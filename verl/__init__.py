# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import os

from packaging.version import parse as parse_version

from .protocol import DataProto
from .utils.device import is_npu_available
from .utils.import_utils import import_external_libs
from .utils.logging_utils import set_basic_config

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()


set_basic_config(level=logging.WARNING)


__all__ = ["DataProto", "__version__"]


modules = os.getenv("VERL_USE_EXTERNAL_MODULES", "")
if modules:
    modules = modules.split(",")
    import_external_libs(modules)


if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    from modelscope.utils.hf_util import patch_hub

    patch_hub()


if is_npu_available:
    # Workaround for torch-npu's lack of support for creating nested tensors from NPU tensors.
    #
    # ```
    # >>> a, b = torch.arange(3).npu(), torch.arange(5).npu() + 3
    # >>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
    # ```
    # throws "not supported in npu" on Ascend NPU.
    # See https://github.com/Ascend/pytorch/blob/294cdf5335439b359991cecc042957458a8d38ae/torch_npu/utils/npu_intercept.py#L109
    # for details.

    import torch

    try:
        if hasattr(torch.nested.nested_tensor, "__wrapped__"):
            torch.nested.nested_tensor = torch.nested.nested_tensor.__wrapped__
        if hasattr(torch.nested.as_nested_tensor, "__wrapped__"):
            torch.nested.as_nested_tensor = torch.nested.as_nested_tensor.__wrapped__
    except AttributeError:
        pass

    # In verl, the driver process aggregates the computation results of workers via Ray.
    # Therefore, after a worker completes its computation job, it will package the output
    # using tensordict and transfer it to the CPU. Since the `to` operation of tensordict
    # is non-blocking, when transferring data from a device to the CPU, it is necessary to
    # ensure that a batch of data has been completely transferred before being used on the
    # host; otherwise, unexpected precision issues may arise. Tensordict has already noticed
    # this problem and fixed it. Ref: https://github.com/pytorch/tensordict/issues/725
    # However, the relevant modifications only cover CUDA and MPS devices and do not take effect
    # for third-party devices such as NPUs. This patch fixes this issue, and the relevant
    # modifications can be removed once the fix is merged into tensordict.

    import tensordict

    if parse_version(tensordict.__version__) < parse_version("0.10.0"):
        from tensordict.base import TensorDictBase

        def _sync_all_patch(self):
            from torch._utils import _get_available_device_type, _get_device_module

            device_type = _get_available_device_type()
            if device_type is None:
                return

            device_module = _get_device_module(device_type)
            device_module.synchronize()

        TensorDictBase._sync_all = _sync_all_patch


# --- Emu3-Stage1 GRPO compat shim --------------------------------------------
# Register the Emu3 text-only submodel as a top-level AutoConfig / AutoModel
# target so loading the SFT snapshot (model_type='emu3_text_model') doesn't
# fall through to the multimodal Emu3Config path — that path triggers vLLM's
# MultiModalBudget which needs Emu3Processor.image_token (absent on the fast
# tokenizer) and dies the rollout actor. See emu3-mask-grpo-2348539.err for
# the failure. Runs on every verl import (TaskRunner, WorkerDict,
# vLLMHttpServer — all live under verl.workers), so the registration reaches
# Ray-spawned actors without any sitecustomize/setup-hook gymnastics.
try:
    from transformers import Emu3ForCausalLM, Emu3TextConfig, Emu3TextModel
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    if 'emu3_text_model' not in CONFIG_MAPPING._extra_content and        'emu3_text_model' not in CONFIG_MAPPING._mapping:
        CONFIG_MAPPING.register('emu3_text_model', Emu3TextConfig)
    if Emu3TextConfig not in MODEL_FOR_CAUSAL_LM_MAPPING._extra_content:
        MODEL_FOR_CAUSAL_LM_MAPPING.register(Emu3TextConfig, Emu3ForCausalLM)
    # vLLM's transformers backend uses AutoModel (not AutoModelForCausalLM)
    # to look up the inner module (embed_tokens+layers+norm, NO lm_head).
    # vLLM then adds its own ParallelLMHead and runs compute_logits on top.
    # Registering Emu3ForCausalLM here (the full causal LM) caused vLLM to
    # double-apply the lm_head: model.forward returned (B*S, vocab) instead
    # of (B*S, hidden), then ParallelLMHead tried to project those 'hidden'
    # states, crashing the rollout worker with
    #   mat1 and mat2 shapes cannot be multiplied (1024x184640 and 4096x184640)
    # See emu3-mask-grpo-2349597.err. Fix: register the inner Emu3TextModel.
    from transformers.models.auto.modeling_auto import MODEL_MAPPING
    if Emu3TextConfig not in MODEL_MAPPING._extra_content:
        MODEL_MAPPING.register(Emu3TextConfig, Emu3TextModel)
except Exception as _e:
    # Don't crash the broader verl import if registration fails — log and
    # continue (downstream Emu3 loads will fail loudly with their own errors).
    logging.getLogger(__name__).warning(
        'Emu3TextConfig registration skipped: %s', _e
    )
