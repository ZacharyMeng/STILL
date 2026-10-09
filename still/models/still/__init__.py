from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from still.models.still.configuration_still import StillConfig
from still.models.still.modeling_still import StillModel, StillModelForCausalLM

AutoConfig.register(StillConfig.model_type, StillConfig)
AutoModel.register(StillConfig, StillModel)
AutoModelForCausalLM.register(StillConfig, StillModelForCausalLM)


__all__ = ['StillConfig', 'StillModelForCausalLM', 'StillModel']