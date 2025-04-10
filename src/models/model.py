from abc import ABC, abstractmethod
from typing import List, Union
#from wandb.apis.public import Run
import os


class Model(ABC):
    name: str

    @staticmethod
    def from_id(model_id: str, **kwargs) -> "Model":
        if "llama" in model_id or "alpaca" in model_id or os.path.exists(model_id):
            from src.models.llama import LlamaModel

            return LlamaModel(model_name_or_path=model_id, **kwargs)
        #if os.path.exists(model_id): #Assumes path exists
            #from src.models.llama import LlamaModel

            #return LlamaModel(model_name_or_path=model_id, **kwargs)
        else:
            raise NotImplementedError(f"Model {model_id} not implemented.")

    @abstractmethod
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        pass

    @abstractmethod
    def generate(self, inputs: Union[str, List[str]], max_tokens: int, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def cond_log_prob(self, inputs: Union[str, List[str]], targets, **kwargs) -> List[List[float]]:
        pass

    #@abstractmethod
    #ef get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
    #    pass