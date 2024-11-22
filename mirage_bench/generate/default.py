from abc import ABC, abstractmethod
from .dataset import HFDataset


class ABCGenerator(ABC):
    @abstractmethod
    def prepare_dataset(self, dataset_name: str, language: str, split: str, filter_start: int = 0, filter_end: int = None):
        pass

    @abstractmethod
    def generate(self, dataset_name: str, language: str, split: str, filter_start: int = 0, filter_end: int = None):
        pass

class DefaultGenerator(ABCGenerator):
    
    def prepare_dataset(self, dataset_name: str, language: str, split: str, filter_start: int = 0, filter_end: int = None):
        hf_dataclass = HFDataset(dataset_name, language, split=split, cache_dir=self.cache_dir)
        hf_dataclass.load_dataset()
        hf_dataclass.format_prompt("prompt", "prompt_mistral", self.tokenizer)
        hf_dataclass.filter_dataset(filter_start, filter_end)
        return hf_dataclass.hf_dataset
