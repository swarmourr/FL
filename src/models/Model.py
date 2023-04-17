
from dataclasses import dataclass
import pickle


@dataclass(frozen=True)
class ModelClass():
    model_name : str
    class_name : str
    model_init : object
    model_local: object


    def __repr__(self) -> str:
        return "- Model name :"+  self.model_name +"\n- Class name :"+  self.class_name+ "\n- Object : "   + str(self.model_init) +"\n- local model path : "   + str(self.model_local)  




