from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np

@dataclass_json
@dataclass(frozen=True)
class Client:
    id : str
    X_train : np.array
    X_test : np.array
    y_train : np.array
    y_test: np.array
    paths : list
    size  : int 


