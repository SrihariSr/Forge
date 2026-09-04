import array

class Dtype:
    def __init__(self, name: str, typecode: str, size: int) -> None:
        self.name = name
        self.typecode = typecode
        self.size = size

    def __repr__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dtype):
            return self.typecode == other.typecode
        return False
    
    def __hash__(self) -> int:
        return hash(self.typecode)

    
float32 = Dtype("float32", "f", 4)
float64 = Dtype("float64", "d", 8)
int32 = Dtype("int32", "i", 4)
int64 = Dtype("int64", "l", 8)

DEFAULT_DTYPE = float32