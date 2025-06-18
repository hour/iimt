from typing import Any

class Base:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()
    
    def to(self, device: str):
        raise NotImplementedError()