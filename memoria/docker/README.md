# Memória de Implementação Docker

Via 

Será utilizado o UBUNTU 20.04 disponibilizado no docker do cuda.

O Cuda Python usa o Numba (Compilador do Anaconda) e o Nvidia Cuda para realizar a compilação para CPU e GPU.

> Numba

  Otimizador just-in-time Python versão Anaconda, capaz de substituir variaveis genericas por especializadas e substituir codigo aplicado por codigo especializado.
  
  Pouco eficiente com strings e muito eficiente com inteiros, pontos flutuantes e dados complexos como vetores.
  
  O Exemplo abaixo tem performance de pior caso (sem jit) 893ns e melhor caso (com jit Numba) 138ns.<br/>
  *(Perceba a anotação/decorator "@jit" que realiza a otimização)*
  
  ```python
  
from numba import jit
import math

@jit
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x);
    y = abs(y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)
  ```
  
> Links

  [docker e cuda](https://hub.docker.com/r/nvidia/cuda)<br/> 
  [Cuda Python](https://developer.nvidia.com/cuda-python) 
