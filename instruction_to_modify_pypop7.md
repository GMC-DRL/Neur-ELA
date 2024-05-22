# Instructions to modify PyPop7

In order to use `Ray` in our framework, we have to modify the source code of `PyPop7`.

We provide guideline for `Fast CMAES` modification as an example here, other ES optimizers can be modified in the similar way.


1. Use this line of code to access source code of `FCMAES`.
```python
from pypop7.optimizers.es import FCMAES
```
The path of FCMAES should be in the format as `_YOUR_PYTHON_PATH/site-packages/pypop7/optimizers/es/fcmaes.py`.

2. Replace the original `fcmaes.py` with [new_fcmaes.py](modification_of_pypop7/fcmaes.py).
3. Access source code of `optimizer.py`, which is often to be `_YOUR_PYTHON_PATH/site-packages/pypop7/optimizers/core/optimizer.py`.
4. Replace the original `optimizer.py` with [new_optimizer.py](modification_of_pypop7/optimizer.py).

Then it should work smoothly when you run `run.py`.
