About 
--------
Pure implementation of Gradient Descent With Line Search and Newton Optimization algorithms in Python.


Requirements
------------
* Numpy

* Sympy

```
pip install numpy
pip install sympy
```

How to run
-------
```
Clone the project in a arbitrary directory, run main.py.
```


*Important tips*
=============
* Use Python operators inside your arbitrary function, for example use: `a**b` for power NOT `a^b`.
* If you use mathematics functions inside your arbitrary function use Python math lib functions, for example use: `exp(y)` for exponential or `acosh(), tan() etc` for trigonometric functions.
* If your arbitrary function contains functions like `exp` or `sin` DO NOT USE variables with names: `e, x, p, s, i, n`. In simpler term DO NOT define `exp(x)` or `sin(s)` , Instead define: `exp(a)` or `sin(x)`.

Some example of valid functions:
-------
```
(x - 4)**4 - (y - 3)**2 + (sin(z*x) * tanh(y**2.7182)) - 4*(z + 5)**4

0.3454663*y**2 + 30*x*y + 21.69*x**2 - 293.074*y

x**3 - 12*x*y + 8*y**3

atan(sinh(x**2 - 2*y))

1000*x**2 + 40*x*y + y**2

exp(a-b) + exp(b-a) - exp(c**2)

x**2 + 2*(y)**2

40.0708*x*y + 592.56*x**2 + 0.01*y**2 - 85.863*x
```