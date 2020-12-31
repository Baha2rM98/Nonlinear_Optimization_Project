from abc import ABC, abstractmethod
from sympy import *
import sympy as sym
import numpy as np


class IterativeMethods(ABC):
    _sym_symbols = list()
    _function = None

    @abstractmethod
    def _assign_symbols(self, symbol: list) -> None:
        pass

    @abstractmethod
    def _check_function_syntax(self, func: str) -> None:
        pass

    @abstractmethod
    def _gradient(self, point: np.ndarray) -> np.ndarray:
        pass


class GradientDescentWithLinearSearch(IterativeMethods):
    __n = None
    iterations = 0

    def __init__(self, symbols_list: list, function: str) -> None:
        self._assign_symbols(symbols_list)
        self._check_function_syntax(function)
        self.__n = self._sym_symbols.__len__()

    def _assign_symbols(self, symbol: list) -> None:
        for i in range(0, symbol.__len__()):
            self._sym_symbols.append(Symbol(symbol[i]))

    def _check_function_syntax(self, func: str) -> None:
        try:
            self._function = parse_expr(func, evaluate=False)
        except Exception:
            raise SyntaxError('Syntax error in function: ' + func)

    def _gradient(self, point: np.ndarray) -> np.ndarray:
        symbol_vars = {}
        for i in range(0, point.size):
            symbol_vars[self._sym_symbols[i]] = point[i]

        gradient = np.zeros((self.__n,), dtype=float)

        for i in range(0, self.__n):
            temp_gradient = sym.diff(self._function, self._sym_symbols[i]).evalf(subs=symbol_vars)
            if type(temp_gradient) == sym.core.mul.Mul:
                raise SyntaxError('The function contains unknown tokens.')
            gradient[i] = temp_gradient

        return gradient

    def gradient_descent(self, first_point: list, step_size: float, epsilon: float = 0.0001):
        if first_point.__len__() != self.__n:
            raise IndexError('Mismatched first point ' + str(first_point) + ' with variables ' + str(self._sym_symbols))

        iterative_point = np.array(first_point)
        gradient_vector = self._gradient(iterative_point) * -1
        self.iterations = 0
        result = {}

        while np.linalg.norm(gradient_vector) > epsilon:
            iterative_point = np.add(iterative_point, step_size * gradient_vector)
            gradient_vector = self._gradient(iterative_point) * -1
            self.iterations += 1
            # print('Optimal point: ', iterative_point, '\t', 'Iterations: ', self.iterations)

        result.update({'Optimal point': iterative_point, 'Iterations': self.iterations})
        return result

    class NewtonMethod(IterativeMethods):
        def _gradient(self, point: np.ndarray) -> np.ndarray:
            pass

        def _check_function_syntax(self, func: str) -> None:
            pass

        def _assign_symbols(self, symbol: list) -> None:
            pass
