from abc import ABC, abstractmethod
from sympy import Symbol, parse_expr, solveset, S
import sympy as sym
import numpy as np
import numpy.linalg as linear_algebra


class IterativeMethods(ABC):
    _sym_symbols = list()
    _function = None
    _n = None

    @abstractmethod
    def __init__(self, symbols_list: list, function: str) -> None:
        self._assign_symbols(symbols_list)
        self._check_function_syntax(function)
        self._n = self._sym_symbols.__len__()

    def _assign_symbols(self, symbol: list) -> None:
        for i in range(0, symbol.__len__()):
            self._sym_symbols.append(Symbol(symbol[i]))

    def _check_function_syntax(self, func: str) -> None:
        try:
            self._function = parse_expr(func, evaluate=False)
        except Exception:
            raise SyntaxError('Syntax error in function: ' + func)

    def _gradient(self, point: np.ndarray) -> np.ndarray:
        symbol_value = dict()
        for i in range(0, point.size):
            symbol_value[self._sym_symbols[i]] = point[i]

        gradient_vector = np.zeros((self._n,), dtype=float)

        for i in range(0, self._n):
            temp_gradient_value = sym.diff(self._function, self._sym_symbols[i]).evalf(subs=symbol_value)
            if type(temp_gradient_value) == sym.core.mul.Mul:
                raise SyntaxError('The function contains unknown tokens.')
            gradient_vector[i] = temp_gradient_value

        return gradient_vector

    def _reset(self) -> None:
        self._sym_symbols = list()
        self._function = None
        self._n = None


class GradientDescentWithLineSearch(IterativeMethods):
    iterations = 0

    def __init__(self, symbols_list: list, function: str) -> None:
        self._reset()
        super().__init__(symbols_list, function)

    def __step_size_line_search(self, x_point: list, d_point: list) -> float:
        temp = list()
        for i in range(0, self._n):
            d_point[i] = str(d_point[i]) + '*t'
            if not str(x_point[i]).__contains__('-'):
                temp.append(d_point[i] + '+' + str(x_point[i]))
            if str(x_point[i]).__contains__('-'):
                temp.append(d_point[i] + str(x_point[i]))

        temp_func = str(self._function)
        for i in range(0, self._n):
            s = str(self._sym_symbols[i])
            if temp_func.__contains__(s):
                temp_func = temp_func.replace(s, '(' + temp[i] + ')')

        t = Symbol('t')
        h = sym.diff(parse_expr(temp_func), t)

        try:
            answers = solveset(h, t, domain=S.Reals)
        except Exception:
            raise ValueError('Convergence to root failed.')

        if type(answers) != sym.sets.sets.FiniteSet:
            raise ValueError(
                'Unacceptable root in finding step size. The equation: '
                + '\"' + str(h) + '\"' + ' has no root in Real numbers system.')

        return float(answers.args[0])

    def gradient_descent(self, first_point: list, epsilon: float = 0.0000001) -> None:
        if first_point.__len__() != self._n:
            raise IndexError('Mismatched first point ' + str(first_point) + ' with variables ' + str(self._sym_symbols))

        iterative_point = np.array(first_point)
        gradient_vector = self._gradient(iterative_point)
        self.iterations = 0

        while linear_algebra.norm(gradient_vector) > epsilon:
            step_size = self.__step_size_line_search(list(iterative_point), list(gradient_vector * -1))
            iterative_point = np.subtract(iterative_point, step_size * gradient_vector)
            gradient_vector = self._gradient(iterative_point)
            self.iterations += 1
            print('Optimal point: ', iterative_point, '\t', 'In iteration: ', self.iterations, '\t', 'Step Size: ',
                  step_size)

        opt_symbol_value = dict()
        for i in range(0, self._n):
            opt_symbol_value[self._sym_symbols[i]] = iterative_point[i]

        print('Optimal value: ' + str(self._function.evalf(subs=opt_symbol_value)))


class NewtonMethod(IterativeMethods):
    iterations = 0

    def __init__(self, symbols_list: list, function: str) -> None:
        self._reset()
        super().__init__(symbols_list, function)

    def __hessian(self, point: np.ndarray) -> np.ndarray:
        symbol_vars = dict()
        for i in range(0, point.size):
            symbol_vars[self._sym_symbols[i]] = point[i]

        symbolic_gradient_vector = []
        for i in range(0, self._n):
            symbolic_gradient_vector.append(sym.diff(self._function, self._sym_symbols[i]))

        hessian_matrix = np.zeros((self._n, self._n), dtype=float)

        for i in range(0, self._n):
            for j in range(0, self._n):
                temp_hessian_value = sym.diff(symbolic_gradient_vector[i], self._sym_symbols[j]).evalf(subs=symbol_vars)
                if type(temp_hessian_value) == sym.core.mul.Mul:
                    raise SyntaxError('The function contains unknown tokens.')
                hessian_matrix[i][j] = temp_hessian_value

        return hessian_matrix

    def newton_opt(self, first_point: list, epsilon: float = 0.00000001) -> None:
        if first_point.__len__() != self._n:
            raise IndexError('Mismatched first point ' + str(first_point) + ' with variables ' + str(self._sym_symbols))

        iterative_point = np.array(first_point)
        self.iterations = 0

        while True:
            t = np.matmul(linear_algebra.inv(self.__hessian(iterative_point)), self._gradient(iterative_point))
            iterative_point = np.subtract(iterative_point, t)
            self.iterations += 1
            print('Optimal point: ', iterative_point, '\t', 'In iteration: ', self.iterations)
            if linear_algebra.norm(t) < epsilon:
                break

        opt_symbol_value = dict()
        for i in range(0, self._n):
            opt_symbol_value[self._sym_symbols[i]] = iterative_point[i]

        print('Optimal value: ' + str(self._function.evalf(subs=opt_symbol_value)))
