from cvx_optimization_iterative_methods import NewtonMethod, GradientDescentWithLineSearch

while True:
    print('Enter the method you want to use:\n\n(1)gradient descent\n(2)newton method')
    option = input()

    if not option.__eq__('1') and not option.__eq__('2'):
        exit(0)

    if option.__eq__('1'):
        print('Gradient Descent with line search:\n')
        print('Enter the number of function\'s variable:')
        n = int(input())
        if n < 1:
            raise Exception('Number of variables must be more than one.')

        print('Enter the function\'s symbols:')
        function_symbols = [input() for i in range(0, n)]

        print('Enter the function:')
        function = input()

        gd = GradientDescentWithLineSearch(function_symbols, function)

        print('Enter the first point as the same order as variables:')
        initial_point = [float(input()) for j in range(0, n)]
        print('Entered function: \"f = ' + function + '\"' + ' in initial point: ' + str(initial_point))

        print('Starting gradient descent with line search:')
        gd.gradient_descent(initial_point)
        print('\n')

    if option.__eq__('2'):
        print('Newton method:\n')
        print('Enter the number of function\'s variable:')
        n = int(input())
        if n < 1:
            raise Exception('Number of variables must be more than one.')

        print('Enter the function\'s symbols:')
        function_symbols = [input() for i in range(0, n)]

        print('Enter the function:')
        function = input()

        newton = NewtonMethod(function_symbols, function)

        print('Enter the first point as the same order as variables:')
        initial_point = [float(input()) for j in range(0, n)]
        print('Entered function: \"f = ' + function + '\"' + ' in initial point: ' + str(initial_point))

        print('Starting newton optimization method:')
        newton.newton_opt(initial_point)
        print('\n')
