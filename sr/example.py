import datetime
import os
import pysr
import sympy
import numpy as np


def write_eq(file, x, g_latex, g_eq, gp_eq, gpp_eq, loss):
    file.write(f'{x},{g_latex},{g_eq},{gp_eq},{gpp_eq},{loss}\n')


if __name__ == '__main__':
    time = datetime.datetime.now().isoformat()
    plot_dir = f'{time}/figures'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    X = 2. * np.random.randn(100, 1)
    y = 2.5382 * np.exp(X[:, 0]) + X[:, 0]**2 - 0.5
    y_prime = 2.5382 * np.exp(X[:, 0]) + 2. * X[:, 0]
    
    pysr.jl.seval('''
    import Pkg
    Pkg.add("Zygote")
    ''')
    pysr.jl.seval('import Zygote')

    objective = '''
    function custom_loss(tree, dataset::Dataset{T, L}, options)::L where {T, L}
        pred, grad, flag = eval_diff_tree_array(tree, dataset.X, options, 1)
        !flag && return L(Inf)
        pred_diff = pred .- dataset.y
        grad_diff = grad .- dataset.weights
        return (sum(pred_diff.^2) + sum(grad_diff.^2)) / length(pred_diff)
    end
    '''

    model = pysr.PySRRegressor(
        binary_operators=['+', '*'],
        # unary_operators=['log'],
        unary_operators=['exp', 'cosh', 'softplus(x) = log(1 + exp(x))'],
        maxsize=30,
        niterations=10000000,
        populations=15,
        population_size=50,
        ncycles_per_iteration=500,
        # elementwise_loss='L2DistLoss()',
        loss_function=objective,
        model_selection='best',
        # nested_constraints={'log': {'log': 0}},
        nested_constraints={
            'exp': {'exp': 0, 'cosh': 0, 'softplus': 0},
            'cosh': {'exp': 0, 'cosh': 0, 'softplus': 0},
            'softplus': {'exp': 0, 'cosh': 0, 'softplus': 0}
        },
        timeout_in_seconds=60,
        enable_autodiff=True,
        temp_equation_file=True,
        tempdir=time,
        delete_tempfiles=False,
        extra_sympy_mappings={'softplus': lambda x: sympy.log(1. + sympy.exp(x))}
    )
    model.fit(X, y, weights=y_prime, variable_names=['x'])
    print(model)

    x = sympy.symbols('x')
    g_latex = model.latex(precision=4)
    g_sympy = model.sympy()
    gp_sympy = sympy.diff(g_sympy, x)
    gpp_sympy = sympy.diff(gp_sympy, x)
    print(g_latex)
    print(g_sympy)
    print(gp_sympy)
    print(gpp_sympy)

    g_lamb = sympy.lambdify(x, g_sympy)
    gp_lamb = sympy.lambdify(x, gp_sympy)
    y_pred = g_lamb(X.reshape(-1))
    y_prime_pred = gp_lamb(X.reshape(-1))
    
    pred_diff = y_pred - y
    grad_diff = y_prime_pred - y_prime
    loss = ((pred_diff**2).sum() + (grad_diff**2).sum()) / len(pred_diff)
    print(loss)

    file = open(f'{time}/equations.csv', 'w')
    file.write('x,g_latex,g_eq,gp_eq,gpp_eq,loss\n')
    write_eq(file, 0, g_latex, g_sympy, gp_sympy, gpp_sympy, loss)
    file.close()
