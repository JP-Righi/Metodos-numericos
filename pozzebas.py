import numpy as np
import pandas as pd


def format_dataframe(df, significant_digits=10):
    return df.applymap(lambda x: f"{x:.{significant_digits}g}")


def jacobi_method_with_errors(A, b, tolerance=1e-11, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    history = [x.copy()]
    error_history = []

    for iteration in range(max_iterations):
        errors = np.zeros(n)
        for i in range(n):
            sum_ax = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum_ax) / A[i, i]

            if x_new[i] != 0:
                errors[i] = abs((x_new[i] - x[i]) / x_new[i])

        error_history.append(errors.copy())
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            iteration_df = pd.DataFrame(history, columns=[f'x{i}' for i in range(n)], index=[f'k={k}' for k in range(len(history))])
            error_df = pd.DataFrame(error_history, columns=[f'Error x{i}' for i in range(n)], index=[f'k={k}' for k in range(1, len(error_history) + 1)])

            iteration_df = format_dataframe(iteration_df, significant_digits=10)
            error_df = format_dataframe(error_df, significant_digits=10)

            return iteration_df, error_df

        x = x_new.copy()

    raise Exception("O método de Jacobi não convergiu após o número máximo de iterações")


def gauss_seidel_method_with_errors(A, b, tolerance=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    history = [x.copy()]
    error_history = []

    for iteration in range(max_iterations):
        x_new = np.copy(x)
        errors = np.zeros(n)

        for i in range(n):
            sum_ax = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            old_value = x_new[i]
            x_new[i] = (b[i] - sum_ax) / A[i, i]

            if x_new[i] != 0:
                errors[i] = abs((x_new[i] - old_value) / x_new[i])

        error_history.append(errors.copy())
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            iteration_df = pd.DataFrame(history, columns=[f'x{i}' for i in range(n)], index=[f'k={k}' for k in range(len(history))])
            error_df = pd.DataFrame(error_history, columns=[f'Error x{i}' for i in range(n)], index=[f'k={k}' for k in range(1, len(error_history) + 1)])

            iteration_df = format_dataframe(iteration_df, significant_digits=10)
            error_df = format_dataframe(error_df, significant_digits=10)

            return iteration_df, error_df

        x = x_new

    raise Exception("O método de Gauss-Seidel não convergiu após o número máximo de iterações")


C = np.array([
    [434, 52, -99, 195, -7, -19, 29],
    [-52, -613, 39, -1, -154, 12, 303],
    [-203, -37, 761, -65, -103, -12, 1],
    [-150, 13, -240, -966, -5, -32, 46],
    [-16, 299, -31, -93, 601, -10, -55],
    [-55, -31, 9, 106, 208, -543, 13],
    [9, 19, -23, 54, -303, -140, 934]
], dtype=float)

d = np.array([-13, 31, -57, 15, 17, -41, 53], dtype=float)

print('JACOBI', end='\n\n')
print(jacobi_method_with_errors(C, d), end='\n\n\n')
print('GAUSS SEIDEL', end='\n\n')
print(gauss_seidel_method_with_errors(C, d), end='\n\n\n')