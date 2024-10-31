import numpy as np
import pandas as pd

def format_dataframe(df, significant_digits=10):
    return df.map(lambda x: f"{x:.{significant_digits}g}")  # Corrigido para applymap

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
            # Criação da tabela de iterações
            iteration_df = pd.DataFrame(history, columns=[f'x{i}' for i in range(n)], index=[f'k={k}' for k in range(len(history))])
            # Criação da tabela de erros
            error_df = pd.DataFrame(error_history, columns=[f'Erro x{i}' for i in range(n)], index=[f'k={k}' for k in range(1, len(error_history) + 1)])

            # Formatação das tabelas
            iteration_df = format_dataframe(iteration_df, significant_digits=10)
            error_df = format_dataframe(error_df, significant_digits=10)

            return iteration_df, error_df

        x = x_new

    raise Exception("O método de Gauss-Seidel não convergiu após o número máximo de iterações")

# Definição da matriz e vetor
C = np.array([
    [-883, 6, 296, 20, 95, -58, -39],
    [28, 946, 19, 105, 10, 292, 64],
    [-7, 260, 589, -51, -154, 26, -16],
    [-272, 39, 10, 976, -123, -72, 15],
    [30, 160, -68, -6, -611, -144, -17],
    [-9, -20, -128, -210, -20, -492, 65],
    [-20, -8, -83, -69, -20, -237, 442]
], dtype=float)

d = np.array([-37, 78, -65, 47, -53, 43, 15], dtype=float)

# Impressão dos resultados
print('\nGAUSS SEIDEL', end='\n\n')

# Chamada do método de Gauss-Seidel
iteration_df, error_df = gauss_seidel_method_with_errors(C, d)

# Impressão da tabela de iterações
print("Tabela de Iterações:")
print(iteration_df)

# Impressão da tabela de erros
print("\nTabela de Erros:")
print(error_df)
