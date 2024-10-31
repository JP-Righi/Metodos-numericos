import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table

def read_input() -> list:
    # input in tilde example format from aa02 test
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

    n = int(input())

    a = []
    for _ in range(n):
        a.append([*map(float, input().replace("−", "-").split())]) 
        # replace method makes it have the right "-"

    b = []
    for _ in range(n):
        b.append(float(input().replace("−", "-"))) 
        # replace method makes it have the right "-"

    return a, b, n 

def create_table(ax, data, title):
    # Create a table and add it to the Axes
    table = Table(ax, bbox=[0, 0, 1, 1])
    n_rows, n_cols = len(data), len(data[0])

    for i in range(n_rows):
        for j in range(n_cols):
            table.add_cell(i, j, width=1/n_cols, height=1/n_rows, text=data[i][j], loc='center', facecolor='white')

    for i in range(n_rows):
        table.add_cell(i, -1, width=1/n_cols, height=1/n_rows, text=f"{i}", loc='center', facecolor='lightgray')

    for j in range(n_cols):
        table.add_cell(-1, j, width=1/n_cols, height=1/n_rows, text=f"x_{{{j+1}, k}}", loc='center', facecolor='lightgray')

    ax.add_table(table)
    ax.axis('off')  # Hide axes

    ax.set_title(title, fontsize=14, fontweight='bold')

def main():
    a, b, n = read_input()
    x = {0: [0]*n}
    error = {0: [float("inf")]*n} 

    k = 0
    while any(e > 1e-5 for e in error[k]):
        for i in range(n):
            acc = 0
            for j in range(i):
                if j != i:
                    acc += a[i][j] * x[k+1][j]

            for j in range(i, n):
                if j != i:
                    acc += a[i][j] * x[k][j]

            if k+1 not in x:
                x[k+1] = [0]*n

            x[k+1][i] = (b[i]-acc)/a[i][i]

            if k+1 not in error:
                error[k+1] = [float("inf")]*n     

            if (k+1) == 1:
                error[k+1][i] = 1
            else:
                error[k+1][i] = abs((x[k+1][i]-x[k][i])/x[k+1][i])

        k += 1

    # Create the PDF
    with PdfPages('output.pdf') as pdf:
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))

        # Prepare data for the first table (results)
        result_table_data = [["k"] + [f"x_{{{i+1},k}}" for i in range(n)]]
        for k, v in x.items():
            result_table_data.append([k] + [f"{single_v:.10g}" for single_v in v])

        create_table(axs[0], result_table_data, "Tabelas de Resultados (Método de Jacobi)")

        # Prepare data for the second table (errors)
        error_table_data = [["K"] + [f"ER_{{{i+1},k}}" for i in range(n)]]
        for k, v in error.items():
            error_row = [k] + ["-" if single_v == float("inf") else f"{single_v:.10g}" for single_v in v]
            error_table_data.append(error_row)

        create_table(axs[1], error_table_data, "Tabela de Erros (Método de Jacobi)")

        # Save the current figure
        pdf.savefig(fig, bbox_inches='tight')

if __name__ == "__main__":
    main()
