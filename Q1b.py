import numpy as np

def gauss_with_total_pivoting(matrix, vector):
    # Convertendo a entrada para um tipo flutuante para garantir operações com decimais
    A = np.array(matrix, dtype=float)
    b = np.array(vector, dtype=float)
    n = len(b)

    # Criando uma matriz aumentada
    Ab = np.hstack([A, b.reshape(-1, 1)])
    col_order = list(range(n))

    # Função para printar a matriz de forma alinhada
    def print_matrix_step(Ab, step):
        print(f"\nEtapa {step}:\n")
        for row in Ab:
            print(" ".join(f"{elem:>10.4f}" for elem in row))
        print()

    # Algoritmo de eliminação de Gauss com pivotamento total
    for i in range(n):
        # Pivotamento total
        max_row, max_col = divmod(np.argmax(np.abs(Ab[i:, i:n])), n - i)
        max_row += i
        max_col += i

        # Trocar linhas
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Trocar colunas
        if i != max_col:
            Ab[:, [i, max_col]] = Ab[:, [max_col, i]]
            col_order[i], col_order[max_col] = col_order[max_col], col_order[i]

        # Exibir a matriz após pivoteamento
        print(f"Etapa {i + 1} após pivoteamento total (linhas e colunas trocadas):")
        print_matrix_step(Ab, i + 1)

        # Eliminação
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

        # Exibir a matriz após zerar todos os elementos abaixo do pivô na coluna atual
        print_matrix_step(Ab, i + 1)

    # Retro-substituição com reordenação das variáveis
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:])) / Ab[i, i]
        print(f"Retro-substituição para x{col_order[i] + 1}: {x[i]:.6f}")

    # Reordenação do resultado final de acordo com as trocas de colunas
    x_final = np.zeros(n)
    for i in range(n):
        x_final[col_order[i]] = x[i]

    # Exibindo a solução final
    print("\nSolução final:")
    print(x_final)

# Exemplo de uso com o sistema fornecido
A = np.array([
    [40, -77, -91, -20, 85, 17, -90],
    [-78, -97, 40, 83, 5, 67, 33],
    [-91, -4, 15, -92, -44, -74, -50],
    [-54, -53, 71, 69, -87, 22, -9],
    [63, -31, -12, 19, -41, -84, -24],
    [-13, 39, -26, 42, 72, 11, -65],
    [42, 79, 49, -91, -79, 29, -23]
], dtype=float)

b = np.array([16, -80, 66, 46, 58, -84, 31], dtype=float)

gauss_with_total_pivoting(A, b)
