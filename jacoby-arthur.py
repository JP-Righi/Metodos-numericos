import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def read_input() -> list:
    # input in tilles example format from aa02 test
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

    n = int(input())
    
    a = []
    for _ in range(n):
        a.append([*map(float, input().replace("−", "-").split())]) 

    b = []
    for _ in range(n):
        b.append(float(input().replace("−", "-"))) 

    return a, b, n 

def main():
    a, b, n = read_input()
    x = {0: [0]*n}
    error = {0: [float("inf")]*n} 
    k = 0

    while any(e > 1e-5 for e in error[k]):
        for i in range(n):
            acc = 0
            for j in range(n):
                if j != i:
                    acc += a[i][j] * x[k][j]

            if k + 1 not in x:
                x[k + 1] = [0] * n

            x[k + 1][i] = (b[i] - acc) / a[i][i]

            if k + 1 not in error:
                error[k + 1] = [float("inf")] * n     

            if (k + 1) == 1:
                error[k + 1][i] = 1
            else:
                error[k + 1][i] = abs((x[k + 1][i] - x[k][i]) / x[k + 1][i])

        k += 1

    # Prepare PDF
    pdf_filename = "resultado_formatado_com_matplotlib.pdf"
    pdf_pages = PdfPages(pdf_filename)

    # Create first table for x values
    plt.figure(figsize=(9, 7))
    plt.title("Valores de x por iteração")
    plt.axis('off')

    string = [["k"] + [f"x_{{{i + 1}, k}}" for i in range(n)]]
    for k, v in x.items():
        string.append([k] + [f"{single_v:.10g}" for single_v in v])

    table = plt.table(cellText=string, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.25, 1.25)
    pdf_pages.savefig()
    plt.close()

    # Create second table for error values
    plt.figure(figsize=(9, 7))
    plt.title("Erros por iteração")
    plt.axis('off')

    error_string = [["K"] + [f"ER{{{i + 1}, k}}" for i in range(n)]]
    for k, v in error.items():
        error_string.append([k] + ["-" if single_v == float("inf") else f"{single_v:.10g}" for single_v in v])

    error_table = plt.table(cellText=error_string, loc='center', cellLoc='center')
    error_table.auto_set_font_size(False)
    error_table.set_fontsize(8)
    error_table.scale(1.25, 1.25)
    pdf_pages.savefig()
    plt.close()

    pdf_pages.close()
    print(f"Arquivo PDF '{pdf_filename}' gerado com sucesso!")

if __name__ == "__main__":
    main()



'''
-883 6 296 20 95 -58 -39
28 946 19 105 10 292 64
-7 260 589 -51 -154 26 -16
-272 39 10 976 -123 -72 15
30 160 -68 -6 -611 -144 -17
-9 -20 -128 -210 -20 -492 65
-20 -8 -83 -69 -20 -237 442

'''


'''
-37, 78, -65, 47, -53, 43, 15
'''