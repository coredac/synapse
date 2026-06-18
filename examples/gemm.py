from synapse.frontend.parser import dump_ast

def gemm(A, B, C):
    for i in range(128):
        for j in range(128):
            acc = 0.0
            for k in range(128):
                acc += A[i][k] * B[k][j]
            C[i][j] = acc

if __name__ == "__main__":
    print(dump_ast(gemm))
