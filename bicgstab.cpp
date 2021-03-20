#include <typeinfo> // for usage of C++ typeid
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "helper_cusolver.h"
#include "mmio.h"

#include "mmio_wrapper.h"

#include "helper_cuda.h"

int main(int argc, char *argv[])
{
    int status = EXIT_FAILURE;
    char *matrix_filename = NULL;
    char *coloring_filename = NULL;

    int symmetrize = 0;
    int debug = 0;
    int maxit = 1;
    double tol = 0.0000001;
    double damping = 0.75;

    printf("WARNING: it is assumed that the matrices are stores in Matrix Market format with double as elementtype\n Usage: ./BiCGStab -F[matrix.mtx] [-E] [-D]\n");

    printf("Starting [%s]\n", argv[0]);
    int i = 0;
    int temp_argc = argc;
    while (argc)
    {
        if (*argv[i] == '-')
        {
            switch (*(argv[i] + 1))
            {
            case 'F':
                matrix_filename = argv[i] + 2;
                break;
            case 'E':
                symmetrize = 1;
                break;
            case 'D':
                debug = 1;
                break;
            case 'C':
                coloring_filename = argv[i] + 2;
                break;
            default:
                fprintf(stderr, "Unknown switch '-%s'\n", argv[i] + 1);
                return status;
            }
        }
        argc--;
        i++;
    }

    argc = temp_argc;

    // Use default input file
    if (matrix_filename == NULL)
    {
        printf("argv[0] = %s", argv[0]);
        matrix_filename = sdkFindFilePath("gr_900_900_crg.mtx", argv[0]);

        if (matrix_filename != NULL)
        {
            printf("Using default input file [%s]\n", matrix_filename);
        }
        else
        {
            printf("Could not find input file = %s\n", matrix_filename);
            return EXIT_FAILURE;
        }
    }
    else
    {
        printf("Using input file [%s]\n", matrix_filename);
    }

    findCudaDevice(argc, (const char **)argv);

    /* load the coefficient matrix */
    const char *element_type = "d";
    int matrixM, matrixN, nnz;
    int matrixSizeAval, matrixSizeAcolsIndex, matrixSizeArowsIndex, mSizeAval, mSizeAcolsIndex, mSizeArowsIndex;
    double *Aval = 0;
    int *AcolsIndex = 0;
    int *ArowsIndex = 0;
    if (loadMMSparseMatrix(matrix_filename, *element_type, true, &matrixM, &matrixN, &nnz, &Aval, &ArowsIndex, &AcolsIndex, symmetrize))
    {
        free(Aval);
        free(AcolsIndex);
        free(ArowsIndex);
        fprintf(stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
        return EXIT_FAILURE;
    }

    matrixSizeAval = nnz;
    matrixSizeAcolsIndex = matrixSizeAval;
    matrixSizeArowsIndex = matrixM + 1;
    int base = ArowsIndex[0];
    if (matrixM != matrixN)
    {
        fprintf(stderr, "!!!! matrix MUST be square, error: m=%d != n=%d\n", matrixM, matrixN);
        return EXIT_FAILURE;
    }
    printf("^^^^ M=%d, N=%d, nnz=%d\n", matrixM, matrixN, nnz);

    // TODO: This is probably going to cause a memory leak, but I don't really care at this point...
    // fix it later definitely!

    // Setting all the matrices and vectors.
    int arraySizeX = matrixN;
    int arraySizeR = matrixM;
    double *x = (double *)malloc(arraySizeX * sizeof(double));
    double *r = (double *)malloc(arraySizeR * sizeof(double));
    for (int i = 0; i < arraySizeX; i++)
    {
        x[i] = 0.1;
    }

    for (int i = 0; i < arraySizeR; i ++)
    {
        r[i] = 0.1;
    }

    return status;
}
