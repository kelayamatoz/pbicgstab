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

static void bicgstab(cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle,
                     int m, int n, int nnz,
                     const cusparseMatDescr_t descra,
                     double *a, int *ia, int *ja,
                     double *x, double *b, double *r, double *result,
                     int maxit)
{
    double zero = 0.0;
    double one = 1.0;
    double mone = -1.0;
    // checkCudaErrors(
    //     cublasDcopy(cublasHandle, n, x, 1, result, 1)
    // );
    checkCudaErrors(
        cusparseDcsrmv(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            n, n, nnz, &one, descra,
            a, ia, ja, x, &zero, result));
}

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
    double *b = (double *)malloc(arraySizeX * sizeof(double));
    double *result = (double *)malloc(arraySizeX * sizeof(double));
    memset(result, 0, sizeof(double) * arraySizeX);
    for (int i = 0; i < arraySizeX; i++)
    {
        x[i] = 0.1;
        b[i] = 0.2;
    }

    // Allocating handles
    cublasHandle_t cublasHandle = 0;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descra = 0;
    cudaStream_t stream = 0;
    cublasCreate(&cublasHandle);
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descra);
    cudaStreamCreate(&stream);
    cublasSetStream(cublasHandle, stream);
    cusparseSetStream(cusparseHandle, stream);

    // Set the test matrix mode
    cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (base)
    {
        cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ONE);
    }
    else
    {
        cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO);
    }

    // Device pointers
    double *devX = 0;
    double *devR = 0;
    double *devB = 0;
    double *devResult = 0;
    double *devAval = 0;
    int *devAcolsIndex = 0;
    int *devArowsIndex = 0;

    // Allocating device memories
    cudaMalloc((void **)&devX, sizeof(double) * arraySizeX);
    cudaMalloc((void **)&devR, sizeof(double) * arraySizeR);
    cudaMalloc((void **)&devB, sizeof(double) * arraySizeX);
    cudaMalloc((void **)&devResult, sizeof(double) * arraySizeX);
    cudaMalloc((void **)&devAval, sizeof(double) * matrixSizeAval);
    cudaMalloc((void **)&devAcolsIndex, sizeof(double) * matrixSizeAcolsIndex);
    cudaMalloc((void **)&devArowsIndex, sizeof(double) * matrixSizeArowsIndex);

    // Transfer data over to dev memory
    cudaMemcpy(devX, x, (size_t)arraySizeX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devR, r, (size_t)arraySizeR * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, (size_t)arraySizeX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devAval, Aval, (size_t)(matrixSizeAval * sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(devAcolsIndex, AcolsIndex, (size_t)(matrixSizeAcolsIndex * sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(devArowsIndex, ArowsIndex, (size_t)(matrixSizeArowsIndex * sizeof(double)), cudaMemcpyHostToDevice);

    /*************** Kernel *********************/
    // TODO: This is where the kernel should stay!
    // Checking if A is read correctly...

    // printf("ia val = \n");
    // for (int i = 0; i < 901; i ++)
    // {
    //     printf("%d, \n", ArowsIndex[i]);
    // }
    printf("matrixSizeAcolsIndex = %d, matrixSizeArowsIndex = %d", matrixSizeAcolsIndex, matrixSizeArowsIndex);

    bicgstab(cublasHandle, cusparseHandle,
             matrixM, matrixN, matrixSizeAval,
             descra,
             devAval, devArowsIndex, devAcolsIndex,
             devX, devB, devR, devResult,
             maxit);

    checkCudaErrors(cudaMemcpy(result, devResult, (size_t)(arraySizeX * sizeof(double)), cudaMemcpyDeviceToHost));
    /********************************************/
    // Cleanup host
    // printf("checking result of residual\n");
    // for (int i = 0; i < arraySizeX; i++)
    // {
    //     printf("%f, ", result[i]);
    // }

    free(result);
    free(x);
    free(r);
    free(Aval);
    free(AcolsIndex);
    free(ArowsIndex);

    // Cleanup dev
    cudaFree(devX);
    cudaFree(devR);
    cudaFree(devB);
    cudaFree(devResult);
    cudaFree(devAval);
    cudaFree(devAcolsIndex);
    cudaFree(devArowsIndex);
    // Destroy handles
    cudaStreamDestroy(stream);
    cublasDestroy(cublasHandle);
    cusparseDestroy(cusparseHandle);
    return status;
}
