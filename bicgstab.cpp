#include <typeinfo> // for usage of C++ typeid
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
                     double *x, double *b, double *r,
                     double *r0_prime, double *r_j, double *p_j, double *Ap_j, double *r_j_buff, double *As_j, double *r0_prime_copy,
                     int maxit)
{
    double zero = 0.0;
    double one = 1.0;
    double mone = -1.0;
    double nrmr = 0.0;
    double tol = -1.0; // TODO: Change it to something different if I need to run multiple iters.
    // r0 = b - A @ x
    checkCudaErrors(
        cusparseDcsrmv(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            n, n, nnz, &one, descra,
            a, ia, ja, x, &zero, r));
    checkCudaErrors(
        cublasDscal(
            cublasHandle,
            n, &mone, r, 1));
    checkCudaErrors(
        cublasDaxpy(
            cublasHandle,
            n, &one, b, 1, r, 1));

    checkCudaErrors(
        cublasDcopy(
            cublasHandle,
            n, r, 1, r0_prime, 1));

    checkCudaErrors(
        cublasDcopy(
            cublasHandle,
            n, r, 1, r0_prime_copy, 1));

    checkCudaErrors(
        cublasDcopy(
            cublasHandle,
            n, r, 1, r0_prime, 1));
    checkCudaErrors(
        cublasDcopy(
            cublasHandle,
            n, r, 1, r_j, 1));
    checkCudaErrors(
        cublasDcopy(
            cublasHandle,
            n, r, 1, p_j, 1));

    // Step 4, Algo 2.3: α_j = (r_j ·r′0)/((Ap_j)·r′0)
    double alpha_j;
    double rj_dot_r0prime = 0.0;
    double Apj_dot_r0prime = 0.0;
    checkCudaErrors(
        cublasDdot(
            cublasHandle, n, r_j, 1, r0_prime, 1, &rj_dot_r0prime));
    checkCudaErrors(
        cusparseDcsrmv(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            n, n, nnz, &one, descra, a, ia, ja, p_j, &zero, Ap_j));

    checkCudaErrors(
        cublasDdot(
            cublasHandle, n, r0_prime, 1, Ap_j, 1, &Apj_dot_r0prime));
    alpha_j = rj_dot_r0prime / Apj_dot_r0prime;

    // Step 5, Algo 2.3: s_j = r_j − α_j dot Apj
    // TODO: Noticing that I'm using r0_prime to replace rj_buff.
    //  If we want to do multiple iterations, we need to find a way to handle rj_buff correctly.
    double negalpha = -alpha_j;
    double *negalpha_Ap_j = Ap_j;
    checkCudaErrors(
        cublasDaxpy(
            cublasHandle, m, &negalpha, Ap_j, 1, r0_prime, 1));
    double *s_j = r0_prime;

    // Step 6, Algo 2.3: ω_j =((Asj)·sj) / ((Asj)·(Asj))
    double Asj_dot_Asj = 0.0;
    double Asj_dot_sj = 0.0;
    checkCudaErrors(
        cusparseDcsrmv(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            n, n, nnz, &one, descra, a, ia, ja, s_j, &zero, As_j));
    checkCudaErrors(cublasDdot(cublasHandle, m, As_j, 1, As_j, 1, &Asj_dot_Asj));
    checkCudaErrors(cublasDdot(cublasHandle, m, As_j, 1, s_j, 1, &Asj_dot_sj));
    double omega_j = Asj_dot_sj / Asj_dot_Asj;

    // Step 7, Algo 2.3: x_j+1 = x_j + α_j * p_j + ω_j * s_j
    checkCudaErrors(cublasDaxpy(cublasHandle, n, &alpha_j, p_j, 1, x, 1));
    checkCudaErrors(cublasDaxpy(cublasHandle, n, &omega_j, s_j, 1, x, 1));

    // Step 8, Algo 2.3: r_j+1 = −ω_j * As_j + s_j
    double negomega = -omega_j;
    checkCudaErrors(cublasDaxpy(cublasHandle, m, &negomega, As_j, 1, s_j, 1));
    double *r_j_plus_1 = s_j;

    // Step 9, 10, 11, Algo 2.3:
    // 9: if ||rj+1||<ε0 then
    // 10: Break;
    // 11: end if
    checkCudaErrors(cublasDnrm2(cublasHandle, n, r_j_plus_1, 1, &nrmr));
    if (nrmr < tol)
    {
        return;
    }

    // Step 12, Algo 2.3: β_j = (α_j / ω_j) × (r_(j+1) · r′0 )/(rj·r′0)
    double alphaj_div_omegaj = alpha_j / omega_j;
    double one_div_rj_dot_r0prime = 1.0 / rj_dot_r0prime;
    double temp_s12 = 0.0;
    checkCudaErrors(
        cublasDdot(cublasHandle, n, r_j_plus_1, 1, r0_prime_copy, 1, &temp_s12));
    double beta_j = alphaj_div_omegaj * temp_s12 * one_div_rj_dot_r0prime;

    // Step 13, Algo 2.3: p_(j+1) = r_(j+1) + β_j * (p_j −ω_j * Ap_j)
    //                            = r_(j+1) + β_j * p_j - β_j * ω_j * Ap_j
    //                            = r_(j+1) + (- β_j * ω_j * Ap_j) + β_j * p_j  ***
    double neg_betaj_omegaj = -beta_j * omega_j;
    checkCudaErrors(cublasDscal(cublasHandle, n, &beta_j, p_j, 1));
    checkCudaErrors(cublasDaxpy(cublasHandle, n, &neg_betaj_omegaj, Ap_j, 1, p_j, 1));
    checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, r_j_plus_1, 1, p_j, 1));
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
    double *devR0_prime = 0;
    double *devR0_prime_copy = 0;
    double *devR_j = 0;
    double *devP_j = 0;
    double *devAp_j = 0;
    double *devR_j_buff = 0;
    double *devAs_j = 0;
    double *devResult = 0;
    double *devAval = 0;
    int *devAcolsIndex = 0;
    int *devArowsIndex = 0;

    // Allocating device memories
    cudaMalloc((void **)&devX, sizeof(double) * arraySizeX);
    cudaMalloc((void **)&devR, sizeof(double) * arraySizeR);
    cudaMalloc((void **)&devR0_prime, sizeof(double) * arraySizeR);
    cudaMalloc((void **)&devR0_prime_copy, sizeof(double) * arraySizeR);
    cudaMalloc((void **)&devR_j, sizeof(double) * arraySizeR);
    cudaMalloc((void **)&devP_j, sizeof(double) * arraySizeR);
    cudaMalloc((void **)&devB, sizeof(double) * arraySizeX);
    cudaMalloc((void **)&devAp_j, sizeof(double) * arraySizeR);
    cudaMalloc((void **)&devResult, sizeof(double) * arraySizeX);
    cudaMalloc((void **)&devAs_j, sizeof(double) * arraySizeR);
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
    printf("matrixSizeAcolsIndex = %d, matrixSizeArowsIndex = %d", matrixSizeAcolsIndex, matrixSizeArowsIndex);
    bicgstab(cublasHandle, cusparseHandle,
             matrixM, matrixN, matrixSizeAval,
             descra,
             devAval, devArowsIndex, devAcolsIndex,
             devX, devB, devR,
             devR0_prime, devR_j, devP_j, devAp_j, devR_j_buff, devAs_j, devR0_prime_copy,
             maxit);
    checkCudaErrors(cudaMemcpy(result, devX, (size_t)(arraySizeX * sizeof(double)), cudaMemcpyDeviceToHost));
    /********************************************/
    // Cleanup host
    printf("checking result of x\n");
    for (int i = 0; i < arraySizeX; i++)
    {
        printf("%f, ", result[i]);
    }
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
    cudaFree(devR0_prime);
    cudaFree(devR0_prime_copy);
    cudaFree(devR_j);
    cudaFree(devP_j);
    cudaFree(devAp_j);
    cudaFree(devAs_j);
    cudaFree(devR_j_buff);
    // Destroy handles
    cudaStreamDestroy(stream);
    cublasDestroy(cublasHandle);
    cusparseDestroy(cusparseHandle);
    return status;
}
