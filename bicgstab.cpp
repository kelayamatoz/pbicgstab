/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

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

//profiling the code
#define TIME_INDIVIDUAL_LIBRARY_CALLS

#define DBICGSTAB_MAX_ULP_ERR 100
#define DBICGSTAB_EPS 1.E-14f //9e-2

#define CLEANUP()                                             \
    do                                                        \
    {                                                         \
        if (x)                                                \
            free(x);                                          \
        if (f)                                                \
            free(f);                                          \
        if (r)                                                \
            free(r);                                          \
        if (rw)                                               \
            free(rw);                                         \
        if (p)                                                \
            free(p);                                          \
        if (pw)                                               \
            free(pw);                                         \
        if (s)                                                \
            free(s);                                          \
        if (t)                                                \
            free(t);                                          \
        if (v)                                                \
            free(v);                                          \
        if (tx)                                               \
            free(tx);                                         \
        if (Aval)                                             \
            free(Aval);                                       \
        if (AcolsIndex)                                       \
            free(AcolsIndex);                                 \
        if (ArowsIndex)                                       \
            free(ArowsIndex);                                 \
        if (Mval)                                             \
            free(Mval);                                       \
        if (devPtrX)                                          \
            checkCudaErrors(cudaFree(devPtrX));               \
        if (devPtrF)                                          \
            checkCudaErrors(cudaFree(devPtrF));               \
        if (devPtrR)                                          \
            checkCudaErrors(cudaFree(devPtrR));               \
        if (devPtrRW)                                         \
            checkCudaErrors(cudaFree(devPtrRW));              \
        if (devPtrP)                                          \
            checkCudaErrors(cudaFree(devPtrP));               \
        if (devPtrS)                                          \
            checkCudaErrors(cudaFree(devPtrS));               \
        if (devPtrT)                                          \
            checkCudaErrors(cudaFree(devPtrT));               \
        if (devPtrV)                                          \
            checkCudaErrors(cudaFree(devPtrV));               \
        if (devPtrAval)                                       \
            checkCudaErrors(cudaFree(devPtrAval));            \
        if (devPtrAcolsIndex)                                 \
            checkCudaErrors(cudaFree(devPtrAcolsIndex));      \
        if (devPtrArowsIndex)                                 \
            checkCudaErrors(cudaFree(devPtrArowsIndex));      \
        if (devPtrMval)                                       \
            checkCudaErrors(cudaFree(devPtrMval));            \
        if (stream)                                           \
            checkCudaErrors(cudaStreamDestroy(stream));       \
        if (cublasHandle)                                     \
            checkCudaErrors(cublasDestroy(cublasHandle));     \
        if (cusparseHandle)                                   \
            checkCudaErrors(cusparseDestroy(cusparseHandle)); \
        fflush(stdout);                                       \
    } while (0)

// TODO: I need to make sure that all the array sizes match the # elements!
static void gpu_pbicgstab(cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle, int m, int n, int nnz,
                          const cusparseMatDescr_t descra, /* the coefficient matrix in CSR format */ // I don't need this since this comes out of the solver.
                          double *a, int *ia, int *ja,
                          const cusparseMatDescr_t descrm, /* the preconditioner in CSR format, lower & upper triangular factor */
                          double *vm, int *im, int *jm,
                          cusparseSolveAnalysisInfo_t info_l, cusparseSolveAnalysisInfo_t info_u, /* the analysis of the lower and upper triangular parts */
                          double *f, double *r_j, double *r_0_prime, double *p_j, double *pw, double *r_j_buff, double *As_j, double *Ap_j, double *x_j,
                          int maxit, double tol, double ttt_sv)
{
    double rho, rhop, beta, alpha_j, negalpha, omega, negomega, temp, temp2, beta_j;
    double nrmr, nrmr0;
    rho = 0.0;
    double zero = 0.0;
    double one = 1.0;
    double mone = -1.0;
    int i = 0;
    int j = 0;
    double ttl, ttl2, ttu, ttu2, ttm, ttm2;
    double ttt_mv = 0.0;

    //WARNING: Analysis is done outside of the function (and the time taken by it is passed to the function in variable ttt_sv)

    //compute initial residual r0=b-Ax0 (using initial guess in x)
    checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, x_j, &zero, r_j));
    checkCudaErrors(cublasDscal(cublasHandle, n, &mone, r_j, 1));
    checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, f, 1, r_j, 1));

    //copy residual r into r_prime and p, algo2.3, 2
    checkCudaErrors(cublasDcopy(cublasHandle, n, r_j, 1, r_0_prime, 1));
    checkCudaErrors(cublasDcopy(cublasHandle, n, r_j, 1, p_j, 1));

    for (j = 0; j < maxit; j ++)
    {
        // Step 4, Algo 2.3: α_j = (r_j ·r′0)/((Ap_j)·r′0)
        double rj_dot_r0prime = 0.0;
        double Apj_dot_r0prime = 0.0;
        checkCudaErrors(cublasDdot(cublasHandle, n, r_j, 1, r_0_prime, 1, &rj_dot_r0prime));
        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, p_j, &zero, Ap_j));
        checkCudaErrors(cublasDcopy(cublasHandle, n, r_j, 1, r_j_buff, 1));
        checkCudaErrors(cublasDdot(cublasHandle, n, r_0_prime, 1, Ap_j, 1, &Apj_dot_r0prime));
        alpha_j = rj_dot_r0prime / Apj_dot_r0prime;

        // Step 5, Algo 2.3: s_j = r_j − α_j dot Apj
        double negalpha = -alpha_j;
        double *negalpha_Ap_j = Ap_j;
        checkCudaErrors(cublasDaxpy(cublasHandle, m, &negalpha, Ap_j, 1, r_j_buff, 1));
        double *s_j = r_j_buff;

        // Step 6, Algo 2.3: ω_j =((Asj)·sj) / ((Asj)·(Asj))
        double Asj_dot_Asj = 0.0;
        double Asj_dot_sj = 0.0;
        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, s_j, &zero, As_j));
        checkCudaErrors(cublasDdot(cublasHandle, m, As_j, 1, As_j, 1, &Asj_dot_Asj));
        checkCudaErrors(cublasDdot(cublasHandle, m, As_j, 1, s_j, 1, &Asj_dot_sj));
        double omega_j = Asj_dot_sj / Asj_dot_Asj;


        // Step 7, Algo 2.3: x_j+1 = x_j + α_j * p_j + ω_j * s_j
        checkCudaErrors(cublasDaxpy(cublasHandle, n, &alpha_j, p_j, 1, x_j, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle, n, &omega_j, s_j, 1, x_j, 1));

        // Step 8, Algo 2.3: r_j+1 = −ω_j * As_j + s_j
        double negomega = -omega_j;
        checkCudaErrors(cublasDaxpy(cublasHandle, m, &negomega, As_j, 1, s_j, 1));
        double *r_j_plus_1 = s_j;

        // Step 9, 10, 11, Algo 2.3:
        // 9: if ||rj+1||<ε0 then
        // 10: Break;
        // 11: end if
        checkCudaErrors(cublasDnrm2(cublasHandle, n, r_j_plus_1, 1, &nrmr));
        if (nrmr < tol) {
            break;
        }

        // Step 12, Algo 2.3: β_j = (α_j / ω_j) × (r_(j+1) · r′0 )/(rj·r′0)
        double alphaj_div_omegaj = alpha_j / omega_j;
        double one_div_rj_dot_r0prime = 1.0 / rj_dot_r0prime;
        double temp_s12 = 0.0;
        checkCudaErrors(cublasDdot(cublasHandle, n, r_j_plus_1, 1, r_0_prime, 1, &temp_s12));
        double beta_j = alphaj_div_omegaj * temp_s12 * one_div_rj_dot_r0prime;

        // Step 13, Algo 2.3: p_(j+1) = r_(j+1) + β_j * (p_j −ω_j * Ap_j)
        //                            = r_(j+1) + β_j * p_j - β_j * ω_j * Ap_j
        //                            = r_(j+1) + (- β_j * ω_j * Ap_j) + β_j * p_j  ***
        double neg_betaj_omegaj = - beta_j * omega_j;
        checkCudaErrors(cublasDscal(cublasHandle, n, &beta_j, p_j, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle, n, &neg_betaj_omegaj, Ap_j, 1, p_j, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, r_j_plus_1, 1, p_j, 1));

        // Step 14, 15, update x to x+1 and r to r+1. It seems that x is already updated at Step 7?
        // Hence I only need to update r to r+1.
        checkCudaErrors(cublasDcopy(cublasHandle, n, r_j_plus_1, 1, r_j, 1));
    }
}

int test_bicgstab(char *matrix_filename, char *coloring_filename,
                  const char *element_type, int symmetrize, int debug, double damping, int maxit, double tol,
                  float err, float eps)
{
    cublasHandle_t cublasHandle = 0;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descra = 0;
    cusparseMatDescr_t descrm = 0;
    cudaStream_t stream = 0;
    cusparseSolveAnalysisInfo_t info_l = 0;
    cusparseSolveAnalysisInfo_t info_u = 0;
    cusparseStatus_t status1, status2, status3;
    double *devPtrAval = 0;
    int *devPtrAcolsIndex = 0;
    int *devPtrArowsIndex = 0;
    double *devPtrMval = 0;
    int *devPtrMcolsIndex = 0;
    int *devPtrMrowsIndex = 0;
    double *devPtrX = 0;
    double *devPtrF = 0;
    double *devPtrR = 0;
    double *devPtrRW = 0;
    double *devPtrP = 0;
    double *devPtrPW = 0;
    double *devPtrS = 0;
    double *devPtrT = 0;
    double *devPtrV = 0;
    double *Aval = 0;
    int *AcolsIndex = 0;
    int *ArowsIndex = 0;
    double *Mval = 0;
    int *MrowsIndex = 0;
    int *McolsIndex = 0;
    double *x = 0;
    double *tx = 0;
    double *f = 0;
    double *r = 0;
    double *rw = 0;
    double *p = 0;
    double *pw = 0;
    double *s = 0;
    double *t = 0;
    double *v = 0;
    int matrixM;
    int matrixN;
    int matrixSizeAval, matrixSizeAcolsIndex, matrixSizeArowsIndex, mSizeAval, mSizeAcolsIndex, mSizeArowsIndex;
    int arraySizeX, arraySizeF, arraySizeR, arraySizeRW, arraySizeP, arraySizePW, arraySizeS, arraySizeT, arraySizeV, nnz, mNNZ;
    long long flops;
    double start, stop;
    int num_iterations, nbrTests, count, base, mbase;
    cusparseOperation_t trans;
    double alpha;
    double ttt_sv = 0.0;

    printf("Testing %cbicgstab\n", *element_type);

    alpha = damping;
    trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    /* load the coefficient matrix */
    if (loadMMSparseMatrix(matrix_filename, *element_type, true, &matrixM, &matrixN, &nnz, &Aval, &ArowsIndex, &AcolsIndex, symmetrize))
    {
        CLEANUP();
        fprintf(stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
        return EXIT_FAILURE;
    }

    matrixSizeAval = nnz;
    matrixSizeAcolsIndex = matrixSizeAval;
    matrixSizeArowsIndex = matrixM + 1;
    base = ArowsIndex[0];
    if (matrixM != matrixN)
    {
        fprintf(stderr, "!!!! matrix MUST be square, error: m=%d != n=%d\n", matrixM, matrixN);
        return EXIT_FAILURE;
    }
    printf("^^^^ M=%d, N=%d, nnz=%d\n", matrixM, matrixN, nnz);

    /* set some extra parameters for lower triangular factor */
    mNNZ = ArowsIndex[matrixM] - ArowsIndex[0];
    mSizeAval = mNNZ;
    mSizeAcolsIndex = mSizeAval;
    mSizeArowsIndex = matrixM + 1;
    mbase = ArowsIndex[0];

    /* compressed sparse row */
    arraySizeX = matrixN;
    arraySizeF = matrixM;
    arraySizeR = matrixM;
    arraySizeRW = matrixM;
    arraySizeP = matrixN;
    arraySizePW = matrixN;
    arraySizeS = matrixM;
    arraySizeT = matrixM;
    arraySizeV = matrixM;

    /* initialize cublas */
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }
    /* initialize cusparse */
    status1 = cusparseCreate(&cusparseHandle);
    if (status1 != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUSPARSE initialization error\n");
        return EXIT_FAILURE;
    }
    /* create three matrix descriptors */
    status1 = cusparseCreateMatDescr(&descra);
    status2 = cusparseCreateMatDescr(&descrm);
    if ((status1 != CUSPARSE_STATUS_SUCCESS) ||
        (status2 != CUSPARSE_STATUS_SUCCESS))
    {
        fprintf(stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n");
        return EXIT_FAILURE;
    }

    /* allocate device memory for csr matrix and vectors */
    checkCudaErrors(cudaMalloc((void **)&devPtrX, sizeof(devPtrX[0]) * arraySizeX));
    checkCudaErrors(cudaMalloc((void **)&devPtrF, sizeof(devPtrF[0]) * arraySizeF));
    checkCudaErrors(cudaMalloc((void **)&devPtrR, sizeof(devPtrR[0]) * arraySizeR));
    checkCudaErrors(cudaMalloc((void **)&devPtrRW, sizeof(devPtrRW[0]) * arraySizeRW));
    checkCudaErrors(cudaMalloc((void **)&devPtrP, sizeof(devPtrP[0]) * arraySizeP));
    checkCudaErrors(cudaMalloc((void **)&devPtrPW, sizeof(devPtrPW[0]) * arraySizePW));
    checkCudaErrors(cudaMalloc((void **)&devPtrS, sizeof(devPtrS[0]) * arraySizeS));
    checkCudaErrors(cudaMalloc((void **)&devPtrT, sizeof(devPtrT[0]) * arraySizeT));
    checkCudaErrors(cudaMalloc((void **)&devPtrV, sizeof(devPtrV[0]) * arraySizeV));
    checkCudaErrors(cudaMalloc((void **)&devPtrAval, sizeof(devPtrAval[0]) * matrixSizeAval));
    checkCudaErrors(cudaMalloc((void **)&devPtrAcolsIndex, sizeof(devPtrAcolsIndex[0]) * matrixSizeAcolsIndex));
    checkCudaErrors(cudaMalloc((void **)&devPtrArowsIndex, sizeof(devPtrArowsIndex[0]) * matrixSizeArowsIndex));
    checkCudaErrors(cudaMalloc((void **)&devPtrMval, sizeof(devPtrMval[0]) * mSizeAval));

    /* allocate host memory for  vectors */
    x = (double *)malloc(arraySizeX * sizeof(x[0]));
    f = (double *)malloc(arraySizeF * sizeof(f[0]));
    r = (double *)malloc(arraySizeR * sizeof(r[0]));
    rw = (double *)malloc(arraySizeRW * sizeof(rw[0]));
    p = (double *)malloc(arraySizeP * sizeof(p[0]));
    pw = (double *)malloc(arraySizePW * sizeof(pw[0]));
    s = (double *)malloc(arraySizeS * sizeof(s[0]));
    t = (double *)malloc(arraySizeT * sizeof(t[0]));
    v = (double *)malloc(arraySizeV * sizeof(v[0]));
    tx = (double *)malloc(arraySizeX * sizeof(tx[0]));
    Mval = (double *)malloc(matrixSizeAval * sizeof(Mval[0]));
    if ((!Aval) || (!AcolsIndex) || (!ArowsIndex) || (!Mval) ||
        (!x) || (!f) || (!r) || (!rw) || (!p) || (!pw) || (!s) || (!t) || (!v) || (!tx))
    {
        CLEANUP();
        fprintf(stderr, "!!!! memory allocation error\n");
        return EXIT_FAILURE;
    }
    /* use streams */
    int useStream = 0;
    if (useStream)
    {

        checkCudaErrors(cudaStreamCreate(&stream));

        if (cublasSetStream(cublasHandle, stream) != CUBLAS_STATUS_SUCCESS)
        {
            CLEANUP();
            fprintf(stderr, "!!!! cannot set CUBLAS stream\n");
            return EXIT_FAILURE;
        }
        status1 = cusparseSetStream(cusparseHandle, stream);
        if (status1 != CUSPARSE_STATUS_SUCCESS)
        {
            CLEANUP();
            fprintf(stderr, "!!!! cannot set CUSPARSE stream\n");
            return EXIT_FAILURE;
        }
    }

    /* clean memory */
    checkCudaErrors(cudaMemset((void *)devPtrX, 0, sizeof(devPtrX[0]) * arraySizeX));
    checkCudaErrors(cudaMemset((void *)devPtrF, 0, sizeof(devPtrF[0]) * arraySizeF));
    checkCudaErrors(cudaMemset((void *)devPtrR, 0, sizeof(devPtrR[0]) * arraySizeR));
    checkCudaErrors(cudaMemset((void *)devPtrRW, 0, sizeof(devPtrRW[0]) * arraySizeRW));
    checkCudaErrors(cudaMemset((void *)devPtrP, 0, sizeof(devPtrP[0]) * arraySizeP));
    checkCudaErrors(cudaMemset((void *)devPtrPW, 0, sizeof(devPtrPW[0]) * arraySizePW));
    checkCudaErrors(cudaMemset((void *)devPtrS, 0, sizeof(devPtrS[0]) * arraySizeS));
    checkCudaErrors(cudaMemset((void *)devPtrT, 0, sizeof(devPtrT[0]) * arraySizeT));
    checkCudaErrors(cudaMemset((void *)devPtrV, 0, sizeof(devPtrV[0]) * arraySizeV));
    checkCudaErrors(cudaMemset((void *)devPtrAval, 0, sizeof(devPtrAval[0]) * matrixSizeAval));
    checkCudaErrors(cudaMemset((void *)devPtrAcolsIndex, 0, sizeof(devPtrAcolsIndex[0]) * matrixSizeAcolsIndex));
    checkCudaErrors(cudaMemset((void *)devPtrArowsIndex, 0, sizeof(devPtrArowsIndex[0]) * matrixSizeArowsIndex));
    checkCudaErrors(cudaMemset((void *)devPtrMval, 0, sizeof(devPtrMval[0]) * mSizeAval));

    memset(x, 0, arraySizeX * sizeof(x[0]));
    memset(f, 0, arraySizeF * sizeof(f[0]));
    memset(r, 0, arraySizeR * sizeof(r[0]));
    memset(rw, 0, arraySizeRW * sizeof(rw[0]));
    memset(p, 0, arraySizeP * sizeof(p[0]));
    memset(pw, 0, arraySizePW * sizeof(pw[0]));
    memset(s, 0, arraySizeS * sizeof(s[0]));
    memset(t, 0, arraySizeT * sizeof(t[0]));
    memset(v, 0, arraySizeV * sizeof(v[0]));
    memset(tx, 0, arraySizeX * sizeof(tx[0]));

    /* create the test matrix and vectors on the host */
    checkCudaErrors(cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL));
    if (base)
    {
        checkCudaErrors(cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ONE));
    }
    else
    {
        checkCudaErrors(cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO));
    }
    checkCudaErrors(cusparseSetMatType(descrm, CUSPARSE_MATRIX_TYPE_GENERAL));
    if (mbase)
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrm, CUSPARSE_INDEX_BASE_ONE));
    }
    else
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrm, CUSPARSE_INDEX_BASE_ZERO));
    }

    //compute the right-hand-side f=A*e, where e=[1, ..., 1]'
    for (int i = 0; i < arraySizeP; i++)
    {
        p[i] = 1.0;
    }

    /* copy the csr matrix and vectors into device memory */
    double start_matrix_copy, stop_matrix_copy, start_preconditioner_copy, stop_preconditioner_copy;

    start_matrix_copy = second();

    checkCudaErrors(cudaMemcpy(devPtrAval, Aval, (size_t)(matrixSizeAval * sizeof(Aval[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrAcolsIndex, AcolsIndex, (size_t)(matrixSizeAcolsIndex * sizeof(AcolsIndex[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrArowsIndex, ArowsIndex, (size_t)(matrixSizeArowsIndex * sizeof(ArowsIndex[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrMval, devPtrAval, (size_t)(matrixSizeAval * sizeof(devPtrMval[0])), cudaMemcpyDeviceToDevice));

    stop_matrix_copy = second();

    fprintf(stdout, "Copy matrix from CPU to GPU, time(s) = %10.8f\n", stop_matrix_copy - start_matrix_copy);

    checkCudaErrors(cudaMemcpy(devPtrX, x, (size_t)(arraySizeX * sizeof(devPtrX[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrF, f, (size_t)(arraySizeF * sizeof(devPtrF[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrR, r, (size_t)(arraySizeR * sizeof(devPtrR[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrRW, rw, (size_t)(arraySizeRW * sizeof(devPtrRW[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrP, p, (size_t)(arraySizeP * sizeof(devPtrP[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrPW, pw, (size_t)(arraySizePW * sizeof(devPtrPW[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrS, s, (size_t)(arraySizeS * sizeof(devPtrS[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrT, t, (size_t)(arraySizeT * sizeof(devPtrT[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrV, v, (size_t)(arraySizeV * sizeof(devPtrV[0])), cudaMemcpyHostToDevice));

    /* --- GPU --- */
    /* create the analysis info (for lower and upper triangular factors) */
    checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_l));
    checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_u));

    /* analyse the lower and upper triangular factors */
    double ttl = second();
    checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
    checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
    checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz, descrm, devPtrAval, devPtrArowsIndex, devPtrAcolsIndex, info_l));
    checkCudaErrors(cudaDeviceSynchronize());
    double ttl2 = second();

    double ttu = second();
    checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
    checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
    checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz, descrm, devPtrAval, devPtrArowsIndex, devPtrAcolsIndex, info_u));
    checkCudaErrors(cudaDeviceSynchronize());
    double ttu2 = second();
    ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
    printf("analysis lower %f (s), upper %f (s) \n", ttl2 - ttl, ttu2 - ttu);

    /* compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) */
    double start_ilu, stop_ilu;
    printf("CUSPARSE csrilu0 ");
    start_ilu = second();
    devPtrMrowsIndex = devPtrArowsIndex;
    devPtrMcolsIndex = devPtrAcolsIndex;
    checkCudaErrors(cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, descra, devPtrMval, devPtrArowsIndex, devPtrAcolsIndex, info_l));
    checkCudaErrors(cudaDeviceSynchronize());
    stop_ilu = second();
    fprintf(stdout, "time(s) = %10.8f \n", stop_ilu - start_ilu);

    /* run the test */
    num_iterations = 1; //10;
    start = second();
    for (count = 0; count < num_iterations; count++)
    {

        gpu_pbicgstab(cublasHandle, cusparseHandle, matrixM, matrixN, nnz,
                      descra, devPtrAval, devPtrArowsIndex, devPtrAcolsIndex,                                                 // I probably don't need descra. The rest are matrix A's sparse info so I gotta keep them.
                      descrm, devPtrMval, devPtrMrowsIndex, devPtrMcolsIndex,                                                 // I definitely don't need the pre-conditioner!
                      info_l, info_u,                                                                                         // Nop.
                      devPtrF, devPtrR, devPtrRW, devPtrP, devPtrPW, devPtrS, devPtrT, devPtrV, devPtrX, maxit, tol, ttt_sv); // Seems that devPtrX is the only thing I need?

        checkCudaErrors(cudaDeviceSynchronize());
    }
    stop = second();
    printf("CUSPARSE iterators");
    fprintf(stdout, "count = %3d\n", num_iterations);
    fprintf(stdout, "time(ns) = %10.8f \n", (stop - start)*1E9);

    /* destroy the analysis info (for lower and upper triangular factors) */
    checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_l));
    checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_u));

    /* copy the result into host memory */
    checkCudaErrors(cudaMemcpy(tx, devPtrX, (size_t)(arraySizeX * sizeof(tx[0])), cudaMemcpyDeviceToHost));

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    int status = EXIT_FAILURE;
    char *matrix_filename = NULL;
    char *coloring_filename = NULL;
    int symmetrize = 0;
    int debug = 0;
    int maxit = 1;
    double tol = 0.0000001; //0.000001; //0.00001; //0.00000001; //0.0001; //0.001; //0.00000001; //0.1; //0.001; //0.00000001;
    double damping = 0.75;

    /* WARNING: it is assumed that the matrices are stores in Matrix Market format */
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

    status = test_bicgstab(matrix_filename, coloring_filename, "d", symmetrize, debug, damping, maxit, tol,
                           DBICGSTAB_MAX_ULP_ERR, DBICGSTAB_EPS);

    return status;
}
