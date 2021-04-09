#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cusolver.h>
#include "mmio.h"
#include "mmio_wrapper.h"
#include "nvgraph.h"

typedef float T;

void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n", status);
        exit(0);
    }
}

int main(int argc, char **argv)
{
    // File loading...
    char *matrix_filename = NULL;
    char *coloring_filename = NULL;

    int symmetrize = 0;
    int debug = 0;
    int maxit = 1;
    int starting_idx = 0;
    double tol = 0.0000001;
    double damping = 0.85;
    const int vertex_numsets = 2, edge_numsets = 1;
    const float alpha1 = 0.85;
    const void *alpha1_p = (const void *)&alpha1;

    printf("WARNING: it is assumed that the matrices are stores in Matrix Market format with double as elementtype\n Usage: ./pagerank -F[matrix.mtx]\n");

    printf("Starting [%s]\n", argv[0]);
    int ii = 0;
    int temp_argc = argc;
    while (argc)
    {
        if (*argv[ii] == '-')
        {
            switch (*(argv[ii] + 1))
            {
            case 'F':
                matrix_filename = argv[ii] + 2;
                break;
            case 'S':
                starting_idx = atoi(argv[ii] + 2);
                printf("Starting from starting index %d\n", starting_idx);
            case 'D':
                debug = 1;
                break;
            case 'C':
                coloring_filename = argv[ii] + 2;
                break;
            default:
                fprintf(stderr, "Unknown switch '-%s'\n", argv[ii] + 1);
                return -1;
            }
        }
        argc--;
        ii++;
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

    std::ifstream infile(matrix_filename);

    std::string sLine;
    getline(infile, sLine);
    if (sLine.find("symmetric") != std::string::npos)
    {
        symmetrize = 1;
    }

    findCudaDevice(argc, (const char **)argv);

    /* load the graph matrix */
    const char *element_type = "d";
    int m, n, nnz;
    double *Aval = 0;
    int *indptr = 0;
    int *indices = 0;
    // PageRank only takes CSC input if I understand it correctly...
    if (loadMMSparseMatrix(matrix_filename, *element_type, false, &m, &n, &nnz, &Aval, &indices, &indptr, symmetrize))
    {
        free(Aval);
        free(indptr);
        free(indices);
        fprintf(stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
        return EXIT_FAILURE;
    }

    T *weights = (T *)malloc(nnz * sizeof(T));
    for (int i = 0; i < nnz; i++)
    {
        weights[i] = (T)Aval[i];
    }

    int base = indices[0];
    if (base)
    {
        for (int i = 0; i <= n; i++)
        {
            indptr[i] -= base;
        }
        for (int i = 0; i < nnz; i++)
        {
            indices[i] -= base;
        }
    }
    if (m != n)
    {
        fprintf(stderr, "!!!! matrix MUST be square, error: m=%d != n=%d\n", m, n);
        return EXIT_FAILURE;
    }
    printf("^^^^ base=%d, M=%d, N=%d, nnz=%d\n", base, m, n, nnz);

    int i, *destination_offsets_h, *source_indices_h;
    float *weights_h, *bookmark_h, *pr_1;
    void **vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t *vertex_dimT;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int cuda_device = 0;
    cuda_device = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    if (deviceProp.major < 3)
    {
        printf("> nvGraph requires device SM 3.0+\n");
        printf("> Waiving.\n");
        exit(EXIT_WAIVED);
    }

    // Allocate host data
    destination_offsets_h = indptr;
    source_indices_h = indices;
    weights_h = weights;
    bookmark_h = (float *)malloc(n * sizeof(float));
    pr_1 = (float *)malloc(n * sizeof(float));
    vertex_dim = (void **)malloc(vertex_numsets * sizeof(void *));
    vertex_dimT = (cudaDataType_t *)malloc(vertex_numsets * sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));

    // Initialize host data
    vertex_dim[0] = (void *)bookmark_h;
    vertex_dim[1] = (void *)pr_1;
    vertex_dimT[0] = CUDA_R_32F;
    vertex_dimT[1] = CUDA_R_32F;

    for (int i = 0; i < n; i ++)
    {
        bookmark_h[i] = 0.0f;
    }
    bookmark_h[starting_idx] = 1.0f;

    // Starting nvgraph
    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr(handle, &graph));

    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void *)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
    for (i = 0; i < 2; ++i)
        check_status(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
    check_status(nvgraphSetEdgeData(handle, graph, (void *)weights_h, 0));

    // First run with default values
    // I know that my implementation is not going to converge so all is fine...

    double start_kernel = second();
    nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.1f, 1);
    double end_kernel = second();
    fprintf(stdout, "pagerank kernel done, time(ms) = %10.8f\n", (end_kernel - start_kernel) * 1000);

    // Get and print result
    check_status(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
    // printf("pr_1, alpha = 0.85\n");
    // for (i = 0; i < n; i++)
    //     printf("%f\n", pr_1[i]);
    // printf("\n");

    //Clean
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(bookmark_h);
    free(pr_1);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    printf("\nDone!\n");
    return EXIT_SUCCESS;
}
