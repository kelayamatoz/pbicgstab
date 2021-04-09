/*
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_cusolver.h"
#include "nvgraph.h"
#include "mmio.h"
#include "mmio_wrapper.h"

/* Single Source Shortest Path (SSSP)
 *  Calculate the shortest path distance from a single vertex in the graph
 *  to all other vertices.
 */

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
    char *matrix_filename = NULL;
    char *coloring_filename = NULL;

    int symmetrize = 0;
    int debug = 0;
    int source_vert = 0;
    printf("WARNING: it is assumed that the matrices are stores in Matrix Market format with double as elementtype\n Usage: ./sssp -F[matrix.mtx] -S[source vertex]\n");

    printf("Starting [%s]\n", argv[0]);
    int ii = 0;
    int temp_argc = argc;
    int base = 1;
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
                source_vert = atoi(argv[ii] + 2);
            case 'B':
                base = atoi(argv[ii] + 2);
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
    printf("Starting from source vertex %d\n", source_vert);

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

    const size_t vertex_numsets = 2, edge_numsets = 1;
    int i, *destination_offsets_h, *source_indices_h;
    float *weights_h, *sssp_1_h, *sssp_2_h;
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

    // Init host data
    destination_offsets_h = (int *)malloc((n + 1) * sizeof(int));
    source_indices_h = (int *)malloc(nnz * sizeof(int));
    weights_h = (float *)malloc(nnz * sizeof(float));
    sssp_1_h = (float *)malloc(n * sizeof(float));
    sssp_2_h = (float *)malloc(n * sizeof(float));
    vertex_dim = (void **)malloc(vertex_numsets * sizeof(void *));
    vertex_dimT = (cudaDataType_t *)malloc(vertex_numsets * sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));

    vertex_dim[0] = (void *)sssp_1_h;
    vertex_dim[1] = (void *)sssp_2_h;
    vertex_dimT[0] = CUDA_R_32F;
    vertex_dimT[1] = CUDA_R_32F;

    for (int i = 0; i < nnz; i++)
    {
        weights_h[i] = 1.0;
    }

    // weights_h [0] = 0.333333;
    // weights_h [1] = 0.500000;
    // weights_h [2] = 0.333333;
    // weights_h [3] = 0.500000;
    // weights_h [4] = 0.500000;
    // weights_h [5] = 1.000000;
    // weights_h [6] = 0.333333;
    // weights_h [7] = 0.500000;
    // weights_h [8] = 0.500000;
    // weights_h [9] = 0.500000;

    for (int i = 0; i <= n; i++)
    {
        destination_offsets_h[i] = indptr[i];
    }

    // destination_offsets_h [0] = 0;
    // destination_offsets_h [1] = 1;
    // destination_offsets_h [2] = 3;
    // destination_offsets_h [3] = 4;
    // destination_offsets_h [4] = 6;
    // destination_offsets_h [5] = 8;
    // destination_offsets_h [6] = 10;

    for (int i = 0; i < nnz; i++)
    {
        source_indices_h[i] = indices[i];
    }

    // source_indices_h [0] = 2;
    // source_indices_h [1] = 0;
    // source_indices_h [2] = 2;
    // source_indices_h [3] = 0;
    // source_indices_h [4] = 4;
    // source_indices_h [5] = 5;
    // source_indices_h [6] = 2;
    // source_indices_h [7] = 3;
    // source_indices_h [8] = 3;
    // source_indices_h [9] = 4;

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
    check_status(nvgraphSetEdgeData(handle, graph, (void *)weights_h, 0));

    // Solve
    printf("Starting nvgraphSssp from source_vert = %d\n", source_vert);
    double start_kernel = second();
    check_status(nvgraphSssp(handle, graph, 0, &source_vert, 0));
    check_status(nvgraphGetVertexData(handle, graph, (void *)sssp_1_h, 0));
    double end_kernel = second();
    fprintf(stdout, "sssp kernel done, time(ms) = %10.8f\n", (end_kernel - start_kernel) * 1000);
    double maxDist = 0.0;
    for (i = 0; i < n; i++)
    {
        double dist = sssp_1_h[i];
        if (dist > maxDist){
            if (dist < 1000.0)
            {
                maxDist = dist;
            }
        }
    }
    printf("max distance = %d\n", (int)maxDist);

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(sssp_1_h);
    free(sssp_2_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    //Clean
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));

    return EXIT_SUCCESS;
}
