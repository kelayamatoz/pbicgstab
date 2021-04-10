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

typedef float T;

void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR: %d\n", status);
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
    printf("WARNING: it is assumed that the matrices are stores in Matrix Market format with double as elementtype\n Usage: ./bfs -F[matrix.mtx] -S[source vertex]\n");

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
    // BFS only takes CSR input if I understand it correctly...
    // if (loadMMSparseMatrix(matrix_filename, *element_type, true, &m, &n, &nnz, &Aval, &indices, &indptr, symmetrize))

    if (loadMMSparseMatrix(matrix_filename, *element_type, true, &m, &n, &nnz, &Aval, &indptr, &indices, symmetrize))
    {
        free(Aval);
        free(indptr);
        free(indices);
        fprintf(stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
        return EXIT_FAILURE;
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
    //Example of graph (CSR format)
    // const size_t n = 7, nnz = 12, 
    const size_t vertex_numsets = 2, edge_numset = 0;
    // int source_offsets_h[] = {0, 1, 3, 4, 6, 8, 10, 12};
    int *source_offsets_h = (int *)malloc(sizeof(int) * (n + 1));
    for (int i = 0; i <= n; i ++)
    {
        source_offsets_h[i] = indptr[i];
    }
    // int destination_indices_h[] = {5, 0, 2, 0, 4, 5, 2, 3, 3, 4, 1, 5};
    int *destination_indices_h = (int *)malloc(sizeof(int) * nnz);
    for (int i = 0; i < nnz; i ++)
    {
        destination_indices_h[i] = indices[i];
    }

    //where to store results (distances from source) and where to store results (predecessors in search tree)
    // int bfs_distances_h[n], bfs_predecessors_h[n];
    int *bfs_distances_h = (int *)malloc(sizeof(int) * n);
    int *bfs_predecessors_h = (int *)malloc(sizeof(int) * n);


    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;
    cudaDataType_t *vertex_dimT;
    size_t distances_index = 0;
    size_t predecessors_index = 1;
    vertex_dimT =
        (cudaDataType_t *)malloc(vertex_numsets * sizeof(cudaDataType_t));
    vertex_dimT[distances_index] = CUDA_R_32I;
    vertex_dimT[predecessors_index] = CUDA_R_32I;
    //Creating nvgraph objects
    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr(handle, &graph));
    // Set graph connectivity and properties (tranfers)

    CSR_input = (nvgraphCSRTopology32I_t)malloc(sizeof(struct
                                                       nvgraphCSCTopology32I_st));
    CSR_input->nvertices = n;
    CSR_input->nedges = nnz;
    CSR_input->source_offsets = source_offsets_h;
    CSR_input->destination_indices = destination_indices_h;
    check_status(nvgraphSetGraphStructure(handle, graph, (void *)CSR_input,
                                          NVGRAPH_CSR_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets,
                                           vertex_dimT));
    //Setting the traversal parameters
    nvgraphTraversalParameter_t traversal_param;
    nvgraphTraversalParameterInit(&traversal_param);
    nvgraphTraversalSetDistancesIndex(&traversal_param, distances_index);
    nvgraphTraversalSetPredecessorsIndex(&traversal_param, predecessors_index);
    nvgraphTraversalSetUndirectedFlag(&traversal_param, false);

    //Computing traversal using BFS algorithm
    printf("Starting from source_vert = %d\n", source_vert);
    double start_kernel = second();
    check_status(nvgraphTraversal(handle, graph, NVGRAPH_TRAVERSAL_BFS,
                                  &source_vert, traversal_param));
    double end_kernel = second();
    fprintf(stdout, "bfs kernel done, time(ms) = %10.8f\n", (end_kernel - start_kernel) * 1000);
    check_status(nvgraphGetVertexData(handle, graph, (void *)bfs_distances_h,
                                      distances_index));
    check_status(nvgraphGetVertexData(handle, graph, (void *)bfs_predecessors_h,
                                      predecessors_index));
    // expect bfs distances_h = (1 0 1 3 3 2 2147483647)
    // for (int i = 0; i < n; i++)
    //     printf("Distance to vertex %d: %i\n", i,
    //            bfs_distances_h[i]);
    // printf("\n");
    // // expect bfs predecessors = (1 -1 1 5 5 0 -1)
    // for (int i = 0; i < n; i++)
    //     printf("%i\n", bfs_predecessors_h[i]);
    int maxDist = 0;
    for (int i = 0; i < n; i ++)
    {
        int bfsDist = bfs_distances_h[i];
        if (bfsDist < 1000 && bfsDist > maxDist)
            maxDist = bfsDist;
    }
    printf("maxDist = %d\n", maxDist);

    free(vertex_dimT);
    free(CSR_input);

    free(source_offsets_h);
    free(destination_indices_h);
    free(bfs_distances_h);
    free(bfs_predecessors_h);
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));
    return 0;
}
