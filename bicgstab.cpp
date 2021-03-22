#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>

using namespace Eigen;
int main(int argc, char **argv)
{
  int status = EXIT_FAILURE;
  char * matrix_filename = NULL;
  int maxit = 1;
  double tol= 0.0000001;
  printf("WARNING: it is assumed that the matrices are stores in Matrix Market format with double as elementtype\n Usage: ./BiCGStab -F[matrix.mtx] [-E] [-D]\n");

  printf("Starting [%s]\n", argv[0]);
  int i=0;
  int temp_argc = argc;
  while (argc) {
    if (*argv[i] == '-') {
        switch (*(argv[i]+1)) { 
        case 'F':
            matrix_filename = argv[i]+2;  
            break;
        default:
            fprintf (stderr, "Unknown switch '-%s'\n", argv[i]+1);
            return status;
        }
    }
    argc--;
    i++;
  }

  argc = temp_argc;


  typedef Eigen::SparseMatrix<float, Eigen::RowMajor>SMatrixXf;
  SMatrixXf A;
  Eigen::loadMarket(A, matrix_filename);
  // int n = 4000;
  // MatrixXd A = MatrixXd::Ones(n, n);
  // MatrixXd B = MatrixXd::Ones(n, n);
  // MatrixXd C = MatrixXd::Ones(n, n);
  // C.noalias() += A * B;
  printf("rows = %ld, cols = %ld, nnz = %ld\n", A.rows(), A.cols(), A.nonZeros());
}
