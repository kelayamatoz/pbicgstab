#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <chrono>


using namespace Eigen;
int main(int argc, char **argv)
{
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
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
        }
    }
    argc--;
    i++;
  }

  argc = temp_argc;
  typedef Eigen::SparseMatrix<float, Eigen::RowMajor>SMatrixXf;
  SMatrixXf A;
  Eigen::loadMarket(A, matrix_filename);
  int m = A.rows();
  int n = A.cols();
  int nnz = A.nonZeros();
  printf("rows = %d, cols = %d, nnz = %d\n", m, n, nnz);
  VectorXd b = VectorXd::Ones(m) * 0.2;
  VectorXd x(n);
  
  printf("Working on a bicgstab solver!\n");
  BiCGSTAB <SparseMatrix<float>, IdentityPreconditioner > solver;
  solver.setMaxIterations(maxit);
  auto t1 = high_resolution_clock::now();
  solver.compute(A);
  solver.solve(b);
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  printf("Solved for 1 iter, time = %fms\n", ms_double.count());
}
