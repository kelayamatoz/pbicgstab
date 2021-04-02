#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <chrono>


using namespace Eigen;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor>SMatrixXf;
struct retVals {
  VectorXf x;
  VectorXf p;
};
    // A = io.mmread(mat_file)
    // M, N = A.shape
    // x = np.ones(N) * 0.1
    // b = np.ones(N) * 0.2
    // r0 = b - A @ x
    // r0_prime = copy.copy(r0)
    // p_j = copy.copy(r0)
    // r_j = copy.copy(r0)
    // for _ in range(max_iter):
    //     alpha_j = (r_j @ r0_prime) / (A @ p_j @ r0_prime)
    //     s_j = r_j - alpha_j * A @ p_j
    //     omega_j = ((A @ s_j) @ s_j) / ((A @ s_j) @ (A @ s_j))
    //     x_new = x + alpha_j * p_j + omega_j * s_j
    //     r_new = s_j - omega_j * A @ s_j
    //     if np.linalg.norm(r_new) < epsilon:
    //         break
    //     beta_j = (alpha_j / omega_j) * (r_new @ r0_prime) / (r_j @ r0_prime)
    //     p_new = r_new + beta_j * (p_j - omega_j * A @ p_j)
    //     x = x_new
    //     p = p_new

retVals bicgstab_no_preconditioning(SMatrixXf A, VectorXf x, VectorXf b, int maxit)
{
  VectorXf r0 = b - A * x;
  VectorXf r0_prime = r0.replicate<1, 1>(); // Column vector
  VectorXf p_j = r0.replicate<1, 1>(); // Column vector
  VectorXf r_j = r0.replicate<1, 1>(); // Column vector
  float alpha_j = (r_j.transpose() * r0_prime / ((A * p_j).transpose() * r0_prime)).value();
  auto s_j = r_j - alpha_j * (A * p_j); // Column
  float omega_j = (((A * s_j).transpose() * s_j) / ((A * s_j).transpose() * (A * s_j))).value();
  VectorXf x_new = x + alpha_j * p_j + omega_j * s_j;
  auto r_new = s_j - omega_j * (A * s_j);
  float beta_j = (alpha_j / omega_j) * (r_new.transpose() * r0_prime).value() / (r_j.transpose() * r0_prime).value();
  VectorXf p_new = r_new + beta_j * (p_j - omega_j * (A * p_j));
  return retVals {x_new, p_new};
}

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
  SMatrixXf A;
  Eigen::loadMarket(A, matrix_filename);
  int m = A.rows();
  int n = A.cols();
  int nnz = A.nonZeros();
  printf("m = %d, n = %d\n", m, n);
  VectorXf x = VectorXf::Ones(n) * 0.1;
  VectorXf b = VectorXf::Ones(m) * 0.2;

  auto t1 = high_resolution_clock::now();
  auto xp = bicgstab_no_preconditioning(A, x, b, maxit);
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  printf("Solved for 1 iter, time = %fms\n", ms_double.count());
  // std::cout << "x = " << xp.x << std::endl;
}
