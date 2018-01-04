#include <Math/IpoptMinimizer.h>

using namespace ROOT;
using namespace ROOT::Math;
using namespace Ipopt;

//_______________________________________________________________________
IpoptMinimizer::IpoptMinimizer() : BasicMinimizer(), fInternalTNLP(this)
{
   fIpotApp = IpoptApplicationFactory();
}

//_______________________________________________________________________
IpoptMinimizer::~IpoptMinimizer()
{
   fIpotApp = nullptr;
}

//_______________________________________________________________________
IpoptMinimizer::IpoptMinimizer::InternalTNLP::InternalTNLP::InternalTNLP(IpoptMinimizer *minimizer)
{
   fNNZerosJacobian = 0;
   fNNZerosHessian = 0;
   fMinimizer = minimizer;
}

//_______________________________________________________________________
IpoptMinimizer::IpoptMinimizer::InternalTNLP::~InternalTNLP()
{
}

//_______________________________________________________________________
/**
 * Give IPOPT the information about the size of the problem (and hence, the size of the arrays that it needs to
 * allocate).
 * \param n  (out), the number of variables in the problem (dimension of $ x$).
 * \param m (out), the number of constraints in the problem (dimension of $ g(x)$).
 * \param nnz_jac_g (out), the number of nonzero entries in the Jacobian.
 * \param nnz_h_lag (out), the number of nonzero entries in the Hessian.
 * \param index_style (out), the numbering style used for row/col entries in the sparse matrix format (C_STYLE: 0-based,
 * FORTRAN_STYLE: 1-based).    default C_STYLE;
 * \return true if everything is right, false in other case.
*/
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
                                                                IndexStyleEnum &index_style)
{
   n = fMinimizer->NDim();
   m = n - fMinimizer->NFree();
   nnz_jac_g = fNNZerosJacobian;
   nnz_h_lag = fNNZerosHessian;
   // use the C style indexing (0-based)
   index_style = TNLP::C_STYLE;

   return true;
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::get_bounds_info(Index n, Number *x_l, Number *x_u, Index m,
                                                                   Number *g_l, Number *g_u)
{
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::get_starting_point(Index n, bool init_x, Number *x, bool init_z,
                                                                      Number *z_L, Number *z_U, Index m,
                                                                      bool init_lambda, Number *lambda)
{
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_f(Index n, const Number *x, bool new_x, Number &obj_value)
{
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f)
{
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g)
{
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_jac_g(Index n, const Number *x, bool new_x, Index m,
                                                              Index nele_jac, Index *iRow, Index *jCol, Number *values)
{
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_h(Index n, const Number *x, bool new_x, Number obj_factor,
                                                          Index m, const Number *lambda, bool new_lambda,
                                                          Index nele_hess, Index *iRow, Index *jCol, Number *values)
{
}

//_______________________________________________________________________
void IpoptMinimizer::IpoptMinimizer::InternalTNLP::finalize_solution(SolverReturn status, Index n, const Number *x,
                                                                     const Number *z_L, const Number *z_U, Index m,
                                                                     const Number *g, const Number *lambda,
                                                                     Number obj_value, const IpoptData *ip_data,
                                                                     IpoptCalculatedQuantities *ip_cq)
{
}

//_______________________________________________________________________
void IpoptMinimizer::SetNNZerosJacobian(UInt_t nzeros)
{
   fInternalTNLP.fNNZerosJacobian = nzeros;
}

//_______________________________________________________________________
void IpoptMinimizer::SetNNZerosHessian(UInt_t nzeros)
{
   fInternalTNLP.fNNZerosHessian = nzeros;
}
