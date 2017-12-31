#include <Math/IpoptMinimizer.h>

using namespace ROOT;
using namespace ROOT::Math;
using namespace Ipopt;
IpoptMinimizer::IpoptMinimizer::InternalTNLP::InternalTNLP::InternalTNLP()
{
}

IpoptMinimizer::IpoptMinimizer::InternalTNLP::~InternalTNLP()
{
}

bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
                                                                IndexStyleEnum &index_style)
{
}

bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::get_bounds_info(Index n, Number *x_l, Number *x_u, Index m,
                                                                   Number *g_l, Number *g_u)
{
}

bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::get_starting_point(Index n, bool init_x, Number *x, bool init_z,
                                                                      Number *z_L, Number *z_U, Index m,
                                                                      bool init_lambda, Number *lambda)
{
}

bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_f(Index n, const Number *x, bool new_x, Number &obj_value)
{
}

bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f)
{
}

bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g)
{
}

bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_jac_g(Index n, const Number *x, bool new_x, Index m,
                                                              Index nele_jac, Index *iRow, Index *jCol, Number *values)
{
}

bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_h(Index n, const Number *x, bool new_x, Number obj_factor,
                                                          Index m, const Number *lambda, bool new_lambda,
                                                          Index nele_hess, Index *iRow, Index *jCol, Number *values)
{
}

void IpoptMinimizer::IpoptMinimizer::InternalTNLP::finalize_solution(SolverReturn status, Index n, const Number *x,
                                                                     const Number *z_L, const Number *z_U, Index m,
                                                                     const Number *g, const Number *lambda,
                                                                     Number obj_value, const IpoptData *ip_data,
                                                                     IpoptCalculatedQuantities *ip_cq)
{
}
