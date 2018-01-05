#include <Math/IpoptMinimizer.h>
#include <Fit/ParameterSettings.h>
#include <Math/IFunction.h>
#include "Math/MultiNumGradFunction.h"
#include "Math/FitMethodFunction.h"
#include <TString.h>

using namespace ROOT;
using namespace ROOT::Math;
using namespace Ipopt;
using namespace ROOT::Fit;
//_______________________________________________________________________
IpoptMinimizer::IpoptMinimizer() : BasicMinimizer()
{
   fIpotApp = IpoptApplicationFactory();
   fInternalTNLP = new InternalTNLP(this);
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
   nlp_lower_bound_inf = -1e19;
   nlp_upper_bound_inf = 1e19;
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
   m = 0; // n - fMinimizer->NFree();//total variables with constraints
   nnz_jac_g = fNNZerosJacobian;
   nnz_h_lag = fNNZerosHessian;
   // use the C style indexing (0-based)
   index_style = TNLP::C_STYLE;

   return true;
}

//_______________________________________________________________________
/** Give IPOPT the value of the bounds on the variables and constraints.
* The values of n and m that you specified in get_nlp_info are passed to you for debug checking.
* Setting a lower bound to a value less than or  equal to the value of the option nlp_lower_bound_inf will cause IPOPT
* to assume no lower bound.
* Likewise, specifying the upper bound above or equal to the value of the option nlp_upper_bound_inf will cause IPOPT to
* assume no upper bound.
* These options, nlp_lower_bound_inf and nlp_upper_bound_inf, are set to \f$ -10^{19}$ and $ 10^{19}\f$, respectively,
* by default.
* \param n (in), the number of variables in the problem (dimension of $ x$).
* \param x_l (out) the lower bounds \f$ x^L$ for $ x\f$.
* \param x_u (out) the upper bounds \f$ x^U$ for $ x\f$.
* \param m (in), the number of constraints in the problem (dimension of \f$ g(x)\f$).
* \param g_l (out) the lower bounds \f$ g^L$ for $ g(x)\f$.
* \param g_u (out) the upper bounds \f$ g^U$ for $ g(x)\f$.
* \return true if everything is right, false in other case.
*/
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::get_bounds_info(Index n, Number *x_l, Number *x_u, Index m,
                                                                   Number *g_l, Number *g_u)
{
   // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
   // If desired, we could assert to make sure they are what we think they are.
   // TODO: print a meesage telling that information is not good whit the variables and constraints
   R__ASSERT(n == (Index)fMinimizer->NDim());
   //   R__ASSERT(m == fMinimizer->NDim() - fMinimizer->NFree());
   for (Index i = 0; i < n; i++) {
      ParameterSettings varsettings;
      if (fMinimizer->GetVariableSettings(i, varsettings)) {
         if (varsettings.HasLowerLimit()) {
            x_l[i] = varsettings.LowerLimit();
         } else {
            x_l[i] = nlp_lower_bound_inf;
         }
         if (varsettings.HasUpperLimit()) {
            x_u[i] = varsettings.UpperLimit();
         } else {
            x_u[i] = nlp_upper_bound_inf;
         }
      } else {
         MATH_ERROR_MSG("IpoptMinimizer::InternalTNLP::get_bounds_info", Form("Variable index = $d not found", i));
      }
   }
   return true;
}

//_______________________________________________________________________
/**
 * Give IPOPT the starting point before it begins iterating.
 * The variables n and m are passed in for your convenience. These variables will have the same values you specified in
 * get_nlp_info.
 * Depending on the options that have been set, IPOPT may or may not require bounds for the primal variables \f$ x\f$,
 * the bound multipliers \f$ z^L\f$ and \f$ z^U\f$, and the constraint multipliers \f$ \lambda \f$.
 * The boolean flags init_x, init_z, and init_lambda tell you whether or not you should provide initial values for \f$
 * x\f$, \f$ z^L\f$, \f$ z^U\f$, or \f$ \lambda \f$ respectively.
 * The default options only require an initial value for the primal variables \f$ x\f$.
 * Note, the initial values for bound multiplier components for ``infinity'' bounds ( \f$ x_L^{(i)}=-\infty \f$ or  \f$
 * x_U^{(i)}=\infty \f$) are ignored.
 *
 * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
 * \param init_x (in), if true, this method must provide an initial value for \f$ x\f$.
 * \param x (out), the initial values for the primal variables, \f$ x\f$.
 * \param init_z (in), if true, this method must provide an initial value for the bound multipliers \f$ z^L\f$ and \f$
 * z^U\f$.
 * \param z_L (out), the initial values for the bound multipliers, \f$ z^L\f$.
 * \param z_U (out), the initial values for the bound multipliers, \f$ z^U\f$.
 * \param m (in), the number of constraints in the problem (dimension of \f$ g(x)\f$).
 * \param init_lambda: (in), if true, this method must provide an initial value for the constraint multipliers, \f$
 * \lambda\f$.
 * \param lambda (out), the initial values for the constraint multipliers, \f$ \lambda\f$.
 */
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::get_starting_point(Index n, bool init_x, Number *x, bool init_z,
                                                                      Number *z_L, Number *z_U, Index m,
                                                                      bool init_lambda, Number *lambda)
{
   R__ASSERT(n == fMinimizer->NDim());
   for (Index i = 0; i < n; i++) {
      ParameterSettings varsettings;
      if (fMinimizer->GetVariableSettings(i, varsettings)) {
         x[i] = varsettings.Value();
      } else {
         MATH_ERROR_MSG("IpoptMinimizer::InternalTNLP::get_starting_point", Form("Variable index = $d not found", i));
      }
   }
   return true;
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_f(Index n, const Number *x, bool new_x, Number &obj_value)
{

   auto fun = fMinimizer->ObjFunction();
   R__ASSERT(n == (Index)fun->NDim());
   obj_value = (*fun)(x);
   return true;
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f)
{
   auto gfun = fMinimizer->GradObjFunction();
   if (!gfun)
      return false;
   else {
      R__ASSERT(n == (Index)gfun->NDim());
      gfun->Gradient(x, grad_f);
   }
   return true;
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g)
{
   return false;
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_jac_g(Index n, const Number *x, bool new_x, Index m,
                                                              Index nele_jac, Index *iRow, Index *jCol, Number *values)
{
   return false;
}

//_______________________________________________________________________
bool IpoptMinimizer::IpoptMinimizer::InternalTNLP::eval_h(Index n, const Number *x, bool new_x, Number obj_factor,
                                                          Index m, const Number *lambda, bool new_lambda,
                                                          Index nele_hess, Index *iRow, Index *jCol, Number *values)
{
   return false;
}

//_______________________________________________________________________
/**
 * This method is called by IPOPT after the algorithm has finished (successfully or even with most errors).
 * \param status (in), gives the status of the algorithm as specified in IpAlgTypes.hpp,
 *      SUCCESS: Algorithm terminated successfully at a locally optimal point, satisfying the convergence tolerances
 * (can be specified by options).
 *      MAXITER_EXCEEDED: Maximum number of iterations exceeded (can be specified by an option).
 *      CPUTIME_EXCEEDED: Maximum number of CPU seconds exceeded (can be specified by an option).
 *      STOP_AT_TINY_STEP: Algorithm proceeds with very little progress.
 *      STOP_AT_ACCEPTABLE_POINT: Algorithm stopped at a point that was converged, not to ``desired'' tolerances, but to
 * ``acceptable'' tolerances (see the acceptable-... options).
 *      LOCAL_INFEASIBILITY: Algorithm converged to a point of local infeasibility. Problem may be infeasible.
 *      USER_REQUESTED_STOP: The user call-back function intermediate_callback (see Section 3.3.4) returned false, i.e.,
 * the user code requested a premature termination of the optimization.
 *      DIVERGING_ITERATES: It seems that the iterates diverge.
 *      RESTORATION_FAILURE: Restoration phase failed, algorithm doesn't know how to proceed.
 *      ERROR_IN_STEP_COMPUTATION: An unrecoverable error occurred while IPOPT tried to compute the search direction.
 *      INVALID_NUMBER_DETECTED: Algorithm received an invalid number (such as NaN or Inf) from the NLP; see also option
 * check_derivatives_for_naninf.
 *      INTERNAL_ERROR: An unknown internal error occurred. Please contact the IPOPT authors through the mailing list.
 * \param n (in), the number of variables in the problem (dimension of $ x$).
 * \param x (in), the final values for the primal variables, $ x_*$.
 * \param z_L (in), the final values for the lower bound multipliers, $ z^L_*$.
 * \param z_U (in), the final values for the upper bound multipliers, $ z^U_*$.
 * \param m (in), the number of constraints in the problem (dimension of $ g(x)$).
 * \param g (in), the final value of the constraint function values, $ g(x_*)$.
 * \param lambda (in), the final values of the constraint multipliers, $ \lambda_*$.
 * \param obj_value (in), the final value of the objective,  $ f(x_*)$.
 * \param ip_data are provided for expert users.
 * \param ip_cq are provided for expert users.
 * This method gives you the return status of the algorithm (SolverReturn), and the values of the variables, the
 * objective and constraint function values when the algorithm exited.
 */

void IpoptMinimizer::IpoptMinimizer::InternalTNLP::finalize_solution(SolverReturn status, Index n, const Number *x,
                                                                     const Number *z_L, const Number *z_U, Index m,
                                                                     const Number *g, const Number *lambda,
                                                                     Number obj_value, const IpoptData *ip_data,
                                                                     IpoptCalculatedQuantities *ip_cq)
{
   // here is where we would store the solution to variables, or write to a file, etc
   // so we could use the solution.

   // For this example, we write the solution to the console
   std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
   for (Index i = 0; i < n; i++) {
      std::cout << "x[" << i << "] = " << x[i] << std::endl;
   }

   std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
   for (Index i = 0; i < n; i++) {
      std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
   }
   for (Index i = 0; i < n; i++) {
      std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
   }

   std::cout << std::endl << std::endl << "Objective value" << std::endl;
   std::cout << "f(x*) = " << obj_value << std::endl;

   std::cout << std::endl << "Final value of the constraints:" << std::endl;
   for (Index i = 0; i < m; i++) {
      std::cout << "g(" << i << ") = " << g[i] << std::endl;
   }
}

void IpoptMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction &func)
{
   // set the function to minimizer
   // need to calculate numerically the derivatives: do via class MultiNumGradFunction
   // no need to clone the passed function
   ROOT::Math::MultiNumGradFunction gradFunc(func);
   //     IGradientFunctionMultiDim gradFunc(func);
   // function is cloned inside so can be delete afterwards
   // called base class method setfunction
   // (note: write explicitly otherwise it will call back itself)
   BasicMinimizer::SetFunction(gradFunc);
   //    BasicMinimizer::SetFunction(func);
}

//_______________________________________________________________________
void IpoptMinimizer::SetNNZerosJacobian(UInt_t nzeros)
{
   fInternalTNLP->fNNZerosJacobian = nzeros;
}

//_______________________________________________________________________
void IpoptMinimizer::SetNNZerosHessian(UInt_t nzeros)
{
   fInternalTNLP->fNNZerosHessian = nzeros;
}

//_______________________________________________________________________
bool IpoptMinimizer::Minimize()
{
   ApplicationReturnStatus status;
   status = fIpotApp->Initialize();
   if (status != Solve_Succeeded) {
      std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
      return (int)status;
   }

   fIpotApp->Options()->SetStringValue("hessian_approximation", "limited-memory");
   status = fIpotApp->OptimizeTNLP(fInternalTNLP);

   if (status == Solve_Succeeded) {
      // Retrieve some statistics about the solve
      Index iter_count = fIpotApp->Statistics()->IterationCount();
      std::cout << std::endl << std::endl << "*** The problem solved in " << iter_count << " iterations!" << std::endl;

      Number final_obj = fIpotApp->Statistics()->FinalObjective();
      std::cout << std::endl
                << std::endl
                << "*** The final value of the objective function is " << final_obj << '.' << std::endl;
   }
}
