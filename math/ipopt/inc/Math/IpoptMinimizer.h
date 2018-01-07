// @(#)root/ipopt:$Id$
// Author: Omar.Zapata@cern.ch Thu Dec 28 2:15:00 2017

/*************************************************************************
 * Copyright (C) 2017-2018, Omar Zapata                                  *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Math_IpoptMinimizer
#define ROOT_Math_IpoptMinimizer

#include "Math/Minimizer.h"

#include "Math/IFunctionfwd.h"

#include "Math/IParamFunctionfwd.h"

#include "Math/BasicMinimizer.h"

#include <vector>
#include <map>
#include <string>

#define HAVE_CSTDDEF
#include <coin/IpTNLP.hpp>
#include <coin/IpSmartPtr.hpp>
#undef HAVE_CSTDDEF

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>

namespace ROOT {

namespace Math {
/**
   enumeration specifying the types of Ipopt solvers
   @ingroup MultiMin
*/
enum EIpoptMinimizerSolver { kMa27, kMa57, kMa77, kMa86, kMa97, kPardiso, kWsmp, kMumps };

//_____________________________________________________________________________________
/**
 * \class IpoptMinimizer
 * IpoptMinimizer class.
 * Implementation for Ipopt (Interior Point OPTimizer) is a software package for large-scale ​nonlinear optimization.
 * It is designed to find (local) solutions of mathematical optimization problems.
 *
 * The following information is required by IPOPT:
 *    - Problem dimensions
 *      -# number of variables
 *      -# number of constraints
 *    - Problem bounds
 *      -# variable bounds
 *      -# constraint bounds
 *    - Initial starting point
 *      -# Initial values for the primal \f$ x\f$ variables
 *      -# Initial values for the multipliers (only required for a warm start option)
 *    - Problem Structure
 *      -# number of nonzeros in the Jacobian of the constraints
 *      -# number of nonzeros in the Hessian of the Lagrangian function
 *      -# sparsity structure of the Jacobian of the constraints
 *      -# sparsity structure of the Hessian of the Lagrangian function
 *    - Evaluation of Problem Functions
 *      -# Information evaluated using a given point ( \f$ x,\lambda, \sigma_f\f$ coming from IPOPT)
 *      -# Objective function, \f$ f(x)\f$
 *      -# Gradient of the objective  \f$ \nabla f(x)\f$
 *      -# Constraint function values, \f$ g(x)\f$
 *      -# Jacobian of the constraints,  \f$ \nabla g(x)^T\f$
 *      -# Hessian of the Lagrangian function,  \f$ \sigma_f \nabla^2 f(x) + \sum_{i=1}^m\lambda_i\nabla^2 g_i(x)\f$
 *
 * (this is not required if a quasi-Newton options is chosen to approximate the second derivatives)
 * The problem dimensions and bounds are straightforward and come solely from the problem definition. The initial
 starting point is used by the algorithm when it begins iterating to solve the problem. If IPOPT has difficulty
 converging, or if it converges to a locally infeasible point, adjusting the starting point may help. Depending on the
 starting point, IPOPT may also converge to different local solutions.

   See <A HREF="https://projects.coin-or.org/Ipopt">Ipopt doc</A>
   from more info on the Ipopt minimization algorithms.

   @ingroup MultiMin
*/
using Ipopt::Number;
using Ipopt::Index;
using Ipopt::SolverReturn;
using Ipopt::IpoptData;
using Ipopt::IpoptCalculatedQuantities;

class IpoptMinimizer : public BasicMinimizer {
private:
   Ipopt::SmartPtr<Ipopt::IpoptApplication> fIpotApp;

protected:
   /**
   * \class InternalTNLP
   * Internal class to create a TNLP object, required for Ipopt minimization
   * in c++, every method is overloaded to pass the information to Ipopt solvers.
   * @ingroup MultiMin
   */
   class InternalTNLP : public Ipopt::TNLP {
      friend class IpoptMinimizer;
      IpoptMinimizer *fMinimizer;
      UInt_t fNNZerosJacobian;
      UInt_t fNNZerosHessian;
      Number nlp_lower_bound_inf;
      Number nlp_upper_bound_inf;

   public:
      InternalTNLP(IpoptMinimizer *minimizer);

      /** default destructor */
      virtual ~InternalTNLP();

      /**
      * Give IPOPT the information about the size of the problem (and hence, the size of the arrays that it needs to
      * allocate).
      * \param n  (out), the number of variables in the problem (dimension of \f$ x\f$).
      * \param m (out), the number of constraints in the problem (dimension of \f$ g(x)\f$).
      * \param nnz_jac_g (out), the number of nonzero entries in the Jacobian.
      * \param nnz_h_lag (out), the number of nonzero entries in the Hessian.
      * \param index_style (out), the numbering style used for row/col entries in the sparse matrix format (C_STYLE:
      * 0-based,
      * FORTRAN_STYLE: 1-based).    default C_STYLE;
      * \return true if everything is right, false in other case.
      */
      virtual bool get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag, IndexStyleEnum &index_style);
      /**
      * Give IPOPT the value of the bounds on the variables and constraints.
      * The values of n and m that you specified in get_nlp_info are passed to you for debug checking.
      * Setting a lower bound to a value less than or  equal to the value of the option nlp_lower_bound_inf will cause
      * IPOPT
      * to assume no lower bound.
      * Likewise, specifying the upper bound above or equal to the value of the option nlp_upper_bound_inf will cause
      * IPOPT to
      * assume no upper bound.
      * These options, nlp_lower_bound_inf and nlp_upper_bound_inf, are set to \f$ -10^{19}\f$ and \f$ 10^{19}\f$,
      * respectively,
      * by default.
      * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
      * \param x_l (out) the lower bounds \f$ x^L\f$ for \f$ x\f$.
      * \param x_u (out) the upper bounds \f$ x^U\f$ for \f$ x\f$.
      * \param m (in), the number of constraints in the problem (dimension of \f$ g(x)\f$).
      * \param g_l (out) the lower bounds \f$ g^L\f$ for \f$ g(x)\f$.
      * \param g_u (out) the upper bounds \f$ g^U\f$ for \f$ g(x)\f$.
      * \return true if everything is right, false in other case.
      */

      virtual bool get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l, Number *g_u);
      /**
      * Give IPOPT the starting point before it begins iterating.
      * The variables n and m are passed in for your convenience. These variables will have the same values you
      * specified in
      * get_nlp_info.
      * Depending on the options that have been set, IPOPT may or may not require bounds for the primal variables \f$
      * x\f$,
      * the bound multipliers \f$ z^L\f$ and \f$ z^U\f$, and the constraint multipliers \f$ \lambda \f$.
      * The boolean flags init_x, init_z, and init_lambda tell you whether or not you should provide initial values for
      * \f$
      * x\f$, \f$ z^L\f$, \f$ z^U\f$, or \f$ \lambda \f$ respectively.
      * The default options only require an initial value for the primal variables \f$ x\f$.
      * Note, the initial values for bound multiplier components for ``infinity'' bounds ( \f$ x_L^{(i)}=-\infty \f$ or
      * \f$
      * x_U^{(i)}=\infty \f$) are ignored.
      *
      * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
      * \param init_x (in), if true, this method must provide an initial value for \f$ x\f$.
      * \param x (out), the initial values for the primal variables, \f$ x\f$.
      * \param init_z (in), if true, this method must provide an initial value for the bound multipliers \f$ z^L\f$ and
      * \f$
      * z^U\f$.
      * \param z_L (out), the initial values for the bound multipliers, \f$ z^L\f$.
      * \param z_U (out), the initial values for the bound multipliers, \f$ z^U\f$.
      * \param m (in), the number of constraints in the problem (dimension of \f$ g(x)\f$).
      * \param init_lambda: (in), if true, this method must provide an initial value for the constraint multipliers, \f$
      * \lambda\f$.
      * \param lambda (out), the initial values for the constraint multipliers, \f$ \lambda\f$.
      * \return true if everything is right, false in other case.
      */

      virtual bool get_starting_point(Index n, bool init_x, Number *x, bool init_z, Number *z_L, Number *z_U, Index m,
                                      bool init_lambda, Number *lambda);
      /**
      * Return the value of the objective function at the point \f$ x\f$.
      * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
      * \param x (in), the values for the primal variables, \f$ x\f$, at which  \f$ f(x)\f$ is to be evaluated.
      * \param new_x (in), false if any evaluation method was previously called with the same values in x, true
      * otherwise.
      * \param obj_value (out) the value of the objective function (\f$ f(x)\f$).
      * The boolean variable new_x will be false if the last call to any of the evaluation methods (eval_*) used the
      * same \f$
      * x\f$ values.  *  * This can be helpful when users have efficient implementations that calculate multiple outputs
      * at
      * once.
      * IPOPT internally caches results from the TNLP and generally, this flag can be ignored.
      * The variable n is passed in for your convenience. This variable will have the same value you specified in
      * get_nlp_info.
      * \return true if everything is right, false in other case.
      */

      virtual bool eval_f(Index n, const Number *x, bool new_x, Number &obj_value);

      /**
       * Return the gradient of the objective function at the point \f$ x\f$.
       * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
       * \param x (in), the values for the primal variables, \f$ x\f$, at which  \f$ \nabla f(x)\f$ is to be evaluated.
       * \param new_x (in), false if any evaluation method was previously called with the same values in x, true
       * otherwise.
       * \param grad_f: (out) the array of values for the gradient of the objective function ( \f$ \nabla f(x)\f$).
       * The gradient array is in the same order as the \f$ x\f$ variables (i.e., the gradient of the objective with
       * respect to x[2] should be put in grad_f[2]).
       * The boolean variable new_x will be false if the last call to any of the evaluation methods (eval_*) used the
       * same \f$ x\f$ values.
       * This can be helpful when users have efficient implementations that calculate multiple outputs at once. IPOPT
       * internally caches results from the TNLP and generally, this flag can be ignored.
       *
       * The variable n is passed in for your convenience. This variable will have the same value you specified in
       * get_nlp_info.
       * \return true if everything is right, false in other case.
       */
      virtual bool eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f);
      /**
       * Return the value of the constraint function at the point \f$ x\f$.
       * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
       * \param x (in), the values for the primal variables, \f$ x\f$, at which the constraint functions,  \f$ g(x)\f$,
       * are to be evaluated.
       * \param new_x (in), false if any evaluation method was previously called with the same values in x, true
       * otherwise.
       * \param m (in), the number of constraints in the problem (dimension of \f$ g(x)\f$).
       * \param g (out) the array of constraint function values, \f$ g(x)\f$.
       * The values returned in g should be only the \f$ g(x)\f$ values, do not add or subtract the bound values \f$
       * g^L\f$ or \f$ g^U\f$.
       * The boolean variable new_x will be false if the last call to any of the evaluation methods (eval_*) used the
       * same \f$ x\f$ values.
       * This can be helpful when users have efficient implementations that calculate multiple outputs at once. IPOPT
       * internally caches results from the TNLP and generally, this flag can be ignored.
       * The variables n and m are passed in for your convenience. These variables will have the same values you
       * specified in get_nlp_info.
       * \return true if everything is right, false in other case.
       */
      virtual bool eval_g(Index n, const Number *x, bool new_x, Index m, Number *g);

      /**
       * Return either the sparsity structure of the Jacobian of the constraints, or the values for the Jacobian of the
       * constraints at the point \f$ x\f$.
       * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
       * \param x (in), the values for the primal variables, \f$ x\f$, at which the constraint Jacobian,  \f$ \nabla
       * g(x)^T\f$, is to be evaluated.
       * \param new_x (in), false if any evaluation method was previously called with the same values in x, true
       * otherwise.
       * \param m (in), the number of constraints in the problem (dimension of \f$ g(x)\f$).
       * \param n_ele_jac (in), the number of nonzero elements in the Jacobian (dimension of iRow, jCol, and values).
       * \param iRow (out), the row indices of entries in the Jacobian of the constraints.
       * \param jCol (out), the column indices of entries in the Jacobian of the constraints.
       * \param values (out), the values of the entries in the Jacobian of the constraints.
       * The Jacobian is the matrix of derivatives where the derivative of constraint \f$ g^{(i)}\f$ with respect to
       * variable \f$ x^{(j)}\f$ is placed in row \f$ i\f$ and column \f$ j\f$. See Appendix A for a discussion of the
       * sparse matrix format used in this method.
       *
       * If the iRow and jCol arguments are not NULL, then IPOPT wants you to fill in the sparsity structure of the
       * Jacobian (the row and column indices only). At this time, the x argument and the values argument will be NULL.
       *
       * If the x argument and the values argument are not NULL, then IPOPT wants you to fill in the values of the
       * Jacobian as calculated from the array x (using the same order as you used when specifying the sparsity
       * structure). At this time, the iRow and jCol arguments will be NULL;
       *
       * The boolean variable new_x will be false if the last call to any of the evaluation methods (eval_*) used the
       * same \f$ x\f$ values. This can be helpful when users have efficient implementations that calculate multiple
       * outputs at once. IPOPT internally caches results from the TNLP and generally, this flag can be ignored.
       *
       * The variables n, m, and nele_jac are passed in for your convenience. These arguments will have the same values
       * you specified in get_nlp_info.
       * \return true if everything is right, false in other case.
       */
      virtual bool eval_jac_g(Index n, const Number *x, bool new_x, Index m, Index nele_jac, Index *iRow, Index *jCol,
                              Number *values);
      /**
       * Return either the sparsity structure of the Hessian of the Lagrangian, or the values of the Hessian of the
       * Lagrangian <a href="https://www.coin-or.org/Ipopt/documentation/node22.html#eq:IpoptLAG">(9)</a> for the given
       * values for \f$ x\f$, \f$ \sigma_f\f$, and \f$ \lambda\f$.
       * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
       * \param x (in), the values for the primal variables, \f$ x\f$, at which the Hessian is to be evaluated.
       * \param new_x (in), false if any evaluation method was previously called with the same values in x, true
       * otherwise.
       * \param obj_factor (in), factor in front of the objective term in the Hessian, \f$ \sigma_f\f$.
       * \param m (in), the number of constraints in the problem (dimension of \f$ g(x)\f$).
       * \param lambda (in), the values for the constraint multipliers,  \f$ \lambda\f$, at which the Hessian is to be
       * evaluated.
       * \param new_lambda (in), false if any evaluation method was previously called with the same values in lambda,
       * true otherwise.
       * \param nele_hess (in), the number of nonzero elements in the Hessian (dimension of iRow, jCol, and values).
       * \param iRow (out), the row indices of entries in the Hessian.
       * \param jCol (out), the column indices of entries in the Hessian.
       * \param values (out), the values of the entries in the Hessian.
       *
       * The Hessian matrix that IPOPT uses is defined in <a
       * href="https://www.coin-or.org/Ipopt/documentation/node22.html#eq:IpoptLAG">(9)</a>. See Appendix A for a
       * discussion of the sparse symmetric matrix format used in this method.
       *
       * If the iRow and jCol arguments are not NULL, then IPOPT wants you to fill in the sparsity structure of the
       * Hessian (the row and column indices for the lower or upper triangular part only). In this case, the x, lambda,
       * and values arrays will be NULL.
       * \return true if everything is right, false in other case.
       */
      virtual bool eval_h(Index n, const Number *x, bool new_x, Number obj_factor, Index m, const Number *lambda,
                          bool new_lambda, Index nele_hess, Index *iRow, Index *jCol, Number *values);
      /**
      * This method is called by IPOPT after the algorithm has finished (successfully or even with most errors).
      * \param status (in), gives the status of the algorithm as specified in IpAlgTypes.hpp,
      *      SUCCESS: Algorithm terminated successfully at a locally optimal point, satisfying the convergence
      * tolerances
      * (can be specified by options).
      *      MAXITER_EXCEEDED: Maximum number of iterations exceeded (can be specified by an option).
      *      CPUTIME_EXCEEDED: Maximum number of CPU seconds exceeded (can be specified by an option).
      *      STOP_AT_TINY_STEP: Algorithm proceeds with very little progress.
      *      STOP_AT_ACCEPTABLE_POINT: Algorithm stopped at a point that was converged, not to ``desired'' tolerances,
      * but to
      * ``acceptable'' tolerances (see the acceptable-... options).
      *      LOCAL_INFEASIBILITY: Algorithm converged to a point of local infeasibility. Problem may be infeasible.
      *      USER_REQUESTED_STOP: The user call-back function intermediate_callback (see Section 3.3.4) returned false,
      * i.e.,
      * the user code requested a premature termination of the optimization.
      *      DIVERGING_ITERATES: It seems that the iterates diverge.
      *      RESTORATION_FAILURE: Restoration phase failed, algorithm doesn't know how to proceed.
      *      ERROR_IN_STEP_COMPUTATION: An unrecoverable error occurred while IPOPT tried to compute the search
      * direction.
      *      INVALID_NUMBER_DETECTED: Algorithm received an invalid number (such as NaN or Inf) from the NLP; see also
      * option
      * check_derivatives_for_naninf.
      *      INTERNAL_ERROR: An unknown internal error occurred. Please contact the IPOPT authors through the mailing
      * list.
      * \param n (in), the number of variables in the problem (dimension of \f$ x\f$).
      * \param x (in), the final values for the primal variables, \f$ x_*\f$.
      * \param z_L (in), the final values for the lower bound multipliers, \f$ z^L_*\f$.
      * \param z_U (in), the final values for the upper bound multipliers, \f$ z^U_*\f$.
      * \param m (in), the number of constraints in the problem (dimension of \f$ g(x)\f$).
      * \param g (in), the final value of the constraint function values, \f$ g(x_*)\f$.
      * \param lambda (in), the final values of the constraint multipliers, \f$ \lambda_*\f$.
      * \param obj_value (in), the final value of the objective,  \f$ f(x_*)\f$.
      * \param ip_data are provided for expert users.
      * \param ip_cq are provided for expert users.
      * This method gives you the return status of the algorithm (SolverReturn), and the values of the variables, the
      * objective and constraint function values when the algorithm exited.
      * \return true if everything is right, false in other case.
      */
      virtual void finalize_solution(SolverReturn status, Index n, const Number *x, const Number *z_L,
                                     const Number *z_U, Index m, const Number *g, const Number *lambda,
                                     Number obj_value, const IpoptData *ip_data, IpoptCalculatedQuantities *ip_cq);

   private:
      /** @name Methods to block default compiler methods.
       * The compiler automatically generates the following three methods.
       *  Since the default compiler implementation is generally not what
       *  you want (for all but the most simple classes), we usually
       *  put the declarations of these methods in the private section
       *  and never implement them. This prevents the compiler from
       *  implementing an incorrect "default" behavior without us
       *  knowing. (See Scott Meyers book, "Effective C++")
       *
       */
      InternalTNLP(const InternalTNLP &);
      InternalTNLP &operator=(const InternalTNLP &);
   };

public:
   /**
      Default constructor
   */
   IpoptMinimizer();
   /**
      Constructor with a string giving name of algorithm
   */
   IpoptMinimizer(const char *type);

   /**
      Destructor
   */
   virtual ~IpoptMinimizer();

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   IpoptMinimizer(const IpoptMinimizer &) : BasicMinimizer() {}

   /**
      Assignment operator
   */
   IpoptMinimizer &operator=(const IpoptMinimizer &rhs)
   {
      if (this == &rhs) {
         return *this; // time saving self-test
      }
      return *this;
   }

public:
   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGenFunction &func);

   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGradFunction &func) { BasicMinimizer::SetFunction(func); }

   /// method to perform the minimization
   virtual bool Minimize();

   virtual void SetNNZerosJacobian(UInt_t nzeros);

   virtual void SetNNZerosHessian(UInt_t nzeros);

   virtual void SetOptionStringValue(const char *var, const char *value);

   /// return expected distance reached from the minimum
   virtual double Edm() const { return 0; } // not impl. }

   /// return pointer to gradient values at the minimum
   //     virtual const double *MinGradient() const;

   /// number of function calls to reach the minimum
   //     virtual unsigned int NCalls() const;

   /// minimizer provides error and error matrix
   virtual bool ProvidesError() const { return false; }

   /// return errors at the minimum
   virtual const double *Errors() const { return 0; }

   /** return covariance matrices elements
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */
   virtual double CovMatrix(unsigned int, unsigned int) const { return 0; }
protected:
   Ipopt::SmartPtr<InternalTNLP> fInternalTNLP;
   ClassDef(IpoptMinimizer, 0) //
};

} // end namespace Math

} // end namespace ROOT

#endif /* ROOT_Math_IpoptMinimizer */
