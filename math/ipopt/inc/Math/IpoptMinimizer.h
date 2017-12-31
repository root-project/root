// @(#)root/ipopt:$Id$
// Author: Omar.Zapata@cern.ch Thu Dec 28 2:15:00 2017

/*************************************************************************
 * Copyright (C) 2017, Omar Andres Zapata Mesa                           *
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
#include <cstddef>
#include <coin/IpTNLP.hpp>
#include <coin/IpSmartPtr.hpp>
#undef HAVE_CSTDDEF

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>

namespace ROOT {

namespace Math {

//_____________________________________________________________________________________
/**
   IpoptMinimizer class.
   Implementation for Ipopt (Interior Point OPTimizer) is a software package for large-scale ​nonlinear optimization.
   It is designed to find (local) solutions of mathematical optimization problems.

   See <A HREF="https://projects.coin-or.org/Ipopt">Ipopt doc</A>
   from more info on the Ipopt minimization algorithms.

   @ingroup MultiMin
*/
using Ipopt::Number;
using Ipopt::Index;
using Ipopt::SolverReturn;
using Ipopt::IpoptData;
using Ipopt::IpoptData;
using Ipopt::IpoptCalculatedQuantities;

class IpoptMinimizer : public ROOT::Math::BasicMinimizer {
protected:
   class InternalTNLP : public Ipopt::TNLP {
      InternalTNLP();

      /** default destructor */
      virtual ~InternalTNLP();

      /**@name Overloaded from TNLP */
      /** Method to return some info about the nlp */
      virtual bool get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag, IndexStyleEnum &index_style);

      /** Method to return the bounds for my problem */
      virtual bool get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l, Number *g_u);

      /** Method to return the starting point for the algorithm */
      virtual bool get_starting_point(Index n, bool init_x, Number *x, bool init_z, Number *z_L, Number *z_U, Index m,
                                      bool init_lambda, Number *lambda);

      /** Method to return the objective value */
      virtual bool eval_f(Index n, const Number *x, bool new_x, Number &obj_value);

      /** Method to return the gradient of the objective */
      virtual bool eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f);

      /** Method to return the constraint residuals */
      virtual bool eval_g(Index n, const Number *x, bool new_x, Index m, Number *g);

      /** Method to return:
       *   1) The structure of the jacobian (if "values" is NULL)
       *   2) The values of the jacobian (if "values" is not NULL)
       */
      virtual bool eval_jac_g(Index n, const Number *x, bool new_x, Index m, Index nele_jac, Index *iRow, Index *jCol,
                              Number *values);

      /** Method to return:
       *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
       *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
       */
      virtual bool eval_h(Index n, const Number *x, bool new_x, Number obj_factor, Index m, const Number *lambda,
                          bool new_lambda, Index nele_hess, Index *iRow, Index *jCol, Number *values);

      /** @name Solution Methods */
      /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
      virtual void finalize_solution(SolverReturn status, Index n, const Number *x, const Number *z_L,
                                     const Number *z_U, Index m, const Number *g, const Number *lambda,
                                     Number obj_value, const IpoptData *ip_data, IpoptCalculatedQuantities *ip_cq);

   private:
      /**@name Methods to block default compiler methods.
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
      if (this == &rhs)
         return *this; // time saving self-test
      return *this;
   }

public:
   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGenFunction &func);

   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGradFunction &func) { BasicMinimizer::SetFunction(func); }

   /// method to perform the minimization
   virtual bool Minimize();

   /// return expected distance reached from the minimum
   virtual double Edm() const { return 0; } // not impl. }

   /// return pointer to gradient values at the minimum
   virtual const double *MinGradient() const;

   /// number of function calls to reach the minimum
   virtual unsigned int NCalls() const;

   /// minimizer provides error and error matrix
   virtual bool ProvidesError() const { return false; }

   /// return errors at the minimum
   virtual const double *Errors() const { return 0; }

   /** return covariance matrices elements
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */
   virtual double CovMatrix(unsigned int, unsigned int) const { return 0; }
};

} // end namespace Fit

} // end namespace ROOT

#endif /* ROOT_Math_IpoptMinimizer */
