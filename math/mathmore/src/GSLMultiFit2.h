// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 17:26:06 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 * This library is free software; you can redistribute it and/or      *
 * modify it under the terms of the GNU General Public License        *
 * as published by the Free Software Foundation; either version 2     *
 * of the License, or (at your option) any later version.             *
 *                                                                    *
 * This library is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
 * General Public License for more details.                           *
 *                                                                    *
 * You should have received a copy of the GNU General Public License  *
 * along with this library (see file COPYING); if not, write          *
 * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
 * 330, Boston, MA 02111-1307 USA, or contact the author.             *
 *                                                                    *
 **********************************************************************/

// Header file for class GSLMultiFit

#ifndef ROOT_Math_GSLMultiFit2
#define ROOT_Math_GSLMultiFit2

#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_multifit_nlinear.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_version.h"
#include "GSLMultiFitFunctionWrapper.h"

#include "Math/IFunction.h"
#include "Math/MinimizerOptions.h"
#include "Math/GenAlgoOptions.h"
#include <string>
#include <vector>
#include <cassert>

//#if GSL_MAJOR_VERSION > 1

namespace ROOT {

   namespace Math {


/**
   GSLMultiFit2, internal class for implementing GSL non linear least square GSL fitting
   New class implementing new GSL non linear fitting methods introduced in GSL 2.2

   @ingroup MultiMin
*/
class GSLMultiFit2 {


public:

   /**
      Default constructor
      No need to specify the type so far since only one solver exists so far
   */
   GSLMultiFit2 (int type = 0) :
      //fSolver(0),
      fVec(0),
      //fTmp(0),
      fCov(0)
      //fJac(0)
   {
      fType = gsl_multifit_nlinear_trust; // default value (unique for the moment)
      fParams = gsl_multifit_nlinear_default_parameters();
      if (type == 1)  // lm
         fParams.trs =  gsl_multifit_nlinear_trs_lm;
      else if (type == 2)
         fParams.trs =  gsl_multifit_nlinear_trs_lmaccel;
      else if (type == 3)
         fParams.trs =  gsl_multifit_nlinear_trs_dogleg;
      else if (type == 4)
         fParams.trs =  gsl_multifit_nlinear_trs_ddogleg;
      else if (type == 5)
         fParams.trs =  gsl_multifit_nlinear_trs_subspace2D;

   }

   ROOT::Math::GenAlgoOptions GetDefaultOptions() const
   {
      // set the extra minimizer options from GSLMultiFit
      ROOT::Math::GenAlgoOptions opt;
      opt.SetValue("scale", fParams.scale->name);
      opt.SetValue("solver", fParams.solver->name);
      opt.SetValue("fdtype", fParams.fdtype);
      opt.SetValue("factor_up", fParams.factor_up);
      opt.SetValue("factor_down", fParams.factor_down);
      opt.SetValue("avmax", fParams.avmax);
      opt.SetValue("h_df", fParams.h_df);
      opt.SetValue("h_fvv", fParams.h_fvv);

      return opt;
   }

   void SetParameters(const ROOT::Math::MinimizerOptions & minimOptions)
   {
      // set the parameters from the minimizer options
      fPrintLevel = minimOptions.PrintLevel();
      if (minimOptions.Tolerance() > 0) fTolerance = minimOptions.Tolerance();
      if (minimOptions.MaxIterations() > 0) fMaxIter = minimOptions.MaxIterations();
      // now specific options to set the specific GSL parameters
      const ROOT::Math::IOptions  * opt = minimOptions.ExtraOptions();
      if (!opt)
         return;
      std::string sval;
      opt->GetValue("scale",sval);
      if (sval == "more")
         fParams.scale = gsl_multifit_nlinear_scale_more;
      else if (sval == "levenberg")
         fParams.scale = gsl_multifit_nlinear_scale_levenberg;
      else if (sval == "marquardt")
         fParams.scale = gsl_multifit_nlinear_scale_marquardt;
      else {
         if (fPrintLevel > 0)
            std::cout << "GSLMultiFit2::SetParameters use default scale : "
                  << fParams.scale->name << std::endl;
      }
      opt->GetValue("solver",sval);
      if (sval == "qr")
         fParams.solver = gsl_multifit_nlinear_solver_qr;
      else if (sval == "cholesky")
         fParams.solver = gsl_multifit_nlinear_solver_cholesky;
#if ((GSL_MAJOR_VERSION >= 2) && (GSL_MINOR_VERSION > 5))
      else if (sval == "mcholesky")
         fParams.solver = gsl_multifit_nlinear_solver_mcholesky;
#endif
      else if (sval == "svd")
         fParams.solver = gsl_multifit_nlinear_solver_svd;
      else {
          if (fPrintLevel > 0)
            std::cout << "GSLMultiFit2::SetParameters use default solver : "
                  << fParams.solver->name << std::endl;
      }
      double val;
      opt->GetValue("factor_up",val);
      fParams.factor_up = val;
      opt->GetValue("factor_down",val);
      fParams.factor_down = val;


   }

   /**
      Destructor (no operations)
   */
   ~GSLMultiFit2 ()  {
      if (fWs) gsl_multifit_nlinear_free(fWs);
      if (fVec != 0) gsl_vector_free(fVec);
      if (fCov != 0) gsl_matrix_free(fCov);
   }

   void PrintOptions() const {
      std::cout << "GSLMultiFit: Parameters used for solving the non-linear fit problem" << std::endl;
      std::cout << "\t\t Solver for trust region   : " << fParams.trs->name << std::endl;
      std::cout << "\t\t Scaling method            : " << fParams.scale->name << std::endl;
      std::cout << "\t\t Solver method for GN step : " << fParams.solver->name << std::endl;
      std::cout << "\t\t Finite difference type    : " << fParams.fdtype << std::endl;
      std::cout << "\t\t Factor TR up              : " << fParams.factor_up << std::endl;
      std::cout << "\t\t Factor TR down            : " << fParams.factor_down << std::endl;
      std::cout << "\t\t Max allowed |a|/|v|       : " << fParams.avmax << std::endl;
      std::cout << "\t\t Step size for deriv       : " << fParams.h_df << std::endl;
      std::cout << "\t\t Step size for fvv         : " << fParams.h_fvv << std::endl;
      std::cout << "\t\t Max number of iterations  : " << fMaxIter << std::endl;
      std::cout << "\t\t Tolerance                 : " << fTolerance << std::endl;
   }

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   GSLMultiFit2(const GSLMultiFit &) {}

   /**
      Assignment operator
   */
   GSLMultiFit2 & operator = (const GSLMultiFit2 & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }


public:


   /// set the solver parameters
   template<class Func>
   int Set(const std::vector<Func> & funcVec, const double * x) {
      // create a vector of the fit contributions
      // create function wrapper from an iterator of functions
      unsigned int npts = funcVec.size();
      if (npts == 0) return -1;

      unsigned int npar = funcVec[0].NDim();

      // Remove unused typedef to remove warning in GCC48
      // http://gcc.gnu.org/gcc-4.8/porting_to.html
      // typedef typename std::vector<Func>  FuncVec;
      //FuncIt funcIter = funcVec.begin();
      SetFunction(funcVec, npts, npar);

       // set initial values
      if (fVec != 0) gsl_vector_free(fVec);
      fVec = gsl_vector_alloc( npar );
      std::copy(x,x+npar, fVec->data);

      if (fCov != 0) gsl_matrix_free(fCov);
      fCov = gsl_matrix_alloc( npar, npar );
      return 0;
   }

   std::string Name() const {
      if (fWs == nullptr) return "undefined";
      return std::string(gsl_multifit_nlinear_name(fWs) );
   }

   int Iterate() {
      return -1;
   }


   int Solve() {
      int npts = fFunc.n;
      int npar = fFunc.p;

      if (fPrintLevel > 0)
         PrintOptions();

      /* allocate workspace with default parameters */
      fWs = gsl_multifit_nlinear_alloc (fType, &fParams, npts, npar);

      /* initialize solver with starting point and weights */
      gsl_multifit_nlinear_init (fVec, &fFunc, fWs);
      // in case of weights
      //gsl_multifit_nlinear_winit (&x.vector, &wts.vector, &fdf, w);

      /* compute initial cost function */
      gsl_vector * f = gsl_multifit_nlinear_residual(fWs);
      double chisq0;
      gsl_blas_ddot(f, f, &chisq0);

      // use slightly larger tolerance for gradient
      const double xtol = fTolerance;
      const double gtol = 10*fTolerance;
      const double ftol = 0.0;

      /* solve the system with a maximum of 100 iterations */
      int info = 0;
      void * callback_params = (fPrintLevel > 0) ? &fPrintLevel : nullptr;
      int status = gsl_multifit_nlinear_driver(fMaxIter, xtol, gtol, ftol,
                                       GSLMultiFit2::Callback, callback_params , &info, fWs);

      /* compute covariance of best fit parameters */
      fJac = gsl_multifit_nlinear_jac (fWs);
      gsl_multifit_nlinear_covar (fJac, 0.0, fCov);

     /* compute final cost */
     double chisq;
     gsl_blas_ddot(f, f, &chisq);

      return status;

   }

   static void Callback(const size_t iter, void * params , const gsl_multifit_nlinear_workspace *w) {
      // return in case of printLevel=0
      if (params == nullptr) return;

      gsl_vector *f = gsl_multifit_nlinear_residual(w);
      gsl_vector *x = gsl_multifit_nlinear_position(w);
      double rcond = 0;

     /* compute reciprocal condition number of J(x) */
      gsl_multifit_nlinear_rcond(&rcond, w);

      printf("iter %2zu: x = {",iter);
      for (unsigned int i = 0; i < x->size; i++)
         printf(" %.4f,",gsl_vector_get(x, i));
      printf(" } cond(J) = %8.4f, |f(x)| = %.4f\n",1.0 / rcond, gsl_blas_dnrm2(f));

      // trust state is not accessible
      //gsl_multifit_nlinear_trust_state * state = (gsl_multifit_nlinear_trust_state *) w->state;
      printf("        step  :  = {");
      for (unsigned int i = 0; i < w->dx->size; i++)
         printf(" %.4f,",gsl_vector_get(w->dx, i));
      printf("\n");
      printf("        gradient  :  = {");
      for (unsigned int i = 0; i < w->g->size; i++)
         printf(" %.4f,",gsl_vector_get(w->g, i));
      printf("\n");
      //if (state) {
      // printf("   diagonal : D = {");
      // for (unsigned int i = 0; i < state->diag->size; i++)
      //    printf(" %.4f,",gsl_vector_get(state->diag, i));
      // }

   }

   int NIter() const {
      return gsl_multifit_nlinear_niter(fWs);
   }

   /// parameter values at the minimum
   const double * X() const {
      if (!fWs) return nullptr;
      gsl_vector * x =  gsl_multifit_nlinear_position(fWs);
      return x->data;
   }

    /// return covariance matrix of the parameters
   const double * CovarMatrix() const {
      return (fCov) ? fCov->data : nullptr;
   }

   /// gradient value at the minimum
   const double * Gradient() const {
      return nullptr;
   }

   // this functions are not used (kept for keeping same interface as old GSLMultiFit class )
   /// test gradient (ask from solver gradient vector)
   int TestGradient(double /* absTol */ ) const {
      return -1;
   }

   /// test using abs and relative tolerance
   ///  |dx| < absTol + relTol*|x| for every component
   int TestDelta(double /* absTol */, double /* relTol */) const {
      return -1;
   }

   // calculate edm  1/2 * ( grad^T * Cov * grad )
   double Edm() const {
      return -1;
   }

protected:

   template<class FuncVector>
   void SetFunction(const FuncVector & f, unsigned int nres, unsigned int npar  ) {
      const void * p = &f;
      assert (p != 0);
      fFunc.f   = &GSLMultiFitFunctionAdapter<FuncVector >::F;
      // set to null for internal finite diff
      fFunc.df  = &GSLMultiFitFunctionAdapter<FuncVector >::Df;
      //fFunc.fdf = &GSLMultiFitFunctionAdapter<FuncVector >::FDf;
      fFunc.fvv = NULL;     /* not using geodesic acceleration */
      fFunc.n = nres;
      fFunc.p = npar;
      fFunc.params =  const_cast<void *>(p);
   }

private:

   int fPrintLevel = 0;
   int fMaxIter = 100;
   double fTolerance = 1.E-6;
   gsl_multifit_nlinear_fdf fFunc;
   gsl_multifit_nlinear_workspace * fWs;
   //gsl_multifit_fdfsolver * fSolver;
   // cached vector to avoid re-allocating every time a new one
   mutable gsl_vector * fVec;
   //mutable gsl_vector * fTmp;
   mutable gsl_matrix * fCov;
   mutable gsl_matrix * fJac;

   const gsl_multifit_nlinear_type * fType;
   gsl_multifit_nlinear_parameters fParams;


};

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLMultiFit2 */
