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

#ifndef ROOT_Math_GSLMultiFit
#define ROOT_Math_GSLMultiFit

#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_multifit_nlin.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_version.h"
#include "GSLMultiFitFunctionWrapper.h"

#include "Math/IFunction.h"
#include <string>
#include <vector>
#include <cassert>


namespace ROOT {

   namespace Math {


/**
   GSLMultiFit, internal class for implementing GSL non linear least square GSL fitting

   @ingroup MultiMin
*/
class GSLMultiFit {

public:

   /**
      Default constructor
      No need to specify the type so far since only one solver exists so far
   */
   GSLMultiFit (const gsl_multifit_fdfsolver_type * type = nullptr) :
      fSolver(nullptr),
      fVec(nullptr),
      fTmp(nullptr),
      fCov(nullptr),
#if GSL_MAJOR_VERSION > 1
      fJac(nullptr),
#endif
      fType(type)
   {
      if (fType == nullptr) fType = gsl_multifit_fdfsolver_lmsder; // default value
   }

   /**
      Destructor (no operations)
   */
   ~GSLMultiFit ()  {
      if (fSolver) gsl_multifit_fdfsolver_free(fSolver);
      if (fVec != nullptr) gsl_vector_free(fVec);
      if (fTmp != nullptr) gsl_vector_free(fTmp);
      if (fCov != nullptr) gsl_matrix_free(fCov);
#if GSL_MAJOR_VERSION > 1
      if (fJac != nullptr) gsl_matrix_free(fJac);
#endif
   }

   // usually copying is non trivial, so we delete this
   GSLMultiFit(const GSLMultiFit &) = delete;
   GSLMultiFit & operator = (const GSLMultiFit & rhs) = delete;
   GSLMultiFit(GSLMultiFit &&) = delete;
   GSLMultiFit & operator = (GSLMultiFit && rhs) = delete;

   /// create the minimizer from the type and size of number of fitting points and number of parameters
   void CreateSolver(unsigned int npoints, unsigned int npar) {
      if (fSolver) gsl_multifit_fdfsolver_free(fSolver);
      fSolver = gsl_multifit_fdfsolver_alloc(fType, npoints, npar);
      if (fVec != nullptr) gsl_vector_free(fVec);
      fVec = gsl_vector_alloc( npar );
      if (fTmp != nullptr) gsl_vector_free(fTmp);
      fTmp = gsl_vector_alloc( npar );
      if (fCov != nullptr) gsl_matrix_free(fCov);
      fCov = gsl_matrix_alloc( npar, npar );
#if GSL_MAJOR_VERSION > 1
      if (fJac != nullptr) gsl_matrix_free(fJac);
      fJac = gsl_matrix_alloc( npoints, npar );
#endif
   }

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
      fFunc.SetFunction(funcVec, npts, npar);
      // create solver object
      CreateSolver( npts, npar );
      assert(fSolver != nullptr);
      // set initial values
      assert(fVec != nullptr);
      std::copy(x,x+npar, fVec->data);
      assert(fTmp != nullptr);
      gsl_vector_set_zero(fTmp);
      assert(fCov != nullptr);
      gsl_matrix_set_zero(fCov);
#if GSL_MAJOR_VERSION > 1
      assert(fJac != nullptr);
      gsl_matrix_set_zero(fJac);
#endif
      return gsl_multifit_fdfsolver_set(fSolver, fFunc.GetFunc(), fVec);
   }

   std::string Name() const {
      if (fSolver == nullptr) return "undefined";
      return std::string(gsl_multifit_fdfsolver_name(fSolver) );
   }

   int Iterate() {
      if (fSolver == nullptr) return -1;
      return gsl_multifit_fdfsolver_iterate(fSolver);
   }

   /// parameter values at the minimum
   const double * X() const {
      if (fSolver == nullptr) return nullptr;
      return fSolver->x->data;
   }

   /// gradient value at the minimum
   const double * Gradient() const {
      if (fSolver == nullptr) return nullptr;
#if GSL_MAJOR_VERSION  > 1
      fType->gradient(fSolver->state, fVec);
#else
      gsl_multifit_gradient(fSolver->J, fSolver->f,fVec);
#endif
      return fVec->data;
   }

   /// return covariance matrix of the parameters
   const double * CovarMatrix() const {
      if (fSolver == nullptr) return nullptr;
      static double kEpsrel = 0.0001;
#if GSL_MAJOR_VERSION > 1
      gsl_multifit_fdfsolver_jac (fSolver, fJac);
      int ret = gsl_multifit_covar(fJac, kEpsrel, fCov);
#else
      int ret = gsl_multifit_covar(fSolver->J, kEpsrel, fCov);
#endif
      if (ret != GSL_SUCCESS) return nullptr;
      return fCov->data;
   }

   /// test gradient (ask from solver gradient vector)
   int TestGradient(double absTol) const {
      if (fSolver == nullptr) return -1;
      Gradient();
      return gsl_multifit_test_gradient( fVec, absTol);
   }

   /// test using abs and relative tolerance
   ///  |dx| < absTol + relTol*|x| for every component
   int TestDelta(double absTol, double relTol) const {
      if (fSolver == nullptr) return -1;
      return gsl_multifit_test_delta(fSolver->dx, fSolver->x, absTol, relTol);
   }

   // calculate edm  1/2 * ( grad^T * Cov * grad )
   double Edm() const {
      // product C * g
      double edm = -1;
      const double * g = Gradient();
      if (g == nullptr) return edm;
      const double * c = CovarMatrix();
      if (c == nullptr) return edm;
      if (fTmp == nullptr) return edm;
      int status =   gsl_blas_dgemv(CblasNoTrans, 1.0, fCov, fVec, 0.,fTmp);
      if (status == 0) status |= gsl_blas_ddot(fVec, fTmp, &edm);
      if (status != 0) return -1;
      // need to divide by 2 ??
      return 0.5*edm;

   }


private:

   GSLMultiFitFunctionWrapper fFunc;
   gsl_multifit_fdfsolver * fSolver;
   // cached vector to avoid re-allocating every time a new one
   mutable gsl_vector * fVec;
   mutable gsl_vector * fTmp;
   mutable gsl_matrix * fCov;
#if GSL_MAJOR_VERSION > 1
   mutable gsl_matrix * fJac;
#endif
   const gsl_multifit_fdfsolver_type * fType;

};

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLMultiFit */
