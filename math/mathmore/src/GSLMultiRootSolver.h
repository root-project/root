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

// Header file for the class GSLMultiRootBaseSolver,
//     GSLMultiRootSolver and GSLMultiRootDerivSolver

#ifndef ROOT_Math_GSLMultiRootSolver
#define ROOT_Math_GSLMultiRootSolver

#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_multiroots.h"
#include "gsl/gsl_blas.h"
#include "GSLMultiRootFunctionWrapper.h"

#include "Math/IFunction.h"
#include "Math/Error.h"

#include <vector>
#include <string>
#include <cassert>


namespace ROOT {

   namespace Math {


/**
   GSLMultiRootBaseSolver, internal class for implementing GSL multi-root finders
   This is the base class for GSLMultiRootSolver (solver not using derivatives) and
   GSLMUltiRootDerivSolver (solver using derivatives)

   @ingroup MultiRoot
*/
class GSLMultiRootBaseSolver {

public:

   /**
      virtual Destructor
   */
   virtual ~GSLMultiRootBaseSolver ()  {}


public:


   /// init the solver with function list and initial values
   bool InitSolver(const std::vector<ROOT::Math::IMultiGenFunction*> & funcVec, const double * x) {
      // create a vector of the fit contributions
      // create function wrapper from an iterator of functions
      unsigned int n = funcVec.size();
      if (n == 0) return false;

      unsigned int ndim = funcVec[0]->NDim();   // should also be = n

      if (ndim != n) {
         MATH_ERROR_MSGVAL("GSLMultiRootSolver::InitSolver","Wrong function dimension",ndim);
         MATH_ERROR_MSGVAL("GSLMultiRootSolver::InitSolver","Number of functions",n);
         return false;
      }


      // set function list and initial values in solver
      int iret = SetSolver(funcVec,x);
      return (iret == 0);
   }

   /// return name
   virtual const std::string & Name() const  = 0;

   /// perform an iteration
   virtual int Iterate() = 0;

   /// solution values at the current iteration
   const double * X() const {
      gsl_vector * x = GetRoot();
      return x->data;
   }

   /// return function values
   const double * FVal() const {
      gsl_vector * f = GetF();
      return f->data;
   }

   /// return function steps
   const double * Dx() const {
      gsl_vector * dx = GetDx();
      return dx->data;
   }

   /// test using abs and relative tolerance
   ///  |dx| < absTol + relTol*|x| for every component
   int TestDelta(double absTol, double relTol) const {
      gsl_vector * x =  GetRoot();
      gsl_vector * dx =  GetDx();
      if (x == 0 || dx == 0) return -1;
      return gsl_multiroot_test_delta(dx, x, absTol, relTol);
   }

   /// test using abs  tolerance
   /// Sum |f|_i < absTol
   int TestResidual(double absTol) const {
      gsl_vector * f =  GetF();
      if (f == 0) return -1;
      return gsl_multiroot_test_residual(f, absTol);
   }


private:

   // accessor to be implemented by the derived classes

   virtual int SetSolver(const std::vector<ROOT::Math::IMultiGenFunction*> & funcVec, const double * x) = 0;

   virtual gsl_vector * GetRoot() const = 0;

   virtual gsl_vector * GetF() const = 0;

   virtual gsl_vector * GetDx() const = 0;


};


/**
   GSLMultiRootSolver, internal class for implementing GSL multi-root finders
   not using derivatives

   @ingroup MultiRoot
*/
class GSLMultiRootSolver : public GSLMultiRootBaseSolver {

public:

   /**
      Constructor from type and simension of system (number of functions)
   */
   GSLMultiRootSolver (const gsl_multiroot_fsolver_type * type, int n ) :
      fSolver(0),
      fVec(0),
      fName(std::string("undefined"))
   {
      CreateSolver(type, n);
   }

   /**
      Destructor (no operations)
   */
   ~GSLMultiRootSolver () override  {
      if (fSolver) gsl_multiroot_fsolver_free(fSolver);
      if (fVec != 0) gsl_vector_free(fVec);
   }

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   GSLMultiRootSolver(const GSLMultiRootSolver &) : GSLMultiRootBaseSolver() {}

   /**
      Assignment operator
   */
   GSLMultiRootSolver & operator = (const GSLMultiRootSolver & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }


public:


   void  CreateSolver(const  gsl_multiroot_fsolver_type * type, unsigned int n) {

      /// create the solver from the type and size of number of fitting points and number of parameters
      if (fSolver) gsl_multiroot_fsolver_free(fSolver);
      fSolver = gsl_multiroot_fsolver_alloc(type, n);
      fName =  std::string(gsl_multiroot_fsolver_name(fSolver) );
   }


   /// set the solver parameters
   int SetSolver(const std::vector<ROOT::Math::IMultiGenFunction*> & funcVec, const double * x) override {
      // create a vector of the fit contributions
      // create function wrapper from an iterator of functions
      assert(fSolver !=0);
      unsigned int n = funcVec.size();

      fFunctions.SetFunctions(funcVec, funcVec.size() );
      // set initial values and create cached vector
      if (fVec != 0) gsl_vector_free(fVec);
      fVec = gsl_vector_alloc( n);
      std::copy(x,x+n, fVec->data);
      // solver should have been already created at this point
      assert(fSolver != 0);
      return gsl_multiroot_fsolver_set(fSolver, fFunctions.GetFunctions(), fVec);
   }

   const std::string & Name() const override {
      return fName; 
   }

   int Iterate() override {
      if (fSolver == 0) return -1;
      return gsl_multiroot_fsolver_iterate(fSolver);
   }

   /// solution values at the current iteration
   gsl_vector * GetRoot() const override {
      if (fSolver == 0) return 0;
      return  gsl_multiroot_fsolver_root(fSolver);
   }

   /// return function values
   gsl_vector * GetF() const override {
      if (fSolver == 0) return 0;
      return  gsl_multiroot_fsolver_f(fSolver);
   }

   /// return function steps
   gsl_vector * GetDx() const override {
      if (fSolver == 0) return 0;
      return gsl_multiroot_fsolver_dx(fSolver);
   }


private:

   GSLMultiRootFunctionWrapper fFunctions;
   gsl_multiroot_fsolver * fSolver;
   // cached vector to avoid re-allocating every time a new one
   mutable gsl_vector * fVec;
   std::string fName;   // solver nane

};

/**
   GSLMultiRootDerivSolver, internal class for implementing GSL multi-root finders
   using derivatives

   @ingroup MultiRoot
*/
class GSLMultiRootDerivSolver : public GSLMultiRootBaseSolver {

public:

   /**
      Constructor
   */
   GSLMultiRootDerivSolver (const gsl_multiroot_fdfsolver_type * type, int n ) :
      fDerivSolver(0),
      fVec(0),
      fName(std::string("undefined"))
   {
      CreateSolver(type, n);
   }

   /**
      Destructor (no operations)
   */
   ~GSLMultiRootDerivSolver () override  {
      if (fDerivSolver) gsl_multiroot_fdfsolver_free(fDerivSolver);
      if (fVec != 0) gsl_vector_free(fVec);
   }

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   GSLMultiRootDerivSolver(const GSLMultiRootDerivSolver &) : GSLMultiRootBaseSolver() {}

   /**
      Assignment operator
   */
   GSLMultiRootDerivSolver & operator = (const GSLMultiRootDerivSolver & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }


public:


   /// create the solver from the type and size of number of fitting points and number of parameters
   void  CreateSolver(const gsl_multiroot_fdfsolver_type * type, unsigned int n) {

      /// create the solver from the type and size of number of fitting points and number of parameters
      if (fDerivSolver) gsl_multiroot_fdfsolver_free(fDerivSolver);
      fDerivSolver = gsl_multiroot_fdfsolver_alloc(type, n);
      fName = std::string(gsl_multiroot_fdfsolver_name(fDerivSolver) );
   }



   /// set the solver parameters for the case of derivative
   int SetSolver(const std::vector<ROOT::Math::IMultiGenFunction*> & funcVec, const double * x) override {
      // create a vector of the fit contributions
      // need to create a vecctor of gradient functions, convert and store in the class
      // the new vector pointer
      assert(fDerivSolver !=0);
      unsigned int n = funcVec.size();
      fGradFuncVec.reserve( n );
      for (unsigned int i = 0; i < n; ++i) {
         ROOT::Math::IMultiGradFunction * func = dynamic_cast<ROOT::Math::IMultiGradFunction *>(funcVec[i] );
         if (func == 0) {
            MATH_ERROR_MSG("GSLMultiRootSolver::SetSolver","Function does not provide gradient interface");
            return -1;
         }
         fGradFuncVec.push_back( func);
      }

      fDerivFunctions.SetFunctions(fGradFuncVec, funcVec.size() );
      // set initial values
      if (fVec != 0) gsl_vector_free(fVec);
      fVec = gsl_vector_alloc( n);
      std::copy(x,x+n, fVec->data);

      return gsl_multiroot_fdfsolver_set(fDerivSolver, fDerivFunctions.GetFunctions(), fVec);
   }

   const std::string & Name() const override {
      return fName; 
   }

   int Iterate() override {
      if (fDerivSolver == 0) return -1;
      return gsl_multiroot_fdfsolver_iterate(fDerivSolver);
   }

   /// solution values at the current iteration
   gsl_vector * GetRoot() const override {
      if (fDerivSolver == 0) return 0;
      return gsl_multiroot_fdfsolver_root(fDerivSolver);
   }

   /// return function values
   gsl_vector * GetF() const override {
      if (fDerivSolver == 0) return 0;
      return  gsl_multiroot_fdfsolver_f(fDerivSolver);
   }

   /// return function steps
   gsl_vector * GetDx() const override {
      if (fDerivSolver == 0) return 0;
      return  gsl_multiroot_fdfsolver_dx(fDerivSolver);
   }



private:

   GSLMultiRootDerivFunctionWrapper fDerivFunctions;
   gsl_multiroot_fdfsolver * fDerivSolver;
   // cached vector to avoid re-allocating every time a new one
   mutable gsl_vector * fVec;
   std::vector<ROOT::Math::IMultiGradFunction*> fGradFuncVec;
   std::string fName;   // solver nane

};

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLMultiRootSolver */
