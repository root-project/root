// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 17:16:32 2006

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

// Header file for class GSLNLSMinimizer

#ifndef ROOT_Math_GSLNLSMinimizer
#define ROOT_Math_GSLNLSMinimizer



#include "Math/BasicMinimizer.h"

#include "Math/IFunctionfwd.h"

#include "Math/IParamFunctionfwd.h"

#include "Math/FitMethodFunction.h"

#include "Math/MinimTransformVariable.h"

#include <vector>

namespace ROOT {

   namespace Math {

      class GSLMultiFit;

//_____________________________________________________________________________________________________
/**
   GSLNLSMinimizer class for Non Linear Least Square fitting
   It Uses the Levemberg-Marquardt algorithm from
   <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Nonlinear-Least_002dSquares-Fitting.html">
   GSL Non Linear Least Square fitting</A>.

   @ingroup MultiMin
*/
class GSLNLSMinimizer : public  ROOT::Math::BasicMinimizer {

public:

   /**
      Default constructor
   */
   GSLNLSMinimizer (int type = 0);

   /**
      Destructor (no operations)
   */
   ~GSLNLSMinimizer () override;

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   GSLNLSMinimizer(const GSLNLSMinimizer &) : ROOT::Math::BasicMinimizer() {}

   /**
      Assignment operator
   */
   GSLNLSMinimizer & operator = (const GSLNLSMinimizer & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }

public:

   /// set the function to minimize
   void SetFunction(const ROOT::Math::IMultiGenFunction & func) override;

   /// set gradient the function to minimize
   void SetFunction(const ROOT::Math::IMultiGradFunction & func) override;


   /// method to perform the minimization
    bool Minimize() override;


   /// return expected distance reached from the minimum
   double Edm() const override { return fEdm; } // not impl. }


   /// return pointer to gradient values at the minimum
   const double *  MinGradient() const override;

   /// number of function calls to reach the minimum
   unsigned int NCalls() const override { return fNCalls; }

   /// number of free variables (real dimension of the problem)
   /// this is <= Function().NDim() which is the total
//   virtual unsigned int NFree() const { return fNFree; }

   /// minimizer provides error and error matrix
   bool ProvidesError() const override { return true; }

   /// return errors at the minimum
   const double * Errors() const override { return (fErrors.size() > 0) ? &fErrors.front() : nullptr; }
//  {
//       static std::vector<double> err;
//       err.resize(fDim);
//       return &err.front();
//    }

   /** return covariance matrices elements
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */
   double CovMatrix(unsigned int , unsigned int ) const override;

   /// return covariance matrix status
   int CovMatrixStatus() const override;

protected:

   /// Internal method to perform minimization
   /// template on the type of method function
   template<class Func>
   bool DoMinimize(const Func & f);


private:

   bool fUseGradFunction = false; // flag to indicate if using external gradients
   unsigned int fNFree;      // dimension of the internal function to be minimized
   unsigned int fNCalls;        // number of function calls

   ROOT::Math::GSLMultiFit * fGSLMultiFit;        // pointer to GSL multi fit solver

   double fEdm;                                   // edm value
   double fLSTolerance;                           // Line Search Tolerance
   std::vector<double> fErrors;
   std::vector<double> fCovMatrix;              //  cov matrix (stored as cov[ i * dim + j]



};

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLNLSMinimizer */
