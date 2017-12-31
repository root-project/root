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
   Implementation for Ipopt (Interior Point OPTimizer) is a software package for large-scale ​nonlinear optimization. It is designed to find (local) solutions of mathematical optimization problems.

   See <A HREF="https://projects.coin-or.org/Ipopt">Ipopt doc</A>
   from more info on the Ipopt minimization algorithms.

   @ingroup MultiMin
*/
class IpoptMinimizer : public ROOT::Math::BasicMinimizer {
protected:
class InternalTNLP:public Ipopt::TNLP
{
    
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
   virtual ~IpoptMinimizer ();

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   IpoptMinimizer(const IpoptMinimizer &) : BasicMinimizer() {}

   /**
      Assignment operator
   */
   IpoptMinimizer & operator = (const IpoptMinimizer & rhs) {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }

public:

   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGenFunction & func);

   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGradFunction & func) { BasicMinimizer::SetFunction(func);}

   /// method to perform the minimization
   virtual  bool Minimize();


   /// return expected distance reached from the minimum
   virtual double Edm() const { return 0; } // not impl. }


   /// return pointer to gradient values at the minimum
   virtual const double *  MinGradient() const;

   /// number of function calls to reach the minimum
   virtual unsigned int NCalls() const;


   /// minimizer provides error and error matrix
   virtual bool ProvidesError() const { return false; }

   /// return errors at the minimum
   virtual const double * Errors() const {
      return 0;
   }

   /** return covariance matrices elements
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */
   virtual double CovMatrix(unsigned int , unsigned int ) const { return 0; }

};

   } // end namespace Fit

} // end namespace ROOT



#endif /* ROOT_Math_IpoptMinimizer */
