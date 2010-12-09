// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Oct 18 11:48:00 2006

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


// Header file for class GSLMinimizer

#ifndef ROOT_Math_GSLMinimizer
#define ROOT_Math_GSLMinimizer

#ifndef ROOT_Math_Minimizer
#include "Math/Minimizer.h"
#endif


#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif

#ifndef ROOT_Math_IParamFunctionfwd
#include "Math/IParamFunctionfwd.h"
#endif

#ifndef ROOT_Math_MinimizerVariable
#include "Math/MinimizerVariable.h"
#endif


#include <vector>
#include <map>
#include <string> 



namespace ROOT { 

namespace Math { 


   /**
      enumeration specifying the types of GSL minimizers
      @ingroup MultiMin
   */
   enum EGSLMinimizerType { 
      kConjugateFR, 
      kConjugatePR, 
      kVectorBFGS, 
      kVectorBFGS2, 
      kSteepestDescent
   };


   class GSLMultiMinimizer; 

   class MinimTransformFunction;


//_____________________________________________________________________________________
/** 
   GSLMinimizer class. 
   Implementation of the ROOT::Math::Minimizer interface using the GSL multi-dimensional 
   minimization algorithms.

   See <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Multidimensional-Minimization.html">GSL doc</A> 
   from more info on the GSL minimization algorithms. 

   The class implements the ROOT::Math::Minimizer interface and can be instantiated using the 
   ROOT plugin manager (plugin name is "GSLMultiMin"). The varius minimization algorithms 
   (conjugatefr, conjugatepr, bfgs, etc..) can be passed as enumerations and also as a string. 
   The default algorithm is conjugatefr (Fletcher-Reeves conjugate gradient algorithm).  

   @ingroup MultiMin
*/ 
class GSLMinimizer : public ROOT::Math::Minimizer {

public: 

   /** 
      Default constructor
   */ 
   GSLMinimizer (ROOT::Math::EGSLMinimizerType type = ROOT::Math::kConjugateFR  ); 

   /**
      Constructor with a string giving name of algorithm 
    */
   GSLMinimizer (const char *  type  ); 

   /** 
      Destructor 
   */ 
   virtual ~GSLMinimizer (); 

private:
   // usually copying is non trivial, so we make this unaccessible

   /** 
      Copy constructor
   */ 
   GSLMinimizer(const GSLMinimizer &) : Minimizer() {}

   /** 
      Assignment operator
   */ 
   GSLMinimizer & operator = (const GSLMinimizer & rhs) { 
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }

public: 

   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGenFunction & func); 

   /// set gradient the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGradFunction & func); 

   /// set free variable 
   virtual bool SetVariable(unsigned int ivar, const std::string & name, double val, double step); 


   /// set lower limit variable  (override if minimizer supports them )
   virtual bool SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower );
   /// set upper limit variable (override if minimizer supports them )
   virtual bool SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper ); 
   /// set upper/lower limited variable (override if minimizer supports them )
   virtual bool SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double /* lower */, double /* upper */); 
   /// set fixed variable (override if minimizer supports them )
   virtual bool SetFixedVariable(unsigned int /* ivar */, const std::string & /* name */, double /* val */);  
   /// set the value of an existing variable 
   virtual bool SetVariableValue(unsigned int ivar, double val );
   /// set the values of all existing variables (array must be dimensioned to the size of existing parameters)
   virtual bool SetVariableValues(const double * x);

   /// method to perform the minimization
   virtual  bool Minimize(); 

   /// return minimum function value
   virtual double MinValue() const { return fMinVal; } 

   /// return expected distance reached from the minimum
   virtual double Edm() const { return 0; } // not impl. }

   /// return  pointer to X values at the minimum 
   virtual const double *  X() const { return &fValues.front(); } 

   /// return pointer to gradient values at the minimum 
   virtual const double *  MinGradient() const; 

   /// number of function calls to reach the minimum 
   virtual unsigned int NCalls() const;  

   /// this is <= Function().NDim() which is the total 
   /// number of variables (free+ constrained ones) 
   virtual unsigned int NDim() const { return fValues.size(); }   

   /// number of free variables (real dimension of the problem) 
   /// this is <= Function().NDim() which is the total number of free parameters
   virtual unsigned int NFree() const { return fObjFunc->NDim(); }  

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


   // method of only GSL  minimizer (not inherited from interface)
 
   /// return pointer to used objective gradient function 
   const ROOT::Math::IMultiGradFunction * ObjFunction() const { return fObjFunc; }

   /// return transformation function (NULL if not having a transformation) 
   const ROOT::Math::MinimTransformFunction * TransformFunction() const; 


protected: 

private: 
   
   // dimension of the function to be minimized 
   unsigned int fDim; 

   ROOT::Math::GSLMultiMinimizer * fGSLMultiMin; 
   const ROOT::Math::IMultiGradFunction * fObjFunc; 
   
   double fMinVal; 
   double fLSTolerance;  // Line Search Tolerance
   std::vector<double> fValues;
   //mutable std::vector<double> fErrors;
   std::vector<double> fSteps;
   std::vector<std::string> fNames;
   std::vector<ROOT::Math::EMinimVariableType> fVarTypes;  // vector specifyng the type of variables
   std::map< unsigned int, std::pair<double, double> > fBounds; // map specifying the bound using as key the parameter index

}; 

   } // end namespace Fit

} // end namespace ROOT



#endif /* ROOT_Math_GSLMinimizer */
