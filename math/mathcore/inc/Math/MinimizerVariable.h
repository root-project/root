// @(#)root/mathmore:$Id$
// Author: L. Moneta 2009

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
// Header file for class MinimizerVariable

#ifndef ROOT_Math_MinimizerVariable
#define ROOT_Math_MinimizerVariable

#ifndef ROOT_Math_MinimizerVariableTransformation
#include "MinimizerVariableTransformation.h"
#endif

#include <memory>

namespace ROOT { 

   namespace Math { 


      /**
         Enumeration describing the status of the variable
         The enumeration are used in the minimizer classes to categorize the variables
      */
      enum EMinimVariableType { 
         kDefault,    // free variable (unlimited)
         kFix,        // fixed variable 
         kBounds,     //  variable has two bounds
         kLowBound,   // variable has a lower bound
         kUpBound     // variable has an upper bounds
      };

/** 
   MinimizerVariable class to perform a transformations on the 
   variables to deal with fixed or limited variables


   @ingroup MultiMin
*/ 


class MinimizerVariable { 

public: 

   /** 
     Default Constructor for  an unlimited variable
   */ 
   MinimizerVariable () : 
      fFix(false), fLowBound(false), fUpBound(false), fBounds(false), 
      fTransform(0), fLower(1), fUpper(0)
   {}

   // constructor for fixed variable
   MinimizerVariable (double value) : 
      fFix(true), fLowBound(false), fUpBound(false), fBounds(false), 
      fTransform(0), fLower(value), fUpper(value) 
   {}

   // constructor for double bound variable
   MinimizerVariable (double lower, double upper, SinVariableTransformation * trafo) : 
      fFix(false), fLowBound(false), fUpBound(false), fBounds(true),      
      fTransform(trafo),
      fLower(lower), fUpper(upper)
   {   }

   // constructor for lower bound variable
   MinimizerVariable (double lower, SqrtLowVariableTransformation * trafo) : 
      fFix(false), fLowBound(true), fUpBound(false), fBounds(false), 
      fTransform(trafo), fLower(lower), fUpper(lower) 
   {}

   // constructor for upper bound variable
   MinimizerVariable (double upper, SqrtUpVariableTransformation * trafo) : 
      fFix(false), fLowBound(true), fUpBound(false), fBounds(false), 
      fTransform(trafo), fLower(upper), fUpper(upper) 
   {}

   // copy constructor 
   MinimizerVariable (const MinimizerVariable & rhs) : 
      fFix(rhs.fFix), fLowBound(rhs.fLowBound), fUpBound(rhs.fUpBound), fBounds(rhs.fBounds), 
      fLower(rhs.fLower), fUpper(rhs.fUpper)
   {
      // swap auto_ptr
      fTransform.reset( const_cast<MinimizerVariable &>( rhs).fTransform.release() ) ;      
   }

   // assignment 
   MinimizerVariable & operator= (const MinimizerVariable & rhs) {
      if (&rhs == this) return *this; 
      fFix = rhs.fFix;
      fLowBound = rhs.fLowBound;  
      fUpBound  = rhs.fUpBound; 
      fBounds   = rhs.fBounds;  
      fLower = rhs.fLower;  fUpper = rhs.fUpper;

      // swap auto_ptr
      fTransform.reset( const_cast<MinimizerVariable &>( rhs).fTransform.release() ) ;     
      return *this;
   }
   

   bool IsFixed() const { return fFix; }
   
   bool IsLimited() const { return fBounds || fLowBound || fUpBound; } 

   bool HasLowerBound() const { return fLowBound || fBounds; }

   bool HasUpperBound() const { return fUpBound || fBounds; }

   double LowerBound() const { return fLower; }

   double UpperBound() const { return fUpper; }

   double FixValue() const { return fLower; }

   // internal to external transformation 
   double InternalToExternal( double x) const { 
      return fTransform->Int2ext(x, fLower, fUpper); 
   }

   // derivative of the internal to external transformation ( d Int2Ext / d int )
   double DerivativeIntToExt ( double x) const { 
      return fTransform->DInt2Ext( x, fLower, fUpper); 
   }

   // etxernal to internal transformation
   double ExternalToInternal(double x) const { 
      return fTransform->Ext2int(x, fLower, fUpper);       
   }

private: 

   bool fFix;         // fix variable
   bool fLowBound;    // has lower bound 
   bool fUpBound;     // has uppper bound param 
   bool fBounds;      // has double bound
   std::auto_ptr< MinimizerVariableTransformation> fTransform; // pointer to the minimizer transformation   
   double fLower;   // lower parameter limit
   double fUpper;   // upper parameter limit
 
}; 

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_MinimizerVariable */


