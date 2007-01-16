// @(#)root/unuran:$Name:  $:$Id: TUnuranDistrMulti.h,v 1.1 2006/11/15 17:40:36 brun Exp $
// Author: L. Moneta Wed Sep 27 17:07:37 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranDistrMulti

#ifndef ROOT_Math_TUnuranDistrMulti
#define ROOT_Math_TUnuranDistrMulti


#include "TF1.h"

#include <vector>
#include <cassert>

/** 
   TUnuranDistrMulti class 
*/ 
class TUnuranDistrMulti {

public: 

   /** 
      Default constructor
   */ 
   TUnuranDistrMulti () : 
     fFunc(0), 
     fHasDomain(0)
   {}
 

   /** 
      Constructor from a TF1 objects
   */ 
   TUnuranDistrMulti (TF1 * func) : 
      fFunc(func), 
      fHasDomain(0)
   {
      assert(func != 0); 
      fDim = func->GetNdim();
   } 

   /** 
      Destructor (no operations)
   */ 
   ~TUnuranDistrMulti (); 


   // usually copying is non trivial, so we make this unaccessible

//    /** 
//       Copy constructor
//    */ 
//    TUnuranDistrMulti(const TUnuranDistrMulti &); 

//    /** 
//       Assignment operator
//    */ 
//    TUnuranDistrMulti & operator = (const TUnuranDistrMulti & rhs); 



   unsigned int NDim() const { 
      return fDim;
   }

   /// evaluate the destribution 
   inline double operator() ( const double * x) const {  
      return fFunc->EvalPar(x); 
   }

   /// evaluate the derivative of the function
   void Gradient( const double * x, double * grad) const;  

   /// evaluate the partial derivative for the given coordinate
   double Derivative( const double * x, int icoord) const; 


   bool GetDomain(const double * xmin, const double * xmax) const { 
      xmin = &fXmin.front(); 
      xmax = &fXmax.front(); 
      return fHasDomain; 
   }

   void SetDomain(double *xmin, double *xmax)  { 
      fXmin = std::vector<double>(xmin,xmin+fDim); 
      fXmax = std::vector<double>(xmax,xmax+fDim); 
      fHasDomain = true;
   }

//    /// get the mode   (x location of function maximum)  
//    double Mode() const { 
//       return fFunc->GetMaximumX(fXmin, fXmax); 
//    }



protected: 


private: 

   mutable TF1 * fFunc; 
   unsigned int fDim;
   std::vector<double> fXmin; 
   std::vector<double> fXmax; 
   bool fHasDomain;


}; 



#endif /* ROOT_Math_TUnuranDistrMulti */
