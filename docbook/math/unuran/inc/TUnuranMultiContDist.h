// @(#)root/unuran:$Id$
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranMultiContDist

#ifndef ROOT_Math_TUnuranMultiContDist
#define ROOT_Math_TUnuranMultiContDist

#ifndef ROOT_Math_TUnuranBaseDist
#include "TUnuranBaseDist.h"
#endif

#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif



#include <vector>

class TF1; 


//___________________________________________________________________________________________
/** 
   TUnuranMultiContDist class describing multi dimensional continuous distributions. 
   It is used by TUnuran to generate a set of random numbers according to this distribution via 
   TUnuran::Sample(double *).  
   The class can be constructed from a multi-dimensional function (TF1 pointer, which can be actually also a 
   TF2 or a TF3). 
   It provides a method to set the domain of the distribution ( SetDomain ) which will correspond to the range 
   of the generated random numbers. By default the domain is [(-inf,-inf,...)(+inf,+inf,...)], indipendently of the 
   range set in the TF1 class used to construct the distribution. 

   The derivatives of the pdf which are used by some UNURAN methods are estimated numerically in the 
   Derivative() method. Some extra information (like distribution mode) can be set using SetMode.   
   Some methods require instead of the pdf the log of the pdf. 
   This can also be controlled by setting a flag when constructing this class. 

*/ 
/////////////////////////////////////////////////////////////
class TUnuranMultiContDist : public TUnuranBaseDist {

public: 


   /** 
      Constructor from a TF1 object representing the Probability density function. 
      The derivatives of the Pdf are estimated, when required by the UNURAN algorithm, 
      using numerical derivation. 
      If a value of dim 0 is passed , the dimension of the function is taken from TF1::GetNdim(). 
      This works only for 2D and 3D (for TF2 and TF3 objects). 
   */ 
   TUnuranMultiContDist (TF1 * func = 0, unsigned int dim = 0, bool isLogPdf = false);  


   /** 
      Constructor as before but from a generic function object interface for multi-dim functions
   */ 
   TUnuranMultiContDist (const ROOT::Math::IMultiGenFunction & pdf, bool isLogPdf = false); 

   /** 
      Destructor 
   */ 
   virtual ~TUnuranMultiContDist ();


   /** 
      Copy constructor
   */ 
   TUnuranMultiContDist(const TUnuranMultiContDist &); 

   /** 
      Assignment operator
   */ 
   TUnuranMultiContDist & operator = (const TUnuranMultiContDist & rhs); 

   /**
      Clone (required by base class)
    */
   virtual TUnuranMultiContDist * Clone() const { return new TUnuranMultiContDist(*this); } 


   /**
      get number of dimension of the distribution
   */
   unsigned int NDim() const { 
      return fPdf->NDim(); 
   }

   /**
      set the domain of the distribution giving an array of minimum and maximum values
      By default otherwise the domain is undefined, i.e. is [-inf,+inf]
      To remove the domain do a SetDomain(0,0). 
      There is no possibility to have a domain defined in only one coordinate. Use instead inf or DOUBLE_MAX to 
      specify un infinite domain in that coordinate
   */
   void SetDomain(const double *xmin, const double *xmax)  { 
      if (xmin == 0 || xmax == 0) return; 
      fXmin = std::vector<double>(xmin,xmin+NDim()); 
      fXmax = std::vector<double>(xmax,xmax+NDim()); 
   }

   /**
      set the mode of the distribution (coordinates of the distribution maximum values)
   */
   void SetMode(const double * x) { 
      fMode = std::vector<double>(x,x+NDim());
   }

   /**
      get the distribution lower domain values. Return a null pointer if domain is not defined 
   */
   const double * GetLowerDomain() const { 
      if (fXmin.size() == 0 || (  fXmin.size() != fXmax.size() )  ) return 0; 
      return &fXmin[0];
   }
   /**
      get the distribution upper domain values. Return a null pointer if domain is not defined 
   */
   const double * GetUpperDomain() const { 
      if (fXmax.size() == 0 || (  fXmin.size() != fXmax.size() )  ) return 0; 
      return &fXmax[0];
   }


   /**
      get the mode (vector of coordinate positions of the maxima of the distribution)
      If a mode has not defined return a NULL pointer
   */
   const double * GetMode( ) const { 
      if (fMode.size() == 0  ) return 0; 
      return &fMode.front(); 
   }


   /**
      flag to control if given function represent the log of a pdf
   */
   bool IsLogPdf() const {  return fIsLogPdf; }

   /**
      evaluate the probability density function, used by UnuRan 
   */
   double Pdf ( const double * x) const;

   /** 
       evaluate the gradient vector of  the Pdf. Used by UnuRan
   */
   void Gradient( const double * x, double * grad) const;  

   /**
      evaluate the partial derivative for the given coordinate. Used by UnuRan
   */
   double Derivative( const double * x, int icoord) const; 



private: 

   const ROOT::Math::IMultiGenFunction  * fPdf;    //pointer to the pdf

   std::vector<double> fXmin;      //vector with lower x values of the domain
   std::vector<double> fXmax;      //vector with upper x values of the domain 
   std::vector<double> fMode;      //vector representing the x coordinates of the maximum of the pdf 

   bool fIsLogPdf;                 //flag to control if function pointer represent log of pdf
   bool  fOwnFunc;                // flag to indicate if class manages the function pointers


   ClassDef(TUnuranMultiContDist,1)  //Wrapper class for multi dimensional continuous distribution


}; 



#endif /* ROOT_Math_TUnuranMultiContDist */
