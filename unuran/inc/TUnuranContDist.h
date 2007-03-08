// @(#)root/unuran:$Name:  $:$Id: TUnuranContDist.h,v 1.3 2007/02/05 10:24:44 moneta Exp $
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranContDist

//////////////////////////////////////////////////////////////////////
// 
//   TUnuranContDistr class 
//   wrapper class for one dimensional continous distribution
// 
///////////////////////////////////////////////////////////////////////

#ifndef ROOT_Math_TUnuranContDist
#define ROOT_Math_TUnuranContDist

#ifndef ROOT_Math_TUnuranBaseDist
#include "TUnuranBaseDist.h"
#endif


class TF1;

//////////////////////////////////////////////////////////////////////
/** 
   TUnuranContDistr class 
   wrapper class for one dimensional continous distribution
*/ 
///////////////////////////////////////////////////////////////////////
class TUnuranContDist : public TUnuranBaseDist {

public: 


   /** 
      Constructor from a TF1 objects specifying the pdf and optionally from another function 
      representing the derivative of the pdf. The flag isLogPdf can be used to pass instead of the pdf 
      (and its derivative) the log (and the derivative of the log) of the pdf. 
      By default the distribution has not domain set (it is defined between [-inf,+inf], no mode, no pdf area and no 
      cdf explicity defined. UnuRan, if needed, can compute some of this quantities, but the user if he knows them can 
      set them in order to speed up the algorithm. For example in case of the Cdf, if the user has not set it, a numerical 
      integration algorithm is used to estimate the Cdf from the Pdf. 
      In case an algorithm requires only the Cdf (no Pdf), an empty distribution can be constructed and then the user must 
      set afterwards the Cdf. 
   */ 
   TUnuranContDist (const TF1 * pdf = 0, const TF1 * deriv = 0, bool isLogPdf = false );

   /** 
      Destructor (no operations)
   */ 
   virtual ~TUnuranContDist () {}


   /** 
      Copy constructor
   */ 
   TUnuranContDist(const TUnuranContDist &); 

   /** 
      Assignment operator
   */ 
   TUnuranContDist & operator = (const TUnuranContDist & rhs); 

   /**
      Clone (required by base class)
    */
   TUnuranContDist * Clone() const { return new TUnuranContDist(*this); } 


   /**
      set cdf distribution. If a method requires it 
      and is not set it is then estimated using numerical 
      integration from the pdf
   */
   void SetCdf(TF1 *  cdf) { fCdf = cdf; }


   /**
      Set the distribution domain. If min < max a domain is defined otherwise is undefined
    */
   void SetDomain(double xmin, double xmax)  { 
      fXmin = xmin; 
      fXmax = xmax; 
      if (fXmin < fXmax) 
         fHasDomain = true;
      else 
         fHasDomain = false;
   }

   /**
      set the distribution mode (x position of its maximum)
   */
   void SetMode(double mode) { fMode = mode; fHasMode=true;}

   /**
      set the area below the pdf
    */
   void SetPdfArea(double area) { fArea = area; fHasArea=true;}

   /**
      check if distribution has a defined domain and return in case its domain
   */
   bool GetDomain(double & xmin, double & xmax) const { 
      xmin = fXmin; 
      xmax = fXmax; 
      return fHasDomain; 
   }

   /**
      check if a cdf fucntion is provided for the distribution 
    */
   bool HasCdf() const { return fCdf != 0; } 

   /**
      check if distribution has a pre-computed mode
    */
   bool HasMode() const { return fHasMode; } 

   
   /**
      check if distribution has a pre-computed area below the Pdf
    */
   bool HasPdfArea() const { return fHasArea; }   

   /**
      return the mode   (x location of  maximum of the pdf)  
   */
   double Mode() const { return fMode; }

   /**
      return area below the pdf
   */
   double PdfArea() const { return fArea; }


   /**
      flag to control if given function represent the log of a pdf
   */
   bool IsLogPdf() const {  return fIsLogPdf; }

   /**
      evaluate the Probability Density function. Used by the UnuRan algorithms 
   */
   double Pdf ( double x) const; 

   /**
      evaluate the derivative of the pdf. Used by  UnuRan 
   */
   double DPdf( double x) const; 

   /**
      evaluate the integral (cdf)  on the domain. Used by Unuran algorithm
   */
   double Cdf(double x) const;   


protected: 


private: 

   const TF1 * fPdf;        //pointer to the pdf
   const TF1 * fDPdf;       //pointer to the derivative of the pdf
   const TF1 * fCdf;        //pointer to the cdf
   double fXmin;            //lower value of the domain 
   double fXmax;            //upper value of the domain
   double fMode;            //mode of the distribution
   double fArea;            //area below pdf
   // flags
   bool  fIsLogPdf;         //flag to control if function pointer represent log of pdf
   bool  fHasDomain;        //flag to control if distribution has a defined domain (otherwise is [-inf,+inf]
   bool  fHasMode;          //flag to control if distribution has a pre-computed mode
   bool  fHasArea;          //flag to control if distribution has a pre-computed area below the pdf

   ClassDef(TUnuranContDist,1)  //Wrapper class for one dimensional continous distribution


}; 



#endif /* ROOT_Math_TUnuranContDist */
