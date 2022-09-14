// @(#)root/unuran:$Id$
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranContDist


#ifndef ROOT_Math_TUnuranContDist
#define ROOT_Math_TUnuranContDist

#include "TUnuranBaseDist.h"

#include "Math/IFunctionfwd.h"

class TF1;




/**
   \class TUnuranContDist
   \ingroup Unuran

   TUnuranContDist class describing one dimensional continuous distribution.
   It is used by TUnuran to generate random numbers according to this distribution via
   TUnuran::Sample()

   The class can be constructed from a function (TF1) representing the probability density
   function of the distribution. Optionally the derivative of the pdf can also be passed.

   It provides a method to set the domain of the distribution ( SetDomain ) which will correspond to the range
   of the generated random numbers. By default the domain is (-inf, + inf), independently of the
   range set in the TF1 class used to construct the distribution.

   In addition, some UNURAN methods requires extra information (cdf function, distribution mode,
   area of pdf, etc...). This information can as well be set.
   Some methods require instead of the pdf the log of the pdf.
   This can also be controlled by setting a flag when constructing this class.
*/


class TUnuranContDist : public TUnuranBaseDist {

public:


   /**
      Constructor from a TF1 objects specifying the pdf and optionally from another function
      representing the derivative of the pdf. The flag isLogPdf can be used to pass instead of the pdf
      (and its derivative) the log (and the derivative of the log) of the pdf.
      By default the distribution has not domain set (it is defined between [-inf,+inf], no mode, no pdf area and no
      cdf explicitly defined. UnuRan, if needed, can compute some of this quantities, but the user if they know them can
      set them in order to speed up the algorithm. For example in case of the Cdf, if the user has not set it, a numerical
      integration algorithm is used to estimate the Cdf from the Pdf.
   */
   explicit TUnuranContDist (TF1 * pdf = nullptr, TF1 * deriv = nullptr, bool isLogPdf = false );
   /**
      Constructor as above but with the possibility to pass also the Cdf.
       In case an algorithm requiring only the Cdf (no Pdf), one can use this constructor passing nullptr for Pdf and derivative of
       the Pdf
   */
   TUnuranContDist (TF1 * pdf, TF1 * deriv, TF1 * cdf, bool isLogPdf = false );
   /**
      Constructor as before but from a generic function object interface for one-dim functions
   */
   explicit TUnuranContDist (const ROOT::Math::IGenFunction & pdf, const ROOT::Math::IGenFunction * dpdf = nullptr, bool isLogPdf = false, bool copyFunc = false);
   /**
      Constructor as before from pointers to generic function object interface for one-dim functions
      which can be use for all algorithms including those requiring only the Cdf
    */
   TUnuranContDist (const ROOT::Math::IGenFunction * pdf, const ROOT::Math::IGenFunction * dpdf,
   const ROOT::Math::IGenFunction * cdf, bool isLogPdf = false, bool copyFunc = false );

   /**
      Destructor
   */
   ~TUnuranContDist () override;


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
   TUnuranContDist * Clone() const override { return new TUnuranContDist(*this); }


   /**
      set cdf distribution. If a method requires it
      and is not set it is then estimated using numerical
      integration from the pdf
   */
   void SetCdf(TF1 *  cdf);

   /**
      set cdf distribution using a generic function interface
   */
   void SetCdf(const ROOT::Math::IGenFunction & cdf);

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
      check if a cdf function is provided for the distribution
    */
   bool HasCdf() const { return fCdf != nullptr; }

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


   const ROOT::Math::IGenFunction * fPdf;       ///< pointer to the pdf
   const ROOT::Math::IGenFunction * fDPdf;      ///< pointer to the derivative of the pdf
   const ROOT::Math::IGenFunction * fCdf;       ///< pointer to the cdf (cumulative dist.)

   double fXmin;            ///< lower value of the domain
   double fXmax;            ///< upper value of the domain
   double fMode;            ///< mode of the distribution
   double fArea;            ///< area below pdf

   // flags
   bool  fIsLogPdf;         ///< flag to control if function pointer represent log of pdf
   bool  fHasDomain;        ///< flag to control if distribution has a defined domain (otherwise is [-inf,+inf]
   bool  fHasMode;          ///< flag to control if distribution has a pre-computed mode
   bool  fHasArea;          ///< flag to control if distribution has a pre-computed area below the pdf
   bool  fOwnFunc;          ///< flag to indicate if class manages the function pointers
   //mutable double fX[1];  ///<! cached vector for using TF1::EvalPar

   ClassDefOverride(TUnuranContDist,1)  //Wrapper class for one dimensional continuous distribution


};



#endif /* ROOT_Math_TUnuranContDist */
