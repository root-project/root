// @(#)root/unuran:$Name:  $:$Id: TUnuranDiscrDist.h,v 1.3 2007/02/05 10:24:44 moneta Exp $
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranDiscrDist


#ifndef ROOT_Math_TUnuranDiscrDist
#define ROOT_Math_TUnuranDiscrDist

#ifndef ROOT_Math_TUnuranBaseDist
#include "TUnuranBaseDist.h"
#endif

#include <vector>


class TF1;

//____________________________________________________________________
/** 
   TUnuranDiscrDistr class 
   wrapper class for one dimensional discrete distribution
*/ 
class TUnuranDiscrDist : public TUnuranBaseDist {

public: 


   /** 
      Constructor from a TF1 objects specifying the pdf
   */ 
   TUnuranDiscrDist (const TF1 * func = 0);

   /** 
      Constructor from a vector of probability
   */ 
   template<class Iterator> 
   TUnuranDiscrDist (Iterator * begin, Iterator * end) : 
      fPVec(begin,end),
      fPmf(0), 
      fCdf(0), 
      fXmin(1), 
      fXmax(-1), 
      fMode(0), 
      fSum(0),
      fHasDomain(0),
      fHasMode(0),
      fHasSum(0)   {}

   /** 
      Destructor (no operations)
   */ 
   ~TUnuranDiscrDist () {}

   /** 
      Copy constructor
   */ 
   TUnuranDiscrDist(const TUnuranDiscrDist &); 

   /** 
      Assignment operator
   */ 
   TUnuranDiscrDist & operator = (const TUnuranDiscrDist & rhs); 

   /**
      Clone (required by base class)
    */
   TUnuranDiscrDist * Clone() const { return new TUnuranDiscrDist(*this); } 


   /**
      set cdf distribution. If a method requires it 
      and is not set it is estimated numerically
   */
   void SetCdf(TF1 *  cdf) { fCdf = cdf; }

   /**
      Set the distribution domain, by default the domain is [0,INT_MAX]
      If xmin >= xmax a domain is removed
    */
   void SetDomain(int xmin, int xmax)  { 
      fXmin = xmin; 
      fXmax = xmax; 
      if (fXmin < fXmax) 
         fHasDomain = true;
      else 
         fHasDomain = false;
   }


   /**
      set the mode of the distribution (location of maximum probability)
    */
   void SetMode(int mode) { fMode = mode; fHasMode=true;}

   /**
      set the value of the sum of the probabilities in the given domain
    */ 
   void SetProbSum(int sum) { fSum = sum; fHasSum=true; }

   /**
      check if distribution has domain and return in case its domain
   */
   bool GetDomain(int & xmin, int & xmax) const { 
      xmin = fXmin; 
      xmax = fXmax; 
      return fHasDomain; 
   }
   
   /**
      get the mode   (x location of function maximum)  
   */
   int Mode() const { return fMode; }

   /**
      return area of the pdf
   */
   double ProbSum() const { return fSum; }


   /**
      flag to control if distribution provides the mode
    */
   bool HasMode() const { return fHasMode; } 

   
   /**
      flag to control if distribution provides the total area of the probability function
    */
   bool HasProbSum() const { return fHasSum; } 

   /**
      flag to control if distribution provides also a Cdf
    */
   bool HasCdf() const { return fCdf != 0; } 


   /**
      retrieve a reference to the vector of the probabilities : Prob(i)
      If the distribution is defined from a function (i.e. for distribution with undefined domain)
      the vector is empty. 
    */
   const std::vector<double> & ProbVec() const { return fPVec; }

   /**
      evaluate the distribution (probability mesh function) at the integer value x. 
      Used internally by UnuRan
      For integer values outside the domain the function must return 0.0
   */ 
   double Pmf ( int x) const; 

   /** 
       evaluate the integral (cdf)  on the given domain
   */
   double Cdf(int x) const;   


protected: 


private: 

   std::vector<double> fPVec;    //Vector of the probabilities 
   const TF1 * fPmf;             //pointer to a function calculating the probability 
   const TF1 * fCdf;             //pointer to the cumulative distribution function
   int   fXmin;                  //lower value of the domain
   int   fXmax;                  //upper value of the domain
   int   fMode;                  //mode of the distribution
   double fSum;                  //total sum of the probabilities in the given domain
   // flags
   bool  fHasDomain;             //flag to control if distribution has a defined domain (otherwise is [0,INT_MAX])
   bool  fHasMode;               //flag to control if distribution has a pre-computed mode
   bool  fHasSum;                //flag to control if distribution has a pre-computed sum of the probabilities

   //ClassDef(TUnuranDiscrDist,1)  //Wrapper class for one dimensional continous distribution


}; 



#endif /* ROOT_Math_TUnuranDiscrDist */
