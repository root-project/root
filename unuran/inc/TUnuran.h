// @(#)root/unuran:$Name:  $:$Id: inc/Math/TUnuran.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
// Author: L. Moneta Tue Sep 26 16:25:09 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuran

#ifndef ROOT_TUnuran
#define ROOT_TUnuran

#include <string> 

#include "TUnuranDistr.h"
#include "TUnuranDistrMulti.h"


/** 
   TUnuran class 
*/ 

//class TUnuranGenerator; 
struct unur_gen; 
typedef struct unur_gen UNUR_GEN; 

// struct unur_urng_generic; 
// typedef struct unur_urng_generic UNUR_URNG; 

struct unur_distr; 
typedef struct unur_distr UNUR_DISTR; 

class TRandom; 
class TUnuranDistr; 


class TUnuran {

public: 

   /** 
      Constructor with a generator instance and level of log output
   */ 
   TUnuran (TRandom * r = 0, unsigned int log = 0); 


   /** 
      Destructor 
   */ 
   ~TUnuran (); 

private:
   // usually copying is non trivial, so we make this unaccessible

   /** 
      Copy constructor
   */ 
   TUnuran(const TUnuran &); 

   /** 
      Assignment operator
   */ 
   TUnuran & operator = (const TUnuran & rhs); 

public: 

   // reinitialize after changing distribuion parameters
   /** 
      initialize with Unuran string interface
   */ 
   bool Init(const std::string & distr, const std::string  & method); 


   /** 
      initialize with a distribution and method
   */ 
   bool Init(const TUnuranDistr & distr, const std::string & method = "method=auto"); 

   /** 
      initialize with a multi dimensional and a method
   */ 
   bool Init(const TUnuranDistrMulti & distr, const std::string & method = "method=hitro",bool useLogpdf = false);

   /**
      change the method
    */
   bool SetMethod(const std::string & method); 

   /**
      reinitialize UNURAN
    */
   bool Rinit(); 

   /// sample 1D distribution
   double Sample();

   /// sample multidimensional distributions
   bool SampleMulti(double * x); 

   /// set log level 
   bool SetLogLevel(unsigned int iflag = 1);  

   /// set stream for log and error 
   bool SetLogStream() { return false;}

protected: 

   bool SetRandomGenerator(); 
   
   bool SetDistribution( );

   bool SetDistributionMulti( );


// private: 

   UNUR_GEN * fGen; 
   UNUR_DISTR * fUdistr;
   TUnuranDistr  fDistr; 
   TUnuranDistrMulti  fDistrMulti; 
   bool fUseLogpdf; 
   TRandom * fRng;
   std::string fMethod;

}; 


#endif /* ROOT_Math_TUnuran */
