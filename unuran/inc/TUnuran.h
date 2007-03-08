// @(#)root/unuran:$Name:  $:$Id: TUnuran.h,v 1.1 2006/11/15 17:40:36 brun Exp $
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

#ifndef ROOT_Math_TUnuranBaseDist
#include "TUnuranBaseDist.h"
#endif


class TUnuranContDist;
class TUnuranDiscrDist;
class TUnuranMultiContDist;
class TUnuranEmpDist;

#include <memory>

////////////////////////////////////////////////////////////////////////
/** 
   TUnuran class. 
   Interface to the UnuRan package for generating non uniform random 
   numbers 
*/ 
///////////////////////////////////////////////////////////////////////


//class TUnuranGenerator; 
struct unur_gen; 
typedef struct unur_gen UNUR_GEN; 

// struct unur_urng_generic; 
// typedef struct unur_urng_generic UNUR_URNG; 

struct unur_distr; 
typedef struct unur_distr UNUR_DISTR; 

class TRandom; 
class TH1; 

class TUnuran {

public: 

   /** 
      Constructor with a generator instance and given level of log output
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


   /** 
      initialize with Unuran string interface
   */ 
   bool Init(const std::string & distr, const std::string  & method); 


   /** 
      Initialize method for continuous one-dimensional distribution. 
      User must provide a distribution object (which is copied inside) and a string for a method. 
      For the list of available method for 1D cont. distribution see the 
      <A href="http://statmath.wu-wien.ac.at/unuran/doc/unuran.html#Methods_005ffor_005fCONT">UnuRan doc</A>. 
      A re-initialization is needed whenever distribution parameters have been changed.      
   */ 
   bool Init(const TUnuranContDist & distr, const std::string & method = "method=auto"); 

   /** 
      Initialize method for continuous multi-dimensional distribution. 
      User must provide a distribution object (which is copied inside) and a string for a method. 
      For the list of available method for multivariate cont. distribution see the 
      <A href="http://statmath.wu-wien.ac.at/unuran/doc/unuran.html#Methods_005ffor_005fCVEC">UnuRan doc</A>
      A re-initialization is needed whenever distribution parameters have been changed.      
      
   */ 
   bool Init(const TUnuranMultiContDist & distr, const std::string & method = "method=hitro");


   /** 
      Initialize method for continuous one-dimensional discrete distribution. 
      User must provide a distribution object (which is copied inside) and a string for a method. 
      For the list of available method for 1D discrete distribution see the 
      <A href="http://statmath.wu-wien.ac.at/unuran/doc/unuran.html#Methods_005ffor_005fDISCR">UnuRan doc</A>
      A re-initialization is needed whenever distribution parameters have been changed.      
      
   */ 
   bool Init(const TUnuranDiscrDist & distr, const std::string & method = "method=auto"); 


   /** 
      Initialize method for continuous empirical distribution. 
      User must provide a distribution object (which is copied inside) and a string for a method.
      The distribution object can represent binned (only 1D) or unbinned (1D or multi-dim) data 
      The method for the unbinned empirical distribution are based on the kernel smoothing, see  
      <A href="http://statmath.wu-wien.ac.at/software/unuran/doc/unuran.html#EMPK">UnuRan doc</A>
      A re-initialization is needed whenever distribution parameters have been changed.      
      
   */ 
   bool Init(const TUnuranEmpDist & distr, const std::string & method = "method=empk"); 


   bool InitPoisson(double mu, std::string method = "dstd");


   /**
      reinitialize UNURAN
    */
   //bool ReInit(); 

   /**
      sample 1D distribution
      User is responsible for having previously correctly initialized with TUnuran::Init
   */
   double Sample();

   /**
      sample multidimensional distributions
      User is responsible for having previously correctly initialized with TUnuran::Init
   */
   bool SampleMulti(double * x); 

   /**
      sample discrete distributions
      User is responsible for having previously correctly initialized with TUnuran::Init
   */
   int SampleDiscr(); 

   /**
      set the seed for the random number generator
    */
   void SetSeed(unsigned int seed); 

   /**
      set log level 
   */
   bool SetLogLevel(unsigned int iflag = 1);  

   /**
      set stream for log and error (not yet implemented)
   */
   bool SetLogStream() { return false;}

   /**
      used Unuran method
    */
   const std::string & MethodName() const { return fMethod; }

protected: 


   bool SetRandomGenerator(); 
   
   bool SetContDistribution(const TUnuranContDist & dist );

   bool SetMultiDistribution(const TUnuranMultiContDist & dist );

   bool SetDiscreteDistribution(const TUnuranDiscrDist & dist );

   bool SetEmpiricalDistribution(const TUnuranEmpDist & dist );

   /**
      change the method and initialize Unuran with the previously given distribution
    */
   bool SetMethodAndInit(); 



// private: 

   UNUR_GEN * fGen;                      //pointer to the UnuRan C generator struct
   UNUR_DISTR * fUdistr;                 //pointer to the UnuRan C distribution struct
// TUnuranContDist         fCDist;       // for continous 1D distribution
// TUnuranDiscrDist        fDDist;       // for discrete distribution
// TUnuranMultiContDist    fMultiCDist;  // for multidimensional continous distribution
   std::auto_ptr<TUnuranBaseDist>         fDist;       // pointer for distribution wrapper
//    std::auto_ptr<TUnuranDiscrDist>        fDDist;       // for discrete distribution
//    std::auto_ptr<TUnuranEmpDist>          fEDist;       // for empirical distribution
//    std::auto_ptr<TUnuranMultiContDist>          fMultiCDist;       // for multi-dim cont. distribution
   //TUnuranEmpirDist        fEDist;       // for empirical 1D distribution
   TRandom * fRng;                       //pointer to random number generator
   std::string fMethod;                  //string representing the method

}; 


#endif /* ROOT_Math_TUnuran */
