// @(#)root/unuran:$Name:  $:$Id: src/TUnuran.cxx,v 1.0 2006/01/01 12:00:00 moneta Exp $
// Author: L. Moneta Tue Sep 26 16:25:09 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuran

#include "TUnuran.h"
#include "UnuranRng.h"
#include "UnuranDistrAdapter.h"

#include "TRandom.h"
#include "TSystem.h"

#include <cassert>

#include <iostream>

#include <unuran.h>

#include "TUnuranDistr.h"
#include "TUnuranDistrMulti.h"


// TUnuran::TUnuran() 
// {
//    // Default constructor implementation.
//    fGen = 0; 
// }



TUnuran::TUnuran(TRandom * r, unsigned int debugLevel) : 
   fGen(0),
   fUdistr(0),
   fRng(r) 
{
   // constructor implementation with a ROOT random generator
   // if no generator is given the ROOT default is used
   if (fRng == 0) fRng = gRandom; 
   // set debug level at global level 
   // (should be in a static  initialization function of the library ? )
   if ( debugLevel > 2) 
      unur_set_default_debug(UNUR_DEBUG_ALL);
   else if (debugLevel > 1) 
      unur_set_default_debug(UNUR_DEBUG_ALL);
   else
      unur_set_default_debug(UNUR_DEBUG_OFF);
      
}


TUnuran::~TUnuran() 
{
   // Destructor implementation
   if (fGen != 0) unur_free(fGen); 
  // we can delete now the distribution object (it is copied in Unuran)
   if (fUdistr != 0) unur_distr_free(fUdistr);
}

//private (no impl.)
TUnuran::TUnuran(const TUnuran &) 
{
   // Implementation of copy constructor.
}

TUnuran & TUnuran::operator = (const TUnuran &rhs) 
{
   // Implementation of assignment operator.
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}


    
bool TUnuran::Init(const TUnuranDistr & distr, const std::string  & method)  
{ 
   //   initialization with a distribution and and generator
   // copy the distribution by value (maybe can done by pointer)
   fDistr = distr;
   fMethod = method; 
   if (! SetDistribution() ) return false;
   if (! SetMethod(method) ) return false;
   if (! SetRandomGenerator() ) return false; 
   return true;
}

    
bool TUnuran::Init(const TUnuranDistrMulti & distr, const std::string  & method, bool useLogpdf)  
{ 
   //  initialization with a distribution and method
   //   I copy the distribution object  (uset it by value)
   fDistrMulti= distr; 
   fUseLogpdf = useLogpdf; 
   fMethod = method;

   if (! SetDistributionMulti() ) return false;
   if (! SetMethod(method) ) return false;
   if (! SetRandomGenerator() ) return false; 
   return true; 
}

bool  TUnuran::Init(const std::string & dist, const std::string & method) 
{
   // initialize with a string
   std::string s = dist + " & " + method; 
   fGen = unur_str2gen(s.c_str() ); 
   if (fGen == 0) { 
      std::cerr << "ERROR: cannot create generator object" << std::endl; 
      return false; 
   } 
   return true; 
}



bool  TUnuran::SetRandomGenerator()
{
   // set an external random generator
   if (fRng == 0) return false; 
   UNUR_URNG * rng = unur_urng_new(&UnuranRng<TRandom>::Rndm, fRng );
   if (rng == 0) return false; 
   unsigned int ret = 0; 
   ret |= unur_urng_set_delete(rng, &UnuranRng<TRandom>::Delete); 
   ret |= unur_urng_set_seed(rng, &UnuranRng<TRandom>::Seed);

   // change generator
   if (fGen == 0) return false; 

   unur_chg_urng( fGen, rng); 
   return (ret ==0) ? true : false; 
   
}

bool  TUnuran::SetDistribution()
{ 
   fUdistr = unur_distr_cont_new(); 
   if (fUdistr == 0) return false; 
   unsigned int ret = 0; 
   ret |= unur_distr_set_extobj(fUdistr, &fDistr);  
   ret |= unur_distr_cont_set_pdf(fUdistr, &UnuranDistr<TUnuranDistr>::Pdf);  
   ret |= unur_distr_cont_set_dpdf(fUdistr, &UnuranDistr<TUnuranDistr>::Dpdf);  
   ret |= unur_distr_cont_set_cdf(fUdistr, &UnuranDistr<TUnuranDistr>::Cdf);  
   ret |= unur_distr_cont_set_mode(fUdistr, fDistr.Mode());  
   double xmin, xmax = 0; 
   if (fDistr.GetDomain(xmin,xmax) ) { 
      ret |= unur_distr_cont_set_domain(fUdistr,xmin,xmax);  
   }
   return (ret ==0) ? true : false; 
}


bool  TUnuran::SetDistributionMulti()
{
   fUdistr = unur_distr_cvec_new(fDistrMulti.NDim() ); 
   if (fUdistr == 0) return false; 
   unsigned int ret = 0; 
   ret |= unur_distr_set_extobj(fUdistr, &fDistrMulti);  
   if (!fUseLogpdf) { 
      ret |= unur_distr_cvec_set_pdf(fUdistr, &UnuranDistrMulti<TUnuranDistrMulti>::Pdf);  
      ret |= unur_distr_cvec_set_dpdf(fUdistr, &UnuranDistrMulti<TUnuranDistrMulti>::Dpdf);  
      ret |= unur_distr_cvec_set_pdpdf(fUdistr, &UnuranDistrMulti<TUnuranDistrMulti>::Pdpdf);  
   }
   else { 
      ret |= unur_distr_cvec_set_logpdf(fUdistr, &UnuranDistrMulti<TUnuranDistrMulti>::Logpdf);  
      ret |= unur_distr_cvec_set_dlogpdf(fUdistr, &UnuranDistrMulti<TUnuranDistrMulti>::Dlogpdf);  
      ret |= unur_distr_cvec_set_pdlogpdf(fUdistr, &UnuranDistrMulti<TUnuranDistrMulti>::Pdlogpdf);  
   }
   //unur_distr_cvec_set_mode(fUdistr, fDistr.Mode());  
//    double xmin, xmax = 0; 
//    if (fDistr.GetDomain(xmin,xmax) ) 
//       unur_distr_cont_set_domain(fUdistr,xmin,xmax);  
   return (ret ==0) ? true : false; 
}


bool TUnuran::SetMethod(const std::string & s) { 
   // set a method from a distribution
   if (fUdistr == 0) return false; 

   struct unur_slist *mlist = NULL;
   UNUR_PAR * par = _unur_str2par(fUdistr, s.c_str(), &mlist);
   if (par == 0) return false;
   fGen = unur_init(par); 
   _unur_slist_free(mlist);
   if (fGen == 0) return false; 
  // we can delete now the distribution object (it is copied in Unuran)
   // otherwise we need to add bookeeping
   unur_distr_free(fUdistr);
   fUdistr = 0;
   fMethod = s; 
   return true;
 }

double TUnuran::Sample()
{
   // sample one-dimensional distribution
   assert(fGen != 0); 
   return unur_sample_cont(fGen);
}

bool TUnuran::SampleMulti(double * x)
{
   // sample multidimensional distribution
   if (fGen == 0) return false;  
   unur_sample_vec(fGen,x);
   return true; 
}

bool TUnuran::Rinit()  
{ 
   //   re-initialize the distribution 
   // (in case the dist has changed somehow externally)
   bool ret = 0; 
   ret &= SetDistribution();
   ret &= SetMethod(fMethod);
   if (fGen == 0) return false; 
   return true; 
}


// bool TUnuran::Rinit (const std::string & s) 
// { 
//    //   re-initialize with a new  method
//    SetDistribution();
//    SetMethod(s);
//    if (fGen == 0) return false; 
//    return true; 
// }

bool  TUnuran::SetLogLevel(unsigned int debugLevel) 
{
   if (fGen == 0) return false; 
   int ret = 0; 
   if ( debugLevel > 2) 
      ret |= unur_chg_debug(fGen, UNUR_DEBUG_ALL);
   else if (debugLevel > 1) 
      ret |= unur_chg_debug(fGen, UNUR_DEBUG_ALL);
   else
      ret |= unur_chg_debug(fGen, UNUR_DEBUG_OFF);

   return (ret ==0) ? true : false; 

}
