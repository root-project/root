// @(#)root/mathmore:$Name:  $:$Id: GSLRndmEngines.cxx,v 1.2 2006/05/26 14:30:17 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
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

// Header file for class GSLRandom
// 
// Created by: moneta  at Sun Nov 21 16:26:03 2004
// 
// Last update: Sun Nov 21 16:26:03 2004
// 



// need to be included later
#include <time.h>
#include <cassert>

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"


#include "Math/GSLRndmEngines.h"

//#include <iostream>

namespace ROOT {
namespace Math {



  /**
     wrapper to gsl_rng and gsl_rng_type
  */
  class GSLRng { 
    public:


    GSLRng() : 
      fRng(0),
      fRngType(0) 
    {
    }

    GSLRng(const gsl_rng_type * type) : 
      fRng(0),
      fRngType(type) 
    {
    }

    ~GSLRng() { /** no op */  } 

    void Allocate() { 
      if (fRngType == 0) SetDefaultType();
      fRng = gsl_rng_alloc( fRngType );
      //std::cout << " allocate   " << fRng <<  std::endl;
    }

    void Free() { 
      //std::cout << "free gslrng " << fRngType <<  "  " << fRng <<  std::endl;
      if (fRng != 0) gsl_rng_free(fRng);       
      fRng = 0; 
    }



    void SetType(const gsl_rng_type * type) { 
      fRngType = type; 
    }

    void SetDefaultType() { 
      // construct default engine
      gsl_rng_env_setup(); 
      fRngType =  gsl_rng_default; 
    }



    inline gsl_rng * Rng() const { return fRng; } 

  private: 

    gsl_rng * fRng; 
    const gsl_rng_type * fRngType; 
  };



  // default constructor 
  GSLRandomEngine::GSLRandomEngine() : 
    fCurTime(0)
  {  
    fRng = new GSLRng();
  } 

  GSLRandomEngine::~GSLRandomEngine() { 
    /* no op , fRng is delete in terminate to avoid problem when copying the instances*/
  }

  void GSLRandomEngine::Initialize() { 
  //----------------------------------------------------
    assert(fRng);
    fRng->Allocate(); 
  }

  void GSLRandomEngine::Terminate() { 
  //----------------------------------------------------
    assert(fRng);
    fRng->Free();
    delete fRng; 
    fRng = 0; 
  }


  double GSLRandomEngine::operator() () { 
    // generate random between 0 and 1. 
    // 0 is excluded 
    assert(fRng);
    return gsl_rng_uniform_pos(fRng->Rng() ); 
  }

  void GSLRandomEngine::RandomArray(double * begin, double * end )  { 
    // generate array of randoms betweeen 0 and 1. 0 is excluded 
    // specialization for double * (to be faster) 
      assert(fRng);
      for ( double * itr = begin; itr != end; ++itr ) { 
	*itr = gsl_rng_uniform_pos(fRng->Rng() ); 
      }
  }

  void GSLRandomEngine::SetSeed(unsigned int seed) { 
  // set the seed, if = 0then the seed is set randomly using an std::rand()
  // seeded with the current time. Be carefuk in case the current time is 
  // the same in consecutive calls 
    if (seed == 0) { 
      // use like in root (use time)
      time_t curtime;  
      time(&curtime); 
      unsigned int ct = static_cast<unsigned int>(curtime);
      if (ct != fCurTime) { 
	fCurTime = ct; 
	// set the seed for rand
	std::srand(ct); 
      }
      seed = std::rand();
    } 

    assert(fRng);
    gsl_rng_set(fRng->Rng(), seed ); 
  }

  std::string GSLRandomEngine::Name() const { 
  //----------------------------------------------------
    assert ( fRng != 0); 
    assert ( fRng->Rng() != 0 ); 
    return std::string( gsl_rng_name( fRng->Rng() ) ); 
  }

  unsigned int GSLRandomEngine::Size() const { 
  //----------------------------------------------------
    return gsl_rng_size( fRng->Rng() ); 
  }


  // Random distributions
  
  double GSLRandomEngine::Gaussian(double sigma)  
  {
    // Gaussian distribution
    //#define HAVE_GSL18
#ifdef HAVE_GSL18
    return gsl_ran_gaussian_ziggurat(  fRng->Rng(), sigma);
#else
    return gsl_ran_gaussian(  fRng->Rng(), sigma);
#endif
  }


  double GSLRandomEngine::GaussianTail(double a , double sigma) 
  {
    // Gaussian Tail distribution: eeturn values larger than a distributed 
    // according to the gaussian 
    return gsl_ran_gaussian_tail(  fRng->Rng(), a, sigma);
  }

  void GSLRandomEngine::Gaussian2D(double sigmaX, double sigmaY, double rho, double &x, double &y) 
  { 
    // Gaussian Bivariate distribution, with correlation coefficient rho
    gsl_ran_bivariate_gaussian(  fRng->Rng(), sigmaX, sigmaY, rho, &x, &y);
  }
  
  double GSLRandomEngine::Exponential(double mu)  
  {
    // Exponential distribution
    return gsl_ran_exponential(  fRng->Rng(), mu);
  }

  double GSLRandomEngine::Cauchy(double a) 
  {
    // Cauchy distribution
    return gsl_ran_cauchy(  fRng->Rng(), a);
  }

  double GSLRandomEngine::Landau() 
  {
    // Landau distribution
    return gsl_ran_landau(  fRng->Rng());
  }

  double GSLRandomEngine::Gamma(double a, double b) 
  {
    // Gamma distribution
    return gsl_ran_gamma(  fRng->Rng(), a, b);
  }

  double GSLRandomEngine::LogNormal(double zeta, double sigma)
  {
    // Log normal distribution
    return gsl_ran_lognormal(  fRng->Rng(), zeta, sigma);
  }

  double GSLRandomEngine::ChiSquare(double nu)
  {
    // Chi square distribution
    return gsl_ran_chisq(  fRng->Rng(), nu);
  }


  double GSLRandomEngine::FDist(double nu1, double nu2)  
  {
    // F distribution
    return gsl_ran_fdist(  fRng->Rng(), nu1, nu2);
  }

  double GSLRandomEngine::tDist(double nu)  
  {
    // t distribution
    return gsl_ran_tdist(  fRng->Rng(), nu);
  }
  
  void GSLRandomEngine::Dir2D(double &x, double &y) 
  { 
    // generate random numbers in a 2D circle of radious 1 
    gsl_ran_dir_2d(  fRng->Rng(), &x, &y);
  }

  void GSLRandomEngine::Dir3D(double &x, double &y, double &z) 
  { 
    // generate random numbers in a 3D sphere of radious 1 
    gsl_ran_dir_3d(  fRng->Rng(), &x, &y, &z);
  }
  
  unsigned int GSLRandomEngine::Poisson(double mu) 
  { 
    // Poisson distribution
    return gsl_ran_poisson(  fRng->Rng(), mu);
  }

  unsigned int GSLRandomEngine::Binomial(double p, unsigned int n) 
  { 
    // Binomial distribution
    return gsl_ran_binomial(  fRng->Rng(), p, n);
  }


  std::vector<unsigned int>  GSLRandomEngine::Multinomial( unsigned int ntot, const std::vector<double> & p ) 
  { 
    // Multinomial distribution  return vector of integers which sum is ntot
    unsigned int * narray = new unsigned int(p.size() ); 
    gsl_ran_multinomial(  fRng->Rng(), p.size(), ntot, &p.front(), narray);
    std::vector<unsigned int> ival( narray, narray+p.size()); 
    return ival; 
  }



  //----------------------------------------------------
  // generators 
  //----------------------------------------------------

  //----------------------------------------------------
  GSLRngMT::GSLRngMT() : 
    GSLRandomEngine(new GSLRng(gsl_rng_mt19937) )
  {}



  GSLRngRanLux::GSLRngRanLux() : 
    GSLRandomEngine(new GSLRng(gsl_rng_ranlux) )
  {}

  // second generation of Ranlux (double precision version)
  GSLRngRanLux2::GSLRngRanLux2() : 
    GSLRandomEngine(new GSLRng(gsl_rng_ranlxs2) )
  {}

  // 48 bits version
  GSLRngRanLux48::GSLRngRanLux48() : 
    GSLRandomEngine(new GSLRng(gsl_rng_ranlxd2) )
  {}

  //----------------------------------------------------
  GSLRngTaus::GSLRngTaus() : 
    GSLRandomEngine(new GSLRng(gsl_rng_taus2) )
  {}

  //----------------------------------------------------
  GSLRngGFSR4::GSLRngGFSR4() : 
    GSLRandomEngine(new GSLRng(gsl_rng_gfsr4) )
  {}

  //----------------------------------------------------
  GSLRngCMRG::GSLRngCMRG() : 
    GSLRandomEngine(new GSLRng(gsl_rng_cmrg) )
  {}

  //----------------------------------------------------
  GSLRngMRG::GSLRngMRG() : 
    GSLRandomEngine(new GSLRng(gsl_rng_mrg) )
  {}


  //----------------------------------------------------
  GSLRngRand::GSLRngRand() : 
    GSLRandomEngine(new GSLRng(gsl_rng_rand) )
  {}

  //----------------------------------------------------
  GSLRngRanMar::GSLRngRanMar() : 
    GSLRandomEngine(new GSLRng(gsl_rng_ranmar) )
  {}

  //----------------------------------------------------
  GSLRngMinStd::GSLRngMinStd() : 
    GSLRandomEngine(new GSLRng(gsl_rng_minstd) )
  {}





} // namespace Math
} // namespace ROOT



