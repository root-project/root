// @(#)root/mathmore:$Id$
// Author: L. Moneta, A. Zsenei   08/2005 

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
#ifndef ROOT_Math_Random
#define ROOT_Math_Random

#include <string> 
#include <vector> 

/**
   @defgroup Random Random number generators and distributions
*/ 



namespace ROOT {
namespace Math {


//_____________________________________________________________________________________
  /**
     User class for MathMore random numbers template on the Engine type. 
     The API of this class followed that of the class ROOT::TRandom. 
     It must be implemented using as Engine one of the derived classes of 
     ROOT::Math::GSLRandomEngine, like ROOT::Math::GSLRngMT

     @ingroup Random 

   */ 
  template < class Engine> 
  class Random { 

  public: 


    /**
       Create a Random generator. Use default engine constructor. 
       Engine will  be initialized via Initialize() function in order to 
       allocate resources
     */
    Random() {
      fEngine.Initialize(); 
    } 

    /**
       Create a Random generator based using teh default enfing constructor and 
       then setting the given seed. 
       Engine will  be initialized via Initialize() function in order to 
       allocate resources
     */
    explicit Random(unsigned int seed)  {
      fEngine.Initialize(); 
      fEngine.SetSeed(seed);
    } 

    /**
       Create a Random generator based on a provided generic engine.
       Engine will  be initialized via Initialize() function in order to 
       allocate resources
     */
    explicit Random(const Engine & e) : fEngine(e) {
      fEngine.Initialize(); 
    } 

    /**
       Destructor: call Terminate() function of engine to free any 
       allocated resource
     */
    ~Random() { 
      fEngine.Terminate(); 
    }

    /**
       Generate random numbers between ]0,1]
       0 is excluded and 1 is included
     */
    double Uniform(double x=1.0) { 
      return x*fEngine(); 
    }
    /** 
       Generate random numbers between ]0,1]
       0 is excluded and 1 is included
       Function to preserve ROOT Trandom compatibility 
     */  
   double Rndm() { 
      return fEngine(); 
    }

    /** 
       Generate an array of random numbers between ]0,1]
       0 is excluded and 1 is included
       Function to preserve ROOT Trandom compatibility 
     */ 
    void RndmArray(int n, double * array) { 
      fEngine.RandomArray(array, array+n);
    }

    /**
       Return the type (name) of the used generator 
     */
    std::string Type() const { 
      return fEngine.Name();
    }

    /**
       Return the size of the generator state 
     */
    unsigned int EngineSize() const { 
      return fEngine.Size();
    }

    /** 
	set the random generator seed 
     */ 
    void SetSeed(unsigned int seed) { 
      return  fEngine.SetSeed(seed);
    }
    
    /** Random  Distributions 
	Use naming and signatures compatible with ROOT TRandom
     **/

    /**
       Gaussian distribution. Default method (use Ziggurat)
     */
    double Gaus(double mean = 0, double sigma = 1) { 
      return mean + fEngine.GaussianZig(sigma);
    }  

    /**
       Gaussian distribution (Box-Muller method)
     */
    double GausBM(double mean = 0, double sigma = 1) { 
      return mean + fEngine.Gaussian(sigma);
    }  

    /**
       Gaussian distribution (Ratio Method)
     */
    double GausR(double mean = 0, double sigma = 1) { 
      return mean + fEngine.GaussianRatio(sigma);
    }  

    /**
       Gaussian Tail distribution
     */
    double GaussianTail(double a, double sigma = 1) { 
      return fEngine.GaussianTail(a,sigma);
    }
  
    /**
       Bivariate Gaussian distribution with correlation
     */
    void Gaussian2D(double sigmaX, double sigmaY, double rho, double &x, double &y) { 
      fEngine.Gaussian2D(sigmaX, sigmaY, rho, x, y);
    }
    
    /**
       Exponential distribution
     */
    double Exp(double tau) { 
      return fEngine.Exponential(tau);
    }
    /**
       Breit Wigner distribution 
    */
    double BreitWigner(double mean = 0., double gamma = 1) { 
      return mean + fEngine.Cauchy( gamma/2.0 );
    } 

    /**
       Landau distribution
     */
    double Landau(double mean = 0, double sigma = 1) { 
      return mean + sigma*fEngine.Landau();
    } 

    /**
       Gamma distribution
     */
    double Gamma(double a, double b) { 
      return fEngine.Gamma(a,b);
    } 

    /**
       Log Normal distribution
     */
    double LogNormal(double zeta, double sigma) { 
      return fEngine.LogNormal(zeta,sigma);
    }

    /**
       Chi square distribution
     */
    double ChiSquare(double nu) { 
      return fEngine.ChiSquare(nu);
    }

    /**
       F distrbution
     */
    double FDist(double nu1, double nu2) { 
      return fEngine.FDist(nu1,nu2);
    }
    
    /**
       t student distribution
     */
    double tDist(double nu) { 
      return fEngine.tDist(nu);
    }

    /**
       generate random numbers in a 2D circle of radious 1 
     */
    void Circle(double &x, double &y, double r = 1) { 
      fEngine.Dir2D(x,y);
      x *= r;
      y *= r;
    } 

    /**
       generate random numbers in a 3D sphere of radious 1 
     */
    void Sphere(double &x, double &y, double &z,double r = 1) { 
      fEngine.Dir3D(x,y,z);
      x *= r;
      y *= r;
      z *= r;
    } 
  
    /**
       Poisson distribution
     */
    unsigned int Poisson(double mu) { 
      return fEngine.Poisson(mu); 
    }

    /**
       Binomial distribution
     */
    unsigned int Binomial(unsigned int ntot, double prob) { 
      return fEngine.Binomial(prob,ntot);
    }

    /**
       Negative Binomial distribution
       First parameter is n, second is probability
       To be consistent with Random::Binomial
     */
     unsigned int NegativeBinomial(double n, double prob) { 
      return fEngine.NegativeBinomial(prob,n);
    }

    /**
       Multinomial distribution
     */
    std::vector<unsigned int> Multinomial( unsigned int ntot, const std::vector<double> & p ) { 
      return fEngine.Multinomial(ntot,p);
    }


  private: 

    Engine fEngine; 

  }; 


} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_Random */



