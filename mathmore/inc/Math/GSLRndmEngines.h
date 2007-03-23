// @(#)root/mathmore:$Name:  $:$Id: GSLRndmEngines.h,v 1.3 2006/06/19 08:44:08 moneta Exp $
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
#ifndef ROOT_Math_GSLRndmEngines
#define ROOT_Math_GSLRndmEngines

#include <string>
#include <vector>

namespace ROOT {
namespace Math {

  //struct GSLRngType; 
  //struct GSLRng; 
  //class  GSLRng; 
  //class GSLRngType;
  //typedef gsl_rng GSLRng; 
  //typedef gsl_rng_type GSLRngType; 
  class GSLRng; 


    /**
     Base class for all GSL engines

     @ingroup Random
    */ 
  class GSLRandomEngine { 

  public: 

    GSLRandomEngine();  

    GSLRandomEngine( GSLRng * rng) : 
      fRng(rng) , 
      fCurTime(0)
    {}

    void Initialize();

    void Terminate(); 

    virtual ~GSLRandomEngine(); 

    /**
       Generate a  random number between ]0,1]
       0 is excluded and 1 is included
     */
    double operator() ();  

    /**
       Generate an array of random numbers. 
       The iterators points to the random numbers 
     */
    template<class Iterator> 
    void RandomArray(Iterator begin, Iterator end) { 
      for ( Iterator itr = begin; itr != end; ++itr ) { 
	*itr = this->operator()(); 
      }
    }

    /**
       Generate an array of random numbers 
       The iterators points to the random numbers 
     */
    void RandomArray(double * begin, double * end);  

    /**
       return name of generator
     */ 
    std::string Name() const; 

    /**
       return the state size of generator 
     */ 
    unsigned int Size() const; 
    
    /** 
	set the random generator seed 
    */ 
    void SetSeed(unsigned int seed); 


    /** distributions implemented using the  
	<A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distributions.html">
	GSL Random number Distributions
     **/

    /**
       Gaussian distribution - default method is Box-Muller
     */
    double Gaussian(double sigma); 

    /**
       Gaussian distribution - Ziggurat method
     */
    double GaussianZig(double sigma);  

    /**
       Gaussian distribution - Ratio method
     */
    double GaussianRatio(double sigma); 
    /**
       Gaussian Tail distribution
     */
    double GaussianTail(double a, double sigma); 
  
    /**
       Bivariate Gaussian distribution with correlation
     */
    void Gaussian2D(double sigmaX, double sigmaY, double rho, double &x, double &y);
    
    /**
       Exponential distribution
     */
    double Exponential(double mu);

    /**
       Cauchy distribution
    */
    double Cauchy(double a); 

    /**
       Landau distribution
     */
    double Landau(); 

    /**
       Gamma distribution
     */
    double Gamma(double a, double b);

    /**
       Log Normal distribution
     */
    double LogNormal(double zeta, double sigma);

    /**
       Chi square distribution
     */
    double ChiSquare(double nu);

    /**
       F distrbution
     */
    double FDist(double nu1, double nu2);
    
    /**
       t student distribution
     */
    double tDist(double nu);

    /**
       generate random numbers in a 2D circle of radious 1 
     */
    void Dir2D(double &x, double &y); 

    /**
       generate random numbers in a 3D sphere of radious 1 
     */
    void Dir3D(double &x, double &y, double &z); 
  
    /**
       Poisson distribution
     */
    unsigned int Poisson(double mu);

    /**
       Binomial distribution
     */
    unsigned int Binomial(double p, unsigned int n);

    /**
       Multinomial distribution
     */
    std::vector<unsigned int> Multinomial( unsigned int ntot, const std::vector<double> & p ); 
  
      

  protected: 

  private: 

    GSLRng * fRng;               // pointer to GSL generator wrapper
    unsigned int  fCurTime;      // current time used to seed the generator

  }; 

  /**
     Mersenne-Twister genertaor

     @ingroup Random
   */
  class GSLRngMT : public GSLRandomEngine { 
  public: 
    GSLRngMT(); 
  };

  /**
     Ranlux generator (James, Luscher) (defaul luxury)

     @ingroup Random
   */
  class GSLRngRanLux : public GSLRandomEngine { 
  public: 
    GSLRngRanLux(); 
  };

  /**
     Second generation of Ranlux generator (with  luxury level of 2)

     @ingroup Random
   */
  class GSLRngRanLux2 : public GSLRandomEngine { 
  public: 
    GSLRngRanLux2(); 
  };

  /**
     48 bits version of Second generation of Ranlux generator (with  luxury level of 2)

     @ingroup Random
   */
  class GSLRngRanLux48 : public GSLRandomEngine { 
  public: 
    GSLRngRanLux48(); 
  };


  /**
     Tausworthe generator by L'Ecuyer

     @ingroup Random
   */
  class GSLRngTaus : public GSLRandomEngine { 
  public: 
    GSLRngTaus(); 
  };

  /**
     Lagged Fibonacci generator by Ziff

     @ingroup Random
   */
  class GSLRngGFSR4 : public GSLRandomEngine { 
  public: 
    GSLRngGFSR4(); 
  };

  /**
     Combined multiple recursive  generator (L'Ecuyer)

     @ingroup Random
   */ 
  class GSLRngCMRG : public GSLRandomEngine { 
  public: 
    GSLRngCMRG(); 
  };

  /**
     5-th order multiple recursive  generator (L'Ecuyer, Blouin and Coutre)

     @ingroup Random
   */ 
  class GSLRngMRG : public GSLRandomEngine { 
  public: 
    GSLRngMRG(); 
  };

  /**
     BSD rand() generator  

     @ingroup Random
   */
  class GSLRngRand : public GSLRandomEngine { 
  public: 
    GSLRngRand(); 
  };

  /**
     RANMAR generator 

     @ingroup Random
   */
  class GSLRngRanMar : public GSLRandomEngine { 
  public: 
    GSLRngRanMar(); 
  };

  /**
     MINSTD generator (Park and Miller)

     @ingroup Random
   */
  class GSLRngMinStd : public GSLRandomEngine { 
  public: 
    GSLRngMinStd(); 
  };
  



} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLRndmEngines */

