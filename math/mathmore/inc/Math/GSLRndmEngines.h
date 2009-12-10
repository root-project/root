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
   class GSLRngWrapper; 

   //_________________________________________________________________
   /**
      GSLRandomEngine
      Base class for all GSL random engines, 
      normally user instantiate the derived classes
      which creates internally the generator.

      The main GSL generators (see 
      <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">
      here</A>) are available as derived classes 
      In addition to generate uniform numbers it provides method for 
      generating numbers according to pre-defined distributions 
      using the GSL functions from 
      <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distributions.html">
      GSL random number distributions</A>. 


       
      @ingroup Random
   */ 
   class GSLRandomEngine { 

   public: 

     /** 
         default constructor. No creation of rng is done. 
         If then Initialize() is called an engine is created 
         based on default GSL type (MT) 
     */
      GSLRandomEngine();  

      /** 
          create from an existing rng. 
          User manage the rng pointer which is then deleted olny by calling Terminate()
      */
      GSLRandomEngine( GSLRngWrapper * rng) : 
         fRng(rng) , 
         fCurTime(0)
      {}

      /**
         initialize the generator 
         If no rng is present the default one based on Mersenne and Twister is created 
       */
      void Initialize();

      /**
         delete pointer to contained rng 
       */
      void Terminate(); 

      /**
         call Terminate()
      */
      virtual ~GSLRandomEngine();

      /**
         Generate a  random number between ]0,1]
         0 is excluded and 1 is included
      */
      double operator() () const;  

      /** 
          Generate an integer number between [0,max-1] (including 0 and max-1)
          if max is larger than available range of algorithm 
          an error message is printed and zero is returned
      */
      unsigned int RndmInt(unsigned int max) const; 

      /**
         Generate an array of random numbers. 
         The iterators points to the random numbers 
      */
      template<class Iterator> 
      void RandomArray(Iterator begin, Iterator end) const { 
         for ( Iterator itr = begin; itr != end; ++itr ) { 
            *itr = this->operator()(); 
         }
      }

      /**
         Generate an array of random numbers 
         The iterators points to the random numbers 
      */
      void RandomArray(double * begin, double * end) const;  

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
      void SetSeed(unsigned int seed) const; 


      /** @name Random Distributions 
          Implemented using the  
          <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distributions.html">
          GSL Random number Distributions</A>
      **/
      //@{
      /**
         Gaussian distribution - default method is Box-Muller (polar method)
      */
      double Gaussian(double sigma) const; 

      /**
         Gaussian distribution - Ziggurat method
      */
      double GaussianZig(double sigma) const;  

      /**
         Gaussian distribution - Ratio method
      */
      double GaussianRatio(double sigma) const; 
      /**
         Gaussian Tail distribution
      */
      double GaussianTail(double a, double sigma) const; 
  
      /**
         Bivariate Gaussian distribution with correlation
      */
      void Gaussian2D(double sigmaX, double sigmaY, double rho, double &x, double &y) const;
    
      /**
         Exponential distribution
      */
      double Exponential(double mu) const;

      /**
         Cauchy distribution
      */
      double Cauchy(double a) const; 

      /**
         Landau distribution
      */
      double Landau() const; 

      /**
         Gamma distribution
      */
      double Gamma(double a, double b) const;

      /**
         Log Normal distribution
      */
      double LogNormal(double zeta, double sigma) const;

      /**
         Chi square distribution
      */
      double ChiSquare(double nu) const;

      /**
         F distrbution
      */
      double FDist(double nu1, double nu2) const;
    
      /**
         t student distribution
      */
      double tDist(double nu) const;

      /**
         generate random numbers in a 2D circle of radious 1 
      */
      void Dir2D(double &x, double &y) const; 

      /**
         generate random numbers in a 3D sphere of radious 1 
      */
      void Dir3D(double &x, double &y, double &z) const; 
  
      /**
         Poisson distribution
      */
      unsigned int Poisson(double mu) const;

      /**
         Binomial distribution
      */
      unsigned int Binomial(double p, unsigned int n) const;

      /**
         Multinomial distribution
      */
      std::vector<unsigned int> Multinomial( unsigned int ntot, const std::vector<double> & p ) const; 

      //@}
  
      

   protected: 

   private: 

      GSLRngWrapper * fRng;               // pointer to GSL generator wrapper
      mutable unsigned int  fCurTime;      // current time used to seed the generator

   }; 
   
   //_____________________________________________________________________________________
   /**
      Mersenne-Twister generator
      gsl_rng_mt19937 from 
      <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      
      @ingroup Random
   */
   class GSLRngMT : public GSLRandomEngine { 
   public: 
      GSLRngMT(); 
   };

   //_____________________________________________________________________________________
   /**
      Ranlux generator (James, Luscher) (defaul luxury)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngRanLux : public GSLRandomEngine { 
   public: 
      GSLRngRanLux(); 
   };

   //_____________________________________________________________________________________
   /**
      Second generation of Ranlux generator (with  luxury level of 2)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngRanLux2 : public GSLRandomEngine { 
   public: 
      GSLRngRanLux2(); 
   };

   //_____________________________________________________________________________________
   /**
      48 bits version of Second generation of Ranlux generator (with  luxury level of 2)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngRanLux48 : public GSLRandomEngine { 
   public: 
      GSLRngRanLux48(); 
   };


   //_____________________________________________________________________________________
   /**
      Tausworthe generator by L'Ecuyer
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngTaus : public GSLRandomEngine { 
   public: 
      GSLRngTaus(); 
   };

   //_____________________________________________________________________________________
   /**
      Lagged Fibonacci generator by Ziff
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngGFSR4 : public GSLRandomEngine { 
   public: 
      GSLRngGFSR4(); 
   };

   //_____________________________________________________________________________________
   /**
      Combined multiple recursive  generator (L'Ecuyer)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */ 
   class GSLRngCMRG : public GSLRandomEngine { 
   public: 
      GSLRngCMRG(); 
   };

   //_____________________________________________________________________________________
   /**
      5-th order multiple recursive  generator (L'Ecuyer, Blouin and Coutre)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */ 
   class GSLRngMRG : public GSLRandomEngine { 
   public: 
      GSLRngMRG(); 
   };

   //_____________________________________________________________________________________
   /**
      BSD rand() generator  
      gsl_rmg_rand from 
      <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Unix-random-number-generators.html">here</A>
      
      @ingroup Random
   */
   class GSLRngRand : public GSLRandomEngine { 
   public: 
      GSLRngRand(); 
   };

   //_____________________________________________________________________________________
   /**
      RANMAR generator 
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Unix-random-number-generators.html">here</A>

      @ingroup Random
   */
   class GSLRngRanMar : public GSLRandomEngine { 
   public: 
      GSLRngRanMar(); 
   };

   //_____________________________________________________________________________________
   /**
      MINSTD generator (Park and Miller)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Unix-random-number-generators.html">here</A>

      @ingroup Random
   */
   class GSLRngMinStd : public GSLRandomEngine { 
   public: 
      GSLRngMinStd(); 
   };
  



} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLRndmEngines */

