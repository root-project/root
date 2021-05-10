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


   class GSLRngWrapper;
   class GSLMCIntegrator; 

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

      friend class GSLMCIntegrator;

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
      GSLRandomEngine( GSLRngWrapper * rng);

      /**
         Copy constructor : clone the contained GSL generator
       */
      GSLRandomEngine(const GSLRandomEngine & eng);

      /**
         Assignment operator : make a deep copy of the contained GSL generator
       */
      GSLRandomEngine & operator=(const GSLRandomEngine & eng);

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
         Generate a  random number between ]0,1]
         0 is excluded and 1 is included
      */
      double Rndm() const { return (*this)(); }

      /**
          Generate an integer number between [0,max-1] (including 0 and max-1)
          if max is larger than available range of algorithm
          an error message is printed and zero is returned
      */
      unsigned long RndmInt(unsigned long max) const;
      /**
          Generate an integer number between [0,max_generator-1] (including 0 and max-1)
          if max is larger than available range of algorithm
          an error message is printed and zero is returned
      */ 
      unsigned long IntRndm() const {
         return RndmInt(MaxInt());   // max return the largest value the generator can give +1
      }

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
         return the minimum integer a generator can handle
         typically this value is 0
       */
      unsigned long MinInt() const; 

      /**
         return the maximum integer +1 a generator can handle

       */
      unsigned long MaxInt() const; 

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
         Multivariate Gaussian distribution
      */
      void GaussianND(const int dim, double *pars, double *covmat, double *genpars) const;

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
         Beta distribution
      */
      double Beta(double a, double b) const;

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
         Rayleigh distribution
      */
      double Rayleigh(double sigma) const;

      /**
         Logistic distribution
      */
      double Logistic(double a) const;

      /**
         Pareto distribution
      */
      double Pareto(double a, double b) const;

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
         Negative Binomial distribution
      */
      unsigned int NegativeBinomial(double p, double n) const;

      /**
         Multinomial distribution
      */
      std::vector<unsigned int> Multinomial( unsigned int ntot, const std::vector<double> & p ) const;

      //@}



   protected:

      /// internal method used by the derived class to set the type of generators
      void SetType(GSLRngWrapper * r) {
         fRng = r;
      }

      /// internal method to return the engine
      /// Used by class like GSLMCIntegrator to set the engine
      GSLRngWrapper * Engine() {
         return fRng; 
      }
      
   private:

      GSLRngWrapper * fRng;                // pointer to GSL generator wrapper (managed by the class)
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
      typedef GSLRandomEngine BaseType; 
      GSLRngMT();
   };

   //_____________________________________________________________________________________
   /**
      Old Ranlux generator (James, Luscher) (default luxury level, p = 223)
      (This is eequivalent to TRandom1 with default luxury level)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngRanLux : public GSLRandomEngine {
   public:
      typedef GSLRandomEngine BaseType; 
      GSLRngRanLux();
   };

   //_____________________________________________________________________________________
   /**
      Second generation of Ranlux generator for single precision with  luxury level of 1
      (It throws away 202 values for every 12 used)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngRanLuxS1 : public GSLRandomEngine {
   public:
      typedef GSLRandomEngine BaseType; 
      GSLRngRanLuxS1();
   };
   typedef GSLRngRanLuxS1 GSLRngRanLux1; // for backward compatibility

   //_____________________________________________________________________________________
   /**
      Second generation of Ranlux generator for Single precision with  luxury level of 2
      (It throws away 397 value for every 12 used)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngRanLuxS2 : public GSLRandomEngine {
   public:
      typedef GSLRandomEngine BaseType; 
      GSLRngRanLuxS2();
   };
   typedef GSLRngRanLuxS2 GSLRngRanLux2; // for backward compatibility

   //_____________________________________________________________________________________
   /**
      Double precision (48 bits) version of Second generation of Ranlux generator with  luxury level of 1
      (It throws away 202 value for every 12 used)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngRanLuxD1 : public GSLRandomEngine {
   public:
      typedef GSLRandomEngine BaseType; 
      GSLRngRanLuxD1();
   };

   //_____________________________________________________________________________________
   /**
      Double precision (48 bits) version of Second generation of Ranlux generator with  luxury level of 2
      (It throws away 397 value for every 12 used)
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngRanLuxD2 : public GSLRandomEngine {
   public:
      typedef GSLRandomEngine BaseType; 
      GSLRngRanLuxD2();
   };
   typedef GSLRngRanLuxD2 GSLRngRanLux48; // for backward compatibility


   //_____________________________________________________________________________________
   /**
      Tausworthe generator by L'Ecuyer
      see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html">here</A>

      @ingroup Random
   */
   class GSLRngTaus : public GSLRandomEngine {
   public:
      typedef GSLRandomEngine BaseType; 
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
      typedef GSLRandomEngine BaseType; 
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
      typedef GSLRandomEngine BaseType; 
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
      typedef GSLRandomEngine BaseType; 
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
      typedef GSLRandomEngine BaseType; 
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
      typedef GSLRandomEngine BaseType; 
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
      typedef GSLRandomEngine BaseType; 
      GSLRngMinStd();
   };

   /** MixMax generator based on ROOT::Math::MixMaxEngine of N=240

       @ingroup Random 
   */
   class GSLRngMixMax : public GSLRandomEngine {
   public:
      typedef GSLRandomEngine BaseType; 
      GSLRngMixMax();
      virtual ~GSLRngMixMax();  // we need a dtcor since is not a standard GSL engine
   };

} // namespace Math
} // namespace ROOT

// random functions specialization for GSL
// needs to be defined after defining GSLRandomEngine class 

#include "Math/GSLRandomFunctions.h"

#endif /* ROOT_Math_GSLRndmEngines */

