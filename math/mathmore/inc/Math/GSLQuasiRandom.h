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

// Header file for class GSLQuasiRandom
//
// Created by: moneta  at Sun Nov 21 16:26:03 2004
//
// Last update: Sun Nov 21 16:26:03 2004
//
#ifndef ROOT_Math_GSLQuasiRandom
#define ROOT_Math_GSLQuasiRandom

#include <string>

namespace ROOT {
namespace Math {


   class GSLQRngWrapper;

   //_________________________________________________________________
   /**
      GSLQuasiRandomEngine
      Base class for all GSL quasi random engines,
      normally user instantiate the derived classes
      which creates internally the generator and uses the class ROOT::Math::QuasiRandom


      @ingroup Random
   */
   class GSLQuasiRandomEngine {

   public:

     /**
         default constructor. No creation of rng is done.
         If then Initialize() is called an engine is created
         based on default GSL type (MT)
     */
      GSLQuasiRandomEngine();

      /**
          create from an existing rng.
          User manage the rng pointer which is then deleted olny by calling Terminate()
      */
      GSLQuasiRandomEngine( GSLQRngWrapper * rng);

      /**
         Copy constructor : clone the contained GSL generator
       */
      GSLQuasiRandomEngine(const GSLQuasiRandomEngine & eng);

      /**
         Assignment operator : make a deep copy of the contained GSL generator
       */
      GSLQuasiRandomEngine & operator=(const GSLQuasiRandomEngine & eng);

      /**
         initialize the generator giving the dimension of the sequence
         If no rng is present the default one based on Mersenne and Twister is created
       */
      void Initialize(unsigned int dimension);

      /**
         delete pointer to contained rng
       */
      void Terminate();

      /**
         call Terminate()
      */
      virtual ~GSLQuasiRandomEngine();

      /**
         Generate a  random number between ]0,1[
      */
      double operator() () const;

      /**
         Fill array x with random numbers between ]0,1[
      */
      bool operator() (double * x) const;

      /**
         Skip the next n random numbers
       */
      bool Skip(unsigned int n) const;

      /**
         Generate an array of quasi random numbers
         The iterators points to the random numbers
      */
      bool GenerateArray(double * begin, double * end) const;

      /**
         return name of generator
      */
      std::string Name() const;

      /**
         return the state size of generator
      */
      unsigned int Size() const;

      /**
         return the dimension of generator
      */
      unsigned int NDim() const;



   protected:

      /// internal method used by the derived class to set the type of generators
      void SetType(GSLQRngWrapper * r) {
         fQRng = r;
      }

   private:

      GSLQRngWrapper * fQRng;                // pointer to GSL generator wrapper (managed by the class)


   };

   //_____________________________________________________________________________________
   /**
      Sobol generator
      gsl_qrng_sobol from
      <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Quasi_002drandom-number-generator-algorithms.html#Quasi_002drandom-number-generator-algorithms">here</A>


      @ingroup Random
   */
   class GSLQRngSobol : public GSLQuasiRandomEngine {
   public:
      GSLQRngSobol();
   };

   //_____________________________________________________________________________________
   /**
      Niederreiter generator
      gsl_qrng_niederreiter_2 from
      <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Quasi_002drandom-number-generator-algorithms.html#Quasi_002drandom-number-generator-algorithms">here</A>

      @ingroup Random
   */
   class GSLQRngNiederreiter2 : public GSLQuasiRandomEngine {
   public:
      GSLQRngNiederreiter2();
   };




} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLQuasiRandom */

