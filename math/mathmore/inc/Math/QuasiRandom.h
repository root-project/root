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
#ifndef ROOT_Math_QuasiRandom
#define ROOT_Math_QuasiRandom

#include <string>

/**
   @defgroup QuasiRandom QuasiRandom number generators and distributions
*/



namespace ROOT {
namespace Math {


//_____________________________________________________________________________________
/**
   User class for MathMore random numbers template on the Engine type.
   The API of this class followed that of the class ROOT::Math::Random
   It must be implemented using as Engine one of the derived classes of
   ROOT::Math::GSLQuasiRandomEngine, like ROOT::Math::GSLQrngSobol

   @ingroup Random

*/
template < class Engine>
class QuasiRandom {

public:


   /**
      Create a QuasiRandom generator. Use default engine constructor.
      Engine will  be initialized via Initialize() function in order to
      allocate resources
   */
   QuasiRandom(unsigned int dimension = 1) {
      fEngine.Initialize(dimension);
   }


   /**
      Create a QuasiRandom generator based on a provided generic engine.
      Engine will  be initialized via Initialize() function in order to
      allocate resources
   */
   explicit QuasiRandom(const Engine & e, unsigned int dimension = 1) : fEngine(e) {
      fEngine.Initialize(dimension);
   }

   /**
      Destructor: call Terminate() function of engine to free any
      allocated resource
   */
   ~QuasiRandom() {
      fEngine.Terminate();
   }

   /**
      Generate next quasi random numbers points
   */
   bool Next(double * x) {
      return fEngine(x);
   }

   /**
      Generate next quasi random numbers point (1 - dimension)
   */
   double Next() {
      return fEngine();
   }

   /**
       Generate quasi random numbers between ]0,1[
       0 and 1 are excluded
       Function to be compatible with  ROOT TRandom compatibility
   */
   double Rndm() {
      return fEngine();
   }

   /**
      skip the next n number and jumb directly to the current state + n
   */
   bool Skip(unsigned int n) {
      return fEngine.Skip(n);
   }
   /**
       Generate an array of random numbers between ]0,1[
       Function to preserve ROOT Trandom compatibility
       The array will be filled as   x1,y1,z1,....x2,y2,z2,...
   */
   bool RndmArray(int n, double * array) {
      return fEngine.GenerateArray(array, array+n*NDim());
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
      Return the dimension of the generator
   */
   unsigned int NDim() const {
      return fEngine.NDim();
   }

   /**
      Return the name of the generator
   */
   std::string Name() const {
      return fEngine.Name();
   }

private:

   Engine fEngine;

};



} // namespace Math
} // namespace ROOT

#ifndef ROOT_Math_GSLQuasiRandom
#include "Math/GSLQuasiRandom.h"
#endif




namespace ROOT {
namespace Math {


typedef QuasiRandom<ROOT::Math::GSLQRngSobol> QuasiRandomSobol;
typedef QuasiRandom<ROOT::Math::GSLQRngNiederreiter2> QuasiRandomNiederreiter;

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_QuasiRandom */



