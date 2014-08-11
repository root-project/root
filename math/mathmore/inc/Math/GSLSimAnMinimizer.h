// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 17:16:32 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
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

// Header file for class GSLSimAnMinimizer

#ifndef ROOT_Math_GSLSimAnMinimizer
#define ROOT_Math_GSLSimAnMinimizer



#ifndef ROOT_Math_BasicMinimizer
#include "Math/BasicMinimizer.h"
#endif


#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif

#ifndef ROOT_Math_IParamFunctionfwd
#include "Math/IParamFunctionfwd.h"
#endif



#ifndef ROOT_Math_GSLSimAnnealing
#include "Math/GSLSimAnnealing.h"
#endif




namespace ROOT {

   namespace Math {



//_____________________________________________________________________________________
/**
   GSLSimAnMinimizer class for minimization using simulated annealing
   using the algorithm from
   <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Simulated-Annealing.html">
   GSL</A>.
   It implements the ROOT::Minimizer interface and
   a plug-in (name "GSLSimAn") exists to instantiate this class via the plug-in manager

   @ingroup MultiMin
*/
class GSLSimAnMinimizer : public  ROOT::Math::BasicMinimizer {

public:

   /**
      Default constructor
   */
   GSLSimAnMinimizer (int type = 0);

   /**
      Destructor (no operations)
   */
   virtual ~GSLSimAnMinimizer ();

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   GSLSimAnMinimizer(const GSLSimAnMinimizer &) : ROOT::Math::BasicMinimizer() {}

   /**
      Assignment operator
   */
   GSLSimAnMinimizer & operator = (const GSLSimAnMinimizer & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }

public:


   /// method to perform the minimization
   virtual  bool Minimize();

   //number of calls
   unsigned int NCalls() const;


protected:

private:


   ROOT::Math::GSLSimAnnealing  fSolver;


};

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLSimAnMinimizer */
