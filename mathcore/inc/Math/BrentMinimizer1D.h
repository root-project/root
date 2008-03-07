// @(#)root/mathcore:$Id$
// Author: David Gonzalez Maline 2/2008
 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 Maline,  CERN/PH-SFT                            *
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

// Header file for class BrentMinimizer1D
// 
// Created by: Maline  at Mon Feb  4 09:32:36 2008
// 
//

#include "IMinimizer1D.h"
#include "Math/IFunction.h"

#ifndef ROOT_Math_BrentMinimizer1D
#define ROOT_Math_BrentMinimizer1D

namespace ROOT { 
namespace Math { 

   class BrentMinimizer1D: IMinimizer1D {
      
   public: 
      
      BrentMinimizer1D(); 
      virtual ~BrentMinimizer1D();
      
   public: 
      
      virtual double XMinimum() const; 
      virtual double XLower() const; 
      virtual double XUpper() const; 
      virtual double FValMinimum() const; 
      virtual double FValLower() const; 
      virtual double FValUpper() const; 
      virtual int Minimize( int maxIter, double absTol, double relTol); 
      virtual int Iterations() const;

      virtual const char * Name() const;  

      int SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup);

   protected:
      const IGenFunction* fFunction;
      double fXMin, fXMax;
      double fXMinimum;
      int fNIter;

   };  // end class BrentMinimizer1D
   
} // end namespace Math
   
} // end namespace ROOT

#endif /* ROOT_Math_BrentMinimizer1D */
