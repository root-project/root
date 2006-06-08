// @(#)root/mathmore:$Name:  $:$Id: WrappedFunction.h,v 1.2 2005/09/19 13:06:53 brun Exp $
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

#ifndef ROOT_Math_WrappedFunction
#define ROOT_Math_WrappedFunction

#include "IGenFunction.h"

#include <iostream>

namespace ROOT {
namespace Math {


/**
   Template class to wrap any C++ callable object which takes one argument 
   i.e. implementing operator() (double x). 
   It provides a ROOT::Math::IGenFunction-like signature
 */
template< class CALLABLE >
class WrappedFunction : public IGenFunction {


 public:

  explicit WrappedFunction( CALLABLE f ) : fFunc( f ) { /**/ }

  virtual WrappedFunction * Clone() const {

    //return new WrappedFunction( *this );
    return new WrappedFunction<CALLABLE>(fFunc);
  }

  virtual double operator() (double x) {
    return fFunc( x );
  }

  virtual ~WrappedFunction() { /**/ }

 private:

  CALLABLE fFunc;


}; // WrappedFunction


} // namespace Math
} // namespace ROOT



#endif // ROOT_Math_WrappedFunction
