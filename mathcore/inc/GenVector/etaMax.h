// @(#)root/mathcore:$Name:  $:$Id: etaMax.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , FNAL MathLib Team                             *
  *                                                                    *
  *                                                                    *
  **********************************************************************/


// Header file for function etaMax
// 
// Created by: Mark Fischler  at Thu Jun 2 2005
// 
// Last update: Fri Jun 3 2005
// 
#ifndef ROOT_MATH_ETAMAX
#define ROOT_MATH_ETAMAX 1

#include <limits>
#include <cmath>

namespace ROOT { 

  namespace Math { 

    /** 
	The following function could be called to provide the maximum possible 
	value of pseudorapidity for a non-zero Rho.  This is log ( max/min ) 
	where max and min are the extrema of positive values for type 
	long double.   
     */ 
    long double etaMax_impl() { 
      return std::log ( std::numeric_limits<long double>::max()/256.0l ) -
             std::log ( std::numeric_limits<long double>::denorm_min()*256.0l )
	     + 16.0 * std::log(2.0);
    // Actual usage of etaMax() simply returns the number 22756, which is 
    // the answer this would supply, rounded to a higher integer.	     
    }

    /** 
	Function providing the maximum possible value of pseudorapidity for
	a non-zero Rho, in the Scalar type with the largest dynamic range.
     */ 
    template <class T> 
    inline
    T etaMax() { 
      return static_cast<T>(22756.0);
    }

  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_MATH_ETAMAX */
