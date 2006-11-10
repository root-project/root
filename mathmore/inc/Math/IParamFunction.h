// @(#)root/mathmore:$Name:  $:$Id: IParamFunction.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

// Header file for class IParamFunction
//
// interface for parameteric functions
// 
// Created by: Lorenzo Moneta  at Wed Nov 10 16:23:49 2004
// 
// Last update: Wed Nov 10 16:23:49 2004
// 
#ifndef ROOT_Math_IParamFunction
#define ROOT_Math_IParamFunction

#include "Math/IGenFunction.h"

#include <vector>

namespace ROOT {
namespace Math {

  /** 
      Interface for 1 Dimensional Parametric Functions.
      A parameteric function is a Generic Function with Parameters, so 
      it is a function object which carries a state, the Parameters. 
      The Parameters are described with a standard vector of doubles.

      @ingroup CppFunctions
  */


  class IParamFunction : public IGenFunction {

  public: 

    virtual ~IParamFunction() {} 
    
    /**
       Access the parameter values
     */
    virtual const std::vector<double> & Parameters() const = 0;

    // set params values (can user change number of params ? ) 
    /**
       Set the parameter values
       @param p vector of doubles containing the parameter values. 
     */
    virtual void SetParameters(const std::vector<double> & p) = 0;

    
    /**
       Return the number of Parameters
     */
    virtual unsigned int NumberOfParameters() const = 0; 



    /**
       Evaluate the derivatives of the function with respect to the parameters at a point x.
       It is optional to be implemented by the derived classes 
    */
    virtual const std::vector<double> & ParameterGradient(double /* x */ ) { 
      // in case of no Gradient provided return Parameters
      return Parameters(); 
    }

    // user override if implements parameter Gradient
    /**
       Return \a true if the calculation of the derivatives with respect to the Parameters is implemented
    */
    virtual bool  ProvidesParamGradient() const  { return false; }

  }; 

} // namespace Math
} // namespace ROOT

#endif /* MATHLIB_IPARAMFUNCTION */
