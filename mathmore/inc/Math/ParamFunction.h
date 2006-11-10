// @(#)root/mathmore:$Name:  $:$Id: ParamFunction.h,v 1.2 2005/09/19 13:06:53 brun Exp $
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

// Header file for class ParamFunction
//
// Base class for Parametric functions
// 
// Created by: Lorenzo Moneta  at Wed Nov 10 16:38:34 2004
// 
// Last update: Wed Nov 10 16:38:34 2004
// 
#ifndef ROOT_Math_ParamFunction
#define ROOT_Math_ParamFunction


#include "Math/IParamFunction.h"

namespace ROOT {
namespace Math {

  /** 
      Base class for 1 Dimensional Parametric Functions.
      A parameteric function is a Generic Function with parameters, so 
      it is a function object which carries a state, the parameters. 
      The parameters are described with a standard vector of doubles.

      This class contains the default implementations for the methods defined in the 
      IParamFunction interface.
      Specific parameteric function classes should derive from this class if they want to profit from 
      default implementations for the abstract methods. 
      The derived classes need to implement only the ParamFunction::operator(double x) and ParamFunction::Clone() methods. 

      @ingroup CppFunctions
  */



  class ParamFunction : public IParamFunction {

  public: 

    /**
       Construct a parameteric function with npar parameters
       @param npar number of parameters (default is zero)
       @param providesGrad flag to specify if function implements the calculation of the derivative
       @param providesParamGrad flag to specify if function implements the calculation of the derivatives with respect to the Parameters 
     */
    ParamFunction(unsigned int npar = 0, bool providesGrad = false, bool providesParamGrad = false);  
    
    // destructor
    virtual ~ParamFunction() {}


    // copying constructors

    // use defaults one ??
    //ParamFunction(const ParamFunction & pf);  
    //ParamFunction & operator = (const ParamFunction &); 


    // cloning
    /**
       Deep copy of function (to be implemented by the derived classes)
     */
    virtual IGenFunction *  Clone() const = 0; 


    
    /**
       Access the parameter values
    */
    const std::vector<double> & Parameters() const { return fParams; } 

    /**
       Set the parameter values
       @param p vector of doubles containing the parameter values. 
     */
    void SetParameters(const std::vector<double> & p)
      // pending : can user change number of Parameters ? 
      { 
	fParams = p;
	fNpar = fParams.size(); 
      } 

    /**
       Return the number of parameters
     */
    unsigned int NumberOfParameters() const { return fNpar; } 

    // this needs to be implemented in derived classes 
    /**
       Evaluate function at a point x  (to be implemented by the derived classes)
     */
    virtual double operator() (double x ) = 0; 

    // user may re-implement this for better efficiency
    // this method is NOT required to  change internal values of parameters. confusing ?? 
    /**
       Evaluate function at a point x and for parameters p.
       This method mey be needed for better efficiencies when for each function evaluation the parameters are changed.
     */
    virtual double operator() (double x, const std::vector<double> & p ) 
      { 
	SetParameters(p); 
	return operator() (x); 
      }

    /**
       Return \a true if the calculation of derivatives is implemented
     */
    bool ProvidesGradient() const {  return fProvGrad; } 

    /**
       Return \a true if the calculation of derivatives with respect to the Parameters is implemented
     */
    bool ProvidesParameterGradient() const {  return fProvParGrad; } 


public: 


protected: 
  
  // Parameters (make protected to be accessible directly by derived classes) 
  std::vector<double> fParams;

  // cache paramGradient for better efficiency (to be used by derived classes) 
  mutable std::vector<double> fParGradient; 


private: 

  // cache number of Parameters for speed efficiency
  unsigned int fNpar; 

  bool fProvGrad; 
  bool fProvParGrad;

}; 

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_ParamFunction */
