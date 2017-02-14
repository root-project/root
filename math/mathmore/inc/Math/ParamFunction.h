// @(#)root/mathmore:$Id$
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

#include <vector>

namespace ROOT {
namespace Math {

//_____________________________________________________________________________________
   /**
      Base template class for all Parametric Functions.
      The template argument is the type of parameteric function interface is implementing like
      Parameteric 1D, Multi-Dim or gradient parametric.

      A parameteric function is a Generic Function with parameters, so
      it is a function object which carries a state, the parameters.
      The parameters are described with a standard vector of doubles.

      This class contains the default implementations for the methods defined in the
      IParamFunction interface for dealing with parameters
      Specific parameteric function classes should derive from this class if they want to profit from
      default implementations for the abstract methods.
      The derived classes need to implement only the DoEvalPar( x, p) and Clone() methods for non-gradient
      parameteric functions or DoParameterDerivative(x,p,ipar) for gradient par functions


      @ingroup ParamFunc
   */


template <class IPFType>
class ParamFunction : public IPFType  {

public:

   typedef IPFType           BaseParFunc;
   typedef typename IPFType::BaseFunc BaseFunc;

   /**
      Construct a parameteric function with npar parameters
      @param npar number of parameters (default is zero)
   */
   ParamFunction(unsigned int npar = 0) :
      fNpar(npar),
      fParams( std::vector<double>(npar) )
   {  }


   // destructor
   virtual ~ParamFunction() {}


   // copying constructors (can use default ones)




   /**
      Access the parameter values
   */
   virtual const double * Parameters() const { return &fParams.front(); }

   /**
      Set the parameter values
      @param p vector of doubles containing the parameter values.
   */
   virtual void SetParameters(const double * p)
   {
      //fParams = std::vector<double>(p,p+fNpar);
      assert(fParams.size() == fNpar);
      std::copy(p,p+fNpar,fParams.begin());
   }

   /**
      Return the number of parameters
   */
   unsigned int NPar() const { return fNpar; }


   //using BaseFunc::operator();

   /**
      Return \a true if the calculation of derivatives is implemented
   */
//   bool ProvidesGradient() const {  return fProvGrad; }

   /**
      Return \a true if the calculation of derivatives with respect to the Parameters is implemented
   */
   //bool ProvidesParameterGradient() const {  return fProvParGrad; }

//    const std::vector<double> & GetParGradient( double x) {
//       BaseParFunc::ParameterGradient(x,&fParGradient[0]);
//       return fParGradient;
//    }



private:

   // cache number of Parameters for speed efficiency
   unsigned int fNpar;

protected:

   // Parameters (make protected to be accessible directly by derived classes)
   std::vector<double> fParams;

};

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_ParamFunction */
