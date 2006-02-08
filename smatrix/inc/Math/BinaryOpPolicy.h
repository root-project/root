// @(#)root/smatrix:$Name:  $:$Id: BinaryOpPolicy.h,v 1.1 2006/02/08 14:45:35 moneta Exp $
// Authors: J. Palacios    2006  
#ifndef ROOT_Math_BinaryOpPolicy_h 
#define ROOT_Math_BinaryOpPolicy_h 1

// Include files

/** @class BinaryOpPolicy BinaryOpPolicy.h Math/BinaryOpPolicy.h
 *  
 *
 *  @author Juan PALACIOS
 *  @date   2006-01-10
 *
 *  Classes to define matrix representation binary combination policy.
 *  At the moment deals with symmetric and generic representation, and
 *  establishes policies for multiplication (and division) and addition
 *  (and subtraction)
 */

//#include "Math/MatrixRepresentations.h"
#include "Math/MatrixRepresentationsStatic.h"

namespace ROOT { 

  namespace Math {


    template <class T, class R1, class R2>
    struct MultPolicy
    {
      typedef MatRepStd<T, R1::kRows, R2::kCols> RepType;
    };

    template <class T, unsigned int D1, unsigned int D2, class R1, class R2>
    struct AddPolicy
    {
      typedef MatRepStd<typename R1::value_type, R1::kRows, R1::kCols> RepType;  
    };

    template <class T, unsigned int D1, unsigned int D2>
    struct AddPolicy<T, D1, D2, MatRepSym<T,D1>, MatRepSym<T,D1> >
    {
      typedef  MatRepSym<T,D1> RepType;
    };

  }  // namespace Math
  
}  // namespace ROOT

#endif // MATH_BINARYOPPOLICY_H
