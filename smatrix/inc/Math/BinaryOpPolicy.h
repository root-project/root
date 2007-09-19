// @(#)root/smatrix:$Id$
// Authors: J. Palacios    2006  
#ifndef ROOT_Math_BinaryOpPolicy 
#define ROOT_Math_BinaryOpPolicy 1

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


#ifndef ROOT_Math_MatrixRepresentationsStatic
#include "Math/MatrixRepresentationsStatic.h"
#endif

namespace ROOT { 

  namespace Math {

    /**
       matrix-matrix multiplication policy
     */
    template <class T, class R1, class R2>
    struct MultPolicy
    {
      enum { 
	N1 = R1::kRows,
	N2 = R2::kCols
      };
      typedef MatRepStd<T, N1, N2> RepType;
    };

    /**
       matrix addition policy
     */
    template <class T, unsigned int D1, unsigned int D2, class R1, class R2>
    struct AddPolicy
    {
      enum { 
	N1 = R1::kRows,
	N2 = R1::kCols
      };
      typedef MatRepStd<typename R1::value_type, N1, N2 > RepType;  
    };

    template <class T, unsigned int D1, unsigned int D2>
    struct AddPolicy<T, D1, D2, MatRepSym<T,D1>, MatRepSym<T,D1> >
    {
      typedef  MatRepSym<T,D1> RepType;
    };

    /**
       matrix transpose policy
     */
    template <class T, unsigned int D1, unsigned int D2, class R>
    struct TranspPolicy
    {
      enum { 
	N1 = R::kRows,
	N2 = R::kCols
      };
      typedef MatRepStd<T, N2, N1> RepType;
    };
    // specialized case of transpose of sym matrices
    template <class T, unsigned int D1, unsigned int D2>
    struct TranspPolicy<T, D1, D2, MatRepSym<T,D1> > 
    {
      typedef MatRepSym<T, D1> RepType;
    };
  }  // namespace Math
  
}  // namespace ROOT

#endif // MATH_BINARYOPPOLICY_H
