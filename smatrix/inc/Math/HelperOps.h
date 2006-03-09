// @(#)root/smatrix:$Name:  $:$Id: HelperOps.h,v 1.2 2006/03/08 15:11:06 moneta Exp $
// Authors: J. Palacios    2006  

#ifndef ROOT_Math_HelperOps_h 
#define ROOT_Math_HelperOps_h 1

// Include files

/** @class HelperOps HelperOps.h Math/HelperOps.h
 *  
 *
 *  @author Juan PALACIOS
 *  @date   2006-01-11
 *
 *  Specialised helper classes for binary operators =, +=, -=
 *  between SMatrices and Expressions with arbitrary representations.
 *  Specialisations at the moment only for Symmetric LHS and Generic RHS
 *  and used to throw static assert.
 */
#include "Math/StaticCheck.h"

namespace ROOT { 

  namespace Math {

    template <class T, unsigned int D1, unsigned int D2, class R>
    class SMatrix;

    template <class A, class T, unsigned int D1, unsigned int D2, class R>
    class Expr;

    //=========================================================================
    template <class T, 
              unsigned int D1, unsigned int D2, 
              class A, class R1, class R2>

    struct Assign
    {
      static void evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
      {
        //for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep[i] = rhs.apply(i);
	unsigned int l = 0; 
        for(unsigned int i=0; i<D1; ++i) 
	  for(unsigned int j=0; j<D2; ++j) { 
	    lhs.fRep[l] = rhs(i,j);
	    l++;
	  }
      }
    };
    
    template <class T, unsigned int D1, unsigned int D2, class A>
    struct Assign<T, D1, D2, A, MatRepSym<T,D1>, MatRepStd<T,D1,D2> > 
    {
      static void evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
                           const Expr<A,T,D1,D2,MatRepStd<T,D1,D2> >& rhs) 
      {
        STATIC_CHECK(0==1, Cannot_assign_general_to_symmetric_matrix);
      }
      
    }; // struct Assign

    //=========================================================================
    template <class T, unsigned int D1, unsigned int D2, class A,
              class R1, class R2>
    struct PlusEquals
    {
      static void evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
      {
        //for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep.Array()[i] += rhs.apply(i);
	unsigned int l = 0; 
        for(unsigned int i=0; i<D1; ++i) 
	  for(unsigned int j=0; j<D2; ++j) { 
	    lhs.fRep[l] += rhs(i,j);
	    l++;
	  }
      }
    };
    
    template <class T, unsigned int D1, unsigned int D2, class A>
    struct PlusEquals<T, D1, D2, A, MatRepSym<T,D1>, MatRepStd<T,D1,D2> > 
    {
      static void evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
                           const Expr<A,T,D1,D2,MatRepStd<T,D1,D2> >& rhs) 
      {        
        STATIC_CHECK(0==1, Cannot_plusEqual_general_to_symmetric_matrix);
      }
    }; // struct PlusEquals

    //=========================================================================
    template <class T, unsigned int D1, unsigned int D2, class A,
              class R1, class R2>
    struct MinusEquals
    {
      static void evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
      {
        //for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep.Array()[i] -= rhs.apply(i);
	unsigned int l = 0; 
        for(unsigned int i=0; i<D1; ++i) 
	  for(unsigned int j=0; j<D2; ++j) { 
	    lhs.fRep[l] -= rhs(i,j);
	    l++;
	  }
      }
    };
    
    template <class T, unsigned int D1, unsigned int D2, class A>
    struct MinusEquals<T, D1, D2, A, MatRepSym<T,D1>, MatRepStd<T,D1,D2> > 
    {
      static void evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
                           const Expr<A,T,D1,D2,MatRepStd<T,D1,D2> >& rhs) 
      {        
        STATIC_CHECK(0==1, Cannot_minusEqual_general_to_symmetric_matrix);
      }
    }; // struct MinusEquals


  }  // namespace Math
  
}  // namespace ROOT

#endif // MATH_HELPEROPS_H
