// @(#)root/smatrix:$Name:  $:$Id: HelperOps.h,v 1.4 2006/03/20 17:11:44 moneta Exp $
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
    // for generic matrices 
    template <class T, 
              unsigned int D1, unsigned int D2, 
              class A, class R1, class R2>

    struct Assign
    {
      static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
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
    // specialization in case of symmetric expression to symmetric matrices :  
    template <class T, 
              unsigned int D1, unsigned int D2, 
              class A>

    struct Assign<T, D1, D2, A, MatRepSym<T,D1>, MatRepSym<T,D1> > 
    {
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
                           const Expr<A,T,D1,D2,MatRepSym<T,D1> >& rhs) 
      {
	unsigned int l = 0; 
        for(unsigned int i=0; i<D1; ++i)
	  // storage of symmetric matrix is in lower block
	  for(unsigned int j=0; j<=i; ++j) { 
	    lhs.fRep.Array()[l] = rhs(i,j);
	    l++;
	  }
      }
    };

    
    // case of general to symmetric matrices (flag an error !) 
    template <class T, unsigned int D1, unsigned int D2, class A>
    struct Assign<T, D1, D2, A, MatRepSym<T,D1>, MatRepStd<T,D1,D2> > 
    {
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
                           const Expr<A,T,D1,D2,MatRepStd<T,D1,D2> >& rhs) 
      {
        STATIC_CHECK(0==1, Cannot_assign_general_to_symmetric_matrix);
      }
      
    }; // struct Assign

    // have a dedicated structure for symmetric matrices to be used when we are sure the resulting expression is 
    // symmetric (like in a smilarity operation). This cannot be used in the opersator= of the SMatrix class
    struct AssignSym
    {
      // assign a symmetric matrix from an expression
    template <class T, 
              unsigned int D,
              class A, 
	      class R>
      static void Evaluate(SMatrix<T,D,D,MatRepSym<T,D> >& lhs,  const Expr<A,T,D,D,R>& rhs) 
      {
        //for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep[i] = rhs.apply(i);
	unsigned int l = 0; 
        for(unsigned int i=0; i<D; ++i)
	  // storage of symmetric matrix is in lower block
	  for(unsigned int j=0; j<=i; ++j) { 
	    lhs.fRep.Array()[l] = rhs(i,j);
	    l++;
	  }
      }


    }; // struct AssignSym 
    

    //=========================================================================
    template <class T, unsigned int D1, unsigned int D2, class A,
              class R1, class R2>
    struct PlusEquals
    {
      static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
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
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
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
      static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
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
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
                           const Expr<A,T,D1,D2,MatRepStd<T,D1,D2> >& rhs) 
      {        
        STATIC_CHECK(0==1, Cannot_minusEqual_general_to_symmetric_matrix);
      }
    }; // struct MinusEquals



  }  // namespace Math
  
}  // namespace ROOT

#endif // MATH_HELPEROPS_H
