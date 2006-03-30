// @(#)root/smatrix:$Name:  $:$Id: HelperOps.h,v 1.5 2006/03/30 08:21:28 moneta Exp $
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


    /** Structure to deal when a submatrix is placed in a matrix.  
	We have different cases according to the matrix representation
    */
    template <class T, unsigned int D1, unsigned int D2,   
              unsigned int D3, unsigned int D4, 
	      class R1, class R2>
    struct PlaceMatrix
    {
      static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const SMatrix<T,D3,D4,R2>& rhs, 
			   unsigned int row, unsigned int col) {

	assert(row+D3 <= D1 && col+D4 <= D2);
	const unsigned int offset = row*D2+col;

	for(unsigned int i=0; i<D3*D4; ++i) {
	  lhs.fRep[offset+(i/D4)*D2+i%D4] = rhs.apply(i);
	}

      }
    }; // struct PlaceMatrix

    template <class T, unsigned int D1, unsigned int D2,   
              unsigned int D3, unsigned int D4, 
	      class A, class R1, class R2>
    struct PlaceExpr { 
    static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D3,D4,R2>& rhs, 
			 unsigned int row, unsigned int col) { 

	assert(row+D3 <= D1 && col+D4 <= D2);
	const unsigned int offset = row*D2+col;

	for(unsigned int i=0; i<D3*D4; ++i) {
	  lhs.fRep[offset+(i/D4)*D2+i%D4] = rhs.apply(i);
	}
      }
  };  // struct PlaceExpr 

  // specialization for general matrix in symmetric matrices
  template <class T, unsigned int D1, unsigned int D2, 
	    unsigned int D3, unsigned int D4 >
  struct PlaceMatrix<T, D1, D2, D3, D4, MatRepSym<T,D1>, MatRepStd<T,D3,D4> > { 
    static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& ,  
			 const SMatrix<T,D3,D4,MatRepStd<T,D3,D4> >& , 
			 unsigned int , unsigned int ) 
    {        
      STATIC_CHECK(0==1, Cannot_Place_Matrix_general_in_symmetric_matrix);
    }
  }; // struct PlaceMatrix

  // specialization for general expression in symmetric matrices
  template <class T, unsigned int D1, unsigned int D2, 
	    unsigned int D3, unsigned int D4, class A >
  struct PlaceExpr<T, D1, D2, D3, D4, A, MatRepSym<T,D1>, MatRepStd<T,D3,D4> > { 
    static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& ,  
			 const Expr<A,T,D3,D4,MatRepStd<T,D3,D4> >& , 
			 unsigned int , unsigned int ) 
    {        
      STATIC_CHECK(0==1, Cannot_Place_Matrix_general_in_symmetric_matrix);
    }
  }; // struct PlaceExpr

  // specialization for symmetric matrix in symmetric matrices

  template <class T, unsigned int D1, unsigned int D2, 
	    unsigned int D3, unsigned int D4 >
  struct PlaceMatrix<T, D1, D2, D3, D4, MatRepSym<T,D1>, MatRepSym<T,D3> > { 
    static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
			 const SMatrix<T,D3,D4,MatRepSym<T,D3> >& rhs, 
			 unsigned int row, unsigned int col ) 
    {        
      // can work only if placed on the diagonal
      assert(row == col); 

      for(unsigned int i=0; i<D3; ++i) {
	for(unsigned int j=0; j<=i; ++j) 
	  lhs.fRep(row+i,col+j) = rhs(i,j);
      }	  
    }
  }; // struct PlaceMatrix

  // specialization for symmetric expression in symmetric matrices
  template <class T, unsigned int D1, unsigned int D2, 
	    unsigned int D3, unsigned int D4, class A >
  struct PlaceExpr<T, D1, D2, D3, D4, A, MatRepSym<T,D1>, MatRepSym<T,D3> > { 
    static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
			 const Expr<A,T,D3,D4,MatRepSym<T,D3> >& rhs, 
			 unsigned int row, unsigned int col ) 
    {        
      // can work only if placed on the diagonal
      assert(row == col); 

      for(unsigned int i=0; i<D3; ++i) {
	for(unsigned int j=0; j<=i; ++j) 
	  lhs.fRep(row+i,col+j) = rhs(i,j);
      }
    }
  }; // struct PlaceExpr



    /** Structure for getting sub matrices 
	We have different cases according to the matrix representations
    */
    template <class T, unsigned int D1, unsigned int D2,   
              unsigned int D3, unsigned int D4, 
	      class R1, class R2>
    struct RetrieveMatrix
    {
      static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const SMatrix<T,D3,D4,R2>& rhs, 
			   unsigned int row, unsigned int col) {
	STATIC_CHECK( D1 <= D3,Smatrix_nrows_too_small); 
	STATIC_CHECK( D2 <= D4,Smatrix_ncols_too_small); 

	assert(row + D1 <= D3);
	assert(col + D2 <= D4);

	for(unsigned int i=0; i<D1; ++i) { 
	  for(unsigned int j=0; j<D2; ++j) 
	    lhs(i,j) = rhs(i+row,j+col);
	}
      }
    };   // struct RetrieveMatrix

  // specialization for getting symmetric matrices from  general matrices (MUST fail)
  template <class T, unsigned int D1, unsigned int D2, 
	    unsigned int D3, unsigned int D4 >
  struct RetrieveMatrix<T, D1, D2, D3, D4, MatRepSym<T,D1>, MatRepStd<T,D3,D4> > { 
    static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& ,  
			 const SMatrix<T,D3,D4,MatRepStd<T,D3,D4> >& , 
			 unsigned int , unsigned int ) 
    {        
      STATIC_CHECK(0==1, Cannot_Sub_Matrix_symmetric_in_general_matrix);
    }
  }; // struct RetrieveMatrix

  // specialization for getting symmetric matrices from  symmetric matrices (OK if row == col)
  template <class T, unsigned int D1, unsigned int D2, 
	    unsigned int D3, unsigned int D4 >
  struct RetrieveMatrix<T, D1, D2, D3, D4, MatRepSym<T,D1>, MatRepSym<T,D3> > { 
    static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
			 const SMatrix<T,D3,D4,MatRepSym<T,D3> >& rhs, 
			 unsigned int row, unsigned int col ) 
    {        
      STATIC_CHECK(  D1 <= D3,Smatrix_dimension1_too_small); 
      // can work only if placed on the diagonal
      assert(row == col); 
      assert(row + D1 <= D3);

      for(unsigned int i=0; i<D1; ++i) {
	for(unsigned int j=0; j<=i; ++j) 
	  lhs(i,j) = rhs(i+row,j+col );	
      }
    }

  }; // struct RetrieveMatrix
    

  }  // namespace Math
  
}  // namespace ROOT

#endif // MATH_HELPEROPS_H
