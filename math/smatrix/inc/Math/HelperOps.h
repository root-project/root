// @(#)root/smatrix:$Id$
// Authors: J. Palacios    2006  

#ifndef ROOT_Math_HelperOps
#define ROOT_Math_HelperOps 1

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
#include <algorithm>  // required by std::copy

namespace ROOT { 

namespace Math {

   template <class T, unsigned int D1, unsigned int D2, class R>
   class SMatrix;

   template <class A, class T, unsigned int D1, unsigned int D2, class R>
   class Expr;

   //=========================================================================
   /** 
       Structure to assign from an expression based to general matrix to general matrix
   */
   template <class T, 
             unsigned int D1, unsigned int D2, 
             class A, class R1, class R2>

   struct Assign
   {
      /** 
          Evaluate the expression from general to general matrices.
          If the matrix to assign the value is in use in the expression, 
          a temporary object is created to store the value (case A = B * A)
      */
      static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
      {
         if (! rhs.IsInUse(lhs.begin() )  ) { 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<D2; ++j) { 
                  lhs.fRep[l] = rhs(i,j);
                  l++;
               }
         }
         // lhs is in use in expression, need to create a temporary with the result
         else { 
            // std::cout << "create temp  for " << typeid(rhs).name() << std::endl;
            T tmp[D1*D2]; 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<D2; ++j) { 
                  tmp[l] = rhs(i,j);
                  l++;
               }
            // copy now the temp object 
            for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep[i] = tmp[i];
         }

      }
	    
   };

   /** 
       Structure to assign from an expression based to symmetric matrix to symmetric matrix
   */
   template <class T, 
             unsigned int D1, unsigned int D2, 
             class A>

   struct Assign<T, D1, D2, A, MatRepSym<T,D1>, MatRepSym<T,D1> > 
   {
      /** 
          Evaluate the expression from  symmetric to symmetric matrices.
          If the matrix to assign the value is in use in the expression, 
          a temporary object is created to store the value (case A = B * A)
      */
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  
                           const Expr<A,T,D1,D2,MatRepSym<T,D1> >& rhs) 
      {
         if (! rhs.IsInUse(lhs.begin() ) ) { 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i)
               // storage of symmetric matrix is in lower block
               for(unsigned int j=0; j<=i; ++j) { 
                  lhs.fRep.Array()[l] = rhs(i,j);
                  l++;
               }
         }
         // create a temporary object to store result
         else { 
            T tmp[MatRepSym<T,D1>::kSize]; 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<=i; ++j) { 
                  tmp[l] = rhs(i,j);
                  l++;
               }
            // copy now the temp object 
            for(unsigned int i=0; i<MatRepSym<T,D1>::kSize; ++i) lhs.fRep.Array()[i] = tmp[i];
         }
      }
   };

    

   /** 
       Dummy Structure which flags an error to avoid assigment from expression based on a 
       general matrix to a symmetric matrix
   */
   template <class T, unsigned int D1, unsigned int D2, class A>
   struct Assign<T, D1, D2, A, MatRepSym<T,D1>, MatRepStd<T,D1,D2> > 
   {
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >&,  
                           const Expr<A,T,D1,D2,MatRepStd<T,D1,D2> >&) 
      {
         STATIC_CHECK(0==1, Cannot_assign_general_to_symmetric_matrix);
      }
      
   }; // struct Assign


   /** 
       Force Expression evaluation from general to symmetric. 
       To be used when is known (like in similarity products) that the result 
       is symmetric
       Note this is function used in the simmilarity product: no check for temporary is 
       done since in that case is not needed
   */
   struct AssignSym
   {
      /// assign a symmetric matrix from an expression
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
      /// assign the symmetric matric  from a general matrix  
      template <class T, 
                unsigned int D,
                class R>
      static void Evaluate(SMatrix<T,D,D,MatRepSym<T,D> >& lhs,  const SMatrix<T,D,D,R>& rhs) 
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
   /** 
       Evaluate the expression performing a += operation
       Need to check whether creating a temporary object with the expression result
       (like in op:  A += A * B )
   */
   template <class T, unsigned int D1, unsigned int D2, class A,
             class R1, class R2>
   struct PlusEquals
   {
      static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
      {
         if (! rhs.IsInUse(lhs.begin() )  ) { 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<D2; ++j) { 
                  lhs.fRep[l] += rhs(i,j);
                  l++;
               }
         }
         else { 
            T tmp[D1*D2]; 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<D2; ++j) { 
                  tmp[l] = rhs(i,j);
                  l++;
               }
            // += now using the temp object 
            for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep[i] += tmp[i];
         }
      }
   };

   /** 
       Specialization for symmetric matrices
       Evaluate the expression performing a += operation for symmetric matrices
       Need to have a separate functions to avoid to modify two times the off-diagonal 
       elements (i.e applying two times the expression)
       Need to check whether creating a temporary object with the expression result
       (like in op:  A += A * B )
   */
   template <class T, 
             unsigned int D1, unsigned int D2, 
             class A>
   struct PlusEquals<T, D1, D2, A, MatRepSym<T,D1>, MatRepSym<T,D1> > 
   {
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  const Expr<A,T,D1,D2, MatRepSym<T,D1> >& rhs) 
      {
         if (! rhs.IsInUse(lhs.begin() )  ) { 
            unsigned int l = 0;  // l span storage of sym matrices
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<=i; ++j) { 
                  lhs.fRep.Array()[l] += rhs(i,j);
                  l++;
               }
         }
         else { 
            T tmp[MatRepSym<T,D1>::kSize]; 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<=i; ++j) { 
                  tmp[l] = rhs(i,j);
                  l++;
               }
            // += now using the temp object 
            for(unsigned int i=0; i<MatRepSym<T,D1>::kSize; ++i) lhs.fRep.Array()[i] += tmp[i];
         }
      }
   };
   /**
      Specialization for symmetrix += general : NOT Allowed operation 
    */    
   template <class T, unsigned int D1, unsigned int D2, class A>
   struct PlusEquals<T, D1, D2, A, MatRepSym<T,D1>, MatRepStd<T,D1,D2> > 
   {
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >&,  
                           const Expr<A,T,D1,D2,MatRepStd<T,D1,D2> >&) 
      {        
         STATIC_CHECK(0==1, Cannot_plusEqual_general_to_symmetric_matrix);
      }
   }; // struct PlusEquals

   //=========================================================================

   /** 
       Evaluate the expression performing a -= operation
       Need to check whether creating a temporary object with the expression result
       (like in op:  A -= A * B )
   */
   template <class T, unsigned int D1, unsigned int D2, class A,
             class R1, class R2>
   struct MinusEquals
   {
      static void Evaluate(SMatrix<T,D1,D2,R1>& lhs,  const Expr<A,T,D1,D2,R2>& rhs) 
      {
         if (! rhs.IsInUse(lhs.begin() )  ) { 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<D2; ++j) { 
                  lhs.fRep[l] -= rhs(i,j);
                  l++;
               }
         }
         else { 
            T tmp[D1*D2]; 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<D2; ++j) { 
                  tmp[l] = rhs(i,j);
                  l++;
               }
            // -= now using the temp object 
            for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep[i] -= tmp[i];
         }
      }
   };
   /** 
       Specialization for symmetric matrices.
       Evaluate the expression performing a -= operation for symmetric matrices
       Need to have a separate functions to avoid to modify two times the off-diagonal 
       elements (i.e applying two times the expression)
       Need to check whether creating a temporary object with the expression result
       (like in op:  A -= A + B )
   */
   template <class T, 
             unsigned int D1, unsigned int D2, 
             class A>
   struct MinusEquals<T, D1, D2, A, MatRepSym<T,D1>, MatRepSym<T,D1> > 
   {
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs,  const Expr<A,T,D1,D2, MatRepSym<T,D1> >& rhs) 
      {
         if (! rhs.IsInUse(lhs.begin() )  ) { 
            unsigned int l = 0;  // l span storage of sym matrices
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<=i; ++j) { 
                  lhs.fRep.Array()[l] -= rhs(i,j);
                  l++;
               }
         }
         else { 
            T tmp[MatRepSym<T,D1>::kSize]; 
            unsigned int l = 0; 
            for(unsigned int i=0; i<D1; ++i) 
               for(unsigned int j=0; j<=i; ++j) { 
                  tmp[l] = rhs(i,j);
                  l++;
               }
            // -= now using the temp object 
            for(unsigned int i=0; i<MatRepSym<T,D1>::kSize; ++i) lhs.fRep.Array()[i] -= tmp[i];
         }
      }
   };

   /**
      Specialization for symmetrix -= general : NOT Allowed operation 
    */
   template <class T, unsigned int D1, unsigned int D2, class A>
   struct MinusEquals<T, D1, D2, A, MatRepSym<T,D1>, MatRepStd<T,D1,D2> > 
   {
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >&,  
                           const Expr<A,T,D1,D2,MatRepStd<T,D1,D2> >&) 
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
    
   /**
      Structure for assignment to a general matrix from iterator. 
      Optionally a check is done that iterator size 
      is not larger than matrix size  
    */
   template <class T, unsigned int D1, unsigned int D2, class R>  
   struct AssignItr { 
      template<class Iterator> 
      static void Evaluate(SMatrix<T,D1,D2,R>& lhs, Iterator begin, Iterator end, 
                           bool triang, bool lower,bool check=true) { 
         // require size match exactly (better)

         if (triang) { 
            Iterator itr = begin; 
            if (lower) { 
               for (unsigned int i = 0; i < D1; ++i) 
                  for (unsigned int j =0; j <= i; ++j) { 
                     // we assume iterator is well bounded within matrix
                     lhs.fRep[i*D2+j] = *itr++;
                  }
	      
            }
            else { // upper 
               for (unsigned int i = 0; i < D1; ++i) 
                  for (unsigned int j = i; j <D2; ++j) { 
                     if (itr != end)  
                        lhs.fRep[i*D2+j] = *itr++;
                     else
                        return;
                  }
	  
            }
         }
         // case of filling the full matrix
         else { 
            if (check) assert( begin + R::kSize == end);
            // copy directly the elements 
            std::copy(begin, end, lhs.fRep.Array() );
         }
      }
	
   }; // struct AssignItr

   /**
      Specialized structure for assignment to a symmetrix matrix from iterator. 
      Optionally a check is done that iterator size 
      is the same as the matrix size  
    */
   template <class T, unsigned int D1, unsigned int D2>  
   struct AssignItr<T, D1, D2, MatRepSym<T,D1> >  { 
      template<class Iterator> 
      static void Evaluate(SMatrix<T,D1,D2,MatRepSym<T,D1> >& lhs, Iterator begin, Iterator end, bool , bool lower, bool check = true) { 

         if (lower) { 
            if (check) {
               assert(begin+ static_cast<const int>( MatRepSym<T,D1>::kSize) == end);
            }
            std::copy(begin, end, lhs.fRep.Array() );
         }
         else { 
            Iterator itr = begin; 
            for (unsigned int i = 0; i < D1; ++i) 
               for (unsigned int j = i; j <D2; ++j) { 
                  if (itr != end) 
                     lhs(i,j) = *itr++;
                  else 
                     return; 
               }
         }
      }

   }; // struct AssignItr
    

}  // namespace Math
  
}  // namespace ROOT

#endif // MATH_HELPEROPS_H
