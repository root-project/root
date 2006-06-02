// @(#)root/smatrix:$Name:  $:$Id: SMatrix.h,v 1.21 2006/05/12 08:12:16 moneta Exp $
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_SMatrix
#define ROOT_Math_SMatrix
// ********************************************************************
//
// source:
//
// type:      source code
//
// created:   20. Mar 2001
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
// Description: A fixed size two dimensional Matrix class
//
// changes:
// 20 Mar 2001 (TG) creation
// 21 Mar 2001 (TG) added operators +=, -=, *=, /=
// 26 Mar 2001 (TG) place_in_row(), place_in_col() added
// 02 Apr 2001 (TG) non-const Array() added
// 03 Apr 2001 (TG) invert() added
// 07 Apr 2001 (TG) CTOR from SVertex (dyadic product) added
// 09 Apr 2001 (TG) CTOR from array added
// 11 Apr 2001 (TG) rows(), cols(), size() replaced by rows, cols, size
// 25 Mai 2001 (TG) row(), col() added
// 04 Sep 2001 (TG) moved inlined functions to .icc file
// 11 Jan 2002 (TG) added operator==(), operator!=()
// 14 Jan 2002 (TG) added more operator==(), operator!=(), operator>(), operator<()
//
// ********************************************************************
// for platform specific configurations
#include "Math/MConfig.h"

#include <iosfwd>


//doxygen tag
/**
   @defgroup SMatrix Matrix and Vector classes

   <ul>
    <li>\ref SVectorDoc
    <li>\ref SMatrixDoc
   </ul>
*/


// expression engine

#include "Math/Expression.h"
//#include "Math/MatrixRepresentations.h"
#include "Math/MatrixRepresentationsStatic.h"


namespace ROOT {

  namespace Math {


    template <class T, unsigned int D> class SVector;

    struct SMatrixIdentity { };
 

/** 
    SMatrix: a generic fixed size D1 x D2 Matrix class.
    The class is template on the scalar type and on the matrix sizes: 
    D1 = number of rows and D2 = number of columns.
    See \ref SMatrixDoc.
    
    @ingroup SMatrix
    @memo SMatrix
    @author T. Glebe
*/
//==============================================================================
// SMatrix: column-wise storage
//==============================================================================
template <class T, 
          unsigned int D1, 
          unsigned int D2 = D1, 
          class R=MatRepStd<T, D1, D2> >
class SMatrix {
public:
  /** @name --- Typedefs --- */
  ///
  typedef T  value_type;

  typedef R  rep_type;

  /** STL iterator interface. */
  typedef T*  iterator;

  /** STL const_iterator interface. */
  typedef const T*  const_iterator;



  /** @name --- Constructors --- */

  /**
      Default constructor:
   */
  SMatrix();
  /// 
  /** 
      construct from an identity matrix 
  */
  SMatrix( SMatrixIdentity ); 
  /** 
      copy constructor (from a matrix of the same representation 
  */ 
  SMatrix(const SMatrix<T,D1,D2,R>& rhs);
  /**
     construct from a matrix with different representation.
     Works only from symmetric to general and not viceversa. 
   */ 
  template <class R2>
  SMatrix(const SMatrix<T,D1,D2,R2>& rhs);

  /**
     construct from an expression. 
     In case of symmetric matrices does not work if expression is of type general 
     matrices. In case one needs to force the assignment from general to symmetric, one can use the 
     ROOT::Math::AssignSym::Evaluate function. 
   */ 
  template <class A, class R2>
  SMatrix(const Expr<A,T,D1,D2,R2>& rhs);

  // new constructs using STL iterator interface
  /**
     Constructor with STL iterator interface. The data will be copied into the matrix
     \param begin start iterator position
     \param end end iterator position
     \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators 
     \param lower if true the lower triangular part is filled 
     
     Size of the matrix must match size of the iterators if triang is false, otherwise the size of the 
     triangular block 
     In the case of symmetric matrices triang is considered always to be true (what-ever the user specifies) and 
     the size of the iterators must be equal to the size of the symmetric representation (number of independent 
     elements), N*(N+1)/2 
     
  */
  template<class InputIterator>
  SMatrix(InputIterator begin, InputIterator end, bool triang = false, bool lower = true);

  /**
    Constructor with STL iterator interface. The data will be copied into the matrix
    \param begin  start iterator position
    \param size   iterator size 
    \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators 
    \param lower if true the lower triangular part is filled 
    
    Size of the matrix must match size of the iterators if triang is false, otherwise the size of the 
    triangular block 
    In the case of symmetric matrices triang is considered always to be true (what-ever the user specifies) and 
    the size of the iterators must be equal to the size of the symmetric representation (number of independent 
    elements), N*(N+1)/2 
    
  */
  template<class InputIterator>
  SMatrix(InputIterator begin, unsigned int size, bool triang = false, bool lower = true);

  // skip this methods (they are too ambigous)
#ifdef OLD_IMPL
  /// 2nd arg: set only diagonal?
  SMatrix(const T& rhs, bool diagonal=false);
  /// constructor via dyadic product
  SMatrix(const SVector<T,D1>& rhs);
  /// constructor via dyadic product
  template <class A>
  SMatrix(const Expr<A,T,D1>& rhs);

  /** constructor via array, triag=true: array contains only upper/lower
      triangular part of a symmetric matrix, len: length of array */
  template <class T1>
  //SMatrix(const T1* a, bool triang=false, unsigned int len=D1*D2);
  // to avoid clash with TRootIOconst
  SMatrix(const T1* a, bool triang, unsigned int len=D1*D2);


  /// assign from a scalar value
  SMatrix<T,D1,D2,R>& operator=(const T& rhs);
#endif

  /**
      construct a symmetric matrix from a SVector containing the lower (upper)
      part of a triangular matrix
  */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SMatrix(const SVector<T, D1*(D2+1)/2> & v, bool lower = true );
#else
  template<unsigned int N>
  SMatrix(const SVector<T,N> & v, bool lower = true );
#endif

  ///
  template <class M>
  SMatrix<T,D1,D2,R>& operator=(const M& rhs);
  
  template <class A, class R2>
  SMatrix<T,D1,D2,R>& operator=(const Expr<A,T,D1,D2,R2>& rhs);

  /// assign from an identity
  SMatrix<T,D1,D2,R> & operator=(SMatrixIdentity ); 


#ifdef OLD_IMPL
  /// return no. of matrix rows
  static const unsigned int kRows = D1;
  /// return no. of matrix columns
  static const unsigned int kCols = D2;
  /// return no of elements: rows*columns
  static const unsigned int kSize = D1*D2;
#else
  enum {
  /// return no. of matrix rows
    kRows = D1,
  /// return no. of matrix columns
    kCols = D2,
  /// return no of elements: rows*columns
    kSize = D1*D2
  };
#endif
  /** @name --- Access functions --- */
  /** access the parse tree with the index starting from zero and following the C convention for the order in accessing 
      the matrix elements. 
   */  
  T apply(unsigned int i) const;

  /// return read-only pointer to internal array
  const T* Array() const;
  /// return pointer to internal array
  T* Array();

  /** @name --- STL-like interface --- 
      The iterators access the matrix element in the order how they are 
      stored in memory. The C (row-major) convention is used, and in the 
      case of symmetric matrices the iterator spans only the lower diagonal 
      block. For example for a symmetric 3x3 matrices the order of the 6 
      elements \f${a_0,...a_5}\f$ is: 
      \f[
       M = \left( \begin{array}{ccc} 
       a_0 & a_1 & a_3  \\ 
       a_1 & a_2  & a_4  \\
       a_3 & a_4 & a_5   \end{array} \right)
       \f]
  */

  /** STL iterator interface. */
  iterator begin();

  /** STL iterator interface. */
  iterator end();

  /** STL const_iterator interface. */
  const_iterator begin() const;

  /** STL const_iterator interface. */
  const_iterator end() const;


  /** @name --- Operators --- */
  /// element wise comparison
  bool operator==(const T& rhs) const;
  /// element wise comparison
  bool operator!=(const T& rhs) const;
  /// element wise comparison
  template <class R2>
  bool operator==(const SMatrix<T,D1,D2,R2>& rhs) const;
  /// element wise comparison
  bool operator!=(const SMatrix<T,D1,D2,R>& rhs) const;
  /// element wise comparison
  template <class A, class R2>
  bool operator==(const Expr<A,T,D1,D2,R2>& rhs) const;
  /// element wise comparison
  template <class A, class R2>
  bool operator!=(const Expr<A,T,D1,D2,R2>& rhs) const;

  /// element wise comparison
  bool operator>(const T& rhs) const;
  /// element wise comparison
  bool operator<(const T& rhs) const;
  /// element wise comparison
  template <class R2>
  bool operator>(const SMatrix<T,D1,D2,R2>& rhs) const;
  /// element wise comparison
  template <class R2>
  bool operator<(const SMatrix<T,D1,D2,R2>& rhs) const;
  /// element wise comparison
  template <class A, class R2>
  bool operator>(const Expr<A,T,D1,D2,R2>& rhs) const;
  /// element wise comparison
  template <class A, class R2>
  bool operator<(const Expr<A,T,D1,D2,R2>& rhs) const;

  /**
     read only access to matrix element, with indices starting from 0
   */ 
  const T& operator()(unsigned int i, unsigned int j) const;
    /**
     read/write access to matrix element with indices starting from 0
   */ 
  T& operator()(unsigned int i, unsigned int j);

  ///
  template <class M>
  SMatrix<T,D1,D2,R>&operator+=(const M& rhs);

  template <class M>
  SMatrix<T,D1,D2,R>& operator-=(const M& rhs);

#ifdef OLD_IMPL
  // this operations are not well defines 
  // in th eold impl they were implemented not as matrix - matrix multiplication, but as 
  //  m(i,j)*m(i,j) multiplication
  SMatrix<T,D1,D2,R>& operator*=(const SMatrix<T,D1,D2,R>& rhs);

  SMatrix<T,D1,D2,R>& operator/=(const SMatrix<T,D1,D2,R>& rhs);
#endif

#ifndef __CINT__
  ///
  template <class A, class R2>
  SMatrix<T,D1,D2,R>& operator+=(const Expr<A,T,D1,D2,R2>& rhs);
  ///
  ///
  template <class A, class R2>
  SMatrix<T,D1,D2,R>& operator-=(const Expr<A,T,D1,D2,R2>& rhs);
  ///
  ///
#ifdef OLD_IMPL
  template <class A, class R2>
  SMatrix<T,D1,D2,R>& operator*=(const Expr<A,T,D1,D2,R2>& rhs);
  ///
  ///
  template <class A, class R2>
  SMatrix<T,D1,D2,R>& operator/=(const Expr<A,T,D1,D2,R2>& rhs);
#endif

#endif

#ifdef OLD_IMPL
  /** @name --- Expert functions --- */

  /**
     invert symmetric, pos. def. Matrix via Dsinv.
     This method change the current matrix
  */
  bool Sinvert();

  /**
     invert symmetric, pos. def. Matrix via Dsinv.
     This method  returns a new matrix. In case the inversion fails
     the current matrix is returned. 
      Return ifail = 0 when successfull 
  */
  SMatrix<T,D1,D2,R>  Sinverse(int & ifail ) const;

  /** determinant of symmetrc, pos. def. Matrix via Dsfact. \textbf{Note:} this
      will destroy the contents of the Matrix!
  */
  bool Sdet(T& det);

  /** determinant of symmetrc, pos. def. Matrix via Dsfact. \textbf{Note:}
      this method will preserve the contents of the Matrix!
  */
  bool Sdet2(T& det) const;
#endif


  /**
     invert square Matrix ( this method change the current matrix)
     The method used for general square matrices is the LU factorization taken from Dinv routine 
     from the CERNLIB (written in C++ from CLHEP authors)
     In case of symmetric matrices Bunch-Kaufman diagonal pivoting method is used
     (The implementation is the one written by the CLHEP authors)
  */
  bool Invert();

  /**
     invert a square Matrix and  returns a new matrix. In case the inversion fails
     the current matrix is returned. 
     Return ifail = 0 when successfull. 
     See ROOT::Math::SMatrix::Invert for the inversion algorithm
  */
  SMatrix<T,D1,D2,R> Inverse(int & ifail ) const;

  /**
      determinant of square Matrix via Dfact. \textbf{Note:} this will destroy
      the contents of the Matrix!
  */
  bool Det(T& det);

  /**
      determinant of square Matrix via Dfact. \textbf{Note:} this will preserve
      the content of the Matrix!
  */
  bool Det2(T& det) const;



  /// place a vector in a Matrix row
  template <unsigned int D>
  SMatrix<T,D1,D2,R>& Place_in_row(const SVector<T,D>& rhs,
				 unsigned int row,
				 unsigned int col);
  /// place a vector expression in a Matrix row 
  template <class A, unsigned int D>
SMatrix<T,D1,D2,R>& Place_in_row(const VecExpr<A,T,D>& rhs,
				 unsigned int row,
				 unsigned int col);
  /// place a vector in a Matrix column
  template <unsigned int D>
SMatrix<T,D1,D2,R>& Place_in_col(const SVector<T,D>& rhs,
				 unsigned int row,
				 unsigned int col);
  /// place a vector expression in a Matrix column
  template <class A, unsigned int D>
SMatrix<T,D1,D2,R>& Place_in_col(const VecExpr<A,T,D>& rhs,
				 unsigned int row,
				 unsigned int col);
  /// place a matrix in this matrix
  template <unsigned int D3, unsigned int D4, class R2>
  SMatrix<T,D1,D2,R>& Place_at(const SMatrix<T,D3,D4,R2>& rhs,
			     unsigned int row,
			     unsigned int col);
  /// place a matrix expression in this matrix
  template <class A, unsigned int D3, unsigned int D4, class R2>
  SMatrix<T,D1,D2,R>& Place_at(const Expr<A,T,D3,D4,R2>& rhs,
			     unsigned int row,
			     unsigned int col);

  /**
      return a full Matrix row as a vector (copy the content in a new vector)
  */
  SVector<T,D2> Row(unsigned int therow) const;

  /**
      return a full Matrix column as a vector (copy the content in a new vector)
  */
  SVector<T,D1> Col(unsigned int thecol) const;

  /**
     return a slice of therow as a vector starting at the colum value col0 until col0+N, 
     where N is the size of the vector (SubVector::kSize )
     Condition  col0+N <= D2 
   */
  template <class SubVector>
  SubVector SubRow(unsigned int therow, unsigned int col0 = 0 ) const;

  /**
     return a slice of the column as a vector starting at the row value row0 until row0+Dsub.
     where N is the size of the vector (SubVector::kSize )
     Condition  row0+N <= D1
   */
  template <class SubVector>
  SubVector SubCol(unsigned int thecol, unsigned int row0 = 0) const;

  /**
     return a submatrix with the upper left corner at the values (row0, col0) and with sizes N1, N2
     where N1 and N2 are the dimension of the sub-matrix (SubMatrix::kRows and SubMatrix::kCols )
     Condition  row0+N1 <= D1 && col0+N2 <=D2
   */
  template <class SubMatrix >
  SubMatrix Sub(unsigned int row0, unsigned int col0) const;

  /**
     return diagonal elements of a matrix as a Vector.
     It works only for squared matrices D1 == D2, otherwise it will produce a compile error
   */
  SVector<T,D1> Diagonal() const;

  /**
     Set the diagonal elements from a Vector
     Require that vector implements ::kSize since a check (statically) is done on 
     diagonal size == vector size
   */
  template <class Vector> 
  void SetDiagonal(const Vector & v);

  /**
     return the upper Triangular block of the matrices (including the diagonal) as
     a vector of sizes N = D1 * (D1 + 1)/2.
     It works only for square matrices with D1==D2, otherwise it will produce a compile error
   */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SVector<T, D1 * (D2 +1)/2> UpperBlock() const;
#else
  template<class SubVector>
  SubVector UpperBlock() const;
#endif
  /**
     return the lower Triangular block of the matrices (including the diagonal) as
     a vector of sizes N = D1 * (D1 + 1)/2.
     It works only for square matrices with D1==D2, otherwise it will produce a compile error
   */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SVector<T, D1 * (D2 +1)/2> LowerBlock() const;
#else
  template<class SubVector>
  SubVector LowerBlock() const;
#endif


  /** 
      check if matrix is sharing same memory location.
      This functionis used by the expression to avoid the alias problem when 
      evaluating them. In case  matrix is in use, a temporary object is automatically 
      created evaluating the expression. Then the correct result is obtained for operations 
      like  A = B * A
   */
  bool IsInUse(const T* p) const; 

  // submatrices

  /// Print: used by operator<<()
  std::ostream& Print(std::ostream& os) const;

public:
  //  T fArray[D1*D2];
  R fRep;
  
}; // end of class SMatrix



//==============================================================================
// operator<<
//==============================================================================
template <class T, unsigned int D1, unsigned int D2, class R>
inline std::ostream& operator<<(std::ostream& os, const ROOT::Math::SMatrix<T,D1,D2,R>& rhs) {
  return rhs.Print(os);
}


  }  // namespace Math

}  // namespace ROOT






#ifndef __CINT__

#include "Math/SMatrix.icc"
// include Matrix-Vector multiplication
#include "Math/MatrixFunctions.h"

#endif //__CINT__

#endif  /* ROOT_Math_SMatrix  */
