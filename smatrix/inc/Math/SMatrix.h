// @(#)root/smatrix:$Name:  $:$Id: SMatrix.h,v 1.7 2005/12/10 21:40:25 moneta Exp $
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


// expression engine

#include "Math/Expression.h"


namespace ROOT {

  namespace Math {

    template <class T, unsigned int D> class SVector;




/** SMatrix.
    A generic fixed size n x m Matrix class.q

    @memo SMatrix
    @author T. Glebe
*/
//==============================================================================
// SMatrix: column-wise storage
//==============================================================================
template <class T, unsigned int D1, unsigned int D2 = D1>
class SMatrix {
public:
  /** @name --- Typedefs --- */
  ///
  typedef T  value_type;

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
  SMatrix(const SMatrix<T,D1,D2>& rhs);
  ///
  template <class A>
  SMatrix(const Expr<A,T,D1,D2>& rhs);

  // new constructs using STL iterator interface
  /**
   * Constructor with STL iterator interface. The data will be copied into the matrix
   * Size of the matrix must match size of the iterators
   */
  template<class InputIterator>
  SMatrix(InputIterator begin, InputIterator end);

  /**
   * Constructor with STL iterator interface. The data will be copied into the matrix
   * In this case the value passed size must be equal to the matrix size (D1*D2)
   */
  template<class InputIterator>
  SMatrix(InputIterator begin, unsigned int size);

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
  SMatrix<T,D1,D2>& operator=(const T& rhs);
#endif

  /**
      construct a symmetric matrix from a SVector containing the upper(lower)
      part of a triangular matrix
  */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SMatrix(const SVector<T, D1*(D2+1)/2> & v, bool lower = false );
#else
  template<unsigned int N>
  SMatrix(const SVector<T,N> & v, bool lower = false );
#endif

  ///
  template <class A>
  SMatrix<T,D1,D2>& operator=(const Expr<A,T,D1,D2>& rhs);


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
  /// access the parse tree
  T apply(unsigned int i) const;
  /// return read-only pointer to internal array
  const T* Array() const;
  /// return pointer to internal array
  T* Array();

  // STL interface

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
  bool operator==(const SMatrix<T,D1,D2>& rhs) const;
  /// element wise comparison
  bool operator!=(const SMatrix<T,D1,D2>& rhs) const;
  /// element wise comparison
  template <class A>
  bool operator==(const Expr<A,T,D1,D2>& rhs) const;
  /// element wise comparison
  template <class A>
  bool operator!=(const Expr<A,T,D1,D2>& rhs) const;

  /// element wise comparison
  bool operator>(const T& rhs) const;
  /// element wise comparison
  bool operator<(const T& rhs) const;
  /// element wise comparison
  bool operator>(const SMatrix<T,D1,D2>& rhs) const;
  /// element wise comparison
  bool operator<(const SMatrix<T,D1,D2>& rhs) const;
  /// element wise comparison
  template <class A>
  bool operator>(const Expr<A,T,D1,D2>& rhs) const;
  /// element wise comparison
  template <class A>
  bool operator<(const Expr<A,T,D1,D2>& rhs) const;

  /// read-only access
  const T& operator()(unsigned int i, unsigned int j) const;
  /// read/write access
  T& operator()(unsigned int i, unsigned int j);

  ///
  SMatrix<T,D1,D2>& operator+=(const SMatrix<T,D1,D2>& rhs);

  SMatrix<T,D1,D2>& operator-=(const SMatrix<T,D1,D2>& rhs);

  SMatrix<T,D1,D2>& operator*=(const SMatrix<T,D1,D2>& rhs);

  SMatrix<T,D1,D2>& operator/=(const SMatrix<T,D1,D2>& rhs);


#ifndef __CINT__
  ///
  template <class A>
  SMatrix<T,D1,D2>& operator+=(const Expr<A,T,D1,D2>& rhs);
  ///
  ///
  template <class A>
  SMatrix<T,D1,D2>& operator-=(const Expr<A,T,D1,D2>& rhs);
  ///
  ///
  template <class A>
  SMatrix<T,D1,D2>& operator*=(const Expr<A,T,D1,D2>& rhs);
  ///
  ///
  template <class A>
  SMatrix<T,D1,D2>& operator/=(const Expr<A,T,D1,D2>& rhs);

#endif

  /** @name --- Expert functions --- */

  /**
     invert symmetric, pos. def. Matrix via Dsinv.
     This method change the current matrix
  */
  bool Sinvert();

  /**
     invert symmetric, pos. def. Matrix via Dsinv.
     This method  returns a new matrix. In case the inversion fails
     the current matrix is returned
  */
  SMatrix<T,D1,D2>  Sinverse() const;

  /** determinant of symmetrc, pos. def. Matrix via Dsfact. \textbf{Note:} this
      will destroy the contents of the Matrix!
  */
  bool Sdet(T& det);

  /** determinant of symmetrc, pos. def. Matrix via Dsfact. \textbf{Note:}
      this method will preserve the contents of the Matrix!
  */
  bool Sdet2(T& det) const;


  /**
     invert square Matrix via Dinv.
     This method change the current matrix
  */
  bool Invert();

  /**
     invert square Matrix via Dinv.
     This method  returns a new matrix. In case the inversion fails
     the current matrix is returned
  */
  SMatrix<T,D1,D2> Inverse() const;

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
  SMatrix<T,D1,D2>& Place_in_row(const SVector<T,D>& rhs,
				 unsigned int row,
				 unsigned int col);
  /// place a vector expression in a Matrix row
  template <class A, unsigned int D>
  SMatrix<T,D1,D2>& Place_in_row(const Expr<A,T,D>& rhs,
				 unsigned int row,
				 unsigned int col);
  /// place a vector in a Matrix column
  template <unsigned int D>
  SMatrix<T,D1,D2>& Place_in_col(const SVector<T,D>& rhs,
				 unsigned int row,
				 unsigned int col);
  /// place a vector expression in a Matrix column
  template <class A, unsigned int D>
  SMatrix<T,D1,D2>& Place_in_col(const Expr<A,T,D>& rhs,
				 unsigned int row,
				 unsigned int col);
  /// place a matrix in this matrix
  template <unsigned int D3, unsigned int D4>
  SMatrix<T,D1,D2>& Place_at(const SMatrix<T,D3,D4>& rhs,
			     unsigned int row,
			     unsigned int col);
  /// place a matrix expression in this matrix
  template <class A, unsigned int D3, unsigned int D4>
  SMatrix<T,D1,D2>& Place_at(const Expr<A,T,D3,D4>& rhs,
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
     return a slice of therow as a vector starting at the colum value col0 until col0+N.
     Condition  col0+N <= D2
   */
  template <unsigned int N>
  SVector<T,N> SubRow(unsigned int therow, unsigned int col0 = 0 ) const;

  /**
     return a slice of the column as a vector starting at the row value row0 until row0+Dsub.
     Condition  row0+N <= D1
   */
  template <unsigned int N>
  SVector<T,N> SubCol(unsigned int thecol, unsigned int row0 = 0) const;

  /**
     return a submatrix with the upper left corner at the values (row0, col0) and with sizes N1, N2
     Condition  row0+N1 <= D1 && col0+N2 <=D2
   */
  template <unsigned int N1, unsigned int N2 >
  SMatrix<T,N1,N2> Sub(unsigned int row0, unsigned int col0) const;

  /**
     return diagonal elements of a matrix as a Vector.
     It works only for squared matrices D1 == D2, otherwise it will produce a compile error
   */
  SVector<T,D1> Diagonal() const;

  /**
     return the upper Triangular block of the matrices (including the diagonal) as
     a vector of sizes N = D1 * (D1 + 1)/2.
     It works only for square matrices with D1==D2, otherwise it will produce a compile error
   */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SVector<T, D1 * (D2 +1)/2> UpperBlock() const;
#else
  template<unsigned int N>
  SVector<T,N> UpperBlock() const;
#endif
  /**
     return the lower Triangular block of the matrices (including the diagonal) as
     a vector of sizes N = D1 * (D1 + 1)/2.
     It works only for square matrices with D1==D2, otherwise it will produce a compile error
   */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SVector<T, D1 * (D2 +1)/2> LowerBlock() const;
#else
  template<unsigned int N>
  SVector<T,N> LowerBlock() const;
#endif




  // submatrices

  /// Print: used by operator<<()
  std::ostream& Print(std::ostream& os) const;

private:
  T fArray[D1*D2];
}; // end of class SMatrix



//==============================================================================
// operator<<
//==============================================================================
template <class T, unsigned int D1, unsigned int D2>
inline std::ostream& operator<<(std::ostream& os, const ROOT::Math::SMatrix<T,D1,D2>& rhs) {
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
