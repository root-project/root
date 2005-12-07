// @(#)root/smatrix:$Name:  $:$Id: MatrixFunctions.h,v 1.2 2005/12/05 16:33:47 moneta Exp $
// Authors: T. Glebe, L. Moneta    2005  

#ifndef ROOT_Math_MatrixFunctions
#define ROOT_Math_MatrixFunctions
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
// Description: Functions/Operators special to Matrix
//
// changes:
// 20 Mar 2001 (TG) creation
// 20 Mar 2001 (TG) Added Matrix * Vector multiplication
// 21 Mar 2001 (TG) Added transpose, product
// 11 Apr 2001 (TG) transpose() speed improvment by removing rows(), cols()
//                  through static members of Matrix and Expr class
//
// ********************************************************************



namespace ROOT { 

  namespace Math { 



#ifdef XXX
//==============================================================================
// SMatrix * SVector
//==============================================================================
template <class T, unsigned int D1, unsigned int D2>
SVector<T,D1> operator*(const SMatrix<T,D1,D2>& rhs, const SVector<T,D2>& lhs)
{
  SVector<T,D1> tmp;
  for(unsigned int i=0; i<D1; ++i) {
    const unsigned int rpos = i*D2;
    for(unsigned int j=0; j<D2; ++j) {
      tmp[i] += rhs.apply(rpos+j) * lhs.apply(j);
    }
  }
  return tmp;
}
#endif


//==============================================================================
// meta_row_dot
//==============================================================================
template <unsigned int I>
struct meta_row_dot {
  template <class A, class B>
  static inline typename A::value_type f(const A& lhs, const B& rhs,
					 const unsigned int offset) {
    return lhs.apply(offset+I) * rhs.apply(I) + meta_row_dot<I-1>::f(lhs,rhs,offset);
  }
};


//==============================================================================
// meta_row_dot<0>
//==============================================================================
template <>
struct meta_row_dot<0> {
  template <class A, class B>
  static inline typename A::value_type f(const A& lhs, const B& rhs,
					 const unsigned int offset) {
    return lhs.apply(offset) * rhs.apply(0);
  }
};

//==============================================================================
// VectorMatrixRowOp
//==============================================================================
template <class Matrix, class Vector, unsigned int D2>
class VectorMatrixRowOp {
public:
  ///
  VectorMatrixRowOp(const Matrix& lhs, const Vector& rhs) :
    lhs_(lhs), rhs_(rhs) {}

  ///
  ~VectorMatrixRowOp() {}

  /// calc $\sum_{j} a_{ij} * v_j$
  inline typename Matrix::value_type apply(unsigned int i) const {
    return meta_row_dot<D2-1>::f(lhs_, rhs_, i*D2);
  }

protected:
  const Matrix& lhs_;
  const Vector& rhs_;
};


//==============================================================================
// meta_col_dot
//==============================================================================
template <unsigned int I>
struct meta_col_dot {
  template <class Matrix, class Vector>
  static inline typename Matrix::value_type f(const Matrix& lhs, const Vector& rhs,
					      const unsigned int offset) {
    return lhs.apply(Matrix::kCols*I+offset) * rhs.apply(I) + 
           meta_col_dot<I-1>::f(lhs,rhs,offset);
  }
};


//==============================================================================
// meta_col_dot<0>
//==============================================================================
template <>
struct meta_col_dot<0> {
  template <class Matrix, class Vector>
  static inline typename Matrix::value_type f(const Matrix& lhs, const Vector& rhs,
					      const unsigned int offset) {
    return lhs.apply(offset) * rhs.apply(0);
  }
};

//==============================================================================
// VectorMatrixColOp
//==============================================================================
template <class Vector, class Matrix, unsigned int D1>
class VectorMatrixColOp {
public:
  ///
  VectorMatrixColOp(const Vector& lhs, const Matrix& rhs) :
    lhs_(lhs), rhs_(rhs) {}

  ///
  ~VectorMatrixColOp() {}

  /// calc $\sum_{j} a_{ij} * v_j$
  inline typename Matrix::value_type apply(unsigned int i) const {
    return meta_col_dot<D1-1>::f(rhs_, lhs_, i);
  }

protected:
  const Vector&    lhs_;
  const Matrix&    rhs_;
};

//==============================================================================
// operator*: SMatrix * SVector
//==============================================================================
template <class T, unsigned int D1, unsigned int D2>
inline Expr<VectorMatrixRowOp<SMatrix<T,D1,D2>,SVector<T,D2>, D2>, T, D1>
 operator*(const SMatrix<T,D1,D2>& lhs, const SVector<T,D2>& rhs) {

  typedef VectorMatrixRowOp<SMatrix<T,D1,D2>,SVector<T,D2>, D2> VMOp;
  return Expr<VMOp, T, D1>(VMOp(lhs,rhs));
}

//==============================================================================
// operator*: SMatrix * Expr<A,T,D2>
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D2>
inline Expr<VectorMatrixRowOp<SMatrix<T,D1,D2>, Expr<A,T,D2>, D2>, T, D1>
 operator*(const SMatrix<T,D1,D2>& lhs, const Expr<A,T,D2>& rhs) {

  typedef VectorMatrixRowOp<SMatrix<T,D1,D2>,Expr<A,T,D2>, D2> VMOp;
  return Expr<VMOp, T, D1>(VMOp(lhs,rhs));
}

//==============================================================================
// operator*: Expr<A,T,D1,D2> * SVector
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D2>
inline Expr<VectorMatrixRowOp<Expr<A,T,D1,D2>, SVector<T,D2>, D2>, T, D1>
 operator*(const Expr<A,T,D1,D2>& lhs, const SVector<T,D2>& rhs) {

  typedef VectorMatrixRowOp<Expr<A,T,D1,D2>,SVector<T,D2>, D2> VMOp;
  return Expr<VMOp, T, D1>(VMOp(lhs,rhs));
}

//==============================================================================
// operator*: Expr<A,T,D1,D2> * Expr<B,T,D2>
//==============================================================================
template <class A, class B, class T, unsigned int D1, unsigned int D2>
inline Expr<VectorMatrixRowOp<Expr<A,T,D1,D2>, Expr<B,T,D2>, D2>, T, D1>
 operator*(const Expr<A,T,D1,D2>& lhs, const Expr<B,T,D2>& rhs) {

  typedef VectorMatrixRowOp<Expr<A,T,D1,D2>,Expr<B,T,D2>, D2> VMOp;
  return Expr<VMOp, T, D1>(VMOp(lhs,rhs));
}

//==============================================================================
// operator*: SVector * SMatrix
//==============================================================================
template <class T, unsigned int D1, unsigned int D2>
inline Expr<VectorMatrixColOp<SVector<T,D1>, SMatrix<T,D1,D2>, D1>, T, D2>
 operator*(const SVector<T,D1>& lhs, const SMatrix<T,D1,D2>& rhs) {

  typedef VectorMatrixColOp<SVector<T,D1>, SMatrix<T,D1,D2>, D1> VMOp;
  return Expr<VMOp, T, D2>(VMOp(lhs,rhs));
}

//==============================================================================
// operator*: SVector * Expr<A,T,D1,D2>
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D2>
inline Expr<VectorMatrixColOp<SVector<T,D1>, Expr<A,T,D1,D2>, D1>, T, D2>
 operator*(const SVector<T,D1>& lhs, const Expr<A,T,D1,D2>& rhs) {

  typedef VectorMatrixColOp<SVector<T,D1>, Expr<A,T,D1,D2>, D1> VMOp;
  return Expr<VMOp, T, D2>(VMOp(lhs,rhs));
}

//==============================================================================
// operator*: Expr<A,T,D1> * SMatrix
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D2>
inline Expr<VectorMatrixColOp<Expr<A,T,D1>, SMatrix<T,D1,D2>, D1>, T, D2>
 operator*(const Expr<A,T,D1>& lhs, const SMatrix<T,D1,D2>& rhs) {

  typedef VectorMatrixColOp<Expr<A,T,D1>, SMatrix<T,D1,D2>, D1> VMOp;
  return Expr<VMOp, T, D2>(VMOp(lhs,rhs));
}

//==============================================================================
// operator*: Expr<A,T,D1> * Expr<B,T,D1,D2>
//==============================================================================
template <class A, class B, class T, unsigned int D1, unsigned int D2>
inline Expr<VectorMatrixColOp<Expr<A,T,D1>, Expr<B,T,D1,D2>, D1>, T, D2>
 operator*(const Expr<A,T,D1>& lhs, const Expr<B,T,D1,D2>& rhs) {

  typedef VectorMatrixColOp<Expr<A,T,D1>, Expr<B,T,D1,D2>, D1> VMOp;
  return Expr<VMOp, T, D2>(VMOp(lhs,rhs));
}

//==============================================================================
// meta_matrix_dot
//==============================================================================
template <unsigned int I>
struct meta_matrix_dot {
  template <class MatrixA, class MatrixB>
  static inline typename MatrixA::value_type f(const MatrixA& lhs, const MatrixB& rhs,
					       const unsigned int offset) {
    return lhs.apply(offset/MatrixB::kCols*MatrixA::kCols + I) *
           rhs.apply(MatrixB::kCols*I + offset%MatrixB::kCols) + 
           meta_matrix_dot<I-1>::f(lhs,rhs,offset);
  }
};


//==============================================================================
// meta_matrix_dot<0>
//==============================================================================
template <>
struct meta_matrix_dot<0> {
  template <class MatrixA, class MatrixB>
  static inline typename MatrixA::value_type f(const MatrixA& lhs, const MatrixB& rhs,
					       const unsigned int offset) {
    return lhs.apply(offset/MatrixB::kCols*MatrixA::kCols) *
           rhs.apply(offset%MatrixB::kCols);
  }
};

//==============================================================================
// MatrixMulOp
//==============================================================================
template <class MatrixA, class MatrixB, class T, unsigned int D>
class MatrixMulOp {
public:
  ///
  MatrixMulOp(const MatrixA& lhs, const MatrixB& rhs) :
    lhs_(lhs), rhs_(rhs) {}

  ///
  ~MatrixMulOp() {}

  /// calc $\sum_{j} a_{ik} * b_{kj}$
  inline T apply(unsigned int i) const {
    return meta_matrix_dot<D-1>::f(lhs_, rhs_, i);
  }

protected:
  const MatrixA&    lhs_;
  const MatrixB&    rhs_;
};


//==============================================================================
// operator* (SMatrix * SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D1, unsigned int D, unsigned int D2>
inline Expr<MatrixMulOp<SMatrix<T,D1,D>, SMatrix<T,D,D2>,T,D>, T, D1, D2>
 operator*(const SMatrix<T,D1,D>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef MatrixMulOp<SMatrix<T,D1,D>, SMatrix<T,D,D2>, T,D> MatMulOp;

  return Expr<MatMulOp,T,D1,D2>(MatMulOp(lhs,rhs));
}

//==============================================================================
// operator* (SMatrix * Expr, binary)
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D, unsigned int D2>
inline Expr<MatrixMulOp<SMatrix<T,D1,D>, Expr<A,T,D,D2>,T,D>, T, D1, D2>
 operator*(const SMatrix<T,D1,D>& lhs, const Expr<A,T,D,D2>& rhs) {
  typedef MatrixMulOp<SMatrix<T,D1,D>, Expr<A,T,D,D2>,T,D> MatMulOp;

  return Expr<MatMulOp,T,D1,D2>(MatMulOp(lhs,rhs));
}

//==============================================================================
// operator* (Expr * SMatrix, binary)
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D, unsigned int D2>
inline Expr<MatrixMulOp<Expr<A,T,D1,D>, SMatrix<T,D,D2>,T,D>, T, D1, D2>
 operator*(const Expr<A,T,D1,D>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef MatrixMulOp<Expr<A,T,D1,D>, SMatrix<T,D,D2>,T,D> MatMulOp;

  return Expr<MatMulOp,T,D1,D2>(MatMulOp(lhs,rhs));
}

//==============================================================================
// operator* (Expr * Expr, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D1, unsigned int D, unsigned int D2>
inline Expr<MatrixMulOp<Expr<A,T,D1,D>, Expr<B,T,D,D2>,T,D>, T, D1, D2>
 operator*(const Expr<A,T,D1,D>& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef MatrixMulOp<Expr<A,T,D1,D>, Expr<B,T,D,D2>, T,D> MatMulOp;

  return Expr<MatMulOp,T,D1,D2>(MatMulOp(lhs,rhs));
}


#ifdef XXX
//==============================================================================
// MatrixMulOp
//==============================================================================
template <class MatrixA, class MatrixB, unsigned int D>
class MatrixMulOp {
public:
  ///
  MatrixMulOp(const MatrixA& lhs, const MatrixB& rhs) :
    lhs_(lhs), rhs_(rhs) {}

  ///
  ~MatrixMulOp() {}

  /// calc $\sum_{j} a_{ik} * b_{kj}$
  inline typename MatrixA::value_type apply(unsigned int i) const {
    return meta_matrix_dot<D-1>::f(lhs_, rhs_, i);
  }

protected:
  const MatrixA&    lhs_;
  const MatrixB&    rhs_;
};


//==============================================================================
// operator* (SMatrix * SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D1, unsigned int D, unsigned int D2>
inline Expr<MatrixMulOp<SMatrix<T,D1,D>, SMatrix<T,D,D2>, D>, T, D1, D2>
 operator*(const SMatrix<T,D1,D>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef MatrixMulOp<SMatrix<T,D1,D>, SMatrix<T,D,D2>, D> MatMulOp;

  return Expr<MatMulOp,T,D1,D2>(MatMulOp(lhs,rhs));
}

//==============================================================================
// operator* (SMatrix * Expr, binary)
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D, unsigned int D2>
inline Expr<MatrixMulOp<SMatrix<T,D1,D>, Expr<A,T,D,D2>, D>, T, D1, D2>
 operator*(const SMatrix<T,D1,D>& lhs, const Expr<A,T,D,D2>& rhs) {
  typedef MatrixMulOp<SMatrix<T,D1,D>, Expr<A,T,D,D2>, D> MatMulOp;

  return Expr<MatMulOp,T,D1,D2>(MatMulOp(lhs,rhs));
}

//==============================================================================
// operator* (Expr * SMatrix, binary)
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D, unsigned int D2>
inline Expr<MatrixMulOp<Expr<A,T,D1,D>, SMatrix<T,D,D2>, D>, T, D1, D2>
 operator*(const Expr<A,T,D1,D>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef MatrixMulOp<Expr<A,T,D1,D>, SMatrix<T,D,D2>, D> MatMulOp;

  return Expr<MatMulOp,T,D1,D2>(MatMulOp(lhs,rhs));
}

//==============================================================================
// operator* (Expr * Expr, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D1, unsigned int D, unsigned int D2>
inline Expr<MatrixMulOp<Expr<A,T,D1,D>, Expr<B,T,D,D2>, D>, T, D1, D2>
 operator*(const Expr<A,T,D1,D>& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef MatrixMulOp<Expr<A,T,D1,D>, Expr<B,T,D,D2>, D> MatMulOp;

  return Expr<MatMulOp,T,D1,D2>(MatMulOp(lhs,rhs));
}
#endif

//==============================================================================
// TransposeOp
//==============================================================================
template <class Matrix, class T, unsigned int D1, unsigned int D2=D1>
class TransposeOp {
public:
  ///
  TransposeOp( const Matrix& rhs) :
    rhs_(rhs) {}

  ///
  ~TransposeOp() {}

  ///
  inline T apply(unsigned int i) const {
    return rhs_.apply( (i%D1)*D2 + i/D1);
  }

protected:
  const Matrix& rhs_;
};


//==============================================================================
// transpose
//==============================================================================
template <class T, unsigned int D1, unsigned int D2>
inline Expr<TransposeOp<SMatrix<T,D1,D2>,T,D1,D2>, T, D2, D1>
 Transpose(const SMatrix<T,D1,D2>& rhs) {
  typedef TransposeOp<SMatrix<T,D1,D2>,T,D1,D2> MatTrOp;

  return Expr<MatTrOp, T, D2, D1>(MatTrOp(rhs));
}

//==============================================================================
// transpose
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D2>
inline Expr<TransposeOp<Expr<A,T,D1,D2>,T,D1,D2>, T, D2, D1>
 Transpose(const Expr<A,T,D1,D2>& rhs) {
  typedef TransposeOp<Expr<A,T,D1,D2>,T,D1,D2> MatTrOp;

  return Expr<MatTrOp, T, D2, D1>(MatTrOp(rhs));
}

//==============================================================================
// product: SMatrix/SVector calculate v^T * A * v
//==============================================================================
template <class T, unsigned int D>
inline T Product(const SMatrix<T,D>& lhs, const SVector<T,D>& rhs) {
  return Dot(rhs, lhs * rhs);
}

//==============================================================================
// product: SVector/SMatrix calculate v^T * A * v
//==============================================================================
template <class T, unsigned int D>
inline T Product(const SVector<T,D>& lhs, const SMatrix<T,D>& rhs) {
  return Dot(lhs, rhs * lhs);
}

//==============================================================================
// product: SMatrix/Expr calculate v^T * A * v
//==============================================================================
template <class A, class T, unsigned int D>
inline T Product(const SMatrix<T,D>& lhs, const Expr<A,T,D>& rhs) {
  return Dot(rhs, lhs * rhs);
}

//==============================================================================
// product: Expr/SMatrix calculate v^T * A * v
//==============================================================================
template <class A, class T, unsigned int D>
inline T Product(const Expr<A,T,D>& lhs, const SMatrix<T,D>& rhs) {
  return Dot(lhs, rhs * lhs);
}

//==============================================================================
// product: SVector/Expr calculate v^T * A * v
//==============================================================================
template <class A, class T, unsigned int D>
inline T Product(const SVector<T,D>& lhs, const Expr<A,T,D,D>& rhs) {
  return Dot(lhs, rhs * lhs);
}

//==============================================================================
// product: Expr/SVector calculate v^T * A * v
//==============================================================================
template <class A, class T, unsigned int D>
inline T Product(const Expr<A,T,D,D>& lhs, const SVector<T,D>& rhs) {
  return Dot(rhs, lhs * rhs);
}

//==============================================================================
// product: Expr/Expr calculate v^T * A * v
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline T Product(const Expr<A,T,D,D>& lhs, const Expr<B,T,D>& rhs) {
  return Dot(rhs, lhs * rhs);
}

//==============================================================================
// product: Expr/Expr calculate v^T * A * v
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline T Product(const Expr<A,T,D>& lhs, const Expr<B,T,D,D>& rhs) {
  return Dot(lhs, rhs * lhs);
}


  }  // namespace Math

}  // namespace ROOT
          

#endif  /* ROOT_Math_MatrixFunctions */
