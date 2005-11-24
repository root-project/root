// @(#)root/smatrix:$Name:  $:$Id: BinaryOperators.hv 1.0 2005/11/24 12:00:00 moneta Exp $
// Authors: T. Glebe, L. Moneta    2005  

#ifndef ROOT_Math_BinaryOperators
#define ROOT_Math_BinaryOperators
//======================================================
//
// ATTENTION: This file was automatically generated,
//            do not edit!
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
//======================================================


namespace ROOT { 

  namespace Math { 



template <class T, unsigned int D> class SVector;
template <class T, unsigned int D1, unsigned int D2> class SMatrix;


//==============================================================================
// AddOp
//==============================================================================
template <class T>
class AddOp {
public:
  static inline T apply(const T& lhs, const T& rhs) {
    return lhs + rhs;
  }
};


//==============================================================================
// operator+ (SVector, binary)
//==============================================================================
template <  class T, unsigned int D>
inline Expr<BinaryOp<AddOp<T>, SVector<T,D>, SVector<T,D>, T>, T, D>
 operator+(const SVector<T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, SVector<T,D>, SVector<T,D>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SVector, binary)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<AddOp<T>, Expr<A,T,D>, SVector<T,D>, T>, T, D>
 operator+(const Expr<A,T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<A,T,D>, SVector<T,D>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SVector, binary)
//==============================================================================
template < class A, class T, unsigned int D>
inline Expr<BinaryOp<AddOp<T>, SVector<T,D>, Expr<A,T,D>, T>, T, D>
 operator+(const SVector<T,D>& lhs, const Expr<A,T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, SVector<T,D>, Expr<A,T,D>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SVector, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<AddOp<T>, Expr<A,T,D>, Expr<B,T,D>, T>, T, D>
 operator+(const Expr<A,T,D>& lhs, const Expr<B,T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<A,T,D>, Expr<B,T,D>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<AddOp<T>, SVector<T,D>, Constant<A>, T>, T, D>
 operator+(const SVector<T,D>& lhs, const A& rhs) {
  typedef BinaryOp<AddOp<T>, SVector<T,D>, Constant<A>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator+ (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<AddOp<T>, Constant<A>, SVector<T,D>, T>, T, D>
 operator+(const A& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, Constant<A>, SVector<T,D>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator+ (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<AddOp<T>, Expr<B,T,D>, Constant<A>, T>, T, D>
 operator+(const Expr<B,T,D>& lhs, const A& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<B,T,D>, Constant<A>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator+ (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<AddOp<T>, Constant<A>, Expr<B,T,D>, T>, T, D>
 operator+(const A& lhs, const Expr<B,T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, Constant<A>, Expr<B,T,D>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator+ (SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<AddOp<T>, SMatrix<T,D,D2>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator+(const SMatrix<T,D,D2>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<AddOp<T>, SMatrix<T,D,D2>, SMatrix<T,D,D2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SMatrix, binary)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<AddOp<T>, Expr<A,T,D,D2>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator+(const Expr<A,T,D,D2>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<A,T,D,D2>, SMatrix<T,D,D2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SMatrix, binary)
//==============================================================================
template < class A, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<AddOp<T>, SMatrix<T,D,D2>, Expr<A,T,D,D2>, T>, T, D, D2>
 operator+(const SMatrix<T,D,D2>& lhs, const Expr<A,T,D,D2>& rhs) {
  typedef BinaryOp<AddOp<T>, SMatrix<T,D,D2>, Expr<A,T,D,D2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SMatrix, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<AddOp<T>, Expr<A,T,D,D2>, Expr<B,T,D,D2>, T>, T, D, D2>
 operator+(const Expr<A,T,D,D2>& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<A,T,D,D2>, Expr<B,T,D,D2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<AddOp<T>, SMatrix<T,D,D2>, Constant<A>, T>, T, D, D2>
 operator+(const SMatrix<T,D,D2>& lhs, const A& rhs) {
  typedef BinaryOp<AddOp<T>, SMatrix<T,D,D2>, Constant<A>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2>(AddOpBinOp(AddOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator+ (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<AddOp<T>, Constant<A>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator+(const A& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<AddOp<T>, Constant<A>, SMatrix<T,D,D2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2>(AddOpBinOp(AddOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator+ (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<AddOp<T>, Expr<B,T,D,D2>, Constant<A>, T>, T, D, D2>
 operator+(const Expr<B,T,D,D2>& lhs, const A& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<B,T,D,D2>, Constant<A>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2>(AddOpBinOp(AddOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator+ (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<AddOp<T>, Constant<A>, Expr<B,T,D,D2>, T>, T, D, D2>
 operator+(const A& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef BinaryOp<AddOp<T>, Constant<A>, Expr<B,T,D,D2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2>(AddOpBinOp(AddOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// MinOp
//==============================================================================
template <class T>
class MinOp {
public:
  static inline T apply(const T& lhs, const T& rhs) {
    return lhs - rhs;
  }
};


//==============================================================================
// operator- (SVector, binary)
//==============================================================================
template <  class T, unsigned int D>
inline Expr<BinaryOp<MinOp<T>, SVector<T,D>, SVector<T,D>, T>, T, D>
 operator-(const SVector<T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, SVector<T,D>, SVector<T,D>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SVector, binary)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<MinOp<T>, Expr<A,T,D>, SVector<T,D>, T>, T, D>
 operator-(const Expr<A,T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<A,T,D>, SVector<T,D>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SVector, binary)
//==============================================================================
template < class A, class T, unsigned int D>
inline Expr<BinaryOp<MinOp<T>, SVector<T,D>, Expr<A,T,D>, T>, T, D>
 operator-(const SVector<T,D>& lhs, const Expr<A,T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, SVector<T,D>, Expr<A,T,D>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SVector, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<MinOp<T>, Expr<A,T,D>, Expr<B,T,D>, T>, T, D>
 operator-(const Expr<A,T,D>& lhs, const Expr<B,T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<A,T,D>, Expr<B,T,D>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<MinOp<T>, SVector<T,D>, Constant<A>, T>, T, D>
 operator-(const SVector<T,D>& lhs, const A& rhs) {
  typedef BinaryOp<MinOp<T>, SVector<T,D>, Constant<A>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator- (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<MinOp<T>, Constant<A>, SVector<T,D>, T>, T, D>
 operator-(const A& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, Constant<A>, SVector<T,D>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator- (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<MinOp<T>, Expr<B,T,D>, Constant<A>, T>, T, D>
 operator-(const Expr<B,T,D>& lhs, const A& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<B,T,D>, Constant<A>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator- (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<MinOp<T>, Constant<A>, Expr<B,T,D>, T>, T, D>
 operator-(const A& lhs, const Expr<B,T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, Constant<A>, Expr<B,T,D>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator- (SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MinOp<T>, SMatrix<T,D,D2>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator-(const SMatrix<T,D,D2>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<MinOp<T>, SMatrix<T,D,D2>, SMatrix<T,D,D2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SMatrix, binary)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MinOp<T>, Expr<A,T,D,D2>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator-(const Expr<A,T,D,D2>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<A,T,D,D2>, SMatrix<T,D,D2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SMatrix, binary)
//==============================================================================
template < class A, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MinOp<T>, SMatrix<T,D,D2>, Expr<A,T,D,D2>, T>, T, D, D2>
 operator-(const SMatrix<T,D,D2>& lhs, const Expr<A,T,D,D2>& rhs) {
  typedef BinaryOp<MinOp<T>, SMatrix<T,D,D2>, Expr<A,T,D,D2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SMatrix, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MinOp<T>, Expr<A,T,D,D2>, Expr<B,T,D,D2>, T>, T, D, D2>
 operator-(const Expr<A,T,D,D2>& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<A,T,D,D2>, Expr<B,T,D,D2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MinOp<T>, SMatrix<T,D,D2>, Constant<A>, T>, T, D, D2>
 operator-(const SMatrix<T,D,D2>& lhs, const A& rhs) {
  typedef BinaryOp<MinOp<T>, SMatrix<T,D,D2>, Constant<A>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2>(MinOpBinOp(MinOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator- (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MinOp<T>, Constant<A>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator-(const A& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<MinOp<T>, Constant<A>, SMatrix<T,D,D2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2>(MinOpBinOp(MinOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator- (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MinOp<T>, Expr<B,T,D,D2>, Constant<A>, T>, T, D, D2>
 operator-(const Expr<B,T,D,D2>& lhs, const A& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<B,T,D,D2>, Constant<A>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2>(MinOpBinOp(MinOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator- (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MinOp<T>, Constant<A>, Expr<B,T,D,D2>, T>, T, D, D2>
 operator-(const A& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef BinaryOp<MinOp<T>, Constant<A>, Expr<B,T,D,D2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2>(MinOpBinOp(MinOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// MulOp
//==============================================================================
template <class T>
class MulOp {
public:
  static inline T apply(const T& lhs, const T& rhs) {
    return lhs * rhs;
  }
};


//==============================================================================
// operator* (SVector, binary)
//==============================================================================
template <  class T, unsigned int D>
inline Expr<BinaryOp<MulOp<T>, SVector<T,D>, SVector<T,D>, T>, T, D>
 operator*(const SVector<T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, SVector<T,D>, SVector<T,D>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SVector, binary)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<MulOp<T>, Expr<A,T,D>, SVector<T,D>, T>, T, D>
 operator*(const Expr<A,T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<A,T,D>, SVector<T,D>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SVector, binary)
//==============================================================================
template < class A, class T, unsigned int D>
inline Expr<BinaryOp<MulOp<T>, SVector<T,D>, Expr<A,T,D>, T>, T, D>
 operator*(const SVector<T,D>& lhs, const Expr<A,T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, SVector<T,D>, Expr<A,T,D>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SVector, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<MulOp<T>, Expr<A,T,D>, Expr<B,T,D>, T>, T, D>
 operator*(const Expr<A,T,D>& lhs, const Expr<B,T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<A,T,D>, Expr<B,T,D>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<MulOp<T>, SVector<T,D>, Constant<A>, T>, T, D>
 operator*(const SVector<T,D>& lhs, const A& rhs) {
  typedef BinaryOp<MulOp<T>, SVector<T,D>, Constant<A>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator* (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<MulOp<T>, Constant<A>, SVector<T,D>, T>, T, D>
 operator*(const A& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, Constant<A>, SVector<T,D>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator* (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<MulOp<T>, Expr<B,T,D>, Constant<A>, T>, T, D>
 operator*(const Expr<B,T,D>& lhs, const A& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<B,T,D>, Constant<A>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator* (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<MulOp<T>, Constant<A>, Expr<B,T,D>, T>, T, D>
 operator*(const A& lhs, const Expr<B,T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, Constant<A>, Expr<B,T,D>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// times (SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MulOp<T>, SMatrix<T,D,D2>, SMatrix<T,D,D2>, T>, T, D, D2>
 times(const SMatrix<T,D,D2>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<MulOp<T>, SMatrix<T,D,D2>, SMatrix<T,D,D2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// times (SMatrix, binary)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MulOp<T>, Expr<A,T,D,D2>, SMatrix<T,D,D2>, T>, T, D, D2>
 times(const Expr<A,T,D,D2>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<A,T,D,D2>, SMatrix<T,D,D2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// times (SMatrix, binary)
//==============================================================================
template < class A, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MulOp<T>, SMatrix<T,D,D2>, Expr<A,T,D,D2>, T>, T, D, D2>
 times(const SMatrix<T,D,D2>& lhs, const Expr<A,T,D,D2>& rhs) {
  typedef BinaryOp<MulOp<T>, SMatrix<T,D,D2>, Expr<A,T,D,D2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// times (SMatrix, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MulOp<T>, Expr<A,T,D,D2>, Expr<B,T,D,D2>, T>, T, D, D2>
 times(const Expr<A,T,D,D2>& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<A,T,D,D2>, Expr<B,T,D,D2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MulOp<T>, SMatrix<T,D,D2>, Constant<A>, T>, T, D, D2>
 operator*(const SMatrix<T,D,D2>& lhs, const A& rhs) {
  typedef BinaryOp<MulOp<T>, SMatrix<T,D,D2>, Constant<A>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2>(MulOpBinOp(MulOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator* (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MulOp<T>, Constant<A>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator*(const A& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<MulOp<T>, Constant<A>, SMatrix<T,D,D2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2>(MulOpBinOp(MulOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator* (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MulOp<T>, Expr<B,T,D,D2>, Constant<A>, T>, T, D, D2>
 operator*(const Expr<B,T,D,D2>& lhs, const A& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<B,T,D,D2>, Constant<A>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2>(MulOpBinOp(MulOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator* (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<MulOp<T>, Constant<A>, Expr<B,T,D,D2>, T>, T, D, D2>
 operator*(const A& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef BinaryOp<MulOp<T>, Constant<A>, Expr<B,T,D,D2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2>(MulOpBinOp(MulOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// DivOp
//==============================================================================
template <class T>
class DivOp {
public:
  static inline T apply(const T& lhs, const T& rhs) {
    return lhs / rhs;
  }
};


//==============================================================================
// operator/ (SVector, binary)
//==============================================================================
template <  class T, unsigned int D>
inline Expr<BinaryOp<DivOp<T>, SVector<T,D>, SVector<T,D>, T>, T, D>
 operator/(const SVector<T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, SVector<T,D>, SVector<T,D>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SVector, binary)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<DivOp<T>, Expr<A,T,D>, SVector<T,D>, T>, T, D>
 operator/(const Expr<A,T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<A,T,D>, SVector<T,D>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SVector, binary)
//==============================================================================
template < class A, class T, unsigned int D>
inline Expr<BinaryOp<DivOp<T>, SVector<T,D>, Expr<A,T,D>, T>, T, D>
 operator/(const SVector<T,D>& lhs, const Expr<A,T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, SVector<T,D>, Expr<A,T,D>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SVector, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<DivOp<T>, Expr<A,T,D>, Expr<B,T,D>, T>, T, D>
 operator/(const Expr<A,T,D>& lhs, const Expr<B,T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<A,T,D>, Expr<B,T,D>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<DivOp<T>, SVector<T,D>, Constant<A>, T>, T, D>
 operator/(const SVector<T,D>& lhs, const A& rhs) {
  typedef BinaryOp<DivOp<T>, SVector<T,D>, Constant<A>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator/ (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline Expr<BinaryOp<DivOp<T>, Constant<A>, SVector<T,D>, T>, T, D>
 operator/(const A& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, Constant<A>, SVector<T,D>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator/ (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<DivOp<T>, Expr<B,T,D>, Constant<A>, T>, T, D>
 operator/(const Expr<B,T,D>& lhs, const A& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<B,T,D>, Constant<A>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator/ (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline Expr<BinaryOp<DivOp<T>, Constant<A>, Expr<B,T,D>, T>, T, D>
 operator/(const A& lhs, const Expr<B,T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, Constant<A>, Expr<B,T,D>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator/ (SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<DivOp<T>, SMatrix<T,D,D2>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator/(const SMatrix<T,D,D2>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<DivOp<T>, SMatrix<T,D,D2>, SMatrix<T,D,D2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SMatrix, binary)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<DivOp<T>, Expr<A,T,D,D2>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator/(const Expr<A,T,D,D2>& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<A,T,D,D2>, SMatrix<T,D,D2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SMatrix, binary)
//==============================================================================
template < class A, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<DivOp<T>, SMatrix<T,D,D2>, Expr<A,T,D,D2>, T>, T, D, D2>
 operator/(const SMatrix<T,D,D2>& lhs, const Expr<A,T,D,D2>& rhs) {
  typedef BinaryOp<DivOp<T>, SMatrix<T,D,D2>, Expr<A,T,D,D2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SMatrix, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<DivOp<T>, Expr<A,T,D,D2>, Expr<B,T,D,D2>, T>, T, D, D2>
 operator/(const Expr<A,T,D,D2>& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<A,T,D,D2>, Expr<B,T,D,D2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<DivOp<T>, SMatrix<T,D,D2>, Constant<A>, T>, T, D, D2>
 operator/(const SMatrix<T,D,D2>& lhs, const A& rhs) {
  typedef BinaryOp<DivOp<T>, SMatrix<T,D,D2>, Constant<A>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2>(DivOpBinOp(DivOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator/ (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<DivOp<T>, Constant<A>, SMatrix<T,D,D2>, T>, T, D, D2>
 operator/(const A& lhs, const SMatrix<T,D,D2>& rhs) {
  typedef BinaryOp<DivOp<T>, Constant<A>, SMatrix<T,D,D2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2>(DivOpBinOp(DivOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator/ (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<DivOp<T>, Expr<B,T,D,D2>, Constant<A>, T>, T, D, D2>
 operator/(const Expr<B,T,D,D2>& lhs, const A& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<B,T,D,D2>, Constant<A>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2>(DivOpBinOp(DivOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator/ (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<DivOp<T>, Constant<A>, Expr<B,T,D,D2>, T>, T, D, D2>
 operator/(const A& lhs, const Expr<B,T,D,D2>& rhs) {
  typedef BinaryOp<DivOp<T>, Constant<A>, Expr<B,T,D,D2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2>(DivOpBinOp(DivOp<T>(),Constant<A>(lhs),rhs));
}



  }  // namespace Math

}  // namespace ROOT
          

#endif  /*ROOT_Math_BinaryOperators */
