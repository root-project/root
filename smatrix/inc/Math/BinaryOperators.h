// @(#)root/smatrix:$Name:  $:$Id: BinaryOperators.h,v 1.1 2005/11/24 16:03:42 brun Exp $
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

#include "Math/BinaryOpPolicy.h"

namespace ROOT { 

  namespace Math { 



template <class T, unsigned int D> class SVector;
template <class T, unsigned int D1, unsigned int D2, class R> class SMatrix;


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
inline VecExpr<BinaryOp<AddOp<T>, SVector<T,D>, SVector<T,D>, T>, T, D>
 operator+(const SVector<T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, SVector<T,D>, SVector<T,D>, T> AddOpBinOp;

  return VecExpr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SVector, binary)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<AddOp<T>, VecExpr<A,T,D>, SVector<T,D>, T>, T, D>
 operator+(const VecExpr<A,T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, VecExpr<A,T,D>, SVector<T,D>, T> AddOpBinOp;

  return VecExpr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SVector, binary)
//==============================================================================
template < class A, class T, unsigned int D>
inline VecExpr<BinaryOp<AddOp<T>, SVector<T,D>, VecExpr<A,T,D>, T>, T, D>
 operator+(const SVector<T,D>& lhs, const VecExpr<A,T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, SVector<T,D>, VecExpr<A,T,D>, T> AddOpBinOp;

  return VecExpr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SVector, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<AddOp<T>, VecExpr<A,T,D>, VecExpr<B,T,D>, T>, T, D>
 operator+(const VecExpr<A,T,D>& lhs, const VecExpr<B,T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, VecExpr<A,T,D>, VecExpr<B,T,D>, T> AddOpBinOp;

  return VecExpr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<AddOp<T>, SVector<T,D>, Constant<A>, T>, T, D>
 operator+(const SVector<T,D>& lhs, const A& rhs) {
  typedef BinaryOp<AddOp<T>, SVector<T,D>, Constant<A>, T> AddOpBinOp;

  return VecExpr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator+ (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<AddOp<T>, Constant<A>, SVector<T,D>, T>, T, D>
 operator+(const A& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, Constant<A>, SVector<T,D>, T> AddOpBinOp;

  return VecExpr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator+ (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<AddOp<T>, VecExpr<B,T,D>, Constant<A>, T>, T, D>
 operator+(const VecExpr<B,T,D>& lhs, const A& rhs) {
  typedef BinaryOp<AddOp<T>, VecExpr<B,T,D>, Constant<A>, T> AddOpBinOp;

  return VecExpr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator+ (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<AddOp<T>, Constant<A>, VecExpr<B,T,D>, T>, T, D>
 operator+(const A& lhs, const VecExpr<B,T,D>& rhs) {
  typedef BinaryOp<AddOp<T>, Constant<A>, VecExpr<B,T,D>, T> AddOpBinOp;

  return VecExpr<AddOpBinOp,T,D>(AddOpBinOp(AddOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator+ (SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<AddOp<T>, SMatrix<T,D,D2,R1>, SMatrix<T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType >
 operator+(const SMatrix<T,D,D2,R1>& lhs, const SMatrix<T,D,D2,R2>& rhs) {
  typedef BinaryOp<AddOp<T>, SMatrix<T,D,D2,R1>, SMatrix<T,D,D2,R2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SMatrix, binary)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<AddOp<T>, Expr<A,T,D,D2,R1>, SMatrix<T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator+(const Expr<A,T,D,D2,R1>& lhs, const SMatrix<T,D,D2,R2>& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<A,T,D,D2,R1>, SMatrix<T,D,D2,R2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SMatrix, binary)
//==============================================================================
template < class A, class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<AddOp<T>, SMatrix<T,D,D2,R1>, Expr<A,T,D,D2,R2>, T>, T, D, D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator+(const SMatrix<T,D,D2,R1>& lhs, const Expr<A,T,D,D2,R2>& rhs) {
  typedef BinaryOp<AddOp<T>, SMatrix<T,D,D2,R1>, Expr<A,T,D,D2,R2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//==============================================================================
// operator+ (SMatrix, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<AddOp<T>, Expr<A,T,D,D2,R1>, Expr<B,T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType >
 operator+(const Expr<A,T,D,D2,R1>& lhs, const Expr<B,T,D,D2,R1>& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<A,T,D,D2,R1>, Expr<B,T,D,D2,R2>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(AddOpBinOp(AddOp<T>(),lhs,rhs));
}


//=============================================================================
// operator+ (SMatrix, binary, Constant)
//=============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<AddOp<T>, SMatrix<T,D,D2,R>, Constant<A>, T>, T, D, D2, R>
 operator+(const SMatrix<T,D,D2,R>& lhs, const A& rhs) {
  typedef BinaryOp<AddOp<T>, SMatrix<T,D,D2,R>, Constant<A>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2,R>(AddOpBinOp(AddOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator+ (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<AddOp<T>, Constant<A>, SMatrix<T,D,D2,R>, T>, T, D, D2, R>
 operator+(const A& lhs, const SMatrix<T,D,D2,R>& rhs) {
  typedef BinaryOp<AddOp<T>, Constant<A>, SMatrix<T,D,D2,R>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2,R>(AddOpBinOp(AddOp<T>(),
                                              Constant<A>(lhs),rhs));
}


//==============================================================================
// operator+ (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<AddOp<T>, Expr<B,T,D,D2,R>, Constant<A>, T>, T, D, D2, R>
 operator+(const Expr<B,T,D,D2,R>& lhs, const A& rhs) {
  typedef BinaryOp<AddOp<T>, Expr<B,T,D,D2,R>, Constant<A>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2,R>(AddOpBinOp(AddOp<T>(),
                                              lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator+ (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<AddOp<T>, Constant<A>, Expr<B,T,D,D2,R>, T>, T, D, D2>
 operator+(const A& lhs, const Expr<B,T,D,D2,R>& rhs) {
  typedef BinaryOp<AddOp<T>, Constant<A>, Expr<B,T,D,D2,R>, T> AddOpBinOp;

  return Expr<AddOpBinOp,T,D,D2,R>(AddOpBinOp(AddOp<T>(),
                                              Constant<A>(lhs),rhs));
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
inline VecExpr<BinaryOp<MinOp<T>, SVector<T,D>, SVector<T,D>, T>, T, D>
 operator-(const SVector<T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, SVector<T,D>, SVector<T,D>, T> MinOpBinOp;

  return VecExpr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SVector, binary)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<MinOp<T>, VecExpr<A,T,D>, SVector<T,D>, T>, T, D>
 operator-(const VecExpr<A,T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, VecExpr<A,T,D>, SVector<T,D>, T> MinOpBinOp;

  return VecExpr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SVector, binary)
//==============================================================================
template < class A, class T, unsigned int D>
inline VecExpr<BinaryOp<MinOp<T>, SVector<T,D>, VecExpr<A,T,D>, T>, T, D>
 operator-(const SVector<T,D>& lhs, const VecExpr<A,T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, SVector<T,D>, VecExpr<A,T,D>, T> MinOpBinOp;

  return VecExpr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SVector, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<MinOp<T>, VecExpr<A,T,D>, VecExpr<B,T,D>, T>, T, D>
 operator-(const VecExpr<A,T,D>& lhs, const VecExpr<B,T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, VecExpr<A,T,D>, VecExpr<B,T,D>, T> MinOpBinOp;

  return VecExpr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<MinOp<T>, SVector<T,D>, Constant<A>, T>, T, D>
 operator-(const SVector<T,D>& lhs, const A& rhs) {
  typedef BinaryOp<MinOp<T>, SVector<T,D>, Constant<A>, T> MinOpBinOp;

  return VecExpr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator- (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<MinOp<T>, Constant<A>, SVector<T,D>, T>, T, D>
 operator-(const A& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, Constant<A>, SVector<T,D>, T> MinOpBinOp;

  return VecExpr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator- (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<MinOp<T>, VecExpr<B,T,D>, Constant<A>, T>, T, D>
 operator-(const VecExpr<B,T,D>& lhs, const A& rhs) {
  typedef BinaryOp<MinOp<T>, VecExpr<B,T,D>, Constant<A>, T> MinOpBinOp;

  return VecExpr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator- (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<MinOp<T>, Constant<A>, VecExpr<B,T,D>, T>, T, D>
 operator-(const A& lhs, const VecExpr<B,T,D>& rhs) {
  typedef BinaryOp<MinOp<T>, Constant<A>, VecExpr<B,T,D>, T> MinOpBinOp;

  return VecExpr<MinOpBinOp,T,D>(MinOpBinOp(MinOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator- (SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<MinOp<T>, SMatrix<T,D,D2,R1>, SMatrix<T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator-(const SMatrix<T,D,D2,R1>& lhs, const SMatrix<T,D,D2,R2>& rhs) {
  typedef BinaryOp<MinOp<T>, SMatrix<T,D,D2,R1>, SMatrix<T,D,D2,R2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SMatrix, binary)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<MinOp<T>, Expr<A,T,D,D2,R1>, SMatrix<T,D,D2,R2>, T>, T, D, D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator-(const Expr<A,T,D,D2,R1>& lhs, const SMatrix<T,D,D2,R2>& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<A,T,D,D2,R1>, SMatrix<T,D,D2,R2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SMatrix, binary)
//==============================================================================
template < class A, class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<MinOp<T>, SMatrix<T,D,D2,R1>, Expr<A,T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator-(const SMatrix<T,D,D2,R1>& lhs, const Expr<A,T,D,D2,R2>& rhs) {
  typedef BinaryOp<MinOp<T>, SMatrix<T,D,D2,R1>, Expr<A,T,D,D2,R2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SMatrix, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<MinOp<T>, Expr<A,T,D,D2,R1>, Expr<B,T,D,D2,R2>, T>, T, D, D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator-(const Expr<A,T,D,D2,R1>& lhs, const Expr<B,T,D,D2,R2>& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<A,T,D,D2,R1>, Expr<B,T,D,D2,R2>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(MinOpBinOp(MinOp<T>(),lhs,rhs));
}


//==============================================================================
// operator- (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<MinOp<T>, SMatrix<T,D,D2,R>, Constant<A>, T>, T, D, D2, R>
 operator-(const SMatrix<T,D,D2,R>& lhs, const A& rhs) {
  typedef BinaryOp<MinOp<T>, SMatrix<T,D,D2,R>, Constant<A>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2,R>(MinOpBinOp(MinOp<T>(),
                                              lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator- (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<MinOp<T>, Constant<A>, SMatrix<T,D,D2,R>, T>, T, D, D2, R>
 operator-(const A& lhs, const SMatrix<T,D,D2,R>& rhs) {
  typedef BinaryOp<MinOp<T>, Constant<A>, SMatrix<T,D,D2,R>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2,R>(MinOpBinOp(MinOp<T>(),Constant<A>(lhs),rhs));
}

//==============================================================================
// operator- (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<MinOp<T>, Expr<B,T,D,D2,R>, Constant<A>, T>, T, D, D2, R>
 operator-(const Expr<B,T,D,D2,R>& lhs, const A& rhs) {
  typedef BinaryOp<MinOp<T>, Expr<B,T,D,D2,R>, Constant<A>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2,R>(MinOpBinOp(MinOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator- (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<MinOp<T>, Constant<A>, Expr<B,T,D,D2,R>, T>, T, D, D2, R>
 operator-(const A& lhs, const Expr<B,T,D,D2,R>& rhs) {
  typedef BinaryOp<MinOp<T>, Constant<A>, Expr<B,T,D,D2,R>, T> MinOpBinOp;

  return Expr<MinOpBinOp,T,D,D2,R>(MinOpBinOp(MinOp<T>(),
                                              Constant<A>(lhs),rhs));
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
inline VecExpr<BinaryOp<MulOp<T>, SVector<T,D>, SVector<T,D>, T>, T, D>
 operator*(const SVector<T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, SVector<T,D>, SVector<T,D>, T> MulOpBinOp;

  return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SVector, binary)
//==============================================================================
// template <class A,  class T, unsigned int D, class R>
// inline VecExpr<BinaryOp<MulOp<T>, VecExpr<A,T,D>, SVector<T,D>, T>, T, D>
//  operator*(const VecExpr<A,T,D,1,R>& lhs, const SVector<T,D>& rhs) {
//   typedef BinaryOp<MulOp<T>, VecExpr<A,T,D,1,R>, SVector<T,D>, T> MulOpBinOp;
//   return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
// }
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<MulOp<T>, Expr<A,T,D>, SVector<T,D>, T>, T, D>
 operator*(const VecExpr<A,T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, VecExpr<A,T,D>, SVector<T,D>, T> MulOpBinOp;
  return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SVector, binary)
//==============================================================================
template < class A, class T, unsigned int D>
inline VecExpr<BinaryOp<MulOp<T>, SVector<T,D>, VecExpr<A,T,D>, T>, T, D>
 operator*(const SVector<T,D>& lhs, const VecExpr<A,T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, SVector<T,D>, VecExpr<A,T,D>, T> MulOpBinOp;
  return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SVector, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<MulOp<T>, VecExpr<A,T,D>, VecExpr<B,T,D>, T>, T, D>
 operator*(const VecExpr<A,T,D>& lhs, const VecExpr<B,T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, VecExpr<A,T,D>, VecExpr<B,T,D>, T> MulOpBinOp;
  return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// operator* (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<MulOp<T>, SVector<T,D>, Constant<A>, T>, T, D>
 operator*(const SVector<T,D>& lhs, const A& rhs) {
  typedef BinaryOp<MulOp<T>, SVector<T,D>, Constant<A>, T> MulOpBinOp;

  return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator* (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<MulOp<T>, Constant<A>, SVector<T,D>, T>, T, D>
 operator*(const A& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, Constant<A>, SVector<T,D>, T> MulOpBinOp;

  return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator* (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<MulOp<T>, VecExpr<B,T,D>, Constant<A>, T>, T, D>
 operator*(const VecExpr<B,T,D>& lhs, const A& rhs) {
  typedef BinaryOp<MulOp<T>, VecExpr<B,T,D>, Constant<A>, T> MulOpBinOp;

  return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator* (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<MulOp<T>, Constant<A>, VecExpr<B,T,D>, T>, T, D>
 operator*(const A& lhs, const VecExpr<B,T,D>& rhs) {
  typedef BinaryOp<MulOp<T>, Constant<A>, VecExpr<B,T,D>, T> MulOpBinOp;

  return VecExpr<MulOpBinOp,T,D>(MulOpBinOp(MulOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// times (SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<MulOp<T>, SMatrix<T,D,D2,R1>, SMatrix<T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 times(const SMatrix<T,D,D2,R1>& lhs, const SMatrix<T,D,D2,R2>& rhs) {
  typedef BinaryOp<MulOp<T>, SMatrix<T,D,D2,R1>, SMatrix<T,D,D2,R2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// times (SMatrix, binary)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<MulOp<T>, Expr<A,T,D,D2,R1>, SMatrix<T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 times(const Expr<A,T,D,D2,R1>& lhs, const SMatrix<T,D,D2,R2>& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<A,T,D,D2,R1>, SMatrix<T,D,D2,R2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// times (SMatrix, binary)
//==============================================================================
template < class A, class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<MulOp<T>, SMatrix<T,D,D2,R1>, Expr<A,T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 times(const SMatrix<T,D,D2,R1>& lhs, const Expr<A,T,D,D2,R2>& rhs) {
  typedef BinaryOp<MulOp<T>, SMatrix<T,D,D2,R1>, Expr<A,T,D,D2,R2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//==============================================================================
// times (SMatrix, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<MulOp<T>, Expr<A,T,D,D2,R1>, Expr<B,T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 times(const Expr<A,T,D,D2,R1>& lhs, const Expr<B,T,D,D2,R2>& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<A,T,D,D2,R1>, Expr<B,T,D,D2,R2>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(MulOpBinOp(MulOp<T>(),lhs,rhs));
}


//=============================================================================
// operator* (SMatrix, binary, Constant)
//=============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<MulOp<T>, SMatrix<T,D,D2,R>, Constant<A>, T>, T, D, D2, R>
 operator*(const SMatrix<T,D,D2,R>& lhs, const A& rhs) {
  typedef BinaryOp<MulOp<T>, SMatrix<T,D,D2,R>, Constant<A>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2,R>(MulOpBinOp(MulOp<T>(),
                                              lhs,Constant<A>(rhs)));
}

//=============================================================================
// operator* (SMatrix, binary, Constant)
//=============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<MulOp<T>, Constant<A>, SMatrix<T,D,D2,R>, T>, T, D, D2, R>
 operator*(const A& lhs, const SMatrix<T,D,D2,R>& rhs) {
  typedef BinaryOp<MulOp<T>, Constant<A>, SMatrix<T,D,D2,R>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2,R>(MulOpBinOp(MulOp<T>(),
                                              Constant<A>(lhs),rhs));
}


//==============================================================================
// operator* (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<MulOp<T>, Expr<B,T,D,D2,R>, Constant<A>, T>, T, D, D2, R>
 operator*(const Expr<B,T,D,D2,R>& lhs, const A& rhs) {
  typedef BinaryOp<MulOp<T>, Expr<B,T,D,D2,R>, Constant<A>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2,R>(MulOpBinOp(MulOp<T>(),
                                              lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator* (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<MulOp<T>, Constant<A>, Expr<B,T,D,D2,R>, T>, T, D, D2, R>
 operator*(const A& lhs, const Expr<B,T,D,D2,R>& rhs) {
  typedef BinaryOp<MulOp<T>, Constant<A>, Expr<B,T,D,D2,R>, T> MulOpBinOp;

  return Expr<MulOpBinOp,T,D,D2,R>(MulOpBinOp(MulOp<T>(),
                                              Constant<A>(lhs),rhs));
}


//=============================================================================
// DivOp
//=============================================================================
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
inline VecExpr<BinaryOp<DivOp<T>, SVector<T,D>, SVector<T,D>, T>, T, D>
 operator/(const SVector<T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, SVector<T,D>, SVector<T,D>, T> DivOpBinOp;

  return VecExpr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SVector, binary)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<DivOp<T>, VecExpr<A,T,D>, SVector<T,D>, T>, T, D>
 operator/(const VecExpr<A,T,D>& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, VecExpr<A,T,D>, SVector<T,D>, T> DivOpBinOp;  
  return VecExpr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SVector, binary)
//==============================================================================
template < class A, class T, unsigned int D>
inline VecExpr<BinaryOp<DivOp<T>, SVector<T,D>, VecExpr<A,T,D>, T>, T, D>
 operator/(const SVector<T,D>& lhs, const VecExpr<A,T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, SVector<T,D>, VecExpr<A,T,D>, T> DivOpBinOp;

  return VecExpr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//=============================================================================
// operator/ (SVector, binary)
//=============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<DivOp<T>, VecExpr<A,T,D>, VecExpr<B,T,D>, T>, T, D>
 operator/(const VecExpr<A,T,D>& lhs, const VecExpr<B,T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, VecExpr<A,T,D>, VecExpr<B,T,D>, T> DivOpBinOp;

  return VecExpr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<DivOp<T>, SVector<T,D>, Constant<A>, T>, T, D>
 operator/(const SVector<T,D>& lhs, const A& rhs) {
  typedef BinaryOp<DivOp<T>, SVector<T,D>, Constant<A>, T> DivOpBinOp;

  return VecExpr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator/ (SVector, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D>
inline VecExpr<BinaryOp<DivOp<T>, Constant<A>, SVector<T,D>, T>, T, D>
 operator/(const A& lhs, const SVector<T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, Constant<A>, SVector<T,D>, T> DivOpBinOp;

  return VecExpr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator/ (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<DivOp<T>, VecExpr<B,T,D>, Constant<A>, T>, T, D>
 operator/(const VecExpr<B,T,D>& lhs, const A& rhs) {
  typedef BinaryOp<DivOp<T>, VecExpr<B,T,D>, Constant<A>, T> DivOpBinOp;

  return VecExpr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator/ (SVector, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D>
inline VecExpr<BinaryOp<DivOp<T>, Constant<A>, VecExpr<B,T,D>, T>, T, D>
 operator/(const A& lhs, const VecExpr<B,T,D>& rhs) {
  typedef BinaryOp<DivOp<T>, Constant<A>, VecExpr<B,T,D>, T> DivOpBinOp;

  return VecExpr<DivOpBinOp,T,D>(DivOpBinOp(DivOp<T>(),Constant<A>(lhs),rhs));
}


//==============================================================================
// operator/ (SMatrix, binary)
//==============================================================================
template <  class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<DivOp<T>, SMatrix<T,D,D2,R1>, SMatrix<T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator/(const SMatrix<T,D,D2,R1>& lhs, const SMatrix<T,D,D2,R2>& rhs) {
  typedef BinaryOp<DivOp<T>, SMatrix<T,D,D2,R1>, SMatrix<T,D,D2,R2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SMatrix, binary)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<DivOp<T>, Expr<A,T,D,D2,R1>, SMatrix<T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator/(const Expr<A,T,D,D2,R1>& lhs, const SMatrix<T,D,D2,R2>& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<A,T,D,D2,R1>, SMatrix<T,D,D2,R2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SMatrix, binary)
//==============================================================================
template < class A, class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<DivOp<T>, SMatrix<T,D,D2,R1>, Expr<A,T,D,D2,R2>, T>, T, D, D2, typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator/(const SMatrix<T,D,D2,R1>& lhs, const Expr<A,T,D,D2,R2>& rhs) {
  typedef BinaryOp<DivOp<T>, SMatrix<T,D,D2,R1>, Expr<A,T,D,D2,R2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//==============================================================================
// operator/ (SMatrix, binary)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R1, class R2>
inline Expr<BinaryOp<DivOp<T>, Expr<A,T,D,D2,R1>, Expr<B,T,D,D2,R2>, T>, T, D, D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>
 operator/(const Expr<A,T,D,D2,R1>& lhs, const Expr<B,T,D,D2,R2>& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<A,T,D,D2,R1>, Expr<B,T,D,D2,R2>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2,typename AddPolicy<T,D,D2,R1,R2>::RepType>(DivOpBinOp(DivOp<T>(),lhs,rhs));
}


//=============================================================================
// operator/ (SMatrix, binary, Constant)
//=============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<DivOp<T>, SMatrix<T,D,D2,R>, Constant<A>, T>, T, D, D2, R>
 operator/(const SMatrix<T,D,D2,R>& lhs, const A& rhs) {
  typedef BinaryOp<DivOp<T>, SMatrix<T,D,D2,R>, Constant<A>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2,R>(DivOpBinOp(DivOp<T>(),
                                              lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator/ (SMatrix, binary, Constant)
//==============================================================================
template <class A,  class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<DivOp<T>, Constant<A>, SMatrix<T,D,D2,R>, T>, T, D, D2, R>
 operator/(const A& lhs, const SMatrix<T,D,D2,R>& rhs) {
  typedef BinaryOp<DivOp<T>, Constant<A>, SMatrix<T,D,D2,R>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2,R>(DivOpBinOp(DivOp<T>(),
                                              Constant<A>(lhs),rhs));
}


//==============================================================================
// operator/ (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<DivOp<T>, Expr<B,T,D,D2,R>, Constant<A>, T>, T, D, D2, R>
 operator/(const Expr<B,T,D,D2,R>& lhs, const A& rhs) {
  typedef BinaryOp<DivOp<T>, Expr<B,T,D,D2,R>, Constant<A>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2,R>(DivOpBinOp(DivOp<T>(),
                                              lhs,Constant<A>(rhs)));
}

//==============================================================================
// operator/ (SMatrix, binary, Constant)
//==============================================================================
template <class A, class B, class T, unsigned int D, unsigned int D2, class R>
inline Expr<BinaryOp<DivOp<T>, Constant<A>, Expr<B,T,D,D2,R>, T>, T, D, D2,R>
 operator/(const A& lhs, const Expr<B,T,D,D2,R>& rhs) {
  typedef BinaryOp<DivOp<T>, Constant<A>, Expr<B,T,D,D2,R>, T> DivOpBinOp;

  return Expr<DivOpBinOp,T,D,D2,R>(DivOpBinOp(DivOp<T>(),Constant<A>(lhs),rhs));
}



  }  // namespace Math

}  // namespace ROOT
          

#endif  /*ROOT_Math_BinaryOperators */
