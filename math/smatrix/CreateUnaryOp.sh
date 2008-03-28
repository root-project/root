#!/bin/bash
#
# Script to generate unary operators for SVector and SMatrix class.
# The operator is applied for each element:
#  C(i,j) = OP( B(i,j) )
#
# author: T. Glebe
#
# MPI fuer Kernphysik
# Saupfercheckweg 1
# D-69117 Heidelberg
#
# 19. Mar 2001 (TG) created
# 04. Apr 2001 (TG) Sqrt added
#
##########################################################################

OUTPUTFILE="UnaryOperators.hh"

# Format: Class name, Function name, operator
OPLIST="
Minus,operator-,-
Fabs,fabs,fabs
Sqr,sqr,square
Sqrt,sqrt,sqrt
"

# generate code:
(
echo "\
#ifndef __UNARYOPERATORS_HH
#define __UNARYOPERATORS_HH
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

template <class T, unsigned int D> class SVector;
template <class T, unsigned int D1, unsigned int D2> class SMatrix;
"

  for i in $OPLIST; do
    CNAM=`echo $i | cut -d, -f1`
    FNAM=`echo $i | cut -d, -f2`
    OPER=`echo $i | cut -d, -f3`

    echo "
//==============================================================================
// $CNAM
//==============================================================================
template <class T>
class $CNAM {
public:
  static inline T apply(const T& rhs) {
    return ${OPER}(rhs);
  }
};

//==============================================================================
// $FNAM (Expr, unary)
//==============================================================================
template <class A, class T, unsigned int D>
inline Expr<UnaryOp<${CNAM}<T>, Expr<A,T,D>, T>, T, D>
 ${FNAM}(const Expr<A,T,D>& rhs) {
  typedef UnaryOp<${CNAM}<T>, Expr<A,T,D>, T> ${CNAM}UnaryOp;

  return Expr<${CNAM}UnaryOp,T,D>(${CNAM}UnaryOp(${CNAM}<T>(),rhs));
}


//==============================================================================
// $FNAM (SVector, unary)
//==============================================================================
template <class T, unsigned int D>
inline Expr<UnaryOp<${CNAM}<T>, SVector<T,D>, T>, T, D>
 ${FNAM}(const SVector<T,D>& rhs) {
  typedef UnaryOp<${CNAM}<T>, SVector<T,D>, T> ${CNAM}UnaryOp;

  return Expr<${CNAM}UnaryOp,T,D>(${CNAM}UnaryOp(${CNAM}<T>(),rhs));
}

//==============================================================================
// $FNAM (MatrixExpr, unary)
//==============================================================================
template <class A, class T, unsigned int D, unsigned int D2>
inline Expr<UnaryOp<${CNAM}<T>, Expr<A,T,D,D2>, T>, T, D, D2>
 ${FNAM}(const Expr<A,T,D,D2>& rhs) {
  typedef UnaryOp<${CNAM}<T>, Expr<A,T,D,D2>, T> ${CNAM}UnaryOp;

  return Expr<${CNAM}UnaryOp,T,D,D2>(${CNAM}UnaryOp(${CNAM}<T>(),rhs));
}


//==============================================================================
// $FNAM (SMatrix, unary)
//==============================================================================
template <class T, unsigned int D, unsigned int D2>
inline Expr<UnaryOp<${CNAM}<T>, SMatrix<T,D,D2>, T>, T, D, D2>
 ${FNAM}(const SMatrix<T,D,D2>& rhs) {
  typedef UnaryOp<${CNAM}<T>, SMatrix<T,D,D2>, T> ${CNAM}UnaryOp;

  return Expr<${CNAM}UnaryOp,T,D,D2>(${CNAM}UnaryOp(${CNAM}<T>(),rhs));
}
"
  done

echo "#endif"
) > $OUTPUTFILE

echo "$OUTPUTFILE generated"
