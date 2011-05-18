#!/bin/bash
#
# Script to generate binary operators for SVector & SMatrix class.
# The operator is applied for each element:
#  C(i,j) = A(i,j) OP B(i,j)
#
# author: T. Glebe
#
# MPI fuer Kernphysik
# Saupfercheckweg 1
# D-69117 Heidelberg
#
# 19. Mar 2001 (TG) created
#
##########################################################################

OUTPUTFILE="BinaryOperators.hh"

# Format: Class name, Function name, operator
OPLIST="
AddOp,operator+,+
MinOp,operator-,-
MulOp,operator*,*
DivOp,operator/,/
"


# generate code:
(
echo "\
#ifndef __BINARYOPERATORS_HH
#define __BINARYOPERATORS_HH
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
  static inline T apply(const T& lhs, const T& rhs) {
    return lhs ${OPER} rhs;
  }
};
"

# =========== SVector / Expr ===============
CMBLIST="
 +SVector<T,D>++SVector<T,D>
 class_A,+Expr<A,T,D>++SVector<T,D>
 +SVector<T,D>+class_A,+Expr<A,T,D>
 class_A,+Expr<A,T,D>+class_B,+Expr<B,T,D>
"
  for j in $CMBLIST; do
    CL1=`echo $j | cut -d+ -f1 | tr '_' ' '`
    TY1=`echo $j | cut -d+ -f2 | tr '_' ' '`
    CL2=`echo $j | cut -d+ -f3 | tr '_' ' '`
    TY2=`echo $j | cut -d+ -f4 | tr '_' ' '`

echo "
//==============================================================================
// $FNAM (SVector, binary)
//==============================================================================
template <${CL1} ${CL2} class T, unsigned int D>
inline Expr<BinaryOp<${CNAM}<T>, ${TY1}, ${TY2}, T>, T, D>
 ${FNAM}(const ${TY1}& lhs, const ${TY2}& rhs) {
  typedef BinaryOp<${CNAM}<T>, ${TY1}, ${TY2}, T> ${CNAM}BinOp;

  return Expr<${CNAM}BinOp,T,D>(${CNAM}BinOp(${CNAM}<T>(),lhs,rhs));
}
"
  done

# =========== SVector Constant ===============
CNSTLIST="
 +SVector<T,D>
 class_B,+Expr<B,T,D>
"

  for j in $CNSTLIST; do
    CL1=`echo $j | cut -d+ -f1 | tr '_' ' '`
    TY1=`echo $j | cut -d+ -f2 | tr '_' ' '`

echo "
//==============================================================================
// $FNAM (SVector, binary, Constant)
//==============================================================================
template <class A, ${CL1} class T, unsigned int D>
inline Expr<BinaryOp<${CNAM}<T>, ${TY1}, Constant<A>, T>, T, D>
 ${FNAM}(const ${TY1}& lhs, const A& rhs) {
  typedef BinaryOp<${CNAM}<T>, ${TY1}, Constant<A>, T> ${CNAM}BinOp;

  return Expr<${CNAM}BinOp,T,D>(${CNAM}BinOp(${CNAM}<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// $FNAM (SVector, binary, Constant)
//==============================================================================
template <class A, ${CL1} class T, unsigned int D>
inline Expr<BinaryOp<${CNAM}<T>, Constant<A>, ${TY1}, T>, T, D>
 ${FNAM}(const A& lhs, const ${TY1}& rhs) {
  typedef BinaryOp<${CNAM}<T>, Constant<A>, ${TY1}, T> ${CNAM}BinOp;

  return Expr<${CNAM}BinOp,T,D>(${CNAM}BinOp(${CNAM}<T>(),Constant<A>(lhs),rhs));
}
"
  done


# =========== SMatrix / Expr ===============
CMBLIST="
 +SMatrix<T,D,D2>++SMatrix<T,D,D2>
 class_A,+Expr<A,T,D,D2>++SMatrix<T,D,D2>
 +SMatrix<T,D,D2>+class_A,+Expr<A,T,D,D2>
 class_A,+Expr<A,T,D,D2>+class_B,+Expr<B,T,D,D2>
"

# component wise multiplication should not occupy operator*()
  if [ "$OPER" == "*" ]; then
     MFNAM="times"
  else
     MFNAM=$FNAM
  fi

  for j in $CMBLIST; do
    CL1=`echo $j | cut -d+ -f1 | tr '_' ' '`
    TY1=`echo $j | cut -d+ -f2 | tr '_' ' '`
    CL2=`echo $j | cut -d+ -f3 | tr '_' ' '`
    TY2=`echo $j | cut -d+ -f4 | tr '_' ' '`

echo "
//==============================================================================
// $MFNAM (SMatrix, binary)
//==============================================================================
template <${CL1} ${CL2} class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<${CNAM}<T>, ${TY1}, ${TY2}, T>, T, D, D2>
 ${MFNAM}(const ${TY1}& lhs, const ${TY2}& rhs) {
  typedef BinaryOp<${CNAM}<T>, ${TY1}, ${TY2}, T> ${CNAM}BinOp;

  return Expr<${CNAM}BinOp,T,D,D2>(${CNAM}BinOp(${CNAM}<T>(),lhs,rhs));
}
"
  done


# =========== SMatrix Constant ===============
CNSTLIST="
 +SMatrix<T,D,D2>
 class_B,+Expr<B,T,D,D2>
"

  for j in $CNSTLIST; do
    CL1=`echo $j | cut -d+ -f1 | tr '_' ' '`
    TY1=`echo $j | cut -d+ -f2 | tr '_' ' '`

echo "
//==============================================================================
// $FNAM (SMatrix, binary, Constant)
//==============================================================================
template <class A, ${CL1} class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<${CNAM}<T>, ${TY1}, Constant<A>, T>, T, D, D2>
 ${FNAM}(const ${TY1}& lhs, const A& rhs) {
  typedef BinaryOp<${CNAM}<T>, ${TY1}, Constant<A>, T> ${CNAM}BinOp;

  return Expr<${CNAM}BinOp,T,D,D2>(${CNAM}BinOp(${CNAM}<T>(),lhs,Constant<A>(rhs)));
}

//==============================================================================
// $FNAM (SMatrix, binary, Constant)
//==============================================================================
template <class A, ${CL1} class T, unsigned int D, unsigned int D2>
inline Expr<BinaryOp<${CNAM}<T>, Constant<A>, ${TY1}, T>, T, D, D2>
 ${FNAM}(const A& lhs, const ${TY1}& rhs) {
  typedef BinaryOp<${CNAM}<T>, Constant<A>, ${TY1}, T> ${CNAM}BinOp;

  return Expr<${CNAM}BinOp,T,D,D2>(${CNAM}BinOp(${CNAM}<T>(),Constant<A>(lhs),rhs));
}
"
  done
done

echo "#endif"
) > $OUTPUTFILE

echo "$OUTPUTFILE generated"