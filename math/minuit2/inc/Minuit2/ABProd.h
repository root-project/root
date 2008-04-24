// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_ABProd
#define ROOT_Minuit2_ABProd

#include "Minuit2/ABObj.h"

namespace ROOT {

   namespace Minuit2 {


template<class M1, class M2>
class ABProd {

private:

  ABProd() : fA(M1()), fB(M2()) {}

  ABProd& operator=(const ABProd&) {return *this;}

  template<class A, class B>
  ABProd& operator=(const ABProd<A,B>&) {return *this;}
  
public:

  ABProd(const M1& a, const M2& b): fA(a), fB(b) {}

  ~ABProd() {}

  ABProd(const ABProd& prod) : fA(prod.fA), fB(prod.fB) {}

  template<class A, class B>
  ABProd(const ABProd<A,B>& prod) : fA(M1(prod.fA)), fB(M2(prod.fB)) {}

  const M1& A() const {return fA;}
  const M2& B() const {return fB;}
 
private:

  M1 fA;
  M2 fB;
};

// ABObj * ABObj
template<class atype, class A, class btype, class B, class T>
inline ABObj<typename AlgebraicProdType<atype, btype>::Type, ABProd<ABObj<atype,A,T>, ABObj<btype,B,T> >,T> operator*(const ABObj<atype,A,T>& a, const ABObj<btype,B,T>& b) {

  return ABObj<typename AlgebraicProdType<atype,btype>::Type, ABProd<ABObj<atype,A,T>, ABObj<btype,B,T> >,T>(ABProd<ABObj<atype,A,T>, ABObj<btype,B,T> >(a, b));
}

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_ABProd
