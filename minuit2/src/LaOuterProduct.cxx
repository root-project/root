// @(#)root/minuit2:$Name:  $:$Id: LaOuterProduct.cpp,v 1.6.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/LaOuterProduct.h"
#include "Minuit2/LAVector.h"
#include "Minuit2/LASymMatrix.h"

namespace ROOT {

   namespace Minuit2 {


int mndspr(const char*, unsigned int, double, const double*, int, double*);

LASymMatrix::LASymMatrix(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>& out) : fSize(0), fNRow(0), fData(0) {
//   std::cout<<"LASymMatrix::LASymMatrix(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>& out)"<<std::endl;
  fNRow = out.Obj().Obj().Obj().size();
  fSize = fNRow*(fNRow+1)/2;
  fData = (double*)StackAllocatorHolder::Get().Allocate(sizeof(double)*fSize);
  memset(fData, 0, fSize*sizeof(double));
  Outer_prod(*this, out.Obj().Obj().Obj(), out.f()*out.Obj().Obj().f()*out.Obj().Obj().f());
}

LASymMatrix& LASymMatrix::operator=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>& out) {
//   std::cout<<"LASymMatrix& LASymMatrix::operator=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>& out)"<<std::endl;
  if(fSize == 0 && fData == 0) {
    fNRow = out.Obj().Obj().Obj().size();
    fSize = fNRow*(fNRow+1)/2;
    fData = (double*)StackAllocatorHolder::Get().Allocate(sizeof(double)*fSize);
    memset(fData, 0, fSize*sizeof(double));
    Outer_prod(*this, out.Obj().Obj().Obj(), out.f()*out.Obj().Obj().f()*out.Obj().Obj().f());
  } else {
    LASymMatrix tmp(out.Obj().Obj().Obj().size());
    Outer_prod(tmp, out.Obj().Obj().Obj());
    tmp *= double(out.f()*out.Obj().Obj().f()*out.Obj().Obj().f());
    assert(fSize == tmp.size());
    memcpy(fData, tmp.Data(), fSize*sizeof(double));
  }
  return *this;
}

void Outer_prod(LASymMatrix& A, const LAVector& v, double f) {

  mndspr("U", v.size(), f, v.Data(), 1, A.Data());
}

  }  // namespace Minuit2

}  // namespace ROOT
