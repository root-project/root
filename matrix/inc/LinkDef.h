/* @(#)root/matrix:$Name:  $:$Id: LinkDef.h,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ function operator+=(TVector&,const TVector&);
#pragma link C++ function operator-=(TVector&,const TVector&);
#pragma link C++ function Add(TVector&,Double_t,const TVector&);
#pragma link C++ function ElementMult(TVector&,const TVector&);
#pragma link C++ function ElementDiv(TVector&,const TVector&);
#pragma link C++ function operator==(const TVector&, const TVector&);
#pragma link C++ function operator*(const TVector&,const TVector&);
#pragma link C++ function Compare(const TVector&,const TVector&);
#pragma link C++ function AreCompatible(const TVector&,const TVector&);
#pragma link C++ function operator+=(TMatrix&,const TMatrix&);
#pragma link C++ function operator-=(TMatrix&,const TMatrix&);
#pragma link C++ function Add(TMatrix&,Double_t,const TMatrix&);
#pragma link C++ function ElementMult(TMatrix&,const TMatrix&);
#pragma link C++ function ElementDiv(TMatrix&,const TMatrix&);
#pragma link C++ function operator==(const TMatrix&,const TMatrix&);
#pragma link C++ function Compare(const TMatrix&,const TMatrix&);
#pragma link C++ function AreCompatible(const TMatrix&,const TMatrix&);
#pragma link C++ function E2Norm(const TMatrix&,const TMatrix&);

// double precision versions
#pragma link C++ function operator+=(TVectorD&,const TVectorD&);
#pragma link C++ function operator-=(TVectorD&,const TVectorD&);
#pragma link C++ function Add(TVectorD&,Double_t,const TVectorD&);
#pragma link C++ function ElementMult(TVectorD&,const TVectorD&);
#pragma link C++ function ElementDiv(TVectorD&,const TVectorD&);
#pragma link C++ function operator==(const TVectorD&, const TVectorD&);
#pragma link C++ function operator*(const TVectorD&,const TVectorD&);
#pragma link C++ function Compare(const TVectorD&,const TVectorD&);
#pragma link C++ function AreCompatible(const TVectorD&,const TVectorD&);
#pragma link C++ function operator+=(TMatrixD&,const TMatrixD&);
#pragma link C++ function operator-=(TMatrixD&,const TMatrixD&);
#pragma link C++ function Add(TMatrixD&,Double_t,const TMatrixD&);
#pragma link C++ function ElementMult(TMatrixD&,const TMatrixD&);
#pragma link C++ function ElementDiv(TMatrixD&,const TMatrixD&);
#pragma link C++ function operator==(const TMatrixD&,const TMatrixD&);
#pragma link C++ function Compare(const TMatrixD&,const TMatrixD&);
#pragma link C++ function AreCompatible(const TMatrixD&,const TMatrixD&);
#pragma link C++ function E2Norm(const TMatrixD&,const TMatrixD&);

#pragma link C++ class TVector-;
#pragma link C++ class TMatrix-;
#pragma link C++ class TLazyMatrix;
#pragma link C++ class TMatrixRow-;
#pragma link C++ class TMatrixColumn-;
#pragma link C++ class TMatrixDiag-;

#pragma link C++ class TVectorD-;
#pragma link C++ class TMatrixD-;
#pragma link C++ class TLazyMatrixD;
#pragma link C++ class TMatrixDRow-;
#pragma link C++ class TMatrixDColumn-;
#pragma link C++ class TMatrixDDiag-;

#endif
