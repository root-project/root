// @(#)root/matrix:$Name:  $:$Id: TMatrixDUtils.cxx,v 1.15 2002/12/10 14:00:48 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix utility classes.                                              //
//                                                                      //
// This file defines utility classes for the Linear Algebra Package.    //
// The following classes are defined here:                              //
//                                                                      //
// Different matrix views without copying data elements :               //
//   TMatrixDRow_const    TMatrixDRow                                   //
//   TMatrixDColumn_const TMatrixDColumn                                //
//   TMatrixDDiag_const   TMatrixDDiag                                  //
//   TMatrixDFlat_const   TMatrixDFlat                                  //
//                                                                      //
//   TElementActionD                                                    //
//   TElementPosActionD                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDBase.h"

ClassImp(TMatrixDRow)
ClassImp(TMatrixDRow_const)
ClassImp(TMatrixDColumn)
ClassImp(TMatrixDColumn_const)
ClassImp(TMatrixDDiag)
ClassImp(TMatrixDDiag_const)
ClassImp(TMatrixDFlat)
ClassImp(TMatrixDFlat_const)

//______________________________________________________________________________
TMatrixDRow_const::TMatrixDRow_const(const TMatrixDBase &matrix,Int_t row)
{
  Assert(matrix.IsValid());
  fRowInd = row-matrix.GetRowLwb();
  if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
    Error("TMatrixDRow_const(const TMatrixDBase &,Int_t)","row index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetElements()+fRowInd*matrix.GetNcols();
  fInc = 1;
}

//______________________________________________________________________________
void TMatrixDRow_const::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixDRow_const.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s,&R__c);
    TMatrixDRow_const::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
    fPtr = fMatrix->GetElements()+fRowInd*fMatrix->GetNcols();
  } else {
    TMatrixDRow_const::Class()->WriteBuffer(R__b,this);
  }
}

//______________________________________________________________________________
TMatrixDRow::TMatrixDRow(TMatrixDBase &matrix,Int_t row)
            :TMatrixDRow_const(matrix,row)
{
}

//______________________________________________________________________________
TMatrixDRow::TMatrixDRow(const TMatrixDRow &mr) : TMatrixDRow_const(mr)
{
  *this = mr;
}

//______________________________________________________________________________
void TMatrixDRow::operator=(Double_t val)
{
  // Assign val to every element of the matrix row.

  Double_t *rp = const_cast<Double_t *>(fPtr);
  for ( ; rp < fPtr+fMatrix->GetNcols(); rp += fInc)
    *rp = val;
}

//______________________________________________________________________________
void TMatrixDRow::operator+=(Double_t val)
{
  // Add val to every element of the matrix row. 

  Double_t *rp = const_cast<Double_t *>(fPtr);
  for ( ; rp < fPtr+fMatrix->GetNcols(); rp += fInc)
    *rp += val;
}

//______________________________________________________________________________
void TMatrixDRow::operator*=(Double_t val)
{
   // Multiply every element of the matrix row with val.

  Double_t *rp = const_cast<Double_t *>(fPtr);
  for ( ; rp < fPtr + fMatrix->GetNcols(); rp += fInc)
    *rp *= val;
}

//______________________________________________________________________________
void TMatrixDRow::operator=(const TMatrixDRow_const &mr)
{
  const TMatrixDBase *mt = mr.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDRow_const &)","matrices not compatible");
    return;
  }

  Double_t *rp1 = const_cast<Double_t *>(fPtr);
  const Double_t *rp2 = mr.GetPtr();
  for ( ; rp1 < fPtr+fMatrix->GetNcols(); rp1 += fInc,rp2 += fInc)
    *rp1 = *rp2;
}

//______________________________________________________________________________
void TMatrixDRow::operator=(const TMatrixDRow &mr)
{
  const TMatrixDBase *mt = mr.GetMatrix();
  if (fMatrix == mt) return;
 
  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDRow &)","matrices not compatible");
    return;
  }

  Double_t *rp1 = const_cast<Double_t *>(fPtr);
  const Double_t *rp2 = mr.GetPtr();
  for ( ; rp1 < fPtr+fMatrix->GetNcols(); rp1 += fInc,rp2 += fInc)
    *rp1 = *rp2;
}

//______________________________________________________________________________
void TMatrixDRow::operator=(const TVectorD &vec)
{
   // Assign a vector to a matrix row. The vector is considered row-vector
   // to allow the assignment in the strict sense.

  Assert(vec.IsValid());

  if (fMatrix->GetColLwb() != vec.GetLwb() || fMatrix->GetNcols() != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-row length");
    return;
  }

  Double_t *rp = const_cast<Double_t *>(fPtr);
  const Double_t *vp = vec.GetElements();
  for ( ; rp < fPtr+fMatrix->GetNcols(); rp += fInc)
    *rp = *vp++;
}

//______________________________________________________________________________
void TMatrixDRow::operator+=(const TMatrixDRow_const &r)
{
  // Add to every element of the matrix row the corresponding element of row r.

  const TMatrixDBase *mt = r.GetMatrix();

  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator+=(const TMatrixDRow_const &)","different row lengths");
    return;
  }

  Double_t *rp1 = const_cast<Double_t *>(fPtr);
  const Double_t *rp2 = r.GetPtr();
  for ( ; rp1 < fPtr+fMatrix->GetNcols(); rp1 += fInc,rp2 += r.GetInc())
   *rp1 += *rp2;
}

//______________________________________________________________________________
void TMatrixDRow::operator*=(const TMatrixDRow_const &r)
{
  // Multiply every element of the matrix row with the
  // corresponding element of row r.

  const TMatrixDBase *mt = r.GetMatrix();

  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator*=(const TMatrixDRow_const &)","different row lengths");
    return;
  }

  Double_t *rp1 = const_cast<Double_t *>(fPtr);
  const Double_t *rp2 = r.GetPtr();
  for ( ; rp1 < fPtr+fMatrix->GetNcols(); rp1 += fInc,rp2 += r.GetInc())
    *rp1 *= *rp2;
}

//______________________________________________________________________________
TMatrixDColumn_const::TMatrixDColumn_const(const TMatrixDBase &matrix,Int_t col)
{
  Assert(matrix.IsValid());
  fColInd = col-matrix.GetColLwb();
  if (fColInd >= matrix.GetNcols() || fColInd < 0) {
    Error("TMatrixDColumn_const(const TMatrixDBase &,Int_t)","column index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetElements()+fColInd;
  fInc = matrix.GetNcols();
}

//______________________________________________________________________________
void TMatrixDColumn_const::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixDColumn.
   
  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s,&R__c);
    TMatrixDColumn_const::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
    fPtr = fMatrix->GetElements()+fColInd;
  } else {
    TMatrixDColumn_const::Class()->WriteBuffer(R__b,this);
  }
}

//______________________________________________________________________________
TMatrixDColumn::TMatrixDColumn(TMatrixDBase &matrix,Int_t col)
               :TMatrixDColumn_const(matrix,col)
{
}

//______________________________________________________________________________
TMatrixDColumn::TMatrixDColumn(const TMatrixDColumn &mc) : TMatrixDColumn_const(mc)
{
  *this = mc;
}

//______________________________________________________________________________
void TMatrixDColumn::operator=(Double_t val)
{
  // Assign val to every element of the matrix column.

  Double_t *cp = const_cast<Double_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp = val;
}

//______________________________________________________________________________
void TMatrixDColumn::operator+=(Double_t val)
{
  // Add val to every element of the matrix column.

  Double_t *cp = const_cast<Double_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp += val;
}

//______________________________________________________________________________
void TMatrixDColumn::operator*=(Double_t val)
{
   // Multiply every element of the matrix column with val.

  Double_t *cp = const_cast<Double_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp *= val;
}

//______________________________________________________________________________
void TMatrixDColumn::operator=(const TMatrixDColumn_const &mc) 
{   
  const TMatrixDBase *mt = mc.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDColumn_const &)","matrices not compatible");
    return;
  }

  Double_t *cp1 = const_cast<Double_t *>(fPtr);
  const Double_t *cp2 = mc.GetPtr();
  for ( ; cp1 < fPtr+fMatrix->GetNoElements(); cp1 += fInc,cp2 += fInc)
    *cp1 = *cp2;
}

//______________________________________________________________________________
void TMatrixDColumn::operator=(const TMatrixDColumn &mc)
{  
  const TMatrixDBase *mt = mc.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDColumn &)","matrices not compatible");
    return;
  }

  Double_t *cp1 = const_cast<Double_t *>(fPtr);
  const Double_t *cp2 = mc.GetPtr();
  for ( ; cp1 < fPtr+fMatrix->GetNoElements(); cp1 += fInc,cp2 += fInc)
    *cp1 = *cp2;
}

//______________________________________________________________________________
void TMatrixDColumn::operator=(const TVectorD &vec)
{
  // Assign a vector to a matrix column.

  Assert(vec.IsValid());

  if (fMatrix->GetRowLwb() != vec.GetLwb() || fMatrix->GetNrows() != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-column length");
    Assert(0);
    return;
  }

  Double_t *cp = const_cast<Double_t *>(fPtr);
  const Double_t *vp = vec.GetElements();
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp = *vp++;

  Assert(vp == vec.GetElements()+vec.GetNrows());
}

//______________________________________________________________________________
void TMatrixDColumn::operator+=(const TMatrixDColumn_const &mc)
{
  const TMatrixDBase *mt = mc.GetMatrix();

  if (fMatrix->GetRowLwb() != mt->GetRowLwb() || fMatrix->GetNrows() != mt->GetNrows()) {
    Error("operator+=(const TMatrixDColumn_const &)","different row lengths");
    return;
  }

  Double_t *cp1 = const_cast<Double_t *>(fPtr);
  const Double_t *cp2 = mc.GetPtr();
  for ( ; cp1 < fPtr+fMatrix->GetNoElements(); cp1 += fInc,cp2 += fInc)
    *cp1 += *cp2;
}

//______________________________________________________________________________
void TMatrixDColumn::operator*=(const TMatrixDColumn_const &mc)
{
  // Multiply every element of the matrix column with the
  // corresponding element of column mc.

  const TMatrixDBase *mt = mc.GetMatrix();

  if (fMatrix->GetRowLwb() != mt->GetRowLwb() || fMatrix->GetNrows() != mt->GetNrows()) {
    Error("operator*=(const TMatrixDColumn_const &)","different row lengths");
    return;
  }

  Double_t *cp1 = const_cast<Double_t *>(fPtr);
  const Double_t *cp2 = mc.GetPtr();
  for ( ; cp1 < fPtr+fMatrix->GetNoElements(); cp1 += fInc,cp2 += fInc)
    *cp1 *= *cp2;
}     

//______________________________________________________________________________
TMatrixDDiag_const::TMatrixDDiag_const(const TMatrixDBase &matrix,Int_t /*dummy*/)
{
  Assert(matrix.IsValid());
  fMatrix = &matrix;
  fNdiag  = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
  fPtr    = matrix.GetElements();
  fInc    = matrix.GetNcols()+1;
}

//______________________________________________________________________________
void TMatrixDDiag_const::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixDDiag.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s,&R__c);
    TMatrixDDiag_const::Class()->ReadBuffer(R__b, this,R__v,R__s,R__c);
    fPtr = fMatrix->GetElements();
  } else {
    TMatrixDDiag_const::Class()->WriteBuffer(R__b,this);
  }
}

//______________________________________________________________________________
TMatrixDDiag::TMatrixDDiag(TMatrixDBase &matrix,Int_t dummy)
             :TMatrixDDiag_const(matrix,dummy)
{
}

//______________________________________________________________________________
TMatrixDDiag::TMatrixDDiag(const TMatrixDDiag &md) : TMatrixDDiag_const(md)
{
  *this = md;
}

//______________________________________________________________________________
void TMatrixDDiag::operator=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  Double_t *dp = const_cast<Double_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp = val;
}

//______________________________________________________________________________
void TMatrixDDiag::operator+=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  Double_t *dp = const_cast<Double_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp += val;
}

//______________________________________________________________________________
void TMatrixDDiag::operator*=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  Double_t *dp = const_cast<Double_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp *= val;
}

//______________________________________________________________________________
void TMatrixDDiag::operator=(const TMatrixDDiag_const &md)
{
  const TMatrixDBase *mt = md.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDDiag_const &)","matrices not compatible");
    return;
  }

  Double_t *dp1 = const_cast<Double_t *>(fPtr);
  const Double_t *dp2 = md.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += fInc)
    *dp1 = *dp2;
}

//______________________________________________________________________________
void TMatrixDDiag::operator=(const TMatrixDDiag &md)
{
  const TMatrixDBase *mt = md.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDDiag &)","matrices not compatible");
    return;
  }

  Double_t *dp1 = const_cast<Double_t *>(fPtr);
  const Double_t *dp2 = md.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += fInc)
    *dp1 = *dp2;
}

//______________________________________________________________________________
void TMatrixDDiag::operator=(const TVectorD &vec)
{
  // Assign a vector to the matrix diagonal.

  Assert(vec.IsValid());

  if (fNdiag != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-diagonal length");
    return;
  }

  Double_t *dp = const_cast<Double_t *>(fPtr);
  const Double_t *vp = vec.GetElements();
  for ( ; vp < vec.GetElements()+vec.GetNrows(); dp += fInc)
    *dp = *vp++;
}

//______________________________________________________________________________
void TMatrixDDiag::operator+=(const TMatrixDDiag_const &d)
{
  // Add to every element of the matrix diagonal the
  // corresponding element of diagonal d.

  if (fNdiag != d.GetNdiags()) {
    Error("operator=(const TMatrixDDiag_const &)","matrix-diagonal's different length");
    return;
  }

  Double_t *dp1 = const_cast<Double_t *>(fPtr);
  const Double_t *dp2 = d.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += d.GetInc())
    *dp1 += *dp2;
}

//______________________________________________________________________________
void TMatrixDDiag::operator*=(const TMatrixDDiag_const &d)
{
  // Add to every element of the matrix diagonal the
  // corresponding element of diagonal d.

  if (fNdiag != d.GetNdiags()) {
    Error("operator*=(const TMatrixDDiag_const &)","matrix-diagonal's different length");
    return;
  }

  Double_t *dp1 = const_cast<Double_t *>(fPtr);
  const Double_t *dp2 = d.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += d.GetInc())
    *dp1 *= *dp2;
}

//______________________________________________________________________________
TMatrixDFlat_const::TMatrixDFlat_const(const TMatrixDBase &matrix,Int_t /*dummy*/)
{
  Assert(matrix.IsValid());
  fMatrix = &matrix;
  fPtr    = matrix.GetElements();
  fNelems = matrix.GetNoElements();
}

//______________________________________________________________________________
void TMatrixDFlat_const::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixDFlat.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s,&R__c);
    TMatrixDFlat_const::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
    fPtr = fMatrix->GetElements();
  } else {
    TMatrixDFlat_const::Class()->WriteBuffer(R__b,this);
  }
}

//______________________________________________________________________________
TMatrixDFlat::TMatrixDFlat(TMatrixDBase &matrix,Int_t dummy)
             :TMatrixDFlat_const(matrix,dummy)
{
}

//______________________________________________________________________________
TMatrixDFlat::TMatrixDFlat(const TMatrixDFlat &mf) : TMatrixDFlat_const(mf)
{
  *this = mf;
}

//______________________________________________________________________________
void TMatrixDFlat::operator=(Double_t val)
{
  // Assign val to every element of the matrix.

  Double_t *fp = const_cast<Double_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ = val;
}

//______________________________________________________________________________
void TMatrixDFlat::operator+=(Double_t val)
{
  // Add val to every element of the matrix.

  Double_t *fp = const_cast<Double_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ += val;
}

//______________________________________________________________________________
void TMatrixDFlat::operator*=(Double_t val)
{
  // Multiply every element of the matrix with val.

  Double_t *fp = const_cast<Double_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ *= val;
}

//______________________________________________________________________________
void TMatrixDFlat::operator=(const TMatrixDFlat_const &mf)
{
  const TMatrixDBase *mt = mf.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDFlat_const &)","matrices not compatible");
    return;
  }

  Double_t *fp1 = const_cast<Double_t *>(fPtr);
  const Double_t *fp2 = mf.GetPtr();
  while (fp1 < fPtr+fMatrix->GetNoElements())
    *fp1++ = *fp2++;
}

//______________________________________________________________________________
void TMatrixDFlat::operator=(const TMatrixDFlat &mf)
{
  const TMatrixDBase *mt = mf.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDFlat &)","matrices not compatible");
    return;
  }

  Double_t *fp1 = const_cast<Double_t *>(fPtr);
  const Double_t *fp2 = mf.GetPtr();
  while (fp1 < fPtr+fMatrix->GetNoElements())
    *fp1++ = *fp2++;
}

//______________________________________________________________________________
void TMatrixDFlat::operator=(const TVectorD &vec)
{
  // Assign a vector to the matrix. The matrix is traversed row-wise

  Assert(vec.IsValid());
  Assert(fMatrix->GetNoElements() == vec.GetNrows());

  if (fMatrix->GetNoElements() != vec.GetNrows()) {
    Error("operator*=(const TVectorD &)","vector length != # matrix-elements");
    return;
  }

  Double_t *fp = const_cast<Double_t *>(fPtr);
  const Double_t *vp = vec.GetElements();
  while (fp < fPtr+fMatrix->GetNoElements())
     *fp++ = *vp++;
}

//______________________________________________________________________________
void TMatrixDFlat::operator+=(const TMatrixDFlat_const &mf)
{
  // Add to every element of the matrix the corresponding element of matrix mf.

  const TMatrixDBase *mt = mf.GetMatrix();

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator+=(const TMatrixDFlat &)","matrices not compatible");
    return;
  }

  Double_t *fp1 = const_cast<Double_t *>(fPtr);
  const Double_t *fp2 = mf.GetPtr();
  while (fp1 < fPtr + fMatrix->GetNoElements())
    *fp1++ += *fp2++;
}

//______________________________________________________________________________
void TMatrixDFlat::operator*=(const TMatrixDFlat_const &mf)
{
  // Multiply every element of the matrix with the corresponding element of diagonal mf.

  const TMatrixDBase *mt = mf.GetMatrix();

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator*=(const TMatrixDFlat_const &)","matrices not compatible");
    return;
  }

  Double_t *fp1 = const_cast<Double_t *>(fPtr);
  const Double_t *fp2 = mf.GetPtr();
  while (fp1 < fPtr + fMatrix->GetNoElements())
    *fp1++ *= *fp2++;
}
