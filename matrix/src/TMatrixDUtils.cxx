// @(#)root/matrix:$Name:  $:$Id: TMatrixDUtils.cxx,v 1.21 2004/05/18 14:01:04 brun Exp $
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
//   TMatrixDRow_const        TMatrixDRow                               //
//   TMatrixDColumn_const     TMatrixDColumn                            //
//   TMatrixDDiag_const       TMatrixDDiag                              //
//   TMatrixDFlat_const       TMatrixDFlat                              //
//   TMatrixDSparseRow_const  TMatrixDSparseRow                         //
//   TMatrixDSparseDiag_const TMatrixDSparseDiag                        //
//                                                                      //
//   TElementActionD                                                    //
//   TElementPosActionD                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDBase.h"

//______________________________________________________________________________
TMatrixDRow_const::TMatrixDRow_const(const TMatrixD &matrix,Int_t row)
{
  Assert(matrix.IsValid());

  fRowInd = row-matrix.GetRowLwb();
  if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
    Error("TMatrixDRow_const(const TMatrixD &,Int_t)","row index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetMatrixArray()+fRowInd*matrix.GetNcols();
  fInc = 1;
}

//______________________________________________________________________________
TMatrixDRow_const::TMatrixDRow_const(const TMatrixDSym &matrix,Int_t row)
{
  Assert(matrix.IsValid());

  fRowInd = row-matrix.GetRowLwb();
  if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
    Error("TMatrixDRow_const(const TMatrixDSym &,Int_t)","row index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetMatrixArray()+fRowInd*matrix.GetNcols();
  fInc = 1;
}

//______________________________________________________________________________
TMatrixDRow::TMatrixDRow(TMatrixD &matrix,Int_t row)
            :TMatrixDRow_const(matrix,row)
{
}

//______________________________________________________________________________
TMatrixDRow::TMatrixDRow(TMatrixDSym &matrix,Int_t row)
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
  const Double_t *vp = vec.GetMatrixArray();
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
TMatrixDColumn_const::TMatrixDColumn_const(const TMatrixD &matrix,Int_t col)
{
  Assert(matrix.IsValid());

  fColInd = col-matrix.GetColLwb();
  if (fColInd >= matrix.GetNcols() || fColInd < 0) {
    Error("TMatrixDColumn_const(const TMatrixD &,Int_t)","column index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetMatrixArray()+fColInd;
  fInc = matrix.GetNcols();
}

//______________________________________________________________________________
TMatrixDColumn_const::TMatrixDColumn_const(const TMatrixDSym &matrix,Int_t col)
{
  Assert(matrix.IsValid());

  fColInd = col-matrix.GetColLwb();
  if (fColInd >= matrix.GetNcols() || fColInd < 0) {
    Error("TMatrixDColumn_const(const TMatrixDSym &,Int_t)","column index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetMatrixArray()+fColInd;
  fInc = matrix.GetNcols();
}

//______________________________________________________________________________
TMatrixDColumn::TMatrixDColumn(TMatrixD &matrix,Int_t col)
               :TMatrixDColumn_const(matrix,col)
{
}

//______________________________________________________________________________
TMatrixDColumn::TMatrixDColumn(TMatrixDSym &matrix,Int_t col)
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
  const Double_t *vp = vec.GetMatrixArray();
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp = *vp++;

  Assert(vp == vec.GetMatrixArray()+vec.GetNrows());
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
TMatrixDDiag_const::TMatrixDDiag_const(const TMatrixD &matrix)
{
  Assert(matrix.IsValid());

  fMatrix = &matrix;
  fNdiag  = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
  fPtr    = matrix.GetMatrixArray();
  fInc    = matrix.GetNcols()+1;
}

//______________________________________________________________________________
TMatrixDDiag_const::TMatrixDDiag_const(const TMatrixDSym &matrix)
{ 
  Assert(matrix.IsValid());
  
  fMatrix = &matrix;
  fNdiag  = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
  fPtr    = matrix.GetMatrixArray();
  fInc    = matrix.GetNcols()+1;
}

//______________________________________________________________________________
TMatrixDDiag::TMatrixDDiag(TMatrixD &matrix)
             :TMatrixDDiag_const(matrix)
{
}

//______________________________________________________________________________
TMatrixDDiag::TMatrixDDiag(TMatrixDSym &matrix)
             :TMatrixDDiag_const(matrix)
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
  const Double_t *vp = vec.GetMatrixArray();
  for ( ; vp < vec.GetMatrixArray()+vec.GetNrows(); dp += fInc)
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
TMatrixDFlat_const::TMatrixDFlat_const(const TMatrixD &matrix)
{
  Assert(matrix.IsValid());

  fMatrix = &matrix;
  fPtr    = matrix.GetMatrixArray();
  fNelems = matrix.GetNoElements();
}

//______________________________________________________________________________
TMatrixDFlat_const::TMatrixDFlat_const(const TMatrixDSym &matrix)
{
  Assert(matrix.IsValid());

  fMatrix = &matrix;
  fPtr    = matrix.GetMatrixArray();
  fNelems = matrix.GetNoElements();
}

//______________________________________________________________________________
TMatrixDFlat::TMatrixDFlat(TMatrixD &matrix)
             :TMatrixDFlat_const(matrix)
{
}

//______________________________________________________________________________
TMatrixDFlat::TMatrixDFlat(TMatrixDSym &matrix)
             :TMatrixDFlat_const(matrix)
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
  const Double_t *vp = vec.GetMatrixArray();
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

//______________________________________________________________________________
TMatrixDSparseRow_const::TMatrixDSparseRow_const(const TMatrixDSparse &matrix,Int_t row)
{
  Assert(matrix.IsValid());

  fRowInd = row-matrix.GetRowLwb();
  if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
    Error("TMatrixDSparseRow_const(const TMatrixDSparse &,Int_t)","row index out of bounds");
    return;
  }

  const Int_t sIndex = matrix.GetRowIndexArray()[fRowInd];
  const Int_t eIndex = matrix.GetRowIndexArray()[fRowInd+1];
  fMatrix  = &matrix;
  fNindex  = eIndex-sIndex;
  fColPtr  = matrix.GetColIndexArray()+sIndex;
  fDataPtr = matrix.GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
TMatrixDSparseRow::TMatrixDSparseRow(TMatrixDSparse &matrix,Int_t row)
                                    : TMatrixDSparseRow_const(matrix,row)
{
}

//______________________________________________________________________________
TMatrixDSparseRow::TMatrixDSparseRow(const TMatrixDSparseRow &mr)
                                    : TMatrixDSparseRow_const(mr)
{
  *this = mr;
} 

//______________________________________________________________________________
Double_t &TMatrixDSparseRow::operator()(Int_t i)
{ 
  const Int_t acoln = i-fMatrix->GetColLwb(); 
  Assert(acoln < fMatrix->GetNcols() && acoln >= 0);
  Int_t index = TMath::BinarySearch(fNindex,fColPtr,acoln);
  if (index >= 0 && fColPtr[index] == acoln)
    return (const_cast<Double_t*>(fDataPtr))[index];
  else {
    TMatrixDBase *mt = const_cast<TMatrixDBase *>(fMatrix);
    const Int_t row = fRowInd+mt->GetRowLwb();
    Double_t val = 0.;
    mt->InsertRow(row,i,&val,1);
    const Int_t sIndex = mt->GetRowIndexArray()[fRowInd];
    const Int_t eIndex = mt->GetRowIndexArray()[fRowInd+1];
    fNindex  = eIndex-sIndex;
    fColPtr  = mt->GetColIndexArray()+sIndex;
    fDataPtr = mt->GetMatrixArray()+sIndex;
    index = TMath::BinarySearch(fNindex,fColPtr,acoln);
    if (index >= 0 && fColPtr[index] == acoln)
      return (const_cast<Double_t*>(fDataPtr))[index];
    else {
      Error("operator()(Int_t","Insert row failed");
      Assert(0);
      return (const_cast<Double_t*>(fDataPtr))[0];
    }
  }
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator=(Double_t val)
{   
  // Assign val to every non-zero (!) element of the matrix row.
  
  Double_t *rp = const_cast<Double_t *>(fDataPtr);
  for ( ; rp < fDataPtr+fNindex; rp++)
    *rp = val;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator+=(Double_t val)
{   
  // Add val to every non-zero (!) element of the matrix row.
  
  Double_t *rp = const_cast<Double_t *>(fDataPtr);
  for ( ; rp < fDataPtr+fNindex; rp++)
    *rp += val;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator*=(Double_t val)
{   
  // Multiply every non-zero (!) element of the matrix row by val.
  
  Double_t *rp = const_cast<Double_t *>(fDataPtr);
  for ( ; rp < fDataPtr+fNindex; rp++)
    *rp *= val;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator=(const TMatrixDSparseRow_const &mr)
{
  const TMatrixDBase *mt = mr.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDSparseRow_const &)","matrices not compatible");
    return;
  }

  Double_t *rp1 = const_cast<Double_t *>(fDataPtr);
  const Double_t *rp2 = mr.GetDataPtr();
  for ( ; rp1 < fDataPtr+fNindex; rp1++,rp2++)
    *rp1 = *rp2;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator=(const TMatrixDSparseRow &mr)
{
  const TMatrixDBase *mt = mr.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDSparseRow &)","matrices not compatible");
    return;
  }

  Double_t *rp1 = const_cast<Double_t *>(fDataPtr);
  const Double_t *rp2 = mr.GetDataPtr();
  for ( ; rp1 < fDataPtr+fNindex; rp1++,rp2++)
    *rp1 = *rp2;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator=(const TVectorD &vec)
{
   // Assign a vector to a matrix row. The vector is considered row-vector
   // to allow the assignment in the strict sense.

  Assert(vec.IsValid());

  if (fMatrix->GetColLwb() != vec.GetLwb() || fMatrix->GetNcols() != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-row length");
    return;
  }

  const Double_t *vp = vec.GetMatrixArray();
  const Int_t row = fRowInd+fMatrix->GetRowLwb();
  const Int_t col = fMatrix->GetColLwb();
  const_cast<TMatrixDBase *>(fMatrix)->InsertRow(row,col,vp,vec.GetNrows());

  const Int_t sIndex = fMatrix->GetRowIndexArray()[fRowInd];
  const Int_t eIndex = fMatrix->GetRowIndexArray()[fRowInd+1];
  fNindex  = eIndex-sIndex;
  fColPtr  = fMatrix->GetColIndexArray()+sIndex;
  fDataPtr = fMatrix->GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator+=(const TMatrixDSparseRow_const &r)
{
  // Add to every element of the matrix row the corresponding element of row r.

  const TMatrixDBase *mt = r.GetMatrix();

  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator+=(const TMatrixDRow_const &)","different row lengths");
    return;
  }

  const Int_t ncols = fMatrix->GetNcols();
  const Int_t row1  = fRowInd+fMatrix->GetRowLwb();
  const Int_t row2  = r.GetRowIndex()+mt->GetRowLwb();
  const Int_t col   = fMatrix->GetColLwb();

  TVectorD v1(ncols);
  TVectorD v2(ncols);
  fMatrix->ExtractRow(row1,col,v1.GetMatrixArray());
  mt     ->ExtractRow(row2,col,v2.GetMatrixArray());
  v1 += v2;
  const_cast<TMatrixDBase *>(fMatrix)->InsertRow(row1,col,v1.GetMatrixArray());

  const Int_t sIndex = fMatrix->GetRowIndexArray()[fRowInd];
  const Int_t eIndex = fMatrix->GetRowIndexArray()[fRowInd+1];
  fNindex  = eIndex-sIndex;
  fColPtr  = fMatrix->GetColIndexArray()+sIndex;
  fDataPtr = fMatrix->GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator*=(const TMatrixDSparseRow_const &r)
{
  // Multiply every element of the matrix row with the
  // corresponding element of row r.

  const TMatrixDBase *mt = r.GetMatrix();

  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator+=(const TMatrixDRow_const &)","different row lengths");
    return;
  }

  const Int_t ncols = fMatrix->GetNcols();
  const Int_t row1  = r.GetRowIndex()+mt->GetRowLwb();
  const Int_t row2  = r.GetRowIndex()+mt->GetRowLwb();
  const Int_t col   = fMatrix->GetColLwb();

  TVectorD v1(ncols);
  TVectorD v2(ncols);
  fMatrix->ExtractRow(row1,col,v1.GetMatrixArray());
  mt     ->ExtractRow(row2,col,v2.GetMatrixArray());

  ElementMult(v1,v2);
  const_cast<TMatrixDBase *>(fMatrix)->InsertRow(row1,col,v1.GetMatrixArray());

  const Int_t sIndex = fMatrix->GetRowIndexArray()[fRowInd];
  const Int_t eIndex = fMatrix->GetRowIndexArray()[fRowInd+1];
  fNindex  = eIndex-sIndex;
  fColPtr  = fMatrix->GetColIndexArray()+sIndex;
  fDataPtr = fMatrix->GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
TMatrixDSparseDiag_const::TMatrixDSparseDiag_const(const TMatrixDSparse &matrix)
{
  Assert(matrix.IsValid());

  fMatrix  = &matrix;
  fNdiag   = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
  fDataPtr = matrix.GetMatrixArray();
}

//______________________________________________________________________________
TMatrixDSparseDiag::TMatrixDSparseDiag(TMatrixDSparse &matrix)
                   :TMatrixDSparseDiag_const(matrix)
{
}

//______________________________________________________________________________
TMatrixDSparseDiag::TMatrixDSparseDiag(const TMatrixDSparseDiag &md)
                  : TMatrixDSparseDiag_const(md)
{
  *this = md;
}

//______________________________________________________________________________
Double_t &TMatrixDSparseDiag::operator()(Int_t i)
{
  Assert(i < fNdiag && i >= 0);
  TMatrixDBase *mt = const_cast<TMatrixDBase *>(fMatrix);
  const Int_t    *pR = mt->GetRowIndexArray();
  const Int_t    *pC = mt->GetColIndexArray();
  Int_t sIndex = pR[i];
  Int_t eIndex = pR[i+1];
  Int_t index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
  if (index >= sIndex && pC[index] == i)
    return (const_cast<Double_t*>(fDataPtr))[index];
  else {
    const Int_t row = i+mt->GetRowLwb();
    const Int_t col = i+mt->GetColLwb();
    Double_t val = 0.;
    mt->InsertRow(row,col,&val,1);
    fDataPtr = mt->GetMatrixArray();
    pR = mt->GetRowIndexArray();
    pC = mt->GetColIndexArray();
    sIndex = pR[i];
    eIndex = pR[i+1];
    index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
    if (index >= sIndex && pC[index] == i)
      return (const_cast<Double_t*>(fDataPtr))[index];
    else {
      Error("operator()(Int_t","Insert row failed");
      Assert(0);
      return (const_cast<Double_t*>(fDataPtr))[index];
    }
  }
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) = val;
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator+=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) += val;
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator*=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) *= val;
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator=(const TMatrixDSparseDiag_const &md)
{
  const TMatrixDBase *mt = md.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDSparseDiag_const &)","matrices not compatible");
    return;
  }

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) = md(i);
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator=(const TMatrixDSparseDiag &md)
{
  const TMatrixDBase *mt = md.GetMatrix();
  if (fMatrix == mt) return;

  if (!AreCompatible(*fMatrix,*mt)) {
    Error("operator=(const TMatrixDSparseDiag &)","matrices not compatible");
    return;
  }

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) = md(i);
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator=(const TVectorD &vec)
{
  // Assign a vector to the matrix diagonal.

  Assert(vec.IsValid());

  if (fNdiag != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-diagonal length");
    return;
  }

  const Double_t *vp = vec.GetMatrixArray();
  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) = vp[i];
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator+=(const TMatrixDSparseDiag_const &md)
{
  // Add to every element of the matrix diagonal the
  // corresponding element of diagonal md.

  if (fNdiag != md.GetNdiags()) {
    Error("operator=(const TMatrixDSparseDiag_const &)","matrix-diagonal's different length");
    return;
  }

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) += md(i);
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator*=(const TMatrixDSparseDiag_const &md)
{
  // Add to every element of the matrix diagonal the
  // corresponding element of diagonal md.

  if (fNdiag != md.GetNdiags()) {
    Error("operator*=(const TMatrixDSparseDiag_const &)","matrix-diagonal's different length");
    return;
  }

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) *= md(i);
}

//______________________________________________________________________________
Double_t Drand(Double_t &ix)
{
  const Double_t a   = 16807.0;
  const Double_t b15 = 32768.0;
  const Double_t b16 = 65536.0;
  const Double_t p   = 2147483647.0;
  Double_t xhi = ix/b16;
  Int_t xhiint = (Int_t) xhi;
  xhi = xhiint;
  Double_t xalo = (ix-xhi*b16)*a;

  Double_t leftlo = xalo/b16;
  Int_t leftloint = (int) leftlo;
  leftlo = leftloint;
  Double_t fhi = xhi*a+leftlo;
  Double_t k = fhi/b15;
  Int_t kint = (Int_t) k;
  k = kint;
  ix = (((xalo-leftlo*b16)-p)+(fhi-k*b15)*b16)+k;
  if (ix < 0.0) ix = ix+p;

  return (ix*4.656612875e-10);
}
