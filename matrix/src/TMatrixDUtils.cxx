// @(#)root/matrix:$Name:  $:$Id: TMatrixDUtils.cxx,v 1.22 2004/05/19 15:47:40 brun Exp $
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
//   TMatrixDSub_const        TMatrixDSub                               //
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

  Assert(fMatrix->IsValid());
  Double_t *rp = const_cast<Double_t *>(fPtr);
  for ( ; rp < fPtr+fMatrix->GetNcols(); rp += fInc)
    *rp = val;
}

//______________________________________________________________________________
void TMatrixDRow::operator+=(Double_t val)
{
  // Add val to every element of the matrix row. 

  Assert(fMatrix->IsValid());
  Double_t *rp = const_cast<Double_t *>(fPtr);
  for ( ; rp < fPtr+fMatrix->GetNcols(); rp += fInc)
    *rp += val;
}

//______________________________________________________________________________
void TMatrixDRow::operator*=(Double_t val)
{
   // Multiply every element of the matrix row with val.

  Assert(fMatrix->IsValid());
  Double_t *rp = const_cast<Double_t *>(fPtr);
  for ( ; rp < fPtr + fMatrix->GetNcols(); rp += fInc)
    *rp *= val;
}

//______________________________________________________________________________
void TMatrixDRow::operator=(const TMatrixDRow_const &mr)
{
  const TMatrixDBase *mt = mr.GetMatrix();
  if (fMatrix == mt && fRowInd == mr.GetRowIndex()) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetNcols() != mt->GetNcols() || fMatrix->GetColLwb() != mt->GetColLwb()) {
    Error("operator=(const TMatrixDRow_const &)", "matrix rows not compatible");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(vec.IsValid());

  if (fMatrix->GetColLwb() != vec.GetLwb() || fMatrix->GetNcols() != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-row length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator+=(const TMatrixDRow_const &)","different row lengths");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator*=(const TMatrixDRow_const &)","different row lengths");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Double_t *cp = const_cast<Double_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp = val;
}

//______________________________________________________________________________
void TMatrixDColumn::operator+=(Double_t val)
{
  // Add val to every element of the matrix column.

  Assert(fMatrix->IsValid());
  Double_t *cp = const_cast<Double_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp += val;
}

//______________________________________________________________________________
void TMatrixDColumn::operator*=(Double_t val)
{
   // Multiply every element of the matrix column with val.

  Assert(fMatrix->IsValid());
  Double_t *cp = const_cast<Double_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp *= val;
}

//______________________________________________________________________________
void TMatrixDColumn::operator=(const TMatrixDColumn_const &mc) 
{
  const TMatrixDBase *mt = mc.GetMatrix();
  if (fMatrix == mt && fColInd == mc.GetColIndex()) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetNrows() != mt->GetNrows() || fMatrix->GetRowLwb() != mt->GetRowLwb()) {
    Error("operator=(const TMatrixDColumn_const &)", "matrix columns not compatible");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(vec.IsValid());

  if (fMatrix->GetRowLwb() != vec.GetLwb() || fMatrix->GetNrows() != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-column length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetRowLwb() != mt->GetRowLwb() || fMatrix->GetNrows() != mt->GetNrows()) {
    Error("operator+=(const TMatrixDColumn_const &)","different row lengths");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetRowLwb() != mt->GetRowLwb() || fMatrix->GetNrows() != mt->GetNrows()) {
    Error("operator*=(const TMatrixDColumn_const &)","different row lengths");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Double_t *dp = const_cast<Double_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp = val;
}

//______________________________________________________________________________
void TMatrixDDiag::operator+=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  Assert(fMatrix->IsValid());
  Double_t *dp = const_cast<Double_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp += val;
}

//______________________________________________________________________________
void TMatrixDDiag::operator*=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  Assert(fMatrix->IsValid());
  Double_t *dp = const_cast<Double_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp *= val;
}

//______________________________________________________________________________
void TMatrixDDiag::operator=(const TMatrixDDiag_const &md)
{
  const TMatrixDBase *mt = md.GetMatrix();
  if (fMatrix == mt) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (this->GetNdiags() != md.GetNdiags()) {
    Error("operator=(const TMatrixDDiag_const &)","diagonals not compatible");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(vec.IsValid());

  if (fNdiag != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-diagonal length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  Double_t *dp = const_cast<Double_t *>(fPtr);
  const Double_t *vp = vec.GetMatrixArray();
  for ( ; vp < vec.GetMatrixArray()+vec.GetNrows(); dp += fInc)
    *dp = *vp++;
}

//______________________________________________________________________________
void TMatrixDDiag::operator+=(const TMatrixDDiag_const &md)
{
  // Add to every element of the matrix diagonal the
  // corresponding element of diagonal md.

  const TMatrixDBase *mt = md.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fNdiag != md.GetNdiags()) {
    Error("operator=(const TMatrixDDiag_const &)","matrix-diagonal's different length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  Double_t *dp1 = const_cast<Double_t *>(fPtr);
  const Double_t *dp2 = md.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += md.GetInc())
    *dp1 += *dp2;
}

//______________________________________________________________________________
void TMatrixDDiag::operator*=(const TMatrixDDiag_const &md)
{
  // Multiply every element of the matrix diagonal with the
  // corresponding element of diagonal md.

  const TMatrixDBase *mt = md.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fNdiag != md.GetNdiags()) {
    Error("operator*=(const TMatrixDDiag_const &)","matrix-diagonal's different length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  Double_t *dp1 = const_cast<Double_t *>(fPtr);
  const Double_t *dp2 = md.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += md.GetInc())
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

  Assert(fMatrix->IsValid());
  Double_t *fp = const_cast<Double_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ = val;
}

//______________________________________________________________________________
void TMatrixDFlat::operator+=(Double_t val)
{
  // Add val to every element of the matrix.

  Assert(fMatrix->IsValid());
  Double_t *fp = const_cast<Double_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ += val;
}

//______________________________________________________________________________
void TMatrixDFlat::operator*=(Double_t val)
{
  // Multiply every element of the matrix with val.

  Assert(fMatrix->IsValid());
  Double_t *fp = const_cast<Double_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ *= val;
}

//______________________________________________________________________________
void TMatrixDFlat::operator=(const TMatrixDFlat_const &mf)
{
  const TMatrixDBase *mt = mf.GetMatrix();
  if (fMatrix == mt) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetNoElements() != mt->GetNoElements()) {
    Error("operator=(const TMatrixDFlat_const &)","matrix lengths different");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  if (fMatrix->GetNoElements() != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != # matrix-elements");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetNoElements() != mt->GetNoElements()) {
    Error("operator+=(const TMatrixDFlat_const &)","matrices lengths different");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetNoElements() != mt->GetNoElements()) {
    Error("operator*=(const TMatrixDFlat_const &)","matrices lengths different");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  Double_t *fp1 = const_cast<Double_t *>(fPtr);
  const Double_t *fp2 = mf.GetPtr();
  while (fp1 < fPtr + fMatrix->GetNoElements())
    *fp1++ *= *fp2++;
}

//______________________________________________________________________________
TMatrixDSub_const::TMatrixDSub_const(const TMatrixD &matrix,Int_t row_lwbs,Int_t row_upbs,
                                     Int_t col_lwbs,Int_t col_upbs)
{
  // make a reference to submatrix [row_lwbs..row_upbs][col_lwbs..col_upbs];
  // The indexing range of the reference is
  // [0..row_upbs-row_lwbs+1][0..col_upb-col_lwbs+1] (default)

  Assert(matrix.IsValid());

  Assert(row_upbs >= row_lwbs && col_upbs >= col_lwbs);
  const Int_t rowLwb = matrix.GetRowLwb();
  const Int_t rowUpb = matrix.GetRowUpb();
  const Int_t colLwb = matrix.GetColLwb();
  const Int_t colUpb = matrix.GetColUpb();
  Assert(row_lwbs >= rowLwb && row_lwbs <= rowUpb);
  Assert(col_lwbs >= colLwb && col_lwbs <= colUpb);
  Assert(row_upbs >= rowLwb && row_upbs <= rowUpb);
  Assert(col_upbs >= colLwb && col_upbs <= colUpb);

  fRowOff    = row_lwbs-rowLwb;
  fColOff    = col_lwbs-colLwb;
  fNrowsSub  = row_upbs-row_lwbs+1;
  fNcolsSub  = col_upbs-col_lwbs+1;

  fMatrix = &matrix;
}

//______________________________________________________________________________
TMatrixDSub_const::TMatrixDSub_const(const TMatrixDSym &matrix,Int_t row_lwbs,Int_t row_upbs,
                                     Int_t col_lwbs,Int_t col_upbs)
{
  // make a reference to submatrix [row_lwbs..row_upbs][col_lwbs..col_upbs];
  // The indexing range of the reference is
  // [0..row_upbs-row_lwbs+1][0..col_upb-col_lwbs+1] (default)

  Assert(matrix.IsValid());

  Assert(row_upbs >= row_lwbs && col_upbs >= col_lwbs);
  const Int_t rowLwb = matrix.GetRowLwb();
  const Int_t rowUpb = matrix.GetRowUpb();
  const Int_t colLwb = matrix.GetColLwb();
  const Int_t colUpb = matrix.GetColUpb();
  Assert(row_lwbs >= rowLwb && row_lwbs <= rowUpb);
  Assert(col_lwbs >= colLwb && col_lwbs <= colUpb);
  Assert(row_upbs >= rowLwb && row_upbs <= rowUpb);
  Assert(col_upbs >= colLwb && col_upbs <= colUpb);

  fRowOff    = row_lwbs-rowLwb;
  fColOff    = col_lwbs-colLwb;
  fNrowsSub  = row_upbs-row_lwbs+1;
  fNcolsSub  = col_upbs-col_lwbs+1;

  fMatrix = &matrix;
}

//______________________________________________________________________________
TMatrixDSub::TMatrixDSub(TMatrixD &matrix,Int_t row_lwbs,Int_t row_upbs,
                         Int_t col_lwbs,Int_t col_upbs)
            :TMatrixDSub_const(matrix,row_lwbs,row_upbs,col_lwbs,col_upbs)
{
}

//______________________________________________________________________________
TMatrixDSub::TMatrixDSub(TMatrixDSym &matrix,Int_t row_lwbs,Int_t row_upbs,
                         Int_t col_lwbs,Int_t col_upbs)
            :TMatrixDSub_const(matrix,row_lwbs,row_upbs,col_lwbs,col_upbs)
{
}

//______________________________________________________________________________
TMatrixDSub::TMatrixDSub(const TMatrixDSub &ms) : TMatrixDSub_const(ms)
{
  *this = ms;
}

//______________________________________________________________________________
void TMatrixDSub::operator=(Double_t val)
{
  // Assign val to every element of the sub matrix.

  Assert(fMatrix->IsValid());

  Double_t *p = (const_cast<TMatrixDBase *>(fMatrix))->GetMatrixArray();
  const Int_t ncols = fMatrix->GetNcols();
  for (Int_t irow = 0; irow < fNrowsSub; irow++) {
    const Int_t off = (irow+fRowOff)*ncols+fColOff;
    for (Int_t icol = 0; icol < fNcolsSub; icol++)
      p[off+icol] = val;
  }
}

//______________________________________________________________________________
void TMatrixDSub::operator+=(Double_t val)
{
  // Add val to every element of the sub matrix.

  Assert(fMatrix->IsValid());

  Double_t *p = (const_cast<TMatrixDBase *>(fMatrix))->GetMatrixArray();
  const Int_t ncols = fMatrix->GetNcols();
  for (Int_t irow = 0; irow < fNrowsSub; irow++) {
    const Int_t off = (irow+fRowOff)*ncols+fColOff;
    for (Int_t icol = 0; icol < fNcolsSub; icol++)
      p[off+icol] = val;
  }
}

//______________________________________________________________________________
void TMatrixDSub::operator*=(Double_t val)
{
  // Multiply every element of the sub matrix by val .

  Assert(fMatrix->IsValid());

  Double_t *p = (const_cast<TMatrixDBase *>(fMatrix))->GetMatrixArray();
  const Int_t ncols = fMatrix->GetNcols();
  for (Int_t irow = 0; irow < fNrowsSub; irow++) {
    const Int_t off = (irow+fRowOff)*ncols+fColOff;
    for (Int_t icol = 0; icol < fNcolsSub; icol++)
      p[off+icol] = val;
  }
}

//______________________________________________________________________________
void TMatrixDSub::operator=(const TMatrixDSub_const &ms)
{
  const TMatrixDBase *mt = ms.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix == mt &&
      (GetNrows()  == ms.GetNrows () && GetNcols()  == ms.GetNcols () &&
       GetRowOff() == ms.GetRowOff() && GetColOff() == ms.GetColOff()) )
    return;

  if (GetNrows() != ms.GetNrows() || GetNcols() != ms.GetNcols()) {
    Error("operator=(const TMatrixDSub_const &)","sub matrices have different size");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  const Int_t rowOff2 = ms.GetRowOff();
  const Int_t colOff2 = ms.GetColOff();

  Bool_t overlap = (fMatrix == mt) &&
                   ( (rowOff2 >= fRowOff && rowOff2 < fRowOff+fNrowsSub) ||
                     (colOff2 >= fColOff && colOff2 < fColOff+fNcolsSub) );

  Double_t *p1 = const_cast<Double_t *>(fMatrix->GetMatrixArray());
  if (!overlap) {
    const Double_t *p2 = mt->GetMatrixArray();

    const Int_t ncols1 = fMatrix->GetNcols();
    const Int_t ncols2 = mt->GetNcols();
    for (Int_t irow = 0; irow < fNrowsSub; irow++) {
      const Int_t off1 = (irow+fRowOff)*ncols1+fColOff;
      const Int_t off2 = (irow+rowOff2)*ncols2+colOff2;
      for (Int_t icol = 0; icol < fNcolsSub; icol++)
        p1[off1+icol] = p2[off2+icol];
    }
  } else {
    const Int_t row_lwbs = rowOff2+mt->GetRowLwb();
    const Int_t row_upbs = row_lwbs+fNrowsSub-1;
    const Int_t col_lwbs = colOff2+mt->GetColLwb();
    const Int_t col_upbs = col_lwbs+fNcolsSub-1;
    TMatrixD tmp; mt->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,tmp);
    const Double_t *p2 = tmp.GetMatrixArray();

    const Int_t ncols1 = fMatrix->GetNcols();
    const Int_t ncols2 = tmp.GetNcols();
    for (Int_t irow = 0; irow < fNrowsSub; irow++) {
      const Int_t off1 = (irow+fRowOff)*ncols1+fColOff;
      const Int_t off2 = irow*ncols2;
      for (Int_t icol = 0; icol < fNcolsSub; icol++)
        p1[off1+icol] = p2[off2+icol];
    }
  }
}

//______________________________________________________________________________
void TMatrixDSub::operator=(const TMatrixDBase &m)
{
  Assert(fMatrix->IsValid());
  Assert(m.IsValid());

  if (fMatrix == &m) return;

  if (fNrowsSub != m.GetNrows() || fNcolsSub != m.GetNcols()) {
    Error("operator=(const TMatrixDBase &)","sub matrices and matrix have different size"); 
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }
  const Int_t row_lwbs = fRowOff+fMatrix->GetRowLwb();
  const Int_t col_lwbs = fColOff+fMatrix->GetColLwb();
  (const_cast<TMatrixDBase *>(fMatrix))->SetSub(row_lwbs,col_lwbs,m);
}

//______________________________________________________________________________
void TMatrixDSub::operator+=(const TMatrixDSub_const &ms)
{
  const TMatrixDBase *mt = ms.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (GetNrows() != ms.GetNrows() || GetNcols() != ms.GetNcols()) {
    Error("operator+=(const TMatrixDSub_const &)","sub matrices have different size");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  const Int_t rowOff2 = ms.GetRowOff();
  const Int_t colOff2 = ms.GetColOff();

  Bool_t overlap = (fMatrix == mt) &&
                   ( (rowOff2 >= fRowOff && rowOff2 < fRowOff+fNrowsSub) ||
                     (colOff2 >= fColOff && colOff2 < fColOff+fNcolsSub) );

  Double_t *p1 = const_cast<Double_t *>(fMatrix->GetMatrixArray());
  if (!overlap) {
    const Double_t *p2 = mt->GetMatrixArray();

    const Int_t ncols1 = fMatrix->GetNcols();
    const Int_t ncols2 = mt->GetNcols();
    for (Int_t irow = 0; irow < fNrowsSub; irow++) {
      const Int_t off1 = (irow+fRowOff)*ncols1+fColOff;
      const Int_t off2 = (irow+rowOff2)*ncols2+colOff2;
      for (Int_t icol = 0; icol < fNcolsSub; icol++)
        p1[off1+icol] += p2[off2+icol];
    }
  } else {
    const Int_t row_lwbs = rowOff2+mt->GetRowLwb();
    const Int_t row_upbs = row_lwbs+fNrowsSub-1;
    const Int_t col_lwbs = colOff2+mt->GetColLwb();
    const Int_t col_upbs = col_lwbs+fNcolsSub-1;
    TMatrixD tmp; mt->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,tmp);
    const Double_t *p2 = tmp.GetMatrixArray();

    const Int_t ncols1 = fMatrix->GetNcols();
    const Int_t ncols2 = tmp.GetNcols();
    for (Int_t irow = 0; irow < fNrowsSub; irow++) {
      const Int_t off1 = (irow+fRowOff)*ncols1+fColOff;
      const Int_t off2 = irow*ncols2;
      for (Int_t icol = 0; icol < fNcolsSub; icol++)
        p1[off1+icol] += p2[off2+icol];
    }
  }
}

//______________________________________________________________________________
void TMatrixDSub::operator*=(const TMatrixDSub_const &ms)
{
  if (fNcolsSub != ms.GetNrows() || fNcolsSub != ms.GetNcols()) {
    Error("operator*=(const TMatrixDSub_const &)","source sub matrix has wrong shape");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  const TMatrixDBase *source = ms.GetMatrix();

  TMatrixD source_sub;
  {
    const Int_t row_lwbs = ms.GetRowOff()+source->GetRowLwb();
    const Int_t row_upbs = row_lwbs+fNrowsSub-1;
    const Int_t col_lwbs = ms.GetColOff()+source->GetColLwb();
    const Int_t col_upbs = col_lwbs+fNcolsSub-1;
    source->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,source_sub);
  }
  
  const Double_t *sp = source_sub.GetMatrixArray();
  const Int_t ncols = fMatrix->GetNcols();

  // One row of the old_target matrix
  Double_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Double_t *trp = work;
  if (fNcolsSub > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Double_t[fNcolsSub];
  }

        Double_t *cp   = const_cast<Double_t *>(fMatrix->GetMatrixArray())+fRowOff*ncols+fColOff;
  const Double_t *trp0 = cp; // Pointer to  target[i,0];
  const Double_t * const trp0_last = trp0+fNrowsSub*ncols;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcolsSub*sizeof(Double_t));         // copy the i-th row of target, Start at target[i,0]
    for (const Double_t *scp = sp; scp < sp+fNcolsSub; ) {  // Pointer to the j-th column of source,
                                                         // Start scp = source[0,0]
      Double_t cij = 0;
      for (Int_t j = 0; j < fNcolsSub; j++) {
        cij += trp[j] * *scp;                            // the j-th col of source
        scp += fNcolsSub;
      }
      *cp++ = cij;
      scp -= source_sub.GetNoElements()-1;               // Set bcp to the (j+1)-th col
    }
    cp   += ncols-fNcolsSub;
    trp0 += ncols;                                      // Set trp0 to the (i+1)-th row
    Assert(trp0 == cp);
  }

  Assert(cp == trp0_last && trp0 == trp0_last);
  if (isAllocated)
    delete [] trp;
}

//______________________________________________________________________________
void TMatrixDSub::operator+=(const TMatrixDBase &mt)
{
  Assert(fMatrix->IsValid());
  Assert(mt.IsValid());

  if (GetNrows() != mt.GetNrows() || GetNcols() != mt.GetNcols()) {
    Error("operator+=(const TMatrixDBase &)","sub matrix and matrix have different size");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  Double_t *p1 = const_cast<Double_t *>(fMatrix->GetMatrixArray());
  const Double_t *p2 = mt.GetMatrixArray();

  const Int_t ncols1 = fMatrix->GetNcols();
  const Int_t ncols2 = mt.GetNcols();
  for (Int_t irow = 0; irow < fNrowsSub; irow++) {
    const Int_t off1 = (irow+fRowOff)*ncols1+fColOff;
    const Int_t off2 = irow*ncols2;
    for (Int_t icol = 0; icol < fNcolsSub; icol++)
      p1[off1+icol] += p2[off2+icol];
  }
}

//______________________________________________________________________________
void TMatrixDSub::operator*=(const TMatrixD &source)
{
  if (fNcolsSub != source.GetNrows() || fNcolsSub != source.GetNcols()) {
    Error("operator*=(const TMatrixD &)","source matrix has wrong shape");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  // Check for A *= A;
  const Double_t *sp;
  TMatrixD tmp;
  if (fMatrix == &source) {
    tmp.ResizeTo(source);
    tmp = source;
    sp = tmp.GetMatrixArray();
  }
  else
    sp = source.GetMatrixArray();

  const Int_t ncols = fMatrix->GetNcols();

  // One row of the old_target matrix
  Double_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Double_t *trp = work;
  if (fNcolsSub > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Double_t[fNcolsSub];
  }

        Double_t *cp   = const_cast<Double_t *>(fMatrix->GetMatrixArray())+fRowOff*ncols+fColOff;
  const Double_t *trp0 = cp;                               // Pointer to  target[i,0];
  const Double_t * const trp0_last = trp0+fNrowsSub*ncols;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcolsSub*sizeof(Double_t));           // copy the i-th row of target, Start at target[i,0]
    for (const Double_t *scp = sp; scp < sp+fNcolsSub; ) { // Pointer to the j-th column of source,
                                                           // Start scp = source[0,0]
      Double_t cij = 0;
      for (Int_t j = 0; j < fNcolsSub; j++) {
        cij += trp[j] * *scp;                              // the j-th col of source
        scp += fNcolsSub;
      }
      *cp++ = cij;
      scp -= source.GetNoElements()-1;                    // Set bcp to the (j+1)-th col
    }
    cp   += ncols-fNcolsSub;
    trp0 += ncols;                                        // Set trp0 to the (i+1)-th row
    Assert(trp0 == cp);
  }

  Assert(cp == trp0_last && trp0 == trp0_last);
  if (isAllocated)
    delete [] trp;
}

//______________________________________________________________________________
void TMatrixDSub::operator*=(const TMatrixDSym &source)
{
  if (fNcolsSub != source.GetNrows() || fNcolsSub != source.GetNcols()) {
    Error("operator*=(const TMatrixDSym &)","source matrix has wrong shape");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  // Check for A *= A;
  const Double_t *sp;
  TMatrixDSym tmp;
  if ((TMatrixDSym *)fMatrix == &source) {
    tmp.ResizeTo(source);
    tmp = source;
    sp = tmp.GetMatrixArray();
  }
  else
    sp = source.GetMatrixArray();

  const Int_t ncols = fMatrix->GetNcols();

  // One row of the old_target matrix
  Double_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Double_t *trp = work;
  if (fNcolsSub > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Double_t[fNcolsSub];
  }

        Double_t *cp   = const_cast<Double_t *>(fMatrix->GetMatrixArray())+fRowOff*ncols+fColOff;
  const Double_t *trp0 = cp;                               // Pointer to  target[i,0];
  const Double_t * const trp0_last = trp0+fNrowsSub*ncols;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcolsSub*sizeof(Double_t));           // copy the i-th row of target, Start at target[i,0]
    for (const Double_t *scp = sp; scp < sp+fNcolsSub; ) { // Pointer to the j-th column of source,
                                                           // Start scp = source[0,0]
      Double_t cij = 0;
      for (Int_t j = 0; j < fNcolsSub; j++) {
        cij += trp[j] * *scp;                              // the j-th col of source
        scp += fNcolsSub;
      }
      *cp++ = cij;
      scp -= source.GetNoElements()-1;                    // Set bcp to the (j+1)-th col
    }
    cp   += ncols-fNcolsSub;
    trp0 += ncols;                                        // Set trp0 to the (i+1)-th row
    Assert(trp0 == cp);
  }

  Assert(cp == trp0_last && trp0 == trp0_last);
  if (isAllocated)
    delete [] trp;
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
  Assert(fMatrix->IsValid());

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
      (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
      return (const_cast<Double_t*>(fDataPtr))[0];
    }
  }
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator=(Double_t val)
{
  // Assign val to every non-zero (!) element of the matrix row.
  
  Assert(fMatrix->IsValid());
  Double_t *rp = const_cast<Double_t *>(fDataPtr);
  for ( ; rp < fDataPtr+fNindex; rp++)
    *rp = val;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator+=(Double_t val)
{
  // Add val to every non-zero (!) element of the matrix row.
  
  Assert(fMatrix->IsValid());
  Double_t *rp = const_cast<Double_t *>(fDataPtr);
  for ( ; rp < fDataPtr+fNindex; rp++)
    *rp += val;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator*=(Double_t val)
{
  // Multiply every element of the matrix row by val.
  
  Assert(fMatrix->IsValid());
  Double_t *rp = const_cast<Double_t *>(fDataPtr);
  for ( ; rp < fDataPtr+fNindex; rp++)
    *rp *= val;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator=(const TMatrixDSparseRow_const &mr)
{
  const TMatrixDBase *mt = mr.GetMatrix();
  if (fMatrix == mt) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator=(const TMatrixDSparseRow_const &)","matrix rows not compatible");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  const Int_t ncols = fMatrix->GetNcols();
  const Int_t row1  = fRowInd+fMatrix->GetRowLwb();
  const Int_t row2  = mr.GetRowIndex()+mt->GetRowLwb();
  const Int_t col   = fMatrix->GetColLwb();

  TVectorD v(ncols);
  mt->ExtractRow(row2,col,v.GetMatrixArray());
  const_cast<TMatrixDBase *>(fMatrix)->InsertRow(row1,col,v.GetMatrixArray());

  const Int_t sIndex = fMatrix->GetRowIndexArray()[fRowInd];
  const Int_t eIndex = fMatrix->GetRowIndexArray()[fRowInd+1];
  fNindex  = eIndex-sIndex;
  fColPtr  = fMatrix->GetColIndexArray()+sIndex;
  fDataPtr = fMatrix->GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
void TMatrixDSparseRow::operator=(const TVectorD &vec)
{
   // Assign a vector to a matrix row. The vector is considered row-vector
   // to allow the assignment in the strict sense.

  Assert(fMatrix->IsValid());
  Assert(vec.IsValid());

  if (fMatrix->GetColLwb() != vec.GetLwb() || fMatrix->GetNcols() != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-row length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator+=(const TMatrixDRow_const &)","different row lengths");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator+=(const TMatrixDRow_const &)","different row lengths");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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
  Assert(fMatrix->IsValid());

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
      (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
      return (const_cast<Double_t*>(fDataPtr))[index];
    }
  }
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator=(Double_t val)
{
  // Assign val to every element of the matrix diagonal.

  Assert(fMatrix->IsValid());
  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) = val;
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator+=(Double_t val)
{
  // Add val to every element of the matrix diagonal.

  Assert(fMatrix->IsValid());
  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) += val;
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator*=(Double_t val)
{
  // Multiply every element of the matrix diagonal by val.

  Assert(fMatrix->IsValid());
  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) *= val;
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator=(const TMatrixDSparseDiag_const &md)
{
  const TMatrixDBase *mt = md.GetMatrix();
  if (fMatrix == mt) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fNdiag != md.GetNdiags()) {
    Error("operator=(const TMatrixDSparseDiag_const &)","matrix-diagonal's different length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) = md(i);
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator=(const TVectorD &vec)
{
  // Assign a vector to the matrix diagonal.

  Assert(fMatrix->IsValid());
  Assert(vec.IsValid());

  if (fNdiag != vec.GetNrows()) {
    Error("operator=(const TVectorD &)","vector length != matrix-diagonal length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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

  const TMatrixDBase *mt = md.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fNdiag != md.GetNdiags()) {
    Error("operator+=(const TMatrixDSparseDiag_const &)","matrix-diagonal's different length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
    return;
  }

  for (Int_t i = 0; i < fNdiag; i++)
    (*this)(i) += md(i);
}

//______________________________________________________________________________
void TMatrixDSparseDiag::operator*=(const TMatrixDSparseDiag_const &md)
{
  // Multiply every element of the matrix diagonal with the
  // corresponding element of diagonal md.

  const TMatrixDBase *mt = md.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fNdiag != md.GetNdiags()) {
    Error("operator*=(const TMatrixDSparseDiag_const &)","matrix-diagonal's different length");
    (const_cast<TMatrixDBase *>(fMatrix))->Invalidate();
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
