// @(#)root/matrix:$Name:  $:$Id: TMatrixFUtils.cxx,v 1.5 2004/05/12 10:39:29 brun Exp $
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
//   TMatrixFRow_const        TMatrixFRow                               //
//   TMatrixFColumn_const     TMatrixFColumn                            //
//   TMatrixFDiag_const       TMatrixFDiag                              //
//   TMatrixFFlat_const       TMatrixFFlat                              //
//   TMatrixFSub_const        TMatrixFSub                               //
//                                                                      //
//   TElementActionF                                                    //
//   TElementPosActionF                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixFBase.h"

//______________________________________________________________________________
TMatrixFRow_const::TMatrixFRow_const(const TMatrixF &matrix,Int_t row)
{
  Assert(matrix.IsValid());

  fRowInd = row-matrix.GetRowLwb();
  if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
    Error("TMatrixFRow_const(const TMatrixF &,Int_t)","row index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetMatrixArray()+fRowInd*matrix.GetNcols();
  fInc = 1;
}

//______________________________________________________________________________
TMatrixFRow_const::TMatrixFRow_const(const TMatrixFSym &matrix,Int_t row)
{
  Assert(matrix.IsValid());

  fRowInd = row-matrix.GetRowLwb();
  if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
    Error("TMatrixFRow_const(const TMatrixFSym &,Int_t)","row index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetMatrixArray()+fRowInd*matrix.GetNcols();
  fInc = 1;
}

//______________________________________________________________________________
TMatrixFRow::TMatrixFRow(TMatrixF &matrix,Int_t row)
            :TMatrixFRow_const(matrix,row)
{
}

//______________________________________________________________________________
TMatrixFRow::TMatrixFRow(TMatrixFSym &matrix,Int_t row)
            :TMatrixFRow_const(matrix,row)
{
}

//______________________________________________________________________________
TMatrixFRow::TMatrixFRow(const TMatrixFRow &mr) : TMatrixFRow_const(mr)
{
  *this = mr;
}

//______________________________________________________________________________
void TMatrixFRow::operator=(Float_t val)
{
  // Assign val to every element of the matrix row.

  Assert(fMatrix->IsValid());
  Float_t *rp = const_cast<Float_t *>(fPtr);
  for ( ; rp < fPtr+fMatrix->GetNcols(); rp += fInc)
    *rp = val;
}

//______________________________________________________________________________
void TMatrixFRow::operator+=(Float_t val)
{
  // Add val to every element of the matrix row. 

  Assert(fMatrix->IsValid());
  Float_t *rp = const_cast<Float_t *>(fPtr);
  for ( ; rp < fPtr+fMatrix->GetNcols(); rp += fInc)
    *rp += val;
}

//______________________________________________________________________________
void TMatrixFRow::operator*=(Float_t val)
{
   // Multiply every element of the matrix row with val.

  Assert(fMatrix->IsValid());
  Float_t *rp = const_cast<Float_t *>(fPtr);
  for ( ; rp < fPtr + fMatrix->GetNcols(); rp += fInc)
    *rp *= val;
}

//______________________________________________________________________________
void TMatrixFRow::operator=(const TMatrixFRow_const &mr)
{
  const TMatrixFBase *mt = mr.GetMatrix();
  if (fMatrix == mt && fRowInd == mr.GetRowIndex()) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetNcols() != mt->GetNcols() || fMatrix->GetColLwb() != mt->GetColLwb()) {
    Error("operator=(const TMatrixFRow_const &)", "matrix rows not compatible");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *rp1 = const_cast<Float_t *>(fPtr);
  const Float_t *rp2 = mr.GetPtr();
  for ( ; rp1 < fPtr+fMatrix->GetNcols(); rp1 += fInc,rp2 += fInc)
    *rp1 = *rp2;
}

//______________________________________________________________________________
void TMatrixFRow::operator=(const TVectorF &vec)
{
   // Assign a vector to a matrix row. The vector is considered row-vector
   // to allow the assignment in the strict sense.

  Assert(fMatrix->IsValid());
  Assert(vec.IsValid());

  if (fMatrix->GetColLwb() != vec.GetLwb() || fMatrix->GetNcols() != vec.GetNrows()) {
    Error("operator=(const TVectorF &)","vector length != matrix-row length");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *rp = const_cast<Float_t *>(fPtr);
  const Float_t *vp = vec.GetMatrixArray();
  for ( ; rp < fPtr+fMatrix->GetNcols(); rp += fInc)
    *rp = *vp++;
}

//______________________________________________________________________________
void TMatrixFRow::operator+=(const TMatrixFRow_const &r)
{
  // Add to every element of the matrix row the corresponding element of row r.

  const TMatrixFBase *mt = r.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator+=(const TMatrixFRow_const &)","different row lengths");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *rp1 = const_cast<Float_t *>(fPtr);
  const Float_t *rp2 = r.GetPtr();
  for ( ; rp1 < fPtr+fMatrix->GetNcols(); rp1 += fInc,rp2 += r.GetInc())
   *rp1 += *rp2;
}

//______________________________________________________________________________
void TMatrixFRow::operator*=(const TMatrixFRow_const &r)
{
  // Multiply every element of the matrix row with the
  // corresponding element of row r.

  const TMatrixFBase *mt = r.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetColLwb() != mt->GetColLwb() || fMatrix->GetNcols() != mt->GetNcols()) {
    Error("operator*=(const TMatrixFRow_const &)","different row lengths");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *rp1 = const_cast<Float_t *>(fPtr);
  const Float_t *rp2 = r.GetPtr();
  for ( ; rp1 < fPtr+fMatrix->GetNcols(); rp1 += fInc,rp2 += r.GetInc())
    *rp1 *= *rp2;
}

//______________________________________________________________________________
TMatrixFColumn_const::TMatrixFColumn_const(const TMatrixF &matrix,Int_t col)
{
  Assert(matrix.IsValid());

  fColInd = col-matrix.GetColLwb();
  if (fColInd >= matrix.GetNcols() || fColInd < 0) {
    Error("TMatrixFColumn_const(const TMatrixF &,Int_t)","column index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetMatrixArray()+fColInd;
  fInc = matrix.GetNcols();
}

//______________________________________________________________________________
TMatrixFColumn_const::TMatrixFColumn_const(const TMatrixFSym &matrix,Int_t col)
{
  Assert(matrix.IsValid());

  fColInd = col-matrix.GetColLwb();
  if (fColInd >= matrix.GetNcols() || fColInd < 0) {
    Error("TMatrixFColumn_const(const TMatrixFSym &,Int_t)","column index out of bounds");
    return;
  }

  fMatrix = &matrix;
  fPtr = matrix.GetMatrixArray()+fColInd;
  fInc = matrix.GetNcols();
}

//______________________________________________________________________________
TMatrixFColumn::TMatrixFColumn(TMatrixF &matrix,Int_t col)
               :TMatrixFColumn_const(matrix,col)
{
}

//______________________________________________________________________________
TMatrixFColumn::TMatrixFColumn(TMatrixFSym &matrix,Int_t col)
               :TMatrixFColumn_const(matrix,col)
{
}

//______________________________________________________________________________
TMatrixFColumn::TMatrixFColumn(const TMatrixFColumn &mc) : TMatrixFColumn_const(mc)
{
  *this = mc;
}

//______________________________________________________________________________
void TMatrixFColumn::operator=(Float_t val)
{
  // Assign val to every element of the matrix column.

  Assert(fMatrix->IsValid());
  Float_t *cp = const_cast<Float_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp = val;
}

//______________________________________________________________________________
void TMatrixFColumn::operator+=(Float_t val)
{
  // Add val to every element of the matrix column.

  Assert(fMatrix->IsValid());
  Float_t *cp = const_cast<Float_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp += val;
}

//______________________________________________________________________________
void TMatrixFColumn::operator*=(Float_t val)
{
   // Multiply every element of the matrix column with val.

  Assert(fMatrix->IsValid());
  Float_t *cp = const_cast<Float_t *>(fPtr);
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp *= val;
}

//______________________________________________________________________________
void TMatrixFColumn::operator=(const TMatrixFColumn_const &mc) 
{
  const TMatrixFBase *mt = mc.GetMatrix();
  if (fMatrix == mt && fColInd == mc.GetColIndex()) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetNrows() != mt->GetNrows() || fMatrix->GetRowLwb() != mt->GetRowLwb()) {
    Error("operator=(const TMatrixFColumn_const &)", "matrix columns not compatible");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return; 
  }

  Float_t *cp1 = const_cast<Float_t *>(fPtr);
  const Float_t *cp2 = mc.GetPtr();
  for ( ; cp1 < fPtr+fMatrix->GetNoElements(); cp1 += fInc,cp2 += fInc)
    *cp1 = *cp2;
}

//______________________________________________________________________________
void TMatrixFColumn::operator=(const TVectorF &vec)
{
  // Assign a vector to a matrix column.

  Assert(fMatrix->IsValid());
  Assert(vec.IsValid());

  if (fMatrix->GetRowLwb() != vec.GetLwb() || fMatrix->GetNrows() != vec.GetNrows()) {
    Error("operator=(const TVectorF &)","vector length != matrix-column length");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *cp = const_cast<Float_t *>(fPtr);
  const Float_t *vp = vec.GetMatrixArray();
  for ( ; cp < fPtr+fMatrix->GetNoElements(); cp += fInc)
    *cp = *vp++;

  Assert(vp == vec.GetMatrixArray()+vec.GetNrows());
}

//______________________________________________________________________________
void TMatrixFColumn::operator+=(const TMatrixFColumn_const &mc)
{
  const TMatrixFBase *mt = mc.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetRowLwb() != mt->GetRowLwb() || fMatrix->GetNrows() != mt->GetNrows()) {
    Error("operator+=(const TMatrixFColumn_const &)","different row lengths");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *cp1 = const_cast<Float_t *>(fPtr);
  const Float_t *cp2 = mc.GetPtr();
  for ( ; cp1 < fPtr+fMatrix->GetNoElements(); cp1 += fInc,cp2 += fInc)
    *cp1 += *cp2;
}

//______________________________________________________________________________
void TMatrixFColumn::operator*=(const TMatrixFColumn_const &mc)
{
  // Multiply every element of the matrix column with the
  // corresponding element of column mc.

  const TMatrixFBase *mt = mc.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix->GetRowLwb() != mt->GetRowLwb() || fMatrix->GetNrows() != mt->GetNrows()) {
    Error("operator*=(const TMatrixFColumn_const &)","different row lengths");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *cp1 = const_cast<Float_t *>(fPtr);
  const Float_t *cp2 = mc.GetPtr();
  for ( ; cp1 < fPtr+fMatrix->GetNoElements(); cp1 += fInc,cp2 += fInc)
    *cp1 *= *cp2;
}

//______________________________________________________________________________
TMatrixFDiag_const::TMatrixFDiag_const(const TMatrixF &matrix)
{
  Assert(matrix.IsValid());

  fMatrix = &matrix;
  fNdiag  = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
  fPtr    = matrix.GetMatrixArray();
  fInc    = matrix.GetNcols()+1;
}

//______________________________________________________________________________
TMatrixFDiag_const::TMatrixFDiag_const(const TMatrixFSym &matrix)
{
  Assert(matrix.IsValid());
  
  fMatrix = &matrix;
  fNdiag  = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
  fPtr    = matrix.GetMatrixArray();
  fInc    = matrix.GetNcols()+1;
}

//______________________________________________________________________________
TMatrixFDiag::TMatrixFDiag(TMatrixF &matrix)
             :TMatrixFDiag_const(matrix)
{
}

//______________________________________________________________________________
TMatrixFDiag::TMatrixFDiag(TMatrixFSym &matrix)
             :TMatrixFDiag_const(matrix)
{
}

//______________________________________________________________________________
TMatrixFDiag::TMatrixFDiag(const TMatrixFDiag &md) : TMatrixFDiag_const(md)
{
  *this = md;
}

//______________________________________________________________________________
void TMatrixFDiag::operator=(Float_t val)
{
  // Assign val to every element of the matrix diagonal.

  Assert(fMatrix->IsValid());
  Float_t *dp = const_cast<Float_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp = val;
}

//______________________________________________________________________________
void TMatrixFDiag::operator+=(Float_t val)
{
  // Assign val to every element of the matrix diagonal.

  Assert(fMatrix->IsValid());
  Float_t *dp = const_cast<Float_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp += val;
}

//______________________________________________________________________________
void TMatrixFDiag::operator*=(Float_t val)
{
  // Assign val to every element of the matrix diagonal.

  Assert(fMatrix->IsValid());
  Float_t *dp = const_cast<Float_t *>(fPtr);
  for (Int_t i = 0; i < fNdiag; i++, dp += fInc)
    *dp *= val;
}

//______________________________________________________________________________
void TMatrixFDiag::operator=(const TMatrixFDiag_const &md)
{
  const TMatrixFBase *mt = md.GetMatrix();
  if (fMatrix == mt) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (this->GetNdiags() != md.GetNdiags()) {
    Error("operator=(const TMatrixFDiag_const &)","diagonals not compatible");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *dp1 = const_cast<Float_t *>(fPtr);
  const Float_t *dp2 = md.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += fInc)
    *dp1 = *dp2;
}

//______________________________________________________________________________
void TMatrixFDiag::operator=(const TVectorF &vec)
{
  // Assign a vector to the matrix diagonal.

  Assert(fMatrix->IsValid());
  Assert(vec.IsValid());

  if (fNdiag != vec.GetNrows()) {
    Error("operator=(const TVectorF &)","vector length != matrix-diagonal length");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *dp = const_cast<Float_t *>(fPtr);
  const Float_t *vp = vec.GetMatrixArray();
  for ( ; vp < vec.GetMatrixArray()+vec.GetNrows(); dp += fInc)
    *dp = *vp++;
}

//______________________________________________________________________________
void TMatrixFDiag::operator+=(const TMatrixFDiag_const &md)
{
  // Add to every element of the matrix diagonal the
  // corresponding element of diagonal md.

  const TMatrixFBase *mt = md.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fNdiag != md.GetNdiags()) {
    Error("operator=(const TMatrixFDiag_const &)","matrix-diagonal's different length");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *dp1 = const_cast<Float_t *>(fPtr);
  const Float_t *dp2 = md.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += md.GetInc())
    *dp1 += *dp2;
}

//______________________________________________________________________________
void TMatrixFDiag::operator*=(const TMatrixFDiag_const &md)
{
  // Multiply every element of the matrix diagonal with the
  // corresponding element of diagonal md.

  const TMatrixFBase *mt = md.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fNdiag != md.GetNdiags()) {
    Error("operator*=(const TMatrixFDiag_const &)","matrix-diagonal's different length");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *dp1 = const_cast<Float_t *>(fPtr);
  const Float_t *dp2 = md.GetPtr();
  for (Int_t i = 0; i < fNdiag; i++, dp1 += fInc, dp2 += md.GetInc())
    *dp1 *= *dp2;
}

//______________________________________________________________________________
TMatrixFFlat_const::TMatrixFFlat_const(const TMatrixF &matrix)
{
  Assert(matrix.IsValid());

  fMatrix = &matrix;
  fPtr    = matrix.GetMatrixArray();
  fNelems = matrix.GetNoElements();
}

//______________________________________________________________________________
TMatrixFFlat_const::TMatrixFFlat_const(const TMatrixFSym &matrix)
{
  Assert(matrix.IsValid());

  fMatrix = &matrix;
  fPtr    = matrix.GetMatrixArray();
  fNelems = matrix.GetNoElements();
}

//______________________________________________________________________________
TMatrixFFlat::TMatrixFFlat(TMatrixF &matrix)
             :TMatrixFFlat_const(matrix)
{
}

//______________________________________________________________________________
TMatrixFFlat::TMatrixFFlat(TMatrixFSym &matrix)
             :TMatrixFFlat_const(matrix)
{
}

//______________________________________________________________________________
TMatrixFFlat::TMatrixFFlat(const TMatrixFFlat &mf) : TMatrixFFlat_const(mf)
{
  *this = mf;
}

//______________________________________________________________________________
void TMatrixFFlat::operator=(Float_t val)
{
  // Assign val to every element of the matrix.

  Assert(fMatrix->IsValid());
  Float_t *fp = const_cast<Float_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ = val;
}

//______________________________________________________________________________
void TMatrixFFlat::operator+=(Float_t val)
{
  // Add val to every element of the matrix.

  Assert(fMatrix->IsValid());
  Float_t *fp = const_cast<Float_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ += val;
}

//______________________________________________________________________________
void TMatrixFFlat::operator*=(Float_t val)
{
  // Multiply every element of the matrix with val.

  Assert(fMatrix->IsValid());
  Float_t *fp = const_cast<Float_t *>(fPtr);
  while (fp < fPtr+fMatrix->GetNoElements())
    *fp++ *= val;
}

//______________________________________________________________________________
void TMatrixFFlat::operator=(const TMatrixFFlat_const &mf)
{
  const TMatrixFBase *mt = mf.GetMatrix();
  if (fMatrix == mt) return;

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetNoElements() != mt->GetNoElements()) {
    Error("operator=(const TMatrixFFlat_const &)","matrix lengths different");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *fp1 = const_cast<Float_t *>(fPtr);
  const Float_t *fp2 = mf.GetPtr();
  while (fp1 < fPtr+fMatrix->GetNoElements())
    *fp1++ = *fp2++;
}

//______________________________________________________________________________
void TMatrixFFlat::operator=(const TVectorF &vec)
{
  // Assign a vector to the matrix. The matrix is traversed row-wise

  Assert(vec.IsValid());

  if (fMatrix->GetNoElements() != vec.GetNrows()) {
    Error("operator=(const TVectorF &)","vector length != # matrix-elements");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *fp = const_cast<Float_t *>(fPtr);
  const Float_t *vp = vec.GetMatrixArray();
  while (fp < fPtr+fMatrix->GetNoElements())
     *fp++ = *vp++;
}

//______________________________________________________________________________
void TMatrixFFlat::operator+=(const TMatrixFFlat_const &mf)
{
  // Add to every element of the matrix the corresponding element of matrix mf.

  const TMatrixFBase *mt = mf.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetNoElements() != mt->GetNoElements()) {
    Error("operator+=(const TMatrixFFlat_const &)","matrices lengths different");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *fp1 = const_cast<Float_t *>(fPtr);
  const Float_t *fp2 = mf.GetPtr();
  while (fp1 < fPtr + fMatrix->GetNoElements())
    *fp1++ += *fp2++;
}

//______________________________________________________________________________
void TMatrixFFlat::operator*=(const TMatrixFFlat_const &mf)
{
  // Multiply every element of the matrix with the corresponding element of diagonal mf.

  const TMatrixFBase *mt = mf.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());
  if (fMatrix->GetNoElements() != mt->GetNoElements()) {
    Error("operator*=(const TMatrixFFlat_const &)","matrices lengths different");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *fp1 = const_cast<Float_t *>(fPtr);
  const Float_t *fp2 = mf.GetPtr();
  while (fp1 < fPtr + fMatrix->GetNoElements())
    *fp1++ *= *fp2++;
}

//______________________________________________________________________________
TMatrixFSub_const::TMatrixFSub_const(const TMatrixF &matrix,Int_t row_lwbs,Int_t row_upbs,
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
TMatrixFSub_const::TMatrixFSub_const(const TMatrixFSym &matrix,Int_t row_lwbs,Int_t row_upbs,
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
TMatrixFSub::TMatrixFSub(TMatrixF &matrix,Int_t row_lwbs,Int_t row_upbs,
                         Int_t col_lwbs,Int_t col_upbs)
            :TMatrixFSub_const(matrix,row_lwbs,row_upbs,col_lwbs,col_upbs)
{
}

//______________________________________________________________________________
TMatrixFSub::TMatrixFSub(TMatrixFSym &matrix,Int_t row_lwbs,Int_t row_upbs,
                         Int_t col_lwbs,Int_t col_upbs)
            :TMatrixFSub_const(matrix,row_lwbs,row_upbs,col_lwbs,col_upbs)
{
}

//______________________________________________________________________________
TMatrixFSub::TMatrixFSub(const TMatrixFSub &ms) : TMatrixFSub_const(ms)
{
  *this = ms;
}

//______________________________________________________________________________
void TMatrixFSub::operator=(Float_t val)
{
  // Assign val to every element of the sub matrix.

  Assert(fMatrix->IsValid());

  Float_t *p = (const_cast<TMatrixFBase *>(fMatrix))->GetMatrixArray();
  const Int_t ncols = fMatrix->GetNcols();
  for (Int_t irow = 0; irow < fNrowsSub; irow++) {
    const Int_t off = (irow+fRowOff)*ncols+fColOff;
    for (Int_t icol = 0; icol < fNcolsSub; icol++)
      p[off+icol] = val;
  }
}

//______________________________________________________________________________
void TMatrixFSub::operator+=(Float_t val)
{
  // Add val to every element of the sub matrix.

  Assert(fMatrix->IsValid());

  Float_t *p = (const_cast<TMatrixFBase *>(fMatrix))->GetMatrixArray();
  const Int_t ncols = fMatrix->GetNcols();
  for (Int_t irow = 0; irow < fNrowsSub; irow++) {
    const Int_t off = (irow+fRowOff)*ncols+fColOff;
    for (Int_t icol = 0; icol < fNcolsSub; icol++)
      p[off+icol] = val;
  }
}

//______________________________________________________________________________
void TMatrixFSub::operator*=(Float_t val)
{
  // Multiply every element of the sub matrix by val .

  Assert(fMatrix->IsValid());

  Float_t *p = (const_cast<TMatrixFBase *>(fMatrix))->GetMatrixArray();
  const Int_t ncols = fMatrix->GetNcols();
  for (Int_t irow = 0; irow < fNrowsSub; irow++) {
    const Int_t off = (irow+fRowOff)*ncols+fColOff;
    for (Int_t icol = 0; icol < fNcolsSub; icol++)
      p[off+icol] = val;
  }
}

//______________________________________________________________________________
void TMatrixFSub::operator=(const TMatrixFSub_const &ms)
{
  const TMatrixFBase *mt = ms.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (fMatrix == mt &&
      (GetNrows()  == ms.GetNrows () && GetNcols()  == ms.GetNcols () &&
       GetRowOff() == ms.GetRowOff() && GetColOff() == ms.GetColOff()) )
    return;

  if (GetNrows() != ms.GetNrows() || GetNcols() != ms.GetNcols()) {
    Error("operator=(const TMatrixFSub_const &)","sub matrices have different size");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  const Int_t rowOff2 = ms.GetRowOff();
  const Int_t colOff2 = ms.GetColOff();

  Bool_t overlap = (fMatrix == mt) &&
                   ( (rowOff2 >= fRowOff && rowOff2 < fRowOff+fNrowsSub) ||
                     (colOff2 >= fColOff && colOff2 < fColOff+fNcolsSub) );

  Float_t *p1 = const_cast<Float_t *>(fMatrix->GetMatrixArray());
  if (!overlap) {
    const Float_t *p2 = mt->GetMatrixArray();

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
    TMatrixF tmp; mt->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,tmp);
    const Float_t *p2 = tmp.GetMatrixArray();

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
void TMatrixFSub::operator=(const TMatrixFBase &m)
{
  Assert(fMatrix->IsValid());
  Assert(m.IsValid());

  if (fMatrix == &m) return;

  if (fNrowsSub != m.GetNrows() || fNcolsSub != m.GetNcols()) {
    Error("operator=(const TMatrixFBase &)","sub matrices and matrix have different size"); 
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }
  const Int_t row_lwbs = fRowOff+fMatrix->GetRowLwb();
  const Int_t col_lwbs = fColOff+fMatrix->GetColLwb();
  (const_cast<TMatrixFBase *>(fMatrix))->SetSub(row_lwbs,col_lwbs,m);
}

//______________________________________________________________________________
void TMatrixFSub::operator+=(const TMatrixFSub_const &ms)
{
  const TMatrixFBase *mt = ms.GetMatrix();

  Assert(fMatrix->IsValid());
  Assert(mt->IsValid());

  if (GetNrows() != ms.GetNrows() || GetNcols() != ms.GetNcols()) {
    Error("operator+=(const TMatrixFSub_const &)","sub matrices have different size");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  const Int_t rowOff2 = ms.GetRowOff();
  const Int_t colOff2 = ms.GetColOff();

  Bool_t overlap = (fMatrix == mt) &&
                   ( (rowOff2 >= fRowOff && rowOff2 < fRowOff+fNrowsSub) ||
                     (colOff2 >= fColOff && colOff2 < fColOff+fNcolsSub) );

  Float_t *p1 = const_cast<Float_t *>(fMatrix->GetMatrixArray());
  if (!overlap) {
    const Float_t *p2 = mt->GetMatrixArray();

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
    TMatrixF tmp; mt->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,tmp);
    const Float_t *p2 = tmp.GetMatrixArray();

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
void TMatrixFSub::operator*=(const TMatrixFSub_const &ms)
{
  if (fNcolsSub != ms.GetNrows() || fNcolsSub != ms.GetNcols()) {
    Error("operator*=(const TMatrixFSub_const &)","source sub matrix has wrong shape");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  const TMatrixFBase *source = ms.GetMatrix();

  TMatrixF source_sub;
  {
    const Int_t row_lwbs = ms.GetRowOff()+source->GetRowLwb();
    const Int_t row_upbs = row_lwbs+fNrowsSub-1;
    const Int_t col_lwbs = ms.GetColOff()+source->GetColLwb();
    const Int_t col_upbs = col_lwbs+fNcolsSub-1;
    source->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,source_sub);
  }
  
  const Float_t *sp = source_sub.GetMatrixArray();
  const Int_t ncols = fMatrix->GetNcols();

  // One row of the old_target matrix
  Float_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Float_t *trp = work;
  if (fNcolsSub > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Float_t[fNcolsSub];
  }

        Float_t *cp   = const_cast<Float_t *>(fMatrix->GetMatrixArray())+fRowOff*ncols+fColOff;
  const Float_t *trp0 = cp; // Pointer to  target[i,0];
  const Float_t * const trp0_last = trp0+fNrowsSub*ncols;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcolsSub*sizeof(Float_t));         // copy the i-th row of target, Start at target[i,0]
    for (const Float_t *scp = sp; scp < sp+fNcolsSub; ) {  // Pointer to the j-th column of source,
                                                         // Start scp = source[0,0]
      Float_t cij = 0;
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
void TMatrixFSub::operator+=(const TMatrixFBase &mt)
{
  Assert(fMatrix->IsValid());
  Assert(mt.IsValid());

  if (GetNrows() != mt.GetNrows() || GetNcols() != mt.GetNcols()) {
    Error("operator+=(const TMatrixFBase &)","sub matrix and matrix have different size");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  Float_t *p1 = const_cast<Float_t *>(fMatrix->GetMatrixArray());
  const Float_t *p2 = mt.GetMatrixArray();

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
void TMatrixFSub::operator*=(const TMatrixF &source)
{
  if (fNcolsSub != source.GetNrows() || fNcolsSub != source.GetNcols()) {
    Error("operator*=(const TMatrixF &)","source matrix has wrong shape");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  // Check for A *= A;
  const Float_t *sp;
  TMatrixF tmp;
  if (fMatrix == &source) {
    tmp.ResizeTo(source);
    tmp = source;
    sp = tmp.GetMatrixArray();
  }
  else
    sp = source.GetMatrixArray();

  const Int_t ncols = fMatrix->GetNcols();

  // One row of the old_target matrix
  Float_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Float_t *trp = work;
  if (fNcolsSub > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Float_t[fNcolsSub];
  }

        Float_t *cp   = const_cast<Float_t *>(fMatrix->GetMatrixArray())+fRowOff*ncols+fColOff;
  const Float_t *trp0 = cp;                               // Pointer to  target[i,0];
  const Float_t * const trp0_last = trp0+fNrowsSub*ncols;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcolsSub*sizeof(Float_t));           // copy the i-th row of target, Start at target[i,0]
    for (const Float_t *scp = sp; scp < sp+fNcolsSub; ) { // Pointer to the j-th column of source,
                                                           // Start scp = source[0,0]
      Float_t cij = 0;
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
void TMatrixFSub::operator*=(const TMatrixFSym &source)
{
  if (fNcolsSub != source.GetNrows() || fNcolsSub != source.GetNcols()) {
    Error("operator*=(const TMatrixFSym &)","source matrix has wrong shape");
    (const_cast<TMatrixFBase *>(fMatrix))->Invalidate();
    return;
  }

  // Check for A *= A;
  const Float_t *sp;
  TMatrixFSym tmp;
  if ((TMatrixFSym *)fMatrix == &source) {
    tmp.ResizeTo(source);
    tmp = source;
    sp = tmp.GetMatrixArray();
  }
  else
    sp = source.GetMatrixArray();

  const Int_t ncols = fMatrix->GetNcols();

  // One row of the old_target matrix
  Float_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Float_t *trp = work;
  if (fNcolsSub > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Float_t[fNcolsSub];
  }

        Float_t *cp   = const_cast<Float_t *>(fMatrix->GetMatrixArray())+fRowOff*ncols+fColOff;
  const Float_t *trp0 = cp;                               // Pointer to  target[i,0];
  const Float_t * const trp0_last = trp0+fNrowsSub*ncols;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcolsSub*sizeof(Float_t));           // copy the i-th row of target, Start at target[i,0]
    for (const Float_t *scp = sp; scp < sp+fNcolsSub; ) { // Pointer to the j-th column of source,
                                                           // Start scp = source[0,0]
      Float_t cij = 0;
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
Float_t Frand(Double_t &ix)
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

  return Float_t(ix*4.656612875e-10);
}
