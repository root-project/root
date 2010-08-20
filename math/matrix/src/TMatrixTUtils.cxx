// @(#)root/matrix:$Id$
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
// Templates of utility classes in the Linear Algebra Package.          //
// The following classes are defined here:                              //
//                                                                      //
// Different matrix views without copying data elements :               //
//   TMatrixTRow_const        TMatrixTRow                               //
//   TMatrixTColumn_const     TMatrixTColumn                            //
//   TMatrixTDiag_const       TMatrixTDiag                              //
//   TMatrixTFlat_const       TMatrixTFlat                              //
//   TMatrixTSub_const        TMatrixTSub                               //
//   TMatrixTSparseRow_const  TMatrixTSparseRow                         //
//   TMatrixTSparseDiag_const TMatrixTSparseDiag                        //
//                                                                      //
//   TElementActionT                                                    //
//   TElementPosActionT                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixTUtils.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TMatrixTSparse.h"
#include "TMath.h"
#include "TVectorT.h"

//______________________________________________________________________________
template<class Element>
TMatrixTRow_const<Element>::TMatrixTRow_const(const TMatrixT<Element> &matrix,Int_t row)
{
// Constructor with row "row" of matrix

   R__ASSERT(matrix.IsValid());

   fRowInd = row-matrix.GetRowLwb();
   if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
      Error("TMatrixTRow_const(const TMatrixT<Element> &,Int_t)","row index out of bounds");
      fMatrix = 0;
      fPtr = 0;
      fInc = 0;
      return;
   }
 
   fMatrix = &matrix;
   fPtr = matrix.GetMatrixArray()+fRowInd*matrix.GetNcols();
   fInc = 1;
}

//______________________________________________________________________________
template<class Element>
TMatrixTRow_const<Element>::TMatrixTRow_const(const TMatrixTSym<Element> &matrix,Int_t row)
{
// Constructor with row "row" of symmetric matrix

   R__ASSERT(matrix.IsValid());

   fRowInd = row-matrix.GetRowLwb();
   if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
      Error("TMatrixTRow_const(const TMatrixTSym &,Int_t)","row index out of bounds");
      fMatrix = 0;
      fPtr = 0;
      fInc = 0;
      return;
   }

   fMatrix = &matrix;
   fPtr = matrix.GetMatrixArray()+fRowInd*matrix.GetNcols();
   fInc = 1;
}

//______________________________________________________________________________
template<class Element>
TMatrixTRow<Element>::TMatrixTRow(TMatrixT<Element> &matrix,Int_t row)
            :TMatrixTRow_const<Element>(matrix,row)
{
// Constructor with row "row" of symmetric matrix
}

//______________________________________________________________________________
template<class Element>
TMatrixTRow<Element>::TMatrixTRow(TMatrixTSym<Element> &matrix,Int_t row)
            :TMatrixTRow_const<Element>(matrix,row)
{
// Constructor with row "row" of symmetric matrix
}

//______________________________________________________________________________
template<class Element>
TMatrixTRow<Element>::TMatrixTRow(const TMatrixTRow<Element> &mr) : TMatrixTRow_const<Element>(mr)
{
// Copy constructor

   *this = mr;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTRow<Element>::operator=(Element val)
{
// Assign val to every element of the matrix row.

   R__ASSERT(this->fMatrix->IsValid());
   Element *rp = const_cast<Element *>(this->fPtr);
   for ( ; rp < this->fPtr+this->fMatrix->GetNcols(); rp += this->fInc)
      *rp = val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTRow<Element>::operator+=(Element val)
{
// Add val to every element of the matrix row.

   R__ASSERT(this->fMatrix->IsValid());
   Element *rp = const_cast<Element *>(this->fPtr);
   for ( ; rp < this->fPtr+this->fMatrix->GetNcols(); rp += this->fInc)
      *rp += val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTRow<Element>::operator*=(Element val)
{
// Multiply every element of the matrix row with val.

   R__ASSERT(this->fMatrix->IsValid());
   Element *rp = const_cast<Element *>(this->fPtr);
   for ( ; rp < this->fPtr + this->fMatrix->GetNcols(); rp += this->fInc)
      *rp *= val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTRow<Element>::operator=(const TMatrixTRow_const<Element> &mr)
{
// Assignment operator

   const TMatrixTBase<Element> *mt = mr.GetMatrix();
   if (this->fMatrix->GetMatrixArray() == mt->GetMatrixArray() && this->fRowInd == mr.GetRowIndex()) return;

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->fMatrix->GetNcols() != mt->GetNcols() || this->fMatrix->GetColLwb() != mt->GetColLwb()) {
      Error("operator=(const TMatrixTRow_const &)", "matrix rows not compatible");
      return;
   }

   Element *rp1 = const_cast<Element *>(this->fPtr);
   const Element *rp2 = mr.GetPtr();
   for ( ; rp1 < this->fPtr+this->fMatrix->GetNcols(); rp1 += this->fInc,rp2 += mr.GetInc())
      *rp1 = *rp2;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTRow<Element>::operator=(const TVectorT<Element> &vec)
{
// Assign a vector to a matrix row. The vector is considered row-vector
// to allow the assignment in the strict sense.

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(vec.IsValid());

   if (this->fMatrix->GetColLwb() != vec.GetLwb() || this->fMatrix->GetNcols() != vec.GetNrows()) {
      Error("operator=(const TVectorT &)","vector length != matrix-row length");
      return;
   }

   Element *rp = const_cast<Element *>(this->fPtr);
   const Element *vp = vec.GetMatrixArray();
   for ( ; rp < this->fPtr+this->fMatrix->GetNcols(); rp += this->fInc)
      *rp = *vp++;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTRow<Element>::operator+=(const TMatrixTRow_const<Element> &r)
{
// Add to every element of the matrix row the corresponding element of row r.

  const TMatrixTBase<Element> *mt = r.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->fMatrix->GetColLwb() != mt->GetColLwb() || this->fMatrix->GetNcols() != mt->GetNcols()) {
      Error("operator+=(const TMatrixTRow_const &)","different row lengths");
      return;
   }

   Element *rp1 = const_cast<Element *>(this->fPtr);
   const Element *rp2 = r.GetPtr();
   for ( ; rp1 < this->fPtr+this->fMatrix->GetNcols(); rp1 += this->fInc,rp2 += r.GetInc())
     *rp1 += *rp2;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTRow<Element>::operator*=(const TMatrixTRow_const<Element> &r)
{
// Multiply every element of the matrix row with the
// corresponding element of row r.

   const TMatrixTBase<Element> *mt = r.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->fMatrix->GetColLwb() != mt->GetColLwb() || this->fMatrix->GetNcols() != mt->GetNcols()) {
      Error("operator*=(const TMatrixTRow_const &)","different row lengths");
      return;
   }

   Element *rp1 = const_cast<Element *>(this->fPtr);
   const Element *rp2 = r.GetPtr();
   for ( ; rp1 < this->fPtr+this->fMatrix->GetNcols(); rp1 += this->fInc,rp2 += r.GetInc())
      *rp1 *= *rp2;
}

//______________________________________________________________________________
template<class Element>
TMatrixTColumn_const<Element>::TMatrixTColumn_const(const TMatrixT<Element> &matrix,Int_t col)
{
// Constructor with column "col" of matrix

   R__ASSERT(matrix.IsValid());

   this->fColInd = col-matrix.GetColLwb();
   if (this->fColInd >= matrix.GetNcols() || this->fColInd < 0) {
      Error("TMatrixTColumn_const(const TMatrixT &,Int_t)","column index out of bounds");
      fMatrix = 0;
      fPtr = 0;
      fInc = 0;
      return;
   }

   fMatrix = &matrix;
   fPtr = matrix.GetMatrixArray()+fColInd;
   fInc = matrix.GetNcols();
}

//______________________________________________________________________________
template<class Element>
TMatrixTColumn_const<Element>::TMatrixTColumn_const(const TMatrixTSym<Element> &matrix,Int_t col)
{
// Constructor with column "col" of matrix

   R__ASSERT(matrix.IsValid());

   fColInd = col-matrix.GetColLwb();
   if (fColInd >= matrix.GetNcols() || fColInd < 0) {
      Error("TMatrixTColumn_const(const TMatrixTSym &,Int_t)","column index out of bounds");
      fMatrix = 0;
      fPtr = 0;
      fInc = 0;
      return;
   }

   fMatrix = &matrix;
   fPtr = matrix.GetMatrixArray()+fColInd;
   fInc = matrix.GetNcols();
}

//______________________________________________________________________________
template<class Element>
TMatrixTColumn<Element>::TMatrixTColumn(TMatrixT<Element> &matrix,Int_t col)
               :TMatrixTColumn_const<Element>(matrix,col)
{
// Constructor with column "col" of matrix
}

//______________________________________________________________________________
template<class Element>
TMatrixTColumn<Element>::TMatrixTColumn(TMatrixTSym<Element> &matrix,Int_t col)
               :TMatrixTColumn_const<Element>(matrix,col)
{
// Constructor with column "col" of matrix
}

//______________________________________________________________________________
template<class Element>
TMatrixTColumn<Element>::TMatrixTColumn(const TMatrixTColumn<Element> &mc) : TMatrixTColumn_const<Element>(mc)
{
// Copy constructor

   *this = mc;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTColumn<Element>::operator=(Element val)
{
// Assign val to every element of the matrix column.

   R__ASSERT(this->fMatrix->IsValid());
   Element *cp = const_cast<Element *>(this->fPtr);
   for ( ; cp < this->fPtr+this->fMatrix->GetNoElements(); cp += this->fInc)
      *cp = val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTColumn<Element>::operator+=(Element val)
{
// Add val to every element of the matrix column.

   R__ASSERT(this->fMatrix->IsValid());
   Element *cp = const_cast<Element *>(this->fPtr);
   for ( ; cp < this->fPtr+this->fMatrix->GetNoElements(); cp += this->fInc)
      *cp += val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTColumn<Element>::operator*=(Element val)
{
// Multiply every element of the matrix column with val.

   R__ASSERT(this->fMatrix->IsValid());
   Element *cp = const_cast<Element *>(this->fPtr);
   for ( ; cp < this->fPtr+this->fMatrix->GetNoElements(); cp += this->fInc)
      *cp *= val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTColumn<Element>::operator=(const TMatrixTColumn_const<Element> &mc)
{
// Assignment operator

   const TMatrixTBase<Element> *mt = mc.GetMatrix();
   if (this->fMatrix->GetMatrixArray() == mt->GetMatrixArray() && this->fColInd == mc.GetColIndex()) return;

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->fMatrix->GetNrows() != mt->GetNrows() || this->fMatrix->GetRowLwb() != mt->GetRowLwb()) {
      Error("operator=(const TMatrixTColumn_const &)", "matrix columns not compatible");
      return;
   }

   Element *cp1 = const_cast<Element *>(this->fPtr);
   const Element *cp2 = mc.GetPtr();
   for ( ; cp1 < this->fPtr+this->fMatrix->GetNoElements(); cp1 += this->fInc,cp2 += mc.GetInc())
      *cp1 = *cp2;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTColumn<Element>::operator=(const TVectorT<Element> &vec)
{
// Assign a vector to a matrix column.

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(vec.IsValid());

   if (this->fMatrix->GetRowLwb() != vec.GetLwb() || this->fMatrix->GetNrows() != vec.GetNrows()) {
      Error("operator=(const TVectorT &)","vector length != matrix-column length");
      return;
   }

   Element *cp = const_cast<Element *>(this->fPtr);
   const Element *vp = vec.GetMatrixArray();
   for ( ; cp < this->fPtr+this->fMatrix->GetNoElements(); cp += this->fInc)
      *cp = *vp++;

   R__ASSERT(vp == vec.GetMatrixArray()+vec.GetNrows());
}

//______________________________________________________________________________
template<class Element>
void TMatrixTColumn<Element>::operator+=(const TMatrixTColumn_const<Element> &mc)
{
// Add to every element of the matrix row the corresponding element of row mc.

   const TMatrixTBase<Element> *mt = mc.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->fMatrix->GetRowLwb() != mt->GetRowLwb() || this->fMatrix->GetNrows() != mt->GetNrows()) {
      Error("operator+=(const TMatrixTColumn_const &)","different row lengths");
      return;
   }

   Element *cp1 = const_cast<Element *>(this->fPtr);
   const Element *cp2 = mc.GetPtr();
   for ( ; cp1 < this->fPtr+this->fMatrix->GetNoElements(); cp1 += this->fInc,cp2 += mc.GetInc())
      *cp1 += *cp2;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTColumn<Element>::operator*=(const TMatrixTColumn_const<Element> &mc)
{
// Multiply every element of the matrix column with the
// corresponding element of column mc.

   const TMatrixTBase<Element> *mt = mc.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->fMatrix->GetRowLwb() != mt->GetRowLwb() || this->fMatrix->GetNrows() != mt->GetNrows()) {
      Error("operator*=(const TMatrixTColumn_const &)","different row lengths");
      return;
   }

   Element *cp1 = const_cast<Element *>(this->fPtr);
   const Element *cp2 = mc.GetPtr();
   for ( ; cp1 < this->fPtr+this->fMatrix->GetNoElements(); cp1 += this->fInc,cp2 += mc.GetInc())
      *cp1 *= *cp2;
}

//______________________________________________________________________________
template<class Element>
TMatrixTDiag_const<Element>::TMatrixTDiag_const(const TMatrixT<Element> &matrix)
{
// Constructor

   R__ASSERT(matrix.IsValid());

   fMatrix = &matrix;
   fNdiag  = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
   fPtr    = matrix.GetMatrixArray();
   fInc    = matrix.GetNcols()+1;
}

//______________________________________________________________________________
template<class Element>
TMatrixTDiag_const<Element>::TMatrixTDiag_const(const TMatrixTSym<Element> &matrix)
{
// Constructor

   R__ASSERT(matrix.IsValid());

   fMatrix = &matrix;
   fNdiag  = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
   fPtr    = matrix.GetMatrixArray();
   fInc    = matrix.GetNcols()+1;
}

//______________________________________________________________________________
template<class Element>
TMatrixTDiag<Element>::TMatrixTDiag(TMatrixT<Element> &matrix)
             :TMatrixTDiag_const<Element>(matrix)
{
// Constructor
}

//______________________________________________________________________________
template<class Element>
TMatrixTDiag<Element>::TMatrixTDiag(TMatrixTSym<Element> &matrix)
             :TMatrixTDiag_const<Element>(matrix)
{
// Constructor
}

//______________________________________________________________________________
template<class Element>
TMatrixTDiag<Element>::TMatrixTDiag(const TMatrixTDiag<Element> &md) : TMatrixTDiag_const<Element>(md)
{
// Copy constructor

   *this = md;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTDiag<Element>::operator=(Element val)
{
// Assign val to every element of the matrix diagonal.

   R__ASSERT(this->fMatrix->IsValid());
   Element *dp = const_cast<Element *>(this->fPtr);
   for (Int_t i = 0; i < this->fNdiag; i++, dp += this->fInc)
      *dp = val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTDiag<Element>::operator+=(Element val)
{
// Assign val to every element of the matrix diagonal.

   R__ASSERT(this->fMatrix->IsValid());
   Element *dp = const_cast<Element *>(this->fPtr);
   for (Int_t i = 0; i < this->fNdiag; i++, dp += this->fInc)
      *dp += val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTDiag<Element>::operator*=(Element val)
{
// Assign val to every element of the matrix diagonal.

   R__ASSERT(this->fMatrix->IsValid());
   Element *dp = const_cast<Element *>(this->fPtr);
   for (Int_t i = 0; i < this->fNdiag; i++, dp += this->fInc)
      *dp *= val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTDiag<Element>::operator=(const TMatrixTDiag_const<Element> &md)
{
// Assignment operator

   const TMatrixTBase<Element> *mt = md.GetMatrix();
   if (this->fMatrix == mt) return;

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->GetNdiags() != md.GetNdiags()) {
      Error("operator=(const TMatrixTDiag_const &)","diagonals not compatible");
      return;
   }

   Element *dp1 = const_cast<Element *>(this->fPtr);
   const Element *dp2 = md.GetPtr();
   for (Int_t i = 0; i < this->fNdiag; i++, dp1 += this->fInc, dp2 += md.GetInc())
      *dp1 = *dp2;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTDiag<Element>::operator=(const TVectorT<Element> &vec)
{
// Assign a vector to the matrix diagonal.

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(vec.IsValid());

   if (this->fNdiag != vec.GetNrows()) {
      Error("operator=(const TVectorT &)","vector length != matrix-diagonal length");
      return;
   }

   Element *dp = const_cast<Element *>(this->fPtr);
   const Element *vp = vec.GetMatrixArray();
   for ( ; vp < vec.GetMatrixArray()+vec.GetNrows(); dp += this->fInc)
      *dp = *vp++;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTDiag<Element>::operator+=(const TMatrixTDiag_const<Element> &md)
{
// Add to every element of the matrix diagonal the
// corresponding element of diagonal md.

   const TMatrixTBase<Element> *mt = md.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fNdiag != md.GetNdiags()) {
      Error("operator=(const TMatrixTDiag_const &)","matrix-diagonal's different length");
      return;
   }

   Element *dp1 = const_cast<Element *>(this->fPtr);
   const Element *dp2 = md.GetPtr();
   for (Int_t i = 0; i < this->fNdiag; i++, dp1 += this->fInc, dp2 += md.GetInc())
      *dp1 += *dp2;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTDiag<Element>::operator*=(const TMatrixTDiag_const<Element> &md)
{
// Multiply every element of the matrix diagonal with the
// corresponding element of diagonal md.

   const TMatrixTBase<Element> *mt = md.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fNdiag != md.GetNdiags()) {
      Error("operator*=(const TMatrixTDiag_const &)","matrix-diagonal's different length");
      return;
   }

   Element *dp1 = const_cast<Element *>(this->fPtr);
   const Element *dp2 = md.GetPtr();
   for (Int_t i = 0; i < this->fNdiag; i++, dp1 += this->fInc, dp2 += md.GetInc())
      *dp1 *= *dp2;
}

//______________________________________________________________________________
template<class Element>
TMatrixTFlat_const<Element>::TMatrixTFlat_const(const TMatrixT<Element> &matrix)
{
// Constructor

   R__ASSERT(matrix.IsValid());

   fMatrix = &matrix;
   fPtr    = matrix.GetMatrixArray();
   fNelems = matrix.GetNoElements();
}

//______________________________________________________________________________
template<class Element>
TMatrixTFlat_const<Element>::TMatrixTFlat_const(const TMatrixTSym<Element> &matrix)
{
// Constructor

   R__ASSERT(matrix.IsValid());

   fMatrix = &matrix;
   fPtr    = matrix.GetMatrixArray();
   fNelems = matrix.GetNoElements();
}

//______________________________________________________________________________
template<class Element>
TMatrixTFlat<Element>::TMatrixTFlat(TMatrixT<Element> &matrix)
             :TMatrixTFlat_const<Element>(matrix)
{
// Constructor
}

//______________________________________________________________________________
template<class Element>
TMatrixTFlat<Element>::TMatrixTFlat(TMatrixTSym<Element> &matrix)
             :TMatrixTFlat_const<Element>(matrix)
{
// Constructor
}

//______________________________________________________________________________
template<class Element>
TMatrixTFlat<Element>::TMatrixTFlat(const TMatrixTFlat<Element> &mf) : TMatrixTFlat_const<Element>(mf)
{
// Copy constructor

   *this = mf;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTFlat<Element>::operator=(Element val)
{
// Assign val to every element of the matrix.

   R__ASSERT(this->fMatrix->IsValid());
   Element *fp = const_cast<Element *>(this->fPtr);
   while (fp < this->fPtr+this->fMatrix->GetNoElements())
      *fp++ = val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTFlat<Element>::operator+=(Element val)
{
// Add val to every element of the matrix.

   R__ASSERT(this->fMatrix->IsValid());
   Element *fp = const_cast<Element *>(this->fPtr);
   while (fp < this->fPtr+this->fMatrix->GetNoElements())
      *fp++ += val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTFlat<Element>::operator*=(Element val)
{
// Multiply every element of the matrix with val.

   R__ASSERT(this->fMatrix->IsValid());
   Element *fp = const_cast<Element *>(this->fPtr);
   while (fp < this->fPtr+this->fMatrix->GetNoElements())
      *fp++ *= val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTFlat<Element>::operator=(const TMatrixTFlat_const<Element> &mf)
{
// Assignment operator

   const TMatrixTBase<Element> *mt = mf.GetMatrix();
   if (this->fMatrix->GetMatrixArray() == mt->GetMatrixArray()) return;

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fMatrix->GetNoElements() != mt->GetNoElements()) {
      Error("operator=(const TMatrixTFlat_const &)","matrix lengths different");
      return;
   }

   Element *fp1 = const_cast<Element *>(this->fPtr);
   const Element *fp2 = mf.GetPtr();
   while (fp1 < this->fPtr+this->fMatrix->GetNoElements())
      *fp1++ = *fp2++;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTFlat<Element>::operator=(const TVectorT<Element> &vec)
{
// Assign a vector to the matrix. The matrix is traversed row-wise

   R__ASSERT(vec.IsValid());

   if (this->fMatrix->GetNoElements() != vec.GetNrows()) {
      Error("operator=(const TVectorT &)","vector length != # matrix-elements");
      return;
   }

   Element *fp = const_cast<Element *>(this->fPtr);
   const Element *vp = vec.GetMatrixArray();
   while (fp < this->fPtr+this->fMatrix->GetNoElements())
       *fp++ = *vp++;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTFlat<Element>::operator+=(const TMatrixTFlat_const<Element> &mf)
{
// Add to every element of the matrix the corresponding element of matrix mf.

   const TMatrixTBase<Element> *mt = mf.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fMatrix->GetNoElements() != mt->GetNoElements()) {
      Error("operator+=(const TMatrixTFlat_const &)","matrices lengths different");
      return;
   }

   Element *fp1 = const_cast<Element *>(this->fPtr);
   const Element *fp2 = mf.GetPtr();
   while (fp1 < this->fPtr + this->fMatrix->GetNoElements())
      *fp1++ += *fp2++;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTFlat<Element>::operator*=(const TMatrixTFlat_const<Element> &mf)
{
// Multiply every element of the matrix with the corresponding element of diagonal mf.

   const TMatrixTBase<Element> *mt = mf.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fMatrix->GetNoElements() != mt->GetNoElements()) {
      Error("operator*=(const TMatrixTFlat_const &)","matrices lengths different");
      return;
   }

   Element *fp1 = const_cast<Element *>(this->fPtr);
   const Element *fp2 = mf.GetPtr();
   while (fp1 < this->fPtr + this->fMatrix->GetNoElements())
      *fp1++ *= *fp2++;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSub_const<Element>::TMatrixTSub_const(const TMatrixT<Element> &matrix,Int_t row_lwbs,Int_t row_upbs,
                                              Int_t col_lwbs,Int_t col_upbs)
{
// make a reference to submatrix [row_lwbs..row_upbs][col_lwbs..col_upbs];
// The indexing range of the reference is
// [0..row_upbs-row_lwbs+1][0..col_upb-col_lwbs+1] (default)

   R__ASSERT(matrix.IsValid());

   fRowOff    = 0;
   fColOff    = 0;
   fNrowsSub  = 0;
   fNcolsSub  = 0;
   fMatrix = &matrix;

   if (row_upbs < row_lwbs) {
      Error("TMatrixTSub_const","Request sub-matrix with row_upbs(%d) < row_lwbs(%d)",row_upbs,row_lwbs);
      return;
   }
   if (col_upbs < col_lwbs) {
      Error("TMatrixTSub_const","Request sub-matrix with col_upbs(%d) < col_lwbs(%d)",col_upbs,col_lwbs);
      return;
   }

   const Int_t rowLwb = matrix.GetRowLwb();
   const Int_t rowUpb = matrix.GetRowUpb();
   const Int_t colLwb = matrix.GetColLwb();
   const Int_t colUpb = matrix.GetColUpb();

   if (row_lwbs < rowLwb || row_lwbs > rowUpb) {
      Error("TMatrixTSub_const","Request row_lwbs sub-matrix(%d) outside matrix range of %d - %d",row_lwbs,rowLwb,rowUpb);
      return;
   }
   if (col_lwbs < colLwb || col_lwbs > colUpb) {
      Error("TMatrixTSub_const","Request col_lwbs sub-matrix(%d) outside matrix range of %d - %d",col_lwbs,colLwb,colUpb);
      return;
   }
   if (row_upbs < rowLwb || row_upbs > rowUpb) {
      Error("TMatrixTSub_const","Request row_upbs sub-matrix(%d) outside matrix range of %d - %d",row_upbs,rowLwb,rowUpb);
      return;
   }
   if (col_upbs < colLwb || col_upbs > colUpb) {
      Error("TMatrixTSub_const","Request col_upbs sub-matrix(%d) outside matrix range of %d - %d",col_upbs,colLwb,colUpb);
      return;
   }

   fRowOff    = row_lwbs-rowLwb;
   fColOff    = col_lwbs-colLwb;
   fNrowsSub  = row_upbs-row_lwbs+1;
   fNcolsSub  = col_upbs-col_lwbs+1;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSub_const<Element>::TMatrixTSub_const(const TMatrixTSym<Element> &matrix,Int_t row_lwbs,Int_t row_upbs,
                                              Int_t col_lwbs,Int_t col_upbs)
{
// make a reference to submatrix [row_lwbs..row_upbs][col_lwbs..col_upbs];
// The indexing range of the reference is
// [0..row_upbs-row_lwbs+1][0..col_upb-col_lwbs+1] (default)

   R__ASSERT(matrix.IsValid());

   fRowOff    = 0;
   fColOff    = 0;
   fNrowsSub  = 0;
   fNcolsSub  = 0;
   fMatrix    = &matrix;

   if (row_upbs < row_lwbs) {
      Error("TMatrixTSub_const","Request sub-matrix with row_upbs(%d) < row_lwbs(%d)",row_upbs,row_lwbs);
      return;
   }
   if (col_upbs < col_lwbs) {
      Error("TMatrixTSub_const","Request sub-matrix with col_upbs(%d) < col_lwbs(%d)",col_upbs,col_lwbs);
      return;
   }

   const Int_t rowLwb = matrix.GetRowLwb();
   const Int_t rowUpb = matrix.GetRowUpb();
   const Int_t colLwb = matrix.GetColLwb();
   const Int_t colUpb = matrix.GetColUpb();

   if (row_lwbs < rowLwb || row_lwbs > rowUpb) {
      Error("TMatrixTSub_const","Request row_lwbs sub-matrix(%d) outside matrix range of %d - %d",row_lwbs,rowLwb,rowUpb);
      return;
   }
   if (col_lwbs < colLwb || col_lwbs > colUpb) {
      Error("TMatrixTSub_const","Request col_lwbs sub-matrix(%d) outside matrix range of %d - %d",col_lwbs,colLwb,colUpb);
      return;
   }
   if (row_upbs < rowLwb || row_upbs > rowUpb) {
      Error("TMatrixTSub_const","Request row_upbs sub-matrix(%d) outside matrix range of %d - %d",row_upbs,rowLwb,rowUpb);
      return;
   }
   if (col_upbs < colLwb || col_upbs > colUpb) {
      Error("TMatrixTSub_const","Request col_upbs sub-matrix(%d) outside matrix range of %d - %d",col_upbs,colLwb,colUpb);
      return;
   }

   fRowOff    = row_lwbs-rowLwb;
   fColOff    = col_lwbs-colLwb;
   fNrowsSub  = row_upbs-row_lwbs+1;
   fNcolsSub  = col_upbs-col_lwbs+1;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSub<Element>::TMatrixTSub(TMatrixT<Element> &matrix,Int_t row_lwbs,Int_t row_upbs,
                                  Int_t col_lwbs,Int_t col_upbs)
            :TMatrixTSub_const<Element>(matrix,row_lwbs,row_upbs,col_lwbs,col_upbs)
{
// Constructor
}

//______________________________________________________________________________
template<class Element>
TMatrixTSub<Element>::TMatrixTSub(TMatrixTSym<Element> &matrix,Int_t row_lwbs,Int_t row_upbs,
                                  Int_t col_lwbs,Int_t col_upbs)
            :TMatrixTSub_const<Element>(matrix,row_lwbs,row_upbs,col_lwbs,col_upbs)
{
// Constructor
}

//______________________________________________________________________________
template<class Element>
TMatrixTSub<Element>::TMatrixTSub(const TMatrixTSub<Element> &ms) : TMatrixTSub_const<Element>(ms)
{
// Copy constructor

   *this = ms;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::Rank1Update(const TVectorT<Element> &v,Element alpha)
{
// Perform a rank 1 operation on the matrix:
//     A += alpha * v * v^T

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(v.IsValid());

   if (v.GetNoElements() < TMath::Max(this->fNrowsSub,this->fNcolsSub)) {
      Error("Rank1Update","vector too short");
      return;
   }

   const Element * const pv = v.GetMatrixArray();
         Element *mp = (const_cast<TMatrixTBase<Element> *>(this->fMatrix))->GetMatrixArray();

   const Int_t ncols = this->fMatrix->GetNcols();
   for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
      const Int_t off = (irow+this->fRowOff)*ncols+this->fColOff;
      const Element tmp = alpha*pv[irow];
      for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
         mp[off+icol] += tmp*pv[icol];
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator=(Element val)
{
// Assign val to every element of the sub matrix.

   R__ASSERT(this->fMatrix->IsValid());

   Element *p = (const_cast<TMatrixTBase<Element> *>(this->fMatrix))->GetMatrixArray();
   const Int_t ncols = this->fMatrix->GetNcols();
   for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
      const Int_t off = (irow+this->fRowOff)*ncols+this->fColOff;
      for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
         p[off+icol] = val;
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator+=(Element val)
{
// Add val to every element of the sub matrix.

   R__ASSERT(this->fMatrix->IsValid());

   Element *p = (const_cast<TMatrixTBase<Element> *>(this->fMatrix))->GetMatrixArray();
   const Int_t ncols = this->fMatrix->GetNcols();
   for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
      const Int_t off = (irow+this->fRowOff)*ncols+this->fColOff;
      for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
         p[off+icol] += val;
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator*=(Element val)
{
// Multiply every element of the sub matrix by val .

   R__ASSERT(this->fMatrix->IsValid());

   Element *p = (const_cast<TMatrixTBase<Element> *>(this->fMatrix))->GetMatrixArray();
   const Int_t ncols = this->fMatrix->GetNcols();
   for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
      const Int_t off = (irow+this->fRowOff)*ncols+this->fColOff;
      for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
         p[off+icol] *= val;
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator=(const TMatrixTSub_const<Element> &ms)
{
// Assignment operator

   const TMatrixTBase<Element> *mt = ms.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->fMatrix == mt &&
       (this->GetNrows()  == ms.GetNrows () && this->GetNcols()  == ms.GetNcols () &&
        this->GetRowOff() == ms.GetRowOff() && this->GetColOff() == ms.GetColOff()) )
     return;

   if (this->GetNrows() != ms.GetNrows() || this->GetNcols() != ms.GetNcols()) {
     Error("operator=(const TMatrixTSub_const &)","sub matrices have different size");
      return;
   }

   const Int_t rowOff2 = ms.GetRowOff();
   const Int_t colOff2 = ms.GetColOff();

   Bool_t overlap = (this->fMatrix == mt) &&
                    ( (rowOff2 >= this->fRowOff && rowOff2 < this->fRowOff+this->fNrowsSub) ||
                      (colOff2 >= this->fColOff && colOff2 < this->fColOff+this->fNcolsSub) );

   Element *p1 = const_cast<Element *>(this->fMatrix->GetMatrixArray());
   if (!overlap) {
      const Element *p2 = mt->GetMatrixArray();

      const Int_t ncols1 = this->fMatrix->GetNcols();
      const Int_t ncols2 = mt->GetNcols();
      for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
         const Int_t off1 = (irow+this->fRowOff)*ncols1+this->fColOff;
         const Int_t off2 = (irow+rowOff2)*ncols2+colOff2;
         for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
            p1[off1+icol] = p2[off2+icol];
      }
   } else {
      const Int_t row_lwbs = rowOff2+mt->GetRowLwb();
      const Int_t row_upbs = row_lwbs+this->fNrowsSub-1;
      const Int_t col_lwbs = colOff2+mt->GetColLwb();
      const Int_t col_upbs = col_lwbs+this->fNcolsSub-1;
      TMatrixT<Element> tmp; mt->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,tmp);
      const Element *p2 = tmp.GetMatrixArray();

      const Int_t ncols1 = this->fMatrix->GetNcols();
      const Int_t ncols2 = tmp.GetNcols();
      for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
         const Int_t off1 = (irow+this->fRowOff)*ncols1+this->fColOff;
         const Int_t off2 = irow*ncols2;
         for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
            p1[off1+icol] = p2[off2+icol];
      }
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator=(const TMatrixTBase<Element> &m)
{
// Assignment operator

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(m.IsValid());

   if (this->fMatrix->GetMatrixArray() == m.GetMatrixArray()) return;

   if (this->fNrowsSub != m.GetNrows() || this->fNcolsSub != m.GetNcols()) {
      Error("operator=(const TMatrixTBase<Element> &)","sub matrices and matrix have different size");
      return;
   }
   const Int_t row_lwbs = this->fRowOff+this->fMatrix->GetRowLwb();
   const Int_t col_lwbs = this->fColOff+this->fMatrix->GetColLwb();
   (const_cast<TMatrixTBase<Element> *>(this->fMatrix))->SetSub(row_lwbs,col_lwbs,m);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator+=(const TMatrixTSub_const<Element> &ms)
{
// Add to every element of the submatrix the corresponding element of submatrix ms.

   const TMatrixTBase<Element> *mt = ms.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());

   if (this->GetNrows() != ms.GetNrows() || this->GetNcols() != ms.GetNcols()) {
      Error("operator+=(const TMatrixTSub_const &)","sub matrices have different size");
      return;
   }

   const Int_t rowOff2 = ms.GetRowOff();
   const Int_t colOff2 = ms.GetColOff();

   Bool_t overlap = (this->fMatrix == mt) &&
                    ( (rowOff2 >= this->fRowOff && rowOff2 < this->fRowOff+this->fNrowsSub) ||
                      (colOff2 >= this->fColOff && colOff2 < this->fColOff+this->fNcolsSub) );

   Element *p1 = const_cast<Element *>(this->fMatrix->GetMatrixArray());
   if (!overlap) {
      const Element *p2 = mt->GetMatrixArray();

      const Int_t ncols1 = this->fMatrix->GetNcols();
      const Int_t ncols2 = mt->GetNcols();
      for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
         const Int_t off1 = (irow+this->fRowOff)*ncols1+this->fColOff;
         const Int_t off2 = (irow+rowOff2)*ncols2+colOff2;
         for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
            p1[off1+icol] += p2[off2+icol];
      }
   } else {
      const Int_t row_lwbs = rowOff2+mt->GetRowLwb();
      const Int_t row_upbs = row_lwbs+this->fNrowsSub-1;
      const Int_t col_lwbs = colOff2+mt->GetColLwb();
      const Int_t col_upbs = col_lwbs+this->fNcolsSub-1;
      TMatrixT<Element> tmp; mt->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,tmp);
      const Element *p2 = tmp.GetMatrixArray();

      const Int_t ncols1 = this->fMatrix->GetNcols();
      const Int_t ncols2 = tmp.GetNcols();
      for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
         const Int_t off1 = (irow+this->fRowOff)*ncols1+this->fColOff;
         const Int_t off2 = irow*ncols2;
         for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
            p1[off1+icol] += p2[off2+icol];
      }
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator*=(const TMatrixTSub_const<Element> &ms)
{
// Multiply submatrix with submatrix ms.

   if (this->fNcolsSub != ms.GetNrows() || this->fNcolsSub != ms.GetNcols()) {
      Error("operator*=(const TMatrixTSub_const &)","source sub matrix has wrong shape");
      return;
   }

   const TMatrixTBase<Element> *source = ms.GetMatrix();

   TMatrixT<Element> source_sub;
   {
      const Int_t row_lwbs = ms.GetRowOff()+source->GetRowLwb();
      const Int_t row_upbs = row_lwbs+this->fNrowsSub-1;
      const Int_t col_lwbs = ms.GetColOff()+source->GetColLwb();
      const Int_t col_upbs = col_lwbs+this->fNcolsSub-1;
      source->GetSub(row_lwbs,row_upbs,col_lwbs,col_upbs,source_sub);
   }

   const Element *sp = source_sub.GetMatrixArray();
   const Int_t ncols = this->fMatrix->GetNcols();

   // One row of the old_target matrix
   Element work[kWorkMax];
   Bool_t isAllocated = kFALSE;
   Element *trp = work;
   if (this->fNcolsSub > kWorkMax) {
      isAllocated = kTRUE;
      trp = new Element[this->fNcolsSub];
   }

         Element *cp   = const_cast<Element *>(this->fMatrix->GetMatrixArray())+this->fRowOff*ncols+this->fColOff;
   const Element *trp0 = cp; // Pointer to  target[i,0];
   const Element * const trp0_last = trp0+this->fNrowsSub*ncols;
   while (trp0 < trp0_last) {
      memcpy(trp,trp0,this->fNcolsSub*sizeof(Element));         // copy the i-th row of target, Start at target[i,0]
      for (const Element *scp = sp; scp < sp+this->fNcolsSub; ) {  // Pointer to the j-th column of source,
                                                                 // Start scp = source[0,0]
         Element cij = 0;
         for (Int_t j = 0; j < this->fNcolsSub; j++) {
            cij += trp[j] * *scp;                            // the j-th col of source
            scp += this->fNcolsSub;
         }
         *cp++ = cij;
         scp -= source_sub.GetNoElements()-1;               // Set bcp to the (j+1)-th col
      }
      cp   += ncols-this->fNcolsSub;
      trp0 += ncols;                                      // Set trp0 to the (i+1)-th row
      R__ASSERT(trp0 == cp);
   }

   R__ASSERT(cp == trp0_last && trp0 == trp0_last);
   if (isAllocated)
      delete [] trp;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator+=(const TMatrixTBase<Element> &mt)
{
// Add to every element of the submatrix the corresponding element of matrix mt.

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt.IsValid());

   if (this->GetNrows() != mt.GetNrows() || this->GetNcols() != mt.GetNcols()) {
      Error("operator+=(const TMatrixTBase<Element> &)","sub matrix and matrix have different size");
      return;
   }

   Element *p1 = const_cast<Element *>(this->fMatrix->GetMatrixArray());
   const Element *p2 = mt.GetMatrixArray();

   const Int_t ncols1 = this->fMatrix->GetNcols();
   const Int_t ncols2 = mt.GetNcols();
   for (Int_t irow = 0; irow < this->fNrowsSub; irow++) {
      const Int_t off1 = (irow+this->fRowOff)*ncols1+this->fColOff;
      const Int_t off2 = irow*ncols2;
      for (Int_t icol = 0; icol < this->fNcolsSub; icol++)
         p1[off1+icol] += p2[off2+icol];
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator*=(const TMatrixT<Element> &source)
{
// Multiply submatrix with matrix source.

   if (this->fNcolsSub != source.GetNrows() || this->fNcolsSub != source.GetNcols()) {
      Error("operator*=(const TMatrixT<Element> &)","source matrix has wrong shape");
      return;
   }

   // Check for A *= A;
   const Element *sp;
   TMatrixT<Element> tmp;
   if (this->fMatrix->GetMatrixArray() == source.GetMatrixArray()) {
      tmp.ResizeTo(source);
      tmp = source;
      sp = tmp.GetMatrixArray();
   }
   else
      sp = source.GetMatrixArray();

   const Int_t ncols = this->fMatrix->GetNcols();

   // One row of the old_target matrix
   Element work[kWorkMax];
   Bool_t isAllocated = kFALSE;
    Element *trp = work;
   if (this->fNcolsSub > kWorkMax) {
      isAllocated = kTRUE;
      trp = new Element[this->fNcolsSub];
   }

         Element *cp   = const_cast<Element *>(this->fMatrix->GetMatrixArray())+this->fRowOff*ncols+this->fColOff;
   const Element *trp0 = cp;                               // Pointer to  target[i,0];
   const Element * const trp0_last = trp0+this->fNrowsSub*ncols;
   while (trp0 < trp0_last) {
      memcpy(trp,trp0,this->fNcolsSub*sizeof(Element));           // copy the i-th row of target, Start at target[i,0]
      for (const Element *scp = sp; scp < sp+this->fNcolsSub; ) { // Pointer to the j-th column of source,
                                                                  // Start scp = source[0,0]
         Element cij = 0;
         for (Int_t j = 0; j < this->fNcolsSub; j++) {
            cij += trp[j] * *scp;                              // the j-th col of source
            scp += this->fNcolsSub;
         }
         *cp++ = cij;
         scp -= source.GetNoElements()-1;                    // Set bcp to the (j+1)-th col
      }
      cp   += ncols-this->fNcolsSub;
      trp0 += ncols;                                        // Set trp0 to the (i+1)-th row
      R__ASSERT(trp0 == cp);
   }
 
   R__ASSERT(cp == trp0_last && trp0 == trp0_last);
   if (isAllocated)
      delete [] trp;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSub<Element>::operator*=(const TMatrixTSym<Element> &source)
{
// Multiply submatrix with matrix source.

   if (this->fNcolsSub != source.GetNrows() || this->fNcolsSub != source.GetNcols()) {
      Error("operator*=(const TMatrixTSym<Element> &)","source matrix has wrong shape");
      return;
   }

   // Check for A *= A;
   const Element *sp;
   TMatrixTSym<Element> tmp;
   if (this->fMatrix->GetMatrixArray() == source.GetMatrixArray()) {
      tmp.ResizeTo(source);
      tmp = source;
      sp = tmp.GetMatrixArray();
   }
   else
      sp = source.GetMatrixArray();

   const Int_t ncols = this->fMatrix->GetNcols();

   // One row of the old_target matrix
   Element work[kWorkMax];
   Bool_t isAllocated = kFALSE;
   Element *trp = work;
   if (this->fNcolsSub > kWorkMax) {
      isAllocated = kTRUE;
      trp = new Element[this->fNcolsSub];
   }

         Element *cp   = const_cast<Element *>(this->fMatrix->GetMatrixArray())+this->fRowOff*ncols+this->fColOff;
   const Element *trp0 = cp;                               // Pointer to  target[i,0];
   const Element * const trp0_last = trp0+this->fNrowsSub*ncols;
   while (trp0 < trp0_last) {
      memcpy(trp,trp0,this->fNcolsSub*sizeof(Element));           // copy the i-th row of target, Start at target[i,0]
      for (const Element *scp = sp; scp < sp+this->fNcolsSub; ) { // Pointer to the j-th column of source,
                                                                // Start scp = source[0,0]
         Element cij = 0;
         for (Int_t j = 0; j < this->fNcolsSub; j++) {
            cij += trp[j] * *scp;                              // the j-th col of source
            scp += this->fNcolsSub;
         }
         *cp++ = cij;
         scp -= source.GetNoElements()-1;                     // Set bcp to the (j+1)-th col
      }
      cp   += ncols-this->fNcolsSub;
      trp0 += ncols;                                         // Set trp0 to the (i+1)-th row
      R__ASSERT(trp0 == cp);
   }

   R__ASSERT(cp == trp0_last && trp0 == trp0_last);
   if (isAllocated)
      delete [] trp;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparseRow_const<Element>::TMatrixTSparseRow_const(const TMatrixTSparse<Element> &matrix,Int_t row)
{
// Constructor with row "row" of matrix

   R__ASSERT(matrix.IsValid());

   fRowInd = row-matrix.GetRowLwb();
   if (fRowInd >= matrix.GetNrows() || fRowInd < 0) {
      Error("TMatrixTSparseRow_const(const TMatrixTSparse &,Int_t)","row index out of bounds");
      fMatrix  = 0;
      fNindex  = 0;
      fColPtr  = 0;
      fDataPtr = 0;
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
template<class Element>
Element TMatrixTSparseRow_const<Element>::operator()(Int_t i) const
{
  R__ASSERT(fMatrix->IsValid());
  const Int_t acoln = i-fMatrix->GetColLwb();
  if (acoln < fMatrix->GetNcols() && acoln >= 0) {
     const Int_t index = TMath::BinarySearch(fNindex,fColPtr,acoln);
     if (index >= 0 && fColPtr[index] == acoln) return fDataPtr[index];
     else                                       return 0.0;
  } else {
     Error("operator()","Request col(%d) outside matrix range of %d - %d",
                        i,fMatrix->GetColLwb(),fMatrix->GetColLwb()+fMatrix->GetNcols());
     return 0.0;
  }
 }

//______________________________________________________________________________
template<class Element>
TMatrixTSparseRow<Element>::TMatrixTSparseRow(TMatrixTSparse<Element> &matrix,Int_t row)
                                    : TMatrixTSparseRow_const<Element>(matrix,row)
{
// Constructor with row "row" of matrix
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparseRow<Element>::TMatrixTSparseRow(const TMatrixTSparseRow<Element> &mr)
                                    : TMatrixTSparseRow_const<Element>(mr)
{
// Copy constructor

   *this = mr;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTSparseRow<Element>::operator()(Int_t i) const
{
  R__ASSERT(this->fMatrix->IsValid());
  const Int_t acoln = i-this->fMatrix->GetColLwb();
  if (acoln < this->fMatrix->GetNcols() && acoln >= 0) {
     const Int_t index = TMath::BinarySearch(this->fNindex,this->fColPtr,acoln);
     if (index >= 0 && this->fColPtr[index] == acoln) return this->fDataPtr[index];
     else                                             return 0.0;
  } else {
     Error("operator()","Request col(%d) outside matrix range of %d - %d",
                        i,this->fMatrix->GetColLwb(),this->fMatrix->GetColLwb()+this->fMatrix->GetNcols());
     return 0.0;
  }
}

//______________________________________________________________________________
template<class Element>
Element &TMatrixTSparseRow<Element>::operator()(Int_t i)
{
// operator() : pick element row(i)

   R__ASSERT(this->fMatrix->IsValid());

   const Int_t acoln = i-this->fMatrix->GetColLwb();
   if (acoln >= this->fMatrix->GetNcols() || acoln < 0) {
      Error("operator()(Int_t","Requested element %d outside range : %d - %d",i,
            this->fMatrix->GetColLwb(),this->fMatrix->GetColLwb()+this->fMatrix->GetNcols());
      return (const_cast<Element*>(this->fDataPtr))[0];
   }

   Int_t index = TMath::BinarySearch(this->fNindex,this->fColPtr,acoln);
   if (index >= 0 && this->fColPtr[index] == acoln)
      return (const_cast<Element*>(this->fDataPtr))[index];
   else {
      TMatrixTBase<Element> *mt = const_cast<TMatrixTBase<Element> *>(this->fMatrix);
      const Int_t row = this->fRowInd+mt->GetRowLwb();
      Element val = 0.;
      mt->InsertRow(row,i,&val,1);
      const Int_t sIndex = mt->GetRowIndexArray()[this->fRowInd];
      const Int_t eIndex = mt->GetRowIndexArray()[this->fRowInd+1];
      this->fNindex  = eIndex-sIndex;
      this->fColPtr  = mt->GetColIndexArray()+sIndex;
      this->fDataPtr = mt->GetMatrixArray()+sIndex;
      index = TMath::BinarySearch(this->fNindex,this->fColPtr,acoln);
      if (index >= 0 && this->fColPtr[index] == acoln)
         return (const_cast<Element*>(this->fDataPtr))[index];
      else {
         Error("operator()(Int_t","Insert row failed");
         return (const_cast<Element*>(this->fDataPtr))[0];
      }
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseRow<Element>::operator=(Element val)
{
// Assign val to every non-zero (!) element of the matrix row.

   R__ASSERT(this->fMatrix->IsValid());
   Element *rp = const_cast<Element *>(this->fDataPtr);
   for ( ; rp < this->fDataPtr+this->fNindex; rp++)
      *rp = val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseRow<Element>::operator+=(Element val)
{
// Add val to every non-zero (!) element of the matrix row.

   R__ASSERT(this->fMatrix->IsValid());
   Element *rp = const_cast<Element *>(this->fDataPtr);
   for ( ; rp < this->fDataPtr+this->fNindex; rp++)
      *rp += val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseRow<Element>::operator*=(Element val)
{
// Multiply every element of the matrix row by val.

   R__ASSERT(this->fMatrix->IsValid());
   Element *rp = const_cast<Element *>(this->fDataPtr);
   for ( ; rp < this->fDataPtr+this->fNindex; rp++)
      *rp *= val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseRow<Element>::operator=(const TMatrixTSparseRow_const<Element> &mr)
{
// Assignment operator

   const TMatrixTBase<Element> *mt = mr.GetMatrix();
   if (this->fMatrix == mt) return;

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fMatrix->GetColLwb() != mt->GetColLwb() || this->fMatrix->GetNcols() != mt->GetNcols()) {
      Error("operator=(const TMatrixTSparseRow_const &)","matrix rows not compatible");
      return;
   }

   const Int_t ncols = this->fMatrix->GetNcols();
   const Int_t row1  = this->fRowInd+this->fMatrix->GetRowLwb();
   const Int_t row2  = mr.GetRowIndex()+mt->GetRowLwb();
   const Int_t col   = this->fMatrix->GetColLwb();

   TVectorT<Element> v(ncols);
   mt->ExtractRow(row2,col,v.GetMatrixArray());
   const_cast<TMatrixTBase<Element> *>(this->fMatrix)->InsertRow(row1,col,v.GetMatrixArray());

   const Int_t sIndex = this->fMatrix->GetRowIndexArray()[this->fRowInd];
   const Int_t eIndex = this->fMatrix->GetRowIndexArray()[this->fRowInd+1];
   this->fNindex  = eIndex-sIndex;
   this->fColPtr  = this->fMatrix->GetColIndexArray()+sIndex;
   this->fDataPtr = this->fMatrix->GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseRow<Element>::operator=(const TVectorT<Element> &vec)
{
// Assign a vector to a matrix row. The vector is considered row-vector
// to allow the assignment in the strict sense.

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(vec.IsValid());

   if (this->fMatrix->GetColLwb() != vec.GetLwb() || this->fMatrix->GetNcols() != vec.GetNrows()) {
      Error("operator=(const TVectorT &)","vector length != matrix-row length");
      return;
   }

   const Element *vp = vec.GetMatrixArray();
   const Int_t row = this->fRowInd+this->fMatrix->GetRowLwb();
   const Int_t col = this->fMatrix->GetColLwb();
   const_cast<TMatrixTBase<Element> *>(this->fMatrix)->InsertRow(row,col,vp,vec.GetNrows());

   const Int_t sIndex = this->fMatrix->GetRowIndexArray()[this->fRowInd];
   const Int_t eIndex = this->fMatrix->GetRowIndexArray()[this->fRowInd+1];
   this->fNindex  = eIndex-sIndex;
   this->fColPtr  = this->fMatrix->GetColIndexArray()+sIndex;
   this->fDataPtr = this->fMatrix->GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseRow<Element>::operator+=(const TMatrixTSparseRow_const<Element> &r)
{
// Add to every element of the matrix row the corresponding element of row r.

   const TMatrixTBase<Element> *mt = r.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fMatrix->GetColLwb() != mt->GetColLwb() || this->fMatrix->GetNcols() != mt->GetNcols()) {
      Error("operator+=(const TMatrixTRow_const &)","different row lengths");
      return;
   }

   const Int_t ncols = this->fMatrix->GetNcols();
   const Int_t row1  = this->fRowInd+this->fMatrix->GetRowLwb();
   const Int_t row2  = r.GetRowIndex()+mt->GetRowLwb();
   const Int_t col   = this->fMatrix->GetColLwb();

   TVectorT<Element> v1(ncols);
   TVectorT<Element> v2(ncols);
   this->fMatrix->ExtractRow(row1,col,v1.GetMatrixArray());
   mt           ->ExtractRow(row2,col,v2.GetMatrixArray());
   v1 += v2;
   const_cast<TMatrixTBase<Element> *>(this->fMatrix)->InsertRow(row1,col,v1.GetMatrixArray());

   const Int_t sIndex = this->fMatrix->GetRowIndexArray()[this->fRowInd];
   const Int_t eIndex = this->fMatrix->GetRowIndexArray()[this->fRowInd+1];
   this->fNindex  = eIndex-sIndex;
   this->fColPtr  = this->fMatrix->GetColIndexArray()+sIndex;
   this->fDataPtr = this->fMatrix->GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseRow<Element>::operator*=(const TMatrixTSparseRow_const<Element> &r)
{
// Multiply every element of the matrix row with the
// corresponding element of row r.

   const TMatrixTBase<Element> *mt = r.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fMatrix->GetColLwb() != mt->GetColLwb() || this->fMatrix->GetNcols() != mt->GetNcols()) {
      Error("operator+=(const TMatrixTRow_const &)","different row lengths");
      return;
   }

   const Int_t ncols = this->fMatrix->GetNcols();
   const Int_t row1  = r.GetRowIndex()+mt->GetRowLwb();
   const Int_t row2  = r.GetRowIndex()+mt->GetRowLwb();
   const Int_t col   = this->fMatrix->GetColLwb();

   TVectorT<Element> v1(ncols);
   TVectorT<Element> v2(ncols);
   this->fMatrix->ExtractRow(row1,col,v1.GetMatrixArray());
   mt           ->ExtractRow(row2,col,v2.GetMatrixArray());

   ElementMult(v1,v2);
   const_cast<TMatrixTBase<Element> *>(this->fMatrix)->InsertRow(row1,col,v1.GetMatrixArray());

   const Int_t sIndex = this->fMatrix->GetRowIndexArray()[this->fRowInd];
   const Int_t eIndex = this->fMatrix->GetRowIndexArray()[this->fRowInd+1];
   this->fNindex  = eIndex-sIndex;
   this->fColPtr  = this->fMatrix->GetColIndexArray()+sIndex;
   this->fDataPtr = this->fMatrix->GetMatrixArray()+sIndex;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparseDiag_const<Element>::TMatrixTSparseDiag_const(const TMatrixTSparse<Element> &matrix)
{
// Constructor

   R__ASSERT(matrix.IsValid());

   fMatrix  = &matrix;
   fNdiag   = TMath::Min(matrix.GetNrows(),matrix.GetNcols());
   fDataPtr = matrix.GetMatrixArray();
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTSparseDiag_const<Element>::operator()(Int_t i) const
{
  R__ASSERT(fMatrix->IsValid());
  if (i < fNdiag && i >= 0) {
     const Int_t   * const pR = fMatrix->GetRowIndexArray();
     const Int_t   * const pC = fMatrix->GetColIndexArray();
     const Element * const pD = fMatrix->GetMatrixArray();
     const Int_t sIndex = pR[i];
     const Int_t eIndex = pR[i+1];
     const Int_t index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
     if (index >= sIndex && pC[index] == i) return pD[index];
     else                                   return 0.0;
  } else {
     Error("operator()","Request diagonal(%d) outside matrix range of 0 - %d",i,fNdiag);
     return 0.0;
  }
  return 0.0;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparseDiag<Element>::TMatrixTSparseDiag(TMatrixTSparse<Element> &matrix)
                   :TMatrixTSparseDiag_const<Element>(matrix)
{
// Constructor
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparseDiag<Element>::TMatrixTSparseDiag(const TMatrixTSparseDiag<Element> &md)
                  : TMatrixTSparseDiag_const<Element>(md)
{
// Constructor

   *this = md;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTSparseDiag<Element>::operator()(Int_t i) const
{
    R__ASSERT(this->fMatrix->IsValid());
    if (i < this->fNdiag && i >= 0) {
       const Int_t   * const pR = this->fMatrix->GetRowIndexArray();
       const Int_t   * const pC = this->fMatrix->GetColIndexArray();
       const Element * const pD = this->fMatrix->GetMatrixArray();
       const Int_t sIndex = pR[i];
       const Int_t eIndex = pR[i+1];
       const Int_t index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
       if (index >= sIndex && pC[index] == i) return pD[index];
       else                                   return 0.0;
    } else {
       Error("operator()","Request diagonal(%d) outside matrix range of 0 - %d",i,this->fNdiag);
       return 0.0;
    }
    return 0.0;
 }

//______________________________________________________________________________
template<class Element>
Element &TMatrixTSparseDiag<Element>::operator()(Int_t i)
{
// operator() : pick element diag(i)

   R__ASSERT(this->fMatrix->IsValid());

   if (i < 0 || i >= this->fNdiag) {
      Error("operator()(Int_t","Requested element %d outside range : 0 - %d",i,this->fNdiag);
      return (const_cast<Element*>(this->fDataPtr))[0];
   }

   TMatrixTBase<Element> *mt = const_cast<TMatrixTBase<Element> *>(this->fMatrix);
   const Int_t *pR = mt->GetRowIndexArray();
   const Int_t *pC = mt->GetColIndexArray();
   Int_t sIndex = pR[i];
   Int_t eIndex = pR[i+1];
   Int_t index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
   if (index >= sIndex && pC[index] == i)
      return (const_cast<Element*>(this->fDataPtr))[index];
   else {
      const Int_t row = i+mt->GetRowLwb();
      const Int_t col = i+mt->GetColLwb();
      Element val = 0.;
      mt->InsertRow(row,col,&val,1);
      this->fDataPtr = mt->GetMatrixArray();
      pR = mt->GetRowIndexArray();
      pC = mt->GetColIndexArray();
      sIndex = pR[i];
      eIndex = pR[i+1];
      index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
      if (index >= sIndex && pC[index] == i)
         return (const_cast<Element*>(this->fDataPtr))[index];
      else {
         Error("operator()(Int_t","Insert row failed");
         return (const_cast<Element*>(this->fDataPtr))[0];
      }
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseDiag<Element>::operator=(Element val)
{
// Assign val to every element of the matrix diagonal.

   R__ASSERT(this->fMatrix->IsValid());
   for (Int_t i = 0; i < this->fNdiag; i++)
      (*this)(i) = val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseDiag<Element>::operator+=(Element val)
{
// Add val to every element of the matrix diagonal.

   R__ASSERT(this->fMatrix->IsValid());
   for (Int_t i = 0; i < this->fNdiag; i++)
      (*this)(i) += val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseDiag<Element>::operator*=(Element val)
{
// Multiply every element of the matrix diagonal by val.

   R__ASSERT(this->fMatrix->IsValid());
   for (Int_t i = 0; i < this->fNdiag; i++)
      (*this)(i) *= val;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseDiag<Element>::operator=(const TMatrixTSparseDiag_const<Element> &md)
{
// Assignment operator

   const TMatrixTBase<Element> *mt = md.GetMatrix();
   if (this->fMatrix == mt) return;

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fNdiag != md.GetNdiags()) {
      Error("operator=(const TMatrixTSparseDiag_const &)","matrix-diagonal's different length");
      return;
   }

   for (Int_t i = 0; i < this->fNdiag; i++)
      (*this)(i) = md(i);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseDiag<Element>::operator=(const TVectorT<Element> &vec)
{
// Assign a vector to the matrix diagonal.

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(vec.IsValid());

   if (this->fNdiag != vec.GetNrows()) {
      Error("operator=(const TVectorT &)","vector length != matrix-diagonal length");
      return;
   }

   const Element *vp = vec.GetMatrixArray();
   for (Int_t i = 0; i < this->fNdiag; i++)
      (*this)(i) = vp[i];
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseDiag<Element>::operator+=(const TMatrixTSparseDiag_const<Element> &md)
{
// Add to every element of the matrix diagonal the
// corresponding element of diagonal md.

   const TMatrixTBase<Element> *mt = md.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fNdiag != md.GetNdiags()) {
      Error("operator+=(const TMatrixTSparseDiag_const &)","matrix-diagonal's different length");
      return;
   }

   for (Int_t i = 0; i < this->fNdiag; i++)
      (*this)(i) += md(i);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparseDiag<Element>::operator*=(const TMatrixTSparseDiag_const<Element> &md)
{
// Multiply every element of the matrix diagonal with the
// corresponding element of diagonal md.

   const TMatrixTBase<Element> *mt = md.GetMatrix();

   R__ASSERT(this->fMatrix->IsValid());
   R__ASSERT(mt->IsValid());
   if (this->fNdiag != md.GetNdiags()) {
      Error("operator*=(const TMatrixTSparseDiag_const &)","matrix-diagonal's different length");
      return;
   }

   for (Int_t i = 0; i < this->fNdiag; i++)
      (*this)(i) *= md(i);
}

//______________________________________________________________________________
Double_t Drand(Double_t &ix)
{
// Random number generator [0....1] with seed ix

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

template class TMatrixTRow_const       <Float_t>;
template class TMatrixTColumn_const    <Float_t>;
template class TMatrixTDiag_const      <Float_t>;
template class TMatrixTFlat_const      <Float_t>;
template class TMatrixTSub_const       <Float_t>;
template class TMatrixTSparseRow_const <Float_t>;
template class TMatrixTSparseDiag_const<Float_t>;
template class TMatrixTRow             <Float_t>;
template class TMatrixTColumn          <Float_t>;
template class TMatrixTDiag            <Float_t>;
template class TMatrixTFlat            <Float_t>;
template class TMatrixTSub             <Float_t>;
template class TMatrixTSparseRow       <Float_t>;
template class TMatrixTSparseDiag      <Float_t>;
template class TElementActionT         <Float_t>;
template class TElementPosActionT      <Float_t>;

template class TMatrixTRow_const       <Double_t>;
template class TMatrixTColumn_const    <Double_t>;
template class TMatrixTDiag_const      <Double_t>;
template class TMatrixTFlat_const      <Double_t>;
template class TMatrixTSub_const       <Double_t>;
template class TMatrixTSparseRow_const <Double_t>;
template class TMatrixTSparseDiag_const<Double_t>;
template class TMatrixTRow             <Double_t>;
template class TMatrixTColumn          <Double_t>;
template class TMatrixTDiag            <Double_t>;
template class TMatrixTFlat            <Double_t>;
template class TMatrixTSub             <Double_t>;
template class TMatrixTSparseRow       <Double_t>;
template class TMatrixTSparseDiag      <Double_t>;
template class TElementActionT         <Double_t>;
template class TElementPosActionT      <Double_t>;
