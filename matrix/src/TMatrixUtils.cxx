// @(#)root/matrix:$Name:  $:$Id: TMatrixUtils.cxx,v 1.2 2000/11/21 20:41:36 brun Exp $
// Author: Fons Rademakers   05/11/97

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
//   TMatrixAction                                                      //
//   TMatrixPosAction                                                   //
//   TLazyMatrix                                                        //
//   THaarMatrix                                                        //
//   TMatrixRow                                                         //
//   TMatrixColumn                                                      //
//   TMatrixDiag                                                        //
//   TMatrixPivoting                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVector.h"
#include "TMatrix.h"
#include "TClass.h"


ClassImp(TLazyMatrix)
ClassImp(TMatrixRow)
ClassImp(TMatrixColumn)
ClassImp(TMatrixDiag)


//______________________________________________________________________________
void TMatrixRow::operator=(Real_t val)
{
   // Assign val to every element of the matrix row.

   if (!fMatrix->IsValid()) {
      Error("operator=", "matrix not initialized");
      return;
   }

   Real_t *rp = fPtr;                    // Row ptr
   for ( ; rp < fPtr + fMatrix->fNelems; rp += fInc)
      *rp = val;
}

//______________________________________________________________________________
void TMatrixRow::operator+=(Double_t val)
{
   // Add val to every element of the matrix row.

   if (!fMatrix->IsValid()) {
      Error("operator+=", "matrix not initialized");
      return;
   }

   Real_t *rp = fPtr;                    // Row ptr
   for ( ; rp < fPtr + fMatrix->fNelems; rp += fInc)
      *rp += val;
}

//______________________________________________________________________________
void TMatrixRow::operator*=(Double_t val)
{
   // Multiply every element of the matrix row with val.

   if (!fMatrix->IsValid()) {
      Error("operator*=", "matrix not initialized");
      return;
   }

   Real_t *rp = fPtr;                    // Row ptr
   for ( ; rp < fPtr + fMatrix->fNelems; rp += fInc)
      *rp *= val;
}

//______________________________________________________________________________
void TMatrixRow::operator=(const TVector &vec)
{
   // Assign a vector to a matrix row. The vector is considered row-vector
   // to allow the assignment in the strict sense.

   if (!fMatrix->IsValid()) {
      Error("operator=", "matrix not initialized");
      return;
   }
   if (!vec.IsValid()) {
      Error("operator=", "vector not initialized");
      return;
   }

   if (fMatrix->fColLwb != vec.fRowLwb || fMatrix->fNcols != vec.fNrows) {
      Error("operator=", "transposed vector cannot be assigned to a row of the matrix");
      return;
   }

   Real_t *rp = fPtr;                          // Row ptr
   Real_t *vp = vec.fElements;                 // Vector ptr
   for ( ; rp < fPtr + fMatrix->fNelems; rp += fInc)
      *rp = *vp++;

   Assert(vp == vec.fElements + vec.fNrows);
}

//______________________________________________________________________________
void TMatrixRow::Streamer(TBuffer &R__b)
{
   // Stream an object of class TMatrixRow.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      TMatrixRow::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
      fPtr = &(fMatrix->fIndex[0][fRowInd]);
   } else {
      TMatrix::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void TMatrixColumn::operator=(Real_t val)
{
   // Assign val to every element of the matrix column.

   if (!fMatrix->IsValid()) {
      Error("operator=", "matrix not initialized");
      return;
   }

   Real_t *cp = fPtr;                    // Column ptr
   while (cp < fPtr + fMatrix->fNrows)
      *cp++ = val;
}

//______________________________________________________________________________
void TMatrixColumn::operator+=(Double_t val)
{
   // Add val to every element of the matrix column.

   if (!fMatrix->IsValid()) {
      Error("operator+=", "matrix not initialized");
      return;
   }

   Real_t *cp = fPtr;                    // Column ptr
   while (cp < fPtr + fMatrix->fNrows)
      *cp++ += val;
}

//______________________________________________________________________________
void TMatrixColumn::operator*=(Double_t val)
{
   // Multiply every element of the matrix column with val.

   if (!fMatrix->IsValid()) {
      Error("operator*=", "matrix not initialized");
      return;
   }

   Real_t *cp = fPtr;                    // Column ptr
   while (cp < fPtr + fMatrix->fNrows)
      *cp++ *= val;
}

//______________________________________________________________________________
void TMatrixColumn::operator=(const TVector &vec)
{
   // Assign a vector to a matrix column.

   if (!fMatrix->IsValid()) {
      Error("operator=", "matrix not initialized");
      return;
   }
   if (!vec.IsValid()) {
      Error("operator=", "vector not initialized");
      return;
   }

   if (fMatrix->fRowLwb != vec.fRowLwb || fMatrix->fNrows != vec.fNrows) {
      Error("operator=", "vector cannot be assigned to a column of the matrix");
      return;
   }

   Real_t *cp = fPtr;                       // Column ptr
   Real_t *vp = vec.fElements;              // Vector ptr
   while (cp < fPtr + fMatrix->fNrows)
      *cp++ = *vp++;

   Assert(vp == vec.fElements + vec.fNrows);
}

//______________________________________________________________________________
void TMatrixColumn::Streamer(TBuffer &R__b)
{
   // Stream an object of class TMatrixColumn.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      TMatrixColumn::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
      fPtr = &(fMatrix->fIndex[fColInd][0]);
   } else {
      TMatrixColumn::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void TMatrixDiag::operator=(Real_t val)
{
   // Assign val to every element of the matrix diagonal.

   if (!fMatrix->IsValid()) {
      Error("operator=", "matrix not initialized");
      return;
   }

   Real_t *dp = fPtr;                // Diag ptr
   Int_t i;
   for (i = 0; i < fNdiag; i++, dp += fInc)
      *dp = val;
}

//______________________________________________________________________________
void TMatrixDiag::operator+=(Double_t val)
{
   // Assign val to every element of the matrix diagonal.

   if (!fMatrix->IsValid()) {
      Error("operator+=", "matrix not initialized");
      return;
   }

   Real_t *dp = fPtr;                // Diag ptr
   Int_t i;
   for (i = 0; i < fNdiag; i++, dp += fInc)
      *dp += val;
}

//______________________________________________________________________________
void TMatrixDiag::operator*=(Double_t val)
{
   // Multiply every element of the matrix diagonal with val.

   if (!fMatrix->IsValid()) {
      Error("operator*=", "matrix not initialized");
      return;
   }

   Real_t *dp = fPtr;                // Diag ptr
   Int_t i;
   for (i = 0; i < fNdiag; i++, dp += fInc)
      *dp *= val;
}

//______________________________________________________________________________
void TMatrixDiag::operator=(const TVector &vec)
{
   // Assign a vector to the matrix diagonal.

   if (!fMatrix->IsValid()) {
      Error("operator=", "matrix not initialized");
      return;
   }
   if (!vec.IsValid()) {
      Error("operator=", "vector not initialized");
      return;
   }

   if (fNdiag != vec.fNrows) {
      Error("operator=", "vector cannot be assigned to the matrix diagonal");
      return;
   }

   Real_t *dp = fPtr;                       // Diag ptr
   Real_t *vp = vec.fElements;              // Vector ptr
   for ( ; vp < vec.fElements + vec.fNrows; dp += fInc)
      *dp = *vp++;

   Assert(dp < fPtr + fMatrix->fNelems + fInc);
}

//______________________________________________________________________________
void TMatrixDiag::Streamer(TBuffer &R__b)
{
   // Stream an object of class TMatrixDiag.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      TMatrixDiag::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
      fPtr = &(fMatrix->fElements[0]);
   } else {
      TMatrixDiag::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TMatrixPivoting::TMatrixPivoting(const TMatrix &m)
    : TMatrix(m), fRowIndex(new Index_t[fNrows]), fPivotRow(0), fPivotCol(0)
{
   Assert(fRowIndex != 0);

   Index_t rp = fElements;               // Fill in the row_index
   for (Index_t *rip = fRowIndex; rip < fRowIndex+fNrows; )
      *rip++ = rp++;
}

//______________________________________________________________________________
TMatrixPivoting::~TMatrixPivoting()
{
   delete [] fRowIndex;
}

//______________________________________________________________________________
void TMatrixPivoting::PickUpPivot()
{
   // Pick up a pivot, an element with the largest abs value from yet
   // not-pivoted rows and cols

   Real_t max_elem = -1;                // Abs value of the largest element
   Index_t *prpp = 0;                   // Position of the pivot in fRowIndex
   Index_t *pcpp = 0;                   // Position of the pivot in fIndex

   Int_t col_odd = 0;                   // Parity of the current column

   for (Index_t *cpp = fIndex; cpp < fIndex + fNcols; cpp++) {
      const Real_t *cp = *cpp;          // Column pointer for the curr col
      if (cp == 0)                      // skip over already pivoted col
         continue;
      Int_t row_odd = 0;                // Parity of the current row
      for (Index_t *rip = fRowIndex; rip < fRowIndex + fNrows; rip++, cp++)
         if (*rip) {                    // only if the row hasn't been pivoted
            const Real_t v = *cp;
            if (TMath::Abs(v) > max_elem) {
               max_elem = TMath::Abs(v); // Note the local max of col elements
               fPivotValue = v;
               prpp = rip;
               pcpp = cpp;
               fPivotOdd = row_odd ^ col_odd;
            }
            row_odd ^= 1;                // Toggle parity for the next row
         }
      col_odd ^= 1;
   }

   if (max_elem < 0 || prpp == 0 || pcpp == 0)
      Error("PickUpPivot", "all the rows and columns have been already pivoted and eliminated");

   fPivotRow = *prpp, *prpp = 0;
   fPivotCol = *pcpp, *pcpp = 0;
}

//______________________________________________________________________________
Double_t TMatrixPivoting::PivotingAndElimination()
{
   // Perform pivoting and gaussian elemination, return the pivot value
   // times pivot parity. The procedure places zeros to the fPivotRow of
   // all not yet pivoted columns. A[i,j] -= A[i,pivot_col]/pivot*A[pivot_row,j]

   PickUpPivot();

   if (fPivotValue == 0)
      return 0;

   Assert(fPivotRow != 0 && fPivotCol != 0);

   Real_t *pcp;                         // Pivot column pointer
   const Index_t *rip;                  // Current ptr in row_index
                                        // Divide the pivoted column by pivot
   for (pcp = fPivotCol, rip = fRowIndex; rip < fRowIndex+fNrows; pcp++, rip++)
     if (*rip)                          // Skip already pivoted rows
        *pcp /= fPivotValue;

   // Eliminate all the elements from the pivot_row in all not pivoted columns
   const Real_t *prp = fPivotRow;
   for (const Index_t *cpp = fIndex; cpp < fIndex + fNcols; cpp++, prp += fNrows) {
      Real_t *cp = *cpp;                // A[*,j]
      if (cp == 0)                      // skip over already pivoted col
         continue;
      const Double_t fac = *prp;        // fac = A[pivot_row,j]
                                        // Do elimination stepping over pivoted rows
      for (pcp = fPivotCol, rip = fRowIndex; rip < fRowIndex+fNrows; pcp++, cp++, rip++)
         if (*rip)
            *cp -= *pcp * fac;
   }

   return fPivotOdd ? -fPivotValue : fPivotValue;
}

//______________________________________________________________________________
void MakeHaarMatrix(TMatrix &m)
{
   // Create an orthonormal (2^n)*(no_cols) Haar (sub)matrix, whose columns
   // are Haar functions. If no_cols is 0, create the complete matrix with
   // 2^n columns. Example, the complete Haar matrix of the second order is:
   // column 1: [ 1  1  1  1]/2
   // column 2: [ 1  1 -1 -1]/2
   // column 3: [ 1 -1  0  0]/sqrt(2)
   // column 4: [ 0  0  1 -1]/sqrt(2)
   // Matrix m is assumed to be zero originally.

   if (!m.IsValid()) {
      Error("MakeHaarMatrix", "matrix not initialized");
      return;
   }

   if (m.fNcols > m.fNrows || m.fNcols <= 0) {
      Error("MakeHaarMatrix", "matrix not right shape");
      return;
   }

   Real_t *cp = m.fElements;
   Real_t *m_end = m.fElements + (m.fNrows*m.fNcols);
   Int_t i;

   Double_t norm_factor = 1/TMath::Sqrt((Double_t)m.fNrows);

   // First column is always 1 (up to normalization)
   for ( i = 0; i < m.fNrows; i++)
      *cp++ = norm_factor;

   // The other functions are kind of steps: stretch of 1 followed by the
   // equally long stretch of -1. The functions can be grouped in families
   // according to their order (step size), differing only in the location
   // of the step
   Int_t step_length = m.fNrows/2;
   while (cp < m_end && step_length > 0) {
      for (Int_t step_position = 0; cp < m_end && step_position < m.fNrows;
           step_position += 2*step_length, cp += m.fNrows) {
         Real_t *ccp = cp + step_position;
         for (i = 0; i < step_length; i++)
            *ccp++ = norm_factor;
         for (i = 0; i < step_length; i++)
            *ccp++ = -norm_factor;
      }
      step_length /= 2;
      norm_factor *= TMath::Sqrt(2.0);
   }

   Assert(step_length != 0 || cp == m_end);
   Assert(m.fNrows != m.fNcols || step_length == 0);
}

//______________________________________________________________________________
THaarMatrix::THaarMatrix(Int_t order, Int_t no_cols)
    : TLazyMatrix(1<<order, no_cols == 0 ? 1<<order : no_cols)
{
   Assert(order > 0 && no_cols >= 0);
}

//______________________________________________________________________________
void THaarMatrix::FillIn(TMatrix &m) const
{
   MakeHaarMatrix(m);
}


#if defined(R__HPUX) || defined(R__MACOSX)

//______________________________________________________________________________
//  These functions should be inline
//______________________________________________________________________________

TMatrixRow::TMatrixRow(const TMatrix &matrix, Int_t row)
       : fMatrix(&matrix), fInc(matrix.fNrows)
{
   if (!matrix.IsValid()) {
      Error("TMatrixRow", "matrix is not initialized");
      return;
   }

   fRowInd = row - matrix.fRowLwb;

   if (fRowInd >= matrix.fNrows || fRowInd < 0) {
      Error("TMatrixRow", "row #%d is not within the matrix", row);
      return;
   }

   fPtr = &(matrix.fIndex[0][fRowInd]);
}

const Real_t &TMatrixRow::operator()(Int_t i) const
{
   // Get hold of the i-th row's element.

   static Real_t err;
   err = 0.0;

   if (!fMatrix->IsValid()) {
      Error("operator()", "matrix is not initialized");
      return err;
   }

   Int_t acoln = i-fMatrix->fColLwb;           // Effective index

   if (acoln >= fMatrix->fNcols || acoln < 0) {
      Error("operator()", "TMatrixRow index %d is out of row boundaries [%d,%d]",
            i, fMatrix->fColLwb, fMatrix->fNcols+fMatrix->fColLwb-1);
      return err;
   }

   return fMatrix->fIndex[acoln][fPtr-fMatrix->fElements];
}

Real_t &TMatrixRow::operator()(Int_t i)
{
   return (Real_t&)((*(const TMatrixRow *)this)(i));
}

TMatrixColumn::TMatrixColumn(const TMatrix &matrix, Int_t col)
       : fMatrix(&matrix)
{
   if (!matrix.IsValid()) {
      Error("TMatrixColumn", "matrix is not initialized");
      return;
   }

   fColInd = col - matrix.fColLwb;

   if (fColInd >= matrix.fNcols || fColInd < 0) {
      Error("TMatrixColumn", "column #%d is not within the matrix", col);
      return;
   }

   fPtr = &(matrix.fIndex[fColInd][0]);
}

const Real_t &TMatrixColumn::operator()(Int_t i) const
{
   // Access the i-th element of the column

   static Real_t err;
   err = 0.0;

   if (!fMatrix->IsValid()) {
      Error("operator()", "matrix is not initialized");
      return err;
   }

   Int_t arown = i-fMatrix->fRowLwb;           // Effective indices

   if (arown >= fMatrix->fNrows || arown < 0) {
      Error("operator()", "TMatrixColumn index %d is out of column boundaries [%d,%d]",
            i, fMatrix->fRowLwb, fMatrix->fNrows+fMatrix->fRowLwb-1);
      return err;
   }

   return fPtr[arown];
}

Real_t &TMatrixColumn::operator()(Int_t i)
{
   return (Real_t&)((*(const TMatrixColumn *)this)(i));
}

TMatrixDiag::TMatrixDiag(const TMatrix &matrix)
       : fMatrix(&matrix), fInc(matrix.fNrows+1),
         fNdiag(TMath::Min(matrix.fNrows, matrix.fNcols))
{
   if (!matrix.IsValid()) {
      Error("TMatrixDiag", "matrix is not initialized");
      return;
   }
   fPtr = &(matrix.fElements[0]);
}

const Real_t &TMatrixDiag::operator()(Int_t i) const
{
   // Get hold of the i-th diag element (indexing always starts at 0,
   // regardless of matrix' col_lwb and row_lwb)

   static Real_t err;
   err = 0.0;

   if (!fMatrix->IsValid()) {
      Error("operator()", "matrix is not initialized");
      return err;
   }

   if (i > fNdiag || i < 1) {
      Error("TMatrixDiag", "TMatrixDiag index %d is out of diag boundaries [1,%d]",
            i, fNdiag);
      return err;
   }

   return fMatrix->fIndex[i-1][i-1];
}

Real_t &TMatrixDiag::operator()(Int_t i)
{
   return (Real_t&)((*(const TMatrixDiag *)this)(i));
}

#endif
