// @(#)root/matrix:$Name:  $:$Id: TMatrixUtils.h,v 1.7 2002/05/10 07:18:59 brun Exp $
// Author: Fons Rademakers   05/11/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixUtils
#define ROOT_TMatrixUtils


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix utility classes.                                              //
//                                                                      //
// This file defines utility classes for the Linear Algebra Package.    //
// The following classes are defined here:                              //
//   TElementAction                                                     //
//   TElementPosAction                                                  //
//   TLazyMatrix                                                        //
//   THaarMatrix                                                        //
//   TMatrixRow                                                         //
//   TMatrixColumn                                                      //
//   TMatrixDiag                                                        //
//   TMatrixPivoting                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrix
#include "TMatrix.h"
#endif


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TElementAction                                                       //
//                                                                      //
// A class to do a specific operation on every vector or matrix element //
// (regardless of it position) as the object is being traversed.        //
// This is an abstract class. Derived classes need to implement the     //
// action function Operation().                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TElementAction {

friend class TMatrix;
friend class TVector;

private:
   virtual void Operation(Real_t &element) = 0;
   void operator=(const TElementAction &) { }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TElementPosAction                                                    //
//                                                                      //
// A class to do a specific operation on every vector or matrix element //
// as the object is being traversed. This is an abstract class.         //
// Derived classes need to implement the action function Operation().   //
// In the action function the location of the current element is        //
// known (fI=row, fJ=columns).                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TElementPosAction {

friend class TMatrix;
friend class TVector;

protected:
   Int_t fI;        // i position of element being passed to Operation()
   Int_t fJ;        // j position of element being passed to Operation()

private:
   virtual void Operation(Real_t &element) = 0;
   void operator=(const TElementPosAction &) { }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLazyMatrix                                                          //
//                                                                      //
// Class used to make a lazy copy of a matrix, i.e. only copy matrix    //
// when really needed (when accessed).                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TLazyMatrix : public TObject {

friend class TMatrix;

protected:
   Int_t fRowUpb;
   Int_t fRowLwb;
   Int_t fColUpb;
   Int_t fColLwb;

   TLazyMatrix(const TLazyMatrix &) : TObject() { }
   void operator=(const TLazyMatrix &) { }

private:
   virtual void FillIn(TMatrix &m) const = 0;

public:
   TLazyMatrix() { fRowUpb = fRowLwb = fColUpb = fColLwb = 0; }
   TLazyMatrix(Int_t nrows, Int_t ncols)
      : fRowUpb(nrows-1), fRowLwb(0), fColUpb(ncols-1), fColLwb(0) { }
   TLazyMatrix(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb)
      : fRowUpb(row_upb), fRowLwb(row_lwb), fColUpb(col_upb), fColLwb(col_lwb) { }

   ClassDef(TLazyMatrix,1)  // Lazy matrix
};


class THaarMatrix : public TLazyMatrix {

private:
   void FillIn(TMatrix &m) const;

public:
   THaarMatrix(Int_t n, Int_t no_cols = 0);
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixRow                                                           //
//                                                                      //
// Class represents a row of a TMatrix.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixRow : public TObject {

friend class TMatrix;
friend class TVector;

private:
   const TMatrix  *fMatrix;  //! the matrix I am a row of
   Int_t           fRowInd;  // effective row index
   Int_t           fInc;     // if ptr = @a[row,i], then ptr+inc = @a[row,i+1]
   Real_t         *fPtr;     //! pointer to the a[row,0]

   TMatrixRow() { fMatrix = 0; fInc = 0; fPtr = 0; }

public:
   TMatrixRow(const TMatrix &matrix, Int_t row);

   void operator=(Real_t val);
   void operator+=(Double_t val);
   void operator*=(Double_t val);

   void operator=(const TVector &vec);

   const Real_t &operator()(Int_t i) const;
   Real_t &operator()(Int_t i);

   ClassDef(TMatrixRow,1)  // One row of a matrix
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixColumn                                                        //
//                                                                      //
// Class represents a column of a TMatrix.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixColumn : public TObject {

friend class TMatrix;
friend class TVector;

private:
   const TMatrix  *fMatrix;         //! the matrix I am a column of
   Int_t           fColInd;         // effective column index
   Real_t         *fPtr;            //! pointer to the a[0,i] column

   TMatrixColumn() { fMatrix = 0; fPtr = 0; }

public:
   TMatrixColumn(const TMatrix &matrix, Int_t col);

   void operator=(Real_t val);
   void operator+=(Double_t val);
   void operator*=(Double_t val);

   void operator=(const TVector &vec);

   const Real_t &operator()(Int_t i) const;
   Real_t &operator()(Int_t i);

   ClassDef(TMatrixColumn,1)  // One column of a matrix
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDiag                                                          //
//                                                                      //
// Class represents the diagonal of a matrix (for easy manipulation).   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDiag : public TObject {

friend class TMatrix;
friend class TVector;

private:
   const TMatrix  *fMatrix;  //! the matrix I am the diagonal of
   Int_t           fInc;     // if ptr=@a[i,i], then ptr+inc = @a[i+1,i+1]
   Int_t           fNdiag;   // number of diag elems, min(nrows,ncols)
   Real_t         *fPtr;     //! pointer to the a[0,0]

   TMatrixDiag() { fMatrix = 0; fInc = 0; fNdiag = 0; fPtr = 0; }

public:
   TMatrixDiag(const TMatrix &matrix);

   void operator=(Real_t val);
   void operator+=(Double_t val);
   void operator*=(Double_t val);

   void operator=(const TVector &vec);

   const Real_t &operator()(Int_t i) const;
   Real_t &operator()(Int_t i);
   Int_t  GetNdiags() const { return fNdiag; }

   ClassDef(TMatrixDiag,1)  // Diagonal of a matrix
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixPivoting                                                      //
//                                                                      //
// This class inherits from TMatrix and it keeps additional information //
// about what is being/has been pivoted.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixPivoting : public TMatrix {

private:
   typedef Real_t* Index_t;             // wanted to have typeof(index[0])
   Index_t *const fRowIndex;            // fRowIndex[i] = ptr to the i-th
                                        // matrix row, or 0 if the row
                                        // has been pivoted. Note,
                                        // pivoted columns are marked
                                        // by setting fIndex[j] to zero.

                                // Information about the pivot that was
                                // just picked up
   Double_t fPivotValue;                // Value of the pivoting element
   Index_t  fPivotRow;                  // pivot's location (ptrs)
   Index_t  fPivotCol;
   Int_t    fPivotOdd;                  // parity of the pivot
                                        // (0 for even, 1 for odd)

   void PickUpPivot();                  // Pick up a pivot from
                                        // not-pivoted rows and cols

public:
   TMatrixPivoting(const TMatrix &m);
   ~TMatrixPivoting();

   Double_t PivotingAndElimination();   // Perform the pivoting, return
                                        // the pivot value times (-1)^(pi+pj)
                                        // (pi,pj - pivot el row & col)
};

//----- inlines ----------------------------------------------------------------

#if !defined(R__HPUX) && !defined(R__MACOSX)

#ifndef __CINT__

inline TMatrixRow::TMatrixRow(const TMatrix &matrix, Int_t row)
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

inline const Real_t &TMatrixRow::operator()(Int_t i) const
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

inline Real_t &TMatrixRow::operator()(Int_t i)
{
   return (Real_t&)((*(const TMatrixRow *)this)(i));
}

inline TMatrixColumn::TMatrixColumn(const TMatrix &matrix, Int_t col)
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

inline const Real_t &TMatrixColumn::operator()(Int_t i) const
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

inline Real_t &TMatrixColumn::operator()(Int_t i)
{
   return (Real_t&)((*(const TMatrixColumn *)this)(i));
}

inline TMatrixDiag::TMatrixDiag(const TMatrix &matrix)
       : fMatrix(&matrix), fInc(matrix.fNrows+1),
         fNdiag(TMath::Min(matrix.fNrows, matrix.fNcols))
{
   if (!matrix.IsValid()) {
      Error("TMatrixDiag", "matrix is not initialized");
      return;
   }
   fPtr = &(matrix.fElements[0]);
}

inline const Real_t &TMatrixDiag::operator()(Int_t i) const
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

   if (i >= fNdiag || i < 0) {
      Error("TMatrixDiag", "TMatrixDiag index %d is out of diag boundaries [0,%d]",
            i, fNdiag-1);
      return err;
   }

   return fMatrix->fIndex[i][i];
}

inline Real_t &TMatrixDiag::operator()(Int_t i)
{
   return (Real_t&)((*(const TMatrixDiag *)this)(i));
}

#endif

#endif

#endif
