// @(#)root/matrix:$Name:  $:$Id: TMatLazy.cxx,v 1.15 2002/12/10 14:00:48 brun Exp $
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
// Lazy Matrix classes.                                                 //
//                                                                      //
//   TMatrixDLazy                                                       //
//   TMatrixDSymLazy                                                    //
//   THaarMatrixD                                                       //
//   THilbertMatrixD                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDBase.h"

ClassImp(TMatrixDLazy)
ClassImp(TMatrixDSymLazy)
ClassImp(THaarMatrixD)
ClassImp(THilbertMatrixD)

//______________________________________________________________________________
THaarMatrixD::THaarMatrixD(Int_t order,Int_t no_cols)
    : TMatrixDLazy(1<<order, no_cols == 0 ? 1<<order : no_cols)
{
  Assert(order > 0 && no_cols >= 0);
}

//______________________________________________________________________________
void THaarMatrixD::FillIn(TMatrixD &m) const
{
  MakeHaarMat(m);
}

//______________________________________________________________________________
void MakeHaarMat(TMatrixD &m)
{
  // Create an orthonormal (2^n)*(no_cols) Haar (sub)matrix, whose columns
  // are Haar functions. If no_cols is 0, create the complete matrix with
  // 2^n columns. Example, the complete Haar matrix of the second order is:
  // column 1: [ 1  1  1  1]/2
  // column 2: [ 1  1 -1 -1]/2
  // column 3: [ 1 -1  0  0]/sqrt(2)
  // column 4: [ 0  0  1 -1]/sqrt(2)
  // Matrix m is assumed to be zero originally.

  Assert(m.IsValid());
  const Int_t no_rows = m.GetNrows();
  const Int_t no_cols = m.GetNcols();
  Assert(no_rows >= no_cols && no_cols > 0);

  // It is easier to calculate a Haar matrix when the elements are stored
  // column-wise . Since we are row-wise, the transposed Haar is calculted

  TMatrixD mtr(no_cols,no_rows);
        Double_t *cp    = mtr.GetElements();
  const Double_t *m_end = mtr.GetElements()+no_rows*no_cols;

  Double_t norm_factor = 1/TMath::Sqrt((Double_t)no_rows);

  // First row is always 1 (up to normalization)
  Int_t j;
  for (j = 0; j < no_rows; j++)
    *cp++ = norm_factor;

  // The other functions are kind of steps: stretch of 1 followed by the
  // equally long stretch of -1. The functions can be grouped in families
  // according to their order (step size), differing only in the location
  // of the step
  Int_t step_length = no_rows/2;
  while (cp < m_end && step_length > 0) {
    for (Int_t step_position = 0; cp < m_end && step_position < no_rows;
           step_position += 2*step_length, cp += no_rows) {
      Double_t *ccp = cp+step_position;
      for (j = 0; j < step_length; j++)
        *ccp++ = norm_factor;
      for (j = 0; j < step_length; j++)
        *ccp++ = -norm_factor;
    }
    step_length /= 2;
    norm_factor *= TMath::Sqrt(2.0);
  }

  Assert(step_length != 0       || cp == m_end);
  Assert(no_rows     != no_cols || step_length == 0);

  m.Transpose(mtr);
}

//______________________________________________________________________________
THilbertMatrixD::THilbertMatrixD(Int_t no_rows,Int_t no_cols)
    : TMatrixDLazy(no_rows,no_cols)
{
  Assert(no_rows > 0 && no_cols > 0);
}

//______________________________________________________________________________
THilbertMatrixD::THilbertMatrixD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
    : TMatrixDLazy(row_lwb,row_upb,col_lwb,col_upb)
{
  Assert(row_upb-row_lwb+1 > 0 && col_upb-col_lwb+1 > 0);
}

//______________________________________________________________________________
void THilbertMatrixD::FillIn(TMatrixD &m) const
{
  MakeHilbertMat(m);
}

//______________________________________________________________________________
void MakeHilbertMat(TMatrixD &m)
{
  // Make a Hilbert matrix. Hilb[i,j] = 1/(i+j+1),
  // i,j=0...max-1 (matrix need not be a square one).

  Assert(m.IsValid());
  const Int_t no_rows = m.GetNrows();
  const Int_t no_cols = m.GetNcols();
  Assert(no_rows > 0 && no_cols > 0);

  Double_t *cp = m.GetElements();
  for (Int_t i = 0; i < no_rows; i++)
    for (Int_t j = 0; j < no_cols; j++)
      *cp++ = 1.0/(i+j+1.0);
}

//______________________________________________________________________________
THilbertMatrixDSym::THilbertMatrixDSym(Int_t no_rows)
    : TMatrixDSymLazy(no_rows)
{
  Assert(no_rows > 0);
}

//______________________________________________________________________________
THilbertMatrixDSym::THilbertMatrixDSym(Int_t row_lwb,Int_t row_upb)
    : TMatrixDSymLazy(row_lwb,row_upb)
{
  Assert(row_upb-row_lwb+1 > 0);
}

//______________________________________________________________________________
void THilbertMatrixDSym::FillIn(TMatrixDSym &m) const
{
  MakeHilbertMat(m);
}

//______________________________________________________________________________
void MakeHilbertMat(TMatrixDSym &m)
{
  // Make a Hilbert matrix. Hilb[i,j] = 1/(i+j+1),
  // i,j=0...max-1 (matrix must be square).

  Assert(m.IsValid());
  const Int_t no_rows = m.GetNrows();
  Assert(no_rows > 0);

  Double_t *cp = m.GetElements();
  for (Int_t i = 0; i < no_rows; i++)
    for (Int_t j = 0; j < no_rows; j++)
      *cp++ = 1.0/(i+j+1.0);
}
