// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann  Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TMatrixTLazy
    \ingroup Matrix

 Templates of Lazy Matrix classes.
~~~
   TMatrixTLazy
   TMatrixTSymLazy
   THaarMatrixT
   THilbertMatrixT
   THilbertMatrixTSym
~~~
*/

#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TMatrixTLazy.h"
#include "TMath.h"

templateClassImp(TMatrixTLazy);
templateClassImp(TMatrixTSymLazy);
templateClassImp(THaarMatrixT);
templateClassImp(THilbertMatrixT);
templateClassImp(THilbertMatrixTSym);


////////////////////////////////////////////////////////////////////////////////

template<class Element>
THaarMatrixT<Element>::THaarMatrixT(Int_t order,Int_t no_cols)
    : TMatrixTLazy<Element>(1<<order, no_cols == 0 ? 1<<order : no_cols)
{
   if (order <= 0)
      Error("THaarMatrixT","Haar order(%d) should be > 0",order);
   if (no_cols < 0)
      Error("THaarMatrixT","#cols(%d) in Haar should be >= 0",no_cols);
}

////////////////////////////////////////////////////////////////////////////////
/// Create an orthonormal (2^n)*(no_cols) Haar (sub)matrix, whose columns
/// are Haar functions. If no_cols is 0, create the complete matrix with
/// 2^n columns. Example, the complete Haar matrix of the second order is:
/// column 1: [ 1  1  1  1]/2
/// column 2: [ 1  1 -1 -1]/2
/// column 3: [ 1 -1  0  0]/sqrt(2)
/// column 4: [ 0  0  1 -1]/sqrt(2)
/// Matrix m is assumed to be zero originally.

template<class Element>
void MakeHaarMat(TMatrixT<Element> &m)
{
   R__ASSERT(m.IsValid());
   const Int_t no_rows = m.GetNrows();
   const Int_t no_cols = m.GetNcols();

   if (no_rows < no_cols) {
      Error("MakeHaarMat","#rows(%d) should be >= #cols(%d)",no_rows,no_cols);
      return;
   }
   if (no_cols <= 0) {
      Error("MakeHaarMat","#cols(%d) should be > 0",no_cols);
      return;
   }

   // It is easier to calculate a Haar matrix when the elements are stored
   // column-wise . Since we are row-wise, the transposed Haar is calculated

   TMatrixT<Element> mtr(no_cols,no_rows);
         Element *cp    = mtr.GetMatrixArray();
   const Element *m_end = mtr.GetMatrixArray()+no_rows*no_cols;

   Element norm_factor = 1/TMath::Sqrt((Element)no_rows);

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
         Element *ccp = cp+step_position;
         for (j = 0; j < step_length; j++)
            *ccp++ = norm_factor;
         for (j = 0; j < step_length; j++)
            *ccp++ = -norm_factor;
      }
      step_length /= 2;
      norm_factor *= TMath::Sqrt(2.0);
   }

   R__ASSERT(step_length != 0       || cp == m_end);
   R__ASSERT(no_rows     != no_cols || step_length == 0);

   m.Transpose(mtr);
}

////////////////////////////////////////////////////////////////////////////////

template<class Element>
void THaarMatrixT<Element>::FillIn(TMatrixT<Element> &m) const
{
   MakeHaarMat(m);
}

////////////////////////////////////////////////////////////////////////////////

template<class Element>
THilbertMatrixT<Element>::THilbertMatrixT(Int_t no_rows,Int_t no_cols)
    : TMatrixTLazy<Element>(no_rows,no_cols)
{
   if (no_rows <= 0)
      Error("THilbertMatrixT","#rows(%d) in Hilbert should be > 0",no_rows);
   if (no_cols <= 0)
      Error("THilbertMatrixT","#cols(%d) in Hilbert should be > 0",no_cols);
}

////////////////////////////////////////////////////////////////////////////////

template<class Element>
THilbertMatrixT<Element>::THilbertMatrixT(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
    : TMatrixTLazy<Element>(row_lwb,row_upb,col_lwb,col_upb)
{
   if (row_upb < row_lwb)
      Error("THilbertMatrixT","row_upb(%d) in Hilbert should be >= row_lwb(%d)",row_upb,row_lwb);
   if (col_upb < col_lwb)
      Error("THilbertMatrixT","col_upb(%d) in Hilbert should be >= col_lwb(%d)",col_upb,col_lwb);
}

////////////////////////////////////////////////////////////////////////////////
/// Make a Hilbert matrix. Hilb[i,j] = 1/(i+j+1),
/// i,j=0...max-1 (matrix need not be a square one).

template<class Element>
void MakeHilbertMat(TMatrixT<Element> &m)
{
   R__ASSERT(m.IsValid());
   const Int_t no_rows = m.GetNrows();
   const Int_t no_cols = m.GetNcols();

   if (no_rows <= 0) {
      Error("MakeHilbertMat","#rows(%d) should be > 0",no_rows);
      return;
   }
   if (no_cols <= 0) {
      Error("MakeHilbertMat","#cols(%d) should be > 0",no_cols);
      return;
   }

   Element *cp = m.GetMatrixArray();
   for (Int_t i = 0; i < no_rows; i++)
      for (Int_t j = 0; j < no_cols; j++)
         *cp++ = 1.0/(i+j+1.0);
}

////////////////////////////////////////////////////////////////////////////////

template<class Element>
void THilbertMatrixT<Element>::FillIn(TMatrixT<Element> &m) const
{
   MakeHilbertMat(m);
}

////////////////////////////////////////////////////////////////////////////////

template<class Element>
THilbertMatrixTSym<Element>::THilbertMatrixTSym(Int_t no_rows)
    : TMatrixTSymLazy<Element>(no_rows)
{
   if (no_rows <= 0)
      Error("THilbertMatrixTSym","#rows(%d) in Hilbert should be > 0",no_rows);
}

////////////////////////////////////////////////////////////////////////////////

template<class Element>
THilbertMatrixTSym<Element>::THilbertMatrixTSym(Int_t row_lwb,Int_t row_upb)
    : TMatrixTSymLazy<Element>(row_lwb,row_upb)
{
   if (row_upb < row_lwb)
      Error("THilbertMatrixTSym","row_upb(%d) in Hilbert should be >= row_lwb(%d)",row_upb,row_lwb);
}

////////////////////////////////////////////////////////////////////////////////
/// Make a Hilbert matrix. Hilb[i,j] = 1/(i+j+1),
/// i,j=0...max-1 (matrix must be square).

template<class Element>
void MakeHilbertMat(TMatrixTSym<Element> &m)
{
   R__ASSERT(m.IsValid());
   const Int_t no_rows = m.GetNrows();
   if (no_rows <= 0) {
      Error("MakeHilbertMat","#rows(%d) should be > 0",no_rows);
      return;
   }

   Element *cp = m.GetMatrixArray();
   for (Int_t i = 0; i < no_rows; i++)
      for (Int_t j = 0; j < no_rows; j++)
         *cp++ = 1.0/(i+j+1.0);
}

////////////////////////////////////////////////////////////////////////////////

template<class Element>
void THilbertMatrixTSym<Element>::FillIn(TMatrixTSym<Element> &m) const
{
   MakeHilbertMat(m);
}

template class TMatrixTLazy      <Float_t>;
template class TMatrixTSymLazy   <Float_t>;
template class THaarMatrixT      <Float_t>;
template class THilbertMatrixT   <Float_t>;
template class THilbertMatrixTSym<Float_t>;

template class TMatrixTLazy      <Double_t>;
template class TMatrixTSymLazy   <Double_t>;
template class THaarMatrixT      <Double_t>;
template class THilbertMatrixT   <Double_t>;
template class THilbertMatrixTSym<Double_t>;
