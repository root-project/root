// @(#)root/matrix:$Name:  $:$Id: TVector.cxx,v 1.11 2002/05/18 08:48:42 brun Exp $
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
// Linear Algebra Package                                               //
//                                                                      //
// The present package implements all the basic algorithms dealing      //
// with vectors, matrices, matrix columns, rows, diagonals, etc.        //
//                                                                      //
// Matrix elements are arranged in memory in a COLUMN-wise              //
// fashion (in FORTRAN's spirit). In fact, it makes it very easy to     //
// feed the matrices to FORTRAN procedures, which implement more        //
// elaborate algorithms.                                                //
//                                                                      //
// Unless otherwise specified, matrix and vector indices always start   //
// with 0, spanning up to the specified limit-1.                        //
//                                                                      //
// The present package provides all facilities to completely AVOID      //
// returning matrices. Use "TMatrix A(TMatrix::kTransposed,B);" and     //
// other fancy constructors as much as possible. If one really needs    //
// to return a matrix, return a TLazyMatrix object instead. The         //
// conversion is completely transparent to the end user, e.g.           //
// "TMatrix m = THaarMatrix(5);" and _is_ efficient.                    //
//                                                                      //
// For usage examples see $ROOTSYS/test/vmatrix.cxx and vvector.cxx     //
// and also:                                                            //
// http://root.cern.ch/root/html/TMatrix.html#TMatrix:description       //
//                                                                      //
// The implementation is based on original code by                      //
// Oleg E. Kiselyov (oleg@pobox.com).                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrix.h"
#include "TROOT.h"
#include "TClass.h"


ClassImp(TVector)

//______________________________________________________________________________
void TVector::Allocate(Int_t nrows, Int_t row_lwb)
{
   // Allocate new vector. Arguments are number of rows and row
   // lowerbound (0 default).

   Invalidate();

   if (nrows <= 0) {
      Error("Allocate", "no of rows has to be positive");
      return;
   }

   fNrows  = nrows;
   fNmem   = nrows;
   fRowLwb = row_lwb;

   //fElements = new Real_t[fNrows];  because of use of ReAlloc()
   fElements = (Real_t*) ::operator new(fNrows*sizeof(Real_t));
   if (fElements)
      memset(fElements, 0, fNrows*sizeof(Real_t));
}

//______________________________________________________________________________
TVector::TVector(Int_t lwb, Int_t upb, Double_t va_(iv1), ...)
{
   // Make a vector and assign initial values. Argument list should contain
   // Double_t values to assign to vector elements. The list must be
   // terminated by the string "END". Example:
   // TVector foo(1,3,0.0,1.0,1.5,"END");

   va_list args;
   va_start(args,va_(iv1));             // Init 'args' to the beginning of
                                        // the variable length list of args

   Allocate(upb-lwb+1, lwb);

   Int_t i;
   (*this)(lwb) = iv1;
   for (i = lwb+1; i <= upb; i++)
      (*this)(i) = (Double_t)va_arg(args,Double_t);

   if (strcmp((char *)va_arg(args,char *),"END"))
      Error("TVector(Int_t, Int_t, ...)", "argument list must be terminated by \"END\"");

   va_end(args);
}

//______________________________________________________________________________
TVector::~TVector()
{
   // TVector destructor.

   if (IsValid())
      ::operator delete(fElements);

   Invalidate();
}

//______________________________________________________________________________
void TVector::Draw(Option_t *option)
{
   // Draw this vector using an intermediate histogram
   // The histogram is named "TVector" by default and no title

   gROOT->ProcessLine(Form("TH1F *R__TVector = new TH1F((TVector&)((TVector*)(0x%lx)));R__TVector->SetBit(kCanDelete);R__TVector->Draw(\"%s\");",
      (Long_t)this,option));
}


//______________________________________________________________________________
void TVector::ResizeTo(Int_t lwb, Int_t upb)
{
   // Resize the vector for a specified number of elements, trying to keep
   // intact as many elements of the old vector as possible. If the vector is
   // expanded, the new elements will be zeroes.

   if (upb-lwb+1 <= 0) {
      Error("ResizeTo", "can't resize vector to a non-positive number of elems");
      return;
   }

   if (!IsValid()) {
      Allocate(upb-lwb+1, lwb);
      return;
   }

   const Int_t old_nrows = fNrows;
   fNrows  = upb-lwb+1;
   fRowLwb = lwb;

   if (old_nrows == fNrows)
      return;                       // The same number of elems

   // If the vector is to grow, reallocate and clear the newly added elements
   if (fNrows > old_nrows) {
      fElements = (Real_t *)TStorage::ReAlloc(fElements, fNrows*sizeof(Real_t),
                                              fNmem*sizeof(Real_t));
      memset(fElements+old_nrows, 0, (fNrows-old_nrows)*sizeof(Real_t));
      fNmem = fNrows;
   } else if (old_nrows - fNrows > (old_nrows>>2)) {
      // Vector is to shrink a lot (more than 1/4 of the original size), reallocate
      fElements = (Real_t *)TStorage::ReAlloc(fElements, fNrows*sizeof(Real_t));
      fNmem = fNrows;
   }

   // If the vector shrinks only a little, don't bother to reallocate

   Assert(fElements != 0);
}

//______________________________________________________________________________
Double_t TVector::Norm1() const
{
   // Compute the 1-norm of the vector SUM{ |v[i]| }.

   if (!IsValid()) {
      Error("Norm1", "vector is not initialized");
      return 0.0;
   }

   Double_t norm = 0;
   Real_t *vp;

   for (vp = fElements; vp < fElements + fNrows; )
      norm += TMath::Abs(*vp++);

   return norm;
}

//______________________________________________________________________________
Double_t TVector::Norm2Sqr() const
{
   // Compute the square of the 2-norm SUM{ v[i]^2 }.

   if (!IsValid()) {
      Error("Norm2Sqr", "vector is not initialized");
      return 0.0;
   }

   Double_t norm = 0;
   Real_t *vp;

   for (vp = fElements; vp < fElements + fNrows; vp++)
      norm += (*vp) * (*vp);

   return norm;
}

//______________________________________________________________________________
Double_t TVector::NormInf() const
{
   // Compute the infinity-norm of the vector MAX{ |v[i]| }.

   if (!IsValid()) {
      Error("NormInf", "vector is not initialized");
      return 0.0;
   }

   Double_t norm = 0;
   Real_t *vp;

   for (vp = fElements; vp < fElements + fNrows; )
      norm = TMath::Max(norm, (Double_t)TMath::Abs(*vp++));

   return norm;
}

//______________________________________________________________________________
Double_t operator*(const TVector &v1, const TVector &v2)
{
   // Compute the scalar product.

   if (!AreCompatible(v1,v2))
      return 0.0;

   Real_t *v1p = v1.fElements;
   Real_t *v2p = v2.fElements;
   Double_t sum = 0.0;

   while (v1p < v1.fElements + v1.fNrows)
      sum += *v1p++ * *v2p++;

   return sum;
}

//______________________________________________________________________________
TVector &TVector::operator*=(Double_t val)
{
   // Multiply every element of the vector with val.

   if (!IsValid()) {
      Error("operator*=", "vector not initialized");
      return *this;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      *ep++ *= val;

   return *this;
}

//______________________________________________________________________________
TVector &TVector::operator*=(const TMatrix &a)
{
   // "Inplace" multiplication target = A*target. A needn't be a square one
   // (the target will be resized to fit).

   if (!a.IsValid()) {
      Error("operator*=(const TMatrix&)", "matrix a is not initialized");
      return *this;
   }
   if (!IsValid()) {
      Error("operator*=(const TMatrix&)", "vector is not initialized");
      return *this;
   }

   if (a.fNcols != fNrows || a.fColLwb != fRowLwb) {
      Error("operator*=(const TMatrix&)", "matrix and vector cannot be multiplied");
      return *this;
   }

   const Int_t old_nrows = fNrows;
   Real_t *old_vector = fElements;        // Save the old vector elem
   fRowLwb = a.fRowLwb;
   Assert((fNrows = a.fNrows) > 0);

   //Assert((fElements = new Real_t[fNrows]) != 0);
   Assert((fElements = (Real_t*) ::operator new(fNrows*sizeof(Real_t))) != 0);
   fNmem = fNrows;

   Real_t *tp = fElements;                     // Target vector ptr
   Real_t *mp = a.fElements;                   // Matrix row ptr
   while (tp < fElements + fNrows) {
      Double_t sum = 0;
      for (const Real_t *sp = old_vector; sp < old_vector + old_nrows; )
         sum += *sp++ * *mp, mp += a.fNrows;
      *tp++ = sum;
      mp -= a.fNelems - 1;       // mp points to the beginning of the next row
   }
   Assert(mp == a.fElements + a.fNrows);

   ::operator delete(old_vector);
   return *this;
}

//______________________________________________________________________________
TVector &TVector::operator=(Real_t val)
{
   // Assign val to every element of the vector.

   if (!IsValid()) {
      Error("operator=", "vector not initialized");
      return *this;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      *ep++ = val;

   return *this;
}

//______________________________________________________________________________
TVector &TVector::operator=(const TMatrixRow &mr)
{
   // Assign a matrix row to a vector. The matrix row is implicitly transposed
   // to allow the assignment in the strict sense.

   if (!IsValid()) {
      Error("operator=(const TMatrixRow&)", "vector is not initialized");
      return *this;
   }
   if (!mr.fMatrix->IsValid()) {
      Error("operator=(const TMatrixRow&)", "matrix is not initialized");
      return *this;
   }

   if (mr.fMatrix->fColLwb != fRowLwb || mr.fMatrix->fNcols != fNrows) {
      Error("operator=(const TMatrixRow&)", "can't assign the transposed row of the matrix to the vector");
      return *this;
   }

   Real_t *rp = mr.fPtr;                       // Row ptr
   Real_t *vp = fElements;                     // Vector ptr
   for ( ; vp < fElements + fNrows; rp += mr.fInc)
      *vp++ = *rp;

   Assert(rp == mr.fPtr + mr.fMatrix->fNelems);

   return *this;
}

//______________________________________________________________________________
TVector &TVector::operator=(const TMatrixColumn &mc)
{
   // Assign a matrix column to a vector.

   if (!IsValid()) {
      Error("operator=(const TMatrixColumn&)", "vector is not initialized");
      return *this;
   }
   if (!mc.fMatrix->IsValid()) {
      Error("operator=(const TMatrixColumn&)", "matrix is not initialized");
      return *this;
   }

   if (mc.fMatrix->fRowLwb != fRowLwb || mc.fMatrix->fNrows != fNrows) {
      Error("operator=(const TMatrixColumn&)", "can't assign a column of the matrix to the vector");
      return *this;
   }

   Real_t *cp = mc.fPtr;                   // Column ptr
   Real_t *vp = fElements;                 // Vector ptr
   while (vp < fElements + fNrows)
      *vp++ = *cp++;

   Assert(cp == mc.fPtr + mc.fMatrix->fNrows);

   return *this;
}

//______________________________________________________________________________
TVector &TVector::operator=(const TMatrixDiag &md)
{
   // Assign the matrix diagonal to a vector.

   if (!IsValid()) {
      Error("operator=(const TMatrixDiag&)", "vector is not initialized");
      return *this;
   }
   if (!md.fMatrix->IsValid()) {
      Error("operator=(const TMatrixDiag&)", "matrix is not initialized");
      return *this;
   }

   if (md.fNdiag != fNrows) {
      Error("operator=(const TMatrixDiag&)", "can't assign the diagonal of the matrix to the vector");
      return *this;
   }

   Real_t *dp = md.fPtr;                  // Diag ptr
   Real_t *vp = fElements;                // Vector ptr
   for ( ; vp < fElements + fNrows; dp += md.fInc)
      *vp++ = *dp;

   Assert(dp < md.fPtr + md.fMatrix->fNelems + md.fInc);

   return *this;
}

//______________________________________________________________________________
TVector &TVector::operator+=(Double_t val)
{
   // Add val to every element of the vector.

   if (!IsValid()) {
      Error("operator+=", "vector not initialized");
      return *this;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      *ep++ += val;

   return *this;
}

//______________________________________________________________________________
TVector &TVector::operator-=(Double_t val)
{
   // Subtract val from every element of the vector.

   if (!IsValid()) {
      Error("operator-=", "vector not initialized");
      return *this;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      *ep++ -= val;

   return *this;
}

//______________________________________________________________________________
Bool_t TVector::operator==(Real_t val) const
{
   // Are all vector elements equal to val?

   if (!IsValid()) {
      Error("operator==", "vector not initialized");
      return kFALSE;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      if (!(*ep++ == val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TVector::operator!=(Real_t val) const
{
   // Are all vector elements not equal to val?

   if (!IsValid()) {
      Error("operator!=", "vector not initialized");
      return kFALSE;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      if (!(*ep++ != val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TVector::operator<(Real_t val) const
{
   // Are all vector elements < val?

   if (!IsValid()) {
      Error("operator<", "vector not initialized");
      return kFALSE;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      if (!(*ep++ < val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TVector::operator<=(Real_t val) const
{
   // Are all vector elements <= val?

   if (!IsValid()) {
      Error("operator<=", "vector not initialized");
      return kFALSE;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      if (!(*ep++ <= val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TVector::operator>(Real_t val) const
{
   // Are all vector elements > val?

   if (!IsValid()) {
      Error("operator>", "vector not initialized");
      return kFALSE;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      if (!(*ep++ > val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TVector::operator>=(Real_t val) const
{
   // Are all vector elements >= val?

   if (!IsValid()) {
      Error("operator>=", "vector not initialized");
      return kFALSE;
   }

   Real_t *ep = fElements;
   while (ep < fElements+fNrows)
      if (!(*ep++ >= val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
TVector &TVector::Abs()
{
   // Take an absolute value of a vector, i.e. apply Abs() to each element.

   if (!IsValid()) {
      Error("Abs", "vector not initialized");
      return *this;
   }

   Real_t *ep;
   for (ep = fElements; ep < fElements+fNrows; ep++)
      *ep = TMath::Abs(*ep);

   return *this;
}

//______________________________________________________________________________
TVector &TVector::Sqr()
{
   // Square each element of the vector.

   if (!IsValid()) {
      Error("Sqr", "vector not initialized");
      return *this;
   }

   Real_t *ep;
   for (ep = fElements; ep < fElements+fNrows; ep++)
      *ep = (*ep) * (*ep);

   return *this;
}

//______________________________________________________________________________
TVector &TVector::Sqrt()
{
   // Take square root of all elements.

   if (!IsValid()) {
      Error("Sqrt", "vector not initialized");
      return *this;
   }

   Real_t *ep;
   for (ep = fElements; ep < fElements+fNrows; ep++)
      if (*ep >= 0)
         *ep = TMath::Sqrt(*ep);
      else
         Error("Sqrt", "(%d)-th element, %g, is negative, can't take the square root",
               (ep-fElements) + fRowLwb, *ep);

   return *this;
}

//______________________________________________________________________________
TVector &TVector::Apply(TElementAction &action)
{
   // Apply action to each element of the vector.

   if (!IsValid())
      Error("Apply(TElementAction&)", "vector not initialized");
   else
      for (Real_t *ep = fElements; ep < fElements+fNrows; ep++)
         action.Operation(*ep);
   return *this;
}

//______________________________________________________________________________
TVector &TVector::Apply(TElementPosAction &action)
{
   // Apply action to each element of the vector. In action the location
   // of the current element is known.

   if (!IsValid()) {
      Error("Apply(TElementPosAction&)", "vector not initialized");
      return *this;
   }

   Real_t *ep = fElements;
   for (action.fI = fRowLwb; action.fI < fRowLwb+fNrows; action.fI++)
      action.Operation(*ep++);

   Assert(ep == fElements+fNrows);

   return *this;
}

//______________________________________________________________________________
Bool_t operator==(const TVector &v1, const TVector &v2)
{
   // Check to see if two vectors are identical.

   if (!AreCompatible(v1, v2)) return kFALSE;
   return (memcmp(v1.fElements, v2.fElements, v1.fNrows*sizeof(Real_t)) == 0);
}

//______________________________________________________________________________
TVector &operator+=(TVector &target, const TVector &source)
{
   // Add the source vector to the target vector.

   if (!AreCompatible(target, source)) {
      Error("operator+=", "vectors are not compatible");
      return target;
   }

   Real_t *sp = source.fElements;
   Real_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNrows; )
      *tp++ += *sp++;

   return target;
}

//______________________________________________________________________________
TVector &operator-=(TVector &target, const TVector &source)
{
   // Subtract the source vector from the target vector.

   if (!AreCompatible(target, source)) {
      Error("operator-=", "vectors are not compatible");
      return target;
   }

   Real_t *sp = source.fElements;
   Real_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNrows; )
      *tp++ -= *sp++;

   return target;
}

//______________________________________________________________________________
TVector &Add(TVector &target, Double_t scalar, const TVector &source)
{
   // Modify addition: target += scalar * source.

   if (!AreCompatible(target, source)) {
      Error("Add", "vectors are not compatible");
      return target;
   }

   Real_t *sp = source.fElements;
   Real_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNrows; )
      *tp++ += scalar * (*sp++);

   return target;
}

//______________________________________________________________________________
TVector &ElementMult(TVector &target, const TVector &source)
{
   // Multiply target by the source, element-by-element.

   if (!AreCompatible(target, source)) {
      Error("ElementMult", "vectors are not compatible");
      return target;
   }

   Real_t *sp = source.fElements;
   Real_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNrows; )
      *tp++ *= *sp++;

   return target;
}

//______________________________________________________________________________
TVector &ElementDiv(TVector &target, const TVector &source)
{
   // Divide target by the source, element-by-element.

   if (!AreCompatible(target, source)) {
      Error("ElementDiv", "vectors are not compatible");
      return target;
   }

   Real_t *sp = source.fElements;
   Real_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNrows; )
      *tp++ /= *sp++;

   return target;
}

//______________________________________________________________________________
void TVector::Print(Option_t *) const
{
   // Print the vector as a list of elements.

   if (!IsValid()) {
      Error("Print", "vector not initialized");
      return;
   }

   printf("\nVector %d is as follows", fNrows);

   printf("\n\n     |   %6d  |", 1);
   printf("\n%s\n", "------------------");
   for (Int_t i = 0; i < fNrows; i++) {
      printf("%4d |", i+fRowLwb);
      printf("%11.4g \n", (*this)(i+fRowLwb));
   }
   printf("\n");
}

//______________________________________________________________________________
void TVector::Streamer(TBuffer &R__b)
{
   // Stream an object of class TVector.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TVector::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         fNmem   = fNrows;
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      R__b >> fRowLwb;
      fNrows = R__b.ReadArray(fElements);
      R__b.CheckByteCount(R__s, R__c, TVector::IsA());
      //====end of old versions

   } else {
      TVector::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void Compare(const TVector &v1, const TVector &v2)
{
   // Compare two vectors and print out the result of the comparison.

   Int_t i;

   if (!AreCompatible(v1, v2)) {
      Error("Compare", "vectors are not compatible");
      return;
   }

   printf("\n\nComparison of two TVectors:\n");

   Double_t norm1 = 0, norm2 = 0;       // Norm of the Matrices
   Double_t ndiff = 0;                  // Norm of the difference
   Int_t    imax = 0;                   // For the elements that differ most
   Real_t   difmax = -1;
   Real_t  *mp1 = v1.fElements;         // Vector element pointers
   Real_t  *mp2 = v2.fElements;

   for (i = 0; i < v1.fNrows; i++) {
      Real_t mv1 = *mp1++;
      Real_t mv2 = *mp2++;
      Real_t diff = TMath::Abs(mv1-mv2);

      if (diff > difmax) {
         difmax = diff;
         imax = i;
      }
      norm1 += TMath::Abs(mv1);
      norm2 += TMath::Abs(mv2);
      ndiff += TMath::Abs(diff);
   }

   imax += v1.fRowLwb;
   printf("\nMaximal discrepancy    \t\t%g", difmax);
   printf("\n   occured at the point\t\t(%d)", imax);
   const Real_t mv1 = v1(imax);
   const Real_t mv2 = v2(imax);
   printf("\n Vector 1 element is    \t\t%g", mv1);
   printf("\n Vector 2 element is    \t\t%g", mv2);
   printf("\n Absolute error v2[i]-v1[i]\t\t%g", mv2-mv1);
   printf("\n Relative error\t\t\t\t%g\n",
          (mv2-mv1)/TMath::Max(TMath::Abs(mv2+mv1)/2,(Real_t)1e-7));

   printf("\n||Vector 1||   \t\t\t%g", norm1);
   printf("\n||Vector 2||   \t\t\t%g", norm2);
   printf("\n||Vector1-Vector2||\t\t\t\t%g", ndiff);
   printf("\n||Vector1-Vector2||/sqrt(||Vector1|| ||Vector2||)\t%g\n\n",
          ndiff/TMath::Max(TMath::Sqrt(norm1*norm2), 1e-7));
}

//______________________________________________________________________________
void VerifyElementValue(const TVector &v, Real_t val)
{
   // Validate that all elements of vector have value val (within 1.e-5).

   Int_t    imax = 0;
   Double_t max_dev = 0;
   Int_t    i;

   for (i = v.GetLwb(); i <= v.GetUpb(); i++) {
      Double_t dev = TMath::Abs(v(i)-val);
      if (dev > max_dev)
         imax = i, max_dev = dev;
   }

   if (max_dev == 0)
      return;
   else if(max_dev < 1e-5)
      printf("Element (%d) with value %g differs the most from what\n"
             "was expected, %g, though the deviation %g is small\n",
             imax, v(imax), val, max_dev);
   else
      Error("VerifyElementValue", "a significant difference from the expected value %g\n"
            "encountered for element (%d) with value %g",
            val, imax, v(imax));
}

//______________________________________________________________________________
void VerifyVectorIdentity(const TVector &v1, const TVector &v2)
{
   // Verify that elements of the two vectors are equal (within 1.e-5).

   Int_t    imax = 0;
   Double_t max_dev = 0;
   Int_t    i;

   if (!AreCompatible(v1, v2)) {
      Error("VerifyVectorIdentity", "vectors are not compatible");
      return;
   }

   for (i = v1.GetLwb(); i <= v1.GetUpb(); i++) {
      Double_t dev = TMath::Abs(v1(i)-v2(i));
      if (dev > max_dev)
         imax = i, max_dev = dev;
   }

   if (max_dev == 0)
      return;
   if (max_dev < 1e-5)
      printf("Two (%d) elements of vectors with values %g and %g\n"
             "differ the most, though the deviation %g is small\n",
             imax, v1(imax), v2(imax), max_dev);
   else
      Error("VerifyVectorIdentity", "a significant difference between the vectors encountered\n"
            "at (%d) element, with values %g and %g",
            imax, v1(imax), v2(imax));
}



#if defined(R__HPUX) || defined(R__MACOSX)

//______________________________________________________________________________
//  These functions should be inline
//______________________________________________________________________________

TVector::TVector(Int_t n)
{
   Allocate(n);
}

TVector::TVector(Int_t lwb, Int_t upb)
{
   Allocate(upb-lwb+1, lwb);
}

Bool_t TVector::IsValid() const
{
   if (fNrows == -1)
      return kFALSE;
   return kTRUE;
}

void TVector::SetElements(const Float_t *elements)
{
  if (!IsValid()) {
    Error("SetElements", "vector is not initialized");
    return;
  }
  memcpy(fElements,elements,fNrows*sizeof(Float_t));
}

TVector::TVector(Int_t n, const Float_t *elements)
{
   Allocate(n);
   SetElements(elements);
}

TVector::TVector(Int_t lwb, Int_t upb, const Float_t *elements)
{
   Allocate(upb-lwb+1, lwb);
   SetElements(elements);
}


Bool_t AreCompatible(const TVector &v1, const TVector &v2)
{
   if (!v1.IsValid()) {
      Error("AreCompatible", "vector 1 not initialized");
      return kFALSE;
   }
   if (!v2.IsValid()) {
      Error("AreCompatible", "vector 2 not initialized");
      return kFALSE;
   }

   if (v1.fNrows != v2.fNrows || v1.fRowLwb != v2.fRowLwb)
      return kFALSE;

   return kTRUE;
}

TVector &TVector::operator=(const TVector &source)
{
   if (this != &source && AreCompatible(*this, source)) {
      TObject::operator=(source);
      memcpy(fElements, source.fElements, fNrows*sizeof(Real_t));
   }
   return *this;
}

TVector::TVector(const TVector &another) : TObject(another)
{
   if (another.IsValid()) {
      Allocate(another.GetUpb()-another.GetLwb()+1, another.GetLwb());
      *this = another;
   } else
      Error("TVector(const TVector&)", "other vector is not valid");
}

void TVector::ResizeTo(Int_t n)
{
   TVector::ResizeTo(0,n-1);
}

void TVector::ResizeTo(const TVector &v)
{
   TVector::ResizeTo(v.GetLwb(), v.GetUpb());
}

Real_t &TVector::operator()(Int_t ind) const
{
   static Real_t err;
   err = 0.0;

   if (!IsValid()) {
      Error("operator()", "vector is not initialized");
      return err;
   }

   Int_t aind = ind - fRowLwb;
   if (aind >= fNrows || aind < 0) {
      Error("operator()", "requested element %d is out of vector boundaries [%d,%d]",
            ind, fRowLwb, fNrows+fRowLwb-1);
      return err;
   }

   return fElements[aind];
}

Real_t &TVector::operator()(Int_t index)
{
   return (Real_t&)((*(const TVector *)this)(index));
}

TVector &TVector::Zero()
{
   if (!IsValid())
      Error("Zero", "vector not initialized");
   else
      memset(fElements, 0, fNrows*sizeof(Real_t));
   return *this;
}

#endif
