// @(#)root/matrix:$Name:  $:$Id: TMatrixD.cxx,v 1.19 2002/05/10 08:46:51 brun Exp $
// Author: Fons Rademakers   03/11/97

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
// with 0, spanning up to the specified limit-1. However, there are     //
// constructors to which one can specify aribtrary lower and upper      //
// bounds, e.g. TMatrixD m(1,10,1,5) defines a matrix that ranges, and  //
// that can be addresses, from 1..10, 1..5 (a(1,1)..a(10,5)).           //
//                                                                      //
// The present package provides all facilities to completely AVOID      //
// returning matrices. Use "TMatrixD A(TMatrixD::kTransposed,B);" and   //
// other fancy constructors as much as possible. If one really needs    //
// to return a matrix, return a TLazyMatrixD object instead. The        //
// conversion is completely transparent to the end user, e.g.           //
// "TMatrixD m = THaarMatrix(5);" and _is_ efficient.                   //
//                                                                      //
// Since TMatrixD et al. are fully integrated in ROOT they of course    //
// can be stored in a ROOT database.                                    //
//                                                                      //
//                                                                      //
// How to efficiently use this package                                  //
// -----------------------------------                                  //
//                                                                      //
// 1. Never return complex objects (matrices or vectors)                //
//    Danger: For example, when the following snippet:                  //
//        TMatrixD foo(int n)                                           //
//        {                                                             //
//           TMatrixD foom(n,n); fill_in(foom); return foom;            //
//        }                                                             //
//        TMatrixD m = foo(5);                                          //
//    runs, it constructs matrix foo:foom, copies it onto stack as a    //
//    return value and destroys foo:foom. Return value (a matrix)       //
//    from foo() is then copied over to m (via a copy constructor),     //
//    and the return value is destroyed. So, the matrix constructor is  //
//    called 3 times and the destructor 2 times. For big matrices,      //
//    the cost of multiple constructing/copying/destroying of objects   //
//    may be very large. *Some* optimized compilers can cut down on 1   //
//    copying/destroying, but still it leaves at least two calls to     //
//    the constructor. Note, TLazyMatrices (see below) can construct    //
//    TMatrixD m "inplace", with only a _single_ call to the            //
//    constructor.                                                      //
//                                                                      //
// 2. Use "two-address instructions"                                    //
//        "void TMatrixD::operator += (const TMatrixD &B);"             //
//    as much as possible.                                              //
//    That is, to add two matrices, it's much more efficient to write   //
//        A += B;                                                       //
//    than                                                              //
//        TMatrixD C = A + B;                                           //
//    (if both operand should be preserved,                             //
//        TMatrixD C = A; C += B;                                       //
//    is still better).                                                 //
//                                                                      //
// 3. Use glorified constructors when returning of an object seems      //
//    inevitable:                                                       //
//        "TMatrixD A(TMatrixD::kTransposed,B);"                        //
//        "TMatrixD C(A,TMatrixD::kTransposeMult,B);"                   //
//                                                                      //
//    like in the following snippet (from $ROOTSYS/test/vmatrix.cxx)    //
//    that verifies that for an orthogonal matrix T, T'T = TT' = E.     //
//                                                                      //
//    TMatrixD haar = THaarMatrix(5);                                   //
//    TMatrixD unit(TMatrixD::kUnit,haar);                              //
//    TMatrixD haar_t(TMatrixD::kTransposed,haar);                      //
//    TMatrixD hth(haar,TMatrixD::kTransposeMult,haar);                 //
//    TMatrixD hht(haar,TMatrixD::kMult,haar_t);                        //
//    TMatrixD hht1 = haar; hht1 *= haar_t;                             //
//    VerifyMatrixIdentity(unit,hth);                                   //
//    VerifyMatrixIdentity(unit,hht);                                   //
//    VerifyMatrixIdentity(unit,hht1);                                  //
//                                                                      //
// 4. Accessing row/col/diagonal of a matrix without much fuss          //
//    (and without moving a lot of stuff around):                       //
//                                                                      //
//        TMatrixD m(n,n); TVectorD v(n); TMatrixDDiag(m) += 4;         //
//        v = TMatrixDRow(m,0);                                         //
//        TMatrixDColumn m1(m,1); m1(2) = 3; // the same as m(2,1)=3;   //
//    Note, constructing of, say, TMatrixDDiag does *not* involve any   //
//    copying of any elements of the source matrix.                     //
//                                                                      //
// 5. It's possible (and encouraged) to use "nested" functions          //
//    For example, creating of a Hilbert matrix can be done as follows: //
//                                                                      //
//    void foo(const TMatrixD &m)                                       //
//    {                                                                 //
//      TMatrixD m1(TMatrixD::kZero,m);                                 //
//      struct MakeHilbert : public TElementPosAction {                 //
//        void Operation(Double_t &element) { element = 1./(fI+fJ-1); } //
//      };                                                              //
//      m1.Apply(MakeHilbert());                                        //
//    }                                                                 //
//                                                                      //
//    of course, using a special method TMatrixD::HilbertMatrix() is    //
//    still more optimal, but not by a whole lot. And that's right,     //
//    class MakeHilbert is declared *within* a function and local to    //
//    that function. It means one can define another MakeHilbert class  //
//    (within another function or outside of any function, that is, in  //
//    the global scope), and it still will be OK. Note, this currently  //
//    is not yet supported by the interpreter CINT.                     //
//                                                                      //
//    Another example is applying of a simple function to each matrix   //
//    element:                                                          //
//                                                                      //
//    void foo(TMatrixD &m, TMatrixD &m1)                               //
//    {                                                                 //
//      typedef  double (*dfunc_t)(double);                             //
//      class ApplyFunction : public TElementActionD {                  //
//        dfunc_t fFunc;                                                //
//        void Operation(Double_t &element) { element=fFunc(element); } //
//      public:                                                         //
//        ApplyFunction(dfunc_t func):fFunc(func) {}                    //
//      };                                                              //
//      ApplyFunction x(TMath::Sin);                                    //
//      m.Apply(x);                                                     //
//    }                                                                 //
//                                                                      //
//    Validation code $ROOTSYS/test/vmatrix.cxx and vvector.cxx contain //
//    a few more examples of that kind.                                 //
//                                                                      //
// 6. Lazy matrices: instead of returning an object return a "recipe"   //
//    how to make it. The full matrix would be rolled out only when     //
//    and where it's needed:                                            //
//       TMatrixD haar = THaarMatrix(5);                                //
//    THaarMatrix() is a *class*, not a simple function. However        //
//    similar this looks to a returning of an object (see note #1       //
//    above), it's dramatically different. THaarMatrix() constructs a   //
//    TLazyMatrixD, an object of just a few bytes long. A special       //
//    "TMatrixD(const TLazyMatrixD &recipe)" constructor follows the    //
//    recipe and makes the matrix haar() right in place. No matrix      //
//    element is moved whatsoever!                                      //
//                                                                      //
// The implementation is based on original code by                      //
// Oleg E. Kiselyov (oleg@pobox.com).                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixD.h"
#include "TROOT.h"
#include "TClass.h"



ClassImp(TMatrixD)

//______________________________________________________________________________
void TMatrixD::Allocate(Int_t no_rows, Int_t no_cols, Int_t row_lwb, Int_t col_lwb)
{
   // Allocate new matrix. Arguments are number of rows, columns, row
   // lowerbound (0 default) and column lowerbound (0 default).

   Invalidate();

   if (no_rows <= 0) {
      Error("Allocate", "no of rows has to be positive");
      return;
   }
   if (no_cols <= 0) {
      Error("Allocate", "no of columns has to be positive");
      return;
   }

   fNrows  = no_rows;
   fNcols  = no_cols;
   fRowLwb = row_lwb;
   fColLwb = col_lwb;
   fNelems = fNrows * fNcols;

   fElements = new Double_t[fNelems];
   if (fElements)
      memset(fElements, 0, fNelems*sizeof(Double_t));

   if (fNcols == 1) {          // Only one col - fIndex is dummy actually
      fIndex = &fElements;
      return;
   }

   fIndex = new Double_t*[fNcols];
   if (fIndex)
      memset(fIndex, 0, fNcols*sizeof(Double_t*));

   Int_t i;
   Double_t *col_p;
   for (i = 0, col_p = &fElements[0]; i < fNcols; i++, col_p += fNrows)
      fIndex[i] = col_p;
}

//______________________________________________________________________________
TMatrixD::~TMatrixD()
{
   // TMatrixD destructor.

   if (IsValid()) {
      if (fNcols != 1)
         delete [] fIndex;
      delete [] fElements;
   }

   Invalidate();
}

//______________________________________________________________________________
void TMatrixD::Draw(Option_t *option)
{
   // Draw this matrix using an intermediate histogram
   // The histogram is named "TMatrixD" by default and no title

   gROOT->ProcessLine(Form("TH2D *R__TV = new TH2D((TMatrixD&)((TMatrixD*)(0x%lx)));R__TV->SetBit(kCanDelete);R__TV->Draw(\"%s\");",
      (Long_t)this,option));
}

//______________________________________________________________________________
void TMatrixD::ResizeTo(Int_t nrows, Int_t ncols)
{
   // Erase the old matrix and create a new one according to new boundaries
   // with indexation starting at 0.

   if (IsValid()) {
      if (fNrows == nrows && fNcols == ncols)
         return;

      if (fNcols != 1)
         delete [] fIndex;
      delete [] fElements;
   }

   Allocate(nrows, ncols);
}

//______________________________________________________________________________
void TMatrixD::ResizeTo(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb)
{
   // Erase the old matrix and create a new one according to new boudaries.

   Int_t new_nrows = row_upb - row_lwb + 1;
   Int_t new_ncols = col_upb - col_lwb + 1;

   if (IsValid()) {
      fRowLwb = row_lwb;
      fColLwb = col_lwb;

      if (fNrows == new_nrows && fNcols == new_ncols)
         return;

      if (fNcols != 1)
         delete [] fIndex;
      delete [] fElements;
   }

   Allocate(new_nrows, new_ncols, row_lwb, col_lwb);
}

//______________________________________________________________________________
TMatrixD::TMatrixD(EMatrixCreatorsOp1 op, const TMatrixD &prototype)
{
   // Create a matrix applying a specific operation to the prototype.
   // Example: TMatrixD a(10,12); ...; TMatrixD b(TMatrixD::kTransposed, a);
   // Supported operations are: kZero, kUnit, kTransposed, kInverted and kInvertedPosDef.

   Invalidate();

   if (!prototype.IsValid()) {
      Error("TMatrixD(EMatrixCreatorOp1)", "prototype matrix not initialized");
      return;
   }

   switch(op) {
      case kZero:
         Allocate(prototype.fNrows, prototype.fNcols,
                  prototype.fRowLwb, prototype.fColLwb);
         break;

      case kUnit:
         Allocate(prototype.fNrows, prototype.fNcols,
                  prototype.fRowLwb, prototype.fColLwb);
         UnitMatrix();
         break;

      case kTransposed:
         Transpose(prototype);
         break;

      case kInverted:
         Invert(prototype);
         break;

      case kInvertedPosDef:
         InvertPosDef(prototype);
         break;

      default:
         Error("TMatrixD(EMatrixCreatorOp1)", "operation %d not yet implemented", op);
   }
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixD &a, EMatrixCreatorsOp2 op, const TMatrixD &b)
{
   // Create a matrix applying a specific operation to two prototypes.
   // Example: TMatrixD a(10,12), b(12,5); ...; TMatrixD c(a, TMatrixD::kMult, b);
   // Supported operations are: kMult (a*b), kTransposeMult (a'*b),
   // kInvMult,kInvPosDefMult (a^(-1)*b) and kAtBA (a'*b*a).

   Invalidate();

   if (!a.IsValid()) {
      Error("TMatrixD(EMatrixCreatorOp2)", "matrix a not initialized");
      return;
   }
   if (!b.IsValid()) {
      Error("TMatrixD(EMatrixCreatorOp2)", "matrix b not initialized");
      return;
   }

   switch(op) {
      case kMult:
         AMultB(a, b);
         break;

      case kTransposeMult:
         AtMultB(a, b);
         break;

      default:
         Error("TMatrixD(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
   }
}

//______________________________________________________________________________
TMatrixD &TMatrixD::UnitMatrix()
{
   // make a unit matrix (matrix need not be a square one). The matrix
   // is traversed in the natural (that is, column by column) order.

   if (!IsValid()) {
      Error("UnitMatrix", "matrix not initialized");
      return *this;
   }

   Double_t *ep = fElements;
   Int_t i, j;

   for (j = 0; j < fNcols; j++)
      for (i = 0; i < fNrows; i++)
         *ep++ = (i==j ? 1.0 : 0.0);

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::HilbertMatrix()
{
   // Make a Hilbert matrix. Hilb[i,j] = 1/(i+j-1), i,j=1...max, OR
   // Hilb[i,j] = 1/(i+j+1), i,j=0...max-1 (matrix need not be a square one).
   // The matrix is traversed in the natural (that is, column by column) order.

   if (!IsValid()) {
      Error("HilbertMatrix", "matrix not initialized");
      return *this;
   }

   Double_t *ep = fElements;
   Int_t i, j;

   for (j = 0; j < fNcols; j++)
      for (i = 0; i < fNrows; i++)
         *ep++ = 1./(i+j+1);

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator=(Double_t val)
{
   // Assign val to every element of the matrix.

   if (!IsValid()) {
      Error("operator=", "matrix not initialized");
      return *this;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      *ep++ = val;

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator+=(Double_t val)
{
   // Add val to every element of the matrix.

   if (!IsValid()) {
      Error("operator+=", "matrix not initialized");
      return *this;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      *ep++ += val;

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator-=(Double_t val)
{
   // Subtract val from every element of the matrix.

   if (!IsValid()) {
      Error("operator-=", "matrix not initialized");
      return *this;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      *ep++ -= val;

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(Double_t val)
{
   // Multiply every element of the matrix with val.

   if (!IsValid()) {
      Error("operator*=", "matrix not initialized");
      return *this;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      *ep++ *= val;

   return *this;
}

//______________________________________________________________________________
Bool_t TMatrixD::operator==(Double_t val) const
{
   // Are all matrix elements equal to val?

   if (!IsValid()) {
      Error("operator==", "matrix not initialized");
      return kFALSE;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      if (!(*ep++ == val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixD::operator!=(Double_t val) const
{
   // Are all matrix elements not equal to val?

   if (!IsValid()) {
      Error("operator!=", "matrix not initialized");
      return kFALSE;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      if (!(*ep++ != val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixD::operator<(Double_t val) const
{
   // Are all matrix elements < val?

   if (!IsValid()) {
      Error("operator<", "matrix not initialized");
      return kFALSE;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      if (!(*ep++ < val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixD::operator<=(Double_t val) const
{
   // Are all matrix elements <= val?

   if (!IsValid()) {
      Error("operator<=", "matrix not initialized");
      return kFALSE;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      if (!(*ep++ <= val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixD::operator>(Double_t val) const
{
   // Are all matrix elements > val?

   if (!IsValid()) {
      Error("operator>", "matrix not initialized");
      return kFALSE;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      if (!(*ep++ > val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixD::operator>=(Double_t val) const
{
   // Are all matrix elements >= val?

   if (!IsValid()) {
      Error("operator>=", "matrix not initialized");
      return kFALSE;
   }

   Double_t *ep = fElements;
   while (ep < fElements+fNelems)
      if (!(*ep++ >= val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Abs()
{
   // Take an absolute value of a matrix, i.e. apply Abs() to each element.

   if (!IsValid()) {
      Error("Abs", "matrix not initialized");
      return *this;
   }

   Double_t *ep;
   for (ep = fElements; ep < fElements+fNelems; ep++)
      *ep = TMath::Abs(*ep);

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Sqr()
{
   // Square each element of the matrix.

   if (!IsValid()) {
      Error("Sqr", "matrix not initialized");
      return *this;
   }

   Double_t *ep;
   for (ep = fElements; ep < fElements+fNelems; ep++)
      *ep = (*ep) * (*ep);

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Sqrt()
{
   // Take square root of all elements.

   if (!IsValid()) {
      Error("Sqrt", "matrix not initialized");
      return *this;
   }

   Double_t *ep;
   for (ep = fElements; ep < fElements+fNelems; ep++)
      if (*ep >= 0)
         *ep = TMath::Sqrt(*ep);
      else
         Error("Sqrt", "(%d,%d)-th element, %g, is negative, can't take the square root",
               (ep-fElements) % fNrows + fRowLwb,
               (ep-fElements) / fNrows + fColLwb, *ep);

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Apply(TElementPosActionD &action)
{
   // Apply action to each element of the matrix. In action the location
   // of the current element is known. The matrix is traversed in the
   // natural (that is, column by column) order.

   if (!IsValid()) {
      Error("Apply(TElementPosActionD&)", "matrix not initialized");
      return *this;
   }

   Double_t *ep = fElements;
   for (action.fJ = fColLwb; action.fJ < fColLwb+fNcols; action.fJ++)
      for (action.fI = fRowLwb; action.fI < fRowLwb+fNrows; action.fI++)
         action.Operation(*ep++);

   Assert(ep == fElements+fNelems);

   return *this;
}

//______________________________________________________________________________
Bool_t operator==(const TMatrixD &im1, const TMatrixD &im2)
{
   // Check to see if two matrices are identical.

   if (!AreCompatible(im1, im2)) return kFALSE;
   return (memcmp(im1.fElements, im2.fElements, im1.fNelems*sizeof(Double_t)) == 0);
}

//______________________________________________________________________________
TMatrixD &operator+=(TMatrixD &target, const TMatrixD &source)
{
   // Add the source matrix to the target matrix.

   if (!AreCompatible(target, source)) {
      Error("operator+=", "matrices are not compatible");
      return target;
   }

   Double_t *sp = source.fElements;
   Double_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNelems; )
      *tp++ += *sp++;

   return target;
}

//______________________________________________________________________________
TMatrixD &operator-=(TMatrixD &target, const TMatrixD &source)
{
   // Subtract the source matrix from the target matrix.

   if (!AreCompatible(target, source)) {
      Error("operator-=", "matrices are not compatible");
      return target;
   }

   Double_t *sp = source.fElements;
   Double_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNelems; )
      *tp++ -= *sp++;

   return target;
}

//______________________________________________________________________________
TMatrixD &Add(TMatrixD &target, Double_t scalar, const TMatrixD &source)
{
   // Modify addition: target += scalar * source.

   if (!AreCompatible(target, source)) {
      Error("Add", "matrices are not compatible");
      return target;
   }

   Double_t *sp = source.fElements;
   Double_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNelems; )
      *tp++ += scalar * (*sp++);

   return target;
}

//______________________________________________________________________________
TMatrixD &ElementMult(TMatrixD &target, const TMatrixD &source)
{
   // Multiply target by the source, element-by-element.

   if (!AreCompatible(target, source)) {
      Error("ElementMult", "matrices are not compatible");
      return target;
   }

   Double_t *sp = source.fElements;
   Double_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNelems; )
      *tp++ *= *sp++;

   return target;
}

//______________________________________________________________________________
TMatrixD &ElementDiv(TMatrixD &target, const TMatrixD &source)
{
   // Divide target by the source, element-by-element.

   if (!AreCompatible(target, source)) {
      Error("ElementDiv", "matrices are not compatible");
      return target;
   }

   Double_t *sp = source.fElements;
   Double_t *tp = target.fElements;
   for ( ; tp < target.fElements+target.fNelems; )
      *tp++ /= *sp++;

   return target;
}

//______________________________________________________________________________
Double_t TMatrixD::RowNorm() const
{
   // Row matrix norm, MAX{ SUM{ |M(i,j)|, over j}, over i}.
   // The norm is induced by the infinity vector norm.

   if (!IsValid()) {
      Error("RowNorm", "matrix not initialized");
      return 0.0;
   }

   Double_t  *ep = fElements;
   Double_t norm = 0;

   // Scan the matrix row-after-row
   while (ep < fElements+fNrows) {
      Int_t j;
      Double_t sum = 0;
      // Scan a row to compute the sum
      for (j = 0; j < fNcols; j++, ep += fNrows)
         sum += TMath::Abs(*ep);
      ep -= fNelems - 1;         // Point ep to the beginning of the next row
      norm = TMath::Max(norm, sum);
   }

   Assert(ep == fElements + fNrows);

   return norm;
}

//______________________________________________________________________________
Double_t TMatrixD::ColNorm() const
{
   // Column matrix norm, MAX{ SUM{ |M(i,j)|, over i}, over j}.
   // The norm is induced by the 1 vector norm.

   if (!IsValid()) {
      Error("ColNorm", "matrix not initialized");
      return 0.0;
   }

   Double_t  *ep = fElements;
   Double_t norm = 0;

   // Scan the matrix col-after-col (i.e. in the natural order of elems)
   while (ep < fElements+fNelems) {
      Int_t i;
      Double_t sum = 0;
      // Scan a col to compute the sum
      for (i = 0; i < fNrows; i++)
         sum += TMath::Abs(*ep++);
      norm = TMath::Max(norm, sum);
   }

   Assert(ep == fElements + fNelems);

   return norm;
}

//______________________________________________________________________________
Double_t TMatrixD::E2Norm() const
{
   // Square of the Euclidian norm, SUM{ m(i,j)^2 }.

   if (!IsValid()) {
      Error("E2Norm", "matrix not initialized");
      return 0.0;
   }

   Double_t  *ep;
   Double_t sum = 0;

   for (ep = fElements; ep < fElements+fNelems; ep++)
      sum += (*ep) * (*ep);

   return sum;
}

//______________________________________________________________________________
Double_t E2Norm(const TMatrixD &m1, const TMatrixD &m2)
{
   // Square of the Euclidian norm of the difference between two matrices.

   if (!AreCompatible(m1, m2)) {
      Error("E2Norm", "matrices are not compatible");
      return 0.0;
   }

   Double_t  *mp1 = m1.fElements;
   Double_t  *mp2 = m2.fElements;
   Double_t sum = 0;

   for (; mp1 < m1.fElements+m1.fNelems; mp1++, mp2++)
      sum += (*mp1 - *mp2) * (*mp1 - *mp2);

   return sum;
}

//______________________________________________________________________________
void TMatrixD::Print(Option_t *) const
{
   // Print the matrix as a table of elements (zeros are printed as dots).

   if (!IsValid()) {
      Error("Print", "matrix not initialized");
      return;
   }

   printf("\nMatrix %dx%d is as follows", fNrows, fNcols);

   Int_t cols_per_sheet = 5;
   Int_t sheet_counter;

   for (sheet_counter = 1; sheet_counter <= fNcols; sheet_counter += cols_per_sheet) {
      printf("\n\n     |");
      Int_t i, j;
      for (j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= fNcols; j++)
         printf("   %6d  |", j+fColLwb-1);
      printf("\n%s\n",
             "------------------------------------------------------------------");
      for (i = 1; i <= fNrows; i++) {
         printf("%4d |", i+fRowLwb-1);
         for (j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= fNcols; j++)
            printf("%11.4g ", (*this)(i+fRowLwb-1, j+fColLwb-1));
         printf("\n");
      }
   }
   printf("\n");
}

//______________________________________________________________________________
void TMatrixD::Transpose(const TMatrixD &prototype)
{
   // Transpose a matrix.

   if (!prototype.IsValid()) {
      Error("Transpose", "prototype matrix not initialized");
      return;
   }

   Allocate(prototype.fNcols,  prototype.fNrows,
            prototype.fColLwb, prototype.fRowLwb);

   Double_t *rsp    = prototype.fElements;    // Row source pointer
   Double_t *tp     = fElements;

   // (This: target) matrix is traversed in the natural, column-wise way,
   // whilst the source (prototype) matrix is scanned row-by-row
   while (tp < fElements + fNelems) {
      Double_t *sp = rsp++;  // sp = @ms[j,i] for i=0

      // Move tp to the next elem in the col and sp to the next el in the curr row
      while (sp < prototype.fElements + prototype.fNelems)
         *tp++ = *sp, sp += prototype.fNrows;
   }

   Assert(tp == fElements + fNelems &&
          rsp == prototype.fElements + prototype.fNrows);
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Invert(Double_t *determ_ptr)
{
   // The most general (Gauss-Jordan) matrix inverse
   //
   // This method works for any matrix (which of course must be square and
   // non-singular). Use this method only if none of specialized algorithms
   // (for symmetric, tridiagonal, etc) matrices isn't applicable/available.
   // Also, the matrix to invert has to be _well_ conditioned:
   // Gauss-Jordan eliminations (even with pivoting) perform poorly for
   // near-singular matrices (e.g., Hilbert matrices).
   //
   // The method inverts matrix inplace and returns the determinant if
   // determ_ptr was specified as not 0. Determinant will be exactly zero
   // if the matrix turns out to be (numerically) singular. If determ_ptr is
   // 0 and matrix happens to be singular, throw up.
   //
   // The algorithm perform inplace Gauss-Jordan eliminations with
   // full pivoting. It was adapted from my Algol-68 "translation" (ca 1986)
   // of the FORTRAN code described in
   // Johnson, K. Jeffrey, "Numerical methods in chemistry", New York,
   // N.Y.: Dekker, c1980, 503 pp, p.221
   //
   // Note, since it's much more efficient to perform operations on matrix
   // columns rather than matrix rows (due to the layout of elements in the
   // matrix), the present method implements a "transposed" algorithm.

   if (!IsValid()) {
      Error("Invert(Double_t*)", "matrix not initialized");
      return *this;
   }

   if (fNrows != fNcols) {
      Error("Invert(Double_t*)", "matrix to invert must be square");
      return *this;
   }

   Double_t determinant = 1;
   const Double_t singularity_tolerance = 1e-35;

   Int_t symmetric = IsSymmetric();

   // store matrix diagonal
   Double_t *diag = 0;
   if (symmetric) {
      diag = new Double_t[fNrows];
      Int_t idiag;
      for (idiag = 0; idiag < fNrows; idiag++)
        diag[idiag] = fElements[idiag*(1+fNrows)];

     // condition the matrix
     Int_t irow;
     for (irow = 0; irow < fNrows; irow++) {
        Int_t icol;
        for (icol = 0; icol < fNcols; icol++) {
           Double_t val = TMath::Sqrt(TMath::Abs(diag[irow]*diag[icol]));
           if (val != 0.0) fElements[irow*fNcols+icol] /= val;
       }
     }
   }

   // Locations of pivots (indices start with 0)
   struct Pivot_t { int row, col; } *const pivots = new Pivot_t[fNcols];
   Bool_t *const was_pivoted = new Bool_t[fNrows];
   memset(was_pivoted, 0, fNrows*sizeof(Bool_t));
   Pivot_t *pivotp;

   for (pivotp = &pivots[0]; pivotp < &pivots[fNcols]; pivotp++) {
      Int_t prow = 0, pcol = 0;         // Location of a pivot to be
      {                                 // Look through all non-pivoted cols
         Double_t max_value = 0;          // (and rows) for a pivot (max elem)
         for (Int_t j = 0; j < fNcols; j++)
            if (!was_pivoted[j]) {
               Double_t *cp;
               Int_t k;
               Double_t curr_value = 0;
               for (k = 0, cp = fIndex[j]; k < fNrows; k++, cp++)
                  if (!was_pivoted[k] && (curr_value = TMath::Abs(*cp)) > max_value)
                     max_value = curr_value, prow = k, pcol = j;
             }
         if (max_value < singularity_tolerance) {
            // free allocated heap memory before returning
            if (symmetric) delete [] diag;
            delete [] pivots;
            delete [] was_pivoted;
            if (determ_ptr) {
               *determ_ptr = 0;
               return *this;
            } else {
               Error("Invert(Double_t*)", "matrix turns out to be singular: can't invert");
               return *this;
            }
         }
         pivotp->row = prow;
         pivotp->col = pcol;
     }

     // Swap prow-th and pcol-th columns to bring the pivot to the diagonal
     if (prow != pcol) {
        Double_t *cr = fIndex[prow];
        Double_t *cc = fIndex[pcol];
        for (Int_t k = 0; k < fNrows; k++) {
           Double_t temp = *cr; *cr++ = *cc; *cc++ = temp;
        }
     }
     was_pivoted[prow] = kTRUE;

     {                                       // Normalize the pivot column and
        Double_t *pivot_cp = fIndex[prow];
        Double_t pivot_val = pivot_cp[prow]; // pivot is at the diagonal
        determinant *= pivot_val;            // correct the determinant
        pivot_cp[prow] = kTRUE;
        for (Int_t k=0; k < fNrows; k++)
           *pivot_cp++ /= pivot_val;
     }

     {                                           // Perform eliminations
        Double_t *pivot_rp = fElements + prow;     // pivot row
        for (Int_t k = 0; k < fNrows; k++, pivot_rp += fNrows)
           if (k != prow) {
              Double_t temp = *pivot_rp;
              *pivot_rp = 0;
              Double_t *pivot_cp = fIndex[prow];          // pivot column
              Double_t *elim_cp  = fIndex[k];             // elimination column
              for (Int_t l = 0; l < fNrows; l++)
                 *elim_cp++ -= temp * *pivot_cp++;
           }
      }
   }

   Int_t no_swaps = 0;                   // Swap exchanged *rows* back in place
   for (pivotp = &pivots[fNcols-1]; pivotp >= &pivots[0]; pivotp--)
      if (pivotp->row != pivotp->col) {
         no_swaps++;
         Double_t *rp = fElements + pivotp->row;
         Double_t *cp = fElements + pivotp->col;
         for (Int_t k = 0; k < fNcols; k++, rp += fNrows, cp += fNrows) {
            Double_t temp = *rp; *rp = *cp; *cp = temp;
         }
      }

   // revert our scaling
   if (symmetric) {
      Int_t irow;
      for (irow = 0; irow < fNrows; irow++) {
         Int_t icol;
         for (icol = 0; icol < fNcols; icol++) {
            Double_t val = TMath::Sqrt(TMath::Abs(diag[irow]*diag[icol]));
            if (val != 0.0) fElements[irow*fNcols+icol] /= val;
         }
         determinant *= TMath::Abs(diag[irow]);
      }
      delete [] diag;
   }

   if (determ_ptr)
      *determ_ptr = (no_swaps & 1 ? -determinant : determinant);

   delete [] was_pivoted;
   delete [] pivots;

   return *this;
}

//______________________________________________________________________________
Bool_t TMatrixD::IsSymmetric() const
{
  Int_t irow;
  for (irow = 0; irow < fNrows; irow++) {
    Int_t icol;
    for (icol = 0; icol < irow; icol++) {
      if (fElements[irow*fNrows+icol] != fElements[icol*fNrows+irow]) {
        return 0;
      }
    }
  }
  return 1;
}

//______________________________________________________________________________
void TMatrixD::Invert(const TMatrixD &m)
{
   // Allocate new matrix and set it to inv(m).

   if (!m.IsValid()) {
      Error("Invert(const TMatrixD&)", "matrix m not initialized");
      return;
   }

   ResizeTo(m);

   *this = m;    // assignment operator

   Invert(0);
}

//______________________________________________________________________________
TMatrixD &TMatrixD::InvertPosDef()
{
   if (!IsValid()) {
      Error("InvertPosDef(Double_t*)", "matrix not initialized");
      return *this;
   }

   if (fNrows != fNcols) {
      Error("InvertPosDef(Double_t*)", "matrix to invert must be square");
      return *this;
   }

   Int_t n = fNrows;
   Double_t *pa = fElements;
   Double_t *pu = new Double_t[n*n];

   // step 1: Cholesky decomposition
   if (Pdcholesky(pa,pu,n))
     Error("InvertPosDef","matrix not positive definite ?");

   Int_t off_n = (n-1)*n;
   Int_t i,l;
   for (i = 0; i < n; i++)
   {
     Int_t off_i = i*n;

   // step 2: Forward substitution
     for (l = i; l < n; l++)
     {
       if (l == i)
         pa[off_n+l] = 1./pu[l*n+l];
       else
       {
         pa[off_n+l] = 0.;
         for (Int_t k = i; k <= l-1; k++)
           pa[off_n+l] = pa[off_n+l]-pu[k*n+l]*pa[off_n+k];
         pa[off_n+l] = pa[off_n+l]/pu[l*n+l];
       }
     }

   // step 3: Back substitution
     for (l = n-1; l >= i; l--)
     {
       Int_t off_l = l*n;
       if (l == n-1)
         pa[off_i+l] = pa[off_n+l]/pu[off_l+l];
       else
       {
         pa[off_i+l] = pa[off_n+l];
         for (Int_t k = n-1; k >= l+1; k--)
           pa[off_i+l] = pa[off_i+l]-pu[off_l+k]*pa[off_i+k];
         pa[off_i+l] = pa[off_i+l]/pu[off_l+l];
       }
     }
   }

   // Fill lower triangle symmetrically
   if (n > 1)
   {
     for (Int_t i = 0; i < n; i++)
     {
       for (Int_t l = 0; l <= i-1; l++)
         pa[i*n+l] = pa[l*n+i];
     }
   }

   delete [] pu;
   return *this;
}

//______________________________________________________________________________
Int_t TMatrixD::Pdcholesky(
const Double_t *a,
      Double_t *u,
const Int_t     n)
{
  //  Program Pdcholesky inverts a positiv definite (n x n) - matrix A,
  //  using the Cholesky decomposition
  //
  //  Input:	a	- (n x n)- Matrix A
  //  		n	- dimensions n of matrices
  //
  //  Output:	u	- (n x n)- Matrix U so that U^T . U = A
  //		return	- 0 decomposition succesful
  //			- 1 decomposition failed

  memset(u,0,n*n*sizeof(Double_t));

  Int_t status = 0;
  for (Int_t k = 0; k < n; k++)
  {
    Double_t s = 0.;
    Int_t off_k = k*n;
    for (Int_t j = k; j < n; j++)
    {
      if (k > 0)
      {
        s = 0.;
        for (Int_t l = 0; l <= k-1; l++)
        {
          Int_t off_l = l*n;
          s += u[off_l+k]*u[off_l+j];
        }
      }
      u[off_k+j] = a[off_k+j]-s;
      if (k == j)
      {
        if (u[off_k+j] < 0) status = 1;
        u[off_k+j] = TMath::Sqrt(TMath::Abs(u[off_k+j]));
      }
      else
        u[off_k+j] = u[off_k+j]/u[off_k+k];
    }
  }
  return status;
}

//______________________________________________________________________________
void TMatrixD::InvertPosDef(const TMatrixD &m)
{
   // Allocate new matrix and set it to inv(m).

   if (!m.IsValid()) {
      Error("InvertPosDef(const TMatrixD&)", "matrix m not initialized");
      return;
   }

   ResizeTo(m);

   *this = m;    // assignment operator

   InvertPosDef();
}

//____________________________________________________________________
TMatrixD TMatrixD::EigenVectors(
TVectorD &eigenValues)
{
  // Return a matrix containing the eigen-vectors; also fill the
  // supplied vector with the eigen values.

  if (IsSymmetric())
  {
    TMatrixD eigenVectors = *this;
    eigenValues.ResizeTo(fNrows);
    TVectorD offDiag(fNrows);
    // Tridiagonalize matrix
    MakeTridiagonal(eigenVectors,eigenValues,offDiag);

    // Make eigenvectors and -values
    MakeEigenVectors(eigenValues,offDiag,eigenVectors);

    // Order eigenvalues and -vectors
    EigenSort(eigenVectors,eigenValues);
    return eigenVectors;
  }
  else
  {
    Error("EigenVectors","Not yet implemented for non-symmetric matrix");
    return *this;
  }
}

//____________________________________________________________________
void TMatrixD::MakeTridiagonal(
TMatrixD &a,
TVectorD &d,
TVectorD &e)
{
  // The comments in this algorithm are modified version of those in
  // "Numerical ...". Please refer to that book (web-page) for more on
  // the algorithm.
  // Begin_Html
  /*
    </PRE>
    PRIVATE METHOD:
    <BR>
    Tridiagonalise the covariance matrix according to the Householder
    method as described in
    <A NAME="tex2html3"
    HREF="http://www.nr.com">Numerical Recipes in C</A>
    section&nbsp;11.2.

    <P>
    The basic idea is to perform <IMG
    WIDTH="48" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img51.gif"
    ALT="$P-2$"> orthogonal transformation, where
    each transformation eat away the off-diagonal elements, except the
    inner most.
    <PRE>
   */
  // End_Html

  Int_t n = a.fNrows;

  if (!a.IsValid()) {
    Error("Maketridiagonal", "matrix not initialized");
    return;
  }

  if (a.fNrows != a.fNcols) {
    Error("Maketridiagonal", "matrix to tridiagonalize must be square");
    return;
  }

  if (!a.IsSymmetric()) {
    Error("MakeTridiagonal", "Can only tridiagonalise symmetric matrix");
    a.Zero();
    d.Zero();
    e.Zero();
    return;
  }

  Double_t *pa = a.fElements;
  Double_t *pd = d.fElements;
  Double_t *pe = e.fElements;

  Int_t i;
  for (i = n-1; i > 0; i--) {
    Int_t    l     = i-1;
    Double_t h     = 0;
    Double_t scale = 0;

    if (l > 0) {
      for (Int_t k = 0; k <= l; k++)
        scale += TMath::Abs(pa[i+k*n]);

      if (scale == 0)
        // Skip transformation
        pe[i] = pa[i+l*n];

      else {
        Int_t k;
        for (k = 0; k <= l; k++) {
          // Use scaled elements of a for transformation
          pa[i+k*n] /= scale;
          // Calculate sigma in h
          h += pa[i+k*n]*pa[i+k*n];
        }

        Double_t f =  pa[i+l*n];
        Double_t g =  (f >= 0. ? -TMath::Sqrt(h) : TMath::Sqrt(h));
        pe[i]      =  scale*g;
        h         -= f*g; // Now h is eq. (11.2.4) in "Numerical ..."
        pa[i+l*n]  =  f-g;
        f          = 0;

        Int_t j;
        for (j = 0; j <= l; j++) {
          // Store the u/H in ith column of a;
          pa[j+i*n] = pa[i+j*n]/h;
          // Form element A dot u in g;
          g = 0;

          Int_t k;
          for (k = 0; k <= j; k++)
            g += pa[j+k*n]*pa[i+k*n];

          for (k = j+1; k <= l; k++)
            g += pa[k+j*n]*pa[i+k*n];

          // Form element of vector p in temporarily unused element of
          // e
          pe[j] =  g/h;
          f    += pe[j]*pa[i+j*n];
        }
        // Form K eq (11.2.11)
        Double_t hh = f/(h+h);

        // Form vector q and store in e overwriting p
        for (j = 0; j <= l; j++) {
          f    = pa[i+j*n];
          pe[j] = g = pe[j]-hh*f;

          Int_t k;
          for (k = 0; k <= j; k++)
            // Reduce a, eq (11.2.13)
            pa[j+k*n] -= (f*pe[k]+g*pa[i+k*n]);
        }
      }
    }
    else
      pe[i] = pa[i+l*n];

    pd[i] = h;
  }

  pd[0] = 0;
  pe[0] = 0;

  for (i = 0; i < n; i++) {
    // Begin accumulation of transformation matrix
    Int_t l = i-1;

    if (pd[i]) {
      // This block is skipped if i = 0;
      Int_t j;
      for (j = 0; j <= l; j++) {
        Double_t g = 0;

        Int_t k;
        for (k = 0; k <= l; k++)
          // Use vector u/H stored in a to form P dot Q
          g += pa[i+k*n]*pa[k+j*n];

        for (k = 0; k <= l; k++)
          pa[k+j*n] -= g*pa[k+i*n];
      }
    }

    pd[i]     = pa[i+i*n];
    pa[i+i*n] = 1;

    Int_t j;
    for (j = 0; j <= l; j++) {
      pa[j+i*n] = pa[i+j*n] = 0;
    }
  }

}

//____________________________________________________________________
void TMatrixD::MakeEigenVectors(
TVectorD &d,
TVectorD &e,
TMatrixD &z)
{
  // Begin_Html
  /*
    </PRE>
    PRIVATE METHOD:
    <BR>
    Find eigenvalues and vectors of tridiagonalised covariance matrix
    according to the <I>QL with implicit shift</I> algorithm from
    <A NAME="tex2html1"
    HREF="http://www.nr.com">Numerical Recipes in C</A>
    section&nbsp;11.3.
    <P>
    The basic idea is to find matrices <IMG
    WIDTH="17" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img41.gif"
    ALT="$\mathsf{Q}$"> and <IMG
    WIDTH="14" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img42.gif"
    ALT="$\mathsf{L}$"> so that
    <!-- MATH
    $\mathsf{C} = \mathsf{Q} \cdot \mathsf{L}$
    -->
    <IMG
    WIDTH="74" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img43.gif"
    ALT="$\mathsf{C} = \mathsf{Q} \cdot \mathsf{L}$">, where  <IMG
    WIDTH="17" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img41.gif"
    ALT="$\mathsf{Q}$"> is orthogonal and
    <IMG
    WIDTH="14" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img42.gif"
    ALT="$\mathsf{L}$"> is lower triangular. The <I>QL</I> algorithm
    consist of a
    sequence of orthogonal transformations
    <BR><P></P>
    <DIV ALIGN="CENTER">

    <!-- MATH
    \begin{displaymath}
    \mathsf{C}_s = \mathsf{Q}_s \cdot \mathsf{L}_s
    \end{displaymath}
    -->


    <IMG
    WIDTH="89" HEIGHT="29" BORDER="0"
    SRC="gif/principal_img44.gif"
    ALT="\begin{displaymath}
    \mathsf{C}_s = \mathsf{Q}_s \cdot \mathsf{L}_s
    \end{displaymath}">
    </DIV>
    <BR CLEAR="ALL">
    <P></P>
    <BR><P></P>
    <DIV ALIGN="CENTER">

    <!-- MATH
    \begin{displaymath}
    \mathsf{C}_{s+1} = \mathsf{L}_s   \cdot \mathsf{Q}_s
    = \mathsf{Q}_s^T \cdot \mathsf{C}_s \cdot \mathsf{Q}_s
    \end{displaymath}
    -->


    <IMG
    WIDTH="215" HEIGHT="31" BORDER="0"
    SRC="gif/principal_img45.gif"
    ALT="\begin{displaymath}
    \mathsf{C}_{s+1} = \mathsf{L}_s \cdot \mathsf{Q}_s
    = \mathsf{Q}_s^T \cdot \mathsf{C}_s \cdot \mathsf{Q}_s
    \end{displaymath}">
    </DIV>
    <BR CLEAR="ALL">
    <P></P>
    (1) If <IMG
    WIDTH="16" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img2.gif"
    ALT="$\mathsf{C}$"> have eigenvalues with different absolute value
    <IMG
    WIDTH="25" HEIGHT="34" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img46.gif"
    ALT="$\vert l_i\vert$">,  then
    <!-- MATH
    $\mathsf{C}_s \rightarrow$
    -->
    <IMG
    WIDTH="45" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img47.gif"
    ALT="$\mathsf{C}_s \rightarrow$">&nbsp;[lower triangular form] as

    <!-- MATH
    $s\rightarrow\infty$
    -->
    <IMG
    WIDTH="57" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img48.gif"
    ALT="$s\rightarrow\infty$">. The eigenvalues appear on the diagonal in
    increasing order of absolute magnitude. (2) If If <IMG
    WIDTH="16" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img2.gif"
    ALT="$\mathsf{C}$"> has an
    eigenvalue <IMG
    WIDTH="25" HEIGHT="34" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img46.gif"
    ALT="$\vert l_i\vert$"> of multiplicty of order <IMG
    WIDTH="13" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img49.gif"
    ALT="$p$">,

    <!-- MATH
    $\mathsf{C}_s \rightarrow$
    -->
    <IMG
    WIDTH="45" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img47.gif"
    ALT="$\mathsf{C}_s \rightarrow$">&nbsp;[lower triangular form] as

    <!-- MATH
    $s\rightarrow\infty$
    -->
    <IMG
    WIDTH="57" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img48.gif"
    ALT="$s\rightarrow\infty$">, except for a diagona block matrix of order <IMG
    WIDTH="13" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img49.gif"
    ALT="$p$">,
    whose eigenvalues
    <!-- MATH
    $\rightarrow l_i$
    -->
    <IMG
    WIDTH="37" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img50.gif"
    ALT="$\rightarrow l_i$">.
    <PRE>
  */
  // End_Html

  Int_t n = z.fNrows;

  Double_t *pd = d.fElements;
  Double_t *pe = e.fElements;
  Double_t *pz = z.fElements;

  // It's convenient to renumber the e vector elements
  Int_t l;
  for (l = 1; l < n; l++)
    pe[l-1] = pe[l];
  pe[n-1] = 0;

  for (l = 0; l < n; l++) {
    Int_t iter = 0;
    Int_t m    = 0;

    do {
      for (m = l; m < n-1; m++) {
        // Look for a single small sub-diagonal element  to split the
        // matrix
        Double_t dd = TMath::Abs(pd[m])+TMath::Abs(pd[m+1]);
        if ((Double_t)(TMath::Abs(pe[m])+dd) == dd)
          break;
      }

      if (m != l) {
        if (iter++ == 30) {
          Error("MakeEigenVectors","too many iterations\n");
          return;
        }

        // Form shift
        Double_t g = (pd[l+1]-pd[l])/(2*pe[l]);
        Double_t r = TMath::Sqrt(g*g+1);
        // This is d_m-k_s
        g          = pd[m]-pd[l]+pe[l]/(g+TMath::Sign(r,g));
        Double_t s = 1;
        Double_t c = 1;
        Double_t p = 0;
        Int_t i    = 0;
        for (i = m-1; i >= l; i--) {
          // A plane rotation as in the original QL, followed by
          // Givens rotations to restore tridiagonal form
          Double_t f = s*pe[i];
          Double_t b = c*pe[i];
          r          = TMath::Sqrt(f*f+g*g);
          pe[i+1]    = r;

          if (r == 0) {
            // Recover from underflow
            pd[i+1] -= p;
            pe[m]   =  0;
            break;
          }
          s       = f/r;
          c       = g/r;
          g       = pd[i+1]-p;
          r       = (pd[i]-g)*s+2*c*b;
          p       = s*r;
          pd[i+1] = g+p;
          g       = c*r-b;

          Int_t k;
          for (k = 0; k < n; k++) {
            // Form Eigenvectors
            f             = pz[k+(i+1)*n];
            pz[k+(i+1)*n] = s*pz[k+i*n]+c*f;
            pz[k+i*n]     = c*pz[k+i*n]-s*f;
          }
        }  // for (i = m)

        if (r == 0 && i >= l)
          continue;

        pd[l] -= p;
        pe[l]  = g;
        pe[m]  = 0;

      } // if (m != l)
    } while (m != l);
  } // for (l = 0)
}

//____________________________________________________________________
void TMatrixD::EigenSort(
TMatrixD &eigenVectors,
TVectorD &eigenValues)
{
  // Begin_Html
  /*
    </PRE>
    PRIVATE METHOD:
    <BR>
    Order the eigenvalues and vectors by ascending eigenvalue. The
    algorithm is a straight insertion. It's taken from
    <A NAME="tex2html2"
    HREF="http://www.nr.com">Numerical Recipes in C</A>
    section 11.1.
    <PRE>
  */
  // End_Html

  Int_t n = eigenVectors.fNrows;

  Double_t *pVec = eigenVectors.fElements;
  Double_t *pVal = eigenValues.fElements;

  Int_t i;
  for (i = 0; i < n; i++) {
    Int_t k = i;
    Double_t p = pVal[i];

    Int_t j;
    for (j = i + 1; j < n; j++)
      if (pVal[j] >= p) {
        k = j;
        p = pVal[j];
      }

    if (k != i) {
      pVal[k] = pVal[i];
      pVal[i] = p;

      for (j = 0; j < n; j++) {
        p           = pVec[j+i*n];
        pVec[j+i*n] = pVec[j+k*n];
        pVec[j+k*n] = p;
      }
    }
  }
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(const TMatrixD &source)
{
   // Compute target = target * source inplace. Strictly speaking, it can't be
   // done inplace, though only the row of the target matrix needs
   // to be saved. "Inplace" multiplication is only possible
   // when the 'source' matrix is square.

   if (!IsValid()) {
      Error("operator*=(const TMatrixD&)", "matrix not initialized");
      return *this;
   }

   if (!source.IsValid()) {
      Error("operator*=(const TMatrixD&)", "source matrix not initialized");
      return *this;
   }

   if (fRowLwb != source.fColLwb || fNcols != source.fNrows ||
       fColLwb != source.fColLwb || fNcols != source.fNcols)
      Error("operator*=(const TMatrixD&)",
            "matrices above are unsuitable for the inplace multiplication");

   // One row of the old_target matrix
   Double_t *const one_row = new Double_t[fNcols];
   const Double_t *one_row_end = &one_row[fNcols];

   Double_t *trp = fElements;                     // Pointer to the i-th row
   for ( ; trp < &fElements[fNrows]; trp++) {   // Go row-by-row in the target
      Double_t *wrp, *orp;                        // work row pointers
      for (wrp = trp, orp = one_row; orp < one_row_end; )
         *orp++ = *wrp, wrp += fNrows;          // Copy a row of old_target

      Double_t *scp = source.fElements;           // Source column pointer
      for (wrp = trp; wrp < fElements+fNelems; wrp += fNrows) {
         Double_t sum = 0;                      // Multiply a row of old_target
         for (orp = one_row; orp < one_row_end; ) // by each col of source
            sum += *orp++ * *scp++;             // to get a row of new_target
         *wrp = sum;
      }
   }
   delete [] one_row;

   return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(const TMatrixDDiag &diag)
{
   // Multiply a matrix by the diagonal of another matrix
   // matrix(i,j) *= diag(j)

   if (!IsValid()) {
      Error("operator*=(const TMatrixDDiag&)", "matrix not initialized");
      return *this;
   }

   if (!diag.fMatrix->IsValid()) {
      Error("operator*=(const TMatrixDDiag&)", "diag matrix not initialized");
      return *this;
   }

   if (fNcols != diag.fNdiag) {
      Error("operator*=(const TMatrixDDiag&)", "matrix cannot be multiplied by the diagonal of the other matrix");
      return *this;
   }

   Double_t *dp = diag.fPtr;                // Diag ptr
   Double_t *mp = fElements;                // Matrix ptr
   Int_t i;
   for ( ; mp < fElements + fNelems; dp += diag.fInc)
      for (i = 0; i < fNrows; i++)
         *mp++ *= *dp;

   Assert(dp < diag.fPtr + diag.fMatrix->fNelems + diag.fInc);

   return *this;
}

//______________________________________________________________________________
void TMatrixD::AMultB(const TMatrixD &a, const TMatrixD &b)
{
   // General matrix multiplication. Create a matrix C such that C = A * B.
   // Note, matrix C needs to be allocated.

   if (!a.IsValid()) {
      Error("AMultB", "matrix a not initialized");
      return;
   }
   if (!b.IsValid()) {
      Error("AMultB", "matrix b not initialized");
      return;
   }

   if (a.fNcols != b.fNrows || a.fColLwb != b.fRowLwb) {
      Error("AMultB", "matrices a and b cannot be multiplied");
      return;
   }

   Allocate(a.fNrows, b.fNcols, a.fRowLwb, b.fColLwb);

   Double_t *arp;                         // Pointer to the i-th row of A
   Double_t *bcp = b.fElements;           // Pointer to the j-th col of B
   Double_t *cp  = fElements;             // C is to be traversed in the natural
   while (cp < fElements + fNelems) {   // order, col-after-col
      for (arp = a.fElements; arp < a.fElements + a.fNrows; ) {
         Double_t cij = 0;
         Double_t *bccp = bcp;            // To scan the jth col of B
         while (arp < a.fElements + a.fNelems)       // Scan the i-th row of A and
            cij += *bccp++ * *arp, arp += a.fNrows;  // the j-th col of B
         *cp++ = cij;
         arp -= a.fNelems - 1;          // arp points to (i+1)-th row
      }
      bcp += b.fNrows;                  // We're done with j-th col of both
   }                                    // B and C. Set bcp to the (j+1)-th col

   Assert(cp == fElements + fNelems && bcp == b.fElements + b.fNelems);
}

//______________________________________________________________________________
void TMatrixD::Mult(const TMatrixD &a, const TMatrixD &b)
{
   // Compute C = A*B. The same as AMultB(), only matrix C is already
   // allocated, and it is *this.

   if (!a.IsValid()) {
      Error("Mult", "matrix a not initialized");
      return;
   }
   if (!b.IsValid()) {
      Error("Mult", "matrix b not initialized");
      return;
   }
   if (!IsValid()) {
      Error("Mult", "matrix not initialized");
      return;
   }

   if (a.fNcols != b.fNrows || a.fColLwb != b.fRowLwb) {
      Error("Mult", "matrices a and b cannot be multiplied");
      return;
   }

   if (fNrows != a.fNrows || fNcols != b.fNcols ||
       fRowLwb != a.fRowLwb || fColLwb != b.fColLwb) {
      Error("Mult", "product A*B is incompatible with the given matrix");
      return;
   }

   Double_t *arp;                         // Pointer to the i-th row of A
   Double_t *bcp = b.fElements;           // Pointer to the j-th col of B
   Double_t *cp = fElements;              // C is to be traversed in the natural
   while (cp < fElements + fNelems) {   // order, col-after-col
      for (arp = a.fElements; arp < a.fElements + a.fNrows; ) {
         Double_t cij = 0;
         Double_t *bccp = bcp;            // To scan the jth col of B
         while (arp < a.fElements + a.fNelems)       // Scan the i-th row of A and
            cij += *bccp++ * *arp, arp += a.fNrows;  // the j-th col of B
         *cp++ = cij;
         arp -= a.fNelems - 1;          // arp points to (i+1)-th row
      }
      bcp += b.fNrows;                  // We're done with j-th col of both
   }                                    // B and C. Set bcp to the (j+1)-th col

   Assert(cp == fElements + fNelems && bcp == b.fElements + b.fNelems);
}

//______________________________________________________________________________
void TMatrixD::AtMultB(const TMatrixD &a, const TMatrixD &b)
{
   // Create a matrix C such that C = A' * B. In other words,
   // c[i,j] = SUM{ a[k,i] * b[k,j] }. Note, matrix C needs to be allocated.

   if (!a.IsValid()) {
      Error("AtMultB", "matrix a not initialized");
      return;
   }
   if (!b.IsValid()) {
      Error("AtMultB", "matrix b not initialized");
      return;
   }

   if (a.fNrows != b.fNrows || a.fRowLwb != b.fRowLwb) {
      Error("AtMultB", "matrices above are unsuitable for A'B multiplication");
      return;
   }

   Allocate(a.fNcols, b.fNcols, a.fColLwb, b.fColLwb);

   Double_t *acp;                         // Pointer to the i-th col of A
   Double_t *bcp = b.fElements;           // Pointer to the j-th col of B
   Double_t *cp = fElements;              // C is to be traversed in the natural
   while (cp < fElements + fNelems) {   // order, col-after-col
      for (acp = a.fElements; acp < a.fElements + a.fNelems; ) {
         Double_t cij = 0;                      // Scan all cols of A
         Double_t *bccp = bcp;                    // To scan the jth col of B
         for (int i = 0; i < a.fNrows; i++)     // Scan the i-th row of A and
            cij += *bccp++ * *acp++;            // the j-th col of B
         *cp++ = cij;
      }
      bcp += b.fNrows;                  // We're done with j-th col of both
   }                                    // B and C. Set bcp to the (j+1)-th col

   Assert(cp == fElements + fNelems && bcp == b.fElements + b.fNelems);
}

//______________________________________________________________________________
Double_t TMatrixD::Determinant() const
{
   // Compute the determinant of a general square matrix.
   // Example: Matrix A; Double_t A.Determinant();
   //
   // Gauss-Jordan transformations of the matrix with a slight
   // modification to take advantage of the *column*-wise arrangement
   // of Matrix elements. Thus we eliminate matrix's columns rather than
   // rows in the Gauss-Jordan transformations. Note that determinant
   // is invariant to matrix transpositions.
   // The matrix is copied to a special object of type TMatrixDPivoting,
   // where all Gauss-Jordan eliminations with full pivoting are to
   // take place.

   if (!IsValid()) {
      Error("Determinant", "matrix not initialized");
      return 0.0;
   }

   if (fNrows != fNcols) {
      Error("Determinant", "can't obtain determinant of a non-square matrix");
      return 0.0;
   }

   if (fRowLwb != fColLwb) {
      Error("Determinant", "row and col lower bounds are inconsistent");
      return 0.0;
   }

   TMatrixDPivoting mp(*this);

   Double_t det = 1;
   Int_t    k;

   for (k = 0; k < fNcols && det != 0; k++)
      det *= mp.PivotingAndElimination();

   return det;
}

//______________________________________________________________________________
void TMatrixD::Streamer(TBuffer &R__b)
{
   // Stream an object of class TMatrixD.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TMatrixD::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         if (fNcols == 1) {
            fIndex = &fElements;
         } else {
            fIndex = new Double_t*[fNcols];
            if (fIndex)
               memset(fIndex, 0, fNcols*sizeof(Double_t*));
            Int_t i;
            Double_t *col_p;
            for (i = 0, col_p = &fElements[0]; i < fNcols; i++, col_p += fNrows)
               fIndex[i] = col_p;
         }
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      R__b >> fNrows;
      R__b >> fNcols;
      R__b >> fRowLwb;
      R__b >> fColLwb;
      fNelems = R__b.ReadArray(fElements);
      if (fNcols == 1) {
         fIndex = &fElements;
      } else {
         fIndex = new Double_t*[fNcols];
         if (fIndex)
            memset(fIndex, 0, fNcols*sizeof(Double_t*));
         Int_t i;
         Double_t *col_p;
         for (i = 0, col_p = &fElements[0]; i < fNcols; i++, col_p += fNrows)
            fIndex[i] = col_p;
      }
      R__b.CheckByteCount(R__s, R__c, TMatrixD::IsA());
      //====end of old versions

   } else {
      TMatrixD::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void Compare(const TMatrixD &matrix1, const TMatrixD &matrix2)
{
   // Compare two matrices and print out the result of the comparison.

   Int_t i, j;

   if (!AreCompatible(matrix1, matrix2)) {
      Error("Compare", "matrices are not compatible");
      return;
   }

   printf("\n\nComparison of two TMatrices:\n");

   Double_t norm1 = 0, norm2 = 0;       // Norm of the Matrices
   Double_t ndiff = 0;                  // Norm of the difference
   Int_t    imax = 0, jmax = 0;         // For the elements that differ most
   Double_t   difmax = -1;
   Double_t  *mp1 = matrix1.fElements;    // Matrix element pointers
   Double_t  *mp2 = matrix2.fElements;

   for (j = 0; j < matrix1.fNcols; j++)      // Due to the column-wise arrangement,
      for (i = 0; i < matrix1.fNrows; i++) { // the row index changes first
         Double_t mv1 = *mp1++;
         Double_t mv2 = *mp2++;
         Double_t diff = TMath::Abs(mv1-mv2);

         if (diff > difmax) {
            difmax = diff;
            imax = i;
            jmax = j;
         }
         norm1 += TMath::Abs(mv1);
         norm2 += TMath::Abs(mv2);
         ndiff += TMath::Abs(diff);
      }

   imax += matrix1.fRowLwb, jmax += matrix1.fColLwb;
   printf("\nMaximal discrepancy    \t\t%g", difmax);
   printf("\n   occured at the point\t\t(%d,%d)", imax, jmax);
   const Double_t mv1 = matrix1(imax,jmax);
   const Double_t mv2 = matrix2(imax,jmax);
   printf("\n Matrix 1 element is    \t\t%g", mv1);
   printf("\n Matrix 2 element is    \t\t%g", mv2);
   printf("\n Absolute error v2[i]-v1[i]\t\t%g", mv2-mv1);
   printf("\n Relative error\t\t\t\t%g\n",
          (mv2-mv1)/TMath::Max(TMath::Abs(mv2+mv1)/2,(Double_t)1e-7));

   printf("\n||Matrix 1||   \t\t\t%g", norm1);
   printf("\n||Matrix 2||   \t\t\t%g", norm2);
   printf("\n||Matrix1-Matrix2||\t\t\t\t%g", ndiff);
   printf("\n||Matrix1-Matrix2||/sqrt(||Matrix1|| ||Matrix2||)\t%g\n\n",
          ndiff/TMath::Max(TMath::Sqrt(norm1*norm2), 1e-7));
}

//______________________________________________________________________________
void VerifyElementValue(const TMatrixD &m, Double_t val)
{
   // Validate that all elements of matrix have value val (within 1.e-5).

   Int_t    imax = 0, jmax = 0;
   Double_t max_dev = 0;
   Int_t    i, j;

   for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++)
      for (j = m.GetColLwb(); j <= m.GetColUpb(); j++) {
         Double_t dev = TMath::Abs(m(i,j)-val);
         if (dev > max_dev)
            imax = i, jmax = j, max_dev = dev;
      }

   if (max_dev == 0)
      return;
   else if(max_dev < 1e-5)
      printf("Element (%d,%d) with value %g differs the most from what\n"
             "was expected, %g, though the deviation %g is small\n",
             imax,jmax,m(imax,jmax),val,max_dev);
   else
      Error("VerifyElementValue", "a significant difference from the expected value %g\n"
            "encountered for element (%d,%d) with value %g",
            val,imax,jmax,m(imax,jmax));
}

//______________________________________________________________________________
void VerifyMatrixIdentity(const TMatrixD &m1, const TMatrixD &m2)
{
   // Verify that elements of the two matrices are equal (within 1.e-5).

   Int_t    imax = 0, jmax = 0;
   Double_t max_dev = 0;
   Int_t    i, j;

   if (!AreCompatible(m1, m2)) {
      Error("VerifyMatrixIdentity", "matrices are not compatible");
      return;
   }

   for (i = m1.GetRowLwb(); i <= m1.GetRowUpb(); i++)
      for (j = m1.GetColLwb(); j <= m1.GetColUpb(); j++) {
         Double_t dev = TMath::Abs(m1(i,j)-m2(i,j));
         if (dev > max_dev)
            imax = i, jmax = j, max_dev = dev;
      }

   if (max_dev == 0)
      return;
   if (max_dev < 1e-5)
      printf("Two (%d,%d) elements of matrices with values %g and %g\n"
             "differ the most, though the deviation %g is small\n",
             imax,jmax,m1(imax,jmax),m2(imax,jmax),max_dev);
   else
      Error("VerifyMatrixIdentity", "a significant difference between the matrices encountered\n"
            "at (%d,%d) element, with values %g and %g",
            imax,jmax,m1(imax,jmax),m2(imax,jmax));
}


#if defined(R__HPUX) || defined(R__MACOSX)

//______________________________________________________________________________
//  These functions should be inline
//______________________________________________________________________________

TMatrixD::TMatrixD(Int_t no_rows, Int_t no_cols)
{
   Allocate(no_rows, no_cols);
}

TMatrixD::TMatrixD(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb)
{
   Allocate(row_upb-row_lwb+1, col_upb-col_lwb+1, row_lwb, col_lwb);
}

Bool_t TMatrixD::IsValid() const
{
   if (fNrows == -1)
      return kFALSE;
   return kTRUE;
}

void TMatrixD::SetElements(const Double_t *elements, Option_t *option)
{
  if (!IsValid()) {
    Error("SetElements", "matrix is not initialized");
    return;
  }

  TString opt = option;
  opt.ToUpper();

  if (opt.Contains("F"))
    memcpy(fElements,elements,fNelems*sizeof(Double_t));
  else
  {
    for (Int_t irow = 0; irow < fNrows; irow++)
    {
      for (Int_t icol = 0; icol < fNcols; icol++)
        fElements[irow+icol*fNrows] = elements[irow*fNcols+icol];
    }
  }
}

TMatrixD::TMatrixD(Int_t no_rows, Int_t no_cols,
                          const Double_t *elements, Option_t *option)
{
  // option="F": array elements contains the matrix stored column-wise
  //             like in Fortran, so a[i,j] = elements[i+no_rows*j],
  // else        it is supposed that array elements are stored row-wise
  //             a[i,j] = elements[i*no_cols+j]

  Allocate(no_rows, no_cols);
  SetElements(elements,option);
}

TMatrixD::TMatrixD(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb,
                          const Double_t *elements, Option_t *option)
{
  Allocate(row_upb-row_lwb+1, col_upb-col_lwb+1, row_lwb, col_lwb);
  SetElements(elements,option);
}

void TMatrixD::GetElements(Double_t *elements, Option_t *option) const
{
  if (!IsValid()) {
    Error("GetElements", "matrix is not initialized");
    return;
  }

  TString opt = option;
  opt.ToUpper();

  if (opt.Contains("F"))
    memcpy(elements,fElements,fNelems*sizeof(Double_t));
  else
  {
    for (Int_t irow = 0; irow < fNrows; irow++)
    {
      for (Int_t icol = 0; icol < fNcols; icol++)
        elements[irow+icol*fNrows] = fElements[irow*fNcols+icol];
    }
  }
}

TMatrixD::TMatrixD(const TLazyMatrixD &lazy_constructor)
{
   Allocate(lazy_constructor.fRowUpb-lazy_constructor.fRowLwb+1,
            lazy_constructor.fColUpb-lazy_constructor.fColLwb+1,
            lazy_constructor.fRowLwb, lazy_constructor.fColLwb);
  lazy_constructor.FillIn(*this);
}

TMatrixD &TMatrixD::operator=(const TLazyMatrixD &lazy_constructor)
{
   if (!IsValid()) {
      Error("operator=(const TLazyMatrixD&)", "matrix is not initialized");
      return *this;
   }
   if (lazy_constructor.fRowUpb != GetRowUpb() ||
       lazy_constructor.fColUpb != GetColUpb() ||
       lazy_constructor.fRowLwb != GetRowLwb() ||
       lazy_constructor.fColLwb != GetColLwb()) {
      Error("operator=(const TLazyMatrixD&)", "matrix is incompatible with "
            "the assigned Lazy matrix");
      return *this;
   }

   lazy_constructor.FillIn(*this);
   return *this;
}

Bool_t AreCompatible(const TMatrixD &im1, const TMatrixD &im2)
{
   if (!im1.IsValid()) {
      Error("AreCompatible", "matrix 1 not initialized");
      return kFALSE;
   }
   if (!im2.IsValid()) {
      Error("AreCompatible", "matrix 2 not initialized");
      return kFALSE;
   }

   if (im1.fNrows  != im2.fNrows  || im1.fNcols  != im2.fNcols ||
       im1.fRowLwb != im2.fRowLwb || im1.fColLwb != im2.fColLwb)
      return kFALSE;

   return kTRUE;
}

TMatrixD &TMatrixD::operator=(const TMatrixD &source)
{
   if (this != &source && AreCompatible(*this, source)) {
      TObject::operator=(source);
      memcpy(fElements, source.fElements, fNelems*sizeof(Double_t));
   }
   return *this;
}

TMatrixD::TMatrixD(const TMatrixD &another) : TObject(another)
{
   if (another.IsValid()) {
      Allocate(another.fNrows, another.fNcols, another.fRowLwb, another.fColLwb);
      *this = another;
   } else
      Error("TMatrixD(const TMatrixD&)", "other matrix is not valid");
}

void TMatrixD::ResizeTo(const TMatrixD &m)
{
   ResizeTo(m.GetRowLwb(), m.GetRowUpb(), m.GetColLwb(), m.GetColUpb());
}

const Double_t &TMatrixD::operator()(int rown, int coln) const
{
   static Double_t err;
   err = 0.0;

   if (!IsValid()) {
      Error("operator()", "matrix is not initialized");
      return err;
   }

   Int_t arown = rown - fRowLwb;          // Effective indices
   Int_t acoln = coln - fColLwb;

   if (arown >= fNrows || arown < 0) {
      Error("operator()", "row index %d is out of matrix boundaries [%d,%d]",
            rown, fRowLwb, fNrows+fRowLwb-1);
      return err;
   }
   if (acoln >= fNcols || acoln < 0) {
      Error("operator()", "col index %d is out of matrix boundaries [%d,%d]",
            coln, fColLwb, fNcols+fColLwb-1);
      return err;
   }

   return (fIndex[acoln])[arown];
}

Double_t &TMatrixD::operator()(Int_t rown, Int_t coln)
{
   return (Double_t&)((*(const TMatrixD *)this)(rown,coln));
}

TMatrixD &TMatrixD::Zero()
{
   if (!IsValid())
      Error("Zero", "matrix not initialized");
   else
      memset(fElements, 0, fNelems*sizeof(Double_t));
   return *this;
}

TMatrixD &TMatrixD::Apply(TElementActionD &action)
{
   if (!IsValid())
      Error("Apply(TElementActionD&)", "matrix not initialized");
   else
      for (Double_t *ep = fElements; ep < fElements+fNelems; ep++)
         action.Operation(*ep);
   return *this;
}

#endif
