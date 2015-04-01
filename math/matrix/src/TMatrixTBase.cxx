// @(#)root/matrix:$Id: 2d00df45ce4c38c7ea0930d6b520cbf4cfb9152e $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

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
// ----------------------                                               //
//                                                                      //
// The present package implements all the basic algorithms dealing      //
// with vectors, matrices, matrix columns, rows, diagonals, etc.        //
// In addition eigen-Vector analysis and several matrix decomposition   //
// have been added (LU,QRH,Cholesky,Bunch-Kaufman and SVD) .            //
// The decompositions are used in matrix inversion, equation solving.   //
//                                                                      //
// For a dense matrix, elements are arranged in memory in a ROW-wise    //
// fashion . For (n x m) matrices where n*m <=kSizeMax (=25 currently)  //
// storage space is available on the stack, thus avoiding expensive     //
// allocation/deallocation of heap space . However, this introduces of  //
// course kSizeMax overhead for each matrix object . If this is an      //
// issue recompile with a new appropriate value (>=0) for kSizeMax      //
//                                                                      //
// Sparse matrices are also stored in row-wise fashion but additional   //
// row/column information is stored, see TMatrixTSparse source for      //
// additional details .                                                 //
//                                                                      //
// Another way to assign and store matrix data is through Use           //
// see for instance stressLinear.cxx file .                             //
//                                                                      //
// Unless otherwise specified, matrix and vector indices always start   //
// with 0, spanning up to the specified limit-1. However, there are     //
// constructors to which one can specify aribtrary lower and upper      //
// bounds, e.g. TMatrixD m(1,10,1,5) defines a matrix that ranges       //
// from 1..10, 1..5 (a(1,1)..a(10,5)).                                  //
//                                                                      //
// The present package provides all facilities to completely AVOID      //
// returning matrices. Use "TMatrixD A(TMatrixD::kTransposed,B);"       //
// and other fancy constructors as much as possible. If one really needs//
// to return a matrix, return a TMatrixTLazy object instead. The        //
// conversion is completely transparent to the end user, e.g.           //
// "TMatrixT m = THaarMatrixT(5);" and _is_ efficient.                  //
//                                                                      //
// Since TMatrixT et al. are fully integrated in ROOT, they of course   //
// can be stored in a ROOT database.                                    //
//                                                                      //
// For usage examples see $ROOTSYS/test/stressLinear.cxx                //
//                                                                      //
// Acknowledgements                                                     //
// ----------------                                                     //
// 1. Oleg E. Kiselyov                                                  //
//  First implementations were based on the his code . We have diverged //
//  quite a bit since then but the ideas/code for lazy matrix and       //
//  "nested function" are 100% his .                                    //
//  You can see him and his code in action at http://okmij.org/ftp      //
// 2. Chris R. Birchenhall,                                             //
//  We adapted his idea of the implementation for the decomposition     //
//  classes instead of our messy installation of matrix inversion       //
//  His installation of matrix condition number, using an iterative     //
//  scheme using the Hage algorithm is worth looking at !               //
//  Chris has a nice writeup (matdoc.ps) on his matrix classes at       //
//   ftp://ftp.mcc.ac.uk/pub/matclass/                                  //
// 3. Mark Fischler and Steven Haywood of CLHEP                         //
//  They did the slave labor of spelling out all sub-determinants       //
//   for Cramer inversion  of (4x4),(5x5) and (6x6) matrices            //
//  The stack storage for small matrices was also taken from them       //
// 4. Roldan Pozo of TNT (http://math.nist.gov/tnt/)                    //
//  He converted the EISPACK routines for the eigen-vector analysis to  //
//  C++ . We started with his implementation                            //
// 5. Siegmund Brandt (http://siux00.physik.uni-siegen.de/~brandt/datan //
//  We adapted his (very-well) documented SVD routines                  //
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
//    the constructor. Note, TMatrixDLazy (see below) can construct     //
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
//    TMatrixD haar = THaarMatrixD(5);                                  //
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
//       TMatrixD m1(TMatrixD::kZero,m);                                //
//       struct MakeHilbert : public TElementPosActionD {               //
//          void Operation(Double_t &element)                           //
//             { element = 1./(fI+fJ-1); }                              //
//       };                                                             //
//       m1.Apply(MakeHilbert());                                       //
//    }                                                                 //
//                                                                      //
//    of course, using a special method THilbertMatrixD() is            //
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
//    void foo(TMatrixD &m,TMatrixD &m1)                                //
//    {                                                                 //
//       typedef  double (*dfunc_t)(double);                            //
//       class ApplyFunction : public TElementActionD {                 //
//          dfunc_t fFunc;                                              //
//          void Operation(Double_t &element)                           //
//               { element=fFunc(element); }                            //
//        public:                                                       //
//          ApplyFunction(dfunc_t func):fFunc(func) {}                  //
//       };                                                             //
//       ApplyFunction x(TMath::Sin);                                   //
//       m.Apply(x);                                                    //
//    }                                                                 //
//                                                                      //
//    Validation code $ROOTSYS/test/vmatrix.cxx and vvector.cxx contain //
//    a few more examples of that kind.                                 //
//                                                                      //
// 6. Lazy matrices: instead of returning an object return a "recipe"   //
//    how to make it. The full matrix would be rolled out only when     //
//    and where it's needed:                                            //
//       TMatrixD haar = THaarMatrixD(5);                               //
//    THaarMatrixD() is a *class*, not a simple function. However       //
//    similar this looks to a returning of an object (see note #1       //
//    above), it's dramatically different. THaarMatrixD() constructs a  //
//    TMatrixDLazy, an object of just a few bytes long. A special       //
//    "TMatrixD(const TMatrixDLazy &recipe)" constructor follows the    //
//    recipe and makes the matrix haar() right in place. No matrix      //
//    element is moved whatsoever!                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTBase                                                         //
//                                                                      //
// Template of base class in the linear algebra package                 //
//                                                                      //
//  matrix properties are stored here, however the data storage is part //
//  of the derived classes                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixTBase.h"
#include "TVectorT.h"
#include "TROOT.h"
#include "TClass.h"
#include "TMath.h"
#include <limits.h>

Int_t gMatrixCheck = 1;

templateClassImp(TMatrixTBase)


//______________________________________________________________________________
template<class Element>
void TMatrixTBase<Element>::DoubleLexSort(Int_t n,Int_t *first,Int_t *second,Element *data)
{
// Lexical sort on array data using indices first and second

   const int incs[] = {1,5,19,41,109,209,505,929,2161,3905,8929,16001,INT_MAX};

   Int_t kinc = 0;
   while (incs[kinc] <= n/2)
      kinc++;
   kinc -= 1;

   // incs[kinc] is the greatest value in the sequence that is also <= n/2.
   // If n == {0,1}, kinc == -1 and so no sort will take place.

   for( ; kinc >= 0; kinc--) {
      const Int_t inc = incs[kinc];

      for (Int_t k = inc; k < n; k++) {
         const Element tmp = data[k];
         const Int_t fi = first [k];
         const Int_t se = second[k];
         Int_t j;
         for (j = k; j >= inc; j -= inc) {
            if ( fi < first[j-inc] || (fi == first[j-inc] && se < second[j-inc]) ) {
               data  [j] = data  [j-inc];
               first [j] = first [j-inc];
               second[j] = second[j-inc];
            } else
               break;
         }
         data  [j] = tmp;
         first [j] = fi;
         second[j] = se;
      }
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTBase<Element>::IndexedLexSort(Int_t n,Int_t *first,Int_t swapFirst,
                                           Int_t *second,Int_t swapSecond,Int_t *index)
{
// Lexical sort on array data using indices first and second

   const int incs[] = {1,5,19,41,109,209,505,929,2161,3905,8929,16001,INT_MAX};

   Int_t kinc = 0;
   while (incs[kinc] <= n/2)
      kinc++;
   kinc -= 1;

   // incs[kinc] is the greatest value in the sequence that is also less
   // than n/2.

   for( ; kinc >= 0; kinc--) {
      const Int_t inc = incs[kinc];

      if ( !swapFirst && !swapSecond ) {
         for (Int_t k = inc; k < n; k++) {
            // loop over all subarrays defined by the current increment
            const Int_t ktemp = index[k];
            const Int_t fi = first [ktemp];
            const Int_t se = second[ktemp];
            // Insert element k into the sorted subarray
            Int_t j;
            for (j = k; j >= inc; j -= inc) {
               // Loop over the elements in the current subarray
               if (fi < first[index[j-inc]] || (fi == first[index[j-inc]] && se < second[index[j-inc]])) {
                  // Swap elements j and j - inc, implicitly use the fact
                  // that ktemp hold element j to avoid having to assign to
                  // element j-inc
                    index[j] = index[j-inc];
               } else {
                  // There are no more elements in this sorted subarray which
                  // are less than element j
                  break;
               }
            } // End loop over the elements in the current subarray
            // Move index[j] out of temporary storage
            index[j] = ktemp;
            // The element has been inserted into the subarray.
         } // End loop over all subarrays defined by the current increment
      } else if ( swapSecond && !swapFirst ) {
         for (Int_t k = inc; k < n; k++) {
            const Int_t ktemp = index[k];
            const Int_t fi = first [ktemp];
            const Int_t se = second[k];
            Int_t j;
            for (j = k; j >= inc; j -= inc) {
               if (fi < first[index[j-inc]] || (fi == first[index[j-inc]] && se < second[j-inc])) {
                  index [j] = index[j-inc];
                  second[j] = second[j-inc];
               } else {
                  break;
               }
            }
            index[j]  = ktemp;
            second[j] = se;
         }
      } else if (swapFirst  && !swapSecond) {
         for (Int_t k = inc; k < n; k++ ) {
            const Int_t ktemp = index[k];
            const Int_t fi = first[k];
            const Int_t se = second[ktemp];
            Int_t j;
            for (j = k; j >= inc; j -= inc) {
               if ( fi < first[j-inc] || (fi == first[j-inc] && se < second[ index[j-inc]])) {
                  index[j] = index[j-inc];
                  first[j] = first[j-inc];
               } else {
                  break;
               }
            }
            index[j] = ktemp;
            first[j] = fi;
         }
      } else { // Swap both
         for (Int_t k = inc; k < n; k++ ) {
            const Int_t ktemp = index[k];
            const Int_t fi = first [k];
            const Int_t se = second[k];
            Int_t j;
            for (j = k; j >= inc; j -= inc) {
               if ( fi < first[j-inc] || (fi == first[j-inc] && se < second[j-inc])) {
                  index [j] = index [j-inc];
                  first [j] = first [j-inc];
                  second[j] = second[j-inc];
               } else {
                  break;
               }
            }
            index[j]  = ktemp;
            first[j]  = fi;
            second[j] = se;
         }
      }
   }
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::SetMatrixArray(const Element *data,Option_t *option)
{
  // Copy array data to matrix . It is assumed that array is of size >= fNelems
  // (=)))) fNrows*fNcols
  // option indicates how the data is stored in the array:
  // option =
  //          'F'   : column major (Fortran) m[i][j] = array[i+j*fNrows]
  //          else  : row major    (C)       m[i][j] = array[i*fNcols+j] (default)

   R__ASSERT(IsValid());

   TString opt = option;
   opt.ToUpper();

   Element *elem = GetMatrixArray();
   if (opt.Contains("F")) {
      for (Int_t irow = 0; irow < fNrows; irow++) {
         const Int_t off1 = irow*fNcols;
         Int_t off2 = 0;
         for (Int_t icol = 0; icol < fNcols; icol++) {
            elem[off1+icol] = data[off2+irow];
            off2 += fNrows;
         }
      }
   }
   else
      memcpy(elem,data,fNelems*sizeof(Element));

   return *this;
}

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTBase<Element>::IsSymmetric() const
{
// Check whether matrix is symmetric

  R__ASSERT(IsValid());

   if ((fNrows != fNcols) || (fRowLwb != fColLwb))
      return kFALSE;

   const Element * const elem = GetMatrixArray();
   for (Int_t irow = 0; irow < fNrows; irow++) {
      const Int_t rowOff = irow*fNcols;
      Int_t colOff = 0;
      for (Int_t icol = 0; icol < irow; icol++) {
         if (elem[rowOff+icol] != elem[colOff+irow])
           return kFALSE;
         colOff += fNrows;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTBase<Element>::GetMatrix2Array(Element *data,Option_t *option) const
{
// Copy matrix data to array . It is assumed that array is of size >= fNelems
// (=)))) fNrows*fNcols
// option indicates how the data is stored in the array:
// option =
//          'F'   : column major (Fortran) array[i+j*fNrows] = m[i][j]
//          else  : row major    (C)       array[i*fNcols+j] = m[i][j] (default)

   R__ASSERT(IsValid());

   TString opt = option;
   opt.ToUpper();

   const Element * const elem = GetMatrixArray();
   if (opt.Contains("F")) {
      for (Int_t irow = 0; irow < fNrows; irow++) {
         const Int_t off1 = irow*fNcols;
         Int_t off2 = 0;
         for (Int_t icol = 0; icol < fNcols; icol++) {
            data[off2+irow] = elem[off1+icol];
            off2 += fNrows;
         }
      }
   }
   else
      memcpy(data,elem,fNelems*sizeof(Element));
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::InsertRow(Int_t rown,Int_t coln,const Element *v,Int_t n)
{
// Copy n elements from array v to row rown starting at column coln

   const Int_t arown = rown-fRowLwb;
   const Int_t acoln = coln-fColLwb;
   const Int_t nr = (n > 0) ? n : fNcols;

   if (gMatrixCheck) {
      if (arown >= fNrows || arown < 0) {
         Error("InsertRow","row %d out of matrix range",rown);
         return *this;
      }

      if (acoln >= fNcols || acoln < 0) {
         Error("InsertRow","column %d out of matrix range",coln);
         return *this;
      }

      if (acoln+nr > fNcols || nr < 0) {
         Error("InsertRow","row length %d out of range",nr);
         return *this;
      }
   }

   const Int_t off = arown*fNcols+acoln;
   Element * const elem = GetMatrixArray()+off;
   memcpy(elem,v,nr*sizeof(Element));

   return *this;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTBase<Element>::ExtractRow(Int_t rown,Int_t coln,Element *v,Int_t n) const
{
// Store in array v, n matrix elements of row rown starting at column coln

   const Int_t arown = rown-fRowLwb;
   const Int_t acoln = coln-fColLwb;
   const Int_t nr = (n > 0) ? n : fNcols;

   if (gMatrixCheck) {
      if (arown >= fNrows || arown < 0) {
         Error("ExtractRow","row %d out of matrix range",rown);
         return;
      }

      if (acoln >= fNcols || acoln < 0) {
         Error("ExtractRow","column %d out of matrix range",coln);
         return;
      }

      if (acoln+n >= fNcols || nr < 0) {
         Error("ExtractRow","row length %d out of range",nr);
         return;
      }
   }

   const Int_t off = arown*fNcols+acoln;
   const Element * const elem = GetMatrixArray()+off;
   memcpy(v,elem,nr*sizeof(Element));
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::Shift(Int_t row_shift,Int_t col_shift)
{
// Shift the row index by adding row_shift and the column index by adding
// col_shift, respectively. So [rowLwb..rowUpb][colLwb..colUpb] becomes
// [rowLwb+row_shift..rowUpb+row_shift][colLwb+col_shift..colUpb+col_shift]

   fRowLwb += row_shift;
   fColLwb += col_shift;

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::Zero()
{
// Set matrix elements to zero

   R__ASSERT(IsValid());
   memset(this->GetMatrixArray(),0,fNelems*sizeof(Element));

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::Abs()
{
// Take an absolute value of a matrix, i.e. apply Abs() to each element.

   R__ASSERT(IsValid());

         Element *ep = this->GetMatrixArray();
   const Element * const fp = ep+fNelems;
   while (ep < fp) {
      *ep = TMath::Abs(*ep);
      ep++;
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::Sqr()
{
// Square each element of the matrix.

   R__ASSERT(IsValid());

         Element *ep = this->GetMatrixArray();
   const Element * const fp = ep+fNelems;
   while (ep < fp) {
      *ep = (*ep) * (*ep);
      ep++;
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::Sqrt()
{
// Take square root of all elements.

   R__ASSERT(IsValid());

         Element *ep = this->GetMatrixArray();
   const Element * const fp = ep+fNelems;
   while (ep < fp) {
      *ep = TMath::Sqrt(*ep);
      ep++;
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::UnitMatrix()
{
// Make a unit matrix (matrix need not be a square one).

   R__ASSERT(IsValid());

   Element *ep = this->GetMatrixArray();
   memset(ep,0,fNelems*sizeof(Element));
   for (Int_t i = fRowLwb; i <= fRowLwb+fNrows-1; i++)
      for (Int_t j = fColLwb; j <= fColLwb+fNcols-1; j++)
         *ep++ = (i==j ? 1.0 : 0.0);

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::NormByDiag(const TVectorT<Element> &v,Option_t *option)
{
// option:
// "D"   :  b(i,j) = a(i,j)/sqrt(abs*(v(i)*v(j)))  (default)
// else  :  b(i,j) = a(i,j)*sqrt(abs*(v(i)*v(j)))  (default)

   R__ASSERT(IsValid());
   R__ASSERT(v.IsValid());

   if (gMatrixCheck) {
      const Int_t nMax = TMath::Max(fNrows,fNcols);
      if (v.GetNoElements() < nMax) {
         Error("NormByDiag","vector shorter than matrix diagonal");
         return *this;
      }
   }

   TString opt(option);
   opt.ToUpper();
   const Int_t divide = (opt.Contains("D")) ? 1 : 0;

   const Element *pV = v.GetMatrixArray();
         Element *mp = this->GetMatrixArray();

   if (divide) {
      for (Int_t irow = 0; irow < fNrows; irow++) {
         if (pV[irow] != 0.0) {
            for (Int_t icol = 0; icol < fNcols; icol++) {
               if (pV[icol] != 0.0) {
                  const Element val = TMath::Sqrt(TMath::Abs(pV[irow]*pV[icol]));
                  *mp++ /= val;
               } else {
                  Error("NormbyDiag","vector element %d is zero",icol);
                  mp++;
               }
            }
         } else {
            Error("NormbyDiag","vector element %d is zero",irow);
            mp += fNcols;
         }
      }
   } else {
      for (Int_t irow = 0; irow < fNrows; irow++) {
         for (Int_t icol = 0; icol < fNcols; icol++) {
            const Element val = TMath::Sqrt(TMath::Abs(pV[irow]*pV[icol]));
            *mp++ *= val;
         }
      }
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTBase<Element>::RowNorm() const
{
// Row matrix norm, MAX{ SUM{ |M(i,j)|, over j}, over i}.
// The norm is induced by the infinity vector norm.

   R__ASSERT(IsValid());

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
         Element norm = 0;

   // Scan the matrix row-after-row
   while (ep < fp) {
      Element sum = 0;
      // Scan a row to compute the sum
      for (Int_t j = 0; j < fNcols; j++)
         sum += TMath::Abs(*ep++);
      norm = TMath::Max(norm,sum);
   }

   R__ASSERT(ep == fp);

   return norm;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTBase<Element>::ColNorm() const
{
// Column matrix norm, MAX{ SUM{ |M(i,j)|, over i}, over j}.
// The norm is induced by the 1 vector norm.

   R__ASSERT(IsValid());

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNcols;
         Element norm = 0;

   // Scan the matrix col-after-col
   while (ep < fp) {
      Element sum = 0;
      // Scan a col to compute the sum
      for (Int_t i = 0; i < fNrows; i++,ep += fNcols)
         sum += TMath::Abs(*ep);
      ep -= fNelems-1;         // Point ep to the beginning of the next col
      norm = TMath::Max(norm,sum);
   }

   R__ASSERT(ep == fp);

   return norm;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTBase<Element>::E2Norm() const
{
// Square of the Euclidian norm, SUM{ m(i,j)^2 }.

   R__ASSERT(IsValid());

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
         Element sum = 0;

   for ( ; ep < fp; ep++)
      sum += (*ep) * (*ep);

   return sum;
}

//______________________________________________________________________________
template<class Element>
Int_t TMatrixTBase<Element>::NonZeros() const
{
// Compute the number of elements != 0.0

   R__ASSERT(IsValid());

   Int_t nr_nonzeros = 0;
   const Element *ep = this->GetMatrixArray();
   const Element * const fp = ep+fNelems;
   while (ep < fp)
      if (*ep++ != 0.0) nr_nonzeros++;

   return nr_nonzeros;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTBase<Element>::Sum() const
{
// Compute sum of elements

   R__ASSERT(IsValid());

   Element sum = 0.0;
   const Element *ep = this->GetMatrixArray();
   const Element * const fp = ep+fNelems;
   while (ep < fp)
      sum += *ep++;

   return sum;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTBase<Element>::Min() const
{
// return minimum matrix element value

   R__ASSERT(IsValid());

   const Element * const ep = this->GetMatrixArray();
   const Int_t index = TMath::LocMin(fNelems,ep);
   return ep[index];
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTBase<Element>::Max() const
{
// return maximum vector element value

   R__ASSERT(IsValid());

   const Element * const ep = this->GetMatrixArray();
   const Int_t index = TMath::LocMax(fNelems,ep);
   return ep[index];
}

//______________________________________________________________________________
template<class Element>
void TMatrixTBase<Element>::Draw(Option_t *option)
{
// Draw this matrix
// The histogram is named "TMatrixT" by default and no title

   gROOT->ProcessLine(Form("THistPainter::PaintSpecialObjects((TObject*)0x%lx,\"%s\");",
                           (ULong_t)this, option));
}

//______________________________________________________________________________
template<class Element>
void TMatrixTBase<Element>::Print(Option_t *option) const
{
   // Print the matrix as a table of elements.
   // By default the format "%11.4g" is used to print one element.
   // One can specify an alternative format with eg
   //  option ="f=  %6.2f  "

   if (!IsValid()) {
      Error("Print","Matrix is invalid");
      return;
   }

   //build format
   const char *format = "%11.4g ";
   if (option) {
      const char *f = strstr(option,"f=");
      if (f) format = f+2;
   }
   char topbar[100];
   snprintf(topbar,100,format,123.456789);
   Int_t nch = strlen(topbar)+1;
   if (nch > 18) nch = 18;
   char ftopbar[20];
   for (Int_t i = 0; i < nch; i++) ftopbar[i] = ' ';
   Int_t nk = 1 + Int_t(TMath::Log10(fNcols));
   snprintf(ftopbar+nch/2,20-nch/2,"%s%dd","%",nk);
   Int_t nch2 = strlen(ftopbar);
   for (Int_t i = nch2; i < nch; i++) ftopbar[i] = ' ';
   ftopbar[nch] = '|';
   ftopbar[nch+1] = 0;

   printf("\n%dx%d matrix is as follows",fNrows,fNcols);

   Int_t cols_per_sheet = 5;
   if (nch <= 8) cols_per_sheet =10;
   const Int_t ncols  = fNcols;
   const Int_t nrows  = fNrows;
   const Int_t collwb = fColLwb;
   const Int_t rowlwb = fRowLwb;
   nk = 5+nch*TMath::Min(cols_per_sheet,fNcols);
   for (Int_t i = 0; i < nk; i++) topbar[i] = '-';
   topbar[nk] = 0;
   for (Int_t sheet_counter = 1; sheet_counter <= ncols; sheet_counter += cols_per_sheet) {
      printf("\n\n     |");
      for (Int_t j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= ncols; j++)
         printf(ftopbar,j+collwb-1);
      printf("\n%s\n",topbar);
      if (fNelems <= 0) continue;
      for (Int_t i = 1; i <= nrows; i++) {
         printf("%4d |",i+rowlwb-1);
         for (Int_t j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= ncols; j++)
            printf(format,(*this)(i+rowlwb-1,j+collwb-1));
            printf("\n");
      }
   }
   printf("\n");
}

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTBase<Element>::operator==(Element val) const
{
// Are all matrix elements equal to val?

   R__ASSERT(IsValid());

   if (val == 0. && fNelems == 0)
      return kTRUE;

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
   for (; ep < fp; ep++)
      if (!(*ep == val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTBase<Element>::operator!=(Element val) const
{
// Are all matrix elements not equal to val?

   R__ASSERT(IsValid());

   if (val == 0. && fNelems == 0)
      return kFALSE;

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
   for (; ep < fp; ep++)
      if (!(*ep != val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTBase<Element>::operator<(Element val) const
{
// Are all matrix elements < val?

   R__ASSERT(IsValid());

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
   for (; ep < fp; ep++)
      if (!(*ep < val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTBase<Element>::operator<=(Element val) const
{
// Are all matrix elements <= val?

   R__ASSERT(IsValid());

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
   for (; ep < fp; ep++)
      if (!(*ep <= val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTBase<Element>::operator>(Element val) const
{
// Are all matrix elements > val?

   R__ASSERT(IsValid());

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
   for (; ep < fp; ep++)
      if (!(*ep > val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTBase<Element>::operator>=(Element val) const
{
// Are all matrix elements >= val?

   R__ASSERT(IsValid());

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
   for (; ep < fp; ep++)
      if (!(*ep >= val))
         return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::Apply(const TElementActionT<Element> &action)
{
// Apply action to each matrix element

   R__ASSERT(IsValid());

   Element *ep = this->GetMatrixArray();
   const Element * const ep_last = ep+fNelems;
   while (ep < ep_last)
      action.Operation(*ep++);

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::Apply(const TElementPosActionT<Element> &action)
{
// Apply action to each element of the matrix. To action the location
// of the current element is passed.

   R__ASSERT(IsValid());

   Element *ep = this->GetMatrixArray();
   for (action.fI = fRowLwb; action.fI < fRowLwb+fNrows; action.fI++)
      for (action.fJ = fColLwb; action.fJ < fColLwb+fNcols; action.fJ++)
         action.Operation(*ep++);

   R__ASSERT(ep == this->GetMatrixArray()+fNelems);

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTBase<Element>::Randomize(Element alpha,Element beta,Double_t &seed)
{
// Randomize matrix element values

   R__ASSERT(IsValid());

   const Element scale = beta-alpha;
   const Element shift = alpha/scale;

         Element *       ep = GetMatrixArray();
   const Element * const fp = ep+fNelems;
   while (ep < fp)
      *ep++ = scale*(Drand(seed)+shift);

   return *this;
}

//______________________________________________________________________________
template<class Element>
Bool_t operator==(const TMatrixTBase<Element> &m1,const TMatrixTBase<Element> &m2)
{
// Check to see if two matrices are identical.

   if (!AreCompatible(m1,m2)) return kFALSE;
   return (memcmp(m1.GetMatrixArray(),m2.GetMatrixArray(),
                   m1.GetNoElements()*sizeof(Element)) == 0);
}

//______________________________________________________________________________
template<class Element>
Element E2Norm(const TMatrixTBase<Element> &m1,const TMatrixTBase<Element> &m2)
{
// Square of the Euclidian norm of the difference between two matrices.

   if (gMatrixCheck && !AreCompatible(m1,m2)) {
      ::Error("E2Norm","matrices not compatible");
      return -1.0;
   }

   const Element *        mp1 = m1.GetMatrixArray();
   const Element *        mp2 = m2.GetMatrixArray();
   const Element * const fmp1 = mp1+m1.GetNoElements();

   Element sum = 0.0;
   for (; mp1 < fmp1; mp1++, mp2++)
      sum += (*mp1 - *mp2)*(*mp1 - *mp2);

   return sum;
}

//______________________________________________________________________________
template<class Element1,class Element2>
Bool_t AreCompatible(const TMatrixTBase<Element1> &m1,const TMatrixTBase<Element2> &m2,Int_t verbose)
{
// Check that matrice sm1 and m2 areboth valid and have identical shapes .
   if (!m1.IsValid()) {
      if (verbose)
         ::Error("AreCompatible", "matrix 1 not valid");
      return kFALSE;
   }
   if (!m2.IsValid()) {
      if (verbose)
         ::Error("AreCompatible", "matrix 2 not valid");
      return kFALSE;
   }

   if (m1.GetNrows()  != m2.GetNrows()  || m1.GetNcols()  != m2.GetNcols() ||
       m1.GetRowLwb() != m2.GetRowLwb() || m1.GetColLwb() != m2.GetColLwb()) {
      if (verbose)
         ::Error("AreCompatible", "matrices 1 and 2 not compatible");
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
void Compare(const TMatrixTBase<Element> &m1,const TMatrixTBase<Element> &m2)
{
// Compare two matrices and print out the result of the comparison.

   if (!AreCompatible(m1,m2)) {
      Error("Compare(const TMatrixTBase<Element> &,const TMatrixTBase<Element> &)","matrices are incompatible");
      return;
   }

   printf("\n\nComparison of two TMatrices:\n");

   Element norm1  = 0;      // Norm of the Matrices
   Element norm2  = 0;
   Element ndiff  = 0;      // Norm of the difference
   Int_t   imax   = 0;      // For the elements that differ most
   Int_t   jmax   = 0;
   Element difmax = -1;

   for (Int_t i = m1.GetRowLwb(); i <= m1.GetRowUpb(); i++) {
      for (Int_t j = m1.GetColLwb(); j < m1.GetColUpb(); j++) {
         const Element mv1 = m1(i,j);
         const Element mv2 = m2(i,j);
         const Element diff = TMath::Abs(mv1-mv2);

         if (diff > difmax) {
            difmax = diff;
            imax = i;
            jmax = j;
         }
         norm1 += TMath::Abs(mv1);
         norm2 += TMath::Abs(mv2);
         ndiff += TMath::Abs(diff);
      }
   }

   printf("\nMaximal discrepancy    \t\t%g", difmax);
   printf("\n   occured at the point\t\t(%d,%d)",imax,jmax);
   const Element mv1 = m1(imax,jmax);
   const Element mv2 = m2(imax,jmax);
   printf("\n Matrix 1 element is    \t\t%g", mv1);
   printf("\n Matrix 2 element is    \t\t%g", mv2);
   printf("\n Absolute error v2[i]-v1[i]\t\t%g", mv2-mv1);
   printf("\n Relative error\t\t\t\t%g\n",
          (mv2-mv1)/TMath::Max(TMath::Abs(mv2+mv1)/2,(Element)1e-7));

   printf("\n||Matrix 1||   \t\t\t%g", norm1);
   printf("\n||Matrix 2||   \t\t\t%g", norm2);
   printf("\n||Matrix1-Matrix2||\t\t\t\t%g", ndiff);
   printf("\n||Matrix1-Matrix2||/sqrt(||Matrix1|| ||Matrix2||)\t%g\n\n",
          ndiff/TMath::Max(TMath::Sqrt(norm1*norm2),1e-7));
}

//______________________________________________________________________________
template<class Element>
Bool_t VerifyMatrixValue(const TMatrixTBase<Element> &m,Element val,Int_t verbose,Element maxDevAllow)
{
// Validate that all elements of matrix have value val within maxDevAllow.

   R__ASSERT(m.IsValid());

   if (m == 0)
      return kTRUE;

   Int_t   imax      = 0;
   Int_t   jmax      = 0;
   Element maxDevObs = 0;

   if (TMath::Abs(maxDevAllow) <= 0.0)
      maxDevAllow = std::numeric_limits<Element>::epsilon();

   for (Int_t i = m.GetRowLwb(); i <= m.GetRowUpb(); i++) {
      for (Int_t j = m.GetColLwb(); j <= m.GetColUpb(); j++) {
         const Element dev = TMath::Abs(m(i,j)-val);
         if (dev > maxDevObs) {
            imax    = i;
            jmax    = j;
            maxDevObs = dev;
         }
      }
   }

   if (maxDevObs == 0)
      return kTRUE;

   if (verbose) {
      printf("Largest dev for (%d,%d); dev = |%g - %g| = %g\n",imax,jmax,m(imax,jmax),val,maxDevObs);
      if(maxDevObs > maxDevAllow)
         Error("VerifyElementValue","Deviation > %g\n",maxDevAllow);
   }

   if(maxDevObs > maxDevAllow)
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
Bool_t VerifyMatrixIdentity(const TMatrixTBase<Element> &m1,const TMatrixTBase<Element> &m2,Int_t verbose,
                            Element maxDevAllow)
{
// Verify that elements of the two matrices are equal within MaxDevAllow .

   if (!AreCompatible(m1,m2,verbose))
      return kFALSE;

   if (m1 == 0 && m2 == 0)
      return kTRUE;

   Int_t   imax      = 0;
   Int_t   jmax      = 0;
   Element maxDevObs = 0;

   if (TMath::Abs(maxDevAllow) <= 0.0)
      maxDevAllow = std::numeric_limits<Element>::epsilon();

   for (Int_t i = m1.GetRowLwb(); i <= m1.GetRowUpb(); i++) {
      for (Int_t j = m1.GetColLwb(); j <= m1.GetColUpb(); j++) {
         const Element dev = TMath::Abs(m1(i,j)-m2(i,j));
         if (dev > maxDevObs) {
            imax = i;
            jmax = j;
            maxDevObs = dev;
         }
      }
   }

   if (maxDevObs == 0)
      return kTRUE;

   if (verbose) {
      printf("Largest dev for (%d,%d); dev = |%g - %g| = %g\n",
              imax,jmax,m1(imax,jmax),m2(imax,jmax),maxDevObs);
      if (maxDevObs > maxDevAllow)
         Error("VerifyMatrixValue","Deviation > %g\n",maxDevAllow);
   }

   if (maxDevObs > maxDevAllow)
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTBase<Element>::Streamer(TBuffer &R__b)
{
// Stream an object of class TMatrixTBase<Element>.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TMatrixTBase<Element>::Class(),this,R__v,R__s,R__c);
      } else {
         Error("TMatrixTBase<Element>::Streamer","Unknown version number: %d",R__v);
         R__ASSERT(0);
      }
      if (R__v < 4) MakeValid();
   } else {
      R__b.WriteClassBuffer(TMatrixTBase<Element>::Class(),this);
   }
}

// trick to return a reference to nan in operator(i,j_ when i,j are outside of range
template<class Element>
struct nan_value_t {
   static Element gNanValue;
};
template<>
Double_t nan_value_t<Double_t>::gNanValue = std::numeric_limits<Double_t>::quiet_NaN();
template<>
Float_t nan_value_t<Float_t>::gNanValue = std::numeric_limits<Float_t>::quiet_NaN();

template<class Element>
Element & TMatrixTBase<Element>::NaNValue()
{
   return nan_value_t<Element>::gNanValue;
}


template class TMatrixTBase<Float_t>;

template Bool_t   operator==          <Float_t>(const TMatrixFBase &m1,const TMatrixFBase &m2);
template Float_t  E2Norm              <Float_t>(const TMatrixFBase &m1,const TMatrixFBase &m2);
template Bool_t   AreCompatible<Float_t,Float_t>
                                               (const TMatrixFBase &m1,const TMatrixFBase &m2,Int_t verbose);
template Bool_t   AreCompatible<Float_t,Double_t>
                                               (const TMatrixFBase &m1,const TMatrixDBase &m2,Int_t verbose);
template void     Compare             <Float_t>(const TMatrixFBase &m1,const TMatrixFBase &m2);
template Bool_t   VerifyMatrixValue   <Float_t>(const TMatrixFBase &m,Float_t val,Int_t verbose,Float_t maxDevAllow);
template Bool_t   VerifyMatrixValue   <Float_t>(const TMatrixFBase &m,Float_t val);
template Bool_t   VerifyMatrixIdentity<Float_t>(const TMatrixFBase &m1,const TMatrixFBase &m2,
                                                Int_t verbose,Float_t maxDevAllowN);
template Bool_t   VerifyMatrixIdentity<Float_t>(const TMatrixFBase &m1,const TMatrixFBase &m2);

template class TMatrixTBase<Double_t>;

template Bool_t   operator==          <Double_t>(const TMatrixDBase &m1,const TMatrixDBase &m2);
template Double_t E2Norm              <Double_t>(const TMatrixDBase &m1,const TMatrixDBase &m2);
template Bool_t   AreCompatible<Double_t,Double_t>
                                               (const TMatrixDBase &m1,const TMatrixDBase &m2,Int_t verbose);
template Bool_t   AreCompatible<Double_t,Float_t>
                                               (const TMatrixDBase &m1,const TMatrixFBase &m2,Int_t verbose);
template void     Compare             <Double_t>(const TMatrixDBase &m1,const TMatrixDBase &m2);
template Bool_t   VerifyMatrixValue   <Double_t>(const TMatrixDBase &m,Double_t val,Int_t verbose,Double_t maxDevAllow);
template Bool_t   VerifyMatrixValue   <Double_t>(const TMatrixDBase &m,Double_t val);
template Bool_t   VerifyMatrixIdentity<Double_t>(const TMatrixDBase &m1,const TMatrixDBase &m2,
                                                 Int_t verbose,Double_t maxDevAllow);
template Bool_t   VerifyMatrixIdentity<Double_t>(const TMatrixDBase &m1,const TMatrixDBase &m2);
