// @(#)root/matrix:$Name:  $:$Id: TMatrixDBase.cxx,v 1.47 2003/09/05 09:21:54 brun Exp $
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
// have been added (LU,QRH,Cholesky and SVD) . The decompositions are   //
// used in matrix inversion, equation solving.                          //
//                                                                      //
// Matrix elements are arranged in memory in a ROW-wise fashion         //
// fashion . For (n x m) matrices where n*m < kSizeMax (=25 currently)  //
// storage space is avaialble on the stack, thus avoiding expensive     //
// allocation/deallocation of heap space . However, this introduces of  //
// course kSizeMax overhead for each matrix object . If this is an      //
// issue recompile with a new appropriate value (>=0) for kSizeMax      //
//                                                                      //
// Another way to assign and store matrix data is through Adoption      //
// see for instance stress_linalg.cxx file .                            //
//                                                                      //
// Unless otherwise specified, matrix and vector indices always start   //
// with 0, spanning up to the specified limit-1. However, there are     //
// constructors to which one can specify aribtrary lower and upper      //
// bounds, e.g. TMatrixD m(1,10,1,5) defines a matrix that ranges,      //
// nad that can be addresses, from 1..10, 1..5 (a(1,1)..a(10,5)).       //
//                                                                      //
// The present package provides all facilities to completely AVOID      //
// returning matrices. Use "TMatrixD A(TMatrixD::kTransposed,B);"       //
// and other fancy constructors as much as possible. If one really needs//
// to return a matrix, return a TMatLazy object instead. The            //
// conversion is completely transparent to the end user, e.g.           //
// "TMatrixD m = THaarMatrixD(5);" and _is_ efficient.                  //
//                                                                      //
// Since TMatrixD et al. are fully integrated in ROOT they of course    //
// can be stored in a ROOT database.                                    //
//                                                                      //
// For usage examples see $ROOTSYS/test/stress_linalg.cxx               //
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
//    the constructor. Note, TMatLazy (see below) can construct         //
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
//        TMatrixColumn m1(m,1); m1(2) = 3; // the same as m(2,1)=3;    //
//    Note, constructing of, say, TMatrixDDiag does *not* involve any   //
//    copying of any elements of the source matrix.                     //
//                                                                      //
// 5. It's possible (and encouraged) to use "nested" functions          //
//    For example, creating of a Hilbert matrix can be done as follows: //
//                                                                      //
//    void foo(const TMatrixD &m)                                       //
//    {                                                                 //
//      TMatrixD m1(TMatrixD::kZero,m);                                 //
//      struct MakeHilbert : public TElementPosActionD {                //
//        void Operation(Double_t &element) { element = 1./(fI+fJ-1); } //
//      };                                                              //
//      m1.Apply(MakeHilbert());                                        //
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
//       TMatrixD haar = THaarMatrixD(5);                               //
//    THaarMatrixD() is a *class*, not a simple function. However       //
//    similar this looks to a returning of an object (see note #1       //
//    above), it's dramatically different. THaarMatrixD() constructs a  //
//    TMatLazy, an object of just a few bytes long. A special           //
//    "TMatrixD(const TMatLazy &recipe)" constructor follows the        //
//    recipe and makes the matrix haar() right in place. No matrix      //
//    element is moved whatsoever!                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix Base class                                                    //
//                                                                      //
//  matrix properties are stored here, however the data storage is part //
//  of the derived classes                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDBase.h"

ClassImp(TMatrixDBase)

//______________________________________________________________________________
void TMatrixDBase::Delete_m(Int_t size,Double_t* m)
{ 
  // delete data pointer m, if it was assigned on the heap

  if (m) {
    if (size > kSizeMax)
    {
      delete [] m;
      m = 0;
    }
  }       
}

//______________________________________________________________________________
Double_t* TMatrixDBase::New_m(Int_t size)
{
  // return data pointer . if requested size <= kSizeMax, assign pointer
  // to the stack space

  if (size == 0) return 0;
  else {
    if ( size <= kSizeMax )
      return fDataStack;
    else {
      Double_t *heap = new Double_t[size];
      return heap;
    }
  }
}

//______________________________________________________________________________
Int_t TMatrixDBase::Memcpy_m(Double_t *newp,const Double_t *oldp,Int_t copySize,
                            Int_t newSize,Int_t oldSize)
{
  // copy copySize doubles from *oldp to *newp . However take care of the
  // situation where both pointers are assigned to the same stack space

  if (copySize == 0 || oldp == newp)
    return 0;
  else {
    if ( newSize <= kSizeMax && oldSize <= kSizeMax ) {
      // both pointers are inside fDataStack, be careful with copy direction !
      if (newp > oldp) {
        for (Int_t i = copySize-1; i >= 0; i--)
          newp[i] = oldp[i];
      } else {
        for (Int_t i = 0; i < copySize; i++)
          newp[i] = oldp[i];
      }
    }
    else
      memcpy(newp,oldp,copySize*sizeof(Double_t));
  }
  return 0;
}

//______________________________________________________________________________
Bool_t TMatrixDBase::IsSymmetric() const
{
  Assert(IsValid());

  if ((fNrows != fNcols) || (fRowLwb != fColLwb))
    return kFALSE;

  const Double_t * const elem = GetElements();
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
void TMatrixDBase::GetMatrixElements(Double_t *data,Option_t *option) const 
{
  // Copy matrix data to array . It is assumed that array is of size >= fNrows*fNcols
  // option indicates how the data is stored in the array:
  // option =
  //          'F'   : column major (Fortran) array[i+j*fNrows] = m[i][j]
  //          else  : row major    (C)       array[i*fNcols+j] = m[i][j] (default)

  Assert(IsValid());

  TString opt = option;
  opt.ToUpper();

  const Double_t * const elem = GetElements();
  if (opt.Contains("F")) {
    for (Int_t irow = 0; irow < fNrows; irow++) {
      const Int_t off1 = irow*fNcols;
      Int_t off2 = 0;
      for (Int_t icol = 0; icol < fNcols; icol++)
        data[off2+irow] = elem[off1+icol];
        off2 += fNrows;
    }
  }
  else
    memcpy(data,elem,fNelems*sizeof(Double_t));      
}

//______________________________________________________________________________
void TMatrixDBase::SetMatrixElements(const Double_t *data,Option_t *option) 
{
  // Copy array data to matrix . It is assumed that array is of size >= fNrows*fNcols
  // option indicates how the data is stored in the array:
  // option =
  //          'F'   : column major (Fortran) m[i][j] = array[i+j*fNrows]
  //          else  : row major    (C)       m[i][j] = array[i*fNcols+j] (default)

  Assert(IsValid());

  TString opt = option;
  opt.ToUpper();

  Double_t *elem = GetElements();
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
    memcpy(elem,data,fNelems*sizeof(Double_t));
}

//______________________________________________________________________________
void TMatrixDBase::Shift(Int_t row_shift,Int_t col_shift)
{
  // Shift the row index by adding row_shift and the column index by adding
  // col_shift, respectively. So [rowLwb..rowUpb][colLwb..colUpb] becomes
  // [rowLwb+row_shift..rowUpb+row_shift][colLwb+col_shift..colUpb+col_shift]

  fRowLwb += row_shift;
  fColLwb += col_shift;
}

//______________________________________________________________________________
void TMatrixDBase::ResizeTo(Int_t nrows,Int_t ncols)
{
  // Set size of the matrix to nrows x ncols
  // New dynamic elements are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleted.

  if (!fIsOwner) {
    Error("ResizeTo(nrows,ncols)","Not owner of data array,cannot resize");
    return;
  }

  if (IsValid()) {
    if (fNrows == nrows && fNcols == ncols)
      return;

    Double_t    *elements_old = GetElements();
    const Int_t  nelems_old   = fNelems;
    const Int_t  nrows_old    = fNrows;
    const Int_t  ncols_old    = fNcols;

    Allocate(nrows,ncols);

    Assert(IsValid());

    // Copy overlap
    const Int_t ncols_copy = TMath::Min(fNcols,ncols_old); 
    const Int_t nrows_copy = TMath::Min(fNrows,nrows_old); 

    const Int_t nelems_new = fNelems;
    Double_t *elements_new = GetElements();
    if (ncols_old < fNcols) {
      for (Int_t i = nrows_copy-1; i >= 0; i--)
        Memcpy_m(elements_new+i*fNcols,elements_old+i*ncols_old,ncols_copy,
                 nelems_new,nelems_old);
    } else {
      for (Int_t i = 0; i < nrows_copy; i++)
        Memcpy_m(elements_new+i*fNcols,elements_old+i*ncols_old,ncols_copy,
                 nelems_new,nelems_old);
    }

    Delete_m(nelems_old,elements_old);
  } else {
    Allocate(nrows,ncols,0,0,1);
  }
}

//______________________________________________________________________________
void TMatrixDBase::ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
{
  // Set size of the matrix to [row_lwb:row_upb] x [col_lwb:col_upb]
  // New dynamic elemenst are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleted.

  if (!fIsOwner) {
    Error("ResizeTo(row_lwb,row_upb,col_lwb,col_upb)","Not owner of data array,cannot resize");
    return;
  }

  const Int_t new_nrows = row_upb-row_lwb+1;
  const Int_t new_ncols = col_upb-col_lwb+1;

  if (IsValid()) {

    if (fNrows  == new_nrows  && fNcols  == new_ncols &&
        fRowLwb == row_lwb    && fColLwb == col_lwb)
       return;

    Double_t    *elements_old = GetElements();
    const Int_t  nelems_old   = fNelems;
    const Int_t  nrows_old    = fNrows;
    const Int_t  ncols_old    = fNcols;
    const Int_t  rowLwb_old   = fRowLwb;
    const Int_t  colLwb_old   = fColLwb;

    Allocate(new_nrows,new_ncols,row_lwb,col_lwb);

    Assert(IsValid());

    // Copy overlap
    const Int_t rowLwb_copy = TMath::Max(fRowLwb,rowLwb_old); 
    const Int_t colLwb_copy = TMath::Max(fColLwb,colLwb_old); 
    const Int_t rowUpb_copy = TMath::Min(fRowLwb+fNrows-1,rowLwb_old+nrows_old-1); 
    const Int_t colUpb_copy = TMath::Min(fColLwb+fNcols-1,colLwb_old+ncols_old-1); 

    const Int_t nrows_copy = rowUpb_copy-rowLwb_copy+1;
    const Int_t ncols_copy = colUpb_copy-colLwb_copy+1;

    Double_t *elements_new = GetElements();
    if (nrows_copy > 0 && ncols_copy > 0) {
      const Int_t colOldOff = colLwb_copy-colLwb_old;
      const Int_t colNewOff = colLwb_copy-fColLwb;
      if (ncols_old < fNcols) {
        for (Int_t i = nrows_copy-1; i >= 0; i--) {
          const Int_t iRowOld = rowLwb_copy+i-rowLwb_old;
          const Int_t iRowNew = rowLwb_copy+i-fRowLwb;
          Memcpy_m(elements_new+iRowNew*fNcols+colNewOff,
                   elements_old+iRowOld*ncols_old+colOldOff,ncols_copy,fNelems,nelems_old);
        }
      } else {
        for (Int_t i = 0; i < nrows_copy; i++) {
          const Int_t iRowOld = rowLwb_copy+i-rowLwb_old;
          const Int_t iRowNew = rowLwb_copy+i-fRowLwb;
          Memcpy_m(elements_new+iRowNew*fNcols+colNewOff,
                   elements_old+iRowOld*ncols_old+colOldOff,ncols_copy,fNelems,nelems_old);
        }
      }
    }

    Delete_m(nelems_old,elements_old);
  } else {
    Allocate(new_nrows,new_ncols,row_lwb,col_lwb,1);
  }
}

//______________________________________________________________________________
Double_t TMatrixDBase::RowNorm() const
{
  // Row matrix norm, MAX{ SUM{ |M(i,j)|, over j}, over i}.
  // The norm is induced by the infinity vector norm.

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNelems;
        Double_t norm = 0;

  // Scan the matrix row-after-row
  while (ep < fp) {
    Double_t sum = 0;
    // Scan a row to compute the sum
    for (Int_t j = 0; j < fNcols; j++)
      sum += TMath::Abs(*ep++);
    norm = TMath::Max(norm,sum);
  }

  Assert(ep == fp);

  return norm;
}

//______________________________________________________________________________
Double_t TMatrixDBase::ColNorm() const    
{
  // Column matrix norm, MAX{ SUM{ |M(i,j)|, over i}, over j}.
  // The norm is induced by the 1 vector norm.

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNcols;
        Double_t norm = 0;

  // Scan the matrix col-after-col
  while (ep < fp) {
    Double_t sum = 0;
    // Scan a col to compute the sum
    for (Int_t i = 0; i < fNrows; i++,ep += fNcols)
      sum += TMath::Abs(*ep);
    ep -= fNelems-1;         // Point ep to the beginning of the next col
    norm = TMath::Max(norm,sum);
  }

  Assert(ep == fp);

  return norm;
}

//______________________________________________________________________________
Double_t TMatrixDBase::E2Norm() const  
{
  // Square of the Euclidian norm, SUM{ m(i,j)^2 }.

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNelems;
        Double_t sum = 0;

  for ( ; ep < fp; ep++)
    sum += (*ep) * (*ep);

  return sum;
}

//______________________________________________________________________________
void TMatrixDBase::Draw(Option_t *option)
{
  // Draw this matrix using an intermediate histogram
  // The histogram is named "TMatrixD" by default and no title

  //create the hist utility manager (a plugin)
  TVirtualUtilHist *util = (TVirtualUtilHist*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilHist");
  if (!util) {
    TPluginHandler *h;
    if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualUtilHist"))) {
      if (h->LoadPlugin() == -1)
        return;
      h->ExecPlugin(0);
      util = (TVirtualUtilHist*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilHist");
    }
  }
 util->PaintMatrix(*this,option);
}

//______________________________________________________________________________
void TMatrixDBase::Print(Option_t *) const
{
  // Print the matrix as a table of elements (zeros are printed as dots).

  Assert(IsValid());

  printf("\nMatrix %dx%d is as follows",fNrows,fNcols);

  const Int_t cols_per_sheet = 5;

  const Int_t ncols  = fNcols;
  const Int_t nrows  = fNrows;
  const Int_t collwb = fColLwb;
  const Int_t rowlwb = fRowLwb;
  for (Int_t sheet_counter = 1; sheet_counter <= ncols; sheet_counter += cols_per_sheet) {
    printf("\n\n     |");
    for (Int_t j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= ncols; j++)
      printf("   %6d  |",j+collwb-1);
    printf("\n%s\n","------------------------------------------------------------------");
    for (Int_t i = 1; i <= nrows; i++) {
      printf("%4d |",i+rowlwb-1);
      for (Int_t j = sheet_counter; j < sheet_counter+cols_per_sheet && j <= ncols; j++)
        printf("%11.4g ",(*this)(i+rowlwb-1,j+collwb-1));
        printf("\n");
    }
  }
  printf("\n");
}

//______________________________________________________________________________
Bool_t TMatrixDBase::operator==(Double_t val) const
{
  // Are all matrix elements equal to val?

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNelems;
  for (; ep < fp; ep++)
    if (!(*ep == val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixDBase::operator!=(Double_t val) const
{
  // Are all matrix elements not equal to val?

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNelems;
  for (; ep < fp; ep++) 
    if (!(*ep != val))  
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixDBase::operator<(Double_t val) const
{
  // Are all matrix elements < val?

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNelems;
  for (; ep < fp; ep++) 
    if (!(*ep < val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixDBase::operator<=(Double_t val) const
{
  // Are all matrix elements <= val?

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNelems;
  for (; ep < fp; ep++) 
    if (!(*ep <= val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixDBase::operator>(Double_t val) const
{
  // Are all matrix elements > val?

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNelems;
  for (; ep < fp; ep++) 
    if (!(*ep > val))  
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixDBase::operator>=(Double_t val) const
{
  // Are all matrix elements >= val?

  Assert(IsValid());

  const Double_t *       ep = GetElements();
  const Double_t * const fp = ep+fNelems;
  for (; ep < fp; ep++) 
    if (!(*ep >= val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Double_t E2Norm(const TMatrixDBase &m1,const TMatrixDBase &m2)
{
  // Square of the Euclidian norm of the difference between two matrices.

  if (!AreCompatible(m1,m2)) {
    ::Error("E2Norm","matrices not compatible");
    return -1.0;
  }

  const Double_t *        mp1 = m1.GetElements();
  const Double_t *        mp2 = m2.GetElements();
  const Double_t * const fmp1 = mp1+m1.GetNoElements();

  Double_t sum = 0.0;
  for (; mp1 < fmp1; mp1++, mp2++)
    sum += (*mp1 - *mp2)*(*mp1 - *mp2);

  return sum;
}

//______________________________________________________________________________
Bool_t AreCompatible(const TMatrixDBase &m1,const TMatrixDBase &m2,Int_t verbose)
{
  if (!m1.IsValid()) {
    if (verbose)
      ::Error("AreCompatible", "matrix 1 not initialized");
    return kFALSE;
  }
  if (!m2.IsValid()) {
    if (verbose)
      ::Error("AreCompatible", "matrix 2 not initialized");
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
Bool_t AreCompatible(const TMatrixDBase &m1,const TMatrixFBase &m2,Int_t verbose)
{
  if (!m1.IsValid()) {
    if (verbose)
      ::Error("AreCompatible", "matrix 1 not initialized");
    return kFALSE;
  }
  if (!m2.IsValid()) {
    if (verbose)
      ::Error("AreCompatible", "matrix 2 not initialized");
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
void Compare(const TMatrixDBase &m1,const TMatrixDBase &m2)
{
  // Compare two matrices and print out the result of the comparison.

  if (!AreCompatible(m1,m2)) {
    Error("Compare(const TMatrixDBase &,const TMatrixDBase &)","matrices are incompatible");
    return;
  }

  printf("\n\nComparison of two TMatrices:\n");

  Double_t norm1  = 0;      // Norm of the Matrices
  Double_t norm2  = 0;
  Double_t ndiff  = 0;      // Norm of the difference
  Int_t    imax   = 0;      // For the elements that differ most
  Int_t    jmax   = 0;
  Double_t difmax = -1;

  for (Int_t i = m1.GetRowLwb(); i <= m1.GetRowUpb(); i++) {
    for (Int_t j = m1.GetColLwb(); j < m1.GetColUpb(); j++) {
      const Double_t mv1 = m1(i,j);
      const Double_t mv2 = m2(i,j);
      const Double_t diff = TMath::Abs(mv1-mv2);

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
  const Double_t mv1 = m1(imax,jmax);
  const Double_t mv2 = m2(imax,jmax);
  printf("\n Matrix 1 element is    \t\t%g", mv1);
  printf("\n Matrix 2 element is    \t\t%g", mv2);
  printf("\n Absolute error v2[i]-v1[i]\t\t%g", mv2-mv1);
  printf("\n Relative error\t\t\t\t%g\n",
         (mv2-mv1)/TMath::Max(TMath::Abs(mv2+mv1)/2,(Double_t)1e-7));

  printf("\n||Matrix 1||   \t\t\t%g", norm1);
  printf("\n||Matrix 2||   \t\t\t%g", norm2);
  printf("\n||Matrix1-Matrix2||\t\t\t\t%g", ndiff);
  printf("\n||Matrix1-Matrix2||/sqrt(||Matrix1|| ||Matrix2||)\t%g\n\n",
         ndiff/TMath::Max(TMath::Sqrt(norm1*norm2),1e-7));
}

//______________________________________________________________________________
Bool_t VerifyMatrixValue(const TMatrixDBase &m,Double_t val,Int_t verbose,Double_t maxDevAllow)
{
  // Validate that all elements of matrix have value val within maxDevAllow.

  Assert(m.IsValid());

  Int_t    imax      = 0;
  Int_t    jmax      = 0;
  Double_t maxDevObs = 0;

  for (Int_t i = m.GetRowLwb(); i <= m.GetRowUpb(); i++) {
    for (Int_t j = m.GetColLwb(); j <= m.GetColUpb(); j++) {
      const Double_t dev = TMath::Abs(m(i,j)-val);
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
Bool_t VerifyMatrixIdentity(const TMatrixDBase &m1,const TMatrixDBase &m2,Int_t verbose,
                            Double_t maxDevAllow)
{
   // Verify that elements of the two matrices are equal within MaxDevAllow .

  if (!AreCompatible(m1,m2))
    return kFALSE;

  Int_t    imax      = 0;
  Int_t    jmax      = 0;
  Double_t maxDevObs = 0;

  for (Int_t i = m1.GetRowLwb(); i <= m1.GetRowUpb(); i++) {
    for (Int_t j = m1.GetColLwb(); j <= m1.GetColUpb(); j++) {
      const Double_t dev = TMath::Abs(m1(i,j)-m2(i,j));
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
    if(maxDevObs > maxDevAllow)
      Error("VerifyMatrixValue","Deviation > %g\n",maxDevAllow);
  }

  if(maxDevObs > maxDevAllow)
    return kFALSE;
  return kTRUE;
}

//______________________________________________________________________________
void TMatrixDBase::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixDBase.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
    if (R__v > 1) {
      TMatrixDBase::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
      return;
    }
    //====process old versions before automatic schema evolution
    TObject::Streamer(R__b);
    R__b >> fNrows;
    R__b >> fNcols;
    R__b >> fRowLwb;
    R__b >> fColLwb;
    R__b.CheckByteCount(R__s,R__c,TMatrixDBase::IsA());
    //====end of old versions
  } else {
    TMatrixDBase::Class()->WriteBuffer(R__b,this);
  }
}
