// @(#)root/matrix:$Name:  $:$Id: TMatrixDSparse.cxx,v 1.9 2004/05/18 20:04:46 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Feb 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSparse                                                       //
//                                                                      //
// Implementation of a general sparse matrix in the Harwell-Boeing      //
// format                                                               //
//                                                                      //
// Besides the usual shape/size decsriptors of a matrix like fNrows,    //
// fRowLwb,fNcols and fColLwb, we also store a row index, fRowIndex and //
// column index, fColIndex only for those elements unequal zero:        //
//                                                                      //
// fRowIndex[0,..,fNrows]:    Stores for each row the index range of    //
//                            the elements in the data and column array //
// fColIndex[0,..,fNelems-1]: Stores the column number for each data    //
//                            element != 0                              //
//                                                                      //
// As an example how to access all sparse data elements:                //
//                                                                      //
// for (Int_t irow = 0; irow < fNrows; irow++) {                        //
//   const Int_t sIndex = fRowIndex[irow];                              //
//   const Int_t eIndex = fRowIndex[irow+1];                            //
//   for (Int_t index = sIndex; index < eIndex; index++) {              //
//     const Int_t icol = fColIndex[index];                             //
//     const Double_t data = fElements[index];                          //
//     printf("data(%d,%d) = %.4e\n",irow+fRowLwb,icol+fColLwb,data);   //
//   }                                                                  //
// }                                                                    //
//                                                                      //
// When checking whether sparse matrices are compatible (like in an     //
// assigment !), not only the shape parameters are compared but also    //
// the sparse structure through fRowIndex and fColIndex .               //
//                                                                      //
// Several methods exist to fill a sparse matrix with data entries.     //
// Most are the same like for dense matrices but some care has to be    //
// taken with regard to performance. In the constructor, always the     //
// shape of the matrix has to be specified in some form . Data can be   //
// entered through the following methods :                              //
// 1. constructor                                                       //
//    TMatrixDSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,         //
//                   Int_t col_upb,Int_t nr_nonzeros,                   //
//                   Int_t *row, Int_t *col,Double_t *data);            //
//    It uses SetMatrixArray(..), see below                             //
// 2. copy constructors                                                 //
// 3. SetMatrixArray(Int_t nr,Int_t *irow,Int_t *icol,Double_t *data)   //
//    where it is expected that the irow,icol and data array contain    //
//    nr entries . Only the entries with non-zero data[i] value are     //
//    inserted !                                                        //
// 4. TMatrixDSparse a(n,m); for(....) { a(i,j) = ....                  //
//    This is a very flexible method but expensive :                    //
//    - if no entry for slot (i,j) is found in the sparse index table   //
//      it will be entered, which involves some memory management !     //
//    - before invoking this method in a loop it is smart to first      //
//      set the index table through a call to SetSparseIndex(..)        //
// 5. SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixDBase &source)    //
//    the matrix to be inserted at position (row_lwb,col_lwb) can be    //
//    both dense or sparse .                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDSparse.h"
#include "Riostream.h"

ClassImp(TMatrixDSparse)

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(Int_t no_rows,Int_t no_cols)
{
  // Space is allocated for row/column indices and data, but the sparse structure
  // information has still to be set !

  Allocate(no_rows,no_cols,0,0,1);
}

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
{
  // Space is allocated for row/column indices and data, but the sparse structure
  // information has still to be set !

  Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb,1);
}

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                               Int_t nr,Int_t *row, Int_t *col,Double_t *data)
{
  // Space is allocated for row/column indices and data. Sparse row/column index
  // structure together with data is coming from the arrays, row, col and data, resp .

  const Int_t irowmin = TMath::LocMin(nr,row);
  const Int_t irowmax = TMath::LocMax(nr,row);
  const Int_t icolmin = TMath::LocMin(nr,col);
  const Int_t icolmax = TMath::LocMax(nr,col);
  
  Assert(row[irowmin] >= row_lwb && row[irowmax] <= row_upb);
  Assert(col[icolmin] >= col_lwb && col[icolmax] <= col_upb);

  Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb,1,nr);

  SetMatrixArray(nr,row,col,data);
}

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(const TMatrixDSparse &another) : TMatrixDBase(another)
{
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb(),1,
           another.GetNoElements());
  memcpy(fRowIndex,another.GetRowIndexArray(),fNrowIndex*sizeof(Int_t));
  memcpy(fColIndex,another.GetColIndexArray(),fNelems*sizeof(Int_t));

  *this = another;
}

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(const TMatrixD &another) : TMatrixDBase(another)
{
  const Int_t nr_nonzeros = another.NonZeros();
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb(),1,nr_nonzeros);
  SetSparseIndex(another);
  *this = another;
}

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(EMatrixCreatorsOp1 op,const TMatrixDSparse &prototype)
{
  // Create a matrix applying a specific operation to the prototype.
  // Supported operations are: kZero, kUnit, kTransposed and kAtA
  
  Assert(this != &prototype);
  Invalidate();
  
  Assert(prototype.IsValid());
  
  Int_t nr_nonzeros = 0;

  switch(op) {
    case kZero:
      Allocate(prototype.GetNrows(),prototype.GetNcols(),
               prototype.GetRowLwb(),prototype.GetColLwb(),1,nr_nonzeros);
      break;
    
    case kUnit:
      {
        const Int_t nrows  = prototype.GetNrows();
        const Int_t ncols  = prototype.GetNcols();
        const Int_t rowLwb = prototype.GetRowLwb();
        const Int_t colLwb = prototype.GetColLwb();
        for (Int_t i = rowLwb; i <= rowLwb+nrows-1; i++)
          for (Int_t j = colLwb; j <= colLwb+ncols-1; j++)
            if (i==j) nr_nonzeros++;
        Allocate(nrows,ncols,rowLwb,colLwb,1,nr_nonzeros);
        UnitMatrix();
        break;
      }
    case kTransposed:
      Allocate(prototype.GetNcols(), prototype.GetNrows(),
               prototype.GetColLwb(),prototype.GetRowLwb(),1,prototype.GetNoElements());
      Transpose(prototype);
      break;

    case kAtA:
      {
        const TMatrixDSparse at(TMatrixDSparse::kTransposed,prototype);
        AMultBt(at,at);
        break;
      }

    default:
      Error("TMatrixDSparse(EMatrixCreatorOp1)","operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(const TMatrixDSparse &a,EMatrixCreatorsOp2 op,const TMatrixDSparse &b)
{
  // Create a matrix applying a specific operation to two prototypes.
  // Supported operations are: kMult (a*b), kMultTranspose (a*b'), kPlus (a+b), kMinus (a-b)

  Invalidate();

  Assert(a.IsValid());
  Assert(b.IsValid());

  switch(op) {
    case kMult:
      AMultB(a,b);
      break;

    case kMultTranspose:
      AMultBt(a,b);
      break;

    case kPlus:
      APlusB(a,b);
      break;

    case kMinus:
      AMinusB(a,b);
      break;

    default:
      Error("TMatrixDSparse(EMatrixCreatorOp2)", "operation %d not yet implemented",op);
  }
}

//______________________________________________________________________________
void TMatrixDSparse::Allocate(Int_t no_rows,Int_t no_cols,Int_t row_lwb,Int_t col_lwb,
                              Int_t init,Int_t nr_nonzeros)
{ 
  // Allocate new matrix. Arguments are number of rows, columns, row lowerbound (0 default)
  // and column lowerbound (0 default), 0 initialization flag and number of non-zero 
  // elements (only relevant for sparse format).
  
  Invalidate();
  
  if (no_rows <= 0 || no_cols <= 0 || nr_nonzeros < 0)
  { 
    Error("Allocate","no_rows=%d no_cols=%d non_zeros=%d",no_rows,no_cols,nr_nonzeros);
    return;
  }
  
  fNrows     = no_rows;
  fNcols     = no_cols;
  fRowLwb    = row_lwb;
  fColLwb    = col_lwb;
  fNrowIndex = fNrows+1;
  fNelems    = nr_nonzeros;
  fIsOwner   = kTRUE;
  fTol       = DBL_EPSILON;
  
  fRowIndex = new Int_t[fNrowIndex];
  if (init)
    memset(fRowIndex,0,fNrowIndex*sizeof(Int_t));

  if (fNelems > 0) {
    fElements = new Double_t[fNelems];
    fColIndex = new Int_t   [fNelems];
    if (init) {
      memset(fElements,0,fNelems*sizeof(Double_t));
      memset(fColIndex,0,fNelems*sizeof(Int_t));
    }
  } else {
    fElements = 0;
    fColIndex = 0;
  }
}

//______________________________________________________________________________
void TMatrixDSparse::SetSparseIndexAB(const TMatrixDSparse &a,const TMatrixDSparse &b)
{
  // Set the row/column indices to the "sum" of matrices a and b
  // It is assumed that enough space was reserved

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
      a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
    Error("SetSparseIndexAB","matrices not compatible");
    return;
  }

  const Int_t * const pRowIndexa = a.GetRowIndexArray();
  const Int_t * const pRowIndexb = b.GetRowIndexArray();
  const Int_t * const pColIndexa = a.GetColIndexArray();
  const Int_t * const pColIndexb = b.GetColIndexArray();

        Int_t * const pRowIndexc = this->GetRowIndexArray();
        Int_t * const pColIndexc = this->GetColIndexArray();

  Int_t nc = 0;
  pRowIndexc[0] = 0;
  for (Int_t irowc = 0; irowc < a.GetNrows(); irowc++) {
    const Int_t sIndexa = pRowIndexa[irowc];
    const Int_t eIndexa = pRowIndexa[irowc+1];
    const Int_t sIndexb = pRowIndexb[irowc];
    const Int_t eIndexb = pRowIndexb[irowc+1];
    Int_t indexb = sIndexb;
    for (Int_t indexa = sIndexa; indexa < eIndexa; indexa++) {
      const Int_t icola = pColIndexa[indexa];
      for (; indexb < eIndexb; indexb++) {
        if (pColIndexb[indexb] >= icola) {
          if (pColIndexb[indexb] == icola)
            indexb++;
          break;
        }
        pColIndexc[nc++] = pColIndexb[indexb];
      }
      pColIndexc[nc++] = pColIndexa[indexa];
    }
    while (indexb < eIndexb) {
      if (pColIndexb[indexb++] > pColIndexa[eIndexa-1])
        pColIndexc[nc++] = pColIndexb[indexb-1];
    }
    pRowIndexc[irowc+1] = nc;
  }
}

//______________________________________________________________________________
void TMatrixDSparse::InsertRow(Int_t rown,Int_t coln,const Double_t *v,Int_t n)
{
  // Insert in row rown, n elements of array v at column coln

  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  const Int_t nr = (n > 0) ? n : fNcols;

  if (arown >= fNrows || arown < 0) {
    Error("InsertRow","row %d out of matrix range",rown);
    return;
  }

  if (acoln >= fNcols || acoln < 0) {
    Error("InsertRow","column %d out of matrix range",coln);
    return;
  }

  if (acoln+nr > fNcols || nr < 0) {
    Error("InsertRow","row length %d out of range",nr);
    return;
  }

  const Int_t sIndex = fRowIndex[arown];
  const Int_t eIndex = fRowIndex[arown+1];

  // check first how many slots are available from [acoln,..,acoln+nr-1]
  // also note lIndex and rIndex so that [sIndex..lIndex] and [rIndex..eIndex-1]
  // contain the row entries except for the region to be inserted

  Int_t nslots = 0;
  Int_t lIndex = sIndex-1;
  Int_t rIndex = sIndex-1;
  Int_t index;
  for (index = sIndex; index < eIndex; index++) {
    const Int_t icol = fColIndex[index];
    rIndex++;
    if (icol >= acoln+nr) break;
    if (icol >= acoln) nslots++;
    else               lIndex++;
  }

  const Int_t nelems_old = fNelems;
  const Int_t    *colIndex_old = fColIndex;
  const Double_t *elements_old = fElements;

  const Int_t ndiff = nr-nslots;
  fNelems += ndiff;
  fColIndex = new Int_t[fNelems];
  fElements = new Double_t[fNelems];

  for (Int_t irow = arown+1; irow < fNrows+1; irow++)
    fRowIndex[irow] += ndiff;

  if (lIndex+1 > 0) {
    memmove(fColIndex,colIndex_old,(lIndex+1)*sizeof(Int_t));
    memmove(fElements,elements_old,(lIndex+1)*sizeof(Double_t));
  }

  if (nelems_old > 0 && nelems_old-rIndex > 0) {
    memmove(fColIndex+rIndex+ndiff,colIndex_old+rIndex,(nelems_old-rIndex)*sizeof(Int_t));
    memmove(fElements+rIndex+ndiff,elements_old+rIndex,(nelems_old-rIndex)*sizeof(Double_t));
  }

  index = lIndex+1;
  for (Int_t i = 0; i < nr; i++) {
    fColIndex[index] = acoln+i;
    fElements[index] = v[i];
    index++;
  }

  if (colIndex_old) delete [] (Int_t*)    colIndex_old;
  if (elements_old) delete [] (Double_t*) elements_old;

  Assert(fNelems == fRowIndex[fNrowIndex-1]);
}

//______________________________________________________________________________
void TMatrixDSparse::ExtractRow(Int_t rown, Int_t coln, Double_t *v,Int_t n) const
{
  // Store in array v, n matrix elements of row rown starting at column coln

  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  const Int_t nr = (n > 0) ? n : fNcols;

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

  const Int_t sIndex = fRowIndex[arown];
  const Int_t eIndex = fRowIndex[arown+1];

  memset(v,0,nr*sizeof(Double_t));
  const Int_t    * const pColIndex = GetColIndexArray();
  const Double_t * const pData     = GetMatrixArray();
  for (Int_t index = sIndex; index < eIndex; index++) {
    const Int_t icol = pColIndex[index];
    if (icol < acoln || icol >= acoln+n) continue;
    v[icol-acoln] = pData[index];
  }
}

//______________________________________________________________________________
void TMatrixDSparse::AMultBt(const TMatrixDSparse &a,const TMatrixDSparse &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B'.
  // Note, matrix C is allocated for constr=1.
  
  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNcols() != b.GetNcols() || a.GetColLwb() != b.GetColLwb()) {
    Error("AMultBt","A and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == &a) {
    Error("AMultB","this = &a");
    Invalidate();
    return;
  }     

  if (this == &b) {
    Error("AMultB","this = &b");
    Invalidate();
    return;
  }     

  const Int_t * const pRowIndexa = a.GetRowIndexArray();
  const Int_t * const pColIndexa = a.GetColIndexArray();
  const Int_t * const pRowIndexb = b.GetRowIndexArray();
  const Int_t * const pColIndexb = b.GetColIndexArray();

  Int_t *pRowIndexc;
  Int_t *pColIndexc;
  if (constr) {
    // make a best guess of the sparse structure; it will guarantee
    // enough allocated space !

    Int_t nr_nonzero_rowa = 0;
    {
      for (Int_t irowa = 0; irowa < a.GetNrows(); irowa++)
        if (pRowIndexa[irowa] < pRowIndexa[irowa+1])
          nr_nonzero_rowa++;
    }
    Int_t nr_nonzero_rowb = 0;
    {
      for (Int_t irowb = 0; irowb < b.GetNrows(); irowb++)
        if (pRowIndexb[irowb] < pRowIndexb[irowb+1])
          nr_nonzero_rowb++;
    }

    Int_t nc = nr_nonzero_rowa*nr_nonzero_rowb; // best guess
    Allocate(a.GetNrows(),b.GetNrows(),a.GetRowLwb(),b.GetRowLwb(),1,nc);

    pRowIndexc = this->GetRowIndexArray();
    pColIndexc = this->GetColIndexArray();

    pRowIndexc[0] = 0;
    Int_t ielem = 0;
    for (Int_t irowa = 0; irowa < a.GetNrows(); irowa++) {
      pRowIndexc[irowa+1] = pRowIndexc[irowa];
      if (pRowIndexa[irowa] >= pRowIndexa[irowa+1]) continue;
      for (Int_t irowb = 0; irowb < b.GetNrows(); irowb++) {
        if (pRowIndexb[irowb] >= pRowIndexb[irowb+1]) continue;
        pRowIndexc[irowa+1]++;
        pColIndexc[ielem++] = irowb;
      }
    }
  } else {
    pRowIndexc = this->GetRowIndexArray();
    pColIndexc = this->GetColIndexArray();
  }

  const Double_t * const pDataa = a.GetMatrixArray();
  const Double_t * const pDatab = b.GetMatrixArray();
  Double_t * const pDatac = this->GetMatrixArray();
  Int_t shift = 0;
  Int_t indexc_r = 0;
  for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
    const Int_t sIndexc = pRowIndexc[irowc]+shift;
    const Int_t eIndexc = pRowIndexc[irowc+1];
    const Int_t sIndexa = pRowIndexa[irowc];
    const Int_t eIndexa = pRowIndexa[irowc+1];
    for (Int_t indexc = sIndexc; indexc < eIndexc; indexc++) {
      const Int_t icolc = pColIndexc[indexc];
      const Int_t sIndexb = pRowIndexb[icolc];
      const Int_t eIndexb = pRowIndexb[icolc+1];
      Double_t sum = 0.0;
      Int_t indexb = sIndexb;
      for (Int_t indexa = sIndexa; indexa < eIndexa && indexb < eIndexb; indexa++) {
        const Int_t icola = pColIndexa[indexa];
        while (indexb < eIndexb && pColIndexb[indexb] <= icola) {
          if (icola == pColIndexb[indexb]) {
            sum += pDataa[indexa]*pDatab[indexb];
            break;
          }
          indexb++;
        }
      }
      if (!constr)
        pDatac[indexc] = sum;
      else {
        if (sum != 0.0) {
          pRowIndexc[irowc+1]  = indexc_r+1;
          pColIndexc[indexc_r] = icolc;
          pDatac[indexc_r] = sum;
          indexc_r++;
        } else
          shift++;
      }
    }
  }

  if (constr)
    SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
void TMatrixDSparse::AMultBt(const TMatrixDSparse &a,const TMatrixD &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B'.
  // Note, matrix C is allocated for constr=1.
  
  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNcols() != b.GetNcols() || a.GetColLwb() != b.GetColLwb()) {
    Error("AMultBt","A and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == &a) {
    Error("AMultB","this = &a");
    Invalidate();
    return;
  }     

  if (this == dynamic_cast<const TMatrixDSparse *>(&b)) {
    Error("AMultB","this = &b");
    Invalidate();
    return;
  }     

  const Int_t * const pRowIndexa = a.GetRowIndexArray();
  const Int_t * const pColIndexa = a.GetColIndexArray();

  Int_t *pRowIndexc;
  Int_t *pColIndexc;
  if (constr) {
    // make a best guess of the sparse structure; it will guarantee
    // enough allocated space !

    Int_t nr_nonzero_rowa = 0;
    {
      for (Int_t irowa = 0; irowa < a.GetNrows(); irowa++)
        if (pRowIndexa[irowa] < pRowIndexa[irowa+1])
          nr_nonzero_rowa++;
    }
    Int_t nr_nonzero_rowb = b.GetNrows();

    Int_t nc = nr_nonzero_rowa*nr_nonzero_rowb; // best guess
    Allocate(a.GetNrows(),b.GetNrows(),a.GetRowLwb(),b.GetRowLwb(),1,nc);

    pRowIndexc = this->GetRowIndexArray();
    pColIndexc = this->GetColIndexArray();

    pRowIndexc[0] = 0;
    Int_t ielem = 0;
    for (Int_t irowa = 0; irowa < a.GetNrows(); irowa++) {
      pRowIndexc[irowa+1] = pRowIndexc[irowa];
      for (Int_t irowb = 0; irowb < b.GetNrows(); irowb++) {
        pRowIndexc[irowa+1]++;
        pColIndexc[ielem++] = irowb;
      }
    }
  } else {
    pRowIndexc = this->GetRowIndexArray();
    pColIndexc = this->GetColIndexArray();
  }

  const Double_t * const pDataa = a.GetMatrixArray();
  const Double_t * const pDatab = b.GetMatrixArray();
  Double_t * const pDatac = this->GetMatrixArray();
  Int_t indexc_r = 0;
  Int_t shift = 0;
  for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
    const Int_t sIndexc = pRowIndexc[irowc]+shift;
    const Int_t eIndexc = pRowIndexc[irowc+1];
    const Int_t sIndexa = pRowIndexa[irowc];
    const Int_t eIndexa = pRowIndexa[irowc+1];
    for (Int_t indexc = sIndexc; indexc < eIndexc; indexc++) {
      const Int_t icolc = pColIndexc[indexc];
      const Int_t off   = icolc*b.GetNcols();
      Double_t sum = 0.0;
      for (Int_t indexa = sIndexa; indexa < eIndexa; indexa++) {
        const Int_t icola = pColIndexa[indexa];
        sum += pDataa[indexa]*pDatab[off+icola];
      }
      if (!constr)
        pDatac[indexc] = sum;
      else {
        if (sum != 0.0) {
          pRowIndexc[irowc+1]  = indexc_r+1;
          pColIndexc[indexc_r] = icolc;
          pDatac[indexc_r] = sum;
          indexc_r++;
        } else
          shift++;
      }
    }
  }

  if (constr)
    SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
void TMatrixDSparse::AMultBt(const TMatrixD &a,const TMatrixDSparse &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B'.
  // Note, matrix C is allocated for constr=1.
  
  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNcols() != b.GetNcols() || a.GetColLwb() != b.GetColLwb()) {
    Error("AMultBt","A and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == dynamic_cast<const TMatrixDSparse *>(&a)) {
    Error("AMultB","this = &a");
    Invalidate();
    return;
  }     

  if (this == &b) {
    Error("AMultB","this = &b");
    Invalidate();
    return;
  }     

  const Int_t * const pRowIndexb = b.GetRowIndexArray();
  const Int_t * const pColIndexb = b.GetColIndexArray();

  Int_t *pRowIndexc;
  Int_t *pColIndexc;
  if (constr) {
    // make a best guess of the sparse structure; it will guarantee
    // enough allocated space !

    Int_t nr_nonzero_rowa = a.GetNrows();
    Int_t nr_nonzero_rowb = 0;
    {
      for (Int_t irowb = 0; irowb < b.GetNrows(); irowb++)
        if (pRowIndexb[irowb] < pRowIndexb[irowb+1])
          nr_nonzero_rowb++;
    }

    Int_t nc = nr_nonzero_rowa*nr_nonzero_rowb; // best guess
    Allocate(a.GetNrows(),b.GetNrows(),a.GetRowLwb(),b.GetRowLwb(),1,nc);

    pRowIndexc = this->GetRowIndexArray();
    pColIndexc = this->GetColIndexArray();

    pRowIndexc[0] = 0;
    Int_t ielem = 0;
    for (Int_t irowa = 0; irowa < a.GetNrows(); irowa++) {
      pRowIndexc[irowa+1] = pRowIndexc[irowa];
      for (Int_t irowb = 0; irowb < b.GetNrows(); irowb++) {
        if (pRowIndexb[irowb] >= pRowIndexb[irowb+1]) continue;
        pRowIndexc[irowa+1]++;
        pColIndexc[ielem++] = irowb;
      }
    }
  } else {
    pRowIndexc = this->GetRowIndexArray();
    pColIndexc = this->GetColIndexArray();
  }

  const Double_t * const pDataa = a.GetMatrixArray();
  const Double_t * const pDatab = b.GetMatrixArray();
  Double_t * const pDatac = this->GetMatrixArray();
  Int_t indexc_r = 0;
  Int_t shift = 0;
  for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
    const Int_t sIndexc = pRowIndexc[irowc]+shift;
    const Int_t eIndexc = pRowIndexc[irowc+1];
    const Int_t off   = irowc*a.GetNcols();
    for (Int_t indexc = sIndexc; indexc < eIndexc; indexc++) {
      const Int_t icolc = pColIndexc[indexc];
      const Int_t sIndexb = pRowIndexb[icolc];
      const Int_t eIndexb = pRowIndexb[icolc+1];
      Double_t sum = 0.0;
      for (Int_t indexb = sIndexb; indexb < eIndexb; indexb++) {
        const Int_t icolb = pColIndexb[indexb];
        sum += pDataa[off+icolb]*pDatab[indexb];
      }
      if (!constr)
        pDatac[indexc] = sum;
      else {
        if (sum != 0.0) {
          pRowIndexc[irowc+1]  = indexc_r+1;
          pColIndexc[indexc_r] = icolc;
          pDatac[indexc_r] = sum;
          indexc_r++;
        } else
          shift++;
      }
    }
  }

  if (constr)
    SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
void TMatrixDSparse::APlusB(const TMatrixDSparse &a,const TMatrixDSparse &b,Int_t constr)
{
  // General matrix addition. Create a matrix C such that C = A + B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
      a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
    Error("APlusB(const TMatrixDSparse &,const TMatrixDSparse &","matrices not compatible");
    return;
  }

  if (this == &a) {
    Error("APlusB","this = &a");
    Invalidate();
    return;
  }     

  if (this == &b) {
    Error("APlusB","this = &b");
    Invalidate();
    return;
  }     

  const Int_t * const pRowIndexa = a.GetRowIndexArray();
  const Int_t * const pRowIndexb = b.GetRowIndexArray();
  const Int_t * const pColIndexa = a.GetColIndexArray();
  const Int_t * const pColIndexb = b.GetColIndexArray();

  if (constr) {
    Int_t nc = 0;
    for (Int_t irowc = 0; irowc < a.GetNrows(); irowc++) {
      const Int_t sIndexa = pRowIndexa[irowc];
      const Int_t eIndexa = pRowIndexa[irowc+1];
      const Int_t sIndexb = pRowIndexb[irowc];
      const Int_t eIndexb = pRowIndexb[irowc+1];
      nc += eIndexa-sIndexa;
      Int_t indexb = sIndexb;
      for (Int_t indexa = sIndexa; indexa < eIndexa; indexa++) {
        const Int_t icola = pColIndexa[indexa];
        for (; indexb < eIndexb; indexb++) {
          if (pColIndexb[indexb] >= icola) {
            if (pColIndexb[indexb] == icola)
              indexb++;
            break;
          }
          nc++;
        }
      }
      while (indexb < eIndexb) {
        if (pColIndexb[indexb++] > pColIndexa[eIndexa-1])
          nc++;
      }
    }

    Allocate(a.GetNrows(),a.GetNcols(),a.GetRowLwb(),a.GetColLwb(),1,nc);
    SetSparseIndexAB(a,b);
  }

  Int_t * const pRowIndexc = this->GetRowIndexArray();
  Int_t * const pColIndexc = this->GetColIndexArray();

  const Double_t * const pDataa = a.GetMatrixArray();
  const Double_t * const pDatab = b.GetMatrixArray();
  Double_t * const pDatac = this->GetMatrixArray();
  Int_t indexc_r = 0;
  Int_t shift = 0;
  for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
    const Int_t sIndexc = pRowIndexc[irowc]+shift;
    const Int_t eIndexc = pRowIndexc[irowc+1];
    const Int_t sIndexa = pRowIndexa[irowc];
    const Int_t eIndexa = pRowIndexa[irowc+1];
    const Int_t sIndexb = pRowIndexb[irowc];
    const Int_t eIndexb = pRowIndexb[irowc+1];
    Int_t indexa = sIndexa;
    Int_t indexb = sIndexb;
    for (Int_t indexc = sIndexc; indexc < eIndexc; indexc++) {
      const Int_t icolc = pColIndexc[indexc];
      Double_t sum = 0.0;
      while (indexa < eIndexa && pColIndexa[indexa] <= icolc) {
        if (icolc == pColIndexa[indexa]) {
          sum += pDataa[indexa];
          break;
        }
        indexa++;
      }
      while (indexb < eIndexb && pColIndexb[indexb] <= icolc) {
        if (icolc == pColIndexb[indexb]) {
          sum += pDatab[indexb];
          break;
        }
        indexb++;
      }

      if (!constr)
        pDatac[indexc] = sum;
      else {
        if (sum != 0.0) {
          pRowIndexc[irowc+1]  = indexc_r+1;
          pColIndexc[indexc_r] = icolc;
          pDatac[indexc_r] = sum;
          indexc_r++;
        } else
          shift++;
      }
    }
  }

  if (constr)
    SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
void TMatrixDSparse::APlusB(const TMatrixDSparse &a,const TMatrixD &b,Int_t constr)
{
  // General matrix addition. Create a matrix C such that C = A + B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
      a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
    Error("APlusB(const TMatrixDSparse &,const TMatrixD &","matrices not compatible");
    return;
  }

  if (this == &a) {
    Error("APlusB","this = &a");
    Invalidate();
    return;
  }     

  if (this == dynamic_cast<const TMatrixDSparse *>(&b)) {
    Error("APlusB","this = &b");
    Invalidate();
    return;
  }     

  if (constr)
    *this = b;

  Int_t * const pRowIndexc = this->GetRowIndexArray();
  Int_t * const pColIndexc = this->GetColIndexArray();

  const Int_t * const pRowIndexa = a.GetRowIndexArray();
  const Int_t * const pColIndexa = a.GetColIndexArray();
      
  const Double_t * const pDataa = a.GetMatrixArray();
  Double_t * const pDatac = this->GetMatrixArray();
  Int_t indexc_r = 0;
  Int_t shift = 0;
  for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
    const Int_t sIndexc = pRowIndexc[irowc]+shift;
    const Int_t eIndexc = pRowIndexc[irowc+1];
    const Int_t sIndexa = pRowIndexa[irowc];
    const Int_t eIndexa = pRowIndexa[irowc+1];
    Int_t indexa = sIndexa;
    for (Int_t indexc = sIndexc; indexc < eIndexc; indexc++) {
      const Int_t icolc = pColIndexc[indexc];
      Double_t sum = pDatac[indexc];
      while (indexa < eIndexa && pColIndexa[indexa] <= icolc) {
        if (icolc == pColIndexa[indexa]) {
          sum += pDataa[indexa];
          break;
        }
        indexa++;
      }

      if (!constr)
        pDatac[indexc] = sum;
      else {
        if (sum != 0.0) {
          pRowIndexc[irowc+1]  = indexc_r+1;
          pColIndexc[indexc_r] = icolc;
          pDatac[indexc_r] = sum;
          indexc_r++;
        } else
          shift++;
      }
    }
  }

  if (constr)
    SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
void TMatrixDSparse::AMinusB(const TMatrixDSparse &a,const TMatrixDSparse &b,Int_t constr)
{
  // General matrix subtraction. Create a matrix C such that C = A - B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
      a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
    Error("AMinusB(const TMatrixDSparse &,const TMatrixDSparse &","matrices not compatible");
    return;
  }

  if (this == &a) {
    Error("AMinusB","this = &a");
    Invalidate();
    return;
  }     

  if (this == &b) {
    Error("AMinusB","this = &b");
    Invalidate();
    return;
  }     

  const Int_t * const pRowIndexa = a.GetRowIndexArray();
  const Int_t * const pRowIndexb = b.GetRowIndexArray();
  const Int_t * const pColIndexa = a.GetColIndexArray();
  const Int_t * const pColIndexb = b.GetColIndexArray();
      
  if (constr) {
    Int_t nc = 0;
    for (Int_t irowc = 0; irowc < a.GetNrows(); irowc++) {
      const Int_t sIndexa = pRowIndexa[irowc];
      const Int_t eIndexa = pRowIndexa[irowc+1];
      const Int_t sIndexb = pRowIndexb[irowc];
      const Int_t eIndexb = pRowIndexb[irowc+1];
      nc += eIndexa-sIndexa;
      Int_t indexb = sIndexb;
      for (Int_t indexa = sIndexa; indexa < eIndexa; indexa++) {
        const Int_t icola = pColIndexa[indexa];
        for (; indexb < eIndexb; indexb++) {
          if (pColIndexb[indexb] >= icola) {
            if (pColIndexb[indexb] == icola)
              indexb++;
            break;
          }
          nc++;
        }
      }
      while (indexb < eIndexb) {
        if (pColIndexb[indexb++] > pColIndexa[eIndexa-1])
          nc++;
      }
    }

    Allocate(a.GetNrows(),a.GetNcols(),a.GetRowLwb(),a.GetColLwb(),1,nc);
    SetSparseIndexAB(a,b);
  }

  Int_t * const pRowIndexc = this->GetRowIndexArray();
  Int_t * const pColIndexc = this->GetColIndexArray();

  const Double_t * const pDataa = a.GetMatrixArray();
  const Double_t * const pDatab = b.GetMatrixArray();
  Double_t * const pDatac = this->GetMatrixArray();
  Int_t indexc_r = 0;
  Int_t shift = 0;
  for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
    const Int_t sIndexc = pRowIndexc[irowc]+shift;
    const Int_t eIndexc = pRowIndexc[irowc+1];
    const Int_t sIndexa = pRowIndexa[irowc];
    const Int_t eIndexa = pRowIndexa[irowc+1];
    const Int_t sIndexb = pRowIndexb[irowc];
    const Int_t eIndexb = pRowIndexb[irowc+1];
    Int_t indexa = sIndexa;
    Int_t indexb = sIndexb;
    for (Int_t indexc = sIndexc; indexc < eIndexc; indexc++) {
      const Int_t icolc = pColIndexc[indexc];
      Double_t sum = 0.0;
      while (indexa < eIndexa && pColIndexa[indexa] <= icolc) {
        if (icolc == pColIndexa[indexa]) {
          sum += pDataa[indexa];
          break;
        }
        indexa++;
      }
      while (indexb < eIndexb && pColIndexb[indexb] <= icolc) {
        if (icolc == pColIndexb[indexb]) {
          sum -= pDatab[indexb];
          break;
        }
        indexb++;
      }

      if (!constr)
        pDatac[indexc] = sum;
      else {
        if (sum != 0.0) {
          pRowIndexc[irowc+1]  = indexc_r+1;
          pColIndexc[indexc_r] = icolc;
          pDatac[indexc_r] = sum;
          indexc_r++;
        } else
          shift++;
      }
    }
  }

  if (constr)
    SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
void TMatrixDSparse::AMinusB(const TMatrixDSparse &a,const TMatrixD &b,Int_t constr)
{
  // General matrix subtraction. Create a matrix C such that C = A - B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
      a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
    Error("AMinusB(const TMatrixDSparse &,const TMatrixD &","matrices not compatible");
    return;
  }

  if (this == &a) {
    Error("AMinusB","this = &a");
    Invalidate();
    return;
  }     

  if (this == dynamic_cast<const TMatrixDSparse *>(&b)) {
    Error("AMinusB","this = &b");
    Invalidate();
    return;
  }     

  if (constr)
    *this = b;

  Int_t * const pRowIndexc = this->GetRowIndexArray();
  Int_t * const pColIndexc = this->GetColIndexArray();

  const Int_t * const pRowIndexa = a.GetRowIndexArray();
  const Int_t * const pColIndexa = a.GetColIndexArray();
      
  const Double_t * const pDataa = a.GetMatrixArray();
  Double_t * const pDatac = this->GetMatrixArray();
  Int_t indexc_r = 0;
  Int_t shift = 0;
  for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
    const Int_t sIndexc = pRowIndexc[irowc]+shift;
    const Int_t eIndexc = pRowIndexc[irowc+1];
    const Int_t sIndexa = pRowIndexa[irowc];
    const Int_t eIndexa = pRowIndexa[irowc+1];
    Int_t indexa = sIndexa;
    for (Int_t indexc = sIndexc; indexc < eIndexc; indexc++) {
      const Int_t icolc = pColIndexc[indexc];
      Double_t sum = -pDatac[indexc];
      while (indexa < eIndexa && pColIndexa[indexa] <= icolc) {
        if (icolc == pColIndexa[indexa]) {
          sum += pDataa[indexa];
          break;
        }
        indexa++;
      }

      if (!constr)
        pDatac[indexc] = sum;
      else {
        if (sum != 0.0) {
          pRowIndexc[irowc+1]  = indexc_r+1;
          pColIndexc[indexc_r] = icolc;
          pDatac[indexc_r] = sum;
          indexc_r++;
        } else
          shift++;
      }
    }
  }

  if (constr)
    SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
void TMatrixDSparse::AMinusB(const TMatrixD &a,const TMatrixDSparse &b,Int_t constr)
{
  // General matrix subtraction. Create a matrix C such that C = A - B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
      a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
    Error("AMinusB(const TMatrixD &,const TMatrixDSparse &","matrices not compatible");
    return;
  }

  if (this == dynamic_cast<const TMatrixDSparse *>(&a)) {
    Error("AMinusB","this = &a");
    Invalidate();
    return;
  }     

  if (this == &b) {
    Error("AMinusB","this = &b");
    Invalidate();
    return;
  }     

  if (constr)
    *this = a;

  Int_t * const pRowIndexc = this->GetRowIndexArray();
  Int_t * const pColIndexc = this->GetColIndexArray();

  const Int_t * const pRowIndexb = b.GetRowIndexArray();
  const Int_t * const pColIndexb = b.GetColIndexArray();
      
  const Double_t * const pDatab = b.GetMatrixArray();
  Double_t * const pDatac = this->GetMatrixArray();
  Int_t indexc_r = 0;
  Int_t shift = 0;
  for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
    const Int_t sIndexc = pRowIndexc[irowc]+shift;
    const Int_t eIndexc = pRowIndexc[irowc+1];
    const Int_t sIndexb = pRowIndexb[irowc];
    const Int_t eIndexb = pRowIndexb[irowc+1];
    Int_t indexb = sIndexb;
    for (Int_t indexc = sIndexc; indexc < eIndexc; indexc++) {
      const Int_t icolc = pColIndexc[indexc];
      Double_t sum = pDatac[indexc];
      while (indexb < eIndexb && pColIndexb[indexb] <= icolc) {
        if (icolc == pColIndexb[indexb]) {
          sum -= pDatab[indexb];
          break;
        }
        indexb++;
      }

      if (!constr)
        pDatac[indexc] = sum;
      else {
        if (sum != 0.0) {
          pRowIndexc[irowc+1]  = indexc_r+1;
          pColIndexc[indexc_r] = icolc;
          pDatac[indexc_r] = sum;
          indexc_r++;
        } else
          shift++;
      }
    }
  }

  if (constr)
    SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
void TMatrixDSparse::GetMatrix2Array(Double_t *data,Option_t * /*option*/) const
{
  // Copy matrix data to array . It is assumed that array is of size >= fNelems

  Assert(IsValid());

  const Double_t * const elem = GetMatrixArray();
  memcpy(data,elem,fNelems*sizeof(Double_t));
}

//______________________________________________________________________________
void TMatrixDSparse::SetMatrixArray(Int_t nr,Int_t *row,Int_t *col,Double_t *data)
{
  // Copy nr elements from row/col index and data array to matrix . It is assumed
  // that arrays are of size >= nr

  Assert(IsValid());
  if (nr <= 0) return;

  const Int_t irowmin = TMath::LocMin(nr,row);
  const Int_t irowmax = TMath::LocMax(nr,row);
  const Int_t icolmin = TMath::LocMin(nr,col);
  const Int_t icolmax = TMath::LocMax(nr,col);

  Assert(row[irowmin] >= fRowLwb && row[irowmax] <= fRowLwb+fNrows-1);
  Assert(col[icolmin] >= fColLwb && col[icolmax] <= fColLwb+fNcols-1);

  DoubleLexSort(nr,row,col,data);

  Int_t nr_nonzeros = 0;
  const Double_t *ep        = data;
  const Double_t * const fp = data+nr;

  while (ep < fp)
    if (*ep++ != 0.0) nr_nonzeros++;

  // if nr_nonzeros != fNelems
  if (nr_nonzeros != fNelems) {
    if (fColIndex) { delete [] fColIndex; fColIndex = 0; }
    if (fElements) { delete [] fElements; fElements = 0; }
    fNelems = nr_nonzeros;
    if (fNelems > 0) {
      fColIndex = new Int_t[nr_nonzeros];
      fElements = new Double_t[nr_nonzeros];
    } else {
      fColIndex = 0;
      fElements = 0;
    }
  }

  if (fNelems <= 0)
    return;

  fRowIndex[0] = 0;
  Int_t ielem = 0;
  nr_nonzeros = 0;
  for (Int_t irow = 1; irow < fNrows+1; irow++) {
    if (row[ielem] < irow) {
      while (ielem < nr) {
        if (data[ielem] != 0.0) {
          fColIndex[nr_nonzeros] = col[ielem]-fColLwb;
          fElements[nr_nonzeros] = data[ielem];
          nr_nonzeros++;
        }
        ielem++;
        if (ielem >= nr || row[ielem] != row[ielem-1])
          break;
      }
    }
    fRowIndex[irow] = nr_nonzeros;
  }
}

//______________________________________________________________________________
void TMatrixDSparse::SetSparseIndex(const TMatrixDBase &source)
{
  // Use non-zero data of matrix source to set the sparse structure

  if (!AreCompatible(*this,source)) {
    Error("SetSparseIndex","matrices not compatible");
    return;
  }

  const Int_t nr_nonzeros = source.NonZeros();

  if (nr_nonzeros != fNelems)
    SetSparseIndex(nr_nonzeros);

  const Double_t *ep = source.GetMatrixArray();
  Int_t nr = 0;
  for (Int_t irow = 0; irow < fNrows; irow++) {
    fRowIndex[irow] = nr;
    for (Int_t icol = 0; icol < fNcols; icol++) {
      if (*ep != 0.0) {
        fColIndex[nr] = icol;
        nr++;
      }
      ep++;
    }
  }
  fRowIndex[fNrows] = nr;
}

//______________________________________________________________________________
void TMatrixDSparse::SetSparseIndex(const TMatrixDSparse &source)
{
  // copy the sparse structure from sparse matrix source

  Assert(source.IsValid());
  if (GetNrows()  != source.GetNrows()  || GetNcols()  != source.GetNcols() ||
      GetRowLwb() != source.GetRowLwb() || GetColLwb() != source.GetColLwb()) {
    Error("SetSparseIndex","matrices not compatible");
    return;
  }

  const Int_t nelem_s = source.GetNoElements();
  SetSparseIndex(nelem_s);

  memmove(fRowIndex,source.GetRowIndexArray(),fNrowIndex*sizeof(Int_t));
  memmove(fColIndex,source.GetColIndexArray(),fNelems*sizeof(Int_t));
}

//______________________________________________________________________________
void TMatrixDSparse::SetSparseIndex(Int_t nelems_new)
{
  // Increase/decrease the number of non-zero elements to nelems_new

  if (nelems_new != fNelems) {
    Int_t nr = TMath::Min(nelems_new,fNelems);
    Int_t *oIp = fColIndex;
    fColIndex = new Int_t[nelems_new];
    memmove(fColIndex,oIp,nr*sizeof(Int_t));
    if (oIp) delete [] oIp;
    Double_t *oDp = fElements;
    fElements = new Double_t[nelems_new];
    memmove(fElements,oDp,nr*sizeof(Double_t));
    if (oDp) delete [] oDp;
    fNelems = nelems_new;
    if (nelems_new > nr) {
      memset(fElements+nr,0,(nelems_new-nr)*sizeof(Double_t));
      memset(fColIndex+nr,0,(nelems_new-nr)*sizeof(Int_t));
    } else {
      for (Int_t irow = 0; irow < fNrowIndex; irow++)
        if (fRowIndex[irow] > nelems_new)
          fRowIndex[irow] = nelems_new;
    }
  }
}

//______________________________________________________________________________
void TMatrixDSparse::ResizeTo(Int_t nrows,Int_t ncols,Int_t nr_nonzeros)
{
  // Set size of the matrix to nrows x ncols with nr_nonzeros non-zero entries
  // if nr_nonzeros > 0 .
  // New dynamic elements are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleted.

  Assert(IsValid());
  if (!fIsOwner) {
    Error("ResizeTo(Int_t,Int_t,Int_t)","Not owner of data array,cannot resize");
    return;
  }

  if (fNelems > 0) {
    if (fNrows == nrows && fNcols == ncols &&
       (fNelems == nr_nonzeros || nr_nonzeros < 0))
      return;
    else if (nrows == 0 || ncols == 0 || nr_nonzeros == 0) {
      fNrows = nrows; fNcols = ncols;
      Clear();
      return;
    }

    const Double_t *elements_old = GetMatrixArray();
    const Int_t    *rowIndex_old = GetRowIndexArray();
    const Int_t    *colIndex_old = GetColIndexArray();

    Int_t nrows_old     = fNrows;
    Int_t nrowIndex_old = fNrowIndex;

    Int_t nelems_new;
    if (nr_nonzeros > 0)
      nelems_new = nr_nonzeros;
    else {
      nelems_new = 0;
      for (Int_t irow = 0; irow < nrows_old; irow++) {
        if (irow >= nrows) continue;
        const Int_t sIndex = rowIndex_old[irow];
        const Int_t eIndex = rowIndex_old[irow+1];
        for (Int_t index = sIndex; index < eIndex; index++) {
          const Int_t icol = colIndex_old[index];
          if (icol < ncols)
            nelems_new++;
        }
      }
    }

    Allocate(nrows,ncols,0,0,1,nelems_new);
    Assert(IsValid());

    Double_t *elements_new = GetMatrixArray();
    Int_t    *rowIndex_new = GetRowIndexArray();
    Int_t    *colIndex_new = GetColIndexArray();

    Int_t nelems_copy = 0;
    rowIndex_new[0] = 0;
    Bool_t cont = kTRUE;
    for (Int_t irow = 0; irow < nrows_old && cont; irow++) {
      if (irow >= nrows) continue;
      const Int_t sIndex = rowIndex_old[irow];
      const Int_t eIndex = rowIndex_old[irow+1];
      for (Int_t index = sIndex; index < eIndex; index++) {
        const Int_t icol = colIndex_old[index];
        if (icol < ncols) {
          rowIndex_new[irow+1]      = nelems_copy+1;
          colIndex_new[nelems_copy] = icol;
          elements_new[nelems_copy] = elements_old[index];
          nelems_copy++;
        }
        if (nelems_copy >= nelems_new) {
          cont = kFALSE;
          break;
        }
      }
    }

    if (rowIndex_old) delete [] (Int_t*)    rowIndex_old;
    if (colIndex_old) delete [] (Int_t*)    colIndex_old;
    if (elements_old) delete [] (Double_t*) elements_old;

    if (nrowIndex_old < fNrowIndex) {
      for (Int_t irow = nrowIndex_old; irow < fNrowIndex; irow++)
        rowIndex_new[irow] = rowIndex_new[nrowIndex_old-1];
    }
  } else {
    const Int_t nelems_new = (nr_nonzeros >= 0) ? nr_nonzeros : 0;
    Allocate(nrows,ncols,0,0,1,nelems_new);
  }
}

//______________________________________________________________________________
void TMatrixDSparse::ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                              Int_t nr_nonzeros)
{
  // Set size of the matrix to [row_lwb:row_upb] x [col_lwb:col_upb] with nr_nonzeros
  // non-zero entries if nr_nonzeros > 0 .
  // New dynamic elemenst are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleted.

  Assert(IsValid());
  if (!fIsOwner) {
    Error("ResizeTo(Int_t,Int_t,Int_t,Int_t,Int_t)","Not owner of data array,cannot resize");
    return;
  }

  const Int_t new_nrows = row_upb-row_lwb+1;
  const Int_t new_ncols = col_upb-col_lwb+1;

  if (fNelems > 0) {
    if (fNrows  == new_nrows && fNcols  == new_ncols &&
        fRowLwb == row_lwb   && fColLwb == col_lwb &&
        (fNelems == nr_nonzeros || nr_nonzeros < 0))
       return;
    else if (new_nrows == 0 || new_ncols == 0 || nr_nonzeros == 0) {
      fNrows = new_nrows; fNcols = new_ncols;
      fRowLwb = row_lwb; fColLwb = col_lwb;
      Clear();
      return;
    }

    const Int_t    *rowIndex_old = GetRowIndexArray();
    const Int_t    *colIndex_old = GetColIndexArray();
    const Double_t *elements_old = GetMatrixArray();

    const Int_t  nrowIndex_old = fNrowIndex;

    const Int_t  nrows_old     = fNrows;
    const Int_t  rowLwb_old    = fRowLwb;
    const Int_t  colLwb_old    = fColLwb;

    Int_t nelems_new;
    if (nr_nonzeros > 0)
      nelems_new = nr_nonzeros;
    else {
      nelems_new = 0;
      for (Int_t irow = 0; irow < nrows_old; irow++) {
        if (irow+rowLwb_old > row_upb || irow+rowLwb_old < row_lwb) continue;
        const Int_t sIndex = rowIndex_old[irow];
        const Int_t eIndex = rowIndex_old[irow+1];
        for (Int_t index = sIndex; index < eIndex; index++) {
          const Int_t icol = colIndex_old[index];
          if (icol+colLwb_old <= col_upb && icol+colLwb_old >= col_lwb)
            nelems_new++;
        }
      }
    }

    Allocate(new_nrows,new_ncols,row_lwb,col_lwb,1,nelems_new);
    Assert(IsValid());

    Int_t    *rowIndex_new = GetRowIndexArray();
    Int_t    *colIndex_new = GetColIndexArray();
    Double_t *elements_new = GetMatrixArray();

    Int_t nelems_copy = 0;
    rowIndex_new[0] = 0;
    Bool_t cont = kTRUE;
    const Int_t row_off = rowLwb_old-row_lwb;
    const Int_t col_off = colLwb_old-col_lwb;
    for (Int_t irow = 0; irow < nrows_old; irow++) {
      if (irow+rowLwb_old > row_upb || irow+rowLwb_old < row_lwb) continue;
      const Int_t sIndex = rowIndex_old[irow];
      const Int_t eIndex = rowIndex_old[irow+1];
      for (Int_t index = sIndex; index < eIndex; index++) {
        const Int_t icol = colIndex_old[index];
        if (icol+colLwb_old <= col_upb && icol+colLwb_old >= col_lwb) {
          rowIndex_new[irow+row_off+1] = nelems_copy+1;
          colIndex_new[nelems_copy]    = icol+col_off;
          elements_new[nelems_copy]    = elements_old[index];
          nelems_copy++;
        }
        if (nelems_copy >= nelems_new) {
          cont = kFALSE;
          break;
        }
      }
    }

    if (rowIndex_old) delete [] (Int_t*)    rowIndex_old;
    if (colIndex_old) delete [] (Int_t*)    colIndex_old;
    if (elements_old) delete [] (Double_t*) elements_old;

    if (nrowIndex_old < fNrowIndex) {
      for (Int_t irow = nrowIndex_old; irow < fNrowIndex; irow++)
        rowIndex_new[irow] = rowIndex_new[nrowIndex_old-1];
    }
  } else {
    const Int_t nelems_new = (nr_nonzeros >= 0) ? nr_nonzeros : 0;
    Allocate(new_nrows,new_ncols,row_lwb,col_lwb,1,nelems_new);
  }
}

//______________________________________________________________________________
void TMatrixDSparse::Use(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                         Int_t nr_nonzeros,Int_t *pRowIndex,Int_t *pColIndex,Double_t *pData)
{
  if (row_upb < row_lwb)
  {
    Error("Use","row_upb=%d < row_lwb=%d",row_upb,row_lwb);
    return;
  }
  if (col_upb < col_lwb)
  {
    Error("Use","col_upb=%d < col_lwb=%d",col_upb,col_lwb);
    return;
  }

  Clear();

  fNrows     = row_upb-row_lwb+1;
  fNcols     = col_upb-col_lwb+1;
  fRowLwb    = row_lwb;
  fColLwb    = col_lwb;
  fNrowIndex = fNrows+1;
  fNelems    = nr_nonzeros;
  fIsOwner   = kFALSE;
  fTol       = DBL_EPSILON;
  
  fElements  = pData;
  fRowIndex  = pRowIndex;
  fColIndex  = pColIndex;
}

//______________________________________________________________________________
TMatrixDSparse TMatrixDSparse::GetSub(Int_t row_lwb,Int_t row_upb,
                                      Int_t col_lwb,Int_t col_upb,Option_t *option) const
{
  // Get submatrix [row_lwb..row_upb][col_lwb..col_upb]; The indexing range of the
  // returned matrix depends on the argument option:
  //
  // option == "S" : return [0..row_upb-row_lwb+1][0..col_upb-col_lwb+1] (default)
  // else          : return [row_lwb..row_upb][col_lwb..col_upb]

  Assert(IsValid());
  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("GetSub","row_lwb out-of-bounds");
    return TMatrixDSparse();
  }
  if (col_lwb < fColLwb || col_lwb > fColLwb+fNcols-1) {
    Error("GetSub","col_lwb out-of-bounds");
    return TMatrixDSparse();
  }
  if (row_upb < fRowLwb || row_upb > fRowLwb+fNrows-1) {
    Error("GetSub","row_upb out-of-bounds");
    return TMatrixDSparse();
  }
  if (col_upb < fColLwb || col_upb > fColLwb+fNcols-1) {
    Error("GetSub","col_upb out-of-bounds");
    return TMatrixDSparse();
  }
  if (row_upb < row_lwb || col_upb < col_lwb) {
    Error("GetSub","row_upb < row_lwb || col_upb < col_lwb");
    return TMatrixDSparse();
  }

  TString opt(option);
  opt.ToUpper();
  const Int_t shift = (opt.Contains("S")) ? 1 : 0;

  Int_t nr_nonzeros = 0;
  Int_t irow;
  for (irow = 0; irow < fNrows; irow++) {
    if (irow+fRowLwb > row_upb || irow+fRowLwb < row_lwb) continue;
    const Int_t sIndex = fRowIndex[irow];
    const Int_t eIndex = fRowIndex[irow+1];
    for (Int_t index = sIndex; index < eIndex; index++) {
      const Int_t icol = fColIndex[index];
      if (icol+fColLwb <= col_upb && icol+fColLwb >= col_lwb)
        nr_nonzeros++;
    }
  }

  const Int_t row_lwb_sub = (shift) ? 0               : row_lwb;
  const Int_t row_upb_sub = (shift) ? row_upb-row_lwb : row_upb;
  const Int_t col_lwb_sub = (shift) ? 0               : col_lwb;
  const Int_t col_upb_sub = (shift) ? col_upb-col_lwb : col_upb;

  TMatrixDSparse sub(row_lwb_sub,row_upb_sub,col_lwb_sub,col_upb_sub);
  sub.SetSparseIndex(nr_nonzeros);

  const Double_t *ep = this->GetMatrixArray();

  Int_t    *rowIndex_sub = sub.GetRowIndexArray();
  Int_t    *colIndex_sub = sub.GetColIndexArray();
  Double_t *ep_sub       = sub.GetMatrixArray();

  Int_t nelems_copy = 0;
  rowIndex_sub[0] = 0;
  const Int_t row_off = fRowLwb-row_lwb;
  const Int_t col_off = fColLwb-col_lwb;
  for (irow = 0; irow < fNrows; irow++) {
    if (irow+fRowLwb > row_upb || irow+fRowLwb < row_lwb) continue;
    const Int_t sIndex = fRowIndex[irow];
    const Int_t eIndex = fRowIndex[irow+1];
    for (Int_t index = sIndex; index < eIndex; index++) {
      const Int_t icol = fColIndex[index];
      if (icol+fColLwb <= col_upb && icol+fColLwb >= col_lwb) {
        rowIndex_sub[irow+row_off+1] = nelems_copy+1;
        colIndex_sub[nelems_copy]    = icol+col_off;
        ep_sub[nelems_copy]          = ep[index];
        nelems_copy++;
      }
    }
  }

  return sub;
}

//______________________________________________________________________________
void TMatrixDSparse::SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixDBase &source)
{
  // Insert matrix source starting at [row_lwb][col_lwb], thereby overwriting the part
  // [row_lwb..row_lwb+nrows_source-1][col_lwb..col_lwb+ncols_source-1];

  Assert(IsValid());
  Assert(source.IsValid());

  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("SetSub","row_lwb out-of-bounds");
    return;
  }
  if (col_lwb < fColLwb || col_lwb > fColLwb+fNcols-1) {
    Error("SetSub","col_lwb out-of-bounds");
    return;
  }
  const Int_t nRows_source = source.GetNrows();
  const Int_t nCols_source = source.GetNcols();
  if (row_lwb+nRows_source > fRowLwb+fNrows || col_lwb+nCols_source > fColLwb+fNcols) {
    Error("SetSub","source matrix too large");
    return;
  }

  // Determine how many non-zero's are already available in
  // [row_lwb..row_lwb+nrows_source-1][col_lwb..col_lwb+ncols_source-1]
  Int_t nr_nonzeros = 0;
  Int_t irow,index;
  for (irow = 0; irow < fNrows; irow++) {
    if (irow+fRowLwb >= row_lwb+nRows_source || irow+fRowLwb < row_lwb) continue;
    const Int_t sIndex = fRowIndex[irow];
    const Int_t eIndex = fRowIndex[irow+1];
    for (Int_t index = sIndex; index < eIndex; index++) {
      const Int_t icol = fColIndex[index];
      if (icol+fColLwb < col_lwb+nCols_source && icol+fColLwb >= col_lwb)
        nr_nonzeros++;
    }
  }

  const Int_t    *rowIndex_s = source.GetRowIndexArray();
  const Int_t    *colIndex_s = source.GetColIndexArray();
  const Double_t *elements_s = source.GetMatrixArray();

  const Int_t nelems_old = fNelems;
  const Int_t    *rowIndex_old = GetRowIndexArray();
  const Int_t    *colIndex_old = GetColIndexArray();
  const Double_t *elements_old = GetMatrixArray();

  const Int_t nelems_new = nelems_old+source.GetNoElements()-nr_nonzeros;
  fRowIndex = new Int_t[fNrowIndex];
  fColIndex = new Int_t[nelems_new];
  fElements = new Double_t[nelems_new];
  fNelems   = nelems_new;

  Int_t    *rowIndex_new = GetRowIndexArray();
  Int_t    *colIndex_new = GetColIndexArray();
  Double_t *elements_new = GetMatrixArray();

  const Int_t row_off = row_lwb-fRowLwb;
  const Int_t col_off = col_lwb-fColLwb;

  Int_t nr = 0;
  rowIndex_new[0] = 0;
  for (irow = 0; irow < fNrows; irow++) {
    rowIndex_new[irow+1] = rowIndex_new[irow];
    Bool_t flagRow = kFALSE;
    if (irow+fRowLwb < row_lwb+nRows_source && irow+fRowLwb >= row_lwb)
      flagRow = kTRUE;

    const Int_t sIndex_o = rowIndex_old[irow];
    const Int_t eIndex_o = rowIndex_old[irow+1];

    if (flagRow) {
      const Int_t icol_left = col_off-1;
      const Int_t left = TMath::BinarySearch(eIndex_o-sIndex_o,colIndex_old+sIndex_o,icol_left)+sIndex_o;
      for (index = sIndex_o; index <= left; index++) {
        rowIndex_new[irow+1]++;
        colIndex_new[nr] = colIndex_old[index];
        elements_new[nr] = elements_old[index];
        nr++;
      }

      const Int_t sIndex_s = rowIndex_s[irow-row_off];
      const Int_t eIndex_s = rowIndex_s[irow-row_off+1];
      for (index = sIndex_s; index < eIndex_s; index++) {
        rowIndex_new[irow+1]++;
        colIndex_new[nr] = colIndex_s[index]+col_off;
        elements_new[nr] = elements_s[index];
        nr++;
      }

      const Int_t icol_right = col_off+nCols_source-1;
      if (colIndex_old) {
        Int_t right = TMath::BinarySearch(eIndex_o-sIndex_o,colIndex_old+sIndex_o,icol_right)+sIndex_o;
        while (right < eIndex_o && colIndex_old[right+1] <= icol_right)
          right++;
        right++;

        for (index = right; index < eIndex_o; index++) {
          rowIndex_new[irow+1]++;
          colIndex_new[nr] = colIndex_old[index];
          elements_new[nr] = elements_old[index];
          nr++;
        }
      }
    } else {
      for (index = sIndex_o; index < eIndex_o; index++) {
        rowIndex_new[irow+1]++;
        colIndex_new[nr] = colIndex_old[index];
        elements_new[nr] = elements_old[index];
        nr++;
      }
    }
  }

  Assert(fNelems == fRowIndex[fNrowIndex-1]);

  if (rowIndex_old) delete [] (Int_t*)    rowIndex_old;
  if (colIndex_old) delete [] (Int_t*)    colIndex_old;
  if (elements_old) delete [] (Double_t*) elements_old;
}

//______________________________________________________________________________
TMatrixDSparse &TMatrixDSparse::Transpose(const TMatrixDSparse &source)
{
  // Transpose a matrix.

  Assert(IsValid());
  Assert(source.IsValid());

  if (fNrows  != source.GetNcols()  || fNcols  != source.GetNrows() ||
      fRowLwb != source.GetColLwb() || fColLwb != source.GetRowLwb())
  {
    Error("Transpose","matrix has wrong shape");
    Invalidate();
    return *this;
  }

  const Int_t nr_nonzeros = source.NonZeros();
  if (nr_nonzeros <= 0)
    return *this;

  const Int_t    * const pRowIndex_s = source.GetRowIndexArray();
  const Int_t    * const pColIndex_s = source.GetColIndexArray();
  const Double_t * const pData_s     = source.GetMatrixArray();

  Int_t    *rownr   = new Int_t[nr_nonzeros];
  Int_t    *colnr   = new Int_t[nr_nonzeros];
  Double_t *pData_t = new Double_t[nr_nonzeros];

  Int_t ielem = 0;
  for (Int_t irow_s = 0; irow_s < source.GetNrows(); irow_s++) {
    const Int_t sIndex = pRowIndex_s[irow_s];
    const Int_t eIndex = pRowIndex_s[irow_s+1];
    for (Int_t index = sIndex; index < eIndex; index++) {
      if (pData_s[index] != 0.0) {
        rownr[ielem]       = pColIndex_s[index]+fRowLwb;
        colnr[ielem]       = irow_s+fColLwb;
        pData_t[ielem]     = pData_s[index];
        ielem++;
      }
    }
  }

  Assert(nr_nonzeros == ielem);

  DoubleLexSort(nr_nonzeros,rownr,colnr,pData_t);
  SetMatrixArray(nr_nonzeros,rownr,colnr,pData_t);

  Assert(fNelems == fRowIndex[fNrowIndex-1]);

  if (pData_t) delete [] pData_t;
  if (rownr)   delete [] rownr;
  if (colnr)   delete [] colnr;

  return *this;
}

//______________________________________________________________________________
TMatrixDBase &TMatrixDSparse::Zero()
{
  Assert(IsValid());

  if (fElements) { delete [] fElements; fElements = 0; }
  if (fColIndex) { delete [] fColIndex; fColIndex = 0; }
  fNelems = 0;
  memset(this->GetRowIndexArray(),0,fNrowIndex*sizeof(Int_t));

  return *this;
}

//______________________________________________________________________________
TMatrixDBase &TMatrixDSparse::UnitMatrix()
{
  // Make a unit matrix (matrix need not be a square one).

  Assert(IsValid());

  Int_t i;

  Int_t nr_nonzeros = 0;
  for (i = fRowLwb; i <= fRowLwb+fNrows-1; i++)
    for (Int_t j = fColLwb; j <= fColLwb+fNcols-1; j++)
      if (i == j) nr_nonzeros++;

  if (nr_nonzeros != fNelems) {
    fNelems = nr_nonzeros;
    Int_t *oIp = fColIndex;
    fColIndex = new Int_t[nr_nonzeros];
    if (oIp) delete [] oIp;
    Double_t *oDp = fElements;
    fElements = new Double_t[nr_nonzeros];
    if (oDp) delete [] oDp;
  }

  Int_t ielem = 0;
  fRowIndex[0] = 0;
  for (i = fRowLwb; i <= fRowLwb+fNrows-1; i++) {
    for (Int_t j = fColLwb; j <= fColLwb+fNcols-1; j++) {
      if (i == j) {
        const Int_t irow = i-fRowLwb;
        fRowIndex[irow+1]  = ielem+1;
        fElements[ielem]   = 1.0;
        fColIndex[ielem++] = j-fColLwb;
      }
    }
  }

  return *this;
}

//______________________________________________________________________________
Double_t TMatrixDSparse::RowNorm() const
{
  // Row matrix norm, MAX{ SUM{ |M(i,j)|, over j}, over i}.                     
  // The norm is induced by the infinity vector norm.

  Assert(IsValid());

  const Double_t *       ep = GetMatrixArray();
  const Double_t * const fp = ep+fNelems;
  const Int_t    * const pR = GetRowIndexArray();
        Double_t norm = 0;
            
  // Scan the matrix row-after-row
  for (Int_t irow = 0; irow < fNrows; irow++) {
    const Int_t sIndex = pR[irow];
    const Int_t eIndex = pR[irow+1];
    Double_t sum = 0;
    for (Int_t index = sIndex; index < eIndex; index++)
      sum += TMath::Abs(*ep++);
    norm = TMath::Max(norm,sum);
  }

  Assert(ep == fp); 

  return norm;
}

//______________________________________________________________________________
Double_t TMatrixDSparse::ColNorm() const
{
  // Column matrix norm, MAX{ SUM{ |M(i,j)|, over i}, over j}.
  // The norm is induced by the 1 vector norm.

  Assert(IsValid());

  const TMatrixDSparse mt(kTransposed,*this);

  const Double_t *       ep = mt.GetMatrixArray();
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
Double_t &TMatrixDSparse::operator()(Int_t rown,Int_t coln)
{
  Assert(IsValid());

  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb; 
  Assert(arown < fNrows && arown >= 0);
  Assert(acoln < fNcols && acoln >= 0);

  Int_t index = -1;
  Int_t sIndex = 0;
  Int_t eIndex = 0;
  if (fNrowIndex > 0 && fRowIndex[fNrowIndex-1] != 0) {
    sIndex = fRowIndex[arown];
    eIndex = fRowIndex[arown+1];
    index = TMath::BinarySearch(eIndex-sIndex,fColIndex+sIndex,acoln)+sIndex;
  }

  if (index >= sIndex && fColIndex[index] == acoln)
    return fElements[index];
  else {
    Double_t val = 0.;
    InsertRow(rown,coln,&val,1);
    sIndex = fRowIndex[arown];
    eIndex = fRowIndex[arown+1];
    index = TMath::BinarySearch(eIndex-sIndex,fColIndex+sIndex,acoln)+sIndex;
    if (index >= sIndex && fColIndex[index] == acoln)
      return fElements[index];
    else {
      Error("operator()(Int_t,Int_t","Insert row failed");
      Assert(0);
      return fElements[0];
    }
  }
}

//______________________________________________________________________________
TMatrixDSparse &TMatrixDSparse::operator=(const TMatrixDSparse &source)
{
  // Notice that the sparsity of the matrix is NOT changed : its fRowIndex/fColIndex
  // are used !

  if (!AreCompatible(*this,source)) {
    Error("operator=(const TMatrixDSparse &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  if (this != &source) {
    TObject::operator=(source);

    const Double_t * const sp = source.GetMatrixArray();
          Double_t * const tp = this->GetMatrixArray();
    memcpy(tp,sp,fNelems*sizeof(Double_t));
    fTol = source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixDSparse &TMatrixDSparse::operator=(const TMatrixD &source)
{
  // Notice that the sparsity of the matrix is NOT changed : its fRowIndex/fColIndex
  // are used !

  if (!AreCompatible(*this,(TMatrixDBase &)source)) {
    Error("operator=(const TMatrixD &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  if (this != (TMatrixDSparse *)&source) {
    TObject::operator=(source);

    const Double_t * const sp = source.GetMatrixArray();
          Double_t * const tp = this->GetMatrixArray();
    for (Int_t irow = 0; irow < fNrows; irow++) {
      const Int_t sIndex = fRowIndex[irow];
      const Int_t eIndex = fRowIndex[irow+1];
      const Int_t off = irow*fNcols;
      for (Int_t index = sIndex; index < eIndex; index++) {
        const Int_t icol = fColIndex[index];
        tp[index] = sp[off+icol];
      }
    }
    fTol = source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixDSparse &TMatrixDSparse::operator=(Double_t val)
{
  // Assign val to every element of the matrix. Check that the row/col
  // indices are set !

  Assert(IsValid());

  if (fRowIndex[fNrowIndex-1] == 0) {
    Error("operator=(Double_t","row/col indices are not set");
    Invalidate();
    return *this;
  }

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ = val;

  return *this;
}

//______________________________________________________________________________
TMatrixDSparse &TMatrixDSparse::operator+=(Double_t val)
{
  // Add val to every element of the matrix.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ += val;

  return *this;
}

//______________________________________________________________________________
TMatrixDSparse &TMatrixDSparse::operator-=(Double_t val)
{
  // Subtract val from every element of the matrix.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ -= val;

  return *this;
}

//______________________________________________________________________________
TMatrixDSparse &TMatrixDSparse::operator*=(Double_t val)
{
  // Multiply every element of the matrix with val.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ *= val;

  return *this;
}

//______________________________________________________________________________
void TMatrixDSparse::Randomize(Double_t alpha,Double_t beta,Double_t &seed)
{
  // randomize matrix element values

  Assert(IsValid());

  const Double_t scale = beta-alpha;
  const Double_t shift = alpha/scale;

  Int_t    * const pRowIndex = GetRowIndexArray();
  Int_t    * const pColIndex = GetColIndexArray();
  Double_t * const ep        = GetMatrixArray();

  const Int_t m = GetNrows();
  const Int_t n = GetNcols();

  // Knuth's algorithm for choosing "length" elements out of NN .
  const Int_t NN     = GetNrows()*GetNcols();
  const Int_t length = (GetNoElements() <= NN) ? GetNoElements() : NN;
  Int_t chosen   = 0;
  Int_t icurrent = 0;
  pRowIndex[0] = 0;
  for (Int_t k = 0; k < NN; k++) {
    const Double_t r = Drand(seed);

    if ((NN-k)*r < length-chosen) {
      pColIndex[chosen] = k%n;
      const Int_t irow  = k/n;

      if (irow > icurrent) {
        for ( ; icurrent < irow; icurrent++)
          pRowIndex[icurrent+1] = chosen;
      }
      ep[chosen] = scale*(Drand(seed)+shift);
      chosen++;
    }  
  }
  for ( ; icurrent < m; icurrent++)
    pRowIndex[icurrent+1] = length;

  Assert(chosen == length);
}

//______________________________________________________________________________
void TMatrixDSparse::RandomizePD(Double_t alpha,Double_t beta,Double_t &seed)
{
  // randomize matrix element values but keep matrix symmetric positive definite

  Assert(IsValid());
  
  const Double_t scale = beta-alpha;
  const Double_t shift = alpha/scale;

  if (fNrows != fNcols || fRowLwb != fColLwb) {
    Error("RandomizePD(Double_t &","matrix should be square");
    return;
  }

  const Int_t n = fNcols;

  Int_t    * const pRowIndex = GetRowIndexArray();
  Int_t    * const pColIndex = GetColIndexArray();
  Double_t * const ep        = GetMatrixArray();

  // We will always have non-zeros on the diagonal, so there
  // is no randomness there. In fact, choose the (0,0) element now
  pRowIndex[0] = 0;
  pColIndex[0] = 0;
  pRowIndex[1] = 1;
  ep[0]        = 1e-8+scale*(Drand(seed)+shift);

  // Knuth's algorithm for choosing length elements out of NN .
  // NN here is the number of elements in the strict lower triangle.
  const Int_t NN = n*(n-1)/2;

  // length is the number of elements that can be stored, minus the number
  // of elements in the diagonal, which will always be in the matrix.
//  Int_t length = (fNelems-n)/2;
  Int_t length = fNelems-n;
  length = (length <= NN) ? length : NN;

  // chosen   : the number of elements that have already been chosen (now 0)
  // nnz      : the number of non-zeros in the matrix (now 1, because the
  //            (0,0) element is already in the matrix.
  // icurrent : the index of the last row whose start has been stored in pRowIndex;

  Int_t chosen   = 0;
  Int_t icurrent = 1;
  Int_t nnz      = 1;
  for (Int_t k = 0; k < NN; k++ ) {
    const Double_t r = Drand(seed);

    if( (NN-k)*r < length-chosen) {
      // Element k is chosen. What row is it in?
      // In a lower triangular matrix (including a diagonal), it will be in
      // the largest row such that row*(row+1)/2 < k. In other words

      Int_t row = (int) TMath::Floor((-1+TMath::Sqrt(1.0+8.0*k))/2);
      // and its column will be the remainder
      Int_t col = k-row*(row+1)/2;
      // but since we are only filling in the *strict* lower triangle of
      // the matrix, we shift the row by 1
      row++;

      if (row > icurrent) {
        // We have chosen a row beyond the current row.
        // Choose a diagonal element for each intermediate row and fix the
        // data structure.
        for ( ; icurrent < row; icurrent++) {
          // Choose the diagonal
          ep[nnz] = 0.0;
          for (Int_t ll = pRowIndex[icurrent]; ll < nnz; ll++)
            ep[nnz] += TMath::Abs(ep[ll]);
          ep[nnz] +=  1e-8+scale*(Drand(seed)+shift);
          pColIndex[nnz] = icurrent;

          nnz++;
          pRowIndex[icurrent+1] = nnz;
        }
      } // end if we have chosen a row beyond the current row;
      ep[nnz] = scale*(Drand(seed)+shift);
      pColIndex[nnz] = col;
      // add the value of this element (which occurs symmetrically in the
      // upper triangle) to the appropriate diagonal element
      ep[pRowIndex[col+1]-1] += TMath::Abs(ep[nnz]);

      nnz++; // We have added another element to the matrix
      chosen++; // And finished choosing another element.
    }  
  }

  Assert(chosen == length);

  // and of course, we must choose all remaining diagonal elements .
  for ( ; icurrent < n; icurrent++) {
    // Choose the diagonal
    ep[nnz] = 0.0;
    for(Int_t ll = pRowIndex[icurrent]; ll < nnz; ll++)
      ep[nnz] += TMath::Abs(ep[ll]);
    ep[nnz] += 1e-8+scale*(Drand(seed)+shift);
    pColIndex[nnz] = icurrent;

    nnz++;
    pRowIndex[icurrent+1] = nnz;
  }
  fNelems = nnz;

  TMatrixDSparse tmp(TMatrixDSparse::kTransposed,*this);
  *this += tmp;

  // make sure to divide the diagonal by 2 becuase the operation
  // *this += tmp; adds the diagonal again
  {
    const Int_t    * const pR = GetRowIndexArray();
    const Int_t    * const pC = GetColIndexArray();
          Double_t * const pD = GetMatrixArray();
    for (Int_t irow = 0; irow < fNrows+1; irow++) {
      const Int_t sIndex = pR[irow];
      const Int_t eIndex = pR[irow+1];
      for (Int_t index = sIndex; index < eIndex; index++) {
        const Int_t icol = pC[index];
        if (irow == icol)
          pD[index] /= 2.;
      }
    }
  }
}

//______________________________________________________________________________
TMatrixDSparse operator+(const TMatrixDSparse &source1,const TMatrixDSparse &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kPlus,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator+(const TMatrixDSparse &source1,const TMatrixD &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kPlus,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator+(const TMatrixD &source1,const TMatrixDSparse &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kPlus,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator+(const TMatrixDSparse &source,Double_t val)
{
  TMatrixDSparse target(source);
  target += val;
  return target;
} 

//______________________________________________________________________________
TMatrixDSparse operator+(Double_t val,const TMatrixDSparse &source)
{
  TMatrixDSparse target(source);
  target += val;
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator-(const TMatrixDSparse &source1,const TMatrixDSparse &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kMinus,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator-(const TMatrixDSparse &source1,const TMatrixD &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kMinus,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator-(const TMatrixD &source1,const TMatrixDSparse &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kMinus,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator-(const TMatrixDSparse &source,Double_t val)
{
  TMatrixDSparse target(source);
  target -= val;
  return target;
} 

//______________________________________________________________________________
TMatrixDSparse operator-(Double_t val,const TMatrixDSparse &source)
{
  TMatrixDSparse target(source);
  target -= val;
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator*(const TMatrixDSparse &source1,const TMatrixDSparse &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator*(const TMatrixDSparse &source1,const TMatrixD &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator*(const TMatrixD &source1,const TMatrixDSparse &source2)
{
  TMatrixDSparse target(source1,TMatrixDSparse::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator*(Double_t val,const TMatrixDSparse &source)
{
  TMatrixDSparse target(source);
  target *= val;
  return target;
}

//______________________________________________________________________________
TMatrixDSparse operator*(const TMatrixDSparse &source,Double_t val)
{
  TMatrixDSparse target(source);
  target *= val;
  return target;
}

//______________________________________________________________________________
TMatrixDSparse &Add(TMatrixDSparse &target,Double_t scalar,const TMatrixDSparse &source)
{
  // Modify addition: target += scalar * source.

  target += scalar * source;
  return target;
}

//______________________________________________________________________________
TMatrixDSparse &ElementMult(TMatrixDSparse &target,const TMatrixDSparse &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementMult(TMatrixDSparse &,const TMatrixDSparse &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Double_t *sp  = source.GetMatrixArray();
        Double_t *tp  = target.GetMatrixArray();
  const Double_t *ftp = tp+target.GetNoElements();
  while ( tp < ftp )
    *tp++ *= *sp++;

  return target;
}

//______________________________________________________________________________
TMatrixDSparse &ElementDiv (TMatrixDSparse &target,const TMatrixDSparse &source)
{
  // Divide target by the source, element-by-element. 
    
  if (!AreCompatible(target,source)) {
    ::Error("ElementDiv(TMatrixD &,const TMatrixD &)","matrices not compatible");
    target.Invalidate();
    return target;
  }
  
  const Double_t *sp  = source.GetMatrixArray();
        Double_t *tp  = target.GetMatrixArray();
  const Double_t *ftp = tp+target.GetNoElements();
  while ( tp < ftp ) {
    Assert(*sp != 0.0);
    *tp++ /= *sp++;
  } 
    
  return target;
}

//______________________________________________________________________________
Bool_t AreCompatible(const TMatrixDSparse &m1,const TMatrixDSparse &m2,Int_t verbose)
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

  const Int_t *pR1 = m1.GetRowIndexArray();
  const Int_t *pR2 = m2.GetRowIndexArray();
  const Int_t nRows = m1.GetNrows();
  if (memcmp(pR1,pR2,(nRows+1)*sizeof(Int_t))) {
    if (verbose)
      ::Error("AreCompatible", "matrices 1 and 2 have different rowIndex");
    for (Int_t i = 0; i < nRows+1; i++)
      printf("%d: %d %d\n",i,pR1[i],pR2[i]);
    return kFALSE;
  }
  const Int_t *pD1 = m1.GetColIndexArray();
  const Int_t *pD2 = m2.GetColIndexArray();
  const Int_t nData = m1.GetNoElements();
  if (memcmp(pD1,pD2,nData*sizeof(Int_t))) {
    if (verbose)
      ::Error("AreCompatible", "matrices 1 and 2 have different colIndex");
    for (Int_t i = 0; i < nData; i++)
      printf("%d: %d %d\n",i,pD1[i],pD2[i]);
    return kFALSE;
  }

  return kTRUE;
}
