// @(#)root/matrix:$Name:  $:$Id: TMatrixDSparse.cxx,v 1.3 2004/05/12 13:56:37 brun Exp $
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
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDSparse.h"
#include "Riostream.h"

ClassImp(TMatrixDSparse)

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(Int_t no_rows,Int_t no_cols,Int_t nr_nonzeros)
{
  // Space is allocated for row/column indices and data, but the sparse structure
  // information has still to be set !

  Allocate(no_rows,no_cols,0,0,1,nr_nonzeros);
}

//______________________________________________________________________________
TMatrixDSparse::TMatrixDSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                               Int_t nr_nonzeros)
{
  // Space is allocated for row/column indices and data, but the sparse structure
  // information has still to be set !

  Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb,1,nr_nonzeros);
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

  SetMatrixArray(row,col,data);
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
  const Double_t *ep = another.GetMatrixArray();
  const Double_t * const fp = another.GetMatrixArray()+another.GetNoElements();
  Int_t nr_nonzeros = 0;
  while (ep < fp)
    if (*ep++ != 0.0) nr_nonzeros++;

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
  fJunk      = 0.0;
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
void TMatrixDSparse::Trim(Int_t nelems_new)
{
  // Increase/decrease the number of non-zero elements to nelems_new

  if (nelems_new != fNelems) {
    const Int_t nr = TMath::Min(nelems_new,fNelems);
    const Int_t * const oIp = fColIndex;
    fColIndex = new Int_t[nelems_new];
    memmove(fColIndex,oIp,nr*sizeof(Int_t));
    if (oIp) delete [] oIp;
    const Double_t * const oDp = fElements;
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
      while (indexb < eIndexb && pColIndexb[indexb++] < icola)
        pColIndexc[nc++] = pColIndexb[indexb-1];
      pColIndexc[nc++] = pColIndexa[indexa];
    }
    pRowIndexc[irowc+1] = nc;
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
    Trim(indexc_r);
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
    Trim(indexc_r);
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
    Trim(indexc_r);
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
        while (indexb < eIndexb && pColIndexb[indexb++] < icola) 
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
    Trim(indexc_r);
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
    Trim(indexc_r);
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
        while (indexb < eIndexb && pColIndexb[indexb++] < icola) 
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
    Trim(indexc_r);
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
    Trim(indexc_r);
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
    Trim(indexc_r);
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
void TMatrixDSparse::SetMatrixArray(Int_t *row,Int_t *col,Double_t *data)
{
  // Copy row/col index and data array to matrix . It is assumed that arrays are of
  //  size >= fNelems

  Assert(IsValid());
  if (fNelems <= 0) return;

  const Int_t irowmin = TMath::LocMin(fNelems,row);
  const Int_t irowmax = TMath::LocMax(fNelems,row);
  const Int_t icolmin = TMath::LocMin(fNelems,col);
  const Int_t icolmax = TMath::LocMax(fNelems,col);

  Assert(row[irowmin] >= fRowLwb && row[irowmax] <= fRowLwb+fNrows-1);
  Assert(col[icolmin] >= fColLwb && col[icolmax] <= fColLwb+fNcols-1);

  DoubleLexSort(fNelems,row,col,data);

  Int_t nr_nonzeros = 0;
  const Int_t nr = fNelems;
  const Double_t *ep        = data;
  const Double_t * const fp = data+nr;

  while (ep < fp)
    if (*ep++ != 0.0) nr_nonzeros++;

  // if nr_nonzeros != fNelems => nr_nonzeros < fNelems !
  if (nr_nonzeros != fNelems) {
    delete [] fColIndex;
    delete [] fElements;
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

  const Double_t *ep = source.GetMatrixArray();
  const Double_t * const fp = source.GetMatrixArray()+source.GetNoElements();
  Int_t nr_nonzeros = 0;
  while (ep < fp)
    if (*ep++ != 0.0) nr_nonzeros++;

  if (nr_nonzeros != fNelems)
    Trim(nr_nonzeros);

  ep = source.GetMatrixArray();
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
  if (nelem_s != fNelems)
    Trim(nelem_s);

  memmove(fRowIndex,source.GetRowIndexArray(),fNrowIndex*sizeof(Int_t));
  memmove(fColIndex,source.GetColIndexArray(),fNelems*sizeof(Int_t));
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

    const Double_t * const elements_old = GetMatrixArray();
    const Int_t    * const rowIndex_old = GetRowIndexArray();
    const Int_t    * const colIndex_old = GetColIndexArray();

    const Int_t nrows_old     = fNrows;
    const Int_t nrowIndex_old = fNrowIndex;

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

    Double_t * const elements_new = GetMatrixArray();
    Int_t    * const rowIndex_new = GetRowIndexArray();
    Int_t    * const colIndex_new = GetColIndexArray();

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

    if (rowIndex_old) delete [] rowIndex_old;
    if (colIndex_old) delete [] colIndex_old;
    if (elements_old) delete [] elements_old;

    if (nrowIndex_old < fNrowIndex) {
      for (Int_t irow = nrowIndex_old; irow < fNrowIndex; irow++)
        rowIndex_new[irow] = rowIndex_new[nrowIndex_old-1];
    }
  } else {
    Allocate(nrows,ncols,0,0,1,nr_nonzeros);
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

    const Int_t    * const rowIndex_old = GetRowIndexArray();
    const Int_t    * const colIndex_old = GetColIndexArray();
    const Double_t * const elements_old = GetMatrixArray();

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

    Int_t    * const rowIndex_new = GetRowIndexArray();
    Int_t    * const colIndex_new = GetColIndexArray();
    Double_t * const elements_new = GetMatrixArray();

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

    if (rowIndex_old) delete [] rowIndex_old;
    if (colIndex_old) delete [] colIndex_old;
    if (elements_old) delete [] elements_old;

    if (nrowIndex_old < fNrowIndex) {
      for (Int_t irow = nrowIndex_old; irow < fNrowIndex; irow++)
        rowIndex_new[irow] = rowIndex_new[nrowIndex_old-1];
    }
  } else {
    Allocate(new_nrows,new_ncols,row_lwb,col_lwb,1,nr_nonzeros);
  }
}

//______________________________________________________________________________
void TMatrixDSparse::Use(TMatrixDSparse &a)
{
  Assert(IsValid());

  Clear();

  fNrows     = a.GetNrows();
  fNcols     = a.GetNcols();
  fRowLwb    = a.GetRowLwb();
  fColLwb    = a.GetColLwb();
  fNrowIndex = a.GetNrows()+1;
  fNelems    = a.GetNoElements();
  fJunk      = 0.0;
  fIsOwner   = kFALSE;
  fTol       = DBL_EPSILON;
  
  fElements  = a.GetMatrixArray();
  fRowIndex  = a.GetRowIndexArray();
  fColIndex  = a.GetColIndexArray();
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

  TMatrixDSparse sub(row_lwb_sub,row_upb_sub,col_lwb_sub,col_upb_sub,nr_nonzeros);

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
void TMatrixDSparse::SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixDSparse &source)
{
  // Insert matrix source starting at [row_lwb][col_lwb], thereby overwriting the part
  // [row_lwb..row_lwb+nrows_source][col_lwb..col_lwb+ncols_source];

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

  const Int_t    * const rowIndex_s = source.GetRowIndexArray();
  const Int_t    * const colIndex_s = source.GetColIndexArray();
  const Double_t * const elements_s = source.GetMatrixArray();

  const Int_t nelems_old = fNelems;
  const Int_t    * const rowIndex_old = GetRowIndexArray();
  const Int_t    * const colIndex_old = GetColIndexArray();
  const Double_t * const elements_old = GetMatrixArray();

  const Int_t nelems_new = nelems_old+source.GetNoElements()-nr_nonzeros;
  Int_t    * const rowIndex_new = new Int_t[fNrowIndex];
  Int_t    * const colIndex_new = new Int_t[nelems_new];
  Double_t * const elements_new = new Double_t[nelems_new];

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
      const Int_t icol_left = col_lwb-fColLwb;
      const Int_t left = TMath::BinarySearch(eIndex_o-sIndex_o,colIndex_old+sIndex_o,icol_left)+sIndex_o;
      for (index = sIndex_o; index < left; index++) {
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

      const Int_t icol_right = col_lwb+nCols_source-fColLwb;
      Int_t right = TMath::Min(TMath::BinarySearch(eIndex_o-sIndex_o,colIndex_old+sIndex_o,
                                                   icol_right)+sIndex_o,sIndex_o);
      while (right < eIndex_o && colIndex_old[right+1] < icol_right)
        right++;
      for (index = right; index < eIndex_o; index++) {
        rowIndex_new[irow+1]++;
        colIndex_new[nr] = colIndex_old[index];
        elements_new[nr] = elements_old[index];
        nr++;
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

  delete [] rowIndex_old;
  delete [] colIndex_old;
  delete [] elements_old;
  fRowIndex = rowIndex_new;
  fColIndex = colIndex_new;
  fElements = elements_new;
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

  const Int_t nr_nonzeros = source.GetNoElements();
  if (nr_nonzeros <= 0)
    return *this;

  const Int_t * const pRowIndex_s = source.GetRowIndexArray();
  const Int_t * const pColIndex_s = source.GetColIndexArray();
  Double_t * const pData = new Double_t[nr_nonzeros];
  memmove(pData,source.GetMatrixArray(),nr_nonzeros*sizeof(Double_t));

  Int_t * const pColIndex_t = new Int_t[nr_nonzeros];
  Int_t * const rownr       = new Int_t[nr_nonzeros];

  Int_t ielem = 0;
  for (Int_t irow_s = 0; irow_s < source.GetNrows(); irow_s++) {
    const Int_t sIndex = pRowIndex_s[irow_s];
    const Int_t eIndex = pRowIndex_s[irow_s+1];
    for (Int_t index = sIndex; index < eIndex; index++) {
      rownr[ielem]       = pColIndex_s[index];
      pColIndex_t[ielem] = irow_s;
      ielem++;
    }
  }

  DoubleLexSort(nr_nonzeros,rownr,pColIndex_t,pData);

  Int_t *pRowIndex_t = new Int_t[fNrows+1];
  pRowIndex_t[0] = 0;
  ielem = 0;
  for (Int_t irow = 1; irow < fNrows+1; irow++) {
    while (ielem < nr_nonzeros) {
      ielem++;
      if (ielem >= nr_nonzeros || rownr[ielem] != rownr[ielem-1])
        break;
    }
    pRowIndex_t[irow] = ielem;
  }

  if (rownr)
    delete [] rownr;

  if (this == &source) {
    if (pRowIndex_s) delete [] pRowIndex_s;
    if (pColIndex_s) delete [] pColIndex_s;
    if (fElements)   delete [] fElements;
  }

  fRowIndex = pRowIndex_t;
  fColIndex = pColIndex_t;
  fElements = pData;

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

  if (memcmp(m1.GetRowIndexArray(),m2.GetRowIndexArray(),(m1.GetNrows()+1)*sizeof(Int_t))) {
    if (verbose)
      ::Error("AreCompatible", "matrices 1 and 2 have different rowIndex");
    const Int_t *p1 = m1.GetRowIndexArray();
    const Int_t *p2 = m2.GetRowIndexArray();
    for (Int_t i = 0; i < m1.GetNrows()+1; i++)
      printf("%d: %d %d\n",i,p1[i],p2[i]);
    return kFALSE;
  }
  if (memcmp(m1.GetColIndexArray(),m2.GetColIndexArray(),m1.GetNoElements()*sizeof(Int_t))) {
    if (verbose)
      ::Error("AreCompatible", "matrices 1 and 2 have different colIndex");
    const Int_t *p1 = m1.GetColIndexArray();
    const Int_t *p2 = m2.GetColIndexArray();
    for (Int_t i = 0; i < m1.GetNoElements(); i++)
      printf("%d: %d %d\n",i,p1[i],p2[i]);
    return kFALSE;
  }

  return kTRUE;
}
