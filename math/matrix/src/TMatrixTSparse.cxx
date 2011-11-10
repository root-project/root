// @(#)root/matrix:$Id$
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
// TMatrixTSparse                                                       //
//                                                                      //
// Template class of a general sparse matrix in the Harwell-Boeing      //
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
// for (Int_t irow = 0; irow < this->fNrows; irow++) {                  //
//   const Int_t sIndex = fRowIndex[irow];                              //
//   const Int_t eIndex = fRowIndex[irow+1];                            //
//   for (Int_t index = sIndex; index < eIndex; index++) {              //
//     const Int_t icol = fColIndex[index];                             //
//     const Element data = fElements[index];                           //
//     printf("data(%d,%d) = %.4e\n",irow+this->fRowLwb,icol+           //
//                                               this->fColLwb,data);   //
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
//    TMatrixTSparse(Int_t row_lwb,Int_t row_upb,Int_t dol_lwb,         //
//                   Int_t col_upb,Int_t nr_nonzeros,                   //
//                   Int_t *row, Int_t *col,Element *data);            //
//    It uses SetMatrixArray(..), see below                             //
// 2. copy constructors                                                 //
// 3. SetMatrixArray(Int_t nr,Int_t *irow,Int_t *icol,Element *data)   //
//    where it is expected that the irow,icol and data array contain    //
//    nr entries . Only the entries with non-zero data[i] value are     //
//    inserted. Be aware that the input data array will be modified     //
//    inside the routine for doing the necessary sorting of indices !   //
// 4. TMatrixTSparse a(n,m); for(....) { a(i,j) = ....                  //
//    This is a very flexible method but expensive :                    //
//    - if no entry for slot (i,j) is found in the sparse index table   //
//      it will be entered, which involves some memory management !     //
//    - before invoking this method in a loop it is smart to first      //
//      set the index table through a call to SetSparseIndex(..)        //
// 5. SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixTBase &source)    //
//    the matrix to be inserted at position (row_lwb,col_lwb) can be    //
//    both dense or sparse .                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixTSparse.h"
#include "TMatrixT.h"
#include "TMath.h"

#ifndef R__ALPHA
templateClassImp(TMatrixTSparse)
#endif

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(Int_t no_rows,Int_t no_cols)
{
  // Space is allocated for row/column indices and data, but the sparse structure
  // information has still to be set !

   Allocate(no_rows,no_cols,0,0,1);
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
{
  // Space is allocated for row/column indices and data, but the sparse structure
  // information has still to be set !

   Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb,1);
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                        Int_t nr,Int_t *row, Int_t *col,Element *data)
{
  // Space is allocated for row/column indices and data. Sparse row/column index
  // structure together with data is coming from the arrays, row, col and data, resp .

   const Int_t irowmin = TMath::LocMin(nr,row);
   const Int_t irowmax = TMath::LocMax(nr,row);
   const Int_t icolmin = TMath::LocMin(nr,col);
   const Int_t icolmax = TMath::LocMax(nr,col);

   if (row[irowmin] < row_lwb || row[irowmax] > row_upb) {
      Error("TMatrixTSparse","Inconsistency between row index and its range");
      if (row[irowmin] < row_lwb) {
         Info("TMatrixTSparse","row index lower bound adjusted to %d",row[irowmin]);
         row_lwb = row[irowmin];
      }
      if (row[irowmax] > row_upb) {
         Info("TMatrixTSparse","row index upper bound adjusted to %d",row[irowmax]);
         col_lwb = col[irowmax];
      }
   }
   if (col[icolmin] < col_lwb || col[icolmax] > col_upb) {
      Error("TMatrixTSparse","Inconsistency between column index and its range");
      if (col[icolmin] < col_lwb) {
         Info("TMatrixTSparse","column index lower bound adjusted to %d",col[icolmin]);
         col_lwb = col[icolmin];
      }
      if (col[icolmax] > col_upb) {
         Info("TMatrixTSparse","column index upper bound adjusted to %d",col[icolmax]);
         col_upb = col[icolmax];
      }
   }

   Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb,1,nr);

   SetMatrixArray(nr,row,col,data);
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(const TMatrixTSparse<Element> &another) : TMatrixTBase<Element>(another)
{
   Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb(),1,
            another.GetNoElements());
   memcpy(fRowIndex,another.GetRowIndexArray(),this->fNrowIndex*sizeof(Int_t));
   memcpy(fColIndex,another.GetColIndexArray(),this->fNelems*sizeof(Int_t));

   *this = another;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(const TMatrixT<Element> &another) : TMatrixTBase<Element>(another)
{
   const Int_t nr_nonzeros = another.NonZeros();
   Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb(),1,nr_nonzeros);
   SetSparseIndex(another);
   *this = another;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(EMatrixCreatorsOp1 op,const TMatrixTSparse<Element> &prototype)
{
  // Create a matrix applying a specific operation to the prototype.
  // Supported operations are: kZero, kUnit, kTransposed and kAtA

   R__ASSERT(prototype.IsValid());

   Int_t nr_nonzeros = 0;

   switch(op) {
      case kZero:
      {
         Allocate(prototype.GetNrows(),prototype.GetNcols(),
                  prototype.GetRowLwb(),prototype.GetColLwb(),1,nr_nonzeros);
         break;
      }
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
      {
          Allocate(prototype.GetNcols(), prototype.GetNrows(),
                   prototype.GetColLwb(),prototype.GetRowLwb(),1,prototype.GetNoElements());
          Transpose(prototype);
          break;
      }
      case kAtA:
      {
         const TMatrixTSparse<Element> at(TMatrixTSparse<Element>::kTransposed,prototype);
         AMultBt(at,at,1);
         break;
      }

      default:
         Error("TMatrixTSparse(EMatrixCreatorOp1)","operation %d not yet implemented", op);
   }
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(const TMatrixTSparse<Element> &a,EMatrixCreatorsOp2 op,const TMatrixTSparse<Element> &b)
{
  // Create a matrix applying a specific operation to two prototypes.
  // Supported operations are: kMult (a*b), kMultTranspose (a*b'), kPlus (a+b), kMinus (a-b)

   R__ASSERT(a.IsValid());
   R__ASSERT(b.IsValid());

   switch(op) {
      case kMult:
         AMultB(a,b,1);
         break;

      case kMultTranspose:
         AMultBt(a,b,1);
         break;

      case kPlus:
         APlusB(a,b,1);
         break;

      case kMinus:
         AMinusB(a,b,1);
         break;

      default:
         Error("TMatrixTSparse(EMatrixCreatorOp2)", "operation %d not yet implemented",op);
   }
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(const TMatrixTSparse<Element> &a,EMatrixCreatorsOp2 op,const TMatrixT<Element> &b)
{
  // Create a matrix applying a specific operation to two prototypes.
  // Supported operations are: kMult (a*b), kMultTranspose (a*b'), kPlus (a+b), kMinus (a-b)

   R__ASSERT(a.IsValid());
   R__ASSERT(b.IsValid());

   switch(op) {
      case kMult:
         AMultB(a,b,1);
         break;

      case kMultTranspose:
         AMultBt(a,b,1);
         break;

      case kPlus:
         APlusB(a,b,1);
         break;

      case kMinus:
         AMinusB(a,b,1);
         break;

      default:
         Error("TMatrixTSparse(EMatrixCreatorOp2)", "operation %d not yet implemented",op);
   }
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element>::TMatrixTSparse(const TMatrixT<Element> &a,EMatrixCreatorsOp2 op,const TMatrixTSparse<Element> &b)
{
  // Create a matrix applying a specific operation to two prototypes.
  // Supported operations are: kMult (a*b), kMultTranspose (a*b'), kPlus (a+b), kMinus (a-b)

   R__ASSERT(a.IsValid());
   R__ASSERT(b.IsValid());

   switch(op) {
      case kMult:
         AMultB(a,b,1);
         break;

      case kMultTranspose:
         AMultBt(a,b,1);
         break;

      case kPlus:
         APlusB(a,b,1);
         break;

      case kMinus:
         AMinusB(a,b,1);
         break;

      default:
         Error("TMatrixTSparse(EMatrixCreatorOp2)", "operation %d not yet implemented",op);
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::Allocate(Int_t no_rows,Int_t no_cols,Int_t row_lwb,Int_t col_lwb,
                              Int_t init,Int_t nr_nonzeros)
{
  // Allocate new matrix. Arguments are number of rows, columns, row lowerbound (0 default)
  // and column lowerbound (0 default), 0 initialization flag and number of non-zero
  // elements (only relevant for sparse format).

   if ( (nr_nonzeros > 0 && (no_rows == 0 || no_cols == 0)) ||
       (no_rows < 0 || no_cols < 0 || nr_nonzeros < 0) )
   {
      Error("Allocate","no_rows=%d no_cols=%d non_zeros=%d",no_rows,no_cols,nr_nonzeros);
      this->Invalidate();
      return;
   }

   this->MakeValid();
   this->fNrows     = no_rows;
   this->fNcols     = no_cols;
   this->fRowLwb    = row_lwb;
   this->fColLwb    = col_lwb;
   this->fNrowIndex = this->fNrows+1;
   this->fNelems    = nr_nonzeros;
   this->fIsOwner   = kTRUE;
   this->fTol       = std::numeric_limits<Element>::epsilon();

   fRowIndex = new Int_t[this->fNrowIndex];
   if (init)
      memset(fRowIndex,0,this->fNrowIndex*sizeof(Int_t));

   if (this->fNelems > 0) {
      fElements = new Element[this->fNelems];
      fColIndex = new Int_t   [this->fNelems];
      if (init) {
         memset(fElements,0,this->fNelems*sizeof(Element));
         memset(fColIndex,0,this->fNelems*sizeof(Int_t));
      }
   } else {
      fElements = 0;
      fColIndex = 0;
   }
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::InsertRow(Int_t rown,Int_t coln,const Element *v,Int_t n)
{
  // Insert in row rown, n elements of array v at column coln

   const Int_t arown = rown-this->fRowLwb;
   const Int_t acoln = coln-this->fColLwb;
   const Int_t nr = (n > 0) ? n : this->fNcols;

   if (gMatrixCheck) {
      if (arown >= this->fNrows || arown < 0) {
         Error("InsertRow","row %d out of matrix range",rown);
         return *this;
      }

      if (acoln >= this->fNcols || acoln < 0) {
         Error("InsertRow","column %d out of matrix range",coln);
         return *this;
      }

      if (acoln+nr > this->fNcols || nr < 0) {
         Error("InsertRow","row length %d out of range",nr);
         return *this;
      }
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

   const Int_t nelems_old = this->fNelems;
   const Int_t    *colIndex_old = fColIndex;
   const Element *elements_old = fElements;

   const Int_t ndiff = nr-nslots;
   this->fNelems += ndiff;
   fColIndex = new Int_t[this->fNelems];
   fElements = new Element[this->fNelems];

   for (Int_t irow = arown+1; irow < this->fNrows+1; irow++)
      fRowIndex[irow] += ndiff;

   if (lIndex+1 > 0) {
      memmove(fColIndex,colIndex_old,(lIndex+1)*sizeof(Int_t));
      memmove(fElements,elements_old,(lIndex+1)*sizeof(Element));
   }

   if (nelems_old > 0 && nelems_old-rIndex > 0) {
      memmove(fColIndex+rIndex+ndiff,colIndex_old+rIndex,(nelems_old-rIndex)*sizeof(Int_t));
      memmove(fElements+rIndex+ndiff,elements_old+rIndex,(nelems_old-rIndex)*sizeof(Element));
   }

   index = lIndex+1;
   for (Int_t i = 0; i < nr; i++) {
      fColIndex[index] = acoln+i;
      fElements[index] = v[i];
      index++;
   }

   if (colIndex_old) delete [] (Int_t*)    colIndex_old;
   if (elements_old) delete [] (Element*) elements_old;

   R__ASSERT(this->fNelems == fRowIndex[this->fNrowIndex-1]);

   return *this;
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::ExtractRow(Int_t rown, Int_t coln, Element *v,Int_t n) const
{
  // Store in array v, n matrix elements of row rown starting at column coln

   const Int_t arown = rown-this->fRowLwb;
   const Int_t acoln = coln-this->fColLwb;
   const Int_t nr = (n > 0) ? n : this->fNcols;

   if (gMatrixCheck) {
      if (arown >= this->fNrows || arown < 0) {
         Error("ExtractRow","row %d out of matrix range",rown);
         return;
      }

      if (acoln >= this->fNcols || acoln < 0) {
         Error("ExtractRow","column %d out of matrix range",coln);
         return;
      }

      if (acoln+nr > this->fNcols || nr < 0) {
         Error("ExtractRow","row length %d out of range",nr);
         return;
      }
   }

   const Int_t sIndex = fRowIndex[arown];
   const Int_t eIndex = fRowIndex[arown+1];

   memset(v,0,nr*sizeof(Element));
   const Int_t   * const pColIndex = GetColIndexArray();
   const Element * const pData     = GetMatrixArray();
   for (Int_t index = sIndex; index < eIndex; index++) {
      const Int_t icol = pColIndex[index];
      if (icol < acoln || icol >= acoln+nr) continue;
       v[icol-acoln] = pData[index];
   }
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::AMultBt(const TMatrixTSparse<Element> &a,const TMatrixTSparse<Element> &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B'.
  // Note, matrix C is allocated for constr=1.

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNcols() != b.GetNcols() || a.GetColLwb() != b.GetColLwb()) {
         Error("AMultBt","A and B columns incompatible");
         return;
      }

      if (!constr && this->GetMatrixArray() == a.GetMatrixArray()) {
         Error("AMultB","this = &a");
         return;
      }

      if (!constr && this->GetMatrixArray() == b.GetMatrixArray()) {
         Error("AMultB","this = &b");
         return;
      }
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

      const Int_t nc = nr_nonzero_rowa*nr_nonzero_rowb; // best guess
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

   const Element * const pDataa = a.GetMatrixArray();
   const Element * const pDatab = b.GetMatrixArray();
   Element * const pDatac = this->GetMatrixArray();
   Int_t indexc_r = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexa = pRowIndexa[irowc];
      const Int_t eIndexa = pRowIndexa[irowc+1];
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         const Int_t sIndexb = pRowIndexb[icolc];
         const Int_t eIndexb = pRowIndexb[icolc+1];
         Element sum = 0.0;
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
         if (sum != 0.0) {
            pColIndexc[indexc_r] = icolc;
            pDatac[indexc_r] = sum;
            indexc_r++;
         }
      }
      pRowIndexc[irowc+1] = indexc_r;
   }

   if (constr)
      SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::AMultBt(const TMatrixTSparse<Element> &a,const TMatrixT<Element> &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B'.
  // Note, matrix C is allocated for constr=1.

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNcols() != b.GetNcols() || a.GetColLwb() != b.GetColLwb()) {
         Error("AMultBt","A and B columns incompatible");
         return;
      }

      if (!constr && this->GetMatrixArray() == a.GetMatrixArray()) {
         Error("AMultB","this = &a");
         return;
      }

      if (!constr && this->GetMatrixArray() == b.GetMatrixArray()) {
         Error("AMultB","this = &b");
         return;
      }
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

      const Int_t nc = nr_nonzero_rowa*nr_nonzero_rowb; // best guess
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

   const Element * const pDataa = a.GetMatrixArray();
   const Element * const pDatab = b.GetMatrixArray();
   Element * const pDatac = this->GetMatrixArray();
   Int_t indexc_r = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexa = pRowIndexa[irowc];
      const Int_t eIndexa = pRowIndexa[irowc+1];
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         const Int_t off = icolc*b.GetNcols();
         Element sum = 0.0;
         for (Int_t indexa = sIndexa; indexa < eIndexa; indexa++) {
            const Int_t icola = pColIndexa[indexa];
            sum += pDataa[indexa]*pDatab[off+icola];
         }
         if (sum != 0.0) {
            pColIndexc[indexc_r] = icolc;
            pDatac[indexc_r] = sum;
            indexc_r++;
         }
      }
      pRowIndexc[irowc+1]= indexc_r;
   }

   if (constr)
      SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::AMultBt(const TMatrixT<Element> &a,const TMatrixTSparse<Element> &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B'.
  // Note, matrix C is allocated for constr=1.

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNcols() != b.GetNcols() || a.GetColLwb() != b.GetColLwb()) {
         Error("AMultBt","A and B columns incompatible");
         return;
      }

      if (!constr && this->GetMatrixArray() == a.GetMatrixArray()) {
         Error("AMultB","this = &a");
         return;
      }

      if (!constr && this->GetMatrixArray() == b.GetMatrixArray()) {
         Error("AMultB","this = &b");
         return;
      }
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

      const Int_t nc = nr_nonzero_rowa*nr_nonzero_rowb; // best guess
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

   const Element * const pDataa = a.GetMatrixArray();
   const Element * const pDatab = b.GetMatrixArray();
   Element * const pDatac = this->GetMatrixArray();
   Int_t indexc_r = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t off = irowc*a.GetNcols();
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         const Int_t sIndexb = pRowIndexb[icolc];
         const Int_t eIndexb = pRowIndexb[icolc+1];
         Element sum = 0.0;
         for (Int_t indexb = sIndexb; indexb < eIndexb; indexb++) {
            const Int_t icolb = pColIndexb[indexb];
            sum += pDataa[off+icolb]*pDatab[indexb];
         }
         if (sum != 0.0) {
            pColIndexc[indexc_r] = icolc;
            pDatac[indexc_r] = sum;
            indexc_r++;
         }
      }
      pRowIndexc[irowc+1] = indexc_r;
   }

   if (constr)
      SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::APlusB(const TMatrixTSparse<Element> &a,const TMatrixTSparse<Element> &b,Int_t constr)
{
  // General matrix addition. Create a matrix C such that C = A + B.
  // Note, matrix C is allocated for constr=1.

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
          a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
         Error("APlusB(const TMatrixTSparse &,const TMatrixTSparse &","matrices not compatible");
         return;
      }

      if (!constr && this->GetMatrixArray() == a.GetMatrixArray()) {
         Error("APlusB","this = &a");
         return;
      }

      if (!constr && this->GetMatrixArray() == b.GetMatrixArray()) {
         Error("APlusB","this = &b");
         return;
      }
   }

   const Int_t * const pRowIndexa = a.GetRowIndexArray();
   const Int_t * const pRowIndexb = b.GetRowIndexArray();
   const Int_t * const pColIndexa = a.GetColIndexArray();
   const Int_t * const pColIndexb = b.GetColIndexArray();

   if (constr) {
      Allocate(a.GetNrows(),a.GetNcols(),a.GetRowLwb(),a.GetColLwb());
      SetSparseIndexAB(a,b);
   }

   Int_t * const pRowIndexc = this->GetRowIndexArray();
   Int_t * const pColIndexc = this->GetColIndexArray();

   const Element * const pDataa = a.GetMatrixArray();
   const Element * const pDatab = b.GetMatrixArray();
   Element * const pDatac = this->GetMatrixArray();
   Int_t indexc_r = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexa = pRowIndexa[irowc];
      const Int_t eIndexa = pRowIndexa[irowc+1];
      const Int_t sIndexb = pRowIndexb[irowc];
      const Int_t eIndexb = pRowIndexb[irowc+1];
      Int_t indexa = sIndexa;
      Int_t indexb = sIndexb;
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         Element sum = 0.0;
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

         if (sum != 0.0) {
            pColIndexc[indexc_r] = icolc;
            pDatac[indexc_r] = sum;
            indexc_r++;
         }
      }
      pRowIndexc[irowc+1] = indexc_r;
   }

   if (constr)
      SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::APlusB(const TMatrixTSparse<Element> &a,const TMatrixT<Element> &b,Int_t constr)
{
  // General matrix addition. Create a matrix C such that C = A + B.
  // Note, matrix C is allocated for constr=1.

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
          a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
         Error("APlusB(const TMatrixTSparse &,const TMatrixT &","matrices not compatible");
         return;
      }

      if (!constr && this->GetMatrixArray() == a.GetMatrixArray()) {
         Error("APlusB","this = &a");
         return;
      }

      if (!constr && this->GetMatrixArray() == b.GetMatrixArray()) {
         Error("APlusB","this = &b");
         return;
      }
   }

   if (constr) {
      Allocate(a.GetNrows(),a.GetNcols(),a.GetRowLwb(),a.GetColLwb());
      SetSparseIndexAB(a,b);
   }

   Int_t * const pRowIndexc = this->GetRowIndexArray();
   Int_t * const pColIndexc = this->GetColIndexArray();

   const Int_t * const pRowIndexa = a.GetRowIndexArray();
   const Int_t * const pColIndexa = a.GetColIndexArray();

   const Element * const pDataa = a.GetMatrixArray();
   const Element * const pDatab = b.GetMatrixArray();
   Element * const pDatac = this->GetMatrixArray();
   Int_t indexc_r = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexa = pRowIndexa[irowc];
      const Int_t eIndexa = pRowIndexa[irowc+1];
      const Int_t off = irowc*this->GetNcols();
      Int_t indexa = sIndexa;
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         Element sum = pDatab[off+icolc];
         while (indexa < eIndexa && pColIndexa[indexa] <= icolc) {
            if (icolc == pColIndexa[indexa]) {
               sum += pDataa[indexa];
               break;
            }
            indexa++;
         }

         if (sum != 0.0) {
            pColIndexc[indexc_r] = icolc;
            pDatac[indexc_r] = sum;
            indexc_r++;
         }
      }
      pRowIndexc[irowc+1] = indexc_r;
   }

   if (constr)
      SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::AMinusB(const TMatrixTSparse<Element> &a,const TMatrixTSparse<Element> &b,Int_t constr)
{
  // General matrix subtraction. Create a matrix C such that C = A - B.
  // Note, matrix C is allocated for constr=1.

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
          a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
         Error("AMinusB(const TMatrixTSparse &,const TMatrixTSparse &","matrices not compatible");
         return;
      }

      if (!constr && this->GetMatrixArray() == a.GetMatrixArray()) {
         Error("AMinusB","this = &a");
         return;
      }

      if (!constr && this->GetMatrixArray() == b.GetMatrixArray()) {
         Error("AMinusB","this = &b");
         return;
      }
   }

   const Int_t * const pRowIndexa = a.GetRowIndexArray();
   const Int_t * const pRowIndexb = b.GetRowIndexArray();
   const Int_t * const pColIndexa = a.GetColIndexArray();
   const Int_t * const pColIndexb = b.GetColIndexArray();

   if (constr) {
      Allocate(a.GetNrows(),a.GetNcols(),a.GetRowLwb(),a.GetColLwb());
      SetSparseIndexAB(a,b);
   }

   Int_t * const pRowIndexc = this->GetRowIndexArray();
   Int_t * const pColIndexc = this->GetColIndexArray();

   const Element * const pDataa = a.GetMatrixArray();
   const Element * const pDatab = b.GetMatrixArray();
   Element * const pDatac = this->GetMatrixArray();
   Int_t indexc_r = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexa = pRowIndexa[irowc];
      const Int_t eIndexa = pRowIndexa[irowc+1];
      const Int_t sIndexb = pRowIndexb[irowc];
      const Int_t eIndexb = pRowIndexb[irowc+1];
      Int_t indexa = sIndexa;
      Int_t indexb = sIndexb;
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         Element sum = 0.0;
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

         if (sum != 0.0) {
            pColIndexc[indexc_r] = icolc;
            pDatac[indexc_r] = sum;
            indexc_r++;
         }
      }
      pRowIndexc[irowc+1] = indexc_r;
   }

   if (constr)
      SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::AMinusB(const TMatrixTSparse<Element> &a,const TMatrixT<Element> &b,Int_t constr)
{
  // General matrix subtraction. Create a matrix C such that C = A - B.
  // Note, matrix C is allocated for constr=1.

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
          a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
          Error("AMinusB(const TMatrixTSparse &,const TMatrixT &","matrices not compatible");
          return;
      }

      if (!constr && this->GetMatrixArray() == a.GetMatrixArray()) {
         Error("AMinusB","this = &a");
         return;
      }

      if (!constr && this->GetMatrixArray() == b.GetMatrixArray()) {
         Error("AMinusB","this = &b");
         return;
      }
   }

   if (constr) {
      Allocate(a.GetNrows(),a.GetNcols(),a.GetRowLwb(),a.GetColLwb());
      SetSparseIndexAB(a,b);
   }

   Int_t * const pRowIndexc = this->GetRowIndexArray();
   Int_t * const pColIndexc = this->GetColIndexArray();

   const Int_t * const pRowIndexa = a.GetRowIndexArray();
   const Int_t * const pColIndexa = a.GetColIndexArray();

   const Element * const pDataa = a.GetMatrixArray();
   const Element * const pDatab = b.GetMatrixArray();
   Element * const pDatac = this->GetMatrixArray();
   Int_t indexc_r = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexa = pRowIndexa[irowc];
      const Int_t eIndexa = pRowIndexa[irowc+1];
      const Int_t off = irowc*this->GetNcols();
      Int_t indexa = sIndexa;
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         Element sum = -pDatab[off+icolc];
         while (indexa < eIndexa && pColIndexa[indexa] <= icolc) {
            if (icolc == pColIndexa[indexa]) {
               sum += pDataa[indexa];
               break;
            }
            indexa++;
         }

         if (sum != 0.0) {
            pColIndexc[indexc_r] = icolc;
            pDatac[indexc_r] = sum;
            indexc_r++;
         }
      }
      pRowIndexc[irowc+1] = indexc_r;
   }

   if (constr)
      SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::AMinusB(const TMatrixT<Element> &a,const TMatrixTSparse<Element> &b,Int_t constr)
{
  // General matrix subtraction. Create a matrix C such that C = A - B.
  // Note, matrix C is allocated for constr=1.

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
          a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
         Error("AMinusB(const TMatrixT &,const TMatrixTSparse &","matrices not compatible");
         return;
      }

      if (!constr && this->GetMatrixArray() == a.GetMatrixArray()) {
         Error("AMinusB","this = &a");
         return;
      }

      if (!constr && this->GetMatrixArray() == b.GetMatrixArray()) {
         Error("AMinusB","this = &b");
         return;
      }
   }

   if (constr) {
      Allocate(a.GetNrows(),a.GetNcols(),a.GetRowLwb(),a.GetColLwb());
      SetSparseIndexAB(a,b);
   }

   Int_t * const pRowIndexc = this->GetRowIndexArray();
   Int_t * const pColIndexc = this->GetColIndexArray();

   const Int_t * const pRowIndexb = b.GetRowIndexArray();
   const Int_t * const pColIndexb = b.GetColIndexArray();

   const Element * const pDataa = a.GetMatrixArray();
   const Element * const pDatab = b.GetMatrixArray();
   Element * const pDatac = this->GetMatrixArray();
   Int_t indexc_r = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexb = pRowIndexb[irowc];
      const Int_t eIndexb = pRowIndexb[irowc+1];
      const Int_t off = irowc*this->GetNcols();
      Int_t indexb = sIndexb;
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         Element sum = pDataa[off+icolc];
         while (indexb < eIndexb && pColIndexb[indexb] <= icolc) {
            if (icolc == pColIndexb[indexb]) {
               sum -= pDatab[indexb];
               break;
            }
            indexb++;
         }

         if (sum != 0.0) {
            pColIndexc[indexc_r] = icolc;
            pDatac[indexc_r] = sum;
            indexc_r++;
         }
      }
      pRowIndexc[irowc+1] = indexc_r;
   }

   if (constr)
      SetSparseIndex(indexc_r);
}

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::GetMatrix2Array(Element *data,Option_t * /*option*/) const
{
  // Copy matrix data to array . It is assumed that array is of size >= fNelems

   R__ASSERT(this->IsValid());

   const Element * const elem = GetMatrixArray();
   memcpy(data,elem,this->fNelems*sizeof(Element));
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::SetMatrixArray(Int_t nr,Int_t *row,Int_t *col,Element *data)
{
  // Copy nr elements from row/col index and data array to matrix . It is assumed
  // that arrays are of size >= nr
  // Note that the input arrays are not passed as const since they will be modified ! 

   R__ASSERT(this->IsValid());
   if (nr <= 0) {
      Error("SetMatrixArray(Int_t,Int_t*,Int_t*,Element*","nr <= 0");
      return *this;
   }

   const Int_t irowmin = TMath::LocMin(nr,row);
   const Int_t irowmax = TMath::LocMax(nr,row);
   const Int_t icolmin = TMath::LocMin(nr,col);
   const Int_t icolmax = TMath::LocMax(nr,col);

   R__ASSERT(row[irowmin] >= this->fRowLwb && row[irowmax] <= this->fRowLwb+this->fNrows-1);
   R__ASSERT(col[icolmin] >= this->fColLwb && col[icolmax] <= this->fColLwb+this->fNcols-1);

   if (row[irowmin] < this->fRowLwb || row[irowmax] > this->fRowLwb+this->fNrows-1) {
      Error("SetMatrixArray","Inconsistency between row index and its range");
      if (row[irowmin] < this->fRowLwb) {
         Info("SetMatrixArray","row index lower bound adjusted to %d",row[irowmin]);
         this->fRowLwb = row[irowmin];
      }
      if (row[irowmax] > this->fRowLwb+this->fNrows-1) {
         Info("SetMatrixArray","row index upper bound adjusted to %d",row[irowmax]);
         this->fNrows = row[irowmax]-this->fRowLwb+1;
      }
   }
   if (col[icolmin] < this->fColLwb || col[icolmax] > this->fColLwb+this->fNcols-1) {
      Error("SetMatrixArray","Inconsistency between column index and its range");
      if (col[icolmin] < this->fColLwb) {
         Info("SetMatrixArray","column index lower bound adjusted to %d",col[icolmin]);
         this->fColLwb = col[icolmin];
      }
      if (col[icolmax] > this->fColLwb+this->fNcols-1) {
         Info("SetMatrixArray","column index upper bound adjusted to %d",col[icolmax]);
         this->fNcols = col[icolmax]-this->fColLwb+1;
      }
   }

   TMatrixTBase<Element>::DoubleLexSort(nr,row,col,data);

   Int_t nr_nonzeros = 0;
   const Element *ep        = data;
   const Element * const fp = data+nr;

   while (ep < fp)
      if (*ep++ != 0.0) nr_nonzeros++;

   if (nr_nonzeros != this->fNelems) {
      if (fColIndex) { delete [] fColIndex; fColIndex = 0; }
      if (fElements) { delete [] fElements; fElements = 0; }
      this->fNelems = nr_nonzeros;
      if (this->fNelems > 0) {
         fColIndex = new Int_t[nr_nonzeros];
         fElements = new Element[nr_nonzeros];
      } else {
         fColIndex = 0;
         fElements = 0;
      }
   }

   if (this->fNelems <= 0)
      return *this;

   fRowIndex[0] = 0;
   Int_t ielem = 0;
   nr_nonzeros = 0;
   for (Int_t irow = 1; irow < this->fNrows+1; irow++) {
      if (ielem < nr && row[ielem] < irow) {
         while (ielem < nr) {
            if (data[ielem] != 0.0) {
               fColIndex[nr_nonzeros] = col[ielem]-this->fColLwb;
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

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::SetSparseIndex(Int_t nelems_new)
{
  // Increase/decrease the number of non-zero elements to nelems_new

   if (nelems_new != this->fNelems) {
      Int_t nr = TMath::Min(nelems_new,this->fNelems);
      Int_t *oIp = fColIndex;
      fColIndex = new Int_t[nelems_new];
      if (oIp) { 
         memmove(fColIndex,oIp,nr*sizeof(Int_t));
         delete [] oIp;
      }
      Element *oDp = fElements;
      fElements = new Element[nelems_new];
      if (oDp) { 
         memmove(fElements,oDp,nr*sizeof(Element));
         delete [] oDp;
      }
      this->fNelems = nelems_new;
      if (nelems_new > nr) {
         memset(fElements+nr,0,(nelems_new-nr)*sizeof(Element));
         memset(fColIndex+nr,0,(nelems_new-nr)*sizeof(Int_t));
      } else {
         for (Int_t irow = 0; irow < this->fNrowIndex; irow++)
            if (fRowIndex[irow] > nelems_new)
               fRowIndex[irow] = nelems_new;
      }
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::SetSparseIndex(const TMatrixTBase<Element> &source)
{
  // Use non-zero data of matrix source to set the sparse structure

   if (gMatrixCheck) {
      R__ASSERT(source.IsValid());
      if (this->GetNrows()  != source.GetNrows()  || this->GetNcols()  != source.GetNcols() ||
          this->GetRowLwb() != source.GetRowLwb() || this->GetColLwb() != source.GetColLwb()) {
         Error("SetSparseIndex","matrices not compatible");
         return *this;
      }
   }

   const Int_t nr_nonzeros = source.NonZeros();

   if (nr_nonzeros != this->fNelems)
      SetSparseIndex(nr_nonzeros);

   if (source.GetRowIndexArray() && source.GetColIndexArray()) {
      memmove(fRowIndex,source.GetRowIndexArray(),this->fNrowIndex*sizeof(Int_t));
      memmove(fColIndex,source.GetColIndexArray(),this->fNelems*sizeof(Int_t));
   } else {
      const Element *ep = source.GetMatrixArray();
      Int_t nr = 0;
      for (Int_t irow = 0; irow < this->fNrows; irow++) {
         fRowIndex[irow] = nr;
         for (Int_t icol = 0; icol < this->fNcols; icol++) {
            if (*ep != 0.0) {
               fColIndex[nr] = icol;
               nr++;
            }
            ep++;
         }
      }
      fRowIndex[this->fNrows] = nr;
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::SetSparseIndexAB(const TMatrixTSparse<Element> &a,const TMatrixTSparse<Element> &b)
{
  // Set the row/column indices to the "sum" of matrices a and b
  // It is checked that enough space has been allocated

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
          a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
         Error("SetSparseIndexAB","source matrices not compatible");
         return *this;
      }

      if (this->GetNrows()  != a.GetNrows()  || this->GetNcols()  != a.GetNcols() ||
          this->GetRowLwb() != a.GetRowLwb() || this->GetColLwb() != a.GetColLwb()) {
         Error("SetSparseIndexAB","matrix not compatible with source matrices");
         return *this;
      }
   }

   const Int_t * const pRowIndexa = a.GetRowIndexArray();
   const Int_t * const pRowIndexb = b.GetRowIndexArray();
   const Int_t * const pColIndexa = a.GetColIndexArray();
   const Int_t * const pColIndexb = b.GetColIndexArray();

   // Count first the number of non-zero slots that are needed
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
         const Int_t icola = (eIndexa > sIndexa && eIndexa > 0) ? pColIndexa[eIndexa-1] : -1;
         if (pColIndexb[indexb++] > icola)
            nc++;
      }
   }

   // Allocate the necessary space in fRowIndex and fColIndex
   if (this->NonZeros() != nc)
      SetSparseIndex(nc);

   Int_t * const pRowIndexc = this->GetRowIndexArray();
   Int_t * const pColIndexc = this->GetColIndexArray();

   nc = 0;
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
         const Int_t icola = (eIndexa > 0) ? pColIndexa[eIndexa-1] : -1;
         if (pColIndexb[indexb++] > icola)
            pColIndexc[nc++] = pColIndexb[indexb-1];
      }
      pRowIndexc[irowc+1] = nc;
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::SetSparseIndexAB(const TMatrixT<Element> &a,const TMatrixTSparse<Element> &b)
{
  // Set the row/column indices to the "sum" of matrices a and b
  // It is checked that enough space has been allocated

   if (gMatrixCheck) {
      R__ASSERT(a.IsValid());
      R__ASSERT(b.IsValid());

      if (a.GetNrows()  != b.GetNrows()  || a.GetNcols()  != b.GetNcols() ||
          a.GetRowLwb() != b.GetRowLwb() || a.GetColLwb() != b.GetColLwb()) {
         Error("SetSparseIndexAB","source matrices not compatible");
         return *this;
      }

      if (this->GetNrows()  != a.GetNrows()  || this->GetNcols()  != a.GetNcols() ||
          this->GetRowLwb() != a.GetRowLwb() || this->GetColLwb() != a.GetColLwb()) {
         Error("SetSparseIndexAB","matrix not compatible with source matrices");
         return *this;
      }
   }

   const Element * const pDataa = a.GetMatrixArray();

   const Int_t * const pRowIndexb = b.GetRowIndexArray();
   const Int_t * const pColIndexb = b.GetColIndexArray();

   // Count first the number of non-zero slots that are needed
   Int_t nc = a.NonZeros();
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexb = pRowIndexb[irowc];
      const Int_t eIndexb = pRowIndexb[irowc+1];
      const Int_t off = irowc*this->GetNcols();
      Int_t indexb = sIndexb;
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         if (pDataa[off+icolc] != 0.0 || pColIndexb[indexb] > icolc) continue;
         for (; indexb < eIndexb; indexb++) {
            if (pColIndexb[indexb] >= icolc) {
               if (pColIndexb[indexb] == icolc) {
                  nc++;
                  indexb++;
               }
               break;
            }
         }
      }
   }

   // Allocate thre necessary space in fRowIndex and fColIndex
   if (this->NonZeros() != nc)
      SetSparseIndex(nc);

   Int_t * const pRowIndexc = this->GetRowIndexArray();
   Int_t * const pColIndexc = this->GetColIndexArray();

   nc = 0;
   pRowIndexc[0] = 0;
   for (Int_t irowc = 0; irowc < this->GetNrows(); irowc++) {
      const Int_t sIndexb = pRowIndexb[irowc];
      const Int_t eIndexb = pRowIndexb[irowc+1];
      const Int_t off = irowc*this->GetNcols();
      Int_t indexb = sIndexb;
      for (Int_t icolc = 0; icolc < this->GetNcols(); icolc++) {
         if (pDataa[off+icolc] != 0.0)
            pColIndexc[nc++] = icolc;
         else if (pColIndexb[indexb] <= icolc) {
            for (; indexb < eIndexb; indexb++) {
               if (pColIndexb[indexb] >= icolc) {
                  if (pColIndexb[indexb] == icolc)
                     pColIndexc[nc++] = pColIndexb[indexb++];
                  break;
               }
            }
         }
      }
      pRowIndexc[irowc+1] = nc;
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::ResizeTo(Int_t nrows,Int_t ncols,Int_t nr_nonzeros)
{
  // Set size of the matrix to nrows x ncols with nr_nonzeros non-zero entries
  // if nr_nonzeros > 0 .
  // New dynamic elements are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleted.

   R__ASSERT(this->IsValid());
   if (!this->fIsOwner) {
      Error("ResizeTo(Int_t,Int_t,Int_t)","Not owner of data array,cannot resize");
      return *this;
   }

   if (this->fNelems > 0) {
      if (this->fNrows == nrows && this->fNcols == ncols &&
         (this->fNelems == nr_nonzeros || nr_nonzeros < 0))
         return *this;
      else if (nrows == 0 || ncols == 0 || nr_nonzeros == 0) {
         this->fNrows = nrows; this->fNcols = ncols;
         Clear();
         return *this;
      }

      const Element *elements_old = GetMatrixArray();
      const Int_t   *rowIndex_old = GetRowIndexArray();
      const Int_t   *colIndex_old = GetColIndexArray();

      Int_t nrows_old     = this->fNrows;
      Int_t nrowIndex_old = this->fNrowIndex;

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
      R__ASSERT(this->IsValid());

      Element *elements_new = GetMatrixArray();
      Int_t   *rowIndex_new = GetRowIndexArray();
      Int_t   *colIndex_new = GetColIndexArray();

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

      if (rowIndex_old) delete [] (Int_t*)   rowIndex_old;
      if (colIndex_old) delete [] (Int_t*)   colIndex_old;
      if (elements_old) delete [] (Element*) elements_old;

      if (nrowIndex_old < this->fNrowIndex) {
          for (Int_t irow = nrowIndex_old; irow < this->fNrowIndex; irow++)
            rowIndex_new[irow] = rowIndex_new[nrowIndex_old-1];
      }
   } else {
      const Int_t nelems_new = (nr_nonzeros >= 0) ? nr_nonzeros : 0;
      Allocate(nrows,ncols,0,0,1,nelems_new);
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                         Int_t nr_nonzeros)
{
  // Set size of the matrix to [row_lwb:row_upb] x [col_lwb:col_upb] with nr_nonzeros
  // non-zero entries if nr_nonzeros > 0 .
  // New dynamic elemenst are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleted.

   R__ASSERT(this->IsValid());
   if (!this->fIsOwner) {
      Error("ResizeTo(Int_t,Int_t,Int_t,Int_t,Int_t)","Not owner of data array,cannot resize");
      return *this;
   }

   const Int_t new_nrows = row_upb-row_lwb+1;
   const Int_t new_ncols = col_upb-col_lwb+1;

   if (this->fNelems > 0) {
      if (this->fNrows  == new_nrows && this->fNcols  == new_ncols &&
          this->fRowLwb == row_lwb   && this->fColLwb == col_lwb &&
          (this->fNelems == nr_nonzeros || nr_nonzeros < 0))
          return *this;
      else if (new_nrows == 0 || new_ncols == 0 || nr_nonzeros == 0) {
         this->fNrows = new_nrows; this->fNcols = new_ncols;
         this->fRowLwb = row_lwb; this->fColLwb = col_lwb;
         Clear();
         return *this;
      }

      const Int_t   *rowIndex_old = GetRowIndexArray();
      const Int_t   *colIndex_old = GetColIndexArray();
      const Element *elements_old = GetMatrixArray();

      const Int_t  nrowIndex_old = this->fNrowIndex;

      const Int_t  nrows_old     = this->fNrows;
      const Int_t  rowLwb_old    = this->fRowLwb;
      const Int_t  colLwb_old    = this->fColLwb;

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
      R__ASSERT(this->IsValid());

      Int_t    *rowIndex_new = GetRowIndexArray();
      Int_t    *colIndex_new = GetColIndexArray();
      Element *elements_new = GetMatrixArray();

      Int_t nelems_copy = 0;
      rowIndex_new[0] = 0;
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
               break;
            }
         }
      }

      if (rowIndex_old) delete [] (Int_t*)   rowIndex_old;
      if (colIndex_old) delete [] (Int_t*)   colIndex_old;
      if (elements_old) delete [] (Element*) elements_old;

      if (nrowIndex_old < this->fNrowIndex) {
         for (Int_t irow = nrowIndex_old; irow < this->fNrowIndex; irow++)
            rowIndex_new[irow] = rowIndex_new[nrowIndex_old-1];
      }
   } else {
      const Int_t nelems_new = (nr_nonzeros >= 0) ? nr_nonzeros : 0;
      Allocate(new_nrows,new_ncols,row_lwb,col_lwb,1,nelems_new);
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::Use(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                      Int_t nr_nonzeros,Int_t *pRowIndex,Int_t *pColIndex,Element *pData)
{
   if (gMatrixCheck) {
      if (row_upb < row_lwb) {
         Error("Use","row_upb=%d < row_lwb=%d",row_upb,row_lwb);
         return *this;
      }
      if (col_upb < col_lwb) {
         Error("Use","col_upb=%d < col_lwb=%d",col_upb,col_lwb);
         return *this;
      }
   }

   Clear();

   this->fNrows     = row_upb-row_lwb+1;
   this->fNcols     = col_upb-col_lwb+1;
   this->fRowLwb    = row_lwb;
   this->fColLwb    = col_lwb;
   this->fNrowIndex = this->fNrows+1;
   this->fNelems    = nr_nonzeros;
   this->fIsOwner   = kFALSE;
   this->fTol       = std::numeric_limits<Element>::epsilon();

   fElements  = pData;
   fRowIndex  = pRowIndex;
   fColIndex  = pColIndex;

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::GetSub(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                       TMatrixTBase<Element> &target,Option_t *option) const
{
  // Get submatrix [row_lwb..row_upb][col_lwb..col_upb]; The indexing range of the
  // returned matrix depends on the argument option:
  //
  // option == "S" : return [0..row_upb-row_lwb+1][0..col_upb-col_lwb+1] (default)
  // else          : return [row_lwb..row_upb][col_lwb..col_upb]

   if (gMatrixCheck) {
      R__ASSERT(this->IsValid());
      if (row_lwb < this->fRowLwb || row_lwb > this->fRowLwb+this->fNrows-1) {
         Error("GetSub","row_lwb out-of-bounds");
         return target;
      }
      if (col_lwb < this->fColLwb || col_lwb > this->fColLwb+this->fNcols-1) {
         Error("GetSub","col_lwb out-of-bounds");
         return target;
      }
      if (row_upb < this->fRowLwb || row_upb > this->fRowLwb+this->fNrows-1) {
         Error("GetSub","row_upb out-of-bounds");
         return target;
      }
      if (col_upb < this->fColLwb || col_upb > this->fColLwb+this->fNcols-1) {
         Error("GetSub","col_upb out-of-bounds");
         return target;
      }
      if (row_upb < row_lwb || col_upb < col_lwb) {
         Error("GetSub","row_upb < row_lwb || col_upb < col_lwb");
         return target;
      }
   }

   TString opt(option);
   opt.ToUpper();
   const Int_t shift = (opt.Contains("S")) ? 1 : 0;

   const Int_t row_lwb_sub = (shift) ? 0               : row_lwb;
   const Int_t row_upb_sub = (shift) ? row_upb-row_lwb : row_upb;
   const Int_t col_lwb_sub = (shift) ? 0               : col_lwb;
   const Int_t col_upb_sub = (shift) ? col_upb-col_lwb : col_upb;

   Int_t nr_nonzeros = 0;
   for (Int_t irow = 0; irow < this->fNrows; irow++) {
      if (irow+this->fRowLwb > row_upb || irow+this->fRowLwb < row_lwb) continue;
      const Int_t sIndex = fRowIndex[irow];
      const Int_t eIndex = fRowIndex[irow+1];
      for (Int_t index = sIndex; index < eIndex; index++) {
         const Int_t icol = fColIndex[index];
         if (icol+this->fColLwb <= col_upb && icol+this->fColLwb >= col_lwb)
            nr_nonzeros++;
      }
   }

   target.ResizeTo(row_lwb_sub,row_upb_sub,col_lwb_sub,col_upb_sub,nr_nonzeros);

   const Element *ep = this->GetMatrixArray();

   Int_t   *rowIndex_sub = target.GetRowIndexArray();
   Int_t   *colIndex_sub = target.GetColIndexArray();
   Element *ep_sub       = target.GetMatrixArray();

   if (target.GetRowIndexArray() && target.GetColIndexArray()) {
      Int_t nelems_copy = 0;
      rowIndex_sub[0] = 0;
      const Int_t row_off = this->fRowLwb-row_lwb;
      const Int_t col_off = this->fColLwb-col_lwb;
      for (Int_t irow = 0; irow < this->fNrows; irow++) {
         if (irow+this->fRowLwb > row_upb || irow+this->fRowLwb < row_lwb) continue;
         const Int_t sIndex = fRowIndex[irow];
         const Int_t eIndex = fRowIndex[irow+1];
         for (Int_t index = sIndex; index < eIndex; index++) {
            const Int_t icol = fColIndex[index];
            if (icol+this->fColLwb <= col_upb && icol+this->fColLwb >= col_lwb) {
               rowIndex_sub[irow+row_off+1] = nelems_copy+1;
               colIndex_sub[nelems_copy]    = icol+col_off;
               ep_sub[nelems_copy]          = ep[index];
               nelems_copy++;
            }
         }
      }
   } else {
      const Int_t row_off = this->fRowLwb-row_lwb;
      const Int_t col_off = this->fColLwb-col_lwb;
      const Int_t ncols_sub = col_upb_sub-col_lwb_sub+1;
      for (Int_t irow = 0; irow < this->fNrows; irow++) {
         if (irow+this->fRowLwb > row_upb || irow+this->fRowLwb < row_lwb) continue;
         const Int_t sIndex = fRowIndex[irow];
         const Int_t eIndex = fRowIndex[irow+1];
         const Int_t off = (irow+row_off)*ncols_sub;
         for (Int_t index = sIndex; index < eIndex; index++) {
            const Int_t icol = fColIndex[index];
            if (icol+this->fColLwb <= col_upb && icol+this->fColLwb >= col_lwb)
               ep_sub[off+icol+col_off] = ep[index];
         }
      }
    }

   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixTBase<Element> &source)
{
  // Insert matrix source starting at [row_lwb][col_lwb], thereby overwriting the part
  // [row_lwb..row_lwb+nrows_source-1][col_lwb..col_lwb+ncols_source-1];

   if (gMatrixCheck) {
      R__ASSERT(this->IsValid());
      R__ASSERT(source.IsValid());

      if (row_lwb < this->fRowLwb || row_lwb > this->fRowLwb+this->fNrows-1) {
         Error("SetSub","row_lwb out-of-bounds");
         return *this;
      }
      if (col_lwb < this->fColLwb || col_lwb > this->fColLwb+this->fNcols-1) {
         Error("SetSub","col_lwb out-of-bounds");
         return *this;
      }
      if (row_lwb+source.GetNrows() > this->fRowLwb+this->fNrows || col_lwb+source.GetNcols() > this->fColLwb+this->fNcols) {
         Error("SetSub","source matrix too large");
         return *this;
      }
   }

   const Int_t nRows_source = source.GetNrows();
   const Int_t nCols_source = source.GetNcols();

   // Determine how many non-zero's are already available in
   // [row_lwb..row_lwb+nrows_source-1][col_lwb..col_lwb+ncols_source-1]
   Int_t nr_nonzeros = 0;
   for (Int_t irow = 0; irow < this->fNrows; irow++) {
      if (irow+this->fRowLwb >= row_lwb+nRows_source || irow+this->fRowLwb < row_lwb) continue;
      const Int_t sIndex = fRowIndex[irow];
      const Int_t eIndex = fRowIndex[irow+1];
      for (Int_t index = sIndex; index < eIndex; index++) {
         const Int_t icol = fColIndex[index];
         if (icol+this->fColLwb < col_lwb+nCols_source && icol+this->fColLwb >= col_lwb)
            nr_nonzeros++;
      }
   }

   const Int_t   *rowIndex_s = source.GetRowIndexArray();
   const Int_t   *colIndex_s = source.GetColIndexArray();
   const Element *elements_s = source.GetMatrixArray();

   const Int_t nelems_old = this->fNelems;
   const Int_t   *rowIndex_old = GetRowIndexArray();
   const Int_t   *colIndex_old = GetColIndexArray();
   const Element *elements_old = GetMatrixArray();

   const Int_t nelems_new = nelems_old+source.NonZeros()-nr_nonzeros;
   fRowIndex = new Int_t[this->fNrowIndex];
   fColIndex = new Int_t[nelems_new];
   fElements = new Element[nelems_new];
   this->fNelems   = nelems_new;

   Int_t   *rowIndex_new = GetRowIndexArray();
   Int_t   *colIndex_new = GetColIndexArray();
   Element *elements_new = GetMatrixArray();

   const Int_t row_off = row_lwb-this->fRowLwb;
   const Int_t col_off = col_lwb-this->fColLwb;

   Int_t nr = 0;
   rowIndex_new[0] = 0;
   for (Int_t irow = 0; irow < this->fNrows; irow++) {
      rowIndex_new[irow+1] = rowIndex_new[irow];
      Bool_t flagRow = kFALSE;
      if (irow+this->fRowLwb < row_lwb+nRows_source && irow+this->fRowLwb >= row_lwb)
         flagRow = kTRUE;

      const Int_t sIndex_o = rowIndex_old[irow];
      const Int_t eIndex_o = rowIndex_old[irow+1];

      if (flagRow) {
         const Int_t icol_left = col_off-1;
         const Int_t left = TMath::BinarySearch(eIndex_o-sIndex_o,colIndex_old+sIndex_o,icol_left)+sIndex_o;
         for (Int_t index = sIndex_o; index <= left; index++) {
            rowIndex_new[irow+1]++;
            colIndex_new[nr] = colIndex_old[index];
            elements_new[nr] = elements_old[index];
            nr++;
         }

         if (rowIndex_s && colIndex_s) {
            const Int_t sIndex_s = rowIndex_s[irow-row_off];
            const Int_t eIndex_s = rowIndex_s[irow-row_off+1];
            for (Int_t index = sIndex_s; index < eIndex_s; index++) {
               rowIndex_new[irow+1]++;
               colIndex_new[nr] = colIndex_s[index]+col_off;
               elements_new[nr] = elements_s[index];
               nr++;
            }
         } else {
            const Int_t off = (irow-row_off)*nCols_source;
            for (Int_t icol = 0; icol < nCols_source; icol++) {
                rowIndex_new[irow+1]++;
                colIndex_new[nr] = icol+col_off;
                elements_new[nr] = elements_s[off+icol];
                nr++;
            }
         }

         const Int_t icol_right = col_off+nCols_source-1;
         if (colIndex_old) {
            Int_t right = TMath::BinarySearch(eIndex_o-sIndex_o,colIndex_old+sIndex_o,icol_right)+sIndex_o;
            while (right < eIndex_o-1 && colIndex_old[right+1] <= icol_right)
               right++;
            right++;

            for (Int_t index = right; index < eIndex_o; index++) {
               rowIndex_new[irow+1]++;
               colIndex_new[nr] = colIndex_old[index];
               elements_new[nr] = elements_old[index];
               nr++;
            }
         }
      } else {
         for (Int_t index = sIndex_o; index < eIndex_o; index++) {
            rowIndex_new[irow+1]++;
            colIndex_new[nr] = colIndex_old[index];
            elements_new[nr] = elements_old[index];
            nr++;
         }
      }
   }

   R__ASSERT(this->fNelems == fRowIndex[this->fNrowIndex-1]);

   if (rowIndex_old) delete [] (Int_t*)    rowIndex_old;
   if (colIndex_old) delete [] (Int_t*)    colIndex_old;
   if (elements_old) delete [] (Element*) elements_old;

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::Transpose(const TMatrixTSparse<Element> &source)
{
  // Transpose a matrix.

   if (gMatrixCheck) {
      R__ASSERT(this->IsValid());
      R__ASSERT(source.IsValid());

      if (this->fNrows  != source.GetNcols()  || this->fNcols  != source.GetNrows() ||
          this->fRowLwb != source.GetColLwb() || this->fColLwb != source.GetRowLwb()) {
         Error("Transpose","matrix has wrong shape");
         return *this;
      }

      if (source.NonZeros() <= 0)
         return *this;
   }

   const Int_t nr_nonzeros = source.NonZeros();

   const Int_t   * const pRowIndex_s = source.GetRowIndexArray();
   const Int_t   * const pColIndex_s = source.GetColIndexArray();
   const Element * const pData_s     = source.GetMatrixArray();

   Int_t   *rownr   = new Int_t  [nr_nonzeros];
   Int_t   *colnr   = new Int_t  [nr_nonzeros];
   Element *pData_t = new Element[nr_nonzeros];

   Int_t ielem = 0;
   for (Int_t irow_s = 0; irow_s < source.GetNrows(); irow_s++) {
      const Int_t sIndex = pRowIndex_s[irow_s];
      const Int_t eIndex = pRowIndex_s[irow_s+1];
      for (Int_t index = sIndex; index < eIndex; index++) {
         rownr[ielem]   = pColIndex_s[index]+this->fRowLwb;
         colnr[ielem]   = irow_s+this->fColLwb;
         pData_t[ielem] = pData_s[index];
         ielem++;
      }
   }

   R__ASSERT(nr_nonzeros >= ielem);

   TMatrixTBase<Element>::DoubleLexSort(nr_nonzeros,rownr,colnr,pData_t);
   SetMatrixArray(nr_nonzeros,rownr,colnr,pData_t);

   R__ASSERT(this->fNelems == fRowIndex[this->fNrowIndex-1]);

   if (pData_t) delete [] pData_t;
   if (rownr)   delete [] rownr;
   if (colnr)   delete [] colnr;

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::Zero()
{
   R__ASSERT(this->IsValid());

   if (fElements) { delete [] fElements; fElements = 0; }
   if (fColIndex) { delete [] fColIndex; fColIndex = 0; }
   this->fNelems = 0;
   memset(this->GetRowIndexArray(),0,this->fNrowIndex*sizeof(Int_t));

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::UnitMatrix()
{
  // Make a unit matrix (matrix need not be a square one).

   R__ASSERT(this->IsValid());

   Int_t i;

   Int_t nr_nonzeros = 0;
   for (i = this->fRowLwb; i <= this->fRowLwb+this->fNrows-1; i++)
      for (Int_t j = this->fColLwb; j <= this->fColLwb+this->fNcols-1; j++)
         if (i == j) nr_nonzeros++;

   if (nr_nonzeros != this->fNelems) {
      this->fNelems = nr_nonzeros;
      Int_t *oIp = fColIndex;
      fColIndex = new Int_t[nr_nonzeros];
      if (oIp) delete [] oIp;
      Element *oDp = fElements;
      fElements = new Element[nr_nonzeros];
      if (oDp) delete [] oDp;
   }

   Int_t ielem = 0;
   fRowIndex[0] = 0;
   for (i = this->fRowLwb; i <= this->fRowLwb+this->fNrows-1; i++) {
      for (Int_t j = this->fColLwb; j <= this->fColLwb+this->fNcols-1; j++) {
         if (i == j) {
            const Int_t irow = i-this->fRowLwb;
            fRowIndex[irow+1]  = ielem+1;
            fElements[ielem]   = 1.0;
            fColIndex[ielem++] = j-this->fColLwb;
         }
      }
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTSparse<Element>::RowNorm() const
{
  // Row matrix norm, MAX{ SUM{ |M(i,j)|, over j}, over i}.
  // The norm is induced by the infinity vector norm.

   R__ASSERT(this->IsValid());

   const Element *       ep = GetMatrixArray();
   const Element * const fp = ep+this->fNelems;
   const Int_t   * const pR = GetRowIndexArray();
         Element norm = 0;

   // Scan the matrix row-after-row
   for (Int_t irow = 0; irow < this->fNrows; irow++) {
      const Int_t sIndex = pR[irow];
      const Int_t eIndex = pR[irow+1];
      Element sum = 0;
      for (Int_t index = sIndex; index < eIndex; index++)
          sum += TMath::Abs(*ep++);
      norm = TMath::Max(norm,sum);
   }

   R__ASSERT(ep == fp);

   return norm;
}

//______________________________________________________________________________
template<class Element>
Element TMatrixTSparse<Element>::ColNorm() const
{
  // Column matrix norm, MAX{ SUM{ |M(i,j)|, over i}, over j}.
  // The norm is induced by the 1 vector norm.

   R__ASSERT(this->IsValid());

   const TMatrixTSparse<Element> mt(kTransposed,*this);

   const Element *       ep = mt.GetMatrixArray();
   const Element * const fp = ep+this->fNcols;
         Element norm = 0;

   // Scan the matrix col-after-col
   while (ep < fp) {
      Element sum = 0;
      // Scan a col to compute the sum
      for (Int_t i = 0; i < this->fNrows; i++,ep += this->fNcols)
         sum += TMath::Abs(*ep);
      ep -= this->fNelems-1;         // Point ep to the beginning of the next col
      norm = TMath::Max(norm,sum);
   }

   R__ASSERT(ep == fp);

   return norm;
}

//______________________________________________________________________________
template<class Element>
Element &TMatrixTSparse<Element>::operator()(Int_t rown,Int_t coln)
{
   R__ASSERT(this->IsValid());

   const Int_t arown = rown-this->fRowLwb;
   const Int_t acoln = coln-this->fColLwb;
   if (arown >= this->fNrows || arown < 0) {
      Error("operator()","Request row(%d) outside matrix range of %d - %d",rown,this->fRowLwb,this->fRowLwb+this->fNrows);
      return fElements[0];
   }
   if (acoln >= this->fNcols || acoln < 0) {
      Error("operator()","Request column(%d) outside matrix range of %d - %d",coln,this->fColLwb,this->fColLwb+this->fNcols);
      return fElements[0];
   }

   Int_t index = -1;
   Int_t sIndex = 0;
   Int_t eIndex = 0;
   if (this->fNrowIndex > 0 && fRowIndex[this->fNrowIndex-1] != 0) {
      sIndex = fRowIndex[arown];
      eIndex = fRowIndex[arown+1];
      index = TMath::BinarySearch(eIndex-sIndex,fColIndex+sIndex,acoln)+sIndex;
   }

   if (index >= sIndex && fColIndex[index] == acoln)
      return fElements[index];
   else {
      Element val = 0.;
      InsertRow(rown,coln,&val,1);
      sIndex = fRowIndex[arown];
      eIndex = fRowIndex[arown+1];
      index = TMath::BinarySearch(eIndex-sIndex,fColIndex+sIndex,acoln)+sIndex;
      if (index >= sIndex && fColIndex[index] == acoln)
         return fElements[index];
      else {
         Error("operator()(Int_t,Int_t","Insert row failed");
         return fElements[0];
      }
   }
}

//______________________________________________________________________________
template <class Element>
Element TMatrixTSparse<Element>::operator()(Int_t rown,Int_t coln) const
{
   R__ASSERT(this->IsValid());
   if (this->fNrowIndex > 0 && this->fRowIndex[this->fNrowIndex-1] == 0) {
      Error("operator()(Int_t,Int_t) const","row/col indices are not set");
      Info("operator()","fNrowIndex = %d fRowIndex[fNrowIndex-1] = %d\n",this->fNrowIndex,this->fRowIndex[this->fNrowIndex-1]);
      return 0.0;
   }
   const Int_t arown = rown-this->fRowLwb;
   const Int_t acoln = coln-this->fColLwb;

   if (arown >= this->fNrows || arown < 0) {
      Error("operator()","Request row(%d) outside matrix range of %d - %d",rown,this->fRowLwb,this->fRowLwb+this->fNrows);
      return 0.0;
   }
   if (acoln >= this->fNcols || acoln < 0) {
      Error("operator()","Request column(%d) outside matrix range of %d - %d",coln,this->fColLwb,this->fColLwb+this->fNcols);
      return 0.0;
   }

   const Int_t sIndex = fRowIndex[arown];
   const Int_t eIndex = fRowIndex[arown+1];
   const Int_t index = (Int_t)TMath::BinarySearch(eIndex-sIndex,fColIndex+sIndex,acoln)+sIndex;
   if (index < sIndex || fColIndex[index] != acoln) return 0.0;
   else                                             return fElements[index];
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::operator=(const TMatrixTSparse<Element> &source)
{
  // Notice that the sparsity of the matrix is NOT changed : its fRowIndex/fColIndex
  // are used !

   if (gMatrixCheck && !AreCompatible(*this,source)) {
      Error("operator=(const TMatrixTSparse &)","matrices not compatible");
      return *this;
   }

   if (this->GetMatrixArray() != source.GetMatrixArray()) {
      TObject::operator=(source);

      const Element * const sp = source.GetMatrixArray();
            Element * const tp = this->GetMatrixArray();
      memcpy(tp,sp,this->fNelems*sizeof(Element));
      this->fTol = source.GetTol();
   }
   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::operator=(const TMatrixT<Element> &source)
{
  // Notice that the sparsity of the matrix is NOT changed : its fRowIndex/fColIndex
  // are used !

   if (gMatrixCheck && !AreCompatible(*this,(TMatrixTBase<Element> &)source)) {
      Error("operator=(const TMatrixT &)","matrices not compatible");
      return *this;
   }

   if (this->GetMatrixArray() != source.GetMatrixArray()) {
      TObject::operator=(source);

      const Element * const sp = source.GetMatrixArray();
            Element * const tp = this->GetMatrixArray();
      for (Int_t irow = 0; irow < this->fNrows; irow++) {
         const Int_t sIndex = fRowIndex[irow];
         const Int_t eIndex = fRowIndex[irow+1];
         const Int_t off = irow*this->fNcols;
         for (Int_t index = sIndex; index < eIndex; index++) {
            const Int_t icol = fColIndex[index];
            tp[index] = sp[off+icol];
         }
      }
      this->fTol = source.GetTol();
   }
   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::operator=(Element val)
{
  // Assign val to every element of the matrix. Check that the row/col
  // indices are set !

   R__ASSERT(this->IsValid());

   if (fRowIndex[this->fNrowIndex-1] == 0) {
      Error("operator=(Element","row/col indices are not set");
      return *this;
   }

   Element *ep = this->GetMatrixArray();
   const Element * const ep_last = ep+this->fNelems;
   while (ep < ep_last)
      *ep++ = val;

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::operator+=(Element val)
{
  // Add val to every element of the matrix.

   R__ASSERT(this->IsValid());

   Element *ep = this->GetMatrixArray();
   const Element * const ep_last = ep+this->fNelems;
   while (ep < ep_last)
      *ep++ += val;

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::operator-=(Element val)
{
  // Subtract val from every element of the matrix.

   R__ASSERT(this->IsValid());

   Element *ep = this->GetMatrixArray();
   const Element * const ep_last = ep+this->fNelems;
   while (ep < ep_last)
      *ep++ -= val;

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::operator*=(Element val)
{
  // Multiply every element of the matrix with val.

   R__ASSERT(this->IsValid());

   Element *ep = this->GetMatrixArray();
   const Element * const ep_last = ep+this->fNelems;
   while (ep < ep_last)
      *ep++ *= val;

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTBase<Element> &TMatrixTSparse<Element>::Randomize(Element alpha,Element beta,Double_t &seed)
{
  // randomize matrix element values

   R__ASSERT(this->IsValid());

   const Element scale = beta-alpha;
   const Element shift = alpha/scale;

   Int_t   * const pRowIndex = GetRowIndexArray();
   Int_t   * const pColIndex = GetColIndexArray();
   Element * const ep        = GetMatrixArray();

   const Int_t m = this->GetNrows();
   const Int_t n = this->GetNcols();

   // Knuth's algorithm for choosing "length" elements out of nn .
   const Int_t nn     = this->GetNrows()*this->GetNcols();
   const Int_t length = (this->GetNoElements() <= nn) ? this->GetNoElements() : nn;
   Int_t chosen   = 0;
   Int_t icurrent = 0;
   pRowIndex[0] = 0;
   for (Int_t k = 0; k < nn; k++) {
      const Element r = Drand(seed);

      if ((nn-k)*r < length-chosen) {
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

   R__ASSERT(chosen == length);

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &TMatrixTSparse<Element>::RandomizePD(Element alpha,Element beta,Double_t &seed)
{
  // randomize matrix element values but keep matrix symmetric positive definite

   const Element scale = beta-alpha;
   const Element shift = alpha/scale;

   if (gMatrixCheck) {
      R__ASSERT(this->IsValid());

      if (this->fNrows != this->fNcols || this->fRowLwb != this->fColLwb) {
         Error("RandomizePD(Element &","matrix should be square");
         return *this;
      }
   }

   const Int_t n = this->fNcols;

   Int_t   * const pRowIndex = this->GetRowIndexArray();
   Int_t   * const pColIndex = this->GetColIndexArray();
   Element * const ep        = GetMatrixArray();

   // We will always have non-zeros on the diagonal, so there
   // is no randomness there. In fact, choose the (0,0) element now
   pRowIndex[0] = 0;
   pColIndex[0] = 0;
   pRowIndex[1] = 1;
   ep[0]        = 1e-8+scale*(Drand(seed)+shift);

   // Knuth's algorithm for choosing length elements out of nn .
   // nn here is the number of elements in the strict lower triangle.
   const Int_t nn = n*(n-1)/2;

   // length is the number of elements that can be stored, minus the number
   // of elements in the diagonal, which will always be in the matrix.
   //  Int_t length = (this->fNelems-n)/2;
   Int_t length = this->fNelems-n;
   length = (length <= nn) ? length : nn;

   // chosen   : the number of elements that have already been chosen (now 0)
   // nnz      : the number of non-zeros in the matrix (now 1, because the
   //            (0,0) element is already in the matrix.
   // icurrent : the index of the last row whose start has been stored in pRowIndex;

   Int_t chosen   = 0;
   Int_t icurrent = 1;
   Int_t nnz      = 1;
   for (Int_t k = 0; k < nn; k++ ) {
      const Element r = Drand(seed);

      if( (nn-k)*r < length-chosen) {
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

   R__ASSERT(chosen == length);

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
   this->fNelems = nnz;

   TMatrixTSparse<Element> tmp(TMatrixTSparse<Element>::kTransposed,*this);
   *this += tmp;

   // make sure to divide the diagonal by 2 becuase the operation
   // *this += tmp; adds the diagonal again
   {
      const Int_t    * const pR = this->GetRowIndexArray();
      const Int_t    * const pC = this->GetColIndexArray();
             Element * const pD = GetMatrixArray();
      for (Int_t irow = 0; irow < this->fNrows+1; irow++) {
         const Int_t sIndex = pR[irow];
         const Int_t eIndex = pR[irow+1];
         for (Int_t index = sIndex; index < eIndex; index++) {
            const Int_t icol = pC[index];
            if (irow == icol)
               pD[index] /= 2.;
         }
      }
   }

   return *this;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator+(const TMatrixTSparse<Element> &source1,const TMatrixTSparse<Element> &source2)
{
  TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kPlus,source2);
  return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator+(const TMatrixTSparse<Element> &source1,const TMatrixT<Element> &source2)
{
   TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kPlus,source2);
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator+(const TMatrixT<Element> &source1,const TMatrixTSparse<Element> &source2)
{
   TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kPlus,source2);
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator+(const TMatrixTSparse<Element> &source,Element val)
{
   TMatrixTSparse<Element> target(source);
   target += val;
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator+(Element val,const TMatrixTSparse<Element> &source)
{
   TMatrixTSparse<Element> target(source);
   target += val;
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator-(const TMatrixTSparse<Element> &source1,const TMatrixTSparse<Element> &source2)
{
   TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kMinus,source2);
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator-(const TMatrixTSparse<Element> &source1,const TMatrixT<Element> &source2)
{
   TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kMinus,source2);
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator-(const TMatrixT<Element> &source1,const TMatrixTSparse<Element> &source2)
{
   TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kMinus,source2);
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator-(const TMatrixTSparse<Element> &source,Element val)
{
   TMatrixTSparse<Element> target(source);
   target -= val;
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator-(Element val,const TMatrixTSparse<Element> &source)
{
   TMatrixTSparse<Element> target(source);
   target -= val;
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator*(const TMatrixTSparse<Element> &source1,const TMatrixTSparse<Element> &source2)
{
   TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kMult,source2);
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator*(const TMatrixTSparse<Element> &source1,const TMatrixT<Element> &source2)
{
   TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kMult,source2);
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator*(const TMatrixT<Element> &source1,const TMatrixTSparse<Element> &source2)
{
   TMatrixTSparse<Element> target(source1,TMatrixTSparse<Element>::kMult,source2);
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator*(Element val,const TMatrixTSparse<Element> &source)
{
   TMatrixTSparse<Element> target(source);
   target *= val;
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> operator*(const TMatrixTSparse<Element> &source,Element val)
{
   TMatrixTSparse<Element> target(source);
   target *= val;
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &Add(TMatrixTSparse<Element> &target,Element scalar,const TMatrixTSparse<Element> &source)
{
  // Modify addition: target += scalar * source.

   target += scalar * source;
   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &ElementMult(TMatrixTSparse<Element> &target,const TMatrixTSparse<Element> &source)
{
  // Multiply target by the source, element-by-element.

   if (gMatrixCheck && !AreCompatible(target,source)) {
      ::Error("ElementMult(TMatrixTSparse &,const TMatrixTSparse &)","matrices not compatible");
      return target;
   }

   const Element *sp  = source.GetMatrixArray();
         Element *tp  = target.GetMatrixArray();
   const Element *ftp = tp+target.GetNoElements();
   while ( tp < ftp )
      *tp++ *= *sp++;

   return target;
}

//______________________________________________________________________________
template<class Element>
TMatrixTSparse<Element> &ElementDiv(TMatrixTSparse<Element> &target,const TMatrixTSparse<Element> &source)
{
   // Divide target by the source, element-by-element.

   if (gMatrixCheck && !AreCompatible(target,source)) {
      ::Error("ElementDiv(TMatrixT &,const TMatrixT &)","matrices not compatible");
      return target;
   }

   const Element *sp  = source.GetMatrixArray();
         Element *tp  = target.GetMatrixArray();
   const Element *ftp = tp+target.GetNoElements();
   while ( tp < ftp ) {
      if (*sp != 0.0)
         *tp++ /= *sp++;
      else {
         Error("ElementDiv","source element is zero");
         tp++;
      }
   }

   return target;
}

//______________________________________________________________________________
template<class Element>
Bool_t AreCompatible(const TMatrixTSparse<Element> &m1,const TMatrixTSparse<Element> &m2,Int_t verbose)
{
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

//______________________________________________________________________________
template<class Element>
void TMatrixTSparse<Element>::Streamer(TBuffer &R__b)
{
// Stream an object of class TMatrixTSparse.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      Clear();
      R__b.ReadClassBuffer(TMatrixTSparse<Element>::Class(),this,R__v,R__s,R__c);
      if (this->fNelems < 0)
         this->Invalidate();
      } else {
         R__b.WriteClassBuffer(TMatrixTSparse<Element>::Class(),this);
   }
}

template class TMatrixTSparse<Float_t>;

#ifndef ROOT_TMatrixFSparsefwd
#include "TMatrixFSparsefwd.h"
#endif
#ifndef ROOT_TMatrixFfwd
#include "TMatrixFfwd.h"
#endif

template TMatrixFSparse  operator+    <Float_t>(const TMatrixFSparse &source1,const TMatrixFSparse &source2);
template TMatrixFSparse  operator+    <Float_t>(const TMatrixFSparse &source1,const TMatrixF       &source2);
template TMatrixFSparse  operator+    <Float_t>(const TMatrixF       &source1,const TMatrixFSparse &source2);
template TMatrixFSparse  operator+    <Float_t>(const TMatrixFSparse &source ,      Float_t         val    );
template TMatrixFSparse  operator+    <Float_t>(      Float_t         val    ,const TMatrixFSparse &source );
template TMatrixFSparse  operator-    <Float_t>(const TMatrixFSparse &source1,const TMatrixFSparse &source2);
template TMatrixFSparse  operator-    <Float_t>(const TMatrixFSparse &source1,const TMatrixF       &source2);
template TMatrixFSparse  operator-    <Float_t>(const TMatrixF       &source1,const TMatrixFSparse &source2);
template TMatrixFSparse  operator-    <Float_t>(const TMatrixFSparse &source ,      Float_t         val    );
template TMatrixFSparse  operator-    <Float_t>(      Float_t         val    ,const TMatrixFSparse &source );
template TMatrixFSparse  operator*    <Float_t>(const TMatrixFSparse &source1,const TMatrixFSparse &source2);
template TMatrixFSparse  operator*    <Float_t>(const TMatrixFSparse &source1,const TMatrixF       &source2);
template TMatrixFSparse  operator*    <Float_t>(const TMatrixF       &source1,const TMatrixFSparse &source2);
template TMatrixFSparse  operator*    <Float_t>(      Float_t         val    ,const TMatrixFSparse &source );
template TMatrixFSparse  operator*    <Float_t>(const TMatrixFSparse &source,       Float_t         val    );

template TMatrixFSparse &Add          <Float_t>(TMatrixFSparse &target,      Float_t         scalar,const TMatrixFSparse &source);
template TMatrixFSparse &ElementMult  <Float_t>(TMatrixFSparse &target,const TMatrixFSparse &source);
template TMatrixFSparse &ElementDiv   <Float_t>(TMatrixFSparse &target,const TMatrixFSparse &source);

template Bool_t          AreCompatible<Float_t>(const TMatrixFSparse &m1,const TMatrixFSparse &m2,Int_t verbose);

#ifndef ROOT_TMatrixDSparsefwd
#include "TMatrixDSparsefwd.h"
#endif
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif

template class TMatrixTSparse<Double_t>;

template TMatrixDSparse  operator+    <Double_t>(const TMatrixDSparse &source1,const TMatrixDSparse &source2);
template TMatrixDSparse  operator+    <Double_t>(const TMatrixDSparse &source1,const TMatrixD       &source2);
template TMatrixDSparse  operator+    <Double_t>(const TMatrixD       &source1,const TMatrixDSparse &source2);
template TMatrixDSparse  operator+    <Double_t>(const TMatrixDSparse &source ,      Double_t        val    );
template TMatrixDSparse  operator+    <Double_t>(      Double_t        val    ,const TMatrixDSparse &source );
template TMatrixDSparse  operator-    <Double_t>(const TMatrixDSparse &source1,const TMatrixDSparse &source2);
template TMatrixDSparse  operator-    <Double_t>(const TMatrixDSparse &source1,const TMatrixD       &source2);
template TMatrixDSparse  operator-    <Double_t>(const TMatrixD       &source1,const TMatrixDSparse &source2);
template TMatrixDSparse  operator-    <Double_t>(const TMatrixDSparse &source ,      Double_t        val    );
template TMatrixDSparse  operator-    <Double_t>(      Double_t        val    ,const TMatrixDSparse &source );
template TMatrixDSparse  operator*    <Double_t>(const TMatrixDSparse &source1,const TMatrixDSparse &source2);
template TMatrixDSparse  operator*    <Double_t>(const TMatrixDSparse &source1,const TMatrixD       &source2);
template TMatrixDSparse  operator*    <Double_t>(const TMatrixD       &source1,const TMatrixDSparse &source2);
template TMatrixDSparse  operator*    <Double_t>(      Double_t        val    ,const TMatrixDSparse &source );
template TMatrixDSparse  operator*    <Double_t>(const TMatrixDSparse &source,       Double_t        val    );

template TMatrixDSparse &Add          <Double_t>(TMatrixDSparse &target,      Double_t        scalar,const TMatrixDSparse &source);
template TMatrixDSparse &ElementMult  <Double_t>(TMatrixDSparse &target,const TMatrixDSparse &source);
template TMatrixDSparse &ElementDiv   <Double_t>(TMatrixDSparse &target,const TMatrixDSparse &source);

template Bool_t          AreCompatible<Double_t>(const TMatrixDSparse &m1,const TMatrixDSparse &m2,Int_t verbose);
