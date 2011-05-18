// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann  Apr 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TDecompSparse.h"
#include "TMath.h"

ClassImp(TDecompSparse)

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Sparse Symmetric Decomposition class                                  //
//                                                                       //
// Solve a sparse symmetric system of linear equations using a method    //
// based on Gaussian elimination as discussed in Duff and Reid,          //
// ACM Trans. Math. Software 9 (1983), 302-325.                          //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TDecompSparse::TDecompSparse()
{
// Default constructor

   fVerbose = 0;
   InitParam();
   memset(fInfo,0,21*sizeof(Int_t));
}

//______________________________________________________________________________
TDecompSparse::TDecompSparse(Int_t nRows,Int_t nr_nonZeros,Int_t verbose)
{
// Constructor for a matrix with nrows and unspecified number of columns .
// nr_nonZeros is the total number of non-zero entries in the matrix .

   fVerbose = verbose;
   InitParam();

   fNrows     = nRows;
   fNnonZeros = nr_nonZeros;

   fRowFact.Set(nr_nonZeros+1);
   fColFact.Set(nr_nonZeros+1);
   fW      .Set(fNrows+1);
   fIkeep  .Set(3*(fNrows+1));
   fIw     .Set((Int_t)(1.3 * (2*fNnonZeros+3*fNrows+1)+1));
   fIw1    .Set(2*(fNrows+1));

   memset(fInfo,0,21*sizeof(Int_t));

   // These parameters can only be set after sparsity/pivoting pattern is known
   fNsteps = 0;
   fMaxfrt = 0;
}

//______________________________________________________________________________
TDecompSparse::TDecompSparse(Int_t row_lwb,Int_t row_upb,Int_t nr_nonZeros,Int_t verbose)
{
// Constructor for a matrix with row range, [row_lwb..row_upb] and unspecified column
// range . nr_nonZeros is the total number of non-zero entries in the matrix .

   fVerbose = verbose;
   InitParam();

   fRowLwb    = row_lwb;
   fColLwb    = row_lwb;
   fNrows     = row_upb-row_lwb+1;
   fNnonZeros = nr_nonZeros;

   fRowFact.Set(nr_nonZeros+1);
   fColFact.Set(nr_nonZeros+1);
   fW      .Set(fNrows+1);
   fIkeep  .Set(3*(fNrows+1));
   fIw     .Set((Int_t)(1.3 * (2*fNnonZeros+3*fNrows+1)+1));
   fIw1    .Set(2*(fNrows+1));

   memset(fInfo,0,21*sizeof(Int_t));

   // These parameters can only be set after sparsity/pivoting pattern is known
   fNsteps = 0;
   fMaxfrt = 0;
}

//______________________________________________________________________________
TDecompSparse::TDecompSparse(const TMatrixDSparse &a,Int_t verbose)
{
// Constructor for matrix A .

   fVerbose = verbose;

   InitParam();
   SetMatrix(a);

   memset(fInfo,0,21*sizeof(Int_t));
}

//______________________________________________________________________________
TDecompSparse::TDecompSparse(const TDecompSparse &another) : TDecompBase(another)
{
// Copy constructor

   *this = another;
}

//______________________________________________________________________________
Int_t TDecompSparse::NonZerosUpperTriang(const TMatrixDSparse &a)
{
// Static function, returning the number of non-zero entries in the upper triangular matrix .

   const Int_t  rowLwb   = a.GetRowLwb();
   const Int_t  colLwb   = a.GetColLwb();
   const Int_t  nrows    = a.GetNrows();;
   const Int_t *pRowIndex = a.GetRowIndexArray();
   const Int_t *pColIndex = a.GetColIndexArray();

   Int_t nr_nonzeros = 0;
   for (Int_t irow = 0; irow < nrows; irow++ ) {
      const Int_t rown = irow+rowLwb;
      for (Int_t index = pRowIndex[irow]; index < pRowIndex[irow+1]; index++ ) {
         const Int_t coln = pColIndex[index]+colLwb;
         if (coln >= rown) nr_nonzeros++;
      }
   }

   return nr_nonzeros;
}

//______________________________________________________________________________
void TDecompSparse::CopyUpperTriang(const TMatrixDSparse &a,Double_t *b)
{
// Static function, copying the non-zero entries in the upper triangle to
// array b . User should allocate enough memory for array b .

   const Int_t     rowLwb    = a.GetRowLwb();
   const Int_t     colLwb    = a.GetColLwb();
   const Int_t     nrows     = a.GetNrows();;
   const Int_t    *pRowIndex = a.GetRowIndexArray();
   const Int_t    *pColIndex = a.GetColIndexArray();
   const Double_t *pData     = a.GetMatrixArray();

   Int_t nr = 0;
   for (Int_t irow = 0; irow < nrows; irow++ ) {
      const Int_t rown = irow+rowLwb;
      for (Int_t index = pRowIndex[irow]; index < pRowIndex[irow+1]; index++ ) {
         const Int_t coln = pColIndex[index]+colLwb;
         if (coln >= rown) b[nr++] = pData[index];
      }
   }
}

//______________________________________________________________________________
void TDecompSparse::SetMatrix(const TMatrixDSparse &a)
{
// Set matrix to be decomposed .

   ResetStatus();

   fA.Use(*const_cast<TMatrixDSparse *>(&a));
   fRowLwb    = fA.GetRowLwb();
   fColLwb    = fA.GetColLwb();
   fNrows     = fA.GetNrows();
   fNnonZeros = NonZerosUpperTriang(a);

   fRowFact.Set(fNnonZeros+1);
   fColFact.Set(fNnonZeros+1);

   const Int_t *rowIndex = a.GetRowIndexArray();
   const Int_t *colIndex = a.GetColIndexArray();

   Int_t nr = 0;
   for (Int_t irow = 0; irow < fNrows; irow++ ) {
      const Int_t rown = irow+fRowLwb;
      for (Int_t index = rowIndex[irow]; index < rowIndex[irow+1]; index++ ) {
         const Int_t coln = colIndex[index]+fColLwb;
         if (coln >= rown) {
            fRowFact[nr+1] = irow+1;
            fColFact[nr+1] = colIndex[index]+1;
            nr++;
         }
      }
   }

   fW    .Set(fNrows+1);
   fIkeep.Set(3*(fNrows+1));
   fIw   .Set((Int_t)(1.3 * (2*fNnonZeros+3*fNrows+1)+1));
   fIw1  .Set(2*(fNrows+1));

   // Determine pivot sequence, set iflag = 0 in order to make InitPivot choose the order.
   Int_t iflag = 0;
   Double_t ops;
   InitPivot(fNrows,fNnonZeros,fRowFact,fColFact,fIw,fIkeep,fIw1,fNsteps,iflag,
             fIcntl,fCntl,fInfo,ops);

   switch ( this->ErrorFlag() ) {
      case -1 :
         Error("SetMatrix(const TMatrixDSparse &","nRows  = %d out of range",fNrows);
         return;
      case -2 :
         Error("SetMatrix(const TMatrixDSparse &","nr_nonzeros  = %d out of range",fNnonZeros);
         return;
      case -3 :
         Error("SetMatrix(const TMatrixDSparse &",
               "insufficient space in fIw of %d suggest reset to %d",fIw.GetSize(),this->IError());
         return;
      case 1 :
         Error("SetMatrix(const TMatrixDSparse &",
               "detected %d entries out of rage in row/col indices; ignored",this->IError());
         return;
   }

   // set fIw and fIw1 in prep for calls to Factor and Solve

//   fIw  .Set((Int_t) 1.2*this->MinRealWorkspace()+1);
   fIw  .Set((Int_t) 3*this->MinRealWorkspace()+1);
   fIw1 .Set(fNrows+1);
   fIw2 .Set(fNsteps+1);
//   fFact.Set((Int_t) 1.2*this->MinRealWorkspace()+1);
   fFact.Set((Int_t) 3*this->MinRealWorkspace()+1);

   SetBit(kMatrixSet);
}

//______________________________________________________________________________
Bool_t TDecompSparse::Decompose()
{
// Decomposition engine .
// If the decomposition succeeds, bit kDecomposed is set .

   if (TestBit(kDecomposed)) return kTRUE;

   if ( !TestBit(kMatrixSet) ) {
      Error("Decompose()","Matrix has not been set");
      return kFALSE;
   }

   Int_t done = 0; Int_t tries = 0;
   do {
      fFact[0] = 0.;
      CopyUpperTriang(fA,fFact.GetArray()+1);

      Factor(fNrows,fNnonZeros,fRowFact,fColFact,fFact,fIw,fIkeep,
             fNsteps,fMaxfrt,fIw1,fIcntl,fCntl,fInfo);

      switch ( this->ErrorFlag() ) {
         case 0 :
            done = 1;
            break;
         case -1 :
            Error("Decompose()","nRows  = %d out of range",fNrows);
            return kFALSE;
         case -2 :
            Error("Decompose()","nr_nonzeros  = %d out of range",fNnonZeros);
            return kFALSE;
         case -3 :
            {
               if (fVerbose)
                  Info("Decompose()","insufficient space of fIw: %d",fIw.GetSize());
               const Int_t nIw_old = fIw.GetSize();
               const Int_t nIw = (this->IError() > fIPessimism*nIw_old) ? this->IError() :
                                                                         (Int_t)(fIPessimism*nIw_old);
               fIw.Set(nIw);
               if (fVerbose)
                  Info("Decompose()","resetting to fIw: %d",nIw);
               fIPessimism *= 1.1;
               break;
            }
         case -4 :
            {
               if (fVerbose)
                  Info("Decompose()","insufficient factorization space: %d",fFact.GetSize());
               const Int_t nFact_old = fFact.GetSize();
               const Int_t nFact = (this->IError() > fRPessimism*nFact_old) ? this->IError() :
                                                                             (Int_t) (fRPessimism*nFact_old);
               fFact.Set(nFact); fFact.Reset(0.0);
               CopyUpperTriang(fA,fFact.GetArray()+1);
               if (fVerbose)
                  Info("Decompose()","reseting to: %d",nFact);
               fRPessimism *= 1.1;
               break;
            }
         case -5 :
            if (fVerbose) {
               Info("Decompose()","matrix apparently numerically singular");
               Info("Decompose()","detected at stage %d",this->IError());
               Info("Decompose()","accept this factorization and hope for the best..");
            }
            done = 1;
            break;
         case -6 :
            if (fVerbose) {
               Info("Decompose()","change of sign of pivots detected at stage %d",this->IError());
               Info("Decompose()","but who cares ");
            }
            done = 1;
            break;
         case -7 :
            Error("Decompose()","value of fNsteps out of range: %d",fNsteps);
            return kFALSE;
         case 1 :
            if (fVerbose) {
               Info("Decompose()","detected %d entries out of range in row/column index",this->IError());
               Info("Decompose()","they are ignored");
            }
            done = 1;
            break;
         case 3 :
            if (fVerbose)
               Info("Decompose()","rank deficient matrix detected; apparent rank = %d",this->IError());
            done = 1;
            break;
         default:
            break;
      }

      tries++;
   } while (!done && tries < 10);

   Int_t ok;
   if ( !done && tries >= 10) {
      ok = kFALSE;
      if (fVerbose)
         Error("Decompose()","did not get a factorization after 10 tries");
   } else {
      ok = kTRUE;
      SetBit(kDecomposed);
   }

   return ok;
}

//______________________________________________________________________________
Bool_t TDecompSparse::Solve(TVectorD &b)
{
// Solve Ax=b . Solution returned in b.

   R__ASSERT(b.IsValid());
   if (TestBit(kSingular)) {
      Error("Solve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("Solve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fNrows != b.GetNrows() || fRowLwb != b.GetLwb())
   {
      Error("Solve(TVectorD &","vector and matrix incompatible");
      return kFALSE;
   }
   b.Shift(-fRowLwb); // make sure rowlwb = 0

   // save bs and store residuals
   TVectorD resid = b;
   TVectorD bSave = b;

   Double_t bnorm = b.NormInf();
   Double_t rnorm = 0.0;

   Int_t done = 0;
   Int_t refactorizations = 0;

   while (!done && refactorizations < 10) {

      Solve(fNrows,fFact,fIw,fW,fMaxfrt,b,fIw1,fNsteps,fIcntl,fInfo);

      // compute residuals
      resid = fA*b-resid;
      rnorm = resid.NormInf();

      if (rnorm < fPrecision*(1.+bnorm)) {
         // residuals are small enough, use this solution
         done = 1;
      } else if (this->GetThresholdPivoting() >= kThresholdPivotingMax
                  || refactorizations > 10)  {
         // ThresholdPivoting parameter is already too high; give up and
         // use this solution, whatever it is (the algorithm may bomb in
         // an outer loop).
         done = 1;
      } else {
         // refactor with a higher Threshold Pivoting parameter
         Double_t tp = this->GetThresholdPivoting();
         tp *= kThresholdPivotingFactor;
         if (tp > kThresholdPivotingMax) tp = kThresholdPivotingMax;
         this->SetThresholdPivoting(tp);
         if (fVerbose)
            Info("Solve","Setting ThresholdPivoting parameter to %.4e for future factorizations",
                  this->GetThresholdPivoting());

         SetMatrix(fA);
         refactorizations++;
         resid = bSave;
         b     = bSave;
      }
   }

   b.Shift(fRowLwb);
   return kTRUE;
}

//______________________________________________________________________________
void TDecompSparse::InitParam()
{
// initializing control parameters

   fPrecision  = kInitPrecision;
   fIPessimism = 1.2;
   fRPessimism = 1.2;

   const Int_t ifrlvl = 5;

   SetVerbose(fVerbose);
   fIcntl[4] = 2139062143;
   fIcntl[5] = 1;
   fIcntl[ifrlvl+1]  = 32639;
   fIcntl[ifrlvl+2]  = 32639;
   fIcntl[ifrlvl+3]  = 32639;
   fIcntl[ifrlvl+4]  = 32639;
   fIcntl[ifrlvl+5]  = 14;
   fIcntl[ifrlvl+6]  = 9;
   fIcntl[ifrlvl+7]  = 8;
   fIcntl[ifrlvl+8]  = 8;
   fIcntl[ifrlvl+9]  = 9;
   fIcntl[ifrlvl+10] = 10;
   fIcntl[ifrlvl+11] = 32639;
   fIcntl[ifrlvl+12] = 32639;
   fIcntl[ifrlvl+13] = 32639;
   fIcntl[ifrlvl+14] = 32689;
   fIcntl[ifrlvl+15] = 24;
   fIcntl[ifrlvl+16] = 11;
   fIcntl[ifrlvl+17] = 9;
   fIcntl[ifrlvl+18] = 8;
   fIcntl[ifrlvl+19] = 9;
   fIcntl[ifrlvl+20] = 10;
   fIcntl[26] = 0;
   fIcntl[27] = 0;
   fIcntl[28] = 0;
   fIcntl[29] = 0;
   fIcntl[30] = 0;
   fCntl[1] = 0.10;
   fCntl[2] = 1.00;
   fCntl[3] = 0.00;
   fCntl[4] = 0.0;
   fCntl[5] = 0.0;

   // set initial value of "Treat As Zero" parameter
   this->SetTreatAsZero(kInitTreatAsZero);

   // set initial value of Threshold parameter
   this->SetThresholdPivoting(kInitThresholdPivoting);

   fNsteps    = 0;
   fMaxfrt    = 0;
   fNrows     = 0;
   fNnonZeros = 0;
}

//______________________________________________________________________________
void TDecompSparse::InitPivot(const Int_t n,const Int_t nz,TArrayI &Airn,TArrayI &Aicn,
                              TArrayI &Aiw,TArrayI &Aikeep,TArrayI &Aiw1,Int_t &nsteps,
                              const Int_t iflag,Int_t *icntl,Double_t *cntl,Int_t *info,
                              Double_t &ops)
{
// Setup Pivoting variables

   Int_t i,iwfr,k,l1,l2,lliw;

   Int_t *irn      = Airn.GetArray();
   Int_t *icn      = Aicn.GetArray();
   Int_t *iw       = Aiw.GetArray();
   Int_t *ikeep    = Aikeep.GetArray();
   Int_t *iw1      = Aiw1.GetArray();
   const Int_t liw = Aiw.GetSize()-1;

   for (i = 1; i < 16; i++)
      info[i] = 0;

   if (icntl[3] > 0 && icntl[2] > 0) {
      ::Info("TDecompSparse::InitPivot","Start with n = %d  nz = %d  liw = %d  iflag = %d",n,nz,liw,iflag);
      nsteps = 0;
      k = TMath::Min(8,nz);
      if (icntl[3] > 1) k = nz;
      if (k > 0) {
         printf("matrix non-zeros:\n");
         for (i = 1; i < k+1; i++) {
            printf("%d %d ",irn[i],icn[i]);
            if (i%5 == 0 || i == k) printf("\n");
         }
      }

      k = TMath::Min(10,n);
      if (icntl[3] > 1) k = n;
      if (iflag == 1 && k > 0) {
         for (i = 1; i < k+1; i++) {
            printf("%d ",ikeep[i]);
            if (i%10 == 0 || i == k) printf("\n");
         }
      }
   }

   if (n >= 1 && n <= icntl[4]) {
      if (nz < 0) {
         info[1] = -2;
         if (icntl[1] > 0)
            ::Error("TDecompSparse::InitPivot","info[1]= %d; value of nz out of range .. = %d",info[1],nz);
         return;
      }
      lliw = liw-2*n;
      l1 = lliw+1;
      l2 = l1+n;
      if (iflag != 1) {
         if (liw < 2*nz+3*n+1) {
            info[1] = -3;
            info[2] = 2*nz+3*n+1;
            if (icntl[1] > 0)
               ::Error("TDecompSparse::InitPivot","info[1]= %d; liw too small, must be increased from %d to at least %d",info[1],liw,info[2]);
            return;
         }
         InitPivot_sub1(n,nz,irn,icn,iw,iw1,iw1+n+1,iw+l1-1,iwfr,icntl,info);
         InitPivot_sub2(n,iw1,iw,lliw,iwfr,iw+l1-1,iw+l2-1,ikeep+n+1,
                        ikeep+2*(n+1),ikeep,icntl[4],info[11],cntl[2]);
      } else {
         if (liw < nz+3*n+1) {
            info[1] = -3;
            info[2] = nz+3*n+1;
            if (icntl[1] > 0)
               ::Error("TDecompSparse::InitPivot","info[1]= %d; liw too small, must be increased from %d to at least %d",info[1],liw,info[2]);
            return;
         }
         InitPivot_sub3(n,nz,irn,icn,ikeep,iw,iw1,iw1+n+1,iw+l1-1,iwfr,icntl,info);
         InitPivot_sub4(n,iw1,iw,lliw,iwfr,ikeep,ikeep+n+1,iw+l1-1,iw+l2-1,info[11]);
      }
      InitPivot_sub5(n,iw1,iw+l1-1,ikeep,ikeep+n+1,ikeep+2*(n+1),iw+l2-1,nsteps,icntl[5]);
      if (nz >= 1) iw[1] = irn[1]+1;
      InitPivot_sub6(n,nz,irn,icn,ikeep,ikeep+2*(n+1),ikeep+n+1,iw+l2-1,
                     nsteps,iw1,iw1+n+1,iw,info,ops);
   } else {
      info[1] = -1;
      if (icntl[1] > 0)
         ::Error("TDecompSparse::InitPivot","info[1]= %d; value of n out of range ... = %d",info[1],n);
      return;
   }

   if (icntl[3] <= 0 || icntl[2] <= 0) return;

   printf("Leaving with nsteps =%d info(1)=%d ops=%14.5e ierror=%d\n",nsteps,info[1],ops,info[2]);
   printf("nrltot=%d nirtot=%d nrlnec=%d nirnec=%d nrladu=%d niradu=%d ncmpa=%d\n",
           info[3],info[4],info[5],info[6],info[7],info[8],info[11]);

   k = TMath::Min(9,n);
   if (icntl[3] > 1) k = n;
   if (k > 0) {
      printf("ikeep[0][.]=\n");
      for (i = 1; i < k+1; i++) {
         printf("%d ",ikeep[i]);
         if (k%10 == 0 || i == k) printf("\n");
      }
   }
   k = TMath::Min(k,nsteps);
   if (k > 0) {
      printf("ikeep[2][.]=\n");
      for (i = 1; i < k+1; i++) {
         printf("%d ",ikeep[2*(n+1)+i]);
         if (k%10 == 0 || i == k) printf("\n");
      }
   }
}

//______________________________________________________________________________
void TDecompSparse::Factor(const Int_t n,const Int_t nz,TArrayI &Airn,TArrayI &Aicn,TArrayD &Aa,
                           TArrayI &Aiw,TArrayI &Aikeep,const Int_t nsteps,Int_t &maxfrt,
                           TArrayI &Aiw1,Int_t *icntl,Double_t *cntl,Int_t *info)
{
// Factorization routine, the workhorse for the decompostion step

   Int_t i,iapos,iblk,ipos,irows,j1,j2,jj,k,kblk,kz,len,ncols,nrows,nz1;

   Int_t    *irn   = Airn.GetArray();
   Int_t    *icn   = Aicn.GetArray();
   Int_t    *iw    = Aiw.GetArray();
   Int_t    *ikeep = Aikeep.GetArray();
   Int_t    *iw1   = Aiw1.GetArray();
   Double_t *a     = Aa.GetArray();

   const Int_t la = Aa.GetSize()-1;
   const Int_t liw = Aiw.GetSize()-1;

   info[1] = 0;
   if (icntl[3] > 0 && icntl[2] > 0) {
      printf("entering Factor with n=%d nz=%d la=%d liw=%d nsteps=%d u=%10.2e\n",
               n,nz,la,liw,nsteps,cntl[1]);
      kz = TMath::Min(6,nz);
      if (icntl[3] > 1) kz = nz;
      if (nz > 0) {
         printf("matrix non-zeros:\n");
         for (i = 1; i < kz+1; i++) {
            printf("%16.3e %d %d ",a[i],irn[i],icn[i]);
            if (i%2 == 0 || i==kz) printf("\n");
         }
      }
      k = TMath::Min(9,n);
      if (icntl[3] > 1) k = n;
      if (k > 0) {
         printf("ikeep(0,.)=\n");
         for (i = 1; i < k+1; i++) {
            printf("%d ",ikeep[i]);
            if (i%10 == 0 || i == k) printf("\n");
         }
      }
      k = TMath::Min(k,nsteps);
      if (k > 0) {
         printf("ikeep(1,.)=\n");
         for (i = 1; i < k+1; i++) {
            printf("%d ",ikeep[n+1+i]);
            if (i%10 == 0 || i == k) printf("\n");
         }
         printf("ikeep(2,.)=\n");
         for (i = 1; i < k+1; i++) {
            printf("%d ",ikeep[2*(n+1)+i]);
            if (i%10 == 0 || i == k) printf("\n");
         }
      }
   }

   if (n < 1 || n > icntl[4])
      info[1] = -1;
   else if (nz < 0)
      info[1] = -2;
   else if (liw < nz) {
      info[1] = -3;
      info[2] = nz;
   } else if (la < nz+n) {
      info[1] = -4;
      info[2] = nz+n;
   } else if (nsteps < 1 || nsteps > n)
      info[1] = -7;
   else {
      Factor_sub1(n,nz,nz1,a,la,irn,icn,iw,liw,ikeep,iw1,icntl,info);
      if (info[1] != -3 && info[1] != -4) {
         Factor_sub2(n,nz1,a,la,iw,liw,ikeep,ikeep+2*(n+1),nsteps,maxfrt,ikeep+n+1,iw1,icntl,cntl,info);
         if (info[1] == 3 && icntl[2] > 0)
            ::Warning("TDecompSparse::Factor","info[1]= %d; matrix is singular. rank=%d",info[1],info[2]);
      }
   }

   if (icntl[1] > 0) {
      switch(info[1]) {
         case -1:
            ::Error("TDecompSparse::Factor","info[1]= %d; value of n out of range ... =%d",info[1],n);
            break;

         case -2:
            ::Error("TDecompSparse::Factor","info[1]= %d; value of nz out of range ... =%d",info[1],nz);
            break;

         case -3:
            ::Error("TDecompSparse::Factor","info[1]= %d; liw too small, must be increased from %d to at least %d",info[1],liw,info[2]);
            break;

         case -4:
            ::Error("TDecompSparse::Factor","info[1]= %d; la too small, must be increased from %d to at least %d",info[1],la,info[2]);
            break;

         case -5:
            ::Error("TDecompSparse::Factor","info[1]= %d; zero pivot at stage %d zero pivot at stage",info[1],info[2]);
            break;

         case -6:
            ::Error("TDecompSparse::Factor","info[1]= %d; change in sign of pivot encountered when factoring allegedly definite matrix",info[1]);
            break;

         case -7:
            ::Error("TDecompSparse::Factor","info[1]= %d; nsteps is out of range",info[1]);
            break;
      }
   }

   if (icntl[3] <= 0 || icntl[2] <= 0 || info[1] < 0)
      return;

   ::Info("TDecompSparse::Factor","leaving Factor with maxfrt=%d info[1]=%d nrlbdu=%d nirbdu=%d ncmpbr=%d ncmpbi=%d ntwo=%d ierror=%d",
          maxfrt,info[1],info[9],info[10],info[12],info[13],info[14],info[2]);

   if (info[1] < 0) return;

   kblk = TMath::Abs(iw[1]+0);
   if (kblk == 0) return;
   if (icntl[3] == 1) kblk = 1;
   ipos = 2;
   iapos = 1;

   for (iblk = 1; iblk < kblk+1; iblk++) {
      ncols = iw[ipos];
      nrows = iw[ipos+1];
      j1 = ipos+2;
      if (ncols <= 0) {
         ncols = -ncols;
         nrows = 1;
         j1 = j1-1;
      }
      ::Info("TDecompSparse::Factor","block pivot =%d nrows =%d ncols =%d",iblk,nrows,ncols);
      j2 = j1+ncols-1;
      ipos = j2+1;

      printf(" column indices =\n");
      for (jj = j1; jj < j2+1; jj++) {
         printf("%d ",iw[jj]);
         if (jj%10 == 0 || jj == j2) printf("\n");
      }

      printf(" real entries .. each row starts on a new line\n");
      len = ncols;
      for (irows = 1; irows < nrows+1; irows++) {
         j1 = iapos;
         j2 = iapos+len-1;
         for (jj = j1; jj < j2+1; jj++) {
            printf("%13.4e ",a[jj]);
            if (jj%5 == 0 || jj == j2) printf("\n");
         }
         len = len-1;
         iapos = j2+1;
      }
   }
}

//______________________________________________________________________________
void TDecompSparse::Solve(const Int_t n,TArrayD &Aa,TArrayI &Aiw,
                          TArrayD &Aw,const Int_t maxfrt,TVectorD &b,TArrayI &Aiw1,
                          const Int_t nsteps,Int_t *icntl,Int_t *info)
{
// Main routine for solving Ax=b

   Int_t i,iapos,iblk,ipos,irows,j1,j2,jj,k,kblk,latop,len,nblk,ncols,nrows;

   Double_t *a   = Aa.GetArray();
   Double_t *w   = Aw.GetArray();
   Int_t    *iw  = Aiw.GetArray();
   Int_t    *iw1 = Aiw1.GetArray();
   Double_t *rhs = new Double_t[n+1];
   rhs[0] = 0.;
   memcpy(rhs+1,b.GetMatrixArray(),n*sizeof(Double_t));
   const Int_t la  = Aa.GetSize()-1;
   const Int_t liw = Aiw.GetSize()-1;

   info[1] = 0;
   k = 0;
   if (icntl[3] > 0 && icntl[2] > 0) {
      printf("nentering Solve with n=%d la=%d liw=%d maxfrt=%d nsteps=%d",n,la,liw,maxfrt,nsteps);

      kblk = TMath::Abs(iw[1]+0);
      if (kblk != 0) {
         if (icntl[3] == 1) kblk = 1;
         ipos = 2;
         iapos = 1;
         for (iblk = 1; iblk < kblk+1; iblk++) {
            ncols = iw[ipos];
            nrows = iw[ipos+1];
            j1 = ipos+2;
            if (ncols <= 0) {
               ncols = -ncols;
               nrows = 1;
               j1 = j1-1;
            }
            printf("block pivot=%d nrows=%d ncols=%d\n",iblk,nrows,ncols);
            j2 = j1+ncols-1;
            ipos = j2+1;
            printf("column indices =\n");
            for (jj = j1; jj < j2+1; jj++) {
               printf("%d ",iw[jj]);
               if (jj%10 == 0 || jj == j2) printf("\n");
            }
            printf("real entries .. each row starts on a new line\n");
            len = ncols;
            for (irows = 1; irows < nrows+1; irows++) {
               j1 = iapos;
               j2 = iapos+len-1;
               for (jj = j1; jj < j2+1; jj++) {
                  printf("%13.3e ",a[jj]);
                  if (jj%5 == 0 || jj == j2) printf("\n");
               }
               len = len-1;
               iapos = j2+1;
            }
         }
      }

      k = TMath::Min(10,n);
      if (icntl[3] > 1) k = n;
      if (n > 0) {
         printf("rhs =\n");
         for (i = 1; i < k+1; i++) {
            printf("%13.3e ",rhs[i]);
            if (i%5 == 0 || i == k) printf("\n");
         }
      }
   }

   nblk = 0;
   if (iw[1] == 0) {
      nblk = 0;
      for (i = 1; i < n+1; i++)
         rhs[i] = 0.0;
   } else {
      nblk = (iw[1] <= 0) ? -iw[1] : iw[1];
      Solve_sub1(n,a,iw+1,w,rhs,iw1,nblk,latop,icntl);
      Solve_sub2(n,a,iw+1,w,rhs,iw1,nblk,latop,icntl);
   }

   if (icntl[3] > 0 && icntl[2] > 0) {
      printf("leaving Solve with:\n");
      if (n > 0) {
         printf("rhs =\n");
         for (i = 1; i < k+1; i++) {
            printf("%13.3e ",rhs[i]);
            if (i%5 == 0 || i == k) printf("\n");
         }
      }
   }

   memcpy(b.GetMatrixArray(),rhs+1,n*sizeof(Double_t));
   delete [] rhs;
}

//______________________________________________________________________________
void TDecompSparse::InitPivot_sub1(const Int_t n,const Int_t nz,Int_t *irn,Int_t *icn,
                                   Int_t *iw,Int_t *ipe,Int_t *iq,Int_t *flag,
                                   Int_t &iwfr,Int_t *icntl,Int_t *info)
{
// Help routine for pivoting setup

   Int_t i,id,j,jn,k,k1,k2,l,last,lr,n1,ndup;

   info[2] = 0;
   for (i = 1; i < n+1; i++)
      ipe[i] = 0;
   lr = nz;

   if (nz != 0) {
      for (k = 1; k < nz+1; k++) {
         i = irn[k];
         j = icn[k];

         Bool_t outRange = (i < 1 || i > n || j < 1 || j > n);
         if (outRange) {
            info[2] = info[2]+1;
            info[1] = 1;
            if (info[2] <= 1 && icntl[2]> 0)
               ::Warning("TDecompSparse::InitPivot_sub1","info[1]= %d; %d th non-zero (in row=%d and column=%d) ignored",info[1],k,i,j);
         }

         if (outRange || i == j) {
            i = 0;
            j = 0;
         } else {
            ipe[i] = ipe[i]+1;
            ipe[j] = ipe[j]+1;
         }
         iw[k] = j;
         lr = lr+1;
         iw[lr] = i;
      }
   }

   iq[1] = 1;
   n1 = n-1;
   if (n1 > 0) {
      for (i = 1; i < n1+1; i++) {
         flag[i] = 0;
         if (ipe[i] == 0) ipe[i] = -1;
         iq[i+1] = ipe[i]+iq[i]+1;
         ipe[i] = iq[i];
      }
   }

   last = ipe[n]+iq[n];
   flag[n] = 0;
   if (lr < last) {
      k1 = lr+1;
      for (k = k1; k < last+1; k++)
         iw[k] = 0;
   }
   ipe[n] = iq[n];
   iwfr = last+1;
   if (nz != 0) {
      for (k = 1; k < nz+1; k++) {
         j = iw[k];
         if (j <= 0)  continue;
         l = k;
         iw[k] = 0;
         for (id = 1; id < nz+1; id++) {
            if (l <= nz)
               l = l+nz;
            else
               l = l-nz;
            i = iw[l];
            iw[l] = 0;
            if (i >= j) {
               l = iq[j]+1;
               iq[j] = l;
               jn = iw[l];
               iw[l] = -i;
            } else {
               l = iq[i]+1;
               iq[i] = l;
               jn = iw[l];
               iw[l] = -j;
            }
            j = jn;
            if (j <= 0) break;
         }
      }
   }

   ndup = 0;

   for (i = 1; i < n+1; i++) {
      k1 = ipe[i]+1;
      k2 = iq[i];
      if (k1 > k2) {
         ipe[i] = 0;
         iq[i] = 0;
      } else {
         for (k = k1; k < k2+1; k++) {
            j = -iw[k];
            if (j <= 0) break;
            l = iq[j]+1;
            iq[j] = l;
            iw[l] = i;
            iw[k] = j;
            if (flag[j] == i) {
               ndup = ndup + 1;
               iw[l] = 0;
               iw[k] = 0;
            }
            flag[j] = i;
         }

         iq[i] = iq[i]-ipe[i];
         if (ndup == 0) iw[k1-1] = iq[i];
      }
   }

   if (ndup != 0) {
      iwfr = 1;
      for (i = 1; i < n+1; i++) {
         k1 = ipe[i]+1;
         if (k1 == 1) continue;
         k2 = iq[i]+ipe[i];
         l = iwfr;
         ipe[i] = iwfr;
         iwfr = iwfr+1;
         for (k = k1; k < k2+1; k++) {
            if (iw[k] == 0) continue;
            iw[iwfr] = iw[k];
            iwfr = iwfr+1;
         }
         iw[l] = iwfr-l-1;
      }
   }

}

//______________________________________________________________________________
void TDecompSparse::InitPivot_sub2(const Int_t n,Int_t *ipe,Int_t *iw,const Int_t lw,
                                   Int_t &iwfr,Int_t *nv,Int_t *nxt,Int_t *lst,Int_t *ipd,
                                   Int_t *flag,const Int_t iovflo,Int_t &ncmpa,
                                   const Double_t fratio)
{
// Help routine for pivoting setup

   Int_t i,id,idl,idn,ie,ip,is,jp,jp1,jp2,js,k,k1,k2,ke,kp,kp0,kp1,
         kp2,ks,l,len,limit,ln,ls,lwfr,md,me,ml,ms,nel,nflg,np,
         np0,ns,nvpiv,nvroot,root;

   for (i = 1; i < n+1; i++) {
      ipd[i]  = 0;
      nv[i]   = 1;
      flag[i] = iovflo;
   }

   js = 0;
   ms = 0;
   ncmpa = 0;
   md     = 1;
   nflg   = iovflo;
   nel    = 0;
   root   = n+1;
   nvroot = 0;
   for (is = 1; is < n+1; is++) {
      k = ipe[is];
      if (k > 0) {
         id = iw[k]+1;
         ns = ipd[id];
         if (ns > 0) lst[ns] = is;
         nxt[is] = ns;
         ipd[id] = is;
         lst[is] = -id;
      } else {
         nel = nel+1;
         flag[is] = -1;
         nxt[is]  = 0;
         lst[is]  = 0;
      }
   }

   for (ml = 1; ml < n+1; ml++) {
      if (nel+nvroot+1 >= n) break;
      for (id = md; id < n+1; id++) {
         ms = ipd[id];
         if (ms > 0) break;
      }

      md = id;
      nvpiv = nv[ms];
      ns = nxt[ms];
      nxt[ms] = 0;
      lst[ms] = 0;
      if (ns > 0) lst[ns] = -id;
      ipd[id] = ns;
      me = ms;
      nel = nel+nvpiv;
      idn = 0;
      kp = ipe[me];
      flag[ms] = -1;
      ip = iwfr;
      len = iw[kp];
      jp = 0;
      for (kp1 = 1; kp1 < len+1; kp1++) {
         kp = kp+1;
         ke = iw[kp];
         if (flag[ke] > -2) {
            if (flag[ke] <= 0) {
               if (ipe[ke] != -root) continue;
               ke = root;
               if (flag[ke] <= 0) continue;
            }
            jp = kp-1;
            ln = len-kp1+1;
            ie = ms;
         } else {
            ie = ke;
            jp = ipe[ie];
            ln = iw[jp];
         }

         for (jp1 = 1; jp1 < ln+1; jp1++) {
            jp = jp+1;
            is = iw[jp];
            if (flag[is] <= 0) {
               if (ipe[is] == -root) {
                  is = root;
                  iw[jp] = root;
                  if (flag[is] <= 0) continue;
               } else
               continue;
            }
            flag[is] = 0;
            if (iwfr >= lw) {
               ipe[ms] = kp;
               iw[kp] = len-kp1;
               ipe[ie] = jp;
               iw[jp] = ln-jp1;
               InitPivot_sub2a(n,ipe,iw,ip-1,lwfr,ncmpa);
               jp2 = iwfr-1;
               iwfr = lwfr;
               if (ip <= jp2) {
                  for (jp = ip; jp < jp2+1; jp++) {
                     iw[iwfr] = iw[jp];
                     iwfr = iwfr+1;
                  }
               }
               ip = lwfr;
               jp = ipe[ie];
               kp = ipe[me];
            }
            iw[iwfr] = is;
            idn = idn+nv[is];
            iwfr = iwfr+1;
            ls = lst[is];
            lst[is] = 0;
            ns = nxt[is];
            nxt[is] = 0;
            if (ns > 0) lst[ns] = ls;
            if (ls < 0) {
               ls = -ls;
               ipd[ls] = ns;
            } else if (ls > 0)
               nxt[ls] = ns;
         }

         if (ie == ms)
            break;
         ipe[ie] = -me;
         flag[ie] = -1;
      }

      nv[ms] = idn+nvpiv;
      if (iwfr != ip) {
         k1 = ip;
         k2 = iwfr-1;
         limit = TMath::Nint(fratio*(n-nel));

         for (k = k1; k < k2+1; k++) {
            is = iw[k];
            if (is == root) continue;
            if (nflg <= 2)  {
               for (i = 1; i < n+1; i++) {
                  if (flag[i] > 0)   flag[i] =  iovflo;
                  if (flag[i] <= -2) flag[i] = -iovflo;
               }
               nflg = iovflo;
            }
            nflg = nflg-1;
            id = idn;
            kp1 = ipe[is]+1;
            np  = kp1;
            kp2 = iw[kp1-1]+kp1-1;

            Int_t skip = 0;
            for (kp = kp1; kp < kp2+1; kp++) {
               ke = iw[kp];
               if (flag[ke] == -1) {
                  if (ipe[ke] != -root) continue;
                  ke = root;
                  iw[kp] = root;
                  if (flag[ke] == -1) continue;
               }
               if (flag[ke] >= 0) {
                  skip = 1;
                  break;
               }
               jp1 = ipe[ke]+1;
               jp2 = iw[jp1-1]+jp1-1;
               idl = id;
               for (jp = jp1; jp < jp2+1; jp++) {
                  js = iw[jp];
                  if (flag[js] <= nflg) continue;
                  id = id+nv[js];
                  flag[js] = nflg;
               }
               if (id <= idl) {
                  Int_t skip2 = 0;
                  for (jp = jp1; jp < jp2+1; jp++) {
                     js = iw[jp];
                     if (flag[js] != 0) {
                        skip2 = 1;
                        break;
                     }
                  }
                  if (skip2) {
                     iw[np] = ke;
                     flag[ke] = -nflg;
                     np = np+1;
                  } else {
                     ipe[ke] = -me;
                     flag[ke] = -1;
                  }
               } else {
                  iw[np] = ke;
                  flag[ke] = -nflg;
                  np = np+1;
               }
            }

            if (!skip)
               np0 = np;
            else {
               np0 = np;
               kp0 = kp;
               for (kp = kp0; kp < kp2+1; kp++) {
                  ks = iw[kp];
                  if (flag[ks] <= nflg) {
                     if (ipe[ks] == -root) {
                        ks = root;
                        iw[kp] = root;
                        if (flag[ks] <= nflg) continue;
                     } else
                        continue;
                  }
                  id = id+nv[ks];
                  flag[ks] = nflg;
                  iw[np] = ks;
                  np = np+1;
               }
            }
 
            Int_t doit = 2;
            if (id < limit) {
               iw[np] = iw[np0];
               iw[np0] = iw[kp1];
               iw[kp1] = me;
               iw[kp1-1] = np-kp1+1;
               js = ipd[id];
               for (l = 1; l < n+1; l++) {
                  if (js <= 0) {
                     doit = 3;
                     break;
                  }
                  kp1 = ipe[js]+1;
                  if (iw[kp1] != me) {
                     doit = 3;
                     break;
                  }
                  kp2 = kp1-1+iw[kp1-1];
                  Int_t stayInLoop = 0;
                  for (kp = kp1; kp < kp2+1; kp++) {
                     ie = iw[kp];
                     if (TMath::Abs(flag[ie]+0) > nflg) {
                        stayInLoop = 1;
                        break;
                     }
                  }
                  if (!stayInLoop) {
                     doit = 1;
                     break;
                  }
                  js = nxt[js];
               }
            }
 
            if (doit == 1) {
               ipe[js] = -is;
               nv[is] = nv[is]+nv[js];
               nv[js] = 0;
               flag[js] = -1;
               ns = nxt[js];
               ls = lst[js];
               if (ns > 0) lst[ns] = is;
               if (ls > 0) nxt[ls] = is;
               lst[is] = ls;
               nxt[is] = ns;
               lst[js] = 0;
               nxt[js] = 0;
               if (ipd[id] == js) ipd[id] = is;
            } else if (doit == 2) {
               if (nvroot == 0) {
                  root = is;
                  ipe[is] = 0;
               } else {
                  iw[k] = root;
                  ipe[is] = -root;
                  nv[root] = nv[root]+nv[is];
                  nv[is] = 0;
                  flag[is] = -1;
               }
               nvroot = nv[root];
            } else if (doit == 3) {
               ns = ipd[id];
               if (ns > 0) lst[ns] = is;
               nxt[is] = ns;
               ipd[id] = is;
               lst[is] = -id;
               md = TMath::Min(md,id);
            }
         }

         for (k = k1; k < k2+1; k++) {
            is = iw[k];
            if (nv[is] == 0) continue;
            flag[is] = nflg;
            iw[ip] = is;
            ip = ip+1;
         }
         iwfr = k1;
         flag[me] = -nflg;
         iw[ip] = iw[k1];
         iw[k1] = ip-k1;
         ipe[me] = k1;
         iwfr = ip+1;
      } else
         ipe[me] = 0;
   }

   for (is = 1; is < n+1; is++) {
      if (nxt[is] != 0 || lst[is] != 0) {
         if (nvroot == 0) {
            root = is;
            ipe[is] = 0;
         } else {
            ipe[is] = -root;
         }
         nvroot = nvroot+nv[is];
         nv[is] = 0;
      }
   }

   for (ie = 1; ie < n+1; ie++)
      if (ipe[ie] > 0) ipe[ie] = -root;

   if (nvroot> 0) nv[root] = nvroot;
}

//______________________________________________________________________________
void TDecompSparse::InitPivot_sub2a(const Int_t n,Int_t *ipe,Int_t *iw,const Int_t lw,
                                    Int_t &iwfr,Int_t &ncmpa)
{
// Help routine for pivoting setup

   Int_t i,ir,k,k1,k2,lwfr;

   ncmpa = ncmpa+1;
   for (i = 1; i < n+1; i++) {
      k1 = ipe[i];
      if (k1 <= 0)  continue;
      ipe[i] = iw[k1];
      iw[k1] = -i;
   }

   iwfr = 1;
   lwfr = iwfr;
   for (ir = 1; ir < n+1; ir++) {
      if (lwfr > lw) break;
      Int_t skip = 1;
      for (k = lwfr; k < lw+1; k++) {
         if (iw[k] < 0) {
            skip = 0;
            break;
         }
      }
      if (skip) break;
      i = -iw[k];
      iw[iwfr] = ipe[i];
      ipe[i] = iwfr;
      k1 = k+1;
      k2 = k+iw[iwfr];
      iwfr = iwfr+1;
      if (k1 <= k2)  {
         for (k = k1; k < k2+1; k++) {
            iw[iwfr] = iw[k];
            iwfr = iwfr+1;
         }
      }
      lwfr = k2+1;
   }
}

//______________________________________________________________________________
void TDecompSparse::InitPivot_sub3(const Int_t n,const Int_t nz,Int_t *irn,Int_t *icn,
                                   Int_t *perm,Int_t *iw,Int_t *ipe,Int_t *iq,
                                   Int_t *flag,Int_t &iwfr,Int_t *icntl,Int_t *info)
{
// Help routine for pivoting setup

   Int_t i,id,in,j,jdummy,k,k1,k2,l,lbig,len;

   info[1] = 0;
   info[2] = 0;
   for (i = 1; i < n+1; i++)
      iq[i] = 0;

   if (nz != 0) {
      for (k = 1; k < nz+1; k++) {
         i = irn[k];
         j = icn[k];
         iw[k] = -i;

         Bool_t outRange = (i < 1 || i > n || j < 1 || j > n);
         if (outRange) {
            info[2] = info[2]+1;
            info[1] = 1;
            if (info[2] <= 1 && icntl[2] > 0)
               ::Warning("TDecompSparse::InitPivot_sub3","info[1]= %d; %d 'th non-zero (in row %d and column %d) ignored",info[1],k,i,j);
         }

         if (outRange || i==j) {
            iw[k] = 0;
         } else {
            if (perm[j] <= perm[i])
              iq[j] = iq[j]+1;
            else
               iq[i] = iq[i]+1;
         }
      }
   }

   iwfr = 1;
   lbig = 0;
   for (i = 1; i < n+1; i++) {
      l = iq[i];
      lbig = TMath::Max(l,lbig);
      iwfr = iwfr+l;
      ipe[i] = iwfr-1;
   }

   if (nz != 0) {
      for (k = 1; k < nz+1; k++) {
         i = -iw[k];
         if (i <= 0) continue;
         l = k;
         iw[k] = 0;
         for (id = 1; id < nz+1; id++) {
            j = icn[l];
            if (perm[i] >= perm[j]) {
               l = ipe[j];
               ipe[j] = l-1;
               in = iw[l];
               iw[l] = i;
            } else {
               l = ipe[i];
               ipe[i] = l-1;
               in = iw[l];
               iw[l] = j;
            }
            i = -in;
            if (i <= 0) continue;
         }
      }

      k = iwfr-1;
      l = k+n;
      iwfr = l+1;
      for (i = 1; i < n+1; i++) {
         flag[i] = 0;
         j = n+1-i;
         len = iq[j];
         if (len > 0)  {
            for (jdummy = 1; jdummy < len+1; jdummy++) {
               iw[l] = iw[k];
               k = k-1;
               l = l-1;
            }
         }
         ipe[j] = l;
         l = l-1;
      }

      if (lbig < icntl[4]) {
         for (i = 1; i < n+1; i++) {
            k = ipe[i];
            iw[k] = iq[i];
            if (iq[i] == 0) ipe[i] = 0;
         }
      } else {
         iwfr = 1;
         for (i = 1; i < n+1; i++) {
            k1 = ipe[i]+1;
            k2 = ipe[i]+iq[i];
            if (k1 > k2) {
               ipe[i] = 0;
            } else {
               ipe[i] = iwfr;
               iwfr = iwfr+1;
               for (k = k1; k < k2+1; k++) {
                  j = iw[k];
                  if (flag[j] == i) continue;
                  iw[iwfr] = j;
                  iwfr = iwfr+1;
                  flag[j] = i;
               }
               k = ipe[i];
               iw[k] = iwfr-k-1;
            }
         }
      }
   }

}

//______________________________________________________________________________
void TDecompSparse::InitPivot_sub4(const Int_t n,Int_t *ipe,Int_t *iw,const Int_t lw,
                                   Int_t &iwfr,Int_t *ips,Int_t *ipv,Int_t *nv,Int_t *flag,
                                   Int_t &ncmpa)
{
// Help routine for pivoting setup

   Int_t i,ie,ip,j,je,jp,jp1,jp2,js,kdummy,ln,lwfr,me,minjs,ml,ms;

   for (i = 1; i < n+1; i++) {
      flag[i] = 0;
      nv[i] = 0;
      j = ips[i];
      ipv[j] = i;
   }

   ncmpa = 0;
   for (ml = 1; ml < n+1; ml++) {
      ms = ipv[ml];
      me = ms;
      flag[ms] = me;
      ip = iwfr;
      minjs = n;
      ie = me;

      for (kdummy = 1; kdummy < n+1; kdummy++) {
         jp = ipe[ie];
         ln = 0;
         if (jp > 0) {
            ln = iw[jp];
            for (jp1 = 1; jp1 < ln+1; jp1++) {
               jp = jp+1;
               js = iw[jp];
               if (flag[js] == me) continue;
               flag[js] = me;
               if (iwfr >= lw) {
                  ipe[ie] = jp;
                  iw[jp] = ln-jp1;
                  InitPivot_sub2a(n,ipe,iw,ip-1,lwfr,ncmpa);
                  jp2 = iwfr-1;
                  iwfr = lwfr;
                  if (ip <= jp2)  {
                     for (jp = ip; jp < jp2+1; jp++) {
                        iw[iwfr] = iw[jp];
                        iwfr = iwfr+1;
                     }
                  }
                  ip = lwfr;
                  jp = ipe[ie];
               }
               iw[iwfr] = js;
               minjs = TMath::Min(minjs,ips[js]+0);
               iwfr = iwfr+1;
            }
         }
         ipe[ie] = -me;
         je = nv[ie];
         nv[ie] = ln+1;
         ie = je;
         if (ie == 0) break;
      }

      if (iwfr <= ip) {
         ipe[me] = 0;
         nv[me] = 1;
      } else {
         minjs = ipv[minjs];
         nv[me] = nv[minjs];
         nv[minjs] = me;
         iw[iwfr] = iw[ip];
         iw[ip] = iwfr-ip;
         ipe[me] = ip;
         iwfr = iwfr+1;
      }
   }
}

//______________________________________________________________________________
void TDecompSparse::InitPivot_sub5(const Int_t n,Int_t *ipe,Int_t *nv,Int_t *ips,Int_t *ne,
                                   Int_t *na,Int_t *nd,Int_t &nsteps,const Int_t nemin)
{
// Help routine for pivoting setup

   Int_t i,ib,iff,il,is,ison,k,l,nr;

   il = 0;
   for (i = 1; i < n+1; i++) {
      ips[i] = 0;
      ne[i] = 0;
   }
   for (i = 1; i < n+1; i++) {
      if (nv[i] > 0) continue;
      iff = -ipe[i];
      is = -ips[iff];
      if (is > 0) ipe[i] = is;
      ips[iff] = -i;
   }

   nr = n+1;
   for (i = 1; i < n+1; i++) {
      if (nv[i] <= 0) continue;
      iff = -ipe[i];
      if (iff != 0) {
         is = -ips[iff];
         if (is > 0)
            ipe[i] = is;
         ips[iff] = -i;
      } else {
         nr = nr-1;
         ne[nr] = i;
      }
   }

   is = 1;
   i = 0;
   for (k = 1; k < n+1; k++) {
      if (i <= 0) {
         i = ne[nr];
         ne[nr] = 0;
         nr = nr+1;
         il = n;
         na[n] = 0;
      }
      for (l = 1; l < n+1; l++) {
         if (ips[i] >= 0) break;
         ison = -ips[i];
         ips[i] = 0;
         i = ison;
         il = il-1;
         na[il] = 0;
      }

      ips[i] = k;
      ne[is] = ne[is]+1;
      if (nv[i] > 0) {
         if (il < n) na[il+1] = na[il+1]+1;
         na[is] = na[il];
         nd[is] = nv[i];

         Bool_t doit = (na[is] == 1 && (nd[is-1]-ne[is-1] == nd[is])) ||
                       (na[is] != 1 && ne[is] < nemin && na[is] != 0 && ne[is-1] < nemin);

         if (doit) {
            na[is-1] = na[is-1]+na[is]-1;
            nd[is-1] = nd[is]+ne[is-1];
            ne[is-1] = ne[is]+ne[is-1];
            ne[is] = 0;
         } else {
            is = is+1;
         }
      }

      ib = ipe[i];
      if (ib >= 0) {
         if (ib > 0)
            na[il] = 0;
         i = ib;
      } else {
         i = -ib;
         il = il+1;
      }
   }

   nsteps = is-1;
}

//______________________________________________________________________________
void TDecompSparse::InitPivot_sub6(const Int_t n,const Int_t nz,Int_t *irn,Int_t *icn,
                                   Int_t *perm,Int_t *na,Int_t *ne,Int_t *nd,const Int_t nsteps,
                                   Int_t *lstki,Int_t *lstkr,Int_t *iw,Int_t *info,Double_t &ops)
{
// Help routine for pivoting setup

   Int_t i,inew,iold,iorg,irow,istki,istkr,itop,itree,jold,jorg,k,lstk,nassr,nelim,nfr,nstk,
         numorg,nz1,nz2,nrladu,niradu,nirtot,nrltot,nirnec,nrlnec;
   Double_t delim;

   if (nz != 0 && irn[1] == iw[1]) {
      irn[1] = iw[1]-1;
      nz2 = 0;
      for (iold = 1; iold < n+1; iold++) {
         inew = perm[iold];
         lstki[inew] = lstkr[iold]+1;
         nz2 = nz2+lstkr[iold];
      }
      nz1 = nz2/2+n;
      nz2 = nz2+n;
   } else {
      for (i = 1; i < n+1; i++)
         lstki[i] = 1;
      nz1 = n;
      if (nz != 0) {
         for (i = 1; i < nz+1; i++) {
            iold = irn[i];
            jold = icn[i];
            if (iold < 1 || iold > n) continue;
            if (jold < 1 || jold > n) continue;
            if (iold == jold) continue;
            nz1 = nz1+1;
            irow = TMath::Min(perm[iold]+0,perm[jold]+0);
            lstki[irow] = lstki[irow]+1;
         }
      }
      nz2 = nz1;
   }

   ops = 0.0;
   istki = 0;
   istkr = 0;
   nrladu = 0;
   niradu = 1;
   nirtot = nz1;
   nrltot = nz1;
   nirnec = nz2;
   nrlnec = nz2;
   numorg = 0;
   itop = 0;
   for (itree = 1; itree < nsteps+1; itree++) {
      nelim = ne[itree];
      delim = Double_t(nelim);
      nfr = nd[itree];
      nstk = na[itree];
      nassr = nfr*(nfr+1)/2;
      if (nstk != 0) nassr = nassr-lstkr[itop]+1;
      nrltot = TMath::Max(nrltot,nrladu+nassr+istkr+nz1);
      nirtot = TMath::Max(nirtot,niradu+nfr+2+istki+nz1);
      nrlnec = TMath::Max(nrlnec,nrladu+nassr+istkr+nz2);
      nirnec = TMath::Max(nirnec,niradu+nfr+2+istki+nz2);
      for (iorg = 1; iorg < nelim+1; iorg++) {
         jorg = numorg+iorg;
         nz2 = nz2-lstki[jorg];
      }
      numorg = numorg+nelim;
      if (nstk > 0) {
         for (k = 1; k < nstk+1; k++) {
            lstk = lstkr[itop];
            istkr = istkr-lstk;
            lstk = lstki[itop];
            istki = istki-lstk;
            itop = itop-1;
         }
      }
      nrladu = nrladu+(nelim* (2*nfr-nelim+1))/2;
      niradu = niradu+2+nfr;
      if (nelim == 1) niradu = niradu-1;
      ops = ops+((nfr*delim*(nfr+1)-(2*nfr+1)*delim*(delim+1)/2+delim*(delim+1)*(2*delim+1)/6)/2);
      if (itree == nsteps || nfr == nelim) continue;
      itop = itop+1;
      lstkr[itop] = (nfr-nelim)* (nfr-nelim+1)/2;
      lstki[itop] = nfr-nelim+1;
      istki = istki+lstki[itop];
      istkr = istkr+lstkr[itop];
      nirtot = TMath::Max(nirtot,niradu+istki+nz1);
      nirnec = TMath::Max(nirnec,niradu+istki+nz2);
   }

   nrlnec = TMath::Max(nrlnec,n+TMath::Max(nz,nz1));
   nrltot = TMath::Max(nrltot,n+TMath::Max(nz,nz1));
   nrlnec = TMath::Min(nrlnec,nrltot);
   nirnec = TMath::Max(nz,nirnec);
   nirtot = TMath::Max(nz,nirtot);
   nirnec = TMath::Min(nirnec,nirtot);
   info[3] = nrltot;
   info[4] = nirtot;
   info[5] = nrlnec;
   info[6] = nirnec;
   info[7] = nrladu;
   info[8] = niradu;
}

//______________________________________________________________________________
void TDecompSparse::Factor_sub1(const Int_t n,const Int_t nz,Int_t &nz1,Double_t *a,
                                const Int_t la,Int_t *irn,Int_t *icn,Int_t *iw,const Int_t liw,
                                Int_t *perm,Int_t *iw2,Int_t *icntl,Int_t *info)
{
// Help routine for factorization

   Int_t i,ia,ich,ii,iiw,inew,iold,ipos,j1,j2,jj,jnew,jold,jpos,k;
   Double_t anext,anow;

   const Double_t zero = 0.0;
   info[1] = 0;
   ia = la;
   for (iold = 1; iold < n+1; iold++) {
      iw2[iold] = 1;
      a[ia] = zero;
      ia = ia-1;
   }

   info[2] = 0;
   nz1 = n;
   if (nz != 0) {
      for (k = 1; k < nz+1; k++) {
         iold = irn[k];
         jold = icn[k];
         Bool_t outRange = (iold < 1 || iold > n || jold < 1 || jold > n);

         inew = perm[iold];
         jnew = perm[jold];

         if (!outRange && inew == jnew) {
            ia = la-n+iold;
            a[ia] = a[ia]+a[k];
            iw[k] = 0;
         } else {
            if (!outRange) {
               inew = TMath::Min(inew,jnew);
               iw2[inew] = iw2[inew]+1;
               iw[k] = -iold;
               nz1 = nz1+1;
            } else {
               info[1] = 1;
               info[2] = info[2]+1;
               if (info[2] <= 1 && icntl[2] > 0)
                  ::Warning("TDecompSparse::Factor_sub1","info[1]= %d; %d 'th non-zero (in row %d and column %d) ignored",
                            info[1],k,irn[k],icn[k]);
               iw[k] = 0;
            }
         }
      }
   }

   if (nz >= nz1 || nz1 == n) {
      k = 1;
      for (i = 1; i < n+1; i++) {
         k = k+iw2[i];
         iw2[i] = k;
      }
   } else {
      k = 1;
      for (i = 1; i < n+1; i++) {
         k = k+iw2[i]-1;
         iw2[i] = k;
      }
   }

   if (nz1 > liw) {
      info[1] = -3;
      info[2] = nz1;
      return;
   }

   if (nz1+n > la) {
      info[1] = -4;
      info[2] = nz1+n;
      return;
   }

   if (nz1 != n) {
      for (k = 1; k < nz+1; k++) {
         iold = -iw[k];
         if (iold <= 0) continue;
         jold = icn[k];
         anow = a[k];
         iw[k] = 0;
         for (ich = 1; ich < nz+1; ich++) {
            inew = perm[iold];
            jnew = perm[jold];
            inew = TMath::Min(inew,jnew);
            if (inew == perm[jold]) jold = iold;
            jpos = iw2[inew]-1;
            iold = -iw[jpos];
            anext = a[jpos];
            a[jpos] = anow;
            iw[jpos] = jold;
            iw2[inew] = jpos;
            if (iold == 0) break;
            anow = anext;
            jold = icn[jpos];
         }
      }

      if (nz < nz1) {
         ipos = nz1;
         jpos = nz1-n;
         for (ii = 1; ii < n+1; ii++) {
            i = n-ii+1;
            j1 = iw2[i];
            j2 = jpos;
            if (j1 <= jpos) {
               for (jj = j1; jj < j2+1; jj++) {
                  iw[ipos] = iw[jpos];
                  a[ipos] = a[jpos];
                  ipos = ipos-1;
                  jpos = jpos-1;
               }
            }
            iw2[i] = ipos+1;
            ipos = ipos-1;
         }
      }
   }

   for (iold = 1; iold < n+1; iold++) {
      inew = perm[iold];
      jpos = iw2[inew]-1;
      ia = la-n+iold;
      a[jpos] = a[ia];
      iw[jpos] = -iold;
   }
   ipos = nz1;
   ia = la;
   iiw = liw;
   for (i = 1; i < nz1+1; i++) {
      a[ia] = a[ipos];
      iw[iiw] = iw[ipos];
      ipos = ipos-1;
      ia = ia-1;
      iiw = iiw-1;
   }
}

//______________________________________________________________________________
void TDecompSparse::Factor_sub2(const Int_t n,const Int_t nz,Double_t *a,const Int_t la,
                                Int_t *iw,const Int_t liw,Int_t *perm,Int_t *nstk,
                                const Int_t nsteps,Int_t &maxfrt,Int_t *nelim,Int_t *iw2,
                                Int_t *icntl,Double_t *cntl,Int_t *info)
{
// Help routine for factorization

   Double_t amax,amult,amult1,amult2,detpiv,rmax,swop,thresh,tmax,uu;
   Int_t ainput,apos,apos1,apos2,apos3,astk,astk2,azero,i,iass;
   Int_t ibeg,idummy,iell,iend,iexch,ifr,iinput,ioldps,iorg,ipiv;
   Int_t ipmnp,ipos,irow,isnpiv,istk,istk2,iswop,iwpos,j,j1;
   Int_t j2,jcol,jdummy,jfirst,jj,jj1,jjj,jlast,jmax,jmxmip,jnew;
   Int_t jnext,jpiv,jpos,k,k1,k2,kdummy,kk,kmax,krow,laell,lapos2;
   Int_t liell,lnass,lnpiv,lt,ltopst,nass,nblk,newel,nfront,npiv;
   Int_t npivp1,ntotpv,numass,numorg,numstk,pivsiz,posfac,pospv1,pospv2;
   Int_t ntwo,neig,ncmpbi,ncmpbr,nrlbdu,nirbdu;

   const Double_t zero = 0.0;
   const Double_t half = 0.5;
   const Double_t one  = 1.0;

   detpiv = 0.0;
   apos3  = 0;
   isnpiv = 0;
   jmax   = 0;
   jpos   = 0;

   nblk = 0;
   ntwo = 0;
   neig = 0;
   ncmpbi = 0;
   ncmpbr = 0;
   maxfrt = 0;
   nrlbdu = 0;
   nirbdu = 0;
   uu = TMath::Min(cntl[1],half);
   uu = TMath::Max(uu,-half);
   for (i = 1; i < n+1; i++)
      iw2[i] = 0;
   iwpos = 2;
   posfac = 1;
   istk = liw-nz+1;
   istk2 = istk-1;
   astk = la-nz+1;
   astk2 = astk-1;
   iinput = istk;
   ainput = astk;
   azero = 0;
   ntotpv = 0;
   numass = 0;

   for (iass = 1; iass < nsteps+1; iass++) {
      nass = nelim[iass];
      newel = iwpos+1;
      jfirst = n+1;
      nfront = 0;
      numstk = nstk[iass];
      ltopst = 1;
      lnass = 0;
      if (numstk != 0) {
         j2 = istk-1;
         lnass = nass;
         ltopst = ((iw[istk]+1)*iw[istk])/2;
         for (iell = 1; iell < numstk+1; iell++) {
            jnext = jfirst;
            jlast = n+1;
            j1 = j2+2;
            j2 = j1-1+iw[j1-1];
            for (jj = j1; jj < j2+1; jj++) {
               j = iw[jj];
               if (iw2[j] > 0) continue;
               jnew = perm[j];
               if (jnew <= numass) nass = nass+1;
               for (idummy = 1; idummy < n+1; idummy++) {
                  if (jnext == n+1) break;
                  if (perm[jnext] > jnew) break;
                  jlast = jnext;
                  jnext = iw2[jlast];
               }
               if (jlast == n+1)
                  jfirst = j;
               else
                  iw2[jlast] = j;
               iw2[j] = jnext;
               jlast = j;
               nfront = nfront+1;
            }
         }
         lnass = nass-lnass;
      }

      numorg = nelim[iass];
      j1 = iinput;
      for (iorg = 1; iorg < numorg+1; iorg++) {
         j = -iw[j1];
         for (idummy = 1; idummy < liw+1; idummy++) {
            jnew = perm[j];
            if (iw2[j] <= 0) {
               jlast = n+1;
               jnext = jfirst;
               for (jdummy = 1; jdummy < n+1; jdummy++) {
                  if (jnext == n+1) break;
                  if (perm[jnext] > jnew) break;
                  jlast = jnext;
                  jnext = iw2[jlast];
               }
               if (jlast == n+1)
                  jfirst = j;
               else
                  iw2[jlast] = j;
               iw2[j] = jnext;
               nfront = nfront+1;
            }
            j1 = j1+1;
            if (j1 > liw) break;
            j = iw[j1];
            if (j < 0) break;
         }
      }

      if (newel+nfront >= istk)
         Factor_sub3(a,iw,istk,istk2,iinput,2,ncmpbr,ncmpbi);
      if (newel+nfront >= istk) {
         info[1] = -3;
         info[2] = liw+1+newel+nfront-istk;
         goto finish;
      }

      j = jfirst;
      for (ifr = 1; ifr < nfront+1; ifr++) {
         newel = newel+1;
         iw[newel] = j;
         jnext = iw2[j];
         iw2[j] = newel-(iwpos+1);
         j = jnext;
      }

      maxfrt = TMath::Max(maxfrt,nfront);
      iw[iwpos] = nfront;
      laell = ((nfront+1)*nfront)/2;
      apos2 = posfac+laell-1;
      if (numstk != 0) lnass = lnass*(2*nfront-lnass+1)/2;

      if (posfac+lnass-1 >= astk || apos2 >= astk+ltopst-1) {
         Factor_sub3(a,iw,astk,astk2,ainput,1,ncmpbr,ncmpbi);
         if (posfac+lnass-1 >= astk || apos2 >= astk+ltopst-1) {
            info[1] = -4;
            info[2] = la+TMath::Max(posfac+lnass,apos2-ltopst+2)-astk;
            goto finish;
         }
      }

      if (apos2 > azero) {
         apos = azero+1;
         lapos2 = TMath::Min(apos2,astk-1);
         if (lapos2 >= apos) {
            for (k= apos; k< lapos2+1; k++)
            a[k] = zero;
         }
         azero = apos2;
      }

      if (numstk != 0) {
         for (iell = 1; iell < numstk+1; iell++) {
            j1 = istk+1;
            j2 = istk+iw[istk];
            for (jj = j1; jj < j2+1; jj++) {
               irow = iw[jj];
               irow = iw2[irow];
               apos = posfac+IDiag(nfront,irow);
               for (jjj = jj; jjj < j2+1; jjj++) {
                  j = iw[jjj];
                  apos2 = apos+iw2[j]-irow;
                  a[apos2] = a[apos2]+a[astk];
                  a[astk] = zero;
                  astk = astk+1;
               }
            }
            istk = j2+1;
         }
      }

      for (iorg = 1; iorg < numorg+1; iorg++) {
         j = -iw[iinput];
         irow = iw2[j];
         apos = posfac+IDiag(nfront,irow);
         for (idummy = 1; idummy < nz+1; idummy++) {
            apos2 = apos+iw2[j]-irow;
            a[apos2] = a[apos2]+a[ainput];
            ainput = ainput+1;
            iinput = iinput+1;
            if (iinput > liw) break;
            j = iw[iinput];
            if (j < 0) break;
         }
      }
      numass = numass+numorg;
      j1 = iwpos+2;
      j2 = iwpos+nfront+1;
      for (k = j1; k < j2+1; k++) {
         j = iw[k];
         iw2[j] = 0;
      }

      lnpiv = -1;
      npiv = 0;
      for (kdummy = 1; kdummy < nass+1; kdummy++) {
         if (npiv == nass) break;
         if (npiv == lnpiv) break;
         lnpiv = npiv;
         npivp1 = npiv+1;
         jpiv = 1;
         for (ipiv = npivp1; ipiv < nass+1; ipiv++) {
            jpiv = jpiv-1;
            if (jpiv == 1) continue;
            apos = posfac+IDiag(nfront-npiv,ipiv-npiv);

            if (uu <= zero) {
               if (TMath::Abs(a[apos]) <= cntl[3]) {
                  info[1] = -5;
                  info[2] = ntotpv+1;
                  goto finish;
               }
               if (ntotpv <=  0) {
                  if (a[apos] > zero) isnpiv = 1;
                  if (a[apos] < zero) isnpiv = -1;
               }
               if ((a[apos] <= zero || isnpiv !=  1) && (a[apos] >= zero || isnpiv != -1)) {
                  if (info[1] != 2) info[2] = 0;
                  info[2] = info[2]+1;
                  info[1] = 2;
                  i = ntotpv+1;
                  if (icntl[2] > 0 && info[2] <= 10)
                     ::Warning("TDecompSparse::Factor_sub2","info[1]= %d; pivot %d has different sign from the previous one",
                               info[1],i);
                  isnpiv = -isnpiv;
               }
               if ((a[apos] > zero && isnpiv ==  1) || (a[apos] < zero && isnpiv == -1) || (uu == zero)) goto hack;
               info[1] = -6;
               info[2] = ntotpv+1;
               goto finish;
            }

            amax = zero;
            tmax = amax;
            j1 = apos+1;
            j2 = apos+nass-ipiv;
            if (j2 >= j1) {
               for (jj = j1; jj < j2+1; jj++) {
                  if (TMath::Abs(a[jj]) <= amax) continue;
                  jmax = ipiv+jj-j1+1;
                  amax = TMath::Abs(a[jj]);
               }
            }
            j1 = j2+1;
            j2 = apos+nfront-ipiv;
            if (j2 >= j1) {
               for (jj = j1; jj < j2+1; jj++)
                  tmax = TMath::Max(TMath::Abs(a[jj]),tmax);
            }
            rmax = TMath::Max(tmax,amax);
            apos1 = apos;
            kk = nfront-ipiv;
            lt = ipiv-(npiv+1);
            if (lt != 0) {
               for (k = 1; k < lt+1; k++) {
                  kk = kk+1;
                  apos1 = apos1-kk;
                  rmax = TMath::Max(rmax,TMath::Abs(a[apos1]));
               }
            }
            if (TMath::Abs(a[apos]) <= TMath::Max(cntl[3],uu*rmax)) {
               if (TMath::Abs(amax) <= cntl[3]) continue;
               apos2 = posfac+IDiag(nfront-npiv,jmax-npiv);
               detpiv = a[apos]*a[apos2]-amax*amax;
               thresh = TMath::Abs(detpiv);
               thresh = thresh/(uu*TMath::Max(TMath::Abs(a[apos])+amax,TMath::Abs(a[apos2])+amax));
               if (thresh <= rmax) continue;
               rmax = zero;
               j1 = apos2+1;
               j2 = apos2+nfront-jmax;
               if (j2 >= j1) {
                  for (jj = j1; jj < j2+1; jj++)
                     rmax = TMath::Max(rmax,TMath::Abs(a[jj]));
               }
               kk = nfront-jmax+1;
               apos3 = apos2;
               jmxmip = jmax-ipiv-1;
               if (jmxmip != 0) {
                  for (k = 1; k < jmxmip+1; k++) {
                     apos2 = apos2-kk;
                     kk = kk+1;
                     rmax = TMath::Max(rmax,TMath::Abs(a[apos2]));
                  }
               }
               ipmnp = ipiv-npiv-1;
               if (ipmnp != 0) {
                  apos2 = apos2-kk;
                  kk = kk+1;
                  for (k = 1; k < ipmnp+1; k++) {
                     apos2 = apos2-kk;
                     kk = kk+1;
                     rmax = TMath::Max(rmax,TMath::Abs(a[apos2]));
                  }
               }
               if (thresh <= rmax) continue;
               pivsiz = 2;
            } else {
               pivsiz = 1;
            }

            irow = ipiv-npiv;
            for (krow = 1; krow < pivsiz+1; krow++) {
               if (irow != 1) {
                  j1 = posfac+irow;
                  j2 = posfac+nfront-(npiv+1);
                  if (j2 >= j1) {
                     apos2 = apos+1;
                     for (jj = j1; jj < j2+1; jj++) {
                        swop = a[apos2];
                        a[apos2] = a[jj];
                        a[jj] = swop;
                        apos2 = apos2+1;
                     }
                  }
                  j1 = posfac+1;
                  j2 = posfac+irow-2;
                  apos2 = apos;
                  kk = nfront-(irow+npiv);
                  if (j2 >= j1) {
                     for (jjj = j1; jjj < j2+1; jjj++) {
                        jj = j2-jjj+j1;
                        kk = kk+1;
                        apos2 = apos2-kk;
                        swop = a[apos2];
                        a[apos2] = a[jj];
                        a[jj] = swop;
                     }
                  }
                  if (npiv != 0) {
                     apos1 = posfac;
                     kk = kk+1;
                     apos2 = apos2-kk;
                     for (jj = 1; jj < npiv+1; jj++) {
                        kk = kk+1;
                        apos1 = apos1-kk;
                        apos2 = apos2-kk;
                        swop = a[apos2];
                        a[apos2] = a[apos1];
                        a[apos1] = swop;
                     }
                  }
                  swop = a[apos];
                  a[apos] = a[posfac];
                  a[posfac] = swop;
                  ipos = iwpos+npiv+2;
                  iexch = iwpos+irow+npiv+1;
                  iswop = iw[ipos];
                  iw[ipos] = iw[iexch];
                  iw[iexch] = iswop;
               }
               if (pivsiz == 1) continue;
               if (krow != 2)  {
                  irow = jmax-(npiv+1);
                  jpos = posfac;
                  posfac = posfac+nfront-npiv;
                  npiv = npiv+1;
                  apos = apos3;
               } else {
                  npiv = npiv-1;
                  posfac = jpos;
               }
            }

            if (pivsiz != 2) {
hack:
               a[posfac] = one/a[posfac];
               if (a[posfac] < zero) neig = neig+1;
               j1 = posfac+1;
               j2 = posfac+nfront-(npiv+1);
               if (j2 >= j1) {
                  ibeg = j2+1;
                  for (jj = j1; jj < j2+1; jj++) {
                     amult = -a[jj]*a[posfac];
                     iend = ibeg+nfront-(npiv+jj-j1+2);
                     for (irow = ibeg; irow < iend+1; irow++) {
                        jcol = jj+irow-ibeg;
                        a[irow] = a[irow]+amult*a[jcol];
                     }
                     ibeg = iend+1;
                     a[jj] = amult;
                  }
               }
               npiv = npiv+1;
               ntotpv = ntotpv+1;
               jpiv = 1;
               posfac = posfac+nfront-npiv+1;
            } else {
               ipos = iwpos+npiv+2;
               ntwo = ntwo+1;
               iw[ipos] = -iw[ipos];
               pospv1 = posfac;
               pospv2 = posfac+nfront-npiv;
               swop = a[pospv2];
               if (detpiv < zero) neig = neig+1;
               if (detpiv > zero && swop < zero) neig = neig+2;
               a[pospv2] = a[pospv1]/detpiv;
               a[pospv1] = swop/detpiv;
               a[pospv1+1] = -a[pospv1+1]/detpiv;
               j1 = pospv1+2;
               j2 = pospv1+nfront-(npiv+1);
               if (j2 >= j1) {
                  jj1 = pospv2;
                  ibeg = pospv2+nfront-(npiv+1);
                  for (jj = j1; jj < j2+1; jj++) {
                     jj1 = jj1+1;
                     amult1 =-(a[pospv1]*a[jj]+a[pospv1+1]*a[jj1]);
                     amult2 =-(a[pospv1+1]*a[jj]+a[pospv2]*a[jj1]);
                     iend = ibeg+nfront-(npiv+jj-j1+3);
                     for (irow = ibeg; irow < iend+1; irow++) {
                        k1 = jj+irow-ibeg;
                        k2 = jj1+irow-ibeg;
                        a[irow] = a[irow]+amult1*a[k1]+amult2*a[k2];
                     }
                     ibeg = iend+1;
                     a[jj] = amult1;
                     a[jj1] = amult2;
                  }
               }
               npiv = npiv+2;
               ntotpv = ntotpv+2;
               jpiv = 2;
               posfac = pospv2+nfront-npiv+1;
            }
         }
      }

      if (npiv != 0) nblk = nblk+1;
      ioldps = iwpos;
      iwpos = iwpos+nfront+2;
      if (npiv != 0) {
         if (npiv <= 1) {
            iw[ioldps] = -iw[ioldps];
            for (k = 1; k < nfront+1; k++) {
               j1 = ioldps+k;
               iw[j1] = iw[j1+1];
            }
            iwpos = iwpos-1;
         } else {
            iw[ioldps+1] = npiv;
         }
      }
      liell = nfront-npiv;

      if (liell != 0 && iass != nsteps) {
         if (iwpos+liell >= istk)
            Factor_sub3(a,iw,istk,istk2,iinput,2,ncmpbr,ncmpbi);
         istk = istk-liell-1;
         iw[istk] = liell;
         j1 = istk;
         kk = iwpos-liell-1;
         for (k = 1; k < liell+1; k++) {
            j1 = j1+1;
            kk = kk+1;
            iw[j1] = iw[kk];
         }
         laell = ((liell+1)*liell)/2;
         kk = posfac+laell;
         if (kk == astk) {
            astk = astk-laell;
         } else {
            kmax = kk-1;
            for (k = 1; k < laell+1; k++) {
               kk = kk-1;
               astk = astk-1;
               a[astk] = a[kk];
            }
            kmax = TMath::Min(kmax,astk-1);
            for ( k = kk; k < kmax+1; k++)
               a[k] = zero;
         }
         azero = TMath::Min(azero,astk-1);
      }
      if (npiv == 0) iwpos = ioldps;
   }

   iw[1] = nblk;
   if (ntwo > 0) iw[1] = -nblk;
   nrlbdu = posfac-1;
   nirbdu = iwpos-1;

   if (ntotpv != n) {
      info[1] = 3;
      info[2] = ntotpv;
   }

finish:
   info[9]  = nrlbdu;
   info[10] = nirbdu;
   info[12] = ncmpbr;
   info[13] = ncmpbi;
   info[14] = ntwo;
   info[15] = neig;
}

//______________________________________________________________________________
void TDecompSparse::Factor_sub3(Double_t *a,Int_t *iw,Int_t &j1,Int_t &j2,const Int_t itop,
                                const Int_t ireal,Int_t &ncmpbr,Int_t &ncmpbi)
{
// Help routine for factorization

   Int_t ipos,jj,jjj;

   ipos = itop-1;
   if (j2 != ipos) {
      if (ireal != 2) {
         ncmpbr = ncmpbr+1;
         if (j1 <= j2) {
            for (jjj = j1; jjj < j2+1; jjj++) {
               jj = j2-jjj+j1;
               a[ipos] = a[jj];
               ipos = ipos-1;
            }
         }
      } else {
         ncmpbi = ncmpbi+1;
         if (j1 <= j2) {
            for (jjj = j1; jjj < j2+1; jjj++) {
               jj = j2-jjj+j1;
               iw[ipos] = iw[jj];
               ipos = ipos-1;
            }
         }
      }
      j2 = itop-1;
      j1 = ipos+1;
   }
}

//______________________________________________________________________________
void TDecompSparse::Solve_sub1(const Int_t n,Double_t *a,Int_t *iw,Double_t *w,
                               Double_t *rhs,Int_t *iw2,const Int_t nblk,Int_t &latop,
                               Int_t *icntl)
{
// Help routine for solving

   Int_t apos,iblk,ifr,ilvl,ipiv,ipos,irhs,irow,ist,j,j1=0,j2,j3,jj,jpiv,k,k1,k2,k3,liell,npiv;
   Double_t w1,w2;

   const Int_t ifrlvl = 5;

   apos = 1;
   ipos = 1;
   j2 = 0;
   iblk = 0;
   npiv = 0;
   for (irow = 1; irow < n+1; irow++) {
      if (npiv <= 0) {
         iblk = iblk+1;
         if (iblk > nblk) break;
         ipos = j2+1;
         iw2[iblk] = ipos;
         liell = -iw[ipos];
         npiv = 1;
         if (liell <= 0)  {
            liell = -liell;
            ipos = ipos+1;
            npiv = iw[ipos];
         }
         j1 = ipos+1;
         j2 = ipos+liell;
         ilvl = TMath::Min(npiv,10);
         if (liell < icntl[ifrlvl+ilvl]) goto hack;
         ifr = 0;
         for (jj = j1; jj < j2+1; jj++) {
            j = TMath::Abs(iw[jj]+0);
            ifr = ifr+1;
            w[ifr] = rhs[j];
         }
         jpiv = 1;
         j3 = j1;

         for (ipiv = 1; ipiv < npiv+1; ipiv++) {
            jpiv = jpiv-1;
            if (jpiv == 1) continue;

            if (iw[j3] >= 0) {
               jpiv = 1;
               j3 = j3+1;
               apos = apos+1;
               ist = ipiv+1;
               if (liell< ist) continue;
               w1 = w[ipiv];
               k = apos;
               for (j = ist; j < liell+1; j++) {
                  w[j] = w[j]+a[k]*w1;
                  k = k+1;
               }
               apos = apos+liell-ist+1;
            } else {
               jpiv = 2;
               j3 = j3+2;
               apos = apos+2;
               ist = ipiv+2;
               if (liell >= ist) {
                  w1 = w[ipiv];
                  w2 = w[ipiv+1];
                  k1 = apos;
                  k2 = apos+liell-ipiv;
                  for (j = ist; j < liell+1; j++) {
                     w[j] = w[j]+w1*a[k1]+w2*a[k2];
                     k1 = k1+1;
                     k2 = k2+1;
                  }
               }
               apos = apos+2*(liell-ist+1)+1;
            }
         }

         ifr = 0;
         for (jj = j1; jj < j2+1; jj++) {
            j = TMath::Abs(iw[jj]+0);
            ifr = ifr+1;
            rhs[j] = w[ifr];
         }
         npiv = 0;
      } else {
hack:
         if (iw[j1] >= 0) {
            npiv = npiv-1;
            apos = apos+1;
            j1 = j1+1;
            if (j1 <= j2) {
               irhs = iw[j1-1];
               w1 = rhs[irhs];
               k = apos;
               for (j = j1; j < j2+1; j++) {
                  irhs = TMath::Abs(iw[j]+0);
                  rhs[irhs] = rhs[irhs]+a[k]*w1;
                  k = k+1;
               }
            }
            apos = apos+j2-j1+1;
         } else {
            npiv = npiv-2;
            j1 = j1+2;
            apos = apos+2;
            if (j1 <= j2) {
               irhs = -iw[j1-2];
               w1 = rhs[irhs];
               irhs = iw[j1-1];
               w2 = rhs[irhs];
               k1 = apos;
               k3 = apos+j2-j1+2;
               for (j = j1; j < j2+1; j++) {
                  irhs = TMath::Abs(iw[j]+0);
                  rhs[irhs] = rhs[irhs]+w1*a[k1]+w2*a[k3];
                  k1 = k1+1;
                  k3 = k3+1;
               }
            }
            apos = apos+2*(j2-j1+1)+1;
         }
      }
   }

   latop = apos-1;
}

//______________________________________________________________________________
void TDecompSparse::Solve_sub2(const Int_t n,Double_t *a,Int_t *iw,Double_t *w,
                               Double_t *rhs,Int_t *iw2,const Int_t nblk,
                               const Int_t latop,Int_t *icntl)
{
// Help routine for solving

   Int_t apos,apos2,i1rhs,i2rhs,iblk,ifr,iipiv,iirhs,ilvl,ipiv,ipos,irhs,ist,
         j,j1=0,j2=0,jj,jj1,jj2,jpiv,jpos=0,k,liell,loop,npiv;
   Double_t w1,w2;

   const Int_t ifrlvl = 5;

   apos = latop+1;
   npiv = 0;
   iblk = nblk+1;
   for (loop = 1; loop < n+1; loop++) {
      if (npiv <= 0) {
         iblk = iblk-1;
         if (iblk < 1) break;
         ipos = iw2[iblk];
         liell = -iw[ipos];
         npiv = 1;
         if (liell <= 0) {
            liell = -liell;
            ipos = ipos+1;
            npiv = iw[ipos];
         }
         jpos = ipos+npiv;
         j2 = ipos+liell;
         ilvl = TMath::Min(10,npiv)+10;
         if (liell < icntl[ifrlvl+ilvl]) goto hack;
         j1 = ipos+1;
         ifr = 0;
         for (jj = j1; jj < j2+1; jj++) {
            j = TMath::Abs(iw[jj]+0);
            ifr = ifr+1;
            w[ifr] = rhs[j];
         }
         jpiv = 1;
         for (iipiv = 1; iipiv < npiv+1; iipiv++) {
            jpiv = jpiv-1;
            if (jpiv == 1) continue;
            ipiv = npiv-iipiv+1;
            if (ipiv == 1 || iw[jpos-1] >= 0) {
               jpiv = 1;
               apos = apos-(liell+1-ipiv);
               ist = ipiv+1;
               w1 = w[ipiv]*a[apos];
               if (liell >= ist) {
                  jj1 = apos+1;
                  for (j = ist; j < liell+1; j++) {
                     w1 = w1+a[jj1]*w[j];
                     jj1 = jj1+1;
                  }
               }
               w[ipiv] = w1;
               jpos = jpos-1;
            } else {
               jpiv = 2;
               apos2 = apos-(liell+1-ipiv);
               apos = apos2-(liell+2-ipiv);
               ist = ipiv+1;
               w1 = w[ipiv-1]*a[apos]+w[ipiv]*a[apos+1];
               w2 = w[ipiv-1]*a[apos+1]+w[ipiv]*a[apos2];
               if (liell >= ist) {
                  jj1 = apos+2;
                  jj2 = apos2+1;
                  for (j = ist; j < liell+1; j++) {
                     w1 = w1+w[j]*a[jj1];
                     w2 = w2+w[j]*a[jj2];
                     jj1 = jj1+1;
                     jj2 = jj2+1;
                  }
               }
               w[ipiv-1] = w1;
               w[ipiv] = w2;
               jpos = jpos-2;
            }
         }
         ifr = 0;
         for (jj = j1; jj < j2+1; jj++) {
            j = TMath::Abs(iw[jj]+0);
            ifr = ifr+1;
            rhs[j] = w[ifr];
         }
         npiv = 0;
      } else {
hack:
         if (npiv == 1 || iw[jpos-1] >= 0) {
            npiv = npiv-1;
            apos = apos-(j2-jpos+1);
            iirhs = iw[jpos];
            w1 = rhs[iirhs]*a[apos];
            j1 = jpos+1;
            if (j1 <= j2) {
               k = apos+1;
               for (j = j1; j < j2+1; j++) {
                  irhs = TMath::Abs(iw[j]+0);
                  w1 = w1+a[k]*rhs[irhs];
                  k = k+1;
               }
            }
            rhs[iirhs] = w1;
            jpos = jpos-1;
         } else {
            npiv = npiv-2;
            apos2 = apos-(j2-jpos+1);
            apos = apos2-(j2-jpos+2);
            i1rhs = -iw[jpos-1];
            i2rhs = iw[jpos];
            w1 = rhs[i1rhs]*a[apos]+rhs[i2rhs]*a[apos+1];
            w2 = rhs[i1rhs]*a[apos+1]+rhs[i2rhs]*a[apos2];
            j1 = jpos+1;
            if (j1 <= j2) {
               jj1 = apos+2;
               jj2 = apos2+1;
               for (j = j1; j < j2+1; j++) {
                  irhs = TMath::Abs(iw[j]+0);
                  w1 = w1+rhs[irhs]*a[jj1];
                  w2 = w2+rhs[irhs]*a[jj2];
                  jj1 = jj1+1;
                  jj2 = jj2+1;
               }
            }
            rhs[i1rhs] = w1;
            rhs[i2rhs] = w2;
            jpos = jpos-2;
         }
      }
   }
}

//______________________________________________________________________________
void TDecompSparse::Print(Option_t *opt) const
{
// Print class members

   TDecompBase::Print(opt);

   printf("fPrecision  = %.3f\n",fPrecision);
   printf("fIPessimism = %.3f\n",fIPessimism);
   printf("fRPessimism = %.3f\n",fRPessimism);

   TMatrixDSparse fact(0,fNrows-1,0,fNrows-1,fNnonZeros,
                       (Int_t*)fRowFact.GetArray(),(Int_t*)fColFact.GetArray(),(Double_t*)fFact.GetArray());
   fact.Print("fFact");
}

//______________________________________________________________________________
TDecompSparse &TDecompSparse::operator=(const TDecompSparse &source)
{
// Assignment operator

   if (this != &source) {
      TDecompBase::operator=(source);
      memcpy(fIcntl,source.fIcntl,31*sizeof(Int_t));
      memcpy(fCntl,source.fCntl,6*sizeof(Double_t));
      memcpy(fInfo,source.fInfo,21*sizeof(Int_t));
      fVerbose    = source.fVerbose;
      fPrecision  = source.fPrecision;
      fIkeep      = source.fIkeep;
      fIw         = source.fIw;
      fIw1        = source.fIw1;
      fIw2        = source.fIw2;
      fNsteps     = source.fNsteps;
      fMaxfrt     = source.fMaxfrt;
      fW          = source.fW;
      fIPessimism = source.fIPessimism;
      fRPessimism = source.fRPessimism;
      if (fA.IsValid())
         fA.Use(*const_cast<TMatrixDSparse *>(&(source.fA)));
      fNrows      = source.fNrows;
      fNnonZeros  = source.fNnonZeros;
      fFact       = source.fFact;
      fRowFact    = source.fRowFact;
      fColFact    = source.fColFact;
   }
   return *this;
}
