// Author: Stefan Schmitt
// DESY, 23/01/09

//  Version 17.1, bug fix with background uncertainty
//
//  History:
//    Version 17.0, possibility to specify an error matrix with SetInput
//    Version 16.1, parallel to changes in TUnfold
//    Version 16.0, parallel to changes in TUnfold
//    Version 15, fix bugs with uncorr. uncertainties, add backgnd subtraction
//    Version 14, remove some print-out, do not add unused sys.errors
//    Version 13, support for systematic errors

////////////////////////////////////////////////////////////////////////
//
//  TUnfoldSys : public TUnfold
//
//  TUnfold is used to decompose a measurement y into several sources x
//  given the measurement uncertainties and a matrix of migrations A
//
//  TUnfoldSys adds error propagation of systematic errors to TUnfold
//  Also, background sources (with errors) can be subtracted.
//
//  *************************************************************
//  * For most applications, it is better to use TUnfoldDensity *
//  * instead of using TUnfoldSys or TUnfold                    *
//  *************************************************************
//
//  If you use this software, please consider the following citation
//       S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]
//
//  More documentation and updates are available on
//      http://www.desy.de/~sschmitt
//
//
//  The following sources of systematic error are considered in TUnfoldSys
//
//  (a) uncorrelated errors on the input matrix histA, taken as the
//      errors provided with the histogram.
//      These are typically statistical errors from finite Monte Carlo samples
//
//  (b) correlated shifts of the input matrix histA. These shifts are taken
//      as one-sigma effects when switchig on a given error soure.
//      several such error sources may be defined
//
//  (c) a systematic error on the regularisation parameter tau
//
//  (d) uncorrelated errors on background sources, taken as the errors
//      provided with the background histograms
//
//  (e) scale errors on background sources
//
// In addition there is the (statistical) uncertainty of the input vector (i)
//
// Source (a) is providede with the original histogram histA
//     TUnfoldSys(histA,...)
//
// Sources (b) are added by calls to
//     AddSysError()
//
// The systematic uncertainty on tau (c) is set by
//     SetTauError()
//
// Backgound sources causing errors of type (d) and (e) are added by
//     SubtractBackground()
//
//
// NOTE:
// ======
//    Systematic errors (a), (b), (c) are propagated to the result
//       AFTER unfolding
//
//    Background errors (d) and (e) are added to the data errors
//       BEFORE unfolding
//
// For this reason:
//  errors of type (d) and (e) are INCLUDED in the standard error matrix
//  and other methods provided by the base class TUnfold:
//      GetOutput()
//      GetEmatrix()
//      ...
//  whereas errors of type (a), (b), (c) are NOT INCLUDED in the methods
//  provided by the base class TUnfold.
//
// Accessing error matrices:
// ===================================
//  The error sources (b),(c) and (e) propagate to shifts of the result.
//  These shifts may be accessed as histograms using the methods
//     GetDeltaSysSource()            corresponds to (b)
//     GetDeltaSysTau()               corresponds to (c)
//     GetDeltaSysBackgroundScale()   corresponds to (e)
//  The error sources (a) and (d) originate from many uncorrelated errors,
//  which in general are NOT uncorrelated on the result vector.
//  Thus, there is no corresponding shift of the output vector, only error
//  matrices are available
//
//  Method to get error matrix       corresponds to error sources
//  ===============================================================
//   GetEmatrixSysUncorr()             (a)
//   GetEmatrixSysSource()             (b)
//   GetEmatrixSysTau()                (c)
//   GetEmatrixSysBackgroundUncorr()   (d)
//   GetEmatrixSysBackgroundScale()    (e)
//   GetEmatrixInput()                 (i)
//   GetEmatrix()                      (i)+(d)+(e)
//   GetEmatrixTotal()                 (i)+(a)+(b)+(c)+(d)+(e)
//
//  Error matrices can be added to existing histograms.
//  This is useful to retreive the sum of several error matrices.
//  If the last argument of the GetEmatrixXXX() methods is set to kFALSE,
//  the histogram is not cleared, but the error matrix is simply added to the
//  existing histogram
//
////////////////////////////////////////////////////////////////////////

/*
  This file is part of TUnfold.

  TUnfold is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  TUnfold is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <TMap.h>
#include <TMath.h>
#include <TObjString.h>
#include <RVersion.h>
#include <cmath>

#include "TUnfoldSys.h"

ClassImp(TUnfoldSys)

TUnfoldSys::TUnfoldSys(void)
{
   // set all pointers to zero
   InitTUnfoldSys();
}

TUnfoldSys::TUnfoldSys
(const TH2 *hist_A, EHistMap histmap, ERegMode regmode,EConstraint constraint)
   : TUnfold(hist_A,histmap,regmode,constraint)
{
   // arguments:
   //    hist_A:  matrix that describes the migrations
   //    histmap: mapping of the histogram axes to the unfolding output
   //    regmode: global regularisation mode
   // data members initialized to something different from zero:
   //    fDA2, fDAcol

   // initialize TUnfold
   InitTUnfoldSys();

   // svae underflow and overflow bins
   fAoutside = new TMatrixD(GetNx(),2);
   // save the romalized errors on hist_A
   // to the matrices fDAinRelSq and fDAinColRelSq
   fDAinColRelSq = new TMatrixD(GetNx(),1);

   Int_t nmax=GetNx()*GetNy();
   Int_t *rowDAinRelSq = new Int_t[nmax];
   Int_t *colDAinRelSq = new Int_t[nmax];
   Double_t *dataDAinRelSq = new Double_t[nmax];

   Int_t da_nonzero=0;
   for (Int_t ix = 0; ix < GetNx(); ix++) {
      Int_t ibinx = fXToHist[ix];
      Double_t sum_sq= fSumOverY[ix]*fSumOverY[ix];
      for (Int_t ibiny = 0; ibiny <= GetNy()+1; ibiny++) {
         Double_t dz;
         if (histmap == kHistMapOutputHoriz) {
            dz = hist_A->GetBinError(ibinx, ibiny);
         } else {
            dz = hist_A->GetBinError(ibiny, ibinx);
         }
         Double_t normerr_sq=dz*dz/sum_sq;
         // quadratic sum of all errors from all bins,
         //   including under/overflow bins
         (*fDAinColRelSq)(ix,0) += normerr_sq;

         if(ibiny==0) {
            // underflow bin
            if (histmap == kHistMapOutputHoriz) {
               (*fAoutside)(ix,0)=hist_A->GetBinContent(ibinx, ibiny);
            } else {
               (*fAoutside)(ix,0)=hist_A->GetBinContent(ibiny, ibinx);
            }
         } else if(ibiny==GetNy()+1) {
            // overflow bins
            if (histmap == kHistMapOutputHoriz) {
               (*fAoutside)(ix,1)=hist_A->GetBinContent(ibinx, ibiny);
            } else {
               (*fAoutside)(ix,1)=hist_A->GetBinContent(ibiny, ibinx);
            }
         } else {
            // error on this bin
            rowDAinRelSq[da_nonzero]=ibiny-1;
            colDAinRelSq[da_nonzero] = ix;
            dataDAinRelSq[da_nonzero] = normerr_sq;
            if(dataDAinRelSq[da_nonzero]>0.0) da_nonzero++;
         }
      }
   }
   if(da_nonzero) {
      fDAinRelSq = CreateSparseMatrix(GetNy(),GetNx(),da_nonzero,
                                      rowDAinRelSq,colDAinRelSq,dataDAinRelSq);
   } else {
      DeleteMatrix(&fDAinColRelSq);
   }
   delete[] rowDAinRelSq;
   delete[] colDAinRelSq;
   delete[] dataDAinRelSq;
}

void TUnfoldSys::AddSysError
(const TH2 *sysError,const char *name,EHistMap histmap,ESysErrMode mode)
{
   // add a correlated error source
   //    sysError: alternative matrix or matrix of absolute/relative shifts
   //    name: name of the error source
   //    histmap: mapping of the histogram axes to the unfolding output
   //    mode: format of the error source

   if(fSysIn->FindObject(name)) {
      Error("AddSysError","Source %s given twice, ignoring 2nd call.\n",name);
   } else {
      // a copy of fA is made. It can be accessed inside the loop
      // without having to take care that the sparse structure of *fA
      // may be accidentally destroyed by asking for an element which is zero.
      TMatrixD aCopy(*fA);

      Int_t nmax= GetNx()*GetNy();
      Double_t *data=new Double_t[nmax];
      Int_t *cols=new Int_t[nmax];
      Int_t *rows=new Int_t[nmax];
      nmax=0;
      for (Int_t ix = 0; ix < GetNx(); ix++) {
         Int_t ibinx = fXToHist[ix];
         Double_t sum=0.0;
         for(Int_t loop=0;loop<2;loop++) {
            for (Int_t ibiny = 0; ibiny <= GetNy()+1; ibiny++) {
               Double_t z;
               // get bin content, depending on histmap
               if (histmap == kHistMapOutputHoriz) {
                  z = sysError->GetBinContent(ibinx, ibiny);
               } else {
                  z = sysError->GetBinContent(ibiny, ibinx);
               }
               // correct to absolute numbers
               if(mode!=kSysErrModeMatrix) {
                  Double_t z0;
                  if((ibiny>0)&&(ibiny<=GetNy())) {
                     z0=aCopy(ibiny-1,ix)*fSumOverY[ix];
                  } else if(ibiny==0) {
                     z0=(*fAoutside)(ix,0);
                  } else {
                     z0=(*fAoutside)(ix,1);
                  }
                  if(mode==kSysErrModeShift) {
                     z += z0;
                  } else if(mode==kSysErrModeRelative) {
                     z = z0*(1.+z);
                  }
               }
               if(loop==0) {
                  // sum up all entries, including overflow bins
                  sum += z;
               } else {
                  if((ibiny>0)&&(ibiny<=GetNy())) {
                     // save normalized matrix of shifts,
                     // excluding overflow bins
                     rows[nmax]=ibiny-1;
                     cols[nmax]=ix;
                     if(sum>0.0) {
                        data[nmax]=z/sum-aCopy(ibiny-1,ix);
                     } else {
                        data[nmax]=0.0;
                     }
                     if(data[nmax] != 0.0) nmax++;
                  }
               }
            }
         }
      }
      if(nmax==0) {
         Error("AddSysError",
               "source %s has no influence and has not been added.\n",name);
      } else {
         TMatrixDSparse *dsys=CreateSparseMatrix(GetNy(),GetNx(),
                                                 nmax,rows,cols,data);
         fSysIn->Add(new TObjString(name),dsys);
      }
      delete[] data;
      delete[] rows;
      delete[] cols;
   }
}

void TUnfoldSys::DoBackgroundSubtraction(void)
{
   // performs background subtraction
   // fY = fYData - fBgrIn
   // fVyy = fVyyData + fBgrErrUncorr^2 + fBgrErrCorr * fBgrErrCorr#
   // fVyyinv = fVyy^(-1)

   if(fYData) {
      DeleteMatrix(&fY);
      DeleteMatrix(&fVyy);
      if(fBgrIn->GetEntries()<=0) {
         // simple copy
         fY=new TMatrixD(*fYData);
         fVyy=new TMatrixDSparse(*fVyyData);
      } else {
         if(GetVyyInv()) {
            Warning("DoBackgroundSubtraction",
                    "inverse error matrix from user input,"
                    " not corrected for background");
         }
         // copy of the data
         fY=new TMatrixD(*fYData);
         // subtract background from fY
         const TObject *key;
         {
            TMapIter bgrPtr(fBgrIn);
            for(key=bgrPtr.Next();key;key=bgrPtr.Next()) {
               const TMatrixD *bgr=(const TMatrixD *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
                  ((const TPair *)*bgrPtr)->Value()
#else
                  fBgrIn->GetValue(((const TObjString *)key)->GetString())
#endif
                  ;
               for(Int_t i=0;i<GetNy();i++) {
                  (*fY)(i,0) -= (*bgr)(i,0);
               }
            }
         }
         // copy original error matrix
         TMatrixD vyy(*fVyyData);
         // determine used bins
         Int_t ny=fVyyData->GetNrows();
         const Int_t *vyydata_rows=fVyyData->GetRowIndexArray();
         const Int_t *vyydata_cols=fVyyData->GetColIndexArray();
         const Double_t *vyydata_data=fVyyData->GetMatrixArray();
         Int_t *usedBin=new Int_t[ny];
         for(Int_t i=0;i<ny;i++) {
            usedBin[i]=0;
         }
         for(Int_t i=0;i<ny;i++) {
            for(Int_t k=vyydata_rows[i];k<vyydata_rows[i+1];k++) {
               if(vyydata_data[k]>0.0) {
                  usedBin[i]++;
                  usedBin[vyydata_cols[k]]++;
               }
            }
         }
         // add uncorrelated background errors
         {
            TMapIter bgrErrUncorrSqPtr(fBgrErrUncorrInSq);
            for(key=bgrErrUncorrSqPtr.Next();key;
                key=bgrErrUncorrSqPtr.Next()) {
               const TMatrixD *bgrerruncorrSquared=(TMatrixD const *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
                  ((const TPair *)*bgrErrUncorrSqPtr)->Value()
#else
                  fBgrErrUncorrInSq->GetValue(((const TObjString *)key)
                                              ->GetString())
#endif
                  ;
               for(Int_t yi=0;yi<ny;yi++) {
                  if(!usedBin[yi]) continue;
                  vyy(yi,yi) +=(*bgrerruncorrSquared)(yi,0);
               }
            }
         }
         // add correlated background errors
         {
            TMapIter bgrErrScalePtr(fBgrErrScaleIn);
            for(key=bgrErrScalePtr.Next();key;key=bgrErrScalePtr.Next()) {
               const TMatrixD *bgrerrscale=(const TMatrixD *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
                  ((const TPair *)*bgrErrScalePtr)->Value()
#else
                  fBgrErrScaleIn->GetValue(((const TObjString *)key)
                                          ->GetString())
#endif
                  ;
               for(Int_t yi=0;yi<ny;yi++) {
                  if(!usedBin[yi]) continue;
                  for(Int_t yj=0;yj<ny;yj++) {
                     if(!usedBin[yj]) continue;
                     vyy(yi,yj) +=(*bgrerrscale)(yi,0)* (*bgrerrscale)(yj,0);
                  }
               }
            }
         }
         delete[] usedBin;
         usedBin=0;

         // convert to sparse matrix
         fVyy=new TMatrixDSparse(vyy);

      }
   } else {
      Fatal("DoBackgroundSubtraction","No input vector defined");
   }
}

Int_t TUnfoldSys::SetInput(const TH1 *hist_y,Double_t scaleBias,
                              Double_t oneOverZeroError,const TH2 *hist_vyy,
                              const TH2 *hist_vyy_inv)
{
   // Define the input data for subsequent calls to DoUnfold(Double_t)
   //  input:   input distribution with errors
   //  scaleBias:  scale factor applied to the bias
   //  oneOverZeroError: for bins with zero error, this number defines 1/error.
   // Return value: number of bins with bad error
   //                 +10000*number of unconstrained output bins
   //         Note: return values>=10000 are fatal errors,
   //               for the given input, the unfolding can not be done!
   // Calls the SetInput method of the base class, then renames the input
   // vectors fY and fVyy, then performs the background subtraction
   // Data members modified:
   //   fYData,fY,fVyyData,fVyy,fVyyinvData,fVyyinv
   // and those modified by TUnfold::SetInput()
   // and those modified by DoBackgroundSubtraction()

   Int_t r=TUnfold::SetInput(hist_y,scaleBias,oneOverZeroError,hist_vyy,
                             hist_vyy_inv);
   fYData=fY;
   fY=0;
   fVyyData=fVyy;
   fVyy=0;
   DoBackgroundSubtraction();

   return r;
}

void TUnfoldSys::SubtractBackground
(const TH1 *bgr,const char *name,Double_t scale,Double_t scale_error)
{
   // Store background source
   //   bgr:    background distribution with uncorrelated errors
   //   name:   name of this background source
   //   scale:  scale factor applied to the background
   //   scaleError: error on scale factor (correlated error)
   //
   // Data members modified:
   //   fBgrIn,fBgrErrUncorrInSq,fBgrErrScaleIn
   // and those modified by DoBackgroundSubtraction()

   // save background source
   if(fBgrIn->FindObject(name)) {
      Error("SubtractBackground","Source %s given twice, ignoring 2nd call.\n",
            name);
   } else {
      TMatrixD *bgrScaled=new TMatrixD(GetNy(),1);
      TMatrixD *bgrErrUncSq=new TMatrixD(GetNy(),1);
      TMatrixD *bgrErrCorr=new TMatrixD(GetNy(),1);
      for(Int_t row=0;row<GetNy();row++) {
         (*bgrScaled)(row,0) = scale*bgr->GetBinContent(row+1);
         (*bgrErrUncSq)(row,0) =
            TMath::Power(scale*bgr->GetBinError(row+1),2.);
         (*bgrErrCorr)(row,0) = scale_error*bgr->GetBinContent(row+1);
      }
      fBgrIn->Add(new TObjString(name),bgrScaled);
      fBgrErrUncorrInSq->Add(new TObjString(name),bgrErrUncSq);
      fBgrErrScaleIn->Add(new TObjString(name),bgrErrCorr);
      if(fYData) {
         DoBackgroundSubtraction();
      } else {
         Info("SubtractBackground",
              "Background subtraction prior to setting input data");
      }
   }
}

void TUnfoldSys::GetBackground
(TH1 *bgrHist,const char *bgrSource,const Int_t *binMap,
 Int_t includeError,Bool_t clearHist) const
{
   if(clearHist) {
       ClearHistogram(bgrHist);
   }
   // add all background sources
   const TObject *key;
   {
      TMapIter bgrPtr(fBgrIn);
      for(key=bgrPtr.Next();key;key=bgrPtr.Next()) {
         TString bgrName=((const TObjString *)key)->GetString();
         if(bgrSource && bgrName.CompareTo(bgrSource)) continue;
         const TMatrixD *bgr=(const TMatrixD *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
            ((const TPair *)*bgrPtr)->Value()
#else
            fBgrIn->GetValue(bgrName)
#endif
            ;
         for(Int_t i=0;i<GetNy();i++) {
            Int_t destBin=binMap[i];
            bgrHist->SetBinContent(destBin,bgrHist->GetBinContent(destBin)+
                                   (*bgr)(i,0));
         }
      }
   }
   // add uncorrelated background errors
   if(includeError &1) {
      TMapIter bgrErrUncorrSqPtr(fBgrErrUncorrInSq);
      for(key=bgrErrUncorrSqPtr.Next();key;key=bgrErrUncorrSqPtr.Next()) {
         TString bgrName=((const TObjString *)key)->GetString();
         if(bgrSource && bgrName.CompareTo(bgrSource)) continue;
         const TMatrixD *bgrerruncorrSquared=(TMatrixD const *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
            ((const TPair *)*bgrErrUncorrSqPtr)->Value()
#else
            fBgrErrUncorrInSq->GetValue(((const TObjString *)key)
                                            ->GetString())
#endif
            ;
         for(Int_t i=0;i<GetNy();i++) {
            Int_t destBin=binMap[i];
            bgrHist->SetBinError
               (destBin,TMath::Sqrt
                ((*bgrerruncorrSquared)(i,0)+
                 TMath::Power(bgrHist->GetBinError(destBin),2.)));
         }
      }
   }
   if(includeError & 2) {
      TMapIter bgrErrScalePtr(fBgrErrScaleIn);
      for(key=bgrErrScalePtr.Next();key;key=bgrErrScalePtr.Next()) {
         TString bgrName=((const TObjString *)key)->GetString();
         if(bgrSource && bgrName.CompareTo(bgrSource)) continue;
         const TMatrixD *bgrerrscale=(TMatrixD const *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
            ((const TPair *)*bgrErrScalePtr)->Value()
#else
            fBgrErrScaleIn->GetValue(((const TObjString *)key)->GetString())
#endif
            ;
         for(Int_t i=0;i<GetNy();i++) {
            Int_t destBin=binMap[i];
            bgrHist->SetBinError(destBin,hypot((*bgrerrscale)(i,0),
                                               bgrHist->GetBinError(destBin)));
         }
      }
   }
}

void TUnfoldSys::InitTUnfoldSys(void)
{
   // initialize pointers and TMaps

   // input
   fDAinRelSq = 0;
   fDAinColRelSq = 0;
   fAoutside = 0;
   fBgrIn = new TMap();
   fBgrErrUncorrInSq = new TMap();
   fBgrErrScaleIn = new TMap();
   fSysIn = new TMap();
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
   fBgrIn->SetOwnerKeyValue();
   fBgrErrUncorrInSq->SetOwnerKeyValue();
   fBgrErrScaleIn->SetOwnerKeyValue();
   fSysIn->SetOwnerKeyValue();
#else
   fBgrIn->SetOwner();
   fBgrErrUncorrInSq->SetOwner();
   fBgrErrScaleIn->SetOwner();
   fSysIn->SetOwner();
#endif
   // results
   fEmatUncorrX = 0;
   fEmatUncorrAx = 0;
   fDeltaCorrX = new TMap();
   fDeltaCorrAx = new TMap();
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
   fDeltaCorrX->SetOwnerKeyValue();
   fDeltaCorrAx->SetOwnerKeyValue();
#else
   fDeltaCorrX->SetOwner();
   fDeltaCorrAx->SetOwner();
#endif
   fDeltaSysTau = 0;
   fDtau=0.0;
   fYData=0;
   fVyyData=0;
}

TUnfoldSys::~TUnfoldSys(void)
{
   // delete all data members
   DeleteMatrix(&fDAinRelSq);
   DeleteMatrix(&fDAinColRelSq);
   delete fBgrIn;
   delete fBgrErrUncorrInSq;
   delete fBgrErrScaleIn;
   delete fSysIn;
   ClearResults();
   delete fDeltaCorrX;
   delete fDeltaCorrAx;
   DeleteMatrix(&fYData);
   DeleteMatrix(&fVyyData);
}

void TUnfoldSys::ClearResults(void)
{
   // clear all data members which depend on the unfolding results
   TUnfold::ClearResults();
   DeleteMatrix(&fEmatUncorrX);
   DeleteMatrix(&fEmatUncorrAx);
   fDeltaCorrX->Clear();
   fDeltaCorrAx->Clear();
   DeleteMatrix(&fDeltaSysTau);
}

void TUnfoldSys::PrepareSysError(void)
{
   // calculations required for syst.error
   // data members modified
   //    fEmatUncorrX, fEmatUncorrAx, fDeltaCorrX, fDeltaCorrAx
   if(!fEmatUncorrX) {
      fEmatUncorrX=PrepareUncorrEmat(GetDXDAM(0),GetDXDAM(1));
   }
   TMatrixDSparse *AM0=0,*AM1=0;
   if(!fEmatUncorrAx) {
      if(!AM0) AM0=MultiplyMSparseMSparse(fA,GetDXDAM(0));
      if(!AM1) {
         AM1=MultiplyMSparseMSparse(fA,GetDXDAM(1));
         Int_t *rows_cols=new Int_t[GetNy()];
         Double_t *data=new Double_t[GetNy()];
         for(Int_t i=0;i<GetNy();i++) {
            rows_cols[i]=i;
            data[i]=1.0;
         }
         TMatrixDSparse *one=CreateSparseMatrix
            (GetNy(),GetNy(),GetNy(),rows_cols, rows_cols,data);
         delete[] data;
         delete[] rows_cols;
         AddMSparse(AM1,-1.,one);
         DeleteMatrix(&one);
         fEmatUncorrAx=PrepareUncorrEmat(AM0,AM1);
      }
   }
   if((!fDeltaSysTau )&&(fDtau>0.0)) {
      fDeltaSysTau=new TMatrixDSparse(*GetDXDtauSquared());
      Double_t scale=2.*TMath::Sqrt(fTauSquared)*fDtau;
      Int_t n=fDeltaSysTau->GetRowIndexArray() [fDeltaSysTau->GetNrows()];
      Double_t *data=fDeltaSysTau->GetMatrixArray();
      for(Int_t i=0;i<n;i++) {
         data[i] *= scale;
      }
   }

   TMapIter sysErrIn(fSysIn);
   const TObjString *key;

   // calculate individual systematic errors
   for(key=(const TObjString *)sysErrIn.Next();key;
       key=(const TObjString *)sysErrIn.Next()) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
      const TMatrixDSparse *dsys=
         (const TMatrixDSparse *)((const TPair *)*sysErrIn)->Value();
#else
      const TMatrixDSparse *dsys=
         (const TMatrixDSparse *)(fSysIn->GetValue(key->GetString()));
#endif
      const TPair *named_emat=(const TPair *)
         fDeltaCorrX->FindObject(key->GetString());
      if(!named_emat) {
         TMatrixDSparse *emat=PrepareCorrEmat(GetDXDAM(0),GetDXDAM(1),dsys);
         fDeltaCorrX->Add(new TObjString(*key),emat);
      }
      named_emat=(const TPair *)fDeltaCorrAx->FindObject(key->GetString());
      if(!named_emat) {
         if(!AM0) AM0=MultiplyMSparseMSparse(fA,GetDXDAM(0));
         if(!AM1) {
            AM1=MultiplyMSparseMSparse(fA,GetDXDAM(1));
            Int_t *rows_cols=new Int_t[GetNy()];
            Double_t *data=new Double_t[GetNy()];
            for(Int_t i=0;i<GetNy();i++) {
               rows_cols[i]=i;
               data[i]=1.0;
            }
            TMatrixDSparse *one=CreateSparseMatrix
               (GetNy(),GetNy(),GetNy(),rows_cols, rows_cols,data);
            delete[] data;
            delete[] rows_cols;
            AddMSparse(AM1,-1.,one);
            DeleteMatrix(&one);
            fEmatUncorrAx=PrepareUncorrEmat(AM0,AM1);
         }
         TMatrixDSparse *emat=PrepareCorrEmat(AM0,AM1,dsys);
         fDeltaCorrAx->Add(new TObjString(*key),emat);
      }
   }
   DeleteMatrix(&AM0);
   DeleteMatrix(&AM1);
}

void TUnfoldSys::GetEmatrixSysUncorr
(TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat)
{
   // get output error contribution from statistical fluctuations in A
   //   ematrix: output error matrix histogram
   //   binMap: see method GetEmatrix()
   //   clearEmat: set kTRUE to clear the histogram prior to adding the errors
   // data members modified:
   //   fVYAx, fESparse, fEAtV, fErrorAStat
   PrepareSysError();
   ErrorMatrixToHist(ematrix,fEmatUncorrX,binMap,clearEmat);
}

TMatrixDSparse *TUnfoldSys::PrepareUncorrEmat
(const TMatrixDSparse *m_0,const TMatrixDSparse *m_1)
{
   // propagate uncorrelated systematic errors to a covariance matrix
   //   m0,m1 : coefficients (matrices) for propagating the errors
   //
   // the error matrix is calculated by standard error propagation, where the
   // derivative of the result vector X wrt the matrix A is given by
   //
   //  dX_k / dA_ij  =  M0_kj * Z0_i  - M1_ki * Z1_j
   //
   // where:
   //   the matrices M0 and M1 are arguments to this function
   //   the vectors Z0, Z1 : GetDXDAZ()
   //
   // The matrix A is calculated from a matrix B as
   //
   //    A_ij = B_ij / sum_k B_kj
   //
   // where k runs over additional indices of B, not present in A.
   // (underflow and overflow bins, used for efficiency corrections)
   //
   // define:   Norm_j = sum_k B_kj   (data member fSumOverY)
   //
   // the derivative of A wrt this input matrix B is given by:
   //
   //   dA_ij / dB_kj = (  delta_ik - A_ij ) * 1/Norm_j
   //
   // The covariance matrix Vxx is:
   //
   //   Vxx_mn  = sum_ijlk [   (dX_m / dA_ij) * (dA_ij / dB_kj) * DB_kj
   //                        * (dX_n / dA_lj) * (dA_lj / dB_kj)  ]
   //
   // where DB_kj is the error on B_kj squared
   // Simplify the sum over k:
   //
   //   sum_k [ (dA_ij / dB_kj) * DB_kj * (dA_lj / dB_kj) ]
   //      =  sum_k [  ( delta_ik - A_ij ) * 1/Norm_j * DB_kj *
   //                * ( delta_lk - A_lj ) * 1/Norm_j ]
   //      =  sum_k [ ( delta_ik*delta_lk - delta_ik*A_lj - delta_lk*A_ij
   //                  + A_ij * A_lj ) * DB_kj / Norm_j^2 ]
   //
   // introduce normalized errors:  Rsq_kj = DB_kj / Norm_j^2
   // after summing over k:
   //   delta_ik*delta_lk*Rsq_kj  ->    delta_il*Rsq_ij
   //   delta_ik*A_lj*Rsq_kj      ->    A_lj*Rsq_ij
   //   delta_lk*A_ij*Rsq_kj      ->    A_ij*Rsq_lj
   //   A_ij*A_lj*Rsq_kj          ->    A_ij*A_lj*sum_k(Rsq_kj)
   //
   // introduce sum of normalized errors squared:   SRsq_j = sum_k(Rsq_kj)
   //
   // Note: Rsq_ij is stored as  fDAinRelSq     (excludes extra indices of B)
   //   and SRsq_j is stored as  fDAinColRelSq  (sum includes all indices of B)
   //
   //  Vxx_nm = sum_ijl [ (dX_m / dA_ij) * (dX_n / dA_lj)
   //     (delta_il*Rsq_ij - A_lj*Rsq_ij - A_ij*Rsq_lj + A_ij*A_lj *SRsq_j) ]
   //
   //  Vxx_nm =    sum_j [ F_mj * F_nj * SRsq_j
   //            - sum_j [ G_mj * F_nj ]
   //            - sum_j [ F_mj * G_nj ]
   //            + sum_ij [  (dX_m / dA_ij) * (dX_n / dA_lj) * Rsq_ij ]
   //
   // where:
   //    F_mj = sum_i [ (dX_m / dA_ij) * A_ij ]
   //    G_mj = sum_i [ (dX_m / dA_ij) * Rsq_ij ]
   //
   // In order to avoid explicitly calculating the 3-dimensional tensor
   // (dX_m/dA_ij) the sums are evaluated further, using
   //    dX_k / dA_ij  =  M0_kj * Z0_i  - M1_ki * Z1_j
   //
   //   F_mj = M0_mj * (A# Z0)_j - (M1 A)_mj Z1_j
   //   G_mj = M0_mj * (Rsq# Z0)_j - (M1 Rsq)_mj Z1_j
   //
   // and
   //
   //   sum_ij [ (dX_m/dA_ij) * (dX_n/dA_ij) * Rsq_ij ] =
   //      sum_j [ M0_mj * M0_nj *  [ sum_i (Z0_i)^2 * Rsq_ij ] ]
   //    + sum_i [ M1_mi * M1_ni *  [ sum_j (Z1_j)^2 * Rsq_ij ] ]
   //    - sum_i [ M1_mi * H_ni + M1_ni * H_mi]
   // where:
   //   H_mi = Z0_i * sum_j [ M0_mj * Z1_j * Rsq_ij ]
   //
   // collect all contributions:
   //   Vxx_nm = r0 -r1 -r2 +r3 +r4 -r5 -r6
   //      r0 = sum_j [ F_mj * F_nj * SRsq_j ]
   //      r1 = sum_j [ G_mj * F_nj ]
   //      r2 = sum_j [ F_mj * G_nj ]
   //      r3 = sum_j [ M0_mj * M0_nj *  [ sum_i (Z0_i)^2 * Rsq_ij ] ]
   //      r4 = sum_i [ M1_mi * M1_ni *  [ sum_j (Z1_j)^2 * Rsq_ij ] ]
   //      r5 = sum_i [ M1_mi * H_ni ]
   //      r6 = sum_i [ M1_ni * H_mi ]

   //======================================================
   // calculate contributions containing matrices F and G
   // r0,r1,r2
   TMatrixDSparse *r=0;
   if(fDAinColRelSq && fDAinRelSq) {
      // calculate matrices (M1*A)_mj * Z1_j  and  (M1*Rsq)_mj * Z1_j
      TMatrixDSparse *M1A_Z1=MultiplyMSparseMSparse(m_1,fA);
      ScaleColumnsByVector(M1A_Z1,GetDXDAZ(1));
      TMatrixDSparse *M1Rsq_Z1=MultiplyMSparseMSparse(m_1,fDAinRelSq);
      ScaleColumnsByVector(M1Rsq_Z1,GetDXDAZ(1));
      // calculate vectors A#*Z0  and  Rsq#*Z0
      TMatrixDSparse *AtZ0 = MultiplyMSparseTranspMSparse(fA,GetDXDAZ(0));
      TMatrixDSparse *RsqZ0=
         MultiplyMSparseTranspMSparse(fDAinRelSq,GetDXDAZ(0));
      //calculate matrix F
      //   F_mj = M0_mj * (A# Z0)_j - (M1 A)_mj Z1_j
      TMatrixDSparse *F=new TMatrixDSparse(*m_0);
      ScaleColumnsByVector(F,AtZ0);
      AddMSparse(F,-1.0,M1A_Z1);
      //calculate matrix G
      //   G_mj = M0_mj * (Rsq# Z0)_j - (M1 Rsq)_mj Z1_j
      TMatrixDSparse *G=new TMatrixDSparse(*m_0);
      ScaleColumnsByVector(G,RsqZ0);
      AddMSparse(G,-1.0,M1Rsq_Z1);
      DeleteMatrix(&M1A_Z1);
      DeleteMatrix(&M1Rsq_Z1);
      DeleteMatrix(&AtZ0);
      DeleteMatrix(&RsqZ0);
      //      r0 = sum_j [ F_mj * F_nj * SRsq_j ]
      r=MultiplyMSparseMSparseTranspVector(F,F,fDAinColRelSq);
      //      r1 = sum_j [ G_mj * F_nj ]
      TMatrixDSparse *r1=MultiplyMSparseMSparseTranspVector(F,G,0);
      //      r2 = sum_j [ F_mj * G_nj ]
      TMatrixDSparse *r2=MultiplyMSparseMSparseTranspVector(G,F,0);
      // r = r0-r1-r2
      AddMSparse(r,-1.0,r1);
      AddMSparse(r,-1.0,r2);
      DeleteMatrix(&r1);
      DeleteMatrix(&r2);
      DeleteMatrix(&F);
      DeleteMatrix(&G);
   }
   //======================================================
   // calculate contribution
   //   sum_ij [ (dX_m/dA_ij) * (dX_n/dA_ij) * Rsq_ij ]
   //  (r3,r4,r5,r6)
   if(fDAinRelSq) {
      // (Z0_i)^2
      TMatrixDSparse Z0sq(*GetDXDAZ(0));
      const Int_t *Z0sq_rows=Z0sq.GetRowIndexArray();
      Double_t *Z0sq_data=Z0sq.GetMatrixArray();
      for(int index=0;index<Z0sq_rows[Z0sq.GetNrows()];index++) {
         Z0sq_data[index] *= Z0sq_data[index];
      }
      // Z0sqRsq =  sum_i (Z_i)^2 * Rsq_ij
      TMatrixDSparse *Z0sqRsq=MultiplyMSparseTranspMSparse(fDAinRelSq,&Z0sq);
      //      r3 = sum_j [ M0_mj * M0_nj *  [ sum_i (Z0_i)^2 * Rsq_ij ] ]
      TMatrixDSparse *r3=MultiplyMSparseMSparseTranspVector(m_0,m_0,Z0sqRsq);
      DeleteMatrix(&Z0sqRsq);

      // (Z1_j)^2
      TMatrixDSparse Z1sq(*GetDXDAZ(1));
      const Int_t *Z1sq_rows=Z1sq.GetRowIndexArray();
      Double_t *Z1sq_data=Z1sq.GetMatrixArray();
      for(int index=0;index<Z1sq_rows[Z1sq.GetNrows()];index++) {
         Z1sq_data[index] *= Z1sq_data[index];
      }
      // Z1sqRsq = sum_j (Z1_j)^2 * Rsq_ij ]
      TMatrixDSparse *Z1sqRsq=MultiplyMSparseMSparse(fDAinRelSq,&Z1sq);
      //      r4 = sum_i [ M1_mi * M1_ni *  [ sum_j (Z1_j)^2 * Rsq_ij ] ]
      TMatrixDSparse *r4=MultiplyMSparseMSparseTranspVector(m_1,m_1,Z1sqRsq);
      DeleteMatrix(&Z1sqRsq);

      // sum_j [ M0_mj * Z1_j * Rsq_ij ]
      TMatrixDSparse *H=MultiplyMSparseMSparseTranspVector
         (m_0,fDAinRelSq,GetDXDAZ(1));
      // H_mi = Z0_i * sum_j [ M0_mj * Z1_j * Rsq_ij ]
      ScaleColumnsByVector(H,GetDXDAZ(0));
      //      r5 = sum_i [ M1_mi * H_ni ]
      TMatrixDSparse *r5=MultiplyMSparseMSparseTranspVector(m_1,H,0);
      //      r6 = sum_i [ H_mi * M1_ni ]
      TMatrixDSparse *r6=MultiplyMSparseMSparseTranspVector(H,m_1,0);
      DeleteMatrix(&H);
      // r =  r0 -r1 -r2 +r3 +r4 +r5 +r6
      if(r) {
         AddMSparse(r,1.0,r3);
         DeleteMatrix(&r3);
      } else {
         r=r3;
         r3=0;
      }
      AddMSparse(r,1.0,r4);
      AddMSparse(r,-1.0,r5);
      AddMSparse(r,-1.0,r6);
      DeleteMatrix(&r4);
      DeleteMatrix(&r5);
      DeleteMatrix(&r6);
   }
   return r;
}

TMatrixDSparse *TUnfoldSys::PrepareCorrEmat
(const TMatrixDSparse *m1,const TMatrixDSparse *m2,const TMatrixDSparse *dsys)
{
   // propagate correlated systematic shift to output vector
   //   m1,m2 : coefficients for propagating the errors
   //   dsys : matrix of correlated shifts from this source

   // delta_m =
   //   sum{i,j}   {
   //      ((*m1)(m,j) * (*fVYAx)(i) - (*m2)(m,i) * (*fX)(j))*dsys(i,j) }
   //   =    sum_j (*m1)(m,j)  sum_i dsys(i,j) * (*fVYAx)(i)
   //     -  sum_i (*m2)(m,i)  sum_j dsys(i,j) * (*fX)(j)

   TMatrixDSparse *dsysT_VYAx = MultiplyMSparseTranspMSparse(dsys,GetDXDAZ(0));
   TMatrixDSparse *delta =  MultiplyMSparseMSparse(m1,dsysT_VYAx);
   DeleteMatrix(&dsysT_VYAx);
   TMatrixDSparse *dsys_X = MultiplyMSparseMSparse(dsys,GetDXDAZ(1));
   TMatrixDSparse *delta2 = MultiplyMSparseMSparse(m2,dsys_X);
   DeleteMatrix(&dsys_X);
   AddMSparse(delta,-1.0,delta2);
   DeleteMatrix(&delta2);
   return delta;
}

void TUnfoldSys::SetTauError(Double_t delta_tau)
{
   // set uncertainty on tau
   fDtau=delta_tau;
   DeleteMatrix(&fDeltaSysTau);
}

Bool_t TUnfoldSys::GetDeltaSysSource(TH1 *hist_delta,const char *name,
                                   const Int_t *binMap)
{
   // calculate systematic shift from a given source
   //    ematrix: output
   //    source: name of the error source
   //    binMap: see method GetEmatrix()
   PrepareSysError();
   const TPair *named_emat=(const TPair *)fDeltaCorrX->FindObject(name);
   const TMatrixDSparse *delta=0;
   if(named_emat) {
      delta=(TMatrixDSparse *)named_emat->Value();
   }
   VectorMapToHist(hist_delta,delta,binMap);
   return delta !=0;
}

Bool_t TUnfoldSys::GetDeltaSysBackgroundScale
(TH1 *hist_delta,const char *source,const Int_t *binMap)
{
   // get correlated shift induced by a background source
   //   delta: output shift vector histogram
   //   source: name of background source
   //   binMap: see method GetEmatrix()
   //   see PrepareSysError()
   PrepareSysError();
   const TPair *named_err=(const TPair *)fBgrErrScaleIn->FindObject(source);
   TMatrixDSparse *dx=0;
   if(named_err) {
      const TMatrixD *dy=(TMatrixD *)named_err->Value();
      dx=MultiplyMSparseM(GetDXDY(),dy);
   }
   VectorMapToHist(hist_delta,dx,binMap);
   if(dx!=0) {
      DeleteMatrix(&dx);
      return kTRUE;
   }
   return kFALSE;
}

Bool_t TUnfoldSys::GetDeltaSysTau(TH1 *hist_delta,const Int_t *binMap)
{
   // calculate systematic shift from tau variation
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   PrepareSysError();
   VectorMapToHist(hist_delta,fDeltaSysTau,binMap);
   return fDeltaSysTau !=0;
}

void TUnfoldSys::GetEmatrixSysSource
(TH2 *ematrix,const char *name,const Int_t *binMap,Bool_t clearEmat)
{
   // calculate systematic shift from a given source
   //    ematrix: output
   //    source: name of the error source
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   PrepareSysError();
   const TPair *named_emat=(const TPair *)fDeltaCorrX->FindObject(name);
   TMatrixDSparse *emat=0;
   if(named_emat) {
      const TMatrixDSparse *delta=(TMatrixDSparse *)named_emat->Value();
      emat=MultiplyMSparseMSparseTranspVector(delta,delta,0);
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
   DeleteMatrix(&emat);
}

void TUnfoldSys::GetEmatrixSysBackgroundScale
(TH2 *ematrix,const char *name,const Int_t *binMap,Bool_t clearEmat)
{
   // calculate systematic shift from a given background scale error
   //    ematrix: output
   //    source: name of the error source
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   PrepareSysError();
   const TPair *named_err=(const TPair *)fBgrErrScaleIn->FindObject(name);
   TMatrixDSparse *emat=0;
   if(named_err) {
      const TMatrixD *dy=(TMatrixD *)named_err->Value();
      TMatrixDSparse *dx=MultiplyMSparseM(GetDXDY(),dy);
      emat=MultiplyMSparseMSparseTranspVector(dx,dx,0);
      DeleteMatrix(&dx);
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
   DeleteMatrix(&emat);
}

void TUnfoldSys::GetEmatrixSysTau
(TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat)
{
   // calculate error matrix from error in regularisation parameter
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   PrepareSysError();
   TMatrixDSparse *emat=0;
   if(fDeltaSysTau) {
      emat=MultiplyMSparseMSparseTranspVector(fDeltaSysTau,fDeltaSysTau,0);
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
   DeleteMatrix(&emat);
}

void TUnfoldSys::GetEmatrixInput
(TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat)
{
   // calculate error matrix from error in input vector alone
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   GetEmatrixFromVyy(fVyyData,ematrix,binMap,clearEmat);
}

void TUnfoldSys::GetEmatrixSysBackgroundUncorr
(TH2 *ematrix,const char *source,const Int_t *binMap,Bool_t clearEmat)
{
   // calculate error matrix contribution originating from uncorrelated errors
   // of one background source
   //    ematrix: output
   //    source: name of the error source
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   const TPair *named_err=(const TPair *)fBgrErrUncorrInSq->FindObject(source);
   TMatrixDSparse *emat=0;
   if(named_err) {
      TMatrixD const *dySquared=(TMatrixD const *)named_err->Value();
      emat=MultiplyMSparseMSparseTranspVector(GetDXDY(),GetDXDY(),dySquared);
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
   DeleteMatrix(&emat);
}

void TUnfoldSys::GetEmatrixFromVyy
(const TMatrixDSparse *vyy,TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat)
{
   // propagate error matrix vyy to the result
   //    vyy: error matrix on input data fY
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   PrepareSysError();
   TMatrixDSparse *em=0;
   if(vyy) {
      TMatrixDSparse *dxdyVyy=MultiplyMSparseMSparse(GetDXDY(),vyy);
      em=MultiplyMSparseMSparseTranspVector(dxdyVyy,GetDXDY(),0);
      DeleteMatrix(&dxdyVyy);
   }
   ErrorMatrixToHist(ematrix,em,binMap,clearEmat);
   DeleteMatrix(&em);
}

void TUnfoldSys::GetEmatrixTotal(TH2 *ematrix,const Int_t *binMap)
{
   // get total error including statistical error
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   GetEmatrix(ematrix,binMap);  // (stat)+(d)+(e)
   GetEmatrixSysUncorr(ematrix,binMap,kFALSE); // (a)
   TMapIter sysErrPtr(fDeltaCorrX);
   const TObject *key;

   for(key=sysErrPtr.Next();key;key=sysErrPtr.Next()) {
      GetEmatrixSysSource(ematrix,
                          ((const TObjString *)key)->GetString(),
                          binMap,kFALSE); // (b)
   }
   GetEmatrixSysTau(ematrix,binMap,kFALSE); // (c)
}

TMatrixDSparse *TUnfoldSys::GetSummedErrorMatrixYY(void)
{
   PrepareSysError();

   // errors from input vector and from background subtraction
   TMatrixDSparse *emat_sum=new TMatrixDSparse(*fVyy);

   // uncorrelated systematic error
   AddMSparse(emat_sum,1.0,fEmatUncorrAx);
   TMapIter sysErrPtr(fDeltaCorrAx);
   const TObject *key;

   // correlated systematic errors
   for(key=sysErrPtr.Next();key;key=sysErrPtr.Next()) {
      const TMatrixDSparse *delta=(TMatrixDSparse *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
                 ((const TPair *)*sysErrPtr)->Value()
#else
         fDeltaCorrAx->GetValue(((const TObjString *)key)
                                ->GetString())
#endif
         ;
      TMatrixDSparse *emat=MultiplyMSparseMSparseTranspVector(delta,delta,0);
      AddMSparse(emat_sum,1.0,emat);
      DeleteMatrix(&emat);
   }
   // error on tau
   if(fDeltaSysTau) {
      TMatrixDSparse *Adx_tau=MultiplyMSparseMSparse(fA,fDeltaSysTau);
      TMatrixDSparse *emat_tau=
         MultiplyMSparseMSparseTranspVector(Adx_tau,Adx_tau,0);
      DeleteMatrix(&Adx_tau);
      AddMSparse(emat_sum,1.0,emat_tau);
      DeleteMatrix(&emat_tau);
   }
   return emat_sum;
}

TMatrixDSparse *TUnfoldSys::GetSummedErrorMatrixXX(void)
{
   PrepareSysError();

   // errors from input vector and from background subtraction
   TMatrixDSparse *emat_sum=new TMatrixDSparse(*GetVxx());

   // uncorrelated systematic error
   AddMSparse(emat_sum,1.0,fEmatUncorrX);
   TMapIter sysErrPtr(fDeltaCorrX);
   const TObject *key;

   // correlated systematic errors
   for(key=sysErrPtr.Next();key;key=sysErrPtr.Next()) {
      const TMatrixDSparse *delta=(TMatrixDSparse *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
                 ((const TPair *)*sysErrPtr)->Value()
#else
         fDeltaCorrX->GetValue(((const TObjString *)key)
                                ->GetString())
#endif
         ;
      TMatrixDSparse *emat=MultiplyMSparseMSparseTranspVector(delta,delta,0);
      AddMSparse(emat_sum,1.0,emat);
      DeleteMatrix(&emat);
   }
   // error on tau
   if(fDeltaSysTau) {
      TMatrixDSparse *emat_tau=
         MultiplyMSparseMSparseTranspVector(fDeltaSysTau,fDeltaSysTau,0);
      AddMSparse(emat_sum,1.0,emat_tau);
      DeleteMatrix(&emat_tau);
   }
   return emat_sum;
}


Double_t TUnfoldSys::GetChi2Sys(void)
{
   // calculate total chi**2 including systematic errors

   TMatrixDSparse *emat_sum=GetSummedErrorMatrixYY();

   Int_t rank=0;
   TMatrixDSparse *v=InvertMSparseSymmPos(emat_sum,&rank);
   TMatrixD dy(*fY, TMatrixD::kMinus, *GetAx());
   TMatrixDSparse *vdy=MultiplyMSparseM(v,&dy);
   DeleteMatrix(&v);
   Double_t r=0.0;
   const Int_t *vdy_rows=vdy->GetRowIndexArray();
   const Double_t *vdy_data=vdy->GetMatrixArray();
   for(Int_t i=0;i<vdy->GetNrows();i++) {
      if(vdy_rows[i+1]>vdy_rows[i]) {
         r += vdy_data[vdy_rows[i]] * dy(i,0);
      }
   }
   DeleteMatrix(&vdy);
   DeleteMatrix(&emat_sum);
   return r;
}

void TUnfoldSys::GetRhoItotal(TH1 *rhoi,const Int_t *binMap,TH2 *invEmat)
{
   // get global correlation coefficients including systematic,statistical,background,tau errors
   //    rhoi: output histogram
   //    binMap: for each global bin, indicate in which histogram bin
   //            to store its content
   //    invEmat: output histogram for inverse of error matrix
   //              (pointer may zero if inverse is not requested)
   ClearHistogram(rhoi,-1.);
   TMatrixDSparse *emat_sum=GetSummedErrorMatrixXX();
   GetRhoIFromMatrix(rhoi,emat_sum,binMap,invEmat);

   DeleteMatrix(&emat_sum);
}

void TUnfoldSys::ScaleColumnsByVector
(TMatrixDSparse *m,const TMatrixTBase<Double_t> *v) const
{
   // scale columns of m by the corresponding rows of v
   // input:
   //   m:  pointer to sparse matrix of dimension NxM
   //   v:  pointer to matrix of dimension Mx1
   if((m->GetNcols() != v->GetNrows())||(v->GetNcols()!=1)) {
      Fatal("ScaleColumnsByVector error",
            "matrix cols/vector rows %d!=%d OR vector cols %d !=1\n",
            m->GetNcols(),v->GetNrows(),v->GetNcols());
   }
   const Int_t *rows_m=m->GetRowIndexArray();
   const Int_t *cols_m=m->GetColIndexArray();
   Double_t *data_m=m->GetMatrixArray();
   const TMatrixDSparse *v_sparse=dynamic_cast<const TMatrixDSparse *>(v);
   if(v_sparse) {
      const Int_t *rows_v=v_sparse->GetRowIndexArray();
      const Double_t *data_v=v_sparse->GetMatrixArray();
      for(Int_t i=0;i<m->GetNrows();i++) {
         for(Int_t index_m=rows_m[i];index_m<rows_m[i+1];index_m++) {
            Int_t j=cols_m[index_m];
            Int_t index_v=rows_v[j];
            if(index_v<rows_v[j+1]) {
               data_m[index_m] *= data_v[index_v];
            } else {
               data_m[index_m] =0.0;
            }
         }
      }
   } else {
      for(Int_t i=0;i<m->GetNrows();i++) {
         for(Int_t index=rows_m[i];index<rows_m[i+1];index++) {
            data_m[index] *= (*v)(cols_m[index],0);
         }
      }
   }
}

void TUnfoldSys::VectorMapToHist
(TH1 *hist_delta,const TMatrixDSparse *delta,const Int_t *binMap)
{
   // sum over bins of *delta, as defined in binMap,fXToHist
   //   hist_delta: histogram to return summed vector
   //   delta: vector to sum and remap
   Int_t nbin=hist_delta->GetNbinsX();
   Double_t *c=new Double_t[nbin+2];
   for(Int_t i=0;i<nbin+2;i++) {
      c[i]=0.0;
   }
   if(delta) {
      Int_t binMapSize = fHistToX.GetSize();
      const Double_t *delta_data=delta->GetMatrixArray();
      const Int_t *delta_rows=delta->GetRowIndexArray();
      for(Int_t i=0;i<binMapSize;i++) {
         Int_t destBinI=binMap ? binMap[i] : i;
         Int_t srcBinI=fHistToX[i];
         if((destBinI>=0)&&(destBinI<nbin+2)&&(srcBinI>=0)) {
            Int_t index=delta_rows[srcBinI];
            if(index<delta_rows[srcBinI+1]) {
               c[destBinI]+=delta_data[index];
            }
         }
      }
   }
   for(Int_t i=0;i<nbin+2;i++) {
      hist_delta->SetBinContent(i,c[i]);
      hist_delta->SetBinError(i,0.0);
   }
   delete[] c;
}
