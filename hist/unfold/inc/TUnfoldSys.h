// Author: Stefan Schmitt
// DESY, 23/01/09

//  Version 17.5, bug fixes in TUnfold fix problem with GetEmatrixSysUncorr
//
//  History:
//    Version 17.4, in parallel to changes in TUnfoldBinning
//    Version 17.3, in parallel to changes in TUnfoldBinning
//    Version 17.2, add methods to find back systematic and background sources
//    Version 17.1, bug fix with background uncertainty
//    Version 17.0, possibility to specify an error matrix with SetInput
//    Version 16.2, bug-fix with the calculation of background errors
//    Version 16.1, parallel to changes in TUnfold
//    Version 16.0, parallel to changes in TUnfold
//    Version 15, fix bugs with uncorr. uncertainties, add backgnd subtraction
//    Version 14, with changes in TUnfoldSys.cxx
//    Version 13, support for systematic errors

#ifndef ROOT_TUnfoldSys
#define ROOT_TUnfoldSys

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//  TUnfoldSys, an extension of the class TUnfold to correct for        //
//  migration effects. It provides methods for background subtraction   //
//  and propagation of systematic uncertainties                         //
//                                                                      //
//  Citation: S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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

#include <TMap.h>
#include <TSortedList.h>
#include "TUnfold.h"


class TUnfoldSys : public TUnfold {
 private:
   void InitTUnfoldSys(void);     // initialize all data members
 protected:
   /// Input: normalized errors from input matrix
   TMatrixDSparse *fDAinRelSq;
   /// Input: normalized column err.sq. (inp.matr.)
   TMatrixD* fDAinColRelSq;
   /// Input: underflow/overflow bins
   TMatrixD* fAoutside;
   /// Input: correlated errors
   TMap *fSysIn;
   /// Input: size of background sources
   TMap *fBgrIn;
   /// Input: uncorr error squared from bgr sources
   TMap *fBgrErrUncorrInSq;
   /// Input: background sources correlated error
   TMap *fBgrErrScaleIn;
   /// Input: error on tau
   Double_t fDtau;
   /// Input: fY prior to bgr subtraction
   TMatrixD *fYData;
   /// Input: error on fY prior to bgr subtraction
   TMatrixDSparse *fVyyData;
   /// Result: syst.error from fDA2 on fX
   TMatrixDSparse *fEmatUncorrX;
   /// Result: syst.error from fDA2 on fAx
   TMatrixDSparse *fEmatUncorrAx;
   /// Result: syst.shift from fSysIn on fX
   TMap *fDeltaCorrX;
   /// Result: syst.shift from fSysIn on fAx
   TMap *fDeltaCorrAx;
   /// Result: systematic shift from tau
   TMatrixDSparse *fDeltaSysTau;
 protected:
   void ClearResults(void) override;     // clear all results
   virtual void PrepareSysError(void); // common calculations for syst.errors
   virtual TMatrixDSparse *PrepareUncorrEmat(const TMatrixDSparse *m1,const TMatrixDSparse *m2); // calculate uncorrelated error matrix
   virtual TMatrixDSparse *PrepareCorrEmat(const TMatrixDSparse *m1,const TMatrixDSparse *m2,const TMatrixDSparse *dsys); // calculate correlated error matrix
   void ScaleColumnsByVector(TMatrixDSparse *m,const TMatrixTBase<Double_t> *v) const; // scale columns of m by the corresponding rows of v
   void VectorMapToHist(TH1 *hist_delta,const TMatrixDSparse *delta,const Int_t  *binMap); // map and sum vector delta, save in hist_delta
   void GetEmatrixFromVyy(const TMatrixDSparse *vyy,TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat); // propagate error matrix vyy to the result
   void DoBackgroundSubtraction(void);
   TMatrixDSparse *GetSummedErrorMatrixYY(void);
   TMatrixDSparse *GetSummedErrorMatrixXX(void);
 public:
   /// type of matrix specified with AddSysError()
   enum ESysErrMode {
      /// matrix is an alternative to the default matrix, the errors are the difference to the original matrix
     kSysErrModeMatrix=0,
     /// matrix gives the absolute shifts
     kSysErrModeShift=1,
     /// matrix gives the relative shifts
     kSysErrModeRelative=2
   };
   TUnfoldSys(const TH2 *hist_A, EHistMap histmap, ERegMode regmode = kRegModeSize,
             EConstraint constraint=kEConstraintArea);      // constructor
   TUnfoldSys(void);            // for derived classes
   ~ TUnfoldSys(void) override;    // delete data members
   void AddSysError(const TH2 *sysError,const char *name, EHistMap histmap,
                    ESysErrMode mode); // add a systematic error source
   void SubtractBackground(const TH1 *hist_bgr,const char *name,
                           Double_t scale=1.0,
                           Double_t scale_error=0.0); // subtract background prior to unfolding
   Int_t SetInput(const TH1 *hist_y,Double_t scaleBias=0.0,Double_t oneOverZeroError=0.0,const TH2 *hist_vyy=nullptr,const TH2 *hist_vyy_inv=nullptr) override; // define input consistently in case of background subtraction
   void SetTauError(Double_t delta_tau); // set uncertainty on tau
   TSortedList *GetBgrSources(void) const; // get names of background sources
   TSortedList *GetSysSources(void) const; // get names of systematic sources
   void GetBackground(TH1 *bgr,const char *bgrSource=nullptr,const Int_t *binMap=nullptr,Int_t includeError=3,Bool_t clearHist=kTRUE) const; // get background as histogram
   void GetEmatrixSysBackgroundUncorr(TH2 *ematrix,const char *source,
                                   const Int_t *binMap=nullptr,Bool_t clearEmat=kTRUE); // get error matrix from uncorrelated error of one background source
   void GetEmatrixSysBackgroundScale(TH2 *ematrix,const char *source,
                                  const Int_t *binMap=nullptr,Bool_t clearEmat=kTRUE); // get error matrix from the scale error of one background source
   Bool_t GetDeltaSysBackgroundScale(TH1 *delta,const char *source,
                                const Int_t *binMap=nullptr); // get correlated uncertainty induced by the scale uncertainty of a background source
   void GetEmatrixSysUncorr(TH2 *ematrix,const Int_t *binMap=nullptr,Bool_t clearEmat=kTRUE); // get error matrix contribution from uncorrelated errors on the matrix A
   void GetEmatrixSysSource(TH2 *ematrix,const char *source,
                            const Int_t *binMap=nullptr,Bool_t clearEmat=kTRUE); // get error matrix from one systematic source
   Bool_t GetDeltaSysSource(TH1 *hist_delta,const char *source,
                          const Int_t *binMap=nullptr); // get systematic shifts from one systematic source
   void GetEmatrixSysTau(TH2 *ematrix,
                      const Int_t *binMap=nullptr,Bool_t clearEmat=kTRUE); // get error matrix from tau variation
   Bool_t GetDeltaSysTau(TH1 *delta,const Int_t *binMap=nullptr); // get correlated uncertainty from varying tau
   void GetEmatrixInput(TH2 *ematrix,const Int_t *binMap=nullptr,Bool_t clearEmat=kTRUE); // get error contribution from input vector
   void GetEmatrixTotal(TH2 *ematrix,const Int_t *binMap=nullptr); // get total error including systematic,statistical,background,tau errors
   void GetRhoItotal(TH1 *rhoi,const Int_t *binMap=nullptr,TH2 *invEmat=nullptr); // get global correlation coefficients including systematic,statistical,background,tau errors
   Double_t GetChi2Sys(void); // get total chi**2 including all systematic errors
   ClassDefOverride(TUnfoldSys, TUnfold_CLASS_VERSION) //Unfolding with support for systematic error propagation
};

#endif
