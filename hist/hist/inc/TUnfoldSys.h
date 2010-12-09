// @(#)root/hist:$Id$
// Author: Stefan Schmitt
// DESY, 23/01/09

//  Version 16, parallel to changes in TUnfold
//
//  History:
//     Version 15, fix bugs with uncorr. uncertainties, add backgnd subtraction
//     Version 14, with changes in TUnfoldSys.cxx
//     Version 13, support for systematic errors

#ifndef ROOT_TUnfoldSys
#define ROOT_TUnfoldSys

#include <TUnfold.h>

class TMap;

class TUnfoldSys : public TUnfold {
 private:
   void InitTUnfoldSys(void);     // initialize all data members
 protected:
   TMatrixDSparse *fDAinRelSq;  // Input: normalized errors from input matrix
   TMatrixD* fDAinColRelSq;     // Input: normalized column err.sq. (inp.matr.)
   TMatrixD* fAoutside;         // Input: underflow/overflow bins
   TMap *fSysIn;                // Input: correlated errors
   TMap *fBgrIn;                // Input: size of background sources
   TMap *fBgrErrUncorrIn;       // Input: uncorrelated error from bgr sources
   TMap *fBgrErrCorrIn;         // Input: background sources correlated error
   Double_t fDtau;              // Input: error on tau
   TMatrixD *fYData;            // Input: fY prior to bgr subtraction
   TMatrixDSparse *fVyyData;    // Input: error on fY prior to bgr subtraction
   TMatrixDSparse *fEmatUncorrX;      // Result: syst.error from fDA2 on fX
   TMatrixDSparse *fEmatUncorrAx;     // Result: syst.error from fDA2 on fAx
   TMap *fDeltaCorrX;           // Result: syst.shift from fSysIn on fX
   TMap *fDeltaCorrAx;          // Result: syst.shift from fSysIn on fAx
   TMatrixDSparse *fDeltaSysTau; // Result: systematic shift from tau
 protected:
   TUnfoldSys(void);            // for derived classes
   virtual void ClearResults(void);     // clear all results
   virtual void PrepareSysError(void); // common calculations for syst.errors
   virtual TMatrixDSparse *PrepareUncorrEmat(const TMatrixDSparse *m1,const TMatrixDSparse *m2); // calculate uncorrelated error matrix
   virtual TMatrixDSparse *PrepareCorrEmat(const TMatrixDSparse *m1,const TMatrixDSparse *m2,const TMatrixDSparse *dsys); // calculate correlated error matrix
   void ScaleColumnsByVector(TMatrixDSparse *m,const TMatrixTBase<Double_t> *v) const; // scale columns of m by the corresponding rows of v
   void VectorMapToHist(TH1 *hist_delta,const TMatrixDSparse *delta,const Int_t  *binMap); // map and sum vector delta, save in hist_delta
   void GetEmatrixFromVyy(const TMatrixDSparse *vyy,TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat); // propagate error matrix vyy to the result
   void DoBackgroundSubtraction(void);
 public:
   enum ESysErrMode { // meaning of the argument to AddSysError()
     kSysErrModeMatrix=0, // matrix is an alternative to the default matrix, the errors are the difference to the original matrix
     kSysErrModeShift=1, // matrix gives the absolute shifts
     kSysErrModeRelative=2 // matrix gives the relative shifts
   };
   TUnfoldSys(const TH2 *hist_A, EHistMap histmap, ERegMode regmode = kRegModeSize,
             EConstraint constraint=kEConstraintNone);      // constructor
   virtual ~ TUnfoldSys(void);    // delete data members
   void AddSysError(const TH2 *sysError,const char *name, EHistMap histmap,
                    ESysErrMode mode); // add a systematic error source
   void GetDeltaSysSource(TH1 *hist_delta,const char *source,
                          const Int_t *binMap=0); // get systematic shifts from one systematic source
   void SubtractBackground(const TH1 *hist_bgr,const char *name,
                           Double_t scale=1.0,
                           Double_t scale_error=0.0); // subtract background prior to unfolding
   virtual Int_t SetInput(const TH1 *hist_y,Double_t scaleBias=0.0,Double_t oneOverZeroError=0.0); // define input consistently in case of background subtraction
   void GetDeltaSysBackgroundScale(TH1 *delta,const char *source,
                                const Int_t *binMap=0); // get correlated uncertainty induced by the scale uncertainty of a background source
   void SetTauError(Double_t delta_tau); // set uncertainty on tau
   void GetDeltaSysTau(TH1 *delta,const Int_t *binMap=0); // get correlated uncertainty from varying tau
   void GetEmatrixSysUncorr(TH2 *ematrix,const Int_t *binMap=0,Bool_t clearEmat=kTRUE); // get error matrix contribution from uncorrelated errors on the matrix A
   void GetEmatrixSysSource(TH2 *ematrix,const char *source,
                            const Int_t *binMap=0,Bool_t clearEmat=kTRUE); // get error matrix from one systematic source
   void GetEmatrixSysBackgroundUncorr(TH2 *ematrix,const char *source,
                                   const Int_t *binMap=0,Bool_t clearEmat=kTRUE); // get error matrix from uncorrelated error of one background source
   void GetEmatrixSysBackgroundScale(TH2 *ematrix,const char *source,
                                  const Int_t *binMap=0,Bool_t clearEmat=kTRUE); // get error matrix from the scale error of one background source
   void GetEmatrixSysTau(TH2 *ematrix,
                      const Int_t *binMap=0,Bool_t clearEmat=kTRUE); // get error matrix from tau variation
   void GetEmatrixInput(TH2 *ematrix,const Int_t *binMap=0,Bool_t clearEmat=kTRUE); // get error contribution from input vector
   void GetEmatrixTotal(TH2 *ematrix,const Int_t *binMap=0); // get total error including systematic,statistical,background,tau errors
   Double_t GetChi2Sys(void); // get total chi**2 including all systematic errors

   ClassDef(TUnfoldSys, 0) //Unfolding with support for systematic error propagation
};

#endif
