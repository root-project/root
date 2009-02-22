// Author: Stefan Schmitt
// DESY, 23/01/09

// Version 13, support for systematic errors

#ifndef ROOT_TUnfoldSys
#define ROOT_TUnfoldSys

#include <TUnfold.h>

class TMap;

class TUnfoldSys : public TUnfold {
 private:
   void InitTUnfoldSys(void);     // initialize all data members
 protected:
   TMatrixDSparse * fDA2;       // Input: uncorrelated errors on fA (squared)
   TMatrixD* fDAcol;            // Input: normalized column errors on fA
   TMatrixD* fAoutside;         // Input: underflow/overflow bins
   TMap *fSysIn;                // Input: correlated errors
   TMatrixDSparse *fESparse;    // Result: sparse version of fE
   TMatrixDSparse *fVYAx;       // Result: V(y-Ax) for syst.errors
   TMatrixDSparse *fEAtV;       // Result: EA#V    for syst.errors  
   TMatrixDSparse *fAE;         // Result: AE for chi**2
   TMatrixDSparse *fAEAtV_one;  // Result: AEA#V-1 for chi**2
   TMatrixD *fErrorUncorrX;     // Result: syst.error from fDA2 on fX
   TMatrixD *fErrorUncorrAx;    // Result: syst.error from fDA2 on fAx
   TMap *fErrorCorrX;           // Result: syst.error from fSysIn
   TMap *fErrorCorrAx;          // Result: syst.error from fSysIn
 protected:
   TUnfoldSys(void);            // for derived classes
   virtual void ClearResults(void);     // clear all results
   virtual void PrepareSysError(void); // common calculations for syst.errors
   virtual TMatrixD *PrepareUncorrEmat(TMatrixDSparse const *m1,TMatrixDSparse const *m2); // calculate uncorrelated error matrix
   virtual TMatrixD *PrepareCorrEmat(TMatrixDSparse const *m1,TMatrixDSparse const *m2,TMatrixDSparse const *dsys); // calculate correlated error matrix
 public:
   enum ESysErrMode { // meaning of the argument to AddSysError()
     kSysErrModeMatrix=0, // matrix is an alternative to the default matrix, the errors are the difference to the original matrix
     kSysErrModeShift=1, // matrix gives the absolute shifts
     kSysErrModeRelative=2 // matrix gives the relative shifts
   };
   TUnfoldSys(TH2 const *hist_A, EHistMap histmap, ERegMode regmode = kRegModeSize);      // constructor
   virtual ~ TUnfoldSys(void);    // delete data members
   void AddSysError(TH2 const *sysError,char const *name, EHistMap histmap,
                    ESysErrMode mode); // add a systematic error source
   void GetEmatrixSysUncorr(TH2 *ematrix,Int_t const *binMap=0,Bool_t clearEmat=kTRUE); // get error matrix contribution from statistical errors on the matrix A
   void GetEmatrixSysSource(TH2 *ematrix,char const *source,
                            Int_t const *binMap=0,Bool_t clearEmat=kTRUE); // get error matrix from one systematic source

   void GetEmatrixSysTotal(TH2 *ematrix,Int_t const *binMap=0,Bool_t clearEmat=kTRUE); // get total systematic error
   void GetEmatrixTotal(TH2 *ematrix,Int_t const *binMap=0); // get total error including statistical error

   Double_t GetChi2Sys(void);

   ClassDef(TUnfoldSys, 0) //Unfolding with support for systematic error propagation
};

#endif
