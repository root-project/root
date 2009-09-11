// Author: Stefan Schmitt
// DESY, 13/10/08

//  Version 13, new methods for derived classes
//
//  History:
//    Version 12, with support for preconditioned matrix inversion
//    Version 11, regularisation methods have return values
//    Version 10, with bug-fix in TUnfold.cxx
//    Version 9, implements method for optimized inversion of sparse matrix
//    Version 8, replace all TMatrixSparse matrix operations by private code
//    Version 7, fix problem with TMatrixDSparse,TMatrixD multiplication
//    Version 6, completely remove definition of class XY
//    Version 5, move definition of class XY from TUnfold.C to this file
//    Version 4, with bug-fix in TUnfold.C
//    Version 3, with bug-fix in TUnfold.C
//    Version 2, with changed ScanLcurve() arguments
//    Version 1, added ScanLcurve() method
//    Version 0, stable version of basic unfolding algorithm


#ifndef ROOT_TUnfold
#define ROOT_TUnfold

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//  TUnfold solves the inverse problem                                  //
//                                                                      //
//                         T                        T                   //
//    chi**2 = 1/2 * (y-Ax) V (y-Ax) + tau (L(x-x0)) L(x-x0)            //
//                                                                      //
//  Monte Carlo input                                                   //
//    y: vector of measured quantities  (dimension ny)                  //
//    V: inverse of covariance matrix for y (dimension ny x ny)         //
//    A: migration matrix               (dimension ny x nx)             //
//    x: unknown underlying distribution (dimension nx)                 //
//  Regularisation                                                      //
//    tau: parameter, defining the regularisation strength              //
//    L: matrix of regularisation conditions (dimension nl x nx)        //
//    x0: underlying distribution bias                                  //
//                                                                      //
//  where chi**2 is minimized as a function of x                        //
//                                                                      //
//  The algorithm is based on "standard" matrix inversion, with the     //
//  known limitations in numerical accuracy and computing cost for      //
//  matrices with large dimensions.                                     //
//                                                                      //
//  Thus the algorithm should not used for large dimensions of x and y  //
//    dim(x) should not exceed O(100)                                   //    
//    dim(y) should not exceed O(500)                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include <TH1D.h>
#include <TH2D.h>
#include <TObject.h>
#include <TArrayI.h>
#include <TSpline.h>
#include <TMatrixDSparse.h>
#include <TMatrixD.h>
#include <TObjArray.h>

class TUnfold:public TObject {
 private:
   void InitTUnfold(void);     // initialize all data members
 protected:
   TMatrixDSparse * fA;        // Input: matrix
   TMatrixDSparse *fLsquared;   // Input: regularisation conditions squared
   TMatrixDSparse *fV;          // Input: covariance matrix for y
   TMatrixD *fY;                // Input: y
   TMatrixD *fX0;               // Input: x0
   Double_t fTau;               // Input: regularisation parameter
   Double_t fBiasScale;         // Input: scale factor for the bias
   TArrayI fXToHist;            // Input: matrix indices -> histogram bins
   TArrayI fHistToX;            // Input: histogram bins -> matrix indices
   TArrayD fSumOverY;           // Input: sum of all columns
   TMatrixDSparse *fEinv;       // Result: inverse error matrix
   TMatrixDSparse *fAtV;        // Result: fA# times fV
   TMatrixD *fE;                // Result: error matrix
   TMatrixD *fX;                // Result: x
   TMatrixDSparse *fAx;         // Result: Ax
   Double_t fChi2A;             // Result: chi**2 contribution from (y-Ax)V(y-Ax)
   Double_t fChi2L;             // Result: chi**2 contribution from tau(x-s*x0)Lsquared(x-s*x0)
   Double_t fRhoMax;            // Result: maximum global correlation
   Double_t fRhoAvg;            // Result: average global correlation
   Int_t fNdf;                  // Result: number of degrees of freedom
 protected:
   TUnfold(void);              // for derived classes
   virtual Double_t DoUnfold(void);     // the unfolding algorithm
   virtual void CalculateChi2Rho(void); // supplementory calculations
   virtual void ClearResults(void);     // clear all results

   static TMatrixDSparse *MultiplyMSparseM(TMatrixDSparse const &a,TMatrixD const &b); // multiply sparse and non-sparse matrix
   static TMatrixDSparse *MultiplyMSparseMSparse(TMatrixDSparse const &a,TMatrixDSparse const &b); // multiply sparse and sparse matrix
   static TMatrixDSparse *MultiplyMSparseTranspMSparse(TMatrixDSparse const &a,TMatrixDSparse const &b); // multiply transposed sparse and sparse matrix
   static Double_t MultiplyVecMSparseVec(TMatrixDSparse const &a,TMatrixD const &v); // scalar product of v and Av
   static TMatrixD *InvertMSparse(TMatrixDSparse const &A); // invert sparse matrix
   static Bool_t InvertMConditioned(TMatrixD &A); // invert matrix including preconditioning
   static void AddMSparse(TMatrixDSparse &dest,Double_t const &f,TMatrixDSparse const &src); // replacement for dest += f*src
   inline Int_t GetNx(void) const {
      return fA->GetNcols();
   } // number of non-zero output bins
   inline Int_t GetNy(void) const {
      return fA->GetNrows();
   } // number of input bins
   void ErrorMatrixToHist(TH2 *ematrix,TMatrixD const *emat,Int_t const *binMap,
                          Bool_t doClear) const; // return an error matrix as histogram
 public:
   enum EHistMap {              // mapping between unfolding matrix and TH2 axes
      kHistMapOutputHoriz = 0,  // map unfolding output to x-axis of TH2 matrix
      kHistMapOutputVert = 1    // map unfolding output to y-axis of TH2 matrix
   };

   enum ERegMode {              // regularisation scheme
      kRegModeNone = 0,         // no regularisation
      kRegModeSize = 1,         // regularise the size of the output
      kRegModeDerivative = 2,   // regularize the 1st derivative of the output
      kRegModeCurvature = 3     // regularize the 2nd derivative of the output
   };
   TUnfold(TH2 const *hist_A, EHistMap histmap, ERegMode regmode = kRegModeSize);      // constructor
   virtual ~ TUnfold(void);    // delete data members
   static void DeleteMatrix(TMatrixD **m); // delete and invalidate pointer
   static void DeleteMatrix(TMatrixDSparse **m); // delete and invalidate pointer
   void SetBias(TH1 const *bias);       // set alternative bias
   Int_t RegularizeSize(int bin, Double_t const &scale = 1.0);   // regularise the size of one output bin
   Int_t RegularizeDerivative(int left_bin, int right_bin, Double_t const &scale = 1.0); // regularize difference of two output bins (1st derivative)
   Int_t RegularizeCurvature(int left_bin, int center_bin, int right_bin, Double_t const &scale_left = 1.0, Double_t const &scale_right = 1.0);  // regularize curvature of three output bins (2nd derivative)
   Int_t RegularizeBins(int start, int step, int nbin, ERegMode regmode);        // regularize a 1-dimensional curve
   Int_t RegularizeBins2D(int start_bin, int step1, int nbin1, int step2, int nbin2, ERegMode regmode);  // regularize a 2-dimensional grid
   Double_t DoUnfold(Double_t const &tau,
                     TH1 const *hist_y, Double_t const &scaleBias=0.0);  // do the unfolding
   Int_t SetInput(TH1 const *hist_y, Double_t const &scaleBias=0.0,Double_t oneOverZeroError=0.0); // define input distribution for ScanLCurve
   virtual Double_t DoUnfold(Double_t const &tau); // Unfold with given choice of tau
   virtual Int_t ScanLcurve(Int_t nPoint,Double_t const &tauMin,
                            Double_t const &tauMax,TGraph **lCurve,
			    TSpline **logTauX=0,TSpline **logTauY=0); // scan the L curve using successive calls to DoUnfold(Double_t)
   TH1D *GetOutput(char const *name, char const *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;    // get unfolding result
   TH1D *GetBias(char const *name, char const *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;      // get bias
   TH1D *GetFoldedOutput(char const *name, char const *title, Double_t y0 = 0.0, Double_t y1 = 0.0) const; // get folded unfolding result
   TH1D *GetInput(char const *name, char const *title, Double_t y0 = 0.0, Double_t y1 = 0.0) const;     // get unfolding input
   TH2D *GetRhoIJ(char const *name, char const *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;     // get correlation coefficients
   TH2D *GetEmatrix(char const *name, char const *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;   // get error matrix
   TH1D *GetRhoI(char const *name, char const *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;      // get global correlation coefficients
   TH2D *GetLsquared(char const *name, char const *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;  // get regularisation conditions squared

   void GetOutput(TH1 *output,Int_t const *binMap=0) const; // get output distribution, averaged over bins
   void GetEmatrix(TH2 *ematrix,Int_t const *binMap=0) const; // get error matrix, averaged over bins
   Double_t GetRhoI(TH1 *rhoi,TH2 *ematrixinv=0,Int_t const *binMap=0) const; // get global correlation coefficients and inverse of error matrix, averaged over bins
   void GetRhoIJ(TH2 *rhoij,Int_t const *binMap=0) const; // get correlation coefficients, averaged over bins
   Double_t const &GetTau(void) const;  // regularisation parameter
   Double_t const &GetRhoMax(void) const;       // maximum global correlation
   Double_t const &GetRhoAvg(void) const;       // average global correlation
   Double_t const &GetChi2A(void) const;        // chi**2 contribution from A
   Double_t const &GetChi2L(void) const;        // chi**2 contribution from L
   Double_t GetLcurveX(void) const;        // x axis of L curve
   Double_t GetLcurveY(void) const;        // y axis of L curve
   Int_t GetNdf(void) const;   // number of degrees of freedom
   Int_t GetNpar(void) const;  // number of parameters

   ClassDef(TUnfold, 0) //Unfolding with support for L-curve analysis
};

#endif
