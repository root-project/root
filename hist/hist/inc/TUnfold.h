// @(#)root/hist:$Id$
// Author: Stefan Schmitt
// DESY, 13/10/08

//  Version 16, some cleanup, more getter functions, query version number
//
//  History:
//    Version 15, simplified L-curve scan, new tau definition, new eror calc.
//    Version 14, with changes in TUnfoldSys.cxx
//    Version 13, new methods for derived classes
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
//                   T   -1            2          T                     //
//    chi**2 = (y-Ax) Vyy  (y-Ax) + tau  (L(x-x0)) L(x-x0)              //
//                                                                      //
//  Monte Carlo input                                                   //
//    y: vector of measured quantities  (dimension ny)                  //
//    Vyy: covariance matrix for y (dimension ny x ny)                  //
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

#define TUnfold_VERSION "V16.0"

class TUnfold : public TObject {
 private:
   void InitTUnfold(void);     // initialize all data members
 public:
   enum EConstraint {
      kEConstraintNone =0, // use no extra constraint
      kEConstraintArea =1  // enforce preservation of the area
   };
   enum ERegMode {              // regularisation scheme
      kRegModeNone = 0,         // no regularisation
      kRegModeSize = 1,         // regularise the size of the output
      kRegModeDerivative = 2,   // regularize the 1st derivative of the output
      kRegModeCurvature = 3,    // regularize the 2nd derivative of the output
      kRegModeMixed = 4         // mixed regularisation pattern
   };
 protected:
   TMatrixDSparse * fA;         // Input: matrix
   TMatrixDSparse *fLsquared;   // Input: regularisation conditions squared
   TMatrixDSparse *fVyy;        // Input: covariance matrix for y
   TMatrixD *fY;                // Input: y
   TMatrixD *fX0;               // Input: x0
   Double_t fTauSquared;        // Input: regularisation parameter
   Double_t fBiasScale;         // Input: scale factor for the bias
   TArrayI fXToHist;            // Input: matrix indices -> histogram bins
   TArrayI fHistToX;            // Input: histogram bins -> matrix indices
   TArrayD fSumOverY;           // Input: sum of all columns
   EConstraint fConstraint;     // Input: type of constraint to use
   ERegMode fRegMode;           // Input: type of regularisation
 private:
   TMatrixD *fX;                // Result: x
   TMatrixDSparse *fVxx;        // Result: covariance matrix on x
   TMatrixDSparse *fVxxInv;     // Result: inverse of covariance matrix on x
   TMatrixDSparse *fAx;         // Result: Ax
   Double_t fChi2A;             // Result: chi**2 contribution from (y-Ax)V(y-Ax)
   Double_t fLXsquared;         // Result: chi**2 contribution from (x-s*x0)Lsquared(x-s*x0)
   Double_t fRhoMax;            // Result: maximum global correlation
   Double_t fRhoAvg;            // Result: average global correlation
   Int_t fNdf;                  // Result: number of degrees of freedom
   TMatrixDSparse *fDXDAM[2];   // Result: part of derivative dx_k/dA_ij
   TMatrixDSparse *fDXDAZ[2];   // Result: part of derivative dx_k/dA_ij
   TMatrixDSparse *fDXDtauSquared;     // Result: derivative dx/dtau
   TMatrixDSparse *fDXDY;       // Result: derivative dx/dy
   TMatrixDSparse *fEinv;       // Result: matrix E^(-1)
   TMatrixDSparse *fE;          // Result: matrix E
 protected:
   TUnfold(void);              // for derived classes
   virtual Double_t DoUnfold(void);     // the unfolding algorithm
   virtual void ClearResults(void);     // clear all results

   TMatrixDSparse *MultiplyMSparseM(const TMatrixDSparse *a,const TMatrixD *b) const; // multiply sparse and non-sparse matrix
   TMatrixDSparse *MultiplyMSparseMSparse(const TMatrixDSparse *a,const TMatrixDSparse *b) const; // multiply sparse and sparse matrix
   TMatrixDSparse *MultiplyMSparseTranspMSparse(const TMatrixDSparse *a,const TMatrixDSparse *b) const; // multiply transposed sparse and sparse matrix
   TMatrixDSparse *MultiplyMSparseMSparseTranspVector
      (const TMatrixDSparse *m1,const TMatrixDSparse *m2,
       const TMatrixTBase<Double_t> *v) const; // calculate M_ij = sum_k [m1_ik*m2_jk*v[k] ]. the pointer v may be zero (means no scaling).
   TMatrixD *InvertMSparse(const TMatrixDSparse *A) const; // invert sparse matrix
   static Bool_t InvertMConditioned(TMatrixD *A); // invert matrix including preconditioning
   void AddMSparse(TMatrixDSparse *dest,Double_t f,const TMatrixDSparse *src); // replacement for dest += f*src
   TMatrixDSparse *CreateSparseMatrix(Int_t nrow,Int_t ncol,Int_t nele,Int_t *row,Int_t *col,Double_t *data) const; // create a TMatrixDSparse from an array
   inline Int_t GetNx(void) const {
      return fA->GetNcols();
   } // number of non-zero output bins
   inline Int_t GetNy(void) const {
      return fA->GetNrows();
   } // number of input bins
   void ErrorMatrixToHist(TH2 *ematrix,const TMatrixDSparse *emat,const Int_t *binMap,
                          Bool_t doClear) const; // return an error matrix as histogram
   inline const TMatrixDSparse *GetDXDY(void) const { return fDXDY; } // access derivative dx/dy
   inline const TMatrixDSparse *GetDXDAM(int i) const { return fDXDAM[i]; } // access matrix parts of the derivative dx/dA
   inline const TMatrixDSparse *GetDXDAZ(int i) const { return fDXDAZ[i]; } // access vector parts of the derivative dx/dA
   inline const TMatrixDSparse *GetDXDtauSquared(void) const { return fDXDtauSquared; } // get derivative dx/dtauSquared
   inline const TMatrixDSparse *GetAx(void) const { return fAx; } // get vector Ax
   inline const TMatrixDSparse *GetEinv(void) const { return fEinv; } // get matrix E^-1
   inline const TMatrixDSparse *GetE(void) const { return fE; } // get matrix E
   inline const TMatrixDSparse *GetVxx(void) const { return fVxx; } // get covariance matrix of x
   inline const TMatrixDSparse *GetVxxInv(void) const { return fVxxInv; } // get inverse of covariance matrix of x
   inline const TMatrixD *GetX(void) const { return fX; } // get result vector x
   static void DeleteMatrix(TMatrixD **m); // delete and invalidate pointer
   static void DeleteMatrix(TMatrixDSparse **m); // delete and invalidate pointer
public:
   enum EHistMap {              // mapping between unfolding matrix and TH2 axes
      kHistMapOutputHoriz = 0,  // map unfolding output to x-axis of TH2 matrix
      kHistMapOutputVert = 1    // map unfolding output to y-axis of TH2 matrix
   };

   TUnfold(const TH2 *hist_A, EHistMap histmap,
           ERegMode regmode = kRegModeSize,
           EConstraint constraint=kEConstraintArea);      // constructor
   virtual ~ TUnfold(void);    // delete data members
   static const char*GetTUnfoldVersion(void);
   void SetBias(const TH1 *bias);       // set alternative bias
   void SetConstraint(EConstraint constraint); // set type of constraint for the next unfolding
   Int_t RegularizeSize(int bin, Double_t scale = 1.0);   // regularise the size of one output bin
   Int_t RegularizeDerivative(int left_bin, int right_bin, Double_t scale = 1.0); // regularize difference of two output bins (1st derivative)
   Int_t RegularizeCurvature(int left_bin, int center_bin, int right_bin, Double_t scale_left = 1.0, Double_t scale_right = 1.0);  // regularize curvature of three output bins (2nd derivative)
   Int_t RegularizeBins(int start, int step, int nbin, ERegMode regmode);        // regularize a 1-dimensional curve
   Int_t RegularizeBins2D(int start_bin, int step1, int nbin1, int step2, int nbin2, ERegMode regmode);  // regularize a 2-dimensional grid
   Double_t DoUnfold(Double_t tau,
                     const TH1 *hist_y, Double_t scaleBias=0.0);  // do the unfolding
   virtual Int_t SetInput(const TH1 *hist_y, Double_t scaleBias=0.0,Double_t oneOverZeroError=0.0); // define input distribution for ScanLCurve
   virtual Double_t DoUnfold(Double_t tau); // Unfold with given choice of tau
   virtual Int_t ScanLcurve(Int_t nPoint,Double_t tauMin,
                            Double_t tauMax,TGraph **lCurve,
			    TSpline **logTauX=0,TSpline **logTauY=0); // scan the L curve using successive calls to DoUnfold(Double_t)
   TH1D *GetOutput(const char *name,const char *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;    // get unfolding result
   TH1D *GetBias(const char *name,const char *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;      // get bias
   TH1D *GetFoldedOutput(const char *name,const char *title, Double_t y0 = 0.0, Double_t y1 = 0.0) const; // get folded unfolding result
   TH1D *GetInput(const char *name,const char *title, Double_t y0 = 0.0, Double_t y1 = 0.0) const;     // get unfolding input
   TH2D *GetRhoIJ(const char *name,const char *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;     // get correlation coefficients
   TH2D *GetEmatrix(const char*name,const char *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;   // get error matrix
   TH1D *GetRhoI(const char*name,const char *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;      // get global correlation coefficients
   TH2D *GetLsquared(const char*name,const char *title, Double_t x0 = 0.0, Double_t x1 = 0.0) const;  // get regularisation conditions squared

   void GetOutput(TH1 *output,const Int_t *binMap=0) const; // get output distribution, averaged over bins
   void GetEmatrix(TH2 *ematrix,const Int_t *binMap=0) const; // get error matrix, averaged over bins
   Double_t GetRhoI(TH1 *rhoi,TH2 *ematrixinv=0,const Int_t *binMap=0) const; // get global correlation coefficients and inverse of error matrix, averaged over bins
   void GetRhoIJ(TH2 *rhoij,const Int_t *binMap=0) const; // get correlation coefficients, averaged over bins
   Double_t GetTau(void) const;  // regularisation parameter
   inline Double_t GetRhoMax(void) const { return fRhoMax; } // maximum global correlation
   inline Double_t GetRhoAvg(void) const { return fRhoAvg; }  // average global correlation
   inline Double_t GetChi2A(void) const { return fChi2A; }  // chi**2 contribution from A
   Double_t GetChi2L(void) const;        // chi**2 contribution from L
   virtual Double_t GetLcurveX(void) const;        // x axis of L curve
   virtual Double_t GetLcurveY(void) const;        // y axis of L curve
   inline Int_t GetNdf(void) const { return fNdf; }   // number of degrees of freedom
   Int_t GetNpar(void) const;  // number of parameters

   ClassDef(TUnfold, 0) //Unfolding with support for L-curve analysis
};

#endif
