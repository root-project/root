// Author: Stefan Schmitt
// DESY, 13/10/08

//  Version 17.1, bug fixes in GetFoldedOutput, GetOutput
//
//  History:
//    Version 17.0, error matrix with SetInput, store fL not fLSquared
//    Version 16.2, in parallel to bug-fix in TUnfoldSys
//    Version 16.1, in parallel to bug-fix in TUnfold.C
//    Version 16.0, some cleanup, more getter functions, query version number
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
//  TUnfold provides functionality to correct data                      //
//   for migration effects.                                             //
//                                                                      //
//  Citation: S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]        //
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

#include <TH1D.h>
#include <TH2D.h>
#include <TObject.h>
#include <TArrayI.h>
#include <TSpline.h>
#include <TMatrixDSparse.h>
#include <TMatrixD.h>
#include <TObjArray.h>
#include <TString.h>

#define TUnfold_VERSION "V17.1"
#define TUnfold_CLASS_VERSION 17


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
   TMatrixDSparse *fL;          // Input: regularisation conditions
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
   Int_t fIgnoredBins;          // number of input bins which are dropped because they have error=0
   Double_t fEpsMatrix;         // machine accuracy for eingenvalue analysis
   TMatrixD *fX;                // Result: x
   TMatrixDSparse *fVyyInv;     // Result: inverse of covariance matrix on y
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
   // Int_t IsNotSymmetric(TMatrixDSparse const &m) const;
   virtual Double_t DoUnfold(void);     // the unfolding algorithm
   virtual void ClearResults(void);     // clear all results
   void ClearHistogram(TH1 *h,Double_t x=0.) const;
   virtual TString GetOutputBinName(Int_t iBinX) const; // name a bin
   TMatrixDSparse *MultiplyMSparseM(const TMatrixDSparse *a,const TMatrixD *b) const; // multiply sparse and non-sparse matrix
   TMatrixDSparse *MultiplyMSparseMSparse(const TMatrixDSparse *a,const TMatrixDSparse *b) const; // multiply sparse and sparse matrix
   TMatrixDSparse *MultiplyMSparseTranspMSparse(const TMatrixDSparse *a,const TMatrixDSparse *b) const; // multiply transposed sparse and sparse matrix
   TMatrixDSparse *MultiplyMSparseMSparseTranspVector
      (const TMatrixDSparse *m1,const TMatrixDSparse *m2,
       const TMatrixTBase<Double_t> *v) const; // calculate M_ij = sum_k [m1_ik*m2_jk*v[k] ]. the pointer v may be zero (means no scaling).
   TMatrixDSparse *InvertMSparseSymmPos(const TMatrixDSparse *A,Int_t *rank) const; // invert symmetric (semi-)positive sparse matrix
   void AddMSparse(TMatrixDSparse *dest,Double_t f,const TMatrixDSparse *src) const; // replacement for dest += f*src
   TMatrixDSparse *CreateSparseMatrix(Int_t nrow,Int_t ncol,Int_t nele,Int_t *row,Int_t *col,Double_t *data) const; // create a TMatrixDSparse from an array
   inline Int_t GetNx(void) const {
      return fA->GetNcols();
   } // number of non-zero output bins
   inline Int_t GetNy(void) const {
      return fA->GetNrows();
   } // number of input bins
   void ErrorMatrixToHist(TH2 *ematrix,const TMatrixDSparse *emat,const Int_t *binMap,Bool_t doClear) const; // return an error matrix as histogram
   Double_t GetRhoIFromMatrix(TH1 *rhoi,const TMatrixDSparse *eOrig,const Int_t *binMap,TH2 *invEmat) const; // return global correlation coefficients
   inline const TMatrixDSparse *GetDXDY(void) const { return fDXDY; } // access derivative dx/dy
   inline const TMatrixDSparse *GetDXDAM(int i) const { return fDXDAM[i]; } // access matrix parts of the derivative dx/dA
   inline const TMatrixDSparse *GetDXDAZ(int i) const { return fDXDAZ[i]; } // access vector parts of the derivative dx/dA
   inline const TMatrixDSparse *GetDXDtauSquared(void) const { return fDXDtauSquared; } // get derivative dx/dtauSquared
   inline const TMatrixDSparse *GetAx(void) const { return fAx; } // get vector Ax
   inline const TMatrixDSparse *GetEinv(void) const { return fEinv; } // get matrix E^-1
   inline const TMatrixDSparse *GetE(void) const { return fE; } // get matrix E
   inline const TMatrixDSparse *GetVxx(void) const { return fVxx; } // get covariance matrix of x
   inline const TMatrixDSparse *GetVxxInv(void) const { return fVxxInv; } // get inverse of covariance matrix of x
   inline const TMatrixDSparse *GetVyyInv(void) const { return fVyyInv; } // get inverse of covariance matrix of y
   inline const TMatrixD *GetX(void) const { return fX; } // get result vector x
   inline Int_t GetRowFromBin(int ix) const { return fHistToX[ix]; } // convert histogram bin number to matrix row
   inline Int_t GetBinFromRow(int ix) const { return fXToHist[ix]; } // convert matrix row to histogram bin number
   static void DeleteMatrix(TMatrixD **m); // delete and invalidate pointer
   static void DeleteMatrix(TMatrixDSparse **m); // delete and invalidate pointer
   Bool_t AddRegularisationCondition(Int_t i0,Double_t f0,Int_t i1=-1,Double_t f1=0.,Int_t i2=-1,Double_t f2=0.); // add regularisation condition for a triplet of output bins
   Bool_t AddRegularisationCondition(Int_t nEle,const Int_t *indices,const Double_t *rowData); // add a regularisation condition
public:
   enum EHistMap {              // mapping between unfolding matrix and TH2 axes
      kHistMapOutputHoriz = 0,  // map unfolding output to x-axis of TH2 matrix
      kHistMapOutputVert = 1    // map unfolding output to y-axis of TH2 matrix
   };

   TUnfold(const TH2 *hist_A, EHistMap histmap,
           ERegMode regmode = kRegModeSize,
           EConstraint constraint=kEConstraintArea);      // constructor
   virtual ~TUnfold(void);    // delete data members
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
   virtual Int_t SetInput(const TH1 *hist_y, Double_t scaleBias=0.0,Double_t oneOverZeroError=0.0,const TH2 *hist_vyy=0,const TH2 *hist_vyy_inv=0); // define input distribution
   virtual Double_t DoUnfold(Double_t tau); // Unfold with given choice of tau
   virtual Int_t ScanLcurve(Int_t nPoint,Double_t tauMin,
                            Double_t tauMax,TGraph **lCurve,
                            TSpline **logTauX=0,TSpline **logTauY=0); // scan the L curve using successive calls to DoUnfold(Double_t) at various tau
   void GetProbabilityMatrix(TH2 *A,EHistMap histmap) const; // get the matrix A of probabilities
   void GetNormalisationVector(TH1 *s,const Int_t *binMap=0) const; // get the vector of normalisation factors, equivalent to the initial bias vector

   void GetOutput(TH1 *output,const Int_t *binMap=0) const; // get output distribution, arbitrary bin mapping

   void GetBias(TH1 *bias,const Int_t *binMap=0) const; // get bias (includind biasScale)

   void GetFoldedOutput(TH1 *folded,const Int_t *binMap=0) const; // get unfolding result folded back through the matrix

   void GetInput(TH1 *inputData,const Int_t *binMap=0) const; // get input data

   void GetRhoIJ(TH2 *rhoij,const Int_t *binMap=0) const; // get correlation coefficients, arbitrary bin mapping

   void GetEmatrix(TH2 *ematrix,const Int_t *binMap=0) const; // get error matrix, arbitrary bin mapping

   Double_t GetRhoI(TH1 *rhoi,const Int_t *binMap=0,TH2 *invEmat=0) const; // get global correlation coefficients, arbitrary bin mapping

   void GetLsquared(TH2 *lsquared) const; // get regularisation conditions squared

   inline Int_t GetNr(void) const { return fL->GetNrows(); } // number of regularisation conditions
   void GetL(TH2 *l) const; // get matrix of regularisation conditions

   void GetInputInverseEmatrix(TH2 *ematrix);   // get input data inverse of error matrix

   Double_t GetTau(void) const;  // get regularisation parameter
   inline Double_t GetRhoMax(void) const { return fRhoMax; } // get maximum global correlation
   inline Double_t GetRhoAvg(void) const { return fRhoAvg; }  // get average global correlation
   inline Double_t GetChi2A(void) const { return fChi2A; }  // get chi**2 contribution from A
   Double_t GetChi2L(void) const;        // get chi**2 contribution from L
   virtual Double_t GetLcurveX(void) const;        // get value on x axis of L curve
   virtual Double_t GetLcurveY(void) const;        // get value on y axis of L curve
   inline Int_t GetNdf(void) const { return fNdf; }   // get number of degrees of freedom
   Int_t GetNpar(void) const;  // get number of parameters

   inline Double_t GetEpsMatrix(void) const { return  fEpsMatrix; } // get accuracy for eingenvalue analysis
   void SetEpsMatrix(Double_t eps); // set accuracy for eigenvalue analysis

   ClassDef(TUnfold, TUnfold_CLASS_VERSION) //Unfolding with support for L-curve analysis
};

#endif
