// -*- mode: c++ -*-
//
// $Id: TPrincipal.h,v 1.1 2000/08/15 07:48:50 brun Exp $
// $Author: brun $
// $Date: 2000/08/15 07:48:50 $
//
#ifndef ROOT_TPrincipal
#define ROOT_TPrincipal

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TVectorD
#include "TVectorD.h"
#endif
#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif

class TPrincipal : public TObject {

protected:
  Int_t      fNumberOfDataPoints;   // Number of data points
  Int_t      fNumberOfVariables;    // Number of variables

  TVectorD   fMeanValues;           // Mean value over all data points
  TVectorD   fSigmas;               // vector of sigmas
  TMatrixD   fCovarianceMatrix;     // Covariance matrix
  
  TMatrixD   fEigenVectors;         // Eigenvector matrix of trans
  TVectorD   fEigenValues;          // Eigenvalue vector of trans

  TVectorD   fOffDiagonal;          // elements of the tridiagonal

  TVectorD   fUserData;             // Vector of original data points 
 
  Double_t   fTrace;                // Trace of covarience matrix 

  TList     *fHistograms;           // List of histograms

  Bool_t     fIsNormalised;         // Normalize matrix?

  void       MakeNormalised();
  void       MakeTridiagonal();
  void       MakeEigenVectors();
  void       MakeOrdered();
  void       MakeRealCode(const char *filename, const char *prefix, Option_t *option="");

public:
  TPrincipal();
  virtual ~TPrincipal();
  TPrincipal(Int_t nVariables, Option_t *opt="N");
  
  virtual void    AddRow(Double_t *x);
  virtual void    Browse(TBrowser *b);
  virtual void    Clear(Option_t *option="");
  const TMatrixD *GetCovarianceMatrix() const {return &fCovarianceMatrix;}
  const TVectorD *GetEigenValues() const      {return &fEigenValues;}
  const TMatrixD *GetEigenVectors() const     {return &fEigenVectors;}
  TList          *GetHistograms() {return fHistograms;}
  const TVectorD *GetMeanValues() const       {return &fMeanValues;}
  virtual const char *GetName() const { return "PCA"; }
  const Double_t *GetRow(Int_t row);
  const TVectorD *GetSigmas() const           {return &fSigmas;}
  const TVectorD *GetUserData() const         {return &fUserData;}
  Bool_t          IsFolder() { return kTRUE;}
  virtual void    MakeCode(const char *filename ="pca", Option_t *option="");  // *MENU*
  virtual void    MakeHistograms(const char *name = "pca", Option_t *option="epsdx"); // *MENU*
  virtual void    MakeMethods(const char *classname = "PCA", Option_t *option=""); // *MENU*
  virtual void    MakePrincipals();            // *MENU*
  virtual void    P2X(Double_t *p, Double_t *x, Int_t nTest);
  virtual void    Print(Option_t *opt="MSE");         // *MENU*
  virtual void    SumOfSquareResiduals(Double_t *x, Double_t *s); 
  void            Test(Option_t *option="");       // *MENU*
  virtual void    X2P(Double_t *x, Double_t *p);

  ClassDef(TPrincipal,1) // Principal Components Analysis
}
;

#endif 
