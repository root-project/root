// @(#)root/tmva $Id: TMVA_MethodLikelihood.h,v 1.7 2006/05/02 23:27:40 helgevoss Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodLikelihood                                                 *
 *                                                                                *
 * Description:                                                                   *
 *      Likelihood analysis ("non-parametric approach")                           *
 *      Also implemented is a "diagonalized likelihood approach",                 *
 *      which improves over the uncorrelated likelihood ansatz by transforming    *
 *      linearly the input variables into a diagonal space, using the square-root *
 *      of the covariance matrix. This approach can be chosen by inserting        *
 *      the letter "D" into the option string.                                    *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_MethodLikelihood.h,v 1.7 2006/05/02 23:27:40 helgevoss Exp $      
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodLikelihood
#define ROOT_TMVA_MethodLikelihood

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodLikelihood                                                //
//                                                                      //
// Likelihood analysis ("non-parametric approach")                      //
// Also implemented is a "diagonalized likelihood approach",            //
// which improves over the uncorrelated likelihood ansatz by            //
// transforming linearly the input variables into a diagonal space,     //
// using the square-root of the covariance matrix                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA_MethodBase.h"
#endif
#ifndef ROOT_TMVA_PDF
#include "TMVA_PDF.h"
#endif
#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif

class TH1D;

// for convenience...
#define TMVA_MethodLikelihood_max_nvar__ 200


class TMVA_MethodLikelihood : public TMVA_MethodBase {

 public:

  TMVA_MethodLikelihood( TString jobName, 
			vector<TString>* theVariables, 
			TTree* theTree = 0,
			TString theOption = "",
			TDirectory* theTargetDir = 0 );
  
  TMVA_MethodLikelihood( vector<TString> *theVariables, 
			 TString theWeightFile,  
			 TDirectory* theTargetDir = NULL );

  virtual ~TMVA_MethodLikelihood( void );
    
  // training method
  virtual void Train( void );

  // write weights to file
  virtual void WriteWeightsToFile( void );
  
  // read weights from file
  virtual void ReadWeightsFromFile( void );

  // calculate the MVA value
  virtual Double_t GetMvaValue( TMVA_Event *e );

  // write method specific histos to target file
  virtual void WriteHistosToFile( void ) ;

  // additional accessor
  Bool_t DecorrVarSpace( void ) { return fDecorrVarSpace; }

  // special normalization class for diagonalized Fisher
  // override bas-class function
  //  void InitNorm( TTree* theTree );

 protected:

 private:

  TFile* fFin;

  TMVA_PDF::SmoothMethod fSmoothMethod;

  // number of events (tot, signal, background)
  Int_t            fNevt;
  Int_t            fNsig;
  Int_t            fNbgd;

  Int_t            fNsmooth;
  Double_t         fEpsilon;
  TMatrixD*        fSqS;
  TMatrixD*        fSqB;

  vector<TH1*>*    fHistSig;
  vector<TH1*>*    fHistBgd; 
  vector<TH1*>*    fHistSig_smooth; 
  vector<TH1*>*    fHistBgd_smooth; 
  
  TList *fSigPDFHist, *fBgdPDFHist;

  vector<TMVA_PDF*>*  fPDFSig;
  vector<TMVA_PDF*>*  fPDFBgd;

  Int_t     fNbins;
  Int_t     fAverageEvtPerBin; // average events per bin. Used to calculate nbins

  // diagonalisation of variable space
  Bool_t    fDecorrVarSpace;
  void GetSQRMats( void );
  void InitLik( void );
   
  ClassDef(TMVA_MethodLikelihood,0) //Likelihood analysis ("non-parametric approach") 
};

#endif // TMVA_MethodLikelihood_H
