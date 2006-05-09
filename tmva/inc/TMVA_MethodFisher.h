// @(#)root/tmva $Id: TMVA_MethodFisher.h,v 1.1 2006/05/08 12:46:31 brun Exp $
// Author: Andreas Hoecker, Xavier Prudent, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodFisher                                                     *
 *                                                                                *
 * Description:                                                                   *
 *      Analysis of Fisher discriminant (Fisher or Mahalanobis approach)          *
 *                                                                                *
 * Original author of this Fisher-Discriminant implementation:                    *
 *      Andre Gaidot, CEA-France;                                                 *
 *      (Translation from FORTRAN)                                                *
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
 * $Id: TMVA_MethodFisher.h,v 1.1 2006/05/08 12:46:31 brun Exp $          
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodFisher
#define ROOT_TMVA_MethodFisher

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodFisher                                                    //
//                                                                      //
// Analysis of Fisher discriminant (Fisher or Mahalanobis approach)     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA_MethodBase.h"
#endif
#ifndef ROOT_TMatrix
#include "TMatrix.h"
#endif

class TH1D;

class TMVA_MethodFisher : public TMVA_MethodBase {

 public:

  TMVA_MethodFisher( TString jobName, 
		    vector<TString>* theVariables, 
		    TTree* theTree = 0, 
		    TString theOption = "Fisher",
		    TDirectory* theTargetDir = 0 );

  TMVA_MethodFisher( vector<TString> *theVariables, 
		     TString theWeightFile,  
		     TDirectory* theTargetDir = NULL );

  virtual ~TMVA_MethodFisher( void );
    
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

  enum FisherMethod { kFisher, kMahalanobis };
  virtual FisherMethod GetMethod( void ) { return fFisherMethod; }

 protected:

 private:

  TH1D *fHistPDF[25]; // histograms containing the pdfs
  Int_t fNbins;
  Int_t fAverageEvtPerBin; // average events per bin. Used to calculate nbins
        
  // number of events (tot, signal, background)
  Int_t fNevt;
  Int_t fNsig;
  Int_t fNbgd;

  // arrays of input evt vs. variable 
  TMatrix *fSig;
  TMatrix *fBgd;

  // Initialization and allocation
  void Init( void );

  // get mean value of variables
  void GetMean( void );

  // get matrix of covariance within class
  void GetCov_WithinClass( void );

  // get matrix of covariance between class
  void GetCov_BetweenClass( void );

  // and the full covariance matrix
  void GetCov_Full( void );

  // get discriminating power
  void GetDiscrimPower( void );

  // nice output
  void PrintCoefficients( void );

  // get Fisher coefficients
  void GetFisherCoeff( void );

  // matrix of variables means: S, B, S+B vs. variables
  TMatrix *fMeanMatx;

  // covariance matrices
  TMatrix *fBetw;
  TMatrix *fWith;
  TMatrix *fCov;     

  //discriminating power
  vector<Double_t> *fDiscrimPow;

  // Fisher coefficients
  vector<Double_t> *fFisherCoeff;
  Double_t fF0;

  FisherMethod fFisherMethod;

  void InitFisher( void );

  ClassDef(TMVA_MethodFisher,0) //Analysis of Fisher discriminant (Fisher or Mahalanobis approach) 
};

#endif // TMVA_MethodFisher_H
