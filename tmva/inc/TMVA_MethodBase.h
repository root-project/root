// @(#)root/tmva $Id: TMVA_MethodBase.h,v 1.14 2006/05/02 23:27:40 helgevoss Exp $   
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodBase                                                       *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual base class for all MVA method                                     *
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
 * $Id: TMVA_MethodBase.h,v 1.14 2006/05/02 23:27:40 helgevoss Exp $   
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodBase
#define ROOT_TMVA_MethodBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodBase                                                      //
//                                                                      //
// Virtual base class for all TMVA method                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Riostream.h"

#ifndef ROOT_TMVA_PDF
#include "TMVA_PDF.h"
#endif
#ifndef ROOT_TMVA_TSpline1
#include "TMVA_TSpline1.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA_Event.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA_Types.h"
#endif

class TTree;
class TDirectory;

class TMVA_MethodBase : public TObject {

 public:

  // default constructur
  TMVA_MethodBase( TString jobName,
		   vector<TString>* theVariables, 
		   TTree* theTree = 0, 
		   TString theOption = "", 
		   TDirectory* theBaseDir = 0 );

  // constructor used for Testing + Application of the MVA, only (no training), 
  // using given weight file
  TMVA_MethodBase( vector<TString> *theVariables, 
		   TString weightFile, 
		   TDirectory* theBaseDir = NULL  );

  // default destructur
  virtual ~TMVA_MethodBase( void );

  // training method
  virtual void Train( void ) = 0;

  // write weights to file
  virtual void WriteWeightsToFile( void ) = 0;
  
  // read weights from file
  virtual void ReadWeightsFromFile( void ) = 0;

  // prepare tree branch with the method's discriminating variable
  virtual void PrepareEvaluationTree( TTree* theTestTree );

  // calculate the MVA value
  virtual Double_t GetMvaValue( TMVA_Event *e ) = 0;

  // evaluate method (resulting discriminating variable) or input varible
  virtual void TestInit(TTree* theTestTree);

  // indivudual initialistion for testing of each method
  // overload this one for individual initialisation of the testing, 
  // it is then called automatically within the global "TestInit" 
  virtual void TestInitLocal(TTree *  /*testTree*/) {
    return ;
  }

  // test the method
  virtual void Test( TTree * theTestTree );

  // write method specific histos to target file
  virtual void WriteHistosToFile( void ) = 0;

  // accessors
  TString GetMethodName( void ) const         { return fMethodName; }
  TMVA_Types::MVA GetMethod    ( void ) const { return fMethod;     }
  TString GetOptions   ( void ) const         { return fOptions;    }
  void    SetMethodName( TString methodName ) { fMethodName = methodName; }
  void    AppendToMethodName( TString methodNameSuffix );

  TString GetJobName   ( void ) const         { return fJobName; }
  void    SetJobName   ( TString jobName )    { fJobName = jobName; }

  TString GetWeightFileExtension( void ) const            { return fFileExtension; }
  void    SetWeightFileExtension( TString fileExtension ) { fFileExtension = fileExtension; } 

  TString GetWeightFileDir( void ) const      { return fFileDir; }
  void    SetWeightFileDir( TString fileDir );

  vector<TString>*  GetInputVars( void ) const { return fInputVars; }
  void              SetInputVars( vector<TString>* theInputVars ) { fInputVars = theInputVars; }

  void     SetWeightFileName( void );
  void     SetWeightFileName( TString );
  TString  GetWeightFileName( void );
  TTree*   GetTrainingTree  ( void ) const { return fTrainingTree; }

  Int_t    GetNvar          ( void ) const { return fNvar; }

  // variables (and private menber functions) for the Evaluation:
  // get the effiency. It fills a histogram for efficiency/vs/bkg
  // and returns the one value fo the efficiency demanded for 
  // in the TString argument. (Watch the string format)
  virtual Double_t  GetEfficiency   ( TString , TTree *);
  virtual Double_t  GetSignificance ( void );
  virtual Double_t  GetSeparation   ( void );
  virtual Double_t  GetmuTransform  ( TTree * );


  // normalisation init
  virtual void InitNorm( TTree* theTree );

  // normalisation accessors
  Double_t GetXminNorm( Int_t ivar  ) const { return (*fXminNorm)[ivar]; }
  Double_t GetXmaxNorm( Int_t ivar  ) const { return (*fXmaxNorm)[ivar]; }
  Double_t GetXminNorm( TString var ) const;
  Double_t GetXmaxNorm( TString var ) const;
  void     SetXminNorm( Int_t ivar,  Double_t x ) { (*fXminNorm)[ivar] = x; }
  void     SetXmaxNorm( Int_t ivar,  Double_t x ) { (*fXmaxNorm)[ivar] = x; }
  void     SetXminNorm( TString var, Double_t x );
  void     SetXmaxNorm( TString var, Double_t x );
  void     UpdateNorm ( Int_t ivar,  Double_t x );

  // main normalization method is in TMVA_Tools
  Double_t Norm       ( Int_t ivar,  Double_t x ) const;
  Double_t Norm       ( TString var, Double_t x ) const;
  
  // member functions for the "evaluation" 
  // accessors
  Bool_t   isOK     ( void  )  const { return fIsOK; }

  void WriteHistosToFile( TDirectory* targetDir );

  enum CutOrientation { Negative = -1, Positive = +1 };
  CutOrientation GetCutOrientation() { return fCutOrientation; }

  enum Type { Signal = 1, Background = 0 };

  Bool_t Verbose( void ) { return fVerbose; }

 public:

  // static pointer to this object
  static TMVA_MethodBase* ThisBase( void ) { return fThisBase; }  

 protected:

  void ResetThisBase( void ) { fThisBase = this; }

 protected:

  TString          fJobName;
  TString          fMethodName;
  TMVA_Types::MVA  fMethod;
  TTree*           fTrainingTree;
  TString          fTestvar;  
  TString          fTestvarPrefix;  
  vector<TString>* fInputVars;
  TString          fOptions;
  TDirectory*      fBaseDir;
  TDirectory*      fLocalTDir;

  Bool_t CheckSanity( TTree* theTree = 0 );

  Int_t       fNvar;

 private:

  TString     fFileExtension;
  TString     fFileDir;


  TH1*        bookNormTH1( TString, Int_t, Double_t, Double_t, TString );

  TString     fWeightFile;

 protected:

  Bool_t    fIsOK;
  TH1*      fHistS_plotbin;
  TH1*      fHistB_plotbin;
  TH1*      fHistS_highbin;
  TH1*      fHistB_highbin;
  TH1*      fEffS;
  TH1*      fEffB;
  TH1*      fEffBvsS;
  TH1*      fRejBvsS;
  TH1*      fHistBhatS;
  TH1*      fHistBhatB;
  TH1*      fHistMuS;
  TH1*      fHistMuB;

  // mu-transform
  Double_t  fX;
  Double_t  fMode;

  TGraph*   fGraphS;
  TGraph*   fGraphB;
  TGraph*   fGrapheffBvsS;
  TMVA_PDF* fSplS;
  TMVA_PDF* fSplB;
  TSpline*  fSpleffBvsS;
  void      muTransform( void );

 private:

  Double_t  fMeanS;
  Double_t  fMeanB;
  Double_t  fRmsS; 
  Double_t  fRmsB;
  Double_t  fXmin;
  Double_t  fXmax;

  // verbose flag (debug messages) 
  Bool_t    fVerbose;

 protected:

  Double_t  fEffSatB;
  Int_t     fNbins;
  Int_t     fNbinsH;

 private:

  // output types
  Double_t  fSeparation;
  Double_t  fSignificance;

  // orientation of cut: depends on signal and background mean values
 protected:

  CutOrientation  fCutOrientation; // +1 if Sig>Bkg, -1 otherwise

  // for root finder
  TMVA_TSpline1*  fSplRefS;
  TMVA_TSpline1*  fSplRefB;

 public:

  // for root finder 
  static Double_t IGetEffForRoot( Double_t );  
  Double_t GetEffForRoot( Double_t );  

 private:

  // normalization
  vector<Double_t>* fXminNorm;
  vector<Double_t>* fXmaxNorm;

  // this carrier
  static TMVA_MethodBase* fThisBase;

  // Init used in the various constructors
  void Init( void );

  ClassDef(TMVA_MethodBase,0)  //Virtual base class for all TMVA method
};

#endif

