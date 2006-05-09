// @(#)root/tmva $Id: TMVA_MethodCuts.h,v 1.1 2006/05/08 12:46:31 brun Exp $
// Author: Andreas Hoecker, Peter Speckmayer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodCuts                                                       *
 *                                                                                *
 * Description:                                                                   *
 *      Multivariate optimisation of signal efficiency for given background       *
 *      efficiency, using rectangular minimum and maximum requirements on         *
 *      input variables                                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *      Helge Voss       <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany    *
 *      Kai Voss         <Kai.Voss@cern.ch>       - U. of Victoria, Canada        *
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
 * $Id: TMVA_MethodCuts.h,v 1.1 2006/05/08 12:46:31 brun Exp $
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodCuts
#define ROOT_TMVA_MethodCuts

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodCuts                                                      //
//                                                                      //
// Multivariate optimisation of signal efficiency for given background  //
// efficiency, using rectangular minimum and maximum requirements on    //
// input variables                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA_MethodBase.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA_BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_PDF
#include "TMVA_PDF.h"
#endif
#ifndef ROOT_TMVA_GeneticBase
#include "TMVA_GeneticBase.h"
#endif

class TRandom;

class TMVA_MethodCuts : public TMVA_MethodBase {

 public:

  TMVA_MethodCuts( TString jobName,
		   vector<TString>* theVariables,
		   TTree* theTree = 0,
		   TString theOption = "MC:150:10000:",
		   TDirectory* theTargetFile = 0 );

  TMVA_MethodCuts( vector<TString> *theVariables,
		   TString theWeightFile,
		   TDirectory* theTargetDir = NULL );

  virtual ~TMVA_MethodCuts( void );

  // training method
  virtual void Train( void );

  // write weights to file
  virtual void WriteWeightsToFile( void );

  // read weights from file
  virtual void ReadWeightsFromFile( void );

  // calculate the MVA value (for CUTs this is just a dummy)
  virtual Double_t GetMvaValue( TMVA_Event *e );

  // write method specific histos to target file
  virtual void WriteHistosToFile( void );

  // indivudual initialistion of testing of each method test the method
  virtual void TestInitLocal(TTree * testTree);

  // also overwrite:
  virtual Double_t GetSignificance( void )   { return 0; }
  virtual Double_t GetSeparation  ( void )   { return 0; }
  virtual Double_t GetmuTransform ( TTree *) { return 0; }
  virtual Double_t GetEfficiency  ( TString, TTree *);

  // accessors for Minuit
  Double_t        ComputeEstimator( Double_t*, Int_t );

  void SetTestSignalEfficiency( Double_t eff ) { fTestSignalEff = eff; }

  // static pointer to this object
  static TMVA_MethodCuts* ThisCuts( void ) { return fgThisCuts; }

 protected:

 private:

  enum ConstrainType { kConstrainEffS = 0,
		       kConstrainEffB } fConstrainType;

  enum FitMethodType { kUseMonteCarlo = 0,
		       kUseGeneticAlgorithm };

  enum EffMethod     { kUseEventSelection = 0,
		       kUsePDFs };

  enum FitParameters { kNotEnforced = 0,
		       kStartFromMin,
		       kStartFromCenter,
		       kStartFromMax,
		       kForceMin,
		       kForceMax,
		       kForceSmart,
		       kForceVerySmart,
		       kStartingValuesAreGiven,
		       kRandomizeStartingValues };

  enum FitType       { kMigrad = 0, kSimplex };

  // general
  FitMethodType           fFitMethod;
  EffMethod               fEffMethod;
  FitType                 fFitType;
  vector<FitParameters>*  fFitParams;
  Double_t                fTestSignalEff;
  Double_t                fEffSMin;
  Double_t                fEffSMax;

  // for the use of the binary tree method
  TMVA_BinarySearchTree*  fBinaryTreeS;
  TMVA_BinarySearchTree*  fBinaryTreeB;

  // GA
  Int_t              fGa_preCalc;
  Int_t              fGa_SC_steps;
  Int_t              fGa_SC_offsteps;
  Double_t           fGa_SC_factor;
  Int_t              fGa_nsteps;

  // minuit
  Double_t           fEffRef;
  Int_t              fNpar;
  vector<Int_t>*     fRangeSign;
  vector<Double_t>*  fPar0;
  TRandom*           fTrandom;

  // Statistics
  vector<Double_t>*  fMeanS;
  vector<Double_t>*  fMeanB;
  vector<Double_t>*  fRmsS;
  vector<Double_t>*  fRmsB;
  vector<Double_t>*  fXmin;
  vector<Double_t>*  fXmax;

  TH1*               fEffBvsSLocal;

  // PDF section
  vector<TH1*>*      fVarHistS;
  vector<TH1*>*      fVarHistB;
  vector<TH1*>*      fVarHistS_smooth;
  vector<TH1*>*      fVarHistB_smooth;
  vector<TMVA_PDF*>* fVarPdfS;
  vector<TMVA_PDF*>* fVarPdfB;

  // MC method
  Int_t              fNRandCuts;
  Double_t**         fCutMin;
  Double_t**         fCutMax;

  static TMVA_MethodCuts*   fgThisCuts;

  void     InitTMinuitAndFit   ( FitParameters fitParam = kNotEnforced,
			         vector<Double_t>* parStart = NULL );
  void     MatchParsToCuts     ( Double_t*, Double_t*, Double_t* );
  void     MatchCutsToPars     ( Double_t*, Double_t*, Double_t* );
  void     Fit                 ( void );
  void     CheckErr            ( TString, Int_t );
  void     Backup              ( Double_t&, Double_t&, Double_t&, vector<Double_t>* );
  void     CreateVariablePDFs  ( void );
  Bool_t   SanityChecks        ( void );
  void     GetEffsfromSelection(  Double_t* cutMin, Double_t* cutMax,
				  Double_t& effS, Double_t& effB);
  void     GetEffsfromPDFs     (  Double_t* cutMin, Double_t* cutMax,
				  Double_t& effS, Double_t& effB );
  void     InitCuts( void );

  ClassDef(TMVA_MethodCuts,0)  //Multivariate optimisation of signal efficiency
};

#endif
