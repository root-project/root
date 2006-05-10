// @(#)root/tmva $Id: TMVA_Factory.h,v 1.2 2006/05/08 20:56:16 brun Exp $   
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Factory                                                          *
 *                                                                                *
 * Description:                                                                   *
 *      This is the main MVA steering class: it creates (books) all MVA methods,  *
 *      and guides them through the training, testing and evaluation phases.      *
 *      It also manages multiple MVA handling in case of distinct phase space     *
 *      requirements (cuts).                                                      *
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
 **********************************************************************************/

#ifndef ROOT_TMVA_Factory
#define ROOT_TMVA_Factory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_Factory                                                         //
//                                                                      //
// This is the main MVA steering class: it creates all MVA methods,     //
// and guides them through the training, testing and evaluation         //
// phases. It also manages multiple MVA handling in case of distinct    //
// phase space requirements (cuts).                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <map>
#include "TCut.h"
#include "TDirectory.h"
#ifndef ROOT_TMVA_Types
#include "TMVA_Types.h"
#endif

class TFile;
class TTree;
class TMVA_MethodBase;

using  std::vector;

class TMVA_Factory : public TObject {

public:

  // no default  constructor
  TMVA_Factory( TString theJobName, TFile* theTargetFile, TString theOption = "" );

  // constructor used if no training, but only testing, application is desired
  TMVA_Factory( TFile*theTargetFile );

  // default destructor
  virtual ~TMVA_Factory();

  // the (colourfull) greeting
  void Greeting(TString="");


  /* 
   * Create signal and background trees from individual ascii files
   * note that the format of the file must be the following:
   *
   *    myVar1/D:myVar2[2]/F:myVar3/I:myString/S
   *    3.1415  6.24   56.14   18   UmmmYeah
   *    4.31534 7.4555 9.1466  8    OhWell
   *    ...
   * The first line says to make a tree with 4 branches.
   * The 1st branch will be called "myVar1"   and will contain a Double_t.
   * The 2nd branch will be called "myVar2"   and will contain a TArrayF.
   * The 3rd branch will be called "myVar3"   and will contain an Int_t.
   * The 4th branch will be called "myString" and will contain a TObjString. 
   */
  Bool_t SetInputTrees( TString signalFileName, 
                        TString backgroundFileName );
  Bool_t SetInputTrees( TTree* inputTree, TCut SigCut, TCut BgCut = "");

  // Set input trees at once
  Bool_t SetInputTrees(TTree* signal, TTree* background);

  // set signal tree
  void SetSignalTree(TTree* signal);

  // set background tree
  void SetBackgroundTree(TTree* background);

  // set test tree
  void SetTestTree(TTree* testTree);

  // set input variable
  void SetInputVariables( vector<TString>* theVariables ) { fInputVariables = theVariables; }

  // prepare input tree for training
  void PrepareTrainingAndTestTree(TCut cut = "",Int_t Ntrain = 0, Int_t Ntest = 0 , TString TreeName="");

  // book multiple MVAs 
  void BookMultipleMVAs(TString theVariable, Int_t nbins, Double_t *array);

  // process Multiple MVAs
  void ProcessMultipleMVA();

  // set number of training events
  // void SetN_training(Int_t Ntrain);
  Bool_t BookMethod( TString theMethodName, TString theOption = "", 
                     TString theNameAppendix = "" );
  Bool_t BookMethod( TMVA_Types::MVA theMethod, TString theOption = "", 
                     TString theNameAppendix = "" );

  // booking the method with a given weight file --> testing or application only
  Bool_t BookMethod( TMVA_MethodBase *theMethod, 
                     TString theNameAppendix = "");

  // training for all booked methods
  void TrainAllMethods     ( void );

  // testing
  void TestAllMethods      ( void );

  // performance evaluation
  void EvaluateAllMethods  ( void );
  void EvaluateAllVariables( TString options = "");
  
  // delete all methods and reset the method vector
  void DeleteAllMethods ( void );

  // accessors
  TTree* GetTrainingTree( void ) const { return fTrainingTree; }
  TTree* GetTestTree    ( void ) const { return fTestTree;     }
  TCut   GetCut         ( void ) { return fCut;          }

  TMVA_MethodBase* GetMVA( TString method );

  Bool_t Verbose        ( void ) const { return fVerbose; }

protected:

  void PlotVariables       ( TTree* theTree);
  void GetCorrelationMatrix( TTree* theTree );

private:
  void SetLocalDir();

  /**
   * Fields
   */  
  TFile*           fSignalFile;
  TFile*           fBackgFile;
  TTree*           fTrainingTree;
  TTree*           fTestTree;
  TTree*           fMultiCutTestTree;
  TTree*           fSignalTree;
  TTree*           fBackgTree;
  TFile*           fTargetFile;


  TCut             fCut;
  TString          fOptions;
  Bool_t           fVerbose;

  vector<TString>*              fInputVariables;
  std::vector<TMVA_MethodBase*> fMethods;
  TString                       fJobName;

  Bool_t fMultipleMVAs;
  Bool_t fMultipleStoredOptions;
  Bool_t fMultiTrain;
  Bool_t fMultiTest;
  Bool_t fMultiEvalVar;
  Bool_t fMultiEval; 
  TCut   fMultiCut;
  Int_t  fMultiNtrain;
  Int_t  fMultiNtest;
    

  // This contains:
  // TString: simple bin name (for directories without special characters)
  // TString: cut (human readable)
  // TCut   : ROOT cut  
  std::map<TString, std::pair<TString,TCut> > fMultipleMVAnames ;

  std::map<TString, std::pair<TString,TString> > fMultipleMVAMethodOptions ;

  TString     fMultiVar1;
  TDirectory* fLocalTDir;
 
  ClassDef(TMVA_Factory,0)  //main MVA steering class: it creates all MVA methods
};

#endif

