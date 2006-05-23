// @(#)root/tmva $Id: Factory.h,v 1.6 2006/05/23 09:53:10 stelzer Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Factory                                                               *
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
// Factory                                                              //
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
#include "TMVA/Types.h"
#endif

class TFile;
class TTree;
class TNtuple;

using std::vector;

namespace TMVA {

   class MethodBase;

   class Factory : public TObject {

   public:

      // no default  constructor
      Factory( TString theJobName, TFile* theTargetFile, TString theOption = "" );

      // constructor used if no training, but only testing, application is desired
      Factory( TFile*theTargetFile );

      // default destructor
      virtual ~Factory();

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
      Bool_t SetInputTrees(TString signalFileName, 
                           TString backgroundFileName );
      Bool_t SetInputTrees(TTree* inputTree, TCut SigCut, TCut BgCut = "");

      // Set input trees at once
      Bool_t SetInputTrees(TTree* signal, TTree* background);

      // set signal tree
      void SetSignalTree(TTree* signal);

      // set background tree
      void SetBackgroundTree(TTree* background);

      // set test tree
      void SetTestTree(TTree* testTree);

      // set Signal and Background weights
      void SetSignalAndBackgroundEvents(Double_t signal, Double_t background);

      // set input variable
      void SetInputVariables( vector<TString>* theVariables ) { fInputVariables = theVariables; }

      // prepare input tree for training
      void PrepareTrainingAndTestTree(TCut cut = "",Int_t Ntrain = 0, Int_t Ntest = 0 , 
                                      TString TreeName="");

      // book multiple MVAs 
      void BookMultipleMVAs(TString theVariable, Int_t nbins, Double_t *array);

      // process Multiple MVAs
      void ProcessMultipleMVA();

      // set number of training events
      // void SetN_training(Int_t Ntrain);
      Bool_t BookMethod( TString theMethodName, TString theOption = "", 
                         TString theNameAppendix = "" );
      Bool_t BookMethod( Types::MVA theMethod, TString theOption = "", 
                         TString theNameAppendix = "" );

      // booking the method with a given weight file --> testing or application only
      Bool_t BookMethod( MethodBase *theMethod, 
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
      TCut   GetCut         ( void ) { return fCut; }

      MethodBase* GetMVA( TString method );

      Bool_t Verbose        ( void ) const { return fVerbose; }
      void SetVerbose       ( Bool_t v=kTRUE ) { fVerbose = v; }
    
   protected:
    
      void PlotVariables       ( TTree* theTree);
      void GetCorrelationMatrix( TTree* theTree );
    
   private:

      // cd to local directory
      void SetLocalDir();  

      TFile*           fSignalFile;         // if two input files: signal file
      TFile*           fBackgFile;          // if two input files: background file
      // TTree used for MVA training: 
      //   contains signal and background events separated by a "type" identifier;
      //   the training tree is built by the Factory from the input tree(s) or ascii files
      TTree*           fTrainingTree;       
      // TTree used for MVA testing:
      //   contains signal and background events separated by a "type" identifier;
      //   the test tree is built by the Factory; it is written to the ROOT output target
      TTree*           fTestTree;           

      // in multi-cut mode (multiple MVAs for different phase space regions), this test 
      // tree combines all sub-cut-trees
      TTree*           fMultiCutTestTree;
      TTree*           fSignalTree;         // cloned from input tree (signal)
      TTree*           fBackgTree;          // cloned from input tree (background)

      // the following events will represent the basis of further event weights;
      // event weights are not yet fully supported by all MVAs... this is on the todo list !
      Double_t         fSignalEvents;       // number of signal events
      Double_t         fBackgroundEvents;   // number of background events

      TFile*           fTargetFile;         // ROOT output file
      TNtuple*         fSigBgdVariables;    // Signal and background ntuple (see example significances.C)

      TCut             fCut;                // preselection cut applied to all input events
      TString          fOptions;            // option string given by construction (presently only "V")
      Bool_t           fVerbose;            // verbose mode

      vector<TString>*         fInputVariables; // names of input variables used in the MVAs
      std::vector<MethodBase*> fMethods;        // all MVA methods
      TString                  fJobName;        // jobname, used as extension in weight file names

      // driving flags for multi-cut (multiple MVAs) environment; required for internal mapping
      Bool_t fMultipleMVAs;                 // multi-cut mode ?
      Bool_t fMultipleStoredOptions;        // multi-cut driving flag
      Bool_t fMultiTrain;                   // multi-cut driving flag
      Bool_t fMultiTest;                    // multi-cut driving flag
      Bool_t fMultiEvalVar;                 // multi-cut driving flag
      Bool_t fMultiEval;                    // multi-cut driving flag

      // in case of multiple MVAs, the cut given in the constructor determines the 
      // distinguished phase-space regions 
      TCut   fMultiCut;                     // phase-space cut
      Int_t  fMultiNtrain;                  // number of training events
      Int_t  fMultiNtest;                   // number of testing events
    
      // maps for multi-cut treatment containing:
      //   TString: simple bin name (for directories without special characters)
      //   TString: cut (human readable)
      //   TCut   : ROOT cut  
      std::map<TString, std::pair<TString,TCut> >    fMultipleMVAnames;         // map of MVA names
      std::map<TString, std::pair<TString,TString> > fMultipleMVAMethodOptions; // map of option strings

      // local directory for each MVA in output target, used to store 
      // specific monitoring histograms (e.g., reference distributions of likelihood method)
      TDirectory* fLocalTDir;

      ClassDef(Factory,0)  // TMVA steering class: it creates all MVA methods, and performs their training and testing
         };
} // namespace TMVA

#endif

