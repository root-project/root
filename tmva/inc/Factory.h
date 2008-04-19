// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Factory                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      This is the main MVA steering class: it creates (books) all MVA methods,  *
 *      and guides them through the training, testing and evaluation phases.      *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_Factory
#define ROOT_TMVA_Factory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Factory                                                              //
//                                                                      //
// This is the main MVA steering class: it creates all MVA methods,     //
// and guides them through the training, testing and evaluation         //
// phases                                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <map>
#include "TCut.h"

#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

class TFile;
class TTree;
class TNtuple;
class TTreeFormula;
class TDirectory;

namespace TMVA {

   class IMethod;

   class Factory : public Configurable {

   public:

      // no default  constructor
      Factory( TString theJobName, TFile* theTargetFile, TString theOption = "" );

      // default destructor
      virtual ~Factory();

      virtual const char*  GetName() const { return "Factory"; }
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
      Bool_t SetInputTrees( TString signalFileName, TString backgroundFileName, 
                            Double_t signalWeight=1.0, Double_t backgroundWeight=1.0 );
      Bool_t SetInputTrees( TTree* inputTree, TCut SigCut, TCut BgCut = "" );

      // add events to training and testing trees
      void AddSignalTrainingEvent    ( std::vector<Double_t>& event, Double_t weight = 1.0 );
      void AddBackgroundTrainingEvent( std::vector<Double_t>& event, Double_t weight = 1.0 );
      void AddSignalTestEvent        ( std::vector<Double_t>& event, Double_t weight = 1.0 );
      void AddBackgroundTestEvent    ( std::vector<Double_t>& event, Double_t weight = 1.0 );

      // Set input trees at once
      Bool_t SetInputTrees( TTree* signal, TTree* background, 
                            Double_t signalWeight=1.0, Double_t backgroundWeight=1.0 );

      // set signal tree
      void AddSignalTree( TTree* signal, Double_t weight=1.0, Types::ETreeType treetype = Types::kMaxTreeType );
      void AddSignalTree( TTree* signal, Double_t weight, const TString& treetype );      
      // ... depreciated, kept for backwards compatibility
      void SetSignalTree( TTree* signal, Double_t weight=1.0 ); 

      // set background tree
      void AddBackgroundTree( TTree* background, Double_t weight=1.0, Types::ETreeType treetype = Types::kMaxTreeType );
      void AddBackgroundTree( TTree* background, Double_t weight, const TString & treetype );
      // ... depreciated, kept for backwards compatibility
      void SetBackgroundTree( TTree* background, Double_t weight=1.0 );

      // set input variable
      void SetInputVariables( std::vector<TString>* theVariables );
      void AddVariable( const TString& expression, char type='F',
                        Double_t min = 0, Double_t max = 0 ) { 
         Data().AddVariable( expression, min, max, type ); 
      }
      void SetWeightExpression( const TString& variable )  { 
         SetSignalWeightExpression    ( variable );
         SetBackgroundWeightExpression( variable );
      }
      void SetSignalWeightExpression( const TString& variable )  { 
         Data().SetSignalWeightExpression(variable);
      }
      void SetBackgroundWeightExpression( const TString& variable ) {
         Data().SetBackgroundWeightExpression(variable);
      }


      // prepare input tree for training
      void PrepareTrainingAndTestTree( const TCut& cut, 
                                       Int_t Ntrain, Int_t Ntest = -1 );

      void PrepareTrainingAndTestTree( const TCut& cut,
                                       Int_t NsigTrain, Int_t NbkgTrain, Int_t NsigTest, Int_t NbkgTest, 
                                       const TString& otherOpt );

      void PrepareTrainingAndTestTree( const TCut& cut, const TString& splitOpt );
      void PrepareTrainingAndTestTree( const TCut& sigcut, const TCut& bkgcut, const TString& splitOpt );

      Bool_t BookMethod( TString theMethodName, TString methodTitle, TString theOption = "" );
      Bool_t BookMethod( Types::EMVA theMethod,  TString methodTitle, TString theOption = "" );
      Bool_t BookMethod( TMVA::Types::EMVA theMethod, TString methodTitle, TString methodOption,
                         TMVA::Types::EMVA theCommittee, TString committeeOption = "" ); 

      // training for all booked methods
      void TrainAllMethods( void );

      // testing
      void TestAllMethods( void );

      // performance evaluation
      void EvaluateAllMethods( void );
      void EvaluateAllVariables( TString options = "" ); 
  
      // delete all methods and reset the method vector
      void DeleteAllMethods( void );

      // accessors
      IMethod* GetMethod( const TString& title ) const;

      Bool_t Verbose( void ) const { return fVerbose; }
      void SetVerbose( Bool_t v=kTRUE ) { fVerbose = v; Data().SetVerbose(Verbose()); }

      // make ROOT-independent C++ class for classifier response 
      // (classifier-specific implementation)
      // If no classifier name is given, help messages for all booked 
      // classifiers are printed
      virtual void MakeClass( const TString& methodTitle = "" ) const;

      // prints classifier-specific hepl messages, dedicated to 
      // help with the optimisation and configuration options tuning.
      // If no classifier name is given, help messages for all booked 
      // classifiers are printed
      void PrintHelpMessage( const TString& methodTitle = "" ) const;
    
   protected:

      //      TString GetDefaultName() const { return TString("default"); } CHGFT
      //      void CheckMethodTitle( TString & title ) { if (title.Length()==0) title=GetDefaultName(); } CHGFT

      DataSet& Data() const { return *fDataSet; }
      DataSet& Data()       { return *fDataSet; }
    
   private:

      // the beautiful greeting message
      void Greetings();

      // cd to local directory
      DataSet*         fDataSet;            // the dataset
      TFile*           fTargetFile;         // ROOT output file
      TString          fOptions;            // option string given by construction (presently only "V")
      Bool_t           fVerbose;            // verbose mode

      std::vector<TTreeFormula*> fInputVarFormulas; // local forulas of the same
      std::vector<IMethod*>      fMethods;          // all MVA methods
      TString                    fJobName;          // jobname, used as extension in weight file names

      // flag determining the way training and test data are assigned to Factory
      enum DataAssignType { kUndefined = 0, kAssignTrees, kAssignEvents };
      DataAssignType             fDataAssignType;         // flags for data assigning
      Bool_t                     fSuspendDATVerification; // do not temporarily verify

      // AssignType can only be event-wise OR tree-wise
      Bool_t VerifyDataAssignType  ( DataAssignType ); 
      void   CreateEventAssignTrees( TTree*&, const TString& name  );
      void   SetInputTreesFromEventAssignTrees();
      TTree*                     fTrainSigAssignTree; // tree for training events
      TTree*                     fTrainBkgAssignTree; // tree for training events
      TTree*                     fTestSigAssignTree;  // tree for test events
      TTree*                     fTestBkgAssignTree;  // tree for test events
      Int_t                      fATreeType;          // type of event (sig = 1, background = 0)
      Float_t                    fATreeWeight;        // weight of the event
      Float_t*                   fATreeEvent;         // event variables

      // local directory for each MVA in output target, used to store 
      // specific monitoring histograms (e.g., reference distributions of likelihood method)
      TDirectory* fLocalTDir;

   protected:

      ClassDef(Factory,0)  // The factory creates all MVA methods, and performs their training and testing
   };

} // namespace TMVA

#endif

