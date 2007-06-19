// @(#)root/tmva $Id: DataSet.h,v 1.8 2007/04/19 06:53:01 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSet                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Contains all the data information                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_DataSet
#define ROOT_TMVA_DataSet

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DataSet                                                              //
//                                                                      //
// Class that contains all the data information                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"
#include "TTree.h"
#include "TCut.h"
#include "TTreeFormula.h"
#include "TMatrixD.h"
#include "TPrincipal.h"

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_VariableInfo
#include "TMVA/VariableInfo.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_VariableTransformBase
#include "TMVA/VariableTransformBase.h"
#endif

namespace TMVA {
   
   class TreeInfo {

   public:

      TreeInfo( TTree* tr, Double_t weight=1.0 ) 
         : fTree(tr), fWeight(weight) {}
      ~TreeInfo() {}

      TTree*   GetTree()   const { return fTree; }
      Double_t GetWeight() const { return fWeight; }

   private:

      TTree*   fTree;    //! pointer to the tree
      Double_t fWeight;  //  weight for the tree
   };

   class DataSet {

   public:

      DataSet();
      virtual ~DataSet();

      const char* GetName() const { return "DataSet"; }

      // the tree data
      void     AddSignalTree    ( TTree* tr, Double_t weight=1.0 );
      void     AddBackgroundTree( TTree* tr, Double_t weight=1.0 );
      UInt_t   NSignalTrees()                const { return fTreeCollection[Types::kSignal].size(); }
      UInt_t   NBackgroundTrees()            const { return fTreeCollection[Types::kBackground].size(); }
      const TreeInfo&  SignalTreeInfo(Int_t i)     const { return fTreeCollection[Types::kSignal][i]; }
      const TreeInfo&  BackgroundTreeInfo(Int_t i) const { return fTreeCollection[Types::kBackground][i]; }
      void     ClearSignalTreeList()     { fTreeCollection[Types::kSignal].clear(); }
      void     ClearBackgroundTreeList() { fTreeCollection[Types::kBackground].clear(); }

      // the variable data
      void     AddVariable( const TString& expression, char varType='F', void* external = 0 );
      void     AddVariable( const TString& expression, Double_t min, Double_t max, char varType, void* external = 0 );
      std::vector<VariableInfo>& GetVariableInfos() { return fVariables; }
      UInt_t   GetNVariables()               const { return fVariables.size(); }
      char     GetVarType(Int_t i)           const { return fVariables[i].GetVarType(); }
      Int_t    FindVar(const TString& var)   const;

      const TString& GetExpression(Int_t i)      const { return fVariables[i].GetExpression(); }
      const TString& GetInternalVarName(Int_t i) const { return fVariables[i].GetInternalVarName(); }

      // the cut
      void SetCut( const TString& cut )      { SetCut(TCut(cut)); }
      void SetCut( const TCut& cut )         { fCut = cut; }
      void SetMultiCut( const TString& cut ) { SetMultiCut(TCut(cut)); }
      void SetMultiCut( const TCut& cut )    { fMultiCut = cut; }
      const TCut& Cut()  const { return fCut; }
      const char* CutS() const { return fCut.GetTitle(); }
      Bool_t HasCut()          { return TString(CutS())!=""; }

      // the internal trees
      TTree* GetTrainingTree()     const { return fTrainingTree; }
      TTree* GetTestTree()         const { return fTestTree; }
      TTree* GetMultiCutTestTree() const { return fMultiCutTestTree; }

      void SetTrainingTree    (TTree* tr) { fTrainingTree = tr; }
      void SetTestTree        (TTree* tr) { fTestTree = tr; }
      void SetMultiCutTestTree(TTree* tr) { fMultiCutTestTree = tr; }

      // ROOT stuff
      TDirectory* LocalRootDir()      const { return fLocalRootDir; }
      TDirectory* BaseRootDir()       const { return fBaseRootDir; }
      void SetBaseRootDir(TDirectory* dir)  { fBaseRootDir = dir; }
      void SetLocalRootDir(TDirectory* dir) { fLocalRootDir = dir; }

      // data preparation
      // prepare input tree for training
      void PrepareForTrainingAndTesting( const TString & splitOpt );

      // auxiliary functions to compute correlations
      void GetCorrelationMatrix( Bool_t isSignal, TMatrixDBase* mat );
      void GetCovarianceMatrix ( Bool_t isSignal, TMatrixDBase*, Bool_t norm = kFALSE );

      void SetVerbose( Bool_t v=kTRUE ) { fVerbose = v; }
      
      // finds transformation in map
      VariableTransformBase* FindTransform( Types::EVariableTransform transform ) const;

      // finds transformation in map or creates new one
      VariableTransformBase* GetTransform( Types::EVariableTransform transform );

      // event reading
      Bool_t ReadEvent        ( TTree* tr, Long64_t evidx ) const;
      Bool_t ReadTrainingEvent( Long64_t evidx ) const { return ReadEvent(GetTrainingTree(), evidx ); }
      Bool_t ReadTestEvent    ( Long64_t evidx ) const { return ReadEvent(GetTestTree(), evidx ); }

      TMVA::Event& GetEvent() { if (fEvent==0) fEvent = new TMVA::Event(fVariables); return *fEvent; }

      UInt_t GetCurrentEvtIdx() const { return fCurrentEvtIdx; } // the current event (to avoid reading of the same event)

      const TMVA::Event& GetEvent() const { return *fEvent; } // Warning, this requires an existing event object

      // correlation matrix 
      const TMatrixD* CorrelationMatrix( Types::ESBType sigbgd ) const { return fDecorrMatrix[sigbgd]; }

      // the weight 
      void SetWeightExpression(const TString& expr) { fWeightExp = expr; }

      // some dataset stats
      Int_t GetNEvtTrain()     const { return fDataStats[Types::kTraining][Types::kSBBoth]; }
      Int_t GetNEvtSigTrain()  const { return fDataStats[Types::kTraining][Types::kSignal]; }
      Int_t GetNEvtBkgdTrain() const { return fDataStats[Types::kTraining][Types::kBackground]; }
      Int_t GetNEvtTest()      const { return fDataStats[Types::kTesting][Types::kSBBoth]; }
      Int_t GetNEvtSigTest()   const { return fDataStats[Types::kTesting][Types::kSignal]; }
      Int_t GetNEvtBkgdTest()  const { return fDataStats[Types::kTesting][Types::kBackground]; }

      // resets branch addresses to current event
      void ResetBranchAndEventAddresses( TTree* );
      void ResetCurrentTree() { fCurrentTree = 0; }

   private:

      void ChangeToNewTree( TTree* tr );
      void PrintCorrelationMatrix( TTree* theTree );

      // verbosity
      Bool_t Verbose() { return fVerbose; }

      // data members

      // ROOT stuff
      TDirectory*                fLocalRootDir;     //! the current directory, where things are created
      TDirectory*                fBaseRootDir;      //! the base directory, usually the root dir of a ROOT-file

      // input trees
      std::vector<TreeInfo>      fTreeCollection[2]; //! list of signal and background trees/weights

      // expressions/formulas
      std::vector<VariableInfo>  fVariables;        //! list of variable expressions/internal names
      std::vector<TString>       fVariableStrings;  //! list of variable expressions
      std::vector<TTreeFormula*> fInputVarFormulas; // local formulas of the same
      TCut                       fCut;              // the pretraining cut
      TCut                       fMultiCut;         // phase-space cut

      TTree*                     fTrainingTree;     //! tree used for training
      TTree*                     fTestTree;         //! tree used for testing 
      TTree*                     fMultiCutTestTree; //! tree used for testing of multicut method

      // data stats
      UInt_t                     fDataStats[Types::kMaxTreeType][Types::kMaxSBType]; //! statistics of the dataset for training/test tree

      TMatrixD*                  fDecorrMatrix[2];     //! Decorrelation matrix [signal/background]

      std::map<Types::EVariableTransform,VariableTransformBase*> fVarTransforms; //! Registered variable transformations
      
      // verbosity
      Bool_t                    fVerbose;           //! Verbosity

      // the event 
      mutable TMVA::Event*      fEvent;             //! the event
      mutable TTree*            fCurrentTree;       //! the tree, events are currently read from
      mutable UInt_t            fCurrentEvtIdx;     //! the current event (to avoid reading of the same event)

      // the weight
      TString                   fWeightExp;        //! the input formula string that is the weight
      TTreeFormula*             fWeightFormula;    //! local weight formula
      
      mutable MsgLogger         fLogger;           //! message logger

   };
}

#endif
