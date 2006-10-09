// @(#)root/tmva $Id: DataSet.h,v 1.27 2006/10/04 22:29:27 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSet                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Class to hold trees and variable definitions                              *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany,                                               *
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
// Class to contain all the data information                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"
#include "TTree.h"
#include "TCut.h"
#include "TTreeFormula.h"
#include "TMatrixD.h"
#include "TPrincipal.h"

#ifndef ROOT_TMVA_VariableInfo
#include "TMVA/VariableInfo.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

namespace TMVA {
   
   class TreeInfo {
   public:
      TreeInfo(TTree* tr, Double_t weight=1.0) : fTree(tr), fWeight(weight) {}
      ~TreeInfo(){}
      TTree* GetTree() const { return fTree; }
      Double_t GetWeight() const { return fWeight; }
   private:
      TTree* fTree;    //! pointer to the tree
      Double_t  fWeight;  //  weight for the tree
   };


   class DataSet {
   public:

      DataSet();
      ~DataSet(){};

      enum TreeType { kTraining=0, kTesting, kMaxTreeType };

      const char* GetName() const { return "DataSet"; }

      // the tree data
      void AddSignalTree(TTree* tr, Double_t weight=1.0);
      void AddBackgroundTree(TTree* tr, Double_t weight=1.0);
      UInt_t NSignalTrees()         { return fSignalTrees.size(); }
      UInt_t NBackgroundTrees()     { return fBackgroundTrees.size(); }
      TTree* SignalTree(int i)     { return fSignalTrees[i].GetTree(); }
      TTree* BackgroundTree(int i) { return fBackgroundTrees[i].GetTree(); }
      Double_t SignalTreeWeight(int i)     { return fSignalTrees[i].GetWeight(); }
      Double_t BackgroundTreeWeight(int i) { return fBackgroundTrees[i].GetWeight(); }
      void ClearSignalTreeList() { fSignalTrees.clear(); }
      void ClearBackgroundTreeList() { fBackgroundTrees.clear(); }

      // the variable data
      void AddVariable(const TString& expression, char varType='F', void* external = 0);
      UInt_t GetNVariables() const { return fVariables.size(); }
      const TString& GetExpression(int i) const { return fVariables[i].GetExpression(); }
      const TString& GetInternalVarName(int i) const { return fVariables[i].GetInternalVarName(); }
      char VarType(int i) const { return fVariables[i].VarType(); }
      char VarTypeOriginal(int i) const { return fVariables[i].VarTypeOriginal(); }
      Int_t FindVar(const TString& var) const;

      // the cut
      void SetCut( const TString& cut ) { fCut = TCut(cut); }
      void SetCut( const TCut& cut ) { fCut = cut; }
      void SetMultiCut( const TString& cut ) { fMultiCut = TCut(cut); }
      void SetMultiCut( const TCut& cut ) { fMultiCut = cut; }
      const TCut& Cut() const { return fCut; }
      const char* CutS() const { return fCut.GetTitle(); }
      Bool_t HasCut() { return TString(CutS())!=""; }

      // the internal trees
      TTree* GetTrainingTree()     const { return fTrainingTree; }
      TTree* GetTestTree()         const { return fTestTree; }
      TTree* GetMultiCutTestTree() const { return fMultiCutTestTree; }

      void SetTrainingTree    (TTree* tr) { fTrainingTree = tr; }
      void SetTestTree        (TTree* tr) { fTestTree = tr; }
      void SetMultiCutTestTree(TTree* tr) { fMultiCutTestTree = tr; }

      // ROOT stuff
      TDirectory* LocalRootDir() { return fLocalRootDir; }
      TDirectory* BaseRootDir()  { return fBaseRootDir; }
      void SetBaseRootDir(TDirectory* dir)  { fBaseRootDir = dir; }
      void SetLocalRootDir(TDirectory* dir) { fLocalRootDir = dir; }

      // data preparation
      // prepare input tree for training
      void PrepareForTrainingAndTesting(Int_t Ntrain = 0, Int_t Ntest = 0, TString TreeName="");

      // plot variables
      // possible values for tree are 'training', 'multi'
      void PlotVariables( TString tree, TString folderName, Types::PreprocessingMethod corr = Types::kNone );

      // auxiliary functions to compute decorrelation
      void GetCorrelationMatrix( Bool_t isSignal, TMatrixDBase* mat );
      void GetCovarianceMatrix ( Bool_t isSignal, TMatrixDBase*, Bool_t norm = kFALSE );
      void GetSQRMats( TMatrixD*& sqS, TMatrixD*& sqB, vector<TString>* theVars );
      void CalculatePrincipalComponents( TTree* originalTree, TPrincipal *&sigPrincipal, 
					 TPrincipal *&bgdPrincipal, vector<TString>* theVars );

      void SetVerbose(Bool_t v=kTRUE) { fVerbose = v; }

      // properties of the dataset
      // normalisation init
      void CalcNorm();
      // normalisation accessors      
      Double_t GetRMS( Int_t ivar, Types::PreprocessingMethod corr = Types::kNone) const {
         return fVariables[ivar].GetRMS(corr); }
      Double_t GetRMS( const TString& var, Types::PreprocessingMethod corr = Types::kNone) const { 
         return GetRMS(FindVar(var), corr); }
      Double_t GetMean( Int_t ivar, Types::PreprocessingMethod corr = Types::kNone) const {
         return fVariables[ivar].GetMean(corr); }
      Double_t GetMean( const TString& var, Types::PreprocessingMethod corr = Types::kNone) const { 
         return GetMean(FindVar(var), corr); }
      Double_t GetXmin( Int_t ivar, Types::PreprocessingMethod corr = Types::kNone) const {
         return fVariables[ivar].GetMin(corr); }
      Double_t GetXmax( Int_t ivar, Types::PreprocessingMethod corr = Types::kNone) const {
         return fVariables[ivar].GetMax(corr); }
      Double_t GetXmin( const TString& var, Types::PreprocessingMethod corr = Types::kNone) const { 
         return GetXmin(FindVar(var), corr); }
      Double_t GetXmax( const TString& var, Types::PreprocessingMethod corr = Types::kNone) const { 
         return GetXmax(FindVar(var), corr); }

      void     SetRMS ( const TString& var, Double_t x, Types::PreprocessingMethod corr = Types::kNone) { 
         SetRMS(FindVar(var), x, corr); }
      void     SetRMS( Int_t ivar, Double_t x, Types::PreprocessingMethod corr = Types::kNone) {
         fVariables[ivar].SetRMS(x, corr); }
      void     SetMean ( const TString& var, Double_t x, Types::PreprocessingMethod corr = Types::kNone) { 
         SetMean(FindVar(var), x, corr); }
      void     SetMean( Int_t ivar, Double_t x, Types::PreprocessingMethod corr = Types::kNone) {
         fVariables[ivar].SetMean(x, corr); }
      void     SetXmin( Int_t ivar, Double_t x, Types::PreprocessingMethod corr = Types::kNone) { 
         fVariables[ivar].SetMin(x, corr); }
      void     SetXmax( Int_t ivar, Double_t x, Types::PreprocessingMethod corr = Types::kNone) {
         fVariables[ivar].SetMax(x, corr); }
      void     SetXmin( const TString& var, Double_t x, Types::PreprocessingMethod corr = Types::kNone) { 
         SetXmin(FindVar(var), x, corr); }
      void     SetXmax( const TString& var, Double_t x, Types::PreprocessingMethod corr = Types::kNone) { 
         SetXmax(FindVar(var), x, corr); }
      void     UpdateNorm ( Int_t ivar, Double_t x, Types::PreprocessingMethod corr = Types::kNone);

      // event reading
      Bool_t ReadEvent(TTree* tr, UInt_t evidx, Types::PreprocessingMethod corr = Types::kNone, Types::SBType type = Types::kSignal) const;
      Bool_t ReadTrainingEvent( UInt_t evidx, Types::PreprocessingMethod corr = Types::kNone, Types::SBType type = Types::kSignal) const { 
         return ReadEvent(GetTrainingTree(),evidx,corr,type); }
      Bool_t ReadTestEvent(     UInt_t evidx, Types::PreprocessingMethod corr = Types::kNone, Types::SBType type = Types::kSignal) const { 
         return ReadEvent(GetTestTree(),evidx, corr,type); }

      TMVA::Event& Event() { if (fEvent==0) fEvent = new TMVA::Event(fVariables); return *fEvent; }
      void BackupEvent() { 
         if (fEventBackup==0) fEventBackup = new TMVA::Event(Event()); 
         else fEventBackup->CopyVarValues( Event() );
      }
      void RestoreEvent() { 
         Event().CopyVarValues( *fEventBackup );
      }
      const TMVA::Event& Event() const { return *fEvent; } // Warning, this requires that the event is already created

      // decorrelation Matrix accessors
      const TMatrixD* CorrelationMatrix(Types::SBType sigbgd) const { return fDecorrMatrix[sigbgd]; }
      TPrincipal* PrincipalComponents (Types::SBType sigbgd)  const { return fPrincipal[sigbgd]; }

      // the weight 
      void SetWeightExpression(const TString& expr) { fWeightExp = expr; }

      // some dataset stats
      Int_t GetNEvtTrain()     const { return fDataStats[kTraining][Types::kSBBoth]; }
      Int_t GetNEvtSigTrain()  const { return fDataStats[kTraining][Types::kSignal]; }
      Int_t GetNEvtBkgdTrain() const { return fDataStats[kTraining][Types::kBackground]; }
      Int_t GetNEvtTest()      const { return fDataStats[kTesting][Types::kSBBoth]; }
      Int_t GetNEvtSigTest()   const { return fDataStats[kTesting][Types::kSignal]; }
      Int_t GetNEvtBkgdTest()  const { return fDataStats[kTesting][Types::kBackground]; }

      // write and read functions
      void WriteVarsToStream(std::ostream& o, Types::PreprocessingMethod corr) const;
      void ReadVarsFromStream(std::istream& istr, Types::PreprocessingMethod corr);
      void WriteCorrMatToStream(std::ostream& o) const;
      void ReadCorrMatFromStream(std::istream& istr);

      // resets branch addresses to current event
      void ResetBranchAndEventAddresses( TTree* );
      void ResetCurrentTree() { fCurrentTree = 0; }

      // transformatoin for preprocessing
      Bool_t ApplyTransformation(Types::PreprocessingMethod corr = Types::kNone, Bool_t useSignal = kTRUE) const;

   private:
      
      // data manipulation helper functions
      // helper functions for writing decorrelated data
      void PreparePreprocessing( TTree* originalTree,
                                 TMatrixD*& sigCorrMat, TMatrixD*& bgdCorrMat );

      void ChangeToNewTree( TTree* tr );
      void PrintCorrelationMatrix( TTree* theTree );

      Double_t GetSeparation( TH1* S, TH1* B ) const;

      // verbosity
      Bool_t Verbose() { return fVerbose; }
      // plot variables
      void PlotVariables( TTree* theTree, TString folderName = "input_variables", Types::PreprocessingMethod corr = Types::kDecorrelated );
      
      // data members

      // ROOT stuff
      TDirectory*               fLocalRootDir;     //! the current directory, where things are created
      TDirectory*               fBaseRootDir;      //! the base directory, usually the root dir of a ROOT-file

      // input trees
      std::vector<TreeInfo>      fSignalTrees;      //! list of signal trees/weights
      std::vector<TreeInfo>      fBackgroundTrees;  //! list of signal trees/weights

      // expressions/formulas
      std::vector<VariableInfo>  fVariables;        //! list of variable expressions/internal names
      std::vector<TTreeFormula*> fInputVarFormulas; // local formulas of the same
      TCut                       fCut;              // the pretraining cut
      TCut                       fMultiCut;         // phase-space cut

      // the internal trees always as correlated and decorrelated version
      TTree*                    fTrainingTree;     //! tree used for training [correlated/decorrelated]
      TTree*                    fTestTree;         //! tree used for testing [correlated/decorrelated]
      TTree*                    fMultiCutTestTree; //! tree used for testing of multicut method [correlated/decorrelated]

      // data stats
      UInt_t                    fDataStats[kMaxTreeType][Types::kMaxSBType];  //! statistics of the dataset for training/test tree

      // 
      TMatrixD*                 fCovarianceMatrix[2];   //! Covariance matrix [signal/background]
      TMatrixD*                 fDecorrMatrix[2];       //! Decorrelation matrix [signal/background]
      TPrincipal*               fPrincipal[2];          //! Principal [signal/background]
      
      // verbosity
      Bool_t                    fVerbose;       //! Verbosity

      // the event 
      mutable TMVA::Event*      fEvent;         //! the event
      mutable TMVA::Event*      fEventBackup;   //! backup of non-preprocessed event (HOPEFULLY A TEMPORARY SOLUTION !!!)
      mutable TTree*            fCurrentTree;   //! the tree, events are currently read from
      mutable UInt_t            fCurrentEvtIdx; //! the current event (to avoid reading of the same event)

      // the weight
      TString                    fWeightExp;     //! the input formula string that is the weight
      TTreeFormula*              fWeightFormula; //! local weight formula
   };
}

#endif
