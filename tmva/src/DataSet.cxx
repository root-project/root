// @(#)root/tmva $Id: DataSet.cxx,v 1.7 2006/11/20 15:35:28 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSet                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      MPI-K Heidelberg, Germany ,                                               *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <sstream>

#include "TMVA/DataSet.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"
#include "TMVA/MsgLogger.h"
#include "TEventList.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TMath.h"
#include "TMatrixF.h"
#include "TVectorF.h"

//_______________________________________________________________________
TMVA::DataSet::DataSet() 
   : fDoPreprocessing( kTRUE ),
     fLocalRootDir( 0 ),
     fCut( "" ),
     fMultiCut( "" ),
     fTrainingTree( 0 ),
     fTestTree( 0 ),
     fMultiCutTestTree( 0 ),
     fVerbose( kFALSE ),
     fEvent( 0 ),
     fEventBackup( 0 ),
     fCurrentTree( 0 ),
     fCurrentEvtIdx( 0 ),
     fWeightExp(""),
     fWeightFormula( 0 ),
     fLogger( GetName(), kINFO )
{
   // constructor

   fDecorrMatrix[0]   = fDecorrMatrix[1] = 0;
   fPrincipal[0]      = fPrincipal[1]    = 0;
   fPreprocessingEnabled[Types::kNone] = kTRUE;
   fPreprocessingEnabled[Types::kDecorrelated] = fPreprocessingEnabled[Types::kPCA] = kFALSE;

   for (Int_t corr=0; corr!=Types::kMaxPreprocessingMethod; corr++) fFlagPreprocessed[corr] = kFALSE;

   for (Int_t dim1=0; dim1!=Types::kMaxTreeType; dim1++) {
      for (Int_t dim2=0; dim2!=Types::kMaxSBType; dim2++) {
         fDataStats[dim1][dim2]=0;
      }
   }
}

TMVA::DataSet::~DataSet() 
{
   // destructor
}

//_______________________________________________________________________
Bool_t TMVA::DataSet::ReadEvent(TTree* tr, UInt_t evidx, Types::EPreprocessingMethod corr, Types::ESBType type) const 
{
   // read event from a tree into memory
   // after the reading the event transformation is called

   if (tr == 0) fLogger << kFATAL << "<ReadEvent> zero Tree Pointer encountered" << Endl;

   Bool_t needRead = kFALSE;
   if (fEvent == 0) {
      needRead = kTRUE;
      fEvent   = new TMVA::Event(fVariables);
   }
   if (tr != fCurrentTree) {
      needRead = kTRUE;
      if (fCurrentTree!=0) fCurrentTree->ResetBranchAddresses();
      fCurrentTree = tr;
      fEvent->SetBranchAddresses(tr);
   }
   if (evidx != fCurrentEvtIdx) {
      needRead = kTRUE;
      fCurrentEvtIdx = evidx;
   }
   if (!needRead) return kTRUE;

   // this needs to be changed, because we don't want to affect the other branches at all
   // pass this task to the event, which should hold list of branches
   std::vector<TBranch*>::iterator brIt = fEvent->Branches().begin();
   for (;brIt!=fEvent->Branches().end(); brIt++) (*brIt)->GetEntry(evidx);
   
   return ApplyTransformation( corr, (type == Types::kTrueType) ? Event().IsSignal() : (type == Types::kSignal) );
}

Bool_t TMVA::DataSet::ApplyTransformation( Types::EPreprocessingMethod corr, Bool_t useSignal ) const 
{ 
   // applies the transformation (none, decorellation, PCA) to the event data

   switch (corr) {

   case Types::kNone: 
      break;

   case Types::kDecorrelated: {
      const Int_t nvar = GetNVariables();

      TVectorD vec( nvar );
      for (Int_t ivar=0; ivar<nvar; ivar++) vec(ivar) = Event().GetVal(ivar);
      // diagonalize variable vectors      
      // figure out which vector to insert in new training tree
      vec *= (useSignal ? *(fDecorrMatrix[Types::kSignal]) : *(fDecorrMatrix[Types::kBackground]));
      
      for (Int_t ivar=0; ivar<nvar; ivar++) fEvent->SetVal(ivar,vec(ivar));

      break;
   }

   case Types::kPCA: {
      const Int_t nvar = GetNVariables();

      Double_t *dv = new Double_t[ nvar ];
      Double_t *rv = new Double_t[ nvar ];
      for (Int_t ivar=0; ivar<nvar; ivar++) dv[ivar] = Event().GetVal(ivar);
      
      // Perform PCA and put it into PCAed events tree
      PrincipalComponents( useSignal ? Types::kSignal : Types::kBackground )->X2P(dv, rv);
      for (Int_t ivar=0; ivar<nvar; ivar++) fEvent->SetVal(ivar, rv[ivar]);
      break;
   }
   default:
      fLogger << kFATAL << "<ApplyTransformation> unknown type: " << corr << Endl;
   }

   return kTRUE;
}

void TMVA::DataSet::ResetBranchAndEventAddresses( TTree* tree )
{
   // resets all branch adresses of the tree given as parameter 
   // to the event memory

   if (tree != 0) {
      tree->ResetBranchAddresses();
      if (fEvent != 0) fEvent->SetBranchAddresses( tree );
      fCurrentTree = tree;
   }
}

//_______________________________________________________________________
void TMVA::DataSet::AddSignalTree(TTree* tr, double weight) 
{
   // add a signal tree to the dataset to be used as input
   // multiple trees is not used at the moment, use chains instead
   fSignalTrees.push_back(TreeInfo(tr,weight));
}

//_______________________________________________________________________
void TMVA::DataSet::AddBackgroundTree(TTree* tr, double weight) 
{
   // add a background tree to the dataset to be used as input
   // multiple trees is not used at the moment, use chains instead
   fBackgroundTrees.push_back(TreeInfo(tr,weight));
}

//_______________________________________________________________________
void TMVA::DataSet::AddVariable(const TString& expression, char varType, void* external) 
{
   // add a variable (can be a complex expression) to the set of variables used in
   // the MV analysis
   fVariables.push_back(VariableInfo(expression, fVariables.size()+1, varType, external));
   fVariableStrings.push_back(expression);
}

//_______________________________________________________________________
// corr==0 - correlated, corr==1 - decorrelated, corr==2 - PCAed
void TMVA::DataSet::CalcNorm(Types::EPreprocessingMethod corr)
{
   // method to calculate minimum, maximum, mean, and RMS for all
   // variables used in the MVA for the given data preprocessing
   // method (none, decorrelate, PCA)

   // preprocessing method has to be enabled 
   if(!PreprocessingEnabled(corr)) return;

   if (GetTrainingTree()==0) return;

   GetTrainingTree()->ResetBranchAddresses();
   ResetCurrentTree();
   
   UInt_t nvar = GetNVariables();

   UInt_t nevts = GetTrainingTree()->GetEntries();
   
   TVectorD x2( nvar ); x2 *= 0;
   TVectorD x0( nvar ); x0 *= 0;   
   
   for (UInt_t ievt=0; ievt<nevts; ievt++) {
      ReadTrainingEvent(ievt,(Types::EPreprocessingMethod)corr);
      
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         Double_t x = Event().GetVal(ivar);
         UpdateNorm( ivar,  x, (Types::EPreprocessingMethod)corr);
         x2(ivar) += x*x;
         x0(ivar) += x;
      }
   }
   
   // get Mean and RMS
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      SetMean( ivar, x0(ivar)/nevts );
      SetRMS ( ivar, TMath::Sqrt( x2(ivar)/nevts - x0(ivar)*x0(ivar)/(nevts*nevts) ) );
   }
   
   fLogger << kVERBOSE << "set minNorm/maxNorm for " 
           << (corr==0?"correlated":corr==1?"decorrelated":"PCA") << " variables to: " << Endl;
   fLogger << setprecision(3);
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++)
      fLogger << "    " << GetInternalVarName(ivar)
              << "\t: [" << GetXmin( ivar, corr ) << "\t, " << GetXmax( ivar, corr ) << "\t] " << Endl;
   fLogger << setprecision(5); // reset to better value       
}

//_______________________________________________________________________
Int_t TMVA::DataSet::FindVar(const TString& var) const
{
   // find variable by name
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) 
      if (var == GetInternalVarName(ivar)) return ivar;
   fLogger << kFATAL << "<FindVar> variable \'" << var << "\' not found" << Endl;
   return -1;
}

//_______________________________________________________________________
void TMVA::DataSet::UpdateNorm ( Int_t ivar,  Double_t x, Types::EPreprocessingMethod corr) 
{
   // update min and max of a given variable and a given preprocessing method
   if (x < GetXmin( ivar, corr )) SetXmin( ivar, x, corr );
   if (x > GetXmax( ivar, corr )) SetXmax( ivar, x, corr );
}

//_______________________________________________________________________
void TMVA::DataSet::PrepareForTrainingAndTesting( Int_t Ntrain, Int_t Ntest, TString TreeName )
{ 
   // The internally used training and testing trees are prepaired in
   // this method 
   // First the variables (expressions) of interest are copied from
   // the given signal and background trees/chains into the local
   // trees (training and testing), according to the specified numbers
   // of training and testing events
   // Second DataSet::CalcNorm is called to determine min, max, mean,
   // and rms for all variables
   // Optionally (if specified as option) the decorrelation and PCA
   // preparation is executed

   fLogger << kINFO << "prepare training and Test samples" << Endl;
   fLogger << kINFO << "" << NSignalTrees()     
           << " signal trees with total number of events     : " << flush;
   fLogger << kINFO << SignalTree(0)->GetEntries() << Endl;
   fLogger << kINFO << NBackgroundTrees() 
           << " background trees with total number of events : " << flush;
   fLogger << kINFO << BackgroundTree(0)->GetEntries() << Endl;
   if (HasCut())
      fLogger << kINFO << "apply cut on input trees               : " << CutS() << Endl;
   else
      fLogger << kINFO << "no cuts applied" << Endl;

   // apply cuts to the input trees and create TEventLists of only the events
   // we would like to use !
   SignalTree(0)->Draw( ">>signalList", Cut(), "goff" );
   TEventList* signalList = (TEventList*)gDirectory->Get("signalList");

   BackgroundTree(0)->Draw( ">>backgList", Cut(), "goff" );
   TEventList* backgList = (TEventList*)gDirectory->Get("backgList");

   if (HasCut()) {
      fLogger << kINFO << "num of signal events passing cut       : " 
              << signalList->GetN() << Endl;
      fLogger << kINFO << "num of background eventspassing cut    : " 
              << backgList->GetN() << Endl;
   }

   Int_t nsig_train(0), nbkg_train(0), nsig_test(0), nbkg_test(0);
   Int_t nsig_test_min(0), nbkg_test_min(0);
   Int_t nsigTot = signalList->GetN();  
   Int_t nbkgTot = backgList->GetN();
   //  Int_t nTot    = nsigTot + nbkgTot;
   Int_t array[3];
   array[1] = nsigTot;
   array[2] = nbkgTot;

   if (Ntrain > 0 && Ntest == 0) {
      array[0]      = Ntrain;
      //      nsig_train    = TMath::MinElement(3,array);
      // to be backward compatible with ROOT 4.02
      nsig_train    = TMath::Min(array[0],TMath::Min(array[1],array[2]));
      nbkg_train    = nsig_train;
      nsig_test_min = nsig_train;
      nbkg_test_min = nsig_train;
    
      if ((nsigTot-nsig_train)<=0 ) 
         fLogger << kWARNING << "<PrepareForTrainingAndTesting> no. of signal events for testing <= 0! " 
                 << Endl;
      if ((nbkgTot-nbkg_test)<=0) 
         fLogger << kWARNING << "<PrepareForTrainingAndTesting> no. of background events for testing <= 0! " 
                 << Endl;
      nsig_test = nsigTot;
      nbkg_test = nbkgTot;    
   } 
   else if (Ntrain > 0 && Ntest > 0) {
      array[0]      = Ntrain;
      //      nsig_train    = TMath::MinElement(3,array);
      // to be backward compatible with ROOT 4.02
      nsig_train    = TMath::Min(array[0],TMath::Min(array[1],array[2]));
      nbkg_train    = nsig_train;
      nsig_test_min = nsig_train;
      nbkg_test_min = nsig_train;

      nsig_test     = TMath::Min(Ntest,nsigTot-nsig_train);
      nbkg_test     = TMath::Min(Ntest,nbkgTot-nsig_train);
      if (nsig_train != Ntrain) 
         fLogger << kWARNING << "<PrepareForTrainingAndTesting> less events for training than requested (1)" 
                 << Endl;
      if (nsig_test<Ntest || nbkg_test<Ntest)
         fLogger << kWARNING << "<PrepareForTrainingAndTesting> less events for testing than requested (2)" 
                 << Endl;
      nsig_test += nsig_train;
      nbkg_test += nsig_train;
   } 
   else if (Ntrain == -1 && Ntest == 0) {
      nsig_train = TMath::Min(nsigTot,nbkgTot);
      nbkg_train = nsig_train;
      nsig_test  = nsigTot;
      nbkg_test  = nbkgTot;
      nsig_test_min=1;
      nbkg_test_min=1;
      fLogger << kWARNING 
              << "<PrepareForTrainingAndTesting> same samples are used for training and testing" 
              << Endl;    
   }
   else if (Ntrain == -1&& Ntest == -1) {
      nsig_train = nsigTot;
      nbkg_train = nbkgTot;
      nbkg_test  = nbkgTot;
      nsig_test  = nsigTot;
      nsig_test_min=1;
      nbkg_test_min=1;
      fLogger << kWARNING 
              << "<PrepareForTrainingAndTesting> same samples are used for training and testing" 
              << Endl;    
      fLogger << kWARNING 
              << "<PrepareForTrainingAndTesting> "
              << "unequal numbers of signal and background events are used in training"
              << Endl;    
   }

   // provide detailed output
   fLogger << kINFO << "num of training signal events          : 1..." << nsig_train << Endl;  
   fLogger << kINFO << "num of training background events      : 1..." << nbkg_train << Endl;  
   fLogger << kINFO << "num of testing  signal events          : " 
           << nsig_test_min+1 << "..." << nsig_test << Endl;
   fLogger << kINFO << "num of testing  background events      : " 
           << nbkg_test_min+1 << "..." << nbkg_test << Endl;
  
   // create new trees
   // variable "type" is used to destinguish "0" = background;  "1" = Signal
   Int_t type; 
   Float_t weight, boostweight;
   const Long_t basketsize = 128000;
   BaseRootDir()->cd();

   TTree* trainingTree = new TTree("TrainingTree", "Variables used for MVA training");
   trainingTree->SetDirectory(0);
   SetTrainingTree(trainingTree);
   trainingTree->Branch( "type",       &type,        "type/I",        basketsize );
   trainingTree->Branch( "weight",     &weight,      "weight/F",      basketsize );
   trainingTree->Branch( "boostweight",&boostweight, "boostweight/F", basketsize );

   if (TreeName.Sizeof() >1) TreeName.Prepend("_");
   TTree* testTree = new TTree("TestTree"+TreeName, "Variables used for MVA testing, and MVA outputs" );
   SetTestTree(testTree);
   testTree->Branch( "type",       &type,        "type/I",        basketsize );
   testTree->Branch( "weight",     &weight,      "weight/F" ,     basketsize );
   testTree->Branch( "boostweight",&boostweight, "boostweight/F", basketsize );

   UInt_t nvars = GetNVariables();
   Float_t* vArr = new Float_t[nvars];
   Int_t*   iArr = new Int_t[nvars];
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) {
      // add Bbanch to training/test Tree
      const char* myVar = GetInternalVarName(ivar).Data();
      char vt = VarType(ivar);   // the variable type, 'F' or 'I'
      if (vt=='F') {
         trainingTree->Branch( myVar,&vArr[ivar], Form("%s/%c", myVar, vt), basketsize );
         testTree->Branch    ( myVar,&vArr[ivar], Form("%s/%c", myVar, vt), basketsize );
      } 
      else if (vt=='I') {
         trainingTree->Branch( myVar,&iArr[ivar], Form("%s/%c", myVar, vt), basketsize );
         testTree->Branch    ( myVar,&iArr[ivar], Form("%s/%c", myVar, vt), basketsize );
      } 
      else {
         fLogger << kFATAL << "<PrepareForTrainingAndTesting> unknown variable type '" 
                 << vt << "' encountered; allowed are: 'F' or 'I'"
                 << Endl;
      }
   } // end of loop over input variables
   
   // loop over signal events first, then over background events
   const char*  kindS[2]      = { "signal", "background" };
   TTree*       tr[2]         = { SignalTree(0), BackgroundTree(0) };
   Double_t     trwgt[2]      = { SignalTreeWeight(0), BackgroundTreeWeight(0) };
   TEventList*  evList[2]     = { signalList, backgList };
   Int_t        n_train[2]    = { nsig_train, nbkg_train };
   Int_t        n_test_min[2] = { nsig_test_min, nbkg_test_min };
   Int_t        n_test[2]     = { nsig_test, nbkg_test };
   // if we work with chains we need to remember the
   // current tree
   // if the chain jumps to a new tree
   // we have to reset the formulas
   for (Int_t sb=0; sb<2; sb++) { // sb=0 - signal, sb=1 - background
      TString currentFileName="";
      TTree * pCurrentTree  = 0; // used for chains only
      Bool_t isChain = (TString("TChain") == tr[sb]->ClassName());
      type = 1-sb;
      fLogger << kINFO << "create test tree: looping over " << kindS[sb] << " events ..." << Endl;
      Int_t ac=0;
      tr[sb]->LoadTree(0);
      ChangeToNewTree( tr[sb]->GetTree() );
      if(isChain) {
         currentFileName = tr[sb]->GetTree()->GetDirectory()->GetFile()->GetName();
         pCurrentTree = tr[sb]->GetTree();
      }

      // loop over events in ntuple
      for (Int_t i = 0; i < tr[sb]->GetEntries(); i++) {
         if (!evList[sb]->Contains(i)) continue;
         // survived the cut
         tr[sb]->LoadTree(i);
         if (isChain) {
            if (tr[sb]->GetTree()->GetDirectory()->GetFile()->GetName() != currentFileName) {
               currentFileName = tr[sb]->GetTree()->GetDirectory()->GetFile()->GetName();
               pCurrentTree = tr[sb]->GetTree();
               ChangeToNewTree( pCurrentTree );
            }
         }
         tr[sb]->GetEntry(i);
         Int_t sizeOfArrays = 1;
         Int_t prevArrExpr = 0;
         for (UInt_t ivar=0; ivar<nvars; ivar++) {
            Int_t ndata = fInputVarFormulas[ivar]->GetNdata();
            if (ndata==1) continue;
            if (sizeOfArrays==1) {
               sizeOfArrays = ndata;
               prevArrExpr = ivar;
            } 
            else if (sizeOfArrays!=ndata) {
               fLogger << kWARNING << "ERROR while preparing training and testing trees:" << Endl;
               fLogger << "   multiple array-type expressions of different length were encountered" << Endl;
               fLogger << "   location of error: event " << i << " in tree " << tr[sb]->GetName()
                       << " of file " << tr[sb]->GetCurrentFile()->GetName() << Endl;
               fLogger << "   expression " << fInputVarFormulas[ivar]->GetTitle() << " has " 
                       << ndata << " entries, while" << Endl;
               fLogger << "   expression " << fInputVarFormulas[prevArrExpr]->GetTitle() << " has "
                       << fInputVarFormulas[prevArrExpr]->GetNdata() << " entries, while" << Endl;
            }
         }

         for (Int_t idata = 0;  idata<sizeOfArrays; idata++) {
            Bool_t containsNaN = kFALSE;
            for (UInt_t ivar=0; ivar<nvars; ivar++) {
               Int_t ndata = fInputVarFormulas[ivar]->GetNdata();
               if (VarType(ivar)=='F') {
                  vArr[ivar] = (ndata == 1 ? 
                                fInputVarFormulas[ivar]->EvalInstance(0) : fInputVarFormulas[ivar]->EvalInstance(idata));
                  if (TMath::IsNaN(vArr[ivar])) {
                     containsNaN = kTRUE;
                     fLogger << kWARNING << "Expression resolves to infinite value (NaN): " << fInputVarFormulas[ivar]->GetTitle() << Endl;
                  }
               } 
               else if (VarType(ivar)=='I') {
                  double tmpVal = ndata == 1 ? 
                     fInputVarFormulas[ivar]->EvalInstance() : fInputVarFormulas[ivar]->EvalInstance(idata);
                  iArr[ivar] = tmpVal >= 0 ? Int_t(tmpVal+.5) : Int_t(tmpVal-0.5);
                  if (TMath::IsNaN(iArr[ivar])) {
                     containsNaN = kTRUE;
                     fLogger << kWARNING << "Expression resolves to infinite value (NaN): " << fInputVarFormulas[ivar]->GetTitle() << Endl;
                  }
               }
            }

            // the weight (can also be an array)
            weight = trwgt[sb]; // multiply by tree weight
            if (fWeightFormula!=0) {
               
               Int_t ndata = fWeightFormula->GetNdata();
               weight *= ndata==1?fWeightFormula->EvalInstance():fWeightFormula->EvalInstance(idata);
               if (TMath::IsNaN(weight)) {
                  containsNaN = kTRUE;
                  fLogger << kWARNING << "Weight expression resolves to infinite value (NaN): " 
                          << fWeightFormula->GetTitle() << Endl;
               }
            }

            // the boost weight is 1 per default
            boostweight = 1;

            if (containsNaN) {
               fLogger << kWARNING << "Event " << i << " rejected" << Endl;
               continue;
            }
            
            ac++;
            if ( ac <= n_train[sb]) {
               trainingTree->Fill();
               fDataStats[Types::kTraining][Types::kSBBoth]++;
               fDataStats[Types::kTraining][sb]++;
            }
            if ((ac > n_test_min[sb])&& (ac <= n_test[sb])) {
               testTree->Fill();
               fDataStats[Types::kTesting][Types::kSBBoth]++;
               fDataStats[Types::kTesting][sb]++;
            }
         }
      }
      tr[sb]->ResetBranchAddresses();

      // some output
      fLogger << kINFO << "collected " << fDataStats[Types::kTraining][sb] << " " << kindS[sb] 
              << " events for the training sample" << Endl;
      fLogger << kINFO << "collected " << fDataStats[Types::kTesting][sb] << " " << kindS[sb] 
              << " events for the test sample" << Endl;
   }

   // sanity check
   if (GetNEvtSigTrain() <= 0 || GetNEvtBkgdTrain() <= 0 ||
       GetNEvtSigTest() <= 0 || GetNEvtBkgdTest() <= 0) {
      fLogger << kFATAL << "zero events encountered for training and/or testing in signal and/or "
              << "background sample" << Endl;
   }

   GetTestTree()->ResetBranchAddresses();
   GetTrainingTree()->ResetBranchAddresses();

   if (Verbose()) {
      GetTrainingTree()->Print();
      GetTestTree()->Print();
      GetTrainingTree()->Show(0);
   }

   BaseRootDir()->cd();

   // print overall correlation matrix between all variables in tree  
   PrintCorrelationMatrix( GetTrainingTree() );

   // Calculate normalization based on the training tree
   CalcNorm(Types::kNone);

   // designed plotting output
   fLogger << kINFO << "plotting variables" << Endl;
   PlotVariables( "TrainingTree", "input_variables" );

   // Now some preprocessing
   Preprocess( Types::kDecorrelated );
   Preprocess( Types::kPCA );

   BaseRootDir()->mkdir("input_expressions")->cd();
   for (UInt_t ivar=0; ivar<nvars; ivar++) fInputVarFormulas[ivar]->Write();
   BaseRootDir()->cd();

   // sanity check (should be removed from all methods!)
   if (UInt_t(GetTrainingTree()->GetListOfBranches()->GetEntries() - 3) != GetNVariables()) {
      fLogger << kFATAL << "<PrepareForTrainingAndTesting> mismatch in number of variables" << Endl;
   }

   // reset the Event references
   ResetCurrentTree();

   delete [] vArr;
   delete [] iArr;
}

//_______________________________________________________________________
Bool_t TMVA::DataSet::Preprocess( Types::EPreprocessingMethod corr ) 
{
   // the dataset can be preprocessed with one of three methods 
   // - None: trivial identity
   // - Decorellation: transformation that decorellates the input variables
   // - PCA: Principal component analysis
   // first the transformations are calculated, then the minimum,
   // maximum, mean, and rms for the transformed set of variables are
   // calculated (in DataSet::CalcNorm)

   if( corr == Types::kNone ) return kTRUE;

   if( ! PreprocessingEnabled(corr) ) return kFALSE;

   if( HasBeenPreprocessed( corr ) ) return kFALSE;

   if( ! PreparePreprocessing( corr, GetTrainingTree() ) ) return kFALSE;

   CalcNorm(corr);

   switch(corr) {
   case Types::kDecorrelated:
      // So, is this the correlation or the covariance matrix?
      if (Verbose()) {
         fLogger << kVERBOSE << "SQRT covariance matrix for signal: " << Endl; 
         TMVA::Tools::FormattedOutput( *fDecorrMatrix[Types::kSignal], fVariableStrings, fLogger );
         fDecorrMatrix[Types::kSignal]->Print();
         fLogger << kVERBOSE << "SQRT covariance matrix for background: " << Endl;
         fDecorrMatrix[Types::kBackground]->Print();
         TMVA::Tools::FormattedOutput( *fDecorrMatrix[Types::kBackground], fVariableStrings, fLogger );
      }
      fLogger << kINFO << "plotting decorrelated variables" << Endl;
      PlotVariables( "TrainingTree", "decorrelated_input_variables", Types::kDecorrelated );
      break;
   case Types::kPCA:
      if (Verbose()) {      
         fLogger << kVERBOSE << "Covariance matrix eigenvectors for signal: " << Endl;
         PrincipalComponents( Types::kSignal )->GetEigenVectors()->Print();
         fLogger << kVERBOSE << "Covariance matrix eigenvectors for background: " << Endl;
         PrincipalComponents( Types::kBackground )->GetEigenVectors()->Print();
      }
      fLogger << kINFO << "plotting principal component variables" << Endl;
      PlotVariables( "TrainingTree", "principal_component_analyzed_input_variables", Types::kPCA );
   default:
      fLogger << kFATAL << "Unknown preprocessing method" << Endl;
   }

   ResetCurrentTree();

   FlagAsPreprocessed( corr );
   return kTRUE;
}

//_______________________________________________________________________
void TMVA::DataSet::ChangeToNewTree( TTree* tr )
{ 
   // While the data gets copied into the local training and testing
   // trees, the input tree can change (for intance when changing from
   // signal to background tree, or using TChains as input) The
   // TTreeFormulas, that hold the input expressions need to be
   // reassociated with the new tree, which is done here

   vector<TTreeFormula*>::const_iterator varFIt = fInputVarFormulas.begin();
   for (;varFIt!=fInputVarFormulas.end();varFIt++) delete *varFIt;
   fInputVarFormulas.clear();
   for (UInt_t i=0; i<GetNVariables(); i++) {
      TTreeFormula* ttf = new TTreeFormula( Form( "Formula%s", GetInternalVarName(i).Data() ),
                                            GetExpression(i).Data(), tr );
      fInputVarFormulas.push_back( ttf );

      // sanity check
      if (ttf->GetNcodes() == 0) {
         fLogger << kFATAL << "expression: " << GetExpression(i) 
                 << " does not appear to depend on any TTree variable --> please check spelling" << Endl;
      }
   }

   // recreate all formulas associated with the new tree
   // clear the old Formulas, if there are any
   //    vector<TTreeFormula*>::const_iterator varFIt = fInputVarFormulas.begin();
   //    for (;varFIt!=fInputVarFormulas.end();varFIt++) delete *varFIt;

   if (fWeightFormula!=0) { delete fWeightFormula; fWeightFormula=0; }
   if (fWeightExp!=TString("")) 
      fWeightFormula = new TTreeFormula("FormulaWeight",fWeightExp.Data(),tr);

   
   return;
}

//_______________________________________________________________________
void TMVA::DataSet::PlotVariables( TString tree, TString folderName, Types::EPreprocessingMethod corr )
{
   // wrapper around PlotVariable(TTree*) that allows to call
   // PlotVariables with a name

   if(!PreprocessingEnabled(corr)) return;
   tree.ToLower();
   if (tree.BeginsWith("train")) {
      PlotVariables( GetTrainingTree(), folderName, corr );
   } 
   else if (tree.BeginsWith("multi")) {
      PlotVariables( GetMultiCutTestTree(), folderName, corr );
   }
}

//_______________________________________________________________________
void TMVA::DataSet::PlotVariables( TTree* theTree, TString folderName, Types::EPreprocessingMethod corr )
{
   // create histograms from the input variables
   // - histograms for all input variables
   // - scatter plots for all pairs of input variables

   if(!PreprocessingEnabled(corr)) return;

   // if decorrelation has not been achieved, the decorrelation tree may be empty
   if (theTree == 0) return;

   theTree->ResetBranchAddresses();
   ResetCurrentTree();

   // create plots of the input variables and check them
   if (theTree==0) {
      fLogger << kINFO << "error: Empty tree in PlotVariables" << Endl;
      return;
   }
   fLogger << kVERBOSE << "plot input variables from '" << theTree->GetName() << "'" << Endl;

   // extension for preprocessor type
   TString prepType = "";
   if      (corr == Types::kDecorrelated) prepType += "_decorr";
   else if (corr == Types::kPCA)          prepType += "_PCA";

   const UInt_t nvar = GetNVariables();

   // compute means and RMSs
   TVectorD x2S( nvar ); x2S *= 0;
   TVectorD x2B( nvar ); x2B *= 0;
   TVectorD x0S( nvar ); x0S *= 0;   
   TVectorD x0B( nvar ); x0B *= 0;      
   TVectorD rmsS( nvar ), meanS( nvar ); 
   TVectorD rmsB( nvar ), meanB( nvar ); 
   
   UInt_t nevts = (UInt_t)theTree->GetEntries();
   UInt_t nS = 0, nB = 0;
   for (UInt_t ievt=0; ievt<nevts; ievt++) {
      ReadTrainingEvent( ievt, corr, Types::kTrueType );

      if (Event().IsSignal()) nS++;
      else                    nB++;

      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         Double_t x = Event().GetVal(ivar);
         if (Event().IsSignal()) {            
            x2S(ivar) += x*x;
            x0S(ivar) += x;
         }
         else {
            x2B(ivar) += x*x;
            x0B(ivar) += x;
         }
      }
   }
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      meanS(ivar) = x0S(ivar)/nS;
      meanB(ivar) = x0B(ivar)/nB;
      rmsS(ivar) = TMath::Sqrt( x2S(ivar)/nS - x0S(ivar)*x0S(ivar)/(nS*nS) );   
      rmsB(ivar) = TMath::Sqrt( x2B(ivar)/nB - x0B(ivar)*x0B(ivar)/(nB*nB) );   
   }

   // Create all histograms
   // do both, scatter and profile plots
   std::vector <TH1F*>      vS ( nvar );
   std::vector <TH1F*>      vB ( nvar );
   std::vector < std::vector <TH2F*> >      mycorrS ( nvar );
   std::vector < std::vector <TH2F*> >      mycorrB ( nvar );
   std::vector < std::vector <TProfile*> >  myprofS ( nvar );
   std::vector < std::vector <TProfile*> >  myprofB ( nvar );
   for (UInt_t ivar=0; ivar < nvar; ivar++){
      mycorrS[ivar].resize(nvar);
      mycorrB[ivar].resize(nvar);
      myprofS[ivar].resize(nvar);
      myprofB[ivar].resize(nvar);
   }
   

   Float_t timesRMS  = 4.0;
   UInt_t  nbins1D   = 70;
   UInt_t  nbins2D   = 300;
   for (UInt_t i=0; i<nvar; i++) {
      TString myVari = GetInternalVarName(i);  

      // choose reasonable histogram ranges, by removing outliers
      Double_t xmin = TMath::Max( GetXmin(i,corr), TMath::Min( meanS(i) - timesRMS*rmsS(i), meanB(i) - timesRMS*rmsB(i) ) );
      Double_t xmax = TMath::Min( GetXmax(i,corr), TMath::Max( meanS(i) + timesRMS*rmsS(i), meanB(i) + timesRMS*rmsB(i) ) );

      vS[i] = new TH1F( Form("%s__S%s", myVari.Data(), prepType.Data()), GetExpression(i), nbins1D, xmin, xmax );
      vB[i] = new TH1F( Form("%s__B%s", myVari.Data(), prepType.Data()), GetExpression(i), nbins1D, xmin, xmax );

      vS[i]->SetXTitle(GetExpression(i));
      vB[i]->SetXTitle(GetExpression(i));
      vS[i]->SetLineColor(4);
      vB[i]->SetLineColor(2);

      // the profile and scatter plots
      for (UInt_t j=i+1; j<nvar; j++) {
         TString myVarj = GetInternalVarName(j);  

         mycorrS[i][j] = new TH2F( Form( "scat_%s_vs_%s_sig%s", myVarj.Data(), myVari.Data(), prepType.Data() ), 
                                   Form( "%s versus %s (signal)%s", myVarj.Data(), myVari.Data(), prepType.Data() ), 
                                   nbins2D, GetXmin(i,corr), GetXmax(i,corr), 
                                   nbins2D, GetXmin(j,corr), GetXmax(j,corr) );
         mycorrB[i][j] = new TH2F( Form( "scat_%s_vs_%s_bgd%s", myVarj.Data(), myVari.Data(), prepType.Data() ), 
                                   Form( "%s versus %s (background)%s", myVarj.Data(), myVari.Data(), prepType.Data() ), 
                                   nbins2D, GetXmin(i,corr), GetXmax(i,corr), 
                                   nbins2D, GetXmin(j,corr), GetXmax(j,corr) );

         myprofS[i][j] = new TProfile( Form( "prof_%s_vs_%s_sig%s", myVarj.Data(), myVari.Data(), prepType.Data() ), 
                                       Form( "profile %s versus %s (signal)%s", myVarj.Data(), myVari.Data(), prepType.Data() ), 
                                       nbins1D, GetXmin(i,corr), GetXmax(i,corr) );
         myprofB[i][j] = new TProfile( Form( "prof_%s_vs_%s_bgd%s", myVarj.Data(), myVari.Data(), prepType.Data() ), 
                                       Form( "profile %s versus %s (background)%s", myVarj.Data(), myVari.Data(), prepType.Data() ), 
                                       nbins1D, GetXmin(i,corr), GetXmax(i,corr) );
      }
   }   
     
   // fill the histograms (this approach should be faster than individual projection
   for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {

      ReadEvent( theTree, ievt, corr, Types::kTrueType );
      Float_t weight = Event().GetWeight();

      for (UInt_t i=0; i<nvar; i++) {
         Float_t vali = Event().GetVal(i);

         // variable histos
         if (Event().IsSignal()) vS[i]->Fill( vali, weight );
         else                    vB[i]->Fill( vali, weight );
         
         // correlation histos
         for (UInt_t j=i+1; j<nvar; j++) {
            Float_t valj = Event().GetVal(j);
            if (Event().IsSignal()) {
               mycorrS[i][j]->Fill( vali, valj, weight );
               myprofS[i][j]->Fill( vali, valj, weight );
            }
            else {
               mycorrB[i][j]->Fill( vali, valj, weight );
               myprofB[i][j]->Fill( vali, valj, weight );
            }
         }
      }
   }

   
   // computes ranking of input variables
   // create the ranking object
   Ranking* ranking = new Ranking( GetName(), "Separation" );
   for (UInt_t i=0; i<nvar; i++) {   
      Double_t sep = GetSeparation( vS[i], vB[i] );
      ranking->AddRank( *new Rank( vS[i]->GetTitle(), sep ) );
   }
   
   // write histograms

   // create directory in output file
   TDirectory* localDir= BaseRootDir()->mkdir( folderName );
   localDir->cd();
   fLogger << kVERBOSE << "change to dir: " << localDir->GetPath() << Endl;
   for (UInt_t i=0; i<nvar; i++) {
      vS[i]->Write();
      vB[i]->Write();
   }

   // correlation plots have dedicated directory
   TString dirName = "CorrelationPlots" + prepType;
   fLogger << kINFO << "create scatter and profile plots in directory: " << dirName
           << Endl;

   TDirectory* dir = 0;
   TObject * o = BaseRootDir()->FindObject(dirName);
   if (o!=0 && o->InheritsFrom("TDirectory")) dir = (TDirectory*)o;
   if (dir == 0) dir = BaseRootDir()->mkdir( dirName );
   dir ->cd();

   for (UInt_t i=0; i<nvar; i++) {
      vS[i]->Write();
      vB[i]->Write();
      for (UInt_t j=i+1; j<nvar; j++) {
         mycorrS[i][j]->Write();
         mycorrB[i][j]->Write();
         myprofS[i][j]->Write();
         myprofB[i][j]->Write();
      }
   }   

   // output ranking results
   if (ranking != 0 && corr == Types::kNone) {
      fLogger << kINFO << "pre-ranking input variables..." << Endl;
      ranking->Print();
   }
   BaseRootDir()->cd();
   theTree->ResetBranchAddresses();
}

//_______________________________________________________________________
void TMVA::DataSet::PrintCorrelationMatrix( TTree* theTree )
{ 
   // calculates the correlation matrices for signal and background, 
   // and print them to standard output

   // first remove type from variable set
   fLogger << kINFO << "compute correlation matrices on tree: " 
           << theTree->GetName() << Endl;

   TBranch*         branch = 0;
   vector<TString>* theVars = new vector<TString>;
   TObjArrayIter branchIter( theTree->GetListOfBranches(), kIterForward );
   while ((branch = (TBranch*)branchIter.Next()) != 0) 
      if ((TString)branch->GetName() != "type"  &&
          (TString)branch->GetName() != "weight"&&
          (TString)branch->GetName() != "boostweight") theVars->push_back( branch->GetName() );

   Int_t nvar = (int)theVars->size();
   TMatrixD* corrMatS = new TMatrixD( nvar, nvar );
   TMatrixD* corrMatB = new TMatrixD( nvar, nvar );

   GetCorrelationMatrix( kTRUE,  corrMatS );
   GetCorrelationMatrix( kFALSE, corrMatB );

   // print the matrix
   fLogger << Endl;
   fLogger << kINFO << "correlation matrix (signal):" << Endl;
   TMVA::Tools::FormattedOutput( *corrMatS, *theVars, fLogger );
   fLogger << Endl;

   fLogger << kINFO << "correlation matrix (background):" << Endl;
   TMVA::Tools::FormattedOutput( *corrMatB, *theVars, fLogger );
   fLogger << Endl;

   // ---- histogramming
   LocalRootDir()->cd();

   // loop over signal and background
   TString      hName[2]  = { "CorrelationMatrixS", "CorrelationMatrixB" };
   TString      hTitle[2] = { "Correlation Matrix (signal)", "Correlation Matrix (background)" };

   // workaround till the TMatrix templates are comonly used
   // this keeps backward compatibility
   TMatrixF*    tmS = new TMatrixF( nvar, nvar );
   TMatrixF*    tmB = new TMatrixF( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      for (Int_t jvar=0; jvar<nvar; jvar++) {
         (*tmS)(ivar, jvar) = (*corrMatS)(ivar,jvar);
         (*tmB)(ivar, jvar) = (*corrMatB)(ivar,jvar);
      }
   }  

   TMatrixF* mObj[2]  = { tmS, tmB };

   // settings
   const Float_t labelSize = 0.055;

   for (Int_t ic=0; ic<2; ic++) { 

      TH2F* h2 = new TH2F( *(mObj[ic]) );
      h2->SetNameTitle( hName[ic], hTitle[ic] );

      for (Int_t ivar=0; ivar<nvar; ivar++) {
         h2->GetXaxis()->SetBinLabel( ivar+1, GetExpression(ivar) );
         h2->GetYaxis()->SetBinLabel( ivar+1, GetExpression(ivar) );
      }

      // present in percent, and round off digits
      // also, use absolute value of correlation coefficient (ignore sign)
      h2->Scale( 100.0  ); 
      for (Int_t ibin=1; ibin<=nvar; ibin++)
         for (Int_t jbin=1; jbin<=nvar; jbin++)
            h2->SetBinContent( ibin, jbin, Int_t(h2->GetBinContent( ibin, jbin )) );

      // style settings
      h2->SetStats( 0 );
      h2->GetXaxis()->SetLabelSize( labelSize );
      h2->GetYaxis()->SetLabelSize( labelSize );
      h2->SetMarkerSize( 1.5 );
      h2->SetMarkerColor( 0 );
      h2->LabelsOption( "d" ); // diagonal labels on x axis
      h2->SetLabelOffset( 0.011 );// label offset on x axis
      h2->SetMinimum( -100.0 );
      h2->SetMaximum( +100.0 );

      // -------------------------------------------------------------------------------------
      // just in case one wants to change the position of the color palette axis
      // -------------------------------------------------------------------------------------
      //     gROOT->SetStyle("Plain");
      //     TStyle* gStyle = gROOT->GetStyle( "Plain" );
      //     gStyle->SetPalette( 1, 0 );
      //     TPaletteAxis* paletteAxis 
      //                   = (TPaletteAxis*)h2->GetListOfFunctions()->FindObject( "palette" );
      // -------------------------------------------------------------------------------------

      // write to file
      h2->Write();
      fLogger << kVERBOSE << "created correlation matrix as 2D histogram: " << h2->GetName() << Endl;
      
      delete h2;
   }
   // ----  

   delete tmS;
   delete tmB;

   delete theVars;
   delete corrMatS;
   delete corrMatB;
}

//_______________________________________________________________________
Double_t TMVA::DataSet::GetSeparation( TH1* S, TH1* B ) const
{
   // compute "separation" defined as
   // <s2> = (1/2) Int_-oo..+oo { (S(x)2 - B(x)2)/(S(x) + B(x)) dx }
   Double_t separation = 0;
   
   // sanity checks
   // signal and background histograms must have same number of bins and 
   // same limits
   if ( (S->GetNbinsX() != B->GetNbinsX()) || (S->GetNbinsX() <= 0) ) {
      fLogger << kFATAL << "<GetSeparation> signal and background"
              << " histograms have different number of bins: " 
              << S->GetNbinsX() << " : " << B->GetNbinsX() << Endl;
   }

   if (S->GetXaxis()->GetXmin() != B->GetXaxis()->GetXmin() || 
       S->GetXaxis()->GetXmax() != B->GetXaxis()->GetXmax() || 
       S->GetXaxis()->GetXmax() <= S->GetXaxis()->GetXmin()) {
      fLogger << kINFO << S->GetXaxis()->GetXmin() << " " << B->GetXaxis()->GetXmin() 
              << " " << S->GetXaxis()->GetXmax() << " " << B->GetXaxis()->GetXmax() 
              << " " << S->GetXaxis()->GetXmax() << " " << S->GetXaxis()->GetXmin() << Endl;
      fLogger << kFATAL << "<GetSeparation> signal and background"
              << " histograms have different or invalid dimensions:" << Endl;
   }

   Int_t nstep  = S->GetNbinsX();
   Double_t intBin = (S->GetXaxis()->GetXmax() - S->GetXaxis()->GetXmin())/nstep;
   for (Int_t bin=0; bin<nstep; bin++) {
      Double_t s = S->GetBinContent( bin );
      Double_t b = B->GetBinContent( bin );
      // separation
      if (s + b > 0) separation += 0.5*(s - b)*(s - b)/(s + b);
   }
   separation *= intBin;

   return separation;
}

//_______________________________________________________________________
Bool_t TMVA::DataSet::PreparePreprocessing( Types::EPreprocessingMethod corr, TTree* originalTree )
{
   // For the preprocessing methods kDecorrelate and kPCA the
   // transformation matrices are calculated.  For the PCA the class
   // TPrincipal is used.

   if( originalTree == 0 ) return kFALSE;

   // creates a deep copy of a tree with all of the values decorrelated
   // the result is saved in passed-by-reference decorrTree

   if (GetNVariables() > 200) { 
      fLogger << kINFO << "----------------------------------------------------------------------------" 
              << Endl;
      fLogger << kINFO 
              << ": More than 200 variables, will not calculate decorrelation matrix "
              << originalTree->GetName() << "!" << Endl;
      fLogger << kINFO << "----------------------------------------------------------------------------" 
              << Endl;
      return kFALSE;
   }
   
   ResetCurrentTree();

   fLogger << kINFO << "calculate matrices to perform data transformation from " << originalTree->GetName() << Endl;

   // get a vector of variable names and figure out number of variables
   TBranch*         branch = 0;
   vector<TString>* theVars = new vector<TString>;
   TObjArrayIter    branchIter( originalTree->GetListOfBranches(), kIterForward );
   Bool_t           hasInteger = kFALSE;
   while ((branch = (TBranch*)branchIter.Next()) != 0) {
      if ((TString)branch->GetName() != "type"  &&
          (TString)branch->GetName() != "weight"&&
          (TString)branch->GetName() != "boostweight") {
         theVars->push_back( branch->GetName() );
         
         // if the data set has an integer (discrete) variable, decorrelation 
         if ((TString)((TLeaf*)branch->GetListOfLeaves()->At(0))->GetTypeName() == "Int_t") hasInteger = kTRUE;
      }
   }

   if (hasInteger) {
      fLogger << kWARNING << "<CreateDecorrelatedTree> tree contains "
              << "integer variable --> will not decorrelate variables" << Endl;
      return kFALSE;
   }

   switch( corr ) {
   case Types::kDecorrelated:
      // get the covariance and square-root matrices for signal and background
      GetSQRMats( fDecorrMatrix[Types::kSignal], fDecorrMatrix[Types::kBackground], theVars );
      break;
   case Types::kPCA:
      // This is getting ridiculous from both computation and storage points of view. 
      // Have to figure out beforehand what the methods need.
      CalculatePrincipalComponents( originalTree,
                                    fPrincipal[Types::kSignal], fPrincipal[Types::kBackground], theVars );
      break;
   default:
      fLogger << kERROR << "Unknown preprocessing method" << Endl;
   }

   return kTRUE;
}

void TMVA::DataSet::GetCorrelationMatrix( Bool_t isSignal, TMatrixDBase* mat )
{
   // computes correlation matrix for variables "theVars" in tree;
   // "theType" defines the required event "type" 
   // ("type" variable must be present in tree)

   // first compute variance-covariance
   GetCovarianceMatrix( isSignal, mat );

   // now the correlation
   UInt_t nvar = GetNVariables(), ivar, jvar;

   for (ivar=0; ivar<nvar; ivar++) {
      for (jvar=0; jvar<nvar; jvar++) {
         if (ivar != jvar) {
            Double_t d = (*mat)(ivar, ivar)*(*mat)(jvar, jvar);
            if (d > 0) (*mat)(ivar, jvar) /= sqrt(d);
            else {
               fLogger << kWARNING << "<GetCorrelationMatrix> zero variances for variables "
                       << "(" << ivar << ", " << jvar << ")" << Endl;
               (*mat)(ivar, jvar) = 0;
            }
         }
      }
   }

   for (UInt_t ivar=0; ivar<nvar; ivar++) (*mat)(ivar, ivar) = 1.0;
}

void TMVA::DataSet::GetCovarianceMatrix( Bool_t isSignal, TMatrixDBase* mat, Bool_t norm )
{
   // compute covariance matrix

   UInt_t nvar = GetNVariables(), ivar = 0, jvar = 0;

   // init matrices
   TVectorD vec(nvar);
   TMatrixD mat2(nvar, nvar);      
   for (ivar=0; ivar<nvar; ivar++) {
      vec(ivar) = 0;
      for (jvar=0; jvar<nvar; jvar++) {
         mat2(ivar, jvar) = 0;
      }
   }

   // if normalisation required, determine min/max
   TVectorF xmin(nvar), xmax(nvar);
   if (norm) {
      for (Int_t i=0; i<GetTrainingTree()->GetEntries(); i++) {
         // fill the event
         ReadTrainingEvent(i);

         for (ivar=0; ivar<nvar; ivar++) {
            if (i == 0) {
               xmin(ivar) = Event().GetVal(ivar);
               xmax(ivar) = Event().GetVal(ivar);
            }
            else {
               xmin(ivar) = TMath::Min( xmin(ivar), Event().GetVal(ivar) );
               xmax(ivar) = TMath::Max( xmax(ivar), Event().GetVal(ivar) );
            }
         }
      }
   }

   // perform event loop
   Int_t ic = 0;
   for (Int_t i=0; i<GetTrainingTree()->GetEntries(); i++) {

      // fill the event
      ReadTrainingEvent(i);

      if (Event().IsSignal() == isSignal) {
         ic++; // count used events
         for (ivar=0; ivar<nvar; ivar++) {

            Double_t xi = ( (norm) ? Tools::NormVariable( Event().GetVal(ivar), xmin(ivar), xmax(ivar) )
                            : Event().GetVal(ivar) );
            vec(ivar) += xi;
            mat2(ivar, ivar) += (xi*xi);

            for (jvar=ivar+1; jvar<nvar; jvar++) {
               Double_t xj =  ( (norm) ? Tools::NormVariable( Event().GetVal(jvar), xmin(ivar), xmax(ivar) )
                                : Event().GetVal(jvar) );
               mat2(ivar, jvar) += (xi*xj);
               mat2(jvar, ivar) = mat2(ivar, jvar); // symmetric matrix
            }
         }
      }
   }

   // variance-covariance
   Double_t n = (Double_t)ic;
   for (ivar=0; ivar<nvar; ivar++) {
      for (jvar=0; jvar<nvar; jvar++) {
         (*mat)(ivar, jvar) = mat2(ivar, jvar)/n - vec(ivar)*vec(jvar)/(n*n);
      }
   }
}

//_______________________________________________________________________
void TMVA::DataSet::GetSQRMats( TMatrixD*& sqS, TMatrixD*& sqB, vector<TString>* theVars )
{
   // compute square-root matrices for signal and background
   if (NULL != sqS)  { delete sqS; sqS = 0;  }
   if (NULL != sqB)  { delete sqB; sqB = 0;  }

   Int_t nvar = (int)theVars->size();
   TMatrixDSym* covMatS = new TMatrixDSym( nvar );
   TMatrixDSym* covMatB = new TMatrixDSym( nvar );

   GetCovarianceMatrix( kTRUE,  covMatS, kFALSE );
   GetCovarianceMatrix( kFALSE, covMatB, kFALSE );

   TMVA::Tools::GetSQRootMatrix( covMatS, sqS );
   TMVA::Tools::GetSQRootMatrix( covMatB, sqB );
}

//_______________________________________________________________________
void TMVA::DataSet::CalculatePrincipalComponents (TTree* originalTree, TPrincipal *&sigPrincipal, TPrincipal *&bgdPrincipal, vector<TString>* theVars )
{
   // calculate the principal components for the signal and the background data
   // it uses the MakePrincipal method of ROOT's TPrincipal class

   if (sigPrincipal != NULL) { delete sigPrincipal; sigPrincipal = 0; }
   if (bgdPrincipal != NULL) { delete bgdPrincipal; sigPrincipal = 0; }

   Int_t nvar = (int)theVars->size();
   sigPrincipal = new TPrincipal (nvar, ""); // Not normalizing and not storing input data, for performance reasons. Should perhaps restore normalization.
   bgdPrincipal = new TPrincipal (nvar, "");

   // Should we shove this into TMVA::Tools?

   TObjArrayIter   branchIter( originalTree->GetListOfBranches(), kIterForward );
   TBranch*      branch = NULL;
   Long64_t      ievt, entries = originalTree->GetEntries();
   Float_t  *      fvec = new Float_t [nvar];
   Double_t *      dvec = new Double_t [nvar];
   Int_t         type, jvar=-1;
   Float_t         weight, boostweight;
   
   while ((branch = (TBranch*)branchIter.Next()) != 0) { 
      if ((TString)branch->GetName() == "type") 
         originalTree->SetBranchAddress( branch->GetName(), &type );
      else if ((TString)branch->GetName() == "weight")
         originalTree->SetBranchAddress( branch->GetName(), &weight );
      else if ((TString)branch->GetName() == "boostweight")
         originalTree->SetBranchAddress( branch->GetName(), &boostweight );
      else
         originalTree->SetBranchAddress( branch->GetName(), &fvec[++jvar] );
   }

   for (ievt=0; ievt<entries; ievt++) {
      originalTree->GetEntry( ievt );
      TPrincipal *princ = type == Types::kSignal ? sigPrincipal : bgdPrincipal;
      for (Int_t i = 0; i < nvar; i++)
         dvec [i] = (Double_t) fvec [i];
      princ->AddRow (dvec);
   }

   sigPrincipal->MakePrincipals();
   bgdPrincipal->MakePrincipals();

   delete fvec; delete dvec;
}

//_______________________________________________________________________
void TMVA::DataSet::WriteVarsToStream(std::ostream& o, Types::EPreprocessingMethod corr) const 
{
   // write the list of variables (name, min, max) for a given data
   // transformation method to the stream

   o << "NVar " << GetNVariables() << endl;
   std::vector<VariableInfo>::const_iterator varIt = fVariables.begin();
   for (;varIt!=fVariables.end(); varIt++) varIt->WriteToStream(o,corr);
}

//_______________________________________________________________________
void TMVA::DataSet::WriteCorrMatToStream(std::ostream& o) const
{
   // write the decorrelation matrix to the stream
   for (Int_t matType=0; matType<2; matType++) {
      o << "# correlation matrix " << endl;
      TMatrixD* mat = fDecorrMatrix[matType];
      o << (matType==0?"signal":"background") << " " << mat->GetNrows() << " x " << mat->GetNcols() << endl;
      for (Int_t row = 0; row<mat->GetNrows(); row++) {
         for (Int_t col = 0; col<mat->GetNcols(); col++) {
            o << setw(15) << (*mat)[row][col];
         }
         o << endl;
      }
   }
   o << "##" << endl;
}

//_______________________________________________________________________
void TMVA::DataSet::ReadVarsFromStream(std::istream& istr, Types::EPreprocessingMethod corr) 
{
   // Read the variables (name, min, max) for a given data
   // transformation method from the stream.

   TString dummy;
   Int_t readNVar;
   istr >> dummy >> readNVar;
   VariableInfo varInfo;

   // we want to make sure all variables are read and that there are no more variables defined
   // first we create a local vector with all the expressions
   std::vector<TString> varExp;
   std::vector<VariableInfo>::iterator varIt = fVariables.begin();
   for (;varIt!=fVariables.end(); varIt++)
      varExp.push_back(varIt->GetExpression());

   // now read 'readNVar' lines
   for (Int_t i=0; i<readNVar; i++) {
      varInfo.ReadFromStream(istr,corr);
      varIt = fVariables.begin();
      for (;varIt!=fVariables.end(); varIt++) {
         if ( varIt->GetExpression() == varInfo.GetExpression()) {
            varInfo.SetExternalLink((*varIt).GetExternalLink());
            (*varIt) = varInfo;
            break;
         }
      }
      if (varIt==fVariables.end()) {
         fLogger << kFATAL << "Trying to read undeclared variable \'" 
                 << varInfo.GetExpression() << "\'" << Endl;
      }
      // now remove the varexp from the local list
      std::vector<TString>::iterator varExpIt = varExp.begin();
      for (;varExpIt!=varExp.end(); varExpIt++) {
         if ( (*varExpIt) == varInfo.GetExpression()) {
            varExp.erase(varExpIt); break;
         }
      }
      if (varExpIt==varExp.end()&& varExp.size()>0) {
         fLogger << kFATAL << "Trying to read variable \'" 
                 << varInfo.GetExpression() << "\' twice" << Endl;
      }      
   }
}

//_______________________________________________________________________
void TMVA::DataSet::ReadCorrMatFromStream(std::istream& istr )
{
   // Read the decorellation matrix from an input stream

   char buf[512];
   istr.getline(buf,512);
   TString strvar, dummy;
   Int_t nrows(0), ncols(0);
   while (!(buf[0]=='#'&& buf[1]=='#')) { // if line starts with ## return
      char* p = buf;
      while(*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512);
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> strvar;
      if (strvar=="signal" || strvar=="background") {
         sstr >> nrows >> dummy >> ncols;
         Int_t matType = (strvar=="signal"?0:1);
         delete fDecorrMatrix[matType];
         TMatrixD* mat = fDecorrMatrix[matType] = new TMatrixD(nrows,ncols);
         // now read all matrix parameters
         for (Int_t row = 0; row<mat->GetNrows(); row++) {
            for (Int_t col = 0; col<mat->GetNcols(); col++) {
               istr >> (*mat)[row][col];
            }
         }
      } // done reading a matrix
      istr.getline(buf,512); // reading the next line
   }
}
