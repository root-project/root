// @(#)root/tmva $Id: DataSet.cxx,v 1.8 2007/02/03 06:40:26 brun Exp $
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
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland,                                                        *
 *      MPI-K Heidelberg, Germany ,                                               *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/DataSet.h"
#include "TMVA/Tools.h"
#include "TMVA/MsgLogger.h"
#include "TEventList.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TMatrixF.h"
#include "TVectorF.h"
#include "TMath.h"

#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif
#ifndef ROOT_TMVA_VariableIdentityTransform
#include "TMVA/VariableIdentityTransform.h"
#endif
#ifndef ROOT_TMVA_VariableDecorrTransform
#include "TMVA/VariableDecorrTransform.h"
#endif
#ifndef ROOT_TMVA_VariablePCATransform
#include "TMVA/VariablePCATransform.h"
#endif

namespace TMVA {
   // calculate the largest common divider
   // this function is not happy if numbers are negative!
   Int_t largestCommonDivider(Int_t a, Int_t b) 
   {
      if (a<b) {Int_t tmp = a; a=b; b=tmp; } // achieve a>=b
      if (b==0) return a;
      Int_t fullFits = a/b;
      return largestCommonDivider(b,a-b*fullFits);
   }
}

//_______________________________________________________________________
TMVA::DataSet::DataSet() 
   : fLocalRootDir( 0 ),
     fCut( "" ),
     fMultiCut( "" ),
     fTrainingTree( 0 ),
     fTestTree( 0 ),
     fMultiCutTestTree( 0 ),
     fVerbose( kFALSE ),
     fEvent( 0 ),
     fCurrentTree( 0 ),
     fCurrentEvtIdx( 0 ),
     fWeightExp(""),
     fWeightFormula( 0 ),
     fLogger( GetName(), kINFO )
{
   // constructor

   fDecorrMatrix[0]   = fDecorrMatrix[1] = 0;

   for (Int_t dim1=0; dim1!=Types::kMaxTreeType; dim1++) {
      for (Int_t dim2=0; dim2!=Types::kMaxSBType; dim2++) {
         fDataStats[dim1][dim2]=0;
      }
   }
}

//_______________________________________________________________________
TMVA::DataSet::~DataSet() 
{
   // destructor
}

//_______________________________________________________________________
Bool_t TMVA::DataSet::ReadEvent( TTree* tr, Long64_t evidx ) const 
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
   for (std::vector<TBranch*>::iterator brIt = fEvent->Branches().begin(); 
        brIt!=fEvent->Branches().end();
        brIt++)
     (*brIt)->GetEntry(evidx);

   return kTRUE;
}

//_______________________________________________________________________
TMVA::VariableTransformBase* TMVA::DataSet::FindTransform( Types::EVariableTransform transform ) const
{
   // finds transformation in map
   std::map<Types::EVariableTransform,VariableTransformBase*>::const_iterator tr = fVarTransforms.find( transform );
   if (tr == fVarTransforms.end()) return 0;
   return tr->second;
}

//_______________________________________________________________________
TMVA::VariableTransformBase* TMVA::DataSet::GetTransform( Types::EVariableTransform transform )
{
   // retrieves transformation
   VariableTransformBase* trbase = FindTransform( transform );
   
   if (trbase != 0) return trbase;
   
   // transformation not yet created
   switch (transform) {
   case Types::kNone:
      trbase = new VariableIdentityTransform( GetVariableInfos() );
      break;
   case Types::kDecorrelated:
      trbase = new VariableDecorrTransform( GetVariableInfos() );
      break;
   case Types::kPCA:
      trbase = new VariablePCATransform( GetVariableInfos() );
      break;
   case Types::kMaxVariableTransform:
   default:
      fLogger << kFATAL << "<ProcessOptions> variable transformation method '" 
              << transform << "' unknown." << Endl;
   }

   fLogger << kINFO << "New variable Transformation " 
           << trbase->GetName() << " requested and created." << Endl;

   trbase->SetRootOutputBaseDir(BaseRootDir());

   // here the actual transformation is computed
   trbase->PrepareTransformation( GetTrainingTree() );

   // plot the variables once in this transformation
   trbase->PlotVariables( GetTrainingTree() );

   // print ranking
   trbase->PrintVariableRanking();

   // add transformation to map
   fVarTransforms[transform] = trbase;

   return trbase;
}

//_______________________________________________________________________
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
void TMVA::DataSet::AddSignalTree( TTree* tr, Double_t weight ) 
{
   // add a signal tree to the dataset to be used as input
   // multiple trees is not used at the moment, use chains instead
   fTreeCollection[Types::kSignal].push_back(TreeInfo(tr,weight));
}

//_______________________________________________________________________
void TMVA::DataSet::AddBackgroundTree( TTree* tr, Double_t weight ) 
{
   // add a background tree to the dataset to be used as input
   // multiple trees is not used at the moment, use chains instead
   fTreeCollection[Types::kBackground].push_back(TreeInfo(tr,weight));
}

//_______________________________________________________________________
void TMVA::DataSet::AddVariable( const TString& expression, char varType, void* external ) 
{
   // add a variable (can be a complex expression) to the set of variables used in
   // the MV analysis   
   this->AddVariable( expression, 0, 0, varType, external );
}

//_______________________________________________________________________
void TMVA::DataSet::AddVariable( const TString& expression, Double_t min, Double_t max, char varType, 
                                 void* external ) 
{
   // add a variable (can be a complex expression) to the set of variables used in
   // the MV analysis
   TString regexpr = expression; // remove possible blanks
   regexpr.ReplaceAll(" ", "" );
   fVariables.push_back(VariableInfo( regexpr, fVariables.size()+1, varType, external, min, max ));
   fVariableStrings.push_back( regexpr );
}

//_______________________________________________________________________
Int_t TMVA::DataSet::FindVar(const TString& var) const
{
   // find variable by name
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) 
      if (var == GetInternalVarName(ivar)) return ivar;
   
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) 
      fLogger << kINFO  <<  GetInternalVarName(ivar) << Endl;
   
   fLogger << kFATAL << "<FindVar> variable \'" << var << "\' not found." << Endl;
 
   return -1;
}

//_______________________________________________________________________
void TMVA::DataSet::PrepareForTrainingAndTesting( const TString& splitOpt, TString )
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

   Configurable splitSpecs(splitOpt);
   splitSpecs.SetName("DataSet");

   TString splitMode("Random");
   Int_t nSigTrainEvents(0); // number of signal training events, 0 - all available
   Int_t nBkgTrainEvents(0); // number of backgd training events, 0 - all available
   Int_t nSigTestEvents (0); // number of signal testing events, 0 - all available
   Int_t nBkgTestEvents (0); // number of backgd testing events, 0 - all available

   UInt_t splitSeed(0);

   splitSpecs.DeclareOptionRef( splitMode, "SplitMode",
                                "Method of picking training and testing events (default: random)" );
   splitSpecs.AddPreDefVal(TString("Random"));
   splitSpecs.AddPreDefVal(TString("Alternate"));
   splitSpecs.AddPreDefVal(TString("Block"));

   splitSpecs.DeclareOptionRef( splitSeed=100, "SplitSeed",
                                "Seed for random event shuffling" );

   splitSpecs.DeclareOptionRef( nSigTrainEvents, "NSigTrain",
                                "Number of signal training events (default: 0 - all)" );
   splitSpecs.DeclareOptionRef( nBkgTrainEvents, "NBkgTrain",
                                "Number of background training events (default: 0 - all)" );
   splitSpecs.DeclareOptionRef( nSigTestEvents,  "NSigTest",
                                "Number of signal testing events (default: 0 - all)" );
   splitSpecs.DeclareOptionRef( nBkgTestEvents,  "NBkgTest",
                                "Number of background testing events (default: 0 - all)" );
   
   splitSpecs.DeclareOptionRef( fVerbose, "V", "Verbosity (default: true)" );
   splitSpecs.ParseOptions();

   fLogger << kINFO << "Prepare training and Test samples:" << Endl;
   fLogger << kINFO << "  Signal tree     - total number of events     : " << SignalTreeInfo(0).GetTree()->GetEntries()     << Endl;
   fLogger << kINFO << "  Background tree - total number of events     : " << BackgroundTreeInfo(0).GetTree()->GetEntries() << Endl;

   // loop over signal events first, then over background events
   const char* kindS[2] = { "signal", "background" };

   // apply cuts to the input trees and create TEventLists
   TEventList*  evList[2]; evList[0]=evList[1]=0;
   if (HasCut()) {
      fLogger << kINFO << "  --> Apply cut on input trees                 : " << CutS() << Endl;
      SignalTreeInfo(0).GetTree()->Draw( ">>signalList", Cut(), "goff" );
      evList[Types::kSignal] = (TEventList*)gDirectory->Get("signalList");
      BackgroundTreeInfo(0).GetTree()->Draw( ">>backgroundList", Cut(), "goff" );
      evList[Types::kBackground] = (TEventList*)gDirectory->Get("backgroundList");

      fLogger << kINFO << "  Signal tree    : number of events passing cut: " << evList[Types::kSignal]->GetN()     << Endl;
      fLogger << kINFO << "  Background tree: number of events passing cut: " << evList[Types::kBackground]->GetN() << Endl;
   } 
   else fLogger << kINFO << "No cuts applied" << Endl;

   // ============================================================
   // create training and test tree
   // ============================================================

   Int_t type;            // variable "type" is used to destinguish "0" = background;  "1" = Signal
   Float_t weight;
   Float_t boostweight=1; // variable "boostweight" is 1 by default
   const Long_t basketsize = 128000;
   //BaseRootDir()->cd();

   TTree* tmpTree[2];
   tmpTree[Types::kSignal] = new TTree("SigTT", "Variables used for MVA training");
   tmpTree[Types::kSignal]->Branch( "type",       &type,        "type/I",        basketsize );
   tmpTree[Types::kSignal]->Branch( "weight",     &weight,      "weight/F",      basketsize );
   tmpTree[Types::kSignal]->Branch( "boostweight",&boostweight, "boostweight/F", basketsize );

   UInt_t nvars = GetNVariables();
   Float_t* vArr = new Float_t[nvars];
   Int_t*   iArr = new Int_t[nvars];
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) {
      // add Bbanch to training/test Tree
      const char* myVar = GetInternalVarName(ivar).Data();
      char vt = VarType(ivar);   // the variable type, 'F' or 'I'
      if (vt=='F') {
         tmpTree[Types::kSignal]->Branch( myVar,&vArr[ivar], Form("%s/%c", myVar, vt), basketsize );
      } else if (vt=='I') {
         tmpTree[Types::kSignal]->Branch( myVar,&iArr[ivar], Form("%s/%c", myVar, vt), basketsize );
      } else
         fLogger << kFATAL << "<PrepareForTrainingAndTesting> unknown variable type '" 
                 << vt << "' encountered; allowed are: 'F' or 'I'"
                 << Endl;
   } // end of loop over input variables
   
   tmpTree[Types::kBackground] = (TTree*)tmpTree[Types::kSignal]->CloneTree(0);
   tmpTree[Types::kBackground]->SetName("BkgTT");

   tmpTree[Types::kSignal]    ->SetDirectory(0);
   tmpTree[Types::kBackground]->SetDirectory(0);

   // if we work with chains we need to remember the current tree
   // if the chain jumps to a new tree we have to reset the formulas
   for (Int_t sb=0; sb<2; sb++) { // sb=0 - signal, sb=1 - background

      fLogger << kINFO << "Create training and testing trees: looping over " << kindS[sb] 
              << " events ..." << Endl;

      TString currentFileName="";
      TTree * pCurrentTree  = 0; // used for chains only

      std::vector<TreeInfo>::const_iterator treeIt = fTreeCollection[sb].begin();
      for (;treeIt!=fTreeCollection[sb].end(); treeIt++) {
         TreeInfo currentInfo = *treeIt;
         Bool_t isChain = (TString("TChain") == currentInfo.GetTree()->ClassName());
         type = 1-sb;
         currentInfo.GetTree()->LoadTree(0);
         ChangeToNewTree( currentInfo.GetTree()->GetTree() );
         if (isChain) {
            currentFileName = currentInfo.GetTree()->GetTree()->GetDirectory()->GetFile()->GetName();
            pCurrentTree = currentInfo.GetTree()->GetTree();
         }

         // loop over events in ntuple
         for (Long64_t evtIdx = 0; evtIdx < currentInfo.GetTree()->GetEntries(); evtIdx++) {
            if (evList[sb]!=0 && !evList[sb]->Contains(evtIdx)) continue;

            // survived the cut
            currentInfo.GetTree()->LoadTree(evtIdx);
            if (isChain) {
               if (currentInfo.GetTree()->GetTree()->GetDirectory()->GetFile()->GetName() != currentFileName) {
                  currentFileName = currentInfo.GetTree()->GetTree()->GetDirectory()->GetFile()->GetName();
                  pCurrentTree = currentInfo.GetTree()->GetTree();
                  ChangeToNewTree( pCurrentTree );
               }
            }
            currentInfo.GetTree()->GetEntry(evtIdx);
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
                  fLogger << kERROR << "ERROR while preparing training and testing trees:" << Endl;
                  fLogger << "   multiple array-type expressions of different length were encountered" << Endl;
                  fLogger << "   location of error: event " << evtIdx 
                          << " in tree " << currentInfo.GetTree()->GetName()
                          << " of file " << currentInfo.GetTree()->GetCurrentFile()->GetName() << Endl;
                  fLogger << "   expression " << fInputVarFormulas[ivar]->GetTitle() << " has " 
                          << ndata << " entries, while" << Endl;
                  fLogger << "   expression " << fInputVarFormulas[prevArrExpr]->GetTitle() << " has "
                          << fInputVarFormulas[prevArrExpr]->GetNdata() << " entries" << Endl;
                  fLogger << kFATAL << "Need to abort" << Endl;
               }
            }
            for (Int_t idata = 0;  idata<sizeOfArrays; idata++) {
               Bool_t containsNaN = kFALSE;
               for (UInt_t ivar=0; ivar<nvars; ivar++) {
                  Int_t ndata = fInputVarFormulas[ivar]->GetNdata();
                  if (VarType(ivar)=='F') {
                     vArr[ivar] = (ndata == 1 ? 
                                   fInputVarFormulas[ivar]->EvalInstance(0) : 
                                   fInputVarFormulas[ivar]->EvalInstance(idata));
                     if (TMath::IsNaN(vArr[ivar])) {
                        containsNaN = kTRUE;
                        fLogger << kWARNING << "Expression resolves to infinite value (NaN): " 
                                << fInputVarFormulas[ivar]->GetTitle() << Endl;
                     }
                  } 
                  else if (VarType(ivar)=='I') {
                     Double_t tmpVal = ndata == 1 ? 
                        fInputVarFormulas[ivar]->EvalInstance() : fInputVarFormulas[ivar]->EvalInstance(idata);
                     iArr[ivar] = tmpVal >= 0 ? Int_t(tmpVal+.5) : Int_t(tmpVal-0.5);
                     if (TMath::IsNaN(iArr[ivar])) {
                        containsNaN = kTRUE;
                        fLogger << kWARNING << "Expression resolves to infinite value (NaN): " 
                                << fInputVarFormulas[ivar]->GetTitle() << Endl;
                     }
                  }
               }

               // the weight (can also be an array)
               weight = currentInfo.GetWeight(); // multiply by tree weight
               if (fWeightFormula!=0) {
               
                  Int_t ndata = fWeightFormula->GetNdata();
                  weight *= ndata==1?fWeightFormula->EvalInstance():fWeightFormula->EvalInstance(idata);
                  if (TMath::IsNaN(weight)) {
                     containsNaN = kTRUE;
                     fLogger << kWARNING << "Weight expression resolves to infinite value (NaN): " 
                             << fWeightFormula->GetTitle() << Endl;
                  }
               }

               if (containsNaN) {
                  fLogger << kWARNING << "Event " << evtIdx;
                  if (sizeOfArrays>1) fLogger << kWARNING << "[" << idata << "]";
                  fLogger << " rejected" << Endl;
                  continue;
               }

               tmpTree[sb]->Fill();
            }
         }
      
         currentInfo.GetTree()->ResetBranchAddresses();
      }
   }

   // ============================================================
   // now the training tree needs to be pruned
   // ============================================================

   Long64_t origSize[2];

   for (Int_t sb = 0; sb<2; sb++ ) {
      origSize[sb] = tmpTree[sb]->GetEntries();
   }

   Long64_t finalNEvents[2][2] = { {nSigTrainEvents, nSigTestEvents},
                                   {nBkgTrainEvents, nBkgTestEvents} };

   fLogger << kINFO << "Number of available training events:" << endl
           << "  signal    : " << origSize[Types::kSignal] << endl
           << "  background: " << origSize[Types::kBackground] << Endl;


   for (Int_t sb = 0; sb<2; sb++) { // sb: 0-signal, 1-background

      if (finalNEvents[sb][Types::kTraining]>origSize[sb])
         fLogger << kFATAL << "More training events requested than available for the " << kindS[sb] << Endl;

      if (finalNEvents[sb][Types::kTesting]>origSize[sb])
         fLogger << kFATAL << "More testing events requested than available for the " << kindS[sb] << Endl;

      if (finalNEvents[sb][Types::kTraining] + finalNEvents[sb][Types::kTesting] > origSize[sb])
         fLogger << kFATAL << "More testing and training events requested than available for the " << kindS[sb] << Endl;

      if (finalNEvents[sb][Types::kTraining]==0 || finalNEvents[sb][Types::kTesting]==0) {
         if (finalNEvents[sb][Types::kTraining]==0 && finalNEvents[sb][Types::kTesting]==0) {
            finalNEvents[sb][Types::kTraining] = finalNEvents[sb][Types::kTesting] = origSize[sb]/2;
         } 
         else if (finalNEvents[sb][Types::kTesting]==0) {
            finalNEvents[sb][Types::kTesting]  = origSize[sb] - finalNEvents[sb][Types::kTraining];
         } 
         else {
            finalNEvents[sb][Types::kTraining]  = origSize[sb] - finalNEvents[sb][Types::kTesting];
         }
      }
   }

   for (Int_t j=0;j<2;j++) {
      for (Int_t i=0;i<2;i++) fDataStats[j][i] = finalNEvents[i][j];
      fDataStats[j][Types::kSBBoth] = fDataStats[j][Types::kSignal] + fDataStats[j][Types::kBackground];
   }

   TRandom rndm( splitSeed ); 
   TEventList* evtList[2][2];

   evtList[Types::kSignal]    [Types::kTraining] = new TEventList();
   evtList[Types::kBackground][Types::kTraining] = new TEventList();
   evtList[Types::kSignal]    [Types::kTesting]  = new TEventList();
   evtList[Types::kBackground][Types::kTesting]  = new TEventList();

   for (Int_t sb = 0; sb<2; sb++ ) { // signal, background
      
      const Long64_t size = origSize[sb];

      if (splitMode == "Random") {

         fLogger << kINFO << "Randomly shuffle events in training and testing trees for " << kindS[sb] << Endl;

         // the index array
         Long64_t* idxArray = new Long64_t[size];
         Bool_t*   allPickedIdxArray = new Bool_t[size];
         for (Int_t i=0; i<size; i++) { idxArray[i]=i; allPickedIdxArray[i] = kFALSE; }
         
         for (Int_t itype=Types::kTraining; itype<Types::kMaxTreeType; itype++) {

            // the selected events
            Bool_t* thisPickedIdxArray = new Bool_t[size];
            for (Int_t i=0; i<size; i++) thisPickedIdxArray[i] = kFALSE;
            
            Long64_t pos = 0;
            for (Long64_t i=0; i<finalNEvents[sb][itype]; i++) {
               do { pos = Long64_t(size * rndm.Rndm()); } while (allPickedIdxArray[idxArray[pos]]);
               thisPickedIdxArray[idxArray[pos]] = kTRUE;
               allPickedIdxArray [idxArray[pos]] = kTRUE;
            }
            for (Long64_t i=0; i<size; i++) if (thisPickedIdxArray[i]) evtList[sb][itype]->Enter(i); 

            delete [] thisPickedIdxArray;
         }

         delete [] idxArray;
         delete [] allPickedIdxArray;
      }
      else if (splitMode == "Alternate") {

         fLogger << kINFO << "Pick alternating training and test events from input tree for " << kindS[sb] << Endl;
         
         Int_t ntrain = finalNEvents[sb][Types::kTraining];
         Int_t ntest  = finalNEvents[sb][Types::kTesting];

         Int_t lcd       = largestCommonDivider(ntrain,ntest);
         Int_t trainfrac = ntrain/lcd;
         Int_t modulo    = (ntrain+ntest)/lcd;

         for (Long64_t i=0; i<finalNEvents[sb][Types::kTraining]+finalNEvents[sb][Types::kTesting]; i++) {
            Bool_t isTrainingEvent = (i%modulo)<trainfrac;
            evtList[sb][isTrainingEvent?Types::kTraining:Types::kTesting]->Enter( i );
         }
      }
      else if (splitMode == "Block") {

         fLogger << kINFO << "Pick block-wise training and test events from input tree for " << kindS[sb] << Endl;
         
         for (Long64_t i=0; i<finalNEvents[sb][Types::kTraining]; i++)
            evtList[sb][Types::kTraining]->Enter( i );
         for (Long64_t i=0; i<finalNEvents[sb][Types::kTesting]; i++)
            evtList[sb][Types::kTesting]->Enter( i + finalNEvents[sb][Types::kTraining]);

      }
      else fLogger << kFATAL << "Unknown type: " << splitMode << Endl;
   }

   fLogger << kINFO << "Merge the signal and background trees for training" << Endl;

   gROOT->cd();

   TList lot;
   tmpTree[Types::kSignal]    ->SetEventList(evtList[Types::kSignal]    [Types::kTraining]);
   tmpTree[Types::kBackground]->SetEventList(evtList[Types::kBackground][Types::kTraining]);
   lot.Add( tmpTree[Types::kSignal]    ->CopyTree("") ); 
   lot.Add( tmpTree[Types::kBackground]->CopyTree("") );
   SetTrainingTree(TTree::MergeTrees(&lot));
   lot.SetOwner(); lot.Clear();

   fLogger << kINFO << "Merge the signal and background trees for testing" << Endl;
   tmpTree[Types::kSignal]    ->SetEventList(evtList[Types::kSignal]    [Types::kTesting]);
   tmpTree[Types::kBackground]->SetEventList(evtList[Types::kBackground][Types::kTesting]);
   lot.Add( tmpTree[Types::kSignal]    ->CopyTree("") ); 
   lot.Add( tmpTree[Types::kBackground]->CopyTree("") );
   SetTestTree(TTree::MergeTrees(&lot));
   lot.SetOwner(); lot.Clear();

   delete tmpTree[Types::kSignal];
   delete tmpTree[Types::kBackground];

   GetTrainingTree()->SetName("TrainingTree");
   GetTrainingTree()->SetTitle("Tree used for MVA training");
   GetTrainingTree()->ResetBranchAddresses();

   GetTestTree()->SetName("TestTree");
   GetTestTree()->SetTitle("Tree used for MVA testing");
   GetTestTree()->ResetBranchAddresses();
   
   fLogger << kINFO << "Collected:" << Endl;
   fLogger << kINFO << "  training signal entries     : " << fDataStats[Types::kTraining][Types::kSignal] << Endl;
   fLogger << kINFO << "  training background entries : " << fDataStats[Types::kTraining][Types::kBackground] << Endl;
   fLogger << kINFO << "  testing  signal entries     : " << fDataStats[Types::kTesting][Types::kSignal] << Endl;
   fLogger << kINFO << "  testing  background entries : " << fDataStats[Types::kTesting][Types::kBackground] << Endl;

   // sanity check
   if (GetNEvtSigTrain() <= 0 || GetNEvtBkgdTrain() <= 0 ||
       GetNEvtSigTest() <= 0 || GetNEvtBkgdTest() <= 0) {
      fLogger << kFATAL << "Zero events encountered for training and/or testing in signal and/or "
              << "background sample" << Endl;
   }

   if (Verbose()) {
      GetTrainingTree()->Print();
      GetTestTree()->Print();
      GetTrainingTree()->Show(0);
   }

   BaseRootDir()->cd();

   // print overall correlation matrix between all variables in tree  
   PrintCorrelationMatrix( GetTrainingTree() );

   GetTransform( Types::kNone );
   
   BaseRootDir()->mkdir("input_expressions")->cd();
   for (UInt_t ivar=0; ivar<nvars; ivar++) fInputVarFormulas[ivar]->Write();
   BaseRootDir()->cd();

   // sanity check (should be removed from all methods!)
   if (UInt_t(GetTrainingTree()->GetListOfBranches()->GetEntries() - 3) != GetNVariables()) 
      fLogger << kFATAL << "<PrepareForTrainingAndTesting> mismatch in number of variables" << Endl;

   // reset the Event references
   ResetCurrentTree();

   delete [] vArr;
   delete [] iArr;
}

//_______________________________________________________________________
void TMVA::DataSet::ChangeToNewTree( TTree* tr )
{ 
   // While the data gets copied into the local training and testing
   // trees, the input tree can change (for intance when changing from
   // signal to background tree, or using TChains as input) The
   // TTreeFormulas, that hold the input expressions need to be
   // reassociated with the new tree, which is done here

   tr->SetBranchStatus("*",1);

   vector<TTreeFormula*>::const_iterator varFIt = fInputVarFormulas.begin();
   for (;varFIt!=fInputVarFormulas.end();varFIt++) delete *varFIt;
   fInputVarFormulas.clear();
   for (UInt_t i=0; i<GetNVariables(); i++) {
      TTreeFormula* ttf = new TTreeFormula( Form( "Formula%s", GetInternalVarName(i).Data() ),
                                            GetExpression(i).Data(), tr );
      fInputVarFormulas.push_back( ttf );
      // sanity check
      if (ttf->GetNcodes() == 0) {
         fLogger << kFATAL << "Expression: " << GetExpression(i) 
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

   tr->SetBranchStatus("*",0);

   for (varFIt = fInputVarFormulas.begin(); varFIt!=fInputVarFormulas.end(); varFIt++) {
      TTreeFormula * ttf = *varFIt;
      for(Int_t bi = 0; bi<ttf->GetNcodes(); bi++)
         tr->SetBranchStatus(ttf->GetLeaf(bi)->GetName(), 1);
   }

   if(fWeightFormula!=0)
      for(Int_t bi = 0; bi<fWeightFormula->GetNcodes(); bi++)
         tr->SetBranchStatus(fWeightFormula->GetLeaf(bi)->GetName(), 1);

   return;
}

//_______________________________________________________________________
void TMVA::DataSet::PrintCorrelationMatrix( TTree* theTree )
{ 
   // calculates the correlation matrices for signal and background, 
   // prints them to standard output, and fills 2D histograms

   // first remove type from variable set
   fLogger << kINFO << "Compute correlation matrices on tree: " 
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
   fLogger << kINFO << "Correlation matrix (signal):" << Endl;
   TMVA::Tools::FormattedOutput( *corrMatS, *theVars, fLogger );
   fLogger << Endl;

   fLogger << kINFO << "Correlation matrix (background):" << Endl;
   TMVA::Tools::FormattedOutput( *corrMatB, *theVars, fLogger );
   fLogger << Endl;

   // ---- histogramming
   LocalRootDir()->cd();

   // loop over signal and background
   TString hName[2]  = { "CorrelationMatrixS", "CorrelationMatrixB" };
   TString hTitle[2] = { "Correlation Matrix (signal)", "Correlation Matrix (background)" };

   // workaround till the TMatrix templates are comonly used
   // this keeps backward compatibility
   TMatrixF* tmS = new TMatrixF( nvar, nvar );
   TMatrixF* tmB = new TMatrixF( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      for (Int_t jvar=0; jvar<nvar; jvar++) {
         (*tmS)(ivar, jvar) = (*corrMatS)(ivar,jvar);
         (*tmB)(ivar, jvar) = (*corrMatB)(ivar,jvar);
      }
   }  

   TMatrixF* mObj[2] = { tmS, tmB };

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
      fLogger << kVERBOSE << "Created correlation matrix as 2D histogram: " << h2->GetName() << Endl;
      
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
                       << "(" << ivar << ", " << jvar << ") = " << d                   
                       << Endl;
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
               xmin(ivar) = GetEvent().GetVal(ivar);
               xmax(ivar) = GetEvent().GetVal(ivar);
            }
            else {
               xmin(ivar) = TMath::Min( xmin(ivar), GetEvent().GetVal(ivar) );
               xmax(ivar) = TMath::Max( xmax(ivar), GetEvent().GetVal(ivar) );
            }
         }
      }
   }
   
   // perform event loop
   Double_t ic = 0;
   for (Int_t i=0; i<GetTrainingTree()->GetEntries(); i++) {

      // fill the event
      ReadTrainingEvent(i);

      if (GetEvent().IsSignal() == isSignal) {

         Double_t weight = GetEvent().GetWeight();
         ic += weight; // count used events

         for (ivar=0; ivar<nvar; ivar++) {

            Double_t xi = ( (norm) ? Tools::NormVariable( GetEvent().GetVal(ivar), xmin(ivar), xmax(ivar) )
                            : GetEvent().GetVal(ivar) );
            vec(ivar) += xi*weight;
            mat2(ivar, ivar) += (xi*xi*weight);

            for (jvar=ivar+1; jvar<nvar; jvar++) {
               Double_t xj =  ( (norm) ? Tools::NormVariable( GetEvent().GetVal(jvar), xmin(ivar), xmax(ivar) )
                                : GetEvent().GetVal(jvar) );
               mat2(ivar, jvar) += (xi*xj*weight);
               mat2(jvar, ivar) = mat2(ivar, jvar); // symmetric matrix
            }
         }
      }
   }

   // variance-covariance
   for (ivar=0; ivar<nvar; ivar++) {
      for (jvar=0; jvar<nvar; jvar++) {
         (*mat)(ivar, jvar) = mat2(ivar, jvar)/ic - vec(ivar)*vec(jvar)/(ic*ic);
      }
   }
}

