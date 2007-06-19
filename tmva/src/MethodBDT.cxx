// @(#)root/tmva $Id: MethodBDT.cxx,v 1.15 2007/04/21 14:20:46 brun Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBDT (BDT = Boosted Decision Trees)                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Analysis of Boosted Decision Trees                                        *
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

//_______________________________________________________________________
//                                                                      
// Analysis of Boosted Decision Trees                                   
//               
// Boosted decision trees have been successfully used in High Energy 
// Physics analysis for example by the MiniBooNE experiment
// (Yang-Roe-Zhu, physics/0508045). In Boosted Decision Trees, the
// selection is done on a majority vote on the result of several decision
// trees, which are all derived from the same training sample by
// supplying different event weights during the training.
//
// Decision trees: 
//
// successive decision nodes are used to categorize the
// events out of the sample as either signal or background. Each node
// uses only a single discriminating variable to decide if the event is
// signal-like ("goes right") or background-like ("goes left"). This
// forms a tree like structure with "baskets" at the end (leave nodes),
// and an event is classified as either signal or background according to
// whether the basket where it ends up has been classified signal or
// background during the training. Training of a decision tree is the
// process to define the "cut criteria" for each node. The training
// starts with the root node. Here one takes the full training event
// sample and selects the variable and corresponding cut value that gives
// the best separation between signal and background at this stage. Using
// this cut criterion, the sample is then divided into two subsamples, a
// signal-like (right) and a background-like (left) sample. Two new nodes
// are then created for each of the two sub-samples and they are
// constructed using the same mechanism as described for the root
// node. The devision is stopped once a certain node has reached either a
// minimum number of events, or a minimum or maximum signal purity. These
// leave nodes are then called "signal" or "background" if they contain
// more signal respective background events from the training sample.
//
// Boosting: 
//
// the idea behind the boosting is, that signal events from the training
// sample, that end up in a background node (and vice versa) are given a
// larger weight than events that are in the correct leave node. This
// results in a re-weighed training event sample, with which then a new
// decision tree can be developed. The boosting can be applied several
// times (typically 100-500 times) and one ends up with a set of decision
// trees (a forest).
//
// Bagging: 
//
// In this particular variant of the Boosted Decision Trees the boosting
// is not done on the basis of previous training results, but by a simple
// stochasitc re-sampling of the initial training event sample.
//
// Analysis: 
//
// applying an individual decision tree to a test event results in a
// classification of the event as either signal or background. For the
// boosted decision tree selection, an event is successively subjected to
// the whole set of decision trees and depending on how often it is
// classified as signal, a "likelihood" estimator is constructed for the
// event being signal or background. The value of this estimator is the
// one which is then used to select the events from an event sample, and
// the cut value on this estimator defines the efficiency and purity of
// the selection.
//
//_______________________________________________________________________

#include <algorithm>
#include "Riostream.h"
#include "TRandom.h"
#include "TMath.h"
#include "TObjString.h"

#include "TMVA/MethodBDT.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TMVA/Ranking.h"
#include "TMVA/SdivSqrtSplusB.h"
#include "TMVA/BinarySearchTree.h"
#include "TMVA/SeparationBase.h"
#include "TMVA/GiniIndex.h"
#include "TMVA/CrossEntropy.h"
#include "TMVA/MisClassificationError.h"

using std::vector;

ClassImp(TMVA::MethodBDT)
 
//_______________________________________________________________________
TMVA::MethodBDT::MethodBDT( TString jobName, TString methodTitle, DataSet& theData, 
                            TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // the standard constructor for the "boosted decision trees" 
   //
   // MethodBDT (Boosted Decision Trees) options:
   // know options:
   // nTrees=Int_t:    number of trees in the forest to be created
   // BoostType=       the boosting type for the trees in the forest (AdaBoost e.t.c..)
   //                  known: AdaBoost
   //                         Bagging
   // SeparationType   the separation criterion applied in the node splitting
   //                  known: GiniIndex
   //                         MisClassificationError
   //                         CrossEntropy
   //                         SDivSqrtSPlusB
   // nEventsMin:      the minimum number of events in a node (leaf criteria, stop splitting)
   // nCuts:           the number of steps in the optimisation of the cut for a node
   // UseYesNoLeaf     decide if the classification is done simply by the node type, or the S/B
   //                  (from the training) in the leaf node
   // UseWeightedTrees use average classification from the trees, or have the individual trees
   //                  trees in the forest weighted (e.g. log(boostweight) from AdaBoost
   // PruneMethod      The Pruning method: 
   //                  known: NoPruning  // switch off pruning completely
   //                         ExpectedError
   //                         CostComplexity 
   //                         CostComplexity2
   // PruneStrength    a parameter to adjust the amount of pruning. 
   //                  Should be large enouth such that overtraining is avoided");
   InitBDT(); 

   // interpretation of configuration option string
   DeclareOptions();
   ParseOptions();
   ProcessOptions();

   // this initialization is only for the training
   if (HasTrainingTree()) {
      fLogger << kVERBOSE << "Method has been called " << Endl;

      // fill the STL Vector with the event sample 
      this->InitEventSample();
   }
   else {      
      fLogger << kWARNING << "No training Tree given: you will not be allowed to call ::Train etc." << Endl;
   }

   // book monitoring histograms (currently for AdaBost, only)
   BaseDir()->cd();
   fBoostWeightHist = new TH1F("BoostWeight","Ada Boost weights",100,1,100);
   fBoostWeightVsTree = new TH1F("BoostWeightVsTree","Ada Boost weights",fNTrees,0,fNTrees);
   
   fErrFractHist = new TH1F("ErrFractHist","error fraction vs tree number",fNTrees,0,fNTrees);

   fNodesBeforePruningVsTree = new TH1I("NodesBeforePruning","nodes before pruning",fNTrees,0,fNTrees);
   fNodesAfterPruningVsTree = new TH1I("NodesAfterPruning","nodes after pruning",fNTrees,0,fNTrees); 

   fMonitorNtuple= new TTree("MonitorNtuple","BDT variables");
   fMonitorNtuple->Branch("iTree",&fITree,"iTree/I");
   fMonitorNtuple->Branch("boostWeight",&fBoostWeight,"boostWeight/D");
   fMonitorNtuple->Branch("errorFraction",&fErrorFraction,"errorFraction/D");
}

//_______________________________________________________________________
TMVA::MethodBDT::MethodBDT( DataSet& theData, 
                            TString theWeightFile,  
                            TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // constructor for calculating BDT-MVA using previously generatad decision trees
   // the result of the previous training (the decision trees) are read in via the
   // weightfile. Make sure the "theVariables" correspond to the ones used in 
   // creating the "weight"-file
   InitBDT();
  
   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodBDT::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // know options:
   // nTrees=Int_t:    number of trees in the forest to be created
   // BoostType=       the boosting type for the trees in the forest (AdaBoost e.t.c..)
   //                  known: AdaBoost
   //                         Bagging
   // SeparationType   the separation criterion applied in the node splitting
   //                  known: GiniIndex
   //                         MisClassificationError
   //                         CrossEntropy
   //                         SDivSqrtSPlusB
   // nEventsMin:      the minimum number of events in a node (leaf criteria, stop splitting)
   // nCuts:           the number of steps in the optimisation of the cut for a node
   // UseYesNoLeaf     decide if the classification is done simply by the node type, or the S/B
   //                  (from the training) in the leaf node
   // UseWeightedTrees use average classification from the trees, or have the individual trees
   //                  trees in the forest weighted (e.g. log(boostweight) from AdaBoost
   // PruneMethod      The Pruning method: 
   //                  known: NoPruning  // switch off pruning completely
   //                         ExpectedError
   //                         CostComplexity 
   //                         CostComplexity2
   // PruneStrength    a parameter to adjust the amount of pruning. Should be large enouth such that overtraining is avoided");

   DeclareOptionRef(fNTrees, "NTrees", "Number of trees in the forest");
   DeclareOptionRef(fBoostType, "BoostType", "Boosting type for the trees in the forest");
   AddPreDefVal(TString("AdaBoost"));
   AddPreDefVal(TString("Bagging"));
   DeclareOptionRef(fUseYesNoLeaf=kTRUE, "UseYesNoLeaf", 
                    "Use Sig or Bkg node type or the ratio S/B as classification in the leaf node");
   DeclareOptionRef(fUseWeightedTrees=kTRUE, "UseWeightedTrees", 
                    "Use weighted trees or simple average in classification from the forest");
   DeclareOptionRef(fSepTypeS="GiniIndex", "SeparationType", "Separation criterion for node splitting");
   AddPreDefVal(TString("MisClassificationError"));
   AddPreDefVal(TString("GiniIndex"));
   AddPreDefVal(TString("CrossEntropy"));
   AddPreDefVal(TString("SDivSqrtSPlusB"));
   DeclareOptionRef(fNodeMinEvents, "nEventsMin", "Minimum number of events in a leaf node (default: max(20, N_train/(Nvar^2)/10) ) ");
   DeclareOptionRef(fNCuts, "nCuts", "Number of steps during node cut optimisation");
   DeclareOptionRef(fPruneStrength, "PruneStrength", "Pruning strength (negative value == automatic adjustment)");
   DeclareOptionRef(fPruneMethodS, "PruneMethod", "Pruning method: NoPruning (switched off), ExpectedError or CostComplexity");
   AddPreDefVal(TString("NoPruning"));
   AddPreDefVal(TString("ExpectedError"));
   AddPreDefVal(TString("CostComplexity"));
   AddPreDefVal(TString("CostComplexity2"));
}

//_______________________________________________________________________
void TMVA::MethodBDT::ProcessOptions() 
{
   // the option string is decoded, for available options see "DeclareOptions"

   MethodBase::ProcessOptions();

   fSepTypeS.ToLower();
   if      (fSepTypeS == "misclassificationerror") fSepType = new MisClassificationError();
   else if (fSepTypeS == "giniindex")              fSepType = new GiniIndex();
   else if (fSepTypeS == "crossentropy")           fSepType = new CrossEntropy();
   else if (fSepTypeS == "sdivsqrtsplusb")         fSepType = new SdivSqrtSplusB();
   else {
      fLogger << kINFO << GetOptions() << Endl;
      fLogger << kFATAL << "<ProcessOptions> unknown Separation Index option called" << Endl;
   }     

   fPruneMethodS.ToLower();
   if      (fPruneMethodS == "expectederror" )   fPruneMethod = DecisionTree::kExpectedErrorPruning;
   else if (fPruneMethodS == "costcomplexity" )  fPruneMethod = DecisionTree::kCostComplexityPruning;
   else if (fPruneMethodS == "costcomplexity2" ) fPruneMethod = DecisionTree::kMCC;
   else if (fPruneMethodS == "nopruning" )       fPruneMethod = DecisionTree::kNoPruning;
   else {
      fLogger << kINFO << GetOptions() << Endl;
      fLogger << kFATAL << "<ProcessOptions> unknown PruneMethod option called" << Endl;
   }     

   if (fPruneStrength < 0) fAutomatic = kTRUE;
   else fAutomatic = kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodBDT::InitBDT( void )
{
   // common initialisation with defaults for the BDT-Method
   SetMethodName( "BDT" );
   SetMethodType( Types::kBDT );
   SetTestvarName();

   fNTrees         = 200;
   fBoostType      = "AdaBoost";
   fNodeMinEvents  = TMath::Max( 20, int( this->Data().GetNEvtTrain() / this->GetNvar()/ this->GetNvar() / 10) );
   fNCuts          = 20; 
   fPruneMethod    = DecisionTree::kMCC;
   fPruneStrength  = 5;     // means automatic determination of the prune strength using a validation sample  
   fDeltaPruneStrength=0.1;

   // reference cut value to distingiush signal-like from background-like events   
   SetSignalReferenceCut( 0 );
}

//_______________________________________________________________________
TMVA::MethodBDT::~MethodBDT( void )
{
   //destructor
   for (UInt_t i=0; i<fEventSample.size(); i++) delete fEventSample[i];
   for (UInt_t i=0; i<fValidationSample.size(); i++) delete fValidationSample[i];
   for (UInt_t i=0; i<fForest.size(); i++) delete fForest[i];
}

//_______________________________________________________________________
void TMVA::MethodBDT::InitEventSample( void )
{
   // write all Events from the Tree into a vector of Events, that are 
   // more easily manipulated.  
   // This method should never be called without existing trainingTree, as it
   // the vector of events from the ROOT training tree
   if (!HasTrainingTree()) fLogger << kFATAL << "<Init> Data().TrainingTree() is zero pointer" << Endl;

   Int_t nevents = Data().GetNEvtTrain();
   Int_t ievt=0;

   for (; ievt<nevents; ievt++) {

      ReadTrainingEvent(ievt);
      // if fAutomatic you need a validation sample, hence split the training sample into 2

      Event* event = new Event( GetEvent() );
      if (ievt%2 == 0 || !fAutomatic ) fEventSample     .push_back( event );
      else                             fValidationSample.push_back( event );      
   }
   
   fLogger << kINFO << "<InitEventSample> Internally I use " << fEventSample.size() 
           << " for Training  and " << fValidationSample.size() 
           << " for Validation " << Endl;
}

//_______________________________________________________________________
void TMVA::MethodBDT::Train( void )
{  
   // some option, not yet set as "choosable option".. more for internal testing
   Bool_t pruneBeforeBoost = kFALSE;
   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;
   if (IsNormalised()) fLogger << kFATAL << "\"Normalise\" option cannot be used with BDT; " 
                               << "please remove the option from the configuration string, or "
                               << "use \"!Normalise\""
                               << Endl;

   fLogger << kINFO << "Training "<< fNTrees << " Decision Trees ... patience please" << Endl;

   Timer timer( fNTrees, GetName() ); 
   Int_t nNodesBeforePruningCount = 0;
   Int_t nNodesAfterPruningCount = 0;

   Int_t nNodesBeforePruning = 0;
   Int_t nNodesAfterPruning = 0;

   SeparationBase *qualitySepType = new GiniIndex();
   for (int itree=0; itree<fNTrees; itree++) {
      timer.DrawProgressBar( itree );

      fForest.push_back( new DecisionTree( fSepType, fNodeMinEvents, fNCuts, qualitySepType ));
      //use for the training of the next tree only those event, that have together
      //95% of the weight. The others are.. not really important for the tree, but
      //might be large in number ==> ommitting them reduces training ime considerably
      // in order to do this properly, I would have to sort the events according to
      // their weights, but I can get "at least 95%" also by ommitting simply all events
      // for which the weight is less than 0.05
      //
      //       std::vector<Event*> sample;
      //       for (std::vector<Event*>::iterator iev=fEventSample.begin(); 
      //       iev != fEventSample.end(); iev++) {
      //    if ((*iev)->GetWeight() > 0.1) sample.push_back(*iev);
      //       }
      //       nNodesBeforePruning = fForest.back()->BuildTree(sample);

      nNodesBeforePruning = fForest.back()->BuildTree(fEventSample);

      if (itree==1 && fgDebugLevel==1) {
         //plot Cost Complexity versus #Nodes for increasing pruning strengths
         DecisionTree *d = new DecisionTree(*(fForest[itree]));         

         TH1D *h=new TH1D("h","CostComplexity",d->GetNNodes(),0,d->GetNNodes());
         ofstream out1("theOriginal.txt");
         ofstream out2("theCopy.txt");
         fForest[itree]->Print(out1);
         out2 << "************* pruned T " << 1 << " ****************" <<endl;
         d->Print(out2);
         
         Int_t count=1;
         h->SetBinContent(count++,d->GetCostComplexity(fPruneStrength));
         while (d->GetNNodes() > 3) {
            d->FillQualityMap();
            d->FillQualityGainMap();
            
            multimap<Double_t, DecisionTreeNode* > qgm = d->GetQualityGainMap();
            
            multimap<Double_t, DecisionTreeNode* >::iterator it=qgm.begin();
            d->PruneNode(it->second);
            out2 << "************* pruned T " << count << " ****************" <<endl;
            d->Print(out2);
            h->SetBinContent(count++,d->GetCostComplexity(fPruneStrength));
         }
         h->Write();
      }

      if (itree==1 && fgDebugLevel==1) {
         //plot Cost Complexity versus #Nodes for increasing pruning strengths
         DecisionTree *d = new DecisionTree(*(fForest[itree]));         

         TH1D *h=new TH1D("h2","Weakestlink",d->GetNNodes(),0,d->GetNNodes());
         ofstream out2("theCopy2.txt");
         out2 << "************* pruned T " << 1 << " ****************" <<endl;
         d->Print(out2);
         Int_t count=1;
         while (d->GetNNodes() > 3) {
            DecisionTreeNode *n = d->GetWeakestLink();
            multimap<Double_t, DecisionTreeNode* > ls = d->GetLinkStrengthMap();
            multimap<Double_t, DecisionTreeNode* >::iterator it=ls.begin();
            fLogger << kINFO << "Nodes before " << d->CountNodes() << Endl;
            h->SetBinContent(count++,it->first);
            fLogger << kINFO << "Prune Node sequence: " << n->GetSequence() << ", depth:" << n->GetDepth() << Endl;
            d->PruneNode(n);
            fLogger << kINFO << "Nodes after  " << d->CountNodes() << Endl;
            for (it=ls.begin();it!=ls.end();it++) cout << it->first << " / ";
            fLogger << kINFO << Endl;                                      
            out2 << "************* pruned T " << count << " ****************" <<endl;
            d->Print(out2);


         }
         h->Write();
      }


      nNodesBeforePruningCount +=nNodesBeforePruning;
      fNodesBeforePruningVsTree->SetBinContent(itree+1,nNodesBeforePruning);
      if (pruneBeforeBoost && fPruneMethod !=  DecisionTree::kNoPruning) {
         fForest.back()->SetPruneMethod(fPruneMethod);
         fForest.back()->SetPruneStrength(fPruneStrength);
         fForest.back()->PruneTree();
         nNodesAfterPruning = fForest.back()->GetNNodes();
         nNodesAfterPruningCount += nNodesAfterPruning;
         fNodesAfterPruningVsTree->SetBinContent(itree+1,nNodesAfterPruning);
      }
      fBoostWeights.push_back( this->Boost(fEventSample, fForest.back(), itree) );


      fITree = itree;
      fMonitorNtuple->Fill();
   }

   // get elapsed time
   fLogger << kINFO << "<Train> elapsed time: " << timer.GetElapsedTime()    
           << "                              " << Endl;    
   if (!pruneBeforeBoost  && fPruneMethod !=  DecisionTree::kNoPruning) {
      fLogger << kINFO << "Pruning "<< fNTrees << " Decision Trees ... patience please" << Endl;
      Timer timer2( fNTrees, GetName() ); 
      TH1D *alpha = new TH1D("alpha","PruneStrengths",fNTrees,0,fNTrees);
      alpha->SetXTitle("#tree");
      alpha->SetYTitle("PruneStrength");
      for (int itree=0; itree<fNTrees; itree++) {
         timer2.DrawProgressBar( itree );
         fForest[itree]->SetPruneMethod(fPruneMethod);
         if (fAutomatic) {
            fPruneStrength = this->PruneTree(fForest[itree], itree);
         }
         else{
            fForest[itree]->SetPruneStrength(fPruneStrength);
            fForest[itree]->PruneTree();
         }
         nNodesAfterPruning = fForest[itree]->GetNNodes();
         nNodesAfterPruningCount += nNodesAfterPruning;
         fNodesAfterPruningVsTree->SetBinContent(itree+1,nNodesAfterPruning);
         alpha->SetBinContent(itree+1,fPruneStrength);
      }
      alpha->Write();
      fLogger << kINFO << "<Train_Prune> elapsed time: " << timer2.GetElapsedTime()    
              << "                              " << Endl;    
   }
   if (DecisionTree::kNoPruning) {
      fLogger << kINFO << "<Train> average number of nodes (w/o pruning) : "
              << nNodesBeforePruningCount/fNTrees << Endl;
   } 
   else { 
      fLogger << kINFO << "<Train> average number of nodes before/after pruning : " 
              << nNodesBeforePruningCount/fNTrees << " / " 
              << nNodesAfterPruningCount/fNTrees
              << Endl;
   }    
}

//_______________________________________________________________________
Double_t  TMVA::MethodBDT::PruneTree( DecisionTree *dt, Int_t itree)
{
   // prune a tree adjusting the prunestrength using the "test sample" until
   // the best efficiency on the test sample is obtained. In principle the
   // test sample should not be used for that but rather a third validation
   // sample, or the trainng sample with "cross validation". The latter is
   // planned but will come later.

   Double_t alpha = 0;
   Double_t delta = fDeltaPruneStrength;

   DecisionTree*  dcopy;
   vector<Double_t> q;
   multimap<Double_t,Double_t> quality;
   Int_t nnodes=dt->GetNNodes();

   // find the maxiumum prune strength that still leaves some nodes 
   Bool_t forceStop = kFALSE;
   Int_t troubleCount=0, previousNnodes=nnodes;


   nnodes=dt->GetNNodes();
   while (nnodes > 3 && !forceStop) {
      dcopy = new DecisionTree(*dt);
      dcopy->SetPruneStrength(alpha+=delta);
      dcopy->PruneTree();
      q.push_back(this->TestTreeQuality((dcopy)));
      quality.insert(pair<const Double_t,Double_t>(q.back(),alpha));
      nnodes=dcopy->GetNNodes();
      if (previousNnodes == nnodes) troubleCount++;
      else { 
         troubleCount=0; // reset counter
         if (nnodes < previousNnodes / 2 ) fDeltaPruneStrength /= 2.;
      }
      previousNnodes = nnodes;
      if (troubleCount > 20) {
         if (itree == 0 && fPruneStrength <=0) {//maybe you need larger stepsize ??
            fDeltaPruneStrength *= 5;
            fLogger << kINFO << "<PruneTree> trouble determining optimal prune strength"
                    << " for Tree " << itree
                    << " --> first try to increase the step size"
                    << " currently Prunestrenght= " << alpha 
                    << " stepsize " << fDeltaPruneStrength << " " << Endl;
            troubleCount = 0;   // try again
            fPruneStrength = 1; // if it was for the first time.. 
         } 
         else if (itree == 0 && fPruneStrength <=2) {//maybe you need much larger stepsize ??
            fDeltaPruneStrength *= 5;
            fLogger << kINFO << "<PruneTree> trouble determining optimal prune strength"
                    << " for Tree " << itree
                    << " -->  try to increase the step size even more.. "
                    << " if that stitill didn't work, TRY IT BY HAND"  
                    << " currently Prunestrenght= " << alpha 
                    << " stepsize " << fDeltaPruneStrength << " " << Endl;
            troubleCount = 0;   // try again
            fPruneStrength = 3; // if it was for the first time.. 
         }
         else{
            forceStop=kTRUE;
            fLogger << kINFO << "<PruneTree> trouble determining optimal prune strength"
                    << " for Tree " << itree << " at tested prune strength: " << alpha 
                    << " --> abort forced, use same strength as for previous tree:"
                    << fPruneStrength << Endl;
         }
      }
      if (fgDebugLevel==1) fLogger << kINFO << "Pruneed with ("<<alpha
                                   << ") give quality: " << q.back()
                                   << " and #nodes: " << nnodes  
                                   << Endl;
      delete dcopy;
   }
   if (!forceStop) {
      multimap<Double_t,Double_t>::reverse_iterator it=quality.rend();
      it++;
      fPruneStrength = it->second;
      // adjust the step size for the next tree.. think that 20 steps are sort of
      // fine enough.. could become a tunable option later..
      fDeltaPruneStrength *= Double_t(q.size())/20.;
   }
   
   char buffer[10];
   sprintf (buffer,"quad%d",itree);
   TH1D *qual=new TH1D(buffer,"Quality of tree prune steps",q.size(),0.,alpha);
   qual->SetXTitle("PruneStrength");
   qual->SetYTitle("TreeQuality (Purity)");
   for (UInt_t i=0; i< q.size(); i++) {
      qual->SetBinContent(i+1,q[i]);
   }
   qual->Write();
   
   dt->SetPruneStrength(fPruneStrength);
   dt->PruneTree();
   
   return fPruneStrength;
  
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::TestTreeQuality( DecisionTree *dt )
{
   // test the tree quality.. in terms of Miscalssification

   Double_t ncorrect=0, nfalse=0;
   for (UInt_t ievt=0; ievt<fValidationSample.size(); ievt++) {
      Bool_t isSignalType= (dt->CheckEvent(*(fValidationSample[ievt])) > 0.5 ) ? 1 : 0;
      
      if (isSignalType == (fValidationSample[ievt]->IsSignal()) ) {
         ncorrect += fValidationSample[ievt]->GetWeight();
      }
      else{
         nfalse += fValidationSample[ievt]->GetWeight();
      }
   }

   return  ncorrect / (ncorrect + nfalse);
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::Boost( vector<TMVA::Event*> eventSample, DecisionTree *dt, Int_t iTree )
{
   // apply the boosting alogrithim (the algorithm is selecte via the the "option" given
   // in the constructor. The return value is the boosting weight 
  
   if      (fBoostType=="AdaBoost") return this->AdaBoost(eventSample, dt);
   else if (fBoostType=="Bagging")  return this->Bagging(eventSample, iTree);
   else {
      fLogger << kINFO << GetOptions() << Endl;
      fLogger << kFATAL << "<Boost> unknown boost option called" << Endl;
   }

   return -1;
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::AdaBoost( vector<TMVA::Event*> eventSample, DecisionTree *dt )
{
   // the AdaBoost implementation.
   // a new training sample is generated by weighting 
   // events that are misclassified by the decision tree. The weight
   // applied is w = (1-err)/err or more general:
   //            w = ((1-err)/err)^beta
   // where err is the fracthin of misclassified events in the tree ( <0.5 assuming
   // demanding the that previous selection was better than random guessing)
   // and "beta" beeing a free parameter (standard: beta = 1) that modifies the
   // boosting.

   Double_t adaBoostBeta=1.;   // that's apparently the standard value :)

   Double_t err=0, sumw=0, sumwfalse=0, count=0;
   vector<Bool_t> correctSelected;
   correctSelected.reserve(eventSample.size());
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      Bool_t isSignalType = (dt->CheckEvent(*(*e),fUseYesNoLeaf) > 0.5 );
      Double_t w = (*e)->GetWeight();
      sumw += w;

      if (isSignalType == (*e)->IsSignal()) { 
         correctSelected.push_back(kTRUE);
      }
      else {
         sumwfalse+= w;
         count++;
         correctSelected.push_back(kFALSE);
      }    
   }
   err = sumwfalse/sumw;

   Double_t newSumw=0;
   Int_t i=0;
   Double_t boostWeight;
   if (err>0) {
      if (adaBoostBeta == 1) {
         boostWeight = (1-err)/err;
      }
      else {
         boostWeight =  TMath::Power((1.0 - err)/err, adaBoostBeta);
      }
   }
   else {
      boostWeight = 1000; // 
   }

   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      if (!correctSelected[i]) {
         (*e)->SetWeight( (*e)->GetWeight() * boostWeight);
      }
      newSumw+=(*e)->GetWeight();    
      i++;
   }

   // re-normalise the Weights
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      (*e)->SetWeight( (*e)->GetWeight() * sumw / newSumw );
   }

   fBoostWeightHist->Fill(boostWeight);

   fBoostWeightVsTree->SetBinContent(fForest.size(),boostWeight);
   
   fErrFractHist->SetBinContent(fForest.size(),err);

   fBoostWeight = boostWeight;
   fErrorFraction = err;
  
   return TMath::Log(boostWeight);
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::Bagging( vector<TMVA::Event*> eventSample, Int_t iTree )
{
   // call it Bootstrapping, re-sampling or whatever you like, in the end it is nothing
   // else but applying "random Weights" to each event.
   Double_t newSumw=0;
   Double_t newWeight;
   TRandom *trandom   = new TRandom(iTree);
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      newWeight = trandom->Rndm();
      (*e)->SetWeight(newWeight);
      newSumw+=(*e)->GetWeight();    
   }
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      (*e)->SetWeight( (*e)->GetWeight() * eventSample.size() / newSumw );
   }
   return 1.;  //here as there are random weights for each event, just return a constant==1;
}

//_______________________________________________________________________
void TMVA::MethodBDT::WriteWeightsToStream( ostream& o) const
{  
   // and save the Weights
   o << "NTrees= " << fForest.size() <<endl; 
   for (UInt_t i=0; i< fForest.size(); i++) {
      o << "Tree " << i << "  boostWeight " << fBoostWeights[i] << endl;
      (fForest[i])->Print(o);
   }
}
  
//_______________________________________________________________________
void  TMVA::MethodBDT::ReadWeightsFromStream( istream& istr )
{
   // read variable names and min/max
   // NOTE: the latter values are mandatory for the normalisation 
   // in the reader application !!!
   TString var, dummy;

   // and read the Weights (BDT coefficients)  
   istr >> dummy >> fNTrees;
   fLogger << kINFO << "Read " << fNTrees << " Decision trees" << Endl;
  
   for (UInt_t i=0;i<fForest.size();i++) delete fForest[i];
   fForest.clear();
   fBoostWeights.clear();
   Int_t iTree;
   Double_t boostWeight;
   for (int i=0;i<fNTrees;i++) {
      istr >> dummy >> iTree >> dummy >> boostWeight;
      if (iTree != i) {
         fForest.back()->Print( cout );
         fLogger << kFATAL << "Error while reading weight file; mismatch Itree=" 
                 << iTree << " i=" << i 
                 << " dummy " << dummy
                 << " boostweight " << boostWeight 
                 << Endl;
      }

      fForest.push_back( new DecisionTree() );
      fForest.back()->Read(istr);
      fBoostWeights.push_back(boostWeight);
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetMvaValue()
{
   // return the MVA value (range [-1;1]) that classifies the
   // event.according to the majority vote from the total number of
   // decision trees
   // In the literature I found that people actually use the 
   // weighted majority vote (using the boost weights) .. However I
   // did not see any improvement in doing so :(  
   // --> this is currently switched off

   Double_t myMVA = 0;
   Double_t norm  = 0;
   for (UInt_t itree=0; itree<fForest.size(); itree++) {
      //
      if (fUseWeightedTrees) { 
         myMVA += fBoostWeights[itree] * fForest[itree]->CheckEvent(GetEvent(),fUseYesNoLeaf);
         norm  += fBoostWeights[itree];
      }
      else { 
         myMVA += fForest[itree]->CheckEvent(GetEvent(),fUseYesNoLeaf);
         norm  += 1;
      }
   }
   return myMVA /= norm;
}

//_______________________________________________________________________
void  TMVA::MethodBDT::WriteMonitoringHistosToFile( void ) const
{
   // here we could write some histograms created during the processing
   // to the output file.
   fLogger << kINFO << "Write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;
 
   fBoostWeightHist->Write();
   fBoostWeightVsTree->Write();
   fErrFractHist->Write();
   fNodesBeforePruningVsTree->Write();
   fNodesAfterPruningVsTree->Write();
   fMonitorNtuple->Write();
}

// return the individual relative variable importance 
//_______________________________________________________________________
vector< Double_t > TMVA::MethodBDT::GetVariableImportance()
{
   // return the relative variable importance, normalized to all
   // variables together having the importance 1. The importance in
   // evaluated as the total separation-gain that this variable had in
   // the decision trees (weighted by the number of events)
  
   fVariableImportance.resize(GetNvar());
   Double_t  sum=0;
   for (int itree = 0; itree < fNTrees; itree++) {
      vector<Double_t> relativeImportance(fForest[itree]->GetVariableImportance());
      for (UInt_t i=0; i< relativeImportance.size(); i++) {
         fVariableImportance[i] += relativeImportance[i];
      } 
   }   
   for (UInt_t i=0; i< fVariableImportance.size(); i++) sum += fVariableImportance[i];
   for (UInt_t i=0; i< fVariableImportance.size(); i++) fVariableImportance[i] /= sum;

   return fVariableImportance;
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetVariableImportance( UInt_t ivar )
{
   // returns the measure for the variable importance of variable "ivar"
   // which is later used in GetVariableImportance() to calculat the
   // relative variable importances

   vector<Double_t> relativeImportance = this->GetVariableImportance();
   if (ivar < (UInt_t)relativeImportance.size()) return relativeImportance[ivar];
   else fLogger << kFATAL << "<GetVariableImportance> ivar = " << ivar << " is out of range " << Endl;

   return -1;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodBDT::CreateRanking()
{
   // computes ranking of input variables

   // create the ranking object
   fRanking = new Ranking( GetName(), "Variable Importance" );
   vector< Double_t> importance(this->GetVariableImportance());

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fRanking->AddRank( *new Rank( GetInputExp(ivar), importance[ivar] ) );
   }

   return fRanking;
}

//_______________________________________________________________________
void TMVA::MethodBDT::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Short description:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "Boosted Decision Trees are a collection of individual decision" << Endl; 
   fLogger << "trees which form a multivariate classifier by (weighted) majority " << Endl; 
   fLogger << "vote of the individual trees. Consecutive decision trees are  " << Endl;
   fLogger << "trained using the original training data set with re-weighted " << Endl;
   fLogger << "events. By default, the AdaBoost method is employed, which gives " << Endl; 
   fLogger << "events that were misclassified in the previous tree a larger " << Endl;
   fLogger << "weight in the training of the following tree." << Endl; 
   fLogger << Endl; 
   fLogger << "Decision trees are a sequence of binary splits of the data sample" << Endl; 
   fLogger << "using a single descriminant variable at a time. A test event " << Endl;
   fLogger << "ending up after the sequence of left-right splits in a final " << Endl; 
   fLogger << "(\"leaf\") node is classified as either signal or background" << Endl; 
   fLogger << "depending on the majority type of training events in that node." << Endl; 
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance optimisation:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "By the nature of the binary splits performed on the individual" << Endl; 
   fLogger << "variables, decision trees do not deal well with linear correlations" << Endl;
   fLogger << "between variables (they need to approximate the linear split in" << Endl; 
   fLogger << "the two dimensional space by a sequence of splits on the two " << Endl; 
   fLogger << "variables individually). Hence decorrelation could be useful " << Endl; 
   fLogger << "to optimise the BDT performance." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance tuning via configuration options:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "The two most important parameters in the configuration are the  " << Endl; 
   fLogger << "minimal number of events requested by a leaf node (option " << Endl; 
   fLogger << "\"nEventsMin\"). If this number is too large, detailed features " << Endl;
   fLogger << "in the parameter space cannot be modeled. If it is too small, " << Endl; 
   fLogger << "the risk to overtain rises." << Endl;
   fLogger << "   (Imagine the decision tree is split until the leaf node contains" << Endl;
   fLogger << "    only a single event. In such a case, no training event is  " << Endl;
   fLogger << "    misclassified, while the situation will look very different" << Endl;
   fLogger << "    for the test sample.)" << Endl;
   fLogger << Endl;
   fLogger << "The default minumal number is currently set to " << Endl;
   fLogger << "   max(20, (N_training_events / N_variables^2 / 10) " << Endl;
   fLogger << "and can be changed by the user." << Endl; 
   fLogger << Endl;
   fLogger << "The other crucial paramter, the pruning strength (\"PruneStrength\")," << Endl;
   fLogger << "is also related to overtraining. It is a regularistion parameter " << Endl;
   fLogger << "that is used when determining after the training which splits " << Endl;
   fLogger << "are considered statistically insignificant and are removed. The" << Endl;
   fLogger << "user is advised to carefully watch the BDT screen output for" << Endl;
   fLogger << "the comparison between efficiencies obtained on the training and" << Endl;
   fLogger << "the independent test sample. They should be equal within statistical" << Endl;
   fLogger << "errors." << Endl;
}

//_______________________________________________________________________
void TMVA::MethodBDT::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // make ROOT-independent C++ class for classifier response (classifier-specific implementation)

   // write BDT-specific classifier response
   fout << "   std::vector<"<<className<<"_DecisionTreeNode*> fForest;       // i.e. root nodes of decision trees" << endl;
   fout << "   std::vector<double>            fBoostWeights; // the weights applied in the individual boosts" << endl;
   fout << "};" << endl;
   fout << "double " << className << "::GetMvaValue__( const std::vector<double>& inputValues ) const{" << endl;
   fout << "   double myMVA = 0;" << endl;
   fout << "   double norm  = 0;" << endl;
   fout << "for (unsigned int itree=0; itree<fForest.size(); itree++){" << endl;
   fout << "      "<<className<<"_DecisionTreeNode *current = fForest[itree];" << endl;
   fout << "      while (current->GetNodeType() == 0){ //intermediate node" << endl;
   fout << "         if (current->GoesRight(inputValues)) current=("<<className<<"_DecisionTreeNode*)current->GetRight();" << endl;
   fout << "         else current=("<<className<<"_DecisionTreeNode*)current->GetLeft();" << endl;
   fout << "      }" << endl;
   if (fUseWeightedTrees) { 
      if (fUseYesNoLeaf) fout << "      myMVA += fBoostWeights[itree] *  current->GetNodeType();" << endl;
      else               fout << "      myMVA += fBoostWeights[itree] *  current->GetPurity();" << endl;
      fout << "      norm  += fBoostWeights[itree];" << endl;
   }
   else {
      if (fUseYesNoLeaf) fout << "      myMVA += current->GetNodeType();" << endl;
      else               fout << "      myMVA += current->GetPurity();" << endl;
      fout << "      norm  += 1.;" << endl;
   }
   fout << "   }" << endl;
   fout << "   return myMVA /= norm;" << endl;
   fout << "};" << endl;

   fout << "void " << className << "::Initialize(){" << endl;
   fout << " " << endl;

   //Now for each decision tree, write directly the constructors of the nodes in the tree structure 
   for (int itree=0; itree<fNTrees; itree++) {
      fout << "  // itree = " << itree << endl;
      fout << "  fBoostWeights.push_back("<<fBoostWeights[itree]<<");" << endl;
      fout << "  fForest.push_back( " << endl;
      this->MakeClassInstantiateNode((DecisionTreeNode*)fForest[itree]->GetRoot(), fout, className);
      fout <<"   );" << endl;
   }
   fout << "   return;" << endl;
   fout << "};" << endl;
   fout << " " << endl;

   fout << "// Clean up" << endl;
   fout << "inline void " << className << "::Clear() " << endl;
   fout << "{" << endl;
   fout << "   for (unsigned int itree=0; itree<fForest.size(); itree++) { " << endl;
   fout << "      delete fForest[itree]; " << endl;
   fout << "   }" << endl;
   fout << "}" << endl;
} 

//_______________________________________________________________________
void TMVA::MethodBDT::MakeClassSpecificHeader(  std::ostream& fout, const TString& className ) const
{
   // specific class header
   fout << "class "<<className<<"_DecisionTreeNode{" << endl;
   fout << "   " << endl;
   fout << "public:" << endl;
   fout << "   " << endl;
   fout << "   // constructor of an essentially \"empty\" node floating in space" << endl;
   fout << "   "<<className<<"_DecisionTreeNode ( " << endl;
   fout << "                      "<<className<<"_DecisionTreeNode* left," << endl;
   fout << "                      "<<className<<"_DecisionTreeNode* right," << endl;
   fout << "                      double cutValue, bool cutType, int selector," << endl; 
   fout << "                      int nodeType, double purity):" << endl;
   fout << "   fLeft(left)," << endl;
   fout << "   fRight(right)," << endl;
   fout << "   fCutValue(cutValue)," << endl;
   fout << "   fCutType(cutType)," << endl;
   fout << "   fSelector(selector)," << endl;
   fout << "   fNodeType(nodeType)," << endl;
   fout << "   fPurity(purity) {}" << endl;
   fout << "   virtual ~"<<className<<"_DecisionTreeNode();  " << endl;
   fout << "   // test event if it decends the tree at this node to the right" << endl;  
   fout << "   virtual bool GoesRight( const std::vector<double>& inputValues ) const;" << endl;
   fout << "   "<<className<<"_DecisionTreeNode* GetRight( void ) {return fRight; };" << endl;
   fout << "   // test event if it decends the tree at this node to the left " << endl;
   fout << "   virtual bool GoesLeft ( const std::vector<double>& inputValues ) const;" << endl;
   fout << "   "<<className<<"_DecisionTreeNode* GetLeft( void ) {return fLeft; };   " << endl;
   fout << "   //return  S/(S+B) (purity) at this node (from  training)" << endl;
   fout << "   double GetPurity( void ) const {return fPurity;} " << endl;
   fout << "   //return the node type" << endl;
   fout << "   int    GetNodeType( void ) const {return fNodeType;}" << endl;
   fout << "private:" << endl;
   fout << "   "<<className<<"_DecisionTreeNode*   fLeft;    // pointer to the left daughter node" << endl;
   fout << "   "<<className<<"_DecisionTreeNode*   fRight;   // pointer to the right daughter node" << endl;
   fout << "   double              fCutValue;// cut value appplied on this node to discriminate bkg against sig" << endl;
   fout << "   bool                fCutType; // true: if event variable > cutValue ==> signal , false otherwise" << endl;
   fout << "   int                 fSelector;// index of variable used in node selection (decision tree)   " << endl;
   fout << "   int                 fNodeType;// Type of node: -1 == Bkg-leaf, 1 == Signal-leaf, 0 = internal " << endl;
   fout << "   double              fPurity;  // Purity of node from training"<< endl;
   fout << "}; " << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "  "<<className<<"_DecisionTreeNode::~"<<className<<"_DecisionTreeNode(){   " << endl;
   fout << "   if (fLeft != NULL) delete fLeft;" << endl;
   fout << "   if (fRight != NULL) delete fRight;" << endl;
   fout << "}; " << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "bool "<<className<<"_DecisionTreeNode::GoesRight(const  std::vector<double>& inputValues) const{" << endl;
   fout << "   // test event if it decends the tree at this node to the right" << endl;
   fout << "   bool result = (inputValues[fSelector] > fCutValue );" << endl;
   fout << "   if (fCutType == true) return result; //the cuts are selecting Signal ;" << endl;
   fout << "   else return !result;" << endl;
   fout << "}" << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "bool "<<className<<"_DecisionTreeNode::GoesLeft(const  std::vector<double>& inputValues) const{" << endl;
   fout << "   // test event if it decends the tree at this node to the left" << endl;
   fout << "   if (!this->GoesRight(inputValues)) return true;" << endl;
   fout << "   else return false;" << endl;
   fout << "}" << endl;
}

//_______________________________________________________________________
void TMVA::MethodBDT::MakeClassInstantiateNode( DecisionTreeNode *n, std::ostream& fout, const TString& className ) const
{
   // recursively descends a tree and writes the node instance to the output streem
   if (n == NULL) {
      fLogger << kFATAL << "MakeClassInstantiateNode: started with undefined node" <<Endl;
      return ;
   }
   fout << "new "<<className<<"_DecisionTreeNode("<<endl;
   if (n->GetLeft() != NULL){
      this->MakeClassInstantiateNode( (DecisionTreeNode*)n->GetLeft() , fout, className);
   }
   else {
      fout << "NULL";
   }
   fout << ", " <<endl;
   if (n->GetRight() != NULL){
      this->MakeClassInstantiateNode( (DecisionTreeNode*)n->GetRight(), fout, className );
   }
   else {
      fout << "NULL";
   }
   fout << ", " <<  endl
        << n->GetCutValue() << ", " 
        << n->GetCutType() << ", " 
        << n->GetSelector() << ", " 
        << n->GetNodeType() << ", " 
        << n->GetPurity() << ") ";    
}
