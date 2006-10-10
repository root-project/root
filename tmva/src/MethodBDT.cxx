// @(#)root/tmva $Id: MethodBDT.cxx,v 1.4 2006/05/26 09:22:13 brun Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodBDT  (Boosted Decision Trees)                             *
 *                                                                                *
 * Description:                                                                   *
 *      Analysis of Boosted Decision Trees                                        *
 *                                                                                *
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

#include "TMVA/MethodBDT.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "Riostream.h"
#include "TRandom.h"
#include <algorithm>
#include "TObjString.h"

using std::vector;

ClassImp(TMVA::MethodBDT)
 
//_______________________________________________________________________
TMVA::MethodBDT::MethodBDT( TString jobName, vector<TString>* theVariables,  
                            TTree* theTree, TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
   // the standard constructor for the "boosted decision trees" 
   //
   // MethodBDT (Boosted Decision Trees) options:
   // format and syntax of option string: "nTrees:BoostType:SeparationType:
   //                                      nEventsMin:dummy:
   //                                      nCuts:SignalFraction"
   // nTrees:          number of trees in the forest to be created
   // BoostType:       the boosting type for the trees in the forest (AdaBoost e.t.c..)
   // SeparationType   the separation criterion applied in the node splitting
   // nEventsMin:      the minimum number of events in a node (leaf criteria, stop splitting)
   // dummy:           a dummy variable, just to keep backward compatible
   // nCuts:  the number of steps in the optimisation of the cut for a node
   // SignalFraction:  scale parameter of the number of Bkg events  
   //                  applied to the training sample to simulate different initial purity
   //                  of your data sample. 
   //
   // known SeparationTypes are:
   //    - MisClassificationError
   //    - GiniIndex
   //    - CrossEntropy
   // known BoostTypes are:
   //    - AdaBoost
   //    - Bagging
   InitBDT();
  
   if (fOptions.Sizeof()<0) {
      cout << "--- " << GetName() << ": using default options= "<< fOptions <<endl;
   }
   cout << "--- "<<GetName() << " options:" << fOptions <<endl;
   fOptions.ToLower();
   TList*  list  = TMVA::Tools::ParseFormatLine( fOptions );
   if (list->GetSize() > 0){
      fNTrees = atoi( ((TObjString*)list->At(0))->GetString() ) ;
   }
   if (list->GetSize() > 1)fBoostType=((TObjString*)list->At(1))->GetString();
   if (list->GetSize() > 2){
      TString sepType=((TObjString*)list->At(2))->GetString();
      if (sepType.Contains("misclassificationerror")) {
         fSepType = new TMVA::MisClassificationError();
      }
      else if (sepType.Contains("giniindex")) {
         fSepType = new TMVA::GiniIndex();
      }
      else if (sepType.Contains("crossentropy")) {
         fSepType = new TMVA::CrossEntropy();
      }
      else if (sepType.Contains("sdivsqrtsplusb")) {
         fSepType = new TMVA::SdivSqrtSplusB();
      }
      else{
         cout <<"--- TMVA::DecisionTree::TrainNode Error!! separation Routine not found\n" << endl;
         cout << sepType <<endl;
         exit(1);
      }

   }
   else{
      cout <<"---" <<GetName() <<": using default GiniIndex as separation criterion"<<endl;
      fSepType = new TMVA::GiniIndex();
   }
   fMethodName = "BDT"+fSepType->GetName();
   fTestvar    = fTestvarPrefix+GetMethodName();

   if (list->GetSize() > 4){
      fNodeMinEvents = atoi( ((TObjString*)list->At(3))->GetString() ) ;
      fDummyOpt      = Double_t(atof( ((TObjString*)list->At(4))->GetString() )) ;
   }
   if (list->GetSize() > 5){
      fNCuts = atoi( ((TObjString*)list->At(5))->GetString() ) ;
   }
   if (list->GetSize() > 6){
      fSignalFraction = atof( ((TObjString*)list->At(6))->GetString() ) ;
   }

   cout << "--- " << GetName() << ": Called with "<<fNTrees <<" trees in the forest"<<endl; 

   cout << "--- " << GetName() << ": Booked with options: "<<endl;
   cout << "--- " << GetName() << ": separation criteria in Node training: "
        << fSepType->GetName()<<endl;
   cout << "--- " << GetName() << ": BoostType: "
        << fBoostType << "   nTress "<< fNTrees<<endl;
   cout << "--- " << GetName() << ": NodeMinEvents:   " << fNodeMinEvents  << endl
        << "--- " << GetName() << ": dummy:  " << fDummyOpt << endl
        << "--- " << GetName() << ": NCuts:           " << fNCuts          << endl
        << "--- " << GetName() << ": SignalFraction:  " << fSignalFraction << endl;

   if (0 != fTrainingTree) {
      if (Verbose())
         cout << "--- " << GetName() << " called " << endl;
      // fill the STL Vector with the event sample 
      this->InitEventSample();
   }
   else{
      cout << "--- " << GetName() << ": Warning: no training Tree given " <<endl;
      cout << "--- " << GetName() << "  you'll not allowed to cal Train e.t.c..."<<endl;
   }

   //book monitoring histograms (currently for AdaBost, only)
   fBoostWeightHist = new TH1F("fBoostWeight","Ada Boost weights",100,1,100);
   fErrFractHist = new TH2F("fErrFractHist","error fraction vs tree number",
                            fNTrees,0,fNTrees,50,0,0.5);
   fMonitorNtuple= new TTree("fMonitorNtuple","BDT variables");
   fMonitorNtuple->Branch("iTree",&fITree,"iTree/I");
   fMonitorNtuple->Branch("boostWeight",&fBoostWeight,"boostWeight/D");
   fMonitorNtuple->Branch("errorFraction",&fErrorFraction,"errorFraction/D");
   fMonitorNtuple->Branch("nNodes",&fNnodes,"nNodes/I");

   delete list;
}

//_______________________________________________________________________
TMVA::MethodBDT::MethodBDT( vector<TString> *theVariables, 
                            TString theWeightFile,  
                            TDirectory* theTargetDir )
   : TMVA::MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
   // constructor for calculating BDT-MVA using previously generatad decision trees
   // the result of the previous training (the decision trees) are read in via the
   // weightfile. Make sure the "theVariables" correspond to the ones used in 
   // creating the "weight"-file
   InitBDT();
}

//_______________________________________________________________________
void TMVA::MethodBDT::InitBDT( void )
{
   // common initialisation with defaults for the BDT-Method
   fMethodName = "BDT";
   fMethod     = TMVA::Types::BDT;
   fNTrees     = 100;
   fBoostType  = "AdaBoost";
   fNodeMinEvents  = 10;
   fDummyOpt = 0.;
   fNCuts          = 20;
   fSignalFraction =-1.;     // -1 means scaling the signal fraction in the is switched off, any
                             // value > 0 would scale the number of background events in the 
                             // training tree by the corresponding number

}

//_______________________________________________________________________
TMVA::MethodBDT::~MethodBDT( void )
{
   //destructor
   for (UInt_t i=0; i<fEventSample.size(); i++) delete fEventSample[i];
   for (UInt_t i=0; i<fForest.size(); i++) delete fForest[i];
}


//_______________________________________________________________________
void TMVA::MethodBDT::InitEventSample( void )
{
   // write all Events from the Tree into a vector of TMVA::Events, that are 
   // more easily manipulated.  
   // This method should never be called without existing trainingTree, as it
   // the vector of events from the ROOT training tree
   if (0 == fTrainingTree) {
      cout << "--- " << GetName() << ": Error in ::Init(): fTrainingTree is zero pointer"
           << " --> exit(1)" << endl;
      exit(1);
   }
   Int_t nevents = fTrainingTree->GetEntries();
   for (int ievt=0; ievt<nevents; ievt++){
      fEventSample.push_back(new TMVA::Event(fTrainingTree, ievt, fInputVars));
      if (fSignalFraction > 0){
         if (fEventSample.back()->GetType2() < 0) fEventSample.back()->SetWeight(fSignalFraction*fEventSample.back()->GetWeight());
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodBDT::Train( void )
{  
   // default sanity checks
   if (!CheckSanity()) { 
      cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
      exit(1);
   }

   cout << "--- " << GetName() << ": I will train "<< fNTrees << " Decision Trees"  
        << " ... patience please" << endl;
   TMVA::Timer timer( fNTrees, GetName() ); 
   for (int itree=0; itree<fNTrees; itree++){
      timer.DrawProgressBar( itree );

      fForest.push_back(new TMVA::DecisionTree(fSepType,
                                               fNodeMinEvents,fNCuts));
      fNnodes = fForest.back()->BuildTree(fEventSample);
      fBoostWeights.push_back( this->Boost(fEventSample, fForest.back(), itree) );
      fITree = itree;

      fMonitorNtuple->Fill();
   }

   // get elapsed time
   cout << "--- " << GetName() << ": elapsed time: " << timer.GetElapsedTime() 
        << endl;    

   // write Weights to file
   WriteWeightsToFile();

   // write monitoring histograms to file
   WriteHistosToFile();
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::Boost( vector<TMVA::Event*> eventSample, TMVA::DecisionTree *dt, Int_t iTree )
{
   // apply the boosting alogrithim (the algorithm is selecte via the the "option" given
   // in the constructor. The return value is the boosting weight 

   if (fOptions.Contains("adaboost")) return this->AdaBoost(eventSample, dt);
   //  else if (fOptions.Contains("epsilonboost")) return this->EpsilonBoost(eventSample, dt);
   else if (fOptions.Contains("bagging")) return this->Bagging(eventSample, iTree);
   else {
      cout << "--- " << this->GetName() << "::Boost: ERROR Unknow boost option called\n";
      cout << fOptions << endl;
      exit(1);
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::AdaBoost( vector<TMVA::Event*> eventSample, TMVA::DecisionTree *dt )
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

   fAdaBoostBeta=1.;   // that's apparently the standard value :)

   Double_t err=0, sumw=0, sumwfalse=0, count=0;
   vector<Bool_t> correctSelected;
   correctSelected.reserve(eventSample.size());
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      Int_t evType= ( 0.5 > dt->CheckEvent(*e) ) ? -1 : 1;
      sumw+=(*e)->GetWeight();

      // I knew I'd get it worng.. 
      // event Type = 0 bkg, 1 sig
      // nodeType:  =-1 bkg  1 sig  
      // if (evType != (*e)->GetType()) { 
      if (evType != (*e)->GetType2()) { 
         sumwfalse+= (*e)->GetWeight();
         count+=1;
         correctSelected.push_back(kFALSE);
      }
      else{
         correctSelected.push_back(kTRUE);
      }    
   }
   err=sumwfalse/sumw;

   Double_t newSumw=0;
   Int_t i=0;
   Double_t boostWeight;
   if (err>0){
      if (fAdaBoostBeta == 1){
         boostWeight = (1-err)/err ;
      }else{
         boostWeight =  pow((1-err)/err,fAdaBoostBeta) ;
      }
   }else{
      boostWeight = 1000; // 
   }

   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
      if (!correctSelected[i]){
         (*e)->SetWeight( (*e)->GetWeight() * boostWeight);
      }
      newSumw+=(*e)->GetWeight();    
      i++;
   }
   //re-normalise the Weights
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
      (*e)->SetWeight( (*e)->GetWeight() * sumw / newSumw );
   }

   fBoostWeightHist->Fill(boostWeight);
   fErrFractHist->Fill(fForest.size(),err);

   fBoostWeight = boostWeight;
   fErrorFraction = err;
  

   return log(boostWeight);
}

//_______________________________________________________________________
// Double_t TMVA::MethodBDT::EpsilonBoost( vector<TMVA::Event*>  /*eventSample*/, TMVA::DecisionTree * /*dt*/ ){
//   cout << "!!! Sorry...EpsilonBoost is not yet implement \n"; exit(1);
//   Double_t boostWeight;
//   return boostWeight;
// }

//_______________________________________________________________________
Double_t TMVA::MethodBDT::Bagging( vector<TMVA::Event*> eventSample, Int_t iTree )
{
   // call it Bootstrapping, re-sampling or whatever you like, in the end it is nothing
   // else but applying "random Weights" to each event.
   Double_t newSumw=0;
   Double_t newWeight;
   TRandom *trandom   = new TRandom(iTree);
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
      newWeight = trandom->Rndm();
      (*e)->SetWeight(newWeight);
      newSumw+=(*e)->GetWeight();    
   }
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
      (*e)->SetWeight( (*e)->GetWeight() * eventSample.size() / newSumw );
   }
   return 1.;  //here as there are random weights for each event, just return a constant==1;
}

//_______________________________________________________________________
void  TMVA::MethodBDT::WriteWeightsToFile( void )
{  
   // write the whole Forest (sample of Decition trees) to a file for later use.
   TString fname = GetWeightFileName();
   cout << "--- " << GetName() << ": creating Weight file: " << fname << endl;
   ofstream fout( fname );
   if (!fout.good( )) { // file not found --> Error
      cout << "--- " << GetName() << ": Error in ::WriteWeightsToFile: "
           << "unable to open output  Weight file: " << fname << endl;
      exit(1);
   }

   // write variable names and min/max 
   // NOTE: the latter values are mandatory for the normalisation 
   // in the reader application !!!
   fout << this->GetMethodName() <<endl;
   fout << "NVars= " << fNvar <<endl; 
   for (Int_t ivar=0; ivar<fNvar; ivar++) {
      TString var = (*fInputVars)[ivar];
      fout << var << "  " << GetXminNorm( var ) << "  " << GetXmaxNorm( var ) << endl;
   }

   // and save the Weights
   fout << "NTrees= " << fForest.size() <<endl; 
   for (UInt_t i=0; i< fForest.size(); i++){
      fout << "-999 *******Tree " << i << "  boostWeight " << fBoostWeights[i] << endl;
      (fForest[i])->Print(fout);
   }

   fout.close();    
}
  
//_______________________________________________________________________
void  TMVA::MethodBDT::ReadWeightsFromFile( void )
{
   // read back the Decicion Trees  from the file
   TString fname = GetWeightFileName();
   cout << "--- " << GetName() << ": reading Weight file: " << fname << endl;
   ifstream fin( fname );

   if (!fin.good( )) { // file not found --> Error
      cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
           << "unable to open input file: " << fname << endl;
      exit(1);
   }

   // read variable names and min/max
   // NOTE: the latter values are mandatory for the normalisation 
   // in the reader application !!!
   TString var, dummy;
   Double_t xmin, xmax;
   fin >> dummy;
   this->SetMethodName(dummy);
   fin >> dummy >> fNvar;
   for (Int_t ivar=0; ivar<fNvar; ivar++) {
      fin >> var >> xmin >> xmax;
      (*fInputVars)[ivar] = var;
      // set min/max
      this->SetXminNorm( ivar, xmin );
      this->SetXmaxNorm( ivar, xmax );
   } 

   // and read the Weights (BDT coefficients)  
   fin >> dummy >> fNTrees;
   cout << "--- " << GetName() << ": Read "<<fNTrees<<" Decision trees\n";
  
   for (UInt_t i=0;i<fForest.size();i++) delete fForest[i];
   fForest.clear();
   fBoostWeights.clear();
   Int_t iTree;
   Double_t boostWeight;
   fin >> var >> var;
   for (int i=0;i<fNTrees;i++){
      fin >> iTree >> dummy >> boostWeight;
      if (iTree != i) {
         cout << "--- "  << ": Error while reading Weight file \n ";
         cout << "--- "  << ": mismatch Itree="<<iTree<<" i="<<i<<endl;
         exit(1);
      }
      TMVA::DecisionTreeNode *n = new TMVA::DecisionTreeNode();
      TMVA::NodeID id;
      n->ReadRec(fin,id);
      fForest.push_back(new TMVA::DecisionTree());
      fForest.back()->SetRoot(n);
      fBoostWeights.push_back(boostWeight);
   }  
   fin.close();      
} 

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetMvaValue(TMVA::Event *e)
{
   //return the MVA value (range [-1;1]) that classifies the
   //event.according to the majority vote from the total number of
   //decision trees
   //In the literature I found that people actually use the 
   //weighted majority vote (using the boost weights) .. However I
   //did not see any improvement in doing so :(  
   // --> this is currently switched off

   const bool useWeightedMajorityVote = kFALSE;

   Double_t myMVA = 0;
   Double_t norm  = 0;
   for (UInt_t itree=0; itree<fForest.size(); itree++){
      //
      if (useWeightedMajorityVote){ 
         myMVA += fBoostWeights[itree] * fForest[itree]->CheckEvent(e);
         norm += fBoostWeights[itree];
      }
      else { 
         myMVA +=  fForest[itree]->CheckEvent(e);
         norm += 1.;
      }
   }
   return myMVA /= Double_t(norm);
}

//_______________________________________________________________________
void  TMVA::MethodBDT::WriteHistosToFile( void )
{
   //here we could write some histograms created during the processing
   //to the output file.
   cout << "--- " << GetName() << ": write " << GetName() 
        <<" special histos to file: " << fBaseDir->GetPath() << endl;
   gDirectory->GetListOfKeys()->Print();
   fLocalTDir = fBaseDir->mkdir(GetName()+GetMethodName());
   fLocalTDir->cd();

   fBoostWeightHist->Write();
   fErrFractHist->Write();
   fMonitorNtuple->Write();
}
