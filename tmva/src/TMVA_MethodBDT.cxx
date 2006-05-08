// @(#)root/tmva $Id: TMVA_MethodBDT.cpp,v 1.17 2006/05/03 08:31:08 helgevoss Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodBDT  (Boosted Decision Trees)                              *
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
//_______________________________________________________________________

#include "TMVA_MethodBDT.h"
#include "TMVA_Tools.h"
#include "TMVA_Timer.h"
#include "Riostream.h"
#include "TRandom.h"
#include <algorithm>
#include "TObjString.h"

#define DEBUG_TMVA_MethodBDT kTRUE

using std::vector;

ClassImp(TMVA_MethodBDT)
 
//_______________________________________________________________________
TMVA_MethodBDT::TMVA_MethodBDT( TString jobName, vector<TString>* theVariables,  
				TTree* theTree, TString theOption, TDirectory* theTargetDir )
  : TMVA_MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  InitBDT();

  if (fOptions.Sizeof()<0) {
    cout << "--- " << GetName() << ": using default options= "<< fOptions <<endl;
  }
  cout << "--- "<<GetName() << " options:" << fOptions <<endl;
  fOptions.ToLower();
  TList*  list  = TMVA_Tools::ParseFormatLine( fOptions );
  if (list->GetSize() > 0){
    fNTrees = atoi( ((TObjString*)list->At(0))->GetString() ) ;
  }
  if (list->GetSize() > 1)fBoostType=((TObjString*)list->At(1))->GetString();
  if (list->GetSize() > 2){
    TString sepType=((TObjString*)list->At(2))->GetString();
    if (sepType.Contains("misclassificationerror")) {
      fSepType = new TMVA_MisClassificationError();
    }
    else if (sepType.Contains("giniindex")) {
      fSepType = new TMVA_GiniIndex();
    }
    else if (sepType.Contains("crossentropy")) {
      fSepType = new TMVA_CrossEntropy();
    }
    else if (sepType.Contains("sdivsqrtsplusb")) {
      fSepType = new TMVA_SdivSqrtSplusB();
    }
    else{
      cout <<"--- TMVA_DecisionTree::TrainNode Error!! separation Routine not found\n" << endl;
      cout << sepType <<endl;
      exit(1);
    }

  }
  else{
    cout <<"---" <<GetName() <<": using default GiniIndex as separation criterion"<<endl;
    fSepType = new TMVA_GiniIndex();
  }
  fMethodName = "BDT"+fSepType->GetName();
  fTestvar    = fTestvarPrefix+GetMethodName();

  if (list->GetSize() > 4){
    fNodeMinEvents = atoi( ((TObjString*)list->At(3))->GetString() ) ;
    fNodeMinSepGain = Double_t(atof( ((TObjString*)list->At(4))->GetString() )) ;
  }
  if (list->GetSize() > 5){
    fNCuts = atoi( ((TObjString*)list->At(5))->GetString() ) ;
  }
  if (list->GetSize() > 6){
    fSignalFraction = atof( ((TObjString*)list->At(6))->GetString() ) ;
  }

  cout << "--- " << GetName() << ": Called with "<<fNTrees <<" trees in the forest"<<endl; 

  cout << "--- " << GetName() << ": Booked with options: "<<endl;
  cout << "--- " << GetName() << ": BoostType: "
       << fBoostType << "   nTress "<< fNTrees<<endl;
  cout << "--- " << GetName() << ": separation criteria in Node training: "
       <<fSepType->GetName()<<endl;
  cout << "--- " << GetName() << ": NodeMinEvents: " << fNodeMinEvents << endl
       << "--- " << GetName() << ": NodeMinSepGain: " << fNodeMinSepGain << endl
       << "--- " << GetName() << ": NCuts        : " << fNCuts         << endl;

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
}

//_______________________________________________________________________
TMVA_MethodBDT::TMVA_MethodBDT( vector<TString> *theVariables, 
				TString theWeightFile,  
				TDirectory* theTargetDir )
  : TMVA_MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
  InitBDT();
}

//_______________________________________________________________________
void TMVA_MethodBDT::InitBDT( void )
{
  fMethodName = "BDT";
  fMethod     = TMVA_Types::BDT;
  fNTrees     = 100;
  fBoostType  = "AdaBoost";
  fNodeMinEvents  = 10;
  fNodeMinSepGain = 0.0002;
  fNCuts          = 20;
  fSignalFraction =-1.;     // -1 means scaling the signal fraction in the is switched off, any
                             // value > 0 would scale the number of background events in the 
                             // training tree by the corresponding number
}

//_______________________________________________________________________
TMVA_MethodBDT::~TMVA_MethodBDT( void )
{
  for (UInt_t i=0; i<fEventSample.size(); i++) delete fEventSample[i];
  for (UInt_t i=0; i<fForest.size(); i++) delete fForest[i];
}


//_______________________________________________________________________
void TMVA_MethodBDT::InitEventSample( void )
{
  // write all Events from the Tree into a vector of TMVA_Events, that are 
  // more easily manipulated 
  // should never be called without existing trainingTree
  if (0 == fTrainingTree) {
    cout << "--- " << GetName() << ": Error in ::Init(): fTrainingTree is zero pointer"
	 << " --> exit(1)" << endl;
    exit(1);
  }
  Int_t nevents = fTrainingTree->GetEntries();
  for (int ievt=0; ievt<nevents; ievt++){
    fEventSample.push_back(new TMVA_Event(fTrainingTree, ievt, fInputVars));
    if (fSignalFraction > 0){
      if (fEventSample.back()->GetType2() < 0) fEventSample.back()->SetWeight(fSignalFraction*fEventSample.back()->GetWeight());
    }
  }
}

//_______________________________________________________________________
void TMVA_MethodBDT::Train( void )
{  
  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    exit(1);
  }

  cout << "--- " << GetName() << ": I will train "<< fNTrees << " Decision Trees"  
       << " ... patience please" << endl;
  TMVA_Timer timer( fNTrees, GetName() ); 
  for (int itree=0; itree<fNTrees; itree++){
    timer.DrawProgressBar( itree );

    fForest.push_back(new TMVA_DecisionTree(fSepType,
      fNodeMinEvents,fNodeMinSepGain,fNCuts));
    fForest.back()->BuildTree(fEventSample);
    this->Boost(fEventSample, fForest.back(), itree);
  }

  // get elapsed time
  cout << "--- " << GetName() << ": elapsed time: " << timer.GetElapsedTime() 
       << endl;    

  // write Weights to file
  WriteWeightsToFile();
}

//_______________________________________________________________________
void TMVA_MethodBDT::Boost( vector<TMVA_Event*> eventSample, TMVA_DecisionTree *dt, Int_t iTree )
{
  if (fOptions.Contains("adaboost")) this->AdaBoost(eventSample, dt);
  else if (fOptions.Contains("epsilonboost")) this->EpsilonBoost(eventSample, dt);
  else if (fOptions.Contains("bagging")) this->Bagging(eventSample, iTree);
  else {
    cout << "--- " << this->GetName() << "::Boost: ERROR Unknow boost option called\n";
    cout << fOptions << endl;
   exit(1);
  }
}

//_______________________________________________________________________
void TMVA_MethodBDT::AdaBoost( vector<TMVA_Event*> eventSample, TMVA_DecisionTree *dt )
{
  fAdaBoostBeta=1.;   // that's apparently the standard value :)
  // in order to perform the boosting, you first have to see how the events of 
  // the original sample were selected with this algorithm... in practice, this is
  // already an information we'd have in the node when they were build, but it's easier
  // right now (.... to be changed later) to simple loop over all the event's again.
  Double_t err=0, sumw=0, sumwfalse=0, count=0;
  vector<Bool_t> correctSelected;
  correctSelected.reserve(eventSample.size());
  for (vector<TMVA_Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
    Int_t evType=dt->CheckEvent(*e);
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
  Double_t newWeight;
  for (vector<TMVA_Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
    if (!correctSelected[i]){
      if (fAdaBoostBeta == 1){
	newWeight = (*e)->GetWeight() * ((1-err)/err) ;
	//	newWeight =  ((1-err)/err) ;
      }else{
	newWeight = (*e)->GetWeight() * pow((1-err)/err,fAdaBoostBeta) ;
	//newWeight =  pow((1-err)/err,fAdaBoostBeta) ;
      }
      (*e)->SetWeight(newWeight);
    }//else (*e)->SetWeight(1.);
    newSumw+=(*e)->GetWeight();    
    i++;
  }
  //re-normalise the Weights
  for (vector<TMVA_Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
    (*e)->SetWeight( (*e)->GetWeight() * sumw / newSumw );
  }
}

//_______________________________________________________________________
void TMVA_MethodBDT::EpsilonBoost( vector<TMVA_Event*>  /*eventSample*/, TMVA_DecisionTree * /*dt*/ ){
  cout << "!!! Sorry...EpsilonBoost is not yet implement \n"; exit(1);
}

//_______________________________________________________________________
void TMVA_MethodBDT::Bagging( vector<TMVA_Event*> eventSample, Int_t iTree )
{
  // call it Bootstrapping, re-sampling or whatever you like, in the end it is nothing
  // else but applying "random Weights" to each event.
  Double_t newSumw=0;
  Double_t newWeight;
  TRandom *trandom   = new TRandom(iTree);
  for (vector<TMVA_Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
    newWeight = trandom->Rndm();
    (*e)->SetWeight(newWeight);
    newSumw+=(*e)->GetWeight();    
  }
  for (vector<TMVA_Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
    (*e)->SetWeight( (*e)->GetWeight() * eventSample.size() / newSumw );
  }
}

//_______________________________________________________________________
void  TMVA_MethodBDT::WriteWeightsToFile( void )
{  
   // write coefficients to file
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
    fout << "-999 *******Tree " << i << endl;
    (fForest[i])->Print(fout);
  }

  fout.close();    
}
  
//_______________________________________________________________________
void  TMVA_MethodBDT::ReadWeightsFromFile( void )
{
   // read coefficients from file
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
  Int_t iTree;
  fin >> var >> var;
  for (int i=0;i<fNTrees;i++){
    fin >> iTree;
    if (iTree != i) {
      cout << "--- "  << ": Error while reading Weight file \n ";
      cout << "--- "  << ": mismatch Itree="<<iTree<<" i="<<i<<endl;
      exit(1);
    }
    TMVA_DecisionTreeNode *n = new TMVA_DecisionTreeNode();
    TMVA_NodeID id;
    n->ReadRec(fin,id);
    fForest.push_back(new TMVA_DecisionTree());
    fForest.back()->SetRoot(n);
  }
  
  fin.close();      
}

//_______________________________________________________________________
Double_t TMVA_MethodBDT::GetMvaValue(TMVA_Event *e)
{
  Double_t myMVA = 0;
  for (UInt_t itree=0; itree<fForest.size(); itree++){
    myMVA += fForest[itree]->CheckEvent(e);
  }
  return myMVA/= Double_t(fForest.size());;
}

//_______________________________________________________________________
void  TMVA_MethodBDT::WriteHistosToFile( void )
{
  cout << "--- " << GetName() << ": write " << GetName() 
       <<" special histos to file: " << fBaseDir->GetPath() << endl;
}
