// @(#)root/tmva $Id: MethodCommittee.cxx,v 1.6 2006/11/20 15:35:28 brun Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCommittee                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
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
//_______________________________________________________________________

#include "TMVA/MethodCommittee.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "Riostream.h"
#include "TDirectory.h"
#include "TRandom.h"
#include <algorithm>
#include "TObjString.h"
#include "TMVA/Ranking.h"
#include "TMVA/Methods.h"

using std::vector;

ClassImp(TMVA::MethodCommittee)
   ;
 
//_______________________________________________________________________
TMVA::MethodCommittee::MethodCommittee( TString jobName, TString committeeTitle, DataSet& theData, 
                                        TString committeeOptions,
                                        Types::EMVA method, TString methodOptions,
                                        TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, committeeTitle, theData, committeeOptions, theTargetDir ),
     fMemberType( method ),
     fMemberOption( methodOptions )
{
   // constructor
   InitCommittee(); // sets default values

   DeclareOptions();

   ParseOptions();
   
   ProcessOptions();

   // book monitoring histograms (currently for AdaBost, only)
   fBoostFactorHist = new TH1F("fBoostFactor","Ada Boost weights",100,1,100);
   fErrFractHist    = new TH2F("fErrFractHist","error fraction vs tree number",
                               fNMembers,0,fNMembers,50,0,0.5);
   fMonitorNtuple   = new TTree("fMonitorNtuple","Committee variables");
   fMonitorNtuple->Branch("iTree",&fITree,"iTree/I");
   fMonitorNtuple->Branch("boostFactor",&fBoostFactor,"boostFactor/D");
   fMonitorNtuple->Branch("errorFraction",&fErrorFraction,"errorFraction/D");
}

//_______________________________________________________________________
TMVA::MethodCommittee::MethodCommittee( DataSet& theData, 
                                        TString theWeightFile,  
                                        TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // constructor for calculating Committee-MVA using previously generatad decision trees
   // the result of the previous training (the decision trees) are read in via the
   // weightfile. Make sure the "theVariables" correspond to the ones used in 
   // creating the "weight"-file
   InitCommittee();
  
   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodCommittee::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // know options:
   // NMembers           <string>     number of members in the committee
   // UseMemberDecision  <bool>       use signal information from event (otherwise assume signal)
   // UseWeightedMembers <bool>       use weighted trees or simple average in classification from the forest
   //
   // BoostType          <string>     boosting type
   //    available values are:        AdaBoost  <default>
   //                                 Bagging

   DeclareOptionRef(fNMembers, "NMembers", "number of members in the committee");
   DeclareOptionRef(fUseMemberDecision=kFALSE, "UseMemberDecision", "use binary information from IsSignal");
   DeclareOptionRef(fUseWeightedMembers=kTRUE, "UseWeightedMembers", "use weighted trees or simple average in classification from the forest");

   DeclareOptionRef(fBoostType, "BoostType", "boosting type");
   AddPreDefVal(TString("AdaBoost"));
   AddPreDefVal(TString("Bagging"));
}

//_______________________________________________________________________
void TMVA::MethodCommittee::ProcessOptions() 
{
   // process user options
   MethodBase::ProcessOptions();
}

//_______________________________________________________________________
void TMVA::MethodCommittee::InitCommittee( void )
{
   // common initialisation with defaults for the Committee-Method
   SetMethodName( "Committee" );
   SetMethodType( TMVA::Types::kCommittee );
   SetTestvarName();

   fNMembers  = 100;
   fBoostType = "AdaBoost";   

   fCommittee.clear();
   fBoostWeights.clear();
}

//_______________________________________________________________________
TMVA::MethodCommittee::~MethodCommittee( void )
{
   //destructor
   for (UInt_t i=0; i<GetCommittee().size(); i++)   delete fCommittee[i];
   fCommittee.clear();
}

//_______________________________________________________________________
void TMVA::MethodCommittee::WriteStateToFile() const
{ 
   // Function to write options and weights to file

   if (GetWeightFileType()==kTEXT) {

      // get the filename
      TString fname(GetWeightFileName());
      fLogger << kINFO << "creating weight file: " << fname << Endl;

      std::ofstream* fout = new std::ofstream( fname );
      if (!fout->good()) { // file not found --> Error
         fLogger << kFATAL << "<WriteStateToFile> "
                 << "unable to open output  weight file: " << fname << endl;
      }

      WriteStateToStream( *fout );
   }
}

//_______________________________________________________________________
void TMVA::MethodCommittee::Train( void )
{  
   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

   fLogger << kINFO << "will train "<< fNMembers << " committee members ... patience please" << Endl;
   TMVA::Timer timer( fNMembers, GetName() ); 
   for (UInt_t imember=0; imember<fNMembers; imember++){
      timer.DrawProgressBar( imember );

      TMVA::IMethod *method = 0;
      
      // initialize methods
      switch(fMemberType) {
      case TMVA::Types::kCuts:       
         method = new TMVA::MethodCuts           ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kFisher:     
         method = new TMVA::MethodFisher         ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kMLP:        
         method = new TMVA::MethodMLP            ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kTMlpANN:    
         method = new TMVA::MethodTMlpANN        ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kCFMlpANN:   
         method = new TMVA::MethodCFMlpANN       ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kLikelihood: 
         method = new TMVA::MethodLikelihood     ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kHMatrix:    
         method = new TMVA::MethodHMatrix        ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kPDERS:      
         method = new TMVA::MethodPDERS          ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kBDT:        
         method = new TMVA::MethodBDT            ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kSVM:        
         method = new TMVA::MethodSVM            ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kRuleFit:    
         method = new TMVA::MethodRuleFit        ( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      case TMVA::Types::kBayesClassifier:    
         method = new TMVA::MethodBayesClassifier( GetJobName(), GetMethodTitle(), Data(), fMemberOption ); break;
      default:
         fLogger << kFATAL << "method: " << fMemberType << " does not exist" << Endl;
      }
      
      // train each of the member methods
      method->Train();

      GetBoostWeights().push_back( this->Boost( method, imember ) );

      GetCommittee().push_back( method );

      fMonitorNtuple->Fill();
   }

   // get elapsed time
   fLogger << kINFO << "elapsed time: " << timer.GetElapsedTime()    
           << "                              " << Endl;    
}

//_______________________________________________________________________
Double_t TMVA::MethodCommittee::Boost( TMVA::IMethod* method, UInt_t imember )
{
   // apply the boosting alogrithim (the algorithm is selecte via the the "option" given
   // in the constructor. The return value is the boosting weight 
  
   if      (fBoostType=="AdaBoost") return this->AdaBoost( method );
   else if (fBoostType=="Bagging")  return this->Bagging( imember );
   else {
      fLogger << kINFO << GetOptions() << Endl;
      fLogger << kFATAL << "<Boost> unknown boost option called" << Endl;
   }
   return 1.0;
}

//_______________________________________________________________________
Double_t TMVA::MethodCommittee::AdaBoost( TMVA::IMethod* method )
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

   Double_t adaBoostBeta = 1.;   // that's apparently the standard value :)

   // should never be called without existing trainingTree
   if (!HasTrainingTree()) fLogger << kFATAL << "<AdaBoost> Data().TrainingTree() is zero pointer" << Endl;

   // give reference to event
   Event& event = Data().Event();

   Double_t err=0, sumw=0, sumwfalse=0, count=0;
   vector<Bool_t> correctSelected;

   // loop over all events in training tree
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // read the Training Event into "event"
      ReadTrainingEvent(ievt);

      // total sum of event weights
      sumw += event.GetBoostWeight();

      // decide whether it is signal or background-like
      Bool_t isSignalType = method->IsSignalLike();
      
      // to prevent code duplication
      if (isSignalType == event.IsSignal()) correctSelected.push_back( kTRUE );
      else {
         sumwfalse += event.GetBoostWeight();
         count += 1;
         correctSelected.push_back( kFALSE );
      }    
   }

   if (0 == sumw) {
      fLogger << kFATAL << "<AdaBoost> fatal error sum of event boostweights is zero" << Endl;
   }

   // compute the boost factor
   err = sumwfalse/sumw;

   Double_t newSumw=0;
   Int_t i=0;
   Double_t boostFactor = 1;
   if (err>0){
      if (adaBoostBeta == 1){
         boostFactor = (1-err)/err ;
      }
      else {
         boostFactor =  pow((1-err)/err,adaBoostBeta) ;
      }
   }
   else {
      boostFactor = 1000; // default
   }

   // now fill new boostweights
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // read the Training Event into "event"
      ReadTrainingEvent(ievt);

      if (!correctSelected[ievt]) event.SetBoostWeight( event.GetBoostWeight() * boostFactor);

      newSumw += event.GetBoostWeight();    
      i++;
   }

   // re-normalise the boostweights
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {
      event.SetBoostWeight( event.GetBoostWeight() * sumw / newSumw );      
   }

   fBoostFactorHist->Fill(boostFactor);
   fErrFractHist->Fill(GetCommittee().size(),err);

   // save for ntuple
   fBoostFactor   = boostFactor;
   fErrorFraction = err;
  
   // return weight factor for this committee member
   return log(boostFactor);
}

//_______________________________________________________________________
Double_t TMVA::MethodCommittee::Bagging( UInt_t imember )
{
   // call it Bootstrapping, re-sampling or whatever you like, in the end it is nothing
   // else but applying "random boostweights" to each event.
   Double_t newSumw = 0;
   TRandom *trandom   = new TRandom( imember );

   // give reference to event
   Event& event = Data().Event();

   // loop over all events in training tree
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // read the Training Event into "event"
      ReadTrainingEvent(ievt);

      Double_t newWeight = trandom->Rndm();
      event.SetBoostWeight( newWeight );
      newSumw += newWeight;
   }

   // re-normalise the boostweights
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {
      event.SetBoostWeight( event.GetBoostWeight() * Data().GetNEvtTrain() / newSumw );      
   }

   // return weight factor for this committee member
   return 1.0;  // here as there are random weights for each event, just return a constant==1;
}

//_______________________________________________________________________
void TMVA::MethodCommittee::WriteWeightsToStream( ostream& o ) const
{
   // write the state of the method to an output stream
   for (UInt_t imember=0; imember<GetCommittee().size(); imember++) {
      o << endl;
      o << "------------------------------ new member: " << imember << " ---------------" << endl;
      o << "boost weight: " << GetBoostWeights()[imember] << endl;
      GetCommittee()[imember]->WriteStateToStream( o );
   }   
}
  
//_______________________________________________________________________
void  TMVA::MethodCommittee::ReadWeightsFromStream( istream& istr )
{
   // read the state of the method from an input stream

   // explicitly destroy objects in vector
   std::vector<IMethod*>::iterator member = GetCommittee().begin();
   for (; member != GetCommittee().end(); member++) delete *member;

   GetCommittee().clear();
   GetBoostWeights().clear();

   TString  dummy;
   UInt_t   imember;
   Double_t boostWeight;
   
   // loop over all members in committee
   for (UInt_t i=0; i<fNMembers; i++) {
       
      istr >> dummy >> dummy >> dummy >> imember;
      istr >> dummy >> dummy >> boostWeight;

      if (imember != i) {
         fLogger << kFATAL << "<ReadWeightsFromStream> fatal error while reading Weight file \n "
                 << ": mismatch imember: " << imember << " != i: " << i << Endl;
      }

      TMVA::IMethod *method = 0;
      
      // initialize methods
      switch(fMemberType) {
      case TMVA::Types::kCuts:       
         method = new TMVA::MethodCuts           ( Data(), "" ); break;
      case TMVA::Types::kFisher:     
         method = new TMVA::MethodFisher         ( Data(), "" ); break;
      case TMVA::Types::kMLP:        
         method = new TMVA::MethodMLP            ( Data(), "" ); break;
      case TMVA::Types::kTMlpANN:    
         method = new TMVA::MethodTMlpANN        ( Data(), "" ); break;
      case TMVA::Types::kCFMlpANN:   
         method = new TMVA::MethodCFMlpANN       ( Data(), "" ); break;
      case TMVA::Types::kLikelihood: 
         method = new TMVA::MethodLikelihood     ( Data(), "" ); break;
      case TMVA::Types::kHMatrix:    
         method = new TMVA::MethodHMatrix        ( Data(), "" ); break;
      case TMVA::Types::kPDERS:      
         method = new TMVA::MethodPDERS          ( Data(), "" ); break;
      case TMVA::Types::kBDT:        
         method = new TMVA::MethodBDT            ( Data(), "" ); break;
      case TMVA::Types::kSVM:        
         method = new TMVA::MethodSVM            ( Data(), "" ); break;
      case TMVA::Types::kRuleFit:    
         method = new TMVA::MethodRuleFit        ( Data(), "" ); break;
      case TMVA::Types::kBayesClassifier:    
         method = new TMVA::MethodBayesClassifier( Data(), "" ); break;
      default:
         fLogger << kFATAL << "<ReadWeightsFromStream> fatal error: method: " 
                 << fMemberType << " does not exist" << endl;
      }

      // read weight file
      method->ReadStateFromStream(istr);
      GetCommittee().push_back(method);
      GetBoostWeights().push_back(boostWeight);
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodCommittee::GetMvaValue()
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
   for (UInt_t itree=0; itree<GetCommittee().size(); itree++) {

      Double_t tmpMVA = fUseMemberDecision ? ( GetCommittee()[itree]->IsSignalLike() ? 1.0 : -1.0 ) : GetCommittee()[itree]->GetMvaValue();

      if (fUseWeightedMembers){ 
         myMVA += GetBoostWeights()[itree] * tmpMVA;
         norm  += GetBoostWeights()[itree];
      }
      else { 
         myMVA += tmpMVA;
         norm  += 1;
      }
   }
   return (norm != 0) ? myMVA /= Double_t(norm) : -999;
}

//_______________________________________________________________________
void  TMVA::MethodCommittee::WriteMonitoringHistosToFile( void ) const
{
   // here we could write some histograms created during the processing
   // to the output file.
   fLogger << kINFO << "write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;

   BaseDir()->mkdir(GetName()+GetMethodName())->cd();

   fBoostFactorHist->Write();
   fErrFractHist->Write();
   fMonitorNtuple->Write();

   BaseDir()->cd();
}

// return the individual relative variable importance 
//_______________________________________________________________________
vector< Double_t > TMVA::MethodCommittee::GetVariableImportance()
{
   // return the relative variable importance, normalized to all
   // variables together having the importance 1. The importance in
   // evaluated as the total separation-gain that this variable had in
   // the decision trees (weighted by the number of events)
  
   fVariableImportance.resize(GetNvar());
   //    Double_t  sum=0;
   //    for (int itree = 0; itree < fNMembers; itree++){
   //       vector<Double_t> relativeImportance(GetCommittee()[itree]->GetVariableImportance());
   //       for (unsigned int i=0; i< relativeImportance.size(); i++) {
   //          fVariableImportance[i] += relativeImportance[i] ;
   //       } 
   //    }   
   //    for (unsigned int i=0; i< fVariableImportance.size(); i++) sum += fVariableImportance[i];
   //    for (unsigned int i=0; i< fVariableImportance.size(); i++) fVariableImportance[i] /= sum;

   return fVariableImportance;
}

//_______________________________________________________________________
Double_t TMVA::MethodCommittee::GetVariableImportance(UInt_t ivar)
{
   // return the variable importance
   vector<Double_t> relativeImportance = this->GetVariableImportance();
   if (ivar < (UInt_t)relativeImportance.size()) return relativeImportance[ivar];
   else  fLogger << kFATAL << "<GetVariableImportance> ivar = " << ivar << " is out of range " << Endl;

   return -1;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodCommittee::CreateRanking()
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
