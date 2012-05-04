// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

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
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

#ifndef ROOT_TMVA_DataSetInfo
#include "TMVA/DataSetInfo.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_ResultsRegression
#include "TMVA/ResultsRegression.h"
#endif
#ifndef ROOT_TMVA_ResultsClassification
#include "TMVA/ResultsClassification.h"
#endif
#ifndef ROOT_TMVA_ResultsMulticlass
#include "TMVA/ResultsMulticlass.h"
#endif
#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif

//_______________________________________________________________________
TMVA::DataSet::DataSet(const DataSetInfo& dsi) 
   : fdsi(dsi),
     fEventCollection(4,(std::vector<Event*>*)0),
     fCurrentTreeIdx(0),
     fCurrentEventIdx(0),
     fHasNegativeEventWeights(kFALSE),
     fLogger( new MsgLogger(TString(TString("Dataset:")+dsi.GetName()).Data()) ),
     fTrainingBlockSize(0)
{
   // constructor
   for (UInt_t i=0; i<4; i++) fEventCollection[i] = new std::vector<Event*>();
   
   fClassEvents.resize(4);
   fBlockBelongToTraining.reserve(10);
   fBlockBelongToTraining.push_back(kTRUE);

   // sampling
   fSamplingRandom = 0;

   Int_t treeNum = 2;
   fSampling.resize( treeNum );  
   fSamplingNEvents.resize( treeNum ); 
   fSamplingWeight.resize(treeNum);
  
   for (Int_t treeIdx = 0; treeIdx < treeNum; treeIdx++) {
      fSampling.at(treeIdx) = kFALSE;
      fSamplingNEvents.at(treeIdx) = 0;
      fSamplingWeight.at(treeIdx) = 1.0;
   }
}

//_______________________________________________________________________
TMVA::DataSet::~DataSet() 
{
   // destructor

   // delete event collection
   Bool_t deleteEvents=true; // dataset owns the events /JS
   DestroyCollection( Types::kTraining, deleteEvents );
   DestroyCollection( Types::kTesting, deleteEvents );
   
   fBlockBelongToTraining.clear();
   // delete results
   for (std::vector< std::map< TString, Results* > >::iterator it = fResults.begin(); it != fResults.end(); it++) {
      for (std::map< TString, Results* >::iterator itMap = (*it).begin(); itMap != (*it).end(); itMap++) {
         delete itMap->second;
      }
   }

   // delete sampling
   if (fSamplingRandom != 0 ) delete fSamplingRandom;

   std::vector< std::pair< Float_t, Long64_t >* >::iterator itEv;
   std::vector< std::vector<std::pair< Float_t, Long64_t >* > >::iterator treeIt;
   for (treeIt = fSamplingEventList.begin(); treeIt != fSamplingEventList.end(); treeIt++ ) {
      for (itEv = (*treeIt).begin(); itEv != (*treeIt).end(); itEv++) {
         delete (*itEv);
      }
   }

   // need also to delete fEventCollections[2] and [3], not sure if they are used
   DestroyCollection( Types::kValidation, deleteEvents );
   DestroyCollection( Types::kTrainingOriginal, deleteEvents );

   delete fLogger;
}

//_______________________________________________________________________
void TMVA::DataSet::IncrementNClassEvents( Int_t type, UInt_t classNumber ) 
{
   if (fClassEvents.size()<(UInt_t)(type+1)) fClassEvents.resize( type+1 );
   if (fClassEvents.at( type ).size() < classNumber+1) fClassEvents.at( type ).resize( classNumber+1 );
   fClassEvents.at( type ).at( classNumber ) += 1;
}

//_______________________________________________________________________
void TMVA::DataSet::ClearNClassEvents( Int_t type ) 
{
   if (fClassEvents.size()<(UInt_t)(type+1)) fClassEvents.resize( type+1 );
   fClassEvents.at( type ).clear();
}

//_______________________________________________________________________
Long64_t TMVA::DataSet::GetNClassEvents( Int_t type, UInt_t classNumber ) 
{
   try {
      return fClassEvents.at(type).at(classNumber);
   } 
   catch (std::out_of_range excpt) {
      ClassInfo* ci = fdsi.GetClassInfo( classNumber );
      Log() << kFATAL << "No " << (type==0?"training":(type==1?"testing":"_unknown_type_")) 
            << " events for class " << (ci==NULL?"_no_name_known_":ci->GetName().Data()) << " (index # "<<classNumber<<")"
            << " available. Check if all class names are spelled correctly and if events are" 
            << " passing the selection cuts." << Endl;
   } 
   catch (...) {
      Log() << kFATAL << "ERROR/CAUGHT : DataSet/GetNClassEvents, .. unknown error" << Endl;
   }
   return 0;
}

//_______________________________________________________________________
void TMVA::DataSet::DestroyCollection(Types::ETreeType type, Bool_t deleteEvents )
{
   // destroys the event collection (events + vector)
   UInt_t i = TreeIndex(type);
   if (i>=fEventCollection.size() || fEventCollection[i]==0) return;
   if (deleteEvents) {
      for (UInt_t j=0; j<fEventCollection[i]->size(); j++) delete (*fEventCollection[i])[j];
   }
   delete fEventCollection[i];
   fEventCollection[i]=0;
}

//_______________________________________________________________________
TMVA::Event* TMVA::DataSet::GetEvent() const
{
   if (fSampling.size() > UInt_t(fCurrentTreeIdx) && fSampling.at(fCurrentTreeIdx)) {
      Long64_t iEvt = fSamplingSelected.at(fCurrentTreeIdx).at( fCurrentEventIdx )->second;
      return (*(fEventCollection.at(fCurrentTreeIdx))).at(iEvt);
   }
   else {
      return (*(fEventCollection.at(fCurrentTreeIdx))).at(fCurrentEventIdx);
   }
}

//_______________________________________________________________________
UInt_t TMVA::DataSet::GetNVariables() const 
{
   // access the number of variables through the datasetinfo
   return fdsi.GetNVariables();
}

//_______________________________________________________________________
UInt_t TMVA::DataSet::GetNTargets() const 
{
   // access the number of targets through the datasetinfo
   return fdsi.GetNTargets();
}

//_______________________________________________________________________
UInt_t TMVA::DataSet::GetNSpectators() const 
{
   // access the number of targets through the datasetinfo
   return fdsi.GetNSpectators();
}

//_______________________________________________________________________
void TMVA::DataSet::AddEvent(Event * ev, Types::ETreeType type) 
{
   // add event to event list
   // after which the event is owned by the dataset
   fEventCollection.at(Int_t(type))->push_back(ev);
   if (ev->GetWeight()<0) fHasNegativeEventWeights = kTRUE;
   fEvtCollIt=fEventCollection.at(fCurrentTreeIdx)->begin();
}

//_______________________________________________________________________
void TMVA::DataSet::SetEventCollection(std::vector<TMVA::Event*>* events, Types::ETreeType type) 
{
   // Sets the event collection (by DataSetFactory)
   Bool_t deleteEvents = true;
   DestroyCollection(type,deleteEvents);

   const Int_t t = TreeIndex(type);
   ClearNClassEvents( type );
   fEventCollection.at(t) = events;
   for (std::vector<Event*>::iterator it = fEventCollection.at(t)->begin(); it < fEventCollection.at(t)->end(); it++) {
      IncrementNClassEvents( t, (*it)->GetClass() );
   }
   fEvtCollIt=fEventCollection.at(fCurrentTreeIdx)->begin();
}

//_______________________________________________________________________
TMVA::Results* TMVA::DataSet::GetResults( const TString & resultsName,
                                          Types::ETreeType type,
                                          Types::EAnalysisType analysistype ) 
{
   //    TString info(resultsName+"/");
   //    switch(type) {
   //    case Types::kTraining: info += "kTraining/";  break;
   //    case Types::kTesting:  info += "kTesting/";   break;
   //    default: break;
   //    }
   //    switch(analysistype) {
   //    case Types::kClassification: info += "kClassification";  break;
   //    case Types::kRegression:     info += "kRegression";      break;
   //    case Types::kNoAnalysisType: info += "kNoAnalysisType";  break;
   //    case Types::kMaxAnalysisType:info += "kMaxAnalysisType"; break;
   //    }

   UInt_t t = TreeIndex(type);
   if (t<fResults.size()) {
      const std::map< TString, Results* >& resultsForType = fResults[t];
      std::map< TString, Results* >::const_iterator it = resultsForType.find(resultsName);
      if (it!=resultsForType.end()) {
         //Log() << kINFO << " GetResults("<<info<<") returns existing result." << Endl;
         return it->second;
      }
   }
   else {
      fResults.resize(t+1);
   }

   // nothing found

   Results * newresults = 0;
   switch(analysistype) {
   case Types::kClassification:
      newresults = new ResultsClassification(&fdsi);
      break;
   case Types::kRegression:
      newresults = new ResultsRegression(&fdsi);
      break;
   case Types::kMulticlass:
      newresults = new ResultsMulticlass(&fdsi);
      break;
   case Types::kNoAnalysisType:
      newresults = new ResultsClassification(&fdsi);
      break;
   case Types::kMaxAnalysisType:
      //Log() << kINFO << " GetResults("<<info<<") can't create new one." << Endl;
      return 0;
      break;
   }

   newresults->SetTreeType( type );
   fResults[t][resultsName] = newresults;

   //Log() << kINFO << " GetResults("<<info<<") builds new result." << Endl;
   return newresults;
}
//_______________________________________________________________________
void TMVA::DataSet::DeleteResults( const TString & resultsName,
                                   Types::ETreeType type,
                                   Types::EAnalysisType /* analysistype */ ) 
{
   // delete the results stored for this particulary 
   //      Method instance  (here appareantly called resultsName instead of MethodTitle
   //      Tree type (Training, testing etc..)
   //      Analysis Type (Classification, Multiclass, Regression etc..)

   if (fResults.empty()) return;

   if (UInt_t(type) > fResults.size()){
      Log()<<kFATAL<< "you asked for an Treetype (training/testing/...)"
           << " whose index " << type << " does not exist " << Endl;
   }
   std::map< TString, Results* >& resultsForType = fResults[UInt_t(type)];
   std::map< TString, Results* >::iterator it = resultsForType.find(resultsName);
   if (it!=resultsForType.end()) {
      Log() << kDEBUG << " Delete Results previous existing result:" << resultsName 
            << " of type " << type << Endl;
      delete it->second;
      resultsForType.erase(it->first);
   }
   else {
      Log() << kINFO << "could not fine Result class of " << resultsName 
            << " of type " << type << " which I should have deleted" << Endl;
   }
}
//_______________________________________________________________________
void TMVA::DataSet::DivideTrainingSet( UInt_t blockNum )
{
   // divide training set
   Int_t tOrg = TreeIndex(Types::kTrainingOriginal),tTrn = TreeIndex(Types::kTraining);
   // not changing anything ??
   if (fBlockBelongToTraining.size() == blockNum) return;
   // storing the original training vector
   if (fBlockBelongToTraining.size() == 1) {
      if (fEventCollection[tOrg] == 0)
         fEventCollection[tOrg]=new std::vector<TMVA::Event*>(fEventCollection[tTrn]->size());
      fEventCollection[tOrg]->clear();
      for (UInt_t i=0; i<fEventCollection[tTrn]->size(); i++)
         fEventCollection[tOrg]->push_back((*fEventCollection[tTrn])[i]);
      fClassEvents[tOrg] = fClassEvents[tTrn];
   }
   //reseting the event division vector
   fBlockBelongToTraining.clear();
   for (UInt_t i=0 ; i < blockNum ; i++) fBlockBelongToTraining.push_back(kTRUE);

   ApplyTrainingSetDivision();
}

//_______________________________________________________________________
void TMVA::DataSet::ApplyTrainingSetDivision()
{
   // apply division of data set
   Int_t tOrg = TreeIndex(Types::kTrainingOriginal), tTrn = TreeIndex(Types::kTraining), tVld = TreeIndex(Types::kValidation);
   fEventCollection[tTrn]->clear();
   if (fEventCollection[tVld]==0)
      fEventCollection[tVld] = new std::vector<TMVA::Event*>(fEventCollection[tOrg]->size());
   fEventCollection[tVld]->clear();

   //creating the new events collections, notice that the events that can't be evenly divided belong to the last event
   for (UInt_t i=0; i<fEventCollection[tOrg]->size(); i++) {
      if (fBlockBelongToTraining[i % fBlockBelongToTraining.size()])
         fEventCollection[tTrn]->push_back((*fEventCollection[tOrg])[i]);
      else
         fEventCollection[tVld]->push_back((*fEventCollection[tOrg])[i]);
   }
}

//_______________________________________________________________________
void TMVA::DataSet::MoveTrainingBlock( Int_t blockInd,Types::ETreeType dest, Bool_t applyChanges )
{
   // move training block 
   if (dest == Types::kValidation)
      fBlockBelongToTraining[blockInd]=kFALSE;
   else
      fBlockBelongToTraining[blockInd]=kTRUE;
   if (applyChanges) ApplyTrainingSetDivision();
}

//_______________________________________________________________________
Long64_t TMVA::DataSet::GetNEvtSigTest()   
{ 
   // return number of signal test events in dataset
   return GetNClassEvents(Types::kTesting, fdsi.GetClassInfo("Signal")->GetNumber() ); 
}

//_______________________________________________________________________
Long64_t TMVA::DataSet::GetNEvtBkgdTest()  
{ 
   // return number of background test events in dataset
   return GetNClassEvents(Types::kTesting, fdsi.GetClassInfo("Background")->GetNumber() ); 
}

//_______________________________________________________________________
Long64_t TMVA::DataSet::GetNEvtSigTrain()  
{ 
   // return number of signal training events in dataset
   return GetNClassEvents(Types::kTraining, fdsi.GetClassInfo("Signal")->GetNumber() ); 
}

//_______________________________________________________________________
Long64_t TMVA::DataSet::GetNEvtBkgdTrain() 
{ 
   // return number of background training events in dataset
   return GetNClassEvents(Types::kTraining, fdsi.GetClassInfo("Background")->GetNumber() ); 
}

//_______________________________________________________________________
void TMVA::DataSet::InitSampling( Float_t fraction, Float_t weight, UInt_t seed  )
{
   // initialize random or importance sampling

   // add a random generator if not yet present
   if (fSamplingRandom == 0 ) fSamplingRandom = new TRandom3( seed );

   // first, clear the lists
   std::vector< std::pair< Float_t, Long64_t >* > evtList;
   std::vector< std::pair< Float_t, Long64_t >* >::iterator it;

   Int_t treeIdx = TreeIndex( GetCurrentType() );

   if (fSamplingEventList.size() < UInt_t(treeIdx+1) ) fSamplingEventList.resize(treeIdx+1);
   if (fSamplingSelected.size() < UInt_t(treeIdx+1) ) fSamplingSelected.resize(treeIdx+1);
   for (it = fSamplingEventList.at(treeIdx).begin(); it != fSamplingEventList.at(treeIdx).end(); it++ ) delete (*it);
   fSamplingEventList.at(treeIdx).clear();
   fSamplingSelected.at(treeIdx).clear();

   if (fSampling.size() < UInt_t(treeIdx+1) )         fSampling.resize(treeIdx+1);
   if (fSamplingNEvents.size() < UInt_t(treeIdx+1) ) fSamplingNEvents.resize(treeIdx+1);
   if (fSamplingWeight.size() < UInt_t(treeIdx+1) )   fSamplingWeight.resize(treeIdx+1);
      
   if (fraction > 0.999999 || fraction < 0.0000001) {
      fSampling.at( treeIdx ) = false;
      fSamplingNEvents.at( treeIdx ) = 0;
      fSamplingWeight.at( treeIdx ) = 1.0;
      return;
   }

   // for the initialization, the sampling has to be turned off, afterwards we will turn it on
   fSampling.at( treeIdx )  = false;

   fSamplingNEvents.at( treeIdx ) = Int_t(fraction*GetNEvents());
   fSamplingWeight.at( treeIdx ) = weight;

   Long64_t nEvts = GetNEvents();
   fSamplingEventList.at( treeIdx ).reserve( nEvts );
   fSamplingSelected.at( treeIdx ).reserve( fSamplingNEvents.at(treeIdx) );
   for (Long64_t ievt=0; ievt<nEvts; ievt++) {
      std::pair<Float_t,Long64_t> *p = new std::pair<Float_t,Long64_t>(1.0,ievt);
      fSamplingEventList.at( treeIdx ).push_back( p );
   }

   // now turn sampling on
   fSampling.at( treeIdx ) = true;
}


//_______________________________________________________________________
void TMVA::DataSet::CreateSampling() const
{
   // create an event sampling (random or importance sampling)

   Int_t treeIdx = TreeIndex( GetCurrentType() );

   if (!fSampling.at(treeIdx) ) return;

   if (fSamplingRandom == 0 )
      Log() << kFATAL
            << "no random generator present for creating a random/importance sampling (initialized?)" << Endl;

   // delete the previous selection
   fSamplingSelected.at(treeIdx).clear();

   // create a temporary event-list
   std::vector< std::pair< Float_t, Long64_t >* > evtList;
   std::vector< std::pair< Float_t, Long64_t >* >::iterator evtListIt;

   // some variables
   Float_t sumWeights = 0;

   // make a copy of the event-list
   evtList.assign( fSamplingEventList.at(treeIdx).begin(), fSamplingEventList.at(treeIdx).end() );

   // sum up all the weights (internal weights for importance sampling)
   for (evtListIt = evtList.begin(); evtListIt != evtList.end(); evtListIt++) {
      sumWeights += (*evtListIt)->first;
   }
   evtListIt = evtList.begin();

   // random numbers
   std::vector< Float_t > rnds;
   rnds.reserve(fSamplingNEvents.at(treeIdx));

   Float_t pos = 0;
   for (Int_t i = 0; i < fSamplingNEvents.at(treeIdx); i++) {
      pos = fSamplingRandom->Rndm()*sumWeights;
      rnds.push_back( pos );
   }
   
   // sort the random numbers
   std::sort(rnds.begin(),rnds.end());
   
   // select the events according to the random numbers
   std::vector< Float_t >::iterator rndsIt = rnds.begin();
   Float_t runningSum = 0.000000001;
   for (evtListIt = evtList.begin(); evtListIt != evtList.end();) {
      runningSum += (*evtListIt)->first;
      if (runningSum >= (*rndsIt)) {
         fSamplingSelected.at(treeIdx).push_back( (*evtListIt) );
         evtListIt = evtList.erase( evtListIt );

         rndsIt++;
         if (rndsIt == rnds.end() ) break;
      }
      else {
         evtListIt++;
      }
   }
}

//_______________________________________________________________________
void TMVA::DataSet::EventResult( Bool_t successful, Long64_t evtNumber )
{
   // increase the importance sampling weight of the event 
   // when not successful and decrease it when successful


   if (!fSampling.at(fCurrentTreeIdx)) return;
   if (fSamplingWeight.at(fCurrentTreeIdx) > 0.99999999999) return;

   Long64_t start = 0;
   Long64_t stop  = fSamplingEventList.at(fCurrentTreeIdx).size() -1;
   if (evtNumber >= 0) {
      start = evtNumber; 
      stop  = evtNumber;
   }
   for ( Long64_t iEvt = start; iEvt <= stop; iEvt++ ){
      if (Long64_t(fSamplingEventList.at(fCurrentTreeIdx).size()) < iEvt) {
         Log() << kWARNING << "event number (" << iEvt 
               << ") larger than number of sampled events (" 
               << fSamplingEventList.at(fCurrentTreeIdx).size() << " of tree " << fCurrentTreeIdx << ")" << Endl;
         return;
      }
      Float_t weight = fSamplingEventList.at(fCurrentTreeIdx).at( iEvt )->first;
      if (!successful) {
         //      weight /= (fSamplingWeight.at(fCurrentTreeIdx)/fSamplingEventList.at(fCurrentTreeIdx).size());
         weight /= fSamplingWeight.at(fCurrentTreeIdx);
         if (weight > 1.0 ) weight = 1.0;
      }
      else {
         //      weight *= (fSamplingWeight.at(fCurrentTreeIdx)/fSamplingEventList.at(fCurrentTreeIdx).size());
         weight *= fSamplingWeight.at(fCurrentTreeIdx);
      }
      fSamplingEventList.at(fCurrentTreeIdx).at( iEvt )->first = weight;
   }
}


//_______________________________________________________________________
TTree* TMVA::DataSet::GetTree( Types::ETreeType type ) 
{ 
   // create the test/trainings tree with all the variables, the weights, the classes, the targets, the spectators, the MVA outputs
   
   Log() << kDEBUG << "GetTree(" << ( type==Types::kTraining ? "training" : "testing" ) << ")" << Endl;

   // the dataset does not hold the tree, this function returns a new tree everytime it is called

   if (type!=Types::kTraining && type!=Types::kTesting) return 0;

   Types::ETreeType savedType = GetCurrentType();

   SetCurrentType(type);
   const UInt_t t = TreeIndex(type);
   if (fResults.size() <= t) {
      Log() << kWARNING << "No results for treetype " << ( type==Types::kTraining ? "training" : "testing" ) 
            << " found. Size=" << fResults.size() << Endl;
   }

   // return number of background training events in dataset
   TString treeName( (type == Types::kTraining ? "TrainTree" : "TestTree" ) );
   TTree *tree = new TTree(treeName,treeName);

   Float_t *varVals = new Float_t[fdsi.GetNVariables()];
   Float_t *tgtVals = new Float_t[fdsi.GetNTargets()];
   Float_t *visVals = new Float_t[fdsi.GetNSpectators()];

   UInt_t cls;
   Float_t weight;
   //   TObjString *className = new TObjString();
   char *className = new char[40];


   //Float_t metVals[fResults.at(t).size()][Int_t(fdsi.GetNTargets()+1)];
   // replace by:  [Joerg]
   Float_t **metVals = new Float_t*[fResults.at(t).size()];
   for(UInt_t i=0; i<fResults.at(t).size(); i++ )
      metVals[i] = new Float_t[fdsi.GetNTargets()+fdsi.GetNClasses()];

   // create branches for event-variables
   tree->Branch( "classID", &cls, "classID/I" ); 
   tree->Branch( "className",(void*)className, "className/C" ); 

   // create all branches for the variables
   Int_t n = 0;
   for (std::vector<VariableInfo>::const_iterator itVars = fdsi.GetVariableInfos().begin(); 
        itVars != fdsi.GetVariableInfos().end(); itVars++) {

      // has to be changed to take care of types different than float: TODO
      tree->Branch( (*itVars).GetInternalName(), &varVals[n], (*itVars).GetInternalName()+TString("/F") ); 
      n++;
   }
   // create the branches for the targets
   n = 0;
   for (std::vector<VariableInfo>::const_iterator itTgts = fdsi.GetTargetInfos().begin(); 
        itTgts != fdsi.GetTargetInfos().end(); itTgts++) {
      // has to be changed to take care of types different than float: TODO
      tree->Branch( (*itTgts).GetInternalName(), &tgtVals[n], (*itTgts).GetInternalName()+TString("/F") ); 
      n++;
   }
   // create the branches for the spectator variables
   n = 0;
   for (std::vector<VariableInfo>::const_iterator itVis = fdsi.GetSpectatorInfos().begin(); 
        itVis != fdsi.GetSpectatorInfos().end(); itVis++) {
      // has to be changed to take care of types different than float: TODO
      tree->Branch( (*itVis).GetInternalName(), &visVals[n], (*itVis).GetInternalName()+TString("/F") ); 
      n++;
   }

   tree->Branch( "weight", &weight, "weight/F" );

   // create all the branches for the results
   n = 0;
   for (std::map< TString, Results* >::iterator itMethod = fResults.at(t).begin(); 
        itMethod != fResults.at(t).end(); itMethod++) {


      Log() << kDEBUG << "analysis type: " << (itMethod->second->GetAnalysisType()==Types::kRegression ? "Regression" :
                                               (itMethod->second->GetAnalysisType()==Types::kMulticlass ? "Multiclass" : "Classification" )) << Endl;
      
      if (itMethod->second->GetAnalysisType() == Types::kClassification) {
         // classification
         tree->Branch( itMethod->first, &(metVals[n][0]), itMethod->first + "/F" );
      } 
      else if (itMethod->second->GetAnalysisType() == Types::kMulticlass) {
         // multiclass classification
         TString leafList("");
         for (UInt_t iCls = 0; iCls < fdsi.GetNClasses(); iCls++) {
            if (iCls > 0) leafList.Append( ":" );
            leafList.Append( fdsi.GetClassInfo( iCls )->GetName() );
            leafList.Append( "/F" );
         }
         Log() << kDEBUG << "itMethod->first " << itMethod->first <<  "    LEAFLIST: " 
               << leafList << "    itMethod->second " << itMethod->second <<  Endl;
         tree->Branch( itMethod->first, (metVals[n]), leafList );
      } 
      else if (itMethod->second->GetAnalysisType() == Types::kRegression) {
         // regression
         TString leafList("");
         for (UInt_t iTgt = 0; iTgt < fdsi.GetNTargets(); iTgt++) {
            if (iTgt > 0) leafList.Append( ":" );
            leafList.Append( fdsi.GetTargetInfo( iTgt ).GetInternalName() );
            //            leafList.Append( fdsi.GetTargetInfo( iTgt ).GetLabel() );
            leafList.Append( "/F" );
         }
         Log() << kDEBUG << "itMethod->first " << itMethod->first <<  "    LEAFLIST: " 
               << leafList << "    itMethod->second " << itMethod->second <<  Endl;
         tree->Branch( itMethod->first, (metVals[n]), leafList );
      } 
      else {
         Log() << kWARNING << "Unknown analysis type for result found when writing TestTree." << Endl;
      }
      n++;

   }

   // loop through all the events
   for (Long64_t iEvt = 0; iEvt < GetNEvents( type ); iEvt++) {
      // write the event-variables
      const Event* ev = GetEvent( iEvt );

      // write the classnumber and the classname
      cls = ev->GetClass();
      weight = ev->GetWeight();
      TString tmp = fdsi.GetClassInfo( cls )->GetName();
      for (Int_t itmp = 0; itmp < tmp.Sizeof(); itmp++) {
         className[itmp] = tmp(itmp);
         className[itmp+1] = 0;
      }

      // write the variables, targets and spectator variables
      for (UInt_t ivar = 0; ivar < ev->GetNVariables();   ivar++) varVals[ivar] = ev->GetValue( ivar );
      for (UInt_t itgt = 0; itgt < ev->GetNTargets();     itgt++) tgtVals[itgt] = ev->GetTarget( itgt );
      for (UInt_t ivis = 0; ivis < ev->GetNSpectators();  ivis++) visVals[ivis] = ev->GetSpectator( ivis );


      // loop through all the results and write the branches
      n=0;
      for (std::map<TString, Results*>::iterator itMethod = fResults.at(t).begin();
           itMethod != fResults.at(t).end(); itMethod++) {

         Results* results = itMethod->second;
         const std::vector< Float_t >& vals = results->operator[](iEvt);

         if (itMethod->second->GetAnalysisType() == Types::kClassification) {
            // classification
            metVals[n][0] = vals[0];
         }
         else if (itMethod->second->GetAnalysisType() == Types::kMulticlass) {
            // multiclass classification
            for (UInt_t nCls = 0, nClsEnd=fdsi.GetNClasses(); nCls < nClsEnd; nCls++) {
               Float_t val = vals.at(nCls);
               metVals[n][nCls] = val;
            }
         }
         else if (itMethod->second->GetAnalysisType() == Types::kRegression) {
            // regression
            for (UInt_t nTgts = 0; nTgts < fdsi.GetNTargets(); nTgts++) {
               Float_t val = vals.at(nTgts);
               metVals[n][nTgts] = val;
            }
         }
         n++;
      }
      // fill the variables into the tree
      tree->Fill();
   }

   Log() << kINFO << "Created tree '" << tree->GetName() << "' with " << tree->GetEntries() << " events" << Endl;

   SetCurrentType(savedType);

   delete[] varVals;
   delete[] tgtVals;
   delete[] visVals;

   for(UInt_t i=0; i<fResults.at(t).size(); i++ )
      delete[] metVals[i];
   delete[] metVals;

   delete[] className;

   return tree;
}

