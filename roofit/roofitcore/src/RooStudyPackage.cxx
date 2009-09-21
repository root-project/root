/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooStudyPackage is a utility class to manage studies that consist of
// repeated applications of generate-and-fit operations on a workspace
//
// END_HTML
//



#include "RooFit.h"
#include "Riostream.h"

#include "RooStudyPackage.h"
#include "RooWorkspace.h"
#include "RooAbsStudy.h"
#include "RooDataSet.h"
#include "RooMsgService.h"
#include "TProof.h"
#include "TTree.h"
#include "TDSet.h"
#include "TFile.h"
#include "TRandom.h"
#include "RooRandom.h"

using namespace std ;

ClassImp(RooStudyPackage)
  ;



//_____________________________________________________________________________
RooStudyPackage::RooStudyPackage() : _ws(0)
{  
}



//_____________________________________________________________________________
RooStudyPackage::RooStudyPackage(RooWorkspace& w) : _ws(&w)
{  
}



//_____________________________________________________________________________
void RooStudyPackage::addStudy(RooAbsStudy& study) 
{
  cout << "RooStudyPackage(" << this << ") addStudy " << &study << endl ;
  _studies.push_back(&study) ;
}



//_____________________________________________________________________________
void RooStudyPackage::driver(Int_t nExperiments)
{
  initialize() ;
  run(nExperiments) ;
  finalize() ;
} 



//_____________________________________________________________________________
void RooStudyPackage::initialize() 
{
  // Make iterator over copy of studies attached to workspace
  for (list<RooAbsStudy*>::iterator iter=_studies.begin() ; iter!=_studies.end() ; iter++) {
    (*iter)->attach(*_ws) ;
    (*iter)->initialize() ;
  }

}


//_____________________________________________________________________________
void RooStudyPackage::run(Int_t nExperiments) 
{

  // Run the requested number of experiments
  Int_t prescale = nExperiments>100 ? Int_t(nExperiments/100) : 1 ;
  for (Int_t i=0 ; i<nExperiments ; i++) {
    if (i%prescale==0) {
      coutP(Generation) << "RooStudyPackage::run(" << GetName() << ") processing experiment " << i << "/" << nExperiments << endl ;
    }    
    runOne() ;
  }
}



//_____________________________________________________________________________
void RooStudyPackage::runOne() 
{
  for (list<RooAbsStudy*>::iterator iter=_studies.begin() ; iter!=_studies.end() ; iter++) {
    (*iter)->execute() ;
  }    
}




//_____________________________________________________________________________
void RooStudyPackage::finalize() 
{   
  // Finalize all studies
  for (list<RooAbsStudy*>::iterator iter=_studies.begin() ; iter!=_studies.end() ; iter++) {
    (*iter)->finalize() ;
  }
}




//_____________________________________________________________________________
void RooStudyPackage::exportData(TList* olist, Int_t seqno)
{
  for (list<RooAbsStudy*>::iterator iter=_studies.begin() ; iter!=_studies.end() ; iter++) {

    (*iter)->finalize() ;

    RooDataSet* summaryData = (*iter)->summaryData() ;
    if (summaryData) {
      summaryData->SetName(Form("%s_%d",summaryData->GetName(),seqno)) ;
      cout << "registering summary dataset: " ; summaryData->Print() ;
      olist->Add(summaryData) ;
    }

    RooLinkedList* detailedData = (*iter)->detailedData() ;
    if (detailedData && detailedData->GetSize()>0) {

      detailedData->SetName(Form("%s_%d",detailedData->GetName(),seqno)) ;
      cout << "registering detailed dataset " << detailedData->IsA()->GetName() << "::" 
	   << detailedData->GetName() << " with " << detailedData->GetSize() << " elements" << endl ;
      TIterator* diter = detailedData->MakeIterator() ;
      TNamed* dobj ;
      while((dobj=(TNamed*)diter->Next())) {
	dobj->SetName(Form("%s_%d",dobj->GetName(),seqno)) ;
      }
      delete diter ;
      olist->Add(detailedData) ;
      (*iter)->releaseDetailData() ;
    }
  }        
}



//_____________________________________________________________________________
Int_t RooStudyPackage::initRandom()
{
  // Choose random seed for this process
  gRandom->SetSeed(0) ;
  Int_t seed = gRandom->Integer(1000000) ;
  RooRandom::randomGenerator()->SetSeed(seed) ;
  gRandom->SetSeed(seed) ;

  return seed ;
}



//_____________________________________________________________________________
void RooStudyPackage::processFile(const char* studyName, Int_t nexp) 
{
  // Read in study package
  string name_fin = Form("study_data_%s.root",studyName) ;
  TFile fin(name_fin.c_str()) ;
  RooStudyPackage* pkg = dynamic_cast<RooStudyPackage*>(fin.Get("studypack")) ;
  if (!pkg) {
    cout << "RooStudyPackage::processFile() ERROR input file " << name_fin << " does not contain a RooStudyPackage named 'studypack'" << endl ;
    return ;
  }
      
  // Initialize random seed
  Int_t seqno = pkg->initRandom() ;
  cout << "RooStudyPackage::processFile() Initial random seed for this run is " << seqno << endl ;

  // Run study
  pkg->driver(nexp) ;

  // Save result
  TList res ;
  pkg->exportData(&res,seqno) ;
  TFile fout(Form("study_result_%s_%d.root",studyName,seqno),"RECREATE") ;
  res.Write() ;
  fout.Close() ;    
}
