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
// RooAbsStudy is an abstract base class for RooStudyManager modules
//
// END_HTML
//



#include "RooFit.h"
#include "Riostream.h"

#include "RooAbsStudy.h"
#include "RooMsgService.h"
#include "RooDataSet.h"
#include "TList.h"
#include "TClass.h"

using namespace std ;

ClassImp(RooAbsStudy)
  ;


//_____________________________________________________________________________
RooAbsStudy::RooAbsStudy(const char* name, const char* title) : TNamed(name,title), _storeDetails(0), _summaryData(0), _detailData(0), _ownDetailData(kTRUE)
{  
  // Constructor
}



//_____________________________________________________________________________
RooAbsStudy::RooAbsStudy(const RooAbsStudy& other) : TNamed(other), _storeDetails(other._storeDetails), _summaryData(other._summaryData), 
						     _detailData(0), _ownDetailData(other._ownDetailData)
{  
  // Copy constructor
}



//_____________________________________________________________________________
RooAbsStudy::~RooAbsStudy() 
{
  // Destructor
  if (_summaryData) delete _summaryData ;
  if (_ownDetailData && _detailData) {
    _detailData->Delete() ;
    delete _detailData ;
  }
}




//_____________________________________________________________________________
void RooAbsStudy::registerSummaryOutput(const RooArgSet& allVars, const RooArgSet& varsWithError, const RooArgSet& varsWithAsymError) 
{
  if (_summaryData) {
    coutW(ObjectHandling) << "RooAbsStudy::registerSummaryOutput(" << GetName() << ") WARNING summary output already registered" << endl ;
    return ;
  }

  string name = Form("%s_summary_data",GetName()) ;
  string title = Form("%s Summary Data",GetTitle()) ;
  _summaryData = new RooDataSet(name.c_str(),title.c_str(),allVars,RooFit::StoreError(varsWithError),RooFit::StoreAsymError(varsWithAsymError)) ;  
}


//_____________________________________________________________________________
void RooAbsStudy::storeSummaryOutput(const RooArgSet& vars) 
{
  if (!_summaryData) {
    coutE(ObjectHandling) << "RooAbsStudy::storeSummaryOutput(" << GetName() << ") ERROR: no summary output data configuration registered" << endl ;
    return ;
  }
  _summaryData->add(vars) ;
}



//_____________________________________________________________________________
void RooAbsStudy::storeDetailedOutput(TNamed& object) 
{
  if (_storeDetails) {

    if (!_detailData) {
      _detailData = new RooLinkedList ;
      _detailData->SetName(Form("%s_detailed_data",GetName())) ;
      //cout << "RooAbsStudy::ctor() detailData name = " << _detailData->GetName() << endl ;
    }

    object.SetName(Form("%s_detailed_data_%d",GetName(),_detailData->GetSize())) ;    
    //cout << "storing detailed data with name " << object.GetName() << endl ;
    _detailData->Add(&object) ;
  } else {
    delete &object ;
  }
}



//_____________________________________________________________________________
void RooAbsStudy::aggregateSummaryOutput(TList* chunkList)
{
  if (!chunkList) return ;

  TIterator* iter = chunkList->MakeIterator() ;
  TObject* obj ;
  while((obj=iter->Next())) {

    //cout << "RooAbsStudy::aggregateSummaryOutput(" << GetName() << ") processing object " << obj->GetName() << endl ;

    RooDataSet* data = dynamic_cast<RooDataSet*>(obj) ;
    if (data) {
      if (TString(data->GetName()).BeginsWith(Form("%s_summary_data",GetName()))) {
	//cout << "RooAbsStudy::aggregateSummaryOutput(" << GetName() << ") found summary block " << data->GetName() << endl ;
	if (!_summaryData) {
	  _summaryData = (RooDataSet*) data->Clone(Form("%s_summary_data",GetName())) ;
	} else {
	  _summaryData->append(*data) ;
	}
      }
    }

    RooLinkedList* dlist = dynamic_cast<RooLinkedList*>(obj) ;
    if (dlist) {
      if (TString(dlist->GetName()).BeginsWith(Form("%s_detailed_data",GetName()))) {
	//cout << "RooAbsStudy::aggregateSummaryOutput(" << GetName() << ") found detail block " <<dlist->GetName() << " with " << dlist->GetSize() << " entries" << endl ;
	TIterator* diter = dlist->MakeIterator() ;
	TNamed* dobj ;
	while((dobj=(TNamed*)diter->Next())) {	  
	  storeDetailedOutput(*dobj) ;
	}
	delete diter ;
      }
    }
  }
}
