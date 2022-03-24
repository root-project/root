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

/**
\file RooAbsStudy.cxx
\class RooAbsStudy
\ingroup Roofitcore

RooAbsStudy is an abstract base class for RooStudyManager modules

**/

#include "Riostream.h"

#include "RooAbsStudy.h"
#include "RooMsgService.h"
#include "RooDataSet.h"
#include "TList.h"
#include "TClass.h"

using namespace std ;

ClassImp(RooAbsStudy);
  ;


////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsStudy::RooAbsStudy(const char* name, const char* title) : TNamed(name,title), _storeDetails(0), _summaryData(0), _detailData(0), _ownDetailData(kTRUE)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsStudy::RooAbsStudy(const RooAbsStudy& other) : TNamed(other), _storeDetails(other._storeDetails), _summaryData(other._summaryData),
                       _detailData(0), _ownDetailData(other._ownDetailData)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsStudy::~RooAbsStudy()
{
  if (_summaryData) delete _summaryData ;
  if (_ownDetailData && _detailData) {
    _detailData->Delete() ;
    delete _detailData ;
  }
}




////////////////////////////////////////////////////////////////////////////////

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


////////////////////////////////////////////////////////////////////////////////

void RooAbsStudy::storeSummaryOutput(const RooArgSet& vars)
{
  if (!_summaryData) {
    coutE(ObjectHandling) << "RooAbsStudy::storeSummaryOutput(" << GetName() << ") ERROR: no summary output data configuration registered" << endl ;
    return ;
  }
  _summaryData->add(vars) ;
}



////////////////////////////////////////////////////////////////////////////////

void RooAbsStudy::storeDetailedOutput(TNamed& object)
{
  if (_storeDetails) {

    if (!_detailData) {
      _detailData = new RooLinkedList ;
      _detailData->SetName(TString::Format("%s_detailed_data_list",GetName())) ;
      //cout << "RooAbsStudy::ctor() detailData name = " << _detailData->GetName() << endl ;
    }

    object.SetName(TString::Format("%s_detailed_data_%d",GetName(),_detailData->GetSize())) ;
    //cout << "storing detailed data with name " << object.GetName() << endl ;
    _detailData->Add(&object) ;
  } else {
    delete &object ;
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooAbsStudy::aggregateSummaryOutput(TList* chunkList)
{
  if (!chunkList) return ;

  TIter iter = chunkList->MakeIterator() ;
  TObject* obj ;
  while((obj=iter.Next())) {

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

    if (auto dlist = dynamic_cast<RooLinkedList*>(obj)) {
      if (TString(dlist->GetName()).BeginsWith(Form("%s_detailed_data",GetName()))) {
        for(auto * dobj : static_range_cast<TNamed*>(*dlist)) storeDetailedOutput(*dobj) ;
      }
    }
  }
}
