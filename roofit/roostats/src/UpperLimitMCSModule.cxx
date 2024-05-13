// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke, Nils Ruthmann
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::UpperLimitMCSModule
    \ingroup Roostats

This class allow to compute in the ToyMcStudy framework the ProfileLikelihood
upper limit for each toy-MC sample generated

*/

#include "Riostream.h"

#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooStats/UpperLimitMCSModule.h"
#include "RooMsgService.h"
#include "RooStats/ConfInterval.h"
#include "RooStats/PointSetInterval.h"
#include "RooStats/LikelihoodInterval.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooRealVar.h"

ClassImp(RooStats::UpperLimitMCSModule);

using namespace RooStats ;

////////////////////////////////////////////////////////////////////////////////

UpperLimitMCSModule::UpperLimitMCSModule(const RooArgSet* poi, double CL) :
  RooAbsMCStudyModule(Form("UpperLimitMCSModule_%s",poi->first()->GetName()),Form("UpperLimitMCSModule_%s",poi->first()->GetName())),
  _parName(poi->first()->GetName()),
  _plc(nullptr),_ul(nullptr),_poi(nullptr), _data(nullptr),_cl(CL), _model(nullptr)
{
  std::cout<<"RooUpperLimitConstructor ParName:"<<_parName<<std::endl;
  std::cout<<"RooUpperLimitConstructor CL:"<<_cl<<std::endl;
  // Constructor of module with parameter to be interpreted as nSignal and the value of the
  // null hypothesis for nSignal (usually zero)
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

UpperLimitMCSModule::UpperLimitMCSModule(const UpperLimitMCSModule& other) :
  RooAbsMCStudyModule(other),
  _parName(other._poi->first()->GetName()),
  _plc(nullptr),_ul(nullptr),_poi(other._poi), _data(nullptr), _cl(other._cl), _model(other._model)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

UpperLimitMCSModule:: ~UpperLimitMCSModule()
{

  if (_plc) {
    delete _plc ;
  }
  if (_data) {
    delete _data ;
  }
  if(_ul){
    delete _ul;
  }
  if(_poi){
     delete _poi;
  }
  if (_model){
    delete _model;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize module after attachment to RooMCStudy object

bool UpperLimitMCSModule::initializeInstance()
{
  // Check that parameter is also present in fit parameter list of RooMCStudy object
  if (!fitParams()->find(_parName.c_str())) {
    coutE(InputArguments) << "UpperLimitMCSModule::initializeInstance:: ERROR: No parameter named " << _parName << " in RooMCStudy!" << std::endl ;
    return false ;
  }

  //Construct the ProfileLikelihoodCalculator
  _poi=new RooArgSet(*(fitParams()->find(_parName.c_str())));
  std::cout<<"RooUpperLimit Initialize Instance: POI Set:"<<std::endl;
  _poi->Print("v");
  std::cout<<"RooUpperLimit Initialize Instance: End:"<<std::endl;



  std::string ulName = "ul_" + _parName;
  std::string ulTitle = "UL for parameter " + _parName;
  _ul = new RooRealVar(ulName.c_str(),ulTitle.c_str(),0) ;


  // Create new dataset to be merged with RooMCStudy::fitParDataSet
  _data = new RooDataSet("ULSigData","Additional data for UL study",RooArgSet(*_ul)) ;

  return true ;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize module at beginning of RooCMStudy run

bool UpperLimitMCSModule::initializeRun(Int_t /*numSamples*/)
{
  _data->reset() ;
  return true ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return auxiliary dataset with results of delta(-log(L))
/// calculations of this module so that it is merged with
/// RooMCStudy::fitParDataSet() by RooMCStudy

RooDataSet* UpperLimitMCSModule::finalizeRun()
{
  return _data ;
}

////////////////////////////////////////////////////////////////////////////////

// bool UpperLimitMCSModule::processAfterFit(Int_t /*sampleNum*/)
// {
//   // Save likelihood from nominal fit, fix chosen parameter to its
//   // null hypothesis value and rerun fit Save difference in likelihood
//   // and associated Gaussian significance in auxiliary dataset

//   RooRealVar* par = static_cast<RooRealVar*>(fitParams()->find(_parName.c_str())) ;
//   par->setVal(_nullValue) ;
//   par->setConstant(true) ;
//   RooFitResult* frnull = refit() ;
//   par->setConstant(false) ;

//   _nll0h->setVal(frnull->minNll()) ;

//   double deltaLL = (frnull->minNll() - nllVar()->getVal()) ;
//   double signif = deltaLL>0 ? sqrt(2*deltaLL) : -sqrt(-2*deltaLL) ;
//   _sig0h->setVal(signif) ;
//   _dll0h->setVal(deltaLL) ;


//   _data->add(RooArgSet(*_nll0h,*_dll0h,*_sig0h)) ;

//   delete frnull ;
//   return true ;

// }

////////////////////////////////////////////////////////////////////////////////

bool UpperLimitMCSModule::processBetweenGenAndFit(Int_t /*sampleNum*/) {
  std::cout<<"after generation Test"<<std::endl;

  if (!fitInitParams() || !genSample() || !fitParams() || !fitModel() ) return false;

  static_cast<RooRealVar*>(_poi->first())->setVal(static_cast<RooRealVar*>(fitInitParams()->find(_parName.c_str()))->getVal());

  //_poi->first()->Print();
  static_cast<RooRealVar*>(_poi->first())->setBins(1000);
  //fitModel()->Print("v");

  std::cout<<"generated Entries:"<<genSample()->numEntries()<<std::endl;

  RooStats::ProfileLikelihoodCalculator plc( *(genSample()), *(fitModel()), *_poi);

  //PLC calculates intervals. for one sided ul multiply testsize by two
  plc.SetTestSize(2*(1-_cl));
  RooStats::ConfInterval* pllint=plc.GetInterval();

  if (!pllint) return false;

  std::cout<<"poi value: "<<(static_cast<RooRealVar*>(_poi->first()))->getVal()<<std::endl;
  std::cout<<(static_cast<RooRealVar*>((fitParams()->find(_parName.c_str()))))->getVal()<<std::endl;
  std::cout<<(static_cast<RooStats::LikelihoodInterval*>(pllint))->UpperLimit(static_cast<RooRealVar&>(*(_poi->first())))<<std::endl;


  //Go to the fit Value for zour POI to make sure upper limit works correct.
  //fitModel()->fitTo(*genSample());



  _ul->setVal((static_cast<RooStats::LikelihoodInterval*>(pllint))->UpperLimit(static_cast<RooRealVar&>(*(fitParams()->find(_parName.c_str())))));

  _data->add(RooArgSet(*_ul));
  std::cout<<"UL:"<<_ul->getVal()<<std::endl;
//   if (_ul->getVal()<1){

//   RooStats::LikelihoodIntervalPlot plotpll((RooStats::LikelihoodInterval*) pllint);
//   TCanvas c1;
//   plotpll.Draw();
//   c1.Print("test.ps");
//   std::cout<<" UL<1 whats going on here?"<<std::endl;
//   abort();
//   }

  delete pllint;


  return true;
}
