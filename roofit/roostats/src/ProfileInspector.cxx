// @(#)root/roostats:$Id: ProfileInspector.cxx 34109 2010-06-24 15:00:16Z moneta $

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 *   Akira Shibata
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class RooStats::ProfileInspector
    \ingroup Roostats

Utility class to plot conditional MLE of nuisance parameters vs. Parameters of Interest

*/


#include "RooStats/ProfileInspector.h"
#include "RooRealVar.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooAbsPdf.h"
#include "RooCurve.h"
#include "TAxis.h"


using namespace RooStats;
using std::string, std::map;

////////////////////////////////////////////////////////////////////////////////

ProfileInspector::ProfileInspector()
{
}

////////////////////////////////////////////////////////////////////////////////
/// ProfileInspector destructor

ProfileInspector::~ProfileInspector()
{
}

////////////////////////////////////////////////////////////////////////////////
///
/// This tool makes a plot of the conditional maximum likelihood estimate of the nuisance parameter
///   vs the parameter of interest
///
/// This enables you to discover if any of the nuisance parameters are behaving strangely
/// curve is the optional parameters, when used you can specify the points previously scanned
/// in the process of plotOn or createHistogram.
/// To do this, you can do the following after the plot has been made:
/// ~~~ {.cpp}
/// profile, RooRealVar * poi, RooCurve * curve ){
///RooCurve * curve = 0;
/// ~~~

TList* ProfileInspector::GetListOfProfilePlots( RooAbsData& data, RooStats::ModelConfig * config)
{

  const RooArgSet* poi_set = config->GetParametersOfInterest();
  const RooArgSet* nuis_params=config->GetNuisanceParameters();
  RooAbsPdf* pdf =  config->GetPdf();


  if(!poi_set){
    std::cout << "no parameters of interest" << std::endl;
    return nullptr;
  }

  if(poi_set->size()!=1){
    std::cout << "only one parameter of interest is supported currently" << std::endl;
    return nullptr;
  }
  RooRealVar* poi = static_cast<RooRealVar*>(poi_set->first());


  if(!nuis_params){
    std::cout << "no nuisance parameters" << std::endl;
    return nullptr;
  }

  if(!pdf){
    std::cout << "pdf not set" << std::endl;
    return nullptr;
  }

  std::unique_ptr<RooAbsReal> nll{pdf->createNLL(data)};
  std::unique_ptr<RooAbsReal> profile{nll->createProfile(*poi)};

  TList * list = new TList;
  Int_t curve_N=100;
  double* curve_x=nullptr;
//   if(curve){
//     curve_N=curve->GetN();
//     curve_x=curve->GetX();
//     } else {
  double max = dynamic_cast<RooAbsRealLValue*>(poi)->getMax();
  double min = dynamic_cast<RooAbsRealLValue*>(poi)->getMin();
  double step = (max-min)/(curve_N-1);
  curve_x=new double[curve_N];
  for(int i=0; i<curve_N; ++i){
     curve_x[i]=min+step*i;
  }
//   }

  map<string, std::vector<double> > name_val;
  for(int i=0; i<curve_N; i++){
    poi->setVal(curve_x[i]);
    profile->getVal();

    for (auto const *nuis_param : dynamic_range_cast<RooRealVar *> (*nuis_params)){
       if(nuis_param) {
          string name = nuis_param->GetName();
          if(nuis_params->empty()) continue;
          if(nuis_param && (! nuis_param->isConstant())){
             if(name_val.find(name)==name_val.end()) name_val[name]=std::vector<double>(curve_N);
             name_val[name][i]=nuis_param->getVal();

             if(i==curve_N-1){
                TGraph* g = new TGraph(curve_N, curve_x, &(name_val[name].front()));
                g->SetName((name+"_"+string(poi->GetName())+"_profile").c_str());
                g->GetXaxis()->SetTitle(poi->GetName());
                g->GetYaxis()->SetTitle(nuis_param->GetName());
                g->SetTitle("");
                list->Add(g);
             }
          }
       }
    }
  }

  delete [] curve_x;

  return list;
}
