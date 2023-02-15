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
\file RooDataHist.cxx
\class RooDataHist
\ingroup Roofitcore

The RooDataHist is a container class to hold N-dimensional binned data. Each bin's central
coordinates in N-dimensional space are represented by a RooArgSet containing RooRealVar, RooCategory
or RooStringVar objects, thus data can be binned in real and/or discrete dimensions.

There is an unbinned equivalent, RooDataSet.

### Inspecting a datahist
Inspect a datahist using Print() to get the coordinates and `weight()` to get the bin contents:
```
datahist->Print("V");
datahist->get(0)->Print("V"); std::cout << "w=" << datahist->weight(0) << std::endl;
datahist->get(1)->Print("V"); std::cout << "w=" << datahist->weight(1) << std::endl;
...
```

### Plotting data.
See RooAbsData::plotOn().

### Creating a datahist using RDataFrame
\see RooAbsDataHelper, rf408_RDataFrameToRooFit.C

**/

#include "RooDataHist.h"

#include "Riostream.h"
#include "RooMsgService.h"
#include "RooDataHistSliceIter.h"
#include "RooAbsLValue.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooMath.h"
#include "RooBinning.h"
#include "RooPlot.h"
#include "RooHistError.h"
#include "RooCategory.h"
#include "RooCmdConfig.h"
#include "RooLinkedListIter.h"
#include "RooTreeDataStore.h"
#include "RooVectorDataStore.h"
#include "RooTrace.h"
#include "RooFormulaVar.h"
#include "RooFormula.h"
#include "RooUniformBinning.h"
#include "RooSpan.h"

#include "ROOT/StringUtils.hxx"

#include "TAxis.h"
#include "TH1.h"
#include "TTree.h"
#include "TBuffer.h"
#include "TMath.h"
#include "Math/Util.h"

using namespace std;

ClassImp(RooDataHist);


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooDataHist::RooDataHist()
{
  TRACE_CREATE
}


std::unique_ptr<RooAbsDataStore> RooDataHist::makeDefaultDataStore(const char* name, const char* title, RooArgSet const& vars)
{
  return RooAbsData::defaultStorageType == RooAbsData::Tree
      ? static_cast<std::unique_ptr<RooAbsDataStore>>(std::make_unique<RooTreeDataStore>(name, title, vars))
      : static_cast<std::unique_ptr<RooAbsDataStore>>(std::make_unique<RooVectorDataStore>(name, title, vars));
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor of an empty data hist from a RooArgSet defining the dimensions
/// of the data space. The range and number of bins in each dimensions are taken
/// from getMin()getMax(),getBins() of each RooAbsArg representing that
/// dimension.
///
/// For real dimensions, the fit range and number of bins can be set independently
/// of the plot range and number of bins, but it is advisable to keep the
/// ratio of the plot bin width and the fit bin width an integer value.
/// For category dimensions, the fit ranges always comprises all defined states
/// and each state is always has its individual bin
///
/// To effectively bin real dimensions with variable bin sizes,
/// construct a RooThresholdCategory of the real dimension to be binned variably.
/// Set the thresholds at the desired bin boundaries, and construct the
/// data hist as a function of the threshold category instead of the real variable.
RooDataHist::RooDataHist(RooStringView name, RooStringView title, const RooArgSet& vars, const char* binningName) :
  RooAbsData(name,title,vars)
{
  // Initialize datastore
  _dstore = makeDefaultDataStore(name, title, _vars);

  initialize(binningName) ;

  registerWeightArraysToDataStore();

  appendToDir(this,true) ;
  TRACE_CREATE
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of a data hist from an existing data collection (binned or unbinned)
/// The RooArgSet 'vars' defines the dimensions of the histogram.
/// The range and number of bins in each dimensions are taken
/// from getMin(), getMax(), getBins() of each argument passed.
///
/// For real dimensions, the fit range and number of bins can be set independently
/// of the plot range and number of bins, but it is advisable to keep the
/// ratio of the plot bin width and the fit bin width an integer value.
/// For category dimensions, the fit ranges always comprises all defined states
/// and each state is always has its individual bin
///
/// To effectively bin real dimensions with variable bin sizes,
/// construct a RooThresholdCategory of the real dimension to be binned variably.
/// Set the thresholds at the desired bin boundaries, and construct the
/// data hist as a function of the threshold category instead of the real variable.
///
/// If the constructed data hist has less dimensions that in source data collection,
/// all missing dimensions will be projected.

RooDataHist::RooDataHist(RooStringView name, RooStringView title, const RooArgSet& vars, const RooAbsData& data, double wgt) :
  RooDataHist(name,title,vars)
{
  add(data,static_cast<const RooFormulaVar*>(nullptr),wgt);
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of a data hist from a map of TH1,TH2 or TH3 that are collated into a x+1 dimensional
/// RooDataHist where the added dimension is a category that labels the input source as defined
/// in the histMap argument. The state names used in histMap must correspond to predefined states
/// 'indexCat'
///
/// The RooArgList 'vars' defines the dimensions of the histogram.
/// The ranges and number of bins are taken from the input histogram and must be the same in all histograms

RooDataHist::RooDataHist(RooStringView name, RooStringView title, const RooArgList& vars, RooCategory& indexCat,
          std::map<string,TH1*> histMap, double wgt) :
  RooAbsData(name,title,RooArgSet(vars,&indexCat))
{
  // Initialize datastore
  _dstore = makeDefaultDataStore(name, title, _vars);

  importTH1Set(vars, indexCat, histMap, wgt, false) ;

  registerWeightArraysToDataStore();
  TRACE_CREATE
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of a data hist from a map of RooDataHists that are collated into a x+1 dimensional
/// RooDataHist where the added dimension is a category that labels the input source as defined
/// in the histMap argument. The state names used in histMap must correspond to predefined states
/// 'indexCat'
///
/// The RooArgList 'vars' defines the dimensions of the histogram.
/// The ranges and number of bins are taken from the input histogram and must be the same in all histograms

RooDataHist::RooDataHist(RooStringView name, RooStringView title, const RooArgList& vars, RooCategory& indexCat,
          std::map<string,RooDataHist*> dhistMap, double wgt) :
  RooAbsData(name,title,RooArgSet(vars,&indexCat))
{
  // Initialize datastore
  _dstore = makeDefaultDataStore(name, title, _vars);

  importDHistSet(vars, indexCat, dhistMap, wgt) ;

  registerWeightArraysToDataStore();
  TRACE_CREATE
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of a data hist from an TH1,TH2 or TH3
/// The RooArgSet 'vars' defines the dimensions of the histogram. The ranges
/// and number of bins are taken from the input histogram, and the corresponding
/// values are set accordingly on the arguments in 'vars'

RooDataHist::RooDataHist(RooStringView name, RooStringView title, const RooArgList& vars, const TH1* hist, double wgt) :
  RooAbsData(name,title,vars)
{
  // Initialize datastore
  _dstore = makeDefaultDataStore(name, title, _vars);

  // Check consistency in number of dimensions
  if (vars.getSize() != hist->GetDimension()) {
    std::stringstream errorMsgStream;
    errorMsgStream << "RooDataHist::ctor(" << GetName() << ") ERROR: dimension of input histogram must match "
                   << "number of dimension variables";
    const std::string errorMsg = errorMsgStream.str();
    coutE(InputArguments) << errorMsg << std::endl;
    throw std::invalid_argument(errorMsg);
  }

  importTH1(vars,*hist,wgt, false) ;

  registerWeightArraysToDataStore();
  TRACE_CREATE
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of a binned dataset from a RooArgSet defining the dimensions
/// of the data space. The range and number of bins in each dimensions are taken
/// from getMin() getMax(),getBins() of each RooAbsArg representing that
/// dimension.
///
/// <table>
/// <tr><th> Optional Argument <th> Effect
/// <tr><td> Import(TH1&, bool impDens) <td> Import contents of the given TH1/2/3 into this binned dataset. The
///                                 ranges and binning of the binned dataset are automatically adjusted to
///                                 match those of the imported histogram.
///
///                                 Please note: for TH1& with unequal binning _only_,
///                                 you should decide if you want to import the absolute bin content,
///                                 or the bin content expressed as density. The latter is default and will
///                                 result in the same histogram as the original TH1. For certain types of
///                                 bin contents (containing efficiencies, asymmetries, or ratio is general)
///                                 you should import the absolute value and set impDens to false
///
///
/// <tr><td> Weight(double)          <td> Apply given weight factor when importing histograms
///
/// <tr><td> Index(RooCategory&)       <td> Prepare import of multiple TH1/1/2/3 into a N+1 dimensional RooDataHist
///                              where the extra discrete dimension labels the source of the imported histogram
///                              If the index category defines states for which no histogram is be imported
///                              the corresponding bins will be left empty.
///
/// <tr><td> Import(const char*, TH1&) <td> Import a THx to be associated with the given state name of the index category
///                              specified in Index(). If the given state name is not yet defined in the index
///                              category it will be added on the fly. The import command can be specified
///                              multiple times.
/// <tr><td> Import(map<string,TH1*>&) <td> As above, but allows specification of many imports in a single operation
/// <tr><td> `GlobalObservables(const RooArgSet&)`      <td> Define the set of global observables to be stored in this RooDataHist.
///                                                          A snapshot of the passed RooArgSet is stored, meaning the values wont't change unexpectedly.
/// </table>
///

RooDataHist::RooDataHist(RooStringView name, RooStringView title, const RooArgList& vars, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3,
          const RooCmdArg& arg4,const RooCmdArg& arg5,const RooCmdArg& arg6,const RooCmdArg& arg7,const RooCmdArg& arg8) :
  RooAbsData(name,title,RooArgSet(vars,(RooAbsArg*)RooCmdConfig::decodeObjOnTheFly("RooDataHist::RooDataHist", "IndexCat",0,0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8)))
{
  // Initialize datastore
  _dstore = makeDefaultDataStore(name, title, _vars);

  // Define configuration for this method
  RooCmdConfig pc(Form("RooDataHist::ctor(%s)",GetName())) ;
  pc.defineObject("impHist","ImportHisto",0) ;
  pc.defineInt("impDens","ImportHisto",0) ;
  pc.defineObject("indexCat","IndexCat",0) ;
  pc.defineObject("impSliceHist","ImportHistoSlice",0,0,true) ; // array
  pc.defineString("impSliceState","ImportHistoSlice",0,"",true) ; // array
  pc.defineObject("impSliceDHist","ImportDataHistSlice",0,0,true) ; // array
  pc.defineString("impSliceDState","ImportDataHistSlice",0,"",true) ; // array
  pc.defineDouble("weight","Weight",0,1) ;
  pc.defineObject("dummy1","ImportDataHistSliceMany",0) ;
  pc.defineObject("dummy2","ImportHistoSliceMany",0) ;
  pc.defineSet("glObs","GlobalObservables",0,0) ;
  pc.defineMutex("ImportHisto","ImportHistoSlice","ImportDataHistSlice") ;
  pc.defineDependency("ImportHistoSlice","IndexCat") ;
  pc.defineDependency("ImportDataHistSlice","IndexCat") ;

  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  // Process & check varargs
  pc.process(l) ;
  if (!pc.ok(true)) {
    throw std::invalid_argument("Invalid command arguments passed to RooDataHist constructor!");
  }

  if(pc.getSet("glObs")) setGlobalObservables(*pc.getSet("glObs"));

  TH1* impHist = static_cast<TH1*>(pc.getObject("impHist")) ;
  bool impDens = pc.getInt("impDens") ;
  double initWgt = pc.getDouble("weight") ;
  const char* impSliceNames = pc.getString("impSliceState","",true) ;
  const RooLinkedList& impSliceHistos = pc.getObjectList("impSliceHist") ;
  RooCategory* indexCat = static_cast<RooCategory*>(pc.getObject("indexCat")) ;
  const char* impSliceDNames = pc.getString("impSliceDState","",true) ;
  const RooLinkedList& impSliceDHistos = pc.getObjectList("impSliceDHist") ;


  if (impHist) {

    // Initialize importing contents from TH1
    importTH1(vars,*impHist,initWgt, impDens) ;

  } else if (indexCat) {


    if (!impSliceHistos.empty()) {

      // Initialize importing mapped set of TH1s
      std::map<std::string,TH1*> hmap ;
      auto hiter = impSliceHistos.begin() ;
      for (const auto& token : ROOT::Split(impSliceNames, ",")) {
        auto histo = static_cast<TH1*>(*hiter);
        assert(histo);
        hmap[token] = histo;
        ++hiter;
      }
      importTH1Set(vars,*indexCat,hmap,initWgt,false) ;
    } else {

      // Initialize importing mapped set of RooDataHists
      std::map<std::string,RooDataHist*> dmap ;
      auto hiter = impSliceDHistos.begin() ;
      for (const auto& token : ROOT::Split(impSliceDNames, ",")) {
        dmap[token] = static_cast<RooDataHist*>(*hiter);
        ++hiter;
      }
      importDHistSet(vars,*indexCat,dmap,initWgt) ;
    }


  } else {

    // Initialize empty
    initialize() ;
    appendToDir(this,true) ;

  }

  registerWeightArraysToDataStore();
  TRACE_CREATE

}




////////////////////////////////////////////////////////////////////////////////
/// Import data from given TH1/2/3 into this RooDataHist

void RooDataHist::importTH1(const RooArgList& vars, const TH1& histo, double wgt, bool doDensityCorrection)
{
  // Adjust binning of internal observables to match that of input THx
  Int_t offset[3]{0, 0, 0};
  adjustBinning(vars, histo, offset) ;

  // Initialize internal data structure
  initialize() ;
  appendToDir(this,true) ;

  // Define x,y,z as 1st, 2nd and 3rd observable
  RooRealVar* xvar = (RooRealVar*) _vars.find(vars.at(0)->GetName()) ;
  RooRealVar* yvar = (RooRealVar*) (vars.at(1) ? _vars.find(vars.at(1)->GetName()) : 0 ) ;
  RooRealVar* zvar = (RooRealVar*) (vars.at(2) ? _vars.find(vars.at(2)->GetName()) : 0 ) ;

  // Transfer contents
  Int_t xmin(0),ymin(0),zmin(0) ;
  RooArgSet vset(*xvar) ;
  xmin = offset[0] ;
  if (yvar) {
    vset.add(*yvar) ;
    ymin = offset[1] ;
  }
  if (zvar) {
    vset.add(*zvar) ;
    zmin = offset[2] ;
  }

  Int_t ix(0),iy(0),iz(0) ;
  for (ix=0 ; ix < xvar->getBins() ; ix++) {
    xvar->setBin(ix) ;
    if (yvar) {
      for (iy=0 ; iy < yvar->getBins() ; iy++) {
        yvar->setBin(iy) ;
        if (zvar) {
          for (iz=0 ; iz < zvar->getBins() ; iz++) {
            zvar->setBin(iz) ;
            double bv = doDensityCorrection ? binVolume(vset) : 1;
            add(vset,bv*histo.GetBinContent(ix+1+xmin,iy+1+ymin,iz+1+zmin)*wgt,bv*TMath::Power(histo.GetBinError(ix+1+xmin,iy+1+ymin,iz+1+zmin)*wgt,2)) ;
          }
        } else {
          double bv = doDensityCorrection ? binVolume(vset) : 1;
          add(vset,bv*histo.GetBinContent(ix+1+xmin,iy+1+ymin)*wgt,bv*TMath::Power(histo.GetBinError(ix+1+xmin,iy+1+ymin)*wgt,2)) ;
        }
      }
    } else {
      double bv = doDensityCorrection ? binVolume(vset) : 1 ;
      add(vset,bv*histo.GetBinContent(ix+1+xmin)*wgt,bv*TMath::Power(histo.GetBinError(ix+1+xmin)*wgt,2)) ;
    }
  }

}

namespace {
bool checkConsistentAxes(const TH1* first, const TH1* second) {
  return first->GetDimension() == second->GetDimension()
      && first->GetNbinsX() == second->GetNbinsX()
      && first->GetNbinsY() == second->GetNbinsY()
      && first->GetNbinsZ() == second->GetNbinsZ()
      && first->GetXaxis()->GetXmin() == second->GetXaxis()->GetXmin()
      && first->GetXaxis()->GetXmax() == second->GetXaxis()->GetXmax()
      && (first->GetNbinsY() == 1 || (first->GetYaxis()->GetXmin() == second->GetYaxis()->GetXmin()
                                   && first->GetYaxis()->GetXmax() == second->GetYaxis()->GetXmax() ) )
      && (first->GetNbinsZ() == 1 || (first->GetZaxis()->GetXmin() == second->GetZaxis()->GetXmin()
                                   && first->GetZaxis()->GetXmax() == second->GetZaxis()->GetXmax() ) );
}

bool hasConsistentLayoutAndBinning(RooDataHist const& h1, RooDataHist const& h2) {

  // Relative tolerance for bin boundary comparison
  constexpr double tolerance = 1e-6;

  auto const& vars1 = *h1.get();
  auto const& vars2 = *h2.get();

  // Check if numer of variables and names is consistent
  if(!vars1.hasSameLayout(vars2)) {
    return false;
  }

  for(std::size_t iVar = 0; iVar < vars1.size(); ++iVar) {
    auto * var1 = dynamic_cast<RooRealVar*>(vars1[iVar]);
    auto * var2 = dynamic_cast<RooRealVar*>(vars2[iVar]);

    // Check if variables are consistently real-valued
    if((!var1 && var2) || (var1 && !var2)) return false;

    // Not a real-valued variable
    if(!var1) continue;

    // Now check the binning
    auto const& bng1 = var1->getBinning();
    auto const& bng2 = var2->getBinning();

    // Compare bin numbers
    if(bng1.numBins() != bng2.numBins()) return false;

    std::size_t nBins = bng1.numBins();

    // Compare bin boundaries
    for(std::size_t iBin = 0; iBin < nBins; ++iBin) {
      double v1 = bng1.binLow(iBin);
      double v2 = bng2.binLow(iBin);
      if(std::abs((v1 - v2) / v1) > tolerance) return false;
    }
    double v1 = bng1.binHigh(nBins - 1);
    double v2 = bng2.binHigh(nBins - 1);
    if(std::abs((v1 - v2) / v1) > tolerance) return false;
  }
  return true;
}
}


////////////////////////////////////////////////////////////////////////////////
/// Import data from given set of TH1/2/3 into this RooDataHist. The category indexCat labels the sources
/// in the constructed RooDataHist. The stl map provides the mapping between the indexCat state labels
/// and the import source

void RooDataHist::importTH1Set(const RooArgList& vars, RooCategory& indexCat, std::map<string,TH1*> hmap, double wgt, bool doDensityCorrection)
{
  RooCategory* icat = (RooCategory*) _vars.find(indexCat.GetName()) ;

  TH1* histo(0) ;
  bool init(false) ;
  for (const auto& hiter : hmap) {
    // Store pointer to first histogram from which binning specification will be taken
    if (!histo) {
      histo = hiter.second;
    } else {
      if (!checkConsistentAxes(histo, hiter.second)) {
        coutE(InputArguments) << "Axes of histogram " << hiter.second->GetName() << " are not consistent with first processed "
            << "histogram " << histo->GetName() << std::endl;
        throw std::invalid_argument("Axes of inputs for RooDataHist are inconsistent");
      }
    }
    // Define state labels in index category (both in provided indexCat and in internal copy in dataset)
    if (!indexCat.hasLabel(hiter.first)) {
      indexCat.defineType(hiter.first) ;
      coutI(InputArguments) << "RooDataHist::importTH1Set(" << GetName() << ") defining state \"" << hiter.first << "\" in index category " << indexCat.GetName() << endl ;
    }
    if (!icat->hasLabel(hiter.first)) {
      icat->defineType(hiter.first) ;
    }
  }

  // Check consistency in number of dimensions
  if (histo && (vars.getSize() != histo->GetDimension())) {
    coutE(InputArguments) << "RooDataHist::importTH1Set(" << GetName() << "): dimension of input histogram must match "
           << "number of continuous variables" << endl ;
    throw std::invalid_argument("Inputs histograms for RooDataHist are not compatible with dimensions of variables.");
  }

  // Copy bins and ranges from THx to dimension observables
  Int_t offset[3] ;
  adjustBinning(vars,*histo,offset) ;

  // Initialize internal data structure
  if (!init) {
    initialize() ;
    appendToDir(this,true) ;
    init = true ;
  }

  // Define x,y,z as 1st, 2nd and 3rd observable
  RooRealVar* xvar = (RooRealVar*) _vars.find(vars.at(0)->GetName()) ;
  RooRealVar* yvar = (RooRealVar*) (vars.at(1) ? _vars.find(vars.at(1)->GetName()) : 0 ) ;
  RooRealVar* zvar = (RooRealVar*) (vars.at(2) ? _vars.find(vars.at(2)->GetName()) : 0 ) ;

  // Transfer contents
  Int_t xmin(0),ymin(0),zmin(0) ;
  RooArgSet vset(*xvar) ;
  double volume = xvar->getMax()-xvar->getMin() ;
  xmin = offset[0] ;
  if (yvar) {
    vset.add(*yvar) ;
    ymin = offset[1] ;
    volume *= (yvar->getMax()-yvar->getMin()) ;
  }
  if (zvar) {
    vset.add(*zvar) ;
    zmin = offset[2] ;
    volume *= (zvar->getMax()-zvar->getMin()) ;
  }
  double avgBV = volume / numEntries() ;

  Int_t ic(0),ix(0),iy(0),iz(0) ;
  for (ic=0 ; ic < icat->numBins(0) ; ic++) {
    icat->setBin(ic) ;
    histo = hmap[icat->getCurrentLabel()] ;
    for (ix=0 ; ix < xvar->getBins() ; ix++) {
      xvar->setBin(ix) ;
      if (yvar) {
        for (iy=0 ; iy < yvar->getBins() ; iy++) {
          yvar->setBin(iy) ;
          if (zvar) {
            for (iz=0 ; iz < zvar->getBins() ; iz++) {
              zvar->setBin(iz) ;
              double bv = doDensityCorrection ? binVolume(vset)/avgBV : 1;
              add(vset,bv*histo->GetBinContent(ix+1+xmin,iy+1+ymin,iz+1+zmin)*wgt,bv*TMath::Power(histo->GetBinError(ix+1+xmin,iy+1+ymin,iz+1+zmin)*wgt,2)) ;
            }
          } else {
            double bv = doDensityCorrection ? binVolume(vset)/avgBV : 1;
            add(vset,bv*histo->GetBinContent(ix+1+xmin,iy+1+ymin)*wgt,bv*TMath::Power(histo->GetBinError(ix+1+xmin,iy+1+ymin)*wgt,2)) ;
          }
        }
      } else {
        double bv = doDensityCorrection ? binVolume(vset)/avgBV : 1;
        add(vset,bv*histo->GetBinContent(ix+1+xmin)*wgt,bv*TMath::Power(histo->GetBinError(ix+1+xmin)*wgt,2)) ;
      }
    }
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Import data from given set of TH1/2/3 into this RooDataHist. The category indexCat labels the sources
/// in the constructed RooDataHist. The stl map provides the mapping between the indexCat state labels
/// and the import source

void RooDataHist::importDHistSet(const RooArgList& /*vars*/, RooCategory& indexCat, std::map<std::string,RooDataHist*> dmap, double initWgt)
{
  auto* icat = static_cast<RooCategory*>(_vars.find(indexCat.GetName()));

  RooDataHist* dhistForBinning = nullptr;

  for (const auto& diter : dmap) {

    std::string const& label = diter.first;
    RooDataHist* dhist = diter.second ;

    if(!dhistForBinning) {
      dhistForBinning = dhist;
    }
    else {
      if(!hasConsistentLayoutAndBinning(*dhistForBinning, *dhist)) {
        coutE(InputArguments) << "Layout or binning of histogram " << dhist->GetName() << " is not consistent with first processed "
            << "histogram " << dhistForBinning->GetName() << std::endl;
        throw std::invalid_argument("Layout or binning of inputs for RooDataHist is inconsistent");
      }
    }

    // Define state labels in index category (both in provided indexCat and in internal copy in dataset)
    if (!indexCat.hasLabel(label)) {
      indexCat.defineType(label) ;
      coutI(InputArguments) << "RooDataHist::importDHistSet(" << GetName() << ") defining state \"" << label << "\" in index category " << indexCat.GetName() << endl ;
    }
    if (!icat->hasLabel(label)) {
      icat->defineType(label) ;
    }
  }

  // adjust the binning of the created histogram
  for(auto * theirVar : dynamic_range_cast<RooRealVar*>(dhistForBinning->_vars)) {
    auto * ourVar = dynamic_cast<RooRealVar*>(_vars.find(theirVar->GetName()));
    if(!theirVar || !ourVar) continue;
    ourVar->setBinning(theirVar->getBinning());
  }

  initialize() ;
  appendToDir(this,true) ;


  for (const auto& diter : dmap) {
    std::string const& label = diter.first;
    RooDataHist* dhist = diter.second ;

    icat->setLabel(label.c_str()) ;

    // Transfer contents
    for (Int_t i=0 ; i<dhist->numEntries() ; i++) {
      _vars.assign(*dhist->get(i)) ;
      add(_vars,dhist->weight()*initWgt, pow(dhist->weightError(SumW2),2) ) ;
    }

  }
}

////////////////////////////////////////////////////////////////////////////////
/// Helper doing the actual work of adjustBinning().

void RooDataHist::_adjustBinning(RooRealVar &theirVar, const TAxis &axis,
    RooRealVar *ourVar, Int_t *offset)
{
   const std::string ourVarName(ourVar->GetName() ? ourVar->GetName() : ""), ownName(GetName() ? GetName() : "");
   // RooRealVar is derived from RooAbsRealLValue which is itself
   // derived from RooAbsReal and a virtual class RooAbsLValue
   // supplying setter fuctions, check if ourVar is indeed derived
   // as real
   if (!dynamic_cast<RooAbsReal *>(ourVar)) {
      coutE(InputArguments) << "RooDataHist::adjustBinning(" << ownName << ") ERROR: dimension " << ourVarName
                            << " must be real\n";
      throw std::logic_error("Incorrect type object (" + ourVarName +
                             ") passed as argument to RooDataHist::_adjustBinning. Please report this issue.");
   }

  const double xlo = theirVar.getMin();
  const double xhi = theirVar.getMax();

  if (axis.GetXbins()->GetArray()) {
    RooBinning xbins(axis.GetNbins(), axis.GetXbins()->GetArray());

    const double tolerance = 1e-6 * xbins.averageBinWidth();

    // Adjust xlo/xhi to nearest boundary
    const double xloAdj = xbins.binLow(xbins.binNumber(xlo + tolerance));
    const double xhiAdj = xbins.binHigh(xbins.binNumber(xhi - tolerance));
    xbins.setRange(xloAdj, xhiAdj);

    theirVar.setBinning(xbins);

    if (true || std::abs(xloAdj - xlo) > tolerance || std::abs(xhiAdj - xhi) > tolerance) {
       coutI(DataHandling) << "RooDataHist::adjustBinning(" << ownName << "): fit range of variable " << ourVarName
                           << " expanded to nearest bin boundaries: [" << xlo << "," << xhi << "] --> [" << xloAdj
                           << "," << xhiAdj << "]"
                           << "\n";
    }

    ourVar->setBinning(xbins);

    if (offset) {
      *offset = xbins.rawBinNumber(xloAdj + tolerance);
    }
  } else {
    RooBinning xbins(axis.GetXmin(), axis.GetXmax());
    xbins.addUniform(axis.GetNbins(), axis.GetXmin(), axis.GetXmax());

    const double tolerance = 1e-6 * xbins.averageBinWidth();

    // Adjust xlo/xhi to nearest boundary
    const double xloAdj = xbins.binLow(xbins.binNumber(xlo + tolerance));
    const double xhiAdj = xbins.binHigh(xbins.binNumber(xhi - tolerance));
    xbins.setRange(xloAdj, xhiAdj);
    theirVar.setRange(xloAdj, xhiAdj);

    if (std::abs(xloAdj - xlo) > tolerance || std::abs(xhiAdj - xhi) > tolerance) {
       coutI(DataHandling) << "RooDataHist::adjustBinning(" << ownName << "): fit range of variable " << ourVarName
                           << " expanded to nearest bin boundaries: [" << xlo << "," << xhi << "] --> [" << xloAdj
                           << "," << xhiAdj << "]"
                           << "\n";
    }

    RooUniformBinning xbins2(xloAdj, xhiAdj, xbins.numBins());
    ourVar->setBinning(xbins2);

    if (offset) {
      *offset = xbins.rawBinNumber(xloAdj + tolerance);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Adjust binning specification on first and optionally second and third
/// observable to binning in given reference TH1. Used by constructors
/// that import data from an external TH1.
/// Both the variables in vars and in this RooDataHist are adjusted.
/// @param vars List with variables that are supposed to have their binning adjusted.
/// @param href Reference histogram that dictates the binning
/// @param offset If not nullptr, a possible bin count offset for the axes x,y,z is saved here as Int_t[3]

void RooDataHist::adjustBinning(const RooArgList& vars, const TH1& href, Int_t* offset)
{
  auto xvar = static_cast<RooRealVar*>( _vars.find(*vars.at(0)) );
  _adjustBinning(*static_cast<RooRealVar*>(vars.at(0)), *href.GetXaxis(), xvar, offset ? &offset[0] : nullptr);

  if (vars.at(1)) {
    auto yvar = static_cast<RooRealVar*>(_vars.find(*vars.at(1)));
    if (yvar)
      _adjustBinning(*static_cast<RooRealVar*>(vars.at(1)), *href.GetYaxis(), yvar, offset ? &offset[1] : nullptr);
  }

  if (vars.at(2)) {
    auto zvar = static_cast<RooRealVar*>(_vars.find(*vars.at(2)));
    if (zvar)
      _adjustBinning(*static_cast<RooRealVar*>(vars.at(2)), *href.GetZaxis(), zvar, offset ? &offset[2] : nullptr);
  }

}


namespace {
/// Clone external weight arrays, unless the external array is nullptr.
void cloneArray(double*& ours, const double* theirs, std::size_t n) {
  if (ours) delete[] ours;
  ours = nullptr;
  if (!theirs) return;
  ours = new double[n];
  std::copy(theirs, theirs+n, ours);
}

/// Allocate and initialise an array with desired size and values.
void initArray(double*& arr, std::size_t n, double val) {
  if (arr) delete[] arr;
  arr = nullptr;
  if (n == 0) return;
  arr = new double[n];
  std::fill(arr, arr+n, val);
}
}


////////////////////////////////////////////////////////////////////////////////
/// Initialization procedure: allocate weights array, calculate
/// multipliers needed for N-space to 1-dim array jump table,
/// and fill the internal tree with all bin center coordinates

void RooDataHist::initialize(const char* binningName, bool fillTree)
{
  _lvvars.clear();
  _lvbins.clear();

  // Fill array of LValue pointers to variables
  for (unsigned int i = 0; i < _vars.size(); ++i) {
    if (binningName) {
      RooRealVar* rrv = dynamic_cast<RooRealVar*>(_vars[i]);
      if (rrv) {
        rrv->setBinning(rrv->getBinning(binningName));
      }
    }

    auto lvarg = dynamic_cast<RooAbsLValue*>(_vars[i]);
    assert(lvarg);
    _lvvars.push_back(lvarg);

    const RooAbsBinning* binning = lvarg->getBinningPtr(nullptr);
    _lvbins.emplace_back(binning ? binning->clone() : nullptr);
  }


  // Allocate coefficients array
  _idxMult.resize(_vars.size()) ;

  _arrSize = 1 ;
  unsigned int n = 0u;
  for (const auto var : _vars) {
    auto arg = dynamic_cast<const RooAbsLValue*>(var);
    assert(arg);

    // Calculate sub-index multipliers for master index
    for (unsigned int i = 0u; i<n; i++) {
      _idxMult[i] *= arg->numBins() ;
    }
    _idxMult[n++] = 1 ;

    // Calculate dimension of weight array
    _arrSize *= arg->numBins() ;
  }

  // Allocate and initialize weight array if necessary
  if (!_wgt) {
    initArray(_wgt, _arrSize, 0.);
    delete[] _errLo; _errLo = nullptr;
    delete[] _errHi; _errHi = nullptr;
    delete[] _sumw2; _sumw2 = nullptr;
    initArray(_binv, _arrSize, 0.);

    // Refill array pointers in data store when reading
    // from Streamer
    if (!fillTree) {
      registerWeightArraysToDataStore();
    }
  }

  if (!fillTree) return ;

  // Fill TTree with bin center coordinates
  // Calculate plot bins of components from master index

  for (Int_t ibin=0 ; ibin < _arrSize ; ibin++) {
    Int_t j(0), idx(0), tmp(ibin) ;
    double theBinVolume(1) ;
    for (auto arg2 : _lvvars) {
      idx  = tmp / _idxMult[j] ;
      tmp -= idx*_idxMult[j++] ;
      arg2->setBin(idx) ;
      theBinVolume *= arg2->getBinWidth(idx) ;
    }
    _binv[ibin] = theBinVolume ;

    fill() ;
  }


}


////////////////////////////////////////////////////////////////////////////////

void RooDataHist::checkBinBounds() const
{
  if (!_binbounds.empty()) return;
  for (auto& it : _lvbins) {
    _binbounds.push_back(std::vector<double>());
    if (it) {
      std::vector<double>& bounds = _binbounds.back();
      bounds.reserve(2 * it->numBins());
      for (Int_t i = 0; i < it->numBins(); ++i) {
        bounds.push_back(it->binLow(i));
        bounds.push_back(it->binHigh(i));
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooDataHist::RooDataHist(const RooDataHist& other, const char* newname) :
  RooAbsData(other,newname), RooDirItem(), _arrSize(other._arrSize), _idxMult(other._idxMult), _pbinvCache(other._pbinvCache)
{
  // Allocate and initialize weight array
  assert(_arrSize == other._arrSize);
  cloneArray(_wgt, other._wgt, other._arrSize);
  cloneArray(_errLo, other._errLo, other._arrSize);
  cloneArray(_errHi, other._errHi, other._arrSize);
  cloneArray(_binv, other._binv, other._arrSize);
  cloneArray(_sumw2, other._sumw2, other._arrSize);

  // Fill array of LValue pointers to variables
  for (const auto rvarg : _vars) {
    auto lvarg = dynamic_cast<RooAbsLValue*>(rvarg);
    assert(lvarg);
    _lvvars.push_back(lvarg);
    const RooAbsBinning* binning = lvarg->getBinningPtr(0);
    _lvbins.emplace_back(binning ? binning->clone() : 0) ;
  }

  registerWeightArraysToDataStore();

 appendToDir(this,true) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Implementation of RooAbsData virtual method that drives the RooAbsData::reduce() methods

RooAbsData* RooDataHist::reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, const char* cutRange,
    std::size_t nStart, std::size_t nStop)
{
  checkInit() ;
  RooArgSet myVarSubset;
  _vars.selectCommon(varSubset, myVarSubset);
  RooDataHist *rdh = new RooDataHist(GetName(), GetTitle(), myVarSubset);

  RooFormulaVar* cloneVar = 0;
  std::unique_ptr<RooArgSet> tmp;
  if (cutVar) {
    // Deep clone cutVar and attach clone to this dataset
    tmp.reset(static_cast<RooArgSet*>(RooArgSet(*cutVar).snapshot()));
    if (!tmp) {
      coutE(DataHandling) << "RooDataHist::reduceEng(" << GetName() << ") Couldn't deep-clone cut variable, abort," << endl ;
      return nullptr;
    }
    cloneVar = (RooFormulaVar*) tmp->find(*cutVar) ;
    cloneVar->attachDataSet(*this) ;
  }

  double lo,hi ;
  const std::size_t nevt = nStop < static_cast<std::size_t>(numEntries()) ? nStop : static_cast<std::size_t>(numEntries());
  for (auto i=nStart; i<nevt ; i++) {
    const RooArgSet* row = get(i) ;

    bool doSelect(true) ;
    if (cutRange) {
      for (const auto arg : *row) {
        if (!arg->inRange(cutRange)) {
          doSelect = false ;
          break ;
        }
      }
    }
    if (!doSelect) continue ;

    if (!cloneVar || cloneVar->getVal()) {
      weightError(lo,hi,SumW2) ;
      rdh->add(*row,weight(),lo*lo) ;
    }
  }

  return rdh ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooDataHist::~RooDataHist()
{
   delete[] _wgt;
   delete[] _errLo;
   delete[] _errHi;
   delete[] _sumw2;
   delete[] _binv;

   removeFromDir(this) ;
  TRACE_DESTROY
}




////////////////////////////////////////////////////////////////////////////////
/// Calculate bin number of the given coordinates. If only a subset of the internal
/// coordinates are passed, the missing coordinates are taken at their current value.
/// \param[in] coord Variables that are representing the coordinates.
/// \param[in] fast If the variables in `coord` and the ones of the data hist have the
/// same size and layout, `fast` can be set to skip checking that all variables are
/// present in `coord`.
Int_t RooDataHist::getIndex(const RooAbsCollection& coord, bool fast) const {
  checkInit() ;
  return calcTreeIndex(coord, fast);
}




////////////////////////////////////////////////////////////////////////////////
/// Calculate the bin index corresponding to the coordinates passed as argument.
/// \param[in] coords Coordinates. If `fast == false`, these can be partial.
/// \param[in] fast   Promise that the coordinates in `coords` have the same order
/// as the internal coordinates. In this case, values are looked up only by index.
std::size_t RooDataHist::calcTreeIndex(const RooAbsCollection& coords, bool fast) const
{
  // With fast, caller promises that layout of `coords` is identical to our internal `vars`.
  // Previously, this was verified with an assert in debug mode like this:
  //
  //    assert(!fast || coords.hasSameLayout(_vars));
  //
  // However, there are usecases where the externally provided `coords` have
  // different names than the internal variables, even though they correspond
  // to each other. For example, if the observables in the computation graph
  // are renamed with `redirectServers`. Hence, we can't do a meaningful assert
  // here.

  if (&_vars == &coords)
    fast = true;

  std::size_t masterIdx = 0;

  for (unsigned int i=0; i < _vars.size(); ++i) {
    const RooAbsArg* internalVar = _vars[i];
    const RooAbsBinning* binning = _lvbins[i].get();

    // Find the variable that we need values from.
    // That's either the variable directly from the external coordinates
    // or we find the external one that has the same name as "internalVar".
    const RooAbsArg* theVar = fast ? coords[i] : coords.find(*internalVar);
    if (!theVar) {
      // Variable is not in external coordinates. Use current internal value.
      theVar = internalVar;
    }
    // If fast is on, users promise that the sets have the same layout:
    //
    //   assert(!fast || strcmp(internalVar->GetName(), theVar->GetName()) == 0);
    //
    // This assert is commented out for the same reasons that applied to the
    // other assert explained above.

    if (binning) {
      assert(dynamic_cast<const RooAbsReal*>(theVar));
      const double val = static_cast<const RooAbsReal*>(theVar)->getVal();
      masterIdx += _idxMult[i] * binning->binNumber(val);
    } else {
      // We are a category. No binning.
      assert(dynamic_cast<const RooAbsCategoryLValue*>(theVar));
      auto cat = static_cast<const RooAbsCategoryLValue*>(theVar);
      masterIdx += _idxMult[i] * cat->getBin(static_cast<const char*>(nullptr));
    }
  }

  return masterIdx ;
}


////////////////////////////////////////////////////////////////////////////////
/// Back end function to plotting functionality. Plot RooDataHist on given
/// frame in mode specified by plot options 'o'. The main purpose of
/// this function is to match the specified binning on 'o' to the
/// internal binning of the plot observable in this RooDataHist.
/// \see RooAbsData::plotOn() for plotting options.
RooPlot *RooDataHist::plotOn(RooPlot *frame, PlotOpt o) const
{
  checkInit() ;
  if (o.bins) return RooAbsData::plotOn(frame,o) ;

  if(!frame) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return nullptr;
  }
  auto var= static_cast<RooAbsRealLValue*>(frame->getPlotVar());
  if(!var) {
    coutE(InputArguments) << ClassName() << "::" << GetName()
    << ":plotOn: frame does not specify a plot variable" << endl;
    return nullptr;
  }

  auto dataVar = static_cast<RooRealVar*>(_vars.find(*var));
  if (!dataVar) {
    coutE(InputArguments) << ClassName() << "::" << GetName()
    << ":plotOn: dataset doesn't contain plot frame variable" << endl;
    return nullptr;
  }

  o.bins = &dataVar->getBinning() ;
  return RooAbsData::plotOn(frame,o) ;
}


////////////////////////////////////////////////////////////////////////////////
/// A vectorized version of interpolateDim for boundary safe quadratic
/// interpolation of one dimensional histograms.
///
/// \param[out] output An array of interpolated weights corresponding to the
///                    values in xVals.
/// \param[in] xVals An array of event coordinates for which the weights should be
///                  calculated.
/// \param[in] correctForBinSize Enable the inverse bin volume correction factor.
/// \param[in] cdfBoundaries Enable the special boundary coundition for a cdf:
///                          Underflow bins are assumed to have weight zero and
///                          overflow bins have weight one. Otherwise, the
///                          histogram is mirrored at the boundaries for the
///                          interpolation.

void RooDataHist::interpolateQuadratic(double* output, RooSpan<const double> xVals,
                                       bool correctForBinSize, bool cdfBoundaries)
{
  const std::size_t nBins = numEntries();
  const std::size_t nEvents = xVals.size();

  RooAbsBinning const& binning = *_lvbins[0];
  // Reuse the output buffer for bin indices and zero-initialize it
  auto binIndices = reinterpret_cast<int*>(output + nEvents) - nEvents;
  std::fill(binIndices, binIndices + nEvents, 0);
  binning.binNumbers(xVals.data(), binIndices, nEvents);

  // Extend coordinates and weights with one extra point before the first bin
  // and one extra point after the last bin. This means the original histogram
  // bins span elements 1 to nBins in coordsExt and weightsExt
  std::vector<double> coordsExt(nBins+3);
  double* binCoords = coordsExt.data() + 2;
  binCoords[0] = binning.lowBound() + 0.5*_binv[0];
  for (std::size_t binIdx = 1; binIdx < nBins ; ++binIdx) {
    if (binning.isUniform()) {
      double binWidth = _binv[0];
      binCoords[binIdx] = binIdx*binWidth + binCoords[0];
    }
    else {
      double binCentDiff = 0.5*_binv[binIdx-1] + 0.5*_binv[binIdx];
      binCoords[binIdx] = binCoords[binIdx-1] + binCentDiff;
    }
  }

  std::vector<double> weightsExt(nBins+3);
  // Fill weights for bins that are inside histogram boundaries
  for (std::size_t binIdx = 0; binIdx < nBins; ++binIdx) {
    weightsExt[binIdx+2] = correctForBinSize ? _wgt[binIdx] / _binv[binIdx] : _wgt[binIdx];
  }

  if (cdfBoundaries) {
    coordsExt[0] = - 1e-10 + binning.lowBound();
    weightsExt[0] = 0.;

    coordsExt[1] = binning.lowBound();
    weightsExt[1] = 0.;

    coordsExt[nBins+2] = binning.highBound();
    weightsExt[nBins+2] = 1.;
  }
  else {
    // Mirror first two bins and last bin
    coordsExt[0] = binCoords[1] - 2*_binv[0] - _binv[1];
    weightsExt[0] = weightsExt[3];

    coordsExt[1] = binCoords[0] - _binv[0];
    weightsExt[1] = weightsExt[2];

    coordsExt[nBins+2] = binCoords[nBins-1] + _binv[nBins-1];
    weightsExt[nBins+2] = weightsExt[nBins+1];
  }

  // We use the current bin center and two bin centers on the left for
  // interpolation if xVal is to the left of the current bin center
  for (std::size_t i = 0; i < nEvents ; ++i) {
    double xVal = xVals[i];
    std::size_t binIdx = binIndices[i] + 2;

    // If xVal is to the right of the current bin center, shift all bin
    // coordinates one step to the right and use that for the interpolation
    if (xVal > coordsExt[binIdx]) {
        binIdx += 1;
    }

    double x1 = coordsExt[binIdx-2];
    double y1 = weightsExt[binIdx-2];

    double x2 = coordsExt[binIdx-1];
    double y2 = weightsExt[binIdx-1];

    double x3 = coordsExt[binIdx];
    double y3 = weightsExt[binIdx];

    // Evaluate a few repeated factors
    double quotient = (x3-x1) / (x2-x1);
    double x1Sqrd = x1*x1;
    double x3Sqrd = x3*x3;
    // Solve coefficients in system of three quadratic equations!
    double secondCoeff = (y3 - y1 - (y2-y1) * quotient) / (x3Sqrd - x1Sqrd - (x2*x2 - x1Sqrd) * quotient);
    double firstCoeff = (y3 - y1 - secondCoeff*(x3Sqrd - x1Sqrd)) / (x3-x1);
    double zerothCoeff = y1 - secondCoeff*x1Sqrd - firstCoeff*x1;
    // Get the interpolated weight using the equation of a second degree polynomial
    output[i] = secondCoeff * xVal * xVal + firstCoeff * xVal + zerothCoeff;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// A vectorized version of interpolateDim for boundary safe linear
/// interpolation of one dimensional histograms.
///
/// \param[out] output An array of interpolated weights corresponding to the
///                    values in xVals.
/// \param[in] xVals An array of event coordinates for which the weights should be
///                  calculated.
/// \param[in] correctForBinSize Enable the inverse bin volume correction factor.
/// \param[in] cdfBoundaries Enable the special boundary coundition for a cdf:
///                          Underflow bins are assumed to have weight zero and
///                          overflow bins have weight one. Otherwise, the
///                          histogram is mirrored at the boundaries for the
///                          interpolation.

void RooDataHist::interpolateLinear(double* output, RooSpan<const double> xVals,
                                    bool correctForBinSize, bool cdfBoundaries)
{
  const std::size_t nBins = numEntries();
  const std::size_t nEvents = xVals.size();

  RooAbsBinning const& binning = *_lvbins[0];
  // Reuse the output buffer for bin indices and zero-initialize it
  auto binIndices = reinterpret_cast<int*>(output + nEvents) - nEvents;
  std::fill(binIndices, binIndices + nEvents, 0);
  binning.binNumbers(xVals.data(), binIndices, nEvents);

  // Extend coordinates and weights with one extra point before the first bin
  // and one extra point after the last bin. This means the original histogram
  // bins span elements 1 to nBins in coordsExt and weightsExt
  std::vector<double> coordsExt(nBins+2);
  double* binCoords = coordsExt.data() + 1;
  binCoords[0] = binning.lowBound() + 0.5*_binv[0];
  for (std::size_t binIdx = 1; binIdx < nBins ; ++binIdx) {
    if (binning.isUniform()) {
      double binWidth = _binv[0];
      binCoords[binIdx] = binIdx*binWidth + binCoords[0];
    }
    else {
      double binCentDiff = 0.5*_binv[binIdx-1] + 0.5*_binv[binIdx];
      binCoords[binIdx] = binCoords[binIdx-1] + binCentDiff;
    }
  }

  std::vector<double> weightsExt(nBins+2);
  // Fill weights for bins that are inside histogram boundaries
  for (std::size_t binIdx = 0; binIdx < nBins; ++binIdx) {
    weightsExt[binIdx+1] = correctForBinSize ? _wgt[binIdx] / _binv[binIdx] : _wgt[binIdx];
  }

  // Fill weights for bins that are outside histogram boundaries
  if (cdfBoundaries) {
    coordsExt[0] = binning.lowBound();
    weightsExt[0] = 0.;
    coordsExt[nBins+1] = binning.highBound();
    weightsExt[nBins+1] = 1.;
  }
  else {
    // Mirror first and last bins
    coordsExt[0] = binCoords[0] - _binv[0];
    weightsExt[0] = weightsExt[1];
    coordsExt[nBins+1] = binCoords[nBins-1] + _binv[nBins-1];
    weightsExt[nBins+1] = weightsExt[nBins];
  }

  // Interpolate between current bin center and one bin center to the left
  // if xVal is to the left of the current bin center
  for (std::size_t i = 0; i < nEvents ; ++i) {
      double xVal = xVals[i];
      std::size_t binIdx = binIndices[i] + 1;

      // If xVal is to the right of the current bin center, interpolate between
      // current bin center and one bin center to the right instead
      if (xVal > coordsExt[binIdx]) { binIdx += 1; }

      double x1 = coordsExt[binIdx-1];
      double y1 = weightsExt[binIdx-1];
      double x2 = coordsExt[binIdx];
      double y2 = weightsExt[binIdx];

      // Find coefficients by solving a system of two linear equations
      double firstCoeff = (y2-y1) / (x2-x1);
      double zerothCoeff = y1 - firstCoeff * x1;
      // Get the interpolated weight using the equation of a straight line
      output[i] = firstCoeff * xVal + zerothCoeff;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// A vectorized version of RooDataHist::weight() for one dimensional histograms
/// with up to one dimensional interpolation.
/// \param[out] output An array of weights corresponding the values in xVals.
/// \param[in] xVals An array of coordinates for which the weights should be
///                  calculated.
/// \param[in] intOrder Interpolation order; 0th and 1st order are supported.
/// \param[in] correctForBinSize Enable the inverse bin volume correction factor.
/// \param[in] cdfBoundaries Enable the special boundary coundition for a cdf:
///                          Underflow bins are assumed to have weight zero and
///                          overflow bins have weight one. Otherwise, the
///                          histogram is mirrored at the boundaries for the
///                          interpolation.

void RooDataHist::weights(double* output, RooSpan<double const> xVals, int intOrder, bool correctForBinSize, bool cdfBoundaries)
{
  auto const nEvents = xVals.size();

  if (intOrder == 0) {
    RooAbsBinning const& binning = *_lvbins[0];

    // Reuse the output buffer for bin indices and zero-initialize it
    auto binIndices = reinterpret_cast<int*>(output + nEvents) - nEvents;
    std::fill(binIndices, binIndices + nEvents, 0);
    binning.binNumbers(xVals.data(), binIndices, nEvents);

    for (std::size_t i=0; i < nEvents; ++i) {
      auto binIdx = binIndices[i];
      output[i] = correctForBinSize ? _wgt[binIdx] / _binv[binIdx] : _wgt[binIdx];
    }
  }
  else if (intOrder == 1) {
    interpolateLinear(output, xVals, correctForBinSize, cdfBoundaries);
  }
    else if (intOrder == 2) {
    interpolateQuadratic(output, xVals, correctForBinSize, cdfBoundaries);
  }
  else {
    // Higher dimensional scenarios not yet implemented
    coutE(InputArguments) << "RooDataHist::weights(" << GetName() << ") interpolation in "
                          << intOrder << " dimensions not yet implemented" << std::endl ;
    // Fall back to 1st order interpolation
    weights(output, xVals, 1, correctForBinSize, cdfBoundaries);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// A faster version of RooDataHist::weight that assumes the passed arguments
/// are aligned with the histogram variables.
/// \param[in] bin Coordinates for which the weight should be calculated.
///                Has to be aligned with the internal histogram variables.
/// \param[in] intOrder Interpolation order, i.e. how many neighbouring bins are
///                     used for the interpolation. If zero, the bare weight for
///                     the bin enclosing the coordinatesis returned.
/// \param[in] correctForBinSize Enable the inverse bin volume correction factor.
/// \param[in] cdfBoundaries Enable the special boundary coundition for a cdf:
///                          underflow bins are assumed to have weight zero and
///                          overflow bins have weight one. Otherwise, the
///                          histogram is mirrored at the boundaries for the
///                          interpolation.

double RooDataHist::weightFast(const RooArgSet& bin, Int_t intOrder, bool correctForBinSize, bool cdfBoundaries)
{
  checkInit() ;

  // Handle illegal intOrder values
  if (intOrder<0) {
    coutE(InputArguments) << "RooDataHist::weight(" << GetName() << ") ERROR: interpolation order must be positive" << endl ;
    return 0 ;
  }

  // Handle no-interpolation case
  if (intOrder==0) {
    const auto idx = calcTreeIndex(bin, true);
    return correctForBinSize ? _wgt[idx] / _binv[idx] : _wgt[idx];
  }

  // Handle all interpolation cases
  return weightInterpolated(bin, intOrder, correctForBinSize, cdfBoundaries);
}


////////////////////////////////////////////////////////////////////////////////
/// Return the weight at given coordinates with optional interpolation.
/// \param[in] bin Coordinates for which the weight should be calculated.
/// \param[in] intOrder Interpolation order, i.e. how many neighbouring bins are
///                     used for the interpolation. If zero, the bare weight for
///                     the bin enclosing the coordinatesis returned.
/// \param[in] correctForBinSize Enable the inverse bin volume correction factor.
/// \param[in] cdfBoundaries Enable the special boundary coundition for a cdf:
///                          underflow bins are assumed to have weight zero and
///                          overflow bins have weight one. Otherwise, the
///                          histogram is mirrored at the boundaries for the
///                          interpolation.
/// \param[in] oneSafe Ignored.

double RooDataHist::weight(const RooArgSet& bin, Int_t intOrder, bool correctForBinSize, bool cdfBoundaries, bool /*oneSafe*/)
{
  checkInit() ;

  // Handle illegal intOrder values
  if (intOrder<0) {
    coutE(InputArguments) << "RooDataHist::weight(" << GetName() << ") ERROR: interpolation order must be positive" << endl ;
    return 0 ;
  }

  // Handle no-interpolation case
  if (intOrder==0) {
    const auto idx = calcTreeIndex(bin, false);
    return correctForBinSize ? _wgt[idx] / _binv[idx] : _wgt[idx];
  }

  // Handle all interpolation cases
  _vars.assignValueOnly(bin) ;

  return weightInterpolated(_vars, intOrder, correctForBinSize, cdfBoundaries);
}


////////////////////////////////////////////////////////////////////////////////
/// Return the weight at given coordinates with interpolation.
/// \param[in] bin Coordinates for which the weight should be calculated.
///                Has to be aligned with the internal histogram variables.
/// \param[in] intOrder Interpolation order, i.e. how many neighbouring bins are
///                     used for the interpolation.
/// \param[in] correctForBinSize Enable the inverse bin volume correction factor.
/// \param[in] cdfBoundaries Enable the special boundary coundition for a cdf:
///                          underflow bins are assumed to have weight zero and
///                          overflow bins have weight one. Otherwise, the
///                          histogram is mirrored at the boundaries for the
///                          interpolation.

double RooDataHist::weightInterpolated(const RooArgSet& bin, int intOrder, bool correctForBinSize, bool cdfBoundaries) {
  VarInfo const& varInfo = getVarInfo();

  const auto centralIdx = calcTreeIndex(bin, true);

  double wInt{0} ;
  if (varInfo.nRealVars == 1) {

    // buffer needs to be 2 x (interpolation order + 1), with the factor 2 for x and y.
    _interpolationBuffer.resize(2 * intOrder + 2);

    // 1-dimensional interpolation
    auto const& realX = static_cast<RooRealVar const&>(*bin[varInfo.realVarIdx1]);
    wInt = interpolateDim(varInfo.realVarIdx1, realX.getVal(), centralIdx, intOrder, correctForBinSize, cdfBoundaries) ;

  } else if (varInfo.nRealVars == 2) {

    // buffer needs to be 2 x 2 x (interpolation order + 1), with one factor 2
    // for x and y, and the other for the number of dimensions.
    _interpolationBuffer.resize(4 * intOrder + 4);

    // 2-dimensional interpolation
    auto const& realX = static_cast<RooRealVar const&>(*bin[varInfo.realVarIdx1]);
    auto const& realY = static_cast<RooRealVar const&>(*bin[varInfo.realVarIdx2]);
    double xval = realX.getVal() ;
    double yval = realY.getVal() ;

    RooAbsBinning const& binningY = realY.getBinning();

    int ybinC = binningY.binNumber(yval) ;
    int ybinLo = ybinC-intOrder/2 - ((yval<binningY.binCenter(ybinC))?1:0) ;
    int ybinM = binningY.numBins() ;

    auto idxMultY = _idxMult[varInfo.realVarIdx2];
    auto offsetIdx = centralIdx - idxMultY * ybinC;

    // Use a class-member buffer to avoid repeated heap allocations.
    double * yarr = _interpolationBuffer.data() + 2 * intOrder + 2; // add offset to skip part reserved for other dim
    double * xarr = yarr + intOrder + 1;
    for (int i=ybinLo ; i<=intOrder+ybinLo ; i++) {
      int ibin ;
      if (i>=0 && i<ybinM) {
        // In range
        ibin = i ;
        xarr[i-ybinLo] = binningY.binCenter(ibin) ;
      } else if (i>=ybinM) {
        // Overflow: mirror
        ibin = 2*ybinM-i-1 ;
        xarr[i-ybinLo] = 2*binningY.highBound()-binningY.binCenter(ibin) ;
      } else {
        // Underflow: mirror
        ibin = -i -1;
        xarr[i-ybinLo] = 2*binningY.lowBound()-binningY.binCenter(ibin) ;
      }
      auto centralIdxX = offsetIdx + idxMultY * ibin;
      yarr[i-ybinLo] = interpolateDim(varInfo.realVarIdx1,xval,centralIdxX,intOrder,correctForBinSize,false) ;
    }

    if (gDebug>7) {
      cout << "RooDataHist interpolating data is" << endl ;
      cout << "xarr = " ;
      for (int q=0; q<=intOrder ; q++) cout << xarr[q] << " " ;
      cout << " yarr = " ;
      for (int q=0; q<=intOrder ; q++) cout << yarr[q] << " " ;
      cout << endl ;
    }
    wInt = RooMath::interpolate(xarr,yarr,intOrder+1,yval) ;

  } else {

    // Higher dimensional scenarios not yet implemented
    coutE(InputArguments) << "RooDataHist::weight(" << GetName() << ") interpolation in "
                          << varInfo.nRealVars << " dimensions not yet implemented" << endl ;
    return weightFast(bin,0,correctForBinSize,cdfBoundaries) ;

  }

  return wInt ;
}


void RooDataHist::initializeAsymErrArrays() const {
  if (!_errLo || !_errHi) {
    initArray(_errLo, _arrSize, -1.);
    initArray(_errHi, _arrSize, -1.);
    registerWeightArraysToDataStore();
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Return the asymmetric errors on the current weight.
/// \see weightError(ErrorType) const for symmetric error.
/// \param[out] lo Low error.
/// \param[out] hi High error.
/// \param[in] etype Type of error to compute. May throw if not supported.
/// Supported errors are
/// - `Poisson` Default. Asymmetric Poisson errors (68% CL).
/// - `SumW2` The square root of the sum of weights. (Symmetric).
/// - `None` Return zero.
void RooDataHist::weightError(double& lo, double& hi, ErrorType etype) const
{
  checkInit() ;

  switch (etype) {

  case Auto:
    throw std::invalid_argument(Form("RooDataHist::weightError(%s) error type Auto not allowed here",GetName())) ;
    break ;

  case Expected:
    throw std::invalid_argument(Form("RooDataHist::weightError(%s) error type Expected not allowed here",GetName())) ;
    break ;

  case Poisson:
    if (_errLo && _errLo[_curIndex] >= 0.0) {
      // Weight is preset or precalculated
      lo = _errLo[_curIndex];
      hi = _errHi[_curIndex];
      return ;
    }

    // We didn't track asymmetric errors so far, so now we need to allocate
    initializeAsymErrArrays();

    // Calculate poisson errors
    double ym,yp ;
    RooHistError::instance().getPoissonInterval(Int_t(weight()+0.5),ym,yp,1) ;
    _errLo[_curIndex] = weight()-ym;
    _errHi[_curIndex] = yp-weight();
    lo = _errLo[_curIndex];
    hi = _errHi[_curIndex];
    return ;

  case SumW2:
    lo = std::sqrt(weightSquared(_curIndex));
    hi = lo;
    return ;

  case None:
    lo = 0 ;
    hi = 0 ;
    return ;
  }
}


// wve adjust for variable bin sizes

////////////////////////////////////////////////////////////////////////////////
/// Perform boundary safe 'intOrder'-th interpolation of weights in dimension 'dim'
/// at current value 'xval'

/// \param[in] iDim Index of the histogram dimension along which to interpolate.
/// \param[in] xval Value of histogram variable at dimension `iDim` for which
///                 we want to interpolate the histogram weight.
/// \param[in] centralIdx Index of the bin that the point at which we
///                       interpolate the histogram weight falls into
///                       (can be obtained with `RooDataHist::calcTreeIndex`).
/// \param[in] intOrder Interpolation order, i.e. how many neighbouring bins are
///                     used for the interpolation.
/// \param[in] correctForBinSize Enable the inverse bin volume correction factor.
/// \param[in] cdfBoundaries Enable the special boundary coundition for a cdf:
///                          underflow bins are assumed to have weight zero and
///                          overflow bins have weight one. Otherwise, the
///                          histogram is mirrored at the boundaries for the
///                          interpolation.
double RooDataHist::interpolateDim(int iDim, double xval, size_t centralIdx, int intOrder, bool correctForBinSize, bool cdfBoundaries)
{
  auto const& binning = static_cast<RooRealVar&>(*_vars[iDim]).getBinning();

  // Fill workspace arrays spanning interpolation area
  int fbinC = binning.binNumber(xval) ;
  int fbinLo = fbinC-intOrder/2 - ((xval<binning.binCenter(fbinC))?1:0) ;
  int fbinM = binning.numBins() ;

  auto idxMult = _idxMult[iDim];
  auto offsetIdx = centralIdx - idxMult * fbinC;

  // Use a class-member buffer to avoid repeated heap allocations.
  double * yarr = _interpolationBuffer.data();
  double * xarr = yarr + intOrder + 1;

  for (int i=fbinLo ; i<=intOrder+fbinLo ; i++) {
    int ibin ;
    if (i>=0 && i<fbinM) {
      // In range
      ibin = i ;
      xarr[i-fbinLo] = binning.binCenter(ibin) ;
      auto idx = offsetIdx + idxMult * ibin;
      yarr[i - fbinLo] = _wgt[idx];
      if (correctForBinSize) yarr[i-fbinLo] /=  _binv[idx] ;
    } else if (i>=fbinM) {
      // Overflow: mirror
      ibin = 2*fbinM-i-1 ;
      if (cdfBoundaries) {
        xarr[i-fbinLo] = binning.highBound()+1e-10*(i-fbinM+1) ;
        yarr[i-fbinLo] = 1.0 ;
      } else {
        auto idx = offsetIdx + idxMult * ibin;
        xarr[i-fbinLo] = 2*binning.highBound()-binning.binCenter(ibin) ;
        yarr[i - fbinLo] = _wgt[idx];
        if (correctForBinSize)
          yarr[i - fbinLo] /= _binv[idx];
      }
    } else {
      // Underflow: mirror
      ibin = -i - 1 ;
      if (cdfBoundaries) {
        xarr[i-fbinLo] = binning.lowBound()-ibin*(1e-10) ;
        yarr[i-fbinLo] = 0.0 ;
      } else {
        auto idx = offsetIdx + idxMult * ibin;
        xarr[i-fbinLo] = 2*binning.lowBound()-binning.binCenter(ibin) ;
        yarr[i - fbinLo] = _wgt[idx];
        if (correctForBinSize)
          yarr[i - fbinLo] /= _binv[idx];
      }
    }
  }
  return RooMath::interpolate(xarr,yarr,intOrder+1,xval) ;
}




////////////////////////////////////////////////////////////////////////////////
/// Increment the bin content of the bin enclosing the given coordinates.
///
/// \param[in] row Coordinates of the bin.
/// \param[in] wgt Increment by this weight.
/// \param[in] sumw2 Optionally, track the sum of squared weights. If a value > 0 or
/// a weight != 1. is passed for the first time, a vector for the squared weights will be allocated.
void RooDataHist::add(const RooArgSet& row, double wgt, double sumw2)
{
  checkInit() ;

  if ((sumw2 > 0. || wgt != 1.) && !_sumw2) {
    // Receiving a weighted entry. SumW2 != sumw from now on.
    _sumw2 = new double[_arrSize];
    std::copy(_wgt, _wgt+_arrSize, _sumw2);

    registerWeightArraysToDataStore();
  }

  const auto idx = calcTreeIndex(row, false);

  _wgt[idx] += wgt ;
  if (_sumw2) _sumw2[idx] += (sumw2 > 0 ? sumw2 : wgt*wgt);

  _cache_sum_valid = false;
}



////////////////////////////////////////////////////////////////////////////////
/// Set a bin content.
/// \param[in] row Coordinates of the bin to be set.
/// \param[in] wgt New bin content.
/// \param[in] wgtErrLo Low error of the bin content.
/// \param[in] wgtErrHi High error of the bin content.
void RooDataHist::set(const RooArgSet& row, double wgt, double wgtErrLo, double wgtErrHi)
{
  checkInit() ;

  initializeAsymErrArrays();

  const auto idx = calcTreeIndex(row, false);

  _wgt[idx] = wgt ;
  _errLo[idx] = wgtErrLo ;
  _errHi[idx] = wgtErrHi ;

  _cache_sum_valid = false;
}



////////////////////////////////////////////////////////////////////////////////
/// Set bin content of bin that was last loaded with get(std::size_t).
/// \param[in] binNumber Optional bin number to set. If empty, currently active bin is set.
/// \param[in] wgt New bin content.
/// \param[in] wgtErr Error of the new bin content. If the weight need not have an error, use 0. or a negative number.
void RooDataHist::set(std::size_t binNumber, double wgt, double wgtErr) {
  checkInit() ;

  if (wgtErr > 0. && !_sumw2) {
    // Receiving a weighted entry. Need to track sumw2 from now on:
    cloneArray(_sumw2, _wgt, _arrSize);

    registerWeightArraysToDataStore();
  }

  _wgt[binNumber] = wgt ;
  if (_errLo) _errLo[binNumber] = wgtErr;
  if (_errHi) _errHi[binNumber] = wgtErr;
  if (_sumw2) _sumw2[binNumber] = wgtErr*wgtErr;

  _cache_sum_valid = false ;
}


////////////////////////////////////////////////////////////////////////////////
/// Set bin content of bin that was last loaded with get(std::size_t).
/// \param[in] wgt New bin content.
/// \param[in] wgtErr Optional error of the bin content.
void RooDataHist::set(double wgt, double wgtErr) {
  if (_curIndex == std::numeric_limits<std::size_t>::max()) {
    _curIndex = calcTreeIndex(_vars, true) ;
  }

  set(_curIndex, wgt, wgtErr);
}


////////////////////////////////////////////////////////////////////////////////
/// Set a bin content.
/// \param[in] row Coordinates to compute the bin from.
/// \param[in] wgt New bin content.
/// \param[in] wgtErr Optional error of the bin content.
void RooDataHist::set(const RooArgSet& row, double wgt, double wgtErr) {
  set(calcTreeIndex(row, false), wgt, wgtErr);
}



////////////////////////////////////////////////////////////////////////////////
/// Add all data points contained in 'dset' to this data set with given weight.
/// Optional cut string expression selects the data points to be added and can
/// reference any variable contained in this data set

void RooDataHist::add(const RooAbsData& dset, const char* cut, double wgt)
{
  RooFormulaVar cutVar("select",cut,*dset.get()) ;
  add(dset,&cutVar,wgt) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Add all data points contained in 'dset' to this data set with given weight.
/// Optional RooFormulaVar pointer selects the data points to be added.

void RooDataHist::add(const RooAbsData& dset, const RooFormulaVar* cutVar, double wgt)
{
  checkInit() ;

  RooFormulaVar* cloneVar = 0;
  std::unique_ptr<RooArgSet> tmp;
  if (cutVar) {
    // Deep clone cutVar and attach clone to this dataset
    tmp.reset(static_cast<RooArgSet*>(RooArgSet(*cutVar).snapshot()));
    if (!tmp) {
      coutE(DataHandling) << "RooDataHist::add(" << GetName() << ") Couldn't deep-clone cut variable, abort," << endl ;
      return ;
    }

    cloneVar = (RooFormulaVar*) tmp->find(*cutVar) ;
    cloneVar->attachDataSet(dset) ;
  }


  Int_t i ;
  for (i=0 ; i<dset.numEntries() ; i++) {
    const RooArgSet* row = dset.get(i) ;
    if (!cloneVar || cloneVar->getVal()) {
       add(*row,wgt*dset.weight(), wgt*wgt*dset.weightSquared()) ;
    }
  }

  _cache_sum_valid = false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the sum of the weights of all bins in the histogram.
///
/// \param[in] correctForBinSize Multiply the sum of weights in each bin
/// with the N-dimensional bin volume, making the return value
/// the integral over the function represented by this histogram.
/// \param[in] inverseBinCor Divide by the N-dimensional bin volume.
double RooDataHist::sum(bool correctForBinSize, bool inverseBinCor) const
{
  checkInit() ;

  // Check if result was cached
  const CacheSumState_t cache_code = !correctForBinSize ? kNoBinCorrection : (inverseBinCor ? kInverseBinCorr : kCorrectForBinSize);
  if (_cache_sum_valid == static_cast<Int_t>(cache_code)) {
    return _cache_sum ;
  }

  ROOT::Math::KahanSum<double> kahanSum;
  for (Int_t i=0; i < _arrSize; i++) {
    const double theBinVolume = correctForBinSize ? (inverseBinCor ? 1/_binv[i] : _binv[i]) : 1.0 ;
    kahanSum += _wgt[i] * theBinVolume;
  }

  // Store result in cache
  _cache_sum_valid = cache_code;
  _cache_sum = kahanSum.Sum();

  return kahanSum.Sum();
}



////////////////////////////////////////////////////////////////////////////////
/// Return the sum of the weights of a multi-dimensional slice of the histogram
/// by summing only over the dimensions specified in sumSet.
///
/// The coordinates of all other dimensions are fixed to those given in sliceSet
///
/// If correctForBinSize is specified, the sum of weights
/// is multiplied by the M-dimensional bin volume, (M = N(sumSet)),
/// making the return value the integral over the function
/// represented by this histogram

double RooDataHist::sum(const RooArgSet& sumSet, const RooArgSet& sliceSet, bool correctForBinSize, bool inverseBinCor)
{
  checkInit() ;

  RooArgSet varSave ;
  varSave.addClone(_vars) ;

  RooArgSet sliceOnlySet(sliceSet);
  sliceOnlySet.remove(sumSet,true,true) ;

  _vars.assign(sliceOnlySet);
  std::vector<double> const * pbinv = nullptr;

  if(correctForBinSize && inverseBinCor) {
     pbinv = &calculatePartialBinVolume(sliceOnlySet);
  } else if(correctForBinSize && !inverseBinCor) {
     pbinv = &calculatePartialBinVolume(sumSet);
  }

  // Calculate mask and refence plot bins for non-iterating variables
  std::vector<bool> mask(_vars.size());
  std::vector<int> refBin(_vars.size());

  for (unsigned int i = 0; i < _vars.size(); ++i) {
    const RooAbsArg*    arg   = _vars[i];
    const RooAbsLValue* argLv = _lvvars[i]; // Same as above, but cross-cast

    if (sumSet.find(*arg)) {
      mask[i] = false ;
    } else {
      mask[i] = true ;
      refBin[i] = argLv->getBin();
    }
  }

  // Loop over entire data set, skipping masked entries
  ROOT::Math::KahanSum<double> total;
  for (Int_t ibin=0; ibin < _arrSize; ++ibin) {

    std::size_t tmpibin = ibin;
    bool skip(false) ;

    // Check if this bin belongs in selected slice
    for (unsigned int ivar = 0; !skip && ivar < _vars.size(); ++ivar) {
      const Int_t idx = tmpibin / _idxMult[ivar] ;
      tmpibin -= idx*_idxMult[ivar] ;
      if (mask[ivar] && idx!=refBin[ivar])
        skip = true ;
    }

    if (!skip) {
      const double theBinVolume = correctForBinSize ? (inverseBinCor ? 1/(*pbinv)[ibin] : (*pbinv)[ibin] ) : 1.0 ;
      total += _wgt[ibin] * theBinVolume;
    }
  }

  _vars.assign(varSave) ;

  return total.Sum();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the sum of the weights of a multi-dimensional slice of the histogram
/// by summing only over the dimensions specified in sumSet.
///
/// The coordinates of all other dimensions are fixed to those given in sliceSet
///
/// If correctForBinSize is specified, the sum of weights
/// is multiplied by the M-dimensional bin volume, (M = N(sumSet)),
/// or the fraction of it that falls inside the range rangeName,
/// making the return value the integral over the function
/// represented by this histogram.
///
/// If correctForBinSize is not specified, the weights are multiplied by the
/// fraction of the bin volume that falls inside the range, i.e. a factor of
/// binVolumeInRange/totalBinVolume.

double RooDataHist::sum(const RooArgSet& sumSet, const RooArgSet& sliceSet,
                          bool correctForBinSize, bool inverseBinCor,
                          const std::map<const RooAbsArg*, std::pair<double, double> >& ranges,
                          std::function<double(int)> getBinScale)
{
  checkInit();
  checkBinBounds();
  RooArgSet varSave;
  varSave.addClone(_vars);
  {
    RooArgSet sliceOnlySet(sliceSet);
    sliceOnlySet.remove(sumSet, true, true);
    _vars.assign(sliceOnlySet);
  }

  // Calculate mask and reference plot bins for non-iterating variables,
  // and get ranges for iterating variables
  std::vector<bool> mask(_vars.size());
  std::vector<int> refBin(_vars.size());
  std::vector<double> rangeLo(_vars.size(), -std::numeric_limits<double>::infinity());
  std::vector<double> rangeHi(_vars.size(), +std::numeric_limits<double>::infinity());

  for (std::size_t i = 0; i < _vars.size(); ++i) {
    const RooAbsArg* arg = _vars[i];
    const RooAbsLValue* argLV = _lvvars[i]; // Same object as above, but cross cast

    RooAbsArg* sumsetv = sumSet.find(*arg);
    RooAbsArg* slicesetv = sliceSet.find(*arg);
    mask[i] = !sumsetv;
    if (mask[i]) {
      assert(argLV);
      refBin[i] = argLV->getBin();
    }

    auto it = ranges.find(sumsetv ? sumsetv : slicesetv);
    if (ranges.end() != it) {
      rangeLo[i] = it->second.first;
      rangeHi[i] = it->second.second;
    }
  }

  // Loop over entire data set, skipping masked entries
  ROOT::Math::KahanSum<double> total;
  for (Int_t ibin = 0; ibin < _arrSize; ++ibin) {
    // Check if this bin belongs in selected slice
    bool skip{false};
    for (int ivar = 0, tmp = ibin; !skip && ivar < int(_vars.size()); ++ivar) {
      const Int_t idx = tmp / _idxMult[ivar];
      tmp -= idx*_idxMult[ivar];
      if (mask[ivar] && idx!=refBin[ivar]) skip = true;
    }

    if (skip) continue;

    // Work out bin volume
    // It's not necessary to figure out the bin volume for the slice-only set explicitely here.
    // We need to loop over the sumSet anyway to get the partial bin containment correction,
    // so we can get the slice-only set volume later by dividing _binv[ibin] / binVolumeSumSetFull.
    double binVolumeSumSetFull = 1.;
    double binVolumeSumSetInRange = 1.;
    for (Int_t ivar = 0, tmp = ibin; ivar < (int)_vars.size(); ++ivar) {
      const Int_t idx = tmp / _idxMult[ivar];
      tmp -= idx*_idxMult[ivar];

      // If the current variable is not in the sumSet, it should not be considered for the bin volume
      const auto arg = _vars[ivar];
      if (!sumSet.find(*arg)) {
          continue;
      }

      if (_binbounds[ivar].empty()) continue;
      const double binLo = _binbounds[ivar][2 * idx];
      const double binHi = _binbounds[ivar][2 * idx + 1];
      if (binHi < rangeLo[ivar] || binLo > rangeHi[ivar]) {
        // bin is outside of allowed range - effective bin volume is zero
        binVolumeSumSetInRange = 0.;
        break;
      }

      binVolumeSumSetFull *= binHi - binLo;
      binVolumeSumSetInRange *= std::min(rangeHi[ivar], binHi) - std::max(rangeLo[ivar], binLo);
    }
    const double corrPartial = binVolumeSumSetInRange / binVolumeSumSetFull;
    if (0. == corrPartial) continue;
    const double corr = correctForBinSize ? (inverseBinCor ? binVolumeSumSetFull / _binv[ibin] : binVolumeSumSetFull ) : 1.0;
    total += getBinScale(ibin)*(_wgt[ibin] * corr * corrPartial);
  }

  _vars.assign(varSave);

  return total.Sum();
}



////////////////////////////////////////////////////////////////////////////////
/// Fill the transient cache with partial bin volumes with up-to-date
/// values for the partial volume specified by observables 'dimSet'

const std::vector<double>& RooDataHist::calculatePartialBinVolume(const RooArgSet& dimSet) const
{
  // The code bitset has all bits set to one whose position corresponds to arguments in dimSet.
  // It is used as the key for the bin volume caching hash map.
  int code{0};
  {
    int i{0} ;
    for (auto const& v : _vars) {
      code += ((dimSet.find(*v) ? 1 : 0) << i) ;
      ++i;
    }
  }

  auto& pbinv = _pbinvCache[code];
  if(!pbinv.empty()) {
    return pbinv;
  }
  pbinv.resize(_arrSize);

  // Calculate plot bins of components from master index
  std::vector<bool> selDim(_vars.size());
  for (std::size_t i = 0; i < selDim.size(); ++i) {
    selDim[i] = (code >> i) & 1 ;
  }

  // Recalculate partial bin volume cache
  for (Int_t ibin=0; ibin < _arrSize ;ibin++) {
    Int_t idx(0), tmp(ibin) ;
    double theBinVolume(1) ;
    for (unsigned int j=0; j < _lvvars.size(); ++j) {
      const RooAbsLValue* arg = _lvvars[j];
      assert(arg);

      idx  = tmp / _idxMult[j];
      tmp -= idx*_idxMult[j];
      if (selDim[j]) {
        theBinVolume *= arg->getBinWidth(idx) ;
      }
    }
    pbinv[ibin] = theBinVolume ;
  }

  return pbinv;
}


////////////////////////////////////////////////////////////////////////////////
/// Sum the weights of all bins.
double RooDataHist::sumEntries() const {
  return ROOT::Math::KahanSum<double>::Accumulate(_wgt, _wgt + _arrSize).Sum();
}



////////////////////////////////////////////////////////////////////////////////
/// Return the sum of weights in all entries matching cutSpec (if specified)
/// and in named range cutRange (if specified)
/// Return the

double RooDataHist::sumEntries(const char* cutSpec, const char* cutRange) const
{
  checkInit() ;

  if (cutSpec==0 && cutRange==0) {
    return sumEntries();
  } else {

    // Setup RooFormulaVar for cutSpec if it is present
    std::unique_ptr<RooFormula> select;
    if (cutSpec) {
      select = std::make_unique<RooFormula>("select",cutSpec,*get());
    }

    // Otherwise sum the weights in the event
    ROOT::Math::KahanSum<> kahanSum;
    for (Int_t i=0; i < _arrSize; i++) {
      get(i) ;
      if ((select && select->eval() == 0.) || (cutRange && !_vars.allInRange(cutRange)))
          continue;

      kahanSum += weight(i);
    }

    return kahanSum.Sum();
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Reset all bin weights to zero

void RooDataHist::reset()
{
  // WVE DO NOT CALL RooTreeData::reset() for binned
  // datasets as this will delete the bin definitions

  std::fill(_wgt, _wgt + _arrSize, 0.);
  delete[] _errLo; _errLo = nullptr;
  delete[] _errHi; _errHi = nullptr;
  delete[] _sumw2; _sumw2 = nullptr;

  registerWeightArraysToDataStore();

  _cache_sum_valid = false;
}



////////////////////////////////////////////////////////////////////////////////
/// Load bin `binNumber`, and return an argset with the coordinates of the bin centre.
/// \note The argset is owned by this data hist, and this function has a side effect, because
/// it alters the currently active bin.
const RooArgSet* RooDataHist::get(Int_t binNumber) const
{
  checkInit() ;
  _curIndex = binNumber;

  return RooAbsData::get(_curIndex);
}



////////////////////////////////////////////////////////////////////////////////
/// Return a RooArgSet with whose coordinates denote the bin centre of the bin
/// enclosing the point in `coord`.
/// \note The argset is owned by this data hist, and this function has a side effect, because
/// it alters the currently active bin.
const RooArgSet* RooDataHist::get(const RooArgSet& coord) const {
  return get(calcTreeIndex(coord, false));
}



////////////////////////////////////////////////////////////////////////////////
/// Return the volume of the bin enclosing coordinates 'coord'.
double RooDataHist::binVolume(const RooArgSet& coord) const {
  checkInit() ;
  return _binv[calcTreeIndex(coord, false)] ;
}


////////////////////////////////////////////////////////////////////////////////
/// Create an iterator over all bins in a slice defined by the subset of observables
/// listed in sliceArg. The position of the slice is given by otherArgs

TIterator* RooDataHist::sliceIterator(RooAbsArg& sliceArg, const RooArgSet& otherArgs)
{
  // Update to current position
  _vars.assign(otherArgs) ;
  _curIndex = calcTreeIndex(_vars, true);

  RooAbsArg* intArg = _vars.find(sliceArg) ;
  if (!intArg) {
    coutE(InputArguments) << "RooDataHist::sliceIterator() variable " << sliceArg.GetName() << " is not part of this RooDataHist" << endl ;
    return 0 ;
  }
  return new RooDataHistSliceIter(*this,*intArg) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Change the name of the RooDataHist

void RooDataHist::SetName(const char *name)
{
  if (_dir) _dir->GetList()->Remove(this);
  // We need to use the function from RooAbsData, because it already overrides TNamed::SetName
  RooAbsData::SetName(name);
  if (_dir) _dir->GetList()->Add(this);
}


////////////////////////////////////////////////////////////////////////////////
/// Change the title of this RooDataHist

void RooDataHist::SetNameTitle(const char *name, const char* title)
{
  SetName(name);
  SetTitle(title);
}


////////////////////////////////////////////////////////////////////////////////
/// Print value of the dataset, i.e. the sum of weights contained in the dataset

void RooDataHist::printValue(ostream& os) const
{
  os << numEntries() << " bins (" << sumEntries() << " weights)" ;
}




////////////////////////////////////////////////////////////////////////////////
/// Print argument of dataset, i.e. the observable names

void RooDataHist::printArgs(ostream& os) const
{
  os << "[" ;
  bool first(true) ;
  for (const auto arg : _vars) {
    if (first) {
      first=false ;
    } else {
      os << "," ;
    }
    os << arg->GetName() ;
  }
  os << "]" ;
}



////////////////////////////////////////////////////////////////////////////////
/// Returns true if dataset contains entries with a non-integer weight.

bool RooDataHist::isNonPoissonWeighted() const
{
  for (Int_t i=0; i < _arrSize; ++i) {
    const double wgt = _wgt[i];
    double intpart;
    if (std::abs(std::modf(wgt, &intpart)) > 1.E-10)
      return true;
  }

  return false;
}


////////////////////////////////////////////////////////////////////////////////
/// Print the details on the dataset contents

void RooDataHist::printMultiline(ostream& os, Int_t content, bool verbose, TString indent) const
{
  RooAbsData::printMultiline(os,content,verbose,indent) ;

  os << indent << "Binned Dataset " << GetName() << " (" << GetTitle() << ")" << endl ;
  os << indent << "  Contains " << numEntries() << " bins with a total weight of " << sumEntries() << endl;

  if (!verbose) {
    os << indent << "  Observables " << _vars << endl ;
  } else {
    os << indent << "  Observables: " ;
    _vars.printStream(os,kName|kValue|kExtras|kTitle,kVerbose,indent+"  ") ;
  }

  if(verbose) {
    if (!_cachedVars.empty()) {
      os << indent << "  Caches " << _cachedVars << endl ;
    }
  }
}

void RooDataHist::printDataHistogram(ostream& os, RooRealVar* obs) const
{
  for(Int_t i=0; i<obs->getBins(); ++i){
    this->get(i);
    obs->setBin(i);
    os << this->weight() << " +/- " << this->weightSquared() << endl;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooDataHist.
void RooDataHist::Streamer(TBuffer &R__b) {
  if (R__b.IsReading()) {

    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c);

    if (R__v > 2) {
      R__b.ReadClassBuffer(RooDataHist::Class(),this,R__v,R__s,R__c);
      R__b.CheckByteCount(R__s, R__c, RooDataHist::IsA());
      initialize(0, false);
    } else {

      // Legacy dataset conversion happens here. Legacy RooDataHist inherits from RooTreeData
      // which in turn inherits from RooAbsData. Manually stream RooTreeData contents on
      // file here and convert it into a RooTreeDataStore which is installed in the
      // new-style RooAbsData base class

      // --- This is the contents of the streamer code of RooTreeData version 2 ---
      UInt_t R__s1, R__c1;
      Version_t R__v1 = R__b.ReadVersion(&R__s1, &R__c1); if (R__v1) { }

      RooAbsData::Streamer(R__b);
      TTree* X_tree(0) ; R__b >> X_tree;
      RooArgSet X_truth ; X_truth.Streamer(R__b);
      TString X_blindString ; X_blindString.Streamer(R__b);
      R__b.CheckByteCount(R__s1, R__c1, TClass::GetClass("RooTreeData"));
      // --- End of RooTreeData-v1 streamer

      // Construct RooTreeDataStore from X_tree and complete initialization of new-style RooAbsData
      _dstore = std::make_unique<RooTreeDataStore>(X_tree,_vars);
      _dstore->SetName(GetName()) ;
      _dstore->SetTitle(GetTitle()) ;
      _dstore->checkInit() ;

      RooDirItem::Streamer(R__b);
      R__b >> _arrSize;
      delete [] _wgt;
      _wgt = new double[_arrSize];
      R__b.ReadFastArray(_wgt,_arrSize);
      delete [] _errLo;
      _errLo = new double[_arrSize];
      R__b.ReadFastArray(_errLo,_arrSize);
      delete [] _errHi;
      _errHi = new double[_arrSize];
      R__b.ReadFastArray(_errHi,_arrSize);
      delete [] _sumw2;
      _sumw2 = new double[_arrSize];
      R__b.ReadFastArray(_sumw2,_arrSize);
      delete [] _binv;
      _binv = new double[_arrSize];
      RooArgSet tmpSet;
      tmpSet.Streamer(R__b);
      double tmp;
      R__b >> tmp; //_curWeight;
      R__b >> tmp; //_curWgtErrLo;
      R__b >> tmp; //_curWgtErrHi;
      R__b >> tmp; //_curSumW2;
      R__b >> tmp; //_curVolume;
      R__b >> _curIndex;
      R__b.CheckByteCount(R__s, R__c, RooDataHist::IsA());
    }

  } else {

    R__b.WriteClassBuffer(RooDataHist::Class(),this);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Return event weights of all events in range [first, first+len).
/// If cacheValidEntries() has been called, out-of-range events will have a weight of 0.
RooSpan<const double> RooDataHist::getWeightBatch(std::size_t first, std::size_t len, bool sumW2 /*=false*/) const {
  return {(sumW2 && _sumw2 ? _sumw2 : _wgt) + first, len};
}


////////////////////////////////////////////////////////////////////////////////
/// Hand over pointers to our weight arrays to the data store implementation.
void RooDataHist::registerWeightArraysToDataStore() const {
  _dstore->setExternalWeightArray(_wgt, _errLo, _errHi, _sumw2);
}


////////////////////////////////////////////////////////////////////////////////
/// Return reference to VarInfo struct with cached histogram variable
/// information that is frequently used for histogram weights retrieval.
///
/// If the `_varInfo` struct was not initialized yet, it will be initialized in
/// this function.
RooDataHist::VarInfo const& RooDataHist::getVarInfo() {

  if(_varInfo.initialized) return _varInfo;

  auto& info = _varInfo;

  {
    // count the number of real vars and get their indices
    info.nRealVars = 0;
    size_t iVar = 0;
    for (const auto real : _vars) {
      if (dynamic_cast<RooRealVar*>(real)) {
        if(info.nRealVars == 0) info.realVarIdx1 = iVar;
        if(info.nRealVars == 1) info.realVarIdx2 = iVar;
        ++info.nRealVars;
      }
      ++iVar;
    }
  }

  {
    // assert that the variables are either real values or categories
    for (unsigned int i=0; i < _vars.size(); ++i) {
      if (_lvbins[i].get()) {
        assert(dynamic_cast<const RooAbsReal*>(_vars[i]));
      } else {
        assert(dynamic_cast<const RooAbsCategoryLValue*>(_vars[i]));
      }
    }
  }

  info.initialized = true;

  return info;
}
