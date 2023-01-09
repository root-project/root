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
\file RooPlot.cxx
\class RooPlot
\ingroup Roofitcore

A RooPlot is a plot frame and a container for graphics objects
within that frame. As a frame, it provides the TH1-style public interface
for setting plot ranges, configuring axes, etc. As a container, it
holds an arbitrary set of objects that might be histograms of data,
curves representing a fit model, or text labels. Use the Draw()
method to draw a frame and the objects it contains. Use the various
add...() methods to add objects to be drawn.  In general, the
add...() methods create a private copy of the object you pass them
and return a pointer to this copy. The caller owns the input object
and this class owns the returned object.
All RooAbsReal and RooAbsData derived classes implement plotOn()
functions that facilitate to plot themselves on a given RooPlot, e.g.
~~~ {.cpp}
RooPlot *frame = x.frame() ;
data.plotOn(frame) ;
pdf.plotOn(frame) ;
~~~
These high level functions also take care of any projections
or other mappings that need to be made to plot a multi-dimensional
object onto a one-dimensional plot.
**/

#include "RooPlot.h"

#include "RooAbsReal.h"
#include "RooAbsRealLValue.h"
#include "RooPlotable.h"
#include "RooArgSet.h"
#include "RooCurve.h"
#include "RooHist.h"
#include "RooMsgService.h"

#include "TClass.h"
#include "TBuffer.h"
#include "TH1D.h"
#include "TBrowser.h"
#include "TVirtualPad.h"

#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"
#include "TAttText.h"
#include "TDirectoryFile.h"
#include "TLegend.h"
#include "strlcpy.h"

#include <algorithm>
#include <cstring>
#include <iostream>

using namespace std;

ClassImp(RooPlot);


bool RooPlot::_addDirStatus = true ;

bool RooPlot::addDirectoryStatus() { return _addDirStatus; }
bool RooPlot::setAddDirectoryStatus(bool flag) { bool ret = flag ; _addDirStatus = flag ; return ret ; }


////////////////////////////////////////////////////////////////////////////////
/// Default constructor
/// coverity[UNINIT_CTOR]

RooPlot::RooPlot()
{
  if (gDirectory && addDirectoryStatus()) {
    SetDirectory(gDirectory);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor of RooPlot with range [xmin,xmax]

RooPlot::RooPlot(double xmin, double xmax)
{
  _hist = new TH1D(histName(),"A RooPlot",100,xmin,xmax) ;
  _hist->Sumw2(false) ;
  _hist->GetSumw2()->Set(0) ;
  _hist->SetDirectory(nullptr);

  // Create an empty frame with the specified x-axis limits.
  initialize();

}



////////////////////////////////////////////////////////////////////////////////
/// Construct of a two-dimensional RooPlot with ranges [xmin,xmax] x [ymin,ymax]

RooPlot::RooPlot(double xmin, double xmax, double ymin, double ymax) :
  _defYmax(0)
{
  _hist = new TH1D(histName(),"A RooPlot",100,xmin,xmax) ;
  _hist->Sumw2(false) ;
  _hist->GetSumw2()->Set(0) ;
  _hist->SetDirectory(nullptr);

  SetMinimum(ymin);
  SetMaximum(ymax);
  initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// Construct a two-dimensional RooPlot with ranges and properties taken
/// from variables var1 and var2

RooPlot::RooPlot(const RooAbsRealLValue &var1, const RooAbsRealLValue &var2) :
  _defYmax(0)
{
  _hist = new TH1D(histName(),"A RooPlot",100,var1.getMin(),var1.getMax()) ;
  _hist->Sumw2(false) ;
  _hist->GetSumw2()->Set(0) ;
  _hist->SetDirectory(nullptr);

  if(!var1.hasMin() || !var1.hasMax()) {
    coutE(InputArguments) << "RooPlot::RooPlot: cannot create plot for variable without finite limits: "
    << var1.GetName() << endl;
    return;
  }
  if(!var2.hasMin() || !var2.hasMax()) {
    coutE(InputArguments) << "RooPlot::RooPlot: cannot create plot for variable without finite limits: "
    << var1.GetName() << endl;
    return;
  }
  SetMinimum(var2.getMin());
  SetMaximum(var2.getMax());
  SetXTitle(var1.getTitle(true));
  SetYTitle(var2.getTitle(true));
  initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// Construct a two-dimensional RooPlot with ranges and properties taken
/// from variables var1 and var2 but with an overriding range definition
/// of [xmin,xmax] x [ymin,ymax]

RooPlot::RooPlot(const RooAbsRealLValue &var1, const RooAbsRealLValue &var2,
       double xmin, double xmax, double ymin, double ymax) :
  _defYmax(0)
{
  _hist = new TH1D(histName(),"A RooPlot",100,xmin,xmax) ;
  _hist->Sumw2(false) ;
  _hist->GetSumw2()->Set(0) ;
  _hist->SetDirectory(nullptr);

  SetMinimum(ymin);
  SetMaximum(ymax);
  SetXTitle(var1.getTitle(true));
  SetYTitle(var2.getTitle(true));
  initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// Create an 1-dimensional with all properties taken from 'var', but
/// with an explicit range [xmin,xmax] and a default binning of 'nbins'

RooPlot::RooPlot(const char* name, const char* title, const RooAbsRealLValue &var, double xmin, double xmax, Int_t nbins)
{
  _hist = new TH1D(name,title,nbins,xmin,xmax) ;
  _hist->Sumw2(false) ;
  _hist->GetSumw2()->Set(0) ;
  _hist->SetDirectory(nullptr);

  // In the past, the plot variable was cloned, but there was no apparent reason for doing so.
  _plotVar = const_cast<RooAbsRealLValue*>(&var);

  TString xtitle= var.getTitle(true);
  SetXTitle(xtitle.Data());

  initialize();

  _normBinWidth = (xmax-xmin)/nbins ;
}


////////////////////////////////////////////////////////////////////////////////
/// Create an 1-dimensional with all properties taken from 'var', but
/// with an explicit range [xmin,xmax] and a default binning of 'nbins'

RooPlot::RooPlot(const RooAbsRealLValue &var, double xmin, double xmax, Int_t nbins)
{
  _hist = new TH1D(histName(),"RooPlot",nbins,xmin,xmax) ;
  _hist->Sumw2(false) ;
  _hist->GetSumw2()->Set(0) ;
  _hist->SetDirectory(nullptr);

  // In the past, the plot variable was cloned, but there was no apparent reason for doing so.
  _plotVar = const_cast<RooAbsRealLValue*>(&var);

  TString xtitle= var.getTitle(true);
  SetXTitle(xtitle.Data());

  TString title("A RooPlot of \"");
  title.Append(var.getTitle());
  title.Append("\"");
  SetTitle(title.Data());
  initialize();

  _normBinWidth = (xmax-xmin)/nbins ;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new frame for a given variable in x. This is just a
/// wrapper for the RooPlot constructor with the same interface.
///
/// More details.
/// \param[in] var The variable on the x-axis
/// \param[in] xmin Left edge of the x-axis
/// \param[in] xmax Right edge of the x-axis
/// \param[in] nBins number of bins on the x-axis
RooPlot* RooPlot::frame(const RooAbsRealLValue &var, double xmin, double xmax, Int_t nBins){
  return new RooPlot(var,xmin,xmax,nBins);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new frame for a given variable in x, adding bin labels.
/// The binning will be extracted from the variable given. The bin
/// labels will be set as "%g-%g" for the left and right edges of each
/// bin of the given variable.
///
/// More details.
/// \param[in] var The variable on the x-axis
RooPlot* RooPlot::frameWithLabels(const RooAbsRealLValue &var){
  RooPlot* pl = new RooPlot();
  int nbins = var.getBinning().numBins();

  pl->_hist = new TH1D(pl->histName(),"RooPlot",nbins,var.getMin(),var.getMax()) ;
  pl->_hist->Sumw2(false) ;
  pl->_hist->GetSumw2()->Set(0) ;
  pl->_hist->SetDirectory(nullptr);

  pl->_hist->SetNdivisions(-nbins);
  for(int i=0; i<nbins; ++i){
    TString s = TString::Format("%g-%g",var.getBinning().binLow(i),var.getBinning().binHigh(i));
    pl->_hist->GetXaxis()->SetBinLabel(i+1,s);
  }

  // In the past, the plot variable was cloned, but there was no apparent reason for doing so.
  pl->_plotVar = const_cast<RooAbsRealLValue*>(&var);

  TString xtitle= var.getTitle(true);
  pl->SetXTitle(xtitle.Data());

  TString title("A RooPlot of \"");
  title.Append(var.getTitle());
  title.Append("\"");
  pl->SetTitle(title.Data());
  pl->initialize();

  pl->_normBinWidth = 1.;
  return pl;
}

////////////////////////////////////////////////////////////////////////////////
/// Return empty clone of current RooPlot

RooPlot* RooPlot::emptyClone(const char* name)
{
  RooPlot* clone = new RooPlot(*_plotVar,_hist->GetXaxis()->GetXmin(),_hist->GetXaxis()->GetXmax(),_hist->GetNbinsX()) ;
  clone->SetName(name) ;
  return clone ;
}


////////////////////////////////////////////////////////////////////////////////
/// Perform initialization that is common to all constructors.

void RooPlot::initialize()
{
  SetName(histName()) ;

  if (gDirectory && addDirectoryStatus()) {
    SetDirectory(gDirectory);
  }

  // We do not have useful stats of our own
  _hist->SetStats(false);
  _hist->SetDirectory(nullptr);
  // Default vertical padding of our enclosed objects
  setPadFactor(0.05);
  // We don't know our normalization yet
  _normNumEvts= 0;
  _normBinWidth = 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Construct automatic name of internal TH1

TString RooPlot::histName() const
{
  if (_plotVar) {
    return TString(Form("frame_%s_%zx",_plotVar->GetName(),(size_t)this)) ;
  } else {
    return TString(Form("frame_%zx",(size_t)this)) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooPlot::~RooPlot()
{
  // Delete the items in our container and our iterator.
  if (_dir) {
    _dir->GetList()->RecursiveRemove(this) ;
  }

  for(auto& item : _items) delete item.first;
  if(_plotVarSet) delete _plotVarSet;
  if(_normVars) delete _normVars;
  delete _hist ;

}


////////////////////////////////////////////////////////////////////////////////
/// Set the directory that this plot is associated to.
/// Setting it to `nullptr` will remove the object from all directories.
/// Like TH1::SetDirectory.
void RooPlot::SetDirectory(TDirectory *dir) {
  if (_dir) {
    _dir->GetList()->RecursiveRemove(this);
  }
  _dir = dir;
  if (_dir) {
    _dir->Append(this);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Install the given set of observables are reference normalization
/// variables for this frame. These observables are e.g. later used
/// to automatically project out observables when plotting functions
/// on this frame. This function is only effective when called the
/// first time on a frame

void RooPlot::updateNormVars(const RooArgSet &vars)
{
  if(0 == _normVars) _normVars= (RooArgSet*) vars.snapshot(true);
}


////////////////////////////////////////////////////////////////////////////////
/// Add a generic object to this plot. The specified options will be
/// used to Draw() this object later. The caller transfers ownership
/// of the object with this call, and the object will be deleted
/// when its containing plot object is destroyed.

void RooPlot::addObject(TObject *obj, Option_t *drawOptions, bool invisible)
{
  if(0 == obj) {
    coutE(InputArguments) << fName << "::addObject: called with a null pointer" << endl;
    return;
  }
  DrawOpt opt(drawOptions) ;
  opt.invisible = invisible ;
  _items.emplace_back(obj,opt.rawOpt());
}


////////////////////////////////////////////////////////////////////////////////
/// Add a TH1 histogram object to this plot. The specified options
/// will be used to Draw() this object later. "SAME" will be added to
/// the options if they are not already present. The caller transfers
/// ownership of the object with this call, and the object will be
/// deleted when its containing plot object is destroyed.

void RooPlot::addTH1(TH1 *hist, Option_t *drawOptions, bool invisible)
{
  if(0 == hist) {
    coutE(InputArguments) << fName << "::addTH1: called with a null pointer" << endl;
    return;
  }
  // check that this histogram is really 1D
  if(1 != hist->GetDimension()) {
    coutE(InputArguments) << fName << "::addTH1: cannot plot histogram with "
    << hist->GetDimension() << " dimensions" << endl;
    return;
  }

  // add option "SAME" if necessary
  TString options(drawOptions);
  options.ToUpper();
  if(!options.Contains("SAME")) options.Append("SAME");

  // update our y-axis label and limits
  updateYAxis(hist->GetMinimum(),hist->GetMaximum(),hist->GetYaxis()->GetTitle());

  // use this histogram's normalization if necessary
  updateFitRangeNorm(hist);

  // add the histogram to our list
  addObject(hist,options.Data(),invisible);
}


namespace {
  // this helper function is intended to translate a graph from a regular axis to a labelled axis
  // this version uses TGraph, which is a parent of RooCurve
  void translateGraph(TH1* hist, RooAbsRealLValue* xvar, TGraph* graph){
    // if the graph already has a labelled axis, don't do anything
    if(graph->GetXaxis()->IsAlphanumeric()) return;
    double xmin = hist->GetXaxis()->GetXmin();
    double xmax = hist->GetXaxis()->GetXmax();
    if(graph->TestBit(TGraph::kIsSortedX)){
      // sorted graphs are "line graphs"
      // evaluate the graph at the lower and upper edge as well as the center of each bin
      std::vector<double> x;
      std::vector<double> y;
      x.push_back(xmin);
      y.push_back(graph->Eval(xvar->getBinning().binLow(0)));
      for(int i=0; i<hist->GetNbinsX(); ++i){
        x.push_back(hist->GetXaxis()->GetBinUpEdge(i+1));
        y.push_back(graph->Eval(xvar->getBinning().binHigh(i)));
        x.push_back(hist->GetXaxis()->GetBinCenter(i+1));
        y.push_back(graph->Eval(xvar->getBinning().binCenter(i)));
      }
      int n = x.size();
      graph->Set(n);
      for(int i=0; i<n; ++i){
        graph->SetPoint(i,x[i],y[i]);
      }
      graph->Sort();
    } else {
      // unsorted graphs are "area graphs"
      std::map<int,double> minValues;
      std::map<int,double> maxValues;
      int n = graph->GetN();
      double x, y;
      // for each bin, find the min and max points to form an envelope
      for(int i=0; i<n; ++i){
        graph->GetPoint(i,x,y);
        int bin = xvar->getBinning().binNumber(x)+1;
        if(maxValues.find(bin)!=maxValues.end()){
          maxValues[bin] = std::max(maxValues[bin],y);
        } else {
          maxValues[bin] = y;
        }
        if(minValues.find(bin)!=minValues.end()){
          minValues[bin] = std::min(minValues[bin],y);
        } else {
          minValues[bin] = y;
        }
      }
      double xminY = graph->Eval(xmin);
      double xmaxY = graph->Eval(xmax);
      graph->Set(hist->GetNbinsX()+2);
      int np=0;
      graph->SetPoint(np,xmin,xminY);
      // assign the calculated envelope boundaries to the bin centers of the bins
      for(auto it = maxValues.begin(); it != maxValues.end(); ++it){
        graph->SetPoint(++np,hist->GetXaxis()->GetBinCenter(it->first),it->second);
      }
      graph->SetPoint(++np,xmax,xmaxY);
      for(auto it = minValues.rbegin(); it != minValues.rend(); ++it){
        graph->SetPoint(++np,hist->GetXaxis()->GetBinCenter(it->first),it->second);
      }
      graph->SetPoint(++np,xmin,xminY);
    }
    // make sure that the graph also has the labels set, such that subsequent calls to translate this graph will not do anything
    graph->GetXaxis()->Set(hist->GetNbinsX(),xmin,xmax);
    for(int i=0; i<hist->GetNbinsX(); ++i){
      graph->GetXaxis()->SetBinLabel(i+1,hist->GetXaxis()->GetBinLabel(i+1));
    }
  }
  // this version uses TGraphErrors, which is a parent of RooHist
  void translateGraph(TH1* hist, RooAbsRealLValue* xvar, TGraphAsymmErrors* graph){
    // if the graph already has a labelled axis, don't do anything
    if(graph->GetXaxis()->IsAlphanumeric()) return;
    int n = graph->GetN();
    double xmin = hist->GetXaxis()->GetXmin();
    double xmax = hist->GetXaxis()->GetXmax();
    double x, y;
    // as this graph is histogram-like, we expect there to be one point per bin
    // we just move these points to the respective bin centers
    for(int i=0; i<n; ++i){
      if(graph->GetPoint(i,x,y)!=i) break;
      int bin = xvar->getBinning().binNumber(x);
      graph->SetPoint(i,hist->GetXaxis()->GetBinCenter(bin+1),y);
      graph->SetPointEXhigh(i,0.5*hist->GetXaxis()->GetBinWidth(bin+1));
      graph->SetPointEXlow(i,0.5*hist->GetXaxis()->GetBinWidth(bin+1));
    }
    graph->GetXaxis()->Set(hist->GetNbinsX(),xmin,xmax);
    // make sure that the graph also has the labels set, such that subsequent calls to translate this graph will not do anything
    for(int i=0; i<hist->GetNbinsX(); ++i){
      graph->GetXaxis()->SetBinLabel(i+1,hist->GetXaxis()->GetBinLabel(i+1));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Add the specified plotable object to our plot. Increase our y-axis
/// limits to fit this object if necessary. The default lower-limit
/// is zero unless we are plotting an object that takes on negative values.
/// This call transfers ownership of the plotable object to this class.
/// The plotable object will be deleted when this plot object is deleted.
void RooPlot::addPlotable(RooPlotable *plotable, Option_t *drawOptions, bool invisible, bool refreshNorm)
{
  // update our y-axis label and limits
  updateYAxis(plotable->getYAxisMin(),plotable->getYAxisMax(),plotable->getYAxisLabel());

  // use this object's normalization if necessary
  updateFitRangeNorm(plotable,refreshNorm) ;

  // add this element to our list and remember its drawing option
  TObject *obj= plotable->crossCast();
  if(0 == obj) {
    coutE(InputArguments) << fName << "::add: cross-cast to TObject failed (nothing added)" << endl;
  }
  else {
    // if the frame axis is alphanumeric, the coordinates of the graph need to be translated to this binning
    if(_hist->GetXaxis()->IsAlphanumeric()){
      if(obj->InheritsFrom(RooCurve::Class())){
        ::translateGraph(_hist,_plotVar,static_cast<RooCurve*>(obj));
      } else if(obj->InheritsFrom(RooHist::Class())){
        ::translateGraph(_hist,_plotVar,static_cast<RooHist*>(obj));
      }
    }

    DrawOpt opt(drawOptions) ;
    opt.invisible = invisible ;
    _items.emplace_back(obj,opt.rawOpt());
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Update our plot normalization over our plot variable's fit range,
/// which will be determined by the first suitable object added to our plot.

void RooPlot::updateFitRangeNorm(const TH1* hist)
{
  const TAxis* xa = ((TH1*)hist)->GetXaxis() ;
  _normBinWidth = (xa->GetXmax()-xa->GetXmin())/hist->GetNbinsX() ;
  _normNumEvts = hist->GetEntries()/_normBinWidth ;
}


////////////////////////////////////////////////////////////////////////////////
/// Update our plot normalization over our plot variable's fit range,
/// which will be determined by the first suitable object added to our plot.

void RooPlot::updateFitRangeNorm(const RooPlotable* rp, bool refreshNorm)
{
  if (_normNumEvts != 0) {

    // If refresh feature is disabled stop here
    if (!refreshNorm) return ;

    double corFac(1.0) ;
    if (dynamic_cast<const RooHist*>(rp)) corFac = _normBinWidth/rp->getFitRangeBinW() ;


    if (std::abs(rp->getFitRangeNEvt()/corFac-_normNumEvts)>1e-6) {
      coutI(Plotting) << "RooPlot::updateFitRangeNorm: New event count of " << rp->getFitRangeNEvt()/corFac
            << " will supercede previous event count of " << _normNumEvts << " for normalization of PDF projections" << endl ;
    }

    // Nominal bin width (i.e event density) is already locked in by previously drawn histogram
    // scale this histogram to match that density
    _normNumEvts = rp->getFitRangeNEvt()/corFac ;
    _normObj = rp ;
    // cout << "correction factor = " << _normBinWidth << "/" << rp->getFitRangeBinW() << endl ;
    // cout << "updating numevts to " << _normNumEvts << endl ;

  } else {

    _normObj = rp ;
    _normNumEvts = rp->getFitRangeNEvt() ;
    if (rp->getFitRangeBinW()) {
      _normBinWidth = rp->getFitRangeBinW() ;
    }

    // cout << "updating numevts to " << _normNumEvts << endl ;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Update our y-axis limits to accomodate an object whose spread
/// in y is (ymin,ymax). Use the specified y-axis label if we don't
/// have one assigned already.

void RooPlot::updateYAxis(double ymin, double ymax, const char *label)
{
  // force an implicit lower limit of zero if appropriate
  if(GetMinimum() == 0 && ymin > 0) ymin= 0;

  // calculate padded values
  double ypad= getPadFactor()*(ymax-ymin);
  ymax+= ypad;
  if(ymin < 0) ymin-= ypad;

  // update our limits if necessary
  if(GetMaximum() < ymax) {
    _defYmax = ymax ;
    SetMaximum(ymax);
    // if we don't do this - Unzoom on y-axis will reset upper bound to 1
    _hist->SetBinContent(1,ymax) ;
  }
  if(GetMinimum() > ymin) {
    _defYmin = ymin ;
    SetMinimum(ymin);
  }

  // use the specified y-axis label if we don't have one already
  if(0 == strlen(_hist->GetYaxis()->GetTitle())) _hist->SetYTitle(label);
}


////////////////////////////////////////////////////////////////////////////////
/// Draw this plot and all of the elements it contains. The specified options
/// only apply to the drawing of our frame. The options specified in our add...()
/// methods will be used to draw each object we contain.

void RooPlot::Draw(Option_t *option)
{
  TString optArg = option ;
  optArg.ToLower() ;

  // This draw options prevents the histogram with one dummy entry
  // to be drawn
  if (optArg.Contains("same")) {
    _hist->Draw("FUNCSAME");
  } else {
    _hist->Draw("FUNC");
  }

  for(auto const& item : _items) {
    TObject &obj = *item.first;
    DrawOpt opt(item.second.c_str()) ;
    if (!opt.invisible) {
       //LM:  in case of a TGraph derived object, do not use default "" option
       // which is "ALP" from 5.34.10 (and will then redrawn the axis) but  use "LP"
       if (!strlen(opt.drawOptions) && obj.IsA()->InheritsFrom(TGraph::Class()) ) strlcpy(opt.drawOptions,"LP",3);
       obj.Draw(opt.drawOptions);
    }
  }

  _hist->Draw("AXISSAME");
}



////////////////////////////////////////////////////////////////////////////////
/// Print frame name

void RooPlot::printName(ostream& os) const
{
  os << GetName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print frame title

void RooPlot::printTitle(ostream& os) const
{
  os << GetTitle() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print frame class name

void RooPlot::printClassName(ostream& os) const
{
  os << ClassName() ;
}



////////////////////////////////////////////////////////////////////////////////

void RooPlot::printArgs(ostream& os) const
{
  if (_plotVar) {
    os << "[" ;
    _plotVar->printStream(os,kName,kInline) ;
    os << "]" ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Print frame arguments

void RooPlot::printValue(ostream& os) const
{
  os << "(" ;
  bool first(true) ;
  for(auto const& item : _items) {
    TObject &obj = *item.first;
    if (first) {
      first=false ;
    } else {
      os << "," ;
    }
    if(obj.IsA()->InheritsFrom(RooPrintable::Class())) {
      auto po = dynamic_cast<RooPrintable&>(obj) ;
      // coverity[FORWARD_NULL]
      po.printStream(os,kClassName|kName,kInline) ;
    }
    // is it a TNamed subclass?
    else {
      os << obj.ClassName() << "::" << obj.GetName() ;
    }
  }
  os << ")" ;
}


////////////////////////////////////////////////////////////////////////////////
/// Frame detailed printing

void RooPlot::printMultiline(ostream& os, Int_t /*content*/, bool verbose, TString indent) const
{
  TString deeper(indent);
  deeper.Append("    ");
  if(0 != _plotVar) {
    os << indent << "RooPlot " << GetName() << " (" << GetTitle() << ") plots variable ";
    _plotVar->printStream(os,kName|kTitle,kSingleLine,"");
  }
  else {
    os << indent << "RooPlot " << GetName() << " (" << GetTitle() << ") has no associated plot variable" << endl ;
  }
  os << indent << "  Plot frame contains " << _items.size() << " object(s):" << endl;

  if(verbose) {
    Int_t i=0 ;
    for(auto const& item : _items) {
      TObject &obj = *item.first;
      os << deeper << "[" << i++ << "] (Options=\"" << item.second << "\") ";
      // Is this a printable object?
      if(obj.IsA()->InheritsFrom(RooPrintable::Class())) {
        auto po = dynamic_cast<RooPrintable&>(obj) ;
        po.printStream(os,kName|kClassName|kArgs|kExtras,kSingleLine) ;
      }
      // is it a TNamed subclass?
      else {
   os << obj.ClassName() << "::" << obj.GetName() << endl;
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return the name of the object at slot 'idx' in this RooPlot.
/// If the given index is out of range, return a null pointer

const char* RooPlot::nameOf(Int_t idx) const
{
  TObject* obj = _items.at(idx).first;
  if (!obj) {
    coutE(InputArguments) << "RooPlot::nameOf(" << GetName() << ") index " << idx << " out of range" << endl ;
    return 0 ;
  }
  return obj->GetName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the name of the object at slot 'idx' in this RooPlot.
/// If the given index is out of range, return a null pointer

TObject* RooPlot::getObject(Int_t idx) const
{
  TObject* obj = _items.at(idx).first;
  if (!obj) {
    coutE(InputArguments) << "RooPlot::getObject(" << GetName() << ") index " << idx << " out of range" << endl ;
    return 0 ;
  }
  return obj ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to the line attributes of the named object in this plot,
/// or zero if the named object does not exist or does not have line attributes.

TAttLine *RooPlot::getAttLine(const char *name) const
{
  return dynamic_cast<TAttLine*>(findObject(name));
}


////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to the fill attributes of the named object in this plot,
/// or zero if the named object does not exist or does not have fill attributes.

TAttFill *RooPlot::getAttFill(const char *name) const
{
  return dynamic_cast<TAttFill*>(findObject(name));
}


////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to the marker attributes of the named object in this plot,
/// or zero if the named object does not exist or does not have marker attributes.

TAttMarker *RooPlot::getAttMarker(const char *name) const
{
  return dynamic_cast<TAttMarker*>(findObject(name));
}


////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to the text attributes of the named object in this plot,
/// or zero if the named object does not exist or does not have text attributes.

TAttText *RooPlot::getAttText(const char *name) const
{
  return dynamic_cast<TAttText*>(findObject(name));
}



////////////////////////////////////////////////////////////////////////////////
/// Return a RooCurve pointer of the named object in this plot,
/// or zero if the named object does not exist or is not a RooCurve

RooCurve* RooPlot::getCurve(const char* name) const
{
  return dynamic_cast<RooCurve*>(findObject(name)) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return a RooCurve pointer of the named object in this plot,
/// or zero if the named object does not exist or is not a RooCurve

RooHist* RooPlot::getHist(const char* name) const
{
  return dynamic_cast<RooHist*>(findObject(name)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Remove object with given name, or last object added if no name is given.

void RooPlot::remove(const char* name, bool deleteToo)
{
  if(name == nullptr) {
    if(!_items.empty()) {
      if(deleteToo) delete _items.back().first;
      _items.pop_back();
    } else {
      coutE(InputArguments) << "RooPlot::remove(" << GetName() << ") ERROR: plot frame is empty, cannot remove last object" << endl ;
    }
  } else {
    auto item = findItem(name);
    if(item == _items.end()) {
      coutE(InputArguments) << "RooPlot::remove(" << GetName() << ") ERROR: no object found with name " << name << endl ;
    } else {
      if(deleteToo) delete item->first;
      _items.erase(item);
    }
  }
}


namespace {

template<class Iter>
void moveBefore(Iter before, Iter target) {
  auto d = std::distance(before, target);
  if(d > 0) std::rotate(before, target, target + 1);
  else if(d < 0) std::rotate(target, target+1, before);
}

} // namespace


////////////////////////////////////////////////////////////////////////////////
/// Change the order in which our contained objects are drawn so that
/// the target object is drawn just before the specified object.
/// Returns false if either object does not exist.

bool RooPlot::drawBefore(const char *before, const char *target)
{
  auto iterBefore = findItem(before);
  auto iterTarget = findItem(target);
  if(iterBefore == _items.end() || iterTarget == _items.end()) return false;
  moveBefore(iterBefore, iterTarget);
  return true;
}


////////////////////////////////////////////////////////////////////////////////
/// Change the order in which our contained objects are drawn so that
/// the target object is drawn just after the specified object.
/// Returns false if either object does not exist.

bool RooPlot::drawAfter(const char *after, const char *target)
{
  auto iterAfter = findItem(after);
  auto iterTarget = findItem(target);
  if(iterAfter == _items.end() || iterTarget == _items.end()) return false;
  moveBefore(iterAfter + 1, iterTarget);
  return true;
}


////////////////////////////////////////////////////////////////////////////////
/// Find the named object in our list of items and return a pointer
/// to it. Return zero and print a warning message if the named
/// object cannot be found. If no name is supplied the last object
/// added is returned.
///
/// Note that the returned pointer is to a
/// TObject and so will generally need casting. Use the getAtt...()
/// methods to change the drawing style attributes of a contained
/// object directly.

TObject *RooPlot::findObject(const char *name, const TClass* clas) const
{
  TObject *ret = nullptr;

  for(auto const& item : _items) {
    TObject &obj = *item.first;
    if ((!name || name[0] == '\0' || !TString(name).CompareTo(obj.GetName()))
        && (!clas || (obj.IsA()==clas))) {
      ret = &obj ;
    }
  }

  if (ret == nullptr) {
    coutE(InputArguments) << "RooPlot::findObject(" << GetName() << ") cannot find object " << (name?name:"<last>") << endl ;
  }
  return ret ;
}


RooPlot::Items::iterator RooPlot::findItem(std::string const& name)
{
  return std::find_if(_items.begin(), _items.end(), [&name](auto const& item){
              return name == item.first->GetName();
          });
}

RooPlot::Items::const_iterator RooPlot::findItem(std::string const& name) const
{
  return std::find_if(_items.begin(), _items.end(), [&name](auto const& item){
              return name == item.first->GetName();
          });
}


////////////////////////////////////////////////////////////////////////////////
/// Return the Draw() options registered for the named object. Return
/// an empty string if the named object cannot be found.

TString RooPlot::getDrawOptions(const char *name) const
{
  auto item = findItem(name);
  if(item == _items.end()) return "";

  return DrawOpt{item->second.c_str()}.drawOptions;
}


////////////////////////////////////////////////////////////////////////////////
/// Register the specified drawing options for the named object.
/// Return false if the named object cannot be found.

bool RooPlot::setDrawOptions(const char *name, TString options)
{
  auto item = findItem(name);
  if(item == _items.end()) return false;

  DrawOpt opt(item->second.c_str()) ;
  strlcpy(opt.drawOptions,options,128) ;
  item->second = opt.rawOpt();
  return true;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns true of object with given name is set to be invisible

bool RooPlot::getInvisible(const char* name) const
{
  auto item = findItem(name);
  if(item == _items.end()) return false;

  return DrawOpt{item->second.c_str()}.invisible ;
}


////////////////////////////////////////////////////////////////////////////////
/// If flag is true object with 'name' is set to be invisible
/// i.e. it is not drawn when Draw() is called

void RooPlot::setInvisible(const char* name, bool flag)
{
  auto item = findItem(name);
  if(item != _items.end()) {
    DrawOpt opt;
    opt.initialize(item->second.c_str());
    opt.invisible = flag;
    item->second = opt.rawOpt();
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Set maximum value of Y axis

void RooPlot::SetMaximum(double maximum)
{
  _hist->SetMaximum(maximum==-1111?_defYmax:maximum) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set minimum value of Y axis

void RooPlot::SetMinimum(double minimum)
{
  _hist->SetMinimum(minimum==-1111?_defYmin:minimum) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate and return reduced chi-squared between a curve and a histogram.
///
/// \param[in] curvename  Name of the curve or nullptr for last curve
/// \param[in] histname   Name of the histogram to compare to or nullptr for last added histogram
/// \param[in] nFitParam  If non-zero, reduce the number of degrees of freedom by this
/// number. This means that the curve was fitted to the data with nFitParam floating
/// parameters, which needs to be reflected in the calculation of \f$\chi^2 / \mathrm{ndf}\f$.
///
/// \return \f$ \chi^2 / \mathrm{ndf} \f$ between the plotted curve and the data.
///
/// \note The \f$ \chi^2 \f$ is calculated between a *plot of the original distribution* and the data.
/// It therefore has more rounding errors than directly calculating the \f$ \chi^2 \f$ from a PDF or
/// function. To do this, use RooChi2Var.
double RooPlot::chiSquare(const char* curvename, const char* histname, int nFitParam) const
{

  // Find curve object
  RooCurve* curve = (RooCurve*) findObject(curvename,RooCurve::Class()) ;
  if (!curve) {
    coutE(InputArguments) << "RooPlot::chiSquare(" << GetName() << ") cannot find curve" << endl ;
    return -1. ;
  }

  // Find histogram object
  RooHist* hist = (RooHist*) findObject(histname,RooHist::Class()) ;
  if (!hist) {
    coutE(InputArguments) << "RooPlot::chiSquare(" << GetName() << ") cannot find histogram" << endl ;
    return -1. ;
  }

  return curve->chiSquare(*hist,nFitParam) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return a RooHist (derives from TGraphAsymErrors) containing the residuals of a histogram.
/// The plotting range of the graph is adapted to the plotting range of the current plot.
///
/// \param histname Name of the data histogram.
/// Passing an empty string or `nullptr` will create residuals of the last-plotted histogram.
/// \param curvename Name of the curve to compare to data.
/// Passing an empty string or `nullptr` will create residuals of the last-plotted curve.
/// \param normalize If true, the residuals are divided by the error
/// of the histogram, effectively returning a pull histogram.
/// \param useAverage If true, the histogram is compared with the curve averaged in each bin.
/// Otherwise, the curve is evaluated at the bin centres, which is not accurate for strongly curved distributions.
RooHist* RooPlot::residHist(const char* histname, const char* curvename, bool normalize, bool useAverage) const
{
  // Find all curve objects with the name "curvename" or the name of the last
  // plotted curve (there might be multiple in the case of multi-range fits).
  std::vector<RooCurve *> curves;

  for(auto it = _items.rbegin(); it != _items.rend(); ++it) {
    TObject &obj = *it->first;
    if(obj.IsA() == RooCurve::Class()) {
      // If no curvename was passed, we take by default the last curve and all
      // other curves that have the same name
      if((!curvename || curvename[0] == '\0') || std::string(curvename) == obj.GetName()) {
        curvename = obj.GetName();
        curves.push_back(static_cast<RooCurve*>(&obj));
      }
    }
  }

  if (curves.empty()) {
    coutE(InputArguments) << "RooPlot::residHist(" << GetName() << ") cannot find curve" << std::endl;
    return nullptr;
  }

  // Find histogram object
  auto hist = static_cast<RooHist*>(findObject(histname,RooHist::Class()));
  if (!hist) {
    coutE(InputArguments) << "RooPlot::residHist(" << GetName() << ") cannot find histogram" << std::endl;
    return nullptr;
  }

  auto residHist = hist->createEmptyResidHist(*curves.front(), normalize);

  // We consider all curves with the same name as long as the ranges don't
  // overlap, to create also the full residual plot in multi-range fits.
  std::vector<std::pair<double, double>> coveredRanges;
  for(RooCurve * curve : curves) {
    const double xmin = curve->GetPointX(0);
    const double xmax = curve->GetPointX(curve->GetN() - 1);

    for(auto const& prevRange : coveredRanges) {
      const double pxmin = prevRange.first;
      const double pxmax = prevRange.second;
      // If either xmin or xmax is within any previous range, the ranges
      // overloap and it makes to sense to also consider this curve for the
      // residuals (i.e., they can't come from a multi-range fit).
      if((pxmax > xmin && pxmin <= xmin) || (pxmax > xmax && pxmin <= xmax)) {
        continue;
      }
    }

    coveredRanges.emplace_back(xmin, xmax);

    hist->fillResidHist(*residHist, *curve, normalize, useAverage);
  }
  residHist->GetHistogram()->GetXaxis()->SetRangeUser(_hist->GetXaxis()->GetXmin(), _hist->GetXaxis()->GetXmax());
  residHist->GetHistogram()->GetXaxis()->SetTitle(_hist->GetXaxis()->GetTitle());
  residHist->GetHistogram()->GetYaxis()->SetTitle(normalize ? "(Data - curve) / #sigma_{data}" : "Data - curve");
  return residHist.release();
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize the DrawOpt helper class

void RooPlot::DrawOpt::initialize(const char* inRawOpt)
{
  if (!inRawOpt) {
    drawOptions[0] = 0 ;
    invisible=false ;
    return ;
  }
  strlcpy(drawOptions,inRawOpt,128) ;
  strtok(drawOptions,":") ;
  const char* extraOpt = strtok(0,":") ;
  if (extraOpt) {
    invisible =  (extraOpt[0]=='I') ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Return the raw draw options

const char* RooPlot::DrawOpt::rawOpt() const
{
  static char buf[128] ;
  strlcpy(buf,drawOptions,128) ;
  if (invisible) {
    strlcat(buf,":I",128) ;
  }
  return buf ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the number of events that is associated with the range [xlo,xhi]
/// This method is only fully functional for ranges not equal to the full
/// range if the object that inserted the normalization data provided
/// a link to an external object that can calculate the event count in
/// in sub ranges. An error will be printed if this function is used
/// on sub-ranges while that information is not available

double RooPlot::getFitRangeNEvt(double xlo, double xhi) const
{
  double scaleFactor = 1.0 ;
  if (_normObj) {
    scaleFactor = _normObj->getFitRangeNEvt(xlo,xhi)/_normObj->getFitRangeNEvt() ;
  } else {
    coutW(Plotting) << "RooPlot::getFitRangeNEvt(" << GetName() << ") WARNING: Unable to obtain event count in range "
          << xlo << " to " << xhi << ", substituting full event count" << endl ;
  }
  return getFitRangeNEvt()*scaleFactor ;
}


////////////////////////////////////////////////////////////////////////////////
/// Set the name of the RooPlot to 'name'

void RooPlot::SetName(const char *name)
{
  if (_dir) _dir->GetList()->Remove(this);
  TNamed::SetName(name) ;
  if (_dir) _dir->GetList()->Add(this);
}


////////////////////////////////////////////////////////////////////////////////
/// Set the name and title of the RooPlot to 'name' and 'title'

void RooPlot::SetNameTitle(const char *name, const char* title)
{
  if (_dir) _dir->GetList()->Remove(this);
  TNamed::SetNameTitle(name,title) ;
  if (_dir) _dir->GetList()->Add(this);
}


////////////////////////////////////////////////////////////////////////////////
/// Set the title of the RooPlot to 'title'

void RooPlot::SetTitle(const char* title)
{
  TNamed::SetTitle(title) ;
  _hist->SetTitle(title) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define default print options, for a given print style

Int_t RooPlot::defaultPrintContents(Option_t* /*opt*/) const
{
  return kName|kArgs|kValue ;
}



/// \see TH1::GetXaxis()
TAxis* RooPlot::GetXaxis() const { return _hist->GetXaxis() ; }
/// \see TH1::GetYaxis()
TAxis* RooPlot::GetYaxis() const { return _hist->GetYaxis() ; }
/// \see TH1::GetNbinsX()
Int_t  RooPlot::GetNbinsX() const { return _hist->GetNbinsX() ; }
/// \see TH1::GetNdivisions()
Int_t  RooPlot::GetNdivisions(Option_t* axis) const { return _hist->GetNdivisions(axis) ; }
/// \see TH1::GetMinimum()
double  RooPlot::GetMinimum(double minval) const { return _hist->GetMinimum(minval) ; }
/// \see TH1::GetMaximum()
double   RooPlot::GetMaximum(double maxval) const { return _hist->GetMaximum(maxval) ; }


/// \see TH1::SetAxisColor()
void RooPlot::SetAxisColor(Color_t color, Option_t* axis) { _hist->SetAxisColor(color,axis) ; }
/// \see TH1::SetAxisRange()
void RooPlot::SetAxisRange(double xmin, double xmax, Option_t* axis) { _hist->SetAxisRange(xmin,xmax,axis) ; }
/// \see TH1::SetBarOffset()
void RooPlot::SetBarOffset(Float_t offset) { _hist->SetBarOffset(offset) ; }
/// \see TH1::SetBarWidth()
void RooPlot::SetBarWidth(Float_t width) { _hist->SetBarWidth(width) ; }
/// \see TH1::SetContour()
void RooPlot::SetContour(Int_t nlevels, const double* levels) { _hist->SetContour(nlevels,levels) ; }
/// \see TH1::SetContourLevel()
void RooPlot::SetContourLevel(Int_t level, double value) { _hist->SetContourLevel(level,value) ; }
/// \see TH1::SetDrawOption()
void RooPlot::SetDrawOption(Option_t* option) { _hist->SetDrawOption(option) ; }
/// \see TH1::SetFillAttributes()
void RooPlot::SetFillAttributes() { _hist->SetFillAttributes() ; }
/// \see TH1::SetFillColor()
void RooPlot::SetFillColor(Color_t fcolor) { _hist->SetFillColor(fcolor) ; }
/// \see TH1::SetFillStyle()
void RooPlot::SetFillStyle(Style_t fstyle) { _hist->SetFillStyle(fstyle) ; }
/// \see TH1::SetLabelColor()
void RooPlot::SetLabelColor(Color_t color, Option_t* axis) { _hist->SetLabelColor(color,axis) ; }
/// \see TH1::SetLabelFont()
void RooPlot::SetLabelFont(Style_t font, Option_t* axis) { _hist->SetLabelFont(font,axis) ; }
/// \see TH1::SetLabelOffset()
void RooPlot::SetLabelOffset(Float_t offset, Option_t* axis) { _hist->SetLabelOffset(offset,axis) ; }
/// \see TH1::SetLabelSize()
void RooPlot::SetLabelSize(Float_t size, Option_t* axis) { _hist->SetLabelSize(size,axis) ; }
/// \see TH1::SetLineAttributes()
void RooPlot::SetLineAttributes() { _hist->SetLineAttributes() ; }
/// \see TH1::SetLineColor()
void RooPlot::SetLineColor(Color_t lcolor) { _hist->SetLineColor(lcolor) ; }
/// \see TH1::SetLineStyle()
void RooPlot::SetLineStyle(Style_t lstyle) { _hist->SetLineStyle(lstyle) ; }
/// \see TH1::SetLineWidth()
void RooPlot::SetLineWidth(Width_t lwidth) { _hist->SetLineWidth(lwidth) ; }
/// \see TH1::SetMarkerAttributes()
void RooPlot::SetMarkerAttributes() { _hist->SetMarkerAttributes() ; }
/// \see TH1::SetMarkerColor()
void RooPlot::SetMarkerColor(Color_t tcolor) { _hist->SetMarkerColor(tcolor) ; }
/// \see TH1::SetMarkerSize()
void RooPlot::SetMarkerSize(Size_t msize) { _hist->SetMarkerSize(msize) ; }
/// \see TH1::SetMarkerStyle()
void RooPlot::SetMarkerStyle(Style_t mstyle) { _hist->SetMarkerStyle(mstyle) ; }
/// \see TH1::SetNdivisions()
void RooPlot::SetNdivisions(Int_t n, Option_t* axis) { _hist->SetNdivisions(n,axis) ; }
/// \see TH1::SetOption()
void RooPlot::SetOption(Option_t* option) { _hist->SetOption(option) ; }
/// Like TH1::SetStats(), but statistics boxes are *off* by default in RooFit.
void RooPlot::SetStats(bool stats) { _hist->SetStats(stats) ; }
/// \see TH1::SetTickLength()
void RooPlot::SetTickLength(Float_t length, Option_t* axis) { _hist->SetTickLength(length,axis) ; }
/// \see TH1::SetTitleFont()
void RooPlot::SetTitleFont(Style_t font, Option_t* axis) { _hist->SetTitleFont(font,axis) ; }
/// \see TH1::SetTitleOffset()
void RooPlot::SetTitleOffset(Float_t offset, Option_t* axis) { _hist->SetTitleOffset(offset,axis) ; }
/// \see TH1::SetTitleSize()
void RooPlot::SetTitleSize(Float_t size, Option_t* axis) { _hist->SetTitleSize(size,axis) ; }
/// \see TH1::SetXTitle()
void RooPlot::SetXTitle(const char *title) { _hist->SetXTitle(title) ; }
/// \see TH1::SetYTitle()
void RooPlot::SetYTitle(const char *title) { _hist->SetYTitle(title) ; }
/// \see TH1::SetZTitle()
void RooPlot::SetZTitle(const char *title) { _hist->SetZTitle(title) ; }




////////////////////////////////////////////////////////////////////////////////
/// Plot RooPlot when double-clicked in browser

void RooPlot::Browse(TBrowser * /*b*/)
{
  Draw();
  gPad->Update();
}




////////////////////////////////////////////////////////////////////////////////

void RooPlot::Streamer(TBuffer &R__b)
{
  // Custom streamer, needed for backward compatibility

  if (R__b.IsReading()) {
    const bool oldAddDir = TH1::AddDirectoryStatus();
    TH1::AddDirectory(false);

    // The default c'tor might have registered this with a TDirectory.
    // Streaming the TNamed will make this not retrievable anymore, so
    // unregister first.
    if (_dir)
      _dir->Remove(this);

    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
    if (R__v > 1) {
      R__b.ReadClassBuffer(RooPlot::Class(),this,R__v,R__s,R__c);
    } else {
      // backward compatible streamer code here
      // Version 1 of RooPlot was deriving from TH1 and RooPrintable
      // Version 2 derives instead from TNamed and RooPrintable
      _hist = new TH1F();
      _hist->TH1::Streamer(R__b);
      SetName(_hist->GetName());
      SetTitle(_hist->GetTitle());
      RooPrintable::Streamer(R__b);
      {
        TList itemsList;
        itemsList.Streamer(R__b);
        RooPlot::fillItemsFromTList(_items, itemsList);
      }
      R__b >> _padFactor;
      R__b >> _plotVar;
      R__b >> _plotVarSet;
      R__b >> _normVars;
      R__b >> _normNumEvts;
      R__b >> _normBinWidth;
      R__b >> _defYmin;
      R__b >> _defYmax;
      R__b.CheckByteCount(R__s, R__c, RooPlot::IsA());
    }

    TH1::AddDirectory(oldAddDir);
    if (_dir)
      _dir->Append(this);

  } else {
    R__b.WriteClassBuffer(RooPlot::Class(),this);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Build a legend that contains all objects that have been drawn on the plot.
std::unique_ptr<TLegend> RooPlot::BuildLegend() const {
  auto leg = std::make_unique<TLegend>(0.5, 0.7, 0.9, 0.9);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  for (std::size_t i=0; i < _items.size(); ++i) {
    leg->AddEntry(getObject(i));
  }

  return leg;
}

/// RooFit-internal function for backwards compatibility.
void RooPlot::fillItemsFromTList(RooPlot::Items & items, TList const& tlist) {
  for(TObject * obj : tlist) {
    items.emplace_back(obj, obj->GetOption());
  }
}

/// Replaces the pointer to the plot variable with a pointer to a clone of the
/// plot variable that is owned by this RooPlot. The RooPlot references the
/// plotted variable by non-owning pointer by default since ROOT 6.28, which
/// resulted in a big speedup when plotting complicated pdfs that are expensive
/// to clone. However, going back to an owned clone is useful in rare cases.
/// For example in the RooUnitTest, where the registered plots need to live
/// longer than the scope of the unit test.
void RooPlot::createInternalPlotVarClone() {
  // If the plot variable is already cloned, we don't need to do anything.
  if(_plotVarSet) return;
  _plotVarSet = static_cast<RooArgSet*>(RooArgSet(*_plotVar).snapshot());
  _plotVar = static_cast<RooAbsRealLValue*>(_plotVarSet->find(_plotVar->GetName()));
}
