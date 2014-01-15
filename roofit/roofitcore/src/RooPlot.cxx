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
// A RooPlot is a plot frame and a container for graphics objects
// within that frame. As a frame, it provides the TH1-style public interface
// for settting plot ranges, configuring axes, etc. As a container, it
// holds an arbitrary set of objects that might be histograms of data,
// curves representing a fit model, or text labels. Use the Draw()
// method to draw a frame and the objects it contains. Use the various
// add...() methods to add objects to be drawn.  In general, the
// add...() methods create a private copy of the object you pass them
// and return a pointer to this copy. The caller owns the input object
// and this class owns the returned object.
// <p>
// All RooAbsReal and RooAbsData derived classes implement plotOn()
// functions that facilitate to plot themselves on a given RooPlot, e.g.
// <pre>
// RooPlot *frame = x.frame() ;
// data.plotOn(frame) ;
// pdf.plotOn(frame) ;
// </pre>
// These high level functions also take care of any projections
// or other mappings that need to be made to plot a multi-dimensional
// object onto a one-dimensional plot.
// END_HTML
//


#include "RooFit.h"

#include "TClass.h"
#include "TH1D.h"
#include "TBrowser.h"
#include "TPad.h"

#include "RooPlot.h"
#include "RooAbsReal.h"
#include "RooAbsRealLValue.h"
#include "RooPlotable.h"
#include "RooArgSet.h"
#include "RooCurve.h"
#include "RooHist.h"
#include "RooMsgService.h"

#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"
#include "TAttText.h"
#include "TDirectory.h"
#include "TDirectoryFile.h"

#include "Riostream.h"
#include <string.h>
#include <assert.h>

using namespace std;

ClassImp(RooPlot)
;

Bool_t RooPlot::_addDirStatus = kTRUE ;

Bool_t RooPlot::addDirectoryStatus() { return _addDirStatus ; }
Bool_t RooPlot::setAddDirectoryStatus(Bool_t flag) { Bool_t ret = flag ; _addDirStatus = flag ; return ret ; }


//_____________________________________________________________________________
RooPlot::RooPlot() : _hist(0), _plotVarClone(0), _plotVarSet(0), _normVars(0), _normObj(0), _dir(0)
{
  // Default constructor
  // coverity[UNINIT_CTOR]

  _iterator= _items.MakeIterator() ;

  if (gDirectory && addDirectoryStatus()) {
    _dir = gDirectory ;
    gDirectory->Append(this) ;
  }
}


//_____________________________________________________________________________
RooPlot::RooPlot(Double_t xmin, Double_t xmax) :
  _hist(0), _items(), _plotVarClone(0), _plotVarSet(0), _normObj(0),
  _defYmin(1e-5), _defYmax(1), _dir(0)
{
  // Constructor of RooPlot with range [xmin,xmax]

  Bool_t histAddDirStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE) ;

  _hist = new TH1D(histName(),"A RooPlot",100,xmin,xmax) ;
  _hist->Sumw2(kFALSE) ;
  _hist->GetSumw2()->Set(0) ;

  
  TH1::AddDirectory(histAddDirStatus) ;


  // Create an empty frame with the specified x-axis limits.
  initialize();

}



//_____________________________________________________________________________
RooPlot::RooPlot(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax) :
  _hist(0), _items(), _plotVarClone(0),
  _plotVarSet(0), _normObj(0), _defYmin(1e-5), _defYmax(0), _dir(0)
{
  // Construct of a two-dimensioanl RooPlot with ranges [xmin,xmax] x [ymin,ymax]

  Bool_t histAddDirStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE) ;

  _hist = new TH1D(histName(),"A RooPlot",100,xmin,xmax) ;
  _hist->Sumw2(kFALSE) ;
  _hist->GetSumw2()->Set(0) ;

  TH1::AddDirectory(histAddDirStatus) ;

  SetMinimum(ymin);
  SetMaximum(ymax);
  initialize();
}


//_____________________________________________________________________________
RooPlot::RooPlot(const RooAbsRealLValue &var1, const RooAbsRealLValue &var2) :
  _hist(0), _items(),
  _plotVarClone(0), _plotVarSet(0), _normObj(0), _defYmin(1e-5), _defYmax(0), _dir(0)
{
  // Construct a two-dimensional RooPlot with ranges and properties taken
  // from variables var1 and var2

  Bool_t histAddDirStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE) ;

  _hist = new TH1D(histName(),"A RooPlot",100,var1.getMin(),var1.getMax()) ;
  _hist->Sumw2(kFALSE) ;
  _hist->GetSumw2()->Set(0) ;

  TH1::AddDirectory(histAddDirStatus) ;

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
  SetXTitle(var1.getTitle(kTRUE));
  SetYTitle(var2.getTitle(kTRUE));
  initialize();
}


//_____________________________________________________________________________
RooPlot::RooPlot(const RooAbsRealLValue &var1, const RooAbsRealLValue &var2,
		 Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax) :
  _hist(0), _items(), _plotVarClone(0),
  _plotVarSet(0), _normObj(0), _defYmin(1e-5), _defYmax(0), _dir(0)
{
  // Construct a two-dimensional RooPlot with ranges and properties taken
  // from variables var1 and var2 but with an overriding range definition
  // of [xmin,xmax] x [ymin,ymax]

  Bool_t histAddDirStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE) ;

  _hist = new TH1D(histName(),"A RooPlot",100,xmin,xmax) ;
  _hist->Sumw2(kFALSE) ;
  _hist->GetSumw2()->Set(0) ;

  TH1::AddDirectory(histAddDirStatus) ;

  SetMinimum(ymin);
  SetMaximum(ymax);
  SetXTitle(var1.getTitle(kTRUE));
  SetYTitle(var2.getTitle(kTRUE));
  initialize();
}


//_____________________________________________________________________________
RooPlot::RooPlot(const char* name, const char* title, const RooAbsRealLValue &var, Double_t xmin, Double_t xmax, Int_t nbins) :
  _hist(0), _items(),
  _plotVarClone(0), _plotVarSet(0), _normObj(0), _defYmin(1e-5), _defYmax(1), _dir(0)
{
  // Create an 1-dimensional with all properties taken from 'var', but
  // with an explicit range [xmin,xmax] and a default binning of 'nbins'

  Bool_t histAddDirStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE) ;

  _hist = new TH1D(name,title,nbins,xmin,xmax) ;
  _hist->Sumw2(kFALSE) ;
  _hist->GetSumw2()->Set(0) ;

  TH1::AddDirectory(histAddDirStatus) ;

  // plotVar can be a composite in case of a RooDataSet::plot, need deepClone
  _plotVarSet = (RooArgSet*) RooArgSet(var).snapshot() ;
  _plotVarClone= (RooAbsRealLValue*)_plotVarSet->find(var.GetName()) ;

  TString xtitle= var.getTitle(kTRUE);
  SetXTitle(xtitle.Data());

  initialize();

  _normBinWidth = (xmax-xmin)/nbins ;
}


//_____________________________________________________________________________
RooPlot::RooPlot(const RooAbsRealLValue &var, Double_t xmin, Double_t xmax, Int_t nbins) :
  _hist(0), _items(),
  _plotVarClone(0), _plotVarSet(0), _normObj(0), _defYmin(1e-5), _defYmax(1), _dir(0)
{
  // Create an 1-dimensional with all properties taken from 'var', but
  // with an explicit range [xmin,xmax] and a default binning of 'nbins'

  Bool_t histAddDirStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE) ;

  _hist = new TH1D(histName(),"RooPlot",nbins,xmin,xmax) ;
  _hist->Sumw2(kFALSE) ;
  _hist->GetSumw2()->Set(0) ;

  TH1::AddDirectory(histAddDirStatus) ;

  // plotVar can be a composite in case of a RooDataSet::plot, need deepClone
  _plotVarSet = (RooArgSet*) RooArgSet(var).snapshot() ;
  _plotVarClone= (RooAbsRealLValue*)_plotVarSet->find(var.GetName()) ;

  TString xtitle= var.getTitle(kTRUE);
  SetXTitle(xtitle.Data());

  TString title("A RooPlot of \"");
  title.Append(var.getTitle());
  title.Append("\"");
  SetTitle(title.Data());
  initialize();

  _normBinWidth = (xmax-xmin)/nbins ;
}



//_____________________________________________________________________________
RooPlot* RooPlot::emptyClone(const char* name)
{
  // Return empty clone of current RooPlot

  RooPlot* clone = new RooPlot(*_plotVarClone,_hist->GetXaxis()->GetXmin(),_hist->GetXaxis()->GetXmax(),_hist->GetNbinsX()) ;
  clone->SetName(name) ;
  return clone ;
}


//_____________________________________________________________________________
void RooPlot::initialize()
{
  // Perform initialization that is common to all constructors.

  SetName(histName()) ;

  if (gDirectory && addDirectoryStatus()) {
    _dir = gDirectory ;
    gDirectory->Append(this) ;
  }

  // We do not have useful stats of our own
  _hist->SetStats(kFALSE);
  // Default vertical padding of our enclosed objects
  setPadFactor(0.05);
  // We don't know our normalization yet
  _normNumEvts= 0;
  _normBinWidth = 0;
  _normVars= 0;
  // Create an iterator over our enclosed objects
  _iterator= _items.MakeIterator();
  assert(0 != _iterator);
}


//_____________________________________________________________________________
TString RooPlot::histName() const
{
  // Construct automatic name of internal TH1
  if (_plotVarClone) {
    return TString(Form("frame_%s_%lx",_plotVarClone->GetName(),(ULong_t)this)) ;
  } else {
    return TString(Form("frame_%lx",(ULong_t)this)) ;
  }
}


//_____________________________________________________________________________
RooPlot::~RooPlot()
{
  // Destructor

  // Delete the items in our container and our iterator.
  if (_dir) {
    if (!_dir->TestBit(TDirectoryFile::kCloseDirectory)) {
      _dir->GetList()->RecursiveRemove(this) ;
    }
  }

  _items.Delete();
  delete _iterator;
  if(_plotVarSet) delete _plotVarSet;
  if(_normVars) delete _normVars;
  delete _hist ;

}


//_____________________________________________________________________________
void RooPlot::updateNormVars(const RooArgSet &vars)
{
  // Install the given set of observables are reference normalization
  // variables for this frame. These observables are e.g. later used
  // to automatically project out observables when plotting functions
  // on this frame. This function is only effective when called the
  // first time on a frame

  if(0 == _normVars) _normVars= (RooArgSet*) vars.snapshot(kTRUE);
}


//_____________________________________________________________________________
Stat_t RooPlot::GetBinContent(Int_t /*i*/) const {
  // A plot object is a frame without any bin contents of its own so this
  // method always returns zero.
  return 0;
}


//_____________________________________________________________________________
Stat_t RooPlot::GetBinContent(Int_t, Int_t) const
{
  // A plot object is a frame without any bin contents of its own so this
  // method always returns zero.
  return 0;
}


//_____________________________________________________________________________
Stat_t RooPlot::GetBinContent(Int_t, Int_t, Int_t) const
{
  // A plot object is a frame without any bin contents of its own so this
  // method always returns zero.
  return 0;
}



//_____________________________________________________________________________
void RooPlot::addObject(TObject *obj, Option_t *drawOptions, Bool_t invisible)
{
  // Add a generic object to this plot. The specified options will be
  // used to Draw() this object later. The caller transfers ownership
  // of the object with this call, and the object will be deleted
  // when its containing plot object is destroyed.

  if(0 == obj) {
    coutE(InputArguments) << fName << "::addObject: called with a null pointer" << endl;
    return;
  }
  DrawOpt opt(drawOptions) ;
  opt.invisible = invisible ;
  _items.Add(obj,opt.rawOpt());
}


//_____________________________________________________________________________
void RooPlot::addTH1(TH1 *hist, Option_t *drawOptions, Bool_t invisible)
{
  // Add a TH1 histogram object to this plot. The specified options
  // will be used to Draw() this object later. "SAME" will be added to
  // the options if they are not already present. The caller transfers
  // ownership of the object with this call, and the object will be
  // deleted when its containing plot object is destroyed.

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


//_____________________________________________________________________________
void RooPlot::addPlotable(RooPlotable *plotable, Option_t *drawOptions, Bool_t invisible, Bool_t refreshNorm)
{
  // Add the specified plotable object to our plot. Increase our y-axis
  // limits to fit this object if necessary. The default lower-limit
  // is zero unless we are plotting an object that takes on negative values.
  // This call transfers ownership of the plotable object to this class.
  // The plotable object will be deleted when this plot object is deleted.

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
    DrawOpt opt(drawOptions) ;
    opt.invisible = invisible ;
    _items.Add(obj,opt.rawOpt());
  }
}


//_____________________________________________________________________________
void RooPlot::updateFitRangeNorm(const TH1* hist)
{
  // Update our plot normalization over our plot variable's fit range,
  // which will be determined by the first suitable object added to our plot.

  const TAxis* xa = ((TH1*)hist)->GetXaxis() ;
  _normBinWidth = (xa->GetXmax()-xa->GetXmin())/hist->GetNbinsX() ;
  _normNumEvts = hist->GetEntries()/_normBinWidth ;
}


//_____________________________________________________________________________
void RooPlot::updateFitRangeNorm(const RooPlotable* rp, Bool_t refreshNorm)
{
  // Update our plot normalization over our plot variable's fit range,
  // which will be determined by the first suitable object added to our plot.

  if (_normNumEvts != 0) {

    // If refresh feature is disabled stop here
    if (!refreshNorm) return ;

    Double_t corFac(1.0) ;
    if (dynamic_cast<const RooHist*>(rp)) corFac = _normBinWidth/rp->getFitRangeBinW() ;


    if (fabs(rp->getFitRangeNEvt()/corFac-_normNumEvts)>1e-6) {
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



//_____________________________________________________________________________
void RooPlot::updateYAxis(Double_t ymin, Double_t ymax, const char *label)
{
  // Update our y-axis limits to accomodate an object whose spread
  // in y is (ymin,ymax). Use the specified y-axis label if we don't
  // have one assigned already.

  // force an implicit lower limit of zero if appropriate
  if(GetMinimum() == 0 && ymin > 0) ymin= 0;

  // calculate padded values
  Double_t ypad= getPadFactor()*(ymax-ymin);
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


//_____________________________________________________________________________
void RooPlot::Draw(Option_t *option)
{
  // Draw this plot and all of the elements it contains. The specified options
  // only apply to the drawing of our frame. The options specified in our add...()
  // methods will be used to draw each object we contain.

  TString optArg = option ;
  optArg.ToLower() ;

  // This draw options prevents the histogram with one dummy entry 
  // to be drawn 
  if (optArg.Contains("same")) {
    _hist->Draw("FUNCSAME");
  } else {
    _hist->Draw("FUNC");
  }

  _iterator->Reset();
  TObject *obj = 0;
  while((obj= _iterator->Next())) {
    DrawOpt opt(_iterator->GetOption()) ;
    if (!opt.invisible) {
       //LM:  in case of a TGraph derived object, do not use default "" option
       // which is "ALP" from 5.34.10 (and will then redrawn the axis) but  use "LP" 
       if (!strlen(opt.drawOptions) && obj->IsA()->InheritsFrom(TGraph::Class()) ) strlcpy(opt.drawOptions,"LP",3); 
       obj->Draw(opt.drawOptions);
    }
  }

  _hist->Draw("AXISSAME");
}



//_____________________________________________________________________________
void RooPlot::printName(ostream& os) const
{
  // Print frame name
  os << GetName() ;
}


//_____________________________________________________________________________
void RooPlot::printTitle(ostream& os) const
{
  // Print frame title
  os << GetTitle() ;
}


//_____________________________________________________________________________
void RooPlot::printClassName(ostream& os) const
{
  // Print frame class name
  os << IsA()->GetName() ;
}



//_____________________________________________________________________________
void RooPlot::printArgs(ostream& os) const
{
  if (_plotVarClone) {
    os << "[" ;
    _plotVarClone->printStream(os,kName,kInline) ;
    os << "]" ;
  }
}



//_____________________________________________________________________________
void RooPlot::printValue(ostream& os) const
{
  // Print frame arguments
  os << "(" ;
  _iterator->Reset();
  TObject *obj = 0;
  Bool_t first(kTRUE) ;
  while((obj= _iterator->Next())) {
    if (first) {
      first=kFALSE ;
    } else {
      os << "," ;
    }
    if(obj->IsA()->InheritsFrom(RooPrintable::Class())) {
      RooPrintable* po = dynamic_cast<RooPrintable*>(obj) ;
      // coverity[FORWARD_NULL]
      po->printStream(os,kClassName|kName,kInline) ;
    }
    // is it a TNamed subclass?
    else {
      os << obj->ClassName() << "::" << obj->GetName() ;
    }
  }
  os << ")" ;
}


//_____________________________________________________________________________
void RooPlot::printMultiline(ostream& os, Int_t /*content*/, Bool_t verbose, TString indent) const
{
  // Frame detailed printing

  TString deeper(indent);
  deeper.Append("    ");
  if(0 != _plotVarClone) {
    os << indent << "RooPlot " << GetName() << " (" << GetTitle() << ") plots variable ";
    _plotVarClone->printStream(os,kName|kTitle,kSingleLine,"");
  }
  else {
    os << indent << "RooPlot " << GetName() << " (" << GetTitle() << ") has no associated plot variable" << endl ;
  }
  os << indent << "  Plot frame contains " << _items.GetSize() << " object(s):" << endl;

  if(verbose) {
    _iterator->Reset();
    TObject *obj = 0;
    Int_t i=0 ;
    while((obj= _iterator->Next())) {
      os << deeper << "[" << i++ << "] (Options=\"" << _iterator->GetOption() << "\") ";
      // Is this a printable object?
      if(obj->IsA()->InheritsFrom(RooPrintable::Class())) {
	RooPrintable* po = dynamic_cast<RooPrintable*>(obj) ;
	if (po) {
	  po->printStream(os,kName|kClassName|kArgs|kExtras,kSingleLine) ;
	}
      }
      // is it a TNamed subclass?
      else {
	os << obj->ClassName() << "::" << obj->GetName() << endl;
      }
    }
  }
}



//_____________________________________________________________________________
const char* RooPlot::nameOf(Int_t idx) const
{
  // Return the name of the object at slot 'idx' in this RooPlot.
  // If the given index is out of range, return a null pointer

  TObject* obj = _items.At(idx) ;
  if (!obj) {
    coutE(InputArguments) << "RooPlot::nameOf(" << GetName() << ") index " << idx << " out of range" << endl ;
    return 0 ;
  }
  return obj->GetName() ;
}



//_____________________________________________________________________________
TObject* RooPlot::getObject(Int_t idx) const
{
  // Return the name of the object at slot 'idx' in this RooPlot.
  // If the given index is out of range, return a null pointer

  TObject* obj = _items.At(idx) ;
  if (!obj) {
    coutE(InputArguments) << "RooPlot::getObject(" << GetName() << ") index " << idx << " out of range" << endl ;
    return 0 ;
  }
  return obj ;
}



//_____________________________________________________________________________
TAttLine *RooPlot::getAttLine(const char *name) const
{
  // Return a pointer to the line attributes of the named object in this plot,
  // or zero if the named object does not exist or does not have line attributes.

  return dynamic_cast<TAttLine*>(findObject(name));
}


//_____________________________________________________________________________
TAttFill *RooPlot::getAttFill(const char *name) const
{
  // Return a pointer to the fill attributes of the named object in this plot,
  // or zero if the named object does not exist or does not have fill attributes.

  return dynamic_cast<TAttFill*>(findObject(name));
}


//_____________________________________________________________________________
TAttMarker *RooPlot::getAttMarker(const char *name) const
{
  // Return a pointer to the marker attributes of the named object in this plot,
  // or zero if the named object does not exist or does not have marker attributes.

  return dynamic_cast<TAttMarker*>(findObject(name));
}


//_____________________________________________________________________________
TAttText *RooPlot::getAttText(const char *name) const
{
  // Return a pointer to the text attributes of the named object in this plot,
  // or zero if the named object does not exist or does not have text attributes.

  return dynamic_cast<TAttText*>(findObject(name));
}



//_____________________________________________________________________________
RooCurve* RooPlot::getCurve(const char* name) const
{
  // Return a RooCurve pointer of the named object in this plot,
  // or zero if the named object does not exist or is not a RooCurve

  return dynamic_cast<RooCurve*>(findObject(name)) ;
}


//_____________________________________________________________________________
RooHist* RooPlot::getHist(const char* name) const
{
  // Return a RooCurve pointer of the named object in this plot,
  // or zero if the named object does not exist or is not a RooCurve

  return dynamic_cast<RooHist*>(findObject(name)) ;
}



//_____________________________________________________________________________
void RooPlot::remove(const char* name, Bool_t deleteToo)
{
  // Remove object with given name, or last object added if no name is given.
  // If deleteToo is true (default), the object removed from the RooPlot is
  // also deleted.

  TObject* obj = findObject(name) ;
  if (!obj) {
    if (name) {
      coutE(InputArguments) << "RooPlot::remove(" << GetName() << ") ERROR: no object found with name " << name << endl ;
    } else {
      coutE(InputArguments) << "RooPlot::remove(" << GetName() << ") ERROR: plot frame is empty, cannot remove last object" << endl ;
    }
    return ;
  }

  _items.Remove(obj) ;

  if (deleteToo) {
    delete obj ;
  }
}


//_____________________________________________________________________________
Bool_t RooPlot::drawBefore(const char *before, const char *target)
{
  // Change the order in which our contained objects are drawn so that
  // the target object is drawn just before the specified object.
  // Returns kFALSE if either object does not exist.

  return _items.moveBefore(before, target, caller("drawBefore"));
}


//_____________________________________________________________________________
Bool_t RooPlot::drawAfter(const char *after, const char *target)
{
  // Change the order in which our contained objects are drawn so that
  // the target object is drawn just after the specified object.
  // Returns kFALSE if either object does not exist.

  return _items.moveAfter(after, target, caller("drawAfter"));
}


//_____________________________________________________________________________
TObject *RooPlot::findObject(const char *name, const TClass* clas) const
{
  // Find the named object in our list of items and return a pointer
  // to it. Return zero and print a warning message if the named
  // object cannot be found. If no name is supplied the last object
  // added is returned.
  //
  // Note that the returned pointer is to a
  // TObject and so will generally need casting. Use the getAtt...()
  // methods to change the drawing style attributes of a contained
  // object directly.

  TObject *obj = 0;
  TObject *ret = 0;

  TIterator* iter = _items.MakeIterator() ;
  while((obj=iter->Next())) {
    if ((!name || !TString(name).CompareTo(obj->GetName())) &&
	(!clas || (obj->IsA()==clas))) {
      ret = obj ;
    }
  }
  delete iter ;

  if (ret==0) {
    coutE(InputArguments) << "RooPlot::findObject(" << GetName() << ") cannot find object " << (name?name:"<last>") << endl ;
  }
  return ret ;
}


//_____________________________________________________________________________
TString RooPlot::getDrawOptions(const char *name) const
{
  // Return the Draw() options registered for the named object. Return
  // an empty string if the named object cannot be found.

  TObjOptLink *link= _items.findLink(name,caller("getDrawOptions"));
  DrawOpt opt(0 == link ? "" : link->GetOption()) ;
  return TString(opt.drawOptions) ;
}


//_____________________________________________________________________________
Bool_t RooPlot::setDrawOptions(const char *name, TString options)
{
  // Register the specified drawing options for the named object.
  // Return kFALSE if the named object cannot be found.

  TObjOptLink *link= _items.findLink(name,caller("setDrawOptions"));
  if(0 == link) return kFALSE;

  DrawOpt opt(link->GetOption()) ;
  strlcpy(opt.drawOptions,options,128) ;
  link->SetOption(opt.rawOpt());
  return kTRUE;
}


//_____________________________________________________________________________
Bool_t RooPlot::getInvisible(const char* name) const
{
  // Returns true of object with given name is set to be invisible
  TObjOptLink *link= _items.findLink(name,caller("getInvisible"));
  if(0 == link) return kFALSE;

  return DrawOpt(link->GetOption()).invisible ;
}


//_____________________________________________________________________________
void RooPlot::setInvisible(const char* name, Bool_t flag)
{
  // If flag is true object with 'name' is set to be invisible
  // i.e. it is not drawn when Draw() is called

  TObjOptLink *link= _items.findLink(name,caller("getInvisible"));

  DrawOpt opt ;

  if(link) {
    opt.initialize(link->GetOption()) ;
    opt.invisible = flag ;
    link->SetOption(opt.rawOpt()) ;
  }

}



//_____________________________________________________________________________
TString RooPlot::caller(const char *method) const
{
  // Utility function
  TString name(fName);
  if(strlen(method)) {
    name.Append("::");
    name.Append(method);
  }
  return name;
}



//_____________________________________________________________________________
void RooPlot::SetMaximum(Double_t maximum)
{
  // Set maximum value of Y axis
  _hist->SetMaximum(maximum==-1111?_defYmax:maximum) ;
}



//_____________________________________________________________________________
void RooPlot::SetMinimum(Double_t minimum)
{
  // Set minimum value of Y axis
  _hist->SetMinimum(minimum==-1111?_defYmin:minimum) ;
}



//_____________________________________________________________________________
Double_t RooPlot::chiSquare(const char* curvename, const char* histname, Int_t nFitParam) const
{
  // Calculate and return reduced chi-squared of curve with given name with respect
  // to histogram with given name. If nFitParam is non-zero, it is used to reduce the
  // number of degrees of freedom for a chi^2 for a curve that was fitted to the
  // data with that number of floating parameters


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


//_____________________________________________________________________________
RooHist* RooPlot::residHist(const char* histname, const char* curvename, bool normalize, bool useAverage) const
{
  // Return a RooHist containing the residuals of histogram 'histname' with respect
  // to curve 'curvename'. If normalize is true the residuals are divided by the error
  // on the histogram, effectively returning a pull histogram

  // Find curve object
  RooCurve* curve = (RooCurve*) findObject(curvename,RooCurve::Class()) ;
  if (!curve) {
    coutE(InputArguments) << "RooPlot::residHist(" << GetName() << ") cannot find curve" << endl ;
    return 0 ;
  }

  // Find histogram object
  RooHist* hist = (RooHist*) findObject(histname,RooHist::Class()) ;
  if (!hist) {
    coutE(InputArguments) << "RooPlot::residHist(" << GetName() << ") cannot find histogram" << endl ;
    return 0 ;
  }

  return hist->makeResidHist(*curve,normalize,useAverage) ;
}



//_____________________________________________________________________________
void RooPlot::DrawOpt::initialize(const char* inRawOpt)
{
  // Initialize the DrawOpt helper class

  if (!inRawOpt) {
    drawOptions[0] = 0 ;
    invisible=kFALSE ;
    return ;
  }
  strlcpy(drawOptions,inRawOpt,128) ;
  strtok(drawOptions,":") ;
  const char* extraOpt = strtok(0,":") ;
  if (extraOpt) {
    invisible =  (extraOpt[0]=='I') ;
  }
}


//_____________________________________________________________________________
const char* RooPlot::DrawOpt::rawOpt() const
{
  // Return the raw draw options
  static char buf[128] ;
  strlcpy(buf,drawOptions,128) ;
  if (invisible) {
    strlcat(buf,":I",128) ;
  }
  return buf ;
}



//_____________________________________________________________________________
Double_t RooPlot::getFitRangeNEvt(Double_t xlo, Double_t xhi) const
{
  // Return the number of events that is associated with the range [xlo,xhi]
  // This method is only fully functional for ranges not equal to the full
  // range if the object that inserted the normalization data provided
  // a link to an external object that can calculate the event count in
  // in sub ranges. An error will be printed if this function is used
  // on sub-ranges while that information is not available

  Double_t scaleFactor = 1.0 ;
  if (_normObj) {
    scaleFactor = _normObj->getFitRangeNEvt(xlo,xhi)/_normObj->getFitRangeNEvt() ;
  } else {
    coutW(Plotting) << "RooPlot::getFitRangeNEvt(" << GetName() << ") WARNING: Unable to obtain event count in range "
		    << xlo << " to " << xhi << ", substituting full event count" << endl ;
  }
  return getFitRangeNEvt()*scaleFactor ;
}


//_____________________________________________________________________________
void RooPlot::SetName(const char *name)
{
  // Set the name of the RooPlot to 'name'

  if (_dir) _dir->GetList()->Remove(this);
  TNamed::SetName(name) ;
  if (_dir) _dir->GetList()->Add(this);
}


//_____________________________________________________________________________
void RooPlot::SetNameTitle(const char *name, const char* title)
{
  // Set the name and title of the RooPlot to 'name' and 'title'
  if (_dir) _dir->GetList()->Remove(this);
  TNamed::SetNameTitle(name,title) ;
  if (_dir) _dir->GetList()->Add(this);
}


//_____________________________________________________________________________
void RooPlot::SetTitle(const char* title)
{
  // Set the title of the RooPlot to 'title'
  TNamed::SetTitle(title) ;
  _hist->SetTitle(title) ;
}



//_____________________________________________________________________________
Int_t RooPlot::defaultPrintContents(Option_t* /*opt*/) const
{
  // Define default print options, for a given print style

  return kName|kArgs|kValue ;
}



TAxis* RooPlot::GetXaxis() const { return _hist->GetXaxis() ; }
TAxis* RooPlot::GetYaxis() const { return _hist->GetYaxis() ; }
Int_t  RooPlot::GetNbinsX() const { return _hist->GetNbinsX() ; }
Int_t  RooPlot::GetNdivisions(Option_t* axis) const { return _hist->GetNdivisions(axis) ; }
Double_t  RooPlot::GetMinimum(Double_t minval) const { return _hist->GetMinimum(minval) ; }
Double_t   RooPlot::GetMaximum(Double_t maxval) const { return _hist->GetMaximum(maxval) ; }


void RooPlot::SetAxisColor(Color_t color, Option_t* axis) { _hist->SetAxisColor(color,axis) ; }
void RooPlot::SetAxisRange(Double_t xmin, Double_t xmax, Option_t* axis) { _hist->SetAxisRange(xmin,xmax,axis) ; }
void RooPlot::SetBarOffset(Float_t offset) { _hist->SetBarOffset(offset) ; }
void RooPlot::SetBarWidth(Float_t width) { _hist->SetBarWidth(width) ; }
void RooPlot::SetContour(Int_t nlevels, const Double_t* levels) { _hist->SetContour(nlevels,levels) ; }
void RooPlot::SetContourLevel(Int_t level, Double_t value) { _hist->SetContourLevel(level,value) ; }
void RooPlot::SetDrawOption(Option_t* option) { _hist->SetDrawOption(option) ; }
void RooPlot::SetFillAttributes() { _hist->SetFillAttributes() ; }
void RooPlot::SetFillColor(Color_t fcolor) { _hist->SetFillColor(fcolor) ; }
void RooPlot::SetFillStyle(Style_t fstyle) { _hist->SetFillStyle(fstyle) ; }
void RooPlot::SetLabelColor(Color_t color, Option_t* axis) { _hist->SetLabelColor(color,axis) ; }
void RooPlot::SetLabelFont(Style_t font, Option_t* axis) { _hist->SetLabelFont(font,axis) ; }
void RooPlot::SetLabelOffset(Float_t offset, Option_t* axis) { _hist->SetLabelOffset(offset,axis) ; }
void RooPlot::SetLabelSize(Float_t size, Option_t* axis) { _hist->SetLabelSize(size,axis) ; }
void RooPlot::SetLineAttributes() { _hist->SetLineAttributes() ; }
void RooPlot::SetLineColor(Color_t lcolor) { _hist->SetLineColor(lcolor) ; }
void RooPlot::SetLineStyle(Style_t lstyle) { _hist->SetLineStyle(lstyle) ; }
void RooPlot::SetLineWidth(Width_t lwidth) { _hist->SetLineWidth(lwidth) ; }
void RooPlot::SetMarkerAttributes() { _hist->SetMarkerAttributes() ; }
void RooPlot::SetMarkerColor(Color_t tcolor) { _hist->SetMarkerColor(tcolor) ; }
void RooPlot::SetMarkerSize(Size_t msize) { _hist->SetMarkerSize(msize) ; }
void RooPlot::SetMarkerStyle(Style_t mstyle) { _hist->SetMarkerStyle(mstyle) ; }
void RooPlot::SetNdivisions(Int_t n, Option_t* axis) { _hist->SetNdivisions(n,axis) ; }
void RooPlot::SetOption(Option_t* option) { _hist->SetOption(option) ; }
void RooPlot::SetStats(Bool_t stats) { _hist->SetStats(stats) ; }
void RooPlot::SetTickLength(Float_t length, Option_t* axis) { _hist->SetTickLength(length,axis) ; }
void RooPlot::SetTitleFont(Style_t font, Option_t* axis) { _hist->SetTitleFont(font,axis) ; }
void RooPlot::SetTitleOffset(Float_t offset, Option_t* axis) { _hist->SetTitleOffset(offset,axis) ; }
void RooPlot::SetTitleSize(Float_t size, Option_t* axis) { _hist->SetTitleSize(size,axis) ; }
void RooPlot::SetXTitle(const char *title) { _hist->SetXTitle(title) ; }
void RooPlot::SetYTitle(const char *title) { _hist->SetYTitle(title) ; }
void RooPlot::SetZTitle(const char *title) { _hist->SetZTitle(title) ; }




//______________________________________________________________________________
void RooPlot::Browse(TBrowser * /*b*/)
{
  // Plot RooPlot when double-clicked in browser
  Draw();
  gPad->Update();
}




//_____________________________________________________________________________
void RooPlot::Streamer(TBuffer &R__b)
{

  // Custom streamer, needed for backward compatibility

  if (R__b.IsReading()) {

    TH1::AddDirectory(kFALSE) ;

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
      _items.Streamer(R__b);
      R__b >> _padFactor;
      R__b >> _plotVarClone;
      R__b >> _plotVarSet;
      R__b >> _normVars;
      R__b >> _normNumEvts;
      R__b >> _normBinWidth;
      R__b >> _defYmin;
      R__b >> _defYmax;
      R__b.CheckByteCount(R__s, R__c, RooPlot::IsA());
    }

    TH1::AddDirectory(kTRUE) ;


  } else {
    R__b.WriteClassBuffer(RooPlot::Class(),this);
  }
}
