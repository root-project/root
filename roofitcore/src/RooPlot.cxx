/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.cc,v 1.18 2001/09/17 18:48:15 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// A RooPlot is a plot frame and a container for graphics objects
// within that frame. As a frame, it provides the TH1 public interface
// for settting plot ranges, configuring axes, etc. As a container, it
// holds an arbitrary set of objects that might be histograms of data,
// curves representing a fit model, or text labels. Use the Draw()
// method to draw a frame and the objects it contains. Use the various
// add...() methods to add objects to be drawn.  In general, the
// add...() methods create a private copy of the object you pass them
// and return a pointer to this copy. The caller owns the input object
// and this class owns the returned object.

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooPlotable.hh"
#include "RooFitCore/RooArgSet.hh"

#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"
#include "TAttText.h"

#include <iostream.h>
#include <string.h>
#include <assert.h>

ClassImp(RooPlot)
  ;

static const char rcsid[] =
"$Id: RooPlot.cc,v 1.18 2001/09/17 18:48:15 verkerke Exp $";

RooPlot::RooPlot(Float_t xmin, Float_t xmax) :
  TH1(histName(),"A RooPlot",0,xmin,xmax), _plotVarClone(0), 
  _plotVarSet(0), _items()
{
  // Create an empty frame with the specified x-axis limits.
  initialize();
}

RooPlot::RooPlot(Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax) :
  TH1(histName(),"A RooPlot",0,xmin,xmax), _plotVarClone(0), 
  _plotVarSet(0), _items()
{
  // Create an empty frame with the specified x- and y-axis limits.
  SetMinimum(ymin);
  SetMaximum(ymax);
  initialize();
}

RooPlot::RooPlot(const RooAbsReal &var) :
  TH1(histName(),"RooPlot",0,var.getPlotMin(),var.getPlotMax()),
  _plotVarClone(0), _plotVarSet(0), _items()
{
  // Create an empty frame with its title and x-axis range and label taken
  // from the specified real variable. We keep a clone of the variable
  // so that we do not depend on its lifetime and are decoupled from
  // any later changes to its state.

  // plotVar can be a composite in case of a RooDataSet::plot, need deepClone
  _plotVarSet = (RooArgSet*) RooArgSet(var).snapshot() ;
  _plotVarClone= (RooAbsReal*)_plotVarSet->find(var.GetName()) ;
  
  TString xtitle(var.GetTitle());
  if(0 != strlen(var.getUnit())) {
    xtitle.Append(" (");
    xtitle.Append(var.getUnit());
    xtitle.Append(")");
  }
  SetXTitle(xtitle.Data());

  TString title("A RooPlot of \"");
  title.Append(var.GetTitle());
  title.Append("\"");
  SetTitle(title.Data());
  initialize();
}

void RooPlot::initialize() {
  // Perform initialization that is common to all constructors.

  // We hold 1D plot objects
  fDimension=1 ;
  // We do not have useful stats of our own
  SetStats(kFALSE);
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


TString RooPlot::histName() const 
{
  return TString(Form("frame(%08x)",this)) ;
}

RooPlot::~RooPlot() {
  // Delete the items in our container and our iterator.

  _items.Delete();
  delete _iterator;
  if(_plotVarSet) delete _plotVarSet;
  if(_normVars) delete _normVars;
}

void RooPlot::updateNormVars(const RooArgSet &vars) {
  if(0 == _normVars) _normVars= (RooArgSet*) vars.snapshot(kTRUE);
}

Stat_t RooPlot::GetBinContent(Int_t i) const {
  // A plot object is a frame without any bin contents of its own so this
  // method always returns zero.
  return 0;
}

Stat_t RooPlot::GetBinContent(Int_t, Int_t) const
{
  // A plot object is a frame without any bin contents of its own so this
  // method always returns zero.
  return 0;
}

Stat_t RooPlot::GetBinContent(Int_t, Int_t, Int_t) const
{
  // A plot object is a frame without any bin contents of its own so this
  // method always returns zero.
  return 0;
}


void RooPlot::addObject(TObject *obj, Option_t *drawOptions) {
  // Add a generic object to this plot. The specified options will be
  // used to Draw() this object later. The caller transfers ownership
  // of the object with this call, and the object will be deleted
  // when its containing plot object is destroyed.

  if(0 == obj) {
    cout << fName << "::addObject: called with a null pointer" << endl;
    return;
  }
  _items.Add(obj,drawOptions);
}

void RooPlot::addTH1(TH1 *hist, Option_t *drawOptions) {
  // Add a TH1 histogram object to this plot. The specified options
  // will be used to Draw() this object later. "SAME" will be added to
  // the options if they are not already present. Note that histograms
  // should probably not be drawn with error bars since they will not
  // be calculated correctly for bins with low statistics, and will
  // not be accounted for in the automatic y-axis range adjustment. To
  // histogram data in a RooDataSet without these problems, use
  // RooDataSet::plotOn(). The caller transfers ownership of the
  // object with this call, and the object will be deleted when its
  // containing plot object is destroyed.

  if(0 == hist) {
    cout << fName << "::addTH1: called with a null pointer" << endl;
    return;
  }
  // check that this histogram is really 1D
  if(1 != hist->GetDimension()) {
    cout << fName << "::addTH1: cannot plot histogram with "
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
  addObject(hist,options.Data());
}

void RooPlot::addPlotable(RooPlotable *plotable, Option_t *drawOptions) {
  // Add the specified plotable object to our plot. Increase our y-axis
  // limits to fit this object if necessary. The default lower-limit
  // is zero unless we are plotting an object that takes on negative values.
  // This call transfers ownership of the plotable object to this class.
  // The plotable object will be deleted when this plot object is deleted.

  // update our y-axis label and limits
  updateYAxis(plotable->getYAxisMin(),plotable->getYAxisMax(),plotable->getYAxisLabel());

  // use this object's normalization if necessary
  updateFitRangeNorm(plotable) ;

  // add this element to our list and remember its drawing option
  TObject *obj= plotable->crossCast();
  if(0 == obj) {
    cout << fName << "::add: cross-cast to TObject failed (nothing added)" << endl;
  }
  else {
    _items.Add(obj,drawOptions);
  }
}

void RooPlot::updateFitRangeNorm(const TH1* hist) {
  // Update our plot normalization over our plot variable's fit range,
  // which will be determined by the first suitable object added to our plot.

  if(_normNumEvts == 0) {
    const TAxis* xa = ((TH1*)hist)->GetXaxis() ;
    _normBinWidth = (xa->GetXmax()-xa->GetXmin())/hist->GetNbinsX() ;
    _normNumEvts = hist->GetEntries()/_normBinWidth ;
  }      
}

void RooPlot::updateFitRangeNorm(const RooPlotable* rp) {
  // Update our plot normalization over our plot variable's fit range,
  // which will be determined by the first suitable object added to our plot.

  if(_normNumEvts == 0) {
    _normNumEvts = rp->getFitRangeNEvt() ;
    _normBinWidth = rp->getFitRangeBinW() ;
  }      
}

void RooPlot::updateYAxis(Double_t ymin, Double_t ymax, const char *label) {
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
  if(GetMaximum() < ymax) SetMaximum(ymax);
  if(GetMinimum() > ymin) SetMinimum(ymin);

  // use the specified y-axis label if we don't have one already
  if(0 == strlen(GetYaxis()->GetTitle())) SetYTitle(label);
}

void RooPlot::Draw(Option_t *options) {
  // Draw this plot and all of the elements it contains. The specified options
  // only apply to the drawing of our frame. The options specified in our add...()
  // methods will be used to draw each object we contain.

  TH1::Draw(options);
  _iterator->Reset();
  TObject *obj(0);
  while(obj= _iterator->Next()) {
    obj->Draw(_iterator->GetOption());
  }
}

void RooPlot::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this plot object to the specified stream.
  //
  //  Standard: plot variable and number of contained objects
  //     Shape: list of our contained objects

  oneLinePrint(os,*this);
  if(opt >= Standard) {
    TString deeper(indent);
    deeper.Append("    ");
    if(0 != _plotVarClone) {
      os << indent << "  Plotting ";
      _plotVarClone->printToStream(os,OneLine,deeper);
    }
    else {
      os << indent << "  No plot variable specified" << endl;
    }
    os << indent << "  Plot contains " << _items.GetSize() << " object(s)" << endl;
    if(opt >= Shape) {
      _iterator->Reset();
      TObject *obj(0);
      while(obj= _iterator->Next()) {
	os << deeper << "(Options=\"" << _iterator->GetOption() << "\") ";
	// Is this a printable object?
	if(obj->IsA()->InheritsFrom(RooPrintable::Class())) {
	  ostream& oldDefault= RooPrintable::defaultStream(&os);
	  obj->Print("1");
	  RooPrintable::defaultStream(&oldDefault);
	}
	// is it a TNamed subclass?
	else if(obj->IsA()->InheritsFrom(TNamed::Class())) {
	  oneLinePrint(os,(const TNamed&)(*obj));
	}
	// at least it is a TObject
	else {
	  os << obj->ClassName() << "::" << obj->GetName() << endl;
	}
      }
    }
  }
}

TAttLine *RooPlot::getAttLine(const char *name) const {
  // Return a pointer to the line attributes of the named object in this plot,
  // or zero if the named object does not exist or does not have line attributes.

  return dynamic_cast<TAttLine*>(findObject(name));
}

TAttFill *RooPlot::getAttFill(const char *name) const {
  // Return a pointer to the fill attributes of the named object in this plot,
  // or zero if the named object does not exist or does not have fill attributes.

  return dynamic_cast<TAttFill*>(findObject(name));
}

TAttMarker *RooPlot::getAttMarker(const char *name) const {
  // Return a pointer to the marker attributes of the named object in this plot,
  // or zero if the named object does not exist or does not have marker attributes.

  return dynamic_cast<TAttMarker*>(findObject(name));
}

TAttText *RooPlot::getAttText(const char *name) const {
  // Return a pointer to the text attributes of the named object in this plot,
  // or zero if the named object does not exist or does not have text attributes.

  return dynamic_cast<TAttText*>(findObject(name));
}

Bool_t RooPlot::drawBefore(const char *before, const char *target) {
  // Change the order in which our contained objects are drawn so that
  // the target object is drawn just before the specified object.
  // Returns kFALSE if either object does not exist.

  return _items.moveBefore(before, target, caller("drawBefore"));
}

Bool_t RooPlot::drawAfter(const char *after, const char *target) {
  // Change the order in which our contained objects are drawn so that
  // the target object is drawn just after the specified object.
  // Returns kFALSE if either object does not exist.

  return _items.moveAfter(after, target, caller("drawAfter"));
}

TObject *RooPlot::findObject(const char *name) const {
  // Find the named object in our list of items and return a pointer
  // to it. Return zero and print a warning message if the named
  // object cannot be found. Note that the returned pointer is to a
  // TObject and so will generally need casting. Use the getAtt...()
  // methods to change the drawing style attributes of a contained
  // object directly.

  TObject *obj= _items.FindObject(name);
  if(0 == obj) {
    cout << fName << "::findObject: cannot find object named \"" << name << "\"" << endl;
  }
  return obj;
}

TString RooPlot::getDrawOptions(const char *name) const {
  // Return the Draw() options registered for the named object. Return
  // an empty string if the named object cannot be found.

  TObjOptLink *link= _items.findLink(name,caller("getDrawOptions"));
  return TString(0 == link ? "" : link->GetOption());
}

Bool_t RooPlot::setDrawOptions(const char *name, TString options) {
  // Register the specified drawing options for the named object.
  // Return kFALSE if the named object cannot be found.

  TObjOptLink *link= _items.findLink(name,caller("setDrawOptions"));
  if(0 == link) return kFALSE;
  link->SetOption(options.Data());
  return kTRUE;
}

TString RooPlot::caller(const char *method) const {
  TString name(fName);
  if(strlen(method)) {
    name.Append("::");
    name.Append(method);
  }
  return name;
}
