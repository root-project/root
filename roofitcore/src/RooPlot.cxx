/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.cc,v 1.9 2001/05/09 00:51:10 david Exp $
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

#include "BaBar/BaBar.hh"

#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooPlotable.hh"
#include "RooFitCore/RooHist.hh"
#include "RooFitCore/RooArgSet.hh"

#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"
#include "TAttText.h"

#include <iostream.h>
#include <string.h>
#include <assert.h>

ClassImp(RooPlot)

static const char rcsid[] =
"$Id: RooPlot.cc,v 1.9 2001/05/09 00:51:10 david Exp $";

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
  _plotVarSet = RooArgSet(var).snapshot() ;
  _plotVarClone= (RooAbsReal*)_plotVarSet->find(var.GetName()) ;
  _plotVarSet->Print("v") ;
  
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

  fDimension=1 ;
  setPadFactor(0.05);
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
}

Stat_t RooPlot::GetBinContent(Int_t i) const {
  // A plot object is a frame without any bin contents of its own so this
  // method always returns zero.
  return 0;
}

TObject *RooPlot::addObject(const TObject *obj, Option_t *drawOptions) {
  // Add a generic object to this plot. The specified options will be
  // used to Draw() this object later. Returns a pointer to a clone of
  // the input object which belongs to this container object (ie, it
  // will be deleted in our destructor). The caller still owns the
  // input object. Returns zero in case of error.

  if(0 == obj) {
    cout << fName << "::addObject: called with a null pointer" << endl;
    return 0;
  }
  TObject *clone= obj->Clone();
  // add option "SAME" for a TH1 object if necessary
  TString options(drawOptions);
  options.ToUpper();
  if(clone->IsA()->InheritsFrom(TH1::Class())) {
    if(!options.Contains("SAME")) options.Append("SAME");
  }
  if(0 != clone) _items.Add(clone,options.Data());

  return clone;
}

void RooPlot::addPlotable(RooPlotable *plotable, Option_t *drawOptions) {
  // Add the specified plotable object to our plot. Increase our y-axis
  // limits to fit this object if necessary. The default lower-limit
  // is zero unless we are plotting an object that takes on negative values.
  // This call transfers ownership of the plotable object to this class.
  // The plotable object will be deleted when this plot object is deleted.

  // get this object's y-axis limits w/o padding
  Double_t ymin= plotable->getYAxisMin();
  if(ymin > 0) ymin= 0;
  Double_t ymax= plotable->getYAxisMax();

  // calculate the padded values
  Double_t ypad= getPadFactor()*(ymax-ymin);
  ymax+= ypad;
  if(ymin < 0) ymin-= ypad;

  // update our limits if necessary
  if(GetMaximum() < ymax) SetMaximum(ymax);
  if(GetMinimum() > ymin) SetMinimum(ymin);

  // use this object's y-axis title if we don't have one already
  if(0 == strlen(GetYaxis()->GetTitle())) SetYTitle(plotable->getYAxisLabel());

  // add this element to our list and remember its drawing option
  TObject *obj= plotable->crossCast();
  if(0 == obj) {
    cout << fName << "::add: cross-cast to TObject failed (nothing added)" << endl;
  }
  else {
    _items.Add(obj,drawOptions);
  }
}

RooHist *RooPlot::addHistogram(const TH1 *data, Option_t *drawOptions) {
  // Convert a one-dimensional TH1 into a RooHist object and add the
  // new RooHist to this plot. Use the specified options provided to
  // Draw() the RooHist later.  The upper limit of our frame's y-axis
  // will be increased if necessary to fit this histogram. You can
  // adjust the axis range by hand with the SetMaximum() method if
  // necessary. The input histogram belongs to the caller, but the
  // returned object belongs to this container object (ie, it will be
  // deleted in our destructor). Returns zero in case of error.
  //
  // Use the addObject() method to add a TH1 object that should *not* be
  // converted into a RooHist (eg, if Poisson errors are not appropriate).
  // In this case, you are responsible for increasing the plot range, using
  // SetMaximum(), if necessary.

  // check for a valid input object
  if(0 == data) {
    cout << fName << "::addHistogram: called with a null pointer" << endl;
    return 0;
  }
  if(data->GetDimension() != 1) {
    cout << fName << "::addHistogram: cannot add TH1 with " << data->GetDimension()
	 << " dimensions" << endl;
    return 0;
  }

  // create a new histogram on the heap
  RooHist *graph= new RooHist(*data);
  if(0 == graph) return 0;

  addPlotable(graph,drawOptions);

  return graph;
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
