/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.cc,v 1.1 2001/03/28 19:21:48 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION -- A RooPlot is a plot frame and a container
// for graphics objects within that frame. As a frame, it provides the
// TH1 public interface for settting plot ranges, configuring axes,
// etc. As a container, it holds an arbitrary set of objects that
// might be histograms of data, curves representing a fit model, or
// text labels. Use the Draw() method to draw a frame and the objects
// it contains. Use the various add...() methods to add objects to be
// drawn.  In general, the add...() methods create a private copy of
// the object you pass them and return a pointer to this copy. The
// caller owns the input object and this class owns the returned
// object.

#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooHist.hh"

#include <iostream.h>
#include <string.h>
#include <assert.h>

ClassImp(RooPlot)

static const char rcsid[] =
"$Id: RooPlot.cc,v 1.1 2001/03/28 19:21:48 davidk Exp $";

RooPlot::RooPlot(Float_t xmin, Float_t xmax) :
  TH1("frame","RooPlotFrame",0,xmin,xmax), _items()
{
  // Create an empty frame with the specified x-axis limits.

  initialize();
}

RooPlot::RooPlot(Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax) :
  TH1("frame","RooPlotFrame",0,xmin,xmax), _items()
{
  // Create an empty frame with the specified x- and y-axis limits.

  SetMinimum(ymin);
  SetMaximum(ymax);
  initialize();
}

RooPlot::RooPlot(RooAbsReal &var) :
  TH1("frame",var.GetTitle(),0,var.getPlotMin(),var.getPlotMax()), _items()
{
  // Create an empty frame with its title and x-axis range and label taken
  // from the specified real variable.

  if(0 != strlen(var.getUnit())) {
    fTitle.Append(" (");
    fTitle.Append(var.getUnit());
    fTitle.Append(")");
  }
  SetXTitle((Text_t*)fTitle.Data());
  initialize();
}

void RooPlot::initialize() {
  // Perform initialization that is common to all constructors.

  _iterator= _items.MakeIterator();
  assert(0 != _iterator);
}

RooPlot::~RooPlot() {
  // Delete the items in our container and our iterator.

  _items.Delete();
  delete _iterator;
}

TObject *RooPlot::addObject(const TObject *obj, const char *drawOptions) {
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
  if(0 != clone) _items.Add(clone,drawOptions);
  return clone;
}

RooHist *RooPlot::addHistogram(const TH1 *data, const char *drawOptions, Bool_t adjustRange) {
  // Convert a one-dimensional TH1 into a RooHist object and add the
  // new RooHist to this plot. Use the specified options provided to
  // Draw() the RooHist later.  The upper limit of our frame's y-axis
  // will be increased if necessary to fit this histogram, unless
  // adjustRange is kFALSE. The input histogram belongs to the caller,
  // but the returned object belongs to this container object (ie, it
  // will be deleted in our destructor). Returns zero in case of
  // error.

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
  RooHist *graph= new RooHist();
  if(0 != graph) return 0;

  // fill our histogram with the contents of each bin of the input histogram
  Int_t nbin= data->GetNbinsX();
  for(Int_t bin= 1; bin <= nbin; bin++) {
    Double_t x= data->GetBinCenter(bin);
    Double_t y= data->GetBinContent(bin);
    // check for a positive integer bin contents
    Int_t n= (Int_t)(y+0.5);
    if(fabs(y-n)>1e-6) {
      cout << "RooPlot::AddHistogram: cannot calculate error for non-integer "
	   << "bin " << bin << " contents: " << y << endl;
      return 0;
    }
    graph->addBin(x,n);
  }

  // adjust our frame's range if necessary (and requested)
  if(adjustRange) {
    Float_t ymax= 1.05*graph->getPlotMax();
    if(GetMaximum() < ymax) SetMaximum(ymax);
  }

  // set the key used to retrieve this object from our list
  graph->SetName(data->GetName());
  graph->SetTitle(data->GetTitle());

  // add this element to our list and remember its drawing option
  _items.Add(graph,drawOptions);
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

void RooPlot::printToStream(ostream& os, PrintOption opt, const char *indent) const {
  oneLinePrint(os,*this);
}
