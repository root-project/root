/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.cc,v 1.6 2001/03/15 22:43:37 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// A RooPlot is a plot frame and a container for graphics objects within that
// frame. As a frame, it provides the TH1F public interface for settting plot
// ranges, configuring axes, etc. As a container, it holds an arbitrary set
// of objects that might be histograms of data, curves representing a fit
// model, or text labels. Use the Draw() method to draw a frame and the objects
// it contains. Use the various add...() methods to add objects to be drawn.

#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooPlotWithErrors.hh"

#include <iostream.h>
#include <string.h>
#include <math.h>

#include "TH1.h"

ClassImp(RooPlot)

static const char rcsid[] =
"$Id: RooPlot.cc,v 1.6 2001/03/15 22:43:37 davidk Exp $";

RooPlot::RooPlot(RooAbsReal &var) :
  TH1F("frame",var.GetTitle(),0,var.getPlotMin(),var.getPlotMax()), _items()
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

RooPlot::RooPlot(Float_t xmin, Float_t xmax) :
  TH1F("frame","RooPlotFrame",0,xmin,xmax), _items()
{
  // Create an empty frame with the specified x-axis limits.

  initialize();
}

RooPlot::RooPlot(Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax) :
  TH1F("frame","RooPlotFrame",0,xmin,xmax), _items()
{
  // Create an empty frame with the specified x- and y-axis limits.

  SetMinimum(ymin);
  SetMaximum(ymax);
  initialize();
}

void RooPlot::initialize() {
  // Perform initialization that is common to all constructors.

  _iterator= _items.MakeIterator();
}

RooPlot::~RooPlot() {
  // Delete the items in our container.

  _items.Delete();
  delete _iterator;
}

TObject *RooPlot::addObject(const TObject *obj, const char *options) {
  // Add a generic object to this plot. The specified options will be used
  // to Draw() this object later. Returns a pointer to a clone of the input
  // object which belongs to this container object (ie, it will be deleted
  // in our destructor). The caller still owns the input object.

  _items.Add(obj->Clone(),options);
  return obj;
}

RooPlotWithErrors *RooPlot::addHistogram(const TH1F *data, const char *options) {
  // Add a histogram of event counts that will be displayed using
  // Poisson error bars. Use the specified options to Draw() this object
  // later. The input histogram belongs to the caller, but the returned
  // plot object belongs to this container object (ie, it will be deleted
  // in our destructor).

  // create a new plot element on the heap
  RooPlotWithErrors *graph= new RooPlotWithErrors();
  // fill this plot with the contents of each bin
  Int_t nbin= data->GetNbinsX();
  for(Int_t bin= 1; bin <= nbin; bin++) {
    Double_t x= data->GetBinCenter(bin);
    Double_t y= data->GetBinContent(bin);
    // check for a positive integer bin contents
    Int_t n= (Int_t)(y+0.5);
    if(fabs(y-n)>1e-6) {
      cout << "RooPlot::AddHistogram: cannot calculate error for non-integer "
	   << "bin contents " << y << endl;
      return 0;
    }
    graph->addBin(x,n);
  }
  // adjust our frame's range if necessary
  Float_t ymax= 1.05*graph->getPlotMax();
  if(GetMaximum() < ymax) SetMaximum(ymax);
  // set the key used to retrieve this object from our list
  graph->SetName(data->GetName());
  graph->SetTitle(data->GetTitle());
  // add this element to our list and remember its drawing option
  _items.Add(graph,options);
  return graph;
}

void RooPlot::Draw(Option_t *options) {
  // Draw this plot and all of the elements it contains.

  TH1F::Draw(options);
  _iterator->Reset();
  TObject *obj(0);
  while(obj= _iterator->Next()) {
    obj->Draw(_iterator->GetOption());
  }
}
