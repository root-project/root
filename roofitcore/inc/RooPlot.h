/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.rdl,v 1.3 2001/04/11 23:25:27 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#ifndef ROO_PLOT
#define ROO_PLOT

#include "TH1.h"
#include "THashList.h"
#include "RooFitCore/RooPrintable.hh"

class RooAbsReal;
class RooHist;

class RooPlot : public TH1, public RooPrintable {
public:
  RooPlot(RooAbsReal &var);
  RooPlot(Float_t xmin= 0, Float_t xmax= 1);
  RooPlot(Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax);
  virtual ~RooPlot();
  TObject *addObject(const TObject *obj, const char *drawOptions= "");
  RooHist *addHistogram(const TH1 *data, const char *drawOptions= "E",
			Bool_t adjustRange= kTRUE);
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  virtual void Draw(Option_t *options= 0);
  inline RooAbsReal *getPlotVar() const { return _plotVar; }
protected:
  void initialize();
  THashList _items;
  RooAbsReal *_plotVar;
  TIterator *_iterator;  //! non-persistent
  RooPlot(const RooPlot& other); // object cannot be copied
  ClassDef(RooPlot,1) // A plot frame and container for graphics objects
};

#endif
