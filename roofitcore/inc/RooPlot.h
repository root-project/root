/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.rdl,v 1.4 2001/03/13 01:15:43 david Exp $
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

class RooAbsReal;
class RooPlotWithErrors;

class RooPlot : public TH1F {
public:
  RooPlot(RooAbsReal &var);
  RooPlot(Float_t xmin= 0, Float_t xmax= 1);
  RooPlot(Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax);
  virtual ~RooPlot();
  TObject *addObject(const TObject *obj, const char *options= "");
  RooPlotWithErrors *addHistogram(const TH1F *data, const char *options= "P");
  virtual void Draw(Option_t *options= 0);
protected:
  void initialize();
  THashList _items;
  TIterator *_iterator;  //! non-persistent
  RooPlot(const RooPlot& other); // object cannot be copied
  ClassDef(RooPlot,1) // a value-added plot (non-persistant)
};

#endif
