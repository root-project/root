/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPlotable.rdl,v 1.13 2005/06/20 15:44:56 wverkerke Exp $
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
#ifndef ROO_PLOTABLE
#define ROO_PLOTABLE

#include "Rtypes.h"
#include "TString.h"
#include "RooPrintable.h"

class TObject;
class RooArgSet;

class RooPlotable : public RooPrintable {
public:
  inline RooPlotable() : _ymin(0), _ymax(0), _normValue(0) { }
  inline virtual ~RooPlotable() { }

  inline const char* getYAxisLabel() const { return _yAxisLabel.Data(); }
  inline void setYAxisLabel(const char *label) { _yAxisLabel= label; }
  inline void updateYAxisLimits(Double_t y) {
    if(y > _ymax) _ymax= y;
    if(y < _ymin) _ymin= y;
  }
  inline void setYAxisLimits(Double_t ymin, Double_t ymax) { 
    _ymin = ymin ;
    _ymax = ymax ;
  }
  inline Double_t getYAxisMin() const { return _ymin; }
  inline Double_t getYAxisMax() const { return _ymax; }

  // the normalization value refers to the full "fit range" instead of
  // the "plot range"
  virtual Double_t getFitRangeNEvt() const = 0;
  virtual Double_t getFitRangeNEvt(Double_t xlo, Double_t xhi) const = 0;
  virtual Double_t getFitRangeBinW() const = 0;

  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  TObject *crossCast();
protected:
  TString _yAxisLabel;
  Double_t _ymin, _ymax, _normValue;
  ClassDef(RooPlotable,1) // Abstract interface for plotable objects
};

#endif
