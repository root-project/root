/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPlotable.h,v 1.14 2007/05/11 09:11:30 verkerke Exp $
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
  inline ~RooPlotable() override { }

  inline const char* getYAxisLabel() const { return _yAxisLabel.Data(); }
  inline void setYAxisLabel(const char *label) { _yAxisLabel= label; }
  inline void updateYAxisLimits(double y) {
    if(y > _ymax) _ymax= y;
    if(y < _ymin) _ymin= y;
  }
  inline void setYAxisLimits(double ymin, double ymax) {
    _ymin = ymin ;
    _ymax = ymax ;
  }
  inline double getYAxisMin() const { return _ymin; }
  inline double getYAxisMax() const { return _ymax; }

  // the normalization value refers to the full "fit range" instead of
  // the "plot range"
  virtual double getFitRangeNEvt() const = 0;
  virtual double getFitRangeNEvt(double xlo, double xhi) const = 0;
  virtual double getFitRangeBinW() const = 0;

  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent= "") const override;

  TObject *crossCast();
protected:
  TString _yAxisLabel;
  double _ymin, _ymax, _normValue;
  ClassDefOverride(RooPlotable,1) // Abstract interface for plotable objects in a RooPlot
};

#endif
