/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlotable.rdl,v 1.3 2001/05/14 22:54:21 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   29-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_PLOTABLE
#define ROO_PLOTABLE

#include "Rtypes.h"
#include "TString.h"
#include "RooFitCore/RooPrintable.hh"

class TObject;
class RooArgSet;

class RooPlotable : public RooPrintable {
public:
  inline RooPlotable() : _ymin(0), _ymax(0), _normValue(0) { }
  inline virtual ~RooPlotable() { }

  inline const char* getYAxisLabel() const { return _yAxisLabel.Data(); }
  inline setYAxisLabel(const char *label) { _yAxisLabel= label; }
  inline void updateYAxisLimits(Double_t y) {
    if(y > _ymax) _ymax= y;
    if(y < _ymin) _ymin= y;
  }
  inline Double_t getYAxisMin() const { return _ymin; }
  inline Double_t getYAxisMax() const { return _ymax; }

  // the normalization value refers to the full "fit range" instead of
  // the "plot range"
  virtual Double_t getFitRangeNorm() const = 0;

  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  TObject *crossCast();
protected:
  TString _yAxisLabel;
  Double_t _ymin, _ymax, _normValue;
  ClassDef(RooPlotable,1) // Abstract interface for plotable objects
};

#endif
