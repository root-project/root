/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_ABS_BINNING
#define ROO_ABS_BINNING

#include "Rtypes.h"
#include "RooFitCore/RooPrintable.hh"
class TIterator ;

class RooAbsBinning : public RooPrintable {
public:

  RooAbsBinning() ;
  RooAbsBinning(const RooAbsBinning&) {}
  virtual RooAbsBinning* clone() const = 0 ;
  virtual ~RooAbsBinning() ;

  Int_t numBins() const { return numBoundaries()-1 ; }
  virtual Int_t numBoundaries() const = 0 ;
  virtual Int_t binNumber(Double_t x) const = 0 ;
  virtual Double_t binCenter(Int_t bin) const = 0 ;
  virtual Double_t binWidth(Int_t bin) const = 0 ;
  virtual Double_t binLow(Int_t bin) const = 0 ;
  virtual Double_t binHigh(Int_t bin) const = 0 ;

  virtual void setRange(Double_t xlo, Double_t xhi) = 0 ;
  virtual void setMin(Double_t xlo) { setRange(xlo,highBound()) ; }
  virtual void setMax(Double_t xhi) { setRange(lowBound(),xhi) ; }

  virtual Double_t lowBound() const = 0 ;
  virtual Double_t highBound() const = 0 ;
  virtual Double_t averageBinWidth() const = 0 ;


  virtual Double_t* array() const = 0 ;

  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }
  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const;

protected:
  

  ClassDef(RooAbsBinning,1) // Abstract base class for binning specification
};

#endif
