/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooUniformBinning.rdl,v 1.1 2002/03/07 06:22:24 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_UNIFORM_BINNING
#define ROO_UNIFORM_BINNING

#include "Rtypes.h"
#include "RooFitCore/RooAbsBinning.hh"
class TIterator ;

class RooUniformBinning : public RooAbsBinning {
public:

  RooUniformBinning() ;
  RooUniformBinning(Double_t xlo, Double_t xhi, Int_t nBins) ;
  RooUniformBinning(const RooUniformBinning& other) ;
  RooAbsBinning* clone() const { return new RooUniformBinning(*this) ; }
  virtual ~RooUniformBinning() ;

  virtual void setRange(Double_t xlo, Double_t xhi) ;

  virtual Int_t numBoundaries() const { return _nbins + 1 ; }
  virtual Int_t binNumber(Double_t x) const  ;

  virtual Double_t lowBound() const { return _xlo ; }
  virtual Double_t highBound() const { return _xhi ; }

  virtual Double_t binCenter(Int_t bin) const ;
  virtual Double_t binWidth(Int_t bin) const ;
  virtual Double_t binLow(Int_t bin) const ;
  virtual Double_t binHigh(Int_t bin) const ;

  virtual Double_t averageBinWidth() const { return _binw ; }
  virtual Double_t* array() const ;

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const;

protected:

  mutable Double_t* _array ; //! do not persist
  Double_t _xlo ;
  Double_t _xhi ;
  Int_t    _nbins ;
  Double_t _binw ;


  ClassDef(RooUniformBinning,1) // Uniform binning specification
};

#endif
