/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooLinTransBinning.rdl,v 1.2 2002/03/11 07:41:02 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_LIN_TRANS_BINNING
#define ROO_LIN_TRANS_BINNING

#include "Rtypes.h"
#include "RooFitCore/RooAbsBinning.hh"

class RooLinTransBinning : public RooAbsBinning {
public:

  RooLinTransBinning() {} ; 
  RooLinTransBinning(const RooAbsBinning& input, Double_t slope=1.0, Double_t offset=0.0) ;
  RooLinTransBinning(const RooLinTransBinning&) ;
  virtual RooAbsBinning* clone() const { return new RooLinTransBinning(*this) ; }
  virtual ~RooLinTransBinning() ;

  virtual Int_t numBoundaries() const { return _input->numBoundaries() ; }
  virtual Int_t binNumber(Double_t x) const { return _input->binNumber(invTrans(x)) ; }
  virtual Double_t binCenter(Int_t bin) const { return trans(_input->binCenter(binTrans(bin))) ; }
  virtual Double_t binWidth(Int_t bin) const { return _slope*_input->binWidth(binTrans(bin)) ; }
  virtual Double_t binLow(Int_t bin) const { if (_slope>0) return trans(_input->binLow(binTrans(bin))) ; else return trans(_input->binHigh(binTrans(bin))) ; }
  virtual Double_t binHigh(Int_t bin) const { if (_slope>0) return trans(_input->binHigh(binTrans(bin))) ; else return trans(_input->binLow(binTrans(bin))) ; }

  virtual void setRange(Double_t xlo, Double_t xhi) ;
  virtual void setMin(Double_t xlo) { setRange(xlo,highBound()) ; }
  virtual void setMax(Double_t xhi) { setRange(lowBound(),xhi) ; }

  virtual Double_t lowBound() const { if (_slope>0) return trans(_input->lowBound()) ; else return trans(_input->highBound()) ; }
  virtual Double_t highBound() const { if (_slope>0) return trans(_input->highBound()) ; else return trans(_input->lowBound()) ; }
  virtual Double_t averageBinWidth() const { return _slope*_input->averageBinWidth() ; }

  virtual Double_t* array() const ;

  void updateInput(const RooAbsBinning& input, Double_t slope=1.0, Double_t offset=0.0) ;

protected:
    
  inline Int_t binTrans(Int_t bin) const { if (_slope>0) return bin ; else return numBins()-bin-1 ; }
  inline Double_t trans(Double_t x) const { return x*_slope + _offset ; }
  inline Double_t invTrans(Double_t x) const { if (_slope==0.) return 0 ; return (x-_offset)/_slope ; }

  Double_t _slope ;
  Double_t _offset ;
  RooAbsBinning* _input ; 
  mutable Double_t *_array ; //!

  ClassDef(RooLinTransBinning,1) // Linear transformation of binning specification
};

#endif
