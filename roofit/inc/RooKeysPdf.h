/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooKeysPdf.rdl,v 1.1 2001/12/01 00:11:21 croat Exp $
 * Authors:
 *   GR, Gerhard Raven, UC, San Diego , Gerhard.Raven@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   04-Jul-2000 GR Created initial version from the original Fortran version
 *                  by Kyle Cranmer, see:
 *                    http://www-wisconsin.cern.ch/~cranmer/keys.html
 *                  Some functionality is still missing (like dealing with
 *                  boundaries)
 *   01-Sep-2000 DK Override useParameters() method to fix the normalization
 *
 * Copyright (C) 2000 UC, San Diego
 *****************************************************************************/
#ifndef ROO_KEYS
#define ROO_KEYS

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;

class RooKeysPdf : public RooAbsPdf {
public:
  enum Mirror { NoMirror, MirrorLeft, MirrorRight, MirrorBoth };
  RooKeysPdf(const char *name, const char *title,
             RooAbsReal& x, RooDataSet& data, Mirror mirror= NoMirror );
  RooKeysPdf(const RooKeysPdf& other, const char* name=0);
  virtual TObject* clone(const char* newname) const {return new RooKeysPdf(*this,newname); }
  virtual ~RooKeysPdf();
  
  void LoadDataSet( RooDataSet& data);
  
protected:
  
  RooRealProxy _x ;
  Double_t evaluate() const;

private:
  
  Double_t evaluateFull(Double_t x);

  Int_t _nEvents;
  Double_t *_dataPts; //!
  Double_t *_weights; //!
  
  enum { _nPoints = 1000 };
  Double_t _lookupTable[_nPoints+1];
  
  Double_t g(Double_t x,Double_t sigma) const;

  Bool_t _mirrorLeft, _mirrorRight;

  // cached info on variable
  Char_t _varName[128];
  Double_t _lo, _hi, _binWidth;
  
  ClassDef(RooKeysPdf,1) // Non-Parametric KEYS PDF
};

#endif
