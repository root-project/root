/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsBinning.h,v 1.13 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_BINNING
#define ROO_ABS_BINNING

#include "Rtypes.h"
#include "RooPrintable.h"
#include "TNamed.h" 
class TIterator ;
class RooAbsRealLValue ;
class RooAbsReal ;

class RooAbsBinning : public TNamed, public RooPrintable {
public:

  RooAbsBinning(const char* name=0) ;
  RooAbsBinning(const RooAbsBinning& other, const char* name=0) : TNamed(name,name), RooPrintable(other) {}
  virtual TObject* Clone(const char* newname=0) const { return clone(newname) ; }
  virtual RooAbsBinning* clone(const char* name=0) const = 0 ;
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
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  virtual void printName(ostream& os) const ;
  virtual void printTitle(ostream& os) const ;
  virtual void printClassName(ostream& os) const ;
  virtual void printArgs(ostream& os) const ;
  virtual void printValue(ostream& os) const ;
  

  virtual Bool_t isParameterized() const { return kFALSE ; }
  virtual RooAbsReal* lowBoundFunc() const { return 0 ; }
  virtual RooAbsReal* highBoundFunc() const { return 0 ; }
  virtual Bool_t isShareable() const { return kTRUE ; }
  virtual void insertHook(RooAbsRealLValue&) const {} ;
  virtual void removeHook(RooAbsRealLValue&) const {} ;

protected:
  

  ClassDef(RooAbsBinning,2) // Abstract base class for binning specification
};

#endif
