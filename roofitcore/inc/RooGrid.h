/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooIntegrator1D.rdl,v 1.6 2001/08/02 23:54:24 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   08-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_GRID
#define ROO_GRID

#include "TObject.h"
#include "RooFitCore/RooPrintable.hh"

class RooAbsFunc;

class RooGrid : public TObject, public RooPrintable {
public:
  RooGrid(const RooAbsFunc &function);
  virtual ~RooGrid();

  // Printing interface
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  inline Bool_t isValid() const { return _valid; }
  inline UInt_t getDimension() const { return _dim; }
  inline Double_t getVolume() const { return _vol; }
  inline UInt_t getNBins() const { return _bins; }
  inline UInt_t getNBoxes() const { return _boxes; }
  inline void setNBoxes(UInt_t boxes) { _boxes= boxes; }

  inline Double_t *createPoint() const { return _valid ? new Double_t[_dim] : 0; }
  inline UInt_t *createIndexVector() const { return _valid ? new UInt_t[_dim] : 0; }

  Bool_t initialize(const RooAbsFunc &function);
  void resize(UInt_t bins);
  void resetValues();
  void generatePoint(const UInt_t box[], Double_t x[], UInt_t bin[], Double_t &vol) const;
  void accumulate(const UInt_t bin[], Double_t amount);
  void refine(Double_t alpha= 1.5);

  void firstBox(UInt_t box[]) const;
  Bool_t nextBox(UInt_t box[]) const;

  enum { maxBins = 50 }; // must be even

  // Accessor for the j-th normalized grid point along the i-th dimension
public:
  inline Double_t coord(Int_t i, Int_t j) const { return _xi[i*_dim + j]; }
  inline Double_t value(Int_t i,Int_t j) const { return _d[i*_dim + j]; }
protected:
  inline Double_t& coord(Int_t i, Int_t j) { return _xi[i*_dim + j]; }
  inline Double_t& value(Int_t i,Int_t j) { return _d[i*_dim + j]; }
  inline Double_t& newCoord(Int_t i) { return _xin[i]; }

protected:

  Bool_t _valid;
  UInt_t _dim,_bins,_boxes;
  Double_t _vol;

  Double_t *_xl;     //! do not persist
  Double_t *_xu;     //! do not persist
  Double_t *_delx;   //! do not persist
  Double_t *_d;      //! do not persist
  Double_t *_xi;     //! do not persist
  Double_t *_xin;    //! do not persist
  Double_t *_weight; //! do not persist

  ClassDef(RooGrid,1) // a multi-dimensional grid
};

#endif


