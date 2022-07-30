/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGrid.h,v 1.10 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_GRID
#define ROO_GRID

#include "TObject.h"
#include "RooPrintable.h"

#include <vector>

class RooAbsFunc;

class RooGrid : public TObject, public RooPrintable {
public:
  RooGrid() {}
  RooGrid(const RooAbsFunc &function);

  // Printing interface
  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override;

  inline void Print(Option_t *options= 0) const override {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  inline bool isValid() const { return _valid; }
  inline UInt_t getDimension() const { return _dim; }
  inline double getVolume() const { return _vol; }
  inline UInt_t getNBins() const { return _bins; }
  inline UInt_t getNBoxes() const { return _boxes; }
  inline void setNBoxes(UInt_t boxes) { _boxes= boxes; }

  bool initialize(const RooAbsFunc &function);
  void resize(UInt_t bins);
  void resetValues();
  void generatePoint(const UInt_t box[], double x[], UInt_t bin[],
           double &vol, bool useQuasiRandom= true) const;
  void accumulate(const UInt_t bin[], double amount);
  void refine(double alpha= 1.5);

  void firstBox(UInt_t box[]) const;
  bool nextBox(UInt_t box[]) const;

  enum { maxBins = 50 }; // must be even

  // Accessor for the j-th normalized grid point along the i-th dimension
public:
  inline double coord(Int_t i, Int_t j) const { return _xi[i*_dim + j]; }
  inline double value(Int_t i,Int_t j) const { return _d[i*_dim + j]; }
protected:
  inline double& coord(Int_t i, Int_t j) { return _xi[i*_dim + j]; }
  inline double& value(Int_t i,Int_t j) { return _d[i*_dim + j]; }
  inline double& newCoord(Int_t i) { return _xin[i]; }

protected:

  bool _valid = false;         ///< Is configuration valid
  UInt_t _dim = 0;             ///< Number of dimensions, bins and boxes
  UInt_t _bins = 0;            ///< Number of bins
  UInt_t _boxes = 0;           ///<Numbser of boxes
  double _vol = 0.0;           ///< Volume

  std::vector<double> _xl;     ///<! Internal workspace
  std::vector<double> _xu;     ///<! Internal workspace
  std::vector<double> _delx;   ///<! Internal workspace
  std::vector<double> _d;      ///<! Internal workspace
  std::vector<double> _xi;     ///<! Internal workspace
  std::vector<double> _xin;    ///<! Internal workspace
  std::vector<double> _weight; ///<! Internal workspace

  ClassDefOverride(RooGrid,1) // Utility class for RooMCIntegrator holding a multi-dimensional grid
};

#endif


