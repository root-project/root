/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.rdl,v 1.6 2001/04/22 18:15:32 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_PLOT
#define ROO_PLOT

#include "TH1.h"
#include "RooFitCore/RooList.hh"
#include "RooFitCore/RooPrintable.hh"

class RooAbsReal;
class RooArgSet ;
class RooHist;
class TAttLine;
class TAttFill;
class TAttMarker;
class TAttText;

class RooPlot : public TH1, public RooPrintable {
public:
  RooPlot(const RooAbsReal &var);
  RooPlot(Float_t xmin= 0, Float_t xmax= 1);
  RooPlot(Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax);
  virtual ~RooPlot();
  Stat_t GetBinContent(Int_t) const;
  TObject *addObject(const TObject* obj, Option_t* drawOptions= "");
  RooHist *addHistogram(const TH1* data, Option_t* drawOptions= "P");
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }
  virtual void Draw(Option_t *options= 0);
  inline RooAbsReal *getPlotVar() const { return _plotVarClone; }

  TObject *findObject(const char *name) const;

  inline Double_t getPadFactor() const { return _padFactor; }
  inline void setPadFactor(Double_t factor) { if(factor > 0) _padFactor= factor; }

  TAttLine *getAttLine(const char *name) const;
  TAttFill *getAttFill(const char *name) const;
  TAttMarker *getAttMarker(const char *name) const;
  TAttText *getAttText(const char *name) const;

  Bool_t drawBefore(const char *before, const char *target);
  Bool_t drawAfter(const char *afger, const char *target);

  TString getDrawOptions(const char *name) const;
  Bool_t setDrawOptions(const char *name, TString options);

  inline void origPrint(Option_t* opt) { TH1::Print(opt) ; }

protected:
  void initialize();
  TString histName() const ; 
  TString caller(const char *method) const;
  Double_t _padFactor; // Scale our y-axis to _padFactor of our maximum contents.
  RooList _items;    // A list of the items we contain.
  RooAbsReal *_plotVarClone; // A clone of the variable we are plotting.
  RooArgSet *_plotVarSet; // A list owning the cloned tree nodes of the plotVarClone ;
  TIterator *_iterator;  //! non-persistent
  RooPlot(const RooPlot& other); // object cannot be copied
  ClassDef(RooPlot,1) // A plot frame and container for graphics objects
};

#endif
