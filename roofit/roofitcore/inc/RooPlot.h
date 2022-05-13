/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPlot.h,v 1.37 2007/06/18 11:52:41 wouter Exp $
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
#ifndef ROO_PLOT
#define ROO_PLOT

#include "RooPrintable.h"
#include "TNamed.h"

#include <memory>
#include <float.h>

class TH1 ;

class RooAbsReal;
class RooAbsRealLValue;
class RooArgSet ;
class RooHist;
class RooCurve ;
class RooPlotable;
class TDirectory ;
class TAttLine;
class TAttFill;
class TAttMarker;
class TAttText;
class TClass ;
class TAxis;
class TBrowser ;
class TLegend;

class RooPlot : public TNamed, public RooPrintable {
public:
  using Items = std::vector<std::pair<TObject*,std::string>>;

  RooPlot() ;
  RooPlot(const char* name, const char* title, const RooAbsRealLValue &var, double xmin, double xmax, Int_t nBins) ;
  RooPlot(const RooAbsRealLValue &var, double xmin, double xmax, Int_t nBins);
  RooPlot(double xmin, double xmax);
  RooPlot(double xmin, double xmax, double ymin, double ymax);
  RooPlot(const RooAbsRealLValue &var1, const RooAbsRealLValue &var2);
  RooPlot(const RooAbsRealLValue &var1, const RooAbsRealLValue &var2,
     double xmin, double xmax, double ymin, double ymax);
  ~RooPlot() override;

  static RooPlot* frame(const RooAbsRealLValue &var, double xmin, double xmax, Int_t nBins);
  static RooPlot* frameWithLabels(const RooAbsRealLValue &var);

  RooPlot* emptyClone(const char* name) ;

  // implement the TH1 interface
  virtual Stat_t GetBinContent(Int_t) const;
  virtual Stat_t GetBinContent(Int_t, Int_t) const;
  virtual Stat_t GetBinContent(Int_t, Int_t, Int_t) const;
  void Draw(Option_t *options= 0) override;

  // forwarding of relevant TH1 interface
  TAxis* GetXaxis() const ;
  TAxis* GetYaxis() const ;
  Int_t GetNbinsX() const ;
  Int_t GetNdivisions(Option_t* axis = "X") const ;
  double GetMinimum(double minval = -FLT_MAX) const ;
  double GetMaximum(double maxval = FLT_MAX) const ;

  void SetAxisColor(Color_t color = 1, Option_t* axis = "X") ;
  void SetAxisRange(double xmin, double xmax, Option_t* axis = "X") ;
  void SetBarOffset(Float_t offset = 0.25) ;
  void SetBarWidth(Float_t width = 0.5) ;
  void SetContour(Int_t nlevels, const double* levels = 0) ;
  void SetContourLevel(Int_t level, double value) ;
  void SetDrawOption(Option_t* option = "") override ;
  void SetFillAttributes() ;
  void SetFillColor(Color_t fcolor) ;
  void SetFillStyle(Style_t fstyle) ;
  void SetLabelColor(Color_t color = 1, Option_t* axis = "X") ;
  void SetLabelFont(Style_t font = 62, Option_t* axis = "X") ;
  void SetLabelOffset(Float_t offset = 0.005, Option_t* axis = "X") ;
  void SetLabelSize(Float_t size = 0.02, Option_t* axis = "X") ;
  void SetLineAttributes() ;
  void SetLineColor(Color_t lcolor) ;
  void SetLineStyle(Style_t lstyle) ;
  void SetLineWidth(Width_t lwidth) ;
  void SetMarkerAttributes() ;
  void SetMarkerColor(Color_t tcolor = 1) ;
  void SetMarkerSize(Size_t msize = 1) ;
  void SetMarkerStyle(Style_t mstyle = 1) ;
  void SetName(const char *name) override ;
  void SetTitle(const char *name) override ;
  void SetNameTitle(const char *name, const char* title) override ;
  void SetNdivisions(Int_t n = 510, Option_t* axis = "X") ;
  void SetOption(Option_t* option = " ") ;
  void SetStats(bool stats = true) ;
  void SetTickLength(Float_t length = 0.02, Option_t* axis = "X") ;
  void SetTitleFont(Style_t font = 62, Option_t* axis = "X") ;
  void SetTitleOffset(Float_t offset = 1, Option_t* axis = "X") ;
  void SetTitleSize(Float_t size = 0.02, Option_t* axis = "X") ;
  void SetXTitle(const char* title) ;
  void SetYTitle(const char* title) ;
  void SetZTitle(const char* title) ;

  // container management
  const char* nameOf(Int_t idx) const ;
  TObject *findObject(const char *name, const TClass* clas=0) const;
  TObject* getObject(Int_t idx) const ;
  Stat_t numItems() const {return _items.size();}

  void addPlotable(RooPlotable *plotable, Option_t *drawOptions= "", bool invisible=false, bool refreshNorm=false);
  void addObject(TObject* obj, Option_t* drawOptions= "", bool invisible=false);
  void addTH1(TH1 *hist, Option_t* drawOptions= "", bool invisible=false);
  std::unique_ptr<TLegend> BuildLegend() const;

  void remove(const char* name=0, bool deleteToo=true) ;

  // ascii printing
  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printArgs(std::ostream& os) const override ;
  void printValue(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override ;

  Int_t defaultPrintContents(Option_t* opt) const override ;

  inline void Print(Option_t *options= 0) const override {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  // data member get/set methods
  inline RooAbsRealLValue *getPlotVar() const { return _plotVarClone; }
  ///Return the number of events in the fit range
  inline double getFitRangeNEvt() const { return _normNumEvts; }
  double getFitRangeNEvt(double xlo, double xhi) const ;
  ///Return the bin width that is being used to normalise the PDF
  inline double getFitRangeBinW() const { return _normBinWidth; }
  inline double getPadFactor() const { return _padFactor; }
  inline void setPadFactor(double factor) { if(factor >= 0) _padFactor= factor; }
  void updateNormVars(const RooArgSet &vars);
  const RooArgSet *getNormVars() const { return _normVars; }

  // get attributes of contained objects
  TAttLine *getAttLine(const char *name=0) const;
  TAttFill *getAttFill(const char *name=0) const;
  TAttMarker *getAttMarker(const char *name=0) const;
  TAttText *getAttText(const char *name=0) const;

  // Convenient type-safe accessors
  RooCurve* getCurve(const char* name=0) const ;
  RooHist* getHist(const char* name=0) const ;


  // rearrange drawing order of contained objects
  bool drawBefore(const char *before, const char *target);
  bool drawAfter(const char *after, const char *target);

  // get/set drawing options for contained objects
  TString getDrawOptions(const char *name) const;
  bool setDrawOptions(const char *name, TString options);

  bool getInvisible(const char* name) const ;
  void setInvisible(const char* name, bool flag=true) ;

  virtual void SetMaximum(double maximum = -1111) ;
  virtual void SetMinimum(double minimum = -1111) ;

  ///Shortcut for RooPlot::chiSquare(const char* pdfname, const char* histname, int nFitParam=0)
  double chiSquare(int nFitParam=0) const { return chiSquare(0,0,nFitParam) ; }
  double chiSquare(const char* pdfname, const char* histname, int nFitParam=0) const ;

  RooHist* residHist(const char* histname=0, const char* pdfname=0,bool normalize=false, bool useAverage=true) const ;
  ///Uses residHist() and sets normalize=true
  RooHist* pullHist(const char* histname=0, const char* pdfname=0, bool useAverage=true) const
    { return residHist(histname,pdfname,true,useAverage); }

  void Browse(TBrowser *b) override ;

  /// \copydoc AddDirectoryStatus()
  static bool addDirectoryStatus() ;
  /// \copydoc AddDirectory()
  static bool setAddDirectoryStatus(bool flag) ;

  /// Configure whether new instances of RooPlot will add themselves to `gDirectory`.
  /// Like TH1::AddDirectory().
  static void AddDirectory(bool add=true) {
    setAddDirectoryStatus(add);
  }
  /// Query whether new instances of RooPlot will add themselves to `gDirectory`.
  /// When a file has been opened before a RooPlot instance is created,
  /// this instance will be associated to the file. Closing the file will e.g.
  /// write the instance to the file, and then delete it.
  /// Like TH1::AddDirectoryStatus().
  static bool AddDirectoryStatus() {
    return addDirectoryStatus();
  }

  void SetDirectory(TDirectory *dir);

  static void fillItemsFromTList(Items & items, TList const& tlist);

protected:

  RooPlot(const RooPlot& other) = delete; // cannot be copied

  class DrawOpt {
    public:

    DrawOpt(const char* _rawOpt=0) : invisible(false) { drawOptions[0] = 0 ; initialize(_rawOpt) ; }
    void initialize(const char* _rawOpt) ;
    const char* rawOpt() const ;

    char drawOptions[128] ;
    bool invisible ;
  } ;


  void initialize();
  TString histName() const ;
  Items::iterator findItem(std::string const& name);
  Items::const_iterator findItem(std::string const& name) const;

  void updateYAxis(double ymin, double ymax, const char *label= "");
  void updateFitRangeNorm(const TH1* hist);
  void updateFitRangeNorm(const RooPlotable* rp, bool refeshNorm=false);

  TH1* _hist = nullptr;      ///< Histogram that we uses as basis for drawing the content
  Items _items;  ///< A list of the items we contain.
  double _padFactor;       ///< Scale our y-axis to _padFactor of our maximum contents.
  RooAbsRealLValue *_plotVarClone = nullptr; ///< A clone of the variable we are plotting.
  RooArgSet *_plotVarSet = nullptr; ///< A list owning the cloned tree nodes of the plotVarClone
  RooArgSet *_normVars = nullptr; ///< Variables that PDF plots should be normalized over

  const RooPlotable* _normObj = nullptr; ///<! Pointer to normalization object ;
  double _normNumEvts;     ///< Number of events in histogram (for normalization)
  double _normBinWidth;    ///< Histogram bin width (for normalization)

  double _defYmin = 1e-5; ///< Default minimum for Yaxis (as calculated from contents)
  double _defYmax = 1.0;  ///< Default maximum for Yaxis (as calculated from contents)

  TDirectory* _dir = nullptr; ///<! non-persistent

  static bool _addDirStatus ; ///< static flag controlling AutoDirectoryAdd feature

  ClassDefOverride(RooPlot,3)        // Plot frame and container for graphics objects
};

#endif
