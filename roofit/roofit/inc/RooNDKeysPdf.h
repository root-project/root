/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooNDKeysPdf.h 44368 2012-05-30 15:38:44Z axel $
 * Authors:                                                                  *
 *   Max Baak, CERN, mbaak@cern.ch *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_NDKEYS_PDF
#define ROO_NDKEYS_PDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooDataSet.h"
#include "RooListProxy.h"
#include "TH1.h"
#include "TAxis.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include <map>
#include <vector>
#include <string>

class RooRealVar;
class RooArgList;
class RooArgSet;
class RooChangeTracker;

#ifndef __CINT__
class VecVecDouble : public std::vector<std::vector<Double_t> >  { } ;
class VecTVecDouble : public std::vector<TVectorD> { } ;
typedef std::pair<Int_t, VecVecDouble::iterator > iiPair;
typedef std::vector< iiPair > iiVec;
typedef std::pair<Int_t, VecTVecDouble::iterator > itPair;
typedef std::vector< itPair > itVec;
#else
class itPair ;
#endif

class RooNDKeysPdf : public RooAbsPdf {

public:

  enum Mirror {NoMirror, MirrorLeft, MirrorRight, MirrorBoth,
               MirrorAsymLeft, MirrorAsymLeftRight,
               MirrorAsymRight, MirrorLeftAsymRight,
               MirrorAsymBoth };

  RooNDKeysPdf() = default;

  RooNDKeysPdf(const char *name, const char *title, const RooArgList &varList, const RooDataSet &data,
               TString options = "ma", Double_t rho = 1, Double_t nSigma = 3, bool rotate = true,
               bool sortInput = true);

  RooNDKeysPdf(const char *name, const char *title, const RooArgList &varList, const TH1 &hist, TString options = "ma",
               Double_t rho = 1, Double_t nSigma = 3, bool rotate = true, bool sortInput = true);

  RooNDKeysPdf(const char *name, const char *title, const RooArgList &varList, const RooDataSet &data,
               const TVectorD &rho, TString options = "ma", Double_t nSigma = 3, bool rotate = true,
               bool sortInput = true);

  RooNDKeysPdf(const char *name, const char *title, const RooArgList &varList, const RooDataSet &data,
               const RooArgList &rhoList, TString options = "ma", Double_t nSigma = 3, bool rotate = true,
               bool sortInput = true);

  RooNDKeysPdf(const char *name, const char *title, const RooArgList &varList, const TH1 &hist,
               const RooArgList &rhoList, TString options = "ma", Double_t nSigma = 3, bool rotate = true,
               bool sortInput = true);

  RooNDKeysPdf(const char *name, const char *title, RooAbsReal &x, const RooDataSet &data, Mirror mirror = NoMirror,
               Double_t rho = 1, Double_t nSigma = 3, bool rotate = true, bool sortInput = true);

  RooNDKeysPdf(const char *name, const char *title, RooAbsReal &x, RooAbsReal &y, const RooDataSet &data,
               TString options = "ma", Double_t rho = 1.0, Double_t nSigma = 3, bool rotate = true,
               bool sortInput = true);

  RooNDKeysPdf(const RooNDKeysPdf& other, const char* name=0);
  ~RooNDKeysPdf() override;

  TObject* clone(const char* newname) const override { return new RooNDKeysPdf(*this,newname); }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

  inline void fixShape(bool fix) {
    createPdf(false);
    _fixedShape=fix;
  }

  TMatrixD getWeights(const int &k) const;

  struct BoxInfo {
    bool filled;
    bool netFluxZ;
    Double_t nEventsBW;
    Double_t nEventsBMSW;
    std::vector<Double_t> xVarLo, xVarHi;
    std::vector<Double_t> xVarLoM3s, xVarLoP3s, xVarHiM3s, xVarHiP3s;
    std::map<Int_t,bool> bpsIdcs;
    std::vector<Int_t> sIdcs;
    std::vector<Int_t> bIdcs;
    std::vector<Int_t> bmsIdcs;
  } ;

protected:

  RooListProxy _varList ;
  RooListProxy _rhoList;

  Double_t evaluate() const override;

  void createPdf(bool firstCall = true);
  void setOptions();
  void initialize();
  void loadDataSet(bool firstCall);
  void mirrorDataSet();
  void loadWeightSet();
  void calculateShell(BoxInfo *bi) const;
  void calculatePreNorm(BoxInfo *bi) const;
  void sortDataIndices(BoxInfo *bi = 0);
  void calculateBandWidth();
  Double_t gauss(std::vector<Double_t> &x, std::vector<std::vector<Double_t>> &weights) const;
  void loopRange(std::vector<Double_t> &x, std::map<Int_t, bool> &ibMap) const;
  void boxInfoInit(BoxInfo *bi, const char *rangeName, Int_t code) const;
  RooDataSet *createDatasetFromHist(const RooArgList &varList, const TH1 &hist) const;
  void updateRho() const;
  void checkInitWeights() const {
    if (_weights == &_weights0 || _weights == &_weights1)
      return;
    const_cast<RooNDKeysPdf*>(this)->calculateBandWidth();
  }

  std::unique_ptr<RooDataSet> _ownedData{nullptr};
  const RooDataSet* _data; //! do not persist
  mutable TString _options;
  Double_t _widthFactor;
  Double_t _nSigma;

  bool _fixedShape{false};
  bool _mirror{false};
  bool _debug{false};   //!
  bool _verbose{false}; //!

  Int_t _nDim{0};
  Int_t _nEvents{0};
  Int_t _nEventsM{0};
  Double_t _nEventsW{0.};
  Double_t _d{0.};
  Double_t _n{0.};

  // cached info on variable
  std::vector<std::vector<Double_t> > _dataPts;
  std::vector<TVectorD> _dataPtsR;
  std::vector<std::vector<Double_t> > _weights0;            // Plain weights
  std::vector<std::vector<Double_t> > _weights1;            // Weights for adaptive kernels
  std::vector<std::vector<Double_t> >* _weights{nullptr};   //! Weights to be used. Points either to _weights0 or _weights1

  std::vector<itVec> _sortTVIdcs; //!

  std::vector<std::string> _varName;
  mutable std::vector<Double_t> _rho;
  RooArgSet _dataVars;
  mutable std::vector<Double_t> _x; // Cache for x values
  std::vector<Double_t> _x0, _x1, _x2;
  std::vector<Double_t> _mean, _sigma;
  std::vector<Double_t> _xDatLo, _xDatHi;
  std::vector<Double_t> _xDatLo3s, _xDatHi3s;

  bool _netFluxZ{false};
  Double_t _nEventsBW{0.};
  Double_t _nEventsBMSW{0.};
  std::vector<Double_t> _xVarLo, _xVarHi;
  std::vector<Double_t> _xVarLoM3s, _xVarLoP3s, _xVarHiM3s, _xVarHiP3s;
  std::map<Int_t,bool> _bpsIdcs;
  std::map<Int_t, bool> _ibNoSort;
  std::vector<Int_t> _sIdcs;
  std::vector<Int_t> _bIdcs;
  std::vector<Int_t> _bmsIdcs;

  // Data for analytical integrals:
  mutable std::map<std::pair<std::string,int>,BoxInfo*> _rangeBoxInfo ;
  mutable BoxInfo _fullBoxInfo ;

  std::vector<Int_t> _idx;
  Double_t _minWeight{0.};
  Double_t _maxWeight{0.};
  std::map<Int_t,Double_t> _wMap;

  TMatrixDSym* _covMat{nullptr};
  TMatrixDSym* _corrMat{nullptr};
  TMatrixD* _rotMat{nullptr};
  TVectorD* _sigmaR{nullptr};
  TVectorD* _dx{nullptr};
  Double_t _sigmaAvgR{0.};

  bool _rotate;
  bool _sortInput;
  Int_t _nAdpt;

  RooChangeTracker *_tracker{nullptr}; //

  ClassDefOverride(RooNDKeysPdf, 1) // General N-dimensional non-parametric kernel estimation p.d.f
};

#endif
