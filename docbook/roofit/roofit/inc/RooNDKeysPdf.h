/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id$
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
#ifndef ROO_KEYS_PDF
#define ROO_KEYS_PDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooRealConstant.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include <map>
#include <vector>
#include <string>

class RooRealVar;
class RooArgList;
class RooArgSet;

using namespace std;

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

  RooNDKeysPdf(const char *name, const char *title,
               const RooArgList& varList, RooDataSet& data, 
	       TString options="a", Double_t rho=1, Double_t nSigma=3, Bool_t rotate=kTRUE) ; 

  RooNDKeysPdf(const char *name, const char *title,
               RooAbsReal& x, RooDataSet& data, 
               Mirror mirror= NoMirror, Double_t rho=1, Double_t nSigma=3, Bool_t rotate=kTRUE) ; 

  RooNDKeysPdf(const char *name, const char *title,
               RooAbsReal& x, RooAbsReal &y, RooDataSet& data, 
               TString options="a", Double_t rho = 1.0, Double_t nSigma=3, Bool_t rotate=kTRUE); 

  RooNDKeysPdf(const RooNDKeysPdf& other, const char* name=0);
  virtual ~RooNDKeysPdf();

  virtual TObject* clone(const char* newname) const { return new RooNDKeysPdf(*this,newname); }
  
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

  inline void fixShape(Bool_t fix) { 
    createPdf(kFALSE);
    _fixedShape=fix; 
  }

  struct BoxInfo {
    Bool_t filled;
    Bool_t netFluxZ;
    Double_t nEventsBW;
    Double_t nEventsBMSW;
    vector<Double_t> xVarLo, xVarHi;
    vector<Double_t> xVarLoM3s, xVarLoP3s, xVarHiM3s, xVarHiP3s;
    map<Int_t,Bool_t> bpsIdcs;
    vector<Int_t> sIdcs;	
    vector<Int_t> bIdcs;
    vector<Int_t> bmsIdcs;
  } ;

protected:
  
  RooListProxy _varList ;
  TIterator* _varItr ;   //! do not persist

  Double_t evaluate() const;


  void     createPdf(Bool_t firstCall=kTRUE) const;  
  void     setOptions() const;
  void     initialize() const; 
  void     loadDataSet(Bool_t firstCall) const;
  void     mirrorDataSet() const;
  void     loadWeightSet() const;
  void     calculateShell(BoxInfo* bi) const;
  void     calculatePreNorm(BoxInfo* bi) const;
  void     sortDataIndices(BoxInfo* bi=0) const;
  void     calculateBandWidth() const;
  Double_t gauss(vector<Double_t>& x, vector<vector<Double_t> >& weights) const;
  void     loopRange(vector<Double_t>& x, map<Int_t,Bool_t>& ibMap) const;
  void     boxInfoInit(BoxInfo* bi, const char* rangeName, Int_t code) const;

  RooDataSet& _data;
  mutable TString _options;
  mutable Double_t _widthFactor;
  mutable Double_t _nSigma;

  mutable Bool_t _fixedShape;
  mutable Bool_t _mirror;
  mutable Bool_t _debug;
  mutable Bool_t _verbose;

  mutable Double_t _sqrt2pi;
  mutable Int_t _nDim;
  mutable Int_t _nEvents;
  mutable Int_t _nEventsM;
  mutable Double_t _nEventsW;
  mutable Double_t _d;
  mutable Double_t _n;
  
  // cached info on variable

  mutable vector<vector<Double_t> > _dataPts;
  mutable vector<TVectorD> _dataPtsR;
  mutable vector<vector<Double_t> > _weights0;
  mutable vector<vector<Double_t> > _weights1;
  mutable vector<vector<Double_t> >* _weights; //!

#ifndef __CINT__
  mutable vector<iiVec> _sortIdcs;
  mutable vector<itVec> _sortTVIdcs;
#endif

  mutable vector<string> _varName;
  mutable vector<Double_t> _rho;
  mutable RooArgSet _dataVars;
  mutable vector<Double_t> _x;
  mutable vector<Double_t> _x0, _x1, _x2;
  mutable vector<Double_t> _mean, _sigma;
  mutable vector<Double_t> _xDatLo, _xDatHi;
  mutable vector<Double_t> _xDatLo3s, _xDatHi3s;

  mutable Bool_t _netFluxZ;
  mutable Double_t _nEventsBW;
  mutable Double_t _nEventsBMSW;
  mutable vector<Double_t> _xVarLo, _xVarHi;
  mutable vector<Double_t> _xVarLoM3s, _xVarLoP3s, _xVarHiM3s, _xVarHiP3s;
  mutable map<Int_t,Bool_t> _bpsIdcs;
  mutable vector<Int_t>	_sIdcs;	
  mutable vector<Int_t> _bIdcs;
  mutable vector<Int_t> _bmsIdcs;

  mutable map<pair<string,int>,BoxInfo*> _rangeBoxInfo ;
  mutable BoxInfo _fullBoxInfo ;

  mutable vector<Int_t> _idx;
  mutable Double_t _minWeight;
  mutable Double_t _maxWeight;
  mutable map<Int_t,Double_t> _wMap;

  mutable TMatrixDSym* _covMat; 
  mutable TMatrixDSym* _corrMat; 
  mutable TMatrixD* _rotMat; 
  mutable TVectorD* _sigmaR; 
  mutable TVectorD* _dx;
  mutable Double_t _sigmaAvgR;

  mutable Bool_t _rotate;

  /// sorter function
  struct SorterTV_L2H {
    Int_t idx;
    
    SorterTV_L2H (Int_t index) : idx(index) {}
    bool operator() (const itPair& a, const itPair& b) {
      const TVectorD& aVec = *(a.second); 
      const TVectorD& bVec = *(b.second); 
      return (aVec[idx]<bVec[idx]);
    }
  };

  ClassDef(RooNDKeysPdf,0) // General N-dimensional non-parametric kernel estimation p.d.f
};

#endif
