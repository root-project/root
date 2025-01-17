/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooMomentMorphFunc
    \ingroup Roofit

**/

#include "Riostream.h"

#include "RooMomentMorphFunc.h"
#include "RooAbsCategory.h"
#include "RooRealConstant.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooCustomizer.h"
#include "RooRealSumFunc.h"
#include "RooAddition.h"
#include "RooMoment.h"
#include "RooLinearVar.h"
#include "RooChangeTracker.h"

#include "TMath.h"
#include "TH1.h"

using std::string, std::vector;

ClassImp(RooMomentMorphFunc)

//_____________________________________________________________________________
RooMomentMorphFunc::RooMomentMorphFunc()
   : _cacheMgr(this, 10, true, true)
{
}

//_____________________________________________________________________________
RooMomentMorphFunc::RooMomentMorphFunc(const char *name, const char *title, RooAbsReal &_m, const RooArgList &varList,
                                       const RooArgList &pdfList, const TVectorD &mrefpoints, Setting setting)
   : RooAbsReal(name, title),
     _cacheMgr(this, 10, true, true),
     m("m", "m", this, _m),
     _varList("varList", "List of variables", this),
     _pdfList("pdfList", "List of pdfs", this),
     _mref(new TVectorD(mrefpoints)),
     _setting(setting)
{
   // observables
  _varList.addTyped<RooAbsReal>(varList);

   // reference p.d.f.s
  _pdfList.addTyped<RooAbsPdf>(pdfList);

   // initialization
   initialize();
}

//_____________________________________________________________________________
RooMomentMorphFunc::RooMomentMorphFunc(const char *name, const char *title, RooAbsReal &_m, const RooArgList &varList,
                                       const RooArgList &pdfList, const RooArgList &mrefList, Setting setting)
   : RooAbsReal(name, title),
     _cacheMgr(this, 10, true, true),
     m("m", "m", this, _m),
     _varList("varList", "List of variables", this),
     _pdfList("pdfList", "List of pdfs", this),
     _mref(new TVectorD(mrefList.size())),
     _setting(setting)
{
   // observables
  _varList.addTyped<RooAbsReal>(varList);

   // reference p.d.f.s
  _pdfList.addTyped<RooAbsPdf>(pdfList);

   // reference points in m

   Int_t i = 0;
   for (auto *mref : mrefList) {
      if (!dynamic_cast<RooAbsReal *>(mref)) {
         coutE(InputArguments) << "RooMomentMorphFunc::ctor(" << GetName() << ") ERROR: mref " << mref->GetName()
                               << " is not of type RooAbsReal" << std::endl;
         throw string("RooPolyMorh::ctor() ERROR mref is not of type RooAbsReal");
      }
      if (!dynamic_cast<RooConstVar *>(mref)) {
         coutW(InputArguments) << "RooMomentMorphFunc::ctor(" << GetName() << ") WARNING mref point " << i
                               << " is not a constant, taking a snapshot of its value" << std::endl;
      }
      (*_mref)[i] = static_cast<RooAbsReal *>(mref)->getVal();
      ++i;
   }

   // initialization
   initialize();
}

//_____________________________________________________________________________
RooMomentMorphFunc::RooMomentMorphFunc(const RooMomentMorphFunc &other, const char *name)
   : RooAbsReal(other, name),
     _cacheMgr(other._cacheMgr, this),
     m("m", this, other.m),
     _varList("varList", this, other._varList),
     _pdfList("pdfList", this, other._pdfList),
     _mref(new TVectorD(*other._mref)),
     _setting(other._setting),
     _useHorizMorph(other._useHorizMorph)
{

   // initialization
   initialize();
}

//_____________________________________________________________________________
RooMomentMorphFunc::~RooMomentMorphFunc()
{
   if (_mref)
      delete _mref;
   if (_M)
      delete _M;
}

//_____________________________________________________________________________
void RooMomentMorphFunc::initialize()
{

   Int_t nPdf = _pdfList.size();

   // other quantities needed
   if (nPdf != _mref->GetNrows()) {
      coutE(InputArguments) << "RooMomentMorphFunc::initialize(" << GetName() << ") ERROR: nPdf != nRefPoints" << std::endl;
      assert(0);
   }

   TVectorD *dm = new TVectorD(nPdf);
   _M = new TMatrixD(nPdf, nPdf);

   // transformation matrix for non-linear extrapolation, needed in evaluate()
   TMatrixD M(nPdf, nPdf);
   for (Int_t i = 0; i < _mref->GetNrows(); ++i) {
      (*dm)[i] = (*_mref)[i] - (*_mref)[0];
      M(i, 0) = 1.;
      if (i > 0)
         M(0, i) = 0.;
   }
   for (Int_t i = 1; i < _mref->GetNrows(); ++i) {
      for (Int_t j = 1; j < _mref->GetNrows(); ++j) {
         M(i, j) = std::pow((*dm)[i], (double)j);
      }
   }
   (*_M) = M.Invert();

   delete dm;
}

//_____________________________________________________________________________
RooMomentMorphFunc::CacheElem *RooMomentMorphFunc::getCache(const RooArgSet * /*nset*/) const
{
   auto cache = static_cast<CacheElem *>(_cacheMgr.getObj(nullptr, static_cast<RooArgSet const*>(nullptr)));
   if (cache) {
      return cache;
   }
   Int_t nVar = _varList.size();
   Int_t nPdf = _pdfList.size();

   RooAbsReal *null = nullptr;
   vector<RooAbsReal *> meanrv(nPdf * nVar, null);
   vector<RooAbsReal *> sigmarv(nPdf * nVar, null);
   vector<RooAbsReal *> myrms(nVar, null);
   vector<RooAbsReal *> mypos(nVar, null);
   vector<RooAbsReal *> slope(nPdf * nVar, null);
   vector<RooAbsReal *> offs(nPdf * nVar, null);
   vector<RooAbsReal *> transVar(nPdf * nVar, null);
   vector<RooAbsReal *> transPdf(nPdf, null);

   RooArgSet ownedComps;

   RooArgList fracl;

   // fraction parameters
   RooArgList coefList("coefList");
   RooArgList coefList2("coefList2");
   for (Int_t i = 0; i < 2 * nPdf; ++i) {
      std::string fracName = Form("frac_%d", i);

      RooRealVar *frac = new RooRealVar(fracName.c_str(), fracName.c_str(), 1.);

      fracl.add(*frac); // to be set later
      if (i < nPdf) {
         coefList.add(*static_cast<RooRealVar *>(fracl.at(i)));
      } else {
         coefList2.add(*static_cast<RooRealVar *>(fracl.at(i)));
      }
      ownedComps.add(*static_cast<RooRealVar *>(fracl.at(i)));
   }

   RooRealSumFunc *theSumFunc = nullptr;
   std::string sumfuncName = Form("%s_sumfunc", GetName());

   if (_useHorizMorph) {
      // mean and sigma
      RooArgList varList(_varList);
      for (Int_t i = 0; i < nPdf; ++i) {
         for (Int_t j = 0; j < nVar; ++j) {

            std::string meanName = Form("%s_mean_%d_%d", GetName(), i, j);
            std::string sigmaName = Form("%s_sigma_%d_%d", GetName(), i, j);

            RooAbsMoment *mom = nVar == 1 ? (static_cast<RooAbsPdf *>(_pdfList.at(i)))->sigma(static_cast<RooRealVar &>(*varList.at(j)))
                                          : (static_cast<RooAbsPdf *>(_pdfList.at(i)))->sigma(static_cast<RooRealVar &>(*varList.at(j)), varList);

            mom->setLocalNoDirtyInhibit(true);
            mom->mean()->setLocalNoDirtyInhibit(true);

            sigmarv[ij(i, j)] = mom;
            meanrv[ij(i, j)] = mom->mean();

            ownedComps.add(*sigmarv[ij(i, j)]);
         }
      }

      // slope and offset (to be set later, depend on m)
      for (Int_t j = 0; j < nVar; ++j) {
         RooArgList meanList("meanList");
         RooArgList rmsList("rmsList");
         for (Int_t i = 0; i < nPdf; ++i) {
            meanList.add(*meanrv[ij(i, j)]);
            rmsList.add(*sigmarv[ij(i, j)]);
         }
         std::string myrmsName = Form("%s_rms_%d", GetName(), j);
         std::string myposName = Form("%s_pos_%d", GetName(), j);
         myrms[j] = new RooAddition(myrmsName.c_str(), myrmsName.c_str(), rmsList, coefList2);
         mypos[j] = new RooAddition(myposName.c_str(), myposName.c_str(), meanList, coefList2);
         ownedComps.add(RooArgSet(*myrms[j], *mypos[j]));
      }

      // construction of unit pdfs
      RooArgList transPdfList;

      for (Int_t i = 0; i < nPdf; ++i) {
         auto& pdf = static_cast<RooAbsPdf&>(_pdfList[i]);
         std::string pdfName = Form("pdf_%d", i);
         RooCustomizer cust(pdf, pdfName.c_str());

         for (Int_t j = 0; j < nVar; ++j) {
            // slope and offset formulas
            std::string slopeName = Form("%s_slope_%d_%d", GetName(), i, j);
            std::string offsetName = Form("%s_offset_%d_%d", GetName(), i, j);
            slope[ij(i, j)] = new RooFormulaVar(slopeName.c_str(), "@0/@1", RooArgList(*sigmarv[ij(i, j)], *myrms[j]));
            offs[ij(i, j)] = new RooFormulaVar(offsetName.c_str(), "@0-(@1*@2)",
                                               RooArgList(*meanrv[ij(i, j)], *mypos[j], *slope[ij(i, j)]));
            ownedComps.add(RooArgSet(*slope[ij(i, j)], *offs[ij(i, j)]));
            // linear transformations, so pdf can be renormalized
            auto& var = static_cast<RooRealVar&>(*_varList[j]);
            std::string transVarName = Form("%s_transVar_%d_%d", GetName(), i, j);
            // transVar[ij(i,j)] = new
            // RooFormulaVar(transVarName.c_str(),transVarName.c_str(),"@0*@1+@2",RooArgList(*var,*slope[ij(i,j)],*offs[ij(i,j)]));

            transVar[ij(i, j)] =
               new RooLinearVar(transVarName.c_str(), transVarName.c_str(), var, *slope[ij(i, j)], *offs[ij(i, j)]);

            // *** WVE this is important *** this declares that frac effectively depends on the morphing parameters
            // This will prevent the likelihood optimizers from erroneously declaring terms constant
            transVar[ij(i, j)]->addServer((RooAbsArg &)m.arg());

            ownedComps.add(*transVar[ij(i, j)]);
            cust.replaceArg(var, *transVar[ij(i, j)]);
         }
         transPdf[i] = static_cast<RooAbsPdf *>(cust.build());
         transPdfList.add(*transPdf[i]);
         ownedComps.add(*transPdf[i]);
      }
      // sum pdf
      theSumFunc = new RooRealSumFunc(sumfuncName.c_str(), sumfuncName.c_str(), transPdfList, coefList);
   } else {
      theSumFunc = new RooRealSumFunc(sumfuncName.c_str(), sumfuncName.c_str(), _pdfList, coefList);
   }

   // *** WVE this is important *** this declares that frac effectively depends on the morphing parameters
   // This will prevent the likelihood optimizers from erroneously declaring terms constant
   theSumFunc->addServer((RooAbsArg &)m.arg());
   theSumFunc->addOwnedComponents(ownedComps);

   // change tracker for fraction parameters
   std::string trackerName = Form("%s_frac_tracker", GetName());
   RooChangeTracker *tracker = new RooChangeTracker(trackerName.c_str(), trackerName.c_str(), m.arg(), true);

   // Store it in the cache
   cache = new CacheElem(*theSumFunc, *tracker, fracl);
   _cacheMgr.setObj(nullptr, nullptr, cache, nullptr);

   return cache;
}

//_____________________________________________________________________________
RooArgList RooMomentMorphFunc::CacheElem::containedArgs(Action)
{
   return RooArgList(*_sumFunc, *_tracker);
}

//_____________________________________________________________________________
RooMomentMorphFunc::CacheElem::~CacheElem()
{
   delete _sumFunc;
   delete _tracker;
}

//_____________________________________________________________________________
double RooMomentMorphFunc::getValV(const RooArgSet *set) const
{
   // Special version of getValV() overrides RooAbsReal::getVal() to save value of current normalization set
   _curNormSet = set ? const_cast<RooArgSet *>(set) : const_cast<RooArgSet *>(static_cast<RooArgSet const*>(&_varList));
   return RooAbsReal::getValV(set);
}

//_____________________________________________________________________________
RooAbsReal *RooMomentMorphFunc::sumFunc(const RooArgSet *nset)
{
   CacheElem *cache = getCache(nset ? nset : _curNormSet);

   if (cache->_tracker->hasChanged(true)) {
      cache->calculateFractions(*this, false); // verbose turned off
   }

   return cache->_sumFunc;
}

//_____________________________________________________________________________
const RooAbsReal *RooMomentMorphFunc::sumFunc(const RooArgSet *nset) const
{
   CacheElem *cache = getCache(nset ? nset : _curNormSet);

   if (cache->_tracker->hasChanged(true)) {
      cache->calculateFractions(*this, false); // verbose turned off
   }

   return cache->_sumFunc;
}

//_____________________________________________________________________________
double RooMomentMorphFunc::evaluate() const
{
   CacheElem *cache = getCache(_curNormSet);

   if (cache->_tracker->hasChanged(true)) {
      cache->calculateFractions(*this, false); // verbose turned off
   }

   double ret = cache->_sumFunc->getVal(_pdfList.nset());
   return ret;
}

//_____________________________________________________________________________
RooRealVar *RooMomentMorphFunc::CacheElem::frac(Int_t i)
{
   return static_cast<RooRealVar *>(_frac.at(i));
}

//_____________________________________________________________________________
const RooRealVar *RooMomentMorphFunc::CacheElem::frac(Int_t i) const
{
   return static_cast<RooRealVar *>(_frac.at(i));
}

//_____________________________________________________________________________
void RooMomentMorphFunc::CacheElem::calculateFractions(const RooMomentMorphFunc &self, bool verbose) const
{
   Int_t nPdf = self._pdfList.size();

   double dm = self.m - (*self._mref)[0];

   // fully non-linear
   double sumposfrac = 0.;
   for (Int_t i = 0; i < nPdf; ++i) {
      double ffrac = 0.;
      for (Int_t j = 0; j < nPdf; ++j) {
         ffrac += (*self._M)(j, i) * (j == 0 ? 1. : std::pow(dm, (double)j));
      }
      if (ffrac >= 0)
         sumposfrac += ffrac;
      // fractions for pdf
      const_cast<RooRealVar *>(frac(i))->setVal(ffrac);
      // fractions for rms and mean
      const_cast<RooRealVar *>(frac(nPdf + i))->setVal(ffrac);
      if (verbose) {
         std::cout << ffrac << std::endl;
      }
   }

   // various mode settings
   int imin = self.idxmin(self.m);
   int imax = self.idxmax(self.m);
   double mfrac = (self.m - (*self._mref)[imin]) / ((*self._mref)[imax] - (*self._mref)[imin]);
   switch (self._setting) {
   case NonLinear:
      // default already set above
      break;

   case SineLinear:
      mfrac =
         std::sin(TMath::PiOver2() * mfrac); // this gives a continuous differentiable transition between grid points.

   // now fall through to Linear case

   case Linear:
      for (Int_t i = 0; i < 2 * nPdf; ++i) const_cast<RooRealVar *>(frac(i))->setVal(0.);
      if (imax > imin) { // m in between mmin and mmax
         const_cast<RooRealVar *>(frac(imin))->setVal(1. - mfrac);
         const_cast<RooRealVar *>(frac(nPdf + imin))->setVal(1. - mfrac);
         const_cast<RooRealVar *>(frac(imax))->setVal(mfrac);
         const_cast<RooRealVar *>(frac(nPdf + imax))->setVal(mfrac);
      } else if (imax == imin) { // m outside mmin and mmax
         const_cast<RooRealVar *>(frac(imin))->setVal(1.);
         const_cast<RooRealVar *>(frac(nPdf + imin))->setVal(1.);
      }
      break;
   case NonLinearLinFractions:
      for (Int_t i = 0; i < nPdf; ++i) const_cast<RooRealVar *>(frac(i))->setVal(0.);
      if (imax > imin) { // m in between mmin and mmax
         const_cast<RooRealVar *>(frac(imin))->setVal(1. - mfrac);
         const_cast<RooRealVar *>(frac(imax))->setVal(mfrac);
      } else if (imax == imin) { // m outside mmin and mmax
         const_cast<RooRealVar *>(frac(imin))->setVal(1.);
      }
      break;
   case NonLinearPosFractions:
      for (Int_t i = 0; i < nPdf; ++i) {
         if (frac(i)->getVal() < 0)
            const_cast<RooRealVar *>(frac(i))->setVal(0.);
         const_cast<RooRealVar *>(frac(i))->setVal(frac(i)->getVal() / sumposfrac);
      }
      break;
   }
}

//_____________________________________________________________________________
int RooMomentMorphFunc::idxmin(const double &mval) const
{
   int imin(0);
   Int_t nPdf = _pdfList.size();
   double mmin = -DBL_MAX;
   for (Int_t i = 0; i < nPdf; ++i) {
      if ((*_mref)[i] > mmin && (*_mref)[i] <= mval) {
         mmin = (*_mref)[i];
         imin = i;
      }
   }
   return imin;
}

//_____________________________________________________________________________
int RooMomentMorphFunc::idxmax(const double &mval) const
{
   int imax(0);
   Int_t nPdf = _pdfList.size();
   double mmax = DBL_MAX;
   for (Int_t i = 0; i < nPdf; ++i) {
      if ((*_mref)[i] < mmax && (*_mref)[i] >= mval) {
         mmax = (*_mref)[i];
         imax = i;
      }
   }
   return imax;
}

//_____________________________________________________________________________
std::list<double> *RooMomentMorphFunc::plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   return sumFunc(nullptr)->plotSamplingHint(obs, xlo, xhi);
}

//_____________________________________________________________________________
std::list<double> *RooMomentMorphFunc::binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   return sumFunc(nullptr)->binBoundaries(obs, xlo, xhi);
}

//_____________________________________________________________________________
bool RooMomentMorphFunc::isBinnedDistribution(const RooArgSet &obs) const
{
   return sumFunc(nullptr)->isBinnedDistribution(obs);
}
