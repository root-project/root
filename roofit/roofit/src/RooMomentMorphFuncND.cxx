/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooMomentMorphFuncND
    \ingroup Roofit

**/

#include <RooMomentMorphFuncND.h>

#include <RooAbsCategory.h>
#include <RooAbsMoment.h>
#include <RooAddPdf.h>
#include <RooAddition.h>
#include <RooChangeTracker.h>
#include <RooConstVar.h>
#include <RooCustomizer.h>
#include <RooFormulaVar.h>
#include <RooLinearVar.h>
#include <RooMoment.h>
#include <RooNumIntConfig.h>
#include <RooRealSumFunc.h>
#include <RooRealVar.h>

#include "RooFit/Detail/Algorithms.h"

#include <Riostream.h>

#include <TMap.h>
#include <TMath.h>
#include <TVector.h>

#include <algorithm>
#include <map>

using std::string, std::vector;


//_____________________________________________________________________________
RooMomentMorphFuncND::RooMomentMorphFuncND() : _cacheMgr(this, 10, true, true), _setting(RooMomentMorphFuncND::Linear), _useHorizMorph(true)
{
   TRACE_CREATE;
}

//_____________________________________________________________________________
RooMomentMorphFuncND::RooMomentMorphFuncND(const char *name, const char *title, const RooArgList &parList, const RooArgList &obsList,
                       const Grid2 &referenceGrid, Setting setting)
   : RooMomentMorphFuncND::Base_t(name, title),
     _cacheMgr(this, 10, true, true),
     _parList("parList", "List of morph parameters", this),
     _obsList("obsList", "List of observables", this),
     _referenceGrid(referenceGrid),
     _pdfList("pdfList", "List of pdfs", this),
     _setting(setting),
     _useHorizMorph(true)
{
   // morph parameters
   _parList.addTyped<RooAbsReal>(parList);

   // observables
   _obsList.addTyped<RooAbsReal>(obsList);

   _pdfList.add(_referenceGrid._pdfList);

   // general initialization
   initialize();

   TRACE_CREATE;
}

//_____________________________________________________________________________
RooMomentMorphFuncND::RooMomentMorphFuncND(const char *name, const char *title, RooAbsReal &_m, const RooArgList &varList,
                       const RooArgList &pdfList, const TVectorD &mrefpoints, Setting setting)
   : RooMomentMorphFuncND::Base_t(name, title),
     _cacheMgr(this, 10, true, true),
     _parList("parList", "List of morph parameters", this),
     _obsList("obsList", "List of observables", this),
     _pdfList("pdfList", "List of pdfs", this),
     _setting(setting),
     _useHorizMorph(true)
{
   // make reference grid
   RooBinning grid(mrefpoints.GetNrows() - 1, mrefpoints.GetMatrixArray());
   _referenceGrid.addBinning(grid);

   for (int i = 0; i < mrefpoints.GetNrows(); ++i) {
      for (int j = 0; j < grid.numBoundaries(); ++j) {
         if (mrefpoints[i] == grid.array()[j]) {
            _referenceGrid.addPdf(*static_cast<Base_t *>(pdfList.at(i)), j);
            break;
         }
      }
   }

   _pdfList.add(_referenceGrid._pdfList);

   // morph parameters
   RooArgList parList;
   parList.add(_m);
   _parList.addTyped<RooAbsReal>(parList);

   // observables
   _obsList.addTyped<RooAbsReal>(varList);

   // general initialization
   initialize();

   TRACE_CREATE;
}

//_____________________________________________________________________________
RooMomentMorphFuncND::RooMomentMorphFuncND(const char *name, const char *title, RooAbsReal &_m, const RooArgList &varList,
                       const RooArgList &pdfList, const RooArgList &mrefList, Setting setting)
   : RooMomentMorphFuncND::Base_t(name, title),
     _cacheMgr(this, 10, true, true),
     _parList("parList", "List of morph parameters", this),
     _obsList("obsList", "List of observables", this),
     _pdfList("pdfList", "List of pdfs", this),
     _setting(setting),
     _useHorizMorph(true)
{
   // make reference grid
   TVectorD mrefpoints(mrefList.size());
   Int_t i = 0;
   for (auto *mref : mrefList) {
      if (!dynamic_cast<RooAbsReal *>(mref)) {
         coutE(InputArguments) << "RooMomentMorphFuncND::ctor(" << GetName() << ") ERROR: mref " << mref->GetName()
                               << " is not of type RooAbsReal" << std::endl;
         throw string("RooMomentMorphFuncND::ctor() ERROR mref is not of type RooAbsReal");
      }
      if (!dynamic_cast<RooConstVar *>(mref)) {
         coutW(InputArguments) << "RooMomentMorphFuncND::ctor(" << GetName() << ") WARNING mref point " << i
                               << " is not a constant, taking a snapshot of its value" << std::endl;
      }
      mrefpoints[i] = static_cast<RooAbsReal *>(mref)->getVal();
      i++;
   }

   RooBinning grid(mrefpoints.GetNrows() - 1, mrefpoints.GetMatrixArray());
   _referenceGrid.addBinning(grid);
   for (i = 0; i < mrefpoints.GetNrows(); ++i) {
      for (int j = 0; j < grid.numBoundaries(); ++j) {
         if (mrefpoints[i] == grid.array()[j]) {
            _referenceGrid.addPdf(static_cast<Base_t &>(pdfList[i]), j);
            break;
         }
      }
   }

   _pdfList.add(_referenceGrid._pdfList);

   // morph parameters
   RooArgList parList;
   parList.add(_m);
   _parList.addTyped<RooAbsReal>(parList);

   // observables
   _obsList.addTyped<RooAbsReal>(varList);

   // general initialization
   initialize();

   TRACE_CREATE;
}

//_____________________________________________________________________________
RooMomentMorphFuncND::RooMomentMorphFuncND(const RooMomentMorphFuncND &other, const char *name)
   : RooMomentMorphFuncND::Base_t(other, name),
     _cacheMgr(other._cacheMgr, this),
     _parList("parList", this, other._parList),
     _obsList("obsList", this, other._obsList),
     _referenceGrid(other._referenceGrid),
     _pdfList("pdfList", this, other._pdfList),
     _setting(other._setting),
     _useHorizMorph(other._useHorizMorph),
     _isPdfMode{other._isPdfMode}
{
   // general initialization
   initialize();

   TRACE_CREATE;
}

//_____________________________________________________________________________
RooMomentMorphFuncND::~RooMomentMorphFuncND()
{
   TRACE_DESTROY;
}

//_____________________________________________________________________________
void RooMomentMorphFuncND::initialize()
{
   for (vector<RooAbsBinning *>::iterator itr = _referenceGrid._grid.begin(); itr != _referenceGrid._grid.end();
        ++itr) {
      _referenceGrid._nnuis.push_back((*itr)->numBins() + 1);
   }

   int nPar = _parList.size();
   int nDim = _referenceGrid._grid.size();
   int nPdf = _referenceGrid._pdfList.size();
   int nRef = _referenceGrid._nref.size();
   int depth = std::pow(2, nPar);

   if (nPar != nDim) {
      coutE(InputArguments) << "RooMomentMorphFuncND::initialize(" << GetName() << ") ERROR: nPar != nDim"
                            << ": " << nPar << " !=" << nDim << std::endl;
      assert(0);
   }

   if (nPdf != nRef) {
      coutE(InputArguments) << "RooMomentMorphFuncND::initialize(" << GetName() << ") ERROR: nPdf != nRef"
                            << ": " << nPdf << " !=" << nRef << std::endl;
      assert(0);
   }

   // Transformation matrix for NonLinear settings
   _M = std::make_unique<TMatrixD>(nPdf, nPdf);
   _MSqr = std::make_unique<TMatrixD>(depth, depth);
   if (_setting == NonLinear || _setting == NonLinearPosFractions || _setting == NonLinearLinFractions) {
      TMatrixD M(nPdf, nPdf);

      vector<vector<double>> dm(nPdf);
      for (int k = 0; k < nPdf; ++k) {
         vector<double> dm2;
         for (int idim = 0; idim < nPar; idim++) {
            double delta = _referenceGrid._nref[k][idim] - _referenceGrid._nref[0][idim];
            dm2.push_back(delta);
         }
         dm[k] = dm2;
      }

      vector<vector<int>> powers;
      for (int idim = 0; idim < nPar; idim++) {
         vector<int> xtmp;
         xtmp.reserve(_referenceGrid._nnuis[idim]);
         for (int ix = 0; ix < _referenceGrid._nnuis[idim]; ix++) {
            xtmp.push_back(ix);
         }
         powers.push_back(xtmp);
      }

      vector<vector<int>> output;
      RooFit::Detail::cartesianProduct(output, powers);
      int nCombs = output.size();

      for (int k = 0; k < nPdf; ++k) {
         int nperm = 0;
         for (int i = 0; i < nCombs; i++) {
            double tmpDm = 1.0;
            for (int ix = 0; ix < nPar; ix++) {
               double delta = dm[k][ix];
               tmpDm *= std::pow(delta, static_cast<double>(output[i][ix]));
            }
            M(k, nperm) = tmpDm;
            nperm++;
         }
      }

      // M.Print();
      (*_M) = M.Invert();
   }

   // Resize transformation vectors
   _squareVec.resize(std::pow(2, nPar));
   _squareIdx.resize(std::pow(2, nPar));
}

//_____________________________________________________________________________
RooMomentMorphFuncND::Grid2::Grid2(const RooMomentMorphFuncND::Grid2 &other)
   : _pdfList(other._pdfList), _pdfMap(other._pdfMap), _nref(other._nref)
{
   for (unsigned int i = 0; i < other._grid.size(); i++) {
      _grid.push_back(other._grid[i]->clone());
   }
}

//_____________________________________________________________________________
RooMomentMorphFuncND::Grid2::~Grid2()
{
   for (RooAbsBinning *binning : _grid) {
      delete binning;
   }
}

//_____________________________________________________________________________
void RooMomentMorphFuncND::Grid2::addPdf(const RooMomentMorphFuncND::Base_t &pdf, int bin_x)
{
   vector<int> thisBoundaries;
   vector<double> thisBoundaryCoordinates;
   thisBoundaries.push_back(bin_x);
   thisBoundaryCoordinates.push_back(_grid[0]->array()[bin_x]);
   _pdfList.add(pdf);
   _pdfMap[thisBoundaries] = _pdfList.size() - 1;
   _nref.push_back(thisBoundaryCoordinates);
}

//_____________________________________________________________________________
void RooMomentMorphFuncND::Grid2::addPdf(const RooMomentMorphFuncND::Base_t &pdf, int bin_x, int bin_y)
{
   vector<int> thisBoundaries;
   vector<double> thisBoundaryCoordinates;
   thisBoundaries.push_back(bin_x);
   thisBoundaryCoordinates.push_back(_grid[0]->array()[bin_x]);
   thisBoundaries.push_back(bin_y);
   thisBoundaryCoordinates.push_back(_grid[1]->array()[bin_y]);
   _pdfList.add(pdf);
   _pdfMap[thisBoundaries] = _pdfList.size() - 1;
   _nref.push_back(thisBoundaryCoordinates);
}

//_____________________________________________________________________________
void RooMomentMorphFuncND::Grid2::addPdf(const RooMomentMorphFuncND::Base_t &pdf, int bin_x, int bin_y, int bin_z)
{
   vector<int> thisBoundaries;
   vector<double> thisBoundaryCoordinates;
   thisBoundaries.push_back(bin_x);
   thisBoundaryCoordinates.push_back(_grid[0]->array()[bin_x]);
   thisBoundaries.push_back(bin_y);
   thisBoundaryCoordinates.push_back(_grid[1]->array()[bin_y]);
   thisBoundaries.push_back(bin_z);
   thisBoundaryCoordinates.push_back(_grid[2]->array()[bin_z]);
   _pdfList.add(pdf);
   _pdfMap[thisBoundaries] = _pdfList.size() - 1;
   _nref.push_back(thisBoundaryCoordinates);
}

//_____________________________________________________________________________
void RooMomentMorphFuncND::Grid2::addPdf(const RooMomentMorphFuncND::Base_t &pdf, vector<int> bins)
{
   vector<double> thisBoundaryCoordinates;
   int nBins = bins.size();
   thisBoundaryCoordinates.reserve(nBins);
   for (int i = 0; i < nBins; i++) {
      thisBoundaryCoordinates.push_back(_grid[i]->array()[bins[i]]);
   }
   _pdfList.add(pdf);
   _pdfMap[bins] = _pdfList.size() - 1;
   _nref.push_back(thisBoundaryCoordinates);
}

//_____________________________________________________________________________
RooMomentMorphFuncND::CacheElem *RooMomentMorphFuncND::getCache(const RooArgSet * /*nset*/) const
{
   auto cache = static_cast<CacheElem *>(_cacheMgr.getObj(nullptr, static_cast<RooArgSet const*>(nullptr)));
   if (cache) {
      return cache;
   }

   int nObs = _obsList.size();
   int nPdf = _referenceGrid._pdfList.size();

   RooAbsReal *null = nullptr;
   vector<RooAbsReal *> meanrv(nPdf * nObs, null);
   vector<RooAbsReal *> sigmarv(nPdf * nObs, null);
   vector<RooAbsReal *> myrms(nObs, null);
   vector<RooAbsReal *> mypos(nObs, null);
   vector<RooAbsReal *> slope(nPdf * nObs, null);
   vector<RooAbsReal *> offsets(nPdf * nObs, null);
   vector<RooAbsReal *> transVar(nPdf * nObs, null);
   vector<RooAbsReal *> transPdf(nPdf, null);

   RooArgSet ownedComps;
   RooArgList fracl;

   // fraction parameters
   RooArgList coefList("coefList");   // fractions multiplied with input pdfs
   RooArgList coefList2("coefList2"); // fractions multiplied with mean position of observable contribution
   RooArgList coefList3("coefList3"); // fractions multiplied with rms position of observable contribution

   for (int i = 0; i < 3 * nPdf; ++i) {
      string fracName = Form("frac_%d", i);
      double initval = _isPdfMode ? 1.0 : 0.0;
      RooRealVar *frac = new RooRealVar(fracName.c_str(), fracName.c_str(), initval); // to be set later

      fracl.add(*frac);
      if (i < nPdf) {
         coefList.add(*static_cast<RooRealVar *>(fracl.at(i)));
      } else if (i < 2 * nPdf) {
         coefList2.add(*static_cast<RooRealVar *>(fracl.at(i)));
      } else {
         coefList3.add(*static_cast<RooRealVar *>(fracl.at(i)));
      }
      ownedComps.add(*static_cast<RooRealVar *>(fracl.at(i)));
   }

   std::unique_ptr<RooAbsReal> theSum;
   string sumName = Form("%s_sum", GetName());

   RooArgList transPdfList;
   if (_useHorizMorph) {
      // mean and sigma
      RooArgList obsList(_obsList);
      for (int i = 0; i < nPdf; ++i) {
         for (int j = 0; j < nObs; ++j) {
            RooAbsMoment *mom = nObs == 1 ? (static_cast<Base_t *>(_pdfList.at(i)))->sigma(static_cast<RooRealVar &>(*obsList.at(j)))
                                          : (static_cast<Base_t *>(_pdfList.at(i)))->sigma(static_cast<RooRealVar &>(*obsList.at(j)), obsList);

            mom->setLocalNoDirtyInhibit(true);
            mom->mean()->setLocalNoDirtyInhibit(true);

            sigmarv[sij(i, j)] = mom;
            meanrv[sij(i, j)] = mom->mean();

            ownedComps.add(*sigmarv[sij(i, j)]);
         }
      }

      // slope and offset (to be set later, depend on nuisance parameters)
      for (int j = 0; j < nObs; ++j) {
         RooArgList meanList("meanList");
         RooArgList rmsList("rmsList");
         for (int i = 0; i < nPdf; ++i) {
            meanList.add(*meanrv[sij(i, j)]);
            rmsList.add(*sigmarv[sij(i, j)]);
         }
         string myrmsName = Form("%s_rms_%d", GetName(), j);
         string myposName = Form("%s_pos_%d", GetName(), j);
         mypos[j] = new RooAddition(myposName.c_str(), myposName.c_str(), meanList, coefList2);
         myrms[j] = new RooAddition(myrmsName.c_str(), myrmsName.c_str(), rmsList, coefList3);
         ownedComps.add(RooArgSet(*myrms[j], *mypos[j]));
      }

      // construction of unit pdfs

      Int_t i = 0;
      for (auto const *pdf : static_range_cast<Base_t *>(_pdfList)) {

         string pdfName = Form("pdf_%d", i);
         RooCustomizer cust(*pdf, pdfName.c_str());

         Int_t j = 0;
         for (auto *var : static_range_cast<RooRealVar *>(obsList)) {
            // slope and offset formulas
            string slopeName = Form("%s_slope_%d_%d", GetName(), i, j);
            string offsetName = Form("%s_offset_%d_%d", GetName(), i, j);

            slope[sij(i, j)] =
               new RooFormulaVar(slopeName.c_str(), "@0/@1", RooArgList(*sigmarv[sij(i, j)], *myrms[j]));
            offsets[sij(i, j)] = new RooFormulaVar(offsetName.c_str(), "@0-(@1*@2)",
                                                   RooArgList(*meanrv[sij(i, j)], *mypos[j], *slope[sij(i, j)]));
            ownedComps.add(RooArgSet(*slope[sij(i, j)], *offsets[sij(i, j)]));

            // linear transformations, so pdf can be renormalized easily
            string transVarName = Form("%s_transVar_%d_%d", GetName(), i, j);
            transVar[sij(i, j)] = new RooLinearVar(transVarName.c_str(), transVarName.c_str(), *var, *slope[sij(i, j)],
                                                   *offsets[sij(i, j)]);

            // *** WVE this is important *** this declares that frac effectively depends on the morphing parameters
            // This will prevent the likelihood optimizers from erroneously declaring terms constant
            transVar[sij(i, j)]->addServerList((RooAbsCollection &)_parList);

            ownedComps.add(*transVar[sij(i, j)]);
            cust.replaceArg(*var, *transVar[sij(i, j)]);
            ++j;
         }
         transPdf[i] = static_cast<Base_t *>(cust.build());
         transPdfList.add(*transPdf[i]);
         ownedComps.add(*transPdf[i]);
         ++i;
      }
   }

   // sum pdf
   RooArgList const &pdfList = _useHorizMorph ? transPdfList : static_cast<RooArgList const &>(_pdfList);
   if (_isPdfMode) {
      theSum = std::make_unique<RooAddPdf>(sumName.c_str(), sumName.c_str(), pdfList, coefList);
   } else {
      theSum = std::make_unique<RooRealSumFunc>(sumName.c_str(), sumName.c_str(), pdfList, coefList);
   }

   // *** WVE this is important *** this declares that frac effectively depends on the morphing parameters
   // This will prevent the likelihood optimizers from erroneously declaring terms constant
   theSum->addServerList((RooAbsCollection &)_parList);
   theSum->addOwnedComponents(ownedComps);

   // change tracker for fraction parameters
   std::string trackerName = std::string(GetName()) + "_frac_tracker";

   // Store it in the cache
   cache = new CacheElem(std::move(theSum),
                         std::make_unique<RooChangeTracker>(trackerName.c_str(), trackerName.c_str(), _parList, true),
                         fracl);
   _cacheMgr.setObj(nullptr, nullptr, cache, nullptr);

   return cache;
}

RooMomentMorphFuncND::CacheElem::CacheElem(std::unique_ptr<RooAbsReal> &&sumFunc,
                                           std::unique_ptr<RooChangeTracker> &&tracker, const RooArgList &flist)
   : _sum(std::move(sumFunc)), _tracker(std::move(tracker))
{
   _frac.add(flist);
}

//_____________________________________________________________________________
RooArgList RooMomentMorphFuncND::CacheElem::containedArgs(Action)
{
   return RooArgList(*_sum, *_tracker);
}

//_____________________________________________________________________________
RooMomentMorphFuncND::CacheElem::~CacheElem() = default;

//_____________________________________________________________________________
double RooMomentMorphFuncND::getValV(const RooArgSet *set) const
{
   // Special version of getValV() overrides Base_t::getValV() to save value of current normalization set
   _curNormSet = set ? const_cast<RooArgSet *>(set) : const_cast<RooArgSet *>(static_cast<RooArgSet const*>(&_obsList));
   return Base_t::getValV(set);
}

//_____________________________________________________________________________
RooMomentMorphFuncND::Base_t *RooMomentMorphFuncND::sumFunc(const RooArgSet *nset)
{
   CacheElem *cache = getCache(nset ? nset : _curNormSet);

   if (cache->_tracker->hasChanged(true)) {
      cache->calculateFractions(*this, false); // verbose turned off
   }
   return cache->_sum.get();
}

//_____________________________________________________________________________
double RooMomentMorphFuncND::evaluate() const
{
   CacheElem *cache = getCache(_curNormSet);

   if (cache->_tracker->hasChanged(true)) {
      cache->calculateFractions(*this, false); // verbose turned off
   }

   double ret = cache->_sum->getVal(_obsList.nset());

   return ret;
}

//_____________________________________________________________________________
RooRealVar *RooMomentMorphFuncND::CacheElem::frac(int i)
{
   return static_cast<RooRealVar *>(_frac.at(i));
}

//_____________________________________________________________________________
const RooRealVar *RooMomentMorphFuncND::CacheElem::frac(int i) const
{
   return static_cast<RooRealVar *>(_frac.at(i));
}

//_____________________________________________________________________________
void RooMomentMorphFuncND::CacheElem::calculateFractions(const RooMomentMorphFuncND &self, bool verbose) const
{
   int nPdf = self._pdfList.size();
   int nPar = self._parList.size();

   double fracLinear(1.);
   double fracNonLinear(1.);

   if (self._setting == NonLinear || self._setting == NonLinearLinFractions || self._setting == NonLinearPosFractions) {
      // Calculate the delta vector
      vector<double> dm2;
      for (int idim = 0; idim < nPar; idim++) {
         double delta = (static_cast<RooRealVar *>(self._parList.at(idim)))->getVal() - self._referenceGrid._nref[0][idim];
         dm2.push_back(delta);
      }

      vector<vector<int>> powers;
      for (int idim = 0; idim < nPar; idim++) {
         vector<int> xtmp;
         xtmp.reserve(self._referenceGrid._nnuis[idim]);
         for (int ix = 0; ix < self._referenceGrid._nnuis[idim]; ix++) {
            xtmp.push_back(ix);
         }
         powers.push_back(xtmp);
      }

      vector<vector<int>> output;
      RooFit::Detail::cartesianProduct(output, powers);
      int nCombs = output.size();

      vector<double> deltavec(nPdf, 1.0);

      int nperm = 0;
      for (int i = 0; i < nCombs; i++) {
         double tmpDm = 1.0;
         for (int ix = 0; ix < nPar; ix++) {
            double delta = dm2[ix];
            tmpDm *= std::pow(delta, static_cast<double>(output[i][ix]));
         }
         deltavec[nperm] = tmpDm;
         nperm++;
      }

      double sumposfrac = 0.0;
      for (int i = 0; i < nPdf; ++i) {
         double ffrac = 0.0;

         for (int j = 0; j < nPdf; ++j) {
            ffrac += (*self._M)(j, i) * deltavec[j] * fracNonLinear;
         }

         if (ffrac >= 0) {
            sumposfrac += ffrac;
         }

         // fractions for pdf
         if (self._setting != NonLinearLinFractions) {
            const_cast<RooRealVar *>(frac(i))->setVal(ffrac);
         }

         // fractions for rms and mean
         const_cast<RooRealVar *>(frac(nPdf + i))->setVal(ffrac);     // need to add up
         const_cast<RooRealVar *>(frac(2 * nPdf + i))->setVal(ffrac); // need to add up

         if (verbose) {
            std::cout << "NonLinear fraction " << ffrac << std::endl;
            frac(i)->Print();
            frac(nPdf + i)->Print();
            frac(2 * nPdf + i)->Print();
         }
      }

      if (self._setting == NonLinearPosFractions) {
         for (int i = 0; i < nPdf; ++i) {
            if (frac(i)->getVal() < 0)
               const_cast<RooRealVar *>(frac(i))->setVal(0.);
            const_cast<RooRealVar *>(frac(i))->setVal(frac(i)->getVal() / sumposfrac);
         }
      }
   }

   if (self._setting == Linear || self._setting == NonLinearLinFractions) {
      // zero all fractions
      // for (int i = 0; i < 3*nPdf; ++i) {
      for (int i = 0; i < nPdf; ++i) {
         double initval = 0;
         const_cast<RooRealVar *>(frac(i))->setVal(initval);
         const_cast<RooRealVar *>(frac(nPdf + i))->setVal(initval);
         const_cast<RooRealVar *>(frac(2 * nPdf + i))->setVal(initval);
      }

      std::vector<double> mtmp;

      // loop over parList
      for (auto *m : static_range_cast<RooRealVar *>(self._parList)) {
         mtmp.push_back(m->getVal());
      }

      self.findShape(mtmp); // this sets _squareVec and _squareIdx quantities

      int depth = std::pow(2, nPar);
      vector<double> deltavec(depth, 1.0);

      int nperm = 0;

      vector<int> xtmp;
      xtmp.reserve(nPar);
      for (int ix = 0; ix < nPar; ix++) {
         xtmp.push_back(ix);
      }

      for (int iperm = 1; iperm <= nPar; ++iperm) {
         do {
            double dtmp = mtmp[xtmp[0]] - self._squareVec[0][xtmp[0]];
            for (int itmp = 1; itmp < iperm; ++itmp) {
               dtmp *= mtmp[xtmp[itmp]] - self._squareVec[0][xtmp[itmp]];
            }
            deltavec[nperm + 1] = dtmp;
            nperm++;
         } while (RooFit::Detail::nextCombination(xtmp.begin(), xtmp.begin() + iperm, xtmp.end()));
      }

      double origFrac1(0.);
      double origFrac2(0.);
      for (int i = 0; i < depth; ++i) {
         double ffrac = 0.;
         for (int j = 0; j < depth; ++j) {
            ffrac += (*self._MSqr)(j, i) * deltavec[j] * fracLinear;
         }

         // set fractions for pdf
         origFrac1 = frac(self._squareIdx[i])->getVal();      // already set in case of smoothlinear
         const_cast<RooRealVar *>(frac(self._squareIdx[i]))->setVal(origFrac1 + ffrac); // need to add up

         // set fractions for rms and mean
         if (self._setting != NonLinearLinFractions) {
            origFrac2 =
               frac(nPdf + self._squareIdx[i])->getVal(); // already set in case of smoothlinear
            const_cast<RooRealVar *>(frac(nPdf + self._squareIdx[i]))->setVal(origFrac2 + ffrac);     // need to add up
            const_cast<RooRealVar *>(frac(2 * nPdf + self._squareIdx[i]))->setVal(origFrac2 + ffrac); // need to add up
         }

         if (verbose) {
            std::cout << "Linear fraction " << ffrac << std::endl;
            frac(self._squareIdx[i])->Print();
            frac(nPdf + self._squareIdx[i])->Print();
            frac(2 * nPdf + self._squareIdx[i])->Print();
         }
      }
   }
}

//_____________________________________________________________________________
void RooMomentMorphFuncND::findShape(const vector<double> &x) const
{
   int nPar = _parList.size();
   int nRef = _referenceGrid._nref.size();

   // Find hypercube enclosing the location to morph to
   // bool isEnclosed = true;
   // for (int i = 0; i < nPar; i++) {
   //    if (x[i] < _referenceGrid._grid[i]->lowBound())
   //       isEnclosed = false;
   //    if (x[i] > _referenceGrid._grid[i]->highBound())
   //       isEnclosed = false;
   // }

   // std::cout << "isEnclosed = " << isEnclosed << std::endl;

   int depth = std::pow(2, nPar);

   vector<vector<double>> boundaries(nPar);
   for (int idim = 0; idim < nPar; idim++) {
      int bin = _referenceGrid._grid[idim]->binNumber(x[idim]);
      double lo = _referenceGrid._grid[idim]->binLow(bin);
      double hi = _referenceGrid._grid[idim]->binHigh(bin);
      boundaries[idim].push_back(lo);
      boundaries[idim].push_back(hi);
   }

   vector<vector<double>> output;
   RooFit::Detail::cartesianProduct(output, boundaries);
   _squareVec = output;

   for (int isq = 0; isq < depth; isq++) {
      for (int iref = 0; iref < nRef; iref++) {
         if (_squareVec[isq] == _referenceGrid._nref[iref]) {
            _squareIdx[isq] = iref;
            break;
         }
      }
   }

   // std::cout << std::endl;

   // for (int isq = 0; isq < _squareVec.size(); isq++) {
   //   std::cout << _squareIdx[isq];
   //   std::cout << " (";
   //   for (int isqq = 0; isqq < _squareVec[isq].size(); isqq++) {
   //     std::cout << _squareVec[isq][isqq] << ((isqq<_squareVec[isq].size()-1)?",":"");
   //   }
   //   std::cout << ") ";
   // }

   // construct transformation matrix for linear extrapolation
   TMatrixD M(depth, depth);

   vector<int> xtmp;
   xtmp.reserve(nPar);
   for (int ix = 0; ix < nPar; ix++) {
      xtmp.push_back(ix);
   }

   for (int k = 0; k < depth; ++k) {
      M(k, 0) = 1.0;

      int nperm = 0;
      vector<double> squareBase = _squareVec[0];

      for (int iperm = 1; iperm <= nPar; ++iperm) {
         do {
            double dtmp = _squareVec[k][xtmp[0]] - squareBase[xtmp[0]];
            for (int itmp = 1; itmp < iperm; ++itmp) {
               dtmp *= _squareVec[k][xtmp[itmp]] - squareBase[xtmp[itmp]];
            }
            M(k, nperm + 1) = dtmp;
            nperm++;
         } while (RooFit::Detail::nextCombination(xtmp.begin(), xtmp.begin() + iperm, xtmp.end()));
      }
   }

   // M.Print();
   (*_MSqr) = M.Invert();
}

//_____________________________________________________________________________
bool RooMomentMorphFuncND::setBinIntegrator(RooArgSet &allVars)
{
   if (allVars.size() == 1) {
      RooAbsReal *temp = const_cast<RooMomentMorphFuncND *>(this);
      temp->specialIntegratorConfig(true)->method1D().setLabel("RooBinIntegrator");
      int nbins = (static_cast<RooRealVar *>(allVars.first()))->numBins();
      temp->specialIntegratorConfig(true)->getConfigSection("RooBinIntegrator").setRealValue("numBins", nbins);
      return true;
   } else {
      std::cout << "Currently BinIntegrator only knows how to deal with 1-d " << std::endl;
      return false;
   }
   return false;
}
