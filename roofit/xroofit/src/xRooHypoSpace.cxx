/*
 * Project: xRooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "xRooFit/xRooHypoSpace.h"

#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooFitResult.h"
#include "RooConstVar.h"
#include "RooAbsPdf.h"

#include "TCanvas.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TPRegexp.h"
#include "TMemFile.h"
#include "TROOT.h"
#include "RooDataSet.h"
#include "TKey.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLine.h"
#include "RooStats/HypoTestInverterResult.h"
#include "TEnv.h"

BEGIN_XROOFIT_NAMESPACE

xRooNLLVar::xRooHypoSpace::xRooHypoSpace(const char *name, const char *title)
   : TNamed(name, title), fPars(std::make_shared<RooArgSet>())
{
   if (name == nullptr || strlen(name) == 0) {
      SetName(TUUID().AsString());
   }
}

xRooNLLVar::xRooHypoSpace::xRooHypoSpace(const RooStats::HypoTestInverterResult *result)
   : fPars(std::make_shared<RooArgSet>())
{
   if (!result)
      return;

   SetNameTitle(result->GetName(), result->GetTitle());

   fPars->addClone(*std::unique_ptr<RooAbsCollection>(result->GetParameters()));
   double spaceSize = 1;
   for (auto p : *fPars) {
      auto v = dynamic_cast<RooRealVar *>(p);
      if (!v)
         continue;
      spaceSize *= (v->getMax() - v->getMin());
   }
   for (int i = 0; i < result->ArraySize(); i++) {
      auto point = result->GetResult(i);
      double xVal = result->GetXValue(i);
      double ssize = spaceSize;
      for (auto p : *fPars) {
         auto v = dynamic_cast<RooRealVar *>(p);
         if (!v)
            continue;
         ssize /= (v->getMax() - v->getMin());
         double remain = std::fmod(xVal, ssize);
         v->setVal((xVal - remain) / ssize);
         xVal = remain;
      }
      emplace_back(xRooHypoPoint(std::make_shared<RooStats::HypoTestResult>(*point), fPars.get()));
   }
   // add any pars we might have missed
   for (auto &p : *this) {
      for (auto a : *p.coords) {
         if (!fPars->find(a->GetName()))
            fPars->addClone(*a);
      }
   }
}

std::shared_ptr<xRooNode> xRooNLLVar::xRooHypoSpace::pdf(const char *parValues) const
{
   return pdf(toArgs(parValues));
}

std::shared_ptr<xRooNode> xRooNLLVar::xRooHypoSpace::pdf(const RooAbsCollection &parValues) const
{
   RooArgList rhs;
   rhs.add(parValues);
   rhs.sort();

   std::shared_ptr<xRooNode> out = nullptr;

   for (auto &[_range, _pdf] : fPdfs) {
      // any pars not in rhs are assumed to have infinite range in rhs
      // and vice versa
      bool collision = true;
      for (auto &_lhs : *_range) {
         auto _rhs = rhs.find(*_lhs);
         if (!_rhs)
            continue;
         if (auto v = dynamic_cast<RooRealVar *>(_rhs); v) {
            if (auto v2 = dynamic_cast<RooRealVar *>(_lhs)) {
               if (!(v->getMin() <= v2->getMax() && v2->getMin() <= v->getMax())) {
                  collision = false;
                  break;
               }
            } else if (auto c2 = dynamic_cast<RooConstVar *>(_lhs)) {
               if (!(v->getMin() <= c2->getVal() && c2->getVal() <= v->getMax())) {
                  collision = false;
                  break;
               }
            }
         } else if (auto c = dynamic_cast<RooConstVar *>(_rhs); c) {
            if (auto v2 = dynamic_cast<RooRealVar *>(_lhs)) {
               if (!(c->getVal() <= v2->getMax() && v2->getMin() <= c->getVal())) {
                  collision = false;
                  break;
               }
            } else if (auto c2 = dynamic_cast<RooConstVar *>(_lhs)) {
               if (!(c->getVal() == c2->getVal())) {
                  collision = false;
                  break;
               }
            }
         }
      }
      if (collision) {
         if (out) {
            throw std::runtime_error("Multiple pdf possibilities");
         }
         out = _pdf;
      }
   }

   return out;
}

RooArgList xRooNLLVar::xRooHypoSpace::toArgs(const char *str)
{

   RooArgList out;

   TStringToken pattern(str, ";");
   while (pattern.NextToken()) {
      TString s = pattern;
      // split by "=" sign
      auto _idx = s.Index('=');
      if (_idx == -1)
         continue;
      TString _name = s(0, _idx);
      TString _val = s(_idx + 1, s.Length());

      if (_val.IsFloat()) {
         out.addClone(RooConstVar(_name, _name, _val.Atof()));
      } else if (_val.BeginsWith('[')) {
         _idx = _val.Index(',');
         if (_idx == -1)
            continue;
         TString _min = _val(0, _idx);
         TString _max = _val(_idx + 1, _val.Length() - _idx - 2);
         out.addClone(RooRealVar(_name, _name, _min.Atof(), _max.Atof()));
      }
   }

   return out;
}

int xRooNLLVar::xRooHypoSpace::AddPoints(const char *parName, size_t nPoints, double low, double high)
{
   if (nPoints == 0)
      return nPoints;

   auto _par = dynamic_cast<RooAbsRealLValue *>(fPars->find(parName));
   if (!_par)
      throw std::runtime_error("Unknown parameter");
   _par->setAttribute("axis");

   if (low < _par->getMin()) {
      Warning("AddPoints", "low edge of hypoSpace %g below lower bound of parameter: %g. Changing to lower bound", low,
              _par->getMin());
      low = _par->getMin();
   }
   if (high > _par->getMax()) {
      Warning("AddPoints", "high edge of hypoSpace %g above upper bound of parameter: %g. Changing to upper bound",
              high, _par->getMax());
      high = _par->getMax();
   }

   if (nPoints == 1) {
      _par->setVal((high + low) * 0.5);
      AddPoint();
      return nPoints;
   }

   double step = (high - low) / (nPoints - 1);
   if (step <= 0)
      throw std::runtime_error("Invalid steps");

   for (size_t i = 0; i < nPoints; i++) {
      _par->setVal((i == nPoints - 1) ? high : (low + step * i));
      AddPoint();
   }
   return nPoints;
}

xRooNLLVar::xRooHypoPoint &xRooNLLVar::xRooHypoSpace::AddPoint(double value)
{
   if (axes().empty()) {
      // set the first poi as the axis variable to scan
      if (poi().empty()) {
         throw std::runtime_error("No POI to scan");
      } else {
         poi().first()->setAttribute("axis");
      }
   }

   if (empty()) {
      // promote all axes to being poi and demote all non-axes to non-poi
      poi().setAttribAll("poi", false);
      axes().setAttribAll("poi");
   }

   return AddPoint(TString::Format("%s=%f", axes().first()->GetName(), value));
}

int xRooNLLVar::xRooHypoSpace::scan(const char *type, size_t nPoints, double low, double high,
                                    const std::vector<double> &nSigmas, double relUncert)
{

   TString sType(type);
   sType.ToLower();
   if (sType.Contains("cls") && !sType.Contains("pcls"))
      sType.ReplaceAll("cls", "pcls");
   if (!sType.Contains("pcls") && !sType.Contains("ts") && !sType.Contains("pnull") && !sType.Contains("plr")) {
      throw std::runtime_error("scan type must be equal to one of: plr, cls, ts, pnull");
   }

   // will scan the first axes variable ... if there is none, specify the first poi as the axis var
   if (axes().empty()) {
      // set the first poi as the axis variable to scan
      if (poi().empty()) {
         throw std::runtime_error("No POI to scan");
      } else {
         poi().first()->setAttribute("axis");
      }
   }

   // promote all axes to being poi and demote all non-axes to non-poi
   poi().setAttribAll("poi", false);
   axes().setAttribAll("poi");

   auto p = dynamic_cast<RooRealVar *>(axes().first());
   if (!p) {
      throw std::runtime_error(TString::Format("%s not scannable", axes().first()->GetName()));
   }

   if (sType.Contains("cls")) {
      if (empty() && relUncert == std::numeric_limits<double>::infinity()) {
         // use default uncertainty precision of 10%
         ::Info("xRooHypoSpace::scan", "Using default precision of 10%% for auto-scan");
         relUncert = 0.1;
      }
      for (auto a : axes()) {
         if (!a->hasRange("physical")) {
            ::Info("xRooHypoSpace::scan", "No physical range set for %s, setting to [0,inf]", p->GetName());
            dynamic_cast<RooRealVar *>(a)->setRange("physical", 0, std::numeric_limits<double>::infinity());
         }
         if (!a->getStringAttribute("altVal") || !strlen(p->getStringAttribute("altVal"))) {
            ::Info("xRooHypoSpace::scan", "No altVal set for %s, setting to 0", a->GetName());
            a->setStringAttribute("altVal", "0");
         }
         // ensure range straddles altVal
         double altVal = TString(a->getStringAttribute("altVal")).Atof();
         auto v = dynamic_cast<RooRealVar *>(a);
         if (v->getMin() >= altVal) {
            ::Info("xRooHypoSpace::scan", "range of POI does not straddle alt value, adjusting minimum to %g",
                   altVal - 1e-5);
            v->setMin(altVal - 1e-5);
         }
         if (v->getMax() <= altVal) {
            ::Info("xRooHypoSpace::scan", "range of POI does not straddle alt value, adjusting maximum to %g",
                   altVal + 1e-5);
            v->setMax(altVal + 1e-5);
         }
         for (auto &[pdf, nll] : fNlls) {
            if (auto _v = dynamic_cast<RooRealVar *>(nll->pars()->find(*a))) {
               _v->setRange(v->getMin(), v->getMax());
            }
         }
      }
   } else if (sType.Contains("plr")) {
      // force use of two-sided test statistic for any new points
      fTestStatType = xRooFit::Asymptotics::TwoSided;
      sType.ReplaceAll("plr", "ts");
   } else if (sType.Contains("pnull") && fTestStatType == xRooFit::Asymptotics::Unknown) {
      // for pnull (aka discovery) "scan" (may just be a single point) default to use of
      // uncapped test stat
      fTestStatType = xRooFit::Asymptotics::Uncapped;
      // and ensure altVal is set
      for (auto a : axes()) {
         if (!a->getStringAttribute("altVal") || !strlen(p->getStringAttribute("altVal"))) {
            ::Info("xRooHypoSpace::scan", "No altVal set for %s, setting to 1", a->GetName());
            a->setStringAttribute("altVal", "1");
         }
      }
   }

   if (high < low || (high == low && nPoints != 1)) {
      // take from parameter
      low = p->getMin("scan");
      high = p->getMax("scan");
   }
   if (!std::isnan(low) && !std::isnan(high) && !(std::isinf(low) && std::isinf(high))) {
      p->setRange("scan", low, high);
   }
   if (p->hasRange("scan")) {
      ::Info("xRooHypoSpace::scan", "Using %s scan range: %g - %g", p->GetName(), p->getMin("scan"), p->getMax("scan"));
   }

   bool doObs = false;
   for (auto nSigma : nSigmas) {
      if (std::isnan(nSigma)) {
         doObs = true;
         break;
      }
   }

   if (fNlls.empty()) {
      // this happens when loaded hypoSpace from a hypoSpaceInverterResult
      // set relUncert to infinity so that we don't test any new points
      relUncert = std::numeric_limits<double>::infinity(); // no NLL available so just get whatever limit we can

      // if any of the defined points are 'expected' data don't do obs
      //      for(auto& hp : *this) {
      //         if(hp.isExpected) {
      //            doObs = false; break;
      //         }
      //      }
   } else {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
//      bool allGen = true;
//      for (auto &[pdf, nll]: fNlls) {
//         auto _d = dynamic_cast<RooDataSet *>(nll->data());
//         if (!_d || !_d->weightVar() || !_d->weightVar()->getStringAttribute("fitResult") ||
//             !_d->weightVar()->getAttribute("expected")) {
//            allGen = false;
//            break;
//         }
//      }
//      if (allGen)
//         doObs = false;
#endif
   }

   // create a fitDatabase if required
   TDirectory *origDir = gDirectory;
   if (!gDirectory || !gDirectory->IsWritable()) {
      // locate a TMemFile in the open list of files and move to that
      // or create one if cannot find
      /*for (auto file : *gROOT->GetListOfFiles()) {
         if (auto f = dynamic_cast<TMemFile *>(file)) {
            f->cd();
            break;
         }
      }
      if (!gDirectory || !gDirectory->IsWritable()) {
         new TMemFile("fitDatabase", "RECREATE");
      }*/
      // now we create a TMemFile of our own, so that we don't get in the way of other hypoSpaces
      fFitDb = std::shared_ptr<TMemFile>(new TMemFile(TString::Format("fitDatabase_%s",GetName()),"RECREATE"),[](TFile *) {});
      // db can last longer than the hypoSpace, so that the fits are fully available in the browser
      // if a scan was initiated through the browser. If user wants to cleanup they can do manually
      // through root's GetListOfFiles()
      // would like to clean it up ourself when the hypoSpace is destroyed, but would need way to keep alive for the browser
   }

   int out = 0;

   if (nPoints == 0) {
      // automatic scan
      if (sType.Contains("cls")) {
         for (double nSigma : nSigmas) {
            xValueWithError res(std::make_pair(0., 0.));
            if (std::isnan(nSigma)) {
               if (!doObs)
                  continue;
               res = findlimit(TString::Format("%s obs", sType.Data()), relUncert);
            } else {
               res =
                  findlimit(TString::Format("%s exp%s%d", sType.Data(), nSigma > 0 ? "+" : "", int(nSigma)), relUncert);
            }
            if (std::isnan(res.first) || std::isnan(res.second)) {
               out = 1;
            } else if (std::isinf(res.second)) {
               out = 2;
            }
         }
      } else {
         throw std::runtime_error(TString::Format("Automatic scanning not yet supported for %s", type));
      }
   } else {
      // add the required points and then compute the required value
      if (nPoints == 1) {
         AddPoint(TString::Format("%s=%g", poi().first()->GetName(), (high + low) / 2.));
         graphs(sType); // triggers computation
      } else {
         double step = (high - low) / (nPoints - 1);
         for (size_t i = 0; i < nPoints; i++) {
            AddPoint(TString::Format("%s=%g", poi().first()->GetName(), low + step * i));
            graphs(sType); // triggers computation
         }
      }
   }

   if (origDir)
      origDir->cd();

   return out;
}

std::map<std::string, xRooNLLVar::xValueWithError>
xRooNLLVar::xRooHypoSpace::limits(const char *opt, const std::vector<double> &nSigmas, double relUncert)
{

   if (fNlls.empty()) {
      // this happens when loaded hypoSpace from a hypoSpaceInverterResult
      // set relUncert to infinity so that we don't test any new points
      relUncert = std::numeric_limits<double>::infinity(); // no NLL available so just get whatever limit we can
   }

   scan(opt, nSigmas, relUncert);

   std::map<std::string, xRooNLLVar::xValueWithError> out;
   for (auto nSigma : nSigmas) {
      auto lim = limit(opt, nSigma);
      if (lim.second < 0)
         lim.second = -lim.second; // make errors positive for this method
      out[std::isnan(nSigma) ? "obs" : TString::Format("%d", int(nSigma)).Data()] = xRooFit::matchPrecision(lim);
   }
   return out;
}

xRooNLLVar::xRooHypoPoint &xRooNLLVar::xRooHypoSpace::AddPoint(const char *coords)
{
   // move to given coords, if any ... will mark them const too
   std::unique_ptr<RooAbsCollection, std::function<void(RooAbsCollection *)>> _snap(fPars->snapshot(),
                                                                                    [&](RooAbsCollection *c) {
                                                                                       *fPars = *c;
                                                                                       delete c;
                                                                                    });
   TStringToken pattern(coords, ";");
   while (pattern.NextToken()) {
      TString s = pattern;
      // split by "=" sign
      auto _idx = s.Index('=');
      if (_idx == -1)
         continue;
      TString _name = s(0, _idx);
      TString _val = s(_idx + 1, s.Length());
      auto _v = dynamic_cast<RooRealVar *>(fPars->find(_name));
      if (!_v)
         continue;

      if (_val.IsFloat()) {
         _v->setConstant();
         _v->setVal(_val.Atof());
      }
   }

   auto _pdf = pdf();

   if (!_pdf)
      throw std::runtime_error("no model at coordinates");

   //   if (std::unique_ptr<RooAbsCollection>(fPars->selectByAttrib("poi", true))->size() == 0) {
   //      throw std::runtime_error(
   //         "No pars designated as POI - set with pars()->find(<parName>)->setAttribute(\"poi\",true)");
   //   }

   if (fNlls.find(_pdf) == fNlls.end()) {
      fNlls[_pdf] = std::make_shared<xRooNLLVar>(_pdf->nll("" /*TODO:allow change dataset name and nll opts*/, {}));
   }

   xRooHypoPoint out;

   out.nllVar = fNlls[_pdf];
   out.fData = fNlls[_pdf]->getData();
   out.isExpected = dynamic_cast<RooDataSet *>(out.fData.first.get()) &&
                    dynamic_cast<RooDataSet *>(out.fData.first.get())->weightVar()->getAttribute("expected");
   // TODO: need to access the genfit of the data and add that to the point, somehow ...

   out.coords.reset(fPars->snapshot()); // should already have altVal prop on poi, and poi labelled
   // ensure all poi are marked const ... required by xRooHypoPoint behaviour
   out.poi().setAttribAll("Constant");
   // and now remove anything that's marked floating

   // do to bug in remove have to ensure not using the hash map otherwise will be doing an invalid read after the
   // deletion of the owned pars
   const_cast<RooAbsCollection *>(out.coords.get())->useHashMapForFind(false);
   const_cast<RooAbsCollection *>(out.coords.get())
      ->remove(*std::unique_ptr<RooAbsCollection>(out.coords->selectByAttrib("Constant", false)), true, true);

   double value = out.fNullVal();
   double alt_value = out.fAltVal();

   auto _type = fTestStatType;
   if (_type == xRooFit::Asymptotics::Unknown) {
      // decide based on values
      if (std::isnan(alt_value)) {
         _type = xRooFit::Asymptotics::TwoSided;
      } else if (value >= alt_value) {
         _type = xRooFit::Asymptotics::OneSidedPositive;
      } else {
         _type = xRooFit::Asymptotics::Uncapped;
      }
   }

   out.fPllType = _type;

   // look for a matching point
   for (auto &p : *this) {
      if (p.nllVar != out.nllVar)
         continue;
      if (p.fData != out.fData)
         continue;
      if (!p.alt_poi().equals(out.alt_poi()))
         continue;
      bool match = true;
      for (auto c : p.alt_poi()) {
         if (auto v = dynamic_cast<RooAbsReal *>(c);
             v && std::abs(v->getVal() - out.alt_poi().getRealValue(v->GetName())) > 1e-12) {
            match = false;
            break;
         } else if (auto cat = dynamic_cast<RooAbsCategory *>(c);
                    cat && cat->getCurrentIndex() ==
                              out.alt_poi().getCatIndex(cat->GetName(), std::numeric_limits<int>().max())) {
            match = false;
            break;
         }
      }
      if (!match)
         continue;
      if (!p.coords->equals(*out.coords))
         continue;
      for (auto c : *p.coords) {
         if (c->getAttribute("poi")) {
            continue; // check poi below
         }
         if (auto v = dynamic_cast<RooAbsReal *>(c);
             v && std::abs(v->getVal() - out.coords->getRealValue(v->GetName())) > 1e-12) {
            match = false;
            break;
         } else if (auto cat = dynamic_cast<RooAbsCategory *>(c);
                    cat && cat->getCurrentIndex() ==
                              out.alt_poi().getCatIndex(cat->GetName(), std::numeric_limits<int>().max())) {
            match = false;
            break;
         }
      }
      if (!match)
         continue;
      // if reached here we can copy over the asimov dataset to save re-generating it
      // first copy over cfit_alt (if its there) because that is also the same since coords and alt_poi values same
      if (auto cfit = p.cfit_alt(true)) {
         out.fAlt_cfit = cfit;
      }
      if (p.asimov(true) && p.asimov(true)->fData.first && (!out.asimov(true) || !out.asimov(true)->fData.first)) {
         out.asimov()->fData = p.asimov(true)->fData;
      }
      if (!p.poi().equals(out.poi()))
         continue;
      for (auto c : p.poi()) {
         if (auto v = dynamic_cast<RooAbsReal *>(c);
             v && std::abs(v->getVal() - out.poi().getRealValue(v->GetName())) > 1e-12) {
            match = false;
            break;
         }
      }
      if (match) {
         // found a duplicate point, return that!
         return p;
      }
   }

   std::string coordString;
   for (auto a : axes()) {
      coordString += TString::Format("%s=%g", a->GetName(), out.coords->getRealValue(a->GetName()));
      coordString += ",";
   }
   coordString.erase(coordString.end() - 1);

   ::Info("xRooHypoSpace::AddPoint", "Added new point @ %s", coordString.c_str());
   return emplace_back(out);
}

bool xRooNLLVar::xRooHypoSpace::AddModel(const xRooNode &_pdf, const char *validity)
{

   if (!_pdf.get<RooAbsPdf>()) {
      throw std::runtime_error("Not a pdf");
   }

   auto pars = _pdf.pars().argList();

   // replace any pars with validity pars and add new pars
   auto vpars = toArgs(validity);
   pars.replace(vpars);
   vpars.remove(pars, true, true);
   pars.add(vpars);

   if (auto existing = pdf(pars)) {
      throw std::runtime_error(std::string("Clashing model: ") + existing->GetName());
   }

   auto myPars = std::shared_ptr<RooArgList>(dynamic_cast<RooArgList *>(pars.snapshot()));
   myPars->sort();

   pars.remove(*fPars, true, true);

   fPars->addClone(pars);

   fPdfs.insert(std::make_pair(myPars, std::make_shared<xRooNode>(_pdf)));

   return true;
}

RooArgList xRooNLLVar::xRooHypoSpace::axes() const
{
   // determine which pars are the minimal set to distinguish all points in the space
   RooArgList out;
   out.setName("axes");

   out.add(*std::unique_ptr<RooAbsCollection>(
      fPars->selectByAttrib("axis", true))); // start with any pars explicitly designated as axes

   bool clash;
   do {
      clash = false;

      std::set<std::vector<double>> coords;
      for (auto &p : *this) {
         std::vector<double> p_coords;
         for (auto o : out) {
            auto _v = dynamic_cast<RooRealVar *>(p.coords->find(o->GetName()));
            p_coords.push_back(
               (_v && _v->isConstant())
                  ? _v->getVal()
                  : std::numeric_limits<double>::infinity()); // non-const coords are treating as non-existent
            // p_coords.push_back(p.coords->getRealValue(o->GetName(), std::numeric_limits<double>::quiet_NaN()));
         }
         if (coords.find(p_coords) != coords.end()) {
            clash = true;
            break;
         }
         coords.insert(p_coords);
      }

      if (clash) {
         // add next best coordinate
         std::map<std::string, std::unordered_set<double>> values;
         for (auto &par : *pars()) {
            if (out.find(*par))
               continue;
            for (auto p : *this) {
               auto _v = dynamic_cast<RooRealVar *>(p.coords->find(par->GetName()));
               values[par->GetName()].insert(
                  (_v && _v->isConstant())
                     ? _v->getVal()
                     : std::numeric_limits<double>::infinity()); // non-const coords are treating as non-existent
               // values[par->GetName()].insert(
               //         p.coords->getRealValue(par->GetName(), std::numeric_limits<double>::quiet_NaN()));
            }
         }

         std::string bestVar;
         size_t maxDiff = 0;
         bool isPOI = false;
         for (auto &[k, v] : values) {
            if (v.size() > maxDiff || (v.size() == maxDiff && !isPOI && pars()->find(k.c_str())->getAttribute("poi"))) {
               bestVar = k;
               isPOI = pars()->find(k.c_str())->getAttribute("poi");
               maxDiff = std::max(maxDiff, v.size());
            }
         }
         if (bestVar.empty()) {
            break;
         }

         out.add(*pars()->find(bestVar.c_str()));
      }
   } while (clash);

   // ensure poi are at the end
   std::unique_ptr<RooAbsCollection> poi(out.selectByAttrib("poi", true));
   out.remove(*poi);
   out.add(*poi);

   return out;
}

RooArgList xRooNLLVar::xRooHypoSpace::poi()
{
   RooArgList out;
   out.setName("poi");
   out.add(*std::unique_ptr<RooAbsCollection>(pars()->selectByAttrib("poi", true)));
   return out;
}

void xRooNLLVar::xRooHypoSpace::LoadFits(const char *apath)
{

   if (!gDirectory)
      return;
   auto dir = gDirectory->GetDirectory(apath);
   if (!dir) {
      // try open file first
      TString s(apath);
      auto f = TFile::Open(s.Contains(":") ? TString(s(0, s.Index(":"))) : s);
      if (f) {
         if (!s.Contains(":"))
            s += ":";
         dir = gDirectory->GetDirectory(s);
         if (dir) {
            LoadFits(s);
            return;
         }
      }
      if (!dir) {
         Error("LoadFits", "Path not found %s", apath);
         return;
      }
   }

   // assume for now all fits in given path will have the same pars
   // so can just look at the float and const pars of first fit result to get all of them
   // tuple is: parName, parValue, parAltValue (blank if nan)
   // key represents the ufit values, value represents the sets of poi for the available cfits (subfits of the ufit)

   std::map<std::set<std::tuple<std::string, double, std::string>>, std::set<std::set<std::string>>> cfits;
   std::set<std::string> allpois;

   int nFits = 0;
   std::function<void(TDirectory *)> processDir;
   processDir = [&](TDirectory *_dir) {
      std::cout << "Processing " << _dir->GetName() << std::endl;
      if (auto keys = _dir->GetListOfKeys(); keys) {
         // first check if dir doesn't contain any RooLinkedList ... this identifies it as not an nll dir
         // so treat any sub-dirs as new nll
         bool isNllDir = false;
         for (auto &&k : *keys) {
            TKey *key = dynamic_cast<TKey *>(k);
            if (strcmp(key->GetClassName(), "RooLinkedList") == 0) {
               isNllDir = true;
               break;
            }
         }

         for (auto &&k : *keys) {
            if (auto subdir = _dir->GetDirectory(k->GetName()); subdir) {
               if (!isNllDir) {
                  LoadFits(subdir->GetPath());
               } else {
                  processDir(subdir);
               }
               continue;
            }
            auto cl = TClass::GetClass((static_cast<TKey *>(k))->GetClassName());
            if (cl->InheritsFrom("RooFitResult")) {
               if (auto cachedFit = _dir->Get<RooFitResult>(k->GetName()); cachedFit) {
                  nFits++;
                  if (nFits == 1) {
                     // for first fit add any missing float pars
                     std::unique_ptr<RooAbsCollection> snap(cachedFit->floatParsFinal().snapshot());
                     snap->remove(*fPars, true, true);
                     fPars->addClone(*snap);
                     // add also the non-string const pars
                     for (auto &p : cachedFit->constPars()) {
                        if (p->getAttribute("global"))
                           continue; // don't consider globals
                        auto v = dynamic_cast<RooAbsReal *>(p);
                        if (!v) {
                           continue;
                        };
                        if (!fPars->contains(*v))
                           fPars->addClone(*v);
                     }
                  }
                  // get names of all the floats
                  std::set<std::string> floatPars;
                  for (auto &p : cachedFit->floatParsFinal())
                     floatPars.insert(p->GetName());
                  // see if

                  // build a set of the const par values
                  std::set<std::tuple<std::string, double, std::string>> constPars;
                  for (auto &p : cachedFit->constPars()) {
                     if (p->getAttribute("global"))
                        continue; // don't consider globals when looking for cfits
                     auto v = dynamic_cast<RooAbsReal *>(p);
                     if (!v) {
                        continue;
                     };
                     constPars.insert(
                        std::make_tuple(v->GetName(), v->getVal(),
                                        v->getStringAttribute("altVal") ? v->getStringAttribute("altVal") : ""));
                  }
                  // now see if this is a subset of any existing cfit ...
                  for (auto &&[key, value] : cfits) {
                     if (constPars == key)
                        continue; // ignore cases where we already recorded this list of constPars
                     if (std::includes(constPars.begin(), constPars.end(), key.begin(), key.end())) {
                        // usual case ... cachedFit has more constPars than one of the fits we have already encountered
                        // (the ufit)
                        // => cachedFit is a cfit of key fr ...
                        std::set<std::string> pois;
                        for (auto &&par : constPars) {
                           if (key.find(par) == key.end()) {
                              pois.insert(std::get<0>(par));
                              allpois.insert(std::get<0>(par));
                           }
                        }
                        if (!pois.empty()) {
                           cfits[constPars].insert(pois);
                           //                                    std::cout << cachedFit->GetName() << " ";
                           //                                    for(auto ff: constPars) std::cout << ff.first << "=" <<
                           //                                    ff.second << " "; std::cout << std::endl;
                        }
                     }
                     /* FOR NOW we will skip cases where we encounter the cfit before the ufit - usually should eval the
                     ufit first
                      * else if (std::includes(key.begin(), key.end(), constPars.begin(), constPars.end())) {
                         // constPars are subset of key
                         // => key is a ufit of the cachedFit
                         // add all par names of key that aren't in constPars ... these are the poi
                         std::set<std::string> pois;
                         for (auto &&par: key) {
                             if (constPars.find(par) == constPars.end()) {
                                 pois.insert(std::get<0>(par));
                                 allpois.insert(std::get<0>(par));
                             }
                         }
                         if (!pois.empty()) {
                             std::cout << "found cfit BEFORE ufit??" << std::endl;
                             value.insert(pois);
                         }
                     } */
                  }
                  // ensure that this combination of constPars has entry in map,
                  // even if it doesn't end up with any poi identified from cfits to it
                  cfits[constPars];
                  delete cachedFit;
               }
            }
         }
      }
   };
   processDir(dir);
   ::Info("xRooHypoSpace::xRooHypoSpace", "%s - Loaded %d fits", apath, nFits);

   if (allpois.size() == 1) {
      ::Info("xRooHypoSpace::xRooHypoSpace", "Detected POI: %s", allpois.begin()->c_str());

      auto nll = std::make_shared<xRooNLLVar>(nullptr, nullptr);
      auto dummyNll = std::make_shared<RooRealVar>(apath, "Dummy NLL", std::numeric_limits<double>::quiet_NaN());
      nll->std::shared_ptr<RooAbsReal>::operator=(dummyNll);
      dummyNll->setAttribute("readOnly");
      // add pars as 'servers' on the dummy NLL
      if (fPars) {
         for (auto &&p : *fPars) {
            dummyNll->addServer(
               *p); // this is ok provided fPars (i.e. hypoSpace) stays alive as long as the hypoPoint ...
         }
         // flag poi
         for (auto &p : allpois) {
            fPars->find(p.c_str())->setAttribute("poi", true);
         }
      }
      nll->reinitialize(); // triggers filling of par lists etc

      for (auto &&[key, value] : cfits) {
         if (value.find(allpois) != value.end()) {
            // get the value of the poi in the key set
            auto _coords = std::make_shared<RooArgSet>();
            for (auto &k : key) {
               auto v = _coords->addClone(RooRealVar(std::get<0>(k).c_str(), std::get<0>(k).c_str(), std::get<1>(k)));
               v->setAttribute("poi", allpois.find(std::get<0>(k)) != allpois.end());
               if (!std::get<2>(k).empty()) {
                  v->setStringAttribute("altVal", std::get<2>(k).c_str());
               }
            }
            xRooNLLVar::xRooHypoPoint hp;
            // hp.fPOIName = allpois.begin()->c_str();
            // hp.fNullVal = _coords->getRealValue(hp.fPOIName.c_str());
            hp.coords = _coords;
            hp.nllVar = nll;

            //                auto altVal =
            //                hp.null_cfit()->constPars().find(hp.fPOIName.c_str())->getStringAttribute("altVal");
            //                if(altVal) hp.fAltVal = TString(altVal).Atof();
            //                else hp.fAltVal = std::numeric_limits<double>::quiet_NaN();

            // decide based on values
            if (std::isnan(hp.fAltVal())) {
               hp.fPllType = xRooFit::Asymptotics::TwoSided;
            } else if (hp.fNullVal() >= hp.fAltVal()) {
               hp.fPllType = xRooFit::Asymptotics::OneSidedPositive;
            } else {
               hp.fPllType = xRooFit::Asymptotics::Uncapped;
            }

            emplace_back(hp);
         }
      }
   } else if (nFits > 0) {
      std::cout << "possible POI: ";
      for (auto p : allpois)
         std::cout << p << ",";
      std::cout << std::endl;
   }
}

void xRooNLLVar::xRooHypoSpace::Print(Option_t * /*opt*/) const
{

   auto _axes = axes();

   size_t badFits = 0;

   for (size_t i = 0; i < size(); i++) {
      std::cout << i << ") ";
      for (auto a : _axes) {
         if (a != _axes.first())
            std::cout << ",";
         std::cout << a->GetName() << "="
                   << at(i).coords->getRealValue(a->GetName(), std::numeric_limits<double>::quiet_NaN());
      }
      std::cout << " status=[ufit:";
      auto ufit = const_cast<xRooHypoPoint &>(at(i)).ufit(true);
      if (!ufit) {
         std::cout << "-";
      } else {
         std::cout << ufit->status();
         badFits += (xRooNLLVar::xRooHypoPoint::allowedStatusCodes.count(ufit->status()) == 0);
      }
      std::cout << ",cfit_null:";
      auto cfit = const_cast<xRooHypoPoint &>(at(i)).cfit_null(true);
      if (!cfit) {
         std::cout << "-";
      } else {
         std::cout << cfit->status();
         badFits += (xRooNLLVar::xRooHypoPoint::allowedStatusCodes.count(cfit->status()) == 0);
      }
      std::cout << ",cfit_alt:";
      auto afit = const_cast<xRooHypoPoint &>(at(i)).cfit_alt(true);
      if (!afit) {
         std::cout << "-";
      } else {
         std::cout << afit->status();
         badFits += (xRooNLLVar::xRooHypoPoint::allowedStatusCodes.count(afit->status()) == 0);
      }
      if (auto asiPoint = const_cast<xRooHypoPoint &>(at(i)).asimov(true)) {
         std::cout << ",asimov.ufit:";
         auto asi_ufit = asiPoint->ufit(true);
         if (!asi_ufit) {
            std::cout << "-";
         } else {
            std::cout << asi_ufit->status();
            badFits += (xRooNLLVar::xRooHypoPoint::allowedStatusCodes.count(asi_ufit->status()) == 0);
         }
         std::cout << ",asimov.cfit_null:";
         auto asi_cfit = asiPoint->cfit_null(true);
         if (!asi_cfit) {
            std::cout << "-";
         } else {
            std::cout << asi_cfit->status();
            badFits += (xRooNLLVar::xRooHypoPoint::allowedStatusCodes.count(asi_cfit->status()) == 0);
         }
      }
      std::cout << "]";
      auto sigma_mu = const_cast<xRooHypoPoint &>(at(i)).sigma_mu(true);
      if (!std::isnan(sigma_mu.first)) {
         std::cout << " sigma_mu=" << sigma_mu.first;
         if (sigma_mu.second)
            std::cout << " +/- " << sigma_mu.second;
      }
      std::cout << std::endl;
   }
   std::cout << "--------------------------" << std::endl;
   std::cout << "Number of bad fits: " << badFits << std::endl;
}

std::shared_ptr<TGraphErrors> xRooNLLVar::xRooHypoSpace::graph(
   const char *opt /*, const std::function<void(xRooNLLVar::xRooHypoSpace*)>& progress*/) const
{

   TString sOpt(opt);
   sOpt.ToLower();

   bool doCLs = sOpt.Contains("cls");
   bool readOnly = sOpt.Contains("readonly");
   bool visualize = sOpt.Contains("visualize") && !readOnly;

   double nSigma =
      (sOpt.Contains("exp"))
         ? (TString(sOpt(sOpt.Index("exp") + 3,
                         sOpt.Index(" ", sOpt.Index("exp")) == -1 ? sOpt.Length() : sOpt.Index(" ", sOpt.Index("exp"))))
               .Atof())
         : std::numeric_limits<double>::quiet_NaN();
   bool expBand =
      !std::isnan(nSigma) && nSigma && !(sOpt(sOpt.Index("exp") + 3) == '+' || sOpt(sOpt.Index("exp") + 3) == '-');

   auto _axes = axes();
   if (_axes.size() != 1)
      return nullptr;

   auto out = std::make_shared<TGraphErrors>();
   out->SetName(GetName());
   out->SetEditable(false);
   const char *sCL = (doCLs) ? "CLs" : "null";

   TString title =
      TString::Format("%s;%s;p_{%s}",
                      (std::isnan(nSigma))
                         ? "Observed"
                         : (!nSigma ? "Expected"
                                    : TString::Format("%s%d#sigma Expected",
                                                      expBand || !nSigma ? "" : ((nSigma < 0) ? "-" : "+"), int(nSigma))
                                         .Data()),
                      _axes.at(0)->GetTitle(), sCL);

   if (std::isnan(nSigma)) {
      out->SetNameTitle(TString::Format("obs_p%s", sCL), title);
      out->SetMarkerStyle(20);
      out->SetMarkerSize(0.5);
      if (sOpt.Contains("ts")) {
         out->SetNameTitle("obs_ts", TString::Format("Observed;%s;%s", _axes.at(0)->GetTitle(),
                                                     (empty() ? "" : front().tsTitle(true).Data())));
      }
   } else {
      out->SetNameTitle(TString::Format("exp%d_p%s", int(nSigma), sCL), title);
      out->SetMarkerStyle(0);
      out->SetMarkerSize(0);
      out->SetLineStyle(2 + int(nSigma));
      if (expBand && nSigma) {
         out->SetFillColor((nSigma == 2) ? kYellow : kGreen);
         // out->SetFillStyle(3005);
         out->SetLineStyle(0);
         out->SetLineWidth(0);
         auto x = out->Clone("up");
         x->SetBit(kCanDelete);
         dynamic_cast<TAttFill *>(x)->SetFillStyle(nSigma == 2 ? 3005 : 3004);
         dynamic_cast<TAttFill *>(x)->SetFillColor(kBlack);
         out->GetListOfFunctions()->Add(x, "F");
         x = out->Clone("down");
         x->SetBit(kCanDelete);
         // dynamic_cast<TAttFill*>(x)->SetFillColor((nSigma==2) ? kYellow : kGreen);
         // dynamic_cast<TAttFill*>(x)->SetFillStyle(1001);
         out->GetListOfFunctions()->Add(x, "F");
      }
      if (sOpt.Contains("ts")) {
         out->SetNameTitle(TString::Format("exp_ts%d", int(nSigma)),
                           TString::Format("Expected;%s;%s", _axes.at(0)->GetTitle(), front().tsTitle(true).Data()));
      }
   }

   auto badPoints = [&]() {
      auto badPoints2 = dynamic_cast<TGraph *>(out->GetListOfFunctions()->FindObject("badPoints"));
      if (!badPoints2) {
         badPoints2 = new TGraph;
         badPoints2->SetBit(kCanDelete);
         badPoints2->SetName("badPoints");
         badPoints2->SetMarkerStyle(5);
         badPoints2->SetMarkerColor(std::isnan(nSigma) ? kRed : kBlue);
         badPoints2->SetMarkerSize(1);
         out->GetListOfFunctions()->Add(badPoints2, "P");
      }
      return badPoints2;
   };
   int nPointsDown = 0;
   bool above = true;
   TStopwatch s;
   s.Start();
   size_t nDone = 0;
   for (auto &p : *this) {
      if (s.RealTime() > 5) {
         if (visualize) {
            // draw readonly version of the graph
            auto gra = graph(sOpt + " readOnly");
            if (gra && gra->GetN()) {
               if (!gPad && gROOT->GetSelectedPad())
                  gROOT->GetSelectedPad()->cd();
               if (gPad)
                  gPad->Clear();
               gra->DrawClone(expBand ? "AF" : "ALP")->SetBit(kCanDelete);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
               if (auto pad = gROOT->GetSelectedPad()) {
                  pad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
               }
#endif
               gSystem->ProcessEvents();
            }
         } else {
            ::Info("xRooHypoSpace::graph", "Completed %d/%d points for %s", int(nDone), int(size()), sOpt.Data());
         }
         s.Start();
      } else {
         s.Continue();
      }
      double _x = p.coords->getRealValue(_axes.at(0)->GetName(), std::numeric_limits<double>::quiet_NaN());
      auto pval = const_cast<xRooHypoPoint &>(p).getVal(sOpt);
      auto idx = out->GetN() - nPointsDown;

      if (std::isnan(pval.first)) {
         if (p.status() != 0) { // if status is 0 then bad pval is really just absence of fits, not bad fits
            badPoints()->SetPoint(badPoints()->GetN(), _x, 0);
         }
      } else {
         out->InsertPointBefore(idx, _x, pval.first);
         out->SetPointError(idx, 0, pval.second);
      }

      if (expBand && nSigma) {
         TString sOpt2 = sOpt;
         sOpt2.ReplaceAll("exp", "exp-");
         pval = const_cast<xRooHypoPoint &>(p).getVal(sOpt2);
         if (std::isnan(pval.first)) {
            if (p.status() != 0) { // if status is 0 then bad pval is really just absence of fits, not bad fits
               badPoints()->SetPoint(badPoints()->GetN(), _x, 0);
            }
         } else {
            out->InsertPointBefore(idx + 1, _x, pval.first);
            out->SetPointError(idx + 1, 0, pval.second);
            nPointsDown++;
            if (out->GetPointY(idx) < pval.first)
               above = false; // the -sigma points are actually above +sigma
         }
      }
      nDone++;
   }

   if (out->GetN() == 0)
      return out;

   if (!expBand) {
      out->Sort();
      if (out->GetListOfFunctions()->FindObject("badPoints")) {
         // try to interpolate the points
         for (int i = 0; i < badPoints()->GetN(); i++) {
            badPoints()->SetPointY(i, out->Eval(badPoints()->GetPointX(i)));
         }
      }
   } else {
      out->Sort(&TGraph::CompareX, true, 0, out->GetN() - nPointsDown - 1);            // sort first half
      out->Sort(&TGraph::CompareX, false, out->GetN() - nPointsDown, out->GetN() - 1); // reverse sort second half

      // now populate the up and down values
      auto up = dynamic_cast<TGraph *>(out->GetListOfFunctions()->FindObject("up"));
      auto down = dynamic_cast<TGraph *>(out->GetListOfFunctions()->FindObject("down"));

      for (int i = 0; i < out->GetN(); i++) {
         if (i < out->GetN() - nPointsDown) {
            up->SetPoint(up->GetN(), out->GetPointX(i), out->GetPointY(i) + out->GetErrorY(i) * (above ? 1. : -1.));
            down->SetPoint(down->GetN(), out->GetPointX(i), out->GetPointY(i) - out->GetErrorY(i) * (above ? 1. : -1.));
         } else {
            up->SetPoint(up->GetN(), out->GetPointX(i), out->GetPointY(i) - out->GetErrorY(i) * (above ? 1. : -1.));
            down->SetPoint(down->GetN(), out->GetPointX(i), out->GetPointY(i) + out->GetErrorY(i) * (above ? 1. : -1.));
         }
      }
   }

   if (visualize) {
      // draw result
      if (!gPad && gROOT->GetSelectedPad())
         gROOT->GetSelectedPad()->cd();
      if (gPad)
         gPad->Clear();
      out->DrawClone(expBand ? "AF" : "ALP")->SetBit(kCanDelete);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
      if (auto pad = gROOT->GetSelectedPad()) {
         pad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
      }
#endif
      gSystem->ProcessEvents();
   }

   return out;
}

std::shared_ptr<TMultiGraph> xRooNLLVar::xRooHypoSpace::graphs(const char *opt)
{
   TString sOpt(opt);
   sOpt.ToLower();
   std::shared_ptr<TMultiGraph> out;
   if (sOpt.Contains("pcls") || sOpt.Contains("pnull") || sOpt.Contains("ts")) {

      bool visualize = sOpt.Contains("visualize");
      sOpt.ReplaceAll("visualize", "");

      auto exp2 = graph(sOpt + " exp2");
      auto exp1 = graph(sOpt + " exp1");
      auto exp = graph(sOpt + " exp");
      bool doObs = true;
      // for(auto& hp : *this) { if(hp.isExpected) {doObs=false; break;} }
      auto obs = (doObs) ? graph(sOpt) : nullptr;

      out = std::make_shared<TMultiGraph>(GetName(), GetTitle());
      if (exp2 && exp2->GetN() > 1)
         out->Add(static_cast<TGraph *>(exp2->Clone()), "FP");
      if (exp1 && exp1->GetN() > 1)
         out->Add(static_cast<TGraph *>(exp1->Clone()), "FP");
      if (exp && exp->GetN() > 1)
         out->Add(static_cast<TGraph *>(exp->Clone()), "LP");
      if (obs && obs->GetN() > 1)
         out->Add(static_cast<TGraph *>(obs->Clone()), "LP");

      if (!out->GetListOfGraphs()) {
         return nullptr;
      }

      TGraph *testedPoints = nullptr;
      if (sOpt.Contains("pcls")) {
         TGraph *line = new TGraph;
         line->SetName("alpha");
         line->SetLineStyle(2);
         line->SetEditable(false);
         line->SetPoint(line->GetN(), out->GetHistogram()->GetXaxis()->GetXmin() - 10, 0.05);
         testedPoints = new TGraph;
         testedPoints->SetName("hypoPoints");
         testedPoints->SetEditable(false);
         testedPoints->SetMarkerStyle(24);
         testedPoints->SetMarkerSize(0.4); // use line to indicate tested points
         if (exp) {
            for (int i = 0; i < exp->GetN(); i++) {
               testedPoints->SetPoint(testedPoints->GetN(), exp->GetPointX(i), 0.05);
            }
         }
         line->SetPoint(line->GetN(), out->GetHistogram()->GetXaxis()->GetXmax() + 10, 0.05);
         line->SetBit(kCanDelete);
         out->GetListOfFunctions()->Add(line, "L");
      }
      if (exp) {
         out->GetHistogram()->GetXaxis()->SetTitle(exp->GetHistogram()->GetXaxis()->GetTitle());
         out->GetHistogram()->GetYaxis()->SetTitle(exp->GetHistogram()->GetYaxis()->GetTitle());
      }
      auto leg = new TLegend(1. - gStyle->GetPadRightMargin() - 0.3, 1. - gStyle->GetPadTopMargin() - 0.35,
                             1. - gStyle->GetPadRightMargin() - 0.05, 1. - gStyle->GetPadTopMargin() - 0.05);
      leg->SetName("legend");
      leg->SetBit(kCanDelete);

      out->GetListOfFunctions()->Add(leg);
      // out->GetListOfFunctions()->Add(out->GetHistogram()->Clone(".axis"),"sameaxis"); // redraw axis

      for (auto g : *out->GetListOfGraphs()) {
         if (auto o = dynamic_cast<TGraph *>(g)->GetListOfFunctions()->FindObject("down")) {
            leg->AddEntry(o, "", "F");
         } else {
            leg->AddEntry(g, "", "LPE");
         }
      }

      if (sOpt.Contains("pcls")) {
         // add current limit estimates to legend
         if (exp2 && exp2->GetN() > 1) {
            auto l = xRooFit::matchPrecision(GetLimit(*graph(sOpt + "exp-2")));
            leg->AddEntry((TObject *)nullptr, TString::Format("-2#sigma: %g +/- %g", l.first, l.second), "");
         }
         if (exp1 && exp1->GetN() > 1) {
            auto l = xRooFit::matchPrecision(GetLimit(*graph(sOpt + "exp-1")));
            leg->AddEntry((TObject *)nullptr, TString::Format("-1#sigma: %g +/- %g", l.first, l.second), "");
         }
         if (exp && exp->GetN() > 1) {
            auto l = xRooFit::matchPrecision(GetLimit(*exp));
            leg->AddEntry((TObject *)nullptr, TString::Format("0#sigma: %g +/- %g", l.first, l.second), "");
         }
         if (exp1 && exp1->GetN() > 1) {
            auto l = xRooFit::matchPrecision(GetLimit(*graph(sOpt + "exp+1")));
            leg->AddEntry((TObject *)nullptr, TString::Format("+1#sigma: %g +/- %g", l.first, l.second), "");
         }
         if (exp2 && exp2->GetN() > 1) {
            auto l = xRooFit::matchPrecision(GetLimit(*graph(sOpt + "exp+2")));
            leg->AddEntry((TObject *)nullptr, TString::Format("+2#sigma: %g +/- %g", l.first, l.second), "");
         }
         if (obs && obs->GetN() > 1) {
            auto l = xRooFit::matchPrecision(GetLimit(*obs));
            leg->AddEntry((TObject *)nullptr, TString::Format("Observed: %g +/- %g", l.first, l.second), "");
         }
      }
      if (testedPoints)
         out->Add(testedPoints, "P");

      if (visualize) {
         if (!gPad && gROOT->GetSelectedPad())
            gROOT->GetSelectedPad()->cd();
         if (gPad)
            gPad->Clear();
         auto gra2 = static_cast<TMultiGraph *>(out->DrawClone("A"));
         gra2->SetBit(kCanDelete);
         if (sOpt.Contains("pcls") || sOpt.Contains("pnull")) {
            gra2->GetHistogram()->SetMinimum(1e-6);
         }
         if (gPad) {
            gPad->RedrawAxis();
            gPad->GetCanvas()->Paint();
            gPad->GetCanvas()->Update();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
            gPad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
#endif
         }
         gSystem->ProcessEvents();
      }
   }

   return out;
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoSpace::GetLimit(const TGraph &pValues, double target)
{

   if (std::isnan(target)) {
      target = 1. - gEnv->GetValue("xRooHypoSpace.CL", 95.) / 100.;
   }

   auto gr = std::make_shared<TGraph>(pValues);
   // remove any nan points and duplicates
   int i = 0;
   std::set<double> existingX;
   while (i < gr->GetN()) {
      if (std::isnan(gr->GetPointY(i))) {
         gr->RemovePoint(i);
      } else if (existingX.find(gr->GetPointX(i)) != existingX.end()) {
         gr->RemovePoint(i);
      } else {
         existingX.insert(gr->GetPointX(i));
         // convert to log ....
         gr->SetPointY(i, log(std::max(gr->GetPointY(i), 1e-10)));
         i++;
      }
   }

   gr->Sort();

   // simple linear extrapolation to critical value ... return nan if problem
   if (gr->GetN() < 2) {
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
   }

   double alpha = log(target);

   bool above = gr->GetPointY(0) > alpha;
   for (int ii = 1; ii < gr->GetN(); ii++) {
      if ((above && (gr->GetPointY(ii) <= alpha)) || (!above && (gr->GetPointY(ii) >= alpha))) {
         // found the limit ... return linearly extrapolated point
         double lim = gr->GetPointX(ii - 1) + (gr->GetPointX(ii) - gr->GetPointX(ii - 1)) *
                                                 (alpha - gr->GetPointY(ii - 1)) /
                                                 (gr->GetPointY(ii) - gr->GetPointY(ii - 1));
         // use points either side as error
         double err = std::max(lim - gr->GetPointX(ii - 1), gr->GetPointX(ii) - lim);
         // give err negative sign to indicate if error due to negative side
         if ((lim - gr->GetPointX(ii - 1)) > (gr->GetPointX(ii) - lim))
            err *= -1;
         return std::pair(lim, err);
      }
   }
   // if reach here need to extrapolate ...
   if ((above && gr->GetPointY(gr->GetN() - 1) <= gr->GetPointY(0)) ||
       (!above && gr->GetPointY(gr->GetN() - 1) >= gr->GetPointY(0))) {
      // extrapolating above based on last two points
      // in fact, if 2nd last point is a p=1 (log(p)=0) then go back
      int offset = 2;
      while (offset < gr->GetN() && gr->GetPointY(gr->GetN() - offset) == 0)
         offset++;
      double x1 = gr->GetPointX(gr->GetN() - offset);
      double y1 = gr->GetPointY(gr->GetN() - offset);
      double m = (gr->GetPointY(gr->GetN() - 1) - y1) / (gr->GetPointX(gr->GetN() - 1) - x1);
      if (m == 0.)
         return std::pair(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
      return std::pair((alpha - y1) / m + x1, std::numeric_limits<double>::infinity());
   } else {
      // extrapolating below based on first two points
      double x1 = gr->GetPointX(0);
      double y1 = gr->GetPointY(0);
      double m = (gr->GetPointY(1) - y1) / (gr->GetPointX(1) - x1);
      if (m == 0.)
         return std::pair(-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity());
      return std::pair((alpha - y1) / m + x1, -std::numeric_limits<double>::infinity());
   }
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoSpace::limit(const char *type, double nSigma) const
{
   TString sOpt = TString::Format("p%s", type);
   if (std::isnan(nSigma)) {
      sOpt += "obs";
   } else {
      sOpt += TString::Format("exp%s%d", nSigma > 0 ? "+" : "", int(nSigma));
   }
   return GetLimit(*graph(sOpt + " readonly"));
}

xRooNLLVar::xValueWithError
xRooNLLVar::xRooHypoSpace::findlimit(const char *opt, double relUncert, unsigned int maxTries)
{
   TString sOpt(opt);
   bool visualize = sOpt.Contains("visualize");
   sOpt.ReplaceAll("visualize", "");
   std::shared_ptr<TGraphErrors> gr = graph(sOpt + " readonly");
   if (visualize) {
      auto gra = graphs(sOpt.Contains("toys") ? "pcls readonly toys" : "pcls readonly");
      if (gra) {
         if (!gPad)
            gra->Draw(); // in 6.28 DrawClone wont make the gPad defined :( ... so Draw then clear and Draw Clone
         gPad->Clear();
         gra->DrawClone("A")->SetBit(kCanDelete);
         gPad->RedrawAxis();
         gra->GetHistogram()->SetMinimum(1e-9);
         gra->GetHistogram()->GetYaxis()->SetRangeUser(1e-9, 1);
         gPad->Modified();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
         gPad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
#endif
         gSystem->ProcessEvents();
      }
   }

   // resync parameter boundaries from nlls (may have been modified by fits)
   for (auto p : axes()) {
      for (auto &[pdf, nll] : fNlls) {
         if (auto _v = dynamic_cast<RooRealVar *>(nll->pars()->find(*p))) {
            dynamic_cast<RooRealVar *>(p)->setRange(_v->getMin(), _v->getMax());
         }
      }
   }

   if (!gr || gr->GetN() < 2) {
      auto v = (axes().empty()) ? nullptr : dynamic_cast<RooRealVar *>(*axes().rbegin());
      if (!v)
         return std::pair(std::numeric_limits<double>::quiet_NaN(), 0.);
      double muMax = std::min(std::min(v->getMax("physical"), v->getMax()), v->getMax("scan"));
      double muMin = std::max(std::max(v->getMin("physical"), v->getMin()), v->getMin("scan"));
      if (!gr || gr->GetN() < 1) {
         if (maxTries == 0 || std::isnan(AddPoint(TString::Format("%s=%g", v->GetName(), muMin)).getVal(sOpt).first)) {
            // first point failed ... give up
            Error("findlimit", "Problem evaluating %s @ %s=%g", sOpt.Data(), v->GetName(), muMin);
            return std::pair(std::numeric_limits<double>::quiet_NaN(), 0.);
         }
         gr.reset();
         return findlimit(opt, relUncert, maxTries - 1); // do this to resync parameter limits
      }

      // can approximate expected limit using
      // mu_hat + sigma_mu*ROOT::Math::gaussian_quantile(1.-alpha/2.,1) for cls
      // or mu_hat + sigma_mu*ROOT::Math::gaussian_quantile((1.-alpha),1) for cls+b
      // get a very first estimate of sigma_mu from ufit to expected data, take error on mu as sigma_mu
      double nextPoint = muMin + (muMax - muMin) / 50;

      // if done an expected limit, assume data is like expected and choose expected limit point as first test point
      if (sOpt.Contains("obs")) {
         TString sOpt2 = sOpt;
         sOpt2.ReplaceAll("obs", "exp");
         auto expLim = findlimit(sOpt2, std::numeric_limits<double>::infinity(), 0);
         if (!std::isnan(expLim.first) && expLim.first < nextPoint)
            nextPoint = expLim.first;
      }

      auto point =
         (sOpt.Contains("exp")) ? back().asimov() : std::shared_ptr<xRooHypoPoint>(&back(), [](xRooHypoPoint *) {});
      point = nullptr;
      if (point && point->ufit()) {
         double rough_sigma_mu = point->mu_hat().getError();
         double another_estimate = point->mu_hat().getVal() + rough_sigma_mu * ROOT::Math::gaussian_quantile(0.95, 1);
         // if (another_estimate < nextPoint) {
         nextPoint = another_estimate;
         ::Info("xRooHypoSpace::findlimit", "Guessing %g based on rough sigma_mu = %g", nextPoint, rough_sigma_mu);
         //}
      }

      if (maxTries == 0 || std::isnan(AddPoint(TString::Format("%s=%g", v->GetName(), nextPoint)).getVal(sOpt).first)) {
         // second point failed ... give up
         Error("findlimit", "Problem evaluating %s @ %s=%g", sOpt.Data(), v->GetName(), nextPoint);
         return std::pair(std::numeric_limits<double>::quiet_NaN(), 0.);
      }
      gr.reset();
      return findlimit(opt, relUncert, maxTries - 1);
   }

   auto lim = GetLimit(*gr);

   if (std::isnan(lim.first)) {
      return lim;
   }

   auto v = dynamic_cast<RooRealVar *>(*axes().rbegin());
   double maxMu = std::min(std::min(v->getMax("physical"), v->getMax()), v->getMax("scan"));
   double minMu = std::max(std::max(v->getMin("physical"), v->getMin()), v->getMin("scan"));

   // static double MIN_LIMIT_UNCERT = 1e-4; // stop iterating once uncert gets this small
   if (lim.first > -std::numeric_limits<double>::infinity() && lim.first < std::numeric_limits<double>::infinity() &&
       (std::abs(lim.second) <= relUncert * std::abs(lim.first) /* || std::abs(lim.second)<MIN_LIMIT_UNCERT*/))
      return lim;

   double nextPoint;

   if (lim.second == std::numeric_limits<double>::infinity()) {
      // limit was found by extrapolating to right
      nextPoint = lim.first;
      if (nextPoint == std::numeric_limits<double>::infinity() || nextPoint > maxMu) {
         nextPoint = gr->GetPointX(gr->GetN() - 1) + (maxMu - minMu) / 50;
      }

      // prefer extrapolation with sigma_mu, if available, if it takes us further
      // as shape of p-value curve is usually
      auto point =
         (sOpt.Contains("exp")) ? back().asimov() : std::shared_ptr<xRooHypoPoint>(&back(), [](xRooHypoPoint *) {});
      point = nullptr;
      if (point && point->ufit()) {
         double rough_sigma_mu = point->mu_hat().getError();
         double another_estimate = point->mu_hat().getVal() + rough_sigma_mu * ROOT::Math::gaussian_quantile(0.95, 1);
         // if (another_estimate < nextPoint) {
         nextPoint = std::max(nextPoint, another_estimate);
         ::Info("xRooHypoSpace::findlimit", "Guessing %g based on rough sigma_mu = %g", nextPoint, rough_sigma_mu);
         //}
      }
      nextPoint = std::min(nextPoint + nextPoint * relUncert * 0.99, maxMu); // ensure we step over location if possible

      if (nextPoint > maxMu)
         return lim;
   } else if (lim.second == -std::numeric_limits<double>::infinity()) {
      // limit from extrapolating to left
      nextPoint = lim.first;
      if (nextPoint < minMu)
         nextPoint = gr->GetPointX(0) - (maxMu - minMu) / 50;
      if (nextPoint < minMu)
         return lim;
   } else {
      nextPoint = lim.first + lim.second * relUncert * 0.99;
   }

   // got here need a new point .... evaluate the estimated lim location +/- the relUncert (signed error takes care of
   // direction)

   ::Info("xRooHypoSpace::findlimit", "%s -- Testing new point @ %s=%g (delta=%g)", sOpt.Data(), v->GetName(),
          nextPoint, lim.second);
   if (maxTries == 0 || std::isnan(AddPoint(TString::Format("%s=%g", v->GetName(), nextPoint)).getVal(sOpt).first)) {
      if (maxTries == 0) {
         Warning("findlimit", "Reached max number of point evaluations");
      } else {
         Error("findlimit", "Problem evaluating %s @ %s=%g", sOpt.Data(), v->GetName(), nextPoint);
      }
      return lim;
   }
   gr.reset();
   return findlimit(opt, relUncert, maxTries - 1);
}

void xRooNLLVar::xRooHypoSpace::Draw(Option_t *opt)
{

   TString sOpt(opt);
   sOpt.ToLower();

   if ((sOpt == "" || sOpt == "same") && !empty()) {
      if (front().fPllType == xRooFit::Asymptotics::OneSidedPositive) {
         sOpt += "pcls"; // default to showing cls p-value scan if drawing a limit
         for (auto &hp : *this) {
            if (!hp.nullToys.empty() || !hp.altToys.empty()) {
               sOpt += " toys";
               break; // default to toys if done toys
            }
         }
      } else if (front().fPllType == xRooFit::Asymptotics::TwoSided) {
         sOpt += "ts";
      }
   }

   // split up by ; and call Draw for each (with 'same' appended)
   auto _axes = axes();
   if (_axes.empty())
      return;

   if (sOpt == "status") {
      // draw the points in the space
      if (_axes.size() <= 2) {
         TGraphErrors *out = new TGraphErrors;
         out->SetBit(kCanDelete);
         out->SetName("points");
         out->SetMarkerSize(0.5);
         TGraph *tsAvail = new TGraph;
         tsAvail->SetName("ts");
         tsAvail->SetBit(kCanDelete);
         tsAvail->SetMarkerStyle(20);
         TGraph *expAvail = new TGraph;
         expAvail->SetName("exp");
         expAvail->SetBit(kCanDelete);
         expAvail->SetMarkerStyle(25);
         expAvail->SetMarkerSize(out->GetMarkerSize() * 1.5);
         TGraph *badPoints = new TGraph;
         badPoints->SetName("bad_ufit");
         badPoints->SetBit(kCanDelete);
         badPoints->SetMarkerStyle(5);
         badPoints->SetMarkerColor(kRed);
         badPoints->SetMarkerSize(out->GetMarkerSize());
         TGraph *badPoints2 = new TGraph;
         badPoints2->SetName("bad_cfit_null");
         badPoints2->SetBit(kCanDelete);
         badPoints2->SetMarkerStyle(2);
         badPoints2->SetMarkerColor(kRed);
         badPoints2->SetMarkerSize(out->GetMarkerSize());

         out->SetTitle(TString::Format("%s;%s;%s", GetTitle(), _axes.at(0)->GetTitle(),
                                       (_axes.size() == 1) ? "" : _axes.at(1)->GetTitle()));
         for (auto &p : *this) {
            bool _readOnly = p.nllVar ? p.nllVar->get()->getAttribute("readOnly") : false;
            if (p.nllVar)
               p.nllVar->get()->setAttribute("readOnly", true);
            double x = p.coords->getRealValue(_axes.at(0)->GetName());
            double y = _axes.size() == 1 ? p.ts_asymp().first : p.coords->getRealValue(_axes.at(1)->GetName());
            out->SetPoint(out->GetN(), x, y);
            if (!std::isnan(p.ts_asymp().first)) {
               if (_axes.size() == 1)
                  out->SetPointError(out->GetN() - 1, 0, p.ts_asymp().second);
               tsAvail->SetPoint(tsAvail->GetN(), x, y);
            } else if (p.fUfit && (std::isnan(p.fUfit->minNll()) ||
                                   xRooNLLVar::xRooHypoPoint::allowedStatusCodes.find(p.fUfit->status()) ==
                                      xRooNLLVar::xRooHypoPoint::allowedStatusCodes.end())) {
               badPoints->SetPoint(badPoints->GetN(), x, y);
            } else if (p.fNull_cfit && (std::isnan(p.fNull_cfit->minNll()) ||
                                        xRooNLLVar::xRooHypoPoint::allowedStatusCodes.find(p.fNull_cfit->status()) ==
                                           xRooNLLVar::xRooHypoPoint::allowedStatusCodes.end())) {
               badPoints2->SetPoint(badPoints2->GetN(), x, y);
            }
            if (!std::isnan(p.ts_asymp(0).first)) {
               expAvail->SetPoint(expAvail->GetN(), x, y);
            } else if (p.asimov() && p.asimov()->fUfit &&
                       (std::isnan(p.asimov()->fUfit->minNll()) ||
                        xRooNLLVar::xRooHypoPoint::allowedStatusCodes.find(p.asimov()->fUfit->status()) ==
                           xRooNLLVar::xRooHypoPoint::allowedStatusCodes.end())) {

            } else if (p.asimov() && p.asimov()->fNull_cfit &&
                       (std::isnan(p.asimov()->fNull_cfit->minNll()) ||
                        xRooNLLVar::xRooHypoPoint::allowedStatusCodes.find(p.asimov()->fNull_cfit->status()) ==
                           xRooNLLVar::xRooHypoPoint::allowedStatusCodes.end())) {
            }
            if (p.nllVar)
               p.nllVar->get()->setAttribute("readOnly", _readOnly);
         }

         if (_axes.size() == 1) {
            TGraph tmp;
            for (int i = 0; i < out->GetN(); i++) {
               if (!std::isnan(out->GetPointY(i)))
                  tmp.SetPoint(tmp.GetN(), out->GetPointX(i), out->GetPointY(i));
            }
            auto fixPoints = [&](TGraph *g) {
               for (int i = 0; i < g->GetN(); i++) {
                  if (std::isnan(g->GetPointY(i)))
                     g->SetPointY(i, std::isnan(tmp.Eval(g->GetPointX(i))) ? 0. : tmp.Eval(g->GetPointX(i)));
               }
            };
            fixPoints(out);
            fixPoints(tsAvail);
            fixPoints(expAvail);
            fixPoints(badPoints);
            fixPoints(badPoints2);
         }

         out->SetMarkerStyle(4);
         out->Draw("AP");
         auto leg = new TLegend(1. - gPad->GetRightMargin() - 0.3, 1. - gPad->GetTopMargin() - 0.35,
                                1. - gPad->GetRightMargin() - 0.05, 1. - gPad->GetTopMargin() - 0.05);
         leg->SetName("legend");
         leg->AddEntry(out, "Uncomputed", "P");

         if (tsAvail->GetN()) {
            out->GetListOfFunctions()->Add(tsAvail, "P");
            leg->AddEntry(tsAvail, "Computed", "P");
         } else {
            delete tsAvail;
         }
         if (expAvail->GetN()) {
            out->GetListOfFunctions()->Add(expAvail, "P");
            leg->AddEntry(expAvail, "Expected computed", "P");
         } else {
            delete expAvail;
         }
         if (badPoints->GetN()) {
            out->GetListOfFunctions()->Add(badPoints, "P");
            leg->AddEntry(badPoints, "Bad ufit", "P");
         } else {
            delete badPoints;
         }
         if (badPoints2->GetN()) {
            out->GetListOfFunctions()->Add(badPoints2, "P");
            leg->AddEntry(badPoints2, "Bad null cfit", "P");
         } else {
            delete badPoints2;
         }
         leg->SetBit(kCanDelete);
         leg->Draw();
         // if(_axes.size()==1) out->GetHistogram()->GetYaxis()->SetRangeUser(0,1);
         gPad->SetGrid(true, _axes.size() > 1);
         if (_axes.size() == 1)
            gPad->SetLogy(false);
      }

      gSystem->ProcessEvents();

      return;
   }

   if (sOpt.Contains("pcls") || sOpt.Contains("pnull") || sOpt.Contains("ts")) {
      auto gra = graphs(sOpt + " readonly");
      if (!gPad && gROOT->GetSelectedPad())
         gROOT->GetSelectedPad()->cd();
      if (!sOpt.Contains("same") && gPad) {
         gPad->Clear();
      }
      if (gra) {
         auto gra2 = static_cast<TMultiGraph *>(gra->DrawClone(sOpt.Contains("same") ? "" : "A"));
         gra2->SetBit(kCanDelete);
         if (sOpt.Contains("pcls") || sOpt.Contains("pnull")) {
            gra2->GetHistogram()->SetMinimum(1e-6);
         }
         if (gPad) {
            gPad->RedrawAxis();
            gPad->GetCanvas()->Paint();
            gPad->GetCanvas()->Update();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
            gPad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
#endif
            gSystem->ProcessEvents();
         }
      }
      if (!sOpt.Contains("same") && gPad) {
         //         auto mg = static_cast<TMultiGraph*>(gPad->GetPrimitive(gra->GetName()));
         //         mg->GetHistogram()->SetMinimum(1e-9);
         //         mg->GetHistogram()->GetYaxis()->SetRangeUser(1e-9,1);
         //         gPad->SetGrid(0, 0);
         //         gPad->SetLogy(1);
      }

      gSystem->ProcessEvents();

      return;
   }

   // graphs("ts visualize");return;

   TGraphErrors *out = new TGraphErrors;
   out->SetName(GetName());

   TString title = (!axes().empty()) ? TString::Format(";%s", axes().first()->GetTitle()) : "";

   auto pllType = xRooFit::Asymptotics::TwoSided;
   if (!empty() && axes().size() == 1) {
      for (auto &p : *this) {
         if (p.fPllType != xRooFit::Asymptotics::TwoSided) {
            pllType = p.fPllType;
         }
      }
      title += ";";
      title += front().tsTitle(true);
   }

   out->SetTitle(title);
   *dynamic_cast<TAttFill *>(out) = *this;
   *dynamic_cast<TAttLine *>(out) = *this;
   *dynamic_cast<TAttMarker *>(out) = *this;
   out->SetBit(kCanDelete);

   if (!gPad)
      TCanvas::MakeDefCanvas();
   TVirtualPad *basePad = gPad;
   if (!sOpt.Contains("same"))
      basePad->Clear();

   bool doFits = false;
   if (sOpt.Contains("fits")) {
      doFits = true;
      sOpt.ReplaceAll("fits", "");
   }

   auto mainPad = gPad;

   out->SetEditable(false);

   if (doFits) {
      gPad->Divide(1, 2);
      gPad->cd(1);
      gPad->SetBottomMargin(gPad->GetBottomMargin() * 2.); // increase margin to be same as before
      gPad->SetGrid();
      out->Draw(sOpt);
      basePad->cd(2);
      mainPad = basePad->GetPad(1);
   } else {
      gPad->SetGrid();
      out->Draw("ALP");
   }

   std::pair<double, double> minMax(std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity());
   for (auto &p : *this) {
      if (p.fPllType != pllType)
         continue; // must all have same pll type
      auto val = p.pll(true).first;
      if (std::isnan(val))
         continue;
      minMax.first = std::min(minMax.first, val);
      minMax.second = std::max(minMax.second, val);
   }
   if (minMax.first < std::numeric_limits<double>::infinity())
      out->GetHistogram()->SetMinimum(minMax.first);
   if (minMax.second > -std::numeric_limits<double>::infinity())
      out->GetHistogram()->SetMaximum(minMax.second);

   TGraph *badPoints = nullptr;

   TStopwatch s;
   s.Start();
   std::shared_ptr<const RooFitResult> ufr;
   for (auto &p : *this) {
      if (p.fPllType != pllType)
         continue; // must all have same pll type
      auto val = p.pll().first;
      if (!ufr)
         ufr = p.ufit();
      if (out->GetN() == 0 && ufr && ufr->status() == 0) {
         out->SetPoint(out->GetN(),
                       ufr->floatParsFinal().getRealValue(axes().first()->GetName(),
                                                          ufr->constPars().getRealValue(axes().first()->GetName())),
                       0.);
         out->SetPointError(out->GetN() - 1, 0, ufr->edm());
      }
      if (auto fr = p.fNull_cfit;
          fr && doFits) { // access member to avoid unnecessarily creating fit result if wasnt needed
         // create a new subpad and draw fitResult on it
         auto _pad = gPad;
         auto pad =
            new TPad(fr->GetName(), TString::Format("%s = %g", poi().first()->GetTitle(), p.fNullVal()), 0, 0, 1., 1);
         pad->SetNumber(out->GetN() + 1); // can't use "0" for a subpad
         pad->cd();
         xRooNode(fr).Draw("goff");
         _pad->cd();
         //_pad->GetListOfPrimitives()->AddFirst(pad);
         pad->AppendPad();
      }
      if (std::isnan(val) && p.status() != 0) {
         if (!badPoints) {
            badPoints = new TGraph;
            badPoints->SetBit(kCanDelete);
            badPoints->SetName("badPoints");
            badPoints->SetMarkerStyle(5);
            badPoints->SetMarkerColor(kRed);
            badPoints->SetMarkerSize(1);
            out->GetListOfFunctions()->Add(badPoints, "P");
         }
         badPoints->SetPoint(badPoints->GetN(), p.fNullVal(), out->Eval(p.fNullVal()));
         mainPad->Modified();
      } else if (!std::isnan(val)) {
         out->SetPoint(out->GetN(), p.coords->getRealValue(axes().first()->GetName()), p.pll().first);
         out->SetPointError(out->GetN() - 1, 0, p.pll().second);
         out->Sort();

         // reposition bad points
         if (badPoints) {
            for (int i = 0; i < badPoints->GetN(); i++)
               badPoints->SetPointY(i, out->Eval(badPoints->GetPointX(i)));
         }

         mainPad->Modified();
      }
      if (s.RealTime() > 3) { // stops the clock
         basePad->GetCanvas()->Paint();
         basePad->GetCanvas()->Update();
         gSystem->ProcessEvents();
         s.Reset();
         s.Start();
      }
      s.Continue();
   }
   basePad->GetCanvas()->Paint();
   basePad->GetCanvas()->Update();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
   basePad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
#endif
   gSystem->ProcessEvents();

   // finish by overlaying ufit
   if (ufr && doFits) {
      auto _pad = gPad;
      auto pad = new TPad(ufr->GetName(), "unconditional fit", 0, 0, 1., 1.);
      pad->SetNumber(-1);
      pad->cd();
      xRooNode(ufr).Draw("goff");
      _pad->cd();
      pad->AppendPad();

      // draw one more pad to represent the selected, and draw the ufit pad onto that pad
      pad = new TPad("selected", "selected", 0, 0, 1, 1);
      pad->Draw();
      pad->cd();
      basePad->GetPad(2)->GetPad(-1)->AppendPad();
      pad->Modified();
      pad->Update();
      gSystem->ProcessEvents();

      basePad->cd();
   }

   if (doFits) {
      if (!xRooNode::gIntObj) {
         xRooNode::gIntObj = new xRooNode::InteractiveObject;
      }
      gPad->GetCanvas()->Connect("Highlighted(TVirtualPad*,TObject*,Int_t,Int_t)", "xRooNode::InteractiveObject",
                                 xRooNode::gIntObj, "Interactive_PLLPlot(TVirtualPad*,TObject*,Int_t,Int_t)");
   }

   return;
}

RooStats::HypoTestInverterResult *xRooNLLVar::xRooHypoSpace::result()
{

   RooStats::HypoTestInverterResult *out = nullptr;

   auto _axes = axes();
   if (_axes.empty())
      return out;

   out = new RooStats::HypoTestInverterResult(GetName(), *dynamic_cast<RooRealVar *>(_axes.at(0)), 0.95);
   out->SetTitle(GetTitle());

   for (auto &p : *this) {
      double _x = p.coords->getRealValue(_axes.at(0)->GetName(), std::numeric_limits<double>::quiet_NaN());
      out->Add(_x, p.result());
   }

   return out;
}

END_XROOFIT_NAMESPACE
