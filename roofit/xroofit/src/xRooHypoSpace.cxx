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
#include "RooDataSet.h"
#include "TKey.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLine.h"
#include "RooStats/HypoTestInverterResult.h"

BEGIN_XROOFIT_NAMESPACE

// bool xRooNLLVar::xRooHypoSpace::AddWorkspace(const char* wsFilename, const char* extraPars){
//
//     auto ws = std::make_shared<xRooNode>(wsFilename);
//
//     ws->browse();
//     std::set<std::shared_ptr<xRooNode>> models;
//     for(auto n : *ws) {
//         if (n->fFolder == "!models") models.insert(n);
//     }
//     if (models.size()!=1) {
//         throw std::runtime_error("More than one model in workspace, use AddModel instead");
//     }
//
//     auto out =  AddModel(*models.begin(),extraPars);
//     if (out) {
//         fWorkspaces.insert(ws); // keep ws open
//     }
//
// }

xRooNLLVar::xRooHypoSpace::xRooHypoSpace(const char *name, const char *title)
   : TNamed(name, title), fPars(std::make_shared<RooArgSet>())
{
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

   if (nPoints == 1) {
      _par->setVal((high + low) * 0.5);
      AddPoint();
      return nPoints;
   }

   double step = (high - low) / nPoints;
   if (step <= 0)
      throw std::runtime_error("Invalid steps");

   for (double v = low + step * 0.5; v <= high; v += step) {
      _par->setVal(v);
      AddPoint();
   }
   return nPoints;
}

double round_to_digits(double value, int digits)
{
   if (value == 0.0)
      return 0.0;
   double factor = pow(10.0, digits - ceil(log10(std::abs(value))));
   return std::round(value * factor) / factor;
};
double round_to_decimal(double value, int decimal_places)
{
   const double multiplier = std::pow(10.0, decimal_places);
   return std::round(value * multiplier) / multiplier;
}

// rounds error to 1 or 2 sig fig and round value to match that precision
std::pair<double, double> matchPrecision(const std::pair<double, double> &in)
{
   auto out = in;
   if (!std::isinf(out.second)) {
      auto tmp = out.second;
      out.second = round_to_digits(out.second, 2);
      int expo = (out.second == 0) ? 0 : (int)std::floor(std::log10(std::abs(out.second)));
      if (TString::Format("%e", out.second)(0) != '1') {
         out.second = round_to_digits(tmp, 1);
         out.first = (expo >= 0) ? round(out.first) : round_to_decimal(out.first, -expo);
      } else if (out.second != 0) {
         out.first = (expo >= 0) ? round(out.first) : round_to_decimal(out.first, -expo + 1);
      }
   }
   return out;
}

std::map<std::string, std::pair<double, double>> xRooNLLVar::xRooHypoSpace::limits(const char *opt, double relUncert)
{
   TString sOpt(opt);
   if (sOpt.Contains("cls")) {
      for (auto p : poi()) {
         if (!p->hasRange("physical")) {
            Info("limits", "No physical range set for %s, setting to [0,inf]", p->GetName());
            dynamic_cast<RooRealVar *>(p)->setRange("physical", 0, std::numeric_limits<double>::infinity());
         }
         if (!p->getStringAttribute("altVal")) {
            Info("limits", "No altVal set for %s, setting to 0", p->GetName());
            p->setStringAttribute("altVal", "0");
         }
         // ensure range straddles altVal
         double altVal = TString(p->getStringAttribute("altVal")).Atof();
         auto v = dynamic_cast<RooRealVar *>(p);
         if (v->getMin() >= altVal) {
            Info("limits", "range of POI does not straddle alt value, adjusting minimum to %g", altVal - 1e-5);
            v->setMin(altVal - 1e-5);
         }
         if (v->getMax() <= altVal) {
            Info("limits", "range of POI does not straddle alt value, adjusting minimum to %g", altVal + 1e-5);
            v->setMax(altVal + 1e-5);
         }
         for (auto &[pdf, nll] : fNlls) {
            if (auto _v = dynamic_cast<RooRealVar *>(nll->pars()->find(*p))) {
               _v->setRange(v->getMin(), v->getMax());
            }
         }
      }
   }

   std::map<std::string, std::pair<double, double>> out;
   std::shared_ptr<TMemFile> memFile;
   if (!gDirectory->IsWritable()) {
      memFile = std::make_shared<TMemFile>("memory", "RECREATE");
   }
   for (int nSigma : {0, 1, 2, -1, -2}) {
      auto lim = FindLimit(TString::Format("%s exp%s%d", opt, nSigma > 0 ? "+" : "", nSigma), relUncert);
      if (lim.second < 0)
         lim.second = -lim.second; // make errors positive for this method
      out[TString::Format("%d", nSigma).Data()] = matchPrecision(lim);
   }

   // don't do the observed limit if all the NLL datas are generated
   bool doObs = true;
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
   bool allGen = true;
   for (auto &[pdf, nll] : fNlls) {
      auto _d = dynamic_cast<RooDataSet *>(nll->data());
      if (!_d || !_d->weightVar() || !_d->weightVar()->getStringAttribute("fitResult")) {
         allGen = false;
         break;
      }
   }
   if (allGen)
      doObs = false;
#endif
   if (doObs) {
      auto lim = FindLimit(TString::Format("%s obs", opt), relUncert);
      if (lim.second < 0)
         lim.second = -lim.second;
      out["obs"] = matchPrecision(lim);
   }
   return out;
}

xRooNLLVar::xRooHypoPoint &xRooNLLVar::xRooHypoSpace::AddPoint(const char *coords)
{
   // move to given coords, if any
   fPars->assignValueOnly(toArgs(coords));

   auto _pdf = pdf();

   if (!_pdf)
      throw std::runtime_error("no model at coordinates");

   if (std::unique_ptr<RooAbsCollection>(fPars->selectByAttrib("poi", true))->size() == 0) {
      throw std::runtime_error(
         "No pars designated as POI - set with pars()->find(<parName>)->setAttribute(\"poi\",true)");
   }

   if (fNlls.find(_pdf) == fNlls.end()) {
      fNlls[_pdf] = std::make_shared<xRooNLLVar>(_pdf->nll("" /*TODO:allow change dataset name and nll opts*/, {}));
   }

   xRooHypoPoint out;

   out.nllVar = fNlls[_pdf];
   out.data = fNlls[_pdf]->getData();

   out.coords.reset(fPars->snapshot()); // should already have altVal prop on poi, and poi labelled
   // ensure all poi are marked const ... required by xRooHypoPoint behaviour
   out.poi().setAttribAll("Constant");
   // and now remove anything that's marked floating
   const_cast<RooAbsCollection *>(out.coords.get())
      ->remove(*std::unique_ptr<RooAbsCollection>(out.coords->selectByAttrib("Constant", false)), true, true);
   double value = out.fNullVal();
   double alt_value = out.fAltVal();

   auto _type = fTestStatType;
   if (_type == xRooFit::Asymptotics::Unknown) {
      // decide based on values
      if (std::isnan(alt_value))
         _type = xRooFit::Asymptotics::TwoSided;
      else if (value >= alt_value)
         _type = xRooFit::Asymptotics::OneSidedPositive;
      else
         _type = xRooFit::Asymptotics::Uncapped;
   }

   out.fPllType = _type;

   // TODO: Check for equivalent point before adding

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

   bool clash;
   do {
      clash = false;

      // add next best coordinate
      std::map<std::string, std::set<double>> values;
      for (auto &par : *pars()) {
         if (out.find(*par))
            continue;
         for (auto p : *this) {
            values[par->GetName()].insert(
               p.coords->getRealValue(par->GetName(), std::numeric_limits<double>::quiet_NaN()));
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

      std::set<std::vector<double>> coords;
      for (auto &p : *this) {
         std::vector<double> p_coords;
         for (auto o : out) {
            p_coords.push_back(p.coords->getRealValue(o->GetName(), std::numeric_limits<double>::quiet_NaN()));
         }
         if (coords.find(p_coords) != coords.end()) {
            clash = true;
            break;
         }
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
            auto cl = TClass::GetClass(((TKey *)k)->GetClassName());
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
   Info("xRooHypoSpace", "%s - Loaded %d fits", apath, nFits);

   if (allpois.size() == 1) {
      Info("xRooHypoSpace", "Detected POI: %s", allpois.begin()->c_str());

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
            if (std::isnan(hp.fAltVal()))
               hp.fPllType = xRooFit::Asymptotics::TwoSided;
            else if (hp.fNullVal() >= hp.fAltVal())
               hp.fPllType = xRooFit::Asymptotics::OneSidedPositive;
            else
               hp.fPllType = xRooFit::Asymptotics::Uncapped;

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
      if (!ufit)
         std::cout << "-";
      else {
         std::cout << ufit->status();
         badFits += (xRooNLLVar::xRooHypoPoint::allowedStatusCodes.count(ufit->status()) == 0);
      }
      std::cout << ",cfit_null:";
      auto cfit = const_cast<xRooHypoPoint &>(at(i)).cfit_null(true);
      if (!cfit)
         std::cout << "-";
      else {
         std::cout << cfit->status();
         badFits += (xRooNLLVar::xRooHypoPoint::allowedStatusCodes.count(cfit->status()) == 0);
      }
      std::cout << ",cfit_alt:";
      auto afit = const_cast<xRooHypoPoint &>(at(i)).cfit_alt(true);
      if (!afit)
         std::cout << "-";
      else {
         std::cout << afit->status();
         badFits += (xRooNLLVar::xRooHypoPoint::allowedStatusCodes.count(afit->status()) == 0);
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

std::shared_ptr<TGraphErrors> xRooNLLVar::xRooHypoSpace::BuildGraph(const char *opt)
{

   TString sOpt(opt);
   sOpt.ToLower();

   bool doCLs = sOpt.Contains("cls");

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
      if (sOpt.Contains("ts"))
         out->SetNameTitle("obs_ts", TString::Format("Observed;%s;Test Statistic", _axes.at(0)->GetTitle()));
   } else {
      out->SetNameTitle(TString::Format("exp%d_p%s", int(nSigma), sCL), title);
      out->SetMarkerStyle(0);
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
      if (sOpt.Contains("ts"))
         out->SetNameTitle(TString::Format("exp_ts%d", int(nSigma)),
                           TString::Format("Expected;%s;Test Statistic", _axes.at(0)->GetTitle()));
   }

   auto badPoints = [&]() {
      auto badPoints2 = dynamic_cast<TGraph *>(out->GetListOfFunctions()->FindObject("badPoints"));
      if (!badPoints2) {
         badPoints2 = new TGraph;
         badPoints2->SetBit(kCanDelete);
         badPoints2->SetName("badPoints");
         badPoints2->SetMarkerStyle(5);
         badPoints2->SetMarkerColor(kRed);
         badPoints2->SetMarkerSize(out->GetMarkerSize());
         out->GetListOfFunctions()->Add(badPoints2, "P");
      }
      return badPoints2;
   };
   int nPointsDown = 0;
   bool above = true;
   for (auto &p : *this) {
      double _x = p.coords->getRealValue(_axes.at(0)->GetName(), std::numeric_limits<double>::quiet_NaN());
      auto pval = p.getVal(sOpt);
      auto idx = out->GetN() - nPointsDown;

      if (std::isnan(pval.first)) {
         badPoints()->SetPoint(badPoints()->GetN(), _x, 0);
      } else {
         out->InsertPointBefore(idx, _x, pval.first);
         out->SetPointError(idx, 0, pval.second);
      }

      if (expBand && nSigma) {
         TString sOpt2 = sOpt;
         sOpt2.ReplaceAll("exp", "exp-");
         pval = p.getVal(sOpt2);
         if (std::isnan(pval.first)) {
            badPoints()->SetPoint(badPoints()->GetN(), _x, 0);
         } else {
            out->InsertPointBefore(idx + 1, _x, pval.first);
            out->SetPointError(idx + 1, 0, pval.second);
            nPointsDown++;
            if (out->GetPointY(idx) < pval.first)
               above = false; // the -sigma points are actually above +sigma
         }
      }
   }

   if (out->GetN() == 0)
      return out;

   if (!expBand) {
      out->Sort();
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

   return out;
}

std::pair<double, double> xRooNLLVar::xRooHypoSpace::GetLimit(const TGraph &pValues, double target)
{

   auto gr = std::make_shared<TGraph>(pValues);
   // remove any nan points and duplicates
   int i = 0;
   std::set<double> existingX;
   while (i < gr->GetN()) {
      if (std::isnan(gr->GetPointY(i)))
         gr->RemovePoint(i);
      else if (existingX.find(gr->GetPointX(i)) != existingX.end()) {
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
      return std::pair(std::numeric_limits<double>::quiet_NaN(), 0);
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
      double x1 = gr->GetPointX(gr->GetN() - 2);
      double y1 = gr->GetPointY(gr->GetN() - 2);
      double m = (gr->GetPointY(gr->GetN() - 1) - y1) / (gr->GetPointX(gr->GetN() - 1) - x1);
      if (m == 0.)
         return std::pair(2. * gr->GetPointX(gr->GetN() - 1) - x1, std::numeric_limits<double>::infinity());
      return std::pair((alpha - y1) / m + x1, std::numeric_limits<double>::infinity());
   } else {
      // extrapolating below based on first two points
      double x1 = gr->GetPointX(0);
      double y1 = gr->GetPointY(0);
      double m = (gr->GetPointY(1) - y1) / (gr->GetPointX(1) - x1);
      if (m == 0.)
         return std::pair(2. * x1 - gr->GetPointX(1), std::numeric_limits<double>::infinity());
      return std::pair((alpha - y1) / m + x1, std::numeric_limits<double>::infinity());
   }
}

std::pair<double, double> xRooNLLVar::xRooHypoSpace::FindLimit(const char *opt, double relUncert)
{
   std::shared_ptr<TGraphErrors> gr = BuildGraph(TString(opt) + " readonly");

   // resync parameter boundaries from nlls (may have been modified by fits)
   for (auto p : poi()) {
      for (auto &[pdf, nll] : fNlls) {
         if (auto _v = dynamic_cast<RooRealVar *>(nll->pars()->find(*p))) {
            dynamic_cast<RooRealVar *>(p)->setRange(_v->getMin(), _v->getMax());
         }
      }
   }

   if (!gr || gr->GetN() < 2) {
      auto v = (poi().empty()) ? nullptr : dynamic_cast<RooRealVar *>(poi().first());
      if (!v)
         return std::pair(std::numeric_limits<double>::quiet_NaN(), 0);
      double muMax = std::min(v->getMax(), v->getMax("physical"));
      double muMin = std::max(v->getMin("physical"), v->getMin());
      if (!gr || gr->GetN() < 1) {
         if (std::isnan(AddPoint(TString::Format("%s=%g", v->GetName(), muMin)).getVal(opt).first)) {
            // first point failed ... give up
            return std::pair(std::numeric_limits<double>::quiet_NaN(), 0);
         }
         return FindLimit(opt, relUncert); // do this to resync parameter limits
      }

      if (std::isnan(
             AddPoint(TString::Format("%s=%g", v->GetName(), muMin + (muMax - muMin) / 50)).getVal(opt).first)) {
         // second point failed ... give up
         return std::pair(std::numeric_limits<double>::quiet_NaN(), 0);
      }
      return FindLimit(opt, relUncert);
   }

   auto lim = GetLimit(*gr);

   if (std::isnan(lim.first)) {
      return lim;
   }

   if (std::abs(lim.second) <= relUncert * std::abs(lim.first))
      return lim;

   double nextPoint;
   auto v = dynamic_cast<RooRealVar *>(poi().first());
   double maxMu = std::min(v->getMax("physical"), v->getMax());
   double minMu = std::max(v->getMin("physical"), v->getMin());
   if (lim.second == std::numeric_limits<double>::infinity()) {
      // limit was found by extrapolating to right
      nextPoint = lim.first;
      if (nextPoint > v->getMax("physical"))
         nextPoint = gr->GetPointX(gr->GetN() - 1) + (maxMu - minMu) / 50;
      if (nextPoint > v->getMax("physical"))
         return lim;
   } else if (lim.second == -std::numeric_limits<double>::infinity()) {
      // limit from extrapolating to left
      nextPoint = lim.first;
      if (nextPoint < v->getMin("physical"))
         nextPoint = gr->GetPointX(0) - (maxMu - minMu) / 50;
      if (nextPoint < v->getMin("physical"))
         return lim;
   } else {
      nextPoint = lim.first + lim.second * relUncert * 0.99;
   }

   // got here need a new point .... evaluate the estimated lim location +/- the relUncert (signed error takes care of
   // direction)

   Info("FindLimit", "%s -- Testing new point @ %s=%g", opt, v->GetName(), nextPoint);
   if (std::isnan(AddPoint(TString::Format("%s=%g", v->GetName(), nextPoint)).getVal(opt).first)) {
      return lim;
   }

   return FindLimit(opt, relUncert);
}

void xRooNLLVar::xRooHypoSpace::Draw(Option_t *opt)
{

   TString sOpt(opt);
   sOpt.ToLower();

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
         auto leg = new TLegend(1. - gPad->GetRightMargin() - 0.3, 1. - gPad->GetTopMargin() - 0.3,
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

      return;
   }

   if (sOpt.Contains("pcls") || sOpt.Contains("pnull")) {
      // bool doCLs = (sOpt.Contains("cls"));
      // const char* sCL = (doCLs) ? "CLs" : "null";

      auto exp2 = BuildGraph(sOpt + " exp2");
      auto exp1 = BuildGraph(sOpt + " exp1");
      auto exp = BuildGraph(sOpt + " exp");
      auto obs = BuildGraph(sOpt);

      if (!sOpt.Contains("same") && gPad) {
         gPad->Clear();
      }
      auto g = dynamic_cast<TGraphErrors *>(exp2->DrawClone("AF"));
      g->SetBit(kCanDelete);
      g->GetHistogram()->SetName(".axis");
      g->GetHistogram()->SetTitle("");
      g->GetHistogram()->SetBit(TH1::kNoTitle);
      exp2->DrawClone("F")->SetBit(kCanDelete);
      exp1->DrawClone("F")->SetBit(kCanDelete);
      exp->DrawClone("LP")->SetBit(kCanDelete);
      obs->DrawClone("LP")->SetBit(kCanDelete);
      TLine l;
      l.SetLineStyle(2);
      l.DrawLine(g->GetHistogram()->GetXaxis()->GetXmin(), 0.05, g->GetHistogram()->GetXaxis()->GetXmax(), 0.05);
      // auto l = gPad->BuildLegend(gPad->GetLeftMargin()+0.05,
      // gPad->GetBottomMargin()+0.05,gPad->GetLeftMargin()+0.35,gPad->GetBottomMargin()+0.25);l->SetName("legend");

      auto leg = new TLegend(1. - gPad->GetRightMargin() - 0.3, 1. - gPad->GetTopMargin() - 0.3,
                             1. - gPad->GetRightMargin() - 0.05, 1. - gPad->GetTopMargin() - 0.05);
      leg->SetName("legend");
      leg->AddEntry(g->GetListOfFunctions()->FindObject("down"), "", "F");
      leg->AddEntry(
         dynamic_cast<TGraph *>(gPad->GetPrimitive(exp1->GetName()))->GetListOfFunctions()->FindObject("down"), "",
         "F");
      leg->AddEntry(gPad->GetPrimitive(exp->GetName()), "", "LPE");
      leg->AddEntry(gPad->GetPrimitive(obs->GetName()), "", "LPE");
      leg->Draw();
      leg->SetBit(kCanDelete);

      g->GetHistogram()->Draw("sameaxis"); // redraw axis

      if (!sOpt.Contains("same")) {
         gPad->SetGrid(0, 0);
         gPad->SetLogy(1);
      }

      gSystem->ProcessEvents();

      return;
   }

   TGraphErrors *out = new TGraphErrors;
   out->SetName(GetName());

   TString title = TString::Format(";%s", poi().first()->GetTitle());

   auto pllType = xRooFit::Asymptotics::TwoSided;
   if (!empty() && axes().size() == 1) {
      auto v = dynamic_cast<RooRealVar *>(axes().first());
      for (auto &p : *this) {
         if (p.fPllType != xRooFit::Asymptotics::TwoSided) {
            pllType = p.fPllType;
         }
      }
      if (pllType == xRooFit::Asymptotics::OneSidedPositive) {
         if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity())
            title += TString::Format(";Lower-Bound One-Sided Limit PLR");
         else if (v)
            title += TString::Format(";One-Sided Limit PLR");
         else
            title += ";q";
      } else if (pllType == xRooFit::Asymptotics::TwoSided) {
         if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity())
            title += TString::Format(";Lower-Bound PLR");
         else if (v)
            title += TString::Format(";PLR");
         else
            title += ";t";
      } else if (pllType == xRooFit::Asymptotics::OneSidedNegative) {
         if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity())
            title += TString::Format(";Lower-Bound One-Sided Discovery PLR");
         else if (v)
            title += TString::Format(";One-Sided Discovery PLR");
         else
            title += ";r";
      } else if (pllType == xRooFit::Asymptotics::Uncapped) {
         if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity())
            title += TString::Format(";Lower-Bound Uncapped PLR");
         else if (v)
            title += TString::Format(";Uncapped PLR");
         else
            title += ";s";
      } else {
         title += ";Test Statistic";
      }
   }

   out->SetTitle(title);
   *dynamic_cast<TAttFill *>(out) = *this;
   *dynamic_cast<TAttLine *>(out) = *this;
   *dynamic_cast<TAttMarker *>(out) = *this;
   out->SetBit(kCanDelete);

   if (!gPad)
      TCanvas::MakeDefCanvas();
   auto basePad = gPad;
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
      auto val = p.pll().first;
      minMax.first = std::min(minMax.first, val);
      minMax.second = std::max(minMax.second, val);
   }
   out->GetHistogram()->SetMinimum(minMax.first);
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
      if (std::isnan(val)) {
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
      } else {
         out->SetPoint(out->GetN(), p.fNullVal(), p.pll().first);
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
         basePad->Update();
         gSystem->ProcessEvents();
         s.Reset();
         s.Start();
      }
      s.Continue();
   }

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

   if (!xRooNode::gIntObj) {
      xRooNode::gIntObj = new xRooNode::InteractiveObject;
   }
   gPad->GetCanvas()->Connect("Highlighted(TVirtualPad*,TObject*,Int_t,Int_t)", "xRooNode::InteractiveObject",
                              xRooNode::gIntObj, "Interactive_PLLPlot(TVirtualPad*,TObject*,Int_t,Int_t)");

   return;
}

RooStats::HypoTestInverterResult *xRooNLLVar::xRooHypoSpace::result()
{

   RooStats::HypoTestInverterResult *out = nullptr;

   auto _axes = axes();
   if (_axes.empty())
      return out;

   out = new RooStats::HypoTestInverterResult(GetName(), *dynamic_cast<RooRealVar *>(_axes.at(0)), 0.05);

   for (auto &p : *this) {
      double _x = p.coords->getRealValue(_axes.at(0)->GetName(), std::numeric_limits<double>::quiet_NaN());
      out->Add(_x, p.result());
   }

   return out;
}

END_XROOFIT_NAMESPACE
