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

/** \class ROOT::Experimental::XRooFit::xRooNode
\ingroup xroofit

The xRooNode class is designed to wrap over a TObject and provide functionality to aid with interacting with that
object, particularly in the case where the object is a RooFit class instance. It is a smart pointer to the object, so
you have access to all the methods of the object too.

xRooNode is designed to work in both python and C++, but examples below are given in python because that is imagined
  be the most common way to use the xRooFit API.

-# [Exploring workspaces](\ref exploring-workspaces)

\anchor exploring-workspaces
## Exploring workspaces

An existing workspace file (either a ROOT file containing a RooWorkspace, or a json HS3 file) can be opened using
 xRooNode like this:

\code{.py}
from ROOT.Experimental import XRooFit
w = XRooFit.xRooNode("workspace.root") # or can use workspace.json for HS3
\endcode

 You can explore the content of the workspace somewhat like you would a file system: each node contains sub-nodes,
 which you can interact with to explore ever deeper. The most relevant methods for navigating the workspace and
exploring the content are:




 */

#include "RVersion.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)

#define protected public
#include "TRootBrowser.h"
#include "RooStats/HistFactory/ParamHistFunc.h"
#define private public
#include "RooAbsArg.h"
#include "RooWorkspace.h"
#include "RooStats/HistFactory/PiecewiseInterpolation.h"
#include "RooStats/HistFactory/FlexibleInterpVar.h"
#include "RooProdPdf.h"
#include "TGFileBrowser.h"
#include "RooFitResult.h"
#include "TPad.h"
#undef private
#include "RooAddPdf.h"
#include "RooRealSumPdf.h"
#include "RooProduct.h"
#include "RooHistFunc.h"
#include "RooConstVar.h"
#include "RooSimultaneous.h"
#undef protected

#define GETWS(a) a->_myws
#define GETWSSETS(w) w->_namedSets
#define GETWSSNAPSHOTS(w) w->_snapshots
#define GETACTBROWSER(b) b->fActBrowser
#define GETROOTDIR(b) b->fRootDir
#define GETLISTTREE(b) b->fListTree
#define GETDMP(o, m) o->m

#else

#include "RooAbsArg.h"
#include "RooWorkspace.h"
#include "RooFitResult.h"
#include "RooConstVar.h"
#include "RooHistFunc.h"
#include "RooRealSumPdf.h"
#include "RooSimultaneous.h"
#include "RooAddPdf.h"
#include "RooProduct.h"
#include "TPad.h"
#include "RooStats/HistFactory/PiecewiseInterpolation.h"
#include "RooStats/HistFactory/FlexibleInterpVar.h"
#include "RooStats/HistFactory/ParamHistFunc.h"
#include "RooProdPdf.h"
#include "TRootBrowser.h"
#include "TGFileBrowser.h"
#include "TF1.h"
#include "TFitParametersDialog.h"

RooWorkspace *GETWS(RooAbsArg *a)
{
   return a->workspace();
}
const auto &GETWSSETS(RooWorkspace *w)
{
   return w->sets();
}
auto &GETWSSNAPSHOTS(RooWorkspace *w)
{
   return w->getSnapshots();
}
auto GETACTBROWSER(TRootBrowser *b)
{
   return b->GetActBrowser();
}
auto GETROOTDIR(TGFileBrowser *b)
{
   return b->GetRootDir();
}
auto GETLISTTREE(TGFileBrowser *b)
{
   return b->GetListTree();
}
#define GETDMP(o, m) \
   *reinterpret_cast<void **>(reinterpret_cast<unsigned char *>(o) + o->Class()->GetDataMemberOffset(#m))

#endif

#include "RooAddition.h"

#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooStringVar.h"
#include "RooBinning.h"
#include "RooUniformBinning.h"

#include "RooAbsData.h"
#include "RooDataHist.h"
#include "RooDataSet.h"

#include "xRooFit/xRooNode.h"
#include "xRooFit/xRooFit.h"

#include "TH1.h"
#include "TBrowser.h"
#include "TROOT.h"
#include "TQObject.h"
#include "TAxis.h"
#include "TGraphAsymmErrors.h"
#include "TMath.h"
#include "TPRegexp.h"
#include "TRegexp.h"
#include "TExec.h"
#include "TPaveText.h"

#include "TGListTree.h"
#include "TGMsgBox.h"
#include "TGedEditor.h"
#include "TGMimeTypes.h"
#include "TH2.h"
#include "RooExtendPdf.h"
#include "RooExtendedBinding.h"

#include "RooStats/HypoTestInverterResult.h"

#include "coutCapture.h"

// #include "RooFitTrees/RooFitResultTree.h"
// #include "RooFitTrees/RooDataTree.h"
#include "TFile.h"
#include "TSystem.h"
#include "TKey.h"
#include "TEnv.h"
#include "TStyle.h"

#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
#include "RooFitHS3/RooJSONFactoryWSTool.h"
#endif

#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 24, 00)
#include "RooBinSamplingPdf.h"
#endif

#include "RooPoisson.h"
#include "RooGaussian.h"
#include "RooFormulaVar.h"
#include "RooGenericPdf.h"
#include "TVectorD.h"
#include "TStopwatch.h"
#include "TTimeStamp.h"

#include <csignal>

#include "TCanvas.h"
#include "THStack.h"

#include "TLegend.h"
#include "TLegendEntry.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TFrame.h"
#include "RooProjectedPdf.h"
#include "TMemFile.h"
#include "TGaxis.h"
#include "TPie.h"
// #include <thread>
// #include <future>

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
#include "RooNaNPacker.h"
#endif

BEGIN_XROOFIT_NAMESPACE

xRooNode::InteractiveObject *xRooNode::gIntObj = nullptr;
std::map<std::string, std::tuple<std::function<double(double, double, double)>, bool>> xRooNode::auxFunctions;
void xRooNode::SetAuxFunction(const char *title, const std::function<double(double, double, double)> &func,
                              bool symmetrize)
{
   auxFunctions[title] = std::make_tuple(func, symmetrize);
}

template <typename T>
const T &_or_func(const T &a, const T &b)
{
   if (a)
      return a;
   return b;
}

////////////////////////////////////////////////////////////////////////////////
/// Create new object of type classname, with given name and title, and own-wrap it
/// i.e. the xRooNode will delete the object when the node (and any that reference it) is destroyed
///
/// \param classname : the type of the object to create
/// \param name : the name to give the object
/// \param title : the title to give the object

xRooNode::xRooNode(const char *classname, const char *name, const char *title)
   : xRooNode(name, std::shared_ptr<TObject>(TClass::GetClass(classname)
                                                ? reinterpret_cast<TObject *>(TClass::GetClass(classname)->New())
                                                : nullptr,
                                             [](TObject *o) {
                                                if (auto w = dynamic_cast<RooWorkspace *>(o); w) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
                                                   w->_embeddedDataList.Delete();
#endif
                                                   xRooNode(*w, std::make_shared<xRooNode>()).sterilize();
                                                }
                                                if (o)
                                                   delete o;
                                             }))
{
   if (auto a = get<TNamed>(); a)
      a->SetName(name);
   SetTitle(title);
}

xRooNode::xRooNode(const char *name, const std::shared_ptr<TObject> &comp, const std::shared_ptr<xRooNode> &parent)
   : TNamed(name, ""), fComp(comp), fParent(parent)
{

   if (!fComp && !fParent && name && strlen(name) > 0) {
      char *_path = gSystem->ExpandPathName(name);
      TString pathName = TString(_path);
      delete[] _path;
      if (!gSystem->AccessPathName(pathName)) {
         // if file is json can try to read
         if (pathName.EndsWith(".json")) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
            fComp = std::make_shared<RooWorkspace>("workspace", name);
            RooJSONFactoryWSTool tool(*get<RooWorkspace>());
            RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
            RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);
            if (!tool.importJSON(pathName.Data())) {
               Error("xRooNode", "Error reading json workspace %s", name);
               fComp.reset();
            }
            RooMsgService::instance().setGlobalKillBelow(msglevel);
#else
            Error("xRooNode", "json format workspaces available only in ROOT 6.26 onwards");
#endif
         } else {

            // using acquire in the constructor seems to cause a mem leak according to valgrind ... possibly because
            // (*this) gets called on it before the node is fully constructed
            auto _file = std::make_shared<TFile>(
               pathName); // acquire<TFile>(name); // acquire file to ensure stays open while we have the workspace
            // actually it appears we don't need to keep the file open once we've loaded the workspace, but should be
            // no harm doing so
            // otherwise the workspace doesn't saveas
            auto keys = _file->GetListOfKeys();
            if (keys) {
               for (auto &&k : *keys) {
                  auto cl = TClass::GetClass((static_cast<TKey *>(k))->GetClassName());
                  if (cl == RooWorkspace::Class() || cl->InheritsFrom("RooWorkspace")) {
                     fComp.reset(_file->Get<RooWorkspace>(k->GetName()), [](TObject *ws) {
                        // memory leak in workspace, some RooLinkedLists aren't cleared, fixed in ROOT 6.28
                        if (ws) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
                           dynamic_cast<RooWorkspace *>(ws)->_embeddedDataList.Delete();
#endif
                           xRooNode(*ws, std::make_shared<xRooNode>()).sterilize();
                           delete ws;
                        }
                     });
                     if (fComp) {
                        TNamed::SetNameTitle(fComp->GetName(), fComp->GetTitle());
                        fParent = std::make_shared<xRooNode>(
                           _file); // keep file alive - seems necessary to save workspace again in some cases
                        break;
                     }
                  }
               }
            }
         }
      } else if (pathName.EndsWith(".root") || pathName.EndsWith(".json")) {
         throw std::runtime_error(TString::Format("%s does not exist", name));
      }
   }

   if (auto _ws = get<RooWorkspace>(); _ws && (!parent || parent->get<TFile>())) {
      RooMsgService::instance()
         .getStream(RooFit::INFO)
         .removeTopic(RooFit::NumIntegration); // stop info message every time

      // check if any of the open files have version numbers greater than our major version
      // may not read correctly
      for (auto f : *gROOT->GetListOfFiles()) {
         if ((dynamic_cast<TFile *>(f)->GetVersion() / 100) > (gROOT->GetVersionInt() / 100)) {
            Warning("xRooNode", "There is file open with version %d > current version %d ... results may be wrong",
                    dynamic_cast<TFile *>(f)->GetVersion(), gROOT->GetVersionInt());
         }
      }

      // load list of colors if there is one
      if (auto colors = dynamic_cast<TSeqCollection *>(_ws->obj(gROOT->GetListOfColors()->GetName()))) {
         gROOT->GetListOfColors()->Clear();
         for (auto col : *colors) {
            gROOT->GetListOfColors()->Add(col);
         }
      }

      // use the datasets if any to 'mark' observables
      int checkCount = 0;
      for (auto &d : _ws->allData()) {
         for (auto &a : *d->get()) {
            if (auto v = _ws->var(a->GetName()); v) {
               v->setAttribute("obs");
            } else if (auto c = _ws->cat(a->GetName()); c) {
               c->setAttribute("obs");
            }
         }
         // count how many ds are checked ... if none are checked will check the first
         checkCount += d->TestBit(1 << 20);
      }

      if (checkCount == 0 && !_ws->allData().empty())
         _ws->allData().back()->SetBit(1 << 20, true);

      if (auto _set = dynamic_cast<RooArgSet *>(GETWSSNAPSHOTS(_ws).find("NominalParamValues")); _set) {
         for (auto s : *_set) {
            if (auto v = dynamic_cast<RooRealVar *>(s); v) {
               _ws->var(s->GetName())->setStringAttribute("nominal", TString::Format("%f", v->getVal()));
            }
         }
      }

      // also flag global observables ... relies on ModelConfig existences
      RooArgSet _allGlobs;
      for (auto &[k, v] : GETWSSETS(_ws)) {
         if (k == "globalObservables" || TString(k).EndsWith("_GlobalObservables")) {
            for (auto &s : v) {
               _allGlobs.add(*s);
               s->setAttribute("obs");
               s->setAttribute("global");
            }
         } else if (TString(k).EndsWith("_Observables")) {
            const_cast<RooArgSet &>(v).setAttribAll("obs");
         } else if (TString(k).EndsWith("_POI")) {
            for (auto &s : v) {
               s->setAttribute("poi");
               auto _v = dynamic_cast<RooRealVar *>(s);
               if (!_v)
                  continue;
               //               if (!_v->hasRange("physical")) {
               //                  _v->setRange("physical", 0, std::numeric_limits<double>::infinity());
               //                  // ensure range of poi is also straddling 0
               //                  if (_v->getMin() >= 0)
               //                     _v->setMin(-1e-5);
               //               }
            }
         } else if (TString(k).EndsWith("_NuisParams")) {
            const_cast<RooArgSet &>(v).setAttribAll("np");
         }
      }
      if (!_allGlobs.empty() && GETWSSETS(_ws).count("globalObservables") == 0) {
         _ws->defineSet("globalObservables", _allGlobs);
      }

      // now check if any pars don't have errors defined (not same as error=0) ... if so, use the first pdf (if there is
      // one) to try setting values from
      if (!_ws->allPdfs().empty()) {
         std::set<RooRealVar *> noErrorPars;
         std::string parNames;
         for (auto &p : np()) { // infer errors on all floating non-poi parameters
            auto v = p->get<RooRealVar>();
            if (!v)
               continue;
            if (!v->hasError()) {
               noErrorPars.insert(v);
               if (!parNames.empty())
                  parNames += ",";
               parNames += v->GetName();
            }
         }
         if (!noErrorPars.empty()) {
            Warning("xRooNode",
                    "Inferring initial errors of %d parameters (%s%s) (give all nuisance parameters an error to avoid "
                    "this msg)",
                    int(noErrorPars.size()), (*noErrorPars.begin())->GetName(), (noErrorPars.size() > 1) ? ",..." : "");
            // get the first top-level pdf
            browse();
            for (auto &a : *this) {
               if (noErrorPars.empty()) {
                  break;
               }
               if (a->fFolder == "!pdfs") {
                  try {
                     auto fr = a->floats().reduced(parNames).fitResult("prefit");
                     if (auto _fr = fr.get<RooFitResult>(); _fr) {
                        std::set<RooRealVar *> foundPars;
                        for (auto &v : noErrorPars) {
                           if (auto arg = dynamic_cast<RooRealVar *>(_fr->floatParsFinal().find(v->GetName()));
                               arg && arg->hasError()) {
                              v->setError(arg->getError());
                              foundPars.insert(v);
                           }
                        }
                        for (auto &v : foundPars) {
                           noErrorPars.erase(v);
                        }
                     }
                  } catch (...) {
                  }
               }
            }
         }
      }
   }

   if (strlen(GetTitle()) == 0) {
      if (fComp) {
         TNamed::SetTitle(fComp->GetTitle());
      } else {
         TNamed::SetTitle(GetName());
      }
   }
}

xRooNode::xRooNode(const TObject &comp, const std::shared_ptr<xRooNode> &parent)
   : xRooNode(/*[](const TObject& c) {
c.InheritsFrom("RooAbsArg");
if (s) {
return (s->getStringAttribute("alias")) ? s->getStringAttribute("alias") : c.GetName();
}
return c.GetName();
}(comp)*/
              (comp.InheritsFrom("RooAbsArg") && dynamic_cast<const RooAbsArg *>(&comp)->getStringAttribute("alias"))
                 ? dynamic_cast<const RooAbsArg *>(&comp)->getStringAttribute("alias")
                 : comp.GetName(),
              std::shared_ptr<TObject>(const_cast<TObject *>(&comp), [](TObject *) {}), parent)
{
}

xRooNode::xRooNode(const std::shared_ptr<TObject> &comp, const std::shared_ptr<xRooNode> &parent)
   : xRooNode(
        [&]() {
           if (auto a = std::dynamic_pointer_cast<RooAbsArg>(comp); a && a->getStringAttribute("alias"))
              return a->getStringAttribute("alias");
           if (comp)
              return comp->GetName();
           return "";
        }(),
        comp, parent)
{
}

xRooNode::xRooNode(double value) : xRooNode(RooFit::RooConst(value)) {}

void xRooNode::Checked(TObject *obj, bool val)
{
   if (obj != this)
      return;

   // cycle through states:
   //   unhidden and selected: tick, no uline
   //   hidden and unselected: notick, uline
   //   unhidden and unselected: tick, uline
   if (auto o = get<RooAbsReal>(); o) {
      if (o->isSelectedComp() && !val) {
         // deselecting and hiding
         o->selectComp(val);
         o->setAttribute("hidden");
      } else if (!o->isSelectedComp() && !val) {
         // selecting
         o->selectComp(!val);
      } else if (val) {
         // unhiding but keeping unselected
         o->setAttribute("hidden", false);
      }
      auto item = GetTreeItem(nullptr);
      item->CheckItem(!o->getAttribute("hidden"));
      if (o->isSelectedComp()) {
         item->ClearColor();
      } else {
         item->SetColor(kGray);
      }
      return;
   }

   if (auto o = get(); o) {
      // if (o->TestBit(1<<20)==val) return; // do nothing
      o->SetBit(1 << 20, val); // TODO: check is 20th bit ok to play with?
      if (auto fr = get<RooFitResult>(); fr) {
         if (auto _ws = ws(); _ws) {
            if (val) {
               // ensure fit result is in genericObjects list ... if not, add a copy ...
               if (!_ws->genobj(fr->GetName())) {
                  _ws->import(*fr);
                  if (auto wfr = dynamic_cast<RooFitResult *>(_ws->genobj(fr->GetName()))) {
                     fr = wfr;
                  }
               }
               RooArgSet _allVars = _ws->allVars();
               _allVars = fr->floatParsFinal();
               _allVars = fr->constPars();
               for (auto &i : fr->floatParsInit()) {
                  auto v = dynamic_cast<RooRealVar *>(_allVars.find(i->GetName()));
                  if (v)
                     v->setStringAttribute("initVal", TString::Format("%f", dynamic_cast<RooRealVar *>(i)->getVal()));
               }
               // uncheck all other fit results
               for (auto oo : _ws->allGenericObjects()) {
                  if (auto ffr = dynamic_cast<RooFitResult *>(oo); ffr && ffr != fr) {
                     ffr->ResetBit(1 << 20);
                  }
               }
            } else
               _ws->allVars() = fr->floatParsInit();
         }
         if (auto item = GetTreeItem(nullptr); item) {
            // update check marks on siblings
            if (auto first = item->GetParent()->GetFirstChild()) {
               do {
                  if (first->HasCheckBox()) {
                     auto _obj = static_cast<xRooNode *>(first->GetUserData());
                     first->CheckItem(_obj->get() && _obj->get()->TestBit(1 << 20));
                  }
               } while ((first = first->GetNextSibling()));
            }
         }
      }
   }
}

void xRooNode::Browse(TBrowser *b)
{
   static bool blockBrowse = false;
   if (blockBrowse)
      return;
   if (b == nullptr) {
      auto b2 = dynamic_cast<TBrowser *>(gROOT->GetListOfBrowsers()->Last());
      if (!b2 || !b2->GetBrowserImp()) { // no browser imp if browser was closed
         blockBrowse = true;
         gEnv->SetValue("X11.UseXft", "no"); // for faster x11
         gEnv->SetValue("X11.Sync", "no");
         gEnv->SetValue("X11.FindBestVisual", "no");
         gEnv->SetValue("Browser.Name", "TRootBrowser"); // forces classic root browser (in 6.26 onwards)
         gEnv->SetValue("Canvas.Name", "TRootCanvas");
         b2 = new TBrowser("nodeBrowser", this, "RooFit Browser");
         blockBrowse = false;
      } else if (strcmp(b2->GetName(), "nodeBrowser") == 0) {
         blockBrowse = true;
         b2->BrowseObject(this);
         blockBrowse = false;
      } else {
         auto _b = dynamic_cast<TGFileBrowser *>(GETACTBROWSER(dynamic_cast<TRootBrowser *>(b2->GetBrowserImp())));
         if (_b)
            _b->AddFSDirectory("Workspaces", nullptr, "SetRootDir");
         /*auto l = Node2::Class()->GetMenuList();
         auto o = new CustomClassMenuItem(TClassMenuItem::kPopupUserFunction,Node2::Class(),
                                          "blah blah blah","BlahBlah",0,"Option_t*",-1,true);
         //o->SetCall(o,"BlahBlah","Option_t*",-1);
         l->AddFirst(o);*/
         // b->BrowseObject(this);
         _b->GotoDir(nullptr);
         _b->Add(this, GetName());
         // b->Add(this);
      }
      return;
   }

   if (auto item = GetTreeItem(b); item) {
      if (!item->IsOpen() && IsFolder())
         return; // no need to rebrowse if closing
      // update check marks on any child items
      if (auto first = item->GetFirstChild()) {
         do {
            if (first->HasCheckBox()) {
               auto _obj = static_cast<xRooNode *>(first->GetUserData());
               first->CheckItem(_obj->get() &&
                                (_obj->get()->TestBit(1 << 20) ||
                                 (_obj->get<RooAbsArg>() && !_obj->get<RooAbsArg>()->getAttribute("hidden"))));
            }
         } while ((first = first->GetNextSibling()));
      }
   }

   browse();

   // for top-level pdfs default to having the .vars browsable too
   if (get<RooAbsPdf>() && fFolder == "!pdfs" && !_IsShowVars_()) {
      fBrowsables.push_back(std::make_shared<xRooNode>(vars()));
   }

   if (auto _fr = get<RooFitResult>(); _fr && fBrowsables.empty()) {
      // have some common drawing options
      fBrowsables.push_back(std::make_shared<xRooNode>(".Draw(\"pull\")", nullptr, *this));
      fBrowsables.push_back(std::make_shared<xRooNode>(".Draw(\"corrcolztext\")", nullptr, *this));
      if (std::unique_ptr<RooAbsCollection>(_fr->floatParsFinal().selectByAttrib("poi", true))->size() == 1) {
         fBrowsables.push_back(std::make_shared<xRooNode>(".Draw(\"impact\")", nullptr, *this));
      }
   }

   if (empty() && fBrowsables.empty()) {
      try {
         if (auto s = get<TStyle>()) {
            s->SetFillAttributes();
            if (auto ed = dynamic_cast<TGedEditor *>(TVirtualPadEditor::GetPadEditor())) {
               ed->SetModel(gPad, s, kButton1Down, true);
            }
         } else if (TString(GetName()).BeginsWith(".Draw(\"") && fParent) {
            fParent->Draw(TString(TString(GetName())(7, strlen(GetName()) - 9)) + b->GetDrawOption());
         } else {
            Draw(b->GetDrawOption());
         }
      } catch (const std::exception &e) {
         new TGMsgBox(
            gClient->GetRoot(),
            (gROOT->GetListOfBrowsers()->At(0))
               ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
               : gClient->GetRoot(),
            "Exception", e.what(),
            kMBIconExclamation); // deletes self on dismiss?
      }
   }

   bool hasFolders = false;
   if (strlen(GetName()) > 0 && GetName()[0] != '!') { // folders don't have folders
      for (auto &c : *this) {
         if (!c->fFolder.empty()) {
            hasFolders = true;
            break;
         }
      }
   }
   // auto _ws = get<RooWorkspace>();
   if (/*_ws*/ hasFolders) {
      // organize in folders
      auto _folders = find(".folders");
      if (!_folders) {
         _folders = emplace_back(std::make_shared<xRooNode>(".folders", nullptr, *this));
      }
      // ensure entry in folders for every folder type ...
      for (auto &v : *this) {
         if (!v->fFolder.empty() && !_folders->find(v->fFolder, false)) {
            _folders->emplace_back(std::make_shared<xRooNode>(v->fFolder.c_str(), nullptr, *this));
         }
      }
      // now just add all the folders
      for (auto &v : *_folders) {
         TString _name = v->GetName();
         if (_name.BeginsWith('!'))
            _name = _name(1, _name.Length()); // strip ! from display
         b->Add(v.get(), _name);
      }
   }

   for (auto &v : *this) {
      if (hasFolders && !v->fFolder.empty())
         continue; // in the folders
      if (strcmp(v->GetName(), ".folders") == 0)
         continue; // never 'browse' the folders property
      auto _fr = v->get<RooFitResult>();
      int _checked = (v->get<RooAbsData>() || _fr) ? v->get()->TestBit(1 << 20) : -1;
      if (_fr && ((_fr->status() == 0 && _fr->numStatusHistory() == 0) || (_fr->floatParsFinal().empty()))) {
         // this is a "PARTIAL" fit result ... don't allow it to be selected
         _checked = -1;
      }
      if (v->get<RooAbsPdf>() && get<RooSimultaneous>())
         _checked = !v->get<RooAbsArg>()->getAttribute("hidden");
      TString _name = v->GetName();
      if (v->get() && _name.BeginsWith(TString(v->get()->ClassName()) + "::")) {
         _name = _name(strlen(v->get()->ClassName()) + 2, _name.Length());
      }
      if (_name.BeginsWith(".")) {
         // property node -- display the  name of the contained object
         if (v->get()) {
            _name = TString::Format("%s: %s::%s", _name.Data(), v->get()->ClassName(),
                                    (v->get<RooAbsArg>() && v->get<RooAbsArg>()->getStringAttribute("alias"))
                                       ? v->get<RooAbsArg>()->getStringAttribute("alias")
                                       : v->get()->GetName());
         }
      } else if (v->get() && !v->get<TFile>() && !TString(v->GetName()).BeginsWith('/'))
         _name = TString::Format("%s::%s", v->get()->ClassName(), _name.Data());
      if (auto _type = v->GetNodeType(); strlen(_type)) {
         // decided not to show const values until figure out how to update if value changes
         /*if (TString(_type)=="Const") _name += TString::Format(" [%s=%g]",_type,v->get<RooConstVar>()->getVal());
         else*/
         _name += TString::Format(" [%s]", _type);
      }
      if (auto fv = v->get<RooFormulaVar>()) {
         TString formu = TString::Format(" [%s]", fv->expression());
         for (size_t i = 0; i < fv->dependents().size(); i++) {
            formu.ReplaceAll(TString::Format("x[%zu]", i), fv->dependents()[i].GetName());
         }
         _name += formu;
      } else if (auto gv = v->get<RooGenericPdf>()) {
         TString formu = TString::Format(" [%s]", gv->expression());
         for (size_t i = 0; i < gv->dependents().size(); i++) {
            formu.ReplaceAll(TString::Format("x[%zu]", i), gv->dependents()[i].GetName());
         }
         _name += formu;
      }
      // tool tip defaults to displaying name and title, so temporarily set name to obj name if has one
      // and set title to the object type
      TString nameSave(v->TNamed::GetName());
      TString titleSave(v->TNamed::GetTitle());
      if (auto o = v->get(); o)
         v->TNamed::SetNameTitle(o->GetName(), o->ClassName());
      b->Add(v.get(), _name, _checked);
      if (auto o = v->get(); o)
         v->TNamed::SetNameTitle(nameSave, titleSave);
      if (_checked != -1) {
         dynamic_cast<TQObject *>(b->GetBrowserImp())
            ->Connect("Checked(TObject *, bool)", ClassName(), v.get(), "Checked(TObject *, bool)");
      }
      if (_fr) {
         if (_fr->status() || _fr->covQual() != 3) { // snapshots or bad fits
            v->GetTreeItem(b)->SetColor((_fr->numStatusHistory() || _fr->floatParsFinal().empty()) ? kRed : kBlue);
         } else if (_fr->numStatusHistory() == 0) { // partial fit result ..
            v->GetTreeItem(b)->SetColor(kGray);
         }
      }
      if ((v->fFolder == "!np" || v->fFolder == "!poi")) {
         if (v->get<RooAbsArg>()->getAttribute("Constant")) {
            v->GetTreeItem(b)->SetColor(kGray);
         } else
            v->GetTreeItem(b)->ClearColor();
      }
      if (auto _htr = v->get<RooStats::HypoTestResult>(); _htr) {
         // check for fit statuses
         if (auto fits = _htr->GetFitInfo()) {
            for (int i = 0; i < fits->numEntries(); i++) {
               // if any fit (other than a genFit) is bad, flag point as bad
               if (fits->get(i)->getCatIndex("type") != 5 && fits->get(i)->getRealValue("status") != 0) {
                  v->GetTreeItem(b)->SetColor(kRed);
                  break;
               }
            }
         } else {
            v->GetTreeItem(b)->SetColor(kBlue); // unknown fit status
         }
      }

      // v.fBrowsers.insert(b);
   }

   // for pdfs, check for datasets too and add to list
   /*if (get<RooAbsPdf>()) {
       auto dsets = datasets();
       if (!dsets.empty()) {
           // check if already have .datasets() in browsables
           bool found(false);
           for(auto& p : fBrowsables) {
               if (TString(p->GetName())==".datasets()") {found=true;
                   // add
                   break;
               }
           }
           if (!found) {
               fBrowsables.push_back(std::make_shared<xRooNode>(dsets));
           }
       }
   }*/
   // browse the browsables too
   for (auto &v : fBrowsables) {
      TString _name = v->GetName();
      if (_name == ".memory")
         continue; // hide the memory from browsing, if put in browsables
      TString nameSave(v->TNamed::GetName());
      TString titleSave(v->TNamed::GetTitle());
      if (auto o = v->get(); o)
         v->TNamed::SetNameTitle(o->GetName(), o->ClassName());
      b->Add(v.get(), _name, -1);
      if (auto o = v->get(); o)
         v->TNamed::SetNameTitle(nameSave, titleSave);
   }

   b->SetSelected(this);
}

void xRooNode::_ShowVars_(bool set)
{
   if (!set) {
      // can't remove as causes a crash, need to remove from the browser first
      /*for(auto itr = fBrowsables.begin(); itr != fBrowsables.end(); ++itr) {
          if (strcmp((*itr)->GetName(),".vars")==0) {
              fBrowsables.erase(itr);
          }
      }*/
   } else {
      auto v = std::make_shared<xRooNode>(vars());
      fBrowsables.push_back(v);
      if (auto l = GetListTree(nullptr)) {
         l->AddItem(GetTreeItem(nullptr), v->GetName(), v.get());
      }
   }
}

bool xRooNode::_IsShowVars_() const
{
   for (auto &b : fBrowsables) {
      if (strcmp(b->GetName(), ".vars") == 0)
         return true;
   }
   return false;
}

bool xRooNode::IsFolder() const
{
   if (strlen(GetName()) > 0 && GetName()[0] == '!')
      return true;
   if (strlen(GetName()) > 0 && GetName()[0] == '.' && !(TString(GetName()).BeginsWith(".Draw(\"")))
      return true;
   if (empty())
      const_cast<xRooNode *>(this)->browse();
   return !empty();
}

class Axis2 : public TAxis {

public:
   using TAxis::TAxis;
   double GetBinWidth(Int_t bin) const override
   {
      if (auto v = var(); v)
         return v->getBinWidth(bin - 1, GetName());
      return 1;
   }
   double GetBinLowEdge(Int_t bin) const override
   {
      if (auto v = rvar(); v) {
         return (bin == v->getBinning(GetName()).numBins() + 1) ? v->getBinning(GetName()).binHigh(bin - 2)
                                                                : v->getBinning(GetName()).binLow(bin - 1);
      }
      return bin - 1;
   }
   double GetBinUpEdge(Int_t bin) const override
   {
      if (auto v = rvar(); v)
         return (bin == 0) ? v->getBinning(GetName()).binLow(bin) : v->getBinning(GetName()).binHigh(bin - 1);
      return bin;
   }

   const char *GetTitle() const override
   {
      return (binning() && strlen(binning()->GetTitle())) ? binning()->GetTitle() : GetParent()->GetTitle();
   }
   void SetTitle(const char *title) override
   {
      if (binning()) {
         const_cast<RooAbsBinning *>(binning())->SetTitle(title);
      } else {
         dynamic_cast<TNamed *>(GetParent())->SetTitle(title);
      }
   }

   void Set(Int_t nbins, const double *xbins) override
   {
      if (auto v = dynamic_cast<RooRealVar *>(rvar()))
         v->setBinning(RooBinning(nbins, xbins), GetName());
      TAxis::Set(nbins, xbins);
   }
   void Set(Int_t nbins, const float *xbins) override
   {
      std::vector<double> bins(nbins + 1);
      for (int i = 0; i <= nbins; i++)
         bins.at(i) = xbins[i];
      return Set(nbins, &bins[0]);
   }
   void Set(Int_t nbins, double xmin, double xmax) override
   {
      if (auto v = dynamic_cast<RooRealVar *>(rvar()))
         v->setBinning(RooUniformBinning(xmin, xmax, nbins), GetName());
      TAxis::Set(nbins, xmin, xmax);
   }

   const RooAbsBinning *binning() const { return var()->getBinningPtr(GetName()); }

   Int_t FindFixBin(const char *label) const override { return TAxis::FindFixBin(label); }
   Int_t FindFixBin(double x) const override { return (binning()) ? (binning()->binNumber(x) + 1) : x; }

private:
   RooAbsLValue *var() const { return dynamic_cast<RooAbsLValue *>(GetParent()); }
   RooAbsRealLValue *rvar() const { return dynamic_cast<RooAbsRealLValue *>(GetParent()); }
};

std::shared_ptr<TObject> xRooNode::getObject(const std::string &name, const std::string &type) const
{
   // if (fParent) return fParent->getObject(name);

   if (auto _owned = find(".memory"); _owned) {
      for (auto &o : *_owned) {
         if (name == o->GetName()) {
            if (type.empty() || o->get()->InheritsFrom(type.c_str()))
               return o->fComp;
         }
      }
   }

   // see if have a provider
   auto _provider = fProvider;
   auto _parent = fParent;
   while (!_provider && _parent) {
      _provider = _parent->fProvider;
      _parent = _parent->fParent;
   }
   if (_provider)
      return _provider->getObject(name, type);

   if (ws()) {
      std::shared_ptr<TObject> out;
      if (auto arg = ws()->arg(name.c_str()); arg) {
         auto _tmp = std::shared_ptr<TObject>(arg, [](TObject *) {});
         if (!type.empty() && arg->InheritsFrom(type.c_str()))
            return _tmp;
         if (!out)
            out = _tmp;
      }
      if (auto arg = ws()->data(name.c_str()); arg) {
         auto _tmp = std::shared_ptr<TObject>(arg, [](TObject *) {});
         if (!type.empty() && arg->InheritsFrom(type.c_str()))
            return _tmp;
         if (!out)
            out = _tmp;
      }
      if (auto arg = ws()->genobj(name.c_str()); arg) {
         auto _tmp = std::shared_ptr<TObject>(arg, [](TObject *) {});
         if (!type.empty() && arg->InheritsFrom(type.c_str()))
            return _tmp;
         if (!out)
            out = _tmp;
      }
      if (auto arg = ws()->embeddedData(name.c_str()); arg) {
         auto _tmp = std::shared_ptr<TObject>(arg, [](TObject *) {});
         if (!type.empty() && arg->InheritsFrom(type.c_str()))
            return _tmp;
         if (!out)
            out = _tmp;
      }
      if (auto arg = GETWSSNAPSHOTS(ws()).find(name.c_str()); arg) {
         auto _tmp = std::shared_ptr<TObject>(arg, [](TObject *) {});
         if (!type.empty() && arg->InheritsFrom(type.c_str()))
            return _tmp;
         if (!out)
            out = _tmp;
      }
      return out;
   }
   if (auto arg = get<RooAbsArg>()) {
      // can try all nodes
      RooArgSet nodes;
      arg->treeNodeServerList(&nodes);
      if (auto server = nodes.find(name.c_str())) {
         return std::shared_ptr<TObject>(server, [](TObject *) {});
      }
   }
   return nullptr;
}

TAxis *xRooNode::GetXaxis() const
{
   if (fXAxis) {
      // check if num bins needs update or not
      if (auto cat = dynamic_cast<RooAbsCategory *>(fXAxis->GetParent());
          cat && cat->numTypes() != fXAxis->GetNbins()) {
         fXAxis.reset();
      } else {
         return fXAxis.get();
      }
   }
   RooAbsLValue *x = nullptr;
   if (auto a = get<RooAbsArg>(); a && a->isFundamental())
      x = dynamic_cast<RooAbsLValue *>(a); // self-axis

   auto _parentX = (!x && fParent && !fParent->get<RooSimultaneous>()) ? fParent->GetXaxis() : nullptr;

   auto o = get<RooAbsReal>();
   if (!o)
      return _parentX;

   if (auto xName = o->getStringAttribute("xvar"); xName) {
      x = dynamic_cast<RooAbsLValue *>(getObject(xName).get());
   }

   // if xvar has become set equal to an arg and this is a pdf, we will allow a do-over
   if (!x) {
      // need to choose from dependent fundamentals, in following order:
      // parentX (if not a glob), robs, globs, vars, args

      if (_parentX && !dynamic_cast<RooAbsArg *>(_parentX->GetParent())->getAttribute("global") &&
          (o->dependsOn(*dynamic_cast<RooAbsArg *>(_parentX->GetParent())) || vars().empty())) {
         x = dynamic_cast<RooAbsLValue *>(_parentX->GetParent());
      } else if (auto _obs = obs(); !_obs.empty()) {
         for (auto &v : _obs) {
            if (!v->get<RooAbsArg>()->getAttribute("global")) {
               x = v->get<RooAbsLValue>();
               if (x)
                  break;
            } else if (!x) {
               x = v->get<RooAbsLValue>();
            }
         }
      } else if (auto _pars = pars(); !_pars.empty()) {
         for (auto &v : _pars) {
            if (!v->get<RooAbsArg>()->getAttribute("Constant")) {
               x = v->get<RooAbsLValue>();
               if (x)
                  break;
            } else if (!x) {
               x = v->get<RooAbsLValue>();
            }
         }
      }

      if (!x) {
         return nullptr;
      }
   }

   if (o != dynamic_cast<TObject *>(x)) {
      o->setStringAttribute("xvar", dynamic_cast<TObject *>(x)->GetName());
   }

   // decide binning to use
   TString binningName = o->getStringAttribute("binning");
   auto _bnames = x->getBinningNames();
   bool hasBinning = false;
   for (auto &b : _bnames) {
      if (b == binningName) {
         hasBinning = true;
         break;
      }
   }
   if (!hasBinning) {
      // doesn't have binning, so clear binning attribute
      // this can happen after Combine of models because binning don't get combined yet (should fix this)
      Warning("GetXaxis", "Binning %s not defined on %s - clearing", binningName.Data(),
              dynamic_cast<TObject *>(x)->GetName());
      o->setStringAttribute("binning", nullptr);
      binningName = "";
   }

   if (binningName == "" && o != dynamic_cast<TObject *>(x)) {
      // has var has a binning matching this nodes name then use that
      auto __bnames = x->getBinningNames();
      for (auto &b : __bnames) {
         if (b == GetName())
            binningName = GetName();
         if (b == o->GetName()) {
            binningName = o->GetName();
            break;
         } // best match
      }
      if (binningName == "") {
         // if we are binned in this var then will define that as a binning
         if (/*o->isBinnedDistribution(*dynamic_cast<RooAbsArg *>(x))*/
             auto bins = _or_func(
                /*o->plotSamplingHint(*dynamic_cast<RooAbsRealLValue
                 *>(x),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity())*/
                (std::list<double> *)(nullptr),
                o->binBoundaries(*dynamic_cast<RooAbsRealLValue *>(x), -std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity()));
             bins) {
            std::vector<double> _bins;
            for (auto &b : *bins) {
               if (_bins.empty() || std::abs(_bins.back() - b) > 1e-5 * _bins.back())
                  _bins.push_back(b);
            }
            fXAxis = std::make_shared<Axis2>(_bins.size() - 1, &_bins[0]);
            // add this binning to the var to avoid recalling ...
            if (auto _v = dynamic_cast<RooRealVar *>(x); _v) {
               _v->setBinning(RooBinning(_bins.size() - 1, &_bins[0], o->GetName()), o->GetName());
               _v->getBinning(o->GetName())
                  .SetTitle(""); // indicates to use the current var title when building histograms etc
                                 //_v->getBinning(o->GetName()).SetTitle(strlen(dynamic_cast<TObject*>(x)->GetTitle()) ?
                                 // dynamic_cast<TObject*>(x)->GetTitle() : dynamic_cast<TObject*>(x)->GetName());
            }
            binningName = o->GetName();
            delete bins;
         } else if (_parentX) {
            // use parent axis binning if defined, otherwise we will default
            binningName = _parentX->GetName();
         }
      }
   }

   if (!fXAxis) {
      if (auto r = dynamic_cast<RooAbsRealLValue *>(x); r) {
         if (r->getBinning(binningName).isUniform()) {
            fXAxis = std::make_shared<Axis2>(x->numBins(binningName), r->getMin(binningName), r->getMax(binningName));
         } else {
            fXAxis = std::make_shared<Axis2>(x->numBins(binningName), r->getBinning(binningName).array());
         }
      } else if (auto cat = dynamic_cast<RooCategory *>(x)) {
         std::vector<double> bins = {};
         for (int i = 0; i <= x->numBins(binningName); i++)
            bins.push_back(i);
         fXAxis = std::make_shared<Axis2>(x->numBins(binningName), &bins[0]);
         // TODO have to load current state of bin labels if was a category (sadly not a virtual method)
         int i = 1;
         std::map<int, std::string> cats; // fill into a map to preserve index ordering
         for (auto &c : *cat) {
            if (cat->isStateInRange(binningName, c.first.c_str())) {
               cats[c.second] = c.first;
            }
         }
         for (auto &[_, label] : cats) {
            fXAxis->SetBinLabel(i++, label.c_str());
         }
      }
   }

   fXAxis->SetName(binningName);
   fXAxis->SetParent(dynamic_cast<TObject *>(x));
   return fXAxis.get();
}

const char *xRooNode::GetIconName() const
{
   if (auto o = get(); o) {
      if (o->InheritsFrom("RooWorkspace"))
         return "TFile";
      if (o->InheritsFrom("RooAbsData"))
         return "TProfile";
      if (o->InheritsFrom("RooSimultaneous"))
         return "TH3D";

      if (o->InheritsFrom("RooProdPdf"))
         return "a.C"; // or nullptr for folder
      if (o->InheritsFrom("RooRealSumPdf") || o->InheritsFrom("RooAddPdf"))
         return "TH2D";
      // if(o->InheritsFrom("RooProduct")) return "TH1D";
      if (o->InheritsFrom("RooFitResult")) {
         if (!gClient->GetMimeTypeList()->GetIcon("xRooFitRooFitResult", true)) {
            gClient->GetMimeTypeList()->AddType("xRooFitRooFitResult", "xRooFitRooFitResult", "package.xpm",
                                                "package.xpm", "->Browse()");
         }
         return "xRooFitRooFitResult";
      }
      if (o->InheritsFrom("RooRealVar") || o->InheritsFrom("RooCategory")) {
         if (get<RooAbsArg>()->getAttribute("obs")) {
            if (!gClient->GetMimeTypeList()->GetIcon("xRooFitObs", true)) {
               gClient->GetMimeTypeList()->AddType("xRooFitObs", "xRooFitObs", "x_pic.xpm", "x_pic.xpm", "->Browse()");
            }
            if (!gClient->GetMimeTypeList()->GetIcon("xRooFitGlobs", true)) {
               gClient->GetMimeTypeList()->AddType("xRooFitGlobs", "xRooFitGlobs", "z_pic.xpm", "z_pic.xpm",
                                                   "->Browse()");
            }
            return (get<RooAbsArg>()->getAttribute("global") ? "xRooFitGlobs" : "xRooFitObs");
         }
         return "TLeaf";
      }
      if (o->InheritsFrom("TStyle")) {
         if (!gClient->GetMimeTypeList()->GetIcon("xRooFitTStyle", true)) {
            gClient->GetMimeTypeList()->AddType("xRooFitTStyle", "xRooFitTStyle", "bld_colorselect.xpm",
                                                "bld_colorselect.xpm", "->Browse()");
         }
         return "xRooFitTStyle";
      }
      if (o->InheritsFrom("RooConstVar")) {
         /*if (!gClient->GetMimeTypeList()->GetIcon("xRooFitRooConstVar",true)) {
             gClient->GetMimeTypeList()->AddType("xRooFitRooConstVar", "xRooFitRooConstVar", "stop_t.xpm", "stop_t.xpm",
         "->Browse()");
         }
         return "xRooFitRooConstVar";*/
         return "TMethodBrowsable-leaf";
      }
      if (o->InheritsFrom("RooStats::HypoTestInverterResult")) {
         if (!gClient->GetMimeTypeList()->GetIcon("xRooFitScanStyle", true)) {
            gClient->GetMimeTypeList()->AddType("xRooFitScanStyle", "xRooFitScanStyle", "f2_s.xpm", "f2_s.xpm",
                                                "->Browse()");
         }
         return "xRooFitScanStyle";
      }
      if (o->InheritsFrom("RooStats::HypoTestResult")) {
         if (!gClient->GetMimeTypeList()->GetIcon("xRooFitTestStyle", true)) {
            gClient->GetMimeTypeList()->AddType("xRooFitTestStyle", "xRooFitTestStyle", "diamond.xpm", "diamond.xpm",
                                                "->Browse()");
         }
         return "xRooFitTestStyle";
      }
      if (o->InheritsFrom("RooStats::HistFactory::FlexibleInterpVar"))
         return "TBranchElement-folder";
      if (o->InheritsFrom("RooAbsPdf")) {
         if (!gClient->GetMimeTypeList()->GetIcon("xRooFitPDFStyle", true)) {
            gClient->GetMimeTypeList()->AddType("xRooFitPDFStyle", "xRooFitPDFStyle", "pdf.xpm", "pdf.xpm",
                                                "->Browse()");
         }
         return "xRooFitPDFStyle";
      }
      if (o->InheritsFrom("RooStats::ModelConfig")) {
         if (!gClient->GetMimeTypeList()->GetIcon("xRooFitMCStyle", true)) {
            gClient->GetMimeTypeList()->AddType("xRooFitMCStyle", "xRooFitMCStyle", "app_t.xpm", "app_t.xpm",
                                                "->Browse()");
         }
         return "xRooFitMCStyle";
      }
      if (auto a = dynamic_cast<RooAbsReal *>(o); a) {
         if (auto _ax = GetXaxis();
             _ax && (a->isBinnedDistribution(*dynamic_cast<RooAbsArg *>(_ax->GetParent())) ||
                     (dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()) &&
                      std::unique_ptr<std::list<double>>(a->binBoundaries(
                         *dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()), -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity()))))) {
            return "TH1D";
         }
         return "TF1";
      }
      return o->ClassName();
   }
   if (!IsFolder()) {
      return "Unknown";
   }
   return nullptr;
}

const char *xRooNode::GetNodeType() const
{
   if (auto o = get(); o && fParent && (fParent->get<RooProduct>() || fParent->get<RooRealSumPdf>())) {
      if (o->InheritsFrom("RooStats::HistFactory::FlexibleInterpVar"))
         return "Overall";
      if (o->InheritsFrom("PiecewiseInterpolation"))
         return (dynamic_cast<RooAbsArg *>(o)->getAttribute("density")) ? "DensityHisto" : "Histo";
      if (o->InheritsFrom("RooHistFunc"))
         return (dynamic_cast<RooAbsArg *>(o)->getAttribute("density")) ? "ConstDensityHisto" : "ConstHisto";
      if (o->InheritsFrom("RooBinWidthFunction"))
         return "Density";
      if (o->InheritsFrom("ParamHistFunc"))
         return "Shape";
      if (o->InheritsFrom("RooRealVar"))
         return "Norm";
      if (o->InheritsFrom("RooConstVar"))
         return "Const";
   }
   return "";
}

xRooNode xRooNode::coords(bool setVals) const
{
   xRooNode out(".coords", nullptr, *this);
   // go up through parents looking for slice obs
   auto _p = std::shared_ptr<xRooNode>(const_cast<xRooNode *>(this), [](xRooNode *) {});
   while (_p) {
      TString pName(_p->GetName());
      // following is commented out while still considering, but idea is to include category in coords
      /*if (auto s = _p->get<RooSimultaneous>(); s && s->indexCat().InheritsFrom("RooCategory") &&
      !out.find(s->indexCat().GetName())) { auto cat = const_cast<RooCategory*>(dynamic_cast<const
      RooCategory*>(&s->indexCat()));
         // check if we have a pdf for every category ... if not then add to cut
         cat->clearRange("coordRange",true);
         bool hasMissing = false;
         std::string includedStates;
         for (auto state : *cat) {
            if (!s->getPdf(state.first.c_str())) {
               hasMissing = true;
            } else {
               if (!includedStates.empty()) {
                  includedStates += ",";
               }
               includedStates += state.first;
            }
         }
         if (hasMissing) {
            if(includedStates.find(",") != std::string::npos) {
               cat->addToRange("coordRange",includedStates.c_str());
            } else {
               cat->setLabel(includedStates);
            }
            out.emplace_back(std::make_shared<xRooNode>(cat->GetName(),_p->getObject<RooAbsArg>(cat->GetName()),_p));
         }
      } else*/
      if (auto pos = pName.Index('='); pos != -1) {
         if (pos > 0 && pName(pos - 1) == '<') {
            // should be a range on a real lvalue, of form low<=name<high
            double low = TString(pName(0, pos - 1)).Atof();
            pName = pName(pos + 1, pName.Length());
            double high = TString(pName(pName.Index('<') + 1, pName.Length())).Atof();
            pName = pName(0, pName.Index('<'));
            if (auto _obs = _p->getObject<RooAbsRealLValue>(pName.Data()); _obs) {
               if (setVals) {
                  _obs->setVal((high + low) / 2.);
                  static_cast<RooRealVar *>(_obs.get())->setRange("coordRange", low, high);
                  _obs->setStringAttribute(
                     "coordRange", "coordRange"); // will need if we allow multi disconnected regions, need comma list
               }
               out.emplace_back(std::make_shared<xRooNode>(_obs->GetName(), _obs, _p));
            } else {
               throw std::runtime_error(TString::Format("Unknown observable: %s", pName.Data()));
            }

         } else if (auto _obs = _p->getObject<RooAbsArg>(pName(0, pos)); _obs) {
            if (setVals) {
               if (auto _cat = dynamic_cast<RooAbsCategoryLValue *>(_obs.get()); _cat) {
                  _cat->setLabel(pName(pos + 1, pName.Length()));
               } else if (auto _var = dynamic_cast<RooAbsRealLValue *>(_obs.get()); _var) {
                  _var->setVal(TString(pName(pos + 1, pName.Length())).Atof());
               }
            }
            out.emplace_back(std::make_shared<xRooNode>(_obs->GetName(), _obs, _p));
         } else {
            throw std::runtime_error("Unknown observable, could not find");
         }
      }
      _p = _p->fParent;
   }
   return out;
}

void xRooNode::_Add_(const char *name, const char *opt)
{
   try {
      Add(name, opt);
   } catch (const std::exception &e) {
      new TGMsgBox(gClient->GetRoot(), gClient->GetRoot(), "Exception", e.what(),
                   kMBIconExclamation); // deletes self on dismiss?
   }
}
void xRooNode::_Vary_(const char *what)
{
   try {
      Vary(what);
   } catch (const std::exception &e) {
      new TGMsgBox(gClient->GetRoot(), gClient->GetRoot(), "Exception", e.what(),
                   kMBIconExclamation); // deletes self on dismiss?
   }
}

xRooNode xRooNode::Remove(const xRooNode &child)
{

   if (strcmp(GetName(), ".poi") == 0) {
      // demote a parameter from being a poi
      auto toRemove =
         (child.get<RooAbsArg>() || !find(child.GetName())) ? child : xRooNode(find(child.GetName())->fComp);
      if (toRemove) {
         if (!toRemove.get<RooAbsArg>()->getAttribute("poi")) {
            throw std::runtime_error(TString::Format("%s is not a poi", toRemove.GetName()));
         }
         toRemove.get<RooAbsArg>()->setAttribute("poi", false);
         return toRemove;
      }
   } else if (strcmp(GetName(), ".factors") == 0 || strcmp(GetName(), ".constraints") == 0 ||
              strcmp(GetName(), ".components") == 0) {
      auto toRemove =
         (child.get<RooAbsArg>() || !find(child.GetName())) ? child : xRooNode(find(child.GetName())->fComp);
      if (auto p = fParent->get<RooProdPdf>(); p) {
         auto pdf = toRemove.get<RooAbsArg>();
         if (!pdf)
            pdf = p->pdfList().find(child.GetName());
         if (!pdf)
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
         auto i = p->pdfList().index(*pdf);
         if (i >= 0) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
            const_cast<RooArgList &>(p->pdfList()).remove(*pdf);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
            p->_pdfNSetList.erase(p->_pdfNSetList.begin() + i);
#else
            auto nset = p->_pdfNSetList.At(i);
            p->_pdfNSetList.Remove(nset);
            delete nset; // I don't think the RooLinkedList owned it so must delete ourself
#endif
            if (p->_extendedIndex == i)
               p->_extendedIndex = -1;
            else if (p->_extendedIndex > i)
               p->_extendedIndex--;
#else
            p->removePdfs(RooArgSet(*pdf));
#endif
            sterilize();
            return xRooNode(*pdf);
         } else {
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
         }
      } else if (auto p2 = fParent->get<RooProduct>(); p2) {
         auto arg = toRemove.get<RooAbsArg>();
         if (!arg)
            arg = p2->components().find(child.GetName());
         if (!arg)
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
            // remove server ... doesn't seem to trigger removal from proxy
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
         p2->_compRSet.remove(*arg);
#else
         const_cast<RooArgList &>(p2->realComponents()).remove(*arg);
#endif
         p2->removeServer(*arg, true);
         sterilize();
         return xRooNode(*arg);
      } else if (fParent->get<RooSimultaneous>()) {
         // remove from all channels
         bool removed = false;
         for (auto &c : fParent->bins()) {
            try {
               c->constraints().Remove(toRemove);
               removed = true;
            } catch (std::runtime_error &) { /* wasn't a constraint in channel */
            }
         }
         sterilize();
         if (!removed)
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
         return toRemove;
      } else if (auto p4 = fParent->get<RooRealSumPdf>(); p4) {
         auto arg = toRemove.get<RooAbsArg>();
         if (!arg)
            arg = p4->funcList().find(child.GetName());
         if (!arg)
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
         // remove, including coef removal ....
         auto idx = p4->funcList().index(arg);

         if (idx != -1) {

            const_cast<RooArgList &>(p4->funcList()).remove(*arg);
            p4->removeServer(*arg, true);
            // have to be careful removing coef because if shared will end up removing them all!!
            std::vector<RooAbsArg *> _coefs;
            for (size_t ii = 0; ii < const_cast<RooArgList &>(p4->coefList()).size(); ii++) {
               if (ii != size_t(idx))
                  _coefs.push_back(const_cast<RooArgList &>(p4->coefList()).at(ii));
            }
            const_cast<RooArgList &>(p4->coefList()).removeAll();
            for (auto &a : _coefs)
               const_cast<RooArgList &>(p4->coefList()).add(*a);

            sterilize();
         } else {
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
         }
         return xRooNode(*arg);
      } else if (auto p5 = fParent->get<RooAddPdf>(); p5) {
         auto arg = toRemove.get<RooAbsArg>();
         if (!arg)
            arg = p5->pdfList().find(child.GetName());
         if (!arg)
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
         // remove, including coef removal ....
         auto idx = p5->pdfList().index(arg);

         if (idx != -1) {

            const_cast<RooArgList &>(p5->pdfList()).remove(*arg);
            p5->removeServer(*arg, true);
            // have to be careful removing coef because if shared will end up removing them all!!
            std::vector<RooAbsArg *> _coefs;
            for (size_t ii = 0; ii < const_cast<RooArgList &>(p5->coefList()).size(); ii++) {
               if (ii != size_t(idx))
                  _coefs.push_back(const_cast<RooArgList &>(p5->coefList()).at(ii));
            }
            const_cast<RooArgList &>(p5->coefList()).removeAll();
            for (auto &a : _coefs)
               const_cast<RooArgList &>(p5->coefList()).add(*a);

            sterilize();
         } else {
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
         }
         return xRooNode(*arg);
      } else if (auto p6 = fParent->get<RooAddition>(); p6) {
         auto arg = toRemove.get<RooAbsArg>();
         if (!arg)
            arg = p6->list().find(child.GetName());
         if (!arg)
            throw std::runtime_error(TString::Format("Cannot find %s in %s", child.GetName(), fParent->GetName()));
         // remove server ... doesn't seem to trigger removal from proxy
         const_cast<RooArgList &>(p6->list()).remove(*arg);
         p6->removeServer(*arg, true);
         sterilize();
         return xRooNode(*arg);
      }
   }

   if (auto w = get<RooWorkspace>(); w) {
      xRooNode out(child.GetName());
      auto arg = w->components().find(child.GetName());
      if (!arg)
         arg = operator[](child.GetName())->get<RooAbsArg>();
      if (!arg) {
         throw std::runtime_error(TString::Format("Cannot find %s in workspace %s", child.GetName(), GetName()));
      }
      // check has no clients ... if so, cannot delete
      if (arg->hasClients()) {
         throw std::runtime_error(
            TString::Format("Cannot remove %s from workspace %s, because it has dependencies - first remove from those",
                            child.GetName(), GetName()));
      }
      const_cast<RooArgSet &>(w->components()).remove(*arg); // deletes arg
      Info("Remove", "Deleted %s from workspace %s", out.GetName(), GetName());
      return out;
   } else if (get<RooProduct>() || get<RooProdPdf>()) {
      return factors().Remove(child);
   } else if (get<RooRealSumPdf>() || get<RooAddPdf>() || get<RooAddition>()) {
      return components().Remove(child);
   }

   throw std::runtime_error("Removal not implemented for object type " +
                            std::string(get() ? get()->ClassName() : "null"));
}

xRooNode xRooNode::Add(const xRooNode &child, Option_t *opt)
{

   class AutoUpdater {
   public:
      AutoUpdater(xRooNode &_n) : n(_n) {}
      ~AutoUpdater() { n.browse(); }
      xRooNode &n;
   };
   AutoUpdater xxx(*this);

   TString sOpt(opt);
   bool considerType(sOpt == "+");

   if (strlen(GetName()) > 0 && GetName()[0] == '!' && fParent) {
      // folder .. pass onto parent and add folder to child folder list
      const_cast<xRooNode &>(child).fFolder += GetName();
      return fParent->Add(child, opt);
   }
   // this is how to get the first real parent ... may be useful at some point?
   /*auto realParent = fParent;
   while(!realParent->get()) {
       realParent = realParent->fParent;
       if (!realParent) throw std::runtime_error("No parentage");
   }*/

   // adding to a collection node will incorporate the child into the parent of the collection
   // in the appropriate way
   if (strcmp(GetName(), ".factors") == 0) {
      // multiply the parent
      return fParent->Multiply(child, opt);
   } else if (strcmp(GetName(), ".components") == 0) {
      // add to the parent
      return fParent->Add(child, opt);
   } else if (strcmp(GetName(), ".variations") == 0) {
      // vary the parent
      return fParent->Vary(child);
   } else if (strcmp(GetName(), ".constraints") == 0) {
      // constrain the parent
      return fParent->Constrain(child);
   } else if (strcmp(GetName(), ".bins") == 0 && fParent->get<RooSimultaneous>()) {
      // adding a channel (should adding a 'bin' be an 'Extend' operation?)
      return fParent->Vary(child);
   } else if ((strcmp(GetName(), ".globs") == 0)) {
      if (child.get<RooAbsArg>() || (!child.fComp && getObject<RooAbsArg>(child.GetName()))) {
         auto out = (child.get<RooAbsArg>()) ? child.get<RooAbsArg>() : getObject<RooAbsArg>(child.GetName()).get();
         out->setAttribute("obs");
         out->setAttribute("global");
         return xRooNode(*out, *this);
      }
      throw std::runtime_error("Failed to add global observable");
   } else if ((strcmp(GetName(), ".poi") == 0)) {
      if (child.get<RooAbsLValue>() || (!child.fComp && getObject<RooAbsLValue>(child.GetName()))) {
         auto out = (child.get<RooAbsArg>()) ? child.get<RooAbsArg>() : getObject<RooAbsArg>(child.GetName()).get();
         out->setAttribute("poi");
         return xRooNode(*out, *this);
      } else if (!child.get() && fParent->get<RooWorkspace>()) {
         // may be creating poi at same time as adding, try add to parent
         auto res = fParent->Add(child);
         if (res.get<RooAbsLValue>())
            return Add(res);
      }
      throw std::runtime_error("Failed to add parameter of interest");
   } else if ((strcmp(GetName(), ".pars") == 0 || strcmp(GetName(), ".vars") == 0) && fParent->get<RooWorkspace>()) {
      // adding a parameter, interpret as factory string unless no "[" then create RooRealVar
      TString fac(child.GetName());
      if (!fac.Contains("["))
         fac += "[1]";
      return xRooNode(*fParent->get<RooWorkspace>()->factory(fac), fParent);
   } else if (strcmp(GetName(), ".datasets()") == 0) {
      // create a dataset - only allowed for pdfs or workspaces
      if (auto _ws = ws(); _ws && fParent) {
         sOpt.ToLower();
         if (!fParent->get<RooAbsPdf>() && (!fParent->get<RooWorkspace>() || sOpt == "asimov")) {
            throw std::runtime_error(
               "Datasets can only be created for pdfs or workspaces (except if generated dataset, then must be pdf)");
         }

         if (sOpt == "asimov" || sOpt == "toy") {
            // generate expected dataset - note that globs will be frozen at this time
            auto _fr = fParent->fitResult();
            if (strlen(_fr->GetName()) == 0) { // ensure fit result has a name so that name is saved inside dataset
               _fr.get<RooFitResult>()->SetName(TUUID().AsString());
            }
            auto ds = fParent->generate(_fr, sOpt == "asimov");
            if (strlen(child.GetName())) {
               ds.SetName(child.GetName());
               ds.get<TNamed>()->SetName(child.GetName());
            }
            if (auto _ds = ds.get<RooAbsData>()) {
               _ws->import(*_ds);
            }
            if (_fr.get<RooFitResult>()->numStatusHistory() == 0) {
               if (!GETWSSNAPSHOTS(_ws).find(_fr->GetName())) {
                  const_cast<RooLinkedList &>(GETWSSNAPSHOTS(_ws)).Add(_fr->Clone());
               }
            } else if (!_ws->obj(_fr->GetName())) {
               _ws->import((*_fr.get<RooFitResult>()));
            } // save fr to workspace, for later retrieval
            return xRooNode(*_ws->data(ds.GetName()), fParent);
         }

         auto parentObs = fParent->obs(); // may own globs so keep alive
         auto _obs = parentObs.argList();
         // put globs in a snapshot
         std::unique_ptr<RooAbsCollection> _globs(_obs.selectByAttrib("global", true));
         // RooArgSet _tmp; _tmp.add(*_globs);_ws->saveSnapshot(child.GetName(),_tmp);
         _obs.remove(*_globs);

         // include any coords
         _obs.add(coords(false).argList(), true);
         // include axis var too, provided it's an observable
         if (auto ax = GetXaxis(); ax && dynamic_cast<RooAbsArg *>(ax->GetParent())->getAttribute("obs")) {
            _obs.add(*dynamic_cast<RooAbsArg *>(ax->GetParent()));
         }
         // check if ws already has a dataset with this name, if it does we may need to extend columns
         if (auto _d = _ws->data(child.GetName()); _d) {
            // add any missing obs
            RooArgSet l(_obs);
            l.remove(*_d->get(), true, true);
            if (!l.empty()) {
               auto _dd = dynamic_cast<RooDataSet *>(_d);
               if (!_dd)
                  throw std::runtime_error("Cannot extend dataset with new columns");
               for (auto &x : l) {
                  _dd->addColumn(*x);
               }
            }
         } else {
            RooRealVar w("weightVar", "weightVar", 1);
            _obs.add(w);
            RooDataSet d(child.GetName(), child.GetTitle(), _obs, RooFit::WeightVar("weightVar"));
            _ws->import(d);
            // seems have to set bits after importing, not before
            if (auto __d = _ws->data(child.GetName()))
               __d->SetBit(1 << 20, _ws->allData().size() == 1); // sets as selected if is only ds
         }
         /*if(!_ws->data(child.GetName())) {
             RooRealVar w("weightVar", "weightVar", 1);
             RooArgSet _obs; _obs.add(w);
             RooDataSet d(child.GetName(), child.GetTitle(), _obs, "weightVar");
             _ws->import(d);
         }*/
         auto out = std::shared_ptr<TObject>(_ws->data(child.GetName()), [](TObject *) {});

         if (out) {
            xRooNode o(out, fParent);
            if (child.get<TH1>())
               o = *child.get();
            return o;
         }
      }
      throw std::runtime_error("Cannot create dataset");
   }

   if (!get()) {
      if (!fParent)
         throw std::runtime_error("Cannot add to null object with no parentage");

      auto _ref = emplace_back(std::shared_ptr<xRooNode>(&const_cast<xRooNode &>(child), [](TObject *) {}));
      try {
         fComp = fParent->Add(*this, "+").fComp;
      } catch (...) {
         resize(size() - 1);
         std::rethrow_exception(std::current_exception());
      }
      resize(size() - 1); // remove the temporarily added node

      if (!fComp) {
         throw std::runtime_error("No object");
      }
   }

   if (auto p = get<RooAbsData>(); p) {
      if (auto bb = getBrowsable(".sourceds"))
         bb->Add(child, opt);
      if (auto _data = child.get<RooDataSet>()) {
         auto ds = dynamic_cast<RooDataSet *>(p);
         if (!ds) {
            throw std::runtime_error("Can only add datasets to a dataset");
         }

         // append any missing globs, and check any existing globs have matching values
         RooArgList globsToAdd;
         auto _globs = globs();
         for (auto &glob : child.globs()) {
            if (auto g = _globs.find(glob->GetName()); !g) {
               globsToAdd.addClone(*glob->get<RooAbsArg>());
            } else if (g->GetContent() != glob->GetContent()) {
               Warning("Add", "Global observable %s=%g in dataset %s mismatches %s value %g ... ignoring latter",
                       g->GetName(), g->GetContent(), GetName(), child.GetName(), glob->GetContent());
            }
         }
         // add any existing globs to list then set the list
         if (auto _dglobs = p->getGlobalObservables()) {
            globsToAdd.addClone(*_dglobs);
         } else {
            for (auto g : _globs)
               globsToAdd.addClone(*g->get<RooAbsArg>());
         }
         p->setGlobalObservables(globsToAdd);

         // append any missing observables to our dataset, then append the dataset

         for (auto col : *_data->get()) {
            if (!p->get()->contains(*col)) {
               ds->addColumn(*col);
            }
         }
         ds->append(*_data);
         ds->SetTitle(TString(ds->GetTitle()) + " + " + _data->GetTitle());
         SetTitle(TString(GetTitle()) + " + " + child.GetTitle());
         return *this;
      }
      auto _h = child.get<TH1>();
      if (!_h) {
         throw std::runtime_error("Can only add histogram or dataset to data");
      }
      auto _pdf = parentPdf();
      if (!_pdf)
         throw std::runtime_error("Could not find pdf");
      auto _ax = _pdf->GetXaxis();
      if (!_ax) {
         throw std::runtime_error("Cannot determine binning to add data");
      }

      RooArgSet obs;
      obs.add(*dynamic_cast<RooAbsArg *>(_ax->GetParent()));
      obs.add(coords().argList()); // will also move obs to coords

      // add any missing obs
      RooArgSet l(obs);
      l.remove(*p->get(), true, true);
      if (!l.empty()) {
         auto _d = dynamic_cast<RooDataSet *>(p);
         if (!_d)
            throw std::runtime_error("Cannot extend dataset with new columns");
         for (auto &x : l) {
            _d->addColumn(*x);
         }
      }

      // before adding, ensure range is good to cover
      for (auto &o : obs) {
         if (auto v = dynamic_cast<RooRealVar *>(o); v) {
            if (auto dv = dynamic_cast<RooRealVar *>(p->get()->find(v->GetName())); dv) {
               if (v->getMin() < dv->getMin())
                  dv->setMin(v->getMin());
               if (v->getMax() > dv->getMax())
                  dv->setMax(v->getMax());
            }
         } else if (auto c = dynamic_cast<RooCategory *>(o); c) {
            if (auto dc = dynamic_cast<RooCategory *>(p->get()->find(c->GetName())); dc) {
               if (!dc->hasLabel(c->getCurrentLabel())) {
                  dc->defineType(c->getCurrentLabel(), c->getCurrentIndex());
               }
            }
         }
      }

      for (int i = 1; i <= _h->GetNbinsX(); i++) {
         if (auto cat = dynamic_cast<RooAbsCategoryLValue *>(_ax->GetParent())) {
            if (!_h->GetXaxis()->GetBinLabel(i)) {
               throw std::runtime_error(
                  TString::Format("Categorical observable %s requires bin labels", _ax->GetParent()->GetName()));
            } else if (!cat->hasLabel(_h->GetXaxis()->GetBinLabel(i))) {
               throw std::runtime_error(TString::Format("Categorical observable %s does not have label %s",
                                                        _ax->GetParent()->GetName(), _h->GetXaxis()->GetBinLabel(i)));
            } else {
               cat->setLabel(_h->GetXaxis()->GetBinLabel(i));
            }
         } else {
            dynamic_cast<RooAbsRealLValue *>(_ax->GetParent())->setVal(_h->GetBinCenter(i));
         }
         p->add(obs, _h->GetBinContent(i));
      }

      return *this;
   }

   if (auto p = get<RooAddPdf>(); p) {
      if ((child.get<RooAbsPdf>() || (!child.fComp && getObject<RooAbsPdf>(child.GetName())))) {
         auto out = (child.fComp) ? acquire(child.fComp) : getObject<RooAbsArg>(child.GetName());
         // don't add a coef if in 'all-extended' mode and this pdf is extendable
         auto _pdf = std::dynamic_pointer_cast<RooAbsPdf>(out);
         if (!_pdf) {
            throw std::runtime_error("Something went wrong with pdf acquisition");
         }

         if (auto _ax = GetXaxis(); _ax && dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()) &&
                                    _pdf->dependsOn(*static_cast<RooAbsArg *>(_ax->GetParent()))) {
            auto _p = _pdf;

            if (auto _boundaries = std::unique_ptr<std::list<double>>(_p->binBoundaries(
                   *dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()), -std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()));
                !_boundaries && _ax->GetNbins() > 0) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 24, 00)
               Warning("Add", "Adding unbinned pdf %s to binned %s - will wrap with RooBinSamplingPdf(...)",
                       _p->GetName(), GetName());
               _p = acquireNew<RooBinSamplingPdf>(TString::Format("%s_binned", _p->GetName()), _p->GetTitle(),
                                                  *dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()), *_p);
               _p->setStringAttribute("alias", std::dynamic_pointer_cast<RooAbsArg>(out)->getStringAttribute("alias"));
               if (!_p->getStringAttribute("alias"))
                  _p->setStringAttribute("alias", out->GetName());
#else
               throw std::runtime_error(
                  "unsupported addition of unbinned pdf to binned model - please upgrade to at least ROOT 6.24");
#endif
               _pdf = _p;
            }
         }

         if (!(_pdf->canBeExtended() && p->coefList().empty())) {
            // if extended, use an extended binding as the coef
            // otherwise e.g. if adding a RooRealSumPdf the stacked histograms will be above the
            // actual pdf histogram because the pdf histogram is just normalized down
            if (_pdf->canBeExtended()) {
               // FIXME: ExtendedBinding needs the obs list passing to it ... should be fixed in RooFit
               // until then, this will return "1" and so the pdf's histograms wont be normalized properly in relation
               // to stacks of its comps
               const_cast<RooArgList &>(p->coefList())
                  .add(*acquireNew<RooExtendedBinding>(TString::Format("%s_extBind", _pdf->GetName()),
                                                       TString::Format("Expected Events of %s", _pdf->GetTitle()),
                                                       *_pdf));
            } else {

               // need to create a coefficient for each existing pdf first, like above
               for (auto i = p->coefList().size(); i < p->pdfList().size(); i++) {
                  const_cast<RooArgList &>(p->coefList())
                     .add(*acquireNew<RooExtendedBinding>(
                        TString::Format("%s_extBind", p->pdfList().at(i)->GetName()),
                        TString::Format("Expected Events of %s", p->pdfList().at(i)->GetTitle()),
                        *static_cast<RooAbsPdf *>(p->pdfList().at(i))));
               }

               const_cast<RooArgList &>(p->coefList()).add(*acquire2<RooAbsArg, RooRealVar>("1", "1", 1));
            }
            // ensure not in no-coef mode any more
            *reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(p) +
                                      p->Class()->GetDataMemberOffset("_allExtendable")) = false;
            *reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(p) +
                                      p->Class()->GetDataMemberOffset("_haveLastCoef")) = true;
         }
         const_cast<RooArgList &>(p->pdfList()).add(*_pdf);
         sterilize();
         return xRooNode(*_pdf, *this);
      } else if ((child.get<TH1>() || child.get<RooAbsReal>() ||
                  (!child.get() && getObject<RooAbsReal>(child.GetName()))) &&
                 !child.get<RooAbsPdf>()) {
         RooRealSumPdf *_pdf = nullptr;
         bool tooMany(false);
         for (auto &pp : factors()) {
            if (auto _p = pp->get<RooRealSumPdf>(); _p) {
               if (_pdf) {
                  _pdf = nullptr;
                  tooMany = true;
                  break;
               } // more than one!
               _pdf = _p;
            }
         }
         if (_pdf) {
            return xRooNode(*_pdf, *this).Add(child);
         } else if (!tooMany) {
            // create a RooRealSumPdf to hold the child
            auto _sumpdf = Add(*acquireNew<RooRealSumPdf>(TString::Format("%s_samples", p->GetName()),
                                                          TString::Format("%s samples", GetTitle()), RooArgList(),
                                                          RooArgList(), true));
            _sumpdf.get<RooAbsArg>()->setStringAttribute("alias", "samples");
            return _sumpdf.Add(child);
         }
      }
   }

   if (auto p = get<RooRealSumPdf>(); p) {
      std::shared_ptr<TObject> out;
      auto cc = child.fComp;
      bool isConverted = (cc != child.convertForAcquisition(*this, sOpt));
      if (child.get<RooAbsReal>()) {
         out = acquire(child.fComp);
         if (std::dynamic_pointer_cast<TH1>(cc) && !TString(cc->GetOption()).Contains("nostyle")) {
            xRooNode(out, *this).styles(cc.get()); // transfer style if adding a histogram
         }
      }
      if (!child.fComp && getObject<RooAbsReal>(child.GetName())) {
         Info("Add", "Adding existing function %s to %s", child.GetName(), p->GetName());
         out = getObject<RooAbsReal>(child.GetName());
      }

      if (!out && !child.fComp) {
         std::shared_ptr<RooAbsArg> _func;
         // a null node .. so create either a new RooProduct or RooHistFunc if has observables (or no deps but has
         // x-axis)
         auto _obs = robs();
         if (!_obs.empty() || GetXaxis()) {
            if (_obs.empty()) {
               // using X axis to construct hist
               auto _ax = dynamic_cast<Axis2 *>(GetXaxis());
               auto t = TH1::AddDirectoryStatus();
               TH1::AddDirectory(false);
               auto h =
                  std::make_unique<TH1D>(child.GetName(), child.GetTitle(), _ax->GetNbins(), _ax->binning()->array());
               TH1::AddDirectory(t);
               h->GetXaxis()->SetName(TString::Format("%s;%s", _ax->GetParent()->GetName(), _ax->GetName()));
               // technically convertForAcquisition has already acquired so no need to re-acquire but should be harmless
               _func = std::dynamic_pointer_cast<RooAbsArg>(acquire(xRooNode(*h).convertForAcquisition(*this)));
            } else if (_obs.size() == 1) {
               // use the single obs to make a TH1D
               auto _x = _obs.at(0)->get<RooAbsLValue>();
               auto _bnames = _x->getBinningNames();
               TString binningName = p->getStringAttribute("binning");
               for (auto &b : _bnames) {
                  if (b == p->GetName()) {
                     binningName = p->GetName();
                     break;
                  }
               }
               auto t = TH1::AddDirectoryStatus();
               TH1::AddDirectory(false);
               auto h = std::make_unique<TH1D>(child.GetName(), child.GetTitle(), _x->numBins(binningName),
                                               _x->getBinningPtr(binningName)->array());
               TH1::AddDirectory(t);
               h->GetXaxis()->SetName(
                  TString::Format("%s;%s", dynamic_cast<TObject *>(_x)->GetName(), binningName.Data()));
               // technically convertForAcquisition has already acquired so no need to re-acquire but should be harmless
               _func = std::dynamic_pointer_cast<RooAbsArg>(acquire(xRooNode(*h).convertForAcquisition(*this)));
               Info("Add", "Created densityhisto factor %s (xaxis=%s) for %s", _func->GetName(), _obs.at(0)->GetName(),
                    p->GetName());
            } else {
               throw std::runtime_error("Unsupported creation of new component in SumPdf for this many obs");
            }
         } else {
            _func = acquireNew<RooProduct>(TString::Format("%s_%s", p->GetName(), child.GetName()), child.GetTitle(),
                                           RooArgList());
         }
         _func->setStringAttribute("alias", child.GetName());
         out = _func;
      }

      if (auto _f = std::dynamic_pointer_cast<RooHistFunc>(
             (child.get<RooProduct>()) ? child.factors()[child.GetName()]->fComp : out);
          _f) {
         // adding a histfunc directly to a sumpdf, should be a density
         _f->setAttribute("density");
         if (_f->getAttribute("autodensity")) {
            // need to divide by bin widths first
            for (int i = 0; i < _f->dataHist().numEntries(); i++) {
               auto bin_pars = _f->dataHist().get(i);
               _f->dataHist().set(*bin_pars, _f->dataHist().weight() / _f->dataHist().binVolume(*bin_pars));
            }
            _f->setAttribute("autodensity", false);
            _f->setValueDirty();
         }

         // promote the axis vars to observables
         // can't use original child as might refer to unacquired deps
         for (auto &x : xRooNode("tmp", _f).vars()) {
            x->get<RooAbsArg>()->setAttribute("obs");
         }
         if (isConverted) {
            Info("Add", "Created %s factor RooHistFunc::%s for %s",
                 _f->getAttribute("density") ? "densityhisto" : "histo", _f->GetName(), p->GetName());
         }
      }

      if (auto _p = std::dynamic_pointer_cast<RooAbsPdf>(out); _p) {
         // adding a pdf to a RooRealSumPdf will replace it with a RooAddPdf and put the RooRealSumPdf inside that
         // if pdf is extended will use in the "no coefficients" state, where the expectedEvents are taking from
         // the pdf integrals
         TString newName(_p->GetName());
         newName.ReplaceAll("_samples", "");
         newName += "_components";
         Warning("Add", "converting samples to components");

         if (auto _ax = GetXaxis(); _ax && dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()) &&
                                    _p->dependsOn(*static_cast<RooAbsArg *>(_ax->GetParent()))) {

            if (auto _boundaries = std::unique_ptr<std::list<double>>(_p->binBoundaries(
                   *dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()), -std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()));
                !_boundaries && _ax->GetNbins() > 0) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 24, 00)
               Warning("Add", "Adding unbinned pdf %s to binned %s - will wrap with RooBinSamplingPdf(...)",
                       _p->GetName(), GetName());
               _p = acquireNew<RooBinSamplingPdf>(TString::Format("%s_binned", _p->GetName()), _p->GetTitle(),
                                                  *dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()), *_p);
               _p->setStringAttribute("alias", std::dynamic_pointer_cast<RooAbsArg>(out)->getStringAttribute("alias"));
               if (!_p->getStringAttribute("alias"))
                  _p->setStringAttribute("alias", out->GetName());
#else
               throw std::runtime_error(
                  "unsupported addition of unbinned pdf to binned model - please upgrade to at least ROOT 6.24");
#endif
            }
         }

         // require to be extended to be in coefficient-free mode ...
         // otherwise would lose the integral of the sumPdf (can't think of way to have a coef be the integral)
         if (!_p->canBeExtended()) {
            _p = acquireNew<RooExtendPdf>(TString::Format("%s_extended", _p->GetName()), _p->GetTitle(), *_p,
                                          *acquire2<RooAbsReal, RooRealVar>("1", "1", 1));
         }

         return *(Replace(*acquireNew<RooAddPdf>(newName, _p->GetTitle(), RooArgList(*p, *_p)))
                     .browse()[1]); // returns second node.
      }

      if (auto _f = std::dynamic_pointer_cast<RooAbsReal>(out); _f) {

         // todo: if adding a pdf, should actually replace RooRealSumPdf with a RooAddPdf and put
         // the sumPdf and *this* pdf inside that pdf
         // only exception is the binSamplingPdf below to integrate unbinned functions across bins

         if (auto _ax = GetXaxis(); _ax && dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()) &&
                                    _f->dependsOn(*static_cast<RooAbsArg *>(_ax->GetParent()))) {

            if (auto _boundaries = std::unique_ptr<std::list<double>>(_f->binBoundaries(
                   *dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()), -std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()));
                !_boundaries && _ax->GetNbins() > 0) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 24, 00)
               Warning(
                  "Add",
                  "Adding unbinned function %s to binned %s - will wrap with RooRealSumPdf(RooBinSamplingPdf(...))",
                  _f->GetName(), GetName());
               auto sumPdf = acquireNew<RooRealSumPdf>(TString::Format("%s_pdfWrapper", _f->GetName()), _f->GetTitle(),
                                                       *_f, *acquire2<RooAbsArg, RooRealVar>("1", "1", 1), true);
               sumPdf->setStringAttribute("alias", _f->getStringAttribute("alias"));
               if (!sumPdf->getStringAttribute("alias"))
                  sumPdf->setStringAttribute("alias", out->GetName());
               _f = acquireNew<RooBinSamplingPdf>(TString::Format("%s_binned", _f->GetName()), _f->GetTitle(),
                                                  *dynamic_cast<RooAbsRealLValue *>(_ax->GetParent()), *sumPdf);
               _f->setStringAttribute("alias", std::dynamic_pointer_cast<RooAbsArg>(out)->getStringAttribute("alias"));
               if (!_f->getStringAttribute("alias"))
                  _f->setStringAttribute("alias", out->GetName());
#else
               throw std::runtime_error(
                  "unsupported addition of unbinned function to binned model - please upgrade to at least ROOT 6.24");
#endif
            }
         }

         const_cast<RooArgList &>(p->coefList()).add(*acquire2<RooAbsArg, RooRealVar>("1", "1", 1));
         const_cast<RooArgList &>(p->funcList()).add(*_f);
         // inherit binning if we dont have one yet
         if (!p->getStringAttribute("binning"))
            p->setStringAttribute("binning", _f->getStringAttribute("binning"));

         xRooNode _out(_f, *this);
         if (auto gf = p->getStringAttribute("global_factors"); gf) {
            TStringToken pattern(gf, ";");
            while (pattern.NextToken()) {
               auto fac = getObject<RooAbsReal>(pattern.Data());
               if (!fac) {
                  throw std::runtime_error(TString::Format("Could not find global factor %s", pattern.Data()));
               }
               _out.Multiply(fac);
            }
         }
         sterilize();
         // clear children for reload and update shared axis
         clear();
         fXAxis.reset();
         p->setStringAttribute("xvar", nullptr);
         browse();
         return _out;
      }
   } else if (auto p2 = get<RooProdPdf>(); p2) {
      // can "add" to a RooProdPdf provided trying to add a RooAbsReal not a RooAbsPdf and have a zero or 1
      // RooRealSumPdf child.convertForAcquisition(*this); - don't convert here because want generated objects named
      // after roorealsumpdf
      if (child.get<RooAbsPdf>() || (!child.get() && getObject<RooAbsPdf>(child.GetName()))) {
         // can add if 0 or 1 RooAddPdf ....
         RooAddPdf *_pdf = nullptr;
         bool tooMany(false);
         for (auto &pp : factors()) {
            if (auto _p = pp->get<RooAddPdf>(); _p) {
               if (_pdf) {
                  _pdf = nullptr;
                  tooMany = true;
                  break;
               } // more than one!
               _pdf = _p;
            }
         }
         if (_pdf) {
            return xRooNode(*_pdf, *this).Add(child);
         } else if (!tooMany) {
            auto out = this->operator[]("components")->Add(child);
            return out;
         }
      } else if ((child.get<TH1>() || child.get<RooAbsReal>() ||
                  (!child.get() && getObject<RooAbsReal>(child.GetName()))) &&
                 !child.get<RooAbsPdf>()) {
         RooRealSumPdf *_pdf = nullptr;
         RooAddPdf *_backup = nullptr;
         bool tooMany(false);
         for (auto &pp : factors()) {
            if (auto _p = pp->get<RooRealSumPdf>(); _p) {
               if (_pdf) {
                  _pdf = nullptr;
                  tooMany = true;
                  break;
               } // more than one!
               _pdf = _p;
            } else if (auto _p2 = pp->get<RooAddPdf>(); _p2) {
               _backup = _p2;
               for (auto &_pdfa : pp->components()) {
                  if (auto _p3 = _pdfa->get<RooRealSumPdf>(); _p3) {
                     if (_pdf) {
                        _pdf = nullptr;
                        tooMany = true;
                        break;
                     } // more than one!
                     _pdf = _p3;
                  }
               }
            }
         }
         if (_pdf) {
            return xRooNode(*_pdf, *this).Add(child);
         } else if (_backup) {
            // added *INSIDE* the addPdf -- will create a RooRealSumPdf to hold it
            return xRooNode(*_backup, *this).Add(child);
         } else if (!tooMany) {
            auto out = this->operator[]("samples")->Add(child);
            // clear our x-axis to re-evaluate
            fXAxis.reset();
            p2->setStringAttribute("xvar", nullptr);
            return out;
         }
      }
   } else if (auto s = get<RooSimultaneous>(); s) {

      // adding to a simultaneous means adding a bin
      return bins().Add(child);

      // if the child is a RooAbsPdf can just add it as a new channel using name of pdf as the channel name
      // if child is a histogram, will create a RooProdPdf

   } else if (auto w = get<RooWorkspace>(); w) {
      child.convertForAcquisition(
         *this, child.get() ? "" : "func" /* if child is a string, allow it to be passed to factory */);
      if (child.get()) {
         if (auto _d = child.get<RooAbsData>()) {
            // don't use acquire method to import, because that adds datasets as Embeddded
            if (!w->import(*_d)) {
               // should upgrade vars with any obs from the dataset
               if (_d->get()) {
                  std::unique_ptr<RooAbsCollection>(w->allVars().selectCommon(*_d->get()))->setAttribAll("obs");
               }
               if (_d->getGlobalObservables()) {
                  std::unique_ptr<RooAbsCollection> globs(w->allVars().selectCommon(*_d->get()));
                  globs->setAttribAll("obs");
                  globs->setAttribAll("global");
               }
               return xRooNode(child.GetName(), *w->data(child.GetName()), *this);
            } else {
               throw std::runtime_error(
                  TString::Format("Could not import dataset %s into workspace %s", child.GetName(), w->GetName())
                     .Data());
            }
         } else {
            auto out = acquire(child.fComp);
            if (out)
               return xRooNode(child.GetName(), out, *this);
         }
      }

      if (!child.empty() || child.fFolder == "!pdfs") {
         // create a RooSimultaneous using the children as the channels
         // children either have "=" in name if specifying channel cat name or otherwise assume
         std::string catName = "channelCat";
         if (!child.empty()) {
            if (TString ss = child.at(0)->GetName(); ss.Contains("=")) {
               catName = ss(0, ss.Index('='));
            }
         }
         auto _cat = acquire<RooCategory>(catName.c_str(), catName.c_str());
         _cat->setAttribute("obs");
         auto out = acquireNew<RooSimultaneous>(child.GetName(), child.GetTitle(), *_cat);
         Info("Add", "Created pdf RooSimultaneous::%s in workspace %s", out->GetName(), w->GetName());
         return xRooNode(out, *this);
      }
   }

   if (sOpt == "pdf") {
      // can only add a pdf to a workspace
      if (get<RooWorkspace>()) {
         const_cast<xRooNode &>(child).fFolder = "!pdfs";
         return Add(child);
      }
   } else if (sOpt == "channel") {
      // can add to a model or to a workspace (creates a RooProdPdf either way)
      if (get<RooSimultaneous>()) {
         return Vary(child);
      } else if (get<RooWorkspace>()) {
         std::shared_ptr<TObject> out;
         child.convertForAcquisition(*this);
         if (child.get<RooAbsPdf>()) {
            out = acquire(child.fComp);
         } else if (!child.fComp) {
            out = acquireNew<RooProdPdf>(child.GetName(),
                                         (strlen(child.GetTitle())) ? child.GetTitle() : child.GetName(), RooArgList());
            Info("Add", "Created channel RooProdPdf::%s in workspace %s", out->GetName(), get()->GetName());
         }
         return xRooNode(out, *this);
      }
   } else if (sOpt == "sample" || sOpt == "func") {
      if (get<RooProdPdf>()) {
         auto _mainChild = mainChild();
         if (_mainChild.get<RooRealSumPdf>()) {
            return _mainChild.Add(child, sOpt == "func" ? "func" : "");
         } else {
            return (*this)["samples"]->Add(child, sOpt == "func" ? "func" : "");
         }
      }
   } else if (sOpt == "dataset") {
      if (get<RooWorkspace>()) {
         // const_cast<xRooNode&>(child).fFolder = "!datasets";return Add(child);
         return (*this).datasets().Add(child);
      }
   }

   if (considerType) {

      // interpret 'adding' here as dependent on the object type ...
      if (get<RooSimultaneous>()) {
         return bins().Add(child);
      } else if (TString(child.GetName()).Contains('=')) {
         return variations().Add(child);
      } else if (get<RooProduct>() || get<RooProdPdf>()) {
         return factors().Add(child);
      }
   }

   // Nov 2022 - removed ability to add placeholders ... could bring back if rediscover need for them
   //    if (!child.get() && child.empty() && strlen(child.GetName())) {
   //        // can add a 'placeholder' node, note it will be deleted at the next browse
   //        xRooNode out(child.GetName(),nullptr,*this);
   //        out.SetTitle(child.GetTitle());
   //        emplace_back(std::make_shared<xRooNode>(out));
   //        // update the parent in the out node so that it's copy of the parent knows it has itself in it
   //        // actually maybe not want this :-/
   //        //out.fParent = std::make_shared<Node2>(*this);
   //        for(auto o : *gROOT->GetListOfBrowsers()) {
   //            if(auto b = dynamic_cast<TBrowser*>(o); b && b->GetBrowserImp()){
   //                if(auto _b = dynamic_cast<TGFileBrowser*>(
   //                dynamic_cast<TRootBrowser*>(b->GetBrowserImp())->fActBrowser ); _b) {
   //                    auto _root = _b->fRootDir;
   //                    if (!_root) _root = _b->fListTree->GetFirstItem();
   //                    if (auto item = _b->fListTree->FindItemByObj(_root,this); item) {
   //                        _b->fListTree->AddItem(item,back()->GetName(),back().get());
   //                    }
   //                }
   //            }
   //        }
   //        return out;
   //    }

   throw std::runtime_error(TString::Format("Cannot add %s to %s", child.GetName(), GetName()));
}

std::string xRooNode::GetPath() const
{
   if (!fParent)
      return GetName();
   return fParent->GetPath() + "/" + GetName();
}

xRooNode::~xRooNode()
{
   // std::cout << "deleting " << GetPath() << std::endl;
}

void xRooNode::SetHidden(bool set)
{
   if (auto a = get<RooAbsArg>()) {
      a->setAttribute("hidden", set);
      //        if(auto item = GetTreeItem(nullptr); item) {
      //            if(set) item->SetColor(kRed);
      //            else item->ClearColor();
      //        }
   }
}
bool xRooNode::IsHidden() const
{
   auto a = get<RooAbsArg>();
   if (a)
      return a->getAttribute("hidden");
   return false;
}

xRooNode xRooNode::Combine(const xRooNode &rhs)
{

   if (get() == rhs.get()) {
      // nothing to do because objects are identical
      return *this;
   }

   // Info("Combine","Combining %s into %s",rhs.GetPath().c_str(),GetPath().c_str());

   // combine components, factors, and variations ... when there is a name clash will combine on that object
   for (auto &c : rhs.components()) {
      if (auto _c = components().find(c->GetName()); _c) {
         _c->Combine(*c);
      } else {
         Add(*c);
      }
   }

   for (auto &f : rhs.factors()) {
      if (auto _f = factors().find(f->GetName()); _f) {
         _f->Combine(*f);
      } else {
         Multiply(*f);
      }
   }

   for (auto &v : rhs.variations()) {
      if (auto _v = variations().find(v->GetName()); _v) {
         _v->Combine(*v);
      } else {
         Vary(*v);
      }
   }

   // todo: Should also transfer over binnings of observables

   return *this;
}

xRooNode xRooNode::shallowCopy(const std::string &name, std::shared_ptr<xRooNode> parent)
{
   xRooNode out(name.c_str(), nullptr,
                parent /*? parent : fParent -- was passing fParent for getObject benefit before fProvider concept*/);
   // if(!parent) out.fAcquirer = true;
   if (!parent)
      out.fProvider = fParent;

   auto o = get();
   if (!o) {
      return out;
   }

   if (auto s = get<RooSimultaneous>(); s) {
      auto chans = bins();
      if (!chans.empty()) {
         // create a new RooSimultaneous with shallow copies of each channel

         std::shared_ptr<RooSimultaneous> pdf = out.acquire<RooSimultaneous>(
            name.c_str(), o->GetTitle(), const_cast<RooAbsCategoryLValue &>(s->indexCat()));

         for (auto &c : chans) {
            TString cName(c->GetName());
            cName = cName(cName.Index('=') + 1, cName.Length());
            // by passing out as the parent, will ensure out acquires everything created
            auto c_copy =
               c->shallowCopy(name + "_" + c->get()->GetName(), std::shared_ptr<xRooNode>(&out, [](xRooNode *) {}));
            pdf->addPdf(*dynamic_cast<RooAbsPdf *>(c_copy.get()), cName);
         }
         out.fComp = pdf;
         return out;
      }
   } else if (auto p = dynamic_cast<RooProdPdf *>(o); p) {
      // main pdf will be copied too
      std::shared_ptr<RooProdPdf> pdf =
         std::dynamic_pointer_cast<RooProdPdf>(out.acquire(std::shared_ptr<TObject>(p->Clone(/*name.c_str()*/)), false,
                                                           true)); // use clone to copy all attributes etc too
      auto main = mainChild();
      if (main) {
         auto newMain =
            std::dynamic_pointer_cast<RooAbsArg>(out.acquire(std::shared_ptr<TObject>(main->Clone()), false, true));
         std::cout << newMain << " " << newMain->GetName() << std::endl;
         // pdf->replaceServer(*pdf->pdfList().find(main->GetName()), *newMain, true, true);
         // const_cast<RooArgList&>(pdf->pdfList()).replace(*pdf->pdfList().find(main->GetName()), *newMain);
         pdf->redirectServers(RooArgList(*newMain));
      }
      out.fComp = pdf;
      out.sterilize();
      return out;
   }

   return out;
}

void xRooNode::Print(Option_t *opt) const
{
   static std::unique_ptr<cout_redirect> capture;
   std::string captureStr;
   bool doCapture = false;
   if (!capture && gROOT->FromPopUp()) { // FromPopUp means user executed from the context menu
      capture = std::make_unique<cout_redirect>(captureStr);
      doCapture = true;
   }

   TString sOpt(opt);
   int depth = 0;
   if (sOpt.Contains("depth=")) {
      depth = TString(sOpt(sOpt.Index("depth=") + 6, sOpt.Length())).Atoi();
      sOpt.ReplaceAll(TString::Format("depth=%d", depth), "");
   }
   int indent = 0;
   if (sOpt.Contains("indent=")) {
      indent = TString(sOpt(sOpt.Index("indent=") + 7, sOpt.Length())).Atoi();
      sOpt.ReplaceAll(TString::Format("indent=%d", indent), "");
   }
   bool _more = sOpt.Contains("m");
   if (_more)
      sOpt.Replace(sOpt.Index("m"), 1, "");
   if (sOpt != "")
      _more = true;

   if (indent == 0) { // only print self if not indenting (will already be printed above if tree traverse)
      std::cout << GetPath();
      if (get() && get() != this) {
         std::cout << ": ";
         if (_more || (get<RooAbsArg>() && get<RooAbsArg>()->isFundamental()) || get<RooConstVar>() ||
             get<RooAbsData>() || get<RooProduct>() || get<RooFitResult>()) {
            auto _deps = coords(false).argList(); // want to revert coords after print
            auto _snap = std::unique_ptr<RooAbsCollection>(_deps.snapshot());
            coords(); // move to coords before printing (in case this matters)
            get()->Print(sOpt);
            if (auto _fr = get<RooFitResult>(); _fr && dynamic_cast<RooStringVar *>(_fr->constPars().find(".log"))) {
               std::cout << "Minimization Logs:" << std::endl;
               std::cout << dynamic_cast<RooStringVar *>(_fr->constPars().find(".log"))->getVal() << std::endl;
            }
            _deps.assignValueOnly(*_snap);
            // std::cout << std::endl;
         } else {
            TString _suffix = "";
            if (auto _type = GetNodeType(); strlen(_type)) {
               // decided not to show const values until figure out how to update if value changes
               /*if (TString(_type)=="Const") _name += TString::Format("
               [%s=%g]",_type,v->get<RooConstVar>()->getVal()); else*/
               _suffix += TString::Format(" [%s]", _type);
            }
            if (auto fv = get<RooFormulaVar>()) {
               TString formu = TString::Format(" [%s]", fv->expression());
               for (size_t i = 0; i < fv->dependents().size(); i++) {
                  formu.ReplaceAll(TString::Format("x[%zu]", i), fv->dependents()[i].GetName());
               }
               _suffix += formu;
            } else if (auto gv = get<RooGenericPdf>()) {
               TString formu = TString::Format(" [%s]", gv->expression());
               for (size_t i = 0; i < gv->dependents().size(); i++) {
                  formu.ReplaceAll(TString::Format("x[%zu]", i), gv->dependents()[i].GetName());
               }
               _suffix += formu;
            }
            std::cout << get()->ClassName() << "::" << get()->GetName() << _suffix.Data() << std::endl;
         }

      } else if (!get()) {
         std::cout << std::endl;
      }
   }
   const_cast<xRooNode *>(this)->browse();
   std::vector<std::string> folderNames;
   for (auto &k : *this) {
      if (std::find(folderNames.begin(), folderNames.end(), k->fFolder) == folderNames.end()) {
         folderNames.push_back(k->fFolder);
      }
   }
   for (auto &f : folderNames) {
      int i = 0;
      int iindent = indent;
      if (!f.empty()) {
         for (int j = 0; j < indent; j++)
            std::cout << " ";
         std::cout << f << std::endl;
         iindent += 1;
      }
      for (auto &k : *this) {
         if (k->fFolder != f) {
            i++;
            continue;
         }
         for (int j = 0; j < iindent; j++)
            std::cout << " ";
         std::cout << i++ << ") " << k->GetName() << " : ";
         if (k->get()) {
            if (_more || (k->get<RooAbsArg>() && k->get<RooAbsArg>()->isFundamental()) || k->get<RooConstVar>() ||
                k->get<RooAbsData>() /*|| k->get<RooProduct>()*/) {
               auto _deps = k->coords(false).argList();
               auto _snap = std::unique_ptr<RooAbsCollection>(_deps.snapshot());
               k->coords();           // move to coords before printing (in case this matters)
               k->get()->Print(sOpt); // assumes finishes with an endl
               _deps.assignValueOnly(*_snap);
            } else {
               TString _suffix = "";
               if (auto _type = k->GetNodeType(); strlen(_type)) {
                  // decided not to show const values until figure out how to update if value changes
                  /*if (TString(_type)=="Const") _name += TString::Format("
                  [%s=%g]",_type,v->get<RooConstVar>()->getVal()); else*/
                  _suffix += TString::Format(" [%s]", _type);
               }
               if (auto fv = k->get<RooFormulaVar>()) {
                  TString formu = TString::Format(" [%s]", fv->expression());
                  for (size_t j = 0; j < fv->dependents().size(); j++) {
                     formu.ReplaceAll(TString::Format("x[%zu]", j), fv->dependents()[j].GetName());
                  }
                  _suffix += formu;
               } else if (auto gv = k->get<RooGenericPdf>()) {
                  TString formu = TString::Format(" [%s]", gv->expression());
                  for (size_t j = 0; j < gv->dependents().size(); j++) {
                     formu.ReplaceAll(TString::Format("x[%zu]", j), gv->dependents()[j].GetName());
                  }
                  _suffix += formu;
               }
               std::cout << k->get()->ClassName() << "::" << k->get()->GetName() << _suffix.Data() << std::endl;
            }
            if (depth != 0) {
               k->Print(sOpt + TString::Format("depth=%dindent=%d", depth - 1, iindent + 1));
            }
         } else
            std::cout << " NULL " << std::endl;
      }
   }
   if (doCapture) {
      capture.reset(); // no captureStr has the string to display
      // inject line breaks to avoid msgbox being too wide
      size_t lastBreak = 0;
      std::string captureStrWithBreaks;
      for (size_t i = 0; i < captureStr.size(); i++) {
         captureStrWithBreaks += captureStr[i];
         if (captureStr[i] == '\n') {
            lastBreak = i;
         }
         if (i - lastBreak > 150) {
            captureStrWithBreaks += '\n';
            lastBreak = i;
         }
      }
      const TGWindow *w =
         (gROOT->GetListOfBrowsers()->At(0))
            ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
            : gClient->GetRoot();
      new TGMsgBox(gClient->GetRoot(), w, GetName(),
                   captureStrWithBreaks.c_str()); //,nullptr,kMBDismiss,nullptr,kVerticalFrame,kTextLeft|kTextCenterY);
   }
}

xRooNode xRooNode::Constrain(const xRooNode &child)
{
   if (!child.get()) {

      if (auto v = get<RooRealVar>(); v) {

         TString constrType = child.GetName();
         double mean = std::numeric_limits<double>::quiet_NaN();
         double sigma = mean;
         if (constrType.BeginsWith("gaussian(")) {
            // extract the mean and stddev parameters
            // if only one given, it is the stddev
            if (constrType.Contains(",")) {
               mean = TString(constrType(9, constrType.Index(',') - 9)).Atof();
               sigma = TString(constrType(constrType.Index(',') + 1, constrType.Index(')') - constrType.Index(',') + 1))
                          .Atof();
            } else {
               mean = std::numeric_limits<double>::quiet_NaN(); // will use the var current value below to set mean
               sigma = TString(constrType(9, constrType.Index(')') - 9)).Atof();
            }
            constrType = "normal";
         } else if (constrType == "normal") {
            mean = 0;
            sigma = 1;
         } else if (constrType == "gaussian") {
            // extract parameters from the variable
            // use current value and error on v as constraint
            if (!v->hasError())
               throw std::runtime_error("No error on parameter for gaussian constraint");
            sigma = v->getError();
            mean = v->getVal();
            constrType = "normal";
         } else if (constrType == "poisson") {
            if (!v->hasError())
               throw std::runtime_error("No error on parameter for poisson constraint");
            mean = 1;
            sigma = pow(v->getVal() / v->getError(), 2);
         }

         if (constrType == "poisson") {
            // use current value and error on v as constraint
            double tau_val = sigma;
            auto globs = acquire<RooRealVar>(Form("globs_%s", v->GetName()), Form("globs_%s", v->GetName()),
                                             v->getVal() * tau_val, (v->getVal() - 5 * v->getError()) * tau_val,
                                             (v->getVal() + 5 * v->getError()) * tau_val);
            globs->setConstant();
            globs->setAttribute("obs");
            globs->setAttribute("global");
            globs->setStringAttribute("nominal", TString::Format("%f", tau_val));
            auto tau = acquireNew<RooConstVar>(TString::Format("tau_%s", v->GetName()), "", tau_val);
            auto constr = acquireNew<RooPoisson>(
               Form("pois_%s", v->GetName()), TString::Format("Poisson Constraint of %s", v->GetTitle()), *globs,
               *acquireNew<RooProduct>(TString::Format("mean_%s", v->GetName()),
                                       TString::Format("Poisson Constraint of %s", globs->GetTitle()),
                                       RooArgList(*v, *tau)),
               true /* no rounding */);

            auto out = Constrain(xRooNode(Form("pois_%s", GetName()), constr));
            if (!v->hasError())
               v->setError(mean / sqrt(tau_val)); // if v doesnt have an uncert, will put one on it now
            Info("Constrain", "Added poisson constraint pdf RooPoisson::%s (tau=%g) for %s", out->GetName(), tau_val,
                 GetName());
            return out;
         } else if (constrType == "normal") {

            auto globs = acquire<RooRealVar>(Form("globs_%s", v->GetName()), Form("globs_%s", v->GetName()), mean,
                                             mean - 10 * sigma, mean + 10 * sigma);
            globs->setAttribute("obs");
            globs->setAttribute("global");
            globs->setConstant();

            globs->setStringAttribute("nominal", TString::Format("%f", mean));
            auto constr = acquireNew<RooGaussian>(
               Form("gaus_%s", v->GetName()), TString::Format("Gaussian Constraint of %s", v->GetTitle()), *globs, *v,
               *acquireNew<RooConstVar>(TString::Format("sigma_%s", v->GetName()), "", sigma));
            auto out = Constrain(xRooNode(Form("gaus_%s", GetName()), constr));
            if (!v->hasError())
               v->setError(sigma); // if v doesnt have an uncert, will put one on it now
            Info("Constrain", "Added gaussian constraint pdf RooGaussian::%s (mean=%g,sigma=%g) for %s", out->GetName(),
                 mean, sigma, GetName());
            return out;
         }
      }
   } else if (auto p = child.get<RooAbsPdf>(); p) {

      auto _me = get<RooAbsArg>();
      if (!_me) {
         throw std::runtime_error("Cannot constrain non arg");
      }

      if (!p->dependsOn(*_me)) {
         throw std::runtime_error("Constraint does not depend on constrainee");
      }

      // find a parent that can swallow this pdf ... either a RooProdPdf or a RooWorkspace
      auto x = fParent;
      while (x && !x->get<RooProdPdf>() && !x->get<RooSimultaneous>() && !x->get<RooWorkspace>()) {
         x = x->fParent;
      }
      if (!x) {
         throw std::runtime_error("Nowhere to put constraint");
      }
      // get datasets of the swallower, and add glob to any globs lists
      auto childGlobs = child.globs();
      if (!childGlobs.empty()) {
         for (auto d : x->datasets()) {
            if (auto globs = d->get<RooAbsData>()->getGlobalObservables()) {
               RooArgSet newGlobs(*globs);
               newGlobs.add(*childGlobs.get<RooArgList>());
               d->get<RooAbsData>()->setGlobalObservables(newGlobs);
            }
         }
         // also add to the workspaces globalObservables lists
         if (x->ws()) {
            for (auto &[k, v] : GETWSSETS(x->ws())) {
               if (k == "globalObservables" || TString(k).EndsWith("_GlobalObservables")) {
                  const_cast<RooArgSet &>(v).add(*childGlobs.get<RooArgList>());
               }
            }
         }
      }
      if (auto s = x->get<RooSimultaneous>(); s) {
         // put into every channel that features parameter
         x->browse();
         for (auto &c : *x) {
            if (auto a = c->get<RooAbsArg>(); a->dependsOn(*_me))
               c->Multiply(child);
         }
         return child;
      } else if (x->get<RooProdPdf>()) {
         return x->Multiply(child);
      } else {
         return x->Add(child, "+");
      }
   }

   throw std::runtime_error(TString::Format("Cannot constrain %s", GetName()));
}

xRooNode xRooNode::Multiply(const xRooNode &child, Option_t *opt)
{

   class AutoUpdater {
   public:
      AutoUpdater(xRooNode &_n) : n(_n) {}
      ~AutoUpdater() { n.browse(); }
      xRooNode &n;
   };
   AutoUpdater xxx(*this);

   if (fBinNumber != -1) {
      // scaling a bin ...
      if (child.get<RooAbsReal>()) { // if not child then let fall through to create a child and call self again below
         // doing a bin-multiplication .. the parent should have a ParamHistFunc called binFactors
         // if it doesn't then create one
         auto o = std::dynamic_pointer_cast<RooAbsReal>(acquire(child.fComp));

         // get binFactor unless parent is a ParamHistFunc already ...

         auto binFactors = (fParent->get<ParamHistFunc>()) ? fParent : fParent->factors().find("binFactors");

         // it can happen in a loop over bins() that another node has moved fParent inside a product
         // so check for fParent having a client with the ORIGNAME:<name> attribute
         if (!binFactors && fParent->get<RooAbsArg>()) {
            for (auto c : fParent->get<RooAbsArg>()->clients()) {
               if (c->IsA() == RooProduct::Class() &&
                   c->getAttribute(TString::Format("ORIGNAME:%s", fParent->get()->GetName()))) {
                  // try getting binFactors out of this
                  binFactors = xRooNode(*c).factors().find("binFactors");
                  break;
               }
            }
         }

         if (!binFactors) {
            fParent
               ->Multiply(TString::Format("%s_binFactors",
                                          (fParent->mainChild().get())
                                             ? fParent->mainChild()->GetName()
                                             : (fParent->get() ? fParent->get()->GetName() : fParent->GetName()))
                             .Data(),
                          "blankshape")
               .SetName("binFactors"); // creates ParamHistFunc with all pars = 1 (shared const)
            binFactors = fParent->factors().find("binFactors");
            if (!binFactors) {
               throw std::runtime_error(
                  TString::Format("Could not create binFactors in parent %s", fParent->GetName()));
            }
            // auto phf = binFactors->get<ParamHistFunc>();

            // create RooProducts for all the bins ... so that added factors don't affect selves
            int i = 1;
            for (auto &b : binFactors->bins()) {
               auto p = acquireNew<RooProduct>(TString::Format("%s_bin%d", binFactors->get()->GetName(), i),
                                               TString::Format("binFactors of bin %d", i), RooArgList());
               p->setStringAttribute("alias", TString::Format("%s=%g", binFactors->GetXaxis()->GetParent()->GetName(),
                                                              binFactors->GetXaxis()->GetBinCenter(i)));
               b->Multiply(*p);
               i++;
            }
         }
         // then scale the relevant bin ... if the relevant bin is a "1" then just drop in our factor (inside a
         // RooProduct though, to avoid it getting modified by subsequent multiplies)
         auto _bin = binFactors->bins().at(fBinNumber - 1);
         if (auto phf = binFactors->get<ParamHistFunc>(); phf && _bin) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
            RooArgList &pSet = phf->_paramSet;
#else
            RooArgList &pSet = const_cast<RooArgList &>(phf->paramList());
#endif
            if (strcmp(_bin->GetName(), "1") == 0) {
               RooArgList all;
               for (std::size_t i = 0; i < pSet.size(); i++) {
                  if (int(i) != fBinNumber - 1) {
                     all.add(*pSet.at(i));
                  } else {
                     all.add(*o);
                  }
               }
               pSet.removeAll();
               pSet.add(all);
            } else {
               _bin->fBinNumber = -1; // to avoid infinite loop
               return _bin->Multiply(child, opt);
            }
            //                } else {else if(_bin->get<RooProduct>()) {
            //                    // multiply the element which will just add it as a factor in the rooproduct
            //                    return _bin->Multiply(child,opt);
            //                } else {
            //                    // not a rooproduct in this bin yet ... so need to replace with a rooproduct and
            //                    multiply that
            //                    // this avoids the undesired behaviour of shared binFactors getting all impacted by
            //                    mulitplies RooArgList all; auto new_p =
            //                    acquireNew<RooProduct>(TString::Format("%s_bin%d",binFactors->get()->GetName(),fBinNumber),TString::Format("binFactors
            //                    of bin %d",fBinNumber),RooArgList(*_bin->get<RooAbsArg>()));
            //                    new_p->setStringAttribute("alias","")
            //                    for (int i = 0; i < phf->_paramSet.size(); i++) {
            //                        if (i != fBinNumber - 1) all.add(*phf->_paramSet.at(i));
            //                        else all.add(*new_p);
            //                    }
            //                    phf->_paramSet.removeAll();
            //                    phf->_paramSet.add(all);
            //                    // now multiply that bin having converted it to RooProduct
            //                    return binFactors->bins().at(fBinNumber - 1)->Multiply(child,opt);
            //                }
         }
         return xRooNode(*o, binFactors);
      }
   } else if (!get() && fParent) {
      // try to 'create' object based on parentage
      // add child as a temporary child to help with decision making
      auto _ref = emplace_back(std::shared_ptr<xRooNode>(&const_cast<xRooNode &>(child), [](TObject *) {}));
      try {
         fComp = fParent->Add(*this, "+").fComp;
      } catch (...) {
         resize(size() - 1);
         std::rethrow_exception(std::current_exception());
      }
      resize(size() - 1); // remove the temporarily added node
   }

   if (!child.get()) {
      TString sOpt(opt);
      sOpt.ToLower();
      if (auto o = getObject<RooAbsReal>(child.GetName())) {
         auto out = Multiply(xRooNode(o, child.fParent));
         // have to protect bin case where get() is null (could change but then must change logic above too)
         if (get()) {
            Info("Multiply", "Scaled %s by existing factor %s::%s",
                 mainChild().get() ? mainChild().get()->GetName() : get()->GetName(), o->ClassName(), o->GetName());
         }
         return out;
      } else if (sOpt == "norm") {
         if (TString(child.GetName()).Contains("[") && ws()) {
            // assume factory method wanted
            auto arg = ws()->factory(child.GetName());
            if (arg) {
               auto out = Multiply(*arg);
               if (get()) {
                  Info("Multiply", "Scaled %s by new norm factor %s",
                       mainChild().get() ? mainChild().get()->GetName() : get()->GetName(), out->GetName());
               }
               return out;
            }
            throw std::runtime_error(TString::Format("Failed to create new normFactor %s", child.GetName()));
         }
         auto out = Multiply(RooRealVar(child.GetName(), child.GetTitle(), 1, -1e-5, 100));
         if (get()) {
            Info("Multiply", "Scaled %s by new norm factor %s",
                 mainChild().get() ? mainChild().get()->GetName() : get()->GetName(), out->GetName());
         }
         return out;
      } else if (sOpt == "shape" || sOpt == "histo" || sOpt == "blankshape") {
         // needs axis defined
         if (auto ax = GetXaxis(); ax) {
            auto h = std::shared_ptr<TH1>(BuildHistogram(dynamic_cast<RooAbsLValue *>(ax->GetParent()), true));
            h->Reset();
            for (int i = 1; i <= h->GetNbinsX(); i++) {
               h->SetBinContent(i, 1);
            }
            h->SetMinimum(0);
            h->SetMaximum(100);
            h->SetName(TString::Format(";%s", child.GetName())); // ; char indicates don't "rename" this thing
            h->SetTitle(child.GetTitle());
            if (sOpt.Contains("shape"))
               h->SetOption(sOpt);
            auto out = Multiply(*h);
            if (get()) {
               Info("Multiply", "Scaled %s by new %s factor %s",
                    mainChild().get() ? mainChild().get()->GetName() : get()->GetName(), sOpt.Data(), out->GetName());
            }
            return out;
         }
      } else if (sOpt == "overall") {
         auto out = Multiply(acquireNew<RooStats::HistFactory::FlexibleInterpVar>(
            child.GetName(), child.GetTitle(), RooArgList(), 1, std::vector<double>(), std::vector<double>()));
         if (get() /* can happen this is null if on a bin node with no shapeFactors*/) {
            Info("Multiply", "Scaled %s by new overall factor %s",
                 mainChild().get() ? mainChild().get()->GetName() : get()->GetName(), out->GetName());
         }
         return out;
      } else if (sOpt == "func" && ws()) {
         // need to get way to get dependencies .. can't pass all as causes circular dependencies issues.
         if (auto arg = ws()->factory(TString("expr::") + child.GetName())) {
            auto out = Multiply(*arg);
            if (get() /* can happen this is null if on a bin node with no shapeFactors*/) {
               Info("Multiply", "Scaled %s by new func factor %s",
                    mainChild().get() ? mainChild().get()->GetName() : get()->GetName(), out->GetName());
            }
            return out;
         }
      }
   }
   if (auto h = child.get<TH1>(); h && strlen(h->GetOption()) == 0 && strlen(opt) > 0) {
      // put the option in the hist
      h->SetOption(opt);
   }
   if (auto w = get<RooWorkspace>(); w) {
      // just acquire
      std::shared_ptr<TObject> out;
      child.convertForAcquisition(*this);
      if (child.get<RooAbsReal>())
         out = acquire(child.fComp);
      return out;
   }

   if (strcmp(GetName(), ".coef") == 0) { // covers both .coef and .coefs
      // need to add this into the relevant coef ... if its not a RooProduct, replace it with one first
      if (auto p = fParent->fParent->get<RooAddPdf>()) {
         // may be in no-coef mode ... in which case must create coefs (use "ExtendedBindings" but note that these need
         // obs list passing to them
         if (p->coefList().empty() && !p->pdfList().empty()) {
            for (auto _pdf : p->pdfList()) {
               const_cast<RooArgList &>(p->coefList())
                  .add(*acquireNew<RooExtendedBinding>(TString::Format("%s_extBind", _pdf->GetName()),
                                                       TString::Format("Expected Events of %s", _pdf->GetTitle()),
                                                       *static_cast<RooAbsPdf *>(_pdf)));
            }
            Info("Multiply", "Created RooExtendedBinding coefficients for all pdfs of %s so that can multiply coef",
                 p->GetName());
            *reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(p) +
                                      p->Class()->GetDataMemberOffset("_allExtendable")) = false;
            *reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(p) +
                                      p->Class()->GetDataMemberOffset("_haveLastCoef")) = true;
         }
         for (size_t i = 0; i < p->pdfList().size(); i++) {
            if (p->pdfList().at(i) == fParent->get<RooAbsArg>()) {
               auto coefs = p->coefList().at(i);
               if (!coefs->InheritsFrom("RooProduct")) {
                  RooArgList oldCoef;
                  if (!(strcmp(coefs->GetName(), "1") == 0 || strcmp(coefs->GetName(), "ONE") == 0))
                     oldCoef.add(*coefs);
                  auto newCoefs = fParent->acquireNew<RooProduct>(
                     TString::Format("coefs_%s", fParent->GetName()),
                     TString::Format("coefficients for %s", fParent->GetName()), oldCoef);
                  RooArgList oldCoefs;
                  for (size_t j = 0; j < p->coefList().size(); j++) {
                     if (i == j) {
                        oldCoefs.add(*newCoefs);
                     } else {
                        oldCoefs.add(*p->coefList().at(j));
                     }
                  }
                  const_cast<RooArgList &>(p->coefList()).removeAll();
                  const_cast<RooArgList &>(p->coefList()).add(oldCoefs);
                  coefs = newCoefs.get();
               }
               return xRooNode(*coefs, fParent).Multiply(child);
            }
         }
      }
      throw std::runtime_error("this coefs case is not supported");
   }

   if (auto p = get<RooProduct>(); p) {
      std::shared_ptr<TObject> out;
      auto cc = child.fComp;
      bool isConverted = (child.convertForAcquisition(*this) != cc);
      if (child.get<RooAbsReal>())
         out = acquire(child.fComp);

      // child may be a histfunc or a rooproduct of a histfunc and a paramhist if has stat errors
      if (auto _f = std::dynamic_pointer_cast<RooHistFunc>(
             (child.get<RooProduct>()) ? child.factors()[child.GetName()]->fComp : out);
          _f && _f->getAttribute("autodensity")) {
         // should we flag this as a density? yes if there's no other term marked as the density
         bool hasDensity = false;
         for (auto &f : factors()) {
            if (f->get<RooAbsArg>()->getAttribute("density")) {
               hasDensity = true;
               break;
            }
         }
         _f->setAttribute("density", !hasDensity && fParent && fParent->get<RooRealSumPdf>());
         if (_f->getAttribute("density")) {

            // need to divide by bin widths first
            for (int i = 0; i < _f->dataHist().numEntries(); i++) {
               auto bin_pars = _f->dataHist().get(i);
               _f->dataHist().set(*bin_pars, _f->dataHist().weight() / _f->dataHist().binVolume(*bin_pars));
            }
            _f->setValueDirty();

            // promote the axis vars to observables
            for (auto &x : xRooNode("tmp", _f).vars()) {
               x->get<RooAbsArg>()->setAttribute("obs");
            }
         }
         _f->setAttribute("autodensity", false);
      }

      if (isConverted && child.get<RooHistFunc>()) {
         Info("Multiply", "Created %s factor %s in %s",
              child.get<RooAbsArg>()->getAttribute("density") ? "densityhisto" : "histo", child->GetName(),
              p->GetName());
      } else if (isConverted && child.get<ParamHistFunc>()) {
         Info("Multiply", "Created shape factor %s in %s", child->GetName(), p->GetName());
      }

      if (auto _f = std::dynamic_pointer_cast<RooAbsReal>(out); _f) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
         p->_compRSet.add(*_f);
#else
         const_cast<RooArgList &>(p->realComponents()).add(*_f);
#endif
         p->setValueDirty();

         browse();
         xRooNode _out(_f, *this);
         for (auto &_par : _out.pars()) {
            if (auto s = _par->get<RooAbsArg>()->getStringAttribute("boundConstraint"); s) {
               bool found = false;
               for (auto &_constr : _par->constraints()) {
                  if (strcmp(s, _constr->get()->GetName()) == 0) {
                     // constraint is already included
                     found = true;
                     break;
                  }
               }
               if (!found) {
                  Info("Multiply", "Pulling in %s boundConstraint: %s", _par->GetName(), s);
                  auto _pdf = getObject<RooAbsPdf>(s);
                  if (!_pdf) {
                     throw std::runtime_error("Couldn't find boundConstraint");
                  }
                  _par->Constrain(_pdf);
               }
            }
         }
         sterilize();
         return _out;
      }
   } else if (auto p2 = get<RooProdPdf>(); p2) {

      std::shared_ptr<TObject> out;
      child.convertForAcquisition(*this);
      if (child.get<RooAbsPdf>()) {
         out = acquire(child.fComp);
      } else if (child.get<RooAbsReal>() && mainChild().get<RooRealSumPdf>()) {
         // cannot multiply a RooProdPdf by a non pdf
         throw std::runtime_error(TString::Format("Cannot multiply %s by non-pdf %s", GetName(), child.GetName()));
         // return mainChild().Add(child); - nov 2022 - used to do this but now replaced with exception above
      } else if (!child.get() || child.get<RooAbsReal>()) {
         // need to create or hide inside a sumpdf or rooadpdf
         std::shared_ptr<RooAbsPdf> _pdf;
         if (!child.get() && strcmp(child.GetName(), "components") == 0) {
            auto _sumpdf = acquireNew<RooAddPdf>(
               Form("%s_%s", p2->GetName(), child.GetName()),
               (strlen(child.GetTitle()) && strcmp(child.GetTitle(), child.GetName())) ? child.GetTitle()
                                                                                       : p2->GetTitle(),
               RooArgList() /*, RooArgList() forces coef-mode if we specify this list */);
            _pdf = _sumpdf;
         } else {
            auto _sumpdf = acquireNew<RooRealSumPdf>(
               Form("%s_%s", p2->GetName(), child.GetName()),
               (strlen(child.GetTitle()) && strcmp(child.GetTitle(), child.GetName())) ? child.GetTitle()
                                                                                       : p2->GetTitle(),
               RooArgList(), RooArgList(), true);
            _sumpdf->setFloor(true);
            _pdf = _sumpdf;
         }
         _pdf->setStringAttribute("alias", child.GetName());
         // transfer axis attributes if present (TODO: should GetXaxis look beyond the immediate parent?)
         _pdf->setStringAttribute("xvar", p2->getStringAttribute("xvar"));
         _pdf->setStringAttribute("binning", p2->getStringAttribute("binning"));
         out = _pdf;
         Info("Multiply", "Created %s::%s in channel %s", _pdf->ClassName(), _pdf->GetName(), p2->GetName());
         if (child.get<RooAbsReal>())
            xRooNode(*out, *this).Add(child);
      }

      if (auto _pdf = std::dynamic_pointer_cast<RooAbsPdf>(out); _pdf) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
         const_cast<RooArgList &>(p2->pdfList()).add(*_pdf);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
         p2->_pdfNSetList.emplace_back(std::make_unique<RooArgSet>("nset"));
#else
         p->_pdfNSetList.Add(new RooArgSet("nset"));
#endif
         if (!p2->canBeExtended() && _pdf->canBeExtended()) {
            p2->_extendedIndex = p2->_pdfList.size() - 1;
         }
#else
         p2->addPdfs(RooArgSet(*_pdf));
#endif
         sterilize();
         browse();
         return xRooNode(_pdf, *this);
      }
   } else if (auto p3 = get<RooRealSumPdf>(); p3) {
      // multiplying all current and future components
      std::shared_ptr<TObject> out;
      child.convertForAcquisition(*this);
      if (child.get<RooAbsReal>()) {
         out = acquire(child.fComp);
         for (auto &c : components()) {
            c->Multiply(out);
         }
         TString s = p3->getStringAttribute("global_factors");
         if (s != "")
            s += ";";
         s += out->GetName();
         p3->setStringAttribute("global_factors", s);
         Info(
            "Multiply",
            "Flagged %s as a global factor in channel %s (is applied to all current and future samples in the channel)",
            out->GetName(), p3->GetName());
         return xRooNode(out, *this);
      }

   } else if (auto p4 = get<RooAbsPdf>(); p4 && !(fParent && fParent->get<RooRealSumPdf>())) {
      // multiply the coefs (if this isn't part of a RooAddPdf or RooRealSumPdf then we will eventually throw exception
      return coefs().Multiply(child);
   } else if (auto p5 = get<RooAbsReal>(); p5 && (!get<RooAbsPdf>() || (fParent && fParent->get<RooRealSumPdf>()))) {
      // replace this obj with a RooProduct to allow for multiplication

      // get the list of clients BEFORE creating the new interpolation ... seems list of clients is inaccurate after
      std::set<RooAbsArg *> cl;
      for (auto &arg : p5->clients()) {
         cl.insert(arg);
      }

      // if multiple clients, see if only one client is in parentage route
      // if so, then assume thats the only client we should replace in
      if (cl.size() > 1) {
         if (cl.count(fParent->get<RooAbsArg>()) > 0) {
            cl.clear();
            cl.insert(fParent->get<RooAbsArg>());
         } else {
            Warning("Multiply", "Scaling %s that has multiple clients", p5->GetName());
         }
      }

      auto new_p = acquireNew<RooProduct>(TString::Format("prod_%s", p5->GetName()), p5->GetTitle(), RooArgList(*p5));
      // copy attributes over
      for (auto &a : p5->attributes())
         new_p->setAttribute(a.c_str());
      for (auto &a : p5->stringAttributes())
         new_p->setStringAttribute(a.first.c_str(), a.second.c_str());
      if (!new_p->getStringAttribute("alias"))
         new_p->setStringAttribute("alias", p5->GetName());
      auto old_p = p5;
      new_p->setAttribute(Form("ORIGNAME:%s", old_p->GetName())); // used in redirectServers to say what this replaces
      for (auto arg : cl) {
         arg->redirectServers(RooArgSet(*new_p), false, true);
      }

      fComp = new_p;
      return Multiply(child);
   }

   // before giving up here, assume user wanted a norm factor type if child is just a name
   if (!child.get() && strlen(opt) == 0)
      return Multiply(child, "norm");

   throw std::runtime_error(
      TString::Format("Cannot multiply %s by %s%s", GetPath().c_str(), child.GetName(),
                      (!child.get() && strlen(opt) == 0) ? " (forgot to specify factor type?)" : ""));
}

xRooNode xRooNode::Replace(const xRooNode &node)
{

   auto p5 = get<RooAbsArg>();
   if (!p5) {
      throw std::runtime_error("Only replacement of RooAbsArg is supported");
   }
   node.convertForAcquisition(*this, "func");

   auto new_p = node.get<RooAbsArg>();
   if (!new_p) {
      throw std::runtime_error(TString::Format("Cannot replace with %s", node.GetName()));
   }
   auto out = acquire(node.fComp);
   new_p = std::dynamic_pointer_cast<RooAbsArg>(out).get();

   std::set<RooAbsArg *> cl;
   for (auto &arg : p5->clients()) {
      if (arg == new_p)
         continue; // do not replace in self ... although redirectServers will prevent that anyway
      cl.insert(arg);
   }

   // if multiple clients, see if only one client is in parentage route
   // if so, then assume thats the only client we should replace in
   if (cl.size() > 1) {
      if (fParent && fParent->get<RooAbsArg>() && cl.count(fParent->get<RooAbsArg>()) > 0) {
         cl.clear();
         cl.insert(fParent->get<RooAbsArg>());
      } else {
         std::stringstream clientList;
         for (auto c : cl)
            clientList << c->GetName() << ",";
         Warning("Replace", "Replacing %s in all clients: %s", p5->GetName(), clientList.str().c_str());
      }
   }

   new_p->setAttribute(Form("ORIGNAME:%s", p5->GetName())); // used in redirectServers to say what this replaces
   for (auto arg : cl) {
      // if RooFormulaVar need to ensure the internal formula has been "constructed" otherwise will try to construct
      // it from the original expression that may have old parameter in it.
      if (auto p = dynamic_cast<RooFormulaVar *>(arg))
         p->ok(); // triggers creation of RooFormula
      arg->redirectServers(RooArgSet(*new_p), false, true);
   }
   return node;
}

xRooNode xRooNode::Vary(const xRooNode &child)
{

   class AutoUpdater {
   public:
      AutoUpdater(xRooNode &_n) : n(_n) {}
      ~AutoUpdater() { n.browse(); }
      xRooNode &n;
   };
   AutoUpdater xxx(*this);

   if (!get() && fParent) {
      // try to 'create' object based on parentage
      // add child as a temporary child to help with decision making
      auto _ref = emplace_back(std::shared_ptr<xRooNode>(&const_cast<xRooNode &>(child), [](TObject *) {}));
      try {
         fComp = fParent->Add(*this, "+").fComp;
      } catch (...) {
         resize(size() - 1);
         std::rethrow_exception(std::current_exception());
      }
      resize(size() - 1); // remove the temporarily added node
   }

   if (auto p = mainChild(); p) {
      // variations applied to the main child if has one
      return p.Vary(child);
   }

   if (auto s = get<RooSimultaneous>(); s && s->indexCat().IsA() == RooCategory::Class()) {
      // name is used as cat label
      std::string label = child.GetName();
      if (auto pos = label.find('='); pos != std::string::npos)
         label = label.substr(pos + 1);
      if (!s->indexCat().hasLabel(label)) {
         static_cast<RooCategory &>(const_cast<RooAbsCategoryLValue &>(s->indexCat())).defineType(label.c_str());
      }
      std::shared_ptr<TObject> out;
      child.convertForAcquisition(*this);
      if (child.get<RooAbsPdf>()) {
         out = acquire(child.fComp); // may create a channel from a histogram
      } else if (!child.fComp) {
         out = acquireNew<RooProdPdf>(TString::Format("%s_%s", s->GetName(), label.c_str()),
                                      (strlen(child.GetTitle())) ? child.GetTitle() : label.c_str(), RooArgList());
         Info("Vary", "Created channel RooProdPdf::%s in model %s", out->GetName(), s->GetName());
      }

      if (auto _pdf = std::dynamic_pointer_cast<RooAbsPdf>(out); _pdf) {
         s->addPdf(*_pdf, label.c_str());
         sterilize();
         // clear children for reload and update shared axis
         clear();
         fXAxis.reset();
         browse();
         return xRooNode(TString::Format("%s=%s", s->indexCat().GetName(), label.data()), _pdf, *this);
      }

   } else if (auto p = get<RooStats::HistFactory::FlexibleInterpVar>(); p) {

      // child needs to be a constvar ...
      child.convertForAcquisition(*this);
      auto _c = child.get<RooConstVar>();
      if (!_c && child.get()) {
         throw std::runtime_error("Only pure consts can be set as variations of a flexible interpvar");
      }
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      double value = (_c ? _c->getVal() : p->_nominal);
      double nomVal = p->_nominal;
#else
      double value = (_c ? _c->getVal() : p->nominal());
      double nomVal = p->nominal();
#endif

      TString cName(child.GetName());
      if (cName == "nominal") {
         p->setNominal(value);
         return *(this->variations().at(cName.Data()));
      }
      if (cName.CountChar('=') != 1) {
         throw std::runtime_error("unsupported variation form");
      }
      std::string parName = cName(0, cName.Index('='));
      double parVal = TString(cName(cName.Index('=') + 1, cName.Length())).Atof();
      if (parVal != 1 && parVal != -1) {
         throw std::runtime_error("unsupported variation magnitude");
      }
      bool high = parVal > 0;

      if (parName.empty()) {
         p->setNominal(value);
      } else {
         auto v = fParent->getObject<RooRealVar>(parName);
         if (!v)
            v = fParent->acquire<RooRealVar>(parName.c_str(), parName.c_str(), -5, 5);
         if (!v->hasError())
            v->setError(1);

         if (!p->findServer(*v)) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
            p->_paramList.add(*v);
            p->_low.push_back(0);
            p->_high.push_back(0);
            p->_interpCode.push_back(4);
#else
            const_cast<RooListProxy &>(p->variables()).add(*v);
            const_cast<std::vector<double> &>(p->low()).push_back(0);
            const_cast<std::vector<double> &>(p->high()).push_back(0);
            const_cast<std::vector<int> &>(p->interpolationCodes()).push_back(4);
#endif
            v->setAttribute(Form("SYMMETRIC%s_%s", high ? "+" : "-", GetName())); // flag for symmetrized
         }

         if (high) {
            p->setHigh(*v, value);
            if (v->getAttribute(Form("SYMMETRIC+_%s", GetName()))) {
               p->setLow(*v, 2 * nomVal - value);
            }
            v->setAttribute(Form("SYMMETRIC-_%s", GetName()), false);
         } else {
            p->setLow(*v, value);
            if (v->getAttribute(Form("SYMMETRIC-_%s", GetName()))) {
               p->setHigh(*v, 2 * nomVal - value);
            }
            v->setAttribute(Form("SYMMETRIC+_%s", GetName()), false);
         }

         /*if (!unconstrained && fParent->pars()[v->GetName()].constraints().empty()) {
             fParent->pars()[v->GetName()].constraints().add("normal");
         }*/
      }
      return *(this->variations().at(cName.Data()));
   } else if (auto p2 = get<PiecewiseInterpolation>(); p2) {
      TString cName(child.GetName());
      if (cName.CountChar('=') != 1) {
         throw std::runtime_error("unsupported variation form");
      }
      TString parName = cName(0, cName.Index('='));
      double parVal = TString(cName(cName.Index('=') + 1, cName.Length())).Atof();
      if (parVal != 1 && parVal != -1) {
         throw std::runtime_error("unsupported variation magnitude");
      }
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      RooHistFunc *f = dynamic_cast<RooHistFunc *>(p2->_nominal.absArg());
      if (!f) {
         throw std::runtime_error(
            TString::Format("Interpolating %s instead of RooHistFunc", p2->_nominal.absArg()->ClassName()));
      }
#else
      RooHistFunc *f = dynamic_cast<RooHistFunc *>(const_cast<RooAbsReal *>(p2->nominalHist()));
      if (!f) {
         throw std::runtime_error(
            TString::Format("Interpolating %s instead of RooHistFunc", p2->nominalHist()->ClassName()));
      }
#endif
      RooHistFunc *nomf = f;
      RooHistFunc *otherf = nullptr;
      size_t i = 0;
      for (auto par : p2->paramList()) {
         if (parName == par->GetName()) {
            f = dynamic_cast<RooHistFunc *>((parVal > 0 ? p2->highList() : p2->lowList()).at(i));
            otherf = dynamic_cast<RooHistFunc *>((parVal > 0 ? p2->lowList() : p2->highList()).at(i));
            break;
         }
         i++;
      }
      if (i == p2->paramList().size() && !child.get<RooAbsReal>()) {

         // need to add the parameter
         auto v = acquire<RooRealVar>(parName, parName, -5, 5);
         if (!v->hasError())
            v->setError(1);

         std::shared_ptr<RooHistFunc> up(
            static_cast<RooHistFunc *>(f->Clone(Form("%s_%s_up", f->GetName(), parName.Data()))));
         std::shared_ptr<RooHistFunc> down(
            static_cast<RooHistFunc *>(f->Clone(Form("%s_%s_down", f->GetName(), parName.Data()))));
         // RooHistFunc doesn't clone it's data hist ... do it ourself (will be cloned again if imported into a ws)
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
         std::unique_ptr<RooDataHist> h1(
            static_cast<RooDataHist *>(f->dataHist().Clone(Form("hist_%s", up->GetName()))));
         std::unique_ptr<RooDataHist> h2(
            static_cast<RooDataHist *>(f->dataHist().Clone(Form("hist_%s", down->GetName()))));
         up->_dataHist = dynamic_cast<RooDataHist *>(f->dataHist().Clone(Form("hist_%s", up->GetName())));
         down->_dataHist = dynamic_cast<RooDataHist *>(f->dataHist().Clone(Form("hist_%s", down->GetName())));
#else
         up->cloneAndOwnDataHist(TString::Format("hist_%s", up->GetName()));
         down->cloneAndOwnDataHist(TString::Format("hist_%s", down->GetName()));
#endif
         auto ups = std::dynamic_pointer_cast<RooHistFunc>(acquire(up, false, true));
         auto downs = std::dynamic_pointer_cast<RooHistFunc>(acquire(down, false, true));
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
         p2->_highSet.add(*ups.get());
         p2->_lowSet.add(*downs.get());
         p2->_interpCode.push_back(4);
         p2->_paramSet.add(*v);
#else
         const_cast<RooArgList &>(p2->highList()).add(*ups);
         const_cast<RooArgList &>(p2->lowList()).add(*downs);
         const_cast<std::vector<int> &>(p2->interpolationCodes()).push_back(4);
         const_cast<RooArgList &>(p2->paramList()).add(*v);
#endif
         p2->setValueDirty();
         f = ((parVal > 0) ? ups : downs).get();
         otherf = ((parVal > 0) ? downs : ups).get();
         // start off with everything being symmetric
         f->setStringAttribute("symmetrizes", otherf->GetName());
         f->setStringAttribute("symmetrize_nominal", nomf->GetName());
         otherf->setStringAttribute("symmetrized_by", f->GetName());

         // constrain par if required
         /*if (!unconstrained && fParent->pars()[v->GetName()].constraints().empty()) {
             fParent->pars()[v->GetName()].constraints().add("normal");
         }*/
      }

      // child.convertForAcquisition(*this);
      if (f) {
         if (child.get())
            xRooNode("tmp", *f, *this) = *child.get();
         f->setValueDirty();
         xRooNode out(*f, *this);
         out.sterilize();
         return out;
      }

   } else if (auto p3 = get<RooConstVar>(); p3) {

      // never vary the universal consts ... its too dangerous
      if (p3->getAttribute("RooRealConstant_Factory_Object")) {
         throw std::runtime_error("Cannot vary pure constants");
      }

      // inject a FlexibleInterpVar ...

      // get the list of clients BEFORE creating the new interpolation ... seems list of clients is inaccurate after
      std::set<RooAbsArg *> cl;
      for (auto &arg : p3->clients()) {
         cl.insert(arg);
      }
      // if multiple clients, see if only one client is in parentage route
      // if so, then assume thats the only client we should replace in
      if (cl.size() > 1) {
         if (cl.count(fParent->get<RooAbsArg>()) > 0) {
            cl.clear();
            cl.insert(fParent->get<RooAbsArg>());
         } else {
            Warning("Vary", "Varying %s that has multiple clients", p3->GetName());
         }
      }
      p3->setStringAttribute("origName", p3->GetName());
      TString n = p3->GetName();
      p3->SetName(Form("%s_nominal", p3->GetName())); // if problems should perhaps not rename here

      auto new_p = acquireNew<RooStats::HistFactory::FlexibleInterpVar>(n, p3->GetTitle(), RooArgList(), p3->getVal(),
                                                                        std::vector<double>(), std::vector<double>());

      // copy attributes over
      for (auto &a : p3->attributes())
         new_p->setAttribute(a.c_str());
      for (auto &a : p3->stringAttributes())
         new_p->setStringAttribute(a.first.c_str(), a.second.c_str());
      // if (!new_p->getStringAttribute("alias")) new_p->setStringAttribute("alias",p->GetName());
      auto old_p = p3;
      new_p->setAttribute(Form("ORIGNAME:%s", old_p->GetName())); // used in redirectServers to say what this replaces
      for (auto arg : cl) {
         arg->redirectServers(RooArgSet(*new_p), false, true);
      }

      fComp = new_p;
      return Vary(child);

   } else if (auto p4 = get<RooAbsReal>(); p4) {
      // inject an interpolation node

      // get the list of clients BEFORE creating the new interpolation ... seems list of clients is inaccurate after
      std::set<RooAbsArg *> cl;
      for (auto &arg : p4->clients()) {
         cl.insert(arg);
      }
      // if multiple clients, see if only one client is in parentage route
      // if so, then assume thats the only client we should replace in
      if (cl.size() > 1) {
         if (cl.count(fParent->get<RooAbsArg>()) > 0) {
            cl.clear();
            cl.insert(fParent->get<RooAbsArg>());
         } else {
            Warning("Vary", "Varying %s that has multiple clients", p4->GetName());
         }
      }
      p4->setStringAttribute("origName", p4->GetName());
      TString n = p4->GetName();
      p4->SetName(Form("%s_nominal", p4->GetName())); // if problems should perhaps not rename here

      auto new_p = acquireNew<PiecewiseInterpolation>(n, p4->GetTitle(), *p4, RooArgList(), RooArgList(), RooArgList());

      // copy attributes over
      for (auto &a : p4->attributes())
         new_p->setAttribute(a.c_str());
      for (auto &a : p4->stringAttributes())
         new_p->setStringAttribute(a.first.c_str(), a.second.c_str());
      // if (!new_p->getStringAttribute("alias")) new_p->setStringAttribute("alias",p->GetName());
      auto old_p = p4;
      new_p->setAttribute(Form("ORIGNAME:%s", old_p->GetName())); // used in redirectServers to say what this replaces
      for (auto arg : cl) {
         arg->redirectServers(RooArgSet(*new_p), false, true);
      }

      fComp = new_p;
      return Vary(child);
   }

   Print();
   throw std::runtime_error(TString::Format("Cannot vary %s with %s", GetName(), child.GetName()));
}

bool xRooNode::SetContent(double value)
{
   return SetContents(RooConstVar(GetName(), GetTitle(), value));
}

bool xRooNode::SetContent(double value, const char *par, double val)
{
   return SetContents(RooConstVar(GetName(), GetTitle(), value), par, val);
}

struct BinningRestorer {
   ~BinningRestorer()
   {
      if (x && b)
         x->setBinning(*b);
      if (b)
         delete b;
   }
   RooRealVar *x = nullptr;
   RooAbsBinning *b = nullptr;
};

xRooNode &xRooNode::operator=(const TObject &o)
{

   if (!get()) {
      fComp = std::shared_ptr<TObject>(const_cast<TObject *>(&o), [](TObject *) {});
      if (fParent && !fParent->find(GetName())) {
         // either a temporary or a placeholder so need to try genuinely adding
         fComp = fParent->Add(*this, "+").fComp;
         if (auto a = get<RooAbsArg>(); a && strcmp(a->GetName(), GetName()) && !a->getStringAttribute("alias")) {
            a->setStringAttribute("alias", GetName());
         }
         if (!fComp)
            throw std::runtime_error("Cannot determine type");
         return *this;
      }
   }

   if (auto h = dynamic_cast<const TH1 *>(&o); h) {
      /*auto f = get<RooHistFunc>();
      if (!f) {
          // if it's a RooProduct locate child with the same name
          if (get<RooProduct>()) {
              f = factors()[GetName()]->get<RooHistFunc>();
          }



      }*/
      bool _isData = get<RooAbsData>();
      BinningRestorer _b;
      if (_isData) {
         // need to ensure x-axis matches this h
         auto ax = GetXaxis();
         if (!ax)
            throw std::runtime_error("no xaxis");
         auto _v = dynamic_cast<RooRealVar *>(ax->GetParent());
         if (_v) {
            _b.x = _v;
            _b.b = dynamic_cast<RooAbsBinning *>(_v->getBinningPtr(nullptr)->Clone());
            if (h->GetXaxis()->IsVariableBinSize()) {
               _v->setBinning(RooBinning(h->GetNbinsX(), h->GetXaxis()->GetXbins()->GetArray()));
            } else {
               _v->setBinning(RooUniformBinning(h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax(), h->GetNbinsX()));
            }
         }
      }

      if (true) {
         for (int bin = 1; bin <= h->GetNbinsX(); bin++) {
            SetBinContent(bin, h->GetBinContent(bin));
            /*double value = h->GetBinContent(bin);
            auto bin_pars = f->dataHist().get(bin - 1);
            if (f->getAttribute("density")) {
                value /= f->dataHist().binVolume(*bin_pars);
            }
            f->dataHist().set(*bin_pars, value);*/
            if (!_isData && h->GetSumw2N() && !SetBinError(bin, h->GetBinError(bin)))
               throw std::runtime_error("Failed setting stat error");
         }
         return *this;
      }
   } else if (auto _c = dynamic_cast<const RooConstVar *>(&o); _c) {

      if (auto a = get<RooAbsArg>();
          (a && a->isFundamental()) || get<RooConstVar>() || get<RooStats::HistFactory::FlexibleInterpVar>()) {
         SetBinContent(1, _c->getVal());
         return *this;
      } else if (get<RooAbsData>()) { // try to do assignment to a dataset (usually setting a bin content)
         SetBinContent(0, _c->getVal());
         return *this;
      }
   }

   throw std::runtime_error("Assignment failed");

   /*

   if (fParent && !fParent->mk()) {
       throw std::runtime_error("mk failure");
   }

   if (fComp) return *this;

   if (o.InheritsFrom("RooAbsArg")) {
       fComp = acquire(std::shared_ptr<TObject>(const_cast<TObject*>(&o),[](TObject* o){}));
       std::dynamic_pointer_cast<RooAbsArg>(fComp)->setStringAttribute("alias",GetName());
   }

   if (fComp && fParent) {
       fParent->incorporate(fComp);
   }


   return *this;
    */
}

void xRooNode::_fit_(const char *constParValues)
{
   try {
      auto _pars = pars();
      // std::unique_ptr<RooAbsCollection> snap(_pars.argList().snapshot());
      TStringToken pattern(constParValues, ",");
      std::map<RooAbsRealLValue *, double> valsToSet;
      while (pattern.NextToken()) {
         auto idx = pattern.Index('=');
         TString pat = (idx == -1) ? TString(pattern) : TString(pattern(0, idx));
         double val =
            (idx == -1) ? std::numeric_limits<double>::quiet_NaN() : TString(pattern(idx + 1, pattern.Length())).Atof();
         for (auto p : _pars.argList()) {
            if (TString(p->GetName()).Contains(TRegexp(pat, true))) {
               p->setAttribute("Constant", true);
               if (!std::isnan(val)) {
                  valsToSet[dynamic_cast<RooAbsRealLValue *>(p)] = val;
                  // dynamic_cast<RooAbsRealLValue *>(p)->setVal(val); // don't set yet, to allow for asimov dataset
                  // creation based on current values
               }
            }
         }
      }
      // use the first selected dataset
      auto _dsets = datasets();
      TString dsetName = "";
      for (auto &d : _dsets) {
         if (d->get()->TestBit(1 << 20)) {
            dsetName = d->get()->GetName();
            break;
         }
      }
      auto _nll = nll(dsetName.Data());
      // can now set the values
      for (auto [p, v] : valsToSet) {
         p->setVal(v);
      }
      _nll.fitConfigOptions()->SetValue("LogSize", 65536);
      _nll.fitConfig()->MinimizerOptions().SetPrintLevel(0);
      auto fr = _nll.minimize();
      //_pars.argList() = *snap; // restore values - irrelevant as SetFitResult will restore values
      if (!fr.get())
         throw std::runtime_error("Fit Failed");
      SetFitResult(fr.get());
      TString statusCodes;
      for (unsigned int i = 0; i < fr->numStatusHistory(); i++) {
         statusCodes += TString::Format("\n%s = %d", fr->statusLabelHistory(i), fr->statusCodeHistory(i));
      }
      const TGWindow *w =
         (gROOT->GetListOfBrowsers()->At(0))
            ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
            : gClient->GetRoot();
      if (fr->status() != 0) {
         new TGMsgBox(gClient->GetRoot(), w, "Fit Finished with Bad Status Code",
                      TString::Format("%s\nData = %s\nFit Status Code = %d\nCov Quality = %d\n-------------%s",
                                      fr->GetName(), dsetName.Data(), fr->status(), fr->covQual(), statusCodes.Data()),
                      kMBIconExclamation, kMBOk);
      } else if (fr->covQual() != 3 && _nll.fitConfig()->ParabErrors()) {
         new TGMsgBox(gClient->GetRoot(), w, "Fit Finished with Bad Covariance Quality",
                      TString::Format("%s\nData = %s\nFit Status Code = %d\nCov Quality = %d\n-------------%s",
                                      fr->GetName(), dsetName.Data(), fr->status(), fr->covQual(), statusCodes.Data()),
                      kMBIconExclamation, kMBOk);
      } else {
         new TGMsgBox(gClient->GetRoot(), w, "Fit Finished Successfully",
                      TString::Format("%s\nData = %s\nFit Status Code = %d\nCov Quality = %d\n-------------%s",
                                      fr->GetName(), dsetName.Data(), fr->status(), fr->covQual(), statusCodes.Data()));
      }
   } catch (const std::exception &e) {
      new TGMsgBox(
         gClient->GetRoot(),
         (gROOT->GetListOfBrowsers()->At(0))
            ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
            : gClient->GetRoot(),
         "Exception", e.what(), kMBIconExclamation, kMBOk); // deletes self on dismiss?
   }
}

void xRooNode::_generate_(const char *datasetName, bool expected)
{
   try {
      datasets().Add(datasetName, expected ? "asimov" : "toy");
   } catch (const std::exception &e) {
      new TGMsgBox(
         gClient->GetRoot(),
         (gROOT->GetListOfBrowsers()->At(0))
            ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
            : gClient->GetRoot(),
         "Exception", e.what(),
         kMBIconExclamation); // deletes self on dismiss?
   }
}

void xRooNode::_scan_(const char *what, double nToys, const char *xvar, int nBinsX, double lowX,
                      double highX /*, const char*, int, double, double*/, const char *constParValues)
{
   try {
      TString sXvar(xvar);
      TString sWhat(what);

      // use the first selected dataset
      auto _dsets = datasets();
      TString dsetName = "";
      for (auto &d : _dsets) {
         if (d->get()->TestBit(1 << 20)) {
            dsetName = d->get()->GetName();
            break;
         }
      }
      auto _pars = pars();
      std::unique_ptr<RooAbsCollection> snap(_pars.argList().snapshot());
      TStringToken pattern(constParValues, ",");
      while (pattern.NextToken()) {
         auto idx = pattern.Index('=');
         TString pat = (idx == -1) ? TString(pattern) : TString(pattern(0, idx));
         double val =
            (idx == -1) ? std::numeric_limits<double>::quiet_NaN() : TString(pattern(idx + 1, pattern.Length())).Atof();
         for (auto par : _pars.argList()) {
            if (TString(par->GetName()).Contains(TRegexp(pat, true))) {
               par->setAttribute("Constant", true);
               if (!std::isnan(val)) {
                  dynamic_cast<RooAbsRealLValue *>(par)->setVal(val);
               }
            }
         }
      }
      auto hs = nll(dsetName.Data()).hypoSpace(sXvar);
      if (nToys) {
         sWhat += " toys";
         if (nToys > 0) {
            sWhat += TString::Format("=%g", nToys);
         }
      }
      hs.SetTitle(sWhat + " scan" + ((dsetName != "") ? TString::Format(" [data=%s]", dsetName.Data()) : ""));
      int scanStatus = hs.scan(sWhat + " visualize", nBinsX, lowX, highX);
      if (scanStatus != 0) {
         new TGMsgBox(
            gClient->GetRoot(),
            (gROOT->GetListOfBrowsers()->At(0))
               ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
               : gClient->GetRoot(),
            "Scan Finished with Bad Status Code",
            TString::Format("%s\nData = %s\nScan Status Code = %d", hs.GetName(), dsetName.Data(), scanStatus),
            kMBIconExclamation, kMBOk);
      }
      hs.SetName(TUUID().AsString());
      if (ws()) {
         if (auto res = hs.result())
            ws()->import(*res);
      }

      _pars.argList() = *snap; // restore pars

   } catch (const std::exception &e) {
      new TGMsgBox(
         gClient->GetRoot(),
         (gROOT->GetListOfBrowsers()->At(0))
            ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
            : gClient->GetRoot(),
         "Exception", e.what(), kMBIconExclamation);
   }
}

void xRooNode::_SetBinContent_(int bin, double value, const char *par, double parVal)
{
   try {
      SetBinContent(bin, value, strlen(par) > 0 ? par : nullptr, parVal);
   } catch (const std::exception &e) {
      new TGMsgBox(gClient->GetRoot(), gClient->GetRoot(), "Exception", e.what(),
                   kMBIconExclamation); // deletes self on dismiss?
   }
}

void xRooNode::_SetContent_(double value)
{
   try {
#if ROOT_VERSION_CODE > ROOT_VERSION(6, 29, 00)
      // if this is a collection of values, populate a TF1 and display as a dialog
      if (!get() && TString(GetName()).BeginsWith("!")) {
         browse();
         RooArgList args;
         for (auto a : *this) {
            if (auto arg = a->get<RooRealVar>())
               args.add(*arg);
         }
         TF1 f(GetName(), 0.0, 1.0, std::min(int(args.size()), 10));
         int i = 0;
         int j = 0;
         for (auto c : args) {
            j++;
            if (j < value) {
               continue;
            }
            auto v = dynamic_cast<RooRealVar *>(c);
            f.SetParName(i, c->GetName());
            if (v) {
               f.SetParLimits(i, v->getMin(), v->getMax());
               if (v->isConstant())
                  f.FixParameter(i, v->getVal());
               else {
                  f.SetParameter(i, v->getVal());
                  f.SetParError(i, v->getError());
               }
            }
            i++;
            if (i == 10) {
               break; // max 10 pars shown
            }
         }
         int ret = 0;
         new TFitParametersDialog(
            gClient->GetDefaultRoot(),
            (gROOT->GetListOfBrowsers()->At(0))
               ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
               : gClient->GetDefaultRoot(),
            &f, nullptr, &ret);
         if (ret) {
            // user has changed parameter values etc, propagate back to parameters
            for (i = 0; i < f.GetNpar(); i++) {
               auto c = args.find(f.GetParName(i));
               auto v = dynamic_cast<RooRealVar *>(c);
               if (v) {
                  v->setVal(f.GetParameter(i));
                  double low, high;
                  f.GetParLimits(i, low, high);
                  if (low == high) {
                     v->setConstant(low); // if low==high==0 then is not marked constant
                  } else {
                     v->setRange(low, high);
                  }
               }
            }
         }
         return;
      }
#endif

      if (!SetContent(value))
         throw std::runtime_error("Failed to SetContent");
   } catch (const std::exception &e) {
      new TGMsgBox(gClient->GetRoot(), gClient->GetRoot(), "Exception", e.what(),
                   kMBIconExclamation); // deletes self on dismiss?
   }
}

bool xRooNode::SetBinContent(int bin, double value, const char *par, double parVal)
{

   // create if needed
   if (!get()) {
      if (fParent && !find(GetName())) {
         // if have a binning we create a histogram to match it
         if (auto ax = GetXaxis(); ax) {
            std::shared_ptr<TH1D> h;
            auto _b = dynamic_cast<Axis2 *>(ax)->binning();
            auto t = TH1::AddDirectoryStatus();
            TH1::AddDirectory(false);
            if (_b->isUniform()) {
               h.reset(new TH1D(GetName(), GetTitle(), _b->numBins(), _b->lowBound(), _b->highBound()));
            } else {
               h.reset(new TH1D(GetName(), GetTitle(), _b->numBins(), _b->array()));
            }
            h->SetOption("nostyle"); // don't transfer style when added
            h->SetDirectory(nullptr);
            TH1::AddDirectory(t);
            h->GetXaxis()->SetName(TString::Format("%s;%s", ax->GetParent()->GetName(), ax->GetName()));
            fComp = h;
         }
         fComp = fParent->Add(*this, "sample").fComp;
      }
   }

   // if it's a RooProduct locate child with the same name
   if (get<RooProduct>()) {
      return factors()[GetName()]->SetBinContent(bin, value, par, parVal);
   }

   if (get<RooAbsData>()) {
      if (auto _data = get<RooDataSet>(); _data) {
         auto _ax = (bin) ? GetXaxis() : nullptr;
         if (!_ax && bin) {
            throw std::runtime_error("Cannot determine binning to fill data");
         }
         if (_ax && _ax->GetNbins() < bin) {
            throw std::out_of_range(TString::Format("%s range %s only has %d bins", _ax->GetParent()->GetName(),
                                                    _ax->GetName(), _ax->GetNbins()));
         }
         RooArgSet obs;

         TString cut = "";

         for (auto _c : coords()) { // coords() moves vars to their respective coordinates too
            if (auto _cat = _c->get<RooAbsCategoryLValue>(); _cat) {
               if (cut != "")
                  cut += " && ";
               cut += TString::Format("%s==%d", _cat->GetName(), _cat->getCurrentIndex());
               obs.add(*_cat); // note: if we ever changed coords to return clones, would need to keep coords alive
            } else if (auto _rv = _c->get<RooAbsRealLValue>(); _rv) {
               // todo: check coordRange is a single range rather than multirange
               if (cut != "")
                  cut += " && ";
               cut +=
                  TString::Format("%s>=%f&&%s<%f", _rv->GetName(), _rv->getMin(_rv->getStringAttribute("coordRange")),
                                  _rv->GetName(), _rv->getMax(_rv->getStringAttribute("coordRange")));
               obs.add(*_rv); // note: if we ever changed coords to return clones, would need to keep coords alive
            } else {
               throw std::runtime_error("SetBinContent of data: Unsupported coordinate type");
            }
         }

         RooFormulaVar cutFormula("cut1", cut, obs); // doing this to avoid complaints about unused vars
         RooFormulaVar icutFormula("icut1", TString::Format("!(%s)", cut.Data()), obs);

         TString cut2;
         if (_ax) {
            cut2 = TString::Format("%s >= %f && %s < %f", _ax->GetParent()->GetName(), _ax->GetBinLowEdge(bin),
                                   _ax->GetParent()->GetName(), _ax->GetBinUpEdge(bin));
            obs.add(*dynamic_cast<RooAbsArg *>(_ax->GetParent()));
         } else {
            cut2 = "1==1";
         }
         RooFormulaVar cutFormula2("cut2", cut + " && " + cut2, obs);
         RooFormulaVar icutFormula2("icut2", TString::Format("!(%s && %s)", cut.Data(), cut2.Data()), obs);

         //            // go up through parents looking for slice obs
         //            auto _p = fParent;
         //            while(_p) {
         //                TString pName(_p->GetName());
         //                if (auto pos = pName.Index('='); pos != -1) {
         //                    if(auto _obs = _p->getObject<RooAbsLValue>(pName(0,pos)); _obs) {
         //                        if(auto _cat = dynamic_cast<RooAbsCategoryLValue*>(_obs.get()); _cat) {
         //                            _cat->setLabel(pName(pos+1,pName.Length()));
         //                            cut += TString::Format("%s%s==%d", (cut=="")?"":" && ",_cat->GetName(),
         //                            _cat->getCurrentIndex());
         //                        } else if(auto _var = dynamic_cast<RooAbsRealLValue*>(_obs.get()); _var) {
         //                            _var->setVal(TString(pName(pos+1,pName.Length())).Atof());
         //                            // TODO: Cut for this!!
         //                        }
         //                        obs.add(*dynamic_cast<RooAbsArg*>(_obs.get()));
         //                    } else {
         //                        throw std::runtime_error("Unknown observable, could not find");
         //                    }
         //                }
         //                _p = _p->fParent;
         //            }

         // add observables to dataset if necessary
         RooArgSet l(obs);
         l.remove(*_data->get(), true, true);
         if (!l.empty()) {
            // addColumns method is buggy: https://github.com/root-project/root/issues/8787
            // incredibly though, addColumn works??
            for (auto &x : l) {
               _data->addColumn(*x);
            }
            // instead create a copy dataset and merge it into current
            // cant use merge because it drops weightVar
            /*RooDataSet tmp("tmp","tmp",l);
            for(int i=0;i<_data->numEntries();i++) tmp.add(l);
            _data->merge(&tmp);*/
            // delete _data->addColumns(l);
         }
         // before adding, ensure range is good to cover
         for (auto &o : obs) {
            if (auto v = dynamic_cast<RooRealVar *>(o); v) {
               if (auto dv = dynamic_cast<RooRealVar *>(_data->get()->find(v->GetName())); dv) {
                  if (v->getMin() < dv->getMin())
                     dv->setMin(v->getMin());
                  if (v->getMax() > dv->getMax())
                     dv->setMax(v->getMax());
               }
            } else if (auto c = dynamic_cast<RooCategory *>(o); c) {
               if (auto dc = dynamic_cast<RooCategory *>(_data->get()->find(c->GetName())); dc) {
                  if (!dc->hasLabel(c->getCurrentLabel())) {
                     dc->defineType(c->getCurrentLabel(), c->getCurrentIndex());
                  }
               }
            }
         }

         // using SetBinContent means dataset must take on a binned form at these coordinates
         // if number of entries doesnt match number of bins then will 'bin' the data
         if (bin) {
            if (auto _nentries = std::unique_ptr<RooAbsData>(_data->reduce(cutFormula))->numEntries();
                _nentries != _ax->GetNbins()) {
               auto _contents = GetBinContents(1, _ax->GetNbins());

               if (_nentries > 0) {
                  Info("SetBinContent", "Binning %s in channel: %s", GetName(), cut.Data());
                  auto _reduced = std::unique_ptr<RooAbsData>(_data->reduce(icutFormula));
                  _data->reset();
                  for (int j = 0; j < _reduced->numEntries(); j++) {
                     auto _obs = _reduced->get(j);
                     _data->add(*_obs, _reduced->weight());
                  }
               }
               for (int i = 1; i <= _ax->GetNbins(); i++) {
                  // can skip over the bin we will be setting to save a reduce step below
                  if (i == bin)
                     continue;
                  dynamic_cast<RooAbsLValue *>(_ax->GetParent())->setBin(i - 1, _ax->GetName());
                  _data->add(obs, _contents.at(i - 1));
               }
            }
         }
         // remove existing entries
         if (std::unique_ptr<RooAbsData>(_data->reduce(cutFormula2))->numEntries() > 0) {
            auto _reduced = std::unique_ptr<RooAbsData>(_data->reduce(icutFormula2));
            _data->reset();
            for (int j = 0; j < _reduced->numEntries(); j++) {
               auto _obs = _reduced->get(j);
               _data->add(*_obs, _reduced->weight());
            }
         }
         if (_ax)
            dynamic_cast<RooAbsLValue *>(_ax->GetParent())->setBin(bin - 1, _ax->GetName());
         _data->add(obs, value);
         if (auto bb = getBrowsable(".sourceds"))
            return bb->SetBinContent(bin, value, par, parVal); // apply to source ds if we have one
         return true;

      } else if (get<RooDataHist>()) {
         throw std::runtime_error("RooDataHist not supported yet");
      }
   }

   if (auto _varies = variations(); !_varies.empty() || (par && strlen(par))) {
      if (!par || strlen(par) == 0) {
         return _varies["nominal"]->SetBinContent(bin, value, par, parVal);
      } else if (auto it = _varies.find(Form("%s=%g", par, parVal)); it) {
         return it->SetBinContent(bin, value);
      } else {
         // need to create the variation : note - if no variations existed up to now this will inject a new node
         // so we should redirect ourself to the new node
         // TODO: Do we need to redirect parents?
         TString s = Form("%s=%g", par, parVal);
         return Vary(s.Data()).SetBinContent(bin, value);
      }
   }

   auto o = get();
   if (auto p = dynamic_cast<RooRealVar *>(o); p) {
      if (!par || strlen(par) == 0) {
         if (p->getMax() < value)
            p->setMax(value);
         if (p->getMin() > value)
            p->setMin(value);
         p->setVal(value);
         sterilize();
         return true;
      }

   } else if (auto c = dynamic_cast<RooConstVar *>(o); c) {

      // if parent is a FlexibleInterpVar, change the value in that .
      if (strcmp(c->GetName(), Form("%g", c->getVal())) == 0) {
         c->SetNameTitle(Form("%g", value), Form("%g", value));
      }
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 24, 00)
      c->_value = value; // in future ROOT versions there is a changeVal method!
#else
      c->changeVal(value);
#endif

      if (fParent && fParent->get<RooStats::HistFactory::FlexibleInterpVar>()) {
         fParent->Vary(*this);
      }

      sterilize();
      return true;
   } else if (auto f = dynamic_cast<RooHistFunc *>(o); f) {
      auto bin_pars = f->dataHist().get(bin - 1);
      if (f->getAttribute("density")) {
         value /= f->dataHist().binVolume(*bin_pars);
      }
      f->dataHist().set(*bin_pars, value);
      f->setValueDirty();

      if (auto otherfName = f->getStringAttribute("symmetrized_by"); otherfName) {
         // broken symmetry, so update flags ...
         f->setStringAttribute("symmetrized_by", nullptr);
         if (auto x = getObject<RooAbsArg>(otherfName); x) {
            x->setStringAttribute("symmetrizes", nullptr);
            x->setStringAttribute("symmetrize_nominal", nullptr);
         }
      } else if (auto otherfName2 = f->getStringAttribute("symmetrizes"); otherfName2) {
         auto nomf = getObject<RooHistFunc>(f->getStringAttribute("symmetrize_nominal"));
         auto otherf = getObject<RooHistFunc>(otherfName2);
         if (nomf && otherf) {
            otherf->dataHist().set(*bin_pars, 2 * nomf->dataHist().weight(bin - 1) - value);
            otherf->setValueDirty();
         }
      }
      sterilize();
      return true;
   } else if (auto f2 = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(o); f2) {
      // changing nominal value
      f2->setNominal(value);
   }
   throw std::runtime_error(TString::Format("unable to set bin content of %s", GetPath().c_str()));
}

bool xRooNode::SetBinData(int bin, double value, const xRooNode &data)
{
   if (data.get<RooAbsData>()) {
      // attach as a child before calling datasets(), so that is included in the list
      push_back(std::make_shared<xRooNode>(data));
   }
   auto node = datasets()[data.GetName()];
   if (data.get<RooAbsData>()) {
      // remove the child we attached
      resize(size() - 1);
   }
   return node->SetBinContent(bin, value);
}

bool xRooNode::SetData(const TObject &obj, const xRooNode &data)
{
   if (data.get<RooAbsData>()) {
      // attach as a child before calling datasets(), so that is included in the list
      push_back(std::make_shared<xRooNode>(data));
   }
   auto node = datasets()[data.GetName()];
   if (data.get<RooAbsData>()) {
      // remove the child we attached
      resize(size() - 1);
   }
   return node->SetContents(obj);
}

bool xRooNode::SetBinError(int bin, double value)
{

   // if it's a RooProduct locate child with the same name
   if (get<RooProduct>()) {
      return factors()[GetName()]->SetBinError(bin, value);
   }

   if (auto _varies = variations(); !_varies.empty()) {
      return _varies["nominal"]->SetBinError(bin, value);
   }

   auto o = get();

   if (auto f = dynamic_cast<RooHistFunc *>(o); f) {

      // if (f->getAttribute("density")) { value /= f->dataHist().binVolume(*bin_pars); } - commented out because DON'T
      // convert .. sumw and sumw2 attributes will be stored not as densities

      // NOTE: Can only do this because factors() makes parents of its children it's own parent (it isn't the parent)
      // If ever make factors etc part of the parentage then this would need tweaking to get to the true parent
      // find first parent that is a RooProduct, that is where the statFactor would live
      // stop as soon as we reach pdf object
      auto _prodParent = fParent;
      while (_prodParent && !_prodParent->get<RooProduct>() && !_prodParent->get<RooAbsPdf>()) {
         if (_prodParent->get<PiecewiseInterpolation>() && strcmp(GetName(), "nominal")) {
            _prodParent.reset();
            break; // only the 'nominal' variation can look for a statFactor outside the variation container
         }
         _prodParent = _prodParent->fParent;
      }
      auto _f_stat =
         (_prodParent && !_prodParent->get<RooAbsPdf>()) ? _prodParent->factors().find("statFactor") : nullptr;
      auto f_stat = (_f_stat) ? _f_stat->get<ParamHistFunc>() : nullptr;
      if (_f_stat && _f_stat->get() && !f_stat) {
         throw std::runtime_error("stat factor must be a paramhistfunc");
      }

      // stat uncertainty lives in the "statFactor" factor, each sample has its own one,
      // but they can share parameters
      if (!f_stat) {
         if (value == 0)
            return true;
         TString parNames;
         for (auto &p : xRooNode("tmp", *f, std::shared_ptr<xRooNode>(nullptr)).vars()) {
            if (parNames != "")
               parNames += ",";
            parNames += p->get()->GetName();
         }
         auto h = std::unique_ptr<TH1>(f->dataHist().createHistogram(parNames
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 27, 00)
                                                                     ,
                                                                     RooCmdArg::none()
#endif
                                                                        ));
         h->Reset();
         h->SetName("statFactor");
         h->SetTitle(TString::Format("StatFactor of %s", f->GetTitle()));
         h->SetOption("blankshape");

         // multiply parent if is nominal
         auto toMultiply = this;
         if (strcmp(GetName(), "nominal") == 0 && fParent && fParent->get<PiecewiseInterpolation>())
            toMultiply = fParent.get();

         f_stat = dynamic_cast<ParamHistFunc *>(toMultiply->Multiply(*h).get());
         if (!f_stat) {
            throw std::runtime_error("Failed creating stat shapeFactor");
         }
      }

      auto phf = f_stat;

      TString prefix = f->getStringAttribute("statPrefix");
      if (value && prefix == "") {
         // find the first parent that can hold components (RooAddPdf, RooRealSumPdf, RooAddition, RooWorkspace) ... use
         // that name for the stat factor
         auto _p = fParent;
         while (_p && !(_p->get()->InheritsFrom("RooRealSumPdf") || _p->get()->InheritsFrom("RooAddPdf") ||
                        _p->get()->InheritsFrom("RooWorkspace") || _p->get()->InheritsFrom("RooAddition"))) {
            _p = _p->fParent;
         }
         prefix = TString::Format("stat_%s", (_p && _p->get<RooAbsReal>()) ? _p->get()->GetName() : f->GetName());
      }
      auto newVar = (value == 0) ? getObject<RooRealVar>("1")
                                 : acquire<RooRealVar>(Form("%s_bin%d", prefix.Data(), bin),
                                                       Form("%s_bin%d", prefix.Data(), bin), 1);
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      RooArgList &pSet = phf->_paramSet;
#else
      RooArgList &pSet = const_cast<RooArgList &>(phf->paramList());
#endif
      auto var = dynamic_cast<RooRealVar *>(&pSet[bin - 1]);

      if (newVar.get() != var) {
         // need to swap out var for newVar
         // replace ith element in list with new func, or inject into RooProduct
         RooArgList all;
         for (std::size_t i = 0; i < pSet.size(); i++) {
            if (int(i) != bin - 1) {
               all.add(*pSet.at(i));
            } else {
               all.add(*newVar);
            }
         }
         pSet.removeAll();
         pSet.add(all);
      }

      xRooNode v((value == 0) ? *var : *newVar, *this);
      auto rrv = dynamic_cast<RooRealVar *>(v.get());
      if (strcmp(rrv->GetName(), "1") != 0) {
         TString origName = (f->getStringAttribute("origName")) ? f->getStringAttribute("origName") : GetName();
         rrv->setStringAttribute(Form("sumw2_%s", origName.Data()), TString::Format("%f", pow(value, 2)));
         auto bin_pars = f->dataHist().get(bin - 1);
         auto _binContent = f->dataHist().weight();
         if (f->getAttribute("density")) {
            _binContent *= f->dataHist().binVolume(*bin_pars);
         }
         rrv->setStringAttribute(Form("sumw_%s", origName.Data()), TString::Format("%f", _binContent));
         double sumw2 = 0;
         double sumw = 0;
         for (auto &[s, sv] : rrv->stringAttributes()) {
            if (s.find("sumw_") == 0) {
               sumw += TString(sv).Atof();
            } else if (s.find("sumw2_") == 0) {
               sumw2 += TString(sv).Atof();
            }
         }
         if (sumw2 && sumw2 != std::numeric_limits<double>::infinity()) {
            double tau = pow(sumw, 2) / sumw2;
            rrv->setError((tau < 1e-15) ? 1e15 : (/*rrv->getVal()*/ 1. / sqrt(tau))); // not sure why was rrv->getVal()?
            rrv->setConstant(false);
            // parameter must be constrained
            auto _constr = v.constraints();
            // std::cout << " setting constraint " << v.GetName() << " nomin=" << tau << std::endl;
            if (_constr.empty()) {
               rrv->setStringAttribute("boundConstraint", _constr.Add("poisson").get()->GetName());
            } else {
               auto _glob = _constr.at(0)->obs().at(0)->get<RooRealVar>();
               // TODO: Update any globs snapshots that are designed to match the nominal
               _glob->setStringAttribute("nominal", TString::Format("%f", tau));
               double _min = tau * (1. - 5. * sqrt(1. / tau));
               double _max = tau * (1. + 5. * sqrt(1. / tau));
               _glob->setRange(_min, _max);
               _glob->setVal(tau);
               _constr.at(0)->pp().at(0)->SetBinContent(0, tau);
               rrv->setStringAttribute("boundConstraint", _constr.at(0)->get()->GetName());
            }
            rrv->setRange(std::max((1. - 5. * sqrt(1. / tau)), 1e-15), 1. + 5. * sqrt(1. / tau));
         } else {
            // remove constraint
            if (auto _constr = v.constraints(); !_constr.empty()) {
               v.constraints().Remove(*_constr.at(0));
            }
            // set const if sumw2 is 0 (i.e. no error)
            rrv->setVal(1);
            rrv->setError(0);
            rrv->setConstant(sumw2 == 0);
         }
      }

      return true;
   }

   throw std::runtime_error(TString::Format("%s SetBinError failed", GetName()));
}

std::shared_ptr<xRooNode> xRooNode::at(const std::string &name, bool browseResult) const
{
   auto res = find(name, browseResult);
   if (res == nullptr)
      throw std::out_of_range(name + " does not exist");
   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// The RooWorkspace this node belong to, if any

RooWorkspace *xRooNode::ws() const
{
   if (auto _w = get<RooWorkspace>(); _w)
      return _w;
   if (auto a = get<RooAbsArg>(); a && GETWS(a)) {
      return GETWS(a);
   }
   if (fParent)
      return fParent->ws();
   return nullptr;
}

xRooNode xRooNode::constraints() const
{

   xRooNode out(".constraints", nullptr, *this);

   std::function<RooAbsPdf *(const xRooNode &n, RooAbsArg &par, std::set<RooAbsPdf *> ignore)> getConstraint;
   getConstraint = [&](const xRooNode &n, RooAbsArg &par, std::set<RooAbsPdf *> ignore) {
      if (auto _pdf = n.get<RooAbsPdf>()) {
         if (ignore.count(_pdf))
            return (RooAbsPdf *)nullptr;
         ignore.insert(_pdf);
      }
      auto o = n.get<RooProdPdf>();
      if (!o) {
         if (n.get<RooSimultaneous>()) {
            // check all channels for a constraint if is simultaneous
            for (auto &c : n.bins()) {
               if (auto oo = getConstraint(*c, par, ignore); oo) {
                  return oo;
               }
            }
            return (RooAbsPdf *)nullptr;
         } else if (n.get<RooAbsPdf>() && n.fParent && n.fParent->get<RooWorkspace>()) {
            // reached top-level pdf, which wasn't a simultaneous, so stop here
            return (RooAbsPdf *)nullptr;
         } else if (auto _ws = n.get<RooWorkspace>(); _ws) {
            // reached a workspace, check for any pdf depending on parameter that isnt the ignore
            for (auto p : _ws->allPdfs()) {
               if (ignore.count(static_cast<RooAbsPdf *>(p)))
                  continue;
               if (p->dependsOn(par)) {
                  out.emplace_back(std::make_shared<xRooNode>(par.GetName(), *p, *this));
               }
            }
         }
         if (!n.fParent)
            return (RooAbsPdf *)nullptr;
         return getConstraint(*n.fParent, par, ignore);
      }
      for (auto p : o->pdfList()) {
         if (ignore.count(static_cast<RooAbsPdf *>(p)))
            continue;
         if (p->dependsOn(par)) {
            out.emplace_back(std::make_shared<xRooNode>(par.GetName(), *p, *this));
         }
      }
      return (RooAbsPdf *)nullptr;
   };

   for (auto &p : vars()) {
      auto v = dynamic_cast<RooAbsReal *>(p->get());
      if (!v)
         continue;
      if (v->getAttribute("Constant") && v != get<RooAbsReal>())
         continue; // skip constants unless we are getting the constraints of a parameter itself
      if (v->getAttribute("obs"))
         continue; // skip observables ... constraints constrain pars not obs
      getConstraint(*this, *v, {get<RooAbsPdf>()});
      /*if (auto c = ; c) {
          out.emplace_back(std::make_shared<Node2>(p->GetName(), *c, *this));
      }*/
   }

   // finish by removing any constraint that contains another constraint for the same par
   // and consolidate common pars
   auto it = out.std::vector<std::shared_ptr<xRooNode>>::begin();
   while (it != out.std::vector<std::shared_ptr<xRooNode>>::end()) {
      bool removeIt = false;
      for (auto &c : out) {
         if (c.get() == it->get())
            continue;
         if ((*it)->get<RooAbsArg>()->dependsOn(*c->get<RooAbsArg>())) {
            removeIt = true;
            std::set<std::string> parNames;
            std::string _cName = c->GetName();
            do {
               parNames.insert(_cName.substr(0, _cName.find(';')));
               _cName = _cName.substr(_cName.find(';') + 1);
            } while (_cName.find(';') != std::string::npos);
            parNames.insert(_cName);
            _cName = it->get()->GetName();
            do {
               parNames.insert(_cName.substr(0, _cName.find(';')));
               _cName = _cName.substr(_cName.find(';') + 1);
            } while (_cName.find(';') != std::string::npos);
            parNames.insert(_cName);
            _cName = "";
            for (auto &x : parNames) {
               if (!_cName.empty())
                  _cName += ";";
               _cName += x;
            }
            c->TNamed::SetName(_cName.c_str());
            break;
         }
      }
      if (removeIt) {
         it = out.erase(it);
      } else {
         ++it;
      }
   }

   // if getting constraints of a fundamental then use the constraint names instead of the par name (because would be
   // all same otherwise)
   if (get<RooAbsArg>() && get<RooAbsArg>()->isFundamental()) {
      for (auto &o : out) {
         o->TNamed::SetName(o->get()->GetName());
      }
   }

   return out;
}

std::shared_ptr<TObject> xRooNode::convertForAcquisition(xRooNode &acquirer, const char *opt) const
{

   TString sOpt(opt);
   sOpt.ToLower();
   TString sName(GetName());
   if (sOpt == "func")
      sName = TString("factory:") + sName;

   // if arg is a histogram, will acquire it as a RooHistFunc unless no conversion
   // todo: could flag not to convert
   if (auto h = get<TH1>(); h) {
      TString sOpt2(h->GetOption());
      std::map<std::string, std::string> stringAttrs;
      while (sOpt2.Contains("=")) {
         auto pos = sOpt2.Index("=");
         auto start = sOpt2.Index(";") + 1;
         if (start > pos)
            start = 0;
         auto end = sOpt2.Index(";", pos);
         if (end == -1)
            end = sOpt2.Length();
         stringAttrs[sOpt2(start, pos - start)] = sOpt2(pos + 1, end - pos - 1);
         sOpt2 = TString(sOpt2(0, start)) + TString(sOpt2(end + 1, sOpt2.Length()));
      }
      TString newObjName = GetName();
      TString origName = GetName();
      if (origName.BeginsWith(';'))
         origName = origName(1, origName.Length());
      if (newObjName.BeginsWith(';')) {
         newObjName =
            newObjName(1, newObjName.Length()); // special case if starts with ';' then don't create a fancy name
      } else if (acquirer.get() && !acquirer.get<RooWorkspace>()) {
         newObjName = TString::Format(
            "%s_%s", (acquirer.mainChild().get()) ? acquirer.mainChild()->GetName() : acquirer->GetName(),
            newObjName.Data());
      }
      // can convert to a RooHistFunc, or RooParamHist if option contains 'shape'
      TString varName = h->GetXaxis()->GetName();
      std::string binningName = newObjName.Data();
      if (auto pos = varName.Index(';'); pos != -1) {
         binningName = varName(pos + 1, varName.Length());
         varName = varName(0, pos);
      }

      if (varName == "xaxis" &&
          !acquirer.get<RooSimultaneous>()) { // default case, try to take axis var and binning from the acquirer
         if (auto ax = acquirer.GetXaxis(); ax) {
            varName = ax->GetParent()->GetName();
            // TODO: check the binning is consistent before using - at least will check nBins below
            binningName = ax->GetName();
         } else if (acquirer.obs().size() == 1)
            varName = acquirer.obs().at(0)->get()->GetName(); // TODO what if no obs but Xaxis var is defined?
      }
      auto x = acquirer.acquire<RooRealVar>(varName, h->GetXaxis()->GetTitle(), h->GetXaxis()->GetXmin(),
                                            h->GetXaxis()->GetXmax());
      if (x->getMin() > h->GetXaxis()->GetXmin())
         x->setMin(h->GetXaxis()->GetXmin());
      if (x->getMax() < h->GetXaxis()->GetXmax())
         x->setMax(h->GetXaxis()->GetXmax());
      if (!x->hasBinning(binningName.c_str())) {
         if (h->GetXaxis()->IsVariableBinSize()) {
            x->setBinning(RooBinning(h->GetNbinsX(), h->GetXaxis()->GetXbins()->GetArray()), binningName.c_str());
         } else {
            x->setBinning(
               RooUniformBinning(h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax(), h->GetXaxis()->GetNbins()),
               binningName.c_str());
         }
         x->getBinning(binningName.c_str()).SetTitle(h->GetXaxis()->GetTitle());
         if (x->getBinningNames().size() == 2) {
            // this was the first binning, so copy it over to be the default binning too
            x->setBinning(x->getBinning(binningName.c_str()));
         }
      } else {
         // TODO check binning is compatible with histogram
         if (x->getBinning(binningName.c_str()).numBins() != h->GetNbinsX()) {
            throw std::runtime_error(
               TString::Format("binning mismatch for binning %s of %s", binningName.c_str(), x->GetName()));
         }
      }

      std::shared_ptr<RooAbsArg> _f;

      // if acquirer is a RooSimultaneous, will use histogram to define a channel
      if (acquirer.get<RooSimultaneous>()) {
         _f = acquirer.acquireNew<RooProdPdf>(newObjName, (strlen(h->GetTitle())) ? h->GetTitle() : h->GetName(),
                                              RooArgList());
         for (auto &[k, v] : stringAttrs) {
            _f->setStringAttribute(k.c_str(), v.c_str());
         }
         x->setAttribute("obs", true);
      } else if (sOpt2.Contains("shape")) {
         RooArgList list;
         for (int i = 0; i < x->getBinning(binningName.c_str()).numBins(); i++) {
            std::shared_ptr<RooAbsArg> arg;
            if (sOpt2.Contains("blankshape")) {
               arg = acquirer.acquire2<RooAbsArg, RooRealVar>("1", "1", 1);
            } else {
               if (!h) {
                  arg = acquirer.acquireNew<RooRealVar>(TString::Format("%s_bin%d", newObjName.Data(), i + 1), "", 1);
               }
               if (h->GetMinimumStored() != -1111 || h->GetMaximumStored() != -1111) {
                  arg = acquirer.acquireNew<RooRealVar>(TString::Format("%s_bin%d", newObjName.Data(), i + 1), "",
                                                        h->GetBinContent(i + 1), h->GetMinimumStored(),
                                                        h->GetMaximumStored());
               } else {
                  arg = acquirer.acquireNew<RooRealVar>(TString::Format("%s_bin%d", newObjName.Data(), i + 1), "",
                                                        h->GetBinContent(i + 1));
               }
            }
            list.add(*arg);
         }
         // paramhistfunc requires the binnings to be loaded as default at construction time
         // so load binning temporarily
         auto tmp = dynamic_cast<RooAbsBinning *>(x->getBinningPtr(nullptr)->Clone());
         x->setBinning(x->getBinning(binningName.c_str()));
         _f = acquirer.acquireNew<ParamHistFunc>(newObjName, h->GetTitle(), *x, list);
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
         dynamic_cast<ParamHistFunc *>(_f.get())->_paramSet.setName("paramSet"); // so can see when print
#else
         const_cast<RooArgList &>(dynamic_cast<ParamHistFunc *>(_f.get())->paramList())
            .setName("paramSet"); // so can see when print
#endif
         x->setBinning(*tmp); // restore binning
         delete tmp;
         for (auto &[k, v] : stringAttrs) {
            _f->setStringAttribute(k.c_str(), v.c_str());
         }
      } else {
         auto dh = acquirer.acquireNew<RooDataHist>(Form("hist_%s", newObjName.Data()), h->GetTitle(), *x,
                                                    binningName.c_str() /* binning name*/);
         if (!dh) {
            throw std::runtime_error("Couldn't make data hist");
         }
         auto f = acquirer.acquireNew<RooHistFunc>(newObjName, h->GetTitle(), *x, *dh,
                                                   0 /*interpolation order between bins*/);
         f->forceNumInt();
         f->setAttribute("autodensity"); // where it gets inserted will determine if this is a density or not
         _f = f;

         for (auto &[k, v] : stringAttrs) {
            _f->setStringAttribute(k.c_str(), v.c_str());
         }

         // need to do these settings here because used in the assignment step
         _f->setStringAttribute("xvar", x->GetName());
         _f->setStringAttribute("binning", binningName.c_str());
         if (strcmp(_f->GetName(), origName.Data()) && !_f->getStringAttribute("alias"))
            _f->setStringAttribute("alias", origName);

         // copy values over using the assignment operator - may convert to a RooProduct if there are stat uncerts
         xRooNode tmp(h->GetName(), _f, acquirer);
         tmp = *h;
         _f = std::dynamic_pointer_cast<RooAbsArg>(tmp.fComp); // in case got upgrade to a RooProduct
      }

      _f->setStringAttribute("xvar", x->GetName());
      _f->setStringAttribute("binning", binningName.c_str());
      // style(h); // will transfer styling to object if necessary - not doing because this method used with plane hists
      // frequently
      if (strcmp(_f->GetName(), origName.Data()) && !_f->getStringAttribute("alias"))
         _f->setStringAttribute("alias", origName);

      fComp = _f;
      return _f;
   } else if (!get() && sName.BeginsWith("factory:") && acquirer.ws()) {
      TString s(sName);
      s = TString(s(8, s.Length()));
      fComp.reset(acquirer.ws()->factory(s), [](TObject *) {});
      if (fComp) {
         const_cast<xRooNode *>(this)->TNamed::SetName(fComp->GetName());
      }
      return fComp;
   }

   return fComp;
}

std::shared_ptr<TStyle> xRooNode::style(TObject *initObject, bool autoCreate) const
{
   return std::dynamic_pointer_cast<TStyle>(styles(initObject, autoCreate).fComp);
}

xRooNode xRooNode::styles(TObject *initObject, bool autoCreate) const
{
   TString t = GetTitle();

   auto arg = get<RooAbsArg>();
   if (!initObject && !arg && !gROOT->GetStyle(t)) {
      return nullptr;
   }

   std::unique_ptr<TObject> argInitObject;

   if (initObject) {
      t = (strlen(initObject->GetTitle())) ? initObject->GetTitle() : initObject->GetName();
   } else if (arg) {
      if (arg->getStringAttribute("style")) {
         t = arg->getStringAttribute("style");
      } else if (autoCreate) {
         // args will default to histo's object styling, whatever that currently may be
         argInitObject = std::make_unique<TH1D>(GetName(), GetTitle(), 1, 0, 1);
         initObject = argInitObject.get();
      } else {
         return nullptr;
      }
   }

   std::shared_ptr<TStyle> style; // use to keep alive for access from GetStyle below, in case getObject has decided to
                                  // return the owning ptr (for some reason)
   if (!gROOT->GetStyle(t)) {
      if ((style = getObject<TStyle>(t.Data()))) {
         // loaded style (from workspace?) so put in list and use that
         gROOT->GetListOfStyles()->Add(style.get());
      } else {
         if (!autoCreate)
            return nullptr;
         // create new style - gets put in style list automatically so don't have to delete
         // acquire them so saved to workspaces for auto reload ...
         style = const_cast<xRooNode &>(*this).acquireNew<TStyle>(t.Data(),
                                                                  TString::Format("Style for %s component", t.Data()));
         if (auto x = dynamic_cast<TAttLine *>(initObject))
            ((TAttLine &)*style) = *x;
         if (auto x = dynamic_cast<TAttFill *>(initObject))
            ((TAttFill &)*style) = *x;
         if (auto x = dynamic_cast<TAttMarker *>(initObject))
            ((TAttMarker &)*style) = *x;
         gROOT->GetListOfStyles()->Add(style.get());
      }
   } else {
      style = std::shared_ptr<TStyle>(gROOT->GetStyle(t), [](TStyle *) {});
   }

   if (arg && !arg->getStringAttribute("style")) {
      arg->setStringAttribute("style", style->GetName());
   }

   return xRooNode(style, *this);
}

std::shared_ptr<TObject> xRooNode::acquire(const std::shared_ptr<TObject> &arg, bool checkFactory, bool mustBeNew)
{
   if (!arg)
      return nullptr;
   if (!fAcquirer && !get<RooWorkspace>() && fParent)
      return fParent->acquire(arg, checkFactory, mustBeNew);

   // if has a workspace and our object is the workspace or is in the workspace then add this object to workspace
   auto _ws = (fAcquirer) ? nullptr : ws();
   if (_ws && (get() == _ws || _ws->arg(GetName()) || (arg && strcmp(arg->GetName(), GetName()) == 0))) {
      RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);
      if (auto a = dynamic_cast<RooAbsArg *>(arg.get()); a) {
         auto out_arg = _ws->arg(a->GetName());
         TString aName = arg->GetName();
         int ii = 1;
         while (out_arg && mustBeNew) {
            a->SetName(TString::Format("%s_%d", aName.Data(), ii++));
            out_arg = _ws->arg(a->GetName());
         }
         if (aName != a->GetName())
            Warning("acquire", "Renaming to %s", a->GetName());
         if (!out_arg) {
            bool done = false;
            if (checkFactory) {
               if (auto res = _ws->factory(arg->GetName()); res) {
                  a = res;
                  done = true;
               }
            }
            if (!done && _ws->import(*a, RooFit::RecycleConflictNodes())) {
               if (GETWS(a) != _ws) {
                  Info("acquire", "A copy of %s has been added to workspace %s", a->GetName(), _ws->GetName());
               }
               RooMsgService::instance().setGlobalKillBelow(msglevel);
               return nullptr;
            }
            // sanitizeWS(); // clears the caches that might exist up to now, as well interfere with getParameters calls
            std::set<std::string> setNames;
            for (auto &aa : GETWSSETS(_ws)) {
               if (TString(aa.first.c_str()).BeginsWith("CACHE_")) {
                  setNames.insert(aa.first);
               }
            }
            for (auto &aa : setNames)
               ws()->removeSet(aa.c_str());
            out_arg = _ws->arg(a->GetName());
            if (GETWS(out_arg) != _ws) { // seems that when objects imported their ws isn't set
               out_arg->setWorkspace(*_ws);
            }
         }
         RooMsgService::instance().setGlobalKillBelow(msglevel);
         return std::shared_ptr<TObject>(out_arg, [](TObject *) {});
      } else if (auto a2 = dynamic_cast<RooAbsData *>(arg.get()); a2) {
         if (_ws->import(*a2, RooFit::Embedded())) {
            RooMsgService::instance().setGlobalKillBelow(msglevel);
            return nullptr;
         }
         RooMsgService::instance().setGlobalKillBelow(msglevel);
         return std::shared_ptr<TObject>(_ws->embeddedData(arg->GetName()), [](TObject *) {});
      } else if (arg->InheritsFrom("RooFitResult") || arg->InheritsFrom("TTree") || arg->IsA() == TStyle::Class() ||
                 arg->InheritsFrom("RooStats::HypoTestInverterResult") ||
                 arg->InheritsFrom("RooStats::HypoTestResult")) {
         // ensure will have a unique name for import if must be new
         TNamed *aNamed = dynamic_cast<TNamed *>(arg.get());
         TString aName = arg->GetName();
         TObject *out_arg = _ws->genobj(arg->GetName());
         int ii = 1;
         while (aNamed && out_arg && mustBeNew) {
            aNamed->SetName(TString::Format("%s;%d", aName.Data(), ii++));
            out_arg = _ws->genobj(aNamed->GetName());
         }
         if (!out_arg) {
            if (aName != arg->GetName()) {
               Warning("acquire", "Renaming to %s", arg->GetName());
            }
            if (_ws->import(*arg, false /*replace existing*/)) {
               RooMsgService::instance().setGlobalKillBelow(msglevel);
               return nullptr;
            }
            out_arg = _ws->genobj(arg->GetName());
         }
         RooMsgService::instance().setGlobalKillBelow(msglevel);
         /* this doesnt work because caller has its own version of fParent, not the one in the browser
         for(auto o : *gROOT->GetListOfBrowsers()) {
             if(auto b = dynamic_cast<TBrowser*>(o); b){
                 if(auto _b = dynamic_cast<TGFileBrowser*>( dynamic_cast<TRootBrowser*>(b->GetBrowserImp())->fActBrowser
         ); _b) { if (auto item = _b->fListTree->FindItemByObj(_b->fRootDir,this); item) { auto _tmp = _b->fListLevel;
                         _b->fListLevel = item;
                         bool _tmp2 = item->IsOpen();
                         item->SetOpen(false);
                         this->Browse(b);
                         item->SetOpen(_tmp2);
                         _b->fListLevel = _tmp;
                     }
                 }
             }
         }*/
         return std::shared_ptr<TObject>(out_arg, [](TObject *) {});
      }
      RooMsgService::instance().setGlobalKillBelow(msglevel);
      // Warning("acquire","Not implemented acquisition of object %s",arg->GetName());
      // return nullptr;
   }
   if (!mustBeNew && fProvider) {
      auto out = fProvider->getObject(arg->GetName(), arg->ClassName());
      if (out)
         return out;
   }
   auto _owned = find(".memory");
   if (!_owned) {
      _owned = emplace_back(std::make_shared<xRooNode>(".memory", nullptr, *this));
   }
   // look for exact name, dont use 'find' because doesnt work if trying to find "1" and it doesn't exist, will get back
   // idx 1 instead
   if (!mustBeNew) {
      for (auto &r : *_owned) {
         if (strcmp(r->GetName(), arg->GetName()) == 0 && strcmp(r->get()->ClassName(), arg->ClassName()) == 0) {
            return r->fComp;
         }
      }
   }
   if (!fProvider)
      std::cout << GetName() << " taking over " << arg->ClassName() << "::" << arg->GetName() << std::endl;
   /*emplace_back(std::make_shared<Node2>(".memory",nullptr,*this))*/
   return _owned->emplace_back(std::make_shared<xRooNode>(arg->GetName(), arg, *this))->fComp;
   // return arg;
}

bool xRooNode::SetXaxis(const char *name, const char *title, int nbins, double low, double high)
{
   RooUniformBinning b(low, high, nbins, name);
   b.SetTitle(title);
   return SetXaxis(b);
}

bool xRooNode::SetXaxis(const char *name, const char *title, int nbins, const double *bins)
{
   RooBinning b(nbins, bins, name);
   b.SetTitle(title);
   return SetXaxis(b);
}

bool xRooNode::SetXaxis(const RooAbsBinning &binning)
{

   auto name = binning.GetName();
   double high = binning.highBound();
   double low = binning.lowBound();
   // int nbins = binning.numBins();
   auto title = binning.GetTitle();

   // if have any dependents and name isn't one of them then stop
   auto _deps = vars();
   /*if(!_deps.empty() && !_deps.find(name)) {
       throw std::runtime_error(TString::Format("%s Does not depend on %s",GetName(),name));
   }*/

   // object will need to exist
   if (!get()) {
      if (fParent && !find(GetName())) {
         fComp = fParent->Add(*this, "+").fComp;
      }
   }

   auto a = get<RooAbsArg>();
   if (!a)
      throw std::runtime_error("Cannot SetXaxis of non-arg");

   auto _x = acquire<RooRealVar>(name, title, low, high);
   _x->setBinning(binning, a->GetName());
   _x->getBinning(a->GetName()).SetTitle(title);
   if (_x->getBinningNames().size() == 2) {
      // this was the first binning, so copy it over to be the default binning too
      _x->setBinning(_x->getBinning(a->GetName()));
   } else {
      // ensure the default binning is wide enough to cover this range
      // the alternative to this would be to ensure setNormRange of all pdfs
      // are set to correct range (then default can be narrower than some of the named binnings)
      if (_x->getMax() < high)
         _x->setMax(high);
      if (_x->getMin() > low)
         _x->setMin(low);
   }

   if (!_deps.find(name) && get<RooAbsPdf>()) {
      // creating a variable for a pdf we will assume it should be an observable
      _x->setAttribute("obs");
   }

   a->setStringAttribute("xvar", _x->GetName());
   a->setStringAttribute("binning", a->GetName());
   fXAxis.reset(); // remove any existing xaxis

   return true;
}

bool xRooNode::SetXaxis(TAxis *ax)
{
   if (!ax)
      return false;
   if (ax->IsVariableBinSize()) {
      return SetXaxis(ax->GetName(), ax->GetTitle(), ax->GetNbins(), ax->GetXbins()->GetArray());
   } else {
      return SetXaxis(ax->GetName(), ax->GetTitle(), ax->GetNbins(), ax->GetXmin(), ax->GetXmax());
   }
}

bool xRooNode::contains(const std::string &name) const
{
   return find(name, false) != nullptr;
}

std::shared_ptr<xRooNode> xRooNode::find(const std::string &name, bool browseResult) const
{
   std::string partname = (name.find('/') != std::string::npos) ? name.substr(0, name.find('/')) : name;
   auto _s = (!get() && fParent) ? fParent->get<RooSimultaneous>()
                                 : get<RooSimultaneous>(); // makes work if doing simPdf.bins()["blah"]
   std::string extra = (_s) ? _s->indexCat().GetName() : "";
   for (auto &child : *this) {
      if (auto _obj = child->get(); name == child->GetName() || partname == child->GetName() ||
                                    (_obj && name == _obj->GetName()) || (_obj && partname == _obj->GetName()) ||
                                    (!extra.empty() && ((extra + "=" + name) == child->GetName() ||
                                                        (extra + "=" + partname) == child->GetName()))) {
         if (browseResult)
            child->browse(); // needed so can go at()->at()->at()...
         if (partname != name && name != child->GetName()) {
            return child->at(name.substr(partname.length() + 1));
         }
         return child;
      }
      if (partname.find('.') != 0) { // do not allow mainChild browsing if trying to find a "." child ... as is done in
                                     // getObject for ".memory"
         if (auto x = mainChild(); x && strcmp(child->GetName(), x.GetName()) == 0) {
            // can browse directly into main children as if their children were our children
            for (auto &child2 : x.browse()) {
               if (auto _obj = child2->get(); name == child2->GetName() || partname == child2->GetName() ||
                                              (_obj && name == _obj->GetName()) ||
                                              (_obj && partname == _obj->GetName())) {
                  if (browseResult)
                     child2->browse(); // needed for onward read (or is it? there's a browse above too??)
                  if (partname != name && name != child2->GetName()) {
                     return child2->at(name.substr(partname.length() + 1));
                  }
                  return child2;
               }
            }
         }
      }
   }
   // before giving up see if partName is numeric and indexes within the range
   if (TString s(partname); s.IsDec() && size_t(s.Atoi()) < size()) {
      auto child2 = at(s.Atoi());
      if (partname != name) {
         return child2->at(name.substr(partname.length() + 1));
      }
      return child2;
   }
   // allow calling of find on a RooWorkspace to access getObject objects ...
   if (get<RooWorkspace>() && name != ".memory") {
      if (auto obj = getObject(name)) {
         auto out = std::make_shared<xRooNode>(obj, *this);
         if (browseResult)
            out->browse();
         return out;
      }
   }
   return nullptr;
}

std::shared_ptr<xRooNode> xRooNode::operator[](const std::string &name)
{
   std::string partname = (name.find('/') != std::string::npos) ? name.substr(0, name.find('/')) : name;
   browse();
   auto _s = (!get() && fParent) ? fParent->get<RooSimultaneous>()
                                 : get<RooSimultaneous>(); // makes work if doing simPdf.bins()["blah"]
   std::string extra = (_s) ? _s->indexCat().GetName() : "";
   std::shared_ptr<xRooNode> folderNode;
   for (auto &child : *this) {
      if (name == child->GetName() || partname == child->GetName() ||
          (!extra.empty() &&
           ((extra + "=" + name) == child->GetName() || (extra + "=" + partname) == child->GetName()))) {
         child->browse(); // needed for onward read (or is it? there's a browse above too??)
         if (partname != name && name != child->GetName()) {
            return child->operator[](name.substr(partname.length() + 1));
         }
         return child;
      }
      if (auto x = mainChild(); strcmp(child->GetName(), x.GetName()) == 0) {
         // can browse directly into main children as if their children were our children
         for (auto &child2 : x.browse()) {
            if (name == child2->GetName() || partname == child2->GetName()) {
               child2->browse(); // needed for onward read (or is it? there's a browse above too??)
               if (partname != name && name != child2->GetName()) {
                  return child2->operator[](name.substr(partname.length() + 1));
               }
               return child2;
            }
         }
      }
      if (child->fFolder == (std::string("!") + partname)) {
         if (!folderNode)
            folderNode = std::make_shared<xRooNode>(child->fFolder.c_str(), nullptr, *this);
         folderNode->push_back(child);
      }
   }
   if (folderNode) {
      if (partname != name) {
         return folderNode->operator[](name.substr(partname.length() + 1));
      }
      return folderNode;
   }
   // before giving up see if partName is numeric and indexes within the range
   if (TString s(partname); s.IsDec() && size_t(s.Atoi()) < size()) {
      auto child2 = at(s.Atoi());
      if (partname != name) {
         return child2->operator[](name.substr(partname.length() + 1));
      }
      return child2;
   }
   auto out = std::make_shared<xRooNode>(partname.c_str(), nullptr, *this); // not adding as child yeeet
   // special case, if creating a node in the workspace with a specific name, it's a folder node ...
   if (get<RooWorkspace>() && partname == "pdfs") {
      out->SetName("!pdfs");
   }
   if (partname != name) {
      return out->operator[](name.substr(partname.length() + 1));
   }
   return out;
}

TGListTreeItem *xRooNode::GetTreeItem(TBrowser *b) const
{
   if (!b) {
      for (auto o : *gROOT->GetListOfBrowsers()) {
         b = dynamic_cast<TBrowser *>(o);
         if (!b || !b->GetBrowserImp())
            continue;
         if (auto out = GetTreeItem(b); out)
            return out;
      }
      return nullptr;
   }
   if (!b->GetBrowserImp())
      return nullptr;
   if (auto _b = dynamic_cast<TGFileBrowser *>(GETACTBROWSER(dynamic_cast<TRootBrowser *>(b->GetBrowserImp()))); _b) {
      auto _root = GETROOTDIR(_b);
      ;
      if (!_root)
         _root = GETLISTTREE(_b)->GetFirstItem();
      GETLISTTREE(_b)->SetColorMode(TGListTree::EColorMarkupMode(TGListTree::kColorUnderline | TGListTree::kColorBox));
      return GETLISTTREE(_b)->FindItemByObj(_root, const_cast<xRooNode *>(this));
   }
   return nullptr;
}

TGListTree *xRooNode::GetListTree(TBrowser *b) const
{
   if (!b) {
      for (auto o : *gROOT->GetListOfBrowsers()) {
         b = dynamic_cast<TBrowser *>(o);
         if (!b || !b->GetBrowserImp())
            continue;
         if (auto out = GetListTree(b); out)
            return out;
      }
      return nullptr;
   }
   if (b->GetBrowserImp()) {
      if (auto _b = dynamic_cast<TGFileBrowser *>(GETACTBROWSER(dynamic_cast<TRootBrowser *>(b->GetBrowserImp())));
          _b) {
         auto _root = GETROOTDIR(_b);
         if (!_root)
            _root = GETLISTTREE(_b)->GetFirstItem();
         if (auto item = GETLISTTREE(_b)->FindItemByObj(_root, const_cast<xRooNode *>(this)); item) {
            return GETLISTTREE(_b);
         }
      }
   }
   return nullptr;
}

void xRooNode::SetName(const char *name)
{
   TNamed::SetName(name);
   if (auto a = get<RooAbsArg>(); a)
      a->setStringAttribute("alias", name);
   for (auto o : *gROOT->GetListOfBrowsers()) {
      if (auto b = dynamic_cast<TBrowser *>(o); b) {
         if (auto item = GetTreeItem(b); item) {
            item->SetText(name);
         }
      }
   }
}

void xRooNode::SetTitle(const char *title)
{
   if (auto o = (get<TNamed>()); o) {
      if (auto c = mainChild(); c.get()) {
         c.SetTitle(title);
      }
      o->SetTitle(title);
   }
   TNamed::SetTitle(title);
}

xRooNode &xRooNode::browse()
{
   if (get<RooArgList>() || (!get() && !(strlen(GetName()) > 0 && (GetName()[0] == '!')) && !fBrowseOperation))
      return *this; // nothing to browse - 'collection' nodes should already be populated except for folders
   // alternative could have been to mandate that the 'components' of a collection node are the children it has.

   auto findByObj = [&](const std::shared_ptr<xRooNode> &n) {
      std::vector<std::shared_ptr<xRooNode>> &nn = *this;
      for (auto &c : nn) {
         if (c->get() == n->get() && strcmp(n->GetName(), c->GetName()) == 0)
            return c;
      }
      return std::shared_ptr<xRooNode>(nullptr);
   };

   auto appendChildren = [&](const xRooNode &n) {
      size_t out = 0;
      const std::vector<std::shared_ptr<xRooNode>> &nn(n);
      for (auto &c : nn) {
         if (auto existing = findByObj(c); existing) {
            existing->fTimes++;
            existing->fFolder = c->fFolder; // transfer folder assignment
         } else {
            emplace_back(c);
         }
         if (!TString(c->GetName()).BeginsWith(".coef"))
            out++; // don't count .coef as a child, as technically part of parent
      }
      return out;
   };

   const std::vector<std::shared_ptr<xRooNode>> &nn2(*this);
   for (auto &c : nn2) {
      if (strlen(c->GetName()) > 0 && (c->GetName()[0] == '.')) {
         c->fTimes = 1;
         continue;
      } // never auto-cleanup property children
      if (strcmp(c->GetName(), "!.pars") == 0) {
         c->fTimes = 1;
         continue;
      } // special collection, also not cleaned up
      if (c->get<RooWorkspace>() || c->get<TFile>()) {
         c->fTimes = 1;
         continue;
      } // workspaces and files not cleaned up: TODO have a nocleanup flag instead
      c->fTimes = 0;
   }

   size_t addedChildren = 0;
   if (fBrowseOperation) {
      addedChildren += appendChildren(fBrowseOperation(this));
   } else {
      if (get<RooWorkspace>()) {
         addedChildren += appendChildren(datasets());
      }

      //    if (get<RooAbsPdf>() && ((fParent && fParent->get<RooWorkspace>()) || !fParent)) {
      //        // top-level pdfs will also list the ".vars" property for -- should make this updateable
      //        //if (auto x = find("!.vars"); !x) { // this is slower because it triggers a browse of !.vars
      //        if(!contains("!.vars")) {
      //            emplace_back(std::make_shared<Node2>("!.vars",nullptr,*this));
      //        } /*else {
      //            x->fTimes++;
      //        }*/
      //    }

      // go through components factors and variations, adding all as children if required
      addedChildren += appendChildren(components());
      if (!get<RooWorkspace>())
         addedChildren += appendChildren(factors());
      // include coefs if any
      auto _coefs = coefs();
      if (_coefs.get() && strcmp(_coefs->GetName(), "1") != 0 && strcmp(_coefs->GetName(), "ONE") != 0) {
         if (_coefs.size() == 1 && _coefs.get<RooAddition>()) {
            if (strcmp(_coefs.at(0)->GetName(), "1") != 0 &&
                strcmp(_coefs.at(0)->GetName(), "ONE") != 0) { // don't add the "1"
               auto coef = std::make_shared<xRooNode>(".coef", *_coefs.at(0)->get(), *this);
               if (auto existing = findByObj(coef); existing) {
                  existing->fTimes++;
                  existing->fFolder = _coefs.at(0)->fFolder; // transfer folder assignment
               } else {
                  emplace_back(coef);
               }
            }
         } else {
            if (auto existing = find(_coefs.GetName()); existing) {
               existing->fTimes++;
               existing->fFolder = _coefs.fFolder; // transfer folder assignment
            } else {
               emplace_back(std::make_shared<xRooNode>(_coefs));
            }
         }
      }
      addedChildren += appendChildren(variations());
      if (get<ParamHistFunc>() || get<RooSimultaneous>())
         addedChildren += appendChildren(bins());
      if (get<RooAbsData>())
         addedChildren += appendChildren(obs());
   }
   // if has no children and is a RooAbsArg, add all the proxies
   if (auto arg = get<RooAbsArg>(); arg && addedChildren == 0) {
      for (int i = 0; i < arg->numProxies(); i++) {
         auto _proxy = arg->getProxy(i);
         if (auto a = dynamic_cast<RooArgProxy *>(_proxy)) {
            auto c = std::make_shared<xRooNode>(TString::Format(".%s", _proxy->name()), *(a->absArg()), *this);
            if (auto existing = findByObj(c); existing) {
               existing->fTimes++;
               existing->fFolder = c->fFolder; // transfer folder assignment
            } else {
               // mark any existing children with the same name for cleanup - this happens e.g. if did a Replace on one
               // of these nodes note that the child nodes will still become reordered (the old node will be deleted,
               // new node will appear at end)
               for (auto &child : *this) {
                  if (strcmp(child->GetName(), c->GetName()) == 0) {
                     child->fTimes = 0;
                  }
               }
               emplace_back(c);
            }
         } else if (auto s = dynamic_cast<RooAbsCollection *>(_proxy)) {
            for (auto a2 : *s) {
               auto c = std::make_shared<xRooNode>(*a2, *this);
               if (arg->numProxies() != 1) {
                  c->fFolder = std::string("!.") +
                               _proxy->name(); // don't put in a folder if there's just 1 proxy (the collection)
               }
               if (auto existing = findByObj(c); existing) {
                  existing->fTimes++;
                  existing->fFolder = c->fFolder; // transfer folder assignment
               } else {
                  emplace_back(c);
               }
            }
         }
      }
      /*for(auto& s : arg->servers()) {
          auto c = std::make_shared<xRooNode>(*s,*this);
          if (auto existing = findByObj(c); existing) {
              existing->fTimes++;
              existing->fFolder = c->fFolder; // transfer folder assignment
          } else {
              emplace_back(c);
          }
      }*/
   } else if (auto ir = get<RooStats::HypoTestInverterResult>()) {
      // check if we already have a hypoSpace in our memory
      bool hasHS = false;
      for (auto &c : fBrowsables) {
         if (strcmp(c->GetName(), ".memory") == 0 && c->get<xRooHypoSpace>()) {
            hasHS = true;
            break;
         }
      }
      if (!hasHS) {
         // add the HS
         auto hs =
            fBrowsables.emplace_back(std::make_shared<xRooNode>(".memory", std::make_shared<xRooHypoSpace>(ir), *this));
         // add the hypoPoints first so they appear first
         auto _axes = hs->get<xRooHypoSpace>()->axes();

         int i = 0;
         for (auto &hp : *hs->get<xRooHypoSpace>()) {
            TString coordString;
            for (auto a : _axes) {
               if (a != _axes.first())
                  coordString += ",";
               coordString +=
                  TString::Format("%s=%g", a->GetName(), hp.coords->getRealValue(a->GetName(), ir->GetXValue(i)));
            }
            auto hpn = emplace_back(std::make_shared<xRooNode>(coordString, hp.hypoTestResult, hs));
            hpn->fTimes++;
            hpn->fBrowsables.emplace_back(std::make_shared<xRooNode>(
               ".memory", std::shared_ptr<xRooNLLVar::xRooHypoPoint>(&hp, [](xRooNLLVar::xRooHypoPoint *) {}), hpn));
            i++;
         }
      } else {
         // ensure all hypoTestResults are flagged as keep-alive
         std::vector<std::shared_ptr<xRooNode>> &nn = *this;
         for (auto &c : nn) {
            if (c->get<RooStats::HypoTestResult>())
               c->fTimes++;
         }
      }
      //      xRooNode tests;
      //      for(int i=0;i<ir->ArraySize();i++) {
      //         tests.push_back(std::make_shared<xRooNode>(TString::Format("%g",ir->GetXValue(i)),*ir->GetResult(i),*this));
      //      }
      //      appendChildren(tests);
   } else if (get<RooStats::HypoTestResult>()) {

      // create the xRooHypoPoint if necessary
      xRooNLLVar::xRooHypoPoint *hp = nullptr;
      for (auto &c : fBrowsables) {
         if (strcmp(c->GetName(), ".memory") == 0 && c->get<xRooNLLVar::xRooHypoPoint>()) {
            hp = c->get<xRooNLLVar::xRooHypoPoint>();
            c->fTimes++; // keep it alive
            break;
         }
      }
      if (!hp) {
         auto shp =
            std::make_shared<xRooNLLVar::xRooHypoPoint>(std::dynamic_pointer_cast<RooStats::HypoTestResult>(fComp));
         fBrowsables.emplace_back(std::make_shared<xRooNode>(".memory", shp, *this));
         hp = shp.get();
      }

      xRooNode fits;

      if (auto fit = hp->ufit()) {
         fits.emplace_back(std::make_shared<xRooNode>(fit, *this))->TNamed::SetName("ufit");
      }
      if (auto fit = hp->cfit_null()) {
         fits.emplace_back(std::make_shared<xRooNode>(fit, *this))->TNamed::SetName("cfit_null");
      }
      if (auto fit = hp->cfit_alt()) {
         fits.emplace_back(std::make_shared<xRooNode>(fit, *this))->TNamed::SetName("cfit_alt");
      }
      if (auto fit = hp->gfit()) {
         fits.emplace_back(std::make_shared<xRooNode>(fit, *this))->TNamed::SetName("gfit");
      }
      if (auto asi = hp->asimov()) {
         auto asiP = fits.emplace_back(std::make_shared<xRooNode>(
            asi->hypoTestResult ? asi->hypoTestResult : std::make_shared<RooStats::HypoTestResult>(asi->result()),
            *this));
         asiP->TNamed::SetName("asimov");
         asiP->fBrowsables.emplace_back(std::make_shared<xRooNode>(".memory", asi, asiP));
      }
      appendChildren(fits);
   }

   // clear anything that has fTimes = 0 still
   auto it = std::vector<std::shared_ptr<xRooNode>>::begin();
   while (it != std::vector<std::shared_ptr<xRooNode>>::end()) {
      if (it->get()->fTimes == 0) {
         for (auto o : *gROOT->GetListOfBrowsers()) {
            auto b = dynamic_cast<TBrowser *>(o);
            if (b && b->GetBrowserImp()) { // browserImp is null if browser was closed
               // std::cout << GetPath() << " Removing " << it->get()->GetPath() << std::endl;

               if (auto _b =
                      dynamic_cast<TGFileBrowser *>(GETACTBROWSER(dynamic_cast<TRootBrowser *>(b->GetBrowserImp())));
                   _b) {
                  auto _root = GETROOTDIR(_b);
                  if (!_root)
                     _root = GETLISTTREE(_b)->GetFirstItem();
                  if (auto item = GETLISTTREE(_b)->FindItemByObj(_root, this); item) {
                     GETLISTTREE(_b)->OpenItem(item);
                  }
               }

               b->RecursiveRemove(
                  it->get()); // problem: if obj is living in a collapsed node it wont actually get deleted
               /*auto _b = dynamic_cast<TGFileBrowser*>( dynamic_cast<TRootBrowser*>(b->GetBrowserImp())->fActBrowser );
               if (_b) {
                   std::cout << _b->fRootDir->GetText() << std::endl;
                   if (auto item = _b->fListTree->FindItemByObj(_b->fRootDir,it->get()); item) {
                       std::cout << "Found obj: " << item << " " << item->GetText() << std::endl;
                       _b->fListTree->RecursiveDeleteItem(_b->fRootDir,it->get());
                   }

                   //b->RecursiveRemove(it->get());
                   if (auto item = _b->fListTree->FindItemByObj(_b->fRootDir,it->get()); item) {
                       std::cout << "Still Found obj: " << item  << std::endl;
                   }
                   _b->fListTree->ClearViewPort();

               }*/
            }
         }
         /*it->get()->ResetBit(TObject::kNotDeleted); ++it;*/ it = erase(it);
      } else {
         ++it;
      }
   }

   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// List of observables (global and regular) of this node.

xRooNode xRooNode::obs() const
{
   xRooNode out(".obs", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".obs").c_str());
   for (auto o : vars()) {
      if (o->get<RooAbsArg>()->getAttribute("obs")) {
         out.get<RooArgList>()->add(*o->get<RooAbsArg>());
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of global observables of this node.

xRooNode xRooNode::globs() const
{
   xRooNode out(".globs", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".globs").c_str());
   for (auto o : obs()) {
      if (o->get<RooAbsArg>()->getAttribute("global")) {
         out.get<RooArgList>()->add(*o->get<RooAbsArg>());
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of regular observables of this node.

xRooNode xRooNode::robs() const
{
   xRooNode out(".robs", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".robs").c_str());
   for (auto o : obs()) {
      if (!o->get<RooAbsArg>()->getAttribute("global")) {
         out.get<RooArgList>()->add(*o->get<RooAbsArg>());
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of parameters (non-observables) of this node.

xRooNode xRooNode::pars() const
{
   if (strcmp(GetName(), ".bins") == 0 && fParent) {
      // return pars of the parent - this method is used by covariances() if e.g. do node.bins().covariances()
      return fParent->pars();
   }
   xRooNode out(".pars", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".pars").c_str());
   for (auto o : vars()) {
      if (!o->get<RooAbsArg>()->getAttribute("obs")) {
         out.get<RooArgList>()->add(*(o->get<RooAbsArg>()));
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of parameters that are currently constant

xRooNode xRooNode::consts() const
{
   xRooNode out(".consts", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".consts").c_str());
   for (auto o : pars()) {
      if (o->get<RooAbsArg>()->getAttribute("Constant") || o->get<RooConstVar>()) {
         out.get<RooArgList>()->add(*o->get<RooAbsArg>());
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of parameters that are currently non-constant
/// These parameters do not have the "Constant" attribute

xRooNode xRooNode::floats() const
{
   xRooNode out(".floats", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".floats").c_str());
   for (auto o : pars()) {
      if (!o->get<RooAbsArg>()->getAttribute("Constant") && !o->get<RooConstVar>()) {
         out.get<RooArgList>()->add(*o->get<RooAbsArg>());
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of parameters of interest: parameters marked as "of interest"
/// These parameters have the "poi" attribute

xRooNode xRooNode::poi() const
{
   xRooNode out(".poi", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".poi").c_str());
   for (auto o : pars()) {
      if (o->get<RooAbsArg>()->getAttribute("poi")) {
         out.get<RooArgList>()->add(*o->get<RooAbsArg>());
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of nuisance parameters: non-constant parameters that are not marked of interest,
///  as well as any parameters that have been marked by the "np" attribute

xRooNode xRooNode::np() const
{
   xRooNode out(".np", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".np").c_str());
   for (auto o : pars()) {
      if (o->get<RooAbsArg>()->getAttribute("np") ||
          (!o->get<RooAbsArg>()->getAttribute("Constant") && !o->get<RooAbsArg>()->getAttribute("poi") &&
           !o->get<RooConstVar>())) {
         out.get<RooArgList>()->add(*o->get<RooAbsArg>());
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of prespecified parameters: non-floatable parameters

xRooNode xRooNode::pp() const
{
   xRooNode out(".pp", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".pp").c_str());
   for (auto o : pars()) {
      if (!o->get<RooAbsArg>()->getAttribute("np") && !o->get<RooAbsArg>()->getAttribute("poi") &&
          (o->get<RooAbsArg>()->getAttribute("Constant") || o->get<RooConstVar>())) {
         out.get<RooArgList>()->add(*o->get<RooAbsArg>());
         out.emplace_back(o);
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// List of variables (observables and parameters) of this node

xRooNode xRooNode::vars() const
{
   xRooNode out(".vars", std::make_shared<RooArgList>(), *this);
   out.get<RooArgList>()->setName((GetPath() + ".vars").c_str());
   if (auto coll = get<RooAbsCollection>(); coll) {
      for (auto &x : *this) {
         for (auto &y : x->vars()) {
            out.push_back(y);
         }
      }
      return out;
   }
   if (auto p = get<RooAbsArg>(); p) {
      // also need to get all constPars so use leafNodeServerList .. will include self if is fundamental, which is what
      // we want
      // ensure all globs appear after robs, as we rely on this ordering for picking "x" var in "reduced" method
      xRooNode _globs;
      RooArgSet allLeaves;
      p->leafNodeServerList(&allLeaves);
      for (auto &c : allLeaves) {
         if (c->isFundamental() || (dynamic_cast<RooConstVar *>(c) && !TString(c->GetName()).IsFloat())) {
            if (!c->getAttribute("global")) {
               out.get<RooArgList>()->add(*c);
               out.emplace_back(std::make_shared<xRooNode>(*c, *this));
            }
            if (c->getAttribute("global")) {
               _globs.emplace_back(std::make_shared<xRooNode>(*c, *this));
               _globs.back()->fFolder = "!globs";
            } else if (c->getAttribute("obs")) {
               out.back()->fFolder = "!robs";
            } else if (c->getAttribute("poi")) {
               out.back()->fFolder = "!poi";
            } else if (c->getAttribute("np") ||
                       (!c->getAttribute("Constant") && !c->getAttribute("poi") && c->IsA() != RooConstVar::Class())) {
               out.back()->fFolder = "!np";
            } else if (!c->getAttribute("Constant") && c->IsA() != RooConstVar::Class()) {
               out.back()->fFolder = "!floats";
            } else {
               out.back()->fFolder = "!pp";
            }
         }
      }
      for (auto g : _globs) {
         out.get<RooArgList>()->add(*g->get<RooAbsArg>());
         out.emplace_back(g);
      }
   } else if (auto p2 = get<RooAbsData>(); p2) {
      for (auto a : *p2->get()) {
         a->setAttribute("obs");
         out.emplace_back(std::make_shared<xRooNode>(*a, *this));
         out.get<RooArgList>()->add(*a);
      }
      if (auto _dglobs = p2->getGlobalObservables()) {
         for (auto &a : *_dglobs) {
            a->setAttribute("obs");
            a->setAttribute("global");
            out.emplace_back(std::make_shared<xRooNode>(*a, *this));
            out.get<RooArgList>()->add(*a);
         }
      } else if (auto _globs = find(".globs"); _globs && _globs->get<RooAbsCollection>()) {
         for (auto &a : *_globs->get<RooAbsCollection>()) {
            a->setAttribute("obs");
            a->setAttribute("global");
            out.emplace_back(std::make_shared<xRooNode>(*a, *this));
            out.get<RooArgList>()->add(*a);
         }
      } else if (auto _ws = ws(); _ws) {
         if (auto _globs2 = dynamic_cast<RooArgSet *>(GETWSSNAPSHOTS(_ws).find(p2->GetName())); _globs2) {
            for (auto a : *_globs2) {
               a->setAttribute("obs");
               a->setAttribute("global");
               out.emplace_back(std::make_shared<xRooNode>(*a, *this));
               out.get<RooArgList>()->add(*a);
            }
         } else if (auto _gl = GETWSSETS(_ws).find("globalObservables"); _gl != GETWSSETS(_ws).end()) {
            for (auto &_g : _gl->second) {
               auto _clone = std::shared_ptr<RooAbsArg>(dynamic_cast<RooAbsArg *>(_g->Clone(_g->GetName())));
               if (auto v = std::dynamic_pointer_cast<RooAbsRealLValue>(_clone); v && _g->getStringAttribute("nominal"))
                  v->setVal(TString(_g->getStringAttribute("nominal")).Atof());
               out.emplace_back(std::make_shared<xRooNode>(_clone, *this));
               out.get<RooArgList>()->add(*_clone);
            }
         } else if (fParent) {
            // note: this is slow in large workspaces ... too many obs to look through?
            std::unique_ptr<RooAbsCollection> _globs3(fParent->obs().get<RooArgList>()->selectByAttrib("global", true));
            // std::unique_ptr<RooAbsCollection> _globs(_ws->allVars().selectByAttrib("global",true)); - tried this to
            // be quicker but it wasn't
            for (auto &_g : *_globs3) {
               auto _clone = std::shared_ptr<RooAbsArg>(dynamic_cast<RooAbsArg *>(_g->Clone(_g->GetName())));
               if (auto v = std::dynamic_pointer_cast<RooAbsRealLValue>(_clone); v && _g->getStringAttribute("nominal"))
                  v->setVal(TString(_g->getStringAttribute("nominal")).Atof());
               out.emplace_back(std::make_shared<xRooNode>(_clone, *this));
               out.get<RooArgList>()->add(*_clone);
            }
         }
      }
   } else if (auto w = get<RooWorkspace>(); w) {
      for (auto a : w->allVars()) {
         out.emplace_back(std::make_shared<xRooNode>(*a, *this));
         out.get<RooArgList>()->add(*a);
      }
      // add all cats as well
      for (auto a : w->allCats()) {
         out.emplace_back(std::make_shared<xRooNode>(*a, *this));
         out.get<RooArgList>()->add(*a);
      }
   }
   return out;
}

xRooNode xRooNode::components() const
{
   xRooNode out(".components", nullptr, *this);

   if (auto p = get<RooAddPdf>(); p) {
      // only add each pdf once (the coefs will be accumulated in coefs() method) ...
      std::set<RooAbsArg *> donePdfs;
      for (auto &o : p->pdfList()) {
         if (donePdfs.count(o))
            continue;
         out.emplace_back(std::make_shared<xRooNode>(*o, *this));
         donePdfs.insert(o);
      }
   } else if (auto p2 = get<RooRealSumPdf>(); p2) {
      // check for common prefixes and suffixes, will use to define aliases to shorten names
      // if have more than 1 function
      //        TString commonPrefix=""; TString commonSuffix="";
      //        if (p->funcList().size() > 1) {
      //            bool checked=false;
      //            for(auto& o : p->funcList()) {
      //                if (!checked) {
      //                    commonPrefix = o->GetName(); commonSuffix = o->GetName(); checked=true;
      //                } else {
      //
      //                }
      //            }
      //        }
      std::set<RooAbsArg *> doneFuncs;
      for (auto &o : p2->funcList()) {
         if (doneFuncs.count(o))
            continue;
         out.emplace_back(std::make_shared<xRooNode>(*o, *this));
         doneFuncs.insert(o);
      }
   } else if (auto p3 = get<RooAddition>(); p3) {
      for (auto &o : p3->list()) {
         out.emplace_back(std::make_shared<xRooNode>(*o, *this));
      }
   } else if (auto p4 = get<RooAbsCollection>(); p4) {
      for (auto &a : *p4) {
         out.emplace_back(std::make_shared<xRooNode>(*a, *this));
      }
   } else if (auto p5 = get<RooWorkspace>(); p5) {
      for (auto &o : p5->components()) {
         // only top-level nodes (only clients are integrals or things that aren't part of the workspace)
         // if (o->hasClients()) continue;
         bool hasClients = false;
         for (auto &c : o->clients()) {
            if (!c->InheritsFrom("RooRealIntegral") && p5 == GETWS(c)) {
               hasClients = true;
               break;
            }
         }
         if (hasClients)
            continue;
         out.emplace_back(std::make_shared<xRooNode>(*o, *this));
         if (o->InheritsFrom("RooAbsPdf")) {
            out.back()->fFolder = "!pdfs";
         } else {
            out.back()->fFolder = "!scratch";
         }
      }
      for (auto &o : p5->allGenericObjects()) {
         if (auto fr = dynamic_cast<RooFitResult *>(o); fr) {
            TString s(fr->GetTitle());
            if (s.Contains(';'))
               s = s(0, s.Index(';'));
            if (auto _pdf = out.find(s.Data()); _pdf) {
               // std::cout << " type = " << _pdf->get()->ClassName() << std::endl;
               out.emplace_back(std::make_shared<xRooNode>(fr->GetName(), *fr, _pdf));
               // for a while, this node's parent pointed to something of type Node2!!
               // how to fix??? - I fxied it with a new constructor to avoid the shared_ptr<Node2> calling the const
               // Node2& constructor via getting wrapped in a Node2(shared_ptr<TObject>) call
               // out.back()->fParent = _pdf;
               // std::cout << " type2 = " << out.back()->fParent->get()->ClassName() << std::endl;
            } else {
               out.emplace_back(std::make_shared<xRooNode>(fr->GetName(), *fr, *this));
            }
            out.back()->fFolder = "!fits";
         } else {
            out.emplace_back(std::make_shared<xRooNode>(*o, *this));
            if (strcmp(out.back()->get()->ClassName(), "TStyle") == 0) {
               out.back()->fFolder = "!styles";
            } else if (strcmp(out.back()->get()->ClassName(), "RooStats::HypoTestInverterResult") == 0) {
               out.back()->fFolder = "!scans";
            } else if (strcmp(out.back()->get()->ClassName(), "RooStats::ModelConfig") == 0) {
               out.back()->fFolder = "!models";
            } else {
               out.back()->fFolder = "!objects";
            }
         }
      }
      for (auto &[k, v] : GETWSSETS(p5)) {
         // skip 'CACHE' sets because they are auto-removed when sanitizing workspaces, which will invalidate these
         // children
         if (k.find("CACHE_") == 0)
            continue;
         out.emplace_back(std::make_shared<xRooNode>(k.c_str(), v, *this));
         out.back()->fFolder = "!sets";
      }

      RooLinkedList snaps = GETWSSNAPSHOTS(p5);
      std::unique_ptr<TIterator> iter(snaps.MakeIterator());
      TObject *snap;
      while ((snap = iter->Next())) {
         out.emplace_back(std::make_shared<xRooNode>(*snap, *this));
         out.back()->fFolder = "!snapshots";
      }
   } else if (auto mc = get<RooStats::ModelConfig>()) {
      // add the pdf as a child, and the external constraints set if its there
      if (mc->GetPdf()) {
         out.emplace_back(std::make_shared<xRooNode>(".pdf", *mc->GetPdf(), *this));
      }
      if (mc->GetExternalConstraints()) {
         out.emplace_back(std::make_shared<xRooNode>(".extCons", *mc->GetExternalConstraints(), *this));
      }
   } else if (strlen(GetName()) > 0 && GetName()[0] == '!' && fParent) {
      // special case of dynamic property
      if (TString(GetName()) == "!.pars") {
         for (auto &c : fParent->pars()) {
            out.emplace_back(c);
         }
      } else {
         // the components of a folder are the children of the parent (after browsing) that live in this folder
         fParent->browse();
         for (auto &c : *fParent) {
            if (c->fFolder == GetName()) {
               out.emplace_back(c);
            }
         }
      }
   }

   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// bins of a channel or sample, or channels of a multi-channel pdf

xRooNode xRooNode::bins() const
{
   xRooNode out(".bins", nullptr, *this);

   if (auto p = get<RooSimultaneous>(); p) {
      std::map<int, std::shared_ptr<xRooNode>> cats; // fill into a map to preserve index ordering
      for (auto &c : p->indexCat()) {                // is alphabetical in labels
         auto pp = p->getPdf(c.first.c_str());
         if (!pp)
            continue;
         cats[c.second] =
            std::make_shared<xRooNode>(TString::Format("%s=%s", p->indexCat().GetName(), c.first.c_str()), *pp, *this);
      }
      for (auto &[_, n] : cats)
         out.emplace_back(n);
   } else if (auto phf = get<ParamHistFunc>(); phf) {
      int i = 1;
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      auto &pSet = phf->_paramSet;
#else
      auto &pSet = phf->paramList();
#endif
      for (auto par : pSet) {
         out.emplace_back(std::make_shared<xRooNode>(*par, *this));
         out.back()->fBinNumber = i;
         i++;
      }
   } else if (auto ax = GetXaxis(); ax) {
      for (int i = 1; i <= ax->GetNbins(); i++) {
         // create a RooProduct of all bin-specific factors of all shapeFactors
         std::vector<RooAbsArg *> _factors;
         for (auto f : factors()) {
            if (f->get<ParamHistFunc>()) {
               if (f->bins()[i - 1]->get<RooProduct>()) {
                  for (auto &ss : f->bins()[i - 1]->factors())
                     _factors.push_back(ss->get<RooAbsArg>());
               } else {
                  _factors.push_back(f->bins()[i - 1]->get<RooAbsArg>());
               }
            }
         }
         out.emplace_back(std::make_shared<xRooNode>(
            TString::Format("%g<=%s<%g", ax->GetBinLowEdge(i), ax->GetParent()->GetName(), ax->GetBinLowEdge(i + 1)),
            _factors.empty() ? nullptr
                             : std::make_shared<RooProduct>(TString::Format("%s.binFactors.bin%d", GetName(), i),
                                                            "binFactors", RooArgList()),
            *this));
         for (auto f : _factors) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
            out.back()->get<RooProduct>()->_compRSet.add(*f);
#else
            const_cast<RooArgList &>(out.back()->get<RooProduct>()->realComponents()).add(*f);
#endif
         }
         out.back()->fBinNumber = i;
      }
   }

   return out;
}

xRooNode xRooNode::coefs(bool recurse) const
{
   RooArgList coefs;

   if (recurse && fParent) {
      // get our coefs and multiply it by the parents coefs ...
      auto ourCoefs = xRooNode::coefs(false);
      auto parentCoefs = fParent->coefs(true);
      if (!parentCoefs.get<RooAbsReal>()) {
         // no coefs to include, just return our coefs
         return ourCoefs;
      }
      if (!ourCoefs.get<RooAbsReal>()) {
         // just return the parent's coefs
         return parentCoefs;
      }
      // if got here, must combine parentCoefs and outCoefs into a RooProduct
      xRooNode out(".recursiveCoefs",
                   std::make_shared<RooProduct>(".recursiveCoefs",
                                                TString::Format("Recursive Coefficients of %s", GetName()),
                                                *ourCoefs.get<RooAbsReal>(), *parentCoefs.get<RooAbsReal>()),
                   *this);
      // keep alive the two coef nodes by adding to out's memory
      auto mem = out.emplace_back(std::make_shared<xRooNode>(".memory", nullptr, *this));
      mem->emplace_back(std::make_shared<xRooNode>(ourCoefs));
      mem->emplace_back(std::make_shared<xRooNode>(parentCoefs));
      return out;
   }

   bool isResidual = false;

   // if parent is a sumpdf or addpdf then include the coefs
   // if func appears multiple times then coefs must be combined into a RooAddition temporary
   if (fParent) {
      // handle case where filters are applied .. need to pass through these
      // do this by iterating while fComp is null
      auto parent = fParent;
      if (!parent->fComp) {
         while (!parent->fComp && parent->fParent) {
            parent = parent->fParent;
         }
         // parent should now be node above the filters ... need parent of that
         parent = parent->fParent;
         if (!parent)
            parent = fParent; // revert t original parent in case something went wrong
      }
      if (auto p = parent->get<RooRealSumPdf>(); p) {
         std::size_t i = 0;
         for (auto &o : p->funcList()) {
            if (o == get()) {
               if (i >= p->coefList().size()) {
                  isResidual = true;
                  coefs.add(p->coefList());
               } else {
                  coefs.add(*p->coefList().at(i));
               }
            }
            i++;
         }
      } else if (auto p2 = parent->get<RooAddPdf>(); p2) {
         std::size_t i = 0;
         if (p2->coefList().empty()) {
            // this can happen if all pdfs are extended then the coef is effectively the
            // expected number of events
            // TODO: test behaviour of xRooNode under this scenario (are histograms correct?)
         } else {
            for (auto &o : p2->pdfList()) {
               if (o == get()) {
                  if (i >= p2->coefList().size()) {
                     isResidual = true;
                     coefs.add(p2->coefList());
                  } else {
                     coefs.add(*p2->coefList().at(i));
                  }
               }
               i++;
            }
         }
      }
   }
   if (isResidual) {
      // return a node representing 1.-sumOfCoefs
      // involves creating sumOfCoefs unless there is only 1 coef, then just use that
      auto coefSum = coefs.empty()
                        ? nullptr
                        : (coefs.size() == 1 ? std::shared_ptr<RooAbsArg>(coefs.at(0), [](RooAbsArg *) {})
                                             : std::make_shared<RooAddition>((isResidual) ? ".sumOfCoefs" : ".coefs",
                                                                             "Coefficients of", coefs));
      xRooNode out(".coef", coefSum ? std::dynamic_pointer_cast<RooAbsArg>(std::make_shared<RooFormulaVar>(
                                         ".coef", "1-sum(otherCoefs)", "1. - @0", *coefSum))
                                    : nullptr /* should we return a "1." instead? */);
      if (coefSum && coefs.size() != 1) {
         out.emplace_back(std::make_shared<xRooNode>(".memory", nullptr, *this))
            ->emplace_back(
               std::make_shared<xRooNode>(".sumOfCoefs", coefSum, out)); // added to keep the sum alive! with the node
      }
      if (!coefs.empty()) {
         out.browse();
      }
      return out;
   } else if (coefs.size() == 1) {
      xRooNode out(".coef", std::shared_ptr<RooAbsArg>(coefs.at(0), [](RooAbsArg *) {}), *this);
      if (!coefs.empty()) {
         out.browse();
      }
      return out;
   } else {
      auto coefSum =
         coefs.empty()
            ? nullptr
            : std::make_shared<RooAddition>(".coefs", TString::Format("Coefficients of %s", GetName()), coefs);
      xRooNode out(".coefs", coefSum, *this);
      if (!coefs.empty())
         out.browse();

      return out;
   }
}

xRooNode xRooNode::factors() const
{
   xRooNode out(".factors", nullptr, *this);

   if (auto p = get<RooProdPdf>(); p) {
      auto _main = mainChild();
      if (auto a = _main.get<RooRealSumPdf>(); a && !a->getStringAttribute("alias")) {
         a->setStringAttribute("alias", "samples");
      } else if (auto a2 = _main.get<RooAddPdf>(); a2 && !a2->getStringAttribute("alias")) {
         a2->setStringAttribute("alias", "components");
      }
      int _npdfs = p->pdfList().size();
      for (auto &o : p->pdfList()) {
         out.emplace_back(std::make_shared<xRooNode>(*o, *this));
         if (_npdfs > 5 && o != _main.get())
            out.back()->fFolder = "!constraints";
      }
   } else if (auto p2 = get<RooProduct>(); p2) {
      for (auto &o : p2->components()) {
         if (o->InheritsFrom("RooProduct")) {
            // get factors of this term
            auto x = xRooNode("tmp", *o, *this).factors();
            for (auto &n : x) {
               out.emplace_back(std::make_shared<xRooNode>(n->GetName(), n->fComp, *this));
            }
         } else {
            out.emplace_back(std::make_shared<xRooNode>(*o, *this));
         }
      }
   } else if (auto w = get<RooWorkspace>(); w) {
      // if workspace, return all functions (not pdfs) that have a RooProduct as one of their clients
      // or not clients
      // exclude obs and globs
      auto oo = obs(); // need to keep alive as may contain owning globs
      auto &_obs = *(oo.get<RooArgList>());
      for (auto a : w->allFunctions()) {
         if (_obs.contains(*a))
            continue;
         bool show(true);
         for (auto c : a->clients()) {
            show = false;
            if (c->InheritsFrom("RooProduct")) {
               show = true;
               break;
            }
         }
         if (show)
            out.emplace_back(std::make_shared<xRooNode>(*a, *this));
      }
   }

   /*
       // if parent is a sumpdf or addpdf then include the coefs
       // if func appears multiple times then coefs must be combined into a RooAddition temporary
       if (fParent) {
           RooArgList coefs;
           if(auto p = fParent->get<RooRealSumPdf>();p) {
               int i=0;
               for(auto& o : p->funcList()) {
                   if (o == get()) {
                       coefs.add( *p->coefList().at(i) );
                   }
                   i++;
               }
           } else if(auto p = fParent->get<RooAddPdf>(); p) {
               int i=0;
               for(auto& o : p->pdfList()) {
                   if (o == get()) {
                       coefs.add( *p->coefList().at(i) );
                   }
                   i++;
               }
           }
           if (!coefs.empty()) {
               if (coefs.size() == 1) {
                   if (strcmp(coefs.at(0)->GetName(),"1")) { // don't add the "1"
                       out.emplace_back(std::make_shared<Node2>(".coef", *coefs.at(0), *this));
                   }
               } else {
                   out.emplace_back(std::make_shared<Node2>(".coefs",
                                                            std::make_shared<RooAddition>(".coefs", "Coefficients of",
                                                                                          coefs), *this));
               }
           }
       }
   */
   return out;
}

xRooNode xRooNode::variations() const
{
   xRooNode out(".variations", nullptr, *this);

   //   if (auto p = get<RooSimultaneous>(); p) {
   //      for (auto &c : p->indexCat()) {
   //         auto pp = p->getPdf(c.first.c_str());
   //         if (!pp)
   //            continue;
   //         out.emplace_back(
   //            std::make_shared<xRooNode>(TString::Format("%s=%s", p->indexCat().GetName(), c.first.c_str()), *pp,
   //            *this));
   //      }
   //   } else
   if (auto p2 = get<PiecewiseInterpolation>(); p2) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      out.emplace_back(std::make_shared<xRooNode>("nominal", p2->_nominal.arg(), *this));
#else
      out.emplace_back(std::make_shared<xRooNode>("nominal", *(p2->nominalHist()), *this));
#endif
      for (size_t i = 0; i < p2->paramList().size(); i++) {
         // TODO: should we only return one if we find they are symmetrized?
         out.emplace_back(std::make_shared<xRooNode>(TString::Format("%s=1", p2->paramList().at(i)->GetName()),
                                                     *p2->highList().at(i), *this));
         out.emplace_back(std::make_shared<xRooNode>(TString::Format("%s=-1", p2->paramList().at(i)->GetName()),
                                                     *p2->lowList().at(i), *this));
      }
   } else if (auto p3 = get<RooStats::HistFactory::FlexibleInterpVar>(); p3) {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      out.emplace_back(std::make_shared<xRooNode>("nominal", RooFit::RooConst(p3->_nominal), *this));
      for (size_t i = 0; i < p3->_paramList.size(); i++) {
         out.emplace_back(std::make_shared<xRooNode>(TString::Format("%s=1", p3->_paramList.at(i)->GetName()),
                                                     RooFit::RooConst(p3->_high.at(i)), *this));
         out.emplace_back(std::make_shared<xRooNode>(TString::Format("%s=-1", p3->_paramList.at(i)->GetName()),
                                                     RooFit::RooConst(p3->_low.at(i)), *this));
      }
#else
      out.emplace_back(std::make_shared<xRooNode>("nominal", RooFit::RooConst(p3->nominal()), *this));
      for (size_t i = 0; i < p3->variables().size(); i++) {
         out.emplace_back(std::make_shared<xRooNode>(TString::Format("%s=1", p3->variables().at(i)->GetName()),
                                                     RooFit::RooConst(p3->high().at(i)), *this));
         out.emplace_back(std::make_shared<xRooNode>(TString::Format("%s=-1", p3->variables().at(i)->GetName()),
                                                     RooFit::RooConst(p3->low().at(i)), *this));
      }
#endif

   } else if (auto p4 = get<ParamHistFunc>(); p4) {
      // I *think* I put this here so that can browse into a ParamHistFunc
      //      int i = 0;
      //      for (auto par : p4->_paramSet) {
      //         TString _name = par->GetName();
      //         // if(auto _v = dynamic_cast<RooRealVar*>(p->_dataSet.get(i)->first()); _v) {
      //         //     _name = TString::Format("%s=%g",_v->GetName(),_v->getVal());
      //         // }
      //         // out.emplace_back(std::make_shared<xRooNode>(_name,*par,*this)); -- -removed cos now have bin()
      //         method i++;
      //      }
   }
   return out;
}

RooArgList xRooNode::argList() const
{
   RooArgList out;
   out.setName(GetName());
   for (auto &k : *this) {
      if (auto o = k->get<RooAbsArg>(); o)
         out.add(*o);
   }
   return out;
}

xRooNode xRooNode::datasets() const
{
   xRooNode out(".datasets()", nullptr, *this);
   // removed the browse operation since no longer showing '.datasets()' in browser
   // and otherwise this means dataset reduction operation will be called every time we 'browse()' the datasets node
   // out.fBrowseOperation = [](xRooNode *f) { return f->fParent->datasets(); };

   if (auto _ws = get<RooWorkspace>(); _ws) {
      for (auto &d : _ws->allData()) {
         out.emplace_back(std::make_shared<xRooNode>(*d, *this));
         out.back()->fFolder = "!datasets";
      }
   } else if (get<RooAbsPdf>() ||
              (!get() && fParent &&
               fParent->get<RooAbsPdf>())) { // second condition handles 'bins' nodes of pdf, which have null ptr
      // only add datasets that have observables that cover all our observables
      auto oo = obs(); // must keep alive in case is owning the globs
      RooArgSet _obs(*oo.get<RooArgList>());
      //_obs.add(coords(true).argList(), true); // include coord observables too, and current xaxis if there's one -
      // added in loop below

      TString cut;
      RooArgSet cutobs;
      for (auto _c : coords()) { // coords() moves vars to their respective coordinates too
         if (auto _cat = _c->get<RooAbsCategoryLValue>(); _cat) {
            if (cut != "")
               cut += " && ";
            cut += TString::Format("%s==%d", _cat->GetName(), _cat->getCurrentIndex());
            _obs.add(*_cat,
                     true); // note: if we ever changed coords to return clones, would need to keep coords alive
            cutobs.add(*_cat);
         } else if (auto _rv = _c->get<RooAbsRealLValue>(); _rv) {
            // todo: check coordRange is a single range rather than multirange
            if (cut != "")
               cut += " && ";
            cut += TString::Format("%s>=%f&&%s<%f", _rv->GetName(), _rv->getMin(_rv->getStringAttribute("coordRange")),
                                   _rv->GetName(), _rv->getMax(_rv->getStringAttribute("coordRange")));
            _obs.add(*_rv,
                     true); // note: if we ever changed coords to return clones, would need to keep coords alive
            cutobs.add(*_rv);
         } else {
            throw std::runtime_error("datasets(): Unsupported coordinate type");
         }
      }
      if (auto s = get<RooSimultaneous>()) {
         // check if we have a pdf for every category ... if not then add to cut
         bool hasMissing = false;
         TString extraCut = "";
         for (auto cat : s->indexCat()) {
            if (!s->getPdf(cat.first.c_str())) {
               hasMissing = true;
            } else {
               if (extraCut != "")
                  extraCut += " || ";
               extraCut += TString::Format("%s==%d", s->indexCat().GetName(), cat.second);
            }
         }
         if (hasMissing) {
            if (cut != "")
               cut += " && ";
            cut += "(" + extraCut + ")";
            cutobs.add(s->indexCat());
         }
      }

      if (auto ax = GetXaxis(); ax && dynamic_cast<RooAbsArg *>(ax->GetParent())->getAttribute("obs")) {
         auto a = dynamic_cast<RooAbsArg *>(ax->GetParent());
         _obs.add(*a, true);
      }
      xRooNode _datasets; // will be any child datasets, along with datasets of the workspace
      for (auto &child : *this) {
         if (child->get<RooAbsData>())
            _datasets.push_back(child);
      }
      if (auto __ws = ws(); __ws) {
         xRooNode _wsNode(*__ws, *this);
         for (auto &d : _wsNode.datasets()) {
            _datasets.push_back(d);
         }
      }

      for (auto &d : _datasets) {
         if (std::unique_ptr<RooAbsCollection>(d->obs().argList().selectCommon(_obs))->size() == _obs.size()) {
            // all obs present .. include

            if (cut != "") {
               RooFormulaVar cutFormula("cut1", cut, cutobs); // doing this to avoid complaints about unused vars
               // TODO: Could consider using a 'filter' node (see filter() method) applied to the dataset instead
               // of creating and using a reduced dataset here
               out.emplace_back(std::make_shared<xRooNode>(
                  std::shared_ptr<RooAbsData>(d->get<RooAbsData>()->reduce(
                     *std::unique_ptr<RooAbsCollection>(d->robs().get<RooArgList>()->selectCommon(_obs)), cutFormula)),
                  *this));
               // put a subset of the globs in the returned dataset too
               out.back()->get<RooAbsData>()->setGlobalObservables(*std::unique_ptr<RooAbsCollection>(
                  d->globs().get<RooArgList>()->selectCommon(*globs().get<RooArgList>())));
               if (d->get()->TestBit(1 << 20))
                  out.back()->get()->SetBit(1 << 20);
               // need to attach the original dataset so that things like SetBinContent can interact with it
               out.back()->fBrowsables.emplace_back(std::make_shared<xRooNode>(".sourceds", d->fComp, *this));
            } else {
               out.emplace_back(std::make_shared<xRooNode>(d->fComp, *this));
            }
         }
      }
      /*else if(auto p = get<RooFitResult>(); p) {
         // look for datasets in workspace that match the fit result name after hashing
         for(auto& _d : xRooNode(*_ws,*this).datasets()) {
             auto _hash = RooAbsTree::nameToHash(_d->get()->GetName());
             if (TString::Format("%d;%d",_hash.first,_hash.second) == p->GetTitle()) {
                 out.emplace_back(std::make_shared<xRooNode>(_d->fComp, *this));
             }
         }
     }*/
   } else if (auto mc = get<RooStats::ModelConfig>()) {
      return xRooNode(*mc->GetPdf(), fParent).datasets();
   }

   return out;
}

std::shared_ptr<xRooNode> xRooNode::getBrowsable(const char *name) const
{
   for (auto b : fBrowsables) {
      if (b && strcmp(b->GetName(), name) == 0)
         return b;
   }
   return nullptr;
}

TGraph *xRooNode::BuildGraph(RooAbsLValue *v, bool includeZeros, TVirtualPad *fromPad) const
{

   if (auto fr = get<RooFitResult>(); fr) {
      return nullptr;
   }

   if (auto theData = get<RooDataSet>(); theData) {

      TH1 *theHist = nullptr;

      if (fromPad) {
         // find first histogram in pad
         for (auto o : *fromPad->GetListOfPrimitives()) {
            theHist = dynamic_cast<TH1 *>(o);
            if (theHist) {
               theHist = static_cast<TH1 *>(theHist->Clone());
               theHist->Reset();
               break;
            } // clone because theHist gets deleted below
         }
      }

      if (!theHist) {
         auto _parentPdf = parentPdf();
         if (!_parentPdf) {
            // can still build graph if v is an obs ... will use v binning
            auto vo = dynamic_cast<TObject *>(v);
            if (v && obs().find(vo->GetName())) {
               if (auto cat = dynamic_cast<RooAbsCategoryLValue *>(v)) {
                  theHist = new TH1D(
                     TString::Format("%s_%s", GetName(), vo->GetName()),
                     TString::Format("my temp hist;%s", strlen(vo->GetTitle()) ? vo->GetTitle() : vo->GetName()),
                     cat->numTypes(), 0, cat->numTypes());
                  int i = 1;
                  std::map<int, std::string> cats; // fill into a map to preserve index ordering
                  for (auto &c : *cat) {
                     cats[c.second] = c.first;
                  }
                  for (auto &[_, label] : cats) {
                     theHist->GetXaxis()->SetBinLabel(i++, label.c_str());
                  }
               } else {
                  auto _binning = v->getBinningPtr(nullptr);
                  if (_binning->isUniform()) {
                     theHist = new TH1D(
                        TString::Format("%s_%s", GetName(), vo->GetName()),
                        TString::Format("my temp hist;%s", strlen(vo->GetTitle()) ? vo->GetTitle() : vo->GetName()),
                        v->numBins(), _binning->lowBound(), _binning->highBound());
                  } else {
                     theHist = new TH1D(
                        TString::Format("%s_%s", GetName(), vo->GetName()),
                        TString::Format("my temp hist;%s", strlen(vo->GetTitle()) ? vo->GetTitle() : vo->GetName()),
                        v->numBins(), _binning->array());
                  }
               }
            } else {
               throw std::runtime_error("Cannot draw dataset without parent PDF");
            }
         } else {
            theHist = _parentPdf->BuildHistogram(v, true);
         }
      }
      if (!theHist)
         return nullptr;
      // this hist will get filled with w*x to track weighted x position per bin
      TH1 *xPos = static_cast<TH1 *>(theHist->Clone("xPos"));
      xPos->Reset();
      TH1 *xPos2 = static_cast<TH1 *>(theHist->Clone("xPos2"));
      xPos2->Reset();
      auto nHist = std::unique_ptr<TH1>(static_cast<TH1 *>(theHist->Clone("nEntries")));
      nHist->Reset();

      auto dataGraph = new TGraphAsymmErrors;
      dataGraph->SetEditable(false);
      dataGraph->SetName(GetName());
      dataGraph->SetTitle(strlen(theData->GetTitle()) ? theData->GetTitle() : theData->GetName());
      // next line triggers creation of the histogram inside the graph, in root 6.22 that isn't protected from being
      // added to gDirectory
      dataGraph->SetTitle(TString::Format("%s;%s;Events", dataGraph->GetTitle(), theHist->GetXaxis()->GetTitle()));
      *static_cast<TAttMarker *>(dataGraph) = *static_cast<TAttMarker *>(theHist);
      *static_cast<TAttLine *>(dataGraph) = *static_cast<TAttLine *>(theHist);
      dataGraph->SetMarkerStyle(20);
      dataGraph->SetLineColor(kBlack);

      auto _obs = obs();

      // auto x = theData->get()->find((v) ? dynamic_cast<TObject*>(v)->GetName() : theHist->GetXaxis()->GetName());
      // const RooAbsReal* xvar = (x) ? dynamic_cast<RooAbsReal*>(x) : nullptr;
      // const RooAbsCategory* xcat = (x && !xvar) ? dynamic_cast<RooAbsCategory*>(x) : nullptr;
      auto x = _obs.find((v) ? dynamic_cast<TObject *>(v)->GetName()
                             : (theHist->GetXaxis()->IsAlphanumeric() ? theHist->GetXaxis()->GetTimeFormatOnly()
                                                                      : theHist->GetXaxis()->GetName()));
      if (x && x->get<RooAbsArg>()->getAttribute("global")) {
         // is global observable ...
         dataGraph->SetPoint(0, x->get<RooAbsReal>()->getVal(), 1e-15);
         dataGraph->SetTitle(TString::Format("%s = %f", dataGraph->GetTitle(), dataGraph->GetPointX(0)));
         delete xPos;
         delete xPos2;
         delete theHist;
         return dataGraph;
      }

      const RooAbsReal *xvar = (x) ? x->get<RooAbsReal>() : nullptr;
      const RooAbsCategory *xcat = (x && !xvar) ? x->get<RooAbsCategory>() : nullptr;

      auto _coords = coords();

      TString pName((fromPad) ? fromPad->GetName() : "");
      auto _pos = pName.Index('=');

      int nevent = theData->numEntries();
      for (int i = 0; i < nevent; i++) {
         theData->get(i);
         bool _skip = false;
         for (auto _c : _coords) {
            if (auto cat = _c->get<RooAbsCategoryLValue>(); cat) {
               if (cat->getIndex() != theData->get()->getCatIndex(cat->GetName())) {
                  _skip = true;
                  break;
               }
            } else if (auto rv = _c->get<RooAbsRealLValue>(); rv) {
               // must be in range
               if (!rv->inRange(theData->get()->getRealValue(rv->GetName()), rv->getStringAttribute("coordRange"))) {
                  _skip = true;
                  break;
               }
            }
         }
         if (_pos != -1) {
            if (auto cat = dynamic_cast<RooAbsCategory *>(theData->get()->find(TString(pName(0, _pos))));
                cat && cat->getLabel() != pName(_pos + 1, pName.Length())) {
               _skip = true;
            }
         }
         if (_skip)
            continue;

         if (xvar) {
            xPos->Fill(xvar->getVal(), xvar->getVal() * theData->weight());
            xPos2->Fill(xvar->getVal(), pow(xvar->getVal(), 2) * theData->weight());
         }

         if (xcat) {
            theHist->Fill(xcat->getLabel(), theData->weight());
            nHist->Fill(xcat->getLabel(), 1);
         } else {
            theHist->Fill((x) ? xvar->getVal() : 0.5, theData->weight());
            nHist->Fill((x) ? xvar->getVal() : 0.5, 1);
         }
      }

      xPos->Divide(theHist);
      xPos2->Divide(theHist);

      // update the x positions to the means for each bin and use poisson asymmetric errors for data ..
      for (int i = 0; i < theHist->GetNbinsX(); i++) {
         if (includeZeros || nHist->GetBinContent(i + 1)) {
            double val = theHist->GetBinContent(i + 1);

            dataGraph->SetPoint(dataGraph->GetN(),
                                (xvar && val) ? xPos->GetBinContent(i + 1) : theHist->GetBinCenter(i + 1), val);

            // x-error will be the (weighted) standard deviation of the x values ...
            double xErr = xPos2->GetBinContent(i + 1) - pow(xPos->GetBinContent(i + 1), 2);
            xErr = (xErr <= 0) ? 0. : sqrt(xErr); // protects against floating point rounding effects

            if (xErr || val) {
               dataGraph->SetPointError(dataGraph->GetN() - 1, xErr, xErr,
                                        val - 0.5 * TMath::ChisquareQuantile(TMath::Prob(1, 1) / 2., 2. * (val)),
                                        0.5 * TMath::ChisquareQuantile(1. - TMath::Prob(1, 1) / 2., 2. * (val + 1)) -
                                           val);
            }
         }
      }

      // transfer limits from theHist to dataGraph hist
      dataGraph->GetHistogram()->GetXaxis()->SetLimits(theHist->GetXaxis()->GetXmin(), theHist->GetXaxis()->GetXmax());
      // and bin labels, if any
      if (xcat) {
         dataGraph->GetHistogram()->GetXaxis()->Set(theHist->GetNbinsX(), 0, theHist->GetNbinsX());
         for (int i = 1; i <= theHist->GetNbinsX(); i++)
            dataGraph->GetHistogram()->GetXaxis()->SetBinLabel(i, theHist->GetXaxis()->GetBinLabel(i));
      }

      delete xPos;
      delete xPos2;
      delete theHist;

      //        std::shared_ptr<TStyle> style; // use to keep alive for access from GetStyle below, in case getObject
      //        has decided to return the owning ptr (for some reason) std::string _title =
      //        strlen(dataGraph->GetTitle()) ? dataGraph->GetTitle() : GetName(); if (!gROOT->GetStyle(_title.c_str()))
      //        {
      //            if ( (style = getObject<TStyle>(_title)) ) {
      //                // loaded style (from workspace?) so put in list and use that
      //                gROOT->GetListOfStyles()->Add(style.get());
      //            } else {
      //                // create new style - gets put in style list automatically so don't have to delete
      //                // acquire them so saved to workspaces for auto reload ...
      //                style = const_cast<xRooNode&>(*this).acquireNew<TStyle>(_title.c_str(),
      //                                           TString::Format("Style for %s component", _title.c_str()));
      //                (TAttLine &) (*style) = *dynamic_cast<TAttLine *>(dataGraph);
      //                (TAttFill &) (*style) = *dynamic_cast<TAttFill *>(dataGraph);
      //                (TAttMarker &) (*style) = *dynamic_cast<TAttMarker *>(dataGraph);
      //                gROOT->GetListOfStyles()->Add(style.get());
      //            }
      //        }
      auto _styleNode = styles(dataGraph);
      if (auto _style = _styleNode.get<TStyle>()) {
         *dynamic_cast<TAttLine *>(dataGraph) = *_style;
         *dynamic_cast<TAttFill *>(dataGraph) = *_style;
         *dynamic_cast<TAttMarker *>(dataGraph) = *_style;
      }
      return dataGraph;
   }

   throw std::runtime_error("Cannot build graph");
}

void xRooNode::SetFitResult(const RooFitResult *fr)
{
   if (fr) {
      if (auto _w = ws(); _w) {
         auto res = acquire(std::shared_ptr<RooFitResult>(const_cast<RooFitResult *>(fr), [](RooFitResult *) {}));
         for (auto o : _w->allGenericObjects()) {
            if (auto _fr = dynamic_cast<RooFitResult *>(o); _fr) {
               _fr->ResetBit(1 << 20);
            }
         }
         res->SetBit(1 << 20);
         // assign values
         auto allVars = _w->allVars();
         allVars = fr->floatParsFinal();
         allVars = fr->constPars();
      } else {
         // need to add to memory as a specific name
         throw std::runtime_error("Not supported yet"); // complication is how to replace an existing fitResult in
                                                        // .memory auto _clone = std::make_shared<RooFitResult>(*fr);
         //_clone->SetName("fitResult");
      }
   } else {
      SetFitResult(fitResult("prefit").get<RooFitResult>());
   }
}

void xRooNode::SetFitResult(const xRooNode &fr)
{
   if (auto _fr = fr.get<const RooFitResult>()) {
      SetFitResult(_fr);
   } else
      throw std::runtime_error("Not a RooFitResult");
}

xRooNode xRooNode::fitResult(const char *opt) const
{

   if (get<RooFitResult>())
      return *this;
   if (get<RooAbsData>()) {
      if (auto _fr = find(".fitResult"); _fr)
         return _fr;
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
      // check if weightVar of RooAbsData has fitResult attribute on it, will be the generation fit result
      if (get<RooDataSet>() && get<RooDataSet>()->weightVar() &&
          get<RooDataSet>()->weightVar()->getStringAttribute("fitResult")) {
         return xRooNode(getObject<const RooFitResult>(get<RooDataSet>()->weightVar()->getStringAttribute("fitResult")),
                         *this);
      }
#endif
      return xRooNode();
   }

   TString sOpt(opt);
   if (sOpt == "prefit") {
      // build a fitResult using nominal values and infer errors from constraints
      // that aren't the 'main' constraints
      // Warning("fitResult","Building prefitResult by examining pdf. Consider setting an explicit prefitResult
      // (SetFitResult(fr)) where fr name is prefitResult");

      // ensure coefs are included if there are any
      auto _coefs = coefs();
      if (_coefs.get()) {
         return xRooNode(RooProduct("tmp", "tmp", RooArgList(*get<RooAbsArg>(), *_coefs.get<RooAbsReal>())))
            .fitResult(opt);
      }

      std::unique_ptr<RooArgList> _pars(dynamic_cast<RooArgList *>(pars().argList().selectByAttrib("Constant", false)));
      auto fr = std::make_shared<RooFitResult>("prefitResult", "Prefit");
      fr->setFinalParList(*_pars);
      for (auto &p : fr->floatParsFinal()) {
         auto _v = dynamic_cast<RooRealVar *>(p);
         if (!_v)
            continue;
         if (auto s = _v->getStringAttribute("nominal"); s)
            _v->setVal(TString(s).Atof());
         auto _constr = xRooNode(fParent->getObject<RooRealVar>(p->GetName()), *this).constraints();
         std::shared_ptr<xRooNode> pConstr;
         for (auto &c : _constr) {
            if (c->get<RooPoisson>() || c->get<RooGaussian>()) {
               // require parameter to be a direct server of the constraint pdf to count
               bool isServer = true;
               if (c->get<RooGaussian>()) {
                  isServer = false;
                  for (auto s : c->get<RooAbsArg>()->servers()) {
                     if (strcmp(s->GetName(), p->GetName()) == 0) {
                        isServer = true;
                        break;
                     }
                  }
               }
               if (isServer) {
                  pConstr = c;
                  break;
               }
            }
         }
         if (pConstr) {
            // there will be 3 deps, one will be this par, the other two are the mean and error (or error^2 in case of
            // poisson use the one that's a ConstVar as the error to break a tie ...
            double prefitVal = 0;
            double prefitError = 0;
            for (auto &_d : pConstr->vars()) {
               if (strcmp(p->GetName(), _d->get()->GetName()) == 0)
                  continue;
               if (auto _c = _d->get<RooConstVar>(); _c && _c->getVal() != 0) {
                  if (prefitError)
                     prefitVal = prefitError; // loading val into error already, so move it over
                  prefitError = _c->getVal();
               } else if (prefitError == 0) {
                  prefitError = _d->get<RooAbsReal>()->getVal();
               } else {
                  prefitVal = _d->get<RooAbsReal>()->getVal();
               }
            }

            if (pConstr->get<RooGaussian>() && pConstr->browse().find(".sigma")) {
               prefitError = pConstr->find(".sigma")->get<RooAbsReal>()->getVal();
            }
            // std::cout << p->GetName() << " extracted " << prefitVal << " " << prefitError << " from ";
            // pConstr->deps().Print();
            if (pConstr->get<RooPoisson>()) {
               // prefitVal will be the global observable value, need to divide that by tau
               prefitVal /= prefitError;
               // prefiterror will be tau ... need 1/sqrt(tau) for error
               prefitError = 1. / sqrt(prefitError);
            }
            if (!_v->getStringAttribute("nominal"))
               _v->setVal(prefitVal);
            _v->setError(prefitError);
         } else {
            // unconstrained, remove error
            _v->removeError();
         }
      }
      auto _args = consts().argList();
      _args.add(pp().argList());
      // global obs are added to constPars list too
      auto _globs = globs(); // keep alive as may own glob
      _args.add(_globs.argList());
      fr->setConstParList(_args);
      std::unique_ptr<RooArgList> _snap(dynamic_cast<RooArgList *>(_pars->snapshot()));
      for (auto &p : *_snap) {
         if (auto atr = p->getStringAttribute("initVal"); atr && dynamic_cast<RooRealVar *>(p))
            dynamic_cast<RooRealVar *>(p)->setVal(TString(atr).Atof());
      }
      fr->setInitParList(*_snap);
      return xRooNode(fr, *this);
   }

   // return first checked fit result present in the workspace
   if (auto _w = ws(); _w) {
      auto checkFr = [&](TObject *o) {
         if (auto _fr = dynamic_cast<RooFitResult *>(o); _fr && _fr->TestBit(1 << 20)) {
            // check all pars match final/const values ... if mismatch need to create a new RooFitResult
            bool match = true;
            for (auto p : pars()) {
               if (!p->get<RooAbsReal>()) {
                  if (auto cat = p->get<RooAbsCategory>();
                      cat && cat->getCurrentIndex() ==
                                _fr->floatParsFinal().getCatIndex(cat->GetName(), std::numeric_limits<int>().max())) {
                     match = false;
                     break;
                  }
               } else if (p->get<RooAbsArg>()->getAttribute("Constant")) {
                  if (_fr->floatParsFinal().find(p->GetName()) ||
                      std::abs(_fr->constPars().getRealValue(p->GetName(), std::numeric_limits<double>::quiet_NaN()) -
                               p->get<RooAbsReal>()->getVal()) > 1e-15) {
                     match = false;
                     break;
                  }
               } else {
                  if (_fr->constPars().find(p->GetName()) ||
                      std::abs(
                         _fr->floatParsFinal().getRealValue(p->GetName(), std::numeric_limits<double>::quiet_NaN()) -
                         p->get<RooAbsReal>()->getVal()) > 1e-15) {
                     match = false;
                     break;
                  }
               }
            }
            if (!match) {
               // create new fit result using covariances from this fit result
               std::unique_ptr<RooArgList> _pars(
                  dynamic_cast<RooArgList *>(pars().argList().selectByAttrib("Constant", false)));
               auto fr = std::make_shared<RooFitResult>(TString::Format("%s-dirty", _fr->GetName()));
               fr->SetTitle(TString::Format("%s parameter snapshot", GetName()));
               fr->setFinalParList(*_pars);
               TMatrixTSym<double> *prevCov = static_cast<TMatrixTSym<double> *>(GETDMP(_fr, _VM));
               if (prevCov) {
                  auto cov = _fr->reducedCovarianceMatrix(*_pars);
                  // make the diagonals all the current error values
                  for (size_t i = 0; i < _pars->size(); i++) {
                     if (auto v = dynamic_cast<RooRealVar *>(_pars->at(i))) {
                        cov(i, i) = pow(v->getError(), 2);
                     } else {
                        cov(i, i) = 0;
                     }
                  }
                  fr->setCovarianceMatrix(cov);
               }

               auto _args = consts().argList();
               _args.add(pp().argList());
               // global obs are added to constPars list too
               auto _globs = globs(); // keep alive as may own glob
               _args.add(_globs.argList());
               fr->setConstParList(_args);
               std::unique_ptr<RooArgList> _snap(dynamic_cast<RooArgList *>(_pars->snapshot()));
               for (auto &p : *_snap) {
                  if (auto atr = p->getStringAttribute("initVal"); atr && dynamic_cast<RooRealVar *>(p))
                     dynamic_cast<RooRealVar *>(p)->setVal(TString(atr).Atof());
               }
               fr->setInitParList(*_snap);
               return xRooNode(fr, *this);
            }
            return xRooNode(*_fr, std::make_shared<xRooNode>(*_w, std::make_shared<xRooNode>()));
         }
         return xRooNode();
      };
      for (auto o : _w->allGenericObjects()) {
         auto out = checkFr(o);
         if (out)
            return out;
      }
      for (auto o : GETWSSNAPSHOTS(_w)) {
         auto out = checkFr(o);
         if (out)
            return out;
      }
   } else {
      // objects not in workspaces are allowed to have a fitResult set in their memory
      // use getObject to get it
      if (auto fr = getObject<RooFitResult>(".fitResult"); fr) {
         return xRooNode(fr, *this);
      }
   }

   // ensure coefs are included if there are any
   auto _coefs = coefs();
   if (_coefs.get()) {
      return xRooNode(RooProduct("tmp", "tmp", RooArgList(*get<RooAbsArg>(), *_coefs.get<RooAbsReal>())))
         .fitResult(opt);
   }

   std::unique_ptr<RooArgList> _pars(dynamic_cast<RooArgList *>(pars().argList().selectByAttrib("Constant", false)));
   auto fr = std::make_shared<RooFitResult>(TUUID().AsString());
   fr->SetTitle(TString::Format("%s uncorrelated parameter snapshot", GetName()));
   fr->setFinalParList(*_pars);
   fr->setStatus(-1);

   TMatrixDSym cov(fr->floatParsFinal().size());
   TMatrixTSym<double> *prevCov = static_cast<TMatrixTSym<double> *>(GETDMP(fr.get(), _VM));
   if (prevCov) {
      for (int i = 0; i < prevCov->GetNcols(); i++) {
         for (int j = 0; j < prevCov->GetNrows(); j++) {
            cov(i, j) = (*prevCov)(i, j);
         }
      }
   }
   int i = 0;
   for (auto &p : fr->floatParsFinal()) {
      if (!prevCov || i >= prevCov->GetNcols()) {
         if (auto v = dynamic_cast<RooRealVar *>(p)) {
            cov(i, i) = pow(v->getError(), 2);
         } else {
            cov(i, i) = 0;
         }
      }
      i++;
   }
   int covQualBackup = fr->covQual();
   fr->setCovarianceMatrix(cov);
   fr->setCovQual(covQualBackup);

   auto _args = consts().argList();
   _args.add(pp().argList());
   // global obs are added to constPars list too
   auto _globs = globs(); // keep alive as may own glob
   _args.add(_globs.argList());
   fr->setConstParList(_args);
   std::unique_ptr<RooArgList> _snap(dynamic_cast<RooArgList *>(_pars->snapshot()));
   for (auto &p : *_snap) {
      if (auto atr = p->getStringAttribute("initVal"); atr && dynamic_cast<RooRealVar *>(p))
         dynamic_cast<RooRealVar *>(p)->setVal(TString(atr).Atof());
   }
   fr->setInitParList(*_snap);

   // return *const_cast<Node2*>(this)->emplace_back(std::make_shared<Node2>(".fitResult",fr,*this));
   return xRooNode(fr, *this);
}

// xRooNode xRooNode::fitTo_(const char* datasetName) const {
//     try {
//         return fitTo(datasetName);
//     } catch(const std::exception& e) {
//         new TGMsgBox(gClient->GetRoot(), gClient->GetRoot(), "Exception", e.what(),kMBIconExclamation); // deletes
//         self on dismiss? return xRooNode();
//     }
// }
//
// xRooNode xRooNode::fitTo(const char* datasetName) const {
//     return fitTo(*datasets().at(datasetName));
// }

void xRooNode::SetRange(const char *range, double low, double high)
{
   if (!std::isnan(low) && !std::isnan(high) && get<RooRealVar>()) {
      if (range && strlen(range)) {
         get<RooRealVar>()->setRange(range, low, high);
      } else {
         get<RooRealVar>()->setRange(low, high);
      }
      return;
   }
   if (auto o = get<RooAbsArg>(); o)
      o->setStringAttribute("range", range);
   // todo: clear the range attribute on all servers
   // could make this controlled by a flag but probably easiest to enforce so you must set range
   // in children after if you wanted to override
}
const char *xRooNode::GetRange() const
{
   std::string &out = fRange;
   if (auto o = get<RooAbsArg>(); o && o->getStringAttribute("range"))
      out = o->getStringAttribute("range");
   auto _parent = fParent;
   while (out.empty() && _parent) {
      if (auto o = _parent->get<RooAbsArg>(); o && o->getStringAttribute("range"))
         out = o->getStringAttribute("range");
      _parent = _parent->fParent;
   }
   return out.c_str();
}

xRooNLLVar xRooNode::nll(const char *_data, std::initializer_list<RooCmdArg> nllOpts) const
{
   return nll(xRooNode(_data), nllOpts);
}

xRooNLLVar xRooNode::nll(const xRooNode &_data) const
{
   return nll(_data, *xRooFit::createNLLOptions());
}

xRooNLLVar xRooNode::nll(const xRooNode &_data, std::initializer_list<RooCmdArg> nllOpts) const
{
   auto defaultOpts = xRooFit::createNLLOptions(); // smart pointer will cleanup the list
   // add user-specified options to list ... if already existing in default list, override and warn
   RooLinkedList l;
   for (auto opt : *defaultOpts) {
      l.Add(opt);
   }
   for (auto &i : nllOpts) {
      if (auto o = l.FindObject(i.GetName())) {
         Info("nll", "Overriding NLL Option: %s", o->GetName());
         l.Remove(o);
      }
      l.Add(const_cast<RooCmdArg *>(&i));
   }

   return nll(_data, l);
}

xRooNode xRooNode::generate(const xRooNode &fr, bool expected, int seed)
{
   if (auto mc = get<RooStats::ModelConfig>()) {
      return xRooNode(*mc->GetPdf(), fParent).generate(fr, expected, seed);
   }

   if (!get<RooAbsPdf>()) {
      // before giving up, if this is a workspace we can proceed if we only have one model
      if (get<RooWorkspace>()) {
         std::shared_ptr<xRooNode> mainModel;
         for (auto &c : const_cast<xRooNode *>(this)->browse()) {
            if (c->get<RooAbsPdf>()) {
               if (!mainModel) {
                  mainModel = c;
               } else {
                  throw std::runtime_error(TString::Format("Workspace has multiple models, you must specify which to "
                                                           "generate with (found at least %s and %s)",
                                                           mainModel->GetName(), c->GetName()));
               }
            }
         }
         if (mainModel)
            return mainModel->generate(fr, expected, seed);
      }
      throw std::runtime_error(TString::Format("%s is not a pdf", GetName()));
   }
   auto _fr = fr.get<RooFitResult>();

   // when generating, will only include channels that are selected
   // any unselected but not hidden channel will have data from the only selected dataset added to it
   if (get<RooSimultaneous>()) {
      std::string selected;
      std::string fromds; // list of channels to take from selected ds
      bool hasDeselected = false;
      for (auto c : bins()) {
         TString cName(c->GetName());
         cName = cName(cName.Index('=') + 1, cName.Length());
         if (!c->get<RooAbsReal>()->isSelectedComp()) {
            hasDeselected = true;
            if (!c->get<RooAbsArg>()->getAttribute("hidden")) {
               if (!fromds.empty())
                  fromds += ",";
               fromds += cName.Data();
            }
         } else {
            if (!selected.empty())
               selected += ",";
            selected += cName.Data();
         }
      }
      if (hasDeselected) {
         std::string dsetName = "";
         if (!fromds.empty()) {
            // use the first selected dataset as protodata
            auto _dsets = datasets();
            for (auto &d : _dsets) {
               if (d->get()->TestBit(1 << 20)) {
                  dsetName = d->get()->GetName();
                  break;
               }
            }
            if (dsetName.empty()) {
               throw std::runtime_error(
                  "Need at least one dataset selected (SetChecked) to use for deselected regions");
            }
         }
         auto result = reduced(selected).generate(fr, expected, seed);
         if (!fromds.empty()) {
            auto ds = reduced(fromds).datasets()[dsetName];
            result.Add(*ds);
            result.SetName(TString(result.GetName()) + "_and_" + dsetName.c_str());
         }
         return result;
      }
   }

   return xRooNode(
      xRooFit::generateFrom(*get<RooAbsPdf>(), (_fr ? *_fr : *(fitResult().get<RooFitResult>())), expected, seed).first,
      *this);
}

xRooNLLVar xRooNode::nll(const xRooNode &_data, const RooLinkedList &opts) const
{
   if (auto mc = get<RooStats::ModelConfig>()) {
      if (mc->GetExternalConstraints()) {
         RooLinkedList optsWithConstraints;
         for (auto o : opts) {
            optsWithConstraints.Add(o->Clone(nullptr));
         }
         optsWithConstraints.Add(RooFit::ExternalConstraints(*mc->GetExternalConstraints()).Clone(nullptr));
         return xRooNode(*mc->GetPdf(), fParent).nll(_data, optsWithConstraints);
      } else {
         return xRooNode(*mc->GetPdf(), fParent).nll(_data, opts);
      }
   }

   if (!get<RooAbsPdf>()) {
      // before giving up, if this is a workspace we can proceed if we only have one model or pdf
      if (get<RooWorkspace>()) {
         std::shared_ptr<xRooNode> mainPdf, mainModel, otherPdf;
         for (auto &c : const_cast<xRooNode *>(this)->browse()) {
            if (c->get<RooAbsPdf>()) {
               if (!mainPdf) {
                  mainPdf = c;
               } else {
                  otherPdf = c;
               }
            } else if (c->get<RooStats::ModelConfig>()) {
               if (!mainModel) {
                  mainModel = c;
               } else {
                  throw std::runtime_error(TString::Format("Workspace has multiple models, you must specify which to "
                                                           "build nll with (found at least %s and %s)",
                                                           mainModel->GetName(), c->GetName()));
               }
            }
         }
         if (mainModel)
            return mainModel->nll(_data, opts);
         if (mainPdf) {
            if (otherPdf) {
               throw std::runtime_error(TString::Format("Workspace has multiple pdfs, you must specify which to "
                                                        "build nll with (found at least %s and %s)",
                                                        mainPdf->GetName(), otherPdf->GetName()));
            }
            return mainPdf->nll(_data, opts);
         }
      }
      throw std::runtime_error(TString::Format("%s is not a pdf", GetName()));
   }

   // if simultaneous and any channels deselected then reduce and return
   if (get<RooSimultaneous>()) {
      std::string selected;
      bool hasDeselected = false;
      for (auto c : bins()) {
         if (!c->get<RooAbsReal>()->isSelectedComp()) {
            hasDeselected = true;
         } else {
            TString cName(c->GetName());
            cName = cName(cName.Index('=') + 1, cName.Length());
            if (!selected.empty())
               selected += ",";
            selected += cName.Data();
         }
      }
      if (hasDeselected)
         return reduced(selected).nll(_data, opts);
   }

   if (!_data.get<RooAbsData>()) {
      // use node name to find dataset and recall
      auto _d = strlen(_data.GetName()) ? datasets().find(_data.GetName()) : nullptr;
      if (strlen(_data.GetName()) == 0) {
         // create the EXPECTED (asimov) dataset with the observables
         auto asi = xRooFit::generateFrom(*get<RooAbsPdf>(), *(fitResult().get<RooFitResult>()), true);
         _d = std::make_shared<xRooNode>(asi.first, *this);
         if (asi.second) {
            _d->emplace_back(
               std::make_shared<xRooNode>(".globs", std::const_pointer_cast<RooAbsCollection>(asi.second), *_d));
         }
      } else if (!_d) {
         throw std::runtime_error(TString::Format("Cannot find dataset %s", _data.GetName()));
      }
      return nll(*_d, opts);
   } else if (!_data.fParent || _data.fParent->fComp != fComp) {
      // dataset is not parented by this node ... meaning it may need to be reduced,
      // do this via the datasets() method by attaching and detaching the dataset
      xRooNode me(*this); // since we are in a const method, need to create a copy node.
      me.push_back(std::make_shared<xRooNode>(_data));
      return nll(*me.datasets().at(_data.GetName()), opts);
   }

   auto _globs = _data.globs(); // keep alive because may own the globs

   auto _opts = std::shared_ptr<RooLinkedList>(new RooLinkedList, [](RooLinkedList *l) {
      if (l)
         l->Delete();
      delete l;
   });
   RooArgSet _globsSet(_globs.argList());
   _opts->Add(RooFit::GlobalObservables(_globsSet).Clone());
   if (GetRange() && strlen(GetRange()))
      _opts->Add(RooFit::Range(GetRange()).Clone());

   // copy over opts ... need to clone each so can safely delete when _opts destroyed
   for (int i = 0; i < opts.GetSize(); i++) {
      if (strlen(opts.At(i)->GetName()) == 0)
         continue; // skipping "none" cmds
      if (strcmp(opts.At(i)->GetName(), "GlobalObservables") == 0) {
         // maybe warn here?
      } else {
         _opts->Add(opts.At(i)->Clone(nullptr)); // nullptr needed because accessing Clone via TObject base class puts
                                                 // "" instead, so doesnt copy names
      }
   }

   // use shared_ptr method so NLLVar will take ownership of datasets etc if created above
   // snapshots the globs out of the nllOpts (see specific constructor of xRooNLLVar)
   auto out = xRooFit::createNLL(std::dynamic_pointer_cast<RooAbsPdf>(fComp),
                                 std::dynamic_pointer_cast<RooAbsData>(_data.fComp), *_opts);
   return out;
}

// xRooNode xRooNode::fitTo(const xRooNode& _data) const {
//
//
//     auto _pdf = get<RooAbsPdf>();
//     if (!_pdf) throw std::runtime_error("Not a pdf");
//
//     auto _globs = _data.globs(); // keep alive because may own the globs
//     RooArgSet globsSet(_globs.argList());
//
//     std::shared_ptr<RooSimultaneous> newPdf;
//     if(auto s = get<RooSimultaneous>(); s) {
//         auto rangeName = GetRange();
//         if (rangeName) {
//             // need to reduce the RooSimultaneous until fix: https://github.com/root-project/root/issues/8231
//             std::vector<TString> chanPatterns;
//             TStringToken pattern(rangeName, ",");
//             while (pattern.NextToken()) {
//                 chanPatterns.emplace_back(pattern);
//             }
//             auto& _cat = const_cast<RooAbsCategoryLValue&>(s->indexCat());
//             newPdf = std::make_shared<RooSimultaneous>(TString::Format("%s_reduced",GetName()),"Reduced model",_cat);
//             for(auto& c : variations()) {
//                 TString cName(c->GetName());
//                 cName = cName(cName.Index('=')+1,cName.Length());
//                 _cat.setLabel(cName);
//                 bool matchAny=false;
//                 for(auto& p : chanPatterns) {
//                     if (cName.Contains(TRegexp(p,true))) { matchAny=true; break; }
//                     if (_cat.hasRange(p) && _cat.inRange(p)) { matchAny=true; break; }
//                 }
//                 if(matchAny) {
//                     newPdf->addPdf( *c->get<RooAbsPdf>(), cName );
//                 }
//             }
//             RooFitResultTree t(newPdf->GetName(),"",*newPdf);
//             auto _fr = std::const_pointer_cast<RooFitResult>(t.fitTo(_data.get<RooAbsData>(), &globsSet));
//             xRooNode parent(_data.GetName(),nullptr,*this);
//             xRooNode out(_fr->GetName(),/*acquire(_fr)*/ _fr,parent);
//             // do full propagation by 'checking' the fr ...
//             out.Checked(&out,true);
//             return out;
//         }
//     }
//
//
//
//     std::string treeName = TString::Format("fits_%s",GetName()).Data();
//
//     auto _frt = getObject<TTree>(treeName); // get existing frt
//
//     std::shared_ptr<RooFitResultTree> t;
//     if (_frt) {
//         t = std::make_shared<RooFitResultTree>(_frt.get());
//     } else {
//         t = std::make_shared<RooFitResultTree>(treeName.c_str(),"",*_pdf);
//     }
//     //t->SetProgress(true);
//     auto _fr = std::const_pointer_cast<RooFitResult>(t->fitTo(_data.get<RooAbsData>(), &globsSet));
//
//
//
//     /*
//     obs().argList() = s; // sets global observables to their values
//     auto _fr =
//     std::shared_ptr<RooFitResult>(_pdf->fitTo(*_data->get<RooAbsData>(),RooFit::GlobalObservables(s),RooFit::Offset(true),RooFit::Save()));
//     _fr->SetName(TUUID().AsString());
//     // restore parameters before returning
//     *std::unique_ptr<RooArgSet>(_pdf->getDependents(_fr->floatParsFinal())) = _fr->floatParsInit();
//     */
//
//     //_fr->SetTitle(TString::Format("%s;%s",GetName(),datasetName));
//     if (!_frt) {
//         t =
//         std::make_shared<RooFitResultTree>(std::dynamic_pointer_cast<TTree>(const_cast<xRooNode*>(this)->acquire(t->fTree)).get());
//     }
//     xRooNode parent(_data.GetName(),nullptr,xRooNode(t,*this));
//     xRooNode out(_fr->GetName(),/*acquire(_fr)*/ _fr,parent);
//     // do full propagation by 'checking' the fr ...
//     out.Checked(&out,true);
//     return out;
// }

std::shared_ptr<xRooNode> xRooNode::parentPdf() const
{
   // find first parent that is a pdf
   auto out = fParent;
   while (out && !out->get<RooAbsPdf>()) {
      out = out->fParent;
   }
   return out;
}

xRooNode xRooNode::reduced(const std::string &_range, bool invert) const
{
   auto rangeName = (_range.empty()) ? GetRange() : _range;
   if (!rangeName.empty()) {
      std::vector<TString> patterns;
      TStringToken pattern(rangeName, ",");
      while (pattern.NextToken()) {
         patterns.emplace_back(pattern);
      }
      if (auto s = get<RooSimultaneous>(); s) {
         // need to reduce the RooSimultaneous until fix: https://github.com/root-project/root/issues/8231
         auto &_cat = const_cast<RooAbsCategoryLValue &>(s->indexCat());
         auto newPdf =
            std::make_shared<RooSimultaneous>(TString::Format("%s_reduced", GetName()), "Reduced model", _cat);
         for (auto &c : bins()) {
            TString cName(c->GetName());
            cName = cName(cName.Index('=') + 1, cName.Length());
            _cat.setLabel(cName);
            bool matchAny = false;
            for (auto &p : patterns) {
               if (cName.Contains(TRegexp(p, true))) {
                  matchAny = true;
                  break;
               }
               if (_cat.hasRange(p) && _cat.inRange(p)) {
                  matchAny = true;
                  break;
               }
            }
            if ((matchAny && !invert) || (!matchAny && invert)) {
               newPdf->addPdf(*c->get<RooAbsPdf>(), cName);
            }
         }
         return xRooNode(newPdf, fParent);
      } else if (get() && !get<RooAbsCollection>() && !components().empty()) {
         // create a new obj and remove non-matching components
         xRooNode out(std::shared_ptr<TObject>(get()->Clone(TString::Format("%s_reduced", get()->GetName()))), fParent);
         // go through components and remove any that don't match pattern
         std::vector<TObject *> funcs; // to be removed
         for (auto &c : out.components()) {
            bool matchAny = false;
            for (auto &p : patterns) {
               if (TString(c->GetName()).Contains(TRegexp(p, true))) {
                  matchAny = true;
                  break;
               }
            }
            if (!((matchAny && !invert) || (!matchAny && invert)))
               funcs.push_back(c->get());
         }
         for (auto &c : funcs)
            out.Remove(*c);
         if (!funcs.empty()) {
            if (auto _pdf = out.get<RooRealSumPdf>(); _pdf) {
               _pdf->setFloor(false); // remove floor if removed some functions, which allows evaluation of negative
                                      // valued components
            }
         }
         out.browse();
         return out;
      } else if (auto fr = get<RooFitResult>()) {
         // reduce the fit result by moving unselected float pars into the constPars list and dropping their covariances
         xRooNode out(std::shared_ptr<TObject>(fr->Clone(TString::Format("%s_reduced", fr->GetName()))), fParent);
         fr = out.get<RooFitResult>();
         RooArgList _pars = fr->floatParsFinal();
         RooArgList _remPars;
         for (auto c : _pars) {
            bool matchAny = false;
            for (auto &p : patterns) {
               if (TString(c->GetName()).Contains(TRegexp(p, true))) {
                  matchAny = true;
                  break;
               }
            }
            if (!((matchAny && !invert) || (!matchAny && invert))) {
               _remPars.add(*c);
            }
         }
         _pars.remove(_remPars, true);

         auto _tmp = fr->reducedCovarianceMatrix(_pars);
         int covQualBackup = fr->covQual();
         fr->setCovarianceMatrix(_tmp);
         fr->setCovQual(covQualBackup);
         const_cast<RooArgList &>(fr->floatParsFinal())
            .remove(_remPars, true); // is this a memory leak ... should delete the remPars?
         return out;

      } else if (!get() || get<RooAbsCollection>()) {
         // filter the children .... handle special case of filtering ".vars" with "x" option too
         xRooNode out(std::make_shared<RooArgList>(), fParent);
         out.SetName(TString(GetName()) + "_reduced");
         size_t nobs = 0;
         bool notAllArgs = false;
         bool isVars = (strcmp(GetName(), ".vars") == 0);
         for (auto c : *this) {
            nobs += (c->fFolder == "!robs" || c->fFolder == "!globs");
            bool matchAny = false;
            for (auto &p : patterns) {
               if (TString(c->GetName()).Contains(TRegexp(p, true)) ||
                   (isVars && p == "x" && (c->fFolder == "!robs" || c->fFolder == "!globs") && nobs == 1)) {
                  matchAny = true;
                  break;
               }
            }
            if ((matchAny && !invert) || (!matchAny && invert)) {
               out.push_back(c);
               if (auto a = c->get<RooAbsArg>()) {
                  out.get<RooArgList>()->add(*a);
               } else {
                  notAllArgs = true;
               }
            }
         }
         if (notAllArgs) {
            out.fComp.reset();
         }
         return out;
      }
   }

   return get<RooArgList>() ? xRooNode(std::make_shared<RooArgList>(), fParent) : *this;
}

// xRooNode xRooNode::generate(bool expected) const {
//
//     auto fr = fitResult();
//     auto _fr = fr.get<RooFitResult>();
//
//     auto _pdf = (get<RooAbsPdf>()) ? std::shared_ptr<const xRooNode>(this, [](const xRooNode*){}) : parentPdf();
//     if (!_pdf) {
//         throw std::runtime_error("Could not find pdf");
//     }
//
//     std::shared_ptr<RooDataTree> t;
//
//     std::shared_ptr<RooSimultaneous> newPdf;
//     if(auto s = _pdf->get<RooSimultaneous>(); s) {
//         auto rangeName = GetRange();
//         if (rangeName) {
//             // need to reduce the RooSimultaneous until fix: https://github.com/root-project/root/issues/8231
//             std::vector<TString> chanPatterns;
//             TStringToken pattern(rangeName, ",");
//             while (pattern.NextToken()) {
//                 chanPatterns.emplace_back(pattern);
//             }
//             auto& _cat = const_cast<RooAbsCategoryLValue&>(s->indexCat());
//             newPdf = std::make_shared<RooSimultaneous>(TString::Format("%s_reduced",GetName()),"Reduced model",_cat);
//             for(auto& c : _pdf->variations()) {
//                 TString cName(c->GetName());
//                 cName = cName(cName.Index('=')+1,cName.Length());
//                 _cat.setLabel(cName);
//                 bool matchAny=false;
//                 for(auto& p : chanPatterns) {
//                     if (cName.Contains(TRegexp(p,true))) { matchAny=true; break; }
//                     if (_cat.hasRange(p) && _cat.inRange(p)) { matchAny=true; break; }
//                 }
//                 if(matchAny) {
//                     newPdf->addPdf( *c->get<RooAbsPdf>(), cName );
//                 }
//             }
//             t = std::make_shared<RooDataTree>(newPdf->GetName(),"",*newPdf);
//             RooArgSet s1(_pdf->obs().argList());
//             RooArgSet s2(_pdf->globs().argList());s1.remove(s2);
//             t->SetObservables(&s1,&s2);
//             auto _data = t->generate(_fr,expected);
//
//             xRooNode parent(_fr ? _fr->GetName() : "unknown",nullptr,xRooNode(t,*this));
//             xRooNode out(_data.first->GetName(),/*acquire(_fr)*/ _data.first,parent);
//             out.emplace_back(std::make_shared<xRooNode>(".globs",std::const_pointer_cast<RooArgSet>(_data.second),out));
//             return out;
//         }
//     }
//
//
//     std::string treeName = TString::Format("gen_%s",_pdf->GetName()).Data();
//
//     auto _frt = getObject<TTree>(treeName); // get existing frt
//
//
//     if (_frt) {
//         t = std::make_shared<RooDataTree>(_frt.get());
//     } else {
//         t = std::make_shared<RooDataTree>(treeName.c_str(),"",*_pdf->get<RooAbsPdf>());
//         RooArgSet s1(_pdf->obs().argList());
//         RooArgSet s2(_pdf->globs().argList());s1.remove(s2);
//         t->SetObservables(&s1,&s2);
//     }
//     auto _data = t->generate(_fr,expected);
//     if (!_frt) {
//         t =
//         std::make_shared<RooDataTree>(std::dynamic_pointer_cast<TTree>(const_cast<xRooNode*>(this)->acquire(t->fTree)).get());
//     }
//     xRooNode parent(_fr ? _fr->GetName() : "unknown",nullptr,xRooNode(t,*this));
//     xRooNode out(_data.first->GetName(),/*acquire(_fr)*/ _data.first,parent);
//     out.emplace_back(std::make_shared<xRooNode>(".globs",std::const_pointer_cast<RooArgSet>(_data.second),out));
//     return out;
// }

class xRooProjectedPdf : public RooProjectedPdf {
public:
   using RooProjectedPdf::RooProjectedPdf;
   double expectedEvents(const RooArgSet *nset) const override
   {
      return static_cast<RooAbsPdf *>(intpdf.absArg())->expectedEvents(nset);
   }
   ExtendMode extendMode() const override { return static_cast<RooAbsPdf *>(intpdf.absArg())->extendMode(); }
   TObject *clone(const char *newname) const override { return new xRooProjectedPdf(*this, newname); }

protected:
   double evaluate() const override
   {
      int code;
      return getProjection(&intobs, _normSet, (_normRange.Length() > 0 ? _normRange.Data() : nullptr), code)->getVal();
   }
};

double new_getPropagatedError(const RooAbsReal &f, const RooFitResult &fr, const RooArgSet &nset = {},
                              RooArgList **pars = nullptr, bool asymHi = false, bool asymLo = false)
{
   // Calling getParameters() might be costly, but necessary to get the right
   // parameters in the RooAbsReal. The RooFitResult only stores snapshots.

   // handle simple case that function is a RooRealVar
   if (auto rrv = dynamic_cast<const RooRealVar *>(&f); rrv) {
      if (auto frrrv = dynamic_cast<RooRealVar *>(fr.floatParsFinal().find(*rrv)); frrrv) {
         rrv = frrrv; // use value from fit result
      }
      if (asymHi) {
         return rrv->getErrorHi();
      } else if (asymLo) {
         return rrv->getErrorLo();
      } else {
         return rrv->getError();
      }
   }

   RooArgList *_pars = (pars) ? *pars : nullptr;

   if (!_pars) {

      RooArgSet allParamsInAbsReal;
      f.getParameters(&nset, allParamsInAbsReal);

      _pars = new RooArgList;
      for (auto *rrvFitRes : static_range_cast<RooRealVar *>(fr.floatParsFinal())) {

         auto rrvInAbsReal = static_cast<RooRealVar const *>(allParamsInAbsReal.find(*rrvFitRes));

         // Strip out parameters with zero error
         if (rrvFitRes->getError() <= std::abs(rrvFitRes->getVal()) * std::numeric_limits<double>::epsilon())
            continue;

         // Ignore parameters in the fit result that this RooAbsReal doesn't depend on
         if (!rrvInAbsReal)
            continue;

         // Checking for float equality is a bad. We check if the values are
         // negligibly far away from each other, relative to the uncertainty.
         if (std::abs(rrvInAbsReal->getVal() - rrvFitRes->getVal()) > 0.01 * rrvFitRes->getError()) {
            std::stringstream errMsg;
            errMsg << "RooAbsReal::getPropagatedError(): the parameters of the RooAbsReal don't have"
                   << " the same values as in the fit result! The logic of getPropagatedError is broken in this case.";

            throw std::runtime_error(errMsg.str());
         }

         _pars->add(*rrvInAbsReal);
      }
   }

   // Make std::vector of variations
   TVectorD F(_pars->size());

   // Create std::vector of plus,minus variations for each parameter
   TMatrixDSym V(_pars->size() == fr.floatParsFinal().size() ? fr.covarianceMatrix()
                                                             : fr.reducedCovarianceMatrix(*_pars));

   // TODO: if _pars includes pars not in fr, need to extend matrix with uncorrelated errors of those pars

   double nomVal = f.getVal(nset);

   for (std::size_t ivar = 0; ivar < _pars->size(); ivar++) {

      auto &rrv = static_cast<RooRealVar &>((*_pars)[ivar]);
      auto *frrrv = static_cast<RooRealVar *>(fr.floatParsFinal().find(rrv));

      double cenVal = rrv.getVal();
      double plusVar, minusVar, errVal;

      if (asymHi || asymLo) {
         errVal = frrrv->getErrorHi();
         rrv.setVal(cenVal + errVal);
         plusVar = f.getVal(nset);
         errVal = frrrv->getErrorLo();
         rrv.setVal(cenVal + errVal);
         minusVar = f.getVal(nset);
         if (asymHi) {
            // pick the one that moved result 'up' most
            plusVar = std::max(plusVar, minusVar);
            minusVar = 2 * nomVal - plusVar; // symmetrizes
         } else {
            // pick the one that moved result 'down' most
            minusVar = std::min(plusVar, minusVar);
            plusVar = 2 * nomVal - minusVar; // symmetrizes
         }
      } else {
         errVal = sqrt(V(ivar, ivar));
         // Make Plus variation
         rrv.setVal(cenVal + errVal);
         plusVar = f.getVal(nset);
         // Make Minus variation
         rrv.setVal(cenVal - errVal);
         minusVar = f.getVal(nset);
      }
      F[ivar] = (plusVar - minusVar) * 0.5;
      rrv.setVal(cenVal);
   }

   // Re-evaluate this RooAbsReal with the central parameters just to be
   // extra-safe that a call to `getPropagatedError()` doesn't change any state.
   // It should not be necessary because thanks to the dirty flag propagation
   // the RooAbsReal is re-evaluated anyway the next time getVal() is called.
   // Still there are imaginable corner cases where it would not be triggered,
   // for example if the user changes the RooFit operation more after the error
   // propagation.
   f.getVal(nset);

   TMatrixDSym C(_pars->size());
   std::vector<double> errVec(_pars->size());
   for (std::size_t i = 0; i < _pars->size(); i++) {
      errVec[i] = std::sqrt(V(i, i));
      for (std::size_t j = i; j < _pars->size(); j++) {
         C(i, j) = V(i, j) / std::sqrt(V(i, i) * V(j, j));
         C(j, i) = C(i, j);
      }
   }

   // Calculate error in linear approximation from variations and correlation coefficient
   double sum = F * (C * F);

   if (!pars) {
      delete _pars;
   } else {
      *pars = _pars;
   }

   return sqrt(sum);
}

class PdfWrapper : public RooAbsPdf {
public:
   // need expPdf option while RooProjectedPdf doesn't support keeping things extended
   PdfWrapper(RooAbsReal &f, RooAbsReal *coef, bool expEvMode = false, RooAbsPdf *expPdf = nullptr)
      : RooAbsPdf(Form("exp_%s", f.GetName())),
        fFunc("func", "func", this, f),
        fCoef("coef", "coef", this),
        fExpPdf("expPdf", "expPdf", this)
   {
      // don't treat pdf as extended if it has a coefficient and is RooAddPdf: RooAddPdf doesn't extend them unless no
      // coefs for any (and all are extendable)
      if (coef) {
         fCoef.setArg(*coef);
      }
      if (expPdf && expPdf->canBeExtended() && !(coef && dynamic_cast<RooAddPdf *>(expPdf))) {
         fExpPdf.setArg(*expPdf);
      } else if (auto _p = dynamic_cast<RooAbsPdf *>(&f);
                 _p && _p->canBeExtended() && !(coef && dynamic_cast<RooAddPdf *>(_p))) {
         fExpPdf.setArg(f); // using self for expectation
      }
      fExpectedEventsMode = expEvMode;
   }
   ~PdfWrapper() override{};
   PdfWrapper(const PdfWrapper &other, const char *name = nullptr)
      : RooAbsPdf(other, name),
        fFunc("func", this, other.fFunc),
        fCoef("coef", this, other.fCoef),
        fExpPdf("expPdf", this, other.fExpPdf),
        fExpectedEventsMode(other.fExpectedEventsMode)
   {
   }
   TObject *clone(const char *newname) const override { return new PdfWrapper(*this, newname); }
   bool isBinnedDistribution(const RooArgSet &obs) const override { return fFunc->isBinnedDistribution(obs); }
   std::list<double> *binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const override
   {
      return fFunc->binBoundaries(obs, xlo, xhi);
   }

   double evaluate() const override
   {
      return (fExpectedEventsMode ? 1. : fFunc) *
             ((fExpPdf.absArg()) ? static_cast<RooAbsPdf *>(fExpPdf.absArg())->expectedEvents(_normSet) : 1.) *
             (fCoef.absArg() ? fCoef : 1.);
   }

   bool selfNormalized() const override
   {
      return true;
   } // so that doesn't try to do an integral because we are passing integration onto fFunc in evaluate

   // faster than full evaluation because doesnt make the integral dependent on the full expression
   double getSimplePropagatedError(const RooFitResult &fr, const RooArgSet &nset_in) const
   {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 28, 00)
      double oo = getPropagatedError(fr, nset_in); // method was improved in 6.28 so use this instead
      if (std::isnan(oo)) {
         // may be consequence of zero uncerts
         // Calling getParameters() might be costly, but necessary to get the right
         // parameters in the RooAbsReal. The RooFitResult only stores snapshots.
         RooArgSet allParamsInAbsReal;
         getParameters(&nset_in, allParamsInAbsReal);

         RooArgList paramList;
         for (auto *rrvFitRes : static_range_cast<RooRealVar *>(fr.floatParsFinal())) {

            auto rrvInAbsReal = static_cast<RooRealVar const *>(allParamsInAbsReal.find(*rrvFitRes));

            // If this RooAbsReal is a RooRealVar in the fit result, we don't need to
            // propagate anything and can just return the error in the fit result
            if (rrvFitRes->namePtr() == namePtr())
               return rrvFitRes->getError();

            // Strip out parameters with zero error
            if (!rrvFitRes->hasError() ||
                rrvFitRes->getError() <= std::abs(rrvFitRes->getVal()) * std::numeric_limits<double>::epsilon())
               continue;

            // Ignore parameters in the fit result that this RooAbsReal doesn't depend on
            if (!rrvInAbsReal)
               continue;

            // Checking for float equality is a bad. We check if the values are
            // negligibly far away from each other, relative to the uncertainty.
            if (std::abs(rrvInAbsReal->getVal() - rrvFitRes->getVal()) > 0.01 * rrvFitRes->getError()) {
               std::stringstream errMsg;
               errMsg
                  << "RooAbsReal::getPropagatedError(): the parameters of the RooAbsReal don't have"
                  << " the same values as in the fit result! The logic of getPropagatedError is broken in this case.";

               throw std::runtime_error(errMsg.str());
            }

            paramList.add(*rrvInAbsReal);
         }
         if (paramList.empty())
            return 0.;

         std::vector<double> plusVar;
         std::vector<double> minusVar;
         plusVar.reserve(paramList.size());
         minusVar.reserve(paramList.size());

         // Create std::vector of plus,minus variations for each parameter
         TMatrixDSym V(paramList.size() == fr.floatParsFinal().size() ? fr.covarianceMatrix()
                                                                      : fr.reducedCovarianceMatrix(paramList));

         for (std::size_t ivar = 0; ivar < paramList.size(); ivar++) {

            auto &rrv = static_cast<RooRealVar &>(paramList[ivar]);

            double cenVal = rrv.getVal();
            double errVal = sqrt(V(ivar, ivar));

            // this next thing happens if the par has errors but the covariance matrix is empty
            // this only happens if the fit was dodgy, so perhaps best to not even try to recover from this
            // screwup ... hence I've commented out this fixup here and will let the errors be nan
            //            if(errVal==0) {
            //               Warning("getPropagatedError","Missing variance for %s",rrv.GetName());
            //               errVal = rrv.getError();
            //               V(ivar,ivar) = errVal*errVal;
            //            }

            // Make Plus variation
            rrv.setVal(cenVal + errVal);
            plusVar.push_back(getVal(nset_in));

            // Make Minus variation
            rrv.setVal(cenVal - errVal);
            minusVar.push_back(getVal(nset_in));
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
            // can try to recover nans ... this stopped being possible in 6.27 onwards because NaNPacker made private
            if (std::isnan(plusVar.back()) && RooNaNPacker::isNaNWithPayload(plusVar.back())) {
               plusVar.back() = -RooNaNPacker::unpackNaN(plusVar.back());
            }
            if (std::isnan(minusVar.back()) && RooNaNPacker::isNaNWithPayload(minusVar.back())) {
               minusVar.back() = -RooNaNPacker::unpackNaN(minusVar.back());
            }
#endif
            // std::cout << plusVar.back() << " and " << minusVar.back() << std::endl;

            rrv.setVal(cenVal);
         }

         // Re-evaluate this RooAbsReal with the central parameters just to be
         // extra-safe that a call to `getPropagatedError()` doesn't change any state.
         // It should not be necessary because thanks to the dirty flag propagation
         // the RooAbsReal is re-evaluated anyway the next time getVal() is called.
         // Still there are imaginable corner cases where it would not be triggered,
         // for example if the user changes the RooFit operation more after the error
         // propagation.
         getVal(nset_in);

         TMatrixDSym C(paramList.size());
         std::vector<double> errVec(paramList.size());
         for (std::size_t i = 0; i < paramList.size(); i++) {
            errVec[i] = std::sqrt(V(i, i));
            for (std::size_t j = i; j < paramList.size(); j++) {
               C(i, j) = V(i, j) / std::sqrt(V(i, i) * V(j, j));
               C(j, i) = C(i, j);
            }
         }

         // Make std::vector of variations
         TVectorD F(plusVar.size());
         for (unsigned int j = 0; j < plusVar.size(); j++) {
            F[j] = (plusVar[j] - minusVar[j]) / 2;
         }

         // Calculate error in linear approximation from variations and correlation coefficient
         double sum = F * (C * F);

         return sqrt(sum);
      }
      return oo;
#else

      // Strip out parameters with zero error
      RooArgList fpf_stripped;
      for (auto *frv : static_range_cast<RooRealVar *>(fr.floatParsFinal())) {
         if (frv->getError() > 1e-20) {
            fpf_stripped.add(*frv);
         }
      }

      // Clone self for internal use
      RooAbsReal *cloneFunc = const_cast<PdfWrapper *>(this); // (RooAbsReal *)fFunc.absArg()->cloneTree();
      // RooAbsPdf *clonePdf = dynamic_cast<RooAbsPdf *>(cloneFunc);
      RooArgSet *errorParams = cloneFunc->getObservables(fpf_stripped);

      RooArgSet *nset =
         nset_in.size() == 0 ? cloneFunc->getParameters(*errorParams) : cloneFunc->getObservables(nset_in);

      // Make list of parameter instances of cloneFunc in order of error matrix
      RooArgList paramList;
      const RooArgList &fpf = fpf_stripped;
      std::vector<int> fpf_idx;
      for (Int_t i = 0; i < fpf.size(); i++) {
         RooAbsArg *par = errorParams->find(fpf[i].GetName());
         if (par) {
            paramList.add(*par);
            fpf_idx.push_back(i);
         }
      }

      std::vector<double> plusVar, minusVar;

      // Create vector of plus,minus variations for each parameter
      TMatrixDSym V(paramList.size() == fr.floatParsFinal().size() ? fr.covarianceMatrix()
                                                                   : fr.reducedCovarianceMatrix(paramList));

      for (Int_t ivar = 0; ivar < paramList.size(); ivar++) {

         RooRealVar &rrv = (RooRealVar &)fpf[fpf_idx[ivar]];

         double cenVal = rrv.getVal();
         double errVal = sqrt(V(ivar, ivar));

         // Make Plus variation
         ((RooRealVar *)paramList.at(ivar))->setVal(cenVal + errVal);
         // plusVar.push_back((fExpectedEventsMode ? 1. : cloneFunc->getVal(nset)) *
         //                   (clonePdf ? clonePdf->expectedEvents(nset) : 1.));
         plusVar.push_back(cloneFunc->getVal(nset));

         // Make Minus variation
         ((RooRealVar *)paramList.at(ivar))->setVal(cenVal - errVal);
         // minusVar.push_back((fExpectedEventsMode ? 1. : cloneFunc->getVal(nset)) *
         //                    (clonePdf ? clonePdf->expectedEvents(nset) : 1.));
         minusVar.push_back(cloneFunc->getVal(nset));

         ((RooRealVar *)paramList.at(ivar))->setVal(cenVal);
      }

      getVal(nset); // reset state

      TMatrixDSym C(paramList.size());
      std::vector<double> errVec(paramList.size());
      for (int i = 0; i < paramList.size(); i++) {
         errVec[i] = sqrt(V(i, i));
         for (int j = i; j < paramList.size(); j++) {
            C(i, j) = V(i, j) / sqrt(V(i, i) * V(j, j));
            C(j, i) = C(i, j);
         }
      }

      // Make vector of variations
      TVectorD F(plusVar.size());
      for (unsigned int j = 0; j < plusVar.size(); j++) {
         F[j] = (plusVar[j] - minusVar[j]) / 2;
      }

      // Calculate error in linear approximation from variations and correlation coefficient
      double sum = F * (C * F);

      // delete cloneFunc;
      delete errorParams;
      delete nset;

      return sqrt(sum);
#endif
   }

private:
   RooRealProxy fFunc;
   RooRealProxy fCoef;
   RooRealProxy fExpPdf;
   bool fExpectedEventsMode = false;
};

const xRooNode *runningNode = nullptr;
void (*gOldHandlerr)(int);

void buildHistogramInterrupt(int signum)
{
   std::cout << "Got signal " << signum << std::endl;
   if (signum == SIGINT) {
      std::cout << "Keyboard interrupt while building histogram" << std::endl;
      // TODO: create a global mutex for this
      runningNode->fInterrupted = true;
   } else {
      gOldHandlerr(signum);
   }
}

void xRooNode::sterilize() const
{
   auto _doSterilize = [](RooAbsArg *obj) {
      if (!obj)
         return;
      for (int i = 0; i < obj->numCaches(); i++) {
         if (auto cache = dynamic_cast<RooObjCacheManager *>(obj->getCache(i))) {
            cache->reset();
         }
      }
      if (RooAbsPdf *p = dynamic_cast<RooAbsPdf *>(obj); p) {
         p->setNormRange(p->normRange());
      }
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      if (RooAbsReal *p = dynamic_cast<RooAbsReal *>(obj); p) {
         // need to forget about any normSet that was passed to getVal(...)
         // doesn't seem necessary in 6.28

         p->setProxyNormSet(nullptr);
         p->_lastNSet = nullptr;
      }
#endif
      obj->setValueDirty();
   };
   if (auto w = get<RooWorkspace>(); w) {
      // sterilizing all nodes
      for (auto &c : w->components()) {
         _doSterilize(c);
      }
      return;
   }
   // recursive through all clients and sterlize their normalization caches
   std::function<void(RooAbsArg *)> func;
   func = [&](RooAbsArg *a) {
      if (!a) {
         return;
      }
      _doSterilize(a); // sterilize first so that cache elements don't appear in the client list
      // safety net in case sterilizing one client deletes another one of our clients
      // monitor for change in clients list size
      // found this was only case in 6.26 (valgrind shows invalid read), in 6.28 these went away
      // might be in 6.28 the client list iterator became able to handle in-loop edits but didn't see
      // in test case that client count changed so just resterilizing if that's the case.
      size_t nClients;
      do {
         nClients = a->clients().size();
         for (auto obj : a->clients()) {
            func(dynamic_cast<RooAbsArg *>(obj));
            if (a->clients().size() != nClients) {
               break; // means sterilizing a client changed our clients, so don't trust the client iterator at this
                      // point
            }
         }
      } while (a->clients().size() != nClients);
   };
   func(get<RooAbsArg>());
}

// observables not in the axisVars are automatically projected over
xRooNode xRooNode::histo(const xRooNode &vars, const xRooNode &fr, bool content, bool errors, bool stack, bool errorsHi,
                         bool errorsLo, int nErrorToys) const
{

   if (!vars.fComp && strlen(vars.GetName())) {
      return histo(xRooNode::vars().reduced(vars.GetName()), fr, content, errors);
   }

   xRooNode out(TString::Format("%s.histo", GetName()), nullptr, *this);

   RooAbsLValue *v = nullptr;
   if (vars.empty()) {
      // does an integral
      out.fComp = std::shared_ptr<TH1>(
         BuildHistogram(nullptr, !content, errors, -1, -1, fr, errorsHi, errorsLo, nErrorToys, nullptr, !stack, false));
   } else if (vars.size() == 1) {
      v = vars.at(0)->get<RooAbsLValue>();
      out.fComp = std::shared_ptr<TH1>(
         BuildHistogram(v, !content, errors, 1, 0, fr, errorsHi, errorsLo, nErrorToys, nullptr, !stack, true));
   } else {
      throw std::runtime_error("multi-dim histo not yet supported");
   }

   return out;
}

xRooNode xRooNode::filter(const xRooNode &range) const
{
   return xRooNode(fComp, xRooNode(range.GetName(), nullptr, *this));
}

TH1 *xRooNode::BuildHistogram(RooAbsLValue *v, bool empty, bool errors, int binStart, int binEnd, const xRooNode &_fr,
                              bool errorsHi, bool errorsLo, int nErrorToys, TH1 *templateHist, bool nostack,
                              bool setInterp) const
{
   auto rar = get<RooAbsReal>();
   if (!rar)
      return nullptr;

   TObject *vv = rar;

   auto t = TH1::AddDirectoryStatus();
   TH1::AddDirectory(false);
   TH1 *h = nullptr;
   if (!v) {
      if (binStart != -1 || binEnd != -1) { // allow v to stay nullptr if doing integral (binStart=binEnd=-1)
         if (auto _ax = GetXaxis())
            v = dynamic_cast<RooAbsLValue *>(_ax->GetParent());
      } else {
         // don't need to integrate if doing a self-histogram
         v = dynamic_cast<RooRealVar *>(rar);
      }
      if (v) {
         vv = dynamic_cast<TObject *>(v);
      } else {
         // make a single-bin histogram of just this value
         h = new TH1D(rar->GetName(), rar->GetTitle(), 1, 0, 1);
         h->GetXaxis()->SetBinLabel(1, rar->GetName());
         h->GetXaxis()->SetTimeFormat(rar->GetName());
      }
   }

   auto x = dynamic_cast<RooRealVar *>(v);
   bool setTitle = false;
   if (templateHist) {
      // using template hist for the binning
      h = static_cast<TH1 *>(templateHist->Clone(rar->GetName()));
      if (h->GetListOfFunctions())
         h->GetListOfFunctions()->Clear();
      h->SetDirectory(0);
      h->SetTitle(rar->GetTitle());
      h->Reset();
   } else if (x) {
      if (x == rar) {
         // self histogram ...
         h = new TH1D(rar->GetName(), rar->GetTitle(), 1, 0, 1);
         h->Sumw2();
         h->GetXaxis()->SetBinLabel(1, rar->GetName());
         h->SetBinContent(1, rar->getVal());
         if (x->getError()) {
            h->SetBinError(1, x->getError());
            h->SetFillStyle(3005);
            h->SetFillColor(h->GetLineColor());
         }
         h->SetMaximum(x->hasMax() ? x->getMax()
                                   : (h->GetBinContent(1) + std::max(std::abs(h->GetBinContent(1) * 0.1), 50.)));
         h->SetMinimum(x->hasMin() ? x->getMin()
                                   : (h->GetBinContent(1) - std::max(std::abs(h->GetBinContent(1) * 0.1), 50.)));
         h->GetXaxis()->SetName(dynamic_cast<TObject *>(v)->GetName());
         h->SetOption("e2");
         h->SetMarkerSize(0);
         h->SetMarkerStyle(0);

         return h;
      }
      auto _ax = GetXaxis();
      TString binningName = (_ax && _ax->GetParent() == x) ? _ax->GetName() : rar->getStringAttribute("binning");
      if (binningName == "")
         binningName = rar->GetName();
      if (x->hasBinning(binningName)) {
         if (x->getBinning(binningName).isUniform()) {
            h = new TH1D(rar->GetName(), rar->GetTitle(), x->numBins(binningName) <= 0 ? 100 : x->numBins(binningName),
                         x->getMin(binningName), x->getMax(binningName));
         } else {
            h = new TH1D(rar->GetName(), rar->GetTitle(), x->numBins(binningName), x->getBinning(binningName).array());
         }
         h->GetXaxis()->SetTitle(x->getBinning(binningName).GetTitle());
         setTitle = true;
      } else if (auto _boundaries =
                    _or_func(/*rar->plotSamplingHint(*x,x->getMin(),x->getMax())*/ (std::list<double> *)(nullptr),
                             rar->binBoundaries(*x, -std::numeric_limits<double>::infinity(),
                                                std::numeric_limits<double>::infinity()));
                 _boundaries) {
         std::vector<double> _bins;
         for (auto &b : *_boundaries) {
            if (_bins.empty() || std::abs(_bins.back() - b) > 1e-5 * _bins.back())
               _bins.push_back(b);
         } // found sometimes get virtual duplicates in the binning
         h = new TH1D(rar->GetName(), rar->GetTitle(), _bins.size() - 1, &_bins[0]);
         delete _boundaries;
      } else if (!x->hasMax() || !x->hasMin()) {
         // use current value of x to estimate range with
         h = new TH1D(rar->GetName(), rar->GetTitle(), v->numBins(), x->getVal() * 0.2, x->getVal() * 5);
      } else {
         h = new TH1D(rar->GetName(), rar->GetTitle(), v->numBins(), x->getBinning().array());
      }
      h->Sumw2();
   } else if (!h) {
      h = new TH1D(rar->GetName(), rar->GetTitle(), v->numBins(rar->GetName()), 0, v->numBins(rar->GetName()));
      if (auto cat = dynamic_cast<RooAbsCategoryLValue *>(v)) {
         int i = 1;
         std::map<int, std::string> cats; // fill into a map to preserve index ordering
         for (auto &c : *cat) {
            cats[c.second] = c.first;
         }
         for (auto &[_, label] : cats) {
            h->GetXaxis()->SetBinLabel(i++, label.c_str());
         }
      }
      h->Sumw2();
   }
   if (auto o = dynamic_cast<TObject *>(v); o && !setTitle) {
      h->GetXaxis()->SetTitle(o->GetTitle());
   }
   TH1::AddDirectory(t);
   if (v) {
      if (h->GetXaxis()->IsAlphanumeric()) {
         // store the variable name in the TimeFormat property as well, b.c. alphanumeric requires axis name to be
         // "xaxis"
         h->GetXaxis()->SetTimeFormat(dynamic_cast<TObject *>(v)->GetName());
      } else {
         h->GetXaxis()->SetName(dynamic_cast<TObject *>(v)->GetName()); // WARNING: messes up display of bin labels
      }
   }

   if (auto s = styles(nullptr, false); s) {
      auto _style = s.get<TStyle>();
      static_cast<TAttLine &>(*h) = *_style;
      static_cast<TAttFill &>(*h) = *_style;
      static_cast<TAttMarker &>(*h) = *_style;
   }
   if (strlen(h->GetXaxis()->GetTitle()) == 0)
      h->GetXaxis()->SetTitle(vv->GetTitle());
   auto p = dynamic_cast<RooAbsPdf *>(rar);

   // possible speed improvement:
   //   if(auto spdf = dynamic_cast<RooRealSumPdf*>(p); spdf && spdf->canBeExtended()) {
   //      p = nullptr; // faster to evaluate sumpdf as a function not a pdf
   //   }

   if (empty && !errors) {
      return h;
   }

   // if (!empty) {

   auto _coefs = coefs();

   RooFitResult *fr = nullptr;
   if (errors) {
      // must ensure the fit result we obtain includes pars from coefficients if present
      if (_fr.get<RooFitResult>()) {
         fr = static_cast<RooFitResult *>(_fr.get<RooFitResult>()->Clone());
      } else {
         auto frn =
            (!_coefs.get() ? *this : xRooNode(RooProduct("tmp", "tmp", RooArgList(*rar, *_coefs.get<RooAbsReal>()))))
               .fitResult();
         if (strlen(_fr.GetName()))
            frn = frn.reduced(_fr.GetName());
         fr = dynamic_cast<RooFitResult *>(frn->Clone());
      }
      if (!GETDMP(fr, _finalPars)) {
         fr->setFinalParList(RooArgList());
      }

      /// Oct2022: No longer doing this because want to allow fitResult to be used to get partial error
      //        // need to add any floating parameters not included somewhere already in the fit result ...
      //        RooArgList l;
      //        for(auto& p : pars()) {
      //            auto vv = p->get<RooRealVar>();
      //            if (!vv) continue;
      //            if (vv == dynamic_cast<RooRealVar*>(v)) continue;
      //            if (vv->isConstant()) continue;
      //            if (fr->floatParsFinal().find(vv->GetName())) continue;
      //            if (fr->_constPars && fr->_constPars->find(vv->GetName())) continue;
      //            l.add(*vv);
      //        }
      //
      //        if (!l.empty()) {
      //            RooArgList l2; l2.addClone(fr->floatParsFinal());
      //            l2.addClone(l);
      //            fr->setFinalParList(l2);
      //        }

      TMatrixTSym<double> *prevCov = static_cast<TMatrixTSym<double> *>(GETDMP(fr, _VM));

      if (!prevCov || size_t(fr->covarianceMatrix().GetNcols()) < fr->floatParsFinal().size()) {
         TMatrixDSym cov(fr->floatParsFinal().size());
         if (prevCov) {
            for (int i = 0; i < prevCov->GetNcols(); i++) {
               for (int j = 0; j < prevCov->GetNrows(); j++) {
                  cov(i, j) = (*prevCov)(i, j);
               }
            }
         }
         int i = 0;
         for (auto &p2 : fr->floatParsFinal()) {
            if (!prevCov || i >= prevCov->GetNcols()) {
               cov(i, i) = pow(dynamic_cast<RooRealVar *>(p2)->getError(), 2);
            }
            i++;
         }
         int covQualBackup = fr->covQual();
         fr->setCovarianceMatrix(cov);
         fr->setCovQual(covQualBackup);
      }

      if (v) {
         // need to remove v from result as we are plotting as function of v
         if (auto _p = fr->floatParsFinal().find(dynamic_cast<TObject *>(v)->GetName()); _p) {
            RooArgList _pars = fr->floatParsFinal();
            _pars.remove(*_p, true);
            auto _tmp = fr->reducedCovarianceMatrix(_pars);
            int covQualBackup = fr->covQual();
            fr->setCovarianceMatrix(_tmp);
            fr->setCovQual(covQualBackup);
            const_cast<RooArgList &>(fr->floatParsFinal())
               .remove(*_p, true); // NOTE: I think this might be a memory leak, should delete _p after removal
         }
      }
      // finally check at least one float has errors defined (might not be cause if in prefit state)
      bool hasErrors = false;
      for (auto pp : fr->floatParsFinal()) {
         if (dynamic_cast<RooRealVar *>(pp)->hasError()) {
            hasErrors = true;
            break;
         }
      }
      if (!hasErrors) {
         errors = false;
         delete fr;
      }
   }

   RooArgSet normSet;
   if (v)
      normSet.add(*dynamic_cast<RooAbsArg *>(v));

   if (binEnd == 0)
      binEnd = h->GetNbinsX();

   bool needBinWidth = false;
   // may have MULTIPLE coefficients for the same pdf!

   if (x && (p || _coefs.get() || rar->getAttribute("density"))) {
      // pdfs of samples embedded in a sumpdf (aka have a coef) will convert their density value to a content
      needBinWidth = true;
   }

   if (auto spdf = dynamic_cast<RooRealSumPdf *>(p);
       spdf && spdf->canBeExtended() && !spdf->getFloor() && !_coefs.get()) {
      p = nullptr; // if pdf has no floor, will evaluate it as a function to allow it to be negative - evaluation should
                   // also be faster (no integral)
      // exception is if RooRealSumPdf is embedded in a RooAddPdf (detected by presence of coefs) ... then it must be
      // evaluated as a pdf technically should check parent is a RooAddPdf, because if was inside a RooRealSumPdf then
      // would be evaluated as a function!
   }

   // check if we need to do any projecting of other observables
   RooAbsReal *oldrar = nullptr;
   auto _obs = obs();

   for (auto o : _obs) {
      if (auto rr = o->get<RooRealVar>(); rr && rr->hasRange("coordRange")) {
         rr->removeRange("coordRange");                 // doesn't actually remove, just sets to -inf->+inf
         rr->setStringAttribute("coordRange", nullptr); // removes the attribute
      }
   }
   // probably should also remove any range on the x-axis variable too, if there is one
   if (auto rr = dynamic_cast<RooRealVar *>(v); rr && rr->hasRange("coordRange")) {
      rr->removeRange("coordRange");                 // doesn't actually remove, just sets to -inf->+inf
      rr->setStringAttribute("coordRange", nullptr); // removes the attribute
   }
   coords(); // loads current coordinates and populates coordRange, if any

   if (auto a = dynamic_cast<RooAbsArg *>(v))
      _obs.get<RooArgList>()->remove(*a);
   if (!_obs.get<RooArgList>()->empty()) {
      oldrar = rar;
      normSet.add(*_obs.get<RooArgList>());
      // check if any obs are restricted range
      bool hasRange = false;
      for (auto o : normSet) {
         if (auto rr = dynamic_cast<RooRealVar *>(o);
             rr && (rr->getStringAttribute("coordRange")) && strlen(rr->getStringAttribute("coordRange"))) {
            hasRange = true;
            break;
         }
      }
      if (p) {
         // need to handle special case of RooSimultaneous ... each pdf needs individually projecting over just its
         // dependent obs
         if (auto s = dynamic_cast<RooSimultaneous *>(p)) {
            auto newrar = new RooSimultaneous("projSim", "projSim", const_cast<RooAbsCategoryLValue &>(s->indexCat()));
            for (auto pdf : bins()) {
               // auto _pdf =
               // pdf->get<RooAbsPdf>()->createProjection(*pdf->get<RooAbsPdf>()->getObservables(*_obs.get<RooArgList>()));
               auto _pdf =
                  new xRooProjectedPdf(TString::Format("%s_projection", pdf->GetName()), "", *pdf->get<RooAbsPdf>(),
                                       *pdf->get<RooAbsPdf>()->getObservables(*_obs.get<RooArgList>()));
               if (hasRange) {
                  dynamic_cast<RooAbsPdf *>(_pdf)->setNormRange("coordRange");
               }
               newrar->addPdf(*_pdf, pdf->coords()[s->indexCat().GetName()]->get<RooCategory>()->getLabel());
            }
            rar = newrar;
         } else {
            rar = p->createProjection(
               *_obs.get<RooArgList>()); // TODO should use xRooProjectedPdf here too, because not fixed range and
                                         // extend behaviour of RooProjectedPdf in ROOT yet
            if (hasRange) {
               dynamic_cast<RooAbsPdf *>(rar)->setNormRange("coordRange");
            }
         }
         if (hasRange)
            p->setNormRange("coordRange"); // should get cleared when we sterilize
      } else {
         if (hasRange) {
            // commented out passing of normset so that getVal of non-pdf is always a 'raw' value (needed for raw eval
            // of RooRealSumPdf)
            rar = std::unique_ptr<RooAbsReal>{rar->createIntegral(
                                                 *_obs.get<RooArgList>(),
                                                 /*RooFit::NormSet(normSet),*/ RooFit::Range("coordRange"))}
                     .release();
         } else {
            rar =
               std::unique_ptr<RooAbsReal>{rar->createIntegral(*_obs.get<RooArgList>() /*, RooFit::NormSet(normSet)*/)}
                  .release();
         }
      }
   }

   bool scaleExpected = (p && p->canBeExtended() && !_coefs.get());
   // Note about above: if pdf has coefficients then its embedded in a RooAddPdf that has coefs defined ...
   // in this case we should *not* scale by expected, since the coefs become the scaling instead
   // we should also not build a stack for this (may be a RooRealSumPdf inside a RooAddPdf, but the
   // samples of the RooRealSumPdf wont be correctly scaled to line up with overall RooRealSumPdf
   // which will be normalized to its coefficient
   if (!nostack && p && p->canBeExtended() && _coefs.get()) {
      nostack = true;
      // if wanted to still hve a stack, would need to scale the stack subcomponents by
      // coefs-value / p_integral(raw) ... since raw p-integral will be what stack integrates to
   }

   std::unique_ptr<RooArgSet> snap(normSet.snapshot());
   TStopwatch timeIt;
   std::vector<double> lapTimes;
   bool warned = false;
   if (binStart == -1 && binEnd == -1) {
      binEnd = 1;
   }
   auto cat = (!x) ? dynamic_cast<RooAbsCategoryLValue *>(v) : nullptr;
   RooArgList *errorPars = nullptr;
   std::unique_ptr<RooAbsCollection> errorParsSnap;

   if (!v) {
      setInterp = false;
   }

   if (setInterp) {
      RooAbsArg *vvv = dynamic_cast<RooAbsArg *>(v);
      // determining if histogram should have interpolation drawing options set on it
      // need to strip namespace to discount the "HistFactory" namespace classes from all being treated as binned
      TString clNameNoNamespace = rar->ClassName();
      clNameNoNamespace = clNameNoNamespace(clNameNoNamespace.Last(':') + 1, clNameNoNamespace.Length());
      setInterp = (clNameNoNamespace.Contains("Hist") || vvv->isCategory() || rar->isBinnedDistribution(*vvv) ||
                   h->GetNbinsX() == 1 || rar->getAttribute("BinnedLikelihood") ||
                   (dynamic_cast<RooAbsRealLValue *>(vvv) &&
                    std::unique_ptr<std::list<double>>(rar->binBoundaries(*dynamic_cast<RooAbsRealLValue *>(vvv),
                                                                          -std::numeric_limits<double>::infinity(),
                                                                          std::numeric_limits<double>::infinity()))))
                     ? false
                     : true;
      if (auto d = dynamic_cast<RooHistFunc *>(rar); d && !d->isBinnedDistribution(*vvv) && h->GetNbinsX() != 1) {
         setInterp = true; // hist func is interpolated, so draw it as such
      }
      if (setInterp && !components().empty()) {
         // check if all components of dOpt are "Hist" type (CMS model support)
         // if so then don't interp;
         bool allHist = true;
         for (auto &s : components()) {
            TString _clName = s->get()->ClassName();
            _clName = _clName(_clName.Last(':') + 1, _clName.Length());
            if (!(s->get() && _clName.Contains("Hist"))) {
               allHist = false;
               break;
            }
         }
         if (allHist)
            setInterp = false;
      }
      if (setInterp) {
         h->SetOption("l"); // does linear interpolation between points
      }
   }

   if (errors) {
      // may be computing potentially asymmetric errors
      // the main histogram will be the error band, and the nominal histogram will be added as a function
      // so that it is drawn over the top of the error band
      // note that this means GetBinContent on returned histogram will return midpoint of the up and down error
      auto l = static_cast<TH1 *>(h->Clone("nominal"));
      l->SetDirectory(0);
      l->SetFillStyle(0);
      h->GetListOfFunctions()->Add(l, (setInterp) ? "lsame" : "histsame");
      h->SetOption(setInterp ? "e3" : "e2"); // default draw option E2 or E3 so error band shown .. could have used
                                             // 'EX0' to draw "classic style"
      // could take this from the 'band' style object if we create one in future?
      h->SetMarkerSize(0);
      h->SetFillStyle(3005);
      h->SetFillColor(h->GetLineColor());
   }

   if (nErrorToys > 0) {
      errors = false; // wont evaluate error on each toy, will estimate for std.dev or normiles of toys
      // need list of errorPars
      auto allPars =
         (!_coefs.get() ? *this : xRooNode(RooProduct("tmp", "tmp", RooArgList(*rar, *_coefs.get<RooAbsReal>()))))
            .pars();
      errorPars = new RooArgList; // will be in same order as appear in fr.
      for (auto a : fr->floatParsFinal()) {
         if (auto par = allPars.get<RooArgList>()->find(*a)) {
            errorPars->add(*par);
         }
      }
      errorParsSnap.reset(errorPars->snapshot());
      auto l = static_cast<TH1 *>(h->Clone("toys"));
      l->Reset(); // removes any functions
      l->SetDirectory(0);
      h->GetListOfFunctions()->Add(
         l, "histsame"); // ensures just this empty hist will be drawn, and not each individual toy

      if (errorsLo || errorsHi)
         empty = false; // must not be empty b.c. calculation of error relies on knowing nominal (see after loop)
   }

   for (int toy = 0; toy < (nErrorToys + 1); toy++) {

      TH1 *main_h = h;
      if (toy > 0) {
         h = static_cast<TH1 *>(main_h->Clone(TString::Format("toy_%d", toy)));
         h->SetDirectory(0);
         h->Reset();
         static_cast<TH1 *>(main_h->GetListOfFunctions()->FindObject("toys"))->GetListOfFunctions()->Add(h);
         // randomize the parameter values according to the fr's covariance matrix
         errorPars->assignValueOnly(fr->randomizePars());
      }

      for (int i = std::max(1, binStart); i <= std::min(h->GetNbinsX(), binEnd); i++) {
         timeIt.Start(true);
         if (x) {
            x->setVal(h->GetBinCenter(i));
         } else if (cat) {
            cat->setLabel(h->GetXaxis()->GetBinLabel(i)); // because order might not match "binning" order
         } else if (v) {
            v->setBin(i - 1);
         }
         if (x && !x->inRange("coordRange"))
            continue;

         double r = 0;
         if (!empty || toy > 0) {
            r = /*(p && p->selfNormalized())*/ rar->getVal(p ? &normSet : nullptr);
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
            if (std::isnan(r) && RooNaNPacker::isNaNWithPayload(r)) {
               r = -RooNaNPacker::unpackNaN(r);
            }
#endif
            if (r && _coefs.get()) {
               r *= _coefs.get<RooAbsReal>()->getVal(normSet);
            }
            if (needBinWidth) {
               r *= h->GetBinWidth(i);
            }
            if (scaleExpected) {
               // std::cout << r << " exp = " << p->expectedEvents(normSet) << " for normRange " << (p->normRange() ?
               // p->normRange() : "null") << std::endl; p->Print();rar->Print();
               r *= (p->expectedEvents(normSet));
            } // do in here in case dependency on var
         }
         h->SetBinContent(i, r);

         if (errors) {
            static_cast<TH1 *>(h->FindObject("nominal"))->SetBinContent(i, r); // transfer nominal to nominal hist
            double res;
            bool doAsym = (errorsHi && errorsLo);
            if (doAsym) {
               errorsHi = false;
            }
            if (p) {
               // std::cout << "computing error of :" << h->GetBinCenter(i) << std::endl;
               // //fr->floatParsFinal().Print(); fr->covarianceMatrix().Print();
               //            res = PdfWrapper((oldrar) ? *rar : *p, _coefs.get<RooAbsReal>(), !v, oldrar ? p : nullptr)
               //                     .getSimplePropagatedError(*fr, normSet);
               res = new_getPropagatedError(
                  PdfWrapper((oldrar) ? *rar : *p, _coefs.get<RooAbsReal>(), !v, oldrar ? p : nullptr), *fr, normSet,
                  &errorPars, errorsHi, errorsLo);
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
               // improved normSet invalidity checking, so assuming no longer need this in 6.28 onwards
               p->_normSet = nullptr;
#endif
            } else {
               //            res = RooProduct("errorEval", "errorEval",
               //                             RooArgList(*rar, !_coefs.get() ? RooFit::RooConst(1) :
               //                             *_coefs.get<RooAbsReal>()))
               //                     .getPropagatedError(
               //                        *fr /*, normSet*/); // should be no need to pass a normSet to a non-pdf (but
               //                        not verified this)
               res = new_getPropagatedError(
                  RooProduct("errorEval", "errorEval",
                             RooArgList(*rar, !_coefs.get() ? RooFit::RooConst(1) : *_coefs.get<RooAbsReal>())),
                  *fr, {}, &errorPars, errorsHi,
                  errorsLo); // should be no need to pass a normSet to a non-pdf (but not verified this)
               // especially important not to pass in the case we are evaluated RooRealSumPdf as a function! otherwise
               // error will be wrong
            }
            if (needBinWidth) {
               res *= h->GetBinWidth(i);
            }
            h->SetBinError(i, res);
            if (doAsym) {
               // compute Hi error
               errorsHi = true;
               errorsLo = false;
               if (p) {
                  res = new_getPropagatedError(
                     PdfWrapper((oldrar) ? *rar : *p, _coefs.get<RooAbsReal>(), !v, oldrar ? p : nullptr), *fr, normSet,
                     &errorPars, errorsHi, errorsLo);
               } else {
                  res = new_getPropagatedError(
                     RooProduct("errorEval", "errorEval",
                                RooArgList(*rar, !_coefs.get() ? RooFit::RooConst(1) : *_coefs.get<RooAbsReal>())),
                     *fr, {}, &errorPars, errorsHi, errorsLo);
               }
               if (needBinWidth) {
                  res *= h->GetBinWidth(i);
               }
               errorsLo = true;
               // lowVal = content - error, highVal = content + res
               // => band/2 = (res+error)/2 and band-mid = (2*content+res-error)/2
               h->SetBinContent(i, h->GetBinContent(i) + (res - h->GetBinError(i)) * 0.5);
               h->SetBinError(i, (res + h->GetBinError(i)) * 0.5);
            }
         }
         timeIt.Stop();
         lapTimes.push_back(timeIt.RealTime());
         double time_estimate =
            (lapTimes.size() > 1)
               ? (h->GetNbinsX() * (std::accumulate(lapTimes.begin() + 1, lapTimes.end(), 0.) / (lapTimes.size() - 1)))
               : 0.;
         if (!warned && (lapTimes.at(0) > 10 || (lapTimes.size() > 2 && time_estimate > 60.))) {
            TTimeStamp t2;
            t2.Add(time_estimate);
            Warning("BuildHistogram", "Building this histogram will take until %s", t2.AsString());
            if (errors) {
               // install interrupt handler
               runningNode = this;
               gOldHandlerr = signal(SIGINT, buildHistogramInterrupt);
            }
            warned = true;
         }
         if (fInterrupted) {
            if (errors) {
               Warning("BuildHistogram", "Skipping errors for remaining bins");
               errors = false;
            }
            fInterrupted = false;
         }
      }
      if (toy > 0) {
         h = main_h;
      }
   }
   if (gOldHandlerr) {
      signal(SIGINT, gOldHandlerr);
      gOldHandlerr = nullptr;
   }
   normSet = *snap;

   if (errorPars) {
      if (errorParsSnap)
         *errorPars = *errorParsSnap;
      delete errorPars;
   }
   if (nErrorToys) {
      // compute main histogram error bar from toys
      // if not doing asymmetric, then will display std.dev
      // otherwise will copy main to nominal and make main error bar s.t. it shows +/-1sigma vals
      if (errorsLo && errorsHi) {
         auto nomHist = static_cast<TH1 *>(h->FindObject("nominal"));
         nomHist->Add(h);
      }
      for (int i = 1; i <= h->GetNbinsX(); i++) {
         std::vector<double> vals;
         vals.reserve(nErrorToys);
         for (int j = 1; j < (nErrorToys + 1); j++) {
            vals.push_back(
               static_cast<TH1 *>(h->FindObject("toys")->FindObject(TString::Format("toy_%d", j)))->GetBinContent(i));
         }
         double upVal, downVal;
         if (errorsLo || errorsHi) {
            std::sort(vals.begin(), vals.end());
            upVal = vals.at(std::round(vals.size() * ROOT::Math::gaussian_cdf(1)));
            downVal = vals.at(std::round(vals.size() * ROOT::Math::gaussian_cdf(-1)));
            if (!errorsLo)
               downVal = 2. * h->GetBinContent(i) - upVal;
            if (!errorsHi)
               upVal = 2. * h->GetBinContent(i) - downVal;
         } else {
            double err = TMath::StdDev(vals.begin(), vals.end());
            upVal = h->GetBinContent(i) + err;
            downVal = h->GetBinContent(i) - err;
         }
         h->SetBinContent(i, (upVal + downVal) * 0.5);
         h->SetBinError(i, (upVal - downVal) * 0.5);
      }
   }

   if (oldrar) {
      std::vector<RooAbsArg *> extra;
      if (auto s = dynamic_cast<RooSimultaneous *>(rar)) {
         // need to delete all the subpdfs we created too
         for (auto _pdf : s->servers()) {
            if (dynamic_cast<RooAbsPdf *>(_pdf))
               extra.push_back(_pdf);
         }
      }
      extra.push_back(rar);
      rar = oldrar;
      xRooNode(*rar).sterilize(); // need to clear the cache of the created integral - do this before deleting things!
      for (auto a : extra)
         delete a;
   } else {
      sterilize(); // needed to forget about the normSet that was passed to getVal()
   }

   if (!p && !rar->getAttribute("density") && !needBinWidth) {
      h->GetYaxis()->SetTitle(rar->getStringAttribute("units"));
   } else if ((p && p->canBeExtended()) || (!p && needBinWidth)) {
      h->GetYaxis()->SetTitle("Events");
   } else {
      h->GetYaxis()->SetTitle("Probability Mass");
   }
   h->GetYaxis()->SetMaxDigits(3);

   if (errors) {
      delete fr;
   }

   // build a stack unless not requested
   if (!nostack) {
      // need to draw copy of hist so shown over the stack
      auto hCopy = static_cast<TH1 *>(h->Clone("copy"));
      hCopy->Reset();
      hCopy->Add(h); // use Reset and Add to clear the function list (dont clear directly as may double-delete if same
                     // object added twice)
      hCopy->SetStats(false);
      h->GetListOfFunctions()->Add(hCopy, TString(h->GetOption()) + "same");
      h->GetListOfFunctions()->Add(hCopy, "axissame"); // prevents stack covering axis
      TString dOpt = (setInterp) ? "LF2" : "";         // should become lf2 if interpolation of histogram is appropriate

      const xRooNode *rarNode = this;
      RooAbsReal *sf = nullptr;
      if (get()->InheritsFrom("RooExtendPdf")) {
         const_cast<xRooNode *>(this)->browse();
         rarNode = find(".pdf").get();
         // rar = rarNode->get<RooAbsReal>();
         sf = find(".n")->get<RooAbsReal>();
      }

      THStack *stack = new THStack("stack", TString::Format("%s;%s", rar->GetTitle(), h->GetXaxis()->GetTitle()));
      int count = 2;
      std::map<std::string, int> colorByTitle; // TODO: should fill from any existing legend
      std::set<std::string> allTitles;
      bool titleMatchName = true;
      std::map<std::string, TH1 *> histGroups;
      std::vector<TH1 *> hhs;
      std::set<TH1 *> histsWithBadTitles; // these histograms will have their titles autoFormatted

      // support for CMS model case where has single component containing many coeffs
      // will build stack by setting each coeff equal to 0 in turn, rebuilding the histogram
      // the difference from the "full" histogram will be the component
      RooArgList cms_coefs;
      if (!rarNode->components().empty()) {
         auto comps = rarNode->components()[0];
         for (auto &c : *comps) {
            if (c->fFolder == "!.coeffs")
               cms_coefs.add(*c->get<RooAbsArg>());
         }
      }
      if (!cms_coefs.empty()) {
         RooRealVar zero("zero", "", 0);
         std::shared_ptr<TH1> prevHist(static_cast<TH1 *>(h->Clone()));
         prevHist->Reset();
         prevHist->Add(h);
         for (auto c : cms_coefs) {
            // seems I have to remake the function each time, as haven't figured out what cache needs clearing?
            std::unique_ptr<RooAbsReal> f(
               dynamic_cast<RooAbsReal *>(rarNode->components()[0]->get()->Clone("tmpCopy")));
            zero.setAttribute(Form("ORIGNAME:%s", c->GetName())); // used in redirectServers to say what this replaces
            f->redirectServers(RooArgSet(zero), false, true);     // each time will replace one additional coef
            // zero.setAttribute(Form("ORIGNAME:%s",c->GetName()),false); (commented out so that on next iteration
            // will still replace all prev)
            auto hh = xRooNode(*f, *this).BuildHistogram(v);
            hh->SetName(c->GetName());
            if (sf)
               hh->Scale(sf->getVal());
            if (strlen(hh->GetTitle()) == 0) {
               hh->SetTitle(c->GetName()); // ensure all hists has titles
               histsWithBadTitles.insert(hh);
            } else if (strcmp(hh->GetName(), hh->GetTitle()) == 0) {
               histsWithBadTitles.insert(hh);
            }
            titleMatchName &= (TString(c->GetName()) == hh->GetTitle() ||
                               TString(hh->GetTitle()).BeginsWith(TString(c->GetName()) + "_"));
            std::shared_ptr<TH1> nextHist(static_cast<TH1 *>(hh->Clone()));
            hh->Add(prevHist.get(), -1.);
            hh->Scale(-1.);
            hhs.push_back(hh);
            prevHist = nextHist;
         }
      } else if (get<RooSimultaneous>()) {
         // need to create a histogram for each sample across all the channels - will rely on functionality below to
         // merge them based on titles

         for (auto &chan : bins()) {
            TString chanName(chan->GetName());
            chanName = chanName(chanName.Index("=") + 1, chanName.Length());
            auto samps = chan->mainChild();
            if (!samps)
               samps = *chan;
            for (auto &samp : samps.components()) {
               auto hh = static_cast<TH1 *>(h->Clone(samp->GetName()));
               hh->Reset();
               hh->SetTitle(samp->GetTitle());
               if (strlen(hh->GetTitle()) == 0) {
                  hh->SetTitle(samp->GetName());
                  histsWithBadTitles.insert(hh);
               } else if (strcmp(hh->GetName(), hh->GetTitle()) == 0) {
                  histsWithBadTitles.insert(hh);
               }
               hh->SetTitle(TString(hh->GetTitle())
                               .ReplaceAll(TString(chan->get()->GetName()) + "_",
                                           "")); // remove occurance of channelname_ in title (usually prefix)
               titleMatchName &= (TString(samp->GetName()) == hh->GetTitle() ||
                                  TString(hh->GetTitle()).BeginsWith(TString(samp->GetName()) + "_"));
               hh->SetBinContent(hh->GetXaxis()->FindFixBin(chanName), samp->GetContent());
               hhs.push_back(hh);
            }
         }
      } else {
         for (auto &samp : rarNode->components()) {
            auto hh = samp->BuildHistogram(
               v, empty, false /* no errors for stack*/, binStart, binEnd, _fr, false, false, 0, h, true,
               setInterp); // passing h to ensure binning is the same for all subcomponent hists
            hh->SetName(samp->GetName());
            if (sf)
               hh->Scale(sf->getVal());
            hhs.push_back(hh);
            if (strlen(hh->GetTitle()) == 0) {
               hh->SetTitle(samp->GetName()); // ensure all hists has titles
               histsWithBadTitles.insert(hh);
            } else if (strcmp(hh->GetName(), hh->GetTitle()) == 0) {
               histsWithBadTitles.insert(hh);
            }
            titleMatchName &= (TString(samp->GetName()) == hh->GetTitle() ||
                               TString(hh->GetTitle()).BeginsWith(TString(samp->GetName()) + "_"));
         }
      }

      if (!hhs.empty()) {
         for (auto &hh : hhs) {
            allTitles.insert(hh->GetTitle());
         }

         // get common prefix to strip off only if all titles match names and
         // any title is longer than 10 chars
         size_t e = std::min(allTitles.begin()->size(), allTitles.rbegin()->size());
         size_t ii = 0;
         bool goodPrefix = false;
         std::string commonSuffix;
         if (titleMatchName && hhs.size() > 1) {
            while (ii < e - 1 && allTitles.begin()->at(ii) == allTitles.rbegin()->at(ii)) {
               ii++;
               if (allTitles.begin()->at(ii) == '_' || allTitles.begin()->at(ii) == ' ')
                  goodPrefix = true;
            }

            // find common suffix if there is one .. must start with a "_"
            bool stop = false;
            while (!stop && commonSuffix.size() < size_t(e - 1)) {
               commonSuffix = allTitles.begin()->substr(allTitles.begin()->length() - commonSuffix.length() - 1);
               for (auto &tt : allTitles) {
                  if (!TString(tt).EndsWith(commonSuffix.c_str())) {
                     commonSuffix = commonSuffix.substr(1);
                     stop = true;
                     break;
                  }
               }
            }
            if (commonSuffix.find('_') == std::string::npos) {
               commonSuffix = "";
            } else {
               commonSuffix = commonSuffix.substr(commonSuffix.find('_'));
            }
         }
         if (!goodPrefix)
            ii = 0;

         // also find how many characters are needed to distinguish all entries (that dont have the same name)
         // then carry on up to first space or underscore
         size_t jj = 0;
         std::map<std::string, std::string> reducedTitles;
         while (reducedTitles.size() != allTitles.size()) {
            jj++;
            std::map<std::string, int> titlesMap;
            for (auto &s : allTitles) {
               if (reducedTitles.count(s))
                  continue;
               titlesMap[s.substr(0, jj)]++;
            }
            for (auto &s : allTitles) {
               if (titlesMap[s.substr(0, jj)] == 1 && (jj >= s.length() || s.at(jj) == ' ' || s.at(jj) == '_')) {
                  reducedTitles[s] = s.substr(0, jj);
               }
            }
         }

         // strip common prefix and suffix before adding
         for (auto ritr = hhs.rbegin(); ritr != hhs.rend(); ++ritr) { // go in reverse order
            if (!histsWithBadTitles.count((*ritr))) {
               continue;
            }
            auto _title = (hhs.size() > 5) ? reducedTitles[(*ritr)->GetTitle()] : (*ritr)->GetTitle();
            _title = _title.substr(ii < _title.size() ? ii : 0);
            if (!commonSuffix.empty() && TString(_title).EndsWith(commonSuffix.c_str()))
               _title = _title.substr(0, _title.length() - commonSuffix.length());
            (*ritr)->SetTitle(_title.c_str());
         }
      }

      for (auto &hh : hhs) {
         // automatically group hists that all have the same title
         if (histGroups.find(hh->GetTitle()) == histGroups.end()) {
            histGroups[hh->GetTitle()] = hh;
         } else {
            // add it into this group
            histGroups[hh->GetTitle()]->Add(hh);
            delete hh;
            hh = nullptr;
            continue;
         }
         auto hhMin = (hh->GetMinimum() == 0) ? hh->GetMinimum(1e-9) : hh->GetMinimum();
         if (!stack->GetHists() && h->GetMinimum() > hhMin) {
            auto newMin = hhMin - (h->GetMaximum() - hhMin) * gStyle->GetHistTopMargin();
            if (hhMin >= 0 && newMin < 0)
               newMin = hhMin * 0.99;
            //            adjustYRange(newMin, h->GetMaximum());
         }

         /*if(stack->GetHists() && stack->GetHists()->GetEntries()>0) {
             // to remove rounding effects on bin boundaries, see if binnings compatible
             auto _h1 = dynamic_cast<TH1*>(stack->GetHists()->At(0));
             if(_h1->GetNbinsX()==hh->GetNbinsX()) TODO ... finish dealing with silly rounding effects
         }*/
         TString thisOpt = TString(hh->GetOption()) == "l" ? "LF2" : ""; // need LF2 to get smooth line with fill
         // uncomment next line to blend continuous with discrete components .. get some unpleasant "poke through"
         // effects though
         // if(auto s = samp->get<RooAbsReal>(); s) thisOpt = s->isBinnedDistribution(*dynamic_cast<RooAbsArg*>(v)) ?
         // "" : "LF2";
         stack->Add(hh, thisOpt);
      }
      //      stack->SetBit(kCanDelete); // should delete its sub histograms
      h->GetListOfFunctions()->AddFirst(stack, "noclear same");
      //      stack->Draw("noclear same");
      //      h->Draw(
      //         dOpt + sOpt +
      //         "same"); // overlay again ..  if stack would cover original hist (negative components) we still see
      //         integral
      //      h->Draw("axissame"); // redraws axis

      TList *ll = stack->GetHists();
      if (ll && ll->GetEntries()) {

         // finally, ensure all hists are styled
         for (auto ho : *ll) {
            TH1 *hh = dynamic_cast<TH1 *>(ho);
            if (!hh)
               continue;
            bool createdStyle = (xRooNode(*hh, *this).styles(nullptr, false).get<TStyle>() == nullptr);

            if (createdStyle) {
               // give hist a color, that isn't the same as any other hists color
               hh->SetFillStyle(1001); // solid fill style
               bool used = false;
               do {
                  hh->SetFillColor((count++));
                  // check not already used this color
                  used = false;
                  for (auto ho2 : *ll) {
                     TH1 *hh2 = dynamic_cast<TH1 *>(ho2);
                     if (!hh2)
                        continue;
                     auto _styleNode = xRooNode(*hh2, *this).styles(hh2, false);
                     auto _style = _styleNode.get<TStyle>();
                     if (hh != hh2 && _style && _style->GetFillColor() == hh->GetFillColor()) {
                        used = true;
                        break;
                     }
                  }
               } while (used);
            }

            auto _styleNode = xRooNode(*hh, *this).styles(hh);
            if (auto _style = _styleNode.get<TStyle>()) {
               *dynamic_cast<TAttLine *>(hh) = *_style;
               *dynamic_cast<TAttFill *>(hh) = *_style;
               *dynamic_cast<TAttMarker *>(hh) = *_style;
            }
            // for stacks, fill color of white should be color 10 unless fill style is 0
            if (hh->GetFillColor() == kWhite && hh->GetFillStyle() != 0) {
               // kWhite means 'transparent' in ROOT ... should really use a FillStyle of 0 for that
               // so assume user wanted actual white, which is color 10
               hh->SetFillColor(10);
            }
         }
      }
   }

   return h;
}

double xRooNode::GetBinData(int bin, const xRooNode &data)
{
   if (data.get<RooAbsData>()) {
      // attach as a child before calling datasets(), so that is included in the list
      push_back(std::make_shared<xRooNode>(data));
   }
   auto node = datasets().find(data.GetName());
   if (data.get<RooAbsData>()) {
      // remove the child we attached
      resize(size() - 1);
   }
   if (!node)
      return std::numeric_limits<double>::quiet_NaN();
   return node->GetBinContent(bin);
}

std::vector<double> xRooNode::GetBinContents(int binStart, int binEnd) const
{
   if (fBinNumber != -1) {
      if (binStart != binEnd || !fParent) {
         throw std::runtime_error(TString::Format("%s is a bin - only has one value", GetName()));
      }
      return fParent->GetBinContents(fBinNumber, fBinNumber);
   }
   std::vector<double> out;
   if (get<RooAbsData>()) {
      auto g = BuildGraph(
         nullptr,
         (binStart != -1 ||
          binEnd != -1) /*include points for zeros unless we are asking for a single point with start=end=-1*/);
      if (!g) {
         return out;
      }
      if (binStart == binEnd && binStart == -1) {
         // integral over all bins if getting bin content -1
         double integral(0);
         for (int i = 0; i < g->GetN(); i++)
            integral += g->GetPointY(i);
         out.push_back(integral);
         delete g;
         return out;
      }
      for (int i = binStart - 1; i < g->GetN() && (binEnd == 0 || i < binEnd); i++) {
         out.push_back(g->GetPointY(i));
      }
      delete g;
      return out;
   }

   bool doIntegral = false;
   if (binStart == binEnd && binStart == -1) {
      binStart = -1;
      binEnd = -1;
      doIntegral = true;
   } // return integral if request bin -1
   auto h = BuildHistogram(nullptr, false, false, binStart, binEnd);
   if (!h) {
      throw std::runtime_error(TString::Format("%s has no content", GetName()));
   }
   if (binEnd == 0) {
      binEnd = h->GetNbinsX();
   }
   if (doIntegral) {
      double tot = 0;
      for (int i = 1; i <= h->GetNbinsX(); i++) {
         tot += h->GetBinContent(i);
      }
      out.push_back(tot);
   } else {
      for (int i = binStart; i <= binEnd; i++) {
         out.push_back(h->GetBinContent(i));
      }
   }
   delete h;
   return out;
}

xRooNode xRooNode::mainChild() const
{
   if (auto a = get<RooAbsArg>(); a) {
      // go through servers looking for 'main' thing
      for (auto &l : a->servers()) {
         if (l->getAttribute("MAIN_MEASUREMENT") || l->InheritsFrom("RooRealSumPdf") || l->InheritsFrom("RooAddPdf")) {
            return xRooNode(*l, *this);
         }
      }
      // the main child of a RooProduct is one that has the same name (/alias) as the product (except if is a bin
      // factor)
      if (a->IsA() == RooProduct::Class() && fBinNumber == -1) {
         for (auto &l : factors()) {
            if (strcmp(l->GetName(), GetName()) == 0) {
               return *l;
            }
         }
      }
   }
   return xRooNode();
}

void xRooNode::Inspect() const
{
   if (auto o = get(); o) {
      o->Inspect();
   } else {
      TNamed::Inspect();
   }
}

bool TopRightPlaceBox(TPad *p, TObject *o, double w, double h, double &xl, double &yb)
{
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
   // reinitialize collide grid because the filling depends on fUxmin and fUxmax (and ymin ymax too)
   // and these aren't filled on the first time we do the placement (they init to 0 and 1), but will be filled
   // subsequently
   for (int i = 0; i < p->fCGnx; i++) {
      for (int j = 0; j < p->fCGny; j++) {
         p->fCollideGrid[i + j * p->fCGnx] = true;
      }
   }
   p->FillCollideGrid(o);
   Int_t iw = (int)(p->fCGnx * w);
   Int_t ih = (int)(p->fCGny * h);

   Int_t nxmax = p->fCGnx - iw - 1 - p->fCGnx * p->GetRightMargin();
   Int_t nymax = p->fCGny - ih - 1 - p->fCGny * p->GetTopMargin();

   for (Int_t j = nymax; j >= 0; j--) {
      for (Int_t i = nxmax; i >= 0; i--) {
         if (p->Collide(i, j, iw, ih)) {
            continue;
         } else {
            xl = (double)(i) / (double)(p->fCGnx);
            yb = (double)(j) / (double)(p->fCGny);
            return true;
         }
      }
   }
   return false;
#else
   return p->PlaceBox(o, w, h, xl, yb, "trw");
#endif
}

TPaveText *getPave(const char *name = "labels", bool create = true, bool doPaint = false)
{
   if (auto p = dynamic_cast<TPaveText *>(gPad->GetPrimitive(name)); p) {
      if (doPaint)
         gPad->PaintModified(); //-- slows down x11 so trying to avoid
      return p;
   }
   if (!create) {
      return nullptr;
   }
   auto l = new TPaveText(gPad->GetLeftMargin() + 0.02, 1. - gPad->GetTopMargin() - 0.08, 0.6,
                          1. - gPad->GetTopMargin() - 0.08);
   l->SetBorderSize(0);
   if (l->GetTextSize() == 0)
      l->SetTextSize(gStyle->GetTitleYSize());

   l->SetBit(kCanDelete);
   // l->SetMargin(0);
   l->SetFillStyle(0);
   l->SetName(name);
   l->Draw();
   l->ConvertNDCtoPad();
   return l;
}

TLegend *getLegend(bool create = true, bool doPaint = false)
{
   if (auto p = dynamic_cast<TLegend *>(gPad->GetPrimitive("legend")); p) {
      double x;
      double y;
      double w = p->GetX2NDC() - p->GetX1NDC();
      double h = p->GetY2NDC() - p->GetY1NDC();
      if (doPaint)
         gPad->PaintModified(); //-- slows down x11 so trying to avoid
      if (TopRightPlaceBox(dynamic_cast<TPad *>(gPad), p, w, h, x, y)) {
         // squash inside the frame ..
         // std::cout << gPad->GetName() << ":" << x << " , " << y << " , " << w << " , " << h << std::endl;
         x = std::max(x, (gPad->GetLeftMargin() + 0.02));
         y = std::max(y, (gPad->GetBottomMargin() + 0.02));
         x = std::min(x, (1. - gPad->GetRightMargin() - 0.02) - w);
         y = std::min(y, (1. - gPad->GetTopMargin() - 0.02) - h);
         h = std::min(h, (1. - gPad->GetTopMargin() - 0.02) - y);
         w = std::min(w, (1. - gPad->GetRightMargin() - 0.02) - x);
         // std::cout << gPad->GetName() << ":" << x << " , " << y << " , " << h << " , " << w << std::endl;
         p->SetX1NDC(x);
         p->SetY1NDC(y);
         p->SetX2NDC(x + w);
         p->SetY2NDC(y + h);
         gPad->Modified();
      }
      return p;
   }
   // look for a parent pad called 'legend' and create it there if existing
   auto p = gPad;
   while ((p != p->GetMother()) && (p = p->GetMother())) {
      if (auto q = dynamic_cast<TVirtualPad *>(p->GetPrimitive("legend")); q) {
         q->Modified();
         p = q;
         break;
      }
   }
   auto tmpPad = gPad;
   TLegend *l = nullptr;
   if (p && strcmp(p->GetName(), "legend") == 0) {
      if (l = dynamic_cast<TLegend *>(p->GetPrimitive("legend")); l || !create)
         return l;
      p->cd();
      l = new TLegend(gPad->GetLeftMargin(), 1. - gPad->GetTopMargin(), 1. - gPad->GetRightMargin(),
                      gPad->GetBottomMargin());
      l->SetBorderSize(1); // ensure has a border
   } else {
      if (!create)
         return nullptr;
      l = new TLegend(0.6, 1. - gPad->GetTopMargin() - 0.08, 0.75, 1. - gPad->GetTopMargin() - 0.08);
      l->SetBorderSize(0);
      // legend text will be required to match y-axis
      if (l->GetTextSize() == 0) {
         l->SetTextSize(gStyle->GetTitleYSize());
         l->SetTextFont(gStyle->GetTitleFont("Y"));
      }
   }
   l->SetBit(kCanDelete);
   // l->SetMargin(0);
   l->SetFillStyle(0);
   l->SetName("legend");
   l->Draw();
   l->ConvertNDCtoPad();
   tmpPad->cd();
   return l;
}

std::string formatLegendString(const std::string &s)
{
   auto i = s.find("\n");
   if (i == std::string::npos) {
      return s;
   }
   return std::string("#splitline{") + s.substr(0, i) + "}{" + formatLegendString(s.substr(i + 1)) + "}";
}

void addLegendEntry(TObject *o, const char *title, const char *opt)
{
   auto l = getLegend();
   if (!l)
      return;
   // check for entry already existing with same title
   for (auto a : *l->GetListOfPrimitives()) {
      if (formatLegendString(title) == dynamic_cast<TLegendEntry *>(a)->GetLabel())
         return;
   }
   if (l->GetListOfPrimitives()->GetEntries() > 20)
      return; // todo: create an 'other' entry?

   l->AddEntry(o, formatLegendString(title).c_str(), opt);
   if (auto nObj = l->GetListOfPrimitives()->GetEntries(); nObj > 0) {
      // each entry takes up 0.05 ... maximum of N*(N+4) (where N is # cols) before next column
      int nn = l->GetNColumns();
      nn *= (nn + 4);
      if (nObj > 1 && (nObj % nn) == 1) {
         l->SetNColumns(l->GetNColumns() + 1);
         if (l->GetBorderSize() == 0) {
            l->SetX1NDC(l->GetX2NDC() - 0.15 * l->GetNColumns());
         }
      }
      if (l->GetBorderSize() == 0) {
         l->SetY1NDC(l->GetY2NDC() - 0.05 * gPad->GetHNDC() * std::ceil((double(nObj) / l->GetNColumns())));
      }
   }

   getLegend(); // to mark modified
}

// this exists to avoid calling update excessively because it slows down x11 ... but still
// need to call update twice if have a legend drawn in order to relocate it.
class PadRefresher {
public:
   PadRefresher(TVirtualPad *p) : fPad(p) { nExisting++; }
   ~PadRefresher()
   {
      if (fPad) {
         getLegend(false, true);
         fPad->GetCanvas()->Paint();
         fPad->GetCanvas()->Update();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
         fPad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
#endif
         fPad->cd();
      }
      nExisting--;
   }
   TVirtualPad *fPad = nullptr;
   static int nExisting;
};

int PadRefresher::nExisting = 0;

void xRooNode::Draw(Option_t *opt)
{
   // in order to catch exceptions to prevent crash of GUI, do this:
   if (gROOT->FromPopUp()) {
      gROOT->SetFromPopUp(false);
      try {
         Draw(opt);
      } catch (const std::exception &e) {
         new TGMsgBox(
            gClient->GetRoot(),
            (gROOT->GetListOfBrowsers()->At(0))
               ? dynamic_cast<TGWindow *>(static_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0))->GetBrowserImp())
               : gClient->GetRoot(),
            "Exception", e.what(),
            kMBIconExclamation); // deletes self on dismiss?
      }
      gROOT->SetFromPopUp(true);
      return;
   }

   TString sOpt2(opt);
   sOpt2.ToLower();
   if (!get() && !IsFolder() && !sOpt2.Contains("x="))
      return;

   if (auto mc = get<RooStats::ModelConfig>()) {
      xRooNode(*mc->GetPdf(), fParent).Draw(opt); // draw the pdf of the config
   } else if (auto ir = get<RooStats::HypoTestInverterResult>()) {
      xRooHypoSpace(ir).Draw(opt);
      gSystem->ProcessEvents();
      return;
   } else if (get<RooStats::HypoTestResult>()) {
      if (gPad)
         gPad->Clear();
      xRooNLLVar::xRooHypoPoint(std::dynamic_pointer_cast<RooStats::HypoTestResult>(fComp)).Draw(opt);
      {
         PadRefresher p(gPad); // refreshes the pad
      }
      gSystem->ProcessEvents();
      return;
   }

   if (sOpt2 == "pcls" && get<RooRealVar>() && fParent && fParent->get<RooAbsPdf>()) {
      // use the first selected dataset
      auto _dsets = fParent->datasets();
      // bool _drawn=false;
      TString dsetName = "";
      for (auto &d : _dsets) {
         if (d->get()->TestBit(1 << 20)) {
            dsetName = d->get()->GetName();
            break;
         }
      }
      auto hs = fParent->nll(dsetName.Data()).hypoSpace(get<RooRealVar>()->GetName());
      hs.limits("cls visualize");
      hs.SetName(TUUID().AsString());
      if (ws()) {
         ws()->import(*hs.result());
      }
      return;
   }

   if (auxFunctions.empty()) {
      // add the defaults: Ratio and Signif
      SetAuxFunction(
         "Ratio",
         [](double a, double b, double) {
            if (a == 0)
               return 0.;
            if (b == 0 && a == 0)
               return 1.;
            return a / b;
         },
         true);
      SetAuxFunction(
         "Signif",
         [](double n, double b, double sigma) {
            double t0 = 0;
            if (sigma <= 0.) {
               // use simplified expression ...
               t0 = 2. * (((n == 0) ? 0 : n * log(n / b)) - (n - b));
            } else {
               double sigma2 = sigma * sigma;
               double b_hathat = 0.5 * (b - sigma2 + sqrt(pow(b - sigma2, 2) + 4 * n * sigma2));
               // double s_hat = n - m;
               // double b_hat = m;
               t0 = 2. * (((n == 0) ? 0 : n * log(n / b_hathat)) + b_hathat - n + pow(b - b_hathat, 2) / (2. * sigma2));
            }
            if (t0 < 0)
               return 0.; // can happen from numerical precision
            return (n >= b) ? sqrt(t0) : -sqrt(t0);
         },
         false);
   }

   TString sOpt(opt);

   RooAbsLValue *v = nullptr;
   std::vector<double> xPoints;
   if (sOpt2.Contains("x=")) {
      // specifying a particular var to scan over ...
      int _idx = sOpt2.Index("x=");
      int _eidx = sOpt2.Index(';', _idx);
      TString varPart = sOpt(_idx + 2, (_eidx < 0 ? sOpt2.Length() : _eidx) - (_idx + 2));
      TString varName = varPart;
      // if varName is of form str(num,num,num) then can infer scan points
      if (auto _idx2 = varPart.Index("("); _idx2 > 0) {
         varName = varPart(0, _idx2);
         TStringToken pattern(TString(varPart(_idx2 + 1, varPart.Length() - _idx2 - 2)), ",");
         double min(0);
         double max(0);
         int nBins = 0;
         int ii = 0;
         while (pattern.NextToken()) {
            TString s = pattern;
            if (ii == 0) {
               nBins = s.Atoi();
            } else if (ii == 1) {
               min = s.Atof();
            } else if (ii == 2) {
               max = s.Atof();
            }
            ii++;
         }
         if (nBins > 100)
            nBins = 100; // limit scanning to 100 points
         if (nBins > 1) {
            for (double x = min; x <= max; x += (max - min) / (nBins - 1)) {
               xPoints.push_back(x);
            }
         } else if (nBins == 1)
            xPoints.push_back((min + max) / 2.);
      }
      v = getObject<RooAbsLValue>(varName.Data()).get();
      if (!v) {
         throw std::runtime_error(TString::Format("Could not find variable %s", varName.Data()));
      }
      if (xPoints.empty() && !obs().find(varName.Data()) &&
          dynamic_cast<RooAbsRealLValue *>(v)) { // will draw obs as regular (e.g. hist)
         double tmp = static_cast<RooAbsRealLValue *>(v)->getVal();
         for (int i = 0; i < v->numBins(GetName()); i++) {
            v->setBin(i, GetName());
            xPoints.push_back(static_cast<RooAbsRealLValue *>(v)->getVal());
         }
         static_cast<RooAbsRealLValue *>(v)->setVal(tmp);
      }
      sOpt2 = TString(sOpt2(0, _idx)) + sOpt2(_idx + 2 + varPart.Length() + 1, sOpt2.Length());
      sOpt = TString(sOpt(0, _idx)) + sOpt(_idx + 2 + varPart.Length() + 1, sOpt.Length());
   }
   TString forceNames = "";
   if (sOpt2.Contains("force")) {
      // force plots show how much NLL changes wrt to a change of variables
      if (get<RooRealVar>() && fParent && fParent->get<RooAbsPdf>()) {
         // assume want force of this parameter from the parent pdf
         TString ff = sOpt(sOpt2.Index("force"), sOpt2.Index("force") + 5);
         sOpt.ReplaceAll(ff, TString::Format("force%s", get()->GetName()));
         fParent->Draw(sOpt);
         return;
      } else if (get<RooAbsPdf>()) {
         // extract the parameter(s) to calculate force for
         forceNames = sOpt(sOpt2.Index("force") + 5, sOpt2.Length());
         sOpt = sOpt(0, sOpt2.Index("force"));
         sOpt2 = sOpt2(0, sOpt2.Index("force"));
      } else {
         Error("Draw", "Can only compute forces with PDFs");
         return; // don't throw because will cause browser to exit if done from there
      }
   }
   bool hasOverlay = sOpt2.Contains("overlay");
   TString overlayName = "";
   if (hasOverlay) {
      // whatever follows overlay is the variation name
      overlayName = sOpt(sOpt2.Index("overlay") + 7, sOpt2.Length());
      sOpt = sOpt(0, sOpt2.Index("overlay"));
      sOpt2 = sOpt2(0, sOpt2.Index("overlay"));
   }
   if (sOpt2.Contains("ratio") && !sOpt2.Contains("auxratio"))
      sOpt += "auxRatio";
   if (sOpt2.Contains("significance") && !sOpt2.Contains("auxsignif"))
      sOpt += "auxSignif";

   std::string auxPlotTitle;
   for (auto &[k, _] : auxFunctions) {
      if (sOpt.Contains(TString::Format("aux%s", k.c_str()))) {
         auxPlotTitle = k;
      }
      sOpt.ReplaceAll(TString::Format("aux%s", k.c_str()), "");
   }

   sOpt.ToLower();
   sOpt.ReplaceAll("ratio", "");
   sOpt.ReplaceAll("significance", ""); // remove old option if still given
   bool nostack = sOpt.Contains("nostack");
   sOpt.ReplaceAll("nostack", "");
   bool hasSame = sOpt.Contains("same");
   sOpt.ReplaceAll("same", "");
   bool hasGoff = sOpt.Contains("goff");
   sOpt.ReplaceAll("goff", "");
   bool hasFR = sOpt.Contains("pull") && !get<RooFitResult>();
   sOpt.ReplaceAll("pull", "");
   bool hasText = sOpt.Contains("text");
   bool hasTexte = sOpt.Contains("texte");
   bool hasErrorOpt = sOpt.Contains("e");
   sOpt.ReplaceAll("e", "");
   if (hasTexte) {
      sOpt.ReplaceAll("txt", "texte");
   } else if (hasText) {
      sOpt.ReplaceAll("txt", "text");
   }
   if (auxPlotTitle == "Signif")
      hasErrorOpt = true; // must calculate error to calculate significance
   if (hasOverlay)
      hasSame = true; // when overlaying must be putting on same

   TVirtualPad *pad = gPad;

   TH1 *hAxis = nullptr;

   auto clearPad = []() {
      gPad->Clear();
      if (gPad->GetNumber() == 0) {
         gPad->SetBottomMargin(gStyle->GetPadBottomMargin());
         gPad->SetTopMargin(gStyle->GetPadTopMargin());
         gPad->SetLeftMargin(gStyle->GetPadLeftMargin());
         gPad->SetRightMargin(gStyle->GetPadRightMargin());
      }
      // if (gPad == gPad->GetCanvas()) {
      //     gPad->GetCanvas()->SetCanvasSize( gPad->GetCanvas()->GetWindowWidth() - 4,
      //     gPad->GetCanvas()->GetWindowHeight() - 28 );
      // }
   };

   if (!hasSame || !pad) {
      if (!pad) {
         TCanvas::MakeDefCanvas();
         pad = gPad;
      }

   } else {
      // get the histogram representing the axes
      hAxis = dynamic_cast<TH1 *>(pad->GetPrimitive("axis"));
      if (!hAxis) {
         for (auto o : *pad->GetListOfPrimitives()) {
            if (hAxis = dynamic_cast<TH1 *>(o); hAxis)
               break;
         }
      }
      if (hAxis && !v) {
         v = getObject<RooAbsLValue>(hAxis->GetXaxis()->IsAlphanumeric() ? hAxis->GetXaxis()->GetTimeFormatOnly()
                                                                         : hAxis->GetXaxis()->GetName())
                .get();
      }
   }

   if (!hasSame) {
      if (gPad != gPad->GetCanvas()) {
         gPad->SetName(GetName()); // only rename the pad if its not the parent canvas
      }
      gPad->SetTitle(GetTitle());
   }

   PadRefresher padRefresh(((!hasSame || hasOverlay || PadRefresher::nExisting == 0) && !hasGoff) ? gPad : nullptr);

   auto adjustYRange = [&](double min, double max, TH1 *hh = nullptr, bool symmetrize = false) {
      if (!hh)
         hh = hAxis;
      // give max and min a buffer ...
      max += gStyle->GetHistTopMargin() * (max - min);
      if (min > 0)
         min = std::max(min * 0.9, min - gStyle->GetHistTopMargin() * (max - min));
      if (hh) {
         double ymin = hh->GetMinimum();
         double ymax = hh->GetMaximum();
         if (hh->GetMaximumStored() == -1111)
            ymax += gStyle->GetHistTopMargin() * (ymax - ymin);
         if (hh->GetMinimumStored() == -1111) {
            if (gStyle->GetHistMinimumZero() && ymax >= 0) {
               ymin = 0;
            } else if (ymin < 0) {
               ymin -= gStyle->GetHistTopMargin() * (ymax - ymin);
            } else {
               ymin = std::max(ymin * 0.9, ymin - gStyle->GetHistTopMargin() * (ymax - ymin));
            }
            // see TGLPlotPainter to complete the mimic, but we leave off here truncating @ 0 if ymax>0
         }
         // make ymax at least 3x bigger than biggest error if has error
         if (hh->GetSumw2()) {
            double smallestErrDown3 = -std::numeric_limits<double>::infinity();
            double smallestErrUp3 = std::numeric_limits<double>::infinity();
            for (int i = 1; i <= hh->GetNbinsX(); i++) {
               smallestErrDown3 = std::max(smallestErrDown3, hh->GetBinContent(i) - 3 * hh->GetBinError(i));
               smallestErrUp3 = std::min(smallestErrUp3, hh->GetBinContent(i) + 3 * hh->GetBinError(i));
            }
            max = std::max(max, smallestErrUp3);
            min = std::min(min, smallestErrDown3);
         }
         bool change = false;
         if (min < ymin) {
            ymin = min;
            change = true;
         }
         if (max > ymax) {
            ymax = max;
            change = true;
         }
         if (change) {
            // note: unfortunately when user 'unzooms' y axis it resets stored minimum to -1111, so lose range
            if (symmetrize) {
               double down = hh->GetBinContent(1) - ymin;
               double up = ymax - hh->GetBinContent(1);
               if (down > up) {
                  ymax = hh->GetBinContent(1) + down;
               } else {
                  ymin = hh->GetBinContent(1) - up;
               }
            }
            if (hh == hAxis && pad && !pad->GetLogy() && ymin > 0 && (log10(ymax) - log10(max)) >= 3) {
               // auto-log the pad
               pad->SetLogy();
            }
            if (hh == hAxis && pad && ymin == 0 && pad->GetLogy()) {
               ymin = 1e-2;
            }
            if (ymin == 0 && ymax > 10)
               ymin = 0.1; // adjust min so if user activates log scale it isn't bad
            hh->SetMinimum(ymin);
            hh->SetMaximum(ymax);
            hh->GetYaxis()->Set(1, ymin, ymax);
            hh->SetAxisRange(ymin, ymax, "Y");
         }
      }
   };

   auto graphMinMax = [](TGraphAsymmErrors *gr) {
      double ymax = -std::numeric_limits<double>::infinity();
      double ymin = std::numeric_limits<double>::infinity();
      for (int i = 0; i < gr->GetN(); i++) {
         ymax = std::max(ymax, gr->GetPointY(i) + gr->GetErrorYhigh(i));
         ymin = std::min(ymin, gr->GetPointY(i) - gr->GetErrorYlow(i));
      }
      return std::make_pair(ymin, ymax);
   };

   if (!xPoints.empty()) {
      // create a graph using GetContent
      TGraphAsymmErrors *out = new TGraphAsymmErrors;
      out->SetName(GetName());
      out->SetTitle(GetTitle());
      out->SetFillColor(out->GetLineColor());
      out->SetMarkerStyle(0);
      out->SetFillStyle(hasErrorOpt ? 3005 : 0);
      double tmp = static_cast<RooAbsRealLValue *>(v)->getVal();
      for (auto &x : xPoints) {
         static_cast<RooAbsRealLValue *>(v)->setVal(x);
         out->AddPoint(x, GetContent());
         if (hasErrorOpt) {
            out->SetPointEYlow(out->GetN() - 1, GetError());
            out->SetPointEYhigh(out->GetN() - 1, out->GetErrorYlow(out->GetN() - 1)); // symmetric error for now
         }
      }
      static_cast<RooAbsRealLValue *>(v)->setVal(tmp);
      out->GetHistogram()->GetXaxis()->SetTitle(static_cast<RooAbsRealLValue *>(v)->GetTitle());
      out->SetBit(kCanDelete);
      out->Draw(TString(hasSame ? "L" : "AL") + (hasErrorOpt ? "3" : ""));
      return;
   }

   if (hasFR) {
      // drawing the fitresult as a pull plot on a subpad, and rest of the draw elsewhere
      clearPad();
      pad->Divide(1, 2, 1e-9, 1e-9); //,0,0);
      pad->GetPad(1)->SetPad(0, 0.2, 1, 1);
      pad->GetPad(2)->SetPad(0, 0, 1, 0.2);
      TString optNoFR(opt);
      optNoFR.ReplaceAll("pull", "");
      pad->cd(1);
      Draw(optNoFR);
      pad->cd(2);
      auto _fr = fitResult();
      _fr.Draw();
      // switch into subpad
      gPad->cd(1);
      gPad->SetFillColor(kGray);
      gPad->GetFrame()->SetFillColor(kWhite);
      gPad->GetFrame()->SetFillStyle(1001);
      gPad->SetTopMargin(0);
      gPad->SetBottomMargin(0);
      gPad->SetName("pull");
      // split the pull graph into individual points -- for benefit of GUI status bar
      auto pullGraph = dynamic_cast<TGraphAsymmErrors *>(gPad->GetPrimitive("pulls"));
      if (!pullGraph) {
         Error("Draw", "Couldn't find pull graph");
         return;
      }
      pullGraph->SetName("nominal");
      TMultiGraph *mg = new TMultiGraph;
      mg->SetName("editables");

      auto scaleHist = static_cast<TH1 *>(pullGraph->FindObject("scales"));
      if (!scaleHist)
         throw std::runtime_error("Could not find scales in fit result");

      for (auto i = 0; i < pullGraph->GetN(); i++) {
         auto g = new TGraphAsymmErrors;
         g->SetName(scaleHist->GetXaxis()->GetBinLabel(i + 1));
         auto _p = dynamic_cast<RooRealVar *>(_fr.get<RooFitResult>()->floatParsFinal().find(g->GetName()));
         if (!_p) {
            Warning("Draw", "Found a non-var in the floatParsFinal list: %s - this shouldn't happen", g->GetName());
            continue;
         }
         g->SetTitle(TString::Format(
            "%s=%g +/- %s [%g,%g]", strlen(_p->GetTitle()) ? _p->GetTitle() : _p->GetName(), _p->getVal(),
            _p->hasAsymError() ? TString::Format("(%g,%g)", _p->getAsymErrorHi(), _p->getAsymErrorLo()).Data()
                               : TString::Format("%g", _p->getError()).Data(),
            scaleHist->GetBinContent(i + 1), scaleHist->GetBinError(i + 1)));
         g->SetPoint(0, pullGraph->GetPointX(i), pullGraph->GetPointY(i));
         g->SetPointEYhigh(0, pullGraph->GetErrorYhigh(i));
         g->SetPointEYlow(0, pullGraph->GetErrorYlow(i));
         g->SetEditable(true);
         g->SetHighlight(true);
         g->SetMarkerStyle(20);
         g->SetMarkerSize(0.5);
         mg->Add(g);
      }
      // gPad->GetListOfPrimitives()->Remove(pullGraph); delete pullGraph;
      mg->Draw("z0p");
      mg->SetBit(kCanDelete);
      auto _thisClone = new xRooNode("node", fComp, fParent);
      _thisClone->SetBit(kCanDelete);
      _thisClone->AppendPad();

      // ensure statusbar visible for interactive plot
      // turned this off for now ... as not needed if doing through browser, status bar already there
      //      if (gPad->GetCanvas() && !gPad->GetCanvas()->TestBit(TCanvas::kShowEventStatus)) {
      //         gPad->GetCanvas()->ToggleEventStatus();
      //      }
      gPad->AddExec("interactivePull", TString::Format("%s::Interactive_Pull()", ClassName()));

      pad->cd();
      return;
   }

   if (auto _simPdf = get<RooSimultaneous>();
       _simPdf && !(v && strcmp(_simPdf->indexCat().GetName(), dynamic_cast<TObject *>(v)->GetName()) == 0)) {
      auto _channels = bins();
      int _size = 0;
      for (auto &_v : _channels) {
         if (!_v->IsHidden())
            _size++;
      }
      if (!hasSame) {
         if (_size > 2) {
            // add a pad for the common legends
            _size++;
         }
         clearPad();
         pad->SetBorderSize(0);
         //            if (pad->GetCanvas() == pad) {
         //                if(_size>4) {
         //                    int n = _size;
         //                    Int_t w = 1, h = 1;
         //                    if (pad->GetCanvas()->GetWindowWidth() > pad->GetCanvas()->GetWindowHeight()) {
         //                        w = TMath::Ceil(TMath::Sqrt(n));
         //                        h = TMath::Floor(TMath::Sqrt(n));
         //                        if (w*h < n) w++;
         //                    } else {
         //                        h = TMath::Ceil(TMath::Sqrt(n));
         //                        w = TMath::Floor(TMath::Sqrt(n));
         //                        if (w*h < n) h++;
         //                    }
         //                    // adjust the window size to display only 4 in the window, with scroll bars
         //                    pad->GetCanvas()->SetCanvasSize( w*((pad->GetCanvas()->GetWindowWidth()-4)/2.) -16
         //                    ,h*((pad->GetCanvas()->GetWindowHeight()-28)/2.) - 16 );
         //                } else {
         //                    //pad->GetCanvas()->Set(
         //                    w*(pad->GetCanvas()->GetWindowWidth()/2.),h*(pad->GetCanvas()->GetWindowHeight()/2.))  )
         //                }
         //            }
         dynamic_cast<TPad *>(pad)->DivideSquare(_size, 1e-9, 1e-9);
         if (_size > 3) {
            auto _pad = pad->GetPad(_size); // will use as the legend pad
            _pad->SetName("legend");
            // stretch the pad all the way to the left
            _pad->SetPad(_pad->GetXlowNDC(), _pad->GetYlowNDC(), 1.0, _pad->GetYlowNDC() + _pad->GetHNDC());
            // and make all the remaining pads transparent
            int x = _size;
            while (pad->GetPad(x + 1)) {
               pad->GetPad(x + 1)->SetFillStyle(0);
               x++;
            }
         }
      }
      int i = 0;
      auto &chanVar = const_cast<RooAbsCategoryLValue &>(_simPdf->indexCat());
      // auto _idx = chanVar.getIndex();
      auto _range = GetRange();
      std::vector<TString> chanPatterns;
      if (_range && strlen(_range)) {
         TStringToken pattern(_range, ",");
         while (pattern.NextToken()) {
            chanPatterns.emplace_back(pattern);
         }
      }
      for (auto &_v : _channels) {
         if (_v->IsHidden())
            continue;
         TString s(_v->GetName());
         pad->cd(++i);
         gPad->SetName(s);
         TString cName = s(s.Index('=') + 1, s.Length());
         chanVar.setLabel(cName);
         bool inRange = chanPatterns.empty();
         for (auto &p : chanPatterns) {
            if (chanVar.inRange(p)) {
               inRange = true;
               break;
            }
         }
         if (!inRange || !_v->get<RooAbsReal>()->isSelectedComp())
            gPad->SetFillColor(kGray);
         if (!hasSame && _size > 1 && (gStyle->GetTitleFont("Y") % 10) == 3)
            gPad->SetLeftMargin(std::min(gPad->GetLeftMargin() * (1. / gPad->GetWNDC()), 0.3));
         _v->Draw(opt);
         gSystem->ProcessEvents();
      }
      pad->cd(0);
      gPad->Modified();
      // gPad->Update();
      return;
   }

   if (!get() || get<RooArgList>()) {
      // is a group draw all the submembers
      browse();
      int _size = 0;
      // int _size = _channels.size(); // size(); if (find("!.vars")) _size--;
      for (auto &_v : *this) {
         if (_v->IsHidden())
            continue;
         if (strcmp(GetName(), ".vars") == 0) {
            // auto hide obs and "1" and const var
            if (_v->get<RooAbsArg>()->getAttribute("obs"))
               continue;
            if (strcmp(_v->get()->GetName(), "1") == 0 || strcmp(_v->get()->GetName(), "ONE") == 0 ||
                TString(_v->get()->GetName()).BeginsWith("binWidth_"))
               continue;
            if (_v->get()->InheritsFrom("RooConstVar"))
               continue;
         }
         TString s(_v->GetName());
         if (s.BeginsWith(".") || s.BeginsWith("!"))
            continue;
         _size++;
      }
      if (!hasSame) {
         clearPad();
         pad->SetBorderSize(0);
         dynamic_cast<TPad *>(pad)->DivideSquare(_size, 1e-9, 1e-9);
      }
      int i = 0;
      for (auto &_v : *this) {
         if (_v->IsHidden())
            continue;
         if (strcmp(GetName(), ".vars") == 0) {
            // auto hide obs and "1" and const var
            if (_v->get<RooAbsArg>()->getAttribute("obs"))
               continue;
            if (strcmp(_v->get()->GetName(), "1") == 0 || strcmp(_v->get()->GetName(), "ONE") == 0 ||
                TString(_v->get()->GetName()).BeginsWith("binWidth_"))
               continue;
            if (_v->get()->InheritsFrom("RooConstVar"))
               continue;
         }
         TString s(_v->GetName());
         if (s.BeginsWith(".") || s.BeginsWith("!"))
            continue;
         pad->cd(++i);
         gPad->SetName(s);
         if (!hasSame && _size > 1 && (gStyle->GetTitleFont("Y") % 10) == 3)
            gPad->SetLeftMargin(std::min(gPad->GetLeftMargin() * (1. / gPad->GetWNDC()), 0.3));
         _v->Draw(opt);
         // pad->Modified();//pad->Update();
         gSystem->ProcessEvents();
      }
      pad->cd(0);
      gPad->Modified();
      // gPad->Update();
      return;
   }

   if (get()->InheritsFrom("RooProdPdf")) {
      // draw the main pdf, if there is one...
      auto _mainChild = mainChild();
      if (_mainChild) {
         _mainChild.Draw(opt);
         gPad->SetName(GetName());
         return;
      }
   }

   if (auto fr = get<RooFitResult>(); fr) {
      if (sOpt.Contains("corr")) {
         // do correlation matrix

         auto hist = fr->correlationHist(fr->GetName());
         hist->SetTitle(fr->GetTitle());
         hist->SetBit(kCanDelete);
         hist->Scale(100);
         hist->SetStats(false);
         hist->SetDirectory(nullptr);
         TString b(gStyle->GetPaintTextFormat());
         gStyle->SetPaintTextFormat(".1f");
         hist->GetXaxis()->SetTickSize(0);
         hist->GetYaxis()->SetTickSize(0);
         hist->SetMinimum(-100);
         hist->Draw(sOpt);
         gStyle->SetPaintTextFormat(b);
         gPad->SetGrid(1, 1);
         gPad->SetLogy(0);
         gPad->SetLogx(0);
         return;
      }

      if (sOpt.Contains("brakdown")) { // e will have been removed above

         // breakdown is quadrature difference between total error and conditional error
         // group by 'group' attribute

         std::string poiName;
         if (sOpt.Contains("brakdown:")) {
            TString sOpt3(opt);
            poiName = sOpt3(sOpt3.Index("breakdown:") + 10, sOpt3.Length());
         } else {
            std::unique_ptr<RooAbsCollection> _poi(fr->floatParsFinal().selectByAttrib("poi", true));
            if (_poi->empty()) {
               throw std::runtime_error("No floating poi in the fit");
            } else if (_poi->size() != 1) {
               throw std::runtime_error("Multiple poi in the fit");
            }
            poiName = _poi->first()->GetName();
         }
         RooRealVar *poi = dynamic_cast<RooRealVar *>(fr->floatParsFinal().find(poiName.c_str()));
         if (!poi) {
            throw std::runtime_error(TString::Format("Cannot find parameter %s", poiName.c_str()));
         }
         std::set<std::string> groups;
         for (auto p : fr->floatParsFinal()) {
            if (p == poi) {
               continue;
            } else if (p->getStringAttribute("group")) {
               groups.insert(p->getStringAttribute("group"));
            } else {
               groups.insert(p->GetTitle());
            }
         }

         auto roundedVal = xRooFit::matchPrecision(std::pair(poi->getVal(), poi->getError()));

         TPie *pie = new TPie(TString::Format("breakdown:%s", poi->GetName()),
                              TString::Format("%s: %g #pm %g", poi->GetTitle(), roundedVal.first, roundedVal.second),
                              groups.size() + 1);

         // for display of errors will go to one extra dp ...
         roundedVal.second *= .1;

         // do breakdown by removing parameters in blocks according to groups and seeing impact on variance
         // this will give the correct sum but will be order-dependent if there are correlations between
         // groups. therefore we will stick with group-by-group
         // RooArgList pars(fr->floatParsFinal()); // pars to not condition on
         // double variance = pow(dynamic_cast<RooRealVar*>(poi)->getError(),2);
         int i = 0;
         for (auto group : groups) {
            RooArgList pars(fr->floatParsFinal()); // pars to not condition on
            double variance = pow(dynamic_cast<RooRealVar *>(poi)->getError(), 2);
            for (auto p : fr->floatParsFinal()) {
               if (p == poi) {
                  continue;
               } else if ((p->getStringAttribute("group") && group == p->getStringAttribute("group")) ||
                          (!p->getStringAttribute("group") && group == p->GetTitle())) {
                  // conditioning on this parameter ... remove from pars list
                  pars.remove(*p);
               }
            }
            int idx = pars.index(poiName.c_str());
            double reducedVar = fr->conditionalCovarianceMatrix(pars)(idx, idx);
            if (reducedVar > variance) {
               Warning("Draw", "breakdown group %s variance bigger than preceding?", group.c_str());
               pie->SetEntryVal(i, 0);
               pie->SetEntryLabel(i, TString::Format("%s: NaN", group.c_str()));
            } else {
               pie->SetEntryVal(i, variance - reducedVar);
               auto r = xRooFit::matchPrecision(
                  std::pair(sqrt(variance - reducedVar), roundedVal.second)); // r.first will be the rounded error
               if (r.first > 0) {
                  pie->SetEntryLabel(i, TString::Format("%s: %g", group.c_str(), r.first));
               } else {
                  pie->SetEntryLabel(i, group.c_str()); // suppress labels for negligible errors.
               }
            }
            pie->SetEntryFillColor(i, TColor::GetColorPalette(TColor::GetNumberOfColors() * i / pie->GetEntries()));
            // variance = reducedVar;
            i++;
         }
         // remaining variance is statistical=
         double variance = fr->conditionalCovarianceMatrix(*poi)(0, 0);
         auto r =
            xRooFit::matchPrecision(std::pair(sqrt(variance), roundedVal.second)); // r.first will be the rounded error
         pie->SetEntryVal(i, variance);
         pie->SetEntryLabel(i, TString::Format("stat: %g", r.first));
         pie->SetEntryFillColor(i, TColor::GetColorPalette(TColor::GetNumberOfColors() * i / pie->GetEntries()));
         pie->SetBit(kCanDelete);
         pie->SetRadius(0.17);
         pie->SetTextSize(gStyle->GetTitleYSize());
         pie->Draw("NOL");
         return;
      }

      // plot pull or impact
      TGraphAsymmErrors *out = new TGraphAsymmErrors;
      out->SetName(TString::Format("%s_pull", fr->GetName()));
      out->SetTitle("Fit Result Pulls");
      std::vector<TString> graphLabels;
      TGraphAsymmErrors *ugraph = new TGraphAsymmErrors;
      ugraph->SetName(TString::Format("%s_pull_unconstrained", fr->GetName()));
      ugraph->SetTitle("Fit Result Pulls");
      std::vector<TString> ugraphLabels;
      std::map<std::string, double> scale;
      std::map<std::string, double> offset;
      for (auto &p : fr->floatParsFinal()) {
         auto _v = dynamic_cast<RooRealVar *>(p);
         if (!_v)
            continue;

         if (std::isnan(_v->getErrorHi()) || std::isnan(_v->getErrorLo())) {
            Warning("Draw", "%s error is invalid", _v->GetName());
         }

         // need to get constraint mean and error parameters ....
         // look for normal gaussian and poisson cases
         double prefitError = 0;
         double prefitVal = 0;
         double customScale = 0;
         if (auto ip =
                dynamic_cast<RooRealVar *>(fr->floatParsInit().find(p->GetName()))) { // handles if no prefit available
            prefitError = ip->getError();
            prefitVal = ip->getVal();
         };

         std::shared_ptr<xRooNode> pConstr;
         if (fParent && fParent->getObject<RooRealVar>(p->GetName())) {
            auto _vv = fParent->getObject<RooRealVar>(p->GetName());
            if (_vv->hasRange("pullScale")) {
               customScale = (_vv->getMax("pullScale") - _vv->getMin("pullScale")) / 2.;
            }
            auto _constr = xRooNode(_vv, *this).constraints();
            for (auto &c : _constr) {
               if (c->get<RooPoisson>() || c->get<RooGaussian>()) {
                  // require parameter to be a direct server of the constraint pdf to count if its a gaussian
                  bool isServer = true;
                  if (c->get<RooGaussian>()) {
                     isServer = false;
                     for (auto s : c->get<RooAbsArg>()->servers()) {
                        if (strcmp(s->GetName(), p->GetName()) == 0) {
                           isServer = true;
                           break;
                        }
                     }
                  }
                  if (isServer) {
                     pConstr = c;
                     break;
                  }
               }
            }
         }
         if (pConstr) {

            // there will be 3 deps, one will be this par, the other two are the mean and error (or error^2 in case of
            // poisson

            // std::cout << p->GetName() << " extracted " << prefitVal << " " << prefitError << " from ";
            // pConstr->deps().Print();
            pConstr->browse();
            if (pConstr->get<RooPoisson>() && pConstr->find(".x")) {
               std::string xName = pConstr->find(".x")->get()->GetName();
               prefitVal = pConstr->find(".x")->get<RooAbsReal>()->getVal();
               for (auto &_d : pConstr->vars()) {
                  if (strcmp(p->GetName(), _d->get()->GetName()) == 0)
                     continue;
                  if (xName == _d->get()->GetName())
                     continue;
                  if (_d->get<RooAbsReal>()->getVal())
                     prefitError = _d->get<RooAbsReal>()->getVal();
               }
               // prefitVal will be the global observable value, need to divide that by tau
               prefitVal /= prefitError;
               // prefiterror will be tau ... need 1/sqrt(tau) for error
               prefitError = 1. / sqrt(prefitError);
            } else if (auto _g = pConstr->get<RooGaussian>(); _g) {
               prefitError =
                  (pConstr->find(".sigma")) ? pConstr->find(".sigma")->get<RooAbsReal>()->getVal() : prefitError;
               prefitVal =
                  (pConstr->find(".x")) ? pConstr->find(".x")->get<RooAbsReal>()->getVal() : 0; // usually the globs
               if (pConstr->find(".x") &&
                   strcmp(p->GetName(), pConstr->find(".x")->get<RooAbsReal>()->GetName()) == 0) {
                  // hybrid construction case,
                  prefitVal = pConstr->find(".mean")->get<RooAbsReal>()->getVal();
               }
            }

            if (customScale)
               prefitError = customScale;
            if (prefitError == 0) {
               Warning("Draw", "failed to determine prefit error of %s, using post-fit error", p->GetName());
               prefitError = _v->getError();
            }
            out->SetPoint(out->GetN(), out->GetN(), (_v->getVal() - prefitVal) / prefitError);
            out->SetPointError(out->GetN() - 1, 0, 0, (-_v->getErrorLo()) / prefitError,
                               (_v->getErrorHi()) / prefitError);
            graphLabels.push_back(p->GetName());
            scale[p->GetName()] = prefitError;
            offset[p->GetName()] = prefitVal;
         } else if (!fParent) {
            // no parent to determine constraints from ... prefitError=0 will be the unconstrained ones
            if (customScale)
               prefitError = customScale;
            if (prefitError == 0) {
               // uses range of var
               prefitError = (std::max({_v->getMax() - _v->getVal(), _v->getVal() - _v->getMin(), 4.}) / 4);
               ugraph->SetPoint(ugraph->GetN(), ugraph->GetN(), (_v->getVal() - prefitVal) / prefitError);
               ugraph->SetPointError(ugraph->GetN() - 1, 0, 0, (-_v->getErrorLo()) / prefitError,
                                     (_v->getErrorHi()) / prefitError);
               ugraphLabels.push_back(p->GetName());
            } else {
               out->SetPoint(out->GetN(), out->GetN(), (_v->getVal() - prefitVal) / prefitError);
               out->SetPointError(out->GetN() - 1, 0, 0, (-_v->getErrorLo()) / prefitError,
                                  (_v->getErrorHi()) / prefitError);
               graphLabels.push_back(p->GetName());
            }
            scale[p->GetName()] = prefitError;
            offset[p->GetName()] = prefitVal;

         } else {
            // unconstrained (or at least couldn't determine constraint) ... use par range if no prefit error
            if (customScale)
               prefitError = customScale;
            if (prefitError == 0) {
               prefitError = (std::max({_v->getMax() - _v->getVal(), _v->getVal() - _v->getMin(), 4.}) / 4);
            }
            ugraph->SetPoint(ugraph->GetN(), ugraph->GetN(), (_v->getVal() - prefitVal) / prefitError);
            ugraph->SetPointError(ugraph->GetN() - 1, 0, 0, (-_v->getErrorLo()) / prefitError,
                                  (_v->getErrorHi()) / prefitError);
            ugraphLabels.push_back(p->GetName());
            scale[p->GetName()] = prefitError;
            offset[p->GetName()] = prefitVal;
         }
      }
      auto graph = out;

      // append ugraph points to end of graph
      for (int i = 0; i < ugraph->GetN(); i++)
         ugraph->SetPointX(i, i + graph->GetN());
      int nUnconstrained = ugraph->GetN();
      TList tmpList;
      tmpList.SetName("tmpList");
      tmpList.Add(ugraph);
      graph->Merge(&tmpList);
      tmpList.RemoveAll();
      delete ugraph;
      for (auto &l : ugraphLabels) {
         graphLabels.push_back(l);
      }

      graph->SetBit(kCanDelete);
      graph->SetMarkerStyle(20);
      graph->SetMarkerSize(0.5);

      graph->SetMaximum(4);
      graph->SetMinimum(-4);

      bool doHorizontal =
         (!sOpt.Contains("impact") && sOpt.Contains("v")) || (sOpt.Contains("impact") && !sOpt.Contains("himpact"));

      std::vector<std::pair<double, std::string>> covariances;
      /*double poiError = 0;*/ std::string poiName;
      double maxImpact = 0;
      if (sOpt.Contains("impact")) {
         if (sOpt.Contains("impact:")) {
            TString sOpt3(opt);
            poiName = sOpt3(sOpt3.Index("impact:") + 7, sOpt3.Length());
         } else {
            std::unique_ptr<RooAbsCollection> _poi(fr->floatParsFinal().selectByAttrib("poi", true));
            if (_poi->empty()) {
               throw std::runtime_error("No floating poi in the fit");
            } else if (_poi->size() != 1) {
               throw std::runtime_error("Multiple poi in the fit");
            }
            poiName = _poi->first()->GetName();
         }
         RooAbsArg *poi = fr->floatParsFinal().find(poiName.c_str());
         if (!poi) {
            throw std::runtime_error(TString::Format("Cannot find parameter %s", poiName.c_str()));
         }
         size_t poiIdx = fr->floatParsFinal().index(*poi);
         // put parameters in order of impact on the poi

         // impact is regression coefficient * npError
         // relevant regression coefficient is cov / (npVariance)
         // i.e. DeltaX/sigmaX = [cov(X,Y)/(sigmaXsigmaY)]DeltaY/sigmaY
         // ... DeltaX = [cov(X,Y)/(sigmaY^2)]DeltaY
         // if DeltaY is just sigmaY then DeltaX = cov(X,Y)/sigmaY

         for (auto &label : graphLabels) {
            covariances.emplace_back(fr->covarianceMatrix()(poiIdx, fr->floatParsFinal().index(label)) /
                                        dynamic_cast<RooRealVar *>(fr->floatParsFinal().find(label))->getError(),
                                     label);
         }
         std::sort(covariances.begin(), covariances.end(),
                   [&](std::pair<double, std::string> i, std::pair<double, std::string> j) {
                      return doHorizontal ? (std::abs(i.first) < std::abs(j.first))
                                          : (std::abs(i.first) > std::abs(j.first));
                   });

         TGraphAsymmErrors sortedGraph;
         std::vector<TString> sortedLabels;
         maxImpact = (doHorizontal) ? covariances.back().first
                                    : covariances.front().first; // note: max impact is likely to be self variance
         for (auto &c : covariances) {
            if (c.second == poi->GetName()) {
               // poiError = sqrt(c.first);
               continue; // skip self
            }
            c.first *= 4. / (maxImpact * 1.2);
            sortedLabels.push_back(c.second);
            size_t i = 0;
            for (; i < graphLabels.size(); i++) {
               if (graphLabels[i] == c.second) {
                  break;
               }
            }
            sortedGraph.AddPoint(sortedGraph.GetN(), graph->GetPointY(i));
            sortedGraph.SetPointError(sortedGraph.GetN() - 1, 0, 0, graph->GetErrorYlow(i), graph->GetErrorYhigh(i));
         }
         graph->Set(0);
         TList tmpList2;
         tmpList2.SetName("tmpList");
         tmpList2.Add(&sortedGraph);
         graph->Merge(&tmpList2);
         tmpList2.RemoveAll();
         graphLabels = sortedLabels;
         graph->SetTitle("Fit Result Impact");
      }

      // create a framing histogram
      TH2D *hist;
      if (doHorizontal) {
         hist = new TH2D(GetName(), fr->GetTitle(), 100, -4, 4, std::max(graph->GetN(), 1), -0.5,
                         std::max(graph->GetN(), 1) - 0.5);
         int i = 1;
         for (auto &l : graphLabels) {
            hist->GetYaxis()->SetBinLabel(i++, l);
         }
         if (!graphLabels.empty())
            hist->GetYaxis()->LabelsOption("v");
         hist->GetXaxis()->SetTitle("(#hat{#theta}-#theta_{i})/#sigma_{i}");
      } else {
         hist = new TH2D(GetName(), fr->GetTitle(), std::max(graph->GetN(), 1), -0.5, std::max(graph->GetN(), 1) - 0.5,
                         100, -4, 4);
         int i = 1;
         for (auto &l : graphLabels) {
            hist->GetXaxis()->SetBinLabel(i++, l);
         }
         if (!graphLabels.empty())
            hist->GetXaxis()->LabelsOption("v");
         hist->GetYaxis()->SetNdivisions(8, 0, 0);
         hist->GetYaxis()->SetTitle("(#hat{#theta}-#theta_{i})/#sigma_{i}");
      }
      hist->SetStats(false);
      hist->SetDirectory(nullptr);
      hist->SetBit(kCanDelete);
      auto histCopy = dynamic_cast<TH1 *>(hist->Clone(".axis"));
      histCopy->SetDirectory(nullptr);
      histCopy->SetBit(kCanDelete);
      auto _axis = (doHorizontal ? histCopy->GetYaxis() : histCopy->GetXaxis());

      /*
            auto t = TH1::AddDirectoryStatus();
            TH1::AddDirectory(false);
            auto hist = new TH1F(TString::Format(".%s_pullFrame", GetName()), fr->GetTitle(), std::max(graph->GetN(),
         1), -0.5, std::max(graph->GetN(), 1) - 0.5); hist->SetStats(false); TH1::AddDirectory(t);
            hist->SetBit(kCanDelete);
            */
      //      auto hist = graph->GetHistogram();
      graph->GetHistogram()->GetXaxis()->Set(std::max(graph->GetN(), 1), -0.5, std::max(graph->GetN(), 1) - 0.5);
      for (int ii = 1; ii <= _axis->GetNbins(); ii++) {
         graph->GetHistogram()->GetXaxis()->SetBinLabel(ii, _axis->GetBinLabel(ii));
      }
      //      int i = 1;
      //      for (auto &l : graphLabels) {
      //         hist->GetXaxis()->SetBinLabel(i++, l);
      //      }
      //      hist->SetMaximum(4);
      //      hist->SetMinimum(-4);
      //      if (graph->GetN())
      //         hist->GetXaxis()->LabelsOption("v");
      //      hist->GetYaxis()->SetNdivisions(8, 0, 0);
      //      hist->GetYaxis()->SetTitle("(#hat{#theta}-#theta_{i})/#sigma_{i}");
      clearPad();
      // create a new pad because adjust the margins ...
      auto oldPad = gPad;
      gPad->Divide(1, 1, 1e-9, 1e-9);
      gPad->cd(1);

      if (doHorizontal) {
         gPad->SetLeftMargin(0.4);
      } else {
         gPad->SetBottomMargin(0.4);
      }

      auto pNamesHist = dynamic_cast<TH1F *>(graph->GetHistogram()->Clone("scales")); // used by interactive "pull" plot
      pNamesHist->Sumw2();
      pNamesHist->SetDirectory(nullptr);

      for (int ii = 1; ii <= graph->GetN(); ii++) { // use graph->GetN() to protect against the 0 pars case
         auto _p = fr->floatParsFinal().find(_axis->GetBinLabel(ii));
         pNamesHist->SetBinContent(ii, offset[_p->GetName()]);
         pNamesHist->SetBinError(ii, scale[_p->GetName()]);
         _axis->SetBinLabel(ii, strlen(_p->GetTitle()) ? _p->GetTitle() : _p->GetName());
      }

      // hist->Draw(); -- now just draw the graph

      if (!sOpt.Contains("impact")) {
         for (int ii = 2; ii >= 1; ii--) {
            auto pullBox = new TGraphErrors;
            pullBox->SetName(TString::Format("%dsigmaBand", ii));
            pullBox->SetBit(kCanDelete);
            pullBox->SetPoint(0, (doHorizontal) ? -ii : -0.5, (doHorizontal) ? -0.5 : 0);
            pullBox->SetPoint(1, (doHorizontal) ? ii : (_axis->GetNbins() - 0.5 - nUnconstrained),
                              (doHorizontal) ? -0.5 : 0);
            pullBox->SetPointError(0, 0, (doHorizontal) ? (_axis->GetNbins() - nUnconstrained) : ii);
            pullBox->SetPointError(1, 0, (doHorizontal) ? (_axis->GetNbins() - nUnconstrained) : ii);
            pullBox->SetFillColor((ii == 2) ? kYellow : kGreen);
            hist->GetListOfFunctions()->Add(pullBox, "3"); // pullBox->Draw("3");
         }
         auto pullLine = new TGraph;
         pullLine->SetName("0sigmaLine");
         pullLine->SetBit(kCanDelete);
         pullLine->SetPoint(0, -0.5, 0);
         pullLine->SetPoint(1, _axis->GetNbins() - 0.5, 0);
         pullLine->SetLineStyle(2);
         pullLine->SetEditable(false);
         hist->GetListOfFunctions()->Add(pullLine, "l"); // pullLine->Draw("l");

         // also draw vertical line separating constrained from unconstrained, if necessary
         if (nUnconstrained > 0) {
            pullLine = new TGraph;
            pullLine->SetName("dividerLine");
            pullLine->SetBit(kCanDelete);
            pullLine->SetPoint(0, graph->GetN() - 0.5 - nUnconstrained, -100);
            pullLine->SetPoint(1, graph->GetN() - 0.5 - nUnconstrained, 100);
            pullLine->SetLineStyle(2);
            pullLine->SetEditable(false);
            hist->GetListOfFunctions()->Add(pullLine, "l"); // pullLine->Draw("l");
         }

         // and draw a pave with fr status info
         TPaveText *pave =
            new TPaveText(gPad->GetLeftMargin(), 1. - gPad->GetTopMargin(), 1. - gPad->GetRightMargin(), 0.98, "NDCNB");
         pave->SetFillStyle(0);
         pave->SetBorderSize(0);
         pave->SetMargin(0.);
         pave->SetName("status");
         pave->SetTextAlign(31);
         pave->AddText(TString::Format("minNLL: %g  edm: %g", fr->minNll(), fr->edm()));
         std::string covQualTxt;
         switch (fr->covQual()) {
         case -1: covQualTxt = "Unknown"; break;
         case 0: covQualTxt = "Not calculated"; break;
         case 1: covQualTxt = "Approximate"; break;
         case 2: covQualTxt = "Forced Positive-Definite"; break;
         case 3: covQualTxt = "Accurate"; break;
         }
         pave->AddText(TString::Format("Cov. Quality: %d (%s)", fr->covQual(), covQualTxt.c_str()))
            ->SetTextColor((fr->covQual() == 3) ? kBlack : kRed);

         std::string statusCodes;
         for (unsigned int i = 0; i < fr->numStatusHistory(); i++) {
            statusCodes += TString::Format(" %s = %d", fr->statusLabelHistory(i), fr->statusCodeHistory(i));
         }
         pave->AddText(statusCodes.c_str())->SetTextColor(fr->status() == 0 ? kBlack : kRed);

         hist->GetListOfFunctions()->Add(pave);

      } else {
         gPad->SetTicks(0, 0); // ensure mirrored ticks aren't drawn in this pad

         if (doHorizontal) {
            // ensure canvas height big enough
            if (int(gPad->GetCanvas()->GetWh()) < pNamesHist->GetNbinsX() * 15) {
               gPad->GetCanvas()->SetCanvasSize(gPad->GetCanvas()->GetWw(), pNamesHist->GetNbinsX() * 15);
            }
         }

         double factor = 475. / gPad->GetCanvas()->GetWh();      // Wh is the full canvas height, not window height
         gPad->SetTopMargin(gStyle->GetPadTopMargin() * factor); // fixed margin height
         gPad->SetBottomMargin(gStyle->GetPadBottomMargin() * factor); // fixed margin height

         TGaxis *axis =
            new TGaxis(_axis->GetXmin(), -4, _axis->GetXmin(), 4, -1.2 * maxImpact, 1.2 * maxImpact, 510, "-S");

         if (doHorizontal) {
            //            _axis->SetLabelSize(
            //               (_axis->GetLabelFont() % 10 > 2)
            //                  ? (20 / factor)
            //                  : ((gPad->AbsPixeltoY(0) - gPad->AbsPixeltoY(20 / factor)) / (gPad->GetY2() -
            //                  gPad->GetY1())));
            //            histCopy->GetXaxis()->SetTickLength(histCopy->GetXaxis()->GetTickLength() * factor);
            //            hist->GetXaxis()->SetTickLength(hist->GetXaxis()->GetTickLength() * factor);
            //            histCopy->GetYaxis()->SetTickLength(histCopy->GetYaxis()->GetTickLength() * factor);
            //            hist->GetYaxis()->SetTickLength(hist->GetYaxis()->GetTickLength() * factor);
            //            histCopy->GetXaxis()->SetTitleOffset(histCopy->GetXaxis()->GetTitleOffset() * factor);
            //            histCopy->GetXaxis()->SetLabelOffset(histCopy->GetXaxis()->GetLabelOffset() * factor);
            //            hist->GetXaxis()->SetTitleOffset(hist->GetXaxis()->GetTitleOffset() * factor);
            //            hist->GetXaxis()->SetLabelOffset(hist->GetXaxis()->GetLabelOffset() * factor);
            //            histCopy->GetXaxis()->SetTitleOffset(histCopy->GetXaxis()->GetTitleOffset() * factor);
            //            histCopy->GetXaxis()->SetLabelOffset(histCopy->GetXaxis()->GetLabelOffset() * factor);
         }
         // copy attributes from TAxis to TGaxis
         axis->ImportAxisAttributes((doHorizontal) ? histCopy->GetXaxis() : histCopy->GetYaxis());
         axis->SetTitle(TString::Format("#Delta %s", fr->floatParsFinal().find(poiName.c_str())->GetTitle()));

         // create impact bar charts
         for (int tt = 0; tt < 2; tt++) {
            auto impact = static_cast<TH1 *>(
               graph->GetHistogram()->Clone(TString::Format("%s_impact+", tt == 0 ? "prefit" : "postfit")));
            impact->SetDirectory(nullptr);
            impact->GetYaxis()->SetTitle(TString::Format("#Delta%s/#sigma", poiName.c_str()));
            impact->SetBarWidth(0.9);
            impact->SetBarOffset(0.05);
            impact->SetLineColor(kBlack);
            impact->SetFillColor(kAzure - 4);
            impact->SetFillStyle(tt == 0 ? 3013 : 1001);
            auto impact2 =
               static_cast<TH1 *>(impact->Clone(TString::Format("%s_impact-", tt == 0 ? "prefit" : "postfit")));
            impact2->SetDirectory(nullptr);
            impact2->SetFillColor(kCyan);
            for (int ii = 1; ii <= pNamesHist->GetNbinsX(); ii++) {
               for (auto &c : covariances) {
                  if (c.second != pNamesHist->GetXaxis()->GetBinLabel(ii))
                     continue;
                  auto vv = dynamic_cast<RooRealVar *>(fr->floatParsFinal().find(c.second.c_str()));
                  auto vv_init = dynamic_cast<RooRealVar *>(fr->floatParsInit().find(c.second.c_str()));
                  impact->SetBinContent(ii, ((tt == 0 && !vv_init->hasError()) || !vv->hasError())
                                               ? 0.
                                               : c.first * vv->getError() / vv->getErrorHi() *
                                                    (tt == 0 ? (vv_init->getErrorHi() / vv->getErrorHi()) : 1.));
                  impact2->SetBinContent(ii, ((tt == 0 && !vv_init->hasError()) || !vv->hasError())
                                                ? 0.
                                                : c.first * vv->getError() / vv->getErrorLo() *
                                                     (tt == 0 ? (vv_init->getErrorLo() / vv->getErrorLo()) : 1.));
               }
            }
            hist->GetListOfFunctions()->Add(impact, (doHorizontal) ? "hbarsamemin0" : "bsamey+");
            hist->GetListOfFunctions()->Add(impact2, (doHorizontal) ? "hbarsamemin0" : "bsamey+");
         }
         // add three lines
         for (int ii = -1; ii <= 1; ii++) {
            auto pullLine = new TGraph;
            pullLine->SetName(TString::Format("%dsigmaLine", ii));
            pullLine->SetBit(kCanDelete);
            pullLine->SetPoint(0, -0.5, ii);
            pullLine->SetPoint(1, hist->GetNbinsY() - 0.5, ii);
            pullLine->SetLineStyle(2);
            pullLine->SetEditable(false);
            hist->GetListOfFunctions()->Add(pullLine, "l");
         }
         hist->GetListOfFunctions()->Add(axis); // draw axis last
         TLegend *leg1 =
            new TLegend(0.02, doHorizontal ? (1. - 0.22 * factor) : 0.02, 0.27, (doHorizontal ? 1. : 0.24));
         leg1->SetFillStyle(0);
         leg1->SetBorderSize(0);
         leg1->SetMargin(0.25);
         leg1->SetNColumns(2);

         leg1->SetTextSize(_axis->GetLabelSize());
         leg1->SetTextFont(_axis->GetLabelFont());
         leg1->AddEntry((TObject *)nullptr, "Hessian Pre-fit", "");
         leg1->AddEntry((TObject *)nullptr, "Impact:", "");
         leg1->AddEntry(hist->FindObject("prefit_impact+"), "#theta = #hat{#theta}+#Delta#theta", "f");
         leg1->AddEntry(hist->FindObject("prefit_impact-"), "#theta = #hat{#theta}-#Delta#theta", "f");
         leg1->AddEntry((TObject *)nullptr, "Hessian Post-fit", "");
         leg1->AddEntry((TObject *)nullptr, "Impact:", "");
         leg1->AddEntry(hist->FindObject("postfit_impact+"), "#theta = #hat{#theta}+#Delta#theta", "f");
         leg1->AddEntry(hist->FindObject("postfit_impact-"), "#theta = #hat{#theta}-#Delta#theta", "f");

         hist->GetListOfFunctions()->Add(leg1);
         if (gStyle->GetOptTitle()) {
            histCopy->SetBit(TH1::kNoTitle);
            TPaveText *title =
               new TPaveText(gPad->GetLeftMargin(), 1. - gPad->AbsPixeltoY(14), 1. - gPad->GetRightMargin(), 1., "NDC");
            title->ConvertNDCtoPad();
            title->SetY1NDC(1. - gPad->GetTopMargin() * 0.6);
            title->SetY2NDC(1);
            title->SetTextSize(
               (title->GetTextFont() % 10 > 2)
                  ? (14 / factor)
                  : ((gPad->AbsPixeltoY(0) - gPad->AbsPixeltoY(10 / factor)) / (gPad->GetY2() - gPad->GetY1())));
            title->SetFillStyle(0);
            title->SetBorderSize(0);
            title->AddText(histCopy->GetTitle());
            hist->GetListOfFunctions()->Add(title);
         }
      }

      graph->SetEditable(false);
      pNamesHist->SetLineWidth(0);
      pNamesHist->SetMarkerSize(0);
      pNamesHist->SetMarkerStyle(0);
      graph->GetListOfFunctions()->Add(pNamesHist, "same"); // graph->SetHistogram(pNamesHist);
      if (doHorizontal) {

         // flip the graph and contained graphs
         for (int p = 0; p < graph->GetN(); p++) {
            graph->SetPoint(p, graph->GetPointY(p), graph->GetPointX(p));
            graph->SetPointError(p, graph->GetErrorYlow(p), graph->GetErrorYhigh(p), graph->GetErrorXlow(p),
                                 graph->GetErrorXhigh(p));
         }
         for (auto f : *hist->GetListOfFunctions()) {
            if (f->InheritsFrom("TH1")) {
               // f->Draw("hbarsamemin0");
            } /*else if (auto g2 = dynamic_cast<TGraphErrors *>(f)) {
               for (int p = 0; p < g2->GetN(); p++) {
                  g2->SetPoint(p, g2->GetPointY(p), g2->GetPointX(p));
                  g2->SetPointError(p, g2->GetErrorY(p), _axis->GetNbins());
               }
               //g2->Draw("3");
            } */
            else if (auto g = dynamic_cast<TGraph *>(f)) {
               for (int p = 0; p < g->GetN(); p++) {
                  g->SetPoint(p, g->GetPointY(p), g->GetPointX(p));
               }
               // g->Draw("l");
            } else if (auto l = dynamic_cast<TLine *>(f)) {
               l->SetX1(l->GetY1());
               l->SetX2(l->GetY2());
               l->SetY1(_axis->GetXmax());
               l->SetY2(_axis->GetXmax());
               // l->Draw();
            }
         }
      }

      graph->SetName("pulls");
      hist->GetListOfFunctions()->Add(graph, "z0p");
      // hist->GetListOfFunctions()->Add(histCopy->Clone(".axis"),(sOpt.Contains("impact") &&
      // !doHorizontal)?"axissamey+":"axissame"); // doesn't display right when zoom the axis
      if (!hasSame) {
         histCopy->Draw((sOpt.Contains("impact") && !doHorizontal)
                           ? "axisy+"
                           : "axis"); // draws the axis, called ".axis" for easy access
      }
      hist->Draw("same");
      //
      //      if(sOpt.Contains("impact")) {
      //         // make main object the histogram
      //         auto h = (TH1*)graph->GetHistogram()->Clone("impact");
      //         graph->GetListOfFunctions()->RemoveAll();
      //         for(int ii=1;ii<=h->GetNbinsX();ii++) h->SetBinContent(ii,-4);
      //         h->GetListOfFunctions()->Add(graph,"z0p");
      //         h->Draw("hbar");
      //      } else {
      //         graph->Draw(sOpt.Contains("impact") ? "az0py+" : "az0p");
      //      }
      auto hh = dynamic_cast<TH1 *>(histCopy->Clone(".axiscopy"));
      hh->SetDirectory(nullptr);
      hh->SetBit(kCanDelete);
      hh->Draw(
         (sOpt.Contains("impact") && !doHorizontal)
            ? "axissamey+"
            : "axissame"); // overlay axis again -- important is last so can remove if don't pad->Update before reclear
      gPad->Modified();
      oldPad->cd();
      // gPad->Update();
      return;
   }

   if (get()->InheritsFrom("RooAbsData")) {
      auto s = parentPdf();
      if (s && s->get<RooSimultaneous>()) {
         // drawing dataset associated to a simultaneous means must find subpads with variation names
         // may not have subpads if drawning a "Yield" plot ...
         bool doneDraw = false;
         for (auto c : s->bins()) {
            auto _pad = dynamic_cast<TPad *>(gPad->GetPrimitive(c->GetName()));
            if (!_pad)
               continue; // channel was hidden?
            // attach as a child before calling datasets(), so that if this dataset is external to workspace it is
            // included still attaching the dataset ensures dataset reduction for the channel is applied
            c->push_back(std::make_shared<xRooNode>(*this));
            auto ds = c->datasets().find(GetName());
            c->resize(c->size() - 1); // remove the child we attached
            if (!ds) {
               std::cout << " no ds " << GetName() << " - this should never happen!" << std::endl;
               continue;
            }
            auto tmp = gPad;
            _pad->cd();
            ds->Draw(opt);
            doneDraw = true;
            tmp->cd();
         }
         if (doneDraw) {
            gPad->Modified();
            return;
         }
      }

      if (!s && hasSame) {
         // draw onto all subpads with = in the name
         // if has no such subpads, draw onto this pad
         bool doneDraw = false;
         for (auto o : *gPad->GetListOfPrimitives()) {
            if (auto p = dynamic_cast<TPad *>(o); p && TString(p->GetName()).Contains('=')) {
               auto _tmp = gPad;
               p->cd();
               Draw(opt);
               _tmp->cd();
               doneDraw = true;
            }
         }
         if (doneDraw) {
            gPad->Modified();
            return;
         }
      }

      auto dataGraph = BuildGraph(v, false, (!s && hasSame) ? gPad : nullptr);
      if (!dataGraph)
         return;

      dataGraph->SetBit(kCanDelete);                                          // will be be deleted when pad is cleared
      dataGraph->SetMarkerSize(dataGraph->GetMarkerSize() * gPad->GetWNDC()); // scale marker sizes to pad size

      if (s && !s->get<RooAbsPdf>()->canBeExtended()) {
         // normalize dataGraph to 1
         double tot = 0;
         for (int i = 0; i < dataGraph->GetN(); i++)
            tot += dataGraph->GetPointY(i);
         dataGraph->Scale(1. / tot);
      }

      if (!hasSame) {
         clearPad();
         dataGraph->Draw("Az0p");
         addLegendEntry(dataGraph, strlen(dataGraph->GetTitle()) ? dataGraph->GetTitle() : GetName(), "pEX0");
         gPad->Modified();
         // gPad->Update();
         return;
      }

      bool noPoint = false;
      if (v && dynamic_cast<RooAbsArg *>(v)->getAttribute("global") && dataGraph->GetN() == 1) {
         // global observable ... if graph has only 1 data point line it up on the histogram value
         for (auto o : *gPad->GetListOfPrimitives()) {
            if (auto h = dynamic_cast<TH1 *>(o);
                h && strcmp(h->GetXaxis()->GetName(), dynamic_cast<TObject *>(v)->GetName()) == 0) {
               dataGraph->SetPointY(0, h->Interpolate(dataGraph->GetPointX(0)));
               noPoint = true;
               break;
            }
         }
      }

      if (auto _pad = dynamic_cast<TPad *>(gPad->FindObject("auxPad")); _pad) {
         if (auto h = dynamic_cast<TH1 *>(_pad->GetPrimitive("auxHist")); h) {
            TString histName = h->GetTitle(); // split it by | char
            TString histType = histName(histName.Index('|') + 1, histName.Length());
            histName = histName(0, histName.Index('|'));
            if (auto mainHist = dynamic_cast<TH1 *>(gPad->GetPrimitive(histName));
                mainHist && auxFunctions.find(h->GetYaxis()->GetTitle()) != auxFunctions.end()) {
               // decide what to do based on title of auxHist (previously used name of y-axis but that changed axis
               // behaviour) use title instead
               auto ratioGraph = dynamic_cast<TGraphAsymmErrors *>(dataGraph->Clone(dataGraph->GetName()));
               ratioGraph->SetBit(kCanDelete);
               for (int i = 0; i < ratioGraph->GetN(); i++) {
                  double val = ratioGraph->GetPointY(i);
                  int binNum = mainHist->FindFixBin(ratioGraph->GetPointX(i));
                  double nom = mainHist->GetBinContent(binNum);
                  double nomerr = mainHist->GetBinError(binNum);
                  double yval =
                     std::get<0>(auxFunctions[h->GetYaxis()->GetTitle()])(ratioGraph->GetPointY(i), nom, nomerr);
                  double yup = std::get<0>(auxFunctions[h->GetYaxis()->GetTitle()])(val + ratioGraph->GetErrorYhigh(i),
                                                                                    nom, nomerr) -
                               yval;
                  double ydown = yval - std::get<0>(auxFunctions[h->GetYaxis()->GetTitle()])(
                                           val - ratioGraph->GetErrorYlow(i), nom, nomerr);
                  if (!std::isnan(yval)) {
                     ratioGraph->SetPointY(i, yval);
                     if (!std::isnan(yup))
                        ratioGraph->SetPointEYhigh(i, yup);
                     if (!std::isnan(ydown))
                        ratioGraph->SetPointEYlow(i, ydown);
                  }
               }
               // remove the zero points
               int i = 0;
               while (i < ratioGraph->GetN()) {
                  if (ratioGraph->GetPointY(i) == 0 && ratioGraph->GetErrorYhigh(i) == 0 &&
                      ratioGraph->GetErrorYlow(i) == 0) {
                     ratioGraph->RemovePoint(i);
                  } else {
                     i++;
                  }
               }
               auto _tmpPad = gPad;
               _pad->cd();
               ratioGraph->Draw("z0psame");
               auto minMax = graphMinMax(ratioGraph);
               adjustYRange(minMax.first, minMax.second, h, std::get<1>(auxFunctions[h->GetYaxis()->GetTitle()]));
               _tmpPad->cd();
            }
         }
      }

      dataGraph->Draw("z0p same");
      addLegendEntry((noPoint) ? nullptr : dataGraph, strlen(dataGraph->GetTitle()) ? dataGraph->GetTitle() : GetName(),
                     noPoint ? "" : "pEX0");

      auto minMax = graphMinMax(dynamic_cast<TGraphAsymmErrors *>(dataGraph));
      adjustYRange(minMax.first, minMax.second);

      gPad->Modified();
      // gPad->Update();
      return;
   }

   // auto _ax = GetXaxis();
   // auto v = (_ax) ? dynamic_cast<RooRealVar*>(/*possibleObs.first()*/_ax->GetParent()) : nullptr;
   // if (!v) { v = get<RooRealVar>(); } // self-axis
   // if (!v) return;

   if (auto lv = get<RooAbsLValue>(); lv && fParent && fParent->get<RooAbsData>()) {
      // drawing an observable from a dataset ... build graph, and exit
      auto gr = fParent->BuildGraph(lv, true);
      gr->SetBit(kCanDelete);
      gr->Draw(hasSame ? "P" : "AP");
      return;
   }

   if (forceNames != "") {
      // drawing a force plot ... build nll and fill a histogram with force terms
      auto _dsets = datasets();
      bool _drawn = false;
      auto _coords = coords();
      auto _fr = fitResult();
      auto initPar = dynamic_cast<RooRealVar *>(_fr.get<RooFitResult>()->floatParsInit().find(forceNames));
      if (!initPar)
         return;
      std::vector<double> valuesToDo = {initPar->getVal()};
      if (initPar->hasError() || initPar->hasAsymError()) {
         valuesToDo.push_back(initPar->getVal() + initPar->getErrorLo());
         valuesToDo.push_back(initPar->getVal() + initPar->getErrorHi());
      }
      int ii = 0;
      for (auto valueToDo : valuesToDo) {
         ii++;
         for (auto &d : _dsets) {
            if (!d->get()->TestBit(1 << 20))
               continue;
            auto emptyHist = BuildHistogram(v, true);
            emptyHist->SetBit(kCanDelete);
            auto _obs = d->obs();
            auto x = _obs.find((v) ? dynamic_cast<TObject *>(v)->GetName() : emptyHist->GetXaxis()->GetName());
            auto _nll = nll(d);
            auto theData = d->get<RooAbsData>();
            int nevent = theData->numEntries();
            for (int i = 0; i < nevent; i++) {
               theData->get(i);
               bool _skip = false;
               for (const auto &_c : _coords) {
                  if (auto cat = _c->get<RooAbsCategoryLValue>(); cat) {
                     if (cat->getIndex() != theData->get()->getCatIndex(cat->GetName())) {
                        _skip = true;
                        break;
                     }
                  }
               }
               if (_skip)
                  continue;

               if (x) {
                  auto val = _nll.pars()->getRealValue(initPar->GetName());
                  if (ii > 1)
                     _nll.pars()->setRealValue(initPar->GetName(), valueToDo);
                  auto nllVal = _nll.getEntryVal(i);
                  _nll.pars()->setRealValue(initPar->GetName(), initPar->getVal());
                  auto nllVal2 = _nll.getEntryVal(i);
                  _nll.pars()->setRealValue(initPar->GetName(), val);
                  emptyHist->Fill(x->get<RooAbsReal>()->getVal(), (nllVal2 - nllVal));
               }
            }
            // include the extendedTerm, distributed evenly over the bins
            // probably should be somehow dependent on data density though (i.e. bins with more data get more of it?)
            auto val = _nll.pars()->getRealValue(initPar->GetName());
            if (ii > 1)
               _nll.pars()->setRealValue(initPar->GetName(), valueToDo);
            auto _extTerm = _nll.extendedTermVal();
            _nll.pars()->setRealValue(initPar->GetName(), initPar->getVal());
            auto _extTerm2 = _nll.extendedTermVal();
            _nll.pars()->setRealValue(initPar->GetName(), val);
            for (int i = 1; i <= emptyHist->GetNbinsX(); i++) {
               emptyHist->SetBinContent(i,
                                        emptyHist->GetBinContent(i) + (_extTerm2 - _extTerm) / emptyHist->GetNbinsX());
               emptyHist->SetBinError(i, 0);
            }
            emptyHist->GetYaxis()->SetTitle("log (L(#theta)/L(#theta_{0}))");
            emptyHist->SetTitle(TString::Format("#theta = %g", (ii > 1) ? valueToDo : val));
            if (ii == 1)
               emptyHist->SetLineColor(kBlack);
            if (ii == 2) {
               emptyHist->SetLineColor(kRed);
            } else if (ii == 3) {
               emptyHist->SetLineColor(kBlue);
            }
            emptyHist->Draw(_drawn ? "same" : "");
            _drawn = true;
         }
      }
      return;
   }

   auto rar = get<RooAbsReal>();
   const xRooNode *rarNode = this;
   if (!rar) {
      // draw a deleteable clone of the object we wrap (since we might own the object)
      get()->DrawClone(opt);
      return;
   }
   //   RooAbsReal *sf = nullptr;
   if (get()->InheritsFrom("RooExtendPdf")) {
      browse();
      rarNode = find(".pdf").get();
      // rar = rarNode->get<RooAbsReal>();
      //      sf = find(".n")->get<RooAbsReal>();
   }

   if (!nostack && !hasOverlay &&
       (rarNode->get()->InheritsFrom("RooRealSumPdf") || rarNode->get()->InheritsFrom("RooAddPdf") ||
        (v && rarNode->get()->InheritsFrom("RooSimultaneous") &&
         strcmp(dynamic_cast<TObject *>(v)->GetName(), rarNode->get<RooSimultaneous>()->indexCat().GetName()) == 0))) {
      nostack = false;
   } else {
      // in all other cases, we do not build a stack
      nostack = true;
   }

   auto h = BuildHistogram(v, false, hasErrorOpt, 1, 0, "", false, false, 0, nullptr, nostack, true /*setInterp*/);
   if (!h) {
      if (get()) {
         // draw a deleteable clone of the object we wrap (since we might own the object)
         get()->DrawClone(opt);
      }
      return;
   }
   h->SetBit(kCanDelete);

   if (!v)
      v = getObject<RooAbsLValue>(h->GetXaxis()->IsAlphanumeric() ? h->GetXaxis()->GetTimeFormatOnly()
                                                                  : h->GetXaxis()->GetName())
             .get();
   RooAbsArg *vv = (v) ? dynamic_cast<RooAbsArg *>(v) : rar;
   if (h->GetXaxis()->IsAlphanumeric()) {
      // do this to get bin labels
      h->GetXaxis()->SetName("xaxis"); // WARNING -- this messes up anywhere we GetXaxis()->GetName()
   }

   // get style now, before we mess with histogram title
   // auto _styleNode = styles(h);

   if (rar->InheritsFrom("RooAbsPdf") && !(rar->InheritsFrom("RooRealSumPdf") || rar->InheritsFrom("RooAddPdf") ||
                                           rar->InheritsFrom("RooSimultaneous"))) {
      // append parameter values to title if has such
      RooArgSet s;
      rar->leafNodeServerList(&s);
      if (v)
         s.remove(*dynamic_cast<RooAbsArg *>(v));
      if (!s.empty()) {
         TString ss = h->GetTitle();
         ss += " [";
         bool first = true;
         for (auto _p : s) {
            auto _v = dynamic_cast<RooRealVar *>(_p);
            if (!_v)
               continue;
            if (!first)
               ss += ",";
            first = false;
            ss += TString::Format("%s=%g", strlen(_p->GetTitle()) ? _p->GetTitle() : _p->GetName(), _v->getVal());
            if (_v->hasError()) {
               ss += TString::Format("#pm %g", _v->getError());
            }
         }
         ss += "]";
         h->SetTitle(ss);
      }
   }

   if (!hasSame) {
      if (obs().find(vv->GetName())) {
         gPad->SetGrid(0, 0);
      } else {
         gPad->SetGrid(1, 1);
      }
   }
   TString dOpt = h->GetOption();
   if (dOpt == "l")
      h->SetFillStyle(0);
   //   // need to strip namespace to discount the "HistFactory" namespace classes from all being treated as binned
   //   TString clNameNoNamespace = rar->ClassName();
   //   clNameNoNamespace = clNameNoNamespace(clNameNoNamespace.Last(':') + 1, clNameNoNamespace.Length());
   //   TString dOpt = (clNameNoNamespace.Contains("Hist") || vv->isCategory() || rar->isBinnedDistribution(*vv) ||
   //                   h->GetNbinsX() == 1 || rar->getAttribute("BinnedLikelihood") ||
   //                   (dynamic_cast<RooAbsRealLValue *>(vv) &&
   //                    std::unique_ptr<std::list<double>>(rar->binBoundaries(*dynamic_cast<RooAbsRealLValue *>(vv),
   //                                                                          -std::numeric_limits<double>::infinity(),
   //                                                                          std::numeric_limits<double>::infinity()))))
   //                     ? ""
   //                     : "LF2";
   //   if (auto d = dynamic_cast<RooHistFunc *>(rar); d && !d->isBinnedDistribution(*vv) && h->GetNbinsX() != 1) {
   //      dOpt = "LF2"; // hist func is interpolated, so draw it as such
   //   }
   //   if (dOpt == "LF2" && !components().empty()) {
   //      // check if all components of dOpt are "Hist" type (CMS model support)
   //      // if so then dOpt="";
   //      bool allHist = true;
   //      for (auto &s : components()) {
   //         TString _clName = s->get()->ClassName();
   //         _clName = _clName(_clName.Last(':') + 1, _clName.Length());
   //         if (!(s->get() && _clName.Contains("Hist"))) {
   //            allHist = false;
   //            break;
   //         }
   //      }
   //      if (allHist)
   //         dOpt = "";
   //   }
   //
   //   if(dOpt=="LF2") {
   //      // ensure any sub hists have lf2 option
   //      TObjLink *lnk = h->GetListOfFunctions()->FirstLink();
   //      while (lnk) {
   //         if(auto hh = dynamic_cast<TH1*>(lnk->GetObject())) {
   //            if(TString(hh->GetName())=="band" && TString(lnk->GetOption())=="e2same") {
   //               lnk->SetOption("LF2 e3same");
   //            } else if(TString(hh->GetName())=="nominal") {
   //               lnk->SetOption("L same");
   //            }
   //         }
   //         lnk = lnk->Next();
   //      }
   //   }

   if (rar == vv && rar->IsA() == RooRealVar::Class()) {
      dOpt += "TEXT";
      // add a TExec to the histogram so that when edited it will propagate to var
      gROOT->SetEditHistograms(true);
   } else {
      gROOT->SetEditHistograms(false);
   }

   if (hasSame) {
      dOpt += " same";
   } else {
      hAxis = h;
   }

   if (dOpt.Contains("TEXT") || sOpt.Contains("text")) {
      // adjust marker size so text is good
      h->SetMarkerSize(gStyle->GetLabelSize("Z") / (0.02 * gPad->GetHNDC()));
   }

   bool hasError(false);
   for (int i = 0; i < h->GetSumw2N(); i++) {
      if (h->GetSumw2()->At(i)) {
         hasError = true;
         break;
      }
   }

   /** This doesn't seem necessary in at least 6.26 any more - pads seem adjusted on their own
   if (!hasSame && h->GetYaxis()->GetTitleFont()%10 == 2) {
       h->GetYaxis()->SetTitleOffset( gPad->GetLeftMargin() / gStyle->GetPadLeftMargin() );
   } */
   // don't this instead - dont want to leave as zero (auto) in case show aux plot
   if (!hasSame && h->GetYaxis()->GetTitleFont() % 10 == 2) {
      h->GetYaxis()->SetTitleOffset(1.);
   }

   TH1 *errHist = nullptr;

   if (!hasSame)
      clearPad();

   if (rar == vv && rar->IsA() == RooRealVar::Class()) {
      // add a TExec to the histogram so that when edited it will propagate to var
      // h->GetListOfFunctions()->Add(h->Clone("self"),"TEXTHIST");
      dOpt = "TEXT";
      auto node = new xRooNode(*this);
      auto _hist = (errHist) ? errHist : h;
      auto hCopy = (errHist) ? nullptr : dynamic_cast<TH1 *>(h->Clone());
      if (hCopy) {
         hCopy->Reset();
         hCopy->Add(_hist);
         hCopy->SetDirectory(nullptr);
      }
      _hist->GetListOfFunctions()->Add(node);
      _hist->GetListOfFunctions()->Add(new TExec(
         ".update",
         TString::Format(
            "gROOT->SetEditHistograms(true);auto h = dynamic_cast<TH1*>(gPad->GetPrimitive(\"%s\")); if(h) { double "
            "range= h->GetMaximum()-h->GetMinimum(); if(auto n "
            "= dynamic_cast<xRooNode*>(h->GetListOfFunctions()->FindObject(\"%s\")); n && "
            "n->TestBit(TObject::kNotDeleted) && n->get<RooRealVar>()->getVal() != h->GetBinContent(1)) {"
            "h->SetBinContent(1, "
            "TString::Format(\"%%.2g\",int(h->GetBinContent(1)/(range*0.01))*range*0.01).Atof());n->SetContent( "
            "h->GetBinContent(1) ); for(auto pp : *h->GetListOfFunctions()) if(auto hh = "
            "dynamic_cast<TH1*>(pp))hh->SetBinContent(1,h->GetBinContent(1));} if(h->GetBinContent(1)==0.) "
            "h->SetBinContent(1,range*0.005); gPad->Modified();gPad->Update(); }",
            _hist->GetName(), node->GetName())));
      if (errHist) {
         errHist->GetListOfFunctions()->Add(h, "TEXT HIST same");
         errHist->SetFillColor(h->GetLineColor());
      } else {
         hCopy->SetBit(kCanDelete);
         hCopy->SetFillStyle(0);
         _hist->GetListOfFunctions()->Add(hCopy, "TEXT HIST same");
         //_hist->SetBinError(1, 0);
      }
      _hist->SetStats(false);
      // if (_hist->GetBinContent(1)==0.) _hist->SetBinContent(1,(_hist->GetMaximum()-_hist->GetMinimum())*0.005);
      _hist->Draw(); //_hist->Draw(((hasError) ? "e2" : ""));
      gPad->Modified();
      return;
   }

   bool overlayExisted = false;
   if (hasOverlay) {
      h->SetName(TString::Format("%s%s", h->GetName(), overlayName.Data()));
      if (auto existing = dynamic_cast<TH1 *>(gPad->GetPrimitive(h->GetName())); existing) {
         existing->Reset();
         existing->Add(h);
         delete h;
         h = existing;
         overlayExisted = true;
      } else {
         TString oldStyle = (rar && rar->getStringAttribute("style")) ? rar->getStringAttribute("style") : "";
         h->SetTitle(overlayName);
         // for overlays will take style from current gStyle before overriding with personal style
         // this ensures initial style will be whatever gStyle is, rather than whatever ours is
         static_cast<TAttLine &>(*h) = *gStyle;
         static_cast<TAttFill &>(*h) = *gStyle;
         static_cast<TAttMarker &>(*h) = *gStyle;
         h->SetFillStyle(0); // explicit default for overlays will be transparent fill

         //            std::shared_ptr<TStyle> style; // use to keep alive for access from GetStyle below, in case
         //            getObject has decided to return the owning ptr (for some reason) if
         //            (!gROOT->GetStyle(h->GetTitle())) {
         //                if ( (style = getObject<TStyle>(h->GetTitle())) ) {
         //                    // loaded style (from workspace?) so put in list and use that
         //                    gROOT->GetListOfStyles()->Add(style.get());
         //                } else {
         //                    // create new style - gets put in style list automatically so don't have to delete
         //                    // acquire them so saved to workspaces for auto reload ...
         //                    style = acquireNew<TStyle>(h->GetTitle(),
         //                                               TString::Format("Style for %s component", h->GetTitle()));
         //                    (TAttLine &) (*style) = *dynamic_cast<TAttLine *>(h);
         //                    (TAttFill &) (*style) = *dynamic_cast<TAttFill *>(h);
         //                    (TAttMarker &) (*style) = *dynamic_cast<TAttMarker *>(h);
         //                    gROOT->GetListOfStyles()->Add(style.get());
         //                }
         //            }
         //            (TAttLine&)(*h) = *(gROOT->GetStyle(h->GetTitle()) ? gROOT->GetStyle(h->GetTitle()) : gStyle);
         //            (TAttFill&)(*h) = *(gROOT->GetStyle(h->GetTitle()) ? gROOT->GetStyle(h->GetTitle()) : gStyle);
         //            (TAttMarker&)(*h) = *(gROOT->GetStyle(h->GetTitle()) ? gROOT->GetStyle(h->GetTitle()) : gStyle);
         auto _styleNode = styles(h);
         rar->setStringAttribute("style", oldStyle == "" ? nullptr : oldStyle.Data()); // restores old style
         if (auto _style = _styleNode.get<TStyle>()) {
            (TAttLine &)(*h) = *_style;
            (TAttFill &)(*h) = *_style;
            (TAttMarker &)(*h) = *_style;
         }
         h->Draw(dOpt == "LF2" ? "e3" : dOpt);
         if (errHist) {
            errHist->SetTitle(overlayName);
            (TAttLine &)(*errHist) = *h;
            errHist->SetFillColor(h->GetLineColor());
         }
      }
   } else {
      //      if (auto _style = _styleNode.get<TStyle>()) {
      //         (TAttLine &)(*h) = *_style;
      //         (TAttFill &)(*h) = *_style;
      //         (TAttMarker &)(*h) = *_style;
      //         if (errHist) {
      //            (TAttLine &)(*errHist) = *h;
      //            errHist->SetFillColor(h->GetLineColor());
      //         }
      //      }
      h->Draw(dOpt);
   }

   if (!hasOverlay && (rarNode->get()->InheritsFrom("RooRealSumPdf") || rarNode->get()->InheritsFrom("RooAddPdf") ||
                       (rarNode->get()->InheritsFrom("RooSimultaneous") &&
                        strcmp(vv->GetName(), rarNode->get<RooSimultaneous>()->indexCat().GetName()) == 0))) {
      if (auto stack = dynamic_cast<THStack *>(h->FindObject("stack"))) {
         // access the stack and set draw options, adjust ranges etc
         TObjLink *lnk = stack->GetHists()->FirstLink();
         while (lnk) {
            TH1 *hh = static_cast<TH1 *>(lnk->GetObject());
            // lnk->SetOption(dOpt); - not needed
            auto hhMin = (hh->GetMinimum() == 0) ? hh->GetMinimum(1e-9) : hh->GetMinimum();
            if (lnk == stack->GetHists()->FirstLink() && h->GetMinimum() > hhMin) {
               auto newMin = hhMin - (h->GetMaximum() - hhMin) * gStyle->GetHistTopMargin();
               if (hhMin >= 0 && newMin < 0)
                  newMin = hhMin * 0.99;
               adjustYRange(newMin, h->GetMaximum());
            }
            addLegendEntry(hh, hh->GetTitle(), "f");
            lnk = lnk->Next();
         }
      }

      //      // build a stack unless not requested
      //      if (!nostack) {
      //         THStack *stack = new THStack(TString::Format("%s_stack", rar->GetName()),
      //                                      TString::Format("%s;%s", rar->GetTitle(), h->GetXaxis()->GetTitle()));
      //         int count = 2;
      //         std::map<std::string, int> colorByTitle; // TODO: should fill from any existing legend
      //         std::set<std::string> allTitles;
      //         bool titleMatchName = true;
      //         std::map<std::string, TH1 *> histGroups;
      //         std::vector<TH1 *> hhs;
      //         std::set<TH1 *> histsWithBadTitles; // these histograms will have their titles autoFormatted
      //
      //         // support for CMS model case where has single component containing many coeffs
      //         // will build stack by setting each coeff equal to 0 in turn, rebuilding the histogram
      //         // the difference from the "full" histogram will be the component
      //         RooArgList cms_coefs;
      //         if (!rarNode->components().empty()) {
      //            auto comps = rarNode->components()[0];
      //            for (auto &c : *comps) {
      //               if (c->fFolder == "!.coeffs")
      //                  cms_coefs.add(*c->get<RooAbsArg>());
      //            }
      //         }
      //         if (!cms_coefs.empty()) {
      //            RooRealVar zero("zero", "", 0);
      //            std::shared_ptr<TH1> prevHist(static_cast<TH1 *>(h->Clone()));
      //            for (auto c : cms_coefs) {
      //               // seems I have to remake the function each time, as haven't figured out what cache needs
      //               clearing? std::unique_ptr<RooAbsReal> f(
      //                  dynamic_cast<RooAbsReal *>(rarNode->components()[0]->get()->Clone("tmpCopy")));
      //               zero.setAttribute(
      //                  Form("ORIGNAME:%s", c->GetName()));            // used in redirectServers to say what this
      //                  replaces
      //               f->redirectServers(RooArgSet(zero), false, true); // each time will replace one additional coef
      //               // zero.setAttribute(Form("ORIGNAME:%s",c->GetName()),false); (commented out so that on next
      //               iteration
      //               // will still replace all prev)
      //               auto hh = xRooNode(*f, *this).BuildHistogram(v);
      //               hh->SetName(c->GetName());
      //               if (sf)
      //                  hh->Scale(sf->getVal());
      //               if (strlen(hh->GetTitle()) == 0) {
      //                  hh->SetTitle(c->GetName()); // ensure all hists has titles
      //                  histsWithBadTitles.insert(hh);
      //               } else if (strcmp(hh->GetName(), hh->GetTitle()) == 0) {
      //                  histsWithBadTitles.insert(hh);
      //               }
      //               titleMatchName &= (TString(c->GetName()) == hh->GetTitle() ||
      //                                  TString(hh->GetTitle()).BeginsWith(TString(c->GetName()) + "_"));
      //               std::shared_ptr<TH1> nextHist(static_cast<TH1 *>(hh->Clone()));
      //               hh->Add(prevHist.get(), -1.);
      //               hh->Scale(-1.);
      //               hhs.push_back(hh);
      //               prevHist = nextHist;
      //            }
      //         } else if (get<RooSimultaneous>()) {
      //            // need to create a histogram for each sample across all the channels - will rely on functionality
      //            below to
      //            // merge them based on titles
      //
      //            for (auto &chan : bins()) {
      //               TString chanName(chan->GetName());
      //               chanName = chanName(chanName.Index("=") + 1, chanName.Length());
      //               auto samps = chan->mainChild();
      //               if (!samps)
      //                  samps = *chan;
      //               for (auto &samp : samps.components()) {
      //                  auto hh = static_cast<TH1 *>(h->Clone(samp->GetName()));
      //                  hh->Reset();
      //                  hh->SetTitle(samp->GetTitle());
      //                  if (strlen(hh->GetTitle()) == 0) {
      //                     hh->SetTitle(samp->GetName());
      //                     histsWithBadTitles.insert(hh);
      //                  } else if (strcmp(hh->GetName(), hh->GetTitle()) == 0) {
      //                     histsWithBadTitles.insert(hh);
      //                  }
      //                  hh->SetTitle(TString(hh->GetTitle())
      //                                  .ReplaceAll(TString(chan->get()->GetName()) + "_",
      //                                              "")); // remove occurance of channelname_ in title (usually
      //                                              prefix)
      //                  titleMatchName &= (TString(samp->GetName()) == hh->GetTitle() ||
      //                                     TString(hh->GetTitle()).BeginsWith(TString(samp->GetName()) + "_"));
      //                  hh->SetBinContent(hh->GetXaxis()->FindFixBin(chanName), samp->GetContent());
      //                  hhs.push_back(hh);
      //               }
      //            }
      //         } else {
      //            for (auto &samp : rarNode->components()) {
      //               auto hh = samp->BuildHistogram(v,false,false,1,0,"",false,false,0,h); // passing h to ensure
      //               binning is the same for all subcomponent hists if (sf)
      //                  hh->Scale(sf->getVal());
      //               hhs.push_back(hh);
      //               if (strlen(hh->GetTitle()) == 0) {
      //                  hh->SetTitle(samp->GetName()); // ensure all hists has titles
      //                  histsWithBadTitles.insert(hh);
      //               } else if (strcmp(hh->GetName(), hh->GetTitle()) == 0) {
      //                  histsWithBadTitles.insert(hh);
      //               }
      //               titleMatchName &= (TString(samp->GetName()) == hh->GetTitle() ||
      //                                  TString(hh->GetTitle()).BeginsWith(TString(samp->GetName()) + "_"));
      //            }
      //         }
      //
      //         if (!hhs.empty()) {
      //            for (auto &hh : hhs) {
      //               allTitles.insert(hh->GetTitle());
      //            }
      //
      //            // get common prefix to strip off only if all titles match names and
      //            // any title is longer than 10 chars
      //            size_t e = std::min(allTitles.begin()->size(), allTitles.rbegin()->size());
      //            size_t ii = 0;
      //            bool goodPrefix = false;
      //            std::string commonSuffix;
      //            if (titleMatchName && hhs.size() > 1) {
      //               while (ii < e - 1 && allTitles.begin()->at(ii) == allTitles.rbegin()->at(ii)) {
      //                  ii++;
      //                  if (allTitles.begin()->at(ii) == '_' || allTitles.begin()->at(ii) == ' ')
      //                     goodPrefix = true;
      //               }
      //
      //               // find common suffix if there is one .. must start with a "_"
      //               bool stop = false;
      //               while (!stop && commonSuffix.size() < size_t(e - 1)) {
      //                  commonSuffix = allTitles.begin()->substr(allTitles.begin()->length() - commonSuffix.length() -
      //                  1); for (auto &t : allTitles) {
      //                     if (!TString(t).EndsWith(commonSuffix.c_str())) {
      //                        commonSuffix = commonSuffix.substr(1);
      //                        stop = true;
      //                        break;
      //                     }
      //                  }
      //               }
      //               if (commonSuffix.find('_') == std::string::npos) {
      //                  commonSuffix = "";
      //               } else {
      //                  commonSuffix = commonSuffix.substr(commonSuffix.find('_'));
      //               }
      //            }
      //            if (!goodPrefix)
      //               ii = 0;
      //
      //            // also find how many characters are needed to distinguish all entries (that dont have the same
      //            name)
      //            // then carry on up to first space or underscore
      //            size_t jj = 0;
      //            std::map<std::string, std::string> reducedTitles;
      //            while (reducedTitles.size() != allTitles.size()) {
      //               jj++;
      //               std::map<std::string, int> titlesMap;
      //               for (auto &s : allTitles) {
      //                  if (reducedTitles.count(s))
      //                     continue;
      //                  titlesMap[s.substr(0, jj)]++;
      //               }
      //               for (auto &s : allTitles) {
      //                  if (titlesMap[s.substr(0, jj)] == 1 && (jj >= s.length() || s.at(jj) == ' ' || s.at(jj) ==
      //                  '_')) {
      //                     reducedTitles[s] = s.substr(0, jj);
      //                  }
      //               }
      //            }
      //
      //            // strip common prefix and suffix before adding
      //            for (auto ritr = hhs.rbegin(); ritr != hhs.rend(); ++ritr) { // go in reverse order
      //               if (!histsWithBadTitles.count((*ritr))) {
      //                  continue;
      //               }
      //               auto _title = (hhs.size() > 5) ? reducedTitles[(*ritr)->GetTitle()] : (*ritr)->GetTitle();
      //               _title = _title.substr(ii < _title.size() ? ii : 0);
      //               if (!commonSuffix.empty() && TString(_title).EndsWith(commonSuffix.c_str()))
      //                  _title = _title.substr(0, _title.length() - commonSuffix.length());
      //               (*ritr)->SetTitle(_title.c_str());
      //            }
      //         }
      //
      //         for (auto &hh : hhs) {
      //            // automatically group hists that all have the same title
      //            if (histGroups.find(hh->GetTitle()) == histGroups.end()) {
      //               histGroups[hh->GetTitle()] = hh;
      //            } else {
      //               // add it into this group
      //               histGroups[hh->GetTitle()]->Add(hh);
      //               delete hh;
      //               hh = nullptr;
      //               continue;
      //            }
      //            auto hhMin = (hh->GetMinimum() == 0) ? hh->GetMinimum(1e-9) : hh->GetMinimum();
      //            if (!stack->GetHists() && h->GetMinimum() > hhMin) {
      //               auto newMin = hhMin - (h->GetMaximum() - hhMin) * gStyle->GetHistTopMargin();
      //               if (hhMin >= 0 && newMin < 0)
      //                  newMin = hhMin * 0.99;
      //               adjustYRange(newMin, h->GetMaximum());
      //            }
      //
      //            /*if(stack->GetHists() && stack->GetHists()->GetEntries()>0) {
      //                // to remove rounding effects on bin boundaries, see if binnings compatible
      //                auto _h1 = dynamic_cast<TH1*>(stack->GetHists()->At(0));
      //                if(_h1->GetNbinsX()==hh->GetNbinsX()) TODO ... finish dealing with silly rounding effects
      //            }*/
      //            TString thisOpt = dOpt;
      //            // uncomment next line to blend continuous with discrete components .. get some unpleasant "poke
      //            through"
      //            // effects though
      //            // if(auto s = samp->get<RooAbsReal>(); s) thisOpt =
      //            s->isBinnedDistribution(*dynamic_cast<RooAbsArg*>(v)) ?
      //            // "" : "LF2";
      //            stack->Add(hh, thisOpt);
      //         }
      //         stack->SetBit(kCanDelete); // should delete its sub histograms
      //         stack->Draw("noclear same");
      //         h->Draw(
      //            dOpt + sOpt +
      //            "same"); // overlay again ..  if stack would cover original hist (negative components) we still see
      //            integral
      //         h->Draw("axissame"); // redraws axis
      //
      //         TList *ll = stack->GetHists();
      //         if (ll && ll->GetEntries()) {
      //
      //            // finally, ensure all hists are styled
      //            for (auto ho : *ll) {
      //               TH1 *hh = dynamic_cast<TH1 *>(ho);
      //               if (!hh)
      //                  continue;
      //               bool createdStyle = (xRooNode(*hh, *this).styles(nullptr, false).get<TStyle>() == nullptr);
      //
      //               if (createdStyle) {
      //                  // give hist a color, that isn't the same as any other hists color
      //                  hh->SetFillStyle(1001); // solid fill style
      //                  bool used = false;
      //                  do {
      //                     hh->SetFillColor((count++));
      //                     // check not already used this color
      //                     used = false;
      //                     for (auto ho2 : *ll) {
      //                        TH1 *hh2 = dynamic_cast<TH1 *>(ho2);
      //                        if (!hh2)
      //                           continue;
      //                        auto _styleNode = xRooNode(*hh2, *this).styles(hh2, false);
      //                        auto _style = _styleNode.get<TStyle>();
      //                        if (hh != hh2 && _style && _style->GetFillColor() == hh->GetFillColor()) {
      //                           used = true;
      //                           break;
      //                        }
      //                     }
      //                  } while (used);
      //               }
      //
      //               auto _styleNode = xRooNode(*hh, *this).styles(hh);
      //               if (auto _style = _styleNode.get<TStyle>()) {
      //                  *dynamic_cast<TAttLine *>(hh) = *_style;
      //                  *dynamic_cast<TAttFill *>(hh) = *_style;
      //                  *dynamic_cast<TAttMarker *>(hh) = *_style;
      //               }
      //               // for stacks, fill color of white should be color 10 unless fill style is 0
      //               if (hh->GetFillColor() == kWhite && hh->GetFillStyle() != 0) {
      //                  // kWhite means 'transparent' in ROOT ... should really use a FillStyle of 0 for that
      //                  // so assume user wanted actual white, which is color 10
      //                  hh->SetFillColor(10);
      //               }
      //               addLegendEntry(hh, hh->GetTitle(), "f");
      //            }
      //         }
      //      }
   } else if (!overlayExisted) {

      if (errHist) {
         addLegendEntry(errHist, strlen(errHist->GetTitle()) ? errHist->GetTitle() : GetName(), "fl");
      } else {
         addLegendEntry(h, strlen(h->GetTitle()) ? h->GetTitle() : GetName(), (hasError) ? "fl" : "l");
      }
   }

   if (errHist) {
      dOpt.ReplaceAll("TEXT", "");
      errHist->Draw(dOpt + (dOpt.Contains("LF2") ? "e3same" : "e2same"));
      double ymax = -std::numeric_limits<double>::infinity();
      double ymin = std::numeric_limits<double>::infinity();
      for (int i = 1; i <= errHist->GetNbinsX(); i++) {
         ymax = std::max(ymax, errHist->GetBinContent(i) + errHist->GetBinError(i));
         ymin = std::min(ymin, errHist->GetBinContent(i) - errHist->GetBinError(i));
      }
      adjustYRange(ymin, ymax);
   } else {
      adjustYRange(h->GetMinimum() * 0.9, h->GetMaximum() * 1.1);
   }

   if ((!auxPlotTitle.empty()) && !hasSame) {
      // create a pad for the ratio ... shift the bottom margin of this pad to make space for it
      double padFrac = 0.3;
      auto _tmpPad = gPad;
      gPad->SetBottomMargin(padFrac);
      auto ratioPad = new TPad("auxPad", "aux plot", 0, 0, 1, padFrac);
      ratioPad->SetFillColor(_tmpPad->GetFillColor());
      ratioPad->SetNumber(1);
      ratioPad->SetBottomMargin(ratioPad->GetBottomMargin() / padFrac);
      ratioPad->SetTopMargin(0.04);
      ratioPad->SetLeftMargin(gPad->GetLeftMargin());
      ratioPad->SetRightMargin(gPad->GetRightMargin());
      ratioPad->cd();
      TH1 *ratioHist = dynamic_cast<TH1 *>((errHist) ? errHist->Clone("auxHist") : h->Clone("auxHist"));
      ratioHist->Reset();
      ratioHist->Add(h); // removes function list
      ratioHist->SetDirectory(nullptr);
      ratioHist->SetTitle((errHist) ? errHist->GetName()
                                    : h->GetName()); // abuse the title string to hold the name of the main hist

      ratioHist->GetYaxis()->SetNdivisions(5, 0, 0);
      ratioHist->GetYaxis()->SetTitle(auxPlotTitle.c_str());
      ratioHist->SetTitle(
         TString::Format("%s|%s", ratioHist->GetTitle(),
                         auxPlotTitle.c_str())); // used when plotting data (above) to decide what to calculate
      ratioHist->SetMaximum();
      ratioHist->SetMinimum(); // resets min and max
      ratioPad->SetGridy();

      for (int i = 1; i <= ratioHist->GetNbinsX(); i++) {
         double val = ratioHist->GetBinContent(i);
         double err = ratioHist->GetBinError(i);
         ratioHist->SetBinContent(i, std::get<0>(auxFunctions[auxPlotTitle])(val, val, err));
         ratioHist->SetBinError(i, std::get<0>(auxFunctions[auxPlotTitle])(val + err, val, err) -
                                      ratioHist->GetBinContent(i));
      }

      double rHeight = 1. / padFrac; //(_tmpPad->GetWNDC())/(gPad->GetHNDC());
      if (ratioHist->GetYaxis()->GetTitleFont() % 10 == 2) {
         ratioHist->GetYaxis()->SetTitleSize(ratioHist->GetYaxis()->GetTitleSize() * rHeight);
         ratioHist->GetYaxis()->SetLabelSize(ratioHist->GetYaxis()->GetLabelSize() * rHeight);
         ratioHist->GetXaxis()->SetTitleSize(ratioHist->GetXaxis()->GetTitleSize() * rHeight);
         ratioHist->GetXaxis()->SetLabelSize(ratioHist->GetXaxis()->GetLabelSize() * rHeight);
         ratioHist->GetYaxis()->SetTitleOffset(ratioHist->GetYaxis()->GetTitleOffset() / rHeight);
      } else {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 26, 00)
         ratioHist->GetYaxis()->SetTitleOffset(ratioHist->GetYaxis()->GetTitleOffset() / rHeight);
#endif
      }
      ratioHist->GetXaxis()->SetTickLength(ratioHist->GetXaxis()->GetTickLength() * rHeight);
      ratioHist->SetStats(false);
      ratioHist->SetBit(TH1::kNoTitle);
      ratioHist->SetBit(kCanDelete);
      if (errHist) {
         auto _h = dynamic_cast<TH1 *>(ratioHist->Clone("auxHist_clone"));
         _h->SetDirectory(nullptr);
         _h->SetFillColor(0);
         ratioHist->GetListOfFunctions()->Add(_h, "histsame");
         //_h->Draw("histsame");
      }
      ratioHist->GetListOfFunctions()->Add(new TExec(
         ".updateAxis",
         TString::Format("auto h1 = (TH1*)%p; auto h2 = (TH1*)%p; if(h2->GetXaxis()->GetFirst() != "
                         "h1->GetXaxis()->GetFirst() || h1->GetXaxis()->GetLast()!=h2->GetXaxis()->GetLast()) "
                         "{h2->GetXaxis()->SetRange(h1->GetXaxis()->GetFirst(),h1->GetXaxis()->GetLast());if(gPad) "
                         "{gPad->GetCanvas()->Paint();gPad->GetCanvas()->Update();}}",
                         (void *)ratioHist, (void *)(h))));
      ratioHist->Draw((errHist ? "e2" : ""));

      _tmpPad->cd();
      ratioPad->Draw();
   } else if (auto ratioPad = dynamic_cast<TPad *>(gPad->GetPrimitive("auxPad")); hasSame && ratioPad) {
      // need to draw histogram in the ratio pad ...
      // if doing overlay need to update histogram

      if (auto hr = dynamic_cast<TH1 *>(ratioPad->GetPrimitive("auxHist"));
          hr && auxFunctions.find(hr->GetYaxis()->GetTitle()) != auxFunctions.end()) {
         TString histName = hr->GetTitle(); // split it by | char
         TString histType = histName(histName.Index('|') + 1, histName.Length());
         histName = histName(0, histName.Index('|'));

         if (auto hnom = dynamic_cast<TH1 *>(gPad->GetPrimitive(histName)); hnom) {
            h = dynamic_cast<TH1 *>(h->Clone(h->GetName()));
            h->SetDirectory(nullptr);
            h->SetBit(kCanDelete);
            for (int i = 1; i <= hnom->GetNbinsX(); i++) {
               double val = h->GetBinContent(i);
               double err = h->GetBinError(i);
               h->SetBinContent(i, std::get<0>(auxFunctions[hr->GetYaxis()->GetTitle()])(
                                      h->GetBinContent(i), hnom->GetBinContent(i), hnom->GetBinError(i)));
               h->SetBinError(i, std::get<0>(auxFunctions[hr->GetYaxis()->GetTitle()])(
                                    val + err, hnom->GetBinContent(i), hnom->GetBinError(i)) -
                                    h->GetBinContent(i));
            }
            auto _tmpPad = gPad;
            ratioPad->cd();
            if (hasOverlay) {
               if (auto existing = dynamic_cast<TH1 *>(ratioPad->GetPrimitive(h->GetName())); existing) {
                  existing->Reset();
                  existing->Add(h);
                  delete h;
                  h = existing;
                  overlayExisted = true;
               } else {
                  h->Draw(dOpt);
               }
            } else {
               h->Draw(dOpt);
            }
            double ymax = -std::numeric_limits<double>::infinity();
            double ymin = std::numeric_limits<double>::infinity();
            for (int i = 1; i <= h->GetNbinsX(); i++) {
               ymax = std::max(ymax, h->GetBinContent(i) + h->GetBinError(i));
               ymin = std::min(ymin, h->GetBinContent(i) - h->GetBinError(i));
            }
            adjustYRange(ymin, ymax, hr, std::get<1>(auxFunctions[hr->GetYaxis()->GetTitle()]));
            // adjustYRange(h->GetMinimum() * (h->GetMinimum()<0 ? 1.1 : 0.9), h->GetMaximum() * (h->GetMinimum()<0 ?
            // 0.9 : 1.1), hr, std::get<1>(auxFunctions[hr->GetYaxis()->GetTitle()]));
            gPad->Modified();
            _tmpPad->cd();
         }
      }
   }

   // see if it's in a simultaneous so need to select a cat
   /*auto _parent = fParent;
   auto _me = rar;
   while(_parent) {
       if (auto s = _parent->get<RooSimultaneous>(); s) {
           for (auto c : s->indexCat()) {
               if (auto p = s->getPdf(c.first.c_str());_me==p) {
                   gPad->SetName(c.first.c_str());
                   break;
               }
           }
           break;
       }
       _me = _parent->get<RooAbsReal>();
       _parent = _parent->fParent;
   }*/

   // now draw selected datasets on top if this was a pdf
   if (auto _pdf = get<RooAbsPdf>();
       !hasSame && _pdf /*&& (_pdf->canBeExtended() || robs().empty())*/ && coefs(true).empty()) {
      auto _dsets = datasets();
      // bool _drawn=false;
      for (auto &d : _dsets) {
         if (d->get()->TestBit(1 << 20)) {
            d->Draw("same");
            //_drawn=true;
         }
      }
      // if (!_drawn && !_dsets.empty()) _dsets[0]->Draw("same"); // always draw if has a dataset
   }

   gPad->Modified();
   // gPad->Update();
   getLegend();
   gPad->Modified();
   // gPad->Update();
}

void xRooNode::SaveAs(const char *filename, Option_t *option) const
{
   TString sOpt(option);
   sOpt.ToLower();
   if (auto w = get<RooWorkspace>(); w) {
      // ensure the current color set is saved in the workspace
      w->import(*gROOT->GetListOfColors(), true);

      if (TString(filename).EndsWith(".json")) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
         // stream with json tool
         RooJSONFactoryWSTool tool(*w);
         if (tool.exportJSON(filename)) {
            Info("SaveAs", "%s saved to %s", w->GetName(), filename);
         } else {
            Error("SaveAs", "Unable to save to %s", filename);
         }
#else
         Error("SaveAs", "json format workspaces only in ROOT 6.26 onwards");
#endif
         return;
      }

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      // before saving, clear the eocache of all owned nodes
      // because causes memory leak when read back in (workspace streamer immediately overwrites the caches)
      // fixed in: https://github.com/root-project/root/pull/12024
      for (auto &c : w->components()) {
         c->_eocache = nullptr;
      }
#endif
      // const_cast<Node2*>(this)->sterilize(); - tried this to reduce mem leak on readback but no improve
      if (!w->writeToFile(filename, sOpt != "update")) {
         Info("SaveAs", "%s saved to %s", w->GetName(), filename);
         // save any fitDatabase that is loaded in memory too
         // TODO: We should do this as well for SaveAs on a scan object
         if (auto fitDb = dynamic_cast<TFile *>(gROOT->GetListOfFiles()->FindObject("fitDatabase"))) {

            std::function<void(TDirectory *, TDirectory *)> CopyDir;

            CopyDir = [&](TDirectory *source, TDirectory *dest) {
               auto dir = dest->GetDirectory(source->GetName());
               if (!dir) {
                  dir = dest->mkdir(source->GetName());
               }
               for (auto k : *source->GetListOfKeys()) {
                  auto key = dynamic_cast<TKey *>(k);
                  const char *classname = key->GetClassName();
                  TClass *cl = gROOT->GetClass(classname);
                  // std::cout << "processing " << key->GetName() << " " << classname << std::endl;
                  if (!cl) {
                     continue;
                  } else if (cl->InheritsFrom(TDirectory::Class())) {
                     CopyDir(source->GetDirectory(key->GetName()), dir);
                  } else {
                     // don't write object if it already exists
                     if (dir->FindKey(key->GetName()))
                        continue;
                     // support FitConfigs ....
                     if (strcmp(classname, "ROOT::Fit::FitConfig") == 0) {
                        auto fc = key->ReadObject<ROOT::Fit::FitConfig>();
                        dir->WriteObject(fc, key->GetName());
                        delete fc;
                     } else {
                        TObject *obj = key->ReadObj();
                        if (obj) {
                           dir->WriteTObject(obj, key->GetName());
                           delete obj;
                        }
                     }
                  }
               }
            };
            CopyDir(fitDb, std::make_unique<TFile>(filename, "UPDATE").get());
            Info("SaveAs", "Saved fitDatabase to %s", filename);
         }

      } else {
         Error("SaveAs", "Unable to save to %s", filename);
      }
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      // restore the cache to every node
      for (auto &c : w->components()) {
         c->setExpensiveObjectCache(w->expensiveObjectCache());
      }
#endif
   }
}

double xRooNode::GetBinError(int bin, const xRooNode &fr, int nToys, bool errorsHi, bool errorsLo) const
{
   auto res = GetBinErrors(bin, bin, fr, nToys, errorsHi, errorsLo);
   if (res.empty())
      return std::numeric_limits<double>::quiet_NaN();
   return res.at(0);
}

std::vector<double> xRooNode::contents() const
{
   std::vector<double> out;
   out.reserve(size());
   for (auto child : *this) {
      out.push_back(child->GetContent());
   }
   return out;
}

TMatrixDSym xRooNode::covariances(const xRooNode &fr) const
{

   auto _fr = fr.get<RooFitResult>();

   if (!_fr) {
      return covariances(fitResult());
   }

   auto rho = _fr->correlationMatrix();

   TMatrixDSym out(size());

   auto _pars = pars();

   // formula for covariance is: C_ij = sum_m[sum_n[ (1/2)(nu[theta_m_up] - nu[theta_m_down])rho_mn(1/2)(nu[theta_n_up]
   // - nu[theta_n_down]) consistent with propagatedError formula

   for (int m = 0; m < rho.GetNrows(); m++) {
      auto p_m = dynamic_cast<RooRealVar *>(_fr->floatParsFinal().at(m));
      if (!p_m)
         continue; // skip categoricals
      auto _p = dynamic_cast<RooAbsRealLValue *>(_pars.get<RooArgList>()->find(p_m->GetName()));
      if (!_p)
         continue;
      auto tmp = _p->getVal();
      _p->setVal(p_m->getVal() + p_m->getErrorHi());
      auto nu_m = contents();
      _p->setVal(p_m->getVal() + p_m->getErrorLo());
      auto nu_m2 = contents();
      _p->setVal(tmp);
      for (int n = 0; n < rho.GetNrows(); n++) {
         auto p_n = dynamic_cast<RooRealVar *>(_fr->floatParsFinal().at(n));
         if (!p_n)
            continue; // skip categoricals
         auto _p2 = dynamic_cast<RooAbsRealLValue *>(_pars.get<RooArgList>()->find(p_n->GetName()));
         if (!_p2)
            continue;
         auto tmp2 = _p2->getVal();
         _p2->setVal(p_n->getVal() + p_n->getErrorHi());
         auto nu_n = (p_n == p_m) ? nu_m : contents();
         _p2->setVal(p_n->getVal() + p_n->getErrorLo());
         auto nu_n2 = (p_n == p_m) ? nu_m2 : contents();
         _p2->setVal(tmp2);
         for (int i = 0; i < out.GetNrows(); i++) {
            for (int j = 0; j < out.GetNrows(); j++) {
               out(i, j) += 0.25 * (nu_m[i] - nu_m2[i]) * rho(m, n) * (nu_n[j] - nu_n2[j]);
            }
         }
      }
   }
   return out;
}

std::pair<double, double> xRooNode::IntegralAndError(const xRooNode &fr, const char *rangeName) const
{
   double out = 1.;
   double err = std::numeric_limits<double>::quiet_NaN();

   std::unique_ptr<RooAbsCollection> _snap;
   RooArgList _pars;
   if (auto _fr = fr.get<RooFitResult>()) {
      _pars.add(pars().argList());
      _snap.reset(_pars.snapshot());
      _pars = _fr->floatParsFinal();
      _pars = _fr->constPars();
   }

   auto _obs = obs();
   auto _coefs = coefs(); // need here to keep alive owned RooProduct
   if (auto c = _coefs.get<RooAbsReal>(); c) {
      out = c->getVal(*_obs.get<RooArgList>()); // assumes independent of observables!
   }

   if (auto p = dynamic_cast<RooAbsPdf *>(get()); p) {
      // prefer to use expectedEvents for integrals of RooAbsPdf e.g. for RooProdPdf wont include constraint terms
      if (rangeName)
         p->setNormRange(rangeName);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 27, 00)
      RooAbsReal::EvalErrorContext _tmp(RooAbsReal::Ignore);
#endif
      out *= p->expectedEvents(*_obs.get<RooArgList>());
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      // improved normSet invalidity checking, so assuming no longer need this in 6.28 onwards
      p->_normSet = nullptr;
#endif
      err = GetBinError(-1, fr);
      if (rangeName)
         p->setNormRange(nullptr);
   } else if (auto p2 = dynamic_cast<RooAbsReal *>(get()); p2) {
      // only integrate over observables we actually depend on
      auto f = std::shared_ptr<RooAbsReal>(
         p2->createIntegral(*std::unique_ptr<RooArgSet>(p2->getObservables(*_obs.get<RooArgList>())),
                            rangeName)); // did use x here before using obs
      RooProduct pr("int_x_coef", "int_x_coef",
                    RooArgList(*f, !_coefs.get() ? RooFit::RooConst(1) : *_coefs.get<RooAbsReal>()));
      out *= f->getVal();
      err = xRooNode(pr, *this).GetBinError(-1, fr);
      sterilize(); // needed so that we can forget properly about the integral we just created (and are deleting)
   } else if (get<RooAbsData>()) {
      out = 0;
      auto vals = GetBinContents(1, 0); // returns all bins
      auto ax = (rangeName) ? GetXaxis() : nullptr;
      auto rv = (ax) ? dynamic_cast<RooRealVar *>(ax->GetParent()) : nullptr;
      auto cv = (ax && !rv) ? dynamic_cast<RooCategory *>(ax->GetParent()) : nullptr;
      int i = 0;
      for (auto &v : vals) {
         i++;
         if (rangeName) {
            if (rv && !rv->inRange(ax->GetBinCenter(i), rangeName))
               continue;
            if (cv && !cv->isStateInRange(rangeName, ax->GetBinLabel(i)))
               continue;
         }
         out += v;
      }
      err = 0; // should this be sqrt(sum(v^2)) or something similar
   } else {
      out = std::numeric_limits<double>::quiet_NaN();
   }
   if (_snap) {
      _pars.RooAbsCollection::operator=(*_snap);
   }
   return std::make_pair(out, err);
}

std::vector<double>
xRooNode::GetBinErrors(int binStart, int binEnd, const xRooNode &_fr, int nToys, bool errorHi, bool errorLo) const
{
   // note: so far this method is inconsistent with the BuildHistogram in ways:
   // no projection over other variables
   // July2023: made RooRealSumPdf evaluate as a function if doesn't have a floor
   // but this method will still evaluate it as a pdf (uses PdfWrapper)
   // but can get away with it while added NaN recovery to getSimplePropagatedError to pickup raw values

   if (fBinNumber != -1) {
      if (binStart != binEnd || !fParent) {
         throw std::runtime_error(TString::Format("%s is a bin - only has one value", GetName()));
      }
      return fParent->GetBinErrors(fBinNumber, fBinNumber, _fr);
   }

   std::vector<double> out;

   auto _hist = BuildHistogram(nullptr, true, true, binStart, binEnd, _fr, errorHi, errorLo, nToys);
   if (!_hist)
      return out;
   if (binEnd == 0) {
      binEnd = _hist->GetNbinsX();
   } else if (binEnd == binStart && binEnd == -1) {
      binStart = 1;
      binEnd = 1; // done an integral, so histogram has only 1 bin
   }
   for (int bin = binStart; bin <= binEnd; bin++) {
      out.push_back(((errorLo && !errorHi) ? (-1.) : 1.) *
                    _hist->GetBinError(bin)); // using same convention as RooFit that Lo errors are negative
   }
   delete _hist;
   return out;

   //   auto o = dynamic_cast<RooAbsReal *>(get());
   //   if (!o)
   //      return out;
   //
   //   std::shared_ptr<RooFitResult> fr = std::dynamic_pointer_cast<RooFitResult>(_fr.fComp);
   //   //= dynamic_cast<RooFitResult*>( _fr.get<RooFitResult>() ? _fr->Clone() : fitResult()->Clone());
   //
   //   auto _coefs = coefs();
   //
   //   if (!fr) {
   //      // need to ensure coefs, if any, are included in fit result retrieval so all pars are loaded
   //      auto frn = (!_coefs.get() ? *this : xRooNode(RooProduct("tmp", "tmp", RooArgList(*o,
   //      *_coefs.get<RooAbsReal>()))))
   //                    .fitResult();
   //      if (strlen(_fr.GetName()))
   //         frn = frn.reduced(_fr.GetName());
   //
   //      // use name to reduce the fit result, if one given
   //      fr = std::dynamic_pointer_cast<RooFitResult>(frn.fComp);
   //   }
   //
   //   if (!GETDMP(fr.get(), _finalPars)) {
   //      fr->setFinalParList(RooArgList());
   //   }
   //
   //   /// Oct2022: No longer doing this because want to allow fitResult to be used to get partial error
   //   //    // need to add any floating parameters not included somewhere already in the fit result ...
   //   //    RooArgList l;
   //   //    for(auto& p : pars()) {
   //   //        auto v = p->get<RooRealVar>();
   //   //        if (!v) continue;
   //   //        if (v->isConstant()) continue;
   //   //        if (fr->floatParsFinal().find(v->GetName())) continue;
   //   //        if (fr->_constPars && fr->_constPars->find(v->GetName())) continue;
   //   //        l.add(*v);
   //   //    }
   //   //
   //   //    if (!l.empty()) {
   //   //        RooArgList l2; l2.addClone(fr->floatParsFinal());
   //   //        l2.addClone(l);
   //   //        fr->setFinalParList(l2);
   //   //    }
   //
   //   TMatrixTSym<double> *prevCov = static_cast<TMatrixTSym<double> *>(GETDMP(fr.get(), _VM));
   //
   //   if (!prevCov || size_t(prevCov->GetNcols()) < fr->floatParsFinal().size()) {
   //      TMatrixDSym cov(fr->floatParsFinal().size());
   //      if (prevCov) {
   //         for (int i = 0; i < prevCov->GetNcols(); i++) {
   //            for (int j = 0; j < prevCov->GetNrows(); j++) {
   //               cov(i, j) = (*prevCov)(i, j);
   //            }
   //         }
   //      }
   //      int i = 0;
   //      for (auto &p : fr->floatParsFinal()) {
   //         if (!prevCov || i >= prevCov->GetNcols()) {
   //            cov(i, i) = pow(dynamic_cast<RooRealVar *>(p)->getError(), 2);
   //         }
   //         i++;
   //      }
   //      int covQualBackup = fr->covQual();
   //      fr->setCovarianceMatrix(cov);
   //      fr->setCovQual(covQualBackup);
   //   }
   //
   //   bool doBinWidth = false;
   //   auto ax = (binStart == -1 && binEnd == -1) ? nullptr : GetXaxis();
   //
   //   auto _obs = obs(); // may own an obs so keep alive here
   //   RooArgList normSet = _obs.argList();
   //   // to give consistency with BuildHistogram method, should be only the axis var if defined
   //   if (ax) {
   //      normSet.clear();
   //      normSet.add(*dynamic_cast<RooAbsArg *>(ax->GetParent()));
   //   }
   //
   //   if (auto p = dynamic_cast<RooAbsPdf *>(o); ax && (p || _coefs.get() || o->getAttribute("density"))) {
   //      // pdfs of samples embedded in a sumpdf (aka have a coef) will convert their density value to a content
   //      doBinWidth = true;
   //   }
   //   if (binEnd == 0) {
   //      if (ax) {
   //         binEnd = ax->GetNbins();
   //      } else {
   //         binEnd = binStart;
   //      }
   //   }
   //   for (int bin = binStart; bin <= binEnd; bin++) {
   //      if (ax)
   //         dynamic_cast<RooAbsLValue *>(ax->GetParent())->setBin(bin - 1, ax->GetName());
   //      // if (!SetBin(bin)) { return out; }
   //
   //      double res;
   //      if (auto p = dynamic_cast<RooAbsPdf *>(o); p) {
   //         // fr->covarianceMatrix().Print();
   //         res = PdfWrapper(*p, _coefs.get<RooAbsReal>(), !ax).getSimplePropagatedError(*fr, normSet);
   // #if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
   //         // improved normSet invalidity checking, so assuming no longer need this in 6.28 onwards
   //         p->_normSet = nullptr;
   // #endif
   //      } else {
   //         //         res = o->getPropagatedError(*fr, normSet);
   //         //         // TODO: What if coef has error? - probably need a FuncWrapper class
   //         //         if (auto c = _coefs.get<RooAbsReal>(); c) {
   //         //            res *= c->getVal(normSet);
   //         //         }
   //         res = RooProduct("errorEval", "errorEval",
   //                          RooArgList(*o, !_coefs.get() ? RooFit::RooConst(1) : *_coefs.get<RooAbsReal>()))
   //                  .getPropagatedError(*fr, normSet);
   //      }
   //      if (doBinWidth) {
   //         res *= ax->GetBinWidth(bin);
   //      }
   //      out.push_back(res);
   //   }
   //
   //   return out;
}

std::string cling::printValue(const xRooNode *v)
{
   if (!v)
      return "nullptr\n";
   if (!v->empty()) {
      std::string out;
      size_t left = v->size();
      for (auto n : *v) {
         left--;
         if (!out.empty())
            out += ",";
         else
            out += "{";
         out += n->GetName();
         if (out.length() > 100 && left > 0) {
            out += TString::Format(",... and %lu more", left);
            break;
         }
      }
      out += "}\n";
      out = std::string(Form("<%s> %s", v->get() ? v->get()->ClassName() : "nullptr", v->GetName())) + out;
      return out;
   }
   std::string out;
   if (!(*v)) {
      return "<nullptr>";
   } else {
      return Form("<%s> %s", v->get() ? v->get()->ClassName() : "nullptr", v->GetName());
   }

   return out;
}

END_XROOFIT_NAMESPACE
