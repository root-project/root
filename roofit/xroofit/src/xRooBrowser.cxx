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

#include "xRooFit/xRooBrowser.h"
#include "xRooFit/xRooNode.h"

#include "TSystem.h"
#include "TKey.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TFile.h"
#include "RooWorkspace.h"

#ifdef XROOFIT_NAMESPACE
namespace XROOFIT_NAMESPACE {
#endif

xRooBrowser::xRooBrowser(xRooNode* o) : TBrowser("RooBrowser", o, "RooFit Browser"), fTopNode(o) {


    fNode = std::shared_ptr<xRooNode>(fTopNode.get(),[](xRooNode*){});

    if (fTopNode) {
        fTopNode->fBrowseOperation = [](xRooNode* in) {
            for (auto file : *gROOT->GetListOfFiles()) {
                auto _file = dynamic_cast<TFile *>(file);
                auto keys = _file->GetListOfKeys();
                if (keys) {
                    for (auto &&k : *keys) {
                        auto cl = TClass::GetClass(((TKey *) k)->GetClassName());
                        if (cl == RooWorkspace::Class() || cl->InheritsFrom("RooWorkspace")) {
                            if (auto w = _file->Get<RooWorkspace>(k->GetName()); w) {
                                if (!in->contains(_file->GetName())) {
                                    in->emplace_back(std::make_shared<xRooNode>(*_file));
                                }
                                if (!in->at(_file->GetName())->contains(w->GetName())) {
                                    in->at(_file->GetName())->emplace_back(
                                            std::make_shared<xRooNode>(*w, in->at(_file->GetName())));
                                }
                            }
                        }
                    }
                }
            }
            return *in;
        };
    }

}

xRooBrowser::xRooBrowser() :xRooBrowser([]() {
    gEnv->SetValue("X11.UseXft","no"); // for faster x11
    gEnv->SetValue("X11.Sync","no");
    gEnv->SetValue("X11.FindBestVisual","no");
    gEnv->SetValue("Browser.Name","TRootBrowser"); // forces classic root browser (in 6.26 onwards)
    return new xRooNode("!Workspaces"); }()) {
/*
    fNode = std::shared_ptr<xRooNode>(dynamic_cast<xRooNode*>(GetSelected()),[](xRooNode*){});

    if (fNode) {
        for (auto file : *gROOT->GetListOfFiles()) {
            auto _file = dynamic_cast<TFile *>(file);
            auto keys = _file->GetListOfKeys();
            if (keys) {
                for (auto &&k : *keys) {
                    auto cl = TClass::GetClass(((TKey *) k)->GetClassName());
                    if (cl == RooWorkspace::Class() || cl->InheritsFrom("RooWorkspace")) {
                        if (auto w = _file->Get<RooWorkspace>(k->GetName()); w) {
                            if (!fNode->contains(_file->GetName())) {
                                fNode->emplace_back(std::make_shared<xRooNode>(*_file));
                            }
                            if (!fNode->at(_file->GetName())->contains(w->GetName())) {
                                fNode->at(_file->GetName())->emplace_back(
                                        std::make_shared<xRooNode>(*w, fNode->at(_file->GetName())));
                            }
                        }
                    }
                }
            }
        }
    }
    */
}

void xRooBrowser::ls(const char* path) const {
   if (!fNode) return;
   if (!path) fNode->Print();
   else {
      // will throw exception if not found
      fNode->at(path)->Print();
   }
}

void xRooBrowser::cd(const char* path) {
   auto _node = fNode->at(path); // throws exception if not found
   fNode = _node;
}

xRooNode* xRooBrowser::GetSelected() { return dynamic_cast<xRooNode*>(TBrowser::GetSelected()); }

#ifdef XROOFIT_NAMESPACE
}
#endif