#include <RooBrowser.h>

#include "RooFit/Detail/RooNode.h"

#include <TSystem.h>
#include <TKey.h>
#include <TROOT.h>
#include <TEnv.h>
#include <TFile.h>
#include <RooWorkspace.h>

RooBrowser::RooBrowser(RooNode* o) : TBrowser("RooBrowser", o, "RooFit Browser"), fTopNode(o) {


    fNode = std::shared_ptr<RooNode>(fTopNode.get(),[](RooNode*){});

    if (fTopNode) {
        fTopNode->fBrowseOperation = [](RooNode* in) {
            for (auto file : *gROOT->GetListOfFiles()) {
                auto _file = dynamic_cast<TFile *>(file);
                auto keys = _file->GetListOfKeys();
                if (keys) {
                    for (auto &&k : *keys) {
                        auto cl = TClass::GetClass(((TKey *) k)->GetClassName());
                        if (cl == RooWorkspace::Class() || cl->InheritsFrom("RooWorkspace")) {
                            if (auto w = _file->Get<RooWorkspace>(k->GetName()); w) {
                                if (!in->contains(_file->GetName())) {
                                    in->emplace_back(std::make_shared<RooNode>(*_file));
                                }
                                if (!in->at(_file->GetName())->contains(w->GetName())) {
                                    in->at(_file->GetName())->emplace_back(
                                            std::make_shared<RooNode>(*w, in->at(_file->GetName())));
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

RooBrowser::RooBrowser() :RooBrowser([]() {
    gEnv->SetValue("X11.UseXft","no"); // for faster x11
    gEnv->SetValue("X11.Sync","no");
    gEnv->SetValue("X11.FindBestVisual","no");
    gEnv->SetValue("Browser.Name","TRootBrowser"); // forces classic root browser (in 6.26 onwards)
    return new RooNode("!Workspaces"); }()) {
/*
    fNode = std::shared_ptr<RooNode>(dynamic_cast<RooNode*>(GetSelected()),[](RooNode*){});

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
                                fNode->emplace_back(std::make_shared<RooNode>(*_file));
                            }
                            if (!fNode->at(_file->GetName())->contains(w->GetName())) {
                                fNode->at(_file->GetName())->emplace_back(
                                        std::make_shared<RooNode>(*w, fNode->at(_file->GetName())));
                            }
                        }
                    }
                }
            }
        }
    }
    */
}

RooBrowser::~RooBrowser() {}

RooNode* RooBrowser::GetSelected() { return dynamic_cast<RooNode*>(TBrowser::GetSelected()); }

void RooBrowser::ls(const char* path) {
    if (!fNode) return;
    if (!path) fNode->Print();
    else {
        // will throw exception if not found
        fNode->at(path)->Print();
    }
}

void RooBrowser::cd(const char* path) {
    auto _node = fNode->at(path); // throws exception if not found
    fNode = _node;
}
