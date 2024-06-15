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

/** \class ROOT::Experimental::XRooFit::xRooBrowser
\ingroup xroofit

 \image html RooBrowser.png width=50%

 \brief A version of the TBrowser that can be used to interact with RooFit models and datasets.

 Also available under the ROOT::Experimental::RooBrowser alias.

To get started with the xRooBrowser, open any ROOT file containing a workspace
 and then create an instance of the xRooBrowser just like
 creating an instance of a `TBrowser`. A window will be displayed with a navigable
 tree structure on the left that lets you explore the content of the workspaces
 present in the loaded file. Note that additional files, <b>including json workspace files</b>,
 can be loaded through the `Browser --> Open` menu in the top left corner.

The context menu for each node (access by right clicking on the node) in the tree structure can be used to get more
information about the node. In particular, the `Draw` command can be selected on many of the nodes that are part of a
statistical model, which will visualize that part of the model in the browser window. A number of options are available
for the `Draw` command, including (some options can be combined):

 - "e" : calculate and visualize propagated model uncertainty
 - "auxratio" : Draw a ratio auxiliary plot below the main plot
 - "auxsignif" : Draw a significance auxiliary plot below the main plot
 - "pull" : show panel of current parameter values, which can be dragged in order to change the values and visualize the
effect on the model (very experimental feature).

 Once a node has been drawn, the styling of subsequent draws can be controlled through `TStyle` objects
 that will now appear in the `objects` folder in the workspace.

A model can be fit to a dataset from the workspace using the `fitTo` context menu command and specifying
 the name of a dataset in the workspace (if no name is given, an expected dataset corresponding to the
 current state of the model will be used). A dialog will display the fit result status code when the
 fit completes and then a `fits` folder will be found under the workspace (the workspace may need to
 be collapsed and re-expanded to make it appear) where the fit result can be found, selected, and visualized.
 In multi-channel models the channels that are included in the fit can be controlled with the checkboxes
 in the browser. Clicking the checkbox will cycle through three states: checked, unchecked with
 grey-underline, and checked with grey-underline. The grey-underline indicates that channel wont be
 included in the fit (and will appear greyed out when the model is visualized)

Many more features are available in the xRooBrowser, and further documentation and development can be found at
 the <a href="https://gitlab.cern.ch/will/xroofit">xRooFit</a> repository, which is the library where the browser has
 been originally developed. The author (Will Buttinger) is also very happy to be contacted with questions or
 feedback about this new functionality.

 */

#include "xRooFit/xRooBrowser.h"
#include "xRooFit/xRooNode.h"

#include "TSystem.h"
#include "TKey.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TFile.h"
#include "RooWorkspace.h"
#include "TRootBrowser.h"
#include "TGMenu.h"
#include "TGFileDialog.h"
#include "TObjString.h"

#define GETPOPUPMENU(b, m)          \
   reinterpret_cast<TGPopupMenu *>( \
      *reinterpret_cast<void **>(reinterpret_cast<unsigned char *>(b) + b->Class()->GetDataMemberOffset(#m)))

BEGIN_XROOFIT_NAMESPACE

xRooBrowser::xRooBrowser(xRooNode *o) : TBrowser("RooBrowser", o, "RooFit Browser"), fTopNode(o)
{

   fNode = std::shared_ptr<xRooNode>(fTopNode.get(), [](xRooNode *) {});

   if (fTopNode) {
      fTopNode->fBrowseOperation = [](xRooNode *in) {
         for (auto file : *gROOT->GetListOfFiles()) {
            auto _file = dynamic_cast<TFile *>(file);
            auto keys = _file->GetListOfKeys();
            if (keys) {
               for (auto &&k : *keys) {
                  auto cl = TClass::GetClass((static_cast<TKey *>(k))->GetClassName());
                  if (cl == RooWorkspace::Class() || cl->InheritsFrom("RooWorkspace")) {
                     if (auto w = _file->Get<RooWorkspace>(k->GetName()); w) {
                        if (!in->contains(_file->GetName())) {
                           in->emplace_back(std::make_shared<xRooNode>(*_file));
                        }
                        if (!in->at(_file->GetName())->contains(w->GetName())) {
                           in->at(_file->GetName())
                              ->emplace_back(std::make_shared<xRooNode>(*w, in->at(_file->GetName())));
                        }
                     }
                  }
               }
            }
         }
         return *in;
      };
   }

   // override file menu event handling so that can intercept "Open"
   if (auto rb = dynamic_cast<TRootBrowser *>(GetBrowserImp())) {
      rb->Disconnect(GETPOPUPMENU(rb, fMenuFile), "Activated(Int_t)", rb, "HandleMenu(Int_t)");
      GETPOPUPMENU(rb, fMenuFile)->Connect("Activated(Int_t)", ClassName(), this, "HandleMenu(Int_t)");
   }
}

void xRooBrowser::HandleMenu(Int_t id)
{
   if (id == TRootBrowser::kOpenFile) {
      static TString dir(".");
      TGFileInfo fi;
      static const char *openFileTypes[] = {"ROOT files", "*.root", "JSON files", "*.json",
                                            "All files",  "*",      nullptr,      nullptr};
      fi.fFileTypes = openFileTypes;
      fi.SetIniDir(dir);
      new TGFileDialog(gClient->GetDefaultRoot(), dynamic_cast<TRootBrowser *>(GetBrowserImp()), kFDOpen, &fi);
      dir = fi.fIniDir;
      std::vector<std::string> filesToOpen;
      if (fi.fMultipleSelection && fi.fFileNamesList) {
         TObjString *el;
         TIter next(fi.fFileNamesList);
         while ((el = static_cast<TObjString *>(next()))) {
            filesToOpen.push_back(gSystem->UnixPathName(el->GetString()));
         }
      } else if (fi.fFilename) {
         filesToOpen.push_back(gSystem->UnixPathName(fi.fFilename));
      }
      if (!filesToOpen.empty()) {
         for (auto &f : filesToOpen) {
            if (TString(f.data()).EndsWith(".json")) {
               fTopNode->push_back(std::make_shared<xRooNode>(f.c_str()));
            } else {
               fTopNode->push_back(std::make_shared<xRooNode>(std::make_shared<TFile>(f.c_str())));
            }
         }
      }
   } else if (auto rb = dynamic_cast<TRootBrowser *>(GetBrowserImp())) {
      rb->HandleMenu(id);
   }
}

xRooBrowser::xRooBrowser()
   : xRooBrowser([]() {
        gEnv->SetValue("X11.UseXft", "no"); // for faster x11
        gEnv->SetValue("X11.Sync", "no");
        gEnv->SetValue("X11.FindBestVisual", "no");
        gEnv->SetValue("Browser.Name", "TRootBrowser"); // forces classic root browser (in 6.26 onwards)
        gEnv->SetValue("Canvas.Name", "TRootCanvas");
        return new xRooNode("!Workspaces");
     }())
{
}

xRooNode *xRooBrowser::Open(const char *filename)
{
   if (TString(filename).EndsWith(".root")) {
      return fTopNode->emplace_back(std::make_shared<xRooNode>(std::make_shared<TFile>(filename))).get();
   } else {
      return fTopNode->emplace_back(std::make_shared<xRooNode>(filename)).get();
   }
}

void xRooBrowser::ls(const char *path) const
{
   if (!fNode)
      return;
   if (!path) {
      fNode->Print();
   } else {
      // will throw exception if not found
      fNode->at(path)->Print();
   }
}

void xRooBrowser::cd(const char *path)
{
   auto _node = fNode->at(path); // throws exception if not found
   fNode = _node;
}

xRooNode *xRooBrowser::GetSelected()
{
   return dynamic_cast<xRooNode *>(TBrowser::GetSelected());
}

END_XROOFIT_NAMESPACE
