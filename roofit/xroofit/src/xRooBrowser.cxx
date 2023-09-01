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
#include "TRootBrowser.h"
#include "TGMenu.h"
#include "TGFileDialog.h"
#include "TObjString.h"

#define GETPOPUPMENU(b,m) ((TGPopupMenu*)(*(void**)(((unsigned char*)b) + b->Class()->GetDataMemberOffset(#m))))

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
                  auto cl = TClass::GetClass(((TKey *)k)->GetClassName());
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
      rb->Disconnect(GETPOPUPMENU(rb,fMenuFile), "Activated(Int_t)", rb, "HandleMenu(Int_t)");
      GETPOPUPMENU(rb,fMenuFile)->Connect("Activated(Int_t)", ClassName(), this, "HandleMenu(Int_t)");
   }
}

void xRooBrowser::HandleMenu(Int_t id)
{
   if (id == TRootBrowser::kOpenFile) {
      static TString dir(".");
      TGFileInfo fi;
      static const char *openFileTypes[] = {"ROOT files", "*.root", "JSON files", "*.json", "All files", "*", 0, 0};
      fi.fFileTypes = openFileTypes;
      fi.SetIniDir(dir);
      new TGFileDialog(gClient->GetDefaultRoot(), dynamic_cast<TRootBrowser *>(GetBrowserImp()), kFDOpen, &fi);
      dir = fi.fIniDir;
      std::vector<std::string> filesToOpen;
      if (fi.fMultipleSelection && fi.fFileNamesList) {
         TObjString *el;
         TIter next(fi.fFileNamesList);
         while ((el = (TObjString *)next())) {
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
        return new xRooNode("!Workspaces");
     }())
{
}

void xRooBrowser::ls(const char *path) const
{
   if (!fNode)
      return;
   if (!path)
      fNode->Print();
   else {
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
