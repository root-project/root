// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2021-01-25
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RBrowserWidget.hxx"

#include <ROOT/Browsable/RProvider.hxx>

#include "TCanvas.h"
#include "TROOT.h"
#include "TClass.h"
#include "TEnv.h"
#include "TWebCanvas.h"

#include <map>

using namespace ROOT;

using namespace std::string_literals;


class RBrowserTCanvasWidget : public RBrowserWidget {

   TString fCanvasName; ///<! canvas name
   TCanvas *fCanvas{nullptr}; ///<! last canvas pointer
   TWebCanvas *fWebCanvas{nullptr};  ///<! web implementation, owned by TCanvas

   std::multimap<TVirtualPad *, std::unique_ptr<Browsable::RHolder>> fObjects; ///<! objects holder, associated with pads

   bool CheckCanvasPointer()
   {
      if (!fCanvas)
         return false;

      auto c = gROOT->GetListOfCanvases()->FindObject(fCanvasName.Data());
      if (c && (fCanvas == c))
         return true;

      fCanvas = nullptr;
      return false;
   }

   void SetPrivateCanvasFields(bool on_init)
   {
      Long_t offset = TCanvas::Class()->GetDataMemberOffset("fCanvasID");
      if (offset > 0) {
         Int_t *id = (Int_t *)((char *) fCanvas + offset);
         if (*id == fCanvas->GetCanvasID()) *id = on_init ? 111222333 : -1;
      } else {
         printf("ERROR: Cannot modify TCanvas::fCanvasID data member\n");
      }

      offset = TCanvas::Class()->GetDataMemberOffset("fPixmapID");
      if (offset > 0) {
         Int_t *id = (Int_t *)((char *) fCanvas + offset);
         if (*id == fCanvas->GetPixmapID()) *id = on_init ? 332211 : -1;
      } else {
         printf("ERROR: Cannot modify TCanvas::fPixmapID data member\n");
      }

      offset = TCanvas::Class()->GetDataMemberOffset("fMother");
      if (offset > 0) {
         TPad **moth = (TPad **)((char *) fCanvas + offset);
         if (*moth == fCanvas->GetMother()) *moth = on_init ? fCanvas : nullptr;
      } else {
         printf("ERROR: Cannot set TCanvas::fMother data member\n");
      }

      offset = TCanvas::Class()->GetDataMemberOffset("fCw");
      if (offset > 0) {
         UInt_t *cw = (UInt_t *)((char *) fCanvas + offset);
         if (*cw == fCanvas->GetWw()) *cw = on_init ? 800 : 0;
      } else {
         printf("ERROR: Cannot set TCanvas::fCw data member\n");
      }

      offset = TCanvas::Class()->GetDataMemberOffset("fCh");
      if (offset > 0) {
         UInt_t *ch = (UInt_t *)((char *) fCanvas + offset);
         if (*ch == fCanvas->GetWh()) *ch = on_init ? 600 : 0;
      } else {
         printf("ERROR: Cannot set TCanvas::fCw data member\n");
      }
   }

   void RegisterCanvasInGlobalLists()
   {
      R__LOCKGUARD(gROOTMutex);
      auto l1 = gROOT->GetListOfCleanups();
      if (!l1->FindObject(fCanvas))
         l1->Add(fCanvas);
      auto l2 = gROOT->GetListOfCanvases();
      if (!l2->FindObject(fCanvas))
         l2->Add(fCanvas);
   }

public:

   RBrowserTCanvasWidget(const std::string &name) : RBrowserWidget(name)
   {
      fCanvasName = name.c_str();

      fCanvas = new TCanvas(kFALSE);
      fCanvas->SetName(fCanvasName);
      fCanvas->SetTitle(fCanvasName);
      fCanvas->ResetBit(TCanvas::kShowEditor);
      fCanvas->ResetBit(TCanvas::kShowToolBar);
      fCanvas->SetBit(TCanvas::kMenuBar, kTRUE);
      fCanvas->SetCanvas(fCanvas);
      fCanvas->SetBatch(kTRUE); // mark canvas as batch
      fCanvas->SetEditable(kTRUE); // ensure fPrimitives are created

      Bool_t readonly = gEnv->GetValue("WebGui.FullCanvas", (Int_t) 1) == 0;

      // create implementation
      fWebCanvas = new TWebCanvas(fCanvas, "title", 0, 0, 800, 600, readonly);

      // use async mode to prevent blocking inside qt5/qt6/cef
      fWebCanvas->SetAsyncMode(kTRUE);

      // assign implementation
      fCanvas->SetCanvasImp(fWebCanvas);
      SetPrivateCanvasFields(true);
      fCanvas->cd();

      RegisterCanvasInGlobalLists();

      // ensure creation of web window
      fWebCanvas->ShowWebWindow("embed");
   }

   RBrowserTCanvasWidget(const std::string &name, std::unique_ptr<TCanvas> &canv) : RBrowserWidget(name)
   {
      fCanvas = canv.release();
      fCanvasName = fCanvas->GetName();
      fCanvas->SetBatch(kTRUE); // mark canvas as batch

      Bool_t readonly = gEnv->GetValue("WebGui.FullCanvas", (Int_t) 1) == 0;

      // create implementation
      fWebCanvas = new TWebCanvas(fCanvas, "title", 0, 0, 800, 600, readonly);

      // use async mode to prevent blocking inside qt5/qt6/cef
      fWebCanvas->SetAsyncMode(kTRUE);

      // assign implementation
      fCanvas->SetCanvasImp(fWebCanvas);
      SetPrivateCanvasFields(true);
      fCanvas->cd();

      RegisterCanvasInGlobalLists();

      // ensure creation of web window
      fWebCanvas->ShowWebWindow("embed");
   }

   RBrowserTCanvasWidget(const std::string &name, TCanvas *canv, TWebCanvas *webcanv) : RBrowserWidget(name)
   {
      fCanvas = canv;
      fCanvasName = fCanvas->GetName();
      fCanvas->SetBatch(kTRUE); // mark canvas as batch

      fWebCanvas = webcanv;
      // use async mode to prevent blocking inside qt5/qt6/cef
      fWebCanvas->SetAsyncMode(kTRUE);

      // assign implementation
      SetPrivateCanvasFields(true);
   }

   virtual ~RBrowserTCanvasWidget()
   {
      if (!fCanvas || !gROOT->GetListOfCanvases()->FindObject(fCanvas))
         return;

      {
         R__LOCKGUARD(gROOTMutex);
         gROOT->GetListOfCleanups()->Remove(fCanvas);
      }

      SetPrivateCanvasFields(false);

      gROOT->GetListOfCanvases()->Remove(fCanvas);

      if ((fCanvas->GetCanvasID() == -1) && (fCanvas->GetCanvasImp() == fWebCanvas)) {
         fCanvas->SetCanvasImp(nullptr);
         delete fWebCanvas;
      }

      fCanvas->Close();
      delete fCanvas;
   }

   std::string GetKind() const override { return "tcanvas"s; }

   void SetActive() override
   {
      if (CheckCanvasPointer())
         fCanvas->cd();
   }

   std::shared_ptr<RWebWindow> GetWindow() override
   {
      if (CheckCanvasPointer())
         return fWebCanvas->GetWebWindow();
      return nullptr;
   }

   std::string GetTitle() override { return fCanvasName.Data(); }

   bool DrawElement(std::shared_ptr<Browsable::RElement> &elem, const std::string &opt = "") override
   {
      if (!elem->IsCapable(Browsable::RElement::kActDraw6))
         return false;

      std::unique_ptr<Browsable::RHolder> obj = elem->GetObject();
      if (!obj)
         return false;

      if (!CheckCanvasPointer())
         return false;

      Browsable::RProvider::ExtendProgressHandle(elem.get(), obj.get());

      std::string drawopt = opt;

      // first remove all objects which may belong to removed pads
      bool find_removed_pad;
      do {
         find_removed_pad = false;
         for (auto &entry : fObjects)
            if ((entry.first != fCanvas) && !fCanvas->FindObject(entry.first)) {
               fObjects.erase(entry.first);
               find_removed_pad = true;
               break;
            }
      } while (find_removed_pad);

      TVirtualPad *pad = fCanvas;
      if (gPad && fCanvas->FindObject(gPad))
         pad = gPad;

      if (drawopt.compare(0,8,"<append>") == 0) {
         drawopt.erase(0,8);
      } else {
         pad->GetListOfPrimitives()->Clear();
         if (pad == fCanvas)
            fObjects.clear();
         else
            fObjects.erase(pad);
         pad->Range(0,0,1,1);  // set default range
      }

      if (drawopt == "<dflt>")
         drawopt = Browsable::RProvider::GetClassDrawOption(obj->GetClass());

      if (Browsable::RProvider::Draw6(pad, obj, drawopt)) {
         fObjects.emplace(pad, std::move(obj));
         pad->Modified();
         fCanvas->UpdateAsync();
         return true;
      }

      return false;
   }

   void CheckModified() override
   {
      if (CheckCanvasPointer() && fCanvas->IsModified())
         fCanvas->UpdateAsync();
   }

   bool IsValid() override
   {
      return CheckCanvasPointer();
   }

};

// ======================================================================

class RBrowserTCanvasProvider : public RBrowserWidgetProvider {
protected:
   std::shared_ptr<RBrowserWidget> Create(const std::string &name) final
   {
      return std::make_shared<RBrowserTCanvasWidget>(name);
   }

   std::shared_ptr<RBrowserWidget> CreateFor(const std::string &name, std::shared_ptr<Browsable::RElement> &elem) final
   {
      auto holder = elem->GetObject();
      if (!holder) return nullptr;

      auto canv = holder->get_unique<TCanvas>();
      if (!canv) return nullptr;

      return std::make_shared<RBrowserTCanvasWidget>(name, canv);
   }

   std::shared_ptr<RBrowserWidget> DetectWindow(RWebWindow &win) final
   {
      TIter iter(gROOT->GetListOfCanvases());

      while (auto canv = static_cast<TCanvas *>(iter())) {
         auto web_canv = dynamic_cast<TWebCanvas *>(canv->GetCanvasImp());
         if (web_canv->GetWebWindow().get() == &win)
            return std::make_shared<RBrowserTCanvasWidget>(canv->GetName(), canv, web_canv);
      }
      return nullptr;
   }

public:
   RBrowserTCanvasProvider() : RBrowserWidgetProvider("tcanvas") {}
   ~RBrowserTCanvasProvider() = default;
} sRBrowserTCanvasProvider;

