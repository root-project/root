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
#include "TWebCanvas.h"

using namespace ROOT::Experimental;

using namespace std::string_literals;


class RBrowserTCanvasWidget : public RBrowserWidget {

   std::unique_ptr<TCanvas> fCanvas; ///<! drawn canvas
   TWebCanvas *fWebCanvas{nullptr};  ///<! web implementation, owned by TCanvas

   std::unique_ptr<Browsable::RHolder> fObject; // TObject drawing in the TCanvas

   void SetPrivateCanvasFields(bool on_init)
   {
      Long_t offset = TCanvas::Class()->GetDataMemberOffset("fCanvasID");
      if (offset > 0) {
         Int_t *id = (Int_t *)((char*) fCanvas.get() + offset);
         if (*id == fCanvas->GetCanvasID()) *id = on_init ? 111222333 : -1;
      } else {
         printf("ERROR: Cannot modify TCanvas::fCanvasID data member\n");
      }

      offset = TCanvas::Class()->GetDataMemberOffset("fPixmapID");
      if (offset > 0) {
         Int_t *id = (Int_t *)((char*) fCanvas.get() + offset);
         if (*id == fCanvas->GetPixmapID()) *id = on_init ? 332211 : -1;
      } else {
         printf("ERROR: Cannot modify TCanvas::fPixmapID data member\n");
      }

      offset = TCanvas::Class()->GetDataMemberOffset("fMother");
      if (offset > 0) {
         TPad **moth = (TPad **)((char*) fCanvas.get() + offset);
         if (*moth == fCanvas->GetMother()) *moth = on_init ? fCanvas.get() : nullptr;
      } else {
         printf("ERROR: Cannot set TCanvas::fMother data member\n");
      }
   }

public:

   RBrowserTCanvasWidget(const std::string &name) : RBrowserWidget(name)
   {
      fCanvas = std::make_unique<TCanvas>(kFALSE);
      fCanvas->SetName(name.c_str());
      fCanvas->SetTitle(name.c_str());
      fCanvas->ResetBit(TCanvas::kShowEditor);
      fCanvas->ResetBit(TCanvas::kShowToolBar);
      fCanvas->SetBit(TCanvas::kMenuBar, kTRUE);
      fCanvas->SetCanvas(fCanvas.get());
      fCanvas->SetBatch(kTRUE); // mark canvas as batch
      fCanvas->SetEditable(kTRUE); // ensure fPrimitives are created

      // create implementation
      fWebCanvas = new TWebCanvas(fCanvas.get(), "title", 0, 0, 800, 600);

      // assign implementation
      fCanvas->SetCanvasImp(fWebCanvas);

      SetPrivateCanvasFields(true);
      fCanvas->cd();
   }

   RBrowserTCanvasWidget(const std::string &name, std::unique_ptr<TCanvas> &canv) : RBrowserWidget(name)
   {
      fCanvas = std::move(canv);
      fCanvas->SetBatch(kTRUE); // mark canvas as batch

      // create implementation
      fWebCanvas = new TWebCanvas(fCanvas.get(), "title", 0, 0, 800, 600);

      // assign implementation
      fCanvas->SetCanvasImp(fWebCanvas);
      SetPrivateCanvasFields(true);
      fCanvas->cd();
   }

   virtual ~RBrowserTCanvasWidget()
   {
      SetPrivateCanvasFields(false);

      gROOT->GetListOfCanvases()->Remove(fCanvas.get());

      fCanvas->Close();
   }

   std::string GetKind() const override { return "tcanvas"s; }

   void SetActive() override
   {
      fCanvas->cd();
   }

   void Show(const std::string &arg) override
   {
      fWebCanvas->ShowWebWindow(arg);
   }

   std::string GetUrl() override
   {
      return "../"s + fWebCanvas->GetWebWindow()->GetAddr() + "/"s;
   }

   std::string GetTitle() override
   {
      return fCanvas->GetName();
   }

   bool DrawElement(std::shared_ptr<Browsable::RElement> &elem, const std::string &opt) override
   {
      if (!elem->IsCapable(Browsable::RElement::kActDraw6))
         return false;

      fObject = elem->GetObject();

      if (fObject && Browsable::RProvider::Draw6(fCanvas.get(), fObject, opt)) {
         fCanvas->ForceUpdate();
         return true;
      }

      return false;
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

public:
   RBrowserTCanvasProvider() : RBrowserWidgetProvider("tcanvas") {}
   ~RBrowserTCanvasProvider() = default;
} sRBrowserTCanvasProvider;

