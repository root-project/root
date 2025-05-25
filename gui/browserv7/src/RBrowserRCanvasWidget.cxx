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

#include <ROOT/RCanvas.hxx>

using namespace std::string_literals;
using namespace ROOT::Browsable;


class RBrowserRCanvasWidget : public ROOT::RBrowserWidget {

   std::shared_ptr<ROOT::Experimental::RCanvas> fCanvas; ///<! drawn canvas

public:

   RBrowserRCanvasWidget(const std::string &name) : ROOT::RBrowserWidget(name)
   {
      fCanvas = ROOT::Experimental::RCanvas::Create(name);

      // ensure creation of web window
      fCanvas->Show("embed");
   }

   RBrowserRCanvasWidget(const std::string &name, std::shared_ptr<ROOT::Experimental::RCanvas> &canv, bool catched_canvas = false) : ROOT::RBrowserWidget(name)
   {
      if (catched_canvas) {
         fCanvas = canv;
      } else {
         fCanvas = std::move(canv);
         // ensure creation of web window
         fCanvas->Show("embed");
      }
   }

   ~RBrowserRCanvasWidget() override = default;

   std::string GetKind() const override { return "rcanvas"s; }

   std::shared_ptr<ROOT::RWebWindow> GetWindow() override { return fCanvas->GetWindow(); }

   std::string GetTitle() override { return fCanvas->GetTitle(); }

   bool DrawElement(std::shared_ptr<RElement> &elem, const std::string &opt = "") override
   {
      if (!elem->IsCapable(RElement::kActDraw7))
         return false;

      auto obj = elem->GetObject();
      if (!obj)
         return false;

      RProvider::ExtendProgressHandle(elem.get(), obj.get());

      std::shared_ptr<ROOT::Experimental::RPadBase> subpad = fCanvas;

      std::string drawopt = opt;

      if (drawopt.compare(0,8,"<append>") == 0) {
         drawopt.erase(0,8);
      } else if (subpad->NumPrimitives() > 0) {
         subpad->Wipe();
         fCanvas->Modified();
         fCanvas->Update(true);
      }

      if (drawopt == "<dflt>")
         drawopt = RProvider::GetClassDrawOption(obj->GetClass());

      if (RProvider::Draw7(subpad, obj, drawopt)) {
         fCanvas->Modified();
         fCanvas->Update(true);
         return true;
      }

      return false;
   }

   void CheckModified() override
   {
      if (fCanvas->IsModified())
         fCanvas->Update();
   }

};

// ======================================================================

class RBrowserRCanvasProvider : public ROOT::RBrowserWidgetProvider {
protected:
   std::shared_ptr<ROOT::RBrowserWidget> Create(const std::string &name) final
   {
      return std::make_shared<RBrowserRCanvasWidget>(name);
   }

   std::shared_ptr<ROOT::RBrowserWidget> CreateFor(const std::string &name, std::shared_ptr<RElement> &elem) final
   {
      auto holder = elem->GetObject();
      if (!holder) return nullptr;

      auto canv = holder->get_shared<ROOT::Experimental::RCanvas>();
      if (!canv) return nullptr;

      return std::make_shared<RBrowserRCanvasWidget>(name, canv);
   }

   std::shared_ptr<ROOT::RBrowserWidget> DetectWindow(ROOT::RWebWindow &win) final
   {
      for(auto & canv : ROOT::Experimental::RCanvas::GetCanvases()) {
         if (canv->GetWindow().get() == &win)
            return std::make_shared<RBrowserRCanvasWidget>(canv->GetTitle(), const_cast<std::shared_ptr<ROOT::Experimental::RCanvas> &>(canv), true);
      }

      return nullptr;
   }


public:
   RBrowserRCanvasProvider() : ROOT::RBrowserWidgetProvider("rcanvas") {}
   ~RBrowserRCanvasProvider() override = default;
} sRBrowserRCanvasProvider;
