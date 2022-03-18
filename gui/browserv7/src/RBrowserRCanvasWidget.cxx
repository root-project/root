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

using namespace ROOT::Experimental;

using namespace std::string_literals;


class RBrowserRCanvasWidget : public RBrowserWidget {

   std::shared_ptr<RCanvas> fCanvas; ///<! drawn canvas

public:

   RBrowserRCanvasWidget(const std::string &name) : RBrowserWidget(name)
   {
      fCanvas = RCanvas::Create(name);
   }

   RBrowserRCanvasWidget(const std::string &name, std::shared_ptr<RCanvas> &canv) : RBrowserWidget(name)
   {
      fCanvas = std::move(canv);
   }

   virtual ~RBrowserRCanvasWidget() = default;

   std::string GetKind() const override { return "rcanvas"s; }

   void Show(const std::string &arg) override
   {
      fCanvas->Show(arg);
   }

   std::string GetUrl() override
   {
      return "../"s + fCanvas->GetWindowAddr() + "/"s;
   }

   std::string GetTitle() override
   {
      return fCanvas->GetTitle();
   }

   bool DrawElement(std::shared_ptr<Browsable::RElement> &elem, const std::string &opt) override
   {
      if (!elem->IsCapable(Browsable::RElement::kActDraw7))
         return false;

      auto obj = elem->GetObject();
      if (!obj)
         return false;

      std::shared_ptr<RPadBase> subpad = fCanvas;

      if (obj && Browsable::RProvider::Draw7(subpad, obj, opt)) {
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

class RBrowserRCanvasProvider : public RBrowserWidgetProvider {
protected:
   std::shared_ptr<RBrowserWidget> Create(const std::string &name) final
   {
      return std::make_shared<RBrowserRCanvasWidget>(name);
   }

   std::shared_ptr<RBrowserWidget> CreateFor(const std::string &name, std::shared_ptr<Browsable::RElement> &elem) final
   {
      auto holder = elem->GetObject();
      if (!holder) return nullptr;

      auto canv = holder->get_shared<RCanvas>();
      if (!canv) return nullptr;

      return std::make_shared<RBrowserRCanvasWidget>(name, canv);
   }

public:
   RBrowserRCanvasProvider() : RBrowserWidgetProvider("rcanvas") {}
   ~RBrowserRCanvasProvider() = default;
} sRBrowserRCanvasProvider;
