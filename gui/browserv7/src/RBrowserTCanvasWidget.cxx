/// \file RBrowserTCanvasWidget.cxx
/// \ingroup rbrowser
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2021-01-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RBrowserWidget.hxx"

#include "TCanvas.h"
#include "TWebCanvas.h"

using namespace ROOT::Experimental;

using namespace std::string_literals;


class RBrowserTCanvasWidget : public RBrowserWidget {

   std::unique_ptr<TCanvas> fCanvas; ///<! drawn canvas
   TWebCanvas *fWebCanvas{nullptr};  ///<! web implementation, owned by TCanvas

   std::unique_ptr<Browsable::RHolder> fObject; // TObject drawing in the TCanvas

public:

   RBrowserTCanvasWidget(const std::string &name) : RBrowserWidget(name)
   {
      fCanvas = std::make_unique<TCanvas>(kFALSE);
      fCanvas->SetName(name.c_str());
      fCanvas->SetTitle(name.c_str());
      fCanvas->ResetBit(TCanvas::kShowEditor);
      fCanvas->ResetBit(TCanvas::kShowToolBar);
      fCanvas->SetCanvas(fCanvas.get());
      fCanvas->SetBatch(kTRUE); // mark canvas as batch
      fCanvas->SetEditable(kTRUE); // ensure fPrimitives are created

      // create implementation
      fWebCanvas = new TWebCanvas(fCanvas.get(), "title", 0, 0, 800, 600);

      // assign implementation
      fCanvas->SetCanvasImp(fWebCanvas);
   }

   virtual ~RBrowserTCanvasWidget() = default;

   std::string GetKind() const override { return "tcanvas"s; }

   void Show(const std::string &arg) override
   {
      fWebCanvas->ShowWebWindow(arg);
   }

   std::string GetUrl() override
   {
      return "../"s + fWebCanvas->GetWebWindow()->GetAddr() + "/"s;
   }

   bool DrawElement(std::shared_ptr<Browsable::RElement> &elem, const std::string &opt) override
   {
      if (!elem->IsCapable(Browsable::RElement::kActDraw6))
         return false;

      fObject = elem->GetObject();
      if (!fObject)
         return false;

      // first take object without ownership
      auto tobj = fObject->get_object<TObject>();
      if (!tobj) {
         // and now with ownership
         auto utobj = fObject->get_unique<TObject>();
         if (!utobj)
            return false;
         tobj = utobj.release();
         tobj->SetBit(TObject::kMustCleanup); // TCanvas should care about cleanup
      }

      fCanvas->GetListOfPrimitives()->Clear();

      fCanvas->GetListOfPrimitives()->Add(tobj, opt.c_str());

      fCanvas->ForceUpdate(); // force update async - do not wait for confirmation

      return true;
   }

   std::string ReplyAfterDraw() override
   {
      return ""s;
   }

};

// ======================================================================

class RBrowserTCanvasProvider : public RBrowserWidgetProvider {
protected:
   std::shared_ptr<RBrowserWidget> Create(const std::string &name) final
   {
      return std::make_shared<RBrowserTCanvasWidget>(name);
   }
public:
   RBrowserTCanvasProvider() : RBrowserWidgetProvider("tcanvas") {}
   ~RBrowserTCanvasProvider() = default;
} sRBrowserTCanvasProvider;
