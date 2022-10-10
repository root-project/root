// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2022-10-07
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RBrowserWidget.hxx"

#include "ROOT/RTreeViewer.hxx"
#include "TTree.h"
#include "ROOT/RBrowser.hxx"

#include "TBufferJSON.h"

using namespace ROOT::Experimental;

using namespace std::string_literals;


class RBrowserTreeWidget : public RBrowserWidget {
   RTreeViewer fViewer;
   std::unique_ptr<Browsable::RHolder> fObject; // tree object
   std::string fTitle;
   bool fFirstSend{false};

public:

   RBrowserTreeWidget(const std::string &name) : RBrowserWidget(name)
   {
      fViewer.SetTitle(name);
      fViewer.SetShowHierarchy(false);
      fViewer.SetCallback([this](const std::string &canvas_name) {
         GetBrowser()->ActivateWidget(canvas_name, "tcanvas");
      });
   }

   virtual ~RBrowserTreeWidget() = default;

   std::string GetKind() const override { return "tree"s; }
   std::string GetTitle() override { return fTitle; }
   std::string GetUrl() override { return "../"s + fViewer.GetWindowAddr() + "/"s; }

   void Show(const std::string &arg) override { fViewer.Show(arg); }

   bool DrawElement(std::shared_ptr<Browsable::RElement> &elem, const std::string &, bool) override
   {
      if (!elem->IsCapable(Browsable::RElement::kActTree))
         return false;

      fObject = elem->GetObject();
      if (!fObject)
         return false;

      auto tree = fObject->Get<TTree>();
      if (!tree) return false;

      fTitle = tree->GetName();
      fViewer.SetTree(const_cast<TTree *>(tree));

      return true;
   }

};

// ======================================================================

class RBrowserTreeProvider : public RBrowserWidgetProvider {
protected:
   std::shared_ptr<RBrowserWidget> Create(const std::string &name) final
   {
      return std::make_shared<RBrowserTreeWidget>(name);
   }
public:
   RBrowserTreeProvider() : RBrowserWidgetProvider("tree") {}
   ~RBrowserTreeProvider() = default;
} sRBrowserTreeProvider;
