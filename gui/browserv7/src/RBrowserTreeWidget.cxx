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
#include "ROOT/RBrowser.hxx"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"

using namespace ROOT::Experimental;

using namespace std::string_literals;


class RBrowserTreeWidget : public RBrowserWidget {
   RTreeViewer fViewer;
   std::unique_ptr<Browsable::RHolder> fObject; // tree object
   std::string fTitle;

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

   bool DrawElement(std::shared_ptr<Browsable::RElement> &elem, const std::string & = "") override
   {
      if (!elem->IsCapable(Browsable::RElement::kActTree))
         return false;

      auto obj = elem->GetObject();
      if (!obj)
         return false;

      auto tree = obj->Get<TTree>();
      if (tree) {
         fObject = std::move(obj);
         fTitle = tree->GetName();
         fViewer.SetTree(const_cast<TTree *>(tree));
         return true;
      }

      auto branch = obj->Get<TBranch>();
      if (branch)
         return fViewer.SuggestBranch(branch);

      auto leaf = obj->Get<TLeaf>();
      if (leaf)
         return fViewer.SuggestLeaf(leaf);

      return false;
   }

   std::string SendWidgetContent() override { return SendWidgetTitle(); }

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
