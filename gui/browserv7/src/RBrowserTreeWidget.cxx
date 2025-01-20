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
#include "ROOT/Browsable/TObjectHolder.hxx"
#include "TTree.h"
#include "TBranch.h"
#include "TBranchBrowsable.h"
#include "TLeaf.h"

using namespace ROOT;

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
   std::shared_ptr<RWebWindow> GetWindow() override { return fViewer.GetWindow(); }

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

      tree = fObject ? fObject->Get<TTree>() : nullptr;

      TTree *new_tree = nullptr;
      std::string expr = elem->GetContent("tree");

      auto branch = obj->Get<TBranch>();
      auto leaf = obj->Get<TLeaf>();
      auto browsable = obj->Get<TVirtualBranchBrowsable>();

      if (branch) {
         new_tree = branch->GetTree();
         if (expr.empty())
            expr = branch->GetFullName().Data();
      } else if (leaf) {
         new_tree = leaf->GetBranch()->GetTree();
         if (expr.empty())
            expr = leaf->GetFullName().Data();
      } else if (browsable) {
         new_tree = browsable->GetBranch()->GetTree();
         if (expr.empty())
            expr = browsable->GetBranch()->GetFullName().Data();
      }

      if (!new_tree || expr.empty())
         return false;

      if (new_tree != tree) {
         fObject = std::make_unique<Browsable::TObjectHolder>(new_tree);
         fTitle = new_tree->GetName();
         fViewer.SetTree(new_tree);
      }

      return fViewer.SuggestExpression(expr);
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
