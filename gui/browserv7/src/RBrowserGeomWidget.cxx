/// \file RBrowserGeomWidget.cxx
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

#include <ROOT/REveGeomViewer.hxx>

using namespace ROOT::Experimental;

using namespace std::string_literals;


class RBrowserGeomWidget : public RBrowserWidget {
   REveGeomViewer fViewer;

public:

   RBrowserGeomWidget(const std::string &name) : RBrowserWidget(name), fViewer()
   {
      fViewer.SetTitle(name);
      fViewer.SetShowHierarchy(false);
   }

   virtual ~RBrowserGeomWidget() = default;

   std::string GetKind() const override { return "geom"s; }

   void Show(const std::string &arg) override
   {
      fViewer.Show(arg);
   }

   std::string GetUrl() override
   {
      return "../"s + fViewer.GetWindowAddr() + "/"s;
   }

};

// ======================================================================

class RBrowserGeomProvider : public RBrowserWidgetProvider {
protected:
   std::shared_ptr<RBrowserWidget> Create(const std::string &name) final
   {
      return std::make_shared<RBrowserGeomWidget>(name);
   }
public:
   RBrowserGeomProvider() : RBrowserWidgetProvider("geom") {}
   ~RBrowserGeomProvider() = default;
} sRBrowserGeomProvider;
