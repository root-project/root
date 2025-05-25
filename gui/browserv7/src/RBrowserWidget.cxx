// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2021-01-22
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RBrowserWidget.hxx"

#include <algorithm>

#include "TSystem.h"
#include "TBufferJSON.h"

using namespace std::string_literals;
using namespace ROOT;

///////////////////////////////////////////////////////////////
/// Returns string which can be send to browser client to set/change
/// title of the widget tab

std::string RBrowserWidget::SendWidgetTitle()
{
   std::vector<std::string> args = { GetName(), GetTitle(), Browsable::RElement::GetPathAsString(GetPath()) };

   return "SET_TITLE:"s + TBufferJSON::ToJSON(&args).Data();
}

///////////////////////////////////////////////////////////////
/// Constructor

RBrowserWidgetProvider::RBrowserWidgetProvider(const std::string &kind)
{
   GetMap().emplace(kind, this);
}

///////////////////////////////////////////////////////////////
/// Destructor

RBrowserWidgetProvider::~RBrowserWidgetProvider()
{
   auto &map = GetMap();
   auto iter = std::find_if(map.begin(), map.end(),
         [this](const ProvidersMap_t::value_type &pair) { return this == pair.second; });

   if (iter != map.end())
      map.erase(iter);
}

///////////////////////////////////////////////////////////////
/// Returns static map of existing providers

RBrowserWidgetProvider::ProvidersMap_t& RBrowserWidgetProvider::GetMap()
{
   static RBrowserWidgetProvider::ProvidersMap_t mMap;
   return mMap;
}

///////////////////////////////////////////////////////////////
/// Returns provider for specified kind

RBrowserWidgetProvider *RBrowserWidgetProvider::GetProvider(const std::string &kind)
{
   auto &map = GetMap();
   auto iter = map.find(kind);
   if (iter == map.end()) {
      // try to load necessary libraries
      if (kind == "geom"s)
         gSystem->Load("libROOTBrowserGeomWidget");
      else if (kind == "tree"s)
         gSystem->Load("libROOTBrowserTreeWidget");
      else if (kind == "tcanvas"s)
         gSystem->Load("libROOTBrowserTCanvasWidget");
      else if (kind == "rcanvas"s)
         gSystem->Load("libROOTBrowserRCanvasWidget");
      iter = map.find(kind);
      if (iter == map.end())
         return nullptr;
   }
   return iter->second;
}


///////////////////////////////////////////////////////////////
/// Create specified widget

std::shared_ptr<RBrowserWidget> RBrowserWidgetProvider::CreateWidget(const std::string &kind, const std::string &name)
{
   auto provider = GetProvider(kind);

   return provider ? provider->Create(name) : nullptr;
}

///////////////////////////////////////////////////////////////
/// Create specified widget for existing object

std::shared_ptr<RBrowserWidget> RBrowserWidgetProvider::CreateWidgetFor(const std::string &kind, const std::string &name, std::shared_ptr<Browsable::RElement> &element)
{
   auto provider = GetProvider(kind);

   return provider ? provider->CreateFor(name, element) : nullptr;
}

///////////////////////////////////////////////////////////////
/// Check if catch window can be identified and normal widget can be created
/// Used for TCanvas created in macro and catch by RBrowser

std::shared_ptr<RBrowserWidget> RBrowserWidgetProvider::DetectCatchedWindow(const std::string &kind, RWebWindow &win)
{
   auto provider = GetProvider(kind);

   return provider ? provider->DetectWindow(win) : nullptr;
}

