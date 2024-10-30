// Author: Sergey Linev, 3.03.2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RGeomHierarchy.hxx>

#include <ROOT/RWebWindow.hxx>

#include "TBufferJSON.h"

using namespace std::string_literals;

using namespace ROOT;

/** \class ROOT::RGeomHierarchy
\ingroup webwidgets

\brief Web-based %ROOT geometry hierarchy viewer
*/



//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

RGeomHierarchy::RGeomHierarchy(RGeomDescription &desc, bool use_server_threads) : fDesc(desc)
{
   fWebWindow = RWebWindow::Create();
   if (use_server_threads)
      fWebWindow->UseServerThreads();

   fWebWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); });
   fWebWindow->SetDefaultPage("file:rootui5sys/geom/index.html");
   fWebWindow->SetGeometry(600, 900); // configure predefined window geometry

   fDesc.AddSignalHandler(this, [this](const std::string &kind) { ProcessSignal(kind); });
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RGeomHierarchy::~RGeomHierarchy()
{
   fDesc.RemoveSignalHandler(this);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process data from client

void RGeomHierarchy::WebWindowCallback(unsigned connid, const std::string &arg)
{
   if (arg.compare(0, 6, "BRREQ:") == 0) {
      // central place for processing browser requests
      auto json = fDesc.ProcessBrowserRequest(arg.substr(6));
      if (json.length() > 0)
         fWebWindow->Send(connid, json);
   } else if (arg.compare(0, 7, "SEARCH:") == 0) {

      std::string query = arg.substr(7);

      if (!query.empty()) {
         std::string hjson, json;
         fDesc.SearchVisibles(query, hjson, json);
         // send reply with appropriate header - NOFOUND, FOUND0:, FOUND1:
         fWebWindow->Send(0, hjson);
         // inform viewer that search is changed
         if (fDesc.SetSearch(query, json))
            fDesc.IssueSignal(this, json.empty() ? "ClearSearch" : "ChangeSearch");
      } else {
         fDesc.SetSearch(""s, ""s);
         fDesc.IssueSignal(this, "ClearSearch");
      }

      auto connids = fWebWindow->GetConnections(connid);

      for (auto id : connids)
         fWebWindow->Send(id, "SETSR:"s + query);

   } else if (arg.compare(0, 7, "SETTOP:") == 0) {
      auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));
      if (path && fDesc.SelectTop(*path)) {
         fDesc.IssueSignal(this, "SelectTop");
         auto connids = fWebWindow->GetConnections(connid);

         for (auto id : connids)
            fWebWindow->Send(id, "UPDATE"s);
      }
   } else if (arg.compare(0, 6, "HOVER:") == 0) {
      auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(6));
      if (path) {
         auto stack = fDesc.MakeStackByPath(*path);
         if (fDesc.SetHighlightedItem(stack))
            fDesc.IssueSignal(this, "HighlightItem");
      }
   } else if (arg.compare(0, 6, "CLICK:") == 0) {
      auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(6));
      if (path) {
         auto stack = fDesc.MakeStackByPath(*path);
         if (fDesc.SetClickedItem(stack))
            fDesc.IssueSignal(this, "ClickItem");
      }
   } else if ((arg.compare(0, 7, "SETVI0:") == 0) || (arg.compare(0, 7, "SETVI1:") == 0)) {
      // change visibility for specified nodeid

      auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));

      bool on = (arg[5] == '1');

      if (path && fDesc.ChangeNodeVisibility(*path, on))
         fDesc.IssueSignal(this, "NodeVisibility");

   } else if ((arg.compare(0, 5, "SHOW:") == 0) || (arg.compare(0, 5, "HIDE:") == 0)) {
      auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(5));
      if (path && fDesc.SetPhysNodeVisibility(*path, arg.compare(0, 5, "SHOW:") == 0))
         fDesc.IssueSignal(this, "NodeVisibility");
   } else if (arg.compare(0, 6, "CLEAR:") == 0) {
      auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(6));
      if (path && fDesc.ClearPhysNodeVisibility(*path))
         fDesc.IssueSignal(this, "NodeVisibility");
   } else if (arg == "CLEARALL"s) {
      if (fDesc.ClearAllPhysVisibility())
         fDesc.IssueSignal(this, "NodeVisibility");
   }
}

/////////////////////////////////////////////////////////////////////////////////
/// Show hierarchy in web window

void RGeomHierarchy::Show(const RWebDisplayArgs &args)
{
   if (args.GetWidgetKind().empty())
      const_cast<RWebDisplayArgs *>(&args)->SetWidgetKind("RGeomHierarchy");

   fWebWindow->SetUserArgs("{ show_columns: true, only_hierarchy: true }");
   RWebWindow::ShowWindow(fWebWindow, args);
}

/////////////////////////////////////////////////////////////////////////////////
/// Update client - reload hierarchy

void RGeomHierarchy::Update()
{
   if (fWebWindow)
      fWebWindow->Send(0, "RELOAD"s);
}

/////////////////////////////////////////////////////////////////////////////////
/// Let browse to specified location

void RGeomHierarchy::BrowseTo(const std::string &itemname)
{
   if (fWebWindow)
      fWebWindow->Send(0, "ACTIV:"s + itemname);
}

/////////////////////////////////////////////////////////////////////////////////
/// Process signals from geometry description object

void RGeomHierarchy::ProcessSignal(const std::string &kind)
{
   if (kind == "HighlightItem") {
      auto stack = fDesc.GetHighlightedItem();
      auto path = fDesc.MakePathByStack(stack);
      if (stack.empty())
         path = {"__OFF__"}; // just clear highlight
      if (fWebWindow)
         fWebWindow->Send(0, "HIGHL:"s + TBufferJSON::ToJSON(&path).Data());
   } else if (kind == "NodeVisibility") {
      // visibility changed from RGeomViewer, update hierarchy
      if (fWebWindow)
         fWebWindow->Send(0, "UPDATE"s);
   } else if (kind == "ActiveItem") {
      BrowseTo(fDesc.GetActiveItem());
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Set handle which will be cleared when connection is closed
/// Must be called after window is shown

void RGeomHierarchy::ClearOnClose(const std::shared_ptr<void> &handle)
{
   if (fWebWindow)
      fWebWindow->SetClearOnClose(handle);
}
