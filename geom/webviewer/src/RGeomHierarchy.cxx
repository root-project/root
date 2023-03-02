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

using namespace std::string_literals;

using namespace ROOT::Experimental;

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

RGeomHierarchy::RGeomHierarchy(RGeomDescription &desc) :
  fDesc(desc)
{
   fWebWindow = RWebWindow::Create();
   fWebWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); });
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RGeomHierarchy::~RGeomHierarchy()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process data from client

void RGeomHierarchy::WebWindowCallback(unsigned connid, const std::string &arg)
{
   if (arg.compare(0,6, "BRREQ:") == 0) {
      // central place for processing browser requests
      auto json = fDesc.ProcessBrowserRequest(arg.substr(6));
      if (json.length() > 0) fWebWindow->Send(connid, json);
   } else if (arg.compare(0, 7, "SEARCH:") == 0) {

      std::string query = arg.substr(7);

      std::string hjson, json;

      /* auto nmatches = */ fDesc.SearchVisibles(query, hjson, json);

      // send reply with appropriate header - NOFOUND, FOUND0:, FOUND1:
      fWebWindow->Send(connid, hjson);

      if (!json.empty())
         fWebWindow->Send(connid, json);
   }

}

/////////////////////////////////////////////////////////////////////////////////
/// Show hierarchy in web window

void RGeomHierarchy::Show(const RWebDisplayArgs &args)
{
   RWebWindow::ShowWindow(fWebWindow, args);
}
