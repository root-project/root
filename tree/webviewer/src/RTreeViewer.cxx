// Author: Sergey Linev, 7.10.2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RTreeViewer.hxx>

// #include <ROOT/RLogger.hxx>
#include <ROOT/RWebWindow.hxx>

#include "TTree.h"
// #include "THttpServer.h"
// #include "TBufferJSON.h"

using namespace std::string_literals;

using namespace ROOT::Experimental;

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

RTreeViewer::RTreeViewer(TTree *tree)
{
   fWebWindow = RWebWindow::Create();
   fWebWindow->SetDefaultPage("file:rootui5sys/tree/index.html");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); });
   fWebWindow->SetGeometry(900, 700); // configure predefined window geometry
   fWebWindow->SetConnLimit(0); // allow any connections numbers at the same time
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue

   if (tree) SetTree(tree);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RTreeViewer::~RTreeViewer()
{

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// assign new TTree to the viewer

void RTreeViewer::SetTree(TTree *tree)
{
   fTree = tree;

   Update();
}

/////////////////////////////////////////////////////////////////////////////////
/// Show or update viewer in web window
/// If web browser already started - just refresh drawing like "reload" button does
/// If no web window exists or \param always_start_new_browser configured, starts new window
/// \param args arguments to display

void RTreeViewer::Show(const RWebDisplayArgs &args, bool always_start_new_browser)
{
   std::string user_args = "";
   if (!GetShowHierarchy()) user_args = "{ nobrowser: true }";
   fWebWindow->SetUserArgs(user_args);

   if (args.GetWidgetKind().empty())
      const_cast<RWebDisplayArgs *>(&args)->SetWidgetKind("RTreeViewer");

   if ((fWebWindow->NumConnections(true) == 0) || always_start_new_browser)
      fWebWindow->Show(args);
   else
      Update();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return URL address of web window used for tree viewer

std::string RTreeViewer::GetWindowAddr() const
{
   return fWebWindow ? fWebWindow->GetAddr() : ""s;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Update tree viewer in all web displays

void RTreeViewer::Update()
{
   fWebWindow->Send(0, "RELOAD");
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Send data for initialize viewer

void RTreeViewer::SendViewerData(unsigned connid)
{
   std::string json = "VIEWER:{}";

   fWebWindow->Send(connid, json);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// receive data from client

void RTreeViewer::WebWindowCallback(unsigned connid, const std::string &arg)
{
   if (arg == "GETVIEWER") {

      SendViewerData(connid);

   } else if (arg == "QUIT_ROOT") {

      fWebWindow->TerminateROOT();

   } else if (arg == "RELOAD") {

      SendViewerData(connid);
   }
}
