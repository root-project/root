/// \file ROOT/RFileDialog.cxx
/// \ingroup rbrowser
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2019-10-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RFileDialog.hxx>

#include <ROOT/RLogger.hxx>
#include <ROOT/RBrowsableSysFile.hxx>
#include <ROOT/RBrowserItem.hxx>


#include "TSystem.h"

#include "TBufferJSON.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <fstream>

using namespace std::string_literals;

using namespace ROOT::Experimental;

using namespace ROOT::Experimental::Browsable;

/** \class RFileDialog
\ingroup rbrowser

web-based FileDialog.
*/

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

RFileDialog::RFileDialog(EDialogTypes kind, const std::string &title)
{
   fKind = kind;
   fTitle = title;

   fWorkingDirectory = gSystem->UnixPathName(gSystem->WorkingDirectory());
   printf("Current dir %s\n", fWorkingDirectory.c_str());

   fBrowsable.SetTopElement(std::make_unique<SysFileElement>("/"));
   fBrowsable.SetWorkingDirectory(fWorkingDirectory);

   fWebWindow = RWebWindow::Create();

   fWebWindow->SetPanelName("rootui5.browser.view.FileDialog");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetCallBacks([this](unsigned connid) { fConnId = connid; SendInitMsg(connid); },
                            [this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); },
                            [this](unsigned connid) { if (fConnId == connid) fConnId = 0; if (fCallback && !fDidSelect) fCallback(""); fDidSelect = true; });
   fWebWindow->SetGeometry(800, 600); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RFileDialog::~RFileDialog()
{
   printf("RFileDialog Destructor\n");
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Assign callback. If file was already selected, immediately call it

void RFileDialog::SetCallback(RFileDialogCallback_t callback)
{
   fCallback = callback;
   if (fDidSelect && fCallback)
      fCallback(fSelect);
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Process browser request

std::string RFileDialog::ProcessBrowserRequest(const std::string &msg)
{

   // not used now, can be reactivated later
   std::unique_ptr<RBrowserRequest> request;

   if (msg.empty()) {
      request = std::make_unique<RBrowserRequest>();
      request->path = "/";
      request->first = 0;
      request->number = 10000;
   } else {
      request = TBufferJSON::FromJSON<RBrowserRequest>(msg);
   }

   if (!request)
      return ""s;

   return "BREPL:"s + fBrowsable.ProcessRequest(*request.get());
}


/////////////////////////////////////////////////////////////////////////////////
/// Show or update RFileDialog in web window
/// If web window already started - just refresh it like "reload" button does
/// Reset result of file selection (if any)

void RFileDialog::Show(const RWebDisplayArgs &args)
{
   fDidSelect = false;
   fSelect.clear();

   if (fWebWindow->NumConnections() == 0) {
      fWebWindow->Show(args);
   } else {
      WebWindowCallback(0, "RELOAD");
   }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Hide ROOT Browser

void RFileDialog::Hide()
{
   fWebWindow->CloseConnections();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process client connect

void RFileDialog::SendInitMsg(unsigned connid)
{
   RBrowserRequest request;
   request.path = "/";
   request.first = 0;
   request.number = 0;

   std::string kindstr;
   switch(fKind) {
      case kOpenFile : kindstr = "OpenFile"; break;
      case kSaveAsFile : kindstr = "SaveAsFile"; break;
      case kNewFile : kindstr = "NewFile"; break;
   }

   auto jpath = TBufferJSON::ToJSON(&fBrowsable.GetWorkingPath());

   std::string jsoncode = "{ \"kind\" : \""s + kindstr + "\", \"title\" : \""s + fTitle + "\", \"path\" : \" "s + jpath.Data() + "\", \"brepl\" : "s + fBrowsable.ProcessRequest(request) + "   }"s;

   fWebWindow->Send(connid, "INMSG:"s + jsoncode);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return the current directory of ROOT

std::string RFileDialog::GetCurrentWorkingDirectory()
{
   return "WORKPATH:"s + fWorkingDirectory;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Sends files list to the browser

void RFileDialog::SendDirContent(unsigned connid)
{
   RBrowserRequest request;
   request.path = "/";
   request.first = 0;
   request.number = 0;
   auto msg = "BREPL:"s + fBrowsable.ProcessRequest(request);

   fWebWindow->Send(connid, msg);
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// receive data from client

void RFileDialog::WebWindowCallback(unsigned connid, const std::string &arg)
{
   size_t len = arg.find("\n");
   if (len != std::string::npos)
      printf("Recv %s\n", arg.substr(0, len).c_str());
   else
      printf("Recv %s\n", arg.c_str());

   if (arg.compare(0, 6, "CHDIR:") == 0) {
      auto chdir = arg.substr(6);
      if (!chdir.empty() && (chdir[0] != '/'))
         fWorkingDirectory += "/"s + chdir;
      else
         fWorkingDirectory = chdir;
      printf("Current dir %s\n", fWorkingDirectory.c_str());
      fBrowsable.SetWorkingDirectory(fWorkingDirectory);

      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
      SendDirContent(connid);
   } else if (arg.compare(0, 7, "SELECT:") == 0) {
      fSelect = fWorkingDirectory + "/"s + arg.substr(7);

      if (fCallback && !fDidSelect)
         fCallback(fSelect);

      fDidSelect = true;

      fWebWindow->Send(connid, "CLOSE:"s); // sending close
   }
}

std::string RFileDialog::Dialog(EDialogTypes kind, const std::string &title)
{
   RFileDialog dlg(kind, title);

   dlg.Show();

   dlg.fWebWindow->WaitForTimed([&](double) {
      if (dlg.fDidSelect) return 1;

      return 0; // continue waiting
   });

   return dlg.fSelect;
}



std::string RFileDialog::OpenFile(const std::string &title)
{
   return Dialog(kOpenFile, title);
}

std::string RFileDialog::SaveAsFile(const std::string &title)
{
   return Dialog(kSaveAsFile, title);
}

std::string RFileDialog::NewFile(const std::string &title)
{
   return Dialog(kNewFile, title);
}


