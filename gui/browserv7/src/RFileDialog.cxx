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

RFileDialog::RFileDialog()
{
   fWorkingDirectory = gSystem->UnixPathName(gSystem->WorkingDirectory());
   printf("Current dir %s\n", fWorkingDirectory.c_str());

   fBrowsable.SetTopElement(std::make_unique<SysFileElement>(fWorkingDirectory));

   fWebWindow = RWebWindow::Create();

   fWebWindow->SetPanelName("rootui5.browser.view.FileDialog");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetCallBacks([this](unsigned connid) { fConnId = connid; SendInitMsg(connid); SendDirContent(connid); },
                            [this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); });
   fWebWindow->SetGeometry(800, 600); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue

   Show();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RFileDialog::~RFileDialog()
{
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

void RFileDialog::Show(const RWebDisplayArgs &args)
{
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
   std::string jsoncode = "{ \"kind\" : \"SaveAs\", \"path\" : \" "s + fWorkingDirectory + "\" }"s;

   fWebWindow->Send(connid, "INMSG:"s + jsoncode);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return the current directory of ROOT

std::string RFileDialog::GetCurrentWorkingDirectory()
{
   return "GETWORKDIR:"s + fWorkingDirectory;
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
      if ((fWorkingDirectory.length()>1) && (fWorkingDirectory[fWorkingDirectory.length()-1] == '/'))
         fWorkingDirectory.resize(fWorkingDirectory.length()-1);
      printf("Current dir %s\n", fWorkingDirectory.c_str());
      fBrowsable.SetTopElement(std::make_unique<SysFileElement>(fWorkingDirectory));
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
      SendDirContent(connid);
   } else if (arg.compare(0, 7, "SELECT:") == 0) {
      auto res = fWorkingDirectory + "/"s + arg.substr(7);

      printf("Select %s\n", res.c_str());
   }
}
