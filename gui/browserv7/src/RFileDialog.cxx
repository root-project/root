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

using namespace ROOT::Experimental::Browsable;

/** \class ROOT::Experimental::RFileDialog
\ingroup rbrowser

web-based FileDialog.
*/

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

ROOT::Experimental::RFileDialog::RFileDialog()
{
   fWorkingDirectory = gSystem->UnixPathName(gSystem->WorkingDirectory());
   printf("Current dir %s\n", fWorkingDirectory.c_str());

   fBrowsable.SetTopElement(std::make_unique<SysFileElement>(fWorkingDirectory));

   fWebWindow = RWebWindow::Create();
   fWebWindow->SetDefaultPage("file:rootui5sys/browser/filedialog.html");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetCallBacks([this](unsigned connid) { fConnId = connid; SendInitMsg(connid); },
                            [this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); });
   fWebWindow->SetGeometry(800, 600); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue

   Show();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

ROOT::Experimental::RFileDialog::~RFileDialog()
{
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Process browser request

std::string ROOT::Experimental::RFileDialog::ProcessBrowserRequest(const std::string &msg)
{
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

void ROOT::Experimental::RFileDialog::Show(const RWebDisplayArgs &args)
{
   if (fWebWindow->NumConnections() == 0) {
      fWebWindow->Show(args);
   } else {
      WebWindowCallback(0, "RELOAD");
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Hide ROOT Browser

void ROOT::Experimental::RFileDialog::Hide()
{
   fWebWindow->CloseConnections();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process client connect

void ROOT::Experimental::RFileDialog::SendInitMsg(unsigned connid)
{
   fWebWindow->Send(connid, "INMSG:"s + fWorkingDirectory);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return the current directory of ROOT

std::string ROOT::Experimental::RFileDialog::GetCurrentWorkingDirectory()
{
   return "GETWORKDIR: { \"path\": \""s + fWorkingDirectory + "\"}"s;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// receive data from client

void ROOT::Experimental::RFileDialog::WebWindowCallback(unsigned connid, const std::string &arg)
{
   size_t len = arg.find("\n");
   if (len != std::string::npos)
      printf("Recv %s\n", arg.substr(0, len).c_str());
   else
      printf("Recv %s\n", arg.c_str());

   if (arg.compare(0,6, "BRREQ:") == 0) {
      // central place for processing browser requests
      auto json = ProcessBrowserRequest(arg.substr(6));
      if (json.length() > 0) fWebWindow->Send(connid, json);
   } else if (arg.compare(0, 11, "GETWORKDIR:") == 0) {
      std::string res = GetCurrentWorkingDirectory();
      fWebWindow->Send(connid, res);
   } else if (arg.compare(0, 6, "CHDIR:") == 0) {
      fWorkingDirectory = arg.substr(6);
      if ((fWorkingDirectory.length()>1) && (fWorkingDirectory[fWorkingDirectory.length()-1] == '/')) fWorkingDirectory.resize(fWorkingDirectory.length()-1);
      printf("Current dir %s\n", fWorkingDirectory.c_str());
      fBrowsable.SetTopElement(std::make_unique<SysFileElement>(fWorkingDirectory));
      gSystem->ChangeDirectory(fWorkingDirectory.c_str());
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   }
}
