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
/// When title not specified, default will be used

RFileDialog::RFileDialog(EDialogTypes kind, const std::string &title, const std::string &fname)
{
   fKind = kind;
   fTitle = title;

   if (fTitle.empty())
      switch (fKind) {
         case kOpenFile: fTitle = "Open file"; break;
         case kSaveAs: fTitle = "Save as file"; break;
         case kNewFile: fTitle = "New file"; break;
      }

   fSelect = fname;

   // TODO: windows
   fBrowsable.SetTopElement(std::make_unique<SysFileElement>("/"));

   auto separ = fSelect.rfind("/");

   if (fSelect.empty() || (separ == std::string::npos)) {
      std::string workdir = gSystem->UnixPathName(gSystem->WorkingDirectory());
      fBrowsable.SetWorkingDirectory(workdir);
   } else {
      std::string workdir = fSelect.substr(0, separ);
      fBrowsable.SetWorkingDirectory(workdir);
      fSelect = fSelect.substr(separ+1);
   }

   fWebWindow = RWebWindow::Create();

   // when dialog used in standalone mode, ui5 panel will be loaded
   fWebWindow->SetPanelName("rootui5.browser.view.FileDialog");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetCallBacks([this](unsigned connid) { SendInitMsg(connid); },
                            [this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); },
                            [this](unsigned) { InvokeCallBack(); });
   fWebWindow->SetGeometry(800, 600); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RFileDialog::~RFileDialog()
{
   InvokeCallBack(); // invoke callback if not yet performed
   printf("RFileDialog Destructor\n");
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Assign callback. If file was already selected, immediately call it

void RFileDialog::SetCallback(RFileDialogCallback_t callback)
{
   fCallback = callback;
   if (fDidSelect)
      InvokeCallBack();
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

   if (fWebWindow->NumConnections() == 0) {
      RWebWindow::ShowWindow(fWebWindow, args);
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


std::string RFileDialog::TypeAsString(EDialogTypes kind)
{
   switch(kind) {
      case kOpenFile: return "OpenFile"s;
      case kSaveAs: return "SaveAs"s;
      case kNewFile: return "NewFile"s;
   }

   return ""s;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Sends initial message to the client

void RFileDialog::SendInitMsg(unsigned connid)
{
   RBrowserRequest request;
   request.path = "/";
   request.first = 0;
   request.number = 0;

   auto jtitle = TBufferJSON::ToJSON(&fTitle);
   auto jpath = TBufferJSON::ToJSON(&fBrowsable.GetWorkingPath());
   auto jfname = TBufferJSON::ToJSON(&fSelect);

   std::string jsoncode = "{ \"kind\" : \""s + TypeAsString(fKind) +
                          "\", \"title\" : "s + jtitle.Data() +
                          ", \"path\" : "s + jpath.Data() +
                          ", \"fname\" : "s + jfname.Data() +
                          ", \"brepl\" : "s + fBrowsable.ProcessRequest(request) + "   }"s;

   fWebWindow->Send(connid, "INMSG:"s + jsoncode);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return the current directory of ROOT

std::string RFileDialog::GetCurrentWorkingDirectory()
{
   return "WORKPATH:"s + TBufferJSON::ToJSON(&fBrowsable.GetWorkingPath()).Data();
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

   if (arg.compare(0, 7, "CHPATH:") == 0) {
      printf("chpath %s\n", arg.substr(7).c_str());
      auto path = TBufferJSON::FromJSON<RElementPath_t>(arg.substr(7));
      if (path) fBrowsable.SetWorkingPath(*path);
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
      SendDirContent(connid);
   } else if (arg.compare(0, 6, "CHDIR:") == 0) {
      printf("chdir dir %s\n", arg.substr(6).c_str());

      auto path = fBrowsable.GetWorkingPath();
      path.emplace_back(arg.substr(6));
      fBrowsable.SetWorkingPath(path);

      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
      SendDirContent(connid);
   } else if (arg.compare(0, 10, "DLGSELECT:") == 0) {
      // selected file name, if file exists - send request for confirmation

      auto path = TBufferJSON::FromJSON<RElementPath_t>(arg.substr(10));

      if (!path) {
         printf("Error to decode JSON %s\n", arg.substr(10).c_str());
         return;
      }

      fSelect = SysFileElement::ProduceFileName(*path);

      bool need_confirm = false;

      if ((GetType() == kSaveAs) || (GetType() == kNewFile))
         if (fBrowsable.GetElementFromTop(*path))
            need_confirm = true;

      if (need_confirm) {
         fWebWindow->Send(connid, "NEED_CONFIRM"s); // sending request for confirmation
      } else {
         fWebWindow->Send(connid, "SELECT_CONFIRMED:"s + fSelect); // sending select confirmation with fully qualified file name
         fDidSelect = true;
      }
   } else if (arg == "DLGNOSELECT") {
      fSelect.clear();
      fDidSelect = true;
      fWebWindow->Send(connid, "NOSELECT_CONFIRMED"s); // sending confirmation of NOSELECT
   } else if (arg == "DLG_CONFIRM_SELECT") {
      fDidSelect = true;
      fWebWindow->Send(connid, "SELECT_CONFIRMED:"s + fSelect);
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Invoke specified callback

void RFileDialog::InvokeCallBack()
{
   if (fCallback) {
      auto func = fCallback;
      fCallback = nullptr;
      func(fSelect);
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Start specified dialog type

std::string RFileDialog::Dialog(EDialogTypes kind, const std::string &title, const std::string &fname)
{
   RFileDialog dlg(kind, title, fname);

   dlg.Show();

   dlg.fWebWindow->WaitForTimed([&](double) {
      if (dlg.fDidSelect) return 1;

      return 0; // continue waiting
   });

   return dlg.fSelect;
}

/////////////////////////////////////////////////////////////////////////////////////
/// Start OpenFile dialog.
/// Blocks until file name is selected or Cancel button is pressed
/// Returns selected file name (or empty string)

std::string RFileDialog::OpenFile(const std::string &title, const std::string &fname)
{
   return Dialog(kOpenFile, title, fname);
}

/////////////////////////////////////////////////////////////////////////////////////
/// Start SaveAs dialog.
/// Blocks until file name is selected or Cancel button is pressed
/// Returns selected file name (or empty string)

std::string RFileDialog::SaveAs(const std::string &title, const std::string &fname)
{
   return Dialog(kSaveAs, title, fname);
}

/////////////////////////////////////////////////////////////////////////////////////
/// Start NewFile dialog.
/// Blocks until file name is selected or Cancel button is pressed
/// Returns selected file name (or empty string)

std::string RFileDialog::NewFile(const std::string &title, const std::string &fname)
{
   return Dialog(kNewFile, title, fname);
}

/////////////////////////////////////////////////////////////////////////////////////
/// Create dialog instance to use as embedded dialog inside other widget
/// Embedded dialog started on the client side where FileDialogController.SaveAs() method called
/// Such method immediately send message with "FILEDIALOG:" prefix
/// On the server side widget should detect such message and call RFileDialog::Embedded()
/// providing received string as second argument.
/// Returned instance of shared_ptr<RFileDialog> may be used to assign callback when file is selected

std::shared_ptr<RFileDialog> RFileDialog::Embedded(const std::shared_ptr<RWebWindow> &window, const std::string &args)
{
   if (args.compare(0, 11, "FILEDIALOG:") != 0)
      return nullptr;

   auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(args.substr(11));

   if (!arr || (arr->size() != 3)) {
      printf("Embedded failure - wrong arguments %s, should be array with three strings\n", args.c_str());
      return nullptr;
   }

   auto kind = kSaveAs;

   if (TypeAsString(kOpenFile) == arr->at(0))
      kind = kOpenFile;
   else if (TypeAsString(kNewFile) == arr->at(0))
      kind = kNewFile;

   auto dialog = std::make_shared<RFileDialog>(kind, "", arr->at(1));
   dialog->Show({window, std::stoi(arr->at(2))});

   // TODO: how one can avoid const_cast for lambda capture to release
   dialog->SetCallback([dialog](const std::string &) mutable { dialog.reset(); }); // use callback to release pointer

   return dialog;
}


