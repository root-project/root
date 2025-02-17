// Authors: Bertrand Bellenot <bertrand.bellenot@cern.ch> Sergey Linev <S.Linev@gsi.de>
// Date: 2019-02-28
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RBrowser.hxx>

#include <ROOT/Browsable/RSysFile.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/Browsable/RProvider.hxx>

#include <ROOT/RLogger.hxx>
#include <ROOT/RFileDialog.hxx>
#include <ROOT/RWebWindowsManager.hxx>

#include "RBrowserWidget.hxx"

#include "TVirtualPad.h"
#include "TString.h"
#include "TSystem.h"
#include "TError.h"
#include "TTimer.h"
#include "TROOT.h"
#include "TBufferJSON.h"
#include "TApplication.h"
#include "TRint.h"
#include "Getline.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <fstream>

using namespace std::string_literals;

namespace ROOT {

class RBrowserTimer : public TTimer {
public:
   RBrowser &fBrowser; ///!< browser processing postponed requests

   /// constructor
   RBrowserTimer(Long_t milliSec, Bool_t mode, RBrowser &br) : TTimer(milliSec, mode), fBrowser(br) {}

   /// timeout handler
   /// used to process postponed requests in main ROOT thread
   void Timeout() override { fBrowser.ProcessPostponedRequests(); }
};


class RBrowserEditorWidget : public RBrowserWidget {
public:

   bool fIsEditor{true};   ///<! either editor or image viewer
   std::string fTitle;
   std::string fFileName;
   std::string fContent;
   bool fFirstSend{false};  ///<! if editor content was send at least once
   std::string fItemPath;   ///<! item path in the browser

   RBrowserEditorWidget(const std::string &name, bool is_editor = true) : RBrowserWidget(name), fIsEditor(is_editor) {}
   virtual ~RBrowserEditorWidget() = default;

   void ResetConn() override { fFirstSend = false; }

   std::string GetKind() const override { return fIsEditor ? "editor"s : "image"s; }
   std::string GetTitle() override { return fTitle; }

   bool DrawElement(std::shared_ptr<Browsable::RElement> &elem, const std::string & = "") override
   {
      if (fIsEditor && elem->IsCapable(Browsable::RElement::kActEdit)) {
         auto code = elem->GetContent("text");
         if (!code.empty()) {
            fFirstSend = false;
            fContent = code;
            fTitle = elem->GetName();
            fFileName = elem->GetContent("filename");
         } else {
            auto json = elem->GetContent("json");
            if (!json.empty()) {
               fFirstSend = false;
               fContent = json;
               fTitle = elem->GetName() + ".json";
               fFileName = "";
            }
         }
         if (!fContent.empty()) {
            // page->fItemPath = item_path;
            return true;
         }
      }

      if (!fIsEditor && elem->IsCapable(Browsable::RElement::kActImage)) {
         auto img = elem->GetContent("image64");
         if (!img.empty()) {
            fFirstSend = false;
            fContent = img;
            fTitle = elem->GetName();
            fFileName = elem->GetContent("filename");
            // fItemPath = item_path;

            return true;
         }
      }

      return false;
   }

   std::string SendWidgetContent() override
   {
      if (fFirstSend) return ""s;

      fFirstSend = true;
      std::vector<std::string> args = { GetName(), fTitle, fFileName, fContent };

      std::string msg = fIsEditor ? "EDITOR:"s : "IMAGE:"s;
      msg += TBufferJSON::ToJSON(&args).Data();
      return msg;
   }

};


class RBrowserInfoWidget : public RBrowserWidget {
public:

   enum { kMaxContentLen = 10000000 };

   std::string fTitle;
   std::string fContent;
   bool fFirstSend{false};  ///<! if editor content was send at least once

   RBrowserInfoWidget(const std::string &name) : RBrowserWidget(name)
   {
      fTitle = "Cling info"s;
      Refresh();
   }

   virtual ~RBrowserInfoWidget() = default;

   void ResetConn() override { fFirstSend = false; }

   std::string GetKind() const override { return "info"s; }
   std::string GetTitle() override { return fTitle; }

   bool DrawElement(std::shared_ptr<Browsable::RElement> &, const std::string & = "") override { return false; }

   void Refresh()
   {
      fFirstSend = false;
      fContent = "";

      std::ostringstream pathtmp;
      pathtmp << gSystem->TempDirectory() << "/info." << gSystem->GetPid() << ".log";

      std::ofstream ofs(pathtmp.str(), std::ofstream::out | std::ofstream::app);
      ofs << "";
      ofs.close();

      gSystem->RedirectOutput(pathtmp.str().c_str(), "a");
      gROOT->ProcessLine(".g");
      gSystem->RedirectOutput(nullptr);

      std::ifstream infile(pathtmp.str());
      if (infile) {
         std::string line;
         while (std::getline(infile, line) && (fContent.length() < kMaxContentLen)) {
            fContent.append(line);
            fContent.append("\n");
         }
      }

      gSystem->Unlink(pathtmp.str().c_str());
   }

   void RefreshFromLogs(const std::string &promt, const std::vector<std::string> &logs)
   {
      int indx = 0, last_prompt = -1;
      for (auto &line : logs) {
         if (line == promt)
            last_prompt = indx;
         indx++;
      }

      if (last_prompt < 0) {
         Refresh();
         return;
      }

      fFirstSend = false;
      fContent = "";

      indx = 0;
      for (auto &line : logs) {
         if ((indx++ > last_prompt) && (fContent.length() < kMaxContentLen)) {
            fContent.append(line);
            fContent.append("\n");
         }
      }
   }


   std::string SendWidgetContent() override
   {
      if (fFirstSend)
         return ""s;

      if (fContent.empty())
         Refresh();

      fFirstSend = true;
      std::vector<std::string> args = { GetName(), fTitle, fContent };

      return "INFO:"s + TBufferJSON::ToJSON(&args).Data();
   }

};


class RBrowserCatchedWidget : public RBrowserWidget {
public:

   RWebWindow  *fWindow{nullptr};   // catched widget, TODO: to be changed to shared_ptr
   std::string fCatchedKind;  // kind of catched widget

   std::string GetKind() const override { return "catched"s; }

   std::string GetUrl() override { return fWindow ? ".."s + fWindow->GetUrl(false) : ""s; }

   std::string GetTitle() override { return fCatchedKind; }

   bool IsValid() override { return fWindow != nullptr; }

   RBrowserCatchedWidget(const std::string &name, RWebWindow *win, const std::string &kind) :
      RBrowserWidget(name),
      fWindow(win),
      fCatchedKind(kind)
   {
   }
};

} // namespace ROOT

using namespace ROOT;


/** \class ROOT::RBrowser
\ingroup rbrowser
\ingroup webwidgets

\brief Web-based %ROOT files and objects browser

\image html v7_rbrowser.png

*/

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

RBrowser::RBrowser(bool use_rcanvas)
{
   if (gROOT->IsWebDisplayBatch()) {
      ::Warning("RBrowser::RBrowser", "The RBrowser cannot run in web batch mode");
      return;
   }

   std::ostringstream pathtmp;
   pathtmp << gSystem->TempDirectory() << "/command." << gSystem->GetPid() << ".log";
   fPromptFileOutput = pathtmp.str();

   SetUseRCanvas(use_rcanvas);

   fBrowsable.CreateDefaultElements();

   fTimer = std::make_unique<RBrowserTimer>(10, kTRUE, *this);

   fWebWindow = RWebWindow::Create();
   fWebWindow->SetDefaultPage("file:rootui5sys/browser/browser.html");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetCallBacks([this](unsigned connid) { fConnId = connid; SendInitMsg(connid); },
                            [this](unsigned connid, const std::string &arg) { ProcessMsg(connid, arg); });
   fWebWindow->SetGeometry(1200, 700); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue

   fWebWindow->GetManager()->SetShowCallback([this](RWebWindow &win, const RWebDisplayArgs &args) -> bool {

      std::string kind;

      if (args.GetWidgetKind() == "RCanvas")
         kind = "rcanvas";
      else if (args.GetWidgetKind() == "TCanvas")
         kind = "tcanvas";
      else if (args.GetWidgetKind() == "RGeomViewer")
         kind = "geom";
      else if (args.GetWidgetKind() == "RTreeViewer")
         kind = "tree";

      if (!fWebWindow || !fCatchWindowShow || kind.empty())
         return false;

      auto widget = RBrowserWidgetProvider::DetectCatchedWindow(kind, win);
      if (widget) {
         widget->fBrowser = this;
         fWidgets.emplace_back(widget);
         fActiveWidgetName = widget->GetName();
      } else {
         widget = AddCatchedWidget(&win, kind);
      }

      if (widget && fWebWindow && (fWebWindow->NumConnections() > 0))
         fWebWindow->Send(0, NewWidgetMsg(widget));

      return widget ? true : false;
   });

   fWebWindow->GetManager()->SetDeleteCallback([this](RWebWindow &win) -> void {
      for (auto &widget : fWidgets) {
         auto catched = dynamic_cast<RBrowserCatchedWidget *>(widget.get());
         if (catched && (catched->fWindow == &win))
            catched->fWindow = nullptr;
      }
   });

   Show();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RBrowser::~RBrowser()
{
   if (fWebWindow) {
      fWebWindow->GetManager()->SetShowCallback(nullptr);
      fWebWindow->GetManager()->SetDeleteCallback(nullptr);
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process browser request

std::string RBrowser::ProcessBrowserRequest(const std::string &msg)
{
   std::unique_ptr<RBrowserRequest> request;

   if (msg.empty()) {
      request = std::make_unique<RBrowserRequest>();
      request->first = 0;
      request->number = 100;
   } else {
      request = TBufferJSON::FromJSON<RBrowserRequest>(msg);
   }

   if (!request)
      return ""s;

   if (request->path.empty() && fWidgets.empty() && fBrowsable.GetWorkingPath().empty())
      fBrowsable.ClearCache();

   return "BREPL:"s + fBrowsable.ProcessRequest(*request.get());
}

/////////////////////////////////////////////////////////////////////////////////
/// Process file save command in the editor

void RBrowser::ProcessSaveFile(const std::string &fname, const std::string &content)
{
   if (fname.empty()) return;
   R__LOG_DEBUG(0, BrowserLog()) << "SaveFile " << fname << "  content length " << content.length();
   std::ofstream f(fname);
   f << content;
}

/////////////////////////////////////////////////////////////////////////////////
/// Process run macro command in the editor

void RBrowser::ProcessRunMacro(const std::string &file_path)
{
   if (file_path.rfind(".py") == file_path.length() - 3) {
      TString exec;
      exec.Form("TPython::ExecScript(\"%s\");", file_path.c_str());
      gROOT->ProcessLine(exec.Data());
   } else {
      gInterpreter->ExecuteMacro(file_path.c_str());
   }
}

/////////////////////////////////////////////////////////////////////////////////
/// Process dbl click on browser item

std::string RBrowser::ProcessDblClick(unsigned connid, std::vector<std::string> &args)
{
   args.pop_back(); // remove exec string, not used now

   std::string opt = args.back();
   args.pop_back(); // remove option

   auto path = fBrowsable.GetWorkingPath();
   path.insert(path.end(), args.begin(), args.end());

   R__LOG_DEBUG(0, BrowserLog()) << "DoubleClick " << Browsable::RElement::GetPathAsString(path);

   auto elem = fBrowsable.GetSubElement(path);
   if (!elem) return ""s;

   auto dflt_action = elem->GetDefaultAction();

   // special case when canvas is clicked - always start new widget
   if (dflt_action == Browsable::RElement::kActCanvas) {
      std::string widget_kind;

      if (elem->IsCapable(Browsable::RElement::kActDraw7))
         widget_kind = "rcanvas";
      else
         widget_kind = "tcanvas";

      std::string name = widget_kind + std::to_string(++fWidgetCnt);

      auto new_widget = RBrowserWidgetProvider::CreateWidgetFor(widget_kind, name, elem);

      if (!new_widget)
         return ""s;

      // assign back pointer
      new_widget->fBrowser = this;
      fWidgets.emplace_back(new_widget);
      fActiveWidgetName = new_widget->GetName();

      return NewWidgetMsg(new_widget);
   }

   // before display tree or geometry ensure that they read and cached inside element
   if (elem->IsCapable(Browsable::RElement::kActGeom) || elem->IsCapable(Browsable::RElement::kActTree)) {
      elem->GetChildsIter();
   }

   fLastProgressSend = 0;
   Browsable::RProvider::ProgressHandle handle(elem.get(), [this, connid](float progress, void *) {
      SendProgress(connid, progress);
   });

   auto widget = GetActiveWidget();
   if (widget && widget->DrawElement(elem, opt)) {
      widget->SetPath(path);
      return widget->SendWidgetContent();
   }

   // check if element was drawn in other widget and just activate that widget
   auto iter = std::find_if(fWidgets.begin(), fWidgets.end(),
         [path](const std::shared_ptr<RBrowserWidget> &wg) { return path == wg->GetPath(); });

   if (iter != fWidgets.end())
      return "SELECT_WIDGET:"s + (*iter)->GetName();

   // check if object can be drawn in RCanvas even when default action is drawing in TCanvas
   if ((dflt_action == Browsable::RElement::kActDraw6) && GetUseRCanvas() && elem->IsCapable(Browsable::RElement::kActDraw7))
      dflt_action = Browsable::RElement::kActDraw7;

   std::string widget_kind;
   switch(dflt_action) {
      case Browsable::RElement::kActDraw6: widget_kind = "tcanvas"; break;
      case Browsable::RElement::kActDraw7: widget_kind = "rcanvas"; break;
      case Browsable::RElement::kActEdit: widget_kind = "editor"; break;
      case Browsable::RElement::kActImage: widget_kind = "image"; break;
      case Browsable::RElement::kActTree: widget_kind = "tree"; break;
      case Browsable::RElement::kActGeom: widget_kind = "geom"; break;
      default: widget_kind.clear();
   }

   if (!widget_kind.empty()) {
      auto new_widget = AddWidget(widget_kind);
      if (new_widget) {
         // draw object before client side is created - should not be a problem
         // after widget add in browser, connection will be established and data provided
         if (new_widget->DrawElement(elem, opt))
            new_widget->SetPath(path);
         return NewWidgetMsg(new_widget);
      }
   }

   if (elem->IsCapable(Browsable::RElement::kActBrowse) && (elem->GetNumChilds() > 0)) {
      // remove extra index in subitems name
      for (auto &pathelem : path)
         Browsable::RElement::ExtractItemIndex(pathelem);
      fBrowsable.SetWorkingPath(path);
      return GetCurrentWorkingDirectory();
   }

   return ""s;
}

/////////////////////////////////////////////////////////////////////////////////
/// Show or update RBrowser in web window
/// If web window already started - just refresh it like "reload" button does
/// If no web window exists or \param always_start_new_browser configured, starts new window
/// \param args display arguments

void RBrowser::Show(const RWebDisplayArgs &args, bool always_start_new_browser)
{
   if (!fWebWindow->NumConnections() || always_start_new_browser) {
      fWebWindow->Show(args);
   } else {
      SendInitMsg(0);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Hide ROOT Browser

void RBrowser::Hide()
{
   if (fWebWindow)
      fWebWindow->CloseConnections();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Return URL parameter for the window showing ROOT Browser
/// See \ref ROOT::RWebWindow::GetUrl docu for more details

std::string RBrowser::GetWindowUrl(bool remote)
{
   if (fWebWindow)
      return fWebWindow->GetUrl(remote);

   return ""s;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Creates new widget

std::shared_ptr<RBrowserWidget> RBrowser::AddWidget(const std::string &kind)
{
   std::string name = kind + std::to_string(++fWidgetCnt);

   std::shared_ptr<RBrowserWidget> widget;

   if (kind == "editor"s)
      widget = std::make_shared<RBrowserEditorWidget>(name, true);
   else if (kind == "image"s)
      widget = std::make_shared<RBrowserEditorWidget>(name, false);
   else if (kind == "info"s)
      widget = std::make_shared<RBrowserInfoWidget>(name);
   else
      widget = RBrowserWidgetProvider::CreateWidget(kind, name);

   if (!widget) {
      R__LOG_ERROR(BrowserLog()) << "Fail to create widget of kind " << kind;
      return nullptr;
   }

   widget->fBrowser = this;
   fWidgets.emplace_back(widget);
   fActiveWidgetName = name;

   return widget;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Add widget catched from external scripts

std::shared_ptr<RBrowserWidget> RBrowser::AddCatchedWidget(RWebWindow *win, const std::string &kind)
{
   if (!win || kind.empty())
      return nullptr;

   std::string name = "catched"s + std::to_string(++fWidgetCnt);

   auto widget = std::make_shared<RBrowserCatchedWidget>(name, win, kind);

   fWidgets.emplace_back(widget);

   fActiveWidgetName = name;

   return widget;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Create new widget and send init message to the client

void RBrowser::AddInitWidget(const std::string &kind)
{
   auto widget = AddWidget(kind);
   if (widget && fWebWindow && (fWebWindow->NumConnections() > 0))
      fWebWindow->Send(0, NewWidgetMsg(widget));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Find widget by name or kind

std::shared_ptr<RBrowserWidget> RBrowser::FindWidget(const std::string &name, const std::string &kind) const
{
   auto iter = std::find_if(fWidgets.begin(), fWidgets.end(),
         [name, kind](const std::shared_ptr<RBrowserWidget> &widget) {
           return kind.empty() ? name == widget->GetName() : kind == widget->GetKind();
   });

   if (iter != fWidgets.end())
      return *iter;

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Close and delete specified widget

void RBrowser::CloseTab(const std::string &name)
{
   auto iter = std::find_if(fWidgets.begin(), fWidgets.end(), [name](std::shared_ptr<RBrowserWidget> &widget) { return name == widget->GetName(); });
   if (iter != fWidgets.end())
      fWidgets.erase(iter);

   if (fActiveWidgetName == name)
      fActiveWidgetName.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Get content of history file

std::vector<std::string> RBrowser::GetRootHistory()
{
   std::vector<std::string> arr;

   std::string path = gSystem->UnixPathName(gSystem->HomeDirectory());
   path += "/.root_hist" ;
   std::ifstream infile(path);

   if (infile) {
      std::string line;
      while (std::getline(infile, line) && (arr.size() < 1000)) {
         if(!(std::find(arr.begin(), arr.end(), line) != arr.end())) {
            arr.emplace_back(line);
         }
      }
   }

   return arr;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Get content of log file

std::vector<std::string> RBrowser::GetRootLogs()
{
   std::vector<std::string> arr;

   std::ifstream infile(fPromptFileOutput);
   if (infile) {
      std::string line;
      while (std::getline(infile, line) && (arr.size() < 10000)) {
         arr.emplace_back(line);
      }
   }

   return arr;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process client connect

void RBrowser::SendInitMsg(unsigned connid)
{
   std::vector<std::vector<std::string>> reply;

   reply.emplace_back(fBrowsable.GetWorkingPath()); // first element is current path

   for (auto &widget : fWidgets) {
      widget->ResetConn();
      reply.emplace_back(std::vector<std::string>({ widget->GetKind(), widget->GetUrl(), widget->GetName(), widget->GetTitle() }));
   }

   if (!fActiveWidgetName.empty())
      reply.emplace_back(std::vector<std::string>({ "active"s, fActiveWidgetName }));

   auto history = GetRootHistory();
   if (history.size() > 0) {
      history.insert(history.begin(), "history"s);
      reply.emplace_back(history);
   }

   auto logs = GetRootLogs();
   if (logs.size() > 0) {
      logs.insert(logs.begin(), "logs"s);
      reply.emplace_back(logs);
   }

   reply.emplace_back(std::vector<std::string>({
      "drawoptions"s,
      Browsable::RProvider::GetClassDrawOption("TH1"),
      Browsable::RProvider::GetClassDrawOption("TH2"),
      Browsable::RProvider::GetClassDrawOption("TProfile")
   }));

   std::string msg = "INMSG:";
   msg.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());

   fWebWindow->Send(connid, msg);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Send generic progress message to the web window
/// Should show progress bar on client side

void RBrowser::SendProgress(unsigned connid, float progr)
{
   long long millisec = gSystem->Now();

   // let process window events
   fWebWindow->Sync();

   if ((!fLastProgressSendTm || millisec > fLastProgressSendTm - 200) && (progr > fLastProgressSend + 0.04) && fWebWindow->CanSend(connid)) {
      fWebWindow->Send(connid, "PROGRESS:"s + std::to_string(progr));

      fLastProgressSendTm = millisec;
      fLastProgressSend = progr;
   }
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Return the current directory of ROOT

std::string RBrowser::GetCurrentWorkingDirectory()
{
   return "WORKPATH:"s + TBufferJSON::ToJSON(&fBrowsable.GetWorkingPath()).Data();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Create message which send to client to create new widget

std::string RBrowser::NewWidgetMsg(std::shared_ptr<RBrowserWidget> &widget)
{
   std::vector<std::string> arr = { widget->GetKind(), widget->GetUrl(), widget->GetName(), widget->GetTitle(),
                                    Browsable::RElement::GetPathAsString(widget->GetPath()) };
   return "NEWWIDGET:"s + TBufferJSON::ToJSON(&arr, TBufferJSON::kNoSpaces).Data();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Check if any widget was modified and update if necessary

void RBrowser::CheckWidgtesModified(unsigned connid)
{
   std::vector<std::string> del_names;

   for (auto &widget : fWidgets)
      if (!widget->IsValid())
         del_names.push_back(widget->GetName());

   if (!del_names.empty())
      fWebWindow->Send(connid, "CLOSE_WIDGETS:"s + TBufferJSON::ToJSON(&del_names, TBufferJSON::kNoSpaces).Data());

   for (auto name : del_names)
      CloseTab(name);

   for (auto &widget : fWidgets)
      widget->CheckModified();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process postponed requests - decouple from websocket handling
/// Only requests which can take longer time should be postponed

void RBrowser::ProcessPostponedRequests()
{
   if (fPostponed.empty())
      return;

   auto arr = fPostponed[0];
   fPostponed.erase(fPostponed.begin(), fPostponed.begin()+1);
   if (fPostponed.empty())
      fTimer->TurnOff();

   std::string reply;
   unsigned connid = std::stoul(arr.back()); arr.pop_back();
   std::string kind = arr.back(); arr.pop_back();

   if (kind == "DBLCLK") {
      reply = ProcessDblClick(connid, arr);
      if (reply.empty()) reply = "NOPE";
   }

   if (!reply.empty())
      fWebWindow->Send(connid, reply);
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Process received message from the client

void RBrowser::ProcessMsg(unsigned connid, const std::string &arg0)
{
   R__LOG_DEBUG(0, BrowserLog()) << "ProcessMsg  len " << arg0.length() << " substr(30) " << arg0.substr(0, 30);

   std::string kind, msg;
   auto pos = arg0.find(":");
   if (pos == std::string::npos) {
      kind = arg0;
   } else {
      kind = arg0.substr(0, pos);
      msg = arg0.substr(pos+1);
   }

   if (kind == "QUIT_ROOT") {

      fWebWindow->TerminateROOT();

   } else if (kind == "BRREQ") {
      // central place for processing browser requests
      auto json = ProcessBrowserRequest(msg);
      if (!json.empty()) fWebWindow->Send(connid, json);

   } else if (kind == "DBLCLK") {

      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(msg);
      if (arr && (arr->size() > 2)) {
         arr->push_back(kind);
         arr->push_back(std::to_string(connid));
         fPostponed.push_back(*arr);
         if (fPostponed.size() == 1)
            fTimer->TurnOn();
      } else {
         fWebWindow->Send(connid, "NOPE");
      }

   } else if (kind == "WIDGET_SELECTED") {
      fActiveWidgetName = msg;
      auto widget = GetActiveWidget();
      if (widget) {
         auto reply = widget->SendWidgetContent();
         if (!reply.empty()) fWebWindow->Send(connid, reply);
      }
   } else if (kind == "CLOSE_TAB") {
      CloseTab(msg);
   } else if (kind == "GETWORKPATH") {
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (kind == "CHPATH") {
      auto path = TBufferJSON::FromJSON<Browsable::RElementPath_t>(msg);
      if (path) fBrowsable.SetWorkingPath(*path);
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (kind == "CMD") {
      std::string sPrompt = "root []";
      TApplication *app = gROOT->GetApplication();
      if (app->InheritsFrom("TRint")) {
         sPrompt = ((TRint*)gROOT->GetApplication())->GetPrompt();
         Gl_histadd((char *)msg.c_str());
      }

      std::ofstream ofs(fPromptFileOutput, std::ofstream::out | std::ofstream::app);
      ofs << sPrompt << msg << std::endl;
      ofs.close();

      gSystem->RedirectOutput(fPromptFileOutput.c_str(), "a");
      gROOT->ProcessLine(msg.c_str());
      gSystem->RedirectOutput(nullptr);

      if (msg == ".g"s) {
         auto widget = std::dynamic_pointer_cast<RBrowserInfoWidget>(FindWidget(""s, "info"s));
         if (!widget) {
            auto new_widget = AddWidget("info"s);
            fWebWindow->Send(connid, NewWidgetMsg(new_widget));
            widget = std::dynamic_pointer_cast<RBrowserInfoWidget>(new_widget);
         } else if (fActiveWidgetName != widget->GetName()) {
            fWebWindow->Send(connid, "SELECT_WIDGET:"s + widget->GetName());
            fActiveWidgetName = widget->GetName();
         }

         if (widget)
            widget->RefreshFromLogs(sPrompt + msg, GetRootLogs());
      }

      CheckWidgtesModified(connid);
   } else if (kind == "GETHISTORY") {

      auto history = GetRootHistory();

      fWebWindow->Send(connid, "HISTORY:"s + TBufferJSON::ToJSON(&history, TBufferJSON::kNoSpaces).Data());
   } else if (kind == "GETLOGS") {

      auto logs = GetRootLogs();
      fWebWindow->Send(connid, "LOGS:"s + TBufferJSON::ToJSON(&logs, TBufferJSON::kNoSpaces).Data());

   } else if (RFileDialog::IsMessageToStartDialog(arg0)) {

      RFileDialog::Embed(fWebWindow, connid, arg0);

   } else if (kind == "SYNCEDITOR") {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(msg);
      if (arr && (arr->size() > 4)) {
         auto editor = std::dynamic_pointer_cast<RBrowserEditorWidget>(FindWidget(arr->at(0)));
         if (editor) {
            editor->fFirstSend = true;
            editor->fTitle = arr->at(1);
            editor->fFileName = arr->at(2);
            if (!arr->at(3).empty()) editor->fContent = arr->at(4);
            if ((arr->size() == 6) && (arr->at(5) == "SAVE"))
               ProcessSaveFile(editor->fFileName, editor->fContent);
            if ((arr->size() == 6) && (arr->at(5) == "RUN")) {
               ProcessSaveFile(editor->fFileName, editor->fContent);
               ProcessRunMacro(editor->fFileName);
               CheckWidgtesModified(connid);
            }
         }
      }
   } else if (kind == "GETINFO") {
      auto info = std::dynamic_pointer_cast<RBrowserInfoWidget>(FindWidget(msg));
      if (info) {
         info->Refresh();
         fWebWindow->Send(connid, info->SendWidgetContent());
      }
   } else if (kind == "NEWWIDGET") {
      auto widget = AddWidget(msg);
      if (widget)
         fWebWindow->Send(connid, NewWidgetMsg(widget));
   } else if (kind == "NEWCHANNEL") {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(msg);
      if (arr && (arr->size() == 2)) {
         auto widget = FindWidget((*arr)[0]);
         if (widget)
            RWebWindow::ShowWindow(widget->GetWindow(), { fWebWindow, connid, std::stoi((*arr)[1]) });
      }
   } else if (kind == "CDWORKDIR") {
      auto wrkdir = Browsable::RSysFile::GetWorkingPath();
      if (fBrowsable.GetWorkingPath() != wrkdir) {
         fBrowsable.SetWorkingPath(wrkdir);
      } else {
         fBrowsable.SetWorkingPath({});
      }
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (kind == "OPTIONS") {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(msg);
      if (arr && (arr->size() == 3)) {
         Browsable::RProvider::SetClassDrawOption("TH1", (*arr)[0]);
         Browsable::RProvider::SetClassDrawOption("TH2", (*arr)[1]);
         Browsable::RProvider::SetClassDrawOption("TProfile", (*arr)[2]);
      }
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Set working path in the browser

void RBrowser::SetWorkingPath(const std::string &path)
{
   auto p = Browsable::RElement::ParsePath(path);
   auto elem = fBrowsable.GetSubElement(p);
   if (elem) {
      fBrowsable.SetWorkingPath(p);
      if (fWebWindow && (fWebWindow->NumConnections() > 0))
         fWebWindow->Send(0, GetCurrentWorkingDirectory());
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Activate widget in RBrowser
/// One should specify title and (optionally) kind of widget like "tcanvas" or "geom"

bool RBrowser::ActivateWidget(const std::string &title, const std::string &kind)
{
   if (title.empty())
      return false;

   for (auto &widget : fWidgets) {

      if (widget->GetTitle() != title)
         continue;

      if (!kind.empty() && (widget->GetKind() != kind))
         continue;

      if (fWebWindow)
         fWebWindow->Send(0, "SELECT_WIDGET:"s + widget->GetName());
      else
         fActiveWidgetName = widget->GetName();
      return true;
   }

   return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Set handle which will be cleared when connection is closed

void RBrowser::ClearOnClose(const std::shared_ptr<void> &handle)
{
   fWebWindow->SetClearOnClose(handle);
}
