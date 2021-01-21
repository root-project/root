/// \file ROOT/RBrowser.cxx
/// \ingroup rbrowser
/// \author Bertrand Bellenot <bertrand.bellenot@cern.ch>
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2019-02-28
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RBrowser.hxx>

#include <ROOT/Browsable/RGroup.hxx>
#include <ROOT/Browsable/RWrapper.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/TObjectHolder.hxx>
#include <ROOT/Browsable/RSysFile.hxx>

#include <ROOT/RLogger.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RFileDialog.hxx>
#include <ROOT/RCanvas.hxx>

#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TWebCanvas.h"
#include "TCanvas.h"
#include "TFolder.h"
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

using namespace ROOT::Experimental;

/** \class ROOT::Experimental::RBrowser
\ingroup rbrowser

web-based ROOT Browser prototype.
*/

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

RBrowser::RBrowser(bool use_rcanvas)
{
   SetUseRCanvas(use_rcanvas);

   auto comp = std::make_shared<Browsable::RGroup>("top","Root browser");

   auto seldir = Browsable::RSysFile::ProvideTopEntries(comp);

   std::unique_ptr<Browsable::RHolder> rootfold = std::make_unique<Browsable::TObjectHolder>(gROOT->GetRootFolder(), kFALSE);
   auto elem_root = Browsable::RProvider::Browse(rootfold);
   if (elem_root)
      comp->Add(std::make_shared<Browsable::RWrapper>("root", elem_root));

   std::unique_ptr<Browsable::RHolder> rootfiles = std::make_unique<Browsable::TObjectHolder>(gROOT->GetListOfFiles(), kFALSE);
   auto elem_files = Browsable::RProvider::Browse(rootfiles);
   if (elem_files)
      comp->Add(std::make_shared<Browsable::RWrapper>("ROOT Files", elem_files));

   fBrowsable.SetTopElement(comp);

   fBrowsable.SetWorkingDirectory(seldir);

   fWebWindow = RWebWindow::Create();
   fWebWindow->SetDefaultPage("file:rootui5sys/browser/browser.html");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetCallBacks([this](unsigned connid) { fConnId = connid; SendInitMsg(connid); },
                            [this](unsigned connid, const std::string &arg) { ProcessMsg(connid, arg); });
   fWebWindow->SetGeometry(1200, 700); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue

   Show();

   // add first canvas by default

   if (GetUseRCanvas())
      AddRCanvas();
   else
      AddCanvas();

   // AddPage(true); // one can add empty editor if necessary
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RBrowser::~RBrowser()
{
   fCanvases.clear();
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Process browser request

std::string RBrowser::ProcessBrowserRequest(const std::string &msg)
{
   std::string res;

   std::unique_ptr<RBrowserRequest> request;

   if (msg.empty()) {
      request = std::make_unique<RBrowserRequest>();
      request->path = "/";
      request->first = 0;
      request->number = 100;
   } else {
      request = TBufferJSON::FromJSON<RBrowserRequest>(msg);
   }

   if (!request)
      return res;

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
/// Process file save command in the editor

long RBrowser::ProcessRunMacro(const std::string &file_path)
{
   return gInterpreter->ExecuteMacro(file_path.c_str());
}

/////////////////////////////////////////////////////////////////////////////////
/// Send editor content to the client

std::string RBrowser::SendPageContent(BrowserPage *editor)
{
   if (!editor) return ""s;

   editor->fFirstSend = true;
   std::vector<std::string> args = { editor->fName, editor->fTitle, editor->fFileName, editor->fContent };

   std::string msg = editor->fIsEditor ? "EDITOR:"s : "IMAGE:"s;
   msg += TBufferJSON::ToJSON(&args).Data();
   return msg;
}

/////////////////////////////////////////////////////////////////////////////////
/// Process dbl click on browser item

std::string RBrowser::ProcessDblClick(const std::string &item_path, const std::string &drawingOptions, const std::string &)
{
   R__LOG_DEBUG(0, BrowserLog()) << "DoubleClick " << item_path;

   auto elem = fBrowsable.GetElement(item_path);
   if (!elem) return ""s;

   auto page = GetActivePage();

   // check if element can provide text for text editor
   if (page && page->fIsEditor && elem->IsCapable(Browsable::RElement::kActEdit)) {
      auto code = elem->GetContent("text");
      if (!code.empty()) {
         page->fContent = code;
         page->fTitle = elem->GetName();
         page->fFileName = elem->GetContent("filename");
      } else {
         auto json = elem->GetContent("json");
         if (!json.empty()) {
            page->fContent = json;
            page->fTitle = elem->GetName() + ".json";
            page->fFileName = "";
         } else {
            page = nullptr;
         }
      }
      if (page) {
         page->fItemPath = item_path;
         return SendPageContent(page);
      }
   }

   // check if element can provide image for the viewer
   if (page && !page->fIsEditor && elem->IsCapable(Browsable::RElement::kActImage)) {
      auto img = elem->GetContent("image64");
      if (!img.empty()) {
         page->fContent = img;
         page->fTitle = elem->GetName();
         page->fFileName = elem->GetContent("filename");
         page->fItemPath = item_path;

         return SendPageContent(page);
      }
   }

   auto rcanv = GetActiveRCanvas();
   if (rcanv && elem->IsCapable(Browsable::RElement::kActDraw7)) {

      std::shared_ptr<RPadBase> subpad = rcanv;

      auto obj = elem->GetObject();
      if (obj && Browsable::RProvider::Draw7(subpad, obj, drawingOptions)) {
         rcanv->Modified();
         rcanv->Update(true);
         return ""s;
         // return "SELECT_TAB:"s + rcanv->GetTitle();
      }
   }

   auto canv = GetActiveCanvas();
   if (canv && elem->IsCapable(Browsable::RElement::kActDraw6)) {

      auto obj = elem->GetObject();

      if (obj && Browsable::RProvider::Draw6(canv, obj, drawingOptions)) {
         canv->ForceUpdate(); // force update async - do not wait for confirmation
         return ""s;
         // return "SELECT_TAB:"s + canv->GetName();
      }
   }

   auto dflt_action = elem->GetDefaultAction();

   if (dflt_action == Browsable::RElement::kActImage) {
      auto viewer = FindPageFor(item_path, false);
      if (viewer) return "SELECT_TAB:"s + viewer->fName;

      auto img = elem->GetContent("image64");
      if (!img.empty()) {
         viewer = AddPage(false);

         viewer->fContent = img;
         viewer->fTitle = elem->GetName();
         viewer->fFileName = elem->GetContent("filename");
         viewer->fItemPath = item_path;

         std::vector<std::string> reply = { viewer->GetKind(), viewer->fName, viewer->fTitle };
         return "NEWTAB:"s + TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data();
      } else {
         return ""s;
      }
   }

   if (dflt_action == Browsable::RElement::kActEdit) {
      auto editor = FindPageFor(item_path, true);
      if (editor) return "SELECT_TAB:"s + editor->fName;

      auto code = elem->GetContent("text");
      if (!code.empty()) {
         editor = AddPage(true);
         editor->fContent = code;
         editor->fTitle = elem->GetName();
         editor->fFileName = elem->GetContent("filename");
      } else {
         auto json = elem->GetContent("json");
         if (!json.empty()) {
            editor = AddPage(true);
            editor->fContent = json;
            editor->fTitle = elem->GetName() + ".json";
            editor->fFileName = "";
         }
      }

      if (editor) {
         editor->fItemPath = item_path;
         std::vector<std::string> reply = { editor->GetKind(), editor->fName, editor->fTitle };
         return "NEWTAB:"s + TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data();
      }
   }

   return ""s;
}

/////////////////////////////////////////////////////////////////////////////////
/// Show or update RBrowser in web window
/// If web window already started - just refresh it like "reload" button does
/// If no web window exists or \param always_start_new_browser configured, starts new window

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
   if (!fWebWindow)
      return;

   fWebWindow->CloseConnections();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Creates new editor, return name

RBrowser::BrowserPage *RBrowser::AddPage(bool is_editor)
{
   fPages.emplace_back(std::make_unique<BrowserPage>(is_editor));

   auto editor = fPages.back().get();

   editor->fName = is_editor ? "CodeEditor"s : "ImageViewer"s;
   editor->fName += std::to_string(fPagesCnt++);
   editor->fTitle = "untitled";

   return editor;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Create new web canvas, invoked when new canvas created on client side

TCanvas *RBrowser::AddCanvas()
{
   TString canv_name;
   canv_name.Form("webcanv%d", (int)(fCanvases.size()+1));

   auto canv = std::make_unique<TCanvas>(kFALSE);
   canv->SetName(canv_name.Data());
   canv->SetTitle(canv_name.Data());
   canv->ResetBit(TCanvas::kShowEditor);
   canv->ResetBit(TCanvas::kShowToolBar);
   canv->SetCanvas(canv.get());
   canv->SetBatch(kTRUE); // mark canvas as batch
   canv->SetEditable(kTRUE); // ensure fPrimitives are created
   fActiveTab = canv->GetName();

   // create implementation
   TWebCanvas *web = new TWebCanvas(canv.get(), "title", 0, 0, 800, 600);

   // assign implementation
   canv->SetCanvasImp(web);

   // initialize web window, but not start new web browser
   web->ShowWebWindow("embed");

   fCanvases.emplace_back(std::move(canv));

   return fCanvases.back().get();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Creates RCanvas for the output

std::shared_ptr<RCanvas> RBrowser::AddRCanvas()
{
   std::string name = "rcanv"s + std::to_string(fRCanvases.size()+1);

   auto canv = RCanvas::Create(name);

   canv->Show("embed");

   fActiveTab = name;

   fRCanvases.emplace_back(canv);

   return canv;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns relative URL for canvas - required for client to establish connection

std::string RBrowser::GetCanvasUrl(TCanvas *canv)
{
   TWebCanvas *web = dynamic_cast<TWebCanvas *>(canv->GetCanvasImp());
   return fWebWindow->GetRelativeAddr(web->GetWebWindow());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns relative URL for canvas - required for client to establish connection

std::string RBrowser::GetRCanvasUrl(std::shared_ptr<RCanvas> &canv)
{
   return "../"s + canv->GetWindowAddr() + "/"s;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns active web canvas (if any)

TCanvas *RBrowser::GetActiveCanvas() const
{
   auto iter = std::find_if(fCanvases.begin(), fCanvases.end(), [this](const std::unique_ptr<TCanvas> &canv) { return fActiveTab == canv->GetName(); });

   if (iter != fCanvases.end())
      return iter->get();

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns active RCanvas (if any)

std::shared_ptr<RCanvas> RBrowser::GetActiveRCanvas() const
{
   auto iter = std::find_if(fRCanvases.begin(), fRCanvases.end(), [this](const std::shared_ptr<RCanvas> &canv) { return fActiveTab == canv->GetTitle(); });

   if (iter != fRCanvases.end())
      return *iter;

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns editor/image page with provided name

RBrowser::BrowserPage *RBrowser::GetPage(const std::string &name) const
{
   if (name.empty()) return nullptr;

   auto iter = std::find_if(fPages.begin(), fPages.end(), [this,name](const std::unique_ptr<BrowserPage> &page) { return name == page->fName; });

   return (iter != fPages.end()) ? iter->get() : nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Find editor for specified item path

RBrowser::BrowserPage *RBrowser::FindPageFor(const std::string &item_path, bool is_editor)
{
   auto iter = std::find_if(fPages.begin(), fPages.end(),
            [this,item_path,is_editor](const std::unique_ptr<BrowserPage> &page) {
              return (item_path == page->fItemPath) && (is_editor == page->fIsEditor);
            });

   return (iter != fPages.end()) ? iter->get() : nullptr;

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Close and delete specified canvas
/// Check both list of TCanvas and list of RCanvas

void RBrowser::CloseTab(const std::string &name)
{
   auto iter1 = std::find_if(fCanvases.begin(), fCanvases.end(), [name](std::unique_ptr<TCanvas> &canv) { return name == canv->GetName(); });
   if (iter1 != fCanvases.end())
      fCanvases.erase(iter1);

   auto iter2 = std::find_if(fRCanvases.begin(), fRCanvases.end(), [name](const std::shared_ptr<RCanvas> &canv) { return name == canv->GetTitle(); });
   if (iter2 != fRCanvases.end()) {
      (*iter2)->Remove();
      fRCanvases.erase(iter2);
   }

   auto iter3 = std::find_if(fPages.begin(), fPages.end(), [name](std::unique_ptr<BrowserPage> &page) { return name == page->fName; });
   if (iter3 != fPages.end())
      fPages.erase(iter3);

   if (fActiveTab == name)
      fActiveTab.clear();
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

   std::ostringstream pathtmp;
   pathtmp << gSystem->TempDirectory() << "/command." << gSystem->GetPid() << ".log";

   std::ifstream infile(pathtmp.str());
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

   for (auto &canv : fCanvases) {
      auto url = GetCanvasUrl(canv.get());
      std::string name = canv->GetName();
      std::vector<std::string> arr = { "root6", url, name };
      reply.emplace_back(arr);
   }

   for (auto &canv : fRCanvases) {
      auto url = GetRCanvasUrl(canv);
      std::string name = canv->GetTitle();
      std::vector<std::string> arr = { "root7", url, name };
      reply.emplace_back(arr);
   }

   for (auto &edit : fPages) {
      edit->fFirstSend = false; // mark that content was not provided
      reply.emplace_back(std::vector<std::string>({ edit->GetKind(), edit->fName, edit->fTitle }));
   }

   if (!fActiveTab.empty())
      reply.emplace_back(std::vector<std::string>({ "active", fActiveTab }));

   auto history = GetRootHistory();
   if (history.size() > 0) {
      history.insert(history.begin(), "history");
      reply.emplace_back(history);
   }

   auto logs = GetRootLogs();
   if (logs.size() > 0) {
      logs.insert(logs.begin(), "logs");
      reply.emplace_back(logs);
   }

   std::string msg = "INMSG:";
   msg.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());

   fWebWindow->Send(connid, msg);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return the current directory of ROOT

std::string RBrowser::GetCurrentWorkingDirectory()
{
   return "WORKPATH:"s + TBufferJSON::ToJSON(&fBrowsable.GetWorkingPath()).Data();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Create requested tab, return message which should be send to client

std::string RBrowser::ProcessNewTab(const std::string &kind)
{
   std::vector<std::string> reply;

   if (kind == "NEWRCANVAS") {
      auto canv = AddRCanvas();
      auto url = GetRCanvasUrl(canv);
      reply = {"root7"s, url, canv->GetTitle()};
   } else if (kind == "NEWTCANVAS") {
      auto canv = AddCanvas();
      auto url = GetCanvasUrl(canv);
      reply = {"root6"s, url, std::string(canv->GetName())};
   } else if ((kind == "NEWEDITOR") || (kind == "NEWVIEWER")) {
      auto edit = AddPage(kind == "NEWEDITOR");
      reply = {edit->GetKind(), edit->fName, edit->fTitle};
   } else {
      return ""s;
   }

   return "NEWTAB:"s + TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data();
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

      std::string reply;

      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(msg);
      if (arr && (arr->size() == 3))
         reply = ProcessDblClick(arr->at(0), arr->at(1), arr->at(2));

      if (!reply.empty())
         fWebWindow->Send(connid, reply);

   } else if (kind == "TAB_SELECTED") {
      fActiveTab = msg;
      std::string reply;
      auto editor = GetActivePage();
      if (editor && !editor->fFirstSend)
         reply = SendPageContent(editor);
      if (!reply.empty()) fWebWindow->Send(connid, reply);
   } else if (kind == "CLOSE_TAB") {
      CloseTab(msg);
   } else if (kind == "GETWORKPATH") {
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (kind == "CHPATH") {
      auto path = TBufferJSON::FromJSON<Browsable::RElementPath_t>(msg);
      if (path) fBrowsable.SetWorkingPath(*path);
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (kind == "CHDIR") {
      fBrowsable.SetWorkingDirectory(msg);
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (kind == "CMD") {
      std::string sPrompt = "root []";
      std::ostringstream pathtmp;
      pathtmp << gSystem->TempDirectory() << "/command." << gSystem->GetPid() << ".log";
      TApplication *app = gROOT->GetApplication();
      if (app->InheritsFrom("TRint")) {
         sPrompt = ((TRint*)gROOT->GetApplication())->GetPrompt();
         Gl_histadd((char *)msg.c_str());
      }

      std::ofstream ofs(pathtmp.str(), std::ofstream::out | std::ofstream::app);
      ofs << sPrompt << msg;
      ofs.close();

      gSystem->RedirectOutput(pathtmp.str().c_str(), "a");
      gROOT->ProcessLine(msg.c_str());
      gSystem->RedirectOutput(0);
   } else if (kind == "GETHISTORY") {

      auto history = GetRootHistory();

      fWebWindow->Send(connid, "HISTORY:"s + TBufferJSON::ToJSON(&history, TBufferJSON::kNoSpaces).Data());
   } else if (kind == "GETLOGS") {

      auto logs = GetRootLogs();
      fWebWindow->Send(connid, "LOGS:"s + TBufferJSON::ToJSON(&logs, TBufferJSON::kNoSpaces).Data());

   } else if (kind == "FILEDIALOG") {
      RFileDialog::Embedded(fWebWindow, arg0);
   } else if (kind == "SYNCEDITOR") {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(msg);
      if (arr && (arr->size() > 4)) {
         auto editor = GetPage(arr->at(0));
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
            }

         }
      }
   } else {
      auto reply = ProcessNewTab(kind);
      if (!reply.empty()) fWebWindow->Send(connid, reply);
   }
}
