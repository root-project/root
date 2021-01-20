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

   AddEditor(true);
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

void RBrowser::ProcessSaveFile(const std::string &arg)
{
   auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg);
   if (!arr || (arr->size()!=2)) {
      R__LOG_ERROR(BrowserLog()) << "SaveFile failure, json array should have two items " << arg;
   } else {
      R__LOG_DEBUG(0, BrowserLog()) << "SaveFile " << arr->at(0) << "  content length " << arr->at(1).length();
      std::ofstream f(arr->at(0));
      f << arr->at(1);
   }
}

/////////////////////////////////////////////////////////////////////////////////
/// Process file save command in the editor

long RBrowser::ProcessRunCommand(const std::string &file_path)
{
   return gInterpreter->ExecuteMacro(file_path.c_str());
}

/////////////////////////////////////////////////////////////////////////////////
/// Send editor content to the client

std::string RBrowser::SendEditorContent(EditorPage *editor)
{
   if (!editor) return ""s;

   editor->fFirstSend = true;
   std::vector<std::string> args = { editor->fName, editor->fFileName, editor->fContent };

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

   auto editor = GetActiveEditor();

   // check if element can provide text for text editor
   if (editor && editor->fIsEditor && elem->IsCapable(Browsable::RElement::kActEdit)) {
      auto code = elem->GetContent("text");
      if (!code.empty()) {
         editor->fContent = code;
         editor->fFileName = elem->GetContent("filename");
         if (editor->fFileName.empty())
            editor->fFileName = elem->GetName();
      } else {
         auto json = elem->GetContent("json");
         if (!json.empty()) {
            editor->fContent = json;
            editor->fFileName = elem->GetName() + ".json";
         } else {
            editor = nullptr;
         }
      }
      if (editor)
         return SendEditorContent(editor);
   }

   // check if element can provide image for the viewer
   if (editor && !editor->fIsEditor && elem->IsCapable(Browsable::RElement::kActImage)) {
      auto img = elem->GetContent("image64");
      if (!img.empty()) {
         editor->fContent = img;
         editor->fFileName = elem->GetContent("filename");
         if (editor->fFileName.empty())
            editor->fFileName = elem->GetName();

         return SendEditorContent(editor);
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

   return ""s;

   // TODO: one can send id of editor or canvas to be sure when sending back reply

   if (drawingOptions == "$$$image$$$") {
      auto img = elem->GetContent("image64");
      if (img.empty())
         return ""s;

      auto fname = elem->GetContent("filename");
      if (fname.empty())
         fname = elem->GetName();

      std::vector<std::string> args = { fname, img };

      return "FIMG:"s + TBufferJSON::ToJSON(&args).Data();
   }

   if (drawingOptions == "$$$execute$$$") {

      std::string ext = item_path.substr(item_path.find_last_of(".") + 1);

      //lower the char
      std::for_each(ext.begin(), ext.end(), [](char & c) {
         c = ::tolower(c);
      });

      if(ext == "c" || ext == "cpp" || ext == "cxx") {
         ProcessRunCommand(elem->GetContent("filename"));
         return "";
      }
   }

   R__LOG_DEBUG(0, BrowserLog()) << "No active canvas to process dbl click";

   return "";
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

RBrowser::EditorPage *RBrowser::AddEditor(bool is_editor)
{
   fEditors.emplace_back(std::make_unique<EditorPage>(is_editor));

   auto editor = fEditors.back().get();

   editor->fName = is_editor ? "CodeEditor"s : "ImageViewer"s;
   editor->fName += std::to_string(fEditorsCnt++);
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
/// Returns active text editor (if any)

RBrowser::EditorPage *RBrowser::GetActiveEditor() const
{
   auto iter = std::find_if(fEditors.begin(), fEditors.end(), [this](const std::unique_ptr<EditorPage> &page) { return fActiveTab == page->fName; });

   if (iter != fEditors.end())
      return iter->get();

   return nullptr;
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

   auto iter3 = std::find_if(fEditors.begin(), fEditors.end(), [name](std::unique_ptr<EditorPage> &page) { return name == page->fName; });
   if (iter3 != fEditors.end())
      fEditors.erase(iter3);

   if (fActiveTab == name)
      fActiveTab.clear();
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

   for (auto &edit : fEditors) {
      edit->fFirstSend = false; // mark that content was not provided
      reply.emplace_back(std::vector<std::string>({ edit->GetKind(), edit->fName, edit->fTitle }));
   }

   if (!fActiveTab.empty())
      reply.emplace_back(std::vector<std::string>({ "active", fActiveTab }));

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
   } else if (kind == "NEWRCANVAS") {

      auto canv = AddRCanvas();
      auto url = GetRCanvasUrl(canv);

      std::vector<std::string> reply = {"root7"s, url, canv->GetTitle()};
      std::string res = "NEWTAB:";
      res.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());
      fWebWindow->Send(connid, res);
   } else if (kind == "NEWTCANVAS") {

      auto canv = AddCanvas();
      auto url = GetCanvasUrl(canv);

      std::vector<std::string> reply = {"root6"s, url, std::string(canv->GetName())};
      std::string res = "NEWTAB:";
      res.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());
      fWebWindow->Send(connid, res);
   } else if ((kind == "NEWEDITOR") || (kind == "NEWVIEWER")) {

      auto edit = AddEditor(kind == "NEWEDITOR");
      std::vector<std::string> reply = { edit->GetKind(), edit->fName, edit->fTitle};
      std::string res = "NEWTAB:";
      res.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());
      fWebWindow->Send(connid, res);

   } else if (kind == "DBLCLK") {

      std::string reply;

      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(msg);
      if (arr && (arr->size() == 3))
         reply = ProcessDblClick(arr->at(0), arr->at(1), arr->at(2));

      if (!reply.empty())
         fWebWindow->Send(connid, reply);

   } else if (kind == "RUNMACRO") {
      ProcessRunCommand(msg);
   } else if (kind == "SELECT_TAB") {
      fActiveTab = msg;
      std::string reply;
      auto editor = GetActiveEditor();
      if (editor && !editor->fFirstSend)
         reply = SendEditorContent(editor);
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
   } else if (kind == "ROOTHIST") {
      std::ostringstream path;
      path << gSystem->UnixPathName(gSystem->HomeDirectory()) << "/.root_hist" ;
      std::ifstream infile(path.str());

      std::vector<std::string> unique_vector;
      std::string line;
      while (std::getline(infile, line)) {
         if(!(std::find(unique_vector.begin(), unique_vector.end(), line) != unique_vector.end())) {
            unique_vector.push_back(line);
         }
      }
      std::string result;
      for (const auto &piece : unique_vector) result += piece + ",";
      fWebWindow->Send(connid, "HIST:"s + result);
   } else if (kind == "LOGS") {
      std::ostringstream pathtmp;
      pathtmp << gSystem->TempDirectory() << "/command." << gSystem->GetPid() << ".log";
      TString result;
      std::ifstream instr(pathtmp.str().c_str());
      result.ReadFile(instr);
      fWebWindow->Send(connid, "LOGS:"s + result.Data());
   } else if (kind == "FILEDIALOG") {
      RFileDialog::Embedded(fWebWindow, arg0);
   } else if (kind == "SAVEFILE") {
      ProcessSaveFile(msg);
   }
}
