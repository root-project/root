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

#include <ROOT/RLogger.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RObjectDrawable.hxx>
#include <ROOT/RDrawableProvider.hxx>
#include <ROOT/RBrowsableSysFile.hxx>
#include <ROOT/RBrowsableTObject.hxx>
#include <ROOT/RCanvas.hxx>

#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TWebCanvas.h"
#include "TCanvas.h"
#include "TBufferJSON.h"
#include "TApplication.h"
#include "TRint.h"
#include "TLeaf.h"
#include "TBranch.h"
#include "TTree.h"
#include "TH1.h"
#include "TFolder.h"
#include "Getline.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <fstream>

using namespace std::string_literals;

using namespace ROOT::Experimental::Browsable;

/** \class ROOT::Experimental::RBrowser
\ingroup rbrowser

web-based ROOT Browser prototype.
*/

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

ROOT::Experimental::RBrowser::RBrowser(bool use_rcanvas)
{
   SetUseRCanvas(use_rcanvas);

   std::string workdir = gSystem->UnixPathName(gSystem->WorkingDirectory());
   std::string homedir = gSystem->UnixPathName(gSystem->HomeDirectory());

   std::string seldir, topdir, toplbl;

   printf("Current dir %s home %s\n", workdir.c_str(), homedir.c_str());

#ifdef WIN32
   auto pos = workdir.find(":");
   if (pos != std::string::npos) {
      toplbl = workdir.substr(0,pos+1);
      seldir = workdir;
   } else {
      seldir = toplbl = "c:"; // suppose that
   }

   topdir = toplbl + "\\";
#else
   topdir = "/";
   toplbl = "fs";
   seldir = "/fs"s + workdir;

#endif

   auto comp = std::make_shared<Browsable::RComposite>("top","very top of Root browser");
   comp->Add(std::make_shared<Browsable::RWrapper>(toplbl,std::make_unique<SysFileElement>(topdir)));
   if (!homedir.empty())
      comp->Add(std::make_shared<Browsable::RWrapper>("home",std::make_unique<SysFileElement>(homedir)));

   std::unique_ptr<RHolder> rootfold = std::make_unique<RTObjectHolder>(gROOT->GetRootFolder(), kFALSE);
   auto elem_root = Browsable::RProvider::Browse(rootfold);
   if (elem_root)
      comp->Add(std::make_shared<Browsable::RWrapper>("root",elem_root));

   std::unique_ptr<RHolder> rootfiles = std::make_unique<RTObjectHolder>(gROOT->GetListOfFiles(), kFALSE);
   auto elem_files = Browsable::RProvider::Browse(rootfiles);
   if (elem_files)
      comp->Add(std::make_shared<Browsable::RWrapper>("ROOT Files",elem_files));


   fBrowsable.SetTopElement(comp);

   fBrowsable.SetWorkingDirectory(seldir);

   fWebWindow = RWebWindow::Create();
   fWebWindow->SetDefaultPage("file:rootui5sys/browser/browser.html");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetCallBacks([this](unsigned connid) { fConnId = connid; SendInitMsg(connid); },
                            [this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); });
   fWebWindow->SetGeometry(1200, 700); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue

   Show();

   // add first canvas by default

   if (GetUseRCanvas())
      AddRCanvas();
   else
      AddCanvas();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

ROOT::Experimental::RBrowser::~RBrowser()
{
   fCanvases.clear();
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Process browser request

std::string ROOT::Experimental::RBrowser::ProcessBrowserRequest(const std::string &msg)
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

bool ROOT::Experimental::RBrowser::ProcessSaveFile(const std::string &file_path)
{
   // Split the path (filename + text)
   std::vector<std::string> split;
   std::string buffer;
   std::istringstream path(file_path);
   if (std::getline(path, buffer, ':'))
      split.push_back(buffer);
   if (std::getline(path, buffer, '\0'))
      split.push_back(buffer);
   // TODO: can be done with RElement as well
   std::ofstream ostrm(split[0]);
   ostrm << split[1];
   return true;
}

/////////////////////////////////////////////////////////////////////////////////
/// Process file save command in the editor

long ROOT::Experimental::RBrowser::ProcessRunCommand(const std::string &file_path)
{
   // Split the path (filename + text)
   std::vector<std::string> split;
   std::string buffer;
   std::istringstream path(file_path);
   if (std::getline(path, buffer, ':'))
      split.push_back(buffer);
   if (std::getline(path, buffer, '\0'))
      split.push_back(buffer);
   return gInterpreter->ExecuteMacro(split[0].c_str());
}

/////////////////////////////////////////////////////////////////////////////////
/// Process dbl click on browser item

std::string ROOT::Experimental::RBrowser::ProcessDblClick(const std::string &item_path, const std::string &drawingOptions)
{
   printf("DoubleClick %s\n", item_path.c_str());

   auto elem = fBrowsable.GetElement(item_path);
   if (!elem) return ""s;

   // TODO: one can send id of editor or canvas to be sure when sending back reply

   if (drawingOptions == "$$$image$$$") {
      auto img = elem->GetContent("image64");
      if (!img.empty())
         return "FIMG:"s + img;
      else
         return ""s;
   }

   if (drawingOptions == "$$$editor$$$") {
      auto code = elem->GetContent("text");
      if (!code.empty())
         return "FREAD:"s + code;
      else
         return ""s;
   }

   auto canv = GetActiveCanvas();
   if (canv) {

      auto obj = elem->GetObject();

      if (obj)
         if (ROOT::Experimental::RDrawableProvider::DrawV6(canv, obj, drawingOptions)) {
            canv->ForceUpdate(); // force update async - do not wait for confirmation
            return "SLCTCANV:"s + canv->GetName();
         }
   }

   auto rcanv = GetActiveRCanvas();
   if (rcanv) {

      std::shared_ptr<RPadBase> subpad = rcanv;

      auto obj = elem->GetObject();
      if (obj)
         if (ROOT::Experimental::RDrawableProvider::DrawV7(subpad, obj, drawingOptions)) {
            rcanv->Modified();
            rcanv->Update(true);
            return "SLCTCANV:"s + rcanv->GetTitle();
         }

/*
      if (rcanv->NumPrimitives() > 0) {
         rcanv->Wipe();
         rcanv->Modified();
         rcanv->Update(true);
      }

      // FIXME: how to proceed with TObject ownership here
      TObject *clone = tobj->Clone();
      TH1 *h1 = dynamic_cast<TH1 *>(clone);
      if (h1) h1->SetDirectory(nullptr);

      std::shared_ptr<TObject> ptr;
      ptr.reset(clone);
      rcanv->Draw<RObjectDrawable>(ptr, drawingOptions);
      rcanv->Modified();

      rcanv->Update(true);

      return "SLCTCANV:"s + rcanv->GetTitle();
      */
   }


   printf("No active canvas to process dbl click\n");


   return "";
}

/////////////////////////////////////////////////////////////////////////////////
/// Show or update RBrowser in web window
/// If web window already started - just refresh it like "reload" button does
/// If no web window exists or \param always_start_new_browser configured, starts new window

void ROOT::Experimental::RBrowser::Show(const RWebDisplayArgs &args, bool always_start_new_browser)
{
   auto number = fWebWindow->NumConnections();

   if ((number == 0) || always_start_new_browser) {
      fWebWindow->Show(args);
   } else {
      for (int n=0;n<number;++n)
         WebWindowCallback(fWebWindow->GetConnectionId(n),"RELOAD");
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Hide ROOT Browser

void ROOT::Experimental::RBrowser::Hide()
{
   if (!fWebWindow)
      return;

   fWebWindow->CloseConnections();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Create new web canvas, invoked when new canvas created on client side

TCanvas *ROOT::Experimental::RBrowser::AddCanvas()
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
   fActiveCanvas = canv->GetName();

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

std::shared_ptr<ROOT::Experimental::RCanvas> ROOT::Experimental::RBrowser::AddRCanvas()
{
   std::string name = "rcanv"s + std::to_string(fRCanvases.size()+1);

   auto canv = ROOT::Experimental::RCanvas::Create(name);

   canv->Show("embed");

   fActiveCanvas = name;

   fRCanvases.emplace_back(canv);

   return canv;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns relative URL for canvas - required for client to establish connection

std::string ROOT::Experimental::RBrowser::GetCanvasUrl(TCanvas *canv)
{
   TWebCanvas *web = dynamic_cast<TWebCanvas *>(canv->GetCanvasImp());
   return fWebWindow->GetRelativeAddr(web->GetWebWindow());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns relative URL for canvas - required for client to establish connection

std::string ROOT::Experimental::RBrowser::GetRCanvasUrl(std::shared_ptr<RCanvas> &canv)
{
   return "../"s + canv->GetWindowAddr() + "/"s;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns active web canvas (if any)

TCanvas *ROOT::Experimental::RBrowser::GetActiveCanvas() const
{
   auto iter = std::find_if(fCanvases.begin(), fCanvases.end(), [this](const std::unique_ptr<TCanvas> &canv) { return fActiveCanvas == canv->GetName(); });

   if (iter != fCanvases.end())
      return iter->get();

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Returns active RCanvas (if any)

std::shared_ptr<ROOT::Experimental::RCanvas> ROOT::Experimental::RBrowser::GetActiveRCanvas() const
{
   auto iter = std::find_if(fRCanvases.begin(), fRCanvases.end(), [this](const std::shared_ptr<RCanvas> &canv) { return fActiveCanvas == canv->GetTitle(); });

   if (iter != fRCanvases.end())
      return *iter;

   return nullptr;

}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Close and delete specified canvas

void ROOT::Experimental::RBrowser::CloseCanvas(const std::string &name)
{
   auto iter = std::find_if(fCanvases.begin(), fCanvases.end(), [name](std::unique_ptr<TCanvas> &canv) { return name == canv->GetName(); });

   if (iter != fCanvases.end())
      fCanvases.erase(iter);

   if (fActiveCanvas == name)
      fActiveCanvas.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process client connect

void ROOT::Experimental::RBrowser::SendInitMsg(unsigned connid)
{
   std::vector<std::vector<std::string>> reply;

   reply.emplace_back(fBrowsable.GetWorkingPath()); // first element is current path

   for (auto &canv : fCanvases) {
      auto url = GetCanvasUrl(canv.get());
      std::string name = canv->GetName();
      std::vector<std::string> arr = {"root6", url, name};
      reply.emplace_back(arr);
   }

   for (auto &canv : fRCanvases) {
      auto url = GetRCanvasUrl(canv);
      std::string name = canv->GetTitle();
      std::vector<std::string> arr = {"root7", url, name};
      reply.emplace_back(arr);
   }

   std::string msg = "INMSG:";
   msg.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());

   printf("Init msg %s\n", msg.c_str());

   fWebWindow->Send(connid, msg);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return the current directory of ROOT

std::string ROOT::Experimental::RBrowser::GetCurrentWorkingDirectory()
{
   return "WORKPATH:"s + TBufferJSON::ToJSON(&fBrowsable.GetWorkingPath()).Data();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// receive data from client

void ROOT::Experimental::RBrowser::WebWindowCallback(unsigned connid, const std::string &arg)
{
   size_t len = arg.find("\n");
   if (len != std::string::npos)
      printf("Recv %s\n", arg.substr(0, len).c_str());
   else
      printf("Recv %s\n", arg.c_str());

   if (arg == "QUIT_ROOT") {

      fWebWindow->TerminateROOT();

   } else if (arg.compare(0,6, "BRREQ:") == 0) {
      // central place for processing browser requests
      auto json = ProcessBrowserRequest(arg.substr(6));
      if (json.length() > 0) fWebWindow->Send(connid, json);
   } else if (arg.compare("NEWRCANVAS") == 0) {

      auto canv = AddRCanvas();
      auto url = GetRCanvasUrl(canv);

      std::vector<std::string> reply = {"root7"s, url, canv->GetTitle()};
      std::string res = "CANVS:";
      res.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());
      fWebWindow->Send(connid, res);
   } else if (arg.compare("NEWTCANVAS") == 0) {

      auto canv = AddCanvas();
      auto url = GetCanvasUrl(canv);

      std::vector<std::string> reply = {"root6"s, url, std::string(canv->GetName())};
      std::string res = "CANVS:";
      res.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());
      fWebWindow->Send(connid, res);
   } else if (arg.compare(0,7, "DBLCLK:") == 0) {

      std::string reply;

      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));
      if (arr && (arr->size() == 2))
         reply = ProcessDblClick(arr->at(0), arr->at(1));

      if (!reply.empty())
         fWebWindow->Send(connid, reply);

   } else if (arg.compare(0,9, "RUNMACRO:") == 0) {
      ProcessRunCommand(arg.substr(9));
   } else if (arg.compare(0,9, "SAVEFILE:") == 0) {
      ProcessSaveFile(arg.substr(9));
   } else if (arg.compare(0,14, "SELECT_CANVAS:") == 0) {
      fActiveCanvas = arg.substr(14);
      printf("Select %s\n", fActiveCanvas.c_str());
   } else if (arg.compare(0,13, "CLOSE_CANVAS:") == 0) {
      CloseCanvas(arg.substr(13));
   } else if (arg == "GETWORKPATH") {
      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (arg.compare(0, 7, "CHPATH:") == 0) {
      printf("Current path %s\n", arg.substr(7).c_str());
      auto path = TBufferJSON::FromJSON<RElementPath_t>(arg.substr(7));

      if (path) fBrowsable.SetWorkingPath(*path);

      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (arg.compare(0, 6, "CHDIR:") == 0) {

      printf("CHDIR %s\n", arg.substr(6).c_str());

      fBrowsable.SetWorkingDirectory(arg.substr(6));

      fWebWindow->Send(connid, GetCurrentWorkingDirectory());
   } else if (arg.compare(0, 4, "CMD:") == 0) {
      std::string sPrompt = "root []";
      std::ostringstream pathtmp;
      pathtmp << gSystem->TempDirectory() << "/command." << gSystem->GetPid() << ".log";
      TApplication *app = gROOT->GetApplication();
      if (app->InheritsFrom("TRint")) {
         sPrompt = ((TRint*)gROOT->GetApplication())->GetPrompt();
         Gl_histadd((char *)arg.substr(4).c_str());
      }

      std::ofstream ofs(pathtmp.str(), std::ofstream::out | std::ofstream::app);
      ofs << sPrompt << arg.substr(4);
      ofs.close();

      gSystem->RedirectOutput(pathtmp.str().c_str(), "a");
      gROOT->ProcessLine(arg.substr(4).c_str());
      gSystem->RedirectOutput(0);
   } else if (arg.compare(0, 9, "ROOTHIST:") == 0) {
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
   } else if (arg.compare(0, 5, "LOGS:") == 0) {
      std::ostringstream pathtmp;
      pathtmp << gSystem->TempDirectory() << "/command." << gSystem->GetPid() << ".log";
      TString result;
      std::ifstream instr(pathtmp.str().c_str());
      result.ReadFile(instr);
      fWebWindow->Send(connid, "LOGS:"s + result.Data());
   } else if (arg.compare(0, 7, "SAVEAS:") == 0) {

      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));

      if (!arr || (arr->size() != 2)) {
         printf("SAVEAS failure - wrong arguments %s, should be array with two strings\n", arg.substr(7).c_str());
      } else {
         printf("Start SAVEAS dialog %s %s\n", arr->at(0).c_str(), arr->at(1).c_str());
         fFileDialog = std::make_unique<RFileDialog>(RFileDialog::kSaveAsFile);
         fFileDialog->SetFileName(arr->at(0));
         fFileDialog->Show({fWebWindow, std::stoi(arr->at(1))});
      }
   } else if (arg == "CLOSESAVEAS") {
      fFileDialog.reset();
   } else if (arg.compare(0, 7, "DOSAVE:") == 0) {
      auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));
      if (!arr || (arr->size() < 2)) {
         printf("DOSAVE failure - wrong arguments %s, should be at least two items\n", arg.substr(7).c_str());
      } else {
         printf("Calling dosave nargs %d\n", (int) arr->size());
         for (auto str : *arr)
            printf("    %s\n", str.c_str());
      }
   }
}


// ============================================================================================

using namespace ROOT::Experimental;

class RV6DrawProvider : public RDrawableProvider {
public:

   RV6DrawProvider()
   {
      RegisterV6(nullptr, [](TVirtualPad *pad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt) -> bool {

         // try take object without ownership
         auto tobj = obj->get_object<TObject>();
         if (!tobj) {
            auto utobj = obj->get_unique<TObject>();
            if (!utobj)
               return false;
            tobj = utobj.release();
            tobj->SetBit(TObject::kMustCleanup); // TCanvas should care about cleanup
         }

         pad->GetListOfPrimitives()->Clear();

         pad->GetListOfPrimitives()->Add(tobj, opt.c_str());

         return true;
      });

      RegisterV6(TLeaf::Class(), [](TVirtualPad *pad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt) -> bool {

         // try take object without ownership
         auto tleaf = obj->get_object<TLeaf>();
         if (!tleaf)
            return false;

         auto ttree = tleaf->GetBranch()->GetTree();
         if (!ttree)
            return false;


         std::string expr = std::string(tleaf->GetName()) + ">>htemp_tree_draw";

         ttree->Draw(expr.c_str(),"","goff");

         if (!gDirectory)
            return false;

         auto htemp = dynamic_cast<TH1*>(gDirectory->FindObject("htemp_tree_draw"));

         if (!htemp)
            return false;

         htemp->SetDirectory(nullptr);
         htemp->SetName(tleaf->GetName());

         pad->GetListOfPrimitives()->Clear();

         pad->GetListOfPrimitives()->Add(htemp, opt.c_str());

         return true;
      });

   }

} newRV6DrawProvider;


class RV7DrawProvider : public RDrawableProvider {
public:
   RV7DrawProvider()
   {
      RegisterV7(nullptr, [] (std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt) -> bool {

         // here clear ownership is required
         // If it possible, TObject will be cloned by RTObjectHolder
         auto tobj = obj->get_shared<TObject>();
         if (!tobj) return false;

         if (subpad->NumPrimitives() > 0) {
            subpad->Wipe();
            subpad->GetCanvas()->Modified();
            subpad->GetCanvas()->Update(true);
         }

         subpad->Draw<RObjectDrawable>(tobj, opt);
         return true;
      });


      RegisterV7(TLeaf::Class(), [](std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt) -> bool {

         // try take object without ownership
         auto tleaf = obj->get_object<TLeaf>();
         if (!tleaf)
            return false;

         auto ttree = tleaf->GetBranch()->GetTree();
         if (!ttree)
            return false;


         std::string expr = std::string(tleaf->GetName()) + ">>htemp_tree_draw";

         ttree->Draw(expr.c_str(),"","goff");

         if (!gDirectory)
            return false;

         auto htemp = dynamic_cast<TH1*>(gDirectory->FindObject("htemp_tree_draw"));

         if (!htemp)
            return false;

         htemp->SetDirectory(nullptr);
         htemp->SetName(tleaf->GetName());

         if (subpad->NumPrimitives() > 0) {
            subpad->Wipe();
            subpad->GetCanvas()->Modified();
            subpad->GetCanvas()->Update(true);
         }

         std::shared_ptr<TH1> shared;
         shared.reset(htemp);

         subpad->Draw<RObjectDrawable>(shared, opt);


         return true;
      });

   }

} newRV7DrawProvider;

