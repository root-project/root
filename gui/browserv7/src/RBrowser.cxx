/// \file ROOT/RBrowser.cxx
/// \ingroup WebGui ROOT7
/// \author Bertrand Bellenot <bertrand.bellenot@cern.ch>
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

#include <ROOT/RBrowserItem.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RObjectDrawable.hxx>

#include "TKey.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TWebCanvas.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TBufferJSON.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <fstream>

using namespace std::string_literals;

/** \class ROOT::Experimental::RBrowser
\ingroup webdisplay

web-based ROOT Browser prototype.
*/

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

ROOT::Experimental::RBrowser::RBrowser(bool use_rcanvas)
{
   SetUseRCanvas(use_rcanvas);

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

   // AddRCanvas();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

ROOT::Experimental::RBrowser::~RBrowser()
{
   fCanvases.clear();
}

TFile *ROOT::Experimental::RBrowser::OpenFile(const std::string &fname)
{
   auto file = dynamic_cast<TFile *>(gROOT->GetListOfFiles()->FindObject(fname.c_str()));

   if (!file)
      file = TFile::Open(fname.c_str());

   return file;
}

/////////////////////////////////////////////////////////////////////
/// Collect information for provided root file

void ROOT::Experimental::RBrowser::Browse(const std::string &path)
{
   fDescPath = path;
   fDesc.clear();
   fSorted.clear();
   std::string keyname, classname, filename = path.substr(1, path.size()-2);
   auto rfile = OpenFile(filename);
   if (rfile) {
      // replace actual user data (TObjString) by the TDirectory...
      int nkeys = rfile->GetListOfKeys()->GetEntries();
      for (int i=0; i<nkeys; ++i) {
         TKey *key = (TKey *)rfile->GetListOfKeys()->At(i);
         keyname = key->GetName();
         if (keyname.back() == '.')
            keyname.pop_back();
         keyname += ";";
         keyname += std::to_string(key->GetCycle());
         classname = key->GetClassName();
         if (classname == "TTree" || classname == "TNtuple" ||
             classname == "TDirectory" || classname == "TDirectoryFile")
            fDesc.emplace_back(keyname, 0);// 1);
         else
            fDesc.emplace_back(keyname, 0);
         auto &item   = fDesc.back();
         item.type    = 0;
         item.size    = 0;
         item.uid     = 0;
         item.gid     = 0;
         item.modtime = 0;
         item.islink  = 0;
         item.isdir   = 0;
         item.icon    = GetClassIcon(classname);
         item.fsize   = "";
         item.mtime   = "";
         item.ftype   = "";
         item.fuid    = "";
         item.fgid    = "";
         item.className = classname;
      }
   }
   for (auto &item : fDesc)
      fSorted.emplace_back(&item);
}

/////////////////////////////////////////////////////////////////////
/// Collect information for provided directory

void ROOT::Experimental::RBrowser::Build(const std::string &path)
{
   fDescPath = path;

   void *dirp;
   TString name;
   std::string spath = path;
   spath.insert(0, ".");
   fDesc.clear();
   fSorted.clear();

   std::string savdir = gSystem->WorkingDirectory();
   if (!gSystem->ChangeDirectory(spath.c_str())) return;

   if ((dirp = gSystem->OpenDirectory(".")) != nullptr) {
      while ((name = gSystem->GetDirEntry(dirp)) != "") {
         if ((name == ".") || (name == "..")) continue;

         FileStat_t stat;

         if (gSystem->GetPathInfo(name, stat)) {
            if (stat.fIsLink) {
               std::cout << "AddFile : Broken symlink of " << name << std::endl;
            } else {
               std::cerr << "Can't read file attributes of \"" <<  name << "\": " << gSystem->GetError() << std::endl;;
            }
            continue;
         }

         int nchilds = R_ISDIR(stat.fMode) ? 1 : 0;
         if (name.EndsWith(".root"))
            nchilds = 1;

         fDesc.emplace_back(name.Data(), nchilds);

         auto &item = fDesc.back();

         // this is construction of current item

         char tmp[256];
         Long64_t _fsize, bsize;

         item.type     = stat.fMode;
         item.size     = stat.fSize;
         item.uid      = stat.fUid;
         item.gid      = stat.fGid;
         item.modtime  = stat.fMtime;
         item.islink   = stat.fIsLink;
         item.isdir    = R_ISDIR(stat.fMode);

         if (item.isdir)
            item.icon = "sap-icon://folder-blank";
         else
            item.icon = GetFileIcon(name);

         // file size
         _fsize = bsize = item.size;
         if (_fsize > 1024) {
            _fsize /= 1024;
            if (_fsize > 1024) {
               // 3.7MB is more informative than just 3MB
               snprintf(tmp, sizeof(tmp), "%lld.%lldM", _fsize/1024, (_fsize%1024)/103);
            } else {
               snprintf(tmp, sizeof(tmp), "%lld.%lldK", bsize/1024, (bsize%1024)/103);
            }
         } else {
            snprintf(tmp, sizeof(tmp), "%lld", bsize);
         }
         item.fsize = tmp;

         // modification time
         time_t loctime = (time_t) item.modtime;
         struct tm *newtime = localtime(&loctime);
         if (newtime) {
            snprintf(tmp, sizeof(tmp), "%d-%02d-%02d %02d:%02d", newtime->tm_year + 1900,
                     newtime->tm_mon+1, newtime->tm_mday, newtime->tm_hour,
                     newtime->tm_min);
            item.mtime = tmp;
         } else {
            item.mtime = "1901-01-01 00:00";
         }

         // file type
         snprintf(tmp, sizeof(tmp), "%c%c%c%c%c%c%c%c%c%c",
                  (item.islink ?
                   'l' :
                   R_ISREG(item.type) ?
                   '-' :
                   (R_ISDIR(item.type) ?
                    'd' :
                    (R_ISCHR(item.type) ?
                     'c' :
                     (R_ISBLK(item.type) ?
                      'b' :
                      (R_ISFIFO(item.type) ?
                       'p' :
                       (R_ISSOCK(item.type) ?
                        's' : '?' )))))),
                  ((item.type & kS_IRUSR) ? 'r' : '-'),
                  ((item.type & kS_IWUSR) ? 'w' : '-'),
                  ((item.type & kS_ISUID) ? 's' : ((item.type & kS_IXUSR) ? 'x' : '-')),
                  ((item.type & kS_IRGRP) ? 'r' : '-'),
                  ((item.type & kS_IWGRP) ? 'w' : '-'),
                  ((item.type & kS_ISGID) ? 's' : ((item.type & kS_IXGRP) ? 'x' : '-')),
                  ((item.type & kS_IROTH) ? 'r' : '-'),
                  ((item.type & kS_IWOTH) ? 'w' : '-'),
                  ((item.type & kS_ISVTX) ? 't' : ((item.type & kS_IXOTH) ? 'x' : '-')));
         item.ftype = tmp;

         struct UserGroup_t *user_group = gSystem->GetUserInfo(item.uid);
         if (user_group) {
            item.fuid = user_group->fUser;
            item.fgid = user_group->fGroup;
            delete user_group;
         } else {
            item.fuid = std::to_string(item.uid);
            item.fgid = std::to_string(item.gid);
         }

         gSystem->ProcessEvents();
      }

      gSystem->FreeDirectory(dirp);
   }

   gSystem->ChangeDirectory(savdir.c_str());

   // now build sorted list - first folders, then files
   // later more complex sorting rules can be applied

   for (auto &item : fDesc)
      if (item.isdir)
         fSorted.emplace_back(&item);

   for (auto &item : fDesc)
      if (!item.isdir)
         fSorted.emplace_back(&item);
}

/////////////////////////////////////////////////////////////////////////////////
/// Get icon for the given class name

std::string ROOT::Experimental::RBrowser::GetClassIcon(std::string &classname)
{
   std::string res;
   if (classname == "TTree" || classname == "TNtuple")
      res = "sap-icon://tree";
   else if (classname == "TDirectory" || classname == "TDirectoryFile")
      res = "sap-icon://folder-blank";
   else
      res = "sap-icon://electronic-medical-record";
   return res;
}

/////////////////////////////////////////////////////////////////////////////////
/// Get icon for the type of given file name

std::string ROOT::Experimental::RBrowser::GetFileIcon(TString &name)
{
   std::string res;
   if ((name.EndsWith(".c")) ||
       (name.EndsWith(".cpp")) ||
       (name.EndsWith(".cxx")) ||
       (name.EndsWith(".c++")) ||
       (name.EndsWith(".cxx")) ||
       (name.EndsWith(".h")) ||
       (name.EndsWith(".hpp")) ||
       (name.EndsWith(".hxx")) ||
       (name.EndsWith(".h++")) ||
       (name.EndsWith(".py")) ||
       (name.EndsWith(".txt")) ||
       (name.EndsWith(".cmake")) ||
       (name.EndsWith(".dat")) ||
       (name.EndsWith(".log")) ||
       (name.EndsWith(".js")))
      res ="sap-icon://document-text";
   else if ((name.EndsWith(".bmp")) ||
            (name.EndsWith(".gif")) ||
            (name.EndsWith(".jpg")) ||
            (name.EndsWith(".png")) ||
            (name.EndsWith(".svg")))
      res = "sap-icon://picture";
   else if (name.EndsWith(".root"))
      res = "sap-icon://org-chart";
   else
      res = "sap-icon://document";
   return res;
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

   if (request->sort == "DBLCLK") {

   }

   // rebuild list only when selected directory changed
   if (!IsBuild() || (request->path != fDescPath)) {
      fDescPath = request->path;
      if (fDescPath.size() > 6 &&
           fDescPath.compare(fDescPath.size() - 6, 6, ".root/") == 0)
         Browse(request->path);
      else
         Build(request->path);
   }

   RBrowserReply reply;
   reply.path = request->path;
   reply.first = request->first;
   reply.nchilds = fDesc.size();

   // return only requested number of nodes
   // no items ownership, RRootBrowserReply must be always temporary object
   // TODO: implement different sorting
   int seq = 0;
   for (auto &node : fSorted) {
      if ((seq >= request->first) && ((seq < request->first + request->number) || (request->number == 0)))
         reply.nodes.emplace_back(node);
      seq++;
   }

   res = "BREPL:";
   res.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kSkipTypeInfo + TBufferJSON::kNoSpaces).Data());

   return res;
}

/////////////////////////////////////////////////////////////////////////////////
/// Process dbl click on browser item

std::string ROOT::Experimental::RBrowser::ProcessDblClick(const std::string &item_path, const std::string &drawingOptions) {
   if (item_path.find(".root") == std::string::npos) {
      std::string res = "FREAD:";
      std::ifstream t(item_path);
      res.append(std::string(std::istreambuf_iterator<char>(t), std::istreambuf_iterator<char>()));
      return res;
   }

   std::string rootFilePath = "", rootFileName = "";

   // Split of the path by /
   std::vector<std::string> split;
   std::string buffer;
   std::istringstream path(item_path);
   while (std::getline(path, buffer, '/')) {
      split.push_back(buffer);
   }

   // Iterate over the split
   // The goal is to have two parts
   // The first one is the relative path of the root file to open it (rootFilePath)
   // And the second if the name of the namecycle (rootFileName)
   for (std::vector<int>::size_type i = 0; i != split.size(); i++) {
      // If the current split contain .root
      if (split[i].find(".root") != std::string::npos) {
         rootFilePath += split[i]; // Add the file to the path
         if (split[i + 1].find("ntuple") != std::string::npos) {
            // TODO
            break;
         } else {
            rootFileName += split[i + 1]; // the add the name of the file then stop
            break;
         }
      } else {
         rootFilePath += split[i] + "/"; // Add the file to the path
      }
   }

   auto file = OpenFile(rootFilePath);
   if (!file) {
      printf("No ROOT file found\n");
      return "";
   }

   TObject *object = nullptr;
   file->GetObject(rootFileName.c_str(), object); // Getting the data of the graphic into the TObject

   if (!object) {
      printf("No ROOT object read\n");
      return "";
   }

   auto canv = GetActiveCanvas();
   if (canv) {
      canv->GetListOfPrimitives()->Clear();

      canv->GetListOfPrimitives()->Add(object, drawingOptions.c_str());

      canv->ForceUpdate(); // force update async - do not wait for confirmation

      return "SLCTCANV:"s + canv->GetName();
   }

   auto rcanv = GetActiveRCanvas();
   if (rcanv) {
      if (rcanv->NumPrimitives() > 0) {
         rcanv->Wipe();
         rcanv->Modified();
         rcanv->Update(true);
      }

      // FIXME: how to proced with ownership here
      TObject *clone = object->Clone();
      TH1 *h1 = dynamic_cast<TH1 *>(clone);
      if (h1) h1->SetDirectory(nullptr);

      std::shared_ptr<TObject> ptr;
      ptr.reset(clone);
      rcanv->Draw<RObjectDrawable>(ptr, drawingOptions);
      rcanv->Modified();

      rcanv->Update(true);

      return "SLCTCANV:"s + rcanv->GetTitle();
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
/// receive data from client

void ROOT::Experimental::RBrowser::WebWindowCallback(unsigned connid, const std::string &arg)
{
   printf("Recv %s\n", arg.c_str());

   if (arg == "QUIT_ROOT") {

      fWebWindow->TerminateROOT();

   } else if (arg.compare(0,6, "BRREQ:") == 0) {
      // central place for processing browser requests
      //if (!fDesc.IsBuild()) fDesc.Build();
      auto json = ProcessBrowserRequest(arg.substr(6));
      if (json.length() > 0) fWebWindow->Send(connid, json);
   } else if (arg.compare("NEWCANVAS") == 0) {

      std::vector<std::string> reply;

      if (GetUseRCanvas()) {
         auto canv = AddRCanvas();

         auto url = GetRCanvasUrl(canv);

         reply = {"root7"s, url, canv->GetTitle()};

      } else {
         // create canvas
         auto canv = AddCanvas();

         auto url = GetCanvasUrl(canv);

         reply = {"root6"s, url, std::string(canv->GetName())};
      }

      std::string res = "CANVS:";
      res.append(TBufferJSON::ToJSON(&reply, TBufferJSON::kNoSpaces).Data());

      fWebWindow->Send(connid, res);
   } else if (arg.compare(0,7, "DBLCLK:") == 0) {

      if (arg.at(8) != '[') {
         auto str = ProcessDblClick(arg.substr(7), "");
         fWebWindow->Send(connid, str);
      } else {
         auto arr = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));
         if (arr) {
            auto str = ProcessDblClick(arr->at(0), arr->at(1));
            if (str.length() > 0) {
               fWebWindow->Send(connid, str);
            }
         }
      }
   } else if (arg.compare(0,14, "SELECT_CANVAS:") == 0) {
      fActiveCanvas = arg.substr(14);
      printf("Select %s\n", fActiveCanvas.c_str());
   } else if (arg.compare(0,13, "CLOSE_CANVAS:") == 0) {
      CloseCanvas(arg.substr(13));
   }
}
