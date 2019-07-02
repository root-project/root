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
#include "ROOT/RMakeUnique.hxx"

#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TBufferJSON.h"

#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>


/** \class ROOT::Experimental::RBrowser
\ingroup webdisplay

web-based ROOT Browser prototype.
*/

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

ROOT::Experimental::RBrowser::RBrowser()
{
   fWebWindow = RWebWindow::Create();
   fWebWindow->SetDefaultPage("file:rootui5sys/browser/browser.html");

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { this->WebWindowCallback(connid, arg); });
   fWebWindow->SetGeometry(1200, 700); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue
   Show();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

ROOT::Experimental::RBrowser::~RBrowser()
{
}

/////////////////////////////////////////////////////////////////////
/// Collect information for provided directory

void ROOT::Experimental::RBrowser::Build(const std::string &path)
{
   fDescPath = path;

   void *dirp;
   const char *name;
   std::string spath = path;
   spath.insert(0, ".");
   fDesc.clear();
   fSorted.clear();

   std::string savdir = gSystem->WorkingDirectory();
   if (!gSystem->ChangeDirectory(spath.c_str())) return;

   if ((dirp = gSystem->OpenDirectory(".")) != nullptr) {
      while ((name = gSystem->GetDirEntry(dirp)) != nullptr) {
         if ((strncmp(name, ".", 1)==0) || (strncmp(name, "..", 2)==0)) continue;

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

         fDesc.emplace_back(name, nchilds);

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

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

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

   // rebuild list only when selected directory changed
   if (!IsBuild() || (request->path != fDescPath)) {
      fDescPath = request->path;
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
/// receive data from client

void ROOT::Experimental::RBrowser::WebWindowCallback(unsigned connid, const std::string &arg)
{
   printf("Recv %s\n", arg.c_str());

   if (arg == "CONN_READY") {

      fConnId = connid;

   } else if (arg == "QUIT_ROOT") {

      fWebWindow->TerminateROOT();

   } else if (arg.compare(0,6, "BRREQ:") == 0) {
      // central place for processing browser requests
      //if (!fDesc.IsBuild()) fDesc.Build();
      auto json = ProcessBrowserRequest(arg.substr(6));
      if (json.length() > 0) fWebWindow->Send(connid, json);
   }
}
