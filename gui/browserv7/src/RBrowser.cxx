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

#include <ROOT/RWebWindowsManager.hxx>
#include <ROOT/RBrowserItem.hxx>
#include <ROOT/TLogger.hxx>
#include "ROOT/TDirectory.hxx"
#include "ROOT/RMakeUnique.hxx"

#include "TString.h"
#include "TROOT.h"
#include "TBufferJSON.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>

/////////////////////////////////////////////////////////////////////
/// Item representing file in RBrowser

ROOT::Experimental::RFileItem::RFileItem(int _id, const std::string &_name, FileStat_t &stat) : RBaseItem(_id, _name)
{
   char tmp[256];
   Long64_t _fsize, bsize;

   type     = stat.fMode;
   size     = stat.fSize;
   uid      = stat.fUid;
   gid      = stat.fGid;
   modtime  = stat.fMtime;
   islink   = stat.fIsLink;
   isdir    = R_ISDIR(type);

   // file size
   _fsize = bsize = size;
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
   fsize = tmp;

   // modification time
   time_t loctime = (time_t) modtime;
   struct tm *newtime = localtime(&loctime);
   if (newtime) {
      snprintf(tmp, sizeof(tmp), "%d-%02d-%02d %02d:%02d", newtime->tm_year + 1900,
               newtime->tm_mon+1, newtime->tm_mday, newtime->tm_hour,
               newtime->tm_min);
      mtime = tmp;
   } else {
      mtime = "1901-01-01 00:00";
   }

   // file type
   snprintf(tmp, sizeof(tmp), "%c%c%c%c%c%c%c%c%c%c",
            (islink ?
             'l' :
             R_ISREG(type) ?
             '-' :
             (R_ISDIR(type) ?
              'd' :
              (R_ISCHR(type) ?
               'c' :
               (R_ISBLK(type) ?
                'b' :
                (R_ISFIFO(type) ?
                 'p' :
                 (R_ISSOCK(type) ?
                  's' : '?' )))))),
            ((type & kS_IRUSR) ? 'r' : '-'),
            ((type & kS_IWUSR) ? 'w' : '-'),
            ((type & kS_ISUID) ? 's' : ((type & kS_IXUSR) ? 'x' : '-')),
            ((type & kS_IRGRP) ? 'r' : '-'),
            ((type & kS_IWGRP) ? 'w' : '-'),
            ((type & kS_ISGID) ? 's' : ((type & kS_IXGRP) ? 'x' : '-')),
            ((type & kS_IROTH) ? 'r' : '-'),
            ((type & kS_IWOTH) ? 'w' : '-'),
            ((type & kS_ISVTX) ? 't' : ((type & kS_IXOTH) ? 'x' : '-')));
   ftype = tmp;

   struct UserGroup_t *user_group = gSystem->GetUserInfo(uid);
   if (user_group) {
      fuid = user_group->fUser;
      fgid = user_group->fGroup;
      delete user_group;
   } else {
      fuid = std::to_string(uid);
      fgid = std::to_string(gid);
   }
}

/////////////////////////////////////////////////////////////////////
/// Add folder

void ROOT::Experimental::RBrowserFSDescription::AddFolder(const char *name)
{
   FileStat_t sbuf;
   int cnt = 0;

   if (gSystem->GetPathInfo(name, sbuf)) {
      if (sbuf.fIsLink) {
         std::cout << "AddFile : Broken symlink of " << name << std::endl;
      } else {
         std::cerr << "Can't read file attributes of \"" <<  name << "\": " << gSystem->GetError() << std::endl;;
      }
      return;
   }

   if (R_ISDIR(sbuf.fMode))
      fDesc.emplace_back(cnt++, name, sbuf);
}

/////////////////////////////////////////////////////////////////////
/// Add file

void ROOT::Experimental::RBrowserFSDescription::AddFile(const char *name)
{
   FileStat_t sbuf;
   int cnt = 0;

   if (gSystem->GetPathInfo(name, sbuf)) {
      if (sbuf.fIsLink) {
         std::cout << "AddFile : Broken symlink of " << name << std::endl;
      } else {
         std::cerr << "Can't read file attributes of \"" <<  name << "\": " << gSystem->GetError() << std::endl;;
      }
      return;
   }

   if (!R_ISDIR(sbuf.fMode))
      fDesc.emplace_back(cnt++, name, sbuf);
}

/////////////////////////////////////////////////////////////////////
/// Collect information for provided directory

void ROOT::Experimental::RBrowserFSDescription::Build(const std::string &path)
{
   void *dirp;
   const char *name;
   std::string spath = path;
   spath.insert(0, ".");
   fDesc.clear();

   std::string savdir = gSystem->WorkingDirectory();
   if (!gSystem->ChangeDirectory(spath.c_str())) return;

   if ((dirp = gSystem->OpenDirectory(".")) == nullptr) {
      gSystem->FreeDirectory(dirp); // probably, not needed
      gSystem->ChangeDirectory(savdir.c_str());
      return;
   }
   while ((name = gSystem->GetDirEntry(dirp)) != nullptr) {
      if (strncmp(name, ".", 1) && strncmp(name, "..", 2))
         AddFolder(name);
      gSystem->ProcessEvents();
   }
   gSystem->FreeDirectory(dirp);

   if ((dirp = gSystem->OpenDirectory(".")) == nullptr) {
      gSystem->FreeDirectory(dirp); // probably, not needed
      gSystem->ChangeDirectory(savdir.c_str());
      return;
   }
   while ((name = gSystem->GetDirEntry(dirp)) != nullptr) {
      if (strncmp(name, ".", 1) && strncmp(name, "..", 2))
         AddFile(name);
      gSystem->ProcessEvents();
   }
   gSystem->FreeDirectory(dirp);
   gSystem->ChangeDirectory(savdir.c_str());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

std::string ROOT::Experimental::RBrowserFSDescription::ProcessBrowserRequest(const std::string &msg)
{
   std::string res;

   auto request = TBufferJSON::FromJSON<RRootBrowserRequest>(msg);

   if (msg.empty()) {
      request = std::make_unique<RRootBrowserRequest>();
      request->path = "/";
      request->first = 0;
      request->number = 100;
   } else if (!request) {
      return res;
   }

   if (1) { //((request->path.compare("/") == 0) && (request->first == 0)) {
      Build(request->path);
      RRootBrowserReply reply;
      reply.path = request->path;
      reply.first = request->first;
      reply.nchilds = fDesc.size();

      for (auto &node : fDesc) {
         reply.nodes.emplace_back(node.name, node.fsize, node.mtime, node.ftype, node.fuid, node.fgid, node.isdir);
      }
      res = "BREPL:";
      res.append(TBufferJSON::ToJSON(&reply, 103).Data());
   }

   return res;
}

/** \class ROOT::Experimental::RBrowser
\ingroup webdisplay

web-based ROOT Browser prototype.
*/

//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

ROOT::Experimental::RBrowser::RBrowser()
{
   fWebWindow = ROOT::Experimental::RWebWindowsManager::Instance()->CreateWindow();
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

   } else if (arg == "GETDRAW") {

   } else if (arg == "QUIT_ROOT") {

      RWebWindowsManager::Instance()->Terminate();

   } else if ((arg.compare(0, 7, "SETVI0:") == 0) || (arg.compare(0, 7, "SETVI1:") == 0)) {
      // change visibility for specified nodeid

   } else if (arg.compare(0,6, "BRREQ:") == 0) {
      // central place for processing browser requests
      //if (!fDesc.IsBuild()) fDesc.Build();
      auto json = fDesc.ProcessBrowserRequest(arg.substr(6));
      if (json.length() > 0) fWebWindow->Send(connid, json);
   }
}
