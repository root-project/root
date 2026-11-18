// @(#)root/eve7:$Id$
// Author: Matevz Tadel, 2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "REveAssetDir.hxx"

#include <ROOT/REveManager.hxx>
#include <ROOT/RWebWindow.hxx>

#include "THttpServer.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"

#include <algorithm>

using namespace ROOT::Experimental::Internal;
namespace REX = ROOT::Experimental;

/** \class ROOT::Experimental::Internal::REveAssetDir
Directories of one kind of file the REve server serves under one http location
prefix, such as the SDF fonts under "sdf-fonts/". Not public API: REveText and
REveLogo each hold one instance and expose its setters.

A file is looked up in the user directories, in the order they were set and
added, then in the fallback directories, which hold files shipped with ROOT and
may be read-only. The shipped files of a kind are in its subdirectory of
GetAssetsDir(). The server serves a requested file from the first directory
that has it. An optional write directory is where new files are written; it is
created only when something is written.

Register() is the only place that passes the directories to the server. The
setters call it, and the REveManager constructor calls RegisterAll(), so the
directories can be set before REveManager::Create().
*/

////////////////////////////////////////////////////////////////////////////////
/// All existing instances, for RegisterAll().

std::vector<REveAssetDir *> &REveAssetDir::Instances()
{
   static std::vector<REveAssetDir *> s_instances;
   return s_instances;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor. `prefix` is the http location prefix, with a trailing slash.

REveAssetDir::REveAssetDir(const std::string &prefix) : fPrefix(prefix)
{
   Instances().push_back(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveAssetDir::~REveAssetDir()
{
   auto &v = Instances();
   v.erase(std::remove(v.begin(), v.end(), this), v.end());
}

////////////////////////////////////////////////////////////////////////////////
/// Return `dir` with a trailing slash; an empty `dir` stays empty.

std::string REveAssetDir::NormalizeDir(const std::string &dir)
{
   if (dir.empty() || dir.back() == '/')
      return dir;
   return dir + '/';
}

////////////////////////////////////////////////////////////////////////////////
/// Return the directory the server maps to "rootui5sys/", found as
/// RWebWindowsManager finds it: $ROOTUI5SYS, else WebGui.RootUi5Path, else the
/// ui5 directory in TROOT::GetDataDir().

std::string REveAssetDir::GetUi5Dir()
{
   TString ui5dir = gSystem->Getenv("ROOTUI5SYS");
   if (ui5dir.Length() == 0)
      ui5dir = gEnv->GetValue("WebGui.RootUi5Path", "");
   if (ui5dir.Length() == 0)
      ui5dir.Form("%s/ui5", TROOT::GetDataDir().Data());
   gSystem->ExpandPathName(ui5dir);
   return ui5dir.Data();
}

////////////////////////////////////////////////////////////////////////////////
/// Return eve7/assets/ in the ui5 directory, where the build installs the
/// RenderCore assets tarball. It has one subdirectory per kind of asset, the
/// fallback directory of that kind.

std::string REveAssetDir::GetAssetsDir()
{
   return GetUi5Dir() + "/eve7/assets/";
}

////////////////////////////////////////////////////////////////////////////////
/// Replace the user directories with `dir`.

void REveAssetDir::SetUserDir(const std::string &dir)
{
   fUserDirs = {NormalizeDir(dir)};
   Register();
}

////////////////////////////////////////////////////////////////////////////////
/// Append `dir` to the user directories.

void REveAssetDir::AddUserDir(const std::string &dir)
{
   fUserDirs.push_back(NormalizeDir(dir));
   Register();
}

////////////////////////////////////////////////////////////////////////////////
/// Append `dir` to the fallback directories.

void REveAssetDir::AddFallbackDir(const std::string &dir)
{
   fFallbackDirs.push_back(NormalizeDir(dir));
   Register();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the path of `file` in the first directory that has it, user
/// directories first, or an empty string if none has it.

std::string REveAssetDir::Find(const std::string &file) const
{
   for (auto *dirs : {&fUserDirs, &fFallbackDirs}) {
      for (auto &dir : *dirs) {
         std::string path = dir + file;
         if (!gSystem->AccessPathName(path.c_str()))
            return path;
      }
   }
   return {};
}

////////////////////////////////////////////////////////////////////////////////
/// Return the first directory that has `file1` or `file2` if it has both, else
/// an empty string. The server then serves both files from that directory, as
/// it serves each file from the first directory that has it.

std::string REveAssetDir::FindPair(const std::string &file1, const std::string &file2) const
{
   for (auto *dirs : {&fUserDirs, &fFallbackDirs}) {
      for (auto &dir : *dirs) {
         bool has1 = !gSystem->AccessPathName((dir + file1).c_str());
         bool has2 = !gSystem->AccessPathName((dir + file2).c_str());
         if (has1 || has2)
            return has1 && has2 ? dir : std::string();
      }
   }
   return {};
}

////////////////////////////////////////////////////////////////////////////////
/// Return the path to write `file` to, creating the write directory if it does
/// not exist. Return an empty string if there is no write directory or it cannot
/// be created.

std::string REveAssetDir::WritePath(const std::string &file) const
{
   if (fWriteDir.empty())
      return {};
   if (gSystem->AccessPathName(fWriteDir.c_str()) && gSystem->mkdir(fWriteDir.c_str(), true))
      return {};
   return fWriteDir + file;
}

////////////////////////////////////////////////////////////////////////////////
/// Pass the directories to the server, user directories first, then fallback
/// directories, each once. Does nothing before REveManager exists or when there
/// are no directories.

void REveAssetDir::Register() const
{
   if (!REX::gEve)
      return;
   auto serv = REX::gEve->GetWebWindow()->GetServer();
   if (!serv)
      return;

   std::vector<std::string> dirs;
   for (auto *src : {&fUserDirs, &fFallbackDirs}) {
      for (auto &dir : *src) {
         if (std::find(dirs.begin(), dirs.end(), dir) == dirs.end())
            dirs.push_back(dir);
      }
   }
   if (dirs.empty())
      return;

   serv->AddLocation(fPrefix.c_str(), dirs.front().c_str());
   for (size_t i = 1; i < dirs.size(); ++i)
      serv->AppendLocation(fPrefix.c_str(), dirs[i].c_str());
}

////////////////////////////////////////////////////////////////////////////////
/// Register all instances. Called from the REveManager constructor.

void REveAssetDir::RegisterAll()
{
   for (auto *ad : Instances())
      ad->Register();
}
