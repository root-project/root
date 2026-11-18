// @(#)root/eve7:$Id$
// Author: Matevz Tadel, 2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveAssetDir
#define ROOT7_REveAssetDir

#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace Internal {

/// Directories of one kind of file the REve server serves under one http
/// location prefix. Private to REve; see the class documentation in the .cxx.
class REveAssetDir {
   std::string fPrefix;                    ///< http location prefix, such as "sdf-fonts/"
   std::vector<std::string> fUserDirs;     ///< set by the user, searched first, in order
   std::vector<std::string> fFallbackDirs; ///< shipped with ROOT, read-only, searched after fUserDirs
   std::string fWriteDir;                  ///< where new files are written; empty if none

   static std::vector<REveAssetDir *> &Instances();

public:
   explicit REveAssetDir(const std::string &prefix);
   ~REveAssetDir();

   REveAssetDir(const REveAssetDir &) = delete;
   REveAssetDir &operator=(const REveAssetDir &) = delete;

   static std::string NormalizeDir(const std::string &dir);
   static std::string GetUi5Dir();
   static std::string GetAssetsDir();

   /// @{ Accessors; the directories passed in are normalized.
   const std::string &GetPrefix() const { return fPrefix; }
   const std::vector<std::string> &GetUserDirs() const { return fUserDirs; }
   const std::string &GetWriteDir() const { return fWriteDir; }
   void SetWriteDir(const std::string &dir) { fWriteDir = NormalizeDir(dir); }
   /// @}

   void SetUserDir(const std::string &dir);
   void AddUserDir(const std::string &dir);
   void AddFallbackDir(const std::string &dir);

   std::string Find(const std::string &file) const;
   std::string FindPair(const std::string &file1, const std::string &file2) const;
   std::string WritePath(const std::string &file) const;

   void Register() const;
   static void RegisterAll();
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
