/// \file v7/src/RFile.cxx
/// \ingroup Base ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-03-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RFile.hxx"

#include <ROOT/RError.hxx>
#include <ROOT/StringUtils.hxx>
#include <Byteswap.h>
#include <TTree.h>
#include <TGraph2D.h>
#include <TH1.h>
#include <TKey.h>

#include <cstring>
#include <iostream>

using ROOT::Experimental::RFile;

static bool HasPrefix(std::string_view str, std::string_view prefix)
{
   if (prefix.empty())
      return true;

   if (str.size() < prefix.size())
      return false;

   return str.compare(0, prefix.size(), prefix) == 0;
}

static bool IsInternalKey(const char *className)
{
   static constexpr const char *fInternalKeyClassNames[] = {"TFile", "FreeSegments", "StreamerInfo", "KeysList"};

   if (strlen(className) == 0)
      return true;
   for (const char *k : fInternalKeyClassNames)
      if (strcmp(className, k) == 0)
         return true;
   return false;
}

static void ThrowIfExtensionIsNotRoot(std::string_view path)
{
   if (path.size() < 5 || path.compare(path.size() - 5, 5, ".root") != 0) {
      throw ROOT::RException(R__FAIL(std::string("Only .root files are supported.")));
   }
}

static ROOT::Experimental::RFileKeyInfo RFileKeyInfoFromTKey(const TKey &tkey)
{
   ROOT::Experimental::RFileKeyInfo keyInfo{};
   keyInfo.fName = tkey.GetName();
   keyInfo.fClassName = tkey.GetClassName();
   keyInfo.fTitle = tkey.GetTitle();
   return keyInfo;
}

namespace {

#pragma pack(push, 1)
struct RTFileKeyHeader {
   std::uint32_t fNBytes;
   std::uint16_t fVersion;
   std::uint32_t fObjLen;
   std::uint32_t fDatetime;
   std::uint16_t fKeyLen;
   std::uint16_t fCycle;
};

/// Contains stripped-down information about a TKey on disk
struct RTFileKey : RTFileKeyHeader {
   union {
      struct {
         std::uint32_t fSeekKey;
         std::uint32_t fSeekPdir;
      } fShort;
      struct {
         std::uint64_t fSeekKey;
         std::uint64_t fSeekPdir;
      } fLong;
   };

   char fObjName[256];
   std::uint8_t fObjNameLen;

   bool IsLong() const { return RByteSwap<sizeof(fVersion)>::bswap(fVersion) >= 1000; }
   std::uint64_t SeekPdir() const
   {
      if (IsLong()) {
         return RByteSwap<sizeof(std::uint64_t)>::bswap(fLong.fSeekPdir);
      } else {
         return RByteSwap<sizeof(std::uint32_t)>::bswap(fShort.fSeekPdir);
      }
   }
};
#pragma pack(pop)
static_assert(std::is_trivial_v<RTFileKey>);

} // namespace

static ROOT::RResult<RTFileKey> ReadKeyFromFile(TFile *file, std::uint64_t keyAddr)
{
   // This is quite ugly, as we need to move the file's internal cursor to retrieve the parent keys.
   // Luckily it is not a problem because our internal iterator saves the last visited address and is
   // therefore resilient to the file pointer being moved in between calls to operator++.
   file->Seek(keyAddr);

   // We know the key is at least this big. Start reading it.
   char keyHeadBuf[sizeof(RTFileKeyHeader)];
   bool failed = file->ReadBuffer(keyHeadBuf, sizeof(keyHeadBuf));
   if (failed) {
      return R__FAIL("Corrupted key data");
   }
   RTFileKey key{};
   memcpy(&key, keyHeadBuf, sizeof(RTFileKeyHeader));

   // Now read the seek key/pdir
   char buf[sizeof(RTFileKey{}.fLong)];
   std::size_t bufSize = key.IsLong() ? sizeof(RTFileKey{}.fLong) : sizeof(RTFileKey{}.fShort);
   failed = file->ReadBuffer(buf, bufSize);
   if (failed) {
      return R__FAIL("Corrupted key data");
   }
   if (key.IsLong()) {
      memcpy(reinterpret_cast<char *>(&key) + sizeof(RTFileKeyHeader), buf, bufSize);
   } else {
      memcpy(&key.fShort.fSeekKey, buf, sizeof(std::uint32_t));
      memcpy(&key.fShort.fSeekPdir, buf + sizeof(std::uint32_t), sizeof(std::uint32_t));
   }

   // Skip the class name
   char classNameLen;
   failed = file->ReadBuffer(&classNameLen, sizeof(classNameLen));
   if (failed) {
      return R__FAIL("Corrupted key data");
   }
   file->Seek(classNameLen, TFile::kCur);

   // Read the object name
   failed = file->ReadBuffer(reinterpret_cast<char *>(&key.fObjNameLen), sizeof(key.fObjNameLen));
   if (failed) {
      return R__FAIL("Corrupted key data");
   }
   file->ReadBuffer(key.fObjName, key.fObjNameLen);
   key.fObjName[key.fObjNameLen] = 0;

   return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<std::string_view, std::string_view> ROOT::Experimental::DecomposePath(std::string_view path)
{
   auto lastSlashIdx = path.rfind('/');
   if (lastSlashIdx == std::string_view::npos)
      return {{}, path};

   auto dirName = path.substr(0, lastSlashIdx + 1);
   auto pathName = path.substr(lastSlashIdx + 1);
   return {dirName, pathName};
}

bool RFile::IsValidPath(std::string_view path)
{
   if (path.empty())
      return false;

   if (path == "." || path == "..")
      return false;

   for (char ch : path) {
      // Disallow control characters, tabs, newlines and whitespace
      if (ch < 33)
         return false;
   }

   return true;
}

std::unique_ptr<RFile> RFile::OpenForReading(std::string_view path)
{
   ThrowIfExtensionIsNotRoot(path);

   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

std::unique_ptr<RFile> RFile::OpenForUpdate(std::string_view path)
{
   ThrowIfExtensionIsNotRoot(path);

   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "UPDATE_WITHOUT_GLOBALREGISTRATION"));
   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

std::unique_ptr<RFile> RFile::Recreate(std::string_view path)
{
   ThrowIfExtensionIsNotRoot(path);

   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "RECREATE_WITHOUT_GLOBALREGISTRATION"));
   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

TKey *RFile::GetTKey(const char *path) const
{
   const auto GetKeyCheckCycle = [](TDirectory *dir, const char *keyPath) -> TKey * {
      TKey *key = dir->FindKey(keyPath);
      if (key) {
         // For some reason, FindKey will not return nullptr if we asked for a specific cycle and that cycle
         // doesn't exist. It will instead return any key whose cycle is *at most* the requested one.
         // This is very confusing, so in RFile we actually return null if the requested cycle is not there.
         short cycle;
         TDirectory::DecodeNameCycle(keyPath, nullptr, cycle);
         // NOTE: cycle == 9999 means that `path` didn't contain a valid cycle (including no cycle at all)
         //       cycle == 10000 means that `path` contained the "any cycle" pattern ("name;*")
         if (cycle >= 9999 || key->GetCycle() == cycle) {
            return key;
         }
      }
      return nullptr;
   };

   // In RFile, differently from TFile, when dealing with a path like "a/b/c", we always consider it to mean
   // "object 'c' in subdirectory 'b' of directory 'a'". We don't try to get any other of the possible combinations,
   // including the object called "a/b/c".
   std::string fullPath = path;
   char *dirName = fullPath.data();
   char *restOfPath = strchr(dirName, '/');
   TDirectory *dir = fFile.get();
   while (restOfPath) {
      // Truncate `dirName` to the position of this '/'.
      *restOfPath = 0;
      ++restOfPath;

      // This can only happen if `path` ends with '/' (which it shouldn't).
      if (R__unlikely(!*restOfPath))
         return nullptr;

      dir = dir->GetDirectory(dirName);
      if (!dir)
         return nullptr;

      dirName = restOfPath;
      restOfPath = strchr(restOfPath, '/');
   }
   // NOTE: after this loop `dirName` contains the base name of the object.

   // First try getting the object from the top-level directory.
   // Don't use GetObjectChecked or similar because they don't handle slashes in paths correctly
   // (note that an object might have a name containing slashes even if it's not in a directory).
   TKey *key = GetKeyCheckCycle(dir, dirName);
   return key;
}

void *RFile::GetUntyped(const char *path, const TClass *type) const
{
   assert(path);
   assert(type);

   if (!fFile)
      throw ROOT::RException(R__FAIL("File has been closed"));

   TKey *key = GetTKey(path);
   void *obj = key ? key->ReadObjectAny(type) : nullptr;

   if (obj) {
      // Disavow any ownership on `obj`
      // NOTE: all these objects inherit from TObject as their first parent, so we can use static_cast.
      if (type->InheritsFrom("TTree"))
         static_cast<TTree *>(obj)->SetDirectory(nullptr);
      else if (type->InheritsFrom("TH1"))
         static_cast<TH1 *>(obj)->SetDirectory(nullptr);
      else if (type->InheritsFrom("TGraph2D"))
         static_cast<TGraph2D *>(obj)->SetDirectory(nullptr);
   }

   return obj;
}

void RFile::PutUntyped(const char *path, const TClass *type, const void *obj, std::uint32_t flags)
{
   assert(path);
   assert(type);

   if (!fFile)
      throw ROOT::RException(R__FAIL("File has been closed"));

   if (!fFile->IsWritable())
      throw ROOT::RException(R__FAIL("File is not writable"));

   // If `path` refers to a subdirectory, make sure we always write in an actual TDirectory,
   // otherwise we may have a mix of top-level objects called "a/b/c" and actual directory
   // structures.
   // Very sadly, TFile does nothing to prevent this and will happily write "a/b" even if there
   // is already a directory "a" containing an object "b". We don't want that kind of ambiguity here.
   const auto tokens = ROOT::Split(path, "/");
   TDirectory *dir = fFile.get();
   std::string fullPathSoFar = "";
   for (auto tokIdx = 0u; tokIdx < tokens.size() - 1; ++tokIdx) {
      fullPathSoFar += tokens[tokIdx];
      // Alas, not only does mkdir not fail if the file already contains an object "a/b" and you try
      // to create dir "a", but even when it does fail it doesn't tell you why.
      // We obviously don't want to allow the coexistence of regular object named "a/b" and the directory
      // named "a", so we manually check if each level of nesting doesn't exist already as a non-directory.
      const TKey *existing = dir->GetKey(tokens[tokIdx].c_str());
      if (existing && strcmp(existing->GetClassName(), "TDirectory") != 0 &&
          strcmp(existing->GetClassName(), "TDirectoryFile") != 0) {
         throw ROOT::RException(R__FAIL(std::string("failed to create directory ") + fullPathSoFar +
                                        ": name already taken by an object of type " + existing->GetClassName()));
      }
      dir = dir->mkdir(tokens[tokIdx].c_str(), "", true);
      if (!dir) {
         throw ROOT::RException(R__FAIL(std::string("failed to create directory ") + fullPathSoFar));
      }
   }

   const bool allowOverwrite = (flags & kPutAllowOverwrite) != 0;
   const bool backupCycle = (flags & kPutOverwriteKeepCycle) != 0;
   const Option_t *writeOpts = "";
   if (!allowOverwrite) {
      const TKey *existing = dir->GetKey(tokens[tokens.size() - 1].c_str());
      if (existing) {
         throw ROOT::RException(R__FAIL(std::string("trying to overwrite object ") + path + " of type " +
                                        existing->GetClassName() + " with another object of type " + type->GetName()));
      }
   } else if (!backupCycle) {
      writeOpts = "WriteDelete";
   }

   int success = dir->WriteObjectAny(obj, type, tokens[tokens.size() - 1].c_str(), writeOpts);

   if (!success) {
      throw ROOT::RException(R__FAIL(std::string("Failed to write ") + path + " to file"));
   }
}

// Returns {fullPath, nestingLevel}
ROOT::RResult<std::pair<std::string, int>> ROOT::Experimental::RFileKeyIterable::RIterator::ReconstructFullKeyPath(
   ROOT::Detail::TKeyMapIterable::TIterator &iter) const
{
   auto seekPdir = iter->fSeekPdir;
   int nesting = 0;
   // We need to accumulate the dir names and append them in reverse.
   std::vector<std::string> dirNames;
   while (seekPdir != 0x64) { // XXX: hardcoded TFile kBEGIN
      auto keyRes = ReadKeyFromFile(fFile, seekPdir);
      if (!keyRes)
         return R__FORWARD_ERROR(keyRes);
      RTFileKey key = keyRes.Unwrap();
      dirNames.push_back(key.fObjName);
      seekPdir = key.SeekPdir();
      ++nesting;
   }

   std::string fullName;
   for (int i = static_cast<int>(dirNames.size()) - 1; i >= 0; --i) {
      fullName += dirNames[i] + "/";
   }
   fullName += iter->fKeyName;
   return std::make_pair(fullName, nesting);
}

void ROOT::Experimental::RFileKeyIterable::RIterator::Advance()
{
   fCurKey = {};

   // We only want to return keys that refer to user objects, not internal ones, therefore we skip
   // all keys that have internal class names.
   while (1) {
      ++fIter;

      if (!fIter.operator->()) {
         // reached end of the iteration
         break;
      }

      if (IsInternalKey(fIter->fClassName.c_str()))
         continue;

      // skip all directories
      if (fIter->fClassName == "TDirectory" || fIter->fClassName == "TDirectoryFile")
         continue;

      // Reconstruct the full path of the key
      // TODO: better error handling
      const auto [fullPath, nesting] = ReconstructFullKeyPath(fIter).Unwrap();

      // skip key if it's not a child of root dir
      if (!HasPrefix(fullPath, fRootDir))
         continue;

      // check that we are in the same directory as "rootDir".
      if (!fRecursive && nesting != fRootDirNesting)
         continue;

      // All checks passed: return this key.
      fCurKey.fName = fullPath;
      fCurKey.fTitle = fIter->fKeyTitle;
      fCurKey.fClassName = fIter->fClassName;
      break;
   }
}

void RFile::Print(std::ostream &out) const
{
   std::vector<RFileKeyInfo> keys;
   auto keysIter = GetKeys();
   for (const auto &key : keysIter) {
      keys.emplace_back(key);
   }

   std::sort(keys.begin(), keys.end(), [](const auto &a, const auto &b) { return a.fName < b.fName; });
   for (const auto &key : keys) {
      out << key.fClassName << " " << key.fName << ": \"" << key.fTitle << "\"\n";
   }
}

TFile *ROOT::Experimental::Internal::GetRFileTFile(ROOT::Experimental::RFile &file)
{
   return file.fFile.get();
}

size_t RFile::Write()
{
   return fFile->Write();
}

void RFile::Close()
{
   fFile.reset();
}

std::optional<ROOT::Experimental::RFileKeyInfo> RFile::GetKeyInfo(std::string_view path) const
{
   TKey *key = GetTKey(std::string(path).c_str());
   if (key)
      return RFileKeyInfoFromTKey(*key);
   return {};
}

void *ROOT::Experimental::Internal::GetRFileObjectFromKey(RFile &file, const RFileKeyInfo &key)
{
   const auto *cls = TClass::GetClass(key.fClassName.c_str());
   void *obj = nullptr;
   if (cls)
      obj = file.GetUntyped(key.fName.c_str(), cls);
   return obj;
}
