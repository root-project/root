/// \file v7/src/TFile.cxx
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TFile.h"
#include "TFile.h"

#include <memory>
#include <mutex>
#include <string>

ROOT::Experimental::TDirectory& ROOT::Experimental::TDirectory::Heap() {
  static TDirectory heapDir;
  return heapDir;
}

namespace {
/// We cannot afford users not closing their files. Yes, we return a unique_ptr -
/// but that might be stored in an object that itself leaks. That would leave
/// the TFile unclosed and data corrupted / not written. Instead, keep a
/// collection of all opened writable TFiles and close them at destruction time,
/// explicitly.
static void AddFilesToClose(std::weak_ptr<ROOT::Experimental::Internal::TFileImplBase> pFile) {
  struct CloseFiles_t {
    std::vector<std::weak_ptr<ROOT::Experimental::Internal::TFileImplBase>> fFiles;
    std::mutex fMutex;
    ~CloseFiles_t() {
      for (auto& wFile: fFiles) {
        if (auto sFile = wFile.lock()) {
          sFile->Flush(); // or Close()? but what if there's still a Write()?
        }
      }
    }
  };
  static CloseFiles_t closer;

  std::lock_guard<std::mutex> lock(closer.fMutex);
  closer.fFiles.emplace_back(pFile);
}

/** \class TFSFile
 TFileImplBase for a file-system (POSIX) style TFile.
 */
class TFileSystemFile: public ROOT::Experimental::Internal::TFileImplBase {
  ::TFile* fOldFile;

public:
  TFileSystemFile(const std::string& name, const std::string& mode):
    fOldFile(::TFile::Open(name.c_str(), mode.c_str())) {
  }

  void Flush() final { fOldFile->Flush(); }

  void Close() final { fOldFile->Close(); }

  ~TFileSystemFile() {
    delete fOldFile;
  }
};
}

ROOT::Experimental::TFilePtr::TFilePtr(std::unique_ptr<ROOT::Experimental::Internal::TFileImplBase>&& impl):
fImpl(std::move(impl))
{
  AddFilesToClose(fImpl);
}

namespace {
static std::string GetV6TFileOpts(const char* mode,
                                  const ROOT::Experimental::TFilePtr::Options_t& opts) {
  std::string ret(mode);
  if (opts.fCachedRead)
    ret += " CACHEREAD ";
  if (opts.fAsynchronousOpen && opts.fAsyncTimeout > 0)
    ret += " TIMEOUT=" + std::to_string(opts.fAsyncTimeout) + " ";
  return ret;
}

static std::mutex& GetCacheDirMutex() {
  static std::mutex sMutex;
  return sMutex;
}

static ROOT::Experimental::TFilePtr
OpenV6TFile(std::string_view name, const char* mode,
            const ROOT::Experimental::TFilePtr::Options_t& opts) {
  // Set and re-set the cache dir.
  // FIXME: do not modify a static here, pass this to the underlying Open.
  struct SetCacheDirRAII_t {
    std::string fOldCacheDir;
    std::lock_guard<std::mutex> fLock;

    SetCacheDirRAII_t(bool need): fLock(GetCacheDirMutex()) {
      if (need)
        fOldCacheDir = TFile::GetCacheFileDir();
    }

    ~SetCacheDirRAII_t() {
      if (!fOldCacheDir.empty())
        TFile::SetCacheFileDir(fOldCacheDir.c_str());
    }
  } setCacheDirRAII(opts.fCachedRead);

  std::unique_ptr<TFileSystemFile> fsf
    = std::make_unique<TFileSystemFile>(name.to_string(), GetV6TFileOpts(mode, opts));
  return ROOT::Experimental::TFilePtr(std::move(fsf));
}
}

ROOT::Experimental::TFilePtr
ROOT::Experimental::TFilePtr::Open(std::string_view name,
                                   const Options_t& opts /*= Options_t()*/) {
  // will become delegation to TFileSystemFile, TWebFile etc.
  return OpenV6TFile(name, "READ", opts);
}

ROOT::Experimental::TFilePtr
ROOT::Experimental::TFilePtr::Create(std::string_view name,
                                     const Options_t& opts /*= Options_t()*/) {
  // will become delegation to TFileSystemFile, TWebFile etc.
  return OpenV6TFile(name, "CREATE", opts);
}

ROOT::Experimental::TFilePtr
ROOT::Experimental::TFilePtr::Recreate(std::string_view name,
                                       const Options_t& opts /*= Options_t()*/) {
  // will become delegation to TFileSystemFile, TWebFile etc.
  return OpenV6TFile(name, "RECREATE", opts);
}

ROOT::Experimental::TFilePtr
ROOT::Experimental::TFilePtr::OpenForUpdate(std::string_view name,
                                            const Options_t& opts /*= Options_t()*/) {
  // will become delegation to TFileSystemFile, TWebFile etc.
  return OpenV6TFile(name, "UPDATE", opts);
}

std::string ROOT::Experimental::TFilePtr::SetCacheDir(std::string_view path) {
  std::lock_guard<std::mutex> lock(GetCacheDirMutex());

  std::string ret = TFile::GetCacheFileDir();
  TFile::SetCacheFileDir(path.to_string().c_str());
  return ret;
}

std::string ROOT::Experimental::TFilePtr::GetCacheDir() {
  return TFile::GetCacheFileDir();
}
