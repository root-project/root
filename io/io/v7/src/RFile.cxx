/// \file v7/src/TFile.cxx
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TFile.hxx"
#include "TFile.h"

#include <memory>
#include <mutex>
#include <string>

ROOT::Experimental::RDirectory &ROOT::Experimental::RDirectory::Heap()
{
   static RDirectory heapDir;
   return heapDir;
}

namespace ROOT {
namespace Experimental {
namespace Internal {
// This will have to move to some "semi-internal" header.
/** \class TFileStorageInterface
 Base class for TFile storage backends.
 */
class TFileStorageInterface {
public:
   virtual void Flush() = 0;
   virtual void Close() = 0;
   virtual ~TFileStorageInterface() = default;
   virtual void WriteMemoryWithType(std::string_view name, const void *address, TClass *cl) = 0;
};

// make_shared<TFile> doesn't work, as TFile() is private. Take detour
// through a friend instead.
class TFileSharedPtrCtor: public ROOT::Experimental::TFile {
public:
   TFileSharedPtrCtor(std::unique_ptr<TFileStorageInterface> &&storage): TFile(std::move(storage)) {}
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

namespace {
/// We cannot afford users not closing their files. Yes, we return a unique_ptr -
/// but that might be stored in an object that itself leaks. That would leave
/// the TFile unclosed and data corrupted / not written. Instead, keep a
/// collection of all opened writable TFiles and close them at destruction time,
/// explicitly.
static void AddFilesToClose(std::weak_ptr<ROOT::Experimental::TFile> pFile)
{
   struct CloseFiles_t {
      std::vector<std::weak_ptr<ROOT::Experimental::TFile>> fFiles;
      std::mutex fMutex;
      ~CloseFiles_t()
      {
         for (auto &wFile: fFiles) {
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

/** \class TV6Storage
 TFile for a ROOT v6 storage backend.
 */
class TV6Storage: public ROOT::Experimental::Internal::TFileStorageInterface {
   ::TFile *fOldFile;

public:
   TV6Storage(const std::string &name, const std::string &mode): fOldFile(::TFile::Open(name.c_str(), mode.c_str())) {}

   void Flush() final { fOldFile->Flush(); }

   void Close() final { fOldFile->Close(); }

   ~TV6Storage() { delete fOldFile; }

   void WriteMemoryWithType(std::string_view name, const void *address, TClass *cl) final
   {
      fOldFile->WriteObjectAny(address, cl, std::string(name).c_str());
   }
};
} // namespace

ROOT::Experimental::TFilePtr::TFilePtr(std::shared_ptr<ROOT::Experimental::TFile> &&file): fFile(std::move(file))
{
   AddFilesToClose(fFile);
}

namespace {
static std::string GetV6TFileOpts(const char *mode, const ROOT::Experimental::TFile::Options_t &opts)
{
   std::string ret(mode);
   if (opts.fCachedRead)
      ret += " CACHEREAD ";
   if (opts.fAsynchronousOpen && opts.fAsyncTimeout > 0)
      ret += " TIMEOUT=" + std::to_string(opts.fAsyncTimeout) + " ";
   return ret;
}

static std::mutex &GetCacheDirMutex()
{
   static std::mutex sMutex;
   return sMutex;
}

static std::unique_ptr<ROOT::Experimental::Internal::TFileStorageInterface>
OpenV6TFile(std::string_view name, const char *mode, const ROOT::Experimental::TFile::Options_t &opts)
{
   // Set and re-set the cache dir.
   // FIXME: do not modify a static here, pass this to the underlying Open.
   struct SetCacheDirRAII_t {
      std::string fOldCacheDir;
      std::lock_guard<std::mutex> fLock;

      SetCacheDirRAII_t(bool need): fLock(GetCacheDirMutex())
      {
         if (need)
            fOldCacheDir = TFile::GetCacheFileDir();
      }

      ~SetCacheDirRAII_t()
      {
         if (!fOldCacheDir.empty())
            TFile::SetCacheFileDir(fOldCacheDir.c_str());
      }
   } setCacheDirRAII(opts.fCachedRead);

   auto v6storage = std::make_unique<TV6Storage>(std::string(name), GetV6TFileOpts(mode, opts));

   using namespace ROOT::Experimental::Internal;
   return std::unique_ptr<TFileStorageInterface>{std::move(v6storage)};
}
} // namespace

ROOT::Experimental::TFilePtr ROOT::Experimental::TFile::Open(std::string_view name,
                                                             const Options_t &opts /*= Options_t()*/)
{
   // will become delegation to TFileSystemFile, TWebFile etc.
   using namespace Internal;
   auto file = std::make_shared<TFileSharedPtrCtor>(OpenV6TFile(name, "READ", opts));
   return ROOT::Experimental::TFilePtr(std::move(file));
}

ROOT::Experimental::TFilePtr ROOT::Experimental::TFile::Create(std::string_view name,
                                                               const Options_t &opts /*= Options_t()*/)
{
   // will become delegation to TFileSystemFile, TWebFile etc.
   using namespace Internal;
   auto file = std::make_shared<TFileSharedPtrCtor>(OpenV6TFile(name, "CREATE", opts));
   return ROOT::Experimental::TFilePtr(std::move(file));
}

ROOT::Experimental::TFilePtr ROOT::Experimental::TFile::Recreate(std::string_view name,
                                                                 const Options_t &opts /*= Options_t()*/)
{
   // will become delegation to TFileSystemFile, TWebFile etc.
   using namespace Internal;
   auto file = std::make_shared<TFileSharedPtrCtor>(OpenV6TFile(name, "RECREATE", opts));
   return ROOT::Experimental::TFilePtr(std::move(file));
}

ROOT::Experimental::TFilePtr ROOT::Experimental::TFile::OpenForUpdate(std::string_view name,
                                                                      const Options_t &opts /*= Options_t()*/)
{
   // will become delegation to TFileSystemFile, TWebFile etc.
   using namespace Internal;
   auto file = std::make_shared<TFileSharedPtrCtor>(OpenV6TFile(name, "UPDATE", opts));
   return ROOT::Experimental::TFilePtr(std::move(file));
}

std::string ROOT::Experimental::TFile::SetCacheDir(std::string_view path)
{
   std::lock_guard<std::mutex> lock(GetCacheDirMutex());

   std::string ret = ::TFile::GetCacheFileDir();
   ::TFile::SetCacheFileDir(std::string(path).c_str());
   return ret;
}

std::string ROOT::Experimental::TFile::GetCacheDir()
{
   std::lock_guard<std::mutex> lock(GetCacheDirMutex());
   return ::TFile::GetCacheFileDir();
}

// Implement outlined, to hide implementation of TFileStorageInterface from
// header.
ROOT::Experimental::TFile::TFile(std::unique_ptr<ROOT::Experimental::Internal::TFileStorageInterface> &&storage)
   : fStorage(std::move(storage))
{}

// Implement outlined, to hide implementation of TFileStorageInterface from
// header.
ROOT::Experimental::TFile::~TFile() = default;

void ROOT::Experimental::TFile::Flush()
{
   fStorage->Flush();
}
void ROOT::Experimental::TFile::Close()
{
   fStorage->Close();
}
void ROOT::Experimental::TFile::WriteMemoryWithType(std::string_view name, const void *address, TClass *cl)
{
   fStorage->WriteMemoryWithType(name, address, cl);
}
