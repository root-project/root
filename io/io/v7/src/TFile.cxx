/// \file TFile.cxx
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

#include <mutex>

ROOT::TDirectory& ROOT::TDirectory::Heap() {
  static TDirectory heapDir;
  return heapDir;
}

namespace {
/// We cannot afford users not closing their files. Yes, we return a unique_ptr -
/// but that might be stored in an object that itself leaks. That would leave
/// the TFile unclosed and data corrupted / not written. Instead, keep a
/// collection of all opened writable TFiles and close them at destruction time,
/// explicitly.
static void AddFilesToClose(ROOT::TCoopPtr<ROOT::Internal::TFileImplBase> pFile) {
  struct CloseFiles_t {
    std::vector<ROOT::TCoopPtr<ROOT::Internal::TFileImplBase>> fFiles;
    std::mutex fMutex;
    ~CloseFiles_t() {
      for (auto& pFile: fFiles)
        if (pFile)
          pFile->Flush(); // or Close()? but what if there's still a Write()?
    }
  };
  static CloseFiles_t closer;

  std::lock_guard<std::mutex> lock(closer.fMutex);
  closer.fFiles.emplace_back(pFile);
}

/** \class TFSFile
 TFileImplBase for a file-system (POSIX) style TFile.
 */
class TFileSystemFile: public ROOT::Internal::TFileImplBase {
  ::TFile* fOldFile;

public:
  TFileSystemFile(const std::string& name, const char* mode):
    fOldFile(::TFile::Open(name.c_str(), mode)) {
  }

  void Flush() final { fOldFile->Flush(); }

  void Close() final { fOldFile->Close(); }

  ~TFileSystemFile() {
    delete fOldFile;
  }
};
}

ROOT::TFilePtr::TFilePtr(TCoopPtr<ROOT::Internal::TFileImplBase> impl):
fImpl(impl)
{
  AddFilesToClose(impl);
}


ROOT::TFilePtr ROOT::TFilePtr::OpenForRead(std::string_view name) {
  // will become delegation to TFileSystemFile, TWebFile etc.
  return TFilePtr(MakeCoop<TFileSystemFile>(name.to_string(), "READ"));
}
ROOT::TFilePtr ROOT::TFilePtr::Create(std::string_view name) {
  // will become delegation to TFileSystemFile, TWebFile etc.
  return TFilePtr(MakeCoop<TFileSystemFile>(name.to_string(), "CREATE"));
}
ROOT::TFilePtr ROOT::TFilePtr::Recreate(std::string_view name) {
  // will become delegation to TFileSystemFile, TWebFile etc.
  return TFilePtr(MakeCoop<TFileSystemFile>(name.to_string(), "RECREATE"));
}
ROOT::TFilePtr ROOT::TFilePtr::OpenForUpdate(std::string_view name) {
  // will become delegation to TFileSystemFile, TWebFile etc.
  return TFilePtr(MakeCoop<TFileSystemFile>(name.to_string(), "UPDATE"));
}
