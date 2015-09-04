/// \file TFile.cxx
/// \ingroup Base
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-31

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TFile.h"

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
static void AddFilesToClose(ROOT::TCoopPtr<ROOT::TFile> pFile) {
  struct CloseFiles_t {
    std::vector<ROOT::TCoopPtr<ROOT::TFile>> fFiles;
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
}

ROOT::TCoopPtr<ROOT::TFile> ROOT::TFile::Read(std::string_view name) {
  return TCoopPtr<TFile>(new TFile()); // will become delegation to TFileSystemFile, TWebFile etc.
}
ROOT::TCoopPtr<ROOT::TFile> ROOT::TFile::Create(std::string_view name) {
  TCoopPtr<TFile> ptr(new TFile()); // will become delegation to TFileSystemFile, TWebFile etc.
  AddFilesToClose(ptr);
  return ptr;
}
ROOT::TCoopPtr<ROOT::TFile> ROOT::TFile::Recreate(std::string_view name) {
  TCoopPtr<TFile> ptr(new TFile()); // will become delegation to TFileSystemFile, TWebFile etc.
  AddFilesToClose(ptr);
  return ptr;
}
