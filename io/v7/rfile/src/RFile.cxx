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
#include <TTree.h>
#include <TGraph2D.h>
#include <TH1.h>
#include <TKey.h>

using ROOT::Experimental::RFile;
// using ROOT::Experimental::RFileRef;

static void CheckExtension(std::string_view path)
{
   if (path.size() < 5 || path.compare(path.size() - 5, 5, ".root") != 0) {
      throw ROOT::RException(R__FAIL(std::string("Only .root files are supported.")));
   }
}

std::unique_ptr<RFile> RFile::OpenForReading(std::string_view path)
{
   CheckExtension(path);

   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

std::unique_ptr<RFile> RFile::OpenForUpdate(std::string_view path)
{
   CheckExtension(path);

   // NOTE: "UPDATE_WITHOUT_GLOBALREGISTRATION" is undocumented but works.
   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "UPDATE_WITHOUT_GLOBALREGISTRATION"));
   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

std::unique_ptr<RFile> RFile::Recreate(std::string_view path)
{
   CheckExtension(path);

   // NOTE: "RECREATE_WITHOUT_GLOBALREGISTRATION" is undocumented but works.
   auto tfile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "RECREATE_WITHOUT_GLOBALREGISTRATION"));
   auto rfile = std::unique_ptr<RFile>(new RFile(std::move(tfile)));
   return rfile;
}

void *RFile::GetUntyped(const char *path, const TClass *type) const
{
   assert(path);

   if (!type) {
      throw ROOT::RException(R__FAIL(std::string("Could not determine type of object ") + path));
   }

   // First try getting the object from the top-level directory.
   // Don't use GetObjectChecked or similar because they don't handle slashes in paths correctly
   // (note that an object might have a name containing slashes even if it's not in a directory).
   TKey *key = fFile->FindKey(path);
   void *obj = nullptr;
   if (key) {
      obj = key->ReadObjectAny(type);
   }

   // If we didn't find the object, try in subdirectories.
   if (!obj) {
      auto tokens = ROOT::Split(path, "/");
      if (tokens.size() > 1) {
         TDirectory *dir = fFile.get();
         for (auto tokenIdx = 0u; tokenIdx < tokens.size() - 1; ++tokenIdx) {
            dir = dir->GetDirectory(tokens[tokenIdx].c_str());
            if (!dir) {
               return nullptr;
            }
         }

         const auto &keyName = tokens[tokens.size() - 1];
         TKey *key = dir->FindKey(keyName.c_str());
         if (!key) {
            return key;
         }
         obj = key->ReadObjectAny(type);
      }
   }

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

void RFile::PutUntyped(const char *path, const TClass *type, void *obj)
{
   if (!type) {
      throw ROOT::RException(R__FAIL(std::string("Could not determine type of object ") + path));
   }
   if (!fFile->IsWritable()) {
      throw ROOT::RException(R__FAIL("File is not writable"));
   }

   int success = fFile->WriteObjectAny(obj, type, path);

   if (!success) {
      throw ROOT::RException(R__FAIL(std::string("Failed to write ") + path + " to file"));
   }
}
