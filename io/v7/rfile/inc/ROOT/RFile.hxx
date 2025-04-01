/// \file ROOT/RFile.hxx
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

#ifndef ROOT7_RFile
#define ROOT7_RFile

#include <ROOT/RError.hxx>
#include <TFile.h>
#include <string_view>
#include <memory>

namespace ROOT {
namespace Experimental {

class RFileKeyIterable final {
   TFile *fFile;
   std::string fRootDir;
   bool fRecursive;

public:
   class RIterator {
      friend class RFileKeyIterable;

      TFile *fFile;
      ROOT::Detail::TKeyMapIterable::TIterator fIter;
      std::string_view fRootDir;
      std::string fCurKeyName;
      int fRootDirNesting = 0;
      bool fRecursive;

      ROOT::RResult<std::pair<std::string, int>>
      ReconstructFullKeyPath(ROOT::Detail::TKeyMapIterable::TIterator &iter) const;

      void Advance();

      RIterator(TFile *file, ROOT::Detail::TKeyMapIterable::TIterator iter, std::string_view rootDir, bool recursive)
         : fFile(file), fIter(iter), fRootDir(rootDir), fRecursive(recursive)
      {
         if (!rootDir.empty()) {
            fRootDirNesting = std::count(rootDir.begin(), rootDir.end(), '/');
            // `rootDir` may or may not end with '/', but we consider it a directory regardless.
            // In other words, like in virtually all filesystem operations, "dir" and "dir/" are equivalent.
            fRootDirNesting += rootDir[rootDir.size() - 1] != '/';
         }

         // Advance the iterator to skip the first key, which is always the TFile key.
         // This will also skip keys until we reach the first correct key we want to return.
         Advance();
      }

   public:
      using iterator = RIterator;
      using iterator_category = std::forward_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = std::string;
      using pointer = value_type *;
      using reference = value_type &;

      iterator operator++()
      {
         Advance();
         return *this;
      }
      reference operator*() { return fCurKeyName; }
      pointer operator->() { return &fCurKeyName; }
      bool operator!=(const iterator &rh) const { return !(*this == rh); }
      bool operator==(const iterator &rh) const { return fIter == rh.fIter; }
   };

   explicit RFileKeyIterable(TFile *file, std::string_view rootDir, bool recursive)
      : fFile(file), fRootDir(rootDir), fRecursive(recursive)
   {
   }

   RIterator begin() const { return {fFile, fFile->WalkTKeys().begin(), fRootDir, fRecursive}; }
   RIterator end() const { return {fFile, fFile->WalkTKeys().end(), fRootDir, fRecursive}; }
};

class RFile final {
   std::unique_ptr<TFile> fFile;

   explicit RFile(std::unique_ptr<TFile> file) : fFile(std::move(file)) {}

   // NOTE: these strings are const char * because they need to be passed to TFile
   /// Gets object `path` from the file and returns an **owning** pointer to it.
   /// The caller should immediately wrap it into a unique_ptr of the type described by `type`.
   [[nodiscard]] void *GetUntyped(const char *path, const TClass *type) const;

   /// Writes `obj` to file, without taking its ownership.
   void PutUntyped(const char *path, const TClass *type, void *obj);

public:
   ///// Factory methods /////

   /// Opens the file for reading
   static std::unique_ptr<RFile> OpenForReading(std::string_view path);

   /// Opens the file for reading/writing, overwriting it if it already exists
   static std::unique_ptr<RFile> Recreate(std::string_view path);

   /// Opens the file for updating
   static std::unique_ptr<RFile> OpenForUpdate(std::string_view path);

   ///// Utility methods /////

   /// Returns true if `path` is a suitable path to store an object into a RFile.
   /// Passing an invalid path to Put will cause it to throw an exception, and
   /// passing an invalid path to Get will always return nullptr.
   static bool IsValidPath(std::string_view path);

   ///// Instance methods /////

   /// Retrieves an object from the file.
   /// If the object is not there returns a null pointer.
   template <typename T>
   std::unique_ptr<T> Get(std::string_view path) const
   {
      if (!IsValidPath(path)) {
         return nullptr;
      }
      std::string pathStr(path);
      const TClass *cls = TClass::GetClass(typeid(T));
      if (!cls) {
         throw ROOT::RException(R__FAIL(std::string("Could not determine type of object ") + pathStr));
      }
      void *obj = GetUntyped(pathStr.c_str(), cls);
      return std::unique_ptr<T>(static_cast<T *>(obj));
   }

   /// Puts an object into the file.
   /// The application retains ownership of the object.
   /// Throws a RException if the file was opened in read-only mode.
   template <typename T>
   void Put(std::string_view path, T &obj)
   {
      if (!IsValidPath(path)) {
         throw RException(R__FAIL(std::string("Invalid object path: ") + std::string(path)));
      }
      std::string pathStr(path);
      const TClass *cls = TClass::GetClass(typeid(T));
      if (!cls) {
         throw ROOT::RException(R__FAIL(std::string("Could not determine type of object ") + pathStr));
      }
      PutUntyped(pathStr.c_str(), cls, &obj);
   }

   /// Returns an iterable over all paths of objects written into this RFile starting at directory "rootDir".
   /// The returned paths are always "absolute" paths: they are not relative to `rootDir`.
   /// Keys relative to directories are not returned.
   /// This recurses on all the subdirectories of `rootDir`. If you only want the immediate children of `rootDir`,
   /// use GetKeysNonRecursive().
   RFileKeyIterable GetKeys(std::string_view rootDir = "") const
   {
      return RFileKeyIterable(fFile.get(), rootDir, /* recursive = */ true);
   }

   /// Returns an iterable over all paths of objects written into this RFile contained in the directory "rootDir".
   /// The returned paths are always "absolute" paths: they are not relative to `rootDir`.
   /// Keys relative to directories are not returned.
   /// This only returns the immediate children of `rootDir`. If you want to recurse into the subdirectories of
   /// `rootDir`, use GetKeys().
   RFileKeyIterable GetKeysNonRecursive(std::string_view rootDir = "") const
   {
      return RFileKeyIterable(fFile.get(), rootDir, /* recursive = */ false);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
