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
#include <iostream>

namespace ROOT {
namespace Experimental {

class RFile;
struct RFileKeyInfo;

namespace Internal {

TFile *GetRFileTFile(RFile &file);
/// Returns an **owning** pointer to the object referenced by `key`. The caller must delete this pointer.
[[nodiscard]] void *GetRFileObjectFromKey(RFile &file, const RFileKeyInfo &key);

}

/// Given a "path-like" string (like foo/bar/baz), returns a pair `{ dirName, baseName }`.
/// `baseName` will be empty if the string ends with '/'.
/// `dirName` will be empty if the string contains no '/'.
/// `dirName`, if not empty, always ends with a '/'.
/// NOTE: this function does no semantic checking or path expansion, nor does it interact with the
/// filesystem in any way (so it won't follow symlink or anything like that).
/// Moreover it doesn't trim the path in any way, so any leading or trailing whitespaces will be preserved.
/// This function does not perform any copy: the returned string_views have the same lifetime as `path`.
std::pair<std::string_view, std::string_view> DecomposePath(std::string_view path);

struct RFileKeyInfo {
  std::string fName;
  std::string fTitle;
  std::string fClassName; 
};

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
      RFileKeyInfo fCurKey;
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
      using value_type = RFileKeyInfo;
      using pointer = value_type *;
      using reference = value_type &;

      iterator operator++()
      {
         Advance();
         return *this;
      }
      reference operator*() { return fCurKey; }
      pointer operator->() { return &fCurKey; }
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
   friend TFile *Internal::GetRFileTFile(RFile &file);
   friend void *Internal::GetRFileObjectFromKey(RFile &file, const RFileKeyInfo &key);

   enum PutFlags {
      kPutAllowOverwrite = 0x1,
      kPutOverwriteKeepCycle = 0x2,   
   };
   
   std::unique_ptr<TFile> fFile;

   explicit RFile(std::unique_ptr<TFile> file) : fFile(std::move(file)) {}

   // NOTE: these strings are const char * because they need to be passed to TFile.
   /// Gets object `path` from the file and returns an **owning** pointer to it.
   /// The caller should immediately wrap it into a unique_ptr of the type described by `type`.
   [[nodiscard]] void *GetUntyped(const char *path, const TClass *type) const;

   /// Writes `obj` to file, without taking its ownership.
   void PutUntyped(const char *path, const TClass *type, const void *obj, std::uint32_t flags);

   // XXX: consider exposing this function
   template <typename T>
   void PutInternal(std::string_view path, const T &obj, std::uint32_t flags)
   {
      if (!IsValidPath(path)) {
         throw RException(R__FAIL(std::string("Invalid object path: ") + std::string(path)));
      }
      std::string pathStr(path);
      const TClass *cls = TClass::GetClass(typeid(T));
      if (!cls) {
         throw ROOT::RException(R__FAIL(std::string("Could not determine type of object ") + pathStr));
      }
      PutUntyped(pathStr.c_str(), cls, &obj, flags);
   }

   /// Retrieves the TKey with path (and cycle) `path`. It will first look at a top-level key with name
   /// equal to `path`, then, if it's not found, it will recurse the subdirectories to look for it.
   /// e.g. path "a/b/c" will first be looked up as the name "a/b/c" in the top-level dir, then as "b/c" in dir "a"
   /// and then as name "c" in dir "a/b".
   /// Even though the RFile always stores the object in a subdirectory in such a way that the object name never
   /// contains a slash, TFile doesn't offer that guarantee, so we need to do this elaborate search to be compatible
   /// with it. This may change in the future if we stop relying on TFile.
   TKey *GetTKey(const char *path) const;

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
   ///
   /// Throws a RException if a directory already exists at `path`.
   /// Throws a RException if an object already exists at `path`.
   /// Throws a RException if the file was opened in read-only mode.
   template <typename T>
   void Put(std::string_view path, const T &obj)
   {
      PutInternal(path, obj, /* flags = */ 0);
   }

   /// Puts an object into the file, overwriting any previously-existing object at that path.
   /// The application retains ownership of the object.
   ///
   /// If an object already exists at that path, it is kept as a backup cycle unless `backupPrevious` is false.
   /// Note that even if `backupPrevious` is false, any existing cycle except the latest will be preserved.
   ///
   /// Throws a RException if a directory already exists at `path`.
   /// Throws a RException if the file was opened in read-only mode.
   template <typename T>
   void Overwrite(std::string_view path, const T &obj, bool backupPrevious = true)
   {
      std::uint32_t flags = kPutAllowOverwrite;
      flags |= backupPrevious * kPutOverwriteKeepCycle;
      PutInternal(path, obj, flags);
   }

   /// Writes all objects to disk with the file structure.
   /// Returns the number of bytes written.
   size_t Write();

   /// Closes the RFile, disallowing any further reading or writing.
   void Close();

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

   /// Retrieves information about the key `path`.
   std::optional<RFileKeyInfo> GetKeyInfo(std::string_view path) const;

   /// Prints the internal structure of this RFile to the given stream.
   void Print(std::ostream &out = std::cout) const;
};

} // namespace Experimental
} // namespace ROOT

#endif
