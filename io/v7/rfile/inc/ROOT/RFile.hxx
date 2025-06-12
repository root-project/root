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
#include <regex>
#include <variant>

namespace ROOT {
namespace Experimental {

class RFile;
struct RFileKeyInfo;

namespace Internal {

TFile *GetRFileTFile(RFile &file);
/// Returns an **owning** pointer to the object referenced by `key`. The caller must delete this pointer.
[[nodiscard]] void *GetRFileObjectFromKey(RFile &file, const RFileKeyInfo &key);

ROOT::RLogChannel &RFileLog();

} // namespace Internal

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
   std::uint16_t fCycle;
};

class RFileKeyIterable final {
   TFile *fFile;
   std::variant<std::string, std::regex> fPattern;
   std::uint32_t fFlags = 0;

public:
   enum EFlags {
      kNone = 0,
      kRecursive = 1 << 0,
   };

   class RIterator {
      friend class RFileKeyIterable;

      TFile *fFile;
      ROOT::Detail::TKeyMapIterable::TIterator fIter;
      std::variant<std::string, std::regex> fPattern;
      RFileKeyInfo fCurKey;
      std::uint16_t fRootDirNesting = 0;
      std::uint32_t fFlags = 0;

      ROOT::RResult<std::pair<std::string, int>>
      ReconstructFullKeyPath(ROOT::Detail::TKeyMapIterable::TIterator &iter) const;

      void Advance();

      RIterator(TFile *file, ROOT::Detail::TKeyMapIterable::TIterator iter,
                std::variant<std::string, std::regex> pattern, std::uint32_t flags)
         : fFile(file), fIter(iter), fPattern(pattern), fFlags(flags)
      {
         if (auto *patternStr = std::get_if<std::string>(&pattern); patternStr && !patternStr->empty()) {
            fRootDirNesting = std::count(patternStr->begin(), patternStr->end(), '/');
            // `patternStr` may or may not end with '/', but we consider it a directory regardless.
            // In other words, like in virtually all filesystem operations, "dir" and "dir/" are equivalent.
            fRootDirNesting += (*patternStr)[patternStr->size() - 1] != '/';
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

   explicit RFileKeyIterable(TFile *file, std::string_view rootDir, std::uint32_t flags)
      : fFile(file), fPattern(std::string(rootDir)), fFlags(flags)
   {
   }

   explicit RFileKeyIterable(TFile *file, const std::regex &regex) : fFile(file), fPattern(regex), fFlags(kRecursive) {}

   RIterator begin() const { return {fFile, fFile->WalkTKeys().begin(), fPattern, fFlags}; }
   RIterator end() const { return {fFile, fFile->WalkTKeys().end(), fPattern, fFlags}; }
};

class RFile final {
   friend TFile *Internal::GetRFileTFile(RFile &file);
   friend void *Internal::GetRFileObjectFromKey(RFile &file, const RFileKeyInfo &key);

   enum PutFlags {
      kPutAllowOverwrite = 0x1,
      kPutOverwriteKeepCycle = 0x2,
   };

   std::unique_ptr<TFile> fFile;

   /// Returns an empty string if `path` is a suitable path to store an object into a RFile,
   /// otherwise returns a description of why that is not the case.
   ///
   /// A valid object path must:
   ///   - not be empty
   ///   - not contain the character '.'
   ///   - not contain ASCII control characters or whitespace characters (including tab or newline).
   ///   - not contain more than RFile::kMaxPathNesting path fragments (i.e. more than RFile::kMaxPathNesting - 1 '/')
   ///
   /// In addition, when *writing* an object to RFile, the character ';' is also banned.
   ///
   /// Passing an invalid path to Put will cause it to throw an exception, and
   /// passing an invalid path to Get will always return nullptr.
   static std::string ValidatePath(std::string_view path);

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
      if (auto err = ValidatePath(path); !err.empty()) {
         throw RException(R__FAIL("Invalid object path '" + std::string(path) + "': " + err));
      }
      if (path.find_first_of(';') != std::string_view::npos) {
         throw RException(
            R__FAIL("Invalid object path '" + std::string(path) +
                    "': character ';' is used to specify an object cycle, which only makes sense when reading."));
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
   // This is arbitrary, but it's useful to avoid pathological cases
   static constexpr int kMaxPathNesting = 1000;

   ///// Factory methods /////

   /// Opens the file for reading
   static std::unique_ptr<RFile> OpenForReading(std::string_view path);

   /// Opens the file for reading/writing, overwriting it if it already exists
   static std::unique_ptr<RFile> Recreate(std::string_view path);

   /// Opens the file for updating
   static std::unique_ptr<RFile> OpenForUpdate(std::string_view path);

   ///// Instance methods /////

   /// Retrieves an object from the file.
   /// `path` should be a string such that `IsValidPath(path) == true`.
   /// If the object is not there returns a null pointer.
   template <typename T>
   std::unique_ptr<T> Get(std::string_view path) const
   {
      if (auto err = ValidatePath(path); !err.empty()) {
         R__LOG_ERROR(Internal::RFileLog()) << "Invalid object path '" << path << "': " << err;
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
   /// `path` should be a string such that `IsValidPath(path) == true`.
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

   /// Returns an iterable over all paths of objects written into this RFile starting at path "rootPath".
   /// The returned paths are always "absolute" paths: they are not relative to `rootPath`.
   /// Keys relative to directories are not returned: only those relative to leaf objects are.
   /// If `rootPath` is the path of a leaf object, only `rootPath` itself will be returned.
   /// This recurses on all the subdirectories of `rootPath`. If you only want the immediate children of `rootPath`,
   /// use GetKeysNonRecursive().
   RFileKeyIterable GetKeys(std::string_view rootPath = "") const
   {
      return RFileKeyIterable(fFile.get(), rootPath, RFileKeyIterable::kRecursive);
   }

   /// Returns an iterable over all paths of objects written into this RFile contained in the directory "rootPath".
   /// The returned paths are always "absolute" paths: they are not relative to `rootPath`.
   /// Keys relative to directories are not returned: only those relative to leaf objects are.
   /// If `rootPath` is the path of a leaf object, only `rootPath` itself will be returned.
   /// This only returns the immediate children of `rootPath`. If you want to recurse into the subdirectories of
   /// `rootPath`, use GetKeys().
   RFileKeyIterable GetKeysNonRecursive(std::string_view rootPath = "") const
   {
      return RFileKeyIterable(fFile.get(), rootPath, RFileKeyIterable::kNone);
   }

   /// Returns an iterable over all paths of objects written into this RFile that match the given regular expression.
   /// The regular expression must match the entire path, so path "a/b/c/d" will be matched by "a.*" or ".*c.*" but
   /// NOT by "a" or "b/c".
   /// Keys relative to directories are not returned: only those relative to leaf objects are.
   RFileKeyIterable GetKeysRegex(const std::regex &pattern) const { return RFileKeyIterable(fFile.get(), pattern); }

   /// \see GetKeysRegex(const std::regex &)
   /// \note This will use the ECMAScript regex grammar. If you want to use a different one, create your own regex
   /// and pass it directly to the other overload of GetKeysRegex().
   RFileKeyIterable GetKeysRegex(std::string_view pattern) const
   {
      std::regex regex{std::string(pattern)};
      return GetKeysRegex(regex);
   }

   /// Retrieves information about the key `path`.
   std::optional<RFileKeyInfo> GetKeyInfo(std::string_view path) const;

   /// Prints the internal structure of this RFile to the given stream.
   void Print(std::ostream &out = std::cout) const;
};

} // namespace Experimental
} // namespace ROOT

#endif
