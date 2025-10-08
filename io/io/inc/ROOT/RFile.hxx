/// \file ROOT/RFile.hxx
/// \ingroup Base ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-03-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#ifndef ROOT7_RFile
#define ROOT7_RFile

#include <ROOT/RError.hxx>

#include <memory>
#include <string_view>
#include <typeinfo>

class TFile;
class TKey;

namespace ROOT {
namespace Experimental {

class RFile;
struct RFileKeyInfo;

namespace Internal {

ROOT::RLogChannel &RFileLog();

} // namespace Internal

/**
\class ROOT::Experimental::RFile
\ingroup RFile
\brief An interface to read from, or write to, a ROOT file, as well as performing other common operations.

## When and why should you use RFile

RFile is a modern and minimalistic interface to ROOT files, both local and remote, that can be used instead of TFile
when you only need basic Put/Get operations and don't need the more advanced TFile/TDirectory functionalities.
It provides:
- a simple interface that makes it easy to do things right and hard to do things wrong;
- more robustness and better error reporting for those operations;
- clearer ownership semantics expressed through the type system.

RFile doesn't cover the entirety of use cases covered by TFile/TDirectory/TDirectoryFile and is not
a 1:1 replacement for them.  It is meant to simplify the most common use cases by following newer standard C++
practices.

## Ownership model

RFile handles ownership via smart pointers, typically std::unique_ptr.

When getting an object from the file (via RFile::Get) you get back a unique copy of the object. Calling `Get` on the
same object twice produces two independent clones of the object. The ownership over that object is solely on the caller
and not shared with the RFile. Therefore, the object will remain valid after closing or destroying the RFile that
generated it. This also means that any modification done to the object are **not** reflected to the file automatically:
to update the object in the file you need to write it again (via RFile::Overwrite).

RFile::Put and RFile::Overwrite are the way to write objects to the file. Both methods take a const reference to the
object to write and don't change the ownership of the object in any way. Calling Put or Overwrite doesn't guarantee that
the object is immediately written to the underlying storage: to ensure that, you need to call RFile::Flush (or close the
file).

## Directories

Even though there is no equivalent of TDirectory in the RFile API, directories are still an existing concept in RFile
(since they are a concept in the ROOT binary format). However they are for now only interacted with indirectly, via the
use of filesystem-like string-based paths. If you Put an object in an RFile under the path "path/to/object", "object"
will be stored under directory "to" which is in turn stored under directory "path". This hierarchy is encoded in the
ROOT file itself and it can provide some optimization and/or conveniencies when querying objects.

For the most part, it is convenient to think about RFile in terms of a key-value storage where string-based paths are
used to refer to arbitrary objects. However, given the hierarchical nature of ROOT files, certain filesystem-like
properties are applied to paths, for ease of use: the '/' character is treated specially as the directory separator;
multiple '/' in a row are collapsed into one (since RFile doesn't allow directories with empty names).

At the moment, RFile doesn't allow getting directories via Get, nor writing ones via Put (this may change in the
future).

## Sample usage
Opening an RFile (for writing) and writing an object to it:
~~~{.cpp}
auto rfile = ROOT::RFile::Recreate("my_file.root");
auto myObj = TH1D("h", "h", 10, 0, 1);
rfile->Put(myObj.GetName(), myObj);
~~~

Opening an RFile (for reading) and reading an object from it:
~~~{.cpp}
auto rfile = ROOT::RFile::Open("my_file.root");
auto myObj = file->Get<TH1D>("h");
~~~
*/
class RFile final {
   /// Flags used in PutInternal()
   enum PutFlags {
      /// When encountering an object at the specified path, overwrite it with the new one instead of erroring out.
      kPutAllowOverwrite = 0x1,
      /// When overwriting an object, preserve the existing one and create a new cycle, rather than removing it.
      kPutOverwriteKeepCycle = 0x2,
   };

   std::unique_ptr<TFile> fFile;

   // Outlined to avoid including TFile.h
   explicit RFile(std::unique_ptr<TFile> file);

   /// Gets object `path` from the file and returns an **owning** pointer to it.
   /// The caller should immediately wrap it into a unique_ptr of the type described by `type`.
   [[nodiscard]] void *GetUntyped(std::string_view path, const std::type_info &type) const;

   /// Writes `obj` to file, without taking its ownership.
   void PutUntyped(std::string_view path, const std::type_info &type, const void *obj, std::uint32_t flags);

   /// \see Put
   template <typename T>
   void PutInternal(std::string_view path, const T &obj, std::uint32_t flags)
   {
      PutUntyped(path, typeid(T), &obj, flags);
   }

   /// Given `path`, returns the TKey corresponding to the object at that path (assuming the path is fully split, i.e.
   /// "a/b/c" always means "object 'c' inside directory 'b' inside directory 'a'").
   /// IMPORTANT: `path` must have been validated/normalized via ValidateAndNormalizePath() (see RFile.cxx).
   TKey *GetTKey(std::string_view path) const;

public:
   // This is arbitrary, but it's useful to avoid pathological cases
   static constexpr int kMaxPathNesting = 1000;

   ///// Factory methods /////

   /// Opens the file for reading. `path` may be a regular file path or a remote URL.
   /// \throw ROOT::RException if the file at `path` could not be opened.
   static std::unique_ptr<RFile> Open(std::string_view path);

   /// Opens the file for reading/writing, overwriting it if it already exists.
   /// \throw ROOT::RException if a file could not be created at `path` (e.g. if the specified
   /// directory tree does not exist).
   static std::unique_ptr<RFile> Recreate(std::string_view path);

   /// Opens the file for updating, creating a new one if it doesn't exist.
   /// \throw ROOT::RException if the file at `path` could neither be read nor created
   /// (e.g. if the specified directory tree does not exist).
   static std::unique_ptr<RFile> Update(std::string_view path);

   ///// Instance methods /////

   // Outlined to avoid including TFile.h
   ~RFile();

   /// Retrieves an object from the file.
   /// `path` should be a string such that `IsValidPath(path) == true`, otherwise an exception will be thrown.
   /// See \ref ValidateAndNormalizePath() for info about valid path names.
   /// If the object is not there returns a null pointer.
   template <typename T>
   std::unique_ptr<T> Get(std::string_view path) const
   {
      void *obj = GetUntyped(path, typeid(T));
      return std::unique_ptr<T>(static_cast<T *>(obj));
   }

   /// Puts an object into the file.
   /// The application retains ownership of the object.
   /// `path` should be a string such that `IsValidPath(path) == true`, otherwise an exception will be thrown.
   /// See \ref ValidateAndNormalizePath() for info about valid path names.
   ///
   /// Throws a RException if `path` already identifies a valid object or directory.
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
   /// Throws a RException if `path` is already the path of a directory.
   /// Throws a RException if the file was opened in read-only mode.
   template <typename T>
   void Overwrite(std::string_view path, const T &obj, bool backupPrevious = true)
   {
      std::uint32_t flags = kPutAllowOverwrite;
      flags |= backupPrevious * kPutOverwriteKeepCycle;
      PutInternal(path, obj, flags);
   }

   /// Writes all objects and the file structure to disk.
   /// Returns the number of bytes written.
   size_t Flush();

   /// Flushes the RFile if needed and closes it, disallowing any further reading or writing.
   void Close();
};

} // namespace Experimental
} // namespace ROOT

#endif
