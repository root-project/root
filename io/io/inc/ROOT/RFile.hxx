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

TODO: more in-depth explanation
*/
class RFile final {
   enum PutFlags {
      kPutAllowOverwrite = 0x1,
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

   /// Opens the file for reading
   static std::unique_ptr<RFile> Open(std::string_view path);

   /// Opens the file for reading/writing, overwriting it if it already exists
   static std::unique_ptr<RFile> Recreate(std::string_view path);

   /// Opens the file for updating
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

   /// Writes all objects to disk with the file structure.
   /// Returns the number of bytes written.
   size_t Flush();

   /// Flushes the RFile if needed and closes it, disallowing any further reading or writing.
   void Close();
};

} // namespace Experimental
} // namespace ROOT

#endif
