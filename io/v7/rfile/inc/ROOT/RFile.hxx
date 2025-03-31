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

#include <TFile.h>

#include <string_view>
#include <memory>

namespace ROOT {
namespace Experimental {

class RFileKeyIterable final {
   TFile *fFile;
   std::string fPattern;

public:
   class RIterator {
      friend class RFileKeyIterable;

      // NOTE: TFile not present because we already skip it in the ctor (and there is only 1 such key per file)
      static constexpr const char *fInternalKeyClasses[] = {"KeysList", "StreamerInfo", "FreeSegments"};

      ROOT::Detail::TKeyMapIterable::TIterator fIter;
      std::string_view fPattern;

      static bool IsInternalKeyClass(const char *className)
      {
         if (strlen(className) == 0)
            return true;
         for (const char *k : fInternalKeyClasses)
            if (strcmp(className, k) == 0)
               return true;
         return false;
      }

      bool MatchesPattern(std::string_view name) const {
         // TODO: more complex pattern handling (glob/regex)
         if (fPattern.empty())
            return true;

         if (name.size() < fPattern.size())
            return false;
                 
          return name.compare(0, fPattern.size(), fPattern) == 0;
      }

      void Advance() {
         // We only want to return keys that refer to user objects, not internal ones, therefore we skip
         // all keys that have internal class names.
         do {
            ++fIter;
         } while (fIter.operator->() &&
                  (IsInternalKeyClass(fIter->fClassName.c_str()) || !MatchesPattern(fIter->fKeyName)));
      }

      RIterator(ROOT::Detail::TKeyMapIterable::TIterator iter, std::string_view pattern)
         : fIter(iter), fPattern(pattern)
      {
         // Advance the iterator to skip the first key, which is always the TFile key.
         // Then skip until the first matching key.
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
      reference operator*() { return fIter->fKeyName; }
      pointer operator->() { return &fIter->fKeyName; }
      bool operator!=(const iterator &rh) const { return !(*this == rh); }
      bool operator==(const iterator &rh) const { return fIter == rh.fIter; }
   };

   explicit RFileKeyIterable(TFile *file, std::string_view pattern) : fFile(file), fPattern(pattern) {}

   RIterator begin() const { return {fFile->WalkTKeys().begin(), fPattern}; }
   RIterator end() const { return {fFile->WalkTKeys().end(), fPattern}; }
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

   ///// Instance methods /////

   // Retrieves an object from the file.
   // If the object is not there, returns an invalid ref.
   template <typename T>
   std::unique_ptr<T> Get(std::string_view path) const
   {
      std::string pathStr(path);
      const TClass *cls = TClass::GetClass(typeid(T));
      void *obj = GetUntyped(pathStr.c_str(), cls);
      return std::unique_ptr<T>(static_cast<T *>(obj));
   }

   // Puts an object into the file.
   // Throws a RException if the file was opened in read-only mode.
   template <typename T>
   void Put(std::string_view path, T &obj)
   {
      std::string pathStr(path);
      const TClass *cls = TClass::GetClass(typeid(T));
      PutUntyped(pathStr.c_str(), cls, &obj);
   }

   RFileKeyIterable GetKeys(std::string_view pattern = "") const { return RFileKeyIterable(fFile.get(), pattern); }
};

} // namespace Experimental
} // namespace ROOT

#endif
