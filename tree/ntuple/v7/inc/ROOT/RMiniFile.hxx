/// \file ROOT/RMiniFile.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-12-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RMiniFile
#define ROOT7_RMiniFile

#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RStringView.hxx>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

class TFile;

namespace ROOT {
namespace Experimental {

namespace Detail {
class RRawFile;
}

// clang-format off
/**
\class ROOT::Experimental::RNTuple
\ingroup NTuple
\brief Entry point for an RNTuple in a ROOT file

The class points to the header and footer keys, which in turn have the references to the pages.
Only the RNTuple key will be listed in the list of keys. Like TBaskets, the pages are "invisible" keys.
Byte offset references in the RNTuple header and footer reference directly the data part of page records,
skipping the TFile key part.
*/
// clang-format on
struct RNTuple {
   /// Allows for evolving the struct in future versions
   std::uint32_t fVersion = 0;
   /// Allows for skipping the struct
   std::uint32_t fSize = sizeof(RNTuple);
   /// The file offset of the header excluding the TKey part
   std::uint64_t fSeekHeader = 0;
   /// The size of the compressed ntuple header
   std::uint32_t fNBytesHeader = 0;
   /// The size of the uncompressed ntuple header
   std::uint32_t fLenHeader = 0;
   /// The file offset of the footer excluding the TKey part
   std::uint64_t fSeekFooter = 0;
   /// The size of the compressed ntuple footer
   std::uint32_t fNBytesFooter = 0;
   /// The size of the uncompressed ntuple footer
   std::uint32_t fLenFooter = 0;
   /// Currently unused, reserved for later use
   std::uint64_t fReserved = 0;

   bool operator ==(const RNTuple &other) const {
      return fVersion == other.fVersion &&
         fSize == other.fSize &&
         fSeekHeader == other.fSeekHeader &&
         fNBytesHeader == other.fNBytesHeader &&
         fLenHeader == other.fLenHeader &&
         fSeekFooter == other.fSeekFooter &&
         fNBytesFooter == other.fNBytesFooter &&
         fLenFooter == other.fLenFooter &&
         fReserved == other.fReserved;
   }
};

namespace Internal {

/// Holds status information of an open ROOT file during writing
struct RTFileControlBlock;

// clang-format off
/**
\class ROOT::Experimental::Internal::RMiniFileReader
\ingroup NTuple
\brief Read RNTuple data blocks from a TFile container, provided by a RRawFile

A RRawFile is used for the byte access.  The class implements a minimal subset of TFile, enough to extract
RNTuple data keys.
*/
// clang-format on
class RMiniFileReader {
private:
   /// The raw file used to read byte ranges
   Detail::RRawFile *fFile = nullptr;
   /// Indicates whether the file is a TFile container or an RNTuple raw file
   bool fIsRaw = false;
   /// Used when the file turns out to be a raw file
   RNTuple GetNTupleRaw(std::string_view ntupleName);
   /// Used when the file turns out to be a TFile container
   RNTuple GetNTupleProper(std::string_view ntupleName);

public:
   RMiniFileReader() = default;
   /// Uses the given raw file to read byte ranges
   explicit RMiniFileReader(Detail::RRawFile *rawFile);
   /// Extracts header and footer location for the RNTuple identified by ntupleName
   RNTuple GetNTuple(std::string_view ntupleName);
   /// Reads a given byte range from the file into the provided memory buffer
   void ReadBlob(void *buffer, size_t nbytes, std::uint64_t offset);
};


// clang-format off
/**
\class ROOT::Experimental::Internal::RMiniFileWriter
\ingroup NTuple
\brief Write RNTuple data blocks in a TFile container

The writer can create a new TFile container for an RNTuple or add an RNTuple to an existing TFile.
Creating a single RNTuple in a new TFile container can be done with a C file stream without a TFile class.
Updaing an existing TFile requires a proper TFile object.  Also, writing a remote file requires a proper TFile object.
A stand-alone version of RNTuple can remove the TFile based writer.
*/
// clang-format on
class RMiniFileWriter {
private:
   struct RFileProper {
      TFile *fFile = nullptr;
      /// Low-level writing using a TFile
      void Write(const void *buffer, size_t nbytes, std::int64_t offset);
      /// Writes an RBlob opaque key with the provided buffer as data record and returns the offset of the record
      std::uint64_t WriteKey(const void *buffer, size_t nbytes, size_t len);
      operator bool() const { return fFile; }
   };

   struct RFileSimple {
      /// For the simplest cases, a C file stream can be used for writing
      FILE *fFile = nullptr;
      /// Keeps track of the seek offset
      std::uint64_t fFilePos = 0;

      RFileSimple() = default;
      RFileSimple(const RFileSimple &other) = delete;
      RFileSimple(RFileSimple &&other) = delete;
      RFileSimple &operator =(const RFileSimple &other) = delete;
      RFileSimple &operator =(RFileSimple &&other) = delete;
      ~RFileSimple();

      /// Writes bytes in the open stream, either at fFilePos or at the given offset
      void Write(const void *buffer, size_t nbytes, std::int64_t offset = -1);
      /// Writes a TKey including the data record, given by buffer, into fFile; returns the file offset to the payload.
      /// The payload is already compressed
      std::uint64_t WriteKey(const void *buffer, std::size_t nbytes, std::size_t len, std::int64_t offset = -1,
                             std::uint64_t directoryOffset = 100,
                             const std::string &className = "",
                             const std::string &objectName = "",
                             const std::string &title = "");
      operator bool() const { return fFile; }
   };

   // TODO(jblomer): wrap in an std::variant with C++17
   /// For updating existing files and for storing more than just an RNTuple in the file
   RFileProper fFileProper;
   /// For simple use cases, survives without libRIO dependency
   RFileSimple fFileSimple;
   /// A simple file can either be written as TFile container or as NTuple raw file
   bool fIsRaw = false;
   /// The identifier of the RNTuple; A single writer object can only write a single RNTuple but multiple
   /// writers can concurrently operate on the same file iff they use a proper TFile object for writing.
   std::string fNTupleName;
   /// The file name without parent directory; only required when writing with a C file stream
   std::string fFileName;
   /// Keeps track of TFile control structures, which need to be updated on committing the data set
   std::unique_ptr<ROOT::Experimental::Internal::RTFileControlBlock> fControlBlock;
   /// Header and footer location of the ntuple, written on Commit()
   RNTuple ntuple;

   explicit RMiniFileWriter(std::string_view name);

   /// For a TFile container written by a C file stream, write the records that constitute an empty file
   void WriteTFileSkeleton(int defaultCompression);
   /// For a raw file, which is necessarily wriiten C file stream, write file header
   void WriteRawFileSkeleton(int defaultCompression);

public:
   /// Create or truncate the local file given by path with the new empty RNTuple identified by ntupleName.
   /// Uses a C stream for writing
   static RMiniFileWriter *Recreate(std::string_view ntupleName, std::string_view path, int defaultCompression,
                                    ENTupleContainerFormat containerFormat);
   /// Create or truncate the local or remote file given by path with the new empty RNTuple identified by ntupleName.
   /// Creates a new TFile object for writing and hands over ownership of the object to the user.
   static RMiniFileWriter *Recreate(std::string_view ntupleName, std::string_view path,
                                    std::unique_ptr<TFile> &file);
   /// Add a new RNTuple identified by ntupleName to the existing TFile.
   static RMiniFileWriter *Append(std::string_view ntupleName, TFile &file);

   RMiniFileWriter(const RMiniFileWriter &other) = delete;
   RMiniFileWriter(RMiniFileWriter &&other) = delete;
   RMiniFileWriter &operator =(const RMiniFileWriter &other) = delete;
   RMiniFileWriter &operator =(RMiniFileWriter &&other) = delete;
   ~RMiniFileWriter();

   /// Writes the compressed header and registeres its location; lenHeader is the size of the uncompressed header.
   std::uint64_t WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader);
   /// Writes the compressed footer and registeres its location; lenFooter is the size of the uncompressed footer.
   std::uint64_t WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter);
   /// Writes a new record as an RBlob key into the file
   std::uint64_t WriteBlob(const void *data, size_t nbytes, size_t len);
   /// Writes the RNTuple key to the file so that the header and footer keys can be found
   void Commit();
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
