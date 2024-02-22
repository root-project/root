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

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleAnchor.hxx>
#include <string_view>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

class TCollection;
class TFile;
class TFileMergeInfo;

namespace ROOT {

namespace Internal {
class RRawFile;
}

namespace Experimental {

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
   ROOT::Internal::RRawFile *fRawFile = nullptr;
   /// Indicates whether the file is a TFile container or an RNTuple bare file
   bool fIsBare = false;
   /// Used when the file container turns out to be a bare file
   RResult<RNTuple> GetNTupleBare(std::string_view ntupleName);
   /// Used when the file turns out to be a TFile container
   RResult<RNTuple> GetNTupleProper(std::string_view ntupleName);

   RNTuple CreateAnchor(std::uint16_t versionEpoch, std::uint16_t versionMajor, std::uint16_t versionMinor,
                        std::uint16_t versionPatch, std::uint64_t seekHeader, std::uint64_t nbytesHeader,
                        std::uint64_t lenHeader, std::uint64_t seekFooter, std::uint64_t nbytesFooter,
                        std::uint64_t lenFooter, std::uint64_t checksum);

public:
   RMiniFileReader() = default;
   /// Uses the given raw file to read byte ranges
   explicit RMiniFileReader(ROOT::Internal::RRawFile *rawFile);
   /// Extracts header and footer location for the RNTuple identified by ntupleName
   RResult<RNTuple> GetNTuple(std::string_view ntupleName);
   /// Reads a given byte range from the file into the provided memory buffer
   void ReadBuffer(void *buffer, size_t nbytes, std::uint64_t offset);
};


// clang-format off
/**
\class ROOT::Experimental::Internal::RNTupleFileWriter
\ingroup NTuple
\brief Write RNTuple data blocks in a TFile or a bare file container

The writer can create a new TFile container for an RNTuple or add an RNTuple to an existing TFile.
Creating a single RNTuple in a new TFile container can be done with a C file stream without a TFile class.
Updating an existing TFile requires a proper TFile object.  Also, writing a remote file requires a proper TFile object.
A stand-alone version of RNTuple can remove the TFile based writer.
*/
// clang-format on
class RNTupleFileWriter {
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
      /// Keeps track of the next key offset
      std::uint64_t fKeyOffset = 0;
      /// Keeps track of TFile control structures, which need to be updated on committing the data set
      std::unique_ptr<ROOT::Experimental::Internal::RTFileControlBlock> fControlBlock;

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
   /// A simple file can either be written as TFile container or as NTuple bare file
   bool fIsBare = false;
   /// The identifier of the RNTuple; A single writer object can only write a single RNTuple but multiple
   /// writers can operate on the same file if (and only if) they use a proper TFile object for writing.
   std::string fNTupleName;
   /// The file name without parent directory; only required when writing with a C file stream
   std::string fFileName;
   /// Header and footer location of the ntuple, written on Commit()
   RNTuple fNTupleAnchor;

   explicit RNTupleFileWriter(std::string_view name);

   /// For a TFile container written by a C file stream, write the header and TFile object
   void WriteTFileSkeleton(int defaultCompression);
   /// The only key that will be visible in file->ls()
   void WriteTFileNTupleKey();
   /// Write the TList with the RNTuple key
   void WriteTFileKeysList();
   /// Write the compressed streamer info record with the description of the RNTuple class
   void WriteTFileStreamerInfo();
   /// Last record in the file
   void WriteTFileFreeList();
   /// For a bare file, which is necessarily written by a C file stream, write file header
   void WriteBareFileSkeleton(int defaultCompression);

public:
   /// For testing purposes, RNTuple data can be written into a bare file container instead of a ROOT file
   enum class EContainerFormat {
      kTFile, // ROOT TFile
      kBare,  // A thin envelope supporting a single RNTuple only
   };

   /// Create or truncate the local file given by path with the new empty RNTuple identified by ntupleName.
   /// Uses a C stream for writing
   static RNTupleFileWriter *Recreate(std::string_view ntupleName, std::string_view path, int defaultCompression,
                                      EContainerFormat containerFormat);
   /// Add a new RNTuple identified by ntupleName to the existing TFile.
   static RNTupleFileWriter *Append(std::string_view ntupleName, TFile &file);

   RNTupleFileWriter(const RNTupleFileWriter &other) = delete;
   RNTupleFileWriter(RNTupleFileWriter &&other) = delete;
   RNTupleFileWriter &operator =(const RNTupleFileWriter &other) = delete;
   RNTupleFileWriter &operator =(RNTupleFileWriter &&other) = delete;
   ~RNTupleFileWriter();

   /// Writes the compressed header and registeres its location; lenHeader is the size of the uncompressed header.
   std::uint64_t WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader);
   /// Writes the compressed footer and registeres its location; lenFooter is the size of the uncompressed footer.
   std::uint64_t WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter);
   /// Writes a new record as an RBlob key into the file
   std::uint64_t WriteBlob(const void *data, size_t nbytes, size_t len);
   /// Reserves a new record as an RBlob key in the file.
   std::uint64_t ReserveBlob(size_t nbytes, size_t len);
   /// Write into a reserved record; the caller is responsible for making sure that the written byte range is in the
   /// previously reserved key.
   void WriteIntoReservedBlob(const void *buffer, size_t nbytes, std::int64_t offset);
   /// Writes the RNTuple key to the file so that the header and footer keys can be found
   void Commit();
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
