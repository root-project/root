/// \file ROOT/RMiniFile.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-12-22

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RMiniFile
#define ROOT_RMiniFile

#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RSpan.hxx>
#include <Compression.h>
#include <string_view>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

class TDirectory;
class TFileMergeInfo;
class TVirtualStreamerInfo;

namespace ROOT {

namespace Internal {
class RRawFile;
}

class RNTupleWriteOptions;

namespace Internal {
/// Holds status information of an open ROOT file during writing
struct RTFileControlBlock;

// clang-format off
/**
\class ROOT::Internal::RMiniFileReader
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
   /// If `fMaxKeySize > 0` and ReadBuffer attempts to read `nbytes > maxKeySize`, it will assume the
   /// blob being read is chunked and read all the chunks into the buffer. This is symmetrical to
   /// what happens in `RNTupleFileWriter::WriteBlob()`.
   std::uint64_t fMaxKeySize = 0;

   /// Used when the file container turns out to be a bare file
   RResult<RNTuple> GetNTupleBare(std::string_view ntupleName);
   /// Used when the file turns out to be a TFile container. The ntuplePath variable is either the ntuple name
   /// or an ntuple name preceded by a directory (`myNtuple` or `foo/bar/myNtuple` or `/foo/bar/myNtuple`)
   RResult<RNTuple> GetNTupleProper(std::string_view ntuplePath);
   /// Loads an RNTuple anchor from a TFile at the given file offset (unzipping it if necessary).
   RResult<RNTuple>
   GetNTupleProperAtOffset(std::uint64_t payloadOffset, std::uint64_t compSize, std::uint64_t uncompLen);

   /// Searches for a key with the given name and type in the key index of the directory starting at offsetDir.
   /// The offset points to the start of the TDirectory DATA section, without the key and without the name and title
   /// of the TFile record (the root directory).
   /// Return 0 if the key was not found. Otherwise returns the offset of found key.
   std::uint64_t SearchInDirectory(std::uint64_t &offsetDir, std::string_view keyName, std::string_view typeName);

public:
   RMiniFileReader() = default;
   /// Uses the given raw file to read byte ranges
   explicit RMiniFileReader(ROOT::Internal::RRawFile *rawFile);
   /// Extracts header and footer location for the RNTuple identified by ntupleName
   RResult<RNTuple> GetNTuple(std::string_view ntupleName);
   /// Reads a given byte range from the file into the provided memory buffer.
   /// If `nbytes > fMaxKeySize` it will perform chunked read from multiple blobs,
   /// whose addresses are listed at the end of the first chunk.
   void ReadBuffer(void *buffer, size_t nbytes, std::uint64_t offset);
   /// Attempts to load the streamer info from the file.
   void LoadStreamerInfo();

   std::uint64_t GetMaxKeySize() const { return fMaxKeySize; }
   /// If the reader is not used to retrieve the anchor, we need to set the max key size manually
   void SetMaxKeySize(std::uint64_t maxKeySize) { fMaxKeySize = maxKeySize; }
};

// clang-format off
/**
\class ROOT::Internal::RNTupleFileWriter
\ingroup NTuple
\brief Write RNTuple data blocks in a TFile or a bare file container

The writer can create a new TFile container for an RNTuple or add an RNTuple to an existing TFile.
Creating a single RNTuple in a new TFile container can be done with a C file stream without a TFile class.
Updating an existing TFile requires a proper TFile object.  Also, writing a remote file requires a proper TFile object.
A stand-alone version of RNTuple can remove the TFile based writer.
*/
// clang-format on
class RNTupleFileWriter {
public:
   /// The key length of a blob. It is always a big key (version > 1000) with class name RBlob.
   static constexpr std::size_t kBlobKeyLen = 42;

private:
   struct RFileProper {
      /// A sub directory in fFile or nullptr if the data is stored in the root directory of the file
      TDirectory *fDirectory = nullptr;
      /// Low-level writing using a TFile
      void Write(const void *buffer, size_t nbytes, std::int64_t offset);
      /// Reserves an RBlob opaque key as data record and returns the offset of the record. If keyBuffer is specified,
      /// it must be written *before* the returned offset. (Note that the array type is purely documentation, the
      /// argument is actually just a pointer.)
      std::uint64_t ReserveBlobKey(size_t nbytes, size_t len, unsigned char keyBuffer[kBlobKeyLen] = nullptr);
      operator bool() const { return fDirectory; }
   };

   struct RFileSimple {
      /// Direct I/O requires that all buffers and write lengths are aligned. It seems 512 byte alignment is the minimum
      /// for Direct I/O to work, but further testing showed that it results in worse performance than 4kB.
      static constexpr int kBlockAlign = 4096;
      /// During commit, WriteTFileKeysList() updates fNBytesKeys and fSeekKeys of the RTFFile located at
      /// fSeekFileRecord. Given that the TFile key starts at offset 100 and the file name, which is written twice,
      /// is shorter than 255 characters, we should need at most ~600 bytes. However, the header also needs to be
      /// aligned to kBlockAlign...
      static constexpr std::size_t kHeaderBlockSize = 4096;

      // fHeaderBlock and fBlock are raw pointers because we have to manually call operator new and delete.
      unsigned char *fHeaderBlock = nullptr;
      std::size_t fBlockSize = 0;
      std::uint64_t fBlockOffset = 0;
      unsigned char *fBlock = nullptr;

      /// For the simplest cases, a C file stream can be used for writing
      FILE *fFile = nullptr;
      /// Whether the C file stream has been opened with Direct I/O, introducing alignment requirements.
      bool fDirectIO = false;
      /// Keeps track of the seek offset
      std::uint64_t fFilePos = 0;
      /// Keeps track of the next key offset
      std::uint64_t fKeyOffset = 0;
      /// Keeps track of TFile control structures, which need to be updated on committing the data set
      std::unique_ptr<ROOT::Internal::RTFileControlBlock> fControlBlock;

      RFileSimple();
      RFileSimple(const RFileSimple &other) = delete;
      RFileSimple(RFileSimple &&other) = delete;
      RFileSimple &operator=(const RFileSimple &other) = delete;
      RFileSimple &operator=(RFileSimple &&other) = delete;
      ~RFileSimple();

      void AllocateBuffers(std::size_t bufferSize);
      void Flush();

      /// Writes bytes in the open stream, either at fFilePos or at the given offset
      void Write(const void *buffer, size_t nbytes, std::int64_t offset = -1);
      /// Writes a TKey including the data record, given by buffer, into fFile; returns the file offset to the payload.
      /// The payload is already compressed
      std::uint64_t WriteKey(const void *buffer, std::size_t nbytes, std::size_t len, std::int64_t offset = -1,
                             std::uint64_t directoryOffset = 100, const std::string &className = "",
                             const std::string &objectName = "", const std::string &title = "");
      /// Reserves an RBlob opaque key as data record and returns the offset of the record. If keyBuffer is specified,
      /// it must be written *before* the returned offset. (Note that the array type is purely documentation, the
      /// argument is actually just a pointer.)
      std::uint64_t ReserveBlobKey(std::size_t nbytes, std::size_t len, unsigned char keyBuffer[kBlobKeyLen] = nullptr);
      operator bool() const { return fFile; }
   };

   /// RFileSimple: for simple use cases, survives without libRIO dependency
   /// RFileProper: for updating existing files and for storing more than just an RNTuple in the file
   std::variant<RFileSimple, RFileProper> fFile;
   /// A simple file can either be written as TFile container or as NTuple bare file
   bool fIsBare = false;
   /// The identifier of the RNTuple; A single writer object can only write a single RNTuple but multiple
   /// writers can operate on the same file if (and only if) they use a proper TFile object for writing.
   std::string fNTupleName;
   /// The file name without parent directory; only required when writing with a C file stream
   std::string fFileName;
   /// Header and footer location of the ntuple, written on Commit()
   RNTuple fNTupleAnchor;
   /// Set of streamer info records that should be written to the file.
   /// The RNTuple class description is always present.
   ROOT::Internal::RNTupleSerializer::StreamerInfoMap_t fStreamerInfoMap;

   explicit RNTupleFileWriter(std::string_view name, std::uint64_t maxKeySize);

   /// For a TFile container written by a C file stream, write the header and TFile object
   void WriteTFileSkeleton(int defaultCompression);
   /// The only key that will be visible in file->ls()
   /// Returns the size on disk of the anchor object
   std::uint64_t WriteTFileNTupleKey(int compression);
   /// Write the TList with the RNTuple key
   void WriteTFileKeysList(std::uint64_t anchorSize);
   /// Write the compressed streamer info record with the description of the RNTuple class
   void WriteTFileStreamerInfo(int compression);
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
   static std::unique_ptr<RNTupleFileWriter> Recreate(std::string_view ntupleName, std::string_view path,
                                                      EContainerFormat containerFormat,
                                                      const ROOT::RNTupleWriteOptions &options);
   /// The directory parameter can also be a TFile object (TFile inherits from TDirectory).
   static std::unique_ptr<RNTupleFileWriter>
   Append(std::string_view ntupleName, TDirectory &fileOrDirectory, std::uint64_t maxKeySize);

   RNTupleFileWriter(const RNTupleFileWriter &other) = delete;
   RNTupleFileWriter(RNTupleFileWriter &&other) = delete;
   RNTupleFileWriter &operator=(const RNTupleFileWriter &other) = delete;
   RNTupleFileWriter &operator=(RNTupleFileWriter &&other) = delete;
   ~RNTupleFileWriter();

   /// Seek a simple writer to offset. Note that previous data is not flushed immediately, but only by the next write
   /// (if necessary).
   void Seek(std::uint64_t offset);

   /// Writes the compressed header and registeres its location; lenHeader is the size of the uncompressed header.
   std::uint64_t WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader);
   /// Writes the compressed footer and registeres its location; lenFooter is the size of the uncompressed footer.
   std::uint64_t WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter);
   /// Writes a new record as an RBlob key into the file
   std::uint64_t WriteBlob(const void *data, size_t nbytes, size_t len);

   /// Prepares buffer for a new record as an RBlob key at offset.
   /// (Note that the array type is purely documentation, the argument is actually just a pointer.)
   static void PrepareBlobKey(std::int64_t offset, size_t nbytes, size_t len, unsigned char buffer[kBlobKeyLen]);

   /// Reserves a new record as an RBlob key in the file. If keyBuffer is specified, it must be written *before* the
   /// returned offset. (Note that the array type is purely documentation, the argument is actually just a pointer.)
   std::uint64_t ReserveBlob(size_t nbytes, size_t len, unsigned char keyBuffer[kBlobKeyLen] = nullptr);
   /// Write into a reserved record; the caller is responsible for making sure that the written byte range is in the
   /// previously reserved key.
   void WriteIntoReservedBlob(const void *buffer, size_t nbytes, std::int64_t offset);
   /// Ensures that the streamer info records passed as argument are written to the file
   void UpdateStreamerInfos(const ROOT::Internal::RNTupleSerializer::StreamerInfoMap_t &streamerInfos);
   /// Writes the RNTuple key to the file so that the header and footer keys can be found
   void Commit(int compression = RCompressionSetting::EDefaults::kUseGeneralPurpose);
};

} // namespace Internal
} // namespace ROOT

#endif
