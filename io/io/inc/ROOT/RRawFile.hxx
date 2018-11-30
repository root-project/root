// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRawFile
#define ROOT_RRawFile

#include <ROOT/RStringView.hxx>

#include <cstddef>
#include <cstdint>
#include <string>

namespace ROOT {
namespace Detail {

class RRawFile;

/**
 * \class RRawFile RRawFile.hxx
 * \ingroup IO
 *
 * The RRawFile provides read-only access to local and remote files. Data can be read either byte-wise or line-wise.
 * The RRawFile base class provides line-wise access and buffering for byte-wise access. Derived classes provide the
 * low-level read operations, e.g. from a local file system or from a web server. The RRawFile is used for non-root
 * RDataSource implementations.
 *
 * Files are addressed by URL consisting of a transport protocol part and a location, like file:///path/to/data
 * If the transport protocol part and the :// separator are missing, the default protocol is local file.
 *
 * RRawFiles are not copyable.
 *
 * TODO(jblomer): Implement derived class for Davix low-level I/O
 */
class RRawFile {
public:
   /// Derived classes do not necessarily need to provide file size information but they can return "not known" instead
   static constexpr std::uint64_t kUnknownFileSize = std::uint64_t(-1);
   /// kAuto detects the line break from the first line, kSystem picks the system's default
   enum class ELineBreaks { kAuto, kSystem, kUnix, kWindows };
   /// On construction, an ROptions parameter can customized the RRawFile behavior
   struct ROptions {
      ELineBreaks fLineBreak;
      /**
       * Read at least fBlockSize bytes at a time. A value of zero turns off I/O buffering. A negative value indicates
       * that the protocol-dependent default block size should be used.
       */
      int fBlockSize;
      ROptions() : fLineBreak(ELineBreaks::kAuto), fBlockSize(-1) { }
   };

private:
   /// Don't set to a different value without adapting ReadAt()
   static constexpr unsigned int kNumBlockBuffers = 2;
   struct RBlockBuffer {
      /// Where in the open file does fBuffer start
      std::uint64_t fBufferOffset;
      /// The number of currently buffered bytes in fBuffer
      size_t fBufferSize;
      /// Points into the I/O buffer with data from the file, not owned.
      unsigned char *fBuffer;

      RBlockBuffer() : fBufferOffset(0), fBufferSize(0), fBuffer(nullptr) { }
      RBlockBuffer(const RBlockBuffer&) = delete;
      RBlockBuffer& operator=(const RBlockBuffer&) = delete;
      ~RBlockBuffer() = default;

      /// Tries to copy up to nbytes starting at offset from fBuffer into buffer.
      size_t Map(void *buffer, size_t nbytes, std::uint64_t offset);
   };
   /// To be used modulo kNumBlockBuffers, points to the last used block buffer in fBlockBuffers
   unsigned int fBlockBufferIdx;
   /// An active buffer and a shadow buffer, which supports "jumping back" to a previously used location in the file
   RBlockBuffer fBlockBuffers[kNumBlockBuffers];
   /// Memory block containing the block buffers consequtively
   unsigned char *fBufferSpace;
   /// The cached file size
   std::uint64_t fFileSize;

protected:
   std::string fUrl;
   ROptions fOptions;
   /// The current position in the file, which can be changed by Seek, Read, and Readln
   std::uint64_t fFilePos;

   /// Derived classes should implement low-level reading without buffering
   virtual size_t DoReadAt(void *buffer, size_t nbytes, std::uint64_t offset) = 0;
   /// Derived classes should return the file size or kUnknownFileSize
   virtual std::uint64_t DoGetSize() = 0;

public:
   RRawFile(const std::string &url, ROptions options);
   RRawFile(const RRawFile&) = delete;
   RRawFile& operator=(const RRawFile&) = delete;
   virtual ~RRawFile();

   /// Factory method that returns a suitable concrete implementation according to the transport in the url
   static RRawFile* Create(std::string_view url, ROptions options = ROptions());
   /// Returns only the file location, e.g. "server/file" for http://server/file
   static std::string GetLocation(std::string_view url);
   /// Returns only the transport protocol, e.g. "http" for http://server/file
   static std::string GetTransport(std::string_view url);

   /**
    * Buffered read from a random position. Returns the actual number of bytes read.
    * Short reads indicate the end of the file
    */
   size_t ReadAt(void *buffer, size_t nbytes, std::uint64_t offset);
   /// Read from fFilePos offset. Returns the actual number of bytes read.
   size_t Read(void *buffer, size_t nbytes);
   /// Change the cursor fFilePos
   void Seek(std::uint64_t offset);
   /// Returns the size of the file
   std::uint64_t GetSize();

   /// Read the next line starting from the current value of fFilePos. Returns false if the end of the file is reached.
   bool Readln(std::string& line);
};

} // namespace Detail
} // namespace ROOT

#endif
