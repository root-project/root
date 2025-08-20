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

#include <string_view>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ROOT {
namespace Internal {

/**
 * \class RRawFile RRawFile.hxx
 * \ingroup IO
 *
 * The RRawFile provides read-only access to local and remote files. Data can be read either byte-wise or line-wise.
 * The RRawFile base class provides line-wise access and buffering for byte-wise access. Derived classes provide the
 * low-level read operations, e.g. from a local file system or from a web server. The RRawFile is used for non-ROOT
 * RDataSource implementations and for RNTuple.
 *
 * Files are addressed by URL consisting of a transport protocol part and a location, like file:///path/to/data
 * If the transport protocol part and the :// separator are missing, the default protocol is local file. Files are
 * opened when required (on reading, getting file size) and closed on object destruction.
 *
 * RRawFiles manage system resources and are therefore made non-copyable. They can be explicitly cloned though.
 *
 * RRawFile objects are conditionally thread safe. See the user manual for further details:
 * https://root.cern/manual/thread_safety/
 */
class RRawFile {
public:
   /// kAuto detects the line break from the first line, kSystem picks the system's default
   enum class ELineBreaks { kAuto, kSystem, kUnix, kWindows };

   /// On construction, an ROptions parameter can customize the RRawFile behavior
   struct ROptions {
      static constexpr size_t kUseDefaultBlockSize = std::size_t(-1); ///< Use protocol-dependent default block size

      ELineBreaks fLineBreak = ELineBreaks::kAuto;
      /// Read at least fBlockSize bytes at a time. A value of zero turns off I/O buffering.
      size_t fBlockSize = kUseDefaultBlockSize;
      // Define an empty constructor to work around a bug in Clang: https://github.com/llvm/llvm-project/issues/36032
      ROptions() {}
   };

   /// Used for vector reads from multiple offsets into multiple buffers. This is unlike readv(), which scatters a
   /// single byte range from disk into multiple buffers.
   struct RIOVec {
      /// The destination for reading
      void *fBuffer = nullptr;
      /// The file offset
      std::uint64_t fOffset = 0;
      /// The number of desired bytes
      std::size_t fSize = 0;
      /// The number of actually read bytes, set by ReadV()
      std::size_t fOutBytes = 0;
   };

   /// Implementations may enforce limits on the use of vector reads. These limits can depend on the server or
   /// the specific file opened and can be queried per RRawFile object through GetReadVLimits().
   /// Note that due to such limits, a vector read with a single request can behave differently from a Read() call.
   struct RIOVecLimits {
      /// Maximum number of elements in a ReadV request vector
      std::size_t fMaxReqs = static_cast<std::size_t>(-1);
      /// Maximum size in bytes of any single request in the request vector
      std::size_t fMaxSingleSize = static_cast<std::size_t>(-1);
      /// Maximum size in bytes of the sum of requests in the vector
      std::uint64_t fMaxTotalSize = static_cast<std::uint64_t>(-1);

      bool HasReqsLimit() const { return fMaxReqs != static_cast<std::size_t>(-1); }
      bool HasSizeLimit() const
      {
         return fMaxSingleSize != static_cast<std::size_t>(-1) || fMaxTotalSize != static_cast<std::uint64_t>(-1);
      }
   };

private:
   /// Don't change without adapting ReadAt()
   static constexpr unsigned int kNumBlockBuffers = 2;
   struct RBlockBuffer {
      /// Where in the open file does fBuffer start
      std::uint64_t fBufferOffset = 0;
      /// The number of currently buffered bytes in fBuffer
      size_t fBufferSize = 0;
      /// Points into the I/O buffer with data from the file, not owned.
      unsigned char *fBuffer = nullptr;

      RBlockBuffer() = default;
      RBlockBuffer(const RBlockBuffer &) = delete;
      RBlockBuffer &operator=(const RBlockBuffer &) = delete;
      ~RBlockBuffer() = default;

      /// Tries to copy up to nbytes starting at offset from fBuffer into buffer.  Returns number of bytes copied.
      size_t CopyTo(void *buffer, size_t nbytes, std::uint64_t offset);
   };
   /// To be used modulo kNumBlockBuffers, points to the last used block buffer in fBlockBuffers
   unsigned int fBlockBufferIdx = 0;
   /// An active buffer and a shadow buffer, which supports "jumping back" to a previously used location in the file
   RBlockBuffer fBlockBuffers[kNumBlockBuffers];
   /// Memory block containing the block buffers consecutively
   std::unique_ptr<unsigned char[]> fBufferSpace;
   /// Used as a marker that the file size was not yet queried
   static constexpr std::uint64_t kUnknownFileSize = std::uint64_t(-1);
   /// The cached file size
   std::uint64_t fFileSize = kUnknownFileSize;
   /// Files are opened lazily and only when required; the open state is kept by this flag
   bool fIsOpen = false;
   /// Runtime switch to decide if reads are buffered or directly sent to ReadAtImpl()
   bool fIsBuffering = true;

protected:
   std::string fUrl;
   ROptions fOptions;
   /// The current position in the file, which can be changed by Seek, Read, and Readln
   std::uint64_t fFilePos = 0;

   /// OpenImpl() is called at most once and before any call to either DoReadAt or DoGetSize. If fOptions.fBlocksize
   /// is negative, derived classes are responsible to set a sensible value. After a call to OpenImpl(),
   /// fOptions.fBlocksize must be larger or equal to zero.
   virtual void OpenImpl() = 0;
   /// Derived classes should implement low-level reading without buffering. Short reads indicate the end of the file,
   /// therefore derived classes should return nbytes bytes if available.
   virtual size_t ReadAtImpl(void *buffer, size_t nbytes, std::uint64_t offset) = 0;
   /// Derived classes should return the file size
   virtual std::uint64_t GetSizeImpl() = 0;

   /// By default implemented as a loop of ReadAt calls but can be overwritten, e.g. XRootD or DAVIX implementations
   virtual void ReadVImpl(RIOVec *ioVec, unsigned int nReq);

   /// Open the file if not already open. Otherwise noop.
   void EnsureOpen();

public:
   RRawFile(std::string_view url, ROptions options);
   RRawFile(const RRawFile &) = delete;
   RRawFile &operator=(const RRawFile &) = delete;
   virtual ~RRawFile() = default;

   /// Create a new RawFile that accesses the same resource.  The file pointer is reset to zero.
   virtual std::unique_ptr<RRawFile> Clone() const = 0;

   /// Factory method that returns a suitable concrete implementation according to the transport in the url
   static std::unique_ptr<RRawFile> Create(std::string_view url, ROptions options = ROptions());
   /// Returns only the file location, e.g. "server/file" for http://server/file
   static std::string GetLocation(std::string_view url);
   /// Returns only the transport protocol in lower case, e.g. "http" for HTTP://server/file
   static std::string GetTransport(std::string_view url);

   /// Buffered read from a random position. Returns the actual number of bytes read.
   /// Short reads indicate the end of the file
   size_t ReadAt(void *buffer, size_t nbytes, std::uint64_t offset);
   /// Read from fFilePos offset. Returns the actual number of bytes read.
   size_t Read(void *buffer, size_t nbytes);
   /// Change the cursor fFilePos
   void Seek(std::uint64_t offset);
   /// Returns the offset for the next Read/Readln call
   std::uint64_t GetFilePos() const { return fFilePos; }
   /// Returns the size of the file
   std::uint64_t GetSize();
   /// Returns the url of the file
   std::string GetUrl() const;

   /// Opens the file if necessary and calls ReadVImpl
   void ReadV(RIOVec *ioVec, unsigned int nReq);
   /// Returns the limits regarding the ioVec input to ReadV for this specific file; may open the file as a side-effect.
   virtual RIOVecLimits GetReadVLimits() { return RIOVecLimits(); }

   /// Turn off buffered reads; all scalar read requests go directly to the implementation. Buffering can be turned
   /// back on.
   void SetBuffering(bool value);
   bool IsBuffering() const { return fIsBuffering; }

   /// Read the next line starting from the current value of fFilePos. Returns false if the end of the file is reached.
   bool Readln(std::string &line);

   /// Once opened, the file stay open until destruction of the RRawFile object
   bool IsOpen() const { return fIsOpen; }
}; // class RRawFile

} // namespace Internal
} // namespace ROOT

#endif
