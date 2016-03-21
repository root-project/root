#ifndef ROOT_TFileBufferRead
#define ROOT_TFileBufferRead

/**
 * Add a buffering layer in front of a TFile.
 *
 * This divides the file into 128MB chunks; the first time a byte from a chunk
 * is accessed, the whole chunk is downloaded to a temporary file on disk.  A
 * file chunk is only downloaded if accessed.
 *
 * This is *not* meant to be a cache: as soon as the TFile is closed, the buffer
 * is removed.  This means repeated access to the same file will still cause it
 * to be downloaded multiple times.
 *
 * Rather, the focus of this buffer is to hide the effects of network latency.
 *
 * Implementation is based on a similar approach from CMSSW:
 * https://github.com/cms-sw/cmssw/blob/CMSSW_8_0_X/Utilities/StorageFactory/src/LocalCacheFile.cc
 */
class TFile;

class TFileBufferRead {
public:
  TFileBufferRead(TFile *file);
  ~TFileBufferRead();

  /**
   * Positional read of the file, using the underlying buffer file.
   *
   * If the buffer file is invalid, then this reads from the underlying file.
   *
   * On error, -1 is returned and errno is set appropriately.
   */
  ssize_t pread(char *into, size_t n, off_t pos);

private:
  /**
   * Create a temporary file on disk and unlink it; use it as a buffer
   * for the source TFile.
   *
   * - tmpdir: Directory where the temporary file should be created.
   *
   * Returns false on failure.
   */
  bool tmpfile(const std::string &tmpdir="");
  /**
   * Make sure the specified byte range is cached in the local buffer.
   *
   * Returns false on failure.
   */
  bool cache(off_t start, off_t end);

  std::vector<char> fPresent;  // A mask of all the currently present file chunks in the buffer.
  TFile  *fFile;               // A copy of the TFile we are buffering.
  ssize_t fSize;               // Size of the source TFile.
  ssize_t fCount;              // Number of file chunks we have buffered.
  ssize_t fTotal;              // Total number of chunks in source TFile.
  int     fFd;                 // File descriptor pointing to the local file.
  bool    fInvalid;            // Set to true if this buffer is in an invalid state
};

#endif  // ROOT_TFileBufferRead
