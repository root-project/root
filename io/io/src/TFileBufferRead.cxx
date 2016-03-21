
#include <string>

#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "TFile.h"
#include "TError.h"

#include "TFileBufferRead.h"


// Download files in 128MB chunks.
#define CHUNK_SIZE 128*1024*1024


TFileBufferRead::TFileBufferRead (TFile *file) :
   fFile(file),
   fSize(-1),
   fCount(0),
   fTotal(-1),
   fFd(-1),
   fInvalid(true)
{
   fSize = fFile->GetSize();
   fTotal = (fSize + CHUNK_SIZE - 1) / CHUNK_SIZE;

   if (tmpfile()) {
      fInvalid = false;
   }
}


bool TFileBufferRead::tmpfile(const std::string &tmpdir) {
  std::string pattern(tmpdir);
  if (pattern.empty()) {
    if (char *p = getenv("TMPDIR")) {
      pattern = p;
    }
  }
  if (pattern.empty()) {
    pattern = "/tmp";
  }
  pattern += "/cmssw-shadow-XXXXXX";

  std::vector<char> temp(pattern.c_str(), pattern.c_str()+pattern.size()+1);
  int fd = mkstemp(&temp[0]);
  if (fd == -1) {
    Warning("TFileBufferRead", "Cannot create temporary file %s: %s (errno=%d)",
                               pattern.c_str(), strerror(errno), errno);
    return false;
  }
  if (-1 == unlink(&temp[0])) {
    Warning("TFileBufferRead", "Cannot unlink temporary file %s: %s (errno=%d)",
                               pattern.c_str(), strerror(errno), errno);
    return false;
  }
  if (-1 == ftruncate(fd, fSize)) {
    Warning("TFileBufferRead", "Cannot resize temporary file %s: %s (errno=%d)",
                               pattern.c_str(), strerror(errno), errno);
    // We ignore this error - it shouldn't be fatal.
  }
  fFd = fd;
  return true;
}


ssize_t TFileBufferRead::pread(char *into, size_t n, off_t pos) {
  if (fInvalid) {
    // Note the order of arguments is different between POSIX read and
    // TFile's ReadBuffer.
    if (!fFile->ReadBuffer(into, pos, n)) {
      if (!errno) {errno = EIO;}
      return -1;
    }
  }

  if (!cache(pos, pos + n)) {
    if (!errno) {errno = EIO;}
    return -1;
  }
  ssize_t retval;
  if (-1 == (retval = ::pread(fFd, into, n, pos))) {
    Warning("TFileBufferRead", "Failed to read from local buffer file: %s (errno=%d)",
                               strerror(errno), errno);
    return -1;
  }
  return retval;
}


bool TFileBufferRead::cache(off_t start, off_t end) {
  start = (start / CHUNK_SIZE) * CHUNK_SIZE;
  end = std::min(end, fSize);

  ssize_t nread = 0;
  size_t index = start / CHUNK_SIZE;

  while (start < end) {
    size_t len = std::min(static_cast<off_t>(fSize - start), static_cast<off_t>(CHUNK_SIZE));
    if (!fPresent[index]) {
      void *window = mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, fFd, start);
      if (window == MAP_FAILED) {
        fInvalid = true;
        Warning("TFileBufferRead", "Unable to map a window of local buffer file: %s (errno=%d)",
                                   strerror(errno), errno);
        fInvalid = true;
        return false;
      }

      
      if (-1 == (nread = fFile->ReadBuffer(static_cast<char *>(window), start, len))) {
        Warning("TFileBufferRead", "Failed to read into the buffer file.");
        fInvalid = true;
        return false;
      }

      munmap(window, len);

      if (static_cast<size_t>(nread) != len)
      {
        Warning("TFileBufferRead", "Unable to cache %lu byte file segment at %ld"
                                   ": got only %ld bytes back.",
                                   len, start, nread);
        fInvalid = true;
        return false;
      }

      fPresent[index] = 1;
      ++fCount;
      if (fCount == fTotal) {
        // TODO: If we provided a more complete wrapper around the source file,
        // we could close it once the file has been fully downloaded to local disk.
      }
    }

    start += len;
    ++index;
  }
}


