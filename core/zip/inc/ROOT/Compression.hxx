/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
 * The parent class of various compression engine types.
 *
 * This is meant to be used only internally within ROOT and does not come
 * with ABI/API stability guarantees.
 */

#include <unistd.h>

namespace ROOT {
namespace Internal {

class CompressionEngine {

public:
   CompressionEngine(int level) : fLevel(level) {}
   virtual ~CompressionEngine();

   /**
    * ROOT's block framing format adds a 9-byte header to all
    * full compression blocks.
    */
   static constexpr unsigned kROOTHeaderSize = 9;

   /**
    * The file-format version written by this compression engine.
    */
   virtual char Version() const = 0;

   /**
    * Set current target of the compression engine.
    */
   void SetTarget(void *buffer, size_t size) {fBuffer = fCur = static_cast<char*>(buffer); fSize = fCap = size;}

   /**
    * Stream as much data as is deemed reasonable by the engine.
    *
    * The engine may decide to not ingest any or all data if it
    * wants to allow more data to accumulate before attempting action.
    *
    * Returns the number of bytes from the buffer ingested, or -1 on error.
    */
   virtual ssize_t Stream(const void * /*buffer*/, size_t /*size*/) {return -1;}

   /**
    * Stream all data in the buffer.
    *
    * Forcibly stream all the data available in the buffer, regardless of
    * how badly this affects the compression performance.
    */
   virtual ssize_t StreamFull(const void *buffer, size_t size) = 0;

   /**
    * Write out sufficient compression state that a subsequent Inflate
    * can start from the output, without any of the prior data available.
    *
    * Returns true if successful.
    */
   virtual bool RestartMarker() {return false;}

   /**
    * Reset all internal state in the engine
    */
   virtual void Reset() {}

   /**
    * Train the compression engine on a particular dataset.
    *
    * After the engine is trained, any subsequent readers must utilize
    * the exact same training set.
    *
    * That means, one should call GetTraining to retrieve the subsequent data,
    * then call SetTraining on the reader.
    *
    * Returns true on success.
    */
   virtual bool Train(const void * /*buffer*/, size_t /*size*/) {return false;}

   /**
    * Return a serialized representation of the training data.
    *
    * - buffer[output] is a reference to a memory pointer.
    * - size[output] is the size of the returned buffer.
    * - Returns true on success.
    *
    * The returned memory is valid until the CompressionEngine is destroyed
    * or Reset() is called.
    */
   virtual bool GetTraining(void *& /*buffer*/, size_t & /*size*/) {return false;}

   /**
    * Set the training of this engine.
    *
    * Take a preexisting serialized training information (as returned by
    * GetTraining) and provide it to this compression engine.
    *
    * Returns true on success.
    */
   virtual bool SetTraining(const void * /*training*/, size_t /*size*/) {return false;}

protected:

   /**
    * Write the ROOT-specific header at a given location.
    */
   static void WriteROOTHeader(void *buffer, const char alg[2], char version, int deflate_size, int inflate_size);

   char   *fBuffer{nullptr}; // The "destination" buffer for compressed data.
   char   *fCur{nullptr};    // Current offset within the destination buffer.
   size_t  fSize{0};         // The total size of the destination buffer.
   size_t  fCap{0};          // The remaining capacity between fCur and the end of fBuffer.
   int     fLevel{6};        // Compression level
};


class DecompressionEngine {

public:
   DecompressionEngine() {}
   virtual ~DecompressionEngine();

   /**
    * Check to see if this decompression engine is compatible with the file format
    *
    * Returns true if there is compatibility.
    */
   virtual bool VersionCompat(char version) const = 0;

   /**
    * Set current target of the compression engine.
    */
   void SetTarget(void *buffer, size_t size) {fBuffer = fCur = static_cast<char*>(buffer); fSize = fCap = size;}

   /**
    * Stream as much data as is deemed reasonable by the engine.
    *
    * The engine may decide to not ingest any or all data if it
    * wants to allow more data to accumulate before attempting action.
    *
    * Returns the number of bytes from the buffer ingested, or -1 on error.
    */
   virtual ssize_t Stream(const void * /*buffer*/, size_t /*size*/) {return -1;}

   /**
    * Stream all data in the buffer.
    *
    * Forcibly stream all the data available in the buffer, regardless of
    * how badly this affects the compression performance.
    */
   virtual ssize_t StreamFull(const void *buffer, size_t size) = 0;

   /**
    * Reset all internal state in the engine
    */
   virtual void Reset() {}

   /**
    * Set the training of this engine.
    *
    * Take a preexisting serialized training information (as returned by
    * GetTraining) and provide it to this compression engine.
    *
    * Returns true on success.
    */
   virtual bool SetTraining(const void * /*training*/, size_t /*size*/) {return false;}

protected:
   char   *fBuffer{nullptr};  // The "destination" buffer for compressed data.
   char   *fCur{nullptr};     // Current offset within the destination buffer.
   size_t  fSize{0};          // The total size of the destination buffer.
   size_t  fCap{0};           // The remaining capacity between fCur and the end of fBuffer.
};

}
}
