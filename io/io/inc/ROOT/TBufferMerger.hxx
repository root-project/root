// @(#)root/io:$Id$
// Author: Philippe Canal, Witold Pokorski, and Guilherme Amadio

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferMerger
#define ROOT_TBufferMerger

#include "TMemFile.h"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

class TBufferFile;

namespace ROOT {
namespace Experimental {

class TBufferMergerFile;

/**
 * \class TBufferMerger TBufferMerger.hxx
 * \ingroup IO
 *
 * TBufferMerger is a class to facilitate writing data in
 * parallel from multiple threads, while writing to a single
 * output file. Its purpose is similar to TParallelMergingFile,
 * but instead of using processes that connect to a network
 * socket, TBufferMerger uses threads that each write to a
 * TBufferMergerFile, which in turn push data into a queue
 * managed by the TBufferMerger.
 */

class TBufferMerger {
public:
   /** Constructor
    * @param name Output file name
    * @param option Output file creation options
    * @param compression Output file compression level
    */
   TBufferMerger(const char *name, Option_t *option = "RECREATE", Int_t compress = 1);

   /** Destructor */
   virtual ~TBufferMerger();

   /** Returns a TBufferMergerFile to which data can be written.
    *  At the end, all TBufferMergerFiles get merged into the output file.
    *  The user is responsible to "cd" into the file to associate objects
    *  such as histograms or trees to it.
    *
    *  After the creation of this file, the user must reset the kMustCleanup
    *  bit on any objects attached to it and take care of their deletion, as
    *  there is a possibility that a race condition will happen that causes
    *  a crash if ROOT manages these objects.
    */
   std::shared_ptr<TBufferMergerFile> GetFile();

   /** Returns the number of buffers currently in the queue. */
   size_t GetQueueSize() const;

   /** Register a user callback function to be called after a buffer has been
    *  removed from the merging queue and finished being processed. This
    *  function can be useful to allow asynchronous launching of new tasks to
    *  push more data into the queue once its size satisfies user requirements.
    */
   void RegisterCallback(const std::function<void(void)> &f);

   /** By default, TBufferMerger will call TFileMerger::PartialMerge() for each
    *  buffer pushed onto its merge queue. This function lets the user change
    *  this behaviour by telling TBufferMerger to accumulate at least @param size
    *  bytes in memory before performing a partial merge and flushing to disk.
    *  This can be useful to avoid an excessive amount of work to happen in the
    *  output thread, as the number of TTree headers (which require compression)
    *  written to disk can be reduced.
    */
   void SetAutoSave(size_t size);

   friend class TBufferMergerFile;

private:
   /** TBufferMerger has no default constructor */
   TBufferMerger();

   /** TBufferMerger has no copy constructor */
   TBufferMerger(const TBufferMerger &);

   /** TBufferMerger has no copy operator */
   TBufferMerger &operator=(const TBufferMerger &);

   void Push(TBufferFile *buffer);
   void WriteOutputFile();

   const std::string fName;
   const std::string fOption;
   const Int_t fCompress;
   size_t fAutoSave;                                             //< AutoSave only every fAutoSave bytes
   std::mutex fQueueMutex;                                       //< Mutex used to lock fQueue
   std::condition_variable fDataAvailable;                       //< Condition variable used to wait for data
   std::queue<TBufferFile *> fQueue;                             //< Queue to which data is pushed and merged
   std::unique_ptr<std::thread> fMergingThread;                  //< Worker thread that writes to disk
   std::vector<std::weak_ptr<TBufferMergerFile>> fAttachedFiles; //< Attached files
   std::function<void(void)> fCallback;                          //< Callback for when data is removed from queue

   ClassDef(TBufferMerger, 0);
};

/**
 * \class TBufferMerger TBufferMerger.hxx
 * \ingroup IO
 *
 * A TBufferMergerFile is similar to a TMemFile, but when data
 * is written to it, it is appended to the TBufferMerger queue.
 * The TBufferMerger merges all data into the output file on disk.
 */

class TBufferMergerFile : public TMemFile {
private:
   TBufferMerger &fMerger; //< TBufferMerger this file is attached to

   /** Constructor. Can only be called by TBufferMerger.
    * @param m Merger this file is attached to. */
   TBufferMergerFile(TBufferMerger &m);

   /** TBufferMergerFile has no default constructor. */
   TBufferMergerFile();

   /** TBufferMergerFile has no copy constructor. */
   TBufferMergerFile(const TBufferMergerFile &);

   /** TBufferMergerFile has no copy operator */
   TBufferMergerFile &operator=(const TBufferMergerFile &);

   friend class TBufferMerger;

public:
   /** Destructor */
   ~TBufferMergerFile();

   using TMemFile::Write;

   /** Write data into a TBufferFile and append it to TBufferMerger.
    * @param name Name
    * @param opt  Options
    * @param bufsize Buffer size
    * This function must be called before the TBufferMergerFile gets destroyed,
    * or no data is appended to the TBufferMerger.
    */
   virtual Int_t Write(const char *name = nullptr, Int_t opt = 0, Int_t bufsize = 0) override;

   ClassDefOverride(TBufferMergerFile, 0);
};

} // namespace Experimental
} // namespace ROOT

#endif
