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

#include "TFileMerger.h"
#include "TMemFile.h"

#include <functional>
#include <memory>
#include <mutex>
#include <queue>

namespace ROOT {

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
    * @param compress Output file compression level
    */
   TBufferMerger(const char *name, Option_t *option = "RECREATE",
                 Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault);

   /** Constructor
    * @param output Output \c TFile
    */
   TBufferMerger(std::unique_ptr<TFile> output);

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

   /** Returns the number of bytes currently buffered (i.e. in the queue). */
   size_t GetBuffered() const
   {
      return fBuffered;
   }

   /** Returns the current value of the auto save setting in bytes (default = 0). */
   size_t GetAutoSave() const;

   /** Returns the current merge options. */
   const char* GetMergeOptions();

   /** By default, TBufferMerger will call TFileMerger::PartialMerge() for each
    *  buffer pushed onto its merge queue. This function lets the user change
    *  this behaviour by telling TBufferMerger to accumulate at least size
    *  bytes in memory before performing a partial merge and flushing to disk.
    *  This can be useful to avoid an excessive amount of work to happen in the
    *  output thread, as the number of TTree headers (which require compression)
    *  written to disk can be reduced.
    */
   void SetAutoSave(size_t size);

   /** Sets the merge options. SetMergeOptions("fast") will disable
    * recompression of input data into the output if they have different
    * compression settings.
    * @param options TFileMerger/TFileMergeInfo merge options
    */
   void SetMergeOptions(const TString& options);

   /** Indicates that any TTree objects in the file should be skipped
    * and thus that steps that are specific to TTree can be skipped */
   void SetNotrees(Bool_t notrees=kFALSE)
   {
      fMerger.SetNotrees(notrees);
   }

   /** Returns whether the file has been marked as not containing any TTree objects
    * and thus that steps that are specific to TTree can be skipped */
   Bool_t GetNotrees() const
   {
      return fMerger.GetNotrees();
   }

   /** Indicates that the temporary keys (corresponding to the object held by the directories
    *  of the TMemFile) should be compressed or not.   Those object are stored in the TMemFile
    * (and thus possibly compressed) when a thread push its data forward (by calling
    * TBufferMergerFile::Write) and the queue is being processed by another.
    * Once the TMemFile is picked (by any thread to be merged), *after* taking the
    * TBufferMerger::fMergeMutex, those object are read back (and thus possibly uncompressed)
    * and then used by merging.
    * In order word, the compression of those objects/keys is only usefull to reduce the size
    * in memory (of the TMemFile) and does not affect (at all) the compression factor of the end
    * result.
    */
   void SetCompressTemporaryKeys(Bool_t request_compression = true)
   {
      fCompressTemporaryKeys = request_compression;
   }

   /** Returns whether to compressed the TKey in the TMemFile for the object held by
    * the TDirectories.  See TBufferMerger::SetCompressTemporaryKeys for more details.
    */
   Bool_t GetCompressTemporaryKeys() const
   {
      return fCompressTemporaryKeys;
   }

   friend class TBufferMergerFile;

private:
   /** TBufferMerger has no default constructor */
   TBufferMerger();

   /** TBufferMerger has no copy constructor */
   TBufferMerger(const TBufferMerger &);

   /** TBufferMerger has no copy operator */
   TBufferMerger &operator=(const TBufferMerger &);

   void Init(std::unique_ptr<TFile>);

   void MergeImpl();

   void Merge();
   void Push(TBufferFile *buffer);
   bool TryMerge(TBufferMergerFile *memfile);

   bool fCompressTemporaryKeys{false};                           //< Enable compression of the TKeys in the TMemFile (save memory at the expense of time, end result is unchanged)
   size_t fAutoSave{0};                                          //< AutoSave only every fAutoSave bytes
   std::atomic<size_t> fBuffered{0};                             //< Number of bytes currently buffered
   TFileMerger fMerger{false, false};                            //< TFileMerger used to merge all buffers
   std::mutex fMergeMutex;                                       //< Mutex used to lock fMerger
   mutable std::mutex fQueueMutex;                               //< Mutex used to lock fQueue
   std::queue<TBufferFile *> fQueue;                             //< Queue to which data is pushed and merged
   std::vector<std::weak_ptr<TBufferMergerFile>> fAttachedFiles; //< Attached files
};

/**
 * \class TBufferMergerFile TBufferMerger.hxx
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

} // namespace ROOT

#endif
