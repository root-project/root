# Parallel scheduling in RDataFrame

RDataFrame allows to process input data in parallel in two ways: with implicit multi-threading and with distributed multiprocessing. This document explains the strategies for the implementations of the two modes.

[//]: # (This should be a good way to write comments in markdown according to https://stackoverflow.com/questions/4823468/comments-in-markdown)
[//]: # (TODO: The document only features a description of the multithreading strategy for RNTuple, add the rest.)

## Multi-threading

### RNTuple

RNTuple data reading is handled in `RNTupleDS`, an `RDataSource`-derived class. The reading is triggered by the
computation graph being run, and nodes requesting data from the data source. The computation graph is started and
managed by an `RLoopManager`, which delegates to the `RNTupleDS` for decisions related to scheduling of different MT
tasks, reading data and giving it back to the `RLoopManager` to run the computation graph until done. This is a
description of the high-level steps involved:

1. When the user triggers the computation graph, `RLoopManager::RunDataSourceMT` prepares the computation graph nodes
   and calls `RDataSource::ProcessMT`. This method is overridden by the RNTuple data source. This method, as well as all
   other methods described from now on, are private to the `RNTupleDS` class.
2. In `RNTupleDS::ProcessMT`:
    a. A `TThreadExecutor` is initialised with the number of threads given by `ROOT::GetThreadPoolSize()`.
    b. An `RSlotStack` is initialised with the same number of slots. A slot is required to apply the computation graph
       to a range of entries.
    c. Optionally, if the user has specified a global entry range to restrict processing to, a list of global entry
       offsets per file is populated by getting the number of entries in each file upfront.
    d. One file-level task per input file is scheduled in a `tbb::task_group`. The task group is executed inside the
       arena managed by `TThreadExecutor`.
3. The `RNTupleDS` constructor opens the first input file, reads its schema and leaves the corresponding page source in
   the staging area. `RNTupleDS::Initialize` starts a background staging thread. The staging thread opens files in
   order, calls `RPageSource::LoadStructure` and puts the page sources into the staging area. The sliding window of
   files that are staged ahead of the next file to be claimed is bounded by the number of slots. `fNextFileToClaim` is
   an atomic counter advanced by the file-level tasks; the staging thread only stages files while
   `fNextFileToStage < min(nFiles, fNextFileToClaim + nSlots)`. A condition variable wakes the staging thread when a
   file is claimed and wakes file-level tasks when their page source is ready.
4. A file-level task atomically claims the next input file. If it claims the last file, it sets the shared tail-mode
   flag. The task then takes the staged page source, calls `RPageSource::Attach` and
   `RPageSource::LoadAllPageLists`, and computes the list of clusters to process. With a global entry range, the
   cluster boundaries are trimmed to the intersection of the file and the requested range; clusters that do not
   contain any requested entry are dropped.
5. The file-level task acquires one processing slot and iterates over its clusters in file order. Before starting, it
   restricts the page source entry range to the whole file (or to the trimmed range when a global entry range is
   active), not to the current cluster. This is important: it lets the page source prefetch the next clusters while the
   computation graph processes the current one. For every cluster, the task inserts the active entry range for its
   slot and calls `RLoopManager::RNTupleThreadTask`. The entry range passed to the loop manager is expressed in global
   entry coordinates by adding the file offset.
6. Tail scheduling starts once every file-level task has been claimed. This is the point at which no new file can
   start and some workers may otherwise be left without work while the remaining clusters of a large file are still
   being processed. A file-level task that observes tail mode schedules one task for each of its remaining clusters in
   the same `tbb::task_group` and then processes its current cluster. This happens at most once per file-level task.
7. A tail cluster task first acquires a processing slot and only then clones the parent page source. It restricts the
   clone to the single cluster it is going to process and calls `RLoopManager::RNTupleThreadTask`, exactly like the
   file-level task does for a cluster. Cloning after acquiring the slot is what keeps the number of concurrently open
   page sources bounded.

The staging window, the file-level tasks and the tail cluster tasks together bound the number of open page sources.
At any point there are at most `nSlots` page sources in the staging area and at most `nSlots` page sources owned by
file-level tasks or their tail cluster tasks. Tail cluster tasks are only created by file-level tasks that have
already claimed a file, and there are at most `nSlots` such file-level tasks active at once. Consequently, the data
source keeps at most `2 * nSlots` page sources open. This property holds also during the transition to tail mode and
for `nSlots == 1`.

This strategy deliberately avoids the eager, recursive splitting of the previous implementation. Splitting a file
into ranges as soon as the computation starts destroys the page source's ability to prefetch across cluster
boundaries and can produce tasks that are too small to amortise their setup cost. With the current strategy, a file
is read sequentially for as long as there is other file-level work to claim, and cluster-level parallelism is only
introduced for the tail of the event loop.
