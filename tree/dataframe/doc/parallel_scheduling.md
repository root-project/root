# Parallel scheduling in RDataFrame

RDataFrame allows to process input data in parallel in two ways: with implicit multi-threading and with distributed multiprocessing. This document explains the strategies for the implementations of the two modes.

[//]: # (This should be a good way to write comments in markdown according to https://stackoverflow.com/questions/4823468/comments-in-markdown)
[//]: # (TODO: The document only features a description of the multithreading strategy for RNTuple, add the rest.)

## Multi-threading

### RNTuple

[//]: # (TODO: The text is meant to document the current status of the MT strategy, knowing that in the future we may adopt a different, more optimised strategy.)

RNTuple data reading is handled in `RNTupleDS`, an `RDataSource`-derived class. The reading is triggered by the
computation graph being run, and nodes requesting data from the data source. The computation graph is started and
managed by an `RLoopManager`, which delegates to the `RNTupleDS` for decisions related to scheduling of different MT
tasks, reading data and giving it back to the `RLoopManager` to run the computation graph until done.  This is a
description of the high-level steps involved:

1. When the user triggers the computation graph, `RLoopManager::RunDataSourceMT` prepares the computation graph nodes
   and calls `RDataSource::ProcessMT`. This method is overridden by the RNTuple data source. This method, as well as all
   other methods methods described from now on, are private to the `RNTupleDS` class.
2. In `RNTupleDS::ProcessMT`:
    a. A `TThreadExecutor` is initialised with the number of threads given by `ROOT::GetThreadPoolSize()`.
    b. An `RSlotStack` is initialised with the same number of slots.
    c. Optionally, if the user has specified a global entry range to restrict processing to, a list of global entry
       offsets per file is populated by getting the number of entries in each file upfront.
    d. `TThreadExecutor::Foreach` is called, spawning one task per input file of the (primary) dataset.
3. The file-based tasks specialise depending on option 2c, respectively:
    a. `RNTupleDS::ScheduleTasksForFile` will schedule a task reading all entries from the file.
    b. `RNTupleDS::ScheduleTasksForFileWithGlobalOffset` will schedule a task with an adjusted list of clusters. The
        user-provided global entry range is matched against the global offsets computed by 2c, and a new list of
        clusters is generated to restrict processing of this page source according to the match. Consequently, some
        tasks may be stopped immediately if the page source boundaries do not overlap with the global entry range.
4. Both file-based task implementations described above eventually call `RNTupleDS::ScheduleTaskForPageSource`. This
   method redirects to the TBB parallel_for overload that takes two arguments, one respecting the
   [Range](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/named_requirements/algorithms/range) and the other respecting the [Body](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/named_requirements/algorithms/par_for_body) requirements. Note that the parallel_for call is hidden by
   `ROOT::Internal::IMTUtils::ParallelFor`, so that the actual TBB implementation is not leaked.
    a. The Range argument is of type `RSplittableRangeOfClusterBoundaries`, a struct representing a range of entries (N
       consecutive clusters) tied to a specific page source to be processed. `ScheduleTaskForPageSource` calls
       parallel_for with an instance of `RSplittableRangeOfClusterBoundaries` that includes the full list of valid clusters of
       the corresponding page source, as defined by step 3.
    b. Immediately after, TBB calls `RSplittableRangeOfClusterBoundaries::SplitWork`, which will split the
       cluster range in roughly two equal parts, recursively until there are only non-splittable tasks based on the
       condition given by `RSplittableRangeOfClusterBoundaries::IsDivisible`. This is the core of the strategy: we cannot know a
       priori if one of the input files will be much larger than the others, or if one thread finishes earlier than
       others. For this reason, we allow tasks to split themselves, so that threads that finish a file-based task
       earlier can pick up one of the smaller cluster-based tasks coming as a result of
       `RSplittableRangeOfClusterBoundaries::SplitWork` of another file-based task. Also, the splitting done with this strategy
       ensures all the ranges of entries are non-overlapping. `RSplittableRangeOfClusterBoundaries::IsDivisible` currently
       dictates that a task can split itself only if it has three or more clusters to process. The reason for requiring
       at least three clusters is that the page source will prefetch the next cluster while processing the current one.
       While TBB allows to provide a proportional splitting, we decide to go for the balanced basic splitting and have
       a task only split into two roughly equal parts, for lack of a motivation to do differently.
    c. The Body argument calls `RNTupleDS::ParallelForBody`. This function is responsible to apply the computation graph
       to a page source restricted to the given range of clusters of the `RSplittableRangeOfClusterBoundaries` argument by
       calling `RLoopManager::RNTupleThreadTask`.

One cause for concern in MT runs is the risk of having too many files open at the same time, if the data source keeps
spawning tasks that open a page source. The strategy described above also limits the number of concurrently open files
through the mechanism of the RDataFrame processing slot. Here is a description of how that works, using concepts
described above with a different perspective:

1. In the beginning of the computation graph run, the pool spawns one task per file. The pool has `N` available threads,
   which start and pick one of the file-based tasks.
2. The file-based task is a function. At the beginning of this function, a page source is created, so that the valid
   list of clusters of step 3 above can be computed.
3. Within `ScheduleTaskForPageSource`, parallel_for is called with the Range and Body arguments. There is no creation
   of new page sources or cloning of existing ones yet.
4. Once `ParallelForBody` starts running, it will request a slot from the `ROOT::Internal::RSlotStackRAII`
   argument. This effectively locks that function to run and apply the whole computation graph on the given range before
   ending. Only as many `ParallelForBody` functions can run as the number of available slots.
5. Only after a slot has been taken, then `ParallelForBody` clones the input page source and restricts the cloned page
   source to the assigned cluster range of the `RSplittableRangeOfClusterBoundaries` argument, with `RPageSource::SetEntryRange`.
   a. In order to save one clone, one of the `ParallelForBody` will just take the input (shared) page source without
      cloning it. At the moment, this is chosen to be the function execution corresponding to a
      `RSplittableRangeOfClusterBoundaries` that processes the first cluster of the page source.

In summary, creation of a page source happens in one of the two methods `ScheduleTasksForFile` or
`ScheduleTasksForFileWithGlobalOffset`. A page source created at the beginning of a file-based task can also be cloned
within a `ParallelForBody`. There are only `N` available threads, each of which could have launched a file-based task,
then end up running a `ParallelForBody`. Thus, there can be at most `2*N` open page sources at the same time.
