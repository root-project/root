# Concurrent Filling

A key feature is the ability to fill histograms from multiple threads in parallel.
It allows to save memory by allocating fewer histograms, instead of duplicating them per thread.
While global index computation is embarrassingly parallel, access to the bin contents requires synchronization.

## Terminology

Here, we discuss histograms as a *concurrent data structure* (as opposed to a *sequential* one).
This means the implementation offers methods to access and modify bin contents from multiple threads.
An application may be *parallel* and rely on the provided thread-safety.

*Atomic instructions* are one way to implement scalable concurrent data structures.
An algorithm for concurrent data structures may be *non-blocking* and *lock-free*.
They refer to the guarantee that suspending one thread does not block other threads.
More precise definitions and discussions are available in computer science literature.

## Bin Content Types for Concurrent Filling

To enable concurrent filling, it would of course be possible to require an appropriate bin content type.
However, `std::atomic` is problematic because it is neither copyable nor movable.
This would complicate many histogram operations, such as clearing or cloning.

An alternative would be a custom type `Atomic<T>` that wraps `std::atomic` and implements a consistent set of operations.
This approach still has two disadvantages:
1. All access to bin contents uses atomic instructions, which is not always needed.
   For example, small histograms per thread are independent and should be used as sequential data structures.
   Similarly, accessing bin contents after concurrent filling most often does not require synchronization.
   For theses cases, "normal" instructions can result in better performance.
2. As the bin content type is a template argument, `RHist<T>` and `RHist<Atomic<T>>` are different types.
   This will be visible in (de)serialization and also for further processing, where code needs to handle these types.

Instead, we use a single bin content type for storage and only a subset of methods is thread-safe.
This is possible because on most architectures, atomic instructions only require natural alignment.
With C++20, `std::atomic_ref` offers a portable way to apply atomic operations to a referenced object.
Until this becomes the required C++ standard, we implement a minimal version ourselves with compiler builtins.

## Concurrent Filling of `RHistEngine`

`RHistEngine` stores bin contents in a `std::vector` with a fixed size after construction.
In this case, the `FillAtomic` method uses atomic instructions to modify the bin contents.
For large histograms and reasonable data, contention on individual bins is expected to be low.

## Concurrent Filling of `RHist`

On the other hand, updates of the (global) histogram statistics (`RHistStats`) can easily lead to contention.
For this reason, `RHist` does **not** offer a `FillAtomic` method because it cannot be implemented efficiently.
Instead, the user has to create a `RHistConcurrentFiller` and (potentially many) `RHistFillContext`s.
These will work together to accumulate the (global) histogram statistics during concurrent filling.
