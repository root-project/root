# Code Architecture

This document lists all classes, describes their responsibilities, and details how they fit together.

## Core Classes

These classes are involved when the user calls `Fill`.
The list is ordered "bottom-up" in terms of functionality.

### `RRegularAxis`, `RVariableBinAxis`, `RCategoricalAxis`

These classes implement a concrete axis type.
Their main task is to compute the linearized index for a single `Fill` argument:
```c++
RLinearizedIndex ComputeLinearizedIndex(double x);
```
The `bool` is used to indicate if the return value is valid.
For example, the argument may be outside the axis with the underflow and overflow bins disabled.
`RLinearizedIndex` is a simple struct with a `std::size_t index` and `bool valid`.
It is chosen over `std::optional` because it unifies the return value construction:
If outside the axis, the validity is just determined by the member property `fEnableFlowBins`.

### `Internal::RAxes`

This class is responsible for managing a histogram's axis configuration.
It stores the axis objects as a `std::vector` of `std::variant`s.
Objects of this class are used internally and not exposed to the user.
Relevant functionality is available through user-facing classes such as `RHistEngine`.

The main function is
```c++
template <typename... A>
RLinearizedIndex ComputeGlobalIndex(const std::tuple<A...> &args) const;
```
that dispatches arguments to the individual `ComputeLinearizedIndex` and combines the results.
If any of the linearized indices is invalid, so will be the combination.

### `RHistEngine`

This class combines an `RAxes` object and storage of bin contents in a `std::vector`.
During `Fill`, it calls `RAxes::ComputeLinearizedIndex` and then updates the bin content.
In contrast to `RHist`, this class supports direct concurrent filling via `FillAtomic`.

### `RHistStats`

Manages the (global) histogram statistics, such as the number of entries.
It also keeps statistics of the unbinned values for each dimension.

### `RHist`

This class combines `RHistEngine`, with its axes and bin contents, and `RHistStats`.
During `Fill`, it delegates to `RHistEngine::Fill` but also updates the histogram statistics.

## Classes for Weighted Filling

### `RDoubleBinWithError`

A special bin content type that also accumulates the sum of weights squared.
It can be used as a template argument to `RHistEngine` and `RHist`.

### `RWeight`

A wrapper `struct` for a single `double` value, used for weighted filling to distinguish its type.
Objects of this type are passed by value.

## Auxiliary Classes

### `RBinIndex`

A single bin index, which is just an integer for normal bins.
`Underflow()` and `Overflow()` are special values and not ordered with respect to others.
Objects of this type are passed by value; most notably to `GetBinContent` and `SetBinContent`.

### `RBinIndexRange`

A range of `RBinIndex` from `begin` (inclusive) to `end` (exclusive).
The class exposes an iterator interface that can be used in range-based loops.
