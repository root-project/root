# Design and Implementation

This document describes key design decisions and implementation choices.

## Templating

Classes are only templated if required for data members, in particular the bin content type `T`.
We use member function templates to accept variable number of arguments (see also below).
Classes are **not** templated to improve performance, in particular not on the axis type(s).
This avoids an explosion of types and simplifies serialization.
Instead axis objects are run-time choices and stored in a `std::variant`.
With a careful design, this still results in excellent performance.

## Performance Optimizations

If required, it would be possible to template performance-critical functions on the axis types.
This was shown beneficial in microbenchmarks for one-dimensional histograms.
However, it will not be implemented until shown useful in a real-world application.
In virtually all cases, filling a (one-dimensional) histogram is negligible compared to reading, decompressing, and processing of data.

The same applies for other optimizations, such as caching the pointer to the axis object stored in the `std::variant`.
Such optimizations should only be implemented with a careful motivation for real-world applications.

## Functions with Variable Number of Arguments

Many member functions have two overloads: one accepting a function parameter pack and one accepting a `std::tuple` or `std::array`.

### Arguments with Different Types

Functions that take arguments with different types expect a `std::tuple`.
An example is `template <typename A...> void Fill(const std::tuple<A...> &args)`.

For user-convenience, a variadic function template forwards to the `std::tuple` overload:
```cpp
template <typename... A> void Fill(const A &...args) {
   Fill(std::forward_as_tuple(args...));
}
```
This will forward the arguments as references, so no copy-constructors are called (that could potentially be expensive).

### Arguments with Same Type

In this case, the function has a `std::size_t N` template argument and accepts a `std::array`.
An example is `template <std::size_t N> const T &GetBinContent(const std::array<RBinIndex, N> &args)`

For user-convenience, a variadic function template forwards to the `std::array` overload:
```cpp
template <typename... A> const T &GetBinContent(const A &...args) {
   std::array<RBinIndex, sizeof...(A)> a{args...};
   return GetBinContent(a);
}
```
This will copy the arguments, which is fine in this case because `RBinIndex` is small (see below).

### Special Arguments

Special arguments are passed last.
Examples include
```cpp
template <typename... A> void Fill(const std::tuple<A...> &args, RWeight w);
template <std::size_t N> void SetBinContent(const std::array<RBinIndex, N> &args, const T &content);
```
The same works for the variadic function templates that will check the type of the last argument.

For profiles, we accept the value with a template type as well to allow automatic conversion to `double`, for example from `int`.

## Miscellaneous

The implementation uses standard [C++17](https://en.cppreference.com/w/cpp/17.html):
 * No backports from later C++ versions, such as `std::span`, and
 * No ROOT types, to make sure the histogram package can be compiled standalone.

Small objects are passed by value instead of by reference (`RBinIndex`, `RWeight`).
