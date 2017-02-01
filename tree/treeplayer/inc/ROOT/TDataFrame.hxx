// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
  \defgroup dataframe Data Frame
The ROOT Data Frame allows to analyse data stored in TTrees with a high level interface, exploiting all the resources available on the machine in a transparent way for the user.

In a nutshell:
~~~ {.cpp}
ROOT::EnableImplicitMT(); // Tell ROOT you want to go parallel
ROOT::TDataFrame d("myTree", file); // Interface to TTree and TChain
auto myHisto = d.Histo("Branch_A"); // This happens in parallel!
myHisto->Draw();
~~~

Calculations are expressed in terms of a type-safe *functional chain of actions and transformations*, `TDataFrame` takes care of their execution. The implementation automatically puts in place several low level optimisations such as multi-thread parallelisation and caching.
The namespace containing the TDataFrame is ROOT::Experimental. This signals the fact that the interfaces may evolve in time.


## Table of Contents
- [Introduction](#introduction)
- [Crash course](#crash-course)
- [More features](#more-features)
- [Transformations](#transformations)
- [Actions](#actions)
- [Parallel execution](#parallel-execution)

## <a name="introduction"></a>Introduction
A pipeline of operations is described to be performed on the data, the framework takes care
of the management of the loop over entries as well as low-level details such as I/O and parallelisation.
`TDataFrame` provides an interface to perform most common operations required by ROOT analyses;
at the same time, the users are not limited to those
common operations: building blocks to trigger custom calculations are available too.

`TDataFrame` is built with a *modular* and *flexible* workflow in mind, summarised as follows:

1.  **build a data-frame** object by specifying your data-set
2.  **apply a series of transformations** to your data
    1.  **filter** (e.g. apply some cuts) or
    2.  create a **temporary branch** (e.g. make available an alias or the result of a non trivial operation involving other branches)
3.  **apply actions** to the transformed data to produce results (e.g. fill a histogram)
4.
<table>
<tr>
   <td>
      <b>TTreeReader</b>
   </td>
   <td>
      <b>ROOT::Experimental::TDataFrame</b>
   </td>
</tr>
<tr>
   <td>
~~~{.cpp}
TTreeReader reader("myTree", file);
TTreeReaderValue&lt;A_t&gt; a(reader, "A");
TTreeReaderValue&lt;B_t&gt; b(reader, "B");
TTreeReaderValue&lt;C_t&gt; c(reader, "C");
while(reader.Next()) {
   if(IsGoodEvent(a, b, c))
      DoStuff(a, b, c);
}
~~~
   </td>
   <td>
~~~{.cpp}
ROOT::Experimental::TDataFrame d("myTree", file, {"A", "B", "C"});
d.Filter(IsGoodEvent).Foreach(DoStuff);
~~~
   </td>
</tr>
<tr>
   <td>
      <b>TTree::Draw</b>
   </td>
   <td>
      <b>ROOT::Experimental::TDataFrame</b>
   </td>
</tr>
<tr>
   <td>
~~~{.cpp}
TTree *t = static_cast&lt;TTree*&gt;(
   file->Get("myTree")
);
t->Draw("var", "var > 2");
~~~
   </td>
   <td>
~~~{.cpp}
ROOT::Experimental::TDataFrame d("myTree", file, "var");
d.Filter([](int v) { return v > 2; }).Histo();
~~~
   </td>
</tr>
</table>

Keep reading to follow a five-minute [crash course](#crash-course) to `TDataFrame`, or jump to an overview of useful [features](#more-features), or a more in-depth explanation of [transformations](#transformations), [actions](#actions) and [parallelism](#multi-thread-execution).

## <a name="crash-course"></a> Crash course
### Filling a histogram
Let's start with a very common task: filling a histogram
~~~{.cpp}
// Fill a TH1F with the "MET" branch
ROOT::Experimental::TDataFrame d("myTree", filePtr); // build a TDataFrame like you would build a TTreeReader
auto h = d.Histo("MET");
h->Draw();
~~~
The first line creates a `TDataFrame` associated to the `TTree` "myTree". This tree has a branch named "MET".

`Histo` is an action; it returns a smart pointer (a `TActionResultPtr` to be precise) to a `TH1F` histogram filled with the `MET` of all events.
If the quantity stored in the branch is a collection, the histogram is filled with its elements.

There are many other possible [actions](#overview), and all their results are wrapped in smart pointers; we'll see why in a minute.

### Applying a filter
Let's now pretend we want to cut over the value of branch "MET" and count how many events pass this cut:
~~~{.cpp}
// Select events with "MET" greater than 4., count events that passed the selection
auto metCut = [](double x) { return x > 4.; }; // a c++11 lambda function checking "x > 4"
ROOT::Experimental::TDataFrame d("myTree", filePtr);
auto c = d.Filter(metCut, {"MET"}).Count();
std::cout << *c << std::endl;
~~~
`Filter` takes a function (a lambda in this example, but it can be any kind of function or even a functor class) and a list of branch names. The filter function is applied to the specified branches for each event; it is required to return a `bool` which signals whether the event passes the filter (`true`) or not (`false`). You can think of your data as "flowing" through the chain of calls, being transformed, filtered and finally used to perform actions. Multiple `Filter` calls can be chained one after another.

### Creating a temporary branch
Let's now consider the case in which "myTree" contains two quantities "x" and "y", but our analysis relies on a derived quantity `z = sqrt(x*x + y*y)`.
Using the `AddBranch` transformation, we can create a new column in the data-set containin the variable "z":
~~~{.cpp}
auto sqrtSum = [](double x, double y) { return sqrt(x*x + y*y); };
auto zCut = [](double z) { return z > 0.; }

ROOT::Experimental::TDataFrame d(treeName, filePtr);
auto zMean = d.AddBranch("z", sqrtSum, {"x","y"})
              .Filter(zCut, {"z"})
              .Mean("z");
std::cout << *zMean << std::endl;
~~~
`AddBranch` creates the variable "z" by applying `sqrtSum` to "x" and "y". Later in the chain of calls we refer to variables created with `AddBranch` as if they were actual tree branches, but they are evaluated on the fly, once per event. As with filters, `AddBranch` calls can be chained with other transformations to create multiple temporary branches.

### Executing multiple actions
As a final example let us apply two different cuts on branch "MET" and fill two different histograms with the "pt\_v" of the filtered events.
You should be able to easily understand what's happening:
~~~{.cpp}
// fill two histograms with the results of two opposite cuts
auto isBig = [](double x) { return x > 10.; };
ROOT::Experimental::TDataFrame d(treeName, filePtr);
auto h1 = d.Filter(isBig, {"MET"}).Histo("pt_v");
auto h2 = d.Histo("pt_v");
h1->Draw();       // event loop is run once here
h2->Draw("SAME"); // no need to run the event loop again
~~~
`TDataFrame` executes all above actions by **running the event-loop only once**. The trick is that actions are not executed at the moment they are called, but they are **lazy**, i.e. delayed until the moment one of their results is accessed through the smart pointer. At that time, the even loop is triggered and *all* results are produced simultaneously.

It is therefore good practice to declare all your filters and actions *before* accessing their results, allowing `TDataFrame` to loop once and produce all results in one go.

### Going parallel
Let's say we would like to run the previous examples in parallel on several cores, dividing events fairly between cores. The only modification required to the snippets would be the addition of this line *before* constructing the main data-frame object:
~~~{.cpp}
ROOT::EnableImplicitMT();
~~~
Simple as that, enjoy your speed-up.

##  <a name="more-features"></a>More features
Here is a list of the most important features that have been omitted in the "Crash course" for brevity's sake.
You don't need to read all these to start using `TDataFrame`, but they are useful to save typing time and runtime.

### Default branch lists
When constructing a `TDataFrame` object, it is possible to specify a **default branch list** for your analysis, in the usual form of a list of strings representing branch names. The default branch list will be used as fallback whenever one specific to the transformation/action is not present.
~~~{.cpp}
// use "b1" and "b2" as default branches for `Filter`, `AddBranch` and actions
ROOT::Experimental::TDataFrame d1(treeName, &file, {"b1","b2"});
// filter acts on default branch list, no need to specify it
auto h = d1.Filter([](int b1, int b2) { return b1 > b2; }).Histo("otherVar");

// just one default branch this time
ROOT::Experimental::TDataFrame d2(treeName, &file, {"b1"});
// we can still specify non-default branch lists
// `Min` here can fall back to the default "b1"
auto min = d2.Filter([](double b2) { return b2 > 0; }, {"b2"}).Min();
~~~
<!-- Commented out until we have the std::array_view in ROOT
### Branches of collection types
We can rely on several features when dealing with branches of collection types (e.g. `vector<double>`, `int[3]`, or anything that you would put in a `TTreeReaderArray` rather than a `TTreeReaderValue`).

First of all, we **never need to spell out the exact type of the collection-type** branch in a transformation or an action. As it would be done when building a `TTreeReaderArray` for that branch, we just need to specify the type of the elements of the collection, in this way:
```c++
ROOT::Experimental::TDataFrame d(treeName, &file, {"vecBranch"});
d.Filter(ROOT::ArrayView<double> vecBranch) { return vecBranch.size() > 0; }).Histo();
```

Moreover, actions detect whenever they are applied to a collection type and **adapt their behaviour to act on all elements of the collection**, for each entry. In the example above, `Histo()` (equivalent to `Histo("vecBranch")`) fills the histogram with the values of all elements of `vecBranch`, for each event.-->

### Branch type guessing and explicit declaration of branch types
C++ is a statically typed language: all types must be known at compile-time. This includes the types of the `TTree` branches we want to work on. For filters, temporary branches and some of the actions, **branch types are deduced from the signature** of the relevant filter function/temporary branch expression/action function:
~~~{.cpp}
// here b1 is deduced to be `int` and b2 to be `double`
dataFrame.Filter([](int x, double y) { return x > 0 && y < 0.; }, {"b1", "b2"});
~~~
If we specify an incorrect type for one of the branches, an exception with an informative message will be thrown at runtime, when the branch value is actually read from the `TTree`: the implementation of `TDataFrame` allows the detection of type mismatches. The same would happen if we swapped the order of "b1" and "b2" in the branch list passed to `Filter`.

Certain actions, on the other hand, do not take a function as argument (e.g. `Histo`), so we cannot deduce the type of the branch at compile-time. In this case **`TDataFrame` tries to guess the type of the branch**, trying out the most common ones and `std::vector` thereof. This is why we never needed to specify the branch types for all actions in the above snippets.

When the branch type is not a common one such as `int`, `double`, `char` or `float` it is therefore good practice to specify it as a template parameter to the action itself, like this:
~~~{.cpp}
dataFrame.Histo("b1"); // OK if b1 is a "common" type
dataFrame.Histo<Object_t>("myObject"); // OK, "myObject" is deduced to be of type `Object_t`
// dataFrame.Histo("myObject"); // THROWS an exception
~~~

### Generic actions
`TDataFrame` strives to offer a comprehensive set of standard actions that can be performed on each event. At the same time, it **allows users to execute arbitrary code (i.e. a generic action) inside the event loop** through the `Foreach` and `ForeachSlot` actions.

`Foreach(f, branchList)` takes a function `f` (lambda expression, free function, functor...) and a list of branches, and executes `f` on those branches for each event. The function passed must return nothing (i.e. `void`). It can be used to perform actions that are not already available in the interface. For example, the following snippet evaluates the root mean square of branch "b":
~~~{.cpp}
// Single-thread evaluation of RMS of branch "b" using Foreach
double sumSq = 0.;
unsigned int n = 0;
ROOT::Experimental::TDataFrame d("bTree", bFilePtr);
d.Foreach([&sumSq, &n](double b) { ++n; sumSq += b*b; }, {"b"});
std::cout << "rms of b: " << std::sqrt(sumSq / n) << std::endl;
~~~
When executing on multiple threads, users are responsible for the thread-safety of the expression passed to `Foreach`.
The code above would need to employ some resource protection mechanism to ensure non-concurrent writing of `rms`; but this is probably too much head-scratch for such a simple operation.

`ForeachSlot` can help in this situation. It is an alternative version of `Foreach` for which the function takes an additional parameter besides the branches it should be applied to: an `unsigned int slot` parameter, where `slot` is a number indicating which thread (0, 1, 2 , ..., poolSize - 1) the function is being run in. We can take advantage of `ForeachSlot` to evaluate a thread-safe root mean square of branch "b":
~~~{.cpp}
// Thread-safe evaluation of RMS of branch "b" using ForeachSlot
ROOT::EnableImplicitMT();
unsigned int nSlots = ROOT::GetImplicitMTPoolSize();
std::vector<double> sumSqs(nSlots, 0.);
std::vector<unsigned int> ns(nSlots, 0);

ROOT::Experimental::TDataFrame d("bTree", bFilePtr);
d.ForeachSlot([&sumSqs, &ns](unsigned int slot, double b) { sumSqs[slot] += b*b; ns[slot] += 1; }, {"b"});
double sumSq = std::accumulate(sumSqs.begin(), sumSqs.end(), 0.); // sum all squares
unsigned int n = std::accumulate(ns.begin(), ns.end(), 0); // sum all counts
std::cout << "rms of b: " << std::sqrt(sumSq / n) << std::endl;
~~~
You see how we created one `double` variable for each thread in the pool, and later merged their results via `std::accumulate`.

### Call graphs (storing and reusing sets of transformations)
**Sets of transformations can be stored as variables** and reused multiple times to create **call graphs** in which several paths of filtering/creation of branches are executed simultaneously; we often refer to this as "storing the state of the chain".

This feature can be used, for example, to create a temporary branch once and use it in several subsequent filters or actions, or to apply a strict filter to the data-set *before* executing several other transformations and actions, effectively reducing the amount of events processed.

Let's try to make this clearer with a commented example:
~~~{.cpp}
// build the data-frame and specify a default branch list
ROOT::Experimental::TDataFrame d(treeName, filePtr, {"var1", "var2", "var3"});

// apply a cut and save the state of the chain
auto filtered = d.Filter(myBigCut);

// plot branch "var1" at this point of the chain
auto h1 = filtered.Histo("var1");

// create a new branch "vec" with a vector extracted from a complex object (only for filtered entries)
// and save the state of the chain
auto newBranchFiltered = filtered.AddBranch("vec", [](const Obj& o) { return o.getVector(); }, {"obj"});

// apply a cut and fill a histogram with "vec"
auto h2 = newBranchFiltered.Filter(cut1).Histo("vec");

// apply a different cut and fill a new histogram
auto h3 = newBranchFiltered.Filter(cut2).Histo("vec");

// Inspect results
h2->Draw(); // first access to an action result: run event-loop!
h3->Draw("SAME"); // event loop does not need to be run again here..
std::cout << "Entries in h1: " << h1->GetEntries() << std::endl; // ..or here
~~~
`TDataFrame` detects when several actions use the same filter or the same temporary branch, and **only evaluates each filter or temporary branch once per event**, regardless of how many times that result is used down the call graph. Objects read from each branch are **built once and never copied**, for maximum efficiency.
When "upstream" filters are not passed, subsequent filters, temporary branch expressions and actions are not evaluated, so it might be advisable to put the strictest filters first in the chain.

##  <a name="transformations"></a>Transformations
### Filters
A filter is defined through a call to `Filter(f, branchList)`. `f` can be a function, a lambda expression, a functor class, or any other callable object. It must return a `bool` signalling whether the event has passed the selection (`true`) or not (`false`). It must perform "read-only" actions on the branches, and should not have side-effects (e.g. modification of an external or static variable) to ensure correct results when implicit multi-threading is active.

`TDataFrame` only evaluates filters when necessary: if multiple filters are chained one after another, they are executed in order and the first one returning `false` causes the event to be discarded and triggers the processing of the next entry. If multiple actions or transformations depend on the same filter, that filter is not executed multiple times for each entry: after the first access it simply serves a cached result.

<!--#### Named filters To be uncommented when the support is added
An optional string parameter `filterName` can be specified to `Filter`, defining a **named filter**. Named filters work as usual, but also keep track of how many entries they accept and reject. Statistics are retrieved through a call to the `Report` method (coming soon).-->

### Temporary branches
Temporary branches are created by invoking `AddBranch(name, f, branchList)`. As usual, `f` can be any callable object (function, lambda expression, functor class...); it takes the values of the branches listed in `branchList` (a list of strings) as parameters, in the same order as they are listed in `branchList`. `f` must return the value that will be assigned to the temporary branch.

A new variable is created called `name`, accessible as if it was contained in the dataset from subsequent transformations/actions.

Use cases include:
- caching the results of complex calculations for easy and efficient multiple access
- extraction of quantities of interest from complex objects
- branch aliasing, i.e. changing the name of a branch

<!-- To be uncommented when the support is added
Temporary branch values can be persistified by saving them to a new `TTree` using the `Snapshot` action.-->
An exception is thrown if the `name` of the new branch is already in use for another branch in the `TTree`.

##  <a name="actions"></a>Actions
### Instant and lazy actions
Actions can be **instant** or **lazy**. Instant actions are executed as soon as they are called, while lazy actions are executed whenever the object they return is accessed for the first time. As a rule of thumb, actions with a return value are lazy, the others are instant.
<!--One notable exception is `Snapshot` (see the table [below](#overview)).

Whenever an action is executed, all (lazy) actions with the same **range** (see later) are executed within the same event loop.

### Ranges (coming soon)
Ranges of entries can (or must) be specified for all actions. **Only the specified range of entries will be processed** during the event loop.
The default range is, of course, beginning to end.-->


### Overview
Here is a quick overview of what actions are present and what they do. Each one is described in more detail in the reference guide.

In the following, whenever we say an action "returns" something, we always mean it returns a smart pointer to it. Also note that all actions are only executed for events that pass all preceding filters.

| **Lazy actions** | **Description** |
|------------------|-----------------|
| Count | Return the number of events processed. |
| Take | Build a collection of values of a branch. |
| Histo | Fill a histogram with the values of a branch that passed all filters. |
| Max | Return the maximum of processed branch values. |
| Mean | Return the mean of processed branch values. |
| Min | Return the minimum of processed branch values. |

| **Instant actions** | **Description** |
|---------------------|-----------------|
| Foreach | Execute a user-defined function on each entry. Users are responsible for the thread-safety of this lambda when executing with implicit multi-threading enabled. |
| ForeachSlot | Same as `Foreach`, but the user-defined function must take an extra `unsigned int slot` as its first parameter. `slot` will take a different value, `0` to `nThreads - 1`, for each thread of execution. This is meant as a helper in writing thread-safe `Foreach` actions when using `TDataFrame` after `ROOT::EnableImplicitMT()`. `ForeachSlot` works just as well with single-thread execution: in that case `slot` will always be `0`. |

<!-- to be added at the correct row when supported -->
<!-- Accumulate | Execute a function with signature `R(R,T)` on each entry. T is a branch, R is an accumulator. Return the final value of the accumulator | coming soon -->
<!-- Reduce | Execute a function with signature `T(T,T)` on each entry. Processed branch values are reduced (e.g. summed, merged) using this function. Return the final result of the reduction operation | coming soon -->
<!-- Sum | Return the sum of processed branch values | coming soon -->
<!-- Head | Take a number `n`, run and pretty-print the first `n` events that passed all filters | coming soon -->
<!-- Snapshot | Save a set of branches and temporary branches to disk, return a new `TDataFrame` that works on the skimmed, augmented or otherwise processed data | coming soon -->
<!-- Tail  | Take a number `n`, run and pretty-print the last `n` events that passed all filters | coming soon -->

##  <a name="parallel-execution"></a>Parallel execution
As pointed out before in this document, `TDataFrame` can transparently perform multi-threaded event loops to speed up the execution of its actions. Users only have to call `ROOT::EnableImplicitMT()` *before* constructing the `TDataFrame` object to indicate that it should take advantage of a pool of worker threads. **Each worker thread processes a distinct subset of entries**, and their partial results are merged before returning the final values to the user.

### Thread safety
`Filter` and `AddBranch` transformations should be inherently thread-safe: they have no side-effects and are not dependent on global state.
Most `Filter`/`AddBranch` functions will in fact be pure in the functional programming sense.
All actions are built to be thread-safe with the exception of `Foreach`, in which case users are responsible of thread-safety, see [here](#generic-actions).

<!--## Example snippets
Here you can find pre-made solutions to common problems. They should work out-of-the-box provided you have our "TDFTestTree.root" in the same directory where you execute the snippet.<br>
Please contact us if you think we are missing important, common use-cases.

```c++
// evaluation of several cuts
```-->

*/
/*




*/

#ifndef ROOT_TDATAFRAME
#define ROOT_TDATAFRAME

#include "TBranchElement.h"
#include "TDirectory.h"
#include "TH1F.h" // For Histo actions
#include "ROOT/TDFOperations.hxx"
#include "ROOT/TDFTraitsUtils.hxx"
#include "ROOT/TTreeProcessorMT.hxx"
#include "ROOT/TSpinMutex.hxx"
#include "TROOT.h" // IsImplicitMTEnabled, GetImplicitMTPoolSize
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

#include <algorithm> // std::find
#include <array>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

namespace ROOT {

using BranchNames = std::vector<std::string>;

// Fwd declarations
namespace Detail {
class TDataFrameImpl;
}

namespace Experimental {

/// Smart pointer for the return type of actions
/**
\class ROOT::Experimental::TActionResultProxy
\ingroup dataframe
\brief A wrapper around the result of TDataFrame actions able to trigger calculations lazily.
\tparam T Type of the action result

A smart pointer which allows to access the result of a TDataFrame action. The
methods of the encapsulated object can be accessed via the arrow operator.
Upon invocation of the arrow operator or dereferencing (`operator*`), the
loop on the events and calculations of all scheduled actions are executed
if needed.
It is possible to iterate on the result proxy if the proxied object is a collection.
~~~{.cpp}
for (auto& myItem : myResultProxy) { ... };
~~~
If iteration is not supported by the type of the proxied object, a compilation error is thrown.

*/
template <typename T>
class TActionResultProxy {
/// \cond HIDDEN_SYMBOLS
   template<typename V, bool isCont = ROOT::Internal::TDFTraitsUtils::TIsContainer<V>::fgValue>
   struct TIterationHelper{
      using Iterator_t = void;
      void GetBegin(const V& ){static_assert(sizeof(V) == 0, "It does not make sense to ask begin for this class.");}
      void GetEnd(const V& ){static_assert(sizeof(V) == 0, "It does not make sense to ask end for this class.");}
   };

   template<typename V>
   struct TIterationHelper<V,true>{
      using Iterator_t = decltype(std::begin(std::declval<V>()));
      static Iterator_t GetBegin(const V& v) {return std::begin(v);};
      static Iterator_t GetEnd(const V& v) {return std::end(v);};
   };
/// \endcond
   using SPT_t = std::shared_ptr<T> ;
   using SPTDFI_t = std::shared_ptr<ROOT::Detail::TDataFrameImpl>;
   using WPTDFI_t = std::weak_ptr<ROOT::Detail::TDataFrameImpl>;
   using ShrdPtrBool_t = std::shared_ptr<bool>;
   friend class ROOT::Detail::TDataFrameImpl;

   ShrdPtrBool_t fReadiness = std::make_shared<bool>(false); ///< State registered also in the TDataFrameImpl until the event loop is executed
   WPTDFI_t fFirstData;                                      ///< Original TDataFrame
   SPT_t fObjPtr;                                            ///< Shared pointer encapsulating the wrapped result
   /// Triggers the event loop in the TDataFrameImpl instance to which it's associated via the fFirstData
   void TriggerRun();
   /// Get the pointer to the encapsulated result.
   /// Ownership is not transferred to the caller.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
   T *Get()
   {
      if (!*fReadiness) TriggerRun();
      return fObjPtr.get();
   }
   TActionResultProxy(SPT_t objPtr, ShrdPtrBool_t readiness, SPTDFI_t firstData)
      : fReadiness(readiness), fFirstData(firstData), fObjPtr(objPtr) { }
   /// Factory to allow to keep the constructor private
   static TActionResultProxy<T> MakeActionResultPtr(SPT_t objPtr, ShrdPtrBool_t readiness, SPTDFI_t firstData)
   {
      return TActionResultProxy(objPtr, readiness, firstData);
   }
public:
   TActionResultProxy() = delete;
   /// Get a reference to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
   T &operator*() { return *Get(); }
   /// Get a pointer to the encapsulated object.
   /// Ownership is not transferred to the caller.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
   T *operator->() { return Get(); }
   /// Return an iterator to the beginning of the contained object if this makes
   /// sense, throw a compilation error otherwise
   typename TIterationHelper<T>::Iterator_t begin()
   {
      if (!*fReadiness) TriggerRun();
      return TIterationHelper<T>::GetBegin(*fObjPtr);
   }
   /// Return an iterator to the end of the contained object if this makes
   /// sense, throw a compilation error otherwise
   typename TIterationHelper<T>::Iterator_t end()
   {
      if (!*fReadiness) TriggerRun();
      return TIterationHelper<T>::GetEnd(*fObjPtr);
   }
};

} // end NS Experimental

} // end NS ROOT

// Internal classes

namespace ROOT {

namespace Detail {
class TDataFrameImpl;
}

namespace Internal {

unsigned int GetNSlots() {
   unsigned int nSlots = 1;
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled()) nSlots = ROOT::GetImplicitMTPoolSize();
#endif // R__USE_IMT
   return nSlots;
}

using TVBPtr_t = std::shared_ptr<TTreeReaderValueBase>;
using TVBVec_t = std::vector<TVBPtr_t>;

template <int... S, typename... BranchTypes>
TVBVec_t BuildReaderValues(TTreeReader &r, const BranchNames &bl, const BranchNames &tmpbl,
                           TDFTraitsUtils::TTypeList<BranchTypes...>,
                           TDFTraitsUtils::TStaticSeq<S...>)
{
   // isTmpBranch has length bl.size(). Elements are true if the corresponding
   // branch is a "fake" branch created with AddBranch, false if they are
   // actual branches present in the TTree.
   std::array<bool, sizeof...(S)> isTmpBranch;
   for (unsigned int i = 0; i < isTmpBranch.size(); ++i)
      isTmpBranch[i] = std::find(tmpbl.begin(), tmpbl.end(), bl.at(i)) != tmpbl.end();

   // Build vector of pointers to TTreeReaderValueBase.
   // tvb[i] points to a TTreeReaderValue specialized for the i-th BranchType,
   // corresponding to the i-th branch in bl
   // For temporary branches (declared with AddBranch) a nullptr is created instead
   // S is expected to be a sequence of sizeof...(BranchTypes) integers
   TVBVec_t tvb{isTmpBranch[S] ? nullptr : std::make_shared<TTreeReaderValue<BranchTypes>>(
                                            r, bl.at(S).c_str())...}; // "..." expands BranchTypes and S simultaneously

   return tvb;
}

template <typename Filter>
void CheckFilter(Filter)
{
   using FilterRet_t = typename TDFTraitsUtils::TFunctionTraits<Filter>::RetType_t;
   static_assert(std::is_same<FilterRet_t, bool>::value, "filter functions must return a bool");
}

void CheckTmpBranch(const std::string& branchName, TTree *treePtr)
{
   auto branch = treePtr->GetBranch(branchName.c_str());
   if (branch != nullptr) {
      auto msg = "branch \"" + branchName + "\" already present in TTree";
      throw std::runtime_error(msg);
   }
}

/// Returns local BranchNames or default BranchNames according to which one should be used
const BranchNames &PickBranchNames(unsigned int nArgs, const BranchNames &bl, const BranchNames &defBl)
{
   bool useDefBl = false;
   if (nArgs != bl.size()) {
      if (bl.size() == 0 && nArgs == defBl.size()) {
         useDefBl = true;
      } else {
         auto msg = "mismatch between number of filter arguments (" + std::to_string(nArgs) +
                    ") and number of branches (" + std::to_string(bl.size() ? bl.size() : defBl.size()) + ")";
         throw std::runtime_error(msg);
      }
   }

   return useDefBl ? defBl : bl;
}

class TDataFrameActionBase {
public:
   virtual ~TDataFrameActionBase() {}
   virtual void Run(unsigned int slot, int entry) = 0;
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   virtual void CreateSlots(unsigned int nSlots) = 0;
};

using ActionBasePtr_t = std::shared_ptr<TDataFrameActionBase>;
using ActionBaseVec_t = std::vector<ActionBasePtr_t>;

// Forward declarations
template <int S, typename T>
T &GetBranchValue(TVBPtr_t &readerValues, unsigned int slot, int entry, const std::string &branch,
                  std::weak_ptr<ROOT::Detail::TDataFrameImpl> df);

template <typename F, typename PrevDataFrame>
class TDataFrameAction final : public TDataFrameActionBase {
   using BranchTypes_t = typename TDFTraitsUtils::TRemoveFirst<typename TDFTraitsUtils::TFunctionTraits<F>::ArgTypes_t>::Types_t;
   using TypeInd_t = typename TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;

   F fAction;
   const BranchNames fBranches;
   const BranchNames fTmpBranches;
   PrevDataFrame *fPrevData;
   std::weak_ptr<ROOT::Detail::TDataFrameImpl> fFirstData;
   std::vector<TVBVec_t> fReaderValues;

public:
   TDataFrameAction(F f, const BranchNames &bl, std::weak_ptr<PrevDataFrame> pd)
      : fAction(f), fBranches(bl), fTmpBranches(pd.lock()->GetTmpBranches()), fPrevData(pd.lock().get()),
        fFirstData(pd.lock()->GetDataFrame()) { }

   TDataFrameAction(const TDataFrameAction &) = delete;

   void Run(unsigned int slot, int entry)
   {
      // check if entry passes all filters
      if (CheckFilters(slot, entry)) ExecuteAction(slot, entry);
   }

   bool CheckFilters(unsigned int slot, int entry)
   {
      // start the recursive chain of CheckFilters calls
      return fPrevData->CheckFilters(slot, entry);
   }

   void ExecuteAction(unsigned int slot, int entry) { ExecuteActionHelper(slot, entry, TypeInd_t(), BranchTypes_t()); }

   void CreateSlots(unsigned int nSlots) { fReaderValues.resize(nSlots); }

   void BuildReaderValues(TTreeReader &r, unsigned int slot)
   {
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }

   template <int... S, typename... BranchTypes>
   void ExecuteActionHelper(unsigned int slot, int entry,
                            TDFTraitsUtils::TStaticSeq<S...>,
                            TDFTraitsUtils::TTypeList<BranchTypes...>)
   {
      // Take each pointer in tvb, cast it to a pointer to the
      // correct specialization of TTreeReaderValue, and get its content.
      // S expands to a sequence of integers 0 to sizeof...(types)-1
      // S and types are expanded simultaneously by "..."
      (void) entry; // avoid bogus unused-but-set-parameter warning by gcc
      fAction(slot, GetBranchValue<S, BranchTypes>(fReaderValues[slot][S], slot, entry, fBranches[S], fFirstData)...);
   }
};

enum class EActionType : short { kHisto1D, kMin, kMax, kMean };

// Utilities to accommodate v7
namespace TDFV7Utils {

template<typename T, bool ISV7HISTO = !std::is_base_of<TH1, T>::value>
struct TIsV7Histo {
   const static bool fgValue = ISV7HISTO;
};

template<typename T, bool ISV7HISTO = TIsV7Histo<T>::fgValue>
struct Histo {
   static void SetCanExtendAllAxes(T& h)
   {
      h.SetCanExtend(TH1::kAllAxes);
   }
   static bool HasAxisLimits(T& h)
   {
      auto xaxis = h.GetXaxis();
      return !(xaxis->GetXmin() == 0. && xaxis->GetXmax() == 0.);
   }
};

template<typename T>
struct Histo<T, true> {
   static void SetCanExtendAllAxes(T&) { }
   static bool HasAxisLimits(T&) {return true;}
};

} // end NS TDFV7Utils

} // end NS Internal

namespace Detail {
// forward declarations for TDataFrameInterface
template <typename F, typename PrevData>
class TDataFrameFilter;
template <typename F, typename PrevData>
class TDataFrameBranch;
class TDataFrameImpl;
}

namespace Experimental {

/**
* \class ROOT::Experimental::TDataFrameInterface
* \ingroup dataframe
* \brief The public interface to the TDataFrame federation of classes: TDataFrameImpl, TDataFrameFilter, TDataFrameBranch
* \tparam T One of the TDataFrameImpl, TDataFrameFilter, TDataFrameBranch classes. The user never specifies this type manually.
*/
template <typename Proxied>
class TDataFrameInterface {
   template<typename T> friend class TDataFrameInterface;
public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Build the dataframe
   /// \param[in] treeName Name of the tree contained in the directory
   /// \param[in] dirPtr TDirectory where the tree is stored, e.g. a TFile.
   /// \param[in] defaultBranches Collection of default branches.
   ///
   /// The default branches are looked at in case no branch is specified in the
   /// booking of actions or transformations.
   TDataFrameInterface(const std::string &treeName, TDirectory *dirPtr, const BranchNames &defaultBranches = {});

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Build the dataframe
   /// \param[in] tree The tree or chain to be studied.
   /// \param[in] defaultBranches Collection of default branches.
   ///
   /// The default branches are looked at in case no branch is specified in the
   /// booking of actions or transformations.
   TDataFrameInterface(TTree &tree, const BranchNames &defaultBranches = {});

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool` signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] bl Names of the branches in input to the filter function.
   ///
   /// Append a filter node at the point of the call graph corresponding to the
   /// object this method is called on.
   /// The callable `f` should not have side-effects (e.g. modification of an
   /// external or static variable) to ensure correct results when implicit
   /// multi-threading is active.
   ///
   /// TDataFrame only evaluates filters when necessary: if multiple filters
   /// are chained one after another, they are executed in order and the first
   /// one returning false causes the event to be discarded.
   /// Even if multiple actions or transformations depend on the same filter,
   /// it is executed once per entry. If its result is requested more than
   /// once, the cached result is served.
   template <typename F>
   TDataFrameInterface<ROOT::Detail::TDataFrameFilter<F, Proxied>> Filter(F f, const BranchNames &bl = {})
   {
      ROOT::Internal::CheckFilter(f);
      auto df = GetDataFrameChecked();
      const BranchNames &defBl = df->GetDefaultBranches();
      auto nArgs = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::ArgTypes_t::fgSize;
      const BranchNames &actualBl = ROOT::Internal::PickBranchNames(nArgs, bl, defBl);
      using DFF_t = ROOT::Detail::TDataFrameFilter<F, Proxied>;
      auto FilterPtr = std::make_shared<DFF_t> (f, actualBl, fProxiedPtr);
      TDataFrameInterface<DFF_t> tdf_f(FilterPtr);
      df->Book(FilterPtr);
      return tdf_f;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a temporary branch
   /// \param[in] name The name of the temporary branch.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the temporary value. Returns the value that will be assigned to the temporary branch.
   /// \param[in] bl Names of the branches in input to the producer function.
   ///
   /// Create a temporary branch that will be visible from all subsequent nodes
   /// of the functional chain. The `expression` is only evaluated for entries that pass
   /// all the preceding filters.
   /// A new variable is created called `name`, accessible as if it was contained
   /// in the dataset from subsequent transformations/actions.
   ///
   /// Use cases include:
   ///
   /// * caching the results of complex calculations for easy and efficient multiple access
   /// * extraction of quantities of interest from complex objects
   /// * branch aliasing, i.e. changing the name of a branch
   ///
   /// An exception is thrown if the name of the new branch is already in use
   /// for another branch in the TTree.
   template <typename F>
   TDataFrameInterface<ROOT::Detail::TDataFrameBranch<F, Proxied>>
   AddBranch(const std::string &name, F expression, const BranchNames &bl = {})
   {
      auto df = GetDataFrameChecked();
      ROOT::Internal::CheckTmpBranch(name, df->GetTree());
      const BranchNames &defBl = df->GetDefaultBranches();
      auto nArgs = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::ArgTypes_t::fgSize;
      const BranchNames &actualBl = ROOT::Internal::PickBranchNames(nArgs, bl, defBl);
      using DFB_t = ROOT::Detail::TDataFrameBranch<F, Proxied>;
      auto BranchPtr = std::make_shared<DFB_t>(name, expression, actualBl, fProxiedPtr);
      TDataFrameInterface<DFB_t> tdf_b(BranchPtr);
      df->Book(BranchPtr);
      return tdf_b;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined calculations.
   /// \param[in] bl Names of the branches in input to the user function.
   ///
   /// The callable `f` is invoked once per entry. This is an *instant action*:
   /// upon invocation, an event loop as well as execution of all scheduled actions
   /// is triggered.
   /// Users are responsible for the thread-safety of this callable when executing
   /// with implicit multi-threading enabled (i.e. ROOT::EnableImplicitMT).
   template <typename F>
   void Foreach(F f, const BranchNames &bl = {})
   {
      namespace IU = ROOT::Internal::TDFTraitsUtils;
      using ArgTypes_t = typename IU::TFunctionTraits<decltype(f)>::ArgTypesNoDecay_t;
      using RetType_t = typename IU::TFunctionTraits<decltype(f)>::RetType_t;
      auto fWithSlot = IU::AddSlotParameter<RetType_t>(f, ArgTypes_t());
      ForeachSlot(fWithSlot, bl);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function requiring a processing slot index on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined calculations.
   /// \param[in] bl Names of the branches in input to the user function.
   ///
   /// Same as `Foreach`, but the user-defined function takes an extra
   /// `unsigned int` as its first parameter, the *processing slot index*.
   /// This *slot index* will be assigned a different value, `0` to `poolSize - 1`,
   /// for each thread of execution.
   /// This is meant as a helper in writing thread-safe `Foreach`
   /// actions when using `TDataFrame` after `ROOT::EnableImplicitMT()`.
   /// The user-defined processing callable is able to follow different
   /// *streams of processing* indexed by the first parameter.
   /// `ForeachSlot` works just as well with single-thread execution: in that
   /// case `slot` will always be `0`.
   template<typename F>
   void ForeachSlot(F f, const BranchNames &bl = {}) {
      auto df = GetDataFrameChecked();
      const BranchNames &defBl= df->GetDefaultBranches();
      auto nArgs = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::ArgTypes_t::fgSize;
      const BranchNames &actualBl = ROOT::Internal::PickBranchNames(nArgs-1, bl, defBl);
      using DFA_t  = ROOT::Internal::TDataFrameAction<decltype(f), Proxied>;
      df->Book(std::make_shared<DFA_t>(f, actualBl, fProxiedPtr));
      df->Run();
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the number of entries processed (*lazy action*)
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   TActionResultProxy<unsigned int> Count()
   {
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto cShared = std::make_shared<unsigned int>(0);
      auto c = df->MakeActionResultPtr(cShared);
      auto cPtr = cShared.get();
      auto cOp = std::make_shared<ROOT::Internal::Operations::CountOperation>(cPtr, nSlots);
      auto countAction = [cOp](unsigned int slot) mutable { cOp->Exec(slot); };
      BranchNames bl = {};
      using DFA_t = ROOT::Internal::TDataFrameAction<decltype(countAction), Proxied>;
      df->Book(std::shared_ptr<DFA_t>(new DFA_t(countAction, bl, fProxiedPtr)));
      return c;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a collection of values of a branch (*lazy action*)
   /// \tparam T The type of the branch.
   /// \tparam COLL The type of collection used to store the values.
   /// \param[in] branchName The name of the branch of which the values are to be collected
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T, typename COLL = std::vector<T>>
   TActionResultProxy<COLL> Take(const std::string &branchName = "")
   {
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "get the values of the branch");
      auto valuesPtr = std::make_shared<COLL>();
      auto values = df->MakeActionResultPtr(valuesPtr);
      auto getOp = std::make_shared<ROOT::Internal::Operations::TakeOperation<T,COLL>>(valuesPtr, nSlots);
      auto getAction = [getOp] (unsigned int slot , const T &v) mutable { getOp->Exec(v, slot); };
      BranchNames bl = {theBranchName};
      using DFA_t = ROOT::Internal::TDataFrameAction<decltype(getAction), Proxied>;
      df->Book(std::shared_ptr<DFA_t>(new DFA_t(getAction, bl, fProxiedPtr)));
      return values;
   }


   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam T The type of the branch the values of which are used to fill the histogram.
   /// \param[in] branchName The name of the branch of which the values are to be collected.
   /// \param[in] model The model to be copied to build the new return value.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   /// The returned histogram is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = double>
   TActionResultProxy<TH1F> Histo(const std::string &branchName, const TH1F &model)
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "fill the histogram");
      auto h = std::make_shared<TH1F>(model);
      return CreateAction<T, ROOT::Internal::EActionType::kHisto1D>(theBranchName, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam T The type of the branch the values of which are used to fill the histogram.
   /// \param[in] branchName The name of the branch of which the values are to be collected.
   /// \param[in] nbins The number of bins.
   /// \param[in] minVal The lower value of the xaxis.
   /// \param[in] maxVal The upper value of the xaxis.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// If no axes boundaries are specified, all entries are buffered: at the end of
   /// the loop on the entries, the histogram is filled. If the axis boundaries are
   /// specified, the histogram (or histograms in the parallel case) are filled. This
   /// latter mode may result in a reduced memory footprint.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = double>
   TActionResultProxy<TH1F> Histo(const std::string &branchName = "", int nBins = 128, double minVal = 0.,
                                double maxVal = 0.)
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "fill the histogram");
      auto h = std::make_shared<TH1F>("", "", nBins, minVal, maxVal);
      if (minVal == maxVal) {
         ROOT::Internal::TDFV7Utils::Histo<TH1F>::SetCanExtendAllAxes(*h);
      }
      return CreateAction<T, ROOT::Internal::EActionType::kHisto1D>(theBranchName, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the minimum of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = double>
   TActionResultProxy<double> Min(const std::string &branchName = "")
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "calculate the minumum");
      auto minV = std::make_shared<T>(std::numeric_limits<T>::max());
      return CreateAction<T, ROOT::Internal::EActionType::kMin>(theBranchName, minV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the maximum of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = double>
   TActionResultProxy<double> Max(const std::string &branchName = "")
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "calculate the maximum");
      auto maxV = std::make_shared<T>(std::numeric_limits<T>::min());
      return CreateAction<T, ROOT::Internal::EActionType::kMax>(theBranchName, maxV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the mean of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = double>
   TActionResultProxy<double> Mean(const std::string &branchName = "")
   {
      auto theBranchName(branchName);
      GetDefaultBranchName(theBranchName, "calculate the mean");
      auto meanV = std::make_shared<T>(0);
      return CreateAction<T, ROOT::Internal::EActionType::kMean>(theBranchName, meanV);
   }

private:
   TDataFrameInterface(std::shared_ptr<Proxied> proxied) : fProxiedPtr(proxied) {}

   /// Get the TDataFrameImpl if reachable. If not, throw.
   std::shared_ptr<ROOT::Detail::TDataFrameImpl> GetDataFrameChecked()
   {
      auto df = fProxiedPtr->GetDataFrame().lock();
      if (!df) {
         throw std::runtime_error("The main TDataFrame is not reachable: did it go out of scope?");
      }
      return df;
   }

   void GetDefaultBranchName(std::string &theBranchName, const std::string &actionNameForErr)
   {
      if (theBranchName.empty()) {
         // Try the default branch if possible
         auto df = GetDataFrameChecked();
         const BranchNames &defBl = df->GetDefaultBranches();
         if (defBl.size() == 1) {
            theBranchName = defBl[0];
         } else {
            std::string msg("No branch in input to ");
            msg += actionNameForErr;
            msg += " and default branch list has size ";
            msg += std::to_string(defBl.size());
            msg += ", need 1";
            throw std::runtime_error(msg);
         }
      }
   }

   /// \cond HIDDEN_SYMBOLS
   template <typename BranchType, typename ActionResultType, enum ROOT::Internal::EActionType, typename ThisType>
   struct SimpleAction {};

   template <typename BranchType, typename ThisType>
   struct SimpleAction<BranchType, TH1F, ROOT::Internal::EActionType::kHisto1D, ThisType> {
      static TActionResultProxy<TH1F> BuildAndBook(ThisType thisFrame, const std::string &theBranchName,
                                                 std::shared_ptr<TH1F> h, unsigned int nSlots)
      {
         // we use a shared_ptr so that the operation has the same scope of the lambda
         // and therefore of the TDataFrameAction that contains it: merging of results
         // from different threads is performed in the operation's destructor, at the
         // moment when the TDataFrameAction is deleted by TDataFrameImpl
         BranchNames bl = {theBranchName};
         auto df = thisFrame->GetDataFrameChecked();
         auto hasAxisLimits = ROOT::Internal::TDFV7Utils::Histo<TH1F>::HasAxisLimits(*h);

         if (hasAxisLimits) {
            auto fillTOOp = std::make_shared<ROOT::Internal::Operations::FillTOOperation<TH1F>>(h, nSlots);
            auto fillLambda = [fillTOOp](unsigned int slot, const BranchType &v) mutable { fillTOOp->Exec(v, slot); };
            using DFA_t = ROOT::Internal::TDataFrameAction<decltype(fillLambda), Proxied>;
            df->Book(std::make_shared<DFA_t>(fillLambda, bl, thisFrame->fProxiedPtr));
         } else {
            auto fillOp = std::make_shared<ROOT::Internal::Operations::FillOperation<TH1F>>(h, nSlots);
            auto fillLambda = [fillOp](unsigned int slot, const BranchType &v) mutable { fillOp->Exec(v, slot); };
            using DFA_t = ROOT::Internal::TDataFrameAction<decltype(fillLambda), Proxied>;
            df->Book(std::make_shared<DFA_t>(fillLambda, bl, thisFrame->fProxiedPtr));
         }
         return df->MakeActionResultPtr(h);
      }
   };

   template <typename BranchType, typename ThisType, typename ActionResultType>
   struct SimpleAction<BranchType, ActionResultType, ROOT::Internal::EActionType::kMin, ThisType> {
      static TActionResultProxy<ActionResultType> BuildAndBook(ThisType thisFrame, const std::string &theBranchName,
                                                             std::shared_ptr<ActionResultType> minV, unsigned int nSlots)
      {
         // see "TActionResultProxy<TH1F> BuildAndBook" for why this is a shared_ptr
         auto minOp = std::make_shared<ROOT::Internal::Operations::MinOperation>(minV.get(), nSlots);
         auto minOpLambda = [minOp](unsigned int slot, const BranchType &v) mutable { minOp->Exec(v, slot); };
         BranchNames bl = {theBranchName};
         using DFA_t = ROOT::Internal::TDataFrameAction<decltype(minOpLambda), Proxied>;
         auto df = thisFrame->GetDataFrameChecked();
         df->Book(std::make_shared<DFA_t>(minOpLambda, bl, thisFrame->fProxiedPtr));
         return df->MakeActionResultPtr(minV);
      }
   };

   template <typename BranchType, typename ThisType, typename ActionResultType>
   struct SimpleAction<BranchType, ActionResultType, ROOT::Internal::EActionType::kMax, ThisType> {
      static TActionResultProxy<ActionResultType> BuildAndBook(ThisType thisFrame, const std::string &theBranchName,
                                                             std::shared_ptr<ActionResultType> maxV, unsigned int nSlots)
      {
         // see "TActionResultProxy<TH1F> BuildAndBook" for why this is a shared_ptr
         auto maxOp = std::make_shared<ROOT::Internal::Operations::MaxOperation>(maxV.get(), nSlots);
         auto maxOpLambda = [maxOp](unsigned int slot, const BranchType &v) mutable { maxOp->Exec(v, slot); };
         BranchNames bl = {theBranchName};
         using DFA_t = ROOT::Internal::TDataFrameAction<decltype(maxOpLambda), Proxied>;
         auto df = thisFrame->GetDataFrameChecked();
         df->Book(std::make_shared<DFA_t>(maxOpLambda, bl, thisFrame->fProxiedPtr));
         return df->MakeActionResultPtr(maxV);
      }
   };

   template <typename BranchType, typename ThisType, typename ActionResultType>
   struct SimpleAction<BranchType, ActionResultType, ROOT::Internal::EActionType::kMean, ThisType> {
      static TActionResultProxy<ActionResultType> BuildAndBook(ThisType thisFrame, const std::string &theBranchName,
                                                             std::shared_ptr<ActionResultType> meanV, unsigned int nSlots)
      {
         // see "TActionResultProxy<TH1F> BuildAndBook" for why this is a shared_ptr
         auto meanOp = std::make_shared<ROOT::Internal::Operations::MeanOperation>(meanV.get(), nSlots);
         auto meanOpLambda = [meanOp](unsigned int slot, const BranchType &v) mutable { meanOp->Exec(v, slot); };
         BranchNames bl = {theBranchName};
         using DFA_t = ROOT::Internal::TDataFrameAction<decltype(meanOpLambda), Proxied>;
         auto df = thisFrame->GetDataFrameChecked();
         df->Book(std::make_shared<DFA_t>(meanOpLambda, bl, thisFrame->fProxiedPtr));
         return df->MakeActionResultPtr(meanV);
      }
   };
   /// \endcond

   template <typename BranchType, ROOT::Internal::EActionType ActionType, typename ActionResultType>
   TActionResultProxy<ActionResultType> CreateAction(const std::string & theBranchName,
                                                   std::shared_ptr<ActionResultType> r)
   {
      // More types can be added at will at the cost of some compilation time and size of binaries.
      using ART_t = ActionResultType;
      using TT_t = decltype(this);
      const auto at = ActionType;
      auto df = GetDataFrameChecked();
      auto tree = static_cast<TTree*>(df->GetDirectory()->Get(df->GetTreeName().c_str()));
      auto branch = tree->GetBranch(theBranchName.c_str());
      unsigned int nSlots = df->GetNSlots();
      if (!branch) {
         // temporary branch
         const auto &type_id = df->GetBookedBranch(theBranchName).GetTypeId();
         if (type_id == typeid(char)) {
            return SimpleAction<char, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (type_id == typeid(int)) {
            return SimpleAction<int, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (type_id == typeid(double)) {
            return SimpleAction<double, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (type_id == typeid(std::vector<double>)) {
            return SimpleAction<std::vector<double>, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (type_id == typeid(std::vector<float>)) {
            return SimpleAction<std::vector<float>, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
      }
      // real branch
      auto branchEl = dynamic_cast<TBranchElement *>(branch);
      if (!branchEl) { // This is a fundamental type
         auto title    = branch->GetTitle();
         auto typeCode = title[strlen(title) - 1];
         if (typeCode == 'B') {
            return SimpleAction<char, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
         // else if (typeCode == 'b') { return SimpleAction<Uchar, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         // else if (typeCode == 'S') { return SimpleAction<Short_t, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         // else if (typeCode == 's') { return SimpleAction<UShort_t, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         else if (typeCode == 'I') {
            return SimpleAction<int, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
         // else if (typeCode == 'i') { return SimpleAction<unsigned int , ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         // else if (typeCode == 'F') { return SimpleAction<float, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         else if (typeCode == 'D') {
            return SimpleAction<double, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
         // else if (typeCode == 'L') { return SimpleAction<Long64_t, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         // else if (typeCode == 'l') { return SimpleAction<ULong64_t, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots); }
         else if (typeCode == 'O') {
            return SimpleAction<bool, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
      } else {
         std::string typeName = branchEl->GetTypeName();
         if (typeName == "vector<double>") {
            return SimpleAction<std::vector<double>, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         } else if (typeName == "vector<float>") {
            return SimpleAction<std::vector<float>, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
         }
      }
      return SimpleAction<BranchType, ART_t, at, TT_t>::BuildAndBook(this, theBranchName, r, nSlots);
   }

   std::shared_ptr<Proxied> fProxiedPtr;
};

/** Achieve a slimmer programming model. See documentation of TDataFrameInterface */
typedef TDataFrameInterface<ROOT::Detail::TDataFrameImpl> TDataFrame; // A typedef and not an alias to trigger auto{loading, parsing}

} // end NS Experimental

namespace Detail {

class TDataFrameBranchBase {
public:
   virtual ~TDataFrameBranchBase() {}
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   virtual void CreateSlots(unsigned int nSlots) = 0;
   virtual std::string GetName() const       = 0;
   virtual void *GetValue(unsigned int slot, int entry) = 0;
   virtual const std::type_info &GetTypeId() const = 0;
};
using TmpBranchBasePtr_t = std::shared_ptr<TDataFrameBranchBase>;

template <typename F, typename PrevData>
class TDataFrameBranch final : public TDataFrameBranchBase {
   using BranchTypes_t = typename Internal
   ::TDFTraitsUtils::TFunctionTraits<F>::ArgTypes_t;
   using TypeInd_t = typename ROOT::Internal::TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;
   using RetType_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::RetType_t;

   const std::string fName;
   F fExpression;
   const BranchNames fBranches;
   BranchNames fTmpBranches;
   std::vector<ROOT::Internal::TVBVec_t> fReaderValues;
   std::vector<std::shared_ptr<RetType_t>> fLastResultPtr;
   std::weak_ptr<TDataFrameImpl> fFirstData;
   PrevData *fPrevData;
   std::vector<int> fLastCheckedEntry = {-1};

public:
   TDataFrameBranch(const std::string &name, F expression, const BranchNames &bl, std::shared_ptr<PrevData> pd)
      : fName(name), fExpression(expression), fBranches(bl), fTmpBranches(pd->GetTmpBranches()),
        fFirstData(pd->GetDataFrame()), fPrevData(pd.get())
   {
      fTmpBranches.emplace_back(name);
   }

   TDataFrameBranch(const TDataFrameBranch &) = delete;

   std::weak_ptr<TDataFrameImpl> GetDataFrame() const { return fFirstData; }

   BranchNames GetTmpBranches() const { return fTmpBranches; }

   void BuildReaderValues(TTreeReader &r, unsigned int slot)
   {
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }

   void *GetValue(unsigned int slot, int entry)
   {
      if (entry != fLastCheckedEntry[slot]) {
         // evaluate this filter, cache the result
         auto newValuePtr = GetValueHelper(BranchTypes_t(), TypeInd_t(), slot, entry);
         fLastResultPtr[slot] = newValuePtr;
         fLastCheckedEntry[slot] = entry;
      }
      return static_cast<void *>(fLastResultPtr[slot].get());
   }

   const std::type_info &GetTypeId() const { return typeid(RetType_t); }

   void CreateSlots(unsigned int nSlots)
   {
      fReaderValues.resize(nSlots);
      fLastCheckedEntry.resize(nSlots);
      fLastResultPtr.resize(nSlots);
   }

   bool CheckFilters(unsigned int slot, int entry)
   {
      // dummy call: it just forwards to the previous object in the chain
      return fPrevData->CheckFilters(slot, entry);
   }

   std::string GetName() const { return fName; }

   template <int... S, typename... BranchTypes>
   std::shared_ptr<RetType_t> GetValueHelper(Internal::TDFTraitsUtils::TTypeList<BranchTypes...>,
                                             ROOT::Internal::TDFTraitsUtils::TStaticSeq<S...>,
                                             unsigned int slot, int entry)
   {
      auto valuePtr = std::make_shared<RetType_t>(fExpression(
         ROOT::Internal::GetBranchValue<S, BranchTypes>(fReaderValues[slot][S], slot, entry, fBranches[S], fFirstData)...));
      return valuePtr;
   }
};

class TDataFrameFilterBase {
public:
   virtual ~TDataFrameFilterBase() {}
   virtual void BuildReaderValues(TTreeReader &r, unsigned int slot) = 0;
   virtual void CreateSlots(unsigned int nSlots) = 0;
};
using FilterBasePtr_t = std::shared_ptr<TDataFrameFilterBase>;
using FilterBaseVec_t = std::vector<FilterBasePtr_t>;

template <typename FilterF, typename PrevDataFrame>
class TDataFrameFilter final : public TDataFrameFilterBase {
   using BranchTypes_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<FilterF>::ArgTypes_t;
   using TypeInd_t = typename ROOT::Internal::TDFTraitsUtils::TGenStaticSeq<BranchTypes_t::fgSize>::Type_t;

   FilterF fFilter;
   const BranchNames fBranches;
   const BranchNames fTmpBranches;
   PrevDataFrame *fPrevData;
   std::weak_ptr<TDataFrameImpl> fFirstData;
   std::vector<ROOT::Internal::TVBVec_t> fReaderValues = {};
   std::vector<int> fLastCheckedEntry = {-1};
   std::vector<int> fLastResult = {true}; // std::vector<bool> cannot be used in a MT context safely

public:
   TDataFrameFilter(FilterF f, const BranchNames &bl, std::shared_ptr<PrevDataFrame> pd)
      : fFilter(f), fBranches(bl), fTmpBranches(pd->GetTmpBranches()), fPrevData(pd.get()),
        fFirstData(pd->GetDataFrame()) { }

   std::weak_ptr<TDataFrameImpl> GetDataFrame() const { return fFirstData; }

   BranchNames GetTmpBranches() const { return fTmpBranches; }

   TDataFrameFilter(const TDataFrameFilter &) = delete;

   bool CheckFilters(unsigned int slot, int entry)
   {
      if (entry != fLastCheckedEntry[slot]) {
         if (!fPrevData->CheckFilters(slot, entry)) {
            // a filter upstream returned false, cache the result
            fLastResult[slot] = false;
         } else {
            // evaluate this filter, cache the result
            fLastResult[slot] = CheckFilterHelper(BranchTypes_t(), TypeInd_t(), slot, entry);
         }
         fLastCheckedEntry[slot] = entry;
      }
      return fLastResult[slot];
   }

   template <int... S, typename... BranchTypes>
   bool CheckFilterHelper(Internal::TDFTraitsUtils::TTypeList<BranchTypes...>,
                          ROOT::Internal::TDFTraitsUtils::TStaticSeq<S...>,
                          unsigned int slot, int entry)
   {
      // Take each pointer in tvb, cast it to a pointer to the
      // correct specialization of TTreeReaderValue, and get its content.
      // S expands to a sequence of integers 0 to `sizeof...(types)-1
      // S and types are expanded simultaneously by "..."
      (void) slot; // avoid bogus unused-but-set-parameter warning by gcc
      (void) entry; // avoid bogus unused-but-set-parameter warning by gcc
      return fFilter(
         ROOT::Internal::GetBranchValue<S, BranchTypes>(fReaderValues[slot][S], slot, entry, fBranches[S], fFirstData)...);
   }

   void BuildReaderValues(TTreeReader &r, unsigned int slot)
   {
      fReaderValues[slot] = ROOT::Internal::BuildReaderValues(r, fBranches, fTmpBranches, BranchTypes_t(), TypeInd_t());
   }

   void CreateSlots(unsigned int nSlots)
   {
      fReaderValues.resize(nSlots);
      fLastCheckedEntry.resize(nSlots);
      fLastResult.resize(nSlots);
   }
};

class TDataFrameImpl {

   ROOT::Internal::ActionBaseVec_t fBookedActions;
   ROOT::Detail::FilterBaseVec_t fBookedFilters;
   std::map<std::string, TmpBranchBasePtr_t> fBookedBranches;
   std::vector<std::shared_ptr<bool>> fResPtrsReadiness;
   std::string fTreeName;
   TDirectory *fDirPtr = nullptr;
   TTree *fTree = nullptr;
   const BranchNames fDefaultBranches;
   // always empty: each object in the chain copies this list from the previous
   // and they must copy an empty list from the base TDataFrameImpl
   const BranchNames fTmpBranches;
   unsigned int fNSlots;
   // TDataFrameInterface<TDataFrameImpl> calls SetFirstData to set this to a
   // weak pointer to the TDataFrameImpl object itself
   // so subsequent objects in the chain can call GetDataFrame on TDataFrameImpl
   std::weak_ptr<TDataFrameImpl> fFirstData;

public:
   TDataFrameImpl(const std::string &treeName, TDirectory *dirPtr, const BranchNames &defaultBranches = {})
      : fTreeName(treeName), fDirPtr(dirPtr), fDefaultBranches(defaultBranches), fNSlots(ROOT::Internal::GetNSlots()) { }

   TDataFrameImpl(TTree &tree, const BranchNames &defaultBranches = {}) : fTree(&tree), fDefaultBranches(defaultBranches), fNSlots(ROOT::Internal::GetNSlots())
   { }

   TDataFrameImpl(const TDataFrameImpl &) = delete;

   void Run()
   {
#ifdef R__USE_IMT
      if (ROOT::IsImplicitMTEnabled()) {
         const auto fileName = fTree ? static_cast<TFile *>(fTree->GetCurrentFile())->GetName() : fDirPtr->GetName();
         const std::string    treeName = fTree ? fTree->GetName() : fTreeName;
         ROOT::TTreeProcessorMT tp(fileName, treeName);
         ROOT::TSpinMutex     slotMutex;
         std::map<std::thread::id, unsigned int> slotMap;
         unsigned int globalSlotIndex = 0;
         CreateSlots(fNSlots);
         tp.Process([this, &slotMutex, &globalSlotIndex, &slotMap](TTreeReader &r) -> void {
            const auto thisThreadID = std::this_thread::get_id();
            unsigned int slot;
            {
               std::lock_guard<ROOT::TSpinMutex> l(slotMutex);
               auto thisSlotIt = slotMap.find(thisThreadID);
               if (thisSlotIt != slotMap.end()) {
                  slot = thisSlotIt->second;
               } else {
                  slot = globalSlotIndex;
                  slotMap[thisThreadID] = slot;
                  ++globalSlotIndex;
               }
            }

            BuildAllReaderValues(r, slot);

            // recursive call to check filters and conditionally execute actions
            while (r.Next())
               for (auto &actionPtr : fBookedActions)
                  actionPtr->Run(slot, r.GetCurrentEntry());
         });
      } else {
#endif // R__USE_IMT
         TTreeReader r;
         if (fTree) {
            r.SetTree(fTree);
         } else {
            r.SetTree(fTreeName.c_str(), fDirPtr);
         }

         CreateSlots(1);
         BuildAllReaderValues(r, 0);

         // recursive call to check filters and conditionally execute actions
         while (r.Next())
            for (auto &actionPtr : fBookedActions)
               actionPtr->Run(0, r.GetCurrentEntry());
#ifdef R__USE_IMT
      }
#endif // R__USE_IMT

      // forget actions and "detach" the action result pointers marking them ready and forget them too
      fBookedActions.clear();
      for (auto readiness : fResPtrsReadiness) {
         *readiness.get() = true;
      }
      fResPtrsReadiness.clear();
   }

   // build reader values for all actions, filters and branches
   void BuildAllReaderValues(TTreeReader &r, unsigned int slot)
   {
      for (auto &ptr : fBookedActions) ptr->BuildReaderValues(r, slot);
      for (auto &ptr : fBookedFilters) ptr->BuildReaderValues(r, slot);
      for (auto &bookedBranch : fBookedBranches) bookedBranch.second->BuildReaderValues(r, slot);
   }

   // inform all actions filters and branches of the required number of slots
   void CreateSlots(unsigned int nSlots)
   {
      for (auto &ptr : fBookedActions) ptr->CreateSlots(nSlots);
      for (auto &ptr : fBookedFilters) ptr->CreateSlots(nSlots);
      for (auto &bookedBranch : fBookedBranches) bookedBranch.second->CreateSlots(nSlots);
   }

   std::weak_ptr<ROOT::Detail::TDataFrameImpl> GetDataFrame() const { return fFirstData; }

   const BranchNames &GetDefaultBranches() const { return fDefaultBranches; }

   const BranchNames GetTmpBranches() const { return fTmpBranches; }

   TTree* GetTree() const {
      if (fTree) {
         return fTree;
      } else {
         auto treePtr = static_cast<TTree*>(fDirPtr->Get(fTreeName.c_str()));
         return treePtr;
      }
   }

   const TDataFrameBranchBase &GetBookedBranch(const std::string &name) const
   {
      return *fBookedBranches.find(name)->second.get();
   }

   void *GetTmpBranchValue(const std::string &branch, unsigned int slot, int entry)
   {
      return fBookedBranches.at(branch)->GetValue(slot, entry);
   }

   TDirectory *GetDirectory() const { return fDirPtr; }

   std::string GetTreeName() const { return fTreeName; }

   void SetFirstData(const std::shared_ptr<TDataFrameImpl>& sp) { fFirstData = sp; }

   void Book(Internal::ActionBasePtr_t actionPtr) { fBookedActions.emplace_back(actionPtr); }

   void Book(ROOT::Detail::FilterBasePtr_t filterPtr) { fBookedFilters.emplace_back(filterPtr); }

   void Book(TmpBranchBasePtr_t branchPtr) { fBookedBranches[branchPtr->GetName()] = branchPtr; }

   // dummy call, end of recursive chain of calls
   bool CheckFilters(int, unsigned int) { return true; }

   unsigned int GetNSlots() {return fNSlots;}

   template<typename T>
   Experimental::TActionResultProxy<T> MakeActionResultPtr(std::shared_ptr<T> r)
   {
      auto readiness = std::make_shared<bool>(false);
      // since fFirstData is a weak_ptr to `this`, we are sure the lock succeeds
      auto df = fFirstData.lock();
      auto resPtr = Experimental::TActionResultProxy<T>::MakeActionResultPtr(r, readiness, df);
      fResPtrsReadiness.emplace_back(readiness);
      return resPtr;
   }
};

} // end NS ROOT::Detail

} // end NS ROOT

// Functions and method implementations
namespace ROOT {

namespace Experimental {

template <typename T>
TDataFrameInterface<T>::TDataFrameInterface(const std::string &treeName, TDirectory *dirPtr,
                                            const BranchNames &defaultBranches)
   : fProxiedPtr(std::make_shared<ROOT::Detail::TDataFrameImpl>(treeName, dirPtr, defaultBranches))
{
   fProxiedPtr->SetFirstData(fProxiedPtr);
}

template <typename T>
TDataFrameInterface<T>::TDataFrameInterface(TTree &tree, const BranchNames &defaultBranches)
   : fProxiedPtr(std::make_shared<ROOT::Detail::TDataFrameImpl>(tree, defaultBranches))
{
   fProxiedPtr->SetFirstData(fProxiedPtr);
}

template<typename T>
void Experimental::TActionResultProxy<T>::TriggerRun()
{
   auto df = fFirstData.lock();
   if (!df) {
      throw std::runtime_error("The main TDataFrame is not reachable: did it go out of scope?");
   }
   df->Run();
}

} // end NS Experimental

namespace Internal {
template <int S, typename T>
T &GetBranchValue(TVBPtr_t &readerValue, unsigned int slot, int entry, const std::string &branch,
                  std::weak_ptr<ROOT::Detail::TDataFrameImpl> df)
{
   if (readerValue == nullptr) {
      // temporary branch
      void *tmpBranchVal = df.lock()->GetTmpBranchValue(branch, slot, entry);
      return *static_cast<T *>(tmpBranchVal);
   } else {
      // real branch
      return **std::static_pointer_cast<TTreeReaderValue<T>>(readerValue);
   }
}

} // end NS Internal

} // end NS ROOT

// FIXME: need to rethink the printfunction

#endif // ROOT_TDATAFRAME
