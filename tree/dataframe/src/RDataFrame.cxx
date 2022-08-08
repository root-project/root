// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RStringView.hxx"
#include "TChain.h"
#include "TDirectory.h"
#include "RtypesCore.h" // for ULong64_t
#include "TTree.h"

#include <memory>  // for make_shared, allocator, shared_ptr
#include <ostream> // ostringstream
#include <stdexcept>
#include <string>
#include <vector>

// clang-format off
/**
* \class ROOT::RDataFrame
* \ingroup dataframe
* \brief ROOT's RDataFrame offers a modern, high-level interface for analysis of data stored in TTree , CSV and other data formats, in C++ or Python.

In addition, multi-threading and other low-level optimisations allow users to exploit all the resources available
on their machines completely transparently.<br>
Skip to the [class reference](#reference) or keep reading for the user guide.

In a nutshell:
~~~{.cpp}
ROOT::EnableImplicitMT(); // Tell ROOT you want to go parallel
ROOT::RDataFrame d("myTree", "file_*.root"); // Interface to TTree and TChain
auto myHisto = d.Histo1D("Branch_A"); // This books the (lazy) filling of a histogram
myHisto->Draw(); // Event loop is run here, upon first access to a result
~~~

Calculations are expressed in terms of a type-safe *functional chain of actions and transformations*, RDataFrame takes
care of their execution. The implementation automatically puts in place several low level optimisations such as
multi-thread parallelization and caching.

\htmlonly
<a href="https://doi.org/10.5281/zenodo.260230"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.260230.svg"
alt="DOI"></a>
\endhtmlonly

## For the impatient user
You can directly see RDataFrame in action in our [tutorials](https://root.cern.ch/doc/master/group__tutorial__dataframe.html), in C++ or Python.

## Table of Contents
- [Cheat sheet](\ref cheatsheet)
- [Introduction](\ref introduction)
- [Crash course](\ref crash-course)
- [Working with collections](\ref collections)
- [Transformations: manipulating data](\ref transformations)
- [Actions: getting results](\ref actions)
- [Distributed execution in Python](\ref distrdf)
- [Performance tips and parallel execution](\ref parallel-execution)
- [More features](\ref more-features)
   - [Systematic variations](\ref systematics)
   - [RDataFrame objects as function arguments and return values](\ref rnode)
   - [Storing RDataFrame objects in collections](\ref RDFCollections)
   - [Executing callbacks every N events](\ref callbacks)
   - [Default column lists](\ref default-branches)
   - [Special helper columns: `rdfentry_` and `rdfslot_`](\ref helper-cols)
   - [Just-in-time compilation: column type inference and explicit declaration of column types](\ref jitting)
   - [User-defined custom actions](\ref generic-actions)
   - [Friend trees](\ref friends)
   - [Reading data formats other than ROOT trees](\ref other-file-formats)
   - [Computation graphs (storing and reusing sets of transformations)](\ref callgraphs)
   - [Visualizing the computation graph](\ref representgraph)
   - [Activating RDataFrame execution logs](\ref rdf-logging)
- [Efficient analysis in Python](\ref python)
- <a class="el" href="classROOT_1_1RDataFrame.html#reference" onclick="javascript:toggleInherit('pub_methods_classROOT_1_1RDF_1_1RInterface')">Class reference</a>

\anchor cheatsheet
## Cheat sheet
These are the operations which can be performed with RDataFrame.

### Transformations
Transformations are a way to manipulate the data.

| **Transformation** | **Description** |
|------------------|--------------------|
| Alias() | Introduce an alias for a particular column name. |
| Define() | Create a new column in the dataset. Example usages include adding a column that contains the invariant mass of a particle, or a selection of elements of an array (e.g. only the `pt`s of "good" muons). |
| DefinePerSample() | Define a new column that is updated when the input sample changes, e.g. when switching tree being processed in a chain. |
| DefineSlot() | Same as Define(), but the user-defined function must take an extra `unsigned int slot` as its first parameter. `slot` will take a different value, `0` to `nThreads - 1`, for each thread of execution. This is meant as a helper in writing thread-safe Define() transformation when using RDataFrame after ROOT::EnableImplicitMT(). DefineSlot() works just as well with single-thread execution: in that case `slot` will always be `0`.  |
| DefineSlotEntry() | Same as DefineSlot(), but the entry number is passed in addition to the slot number. This is meant as a helper in case the expression depends on the entry number. For details about entry numbers in multi-threaded runs, see [here](\ref helper-cols). |
| Filter() | Filter rows based on user-defined conditions. |
| Range() | Filter rows based on entry number (single-thread only). |
| Redefine() | Overwrite the value and/or type of an existing column. See Define() for more information. |
| RedefineSlot() | Overwrite the value and/or type of an existing column. See DefineSlot() for more information. |
| RedefineSlotEntry() | Overwrite the value and/or type of an existing column. See DefineSlotEntry() for more information. |
| Vary() | Register systematic variations for an existing column. Varied results are then extracted via VariationsFor(). |


### Actions
Actions aggregate data into a result. Each one is described in more detail in the reference guide.

In the following, whenever we say an action "returns" something, we always mean it returns a smart pointer to it. Actions only act on events that pass all preceding filters.

Lazy actions only trigger the event loop when one of the results is accessed for the first time, making it easy to
produce many different results in one event loop. Instant actions trigger the event loop instantly.


| **Lazy action** | **Description** |
|------------------|-----------------|
| Aggregate() | Execute a user-defined accumulation operation on the processed column values. |
| Book() | Book execution of a custom action using a user-defined helper object. |
| Cache() | Cache column values in memory. Custom columns can be cached as well, filtered entries are not cached. Users can specify which columns to save (default is all). |
| Count() | Return the number of events processed. Useful e.g. to get a quick count of the number of events passing a Filter. |
| Display() | Provides a printable representation of the dataset contents. The method returns a ROOT::RDF::RDisplay() instance which can print a tabular representation of the data or return it as a string. |
| Fill() | Fill a user-defined object with the values of the specified columns, as if by calling `Obj.Fill(col1, col2, ...)`. |
| Graph() | Fills a TGraph with the two columns provided. If multi-threading is enabled, the order of the points may not be the one expected, it is therefore suggested to sort if before drawing. |
| GraphAsymmErrors() | Fills a TGraphAsymmErrors. If multi-threading is enabled, the order of the points may not be the one expected, it is therefore suggested to sort if before drawing. |
| Histo1D(), Histo2D(), Histo3D() | Fill a one-, two-, three-dimensional histogram with the processed column values. |
| HistoND() | Fill an N-dimensional histogram with the processed column values. |
| Max() | Return the maximum of processed column values. If the type of the column is inferred, the return type is `double`, the type of the column otherwise.|
| Mean() | Return the mean of processed column values.|
| Min() | Return the minimum of processed column values. If the type of the column is inferred, the return type is `double`, the type of the column otherwise.|
| Profile1D(), Profile2D() | Fill a one- or two-dimensional profile with the column values that passed all filters. |
| Reduce() | Reduce (e.g. sum, merge) entries using the function (lambda, functor...) passed as argument. The function must have signature `T(T,T)` where `T` is the type of the column. Return the final result of the reduction operation. An optional parameter allows initialization of the result object to non-default values. |
| Report() | Obtain statistics on how many entries have been accepted and rejected by the filters. See the section on [named filters](#named-filters-and-cutflow-reports) for a more detailed explanation. The method returns a ROOT::RDF::RCutFlowReport instance which can be queried programmatically to get information about the effects of the individual cuts. |
| Stats() | Return a TStatistic object filled with the input columns. |
| StdDev() | Return the unbiased standard deviation of the processed column values. |
| Sum() | Return the sum of the values in the column. If the type of the column is inferred, the return type is `double`, the type of the column otherwise. |
| Take() | Extract a column from the dataset as a collection of values, e.g. a `std::vector<float>` for a column of type `float`. |

| **Instant action** | **Description** |
|---------------------|-----------------|
| Foreach() | Execute a user-defined function on each entry. Users are responsible for the thread-safety of this callable when executing with implicit multi-threading enabled. |
| ForeachSlot() | Same as Foreach(), but the user-defined function must take an extra `unsigned int slot` as its first parameter. `slot` will take a different value, `0` to `nThreads - 1`, for each thread of execution. This is meant as a helper in writing thread-safe Foreach() actions when using RDataFrame after ROOT::EnableImplicitMT(). ForeachSlot() works just as well with single-thread execution: in that case `slot` will always be `0`. |
| Snapshot() | Write the processed dataset to disk, in a new TTree and TFile. Custom columns can be saved as well, filtered entries are not saved. Users can specify which columns to save (default is all). Snapshot, by default, overwrites the output file if it already exists. Snapshot() can be made *lazy* setting the appropriate flag in the snapshot options.|


### Queries

These operations do not modify the dataframe or book computations but simply return information on the RDataFrame object.

| **Operation** | **Description** |
|---------------------|-----------------|
| Describe() | Get useful information describing the dataframe, e.g. columns and their types. |
| GetColumnNames() | Get the names of all the available columns of the dataset. |
| GetColumnType() | Return the type of a given column as a string. |
| GetColumnTypeNamesList() | Return the list of type names of columns in the dataset. |
| GetDefinedColumnNames() | Get the names of all the defined columns. |
| GetFilterNames() | Return the names of all filters in the computation graph. |
| GetNRuns() | Return the number of event loops run by this RDataFrame instance so far. |
| GetNSlots() | Return the number of processing slots that RDataFrame will use during the event loop (i.e. the concurrency level). |
| SaveGraph() | Store the computation graph of an RDataFrame in [DOT format (graphviz)](https://en.wikipedia.org/wiki/DOT_(graph_description_language)) for easy inspection. See the [relevant section](\ref representgraph) for details. |

\anchor introduction
## Introduction
Users define their analysis as a sequence of operations to be performed on the dataframe object; the framework
takes care of the management of the loop over entries as well as low-level details such as I/O and parallelization.
RDataFrame provides methods to perform most common operations required by ROOT analyses;
at the same time, users can just as easily specify custom code that will be executed in the event loop.

RDataFrame is built with a *modular* and *flexible* workflow in mind, summarised as follows:

1. Construct a dataframe object by specifying a dataset. RDataFrame supports TTree as well as TChain, [CSV files](https://root.cern/doc/master/df014__CSVDataSource_8C.html), [SQLite files](https://root.cern/doc/master/df027__SQliteDependencyOverVersion_8C.html), [RNTuples](https://root.cern/doc/master/structROOT_1_1Experimental_1_1RNTuple.html), and it can be extended to custom data formats. From Python, [NumPy arrays can be imported into RDataFrame](https://root.cern/doc/master/df032__MakeNumpyDataFrame_8py.html) as well.

2. Transform the dataframe by:

   - [Applying filters](https://root.cern/doc/master/classROOT_1_1RDataFrame.html#transformations). This selects only specific rows of the dataset.

   - [Creating custom columns](https://root.cern/doc/master/classROOT_1_1RDataFrame.html#transformations). Custom columns can, for example, contain the results of a computation that must be performed for every row of the dataset.

3. [Produce results](https://root.cern/doc/master/classROOT_1_1RDataFrame.html#actions). *Actions* are used to aggregate data into results. Most actions are *lazy*, i.e. they are not executed on the spot, but registered with RDataFrame and executed only when a result is accessed for the first time.

Make sure to book all transformations and actions before you access the contents of any of the results. This lets RDataFrame accumulate work and then produce all results at the same time, upon first access to any of them.

The following table shows how analyses based on TTreeReader and TTree::Draw() translate to RDataFrame. Follow the
[crash course](#crash-course) to discover more idiomatic and flexible ways to express analyses with RDataFrame.
<table>
<tr>
   <td>
      <b>TTreeReader</b>
   </td>
   <td>
      <b>ROOT::RDataFrame</b>
   </td>
</tr>
<tr>
   <td>
~~~{.cpp}
TTreeReader reader("myTree", file);
TTreeReaderValue<A_t> a(reader, "A");
TTreeReaderValue<B_t> b(reader, "B");
TTreeReaderValue<C_t> c(reader, "C");
while(reader.Next()) {
   if(IsGoodEvent(*a, *b, *c))
      DoStuff(*a, *b, *c);
}
~~~
   </td>
   <td>
~~~{.cpp}
ROOT::RDataFrame d("myTree", file, {"A", "B", "C"});
d.Filter(IsGoodEvent).Foreach(DoStuff);
~~~
   </td>
</tr>
<tr>
   <td>
      <b>TTree::Draw</b>
   </td>
   <td>
      <b>ROOT::RDataFrame</b>
   </td>
</tr>
<tr>
   <td>
~~~{.cpp}
auto *tree = file->Get<TTree>("myTree");
tree->Draw("x", "y > 2");
~~~
   </td>
   <td>
~~~{.cpp}
ROOT::RDataFrame df("myTree", file);
auto h = df.Filter("y > 2").Histo1D("x");
h->Draw()
~~~
   </td>
</tr>
<tr>
   <td>
~~~{.cpp}
tree->Draw("jet_eta", "weight*(event == 1)");
~~~
   </td>
   <td>
~~~{.cpp}
df.Filter("event == 1").Histo1D("jet_eta", "weight");
// or the fully compiled version:
df.Filter([] (ULong64_t e) { return e == 1; }, {"event"}).Histo1D<RVec<float>>("jet_eta", "weight");
~~~
   </td>
</tr>
<tr>
   <td>
~~~{cpp}
// object selection: for each event, fill histogram with array of selected pts
tree->Draw('Muon_pt', 'Muon_pt > 100')
~~~
   </td>
   <td>
~~~{cpp}
// with RDF, arrays are read as ROOT::VecOps::RVec objects
df.Define("good_pt", "Muon_pt[Muon_pt > 100]").Histo1D("good_pt")
~~~
   </td>
</tr>
</table>

\anchor crash-course
## Crash course
All snippets of code presented in the crash course can be executed in the ROOT interpreter. Simply precede them with
~~~{.cpp}
using namespace ROOT; // RDataFrame's namespace
~~~
which is omitted for brevity. The terms "column" and "branch" are used interchangeably.

### Creating an RDataFrame
RDataFrame's constructor is where the user specifies the dataset and, optionally, a default set of columns that
operations should work with. Here are the most common methods to construct an RDataFrame object:
~~~{.cpp}
// single file -- all constructors are equivalent
TFile *f = TFile::Open("file.root");
TTree *t = f.Get<TTree>("treeName");

RDataFrame d1("treeName", "file.root");
RDataFrame d2("treeName", f); // same as TTreeReader
RDataFrame d3(*t);

// multiple files -- all constructors are equivalent
TChain chain("myTree");
chain.Add("file1.root");
chain.Add("file2.root");

RDataFrame d4("myTree", {"file1.root", "file2.root"});
std::vector<std::string> files = {"file1.root", "file2.root"};
RDataFrame d5("myTree", files);
RDataFrame d6("myTree", "file*.root"); // the glob is passed as-is to TChain's constructor
RDataFrame d7(chain);
~~~
Additionally, users can construct an RDataFrame with no data source by passing an integer number. This is the number of rows that
will be generated by this RDataFrame.
~~~{.cpp}
RDataFrame d(10); // a RDF with 10 entries (and no columns/branches, for now)
d.Foreach([] { static int i = 0; std::cout << i++ << std::endl; }); // silly example usage: count to ten
~~~
This is useful to generate simple datasets on the fly: the contents of each event can be specified with Define() (explained below). For example, we have used this method to generate [Pythia](https://pythia.org/) events and write them to disk in parallel (with the Snapshot action).

For data sources other than TTrees and TChains, RDataFrame objects are constructed using ad-hoc factory functions (see e.g. MakeCsvDataFrame(), MakeSqliteDataFrame(), MakeArrowDataFrame()):

~~~{.cpp}
auto df = ROOT::RDF::MakeCsvDataFrame("input.csv");
// use df as usual
~~~

### Filling a histogram
Let's now tackle a very common task, filling a histogram:
~~~{.cpp}
// Fill a TH1D with the "MET" branch
RDataFrame d("myTree", "file.root");
auto h = d.Histo1D("MET");
h->Draw();
~~~
The first line creates an RDataFrame associated to the TTree "myTree". This tree has a branch named "MET".

Histo1D() is an *action*; it returns a smart pointer (a ROOT::RDF::RResultPtr, to be precise) to a TH1D histogram filled
with the `MET` of all events. If the quantity stored in the column is a collection (e.g. a vector or an array), the
histogram is filled with all vector elements for each event.

You can use the objects returned by actions as if they were pointers to the desired results. There are many other
possible [actions](\ref cheatsheet), and all their results are wrapped in smart pointers; we'll see why in a minute.

### Applying a filter
Let's say we want to cut over the value of branch "MET" and count how many events pass this cut. This is one way to do it:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto c = d.Filter("MET > 4.").Count(); // computations booked, not run
std::cout << *c << std::endl; // computations run here, upon first access to the result
~~~
The filter string (which must contain a valid C++ expression) is applied to the specified columns for each event;
the name and types of the columns are inferred automatically. The string expression is required to return a `bool`
which signals whether the event passes the filter (`true`) or not (`false`).

You can think of your data as "flowing" through the chain of calls, being transformed, filtered and finally used to
perform actions. Multiple Filter() calls can be chained one after another.

Using string filters is nice for simple things, but they are limited to specifying the equivalent of a single return
statement or the body of a lambda, so it's cumbersome to use strings with more complex filters. They also add a small
runtime overhead, as ROOT needs to just-in-time compile the string into C++ code. When more freedom is required or
runtime performance is very important, a C++ callable can be specified instead (a lambda in the following snippet,
but it can be any kind of function or even a functor class), together with a list of column names.
This snippet is analogous to the one above:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto metCut = [](double x) { return x > 4.; }; // a C++11 lambda function checking "x > 4"
auto c = d.Filter(metCut, {"MET"}).Count();
std::cout << *c << std::endl;
~~~

An example of a more complex filter expressed as a string containing C++ code is shown below

~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto df = d.Define("p", "std::array<double, 4> p{px, py, pz}; return p;")
           .Filter("double p2 = 0.0; for (auto&& x : p) p2 += x*x; return sqrt(p2) < 10.0;");
~~~

The code snippet above defines a column `p` that is a fixed-size array using the component column names and then
filters on its magnitude by looping over its elements. It must be noted that the usage of strings to define columns
like the one above is currently the only possibility when using PyROOT. When writing expressions as such, only constants
and data coming from other columns in the dataset can be involved in the code passed as a string. Local variables and
functions cannot be used, since the interpreter will not know how to find them. When capturing local state is necessary,
it must first be declared to the ROOT C++ interpreter.

More information on filters and how to use them to automatically generate cutflow reports can be found [below](#Filters).

### Defining custom columns
Let's now consider the case in which "myTree" contains two quantities "x" and "y", but our analysis relies on a derived
quantity `z = sqrt(x*x + y*y)`. Using the Define() transformation, we can create a new column in the dataset containing
the variable "z":
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto sqrtSum = [](double x, double y) { return sqrt(x*x + y*y); };
auto zMean = d.Define("z", sqrtSum, {"x","y"}).Mean("z");
std::cout << *zMean << std::endl;
~~~
Define() creates the variable "z" by applying `sqrtSum` to "x" and "y". Later in the chain of calls we refer to
variables created with Define() as if they were actual tree branches/columns, but they are evaluated on demand, at most
once per event. As with filters, Define() calls can be chained with other transformations to create multiple custom
columns. Define() and Filter() transformations can be concatenated and intermixed at will.

As with filters, it is possible to specify new columns as string expressions. This snippet is analogous to the one above:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto zMean = d.Define("z", "sqrt(x*x + y*y)").Mean("z");
std::cout << *zMean << std::endl;
~~~

Again the names of the columns used in the expression and their types are inferred automatically. The string must be
valid C++ and it is just-in-time compiled. The process has a small runtime overhead and like with filters it is currently the only possible approach when using PyROOT.

Previously, when showing the different ways an RDataFrame can be created, we showed a constructor that takes a
number of entries as a parameter. In the following example we show how to combine such an "empty" RDataFrame with Define()
transformations to create a dataset on the fly. We then save the generated data on disk using the Snapshot() action.
~~~{.cpp}
RDataFrame d(100); // an RDF that will generate 100 entries (currently empty)
int x = -1;
auto d_with_columns = d.Define("x", [&x] { return ++x; })
                       .Define("xx", [&x] { return x*x; });
d_with_columns.Snapshot("myNewTree", "newfile.root");
~~~
This example is slightly more advanced than what we have seen so far. First, it makes use of lambda captures (a
simple way to make external variables available inside the body of C++ lambdas) to act on the same variable `x` from
both Define() transformations. Second, we have *stored* the transformed dataframe in a variable. This is always
possible, since at each point of the transformation chain users can store the status of the dataframe for further use (more
on this [below](#callgraphs)).

You can read more about defining new columns [here](#custom-columns).

\image html RDF_Graph.png "A graph composed of two branches, one starting with a filter and one with a define. The end point of a branch is always an action."


### Running on a range of entries
It is sometimes necessary to limit the processing of the dataset to a range of entries. For this reason, the RDataFrame
offers the concept of ranges as a node of the RDataFrame chain of transformations; this means that filters, columns and
actions can be concatenated to and intermixed with Range()s. If a range is specified after a filter, the range will act
exclusively on the entries passing the filter -- it will not even count the other entries! The same goes for a Range()
hanging from another Range(). Here are some commented examples:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
// Here we store a dataframe that loops over only the first 30 entries in a variable
auto d30 = d.Range(30);
// This is how you pick all entries from 15 onwards
auto d15on = d.Range(15, 0);
// We can specify a stride too, in this case we pick an event every 3
auto d15each3 = d.Range(0, 15, 3);
~~~
Note that ranges are not available when multi-threading is enabled. More information on ranges is available
[here](#ranges).

### Executing multiple actions in the same event loop
As a final example let us apply two different cuts on branch "MET" and fill two different histograms with the "pt_v" of
the filtered events.
By now, you should be able to easily understand what is happening:
~~~{.cpp}
RDataFrame d("treeName", "file.root");
auto h1 = d.Filter("MET > 10").Histo1D("pt_v");
auto h2 = d.Histo1D("pt_v");
h1->Draw();       // event loop is run once here
h2->Draw("SAME"); // no need to run the event loop again
~~~
RDataFrame executes all above actions by **running the event-loop only once**. The trick is that actions are not
executed at the moment they are called, but they are **lazy**, i.e. delayed until the moment one of their results is
accessed through the smart pointer. At that time, the event loop is triggered and *all* results are produced
simultaneously.

It is therefore good practice to declare all your transformations and actions *before* accessing their results, allowing
RDataFrame to run the loop once and produce all results in one go.

### Going parallel
Let's say we would like to run the previous examples in parallel on several cores, dividing events fairly between cores.
The only modification required to the snippets would be the addition of this line *before* constructing the main
dataframe object:
~~~{.cpp}
ROOT::EnableImplicitMT();
~~~
Simple as that. More details are given [below](#parallel-execution).

\anchor collections
## Working with collections and object selections

RDataFrame reads collections as the special type [ROOT::RVec](https://root.cern/doc/master/classROOT_1_1VecOps_1_1RVec.html): for example, a column containing an array of floating point numbers can be read as a ROOT::RVecF. C-style arrays (with variable or static size), STL vectors and most other collection types can be read this way.

RVec is a container similar to std::vector (and can be used just like a std::vector) but it also offers a rich interface to operate on the array elements in a vectorised fashion, similarly to Python's NumPy arrays.

For example, to fill a histogram with the "pt" of selected particles for each event, Define() can be used to create a column that contains the desired array elements as follows:

~~~{.cpp}
// h is filled with all the elements of `good_pts`, for each event
auto h = df.Define("good_pts", [](const ROOT::RVecF &pt) { return pt[pt > 0]; })
           .Histo1D("good_pts");
~~~

And in Python:

~~~{.py}
h = df.Define("good_pts", "pt[pt > 0]").Histo1D("good_pts")
~~~

Learn more at ROOT::VecOps::RVec.

\anchor transformations
## Transformations: manipulating data
\anchor Filters
### Filters
A filter is created through a call to `Filter(f, columnList)` or `Filter(filterString)`. In the first overload, `f` can
be a function, a lambda expression, a functor class, or any other callable object. It must return a `bool` signalling
whether the event has passed the selection (`true`) or not (`false`). It should perform "read-only" operations on the
columns, and should not have side-effects (e.g. modification of an external or static variable) to ensure correctness
when implicit multi-threading is active. The second overload takes a string with a valid C++ expression in which column
names are used as variable names (e.g. `Filter("x[0] + x[1] > 0")`). This is a convenience feature that comes with a
certain runtime overhead: C++ code has to be generated on the fly from this expression before using it in the event
loop. See the paragraph about "Just-in-time compilation" below for more information.

RDataFrame only evaluates filters when necessary: if multiple filters are chained one after another, they are executed
in order and the first one returning `false` causes the event to be discarded and triggers the processing of the next
entry. If multiple actions or transformations depend on the same filter, that filter is not executed multiple times for
each entry: after the first access it simply serves a cached result.

\anchor named-filters-and-cutflow-reports
#### Named filters and cutflow reports
An optional string parameter `name` can be passed to the Filter() method to create a **named filter**. Named filters
work as usual, but also keep track of how many entries they accept and reject.

Statistics are retrieved through a call to the Report() method:

- when Report() is called on the main RDataFrame object, it returns a ROOT::RDF::RResultPtr<RCutFlowReport> relative to all
named filters declared up to that point
- when called on a specific node (e.g. the result of a Define() or Filter()), it returns a ROOT::RDF::RResultPtr<RCutFlowReport>
relative all named filters in the section of the chain between the main RDataFrame and that node (included).

Stats are stored in the same order as named filters have been added to the graph, and *refer to the latest event-loop*
that has been run using the relevant RDataFrame.

\anchor ranges
### Ranges
When RDataFrame is not being used in a multi-thread environment (i.e. no call to EnableImplicitMT() was made),
Range() transformations are available. These act very much like filters but instead of basing their decision on
a filter expression, they rely on `begin`,`end` and `stride` parameters.

- `begin`: initial entry number considered for this range.
- `end`: final entry number (excluded) considered for this range. 0 means that the range goes until the end of the dataset.
- `stride`: process one entry of the [begin, end) range every `stride` entries. Must be strictly greater than 0.

The actual number of entries processed downstream of a Range() node will be `(end - begin)/stride` (or less if less
entries than that are available).

Note that ranges act "locally", not based on the global entry count: `Range(10,50)` means "skip the first 10 entries
*that reach this node*, let the next 40 entries pass, then stop processing". If a range node hangs from a filter node,
and the range has a `begin` parameter of 10, that means the range will skip the first 10 entries *that pass the
preceding filter*.

Ranges allow "early quitting": if all branches of execution of a functional graph reached their `end` value of
processed entries, the event-loop is immediately interrupted. This is useful for debugging and quick data explorations.

\anchor custom-columns
### Custom columns
Custom columns are created by invoking `Define(name, f, columnList)`. As usual, `f` can be any callable object
(function, lambda expression, functor class...); it takes the values of the columns listed in `columnList` (a list of
strings) as parameters, in the same order as they are listed in `columnList`. `f` must return the value that will be
assigned to the temporary column.

A new variable is created called `name`, accessible as if it was contained in the dataset from subsequent
transformations/actions.

Use cases include:
- caching the results of complex calculations for easy and efficient multiple access
- extraction of quantities of interest from complex objects
- branch aliasing, i.e. changing the name of a branch

An exception is thrown if the `name` of the new column/branch is already in use for another branch in the TTree.

It is also possible to specify the quantity to be stored in the new temporary column as a C++ expression with the method
`Define(name, expression)`. For example this invocation

~~~{.cpp}
df.Define("pt", "sqrt(px*px + py*py)");
~~~

will create a new column called "pt" the value of which is calculated starting from the columns px and py. The system
builds a just-in-time compiled function starting from the expression after having deduced the list of necessary branches
from the names of the variables specified by the user.

#### Custom columns as function of slot and entry number

It is possible to create custom columns also as a function of the processing slot and entry numbers. The methods that can
be invoked are:
- `DefineSlot(name, f, columnList)`. In this case the callable f has this signature `R(unsigned int, T1, T2, ...)`: the
first parameter is the slot number which ranges from 0 to ROOT::GetThreadPoolSize() - 1.
- `DefineSlotEntry(name, f, columnList)`. In this case the callable f has this signature `R(unsigned int, ULong64_t,
T1, T2, ...)`: the first parameter is the slot number while the second one the number of the entry being processed.

\anchor actions
## Actions: getting results
### Instant and lazy actions
Actions can be **instant** or **lazy**. Instant actions are executed as soon as they are called, while lazy actions are
executed whenever the object they return is accessed for the first time. As a rule of thumb, actions with a return value
are lazy, the others are instant.

\anchor distrdf
## Distributed execution

RDataFrame applications can be executed in parallel through distributed computing frameworks on a set of remote machines
thanks to the Python package `ROOT.RDF.Experimental.Distributed`. This experimental, **Python-only** package allows to scale the
optimized performance RDataFrame can achieve on a single machine to multiple nodes at the same time. It is designed so
that different backends can be easily plugged in, currently supporting [Apache Spark](http://spark.apache.org/) and
[Dask](https://dask.org/). To make use of distributed RDataFrame, you only need to switch `ROOT.RDataFrame` with
the backend-specific `RDataFrame` of your choice, for example:

~~~{.py}
import ROOT

# Point RDataFrame calls to the Spark specific RDataFrame
RDataFrame = ROOT.RDF.Experimental.Distributed.Spark.RDataFrame

# It still accepts the same constructor arguments as traditional RDataFrame
df = RDataFrame("mytree", "myfile.root")

# Continue the application with the traditional RDataFrame API
sum = df.Filter("x > 10").Sum("y")
h = df.Histo1D(("name", "title", 10, 0, 10), "x")

print(sum.GetValue())
h.Draw()
~~~

The main goal of this package is to support running any RDataFrame application distributedly. Nonetheless, not all
RDataFrame operations currently work with this package. The subset that is currently available is:
- AsNumpy
- Count
- Define
- DefinePerSample
- Fill
- Filter
- Graph
- Histo[1,2,3]D
- HistoND
- Max
- Mean
- Min
- Profile[1,2,3]D
- Redefine
- Snapshot
- Sum

with support for more operations coming in the future. Data sources other than TTree and TChain (e.g. CSV, RNTuple) are
currently not supported.

**Note** that the distributed RDataFrame module is available in a ROOT installation if the following criteria are met:
- PyROOT is available
- RDataFrame is available
- The version of the Python interpreter used to build ROOT is greater or equal than 3.7

### Connecting to a Spark cluster

In order to distribute the RDataFrame workload, you can connect to a Spark cluster you have access to through the
official [Spark API](https://spark.apache.org/docs/latest/rdd-programming-guide.html#initializing-spark), then hook the
connection instance to the distributed `RDataFrame` object like so:

~~~{.py}
import pyspark
import ROOT

# Create a SparkContext object with the right configuration for your Spark cluster
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)

# Point RDataFrame calls to the Spark specific RDataFrame
RDataFrame = ROOT.RDF.Experimental.Distributed.Spark.RDataFrame

# The Spark RDataFrame constructor accepts an optional "sparkcontext" parameter
# and it will distribute the application to the connected cluster
df = RDataFrame("mytree", "myfile.root", sparkcontext = sc)
~~~

If an instance of [SparkContext](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.html)
is not provided, the default behaviour is to create one in the background for you.

### Connecting to a Dask cluster

Similarly, you can connect to a Dask cluster by creating your own connection object which internally operates with one
of the cluster schedulers supported by Dask (more information in the
[Dask distributed docs](http://distributed.dask.org/en/stable/)):

~~~{.py}
import ROOT
from dask.distributed import Client

# Point RDataFrame calls to the Dask specific RDataFrame
RDataFrame = ROOT.RDF.Experimental.Distributed.Dask.RDataFrame

# In a Python script the Dask client needs to be initalized in a context
# Jupyter notebooks / Python session don't need this
if __name__ == "__main__":
    # With an already setup cluster that exposes a Dask scheduler endpoint
    client = Client("dask_scheduler.domain.com:8786")

    # The Dask RDataFrame constructor accepts the Dask Client object as an optional argument
    df = RDataFrame("mytree","myfile.root", daskclient=client)
    # Proceed as usual
    df.Define("x","someoperation").Histo1D(("name", "title", 10, 0, 10), "x")
~~~

If an instance of [distributed.Client](http://distributed.dask.org/en/stable/api.html#distributed.Client) is not
provided to the RDataFrame object, it will be created for you and it will run the computations in the local machine
using all cores available.

### Choosing the number of distributed tasks

A distributed RDataFrame has internal logic to define in how many chunks the input dataset will be split before sending
tasks to the distributed backend. Each task reads and processes one of said chunks. The logic is backend-dependent, but
generically tries to infer how many cores are available in the cluster through the connection object. The number of
tasks will be equal to the inferred number of cores. There are cases where the connection object of the chosen backend
doesn't have information about the actual resources of the cluster. An example of this is when using Dask to connect to
a batch system. The client object created at the beginning of the application does not automatically know how many cores
will be available during distributed execution, since the jobs are submitted to the batch system after the creation of
the connection. In such cases, the logic is to default to process the whole dataset in 2 tasks.

The number of tasks submitted for distributed execution can be also set programmatically, by providing the optional
keyword argument `npartitions` when creating the RDataFrame object. This parameter is accepted irrespectively of the
backend used:

~~~{.py}
import ROOT

# Define correct imports and access the distributed RDataFrame appropriate for the
# backend used in the analysis
RDataFrame = ROOT.RDF.Experimental.Distributed.[BACKEND].RDataFrame

if __name__ == "__main__":
    # The `npartitions` optional argument tells the RDataFrame how many tasks are desired
    df = RDataFrame("mytree","myfile.root", npartitions=NPARTITIONS)
    # Proceed as usual
    df.Define("x","someoperation").Histo1D(("name", "title", 10, 0, 10), "x")
~~~

Note that when processing a TTree or TChain dataset, the `npartitions` value should not exceed the number of clusters in
the dataset. The number of clusters in a TTree can be retrieved by typing `rootls -lt myfile.root` at a command line.

### Distributed Snapshot

The Snapshot operation behaves slightly differently when executed distributedly. First off, it requires the path
supplied to the Snapshot call to be accessible from any worker of the cluster and from the client machine (in general
it should be provided as an absolute path). Another important difference is that `n` separate files will be produced,
where `n` is the number of dataset partitions. As with local RDataFrame, the result of a Snapshot on a distributed
RDataFrame is another distributed RDataFrame on which we can define a new computation graph and run more distributed
computations.

### Distributed RunGraphs

Submitting multiple distributed RDataFrame executions is supported through the RunGraphs function. Similarly to its
local counterpart, the function expects an iterable of objects representing an RDataFrame action. Each action will be
triggered concurrently to send multiple computation graphs to a distributed cluster at the same time:

~~~{.py}
import ROOT
RDataFrame = ROOT.RDF.Experimental.Distributed.Dask.RDataFrame
RunGraphs = ROOT.RDF.Experimental.Distributed.RunGraphs

# Create 3 different dataframes and book an histogram on each one
histoproxies = [
   RDataFrame(100)
         .Define("x", "rdfentry_")
         .Histo1D(("name", "title", 10, 0, 100), "x")
   for _ in range(4)
]

# Execute the 3 computation graphs
RunGraphs(histoproxies)
# Retrieve all the histograms in one go
histos = [histoproxy.GetValue() for histoproxy in histoproxies]
~~~

Every distributed backend supports this feature and graphs belonging to different backends can be still triggered with
a single call to RunGraphs (e.g. it is possible to send a Spark job and a Dask job at the same time).

### Histogram models in distributed mode

When calling a Histo*D operation in distributed mode, remember to pass to the function the model of the histogram to be
computed, e.g. the axis range and the number of bins:

~~~{.py}
import ROOT
RDataFrame = ROOT.RDF.Experimental.Distributed.[BACKEND].RDataFrame

if __name__ == "__main__":
    df = RDataFrame("mytree","myfile.root").Define("x","someoperation")
    # The model can be passed either as a tuple with the arguments in the correct order
    df.Histo1D(("name", "title", 10, 0, 10), "x")
    # Or by creating the specific struct
    model = ROOT.RDF.TH1DModel("name", "title", 10, 0, 10)
    df.Histo1D(model, "x")
~~~

Without this, two partial histograms resulting from two distributed tasks would have incompatible binning, thus leading
to errors when merging them. Failing to pass a histogram model will raise an error on the client side, before starting
the distributed execution.


\anchor parallel-execution
## Performance tips and parallel execution
As pointed out before in this document, RDataFrame can transparently perform multi-threaded event loops to speed up
the execution of its actions. Users have to call ROOT::EnableImplicitMT() *before* constructing the RDataFrame
object to indicate that it should take advantage of a pool of worker threads. **Each worker thread processes a distinct
subset of entries**, and their partial results are merged before returning the final values to the user.
There are no guarantees on the order in which threads will process the batches of entries.
In particular, note that this means that, for multi-thread event loops, there is no
guarantee on the order in which Snapshot() will _write_ entries: they could be scrambled with respect to the input dataset. The values of the special `rdfentry_` column will also not correspond to the entry numbers in the input dataset (e.g. TChain) in multi-thread runs.

\warning By default, RDataFrame will use as many threads as the hardware supports, using up **all** the resources on
a machine. This might be undesirable on shared computing resources such as a batch cluster. Therefore, when running on shared computing resources, use
~~~{.cpp}
ROOT::EnableImplicitMT(i)
~~~
replacing `i` with the number of CPUs/slots that were allocated for this job.

### Thread-safety of user-defined expressions
RDataFrame operations such as Histo1D() or Snapshot() are guaranteed to work correctly in multi-thread event loops.
User-defined expressions, such as strings or lambdas passed to Filter(), Define(), Foreach(), Reduce() or Aggregate()
will have to be thread-safe, i.e. it should be possible to call them concurrently from different threads.

Note that simple Filter() and Define() transformations will inherently satisfy this requirement: Filter() / Define()
expressions will often be *pure* in the functional programming sense (no side-effects, no dependency on external state),
which eliminates all risks of race conditions.

In order to facilitate writing of thread-safe operations, some RDataFrame features such as Foreach(), Define() or \link ROOT::RDF::RResultPtr::OnPartialResult OnPartialResult()\endlink
offer thread-aware counterparts (ForeachSlot(), DefineSlot(), \link ROOT::RDF::RResultPtr::OnPartialResultSlot OnPartialResultSlot()\endlink): their only difference is that they
will pass an extra `slot` argument (an unsigned integer) to the user-defined expression. When calling user-defined code
concurrently, RDataFrame guarantees that different threads will employ different values of the `slot` parameter,
where `slot` will be a number between 0 and `GetNSlots() - 1`.
In other words, within a slot, computation runs sequentially and events are processed sequentially.
Note that the same slot might be associated to different threads over the course of a single event loop, but two threads
will never receive the same slot at the same time.
This extra parameter might facilitate writing safe parallel code by having each thread write/modify a different
processing slot, e.g. a different element of a list. See [here](#generic-actions) for an example usage of ForeachSlot().

### Parallel execution of multiple RDataFrame event loops
A complex analysis may require multiple separate RDataFrame computation graphs to produce all desired results. This poses the challenge that the
event loops of each computation graph can be parallelized, but the different loops run sequentially, one after the other.
On many-core architectures it might be desirable to run different event loops concurrently to improve resource usage.
ROOT::RDF::RunGraphs() allows running multiple RDataFrame event loops concurrently:
~~~{.cpp}
ROOT::EnableImplicitMT();
ROOT::RDataFrame df1("tree1", "f1.root");
ROOT::RDataFrame df2("tree2", "f2.root");
auto histo1 = df1.Histo1D("x");
auto histo2 = df2.Histo1D("y");

// just accessing result pointers, the event loops of separate RDataFrames run one after the other
histo1->Draw(); // runs first multi-thread event loop
histo2->Draw(); // runs second multi-thread event loop

// alternatively, with ROOT::RDF::RunGraphs, event loops for separate computation graphs can run concurrently
ROOT::RDF::RunGraphs({histo1, histo2});
histo1->Draw(); // results can then be used as usual
~~~

### Performance considerations

To obtain the maximum performance out of RDataFrame, make sure to avoid just-in-time compiled versions of transformations and actions if at all possible.
For instance, `Filter("x > 0")` requires just-in-time compilation of the corresponding C++ logic, while the equivalent `Filter([](float x) { return x > 0.; }, {"x"})` does not.
Similarly, `Histo1D("x")` requires just-in-time compilation after the type of `x` is retrieved from the dataset, while `Histo1D<float>("x")` does not; the latter spelling
should be preferred for performance-critical applications.

Python applications cannot easily specify template parameters or pass C++ callables to RDataFrame.
See [Efficient analysis in Python](#python) for possible ways to speed up hot paths in this case.

Just-in-time compilation happens once, right before starting an event loop. To reduce the runtime cost of this step, make sure to book all operations *for all RDataFrame computation graphs*
before the first event loop is triggered: just-in-time compilation will happen once for all code required to be generated up to that point, also across different computation graphs.

Also make sure not to count the just-in-time compilation time (which happens once before the event loop and does not depend on the size of the dataset) as part of the event loop runtime (which scales with the size of the dataset). RDataFrame has an experimental logging feature that simplifies measuring the time spent in just-in-time compilation and in the event loop (as well as providing some more interesting information). See [Activating RDataFrame execution logs](\ref rdf-logging).

### Memory usage

There are two reasons why RDataFrame may consume more memory than expected. Firstly, each result is duplicated for each worker thread, which e.g. in case of many (possibly multi-dimensional) histograms with fine binning can result in visible memory consumption during the event loop. The thread-local copies of the results are destroyed when the final result is produced. Reducing the number of threads or using coarser binning will reduce the memory usage.

Secondly, just-in-time compilation of string expressions or non-templated actions (see the previous paragraph) causes Cling, ROOT's C++ interpreter, to allocate some memory for the generated code that is only released at the end of the application. This commonly results in memory usage creep in long-running applications that create many RDataFrames one after the other. Possible mitigations include creating and running each RDataFrame event loop in a sub-process, or booking all operations for all different RDataFrame computation graphs before the first event loop is triggered, so that the interpreter is invoked only once for all computation graphs:

~~~{.cpp}
// assuming df1 and df2 are separate computation graphs, do:
auto h1 = df1.Histo1D("x");
auto h2 = df2.Histo1D("y");
h1->Draw(); // we just-in-time compile everything needed by df1 and df2 here
h2->Draw("SAME");

// do not:
auto h1 = df1.Histo1D("x");
h1->Draw(); // we just-in-time compile here
auto h2 = df2.Histo1D("y");
h2->Draw("SAME"); // we just-in-time compile again here, as the second Histo1D call is new
~~~

\anchor more-features
## More features
Here is a list of the most important features that have been omitted in the "Crash course" for brevity.
You don't need to read all these to start using RDataFrame, but they are useful to save typing time and runtime.

\anchor systematics
### Systematic variations

Starting from ROOT v6.26, RDataFrame provides a flexible syntax to define systematic variations.
This is done in two steps: a) variations for one or more existing columns are registered via Vary() and b) variations
of normal RDataFrame results are extracted with a call to VariationsFor(). In between these steps, no other change
to the analysis code is required: the presence of systematic variations for certain columns is automatically propagated
through filters, defines and actions, and RDataFrame will take these dependencies into account when producing varied
results. VariationsFor() is included in header `ROOT/RDFHelpers.hxx`, which compiled C++ programs must include
explicitly.

An example usage of Vary() and VariationsFor() in C++:

~~~{.cpp}
auto nominal_hx =
   df.Vary("pt", "ROOT::RVecD{pt*0.9f, pt*1.1f}", {"down", "up"})
     .Filter("pt > pt_cut")
     .Define("x", someFunc, {"pt"})
     .Histo1D<float>("x");

// request the generation of varied results from the nominal
ROOT::RDF::Experimental::RResultMap<TH1D> hx = ROOT::RDF::Experimental::VariationsFor(nominal_hx);

// the event loop runs here, upon first access to any of the results or varied results:
hx["nominal"].Draw(); // same effect as nominal_hx->Draw()
hx["pt:down"].Draw("SAME");
hx["pt:up"].Draw("SAME");
~~~

A list of variation "tags" is passed as last argument to Vary(): they give a name to the varied values that are returned
as elements of an RVec of the appropriate type. The number of variation tags must correspond to the number of elements
the RVec returned by the expression (2 in the example above: the first element will correspond to tag "down", the second
to tag "up"). The _full_ variation name will be composed of the varied column name and the variation tags (e.g.
"pt:down", "pt:up" in this example). Python usage looks similar.

Note how we use the "pt" column as usual in the Filter() and Define() calls and we simply use "x" as the value to fill
the resulting histogram. To produce the varied results, RDataFrame will automatically execute the Filter and Define
calls for each variation and fill the histogram with values and cuts that depend on the variation.

There is no limitation to the complexity of a Vary() expression, and just like for Define() and Filter() calls users are
not limited to string expressions but they can also pass any valid C++ callable, including lambda functions and
complex functors. The callable can be applied to zero or more existing columns and it will always receive their
_nominal_ values in input.

**Varying multiple columns in lockstep**

In the following Python snippet we use the Vary() signature that allows varying multiple columns simultaneously or
"in lockstep":

~~~{.python}
df.Vary(["pt", "eta"],
        "RVec<RVecF>{{pt*0.9, pt*1.1}, {eta*0.9, eta*1.1}}",
        variationTags=["down", "up"],
        variationName="ptAndEta")
~~~

The expression returns an RVec of two RVecs: each inner vector contains the varied values for one column, and the
inner vectors follow the same ordering as the column names passed as first argument. Besides the variation tags, in
this case we also have to explicitly pass a variation name as there is no one column name that can be used as default.

The call above will produce variations "ptAndEta:down" and "ptAndEta:up".

**Combining multiple variations**

Even if a result depends on multiple variations, only one is applied at a time, i.e. there will be no result produced
by applying multiple systematic variations at the same time.
For example, in the following example snippet, the RResultMap instance `all_h` will contain keys "nominal", "pt:down",
"pt:up", "eta:0", "eta:1", but no "pt:up&&eta:0" or similar:

~~~{.cpp}
auto df = _df.Vary("pt",
                   "ROOT::RVecD{pt*0.9, pt*1.1}",
                   {"down", "up"})
             .Vary("eta",
                   [](float eta) { return RVecF{eta*0.9f, eta*1.1f}; },
                   {"eta"},
                   2);

auto nom_h = df.Histo2D(histoModel, "pt", "eta");
auto all_hs = VariationsFor(nom_h);
all_hs.GetKeys(); // returns {"nominal", "pt:down", "pt:up", "eta:0", "eta:1"}
~~~

Note how we passed the integer `2` instead of a list of variation tags to the second Vary() invocation: this is a
shorthand that automatically generates tags 0 to N-1 (in this case 0 and 1).

\note As of v6.26, VariationsFor() and RResultMap are in the `ROOT::RDF::Experimental` namespace, to indicate that these
      interfaces might still evolve and improve based on user feedback. We expect that some aspects of the related
      programming model will be streamlined in future versions.

\note As of v6.26, the results of a Snapshot(), Report() or Display() call cannot be varied (i.e. it is not possible to
      call VariationsFor() on them. These limitations will be lifted in future releases.

\anchor rnode
### RDataFrame objects as function arguments and return values
RDataFrame variables/nodes are relatively cheap to copy and it's possible to both pass them to (or move them into)
functions and to return them from functions. However, in general each dataframe node will have a different C++ type,
which includes all available compile-time information about what that node does. One way to cope with this complication
is to use template functions and/or C++14 auto return types:
~~~{.cpp}
template <typename RDF>
auto ApplySomeFilters(RDF df)
{
   return df.Filter("x > 0").Filter([](int y) { return y < 0; }, {"y"});
}
~~~

A possibly simpler, C++11-compatible alternative is to take advantage of the fact that any dataframe node can be
converted (implicitly or via an explicit cast) to the common type ROOT::RDF::RNode:
~~~{.cpp}
// a function that conditionally adds a Range to an RDataFrame node.
RNode MaybeAddRange(RNode df, bool mustAddRange)
{
   return mustAddRange ? df.Range(1) : df;
}
// use as :
ROOT::RDataFrame df(10);
auto maybeRangedDF = MaybeAddRange(df, true);
~~~

The conversion to ROOT::RDF::RNode is cheap, but it will introduce an extra virtual call during the RDataFrame event
loop (in most cases, the resulting performance impact should be negligible). Python users can perform the conversion with the helper function `ROOT.RDF.AsRNode`.

As a final note, remember that RDataFrame actions do not return another dataframe, but a \link ROOT::RDF::RResultPtr ROOT::RDF::RResultPtr<T>\endlink, where T is the
type of the result of the action.

\anchor RDFCollections
### Storing RDataFrame objects in collections

ROOT::RDF::RNode also makes it simple to store RDataFrame nodes in collections, e.g. a `std::vector<RNode>` or a `std::map<std::string, RNode>`:

~~~{.cpp}
std::vector<ROOT::RDF::RNode> dfs;
dfs.emplace_back(ROOT::RDataFrame(10));
dfs.emplace_back(dfs[0].Define("x", "42.f"));
~~~

\anchor callbacks
### Executing callbacks every N events
It's possible to schedule execution of arbitrary functions (callbacks) during the event loop.
Callbacks can be used e.g. to inspect partial results of the analysis while the event loop is running,
drawing a partially-filled histogram every time a certain number of new entries is processed, or
displaying a progress bar while the event loop runs.

For example one can draw an up-to-date version of a result histogram every 100 entries like this:
~~~{.cpp}
auto h = df.Histo1D("x");
TCanvas c("c","x hist");
h.OnPartialResult(100, [&c](TH1D &h_) { c.cd(); h_.Draw(); c.Update(); });
// event loop runs here, this final `Draw` is executed after the event loop is finished
h->Draw();
~~~

Callbacks are registered to a ROOT::RDF::RResultPtr and must be callables that takes a reference to the result type as argument
and return nothing. RDataFrame will invoke registered callbacks passing partial action results as arguments to them
(e.g. a histogram filled with a part of the selected events).

Read more on ROOT::RDF::RResultPtr::OnPartialResult() and ROOT::RDF::RResultPtr::OnPartialResultSlot().

\anchor default-branches
### Default column lists
When constructing an RDataFrame object, it is possible to specify a **default column list** for your analysis, in the
usual form of a list of strings representing branch/column names. The default column list will be used as a fallback
whenever a list specific to the transformation/action is not present. RDataFrame will take as many of these columns as
needed, ignoring trailing extra names if present.
~~~{.cpp}
// use "b1" and "b2" as default columns
RDataFrame d1("myTree", "file.root", {"b1","b2"});
auto h = d1.Filter([](int b1, int b2) { return b1 > b2; }) // will act on "b1" and "b2"
           .Histo1D(); // will act on "b1"

// just one default column this time
RDataFrame d2("myTree", "file.root", {"b1"});
auto min = d2.Filter([](double b2) { return b2 > 0; }, {"b2"}) // we can still specify non-default column lists
             .Min(); // returns the minimum value of "b1" for the filtered entries
~~~

\anchor helper-cols
### Special helper columns: rdfentry_ and rdfslot_
Every instance of RDataFrame is created with two special columns called `rdfentry_` and `rdfslot_`. The `rdfentry_`
column is of type `ULong64_t` and it holds the current entry number while `rdfslot_` is an `unsigned int`
holding the index of the current data processing slot.
For backwards compatibility reasons, the names `tdfentry_` and `tdfslot_` are also accepted.
These columns are ignored by operations such as [Cache](classROOT_1_1RDF_1_1RInterface.html#aaaa0a7bb8eb21315d8daa08c3e25f6c9)
or [Snapshot](classROOT_1_1RDF_1_1RInterface.html#a233b7723e498967f4340705d2c4db7f8).

\warning Note that in multi-thread event loops the values of `rdfentry_` _do not_ correspond to what would be the entry numbers
of a TChain constructed over the same set of ROOT files, as the entries are processed in an unspecified order.

\anchor jitting
### Just-in-time compilation: column type inference and explicit declaration of column types
C++ is a statically typed language: all types must be known at compile-time. This includes the types of the TTree
branches we want to work on. For filters, defined columns and some of the actions, **column types are deduced from the
signature** of the relevant filter function/temporary column expression/action function:
~~~{.cpp}
// here b1 is deduced to be `int` and b2 to be `double`
df.Filter([](int x, double y) { return x > 0 && y < 0.; }, {"b1", "b2"});
~~~
If we specify an incorrect type for one of the columns, an exception with an informative message will be thrown at
runtime, when the column value is actually read from the dataset: RDataFrame detects type mismatches. The same would
happen if we swapped the order of "b1" and "b2" in the column list passed to Filter().

Certain actions, on the other hand, do not take a function as argument (e.g. Histo1D()), so we cannot deduce the type of
the column at compile-time. In this case **RDataFrame infers the type of the column** from the TTree itself. This
is why we never needed to specify the column types for all actions in the above snippets.

When the column type is not a common one such as `int`, `double`, `char` or `float` it is nonetheless good practice to
specify it as a template parameter to the action itself, like this:
~~~{.cpp}
df.Histo1D("b1"); // OK, the type of "b1" is deduced at runtime
df.Min<MyNumber_t>("myObject"); // OK, "myObject" is deduced to be of type `MyNumber_t`
~~~

Deducing types at runtime requires the just-in-time compilation of the relevant actions, which has a small runtime
overhead, so specifying the type of the columns as template parameters to the action is good practice when performance is a goal.

When strings are passed as expressions to Filter() or Define(), fundamental types are passed as constants. This avoids certaincommon mistakes such as typing `x = 0` rather than `x == 0`:

~~~{.cpp}
// this throws an error (note the typo)
df.Define("x", "0").Filter("x = 0");
~~~

\anchor generic-actions
### User-defined custom actions
RDataFrame strives to offer a comprehensive set of standard actions that can be performed on each event. At the same
time, it allows users to inject their own action code to perform arbitrarily complex data reductions.

#### Implementing custom actions with Book()

Through the Book() method, users can implement a custom action and have access to the same features
that built-in RDataFrame actions have, e.g. hooks to events related to the start, end and execution of the
event loop, or the possibility to return a lazy RResultPtr to an arbitrary type of result:

~~~{.cpp}
#include <ROOT/RDataFrame.hxx>
#include <memory>

class MyCounter : public ROOT::Detail::RDF::RActionImpl<MyCounter> {
   std::shared_ptr<int> fFinalResult = std::make_shared<int>(0);
   std::vector<int> fPerThreadResults;

public:
   // We use a public type alias to advertise the type of the result of this action
   using Result_t = int;

   MyCounter(unsigned int nSlots) : fPerThreadResults(nSlots) {}

   // Called before the event loop to retrieve the address of the result that will be filled/generated.
   std::shared_ptr<int> GetResultPtr() const { return fFinalResult; }

   // Called at the beginning of the event loop.
   void Initialize() {}

   // Called at the beginning of each processing task.
   void InitTask(TTreeReader *, int) {}

   /// Called at every entry.
   void Exec(unsigned int slot)
   {
      fPerThreadResults[slot]++;
   }

   // Called at the end of the event loop.
   void Finalize()
   {
      *fFinalResult = std::accumulate(fPerThreadResults.begin(), fPerThreadResults.end(), 0);
   }

   // Called by RDataFrame to retrieve the name of this action.
   std::string GetActionName() const { return "MyCounter"; }
};

int main() {
   ROOT::RDataFrame df(10);
   ROOT::RDF::RResultPtr<int> resultPtr = df.Book<>(MyCounter{df.GetNSlots()}, {});
   // The GetValue call triggers the event loop
   std::cout << "Number of processed entries: " <<  resultPtr.GetValue() << std::endl;
}
~~~

See the Book() method for more information and [this tutorial](https://root.cern/doc/master/df018__customActions_8C.html)
for a more complete example.

#### Injecting arbitrary code in the event loop with Foreach() and ForeachSlot()

Foreach() takes a callable (lambda expression, free function, functor...) and a list of columns and
executes the callable on the values of those columns for each event that passes all upstream selections.
It can be used to perform actions that are not already available in the interface. For example, the following snippet
evaluates the root mean square of column "x":
~~~{.cpp}
// Single-thread evaluation of RMS of column "x" using Foreach
double sumSq = 0.;
unsigned int n = 0;
df.Foreach([&sumSq, &n](double x) { ++n; sumSq += x*x; }, {"x"});
std::cout << "rms of x: " << std::sqrt(sumSq / n) << std::endl;
~~~
In multi-thread runs, users are responsible for the thread-safety of the expression passed to Foreach():
thread will execute the expression concurrently.
The code above would need to employ some resource protection mechanism to ensure non-concurrent writing of `rms`; but
this is probably too much head-scratch for such a simple operation.

ForeachSlot() can help in this situation. It is an alternative version of Foreach() for which the function takes an
additional "processing slot" parameter besides the columns it should be applied to. RDataFrame
guarantees that ForeachSlot() will invoke the user expression with different `slot` parameters for different concurrent
executions (see [Special helper columns: rdfentry_ and rdfslot_](\ref helper-cols) for more information on the slot parameter).
We can take advantage of ForeachSlot() to evaluate a thread-safe root mean square of column "x":
~~~{.cpp}
// Thread-safe evaluation of RMS of column "x" using ForeachSlot
ROOT::EnableImplicitMT();
const unsigned int nSlots = df.GetNSlots();
std::vector<double> sumSqs(nSlots, 0.);
std::vector<unsigned int> ns(nSlots, 0);

df.ForeachSlot([&sumSqs, &ns](unsigned int slot, double x) { sumSqs[slot] += x*x; ns[slot] += 1; }, {"x"});
double sumSq = std::accumulate(sumSqs.begin(), sumSqs.end(), 0.); // sum all squares
unsigned int n = std::accumulate(ns.begin(), ns.end(), 0); // sum all counts
std::cout << "rms of x: " << std::sqrt(sumSq / n) << std::endl;
~~~
Notice how we created one `double` variable for each processing slot and later merged their results via `std::accumulate`.


\anchor friends
### Friend trees
Friend TTrees are supported by RDataFrame.
Friend TTrees with a TTreeIndex are supported starting from ROOT v6.24.

To use friend trees in RDataFrame, it is necessary to add the friends directly to
the tree and instantiate an RDataFrame with the main tree:

~~~{.cpp}
TTree t([...]);
TTree ft([...]);
t.AddFriend(&ft, "myFriend");

RDataFrame d(t);
auto f = d.Filter("myFriend.MyCol == 42");
~~~

Columns coming from the friend trees can be referred to by their full name, like in the example above,
or the friend tree name can be omitted in case the column name is not ambiguous (e.g. "MyCol" could be used instead of
      "myFriend.MyCol" in the example above).


\anchor other-file-formats
### Reading data formats other than ROOT trees
RDataFrame can be interfaced with RDataSources. The ROOT::RDF::RDataSource interface defines an API that RDataFrame can use to read arbitrary columnar data formats.

RDataFrame calls into concrete RDataSource implementations to retrieve information about the data, retrieve (thread-local) readers or "cursors" for selected columns
and to advance the readers to the desired data entry.
Some predefined RDataSources are natively provided by ROOT such as the ROOT::RDF::RCsvDS which allows to read comma separated files:
~~~{.cpp}
auto tdf = ROOT::RDF::MakeCsvDataFrame("MuRun2010B.csv");
auto filteredEvents =
   tdf.Filter("Q1 * Q2 == -1")
      .Define("m", "sqrt(pow(E1 + E2, 2) - (pow(px1 + px2, 2) + pow(py1 + py2, 2) + pow(pz1 + pz2, 2)))");
auto h = filteredEvents.Histo1D("m");
h->Draw();
~~~

See also MakeNumpyDataFrame (Python-only), MakeNTupleDataFrame(), MakeArrowDataFrame(), MakeSqliteDataFrame().

\anchor callgraphs
### Computation graphs (storing and reusing sets of transformations)

As we saw, transformed dataframes can be stored as variables and reused multiple times to create modified versions of the dataset. This implicitly defines a **computation graph** in which
several paths of filtering/creation of columns are executed simultaneously, and finally aggregated results are produced.

RDataFrame detects when several actions use the same filter or the same defined column, and **only evaluates each
filter or defined column once per event**, regardless of how many times that result is used down the computation graph.
Objects read from each column are **built once and never copied**, for maximum efficiency.
When "upstream" filters are not passed, subsequent filters, temporary column expressions and actions are not evaluated,
so it might be advisable to put the strictest filters first in the graph.

\anchor representgraph
### Visualizing the computation graph
It is possible to print the computation graph from any node to obtain a [DOT (graphviz)](https://en.wikipedia.org/wiki/DOT_(graph_description_language)) representation either on the standard output
or in a file.

Invoking the function ROOT::RDF::SaveGraph() on any node that is not the head node, the computation graph of the branch
the node belongs to is printed. By using the head node, the entire computation graph is printed.

Following there is an example of usage:
~~~{.cpp}
// First, a sample computational graph is built
ROOT::RDataFrame df("tree", "f.root");

auto df2 = df.Define("x", []() { return 1; })
             .Filter("col0 % 1 == col0")
             .Filter([](int b1) { return b1 <2; }, {"cut1"})
             .Define("y", []() { return 1; });

auto count =  df2.Count();

// Prints the graph to the rd1.dot file in the current directory
ROOT::RDF::SaveGraph(df, "./mydot.dot");
// Prints the graph to standard output
ROOT::RDF::SaveGraph(df);
~~~

The generated graph can be rendered using one of the graphviz filters, e.g. `dot`. For instance, the image below can be generated with the following command:
~~~{.sh}
$ dot -Tpng computation_graph.dot -ocomputation_graph.png
~~~

\image html RDF_Graph2.png

\anchor rdf-logging
### Activating RDataFrame execution logs

RDataFrame has experimental support for verbose logging of the event loop runtimes and other interesting related information. It is activated as follows:
~~~{.cpp}
#include <ROOT/RLogger.hxx>

// this increases RDF's verbosity level as long as the `verbosity` variable is in scope
auto verbosity = ROOT::Experimental::RLogScopedVerbosity(ROOT::Detail::RDF::RDFLogChannel(), ROOT::Experimental::ELogLevel::kInfo);
~~~

or in Python:
~~~{.python}
import ROOT

verbosity = ROOT.Experimental.RLogScopedVerbosity(ROOT.Detail.RDF.RDFLogChannel(), ROOT.Experimental.ELogLevel.kInfo)
~~~
*/
// clang-format on

namespace ROOT {

using ROOT::RDF::ColumnNames_t;
using ColumnNamesPtr_t = std::shared_ptr<const ColumnNames_t>;

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe.
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] dirPtr TDirectory where the tree is stored, e.g. a TFile.
/// \param[in] defaultColumns Collection of default columns.
///
/// The default columns are looked at in case no column is specified in the
/// booking of actions or transformations.
/// \see ROOT::RDF::RInterface for the documentation of the methods available.
RDataFrame::RDataFrame(std::string_view treeName, TDirectory *dirPtr, const ColumnNames_t &defaultColumns)
   : RInterface(std::make_shared<RDFDetail::RLoopManager>(nullptr, defaultColumns))
{
   if (!dirPtr) {
      auto msg = "Invalid TDirectory!";
      throw std::runtime_error(msg);
   }
   const std::string treeNameInt(treeName);
   auto tree = static_cast<TTree *>(dirPtr->Get(treeNameInt.c_str()));
   if (!tree) {
      auto msg = "Tree \"" + treeNameInt + "\" cannot be found!";
      throw std::runtime_error(msg);
   }
   GetProxiedPtr()->SetTree(std::shared_ptr<TTree>(tree, [](TTree *) {}));
}

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe.
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] filenameglob TDirectory where the tree is stored, e.g. a TFile.
/// \param[in] defaultColumns Collection of default columns.
///
/// The filename glob supports the same type of expressions as TChain::Add(), and it is passed as-is to TChain's
/// constructor.
///
/// The default columns are looked at in case no column is specified in the
/// booking of actions or transformations.
/// \see ROOT::RDF::RInterface for the documentation of the methods available.
RDataFrame::RDataFrame(std::string_view treeName, std::string_view filenameglob, const ColumnNames_t &defaultColumns)
   : RInterface(std::make_shared<RDFDetail::RLoopManager>(nullptr, defaultColumns))
{
   const std::string treeNameInt(treeName);
   const std::string filenameglobInt(filenameglob);
   auto chain = std::make_shared<TChain>(treeNameInt.c_str());
   chain->Add(filenameglobInt.c_str());
   GetProxiedPtr()->SetTree(chain);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe.
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] fileglobs Collection of file names of filename globs
/// \param[in] defaultColumns Collection of default columns.
///
/// The filename globs support the same type of expressions as TChain::Add(), and each glob is passed as-is
/// to TChain's constructor.
///
/// The default columns are looked at in case no column is specified in the booking of actions or transformations.
/// \see ROOT::RDF::RInterface for the documentation of the methods available.
RDataFrame::RDataFrame(std::string_view treeName, const std::vector<std::string> &fileglobs,
                       const ColumnNames_t &defaultColumns)
   : RInterface(std::make_shared<RDFDetail::RLoopManager>(nullptr, defaultColumns))
{
   std::string treeNameInt(treeName);
   auto chain = std::make_shared<TChain>(treeNameInt.c_str());
   for (auto &f : fileglobs)
      chain->Add(f.c_str());
   GetProxiedPtr()->SetTree(chain);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe.
/// \param[in] tree The tree or chain to be studied.
/// \param[in] defaultColumns Collection of default column names to fall back to when none is specified.
///
/// The default columns are looked at in case no column is specified in the
/// booking of actions or transformations.
/// \see ROOT::RDF::RInterface for the documentation of the methods available.
RDataFrame::RDataFrame(TTree &tree, const ColumnNames_t &defaultColumns)
   : RInterface(std::make_shared<RDFDetail::RLoopManager>(&tree, defaultColumns))
{
}

//////////////////////////////////////////////////////////////////////////
/// \brief Build a dataframe that generates numEntries entries.
/// \param[in] numEntries The number of entries to generate.
///
/// An empty-source dataframe constructed with a number of entries will
/// generate those entries on the fly when some action is triggered,
/// and it will do so for all the previously-defined columns.
/// \see ROOT::RDF::RInterface for the documentation of the methods available.
RDataFrame::RDataFrame(ULong64_t numEntries)
   : RInterface(std::make_shared<RDFDetail::RLoopManager>(numEntries))

{
}

//////////////////////////////////////////////////////////////////////////
/// \brief Build dataframe associated to data source.
/// \param[in] ds The data source object.
/// \param[in] defaultColumns Collection of default column names to fall back to when none is specified.
///
/// A dataframe associated to a data source will query it to access column values.
/// \see ROOT::RDF::RInterface for the documentation of the methods available.
RDataFrame::RDataFrame(std::unique_ptr<ROOT::RDF::RDataSource> ds, const ColumnNames_t &defaultColumns)
   : RInterface(std::make_shared<RDFDetail::RLoopManager>(std::move(ds), defaultColumns))
{
}

//////////////////////////////////////////////////////////////////////////
/// \brief Build dataframe from an RDatasetSpec object.
/// \param[in] spec The dataset specification object.
///
/// A dataset specification includes trees and file names,
/// as well as an optional friend list and/or entry range.
///
/// ### Example usage:
/// ~~~{.py}
/// spec = ROOT.RDF.Experimental.RDatasetSpec("tree", "file.root", (3, 5))
/// spec.AddFriend([("tree1", "a.root"), ("tree2", "b.root")], "alias")
/// df = ROOT.RDataFrame(spec)
/// ~~~
RDataFrame::RDataFrame(ROOT::RDF::Experimental::RDatasetSpec spec)
   : RInterface(std::make_shared<RDFDetail::RLoopManager>(std::move(spec)))
{
}

} // namespace ROOT

namespace cling {
//////////////////////////////////////////////////////////////////////////
/// Print an RDataFrame at the prompt
std::string printValue(ROOT::RDataFrame *tdf)
{
   auto &df = *tdf->GetLoopManager();
   auto *tree = df.GetTree();
   auto defCols = df.GetDefaultColumnNames();

   std::ostringstream ret;
   if (tree) {
      ret << "A data frame built on top of the " << tree->GetName() << " dataset.";
      if (!defCols.empty()) {
         if (defCols.size() == 1)
            ret << "\nDefault column: " << defCols[0];
         else {
            ret << "\nDefault columns:\n";
            for (auto &&col : defCols) {
               ret << " - " << col << "\n";
            }
         }
      }
   } else if (auto ds = tdf->fDataSource) {
      ret << "A data frame associated to the data source \"" << cling::printValue(ds) << "\"";
   } else {
      ret << "An empty data frame that will create " << df.GetNEmptyEntries() << " entries\n";
   }

   return ret.str();
}
} // namespace cling
