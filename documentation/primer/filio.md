# File I/O and Parallel Analysis #

## Storing ROOT Objects ##

ROOT offers the possibility to write instances of classes on
disk, into a *ROOT-file* (see the `TFile` class for more details).
One says that the object is made "persistent" by storing
it on disk. When reading the file back, the object is reconstructed
in memory. The requirement to be satisfied to perform I/O of instances
of a certain class is that the ROOT type system is aware of the layout
in memory of that class.
This topic is beyond the scope of this document: it is worth to mention
that I/O can be performed out of the box for the almost complete set
of ROOT classes.

We can explore this functionality with histograms and two simple macros.

``` {.cpp}
@ROOT_INCLUDE_FILE macros/write_to_file.C
```

Not bad, eh ? Especially for a language that does not foresees
persistency natively like C++. The *RECREATE* option forces ROOT to
create a new file even if a file with the same name exists on disk.

Now, you may use the Cling command line to access information in the file
and draw the previously written histogram:

``` {.cpp}
>  root my_rootfile.root
root [0]
Attaching file my_rootfile.root as _file0...
root [1] _file0->ls()
TFile**     my_rootfile.root
 TFile*     my_rootfile.root
  KEY: TH1F	my_histogram;1 My Title
root [2] my_histogram->Draw()
```
\newpage
Alternatively, you can use a simple macro to carry out the job:

``` {.cpp}
@ROOT_INCLUDE_FILE macros/read_from_file.C
```

## N-tuples in ROOT ##

### Storing simple N-tuples ###

Up to now we have seen how to manipulate input read from ASCII files.
ROOT offers the possibility to do much better than that, with its own
n-tuple classes. Among the many advantages provided by these classes one
could cite

-   Optimised disk I/O.

-   Possibility to store many n-tuple rows.

-   Write the n-tuples in ROOT files.

-   Interactive inspection with `TBrowser`.

-   Store not only numbers, but also *objects* in the columns.

In this section we will discuss briefly the `TNtuple` class, which is a
simplified version of the `TTree` class. A ROOT `TNtuple` object can
store rows of float entries. Let's tackle the problem according to the
usual strategy commenting a minimal example

``` {.cpp}
@ROOT_INCLUDE_FILE macros/write_ntuple_to_file.C
```

This data written to this example n-tuple represents, in the statistical
sense, three independent variables (Potential or Voltage, Pressure and
Temperature), and one variable (Current) which depends on the others
according to very simple laws, and an additional Gaussian smearing. This
set of variables mimics a measurement of an electrical resistance while
varying pressure and temperature.

Imagine your task now consists in finding the relations among the
variables -- of course without knowing the code used to generate them.
You will see that the possibilities of the `NTuple` class enable you to
perform this analysis task. Open the ROOT file (`cond_data.root`)
written by the macro above in an interactive session and use a
`TBrowser` to interactively inspect it:

``` {.cpp}
root[0] TBrowser b
```
You find the columns of your n-tuple written as *leafs*. Simply clicking
on them you can obtain histograms of the variables!

Next, try the following commands at the shell prompt and in the
interactive ROOT shell, respectively:

``` {.cpp}
> root conductivity_experiment.root
Attaching file conductivity_experiment.root as _file0...
root [0] cond_data->Draw("Current:Potential")
```

You just produced a correlation plot with one single line of code!

Try to extend the syntax typing for example

``` {.cpp}
root [1] cond_data->Draw("Current:Potential","Temperature<270")
```

What do you obtain ?

Now try

``` {.cpp}
root [2] cond_data->Draw("Current/Potential:Temperature")
```

It should have become clear from these examples how to navigate in such
a multi-dimensional space of variables and unveil relations between
variables using n-tuples.

### Reading N-tuples

For completeness, you find here a small macro to read the data back from
a ROOT n-tuple

``` {.cpp}
@ROOT_INCLUDE_FILE macros/read_ntuple_from_file.C
```

The macro shows the easiest way of accessing the content of a n-tuple:
after loading the n-tuple, its branches are assigned to variables and
`GetEntry(long)` automatically fills them with the content for a
specific row. By doing so, the logic for reading the n-tuple and the
code to process it can be split and the source code remains clear.

### Storing Arbitrary N-tuples ###

It is also possible to write n-tuples of arbitrary type by using ROOT's
`TBranch` class. This is especially important as `TNtuple::Fill()`
accepts only floats. The following macro creates the same n-tuple as
before but the branches are booked directly. The `Fill()` function then
fills the current values of the connected variables to the tree.

``` {.cpp}
@ROOT_INCLUDE_FILE macros/write_ntuple_to_file_advanced.C
```

The `Branch()` function requires a pointer to a variable and a
definition of the variable type. The following table lists some of the possible
values.
Please note that ROOT is not checking the input and mistakes are likely
to result in serious problems. This holds especially if values are read
as another type than they have been written, e.g. when storing a
variable as float and reading it as double.

List of variable types that can be used to define the type of a branch in ROOT:

  type               size     C++             identifier
  ------------------ -------- --------------- ------------
  signed integer     32 bit   int             I
                     64 bit   long            L
  unsigned integer   32 bit   unsigned int    i
                     64 bit   unsigned long   l
  floating point     32 bit   float           F
                     64 bit   double          D
  boolean            -        bool            O


### Processing N-tuples Spanning over Several Files ###

Usually n-tuples or trees span over many files and it would be difficult
to add them manually. ROOT thus kindly provides a helper class in the
form of `TChain`. Its usage is shown in the following macro which is
very similar to the previous example. The constructor of a `TChain`
takes the name of the `TTree` (or `TNuple`) as an argument. The files
are added with the function `Add(fileName)`, where one can also use
wild-cards as shown in the example.

``` {.cpp}
@ROOT_INCLUDE_FILE macros/read_ntuple_with_chain.C
```

### *For the advanced user:* Processing trees with a selector script ###


Another very general and powerful way of processing a `TChain` is
provided via the method `TChain::Process()`. This method takes as
arguments an instance of a -- user-implemented-- class of type
`TSelector`, and -- optionally -- the number of entries and the first
entry to be processed. A template for the class `TSelector` is provided
by the method `TTree::MakeSelector`, as is shown in the little macro
`makeSelector.C` below.

It opens the n-tuple `conductivity_experiment.root` from the example
above and creates from it the header file `MySelector.h` and a template
to insert your own analysis code, `MySelector.C`.
\newpage

``` {.cpp}
@ROOT_INCLUDE_FILE macros/makeMySelector.C
```

The template contains the entry points `Begin()` before processing 
of the `TChain` starts, `Process()` called for
every entry of the chain, and `Terminate()`
called after the last entry has been processed. Typically,
initialization like booking of histograms is performed in
`Begin()`, the analysis, i.e. the selection of entries,
calculations and filling of histograms, is done in `Process()`, and
final operations like plotting and storing of results happen in
`Terminate()`.

A simple example of a selector class is shown in the macro
`MySelector.C`. The example is executed with the following sequence of
commands:

``` {.cpp}
> TChain *ch=new TChain("cond_data", "Chain for Example N-Tuple");
> ch->Add("conductivity_experiment*.root");
> ch->Process("MySelector.C+");
```

As usual, the "`+`" appended to the name of the macro to be executed
initiates the compilation of the `MySelector.C` with the system compiler
in order to improve performance.

The code in `MySelector.C`, shown in the listing below, books some
histograms in `Begin()` and adds them to the instance `fOutput`,
which is of the class `TList`. The final processing in
`Terminate()` allows to access histograms and store, display or save
them as pictures. This is shown in the example via the `TList`
`fOutput`. See the commented listing below for more details; most of the
text is actually comments generated automatically by
`TTree::MakeSelector`.

``` {.cpp}
@ROOT_INCLUDE_FILE macros/MySelector.C
```

### Optimisation Regarding N-tuples ###

ROOT automatically applies compression algorithms on n-tuples to reduce
the memory consumption. A value that is in most cases the same will
consume only small space on your disk (but it has to be decompressed on
reading). Nevertheless, you should think about the design of your
n-tuples and your analyses as soon as the processing time exceeds some
minutes.

-   Try to keep your n-tuples simple and use appropriate variable types.
    If your measurement has only a limited precision, it is needless to
    store it with double precision.

-   Experimental conditions that do not change with every single
    measurement should be stored in a separate tree. Although the
    compression can handle redundant values, the processing time
    increase with every variable that has to be filled.

-   The function `SetCacheSize(long)` specifies the size of the cache
    for reading a `TTree` object from a file. The default value is 30MB.
    A manual increase may help in certain situations. Please note that
    the caching mechanism can cover only one `TTree` object per `TFile`
    object.

-   You can select the branches to be covered by the caching algorithm
    with `AddBranchToCache` and deactivate unneeded branches with
    `SetBranchStatus`. This mechanism can result in a significant
    speed-up for simple operations on trees with many branches.

-   You can measure the performance easily with `TTreePerfStats`. The
    ROOT documentation on this class also includes an introductory
    example. For example, `TTreePerfStats` can show you that it is
    beneficial to store meta data and payload data separately, i.e.
    write the meta data tree in a bulk to a file at the end of your job
    instead of writing both trees interleaved.
