\defgroup Tutorials Tutorials
\brief A collection of C++ macros, Python scripts and notebooks helping to learn ROOT by example.

You can execute the scripts in `$ROOTSYS/tutorials` (or sub-directories)
by setting your current directory in the script directory or from any
user directory with write access.

Several tutorials create new files. If you have write access to
the tutorials directory, the new files will be created in the tutorials
directory, otherwise they will be created in the user directory.

You can start by executing the standard ROOT demos with a session like:

```
  root > .x demos.C
```
or

```
  root > .x $ROOTSYS/tutorials/demos.C
```

You can execute the standard ROOT graphics benchmark with

```
  root > .x benchmarks.C
```

or

```
  root > .x $ROOTSYS/tutorials/benchmarks.C
```

## Get started

If you have never used ROOT before and don’t know where to start, we recommend that you first explore the [ROOT introductory course](https://github.com/root-project/student-course). You can also watch the recording of the course, but you should follow the material along on your PC. You also may want to have a look at the documentation of these modules:

- @ref tutorial_hist
- @ref tutorial_graphs
- @ref tutorial_fit
- @ref tutorial_tree
- @ref tutorial_ntuple
- @ref tutorial_dataframe
- @ref tutorial_roofit

The `$ROOTSYS/tutorials` directory includes several sub-directories:

\defgroup tutorial_hist Histograms tutorials
\ingroup Tutorials
\brief Examples showing the "histograms' classes" usage.

\defgroup tutorial_analysis Data analysis tutorials
\ingroup Tutorials
\brief Various examples of data analysis workflows.

\defgroup tutorial_visualisation Visualisation tutorials
\ingroup Tutorials
\brief All examples related to visually representing the data, including graphs, event displays, web graphics, GUIs and more. 

\defgroup tutorial_exp Experimental API tutorials
\ingroup Tutorials
\brief Various examples showing the experimental API.

\defgroup tutorial_evegen Event generation tutorials
\ingroup Tutorials
\brief Examples showing event generation with pythia and Monte Carlo.

\defgroup tutorial_roofit RooFit Tutorials
\ingroup Tutorials
\brief These tutorials illustrate the main features of [RooFit](group__Roofitmain.html): the name of the examples and their short description help in figuring out their objective.

\defgroup tutorial_histfactory HistFactory Tutorials
\ingroup Tutorials
\brief These tutorials illustrate the usage of the histfactory.

\defgroup tutorial_http HTTP tutorials
\ingroup Tutorials
\brief Examples showing the HTTP interface.

\defgroup tutorial_io IO tutorials
\ingroup Tutorials
\brief These tutorials illustrate some of the capabilities of the ROOT IO subsystem, including TTree, RNTuple, SQL and XML.

\defgroup tutorial_math Math tutorials
\ingroup Tutorials
\brief Examples showing the Math classes.

\defgroup tutorial_multicore Multicore tutorials
\ingroup Tutorials
\brief These examples aim to illustrate the multicore features of ROOT, such as thread awareness and safety, multithreading and multiprocessing.

\defgroup tutorial_roostats RooStats Tutorials
\ingroup Tutorials
\brief These tutorials illustrate the main features of RooStats.

\defgroup tutorial_ml Machine Learning tutorials
\ingroup Tutorials
\brief Examples showing how to use Machine Learning from ROOT. 

\defgroup tutorial_geometry Geometry tutorials
\ingroup Tutorials
\brief Examples showing how to make use of geometry classes in ROOT.

\defgroup tutorial_legacy Legacy tutorials
\ingroup Tutorials
\brief Legacy Tutorials
