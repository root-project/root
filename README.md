The ROOT System
===============

The ROOT system provides a set of OO frameworks with all the functionality
needed to handle and analyze large amounts of data in a very efficient way.
Having the data defined as a set of objects, specialized storage methods are
used to get direct access to the separate attributes of the selected objects,
without having to touch the bulk of the data. Included are histograming
methods in an arbitrary number of dimensions, curve fitting, function
evaluation, minimization, graphics and visualization classes to allow
the easy setup of an analysis system that can query and process the data
interactively or in batch mode, as well as a general parallel processing
framework, PROOF, that can considerably speed up an analysis.

Thanks to the built-in C++ interpreter cling, the command, the
scripting and the programming language are all C++. The interpreter
allows for fast prototyping of the macros since it removes the, time
consuming, compile/link cycle. It also provides a good environment to
learn C++. If more performance is needed the interactively developed
macros can be compiled using a C++ compiler via a machine independent
transparent compiler interface called ACliC.


The system has been designed in such a way that it can query its databases
in parallel on clusters of workstations or many-core machines. ROOT is
an open system that can be dynamically extended by linking external
libraries. This makes ROOT a premier platform on which to build data
acquisition, simulation and data analysis systems.

ROOT is available under the LGPL license.
