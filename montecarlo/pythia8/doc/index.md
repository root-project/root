\defgroup pythia8 Pythia8
\ingroup montecarlo
\brief The Pythia8 interface

The pythia8 directory is an interface to the C++ version of Pythia 8.1 event generators, 
written by T.Sjostrand.
The user is assumed to be familiar with the Pythia8 package.
Only a basic interface to Pythia8 is provided. Because Pythia8 is
also written in C++, its functions/classes can be called directly from a
compiled C++ script.

To call Pythia functions not available in this interface a dictionary must
be generated.

See pythia8.C for an example of use from ROOT interpreter.

See also the [complete Pythia8 documentation](http://home.thep.lu.se/~torbjorn/pythiaaux/recent.html)
