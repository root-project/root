                README File for ROOT Binary Distribution
                ----------------------------------------

Contents
========

Directory root/ :

README      - directory containing important information
LICENSE     - usage terms and conditions
configure   - build configuration script
bin         - directory containing executables
include     - directory containing the ROOT header files
lib         - directory containing the ROOT libraries (in shared library format)
etc         - directory containing default resource and mime type files
etc/proof   - directory containing settings for the PROOF system
macros      - directory containing system macros
icons       - directory containing xpm icons
test        - some ROOT test programs
tutorials   - example macros that can be executed by the bin/root module


Environment variables
=====================

To set the needed environment variables, PATH and LD_LIBRARY_PATH, use
the following convenience script. For the sh shell family do:

   . <pathname>/root/bin/thisroot.sh

and for the csh shell family do:

   source <pathname>/root/bin/thisroot.csh

where <pathname> is the location where you unpacked the ROOT distribution.

Typically add these lines to your .profile or .login files.


Running interactive ROOT and the tutorial macros
================================================

To run the example macros, go to the root/tutorials directory and do, e.g.:

$ root
root [0] .x benchmarks.C
  -- this will run all tutorials and will benchmark your machine.
  -- see http://root.cern.ch/drupal/content/benchmarking for the
     normalization and comparisons.
root [1] .x demos.C
  -- Click on any button you like to run the corresponding tutorial.
  -- Move the objects on the canvas around using the mouse.
root [2] .q


Compiling and running ROOT test programs
========================================

To run some ROOT test programs, go to the root/test directory and do
(after having selected the machine dependent flags in the Makefile):

$ make
$ ./Event
$ ./hsimple
$ ./minexam
$ ./tcollex
$ ./tstring
etc...
