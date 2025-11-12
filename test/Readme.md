# ROOT testing

This directory contains some ROOT test programs.

All executables for the tests produced when ROOT compiled with `-Dtesting=ON` flag


hsimple.cxx        - Simple test program that creates and saves some histograms.

MainEvent.cxx      - Simple test program that creates a ROOT Tree object and
                     fills it with some simple structures but also with complete
                     histograms. This program uses the files Event.cxx,
                     EventCint.cxx and Event.h. An example of a procedure to
                     link this program is in bind_Event. Note that the Makefile
                     invokes the rootcint utility to generate the CINT interface
                     EventCint.cxx.

Event.cxx          - Implementation for classes Event and Track.

minexam.cxx        - Simple test program to test data fitting.

ctorture.cxx       - Test program for the class TComplex.

tcollex.cxx        - Example usage of the ROOT collection classes.

tcollbm.cxx        - Benchmarks of ROOT collection classes.

tstring.cxx        - Example usage of the ROOT string class.

vmatrix.cxx        - Verification program for the TMatrix class.

vvector.cxx        - Verification program for the TVector class.

stressLinear.cxx   - Stress testing of the matrix/vector and linear algebra classes.

stressGraphics.cxx - Stress graphics tests for image production.

QpRandomDriver.cxx - Verfication program for Quadratic programming classes in Quadp library.

vlazy.cxx          - Verification program for lazy matrices.

hsimple.cxx        - Small program showing batch usage of histograms.

hworld.cxx         - Small program showing basic graphics.

guitest.cxx        - Example usage of the ROOT GUI classes.

guiviewer.cxx      - Another ROOT GUI example program.

Hello.cxx          - Dancing text example.

Aclock.cxx         - Analog clock (a la X11 xclock)

Tetris.cxx         - The famous tetris game (using ROOT basic graphics).

stress.cxx         - Important ROOT stress testing program.

bench.cxx          - STL and ROOT container test and benchmarking program.

DrawTest.sh        - Entry script to extensive TTree query test suite.

dt_*               - Scripts used by DrawTest.sh.
