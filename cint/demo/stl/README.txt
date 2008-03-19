demo/stl/README.txt

 This directory contains simple STL programming examples using CINT.
STL processing is extremely challanging for C++ language processor
maker. Please understand some examples included in this directory do
not run correctly on CINT yet.

 I believe that concept of "Generic Programming" introduced with
STL(Standard Template Library) is a real breakthrough. Software 
development methodology should go toward this direction. 

 However, running STL on an interpreter does not seems to be a good
idea. STL focuses on generarity and run-time efficiency. Compile-time
efficiency is largely sacrificed. Cint suffers from the compile-time 
inefficiency as well. Cint instantiates templates on-the-fly when it
runs STL based program. Significant run-time overhead is added.
For this reason, running STL based program on Cint is only for 
experimental purpose.

 I'm still unsure if I should and I could invent a technology to make
STL suitable on the interpreter. One idea is to embed an universally
polymorphic object which is instantiated with all of the STL containers
and algorithms.  Cint can make use of compiled version of STL algorithms,
data structures and iterators. Only user object is interpreted.

 Yet, another question comes arise. C++ is the best language to realize
Generic Programming at this moment. But how about in the future? A better
language may come out. Shall I invent a new technology on C++ based STL 
or will it end-up nonsense. 

1998 Jan 10
 Masharu Goto   cint@pcroot.cern.ch , gotom@jpn.hp.com

2001/Sep/29
 gcc-3.00 is supported from cint5.15.14. However, there are many problems
using this compiler with Cint. You will see problems running demo programs
in this directory. The author is working to solve those problems.



