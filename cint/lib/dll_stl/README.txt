lib/dll_stl/README.txt

ABSTRACT:

 This directory contains build environment for precompiled STL container.
 STL precompiled containers are premature and only experimental. You need
 files in $CINTSYSDIR/stl and $CINTSYSDIR/lib/prec_stl directories to do
 this. Run setup script as follows, depende on your computer platform.

   Linux2.0 - egcs : SGI STL implementation 1997
     $ sh setup 

   HP-UX10.2/11.0 - aCC : RogueWave STL implementation 1997
     $ sh setup 

   SGI IRIX6 - KCC : RogueWave STL implementation
     $ sh setup

   Windows-NT/9x - Visual C++ 5.0 : P.J. Plauger STL implementation
     c:\cint\lib\dll_stl> setup.bat

   Windows-NT/9x - Borland C++ Builder 3.0 : RogueWave STL implementation
     c:\cint\lib\dll_stl> setupbc.bat

 Then, following files will be generated. These DLLs include several 
 instantiated STL template classes.

    stl/string.dll
    stl/vector.dll
    stl/list.dll
    stl/deque.dll
    stl/map.dll
    stl/set.dll
    stl/multimap.dll
    stl/multiset.dll
    stl/stack.dll
    stl/queue.dll
    stl/valarray.dll  (VC++5.0/6.0, BC++5.3, ARM/Linux only)
    stl/exception.dll
    stl/stdexcept.dll


FILES:

 This directory orignially contains following files. Other files will be
 created after running setup.bat script.

  README.txt  : this file
  setup       : setup shell script for Linux egcs
  setup.bat   : setup batch script for VC++5.0
  setupbc.bat : setup batch script for BC++5.3
  str.h       : string precompiled library
  vec.h       : vector precompiled library
  lst.h       : list precompiled library
  dqu.h       : deque precompiled library
  mp.h        : map precompiled library
  st.h        : set precompiled library
  multmp.h    : multimap precompiled library
  multst.h    : multiset precompiled library
  stk.h       : stack precompiled library
  que.h       : queue precompiled library
  vary.h      : valarray precompiled library (VC++5.0 and BC++5.3 only)
  eh.h        : exception precompiled library
  se.h        : stdexcept precompiled library

