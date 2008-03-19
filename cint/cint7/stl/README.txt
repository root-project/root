/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
stl/README.txt

 This directory contains modified STL HP reference implementation. 
 Cint is not quite ready for supporting STL.

DLL files:
 Several DLL (shared library) files are created by Cint setup
 script. Those libraries provides precompiled STL container for
 Cint interpreter. Those libraries are created under lib/dll_stl.

   stl/string.dll
   stl/vector.dll
   stl/list.dll
   stl/deque.dll
   stl/map.dll
   stl/map2.dll
   stl/multimap.dll
   stl/multimap2.dll
   stl/set.dll
   stl/multiset.dll
   stl/stack.dll
   stl/queue.dll
   stl/exception.dll
   stl/valarray.dll   (right now, only in Windows)
