del ..\..\stl\string.dll
del ..\..\stl\vector.dll
del ..\..\stl\list.dll
del ..\..\stl\deque.dll
del ..\..\stl\map.dll
del ..\..\stl\map2.dll
del ..\..\stl\set.dll
del ..\..\stl\multimap.dll
del ..\..\stl\multimap2.dll
del ..\..\stl\multiset.dll
del ..\..\stl\valarray.dll
del ..\..\stl\stack.dll
del ..\..\stl\queue.dll
del ..\..\stl\exception.dll
del G__*
del *.dll

makecint -mk Makestr -dl string.dll -H str.h  -cint -Z0
nmake -f Makestr CFG="string - Win32 Release"
move Release\string.dll ..\..\stl\string.dll

makecint -mk Makevec -dl vector.dll -H vec.h  -cint -Z0
nmake -f Makevec CFG="vector - Win32 Release"
move Release\vector.dll ..\..\stl\vector.dll

makecint -mk Makelist -dl list.dll -H lst.h  -cint -Z0
nmake -f Makelist CFG="list - Win32 Release"
move Release\list.dll ..\..\stl\list.dll

makecint -mk Makedeque -dl deque.dll -H dqu.h  -cint -Z0
nmake -f Makedeque CFG="deque - Win32 Release"
move Release\deque.dll ..\..\stl\deque.dll

makecint -mk Makemap -dl map.dll -H mp.h  -cint -Z0
nmake -f Makemap CFG="map - Win32 Release"
move Release\map.dll ..\..\stl\map.dll

makecint -mk Makemap2 -dl map2.dll -DG__MAP2 -H mp.h  -cint -Z0
nmake -f Makemap2 CFG="map2 - Win32 Release"
move Release\map2.dll ..\..\stl\map2.dll

makecint -mk Makeset -dl set.dll -H st.h  -cint -Z0
nmake -f Makeset CFG="set - Win32 Release"
move Release\set.dll ..\..\stl\set.dll

del G__*
del Release\*.obj
del Release\*.lib
del Release\*.pch

makecint -mk Makemmap -dl multimap.dll -H multmp.h  -cint  -Z0
nmake -f Makemmap CFG="multimap - Win32 Release"
move Release\multimap.dll ..\..\stl\multimap.dll

makecint -mk Makemmap2 -dl multimap2.dll -DG__MAP2 -H multmp.h  -cint  -Z0
nmake -f Makemmap2 CFG="multimap2 - Win32 Release"
move Release\multimap2.dll ..\..\stl\multimap2.dll

makecint -mk Makemset -dl multiset.dll -H multst.h  -cint  -Z0
nmake -f Makemset CFG="multiset - Win32 Release"
move Release\multiset.dll ..\..\stl\multiset.dll

makecint -mk Makestk -dl stack.dll -H stk.h  -cint  -Z0
nmake -f Makestk CFG="stack - Win32 Release"
move Release\stack.dll ..\..\stl\stack.dll

makecint -mk Makeque -dl queue.dll -H que.h  -cint -Z0
nmake -f Makeque CFG="queue - Win32 Release"
move Release\queue.dll ..\..\stl\queue.dll

makecint -mk Makevary -dl valarray.dll -H vary.h  -cint -Z0
nmake -f Makevary CFG="valarray - Win32 Release"
move Release\valarray.dll ..\..\stl\valarray.dll

makecint -mk Makeeh -dl exception.dll -H eh.h -cint  -Z0
nmake -f Makeeh CFG="exception - Win32 Release"
move Release\exception.dll ..\..\stl\exception.dll

rem del Release\*
rem rmdir Release
del G__*
del Make*
del *.def
