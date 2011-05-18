del ..\..\stl\string.dll
del ..\..\stl\vector.dll
del ..\..\stl\vectorbool.dll
del ..\..\stl\list.dll
del ..\..\stl\deque.dll
del ..\..\stl\pair.dll
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
del ..\..\stl\stdexcept.dll
del ..\..\stl\climits.dll
del G__*
del *.dll

makecint -mk Makestr -dl string.dll -H str.h  -cint -Z0
nmake /nologo -f Makestr CFG="string - Win32 Release"
move Release\string.dll ..\..\stl\string.dll

makecint -mk Makevec -dl vector.dll -H vec.h  -cint -Z0
nmake /nologo -f Makevec CFG="vector - Win32 Release" MACRO=/wd4181
move Release\vector.dll ..\..\stl\vector.dll

makecint -mk Makevecbool -dl vectorbool.dll -H vecbool.h  -cint -Z0
nmake /nologo -f Makevecbool CFG="vectorbool - Win32 Release"
move Release\vectorbool.dll ..\..\stl\vectorbool.dll

makecint -mk Makelist -dl list.dll -H lst.h  -cint -Z0
nmake /nologo -f Makelist CFG="list - Win32 Release"
move Release\list.dll ..\..\stl\list.dll

makecint -mk Makedeque -dl deque.dll -H dqu.h  -cint -Z0
nmake /nologo -f Makedeque CFG="deque - Win32 Release"
move Release\deque.dll ..\..\stl\deque.dll

rem makecint -mk Makepair -dl pair.dll -H pr.h  -cint -Z0
rem nmake /nologo -f Makepair CFG="pair - Win32 Release"
rem move Release\pair.dll ..\..\stl\pair.dll

makecint -mk Makemap -dl map.dll -H mp.h  -cint -Z0
nmake /nologo -f Makemap CFG="map - Win32 Release"
move Release\map.dll ..\..\stl\map.dll

makecint -mk Makemap2 -dl map2.dll -DG__MAP2 -H mp.h  -cint -Z0
nmake /nologo -f Makemap2 CFG="map2 - Win32 Release"
move Release\map2.dll ..\..\stl\map2.dll

makecint -mk Makeset -dl set.dll -H st.h  -cint -Z0
nmake /nologo -f Makeset CFG="set - Win32 Release"
move Release\set.dll ..\..\stl\set.dll

del G__*
del Release\*.obj
del Release\*.lib
del Release\*.pch

makecint -mk Makemmap -dl multimap.dll -H multmp.h  -cint  -Z0
nmake /nologo -f Makemmap CFG="multimap - Win32 Release"
move Release\multimap.dll ..\..\stl\multimap.dll

makecint -mk Makemmap2 -dl multimap2.dll -DG__MAP2 -H multmp.h  -cint  -Z0
nmake /nologo -f Makemmap2 CFG="multimap2 - Win32 Release"
move Release\multimap2.dll ..\..\stl\multimap2.dll

makecint -mk Makemset -dl multiset.dll -H multst.h  -cint  -Z0
nmake /nologo -f Makemset CFG="multiset - Win32 Release"
move Release\multiset.dll ..\..\stl\multiset.dll

makecint -mk Makestk -dl stack.dll -H stk.h  -cint  -Z0
nmake /nologo -f Makestk CFG="stack - Win32 Release"
move Release\stack.dll ..\..\stl\stack.dll

makecint -mk Makeque -dl queue.dll -H que.h  -cint -Z0
nmake /nologo -f Makeque CFG="queue - Win32 Release"
move Release\queue.dll ..\..\stl\queue.dll

makecint -mk Makevary -dl valarray.dll -H vary.h  -cint -Z0
nmake /nologo -f Makevary CFG="valarray - Win32 Release" MACRO="/wd4800 /wd4804"
move Release\valarray.dll ..\..\stl\valarray.dll

makecint -mk Makeeh -dl exception.dll -H cinteh.h -cint  -Z0
nmake /nologo -f Makeeh CFG="exception - Win32 Release"
move Release\exception.dll ..\..\stl\exception.dll

makecint -mk Makese -dl stdexcept.dll -H se.h -cint  -Z0
nmake /nologo -f Makese CFG="stdexcept - Win32 Release"
move Release\stdexcept.dll ..\..\stl\stdexcept.dll

makecint -mk Makeclimits -dl climits.dll -H clim.h -cint -Z1
nmake /nologo -f Makeclimits CFG="climits - Win32 Release"
move Release\climits.dll ..\..\stl\climits.dll

rem del Release\*
rem rmdir Release
del G__*
del Make*
del *.def
