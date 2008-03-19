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

makecint -mk Makestr -dl string.dll -H str.h  -cint  -Z0
make.exe -f Makestr 
move string.dll ..\..\stl\string.dll

makecint -mk Makevec -dl vector.dll -H vec.h  -cint  -Z0
make.exe -f Makevec 
move vector.dll ..\..\stl\vector.dll

makecint -mk Makevecbool -dl vectorbool.dll -H vecbool.h  -cint  -Z0
make.exe -f Makevecbool 
move vectorbool.dll ..\..\stl\vectorbool.dll

makecint -mk Makelist -dl list.dll -H lst.h  -cint  -Z0
make.exe -f Makelist 
move list.dll ..\..\stl\list.dll

makecint -mk Makedeque -dl deque.dll -H dqu.h  -cint  -Z0
make.exe -f Makedeque 
move deque.dll ..\..\stl\deque.dll

rem makecint -mk Makepair -dl pair.dll -H pr.h  -cint  -Z0
rem make.exe -f Makepair
rem move pair.dll ..\..\stl\pair.dll

makecint -mk Makemap -dl map.dll -H mp.h  -cint  -Z0
make.exe -f Makemap 
move map.dll ..\..\stl\map.dll

makecint -mk Makemap2 -dl map2.dll -DG__MAP2 -H mp.h  -cint  -Z0
make.exe -f Makemap2 
move map2.dll ..\..\stl\map2.dll

makecint -mk Makeset -dl set.dll -H st.h  -cint  -Z0
make.exe -f Makeset 
move set.dll ..\..\stl\set.dll

del G__*
del *.obj
del *.lib
del *.tds

makecint -mk Makemmap -dl multimap.dll -H multmp.h  -cint  -Z0
make.exe -f Makemmap 
move multimap.dll ..\..\stl\multimap.dll

makecint -mk Makemmap2 -dl multimap2.dll -DG__MAP2 -H multmp.h  -cint  -Z0
make.exe -f Makemmap2 
move multimap2.dll ..\..\stl\multimap2.dll

makecint -mk Makemset -dl multiset.dll -H multst.h  -cint  -Z0
make.exe -f Makemset 
move multiset.dll ..\..\stl\multiset.dll

makecint -mk Makestk -dl stack.dll -H stk.h  -cint  -Z0
make.exe -f Makestk
move stack.dll ..\..\stl\stack.dll

makecint -mk Makeque -dl queue.dll -H que.h  -cint  -Z0
make.exe -f Makeque
move queue.dll ..\..\stl\queue.dll

makecint -mk Makevary -dl valarray.dll -H vary.h  -cint  -Z0
make.exe -f Makevary 
move valarray.dll ..\..\stl\valarray.dll

makecint -mk Makeeh -dl exception.dll -H cinteh.h -cint  -Z0
make.exe -f Makeeh
move exception.dll ..\..\stl\exception.dll

makecint -mk Makese -dl stdexcept.dll -H se.h -cint  -Z0
make.exe -f Makese
move stdexcept.dll ..\..\stl\stdexcept.dll

makecint -mk Makeclimits -dl climits.dll -H clim.h -cint -Z1
make.exe -f Makeclimits
move climits.dll ..\..\stl\climits.dll

del G__*
del Make*
del *.def
del *.obj
del *.lib
del *.tds
