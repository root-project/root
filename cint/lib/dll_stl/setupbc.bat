del ..\..\stl\string.dll
del ..\..\stl\vector.dll
del ..\..\stl\list.dll
del ..\..\stl\deque.dll
del ..\..\stl\map.dll
del ..\..\stl\set.dll
del ..\..\stl\multimap.dll
del ..\..\stl\multiset.dll
del ..\..\stl\valarray.dll
del ..\..\stl\stack.dll
del ..\..\stl\queue.dll
del ..\..\stl\exception.dll
del G__*
del *.dll

makecint -mk Makestr -dl string.dll -H str.h  -cint -M0x10 -Z0
make.exe -f Makestr 
move string.dll ..\..\stl\string.dll

makecint -mk Makevec -dl vector.dll -H vec.h  -cint -M0x10 -Z0
make.exe -f Makevec 
move vector.dll ..\..\stl\vector.dll

makecint -mk Makelist -dl list.dll -H lst.h  -cint -M0x10 -Z0
make.exe -f Makelist 
move list.dll ..\..\stl\list.dll

makecint -mk Makedeque -dl deque.dll -H dqu.h  -cint -M0x10 -Z0
make.exe -f Makedeque 
move deque.dll ..\..\stl\deque.dll

makecint -mk Makemap -dl map.dll -H mp.h  -cint -M0x10 -Z0
make.exe -f Makemap 
move map.dll ..\..\stl\map.dll

makecint -mk Makeset -dl set.dll -H st.h  -cint -M0x10 -Z0
make.exe -f Makeset 
move set.dll ..\..\stl\set.dll

del G__*
del *.obj
del *.lib
del *.tds

makecint -mk Makemmap -dl multimap.dll -H multmp.h  -cint -M0x10 -Z0
make.exe -f Makemmap 
move multimap.dll ..\..\stl\multimap.dll

makecint -mk Makemset -dl multiset.dll -H multst.h  -cint -M0x10 -Z0
make.exe -f Makemset 
move multiset.dll ..\..\stl\multiset.dll

makecint -mk Makestk -dl stack.dll -H stk.h  -cint -M0x10 -Z0
make.exe -f Makestk
move stack.dll ..\..\stl\stack.dll

makecint -mk Makeque -dl queue.dll -H que.h  -cint -M0x10 -Z0
make.exe -f Makeque
move queue.dll ..\..\stl\queue.dll

makecint -mk Makevary -dl valarray.dll -H vary.h  -cint -M0x10 -Z0
make.exe -f Makevary 
move valarray.dll ..\..\stl\valarray.dll

makecint -mk Makeeh -dl exception.dll -H eh.h -cint -M0x10 -Z0
make.exe -f Makeeh
move exception.dll ..\..\stl\exception.dll

del G__*
del Make*
del *.def
del *.obj
del *.lib
del *.tds
