#
# cint dynimic link standard library makefile
#
done: statistics.dl array.dl carray.dl fft.dl xgraph.dl lsm.dl ReadF.dl RegE.dl
	rm G__* *.o | echo 'G__* not exist'
	echo > done

statistics.dl: statistics.c ../cint
	makecint -mk Makeit -dl statistics.dl -c statistics.c -cint -Z0
	make -f Makeit

array.dl: array.c ../cint
	makecint -mk Makeit  -dl array.dl -c array.c -cint -Z0
	make -f Makeit

carray.dl: carray.c ../cint
	makecint -mk Makeit  -dl carray.dl -c carray.c -cint -Z0
	make -f Makeit

fft.dl: fft.c ../cint
	makecint -mk Makeit  -dl fft.dl -c fft.c -cint -Z0
	make -f Makeit

lsm.dl: lsm.c ../cint
	makecint -mk Makeit  -dl lsm.dl -c lsm.c -cint -Z0
	make -f Makeit

xgraph.dl: xgraph.c ../cint
	makecint -mk Makeit  -dl xgraph.dl -c xgraph.c -cint -Z0
	make -f Makeit

ReadF.dl: ReadF.h ReadF.C ../cint
	makecint -mk Makeit  -dl ReadF.dl -H ReadF.h -C++ ReadF.C -cint -Z0
	make -f Makeit

RegE.dl: RegE.h RegE.C ../cint
	makecint -mk Makeit  -dl RegE.dl -H RegE.h -C++ RegE.C -cint -Z0
	make -f Makeit

