
all:	Foo.so  reader writer 

Foo.so:Foo.h Foo.cxx
	rootcint -f FooDict.cxx -c Foo.h+
	g++ -Wall -c -g `root-config --cflags` Foo.cxx 
	g++ -Wall -c -g `root-config --cflags` FooDict.cxx 
	g++ -shared -Wl,-soname,Foo.so -o Foo.so Foo.o FooDict.o

reader:reader.C
	g++ -Wall -c -g `root-config --cflags` $<  
	g++ -o $@ reader.o `root-config --libs` \
		-Wl,-rpath,`root-config --libdir`

writer:writer.C Foo.so 
	g++ -Wall -c -g `root-config --cflags` $< 
	g++ -o $@ writer.o Foo.so `root-config --libs` \
		-Wl,-rpath,`root-config --libdir` -Wl,-rpath,.

dist:
	mkdir -p treeproblem 
	cp Foo.{cxx,h} tree.{C,h} reader.C writer.C Makefile README \
		treeproblem
	tar -czvf treeproblem.tar.gz treeproblem 
	rm -rf treeproblem 

check:Foo.so reader writer 
	root -b -l -q writer.C 
	root -b -l -q reader.C

clean:
	rm -f *.root *~ *.o *Dict.cxx *Dict.h *.so reader writer
