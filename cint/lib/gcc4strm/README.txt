lib/gcc4strm/README

 lib/gcc4strm directory exists for creating iostream library linkage file
src/gcc4strm.cxx and src/gcc4strm.h for g++ 3.00 compiler. These files 
contain interface methods for iostream library. You can create those by 
doing 'make' under this directory. Usually nobody but only author should 
do this. User doesn't need to recognize this.
 Files in this directory are originally copied from lib/snstream/* and 
modified for gcc4strm.

 cbstream.cpp is based on template based stream library.

 Creating src/gcc4strm.cxx

 1) Just do 'make' in this directory. 
