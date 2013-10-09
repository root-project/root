lib/libcxxstrm/README

 lib/libcxxstrm directory exists for creating iostream library linkage file
src/libcxxstrm.cxx and src/libcxxstrm.h for clang++ compiler with libc++.
These files contain interface methods for iostream library. You can create
those by doing 'make' under this directory. Usually nobody but only author
should do this. User doesn't need to recognize this.
 Files in this directory are originally copied from lib/gcc4strm/* and
modified for libcxxstrm.

 cbstream.cpp is based on template based stream library.

 Creating src/libcxxstrm.cxx

 1) Just do 'make' in this directory.
