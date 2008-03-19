lib/snstream/README

 lib/snstream directory exists for creating iostream library linkage file
src/sunstrm.cxx and src/sunstrm.h for Sun CC5 compiler. These files 
contain interface methods for iostream library. You can create those by 
doing 'make' under this directory. Usually nobody but only author should 
do this. User doesn't need to recognize this.

 cbstream.cpp is based on template based stream library.

 Creating src/sunstrm.cxx

 1) Edit src/newlink.c
  Start a text editor, look for '#define G__N_EXPLICITDESTRUCTOR'. Please
  comment this line out. 

 2) Create special 'cint'
  Go back to $CINTSYSDIR and do make to create special 'cint'. You may need
  to specify src/fakestrm.cxx in src/Makefile.

 3) Come to src/snstream directory and do make to create src/snstrm.cxx.

 4) Edit src/newlink.c
  Edit src/newlink.c and restore '#define G__N_EXPLICITDESTRUCTOR' macro.

 5) Go back to $CINTSYSDIR and do make to create updated cint.
