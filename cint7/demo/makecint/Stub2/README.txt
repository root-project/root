demo/makecint/Stub2/README

 This is the simplest example of how you can derive compiled class from
interpreted class. 2 techniques are demonstrated in this example.

  - Accessing compiled protected member from interpreted derived class
     This feature is enabled by following pragma statement.
        #pragma link C++ class+protected [classname];
  - Virtual function resolution in mixed interpreted/compiled class environment
     This feature is enabled by -i++ option of makecint. 

OUTLINE =====================================================================

1) Files
 This directory contains

	README       : This file
	compiled.h   : Compiled class/function header
	compiled.cxx : Compiled function definition
	stub.h       : Stub class/function header
	main.cxx     : Interpreted function
	setup	     : Shell script to setup and test 
	setupcygwin  : Shell script to setup and test for Cygwin
	#setup.bat    : WinNT batch file to setup and test , VC++
	#setupbc.bat  : WinNT batch file to setup and test , C++Builder

2) Simple try

 You can interpret all of those files by cint as follows, by the way, and
you will get the same result with the precompiled version.
	
    $ cint compiled.cxx main.cxx

Or you could try compiling everything as follows.

    $ g++ compiled.cxx main.cxx
    $ a.out

Those results do not match exactly. Special Cint feature is used in main.cxx
and it is disabled in g++ compilation.


3) Precompile compiled.h/cxx
 You can compile compiled.h/cxx by following commands.

    $ makecint -mk Make1 -dl compiled.dll -H compiled.h -C++ compiled.cxx
    $ make -f Make1 

 In case of Windows (including Cygwin), you need to compile compiled.dll
 and stub.dll as one DLL. In this case, 4) is not necessary.

    $ makecint -mk Make1 -dl compiled.dll -H compiled.h -C++ compiled.cxx -i++ stub.h
    $ make -f Make1 


4) Create stub library
 In order to derive compiled class from an interpreted class you need to
create a stub library as follows.

    $ makecint -mk Make2 -dl stub.dll -H dmy.h -i++ stub.h 
    $ make -f Make2


5) Run it in mixed interpreted/compiled mode
 Now, you can run the program. In this case, compiled.h/cxx and interface
of stub.h are precompiled. Body of stub.h is interpreted in main.cxx.

    $ cint main.cxx


# You can try all of above by running 'setup' script.
