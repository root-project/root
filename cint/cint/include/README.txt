File: $CINTSYSDIR/include/README

About this file:

 	This file contains information about $CINTSYSDIR/include directory 
	and files under the directory.


$CINTSYSDIR environment variable:

	Environment variable CINTSYSDIR must point to the directory where 
	you have installed cint. For example, if you installed cint into
	/usr/local/cint, 

	    ksh,bsh
		$ CINTSYSDIR=/usr/local/cint
		$ export CINTSYSDIR
	    csh
		% setenv CINTSYSDIR /usr/local/cint

	If you don't do this, you can not utilize files and class libraries
	uner $CINTSYSDIR/include.

$CINTSYSDIR/include directory:

	This directory contains Cint's standard header files other than
	STL library. (STL header files are located under $CINTSYSDIR/stl 
        directory)  Files in this directory can be included or linked by
	'#include <xxx>' statement.


DLL files:
	Several DLL (shared library) files are created by Cint setup
	script. Those libraries can expand Cint capability beyond ANSI/ISO
	C/C++.

  include/stdfunc.dll  (created under lib/stdstrct)
	This provides C standard library function. Cint mostly works fine
	without this file. However, when it comes to a complicated function
	overloading resolution, this library helps.

  include/posix.dll    (created under lib/posix)
	This provides subset of POSIX system calls. Emulation library
	is provides under Windows.
	ROOT/CINT may not need this because ROOT has its own library. 

  include/win32api.dll (created under lib/win32api)
	This provides subset of Win32 API. Windows only.
	ROOT/CINT may not need this because ROOT has its own library. 

  include/cintsock.dll (created under lib/socket)
	This provides TCP/IP socket library.  
	ROOT/CINT may not need this because ROOT has its own library.

  include/ipc.dll      (created under lib/ipc)
	This provides shared memory, semaphore and inter process messaging.
	ROOT/CINT may not need this because ROOT has its own library. 
	
  include/pthread.dll  (created under lib/pthread)
	This provides pthread library. Please be careful when you use
	this library because Cint itself is not thread safe. Please examine
	example in demo/mthread.
	ROOT/CINT may not need this because ROOT has its own library. 


Auxiliary files:
	Several DLL (shared library) files are created by Cint setup
	script. Those libraries provides auxiliary capability. Those
	files are not essential to Cint.

  include/statistics.dll
	Calculate standard deviation.

  include/array.h/.c/.dll
  include/carray.h/.c/.dll
	Array and complex array class.

  include/fft.h/.c/.dll
	Fast Fourier Transform library.

  include/lsm.h/.c/.dll
	Least Square Method. Line regression.

  include/ReadF.h/.cxx/.dll
	'awk' like parsing class ReadFile.

  include/RegE.h/.cxx/.dll
	Regular expression class.


