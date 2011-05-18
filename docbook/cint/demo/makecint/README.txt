demo/makecint directory

# MAKECINT DEMONSTRATION
 $CINTSYSDIR/demo/makecint contains example application which can be embedded
into cint. You need C/C++ compiler to try these.
 In each sub-directory, there are demonstration script named as follows.

    setup         : Embed C/C++ program as archived library in UNIX
    setupdll      : Embed C/C++ program as DLL in UNIX

    setup.bat     : Embed C/C++ program as archived library in WinNT VC++
    setupdll.bat  : Embed C/C++ program as DLL in WinNT VC++

    scsetup.bat   : Embed C/C++ program as archived library in WinNT SC++
    scdll.bat     : Embed C/C++ program as DLL in WinNT SC++

    bcdll.bat     : Embed C/C++ program as DLL in WinNT C++Builder


# Embedding C++ program
 It is strongly recommended to start from 'Complex' which is the simplest.
Then proceed to 'ReadFile' and 'Array'. Some of these may have problems 
with your environment or C++ compiler. In that case , you need to modify 
Makefile or other files by yourself.

	1 Complex
	2 ReadFile
	3 DArray
	4 Array  << This example was made for C++3.0 and may not work any more
	5 Stub
	6 Stub2
	7 exception
        8 p2f
 
 Demo programs are only tested on HP-UX 9.05, Linux 1.1.0, Linux1.3, Linux2.0,
AIX3.2, AIX 4.1 WinNT-VC++/SC++. Operation in other environment has not been 
tested.  Linux2.0 is the most stable environment at this moment.


# Embedding C program
 If you only have old K&R style C compiler, like SunOS4.x without any 
commercial compilers , try example in KRcc.

	1 KRcc


# Embedding Cint into user's main program.
 If you want to embed Cint into your main program, refer to "UserMain" 
example.

        1 UserMain
