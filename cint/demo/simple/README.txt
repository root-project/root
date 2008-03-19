File: $CINTSYSDIR/demo/simple/README

 Simple example programs to be interpreted on cint. To try them, simply give
the file name to the cint.

	$ cint virtualfunc1.c
	$ cint template2.c
	      .
	      .

 It is also nice to try using debugger features. You can start cint by -S
option as follows. Cint stops at the first statement in main() function,
then you can use debugger commands. For debugger tutorial, try 
demo/debug/debug.c first.

	$ cint ../debug/debug.c

	$ cint -S virtualfunc1.c
	$ cint -S template2.c
	      .
	      .
