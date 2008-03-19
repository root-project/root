lib/wintcldl83/README

 This directory includes WildCard interpreter, a marriage of C/C++ and Tcl/Tk
interpreters.  Now it works on HP-UX, Linux and Windows-NT/9x. UNIX version
is in lib/WildCard directory. This directory is dedicated to Windows-NT/9x
environment using Visual C++ 6.0. 

What you need:=============================================================

 What you need to run WildCard are

    + Windows-NT or Windows-9x operating systems
    + Tcl/Tk8.3 or later
    + cint revision 5.15.05 or later
    + Visual C++ 6.0

Compile: ==================================================================

 1) Install CINT
   Please refer to ../../README and ../../platform/README to install CINT.

 2) Install Tcl/Tk8.3.
   You need to get Tcl/Tk8.3 or later. 
  Please install it to C:\Program Files\TCL directory (this is the default).
  Make sure you install Tcl/Tk8.3 with library and headers.

 3) Run C:\CINT\LIB\WINTCLDL83\setup.bat
   Run SETUP.BAT from command prompt.

 4) Adding path
   Edit C:\AUTOEXEC.BAT using notepad or other text editor. Please add 
  following lines at the end.

   SET PATH=%PATH%;C:\CINT\LIB\WINTCLDL83;C:\Program Files\TCL\BIN

 5) Re-boot or re-login. 


What WildCard can do ======================================================
 WildCard has full functionality of CINT and Tcl/Tk interpreters. You can
enjoy 100% compatibility from cint and wish in one environment. Take a look
at wildc.wc and calc.wc examples. To start wildc interpreter, use following
command.

  $ wildc wildc.wc

To monitor what is going on

   $ wildc -T wildc.wc

Be careful that wildc is a batch command.


WildCard Commands ========================================================

How to start WildCard:
	WildCard interpreter can be started by 'wildc' command. You can give
	mixed C/C++ and Tcl/Tk script. WildCard interpreter usually accepts
        source code with .wc extension.

    Example:
	$ wildc grcalc.wc            # starts C/C++ interpreter


How to debug WildC script:
        You can give -S, -s, -t, -T and -b debug option to the wildc. 
        Debugger works in the same way as CINT. You can set break point, 
        step through and trace your source code.

    Example:
        $ wildc -S grcalc.wc

        You MUST USE 'p WildCard_Exit()' to terminate process in the debug
        mode. Otherwise, behavior is not guaranteed.

    Example:
        wildc> p WildCard_Exit()


In Tcl:

  ceval {C/C++ expression}
	ceval can be used in Tcl/Tk script to evaluate C/C++ expression. 
	Result is returned as form of string.

    Example:
	ceval 1+2
	ceval printf("abc\n");
	button .b -text "button"
	bind .b <Button-1> {ceval CallBack()}

In C/C++:

  #pragma tcl <interp> <link variable list>
  #pragma endtcl
	#pragma tcl statement starts Tcl/Tk interpreter. 
	If it appears within a function, Tcl/Tk script will be evaluated 
	when the function is executed.
	If it appears in a global scope, Tcl/Tk script will be immediately 
	evaluated. #pragma endtcl can be omitted in global scope. In this
	case, WildCard_MainLoop() is implicitly called to start X11 event
	loop.

    Example:
	f() {
	  #pragma tcl interp
	    # tcl/tk commands evaluated when f() is called
	  #pragma endtcl
	}
	#pragma tcl interp
	  # tcl/tk commands evaluated when source code loading
	#pragma endtcl
	#pragma tcl interp
	  # tcl/tk commands evaluated when source code loading 
	  # and implicitly calls WildCard_MainLoop() because #pragma
	  # endtcl is omitted.


	A parameter <interp> must be a valid Tcl_Interp* object. A global
	object 'interp' is preregistered in the WildCard environment for
	user's convenience.
	After the <interp> you can list up arbitrary number of C/C++ variables
	to link to Tcl/Tk interpreters. Variables must be type of int, double
	or char* and must be a simple object name. Variables are unlinked at
	#pragma endtcl.

    Example:
        f(char *color) {
	  static int i=0;
	  #pragma tcl interp color i
	    button -.b$i -text "Button Test" -bg $color
	  #pragma endtcl
	  ++i;
	}


  WildCard_MainLoop();
  WildCard_Exit();
	WildCard_MainLoop() and WildCard_Exit() are API function for the
	WildCard interpreter. WildCard_MainLoop() starts X11 event loop
	to get mouse and keyboard events. The events are handled by Tcl/Tk
	interpreter.
	WildCard_Exit() must be used to quit WildCard interpreter. exit()
	will not work.

    Example:
        void DrawGraphics() {
	  #pragma tcl interp
	    button .b -text "Exit"
	    bind .b <Button-1> {ceval Exit_CallBack()}
	    pack .b
	  #pragma endtcl
	}
        void Exit_CallBack() { WildCard_Exit(); }
	main() {
	  DrawGraphics();
          WildCard_MainLoop();
	}

  Tcl_xxx()
  Tk_xxx()
	The WildCard interpreter embeds all of the Tcl/Tk methods as 
	precompiled library. You can access Tcl/Tk intrinsics from
	C/C++ interpreter. You don't need to use them in most of the cases.

    Example:
        int i; 
	Tcl_LinkVar(interp,"i",(char*)(&i),TCL_LINK_INT);
	    // parenthesis around (&i) is needed in WildCard.


