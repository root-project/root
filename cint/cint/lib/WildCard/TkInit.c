/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* /% C %/ */
/***********************************************************************
 * The WildCard interpreter
 ************************************************************************
 * Source file TkInit.c
 ************************************************************************
 * Description:
 *  This file is modified based on tkMain.c. Function Tk_Main() is split
 *  into 3 parts  G__Tk_Init(), G__Tk_MainLoop() and G__Tk_Exit()
 *  Original version of Tk_Main() is included in the -ltk library. This
 *  source file provides, initialization only version of Tk_Main() that
 *  is G__Tk_Init().
 ************************************************************************
 * Copyright (c) 1990-1994 The Regents of the University of California.
 * Copyright (c) 1994-1995 Sun Microsystems, Inc.
 * Copyright (c) 1996-1996 Masaharu Goto
 * Original      tk/tkMain.c
 * Modifier      Masaharu Goto
 * Date          9/Oct/1996
 *
 ************************************************************************/

/* 
 * tkMain.c --
 *
 *	This file contains a generic main program for Tk-based applications.
 *	It can be used as-is for many applications, just by supplying a
 *	different appInitProc procedure for each specific application.
 *	Or, it can be used as a template for creating new main programs
 *	for Tk applications.
 *
 * Copyright (c) 1990-1994 The Regents of the University of California.
 * Copyright (c) 1994-1995 Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 */

static char sccsid[] = "@(#) tkMain.c 1.114 95/06/08 11:43:00";

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <tcl.h>
#include <tk.h>
#ifdef NO_STDLIB_H
#   include "compat/stdlib.h"
#else
#   include <stdlib.h>
#endif

/*
 * Declarations for various library procedures and variables (don't want
 * to include tkInt.h or tkPort.h here, because people might copy this
 * file out of the Tk source directory to make their own modified versions).
 * Note: don't declare "exit" here even though a declaration is really
 * needed, because it will conflict with a declaration elsewhere on
 * some systems.
 */

extern int		isatty _ANSI_ARGS_((int fd));
extern int		read _ANSI_ARGS_((int fd, char *buf, size_t size));
extern char *		strrchr _ANSI_ARGS_((CONST char *string, int c));

/*
 * Global variables used by the main program:
 */

static Tk_Window mainWindow;	/* The main window for the application.  If
				 * NULL then the application no longer
				 * exists. */
Tcl_Interp *interp;	/* Interpreter for this application. */
static Tcl_DString command;	/* Used to assemble lines of terminal input
				 * into Tcl commands. */
static int tty;			/* Non-zero means standard input is a
				 * terminal-like device.  Zero means it's
				 * a file. */
static char errorExitCmd[] = "exit 1";

/*
 * Command-line options:
 */

static int synchronize = 0;
static char *fileName = NULL;
static char *name = NULL;
static char *display = NULL;
static char *geometry = NULL;

static Tk_ArgvInfo argTable[] = {
    {"-display", TK_ARGV_STRING, (char *) NULL, (char *) &display,
	"Display to use"},
    {"-geometry", TK_ARGV_STRING, (char *) NULL, (char *) &geometry,
	"Initial geometry for window"},
    {"-name", TK_ARGV_STRING, (char *) NULL, (char *) &name,
	"Name to use for application"},
    {"-sync", TK_ARGV_CONSTANT, (char *) 1, (char *) &synchronize,
	"Use synchronous mode for display server"},
    {(char *) NULL, TK_ARGV_END, (char *) NULL, (char *) NULL,
	(char *) NULL}
};

/*
 * Forward declarations for procedures defined later in this file:
 */

static void		Prompt _ANSI_ARGS_((Tcl_Interp *interp, int partial));
static void		StdinProc _ANSI_ARGS_((ClientData clientData,
			    int mask));

/*
 *----------------------------------------------------------------------
 *
 * Tk_Main --
 *
 *	Main program for Wish and most other Tk-based applications.
 *
 * Results:
 *	None. This procedure never returns (it exits the process when
 *	it's done.
 *
 * Side effects:
 *	This procedure initializes the Tk world and then starts
 *	interpreting commands;  almost anything could happen, depending
 *	on the script being interpreted.
 *
 *----------------------------------------------------------------------
 */
/**********************************************************************
* Initialization part of Tk_Main()
**********************************************************************/
#define HPFIX

void
G__Tk_Init(argc, argv, appInitProc)
    int argc;				/* Number of arguments. */
    char **argv;			/* Array of argument strings. */
#ifdef HPFIX
    int (*appInitProc) _ANSI_ARGS_((Tcl_Interp*));
#else
    Tcl_AppInitProc *appInitProc;	/* Application-specific initialization
					 * procedure to call after most
					 * initialization but before starting
					 * to execute commands. */
#endif
{
    char *args, *p, *msg, *argv0, *class;
    char buf[20];
    int code;
    size_t length;
    char *dispdev;

    interp = Tcl_CreateInterp();
#ifdef TCL_MEM_DEBUG
    Tcl_InitMemory(interp);
#endif

    /*
     * Parse command-line arguments.  A leading "-file" argument is
     * ignored (a historical relic from the distant past).  If the
     * next argument doesn't start with a "-" then strip it off and
     * use it as the name of a script file to process.  Also check
     * for other standard arguments, such as "-geometry", anywhere
     * in the argument list.
     */

    argv0 = argv[0];
    if (argc > 1) {
	length = strlen(argv[1]);
	if ((length >= 2) && (strncmp(argv[1], "-file", length) == 0)) {
	    argc--;
	    argv++;
	}
    }
    if ((argc > 1) && (argv[1][0] != '-')) {
	fileName = argv[1];
	argc--;
	argv++;
    }
    if (Tk_ParseArgv(interp, (Tk_Window) NULL, &argc, argv, argTable, 0)
	    != TCL_OK) {
	fprintf(stderr, "%s\n", interp->result);
	exit(1);
    }
    if (name == NULL) {
	if (fileName != NULL) {
	    p = fileName;
	} else {
	    p = argv[0];
	}
	name = strrchr(p, '/');
	if (name != NULL) {
	    name++;
	} else {
	    name = p;
	}
    }

    /*
     * Make command-line arguments available in the Tcl variables "argc"
     * and "argv".
     */

    args = Tcl_Merge(argc-1, argv+1);
    Tcl_SetVar(interp, "argv", args, TCL_GLOBAL_ONLY);
    ckfree(args);
    sprintf(buf, "%d", argc-1);
    Tcl_SetVar(interp, "argc", buf, TCL_GLOBAL_ONLY);
    Tcl_SetVar(interp, "argv0", (fileName != NULL) ? fileName : argv0,
	    TCL_GLOBAL_ONLY);

    /*
     * If a display was specified, put it into the DISPLAY
     * environment variable so that it will be available for
     * any sub-processes created by us.
     */

    if (display != NULL) {
	Tcl_SetVar2(interp, "env", "DISPLAY", display, TCL_GLOBAL_ONLY);
    }

    /*
     * Initialize the Tk application.  If a -name option was provided,
     * use it;  otherwise, if a file name was provided, use the last
     * element of its path as the name of the application; otherwise
     * use the last element of the program name.  For the application's
     * class, capitalize the first letter of the name.
     */

    if (name == NULL) {
	p = (fileName != NULL) ? fileName : argv0;
	name = strrchr(p, '/');
	if (name != NULL) {
	    name++;
	} else {
	    name = p;
	}
    }
    class = (char *) ckalloc((unsigned) (strlen(name) + 1));
    strcpy(class, name);
    class[0] = toupper((unsigned char) class[0]);

#ifdef G__LINUX120
    mainWindow = Tk_CreateMainWindow(interp, display, name, class);
#else
    dispdev=getenv("DISPLAY");
    if(!dispdev) dispdev=":0.0";
    mainWindow = Tk_CreateWindow(interp, (Tk_Window)NULL, name, dispdev);
#endif

    ckfree(class);
    if (mainWindow == NULL) {
	fprintf(stderr, "%s\n", interp->result);
	exit(1);
    }
    if (synchronize) {
	XSynchronize(Tk_Display(mainWindow), True);
    }

    /*
     * Set the "tcl_interactive" variable.
     */

    tty = isatty(0);
    Tcl_SetVar(interp, "tcl_interactive",
	    ((fileName == NULL) && tty) ? "1" : "0", TCL_GLOBAL_ONLY);

    /*
     * Set the geometry of the main window, if requested.  Put the
     * requested geometry into the "geometry" variable.
     */

    if (geometry != NULL) {
	Tcl_SetVar(interp, "geometry", geometry, TCL_GLOBAL_ONLY);
	code = Tcl_VarEval(interp, "wm geometry . ", geometry, (char *) NULL);
	if (code != TCL_OK) {
	    fprintf(stderr, "%s\n", interp->result);
	}
    }

    /*
     * Invoke application-specific initialization.
     */

    if ((*appInitProc)(interp) != TCL_OK) {
	fprintf(stderr, "application-specific initialization failed: %s\n",
		interp->result);
    }




    /*
     * Invoke the script specified on the command line, if any.
     */

    if (fileName != NULL) {
	code = Tcl_VarEval(interp, "source ", fileName, (char *) NULL);
	if (code != TCL_OK) {
	    goto error;
	}
	tty = 0;
    } else {
	/*
	 * Commands will come from standard input, so set up an event
	 * handler for standard input.  Evaluate the .rc file, if one
	 * has been specified, set up an event handler for standard
	 * input, and print a prompt if the input device is a terminal.
	 */

#if 0
	if (tcl_RcFileName != NULL) {
	    Tcl_DString buffer;
	    char *fullName;
	    FILE *f;
    
	    fullName = Tcl_TildeSubst(interp, tcl_RcFileName, &buffer);
	    if (fullName == NULL) {
		fprintf(stderr, "%s\n", interp->result);
	    } else {
		f = fopen(fullName, "r");
		if (f != NULL) {
		    code = Tcl_EvalFile(interp, fullName);
		    if (code != TCL_OK) {
			fprintf(stderr, "%s\n", interp->result);
		    }
		    fclose(f);
		}
	    }
	    Tcl_DStringFree(&buffer);
	}
#endif
	Tk_CreateFileHandler(0, TK_READABLE, StdinProc, (ClientData) 0);
	if (tty) {
	    /* Prompt(interp, 0); */
	}
    }
    fflush(stdout);
    Tcl_DStringInit(&command);
    Tcl_ResetResult(interp);

    /*
     * Loop infinitely, waiting for commands to execute.  When there
     * are no windows left, Tk_MainLoop returns and we exit.
     */

    /*************************************************************
    * This function returns after the initialization
    *************************************************************/
    return;

error:
    msg = Tcl_GetVar(interp, "errorInfo", TCL_GLOBAL_ONLY);
    if (msg == NULL) {
	msg = interp->result;
    }
    fprintf(stderr, "%s\n", msg);
    Tcl_Eval(interp, errorExitCmd);
}

/**********************************************************************
* Dummy function 
**********************************************************************/
void G__Tk_MainLoop() {
  Tk_MainLoop();
}

/**********************************************************************
* Dummy function 
**********************************************************************/
G__Tk_Exit() {
    /*
     * Don't exit directly, but rather invoke the Tcl "exit" command.
     * This gives the application the opportunity to redefine "exit"
     * to do additional cleanup.
     */

    Tcl_Eval(interp, "exit");
    exit(1);
}

/*
 *----------------------------------------------------------------------
 *
 * StdinProc --
 *
 *	This procedure is invoked by the event dispatcher whenever
 *	standard input becomes readable.  It grabs the next line of
 *	input characters, adds them to a command being assembled, and
 *	executes the command if it's complete.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Could be almost arbitrary, depending on the command that's
 *	typed.
 *
 *----------------------------------------------------------------------
 */

    /* ARGSUSED */
static void
StdinProc(clientData, mask)
    ClientData clientData;		/* Not used. */
    int mask;				/* Not used. */
{
#define BUFFER_SIZE 4000
    char input[BUFFER_SIZE+1];
    static int gotPartial = 0;
    char *cmd;
    int code, count;

    count = read(fileno(stdin), input, BUFFER_SIZE);
    if (count <= 0) {
	if (!gotPartial) {
	    if (tty) {
		Tcl_Eval(interp, "exit");
		exit(1);
	    } else {
		Tk_DeleteFileHandler(0);
	    }
	    return;
	} else {
	    count = 0;
	}
    }
    cmd = Tcl_DStringAppend(&command, input, count);
    if (count != 0) {
	if ((input[count-1] != '\n') && (input[count-1] != ';')) {
	    gotPartial = 1;
	    goto prompt;
	}
	if (!Tcl_CommandComplete(cmd)) {
	    gotPartial = 1;
	    goto prompt;
	}
    }
    gotPartial = 0;

    /*
     * Disable the stdin file handler while evaluating the command;
     * otherwise if the command re-enters the event loop we might
     * process commands from stdin before the current command is
     * finished.  Among other things, this will trash the text of the
     * command being evaluated.
     */

#ifndef TCL_EVAL_GLOBAL
#define TCL_EVAL_GLOBAL 0
#endif

    Tk_CreateFileHandler(0, 0, StdinProc, (ClientData) 0);
    code = Tcl_RecordAndEval(interp, cmd, TCL_EVAL_GLOBAL);
    Tk_CreateFileHandler(0, TK_READABLE, StdinProc, (ClientData) 0);
    Tcl_DStringFree(&command);
    if (*interp->result != 0) {
	if ((code != TCL_OK) || (tty)) {
	    /*
	     * The statement below used to call "printf", but that resulted
	     * in core dumps under Solaris 2.3 if the result was very long.
	     */

	    puts(interp->result);
	}
    }

    /*
     * Output a prompt.
     */

    prompt:
    if (tty) {
	Prompt(interp, gotPartial);
    }
    Tcl_ResetResult(interp);
}

/*
 *----------------------------------------------------------------------
 *
 * Prompt --
 *
 *	Issue a prompt on standard output, or invoke a script
 *	to issue the prompt.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	A prompt gets output, and a Tcl script may be evaluated
 *	in interp.
 *
 *----------------------------------------------------------------------
 */

static void
Prompt(interp, partial)
    Tcl_Interp *interp;			/* Interpreter to use for prompting. */
    int partial;			/* Non-zero means there already
					 * exists a partial command, so use
					 * the secondary prompt. */
{
    char *promptCmd;
    int code;

    promptCmd = Tcl_GetVar(interp,
	partial ? "tcl_prompt2" : "tcl_prompt1", TCL_GLOBAL_ONLY);
    if (promptCmd == NULL) {
	defaultPrompt:
	if (!partial) {
	    fputs("% ", stdout);
	}
    } else {
	code = Tcl_Eval(interp, promptCmd);
	if (code != TCL_OK) {
	    Tcl_AddErrorInfo(interp,
		    "\n    (script that generates prompt)");
	    fprintf(stderr, "%s\n", interp->result);
	    goto defaultPrompt;
	}
    }
    fflush(stdout);
}

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
