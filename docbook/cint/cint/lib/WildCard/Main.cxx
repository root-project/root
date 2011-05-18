/* /% C++ %/ */
/***********************************************************************
 * The WildCard interpreter
 ************************************************************************
 * Source file Main.c
 ************************************************************************
 * Description:
 *  main function to the WildCard interpreter. Select between CINT and Tcl
 *  depending on argument file extension.
 ************************************************************************
 * Copyright(c) 1996-1997  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <tk.h>
#include "G__ci.h"

#ifdef __cplusplus
extern "C" {
#endif
extern int Tcl_AppInit(Tcl_Interp *interp);
extern void G__Tk_Init(int argc,char **argv,int (*p2f)(Tcl_Interp*));
extern void G__Tk_Exit();
extern void WildCard_MainLoop();
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
int main(int argc,char **argv)
#else
int main(argc,argv)
int argc;
char **argv;
#endif
{
  int result;
  int flag=0;

  if(argc>1) {
    int i;

    /************************************************************
    * Scan arguments. If there is string sequence .c .C .h .H 
    * WildCard takes it as a C/C++ source code
    ************************************************************/
    for(i=1;i<argc;i++) {
      if(strstr(argv[i],".c") || strstr(argv[i],".C") ||
	 strstr(argv[i],".wc") || strstr(argv[i],".WC") ||
	 strstr(argv[i],".h") || strstr(argv[i],".H")) {
	flag=1;
	break;
      }
    }
  }
  else {
    char select[80];
    strcpy(select,G__input("Start WildCard as CINT mode(c) or Tcl mode(t)>"));
    switch(tolower(select[0])) {
    case 't':
      break;
    case 'c':
    default:
      flag=1;
      break;
    }
  }

  /************************************************************
   * Use CINT C/C++ interpreter as WildCard main environment
   ************************************************************/
  if(flag) {
    char *tclargv[1] ;
    tclargv[0] = "wildc";
    G__Tk_Init(1, tclargv, Tcl_AppInit); /* Initialize Tcl/Tk */
    G__setothermain(0);
    result=G__main(argc,argv); /* Start CINT as main environment */
    G__Tk_Exit();
    return(result);
  }

  /************************************************************
   * Use Tcl interpreter as WildCard main environment
   ************************************************************/
  else {
    G__setothermain(2); /*This forces G__main to return after initialization*/
    result=G__main(1,argv); /* CINT initialization */
    G__setothermain(0); /* Reset G__othermain flag */
#ifdef G__NEVER
    Tk_Main(argc, argv, Tcl_AppInit); /* Start WISH as main environment */
#else
    G__Tk_Init(argc, argv, Tcl_AppInit); /* Initialize Tcl/Tk */
    WildCard_MainLoop();
    G__Tk_Exit();
#endif
    return 0;	/* Needed only to prevent compiler warning. */
  }
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
