/* /% C %/ */
/***********************************************************************
 * The WildCard interpreter
 ************************************************************************
 * Source file TclIns.c
 ************************************************************************
 * Description:
 *  This source file contains interface routines between CINT and Tcl
 *  interpreters.
 ************************************************************************
 * Copyright(c) 1996-1997  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <tk.h>
#include "G__ci.h"

/* CINT internal variables. This is not a good manner. Maybe clean this up
* later. */
extern struct G__input_file G__ifile;
extern int G__return;  /* 0:normal,1:return,2:exit()or'q',3:'qq'(not used)*/
extern int G__func_now;
extern int G__prerun;
extern short G__dispsource;
extern FILE *G__serr;
extern int G__ismain;           /* is there a main function */
extern int G__step;
extern int G__steptrace;
extern int G__debug;
extern int G__debugtrace;
extern int G__prerun;

/* Tk main interpreter */
extern Tcl_Interp *interp;

/**********************************************************************
* cintfront()
* WindCard_MainLoop()
*
*  Tcl/Tk Event Loop and CINT prompt switched by SIGINT
**********************************************************************/
static void cintfront() {
  signal(SIGINT,cintfront);
  while(0==G__pause());
  signal(SIGINT,cintfront);
  fprintf(stdout,"%% ");
  fflush(stdout);
}

void WildCard_MainLoop() {
  signal(SIGINT,cintfront);
  G__init_process_cmd();
  Tk_MainLoop();
}

/**********************************************************************
* G__TclInserted()
*
*  Tcl/tk script can be inserted in a C/C++ source code. G__TclInserted
*  is a callback function when #pragma tcl statement appears in C/C++.
**********************************************************************/
static void G__TclInserted(args)
char *args;
{
  char line[G__LONGLINE];
  char argbuf[G__LONGLINE];
  int argn=0;
  char *arg[G__MAXARG];
  int code;
  Tcl_Interp *localinterp = (Tcl_Interp*)NULL;
  int flag;
  char tempfilename[G__MAXFILENAME];
  FILE *fp;

  /*************************************************************
   * execute tcl script only in following case
   *  prerun in global scope
   *  execution in function
   *************************************************************/
  if((1==G__prerun && -1==G__func_now) || (0==G__prerun && -1!=G__func_now)) {
    flag=1;
  }
  else {
    flag=0;
  }

  /*************************************************************
   * Get handle to Tcl interpreter.  If omitted, global wish
   * interpreter is used (which is the normal case). But you can
   * instantiate another Tcl interpreter too.
   *************************************************************/
  if(flag) {
    if(args && args[0]) localinterp = (Tcl_Interp*)G__int(G__calc(args));
    if(!localinterp) localinterp = interp;

    /*************************************************************
     * get tmpnam for tmp file
     *************************************************************/
    do {
      G__tmpnam(tempfilename);
      fp = fopen(tempfilename,"w");
    } while((FILE*)NULL==fp && G__setTMPDIR(tempfilename)) ;
    if(!fp) {
      G__genericerror("TCL script can not execute because tmpfile can't open");
      G__return=2;
      return;
    }
  }

  /*************************************************************
   * Copy Tcl/Tk script to a tmpfile, 
   *************************************************************/
  /* Read and copy Tcl statements until #pragma endtcl or EOF */
  while((argn<2 || strcmp(arg[1],"#pragma")!=0 || strcmp(arg[2],"endtcl")!=0)
	&& G__readline(G__ifile.fp,line,argbuf,&argn,arg)) {
    ++G__ifile.line_number;
    if(G__dispsource) fprintf(G__serr,"%s\n%-4d ",arg[0],G__ifile.line_number);
    if(0==argn || '#'!=arg[1][0]) {
      if(flag) fprintf(fp,"%s\n",arg[0]);
    }
  }
  fclose(fp);

  /*************************************************************
   * Load evaluate Tcl/Tk script
   *************************************************************/
  if(flag) {
    code=Tcl_EvalFile(localinterp,tempfilename);
    remove(tempfilename);
    /* set error return flag if Tcl returns error */
    if(code!=TCL_OK) {
      fprintf(G__serr,"Error running tcl script\n");
      fprintf(G__serr,"%s\n",localinterp->result);
      G__return=2;
    }
    if(G__NOMAIN==G__ismain) G__ismain=G__TCLMAIN;
  }

  /*************************************************************
   * If source file ended without #pragma endtcl, Tcl/Tk's Event
   * loop appears as frontend.
   *************************************************************/
  if((argn<2 || strcmp(arg[1],"#pragma")!=0 || strcmp(arg[2],"endtcl")!=0)) {
    G__prerun = 0;
    G__debug=G__debugtrace;
    G__step=G__steptrace;
    G__setdebugcond();
    WildCard_MainLoop();
  }
}

/**********************************************************************
* G__cinttk_init()
*
*  Add Tcl/Tk specific #pragma statement in cint parser.
**********************************************************************/
void G__cinttk_init()
{
  G__addpragma("tcl",G__TclInserted);
}



/**********************************************************************
* C/C++ expression evaluater in the Tcl wrapper
**********************************************************************/
int G__Tcl_calc(clientdata,interp,argc,argv)
ClientData clientdata;
Tcl_Interp *interp;
int argc;
char *argv[];
{
  G__value result;
  char buf[G__LONGLINE];
  char *p; 
  int i;
  p = buf;
  *p='\0';
  for (i=1;i<argc;i++) {
    sprintf(p,"%s ",argv[i]);
    p = p+strlen(p);
  }
  result=G__calc(buf);
  G__valuemonitor(result,buf);
  Tcl_SetResult(interp,buf,TCL_VOLATILE);
  return(TCL_OK);
}

/**********************************************************************
* C/C++ statements evaluater in the Tcl wrapper
* NOT WORKING WELL YET.
**********************************************************************/
int G__Tcl_cint(clientdata,interp,argc,argv)
ClientData clientdata;
Tcl_Interp *interp;
int argc;
char *argv[];
{
  int result;
  char buf[G__LONGLINE];
  char prompt[G__LONGLINE];
  char *p; 
  int i;
  p = buf;
  *p='\0';
  for (i=1;i<argc;i++) {
    sprintf(p,"%s ",argv[i]);
    p = p+strlen(p);
  }
  i=0;
  strcpy(prompt,"wildc>");
  result=G__process_cmd(buf,prompt,&i);
  return(TCL_OK);
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
