/* /% C %/ */
/***********************************************************************
 * The WildCard interpreter
 ************************************************************************
 * Source file WildCard.c
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

/* Tk main interpreter */
extern Tcl_Interp *interp;
extern FILE *G__serr;

/**********************************************************************
* cintfront()
* WindCard_MainLoop()
*
*  Tcl/Tk Event Loop and CINT prompt switched by SIGINT
**********************************************************************/
static void cintfront(sigdmy)
int sigdmy;
{
  signal(SIGINT,cintfront);
  while(0==G__pause());
  signal(SIGINT,cintfront);
  fprintf(stdout,"%% ");
  fflush(stdout);
}

void WildCard_MainLoop() {
  signal(SIGINT,cintfront);
  G__init_process_cmd();
  fprintf(stdout,"%% ");
  fflush(stdout);
  Tk_MainLoop();
}

/**********************************************************************
* WildCard_Exit();
*
*  Tcl/Tk Event Loop and CINT prompt switched by SIGINT
**********************************************************************/
void WildCard_Exit() {
    /*
     * Don't exit directly, but rather invoke the Tcl "exit" command.
     * This gives the application the opportunity to redefine "exit"
     * to do additional cleanup.
     */
    static int stat=0;
    if(0==stat) {
      stat=1;
      Tcl_Eval(interp, "exit");
      exit(1);
      stat=0;
    }
}

/**********************************************************************
* WildCard_AllocConsole();
**********************************************************************/
int WildCard_AllocConsole() {
    return(0);
}

/**********************************************************************
* WildCard_FreeConsole();
**********************************************************************/
int WildCard_FreeConsole() {
    return(0);
}

/***************************************************************************
* G__concatcommand()
***************************************************************************/
int G__concatcommand(char *command,char *arg)
{
  static int brace = 0;
  static int doublequote = 0 ;
  static int singlequote = 0 ;
  static int pos = 0;
  int i,len;
  int backslash=0;
  len = strlen(arg);
  for(i=0;i<len;i++) {
    switch(arg[i]) {
    case '"':
      if(0==singlequote) doublequote ^= 1;
      break;
    case '\'':
      if(0==doublequote) singlequote ^= 1;
      break;
    case '{':
      if(0==singlequote&&0==doublequote) ++brace;
      break;
    case '}':
      if(0==singlequote&&0==doublequote) --brace;
      break;
    case '\\': 
      ++i;
      if(i==len) {
	arg[i-1]=' ';
	backslash=1;
      }
      break;
    }
  }
  strcpy(command+pos,arg);
  pos = strlen(command);

  if(0==singlequote&&0==doublequote&&0==brace&&0==backslash) {
    pos=0;
    return(1);
  }
  else return(0);
}

/**********************************************************************
* G__TclInserted()
*
*  Tcl/tk script can be inserted in a C/C++ source code. G__TclInserted
*  is a callback function when #pragma tcl statement appears in C/C++.
*
*  #pragma tcl [interp] <var1 <var2 <var3 <...>>>>
*              -----------------------------------
*                            args
**********************************************************************/
void G__TclInserted(args)
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

  char command[G__LONGLINE];
  int p;

  /* pragma arguments */
  char argbufb[G__ONELINE];
  char *argv[G__MAXARG];
  int argc=0;
  int i;
  G__value reg;

  /*************************************************************
   * execute tcl script only in following case
   *  prerun in global scope
   *  execution in function
   *************************************************************/
  G__set_asm_noverflow(0); /* Disable loop compilation */
  if(0==G__get_no_exec() && 0==G__get_no_exec_compile() &&
     ((1==G__getPrerun() && -1==G__getFuncNow()) ||
      (0==G__getPrerun() && -1!=G__getFuncNow()))){
    flag=1;
  }
  else {
    flag=0;
  }

  /*************************************************************
   * parse '#pragma tcl' arguments
   *************************************************************/
  if(flag) {
    /*************************************************************
     * Split arguments
     *************************************************************/
    strcpy(argbufb,args);
    G__split(args,argbufb,&argc,argv);

    if(0==argc) {
#ifdef G__WIN32
      G__allocconsole();
#endif
      fprintf(G__getSerr(),"Error: %s\n",args);
      G__genericerror(" Usage:#pragma tcl [interp] <var1 <var2 <var3<...>>>>");
      G__setReturn(2);
      return;
    }
    else {

      /*************************************************************
       * Get Tcl interp
       *************************************************************/
      localinterp = (Tcl_Interp*)G__int(G__calc(argv[1]));
      if(!localinterp) localinterp = interp;

      /*************************************************************
       * Link Variable
       *************************************************************/
      for(i=2;i<=argc;i++) {
	reg = G__calc(argv[i]);
	if(reg.ref) {
	  switch(reg.type) {
	  case 'C':
	    Tcl_LinkVar(interp,argv[i],(char*)reg.ref,TCL_LINK_STRING);
	    break;
	  case 'i':
	    Tcl_LinkVar(interp,argv[i],(char*)reg.ref,TCL_LINK_INT);
	    break;
	  case 'd':
	    Tcl_LinkVar(interp,argv[i],(char*)reg.ref,TCL_LINK_DOUBLE);
	    break;
	  default:
#ifdef G__WIN32
	    G__allocconsole();
#endif
	    fprintf(G__getSerr()
		    ,"Error: #pragma tcl, variable %s improper type"
		    ,argv[i]);
	    G__genericerror(NULL);
	    G__setReturn(2);
	    while((--i)>0) Tcl_UnlinkVar(interp,argv[i]);
	    return;
	  }
	}
	else {
#ifdef G__WIN32
	  G__allocconsole();
#endif
	  fprintf(G__getSerr()
		  ,"Error: #pragma tcl, variable %s can not get reference"
		  ,argv[i]);
	  G__genericerror(NULL);
	  G__setReturn(2);
	  while((--i)>0) Tcl_UnlinkVar(interp,argv[i]);
	  return;
	}
      }
    }
  }

  /*************************************************************
   * Read and Evaluate Tcl/Tk command
   *************************************************************/
#ifdef G__WIN32
  if(G__getDispsource()) G__allocconsole();
#endif
  /* Read and copy Tcl statements until #pragma endtcl or EOF */
  while((argn<2 || strcmp(arg[1],"#pragma")!=0 || strcmp(arg[2],"endtcl")!=0)
	&& G__readline(G__getIfileFp(),line,argbuf,&argn,arg)) {
    G__incIfileLineNumber();
    if(G__getDispsource()) fprintf(G__getSerr(),"%s\n%-4d "
				   ,arg[0],G__getIfileLineNumber());
    if(0==argn || '#'!=arg[1][0]) {
      if(flag) {
        if(G__concatcommand(command,arg[0])) {
	  code=Tcl_Eval(localinterp,command);
	  if(TCL_OK!=code) {
#ifdef G__WIN32
	    G__allocconsole();
#endif
	    fprintf(G__getSerr(),"%s\n",localinterp->result);
	  }
	}
      }
    }
  }
  if(flag) {
    if(G__NOMAIN==G__getIsMain()) G__setIsMain(G__TCLMAIN);
  }


  /*************************************************************
   * If source file ended without #pragma endtcl, Tcl/Tk's Event
   * loop appears as frontend.
   *************************************************************/
  if((argn<2 || strcmp(arg[1],"#pragma")!=0 || strcmp(arg[2],"endtcl")!=0)) {
    G__setPrerun(0);
    G__setDebug(G__getDebugTrace());
    G__setStep(G__getStepTrace());
    G__setdebugcond();
    WildCard_MainLoop();
    /* WildCard_MainLoop() will never return */
    exit(0);
  }

  /*************************************************************
   * Unlink variable
   *************************************************************/
  i=argc;
  while(i>1) {
    Tcl_UnlinkVar(interp,argv[i]);
    --i;
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
*
*  ceval [expression]           => G__calc(expression)
**********************************************************************/
int CevalCmd(clientdata,interp,argc,argv)
ClientData clientdata;
Tcl_Interp *interp;
int argc;
char *argv[];
{
  G__value result;
  char buf[G__LONGLINE];
  char *p;
  int i;

  /***************************************************
  * concatinate arguments
  ***************************************************/
  p = buf;
  *p='\0';
  for (i=1;i<argc;i++) {
    sprintf(p,"%s ",argv[i]);
    p = p+strlen(p);
  }

  /***************************************************
  * evaluate C/C++ expression by CINT
  ***************************************************/
  result=G__calc(buf);

  /***************************************************
  * return result to Tcl interpreter
  ***************************************************/
  switch(result.type) {
  case 'd':
  case 'f':
    sprintf(buf,"%g",G__double(result));
    break;
  case 'C':
    sprintf(buf,"%s",(char*)G__int(result));
    break;
  case '\0':
    strcpy(buf,"NULL");
    break;
  default:
    sprintf(buf,"%d",G__int(result));
    break;
  }
  Tcl_SetResult(interp,buf,TCL_VOLATILE);
  G__security_recover(G__serr);
  return(TCL_OK);
}

/**********************************************************************
* C/C++ declaration
*
*  cdecl [declaration]           => G__process_cmd(expression)
**********************************************************************/
int CdeclCmd(clientdata,interp,argc,argv)
ClientData clientdata;
Tcl_Interp *interp;
int argc;
char *argv[];
{
  int result;
  char buf[G__LONGLINE];
  char *p;
  int i;
  char prompt[G__ONELINE];
  int more=0;

  /***************************************************
  * concatinate arguments
  ***************************************************/
  p = buf;
  *p='{';
  ++p;
  *p='\0';
  for (i=1;i<argc;i++) {
    sprintf(p,"%s ",argv[i]);
    p = p+strlen(p);
  }
  *p=';';
  ++p;
  *p='}';
  ++p;
  *p='\0';

  /***************************************************
  * evaluate C/C++ expression by CINT
  ***************************************************/
  result=G__process_cmd(buf,prompt,&more,0,0);

  sprintf(buf,"%d",result);
  Tcl_SetResult(interp,buf,TCL_VOLATILE);
  G__security_recover(G__serr);
  return(TCL_OK);
}

/**********************************************************************
* C/C++ statements evaluater in the Tcl wrapper
* NOT WORKING WELL YET.
**********************************************************************/
int CintCmd(clientdata,interp,argc,argv)
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
  result=G__process_cmd(buf,prompt,&i,0,0);
  G__security_recover(G__serr);
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
