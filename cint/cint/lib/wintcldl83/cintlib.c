/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* /% C %/ */
/***********************************************************************
 * WildCard ,  CINT + Tcl/Tk on Windows-NT/95
 ************************************************************************
 * source file cintlib.c
 ************************************************************************
 * Description:
 *  wildc.dll initialization
 ************************************************************************
 * Copyright(c) 1996-1997  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 *  This file is modified based on example.c from Tcl7.5/Tk4.1 binary
 * distribution from Sun-Microsystems.
 ************************************************************************/
/*
 * example.c --
 *
 *	This file is an example of a Tcl dynamically loadable extension.
 *
 * Copyright (c) 1996 by Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 * SCCS: @(#) example.c 1.4 96/04/26 10:42:55
 */

#include <tcl.h>
#undef G__SPECIALSTDIO
#include "G__ci.h"

#if defined(__WIN32__) || defined(G__WIN32)
#   define WIN32_LEAN_AND_MEAN
#   include <windows.h>
#   undef WIN32_LEAN_AND_MEAN

/*
 * VC++ has an alternate entry point called DllMain, so we need to rename
 * our entry point.
 */

#   if defined(_MSC_VER)
#	define EXPORT(a,b) __declspec(dllexport) a b
#	define DllEntryPoint DllMain
#   else
#	if defined(__BORLANDC__)
#	    define EXPORT(a,b) a _export b
#	else
#	    define EXPORT(a,b) a b
#	endif
#   endif
#else
#   define EXPORT(a,b) a b
#endif

/*
 * Declarations for functions defined in this file.
 */

extern void G__add_setup_func(char *libname,G__incsetup func);
extern int G__security_recover(FILE* fout);
extern int G__process_cmd(char *cmd,char *prompt,int *more);
extern void G__c_setupwildc();
extern void WildCard_MainLoop();
extern void WildCard_Exit();
extern int WildCard_AllocConsole();
extern int WildCard_FreeConsole();
#ifdef G__OLDIMPLEMENTATION562
static int G__noconsole=1;
static int G__lockstdio=0;
extern FILE *G__stdout;
extern FILE *G__stderr;
extern FILE *G__stdin;
#endif
#ifdef G__OLDIMPLEMENTATION614
extern FILE *G__serr;
extern FILE *G__sout;
extern FILE *G__sin;
#endif


EXTERN EXPORT(int,Wildc_Init) _ANSI_ARGS_((Tcl_Interp *interp));

Tcl_Interp *interp;

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
      WildCard_AllocConsole();
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
	    Tcl_LinkVar(localinterp,argv[i],(char*)reg.ref,TCL_LINK_STRING);
	    break;
	  case 'i':
	    Tcl_LinkVar(localinterp,argv[i],(char*)reg.ref,TCL_LINK_INT);
	    break;
	  case 'd':
	    Tcl_LinkVar(localinterp,argv[i],(char*)reg.ref,TCL_LINK_DOUBLE);
	    break;
	  default:
	    WildCard_AllocConsole();
	    fprintf(G__getSerr()
		    ,"Error: #pragma tcl, variable %s improper type"
		    ,argv[i]);
	    G__genericerror(NULL);
	    G__setReturn(2);
	    while((--i)>0) Tcl_UnlinkVar(localinterp,argv[i]);
	    return;
	  }
	}
	else {
	  WildCard_AllocConsole();
	  fprintf(G__getSerr()
		  ,"Error: #pragma tcl, variable %s can not get reference"
		  ,argv[i]);
	  G__genericerror(NULL);
	  G__setReturn(2);
	  while((--i)>0) Tcl_UnlinkVar(localinterp,argv[i]);
	  return;
	}
      }

      /*************************************************************
       * get tmpnam for tmp file
       *************************************************************/
    }
  }

  /*************************************************************
   * Read and Evaluate Tcl/Tk command
   *************************************************************/
  if(G__getDispsource()) WildCard_AllocConsole();
  /* Read and copy Tcl statements until #pragma endtcl or EOF */
  while((argn<2 || strcmp(arg[1],"#pragma")!=0 || strcmp(arg[2],"endtcl")!=0)
	&& G__readline(G__getIfileFp(),line,argbuf,&argn,arg)) {
    G__incIfileLineNumber();
    if(G__getDispsource()) G__fprintf(G__getSerr(),"%s\n%-4d "
                               ,arg[0],G__getIfileLineNumber());
    if(0==argn || '#'!=arg[1][0]) {
      if(flag) {
        if(G__concatcommand(command,arg[0])) {
           code=Tcl_Eval(localinterp,command);
           if(TCL_OK!=code) {
             WildCard_AllocConsole();
             fprintf(G__getSerr(),"%s",localinterp->result);
             G__printlinenum();
           }
        }
      }
    }
  }
  /*
  if(flag) {
    if(G__NOMAIN==G__getIsMain()) G__setIsMain(G__TCLMAIN);
  }
  */

  /*************************************************************
   * Unlink variable
   *************************************************************/
  i=argc;
  while(i>1) {
    Tcl_UnlinkVar(localinterp,argv[i]);
    --i;
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
    /* WildCard_MainLoop(); */
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
static int WildcCmd(clientData, localinterp, argc, argv)
ClientData clientData;
Tcl_Interp *localinterp;
int argc;
char **argv;
{
    /*
     * Note that we must be very careful not to use TCL_DYNAMIC here
     * if we are compiling with a compiler other than Borland C++ 4.5
     * because the malloc/free routines are different.  Instead, we
     * should explicitly set the free proc to point to the free()
     * supplied with the run-time library used by the extension.
     */
    FILE *fpstdout;
    FILE *fpstderr;
    FILE *fpstdin;
    char *arg[] = { "cint" , "" };
    interp=localinterp;

    G__setothermain(2);
#ifndef G__OLDIMPLEMENTATION614
    G__setautoconsole(1);
    G__setmasksignal(1);
#endif
    G__main(argc,argv);
#ifndef G__OLDIMPLEMENTATION614
    G__security_recover(G__getSerr());
#else
    G__security_recover(G__serr);
#endif

    return TCL_OK;
}

/**********************************************************************
* C/C++ expression evaluater in the Tcl wrapper
*
*  ceval [expression]           => G__calc(expression)
**********************************************************************/
static int CevalCmd(clientdata,localinterp,argc,argv)
ClientData clientdata;
Tcl_Interp *localinterp;
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
  Tcl_SetResult(localinterp,buf,TCL_VOLATILE);
#ifndef G__OLDIMPLEMENTATION614
    G__security_recover(G__getSerr());
#else
    G__security_recover(G__serr);
#endif
  return(TCL_OK);
}
/**********************************************************************
* C/C++ declaration
*
*  cdecl [declaration]           => G__process_cmd(expression)
**********************************************************************/
static int CdeclCmd(clientdata,localinterp,argc,argv)
ClientData clientdata;
Tcl_Interp *localinterp;
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
  result=G__process_cmd(buf,prompt,&more);

  sprintf(buf,"%d",result);
  Tcl_SetResult(localinterp,buf,TCL_VOLATILE);
#ifndef G__OLDIMPLEMENTATION614
    G__security_recover(G__getSerr());
#else
    G__security_recover(G__serr);
#endif
  return(TCL_OK);
}

/*
 *----------------------------------------------------------------------
 *
 * DllEntryPoint --
 *
 *	This wrapper function is used by Windows to invoke the
 *	initialization code for the DLL.  If we are compiling
 *	with Visual C++, this routine will be renamed to DllMain.
 *	routine.
 *
 * Results:
 *	Returns TRUE;
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

#ifdef __WIN32__
BOOL APIENTRY
DllEntryPoint(hInst, reason, reserved)
    HINSTANCE hInst;		/* Library instance handle. */
    DWORD reason;		/* Reason this function is being called. */
    LPVOID reserved;		/* Not used. */
{
    switch (reason) {
    case DLL_PROCESS_ATTACH:
      break;
    case DLL_THREAD_ATTACH:
      break;
    case DLL_THREAD_DETACH:
      break;
    case DLL_PROCESS_DETACH:
      G__scratch_all();
      break;
    }
    return TRUE;
}
#endif

/*
 *----------------------------------------------------------------------
 *
 * Example_Init --
 *
 *	This procedure initializes the example command.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

EXPORT(int,Wildc_Init)(localinterp)
    Tcl_Interp *localinterp;
{
  char *arg[] = { "cint" , "" };
  interp=localinterp;
  Tcl_CreateCommand(localinterp, "wildc", WildcCmd, NULL, NULL);
  Tcl_CreateCommand(localinterp, "ceval", CevalCmd, NULL, NULL);
  Tcl_CreateCommand(localinterp, "cdecl",CdeclCmd,NULL,NULL);
  G__add_setup_func("CINTLIB",G__c_setupwildc);
  G__setothermain(2);
#ifndef G__OLDIMPLEMENTATION614
  G__setautoconsole(1);
  G__setmasksignal(1);
#endif
  G__main(1,arg);
  return Tcl_PkgProvide(localinterp, "cint", "1.0");
}


/**********************************************************************
* cintfront()
* WindCard_MainLoop()
*
*  Tcl/Tk Event Loop and CINT prompt switched by SIGINT
**********************************************************************/
static void cintfront(sigdmy)
int sigdmy;
{
  /* signal(SIGINT,cintfront); */
  while(0==G__pause());
  /* signal(SIGINT,cintfront); */
  fprintf(stdout,"%% ");
  fflush(stdout);
}

void WildCard_MainLoop() {
  /* signal(SIGINT,cintfront); */
  G__init_process_cmd();
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
     static int flag=0;
     if(0==flag) {
       flag=1;
       G__scratch_all();
       Tcl_Eval(interp, "exit");
       exit(1);
     }
}


#if 1
/**************************************************************************
* Create new console window and re-open stdio ports
**************************************************************************/



/**************************************************************************
* WildCard_AllocConsole()
**************************************************************************/
int WildCard_AllocConsole()
{
#ifndef G__OLDIMPLEMENTATION562
  return(G__AllocConsole());
#else
  BOOL result=TRUE;
  if(G__noconsole) {
	result=FreeConsole();
    result = AllocConsole();
	SetConsoleTitle("WILDC++(CINT-Tcl/Tk)");
	if(TRUE==result) {
	  G__stdout=G__sout=freopen("CONOUT$","w",stdout);
	  G__stderr=G__serr=freopen("CONOUT$","w",stderr);
	  G__stdin=G__sin=freopen("CONIN$","r",stdin);
	}
	G__noconsole=0;
  }
  return result;
#endif
}

/**************************************************************************
* WildCard_FreeConsole()
**************************************************************************/
int WildCard_FreeConsole()
{
#ifndef G__OLDIMPLEMENTATION562
  return(G__FreeConsole());
#else
  int result;
  if(!G__noconsole && !G__lockstdio) {
    G__noconsole=1;
	result=FreeConsole();
  }
  else {
	result=FALSE;
  }
  return result;
#endif
}

#ifdef G__OLDIMPLEMENTATION562
/**************************************************************************
* G__printf()
**************************************************************************/
int G__printf(char *fmt,...)
{
  int result;
  va_list argptr;
  va_start(argptr,fmt);
  G__lockstdio=1;
  if(G__noconsole) WildCard_AllocConsole();
  result = vprintf(fmt,argptr);
  G__lockstdio=0;
  va_end(argptr);
  return(result);
}


/**************************************************************************
* G__fprintf()
**************************************************************************/
int G__fprintf(FILE *fp,char *fmt,...)
{
  int result;
  va_list argptr;
  va_start(argptr,fmt);
  G__lockstdio=1;
  if(stdout==fp||stderr==fp) {
    if(G__noconsole) WildCard_AllocConsole();
  }
  result = vfprintf(fp,fmt,argptr);
  G__lockstdio=0;
  va_end(argptr);
  return(result);
}

/**************************************************************************
* G__fputc()
**************************************************************************/
int G__fputc(int character,FILE *fp)
{
  int result;
  G__lockstdio=1;
  if(stdout==fp||stderr==fp) {
    if(G__noconsole) WildCard_AllocConsole();
  }
  result=fputc(character,fp);
  G__lockstdio=0;
  return(result);
}

/**************************************************************************
* G__putchar()
**************************************************************************/
int G__putchar(int character)
{
   int result;
   G__lockstdio=1;
   if(G__noconsole) WildCard_AllocConsole();
   result=putchar(character);
   G__lockstdio=0;
   return(result);
}

/**************************************************************************
* G__fputs()
**************************************************************************/
int G__fputs(char *string,FILE *fp)
{
  int result;
  G__lockstdio=1;
  if(stdout==fp||stderr==fp) {
    if(G__noconsole) WildCard_AllocConsole();
  }
  result=fputs(string,fp);
  G__lockstdio=0;
  return(result);
}

/**************************************************************************
* G__puts()
**************************************************************************/
int G__puts(char *string)
{
   int result;
   G__lockstdio=1;
   if(G__noconsole) WildCard_AllocConsole();
   result=puts(string);
   G__lockstdio=0;
   return(result);
}

/**************************************************************************
* G__fgets()
**************************************************************************/
char *G__fgets(char *string,int n,FILE *fp)
{
  char *result;
  G__lockstdio=1;
  if(fp==stdin) {
    if(G__noconsole) WildCard_AllocConsole();
  }
  result=fgets(string,n,fp);
  G__lockstdio=0;
  return(result);
}
/**************************************************************************
* G__gets()
**************************************************************************/
char *G__gets(char *buffer)
{
   char *result;
   G__lockstdio=1;
   if(G__noconsole) WildCard_AllocConsole();
   result=gets(buffer);
   G__lockstdio=0;
   return(result);
}
#endif /* ON562 */



#else /* following case is never used */

/**************************************************************************
* STDIO function emulated by Tcl puts
**************************************************************************/

#define G__DISP1
/**************************************************************************
* G__TclDisplay()
**************************************************************************/
int G__TclDisplay(char *string)
{
#if defined(G__DISP2)
  char com[G__ONELINE];
  char *p;
  static int stat=1;
  static int line=0;
  if(stat) {
    sprintf(com,"text .G__text -relief raised -bd 2");
    Tcl_Eval(interp,com);
    sprintf(com,"pack .G__text");
    Tcl_Eval(interp,com);
    stat=0;
  }
  p = strchr(string,'\n');
  if(p) {
    ++line;
    if(line>15) {
      line=0;
      sprintf(com,".G__text delete 1.0 end");
      Tcl_Eval(interp,com);
    }
  }
  sprintf(com,".G__text insert end {%s}",string);
  Tcl_Eval(interp,com);
#elif defined(G__DISP1)
  char com[G__LONGLINE];
  sprintf(com,"puts -nonewline {%s}",string);
  Tcl_Eval(interp,com);
#else
  static char buf[G__LONGLINE];
  static int pos=0;
  char com[G__LONGLINE];
  int i;
  int len;
  len=strlen(string);
  for(i=0;i<len;i++) {
    switch(string[i]) {
    case '\n':
      buf[pos]='\0';
      sprintf(com,"puts {%s}",buf);
      Tcl_Eval(interp,com);
      pos=0;
      break;
    default:
      buf[pos++] = string[i];
      break;
    }
  }
#endif
  return 1;
}

/**************************************************************************
* G__printf()
**************************************************************************/
int G__printf(char *fmt,...)
{
  char buf[G__LONGLINE];
  int result;
  va_list argptr;
  va_start(argptr,fmt);
  result = vsprintf(buf,fmt,argptr);
  va_end(argptr);
  G__TclDisplay(buf);
  return(result);
}

/**************************************************************************
* G__fprintf()
**************************************************************************/
int G__fprintf(FILE *fp,char *fmt,...)
{
  char buf[G__LONGLINE];
  int result;
  va_list argptr;
  va_start(argptr,fmt);
  if(stdout==fp||stderr==fp) {
    result = vsprintf(buf,fmt,argptr);
    G__TclDisplay(buf);
  }
  else {
    result = vfprintf(fp,fmt,argptr);
  }
  va_end(argptr);
  return(result);
}

/**************************************************************************
* G__fputc()
**************************************************************************/
int G__fputc(int character,FILE *fp)
{
  char buf[10];
  if(stdout==fp||stderr==fp) {
    sprintf(buf,"%c",character);
    G__TclDisplay(buf);
  }
  else {
    fputc(character,fp);
  }
  return(character);
}

/**************************************************************************
* G__putchar()
**************************************************************************/
int G__putchar(int character)
{
  char buf[10];
  sprintf(buf,"%c",character);
  G__TclDisplay(buf);
  return(character);
}

/**************************************************************************
* G__fputs()
**************************************************************************/
int G__fputs(char *string,FILE *fp)
{
  if(stdout==fp||stderr==fp) {
    G__TclDisplay(string);
  }
  else {
    return(fputs(string,fp));
  }
}

/**************************************************************************
* G__puts()
**************************************************************************/
int G__puts(char *string)
{
  G__TclDisplay(string);
  return(1);
}

/**************************************************************************
* G__fgets()
**************************************************************************/
char *G__fgets(char *string,int n,FILE *fp)
{
  static int stat=1;
  if(fp==stdin) {
    if(stat) {
      stat=0;
      G__printf("Warning: fgets() can not use\n");
      return("");
    }
  }
  else {
    return(fgets(string,n,fp));
  }
}
/**************************************************************************
* G__gets()
**************************************************************************/
char *G__gets(char *buffer)
{
  static int stat=1;
  if(stat) {
    stat=0;
    G__printf("Warning: gets() can not use\n");
  }
  return("");
}

#endif /* G__SPECIALSTDIO */
