/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file intrpt.c
 ************************************************************************
 * Description:
 *  Signal handling function
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "common.h"

extern int G__browsing; /* used in disp.c and intrpt.c */

#ifdef G__EH_SIGNAL
static int G__error_handle_flag=0;
#endif

#ifdef G__EH_SIGNAL
/************************************************************************
* New interrupt routine
************************************************************************/
void G__error_handle(signame)
int signame;
{
  G__error_handle_flag=1;
  switch(signame) {
#ifdef SIGFPE
  case SIGFPE:
    G__genericerror("Floating exception");
    signal(SIGFPE,G__error_handle);
    break;
#endif
#ifdef SIGSEGV
  case SIGSEGV:
    G__genericerror("Segmentation violation");
    signal(SIGSEGV,G__error_handle);
    break;
#endif
#ifdef SIGILL
  case SIGILL:
    G__genericerror("Illegal instruction");
    signal(SIGILL,G__error_handle);
    break;
#endif
#ifdef SIGEMT
  case SIGEMT:
    G__genericerror("Out of memory");
    signal(SIGEMT,G__error_handle);
    break;
#endif
#ifdef SIGBUS
  case SIGBUS:
    G__genericerror("Bus error");
    signal(SIGBUS,G__error_handle);
    break;
#endif	
  default:
    G__genericerror("Error: Unknown");
    break;
  }
}
#endif

/************************************************************************
* Interrupt routines
*
*  keyboard interrupt
*  floating exception
*  out of memory
*  kill process
*  segmentation violation
*  bus error
*  time out
************************************************************************/

/******************************************************************
* G__breakkey()
******************************************************************/
#ifdef _AIX
void G__breakkey(int signame)
#else
void G__breakkey(signame)
int signame;
#endif
{
  /*********************************************************
   * stop browsing 
   *********************************************************/
  G__browsing=0;

  /*********************************************************
   * G__step is set so that the program stops at next 
   * interpreted statement.
   *********************************************************/
  G__step++;
  G__setdebugcond();
  G__disp_mask=0;

  /*********************************************************
   * immediate pause in prerun
   *********************************************************/
  if(G__prerun) {
    G__fprinterr(G__serr,"\n!!! Pause at prerun. signal(%d)\n",signame);
    G__step--;
    G__setdebugcond();
    G__pause();
  }
  /*********************************************************
   * immediate pause if called twice
   *********************************************************/
  else if(G__step>1) {
    G__fprinterr(G__serr,"\n!!! Break in the middle of compiled statement. signal(%d)\n",signame);
    G__pause();
    if(G__return>G__RETURN_NORMAL) {
      G__fprinterr(G__serr, "!!! Sorry, continue until compiled code finishes\n");
      G__fprinterr(G__serr, "!!! Use qqq for immediate termination\n");
    }
  }
  else if(G__asm_exec) {
    G__fprinterr(G__serr, "\n!!! Middle of loop compilation run. signal(%d)\n",signame);
  }
  signal(SIGINT,G__breakkey);
}


/******************************************************************
* G__killproc()
******************************************************************/
void G__killproc(signame)
int signame;
{
  fprintf(G__sout,"\n!!! Process killed by interrupt. signal(%d)\n",signame);
  G__exit(EXIT_FAILURE);
}

/******************************************************************
* G__errorprompt()
******************************************************************/
int G__errorprompt(nameoferror)
char *nameoferror;
{

#ifdef G__EH_SIGNAL
  if(G__error_handle_flag) {
    G__error_handle_flag=0;
    return(0);
  }
#endif

  G__step=1;
  G__setdebugcond();

  G__genericerror(nameoferror);
#ifdef G__MEMTEST
  fprintf(G__memhist,"%s\n",nameoferror);
#endif

  G__no_exec=0;
  fflush(G__sout);
  fflush(G__serr);


#ifdef SIGALRM
  G__fprinterr(G__serr,
	  "Press return or process will be terminated in %dsec by timeout\n"
	  ,G__TIMEOUT);
  signal(SIGALRM,G__timeout);
  alarm(G__TIMEOUT);
#endif

  G__pause();

#ifdef SIGALRM
  alarm(0);
  G__fprinterr(G__serr,"Time out cancelled\n");
#endif

  while(G__return<G__RETURN_EXIT1) {
    if(G__pause()) break;
  }

  if(G__return>=G__RETURN_EXIT1) {
    G__close_inputfiles();
    exit(EXIT_FAILURE);
  }
  return(0);
}

/******************************************************************
* G__timeout()
******************************************************************/
void G__timeout(signame)
int signame;
{
  G__fprinterr(G__serr,"\nsignal(%d) Error time out. Exit program.\n",signame);

  G__close_inputfiles();
  exit(EXIT_FAILURE);
}

/******************************************************************
* G__floatexception()
******************************************************************/
#ifdef _AIX
void G__floatexception(int signame)
#else
void G__floatexception(signame)
int signame;
#endif
{
  G__fprinterr(G__serr,"signal(%d) ",signame); 
  signal(SIGFPE,G__floatexception);
  G__errorprompt("Error: Floating point exception");
}

/******************************************************************
* G__segmentviolation()
******************************************************************/
#ifdef _AIX
void G__segmentviolation(int signame)
#else
void G__segmentviolation(signame)
int signame;
#endif
{
  G__fprinterr(G__serr,"signal(%d) ",signame); 
  signal(SIGSEGV,G__segmentviolation);
  G__errorprompt("Error: Segmentation violation");
}

/******************************************************************
* G__outofmemory()
******************************************************************/
#ifdef _AIX
void G__outofmemory(int signame)
#else
void G__outofmemory(signame)
int signame;
#endif
{
  G__fprinterr(G__serr,"signal(%d) ",signame); 
#ifdef SIGEMT
  signal(SIGEMT,G__outofmemory);
#endif
  G__errorprompt("Error: Out of memory");
}

/******************************************************************
* G__buserror()
******************************************************************/
#ifdef _AIX
void G__buserror(int signame)
#else
void G__buserror(signame)
int signame;
#endif
{
  G__fprinterr(G__serr,"signal(%d) ",signame); 
#ifdef SIGBUS
  signal(SIGBUS,G__buserror);
#endif
  G__errorprompt("Error: Bus error");
}

#ifndef G__OLDIMPLEMENTATION1946
/******************************************************************
* G__errorexit()
******************************************************************/
#ifdef _AIX
void G__errorexit(int signame)
#else
void G__errorexit(signame)
int signame;
#endif
{
  G__fprinterr(G__serr,"Error: caught signal(%d)\n",signame); 
  signal(signame,(void (*)())SIG_DFL);
  exit(EXIT_FAILURE);
}
#endif


/************************************************************************
* End of Interrupt routines
************************************************************************/


#ifdef G__SIGNAL
/******************************************************************
* G__call_interrupt()
******************************************************************/
int G__call_interruptfunc(func)
char *func;
{
#ifdef G__ASM
  G__ALLOC_ASMENV;
#endif
  int store_var_type;

#ifdef G__ASM
  G__STORE_ASMENV;
#endif
  store_var_type = G__var_type;
  G__var_type='p';
  G__getexpr(func);
#ifdef G__ASM
  G__RECOVER_ASMENV;
#endif
  G__var_type=store_var_type;
  return(0);
}

/******************************************************************
* G__fsigabrt()
******************************************************************/
void G__fsigabrt()
{
  char temp[G__ONELINE];
  signal(SIGABRT,(void (*)())SIG_DFL);
  if(G__SIGABRT) {
#define G__OLDIMPLEMENTATION1945
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGABRT,SIGABRT);
#else
    sprintf(temp,"%s()",G__SIGABRT);
#endif
    G__SIGABRT = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigfpe()
******************************************************************/
void G__fsigfpe()
{
  char temp[G__ONELINE];
  signal(SIGFPE,G__floatexception);
  if(G__SIGFPE) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGFPE,SIGFPE);
#else
    sprintf(temp,"%s()",G__SIGFPE);
#endif
    G__SIGFPE = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigill()
******************************************************************/
void G__fsigill()
{
  char temp[G__ONELINE];
  signal(SIGILL,(void (*)())SIG_DFL);
  if(G__SIGILL) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGILL,SIGILL);
#else
    sprintf(temp,"%s()",G__SIGILL);
#endif
    G__SIGILL = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigint()
******************************************************************/
void G__fsigint()
{
  char temp[G__ONELINE];
  signal(SIGINT,G__breakkey);
  if(G__SIGINT) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGINT,SIGINT);
#else
    sprintf(temp,"%s()",G__SIGINT);
#endif
    G__SIGINT = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigsegv()
******************************************************************/
void G__fsigsegv()
{
  char temp[G__ONELINE];
  signal(SIGSEGV,G__segmentviolation);
  if(G__SIGSEGV) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGSEGV,SIGSEGV);
#else
    sprintf(temp,"%s()",G__SIGSEGV);
#endif
    G__SIGSEGV = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigterm()
******************************************************************/
void G__fsigterm()
{
  char temp[G__ONELINE];
  signal(SIGTERM,(void (*)())SIG_DFL);
  if(G__SIGTERM) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGTERM,SIGTERM);
#else
    sprintf(temp,"%s()",G__SIGTERM);
#endif
    G__SIGTERM = NULL;
    G__call_interruptfunc(temp);
  }
}

#ifdef SIGHUP
/******************************************************************
* G__fsighup()
******************************************************************/
void G__fsighup()
{
  char temp[G__ONELINE];
  signal(SIGHUP,(void (*)())SIG_DFL);
  if(G__SIGHUP) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGHUP,SIGHUP);
#else
    sprintf(temp,"%s()",G__SIGHUP);
#endif
    G__SIGHUP = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGQUIT
/******************************************************************
* G__fsigquit()
******************************************************************/
void G__fsigquit()
{
  char temp[G__ONELINE];
  signal(SIGQUIT,(void (*)())SIG_DFL);
  if(G__SIGQUIT) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGQUIT,SIGQUIT);
#else
    sprintf(temp,"%s()",G__SIGQUIT);
#endif
    G__SIGQUIT = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGTSTP
/******************************************************************
* G__fsigtstp()
******************************************************************/
void G__fsigtstp()
{
  char temp[G__ONELINE];
  signal(SIGTSTP,(void (*)())SIG_DFL);
  if(G__SIGTSTP) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGTSTP,SIGTSTP);
#else
    sprintf(temp,"%s()",G__SIGTSTP);
#endif
    G__SIGTSTP = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGTTIN
/******************************************************************
* G__fsigttin()
******************************************************************/
void G__fsigttin()
{
  char temp[G__ONELINE];
  signal(SIGTTIN,(void (*)())SIG_DFL);
  if(G__SIGTTIN) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGTTIN,SIGTTIN);
#else
    sprintf(temp,"%s()",G__SIGTTIN);
#endif
    G__SIGTTIN = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGTTOU
/******************************************************************
* G__fsigttou()
******************************************************************/
void G__fsigttou()
{
  char temp[G__ONELINE];
  signal(SIGTTOU,(void (*)())SIG_DFL);
  if(G__SIGTTOU) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGTTOU,SIGTTOU);
#else
    sprintf(temp,"%s()",G__SIGTTOU);
#endif
    G__SIGTTOU = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGALRM
/******************************************************************
* G__fsigalrm()
******************************************************************/
void G__fsigalrm()
{
  char temp[G__ONELINE];
  signal(SIGALRM,(void (*)())SIG_DFL);
  if(G__SIGALRM) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGALRM,SIGALRM);
#else
    sprintf(temp,"%s()",G__SIGALRM);
#endif
    G__SIGALRM = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGUSR1
/******************************************************************
* G__fsigusr1()
******************************************************************/
void G__fsigusr1()
{
  char temp[G__ONELINE];
  signal(SIGUSR1,(void (*)())SIG_DFL);
  if(G__SIGUSR1) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGUSR1,SIGUSR1);
#else
    sprintf(temp,"%s()",G__SIGUSR1);
#endif
    G__SIGUSR1 = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGUSR2
/******************************************************************
* G__fsigusr2()
******************************************************************/
void G__fsigusr2()
{
  char temp[G__ONELINE];
  signal(SIGUSR2,(void (*)())SIG_DFL);
  if(G__SIGUSR2) {
#ifndef G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s(%d)",G__SIGUSR2,SIGUSR2);
#else
    sprintf(temp,"%s()",G__SIGUSR2);
#endif
    G__SIGUSR2 = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#endif /* G__SIGNAL */

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
