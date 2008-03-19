/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file newsos.c
 ************************************************************************
 * Description:
 *  Missing ANSI-C function for NewsOS
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(G__NONANSI) || defined(G__NEWSOS6) || defined(G__NEWSOS4)

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <assert.h>
#include <time.h>
#include <ctype.h>
#include <locale.h>
#include <setjmp.h>
#include "newsos.h"

extern "C" {

extern FILE *G__serr;

/************************************************************************
* ANSI library implemented
************************************************************************/
int fsetpos(fp,position)
FILE *fp;
fpos_t *position;
{
        if(fp) 
          fseek(fp,*position,SEEK_SET);
        return(0);
}


int fgetpos(fp,position)
FILE *fp;
fpos_t *position;
{
        if(fp)
          *position = ftell(fp);
        return(0);
}

/************************************************************************
* 
************************************************************************/
G__newsos_nosupport(funcname)
char *funcname;
{
        G__fprinterr(G__serr,
                "Limitation: %s() not supported for NewsOS\n",funcname);
}        

/************************************************************************
* Unsupported dummy function
************************************************************************/


int raise(signal)
int signal;
{
        int result;
#ifdef G__NEVER
        G__newss_nosupport("raise");
#else
        switch(signal) {
        case SIGINT: G__fsigint(); break;
        case SIGILL: G__fsigill(); break;
        case SIGFPE: G__fsigfpe(); break;
        case SIGABRT: G__fsigabrt(); break;
        case SIGSEGV: G__fsigsegv(); break;
        case SIGTERM: G__fsigterm(); break;
        default: break;
        }
#endif
        return(result);
}

char *strerror(error)
int error;
{
        char *result;
        G__newsos_nosupport("strerror");
        return(result);
}

double difftime(newtime,oldtime)
#ifdef G__NEWSOS6
time_t newtime,oldtime;
#else
int newtime,oldtime;
#endif
{
        double result;
        G__newsos_nosupport("difftime");
        return(result);
}


unsigned long strtoul(sqrt,tailptr,base)
char *sqrt;
char **tailptr;
int base;
{
        unsigned long result;
        G__newsos_nosupport("strtoul");
        return(result);
}

#ifdef G__NEWSOS4
/*********************************************************************
* NewsOS 4.x specifix dummy functions
*********************************************************************/
unsigned long clock() {
        unsigned long result;
        G__newsos_nosupport("clock");
        return(result);
}

int setvbuf(fp,buffer,mode,size)
FILE *fp;
char *buffer;
int mode;
unsigned int size;
{
        int result;
        G__newsos_nosupport("setvbuf");
        return(result);
}

unsigned int strxfrm(string1,string2,n)
char *string1;
char *string2;
unsigned int n;
{
        unsigned int result;
        G__newsos_nosupport("strxfrm");
        return(result);
}
#endif

} /* extern "C" */

#else

#ifndef __GNUC__
/* Prevent "empty translation unit" warnings. */
static char G__file_intentionally_empty_newsos = 69;
#endif


#endif /* G__NONANSI || G__NEWSOS6 || G__NEWSOS4 */
