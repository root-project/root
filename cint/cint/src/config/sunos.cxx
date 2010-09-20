/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file sunos.c
 ************************************************************************
 * Description:
 *  Missing ANSI-C function for SunOS4.1.2.
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(G__NONANSI) || defined(G__SUNOS4)

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
#include "sunos.h"

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
G__sunos_nosupport(funcname)
char *funcname;
{
        G__fprinterr(
                "Limitation: %s() not supported for SunOS\n",funcname);
}        

/************************************************************************
* Unsupported dummy function
************************************************************************/
void *memmove(region1,region2,count)
void *region1,*region2;
size_t count;
{
        void *result;
        G__sunos_nosupport("memmove");
        return(result);
}


int raise(signal)
int signal;
{
        int result;
#ifdef G__NEVER
        G__sunos_nosupport("raise");
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
        G__sunos_nosupport("strerror");
        return(result);
}

double difftime(newtime,oldtime)
time_t newtime,oldtime;
{
        double result;
        G__sunos_nosupport("difftime");
        return(result);
}

int labs(n)
long n;
{
        int result;
        G__sunos_nosupport("labs");
        return(result);
}


unsigned long strtoul(sqrt,tailptr,base)
char *sqrt;
char **tailptr;
int base;
{
        unsigned long result;
        G__sunos_nosupport("strtoul");
        return(result);
}

} /* extern "C" */

#else

/* Prevent "empty translation unit" warnings. */
static char G__file_intentionally_empty_sunos = 69;

#endif /* G__NONANSI || G__SUNOS4 */
