/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file macos.c
 ************************************************************************
 * Description:
 *  MacOS missing library. Needed only for Macintosh environment
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

/* #include "common.h" */


#include <stdio.h>

#ifndef G__OLDIMPLEMENTATION463
extern FILE *G__serr;
#endif

/***********************************************************************
* getopt();
*
***********************************************************************/
int optind=1;
char *optarg;

int getopt(argc, argv,optlist)
int argc;
char **argv;
char *optlist;
{
  int optkey;
  char *p;
  if(optind<argc) {
    if('-'==argv[optind][0]) {
      optkey = argv[optind][1] ;
      p = optlist;
      while(*p) {
	if( (*p) == optkey ) {
	  ++p;
	  if(':'==(*p)) { /* option with argument */
	    if(argv[optind][2]) { /* -aARGUMENT */
	      optarg=argv[optind]+2;
	      optind+=1;
	      return(argv[optind-1][1]);
	    }
	    else { /* -a ARGUMENT */
	      optarg=argv[optind+1];
	      optind+=2;
	      return(argv[optind-2][1]);
	    }
	  }
	  else { /* option without argument */
	    ++optind;
	    optarg=(char*)NULL;
	    return(argv[optind-1][1]);
	  }
	}
	++p;
      }
      G__fprinterr("Error: Unknown option %s\n",argv[optind]);
      ++optind;
      return(0);
    }
    else {
      return(EOF);
    }
  }
  else {
    return (EOF);
  }
}

/***********************************************************************
* alarm()
*
***********************************************************************/
void alarm(iwait)
int iwait;
{ 
  ;
}


