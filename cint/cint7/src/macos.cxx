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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/* #include "common.h" */


#include <stdio.h>

int G__printf(char* fmt,...);
int G__fprinterr(char* fmt,...);

extern "C" {

extern FILE *G__serr;

/***********************************************************************
* getopt();
*
***********************************************************************/
int optind=1;
char *optarg;

int getopt(int argc,char **argv,char *optlist)
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
void alarm(int /* iwait */)
{ 
  ;
}

} /* extern "C" */

