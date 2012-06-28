/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************************
* ifdef.cxx
*
*  #ifdef,#ifndef,#if,#elif,#else,#endif resolver
*
*  Author   : Masaharu Goto
#  Date     : 8 Feb 1994
*  Date     : 8 Feb 2001
*
********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <string>
#include <map>
#ifndef __hpux
using namespace std;
#endif

#define MAXARG  2500
#define MAXLINE 5000

#ifdef G__GET
extern "C" {
extern int resolved;
extern int unresolved;
}
#endif


/************************************************
*  Global variable allocation and function"
* definition
************************************************/
#define MAXFILENAME 300
#define MAXSYMNAME  256

/****************************************************************************
* Table of explicitly defined macro
****************************************************************************/
map<string,int> defined;
map<string,int> wildcard;


/****************************************************************************
* other globals
****************************************************************************/

#define SHOW_UNRESOLVED_COND    0x0001
#define SHOW_RESOLVED_COND      0x0002
#define SHOW_DELETED_LINE       0x0004
#define SHOW_STDOUT             0x0008
#define MASK_NORMAL_OUTPUT      0x0100

int vmode= (SHOW_RESOLVED_COND) ;
FILE *out;
FILE *nout;

extern "C" {
char *G__calc(char*);
char *G__getexpr(char*);
char *G__getandor(char*);
#if !defined(__hpux) && !defined(__APPLE__) 
char getopt(int argc,char **argv,char *optlist);
#endif
extern int optind;
extern char *optarg;
}

void G__awk(FILE* G__fp);
int G__readlineawk(FILE* fp,char* line,char* argbuf,int* argn,char *arg[]);
int G__splitawk(char* string,int* argc,char* argv[MAXARG]);


/****************************************************************************
* G__defined()
*
* Returns 1 : if the macro is explicitly defined
*        -1 : if the macro is explicitly undefined
*         0 : if the macro is not explicitly defined or undefined
****************************************************************************/
extern "C" int G__defined(char* macro)
{
  int flag=0;
  int wildlen=0;
  int len = 0;

  flag = defined[macro];
  if(flag) return(flag);

  map<string,int>::iterator first = wildcard.begin();
  map<string,int>::iterator last  = wildcard.end();
  const char *temp;
  while(first!=last) {
    temp = (*first).first.c_str();
    len = strlen(temp);
    if(len>wildlen && strncmp(macro,temp,len)==0) {
      flag = (*first).second;
    }
    ++first;
  }
  return(flag);
}

/*************************************************************
* G__define
*************************************************************/
void G__define(char *arg) {
  char temp[MAXSYMNAME];

  if(vmode&SHOW_RESOLVED_COND) fprintf(out,"#define %s\n",arg);

  char *pe=strchr(arg,'=');
  if(pe) {
#ifdef G__GET
    G__getexpr(arg);
#endif
    strcpy(temp,arg);
    pe=strchr(temp,'=');
    *pe='\0';
    /* printf("#define %s %s\n",temp,pe+1); */
  }
  else {
    sprintf(temp,"%s=1",arg);
#ifdef G__GET
    G__getexpr(temp);
#endif
    strcpy(temp,arg);
    /* printf("#define %s\n",temp); */
  }
  pe = strchr(temp,'*');
  if(pe) {
    *pe = 0;
    wildcard[temp] = 1;
  }
  else {
    defined[temp] = 1;
  }
}

/*************************************************************
* G__undefine
*************************************************************/
void G__undefine(char *arg) {
  char temp[MAXSYMNAME];

  if(vmode&SHOW_RESOLVED_COND) fprintf(out,"#undef  %s\n",arg);

  sprintf(temp,"%s=0",arg);
#ifdef G__GET
  G__getexpr(temp);
#endif
  char *pe = strchr(temp,'*');
  if(pe) {
    *pe = 0;
    wildcard[temp] = -1;
  }
  else {
    defined[arg] = -1;
  }
}

/*************************************************************
* G__deffile
*************************************************************/
void G__deffile(char *optarg) {
  char G__oneline[MAXLINE];
  char G__argbuf[MAXLINE];
  char *arg[MAXARG];
  int argn;
  FILE* fp = fopen(optarg,"r");
  int line=0;

  if(!fp) {
    fprintf(out,"Warning: '%s' not found\n",optarg);
    return;
  }

  if(vmode&SHOW_RESOLVED_COND) 
    fprintf(out,"--- Read condition from file '%s' ---\n",optarg);
  
  /*************************************************************
   * User defined BEGIN procedure and local variable
   *************************************************************/
  /************************************************
   *  Local variable allocation and processing"
   * before reading file
   ************************************************/
  int narg;
  
  /********************************************************
   * Read lines until end of file
   ********************************************************/
  while(G__readlineawk(fp,G__oneline,G__argbuf,&argn,arg)!=0) {
    /*************************************************************
     * User defined LINE procedure
     *************************************************************/
    /************************************************
     *  If input line is "abcdefg hijklmn opqrstu"
     *
     *           arg[0]
     *             |
     *     +-------+-------+
     *     |       |       |
     *  abcdefg hijklmn opqrstu
     *     |       |       |
     *   arg[1]  arg[2]  arg[3]    argn=3
     *
     ************************************************/
    line++;
    
    /************************************************
     * #define
     ************************************************/
    if((strcmp(arg[1],"#define")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"define")==0)) {
      if(strcmp(arg[1],"#")==0) narg=3;
      else                      narg=2;
      G__define(arg[narg]);
    }

    /************************************************
     * #undef
     ************************************************/
    else if((strcmp(arg[1],"#undef")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"undef")==0)) {
      if(strcmp(arg[1],"#")==0) narg=3;
      else                      narg=2;
      G__undefine(arg[narg]);
    }

    /************************************************
     * #include
     ************************************************/
    else if((strcmp(arg[1],"#include")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"include")==0)) {
      if(strcmp(arg[1],"#")==0) narg=3;
      else                      narg=2;
      int i=0;
      if('<'==arg[narg][0]) {
	while(arg[narg][i+1]) { arg[narg][i]=arg[narg][i+1]; ++i; }
	arg[narg][i-1] = 0;
      }
      G__deffile(arg[narg]);
    }

#ifdef NEVER
    /************************************************
     * #if, ifndef, ifdef, else, elif, else, endif
     ************************************************/
    else if((strcmp(arg[1],"#if")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"if")==0)) {
      if(vmode & SHOW_RESOLVED_COND) 
	fprintf(out,"%s\n",arg[0]);
    }
    else if((strcmp(arg[1],"#ifdef")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"ifdef")==0)) {
      if(vmode & SHOW_RESOLVED_COND) 
	fprintf(out,"%s\n",arg[0]);
    }
    else if((strcmp(arg[1],"#ifndef")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"ifndef")==0)) {
      if(vmode & SHOW_RESOLVED_COND) 
	fprintf(out,"%s\n",arg[0]);
    }
    else if((strcmp(arg[1],"#elif")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"elif")==0)) {
      if(vmode & SHOW_RESOLVED_COND) 
	fprintf(out,"%s\n",arg[0]);
    }
    else if((strcmp(arg[1],"#else")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"else")==0)) {
      if(vmode & SHOW_RESOLVED_COND) 
	fprintf(out,"%s\n",arg[0]);
    }
    else if((strcmp(arg[1],"#endif")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"endif")==0)) {
      if(vmode & SHOW_RESOLVED_COND) 
	fprintf(out,"%s\n",arg[0]);
    }
#endif
    
    
    /************************************************************/
  }
  /*************************************************************
   * User defined END procedure
   *************************************************************/
  /************************************************
   * Processing after reading file
   ************************************************/
  /*******************************************************
   * close file
   ********************************************************/

  if(fp!=stdin) fclose(fp);

  if(vmode&SHOW_RESOLVED_COND) 
    fprintf(out,"--- End reading file '%s' ---\n",optarg);
}


char G__tmp[2];
/*************************************************************
* main function
*************************************************************/
int main(int argc,char *argv[])
{
  FILE *fp;
#ifndef G__OLDIMPLEMENTATION1616
  int G__c;
#else
  char G__c;
#endif
  char G__optdef[200];
  
  G__tmp[0]='\0';
  out = stderr;
  nout = stdout;
  
  sprintf(G__optdef, "D:U:gqGrudvVsmf:o:?");
  
  while ((G__c = getopt(argc,argv,G__optdef))!=EOF){
    switch(G__c) {
      /************************************************
       *  Program argument/option definition
       ************************************************/
    case 'f':
      G__deffile(optarg);
      break;
    case 'D':
      G__define(optarg);
      break;
    case 'U':
      G__undefine(optarg);
      break;
    case 's':
      vmode |= SHOW_STDOUT;
      out = stdout;
      break;
    case 'm':
      vmode |= MASK_NORMAL_OUTPUT;
      break;
    case 'o':
      nout = fopen(optarg,"w");
      if(!nout) {
	fprintf(out,"Error: can not open output file '%s'\n",optarg);
	exit(1);
      }
      break;
    case 'r':
      vmode |= SHOW_RESOLVED_COND;
      break;
    case 'u':
      vmode |= SHOW_UNRESOLVED_COND;
      break;
    case 'd':
      vmode |= SHOW_DELETED_LINE;
      break;
    case 'q':
      vmode= 0;
      break;
    case 'g':
      vmode = (SHOW_STDOUT | SHOW_UNRESOLVED_COND | SHOW_RESOLVED_COND |
	       MASK_NORMAL_OUTPUT);
      out = stdout;
      break;
    case 'G':
      vmode = (SHOW_STDOUT | SHOW_RESOLVED_COND |
	       MASK_NORMAL_OUTPUT);
      out = stdout;
      break;
    case 'v':
      vmode= (SHOW_UNRESOLVED_COND | SHOW_RESOLVED_COND);
      break;
    case 'V':
      vmode= (SHOW_UNRESOLVED_COND | SHOW_RESOLVED_COND | SHOW_DELETED_LINE);
      break;
    case '?':
      fprintf(stderr,"%s: #ifdef/#else/#endif symbolic resolver\n",argv[0]);
      fprintf(stderr,"Author: Masaharu Goto - 8 Feb 1994\n");
      fprintf(stderr,"Usage: %s <-dgmqrsuvV> <-f[deffile]> <-D[defined]<=[value]>> <-U[undefined]> <-o[outfile]> [source.c]\n",argv[0]);
      fprintf(stderr,"Option:\n");
      fprintf(stderr,"   -D [macro]<=[value]>:resolve '#ifdef [macro]' as true\n");
      fprintf(stderr,"   -U [macro]          :resolve '#ifdef [macro]' as false\n");
#if 0
      fprintf(stderr,"    (unspecified macro):#if,#ifdef remains\n");
#endif
      fprintf(stderr,"   -f [deffile]        :define/undef macros by file\n");
      fprintf(stderr,"                           #define [macro]<=[value]>\n");
      fprintf(stderr,"                           #undef  [macro]\n");
      fprintf(stderr,"   -o [outfile]        :specify output file (default - stdout)\n");
      fprintf(stderr,"   -m                  :no output\n");
      fprintf(stderr," debug  . . . . . . . . . . . . . . . . . . . . . . . . .\n");
      fprintf(stderr,"   -q                  :quiet mode\n");
      fprintf(stderr,"   -g                  :preview all #ifdef/#else/#endif (no output)\n");
      fprintf(stderr,"   -G                  :preview resolved #ifdef/#else/#endif (no output)\n");
#if 0
      fprintf(stderr,"   -v                  :verbose mode1. display deleted statements\n");
      fprintf(stderr,"   -V                  :verbose mode2. display all #if,#ifdef command\n");
#endif
      fprintf(stderr,"   -d                  :display deleted lines\n");
      fprintf(stderr,"   -r                  :display resolved conditions\n");
      fprintf(stderr,"   -u                  :display unresolved conditions\n");
#if 0
      fprintf(stderr,"   -s                  :display debug information to stdout (default - stderr)\n");
#endif
      fprintf(stderr,"Example:\n");
      fprintf(stderr,"  %s -DONLINE -DMAX=1024 -UDEBUG -onewsource.c source.c\n",argv[0]);
      fprintf(stderr,"  %s -g source.c\n",argv[0]);
      fprintf(stderr,"  %s -G -d -DDEBUG1 source.c\n",argv[0]);
      exit(EXIT_SUCCESS);
      break;
      default :
	fprintf(stderr,"Illegal option -%c\n",G__c);
      break;
    }
  }
  
  if(argv[optind]) {
    fp=fopen(argv[optind],"r");
    if(fp==NULL) {
      fprintf(stderr,"Can not open file %s\n",argv[1]);
      exit(1);
    }
  }
  else {
    fp=stdin;
  }

  G__awk(fp);

  if(fp!=stdin) fclose(fp);

  if(nout && nout!=stdout && nout!=stderr) fclose(nout);

  return(0);
}


/*************************************************************
 * disp()
 *************************************************************/
char *disp(int line,int nest,char* arg0)
{
  static char buf[MAXLINE];
#ifndef OLD
  int i;
  sprintf(buf,"%5d: ",line);
  for(i=1;i<nest;i++) strcat(buf,"| ");
  strcat(buf,arg0);
#else
  sprintf(buf,"(%4d) %d %s",line,nest,arg0);
#endif
  return(buf);
}

/*************************************************************
 * inden()
 *************************************************************/
char *inden(int nest)
{
  static char buf[MAXLINE];
  int i;
  sprintf(buf,"       ");
  for(i=1;i<nest;i++) strcat(buf,"  ");
  return(buf);
}

/*************************************************************
 * G__awk(fp)
 *************************************************************/
void G__awk(FILE* fp)
{
  char G__oneline[MAXLINE];
  char G__argbuf[MAXLINE];
  char *arg[MAXARG];
  int argn;
  int line=0;
  
  /*************************************************************
   * User defined BEGIN procedure and local variable
   *************************************************************/
  /************************************************
   *  Local variable allocation and processing"
   * before reading file
   ************************************************/
  int nest=0;
  int skipnest=0;
  int i;
  int mask=0;
  int narg;
  char *p,*cond;
  char result[200];
  int ifdefs;
  int skip[30],elseskip[30],unres[30];
  
  for(i=0;i<30;i++) {
    skip[i]=0;
    elseskip[i]=0;
    unres[i]=0;
  }
  /********************************************************
   * Read lines until end of file
   ********************************************************/
  while(G__readlineawk(fp,G__oneline,G__argbuf,&argn,arg)!=0) {
    /*************************************************************
     * User defined LINE procedure
     *************************************************************/
    /************************************************
     *  If input line is "abcdefg hijklmn opqrstu"
     *
     *           arg[0]
     *             |
     *     +-------+-------+
     *     |       |       |
     *  abcdefg hijklmn opqrstu
     *     |       |       |
     *   arg[1]  arg[2]  arg[3]    argn=3
     *
     ************************************************/
    line++;
    
    mask=0;
    ifdefs=0;
    
    /************************************************
     * #ifdef
     ************************************************/
    if((strcmp(arg[1],"#ifdef")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"ifdef")==0)) {
      ifdefs=1;
      if(strcmp(arg[1],"#")==0) narg=3;
      else                      narg=2;
      if(skip[nest]==0) {
	nest++;
	switch(G__defined(arg[narg])) {
	case -1:
	  skip[nest]=1;
	  elseskip[nest]=0;
	  unres[nest]=0;
	  mask=1;
	  if(vmode&SHOW_RESOLVED_COND) 
	    fprintf(out,"%s  ALWAYS FALSE\n",disp(line,nest,arg[0]));
	  break;
	case 1:
	  skip[nest]=0;
	  elseskip[nest]=1;
	  unres[nest]=0;
	  mask=1;
	  if(vmode&SHOW_RESOLVED_COND) 
	    fprintf(out,"%s  ALWAYS TRUE\n",disp(line,nest,arg[0]));
	  break;
	case 0:
	  skip[nest]=0;
	  elseskip[nest]=0;
	  unres[nest]=1;
	  if(vmode&SHOW_UNRESOLVED_COND) 
	    fprintf(out,"%s  unresolved\n",disp(line,nest,arg[0]));
	  break;
	}
	skipnest=0;
      }
      else {
	skipnest++;
      }
    }
    
    /************************************************
     * #ifndef
     ************************************************/
    if((strcmp(arg[1],"#ifndef")==0) ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"ifndef")==0)) {
      ifdefs=1;
      if(strcmp(arg[1],"#")==0) narg=3;
      else                      narg=2;
      if(skip[nest]==0) {
	nest++;
	switch(G__defined(arg[narg])) {
	case 1:
	  skip[nest]=1;
	  elseskip[nest]=0;
	  unres[nest]=0;
	  mask=1;
	  if(vmode&SHOW_RESOLVED_COND) 
	    fprintf(out,"%s  ALWAYS FALSE\n",disp(line,nest,arg[0]));
	  break;
	case -1:
	  skip[nest]=0;
	  elseskip[nest]=1;
	  unres[nest]=0;
	  mask=1;
	  if(vmode&SHOW_RESOLVED_COND) 
	    fprintf(out,"%s  ALWAYS TRUE\n",disp(line,nest,arg[0]));
	  break;
	case 0:
	  skip[nest]=0;
	  elseskip[nest]=0;
	  unres[nest]=1;
	  if(vmode&SHOW_UNRESOLVED_COND) 
	    fprintf(out,"%s  unresolved\n",disp(line,nest,arg[0]));
	  break;
	}
	skipnest=0;
      }
      else {
	skipnest++;
      }
    }
    
    
    /************************************************
     * #if
     ************************************************/
#ifdef G__GET
    if(strcmp(arg[1],"#if")==0 ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"if")==0)) {
      ifdefs=1;
      if(skip[nest]==0) {
	nest++;
	p=strstr(arg[0],"if");
	cond=p+3;
	resolved=0;
	unresolved=0;
	strcpy(result,G__calc(cond));
	
	if(unresolved==0||
	   (isdigit(result[0])&&result[1]=='\0')) {
	  if(atoi(result)==0) {
	    skip[nest]=1;
	    elseskip[nest]=0;
	    unres[nest]=0;
	    mask=1;
	    if(vmode&SHOW_RESOLVED_COND) 
	      fprintf(out,"%s -> %s ALWAYS FALSE\n"
		      ,disp(line,nest,arg[0]),result);
	  }
	  else {
	    skip[nest]=0;
	    elseskip[nest]=1;
	    unres[nest]=0;
	    mask=1;
	    if(vmode&SHOW_RESOLVED_COND) 
	      fprintf(out,"%s -> %s ALWAYS TRUE\n"
		      ,disp(line,nest,arg[0]),result);
	  }
	}
	else {
	  skip[nest]=0;
	  elseskip[nest]=0;
	  unres[nest]=1;
	  mask=1;
	  if(resolved==0) {
	    if(vmode&SHOW_UNRESOLVED_COND) 
	      fprintf(out,"%s %d,%d unresolved\n"
		      ,disp(line,nest,arg[0]),resolved,unresolved);
	    if(0==(vmode&MASK_NORMAL_OUTPUT))
	      fprintf(nout,"%s\n",arg[0]);
	  }
	  else {
	    if(vmode&SHOW_RESOLVED_COND) 
	      fprintf(out,"%s\n%s#if %s\n                    resolved=%d,unresolved=%d PARTLY RESOLVED\n"
		      ,disp(line,nest,arg[0]),inden(nest),result,resolved,unresolved);
	    *cond='\0';
	    if(0==(vmode&MASK_NORMAL_OUTPUT))
	      fprintf(nout,"%s %s\n",arg[0],result);
	  }
	}
      }
      else {
	skipnest++;
      }
    }
    
    /************************************************
     * #elif
     ************************************************/
    if(strcmp(arg[1],"#elif")==0 ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"elif")==0)) {
      ifdefs=1;
      if(skipnest==0) {
	if(elseskip[nest]) {
	  skip[nest]=1;
	  elseskip[nest]=1;
	  mask=1;
	  if(vmode&SHOW_RESOLVED_COND) 
	    fprintf(out,"%s  ALWAYS SKIP\n",disp(line,nest,arg[0]));
	}
	else {
	  p=strstr(arg[0],"elif");
	  cond=p+5;
	  resolved=0;
	  unresolved=0;
	  strcpy(result,G__calc(cond));
	  
	  if(unresolved==0||
	     (isdigit(result[0])&&result[1]=='\0')) {
	    if(atoi(result)==0) {
	      skip[nest]=1;
	      elseskip[nest]=0;
	      mask=1;
	      if(vmode&SHOW_RESOLVED_COND) 
		fprintf(out,"%s -> %s ALWAYS FALSE\n"
			,disp(line,nest,arg[0]),result);
	    }
	    else {
	      skip[nest]=0;
	      elseskip[nest]=1;
	      mask=1;
	      if(vmode&SHOW_RESOLVED_COND) 
		fprintf(out,"%s -> %s ALWAYS TRUE\n"
			,disp(line,nest,arg[0]),result);
	      if(unres[nest]) {
		sprintf(p,"else");
		if(0==(vmode&MASK_NORMAL_OUTPUT))
		  fprintf(nout,"%s\n",arg[0]);
	      }
	    }
	  }
	  else {
	    if(unres[nest]==0) {
	      p[0]='i';
	      p[1]='f';
	      p[2]=' ';
	      p[3]=' ';
	    }
	    if(resolved==0) {
	      if(vmode&SHOW_UNRESOLVED_COND) 
		fprintf(out,"%s %d,%d unresolved\n"
			,disp(line,nest,arg[0]),resolved,unresolved);
	      if(0==(vmode&MASK_NORMAL_OUTPUT))
		fprintf(nout,"%s\n",arg[0]);
	    }
	    else {
	      if(vmode&SHOW_RESOLVED_COND) 
		fprintf(out,"%s\n%s#elif %s\n                    resolved=%d,unresolved=%d PARTLY RESOLVED\n"
			,disp(line,nest,arg[0]),inden(nest),result,resolved,unresolved);
	      *cond='\0';
	      if(0==(vmode&MASK_NORMAL_OUTPUT))
		fprintf(nout,"%s %s\n",arg[0],result);
	    }
	    mask=1;
	    skip[nest]=0;
	    elseskip[nest]=0;
	    unres[nest]++;
	  }
	}
      }
    }
    
#else
    if(strcmp(arg[1],"#if")==0 ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"if")==0)) {
      ifdefs=1;
      if(skip[nest]==0) {
	nest++;
      }
      else {
	skipnest++;
      }
    }
    /************************************************
     * #elif
     ************************************************/
    if(strcmp(arg[1],"#elif")==0 ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"elif")==0)) {
    }
#endif
    
    
    /************************************************
     * #else
     ************************************************/
    if(strcmp(arg[1],"#else")==0 ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"else")==0)) {
      ifdefs=1;
      if(skipnest==0) {
	if(elseskip[nest]) {
	  skip[nest]=1;
	  elseskip[nest]=1;
	  mask=1;
	  if(vmode&SHOW_RESOLVED_COND) 
	    fprintf(out,"%s   ALWAYS SKIP\n",disp(line,nest,arg[0]));
	}
	else if(unres[nest]==0) {
	  skip[nest]=0;
	  elseskip[nest]=1;
	  mask=1;
	  if(vmode&SHOW_RESOLVED_COND) 
	    fprintf(out,"%s   ALWAYS TRUE\n",disp(line,nest,arg[0]));
	}
	else {
	  if(vmode&SHOW_UNRESOLVED_COND) 
	    fprintf(out,"%s   unresolved\n",disp(line,nest,arg[0]));
	  skip[nest]=0;
	  elseskip[nest]=1;
	}
      }
    }
    
    
    /************************************************
     * #endif
     ************************************************/
    if(strcmp(arg[1],"#endif")==0 ||
       (strcmp(arg[1],"#")==0 && strcmp(arg[2],"endif")==0)) {
      ifdefs=1;
      if(skipnest==0) {
	if(unres[nest]==0) {
	  mask=1;
	  if(vmode&SHOW_RESOLVED_COND) 
	    fprintf(out,"%s END ALWAYS\n",disp(line,nest,arg[0]));
	}
	else {
	  if(vmode&SHOW_UNRESOLVED_COND) 
	    fprintf(out,"%s  unresolved\n",disp(line,nest,arg[0]));
	}
	nest--;
      }
      else {
	skipnest--;
      }
    }
    
    if(skip[nest]==0&&mask==0) {
      if(0==(vmode&MASK_NORMAL_OUTPUT))
	fprintf(nout,"%s\n",arg[0]);
    }
    else if((vmode&SHOW_DELETED_LINE) && ifdefs==0) {
      fprintf(out,"D%4d: %s\n",line,arg[0]);
    }
    
    
    /************************************************************/
  }
  /*************************************************************
   * User defined END procedure
   *************************************************************/
  /************************************************
   * Processing after reading file
   ************************************************/
}


/****************************************************************
 * G__readlineawk(fp,line,argbuf,argn,arg)
 ****************************************************************/
int G__readlineawk(FILE* fp,char* line,char* argbuf,int* argn,char *arg[])
{
  char *null_fgets;
  null_fgets=fgets(line,MAXLINE,fp);
  if(null_fgets!=NULL) {
    char *p;
    p = strchr(line,'\n');
    if(p) *p=0;
    p = strchr(line,'\r');
    if(p) *p=0;
    strcpy(argbuf,line);
    G__splitawk(argbuf,argn,arg);
    /*line[strlen(line)-1]='\0';*/
  }
  else {
    strcpy(line,"");
    strcpy(argbuf,"");
    *argn=0;
  }
  arg[0] = &line[0];
  if(*argn==0 || arg[1]==NULL) arg[1]=G__tmp;
  if(null_fgets==NULL) return(0);
  else                 return(1);
}

/****************************************************************
 * G__splitawk(string,argc,argv)
 * split arguments separated by space string.
 * CAUTION: input string will be modified. If you want to keep
 *         the original string, you should copy it to another string.
 ****************************************************************/
int G__splitawk(char* string,int* argc,char* argv[MAXARG])
{
  int lenstring;
  int i=0;
  int flag=0;
  int n_eof=1;
  int single_quote=0,double_quote=0,back_slash=0;
  
  while((string[i]!='\n')&&
#ifdef G__OLDIMPLEMENTATION1616
	(string[i]!=EOF)&&
#endif
	(i<MAXLINE-1)) i++;
  string[i]='\0';
  lenstring=i;
#ifdef G__OLDIMPLEMENTATION1616
  if(string[i]==EOF) n_eof=0;
#endif
  
  *argc=0;
  for(i=0;i<lenstring;i++) {
    switch(string[i]) {
    case '\\':
      if(back_slash==0) back_slash=1;
      else              back_slash=0;
      break;
    case '\'':
      if((double_quote==0)&&(back_slash==0)) {
	if(single_quote==0) single_quote=1;
	else                single_quote=0;
	string[i]='\0';
	flag=0;
      }
      break;
    case '"' :
      if((single_quote==0)&&(back_slash==0)) {
	if(double_quote==0) double_quote=1;
	else                double_quote=0;
	string[i]='\0';
	flag=0;
      }
      break;
      default  :
	if((isspace(string[i]))&&(back_slash==0)&&
	   (single_quote==0)&&(double_quote==0)) {
	  string[i]='\0';
	  flag=0;
	}
	else {
	  if(flag==0) {
	    (*argc)++;
	    argv[*argc] = &string[i];
	    flag=1;
	  }
	}
      back_slash=0;
      break;
    }
  }
  return(n_eof);
}

