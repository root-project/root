/* /% C %/ */
/***********************************************************************
 * batch.exe 
 ************************************************************************
 * Source file batch.c
 ************************************************************************
 * Description:
 *  This tool emulates Windows batch command processor. It turns out 
 * each Windows-xx system has different interntal implementation. It
 * causes problem when running an automation script. This tool provides 
 * stable mean of running batch script.
 ************************************************************************
 * Copyright(c) 2003~2003  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h> 
#endif

/****************************************************************
*
****************************************************************/
#define G__MAXFILENAME  1024
#define G__LONGLINE     1024
#define G__MAXLINE      1024
#define G__MAXARG       256
#define G__MAXNAME      1024

/****************************************************************
* G__split(original,stringbuf,argc,argv)
* split arguments separated by space char.
* CAUTION: input string will be modified. If you want to keep
*         the original string, you should copy it to another string.
****************************************************************/
int G__split(char* line,char* string,int *argc,char* argv[])
{
  int lenstring;
  int i=0;
  int flag=0;
  int n_eof=1;
  int single_quote=0,double_quote=0,back_slash=0;
  
  while((string[i]!='\n')&&
	(string[i]!='\0')
#ifdef G__OLDIMPLEMENTATION1616
	&&(string[i]!=EOF)
#endif
	) i++;
  string[i]='\0';
  line[i]='\0';
  lenstring=i;
#ifdef G__OLDIMPLEMENTATION1616
  if(string[i]==EOF) n_eof=0;
#endif
  argv[0]=line;

  *argc=0;
  for(i=0;i<lenstring;i++) {
    switch(string[i]) {
    case '\\':
      if(back_slash==0) back_slash=1;
      else              back_slash=0;
      break;
    case '\'':
      if((double_quote==0)&&(back_slash==0)) {
	single_quote ^= 1;
	string[i]='\0';
	flag=0;
      }
      break;
    case '"' :
      if((single_quote==0)&&(back_slash==0)) {
	double_quote ^= 1;
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

/****************************************************************
* G__readline(fp,line,argbuf,argn,arg)
****************************************************************/
int G__readline(FILE* fp,char* line,char* argbuf,int* argn,char* arg[])
{
  /* int i; */
  char *null_fgets;
  null_fgets=fgets(line,G__LONGLINE*2,fp);
  if(null_fgets!=NULL) {
    strcpy(argbuf,line);
    G__split(line,argbuf,argn,arg);
  }
  else {
    line[0]='\0';;
    argbuf='\0';
    *argn=0;
    arg[0]=line;
  }
  if(null_fgets==NULL) return(0);
  else                 return(1);
}


/****************************************************************
* G__batch()
****************************************************************/
int G__batch(char* fname,int isdebug,int istrace) {
  FILE *fp;
  char line[G__MAXLINE];
  char argbuf[G__MAXLINE];
  char *arg[G__MAXARG];
  int argn;
  int result =0;

  fp = fopen(fname,"r");

  while(G__readline(fp,line,argbuf,&argn,arg)) {
    if(argn>=1 && (strcmp(arg[1],"REM")==0 || strcmp(arg[1],"rem")==0)) {
      if(istrace) printf("comment: %s\n",line);
    }
    else {
      if(strcmp(arg[1],"cd")==0 || strcmp(arg[1],"CD")==0) {
	if(istrace) printf("chngdir: %s\n",line);
	if(!isdebug) {
#if defined(_WIN32)
	  if(FALSE==SetCurrentDirectory(arg[2]))
	    fprintf(stderr,"can not change directory to %s\n",arg[2]);
#else
	  if(0!=chdir(arg[2])) 
	    fprintf(stderr,"can not change directory to %s\n",arg[2]);
#endif
	}
      }
      if(strcmp(arg[1],"echo")==0 || strcmp(arg[1],"ECHO")==0) {
	if(istrace) printf("echo   : %s\n",line);
	if(!isdebug) {
	  char *p = strstr(line,"echo");
	  if(!p) p=strstr(line,"ECHO");
	  if(p) {
	    p+=5;
	    printf("%s\n",p);
	  }
	}
      }
      else {
	if(istrace) printf("execute: %s\n",line);
	if(!isdebug) result |= system(line);
      }
    }
  }
  fclose(fp);

  return(result);
}

/****************************************************************
* main()
****************************************************************/
int main(int argc,char** argv) {
  int i=1;
  int istrace=0;
  int isdebug=0;
  int result=0;
  int isall=1;
  
  while(i<argc && '-'==argv[i][0]) {
    if(strcmp(argv[i],"-t")==0) istrace=1;
    else if(strcmp(argv[i],"-p")==0) { isdebug=1; istrace=1; }
    else if(strcmp(argv[i],"-a")==0) isall=1;
    else if(strcmp(argv[i],"-e")==0) isall=0;
    else {
      printf("batch.exe  : batch executor\n");
      printf("Usage:  batch.exe [-aept] [filename]\n");
      printf("Option: -a run all commands in one subproces\n");
      printf("        -e run each command in separate subprocess\n");
      printf("        -p preview without executing commands\n");
      printf("        -t trace mode\n");
    }
    ++i;
  }
  while(i<argc) {
    if(isall) {
      if(istrace) printf("execute: %s\n",argv[i]);
      if(!isdebug) result |= system(argv[i]);
    }
    else {
      result |= G__batch(argv[i],isdebug,istrace);
    }
    ++i;
  }
  return(result);
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
