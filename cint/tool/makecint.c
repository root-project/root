/* /% C %/ */
/***********************************************************************
 * makecint (C/C++ interpreter-compiler)
 ************************************************************************
 * Source file makecint.c
 ************************************************************************
 * Description:
 *  This tool creates Makefile for encapsurating arbitrary C/C++ object
 * into Cint as Dynamic Link Library or archived library
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto (cint@pcroot.cern.ch)
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

#ifdef G__ROOT
#ifdef HAVE_CONFIG
#include "config.h"
#endif
#endif

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif


#define G__MAXFILENAME  512
#define G__LONGLINE     512
#define G__MAXLINE      512
#define G__MAXARG       100
#define G__MAXNAME      100

char G__cintsysdir[G__MAXFILENAME] ; 

char G__CSRCPOST[20];
char G__CHDRPOST[20];
char G__CPPSRCPOST[20];
char G__CPPHDRPOST[20];
char G__DLLPOST[20];
char G__OBJPOST[20];

int G__isDLL=1;
char G__object[G__MAXFILENAME];
char G__makefile[G__MAXFILENAME];
char G__DLLID[G__MAXNAME];
char G__INITFUNC[G__LONGLINE];

#ifdef G__DJGPP
char G__DJGPPDIR[G__MAXFILENAME];
#endif

struct G__string_list {
  char *string;
  char *object;
  char *misc;
  struct G__string_list *next;
};

struct G__string_list *G__MACRO;
struct G__string_list *G__IPATH;
struct G__string_list *G__CHDR;
struct G__string_list *G__CPPHDR;
struct G__string_list *G__CSRC;
struct G__string_list *G__CPPSRC;
struct G__string_list *G__LIB;
struct G__string_list *G__CCOPT;
struct G__string_list *G__CIOPT;
struct G__string_list *G__CSTUB;
struct G__string_list *G__CPPSTUB;

char G__preprocess[10];
int G__ismain=0;

enum G__MODE { G__IDLE, G__CHEADER, G__CSOURCE, G__CPPHEADER, G__CPPSOURCE
	     , G__LIBRARY , G__CSTUBFILE , G__CPPSTUBFILE
	     , G__COMPILEROPT, G__CINTOPT
};

enum G__PRINTMODE { G__PRINTOBJECT, G__PRINTSTRING , G__PRINTOBJECT_WPLUS
		      , G__PRINTOBJECT_WSRC };


/**************************************************************************
* Path separator
**************************************************************************/
#if defined(G__NONANSI)
char *G__psep = "/";
#elif defined(G__WIN32)
const char *G__psep = "\\";
#elif defined(__MWERKS__)
const char *G__psep = ":";
#else
const char *G__psep = "/";
#endif

/****************************************************************
* G__printtitle()
****************************************************************/
void G__printtitle()
{
  printf("################################################################\n");
#if defined(G__DJGPP)
  printf("# makecint : interpreter-compiler for cint (MS-DOS DJGPP version)\n");
#elif (G__CYGWIN>=50)
  printf("# makecint : interpreter-compiler for cint (Windows Cygwin DLL version)\n");
#elif defined(G__CYGWIN)
  printf("# makecint : interpreter-compiler for cint (Windows Cygwin version)\n");
#elif defined(G__MINGW)
  printf("# makecint : interpreter-compiler for cint (Windows Mingw version)\n");
#else
  printf("# makecint : interpreter-compiler for cint (UNIX version)\n");
#endif
  printf("#\n");
  printf("# Copyright(c) 1995~2004 Masaharu Goto (cint@pcroot.cern.ch)\n");
  printf("#                        (cint mailing list 'cint@root.cern.ch')\n");
  printf("################################################################\n");
}

/****************************************************************
* G__displayhelp()
****************************************************************/
void G__displayhelp()
{
  printf("Usage :\n");
  printf(" makecint -mk [Makefile] -o [Object] -H [C++header] -C++ [C++source]\n");
  printf("          <-m> <-p>      -dl [DLL]   -h [Cheader]   -C   [Csource]\n");
  printf("                          -l [Lib] -i [StubC] -i++ [StubC++]\n");
  printf("  -o [obj]      :Object name\n");
  printf("  -dl [dynlib]  :Generate dynamic link library object\n");
  printf("  -mk [mkfile]  :Create makefile (no actual compilation)\n");
  printf("  -p            :Use preprocessor for header files\n");
  printf("  -m            :Needed if main() is included in the source file\n");
  printf("  -D [macro]    :Define macro\n");
  printf("  -I [incldpath]:Set Include file search path\n");
  printf("  -H [sut].h    :C++ header as parameter information file\n");
  printf("  -h [sut].h    :C header as parameter information file\n");
  printf("    +P          :Turn on preprocessor mode for following header files\n");
  printf("    -P          :Turn off preprocessor mode for following header files\n");
  printf("    +V          :Turn on class title loading for following header files\n");
  printf("    -V          :Turn off class title loading for following header files\n");
  printf("  -C++ [sut].C  :Link C++ object. Not accessed unless -H [sut].h is given\n");
  printf("  -C [sut].c    :Link C object. Not accessed unless -h [sut].h is given\n");
  printf("  -i++ [stub].h :C++ STUB function parameter information file\n");
  printf("  -i [stub].h   :C STUB function parameter information file\n");
  printf("  -c [sut].c    :Same as '-h [sut].c -C [sut].c'\n");
  printf("  -l -l[lib]    :Compiled object, Library or linker options\n");
#ifndef G__OLDIMPLEMENTATION1452
  printf("  -u [file]     :Generate dummy class for undefined typename\n");
  printf("  -U [dir]      :Directory to disable interface method generation\n");
  printf("  -Y [0|1]      :Ignore std namespace (default=1:ignore)\n");
  printf("  -Z [0|1]      :Automatic loading of standard header files\n");
#endif
  printf("  -cc   [opt]   :Compiler option\n");
  printf("  -cint [opt]   :Cint option\n");
  printf("  -B [funcname] :Initialization function name\n");
  printf("  -y [LIBNAME]  :Name of CINT core DLL, LIBCINT or WILDC(WinNT/95 only)\n");
}

/****************************************************************
* G__displaytodo()
****************************************************************/
void G__displaytodo()
{
  printf("%s is created. Makecint success.\n",G__makefile);
  printf("Do 'make -f %s' to compile the object\n",G__makefile);
}

/****************************************************************
* G__split(original,stringbuf,argc,argv)
* split arguments separated by space char.
* CAUTION: input string will be modified. If you want to keep
*         the original string, you should copy it to another string.
****************************************************************/
int G__split(line,string,argc,argv)
char *string,*line;
char *argv[];
int *argc;
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
int G__readline(fp,line,argbuf,argn,arg)
FILE *fp;
int *argn;
char *line,*argbuf;
char *arg[];
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

/**************************************************************************
* G__strrstr()
**************************************************************************/
char *G__strrstr(string1,string2)
char *string1,*string2;
{
  char *p=NULL,*s,*result=NULL;
  s=string1;
  while((p=strstr(s,string2))) { /* assignment intended here */
    result=p;
    s=p+1;
  }
  return(result);
}


/**************************************************************************
* G__copyfile()
**************************************************************************/
int G__copyfile(to,from)
FILE *to,*from;
{
  int c=0;
  while(EOF != (c=fgetc(from))) {
    fputc(c,to);
  } 
  return(0);
}


/******************************************************************
* G__getcintsysdir()
*
*  print out error message for unsupported capability.
******************************************************************/
int G__getcintsysdir()
{
  char *env;
  if('\0'==G__cintsysdir[0]) {
#ifdef G__ROOT
# ifdef ROOTBUILD
    env = "cint";
# else
#  ifdef CINTINCDIR
    env = CINTINCDIR;
#  else
    env=getenv("ROOTSYS");
#  endif
# endif
#else
    env=getenv("CINTSYSDIR");
# ifdef CINTSYSDIR
    if(!env || !env[0]) env = CINTSYSDIR;
# endif
#endif
    if(env) {
#ifdef G__ROOT
# ifdef ROOTBUILD
      sprintf(G__cintsysdir, "%s", env);
# else
#  ifdef CINTINCDIR
      sprintf(G__cintsysdir, "%s", CINTINCDIR);
#  else
      sprintf(G__cintsysdir, "%s%scint", env, G__psep);
#  endif
# endif
#else
      strcpy(G__cintsysdir,env);
#endif
      return(0);
    }
    else {
#ifdef G__ROOT
      fprintf(stderr,"Error: environment variable ROOTSYS is not set. makecint aborted.\n");   
#else
      fprintf(stderr,"Error: environment variable CINTSYSDIR is not set. makecint aborted.\n");
#endif
      G__cintsysdir[0]='\0';
      return(1);
    }
  }
  return(0);
}


/******************************************************************
* G__readMAKEINFO
******************************************************************/
void G__readMAKEINFO()
{
  char makeinfo[G__MAXFILENAME];
  FILE *fp;
  char line[G__MAXLINE];
  char argbuf[G__MAXLINE];
  char *arg[G__MAXARG];
  int argn;

#ifdef G__DJGPP
  char comm[G__LONGLINE];
  FILE* from;
  FILE* to;
  if(getenv("DJGPPDIR")) strcpy(G__DJGPPDIR,getenv("DJGPPDIR"));
#endif

  /* Get $CINTSYSDIR/MAKEINFO file name */
  if(G__getcintsysdir()) exit(EXIT_FAILURE);
  sprintf(makeinfo,"%s/MAKEINFO",G__cintsysdir);

  /* Open MAKEINFO file */
  fp = fopen(makeinfo,"r");
  if(!fp) {
    fprintf(stderr,"Error: can not open %s. Makecint aborted\n",makeinfo);
    fprintf(stderr
,"!!!Advice. There are examples of MAKEINFO files under %s/platform/ directory.\n"
	    ,G__cintsysdir);
    fprintf(stderr
	    ,"Please refer to these examples and create for your platform\n");
    exit(EXIT_FAILURE);
  }

  /* Read the MAKEINFO file */
  while(G__readline(fp,line,argbuf,&argn,arg)) {
    if(argn>2 && strcmp(arg[1],"CSRCPOST")==0) strcpy(G__CSRCPOST,arg[3]);
    else if(argn>2 && strcmp(arg[1],"CHDRPOST")==0) strcpy(G__CHDRPOST,arg[3]);
    else if(argn>2 && strcmp(arg[1],"CPPHDRPOST")==0) 
      strcpy(G__CPPHDRPOST,arg[3]);
    else if(argn>2 && strcmp(arg[1],"CPPSRCPOST")==0) 
      strcpy(G__CPPSRCPOST,arg[3]);
    else if(argn>2 && strcmp(arg[1],"OBJPOST")==0) strcpy(G__OBJPOST,arg[3]);
    else if(argn>2 && strcmp(arg[1],"DLLPOST")==0) {
      strcpy(G__DLLPOST,arg[3]);
#ifndef G__OLDIMPLEMENTATION1640
      if(argn>3) {
	char *p = strrchr(G__object,'.');
	if(p) {
	  if('.'==G__DLLPOST[0]) strcpy(p,G__DLLPOST);
	  else                   strcpy(p+1,G__DLLPOST);
	}
      }
#endif
    }
#ifdef G__DJGPP
    else if(argn>2 && strcmp(arg[1],"DJGPPDIR")==0 && 0==G__DJGPPDIR[0]) 
      strcpy(G__DJGPPDIR,arg[3]);
#endif
  }
  fclose(fp);

#ifdef G__DJGPP
  sprintf(comm,"%s/lib/crt0.o",G__DJGPPDIR);
  from = fopen(comm,"rb");
  to = fopen("crt0.o","wb");
  if(from && to) G__copyfile(to,from);
  else fprintf(stderr,"Error: DJGPP's crt0.o not found\n");
  if(from) fclose(from);
  if(to) fclose(to);
#endif
}

/******************************************************************
* G__storestringlist
******************************************************************/
struct G__string_list* G__storestringlist(list,string)
struct G__string_list *list;
char *string;
{
  struct G__string_list *p;
  if(!list) {
    p = (struct G__string_list*)malloc(sizeof(struct G__string_list));
    p->string = (char*)malloc(strlen(string)+1);
    strcpy(p->string,string);
    p->object = (char*)NULL;
    p->misc = (char*)NULL;
    p->next = (struct G__string_list*)NULL;
    return(p);
  }
  else {
    p = list;
    while(p->next) p=p->next;
    p->next = (struct G__string_list*)malloc(sizeof(struct G__string_list));
    p = p->next;
    p->string = (char*)malloc(strlen(string)+1);
    strcpy(p->string,string);
    p->object = (char*)NULL;
    p->misc = (char*)NULL;
    p->next = (struct G__string_list*)NULL;
    return(list);
  }
}

/******************************************************************
* G__freestringlist
******************************************************************/
void G__freestringlist(list)
struct G__string_list *list;
{
  struct G__string_list *p;
  p = list;
  if(p) {
    if(p->string) free((void*)p->string);
    if(p->object) free((void*)p->object);
    if(p->misc) free((void*)p->misc);
    if(p->next) G__freestringlist(p->next);
    free((void*)p);
  }
}


/******************************************************************
* G__cleanup()
******************************************************************/
void G__cleanup()
{
  if(G__MACRO) G__freestringlist(G__MACRO);
  if(G__IPATH) G__freestringlist(G__IPATH);
  if(G__CHDR)  G__freestringlist(G__CHDR);
  if(G__CPPHDR) G__freestringlist(G__CPPHDR);
  if(G__CSRC)   G__freestringlist(G__CSRC);
  if(G__CPPSRC) G__freestringlist(G__CPPSRC);
  if(G__LIB)    G__freestringlist(G__LIB);
  if(G__CCOPT)    G__freestringlist(G__CCOPT);
  if(G__CIOPT)    G__freestringlist(G__CIOPT);
  if(G__CSTUB)   G__freestringlist(G__CSTUB);
  if(G__CPPSTUB)   G__freestringlist(G__CPPSTUB);
}

/******************************************************************
* G__ispostfix()
*  test if the string has specific postfix
*****************************************************************/
int G__ispostfix(string,postfix)
char *string;
char *postfix;
{
  int lenpost;
  int len;
  lenpost = strlen(postfix);
  len = strlen(string);

  if(len>=lenpost && strcmp(string+len-lenpost,postfix)==0) return(1);
  else return(0);
}

/******************************************************************
* G__replacepostfix()
*  test if the string has specific postfix
*****************************************************************/
char *G__replacepostfix(string,postfix,buf)
char *string;
char *postfix;
char *buf;
{
  int lenpost;
  int len;
  char *postpos;
  lenpost = strlen(postfix);
  len = strlen(string);

  if(len>=lenpost && strcmp(string+len-lenpost,postfix)==0) {
    strcpy(buf,string);
    postpos = G__strrstr(buf,postfix);
    return(postpos);
  }
  else return((char*)NULL);
}

/******************************************************************
* G__checksourcefiles()
*
* check string list and identify it the item is 
*   1) source file      string="xxx.C" object="xxx.o"  misc=NULL
*   2) object file      string=NULL    object="xxx.o"  misc=NULL
*   3) others(misc)     string=NULL    object=NULL     misc="xxx"
******************************************************************/
void G__checksourcefiles(list,srcpost,objpost)
struct G__string_list *list;
char *srcpost;
char *objpost;
{
  char buf[G__MAXFILENAME];
  struct G__string_list *p;
  char *postfix;

  p = list;
  while(p) {
    /* assignment intended in following if statement */
    if((postfix=G__replacepostfix(p->string,srcpost,buf))||
       (postfix=G__replacepostfix(p->string,".c",buf))||
       (postfix=G__replacepostfix(p->string,".C",buf))||
       (postfix=G__replacepostfix(p->string,".cc",buf))||
       (postfix=G__replacepostfix(p->string,".CC",buf))||
       (postfix=G__replacepostfix(p->string,".cxx",buf))||
       (postfix=G__replacepostfix(p->string,".CXX",buf))||
       (postfix=G__replacepostfix(p->string,".cpp",buf))||
       (postfix=G__replacepostfix(p->string,".CPP",buf))) {
      strcpy(postfix,objpost);
      p->object = (char*)malloc(strlen(buf)+1);
      strcpy(p->object,buf);
    }
    else if(G__ispostfix(p->string,objpost)) {
      p->object = p->string;
      p->string = (char*)NULL;
    }
    else {
      fprintf(stderr
	      ,"makecint: Warning unrecognized name %s given as source file\n"
	      ,p->string);
      p->misc = p->string;
      p->string = (char*)NULL;
    }
    p = p->next;
  }
}

/******************************************************************
* G__printstringlist()
******************************************************************/
void G__printstringlist(fp,list,mode)
FILE *fp;
struct G__string_list *list;
int mode;
{
  struct G__string_list *p;

  p = list;
  while(p) {
    switch(mode) {
    case G__PRINTSTRING:
      if(p->string) fprintf(fp,"\\\n\t\t%s ",p->string);
      else if(p->misc)   fprintf(fp,"\\\n\t\t%s ",p->misc);
      break;
    case G__PRINTOBJECT:
      if(p->object) fprintf(fp,"\\\n\t\t%s ",p->object);
      break;
    case G__PRINTOBJECT_WPLUS:
      if(p->object) fprintf(fp,"+\n%s",p->object);
      break;
    case G__PRINTOBJECT_WSRC:
      if(p->object && p->string) fprintf(fp,"\\\n\t\t%s ",p->object);
      break;
    default:
      fprintf(stderr,"makecint internal error G__printstringlist()\n");
      break;
    }
    p = p->next;
  }
}

/******************************************************************
* G__printstringlist_noopt()
******************************************************************/
void G__printstringlist_noopt(fp,list,mode)
FILE *fp;
struct G__string_list *list;
int mode;
{
  struct G__string_list *p;

  p = list;
  while(p) {
    switch(p->string[0]) {
    case '+':
    case '-':
      /* +P,-P,+V,-V options are ignored */
      break;
    default:
      switch(mode) {
      case G__PRINTSTRING:
	if(p->string) fprintf(fp,"\\\n\t\t%s ",p->string);
	else if(p->misc)   fprintf(fp,"\\\n\t\t%s ",p->misc);
	break;
      case G__PRINTOBJECT:
	if(p->object) fprintf(fp,"\\\n\t\t%s ",p->object);
	break;
      default:
	fprintf(stderr,"makecint internal error G__printstringlist()\n");
	break;
      }
      break;
    }
    p = p->next;
  }
}


/******************************************************************
* G__printstringlist()
******************************************************************/
void G__printsourcecompile(fp,list,headers,compiler)
FILE *fp;
struct G__string_list *list;
char *headers;
char *compiler;
{
  struct G__string_list *p;

  p = list;
  while(p) {
    if(p->string && p->object) {
      fprintf(fp,"%s : %s %s\n",p->object,p->string,headers);
#ifdef CINTSYSDIR
      fprintf(fp,"\t%s $(IPATH) $(CINTIPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o %s -c %s\n"
	      ,compiler,p->object,p->string);
#else
      fprintf(fp,"\t%s $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o %s -c %s\n"
	      ,compiler,p->object,p->string);
#endif
      fprintf(fp,"\n");
    }
    p = p->next;
  }
}


  
/******************************************************************
* G__readargument
******************************************************************/
int G__readargument(argc,argv)
int argc;
char **argv;
{
  char *p;
  char buf[G__MAXFILENAME];
  int mode=G__IDLE;
  int i=1;
  while(i<argc) {
    /*************************************************************************
    * options with no argument
    *************************************************************************/
    if(strcmp(argv[i],"-p")==0) {
      strcpy(G__preprocess,argv[i]);
      mode = G__IDLE;
    }
#ifndef G__OLDIMPLEMENTATION1452
    /*************************************************************************/
    else if(strcmp(argv[i],"-u")==0) {
      G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
      G__CIOPT=G__storestringlist(G__CIOPT,argv[++i]);
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-u",2)==0) {
      G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-U")==0) {
      G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
      G__CIOPT=G__storestringlist(G__CIOPT,argv[++i]);
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-U",2)==0) {
      G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-Y")==0) {
      G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
      G__CIOPT=G__storestringlist(G__CIOPT,argv[++i]);
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-Y",2)==0) {
      G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-Z")==0) {
      G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
      G__CIOPT=G__storestringlist(G__CIOPT,argv[++i]);
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-Z",2)==0) {
      G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
      mode = G__IDLE;
    }
#endif
    /*************************************************************************/
    else if(strcmp(argv[i],"-m")==0) {
      G__ismain = 1;
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-?")==0) {
      G__displayhelp();
      mode = G__IDLE;
      exit(EXIT_SUCCESS);
    }
    /*************************************************************************
    * options with 1 argument
    *************************************************************************/
    else if(strcmp(argv[i],"-D")==0) {
      sprintf(buf,"%s%s",argv[i],argv[i+1]);
      G__MACRO = G__storestringlist(G__MACRO,buf);
      i++;
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-D",2)==0
	    && G__COMPILEROPT!=mode && G__CINTOPT!=mode
	    ) {
      G__MACRO = G__storestringlist(G__MACRO,argv[i]);
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-I")==0) {
      sprintf(buf,"%s%s",argv[i],argv[i+1]);
      G__IPATH = G__storestringlist(G__IPATH,buf);
      i++;
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-I",2)==0 
	    && G__COMPILEROPT!=mode && G__CINTOPT!=mode
	    ) {
      G__IPATH = G__storestringlist(G__IPATH,argv[i]);
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-B")==0) {
      ++i;
      sprintf(G__INITFUNC,"-B%s",argv[i]);
    }
    else if(strcmp(argv[i],"-y")==0) {
      ++i;
      /* WinNT/95 only */
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-o")==0) {
      i++;
      strcpy(G__object,argv[i]);
      p = strrchr(G__object,'/');
      if(!p) p = G__object;
      else p++;
      strcpy(G__DLLID,p);
      p = strchr(G__DLLID,'.');
      if(p) *p = '\0';
      G__isDLL = 0;
      mode = G__IDLE;
#ifdef G__DJGPP
      if(!strstr(G__object,".EXE") && !strstr(G__object,".exe")) {
        strcat(G__object,".exe");
      }
#endif
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-dl")==0 || strcmp(argv[i],"-sl")==0 ||
	    strcmp(argv[i],"-dll")==0 || strcmp(argv[i],"-DLL")==0 ||
	    strcmp(argv[i],"-so")==0) {
      i++;
      strcpy(G__object,argv[i]);
      p = strrchr(G__object,'/');
      if(!p) p = G__object;
      else p++;
      strcpy(G__DLLID,p);
      p = strchr(G__DLLID,'.');
      if(p) *p = '\0';
      G__isDLL = 1;
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-mk")==0) {
      i++;
      strcpy(G__makefile,argv[i]);
      mode = G__IDLE;
    }
    /*************************************************************************
    * options with multiple argument
    *************************************************************************/
    else if(strcmp(argv[i],"-h")==0) {
      mode = G__CHEADER;
    }
    else if(strcmp(argv[i],"-H")==0) {
      mode = G__CPPHEADER;
    }
    else if(strcmp(argv[i],"-C")==0) {
      mode = G__CSOURCE;
    }
    else if(strcmp(argv[i],"-C++")==0) {
      mode = G__CPPSOURCE;
    }
    else if(strcmp(argv[i],"-l")==0) {
      mode = G__LIBRARY;
    }
    else if(strcmp(argv[i],"-cc")==0) {
      mode = G__COMPILEROPT;
    }
    else if(strcmp(argv[i],"-cint")==0) {
      mode = G__CINTOPT;
    }
    else if(strcmp(argv[i],"-i")==0) {
      mode = G__CSTUBFILE;
    }
    else if(strcmp(argv[i],"-i++")==0) {
      mode = G__CPPSTUBFILE;
    }
    else if(strcmp(argv[i],"-c")==0) {
      /* fprintf(stderr,"makecint: -c being obsoleted. no guarantee\n"); */
      mode = G__CHEADER;
    }
    /*************************************************************************/
    else {
      switch(mode) {
      case G__CHEADER:
	G__CHDR=G__storestringlist(G__CHDR,argv[i]);
	break;
      case G__CSOURCE:
	G__CSRC=G__storestringlist(G__CSRC,argv[i]);
	break;
      case G__CPPHEADER:
	G__CPPHDR=G__storestringlist(G__CPPHDR,argv[i]);
	break;
      case G__CPPSOURCE:
	G__CPPSRC=G__storestringlist(G__CPPSRC,argv[i]);
	break;
      case G__LIBRARY:
	G__LIB=G__storestringlist(G__LIB,argv[i]);
	break;
      case G__COMPILEROPT:
	G__CCOPT=G__storestringlist(G__CCOPT,argv[i]);
	break;
      case G__CINTOPT:
	G__CIOPT=G__storestringlist(G__CIOPT,argv[i]);
	break;
      case G__CSTUBFILE:
	G__CSTUB=G__storestringlist(G__CSTUB,argv[i]);
	break;
      case G__CPPSTUBFILE:
	G__CPPSTUB=G__storestringlist(G__CPPSTUB,argv[i]);
	break;
      case G__IDLE:
      default:
	break;
      }
    }
    ++i;
  }
  return(0);
}

/******************************************************************
* G__check
******************************************************************/
int G__check(buf,item,where)
char *buf;
char *item;
char *where;
{
  if((!buf) || '\0'==buf[0]) {
    fprintf(stderr,"Error: %s must be set %s\n",item,where);
    return(1);
  }
  return(0);
}

/******************************************************************
* G__checksetup
******************************************************************/
int G__checksetup()
{
  int error=0;
  if(G__isDLL) {
    error+=G__check(G__object,"'-dl [DLL]'","in the command line");
    error+=G__check(G__DLLPOST,"DLLPOST","in the $(CINTSYSDIR)/MAKEINFO file");
  }
  else {
    error+=G__check(G__object,"'-o [Object]'","in the command line");
  }
  error+=G__check(G__makefile,"'-mk [Makefile]'","in the command line");
  error+=G__check(G__OBJPOST,"OBJPOST","in the $(CINTSYSDIR)/MAKEINFO file");
  if(G__CHDR) {
    error+=G__check(G__CHDRPOST,"CSRCPOST"
		    ,"in the $(CINTSYSDIR)/MAKEINFO file");
  }
  if(G__CPPHDR) {
    error+=G__check(G__CHDRPOST,"CPPSRCPOST"
		    ,"in the $(CINTSYSDIR)/MAKEINFO file");
  }
  return(error);
}

/******************************************************************
* G__outputmakefile
******************************************************************/
void G__outputmakefile(argc,argv)
int argc;
char **argv;
{
  char makeinfo[G__MAXFILENAME];
  FILE *makeinfofp;
  FILE *fp;
  int i;
#ifdef G__DJGPP
  char cintsysdir[G__MAXFILENAME];
  int j,k;
#endif

  fp = fopen(G__makefile,"w");
  if(!fp) {
    fprintf(stderr,"Error: can not create %s\n",G__makefile);
    exit(EXIT_FAILURE);
  }
  sprintf(makeinfo,"%s/MAKEINFO",G__cintsysdir);
  makeinfofp = fopen(makeinfo,"r");

  fprintf(fp,"############################################################\n");
  fprintf(fp,"# Automatically created makefile for %s\n",G__object);
  fprintf(fp,"############################################################\n");

  fprintf(fp,"\n");
  fprintf(fp,"# Copying $CINTSYSDIR/MAKEINFO #############################\n");
  fprintf(fp,"\n");

  /***************************************************************************
   * Copy platform dependent information fro $CINTSYSDIR/MAKEINFO 
   ***************************************************************************/
  G__copyfile(fp,makeinfofp);
  fclose(makeinfofp);

  fprintf(fp,"\n");
  fprintf(fp,"# End of $CINTSYSDIR/MAKEINFO ##############################\n");
  fprintf(fp,"\n");

  /***************************************************************************
   * Print out variables
   ***************************************************************************/
  fprintf(fp,"# Set variables ############################################\n");
#if defined(G__MINGW)
  fprintf(fp,"IPATH      = $(subst \\,/,$(SYSIPATH))\n");
#else
  fprintf(fp,"IPATH      = $(SYSIPATH) ");
#endif
  G__printstringlist(fp,G__IPATH,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"MACRO      = $(SYSMACRO)");
  G__printstringlist(fp,G__MACRO,G__PRINTSTRING);
  fprintf(fp,"\n");
#ifdef G__DJGPP
  j=k=0;
  if(':'==G__cintsysdir[1] && '\\'==G__cintsysdir[2]) {
    strcpy(cintsysdir,"/");
    k=1;
    j=3;
  }
  while(G__cintsysdir[j]) {
    switch(G__cintsysdir[j]) {
    case '\\':
      cintsysdir[k++]='/';
      j++;
      break;
    default:
      cintsysdir[k++]=G__cintsysdir[j++];
      break;
    }
  }
  cintsysdir[k]='\0';
  fprintf(fp,"CINTSYSDIR = %s\n",cintsysdir);
#else
  fprintf(fp,"CINTSYSDIR = %s\n",G__cintsysdir);
#endif
#ifdef __hpux
  fprintf(fp,"CINTIPATH  = \n");
#elif defined(G__MINGW)
  fprintf(fp,"CINTIPATH  = -I$(subst \\,/,$(CINTSYSDIR))\n");
#else
  fprintf(fp,"CINTIPATH  = -I$(CINTSYSDIR)\n");
#endif
  fprintf(fp,"OBJECT     = %s\n",G__object);
  if(G__isDLL) {
    fprintf(fp,"OPTION     = $(CCDLLOPT)\n");
    /* fprintf(fp,"DLLSPEC    = -N%s\n",G__DLLID); */
    fprintf(fp,"DLLSPEC    =\n");
  }
  else {
    fprintf(fp,"OPTION     =\n");
    fprintf(fp,"DLLSPEC    =\n");
  }

  fprintf(fp,"LINKSPEC   =");
  if(G__CHDR) fprintf(fp," -DG__CLINK_ON");
  if(G__CPPHDR) fprintf(fp," -DG__CPPLINK_ON");
  fprintf(fp,"\n");
  fprintf(fp,"\n");
  fprintf(fp,"# Set File names ###########################################\n");

  if(G__CHDR) {
    fprintf(fp,"CIFC       = G__c_%s%s\n",G__DLLID,G__CSRCPOST);
    fprintf(fp,"CIFH       = G__c_%s%s\n",G__DLLID,G__CHDRPOST);
    fprintf(fp,"CIFO       = G__c_%s%s\n",G__DLLID,G__OBJPOST);
  }
  else {
    fprintf(fp,"CIFC       =\n");
    fprintf(fp,"CIFH       =\n");
    fprintf(fp,"CIFO       =\n");
  }
  if(G__CPPHDR) {
    fprintf(fp,"CPPIFC     = G__cpp_%s%s\n",G__DLLID,G__CPPSRCPOST);
    fprintf(fp,"CPPIFH     = G__cpp_%s%s\n",G__DLLID,G__CPPHDRPOST);
    fprintf(fp,"CPPIFO     = G__cpp_%s%s\n",G__DLLID,G__OBJPOST);
  }
  else {
    fprintf(fp,"CPPIFC     =\n");
    fprintf(fp,"CPPIFH     =\n");
    fprintf(fp,"CPPIFO     =\n");
  }
  fprintf(fp,"\n");

  fprintf(fp,"LIBS       = ");
  G__printstringlist(fp,G__LIB,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"CCOPT      = ");
  G__printstringlist(fp,G__CCOPT,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"CINTOPT      = ");
  G__printstringlist(fp,G__CIOPT,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"COFILES    = ");
  G__printstringlist(fp,G__CSRC,G__PRINTOBJECT);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"RMCOFILES  = ");
  G__printstringlist(fp,G__CSRC,G__PRINTOBJECT_WSRC);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"CHEADER    = ");
  G__printstringlist_noopt(fp,G__CHDR,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"CHEADERCINT = ");
  G__printstringlist(fp,G__CHDR,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"CSTUB      = ");
  G__printstringlist_noopt(fp,G__CSTUB,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"CSTUBCINT  = ");
  G__printstringlist(fp,G__CSTUB,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"CPPOFILES  = ");
  G__printstringlist(fp,G__CPPSRC,G__PRINTOBJECT);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"RMCPPOFILES = ");
  G__printstringlist(fp,G__CPPSRC,G__PRINTOBJECT_WSRC);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"CPPHEADER  = ");
  G__printstringlist_noopt(fp,G__CPPHDR,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"CPPHEADERCINT  = ");
  G__printstringlist(fp,G__CPPHDR,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fprintf(fp,"CPPSTUB    = ");
  G__printstringlist_noopt(fp,G__CPPSTUB,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"CPPSTUBCINT = ");
  G__printstringlist(fp,G__CPPSTUB,G__PRINTSTRING);
  fprintf(fp,"\n");
  fprintf(fp,"\n");


  /***************************************************************************
   * Link Object
   ***************************************************************************/
  fprintf(fp,"# Link Object #############################################\n");
  if(G__isDLL) {
#if defined(_AIX)
    fprintf(fp,"$(OBJECT) : $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO)\n");
    fprintf(fp,"\t$(LDDLL) $(LDDLLOPT) -o $(OBJECT) $(COFILES) $(CIFO) $(CPPIFO) $(CPPOFILES) $(LIBS)\n");

#elif (G__CYGWIN>=50)
    fprintf(fp,"$(OBJECT) : $(CINTLIB) $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO)\n");
    fprintf(fp,"\t$(LD) $(LDDLLOPT) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(COFILES) $(CIFO) $(CPPIFO) $(CPPOFILES) $(LIBS) -L$(CINTSYSDIR) -lcint\n");
#if 0 /* DEBUG */
    fprintf(fp,"\t$(DLLTOOL) --as=$(AS) --dllname $(OBJECT) --def %s.def --output-lib %s.lib\n",G__DLLID,G__DLLID);
#endif

#elif defined(G__CYGWIN)||defined(G__MINGW)
    /* Problem reported by Joseph Canedo 2001/6/7 $(CINTLIB) added for DLL */
    /* This is not a good fix. Use G__CYGWIN_DLL instead */
    fprintf(fp,"$(OBJECT) : $(CINTLIB) $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO)\n");
    fprintf(fp,"\t$(LD) $(LDDLLOPT) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(COFILES) $(CIFO) $(CPPIFO) $(CPPOFILES) $(CINTLIB) $(LIBS)\n");
#else
    fprintf(fp,"$(OBJECT) : $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO)\n");
    fprintf(fp,"\t$(LD) $(LDDLLOPT) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(COFILES) $(CIFO) $(CPPIFO) $(CPPOFILES) $(LIBS)\n");
#endif
  }
  else if(G__ismain) {
#ifdef _AIX
    fprintf(fp
   ,"$(OBJECT) : $(CINTLIB) $(READLINEA) $(DLFCN) G__setup.o $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) \n");
    fprintf(fp,"\t$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(DLFCN) $(LIBS) $(LDOPT)\n");
#elif (G__CYGWIN>=50)
    fprintf(fp
   ,"$(OBJECT) : $(CINTLIB) $(READLINEA) $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) \n");
    fprintf(fp,"\t$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) $(READLINEA) $(LIBS) $(LDOPT)\n");
#else
    fprintf(fp
   ,"$(OBJECT) : $(CINTLIB) $(READLINEA) G__setup.o $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) \n");
    fprintf(fp,"\t$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(LIBS) $(LDOPT)\n");
#endif
  }
  else {
#ifdef _AIX
    fprintf(fp
   ,"$(OBJECT) : $(MAINO) $(CINTLIB) $(READLINEA) $(DLFCN) G__setup.o $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) \n");
    fprintf(fp,"\trm -f shr.o $(OBJECT).nm $(OBJECT).exp\n");
    fprintf(fp,"\t$(NM) $(MAINO) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(DLFCN) $(LIBS) $(NMOPT)\n");
    fprintf(fp,"\trm -f shr.o\n");
    fprintf(fp,"\techo \"#!\" > $(OBJECT).exp ; cat $(OBJECT).nm >> $(OBJECT).exp\n");
    fprintf(fp,"\trm -f $(OBJECT).nm\n");
    fprintf(fp,"\t$(LD) $(OPTIMIZE) -bE:$(OBJECT).exp -bM:SRE  $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(MAINO) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(DLFCN) $(LIBS) $(LDOPT)\n");
#elif (G__CYGWIN>=50)
    fprintf(fp
   ,"$(OBJECT) : G__main.o $(CINTLIB) $(READLINEA) $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) \n");
    fprintf(fp,"\t$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) G__main.o $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) $(READLINEA) $(LIBS) $(LDOPT)\n");
#else
    fprintf(fp
   ,"$(OBJECT) : $(MAINO) $(CINTLIB) $(READLINEA) G__setup.o $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) \n");
    fprintf(fp,"\t$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(MAINO) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(LIBS) $(LDOPT)\n");
#endif
  }
  fprintf(fp,"\n");

  /***************************************************************************
   * Compile user source
   ***************************************************************************/
  fprintf(fp,"# Compile User C source files ##############################\n");
  G__printsourcecompile(fp,G__CSRC ,"$(CHEADER)","$(CC)");
  fprintf(fp,"\n");
  fprintf(fp,"# Compile User C++ source files ############################\n");
  G__printsourcecompile(fp,G__CPPSRC ,"$(CPPHEADER)","$(CPP)");
  fprintf(fp,"\n");

  /***************************************************************************
   * Compille Initialization routine
   ***************************************************************************/
  if(!G__isDLL) {
#if (G__CYGWIN>=50)
    fprintf(fp,"\n");
    fprintf(fp,"# Compile main function  #################################\n");
    fprintf(fp,"G__main.o : G__main.cxx\n");
    fprintf(fp,"\t$(CPP) $(LINKSPEC) $(CINTIPATH) $(OPTIMIZE) $(OPTION) -o G__main.o -c G__main.cxx\n");
#else
    fprintf(fp,"# Compile dictionary setup routine #######################\n");
    fprintf(fp,"G__setup.o : $(CINTSYSDIR)/main/G__setup.c $(CINTSYSDIR)/G__ci.h\n");
    fprintf(fp,"\t$(CC) $(LINKSPEC) $(CINTIPATH) $(OPTIMIZE) $(OPTION) -o G__setup.o -c $(CINTSYSDIR)/main/G__setup.c\n");
#endif
  }
  fprintf(fp,"\n");
    
  /***************************************************************************
   * Interface routine
   ***************************************************************************/
  if(G__CHDR) {
    fprintf(fp,"# Compile C Interface routine ############################\n");
    fprintf(fp,"$(CIFO) : $(CIFC)\n");
    fprintf(fp,"\t$(CC) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CIFC)\n");
    fprintf(fp,"\n");
    fprintf(fp,"# Create C Interface routine #############################\n");
#if defined(CINTSYSDIR)
    fprintf(fp,"$(CIFC) : $(CHEADER) $(CSTUB)\n");
#elif defined(G__DJGPP)
    fprintf(fp,"$(CIFC) : $(CHEADER) $(CSTUB) $(CINTSYSDIR)/cint.exe\n");
#else
    fprintf(fp,"$(CIFC) : $(CHEADER) $(CSTUB) $(CINTSYSDIR)/cint\n");
#endif
    /* Following line needs explanation. -K is used at the beginning and 
     * later again $(KRMODE) may be set to -K. When -K is given after -c-2
     * it will set G__clock flags so that it will create K&R compatible 
     * function headers. This is not a good manner but -K -c-2 and -c-2 -K
     * has different meaning. */
#ifdef CINTSYSDIR
    fprintf(fp,"\tcint %s -K -w%d -z%s -n$(CIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT %s -c-2 $(KRMODE) $(IPATH) $(MACRO) $(CINTOPT) $(CHEADERCINT)" 
	    ,G__INITFUNC,G__isDLL,G__DLLID,G__preprocess);
#else
    fprintf(fp,"\t$(CINTSYSDIR)/cint %s -K -w%d -z%s -n$(CIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT %s -c-2 $(KRMODE) $(IPATH) $(MACRO) $(CINTOPT) $(CHEADERCINT)" 
	    ,G__INITFUNC,G__isDLL,G__DLLID,G__preprocess);
#endif
    if(G__CSTUB) fprintf(fp," +STUB $(CSTUBCINT) -STUB\n");
    else      fprintf(fp,"\n");
    fprintf(fp,"\n");
  }
  if(G__CPPHDR) {
    fprintf(fp,"# Compile C++ Interface routine ##########################\n");
    fprintf(fp,"$(CPPIFO) : $(CPPIFC)\n");
    fprintf(fp,"\t$(CPP) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CPPIFC)\n");
    fprintf(fp,"\n");
    fprintf(fp,"# Create C++ Interface routine ###########################\n");
#if defined(CINTSYSDIR)
    fprintf(fp,"$(CPPIFC) : $(CPPHEADER) $(CPPSTUB)\n");
#elif defined(G__DJGPP)
    fprintf(fp,"$(CPPIFC) : $(CPPHEADER) $(CPPSTUB) $(CINTSYSDIR)/cint.exe\n");
#else
    fprintf(fp,"$(CPPIFC) : $(CPPHEADER) $(CPPSTUB) $(CINTSYSDIR)/cint\n");
#endif
#ifdef CINTSYSDIR
    fprintf(fp,"\tcint %s -w%d -z%s -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT %s -c-1 -A $(IPATH) $(MACRO) $(CINTOPT) $(CPPHEADERCINT)"
	    ,G__INITFUNC,G__isDLL,G__DLLID,G__preprocess);
#else
    fprintf(fp,"\t$(CINTSYSDIR)/cint %s -w%d -z%s -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT %s -c-1 -A $(IPATH) $(MACRO) $(CINTOPT) $(CPPHEADERCINT)"
	    ,G__INITFUNC,G__isDLL,G__DLLID,G__preprocess);
#endif
    if(G__CPPSTUB) fprintf(fp," +STUB $(CPPSTUBCINT) -STUB\n");
    else        fprintf(fp,"\n");
    fprintf(fp,"\n");
  }


  fprintf(fp,"\n");
  fprintf(fp,"# Clean up #################################################\n");
  fprintf(fp,"clean :\n");
  if(G__isDLL) {
    fprintf(fp,"\t$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES)\n");
  }
  else {
#ifdef _AIX
    fprintf(fp,"\t$(RM) $(OBJECT) $(OBJECT).exp $(OBJECT).nm shr.o core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(COFILES) $(CPPOFILES) G__setup.o\n");
#elif (G__CYGWIN>=50)
    fprintf(fp,"\t$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES) G__main.o\n");
#else
    fprintf(fp,"\t$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES) G__setup.o\n");
#endif
  }
  fprintf(fp,"\n");

  fprintf(fp,"# re-makecint ##############################################\n");
  fprintf(fp,"makecint :\n");
  fprintf(fp,"\tmakecint ");
  for(i=1;i<argc;i++) {
    fprintf(fp,"%s ",argv[i]);
  }
  fprintf(fp,"\n");
  fprintf(fp,"\n");

  fclose(fp);
}

#if (G__CYGWIN>=50)
/******************************************************************
* G__outputmain()
******************************************************************/
void G__outputmain()
{
  FILE *mainfp;
  char G__DLLID[10] = "";
  /*****************************************************************
  * creating G__main.cxx
  *****************************************************************/
  mainfp = fopen("G__main.cxx","w");
  fprintf(mainfp,"/******************************************************\n");
  fprintf(mainfp,"* G__main.cxx\n");
  fprintf(mainfp,"*  automatically generated main() function for cint\n");
  fprintf(mainfp,"*  Cygwin environment\n");
  fprintf(mainfp,"******************************************************/\n");
  fprintf(mainfp,"#include <stdio.h>\n");
  fprintf(mainfp,"extern \"C\" {\n");
  fprintf(mainfp,"extern void G__setothermain(int othermain);\n");
  fprintf(mainfp,"extern int G__main(int argc,char **argv);\n");
  fprintf(mainfp,"extern void G__set_p2fsetup(void (*p2f)());\n");
  fprintf(mainfp,"extern void G__free_p2fsetup();\n");
  if(G__CHDR) fprintf(mainfp,"extern void G__c_setup%s();\n",G__DLLID);
  if(G__CPPHDR) fprintf(mainfp,"extern void G__cpp_setup%s();\n",G__DLLID);
  fprintf(mainfp,"}\n");
  fprintf(mainfp,"\n");
#ifndef G__OLDIMPLEMENTATION874
  if(G__ismain) {
    fprintf(mainfp,"class G__DMYp2fsetup {\n");
    fprintf(mainfp," public:\n");
    fprintf(mainfp,"  G__DMYp2fsetup() { \n");
    if(G__CHDR) fprintf(mainfp,"    G__set_p2fsetup(G__c_setup%s);\n",G__DLLID);
    if(G__CPPHDR) fprintf(mainfp,"    G__set_p2fsetup(G__cpp_setup%s);\n",G__DLLID);
    fprintf(mainfp,"  }\n");
    fprintf(mainfp,"} G__DMY;\n");
  }
  else {
#endif
    fprintf(mainfp,"int main(int argc,char **argv)\n");
    fprintf(mainfp,"{\n");
    fprintf(mainfp,"  int result;\n");
    if(G__CHDR) fprintf(mainfp,"  G__set_p2fsetup(G__c_setup%s);\n",G__DLLID);
    if(G__CPPHDR) fprintf(mainfp,"  G__set_p2fsetup(G__cpp_setup%s);\n",G__DLLID);
    fprintf(mainfp,"  G__setothermain(0);\n");
    fprintf(mainfp,"  result=G__main(argc,argv);\n");
    fprintf(mainfp,"  G__free_p2fsetup();\n");
    fprintf(mainfp,"  return(result);\n");
    fprintf(mainfp,"}\n");
#ifndef G__OLDIMPLEMENTATION874
  }
#endif

  fclose(mainfp);
  /*****************************************************************
  * end of creating G__main.cxx
  *****************************************************************/
}
#endif

/******************************************************************
* G__makecint
******************************************************************/
int G__makecint(argc,argv)
int argc;
char **argv;
{
  G__printtitle();
  G__readargument(argc,argv);
  G__readMAKEINFO();
  G__checksourcefiles(G__CSRC,G__CSRCPOST,G__OBJPOST);
  G__checksourcefiles(G__CPPSRC,G__CPPSRCPOST,G__OBJPOST);
  if(G__checksetup()) {
    fprintf(stderr,"!!!makecint aborted!!!  makecint -? for help\n");
    exit(EXIT_FAILURE);
  }
#if (G__CYGWIN>=50)
  if(!G__isDLL) G__outputmain();
#endif
  G__outputmakefile(argc,argv);
  G__cleanup();
  G__displaytodo();
  return(EXIT_SUCCESS);
}


/******************************************************************
* main
******************************************************************/
int main(argc,argv)
int argc;
char **argv;
{
  return(G__makecint(argc,argv));
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
