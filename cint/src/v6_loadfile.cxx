/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file loadfile.c
 ************************************************************************
 * Description:
 *  Loading source file
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__OLDIMPLEMENTATION1920 
/* Define one of following */
#if 0
#define G__OLDIMPLEMENTATION1923 /* keep tmpfile for -cN +V +P */
#else
#define G__OLDIMPLEMENTATION1922 /* keep opening all header files for +V +P */
#endif
#else  /* 1920 */
#define G__OLDIMPLEMENTATION1922 
#define G__OLDIMPLEMENTATION1923 
#endif /* 1920 */

#ifdef G__ROOT
#ifdef HAVE_CONFIG
#include "config.h"
#endif
#endif

#include "common.h"

#ifndef G__PHILIPPE28
#ifndef G__TESTMAIN
#include <sys/stat.h>
#endif
#endif

#ifdef G__WIN32
#include <windows.h>
#endif

#define G__OLDIMPLEMENTATION1849
#ifndef G__OLDIMPLEMENTATION1849
#define G__RESTORE_LOADFILEENV \
  G__func_now = store_func_now; \
  G__macroORtemplateINfile = store_macroORtemplateINfile; \
  G__var_type = store_var_type; \
  G__tagnum = store_tagnum; \
  G__typenum = store_typenum; \
  G__nobreak=store_nobreak; \
  G__prerun=store_prerun; \
  G__p_local=store_p_local; \
  G__asm_noverflow = store_asm_noverflow; \
  G__no_exec_compile = store_no_exec_compile; \
  G__asm_exec = store_asm_exec; \
  G__ifile = store_file ; \
  G__eof = 0; \
  G__step=store_step; \
  G__globalcomp=G__store_globalcomp; \
  G__iscpp=store_iscpp; \
  G__security = store_security
#endif
  

/******************************************************************
* Define G__EDU_VERSION for CINT C++ educational version.
* If G__EDU_VERSION is defined, CINT will search ./include and
* ./stl directory for standard header files.
******************************************************************/
/* #define G__EDU_VERSION */


#ifndef G__OLDIMPLEMENTATION1207
extern int G__ispermanentsl;
extern G__DLLINIT G__initpermanentsl;
#endif

#ifndef G__OLDIMPLEMENTATION1210
static G__IgnoreInclude G__ignoreinclude = (G__IgnoreInclude)NULL;

/******************************************************************
* G__set_ignoreinclude
******************************************************************/
void G__set_ignoreinclude(ignoreinclude)
G__IgnoreInclude ignoreinclude;
{
  G__ignoreinclude = ignoreinclude;
}
#endif

static int G__kindofheader = G__USERHEADER;

#ifndef G__OLDIMPLEMENTATION970
static int G__copyflag = 0;

#ifndef G__PHILIPPE1
int (*G__ScriptCompiler) G__P((G__CONST char*,G__CONST char*)) = 0;

/******************************************************************
* G__RegisterScriptCompiler()
******************************************************************/
void G__RegisterScriptCompiler(p2f)
int(*p2f) G__P((G__CONST char*,G__CONST char*));
{
  G__ScriptCompiler = p2f;
}
#endif

#ifndef G__OLDIMPLEMENTATION1920
/******************************************************************
* G__copytotmpfile()
******************************************************************/
static FILE* G__copytotmpfile(prepname)
char *prepname;
{
  FILE *ifp;
  FILE *ofp;
  ifp = fopen(prepname,"rb");
  if(!ifp) {
    G__genericerror("Internal error: G__copytotmpfile() 1\n");
    return((FILE*)NULL);
  }
  ofp = tmpfile();
  if(!ofp) {
    G__genericerror("Internal error: G__copytotmpfile() 2\n");
    fclose(ifp);
    return((FILE*)NULL);
  }
  G__copyfile(ofp,ifp);
  fclose(ifp);
  fseek(ofp,0L,SEEK_SET);
  return(ofp);
}
#endif

/******************************************************************
* G__copysourcetotmp()
******************************************************************/
static void G__copysourcetotmp(prepname,pifile,fentry)
char *prepname;
struct G__input_file *pifile;
int fentry;
{
  if(G__copyflag && 0==prepname[0]) {
    FILE *fpout;
#ifndef G__OLDIMPLEMENTATION1920
    fpout = tmpfile();
    if(!fpout) {
      G__genericerror("Internal error: can not open tmpfile.");
      return;
    }
    /*strcpy(prepname,"(tmpfile)");*/
    sprintf(prepname,"(tmp%d)",fentry);
    G__copyfile(fpout,pifile->fp);
    fseek(fpout,0L,SEEK_SET);
    G__srcfile[fentry].prepname = (char*)malloc(strlen(prepname)+1);
    strcpy(G__srcfile[fentry].prepname,prepname);
    G__srcfile[fentry].fp = fpout;
    fclose(pifile->fp);
    pifile->fp = fpout;
#else /* 1920 */
    G__tmpnam(prepname); /* not used anymore */
    fpout = fopen(prepname,"wb");
    if(!fpout) {
      G__fprinterr(G__serr,"cannot open tmp file %s",prepname);
      G__genericerror((char*)NULL);
      prepname[0] = 0;
      return;
    }
    G__copyfile(fpout,pifile->fp);
    G__srcfile[fentry].prepname = (char*)malloc(strlen(prepname)+1);
    strcpy(G__srcfile[fentry].prepname,prepname);
    fclose(pifile->fp);
    fclose(fpout);
#ifndef G__WIN32
    pifile->fp = fopen(prepname,"r");
#else
    pifile->fp = fopen(prepname,"rb");
#endif
#endif /* 1920 */
  }
}

/**************************************************************************
* G__setcopyflag()
**************************************************************************/
void G__setcopyflag(flag)
int flag;
{
  G__copyflag = flag;
}
#endif

/******************************************************************
* G__ispreprocessfilekey
******************************************************************/
static int G__ispreprocessfilekey(filename)
char *filename;
{
  struct G__Preprocessfilekey *pkey;
  /* char *p; */

  pkey = &G__preprocessfilekey;

  /* check match of the keystring */
  while(pkey->next) {
    if(pkey->keystring && strstr(filename,pkey->keystring)) {
       return(1);
    }
    pkey=pkey->next;
  }

  /* No match */
  return(0);
}

/******************************************************************
* G__include_file()
*
* Keyword '#include' is read in G__exec_statement();
* G__include_file() read include filename, and load it.
*
*   #include   <stdio.h>    \n
*            ^---------------^       do nothing
*
*   #include  comment   "header.h" comment   \n
*            ^--------------------------------^    load "header.h"
*
******************************************************************/
int G__include_file()
{
  int result;
  int c;
  char filename[G__MAXFILENAME];
  int i=0;
  int storeit=0;
  /* char sysinclude[G__MAXFILENAME]; */
  int store_cpp;
  int store_globalcomp;
  int expandflag=0;
#ifndef G__OLDIMPLEMENTATION1725
  static int G__gcomplevel=0;
#endif

  while((c=G__fgetc())!='\n' && c!='\r'
#ifndef G__OLDIMPLEMENTATION1261
	&& c!='#'
#endif
	) {
    switch(c) {
    case '<':
      if(storeit==0) storeit=1;
      break;
    case '>':
      storeit = -1;
      G__kindofheader=G__SYSHEADER;
      break;
    case '\"':
      switch(storeit) {
      case 0:
	storeit=1;
	break;
      case 1:
	storeit = -1;
	G__kindofheader=G__USERHEADER;
	break;
      }
      break;
    default:
      if(!isspace(c)) {
	if(1==storeit) {
	  filename[i++]=c;
	  filename[i]='\0';
	}
	else if(-1!=storeit) {
	  storeit=1;
	  expandflag=1;
	  filename[i++]=c;
	  filename[i]='\0';
	}
      }
      else if(expandflag) {
	storeit = -1;
      }
      break;
    }
  }

#ifndef G__OLDIMPLEMENTATION460
  if(expandflag) {
    /* Following stupid code is written to avoid HP-UX CC -O bug */
    struct G__var_array *var;
    int ig15;
    int hash;
    G__hash(filename,hash,ig15);
    var = G__getvarentry(filename,hash,&ig15,&G__global,G__p_local);
    if(var) {
      strcpy(filename,*(char**)var->p[ig15]);
      G__kindofheader=G__USERHEADER;
    }
    else {
      G__fprinterr(G__serr,"Error: cannot expand #include %s",filename);
      G__genericerror(NULL);
#ifndef G__OLDIMPLEMENTATION1261
      if('#'==c) G__fignoreline();
#endif
      return(G__LOADFILE_FAILURE);
    }
  }
#endif

  store_cpp=G__cpp;
  G__cpp=G__include_cpp;

  if(G__USERHEADER==G__kindofheader) {
#ifndef G__OLDIMPLEMENTATION1725
    store_globalcomp = G__globalcomp;
    if(++G__gcomplevel>=G__gcomplevellimit) G__globalcomp=G__NOLINK;
    result = G__loadfile(filename);
    --G__gcomplevel;
    G__globalcomp=store_globalcomp;
#else
    result = G__loadfile(filename);
#endif
  }
  else {
    /* <xxx.h> , 'xxx.h' */
    store_globalcomp=G__globalcomp;
    /* G__globalcomp=G__NOLINK; */
#ifndef G__OLDIMPLEMENTATION1725
    if(++G__gcomplevel>=G__gcomplevellimit) G__globalcomp=G__NOLINK;
#endif
    result = G__loadfile(filename);
#ifndef G__OLDIMPLEMENTATION1725
    --G__gcomplevel;
#endif
    G__globalcomp=store_globalcomp;
  }
  G__kindofheader = G__USERHEADER;

  G__cpp=store_cpp;

#ifndef G__OLDIMPLEMENTATION1261
  if('#'==c) {
    if(G__LOADFILE_FAILURE==result && G__ispragmainclude) {
      G__ispragmainclude=0;
      c = G__fgetname(filename,"\n\r");
#ifndef G__OLDIMPLEMENTATION1725
      store_globalcomp = G__globalcomp;
      if(++G__gcomplevel>=G__gcomplevellimit) G__globalcomp=G__NOLINK;
      if('\n'!=c && '\r'!=c) result = G__include_file();
#ifndef G__OLDIMPLEMENTATION1725
      --G__gcomplevel;
#endif
      G__globalcomp=store_globalcomp;
#else
      if('\n'!=c && '\r'!=c) result = G__include_file();
#endif
    }
    else {
      G__fignoreline();
    }
  }
#endif

  return(result);
}

/******************************************************************
* G__getmakeinfo()
*
******************************************************************/
char *G__getmakeinfo(item)
char *item;
{
  char makeinfo[G__MAXFILENAME];
  FILE *fp;
  char line[G__LARGEBUF];
  char argbuf[G__LARGEBUF];
  char *arg[G__MAXARG];
  int argn;
  char *p;
  static char buf[G__ONELINE];

  buf[0]='\0';

#ifdef G__NOMAKEINFO
  return("");
#endif

#ifndef G__OLDIMPLEMENTATION466
  /****************************************************************
  * Environment variable overrides MAKEINFO file if exists.
  ****************************************************************/
  if((p=getenv(item)) && p[0] && !isspace(p[0])) {
    strcpy(buf,p);
    return(buf);
  }
#endif

  /****************************************************************
  * Get information from MAKEINFO file.
  ****************************************************************/
  /* Get $CINTSYSDIR/MAKEINFO file name */
  if(G__getcintsysdir()) return(buf);
#ifdef G__VMS
  sprintf(makeinfo,"%sMAKEINFO.txt",G__cintsysdir);
#else
  sprintf(makeinfo,"%s/MAKEINFO",G__cintsysdir);
#endif

  /* Open MAKEINFO file */
  fp = fopen(makeinfo,"r");
  if(!fp) {
    G__fprinterr(G__serr,"Error: cannot open %s\n",makeinfo);
    G__fprinterr(G__serr,
     "!!! There are examples of MAKEINFO files under %s/platform/ !!!\n"
	    ,G__cintsysdir);
    G__fprinterr(G__serr,
	    "Please refer to these examples and create for your platform\n");
    return(buf);
  }

  /* Read the MAKEINFO file */
  while(G__readline(fp,line,argbuf,&argn,arg)) {
    if(argn>2 && strcmp(arg[1],item)==0) {
      p = strchr(arg[0],'=');
      if(p) {
	do {
	  ++p;
	} while(isspace(*p));
	strcpy(buf,p);
	fclose(fp);
	return(buf);
      }
      else {
	G__fprinterr(G__serr,"MAKEINFO syntax error\n");
      }
    }
  }
  fclose(fp);
  return(buf);
}

#ifndef G__OLDIMPLEMENTATION1645
/******************************************************************
* G__getmakeinfo1()
*
******************************************************************/
char *G__getmakeinfo1(item)
char *item;
{
  char *buf = G__getmakeinfo(item);
  char *p = buf;
  while(*p && !isspace(*p)) ++p;
  *p = 0;
  return(buf);
}
#endif

#ifndef G__OLDIMPLEMENTATION1963
/******************************************************************
* G__SetCINTSYSDIR()
*
******************************************************************/
void G__SetCINTSYSDIR(cintsysdir)
char *cintsysdir;
{
  strcpy(G__cintsysdir,cintsysdir);
}
#endif

#ifndef G__OLDIMPLEMENTATION1731
/******************************************************************
 * G__SetUseCINTSYSDIR()
 ******************************************************************/
static int G__UseCINTSYSDIR=0;
void G__SetUseCINTSYSDIR(UseCINTSYSDIR)
int UseCINTSYSDIR;
{
  G__UseCINTSYSDIR=UseCINTSYSDIR;
}
#endif

/******************************************************************
* G__getcintsysdir()
*
*  print out error message for unsupported capability.
******************************************************************/
int G__getcintsysdir()
{
  char *env;
  if('*'==G__cintsysdir[0]) {
#if defined(G__ROOT)
# ifdef ROOTBUILD
    env = "cint";
# else
#  ifdef CINTINCDIR
    env = CINTINCDIR;
#  else
#ifndef G__OLDIMPLEMENTATION1731
    if(G__UseCINTSYSDIR) env=getenv("CINTSYSDIR");
    else                 env=getenv("ROOTSYS");
#else /* 1731 */
    env=getenv("ROOTSYS");
#endif /* 1731 */
#  endif
# endif
#elif defined(G__WILDC)
    env=getenv("WILDCDIR");
    if(!env) env=getenv("CINTSYSDIR");
    if(!env) env="C:\\WILDC";
#else
    env=getenv("CINTSYSDIR");
# ifdef CINTSYSDIR
    if(!env || !env[0]) env = CINTSYSDIR;
# endif
#endif
    if(env) {
#ifdef G__ROOT
#ifdef G__VMS
/*      sprintf(G__cintsysdir,env);
      strcpy(&G__cintsysdir[strlen(G__cintsysdir)-1],".cint]");*/
      sprintf(G__cintsysdir,"%s[cint]",env);
#else /* G__VMS */
# ifdef ROOTBUILD
      sprintf(G__cintsysdir, "%s", env);
# else /* ROOTBUILD */
#  ifdef CINTINCDIR
      sprintf(G__cintsysdir, "%s", CINTINCDIR);
#  else
#ifndef G__OLDIMPLEMENTATION1731
      if(G__UseCINTSYSDIR) strcpy(G__cintsysdir,env);
      else                 sprintf(G__cintsysdir, "%s%scint", env, G__psep);
#else /* 1731 */
      sprintf(G__cintsysdir, "%s%scint", env, G__psep);
#endif /* 1731 */
#  endif
# endif /* ROOTBUILD */
#endif /* G__VMS */

#else /* G__ROOT */
      strcpy(G__cintsysdir,env);
#endif /* G__ROOT */
      return(EXIT_SUCCESS);
    }
    else {
#ifdef G__EDU_VERSION
      sprintf(G__cintsysdir,".");
      return(EXIT_SUCCESS);
#else
#ifndef G__OLDIMPLEMENTATION1314
#ifdef G__WIN32
      HMODULE hmodule=0;
      if(GetModuleFileName(hmodule,G__cintsysdir,G__MAXFILENAME)) {
        char *p = G__strrstr(G__cintsysdir,G__psep);
        if(p) *p = 0;
# ifdef G__ROOT
        p = G__strrstr(G__cintsysdir,G__psep);
        if(p) *p = 0;
	strcat(G__cintsysdir,G__psep);
	strcat(G__cintsysdir,"cint");
# endif
	return(EXIT_SUCCESS);
      }
#endif
#endif
#if defined(G__ROOT)
      G__fprinterr(G__serr,"Warning: environment variable ROOTSYS is not set. Standard include files ignored\n");
#elif defined(G__WILDC)
      G__fprinterr(G__serr,"Warning: environment variable WILDCDIR is not set. Standard include files ignored\n");
#else
      G__fprinterr(G__serr,"Warning: environment variable CINTSYSDIR is not set. Standard include files ignored\n");
#endif
      G__cintsysdir[0]='\0';
      return(EXIT_FAILURE);
#endif
    }
  }
  return(EXIT_SUCCESS);
}

/******************************************************************
* G__isfilebusy()
******************************************************************/
int G__isfilebusy(ifn)
int ifn;
{
  struct G__ifunc_table *ifunc;
  int flag=0;
  int i1;
  int i2;

  /*********************************************************************
  * check global function busy status
  *********************************************************************/
  ifunc = &G__ifunc;
  while(ifunc) {
    for(i1=0;i1<ifunc->allifunc;i1++) {
      if( 0!=ifunc->busy[i1] && ifunc->pentry[i1]->filenum>=ifn ) {
	G__fprinterr(G__serr,"Function %s() busy. loaded after \"%s\"\n"
		,ifunc->funcname[i1],G__srcfile[ifn].filename);
	flag++;
      }
    }
    ifunc=ifunc->next;
  }

  /*********************************************************************
  * check member function busy status
  *********************************************************************/
  if(0==G__nfile || ifn<0 || G__nfile<=ifn ||
     (struct G__dictposition*)NULL==G__srcfile[ifn].dictpos ||
     -1==G__srcfile[ifn].dictpos->tagnum) return(flag);
  for(i2=G__srcfile[ifn].dictpos->tagnum;i2<G__struct.alltag;i2++) {
    ifunc = G__struct.memfunc[i2];
    while(ifunc) {
      for(i1=0;i1<ifunc->allifunc;i1++) {
	if(0!=ifunc->busy[i1]&&ifunc->pentry[i1]->filenum>=ifn) {
	  G__fprinterr(G__serr,"Function %s() busy. loaded after\"%s\"\n"
		  ,ifunc->funcname[i1],G__srcfile[ifn].filename);
	  flag++;
	}
      }
      ifunc=ifunc->next;
    }
  }

  return(flag);
}

#ifndef G__OLDIMPLEMENTATION1196
/******************************************************************
* G__matchfilename(i,filename)
******************************************************************/
int G__matchfilename(i1,filename)
int i1;
char* filename;
{
#if !defined(G__PHILIPPE28) && !defined(__CINT__)

#ifdef G__WIN32
  char i1name[_MAX_PATH],fullfile[_MAX_PATH];
#else 
  struct stat statBufItem;  
  struct stat statBuf;  
#endif

  if((strcmp(G__srcfile[i1].filename,filename)==0)) return(1);

#ifdef G__WIN32
  _fullpath( i1name, G__srcfile[i1].filename, _MAX_PATH ); 
  _fullpath( fullfile, filename, _MAX_PATH );
  if((stricmp(i1name, fullfile)==0)) return 1;
#else
  if (   ( 0 == stat( filename, & statBufItem ) )
      && ( 0 == stat( G__srcfile[i1].filename, & statBuf ) ) 
      && ( statBufItem.st_ino == statBuf.st_ino ) ) {
     return 1;
  }
#endif
  return 0;

#else /* PHILIPPE28 */

  char *filenamebase;
  if((strcmp(G__srcfile[i1].filename,filename)==0)) return(1);
  filenamebase = G__strrstr(G__srcfile[i1].filename,"./");
  if(filenamebase) {
    char *parentdir = G__strrstr(G__srcfile[i1].filename,"../");
    if(!parentdir && strcmp(filename,filenamebase+2)==0) {
      char buf[G__ONELINE];
#if defined(G__WIN32)
      char *p;
#endif
      if(filenamebase==G__srcfile[i1].filename) return(1);
#if defined(G__WIN32)
      GetCurrentDirectory(G__ONELINE,buf);
      p=buf;
      while((p=strchr(p,'\\'))) *p='/';
      if(strlen(buf)>1 && ':'==buf[1]) {
	char buf2[G__ONELINE];
	strcpy(buf2,buf+2);
	strcpy(buf,buf2);
      }
#elif defined(G__POSIX) || defined(G__ROOT)
      getcwd(buf,G__ONELINE);
#else
      buf[0] = 0;
#endif
      if(strncmp(buf,G__srcfile[i1].filename
		 ,filenamebase-G__srcfile[i1].filename-1)==0) return(1);
    }
  }
  return(0);
#endif /* PHILIPPE28 */
}
#endif

/******************************************************************
* G__stripfilename(filename)
******************************************************************/
char* G__stripfilename(filename)
char* filename;
{
  char *filenamebase;
  if(!filename) return("");
  filenamebase = G__strrstr(filename,"./");
  if(filenamebase) {
    char *parentdir = G__strrstr(filename,"../");
    char buf[G__ONELINE];
#if defined(G__WIN32)
    char *p;
#endif
    if(parentdir) return(filename);
    if(filenamebase==filename) return(filenamebase+2);
#if defined(G__WIN32)
    GetCurrentDirectory(G__ONELINE,buf);
    p=buf;
    while((p=strchr(p,'\\'))) *p='/';
    if(strlen(buf)>1 && ':'==buf[1]) {
      char buf2[G__ONELINE];
      strcpy(buf2,buf+2);
      strcpy(buf,buf2);
    }
#elif defined(G__POSIX) || defined(G__ROOT)
    getcwd(buf,G__ONELINE);
#else
    buf[0] = 0;
#endif
    if(strncmp(buf,filename,filenamebase-filename-1)==0) 
      return(filenamebase+2);
    else 
      return(filename);
  }
  else return(filename);
}

#ifndef G__OLDIMPLEMENTATION1273
/******************************************************************
* G__smart_unload()
******************************************************************/
void G__smart_unload(ifn)
int ifn;
{
  struct G__dictposition *dictpos= G__srcfile[ifn].dictpos;
  struct G__dictposition *hasonlyfunc = G__srcfile[ifn].hasonlyfunc;
  struct G__ifunc_table *ifunc;
  struct G__var_array *var;
  int nfile;
  int allsl;

  if(G__nfile == hasonlyfunc->nfile) {
    var = &G__global;
    while(var->next) var=var->next;
    if(var == hasonlyfunc->var && var->allvar == hasonlyfunc->ig15) {
      G__scratch_upto(G__srcfile[ifn].dictpos);
      return;
    }
  }

  /* disable functions */
  ifunc = dictpos->ifunc;
  ifn = dictpos->ifn;
  while(ifunc && (ifunc!=hasonlyfunc->ifunc || ifn!=hasonlyfunc->ifn)) {
    ifunc->hash[ifn] = 0;
#ifndef G__OLDIMPLEMENTATION1706
    ifunc->funcname[ifn][0] = 0;
#endif
    if(++ifn>=G__MAXIFUNC) {
      ifunc = ifunc->next;
      ifn = 0;
    }
  }

  /* disable file entry */
  for(nfile=dictpos->nfile;nfile<hasonlyfunc->nfile;nfile++) {
    G__srcfile[nfile].hash = 0;
    G__srcfile[nfile].filename[0] = 0;
    if(G__srcfile[nfile].fp) fclose(G__srcfile[nfile].fp);
    G__srcfile[nfile].fp = 0;
  }

  /* unload shared library */
  for(allsl=dictpos->allsl;allsl<hasonlyfunc->allsl;allsl++) {
    G__smart_shl_unload(allsl);
  }
}
#endif


/******************************************************************
* G__unloadfile(filename)
*
*  1) check if function is busy. If busy return -1
*  2) Unload file and return 0
*
******************************************************************/
int G__unloadfile(filename)
char *filename;
{
  int ifn;
  int i1=0;
  int i2;
  int hash;
  /* int from = -1 ,to = -1, next; */
  int flag;
#ifndef G__OLDIMPLEMENTATION1765
  char buf[G__MAXFILENAME];
  char *fname;
  char *scope;
  int envtagnum;
#endif

#ifndef G__OLDIMPLEMENTATION1345
  G__LockCriticalSection();
#endif

#ifndef G__OLDIMPLEMENTATION1765
  strcpy(buf,filename);
  fname = G__strrstr(buf,"::");
  if(fname) {
    scope = buf;
    *fname = 0;
    fname+=2;
    if(0==scope[0]) envtagnum = -1;
    else {
      envtagnum = G__defined_tagname(scope,2);
      if(-1==envtagnum) {
	G__fprinterr(G__serr,"Error: G__unloadfile() File \"%s\" scope not found ",scope);
	G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION1345
	G__UnlockCriticalSection();
#endif
	return(G__UNLOADFILE_FAILURE);
      }
    }
  }
  else {
    fname = filename;
    envtagnum = -1;
  }
#endif

  /******************************************************************
  * check if file is already loaded.
  * if not so, return
  ******************************************************************/
#ifndef G__OLDIMPLEMENTATION1765
  G__hash(fname,hash,i2);
#else
  G__hash(filename,hash,i2);
#endif

  flag=0;
  while(i1<G__nfile) {
#ifndef G__OLDIMPLEMENTATION1196
#ifndef G__OLDIMPLEMENTATION1765
    if(G__matchfilename(i1,fname)
       && (-1==envtagnum||(envtagnum==G__srcfile[i1].parent_tagnum))){
#else
    if(G__matchfilename(i1,filename)) {
#endif
#else
    if((G__srcfile[i1].hash==hash&&strcmp(G__srcfile[i1].filename,filename)==0)
       ){
#endif
      flag=1;
      break;
    }
    i1++;
  }

  if(flag==0) {
    G__fprinterr(G__serr,"Error: G__unloadfile() File \"%s\" not loaded ",filename);
    G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION1345
    G__UnlockCriticalSection();
#endif
    return(G__UNLOADFILE_FAILURE);
  }

  /*******************************************************
  * set G__ifile index number to ifn.
  *******************************************************/
  ifn = i1;


  /*********************************************************************
  * if function in unloaded files are busy, cancel unloading
  *********************************************************************/
  if(G__isfilebusy(ifn)) {
    G__fprinterr(G__serr,
  "Error: G__unloadfile() Can not unload \"%s\", file busy " ,filename);
    G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION1345
    G__UnlockCriticalSection();
#endif
    return(G__UNLOADFILE_FAILURE);
  }

#ifndef G__OLDIMPLEMENTATION1273
  if(G__srcfile[ifn].hasonlyfunc && G__do_smart_unload) {
    G__smart_unload(ifn);
  }
  else {
    G__scratch_upto(G__srcfile[ifn].dictpos);
  }
#else
  G__scratch_upto(G__srcfile[ifn].dictpos);
#endif

  if(G__debug) {
    G__fprinterr(G__serr,"File=%s unloaded\n",filename);
  }

#ifndef G__OLDIMPLEMENTATION1345
  G__UnlockCriticalSection();
#endif
  return(G__UNLOADFILE_SUCCESS);

}

/******************************************************************
* G__isbinaryfile()
*
******************************************************************/
static int G__isbinaryfile(filename)
char *filename;
{
  int c;
  int prev=0;
  int i;
  int badflag=0;
  int comflag=0;
#ifdef G__VISUAL /* ON959 */
  char buf[11];  
#endif
#ifndef G__OLDIMPLEMENTATION1217
  int unnamedmacro = 0;
  int alphaflag=0;
#endif
#ifndef G__OLDIMPLEMENTATION1480
  int store_lang = G__lang;
#endif

#ifndef G__OLDIMPLEMENTATION1480
  if(G__ONEBYTE!=G__lang) G__lang = G__UNKNOWNCODING;
#else
#ifndef G__OLDIMPLEMENTATION1344
  G__lang = G__UNKNOWNCODING;
#endif
#endif

  /* Read 10 byte from beginning of the file.
   * Set badflag if unprintable char is found. */
  for(i=0;i<10;i++) {
    c=fgetc(G__ifile.fp);
#ifndef G__OLDIMPLEMENTATION1344
    if(G__IsDBCSLeadByte(c)) {
      c=fgetc(G__ifile.fp);
      if(c!=EOF) G__CheckDBCS2ndByte(c);
    } else
#endif
    if(!isprint(c) && '\t'!=c && '\n'!=c && '\r'!=c && EOF!=c && 0==comflag) {
      ++badflag;
    }
    else if('/'==prev && ('/'==c||'*'==c)) {
      comflag=1; /* set comment flag */
    }
#ifndef G__OLDIMPLEMENTATION1217
    else if('{'==c && 0==alphaflag && 0==comflag) {
      unnamedmacro=1;
    }
    else if(isalpha(c)) {
      ++alphaflag;
    }
#endif
    prev = c;
    if(EOF==c) break;
#ifdef G__VISUAL /* ON959 */
    buf[i] = c;
#endif
  }


  if(badflag) {
    G__fprinterr(G__serr,"Error: Bad source file(binary) %s",filename);
    G__genericerror((char*)NULL);
    G__return=G__RETURN_EXIT1;
#ifndef G__OLDIMPLEMENTATION1480
    G__lang = store_lang;
#endif
    return(1);
  }
#ifndef G__OLDIMPLEMENTATION1217
  else if(unnamedmacro) {
    G__fprinterr(G__serr,"Error: Bad source file(unnamed macro) %s",filename);
    G__genericerror((char*)NULL);
    G__fprinterr(G__serr,"  unnamed macro has to be executed by 'x' command\n");
    G__return=G__RETURN_EXIT1;
#ifndef G__OLDIMPLEMENTATION1480
    G__lang = store_lang;
#endif
    return(1);
  }
#endif
  else {
#ifdef G__VISUAL /* ON959 */
    buf[10] =0;
    if(strncmp(buf,"Microsoft ",10)==0) {
      /* Skip following compiler message
Microsoft (R) 32-bit C/C++ Optimizing Compiler Version 11.00.7022 for 80x86
Copyright (C) Microsoft Corp 1984-1997. All rights reserved.

2.c
       */
      G__fignoreline(); c=G__fgetc(); /* \r\n */
      G__fignoreline(); c=G__fgetc(); /* \r\n */
      G__fignoreline(); c=G__fgetc(); /* \r\n */
      G__fignoreline(); c=G__fgetc(); /* \r\n */
    }
    else {
      fseek(G__ifile.fp,SEEK_SET,0);
    }
#else
    fseek(G__ifile.fp,SEEK_SET,0);
#endif
  }
#ifndef G__OLDIMPLEMENTATION1480
  G__lang = store_lang;
#endif
  return(0);
}

#ifndef G__OLDIMPLEMENTATION1273
/******************************************************************
*  G__checkIfOnlyFunction()
******************************************************************/
static void G__checkIfOnlyFunction(fentry)
int fentry;
{
  struct G__var_array *var;
  struct G__Deffuncmacro *deffuncmacro;
  struct G__Definedtemplateclass *definedtemplateclass;
  struct G__Definetemplatefunc *definedtemplatefunc;   
  struct G__dictposition* dictpos = G__srcfile[fentry].dictpos;
  int varflag = 1;
#ifndef G__OLDIMPLEMENTATION2014
  int tagflag ;

  if(dictpos->tagnum == G__struct.alltag) {
    tagflag = 1;
    if(dictpos->ptype && (char*)G__PVOID!=dictpos->ptype) {
      int i;
      for(i=0; i<G__struct.alltag; i++) {
	if(dictpos->ptype[i]!=G__struct.type[i]) {
	  tagflag=0;
	  break;
	}
      }
    }
  }
  else {
    tagflag = 0;
  }
#endif

  var = &G__global;
  while(var->next) var=var->next;

  if(dictpos->var == var && dictpos->ig15 == var->allvar) {
    varflag = 1;
  }
  else {
    struct G__var_array *var2 = dictpos->var;
    int ig152 = dictpos->ig15;
    while(var2 && (var2 != var || ig152 != var->allvar)) {
      if('p'!=var2->type[ig152]) { 
	varflag = 0;
	break;
      }
      if(++ig152>=G__MEMDEPTH) {
	var2 = var2->next;
	ig152=0;
      }
    }
  }

  deffuncmacro = &G__deffuncmacro;
  while(deffuncmacro->next) deffuncmacro=deffuncmacro->next;

  definedtemplateclass = &G__definedtemplateclass;
  while(definedtemplateclass->next)
    definedtemplateclass=definedtemplateclass->next;

  definedtemplatefunc = &G__definedtemplatefunc;
  while(definedtemplatefunc->next)
    definedtemplatefunc=definedtemplatefunc->next;

  if(
#ifndef G__OLDIMPLEMENTATION2014
     tagflag &&
#else
     dictpos->tagnum == G__struct.alltag &&
#endif
     dictpos->typenum == G__newtype.alltype &&
     varflag &&
     dictpos->deffuncmacro == deffuncmacro &&
     dictpos->definedtemplateclass == definedtemplateclass &&
     dictpos->definedtemplatefunc == definedtemplatefunc) {
    G__srcfile[fentry].hasonlyfunc = 
      (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
#ifndef G__OLDIMPLEMENTATION2014
    G__srcfile[fentry].hasonlyfunc->ptype = (char*)G__PVOID;
#endif
    G__store_dictposition(G__srcfile[fentry].hasonlyfunc);
  }
}
#endif

#ifndef G__OLDIMPLEMENTATION1919
/******************************************************************
* G__loadfile_tmpfile(fp)
*
******************************************************************/
int G__loadfile_tmpfile(fp)
FILE *fp;
{
  int store_prerun;
  struct G__var_array *store_p_local;
  int store_var_type,store_tagnum,store_typenum;
  int fentry;
  int store_nobreak;
  int store_step;
  struct G__input_file store_file;
  int store_macroORtemplateINfile;
  short store_iscpp;
  G__UINT32 store_security;
  int store_func_now;
  int pragmacompile_iscpp;
  int pragmacompile_filenum;
  int store_asm_noverflow;
  int store_no_exec_compile;
  int store_asm_exec;
  int store_return;
  long store_struct_offset;
  int hash,temp;
#ifndef G__OLDIMPLEMENTATION1536
  char hdrprop = G__NONCINTHDR;
#endif

  /******************************************************************
  * check if number of loaded file exceeds G__MAXFILE
  * if so, restore G__ifile reset G__eof and return.
  ******************************************************************/
  if(G__nfile==G__MAXFILE) {
    G__fprinterr(G__serr,"Limitation: Sorry, can not load any more files\n");
    return(G__LOADFILE_FATAL);
  }

  if(!fp) {
    G__genericerror("Internal error: G__loadfile_tmpfile((FILE*)NULL)");
    return(G__LOADFILE_FATAL);
  }

#ifndef G__OLDIMPLEMENTATION1345
  G__LockCriticalSection();
#endif

  /*************************************************
  * store current input file information
  *************************************************/
  store_file = G__ifile;
  store_step = G__step;
  G__step=0;
  G__setdebugcond();

  /* pre run , read whole ifuncs to allocate global variables and
     make ifunc table */

  /**********************************************
  * store iscpp (is C++) flag. This flag is modified in G__preprocessor()
  * function and restored in G__loadfile() before return.
  **********************************************/
  store_iscpp=G__iscpp;

  /**********************************************
  * filenum and line_number.
  **********************************************/
  G__ifile.line_number = 1;
  G__ifile.fp = fp;
  G__ifile.filenum = G__nfile ;
  fentry = G__nfile;
  /* strcpy(G__ifile.name,"(tmpfile)"); */
  sprintf(G__ifile.name,"(tmp%d)",fentry);
  G__hash(G__ifile.name,hash,temp);

  G__srcfile[fentry].dictpos
    = (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
#ifndef G__OLDIMPLEMENTATION2014
  G__srcfile[fentry].dictpos->ptype = (char*)NULL;
#endif
  G__store_dictposition(G__srcfile[fentry].dictpos);

  G__srcfile[fentry].hdrprop = hdrprop;

  store_security = G__security;
  G__srcfile[fentry].security = G__security;

  G__srcfile[fentry].prepname = (char*)NULL;
  G__srcfile[fentry].hash = hash;
  G__srcfile[fentry].filename = (char*)malloc(strlen(G__ifile.name)+1);
  strcpy(G__srcfile[fentry].filename,G__ifile.name);
  G__srcfile[fentry].fp=G__ifile.fp;

  G__srcfile[fentry].included_from = store_file.filenum;

  G__srcfile[fentry].ispermanentsl = G__ispermanentsl;
  G__srcfile[fentry].initsl = (G__DLLINIT)NULL;
  G__srcfile[fentry].hasonlyfunc = (struct G__dictposition*)NULL;
  G__srcfile[fentry].parent_tagnum = G__get_envtagnum();
  G__srcfile[fentry].slindex = -1;

  ++G__nfile;

  if(G__debugtrace) {
    G__fprinterr(G__serr,"LOADING tmpfile\n");
  }
  if(G__debug) {
    G__fprinterr(G__serr,"%-5d",G__ifile.line_number);
  }

  /******************************************************
   * store parser parameters
   ******************************************************/
  store_prerun=G__prerun;
  store_p_local=G__p_local;
  if(0==G__def_struct_member||-1==G__tagdefining||
     ('n'!=G__struct.type[G__tagdefining]
      && 'c'!=G__struct.type[G__tagdefining]
      && 's'!=G__struct.type[G__tagdefining]
     )) {
    G__p_local=NULL;
  }

  G__eof = 0;
  G__prerun = 1;
  G__switch = 0;
  G__mparen = 0;
  store_nobreak=G__nobreak;
  G__nobreak=1;

  store_var_type = G__var_type;
  store_tagnum = G__tagnum;
  store_typenum = G__typenum;
  store_func_now = G__func_now;
  G__func_now = -1;
  store_macroORtemplateINfile = G__macroORtemplateINfile;
  G__macroORtemplateINfile = 0;
  store_asm_noverflow = G__asm_noverflow;
  store_no_exec_compile = G__no_exec_compile;
  store_asm_exec = G__asm_exec;
  G__asm_noverflow = 0;
  G__no_exec_compile = 0;
  G__asm_exec = 0;
  store_return=G__return;
  G__return=G__RETURN_NON;

  store_struct_offset = G__store_struct_offset;
  G__store_struct_offset = 0;

  /******************************************************
   * read source file
   ******************************************************/
  while (!G__eof && G__return<G__RETURN_EXIT1) G__exec_statement();


  /******************************************************
   * restore parser parameters
   ******************************************************/
  G__store_struct_offset = store_struct_offset;
#ifndef G__OLDIMPLEMENTATION487
  pragmacompile_filenum = G__ifile.filenum;
  pragmacompile_iscpp = G__iscpp;
  G__func_now = store_func_now;
#endif
  G__macroORtemplateINfile = store_macroORtemplateINfile;
  G__var_type = store_var_type;
  G__tagnum = store_tagnum;
  G__typenum = store_typenum;

  G__nobreak=store_nobreak;
  G__prerun=store_prerun;
  G__p_local=store_p_local;

#ifndef G__OLDIMPLEMENTATION974
  G__asm_noverflow = store_asm_noverflow;
  G__no_exec_compile = store_no_exec_compile;
  G__asm_exec = store_asm_exec;
#endif

  /******************************************************
   * restore input file information to G__ifile
   * and reset G__eof to 0.
   ******************************************************/
  G__ifile = store_file ;
  G__eof = 0;
  G__step=store_step;
  G__setdebugcond();
  G__globalcomp=G__store_globalcomp;
  G__iscpp=store_iscpp;
#ifdef G__SECURITY
  G__security = store_security;
#endif
  if(G__return>G__RETURN_NORMAL) {
#ifndef G__OLDIMPLEMENTATION1345
    G__UnlockCriticalSection();
#endif
#ifndef G__OLDIMPLEMENTATION1849
    G__return=store_return;
#endif
    return(G__LOADFILE_FAILURE);
  }

#ifndef G__OLDIMPLEMENTATION1849
  G__return=store_return;
#endif

#ifndef G__OLDIMPLEMENTATION487
#ifdef G__AUTOCOMPILE
  /*************************************************************
   * if '#pragma compile' appears in source code.
   *************************************************************/
  if(G__fpautocc && G__autoccfilenum == pragmacompile_filenum) {
    store_iscpp = G__iscpp;
    G__iscpp=pragmacompile_iscpp;
    G__autocc();
    G__iscpp = store_iscpp;
  }
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1273
  G__checkIfOnlyFunction(fentry);
#endif

#ifndef G__OLDIMPLEMENTATION1345
  G__UnlockCriticalSection();
#endif
#ifndef G__OLDIMPLEMENTATION1920
  return(fentry+2);
#else
  return(G__LOADFILE_SUCCESS);
#endif
}
#endif

/******************************************************************
* G__loadfile(filename)
*
*  0) If .sl .dl .so .dll .DLL call G__shl_load()
*  1) check G__MAXFILE                       return -2 if fail(fatal)
*  2) check if file is already loaded        return 1 if already loaded
*  3) Open filename
*  4) If fp==NULL, search include path
*  5) Set filename and linenumber
*  6) If still fp==NULL                      return -1
*  7) LOAD filename
*  8) If G__return>G__RETURN_NORMAL          return -1
*  9)                                        return 0
*
******************************************************************/
int G__loadfile(filenamein)
char *filenamein;
{
#ifndef G__PHILIPPE0
  FILE *tmpfp;
#endif
#ifndef G__PHILIPPE1
  int external_compiler = 0;
  char* compiler_option = "";
#endif
  int store_prerun;
  int i1=0;
  struct G__var_array *store_p_local;
  int store_var_type,store_tagnum,store_typenum;
  int fentry;
  struct G__includepath *ipath;
  int store_nobreak;
#ifdef G__TMPFILE
  char prepname[G__MAXFILENAME];
#else
  char prepname[L_tmpnam+10];
#endif
  int store_step;
  int null_entry = -1;
  struct G__input_file store_file;
  int hash;
  int temp;
  int store_macroORtemplateINfile;
  int len;
#ifndef G__OLDIMPLEMENTATION1705
  int len1;
  char *dllpost;
#endif
  short store_iscpp;
  G__UINT32 store_security;
#ifndef G__OLDIMPLEMENTATION460
  char addpost[3][8];
  int i2;
#endif
#ifndef G__OLDIMPLEMENTATION487
  int store_func_now;
  int pragmacompile_iscpp;
  int pragmacompile_filenum;
#endif
#ifndef G__OLDIMPLEMENTATION974
  int store_asm_noverflow;
  int store_no_exec_compile;
  int store_asm_exec;
#endif
#if defined(R__FBSD) || defined(R__OBSD)
  char soext[]=SOEXT;
#endif
#ifndef G__OLDIMPLEMENTATION1849
  int store_return;
#endif
#ifndef G__OLDIMPLEMENTATION1536
  char hdrprop = G__NONCINTHDR;
#endif
  char filename[G__ONELINE];
  strcpy(filename,filenamein);


#ifndef G__OLDIMPLEMENTATION464
  /*************************************************
  * delete space chars at the end of filename
  *************************************************/
  len = strlen(filename);
  while(len>1&&isspace(filename[len-1])) {
    filename[--len]='\0';
  }
#endif

#ifndef G__PHILIPPE1
  /*************************************************
  * Check if the filename as an extension ending in
  * ++, like script.cxx++ or script.C++
  * ending with only one + means to keep the shared
  * library after the end of this process.
  * The + or ++ can also be followed by either a 'g'
  * or an 'O' which means respectively to compile
  * in debug or optimized mode.
  *************************************************/  
#ifndef G__OLDIMPLEMENTATION1734
  compiler_option = 0;
  if ( len>2 && (strncmp(filename+len-2,"+",1)==0 )
       && (strcmp(filename+len-1,"O")==0
           || strcmp(filename+len-1,"g")==0 )
     ) {
     compiler_option = filename+len-1;
     len -= 1;
  }
  if ( len>1 && (strncmp(filename+len-1,"+",1)==0 ) ) {
    if ( len>2 && (strncmp(filename+len-2,"++",2)==0 ) ) {
#ifndef G__OLDIMPLEMENTATION1303
      if (compiler_option) {
         switch(compiler_option[0]) {
            case 'O': compiler_option = "kfO"; break;
            case 'g': compiler_option = "kfg"; break;
            default: G__genericerror("Should not have been reached!");
         }
      } else {
         compiler_option = "kf";
      }
#endif 
      len -= 2;
    } else {
      if (compiler_option) {
         switch(compiler_option[0]) {
            case 'O': compiler_option = "kO"; break;
            case 'g': compiler_option = "kg"; break;
            default: G__genericerror("Should not have been reached!");
         }
      } else {
         compiler_option = "k";
      }
      len -= 1;
    } 
    
    filename[len]='\0';
    external_compiler = 1; /* Request external compilation
			    * if available (in ROOT) */
    if (G__ScriptCompiler!=0) {
      if ( (*G__ScriptCompiler)(filename,compiler_option) )
	return(G__LOADFILE_SUCCESS);
      else
	return(G__LOADFILE_FAILURE);
    }
  }
#else /* 1734 */
  if ( len>1&& (strcmp(filename+len-1,"+")==0 ) ) {
    if (len>2 && (strcmp(filename+len-2,"++")==0 ) ) {
#ifndef G__OLDIMPLEMENTATION1303
      compiler_option = "kf";
#endif
      len -= 2;
    } else {
      compiler_option = "k";
      len -= 1;
    } 
    filename[len]='\0';
    external_compiler = 1; /* Request external compilation
			    * if available (in ROOT) */
    if (G__ScriptCompiler!=0) {
      if ( (*G__ScriptCompiler)(filename,compiler_option) )
	return(G__LOADFILE_SUCCESS);
      else
	return(G__LOADFILE_FAILURE);
    }
  }
#endif /* 1734 */
#endif /* PHIL1 */

#ifndef G__OLDIMPLEMENTATION1345
  G__LockCriticalSection();
#endif

  /*************************************************
  * store current input file information
  *************************************************/
  store_file = G__ifile;
  store_step = G__step;
  G__step=0;
  G__setdebugcond();

  /* pre run , read whole ifuncs to allocate global variables and
     make ifunc table */


  /******************************************************************
  * check if number of loaded file exceeds G__MAXFILE
  * if so, restore G__ifile reset G__eof and return.
  ******************************************************************/
  if(G__nfile==G__MAXFILE) {
    G__fprinterr(G__serr,"Limitation: Sorry, can not load any more files\n");
    G__ifile = store_file ;
    G__eof = 0;
    G__step=store_step;
    G__setdebugcond();
#ifndef G__OLDIMPLEMENTATION1345
    G__UnlockCriticalSection();
#endif
    return(G__LOADFILE_FATAL);
  }


  G__hash(filename,hash,temp);

  /******************************************************************
  * check if file is already loaded.
  * if so, restore G__ifile reset G__eof and return.
  ******************************************************************/
  while(i1<G__nfile) {
    /***************************************************
     * This entry was unloaded by G__unloadfile()
     * Then remember the entry index into 'null_entry'.
     ***************************************************/
    if((char*)NULL==G__srcfile[i1].filename) {
      if(null_entry == -1) {
	null_entry = i1;
      }
    }
    /***************************************************
     * check if alreay loaded
     ***************************************************/
#ifndef G__OLDIMPLEMENTATION1196
    if(G__matchfilename(i1,filename)
#ifndef G__OLDIMPLEMENTATION1756
       &&G__get_envtagnum()==G__srcfile[i1].parent_tagnum
#endif
       ){
#else
    if(hash==G__srcfile[i1].hash&&strcmp(G__srcfile[i1].filename,filename)==0
       ){
#endif
      if(G__prerun==0 || G__debugtrace)
	if(G__dispmsg>=G__DISPNOTE) {
	  G__fprinterr(G__serr,"Note: File \"%s\" already loaded\n",filename);
	}
      /******************************************************
       * restore input file information to G__ifile
       * and reset G__eof to 0.
       ******************************************************/
      G__ifile = store_file ;
      G__eof = 0;
      G__step=store_step;
      G__setdebugcond();
#ifndef G__OLDIMPLEMENTATION1345
      G__UnlockCriticalSection();
#endif
      return(G__LOADFILE_DUPLICATE);
    }
    else {
      ++i1;
    }
  }

  /**********************************************
  * Get actual open file name.
  **********************************************/
  if(!G__cpp) G__cpp = G__ispreprocessfilekey(filename);

  /**********************************************
  * store iscpp (is C++) flag. This flag is modified in G__preprocessor()
  * function and restored in G__loadfile() before return.
  **********************************************/
  store_iscpp=G__iscpp;
  /**********************************************
  * Get actual open file name.
  **********************************************/
  G__preprocessor(prepname,filename,G__cpp,G__macros,G__undeflist
		  ,G__ppopt,G__allincludepath);

  /**********************************************
  * open file
  **********************************************/
  if(prepname[0]) {
    /**********************************************
     * -p option. open preprocessed tmpfile
     **********************************************/
    sprintf(G__ifile.name,"%s",filename);
#ifndef G__OLDIMPLEMENTATION1920
#ifndef G__OLDIMPLEMENTATION1922
    if(G__fons_comment && G__cpp && G__NOLINK!=G__globalcomp) {
#ifndef G__WIN32
      G__ifile.fp = fopen(prepname,"r");
#else
      G__ifile.fp = fopen(prepname,"rb");
#endif
    }
    else {
      G__ifile.fp = G__copytotmpfile(prepname);
#ifndef G__OLDIMPLEMENTATION2092
      if(G__ifile.fp) {
        remove(prepname);
        strcpy(prepname,"(tmpfile)");
      }
      else {
#ifndef G__WIN32
        G__ifile.fp = fopen(prepname,"r");
#else /* G__WIN32 */
        G__ifile.fp = fopen(prepname,"rb");
#endif /* G__WIN32 */
      }
#else /* 2092 */
      remove(prepname);
      strcpy(prepname,"(tmpfile)");
#endif /* 2092 */
    }
#else /* 1922 */
    G__ifile.fp = G__copytotmpfile(prepname);
    remove(prepname);
    strcpy(prepname,"(tmpfile)");
#endif /* 1922 */
#else /* 1920 */
#ifndef G__WIN32
    G__ifile.fp = fopen(prepname,"r");
#else /* G__WIN32 */
    G__ifile.fp = fopen(prepname,"rb");
#endif /* G__WIN32 */
#endif /* 1920 */
    G__kindofheader = G__USERHEADER;
  }
  else {
    strcpy(addpost[0],"");
    strcpy(addpost[1],".h");

#ifndef G__OLDIMPLEMENTATION800
    strcpy(addpost[2],"");
    for(i2=0;i2<3;i2++) {
      if(2==i2) {
	if((len>3&& (strcmp(filename+len-3,".sl")==0 ||
		     strcmp(filename+len-3,".dl")==0 ||
		     strcmp(filename+len-3,".so")==0))) {
#ifndef G__OLDIMPLEMENTATION1645
	  strcpy(filename+len-3,G__getmakeinfo1("DLLPOST"));
#else
	  strcpy(filename+len-3,G__getmakeinfo("DLLPOST"));
#endif
	}
	else if((len>4&& (strcmp(filename+len-4,".dll")==0 ||
			  strcmp(filename+len-4,".DLL")==0))) {
#ifndef G__OLDIMPLEMENTATION1645
	  strcpy(filename+len-4,G__getmakeinfo1("DLLPOST"));
#else
	  strcpy(filename+len-4,G__getmakeinfo("DLLPOST"));
#endif
	}
	else if((len>2&& (strcmp(filename+len-2,".a")==0 ||
			  strcmp(filename+len-2,".A")==0))) {
#ifndef G__OLDIMPLEMENTATION1645
	  strcpy(filename+len-2,G__getmakeinfo1("DLLPOST"));
#else
	  strcpy(filename+len-2,G__getmakeinfo("DLLPOST"));
#endif
	}
#if defined(R__FBSD) || defined(R__OBSD)
	else if (len>strlen(soext) &&
		 strcmp(filename+len-strlen(soext),soext)==0) {
#ifndef G__OLDIMPLEMENTATION1645
	  strcpy(filename+len-strlen(soext),G__getmakeinfo1("DLLPOST"));
#else
	  strcpy(filename+len-strlen(soext),G__getmakeinfo("DLLPOST"));
#endif
	}
#endif
      }
#else
    for(i2=0;i2<2;i2++) {
#endif

#ifndef G__OLDIMPLEMENTATION794
      G__ifile.fp = NULL;
      /**********************************************
       * If it's a "" header with a relative path, first
       * try relative to the current input file.
       * (This corresponds to the behavior of gcc.)
       **********************************************/
      if (G__USERHEADER == G__kindofheader &&
#ifdef G__WIN32
          filename[0] != '/' &&
          filename[0] != '\\' &&
#else
          filename[0] != G__psep[0] &&
#endif
          store_file.name[0] != '\0') {
        char* p;
        strcpy (G__ifile.name, store_file.name);
#ifdef G__WIN32
        p = strrchr (G__ifile.name, '/');
        {
          char* q = strrchr (G__ifile.name, '\\');
          if (q && q > p)
            p = q;
        }
#else
        p = strrchr (G__ifile.name, G__psep[0]);
#endif
        if (p == 0) p = G__ifile.name;
        else ++p;
        strcpy (p, filename);
        strcat (p, addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
      }
      if (G__ifile.fp) break;
#endif
      /**********************************************
       * try ./filename
       **********************************************/
      if(G__USERHEADER==G__kindofheader) {
#ifdef G__VMS
        sprintf(G__ifile.name,"%s",filename);
#else
	sprintf(G__ifile.name,"%s%s",filename,addpost[i2]);
#endif
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
      }
      else {
	G__ifile.fp=NULL;
	G__kindofheader = G__USERHEADER;
      }
      if(G__ifile.fp) break;

      /**********************************************
       * try includepath/filename
       **********************************************/
      ipath = &G__ipathentry;
      while(G__ifile.fp==NULL && ipath->pathname) {
	sprintf(G__ifile.name,"%s%s%s%s"
		,ipath->pathname,G__psep,filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
	ipath = ipath->next;
#ifndef G__OLDIMPLEMENTATION1451
	{
	  struct G__ConstStringList* sysdir = G__SystemIncludeDir;
	  while(sysdir) {
	    if(strncmp(sysdir->string,G__ifile.name,sysdir->hash)==0) {
	      G__globalcomp=G__NOLINK;
#ifndef G__OLDIMPLEMENTATION1536
	      hdrprop = G__CINTHDR;
#endif
	    }
	    sysdir = sysdir->prev;
	  }
	}
#endif
      }
      if(G__ifile.fp) break;

      /**********************************************
       * try $CINTSYSDIR/include/filename
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"%s%sinclude%s%s%s",G__cintsysdir,G__psep
		,G__psep,filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
#ifndef G__OLDIMPLEMENTATION1271
	if(G__ifile.fp && G__autoload_stdheader) {
	  G__globalcomp=G__store_globalcomp;
	  G__gen_linksystem(filename);
	}
#endif
#ifndef G__OLDIMPLEMENTATION1536
	hdrprop = G__CINTHDR;
#endif
	G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;

      /**********************************************
       * try $CINTSYSDIR/stl
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"%s%sstl%s%s%s",G__cintsysdir,G__psep
		,G__psep,filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
#ifndef G__OLDIMPLEMENTATION1271
	if(G__ifile.fp && G__autoload_stdheader) {
	  G__globalcomp=G__store_globalcomp;
	  G__gen_linksystem(filename);
	}
#endif
#ifndef G__OLDIMPLEMENTATION1536
	hdrprop = G__CINTHDR;
#endif
	G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;

#ifndef G__OLDIMPLEMENTATION1041
      /**********************************************
       * try $CINTSYSDIR/lib
       **********************************************/
      /* G__getcintsysdir(); */
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"%s%slib%s%s%s",G__cintsysdir,G__psep
		,G__psep,filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
	/* G__globalcomp=G__NOLINK; */
      }
      if(G__ifile.fp) break;
#endif

#ifdef G__EDU_VERSION
      /**********************************************
       * try include/filename
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"include%s%s%s"
		,G__psep,filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
#ifndef G__OLDIMPLEMENTATION1271
	if(G__ifile.fp && G__autoload_stdheader) {
	  G__globalcomp=G__store_globalcomp;
	  G__gen_linksystem(filename);
	}
#endif
#ifndef G__OLDIMPLEMENTATION1536
	hdrprop = G__CINTHDR;
#endif
	G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;

      /**********************************************
       * try stl
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"stl%s%s%s"
		,G__psep,filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
#ifndef G__OLDIMPLEMENTATION1271
	if(G__ifile.fp && G__autoload_stdheader) {
	  G__globalcomp=G__store_globalcomp;
	  G__gen_linksystem(filename);
	}
#endif
#ifndef G__OLDIMPLEMENTATION1536
	hdrprop = G__CINTHDR;
#endif
	G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;
#endif /* G__EDU_VERSION */

#ifdef G__VISUAL
      /**********************************************
       * try /msdev/include
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"/msdev/include/%s%s",filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
	G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
#endif /* G__VISUAL */

#ifdef G__SYMANTEC
      /**********************************************
       * try /sc/include
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"/sc/include/%s%s",filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
	G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
#endif /* G__SYMANTEC */

#ifdef G__VMS
       /**********************************************
       * try $ROOTSYS[include]
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
	/*  sprintf(G__ifile.name,getenv("ROOTSYS"));
	    sprintf(&G__ifile.name[strlen(G__ifile.name)-1],".include]%s",filename);*/
	sprintf(G__ifile.name,"%s[include]%s",getenv("ROOTSYS"),filename);
	
	G__ifile.fp = fopen(G__ifile.name,"r");
	/*G__globalcomp=G__store_globalcomp;*/
      }
      if(G__ifile.fp) break;
      
       /**********************************************
       * try $ROOTSYS[cint.include]
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
	/*   sprintf(G__ifile.name,"%s",G__cintsysdir);
	     sprintf(&G__ifile.name[strlen(G__ifile.name)-1],".include]%s",filename);*/
	sprintf(G__ifile.name,"%s[include]%s",G__cintsysdir,filename);
	
	G__ifile.fp = fopen(G__ifile.name,"r");
#ifndef G__OLDIMPLEMENTATION1536
	hdrprop = G__CINTHDR;
#endif
	G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;
      
       /**********************************************
       * try sys$common:[decc$lib.reference.decc$rtldef..]
       **********************************************/

      sprintf(G__ifile.name,"sys$common:decc$lib.reference.decc$rtdef]%s",filename);
      printf("Trying to open %s\n",G__ifile.name,"r");
      
      G__ifile.fp = fopen(G__ifile.name,"r");
      G__globalcomp=G__store_globalcomp;
      
      if(G__ifile.fp) break;
      
#endif  /*G__VMS*/

      /**********************************************
       * try /usr/include/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"/usr/include/%s%s",filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
	G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;

#ifdef __GNUC__
      /**********************************************
       * try /usr/include/g++/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"/usr/include/g++/%s%s",filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
	G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
#endif /* __GNUC__ */

/* #ifdef __hpux */
      /**********************************************
       * try /usr/include/CC/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"/usr/include/CC/%s%s",filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
	G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
/* #endif __hpux */

/* #ifdef __hpux */
      /**********************************************
       * try /usr/include/codelibs/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
	sprintf(G__ifile.name,"/usr/include/codelibs/%s%s"
		,filename,addpost[i2]);
#ifndef G__WIN32
	G__ifile.fp = fopen(G__ifile.name,"r");
#else
	G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
	G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
/* #endif __hpux */
    }
  }
    

  /**********************************************
  * filenum and line_number.
  **********************************************/
  G__ifile.line_number = 1;
  /********************************************
  * if there is null_entry which has been unloaded,
  * use that index. null_entry is found above.
  ********************************************/
  if(null_entry == -1) {
    G__ifile.filenum = G__nfile ;
  }
  else {
    G__ifile.filenum = null_entry ;
  }


  if (!G__ifile.fp) {
    /******************************************************
     * restore input file information to G__ifile
     * and reset G__eof to 0.
     ******************************************************/
    G__ifile = store_file ;
    G__eof = 0;
    G__step=store_step;
    G__globalcomp=G__store_globalcomp;
#ifndef G__OLDIMPLEMENTATION782
    if(0==G__ispragmainclude) {
      G__fprinterr(G__serr,"Error: cannot open file \"%s\" ", filename);
      G__genericerror((char*)NULL);
    }
#else
    G__fprinterr(G__serr,"Error: cannot open file \"%s\" ", filename);
    G__genericerror((char*)NULL);
#endif
    G__iscpp=store_iscpp;
#ifndef G__OLDIMPLEMENTATION1345
    G__UnlockCriticalSection();
#endif
    return(G__LOADFILE_FAILURE);
  }
  else {
#ifndef G__OLDIMPLEMENTATION1210
    if(G__ignoreinclude && (*G__ignoreinclude)(filename,G__ifile.name)) {
#ifndef G__PHILIPPE7
      /* Close file for process max file open limitation with -cN option */
      /* fclose(G__srcfile[fentry].fp); */
      fclose(G__ifile.fp);
      /* since we ignore the file, we can assume that it has no template
	 nor any references... */
      
      /******************************************************
       * restore input file information to G__ifile
       * and reset G__eof to 0.
       ******************************************************/
      G__ifile = store_file ;
      G__eof = 0;
      G__step=store_step;
      G__setdebugcond();
      G__globalcomp=G__store_globalcomp;
      G__iscpp=store_iscpp;
#endif
#ifndef G__OLDIMPLEMENTATION1345
      G__UnlockCriticalSection();
#endif
      return(G__LOADFILE_SUCCESS);
    }
#endif
    G__srcfile[G__nfile].dictpos
      = (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
#ifndef G__OLDIMPLEMENTATION2014
    G__srcfile[G__nfile].dictpos->ptype = (char*)NULL;
#endif
    G__store_dictposition(G__srcfile[G__nfile].dictpos);
    /***************************************************
     * set
     *  char  G__filenameary[][]
     *  FILE *G__filearray
     *  int  G__nfile
     ***************************************************/
    /********************************************
     * if there is null_entry which has been
     * unloaded, use that index.
     ********************************************/
    if(null_entry == -1) {
      fentry = G__nfile;
      G__nfile++;
    }
    else {
      fentry=null_entry;
    }

#ifndef G__OLDIMPLEMENTATION1536
    G__srcfile[fentry].hdrprop = hdrprop;
#endif

#ifdef G__SECURITY
    store_security = G__security;
    G__srcfile[fentry].security = G__security;
#endif

    G__srcfile[fentry].hash=hash;
    if(prepname[0]) {
      G__srcfile[fentry].prepname = (char*)malloc(strlen(prepname)+1);
      strcpy(G__srcfile[fentry].prepname,prepname);
    }
    else {
      G__srcfile[fentry].prepname = (char*)NULL;
    }
    if(G__globalcomp<G__NOLINK) {
      G__srcfile[fentry].filename = (char*)malloc(strlen(G__ifile.name)+1);
      strcpy(G__srcfile[fentry].filename,G__ifile.name);
    }
    else {
      G__srcfile[fentry].filename = (char*)malloc(strlen(filename)+1);
      strcpy(G__srcfile[fentry].filename,filename);
    }
    G__srcfile[fentry].fp=G__ifile.fp;
#ifndef G__OLDIMPLEMENTATION952
    G__srcfile[fentry].included_from = store_file.filenum;
#endif
#ifndef G__OLDIMPLEMENTATION1207
    G__srcfile[fentry].ispermanentsl = G__ispermanentsl;
    G__srcfile[fentry].initsl = (G__DLLINIT)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION1273
    G__srcfile[fentry].hasonlyfunc = (struct G__dictposition*)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION1756
    G__srcfile[fentry].parent_tagnum = G__get_envtagnum();
#endif
#ifndef G__OLDIMPLEMENTATION1908
    G__srcfile[fentry].slindex = -1;
#endif
  }

  if(G__debugtrace) {
    G__fprinterr(G__serr,"LOADING file=%s:%s:%s\n",filename,G__ifile.name,prepname);
  }
  if(G__debug) {
    G__fprinterr(G__serr,"%-5d",G__ifile.line_number);
  }

  store_prerun=G__prerun;
  store_p_local=G__p_local;
#ifndef G__OLDIMPLEMENTATION616
  if(0==G__def_struct_member||-1==G__tagdefining||
     ('n'!=G__struct.type[G__tagdefining]
#ifndef G__OLDIMPLEMENTATION1608
      && 'c'!=G__struct.type[G__tagdefining]
      && 's'!=G__struct.type[G__tagdefining]
#endif
     )) {
    G__p_local=NULL;
  }
#else
  G__p_local=NULL;
#endif

  G__eof = 0;
  G__prerun = 1;
  G__switch = 0;
  G__mparen = 0;
  store_nobreak=G__nobreak;
  G__nobreak=1;

  store_var_type = G__var_type;
  store_tagnum = G__tagnum;
  store_typenum = G__typenum;
#ifndef G__OLDIMPLEMENTATION487
  store_func_now = G__func_now;
  G__func_now = -1;
#endif
  store_macroORtemplateINfile = G__macroORtemplateINfile;
  G__macroORtemplateINfile = 0;
#ifndef G__OLDIMPLEMENTATION974
  store_asm_noverflow = G__asm_noverflow;
  store_no_exec_compile = G__no_exec_compile;
  store_asm_exec = G__asm_exec;
  G__asm_noverflow = 0;
  G__no_exec_compile = 0;
  G__asm_exec = 0;
#endif
#ifndef G__OLDIMPLEMENTATION1849
  store_return=G__return;
  G__return=G__RETURN_NON;
#endif

#ifdef G__SHAREDLIB
  len = strlen(filename);
#ifndef G__OLDIMPLEMENTATION1705
  dllpost = G__getmakeinfo1("DLLPOST");
#endif
  if((len>3&& (strcmp(filename+len-3,".sl")==0 ||
	       strcmp(filename+len-3,".dl")==0 ||
	       strcmp(filename+len-3,".so")==0)) ||
     (len>4&& (strcmp(filename+len-4,".dll")==0 ||
	       strcmp(filename+len-4,".DLL")==0)) ||
#if defined(R__FBSD) || defined(R__OBSD)
     (len>strlen(soext) && strcmp(filename+len-strlen(soext), soext)==0) ||
#endif
#ifndef G__OLDIMPLEMENTATION1705
     (
#ifndef G__OLDIMPLEMENTATION1873
      dllpost[0] && 
#endif
      len>(len1=strlen(dllpost)) && strcmp(filename+len-len1,dllpost)==0) ||
#endif
     (len>2&& (strcmp(filename+len-2,".a")==0 ||
	       strcmp(filename+len-2,".A")==0))
     ) {
    /* Caution, G__ifile.fp is left openned.
     * This may cause trouble in future */
    fclose(G__srcfile[fentry].fp);
#ifndef G__OLDIMPLEMENTATION2224
    if (G__ifile.fp == G__srcfile[fentry].fp) {
      /* Since the file is closed, the FILE* pointer is now invalid and thus
	 we have to remove it from G__ifile! */
      G__ifile.fp=(FILE*)NULL;
    }
#endif
    G__srcfile[fentry].fp=(FILE*)NULL;
#ifndef G__OLDIMPLEMENTATION1908
    G__srcfile[fentry].slindex = G__shl_load(G__ifile.name);
#else
    G__shl_load(G__ifile.name);
#endif
#ifndef G__OLDIMPLEMENTATION1207
    if(G__ispermanentsl) {
      G__srcfile[fentry].initsl = G__initpermanentsl;
    }
#endif
  }
  else {
    if(G__globalcomp>1 && strcmp(filename+strlen(filename)-4,".sut")==0) {
      G__ASSERT(G__sutpi && G__ifile.fp);
      G__copyfile(G__sutpi,G__ifile.fp);
    }
    else {
#ifndef G__OLDIMPLEMENTATION1072
      long store_struct_offset = G__store_struct_offset;
      G__store_struct_offset = 0;
#endif
      if(G__isbinaryfile(filename)) {
	G__iscpp=store_iscpp;
#ifdef G__SECURITY
	G__security = store_security;
#endif
#ifndef G__OLDIMPLEMENTATION1345
	G__UnlockCriticalSection();
#endif
#ifndef G__OLDIMPLEMENTATION1849
	G__RESTORE_LOADFILEENV;
#endif
	return(G__LOADFILE_FAILURE);
      }
#ifndef G__OLDIMPLEMENTATION970
      if(G__copyflag) G__copysourcetotmp(prepname,&G__ifile,fentry);
#endif
      while (!G__eof && G__return<G__RETURN_EXIT1) G__exec_statement();
#ifndef G__OLDIMPLEMENTATION1072
      G__store_struct_offset = store_struct_offset;
#endif
    }
  }
#else /* of G__SHAREDLIB */
  if(G__globalcomp>1 && strcmp(filename+strlen(filename)-4,".sut")==0) {
    G__ASSERT(G__sutpi && G__ifile.fp);
    G__copyfile(G__sutpi,G__ifile.fp);
  }
  else {
    if(G__isbinaryfile(filename)) {
      G__iscpp=store_iscpp;
#ifdef G__SECURITY
      G__security = store_security;
#endif
#ifndef G__OLDIMPLEMENTATION1345
      G__UnlockCriticalSection();
#endif
#ifndef G__OLDIMPLEMENTATION1849
      G__RESTORE_LOADFILEENV;
#endif
      return(G__LOADFILE_FAILURE);
    }
    while (!G__eof && G__return<G__RETURN_EXIT1) G__exec_statement();
  }
#endif  /* of G__SHAREDLIB */

  /******************************************************
   * Avoid file array overflow when G__globalcomp
   ******************************************************/
  if(G__NOLINK!=G__globalcomp && G__srcfile[fentry].fp) {
    if(!G__macroORtemplateINfile
#ifndef G__OLDIMPLEMENTATION1923
       && (!G__fons_comment || !G__cpp)
#endif
       ) {
#ifdef G__OLDIMPLEMENTATION1562
      /* Close file for process max file open limitation with -cN option */
      fclose(G__srcfile[fentry].fp);
#endif
#ifndef G__PHILIPPE0
      /* After closing the file let's make sure that all reference to
	 the file pointer are reset. When preprocessor is used, we
	 will have several logical file packed in one file. */
      tmpfp = G__srcfile[fentry].fp;
      for(i1=0;i1<G__nfile;i1++) {
        if (G__srcfile[i1].fp==tmpfp){
	  G__srcfile[i1].fp = (FILE*)NULL;
	}
      }
#ifndef G__OLDIMPLEMENTATION1562
      /* Close file for process max file open limitation with -cN option */
      fclose(tmpfp);
#endif
#else
#ifndef G__OLDIMPLEMENTATION1562
      /* Close file for process max file open limitation with -cN option */
      fclose(G__srcfile[fentry].fp);
#endif
      G__srcfile[fentry].fp = (FILE*)NULL;
#endif
    }
  }


  /******************************************************
   * restore parser parameters
   ******************************************************/
#ifndef G__OLDIMPLEMENTATION487
  pragmacompile_filenum = G__ifile.filenum;
  pragmacompile_iscpp = G__iscpp;
  G__func_now = store_func_now;
#endif
  G__macroORtemplateINfile = store_macroORtemplateINfile;
  G__var_type = store_var_type;
  G__tagnum = store_tagnum;
  G__typenum = store_typenum;

  G__nobreak=store_nobreak;
  G__prerun=store_prerun;
  G__p_local=store_p_local;

#ifndef G__OLDIMPLEMENTATION974
  G__asm_noverflow = store_asm_noverflow;
  G__no_exec_compile = store_no_exec_compile;
  G__asm_exec = store_asm_exec;
#endif

  /******************************************************
   * restore input file information to G__ifile
   * and reset G__eof to 0.
   ******************************************************/
  G__ifile = store_file ;
  G__eof = 0;
  G__step=store_step;
  G__setdebugcond();
  G__globalcomp=G__store_globalcomp;
  G__iscpp=store_iscpp;
#ifdef G__SECURITY
  G__security = store_security;
#endif
  if(G__return>G__RETURN_NORMAL) {
#ifndef G__OLDIMPLEMENTATION1345
    G__UnlockCriticalSection();
#endif
#ifndef G__OLDIMPLEMENTATION1849
    G__return=store_return;
#endif
    return(G__LOADFILE_FAILURE);
  }

#ifndef G__OLDIMPLEMENTATION1849
  G__return=store_return;
#endif

#ifndef G__OLDIMPLEMENTATION487
#ifdef G__AUTOCOMPILE
  /*************************************************************
   * if '#pragma compile' appears in source code.
   *************************************************************/
  if(G__fpautocc && G__autoccfilenum == pragmacompile_filenum) {
    store_iscpp = G__iscpp;
    G__iscpp=pragmacompile_iscpp;
    G__autocc();
    G__iscpp = store_iscpp;
  }
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1273
  G__checkIfOnlyFunction(fentry);
#endif

#ifndef G__OLDIMPLEMENTATION1345
  G__UnlockCriticalSection();
#endif
  return(G__LOADFILE_SUCCESS);
}

/**************************************************************************
* G__preprocessor()
*
* Use C/C++ preprocessor prior to interpretation.
*  Name of preprocessor must be defined in $CINTSYSDIR/MAKEINFO file as
*  CPPPREP and CPREP.
*
**************************************************************************/
int G__preprocessor(outname,inname,cppflag,macros,undeflist,ppopt,includepath)
char *outname,*inname;
int cppflag;
char *macros,*undeflist,*ppopt,*includepath;
{
  char temp[G__LARGEBUF];
  /* char *envcpp; */
  char tmpfile[G__MAXFILENAME];
  int tmplen;
  FILE *fp;
  int flag=0;
#ifndef G__OLDIMPLEMENTATION503
  int inlen;
  char *post;

  inlen = strlen(inname);
  post = strrchr(inname,'.');
#endif

#ifndef G__OLDIMPLEMENTATION503
  if(post && inlen>2) {
#else
  if(strlen(inname)>2) {
#endif
    if(0==strcmp(inname+strlen(inname)-2,".c")) {
      if(!G__cpplock) G__iscpp = 0;
      flag=1;
    }
    else if(0==strcmp(inname+strlen(inname)-2,".C")) {
      if(!G__clock) G__iscpp = 1;
      flag=1;
    }
    else if(0==strcmp(inname+strlen(inname)-2,".h")) {
      flag=1;
    }
    else if(0==strcmp(inname+strlen(inname)-2,".H")) {
      flag=1;
    }
  }
#ifndef G__OLDIMPLEMENTATION503
  if(flag==0 && post && inlen>3) {
#else
  else if(strlen(inname)>3) {
#endif
    if(0==strcmp(inname+strlen(inname)-3,".cc") ||
       0==strcmp(inname+strlen(inname)-3,".CC") ||
       0==strcmp(inname+strlen(inname)-3,".hh") ||
       0==strcmp(inname+strlen(inname)-3,".HH") ||
       0==strcmp(inname+strlen(inname)-3,".wc") ||
       0==strcmp(inname+strlen(inname)-3,".WC")) {
      G__iscpp = 1;
      flag=1;
    }
  }
#ifndef G__OLDIMPLEMENTATION503
  if(flag==0 && post && inlen>4) {
#else
  else if(strlen(inname)>4) {
#endif
    if(0==strcmp(inname+strlen(inname)-4,".cxx") ||
       0==strcmp(inname+strlen(inname)-4,".CXX") ||
       0==strcmp(inname+strlen(inname)-4,".cpp") ||
       0==strcmp(inname+strlen(inname)-4,".CPP") ||
       0==strcmp(inname+strlen(inname)-4,".hxx") ||
       0==strcmp(inname+strlen(inname)-4,".HXX") ||
       0==strcmp(inname+strlen(inname)-4,".hpp") ||
       0==strcmp(inname+strlen(inname)-4,".HPP")) {
      G__iscpp=1;
      flag=1;
    }
  }
#ifndef G__OLDIMPLEMENTATION503
  if(flag==0 && post) {
#else
  else {
#endif
    if('\0'==G__cppsrcpost[0]) {
#ifndef G__OLDIMPLEMENTATION1645
      strcpy(G__cppsrcpost,G__getmakeinfo1("CPPSRCPOST"));
#else
      strcpy(G__cppsrcpost,G__getmakeinfo("CPPSRCPOST"));
#endif
    }
    if('\0'==G__csrcpost[0]) {
#ifndef G__OLDIMPLEMENTATION1645
      strcpy(G__csrcpost,G__getmakeinfo1("CSRCPOST"));
#else
      strcpy(G__csrcpost,G__getmakeinfo("CSRCPOST"));
#endif
    }
    if('\0'==G__cpphdrpost[0]) {
#ifndef G__OLDIMPLEMENTATION1645
      strcpy(G__cpphdrpost,G__getmakeinfo1("CPPHDRPOST"));
#else
      strcpy(G__cpphdrpost,G__getmakeinfo("CPPHDRPOST"));
#endif
    }
    if('\0'==G__chdrpost[0]) {
#ifndef G__OLDIMPLEMENTATION1645
      strcpy(G__chdrpost,G__getmakeinfo1("CHDRPOST"));
#else
      strcpy(G__chdrpost,G__getmakeinfo("CHDRPOST"));
#endif
    }
    if(0==strcmp(inname+strlen(inname)-strlen(G__cppsrcpost),G__cppsrcpost)) {
      if(!G__clock) G__iscpp=1;
      flag=1;
    }
    else if(0==strcmp(inname+strlen(inname)-strlen(G__csrcpost),G__csrcpost)) {
      if(!G__cpplock) G__iscpp=0;
      flag=1;
    }
    else if(0==strcmp(inname+strlen(inname)-strlen(G__cpphdrpost)
		      ,G__cpphdrpost)) {
      flag=1;
    }
    else if(0==strcmp(inname+strlen(inname)-strlen(G__chdrpost),G__chdrpost)) {
      if(!G__cpplock) G__iscpp=0;
      flag=1;
    }
  }
#ifndef G__OLDIMPLEMENTATION503
  else if(flag==0&&!post) {
    if(!G__clock) G__iscpp=1;
    flag=1;
  }
#endif

  /* If using C preprocessor '-p' option' */
  if(cppflag && flag) {

    /* Determine what C/C++ preprocessor to use */
    if('\0'==G__ccom[0]) {
      switch(G__globalcomp) {
      case G__CPPLINK: /* C++ link */
	strcpy(G__ccom,G__getmakeinfo("CPPPREP"));
	break;
      case G__CLINK: /* C link */
	strcpy(G__ccom,G__getmakeinfo("CPREP"));
	break;
      default:
	if(G__iscpp) strcpy(G__ccom,G__getmakeinfo("CPPPREP"));
	else         strcpy(G__ccom,G__getmakeinfo("CPREP"));
	break;
      }
      if('\0'==G__ccom[0]) {
#ifdef __GNUC__
	sprintf(G__ccom,"g++ -E");
#else
	sprintf(G__ccom,"CC -E");
#endif
      }
    }

    /* Get tmpfile name if necessary */
#ifndef G__OLDIMPLEMENTATION1099
    if((strlen(inname)>2 && (0==strcmp(inname+strlen(inname)-2,".H")||
			     0==strcmp(inname+strlen(inname)-2,".h"))) ||
       (strlen(inname)>3 && (0==strcmp(inname+strlen(inname)-3,".hh")||
			     0==strcmp(inname+strlen(inname)-3,".HH"))) ||
       (strlen(inname)>4 && (0==strcmp(inname+strlen(inname)-4,".hpp")||
			     0==strcmp(inname+strlen(inname)-4,".HPP")||
			     0==strcmp(inname+strlen(inname)-4,".hxx")||
			     0==strcmp(inname+strlen(inname)-4,".HXX"))) ||
       (!strchr(inname,'.'))
       ) 
#else
    if(0==strcmp(inname+strlen(inname)-2,".H")||
       0==strcmp(inname+strlen(inname)-2,".h")) 
#endif
    {
      /* if header file, create tmpfile name as xxx.C */
      do {
	G__tmpnam(tmpfile); /* can't replace this with tmpfile() */
	tmplen=strlen(tmpfile);
	if(G__CPPLINK==G__globalcomp || G__iscpp) {
	  if('\0'==G__cppsrcpost[0]) {
#ifndef G__OLDIMPLEMENTATION1645
	    strcpy(G__cppsrcpost,G__getmakeinfo1("CPPSRCPOST"));
#else
	    strcpy(G__cppsrcpost,G__getmakeinfo("CPPSRCPOST"));
#endif
	  }
	  strcpy(tmpfile+tmplen,G__cppsrcpost);
	}
	else {
	  if('\0'==G__csrcpost[0]) {
#ifndef G__OLDIMPLEMENTATION1645
	    strcpy(G__csrcpost,G__getmakeinfo1("CSRCPOST"));
#else
	    strcpy(G__csrcpost,G__getmakeinfo("CSRCPOST"));
#endif
	  }
	  strcpy(tmpfile+tmplen,G__csrcpost);
	}
	fp = fopen(tmpfile,"w");
      } while((FILE*)NULL==fp && G__setTMPDIR(tmpfile));
      if(fp) {
	fprintf(fp,"#include \"%s\"\n\n\n",inname);
	fclose(fp);
      }
    }
    else {
      /* otherwise, simply copy the inname */
      strcpy(tmpfile,inname);
      tmplen=0;
    }

    /* Get output file name */
    G__getcintsysdir();
    G__tmpnam(outname); /* can't replace this with tmpfile() */
#if defined(G__SYMANTEC) && (!defined(G__TMPFILE))
    /* NEVER DONE */
    { int len_outname = strlen(outname);
      outname[len_outname]   = '.';
      outname[len_outname+1] = '\0';
    }
#endif

#if defined(G__SYMANTEC)
    /**************************************************************
     * preprocessor statement for Symantec C++
     ***************************************************************/
    if(G__cintsysdir[0]) {
      sprintf(temp,"%s %s %s -I. %s %s -D__CINT__ -I%s/include -I%s/stl -I%s/lib %s -o%s"
	      ,G__ccom ,macros,undeflist,ppopt ,includepath
	      ,G__cintsysdir,G__cintsysdir,G__cintsysdir,tmpfile,outname);
    }
    else {
      sprintf(temp,"%s %s %s %s -I. %s -D__CINT__ %s -o%s" ,G__ccom
	      ,macros,undeflist,ppopt ,includepath ,tmpfile,outname);
    }
#elif defined(G__BORLAND)
    /**************************************************************
     * #elif defined(G__BORLAND)
     *      preprocessor statement for Borland C++ to be implemented
     ***************************************************************/
    strcat(outname,".i");
    if(G__cintsysdir[0]) {
      sprintf(temp,"%s %s %s -I. %s %s -D__CINT__ -I%s/include -I%s/stl -I%s/lib -o%s %s"
	      ,G__ccom ,macros,undeflist,ppopt ,includepath
	      ,G__cintsysdir,G__cintsysdir,G__cintsysdir,outname,tmpfile);
    }
    else {
      sprintf(temp,"%s %s %s %s -I. %s -D__CINT__ -o%s %s" ,G__ccom
	      ,macros,undeflist,ppopt ,includepath ,outname,tmpfile);
    }
#else
    /**************************************************************
     * #elif defined(G__VISUAL)
     *      preprocessor statement for Visual C++ to be implemented
     ***************************************************************/
    /**************************************************************
     * preprocessor statement for UNIX
     ***************************************************************/
   if(G__cintsysdir[0]) {
      sprintf(temp,"%s %s %s -I. %s %s -D__CINT__ -I%s/include -I%s/stl -I%s/lib %s > %s"
              ,G__ccom ,macros,undeflist,ppopt ,includepath
              ,G__cintsysdir,G__cintsysdir,G__cintsysdir,tmpfile,outname);
    }
    else {
      sprintf(temp,"%s %s %s %s -I. %s -D__CINT__ %s > %s" ,G__ccom
              ,macros,undeflist,ppopt ,includepath ,tmpfile,outname);
    }
#endif
    if(G__debugtrace||G__steptrace||G__step||G__asm_dbg)
      G__fprinterr(G__serr," %s\n",temp);
    system(temp);

    if(tmplen) remove(tmpfile);
  }

  else {
    /* Didn't use C preprocessor because cppflag is not set or file name
     * suffix does not match */
    outname[0]='\0';
  }
  return(0);
}


#ifndef G__OLDIMPLEMENTATION486
/**************************************************************************
* G__difffile()
**************************************************************************/
int G__difffile(file1,file2)
char *file1;
char *file2;
{
  FILE *fp1;
  FILE *fp2;
  int c1,c2;
  int unmatch=0;

  fp1=fopen(file1,"r");
  fp2=fopen(file2,"r");
  if(fp1 && fp2) {
    do {
      c1=fgetc(fp1);
      c2=fgetc(fp2);
      if(c1!=c2) {
	++unmatch;
	break;
      }
    } while(EOF!=c1 && EOF!=c2);
    if(c1!=c2) ++unmatch;
  }
  else {
    unmatch=1;
  }
  if(fp1) fclose(fp1);
  if(fp2) fclose(fp2);

  return(unmatch);
}
#endif

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


#ifdef G__TMPFILE
static char G__tmpdir[G__MAXFILENAME];
static char G__mfpname[G__MAXFILENAME];
#elif !defined(G__OLDIMPLEMENTATION2092)
static char G__mfpname[G__MAXFILENAME];
#endif


/**************************************************************************
* G__setTMPDIR()
**************************************************************************/
int G__setTMPDIR(badname)
char *badname;
{
#ifndef G__TMPFILE
  G__fprinterr(G__serr,"CAUTION: tmpfile %s can't open\n",badname);
  return(0);
#else
  char *p;
  G__fprinterr(G__serr,"CINT CAUTION: tmpfile %s can't open\n",badname);
  G__fprinterr(G__serr,"Input another temp directory or '*' to give up\n");
  G__fprinterr(G__serr,"(Setting CINTTMPDIR environment variable avoids this interrupt)\n");
  strcpy(G__tmpdir,G__input("Input TMPDIR > "));
  p = strchr(G__tmpdir,'\r');
  if(p) *p = '\0';
  p = strchr(G__tmpdir,'\n');
  if(p) *p = '\0';
  if('*'==G__tmpdir[0]) {
    G__tmpdir[0]='\0';
    return(0);
  }
  else {
    return(1);
  }
#endif
}

/**************************************************************************
* G__tmpnam()
**************************************************************************/
char* G__tmpnam(name)
char *name;
{
#if defined(G__TMPFILE) 
  const char *appendix="_cint";
  static char tempname[G__MAXFILENAME];
#ifndef G__OLDIMPLEMENTATION2174
  int pid = getpid();
  int now = clock();
#endif
  char *tmp;
  if('\0'==G__tmpdir[0]) {
    if((tmp=getenv("CINTTMPDIR"))) strcpy(G__tmpdir,tmp);
    else if((tmp=getenv("TEMP"))) strcpy(G__tmpdir,tmp);
    else if((tmp=getenv("TMP"))) strcpy(G__tmpdir,tmp);
    else strcpy(G__tmpdir,".");
  }
  if(name) {
    strcpy(name,(tmp=tempnam(G__tmpdir,"")));
    free((void*)tmp);
#ifndef G__OLDIMPLEMENTATION2174
    if(strlen(name)<G__MAXFILENAME-10) 
      sprintf(name+strlen(name),"%d%d",pid%10000,now%10000);
    if(strlen(name)<G__MAXFILENAME-6) strcat(name,appendix);
#else
    if(strlen(name)<L_tmpnam-6) strcat(name,appendix);
#endif
    return(name);
  }
  else {
    strcpy(tempname,(tmp=tempnam(G__tmpdir,"")));
    free((void*)tmp);
#ifndef G__OLDIMPLEMENTATION2174
    if(strlen(name)<G__MAXFILENAME-10) 
      sprintf(name+strlen(name),"%d%d",pid%10000,now%10000);
    if(strlen(tempname)<G__MAXFILENAME-6) strcat(tempname,appendix);
#else
    if(strlen(tempname)<L_tmpnam-6) strcat(tempname,appendix);
#endif
    return(tempname);
  }

#elif defined(__CINT__)
  const char *appendix="_cint";
  tmpnam(name);
  if(strlen(name)<G__MAXFILENAME-6) strcat(name,appendix);
  return(name);

#elif /*defined(G__NEVER) && */ ((__GNUC__>=3)||((__GNUC__>=2)&&(__GNUC_MINOR__>=96)))&&(defined(__linux)||defined(__linux__))
  /* After all, mkstemp creates more problem than a solution. */
  const char *appendix="_cint";
  strcpy(name,"/tmp/XXXXXX");
  close(mkstemp(name));/*mkstemp not only generate file name but also opens the file*/
  remove(name); /* mkstemp creates this file anyway. Delete it. questionable */
  if(strlen(name)<G__MAXFILENAME-6) strcat(name,appendix);
  return(name);

#else
  const char *appendix="_cint";
  tmpnam(name);
  if(strlen(name)<G__MAXFILENAME-6) strcat(name,appendix);
  return(name);

#endif
}

#ifndef G__OLDIMPLEMENTATION2092
static int G__istmpnam=0;
#endif

/**************************************************************************
* G__openmfp()
**************************************************************************/
void G__openmfp()
{
#ifndef G__TMPFILE
#ifndef G__OLDIMPLEMENTATION2092
  G__mfp=tmpfile();
  if(!G__mfp) {
    do {
      G__tmpnam(G__mfpname); /* Only VC++ uses this */
      G__mfp=fopen(G__mfpname,"wb+");
    } while((FILE*)NULL==G__mfp && G__setTMPDIR(G__mfpname));
    G__istmpnam=1;
  }
#else
  G__mfp=tmpfile();
#endif
#else
  do {
    G__tmpnam(G__mfpname); /* Only VC++ uses this */
    G__mfp=fopen(G__mfpname,"wb+");
  } while((FILE*)NULL==G__mfp && G__setTMPDIR(G__mfpname));
#endif
}

/**************************************************************************
* G__closemfp()
**************************************************************************/
int G__closemfp()
{
#ifndef G__TMPFILE
  int result=0;
#ifndef G__OLDIMPLEMENTATION2092
  if(!G__istmpnam) {
    if(G__mfp) result=fclose(G__mfp);
    G__mfp = (FILE*)NULL;
  }
  else {
    if(G__mfp) fclose(G__mfp);
    G__mfp = (FILE*)NULL;
    if(G__mfpname[0]) result=remove(G__mfpname);
    G__mfpname[0]=0;
    G__istmpnam=0;
  }
#else
  if(G__mfp) result=fclose(G__mfp);
  G__mfp = (FILE*)NULL;
#endif
  return(result);
#else
  int result=0;
  if(G__mfp) fclose(G__mfp);
  G__mfp = (FILE*)NULL;
  if(G__mfpname[0]) result=remove(G__mfpname);
  G__mfpname[0]=0;
  return(result);
#endif
}


#if 0
/* Just for experimenting Windows Server OS tmpfile patch */
FILE* G__dmy_tmpfile() { return((FILE*)NULL); }
#endif

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
