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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/* Define one of following */
#define G__OLDIMPLEMENTATION1922 /* keep opening all header files for +V +P */

#include <string>
#include <list>

#ifdef _WIN32
#include "process.h"
#endif

#if defined(G__ROOT)
#include "RConfigure.h"
#endif

#if defined(G__HAVE_CONFIG)
#include "configcint.h"
#endif

#include "common.h"
#include "Api.h"
#include "FastAllocString.h"

#ifndef G__TESTMAIN
#include <sys/stat.h>
#endif

#ifdef G__WIN32
#include <windows.h>
#endif

#define G__OLDIMPLEMENTATION1849

#ifdef G__SHAREDLIB
extern std::list<G__DLLINIT>* G__initpermanentsl;
#endif

extern "C" {

/******************************************************************
* Define G__EDU_VERSION for CINT C++ educational version.
* If G__EDU_VERSION is defined, CINT will search ./include and
* ./stl directory for standard header files.
******************************************************************/
/* #define G__EDU_VERSION */

static G__IgnoreInclude G__ignoreinclude = (G__IgnoreInclude)NULL;

/******************************************************************
* G__set_ignoreinclude
******************************************************************/
void G__set_ignoreinclude(G__IgnoreInclude ignoreinclude)
{
  G__ignoreinclude = ignoreinclude;
}

static int G__kindofheader = G__USERHEADER;

static int G__copyflag = 0;

int (*G__ScriptCompiler) G__P((G__CONST char*,G__CONST char*)) = 0;

/******************************************************************
* G__RegisterScriptCompiler()
******************************************************************/
void G__RegisterScriptCompiler(int(*p2f) G__P((G__CONST char*,G__CONST char*)))
{
  G__ScriptCompiler = p2f;
}

/******************************************************************
* G__copytotmpfile()
******************************************************************/
static FILE* G__copytotmpfile(char *prepname)
{
  FILE *ifp;
  FILE *ofp;
  ifp = fopen(prepname,"rb");
  if(!ifp) {
    G__genericerror("Internal error: G__copytotmpfile() 1\n");
    return((FILE*)NULL);
  }
  ofp = fopen(G__tmpnam(0),"w+b");
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

/******************************************************************
* G__copysourcetotmp()
******************************************************************/
static void G__copysourcetotmp(G__FastAllocString &prepname,G__input_file *pifile, int fentry)
{
  if(G__copyflag && 0==prepname[0]) {
    FILE *fpout;
    fpout = fopen(G__tmpnam(0),"w+b");
    if(!fpout) {
      G__genericerror("Internal error: can not open tmpfile.");
      return;
    }
    prepname.Format("(tmp%d)",fentry);
    G__copyfile(fpout,pifile->fp);
    fseek(fpout,0L,SEEK_SET);
    G__srcfile[fentry].prepname = (char*)malloc(strlen(prepname)+1);
    strcpy(G__srcfile[fentry].prepname,prepname); // Okay, we allocated the right size
    G__srcfile[fentry].fp = fpout;
    fclose(pifile->fp);
    pifile->fp = fpout;
  }
}

/**************************************************************************
* G__setcopyflag()
**************************************************************************/
void G__setcopyflag(int flag)
{
  G__copyflag = flag;
}

/******************************************************************
* G__ispreprocessfilekey
******************************************************************/
static int G__ispreprocessfilekey(char *filename)
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
  char c;
  G__FastAllocString filename(G__MAXFILENAME);
  int i=0;
  int storeit=0;
  /* char sysinclude[G__MAXFILENAME]; */
  int store_cpp;
  int store_globalcomp;
  int expandflag=0;
  static int G__gcomplevel=0;

  while((c=G__fgetc())!='\n' && c!='\r'
        && c!='#'&& c!=-1
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
          filename.Set(i++, c);
          filename.Set(i, 0);
        }
        else if(-1!=storeit) {
          storeit=1;
          expandflag=1;
          filename.Set(i++, c);
          filename.Set(i, 0);
        }
      }
      else if(expandflag) {
        storeit = -1;
      }
      break;
    }
  }

  if(expandflag) {
    /* Following stupid code is written to avoid HP-UX CC -O bug */
    struct G__var_array *var;
    int ig15;
    int hash;
    G__hash(filename,hash,ig15);
    var = G__getvarentry(filename,hash,&ig15,&G__global,G__p_local);
    if(var) {
      filename = *(char**)var->p[ig15];
      G__kindofheader=G__USERHEADER;
    }
    else {
       G__fprinterr(G__serr,"Error: cannot expand #include %s",filename());
      G__genericerror(NULL);
      if('#'==c) G__fignoreline();
      return(G__LOADFILE_FAILURE);
    }
  }

  store_cpp=G__cpp;
  G__cpp=G__include_cpp;

  if(G__USERHEADER==G__kindofheader) {
    store_globalcomp = G__globalcomp;
    if(++G__gcomplevel>=G__gcomplevellimit) G__globalcomp=G__NOLINK;
    result = G__loadfile(filename);
    --G__gcomplevel;
    G__globalcomp=store_globalcomp;
  }
  else {
    /* <xxx.h> , 'xxx.h' */
    store_globalcomp=G__globalcomp;
    /* G__globalcomp=G__NOLINK; */
    if(++G__gcomplevel>=G__gcomplevellimit) G__globalcomp=G__NOLINK;
    result = G__loadfile(filename);
    --G__gcomplevel;
    G__globalcomp=store_globalcomp;
  }
  G__kindofheader = G__USERHEADER;

  G__cpp=store_cpp;

  if('#'==c) {
    if(G__LOADFILE_FAILURE==result && G__ispragmainclude) {
      G__ispragmainclude=0;
      c = G__fgetname(filename, 0, "\n\r");
      store_globalcomp = G__globalcomp;
      if(++G__gcomplevel>=G__gcomplevellimit) G__globalcomp=G__NOLINK;
      if('\n'!=c && '\r'!=c) result = G__include_file();
      --G__gcomplevel;
      G__globalcomp=store_globalcomp;
    }
    else {
      G__fignoreline();
    }
  }

  return(result);
}

/******************************************************************
* G__getmakeinfo()
*
******************************************************************/
#ifdef G__HAVE_CONFIG
const char *G__getmakeinfo(const char *item)
{
  if (!strcmp(item,"CPP")) return G__CFG_CXX;
  else if (!strcmp(item,"CC")) return G__CFG_CC;
  else if (!strcmp(item,"DLLPOST")) return G__CFG_SOEXT;
  else if (!strcmp(item,"CSRCPOST")) return ".c";
  else if (!strcmp(item,"CPPSRCPOST")) return ".cxx";
  else if (!strcmp(item,"CHDRPOST")) return ".h";
  else if (!strcmp(item,"CPPHDRPOST")) return ".h";
  else if (!strcmp(item,"INPUTMODE")) return G__CFG_INPUTMODE;
  else if (!strcmp(item,"INPUTMODELOCK")) return G__CFG_INPUTMODELOCK;
  else if (!strcmp(item,"CPREP")) return G__CFG_CPP;
  else if (!strcmp(item,"CPPPREP")) return G__CFG_CPP;
  else {
     printf("G__getmakeinfo for G__HAVE_CONFIG: %s not implemented yet!\n",
              item);
     return "";
  }
}
#elif defined(G__NOMAKEINFO)
const char *G__getmakeinfo(const char *)
{
  return("");
}
#else
const char *G__getmakeinfo(const char *item)
{
  G__FastAllocString makeinfo(G__MAXFILENAME);
  FILE *fp;
  G__FastAllocString line(G__LARGEBUF);
  G__FastAllocString argbuf(G__LARGEBUF);
  char *arg[G__MAXARG];
  int argn;
  char *p;
  static G__FastAllocString buf(G__ONELINE);

  buf[0]='\0';

  /****************************************************************
  * Environment variable overrides MAKEINFO file if exists.
  ****************************************************************/
  if((p=getenv(item)) && p[0] && !isspace(p[0])) {
    buf = p;
    return(buf);
  }

  /****************************************************************
  * Get information from MAKEINFO file.
  ****************************************************************/
  /* Get $CINTSYSDIR/MAKEINFO file name */
  if(G__getcintsysdir()) return(buf);
  makeinfo.Format("%s/%s/MAKEINFO",G__cintsysdir, G__CFG_COREVERSION);

  /* Open MAKEINFO file */
  fp = fopen(makeinfo,"r");
  if(!fp) {
     G__fprinterr(G__serr,"Error: cannot open %s\n",makeinfo());
    G__fprinterr(G__serr,
     "!!! There are examples of MAKEINFO files under %s/platform/ !!!\n"
            ,G__cintsysdir);
    G__fprinterr(G__serr,
            "Please refer to these examples and create for your platform\n");
    return(buf);
  }

  /* Read the MAKEINFO file */
  while(G__readline_FastAlloc(fp,line,argbuf,&argn,arg)) {
    if(argn>2 && strcmp(arg[1],item)==0) {
      p = strchr(arg[0],'=');
      if(p) {
        do {
          ++p;
        } while(isspace(*p));
        buf = p;
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
#endif

/******************************************************************
* G__getmakeinfo1()
*
* TO BE FIXED. ONE SHOULD CALL getmakeinfo instead
*
*
******************************************************************/
const char *G__getmakeinfo1(const char *item)
{
  return G__getmakeinfo(item);
  //char *buf = (char*)G__getmakeinfo(item);
//#ifndef G__HAVE_CONFIG
//  char *p = buf;
//  while(*p && !isspace(*p)) ++p;
//  *p = 0;
//#endif
//  return(buf);
}

/******************************************************************
* G__SetCINTSYSDIR()
*
******************************************************************/
void G__SetCINTSYSDIR(char *cintsysdir)
{
  G__strlcpy(G__cintsysdir,cintsysdir,sizeof(G__cintsysdir));
}

/******************************************************************
 * G__SetUseCINTSYSDIR()
 ******************************************************************/
static int G__UseCINTSYSDIR=0;
void G__SetUseCINTSYSDIR(int UseCINTSYSDIR)
{
  G__UseCINTSYSDIR=UseCINTSYSDIR;
}

/******************************************************************
* G__getcintsysdir()
*
*  print out error message for unsupported capability.
******************************************************************/
int G__getcintsysdir()
{
  const char *env;
  if('*'==G__cintsysdir[0]) {
#if defined(G__ROOT)
# ifdef ROOTBUILD
    env = "cint";
# else
#  ifdef CINTINCDIR
    env = CINTINCDIR;
#  else
    if(G__UseCINTSYSDIR) env=getenv("CINTSYSDIR");
    else                 env=getenv("ROOTSYS");
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
# ifdef G__CFG_DATADIRCINT
    if(!env || !env[0]) env = G__CFG_DATADIRCINT;
# endif
#endif
    if(env) {
#ifdef G__ROOT
# ifdef ROOTBUILD
      G__snprintf(G__cintsysdir,sizeof(G__cintsysdir), "%s", env);
# else /* ROOTBUILD */
#  ifdef CINTINCDIR
      G__snprintf(G__cintsysdir,sizeof(G__cintsysdir), "%s", CINTINCDIR);
#  else
       if(G__UseCINTSYSDIR) {
          G__strlcpy(G__cintsysdir,env,sizeof(G__cintsysdir));
       } else {
          G__strlcpy(G__cintsysdir,env,sizeof(G__cintsysdir));
          G__strlcat(G__cintsysdir,"/cint",sizeof(G__cintsysdir));
       }
#  endif
# endif /* ROOTBUILD */

#else /* G__ROOT */
      G__strlcpy(G__cintsysdir,env,sizeof(G__cintsysdir));
#endif /* G__ROOT */
      return(EXIT_SUCCESS);
    }
    else {
#ifdef G__EDU_VERSION
      G__snprintf(G__cintsysdir,sizeof(G__cintsysdir),".");
      return(EXIT_SUCCESS);
#else
#ifdef G__WIN32
      HMODULE hmodule=0;
      if(GetModuleFileName(hmodule,G__cintsysdir,G__MAXFILENAME)) {
        char *p = (char*)G__strrstr(G__cintsysdir,(char*)G__psep);
        if(p) *p = 0;
# ifdef G__ROOT
        p = (char*)G__strrstr(G__cintsysdir,(char*)G__psep);
        if(p) *p = 0;
        G__strlcat(G__cintsysdir,G__psep,sizeof(G__cintsysdir));
        G__strlcat(G__cintsysdir,"cint",sizeof(G__cintsysdir));
# endif
        return(EXIT_SUCCESS);
      }
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
int G__isfilebusy(int ifn)
{
  struct G__ifunc_table_internal *ifunc;
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

/******************************************************************
* G__matchfilename(i,filename)
******************************************************************/
int G__matchfilename(int i1,const char *filename)
{
#if  !defined(__CINT__)

  if (G__srcfile[i1].filename==0) {
    return 0;
  }
  if((strcmp(G__srcfile[i1].filename,filename)==0)) {
     return(1);
  }

#ifdef G__WIN32
   G__FastAllocString i1name(_MAX_PATH);
   G__FastAllocString fullfile(_MAX_PATH);
  _fullpath( i1name, G__srcfile[i1].filename, _MAX_PATH );
  _fullpath( fullfile, filename, _MAX_PATH );
  if((stricmp(i1name, fullfile)==0)) return 1;
#else
  struct stat statBufItem;
  struct stat statBuf;
  if (   ( 0 == stat( filename, & statBufItem ) )
      && ( 0 == stat( G__srcfile[i1].filename, & statBuf ) )
      && ( statBufItem.st_dev == statBuf.st_dev )     // Files on same device
      && ( statBufItem.st_ino == statBuf.st_ino )     // Files on same inode (but this is not unique on AFS so we need the next 2 test
      && ( statBufItem.st_size == statBuf.st_size )   // Files of same size
      && ( statBufItem.st_mtime == statBuf.st_mtime ) // Files modified at the same time
     ) {
     return 1;
  }
#endif
  return 0;

#else /* PHILIPPE28 */

  char *filenamebase;
  if((strcmp(G__srcfile[i1].filename,filename)==0)) return(1);
  filenamebase = (char*)G__strrstr(G__srcfile[i1].filename,"./");
  if(filenamebase) {
    char *parentdir = (char*)G__strrstr(G__srcfile[i1].filename,"../");
    if(!parentdir && strcmp(filename,filenamebase+2)==0) {
      G__FastAllocString buf(G__ONELINE);
#if defined(G__WIN32)
      char *p;
#endif
      if(filenamebase==G__srcfile[i1].filename) return(1);
#if defined(G__WIN32)
      GetCurrentDirectory(G__ONELINE,buf);
      p=buf;
      while((p=strchr(p,'\\'))) *p='/';
      if(strlen(buf)>1 && ':'==buf[1]) {
        G__FastAllocString buf2(buf+2);
        buf.Swap(buf2);
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

/******************************************************************
* G__stripfilename(filename)
******************************************************************/
const char* G__stripfilename(const char *filename)
{
  const char *filenamebase;
  if(!filename) return("");
  filenamebase = G__strrstr(filename,"./");
  if(filenamebase) {
    const char *parentdir = G__strrstr(filename,"../");
    G__FastAllocString buf(G__ONELINE);
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
      G__FastAllocString buf2(buf+2);
      buf.Swap(buf2);
    }
#elif defined(G__POSIX) || defined(G__ROOT)
    if (!getcwd(buf,G__ONELINE)) {
       G__fprinterr(G__serr, "Error retrieving CWD\n");
       buf[0] = 0;
    }
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

/******************************************************************
* G__smart_unload()
******************************************************************/
void G__smart_unload(int ifn)
{
  struct G__dictposition *dictpos= G__srcfile[ifn].dictpos;
  struct G__dictposition *hasonlyfunc = G__srcfile[ifn].hasonlyfunc;
  struct G__ifunc_table_internal *ifunc;
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
  ifunc = G__get_ifunc_internal(dictpos->ifunc);
  ifn = dictpos->ifn;
  while(ifunc && (ifunc!=G__get_ifunc_internal(hasonlyfunc->ifunc) || ifn!=hasonlyfunc->ifn)) {
    ifunc->hash[ifn] = 0;
    if(++ifn>=G__MAXIFUNC) {
      ifunc = ifunc->next;
      ifn = 0;
    }
  }

  /* disable file entry */
  ++G__srcfile_serial;
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


/******************************************************************
* G__unloadfile(filename)
*
*  1) check if function is busy. If busy return -1
*  2) Unload file and return 0
*
******************************************************************/
int G__unloadfile(const char *filename)
{
  int ifn;
  int i1=0;
  int i2;
  int hash;
  /* int from = -1 ,to = -1, next; */
  int flag;
  G__FastAllocString buf(filename);
  char *fname;
  char *scope;
  int envtagnum;

  G__LockCriticalSection();

  fname = (char*)G__strrstr(buf,"::");
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
        G__UnlockCriticalSection();
        return(G__UNLOADFILE_FAILURE);
      }
    }
  }
  else {
    fname = (char*)filename;
    envtagnum = -1;
  }

  /******************************************************************
  * check if file is already loaded.
  * if not so, return
  ******************************************************************/
  G__hash(fname,hash,i2);

  flag=0;
  while(i1<G__nfile) {
    if(G__matchfilename(i1,fname)
       && (-1==envtagnum||(envtagnum==G__srcfile[i1].parent_tagnum))){
      flag=1;
      break;
    }
    i1++;
  }

  if(flag==0) {
    G__fprinterr(G__serr,"Error: G__unloadfile() File \"%s\" not loaded ",filename);
    G__genericerror((char*)NULL);
    G__UnlockCriticalSection();
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
    G__UnlockCriticalSection();
    return(G__UNLOADFILE_FAILURE);
  }

  if(G__srcfile[ifn].hasonlyfunc && G__do_smart_unload) {
    G__smart_unload(ifn);
  }
  else {
    G__scratch_upto(G__srcfile[ifn].dictpos);
  }

  if(G__debug) {
    G__fprinterr(G__serr,"File=%s unloaded\n",filename);
  }

  G__UnlockCriticalSection();
  return(G__UNLOADFILE_SUCCESS);

}

/******************************************************************
* G__isbinaryfile()
*
******************************************************************/
static int G__isbinaryfile(char *filename)
{
  int c;
  int prev=0;
  int i;
  int badflag=0;
  int comflag=0;
#ifdef G__VISUAL /* ON959 */
  char buf[11];
#endif
  int unnamedmacro = 0;
  int alphaflag=0;
  short store_lang = G__lang;

  if(G__ONEBYTE!=G__lang) G__lang = G__UNKNOWNCODING;

  /* Read 10 byte from beginning of the file.
   * Set badflag if unprintable char is found. */
  for(i=0;i<10;i++) {
    c=fgetc(G__ifile.fp);
    if(G__IsDBCSLeadByte(c)) {
      c=fgetc(G__ifile.fp);
      if(c!=EOF) G__CheckDBCS2ndByte(c);
    } else
    if(!isprint(c) && '\t'!=c && '\n'!=c && '\r'!=c && EOF!=c && 0==comflag) {
      ++badflag;
    }
    else if('/'==prev && ('/'==c||'*'==c)) {
      comflag=1; /* set comment flag */
    }
    else if('{'==c && 0==alphaflag && 0==comflag) {
      unnamedmacro=1;
    }
    else if(isalpha(c)) {
      ++alphaflag;
    }
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
    G__lang = store_lang;
    return(1);
  }
  else if(unnamedmacro) {
    G__fprinterr(G__serr,"Error: Bad source file(unnamed macro) %s",filename);
    G__genericerror((char*)NULL);
    G__fprinterr(G__serr,"  unnamed macro has to be executed by 'x' command\n");
    G__return=G__RETURN_EXIT1;
    G__lang = store_lang;
    return(1);
  }
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
  G__lang = store_lang;
  return(0);
}

/******************************************************************
*  G__checkIfOnlyFunction()
******************************************************************/
static void G__checkIfOnlyFunction(int fentry)
{
  struct G__var_array *var;
  struct G__Deffuncmacro *deffuncmacro;
  struct G__Definedtemplateclass *definedtemplateclass;
  struct G__Definetemplatefunc *definedtemplatefunc;
  struct G__dictposition* dictpos = G__srcfile[fentry].dictpos;
  if (!dictpos) return;

  int varflag = 1;
  int tagflag;

  // Sum the number of G__struct slot used by any of the file
  // we enclosed:
  int nSubdefined = 0;
  for(int filecur = fentry+1; filecur < G__nfile; ++filecur) {
    nSubdefined += G__srcfile[filecur].definedStruct;
  }
  G__srcfile[fentry].definedStruct = G__struct.nactives - dictpos->nactives - nSubdefined;
  
  tagflag = ( G__srcfile[fentry].definedStruct == 0 ); 

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
     tagflag &&
     dictpos->typenum == G__newtype.alltype &&
     varflag &&
     dictpos->deffuncmacro == deffuncmacro &&
     dictpos->definedtemplateclass == definedtemplateclass &&
     dictpos->definedtemplatefunc == definedtemplatefunc) {
    G__srcfile[fentry].hasonlyfunc =
      (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
    G__store_dictposition(G__srcfile[fentry].hasonlyfunc);
  } 
}

/******************************************************************
* G__loadfile_tmpfile(fp)
*
******************************************************************/
int G__loadfile_tmpfile(FILE *fp)
{
  int store_prerun;
  struct G__var_array *store_p_local;
  char store_var_type;
  int store_tagnum,store_typenum;
  int fentry;
  int store_nobreak;
  int store_step;
  struct G__input_file store_file;
  int store_macroORtemplateINfile;
  short store_iscpp;
  G__UINT32 store_security;
  int store_func_now;
  short pragmacompile_iscpp;
  int pragmacompile_filenum;
  int store_asm_noverflow;
  int store_no_exec_compile;
  int store_asm_exec;
  int store_return;
  long store_struct_offset;
  int hash,temp;
  char hdrprop = G__NONCINTHDR;

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

  G__LockCriticalSection();

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
  G__snprintf(G__ifile.name,sizeof(G__ifile.name),"(tmp%d)",fentry);
  G__hash(G__ifile.name,hash,temp);

  G__srcfile[fentry].dictpos
    = (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
  G__store_dictposition(G__srcfile[fentry].dictpos);

  G__srcfile[fentry].hdrprop = hdrprop;

  store_security = G__security;
  G__srcfile[fentry].security = G__security;

  G__srcfile[fentry].prepname = (char*)NULL;
  G__srcfile[fentry].hash = hash;
  G__srcfile[fentry].filename = (char*)malloc(strlen(G__ifile.name)+1);
  strcpy(G__srcfile[fentry].filename,G__ifile.name); // Okay, we allocated the right size
  G__srcfile[fentry].fp=G__ifile.fp;

  G__srcfile[fentry].included_from = store_file.filenum;

  G__srcfile[fentry].ispermanentsl = G__ispermanentsl;
  G__srcfile[fentry].initsl = 0;
  G__srcfile[fentry].hasonlyfunc = (struct G__dictposition*)NULL;
  G__srcfile[fentry].definedStruct = 0;
  G__srcfile[fentry].parent_tagnum = G__get_envtagnum();
  G__srcfile[fentry].slindex = -1;

  ++G__nfile;
  ++G__srcfile_serial;

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
  while (!G__eof && (G__return < G__RETURN_EXIT1)) {
     int brace_level = 0;
     G__exec_statement(&brace_level);
  }


  /******************************************************
   * restore parser parameters
   ******************************************************/
  G__store_struct_offset = store_struct_offset;
  pragmacompile_filenum = G__ifile.filenum;
  pragmacompile_iscpp = G__iscpp;
  G__func_now = store_func_now;
  G__macroORtemplateINfile = store_macroORtemplateINfile;
  G__var_type = store_var_type;
  G__tagnum = store_tagnum;
  G__typenum = store_typenum;

  G__nobreak=store_nobreak;
  G__prerun=store_prerun;
  G__p_local=store_p_local;

  G__asm_noverflow = store_asm_noverflow;
  G__no_exec_compile = store_no_exec_compile;
  G__asm_exec = store_asm_exec;

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
    G__UnlockCriticalSection();
    return(G__LOADFILE_FAILURE);
  }


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

  G__checkIfOnlyFunction(fentry);

  G__UnlockCriticalSection();
  return(fentry+2);
}
   
// *****************************************************************
// * G__statfilename(filename)
// *
// * Use the same search algorithm as G__loadfile to do a 'stat' on the file
// *
// *****************************************************************

int G__statfilename(const char *filenamein, struct stat *statBuf)
{
   G__FastAllocString filename(filenamein);
   G__FastAllocString workname(G__ONELINE);
   int hash,temp;
   char addpost[3][8];
   int res = -1;
   
   /*************************************************
    * delete space chars at the end of filename
    *************************************************/
   size_t len = strlen(filename);
   while(len>1&&isspace(filename[len-1])) {
      filename[--len]='\0';
   }
   
   G__hash(filename,hash,temp);

   G__strlcpy(addpost[0],"",G__LENPOST);
   G__strlcpy(addpost[1],".h",G__LENPOST);
   
   G__strlcpy(addpost[2],"",G__LENPOST);
   for(int i2=0;i2<3;i2++) {
      if(2==i2) {
         if((len>3&& (strcmp(filename+len-3,".sl")==0 ||
                      strcmp(filename+len-3,".dl")==0 ||
                      strcmp(filename+len-3,".so")==0))) {
            filename[len - 3] = 0;
            filename += G__getmakeinfo1("DLLPOST");
         }
         else if((len>4&& (strcmp(filename+len-4,".dll")==0 ||
                           strcmp(filename+len-4,".DLL")==0))) {
            filename[len - 4] = 0;
            filename += G__getmakeinfo1("DLLPOST");
         }
         else if((len>2&& (strcmp(filename+len-2,".a")==0 ||
                           strcmp(filename+len-2,".A")==0))) {
            filename[len - 2] = 0;
            filename += G__getmakeinfo1("DLLPOST");
         }
#if defined(R__FBSD) || defined(R__OBSD)
         else if (len>strlen(soext) &&
                  strcmp(filename+len-strlen(soext),soext)==0) {
            filename[len - strlen(soext)] = 0;
            filename += G__getmakeinfo1("DLLPOST");
         }
#endif
      }
      
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
          G__ifile.name[0] != '\0') {
         char* p;
         workname = G__ifile.name;
#ifdef G__WIN32
         p = strrchr (workname, '/');
         {
            char* q = strrchr (workname, '\\');
            if (q && q > p)
               p = q;
         }
#else
         p = strrchr (workname, G__psep[0]);
#endif
         if (p == 0) p = workname;
         else ++p;
         G__strlcpy (p, filename, G__MAXFILENAME-(p-G__ifile.name));
         G__strlcat (p, addpost[i2], G__MAXFILENAME-(p-G__ifile.name));
         
         res = stat( workname, statBuf );
         
         if (res==0) return res;
      }
      /**********************************************
       * try ./filename
       **********************************************/
      if(G__USERHEADER==G__kindofheader) {
         workname.Format("%s%s",filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }  else {
         // Do we need that or not?
         // G__kindofheader = G__USERHEADER;
      }
      /**********************************************
       * try includepath/filename
       **********************************************/
      struct G__includepath *ipath = &G__ipathentry;
      while(res!=0 && ipath->pathname) {
         workname.Format("%s%s%s%s"
                         ,ipath->pathname,G__psep,filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         ipath = ipath->next;
      }
      if (res==0) return res;
      
      G__getcintsysdir();

      /**********************************************
       * try $CINTSYSDIR/stl
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("%s%s%s%sstl%s%s%s",G__cintsysdir,G__psep,G__CFG_COREVERSION
                         ,G__psep,G__psep,filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }

      /**********************************************
       * try $CINTSYSDIR/lib
       **********************************************/
      /* G__getcintsysdir(); */
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("%s%s%s%slib%s%s%s",G__cintsysdir,G__psep,G__CFG_COREVERSION
                         ,G__psep,G__psep,filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }

#ifdef G__EDU_VERSION
      /**********************************************
       * try include/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("include%s%s%s"
                         ,G__psep,filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
      /**********************************************
       * try stl
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("stl%s%s%s"
                         ,G__psep,filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }         
#endif /* G__EDU_VERSION */
      
#ifdef G__VISUAL
      /**********************************************
       * try /msdev/include
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("/msdev/include/%s%s",filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif /* G__VISUAL */
         
#ifdef G__SYMANTEC
      /**********************************************
       * try /sc/include
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("/sc/include/%s%s",filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif // G__SYMANTEC
         
#ifndef G__WIN32
      /**********************************************
       * try /usr/include/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("/usr/include/%s%s",filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif
      
#ifdef __GNUC__
      /**********************************************
       * try /usr/include/g++/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("/usr/include/g++/%s%s",filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif /* __GNUC__ */
      
#ifndef G__WIN32
      /* #ifdef __hpux */
      /**********************************************
       * try /usr/include/CC/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("/usr/include/CC/%s%s",filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }         
#endif
         
#ifndef G__WIN32
      /**********************************************
       * try /usr/include/codelibs/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         workname.Format("/usr/include/codelibs/%s%s"
                         ,filename(),addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif
   }
   return -1;
}

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
int G__loadfile(const char *filenamein)
{
  FILE *tmpfp;
  int external_compiler = 0;
  const char* compiler_option = "";
  int store_prerun;
  short i1=0;
  struct G__var_array *store_p_local;
  char store_var_type;
  int store_tagnum,store_typenum;
  short fentry;
  struct G__includepath *ipath;
  int store_nobreak;
#ifdef G__TMPFILE
  G__FastAllocString prepname(G__MAXFILENAME);
#else
  G__FastAllocString prepname(L_tmpnam+10);
#endif
  int store_step;
  short null_entry = -1;
  struct G__input_file store_file;
  int hash;
  int temp;
  int store_macroORtemplateINfile;
#ifdef G__SHAREDLIB
  size_t len1;
  const char *dllpost;
#endif //G__SHAREDLIB
  short store_iscpp;
  G__UINT32 store_security;
  char addpost[3][8];
  int i2;
  int store_func_now;
  short pragmacompile_iscpp;
  int pragmacompile_filenum;
  int store_asm_noverflow;
  int store_no_exec_compile;
  int store_asm_exec;
#if defined(R__FBSD) || defined(R__OBSD)
  char soext[]=SOEXT;
#endif
  char hdrprop = G__NONCINTHDR;
  G__FastAllocString filename(filenamein);


  /*************************************************
  * delete space chars at the end of filename
  *************************************************/
  size_t len = strlen(filename);
  while(len>1&&isspace(filename[len-1])) {
    filename[--len]='\0';
  }

  /*************************************************
  * Check if the filename as an extension ending in
  * ++, like script.cxx++ or script.C++
  * ending with only one + means to keep the shared
  * library after the end of this process.
  * The + or ++ can also be followed by either a 'g'
  * or an 'O' which means respectively to compile
  * in debug or optimized mode.
  *************************************************/
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
      if (compiler_option) {
         switch(compiler_option[0]) {
            case 'O': compiler_option = "kfO"; break;
            case 'g': compiler_option = "kfg"; break;
            default: G__genericerror("Should not have been reached!");
         }
      } else {
         compiler_option = "kf";
      }
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

  G__LockCriticalSection();

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
    G__UnlockCriticalSection();
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
    if(G__matchfilename(i1,filename)
       &&G__get_envtagnum()==G__srcfile[i1].parent_tagnum
       ){
      if(G__prerun==0 || G__debugtrace)
        if(G__dispmsg>=G__DISPNOTE) {
            static const char *excludelist [] = {
                "stdfunc.dll","stdcxxfunc.dll","posix.dll","ipc.dll","posix.dll"
                "string.dll","vector.dll","vectorbool.dll","list.dll","deque.dll",
                "map.dll", "map2.dll","set.dll","multimap.dll","multimap2.dll",
                "multiset.dll","stack.dll","queue.dll","valarray.dll",
                "exception.dll","stdexcept.dll","complex.dll","climits.dll" };
            static const unsigned int excludelistsize = sizeof(excludelist)/sizeof(excludelist[0]);
            static int excludelen[excludelistsize] = {-1};
            if (excludelen[0] == -1) {
               for (unsigned int i = 0; i < excludelistsize; ++i)
                  excludelen[i] = (int)strlen(excludelist[i]);
            }
            bool cintdlls = false;
            long filelen = strlen(filename);
            for (unsigned int i = 0; !cintdlls && i < excludelistsize; ++i) {
               if (filelen>=excludelen[i]) {
                  cintdlls = (!strncmp(filename+filelen-excludelen[i], excludelist[i], excludelen[i]));
               }
            }
            if (!cintdlls) G__fprinterr(G__serr,"Note: File \"%s\" already loaded\n",filename());
        }
      /******************************************************
       * restore input file information to G__ifile
       * and reset G__eof to 0.
       ******************************************************/
      G__ifile = store_file ;
      G__eof = 0;
      G__step=store_step;
      G__setdebugcond();
      G__UnlockCriticalSection();
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
  std::string macros(G__macros);
  macros += " -DG__EXTERNAL_CPP ";
  int pres= G__preprocessor(prepname,filename,G__cpp,macros.c_str(),G__undeflist
                            ,G__ppopt,G__allincludepath);
  if (pres!=0) {
     G__fprinterr(G__serr,"Error: external preprocessing failed.");
     G__genericerror((char*)NULL);
      G__UnlockCriticalSection();
     return(G__LOADFILE_FAILURE);
  }

  /**********************************************
  * open file
  **********************************************/
  if(prepname[0]) {
    /**********************************************
     * -p option. open preprocessed tmpfile
     **********************************************/
     G__strlcpy(G__ifile.name, filename, G__MAXFILENAME);
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
      if(G__ifile.fp) {
        remove(prepname);
        prepname = "(tmpfile)";
      }
      else {
#ifndef G__WIN32
        G__ifile.fp = fopen(prepname,"r");
#else /* G__WIN32 */
        G__ifile.fp = fopen(prepname,"rb");
#endif /* G__WIN32 */
      }
    }
#else /* 1922 */
    G__ifile.fp = G__copytotmpfile(prepname);
    remove(prepname);
    prepname = "(tmpfile)";
#endif /* 1922 */
    G__kindofheader = G__USERHEADER;
  }
  else {
    G__strlcpy(addpost[0],"",G__LENPOST);
    G__strlcpy(addpost[1],".h",G__LENPOST);

    G__strlcpy(addpost[2],"",G__LENPOST);
    for(i2=0;i2<3;i2++) {
      if(2==i2) {
        if((len>3&& (strcmp(filename+len-3,".sl")==0 ||
                     strcmp(filename+len-3,".dl")==0 ||
                     strcmp(filename+len-3,".so")==0))) {
           filename[len - 3] = 0;
           filename += G__getmakeinfo1("DLLPOST");
        }
        else if((len>4&& (strcmp(filename+len-4,".dll")==0 ||
                          strcmp(filename+len-4,".DLL")==0))) {
           filename[len - 4] = 0;
           filename += G__getmakeinfo1("DLLPOST");
        }
        else if((len>2&& (strcmp(filename+len-2,".a")==0 ||
                          strcmp(filename+len-2,".A")==0))) {
           filename[len - 2] = 0;
           filename += G__getmakeinfo1("DLLPOST");
        }
#if defined(R__FBSD) || defined(R__OBSD)
        else if (len>strlen(soext) &&
                 strcmp(filename+len-strlen(soext),soext)==0) {
           filename[len - strlen(soext)] = 0;
           filename += G__getmakeinfo1("DLLPOST");
        }
#endif
      }

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
        G__strlcpy (G__ifile.name, store_file.name,G__MAXFILENAME);
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
        G__strlcpy (p, filename,G__MAXFILENAME-(p-G__ifile.name));
        G__strlcat (p, addpost[i2],G__MAXFILENAME-(p-G__ifile.name));
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
      }
      if (G__ifile.fp) break;
      /**********************************************
       * try ./filename
       **********************************************/
      if(G__USERHEADER==G__kindofheader && filename()[0]!='/' && filename()[0]!='\\') {
        G__snprintf(G__ifile.name,G__MAXFILENAME,"./%s%s",filename(),addpost[i2]);
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
       * try filename
       **********************************************/
      if(G__USERHEADER==G__kindofheader) {
         snprintf(G__ifile.name,G__MAXFILENAME,"%s%s",filename(),addpost[i2]);
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
        G__snprintf(G__ifile.name,G__MAXFILENAME,"%s%s%s%s"
                ,ipath->pathname,G__psep,filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        ipath = ipath->next;
        {
          struct G__ConstStringList* sysdir = G__SystemIncludeDir;
          while(sysdir) {
            if(strncmp(sysdir->string,G__ifile.name,sysdir->hash)==0) {
              G__globalcomp=G__NOLINK;
              hdrprop = G__CINTHDR;
            }
            sysdir = sysdir->prev;
          }
        }
      }
      if(G__ifile.fp) break;

      /**********************************************
       * try $CINTSYSDIR/include/filename
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
        G__snprintf(G__ifile.name,G__MAXFILENAME,"%s%s%s%sinclude%s%s%s",G__cintsysdir,G__psep,G__CFG_COREVERSION
                ,G__psep,G__psep,filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        if(G__ifile.fp && G__autoload_stdheader) {
          G__globalcomp=G__store_globalcomp;
          G__gen_linksystem(filename);
        }
        hdrprop = G__CINTHDR;
        G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;

      /**********************************************
       * try $CINTSYSDIR/stl
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
        G__snprintf(G__ifile.name,G__MAXFILENAME,"%s%s%s%sstl%s%s%s",G__cintsysdir, G__psep, G__CFG_COREVERSION,
                G__psep, G__psep, filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        if(G__ifile.fp && G__autoload_stdheader) {
          G__globalcomp=G__store_globalcomp;
          G__gen_linksystem(filename);
        }
        hdrprop = G__CINTHDR;
        G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;

      /**********************************************
       * try $CINTSYSDIR/lib
       **********************************************/
      /* G__getcintsysdir(); */
      if('\0'!=G__cintsysdir[0]) {
        G__snprintf(G__ifile.name,G__MAXFILENAME,"%s%s%s%slib%s%s%s",G__cintsysdir, G__psep, G__CFG_COREVERSION,
                G__psep, G__psep,filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        /* G__globalcomp=G__NOLINK; */
      }
      if(G__ifile.fp) break;

#ifdef G__EDU_VERSION
      /**********************************************
       * try include/filename
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
        G__snprintf(G__ifile.name,G__MAXFILENAME,"include%s%s%s"
                ,G__psep,filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        if(G__ifile.fp && G__autoload_stdheader) {
          G__globalcomp=G__store_globalcomp;
          G__gen_linksystem(filename);
        }
        hdrprop = G__CINTHDR;
        G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;

      /**********************************************
       * try stl
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
        G__snprintf(G__ifile.name,G__MAXFILENAME,"stl%s%s%s"
                ,G__psep,filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        if(G__ifile.fp && G__autoload_stdheader) {
          G__globalcomp=G__store_globalcomp;
          G__gen_linksystem(filename);
        }
        hdrprop = G__CINTHDR;
        G__globalcomp=G__NOLINK;
      }
      if(G__ifile.fp) break;
#endif /* G__EDU_VERSION */

#ifdef G__VISUAL
      /**********************************************
       * try /msdev/include
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         G__snprintf(G__ifile.name,G__MAXFILENAME,"/msdev/include/%s%s",filename(),addpost[i2]);
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
         G__snprintf(G__ifile.name,G__MAXFILENAME,"/sc/include/%s%s",filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
#endif /* G__SYMANTEC */

#ifndef G__WIN32
      /**********************************************
       * try /usr/include/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         G__snprintf(G__ifile.name,G__MAXFILENAME,"/usr/include/%s%s",filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
#endif

#ifdef __GNUC__
      /**********************************************
       * try /usr/include/g++/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         G__snprintf(G__ifile.name,G__MAXFILENAME,"/usr/include/g++/%s%s",filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
#endif /* __GNUC__ */

#ifndef G__WIN32
/* #ifdef __hpux */
      /**********************************************
       * try /usr/include/CC/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         G__snprintf(G__ifile.name,G__MAXFILENAME,"/usr/include/CC/%s%s",filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
/* #endif __hpux */
#endif

#ifndef G__WIN32
/* #ifdef __hpux */
      /**********************************************
       * try /usr/include/codelibs/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
        G__snprintf(G__ifile.name,G__MAXFILENAME,"/usr/include/codelibs/%s%s"
                    ,filename(),addpost[i2]);
#ifndef G__WIN32
        G__ifile.fp = fopen(G__ifile.name,"r");
#else
        G__ifile.fp = fopen(G__ifile.name,"rb");
#endif
        G__globalcomp=G__store_globalcomp;
      }
      if(G__ifile.fp) break;
/* #endif __hpux */
#endif
    }
  }

  /***************************************************
  * check if alreay loaded
  ***************************************************/
  for(int otherfile = 0; otherfile < G__nfile; ++otherfile) {
     if(G__matchfilename(otherfile,G__ifile.name)
        &&G__get_envtagnum()==G__srcfile[otherfile].parent_tagnum
        ){
       if(G__prerun==0 || G__debugtrace)
         if(G__dispmsg>=G__DISPNOTE) {
           G__fprinterr(G__serr,"Note: File \"%s\" already loaded\n",G__ifile.name);
         }
       /******************************************************
        * restore input file information to G__ifile
        * and reset G__eof to 0.
        ******************************************************/
       fclose(G__ifile.fp);
       G__ifile = store_file ;
       G__eof = 0;
       G__step=store_step;
       G__setdebugcond();
       G__UnlockCriticalSection();
       return(G__LOADFILE_DUPLICATE);
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
    if(0==G__ispragmainclude) {
       G__fprinterr(G__serr,"Error: cannot open file \"%s\" ", filename());
      G__genericerror((char*)NULL);
    }
    G__iscpp=store_iscpp;
    G__UnlockCriticalSection();
    return(G__LOADFILE_FAILURE);
  }
  else {
    if(G__ignoreinclude && (*G__ignoreinclude)(filename,G__ifile.name)) {
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
      G__UnlockCriticalSection();
      return(G__LOADFILE_SUCCESS);
    }

    /********************************************
     * if there is null_entry which has been
     * unloaded, use that index.
     ********************************************/
    if(null_entry == -1) {
      fentry = G__nfile;
      // ++G__nfile; after we've stored the dict pos!
    }
    else {
      fentry=null_entry;
    }
    // G__ignoreinclude might have loaded some more libs,
    // so update G__ifile to point to fentry:
    G__ifile.filenum = fentry;
    G__srcfile[fentry].dictpos
      = (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
    G__store_dictposition(G__srcfile[fentry].dictpos);
    if(null_entry == -1) {
       ++G__nfile;
       ++G__srcfile_serial;
    }

    /***************************************************
     * set
     *  char  G__filenameary[][]
     *  FILE *G__filearray
     *  int  G__nfile
     ***************************************************/

    G__srcfile[fentry].hdrprop = hdrprop;

#ifdef G__SECURITY
    store_security = G__security;
    G__srcfile[fentry].security = G__security;
#endif

    G__srcfile[fentry].hash=hash;
    if(prepname[0]) {
      G__srcfile[fentry].prepname = (char*)malloc(strlen(prepname)+1);
      strcpy(G__srcfile[fentry].prepname,prepname); // Okay, we allocated the right size
    }
    else {
      G__srcfile[fentry].prepname = (char*)NULL;
    }
    if(G__globalcomp<G__NOLINK) {
      G__srcfile[fentry].filename = (char*)malloc(strlen(G__ifile.name)+1);
      strcpy(G__srcfile[fentry].filename,G__ifile.name); // Okay, we allocated the right size
    }
    else {
      G__srcfile[fentry].filename = (char*)malloc(strlen(filename)+1);
      strcpy(G__srcfile[fentry].filename,filename); // Okay, we allocated the right size
    }
    G__srcfile[fentry].fp=G__ifile.fp;
    G__srcfile[fentry].included_from = store_file.filenum;
    G__srcfile[fentry].ispermanentsl = G__ispermanentsl;
    G__srcfile[fentry].initsl = 0;
    G__srcfile[fentry].hasonlyfunc = (struct G__dictposition*)NULL;
    G__srcfile[fentry].definedStruct = 0;
    G__srcfile[fentry].parent_tagnum = G__get_envtagnum();
    G__srcfile[fentry].slindex = -1;
    G__srcfile[fentry].breakpoint = 0;
  }

  if(G__debugtrace) {
     G__fprinterr(G__serr,"LOADING file=%s:%s:%s\n",filename(),G__ifile.name,prepname());
  }
  if(G__debug) {
    G__fprinterr(G__serr,"%-5d",G__ifile.line_number);
  }

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

#ifdef G__SHAREDLIB
  len = strlen(filename);
  dllpost = G__getmakeinfo1("DLLPOST");
  if((len>3&& (strcmp(filename+len-3,".sl")==0 ||
               strcmp(filename+len-3,".dl")==0 ||
               strcmp(filename+len-3,".so")==0)) ||
     (len>4&& (strcmp(filename+len-4,".dll")==0 ||
               strcmp(filename+len-4,".DLL")==0)) ||
#if defined(R__FBSD) || defined(R__OBSD)
     (len>strlen(soext) && strcmp(filename+len-strlen(soext), soext)==0) ||
#endif
     (
      dllpost[0] &&
      len>(len1=strlen(dllpost)) && strcmp(filename+len-len1,dllpost)==0) ||
     (len>2&& (strcmp(filename+len-2,".a")==0 ||
               strcmp(filename+len-2,".A")==0))
     ) {
    /* Caution, G__ifile.fp is left openned.
     * This may cause trouble in future */
    fclose(G__srcfile[fentry].fp);
    if (G__ifile.fp == G__srcfile[fentry].fp) {
      /* Since the file is closed, the FILE* pointer is now invalid and thus
         we have to remove it from G__ifile! */
      G__ifile.fp=(FILE*)NULL;
    }
    G__srcfile[fentry].fp=(FILE*)NULL;
    G__ifile.line_number = -1;
    std::list<G__DLLINIT>* store_initpermanentsl = 0;
    if (G__initpermanentsl && !G__initpermanentsl->empty()) {
       store_initpermanentsl = G__initpermanentsl;
       G__initpermanentsl = new std::list<G__DLLINIT>;
    }
    int store_nlibs_highwatermark = G__nlibs_highwatermark;
    G__nlibs_highwatermark = G__nlibs;
    {
#if !defined(ROOTBUILD) && !defined(G__BUILDING_CINTTMP)
      int allsl = G__shl_load(G__ifile.name);
      if (allsl != -1) {
         G__srcfile[fentry].slindex = allsl;
      }
#else
      // don't load any shared libs
      // G__srcfile[fentry].slindex stays at -1
#endif
    }
    G__nlibs_highwatermark = store_nlibs_highwatermark;
    if (G__initpermanentsl) {
       if(G__ispermanentsl) {
          if (!G__srcfile[fentry].initsl)
             G__srcfile[fentry].initsl = new std::list<G__DLLINIT>;
          G__srcfile[fentry].initsl->insert(G__srcfile[fentry].initsl->end(),
                                           G__initpermanentsl->begin(),
                                           G__initpermanentsl->end());
       }
       G__initpermanentsl->clear();
    }
    if (store_initpermanentsl) {
       delete G__initpermanentsl;
       G__initpermanentsl = store_initpermanentsl;
    }
  }
  else {
    if(G__globalcomp>1 && strcmp(filename+strlen(filename)-4,".sut")==0) {
      G__ASSERT(G__sutpi && G__ifile.fp);
      G__copyfile(G__sutpi,G__ifile.fp);
    }
    else {
      long store_struct_offset = G__store_struct_offset;
      G__store_struct_offset = 0;
      if(G__isbinaryfile(filename)) {
        G__iscpp=store_iscpp;
#ifdef G__SECURITY
        G__security = store_security;
#endif
        G__UnlockCriticalSection();
        return(G__LOADFILE_FAILURE);
      }
      if (G__copyflag) {
         G__copysourcetotmp(prepname, &G__ifile, fentry);
      }
      while (!G__eof && (G__return < G__RETURN_EXIT1)) {
         int brace_level = 0;
         G__exec_statement(&brace_level);
      }
      G__store_struct_offset = store_struct_offset;
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
      G__UnlockCriticalSection();
      return(G__LOADFILE_FAILURE);
    }
    while (!G__eof && (G__return < G__RETURN_EXIT1)) {
       int brace_level = 0;
       G__exec_statement(&brace_level);
    }
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
      /* After closing the file let's make sure that all reference to
         the file pointer are reset. When preprocessor is used, we
         will have several logical file packed in one file. */
      tmpfp = G__srcfile[fentry].fp;
      for(i1=0;i1<G__nfile;i1++) {
        if (G__srcfile[i1].fp==tmpfp){
          G__srcfile[i1].fp = (FILE*)NULL;
        }
      }
      /* Close file for process max file open limitation with -cN option */
      fclose(tmpfp);
    }
  }


  /******************************************************
   * restore parser parameters
   ******************************************************/
  pragmacompile_filenum = G__ifile.filenum;
  pragmacompile_iscpp = G__iscpp;
  G__func_now = store_func_now;
  G__macroORtemplateINfile = store_macroORtemplateINfile;
  G__var_type = store_var_type;
  G__tagnum = store_tagnum;
  G__typenum = store_typenum;

  G__nobreak=store_nobreak;
  G__prerun=store_prerun;
  G__p_local=store_p_local;

  G__asm_noverflow = store_asm_noverflow;
  G__no_exec_compile = store_no_exec_compile;
  G__asm_exec = store_asm_exec;

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
    G__UnlockCriticalSection();
    return(G__LOADFILE_FAILURE);
  }


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

  G__checkIfOnlyFunction(fentry);

  G__UnlockCriticalSection();
  return(G__LOADFILE_SUCCESS);
}

/**************************************************************************
* G__setfilecontext()
*
* Set the current G__ifile to filename, allocate an entry in G__srcfiles
* if necessary. Set ifile to the previous G__ifile to it can be popped.
* This function does not load any file, it just provides a valid G__ifile
* context for type manipulations.
*
* Returns 0 in case of error.
* Returns 1 if file entry is newly allocated, 2 otherwise.
*
**************************************************************************/

int  G__setfilecontext(const char* filename, G__input_file* ifile)
{
   if (!filename) return 0;
   
   short null_entry = -1;
   short found_entry = -1;
   // find G__srcfile index matching filename
   for (short i = 0; (found_entry == -1) && i < G__nfile; ++i)
      if (G__srcfile[i].filename) {
          if (!strcmp(G__srcfile[i].filename, filename))
             found_entry = i;
      }
      else if (null_entry == -1)
         null_entry = i;

   if (found_entry == -1) {
      short fentry = null_entry;
      if (fentry == -1)
         fentry = G__nfile;

      G__srcfile[fentry].dictpos
         = (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
      G__store_dictposition(G__srcfile[fentry].dictpos);
      if (null_entry == -1) {
         ++G__nfile;
         ++G__srcfile_serial;
      }

#ifdef G__SECURITY
      G__srcfile[fentry].security = G__security;
#endif
      int hash = 0;
      int temp = 0;
      G__hash(filename,hash,temp);
      G__srcfile[fentry].hash = hash;
      G__srcfile[fentry].prepname = NULL;
      G__srcfile[fentry].filename = (char*)malloc(strlen(filename)+1);
      strcpy(G__srcfile[fentry].filename,filename); // Okay, we allocated the right size
      G__srcfile[fentry].fp = 0;
      G__srcfile[fentry].included_from = G__ifile.filenum;
      G__srcfile[fentry].ispermanentsl = 1;
      G__srcfile[fentry].initsl = 0;
      G__srcfile[fentry].hasonlyfunc = (struct G__dictposition*)NULL;
      G__srcfile[fentry].definedStruct = 0;
      G__srcfile[fentry].parent_tagnum = G__get_envtagnum();
      G__srcfile[fentry].slindex = -1;
      G__srcfile[fentry].breakpoint = 0;
      found_entry = fentry;
   }

   if (ifile) *ifile = G__ifile;
   G__ifile.fp = G__srcfile[found_entry].fp;
   G__ifile.filenum = found_entry;
   G__strlcpy(G__ifile.name, G__srcfile[found_entry].filename,G__MAXFILENAME);
   G__ifile.line_number = 0;
   G__ifile.str = 0;
   G__ifile.pos = 0;
   G__ifile.vindex = 0;
   if (found_entry != -1)
      return 2;
   return 1;
}


/**************************************************************************
* G__preprocessor()
*
* Use C/C++ preprocessor prior to interpretation.
*  Name of preprocessor must be defined in $CINTSYSDIR/MAKEINFO file as
*  CPPPREP and CPREP.
*
**************************************************************************/
int G__preprocessor(      char *outname,const char *inname,int cppflag
                   ,const char *macros,const char *undeflist,const char *ppopt
                   ,const char *includepath)
{
  /* char *envcpp; */
  G__FastAllocString tmpfile(G__MAXFILENAME);
  size_t tmplen;
  FILE *fp;
  int flag=0;
  char *post;

  size_t inlen = strlen(inname);
  post = (char*)strrchr(inname,'.');

  if(post && inlen>2) {
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
  if(flag==0 && post && inlen>3) {
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
  if(flag==0 && post && inlen>4) {
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
  if(flag==0 && post) {
    if('\0'==G__cppsrcpost[0]) {
      G__strlcpy(G__cppsrcpost,G__getmakeinfo1("CPPSRCPOST"),G__LENPOST);
    }
    if('\0'==G__csrcpost[0]) {
      G__strlcpy(G__csrcpost,G__getmakeinfo1("CSRCPOST"),G__LENPOST);
    }
    if('\0'==G__cpphdrpost[0]) {
      G__strlcpy(G__cpphdrpost,G__getmakeinfo1("CPPHDRPOST"),G__LENPOST);
    }
    if('\0'==G__chdrpost[0]) {
      G__strlcpy(G__chdrpost,G__getmakeinfo1("CHDRPOST"),G__LENPOST);
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
  else if(flag==0&&!post) {
    if(!G__clock) G__iscpp=1;
    flag=1;
  }

  /* If using C preprocessor '-p' option' */
  if(cppflag && flag) {

    /* Determine what C/C++ preprocessor to use */
    if('\0'==G__ccom[0]) {
      switch(G__globalcomp) {
      case G__CPPLINK: /* C++ link */
        G__strlcpy(G__ccom,G__getmakeinfo("CPPPREP"),G__MAXFILENAME);
        break;
      case G__CLINK: /* C link */
        G__strlcpy(G__ccom,G__getmakeinfo("CPREP"),G__MAXFILENAME);
        break;
      default:
        if(G__iscpp) G__strlcpy(G__ccom,G__getmakeinfo("CPPPREP"),G__MAXFILENAME);
        else         G__strlcpy(G__ccom,G__getmakeinfo("CPREP"),G__MAXFILENAME);
        break;
      }
      if('\0'==G__ccom[0]) {
#ifdef __GNUC__
        G__snprintf(G__ccom,G__MAXFILENAME,"g++ -E");
#else
        G__snprintf(G__ccom,G__MAXFILENAME,"CC -E");
#endif
      }
    }

    /* Get tmpfile name if necessary */
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
    {
      /* if header file, create tmpfile name as xxx.C */
      do {
        G__tmpnam(tmpfile); /* can't replace this with tmpfile() */
        tmplen=strlen(tmpfile);
        if(G__CPPLINK==G__globalcomp || G__iscpp) {
          if('\0'==G__cppsrcpost[0]) {
            G__strlcpy(G__cppsrcpost,G__getmakeinfo1("CPPSRCPOST"),G__LENPOST);
          }
          tmpfile += G__cppsrcpost;
        }
        else {
          if('\0'==G__csrcpost[0]) {
            G__strlcpy(G__csrcpost,G__getmakeinfo1("CSRCPOST"),G__LENPOST);
          }
          tmpfile += G__csrcpost;
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
      tmpfile = inname;
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

    G__FastAllocString temp(G__LARGEBUF);
#if defined(G__SYMANTEC)
    /**************************************************************
     * preprocessor statement for Symantec C++
     ***************************************************************/
    if(G__cintsysdir[0]) {
       temp.Format("%s %s %s -I. %s %s -D__CINT__ -I%s/%s/include -I%s/%s/stl -I%s/%s/lib %s -o%s"
                   ,G__ccom ,macros,undeflist,ppopt ,includepath,
                   ,G__cintsysdir, G__CFG_COREVERSION, G__cintsysdir,G__CFG_COREVERSION
                   ,G__cintsysdir, G__CFG_COREVERSION, tmpfile(),outname);
    }
    else {
       temp.Format("%s %s %s %s -I. %s -D__CINT__ %s -o%s" ,G__ccom
                   ,macros,undeflist,ppopt ,includepath ,tmpfile(),outname);
    }
#elif defined(G__BORLAND)
    /**************************************************************
     * #elif defined(G__BORLAND)
     *      preprocessor statement for Borland C++ to be implemented
     ***************************************************************/
    G__strlcat(outname,".i",G__MAXFILENAME);
    if(G__cintsysdir[0]) {
       temp.Format("%s %s %s -I. %s %s -D__CINT__ -I%s/%s/include -I%s/%s/stl -I%s/%s/lib -o%s %s"
                   ,G__ccom ,macros,undeflist,ppopt ,includepath
                   ,G__cintsysdir,G__CFG_COREVERSION,G__cintsysdir,G__CFG_COREVERSION,
                   G__cintsysdir,G__CFG_COREVERSION,outname,tmpfile());
    }
    else {
       temp.Format("%s %s %s %s -I. %s -D__CINT__ -o%s %s" ,G__ccom
                   ,macros,undeflist,ppopt ,includepath ,outname,tmpfile());
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
       temp.Format("%s %s %s -I. %s %s -D__CINT__ -I%s/%s/include -I%s/%s/stl -I%s/%s/lib %s > %s"
                   ,G__ccom ,macros,undeflist,ppopt ,includepath
                   ,G__cintsysdir,G__CFG_COREVERSION,G__cintsysdir,G__CFG_COREVERSION,
                   G__cintsysdir,G__CFG_COREVERSION,tmpfile(),outname);
    }
    else {
       temp.Format("%s %s %s %s -I. %s -D__CINT__ %s > %s" ,G__ccom
                   ,macros,undeflist,ppopt ,includepath ,tmpfile(),outname);
    }
#endif
     if(G__debugtrace||G__steptrace||G__step||G__asm_dbg) {
        G__fprinterr(G__serr," %s\n",temp.data());
     }
    int pres = system(temp);

    if(tmplen) remove(tmpfile);
    return pres;
  }

  else {
    /* Didn't use C preprocessor because cppflag is not set or file name
     * suffix does not match */
    outname[0]='\0';
  }
  return(0);
}


/**************************************************************************
* G__difffile()
**************************************************************************/
int G__difffile(char *file1,char *file2)
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

/**************************************************************************
* G__copyfile()
**************************************************************************/
int G__copyfile(FILE *to,FILE *from)
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
#else
static char G__mfpname[G__MAXFILENAME];
#endif


/**************************************************************************
* G__setTMPDIR()
**************************************************************************/
int G__setTMPDIR(char *badname)
{
#ifndef G__TMPFILE
  G__fprinterr(G__serr,"CAUTION: tmpfile %s can't open\n",badname);
  return(0);
#else
  char *p;
  G__fprinterr(G__serr,"CINT CAUTION: tmpfile %s can't open\n",badname);
  G__fprinterr(G__serr,"Input another temp directory or '*' to give up\n");
  G__fprinterr(G__serr,"(Setting CINTTMPDIR environment variable avoids this interrupt)\n");
  G__strlcpy(G__tmpdir,G__input("Input TMPDIR > "),G__MAXFILENAME);
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

class G__Tmpnam_Files {
public:
  G__Tmpnam_Files() {}
  ~G__Tmpnam_Files() {
    for (std::list<std::string>::iterator iFile=fFiles.begin();
        iFile!=fFiles.end(); ++iFile)
      unlink(iFile->c_str());
  }
  void Add(const char* name) {fFiles.push_back(name);}
  std::list<std::string> fFiles;
};

/**************************************************************************
* G__tmpnam()
**************************************************************************/
char* G__tmpnam(char *name)
{
  static G__Tmpnam_Files G__tmpfiles;
#if defined(G__TMPFILE)
  const char *appendix="_cint";
  static char tempname[G__MAXFILENAME];
  int pid = getpid();
  int now = clock();
  char *tmp;
  if('\0'==G__tmpdir[0]) {
    if((tmp=getenv("CINTTMPDIR"))) G__strlcpy(G__tmpdir,tmp,G__MAXFILENAME);
    else if((tmp=getenv("TEMP"))) G__strlcpy(G__tmpdir,tmp,G__MAXFILENAME);
    else if((tmp=getenv("TMP"))) G__strlcpy(G__tmpdir,tmp,G__MAXFILENAME);
    else G__strlcpy(G__tmpdir,".",G__MAXFILENAME);
  }
  if(name) {
    G__strlcpy(name,(tmp=tempnam(G__tmpdir,"")),G__MAXFILENAME);
    free((void*)tmp);
    if(strlen(name)<G__MAXFILENAME-10)
      G__snprintf(name+strlen(name),G__MAXFILENAME-strlen(name),"%d%d",pid%10000,now%10000);
    if(strlen(name)<G__MAXFILENAME-6) G__strlcat(name,appendix,G__MAXFILENAME);
    G__tmpfiles.Add(name);
    return(name);
  }
  else {
    G__strlcpy(tempname,(tmp=tempnam(G__tmpdir,"")),G__MAXFILENAME);
    free((void*)tmp);
    size_t lentemp=strlen(tempname);
    if(lentemp<G__MAXFILENAME-10)
      G__snprintf(tempname+lentemp,G__MAXFILENAME-lentemp,"%d%d",pid%10000,now%10000);
    if(strlen(tempname)<G__MAXFILENAME-strlen(appendix)-1)
      G__strlcat(tempname,appendix,G__MAXFILENAME);
    G__tmpfiles.Add(tempname);
    return(tempname);
  }

#elif defined(__CINT__)
  static char tempname[G__MAXFILENAME];
  const char *appendix="_cint";

  if (name==0) name = tempname;
  tmpnam(name);
  if(strlen(name)<G__MAXFILENAME-6) G__strlcat(name,appendix,G__MAXFILENAME);
  G__tmpfiles.Add(name);
  return(name);

#elif /*defined(G__NEVER) && */ ((__GNUC__>=3)||((__GNUC__>=2)&&(__GNUC_MINOR__>=96)))&&(defined(__linux)||defined(__linux__))
  /* After all, mkstemp creates more problem than a solution. */
  static char tempname[G__MAXFILENAME];
  const char *appendix="_cint";
  static char tmpdir[G__MAXFILENAME];

  char *tmp;
  if('\0'==tmpdir[0]) {
    if((tmp=getenv("CINTTMPDIR"))) G__strlcpy(tmpdir,tmp,G__MAXFILENAME);
    else if((tmp=getenv("TEMP"))) G__strlcpy(tmpdir,tmp,G__MAXFILENAME);
    else if((tmp=getenv("TMP"))) G__strlcpy(tmpdir,tmp,G__MAXFILENAME);
    else G__strlcpy(tmpdir,"/tmp",G__MAXFILENAME);
  }

  if (name==0) name = tempname;
  G__strlcpy(name, tmpdir, G__MAXFILENAME);
  G__strlcat(name,"/XXXXXX", G__MAXFILENAME);
  mode_t old_umask = umask(077); // be restrictive for mkstemp()
  int temp_fileno = mkstemp(name);/*mkstemp not only generate file name but also opens the file*/
  umask(old_umask);
  if (temp_fileno >= 0) {
     close(temp_fileno);
  }
  remove(name); /* mkstemp creates this file anyway. Delete it. questionable */
  if(strlen(name)<G__MAXFILENAME-6) G__strlcat(name,appendix,G__MAXFILENAME);
  G__tmpfiles.Add(name);
  return(name);

#else
  static char tempname[G__MAXFILENAME];
  const char *appendix="_cint";

  if (name==0) name = tempname;
  tmpnam(name);
  if(strlen(name)<G__MAXFILENAME-6) G__strlcat(name,appendix,G__MAXFILENAME);
  G__tmpfiles.Add(name);
  return(name);

#endif
}

static int G__istmpnam=0;

/**************************************************************************
* G__openmfp()
**************************************************************************/
void G__openmfp()
{
#ifndef G__TMPFILE
  // coverity[secure_temp]: we don't care about predictable names.
  G__mfp=tmpfile();
  if(!G__mfp) {
    do {
      G__tmpnam(G__mfpname); /* Only VC++ uses this */
      G__mfp=fopen(G__mfpname,"wb+");
    } while((FILE*)NULL==G__mfp && G__setTMPDIR(G__mfpname));
    G__istmpnam=1;
  }
#else
  do {
    G__tmpnam(G__mfpname); /* Only VC++ uses this */
    G__mfp=fopen(G__mfpname,"wb+");
  } while((FILE*)NULL==G__mfp && G__setTMPDIR(G__mfpname));
#endif
}

namespace {
   static struct G__MfpCloser {
      ~G__MfpCloser() {
         G__closemfp();
      }
   } sMfpCloser;
}

/**************************************************************************
* G__closemfp()
**************************************************************************/
int G__closemfp()
{
#ifndef G__TMPFILE
  int result=0;
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

/**************************************************************************
* G__get_ifile()
**************************************************************************/
struct G__input_file *G__get_ifile()
{
   return &G__ifile;
}

// G__register_sharedlib(const char *libname) 
int G__register_sharedlib(const char *libname)
{
   // Register (if not already registered) in G__srcfile a library that
   // is indirectly loaded (via a hard link) and has a CINT dictionary.

   int null_entry = -1;
   int i1 = 0;
   int hash,temp;
   
   G__LockCriticalSection();
   
   /*************************************************
    * store current input file information
    *************************************************/
   G__setdebugcond();
      
   /******************************************************************
    * check if number of loaded file exceeds G__MAXFILE
    * if so, restore G__ifile reset G__eof and return.
    ******************************************************************/
   if(G__nfile==G__MAXFILE) {
      G__fprinterr(G__serr,"Limitation: Sorry, can not load any more files\n");
      G__setdebugcond();
      G__UnlockCriticalSection();
      return(G__LOADFILE_FATAL);
   }
      
   G__hash(libname,hash,temp);
   
   
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
      if(G__matchfilename(i1,libname)
         &&G__get_envtagnum()==G__srcfile[i1].parent_tagnum
         ){
         if(G__prerun==0 || G__debugtrace)
            if(G__dispmsg>=G__DISPNOTE) {
               static const char *excludelist [] = {
                  "stdfunc.dll","stdcxxfunc.dll","posix.dll","ipc.dll","posix.dll"
                  "string.dll","vector.dll","vectorbool.dll","list.dll","deque.dll",
                  "map.dll", "map2.dll","set.dll","multimap.dll","multimap2.dll",
                  "multiset.dll","stack.dll","queue.dll","valarray.dll",
               "exception.dll","stdexcept.dll","complex.dll","climits.dll" };
               static const unsigned int excludelistsize = sizeof(excludelist)/sizeof(excludelist[0]);
               static int excludelen[excludelistsize] = {-1};
               if (excludelen[0] == -1) {
                  for (unsigned int i = 0; i < excludelistsize; ++i)
                     excludelen[i] = (int)strlen(excludelist[i]);
               }
               bool cintdlls = false;
               long len = strlen(libname);
               for (unsigned int i = 0; !cintdlls && i < excludelistsize; ++i) {
                  if (len>=excludelen[i]) {
                     cintdlls = (!strncmp(libname+len-excludelen[i], excludelist[i], excludelen[i]));
                  }
               }
            }
         /******************************************************
          * restore input file information to G__ifile
          * and reset G__eof to 0.
          ******************************************************/
         G__UnlockCriticalSection();
         return(G__LOADFILE_DUPLICATE);
      }
      else {
         ++i1;
      }
   }
   
   int fentry;
   if (null_entry != -1) {
      fentry = null_entry;
   } else {
      fentry = G__nfile;
      ++G__nfile;
      ++G__srcfile_serial;
   }
 
   G__srcfile[fentry].dictpos
      = (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
   G__store_dictposition(G__srcfile[fentry].dictpos);
   
   G__srcfile[fentry].hdrprop = G__NONCINTHDR;
   
   G__srcfile[fentry].security = G__security;
   
   G__srcfile[fentry].prepname = (char*)NULL;
   G__srcfile[fentry].hash = hash;
   G__srcfile[fentry].filename = (char*)malloc(strlen(libname)+1);
   strcpy(G__srcfile[fentry].filename,libname); // Okay, we allocated the right size
   G__srcfile[fentry].fp=0;
   
   G__srcfile[fentry].included_from = G__ifile.filenum;
   
   G__srcfile[fentry].ispermanentsl = 2;
   G__srcfile[fentry].initsl = 0;
   G__srcfile[fentry].hasonlyfunc = (struct G__dictposition*)NULL;
   G__srcfile[fentry].definedStruct = 0;
   G__srcfile[fentry].parent_tagnum = G__get_envtagnum();
   G__srcfile[fentry].slindex = -1;

   G__UnlockCriticalSection();
   return(fentry);
}
   
// G__unregister_sharedlib(const char *libname) 
int G__unregister_sharedlib(const char *libname)
{
   // Unregister (if and only if it has been registered by G__register_sharedlib) 
   // in G__srcfile a library.
   
   G__LockCriticalSection();
   
   /******************************************************************
    * check if file is already loaded.
    * if not so, return
    ******************************************************************/
   int i2;
   int hash;
   G__hash(libname,hash,i2);
   
   bool flag = false;
   int ifn;
   for(ifn = G__nfile-1; ifn>0; --ifn) {
      if(G__srcfile[ifn].ispermanentsl == 2 
         && G__matchfilename(ifn,libname)) {
         flag = true;
         break;
      }
   }
   
   if (flag) {
      // File found
      
      // No check for busy-ness is need.  If we get there the library
      // is already being unloaded.
      
      // No active unload to do, since we did not load this file directly.

      if (G__srcfile[ifn].dictpos) {
         free((void*) G__srcfile[ifn].dictpos);
         G__srcfile[ifn].dictpos = 0;
      }
      if (G__srcfile[ifn].hasonlyfunc) {
         free((void*) G__srcfile[ifn].hasonlyfunc);
         G__srcfile[ifn].hasonlyfunc = 0;
      }
      if (G__srcfile[ifn].filename) {
         // --
#ifndef G__OLDIMPLEMENTATION1546
         size_t len = strlen(G__srcfile[ifn].filename);
         if (
             (len > strlen(G__NAMEDMACROEXT2)) &&
             !strcmp(G__srcfile[ifn].filename + len - strlen(G__NAMEDMACROEXT2), G__NAMEDMACROEXT2)
             ) {
            remove(G__srcfile[ifn].filename);
         }
#endif // G__OLDIMPLEMENTATION1546
         free((void*) G__srcfile[ifn].filename);
         G__srcfile[ifn].filename = 0;
      }
      G__srcfile[ifn].hash = 0;
      G__srcfile[ifn].ispermanentsl = 0;
      G__srcfile[ifn].included_from = -1;
      
      if(G__debug) {
         G__fprinterr(G__serr,"File=%s unregistered\n",libname);
      }
      while(G__nfile && G__srcfile[G__nfile-1].filename==0) {
         --G__nfile;
      }
      ++G__srcfile_serial;
   }
   
   G__UnlockCriticalSection(); 
   return G__UNLOADFILE_SUCCESS;
   
}
   
} /* extern "C" */

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
