/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file pause.c
 ************************************************************************
 * Description:
 *  Interactive interface
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
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


/* 1723 is not needed because freopen deals with both stdout and cout */
#define G__OLDIMPLEMENTATION1723

#ifndef G__OLDIMPLEMENTATION1635
extern void G__redirectcout G__P((const char* filename)) ;
void G__unredirectcout() ;
extern void G__redirectcerr G__P((const char* filename)) ;
extern void G__unredirectcerr() ;
extern void G__redirectcin G__P((const char* filename)) ;
extern void G__unredirectcin() ;
#endif

#ifndef G__OLDIMPLEMENTATION856
#if defined(G__WIN32)
#include <windows.h>
#elif defined(G__POSIX)
#include <unistd.h> /* already included in G__ci.h */
#endif
#endif

extern void G__setcopyflag G__P((int flag));

#define G__NUM_STDIN   0
#define G__NUM_STDOUT  1
#define G__NUM_STDERR  2
#ifndef G__OLDIMPLEMENTATION1722
#define G__NUM_STDBOTH 3
#endif

#ifdef G__OLDIMPLEMENTATION2137
struct G__store_env {
  struct G__var_array *var_local;
  long struct_offset;
  int tagnum;
  int exec_memberfunc;
};

struct G__view {
  struct G__input_file file;
  struct G__var_array *var_local;
  long struct_offset;
  int tagnum;
  int exec_memberfunc;
};
#endif

#ifdef G__BORLANDCC5
void G__decrement_undo_index(int *pi);
void G__increment_undo_index(int *pi);
int G__is_valid_dictpos(struct G__dictposition *dict);
void G__show_undo_position(int index);
void G__init_undo(void);
int G__clearfilebusy(int ifn);
void G__storerewindposition(void);
#ifndef G__OLDIMPLEMENTATION1917
static void G__display_keyword(FILE *fout,char *keyword,FILE *keyfile);
#else
static void G__display_keyword(FILE *fout,char *keyword,char *fname);
#endif
void G__rewinddictionary(void);
void G__UnlockCriticalSection(void);
void G__LockCriticalSection(void); 
int G__IsBadCommand(char *com);
static void G__unredirectoutput(FILE **sout,FILE **serr,FILE **sin,char *keyword,char *pipefile);
static void G__redirectoutput(char *com,FILE **psout,FILE **pserr,FILE **psin,int asemicolumn,char *keyword,char *pipefile);
void G__cancel_undo_position(void);
static int G__atevaluate(G__value buf);
#endif

/**************************************************************************
* G__SET_TEMPENV
*
**************************************************************************/
#define G__SET_TEMPENV                                 \
      store.var_local = G__p_local;                    \
      store.struct_offset = G__store_struct_offset;    \
      store.tagnum=G__tagnum;                          \
      store.exec_memberfunc = G__exec_memberfunc;      \
      G__p_local=view.var_local;                       \
      G__store_struct_offset=view.struct_offset;       \
      G__tagnum=view.tagnum;                           \
      G__exec_memberfunc=view.exec_memberfunc;         \
      G__storerewindposition()

/**************************************************************************
* G__RESET_TEMPENV
*
**************************************************************************/
#define G__RESET_TEMPENV                               \
      G__p_local=store.var_local;                      \
      G__store_struct_offset=store.struct_offset;      \
      G__tagnum=store.tagnum;                          \
      G__exec_memberfunc=store.exec_memberfunc

#ifndef G__OLDIMPLEMENTATION696
/**************************************************************************
* G__STORE_EVALENV
*
**************************************************************************/
#define G__STORE_EVALENV                                  \
      struct G__input_file eval_ifile;                    \
      struct G__input_file eval_view;                     \
      struct G__var_array *eval_local=G__p_local;         \
      long eval_struct_offset=G__store_struct_offset;     \
      int eval_tagnum=G__tagnum;                          \
      int eval_exec_memberfunc=G__exec_memberfunc;        \
      int store_step=G__step;                             \
      eval_ifile=G__ifile;                                \
      eval_view=view.file;                                \
      if(strncmp("s",com,1)==0) G__step=1;                \
      else if(strncmp("S",com,1)==0) G__step=0;           \
      else if(G__stepover) G__step=0

/**************************************************************************
* G__RESTORE_EVALENV
*
**************************************************************************/
#define G__RESTORE_EVALENV                                \
      view.file=eval_view;                                \
      G__ifile=eval_ifile;                                \
      G__p_local=eval_local;                              \
      G__store_struct_offset=eval_struct_offset;          \
      G__tagnum=eval_tagnum;                              \
      G__exec_memberfunc=eval_exec_memberfunc;            \
      if((char*)NULL==strstr(command,"G__stepmode(")) G__step=store_step

#endif



#ifdef G__ASM_DBG
extern int G__asm_step;
#endif

static int G__pause_return=0;
#ifdef G__OLDIMPLEMENTATION463
static FILE *fout=(FILE*)NULL;
#endif


#ifdef G__ROOT
static int G__rootmode=G__INPUTROOTMODE;
#else
static int G__rootmode=G__INPUTCINTMODE;
#endif

#ifndef G__OLDIMPLEMENTATION1795
static int G__lockinputmode=0;
#endif



#ifndef G__OLDIMPLEMENTATION1035
/************************************************************************
* Thread Safe protection for ROOT
*
* You need to write 4 functions. In case of Windows
*
*************************************************************************
* #include <windows.h>
* static DWORD R__ProtectThreadId;
* static CRITICAL_SECTION R__ProtectCriticalSection;
*
* void R__StoerLockThread() {
*   R__ProtectThreadId=GetCurrentThreadId();
* }
*
* int R__IsSameThread() {
*   if(R__ProtectThreadId==GetCurrentThreadId()) return(1);
*   else return(0);
* }
*
* void R__EnterCriticalSection() {
*   EnterCriticalSection(&R__ProtectCriticalSection);
* }
*
* void R__LeaveCriticalSection() {
*   LeaveCriticalSection(&R__ProtectCriticalSection);
* }
*************************************************************************
* 
* and then initialize the library as follows. You need to register above
* 4 functions to G__SetCriticalSectionEnv().
*
*************************************************************************
* #include <G__ci.h>
* void InitThreadLock() {
*   InitializeCriticalSection(&R__ProtectCriticalSection);
*   G__SetCriticalSectionEnv(R__IsSameThread,R__StoreLockThread
*                            ,R__EnterCriticalSection,R__LeaveCriticalSection);
* }
*
************************************************************************/

/************************************************************************
* Thread protection lock
************************************************************************/
static int lockcount=0;
static int (*G__IsSameThread)();
static void (*G__StoreLockThread)();
static void (*G__EnterCriticalSection)();
static void (*G__LeaveCriticalSection)();

/************************************************************************
* G__SetCriticalSectionEnv
************************************************************************/
void G__SetCrititalSectionEnv(issamethread,storelockthread,entercs,leavecs)
int (*issamethread)();
void (*storelockthread)();
void (*entercs)();
void (*leavecs)();
{
  G__IsSameThread=issamethread;
  G__StoreLockThread=storelockthread;
  G__EnterCriticalSection=entercs;
  G__LeaveCriticalSection=leavecs;
}

/************************************************************************
* G__LockCriticalSection
************************************************************************/
void G__LockCriticalSection() {
  if(!G__IsSameThread||!G__EnterCriticalSection) return;
#ifndef __CINT__
  if(lockcount&&(*G__IsSameThread)()) {
    /* recursive call in same thread, should not lock */
    ++lockcount;
    return;
  }
#endif
  (*G__EnterCriticalSection)();
  (*G__StoreLockThread)();
  ++lockcount;
  return;
}

/************************************************************************
* G__UnlockCriticalSection
************************************************************************/
void G__UnlockCriticalSection() {
  if(!G__IsSameThread||!G__LeaveCriticalSection) return;
#ifndef __CINT__
  if(lockcount&&(*G__IsSameThread)()) {
    /* recursive call in same thread, should not unlock */
    --lockcount;
    return;
  }
#endif
  --lockcount;
  (*G__LeaveCriticalSection)();
  return;
}
#endif

#ifndef G__OLDIMPLEMENTATION1968
/************************************************************************
* G__autoloading()  com="{  TUnknown x;" ,  "new TUnknown
************************************************************************/
int G__autoloading(com)
char *com;
{
  int i=0,j=0;
  char classname[G__ONELINE];
  while(com[i] && !isalpha(com[i])) ++i;
  while(com[i] && (isalnum(com[i]) || '_'==com[i])) classname[j++]=com[i++];
  classname[j]=0;
  if(0==strcmp(classname,"new")){
    j=0;
    while(com[i] && !isalpha(com[i])) ++i;
    while(com[i] && (isalnum(com[i]) || '_'==com[i])) classname[j++]=com[i++];
    classname[j]=0;
  }
  if(classname[0] && -1==G__defined_tagname(classname,2)) {
    char *dllpost = G__getmakeinfo1("DLLPOST");
    char fname[G__MAXFILENAME];
    char prompt[G__ONELINE];
    FILE *fp;
    sprintf(fname,"%s%s",classname,dllpost);
#if 1
    fp=fopen(fname,"r");
    if(!fp) return 0;
    else fclose(fp);
#endif
    sprintf(prompt,"Will you try loading %s(y/n)? ",fname);
    strcpy(prompt,G__input(prompt));
    if(tolower(prompt[0])!='y') return(0);
    G__security_recover(G__sout); /* QUESTIONABLE */
    switch(G__loadfile(fname)) {
    case G__LOADFILE_SUCCESS:
      fprintf(G__sout,"%s loaded\n",fname);
      return(1);
    default:
      fprintf(G__sout,"Error: failed to load %s\n",fname);
      return(0);
    }
  }
  return(0);
}

/************************************************************************
************************************************************************/
int (*G__pautoloading) G__P((char*)) = G__autoloading;

/************************************************************************
* G__set_autoloading
************************************************************************/
void G__set_autoloading(p2f)
int (*p2f) G__P((char*));
{
  G__pautoloading = p2f;
}
#endif

#ifndef G__OLDIMPLEMENTATION1094
/************************************************************************
* G__is_valid_dictpos
************************************************************************/
int G__is_valid_dictpos(dict)
struct G__dictposition *dict;
{
  int flag=0;
  struct G__var_array *var = &G__global; 
  struct G__ifunc_table *ifunc = &G__ifunc;

  while(var) {
    if(var==dict->var && var->allvar>=dict->var->allvar) {
      flag=1;
      break;
    }
    var=var->next;
  }
  if(0==flag) return(0);

  flag=0;
  while(ifunc) {
    if(ifunc==dict->ifunc && ifunc->allifunc>=dict->ifunc->allifunc) {
      flag=1;
      break;
    }
    ifunc=ifunc->next;
  }
  if(0==flag) return(0);

  if(G__struct.alltag<dict->tagnum) return(0);
  if(G__newtype.alltype<dict->typenum) return(0);
#ifdef G__SHAREDLIB  /* ???, reported by A.Yu.Isupov, anyway no harm */
  if(G__allsl<dict->allsl) return(0);
#endif
  if(G__nfile<dict->nfile) return(0);
  
  return(1);
}

/************************************************************************
* Undo rewinding
************************************************************************/
#define G__MAXUNDO 10
static struct G__dictposition undodictpos[G__MAXUNDO];
static int undoindex;

/******************************************************************
* G__init_undo()
******************************************************************/
void G__init_undo() 
{
  int i;
  undoindex=0;
  for(i=0;i<G__MAXUNDO;i++) {
    undodictpos[i].var=(struct G__var_array*)NULL;
#ifndef G__OLDIMPLEMENTATION2024
    undodictpos[i].ptype=(char*)NULL;
#endif
  }
}

/******************************************************************
* G__increment_undo_index()
******************************************************************/
void G__increment_undo_index(pi)
int *pi;
{
  if(++(*pi)>=G__MAXUNDO) *pi=0;
}

/******************************************************************
* G__decrement_undo_index()
******************************************************************/
void G__decrement_undo_index(pi)
int *pi;
{
  if(--(*pi)<0) *(pi)=G__MAXUNDO-1;
}

/******************************************************************
* G__cancel_undo_position()
******************************************************************/
void G__cancel_undo_position()
{
  G__decrement_undo_index(&undoindex);
  undodictpos[undoindex].var=(struct G__var_array*)NULL;
#ifndef G__OLDIMPLEMENTATION2024
  if(undodictpos[undoindex].ptype && 
     undodictpos[undoindex].ptype!=(char*)G__PVOID) {
    free((void*)undodictpos[undoindex].ptype);
    undodictpos[undoindex].ptype = (char*)NULL;
  }
#endif
}

/******************************************************************
* G__store_undo_position()
******************************************************************/
void G__store_undo_position()
{
  G__store_dictposition(&undodictpos[undoindex]);
  G__increment_undo_index(&undoindex);
}

/******************************************************************
* G__show_undo_position()
******************************************************************/
void G__show_undo_position(index)
int index;
{
  int nfile=undodictpos[index].nfile;
  int tagnum=undodictpos[index].tagnum;
  int typenum=undodictpos[index].typenum;
  struct G__ifunc_table *ifunc=undodictpos[index].ifunc;
  int ifn=undodictpos[index].ifn;
  struct G__var_array *var=undodictpos[index].var;
  int ig15=undodictpos[index].ig15;

  fprintf(G__sout,"!!! Following objects will be removed by undo !!!\n");

  fprintf(G__sout,"Src File : ");
  while(nfile<G__nfile) 
    fprintf(G__sout,"%s ",G__srcfile[nfile++].filename);
  fprintf(G__sout,"\n");

  fprintf(G__sout,"Class    : ");
  while(tagnum<G__struct.alltag) 
    fprintf(G__sout,"%s ",G__fulltagname(tagnum++,1));
  fprintf(G__sout,"\n");

  fprintf(G__sout,"Typedef  : ");
  while(typenum<G__newtype.alltype) 
    fprintf(G__sout,"%s ",G__newtype.name[typenum++]);
  fprintf(G__sout,"\n");

  fprintf(G__sout,"Function : ");
  while(ifunc) {
    while(ifn<ifunc->allifunc) {
      fprintf(G__sout,"%s ",ifunc->funcname[ifn++]);
    }
    ifn=0;
    ifunc=ifunc->next;
  }
  fprintf(G__sout,"\n");

  fprintf(G__sout,"Variable : ");
  while(var) {
    while(ig15<var->allvar) {
      fprintf(G__sout,"%s ",var->varnamebuf[ig15++]);
    }
    ig15=0;
    var=var->next;
  }
  fprintf(G__sout,"\n");
}

/******************************************************************
* G__rewind_undo_position()
******************************************************************/
void G__rewind_undo_position()
{
  G__decrement_undo_index(&undoindex);
  if(undodictpos[undoindex].var &&
     G__is_valid_dictpos(&undodictpos[undoindex])){
    char buf[G__ONELINE];
    G__show_undo_position(undoindex);
    strcpy(buf,G__input("Are you sure? (y/n) "));
    if('y'==tolower(buf[0])) {
      G__scratch_upto(&undodictpos[undoindex]);
      undodictpos[undoindex].var=(struct G__var_array*)NULL;
      G__fprinterr(G__serr,"!!! Dictionary position rewound !!!\n");
    }
    else {
      G__increment_undo_index(&undoindex);
    }
  }
  else {
    G__fprinterr(G__serr,"!!! No undo rewinding buffer !!!\n");
    G__init_undo();
  }
}
#endif

#define G__OLDIMPLEMENTATION1156
#ifndef G__OLDIMPLEMENTATION1156
/**************************************************************************
* G__init_env()
*
*  Explicit initialization of all necessary global variables
**************************************************************************/
int G__init_env()
{
  int i;
  /* G__p_ifunc = &G__ifunc ; */

  G__exec_memberfunc = 0;
  G__memberfunc_tagnum = -1;
  G__memberfunc_struct_offset=0;

#ifdef G__ASM
  /***************************************************
   * loop compiler variables initialization
   ***************************************************/
  G__asm_name_p=0;
#ifdef G__ASM_WHOLEFUNC
  G__asm_wholefunction = G__ASM_FUNC_NOP;
#endif
#ifndef G__OLDIMPLEMENTATION517
  G__asm_wholefunc_default_cp=0;
#endif
  G__abortbytecode();
  G__asm_dbg=0;
  G__asm_cp=0;               /* compile program counter */
  G__asm_dt=G__MAXSTACK-1;   /* constant data address */
#ifdef G__ASM_IFUNC
  G__asm_inst = G__asm_inst_g;
  G__asm_stack = G__asm_stack_g;
  G__asm_name = G__asm_name_g;
  G__asm_name_p = 0;
#endif
#endif

  G__debug=0;            /* trace input file */
  G__breakdisp=0;        /* display input file at break point */
  G__break=0;            /* break flab */
  G__step=0;             /* step execution flag */
  G__charstep=0;         /* char step flag */
  G__break_exit_func=0;  /* break at function exit */
  /* G__no_exec_stack = -1; */ /* stack for G__no_exec */
  G__no_exec=0;          /* no execution(ignore) flag */
  G__no_exec_compile=0;
  G__var_type='p';      /* variable decralation type */
  G__var_typeB='p';
  G__prerun=0;           /* pre-run flag */
  G__funcheader=0;       /* function header mode */
  G__return=G__RETURN_NON; /* return flag of i function */
  /* G__extern_global=0;    number of globals defined in c function */
  G__disp_mask=0;        /* temporary read count */
  G__temp_read=0;        /* temporary read count */
  G__switch=0;           /* switch case control flag */
  G__mparen=0;           /* switch case break nesting control */
  G__eof_count=0;        /* end of file error flag */
  G__globalvarpointer=G__PVOID; /* make compiled func's global table */

  G__xfile[0]='\0';
  G__tempc[0]='\0';

  G__doingconstruction=0;

  G__nobreak=0;

#ifdef G__FRIEND
  G__friendtagnum = -1;
#endif
  G__def_struct_member=0;
#ifndef G__OLDIMPLEMENTATION440
  G__tmplt_def_tagnum = -1;
#endif
  G__def_tagnum = -1;
  G__tagdefining = -1;
  G__tagnum = -1;
  G__typenum = -1;
  G__iscpp = 1;
  G__cpplock=0;
  G__clock=0;
  G__constvar = 0;
  G__unsigned = 0;
  G__ansiheader = 0;
  G__enumdef=0;
  G__store_struct_offset=0;
  G__decl=0;
#ifdef G__OLDIMPLEMENTATION435
  G__allstring=0;
#endif
  G__longjump=0;
  G__coredump=0;
  G__definemacro=0;

  G__noerr_defined=0;

  G__static_alloc=0;
  G__func_now = -1;
  G__func_page = 0;
  G__varname_now=NULL;
  G__twice=0;
  G__breaksignal=0;

  G__bitfield=0;

  G__templevel = 0;

  G__reftype=G__PARANORMAL;

  G__access=G__PUBLIC;

  G__stepover=0;

  G__steptrace = 0;
  G__debugtrace = 0;

  G__p_local = NULL ;    /* local variable array */

  G__cpp_aryconstruct=0;
  G__cppconstruct=0;

#ifndef G__OLDIMPLEMENTATION954
  G__exceptionbuffer = G__null;
#endif

  G__virtual=0;
#ifdef G__AUTOCOMPILE
  G__fpautocc=(FILE*)NULL;
  G__compilemode=1;
#endif
  G__typedefnindex = 0;
  G__oprovld = 0;
  G__p2arylabel[0]=0;

  G__interactive=0;
  G__interactivereturnvalue=G__null;

  G__isfuncreturnp2f=0;

  G__typepdecl=0;

  G__macroORtemplateINfile=0;

  G__macro_defining=0;

  G__nonansi_func = 0;

  G__parseextern=0;

  G__istrace = 0;

  G__pbreakcontinue = (struct G__breakcontinue_list*)NULL;

#ifdef G__SECURITY
  G__castcheckoff=0;
  G__security_error = G__NOERROR;
#endif

  return(0);
}
#endif

#ifndef G__OLDIMPLEMENTATION1031
/************************************************************************
* error recovery
************************************************************************/
static struct G__dictposition errordictpos;
#ifndef G__OLDIMPLEMENTATION1422
static struct G__input_file errorifile;
#endif

#ifndef G__OLDIMPLEMENTATION2227
/******************************************************************
* G__clear_errordictpos()
******************************************************************/
void G__clear_errordictpos() 
{
  if(0!=errordictpos.ptype && (char*)G__PVOID!=errordictpos.ptype) {
    free((void*)errordictpos.ptype);
    errordictpos.ptype = (char*)NULL;
  }
}
#endif

#ifndef G__OLDIMPLEMENTATION1066
/******************************************************************
* G__clearfilebusy()
******************************************************************/
int G__clearfilebusy(ifn)
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
    if(ifunc->allifunc>G__MAXIFUNC) {
      G__fprinterr(G__serr,"Internal error: G__clearfilebusy() FATAL! Save data and terminate session");
      G__printlinenum();
      return(0);
    }
    for(i1=0;i1<ifunc->allifunc;i1++) {
      if( 0!=ifunc->busy[i1] && ifunc->pentry[i1]->filenum>=ifn ) {
	ifunc->busy[i1] = 0;
	G__fprinterr(G__serr,"Function %s() busy flag cleared\n"
		,ifunc->funcname[i1]);
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
	  ifunc->busy[i1] = 0;
	  G__fprinterr(G__serr,"Function %s() busy flag cleared\n"
		  ,ifunc->funcname[i1]);
	  flag++;
	}
      }
      ifunc=ifunc->next;
    }
  }

#ifndef G__OLDIMPLEMENTATION1156
  G__init_env();
#endif

  return(flag);
}
#endif

/************************************************************************
* G__storerewindposition
************************************************************************/
void G__storerewindposition()
{
   G__store_dictposition(&errordictpos);
#ifndef G__OLDIMPLEMENTATION1422
   errorifile = G__ifile;
#endif
}



/************************************************************************
* G__rewinddictionary
************************************************************************/
void G__rewinddictionary()
{
  if(errordictpos.var) {
#ifndef G__OLDIMPLEMENTATION1094
    if(G__is_valid_dictpos(&errordictpos)) {
#else
    if(errordictpos.nfile<=G__nfile) {
#endif
      G__clearfilebusy(errordictpos.nfile);
      G__scratch_upto(&errordictpos);
#ifndef G__ROOT
      G__fprinterr(G__serr,"!!!Dictionary position rewound... ");
#endif
    }
    else {
      G__fprinterr(G__serr,"!!!Dictionary position not recovered because G__unloadfile() is used in a macro!!!\n");
    }
  }
#ifndef G__OLDIMPLEMENTATION1422
#ifndef G__OLDIMPLEMENTATION1448
  /* If the file info saved was related to a temporary file
   * there is no use to reput it */
#ifndef G__OLDIMPLEMENTATION1601
  /* We use '<' here, because if the file with an error was a temporary
     file, it probably has been closed by now and the 'fence' has been
     moved :( */
  if(errorifile.filenum<G__gettempfilenum()) G__ifile = errorifile;
#else
  if(errorifile.filenum!=G__MAXFILE-1) G__ifile = errorifile;
#endif
#else
  G__ifile = errorifile;
#endif
#endif
  errordictpos.var = (struct G__var_array*)NULL;
}
#endif

#ifndef G__OLDIMPLEMENTATION850
/************************************************************************
* G__atevaluate
************************************************************************/
static int G__atevaluate(buf)
G__value buf;
{
  G__value result;
  char com[G__ONELINE],buf2[G__ONELINE];
  int known=0;
  int store_break=G__break;
  int store_step=G__step;
  int store_dispsource=G__dispsource;
#ifndef G__OLDIMPLEMENTATION896
  int store_asm_exec = G__asm_exec;
#endif
#ifndef G__OLDIMPLEMENTATION1184
  int store_debug = G__debug;
#endif
#ifndef G__OLDIMPLEMENTATION2001
  void *store_errmsgcallback;
#endif
#ifndef G__OLDIMPLEMENTATION2003
  int store_mask_error = G__mask_error;
#endif

  if(G__return>G__RETURN_NORMAL && G__security_error) return(0);

#ifndef G__OLDIMPLEMENTATION896
  G__asm_exec = 0;
#endif

#ifndef G__OLDIMPLEMENTATION2191
  if('1'==buf.type || 'a'==buf.type 
     || 'n'==buf.type || 'm'==buf.type || 'q'==buf.type) return(0);
#else
  if('Q'==buf.type || 'a'==buf.type) return(0);
#endif
  G__valuemonitor(buf,buf2);
  sprintf(com,"G__ateval(%s)",buf2);
  G__break=0; G__step=0; G__dispsource=0;
#ifndef G__OLDIMPLEMENTATION1184
  G__debug=0;
#endif
  G__setdebugcond();
#ifndef G__OLDIMPLEMENTATION2001
  store_errmsgcallback = G__get_errmsgcallback();
  G__set_errmsgcallback(G__mask_errmsg);
#endif
#ifndef G__OLDIMPLEMENTATION2003
  G__mask_error = 1;
#endif
  result=G__getfunction(com,&known,G__TRYNORMAL);
#ifndef G__OLDIMPLEMENTATION2003
  G__mask_error = store_mask_error;
#endif
#ifndef G__OLDIMPLEMENTATION2001
  G__set_errmsgcallback(store_errmsgcallback);
#endif
  G__break=store_break; G__step=store_step; G__dispsource=store_dispsource;
#ifndef G__OLDIMPLEMENTATION1184
  G__debug=store_debug;
#endif
#ifndef G__OLDIMPLEMENTATION896
  G__asm_exec = store_asm_exec;
#endif
  if(known) return((int)G__int(result));
  return(0);
}
#endif

#ifndef G__OLDIMPLEMENTATION1253
/************************************************************************
* G__display_tempobj()
*
************************************************************************/
static void G__display_tempobj(fout)
FILE *fout;
{
  struct G__tempobject_list *p = G__p_tempbuf;
  char buf[G__ONELINE];
  fprintf(fout,"current tempobj stack level = %d\n",G__templevel);
  do {
    G__valuemonitor(p->obj,buf);
    fprintf(fout,"level%-3d %2d %s\n",p->level,p->cpplink,buf);
    p = p->prev;
  } while(p);
}
#endif

#ifndef G__OLDIMPLEMENTATION577
/************************************************************************
* G__display_keyword()
*
*  display keyword for '/[keyword]' debugger command
*
************************************************************************/
static void G__display_keyword(fout,keyword,
#ifndef G__OLDIMPLEMENTATION1917
			       keyfile
#else
			       fname
#endif
)
FILE *fout;
char *keyword;
#ifndef G__OLDIMPLEMENTATION1917
FILE *keyfile;
#else
char *fname;
#endif
{
  char line[G__LONGLINE];
  char *null_fgets;
#ifdef G__OLDIMPLEMENTATION1917
  FILE *keyfile;

  keyfile = fopen(fname,"r");
#endif

  if(keyfile) {
#ifndef G__OLDIMPLEMENTATION1917
    fseek(keyfile,0L,SEEK_SET);
#endif
    null_fgets=fgets(line,G__LONGLINE-1,keyfile);
    while((char*)NULL!=null_fgets) {
      if(strstr(line,keyword)) {
	if(G__more(fout,line)) break;
      }
      null_fgets=fgets(line,G__LONGLINE-1,keyfile);
    }
#ifdef G__OLDIMPLEMENTATION1917
    fclose(keyfile);
#endif
  }
  else {
    G__genericerror("Warning: can't open file. keyword search unsuccessful");
  }
}
#endif

#ifndef G__OLDIMPLEMENTATION1108
/************************************************************************
* G__reloadfile
*
************************************************************************/
int G__reloadfile(filename)
char *filename;
{
  int i,j=0;
  char *storefname[G__MAXFILE];
  int storecpp[G__MAXFILE];
  int storen=0;
  int flag=0;
  int store_cpp=G__cpp;
  int store_prerun=G__prerun;
  if(!filename || 0==filename[0]) {
    G__fprinterr(G__serr,"Error: no file specified\n");
    return(-1);
  }

  for(i=0;i<G__nfile;i++) {
    if(!flag &&
#ifndef G__OLDIMPLEMENTATION1196
       G__matchfilename(i,filename)
#else
       (strcmp(filename,G__srcfile[i].filename)==0)
#endif
       ) {
#ifndef G__OLDIMPLEMENTATION1273
      if(G__srcfile[i].hasonlyfunc && G__do_smart_unload) {
	G__smart_unload(i);
	flag = 0;
      }
      else flag=1;
#else
      flag=1;
#endif
      j=i;
      while(-1!=G__srcfile[j].included_from
#ifndef G__PHILIPPE18
            /* do not take the tempfile in consideration! */
#ifndef G__OLDIMPLEMENTATION1601
            && (G__srcfile[j].included_from<=G__gettempfilenum())
#else
            && ((G__MAXFILE-1)!=G__srcfile[j].included_from)
#endif
#endif
	    ) {
	j=G__srcfile[j].included_from;
      }
      break;
    }
  }

  if(flag) {
    for(i=j;i<G__nfile;i++) { 
      if(G__srcfile[i].filename[0]) {
	if(G__srcfile[i].prepname) storecpp[storen] = 1;
	else                       storecpp[storen] = 0;
	storefname[storen] = (char*)malloc(strlen(G__srcfile[i].filename)+1);
	strcpy(storefname[storen],G__srcfile[i].filename);
	++storen;
      }
    }
  }
  else {
    return(G__loadfile(filename));
  }

  if(G__UNLOADFILE_SUCCESS!=G__unloadfile(storefname[0])) {
    return(1);
  }

  G__storerewindposition();
  G__init_undo();

  for(i=0;i<storen;i++) {
    G__prerun=1;
    G__cpp=storecpp[i];
    fprintf(G__sout,"reloading %s  %d\n",storefname[i],storecpp[i]);
    G__loadfile(storefname[i]);
    G__return=0;
    G__security_error=0;
  }

  G__prerun=store_prerun;
  G__cpp=store_cpp;

  return(0);
}
#endif

#ifndef G__OLDIMPLEMENTATION967
/************************************************************************
* G__display_classkeyword()
*
************************************************************************/
void G__display_classkeyword(fout,classnamein,keyword,base) 
FILE *fout;
char *classnamein;
char *keyword;
int base;
{
#ifndef G__OLDIMPLEMENTATION1823
  char buf[G__BUFLEN];
  char *classname=buf;
#else
  char classname[G__ONELINE];
#endif
#ifdef G__OLDIMPLEMENTATION1320
  char *p=classname;
#endif
#ifndef G__OLDIMPLEMENTATION2092
  int istmpnam=0;
#endif

#ifndef G__OLDIMPLEMENTATION1823
  if(strlen(classnamein)>G__BUFLEN-5) {
    classname = (char*)malloc(strlen(classnamein)+5);
  }
#endif

  G__more_pause((FILE*)NULL,0);
  strcpy(classname,classnamein);
#ifdef G__OLDIMPLEMENTATION1320
  while(*p && isspace(*p)) ++p;
#ifndef G__OLDIMPLEMENTATION1048
  while(*p && (!isspace(*p)||('>'==*(p-1)&&'>'==*(p+1)))) ++p;
#else
  while(*p && !isspace(*p)) ++p;
#endif
  if(*p) *p=0;
#endif
  if(keyword&&keyword[0]) {
#ifndef G__TMPFILE
    char tname[L_tmpnam+10];
#else
    char tname[G__MAXFILENAME];
#endif
    FILE *G__temp;
    do {
#if !defined(G__OLDIMPLEMENTATION2092)
      G__temp=tmpfile();
      if(!G__temp) {
        G__tmpnam(tname); /* not used anymore */
        G__temp=fopen(tname,"w");
        istmpnam=1;
      }
#elif !defined(G__OLDIMPLEMENTATION1917)
      G__temp=tmpfile();
#else
      G__tmpnam(tname); /* not used anymore */
      G__temp=fopen(tname,"w");
#endif
    } while((FILE*)NULL==G__temp && G__setTMPDIR(tname));
    if(G__temp) {
      G__display_class(G__temp,classname,base,0);
#if !defined(G__OLDIMPLEMENTATION2092)
      if(!istmpnam) {
        fseek(G__temp,0L,SEEK_SET);
        G__display_keyword(fout,keyword,G__temp);
        fclose(G__temp);
      }
      else {
        G__display_keyword(fout,keyword,G__temp);
        fclose(G__temp);
        remove(tname);
      }
#elif !defined(G__OLDIMPLEMENTATION1917)
      fseek(G__temp,0L,SEEK_SET);
      G__display_keyword(fout,keyword,G__temp);
      fclose(G__temp);
#else
      fclose(G__temp);
      G__display_keyword(fout,keyword,tname);
      remove(tname);
#endif
    }
  }
  else {
    G__display_class(fout,classname,base,0);
  }
#ifndef G__OLDIMPLEMENTATION1823
  if(buf!=classname) free((void*)classname);
#endif
}
#endif

#ifdef G__SECURITY
/******************************************************************
* G__security_recover()
******************************************************************/
int G__security_recover(fout)
FILE *fout;
{
  int err = G__security_error ;
#ifndef G__OLDIMPLEMENTATION1067
  if(G__security_error) {
#else
  if(G__return>G__RETURN_NORMAL && G__security_error) {
#endif
    if((G__security_error&(G__DANGEROUS|G__FATAL))) {
#ifndef G__OLDIMPLEMENTATION1485
      if(fout==G__serr) {
#ifdef G__ROOT
	G__fprinterr(G__serr,"*** Fatal error in interpreter... restarting interpreter ***\n");
#else
	G__fprinterr(G__serr,"!!!Fatal error. Sorry, terminate cint session!!!\n");
#endif
      } else 
#endif
      if(fout) {
#ifdef G__ROOT
	fprintf(fout,"*** Fatal error in interpreter... restarting interpreter ***\n");
#else
	fprintf(fout,"!!!Fatal error. Sorry, terminate cint session!!!\n");
#endif
      }
    }
    else {
#ifndef G__OLDIMPLEMENTATION1031
      G__rewinddictionary();
#endif
#ifndef G__OLDIMPLEMENTATION1485
      if(fout==G__serr) {
#ifdef G__ROOT
	G__fprinterr(G__serr,"*** Interpreter error recovered ***\n");
#else
	G__fprinterr(G__serr,"!!!Error recovered!!!\n");
#endif
      } else
#endif
      if(fout) {
#ifdef G__ROOT
	fprintf(fout,"*** Interpreter error recovered ***\n");
#else
	fprintf(fout,"!!!Error recovered!!!\n");
#endif
      }
      G__return=G__RETURN_NON;
#ifndef G__OLDIMPLEMENTATION1067
      G__security_error = G__NOERROR; 
#endif
      /* G__security_error = G__NOERROR; */ /* don't remember why not doing */
#ifndef G__OLDIMPLEMENTATION1421
      G__prerun = 0;
      G__decl = 0;
#endif
    }
  }
#ifndef G__OLDIMPLEMENTATION1031
  errordictpos.var = (struct G__var_array*)NULL;
#endif

  return(err);
}
#endif

/******************************************************************
* Inevitable static variable
******************************************************************/
static struct G__store_env store;
static struct G__view view;
#ifndef G__OLDIMPLEMENTATION860
static int init_process_cmd_called=0;
#endif
#ifndef G__OLDIMPLEMENTATION495
static char stdoutsav[64];
static char stderrsav[64];
static char stdinsav[64];
#endif

/******************************************************************
* int G__init_process_cmd()
******************************************************************/
int G__init_process_cmd()
{
#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif
  view.file = G__ifile;
  view.var_local = G__p_local;
  view.struct_offset=G__store_struct_offset;
  view.tagnum=G__tagnum;
  view.exec_memberfunc=G__exec_memberfunc;
#ifndef G__OLDIMPLEMENTATION860
  init_process_cmd_called=1;
#endif
#ifndef G__OLDIMPLEMENTATION1035
  G__UnlockCriticalSection();
#endif
  return(0);
}

/******************************************************************
* int G__pause()
******************************************************************/
int G__pause()
{
#ifndef G__OLDIMPLEMENTATION432
  char *p;
  char cintname[G__ONELINE];
  char filename[G__ONELINE];
#endif
  char command[G__LONGLINE];
  char prompt[G__ONELINE];
  int ignore=G__PAUSE_NORMAL;
  int more = 0;

  void (*oldhandler)();


#ifdef G__NEVER
  static int flag=0;
#endif

#ifndef G__OLDIMPLEMENTATION432
  p = strrchr(G__nam,G__psep[0]);
  if(p && *(p+1)) strcpy(cintname,p+1);
  else            strcpy(cintname,G__nam);
#endif

  prompt[0] = '\0';

  /************************************************
  * Do not pause while no execution
  ************************************************/
  if(G__no_exec!=0) return(ignore);

  /************************************************
  * Do not pause if breakfile is specified but
  * doesn't match to current source file
  ************************************************/
#ifdef G__OLDIMPLEMENTATION473
  if((G__step!=0)&&(G__break==0)&&(strcmp(G__breakfile,"")!=0)) {
    if(strcmp(G__breakfile,G__ifile.name)!=0) return(ignore);
  }
#endif

  /************************************************
  * Do not pause if assertion is set and the
  * condition is faulse
  ************************************************/
  if((G__step==0)&&(strcmp(G__assertion,"")!=0)&&
     (!G__test(G__assertion))) {
#ifndef G__OLDIMPLEMENTATION1486
    if(G__security_error) {
      G__fprinterr(G__serr,"Warning: Assertion failed, delete assert expression %s\n"
		   ,G__assertion);
      G__assertion[0] = 0;
    }
#endif
    return(ignore);
  }

  /************************************************
  * Prompt/comments printed out to stdout
  * Put caridge return
  ************************************************/
#ifdef G__OLDIMPLEMENTATION463
  fout=G__sout;
#endif
  fprintf(G__sout,"\n");


  G__init_process_cmd();

  /************************************************
  * If Display mode, display source code
  ************************************************/
  if((G__breakdisp!=0)&&(G__ifile.name[0]!='\0')) {
    G__pr(G__sout,view.file);
  }

  /************************************************
  * Infinite loop, until space or 'S'(Step)
  * command
  ************************************************/
  while(1) {

    /************************************
    * Set prompt string
    ************************************/
    if (prompt[0]) strcpy(command, prompt);
    else {
      if('\0'==view.file.name[0]) {
#ifndef G__OLDIMPLEMENTATION432
	sprintf(command,"%s> ", cintname);
#else
	sprintf(command,"%s> ", G__nam);
#endif
      }
      else {
#ifndef G__OLDIMPLEMENTATION432
	p = strrchr(view.file.name,G__psep[0]);
	if(p && *(p+1)) strcpy(filename,p+1);
	else            strcpy(filename,view.file.name);
	sprintf(command,"FILE:%s LINE:%d %s> "
		,G__stripfilename(filename),view.file.line_number,cintname);
#else
	sprintf(command,"FILE:%s LINE:%d %s> "
		,view.file.name,view.file.line_number,G__nam);
#endif
      }
    }

    if (more == 0) {
#ifdef G__ASM
       G__abortbytecode();
#endif

    /************************************
    * flush files
    ************************************/
       G__disp_mask=0;
       fflush(G__sout);
       fflush(G__serr);
#ifdef G__DUMPFILE
       if(G__dumpfile!=NULL) fflush(G__dumpfile);
#endif

       /* if G__atpause is defined , evaluate the function */
       if(G__atpause) {
#ifndef G__OLDIMPLEMENTATION875
	 G__p2f_void_void((void*)G__atpause);
#else
         (*G__atpause)();
#endif
       }

    /************************************
    * Exit program if Break key
    ************************************/
#ifndef G__ROOT /* This is only defined with ROOT */
       signal(SIGINT,G__killproc);
#endif

    /***********************************************
    * For ATEC replacement
    * If execute readline dump file mode, get
    * debug command from the readline dump file
    * else get command from stdin.
    ***********************************************/

       G__SET_TEMPENV;

       G__in_pause=1;

    } else {

#ifndef G__ROOT /* This is only defined with ROOT */
       signal(SIGINT,G__exit);
#endif

    }

    strcpy(command,G__input(command));

    if (more == 0) {
       G__in_pause=0;

       G__RESET_TEMPENV;
    }

    /************************************
    * G__pause() when break key
    ************************************/
    oldhandler = signal(SIGINT,G__breakkey);
#ifdef G__OLDIMPLEMENTATION1352
    alarm(0);
#endif

    G__pause_return=0;

    ignore = G__process_cmd(command, prompt, &more,(int*)NULL,(G__value*)NULL);
#ifndef G__OLDIMPLEMENTATION2139
    if(G__return==G__RETURN_IMMEDIATE) break;
#endif
    if(ignore/G__PAUSE_ERROR_OFFSET) break;
    if(G__pause_return) break;
  }

  G__pause_return=0;
#ifdef G__ROOT
  signal(SIGINT,oldhandler);
#endif
  return(ignore);
}

#ifndef G__OLDIMPLEMENTATION464
/******************************************************************
* G__update_stdio()
******************************************************************/
int G__update_stdio()
{
  char command[G__LONGLINE];

#ifndef G__OLDIMPLEMENTATION713
  G__intp_sout = G__sout;
  G__intp_serr = G__serr;
  G__intp_sin = G__sin;
  sprintf(command,"stderr=%ld",(long)G__intp_serr);
  G__getexpr(command);
  sprintf(command,"stdout=%ld",(long)G__intp_sout);
  G__getexpr(command);
  sprintf(command,"stdin=%ld",(long)G__intp_sin);
  G__getexpr(command);
#else
  sprintf(command,"stderr=%ld",G__serr);
  G__getexpr(command);
  sprintf(command,"stdout=%ld",G__sout);
  G__getexpr(command);
  sprintf(command,"stdin=%ld",G__sin);
  G__getexpr(command);
#endif
  return(0);
}

#define G__OLDIMPLEMENTATION1983
#ifndef G__OLDIMPLEMENTATION1983
#ifdef G__WIN32 /*vo*/
static HANDLE handleout;
#endif
#endif

/******************************************************************
* G__redirectoutput
******************************************************************/
static void G__redirectoutput(com,psout,pserr,psin,asemicolumn,keyword,pipefile)
char *com;
FILE **psout;
FILE **pserr;
FILE **psin;
int asemicolumn;
char *keyword;
char *pipefile;
{
  char *semicolumn;
  char *redirect;
  char *redirectin;
  char *singlequote;
  char *doublequote;
  char *paren;
  char *blacket;
  /* int issemicolumn; */
  char *openmode;
  char filename[G__MAXFILENAME];
  int i=0;
  int j=1;
  int mode=0; /* 0:stdout, 1:stderr, 2:stdout+stderr */

  redirect=strrchr(com,'>');
  redirectin=strrchr(com,'<');
  semicolumn=strrchr(com,';');
  singlequote=strrchr(com,'\'');
  doublequote=strrchr(com,'"');
  paren=strrchr(com,')');
  blacket=strrchr(com,']');

  pipefile[0] = 0;
  keyword[0] = 0;

  if(0==asemicolumn ||
     (semicolumn && semicolumn<redirect &&
      ((char*)NULL==singlequote||semicolumn>singlequote) &&
      ((char*)NULL==doublequote||semicolumn>doublequote))) {

    if(redirect && 
#ifndef G__OLDIMPLEMENTATION493
       (isspace(*(redirect-1)) ||
	(*(redirect-1)=='2' && isspace(*(redirect-2))) ||
	(*(redirect-1)=='>' && isspace(*(redirect-2))) ||
	(*(redirect-1)=='>' && *(redirect-2)=='2' && isspace(*(redirect-3))))&&
#endif
       ((char*)NULL==singlequote||redirect>singlequote) &&
       ((char*)NULL==doublequote||redirect>doublequote) &&
       ((char*)NULL==paren||redirect>paren) &&
       ((char*)NULL==blacket||redirect>blacket)) {

#ifndef G__OLDIMPLEMENTATION1722
      /* check if redirect both mode
       *   cint> command >& filename
       *                 ^^          */
      if('&'==(*(redirect+1))) {
	mode=G__NUM_STDBOTH;
	++j;
      }
#endif

      /* get filename to redirect 
       *    cint> command > filename
       *                  ^ -------- */
      while((*(redirect+j))) {
	if(!isspace(*(redirect+j))) {
	  filename[i++] = *(redirect+j);
	}
	else if(i) break;
	++j;
      }
      filename[i] = '\0';
      strcpy(pipefile,filename);

      /* get filename to redirect 
       *    cint> command > filename  /keyword
       *                            ^-------- */
      while((*(redirect+j))&&isspace((*(redirect+j)))) ++j;
      if('/'== *(redirect+j)) {
	++j;
	i=0;
	while((*(redirect+j))) keyword[i++] = *(redirect+j++);
	keyword[i] = '\0';
      }


      /* check if append mode
       *   cint> command >> filename 
       *                 ^^          */
      --redirect;
      if('>'==(*redirect)) {
	--redirect;
	openmode="a";
      }
      else {
	openmode="w";
      }

      /* check if stderr 
       *   cint> command 2> filename 
       *                 ^^          */
#ifndef G__OLDIMPLEMENTATION1722
      if(mode==G__NUM_STDBOTH) {
      }
      else 
#endif
      if('2'==(*redirect)) {
	--redirect;
	mode=G__NUM_STDERR; /* stderr */
      }
      else {
	mode=G__NUM_STDOUT; /* stdout */
      }

      /* check if there is a space char right before redirection
       *   cint> command > filename
       *                ^          */
      if(isspace(*redirect) && filename[0]) {
	/* cut command string at redirect command
	 *   cint> command > filename
	 *                ^0          */
	*redirect='\0';


	/* open redirect file */
#ifdef G__REDIRECTIO
	switch(mode) {
	case G__NUM_STDOUT: /* stdout */
	  *psout = G__sout;
#ifndef G__WIN32
	  if (!strlen(stdoutsav)) strcpy(stdoutsav,ttyname(STDOUT_FILENO));
#endif
	  G__sout = freopen(filename,openmode,G__sout);
#ifndef G__OLDIMPLEMENTATION1723
	  G__redirectcout(filename);
#endif
#ifndef G__OLDIMPLEMENTATION1983
#ifdef G__WIN32 /*vo*/
	  handleout = GetStdHandle(STD_OUTPUT_HANDLE);
#endif
#endif /* 1983 */
#ifndef G__OLDIMPLEMENTATION713
	  G__redirect_on();
#endif
	  break;
	case G__NUM_STDERR: /* stderr */
	  *pserr = G__serr;
#ifndef G__WIN32
	  if (!strlen(stderrsav)) strcpy(stderrsav,ttyname(STDERR_FILENO));
#endif
	  G__serr = freopen(filename,openmode,G__serr);
	  /*DEBUG G__dumpfile = G__serr; */
#ifndef G__OLDIMPLEMENTATION1723
	  G__redirectcerr(filename);
#endif
	  break;
#ifndef G__OLDIMPLEMENTATION1722
	case G__NUM_STDBOTH: /* stdout + stderr */
	  *psout = G__sout;
	  *pserr = G__serr;
#ifndef G__WIN32
	  if (!strlen(stdoutsav)) strcpy(stdoutsav,ttyname(STDOUT_FILENO));
	  if (!strlen(stderrsav)) strcpy(stderrsav,ttyname(STDERR_FILENO));
#endif
	  G__sout = freopen(filename,openmode,G__sout);
	  G__serr = freopen(filename,"a",G__serr);
#ifndef G__OLDIMPLEMENTATION1983
#ifdef G__WIN32 /*vo*/
	  handleout = GetStdHandle(STD_OUTPUT_HANDLE);
#endif
#endif /* 1983 */
#ifndef G__OLDIMPLEMENTATION1723
	  G__redirectcout(filename);
	  G__redirectcerr(filename);
#endif
#ifndef G__OLDIMPLEMENTATION713
	  G__redirect_on();
#endif
	  break;
#endif /* 1722 */
	}
#else
	switch(mode) {
	case G__NUM_STDOUT: /* stdout */
	  *psout = G__sout;
	  G__sout = fopen(filename,openmode);
	  break;
	case G__NUM_STDERR: /* stderr */
	  *pserr = G__serr;
	  /*DEBUG G__dumpfile = 0; */
	  G__serr = fopen(filename,openmode);
	  break;
#ifndef G__OLDIMPLEMENTATION1722
	case G__NUM_STDBOTH: /* stdout & stderr */
	  *psout = G__sout;
	  *pserr = G__serr;
	  G__sout = fopen(filename,openmode);
	  G__serr = fopen(filename,"a");
	  break;
#endif
	}
	G__update_stdio(); /* update stdout,stderr,stdin in interpreter */
#endif
      } /* isspace */
    } /* redirect && ... */
  }

#ifdef G__REDIRECTIO
  if(0==asemicolumn ||
     (semicolumn && semicolumn<redirectin &&
      ((char*)NULL==singlequote||semicolumn>singlequote) &&
      ((char*)NULL==doublequote||semicolumn>doublequote))) {
    if(redirectin &&
#ifndef G__OLDIMPLEMENTATION493
       isspace(*(redirectin-1)) &&
#endif
       ((char*)NULL==singlequote||redirectin>singlequote) &&
       ((char*)NULL==doublequote||redirectin>doublequote) &&
       ((char*)NULL==paren||redirectin>paren) &&
       ((char*)NULL==blacket||redirectin>blacket)) {
      /* get filename to redirect
       *    cint> command < filename
       *                  ^ -------- */
      i=0;
      j=1;
      while((*(redirectin+j))) {
	if(!isspace(*(redirectin+j))) {
	  filename[i++] = *(redirectin+j);
	}
	else if(i) break;
	++j;
      }
      filename[i] = '\0';

      *redirectin = '\0';
      --redirectin;
      if(isspace(*redirectin)) {
	*psin = G__sin;
#ifndef G__WIN32
	if (!strlen(stdinsav)) strcpy(stdinsav,ttyname(STDIN_FILENO));
#endif
	G__sin = freopen(filename,"r",G__sin);
#ifndef G__OLDIMPLEMENTATION1723
	G__redirectcin(filename);
#endif
      }
    }
  }
#endif
}

/******************************************************************
* G__unredirectoutput
******************************************************************/
static void G__unredirectoutput(sout,serr,sin,keyword,pipefile)
FILE **sout;
FILE **serr;
FILE **sin;
char *keyword;
char *pipefile;
{
#ifdef G__REDIRECTIO
#ifndef G__OLDIMPLEMENTATION713
  G__redirect_off();
#endif
  if(*sout) {
#ifdef G__WIN32
    G__sout = freopen("CONOUT$","w",G__sout);
#else
    G__sout = freopen(stdoutsav,"w",G__sout);
#endif
    *sout = (FILE*)NULL;
#ifndef G__OLDIMPLEMENTATION1723
    G__unredirectcout();
#endif
#ifndef G__OLDIMPLEMENTATION1983
#ifdef G__WIN32 /*vo*/
    SetStdHandle(STD_OUTPUT_HANDLE, handleout);
#endif
#endif
  }
  if(*serr) {
#ifdef G__WIN32
    G__serr = freopen("CONOUT$","w",G__serr);
#else
    G__serr = freopen(stderrsav,"w",G__serr);
#endif
    *serr = (FILE*)NULL;
#ifndef G__OLDIMPLEMENTATION1723
    G__unredirectcerr();
#endif
  }
  if(*sin) {
#ifdef G__WIN32
    *sin = freopen("CONIN$","r",*sin);
#else
    *sin = freopen(stdinsav,"r",*sin);
#endif
    *sin = (FILE*)NULL;
#ifndef G__OLDIMPLEMENTATION1723
    G__unredirectcin();
#endif
  }
#else /* G__REDIRECTIO */
  int flag=0; 
#ifndef G__OLDIMPLEMENTATION713
  G__redirect_off();
#endif
  if(*sout) {
    fclose(G__sout);
    G__sout = *sout;
    ++flag;
  }
  if(*serr) {
    fclose(G__serr);
    G__serr = *serr;
    ++flag;
  }
  if(*sin) {
    fclose(G__sin);
    G__sin = *sin;
    ++flag;
  }
  if(flag) G__update_stdio();
#endif /* G__REDIRECTIO */

  if(pipefile[0] && keyword[0]) {
#ifndef G__OLDIMPLEMENTATION1917
    FILE *keyfile = fopen(pipefile,"r");
    G__display_keyword(G__sout,keyword,keyfile);
    fclose(keyfile);
#else
    G__display_keyword(G__sout,keyword,pipefile);
#endif
  }
}
#endif

#ifndef G__OLDIMPLEMENTATION892
/******************************************************************
* G__IsBadCommand
******************************************************************/
int G__IsBadCommand(com)
char *com;
{
  int i=0;
  int nest=0;
  int single_quote=0;
  int double_quote=0;
#ifndef G__OLDIMPLEMENTATION1774
  int semicolumnattheend=0;
#endif
  while(com[i]!='\0') {
  readagain:
    switch(com[i]) {
    case '"':
      if(single_quote==0) double_quote ^= 1;
      break;
    case '\'':
      if(double_quote==0) single_quote ^= 1;
      break;
    case '{':
    case '(':
    case '[':
      if((single_quote==0)&&(double_quote==0)) nest++;
      break;
    case '}':
    case ')':
    case ']':
      if((single_quote==0)&&(double_quote==0)) nest--;
      break;
    case '\\':
      ++i;
#ifndef G__OLDIMPLEMENTATION1245
      if(0==com[i] || '\n'==com[i]) {
	--i;
	strcpy(com+i,G__input("> "));
#ifndef G__OLDIMPLEMENTATION2139
	if(G__return==G__RETURN_IMMEDIATE) return(-1);
#endif
      }
#endif
      break;
#ifndef G__OLDIMPLEMENTATION1113
    case '/':
      if((single_quote==0)&&(double_quote==0)) {
	if('/'==com[i+1]) {
	  com[i]=0;
	  com[i+1]=0;
	}
      }
      break;
#endif
    }
#ifndef G__OLDIMPLEMENTATION1774
    if(';'==com[i]) {
      if((single_quote==0)&&(double_quote==0)&&(nest==0)) semicolumnattheend=1;
    }
    else {
      if(!isspace(com[i])) semicolumnattheend=0;
    }
#endif
    ++i;
  }
  /* #define G__OLDIMPLEMENTATION1774 */
#ifndef G__OLDIMPLEMENTATION1774
#ifndef G__OLDIMPLEMENTATION1801
  if(nest>0 && '{'!=com[0]) {
    if(0==strncmp(com,"for(",4) || 0==strncmp(com,"for ",4) 
      || 0==strncmp(com,"while(",6) || 0==strncmp(com,"while ",6) 
      || 0==strncmp(com,"do ",3) || 0==strncmp(com,"do{",3) 
      || 0==strncmp(com,"namespace ",10) || 0==strncmp(com,"namespace{",10)) {
      strcpy(com+i,G__input("end with '}', '@':abort > "));
    }
    else {
      strcpy(com+i,G__input("end with ';', '@':abort > "));
    }
#ifndef G__OLDIMPLEMENTATION2139
    if(G__return==G__RETURN_IMMEDIATE) return(-1);
#endif
    if('@'==com[i]) {
      com[0]=0;
      return(0);
    }
    goto readagain;
  }
#endif
  if(0<nest) return(1);
  if(G__INPUTCXXMODE==G__rootmode && 0==nest && 0==semicolumnattheend
     && '#'!=com[0]
#ifndef G__OLDIMPLEMENTATION1801
     && 0!=strncmp(com,"for(",4) && 0!=strncmp(com,"for ",4) 
     && 0!=strncmp(com,"while(",6) && 0!=strncmp(com,"while ",6) 
     && 0!=strncmp(com,"do ",3) && 0!=strncmp(com,"do{",3) 
     && 0!=strncmp(com,"namespace ",10) && 0!=strncmp(com,"namespace{",10)
#endif
     ) {
    strcpy(com+i,G__input("end with ';', '@':abort > "));
#ifndef G__OLDIMPLEMENTATION2139
    if(G__return==G__RETURN_IMMEDIATE) return(-1);
#endif
    if('@'==com[i]) {
      com[0]=0;
      return(0);
    }
    goto readagain;
  }
  if(single_quote || double_quote || nest<0) return(-1);
  return(0);
#else
  if(0!=nest || single_quote || double_quote) return(1);
  else return(0);
#endif
}
#endif

#ifndef G__OLDIMPLEMENTATION1795
/******************************************************************
* G__ReadInputMode
******************************************************************/
int G__ReadInputMode()
{
  static int inputmodeflag=0;
  if(inputmodeflag==0) {
    char *inputmodebuf;
    inputmodeflag=1;
    inputmodebuf=G__getmakeinfo1("INPUTMODE");
    if(inputmodebuf && inputmodebuf[0]) {
      if(strstr(inputmodebuf,"c++")||strstr(inputmodebuf,"C++")) 
	G__rootmode=G__INPUTCXXMODE;
      else if(strstr(inputmodebuf,"root")||strstr(inputmodebuf,"ROOT")) 
	G__rootmode=G__INPUTROOTMODE;
      else if(strstr(inputmodebuf,"cint")||strstr(inputmodebuf,"CINT"))
	G__rootmode=G__INPUTCINTMODE;
    }
    inputmodebuf=G__getmakeinfo1("INPUTMODELOCK");
    if(inputmodebuf && inputmodebuf[0]) {
      if(strstr(inputmodebuf,"on")||strstr(inputmodebuf,"ON")) 
	G__lockinputmode=1;
      else if(strstr(inputmodebuf,"off")||strstr(inputmodebuf,"OFF")) 
	G__lockinputmode=0;
    }
  }
  return(G__rootmode);
}
#endif

/******************************************************************
* G__debugvariable()
******************************************************************/
void G__debugvariable(fp,var,name) 
FILE* fp;
struct G__var_array *var;
char* name;
{
  int ig15;
  int i;
  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if(var->hash[ig15] && strcmp(var->varnamebuf[ig15],name)==0) {
        fprintf(fp
         ,"%s p=%ld type=%c typenum=%d tagnum=%d const=%x static=%d\n paran=%d "
		,var->varnamebuf[ig15]
		,var->p[ig15]
		,var->type[ig15]
		,var->p_typetable[ig15]
		,var->p_tagtable[ig15]
		,var->constvar[ig15]
		,var->statictype[ig15]
		,var->paran[ig15]
	        );
        i=0;
        while(var->varlabel[ig15][i]) {
          fprintf(fp,"[%d]",var->varlabel[ig15][i++]);
        }
        fprintf(fp,"\n");
      }
    }
    var=var->next;
  }
}

/******************************************************************
* intG__process_cmd()
******************************************************************/
int G__process_cmd(line, prompt, more, err, rslt)
char *line;
char *prompt;
int  *more;
int  *err;
G__value *rslt;
{
  FILE *tempfp;   /* used for input dump file */
  char command[G__LONGLINE];
  char syscom[G__LONGLINE];
  char editor[64];
  int temp,temp1=0,temp2;
  int index = -1;
  int ignore=G__PAUSE_NORMAL;
  short double_quote,single_quote;
#ifdef G__ASM
  G__ALLOC_ASMENV;
#endif
  char store_var_type;

/* pass to parent otherwise not re-entrant */
#ifdef G__TMPFILE
  static char tname[G__MAXFILENAME];
#if !defined(G__OLDIMPLEMENTATION2092)
  char sname[G__MAXFILENAME];
#elif !defined(G__OLDIMPLEMENTATION1794)
#else
  char sname[G__MAXFILENAME];
#endif
#else
  static char tname[L_tmpnam+10];
#if !defined(G__OLDIMPLEMENTATION2092)
  char sname[L_tmpnam+10];
#elif !defined(G__OLDIMPLEMENTATION1794)
#else
  char sname[L_tmpnam+10];
#endif
#endif
  static struct G__input_file ftemp;

  /* struct G__ifunc_table *ifunc; */

  struct G__var_array *local;

  char *com,*string;
  char *stringb;
  int line_number,filenum;

  G__value buf;

  FILE *G__temp;

  char *evalbase;
  /* void *evalp; */
  int base=0 /* ,digit */ ,num;
  char evalresult[G__ONELINE];
#ifndef G__OLDIMPLEMENTATION464
  FILE* store_stderr=NULL;
  FILE* store_stdout=NULL;
  FILE* store_stdin=NULL;
  char keyword[G__ONELINE];
  char pipefile[G__MAXFILENAME];
  int dmy = 0;
#ifndef G__OLDIMPLEMENTATION1474
  int noprintflag=0;
#endif
#ifndef G__OLDIMPLEMENTATION2092
  int istmpnam=0;
#endif

  if(!err) err = &dmy;

#ifndef G__OLDIMPLEMENTATION1795
 G__ReadInputMode();
#endif

#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif

  keyword[0] = 0;
  pipefile[0] = 0;
#endif

#ifndef G__OLDIMPLEMENTATION860
  if(0==init_process_cmd_called) 
    G__fprinterr(G__serr,"Internal error: G__init_process_cmd must be called before G__process_cmd\n");
#endif

#ifndef G__OLDIMPLEMENTATION685
  if(strlen(line)>G__LONGLINE-5) {
    G__fprinterr(G__serr,"!!! User command too long (%d>%d)!!!\n"
		 ,strlen(line),G__LONGLINE-5);
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
    return(0);
  }
#endif


  *prompt = '\0';
#ifdef G__OLDIMPLEMENTATION463
  if(!fout) fout=G__sout;
#endif

  
  if ( (com = getenv("EDITOR")) ) strcpy(editor, com);
#ifdef G__WIN32
  else                            strcpy(editor, "notepad");
#else
  else                            strcpy(editor, "vi");
#endif
  
  /*** begin ***/
  
  temp = 0;
  while (isspace(line[temp])) ++temp;
  
  if (*more == 0) {
    strcpy(command, line+temp);
    com = strchr(command,'\n');
    if (com) *com = '\0';
    else {
      com = strchr(command,'\r');
      if (com) *com = '\0';
    }
    
    /* #ifdef G__ROOT */
    if(G__INPUTROOTMODE&G__rootmode) {
      if (command[0] == '.') {
	strcpy(syscom, command+1);
	strcpy(command, syscom);
      } 
      else if (strcmp(command, "?") == 0) {
	strcpy(command, "help");
      } 
      else if ('\0'!=command[0] && command[0] != '{') {
#ifndef G__OLDIMPLEMENTATION892
#ifndef G__OLDIMPLEMENTATION1774
	switch(G__IsBadCommand(command)) {
        case 0:
	  break;
        case 1:
	  com=command;
	  goto multi_line_command;
        case -1:
	default:
	  fprintf(stderr,"!!!Bad command input. Ignored!!!\n");
	  G__UnlockCriticalSection();
	  return(ignore=G__PAUSE_NORMAL);
	  break;
	}
#else /* 1774 */
	if(G__IsBadCommand(command)) {
	  fprintf(stderr,"!!!Bad command input. Ignored!!!\n");
#ifndef G__OLDIMPLEMENTATION1035
	  G__UnlockCriticalSection();
#endif
	  return(ignore=G__PAUSE_NORMAL);
	}
#endif /* 1774 */
#endif
#ifndef G__OLDIMPLEMENTATION464
	G__redirectoutput(command,&store_stdout,&store_stderr,&store_stdin,1
			  ,keyword,pipefile);
#endif
	temp = strlen(command)-1;
	while (isspace(command[temp])) --temp;
	if (command[temp] == ';') {
	  sprintf(syscom, "{%s}", command);
#ifndef G__OLDIMPLEMENTATION1474
#ifndef G__OLDIMPLEMENTATION1774
	  if(G__INPUTCXXMODE!=G__rootmode) noprintflag=1;
#else
	  noprintflag=1;
#endif
#endif
	}
	else
	  sprintf(syscom, "{%s;}", command);
	strcpy(command, syscom);
      }
    }
    /* #endif */
  } else {
    sprintf(command, "{%s", line+temp);
    com = strchr(command,'\n');
    if (com) *com = '\0';
    else {
      com = strchr(command,'\r');
      if (com) *com = '\0';
    }
  }

  com=command;

#ifndef G__OLDIMPLEMENTATION464
  if(strlen(com)>=3&&
     0!=strncmp(com,"{",1)&&0!=strncmp(com,"p",1)&&
     0!=strncmp(com,">",1)&&0!=strncmp(com,"<",1)&&
     0!=strncmp(com,"2>",2)) {
    G__redirectoutput(com,&store_stdout,&store_stderr,&store_stdin,0
		      ,keyword,pipefile);
  }
#endif
  
  temp=0;
  if(isalpha(command[temp])) 
    while(isalpha(command[temp])) ++temp;
  else if(!isdigit(command[temp]))
    ++temp;
  string=command+temp;

  stringb=string;
  while(isspace(*stringb)&&(*stringb)) ++stringb;
  
#ifndef G__OLDIMPLEMENTATION1031
  G__storerewindposition();
#endif

  /************************************
   * cint>    X  argument
   *          ^^
   *          com string
   ************************************/
  
  /************************************
   * Break prompt command parser
   ************************************/
  
#ifndef G__OLDIMPLEMENTATION967
    if(strncmp("class",com,3)==0) {
      char *p = strchr(string,'/');
      if(p) {
#ifndef G__OLDIMPLEMENTATION1320
	*p = 0;
#endif
	++p;
      }
      G__display_classkeyword(G__sout,string,p,0);
    }
    else if(strncmp("Class",com,3)==0) {
      char *p = strchr(string,'/');
      if(p) {
#ifndef G__OLDIMPLEMENTATION1320
	*p = 0;
#endif
	++p;
      }
      G__display_classkeyword(G__sout,string,p,1);
    }
#else
    if(strncmp("class",com,3)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_class(G__sout,string,0,0);
    }
    else if(strncmp("Class",com,3)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_class(G__sout,string,1,0);
    }
#endif

    else if(strncmp("typedef",com,3)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_typedef(G__sout,string,0);
    }

#ifndef G__OLDIMPLEMENTATION873
    else if(strncmp("newtypes",com,4)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_newtypes(G__sout,stringb);
    }
#endif

#ifndef G__OLDIMPLEMENTATION1253
    else if(strncmp("tempobj",com,5)==0) {
      fprintf(G__sout,"!!!Display temp object stack!!!\n");
      G__display_tempobj(G__sout);
    }
#endif

    else if(strncmp("template",com,4)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_template(G__sout,string);
    }

    else if(strncmp("macro",com,3)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_macro(G__sout,string);
    }

    else if(strncmp("include",com,4)==0) {
      if(0==(*stringb)) G__display_includepath(G__sout);
      else {
	string = stringb+strlen(stringb)-1;
	while(isspace(*string)) *(string--) = 0;
	G__add_ipath(stringb);
      }
    }

#ifndef G__OLDIMPLEMENTATION1480
    else if(strncmp("language",com,4)==0) {
      int ix=0;
      while(stringb[ix]) { stringb[ix] = toupper(stringb[ix]); ++ix; }
      if(strncmp(stringb,"EUC",3)==0) G__lang = G__EUC;
      else if(strncmp(stringb,"SJIS",3)==0) G__lang = G__SJIS;
      else if(strncmp(stringb,"JIS",3)==0) G__lang = G__JIS;
      else if(strncmp(stringb,"EUROPEAN",3)==0) G__lang = G__ONEBYTE;
      else if(strncmp(stringb,"UNKNOWN",3)==0) G__lang = G__UNKNOWNCODING;
      else G__lang = G__int(G__calc(stringb));
    }
#endif

    else if(strncmp("J",com,1)==0) {
      G__dispmsg = G__int(G__getexpr(string));
    }

    else if(strncmp("file",com,4)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_files(G__sout);
    }

    else if(strncmp("string",com,4)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_string(G__sout);
    }

    else if(strncmp("garbage",com,4)==0) {
      G__more_pause((FILE*)NULL,1);
      G__disp_garbagecollection(G__sout);
    }
    else if(strncmp("Garbage",com,4)==0) {
      G__more_pause((FILE*)NULL,1);
      G__garbagecollection();
      G__disp_garbagecollection(G__sout);
    }

    else if(strncmp("trace",com,3)==0) {
      G__set_tracemode(string);
    }

    else if(strncmp("deltrace",com,4)==0) {
      G__del_tracemode(string);
    }

    else if(strncmp("break",com,4)==0) {
      G__set_classbreak(string);
    }

    else if(strncmp("delbreak",com,4)==0) {
      G__del_classbreak(string);
    }

#ifndef G__OLDIMPLEMENTATION856
    else if(strncmp("ls",com,2)==0 ||
	    strncmp("ll",com,2)==0 ||
	    strncmp("dir",com,3)==0 ||
	    strncmp("pwd",com,3)==0 ||
	    strncmp("cp",com,2)==0 ||
	    strncmp("copy",com,4)==0 ||
	    strncmp("gmake",com,5)==0 ||
	    strncmp("make",com,4)==0 ||
	    strncmp("mv",com,2)==0 ||
	    strncmp("move",com,4)==0 ||
	    strncmp("mkdir",com,5)==0 ||
	    strncmp("vi",com,2)==0 ||
	    strncmp("notepad",com,7)==0 ||
	    strncmp("grep",com,4)==0 ||
	    strncmp("fgrep",com,5)==0 ||
	    strncmp("egrep",com,5)==0 ||
	    strncmp("diff",com,4)==0 ||
	    strncmp("wc",com,2)==0 ||
	    strncmp("cat",com,3)==0 ||
	    strncmp("more",com,4)==0 ||
	    strncmp("less",com,4)==0 ||
	    strncmp("echo",com,4)==0 ||
	    strncmp("cint",com,4)==0 ||
	    strncmp("du",com,2)==0 ||
	    strncmp("sh",com,2)==0 ||
	    strncmp("csh",com,3)==0 ||
	    strncmp("ksh",com,3)==0 ||
	    strncmp("bash",com,4)==0 ||
	    strncmp("man",com,3)==0 ||
	    strncmp("type",com,4)==0) {
      system(com);
    }
    else if(strncmp("rm",com,2)==0 ||
	    strncmp("del",com,4)==0 ||
	    strncmp("rmdir",com,5)==0) {
      char localbuf[80];
      strcpy(localbuf,G__input("Are you sure(y/n)? "));
      if(tolower(localbuf[0])=='y') system(com);
      else fprintf(G__sout,"aborted\n");
    }
    else if(strncmp("set",com,3)==0 ||
	    strncmp("export",com,6)==0) {
      char *plocal = strchr(stringb,'=');
      if(plocal) {
#if defined(G__WIN32)
	*plocal++ = 0;
	if(FALSE==SetEnvironmentVariable(stringb,plocal)) 
	  G__fprinterr(G__serr,"can not set environment variable %s=%s\n"
		  ,stringb,plocal);
#elif defined(G__POSIX)
	if(0!=putenv(stringb))  /* DOES NOT WORK , WHY ??? */
	  G__fprinterr(G__serr,"can not set environment variable %s\n",stringb);
#endif
      }
      else {
	system(com);
      }
    }
    else if(strncmp("cd",com,2)==0) {
#if defined(G__WIN32)
      if(FALSE==SetCurrentDirectory(stringb))
	G__fprinterr(G__serr,"can not change directory to %s\n",stringb);
#elif defined(G__POSIX)
      if(0!=chdir(stringb)) 
	G__fprinterr(G__serr,"can not change directory to %s\n",stringb);
#endif
    }
#endif

#ifndef G__OLDIMPLEMENTATION1094
    else if(strncmp("undo",com,4)==0||strncmp("rewind",com,3)==0) {
      G__rewind_undo_position();
    }
#endif

#ifdef G__SECURITY
    else if(strncmp("where",com,4)==0) {
      fprintf(G__sout,"FILE:%s LINE:%d interpreter=%s\n"
	      ,G__stripfilename(view.file.name),view.file.line_number,G__nam);
    }

    else if(strncmp("security",com,4)==0) {
      fprintf(G__sout,"security ");
      switch(G__security) {
      case 0x0000000f: fprintf(G__sout,"level0"); break;
      case 0x000000ff: fprintf(G__sout,"level1"); break;
      case 0x00000fff: fprintf(G__sout,"level2"); break;
      case 0x0000ffff: fprintf(G__sout,"level3"); break;
      case 0x000fffff: fprintf(G__sout,"level4"); break;
      case 0x00ffffff: fprintf(G__sout,"level5"); break;
      case 0x0fffffff: fprintf(G__sout,"level6"); break;
      case 0xffffffff: fprintf(G__sout,"level7"); break;
      default: break;
      }
#ifdef G__64BIT
      fprintf(G__sout,"  0x%x\n",G__security);
#else
      fprintf(G__sout,"  0x%lx\n",G__security);
#endif
    }
    else if(strncmp("Security",com,4)==0) {
      G__security = G__int(G__calc_internal(string));
    }

#ifndef G__OLDIMPLEMENTATION1033
    else if(strncmp("refcount",com,3)==0) {
      if(G__security&G__SECURE_GARBAGECOLLECTION) {
        G__security &= ~G__SECURE_GARBAGECOLLECTION;
        fprintf(G__sout,"!!!new/delete reference count control turned off");
      }
      else {
        G__security |= G__SECURE_GARBAGECOLLECTION;
        fprintf(G__sout,"!!!new/delete reference count control turned on");
      }
      fprintf(G__sout,"  0x%lx\n",G__security);
    }
#endif

#endif /* G__SECURITY */

    else if(strncmp("scratch",com,4)==0) {
      if(!G__isfilebusy(0)) {
	G__scratch_all();
#ifndef G__OLDIMPLEMENTATION1094
	G__init_undo();
#endif
      }
    }

#ifndef G__ROOT
    else if(strncmp(".",com,1)==0) {
#ifndef G__OLDIMPLEMENTATION1795
      if(0==G__lockinputmode) {
	G__rootmode ^= 1;
	fprintf(G__sout,"!!!Debugger Command mode switched as follows!!!\n");
	if(G__INPUTROOTMODE&G__rootmode) {
	  fprintf(G__sout,"    > .[command]\n");
	  fprintf(G__sout,"    > [statement]\n");
	}
	else {
	  fprintf(G__sout,"    > [command]\n");
	  fprintf(G__sout,"    > { [statement] }\n");
	}
      }
      else {
	fprintf(G__sout,"!!!Debugger Command mode locked!!!\n");
      }
#else
      G__rootmode ^= 1;
      fprintf(G__sout,"!!!Debugger Command mode switched as follows!!!\n");
      if(G__rootmode) {
	fprintf(G__sout,"    > .[command]\n");
	fprintf(G__sout,"    > [statement]\n");
      }
      else {
	fprintf(G__sout,"    > [command]\n");
	fprintf(G__sout,"    > { [statement] }\n");
      }
#endif
    }
#endif


    else if(strncmp("reset",com,4)==0) {
#ifdef G__ROOT
      G__fprinterr(G__serr,"!!! Sorry, can not reset interpreter !!!\n");
#else
      if(!G__isfilebusy(0)) { 
	int store_othermain=G__othermain;
	G__scratch_all(); /* To make memory leak check work */
	if(G__commandline[0]) G__init_cint(G__commandline);
	else {
#ifndef G__OLDIMPLEMENTATION1961
	  G__init_cint(G__nam);
#else
	  G__init_cint("cint");
#endif
	}
	G__othermain=store_othermain;
#ifndef G__OLDIMPLEMENTATION1094
	G__init_undo();
#endif
      }
      else {
	G__fprinterr(G__serr,"!!! Sorry, can not reset interpreter !!!\n");
      }
#endif
    }

#ifndef G__FONS4
    else if(strncmp("function",com,4)==0) {
#else
    else if(strncmp("function",com,4)==0 || strncmp("L",com,1)==0) {
#endif

      /*******************************************************
       * List up interpreted functions and shared library
       *******************************************************/
      G__more_pause((FILE*)NULL,1);
      G__listfunc(G__sout,G__PUBLIC_PROTECTED_PRIVATE,(char*)NULL
		    ,(struct G__ifunc_table*)NULL);
      G__listshlfunc(G__sout);
    }

    else if(strncmp("interactive",com,5)==0) {
      G__interactive^=1;
      fprintf(G__sout,"interactive mode=%d\n",G__interactive);
    }

    else if(strncmp("coverage",com,5)==0) {
      temp=0;
      while(isspace(string[temp])) temp++;
#ifndef G__OLDIMPLEMENTATION1008
      if(isalpha(string[temp]))
	tempfp=fopen(string+temp,"w");
      else
	tempfp=(FILE*)NULL;
#else
      tempfp=fopen(string+temp,"w");
#endif
      if(tempfp) {
	fprintf(G__sout,"saving trace coverage to %s\n",string+temp);
	G__dump_tracecoverage(tempfp);
	fclose(tempfp);
      }
      else {
	fprintf(G__sout,"can not open file %s\n",string+temp);
      }
    }

    else if(strncmp("debug",com,5)==0) {
/* #ifdef G__ASM_DBG */
      G__asm_dbg ^= 1;
      fprintf(G__sout,"G__asm_dbg=%d\n",G__asm_dbg);
/* #endif */
    }

    else if(strncmp("asmstep",com,5)==0) {
#ifdef G__ASM_DBG
      G__asm_step ^= 1;
      fprintf(G__sout,"G__asm_step=%d\n",G__asm_step);
#endif
    }

    else if(strncmp("dasm",com,4)==0) {
#if defined(G__ASM_DBG) || !defined(G__OLDIMPLEMENTATION1270)
      G__dasm(G__sout,0);
#endif
    }

    else if(strncmp("stack",com,4)==0) {
#ifdef G__ASM_DBG
      /* G__display_stack(); */
#endif
    }

#ifndef G__OLDIMPLEMENTATION1815
    else if(strncmp("exception",com,4)==0) {
      G__catchexception ^= 1;
      fprintf(G__sout,"G__catchexception=%d\n",G__catchexception);
    }
#endif

    else if(strncmp("status",com,4)==0) {
#ifdef G__ASM_DBG
      fprintf(G__sout,"G__asm_noverflow=%d\n",G__asm_noverflow);
      fprintf(G__sout,"G__no_exec_compile=%d\n",G__no_exec_compile);
      fprintf(G__sout,"G__no_exec=%d\n",G__no_exec);
      fprintf(G__sout,"G__asm_exec=%d\n",G__asm_exec);
      fprintf(G__sout,"G__asm_loopcompile=%d\n",G__asm_loopcompile);
      fprintf(G__sout,"G__asm_cp=%d\n",G__asm_cp);
      fprintf(G__sout,"G__asm_dt=%d\n",G__asm_dt);
#endif
    }

    else if(strncmp("OPTIMIZE",com,1)==0) {
      G__asm_loopcompile = G__int(G__calc_internal(string));
#ifndef G__OLDIMPLEMENTATION1155
      G__asm_loopcompile_mode = G__asm_loopcompile; 
#endif
    }
#ifdef G__ASM_WHOLEFUNC
    else if(strncmp("WHOLEFUNC",com,3)==0) {
      G__asm_wholefunction = G__int(G__calc_internal(string));
    }
#endif

    /* interactive return of undefined symbol */
    else if(strncmp("return",com,1)==0) {
      if(G__interactive_undefined) {
	G__interactivereturnvalue = G__calc_internal(string);
	G__pause_return=1;
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
	ignore = G__PAUSE_NORMAL;
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(ignore);
      }
      else {
	G__fprinterr(G__serr,"!!! Use 'return' command at your own risk !!!\n");
	G__interactivereturnvalue = G__calc_internal(string);
	G__return=G__RETURN_IMMEDIATE;
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
	ignore = 2;
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(ignore);
      }
    }

#ifndef G__OLDIMPLEMENTATION1722
    else if(strncmp(">&",com,2)==0 || strncmp(">>&",com,3)==0) {
      if(strncmp(">>&",com,3)==0) index=3;
      else                        index=2;
      while(isspace(command[index])&&command[index]!='\0') index++;
      if((*(command+index))) {
	if(G__sout!=G__stdout) {
	  fprintf(G__stdout,"Old save file closed\n");
	  fclose(G__sout);
	}
	if(G__serr!=G__stderr && G__serr!=G__sout) {
	  fprintf(G__stdout,"Old save file closed\n");
	  fclose(G__serr);
	}
	if(strncmp(">>",com,2)!=0) {
	  G__sout=fopen(command+index,"w");
	  fclose(G__sout);
	}
	G__serr=G__sout=fopen(command+index,"a");
	G__redirectcout(command+index);
	G__redirectcerr(command+index);
	if(G__sout) {
	  fprintf(G__stdout,"Output will be saved in file %s! ('>&' to display on screen)\n" 
		 ,command+index);
	}
	else {
	  G__sout = G__stdout;
	  G__serr = G__stderr;
	  fprintf(G__stdout,"Can not open file %s\n",command+index);
	}
      }
      else {
	if(G__sout && G__sout!=G__stdout) {
	  G__unredirectcout();
	  G__unredirectcerr();
	  fclose(G__sout);
	}
	G__sout = G__stdout;
	G__serr = G__stderr;
	fprintf(G__stdout,"Output will be displayed on screen!\n");
      }
#ifndef G__OLDIMPLEMENTATION713
      G__update_stdio();
#endif
    }
#endif /* 1722 */

    else if(strncmp(">",com,1)==0) {
      if(strncmp(">>",com,2)==0) index=2;
      else                       index=1;
      while(isspace(command[index])&&command[index]!='\0') index++;
      if((*(command+index))) {
	if(G__sout!=G__stdout) {
	  fprintf(G__stdout,"Old save file closed\n");
	  fclose(G__sout);
	}
#ifndef G__OLDIMPLEMENTATION1635
	if(strncmp(">>",com,2)!=0) {
	  G__sout=fopen(command+index,"w");
	  fclose(G__sout);
	}
	G__sout=fopen(command+index,"a");
	G__redirectcout(command+index);
#else /* 1635 */
	if(strncmp(">>",com,2)==0) G__sout=fopen(command+index,"a");
	else                       G__sout=fopen(command+index,"w");
#endif /* 1635 */
	if(G__sout) {
	  fprintf(G__stdout,"Output will be saved in file %s! ('>' to display on screen)\n" 
		 ,command+index);
	}
	else {
	  G__sout = G__stdout;
	  fprintf(G__stdout,"Can not open file %s\n",command+index);
	}
      }
      else {
	if(G__sout && G__sout!=G__stdout) {
#ifndef G__OLDIMPLEMENTATION1635
	  G__unredirectcout();
#endif /* 1635 */
	  fclose(G__sout);
	}
	G__sout = G__stdout;
	fprintf(G__stdout,"Output will be displayed on screen!\n");
      }
#ifndef G__OLDIMPLEMENTATION713
      G__update_stdio();
#endif
    }

    else if(strncmp("2>",com,2)==0) {
      if(strncmp("2>>",com,3)==0) index=3;
      else                        index=2;
      while(isspace(command[index])&&command[index]!='\0') index++;
      if((*(command+index))) {
#ifndef G__OLDIMPLEMENTATION1635
	if(strncmp(">>",com,2)!=0) {
	  G__serr=fopen(command+index,"w");
	  fclose(G__serr);
	  /* fclose(G__sout); */
	}
	G__serr=fopen(command+index,"a");
	G__redirectcerr(command+index);
#else /* 1635 */
	if(strncmp("2>>",com,3)==0) G__serr=fopen(command+index,"a");
	else                        G__serr=fopen(command+index,"w");
#endif /* 1635 */
	if(G__serr) {
	  fprintf(G__stdout,"Error will be saved in file %s! ('2>' to display on screen)\n" 
		 ,command+index);
	}
	else {
	  G__serr = G__stderr;
	  fprintf(G__stdout,"Can not open file %s\n",command+index);
	}
      }
      else {
	if(G__serr && G__serr!=G__stderr) {
#ifndef G__OLDIMPLEMENTATION1635
	  G__unredirectcerr();
#endif /* 1635 */
	  fclose(G__serr);
	}
	G__serr = G__stderr;
	fprintf(G__sout,"Error will be displayed on screen!\n");
      }
#ifndef G__OLDIMPLEMENTATION713
      G__update_stdio();
#endif
    }


    else if(strncmp("<",com,1)==0) {
      /*******************************************************
       * Execute dumpped readline file
       *******************************************************/
      index=1;
      while(isspace(command[index])&&command[index]!='\0') index++;
      tempfp=fopen(command+index,"r");
      if(tempfp) {
	fprintf(G__sout,"Execute readline dumpfile %s\n"
		,command+index);
	G__pushdumpinput(tempfp,1);
      }
      else {
	fprintf(G__sout,"Can not open file %s\n",command+1);
      }
#ifndef G__OLDIMPLEMENTATION713
      G__update_stdio();
#endif
    }

#ifndef G__FONS4
    else if (strncmp("X",com,1)==0) {
#else
    else if (strncmp("x",com,1)==0) {
#endif
      /*******************************************************
       * Execute C/C++ source file. Filename minus extension
       * must match a function in the source file. This function
       * will be automatically executed.
       *******************************************************/ 
#ifndef G__FONS85
      char *com1;
#endif
      temp=0;
      while(isspace(string[temp])) temp++;
#ifndef G__FONS4
      if (string[temp] == '\0') {
        G__fprinterr(G__serr,"Error: no file specified\n");
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
        return(ignore);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1094
      G__store_undo_position();
#endif
#ifndef G__OLDIMPLEMENTATION1105
      com = strchr(string+temp, '(');
#else
      com = strrchr(string+temp, '(');
#endif
      if (com) {
	char *px = com-1;
	*com = '\0';
	while(isspace(*px)) *px-- = '\0';
      }
      temp2 = G__prerun;
      G__prerun = 1;  /* suppress warning message if file already loaded */
      temp1 = G__loadfile(string+temp);
      if (temp1 == 1) {
#ifndef G__FONS20
	 G__prerun = 0;
#endif 
         G__unloadfile(string+temp);
#ifndef G__OLDIMPLEMENTATION1102
	 G__storerewindposition();
#endif
         if (G__loadfile(string+temp)) {
            G__prerun = temp2;
	    G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	    G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifndef G__OLDIMPLEMENTATION1094
	    if(G__security_error) G__cancel_undo_position();
#endif
#ifdef G__SECURITY
	    *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	    G__UnlockCriticalSection();
#endif
            return(ignore);
         }
      }
#ifdef G__OLDIMPLEMENTATION487
#ifdef G__AUTOCOMPILE
      /*************************************************************
       * if '#pragma compile' appears in source code.
       *************************************************************/
      if(G__fpautocc) G__autocc();
#endif
#endif

      G__prerun = temp2;
      com1= strrchr(string+temp,'.');
      if(com) {
	char *px = com-1;
	*com='(';
	while(0==(*px)) *px-- = ' ';
      }
      com = com1;
      if (com) {
#ifndef G__FONS20
	 char *s;
         *com = '\0';
	 s = strrchr(string+temp,'/');
	 if(!s) s=strrchr(string+temp,'\\');
	 if(!s) s=string+temp;
	 else   s++;
	 strcpy(syscom, s);
#ifndef G__OLDIMPLEMENTATION1172
         s=syscom;
         while(s && *s) {
           if('-'==(*s)) *s='_';
           ++s;
         }
#endif
	 string = strchr(com+1, '(');
	 if (string)
	   strcat(syscom, string);
	 else
	   strcat(syscom, "()");
#else
         *com = '\0';
	 com = strrchr(string+temp,'/');
	 if(!com) strrchr(string+temp,'\\');
	 if(!com) com=string+temp;
	 else     com++;
         sprintf(syscom,"%s()",com);
#endif
#ifndef G__FONS4
         buf=G__calc_internal(syscom);
	 if(rslt) *rslt = buf;
         G__in_pause=1;
         G__valuemonitor(buf,syscom);
         G__in_pause=0;
#ifndef G__OLDIMPLEMENTATION1259
	 if(buf.isconst&(G__CONSTVAR|G__CONSTFUNC)) {
	   char tmp[G__ONELINE];
	   sprintf(tmp,"(const %s",syscom+1);
	   strcpy(syscom,tmp);
	 }
	 if(buf.isconst&G__PCONSTVAR) {
	   char tmp2[G__ONELINE];
	   char *ptmp = strchr(syscom,')');
	   strcpy(tmp2,ptmp);
	   strcpy(ptmp,"const");
	   strcat(syscom,tmp2);
	 }
#endif
#ifndef G__OLDIMPLEMENTATION1079
         if(buf.type && 0==G__atevaluate(buf)) fprintf(G__sout,"%s\n",syscom);
#else
         fprintf(G__sout,"%s\n",syscom);
#endif
#ifdef G__SECURITY
	 *err |= G__security_recover(G__serr);
#endif
#else
         G__calc_internal(syscom);
#endif
      } else {
         fprintf(G__sout, "Expecting . in filename\n");
      }
    }

#ifndef G__OLDIMPLEMENTATION1108
    else if(
#ifndef G__OLDIMPLEMENTATION1273
	    (G__do_smart_unload && strncmp("L",com,1)==0) ||
	    strncmp("Lall",com,2)==0
#else
	    strncmp("Lall",com,2)==0
#endif
	    ) {
      temp=0;
      while(isspace(string[temp])) temp++;
      G__UnlockCriticalSection();
      G__reloadfile(string+temp);
      G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
			  ,keyword,pipefile);
      *err |= G__security_recover(G__serr);
      return(ignore);
    }
#endif

#ifndef G__FONS4
    else if (strncmp("L",com,1)==0 || strncmp("Load",com,4)==0) {
      /*******************************************************
       * Load(Re-Load) a C/C++ source file.
       *******************************************************/

      temp=0;
      while(isspace(string[temp])) temp++;
      if (string[temp] == '\0') {
         G__fprinterr(G__serr,"Error: no file specified\n");
#ifndef G__OLDIMPLEMENTATION464
	 G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	 G__UnlockCriticalSection();
#endif
         return(ignore);
      }
#ifndef G__OLDIMPLEMENTATION1094
      G__store_undo_position();
#endif
      temp2 = G__prerun;
      G__prerun = 1;  /* suppress warning message if file already loaded */
      temp1 = G__loadfile(string+temp);
      if (temp1 == 1) {
         G__unloadfile(string+temp);
#ifndef G__OLDIMPLEMENTATION1102
	 G__storerewindposition();
#endif
         if (G__loadfile(string+temp)) {
            G__prerun = temp2;
            G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	    G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifndef G__OLDIMPLEMENTATION1094
	    if(G__security_error) G__cancel_undo_position();
#endif
#ifdef G__SECURITY
	    *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	    G__UnlockCriticalSection();
#endif
            return(ignore);
         }
      }
#ifdef G__OLDIMPLEMENTATION487
#ifdef G__AUTOCOMPILE
      /*************************************************************
       * if '#pragma compile' appears in source code.
       *************************************************************/
      if(G__fpautocc) G__autocc();
#endif
#endif
      G__prerun = temp2;
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
    }

    else if (strncmp("U",com,1)==0) {
      /*******************************************************
       * Load a C/C++ source file.
       *******************************************************/
      temp=0;
      while(isspace(string[temp])) temp++;
      if (string[temp] == '\0') {
         G__fprinterr(G__serr,"Error: no file specified\n");
#ifndef G__OLDIMPLEMENTATION464
	 G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	 G__UnlockCriticalSection();
#endif
         return(ignore);
      }
#ifndef G__OLDIMPLEMENTATION1094
      if(G__UNLOADFILE_SUCCESS==G__unloadfile(string+temp))
	G__init_undo();
#else
      G__unloadfile(string+temp);
#endif
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
    }
#endif

    else if(strncmp("n",com,1)==0) {
      /*******************************************************
       * Create New readline file and start dump.
       *******************************************************/
      index=1;
      while(isspace(command[index])&&command[index]!='\0') index++;
      tempfp=fopen(command+index,"w");
      if(tempfp) {
	fprintf(G__sout
		,"Create new readline dumpfile %s and start dump\n"
		,command+index);
	G__pushdumpinput(tempfp,0);
      }
      else {
	fprintf(G__sout,"Can not open file %s\n",command+1);
      }
    }

    else if(strncmp("y",com,1)==0) {
      /*******************************************************
       * Append readline input to file.
       *******************************************************/
      index=1;
      while(isspace(command[index])&&command[index]!='\0') index++;
      tempfp=fopen(command+index,"a");
      if(tempfp) {
	fprintf(G__sout
		,"Append readline dump to %s\n"
		,command+index);
	G__pushdumpinput(tempfp,0);
      }
      else {
	fprintf(G__sout,"Can not open file %s\n",command+1);
      }
    }

    else if(strncmp("z",com,1)==0) {
      /*******************************************************
       * Stop readline dump.
       *******************************************************/
      if(G__dumpreadline[0]) {
	fprintf(G__sout,"Stop readline dump.\n");
	fclose(G__dumpreadline[0]);
	G__popdumpinput();
      }
      else {
	fprintf(G__sout,"No readline dumpfile in the stack\n");
      }
    }

#ifndef G__OLDIMPLEMENTATION1546
    else if(strncmp("save",com,4)==0) {
      if(G__emergencycallback) (*G__emergencycallback)();
      else fprintf(G__sout,"!!!No emergency callback\n");
    }
#endif

#ifndef G__OLDIMPLEMENTATION970
    else if(strncmp("COPYFLAG",com,1)==0) {
      fprintf(G__sout,"set source file copy flag '%s'\n",string);
      G__setcopyflag(atoi(string));
    }
#endif

#ifndef G__OLDIMPLEMENTATION1097
    else if(strncmp("AUTOVAR",com,1)==0) {
      fprintf(G__sout,"set automatic variable allocation mode '%s'\n",string);
      G__automaticvar = atoi(string);
    }
#endif

    else if(strncmp("help",com,1)==0||strncmp("?",com,1)==0) {
      /*******************************************************
       * Help for break prompt command
       *******************************************************/
      G__more_pause((FILE*)NULL,1);

      G__display_note();

#ifndef G__SMALLOBJECT
#ifdef G__ROOT
      G__more(G__sout,"CINT/ROOT C/C++ interpreter interface.\n");
#else
      G__more(G__sout,"cint (C/C++ interpreter) debugger usage:\n");
#endif
      if(G__INPUTROOTMODE&G__rootmode) {
	G__more(G__sout,"All commands must be preceded by a . (dot), except\n");
	G__more(G__sout,"for the evaluation statement { } and the ?.\n");
	G__more(G__sout,"===========================================================================\n");
      }
#ifndef G__ROOT
      G__more(G__sout,"Dump:        n [file]  : create new readline dumpfile and start dump\n");
      G__more(G__sout,"             y [file]  : append readline dump to [file]\n");
      G__more(G__sout,"             z         : stop readline dump\n");
      G__more(G__sout,"             < [file]  : input redirection from [file](execute command dump)\n");
#endif
      G__more(G__sout,"             > [file]  : output redirection to [file]\n");
      G__more(G__sout,"             2> [file] : error redirection to [file]\n");
      G__more(G__sout,"             >& [file] : output&error redirection to [file]\n");
#ifndef G__ROOT
      G__more(G__sout,"             .         : switch command input mode\n");
#endif
      G__more(G__sout,"Help:        ?         : help\n");
      G__more(G__sout,"             help      : help\n");
      G__more(G__sout,"             /[keyword] : search keyword in help information\n");
#if (!defined(G__ROOT)) && (!defined(G__WIN32))
      G__more(G__sout,"Completion:  [nam][Tab] : complete symbol name start with [nam]\n");
      G__more(G__sout,"             [nam][Tab][Tab] : list up all symbol name start with [nam]]\n");
#endif
      G__more(G__sout,"Shell:       ![shell]  : execute shell command\n");
      G__more(G__sout,"Source:      v <[line]>: view source code <around [line]>\n");
      G__more(G__sout,"             V [stack] : view source code in function call stack\n");
      G__more(G__sout,"             t         : show function call stack\n");
      G__more(G__sout,"             f [file]  : select file to debug\n");
      G__more(G__sout,"             T         : turn on/off trace mode for all source\n");
      G__more(G__sout,"             J [stat]  : Set warning level [0-5]\n");
#ifndef G__OLDIMPLEMENTATION1097
      G__more(G__sout,"             A [1|0]   : allowing automatic variable on/off\n");
#endif
      G__more(G__sout,"             trace <classname> : turn on trace mode for class\n");
      G__more(G__sout,"             deltrace <classname> : turn off trace mode for class\n");
#ifndef G__ROOT
      G__more(G__sout,"             break [classname] : set break point at every [classname] memberfunc\n");
      G__more(G__sout,"             delbreak [classname] : turn off memberfunc break point\n");
#endif
      /* G__more(G__sout,"             D         : toggle source code display mode\n"); */
      G__more(G__sout,"Evaluation:  p [expr]  : evaluate expression (no declaration/loop/condition)\n");
      G__more(G__sout,"Evaluation:  s [expr]  : step into expression (no declaration/loop/condition)\n");
      G__more(G__sout,"Evaluation:  S [expr]  : step over expression (no declaration/loop/condition)\n");
      G__more(G__sout,"             {[statements]} : evaluate statement (any kind)\n");
#ifndef G__FONS3
      G__more(G__sout,"             x [file]  : load [file] and evaluate {statements} in the file\n");
      G__more(G__sout,"             X [file]  : load [file] and execute function [file](wo extension)\n");
#endif
      G__more(G__sout,"             E <[file]>: open editor and evaluate {statements} in the file\n");
#ifndef G__FONS4
      G__more(G__sout,"Load/Unload: L [file]  : load [file]\n");
#ifndef G__OLDIMPLEMENTATION1108
      G__more(G__sout,"             La [file] : reload all files loaded after [file]\n");
#endif
      G__more(G__sout,"             U [file]  : unload [file]\n");
#endif
#ifndef G__OLDIMPLEMENTATION970
      G__more(G__sout,"             C [1|0]   : copy source to $TMPDIR (on/off)\n");
#endif
#ifndef G__ROOT
#ifndef G__FONS2
      G__more(G__sout,"             reset     : reset interpreter environment\n");
#endif
#endif
#ifndef G__OLDIMPLEMENTATION1094
      G__more(G__sout,"             undo      : undo previous declarations\n");
#endif
      G__more(G__sout,"             lang      : local language (EUC,SJIS,EUROPEAN,UNKNOWN)\n");
      G__more(G__sout,"Monitor:     g <[var]> : list global variable\n");
      G__more(G__sout,"             l <[var]> : list local variable\n");
#ifndef G__OLDIMPLEMENTATION553
      G__more(G__sout,"             proto <[scope]::>[func] : show function prototype\n");
#endif
      G__more(G__sout,"             class <[name]> : show class definition (one level)\n");
      G__more(G__sout,"             Class <[name]> : show class definition (all level)\n");
      G__more(G__sout,"             typedef <name> : show typedefs\n");
      G__more(G__sout,"             function  : show interpreted functions\n");
      G__more(G__sout,"             macro     : show macro functions\n");
      G__more(G__sout,"             template  : show templates\n");
      G__more(G__sout,"             include   : show include paths\n");
      G__more(G__sout,"             file      : show loaded files\n");
      G__more(G__sout,"             where     : show current file position\n");
      G__more(G__sout,"             security  : show security level\n");
#ifndef G__OLDIMPLEMENTATION1033
      G__more(G__sout,"             refcount  : reference count control on/off\n");
#endif
      G__more(G__sout,"             garbage   : show garbage collection buffer\n");
      G__more(G__sout,"             Garbage   : Do garbage collection\n");
      G__more(G__sout,"             cover [file] : save trace coverage\n");
      G__more(G__sout,"             return [val] : return undefined symbol value\n");
      G__more(G__sout,"Run:         S         : step over function/loop\n");
      G__more(G__sout,"             s         : step into function/loop\n");
      G__more(G__sout,"             i         : ignore and step over\n");
      G__more(G__sout,"             c <[line]>: continue <to [line]>\n");
      G__more(G__sout,"             e         : step out from function\n");
      G__more(G__sout,"             f [file]  : select file to debug\n");
      G__more(G__sout,"             b [line]  : set break point\n");
      G__more(G__sout,"             db [line] : delete break point\n");
      G__more(G__sout,"             a [assert]: break only if assertion is true\n");
      G__more(G__sout,"             O [0~4]   : Set bytecode compiler mode\n");
      G__more(G__sout,"             debug     : bytecode status display on/off\n");
#ifdef G__ASM_DBG
      G__more(G__sout,"             asmstep   : bytecode step mode on/off\n");
      G__more(G__sout,"             status    : show bytecode exec flags\n");
#endif
      G__more(G__sout,"             dasm      : disassembler\n");
#endif
      G__more(G__sout,"Quit:        q         : quit cint\n");
      G__more(G__sout,"             qqq       : quit cint - mandatory\n");
      G__more(G__sout,"             qqqqq     : exit process immediately\n");
      G__more(G__sout,"             qqqqqqq   : abort process\n");
#ifndef G__OLDIMPLEMENTATION1546
      G__more(G__sout,"             save      : call emergency routine to save important data\n");
#endif
    }

    else if(strncmp("/",com,1)==0) {
      G__more_pause((FILE*)NULL,1);
      /*******************************************************
       * Display keyword help information
       *******************************************************/
      do {
#if defined(G__OLDIMPLEMENTATION2092_YET)
	G__temp=tmpfile();
        if(!G__temp) {
	  G__tmpnam(tname); /* not used anymore */
	  G__temp=fopen(tname,"w");
          istmpnam=1;
        }
#elif !defined(G__OLDIMPLEMENTATION1917)
	G__temp=tmpfile();
#else
	G__tmpnam(tname); /* not used anymore */
	G__temp=fopen(tname,"w");
#endif
      } while((FILE*)NULL==G__temp && G__setTMPDIR(tname));
      if(G__temp) {
	G__cintrevision(G__temp);
	G__list_sut(G__temp);
	/* compiled and interpreterd objects */
	G__display_class(G__temp,"",0,0);
	G__display_typedef(G__temp,"",0);
	G__display_string(G__temp);
	G__display_template(G__temp," ");
	G__display_macro(G__temp,"");
	G__display_files(G__temp);
	G__listfunc(G__temp,G__PUBLIC_PROTECTED_PRIVATE,(char*)NULL
		    ,(struct G__ifunc_table*)NULL);
	G__varmonitor(G__temp,&G__global,"","",0);
#if defined(G__OLDIMPLEMENTATION2092_YET)
        if(istmpnam) fclose(G__temp);
#elif !defined(G__OLDIMPLEMENTATION1917)
#else
	fclose(G__temp);
#endif
      
	if(command[strlen(command)-1]==' ') command[strlen(command)-1]='\0';
#if defined(G__OLDIMPLEMENTATION2092_YET)
        if(!istmpnam) {
	  G__display_keyword(G__sout,command+1,G__temp);
	  fclose(G__temp);
        }
        else {
	  G__display_keyword(G__sout,command+1,tname);
	  remove(tname);
        }
#elif !defined(G__OLDIMPLEMENTATION1917)
	G__display_keyword(G__sout,command+1,G__temp);
	fclose(G__temp);
#else
#ifndef G__OLDIMPLEMENTATION577
	G__display_keyword(G__sout,command+1,tname);
#else
	sprintf(syscom,"grep '%s' %s",command+1,tname);
	system(syscom);
#endif
	remove(tname);
#endif
      }
      else {
	G__fprinterr(G__serr,"Error: Tempfile G__temp can not open\n");
      }
    }

#ifndef G__OLDIMPLEMENTATION2031
    else if(strncmp("$",com,1)==0) {
      /*******************************************************
       * Execute shell command
       *******************************************************/
      char *combuf = (char*)malloc(strlen(string)+30);
      sprintf(combuf,"sh -I -c %s",string);
      system(combuf);
      free((void*)combuf);
    }
#endif

    else if(strncmp("!",com,1)==0) {
      /*******************************************************
       * Execute shell command
       *******************************************************/
      system(string);
    }


    else if(strncmp("a",com,1)==0) {
      /*******************************************************
       * Set assertion for break
       *******************************************************/
      command[0]=' ';
      if(command[1]=='\0') {
	G__assertion[0]='\0'; /* sprintf(G__assertion,""); */
	fprintf(G__sout,"Break assertion is deleted\n");
      }
      else {
	sprintf(G__assertion,"%s",command+1);
	fprintf(G__sout,"Break only if (%s) is true\n"
		,G__assertion);
      }
    }

#if 1
    else if(strncmp("G",com,1)==0) {
      G__more_pause((FILE*)NULL,1);
      index=1;
      while(isspace(command[index])) index++;
      temp=index;
      while(command[temp]&&(!isspace(command[temp]))) temp++;
      command[temp]='\0';
      G__debugvariable(G__sout,&G__global,command+index);
    }
#endif

#ifndef G__FONS23
    else if(strncmp("g",com,1)==0 || strncmp("G",com,1)==0 ||
            strncmp("l",com,1)==0) {
#else
    else if(strncmp("g",com,1)==0 || strncmp("l",com,1)==0) {
#endif
      /*******************************************************
       * Monitor global variables
       * Monitor local variables
       *******************************************************/
      G__more_pause((FILE*)NULL,1);
      index=1;
      while(isspace(command[index])) index++;
      temp=index;
      while(command[temp]&&(!isspace(command[temp]))) temp++;
      command[temp]='\0';

#ifdef G__ASM
      G__STORE_ASMENV;
#endif
      store_var_type = G__var_type;
      G__var_type = 'p';

      G__SET_TEMPENV;
      if('g'==command[0]) G__varmonitor(G__sout,&G__global,command+index,"",0);
#ifndef G__FONS23
      else if('G'==command[0]) G__varmonitor(G__sout,&G__global,command+index,"",0);
#endif
#ifndef G__OLDIMPLEMENTATION2159
      else if(G__cintv6) {
        if(G__bc_setdebugview(temp1,&view)) 
	  G__varmonitor(G__sout,view.var_local,command+index,"",view.localmem);
      }
#endif
      else G__varmonitor(G__sout,G__p_local,command+index,"",0);
      G__RESET_TEMPENV;

#ifdef G__ASM
      G__RECOVER_ASMENV;
#endif
      G__var_type = store_var_type;
    }

    else if(strncmp("t",com,1)==0) {
      /*******************************************************
       * show function call stack
       *******************************************************/
#ifndef G__OLDIMPLEMENTATION2137
      if(G__cintv6) G__bc_showstack(G__sout);
      else          G__showstack(G__sout);
#else
      G__showstack(G__sout);
#endif
    }

    else if(strncmp("T",com,1)==0) {
      /*******************************************************
       * Toggle trace mode
       *******************************************************/
      if(G__istrace==0) {
	G__istrace=1;
#ifdef G__ASM_DBG
	if(string[0]) G__istrace = atoi(string);
#endif
	fprintf(G__sout,"\nTrace mode on %d\n",G__istrace);
      }
      else {
	fprintf(G__sout,"\nTrace mode off\n");
	G__istrace=0;
      }
      G__debug = G__istrace;
      G__setdebugcond();
    }

    else if(strncmp("D",com,1)==0) {
      /*******************************************************
       * Toggle display mode
       *******************************************************/
      if(G__breakdisp==0) {
	G__breakdisp=1;
	fprintf(G__sout,"\nDisplay mode on\n");
      }
      else {
	fprintf(G__sout,"\nDisplay mode off\n");
	G__breakdisp=0;
      }
    }

    else if(strncmp("db",com,2)==0) {
      /*******************************************************
       * Delete break point
       *******************************************************/
      if(1<G__findposition(string,view.file,&line_number,&filenum)
#ifndef G__OLDIMPLEMENTATION2013
	 && filenum>=0 && line_number>=0
#endif
	 ) {
	G__srcfile[filenum].breakpoint[line_number] &= G__NOBREAK;
	fprintf(G__sout,"Break point line %d %s deleted\n"
		,line_number,G__srcfile[filenum].filename);
	G__step=0;
	G__charstep=0;
	G__setdebugcond();
      }
      else {
	G__fprinterr(G__serr,"can not determine where to delete break point\n");
      }
    }

    else if(strncmp("break",com,1)==0) {
      /*******************************************************
       * Set break point
       *******************************************************/
      if(1<G__findposition(string,view.file,&line_number,&filenum) &&
#ifndef G__OLDIMPLEMENTATION2013
	 filenum>=0 && line_number>=0 &&
#endif
	 G__srcfile[filenum].breakpoint && G__srcfile[filenum].maxline
	 ) {
	fprintf(G__sout,"Break point set to line %d %s\n"
		,line_number,G__srcfile[filenum].filename);
	G__srcfile[filenum].breakpoint[line_number] |= G__BREAK;
	G__step=0;
	G__charstep=0;
	G__setdebugcond();
      }
      else if(0==G__srcfile[filenum].maxline) {
	if((FILE*)NULL==G__srcfile[filenum].fp) {
	  G__fprinterr(G__serr,"Can not put break point in included file\n");
	}
	else {
	  G__fprinterr(G__serr,"Setting break point suspended\n");
	  temp=0;
	  while(isspace(string[temp])) ++temp;
	  if('\0'==string[temp]) {
	    sprintf(G__breakline,"%d",G__ifile.line_number);
	    strcpy(G__breakfile,G__srcfile[G__ifile.filenum].filename);
	  }
	  else {
	    strcpy(G__breakline,string+temp);
	  }
	}
      }
      else {
	G__fprinterr(G__serr,"Can not determine where to put break point\n");
      }
    }

    else if(strncmp("continue",com,1)==0) {
      /*******************************************************
       * Continue
       *******************************************************/
      G__stepover=0;
      G__step=0;
      G__charstep=0;
      G__steptrace=0;
      G__setdebugcond();
      if(strlen(string)==0) {
	G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(ignore);
      }
      else if(1<G__findposition(string,view.file,&line_number,&filenum)
#ifndef G__OLDIMPLEMENTATION2013
	      && filenum>=0 && line_number>=0 
#endif
	      ) {
	G__srcfile[filenum].breakpoint[line_number] |= G__CONTUNTIL;
	G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(ignore);
      }
      else {
	G__fprinterr(G__serr,"can not determine where to stop\n");
      }
    }

    else if(strncmp("f",com,1)==0) {
      strcpy(syscom,string);
      strcpy(command,syscom);
      /*******************************************************
       * Set view file
       *******************************************************/
      for(temp=0;temp<G__nfile;temp++) {
#ifndef G__OLDIMPLEMENTATION1196
	if(G__matchfilename(temp,string)) break;
#else
	if(strcmp(string,G__srcfile[temp].filename)==0) break;
#endif
      }
      if(temp>=G__nfile) {
	G__fprinterr(G__serr,"filename %s not loaded\n",string);
      }
      else {
	view.file.filenum=temp;
	view.file.fp=G__srcfile[temp].fp;
	strcpy(view.file.name,G__srcfile[temp].filename);
	view.file.line_number=1;
	G__pr(G__sout,view.file);
      }
    }

    else if(strncmp("+",com,1)==0 || strncmp("-",com,1)==0) {
      temp=atoi(command);
      sprintf(command,"%d",view.file.line_number + temp);
      goto vcommand;
    }

#ifndef G__OLDIMPLEMENTATION553
    else if(strncmp("proto",com,3)==0) {
      G__more_pause((FILE*)NULL,1);
      G__display_proto(G__sout,string);
    }
#endif

    else if(strncmp("view",com,1)==0) {
      strcpy(syscom,string);
      strcpy(command,syscom);
    vcommand:
      /*******************************************************
       * Display source code
       *******************************************************/
      if(0<G__findposition(command,view.file,&line_number,&filenum)&&filenum>=0
#ifndef G__OLDIMPLEMENTATION2013
	 && line_number>=0 
#endif
	 ) {
	view.file.filenum = filenum;
	view.file.fp = G__srcfile[filenum].fp;
	strcpy(view.file.name,G__srcfile[filenum].filename);
	view.file.line_number = line_number;
	G__pr(G__sout,view.file);
      }
      else {
	fprintf(G__sout,"No proper file view.file. Can not display source! Use 'f [file]' command\n");
      }
    }

    else if(strncmp("View",com,1)==0) {
      /*******************************************************
       * show source code in stack
       *******************************************************/
      temp1=atoi(command+1);
      temp=0;
      local=G__p_local;
#ifndef G__OLDIMPLEMENTATION2137
      if(G__cintv6) {
        if(G__bc_setdebugview(temp1,&view)) G__pr(G__sout,view.file);
      }
      else {
        while(local && temp<temp1-1) {
	  ++temp;
	  local=local->prev_local;
        }
        if(0==temp1) {
	  view.file = G__ifile;
	  view.var_local = G__p_local;
	  view.struct_offset=G__store_struct_offset;
	  view.tagnum=G__tagnum;
	  view.exec_memberfunc=G__exec_memberfunc;
	  G__pr(G__sout,view.file);
        }
        else if(local && local->prev_local) {
	  view.file.filenum = local->prev_filenum ;
	  strcpy(view.file.name,G__srcfile[view.file.filenum].filename);
	  view.file.fp = G__srcfile[view.file.filenum].fp;
	  view.file.line_number = local->prev_line_number;
	  view.var_local = local->prev_local;
	  view.struct_offset=view.var_local->struct_offset;
	  view.tagnum=view.var_local->tagnum;
	  view.exec_memberfunc=view.var_local->exec_memberfunc;
	  G__pr(G__sout,view.file);
        }
        else {
	  fprintf(G__sout,"Stack isn't that deep\n");
        }
      }
#else
      while(local && temp<temp1-1) {
	++temp;
	local=local->prev_local;
      }
      if(0==temp1) {
	view.file = G__ifile;
	view.var_local = G__p_local;
	view.struct_offset=G__store_struct_offset;
	view.tagnum=G__tagnum;
	view.exec_memberfunc=G__exec_memberfunc;
	G__pr(G__sout,view.file);
      }
      else if(local && local->prev_local) {
	view.file.filenum = local->prev_filenum ;
	strcpy(view.file.name,G__srcfile[view.file.filenum].filename);
	view.file.fp = G__srcfile[view.file.filenum].fp;
	view.file.line_number = local->prev_line_number;
	view.var_local = local->prev_local;
	view.struct_offset=view.var_local->struct_offset;
	view.tagnum=view.var_local->tagnum;
	view.exec_memberfunc=view.var_local->exec_memberfunc;
	G__pr(G__sout,view.file);
      }
      else {
	fprintf(G__sout,"Stack isn't that deep\n");
      }
#endif
    }


    else if(strncmp("ignore",com,1)==0) {
      /*******************************************************
       * Ignore current statement and step to the next
       *******************************************************/
      G__stepover=0;
      ignore=G__PAUSE_IGNORE;
      G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
      G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return(ignore);
    }

    else if(strncmp("S",com,1)==0 && 0==(*stringb)) {
      /*****************************************************
       * Step over
       *****************************************************/
      G__stepover=3;
      ignore=G__PAUSE_STEPOVER;
      G__step=1;
      G__charstep=0;
      G__setdebugcond();
      G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
      G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return(ignore);
    }

    else if(strncmp("s",com,1)==0 && 0==(*stringb)) {
      /*****************************************************
       * Step into
       *****************************************************/
      G__stepover=0;
      ignore=G__PAUSE_NORMAL;
      G__step=1;
      G__charstep=0;
      G__setdebugcond();
      G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
      G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return(ignore);
    }

    else if(strncmp("e",com,1)==0 && 'x'!=com[1]) {
      /*****************************************************
       * Step out 
       *****************************************************/
      G__stepover=0;
      G__step=0;
      G__charstep=0;
      G__break_exit_func=1;
      G__setdebugcond();
      G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
      G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return(ignore);
    }

    else if( strncmp(command,"qqqqqqq",7)==0 ||
	     strncmp(command,"QQQQQQQ",7)==0) {
      abort();
    }
    else if( strncmp(command,"qqqqq",5)==0 ||
	     strncmp(command,"QQQQQ",5)==0) {
      G__fprinterr(G__serr,"  Bye... (try 'qqqqqqq' if still running)\n");
      exit(EXIT_FAILURE);
    }

    else if( strncmp(command,"qqq",3)==0 ||
	     strncmp(command,"QQQ",3)==0) {
      fprintf(G__sout,"*** Process will be killed ***\n");
      strcpy(command,G__input("Are you sure(y/Y/n)? "));
      if(command[0]=='Y') {
	G__fprinterr(G__serr,"  Bye... (try 'qqqqq' if still running)\n");
	exit(EXIT_FAILURE); 
      }
      else if(command[0]=='y') {
	G__fprinterr(G__serr,"  Bye... (try 'qqqqq' if still running)\n");
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
	G__closemfp();
	G__mfp=NULL;
	G__close_inputfiles();
	exit(EXIT_SUCCESS); 
      }
      else {
	G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(0);
      }
    }

#ifndef G__OLDIMPLEMENTATION743
    else if( strncmp(command,"qq",2)==0 ||
	     strncmp(command,"QQ",2)==0 ) {
      G__return=G__RETURN_EXIT1;
#ifndef G__ROOT
      fprintf(G__sout,"Exit current interpretation.\n");
#endif
      /*****************************************
       * return 2 so that user can utilize 
       * while(G__pause()==0) ;
       *****************************************/
      G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
      G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
			  ,keyword,pipefile);
#endif
      ignore = 2;
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return(ignore);
    }
#endif
    
    else if(strncmp("q",com,1)==0 || strncmp("exit",com,4)==0) {
      if(G__doingconstruction) {
	G__fprinterr(G__serr,"Use 'qqq' when you quit in the middle of object construction (%d)\n"
		,G__doingconstruction);
	G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(ignore);
      }
      else {
	G__fprinterr(G__serr,"  Bye... (try 'qqq' if still running)\n");
      }

      G__stepover=0;
      G__step=0;
      G__break=0;
      G__steptrace=0;
      G__setdebugcond();

      /*******************************************************
       * Quit session and stop process
       *******************************************************/
      if(G__othermain) {
        /*******************************************************
         * QQ : exit all interpretation
	 * Q  : exit current interpretation
	 *******************************************************/
	if(command[1]=='q') {
	  G__return=G__RETURN_EXIT2;
	  fprintf(G__sout,"Exit current command file.\n");
	}
	else {
	  G__return=G__RETURN_EXIT1;
#ifndef G__ROOT
	  fprintf(G__sout,"Exit current interpretation.\n");
#endif
	}
	/*****************************************
	* return 2 so that user can utilize 
	* while(G__pause()==0) ;
	*****************************************/
	G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
	ignore = 2;
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(ignore);
      }
      else {
        /*******************************************************
	 * Q  : exit process
	 *******************************************************/
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__DEBUG
	 G__exit(EXIT_SUCCESS);
#else
	 G__close_inputfiles();
	 exit(EXIT_SUCCESS); 
#endif
      }
    }


#ifndef G__FONS3
    else if(strncmp("E",com,1)==0 || strncmp("x",com,1)==0) {
#else
    else if(strncmp("E",com,1)==0) {
#endif
      /*******************************************************
       * Open tempfile to edit 
       *******************************************************/
      temp=1;
      while(isspace(command[temp])) temp++;
#ifndef G__FONS3
      if (com[0] == 'E') {
#endif
	if(command[temp]=='\0') {
	  if('\0'==G__tempc[0]) G__tmpnam(G__tempc); /* E command, rare case */
	  sprintf(syscom,"%s %s",editor, G__tempc);
	  system(syscom);
	  sprintf(syscom,G__tempc);
	}
	else {
	  sprintf(syscom,"%s %s",editor, command+temp);
	  system(syscom);
	  sprintf(syscom,command+temp);
	}
#ifndef G__FONS3
      } else {
	if (command[temp]=='\0') {
	  G__fprinterr(G__serr,"Error: no file specified\n");
#ifndef G__OLDIMPLEMENTATION464
	  G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
          *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	  G__UnlockCriticalSection();
#endif
	  return(ignore);
	}
	sprintf(syscom,command+temp);
      }
#endif

      /*******************************************************
       * Execute temp file
       *******************************************************/
#ifndef G__OLDIMPLEMENTATION1094
      G__store_undo_position();
#ifdef G__ROOT
      if(strstr(syscom,"rootlogon.")) G__init_undo();
#endif
#endif
      {
        struct G__store_env store;
        G__SET_TEMPENV; 
        buf=G__exec_tempfile(syscom);
        if(rslt) *rslt = buf;
#ifndef G__OLDIMPLEMENTATION901
        if(G__ifile.filenum>=0) 
	  G__security = G__srcfile[G__ifile.filenum].security;
        else
	  G__security = G__SECURE_LEVEL0;
#else
        G__security = G__srcfile[G__ifile.filenum].security;
#endif
#ifndef G__ROOT
        G__in_pause=1;
        G__valuemonitor(buf,syscom);
        G__in_pause=0;
#ifndef G__OLDIMPLEMENTATION1259
        if(buf.isconst&(G__CONSTVAR|G__CONSTFUNC)) {
	  char tmp[G__ONELINE];
          sprintf(tmp,"(const %s",syscom+1);
          strcpy(syscom,tmp);
        }
        if(buf.isconst&G__PCONSTVAR) {
	  char tmp2[G__ONELINE];
          char *ptmp = strchr(syscom,')');
          strcpy(tmp2,ptmp);
          strcpy(ptmp,"const");
          strcat(syscom,tmp2);
        }
#endif
#ifndef G__OLDIMPLEMENTATION1079
        if(buf.type && 0==G__atevaluate(buf)) fprintf(G__sout,"%s\n",syscom);
#else
        if(buf.type) fprintf(G__sout,"%s\n",syscom);
#endif
#endif
        G__RESET_TEMPENV; 
      }
#ifndef G__OLDIMPLEMENTATION1094
      if(G__security_error) G__cancel_undo_position();
#endif
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif

      if(G__return!=G__RETURN_NON) {
	G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
        *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(ignore);
      }
      
    }

    else if(strncmp("{",com,1)==0) {
      /*******************************************************
       * Evaluate sequencial statements
       *******************************************************/
#ifndef G__OLDIMPLEMENTATION1774
    multi_line_command:
#endif
      if (*more == 0) {
#if !defined(G__OLDIMPLEMENTATION2092)
	ftemp.fp = tmpfile();
        if(!ftemp.fp) {
	  do {
	    G__tmpnam(tname); /* not used anymore */
	    ftemp.fp = fopen(tname,"w");
	  } while((FILE*)NULL==ftemp.fp && G__setTMPDIR(tname));
          istmpnam=1;
        }
#elif !defined(G__OLDIMPLEMENTATION1794)
	ftemp.fp = tmpfile();
#else /* 1794 */
	do {
	  G__tmpnam(tname); /* not used anymore */
	  ftemp.fp = fopen(tname,"w");
	} while((FILE*)NULL==ftemp.fp && G__setTMPDIR(tname));
#endif /* 1794 */
      } 
      else {
	com = command+1;
#ifndef G__OLDIMPLEMENTATION1774
	if('@'==com[0]) {
	  if(ftemp.fp) fclose(ftemp.fp);
	  ftemp.fp = (FILE*)NULL;
	  *more=0;
	  G__fprinterr(G__serr,"!!!command line input aborted!!!\n");
	  G__UnlockCriticalSection();
	  return(0);
	}
#endif
      }

      if(!ftemp.fp) {
	G__fprinterr(G__serr,"Error: could not create file %s\n",tname);
      }
      else {
	temp = *more; 
        temp1 = 0;
	double_quote=0; single_quote=0;
	while(com[temp1]!='\0') {
	  switch(com[temp1]) {
	  case '"':
	    if(single_quote==0) double_quote ^= 1;
	    break;
	  case '\'':
	    if(double_quote==0) single_quote ^= 1;
	    break;
	  case '{':
#ifndef G__OLDIMPLEMENTATION1774
	  case '(':
	  case '[':
#endif
	    if((single_quote==0)&&(double_quote==0)) temp++;
	    break;
	  case '}':
#ifndef G__OLDIMPLEMENTATION1774
	  case ')':
	  case ']':
#endif
	    if((single_quote==0)&&(double_quote==0)) temp--;
	    break;
#ifndef G__OLDIMPLEMENTATION1679
	  case '\\':
	    ++temp1;
	    break;
#endif
	  }
	  ++temp1;
	}
	if (temp>0) {
	  fprintf(ftemp.fp,"%s\n",com);
#ifndef G__OLDIMPLEMENTATION1774
	  strcpy(prompt,"end with '}', '@':abort > ");
#else
	  strcpy(prompt,"end with '}'> ");
#endif
          *more = temp;
#ifndef G__FONS23
	  G__pause_return=0;
#else
	  G__pause_return=1;
#endif
#ifdef G__SECURITY
          *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	  G__UnlockCriticalSection();
#endif
          return(0);
	} else {
	  string = strrchr(com,'}');
	  if(string) {
#ifndef G__OLDIMPLEMENTATION464
	    G__redirectoutput(string
			      ,&store_stdout,&store_stderr,&store_stdin,0
		      ,keyword,pipefile);
#endif
	    *string = '\0';
	  }
	  fprintf(ftemp.fp,"%s\n",com);
	  fprintf(ftemp.fp,"}\n");
          *more = 0;
          prompt[0] = '\0';
        }
#if !defined(G__OLDIMPLEMENTATION2092)
        if(istmpnam) fclose(ftemp.fp);
#elif !defined(G__OLDIMPLEMENTATION1794)
#else
	fclose(ftemp.fp);
#endif

	/*******************************************************
	 * Execute temp file
	 *******************************************************/
#ifndef G__OLDIMPLEMENTATION1094
	G__store_undo_position();
#endif
	G__more_pause((FILE*)NULL,1);
        { 
          struct G__store_env store;
          G__SET_TEMPENV;
#if !defined(G__OLDIMPLEMENTATION2092)
          if(!istmpnam) {
            G__command_eval=1 ;
            buf=G__exec_tempfile_fp(ftemp.fp);
	    if(G__security_error&&G__pautoloading&&(*G__pautoloading)(com)) {
	      buf=G__exec_tempfile_fp(ftemp.fp);
	    }
            if(rslt) *rslt = buf;
	    if(ftemp.fp) fclose(ftemp.fp);
	    ftemp.fp = (FILE*)NULL;
          }
          else {
            strcpy(sname,tname);
            G__command_eval=1 ;
            buf=G__exec_tempfile(sname);
	    if(G__security_error&&G__pautoloading&&(*G__pautoloading)(com)) {
	      buf=G__exec_tempfile(sname);
	    }
            if(rslt) *rslt = buf;
            remove(sname);
          }
#elif !defined(G__OLDIMPLEMENTATION1794)
          G__command_eval=1 ;
          buf=G__exec_tempfile_fp(ftemp.fp);
#ifndef G__OLDIMPLEMENTATION1968
	  if(G__security_error && G__pautoloading && (*G__pautoloading)(com)) {
	    buf=G__exec_tempfile_fp(ftemp.fp);
	  }
#endif
          if(rslt) *rslt = buf;
	  if(ftemp.fp) fclose(ftemp.fp);
	  ftemp.fp = (FILE*)NULL;
#else /* 1794 */
          strcpy(sname,tname);
#ifndef G__OLDIMPLEMENTATION1476
          G__command_eval=1 ;
#endif
          buf=G__exec_tempfile(sname);
#ifndef G__OLDIMPLEMENTATION1968
#ifndef G__OLDIMPLEMENTATION2006
	  if(G__security_error && G__pautoloading && (*G__pautoloading)(com)) {
#else
	  if(G__security_error && G__autoloading(com)) {
#endif
	    buf=G__exec_tempfile(sname);
	  }
#endif
          if(rslt) *rslt = buf;
          remove(sname);
#endif /* 1794 */
          G__in_pause=1;
          G__valuemonitor(buf,syscom);
          G__in_pause=0;
#ifndef G__OLDIMPLEMENTATION1259
          if(buf.isconst&(G__CONSTVAR|G__CONSTFUNC)) {
            char tmp[G__ONELINE];
            sprintf(tmp,"(const %s",syscom+1);
            strcpy(syscom,tmp);
          }
          if(buf.isconst&G__PCONSTVAR) {
            char tmp2[G__ONELINE];
            char *ptmp = strchr(syscom,')');
            strcpy(tmp2,ptmp);
            strcpy(ptmp,"const");
            strcat(syscom,tmp2);
          }
#endif
          G__RESET_TEMPENV;
        }
#ifndef G__OLDIMPLEMENTATION1004
	if(-1==G__func_now) G__p_local=0;
#endif
	if(buf.type && 0==G__atevaluate(buf)
#ifndef G__OLDIMPLEMENTATION1474
	   && !noprintflag
#endif
	   ) fprintf(G__sout,"%s\n",syscom);
#ifndef G__OLDIMPLEMENTATION1474
	noprintflag = 0;
#endif
#ifndef G__OLDIMPLEMENTATION1476
	G__command_eval=0 ;
	G__free_tempobject();
#endif
#ifndef G__OLDIMPLEMENTATION1004
	if(-1==G__func_now) G__p_local=store.var_local;
#endif
#ifndef G__OLDIMPLEMENTATION1094
	if(G__security_error) G__cancel_undo_position();
#endif
#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif

	if(G__return!=G__RETURN_NON) {
	  G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	  G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
          *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	  G__UnlockCriticalSection();
#endif
	  return(ignore);
	}
	
      }
    }

#ifndef G__OLDIMPLEMENTATION696
    else if(strncmp("p",com,1)==0 || 
	    strncmp("s",com,1)==0 || strncmp("S",com,1)==0) {
      G__STORE_EVALENV;
#else
    else if(strncmp("p",com,1)==0) {
#endif
      /*******************************************************
       * Evaluate statement
       *******************************************************/
      G__more_pause((FILE*)NULL,1);
      strcpy(syscom,string);
      strcpy(command,syscom);
      if(strlen(command)>0) {

#ifndef G__OLDIMPLEMENTATION464
	G__redirectoutput(command,&store_stdout,&store_stderr,&store_stdin,1
		      ,keyword,pipefile);
#endif
	
#ifdef G__ASM
	G__STORE_ASMENV;
#endif
#ifndef G__OLDIMPLEMENTATION1965
	G__command_eval=1 ;
#endif
	store_var_type = G__var_type;
	G__var_type='p';
	
	single_quote=0;
	double_quote=0;
	evalbase=command;
	temp=1;
	while(temp) {
	  evalbase++;
	  switch(*evalbase) {
	  case '"':
	    if(single_quote==0) {
	      double_quote ^= 1;
	    }
	    break;
	  case '\'':
	    if(double_quote==0) {
	      single_quote ^= 1;
	    }
	    break;
	  case '\\':
	    if(double_quote==0&&single_quote==0) {
	      temp=0;
	    }
	    else {
	      evalbase++;
	    }
	    break;
	  case '\0':
	    if(double_quote==0&&single_quote==0) {
	      temp=0;
	      evalbase=(char *)NULL;
	    }
	    break;
	  }
	}
	G__SET_TEMPENV;
	if(evalbase) {
	  *evalbase='\0';
	  temp1=1;
	  num=0;
	  if(isdigit(evalbase[temp1])) {
	    while(isdigit(evalbase[temp1])) {
	      syscom[temp1-1]=evalbase[temp1];
	      ++temp1;
	    }
	    syscom[temp1-1]='\0';
	    num=atoi(syscom);
	  }
	  switch(tolower(evalbase[temp1])) {
#ifndef G__OLDIMPLEMENTATION1979
          case 'v':
	    base = 100;
	    break;
#endif
	  case 'x':
	  case 'h':
	    base=16;
	    break;
	  case 'b':
	  case 'z':
	    base=2;
	    break;
	  case 'o':
	    base=8;
	    break;
	  case 'd':
	    base=10;
	    break;
	  case 'n':
	    base=0;
	    break;
	  }
	  if(num<=0) {
	    buf = G__calc_internal(command);
	    if(rslt) *rslt = buf;
	    G__in_pause=1;
	    if(base==0) {
	      G__valuemonitor(buf,syscom);
	    }
#ifndef G__OLDIMPLEMENTATION1979
	    else if(base==100) {
	      sprintf(syscom,"{d=%g i=%ld,reftype=%d} type=%c,tag=%d,type=%d,ref=%lx,isconst=%d"
		      ,buf.obj.d,buf.obj.i,buf.obj.reftype.reftype
		      ,buf.type,buf.tagnum,buf.typenum,buf.ref,buf.isconst);
	    }
#endif
	    else {
	      G__getbase(buf.obj.i ,base ,0,evalresult);
	      sprintf(syscom,"0%c%s" ,evalbase[temp1] ,evalresult);
	    }
	    G__in_pause=0;
	    fprintf(G__sout,"%s\n",syscom);
	  }
	  else {
	    for(temp=0;temp<num;temp++) {
	      if(temp%2==0) {
		sprintf(syscom,"&%s+%d" ,command+1 ,temp);
		buf=G__calc_internal(syscom);
		if(rslt) *rslt = buf;
		fprintf(G__sout,"\n0x%lx: " ,buf.obj.i);
	      }
	      sprintf(syscom,"*(&%s+%d)" ,command+1 ,temp);
	      buf = G__calc_internal(syscom);
	      if(rslt) *rslt = buf;
	      G__in_pause=1;
	      if(base==0) {
		G__valuemonitor(buf,syscom);
	      }
	      else {
		G__getbase(buf.obj.i,base ,0,evalresult);
		sprintf(syscom,"0%c%s" ,evalbase[temp1] ,evalresult);
	      }
	      G__in_pause=0;
	      fprintf(G__sout,"%30s ",syscom);
	    }
	    fprintf(G__sout,"\n");
	  }
	}
	else {
	  buf = G__calc_internal(command);
#ifndef G__OLDIMPLEMENTATION1968
#ifndef G__OLDIMPLEMENTATION2006
	  if(G__security_error && G__pautoloading && (*G__pautoloading)(com)) {
#else
	  if(G__security_error && G__autoloading(command)) {
#endif
	    buf = G__calc_internal(command);
	  }
#endif
	  if(rslt) *rslt = buf;
#ifndef G__OLDIMPLEMENTATION696
	  if((char*)NULL==strstr(command,"G__stepmode(")) G__step=store_step;
#endif
	  G__in_pause=1;
	  G__valuemonitor(buf,syscom);
	  G__in_pause=0;
#ifndef G__OLDIMPLEMENTATION1259
	  if(buf.isconst&(G__CONSTVAR|G__CONSTFUNC)) {
	    char tmp[G__ONELINE];
	    sprintf(tmp,"(const %s",syscom+1);
	    strcpy(syscom,tmp);
	  }
	  if(buf.isconst&G__PCONSTVAR) {
	    char tmp2[G__ONELINE];
	    char *ptmp = strchr(syscom,')');
	    strcpy(tmp2,ptmp);
	   strcpy(ptmp,"const");
	   strcat(syscom,tmp2);
	  }
#endif
#ifdef G__OLDIMPLEMENTATION850
	  if(buf.type) fprintf(G__sout,"%s\n",syscom);
	  if('u'==buf.type && buf.obj.i) {
	    G__objectmonitor(G__sout,buf.obj.i,buf.tagnum,"");
	  }
#endif
	}
	G__RESET_TEMPENV;
	
#ifdef G__ASM
	G__RECOVER_ASMENV;
#endif
	G__var_type = store_var_type;
	
#ifdef G__OLDIMPLEMENTATION1965
	/*****************************************************
	 * free temp object buffer
	 *****************************************************/
	G__free_tempobject();
#endif

#ifndef G__OLDIMPLEMENTATION850
	if(buf.type && 0==G__atevaluate(buf)) {
	  fprintf(G__sout,"%s\n",syscom);
	  if('u'==buf.type && buf.obj.i) {
	    G__objectmonitor(G__sout,buf.obj.i,buf.tagnum,"");
	  }
	}
#endif
#ifndef G__OLDIMPLEMENTATION1965
	G__command_eval=0 ;
	G__free_tempobject();
#endif


#ifdef G__SECURITY
	*err |= G__security_recover(G__serr);
#endif
	
	if(G__return!=G__RETURN_NON) {
	  G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	  G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifndef G__OLDIMPLEMENTATION696
	  G__RESTORE_EVALENV;
#endif
#ifdef G__SECURITY
          *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	  G__UnlockCriticalSection();
#endif
	  return(ignore);
	}
      }
      else {
	ignore=G__stepover;
	G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
	G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifndef G__OLDIMPLEMENTATION696
	G__RESTORE_EVALENV;
#endif
#ifdef G__SECURITY
        *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
	G__UnlockCriticalSection();
#endif
	return(ignore);
      }
#ifndef G__OLDIMPLEMENTATION696
      G__RESTORE_EVALENV; 
#endif
    }

    else if(strcmp(com,"P")==0) { /* Dummy pause when reading from dump file */
      ignore=G__PAUSE_NORMAL;
      G__pause_return=0;
#ifndef G__OLDIMPLEMENTATION464
      G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return(ignore);
    }

    else if(strcmp(com,"")==0) {
      ignore=G__stepover;
      G__pause_return=1;
#ifndef G__OLDIMPLEMENTATION464
      G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return(ignore);
    }

    else {
      G__fprinterr(G__serr,"Unknown interpreter command '%s'\n",com);
    }

#ifndef G__OLDIMPLEMENTATION464
    G__unredirectoutput(&store_stdout,&store_stderr,&store_stdin
		      ,keyword,pipefile);
#endif
#ifdef G__SECURITY
    *err |= G__security_recover(G__serr);
#endif
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
  return(ignore); /* never happens, avoiding lint error */
}



/**************************************************************************
* G__setaccess
*
*
**************************************************************************/
int G__setaccess(statement,iout)
char *statement;
int iout;
{
  if(7==iout && strcmp(statement,"public:")==0) {
    G__access=G__PUBLIC;
  }
  else if(10==iout && strcmp(statement,"protected:")==0) {
    G__access=G__PROTECTED;
  }
  else if(8==iout && strcmp(statement,"private:")==0) {
    G__access=G__PRIVATE;
  }
  return(0);
}

#ifndef G__OLDIMPLEMENTATION1273
/**************************************************************************
* G__set_smartunload()
*
**************************************************************************/
void G__set_smartunload(smartunload)
int smartunload;
{
  G__do_smart_unload = smartunload;
}
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
