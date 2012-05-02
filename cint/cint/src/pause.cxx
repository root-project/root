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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Api.h"
#include "DataMemberHandle.h"
#include "floatutilities.h"

FILE* G__sout_keeper;

extern "C" {

/* 1723 is not needed because freopen deals with both stdout and cout */
#define G__OLDIMPLEMENTATION1723

extern void G__redirectcout G__P((const char* filename)) ;
void G__unredirectcout() ;
extern void G__redirectcerr G__P((const char* filename)) ;
extern void G__unredirectcerr() ;
extern void G__redirectcin G__P((const char* filename)) ;
extern void G__unredirectcin() ;
extern int G__EnableAutoDictionary;

#if defined(G__WIN32)
#include <windows.h>
#elif defined(G__POSIX)
#include <unistd.h> /* already included in G__ci.h */
#endif

extern void G__setcopyflag G__P((int flag));

#define G__NUM_STDIN   0
#define G__NUM_STDOUT  1
#define G__NUM_STDERR  2
#define G__NUM_STDBOTH 3


static void G__unredirectoutput(FILE **sout,FILE **serr,FILE **sin,const char *keyword,const char *pipefile);

#ifdef G__BORLANDCC5
void G__decrement_undo_index(int *pi);
void G__increment_undo_index(int *pi);
int G__is_valid_dictpos(struct G__dictposition *dict);
void G__show_undo_position(int index);
void G__init_undo(void);
int G__clearfilebusy(int ifn);
void G__storerewindposition(void);
static void G__display_keyword(FILE *fout,const char *keyword,FILE *keyfile);
void G__rewinddictionary(void);
void G__UnlockCriticalSection(void);
void G__LockCriticalSection(void); 
int G__IsBadCommand(char *com, const unsigned int comlen);
static void G__redirectoutput(char *com,FILE **psout,FILE **pserr,FILE **psin,int asemicolumn,char *keyword,char *pipefile);
void G__cancel_undo_position(void);
static int G__atevaluate(G__value buf);
#endif

static G__input_file G__multi_line_temp;

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


#ifdef G__ASM_DBG
extern int G__asm_step;
#endif

static int G__pause_return = 0;

#ifdef G__ROOT
static int G__rootmode = G__INPUTROOTMODE;
#else
static int G__rootmode = G__INPUTCINTMODE;
#endif

static int G__lockinputmode = 0;

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
static int lockcount = 0;
static int(*G__IsSameThread)();
static void(*G__StoreLockThread)();
static void(*G__EnterCriticalSection)();
static void(*G__LeaveCriticalSection)();

/************************************************************************
* G__SetCriticalSectionEnv
************************************************************************/
void G__SetCriticalSectionEnv(int(*issamethread)(), void(*storelockthread)(), void(*entercs)(), void(*leavecs)())
{
   G__IsSameThread = issamethread;
   G__StoreLockThread = storelockthread;
   G__EnterCriticalSection = entercs;
   G__LeaveCriticalSection = leavecs;
}

/************************************************************************
* G__LockCriticalSection
************************************************************************/
void G__LockCriticalSection()
{
   if (!G__IsSameThread || !G__EnterCriticalSection) return;
#ifndef __CINT__
   if (lockcount && (*G__IsSameThread)()) {
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
void G__UnlockCriticalSection()
{
   if (!G__IsSameThread || !G__LeaveCriticalSection) return;
#ifndef __CINT__
   if (lockcount && (*G__IsSameThread)()) {
      /* recursive call in same thread, should not unlock */
      --lockcount;
      return;
   }
#endif
   --lockcount;
   (*G__LeaveCriticalSection)();
   return;
}

/************************************************************************
* G__autoloading()  com="{  TUnknown x;" ,  "new TUnknown
************************************************************************/
int G__autoloading(char *com)
{
   int i = 0, j = 0;
   G__FastAllocString classname(G__ONELINE);
   while (com[i] && !isalpha(com[i])) ++i;
   while (com[i] && (isalnum(com[i]) || '_' == com[i]))
      classname.Set(j++, com[i++]);
   classname.Set(j, 0);
   if (0 == strcmp(classname, "new")) {
      j = 0;
      while (com[i] && !isalpha(com[i])) ++i;
      while (com[i] && (isalnum(com[i]) || '_' == com[i]))
         classname.Set(j++, com[i++]);
      classname.Set(j, 0);
   }
   if (classname[0] && -1 == G__defined_tagname(classname, 2)) {
      const char *dllpost = G__getmakeinfo1("DLLPOST");
      G__FastAllocString fname(classname);
      fname += dllpost;
      FILE *fp = fopen(fname, "r");
      if (!fp) return 0;
      else fclose(fp);
      G__FastAllocString prompt(G__ONELINE);
      prompt.Format("Will you try loading %s(y/n)? ", fname());
      prompt = G__input(prompt);
      if (tolower(prompt[0]) != 'y') return(0);
      G__security_recover(G__sout); /* QUESTIONABLE */
      switch (G__loadfile(fname)) {
         case G__LOADFILE_SUCCESS:
            fprintf(G__sout, "%s loaded\n", fname());
            return(1);
         default:
            fprintf(G__sout, "Error: failed to load %s\n", fname());
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
void G__set_autoloading(int(*p2f) G__P((char*)))
{
   G__pautoloading = p2f;
}

/************************************************************************
* G__is_valid_dictpos
************************************************************************/
int G__is_valid_dictpos(G__dictposition *dict)
{
   int flag = 0;
   struct G__var_array *var = &G__global;
   struct G__ifunc_table_internal *ifunc = &G__ifunc;

   while (var) {
      if (var == dict->var && var->allvar >= dict->var->allvar) {
         flag = 1;
         break;
      }
      var = var->next;
   }
   if (0 == flag) return(0);

   flag = 0;
   while (ifunc) {
      G__ifunc_table_internal* dict_ifunc = G__get_ifunc_internal(dict->ifunc);
      if (ifunc == dict_ifunc && ifunc->allifunc >= dict_ifunc->allifunc) {
         flag = 1;
         break;
      }
      ifunc = ifunc->next;
   }
   if (0 == flag) return(0);

   if (G__struct.alltag < dict->tagnum) return(0);
   if (G__newtype.alltype < dict->typenum) return(0);
#ifdef G__SHAREDLIB  /* ???, reported by A.Yu.Isupov, anyway no harm */
   if (G__allsl < dict->allsl) return(0);
#endif
   if (G__nfile < dict->nfile) return(0);

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
   undoindex = 0;
   for (i = 0;i < G__MAXUNDO;i++) {
      undodictpos[i].var = (struct G__var_array*)NULL;
   }
}

/******************************************************************
* G__increment_undo_index()
******************************************************************/
void G__increment_undo_index(int *pi)
{
   if (++(*pi) >= G__MAXUNDO) *pi = 0;
}

/******************************************************************
* G__decrement_undo_index()
******************************************************************/
void G__decrement_undo_index(int *pi)
{
   if (--(*pi) < 0) *(pi) = G__MAXUNDO - 1;
}

/******************************************************************
* G__cancel_undo_position()
******************************************************************/
void G__cancel_undo_position()
{
   G__decrement_undo_index(&undoindex);
   undodictpos[undoindex].var = (struct G__var_array*)NULL;
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
void G__show_undo_position(int index)
{
   int nfile = undodictpos[index].nfile;
   int tagnum = undodictpos[index].tagnum;
   int typenum = undodictpos[index].typenum;
   struct G__ifunc_table_internal *ifunc = G__get_ifunc_internal(undodictpos[index].ifunc);
   int ifn = undodictpos[index].ifn;
   struct G__var_array *var = undodictpos[index].var;
   int ig15 = undodictpos[index].ig15;

   fprintf(G__sout, "!!! Following objects will be removed by undo !!!\n");

   fprintf(G__sout, "Src File : ");
   while (nfile < G__nfile)
      fprintf(G__sout, "%s ", G__srcfile[nfile++].filename);
   fprintf(G__sout, "\n");

   fprintf(G__sout, "Class    : ");
   while (tagnum < G__struct.alltag)
      fprintf(G__sout, "%s ", G__fulltagname(tagnum++, 1));
   fprintf(G__sout, "\n");

   fprintf(G__sout, "Typedef  : ");
   while (typenum < G__newtype.alltype)
      fprintf(G__sout, "%s ", G__newtype.name[typenum++]);
   fprintf(G__sout, "\n");

   fprintf(G__sout, "Function : ");
   while (ifunc) {
      while (ifn < ifunc->allifunc) {
         fprintf(G__sout, "%s ", ifunc->funcname[ifn++]);
      }
      ifn = 0;
      ifunc = ifunc->next;
   }
   fprintf(G__sout, "\n");

   fprintf(G__sout, "Variable : ");
   while (var) {
      while (ig15 < var->allvar) {
         fprintf(G__sout, "%s ", var->varnamebuf[ig15++]);
      }
      ig15 = 0;
      var = var->next;
   }
   fprintf(G__sout, "\n");
}

/******************************************************************
* G__rewind_undo_position()
******************************************************************/
void G__rewind_undo_position()
{
   G__decrement_undo_index(&undoindex);
   if (undodictpos[undoindex].var &&
         G__is_valid_dictpos(&undodictpos[undoindex])) {
      G__FastAllocString buf(G__ONELINE);
      G__show_undo_position(undoindex);
      buf = G__input("Are you sure? (y/n) ");
      if ('y' == tolower(buf[0])) {
         G__scratch_upto(&undodictpos[undoindex]);
         undodictpos[undoindex].var = (struct G__var_array*)NULL;
         G__fprinterr(G__serr, "!!! Dictionary position rewound !!!\n");
      }
      else {
         G__increment_undo_index(&undoindex);
      }
   }
   else {
      G__fprinterr(G__serr, "!!! No undo rewinding buffer !!!\n");
      G__init_undo();
   }
}

#define G__OLDIMPLEMENTATION1156

/************************************************************************
* error recovery
************************************************************************/
static struct G__dictposition errordictpos;
static struct G__input_file errorifile;

/******************************************************************
* G__clear_errordictpos()
******************************************************************/
void G__clear_errordictpos()
{
   // Used to free the data member ptype from the global variable errordictpos.
}

/******************************************************************
* G__clearfilebusy()
******************************************************************/
int G__clearfilebusy(int ifn)
{
   struct G__ifunc_table_internal *ifunc;
   int flag = 0;
   int i1;
   int i2;
   /*********************************************************************
   * check global function busy status
   *********************************************************************/
   ifunc = &G__ifunc;
   while (ifunc) {
      if (ifunc->allifunc > G__MAXIFUNC) {
         G__fprinterr(G__serr, "Internal error: G__clearfilebusy() FATAL! Save data and terminate session");
         G__printlinenum();
         return(0);
      }
      for (i1 = 0;i1 < ifunc->allifunc;i1++) {
         if (0 != ifunc->busy[i1] && ifunc->pentry[i1]->filenum >= ifn) {
            ifunc->busy[i1] = 0;
            G__fprinterr(G__serr, "Function %s() busy flag cleared\n"
                         , ifunc->funcname[i1]);
            flag++;
         }
      }
      ifunc = ifunc->next;
   }
   /*********************************************************************
   * check member function busy status
   *********************************************************************/
   if (0 == G__nfile || ifn < 0 || G__nfile <= ifn ||
         (struct G__dictposition*)NULL == G__srcfile[ifn].dictpos ||
         -1 == G__srcfile[ifn].dictpos->tagnum) return(flag);
   for (i2 = G__srcfile[ifn].dictpos->tagnum;i2 < G__struct.alltag;i2++) {
      ifunc = G__struct.memfunc[i2];
      while (ifunc) {
         for (i1 = 0;i1 < ifunc->allifunc;i1++) {
            if (0 != ifunc->busy[i1] && ifunc->pentry[i1]->filenum >= ifn) {
               ifunc->busy[i1] = 0;
               G__fprinterr(G__serr, "Function %s() busy flag cleared\n"
                            , ifunc->funcname[i1]);
               flag++;
            }
         }
         ifunc = ifunc->next;
      }
   }
   return(flag);
}

/************************************************************************
* G__storerewindposition
************************************************************************/
void G__storerewindposition()
{
   G__store_dictposition(&errordictpos);
   errorifile = G__ifile;
}

/************************************************************************
* G__rewinddictionary
************************************************************************/
void G__rewinddictionary()
{
  if(errordictpos.var) {
    if(G__is_valid_dictpos(&errordictpos)) {
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
  /* If the file info saved was related to a temporary file
   * there is no use to reput it */
  /* We use '<' here, because if the file with an error was a temporary
     file, it probably has been closed by now and the 'fence' has been
     moved :( */
  if(errorifile.filenum<G__gettempfilenum()) G__ifile = errorifile;
  errordictpos.var = (struct G__var_array*)NULL;
}

   
static void G__define_limit_var(G__DataMemberHandle &member, const char *name, double up, double down)
{
   char store_var_type = G__var_type;
   G__var_type = 'd';
   G__value value = G__null;
   value.type = 'd';
   value.obj.d = up/down;
   G__FastAllocString tempname(name);
   G__letvariable(tempname, value, &G__global, G__p_local, member);
   G__var_type = store_var_type;   
      
}

#if defined(WIN32) || defined(__SUNPRO_CC)
using namespace Cint::FloatUtilities::DirectCompare;
#endif

/************************************************************************
* G__atevaluate
************************************************************************/
static int G__atevaluate(G__value buf)
{
   G__value result;
   G__FastAllocString com(G__ONELINE);
   G__FastAllocString buf2(G__ONELINE);
   int known = 0;
   int store_break = G__break;
   int store_step = G__step;
   int store_dispsource = G__dispsource;
   int store_asm_exec = G__asm_exec;
   int store_debug = G__debug;
   int store_mask_error = G__mask_error;

   if (G__return > G__RETURN_NORMAL && G__security_error) return(0);

   G__asm_exec = 0;

#ifndef G__OLDIMPLEMENTATION2191
   if ('1' == buf.type || 'a' == buf.type
         || 'n' == buf.type || 'm' == buf.type || 'q' == buf.type) return(0);
#else
   if ('Q' == buf.type || 'a' == buf.type) return(0);
#endif
   G__valuemonitor(buf, buf2);
   bool initedvar = false;
   G__DataMemberHandle member;
   if (buf.type == 'f' || buf.type == 'd') {
      if (isinf(buf.obj.d)) {
         G__define_limit_var(member,"inf",1.0,0.0);
         initedvar = true;
#ifdef G__WIN32
         char c;
         for(c = 0; c<strlen(buf2); ++c) {
            if (buf2[c]=='1') break;
         }
         if (c!=strlen(buf2)) {
            buf2.Replace(c,"inf");
         }
#endif
      } else if (isnan(buf.obj.d)) {
         G__define_limit_var(member,"nan",0.0,0.0);
         initedvar = true;
#ifdef G__WIN32
         char c;
         for(c = 0; c<strlen(buf2); ++c) {
            if (buf2[c]=='1') break;
         }
         if (c!=strlen(buf2)) {
            buf2.Replace(c,"nan");
         }
#endif
      }                 
   }
   com.Format("G__ateval(%s)", buf2());
   G__break = 0;
   G__step = 0;
   G__dispsource = 0;
   G__debug = 0;
   G__setdebugcond();
   G__mask_error = 1;
   result = G__getfunction(com, &known, G__TRYNORMAL);
   G__mask_error = store_mask_error;
   G__break = store_break;
   G__step = store_step;
   G__dispsource = store_dispsource;
   G__debug = store_debug;
   G__asm_exec = store_asm_exec;
   if (initedvar) {
      member.DeleteVariable();
   }
   if (known) return((int)G__int(result));
   return(0);
}

/************************************************************************
* G__display_tempobj()
************************************************************************/
static void G__display_tempobj(FILE *fout)
{
   struct G__tempobject_list *p = G__p_tempbuf;
   G__FastAllocString buf(G__ONELINE);
   fprintf(fout, "current tempobj stack level = %d\n", G__templevel);
   do {
      G__valuemonitor(p->obj, buf);
      fprintf(fout, "level%-3d %2d %s\n", p->level, p->cpplink, buf());
      p = p->prev;
   }
   while (p);
}

/************************************************************************
* G__display_keyword()
*
*  display keyword for '/[keyword]' debugger command
*
************************************************************************/
static void G__display_keyword(FILE* fout, const char* keyword, FILE* keyfile)
{
   G__FastAllocString line(G__LONGLINE);
   char *null_fgets;
   if (keyfile) {
      fseek(keyfile, 0L, SEEK_SET);
      null_fgets = fgets(line, G__LONGLINE - 1, keyfile);
      while ((char*)NULL != null_fgets) {
         if (strstr(line, keyword)) {
            if (G__more(fout, line)) break;
         }
         null_fgets = fgets(line, G__LONGLINE - 1, keyfile);
      }
   }
   else {
      G__genericerror("Warning: can't open file. keyword search unsuccessful");
   }
}

/************************************************************************
* G__reloadfile
************************************************************************/
int G__reloadfile(char *filename, bool keep)
{
   int i, j = 0;
   char* storefname[G__MAXFILE];
   int storecpp[G__MAXFILE];
   int storen = 0;
   int flag = 0;
   int store_cpp = G__cpp;
   int store_prerun = G__prerun;
   if (!filename || 0 == filename[0]) {
      G__fprinterr(G__serr, "Error: no file specified\n");
      return(-1);
   }

   storefname[0] = 0;
   for (i = 0;i < G__nfile;i++) {
      if (!flag &&
            G__matchfilename(i, filename)
         ) {
         if (!keep && G__srcfile[i].hasonlyfunc && G__do_smart_unload) {
            G__smart_unload(i);
            flag = 0;
         }
         else flag = 1;
         if (!keep) {
            j = i;
            while (-1 != G__srcfile[j].included_from
                   /* do not take the tempfile in consideration! */
                   && (G__srcfile[j].included_from <= G__gettempfilenum())
                   ) {
               if (G__srcfile[G__srcfile[j].included_from].filename == 0) {
                  /* It is possibly a closed temporary file let's ignore
                     it to */
                  break;
               }
               // sanity check; this can only happen if something went wrong during loadfile
               if ( j != G__srcfile[j].included_from)
                  j = G__srcfile[j].included_from;
               else break;
            }
         }
         break;
      }
   }

   if (flag) {
      if (keep) return 1;
      for (i = j;i < G__nfile;i++) {
         if (G__srcfile[i].filename[0]) {
            if (G__srcfile[i].prepname) storecpp[storen] = 1;
            else                       storecpp[storen] = 0;
            storefname[storen] = (char*)malloc(strlen(G__srcfile[i].filename) + 1);
            strcpy(storefname[storen], G__srcfile[i].filename); // Okay, we allocated enough space
            ++storen;
         }
      }
   }
   else {
      return(G__loadfile(filename));
   }

   if (!storefname[0] || G__UNLOADFILE_SUCCESS != G__unloadfile(storefname[0])) {
      return(1);
   }

   G__storerewindposition();
   G__init_undo();

   for (i = 0;i < storen;i++) {
      G__prerun = 1;
      G__cpp = storecpp[i];
      fprintf(G__sout, "reloading %s  %d\n", storefname[i], storecpp[i]);
      G__loadfile(storefname[i]);
      G__return = 0;
      G__security_error = 0;
   }

   G__prerun = store_prerun;
   G__cpp = store_cpp;

   return(0);
}

/************************************************************************
* G__display_classkeyword()
************************************************************************/
void G__display_classkeyword(FILE *fout, const char *classnamein, const char *keyword, int base)
{
   // --
   G__FastAllocString classname(classnamein);
   int istmpnam = 0;

   G__more_pause((FILE*)NULL, 0);
   if (keyword && keyword[0]) {
#ifndef G__TMPFILE
      G__FastAllocString tname(L_tmpnam+10);
#else
      G__FastAllocString tname(G__MAXFILENAME);
#endif
      FILE *G__temp;
      do {
         // coverity[secure_temp]: we don't care about predictable names.
         G__temp = tmpfile();
         if (!G__temp) {
            G__tmpnam(tname); /* not used anymore */
            G__temp = fopen(tname, "w");
            istmpnam = 1;
         }
      }
      while ((FILE*)NULL == G__temp && G__setTMPDIR(tname));
      if (G__temp) {
         G__display_class(G__temp, classname, base, 0);
         if (!istmpnam) {
            fseek(G__temp, 0L, SEEK_SET);
            G__display_keyword(fout, keyword, G__temp);
            fclose(G__temp);
         }
         else {
            G__display_keyword(fout, keyword, G__temp);
            fclose(G__temp);
            remove(tname);
         }
      }
   }
   else {
      G__display_class(fout, classname, base, 0);
   }
   // --
}

#ifdef G__SECURITY
/******************************************************************
* G__security_recover()
******************************************************************/
int G__security_recover(FILE *fout)
{
  int err = G__security_error ;
  if(G__security_error) {
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
      G__rewinddictionary();
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
      G__security_error = G__NOERROR; 
      /* G__security_error = G__NOERROR; */ /* don't remember why not doing */
      G__prerun = 0;
      G__decl = 0;
    }
  }
  errordictpos.var = (struct G__var_array*)NULL;

  return(err);
}
#endif

/******************************************************************
* Inevitable static variable
******************************************************************/
static struct G__store_env store;
static struct G__view view;
static int init_process_cmd_called=0;
#if defined(G__REDIRECTIO) && !defined(G__WIN32)
static char stdoutsav[64];
static char stderrsav[64];
static char stdinsav[64];
#endif

/******************************************************************
* int G__init_process_cmd()
******************************************************************/
int G__init_process_cmd()
{
   G__LockCriticalSection();
   init_process_cmd_called = 1;
   view.file = G__ifile;
   view.var_local = G__p_local;
   view.struct_offset = G__store_struct_offset;
   view.tagnum = G__tagnum;
   view.exec_memberfunc = G__exec_memberfunc;
   G__UnlockCriticalSection();
   return 0;
}

/******************************************************************
* int G__pause()
******************************************************************/
int G__pause()
{
   int ignore = G__PAUSE_NORMAL;
   int more = 0;
   void(*oldhandler)(int) = 0;
   char* p = 0;
   G__FastAllocString cintname(G__ONELINE);
   G__FastAllocString filename(G__ONELINE);
   G__FastAllocString command(G__LONGLINE);
   G__FastAllocString prompt(G__ONELINE);
   p = strrchr(G__nam, G__psep[0]);
   if (p && *(p + 1)) {
      cintname = p + 1;
   }
   else {
      cintname = G__nam;
   }
   prompt[0] = '\0';
   // Do not pause while skipping code.
   if (G__no_exec) {
      // -- We are skipping code, return immediately.
      return ignore;
   }
   //
   //  Do not pause if breakfile is specified but
   //  does not match to current source file.
   //
   //  Do not pause if assertion is set and the
   //  condition is false.
   //
   if (!G__step && strcmp(G__assertion, "") && !G__test(G__assertion)) {
      if (G__security_error) {
         G__fprinterr(G__serr, "Warning: Assertion failed, delete assert expression %s\n", G__assertion);
         G__assertion[0] = 0;
      }
      return ignore;
   }
   fprintf(G__sout, "\n");
   G__init_process_cmd();
   // If display mode, display source code.
   if (G__breakdisp && G__ifile.name[0]) {
      G__pr(G__sout, view.file);
   }
   //
   //  Infinite loop, until space or 'S'(Step) command.
   //
   while (1) {
      // Set prompt string.
      if (prompt[0]) {
         command = prompt;
      }
      else {
         if (!view.file.name[0]) {
            command.Format("%s> ", cintname());
         }
         else {
            p = strrchr(view.file.name, G__psep[0]);
            if (p && *(p + 1)) {
               filename = p + 1;
            }
            else {
               filename = view.file.name;
            }
            command.Format("FILE:%s LINE:%d %s> ", G__stripfilename(filename), view.file.line_number, cintname());
         }
      }
      if (!more) {
         // --
#ifdef G__ASM
         G__abortbytecode();
#endif // G__ASM
         //
         //  Flush files.
         //
         G__disp_mask = 0;
         fflush(G__sout);
         fflush(G__serr);
#ifdef G__DUMPFILE
         if (G__dumpfile) {
            fflush(G__dumpfile);
         }
#endif // G__DUMPFILE
         // If G__atpause is defined, evaluate the function.
         if (G__atpause) {
            G__p2f_void_void((void*) G__atpause);
         }
         // Exit program if break key pressed.
#ifndef G__ROOT
         signal(SIGINT, G__killproc);
#endif // G__ROOT
         //
         //  If execute readline dump file mode, get
         //  debug command from the readline dump file
         //  else get command from stdin.
         //
         G__SET_TEMPENV;
         G__in_pause = 1;
      }
      else {
         // --
#ifndef G__ROOT
         signal(SIGINT, G__exit);
#endif // G__ROOT
         // --
      }
      char* tmp = G__input(command);
      command = tmp;
      if (!more) {
         G__in_pause = 0;
         G__RESET_TEMPENV;
      }
      // G__pause() when break key.
      oldhandler = (void(*)(int)) signal(SIGINT, G__breakkey);
      G__pause_return = 0;
      ignore = G__process_cmd(command, prompt, &more, 0, 0);
      if (G__return == G__RETURN_IMMEDIATE) {
         break;
      }
      if (ignore / G__PAUSE_ERROR_OFFSET) {
         break;
      }
      if (G__pause_return) {
         break;
      }
   }
   G__pause_return = 0;
#ifdef G__ROOT
   signal(SIGINT, oldhandler);
#endif // G__ROOT
   return ignore;
}

/******************************************************************
* G__update_stdio()
******************************************************************/
int G__update_stdio()
{
   G__FastAllocString command(G__LONGLINE);

   G__intp_sout = G__sout;
   G__intp_serr = G__serr;
   G__intp_sin = G__sin;
   command.Format("stderr=%ld", (long)G__intp_serr);
   G__getexpr(command);
   command.Format("stdout=%ld", (long)G__intp_sout);
   G__getexpr(command);
   command.Format("stdin=%ld", (long)G__intp_sin);
   G__getexpr(command);
   return(0);
}

#define G__OLDIMPLEMENTATION1983

/******************************************************************
* G__redirectoutput
******************************************************************/
static void G__redirectoutput(char *com
                              , FILE **psout, FILE **pserr
#ifdef G__REDIRECTIO
                              , FILE ** psin
#else
                              , FILE ** /* psin */
#endif
                              , int asemicolumn, char *keyword, char *pipefile)
{
   char *semicolumn;
   char *redirect;
   char *redirectin;
   char *singlequote;
   char *doublequote;
   char *paren;
   char *blacket;
   /* int issemicolumn; */
   const char *openmode;
   G__FastAllocString filename(G__MAXFILENAME);
   int i = 0;
   int j = 1;
   int mode = 0; /* 0:stdout, 1:stderr, 2:stdout+stderr */

   redirect = strrchr(com, '>');
   redirectin = strrchr(com, '<');
   semicolumn = strrchr(com, ';');
   singlequote = strrchr(com, '\'');
   doublequote = strrchr(com, '"');
   paren = strrchr(com, ')');
   blacket = strrchr(com, ']');

   pipefile[0] = 0;
   keyword[0] = 0;

   if (0 == asemicolumn ||
         (semicolumn && semicolumn < redirect &&
          ((char*)NULL == singlequote || semicolumn > singlequote) &&
          ((char*)NULL == doublequote || semicolumn > doublequote))) {

      if (redirect &&
            (isspace(*(redirect - 1)) ||
             (*(redirect - 1) == '2' && isspace(*(redirect - 2))) ||
             (*(redirect - 1) == '>' && isspace(*(redirect - 2))) ||
             (*(redirect - 1) == '>' && *(redirect - 2) == '2' && isspace(*(redirect - 3)))) &&
            ((char*)NULL == singlequote || redirect > singlequote) &&
            ((char*)NULL == doublequote || redirect > doublequote) &&
            ((char*)NULL == paren || redirect > paren) &&
            ((char*)NULL == blacket || redirect > blacket)) {

         /* check if redirect both mode
          *   cint> command >& filename
          *                 ^^          */
         if ('&' == (*(redirect + 1))) {
            mode = G__NUM_STDBOTH;
            ++j;
         }

         /* get filename to redirect
          *    cint> command > filename
          *                  ^ -------- */
         while ((*(redirect + j))) {
            if (!isspace(*(redirect + j))) {
               filename.Set(i++, *(redirect + j));
            }
            else if (i) break;
            ++j;
         }
         filename.Set(i, 0);
         strcpy(pipefile, filename); // Legacy, we don't known the input buffer size.

         /* get filename to redirect
          *    cint> command > filename  /keyword
          *                            ^-------- */
         while ((*(redirect + j)) && isspace((*(redirect + j)))) ++j;
         if ('/' == *(redirect + j)) {
            ++j;
            i = 0;
            while ((*(redirect + j))) keyword[i++] = *(redirect + j++);
            keyword[i] = '\0';
         }


         /* check if append mode
          *   cint> command >> filename 
          *                 ^^          */
         --redirect;
         if ('>' == (*redirect)) {
            --redirect;
            openmode = "a";
         }
         else {
            openmode = "w";
         }

         /* check if stderr
          *   cint> command 2> filename 
          *                 ^^          */
         if (mode == G__NUM_STDBOTH) {}
         else
            if ('2' == (*redirect)) {
               --redirect;
               mode = G__NUM_STDERR; /* stderr */
            }
            else {
               mode = G__NUM_STDOUT; /* stdout */
            }

         /* check if there is a space char right before redirection
          *   cint> command > filename
          *                ^          */
         if (isspace(*redirect) && filename[0]) {
            /* cut command string at redirect command
             *   cint> command > filename
             *                ^0          */
            *redirect = '\0';


            /* open redirect file */
#ifdef G__REDIRECTIO
            switch (mode) {
               case G__NUM_STDOUT:  /* stdout */
               case G__NUM_STDBOTH:  /* stdout and stderr */
               {
                  *psout = G__sout;
#ifndef G__WIN32
                  if (!strlen(stdoutsav)) {
                     const char *stdout_ttyname = ttyname(STDOUT_FILENO);
                     if (stdout_ttyname) G__strlcpy(stdoutsav, stdout_ttyname, sizeof(stdoutsav) );
                     else {
                        G__fprinterr(G__serr, "Error: stdout was already redirected to a file, it will be redirected but we will not be able to undo this redirection!\n");
                     }
                  }
#endif
                  G__sout = freopen(filename, openmode, G__sout);
                  if (!G__sout) {
                     FILE* donttouch = 0;
                     G__sout = *psout;
                     G__unredirectoutput(psout, &donttouch, &donttouch, "", "");
                     G__fprinterr(G__serr, "Error: cannot open pipe output file %s!\n",
                                  filename());
                  }
                  else
                     G__redirect_on();
                  if (mode == G__NUM_STDOUT)
                     break;
               }
               case G__NUM_STDERR: /* stderr */
               {
                  *pserr = G__serr;
#ifndef G__WIN32
                  if (!strlen(stderrsav)) {
                     const char *stderr_ttyname = ttyname(STDERR_FILENO);
                     if (stderr_ttyname) G__strlcpy(stderrsav, stderr_ttyname, sizeof(stderrsav));
                     else {
                        G__fprinterr(G__serr, "Error: stderr was already redirected to a file, it will be redirected but we will not be able to undo this redirection!\n");
                     }
                  }
#endif
                  G__serr = freopen(filename, openmode, G__serr);
                  if (!G__serr) {
                     FILE* donttouch = 0;
                     G__serr = *pserr;
                     G__unredirectoutput(&donttouch, pserr, &donttouch, "", "");
                     G__fprinterr(G__serr, "Error: cannot open error pipe output file %s!\n",
                                  filename());
                  }
                  /*DEBUG G__dumpfile = G__serr; */
                  break;
               }
            }
#else
            switch (mode) {
               case G__NUM_STDOUT: /* stdout */
                  *psout = G__sout;
                  G__sout = fopen(filename, openmode);
                  break;
               case G__NUM_STDERR: /* stderr */
                  *pserr = G__serr;
                  /*DEBUG G__dumpfile = 0; */
                  G__serr = fopen(filename, openmode);
                  break;
               case G__NUM_STDBOTH: /* stdout & stderr */
                  *psout = G__sout;
                  *pserr = G__serr;
                  G__sout = fopen(filename, openmode);
                  G__serr = fopen(filename, "a");
                  break;
            }
            G__update_stdio(); /* update stdout,stderr,stdin in interpreter */
#endif
         } /* isspace */
      } /* redirect && ... */
   }
#ifdef G__REDIRECTIO
   if (
      !asemicolumn ||
      (
         semicolumn &&
         (semicolumn < redirectin) &&
         (!singlequote || (semicolumn > singlequote)) &&
         (!doublequote || (semicolumn > doublequote))
      )
   ) {
      if (redirectin &&
            isspace(*(redirectin - 1)) &&
            ((char*)NULL == singlequote || redirectin > singlequote) &&
            ((char*)NULL == doublequote || redirectin > doublequote) &&
            ((char*)NULL == paren || redirectin > paren) &&
            ((char*)NULL == blacket || redirectin > blacket)) {
         /* get filename to redirect
          *    cint> command < filename
          *                  ^ -------- */
         i = 0;
         j = 1;
         while ((*(redirectin + j))) {
            if (!isspace(*(redirectin + j))) {
               filename.Set(i++, *(redirectin + j));
            }
            else if (i) break;
            ++j;
         }
         filename.Set(i, 0);

         *redirectin = '\0';
         --redirectin;
         if (isspace(*redirectin)) {
            *psin = G__sin;
#ifndef G__WIN32
            if (!strlen(stdinsav) && ttyname(STDIN_FILENO)) {
               G__strlcpy(stdinsav, ttyname(STDIN_FILENO), sizeof(stdinsav));
            } else {
               stdinsav[0] = 0;
            }
#endif
            G__sin = freopen(filename, "r", G__sin);
            if (!G__sin) {
               FILE* donttouch = 0;
               G__sin = *psin;
               G__unredirectoutput(&donttouch, &donttouch, psin, "", "");
               G__fprinterr(G__serr, "Error: cannot open input pipe from file %s!\n",
                            filename());
            }
         }
      }
   }
#endif
}

/******************************************************************
* G__unredirectoutput
******************************************************************/
static void G__unredirectoutput(FILE **sout, FILE **serr, FILE **sin, const char *keyword, const char *pipefile)
{
   // --
#ifdef G__REDIRECTIO
   G__redirect_off();
   if (*sout) {
      if (G__sout_keeper) {
         fclose(G__sout);
         G__sout = G__sout_keeper;
      } else {
#ifdef G__WIN32
      G__sout = freopen("CONOUT$", "w", G__sout);
#else
      G__sout = freopen(stdoutsav, "w", G__sout);
#endif
      }
      *sout = (FILE*)NULL;
   }
   if (*serr) {
#ifdef G__WIN32
      G__serr = freopen("CONOUT$", "w", G__serr);
#else
      G__serr = freopen(stderrsav, "w", G__serr);
#endif
      *serr = (FILE*)NULL;
   }
   if (*sin) {
#ifdef G__WIN32
      *sin = freopen("CONIN$", "r", *sin);
#else
      *sin = freopen(stdinsav, "r", *sin);
#endif
      *sin = (FILE*)NULL;
   }
#else /* G__REDIRECTIO */
   int flag = 0;
   G__redirect_off();
   if (*sout) {
      fclose(G__sout);
      G__sout = *sout;
      ++flag;
   }
   if (*serr) {
      fclose(G__serr);
      G__serr = *serr;
      ++flag;
   }
   if (*sin) {
      fclose(G__sin);
      G__sin = *sin;
      ++flag;
   }
   if (flag) G__update_stdio();
#endif /* G__REDIRECTIO */
   if (pipefile[0] && keyword[0]) {
      FILE *keyfile = fopen(pipefile, "r");
      G__display_keyword(G__sout, keyword, keyfile);
      fclose(keyfile);
   }
}

/******************************************************************
* G__IsBadCommand
******************************************************************/
int G__IsBadCommand(char *com, const unsigned int comlen)
{
   int i = 0;
   int nest = 0;
   int single_quote = 0;
   int double_quote = 0;
#ifndef G__OLDIMPLEMENTATION1774
   int semicolumnattheend = 0;
#endif
   while (com[i] != '\0') {
readagain:
      switch (com[i]) {
         case '"':
            if (single_quote == 0) double_quote ^= 1;
            break;
         case '\'':
            if (double_quote == 0) single_quote ^= 1;
            break;
         case '{':
         case '(':
         case '[':
            if ((single_quote == 0) && (double_quote == 0)) nest++;
            break;
         case '}':
         case ')':
         case ']':
            if ((single_quote == 0) && (double_quote == 0)) nest--;
            break;
         case '\\':
            ++i;
            if (0 == com[i] || '\n' == com[i]) {
               --i;
               char * input = G__input("> ");
               if(strlen(input) + i >= comlen) return -2;
               strcpy(com + i, input); 
               if (G__return == G__RETURN_IMMEDIATE) return(-1);
            }
            break;
         case '/':
            if ((single_quote == 0) && (double_quote == 0)) {
               if ('/' == com[i+1]) {
                  com[i] = 0;
                  com[i+1] = 0;
                  --i;
               }
            }
            break;
      }
#ifndef G__OLDIMPLEMENTATION1774
      if (';' == com[i]) {
         if ((single_quote == 0) && (double_quote == 0) && (nest == 0)) semicolumnattheend = 1;
      }
      else {
         if (!isspace(com[i])) semicolumnattheend = 0;
      }
#endif
      ++i;
   }
   /* #define G__OLDIMPLEMENTATION1774 */
#ifndef G__OLDIMPLEMENTATION1774
   if (nest > 0 && '{' != com[0]) {
      if (0 == strncmp(com, "for(", 4) || 0 == strncmp(com, "for ", 4)
            || 0 == strncmp(com, "while(", 6) || 0 == strncmp(com, "while ", 6)
            || 0 == strncmp(com, "do ", 3) || 0 == strncmp(com, "do{", 3)
            || 0 == strncmp(com, "namespace ", 10) || 0 == strncmp(com, "namespace{", 10)) {
         char * input = G__input("end with '}', '@':abort > ");
         if(strlen(input) + i >= comlen) return -2;
         G__strlcpy(com + i, input,comlen);
      }
      else {
         char * input = G__input("end with '}', '@':abort > ");
         if(strlen(input) + i >= comlen) return -2;
         G__strlcpy(com + i, input,comlen);
      }
      if (G__return == G__RETURN_IMMEDIATE) return(-1);
      if ('@' == com[i]) {
         com[0] = 0;
         return(0);
      }
      goto readagain;
   }
   if (0 < nest) return(1);
   if (G__INPUTCXXMODE == G__rootmode && 0 == nest && 0 == semicolumnattheend
         && '#' != com[0]
         && 0 != strncmp(com, "for(", 4) && 0 != strncmp(com, "for ", 4)
         && 0 != strncmp(com, "while(", 6) && 0 != strncmp(com, "while ", 6)
         && 0 != strncmp(com, "do ", 3) && 0 != strncmp(com, "do{", 3)
         && 0 != strncmp(com, "namespace ", 10) && 0 != strncmp(com, "namespace{", 10)
      ) {
      char * input = G__input("end with ';', '@':abort > ");
      if(strlen(input) + i >= comlen) return -2;
      G__strlcpy(com + i, input,comlen);
      if (G__return == G__RETURN_IMMEDIATE) return(-1);
      if ('@' == com[i]) {
         com[0] = 0;
         return(0);
      }
      goto readagain;
   }
   if (single_quote || double_quote || nest < 0) return(-1);
   return(0);
#else
   if (0 != nest || single_quote || double_quote) return(1);
   else return(0);
#endif
}

/******************************************************************
* G__ReadInputMode
******************************************************************/
int G__ReadInputMode()
{
  static int inputmodeflag=0;
  if(inputmodeflag==0) {
    const char *inputmodebuf;
    inputmodeflag=1;
    inputmodebuf=getenv("INPUTMODE");
    if(inputmodebuf==0) inputmodebuf=G__getmakeinfo1("INPUTMODE");
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

/******************************************************************
* G__debugvariable()
******************************************************************/
void G__debugvariable(FILE *fp,G__var_array *var,char *name) 
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
           fprintf(fp,"[%lu]",(unsigned long)var->varlabel[ig15][i++]);
        }
        fprintf(fp,"\n");
      }
    }
    var=var->next;
  }
}

#ifdef G__WIN32
const char* G__tmpfilenam();
#endif

static void G__create_input_tmpfile(G__input_file& ftemp)
{
#ifdef G__WIN32
   G__strlcpy(ftemp.name, G__tmpfilenam(),G__MAXFILENAME);
   ftemp.fp = fopen(ftemp.name, "w+bTD"); // write and read (but write first), binary, temp, and delete when closed
#else
   // coverity[secure_temp]: we don't care about predictable names.
   ftemp.fp = tmpfile();
   G__strlcpy(ftemp.name, "(tmpfile)",G__MAXFILENAME);
#endif
}

static void G__remove_input_tmpfile(G__input_file& ftemp)
{
#ifdef G__WIN32
   unlink(ftemp.name);
#else
   if (!&ftemp) { // FIXME: What could this possibly mean?
      return;
   }
#endif
}

/******************************************************************
* int G__process_cmd()
******************************************************************/
int G__process_cmd(char* line, char* prompt, int* more, int* err, G__value* rslt)
{
   FILE *tempfp;   /* used for input dump file */
   G__FastAllocString command(G__LONGLINE);
   G__FastAllocString syscom(G__LONGLINE);
   G__FastAllocString editor(64);
   size_t temp, temp1 = 0, temp2;
   int index = -1;
   int ignore = G__PAUSE_NORMAL;
   short double_quote, single_quote;
#ifdef G__ASM
   G__ALLOC_ASMENV;
   (void)store_asm_loopcompile; // set but not used
#endif
   char store_var_type;
   /* pass to parent otherwise not re-entrant */
#ifdef G__TMPFILE
   static char tname[G__MAXFILENAME];
   G__FastAllocString sname(G__MAXFILENAME);
#else
   static char tname[L_tmpnam+10];
   G__FastAllocString sname(L_tmpnam+10);
#endif
   struct G__input_file ftemp;
   ftemp.fp = 0;
   struct G__var_array *local;
   char *com, *string;
   char *stringb;
   int line_number, filenum;
   G__value buf;
   FILE *G__temp;
   char *evalbase;
   int base = 0 /* ,digit */;
   unsigned int num;
   G__FastAllocString evalresult(G__ONELINE);
   FILE* store_stderr = NULL;
   FILE* store_stdout = NULL;
   FILE* store_stdin = NULL;
   G__FastAllocString keyword(G__ONELINE);
   G__FastAllocString pipefile(G__MAXFILENAME);
   int dmy = 0;
   int noprintflag = 0;
   int istmpnam = 0;
   if (!err) err = &dmy;
   G__ReadInputMode();
   G__LockCriticalSection();
   keyword[0] = 0;
   pipefile[0] = 0;
   if (0 == init_process_cmd_called)
      G__fprinterr(G__serr, "Internal error: G__init_process_cmd must be called before G__process_cmd\n");

   if (strlen(line) > G__LONGLINE - 5) {
      G__fprinterr(G__serr, "!!! User command too long (%d>%d)!!!\n", strlen(line), G__LONGLINE - 5);
      G__UnlockCriticalSection();
      return(0);
   }
   *prompt = '\0';
   if ((com = getenv("EDITOR"))) editor = com;
#ifdef G__WIN32
   else editor = "notepad";
#else
   else editor = "vi";
#endif
   temp = 0;
   while (isspace(line[temp])) ++temp;
   if (*more == 0) {
      command = line + temp;
      com = strchr(command, '\n');
      if (com) *com = '\0';
      else {
         com = strchr(command, '\r');
         if (com) *com = '\0';
      }
      if (G__INPUTROOTMODE&G__rootmode) {
         if (command[0] == '.') {
            syscom = command + 1;
            command = syscom;
         }
         else if (strcmp(command, "?") == 0) {
            command = "help";
         }
         else if ('\0' != command[0] && command[0] != '{') {
#ifndef G__OLDIMPLEMENTATION1774
            switch (G__IsBadCommand(command, G__LONGLINE)) {
               case 0:
                  break;
               case 1:
                  com = command;
                  goto multi_line_command;
               case -2:
                  fprintf(stderr, "!!!command input too long.\n");
                  // intentional fall-through
               case -1:
               default:
                  fprintf(stderr, "!!!Bad command input. Ignored!!!\n");
                  G__UnlockCriticalSection();
                  return(ignore = G__PAUSE_NORMAL);
                  break;
            }
#else /* 1774 */
            if (G__IsBadCommand(command), G__LONGLINE) {
               fprintf(stderr, "!!!Bad command input. Ignored!!!\n");
               G__UnlockCriticalSection();
               return(ignore = G__PAUSE_NORMAL);
            }
#endif /* 1774 */
            G__redirectoutput(command, &store_stdout, &store_stderr, &store_stdin, 1, keyword, pipefile);
            if (strlen(command) > 0) {
               temp = strlen(command) - 1;
               while (temp && isspace(command[temp])) --temp;
            } else temp = 0;
            if (command[temp] == ';') {
               syscom.Format("{%s}", command());
#ifndef G__OLDIMPLEMENTATION1774
               if (G__INPUTCXXMODE != G__rootmode) noprintflag = 1;
#else
               noprintflag = 1;
#endif
            }
            else {
               syscom.Format("{%s;}", command());
            }
            command = syscom;
         }
      }
   }
   else {
      command.Format("{%s", line + temp);
      com = strchr(command, '\n');
      if (com) *com = '\0';
      else {
         com = strchr(command, '\r');
         if (com) *com = '\0';
      }
   }
   com = command;
   if (
      strlen(com) >= 3 &&
      0 != strncmp(com, "{", 1) &&
      0 != strncmp(com, "p", 1) &&
      0 != strncmp(com, ">", 1) &&
      0 != strncmp(com, "<", 1) &&
      0 != strncmp(com, "2>", 2)
   ) {
      G__redirectoutput(com, &store_stdout, &store_stderr, &store_stdin, 0, keyword, pipefile);
   }
   temp = 0;
   if (isalpha(command[temp])) {
      while (isalpha(command[temp])) ++temp;
   }
   else if (command[temp] && !isdigit(command[temp])) {
      ++temp;
   }
   string = command + temp;
   stringb = string;
   while (isspace(*stringb) && (*stringb)) ++stringb;
   G__storerewindposition();
   /************************************
    *
    * Break prompt command parser
    * cint>    X  argument
    *          ^^
    *          com string
    *
    ************************************/
   if (strncmp("class", com, 3) == 0) {
      char *p = strchr(string, '/');
      if (p) {
         *p = 0;
         ++p;
      }
      G__display_classkeyword(G__sout, string, p, 0);
   }
   else if (strncmp("Class", com, 3) == 0) {
      char *p = strchr(string, '/');
      if (p) {
         *p = 0;
         ++p;
      }
      G__display_classkeyword(G__sout, string, p, 1);
   }
   else if (strncmp("typedef", com, 3) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__display_typedef(G__sout, string, 0);
   }
   else if (strncmp("newtypes", com, 4) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__display_newtypes(G__sout, stringb);
   }
   else if (strncmp("tempobj", com, 5) == 0) {
      fprintf(G__sout, "!!!Display temp object stack!!!\n");
      G__display_tempobj(G__sout);
   }
   else if (strncmp("template", com, 4) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__display_template(G__sout, string);
   }
   else if (strncmp("macro", com, 3) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__display_macro(G__sout, string);
   }
   else if (strncmp("include", com, 4) == 0) {
      if (0 == (*stringb)) G__display_includepath(G__sout);
      else {
         string = stringb + strlen(stringb) - 1;
         while (isspace(*string)) *(string--) = 0;
         G__add_ipath(stringb);
      }
   }
   else if (strncmp("language", com, 4) == 0) {
      int ix = 0;
      while (stringb[ix]) {
         stringb[ix] = toupper(stringb[ix]);
         ++ix;
      }
      if (strncmp(stringb, "EUC", 3) == 0) G__lang = G__EUC;
      else if (strncmp(stringb, "SJIS", 3) == 0) G__lang = G__SJIS;
      else if (strncmp(stringb, "JIS", 3) == 0) G__lang = G__JIS;
      else if (strncmp(stringb, "EUROPEAN", 3) == 0) G__lang = G__ONEBYTE;
      else if (strncmp(stringb, "UNKNOWN", 3) == 0) G__lang = G__UNKNOWNCODING;
      else G__lang = (short)G__int(G__calc(stringb));
   }
   else if (strncmp("J", com, 1) == 0) {
      G__dispmsg = G__int(G__getexpr(string));
   }
   else if (strncmp("file", com, 4) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__display_files(G__sout);
   }
   else if (strncmp("string", com, 4) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__display_string(G__sout);
   }
   else if (strncmp("garbage", com, 4) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__disp_garbagecollection(G__sout);
   }
   else if (strncmp("Garbage", com, 4) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__garbagecollection();
      G__disp_garbagecollection(G__sout);
   }
   else if (strncmp("trace", com, 3) == 0) {
      G__set_tracemode(string);
   }
   else if (strncmp("deltrace", com, 4) == 0) {
      G__del_tracemode(string);
   }
   else if (strncmp("break", com, 4) == 0) {
      G__set_classbreak(string);
   }
   else if (strncmp("delbreak", com, 4) == 0) {
      G__del_classbreak(string);
   }
   else if (strncmp("ls", com, 2) == 0 ||
            strncmp("ll", com, 2) == 0 ||
            strncmp("dir", com, 3) == 0 ||
            strncmp("pwd", com, 3) == 0 ||
            strncmp("cp", com, 2) == 0 ||
            strncmp("copy", com, 4) == 0 ||
            strncmp("gmake", com, 5) == 0 ||
            strncmp("make", com, 4) == 0 ||
            strncmp("mv", com, 2) == 0 ||
            strncmp("move", com, 4) == 0 ||
            strncmp("mkdir", com, 5) == 0 ||
            strncmp("vi", com, 2) == 0 ||
            strncmp("notepad", com, 7) == 0 ||
            strncmp("grep", com, 4) == 0 ||
            strncmp("fgrep", com, 5) == 0 ||
            strncmp("egrep", com, 5) == 0 ||
            strncmp("diff", com, 4) == 0 ||
            strncmp("wc", com, 2) == 0 ||
            strncmp("cat", com, 3) == 0 ||
            strncmp("more", com, 4) == 0 ||
            strncmp("less", com, 4) == 0 ||
            strncmp("echo", com, 4) == 0 ||
            strncmp("cint", com, 4) == 0 ||
            strncmp("du", com, 2) == 0 ||
            strncmp("sh", com, 2) == 0 ||
            strncmp("csh", com, 3) == 0 ||
            strncmp("ksh", com, 3) == 0 ||
            strncmp("bash", com, 4) == 0 ||
            strncmp("man", com, 3) == 0 ||
            strncmp("type", com, 4) == 0) {
      int ret = system(com);
      if (rslt) G__letint(rslt, 'i', ret);
   }
   else if (strncmp("rm", com, 2) == 0 ||
            strncmp("del", com, 4) == 0 ||
            strncmp("rmdir", com, 5) == 0) {
      G__FastAllocString localbuf(G__input("Are you sure(y/n)? "));
      if (tolower(localbuf[0]) == 'y') {
         int ret = system(com);
         if (rslt) G__letint(rslt, 'i', ret);
      }
      else fprintf(G__sout, "aborted\n");
   }
   else if (strncmp("set", com, 3) == 0 ||
            strncmp("export", com, 6) == 0) {
      char *plocal = strchr(stringb, '=');
      if (plocal) {
#if defined(G__WIN32)
         *plocal++ = 0;
         if (FALSE == SetEnvironmentVariable(stringb, plocal))
            G__fprinterr(G__serr, "can not set environment variable %s=%s\n", stringb, plocal);
#elif defined(G__POSIX)
         if (0 != putenv(stringb))  /* DOES NOT WORK , WHY ??? */
            G__fprinterr(G__serr, "can not set environment variable %s\n", stringb);
#endif
      }
      else {
         int ret = system(com);
         if (rslt) G__letint(rslt, 'i', ret);
      }
   }
   else if (strncmp("cd", com, 2) == 0) {
#if defined(G__WIN32)
      if (FALSE == SetCurrentDirectory(stringb)) {
         G__fprinterr(G__serr, "can not change directory to %s\n", stringb);
      }
#elif defined(G__POSIX)
      if (0 != chdir(stringb)) {
         G__fprinterr(G__serr, "can not change directory to %s\n", stringb);
      }
#endif
   }
   else if (strncmp("undo", com, 4) == 0 || strncmp("rewind", com, 3) == 0) {
      G__rewind_undo_position();
   }
#ifdef G__SECURITY
   else if (strncmp("where", com, 4) == 0) {
      fprintf(G__sout, "FILE:%s LINE:%d interpreter=%s\n", G__stripfilename(view.file.name), view.file.line_number, G__nam);
   }
   else if (strncmp("security", com, 4) == 0) {
      fprintf(G__sout, "security ");
      switch (G__security) {
         case 0x0000000f:
            fprintf(G__sout, "level0");
            break;
         case 0x000000ff:
            fprintf(G__sout, "level1");
            break;
         case 0x00000fff:
            fprintf(G__sout, "level2");
            break;
         case 0x0000ffff:
            fprintf(G__sout, "level3");
            break;
         case 0x000fffff:
            fprintf(G__sout, "level4");
            break;
         case 0x00ffffff:
            fprintf(G__sout, "level5");
            break;
         case 0x0fffffff:
            fprintf(G__sout, "level6");
            break;
         case 0xffffffff:
            fprintf(G__sout, "level7");
            break;
         default:
            break;
      }
#ifdef G__64BIT
      fprintf(G__sout, "  0x%x\n", G__security);
#else
      fprintf(G__sout, "  0x%lx\n", G__security);
#endif
   }
   else if (strncmp("Security", com, 4) == 0) {
      G__security = G__int(G__calc_internal(string));
   }
   else if (strncmp("refcount", com, 3) == 0) {
      if (G__security&G__SECURE_GARBAGECOLLECTION) {
         G__security &= ~G__SECURE_GARBAGECOLLECTION;
         fprintf(G__sout, "!!!new/delete reference count control turned off");
      }
      else {
         G__security |= G__SECURE_GARBAGECOLLECTION;
         fprintf(G__sout, "!!!new/delete reference count control turned on");
      }
      fprintf(G__sout, "  0x%x\n", (unsigned int)G__security);
   }
#endif /* G__SECURITY */
   else if (strncmp("scratch", com, 4) == 0) {
      if (!G__isfilebusy(0)) {
         G__scratch_all();
         G__init_undo();
      }
   }
#ifndef G__ROOT
   else if (strncmp(".", com, 1) == 0) {
      if (0 == G__lockinputmode) {
         G__rootmode ^= 1;
         fprintf(G__sout, "!!!Debugger Command mode switched as follows!!!\n");
         if (G__INPUTROOTMODE&G__rootmode) {
            fprintf(G__sout, "    > .[command]\n");
            fprintf(G__sout, "    > [statement]\n");
         }
         else {
            fprintf(G__sout, "    > [command]\n");
            fprintf(G__sout, "    > { [statement] }\n");
         }
      }
      else {
         fprintf(G__sout, "!!!Debugger Command mode locked!!!\n");
      }
   }
#endif
   else if (strncmp("reset", com, 4) == 0) {
#ifdef G__ROOT
      G__fprinterr(G__serr, "!!! Sorry, can not reset interpreter !!!\n");
#else
      if (!G__isfilebusy(0)) {
         int store_othermain = G__othermain;
         G__scratch_all(); /* To make memory leak check work */
         if (G__commandline[0]) G__init_cint(G__commandline);
         else {
            G__init_cint(G__nam);
         }
         G__othermain = store_othermain;
         G__init_undo();
      }
      else {
         G__fprinterr(G__serr, "!!! Sorry, can not reset interpreter !!!\n");
      }
#endif
   }
   else if (strncmp("function", com, 4) == 0) {
      /*******************************************************
       * List up interpreted functions and shared library
       *******************************************************/
      G__more_pause((FILE*)NULL, 1);
      G__listfunc(G__sout, G__PUBLIC_PROTECTED_PRIVATE, (char*)NULL
                  , (struct G__ifunc_table*)NULL);
      G__listshlfunc(G__sout);
   }
   else if (strncmp("interactive", com, 5) == 0) {
      G__interactive ^= 1;
      fprintf(G__sout, "interactive mode=%d\n", G__interactive);
   }
   else if (strncmp("coverage", com, 5) == 0) {
      temp = 0;
      while (isspace(string[temp])) temp++;
      if (isalpha(string[temp]))
         tempfp = fopen(string + temp, "w");
      else
         tempfp = (FILE*)NULL;
      if (tempfp) {
         fprintf(G__sout, "saving trace coverage to %s\n", string + temp);
         G__dump_tracecoverage(tempfp);
         fclose(tempfp);
      }
      else {
         fprintf(G__sout, "can not open file %s\n", string + temp);
      }
   }
   else if (strncmp("debug", com, 5) == 0) {
      /* #ifdef G__ASM_DBG */
      G__asm_dbg ^= 1;
      fprintf(G__sout, "G__asm_dbg=%d\n", G__asm_dbg);
      /* #endif */
   }
   else if (strncmp("asmstep", com, 5) == 0) {
#ifdef G__ASM_DBG
      G__asm_step ^= 1;
      fprintf(G__sout, "G__asm_step=%d\n", G__asm_step);
#endif
   }
   else if (strncmp("dasm", com, 4) == 0) {
      G__dasm(G__sout, 0);
   }
   else if (strncmp("stack", com, 4) == 0) {
#ifdef G__ASM_DBG
      /* G__display_stack(); */
#endif
   }
   else if (strncmp("exception", com, 4) == 0) {
      G__catchexception ^= 1;
      fprintf(G__sout, "G__catchexception=%d\n", G__catchexception);
   }
   else if (strncmp("preproc", com, 4) == 0) {
      G__cpp ^= 1;
      fprintf(G__sout, "G__cpp=%d\n", G__cpp);
   }
   else if (strncmp("autodict", com, 8) == 0) {
      G__EnableAutoDictionary ^= 1;
      fprintf(G__sout, "Automatic building of dictionaries now %s\n",
              G__EnableAutoDictionary ? "on " : "off");
   }
   else if (strncmp("status", com, 4) == 0) {
#ifdef G__ASM_DBG
      fprintf(G__sout, "G__asm_noverflow=%d\n", G__asm_noverflow);
      fprintf(G__sout, "G__no_exec_compile=%d\n", G__no_exec_compile);
      fprintf(G__sout, "G__no_exec=%d\n", G__no_exec);
      fprintf(G__sout, "G__asm_exec=%d\n", G__asm_exec);
      fprintf(G__sout, "G__asm_loopcompile=%d\n", G__asm_loopcompile);
      fprintf(G__sout, "G__asm_cp=%d\n", G__asm_cp);
      fprintf(G__sout, "G__asm_dt=%d\n", G__asm_dt);
#endif
   }
   else if (strncmp("OPTIMIZE", com, 1) == 0) {
      G__asm_loopcompile = G__int(G__calc_internal(string));
      G__asm_loopcompile_mode = G__asm_loopcompile;
   }
#ifdef G__ASM_WHOLEFUNC
   else if (strncmp("WHOLEFUNC", com, 3) == 0) {
      G__asm_wholefunction = G__int(G__calc_internal(string));
   }
#endif
   /* interactive return of undefined symbol */
   else if (strncmp("return", com, 1) == 0) {
      if (G__interactive_undefined) {
         G__interactivereturnvalue = G__calc_internal(string);
         G__pause_return = 1;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                             , keyword, pipefile);
         ignore = G__PAUSE_NORMAL;
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      else {
         G__fprinterr(G__serr, "!!! Use 'return' command at your own risk !!!\n");
         G__interactivereturnvalue = G__calc_internal(string);
         G__return = G__RETURN_IMMEDIATE;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                             , keyword, pipefile);
         ignore = 2;
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
   }
   else if (strncmp(">&", com, 2) == 0 || strncmp(">>&", com, 3) == 0) {
      if (strncmp(">>&", com, 3) == 0) index = 3;
      else                        index = 2;
      while (isspace(command[index]) && command[index] != '\0') index++;
      if ((*(command + index))) {
         if (G__sout != G__stdout) {
            fprintf(G__stdout, "Old save file closed\n");
            fclose(G__sout);
         }
         if (G__serr != G__stderr && G__serr != G__sout) {
            fprintf(G__stdout, "Old save file closed\n");
            fclose(G__serr);
         }
         if (strncmp(">>", com, 2) != 0) {
            G__sout = fopen(command + index, "w");
            fclose(G__sout);
         }
         G__serr = G__sout = fopen(command + index, "a");
         G__redirectcout(command + index);
         G__redirectcerr(command + index);
         if (G__sout) {
            fprintf(G__stdout, "Output will be saved in file %s! ('>&' to display on screen)\n"
                    , command + index);
         }
         else {
            G__sout = G__stdout;
            G__serr = G__stderr;
            fprintf(G__stdout, "Can not open file %s\n", command + index);
         }
      }
      else {
         if (G__sout && G__sout != G__stdout) {
            G__unredirectcout();
            G__unredirectcerr();
            fclose(G__sout);
         }
         G__sout = G__stdout;
         G__serr = G__stderr;
         fprintf(G__stdout, "Output will be displayed on screen!\n");
      }
      G__update_stdio();
   }
   else if (strncmp(">", com, 1) == 0) {
      if (strncmp(">>", com, 2) == 0) index = 2;
      else                       index = 1;
      while (isspace(command[index]) && command[index] != '\0') index++;
      if ((*(command + index))) {
         if (G__sout != G__stdout) {
            fprintf(G__stdout, "Old save file closed\n");
            fclose(G__sout);
         }
         if (strncmp(">>", com, 2) != 0) {
            G__sout = fopen(command + index, "w");
            fclose(G__sout);
         }
         G__sout = fopen(command + index, "a");
         G__redirectcout(command + index);
         if (G__sout) {
            fprintf(G__stdout, "Output will be saved in file %s! ('>' to display on screen)\n"
                    , command + index);
         }
         else {
            G__sout = G__stdout;
            fprintf(G__stdout, "Can not open file %s\n", command + index);
         }
      }
      else {
         if (G__sout && G__sout != G__stdout) {
            G__unredirectcout();
            fclose(G__sout);
         }
         G__sout = G__stdout;
         fprintf(G__stdout, "Output will be displayed on screen!\n");
      }
      G__update_stdio();
   }
   else if (strncmp("2>", com, 2) == 0) {
      if (strncmp("2>>", com, 3) == 0) index = 3;
      else                             index = 2;
      while (isspace(command[index]) && command[index] != '\0') index++;
      if ((*(command + index))) {
         if (index==2) { // strncmp("2>>", com, 3) != 0) {
            G__serr = fopen(command + index, "w");
            fclose(G__serr);
            /* fclose(G__sout); */
         }
         G__serr = fopen(command + index, "a");
         G__redirectcerr(command + index);
         if (G__serr) {
            fprintf(G__stdout, "Error will be saved in file %s! ('2>' to display on screen)\n"
                    , command + index);
         }
         else {
            G__serr = G__stderr;
            fprintf(G__stdout, "Can not open file %s\n", command + index);
         }
      }
      else {
         if (G__serr && G__serr != G__stderr) {
            G__unredirectcerr();
            fclose(G__serr);
         }
         G__serr = G__stderr;
         fprintf(G__sout, "Error will be displayed on screen!\n");
      }
      G__update_stdio();
   }
   else if (strncmp("<", com, 1) == 0) {
      /*******************************************************
       * Execute dumpped readline file
       *******************************************************/
      index = 1;
      while (isspace(command[index]) && command[index] != '\0') index++;
      tempfp = fopen(command + index, "r");
      if (tempfp) {
         fprintf(G__sout, "Execute readline dumpfile %s\n"
                 , command + index);
         G__pushdumpinput(tempfp, 1);
      }
      else {
         fprintf(G__sout, "Can not open file %s\n", command + 1);
      }
      G__update_stdio();
   }
   else if (strncmp("X", com, 1) == 0) {
      bool keepIfLoaded = com[1] == 'k';
      /*******************************************************
       * Execute C/C++ source file. Filename minus extension
       * must match a function in the source file. This function
       * will be automatically executed.
       *******************************************************/
      struct G__store_env store;
      G__SET_TEMPENV;

      char *com1;
      temp = 0;
      while (isspace(string[temp])) temp++;
      if (string[temp] == '\0') {
         G__fprinterr(G__serr, "Error: no file specified\n");
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                             , keyword, pipefile);
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      G__store_undo_position();
      com = strchr(string + temp, '(');
      if (com) {
         char *px = com - 1;
         *com = '\0';
         while (isspace(*px)) *px-- = '\0';
      }
      temp2 = G__prerun;
      G__prerun = 1;  /* suppress warning message if file already loaded */
      temp1 = G__loadfile(string + temp);
      if (!keepIfLoaded && temp1 == 1) {
         G__prerun = 0;
         G__unloadfile(string + temp);
         G__storerewindposition();
         if (G__loadfile(string + temp)) {
            G__prerun = temp2;
            G__pause_return = 1;
            G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                                , keyword, pipefile);
            if (G__security_error) G__cancel_undo_position();
#ifdef G__SECURITY
            *err |= G__security_recover(G__serr);
#endif
            G__UnlockCriticalSection();
            return(ignore);
         }
      }

      G__prerun = temp2;
      com1 = strrchr(string + temp, '.');
      if (com) {
         char *px = com - 1;
         *com = '(';
         while (0 == (*px)) *px-- = ' ';
      }
      com = com1;
      if (com) {
         char *s;
         *com = '\0';
         s = strrchr(string + temp, '/');
         if (!s) s = strrchr(string + temp, '\\');
         if (!s) s = string + temp;
         else   s++;
         syscom = s;
         s = syscom;
         while (s && *s) {
            if ('-' == (*s)) *s = '_';
            ++s;
         }
         string = strchr(com + 1, '(');
         if (string)
            syscom += string;
         else
            syscom += "()";
         buf = G__calc_internal(syscom);
         if (rslt) *rslt = buf;
         G__in_pause = 1;
         G__valuemonitor(buf, syscom);
         G__in_pause = 0;
#ifndef G__OLDIMPLEMENTATION1259
         if (buf.isconst&(G__CONSTVAR | G__CONSTFUNC)) {
            G__FastAllocString tmp(G__ONELINE);
            tmp.Format("(const %s", syscom + 1);
            syscom = tmp;
         }
         if (buf.isconst&G__PCONSTVAR) {
            G__FastAllocString tmp2(G__ONELINE);
            char *ptmp = strchr(syscom, ')');
            tmp2 = ptmp;
            syscom.Replace((ptmp-syscom()), "const");
            syscom += tmp2;
         }
#endif
         if (buf.type && 0 == G__atevaluate(buf)) fprintf(G__sout, "%s\n", syscom());
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
      }
      else {
         fprintf(G__sout, "Expecting . in filename\n");
      }
      G__RESET_TEMPENV;
   }
   else if (
      (G__do_smart_unload &&
       strncmp("L", com, 1) == 0 &&
       strncmp("Lk", com, 2) != 0
      ) ||
      strncmp("Lall", com, 2) == 0
   ) {
      bool keepIfLoaded = com[1] == 'k';
      temp = 0;
      while (isspace(string[temp])) temp++;
      G__UnlockCriticalSection();
      G__reloadfile(string + temp, keepIfLoaded);
      G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                          , keyword, pipefile);
      *err |= G__security_recover(G__serr);
      return(ignore);
   }
   else if (strncmp("L", com, 1) == 0) {
      /*******************************************************
       * Load(Re-Load) a C/C++ source file.
       *******************************************************/
      bool keepIfLoaded = com[1] == 'k';
      temp = 0;
      while (isspace(string[temp])) temp++;
      if (string[temp] == '\0') {
         G__fprinterr(G__serr, "Error: no file specified\n");
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      G__store_undo_position();
      temp2 = G__prerun;
      G__prerun = 1;  /* suppress warning message if file already loaded */
      temp1 = G__loadfile(string + temp);
      if (!keepIfLoaded && temp1 == 1) {
         G__unloadfile(string + temp);
         G__storerewindposition();
         if (G__loadfile(string + temp)) {
            G__prerun = temp2;
            G__pause_return = 1;
            G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
            if (G__security_error) G__cancel_undo_position();
#ifdef G__SECURITY
            *err |= G__security_recover(G__serr);
#endif
            G__UnlockCriticalSection();
            return(ignore);
         }
      }
      G__prerun = temp2;
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
   }
   else if (strncmp("U", com, 1) == 0) {
      /*******************************************************
       * Load a C/C++ source file.
       *******************************************************/
      temp = 0;
      while (isspace(string[temp])) temp++;
      if (string[temp] == '\0') {
         G__fprinterr(G__serr, "Error: no file specified\n");
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      if (G__UNLOADFILE_SUCCESS == G__unloadfile(string + temp))
         G__init_undo();
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
   }
   else if (strncmp("n", com, 1) == 0) {
      /*******************************************************
       * Create New readline file and start dump.
       *******************************************************/
      index = 1;
      while (isspace(command[index]) && command[index] != '\0') index++;
      tempfp = fopen(command + index, "w");
      if (tempfp) {
         fprintf(G__sout, "Create new readline dumpfile %s and start dump\n", command + index);
         G__pushdumpinput(tempfp, 0);
      }
      else {
         fprintf(G__sout, "Can not open file %s\n", command + 1);
      }
   }
   else if (strncmp("y", com, 1) == 0) {
      /*******************************************************
       * Append readline input to file.
       *******************************************************/
      index = 1;
      while (isspace(command[index]) && command[index] != '\0') index++;
      tempfp = fopen(command + index, "a");
      if (tempfp) {
         fprintf(G__sout, "Append readline dump to %s\n", command + index);
         G__pushdumpinput(tempfp, 0);
      }
      else {
         fprintf(G__sout, "Can not open file %s\n", command + 1);
      }
   }
   else if (strncmp("z", com, 1) == 0) {
      /*******************************************************
       * Stop readline dump.
       *******************************************************/
      if (G__dumpreadline[0]) {
         fprintf(G__sout, "Stop readline dump.\n");
         fclose(G__dumpreadline[0]);
         G__popdumpinput();
      }
      else {
         fprintf(G__sout, "No readline dumpfile in the stack\n");
      }
   }
#ifndef G__OLDIMPLEMENTATION1546
   else if (strncmp("save", com, 4) == 0) {
      if (G__emergencycallback)(*G__emergencycallback)();
      else fprintf(G__sout, "!!!No emergency callback\n");
   }
#endif
   else if (strncmp("COPYFLAG", com, 1) == 0) {
      fprintf(G__sout, "set source file copy flag '%s'\n", string);
      G__setcopyflag(atoi(string));
   }
   else if (strncmp("AUTOVAR", com, 1) == 0) {
      fprintf(G__sout, "set automatic variable allocation mode '%s'\n", string);
      G__automaticvar = atoi(string);
   }
   else if (strncmp("help", com, 1) == 0 || strncmp("?", com, 1) == 0) {
      /*******************************************************
       * Help for break prompt command
       *******************************************************/
      G__more_pause((FILE*)NULL, 1);
      G__display_note();
#ifndef G__SMALLOBJECT
#ifdef G__ROOT
      G__more(G__sout, "CINT/ROOT C/C++ interpreter interface.\n");
#else
      G__more(G__sout, "cint (C/C++ interpreter) debugger usage:\n");
#endif
      if (G__INPUTROOTMODE&G__rootmode) {
         G__more(G__sout, "All commands must be preceded by a . (dot), except\n");
         G__more(G__sout, "for the evaluation statement { } and the ?.\n");
         G__more(G__sout, "===========================================================================\n");
      }
#ifndef G__ROOT
      G__more(G__sout, "Dump:        n [file]  : create new readline dumpfile and start dump\n");
      G__more(G__sout, "             y [file]  : append readline dump to [file]\n");
      G__more(G__sout, "             z         : stop readline dump\n");
      G__more(G__sout, "             < [file]  : input redirection from [file](execute command dump)\n");
#endif
      G__more(G__sout, "             > [file]  : output redirection to [file]\n");
      G__more(G__sout, "             2> [file] : error redirection to [file]\n");
      G__more(G__sout, "             >& [file] : output&error redirection to [file]\n");
#ifndef G__ROOT
      G__more(G__sout, "             .         : switch command input mode\n");
#endif
      G__more(G__sout, "Help:        ?         : help\n");
      G__more(G__sout, "             help      : help\n");
      G__more(G__sout, "             /[keyword] : search keyword in help information\n");
#if (!defined(G__ROOT)) && (!defined(G__WIN32))
      G__more(G__sout, "Completion:  [nam][Tab] : complete symbol name start with [nam]\n");
      G__more(G__sout, "             [nam][Tab][Tab] : list up all symbol name start with [nam]]\n");
#endif
      G__more(G__sout, "Shell:       ![shell]  : execute shell command\n");
      G__more(G__sout, "Source:      v <[line]>: view source code <around [line]>\n");
      G__more(G__sout, "             V [stack] : view source code in function call stack\n");
      G__more(G__sout, "             t         : show function call stack\n");
      G__more(G__sout, "             f [file]  : select file to debug\n");
      G__more(G__sout, "             T         : turn on/off trace mode for all source\n");
      G__more(G__sout, "             J [stat]  : Set warning level [0-5]\n");
      G__more(G__sout, "             A [1|0]   : allowing automatic variable on/off\n");
      G__more(G__sout, "             trace <classname> : turn on trace mode for class\n");
      G__more(G__sout, "             deltrace <classname> : turn off trace mode for class\n");
#ifndef G__ROOT
      G__more(G__sout, "             break [classname] : set break point at every [classname] memberfunc\n");
      G__more(G__sout, "             delbreak [classname] : turn off memberfunc break point\n");
#endif
      /* G__more(G__sout,"             D         : toggle source code display mode\n"); */
      G__more(G__sout, "Evaluation:  p [expr]  : evaluate expression (no declaration/loop/condition)\n");
      G__more(G__sout, "Evaluation:  s [expr]  : step into expression (no declaration/loop/condition)\n");
      G__more(G__sout, "Evaluation:  S [expr]  : step over expression (no declaration/loop/condition)\n");
      G__more(G__sout, "             {[statements]} : evaluate statement (any kind)\n");
#ifndef G__ROOT
      G__more(G__sout, "             x [file]  : load [file] and evaluate {statements} in the file\n");
#else
      G__more(G__sout, "             x [file]  : load [file] and execute function [file](w/o extension)\n");
      G__more(G__sout, "             xk [file] : keep [file] if already loaded else load it, and execute function [file](w/o extension)\n");
#endif
      G__more(G__sout, "             X [file]  : load [file] and execute function [file](w/o extension)\n");
      G__more(G__sout, "             Xk [file] : keep [file] it already loaded else load it. and execute function [file](w/o extension)\n");
      G__more(G__sout, "             E <[file]>: open editor and evaluate {statements} in the file\n");
      G__more(G__sout, "Load/Unload: L [file]  : load [file]\n");
      G__more(G__sout, "             Lk [file] : keep [file] if already loaded, else load it\n");
      G__more(G__sout, "             La [file] : reload all files loaded after [file]\n");
      G__more(G__sout, "             U [file]  : unload [file]\n");
      G__more(G__sout, "             C [1|0]   : copy source to $TMPDIR (on/off)\n");
#ifndef G__ROOT
      G__more(G__sout, "             reset     : reset interpreter environment\n");
#endif
      G__more(G__sout, "             undo      : undo previous declarations\n");
      G__more(G__sout, "             lang      : local language (EUC,SJIS,EUROPEAN,UNKNOWN)\n");
      G__more(G__sout, "             preproc   : toggle preprocessing of interpreted files\n");
      G__more(G__sout, "Monitor:     g <[var]> : list global variable\n");
      G__more(G__sout, "             l <[var]> : list local variable\n");
      G__more(G__sout, "             proto <[scope]::>[func] : show function prototype\n");
      G__more(G__sout, "             class <[name]> : show class definition (one level)\n");
      G__more(G__sout, "             Class <[name]> : show class definition (all level)\n");
      G__more(G__sout, "             typedef <name> : show typedefs\n");
      G__more(G__sout, "             function  : show interpreted functions\n");
      G__more(G__sout, "             macro     : show macro functions\n");
      G__more(G__sout, "             template  : show templates\n");
      G__more(G__sout, "             include   : show include paths\n");
      G__more(G__sout, "             file      : show loaded files\n");
      G__more(G__sout, "             where     : show current file position\n");
      G__more(G__sout, "             security  : show security level\n");
      G__more(G__sout, "             refcount  : reference count control on/off\n");
      G__more(G__sout, "             garbage   : show garbage collection buffer\n");
      G__more(G__sout, "             Garbage   : Do garbage collection\n");
      G__more(G__sout, "             cover [file] : save trace coverage\n");
      G__more(G__sout, "             return [val] : return undefined symbol value\n");
      G__more(G__sout, "Run:         S         : step over function/loop\n");
      G__more(G__sout, "             s         : step into function/loop\n");
      G__more(G__sout, "             i         : ignore and step over\n");
      G__more(G__sout, "             c <[line]>: continue <to [line]>\n");
      G__more(G__sout, "             e         : step out from function\n");
      G__more(G__sout, "             f [file]  : select file to debug\n");
      G__more(G__sout, "             b [line]  : set break point\n");
      G__more(G__sout, "             db [line] : delete break point\n");
      G__more(G__sout, "             a [assert]: break only if assertion is true\n");
      G__more(G__sout, "             O [0~4]   : Set bytecode compiler mode\n");
      G__more(G__sout, "             debug     : bytecode status display on/off\n");
#ifdef G__ASM_DBG
      G__more(G__sout, "             asmstep   : bytecode step mode on/off\n");
      G__more(G__sout, "             status    : show bytecode exec flags\n");
#endif
      G__more(G__sout, "             dasm      : disassembler\n");
#endif
      G__more(G__sout, "             except    : toggle exception catching\n");
      G__more(G__sout, "Quit:        q         : quit cint\n");
      G__more(G__sout, "             qqq       : quit cint - mandatory\n");
      G__more(G__sout, "             qqqqq     : exit process immediately\n");
      G__more(G__sout, "             qqqqqqq   : abort process\n");
#ifndef G__OLDIMPLEMENTATION1546
      G__more(G__sout, "             save      : call emergency routine to save important data\n");
#endif
   }
   else if (strncmp("/", com, 1) == 0) {
      G__more_pause((FILE*)NULL, 1);
      /*******************************************************
       * Display keyword help information
       *******************************************************/
      do {
#if defined(G__OLDIMPLEMENTATION2092_YET)
         G__temp = tmpfile();
         if (!G__temp) {
            G__tmpnam(tname); /* not used anymore */
            G__temp = fopen(tname, "w");
            istmpnam = 1;
         }
#elif !defined(G__OLDIMPLEMENTATION1917)
         // coverity[secure_temp]: we don't care about predictable names.
         G__temp = tmpfile();
#else
        G__tmpnam(tname); /* not used anymore */
        G__temp=fopen(tname,"w");
#endif
      }
      while ((FILE*)NULL == G__temp && G__setTMPDIR(tname));
      if (G__temp) {
         G__cintrevision(G__temp);
         G__list_sut(G__temp);
         /* compiled and interpreterd objects */
         static char emptystring[1] = {0};
         G__display_class(G__temp, emptystring, 0, 0);
         G__display_typedef(G__temp, "", 0);
         G__display_string(G__temp);
         G__display_template(G__temp, " ");
         G__display_macro(G__temp, "");
         G__display_files(G__temp);
         G__listfunc(G__temp, G__PUBLIC_PROTECTED_PRIVATE, (char*)NULL
                     , (struct G__ifunc_table*)NULL);
         G__varmonitor(G__temp, &G__global, "", "", 0);
#if defined(G__OLDIMPLEMENTATION2092_YET)
         if (istmpnam) fclose(G__temp);
#elif !defined(G__OLDIMPLEMENTATION1917)
#else
        fclose(G__temp);
#endif

         if (command[strlen(command)-1] == ' ') command[strlen(command)-1] = '\0';
#if defined(G__OLDIMPLEMENTATION2092_YET)
         if (!istmpnam) {
            G__display_keyword(G__sout, command + 1, G__temp);
            fclose(G__temp);
         }
         else {
            G__display_keyword(G__sout, command + 1, tname);
            remove(tname);
         }
#elif !defined(G__OLDIMPLEMENTATION1917)
        G__display_keyword(G__sout,command+1,G__temp);
        fclose(G__temp);
#else
        G__display_keyword(G__sout,command+1,tname);
        remove(tname);
#endif
      }
      else {
         G__fprinterr(G__serr, "Error: Tempfile G__temp can not open\n");
      }
   }
   else if (strncmp("$", com, 1) == 0) {
      /*******************************************************
       * Execute shell command
       *******************************************************/
      G__FastAllocString combuf(strlen(string) + 30);
      combuf.Format("sh -I -c %s", string);
      int ret = system(combuf);
      if (rslt) G__letint(rslt, 'i', ret);
   }
   else if (strncmp("!", com, 1) == 0) {
      /*******************************************************
       * Execute shell command
       *******************************************************/
      int ret = system(string);
      if (rslt) G__letint(rslt, 'i', ret);
   }
   else if (strncmp("a", com, 1) == 0) {
      /*******************************************************
       * Set assertion for break
       *******************************************************/
      command[0] = ' ';
      if (command[1] == '\0') {
         G__assertion[0] = '\0';
         fprintf(G__sout, "Break assertion is deleted\n");
      }
      else {
         G__strlcpy(G__assertion, command + 1, sizeof(G__assertion));
         fprintf(G__sout, "Break only if (%s) is true\n"
                 , G__assertion);
      }
   }
   else if (strncmp("G", com, 1) == 0) {
      G__more_pause((FILE*)NULL, 1);
      index = 1;
      while (isspace(command[index])) index++;
      temp = index;
      while (command[temp] && (!isspace(command[temp]))) temp++;
      command[temp] = '\0';
      G__debugvariable(G__sout, &G__global, command + index);
   }
   else if (strncmp("g", com, 1) == 0 || strncmp("G", com, 1) == 0 || strncmp("l", com, 1) == 0) {
      /*******************************************************
       * Monitor global variables
       * Monitor local variables
       *******************************************************/
      G__more_pause((FILE*)NULL, 1);
      index = 1;
      while (isspace(command[index])) index++;
      temp = index;
      while (command[temp] && (!isspace(command[temp]))) temp++;
      command[temp] = '\0';
#ifdef G__ASM
      G__STORE_ASMENV;
#endif
      store_var_type = G__var_type;
      G__var_type = 'p';
      G__SET_TEMPENV;
      if ('g' == command[0]) G__varmonitor(G__sout, &G__global, command + index, "", 0);
      else if ('G' == command[0]) G__varmonitor(G__sout, &G__global, command + index, "", 0);
      else if (G__cintv6) {
         if (G__bc_setdebugview(temp1, &view))
            G__varmonitor(G__sout, view.var_local, command + index, "", view.localmem);
      }
      else G__varmonitor(G__sout, G__p_local, command + index, "", 0);
      G__RESET_TEMPENV;
#ifdef G__ASM
      G__RECOVER_ASMENV;
#endif
      G__var_type = store_var_type;
   }
   else if (strncmp("t", com, 1) == 0) {
      /*******************************************************
       * show function call stack
       *******************************************************/
      if (G__cintv6) G__bc_showstack(G__sout);
      else G__showstack(G__sout);
   }
   else if (strncmp("T", com, 1) == 0) {
      /*******************************************************
       * Toggle trace mode
       *******************************************************/
      if (G__istrace == 0) {
         G__istrace = 1;
#ifdef G__ASM_DBG
         if (string[0]) G__istrace = atoi(string);
#endif
         fprintf(G__sout, "\nTrace mode on %d\n", G__istrace);
      }
      else {
         fprintf(G__sout, "\nTrace mode off\n");
         G__istrace = 0;
      }
      G__debug = G__istrace;
      G__setdebugcond();
   }
   else if (strncmp("D", com, 1) == 0) {
      /*******************************************************
       * Toggle display mode
       *******************************************************/
      if (G__breakdisp == 0) {
         G__breakdisp = 1;
         fprintf(G__sout, "\nDisplay mode on\n");
      }
      else {
         fprintf(G__sout, "\nDisplay mode off\n");
         G__breakdisp = 0;
      }
   }
   else if (strncmp("db", com, 2) == 0) {
      /*******************************************************
       * Delete break point
       *******************************************************/
      if (1 < G__findposition(string, &view.file, &line_number, &filenum)
            && filenum >= 0 && line_number >= 0
         ) {
         G__srcfile[filenum].breakpoint[line_number] &= G__NOBREAK;
         fprintf(G__sout, "Break point line %d %s deleted\n"
                 , line_number, G__srcfile[filenum].filename);
         G__step = 0;
         G__charstep = 0;
         G__setdebugcond();
      }
      else {
         G__fprinterr(G__serr, "can not determine where to delete break point\n");
      }
   }
   else if (strncmp("break", com, 1) == 0) {
      /*******************************************************
       * Set break point
       *******************************************************/
      if (1 < G__findposition(string, &view.file, &line_number, &filenum) &&
            filenum >= 0 && line_number >= 0 &&
            G__srcfile[filenum].breakpoint && G__srcfile[filenum].maxline
         ) {
         fprintf(G__sout, "Break point set to line %d %s\n"
                 , line_number, G__srcfile[filenum].filename);
         G__srcfile[filenum].breakpoint[line_number] |= G__BREAK;
         G__step = 0;
         G__charstep = 0;
         G__setdebugcond();
      }
      else if (0 == G__srcfile[filenum].maxline) {
         if ((FILE*)NULL == G__srcfile[filenum].fp) {
            G__fprinterr(G__serr, "Can not put break point in included file\n");
         }
         else {
            G__fprinterr(G__serr, "Setting break point suspended\n");
            temp = 0;
            while (isspace(string[temp])) ++temp;
            if ('\0' == string[temp]) {
               G__snprintf(G__breakline, sizeof(G__breakline), "%d", G__ifile.line_number);
               G__strlcpy(G__breakfile, G__srcfile[G__ifile.filenum].filename,sizeof(G__breakfile));
            }
            else {
               G__strlcpy(G__breakline, string + temp,sizeof(G__breakline));
            }
         }
      }
      else {
         G__fprinterr(G__serr, "Can not determine where to put break point\n");
      }
   }
   else if (strncmp("continue", com, 1) == 0) {
      /*******************************************************
       * Continue
       *******************************************************/
      G__stepover = 0;
      G__step = 0;
      G__charstep = 0;
      G__steptrace = 0;
      G__setdebugcond();
      if (strlen(string) == 0) {
         G__pause_return = 1;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      else if (1 < G__findposition(string, &view.file, &line_number, &filenum) && filenum >= 0 && line_number >= 0) {
         G__srcfile[filenum].breakpoint[line_number] |= G__CONTUNTIL;
         G__pause_return = 1;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      else {
         G__fprinterr(G__serr, "can not determine where to stop\n");
      }
   }
   else if (strncmp("f", com, 1) == 0) {
      syscom = string;
      command = syscom;
      /*******************************************************
       * Set view file
       *******************************************************/
      for (temp = 0;(long)temp < G__nfile;temp++) {
         if (G__matchfilename(temp, string)) break;
      }
      if (temp >= (size_t)G__nfile) {
         G__fprinterr(G__serr, "filename %s not loaded\n", string);
      }
      else {
         view.file.filenum = temp;
         view.file.fp = G__srcfile[temp].fp;
         G__strlcpy(view.file.name, G__srcfile[temp].filename,G__MAXFILENAME);
         view.file.line_number = 1;
         G__pr(G__sout, view.file);
      }
   }
   else if (strncmp("+", com, 1) == 0 || strncmp("-", com, 1) == 0) {
      temp = atoi(command);
      command.Format("%d", view.file.line_number + temp);
      goto vcommand;
   }
   else if (strncmp("proto", com, 3) == 0) {
      G__more_pause((FILE*)NULL, 1);
      G__display_proto(G__sout, string);
   }
   else if (strncmp("view", com, 1) == 0) {
      syscom = string;
      command = syscom;
vcommand:
      /*******************************************************
       * Display source code
       *******************************************************/
      if (0 < G__findposition(command, &view.file, &line_number, &filenum) && filenum >= 0
            && line_number >= 0
         ) {
         view.file.filenum = filenum;
         view.file.fp = G__srcfile[filenum].fp;
         G__strlcpy(view.file.name, G__srcfile[filenum].filename,G__MAXFILENAME);
         view.file.line_number = line_number;
         G__pr(G__sout, view.file);
      }
      else {
         fprintf(G__sout, "No proper file view.file. Can not display source! Use 'f [file]' command\n");
      }
   }
   else if (strncmp("View", com, 1) == 0) {
      /*******************************************************
       * show source code in stack
       *******************************************************/
      temp1 = atoi(command + 1);
      temp = 0;
      local = G__p_local;
      if (G__cintv6) {
         if (G__bc_setdebugview(temp1, &view)) G__pr(G__sout, view.file);
      }
      else {
         while (local && temp < temp1 - 1) {
            ++temp;
            local = local->prev_local;
         }
         if (0 == temp1) {
            view.file = G__ifile;
            view.var_local = G__p_local;
            view.struct_offset = G__store_struct_offset;
            view.tagnum = G__tagnum;
            view.exec_memberfunc = G__exec_memberfunc;
            G__pr(G__sout, view.file);
         }
         else if (local && local->prev_local) {
            view.file.filenum = local->prev_filenum ;
            G__strlcpy(view.file.name, G__srcfile[view.file.filenum].filename,G__MAXFILENAME);
            view.file.fp = G__srcfile[view.file.filenum].fp;
            view.file.line_number = local->prev_line_number;
            view.var_local = local->prev_local;
            view.struct_offset = view.var_local->struct_offset;
            view.tagnum = view.var_local->tagnum;
            view.exec_memberfunc = view.var_local->exec_memberfunc;
            G__pr(G__sout, view.file);
         }
         else {
            fprintf(G__sout, "Stack isn't that deep\n");
         }
      }
   }
   else if (strncmp("ignore", com, 1) == 0) {
      /*******************************************************
       * Ignore current statement and step to the next
       *******************************************************/
      G__stepover = 0;
      ignore = G__PAUSE_IGNORE;
      G__pause_return = 1;
      G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                          , keyword, pipefile);
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
      G__UnlockCriticalSection();
      return(ignore);
   }
   else if (strncmp("S", com, 1) == 0 && 0 == (*stringb)) {
      /*****************************************************
       * Step over
       *****************************************************/
      G__stepover = 3;
      ignore = G__PAUSE_STEPOVER;
      G__step = 1;
      G__charstep = 0;
      G__setdebugcond();
      G__pause_return = 1;
      G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
      G__UnlockCriticalSection();
      return(ignore);
   }
   else if (strncmp("s", com, 1) == 0 && 0 == (*stringb)) {
      /*****************************************************
       * Step into
       *****************************************************/
      G__stepover = 0;
      ignore = G__PAUSE_NORMAL;
      G__step = 1;
      G__charstep = 0;
      G__setdebugcond();
      G__pause_return = 1;
      G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
      G__UnlockCriticalSection();
      return(ignore);
   }
   else if (strncmp("e", com, 1) == 0 && 'x' != com[1]) {
      /*****************************************************
       * Step out 
       *****************************************************/
      G__stepover = 0;
      G__step = 0;
      G__charstep = 0;
      G__break_exit_func = 1;
      G__setdebugcond();
      G__pause_return = 1;
      G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                          , keyword, pipefile);
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
      G__UnlockCriticalSection();
      return(ignore);
   }
   else if (strncmp(command, "qqqqqqq", 7) == 0 || strncmp(command, "QQQQQQQ", 7) == 0) {
      abort();
   }
   else if (strncmp(command, "qqqqq", 5) == 0 || strncmp(command, "QQQQQ", 5) == 0) {
      G__fprinterr(G__serr, "  Bye... (try 'qqqqqqq' if still running)\n");
      exit(EXIT_FAILURE);
   }
   else if (strncmp(command, "qqq", 3) == 0 || strncmp(command, "QQQ", 3) == 0) {
      fprintf(G__sout, "*** Process will be killed ***\n");
      command = G__input("Are you sure(y/Y/n)? ");
      if (command[0] == 'Y') {
         G__fprinterr(G__serr, "  Bye... (try 'qqqqq' if still running)\n");
         exit(EXIT_FAILURE);
      }
      else if (command[0] == 'y') {
         G__fprinterr(G__serr, "  Bye... (try 'qqqqq' if still running)\n");
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                             , keyword, pipefile);
         G__closemfp();
         G__mfp = NULL;
         G__close_inputfiles();
         exit(EXIT_SUCCESS);
      }
      else {
         G__pause_return = 1;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(0);
      }
   }
   else if (strncmp(command, "qq", 2) == 0 || strncmp(command, "QQ", 2) == 0) {
      G__return = G__RETURN_EXIT1;
#ifndef G__ROOT
      fprintf(G__sout, "Exit current interpretation.\n");
#endif
      /*****************************************
       * return 2 so that user can utilize 
       * while(G__pause()==0) ;
       *****************************************/
      G__pause_return = 1;
      G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
      ignore = 2;
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
      G__UnlockCriticalSection();
      return(ignore);
   }
   else if (strncmp("q", com, 1) == 0 || strncmp("exit", com, 4) == 0) {
      if (G__doingconstruction) {
         G__fprinterr(G__serr, "Use 'qqq' when you quit in the middle of object construction (%d)\n", G__doingconstruction);
         G__pause_return = 1;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      else {
         G__fprinterr(G__serr, "  Bye... (try 'qqq' if still running)\n");
      }

      G__stepover = 0;
      G__step = 0;
      G__break = 0;
      G__steptrace = 0;
      G__setdebugcond();
      /*******************************************************
       * Quit session and stop process
       *******************************************************/
      if (G__othermain) {
         /*******************************************************
          * QQ : exit all interpretation
          * Q  : exit current interpretation
          *******************************************************/
         if (command[1] == 'q') {
            G__return = G__RETURN_EXIT2;
            fprintf(G__sout, "Exit current command file.\n");
         }
         else {
            G__return = G__RETURN_EXIT1;
#ifndef G__ROOT
            fprintf(G__sout, "Exit current interpretation.\n");
#endif
         }
         /*****************************************
         * return 2 so that user can utilize 
         * while(G__pause()==0) ;
         *****************************************/
         G__pause_return = 1;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
         ignore = 2;
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      else {
         /*******************************************************
          * Q  : exit process
          *******************************************************/
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__DEBUG
         G__exit(EXIT_SUCCESS);
#else
         G__close_inputfiles();
         exit(EXIT_SUCCESS);
#endif
      }
   }
   else if (strncmp("E", com, 1) == 0 || strncmp("x", com, 1) == 0) {
      /*******************************************************
       * Open tempfile to edit 
       *******************************************************/
      temp = 1;
      while (isspace(command[temp])) temp++;
      if (com[0] == 'E') {
         if (command[temp] == '\0') {
            if ('\0' == G__tempc[0]) G__tmpnam(G__tempc); /* E command, rare case */
            syscom.Format("%s %s", editor(), G__tempc);
            int ret = system(syscom);
            if (rslt) G__letint(rslt, 'i', ret);
            syscom = G__tempc;
         }
         else {
            syscom.Format("%s %s", editor(), command + temp);
            int ret = system(syscom);
            if (rslt) G__letint(rslt, 'i', ret);
            syscom = command + temp;
         }
      }
      else {
         if (command[temp] == '\0') {
            G__fprinterr(G__serr, "Error: no file specified\n");
            G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
            *err |= G__security_recover(G__serr);
#endif
            G__UnlockCriticalSection();
            return(ignore);
         }
         syscom = command + temp;
      }
      /*******************************************************
       * Execute temp file
       *******************************************************/
      G__store_undo_position();
#ifdef G__ROOT
      if (strstr(syscom, "rootlogon.")) G__init_undo();
#endif
      {
         struct G__store_env store;
         G__SET_TEMPENV;
         buf = G__exec_tempfile(syscom);
         if (rslt) *rslt = buf;
         if (G__ifile.filenum >= 0)
            G__security = G__srcfile[G__ifile.filenum].security;
         else
            G__security = G__SECURE_LEVEL0;
#ifndef G__ROOT
         G__in_pause = 1;
         G__valuemonitor(buf, syscom);
         G__in_pause = 0;
#ifndef G__OLDIMPLEMENTATION1259
         if (buf.isconst&(G__CONSTVAR | G__CONSTFUNC)) {
            G__FastAllocString tmp(G__ONELINE);
            tmp.Format("(const %s", syscom + 1);
            syscom = tmp;
         }
         if (buf.isconst&G__PCONSTVAR) {
            char *ptmp = strchr(syscom, ')');
            G__FastAllocString tmp2(ptmp);
            syscom.Replace((ptmp-syscom()), "const");
            syscom += tmp2;
         }
#endif
         if (buf.type && 0 == G__atevaluate(buf)) fprintf(G__sout, "%s\n", syscom());
#endif
         G__RESET_TEMPENV;
      }
      if (G__security_error) G__cancel_undo_position();
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif

      if (G__return != G__RETURN_NON) {
         G__pause_return = 1;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
   }
   else if (strncmp("{", com, 1) == 0) {
      /*******************************************************
       * Evaluate sequencial statements
       *******************************************************/
#ifndef G__OLDIMPLEMENTATION1774
multi_line_command:
#endif
      if (*more == 0) {
         G__create_input_tmpfile(ftemp);
         if (!ftemp.fp) {
            do {
               G__tmpnam(tname); /* not used anymore */
               ftemp.fp = fopen(tname, "w");
            }
            while ((FILE*)NULL == ftemp.fp && G__setTMPDIR(tname));
            istmpnam = 1;
         }
      }
      else {
         com = command + 1;
         ftemp = G__multi_line_temp;
         G__multi_line_temp.fp = 0;
#ifndef G__OLDIMPLEMENTATION1774
         if ('@' == com[0]) {
            if (ftemp.fp) fclose(ftemp.fp);
            ftemp.fp = (FILE*)NULL;
            *more = 0;
            G__fprinterr(G__serr, "!!!command line input aborted!!!\n");
            G__UnlockCriticalSection();
            return(0);
         }
#endif
      }
      if (!ftemp.fp) {
         G__fprinterr(G__serr, "Error: could not create file %s\n", tname);
      }
      else {
         temp = *more;
         temp1 = 0;
         double_quote = 0;
         single_quote = 0;
         while (com[temp1] != '\0') {
            switch (com[temp1]) {
               case '"':
                  if (single_quote == 0) double_quote ^= 1;
                  break;
               case '\'':
                  if (double_quote == 0) single_quote ^= 1;
                  break;
               case '{':
#ifndef G__OLDIMPLEMENTATION1774
               case '(':
               case '[':
#endif
                  if ((single_quote == 0) && (double_quote == 0)) temp++;
                  break;
               case '}':
#ifndef G__OLDIMPLEMENTATION1774
               case ')':
               case ']':
#endif
                  if ((single_quote == 0) && (double_quote == 0)) temp--;
                  break;
               case '\\':
                  ++temp1;
                  break;
            }
            ++temp1;
         }
         if (temp > 0) {
            fprintf(ftemp.fp, "%s\n", com);
           G__multi_line_temp = ftemp;

#ifndef G__OLDIMPLEMENTATION1774
           strcpy(prompt, "end with '}', '@':abort > "); // Legacy, we don't known the input buffer size.
#else
           strcpy(prompt, "end with '}'> "); // Legacy, we don't known the input buffer size.
#endif
            *more = temp;
            G__pause_return = 0;
#ifdef G__SECURITY
            *err |= G__security_recover(G__serr);
#endif
            G__UnlockCriticalSection();
            return(0);
         }
         else {
            string = strrchr(com, '}');
            if (string) {
               G__redirectoutput(string
                                 , &store_stdout, &store_stderr, &store_stdin, 0
                                 , keyword, pipefile);
               *string = '\0';
            }
            fprintf(ftemp.fp, "%s\n", com);
            fprintf(ftemp.fp, "}\n");
            *more = 0;
            prompt[0] = '\0';
         }
         if (istmpnam) fclose(ftemp.fp);
         else G__remove_input_tmpfile(ftemp);

         /*******************************************************
          * Execute temp file
          *******************************************************/
         G__store_undo_position();
         ++G__templevel;
         G__more_pause((FILE*)NULL, 1);
         {
            struct G__store_env store;
            G__SET_TEMPENV;
            if (!istmpnam) {
               G__command_eval = 1 ;
               buf = G__exec_tempfile_fp(ftemp.fp);
               if (G__security_error && G__pautoloading && (*G__pautoloading)(com)) {
                  buf = G__exec_tempfile_fp(ftemp.fp);
               }
               if (rslt) *rslt = buf;
               if (ftemp.fp) fclose(ftemp.fp);
               ftemp.fp = (FILE*)NULL;
            }
            else {
               sname = tname;
               G__command_eval = 1 ;
               buf = G__exec_tempfile(sname);
               if (G__security_error && G__pautoloading && (*G__pautoloading)(com)) {
                  buf = G__exec_tempfile(sname);
               }
               if (rslt) *rslt = buf;
               remove(sname);
            }
            G__in_pause = 1;
            G__valuemonitor(buf, syscom);
            G__in_pause = 0;
#ifndef G__OLDIMPLEMENTATION1259
            if (buf.isconst&(G__CONSTVAR | G__CONSTFUNC)) {
               G__FastAllocString tmp(G__LONGLINE);
               tmp.Format("(const %s", syscom + 1);
               syscom = tmp;
            }
            if (buf.isconst&G__PCONSTVAR) {
               char *ptmp = strchr(syscom, ')');
               G__FastAllocString tmp2(ptmp);
               syscom.Replace((ptmp-syscom()), "const");
               syscom += tmp2;
            }
#endif
            G__RESET_TEMPENV;
         }
         if (-1 == G__func_now) {
            store.var_local = G__p_local;
            G__p_local = 0;
         }
         if (buf.type && 0 == G__atevaluate(buf)
               && !noprintflag
             ) fprintf(G__sout, "%s\n", syscom());
         noprintflag = 0;
         G__command_eval = 0 ;
         G__free_tempobject();
         --G__templevel;
         if (-1 == G__func_now) G__p_local = store.var_local;
         if (G__security_error) G__cancel_undo_position();
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif

         if (G__return != G__RETURN_NON) {
            G__pause_return = 1;
            G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                                , keyword, pipefile);
#ifdef G__SECURITY
            *err |= G__security_recover(G__serr);
#endif
            G__UnlockCriticalSection();
            return(ignore);
         }

      }
   }
   else if (strncmp("p", com, 1) == 0 || strncmp("s", com, 1) == 0 || strncmp("S", com, 1) == 0) {
      G__STORE_EVALENV;
      /*******************************************************
       * Evaluate statement
       *******************************************************/
      G__more_pause((FILE*)NULL, 1);
      syscom = string;
      command = syscom;
      if (strlen(command) > 0) {
         G__redirectoutput(command, &store_stdout, &store_stderr, &store_stdin, 1, keyword, pipefile);
#ifdef G__ASM
         G__STORE_ASMENV;
#endif
         G__command_eval = 1 ;
         store_var_type = G__var_type;
         G__var_type = 'p';

         single_quote = 0;
         double_quote = 0;
         evalbase = command;
         temp = 1;
         while (temp) {
            evalbase++;
            switch (*evalbase) {
               case '"':
                  if (single_quote == 0) {
                     double_quote ^= 1;
                  }
                  break;
               case '\'':
                  if (double_quote == 0) {
                     single_quote ^= 1;
                  }
                  break;
               case '\\':
                  if (double_quote == 0 && single_quote == 0) {
                     temp = 0;
                  }
                  else {
                     evalbase++;
                  }
                  break;
               case '\0':
                  if (double_quote == 0 && single_quote == 0) {
                     temp = 0;
                     evalbase = (char *)NULL;
                  }
                  break;
            }
         }
         G__SET_TEMPENV;
         if (evalbase) {
            *evalbase = '\0';
            temp1 = 1;
            num = 0;
            if (isdigit(evalbase[temp1])) {
               while (isdigit(evalbase[temp1])) {
                  syscom[temp1-1] = evalbase[temp1];
                  ++temp1;
               }
               syscom[temp1-1] = '\0';
               num = atoi(syscom);
            }
            switch (tolower(evalbase[temp1])) {
               case 'v':
                  base = 100;
                  break;
               case 'x':
               case 'h':
                  base = 16;
                  break;
               case 'b':
               case 'z':
                  base = 2;
                  break;
               case 'o':
                  base = 8;
                  break;
               case 'd':
                  base = 10;
                  break;
               case 'n':
                  base = 0;
                  break;
            }
            if (num <= 0) {
               buf = G__calc_internal(command);
               if (rslt) *rslt = buf;
               G__in_pause = 1;
               if (base == 0) {
                  G__valuemonitor(buf, syscom);
               }
               else if (base == 100) {
                  syscom.Format("{d=%g i=%ld,reftype=%d} type=%c,tag=%d,type=%d,ref=%lx,isconst=%d"
                          , buf.obj.d, buf.obj.i, buf.obj.reftype.reftype
                          , buf.type, buf.tagnum, buf.typenum, buf.ref, buf.isconst);
               }
               else {
                  G__getbase(buf.obj.i , base , 0, evalresult);
                  syscom.Format("0%c%s" , evalbase[temp1] , evalresult());
               }
               G__in_pause = 0;
               fprintf(G__sout, "%s\n", syscom());
            }
            else {
               for (temp = 0;temp < num;temp++) {
                  if (temp % 2 == 0) {
                     syscom.Format("&%s+%d" , command + 1 , temp);
                     buf = G__calc_internal(syscom);
                     if (rslt) *rslt = buf;
                     fprintf(G__sout, "\n0x%lx: " , buf.obj.i);
                  }
                  syscom.Format("*(&%s+%d)" , command + 1 , temp);
                  buf = G__calc_internal(syscom);
                  if (rslt) *rslt = buf;
                  G__in_pause = 1;
                  if (base == 0) {
                     G__valuemonitor(buf, syscom);
                  }
                  else {
                     G__getbase(buf.obj.i, base , 0, evalresult);
                     syscom.Format("0%c%s" , evalbase[temp1] , evalresult());
                  }
                  G__in_pause = 0;
                  fprintf(G__sout, "%30s ", syscom());
               }
               fprintf(G__sout, "\n");
            }
         }
         else {
            buf = G__calc_internal(command);
            if (G__security_error && G__pautoloading && (*G__pautoloading)(com)) {
               buf = G__calc_internal(command);
            }
            if (rslt) *rslt = buf;
            if ((char*)NULL == strstr(command, "G__stepmode(")) G__step = store_step;
            G__in_pause = 1;
            G__valuemonitor(buf, syscom);
            G__in_pause = 0;
#ifndef G__OLDIMPLEMENTATION1259
            if (buf.isconst&(G__CONSTVAR | G__CONSTFUNC)) {
               G__FastAllocString tmp(G__ONELINE);
               tmp.Format("(const %s", syscom + 1);
               syscom = tmp;
            }
            if (buf.isconst&G__PCONSTVAR) {
               char *ptmp = strchr(syscom, ')');
               G__FastAllocString tmp2(ptmp);
               syscom.Replace((ptmp-syscom()), "const");
               syscom += tmp2;
            }
#endif
         }
         G__RESET_TEMPENV;

#ifdef G__ASM
         G__RECOVER_ASMENV;
#endif
         G__var_type = store_var_type;
         if (buf.type && 0 == G__atevaluate(buf)) {
            fprintf(G__sout, "%s\n", syscom());
            if ('u' == buf.type && buf.obj.i) {
               G__objectmonitor(G__sout, buf.obj.i, buf.tagnum, "");
            }
         }
         G__command_eval = 0 ;
         G__free_tempobject();
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         if (G__return != G__RETURN_NON) {
            G__pause_return = 1;
            G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                                , keyword, pipefile);
            G__RESTORE_EVALENV;
#ifdef G__SECURITY
            *err |= G__security_recover(G__serr);
#endif
            G__UnlockCriticalSection();
            return(ignore);
         }
      }
      else {
         ignore = G__stepover;
         G__pause_return = 1;
         G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                             , keyword, pipefile);
         G__RESTORE_EVALENV;
#ifdef G__SECURITY
         *err |= G__security_recover(G__serr);
#endif
         G__UnlockCriticalSection();
         return(ignore);
      }
      G__RESTORE_EVALENV;
   }
   else if (strcmp(com, "P") == 0) { /* Dummy pause when reading from dump file */
      ignore = G__PAUSE_NORMAL;
      G__pause_return = 0;
      G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                          , keyword, pipefile);
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
      G__UnlockCriticalSection();
      return(ignore);
   }
   else if (strcmp(com, "") == 0) {
      ignore = G__stepover;
      G__pause_return = 1;
      G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin
                          , keyword, pipefile);
#ifdef G__SECURITY
      *err |= G__security_recover(G__serr);
#endif
      G__UnlockCriticalSection();
      return(ignore);
   }
   else {
      G__fprinterr(G__serr, "Unknown interpreter command '%s'\n", com);
   }
   G__unredirectoutput(&store_stdout, &store_stderr, &store_stdin, keyword, pipefile);
#ifdef G__SECURITY
   *err |= G__security_recover(G__serr);
#endif
   G__UnlockCriticalSection();
   return(ignore); /* never happens, avoiding lint error */
}

/**************************************************************************
* G__setaccess
**************************************************************************/
int G__setaccess(char *statement,int iout)
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

/**************************************************************************
* G__set_smartunload()
**************************************************************************/
void G__set_smartunload(int smartunload)
{
  G__do_smart_unload = smartunload;
}

} /* extern "C" */

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
