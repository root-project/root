/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file parse.c
 ************************************************************************
 * Description:
 *  Cint parser functions
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "configcint.h"
#include "Dict.h"
#include "bc_exec.h"
#include <deque>
#include <stack>

using namespace Cint::Internal;

//______________________________________________________________________________
static const int G__IFDEF_NORMAL = 1;
static const int G__IFDEF_EXTERNBLOCK = 2;
static const int G__IFDEF_ENDBLOCK = 4;

static int G__externblock_iscpp = 0;

//______________________________________________________________________________
static const int G__NONBLOCK = 0;
static const int G__IFSWITCH = 1;
static const int G__DOWHILE = 8;

static int G__ifswitch = G__NONBLOCK;

#ifdef G__WIN32
static void G__toUniquePath(char* s);
#endif // G__WIN32
static int G__setline(char* statement, int c, int* piout);
static int G__pp_ifdefextern(char* temp);
static void G__pp_undef();
static int G__exec_try(char* statement);
static int G__ignore_catch();
static int G__exec_throw(char* statement);
static int G__exec_function(char* statement, int* pc, int* piout, int* plargestep, G__value* presult);
static struct G__breakcontinue_list* G__alloc_breakcontinue_list();
static void G__store_breakcontinue_list(int destination, int breakcontinue);
static void G__free_breakcontinue_list(G__breakcontinue_list* pbreakcontinue);
static void G__set_breakcontinue_destination(int break_dest, int continue_dest, G__breakcontinue_list* pbreakcontinue);
static G__value G__exec_switch();
static G__value G__exec_switch_case(char* casepara);
static G__value G__exec_if();
static G__value G__exec_else_if();
static G__value G__exec_do();
static G__value G__exec_for();
static G__value G__exec_while();
static G__value G__exec_loop(char* forinit, char* condition, int naction, char** foraction);
static G__value G__return_value(const char* statement);
static int G__search_gotolabel(char* label, fpos_t* pfpos, int line, int* pmparen);
static int G__label_access_scope(char* statement, int* piout, int* pspaceflag, int mparen);
static int G__IsFundamentalDecl();
static void G__unsignedintegral();
static void G__externignore();
static void G__parse_friend();
static int G__keyword_anytime_5(char* statement);
static int G__keyword_anytime_6(char* statement);
static int G__keyword_anytime_7(char* statement);
static int G__keyword_anytime_8(char* statement);

#ifndef G__SECURITY
/**************************************************************************
* G__DEFVAR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFVAR(TYPE)                            \
         G__var_type=TYPE + G__unsigned;           \
         G__define_var(-1,::Reflex::Type()); \
         spaceflag = -1;                           \
         iout=0;                                   \
         if(mparen==0) return(G__null)
/**************************************************************************
* G__DEFREFVAR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFREFVAR(TYPE)                         \
         G__var_type=TYPE + G__unsigned;           \
         G__reftype=G__PARAREFERENCE;              \
         G__define_var(-1,::Reflex::Type()); \
         G__reftype=G__PARANORMAL;                 \
         spaceflag = -1;                           \
         iout=0;                                   \
         if(mparen==0) return(G__null)
/**************************************************************************
* G__DEFSTR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFSTR(STRTYPE)                      \
         G__var_type='u';                       \
         G__define_struct(STRTYPE);             \
         spaceflag = -1;                        \
         iout=0;                                \
         if(mparen==0) return(G__null)
#else
/**************************************************************************
* G__DEFVAR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFVAR(TYPE)                                     \
         G__var_type=TYPE + G__unsigned;                    \
         G__define_var(-1,::Reflex::Type());                \
         spaceflag = -1; /* Flag that any following whitespace does not trigger any semantic action. */ \
         iout=0;         /* Reset the statement buffer. */  \
         if(!*mparen||G__return>G__RETURN_NORMAL) return(G__null)
/**************************************************************************
* G__DEFREFVAR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFREFVAR(TYPE)                         \
         G__var_type=TYPE + G__unsigned;           \
         G__reftype=G__PARAREFERENCE;              \
         G__define_var(-1,::Reflex::Type()); \
         G__reftype=G__PARANORMAL;                 \
         spaceflag = -1;                           \
         iout=0;                                   \
         if(!*mparen||G__return>G__RETURN_NORMAL) return(G__null)
/**************************************************************************
* G__DEFSTR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFSTR(STRTYPE)                      \
         G__var_type='u';                       \
         G__define_struct(STRTYPE);             \
         spaceflag = -1;                        \
         iout=0;                                \
         if(!*mparen||G__return>G__RETURN_NORMAL) return(G__null)
#endif


/***********************************************************************
* switch statement jump buffer
***********************************************************************/
static int G__prevcase=0; // Communication between G__exec_switch() and G__exec_statement()

#ifdef G__WIN32
/***********************************************************************
* G__toUniquePath
***********************************************************************/
static void G__toUniquePath(char *s)
{
   // -- FIXME: Describe this function!
   if (!s) {
      return;
   }
   char* d = (char*) malloc(strlen(s) + 1);
   int j = 0;
   for (int i = 0; s[i]; ++i) {
      d[j] = s[i];
      if (!i || (s[i] != '\\') || (s[i-1] != '\\')) {
         ++j;
      }
   }
   d[j] = 0;
   strcpy(s, d);
   free(d);
}
#endif

//______________________________________________________________________________
static int G__setline(char* statement, int c, int* piout)
{
   // -- FIXME: Describe this function!
   if ((c != '\n') && (c != '\r')) {
      c = G__fgetname(statement + 1, "\n\r");
      //
      //
      if (!isdigit(statement[1])) {
         // -- We have #define or #if or #ifdef or #ifndef.
         *piout = strlen(statement);
         return 1;
      }
      else {
         // -- We have #<digit>.
         if ((c == '\n') || (c == '\r')) {
            // -- We have #<line>.
            G__ifile.line_number = atoi(statement + 1);
         }
         else {
            // -- We have #<line> ...".
            G__ifile.line_number = atoi(statement + 1) - 1;
            c = G__fgetname(statement, "\n\r");
            if (statement[0] == '"') {
               // -- We have #<line> "<filename>".
               G__getcintsysdir();
               G__StrBuf sysinclude_sb(G__MAXFILENAME);
               char *sysinclude = sysinclude_sb;
               sprintf(sysinclude, "%s/%s/include/", G__cintsysdir, G__CFG_COREVERSION);
               G__StrBuf sysstl_sb(G__MAXFILENAME);
               char *sysstl = sysstl_sb;
               sprintf(sysstl, "%s/%s/stl/", G__cintsysdir, G__CFG_COREVERSION);
               int len = strlen(sysinclude);
               int lenstl = strlen(sysstl);
#ifdef G__WIN32
               G__toUniquePath(sysinclude);
               G__toUniquePath(sysstl);
               G__toUniquePath(statement);
#endif // G__WIN32
               if (
                  !strncmp(sysinclude, statement + 1, (size_t)len) ||
                  !strncmp(sysstl, statement + 1, (size_t)lenstl)
               ) {
                  G__globalcomp = G__NOLINK;
               }
               else if (G__ifile.fp != G__mfp) {
                  G__globalcomp = G__store_globalcomp;
                  struct G__ConstStringList* sysdir = G__SystemIncludeDir;
                  while (sysdir) {
                     if (!strncmp(sysdir->string, statement + 1, sysdir->hash)) {
                        G__globalcomp = G__NOLINK;
                     }
                     sysdir = sysdir->prev;
                  }
               }
               statement[strlen(statement)-1] = '\0';
               strcpy(G__ifile.name, statement + 1);
               int hash = 0;
               int temp = 0;
               G__hash(G__ifile.name, hash, temp);
               temp = 0;
               int null_entry = -1;
               int i = 0;
               for (; i < G__nfile; ++i) {
                  if (!G__srcfile[i].filename && (null_entry == -1)) {
                     null_entry = i;
                  }
                  if (G__matchfilename(i, G__ifile.name)) {
                     temp = 1;
                     break;
                  }
               }
               if (temp) {
                  // --
                  G__ifile.filenum = i;
                  G__security = G__srcfile[i].security;
               }
               else if (null_entry != -1) {
                  // --
                  G__srcfile[null_entry].hash = hash;
                  G__srcfile[null_entry].filename = (char*) malloc(strlen(statement + 1) + 1);
                  strcpy(G__srcfile[null_entry].filename, statement + 1);
                  G__srcfile[null_entry].prepname = 0;
                  G__srcfile[null_entry].fp = 0;
                  G__srcfile[null_entry].maxline = 0;
                  G__srcfile[null_entry].breakpoint = 0;
                  G__srcfile[null_entry].security = G__security;
                  //
                  //  If we are using a preprocessed file, the logical file
                  //  is actually located in the result file from the preprocessor.
                  //
                  //  We need to need to carry this information on.
                  //
                  //  If the previous file (G__ifile) was preprocessed, this one
                  //  should also be.
                  //
                  if (G__cpp && G__srcfile[G__ifile.filenum].prepname[0]) {
                     G__srcfile[null_entry].prepname = (char*) malloc(strlen(G__srcfile[G__ifile.filenum].prepname) + 1);
                     strcpy(G__srcfile[null_entry].prepname, G__srcfile[G__ifile.filenum].prepname);
                     G__srcfile[null_entry].fp = G__ifile.fp;
                  }
                  G__srcfile[null_entry].included_from = G__ifile.filenum;
                  G__srcfile[null_entry].ispermanentsl = 0;
                  G__srcfile[null_entry].initsl = 0;
                  G__srcfile[null_entry].hasonlyfunc = 0;
                  G__ifile.filenum = null_entry;
               }
               else {
                  // --
                  if (G__nfile == (G__gettempfilenum() + 1)) {
                     G__fprinterr(G__serr, "Limitation: Sorry, can not create any more file entries.\n");
                  }
                  else {
                     // --
                     G__srcfile[G__nfile].hash = hash;
                     G__srcfile[G__nfile].filename = (char*) malloc(strlen(statement + 1) + 1);
                     strcpy(G__srcfile[G__nfile].filename, statement + 1);
                     G__srcfile[G__nfile].prepname = 0;
                     G__srcfile[G__nfile].fp = 0;
                     G__srcfile[G__nfile].maxline = 0;
                     G__srcfile[G__nfile].breakpoint = 0;
                     G__srcfile[G__nfile].security = G__security;
                     //
                     //  If we are using a preprocessed file, the logical file
                     //  is actually located in the result file from the preprocessor.
                     //
                     //  We need to need to carry this information on.
                     //
                     //  If the previous file (G__ifile) was preprocessed, this one
                     //  should also be.
                     //
                     if (
                        G__cpp &&
                        G__srcfile[G__ifile.filenum].prepname &&
                        G__srcfile[G__ifile.filenum].prepname[0]
                     ) {
                        // --
                        G__srcfile[G__nfile].prepname = (char*) malloc(strlen(G__srcfile[G__ifile.filenum].prepname) + 1);
                        strcpy(G__srcfile[G__nfile].prepname, G__srcfile[G__ifile.filenum].prepname);
                        G__srcfile[G__nfile].fp = G__ifile.fp;
                     }
                     G__srcfile[G__nfile].included_from = G__ifile.filenum;
                     G__srcfile[G__nfile].ispermanentsl = 0;
                     G__srcfile[G__nfile].initsl = 0;
                     G__srcfile[G__nfile].hasonlyfunc = 0;
                     G__ifile.filenum = G__nfile;
                     ++G__nfile;
                  }
               }
            }
            if ((c != '\n') && (c != '\r')) {
               while (((c = G__fgetc()) != '\n') && (c != '\r')) {
                  // intentionally empty
               }
            }
         }
      }
   }
   return 0;
}

/***********************************************************************
* G__pp_ifdefextern()
*
*   #ifdef __cplusplus
*   extern "C" {       ^
*   #endif
*
*   #ifdef __cplusplus ^
*   }
*   #endif
***********************************************************************/
static int G__pp_ifdefextern(char* temp)
{
  int cin;
  fpos_t pos;
  int linenum = G__ifile.line_number;
  fgetpos(G__ifile.fp,&pos);

  cin = G__fgetname(temp,"\"}#");

  if('}'==cin) {
    /******************************
     *   #ifdef __cplusplus
     *   }
     *   #endif
     *****************************/
    G__fignoreline();
    do {
      cin = G__fgetstream(temp,"#");
      cin = G__fgetstream(temp,"\n\r");
    } while(strcmp(temp,"endif")!=0); 
    return(G__IFDEF_ENDBLOCK);
  }

  if('#'!=cin && strcmp(temp,"extern")==0) {
    /******************************
     *   #ifdef __cplusplus
     *   extern "C" {
     *   #endif
     *****************************/
    /******************************
     *   #ifdef __cplusplus
     *   extern "C" {  ...  }
     *   #endif
     *****************************/
    
    G__var_type='p';
    if('{'!=cin) cin = G__fgetspace();
    if('"'==cin) {
      /* extern "C" {  } */
      int flag=0;
      int store_iscpp=G__iscpp;
      int store_externblock_iscpp=G__externblock_iscpp;
      G__StrBuf fname_sb(G__MAXFILENAME);
      char *fname = fname_sb;
      cin = G__fgetstream(fname,"\"");

      temp[0] = 0;
      do {
      cin = G__fgetstream(temp,"{\r\n");
      } while (0==temp[0] && (cin == '\r' || cin == '\n'));
      if(0!=temp[0] || '{'!=cin)  goto goback;

      cin = G__fgetstream(temp,"\n\r");
      if (cin=='}' && 0==strcmp(fname,"C")) {
        goto goback;
      }
      cin = G__fgetstream(temp,"#\n\r");
      if ( (cin=='\n'||cin=='\r') && temp[0]==0) {
         cin = G__fgetstream(temp,"#\n\r");
      }
      if('#'!=cin) goto goback;
      cin = G__fgetstream(temp,"\n\r");
      if ( (cin=='\n'||cin=='\r') && temp[0]==0) {
         cin = G__fgetstream(temp,"#\n\r");
      }
      if(strcmp(temp,"endif")!=0) goto goback;

      if(0==strcmp(fname,"C")) {
        G__externblock_iscpp = (G__iscpp||G__externblock_iscpp);
        G__iscpp=0; 
      }
      else {
        G__loadfile(fname);
        G__SetShlHandle(fname);
        flag=1;
      }
      int brace_level = 1;
      G__exec_statement(&brace_level);
      G__iscpp=store_iscpp;
      G__externblock_iscpp = store_externblock_iscpp;
      if(flag) G__ResetShlHandle();
      return(G__IFDEF_EXTERNBLOCK);
    }
  }

 goback:
  fsetpos(G__ifile.fp,&pos);
  G__ifile.line_number = linenum;
  return(G__IFDEF_NORMAL);
}

/***********************************************************************
* G__pp_undef()
*
* Called by
*   G__exec_statement(&brace_level);   '#undef'
*
*   #undef
***********************************************************************/
static void G__pp_undef()
{
   G__StrBuf temp_sb(G__MAXNAME);
   char *temp = temp_sb;
   G__fgetname(temp,"\n\r");

   typedef std::deque< ::Reflex::Member> rlist;
   rlist toberemoved;
   ::Reflex::Scope varscope ( ::Reflex::Scope::GlobalScope() );
   for(::Reflex::Member_Iterator i = varscope.DataMember_Begin();
       i != varscope.DataMember_End(); ++i )
   {
      if(i->Name() == temp && G__get_type(i->TypeOf())=='p') //CHECKME =='p' seems to restrictive shouldn't it be tolower(type)=='p' ?
      {
         toberemoved.push_back(*i);
      }
   }
   for(rlist::const_iterator j = toberemoved.begin();
      j != toberemoved.end(); ++j)
   {
      varscope.RemoveDataMember(*j);
   }
}

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Statements.
//

//______________________________________________________________________________
//
//  Exceptions.  try, throw, catch
//

//______________________________________________________________________________
static int G__exec_try(char* statement)
{
   // -- FIXME: Describe this function!
   int brace_level = 0;
   G__exec_statement(&brace_level);
   if (G__return == G__RETURN_TRY) {
      G__no_exec = 0;
      G__return = G__RETURN_NON;
      return G__exec_catch(statement);
   }
   return 0;
}

/***********************************************************************
* G__ignore_catch()
*
***********************************************************************/
static int G__ignore_catch()
{
   if(G__asm_noverflow) {
      fpos_t fpos1;
      fseek(G__ifile.fp,-1,SEEK_CUR);
      fseek(G__ifile.fp,-1,SEEK_CUR);
      while(fgetc(G__ifile.fp)!='a') {
         fseek(G__ifile.fp,-1,SEEK_CUR);
         fseek(G__ifile.fp,-1,SEEK_CUR);
      }
      while(fgetc(G__ifile.fp)!='c') {
         fseek(G__ifile.fp,-1,SEEK_CUR);
         fseek(G__ifile.fp,-1,SEEK_CUR);
      }
      fseek(G__ifile.fp,-1,SEEK_CUR);
      fgetpos(G__ifile.fp,&fpos1);
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CATCH\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__CATCH;
      G__asm_inst[G__asm_cp+1]=G__ifile.filenum;
      G__asm_inst[G__asm_cp+2]=G__ifile.line_number;
#if defined(G__NONSCALARFPOS2)
      G__asm_inst[G__asm_cp+3]=(long)fpos1.__pos;
#elif defined(G__NONSCALARFPOS_QNX)
      G__asm_inst[G__asm_cp+3]=(long)fpos1._Off;
#else
      G__asm_inst[G__asm_cp+3]=(long)fpos1;
#endif
      G__inc_cp_asm(5,0);
      G__fignorestream("(");
   }

   G__fignorestream(")");
   G__no_exec = 1;
   int brace_level = 0;
   G__exec_statement(&brace_level);
   G__no_exec = 0;
   return(0);
}

/***********************************************************************
* G__exec_throw()
*
***********************************************************************/
static int G__exec_throw(char* statement)
{
   int iout;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   G__fgetstream(buf,";");
   if(isdigit(buf[0])||'.'==buf[0]) {
      strcpy(statement,buf);
      iout=5;
   }
   else {
      sprintf(statement,"new %s",buf);
      iout=strlen(statement);
   }
   if(iout>4) {
      int largestep=0;
      if(G__breaksignal && G__beforelargestep(statement,&iout,&largestep)>=1)
         return(1);
      G__exceptionbuffer = G__getexpr(statement);
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
         if(G__asm_dbg) G__fprinterr(G__serr,"%3x: THROW\n",G__asm_cp);
#endif
         G__asm_inst[G__asm_cp]=G__THROW;
         G__inc_cp_asm(1,0);
      }
      if(largestep) G__afterlargestep(&largestep);
      G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
      if('U'==G__get_type(G__exceptionbuffer)) G__value_typenum(G__exceptionbuffer) = G__deref(G__value_typenum(G__exceptionbuffer));
   }
   else {
      G__exceptionbuffer = G__null;
   }
   if(0==G__no_exec_compile) {
      G__no_exec=1;
      G__return=G__RETURN_TRY;
   }
   return(0);
}

/***********************************************************************
* G__exec_function()
*
***********************************************************************/
static int G__exec_function(char* statement, int* pc, int* piout, int* plargestep, G__value* presult)
{
   // -- Function call.
   //
   // Return 0 if function called, return 1 if function is *not* called.
   //
   if ((*pc == ';') || G__isoperator(*pc) || (*pc == ',') || (*pc == '.') || (*pc == '[')) {
      //fprintf(stderr, "G__exec_function: Function call is followed by an operator.\n");
      if ((*pc != ';') && (*pc != ',')) {
         statement[(*piout)++] = *pc;
         *pc = G__fgetstream_new(statement + (*piout) , ";");
      }
      if (G__breaksignal) {
         int ret = G__beforelargestep(statement, piout, plargestep);
         if (ret > 1) {
            return 1;
         }
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         G__asm_clear();
      }
#endif // G__ASM
      //
      // Evaluate the expression which contains the function call.
      //
      //fprintf(stderr, "G__exec_function: Calling G__getexpr(): '%s'\n", statement);
      *presult = G__getexpr(statement);
   }
   else if (*pc == '(') {
      //fprintf(stderr, "G__exec_function: Function call is followed by '('.\n");
      int len = strlen(statement);
      statement[len++] = *pc;
      *pc = G__fgetstream_newtemplate(statement + len, ")");
      len = strlen(statement);
      statement[len++] = *pc;
      statement[len] = 0;
      *pc = G__fgetspace();
      while (*pc != ';') {
         len = strlen(statement);
         statement[len++] = *pc;
         *pc = G__fgetstream_newtemplate(statement + len, ");");
         if (*pc == ';') {
            break;
         }
         len = strlen(statement);
         statement[len++] = *pc;
         statement[len] = 0;
         *pc = G__fgetspace();
      }
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         G__asm_clear();
      }
#endif // G__ASM
      //
      // Evaluate the expression which contains the function call.
      //
      //fprintf(stderr, "G__exec_function: Calling G__getexpr(): '%s'\n", statement);
      *presult = G__getexpr(statement);
   }
   else {
      // -- Function-style macro without ';' at the end.
      //fprintf(stderr, "G__exec_function: We have a function-style macro call.\n");
      if (G__breaksignal) {
         int ret = G__beforelargestep(statement, piout, plargestep);
         if (ret > 1) {
            return 1;
         }
      }
      //
      // Expand the function-style macro.
      //
      // Note: This call requires the macro to have balanced curly braces.
      // FIXME: This is a terrible hack!
      *presult = G__execfuncmacro(statement, piout);
      if (!(*piout))  {
         // -- It was not a macro.
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: %s Missing ';'", statement);
            G__printlinenum();
         }
      }
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
   }
   if (G__p_tempbuf->level >= G__templevel && G__p_tempbuf->prev) {
      G__free_tempobject();
   }
   if (*plargestep) {
      G__afterlargestep(plargestep);
   }
   return 0;
}

#ifdef G__ASM
/***********************************************************************
* G__alloc_breakcontinue_list
*
***********************************************************************/
static G__breakcontinue_list* G__alloc_breakcontinue_list()
{
  struct G__breakcontinue_list *p;
  p = G__pbreakcontinue;
  G__pbreakcontinue = (struct G__breakcontinue_list*)NULL;
  return(p);
}

/***********************************************************************
* G__store_breakcontinue_list
*
***********************************************************************/
static void G__store_breakcontinue_list(int idx,int isbreak)
{
   // -- FIXME: Describe this function!
   struct G__breakcontinue_list* p = (struct G__breakcontinue_list*) malloc(sizeof(struct G__breakcontinue_list));
   p->next = G__pbreakcontinue;
   p->isbreak = isbreak;
   p->idx = idx;
   G__pbreakcontinue = p;
}

/***********************************************************************
* G__free_breakcontinue_list
*
***********************************************************************/
static void G__free_breakcontinue_list(G__breakcontinue_list *oldlist)
{
   // -- FIXME: Describe this function!
   while (G__pbreakcontinue) {
      struct G__breakcontinue_list* p = G__pbreakcontinue->next;
      free(G__pbreakcontinue);
      G__pbreakcontinue = p;
   }
   G__pbreakcontinue = oldlist;
}

/***********************************************************************
* G__set_breakcontinue_destination
*
***********************************************************************/
static void G__set_breakcontinue_destination(int break_destidx, int continue_destidx, G__breakcontinue_list* oldlist)
{
    // -- FIXME: Describe this function!
   while (G__pbreakcontinue) {
      if (G__pbreakcontinue->isbreak) {
         // -- This entry in the list is for a break statement.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "  %x: assigned JMP %x (for break)  %s:%d\n", G__pbreakcontinue->idx, break_destidx, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__pbreakcontinue->idx] = break_destidx;
      }
      else {
         // -- This entry in the list is for a continue statement.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "  %x: assigned JMP %x (for continue)  %s:%d\n", G__pbreakcontinue->idx, continue_destidx, __FILE__, __LINE__
            );
         }
#endif // G__ASM_DBG
         G__asm_inst[G__pbreakcontinue->idx] = continue_destidx;
      }
      struct G__breakcontinue_list* p = G__pbreakcontinue->next;
      free(G__pbreakcontinue);
      G__pbreakcontinue = p;
   }
   G__pbreakcontinue = oldlist;
}

//______________________________________________________________________________
static void G__set_breakcontinue_breakdestination(int break_destidx, G__breakcontinue_list* oldlist)
{
   // -- FIXME: Describe this function!
   struct G__breakcontinue_list* p = G__pbreakcontinue;
   while (p) {
      if (p->isbreak) {
         // -- This entry in the list is for a break statement.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "  %x: assigned JMP %x (for break)  %s:%d\n", p->idx, break_destidx, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[p->idx] = break_destidx;
         struct G__breakcontinue_list* next = p->next;
         free(p);
         p = next;
      }
      else {
         // -- This entry in the list is for a continue statement.
         struct G__breakcontinue_list* next = p->next;
         p->next = oldlist;
         oldlist = p;
         p = next;
      }
   }
   G__pbreakcontinue = oldlist;
}
#endif /* G__ASM */

/***********************************************************************
* G__exec_switch()
*
* Called by
*   G__exec_statement(&brace_level); 
*
***********************************************************************/
static G__value G__exec_switch()
{
   // -- Handle switch.
   //
   // Note: G__no_exec is always zero when entered.
   //
   //fprintf(stderr, "G__exec_switch: Begin.\n");
   //fprintf(stderr, "G__exec_switch: G__no_exec: %d G__asm_noverflow: %d G__no_exec_compile: %d G__asm_wholefunction: %d\n", G__no_exec, G__asm_noverflow, G__no_exec_compile, G__asm_wholefunction);
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 30);
   //   fprintf(stderr, "G__exec_switch: begin: peek ahead: '%s'\n", buf);
   //}
   int store_ifswitch = G__ifswitch;
   G__ifswitch = G__IFSWITCH;
   //
   //
   //
   int store_G__prevcase = G__prevcase;
   G__prevcase = 0;
   //
   //  Scan the switch condition.
   //
   G__StrBuf condition_sb(G__ONELINE);
   char *condition = condition_sb;
   G__fgetstream(condition, ")");
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 30);
   //   fprintf(stderr, "G__exec_switch: after scanning switch condition text: peek ahead: '%s'\n", buf);
   //}
   //
   //  Handle a breakpoint now.
   //
   int iout = 0;
   int largestep = 0;
   if (G__breaksignal) {
      // -- We have hit a breakpoint.
      int ret = G__beforelargestep(condition, &iout, &largestep);
      if (ret > 1) {
         // Restore state.
         G__ifswitch = store_ifswitch;
         G__prevcase = store_G__prevcase;
         // And return immediately.
         return G__null;
      }
   }
   //
   //  Evaluate the switch expression.
   //
   G__value reg = G__getexpr(condition);
   if (largestep) {
      G__afterlargestep(&largestep);
   }
#ifdef G__ASM
   //
   //  Allocate a break/continue buffer.
   //
   struct G__breakcontinue_list* store_pbreakcontinue = 0;
   int allocflag = 0;
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
      store_pbreakcontinue = G__alloc_breakcontinue_list();
      allocflag = 1;
   }
#endif // G__ASM
   //
   //  Skip until we have the beginning of the body.
   //
   // FIXME: We should only allow whitespace here!
   G__fignorestream("{");
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 30);
   //   fprintf(stderr, "G__exec_switch: after skipping to switch body: peek ahead: '%s'\n", buf);
   //}
   // Initialize our result.
   G__value result = G__null;
   if (G__no_exec_compile) {
      // -- Generate code for the whole switch block, but take no semantic actions.
      //{
      //   char buf[128];
      //   G__fgetstream_peek(buf, 30);
      //   fprintf(stderr, "G__exec_switch: compile whole switch block, peek ahead: '%s'\n", buf);
      //}
      // Flag that we need to generate code for case clauses.
      int store_G__switch = G__switch;
      G__switch = 1;
      // Tell the parser to finish the switch block.
      int brace_level = 1;
      // Call the parser.
      G__exec_statement(&brace_level);
      // Restore state.
      G__switch = store_G__switch;
   }
   else {
      // -- We are going to execute the switch.
      //
      //  Skip code while testing case expressions
      //  until we get a match, or reach the end of the block.
      //
      { // Delimiter for push/popd of environment.
         int store_no_exec_compile = G__no_exec_compile;
         //fprintf(stderr, "G__exec_switch: before case search: G__asm_noverflow: %d G__no_exec_compile: %d\n", G__asm_noverflow, G__no_exec_compile);
#ifdef G__ASM
         //
         //  If we are not generating bytecode, then
         //  skip code.  Otherwise do not skip code,
         //  do not execute, just generate bytecode.
         //
         if (!G__asm_noverflow) {
            // -- We are *not* generating bytecode.
            // Flag we are skipping code.
            G__no_exec = 1;
         }
         else {
            // -- We are generating bytecode.
            // Flag we are *not* executing, but we should generate bytecode.
            G__no_exec_compile = 1;
         }
#else // G__ASM
         // Flag we are skipping code.
         G__no_exec = 1;
#endif // G__ASM
         // Tell the parser to evaluate case expressions.
         int store_G__switch = G__switch;
         G__switch = 1;
         // Tell the parser it should return control after evaluating a case expression.
         int store_G__switch_searching = G__switch_searching;
         G__switch_searching = 1;
         // Flag that we have not evaluated a case expression yet.
         result = G__start;
         // Flag that we have not matched a case expression yet.
         int isequal = 0;
         // FIXME: This does not handle the default case properly!
         // FIXME: We must find it first, remember where it is, then
         // FIXME: come back to it if nothing else matches.
         while (
                (G__value_typenum(result) != G__value_typenum(G__null)) && // not the end of the switch block, and
                (G__value_typenum(result) != G__value_typenum(G__default)) && // not the default case, and
                !isequal // not a case expression match
                ) {
            //fprintf(stderr, "G__exec_switch: Parsing next case clause during search.\n");
            // Tell the parser to stop at the closing curly brace of the switch.
            int brace_level = 1;
            // And call the parser to get the next case clause.
            result = G__exec_statement(&brace_level);
            //fprintf(stderr, "G__exec_switch: Case clause parse has returned.\n");
            // Test the case expression against the switch expression.
            isequal = G__cmp(result, reg);
         }
         //fprintf(stderr, "G__exec_switch: Case clause search has finished.\n");
         // Restore state.
         G__switch = store_G__switch;
         G__switch_searching = store_G__switch_searching;
         G__no_exec = 0;
         G__no_exec_compile = store_no_exec_compile;
         //fprintf(stderr, "G__exec_switch: after case search: G__asm_noverflow: %d G__no_exec_compile: %d\n", G__asm_noverflow, G__no_exec_compile);
      }
      if (G__value_typenum(result) != G__value_typenum(G__null)) {
         // -- Case is a match or is the default case..
         // FIXME: This is wrong if the default case is not at the end!
         //fprintf(stderr, "G__exec_switch: We have matched a case.\n");
         //
         //  Check if we have reached a breakpoint.
         //
         if (
            !G__nobreak &&
            !G__disp_mask &&
            !G__no_exec_compile &&
            G__srcfile[G__ifile.filenum].breakpoint &&
            (G__srcfile[G__ifile.filenum].maxline > G__ifile.line_number)
         ) {
            G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number] |= G__TRACED;
         }
         //
         //  Peek ahead, we need to know whether or not the
         //  case clause is a block or not.
         //
         //int c = G__fgetspace_peek();
         //int isblock = 0;
         //if (c == '{') {
         //   isblock = 1;
         //}
         //
         //  Execute the case until it breaks,
         //  continues, gotos, or reaches the
         //  end of the block.
         //
         // Tell the parser to handle case clauses.
         int brace_level = 1;
         { // Delimiter for push/popd of environment.
            int store_G__switch = G__switch;
            G__switch = 1;
            // Tell the parser to go until the end of the switch block.
            // Call the parser.
            //fprintf(stderr, "G__exec_switch: Running the case block.\n");
            //fprintf(stderr, "G__exec_switch: just before running case block: G__asm_noverflow: %d G__no_exec_compile: %d\n", G__asm_noverflow, G__no_exec_compile);
            result = G__exec_statement(&brace_level);
            //fprintf(stderr, "G__exec_switch: Case block parse has returned.\n");
            // Restore state.
            G__switch = store_G__switch;
         }
         //
         //  Check if user requested an immediate return.
         //
         if (G__return != G__RETURN_NON) {
            // -- The user requested an immediate return.
            // Restore state.
            G__ifswitch = store_ifswitch;
            G__prevcase = store_G__prevcase;
            // And return.
            return result;
         }
         // Check for a break, continue, or goto.
         if (G__value_typenum(result) == G__value_typenum(G__block_break)) {
            // -- The case ended with a break, continue, or goto.
            //fprintf(stderr, "G__exec_switch: Case ended with a break, continue, or goto.\n");
            // Check for goto.
            if (result.ref == G__block_goto.ref) {
               // -- The case did a goto, return that.
               //fprintf(stderr, "G__exec_switch: Case ended with a goto.\n");
               // Restore state.
               G__ifswitch = store_ifswitch;
               G__prevcase = store_G__prevcase;
               // And return the goto result to the parser (our caller).
               return result;
            }
            //
            //  Consume the rest of the switch block.
            //
            //{
            //   char buf[128];
            //   G__fgetstream_peek(buf, 30);
            //   fprintf(stderr, "G__exec_switch: consume rest of switch block, peek ahead: '%s'\n", buf);
            //}
            int store_no_exec_compile = G__no_exec_compile;
            int store_G__switch = G__switch;
#ifdef G__ASM
            if (!G__asm_noverflow) {
               // Flag that we are skipping code.
               G__no_exec = 1;
            }
            else {
               // Flag that we are not executing, but we are generating bytecode.
               G__no_exec_compile = 1;
               // Flag that we need to generate code for case clauses.
               G__switch = 1;
            }
#else // G__ASM
            // Flag that we are skipping code.
            G__no_exec = 1;
#endif // G__ASM
            // Tell the parser to finish the switch block.
            // Call the parser.
            G__exec_statement(&brace_level);
            // Restore state.
            G__no_exec = 0;
            G__no_exec_compile = store_no_exec_compile;
            G__switch = store_G__switch;
         }
      }
   }
   //
   //  We have parsed until the end of
   //  the switch block.
   //
#ifdef G__ASM
   //
   //  Backpatch any breaks now that we know where the end is.
   //
   if (G__asm_noverflow) {
      if (allocflag) {
         //fprintf(stderr, "G__exec_switch: Begin backpatch of jumps for break.\n");
         G__set_breakcontinue_breakdestination(G__asm_cp, store_pbreakcontinue);
         //fprintf(stderr, "G__exec_switch: End backpatch of jumps for break.\n");
         allocflag = 0;
      }
   }
#endif // G__ASM
#ifdef G__ASM
   //
   //  Free the break/continue buffer.
   //
   if (allocflag) {
      G__free_breakcontinue_list(store_pbreakcontinue);
      allocflag = 0;
   }
#endif // G__ASM
#ifdef G__ASM
   //
   //  Backpatch the final case jump.
   //
   if (G__asm_noverflow) {
      if (G__prevcase) {
         // -- Backpatch the jump from the previous case.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "   %3x: CNDJMP %x assigned (switch, next case) %s:%d\n", G__prevcase - 1, G__asm_cp, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
         G__asm_inst[G__prevcase] = G__asm_cp;
      }
   }
#endif // G__ASM
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 30);
   //   fprintf(stderr, "G__exec_switch: done with switch block, peek ahead: '%s'\n", buf);
   //}
   // Restore state.
   G__no_exec = 0;
   if (G__asm_noverflow && G__no_exec_compile) {
      //fprintf(stderr, "G__exec_switch: End.\n");
      // Restore state.
      G__ifswitch = store_ifswitch;
      G__prevcase = store_G__prevcase;
      // All done, return null to the caller.
      return G__null;
   }
   else {
      //
      // If switch block was exited with a break,
      // consume the break return status and return
      // null instead, but do return to caller now.
      //
      if(G__value_typenum(result)==G__value_typenum(G__block_break) && result.obj.i==G__BLOCK_BREAK) {
         // -- Case did a break, return now.
         //fprintf(stderr, "G__exec_switch: Last statement executed was a break.\n");
         //fprintf(stderr, "G__exec_switch: End.\n");
         // Restore state.
         G__ifswitch = store_ifswitch;
         G__prevcase = store_G__prevcase;
         // All done, consume the break return status and return null instead.
         return G__null;
      }
      //
      // If switch block was exited with a continue,
      // return that fact to the enclosing statement,
      // it must handle it.
      //
      if ((G__value_typenum(result) == G__value_typenum(G__block_break)) && (result.obj.i == G__BLOCK_BREAK)) {
         // -- Case did a continue, return now.
         //fprintf(stderr, "G__exec_switch: Last statement executed was a continue.\n");
         //fprintf(stderr, "G__exec_switch: End.\n");
         // Restore state.
         G__ifswitch = store_ifswitch;
         G__prevcase = store_G__prevcase;
         // All done, return the continue status to our caller.
         return result;
      }
   }
   // Restore state.
   G__ifswitch = store_ifswitch;
   G__prevcase = store_G__prevcase;
   // All done, return status of last executed command to caller.
   //fprintf(stderr, "G__exec_switch: End.\n");
   return result;
}

//______________________________________________________________________________
static G__value G__exec_switch_case(char* casepara)
{
   // -- Handle a case expression inside a switch statement.
   //
   // Note: G__no_exec may be true when entered.
   //
   //fprintf(stderr, "G__exec_switch_case: Begin.\n");
#ifdef G__ASM
   // For remembering where to backpatch the jump into next case block.
   int jmp1 = 0;
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
      //
      //  Terminate previous case and backpatch its
      //  conditional jump on the result of its case
      //  expression.
      //
      if (G__prevcase) {
         // --  There was a previous case clause.
         //
         //  End the previous case clause with a jump
         //  to the code block of this clause, which is
         //  after the case expression test of this clause.
         //
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: JMP (for case, end of case, jump into next case block body, intentional fallthrough, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__JMP;
         // Remember location so we can backpatch it later.
         jmp1 = G__asm_cp + 1;
         G__inc_cp_asm(2, 0);
         //
         //  Backpatch previous case expression test to jump to this case if not equal.
         //
         G__asm_inst[G__prevcase] = G__asm_cp;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "   %3x: CNDJMP %x assigned (for case expression not equal, jump to next case test)  %s:%d\n", G__prevcase - 1, G__asm_cp, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         // --
      }
      //
      //  Make a copy of the switch expression on the data stack.
      //
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHCPY (for case, copy selector value for test against case expression)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHCPY;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   //
   //  Evaluate the case expression and stack it on the data stack.
   //
   int store_no_exec = G__no_exec;
   G__no_exec = 0;
   int store_no_exec_compile = G__no_exec_compile;
   if (G__no_exec_compile && G__switch_searching) {
      G__no_exec_compile = 0;
   }
   G__value result = G__getexpr(casepara);
   G__no_exec_compile = store_no_exec_compile;
   G__no_exec = store_no_exec;
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
      //
      //  Compare the switch expression and the case expression.
      //
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: OP2_OPTIMIZED == (for case, test selector against case expression)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__OP2_OPTIMIZED;
      G__asm_inst[G__asm_cp+1] = (long) G__CMP2_equal;
      G__inc_cp_asm(2, 0);
      //
      //  Jump to next case if not equal.
      //
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: CNDJMP (for case, jump to next case test if no match with selector value, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__CNDJMP;
      // Remember location, so it can be backpatched when the next case clause is seen.
      G__prevcase = G__asm_cp + 1;
      G__inc_cp_asm(2, 0);
      //
      //  Backpatch the jump into our code block
      //  in the previous case clause.
      //
      if (jmp1) {
         // -- We had a previous case clause.
         G__asm_inst[jmp1] = G__asm_cp;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "   %3x: JMP %x assigned (for case, jump into this case block body on intentional fallthrough)  %s:%d\n", jmp1 - 1, G__asm_cp, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         // --
      }
   }
#endif // G__ASM
   //fprintf(stderr, "G__exec_switch_case: End.\n");
   // And return the result of evaluating the case expression.
   return result;
}

//______________________________________________________________________________
static G__value G__exec_if()
{
   // -- Handle an if (...) statement.
   //
   // Note: The value of G__no_exec is zero on entry.
   //
   //fprintf(stderr, "---if\n");
   bool localFALSE = false;
   int iout = 0;
   int asm_jumppointer = 0;
   int largestep = 0;
   G__value result = G__null;
   //
   //  Flag that we are running an if statement.
   //
   int store_ifswitch = G__ifswitch;
   G__ifswitch = G__IFSWITCH;
   //
   //  Get the text of the if condition from the input file.
   //
   char* condition = (char*) malloc(G__LONGLINE);
   G__fgetstream_new(condition, ")");
   condition = (char*) realloc((void*) condition, strlen(condition) + 10);
   //
   //  If the previous read hit a breakpoint, pause, and exit if requested.
   //
   if (G__breaksignal) {
      if (G__beforelargestep(condition, &iout, &largestep) > 1) {
         G__ifswitch = store_ifswitch;
         free(condition);
         //fprintf(stderr, "---end if\n");
         return G__null;
      }
   }
   //
   //  Peek ahead, we need to know whether or not the
   //  then clause is a block or not.
   //
   int c = G__fgetspace_peek();
   int isblock = 0;
   if (c == '{') {
      isblock = 1;
   }
   //fprintf(stderr, "G__exec_if: isblock: %d\n", isblock);
   //
   //  Test the if condition.
   //
   int condval = G__test(condition);
   if (largestep) {
      G__afterlargestep(&largestep);
   }
   // Free any generated temporaries.
   if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
      G__free_tempobject();
   }
#ifdef G__ASM
   //
   //  Generate bytecode for the jump skipping to the else clause.
   //
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: CNDJMP (if test, jmp to else, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__CNDJMP;
      // Remember position for backpatching the jump destination.
      asm_jumppointer = G__asm_cp + 1;
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   //
   //  Now either run or skip the then clause.
   //
   if (condval) {
      //fprintf(stderr, "G__exec_if: cond was true.\n");
      // -- Condition was true, we are going to execute the body.
      // Flag that we are not going to skip the body.
      G__no_exec = 0;
      //
      //  Run the body of the then clause.
      //
      //fprintf(stderr, "--- then\n");
      // Tell parser to execute only one statement or a block.
      int brace_level = 0;
      result = G__exec_statement(&brace_level);
      //
      //  Return immediately if requested by user.
      //
      if (G__return != G__RETURN_NON) {
         // --
#ifdef G__ASM
         //
         //  Generate bytecode, backpatch the jump for skipping the true clause.
         //
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
            //
            //  Backpatch the jump after the condition test,
            //  now that where know where the else clause is.
            //
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "   %x: CNDJMP assigned to %x  (if stmt, jmp to else)  %s:%d\n", asm_jumppointer - 1, G__asm_cp, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[asm_jumppointer] = G__asm_cp;
            // Remember where to backpatch the jump destination.
            asm_jumppointer = G__asm_cp - 1; // to skip else clause
         }
#endif // G__ASM
         G__ifswitch = store_ifswitch;
         free(condition);
         //fprintf(stderr, "---end if\n");
         return result;
      }
      //
      //  Check if the then clause terminated early due
      //  to a break, continue, or goto statement.
      //
      if (G__value_typenum(result) == G__value_typenum(G__block_break)) {
         // -- Statement block was exited by break, continue, or goto.
         if (result.ref == G__block_goto.ref) {
            // -- Exited by goto.
         }
         else {
            // -- Exited by break or continue.
            //fprintf(stderr, "G__exec_if: break or continue executed from an if statement then clause.\n");
            if (!G__asm_noverflow) {
               // -- We are *not* generating bytecode, ignore until end then clause.
               if (isblock) {
                  //fprintf(stderr, "G__exec_if: ignoring until next '}' because of break or continue.\n");
                  //{
                  //   char buf[128];
                  //   G__fgetstream_peek(buf, 30);
                  //   fprintf(stderr, "G__exec_if: peek ahead: '%s'\n", buf);
                  //}
                  // Tell the parser to ignore statements.
                  G__no_exec = 1;
                  // Note: We must do this because we may expand a macro while skipping.
                  //G__fignorestream("}");
                  G__exec_statement(&brace_level);
                  // Note: This is always zero on entry to exec_if.
                  G__no_exec = 0;
                  //{
                  //   char buf[128];
                  //   G__fgetstream_peek(buf, 30);
                  //   fprintf(stderr, "G__exec_if: peek ahead: '%s'\n", buf);
                  //}
               }
            }
            else {
               // -- We are generating bytecode, compile to end of then clause.
               if (isblock) {
                  //fprintf(stderr, "G__exec_if: compiling until next '}' because of break or continue.\n");
                  // Flag that we are not executing, but we are generating bytecode.
                  int store_no_exec_compile = G__no_exec_compile;
                  G__no_exec_compile = 1;
                  // Tell the parser to go until a right curly brace is seen.
                  // And call the parser.
                  G__exec_statement(&brace_level);
                  // Done, restore state.
                  G__no_exec_compile = store_no_exec_compile;
               }
            }
         }
      }
      //
      //  Remember that we ran the body.
      //
      localFALSE = false;
   }
   else {
      // -- Condition was false, we are *not* going to execute the body.
      //fprintf(stderr, "G__exec_if: cond was false.\n");
#ifdef G__ASM
      // Save this, we are about to temporarily change it.
      int store_no_exec_compile = G__no_exec_compile;
      if (G__asm_noverflow) {
         // -- Flag that we are *not* executing, but we should still generate bytecode.
         G__no_exec_compile = 1;
      }
      else {
         // -- Flag that we are going to skip the body.
         // Note: G__no_exec is always 0 when exec_if is entered.
         G__no_exec = 1;
      }
#else // G__ASM
      // Flag that we are going to skip the body.
      // Note: G__no_exec is always 0 when exec_if is entered.
      G__no_exec = 1;
#endif // G__ASM
      //
      //  Skip the body.
      // 
      //fprintf(stderr, "---then\n");
      // Tell parser to execute only one statement or a block.
      int brace_level = 0;
      // Call parser.
      G__exec_statement(&brace_level);
      // Done, restore state.
#ifdef G__ASM
      G__no_exec_compile = store_no_exec_compile;
#endif // G__ASM
      // Note: G__no_exec is always 0 when exec_if is entered.
      G__no_exec = 0;
      // Remember that we skipped the body.
      localFALSE = true;
   }
#ifdef G__ASM
   //
   //  Generate bytecode, make a jump for skipping the else clause,
   //  and backpatch the jump for skipping the true clause.
   //
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
      //
      //  Generate a jump for skipping past the else clause.
      //
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: JMP (if stmt, end of then, skip else, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__JMP;
      G__inc_cp_asm(2, 0);
      //
      //  Backpatch the jump after the condition test,
      //  now that where know where the else clause is.
      //
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "   %x: CNDJMP assigned to %x  (if stmt, jmp to else)  %s:%d\n", asm_jumppointer - 1, G__asm_cp, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[asm_jumppointer] = G__asm_cp;
      // Remember where to backpatch the jump destination.
      asm_jumppointer = G__asm_cp - 1; // to skip else clause
   }
#endif // G__ASM
   //fprintf(stderr, "---end then\n");
   //
   //  Check for an else clause, and run it if it is there.
   //
   // First, remember the current file position.
   fpos_t store_fpos;
   fgetpos(G__ifile.fp, &store_fpos);
   int store_line_number = G__ifile.line_number;
   // Skip any whitespace.
   c = ' ';
   while (isspace(c)) {
      // -- Skip whitespace and comments, handle preprocessor directives.
      c = G__fgetc();
      G__temp_read++;
      while ((c == '/') || (c == '#')) {
         // -- Scan any possible comments or preprocessor directives.
         if (c == '/') {
            // -- Possible comment, get next character.
            c = G__fgetc();
            switch (c) {
               case '*':
                  // -- C comment, skip it.
                  if (G__skip_comment() == EOF) {
                     // -- Nothing more to read, all done.
#ifdef G__ASM
                     //
                     //  Generate bytecode, backpatch the jump skipping the else clause.
                     //
                     if (G__asm_noverflow) {
                        // -- We are generating bytecode.
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(G__serr, "   %x: JMP assigned to %x (if stmt, jmp skipping the else clause)  %s:%d\n", asm_jumppointer - 1, G__asm_cp, __FILE__, __LINE__);
                        }
#endif // G__ASM_DBG
                        // Backpatch the jump destination for skipping the else
                        // clause, now that we know where it ends.
                        G__asm_inst[asm_jumppointer] = G__asm_cp;
                        // Remember where the if (...) {...} else {...} statement ended.
                        G__asm_cond_cp = G__asm_cp;
                     }
#endif // G__ASM
                     G__ifswitch = store_ifswitch;
                     free(condition);
                     //fprintf(stderr, "---end if\n");
                     return G__null;
                  }
                  break;
               case '/':
                  // -- C++ comment, ignore the rest of the line.
                  G__fignoreline();
                  break;
               default:
                  // -- Syntax error.
                  G__commenterror();
                  break;
            }
            fgetpos(G__ifile.fp, &store_fpos);
            store_line_number = G__ifile.line_number;
            c = G__fgetc();
            G__temp_read = 1;
         }
         else if (c == '#') {
            // -- A preprocessor directive.
            G__pp_command();
            c = G__fgetc();
            G__temp_read = 1;
         }
      }
      if (c == EOF) {
         // -- Hit end of file, nothing more to do.
         G__genericerror("Error: unexpected if() { } EOF");
         if (G__key) {
            system("key .cint_key -l execute");
         }
         G__eof = 2;
#ifdef G__ASM
         //
         //  Generate bytecode, backpatch the jump skipping the else clause.
         //
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "   %x: JMP assigned to %x (if stmt, jmp skipping the else clause)  %s:%d\n", asm_jumppointer - 1, G__asm_cp, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            // Backpatch the jump destination for skipping the else
            // clause, now that we know where it ends.
            G__asm_inst[asm_jumppointer] = G__asm_cp;
            // Remember where the if (...) {...} else {...} statement ended.
            G__asm_cond_cp = G__asm_cp;
         }
#endif // G__ASM
         G__ifswitch = store_ifswitch;
         free(condition);
         //fprintf(stderr, "---end if\n");
         return G__null;
      }
   }
   // Now read the next four characters.
   char statement[10] = { 0 };
   statement[0] = c;
   for (iout = 1; iout <= 3; ++iout) {
      c = G__fgetc();
      G__temp_read++;
      if (c == EOF) {
         iout = 4;
         statement[0] = '\0';
      }
      statement[iout] = c;
   }
   statement[4] = '\0';
   //fprintf(stderr, "G__exec_if: stmt: %s\n", statement);
   // Check to see if it is an else (...) clause.
   if (!strcmp(statement, "else")) {
      // -- We have seen the else keyword, execute the else clause.
      //fprintf(stderr, "---else\n");
      G__temp_read = 0;
      if (localFALSE || G__asm_wholefunction) {
         // -- We need to run the else clause.
         // Peek ahead one character to see if the else clause is a block.
         int peekc = G__fgetspace_peek();
         int peek_isblock = 0;
         if (peekc == '{') {
            peek_isblock = 1;
         }
         //fprintf(stderr, "G__exec_if: isblock: %d\n", peek_isblock);
         // Flag that we are going to run the clause.
         // Note: G__no_exec is always zero on entry to exec_if.
         // FIXME: This is probably not necessary?
         G__no_exec = 0;
         //
         //  Execute the else clause.
         //
         // Tell the parser to execute only one statement or block.
         int brace_level = 0;
         // Call the parser.
         result = G__exec_statement(&brace_level);
         //
         //  Check if the else clause terminated early due
         //  to a break, continue, or goto statement.
         //
         if (G__value_typenum(result) == G__value_typenum(G__block_break)) {
            // -- Statement block was exited by break, continue, or goto.
            if (result.ref == G__block_goto.ref) {
               // -- Exited by goto.
               // Return the result to our caller, it must handle it.
               //fprintf(stderr, "---end if\n");
               return result;
            }
            else {
               // -- Exited by break or continue.
               //fprintf(stderr, "G__exec_if: break or continue executed from an if statement else clause.\n");
               if (!G__asm_noverflow) {
                  // -- We are *not* generating bytecode, ignore until end then clause.
                  if (peek_isblock) {
                     // -- We are *not* in the if () break; case.
                     // Tell the parser to ignore statements.
                     G__no_exec = 1;
                     // Note: We must do this because we may expand a macro while skipping.
                     //G__fignorestream("}");
                     G__exec_statement(&brace_level);
                     // Note: This is always zero on entry to exec_if.
                     G__no_exec = 0;
                  }
               }
               else {
                  // -- We are generating bytecode, compile to end of then clause.
                  if (peek_isblock) {
                     // -- We are *not* in the if () break; case.
                     // Flag that we are not executing, but we are generating bytecode.
                     int store_no_exec_compile = G__no_exec_compile;
                     G__no_exec_compile = 1;
                     // Tell the parser to go until a right curly brace is seen.
                     // And call the parser.
                     G__exec_statement(&brace_level);
                     // Done, restore state.
                     G__no_exec_compile = store_no_exec_compile;
                  }
               }
            }
         }
      }
      else {
         // -- We are going to skip the else clause.
         // Save state.
         int store_no_exec_compile = G__no_exec_compile;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
            // Flag that we are going to skip the else clause, but we are going to generate bytecode anyway.
            G__no_exec_compile = 1;
         }
         else {
            // Flag that we are going to skip the else clause.
            // FIXME: Should this be a save and restore?
            G__no_exec = 1;
         }
#else // G__ASM
         // Flag that we are going to skip the else clause.
         // FIXME: Should this be a save and restore?
         G__no_exec = 1;
#endif // G__ASM
         //
         //  Skip the else clause.
         //
         // Tell the parser to execute only one statement or block.
         int brace_level = 0;
         // Call the parser.
         G__exec_statement(&brace_level);
         // Restore state.
         G__no_exec_compile = store_no_exec_compile;
         G__no_exec = 0;
      }
      // If we were asked to return immediately, then do so.
      if (G__return != G__RETURN_NON) {
         // --
#ifdef G__ASM
         //
         //  Generate bytecode, backpatch the jump skipping the else clause.
         //
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "   %x: JMP assigned to %x (if stmt, jmp skipping the else clause)  %s:%d\n", asm_jumppointer - 1, G__asm_cp, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            // Backpatch the jump destination for skipping the else
            // clause, now that we know where it ends.
            G__asm_inst[asm_jumppointer] = G__asm_cp;
            // Remember where the if (...) {...} else {...} statement ended.
            G__asm_cond_cp = G__asm_cp;
         }
#endif // G__ASM
         G__ifswitch = store_ifswitch;
         free(condition);
         //fprintf(stderr, "---end if\n");
         return result;
      }
   }
   else {
      // -- No else clause, rewind input.
      //fprintf(stderr, "---no else\n");
      G__ifile.line_number = store_line_number;
      fsetpos(G__ifile.fp, &store_fpos);
      statement[0] = '\0';
      if (G__dispsource) {
         G__disp_mask = G__temp_read;
      }
      G__temp_read = 0;
   }
   // Restore state.
   G__no_exec = 0;
   //fprintf(stderr, "---end else\n");
#ifdef G__ASM
   //
   //  Generate bytecode, backpatch the jump skipping the else clause.
   //
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "   %x: JMP assigned to %x (if stmt, jmp skipping the else clause)  %s:%d\n", asm_jumppointer - 1, G__asm_cp, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      // Backpatch the jump destination for skipping the else
      // clause, now that we know where it ends.
      G__asm_inst[asm_jumppointer] = G__asm_cp;
      // Remember where the if (...) {...} else {...} statement ended.
      G__asm_cond_cp = G__asm_cp;
   }
#endif // G__ASM
   // Restore state.
   G__ifswitch = store_ifswitch;
   free(condition);
   // And return.
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 10);
   //   fprintf(stderr, "G__exec_if: peek ahead: '%s'\n", buf);
   //}
   //fprintf(stderr, "---end if ne: %d nec: %d ty: '%c'\n", G__no_exec, G__no_exec_compile, G__value_typenum(result));
   return result;
}


/***********************************************************************
* G__exec_else_if()
*
* Called by
*   G__exec_statement(&brace_level); 
*
***********************************************************************/
static G__value G__exec_else_if()
{
   // -- Skip an if statement during the parse.
   //
   // Note: G__no_exec is set to one when we are entered.
   //
   int iout;
   fpos_t store_fpos;
   int store_line_number;
   int c;
   char statement[10]; // only read comment and else.
   G__value result;
   int store_ifswitch = G__ifswitch;
   G__ifswitch = G__IFSWITCH;
#ifdef G__ASM
   if (!G__no_exec_compile)
      if (!G__xrefflag) {
         G__asm_noverflow = 0;
      }
#endif // G__ASM
   result = G__null;
   /******************************************************
    * G__no_exec==1 when this function is called.
    * nothing is executed in this function, but just skip
    * else if clause.
    *******************************************************/
   G__fignorestream(")");
   int brace_level = 0;
   G__exec_statement(&brace_level);
   fgetpos(G__ifile.fp, &store_fpos);
   store_line_number = G__ifile.line_number;
   // Reading else keyword.
   c = ' ';
   while (isspace(c)) {
      /* increment temp_read */
      c = G__fgetc();
      G__temp_read++;
      if (c == '/') {
         c = G__fgetc();
         /***********************
          *
          ***********************/
         switch (c) {
            case '*':
               if (G__skip_comment() == EOF) {
                  G__ifswitch = store_ifswitch;
                  return(G__null);
               }
               break;
            case '/':
               G__fignoreline();
               break;
            default:
               G__commenterror();
               break;
         }
         fgetpos(G__ifile.fp, &store_fpos);
         store_line_number = G__ifile.line_number;
         c = G__fgetc();
         G__temp_read = 1;
      }
      else if ('#' == c) {
         G__pp_command();
         c = G__fgetc();
         G__temp_read = 1;
      }
      if (c == EOF) {
         G__genericerror("Error: unexpected if() { } EOF");
         if (G__key != 0) system("key .cint_key -l execute");
         G__eof = 2;
         G__ifswitch = store_ifswitch;
         return(G__null);
      }
   }
   statement[0] = c;
   for (iout = 1; iout <= 3; ++iout) {
      c = G__fgetc();
      /* increment temp_read */
      G__temp_read++;
      if (c == EOF) {
         iout = 4;
         statement[0] = '\0';
      }
      statement[iout] = c;
   }
   statement[4] = '\0';
   if (!strcmp(statement, "else")) {
      // The else found. Skip elseclause.
      G__temp_read = 0;
      int local_brace_level = 0;
      result = G__exec_statement(&local_brace_level);
   }
   else {
      // no else  push back.
      G__ifile.line_number = store_line_number;
      fsetpos(G__ifile.fp, &store_fpos);
      statement[0] = '\0';
      if (G__dispsource) {
         G__disp_mask = G__temp_read;
      }
      G__temp_read = 0;
   }
   G__no_exec = 0;
   G__ifswitch = store_ifswitch;
   return result;
}

//______________________________________________________________________________
//
//  Iteration statements.  while, do, for
//

/***********************************************************************
* G__exec_do()
*
* Called by
*   G__exec_statement(&brace_level);   'do { } while();'
*
*  do { statement list } while(condition);
***********************************************************************/
//______________________________________________________________________________
static G__value G__exec_do()
{
   // -- Handle the do {...} while (...) statement.
   //
   // Note: The parse position is:
   //
   //      do {...} while (...);
   //        ^
   //
   // or:
   //
   //      do stmt; while (...);
   //        ^
   //
   // or:
   //
   //      do{...} while (...);
   //       ^
   //
   // so the next character to be read is either an open curly brace
   // or the first character of a statement.
   //
   // Note: The value of G__no_exec is always zero on entry.
   //
   // Flag that we are in a do-while statement.
   int store_ifswitch = G__ifswitch;
   G__ifswitch = G__DOWHILE;
   // Free any temporaries.
   // FIXME: This should not be here, temporaries should be freed at statement end, not at the beginning of one.
   if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
      G__free_tempobject();
   }
#ifdef G__ASM
   // Remember the bytecode generation state.
   int store_asm_noverflow = G__asm_noverflow;
#endif // G__ASM
   //
   //  Remember the position of the beginning of the body.
   //
   fpos_t store_fpos;
   fgetpos(G__ifile.fp, &store_fpos);
   int store_line_number = G__ifile.line_number;
#ifdef G__ASM
   //
   //  Start bytecode generation, if enabled.
   //
   // Rember the position of the beginning of the body for the jump at the end.
   int asm_start_pc = G__asm_cp;
   if (G__asm_loopcompile) {
      // -- We are generating bytecode for loops.
      //
      // Reset and initialize the bytecode generation environment.
      //
      if (!G__asm_noverflow) {
         // -- If bytecode generation is turned off, turn it on.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "\nLoop compile start (for do).  Erasing old bytecode and resetting pc.");
            G__printlinenum();
         }
#endif // G__ASM_DBG
         G__asm_noverflow = 1;
         G__clear_asm();
         // Rember the position of the beginning of the body for the jump at the end.
         asm_start_pc = G__asm_cp;
      }
      // Clear the data stack.
      G__asm_clear();
   }
#endif // G__ASM
#ifdef G__ASM
   //
   //  Setup the breakcontinue buffer.
   //
   struct G__breakcontinue_list* store_pbreakcontinue = 0;
   int allocflag = 0;
   if (G__asm_noverflow) {
      //fprintf(stderr, "G__exec_do: Begin alloc G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
      store_pbreakcontinue = G__alloc_breakcontinue_list();
      allocflag = 1;
      //fprintf(stderr, "G__exec_do: End alloc G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
   }
#endif // G__ASM
   //
   //  Peek ahead, we need to know whether or not the
   //  body of the do is a block or not.
   //
   int c = G__fgetspace_peek();
   int isblock = 0;
   if (c == '{') {
      isblock = 1;
   }
   //fprintf(stderr, "G__exec_do: isblock: %d\n", isblock);
   //
   //  Run the body once, unconditionally.
   //
   // Tell the parser to execute only one block or statement.
   int outer_brace_level = 0;
   // Call the parser.
   G__value result = G__exec_statement(&outer_brace_level);
   //fprintf(stderr, "G__exec_do: Just finished body.  G__value_typenum(result): %d isbreak: %d\n", G__value_typenum(result), G__value_typenum(result) == G__value_typenum(G__block_break));
   //
   //  Finished, check return code.
   //
   if (G__return != G__RETURN_NON) {
      // -- User asked for an immediate return.
#ifdef G__ASM
      // Free the breakcontinue buffer;
      if (allocflag) {
         //fprintf(stderr, "G__exec_do: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
         G__free_breakcontinue_list(store_pbreakcontinue);
         allocflag = 0;
         //fprintf(stderr, "G__exec_do: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
      }
      // If no bytecode errors, restore bytecode generation
      // flag, otherwise propagate error.
      if (G__asm_noverflow) {
         G__asm_noverflow = store_asm_noverflow;
      }
#endif // G__ASM
      // Restore the global structure flag for break/continue usage.
      G__ifswitch = store_ifswitch;
      // And we are done.
      return result;
   }
   //
   //  Did the body terminate on a flow of control statment?
   //
   int executed_break = 0;
   int executed_continue = 0;
   if (G__value_typenum(result) == G__value_typenum(G__block_break)) {
      // -- The body did a goto, break, or continue.
      //fprintf(stderr, "G__exec_do: Body exited with a break or continue.\n");
      if (result.ref == G__block_goto.ref) {
         // -- The body did a goto.
#ifdef G__ASM
         // Free the breakcontinue buffer, bytecode only.
         if (allocflag) {
            //fprintf(stderr, "G__exec_do: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
            G__free_breakcontinue_list(store_pbreakcontinue);
            allocflag = 0;
            //fprintf(stderr, "G__exec_do: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
         }
         // If no bytecode errors, restore bytecode generation
         // flag, otherwise propagate error.
         if (G__asm_noverflow) {
            G__asm_noverflow = store_asm_noverflow;
         }
#endif // G__ASM
         // Restore the containing loop/switch type for break/continue.
         G__ifswitch = store_ifswitch;
         // And we are done.
         return result;
      }
      if (result.obj.i == G__BLOCK_BREAK) {
         // -- The body did a break.
         //fprintf(stderr, "G__exec_do: Recognized a break on first iteration.\n");
         if (!G__asm_noverflow) {
            // -- The body did a break and bytecode compilation is disabled, return immediately.
            if (isblock) {
               // -- We are *not* in the do break; while (...) case.
               //fprintf(stderr, "G__exec_do: ignoring until next '}' because of a break statement.\n");
               //{
               //   char buf[128];
               //   G__fgetstream_peek(buf, 10);
               //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
               //}
               // Tell the parser to ignore statements.
               // Note: This is always zero on entry to exec_do.
               G__no_exec = 1;
               // Note: We must do this because we may expand a macro while skipping.
               //G__fignorestream("}");
               G__exec_statement(&outer_brace_level);
               // Note: This is always zero on entry to exec_do.
               G__no_exec = 0;
               //{
               //   char buf[128];
               //   G__fgetstream_peek(buf, 10);
               //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
               //}
            }
            // Skip to the end of the while clause before exiting.
            //fprintf(stderr, "G__exec_do: ignoring until next ';' because of a break statement.\n");
            //{
            //   char buf[128];
            //   G__fgetstream_peek(buf, 10);
            //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
            //}
            G__fignorestream(";");
            //{
            //   char buf[128];
            //   G__fgetstream_peek(buf, 10);
            //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
            //}
#ifdef G__ASM
            // Free the breakcontinue buffer, bytecode only.
            if (allocflag) {
               //fprintf(stderr, "G__exec_do: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
               G__free_breakcontinue_list(store_pbreakcontinue);
               allocflag = 0;
               //fprintf(stderr, "G__exec_do: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
            }
            // If no bytecode errors, restore bytecode generation
            // flag, otherwise propagate error.
            if (G__asm_noverflow) {
               G__asm_noverflow = store_asm_noverflow;
            }
#endif // G__ASM
            // Restore the containing loop type flag for break/continue.
            G__ifswitch = store_ifswitch;
            // And we are done.
            return G__null;
         }
         // -- The body did a break and bytecode compilation is enabled.
         executed_break = 1;
         if (isblock) {
            // Parse the rest of the body, generating code.
            int store_no_exec_compile = G__no_exec_compile;
            G__no_exec_compile = 1;
            // Tell parser to go until a right curly brace is seen.
            // Call the parser.
            G__exec_statement(&outer_brace_level);
            G__no_exec_compile = store_no_exec_compile;
         }
      }
      else if (result.obj.i == G__BLOCK_CONTINUE) {
         // -- The body did a continue.
         //fprintf(stderr, "G__exec_do: Recognized a continue on first iteration.\n");
         executed_continue = 1;
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
            if (isblock) {
               // -- The body is a block.
               // Parse the rest of the block, generating code.
               // Flag that we are only going to generate bytecode.
               int store_no_exec_compile = G__no_exec_compile;
               G__no_exec_compile = 1;
               // Tell parser to go until a right curly brace is seen.
               // Call the parser.
               G__exec_statement(&outer_brace_level);
               // Restore state.
               G__no_exec_compile = store_no_exec_compile;
            }
         }
         else {
            // -- We are *not* generating bytecode.
            if (isblock) {
               // -- The body is a block.
               //fprintf(stderr, "G__exec_do: ignoring until next '}' because of a continue.\n");
               //{
               //   char buf[128];
               //   G__fgetstream_peek(buf, 10);
               //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
               //}
               // Tell the parser to ignore statements.
               // Note: This is always zero on entry to exec_if.
               G__no_exec = 1;
               // Note: We must do this because we may expand a macro while skipping.
               //G__fignorestream("}");
               G__exec_statement(&outer_brace_level);
               // Note: This is always zero on entry to exec_if.
               G__no_exec = 0;
               //{
               //   char buf[128];
               //   G__fgetstream_peek(buf, 10);
               //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
               //}
            }
         }
      }
      else {
         G__fprinterr(G__serr, "G__exec_do: Body exited with a break status, but not from break or continue.\n");
      }
   }
   //
   //  do { ... } while (...);
   //            ^ 
   //
   //  Scan the while keyword.
   //fprintf(stderr, "G__exec_do: Scan the while keyword.\n");
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 10);
   //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
   //}
   G__StrBuf condition_sb(G__ONELINE);
   char *condition = condition_sb;
   G__fgetstream(condition, "(");
   if (strcmp(condition, "while")) {
      G__fprinterr(G__serr, "Syntax error: do {} %s(); Should be do {} while (); FILE: %s LINE: %d\n", condition, G__ifile.name, G__ifile.line_number);
   }
   //
   //  do { ... } while (...);
   //                    ^
   //
   //  Scan the do loop condition.
   //fprintf(stderr, "G__exec_do: Scan the do loop condition.\n");
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 10);
   //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
   //}
   G__fgetstream(condition, ")");
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 10);
   //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
   //}
   //
   //  do { ... } while (...);
   //                       ^
   //--
   //
   //  Check for a breakpoint.
   //
   if (G__breaksignal) {
      G__break = 0;
      G__setdebugcond();
      G__pause();
      if (G__return > G__RETURN_NORMAL) {
         // -- The user requested an immediate return.
#ifdef G__ASM
         // Free the breakcontinue buffer;
         if (allocflag) {
            //fprintf(stderr, "G__exec_do: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
            G__free_breakcontinue_list(store_pbreakcontinue);
            allocflag = 0;
            //fprintf(stderr, "G__exec_do: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
         }
         // If no bytecode errors, restore bytecode generation
         // flag, otherwise propagate error.
         if (G__asm_noverflow) {
            G__asm_noverflow = store_asm_noverflow;
         }
#endif // G__ASM
         // Restore the global structure flag for break/continue usage.
         G__ifswitch = store_ifswitch;
         // And return now.
         return G__null;
      }
   }
#ifdef G__ASM
   //
   //  Remember the position of the do loop condition test
   //  as the target for a continue statement.
   //
   int store_asm_cp = G__asm_cp;
#endif // G__ASM
   //
   //  If we have already terminated with a break,
   //  just compile the condition, do not test it.
   //
   int store_no_exec_compile = G__no_exec_compile;
   if (executed_break) {
      G__no_exec_compile = 1;
   }
   //
   //  Test the loop condition.
   //
   //fprintf(stderr, "G__exec_do: Testing condition '%s' nec: %d\n", condition, G__no_exec_compile);
   int cond = G__test(condition);
   //fprintf(stderr, "G__exec_do: Testing condition result: %d\n", cond);
   //
   //  If we have already terminated with a break,
   //  restore state.
   //
   if (executed_break) {
      G__no_exec_compile = store_no_exec_compile;
   }
   //
   //  Free any temporaries generated during the test of the condition.
   //
   if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
      G__free_tempobject();
   }
#ifdef G__ASM
   //
   //  Generate code to exit do block after test,
   //  code to repeat do block, and backpatch any
   //  break or continue jumps in the generated
   //  bytecode for the body.
   //
   //fprintf(stderr, "G__exec_do: Finished test of loop condition: G__asm_noverflow: %d  %s:%d\n", G__asm_noverflow, __FILE__, __LINE__);
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: CNDJMP %x (for exiting a do block)  %s:%d\n", G__asm_cp, G__asm_dt, G__asm_cp + 4, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__CNDJMP;
      G__asm_inst[G__asm_cp+1] = G__asm_cp + 4;
      G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: JMP %x (to repeat a do block)  %s:%d\n", G__asm_cp, G__asm_dt, asm_start_pc, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__JMP;
      G__asm_inst[G__asm_cp+1] = asm_start_pc;
      G__inc_cp_asm(2, 0);
      // Backpatch break and continue jumps.
      if (allocflag) {
         //fprintf(stderr, "G__exec_do: Begin set destination break: %x continue: %x G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__asm_cp, store_asm_cp, G__pbreakcontinue, store_pbreakcontinue);
         G__set_breakcontinue_destination(G__asm_cp /* break dest */, store_asm_cp /* continue dest */, store_pbreakcontinue);
         //fprintf(stderr, "G__exec_do: End set destination G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
         allocflag = 0;
      }
      G__asm_inst[G__asm_cp] = G__RETURN;
      // Turn off bytecode generation.
      if (!G__xrefflag) {
         G__asm_noverflow = 0;
      }
      if (G__asm_loopcompile > 1) {
         // -- We are optimizing compiled bytecode in loops.
         G__asm_optimize(&asm_start_pc);
      }
      G__asm_noverflow = 1;
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "\nBytecode loop compilation successful. (for do)");
         G__printlinenum();
      }
   }
#endif // G__ASM
#ifdef G__ASM
   //
   //  We are done with the break/continue buffer, free it.
   //
   if (allocflag) {
      //fprintf(stderr, "G__exec_do: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
      G__free_breakcontinue_list(store_pbreakcontinue);
      allocflag = 0;
      //fprintf(stderr, "G__exec_do: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
   }
#endif // G__ASM
   //
   //  If we are just generating bytecode, or we have
   //  already executed a break, then do not repeat the body.
   //
   if (G__no_exec_compile || executed_break) {
      cond = 0;
   }
   //
   //  Continue executing body while the condition is true.
   //
   if (cond) {
      // -- We are to execute the body at least once more.
      if (G__asm_noverflow) {
         // -- We have generated bytecode, execute it.
         // Bytecode generation must be off here or we will overwrite the generated code!
         G__asm_noverflow = 0;
         // Execute the bytecode, beware this may call parser routines, make sure bytecode generation is off.
         G__exec_asm(asm_start_pc, 0, &result, 0);
         // Re-enable bytecode generation.
         G__asm_noverflow = 1;
         if (G__return != G__RETURN_NON) {
            // -- The user requested an immediate exit.
            // If no bytecode errors, restore bytecode generation
            // flag, otherwise propagate error.
            if (G__asm_noverflow) {
               G__asm_noverflow = store_asm_noverflow;
            }
            // Restore the global structure flag for break/continue usage.
            G__ifswitch = store_ifswitch;
            // And return now.
            return result;
         }
      }
      else {
         // -- We are going to interpret the body.
         while (cond) {
            // -- Execute the body again.
            // Reset continue statement flag.
            executed_continue = 0;
            //
            //  Reset the parse to the beginning of the body.
            //
            G__ifile.line_number = store_line_number;
            fsetpos(G__ifile.fp, &store_fpos);
            //
            //  If we are tracing from prerun, print the line number on a new line.
            //
            //  Note: This is what the get character function does on end-of-line.
            //
            if (G__debug) {
               G__fprinterr(G__serr, "\n%-5d", G__ifile.line_number);
            }
            //
            //  Run the body.
            //
            // Tell the parser to execute only one statement or block.
            int brace_level = 0;
            // Call the parser.
            result = G__exec_statement(&brace_level);
            // Check if the user requested an immediate return.
            if (G__return != G__RETURN_NON) {
               // -- The user requested an immediate return.
               // If no bytecode errors, restore bytecode generation
               // flag, otherwise propagate error.
               if (G__asm_noverflow) {
                  G__asm_noverflow = store_asm_noverflow;
               }
               // Restore the global structure flag for break/continue usage.
               G__ifswitch = store_ifswitch;
               // And return now.
               return result;
            }
            //
            //  Check if the body terminated early due to a break, continue, or goto.
            //
            if (G__value_typenum(result) == G__value_typenum(G__block_break)) {
               // -- The body executed a goto, break, or continue.
               //fprintf(stderr, "G__exec_do: Body exited with a break or continue on second or greater iteration.\n");
               if (result.ref == G__block_goto.ref) {
                  // -- The body executed a goto.
#ifdef G__ASM
                  // If no bytecode errors, restore bytecode generation
                  // flag, otherwise propagate error.
                  if (G__asm_noverflow) {
                     G__asm_noverflow = store_asm_noverflow;
                  }
#endif // G__ASM
                  // Restore the containing loop type flag for break/continue.
                  G__ifswitch = store_ifswitch;
                  // And return now.
                  return result;
               }
               if (result.obj.i == G__BLOCK_BREAK) {
                  // -- The body executed a break statement.
                  //fprintf(stderr, "G__exec_do: Recognized a break on second or greater iteration.\n");
                  // If the body was a block, skip the rest of it.
                  if (isblock) {
                     //fprintf(stderr, "G__exec_do: ignoring until next '}' because of a break statement.\n");
                     //{
                     //   char buf[128];
                     //   G__fgetstream_peek(buf, 10);
                     //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
                     //}
                     // Tell the parser to ignore statements.
                     // Note: This is always zero on entry to exec_do.
                     G__no_exec = 1;
                     // Note: We must do this because we may expand a macro while skipping.
                     //G__fignorestream("}");
                     G__exec_statement(&brace_level);
                     // Note: This is always zero on entry to exec_do.
                     G__no_exec = 0;
                     //{
                     //   char buf[128];
                     //   G__fgetstream_peek(buf, 10);
                     //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
                     //}
                  }
                  // Skip to the end of the while clause before exiting.
                  //fprintf(stderr, "G__exec_do: ignoring until next ';' because of a break statement.\n");
                  //{
                  //   char buf[128];
                  //   G__fgetstream_peek(buf, 10);
                  //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
                  //}
                  G__fignorestream(";");
                  //{
                  //   char buf[128];
                  //   G__fgetstream_peek(buf, 10);
                  //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
                  //}
#ifdef G__ASM
                  // If no bytecode errors, restore bytecode generation
                  // flag, otherwise propagate error.
                  if (G__asm_noverflow) {
                     G__asm_noverflow = store_asm_noverflow;
                  }
#endif // G__ASM
                  // Restore the containing loop type flag for break/continue.
                  G__ifswitch = store_ifswitch;
                  // And we are done.
                  return G__null;
               }
               else if (result.obj.i == G__BLOCK_CONTINUE) {
                  // -- The body executed a continue.
                  //fprintf(stderr, "G__exec_do: Recognized a continue on second or greater iteration.\n");
                  executed_continue = 1;
                  if (isblock) {
                     // -- The body is a block.
                     //fprintf(stderr, "G__exec_do: ignoring until next '}' because of a continue on second or greater iteration.\n");
                     //{
                     //   char buf[128];
                     //   G__fgetstream_peek(buf, 10);
                     //   fprintf(stderr, "G__exec_do: peek ahead: '%s'\n", buf);
                     //}
                     // Tell the parser to ignore statements.
                     // Note: This is always zero on entry to exec_if.
                     G__no_exec = 1;
                     // Note: We must do this because we may expand a macro while skipping.
                     //G__fignorestream("}");
                     G__exec_statement(&brace_level);
                     // Note: This is always zero on entry to exec_if.
                     G__no_exec = 0;
                     //{
                        //char buf[128];
                        //G__fgetstream_peek(buf, 10);
                        //fprintf(stderr, "G__exec_do: peek ahead after skipping for continue: '%s'\n", buf);
                     //}
                  }
               }
               else {
                  G__fprinterr(G__serr, "G__exec_do: Body exited with a break status, but not from break or continue.\n");
               }
            }
            //
            //  Check for a breakpoint.
            //
            if (G__breaksignal) {
               G__break = 0;
               G__setdebugcond();
               G__pause();
               if (G__return > G__RETURN_NORMAL) {
                  // -- The user requested an immediate return.
#ifdef G__ASM
                  // If no bytecode errors, restore bytecode generation
                  // flag, otherwise propagate error.
                  if (G__asm_noverflow) {
                     G__asm_noverflow = store_asm_noverflow;
                  }
#endif // G__ASM
                  // Restore the containing loop type flag for break/continue.
                  G__ifswitch = store_ifswitch;
                  // And we are done.
                  return G__null;
               }
            }
            // Test the do loop condition again.
            //fprintf(stderr, "G__exec_do: Testing condition '%s'\n", condition);
            cond = G__test(condition);
            //fprintf(stderr, "G__exec_do: Testing condition result: %d\n", cond);
         }
      }
   }
   //
   //  At this point we are done with execution.
   //
   // Skip any whitespace after the condition to the end of the statement.
   G__fignorestream(";");
   //
   //  If we are tracing from prerun, print a ';' to mark the end of the statement.
   //
   if (G__debug) {
      G__fprinterr(G__serr, ";");
   }
   // Restore state.
#ifdef G__ASM
   // If no bytecode errors, restore bytecode generation
   // flag, otherwise propagate error.
   if (G__asm_noverflow) {
      G__asm_noverflow = store_asm_noverflow;
   }
#endif // G__ASM
   // Restore the global structure flag for break/continue usage.
   G__ifswitch = store_ifswitch;
   // And we are done.
   result = G__null;
   return result;
}


/***********************************************************************
* G__exec_for()
*
* Called by
*   G__exec_statement(&brace_level); 
*
***********************************************************************/
static G__value G__exec_for()
{
   // -- Handle the for (...; ...; ...) statement.
   //
   // The parser state is:
   //
   //      for (...; ...; ...) {...}
   //          ^
   //
   // Note: G__no_exec is always zero when we are entered.
   //
   // Change ifswitch state.
   int store_ifswitch = G__ifswitch;
   G__ifswitch = G__DOWHILE;
   //
   //  Execute the loop initializer.
   //
   // Tell the parser to execute only one statement.
   int brace_level = 0;
   // Call the parser.
   G__exec_statement(&brace_level);
   if (G__return > G__RETURN_NORMAL) {
      // -- Exit immediately if the user requested it during tracing.
      G__ifswitch = store_ifswitch;
      return G__null;
   }
   char* condition = (char*) malloc(G__LONGLINE);
   int c = G__fgetstream(condition, ";)");
   if (c == ')') {
      // -- Syntax error, the third clause of the for was missing.
      G__genericerror("Error: for statement syntax error");
      G__ifswitch = store_ifswitch;
      free(condition);
      return G__null;
   }
   // If there is no condition text, make it always true, as the standard requires.
   if (!condition[0]) {
      strcpy(condition, "1");
   }
   // FIXME: Why do we make the condition buffer bigger?
   condition = (char*) realloc(condition, strlen(condition) + 10);
   //
   //  Collect the third clause of the for,
   //  separating it on commas.
   //
   G__StrBuf foractionbuf_sb(G__ONELINE);
   char *foractionbuf = foractionbuf_sb;
   char* foraction[10];
   int naction = 0;
   char* p = foractionbuf;
   do {
      // -- Collect one clause of a comma operator expression.
      // Scan until a comma or the end of the head of the for.
      c = G__fgetstream(p, "),");
      if (G__return > G__RETURN_NORMAL) {
         // FIXME: Is this error message correct?
         G__fprinterr(G__serr, "Error: for statement syntax error. ';' needed\n");
         // Restore the ifswitch state.
         G__ifswitch = store_ifswitch;
         // Cleanup malloc'ed memory.
         free(condition);
         // And exit in error.
         return G__null;
      }
      // Collect this clause.
      foraction[naction++] = p;
      // Move past scanned text in buffer.
      p += strlen(p) + 1;
   }
   while (c != ')');
   //
   //  Execute the body of the loop.
   //
   G__value result  = G__exec_loop(0, condition, naction, foraction);
   // Cleanup malloc'ed memory.
   free(condition);
   // Restore the ifswitch state.
   G__ifswitch = store_ifswitch;
   // And we are done.
   return result;
}

/***********************************************************************
* G__exec_while()
*
* Called by
*   G__exec_statement(&brace_level); 
*
***********************************************************************/
static G__value G__exec_while()
{
   // -- Handle the while (...) do {...} statement.
   //
   // The parser state is:
   //
   //      while (...) do {...}
   //            ^
   //
   // Note: G__no_exec is always zero when we are entered.
   //
   //--
   //
   //  Set the ifswitch state.
   //
   int store_ifswitch = G__ifswitch;
   G__ifswitch = G__DOWHILE;
   //
   //  Scan in the while condition.
   //
   char* condition = (char*) malloc(G__LONGLINE);
   G__fgetstream(condition, ")");
   // FIXME: Why do we make the condition buffer bigger?
   condition = (char*) realloc((void*) condition, strlen(condition) + 10);
   //
   //  Execute the body of the loop.
   //
   G__value result = G__exec_loop(0, condition, 0, 0);
   // Cleanup malloc'ed memory.
   free(condition);
   // Restore the ifswitch state.
   G__ifswitch = store_ifswitch;
   // And we are done.
   return result;
}

/***********************************************************************
* G__exec_loop()
*
*
***********************************************************************/
static G__value G__exec_loop(char *forinit,char *condition,int naction,char **foraction)   {
   // -- Execute a loop, handles most of for, and while.
   //
   // Note: G__no_exec is always zero when we are entered.
   //
   // Note: The parser state is:
   //
   //      for (...;...;...) ...
   //                      ^
   // or:
   //
   //      while (...) ...
   //                ^
   //
   //fprintf(stderr, "G__exec_loop: at begin, G__no_exec_compile: %d\n", G__no_exec_compile);
#ifdef G__ASM
   int store_asm_noverflow = 0;
   int store_asm_cp = 0;
   struct G__breakcontinue_list* store_pbreakcontinue = 0;
   int asm_start_pc = 0;
   int asm_jumppointer = 0;
   int allocflag = 0;
#endif // G__ASM
   //
   // Allocate a return value.
   //
   G__value result;
   //
   //  Allow a single bytecode error message.
   //
#ifdef G__ASM_DBG
   int dispstat = 0;
#endif
   //
   //  Remember old if/switch state, change to dowhile state.
   //
   int store_ifswitch = G__ifswitch;
   G__ifswitch = G__DOWHILE;
#ifdef G__ASM
   //
   //  Save bytecode compilation state.
   //
   store_asm_noverflow = G__asm_noverflow;
#endif // G__ASM
   //
   //  Pause before starting loop, if requested.
   //
   int largestep = 0;
   {
      int junk1 = 0;
      if (G__breaksignal) {
         int ret = G__beforelargestep(condition, &junk1, &largestep);
         if (ret > 1) {
            G__ifswitch = store_ifswitch;
            return G__null;
         }
      }
   }
   //
   //  Remember current file position and line number.
   //
   fpos_t store_fpos;
   fgetpos(G__ifile.fp, &store_fpos);
   int store_line_number = G__ifile.line_number;
   //
   //  Look ahead to see if loop body is a block or not.
   //
   int isblock = 0;
   int c = G__fgetspace_peek();
   if (c == '{') {
      isblock = 1;
   }
   //
   //  Start loop bytecode compilation, if requested.
   //
#ifdef G__ASM
   if (G__asm_loopcompile) {
      // -- We are generating bytecode for loops.
      //
      // Reset and initialize the bytecode generation environment.
      //
      if (!G__asm_noverflow) {
         // -- If bytecode generation is turned off, turn it on.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "\nLoop compile start (for a for or while).  Erasing old bytecode and resetting pc.");
            G__printlinenum();
         }
#endif // G__ASM_DBG
         G__asm_noverflow = 1;
         G__clear_asm();
      }
   }
#endif // G__ASM
   //
   //  If there is a forinit clause, evaluate it.
   //
   if (forinit) {
      //fprintf(stderr, "G__exec_loop: begin forinit expression evaluation ...\n");
      G__getexpr(forinit);
      //fprintf(stderr, "G__exec_loop: end   forinit expression evaluation.\n");
   }
   //
   //  Free any generated temporaries.
   //
   if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
      G__free_tempobject();
   }
#ifdef G__ASM
   //
   //  Remember start of loop body if generating bytecode.
   //
   if (G__asm_noverflow) {
      asm_start_pc = G__asm_cp;
   }
#endif // G__ASM
#ifdef G__ASM
   //
   //  Allocate a breakcontinue buffer, if generating bytecode.
   //
   if (G__asm_noverflow) {
      //fprintf(stderr, "G__exec_loop: Begin alloc G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
      store_pbreakcontinue = G__alloc_breakcontinue_list();
      allocflag = 1;
      //fprintf(stderr, "G__exec_loop: End alloc G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
   }
#endif // G__ASM
   //
   //  Make our first test of the loop condition.
   //
   //fprintf(stderr, "G__exec_loop: begin first test of loop condition ...\n");
   int cond = G__test(condition);
   //fprintf(stderr, "G__exec_loop: end   first test of loop condition.\n");
   //
   //  Free any generated temporaries.
   //
   if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
      G__free_tempobject();
   }
   //
   //  If we are just trying to bytecode compile,
   //  then force at least one execution of the loop.
   //
   if (G__no_exec_compile) {
      cond = 1;
   }
   // Save state.
   int store_no_exec_compile = G__no_exec_compile;
#ifdef G__ASM
   //
   //  If we are generating bytecode and
   //  the first test of the loop condition
   //  was false, force at least one
   //  execution of the loop so the code
   //  is seen.  We may be inside another
   //  loop which is generating code.
   //
   if (G__asm_noverflow && !cond) {
      G__no_exec_compile = 1;
      cond = 1;
   }
   //
   //  Free the breakcontinue buffer if we are not going
   //  to execute the loop.
   //
   if (!cond && allocflag) {
      //fprintf(stderr, "G__exec_loop: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
      G__free_breakcontinue_list(store_pbreakcontinue);
      allocflag = 0;
      //fprintf(stderr, "G__exec_loop: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
   }
#endif // G__ASM
   //
   //  Execute the loop while the condition is true.
   //
   //fprintf(stderr, "G__exec_loop: cond: %d\n", cond);
   int executed_break = 0;
   int executed_continue = 0;
   if (!cond) {
      // -- We are not going to run the body, skip it.
      // Tell the parser we are skipping code.
      G__no_exec = 1;
      // Tell the parser to execute only one statement or block.
      int brace_level = 0;
      // And call the parser.
      result = G__exec_statement(&brace_level);
      // Restore state.
      G__no_exec = 0;
   }
   else {
      while (cond) {
         // -- Execute body of loop.
         executed_continue = 0;
#ifdef G__ASM
         //
         //  Generate an G__CNDJMP <dst> bytecode instruction.
         //
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: CNDJMP (for or while, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__CNDJMP;
            asm_jumppointer = G__asm_cp + 1;
            G__inc_cp_asm(2, 0);
            G__asm_clear();
         }
#endif // G__ASM
         //
         // Rewind file to beginning of loop body.
         //
         G__ifile.line_number = store_line_number;
         fsetpos(G__ifile.fp, &store_fpos);
         //
         //  Execute the loop body.
         //
         //fprintf(stderr, "G__exec_loop: begin execution of body ...\n");
         // Tell the parser to execute only one statement or block.
         int brace_level = 0;
         // And call the parser.
         result = G__exec_statement(&brace_level);
         //fprintf(stderr, "G__exec_loop: end execution of body.\n");
         //
         //  Check if the user asked for an immediate exit.
         //
         if (G__return != G__RETURN_NON) {
            // -- The user requested that we exit immediately.
#ifdef G__ASM
            if (allocflag) {
               //fprintf(stderr, "G__exec_loop: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
               G__free_breakcontinue_list(store_pbreakcontinue);
               //fprintf(stderr, "G__exec_loop: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
            }
            // If no bytecode errors, restore bytecode generation
            // flag, otherwise propagate error.
            if (G__asm_noverflow) {
               G__asm_noverflow = store_asm_noverflow;
            }
#endif // G__ASM
            G__ifswitch = store_ifswitch;
            return result;
         }
         //
         //  Check to see if the body was exited
         //  by a flow control statement.
         //
         if (G__value_typenum(result) == G__value_typenum(G__block_break)) {
            switch (result.obj.i) {
               case G__BLOCK_BREAK:
                  // -- Body exited by either a break or a goto statement.
                  //
                  //  Pause on break, if requested.
                  //
                  if (largestep) {
                     G__afterlargestep(&largestep);
                  }
                  if (result.ref == G__block_goto.ref) {
                     // -- The body did a goto, return immediately.
#ifdef G__ASM
                     if (allocflag) {
                        //fprintf(stderr, "G__exec_loop: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
                        G__free_breakcontinue_list(store_pbreakcontinue);
                        //fprintf(stderr, "G__exec_loop: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
                     }
                     // If no bytecode errors, restore bytecode generation
                     // flag, otherwise propagate error.
                     if (G__asm_noverflow) {
                        G__asm_noverflow = store_asm_noverflow;
                     }
#endif // G__ASM
                     // Restore state.
                     G__ifswitch = store_ifswitch;
                     //fprintf(stderr, "G__exec_loop: End.  The body did a goto.\n");
                     // And return, returning the goto result to our caller.
                     return result;
                  }
                  else if (!G__asm_noverflow) {
                     // -- The body did a break and bytecode compilation is disabled, return immediately.
                     if (isblock) {
                        //fprintf(stderr, "G__exec_loop: Body did a break, ignoring code until next '}'.\n");
                        //{
                        //   char buf[128];
                        //   G__fgetstream_peek(buf, 10);
                        //   fprintf(stderr, "G__exec_loop: peek ahead: '%s'\n", buf);
                        //}
                        // Tell the parser to ignore statements.
                        // Note: This is always zero on entry to exec_loop.
                        G__no_exec = 1;
                        // Note: We must do this because we may expand a macro while skipping.
                        //G__fignorestream("}");
                        G__exec_statement(&brace_level);
                        // Note: This is always zero on entry to exec_loop.
                        G__no_exec = 0;
                        //{
                        //   char buf[128];
                        //   G__fgetstream_peek(buf, 10);
                        //   fprintf(stderr, "G__exec_loop: peek ahead: '%s'\n", buf);
                        //}
                     }
#ifdef G__ASM
                     if (allocflag) {
                        //fprintf(stderr, "G__exec_loop: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
                        G__free_breakcontinue_list(store_pbreakcontinue);
                        //fprintf(stderr, "G__exec_loop: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
                     }
                     // If no bytecode errors, restore bytecode generation
                     // flag, otherwise propagate error.
                     if (G__asm_noverflow) {
                        G__asm_noverflow = store_asm_noverflow;
                     }
#endif // G__ASM
                     // Restore state.
                     G__ifswitch = store_ifswitch;
                     //fprintf(stderr, "G__exec_loop: End.  The body did a break.\n");
                     // And return now, we do not propagate the break status upwards.
                     return G__null;
                  }
                  // The body did a break, and we are generating bytecode.
                  executed_break = 1;
                  // We do not propagate the break status upwards.
                  result = G__null;
                  if (isblock) {
                     // -- Parse the rest of the body, generating code.
                     //fprintf(stderr, "G__exec_loop: Body did a break, generating code for rest of body block.\n"); 
                     int local_store_no_exec_compile = G__no_exec_compile;
                     G__no_exec_compile = 1;
                     // Tell parser to go until a right curly brace is seen.
                     // Call the parser.
                     G__exec_statement(&brace_level);
                     G__no_exec_compile = local_store_no_exec_compile;
                  }
                  break;
               case G__BLOCK_CONTINUE:
                  // -- The body did a continue.
                  //fprintf(stderr, "G__exec_loop: The body did a continue.\n");
                  executed_continue = 1;
                  // We do not propagate the continue status.
                  result = G__null;
                  if (G__asm_noverflow) {
                     // -- We are generating bytecode.
                     if (isblock) {
                        // -- Parse the rest of the body, generating code.
                        //fprintf(stderr, "G__exec_loop: Body did a continue, generating code for rest of body block.\n"); 
                        int local_store_no_exec_compile = G__no_exec_compile;
                        G__no_exec_compile = 1;
                        // Tell parser to go until a right curly brace is seen.
                        // And call parser.
                        G__exec_statement(&brace_level);
                        // Restore state.
                        G__no_exec_compile = local_store_no_exec_compile;
                     }
                  }
                  else {
                     // -- We are *not* generating bytecode.
                     if (isblock) {
                        // -- The body is a block.
                        //fprintf(stderr, "G__exec_loop: Body did a continue, ignoring code until next '}'.\n");
                        //{
                        //   char buf[128];
                        //   G__fgetstream_peek(buf, 10);
                        //   fprintf(stderr, "G__exec_loop: peek ahead: '%s'\n", buf);
                        //}
                        // Tell the parser to ignore statements.
                        // Note: This is always zero on entry to exec_loop.
                        G__no_exec = 1;
                        // Note: We must do this because we may expand a macro while skipping.
                        //G__fignorestream("}");
                        G__exec_statement(&brace_level);
                        // Note: This is always zero on entry to exec_loop.
                        G__no_exec = 0;
                        //{
                        //   char buf[128];
                        //   G__fgetstream_peek(buf, 10);
                        //   fprintf(stderr, "G__exec_loop: peek ahead: '%s'\n", buf);
                        //}
                     }
                  }
                  break;
            }
         }
#ifdef G__ASM
         store_asm_cp = G__asm_cp;
#endif // G__ASM
         int local_store_no_exec_compile = G__no_exec_compile;
         if (executed_break) {
            G__no_exec_compile = 1;
         }
         //
         //  Execute the bottom of loop expression.
         //
         if (naction) {
            //fprintf(stderr, "G__exec_loop: begin executing loop actions ...\n");
            for (int i = 0; i < naction; ++i) {
               G__getexpr(foraction[i]);
            }
            //fprintf(stderr, "G__exec_loop: end executing loop actions.\n");
         }
         if (executed_break) {
            G__no_exec_compile = local_store_no_exec_compile;
         }
#ifdef G__ASM
         //
         //  If bytecode generation has succeeded up to this point,
         //  finish off bytecode generation and run it, otherwise
         //  print an error message and continue interpreting.
         if (G__asm_noverflow) {
            //
            //  Generate a G__JMP <dst> bytecode instruction.
            //
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: JMP %x (for or while, jump to begin of loop body) %s:%d\n", G__asm_cp, G__asm_dt, asm_start_pc, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__JMP;
            G__asm_inst[G__asm_cp+1] = asm_start_pc;
            G__inc_cp_asm(2, 0);
            //
            // Check breakcontinue buffer and assign destination.
            //
            //    break:    G__asm_cp
            //    continue: store_asm_cp
            //
            //fprintf(stderr, "G__exec_loop: break: %x continue: %x allocflag: %d G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__asm_cp, store_asm_cp, allocflag, G__pbreakcontinue, store_pbreakcontinue);
            if (allocflag) {
               //fprintf(stderr, "G__exec_loop: Begin set destination break: %x continue: %x allocflag: %d G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__asm_cp, store_asm_cp, allocflag, G__pbreakcontinue, store_pbreakcontinue);
               G__set_breakcontinue_destination(G__asm_cp, store_asm_cp, store_pbreakcontinue);
               allocflag = 0;
               //fprintf(stderr, "G__exec_loop: End set destination G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
            }
            //
            //  Generate a G__RETURN bytecode instruction.
            //
            G__asm_inst[G__asm_cp] = G__RETURN;
            G__asm_inst[asm_jumppointer] = G__asm_cp;
            // Stop generating bytecode.
            G__asm_noverflow = 0;
            //
            //  Optimize generated bytecode.
            //
            if (G__asm_loopcompile > 1) {
               //fprintf(stderr, "\nG__exec_loop: Begin bytecode optimize.\n");
               G__asm_optimize(&asm_start_pc);
               //fprintf(stderr, "\nG__exec_loop: End bytecode optimize.\n");
            }
            // FIXME: Are we ignoring bytecode generation errors from G__asm_optimize here?
            // Reenable bytecode generation.
            G__asm_noverflow = 1;
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "\nBytecode loop compilation successful. (for for or while)");
               G__printlinenum();
            }
            //
            //  Exit loop on a break.
            //
            if (executed_break) {
               break;
            }
            //
            //  If we are not here just to generate bytecode,
            //  then execute the bytecode for the rest of
            //  the loop iterations and return.
            //
            //fprintf(stderr, "G__exec_loop: G__no_exec_compile: %d\n", G__no_exec_compile);
            if (!G__no_exec_compile) {
               //fprintf(stderr, "G__exec_loop: Beginning to execute bytecode.\n", G__no_exec_compile);
               // Note: Bytecode generation must be off here, or we will overwrite the generated code!
               // Disable bytecode generation.
               G__asm_noverflow = 0;
               /*int status =*/ G__exec_asm(asm_start_pc, 0, &result, 0);
               // Reenable bytecode generation.
               G__asm_noverflow = 1;
               if (G__return != G__RETURN_NON) {
                  // -- User requested that we return immediately.
                  // If no bytecode errors, restore bytecode generation
                  // flag, otherwise propagate error.
                  if (G__asm_noverflow) {
                     G__asm_noverflow = store_asm_noverflow;
                  }
                  G__ifswitch = store_ifswitch;
                  return result;
               }
            }
            //
            //  Exit loop, we are done.
            //
            break;
         }
         else {
            //
            //  Print an error message if we have not yet.
            //
#ifdef G__ASM_DBG
            if (G__asm_dbg && !dispstat) {
               G__fprinterr(G__serr, "\nG__exec_loop: Bytecode loop compilation failed.  %s:%d", __FILE__, __LINE__);
               G__printlinenum();
               // Turn off any further failure messages.
               dispstat = 1;
            }
#endif // G__ASM_DBG
            if (allocflag) {
               //fprintf(stderr, "G__exec_loop: Begin free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
               G__free_breakcontinue_list(store_pbreakcontinue);
               allocflag = 0;
               //fprintf(stderr, "G__exec_loop: End free G__pbreakcontinue: %p store_pbreakcontinue: %p\n", G__pbreakcontinue, store_pbreakcontinue);
            }
         }
#endif // G__ASM
         //
         //  Exit loop on a break.
         //
         if (executed_break) {
            break;
         }
         //
         //  If we were executing the loop body
         //  just for bytecode compilation, force
         //  the loop to exit now.
         //
         //  Otherwise, test the loop condition.
         //
         if (G__no_exec_compile) {
            // -- Force the loop to exit now.
            cond = 0;
         }
         else {
            // -- Test the loop condition.
            //fprintf(stderr, "G__exec_loop: Testing condition '%s'.\n", condition);
            cond = G__test(condition);
            //fprintf(stderr, "G__exec_loop: Testing condition result: %d.\n", cond);
            // Free any generated temporaries.
            if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
               G__free_tempobject();
            }
         }
      }
   }
   // Restore state, we may have forced an execution of the body just to generate code.
   G__no_exec_compile = store_no_exec_compile;
   if (G__return != G__RETURN_NON) {
      // -- User asked us to return immediately.
#ifdef G__ASM
      // If no bytecode errors, restore bytecode generation
      // flag, otherwise propagate error.
      if (G__asm_noverflow) {
         G__asm_noverflow = store_asm_noverflow;
      }
#endif // G__ASM
      G__ifswitch = store_ifswitch;
      return result;
   }
   // Restore state.
   G__no_exec = 0;
   //
   //  Pause after loop exit, if requested.
   //
   if (largestep) {
      G__afterlargestep(&largestep);
   }
#ifdef G__ASM
   // If no bytecode errors, restore bytecode generation
   // flag, otherwise propagate error.
   if (G__asm_noverflow) {
      G__asm_noverflow = store_asm_noverflow;
   }
#endif // G__ASM
   // Restore state.
   G__ifswitch = store_ifswitch;
   // And return status of last executed statement.
#ifdef G__ASM
   //fprintf(stderr, "G__exec_loop: End. ne: %d nec: %d ano: %d\n", G__no_exec, G__no_exec_compile, G__asm_noverflow);
#else // G__ASM
   //fprintf(stderr, "G__exec_loop: End. ne: %d nec: %d\n", G__no_exec, G__no_exec_compile);
#endif // G__ASM
   return result;
}

//______________________________________________________________________________
//
//  Jump statements.  return, goto
//

/***********************************************************************
* G__return_value()
*
* Called by
*    G__exec_statement   'return;'
*    G__exec_statement   'return(result);'
*    G__exec_statement   'return result;'
*
***********************************************************************/
static G__value G__return_value(const char* statement)
{
   // -- Handle the return statement.
   G__value buf;
   if (G__breaksignal) {
      // -- Give user control, if he asked for it.
      G__break = 0;
      G__setdebugcond();
      G__pause();
      if (G__return > G__RETURN_NORMAL) {
         return G__null;
      }
   }
   if (G__dispsource) {
      // -- If allowed, print a closing curly brace to indicate function exit.
      if (
            (G__break || G__step || G__debug) &&
            (G__prerun || !G__no_exec) &&
            !G__disp_mask
         ) {
         G__fprinterr(G__serr, "}\n");
      }
   }
   //
   //  Free any generated temporaries.
   //
   if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
      G__free_tempobject();
   }
   //
   //  Evaluate any expression which was given.
   //
   // FIXME: There are problems with the handling of G__no_exec here!
   if (!statement[0]) {
      // -- We do *not* have an expression.
      G__no_exec = 1;
      buf = G__null;
   }
   else {
      // -- We have an expression.
      // Evaluate it.
      G__no_exec = 0;
      --G__templevel;
      buf = G__getexpr(statement);
      ++G__templevel;
   }
   if (G__no_exec_compile) {
      // -- We are just generating bytecode, not executing.
      //
      //  Generate an RTN_FUNC instruction.
      //
#ifdef G__ASM
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: RTN_FUNC  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__RTN_FUNC;
      if (statement[0]) {
         G__asm_inst[G__asm_cp+1] = 1;
      }
      else {
         G__asm_inst[G__asm_cp+1] = 0;
      }
      G__inc_cp_asm(2, 0);
#endif // G__ASM
      // --
   }
   else {
      // -- We are *not* just generating bytecode.
#ifdef G__ASM
      if (!G__xrefflag) {
         // -- Turn off bytecode generation.
         // FIXME: Should a return statement really turn off bytecode generation?
         G__asm_noverflow = 0;
      }
#endif // G__ASM
      G__return = G__RETURN_NORMAL;
   }
   return buf;
}

/**************************************************************************
* G__search_gotolabel()
*
*    Searches for goto label from given fpos and line_number. If label is
*    found, it returns label of {} nesting as mparen. fpos and line_numbers
*    are set inside this function. If label is not found 0 is returned.
**************************************************************************/
static int G__search_gotolabel(char *label,fpos_t *pfpos,int line,int *pmparen)
    /* label: NULL if upper level call */
{
   // -- Searches for goto label from given fpos and line_number.
   //
   // If label is found, it returns label in G__gotolabel, and the
   // level of curly brace nesting in pmparen.  The file position
   // and line number is left set to just after the colon of the
   // label.  If the label is not found 0 is returned, otherwise
   // the level of brace nesting is returned.
   //
   // Note: The label parameter is 0 if G__gotolabel is already set.
   //
   if (label) {
      strcpy(G__gotolabel, label);
   }
   int c = 0;
   int mparen = 0;
   if (G__breaksignal) {
      G__beforelargestep(G__gotolabel, &c, &mparen);
      if (!G__gotolabel[0]) {
         // -- Ignore goto.
         return -1;
      }
      if (mparen) {
         // -- 'S' command.
         G__step = 1;
         G__setdebugcond();
      }
   }
   mparen = 0;
   // Set file position, line number and source code display mask.
   fsetpos(G__ifile.fp, pfpos);
   G__ifile.line_number = line;
   // Tell parser we are skipping code.
   G__no_exec = 1;
   int single_quote = 0;
   int double_quote = 0;
   do {
      G__StrBuf token_sb(G__LONGLINE);
      char *token = token_sb;
      // The extraneous punctuation is here to keep from overflowing token.
      c = G__fgetstream(token, "'\"{}:();");
      if (c == EOF) {
         break;
      }
      switch (c) {
         case '\'':
            if (!double_quote) {
               single_quote ^= 1;
            }
            break;
         case '"':
            if (!single_quote) {
               double_quote ^= 1;
            }
            break;
         case '{':
            if (!single_quote && !double_quote) {
               ++mparen;
            }
            break;
         case '}':
            if (!single_quote && !double_quote) {
               --mparen;
            }
            break;
         case ':':
            if (!single_quote && !double_quote) {
               if (!strcmp(G__gotolabel, token)) {
                  // -- goto label found.
                  if (G__dispsource) {
                     G__disp_mask = 0;
                  }
                  if (
                        !G__nobreak &&
                        !G__disp_mask &&
                        !G__no_exec_compile &&
                        G__srcfile[G__ifile.filenum].breakpoint &&
                        (G__srcfile[G__ifile.filenum].maxline > G__ifile.line_number)
                     ) {
                     G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number] |= G__TRACED;
                  }
                  G__gotolabel[0] = '\0';
                  G__no_exec = 0;
                  *pmparen = mparen;
                  return mparen;
               }
            }
            break;
         default:
            // -- ;,(,) just punctuation.
            break;
      }
   }
   while (mparen);
   // goto label not found.
   return 0;
}

//______________________________________________________________________________
static int G__label_access_scope(char* statement, int* piout, int* pspaceflag, int mparen)
{
   static int memfunc_def_flag = 0;
   int ispntr;
   int line;
   ::Reflex::Scope store_tagdefining;
   fpos_t pos;
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   // Look ahead to see if we have a "::".
   int c = G__fgetc();
   if (c == ':') {
      // X::memberfunc() {...};
      //    ^  c == ':'
      // -- Member function definition.
      if (
            G__prerun &&
            (!G__func_now) &&
            (
             !G__def_tagnum ||
             ((G__def_tagnum.IsTopScope()) || (G__struct.type[G__get_tagnum(G__def_tagnum)] == 'n')) ||
             memfunc_def_flag ||
             (G__tmplt_def_tagnum)
            )
         ) {
         // --
         ::Reflex::Scope store_def_tagnum = G__def_tagnum;
         int store_def_struct_member = G__def_struct_member;
         // X<T>::TYPE X<T>::f()
         //      ^
         fgetpos(G__ifile.fp, &pos);
         line = G__ifile.line_number;
         if (G__dispsource) G__disp_mask = 1000;
         c = G__fgetname_template(temp, "(;&*");
         if (isspace(c) || (c == '&') || (c == '*')) {
            c = G__fgetspace();
            for (; (c == '&') || (c == '*');) {
               c = G__fgetspace();
            }
            if (
                  // --
#ifndef G__STD_NAMESPACE // ON780
                  (isalpha(c) && strcmp(temp, "operator")) || (!strcmp(statement, "std:") && G__ignore_stdnamespace)
#else // G__STD_NAMESPACE
                  isalpha(c) && strcmp(temp, "operator")
#endif // G__STD_NAMESPACE
                  // --
               ) {
               // --
               //
               // X<T>::TYPE X<T>::f()
               //      space^^alpha , taking as a nested class specification
               //
               fsetpos(G__ifile.fp, &pos);
               G__ifile.line_number = line;
               if (G__dispsource) {
                  G__disp_mask = 0;
               }
               statement[(*piout)++] = ':';
               return 0;
            }
         }
         fsetpos(G__ifile.fp, &pos);
         G__ifile.line_number = line;
         if (G__dispsource) {
            G__disp_mask = 0;
         }
         c = ':';
         // tag name
         statement[*piout-1] = '\0';
         if (statement[0] == '*') {
            ispntr = 1;
            G__var_type = toupper(G__var_type);
         }
         else {
            ispntr = 0;
         }
         G__def_tagnum = G__Dict::GetDict().GetScope(G__defined_tagname(statement+ispntr,0));
         store_tagdefining = G__tagdefining;
         G__tagdefining = G__def_tagnum;
         memfunc_def_flag = 1;
         G__def_struct_member = 1;
         // basically, make_ifunctable
         int brace_level = 0;
         G__exec_statement(&brace_level);
         memfunc_def_flag = 0;
         G__def_tagnum = store_def_tagnum;
         G__def_struct_member = store_def_struct_member;
         G__tagdefining = store_tagdefining;
         *piout = 0;
         *pspaceflag = 0;
         if (!mparen) {
            return 1;
         }
      }
      else {
         // -- ambiguity resolution operator
         statement[(*piout)++] = c;
      }
   }
   else {
      // -- We have "public:x", or "private:x", or "protected:x", or "case 0:abcde..."
      //                    ^               ^                 ^              ^  c == 'a'
      //
      // Undo the lookahead.
      //
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (c == '\n') {
         --G__ifile.line_number;
      }
      if (G__dispsource) {
         G__disp_mask = 1;
      }
      //
      // Set public, private, or protected, otherwise ignore.
      //
      if (
         G__prerun ||
         (
          (statement[0] == 'p') &&
          (
           !strcmp("public:", statement) ||
           !strcmp("private:", statement) ||
           !strcmp("protected:", statement)
          )
         )
      ) {
         // -- We are in prerun, or we have public, private, or protected.
         statement[*piout] = '\0';
         G__setaccess(statement, *piout);
         *piout = 0;
         *pspaceflag = 0;
      }
      else {
         // -- Ignore a statement label if we are not in a switch, and not ?:
         statement[*piout] = '\0';
         if (!G__switch && !strchr(statement, '?')) {
            int itmp = 0;
            int ctmp = G__getstream(statement, &itmp, temp, "+-*%/&|<>=^!");
            if (ctmp && strncmp(statement, "case", 4)) {
               G__fprinterr(G__serr, "Error: illegal label name %s", statement);
               G__genericerror(0);
            }
            if (G__ASM_FUNC_COMPILE == G__asm_wholefunction) {
               G__add_label_bytecode(statement);
            }
            *piout = 0;
            *pspaceflag = 0;
         }
      }
   }
   return 0;
}

/***********************************************************************
 * G__IsFundamentalDecl()
 ***********************************************************************/
static int G__IsFundamentalDecl()
{
  G__StrBuf type_name_sb(G__ONELINE);
  char *type_name = type_name_sb;
  int c;
  fpos_t pos;
  int result=1;
  int tagnum;

  /* store file position */
  int linenum = G__ifile.line_number;
  fgetpos(G__ifile.fp,&pos);
  G__disp_mask = 1000;

  c=G__fgetname_template(type_name,"(");
  if(strcmp(type_name,"struct")==0 || strcmp(type_name,"class")==0 ||
     strcmp(type_name,"union")==0) {
    result=0;
  }
  else {
    tagnum = G__defined_tagname(type_name,1);
    if(-1!=tagnum) result = 0;
    else {
       ::Reflex::Type typenum = G__find_typedef(type_name);        
      if(typenum) {
        switch(G__get_type(typenum)) {
        case 'c':
        case 's':
        case 'i':
        case 'l':
        case 'b':
        case 'r':
        case 'h':
        case 'k':
          result=1;
          break;
        default:
          result=0;
        }
      }
      else {
        if(strcmp(type_name,"unsigned")==0 ||
           strcmp(type_name,"char")==0 ||
           strcmp(type_name,"short")==0 ||
           strcmp(type_name,"int")==0 ||
           strcmp(type_name,"long")==0) result=1;
        else result=0;
      }
    }
  }

  /* restore file position */
  G__ifile.line_number = linenum;
  fsetpos(G__ifile.fp,&pos);
  G__disp_mask = 0;
  return result;
}

//______________________________________________________________________________
static void G__unsignedintegral()
{
   // -- FIXME: Describe this function!
   // -- FIXME: This routine can not handle 'unsigned int*GetXYZ();' [it must have a space after the *]

   G__StrBuf name_sb(G__MAXNAME);
   char *name = name_sb;
   fpos_t pos;
   G__unsigned = -1;
   fgetpos(G__ifile.fp, &pos);
   G__fgetname(name, "(");
   if (strcmp(name, "int") == 0)         G__var_type = 'i' -1;
   else if (strcmp(name, "char") == 0)   G__var_type = 'c' -1;
   else if (strcmp(name, "short") == 0)  G__var_type = 's' -1;
   else if (strcmp(name, "long") == 0)   G__var_type = 'l' -1;
   else if (strcmp(name, "int*") == 0)   G__var_type = 'I' -1;
   else if (strcmp(name, "char*") == 0)  G__var_type = 'C' -1;
   else if (strcmp(name, "short*") == 0) G__var_type = 'S' -1;
   else if (strcmp(name, "long*") == 0)  G__var_type = 'L' -1;
   else if (strcmp(name, "int&") == 0) {
      G__var_type = 'i' -1;
      G__reftype = G__PARAREFERENCE;
   }
   else if (strcmp(name, "char&") == 0) {
      G__var_type = 'c' -1;
      G__reftype = G__PARAREFERENCE;
   }
   else if (strcmp(name, "short&") == 0) {
      G__var_type = 's' -1;
      G__reftype = G__PARAREFERENCE;
   }
   else if (strcmp(name, "long&") == 0) {
      G__var_type = 'l' -1;
      G__reftype = G__PARAREFERENCE;
   }
   else if (strchr(name, '*')) {
      bool nomatch = false;
      if (strncmp(name, "int*", 4) == 0 || strncmp(name, "*", 1) == 0 )        G__var_type = 'I' -1;
      else if (strncmp(name, "char*", 5) == 0)  G__var_type = 'C' -1;
      else if (strncmp(name, "short*", 6) == 0) G__var_type = 'S' -1;
      else if (strncmp(name, "long*", 5) == 0)  G__var_type = 'L' -1;
      else { nomatch = true; }
      
      if (nomatch) {
         G__var_type = 'i' -1;
         fsetpos(G__ifile.fp, &pos);
      } else {
         if (strstr(name, "******")) G__reftype = G__PARAP2P + 4;
         else if (strstr(name, "*****")) G__reftype = G__PARAP2P + 3;
         else if (strstr(name, "****")) G__reftype = G__PARAP2P + 2;
         else if (strstr(name, "***")) G__reftype = G__PARAP2P + 1;
         else if (strstr(name, "**")) G__reftype = G__PARAP2P;
      }
   }
   else {
      G__var_type = 'i' -1;
      fsetpos(G__ifile.fp, &pos);
   }
   G__define_var(-1, ::Reflex::Type());
   G__reftype = G__PARANORMAL;
   G__unsigned = 0;
}


//______________________________________________________________________________
static void G__externignore()
{
   // -- Handle extern "...", EXTERN "..."
   int flag = 0;
   G__StrBuf fname_sb(G__MAXFILENAME);
   char *fname = fname_sb;
   int c = G__fgetstream(fname, "\"");
   int store_iscpp = G__iscpp;
   // FIXME: We should handle "C++" as well!
   if (!strcmp(fname, "C")) {
      // -- Handle extern "C", EXTERN "C"
      G__iscpp = 0;
   }
   else {
      // -- Handle extern "filename", EXTERN "filename"
      // FIXME: This is a CINT extension to the standard (ok, the standard says "implementation defined")!
      G__loadfile(fname);
      G__SetShlHandle(fname);
      flag = 1;
   }
   //
   //  Skip any whitespace following the double quote.
   //
   c = G__fgetspace();
   fseek(G__ifile.fp, -1, SEEK_CUR);
   if (G__dispsource) {
      G__disp_mask = 1;
   }
   //
   //  If extern "C" { ... }, we must handle
   //  everything inside the curly braces.
   //
   int brace_level = 0;
   G__exec_statement(&brace_level);
   //
   //  Restore state.
   //
   G__iscpp = store_iscpp;
   if (flag) {
      G__ResetShlHandle();
   }
}



#ifdef G__FRIEND
//______________________________________________________________________________
static void G__parse_friend()
{
   // -- FIXME: Describe me!
   // friend class A;
   // friend type func(param);
   // friend type operator<<(param);
   // friend A<T,U> operator<<(param);
   // friend const A<T,U> operator<<(param);
   //--
   // We do not need to autoload friend declaration.
   int autoload_old = G__set_class_autoloading(0);
   fpos_t pos;
   fgetpos(G__ifile.fp, &pos);
   int line_number = G__ifile.line_number;
   G__StrBuf classname_sb(G__LONGLINE);
   char *classname = classname_sb;
   int c = G__fgetname_template(classname, ";");
   int tagtype = 0;
   if (c == ';') {
      tagtype = 'c';
   }
   else if (isspace(c)) {
      if (!strcmp(classname, "class")) {
         c = G__fgetname_template(classname, ";");
         tagtype = 'c';
      }
      else if (!strcmp(classname, "struct")) {
         c = G__fgetname_template(classname, ";");
         tagtype = 's';
      }
      else {
         if (!strcmp(classname, "const") || !strcmp(classname, "volatile") || !strcmp(classname, "register")) {
            c = G__fgetname_template(classname, ";");
         }
         if ((c == ';') || (c == ',')) {
            tagtype = 'c';
         }
      }
   }
   ::Reflex::Scope envtagnum = G__get_envtagnum();
   if (!envtagnum) {
      G__genericerror("Error: friend keyword appears outside class definition");
   }

   ::Reflex::Scope store_tagnum = G__tagnum;
   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   int store_def_struct_member = G__def_struct_member;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   int store_access = G__access;

   G__friendtagnum = envtagnum;

   if (!G__tagnum.IsTopScope()) {
      G__tagnum = G__tagnum.DeclaringScope();
   }
   if (!G__def_tagnum.IsTopScope()) {
      G__def_tagnum = G__def_tagnum.DeclaringScope();
   }
   if (!G__tagdefining.IsTopScope()) {
      G__tagdefining = G__tagdefining.DeclaringScope();
   }
   if (!G__tagdefining.IsTopScope() || !G__def_tagnum.IsTopScope()) {
      G__def_struct_member = 1;
   } else {
      G__def_struct_member = 0;
   }
   G__access = G__PUBLIC;
   G__var_type = 'p';

   if (tagtype) {
      while (classname[0]) {
         ::Reflex::Scope def_tagnum = G__def_tagnum;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         ::Reflex::Scope tagdefining = G__tagdefining;
         ::Reflex::Scope friendtagnum = G__Dict::GetDict().GetScope(G__defined_tagname(classname, 2));
         G__def_tagnum = def_tagnum;
         G__tagdefining = tagdefining;
         if (!friendtagnum || friendtagnum.IsTopScope())
            friendtagnum = G__Dict::GetDict().GetScope(G__search_tagname(classname, tagtype));
         if (friendtagnum.IsTopScope())
            friendtagnum = Reflex::Dummy::Scope();
         /* friend class ... ; */
         if (envtagnum) {
            G__friendtag *friendtag = G__struct.friendtag[G__get_tagnum(friendtagnum)];
            if (friendtag) {
               while (friendtag->next) friendtag = friendtag->next;
               friendtag->next = G__new_friendtag(G__get_tagnum(envtagnum));
            }
            else {
               G__struct.friendtag[G__get_tagnum(friendtagnum)] = G__new_friendtag(G__get_tagnum(envtagnum));
               friendtag = G__struct.friendtag[G__get_tagnum(friendtagnum)];
            }
         }
         if (';' != c) {
            c = G__fgetstream(classname, ";,");
         }
         else {
            classname[0] = '\0';
         }
      }
   }
   else {
      /* friend type f() {  } ; */
      fsetpos(G__ifile.fp, &pos);
      G__ifile.line_number = line_number;
      /* friend function belongs to the inner-most namespace
       * not the parent class! In fact, this fix is not perfect, because
       * a friend function can also be a member function. This fix works
       * better only because there is no strict checking for non-member
       * function. */
      if (
         G__NOLINK != G__globalcomp &&
         G__def_tagnum &&
         !G__def_tagnum.IsNamespace()
      ) {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: This friend declaration may cause creation of wrong stub function in dictionary. Use '#pragma link off function ...;' to avoid it.");
            G__printlinenum();
         }
      }
      while (G__def_tagnum && !G__def_tagnum.IsNamespace()) {
         G__def_tagnum = G__def_tagnum.DeclaringScope();
         G__tagdefining = G__def_tagnum;
         G__tagnum = G__def_tagnum;
      }
      int brace_level = 0;
      G__exec_statement(&brace_level);
   }
   G__access = store_access;
   G__tagdefining = store_tagdefining;
   G__def_struct_member = store_def_struct_member;
   G__def_tagnum = store_def_tagnum;
   G__tagnum = store_tagnum;
   G__friendtagnum = ::Reflex::Scope();
   // Restore the autoload flag.
   G__set_class_autoloading(autoload_old);
}
#endif // G__FRIEND

/***********************************************************************
* G__keyword_anytime_5()
*
***********************************************************************/
static int G__keyword_anytime_5(char *statement)
{
  int c=0;
  int iout=0;

  if((G__prerun||(G__ASM_FUNC_COMPILE==G__asm_wholefunction&&0==G__ansiheader))
     && G__NOLINK == G__globalcomp
#ifdef G__OLDIMPLEMENTATION1083_YET
     && (G__func_now>=0 || G__def_struct_member )
#else
     && G__func_now 
#endif
     && strcmp(statement,"const")==0
     && G__IsFundamentalDecl()
     ) {
      // -- We have a function-local const of non-class type.
      G__constvar = G__CONSTVAR;
      G__const_setnoerror();
      // FIXME: Pretend a function-local const is a static, why?
      //int rslt = G__keyword_anytime_6("static");
      ::Reflex::Scope store_local = G__p_local;
      if (G__prerun && G__p_local && (!G__func_now)) {
         G__p_local = ::Reflex::Scope();
      }
      int store_no_exec = G__no_exec;
      G__no_exec = 0;
      G__static_alloc = 1;
      int brace_level = 0;
      G__exec_statement(&brace_level);
      G__static_alloc = 0;
      G__no_exec = store_no_exec;
      G__p_local = store_local;
      G__const_resetnoerror();
      G__security_error = G__NOERROR;
      G__return = G__RETURN_NON;
      //return rslt;
      return 1;
  }

  if(statement[0]!='#') return(0);

   //
   // Either we are here:
   //
   //      #ifdef TRUE
   //      #else        <---
   //      #endif
   //
   // so we skip lines until the #endif is seen,
   // or we are here:
   //
   //      #line ddd    <---
   //
   // and we scan the line number given, and set it.
   //
   if (!strcmp(statement, "#else")) {
      G__pp_skip(1);
      return 1;
   }
   if (!strcmp(statement, "#elif")) {
      G__pp_skip(1);
      return 1;
   }
   if (!strcmp(statement, "#line")) {
      G__setline(statement, c, &iout);
      // Restore statement[0] as we found it
      // because our callers might look at it!
      statement[0] = '#';
      return 1;
   }
   return 0;
}

/***********************************************************************
* G__keyword_anytime_6()
*
***********************************************************************/
static int G__keyword_anytime_6(char *statement)
{
   // -- Handle "static", "return", "#ifdef", "#endif", "#undef", and "#ident"
   if (!strcmp(statement, "static")) {
      // -- We have the static keyword.
      //
      // If in prerun then allocate memory,
      // other wise are executing, so get
      // preallocated memory.
      //
      ::Reflex::Scope store_local = G__p_local;
      if (G__prerun && G__func_now) {
         // -- Function local static during prerun, put into global variable array.
         G__p_local = ::Reflex::Scope();
      }
      // Never skip a static variable declaration,
      // even during if/then/else, switch, goto,
      // break, and continue.
      int store_no_exec = G__no_exec;
      G__no_exec = 0;
      // Flag we are doing a static allocation.
      G__static_alloc = 1;
      // Parse and execute the rest of the declaration.
      int brace_level = 0;
      G__exec_statement(&brace_level);
      // Flag we are done with static allocation.
      G__static_alloc = 0;
      // Restore state.
      G__no_exec = store_no_exec;
      G__p_local = store_local;
      // Return with success.
      return 1;
   }
   if (G__no_exec && !strcmp(statement, "return")) {
      // -- We have the return keyword.
      // Skip the rest of the statement.
      G__fignorestream(";");
      // Return with success.
      return 1;
   }
   if (statement[0] != '#') {
      // -- We do not have a preprocessor directive, return in error.
      return 0;
   }
   /***********************************
    * 1)
    *  #ifdef macro   <---
    *  #endif
    * 2)
    *  #ifdef macro   <---
    *  #else
    *  #endif
    ***********************************/
   if (!strcmp(statement, "#ifdef")) {
      int stat = G__pp_ifdef(1);
      return stat;
   }
   /***********************************
    * This part comes in following cases
    * 1)
    *  #ifdef TRUE
    *  #endif       <---
    * 2)
    *  #ifdef FAULSE
    *  #else
    *  #endif       <---
    ***********************************/
   if (!strcmp(statement, "#endif")) {
      return 1;
   }
   if (!strcmp(statement, "#undef")) {
      G__pp_undef();
      return 1;
   }
   if (!strcmp(statement, "#ident")) {
      G__fignoreline();
      return 1;
   }
   return 0;
}

/***********************************************************************
* G__keyword_anytime_7()
*
***********************************************************************/
static int G__keyword_anytime_7(char *statement)
{
   /***********************************
   * 1)
   *  #ifndef macro   <---
   *  #endif 
   * 2)
   *  #ifndef macro   <---
   *  #else
   *  #endif
   ***********************************/
   if(strcmp(statement,"#define")==0){
      ::Reflex::Scope store_tagnum=G__tagnum;
      ::Reflex::Type store_typenum=G__typenum;
      ::Reflex::Scope store_local=G__p_local;
      //
      //  Parse the macro definition.
      //
      G__p_local=::Reflex::Scope();
      G__var_type='p';
      G__definemacro=1;
      G__define();
      //
      //  Restore state.
      //
      G__definemacro=0;
      G__p_local=store_local;
      G__tagnum=store_tagnum;
      G__typenum=store_typenum;
      return(1);
   }
   if(strcmp(statement,"#ifndef")==0){
      int stat = G__pp_ifdef(0);
      return(stat);
   }
   if(strcmp(statement,"#pragma")==0){
      G__pragma();
      return(1);
   }
   return(0);
}

/***********************************************************************
* G__keyword_anytime_8()
*
***********************************************************************/
static int G__keyword_anytime_8(char* statement)
{
  /***********************************
   * template  <class T> class A { ... };
   * template  A<int>;
   * template  class A<int>;
   *          ^
   ***********************************/
  if(strcmp(statement,"template")==0){
    int c;
    fpos_t pos;
    int line_number;
    G__StrBuf tcname_sb(G__ONELINE);
    char *tcname = tcname_sb;
    line_number = G__ifile.line_number;
    fgetpos(G__ifile.fp,&pos);
    c=G__fgetspace();
    if('<'==c) {
      /* if '<' comes, this is an ordinary template declaration */
      G__ifile.line_number = line_number;
      fsetpos(G__ifile.fp,&pos);
      return(0);
    }
    /* template  A<int>; this is a template instantiation */
    tcname[0] = c;
    fseek(G__ifile.fp,-1,SEEK_CUR);
    G__disp_mask=1;
    c=G__fgetname_template(tcname,";");
    if(strcmp(tcname,"class")==0 ||
       strcmp(tcname,"struct")==0) {
      c=G__fgetstream_template(tcname,";");
    }
    else if(isspace(c)) {
      int len = strlen(tcname);
      int store_c;
      while(len && ('&'==tcname[len-1] || '*'==tcname[len-1])) --len;
      store_c = tcname[len];
      tcname[len] = 0;
      if(G__istypename(tcname)) {
        G__ifile.line_number = line_number;
        fsetpos(G__ifile.fp,&pos);
        int brace_level = 0;
        G__exec_statement(&brace_level); 
        return(1);
      }
      else {
        tcname[len] = store_c;
        c=G__fgetstream_template(tcname+strlen(tcname),";");
      }
    }
    if(!G__defined_templateclass(tcname)) {
      G__instantiate_templateclass(tcname,0);
    }
    return(1);
  }
  if(strcmp(statement,"explicit")==0){
    G__isexplicit = 1;
    return(1);
  }
  return(0);
}

/***********************************************************************
* G__alloc_exceptionbuffer
*
***********************************************************************/
G__value Cint::Internal::G__alloc_exceptionbuffer(int tagnum) 
{
  G__value buf = G__null;

  /* create class object */
  buf.obj.i = (long)malloc((size_t)G__struct.size[tagnum]);
  G__value_typenum(buf) = G__Dict::GetDict().GetType(tagnum);
  buf.ref = G__p_tempbuf->obj.obj.i;

  return(buf);
}

/***********************************************************************
* G__free_exceptionbuffer
*
***********************************************************************/
int Cint::Internal::G__free_exceptionbuffer()
{
  if(G__get_type(G__exceptionbuffer)=='u' && G__exceptionbuffer.obj.i) {
    G__StrBuf destruct_sb(G__ONELINE);
    char *destruct = destruct_sb;
    ::Reflex::Scope store_tagnum=G__tagnum;
    char *store_struct_offset = G__store_struct_offset;
    int dmy=0;
    G__set_G__tagnum(G__exceptionbuffer);
    G__store_struct_offset = (char*)G__exceptionbuffer.obj.i;
    if(G__CPPLINK==G__struct.iscpplink[G__get_tagnum(G__tagnum)]) {
      G__globalvarpointer = G__store_struct_offset;
    }
    else G__globalvarpointer = G__PVOID;
    sprintf(destruct,"~%s()",G__tagnum.Name(::Reflex::SCOPED).c_str());
    if(G__dispsource) {
      G__fprinterr(G__serr,"!!!Destructing exception buffer %s %lx"
                   ,destruct,G__exceptionbuffer.obj.i);
      G__printlinenum();
    }
    G__getfunction(destruct,&dmy ,G__TRYDESTRUCTOR);
    if(G__CPPLINK!=G__struct.iscpplink[G__get_tagnum(G__tagnum)]) 
      free((void*)G__store_struct_offset);
    /* do nothing here, exception object shouldn't be stored in legacy temp buf */
    G__tagnum = store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__globalvarpointer = G__PVOID;
  }
  G__exceptionbuffer = G__null;
  return(0);
}

/***********************************************************************
 * G__display_tempobject()
 ***********************************************************************/
void Cint::Internal::G__display_tempobject(const char* action)
{
   G__fprinterr(G__serr, "\nG__display_tempobject: Current tempobject list:\n");
   for (G__tempobject_list* p = G__p_tempbuf; p && p->prev; p = p->prev) {
      if (G__value_typenum(p->obj)) {
         G__fprinterr(G__serr, "G__display_tempobject: %s: level: %d  typename: '%s'  addr: 0x%08lx  %s:%d\n", action, p->level, G__value_typenum(p->obj).Name(::Reflex::SCOPED).c_str() , (void*)p->obj.obj.i, __FILE__, __LINE__);
      }
      else {
         G__fprinterr(G__serr, "G__display_tempobject: %s: level: %d  typename: invalid  addr: 0x%08lx  %s:%d\n", action, p->level, (void*) p->obj.obj.i, __FILE__, __LINE__);
      }
   }
   G__fprinterr(G__serr, "G__display_tempobject: Current tempobject list end.\n");
}
/***********************************************************************
* G__defined_macro()
*
* Search for macro symbol
*
***********************************************************************/
int Cint::Internal::G__defined_macro(const char *macro)
{
   int hash,hashout;
   G__hash(macro,hash,hashout);

   if(682==hash && strcmp(macro,"__CINT__")==0) return(1);
   if(!G__cpp && 1704==hash && strcmp(macro,"__CINT_INTERNAL_CPP__")==0) return(1);
   if(
      (G__iscpp || G__externblock_iscpp)
      && 1193==hash && strcmp(macro,"__cplusplus")==0) return(1);

   {
      ::Reflex::Scope varscope( ::Reflex::Scope::GlobalScope() );
      for(::Reflex::Member_Iterator iout = varscope.DataMember_Begin();
          iout != varscope.DataMember_End(); ++iout) 
      {
         char type = G__get_type(iout->TypeOf());
         if((tolower(type)=='p' || 'T'==type) &&
            /* hash == var->hash[iout] && */
            iout->Name() == macro) 
         {
            return(1); /* found */
         }
      }
   }

#ifndef G__OLDIMPLEMENTATION869
   { /* Following fix is not completely correct. It confuses typedef names
     * as macro */
      /* look for typedef names defined by '#define foo int' */
      ::Reflex::Type stat;
      ::Reflex::Scope save_tagnum = G__def_tagnum;
      G__def_tagnum = ::Reflex::Scope() ;
      stat = G__find_typedef(macro);
      G__def_tagnum = save_tagnum;
      if(stat) return(1);
   }
#endif
   /* search symbol macro table */
   if(macro!=G__replacesymbol(macro)) return(1);
   /* search  function macro table */
   {
      struct G__Deffuncmacro *deffuncmacro;
      deffuncmacro = &G__deffuncmacro;
      while(deffuncmacro->next) {
         if(deffuncmacro->name && strcmp(macro,deffuncmacro->name)==0) {
            return(1);
         }
         deffuncmacro=deffuncmacro->next;
      }
   }
   return(0); /* not found */
}


/***********************************************************************
* G__pp_command()
*
*  # if      COND
*  # ifdef   MACRO
*  # ifndef  MACRO
*
*  # elif    COND
*  # else
*
*  # endif
*   ^
*
* to be added
*  # num
*  # define MACRO
*
***********************************************************************/
int Cint::Internal::G__pp_command()
{
  int c;
  G__StrBuf condition_sb(G__ONELINE);
  char *condition = condition_sb;
  c=G__fgetname(condition,"\n\r");
  if(isdigit(condition[0])) {
    if('\n'!=c && '\r'!=c) G__fignoreline();
    G__ifile.line_number=atoi(condition);
  }
  else if(strncmp(condition,"el",2)==0)     G__pp_skip(1);
  else if(strncmp(condition,"ifdef",5)==0)  G__pp_ifdef(1);
  else if(strncmp(condition,"ifndef",6)==0) G__pp_ifdef(0);
  else if(strncmp(condition,"if",2)==0)     G__pp_if();
  else if('\n'!=c && '\r'!=c)               G__fignoreline();
  return(0);
}

/***********************************************************************
* G__pp_skip()
*
* Called by
*   G__pp_if              if condition is faulse
*   G__pp_ifdef           if condition is faulse
*   G__exec_statement     '#else'
*   G__exec_statement     '#elif'
*
*   #if [condition] , #ifdef, #ifndef
*         // read through this block
*   #else
*   #endif
***********************************************************************/
void Cint::Internal::G__pp_skip(int elifskip)
{
  G__StrBuf oneline_sb(G__LONGLINE*2);
  char *oneline = oneline_sb;
  G__StrBuf argbuf_sb(G__LONGLINE*2);
  char *argbuf = argbuf_sb;
  char *arg[G__ONELINE];
  int argn;
  
  FILE *fp;
  int nest=1;
  G__StrBuf condition_sb(G__ONELINE);
  char *condition = condition_sb;
  G__StrBuf temp_sb(G__ONELINE);
  char *temp = temp_sb;
  int i;
  
  fp=G__ifile.fp;
  
  /* elace traced mark */
  if(0==G__nobreak && 0==G__disp_mask&&
     G__srcfile[G__ifile.filenum].breakpoint&&
     G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number) {
    G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
      &=G__NOTRACED;
  }
  
  /********************************************************
   * Read lines until end of conditional compilation
   ********************************************************/
  while(nest && G__readline(fp,oneline,argbuf,&argn,arg)!=0) {
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
    ++G__ifile.line_number;
    
    if(argn>0 && arg[1][0]=='#') {
      const char* directive = arg[1]+1; // with "#if" directive will point to "if"
      int directiveArgI = 1;
      if(arg[1][1]==0
         || strcmp(arg[1],"#pragma")==0
         ) {
         directive = arg[2];
         directiveArgI = 2;
      }

      if(strncmp(directive,"if",2)==0) {
        ++nest;
      }
      else if(strncmp(directive,"else",4)==0) {
        if(nest==1 && elifskip==0) nest=0;
      }
      else if(strncmp(directive,"endif",5)==0) {
        --nest;
      }
      else if(strncmp(directive,"elif",4)==0) {
        if(nest==1 && elifskip==0) {
          int store_no_exec_compile=G__no_exec_compile;
          int store_asm_wholefunction=G__asm_wholefunction;
          int store_asm_noverflow=G__asm_noverflow;
          G__no_exec_compile=0;
          G__asm_wholefunction=0;
          G__abortbytecode();
          strcpy(condition,"");
          for(i=directiveArgI+1;i<=argn;i++) 
            strcat(condition, arg[i]);
          i = strlen (oneline) - 1;
          while (i >= 0 && (oneline[i] == '\n' || oneline[i] == '\r'))
            --i;
          if (oneline[i] == '\\') {
            int len = strlen (condition);
            while (1) {
              G__fgetstream (condition+len, "\n\r");
              if (condition[len] == '\\' && (condition[len+1] == '\n' ||
                                             condition[len+1] == '\r')) {
                char* p = condition + len;
                memmove (p, p+2, strlen (p+2) + 1);
              }
              len = strlen (condition) - 1;
              while (len>0 && (condition[len]=='\n' || condition[len]=='\r'))
                --len;
              if (condition[len] != '\\') break;
            }
          }

          /* remove comments */
          char* posComment = strstr(condition, "/*");
          if (!posComment) posComment = strstr(condition, "//");
          while (posComment) {
             if (posComment[1]=='*') {
                char* posCXXComment = strstr(condition, "//");
                if (posCXXComment && posCXXComment < posComment)
                   posComment = posCXXComment;
             }
             if (posComment[1]=='*') {
                const char* posCommentEnd = strstr(posComment+2,"*/");
                // we can have
                // #if A /*
                //   comment */ || B
                // #endif
                if (!posCommentEnd) {
                  if (G__skip_comment()) 
                     break;
                  if (G__fgetstream (posComment, "\r\n") == EOF)
                     break;
                } else {
                   strcpy(temp, posCommentEnd+2);
                   strcpy(posComment, temp);
                }
                posComment = strstr(posComment, "/*");
                if (!posComment) posComment = strstr(condition, "//");
             } else {
                posComment[0]=0;
                posComment=0;
             }
          }

          G__noerr_defined=1;
          if(G__test(condition)) {
            nest=0;
          }
          G__no_exec_compile=store_no_exec_compile;
          G__asm_wholefunction=store_asm_wholefunction;
          G__asm_noverflow=store_asm_noverflow;
          G__noerr_defined=0;
        }
      }
    }
  }
  
  /* set traced mark */
  if(0==G__nobreak && 
     0==G__disp_mask&& 0==G__no_exec_compile &&
     G__srcfile[G__ifile.filenum].breakpoint&&
     G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number) {
    G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
      |=(!G__no_exec);
  }
  
  if(G__dispsource) {
    if((G__debug||G__break||G__step
        )&&
       ((G__prerun!=0)||(G__no_exec==0))&&
       (G__disp_mask==0)){
      G__fprinterr(G__serr, "# conditional interpretation, SKIPPED");
      G__fprinterr(G__serr,"\n%-5d",G__ifile.line_number-1);
      G__fprinterr(G__serr,"%s",arg[0]);
      G__fprinterr(G__serr,"\n%-5d",G__ifile.line_number);
    }
  }
}

/***********************************************************************
* G__pp_if()
*
* Called by
*   G__exec_statement(&brace_level);   '#if'
*
*   #if [condition]
*   #else
*   #endif
***********************************************************************/
int Cint::Internal::G__pp_if()
{
   // -- FIXME: Describe this function!
   G__StrBuf condition_sb(G__LONGLINE);
   char *condition = condition_sb;
   int c, len = 0;
   int store_no_exec_compile;
   int store_asm_wholefunction;
   int store_asm_noverflow;
   int haveOpenDefined = -1; // need to convert defined FOO to defined(FOO)
   do {
      c = G__fgetstream(condition + len, " \n\r");
      len = strlen(condition);
      if (len > 0 && (condition[len] == '\n' || condition[len] == '\r')) --len;
      if (haveOpenDefined != -1) {
         if (condition[len - 1] == ')') {
            // already have enclosing (); remove duplicate opening '('
            for (; haveOpenDefined < len - 1; ++haveOpenDefined) {
               condition[haveOpenDefined] = condition[haveOpenDefined + 1];
            }
            condition[haveOpenDefined] = 0;
            --len;
         }
         else {
            condition[len] = ')';
            condition[len + 1] = 0; // this might be the end, so terminate it
            ++len;
         }
         haveOpenDefined = -1;
      }
      else
         if (c == ' ' && len > 6 && !strcmp(condition + len - 7, "defined")) {
            haveOpenDefined = len;
            condition[len] = '(';
            ++len;
         }
   }
   while ( (len > 0 && '\\' == condition[len - 1]) || c == ' ');
   {
      char *p;
      while ((p = strstr(condition, "\\\n")) != 0) {
         memmove(p, p + 2, strlen(p + 2) + 1);
      }
   }
   // This supresses error message when undefined
   // macro is refered in the #if defined(macro)
   G__noerr_defined = 1;
   //
   // false
   // skip line until #else,
   // #endif or #elif.
   // Then, return to evaluation
   //
   store_no_exec_compile = G__no_exec_compile;
   store_asm_wholefunction = G__asm_wholefunction;
   store_asm_noverflow = G__asm_noverflow;
   G__no_exec_compile = 0;
   G__asm_wholefunction = 0;
   if (!G__xrefflag) {
      G__asm_noverflow = 0;
   }
   if (!G__test(condition)) {
      //
      // SKIP
      //
      G__pp_skip(0);
   }
   else {
      int stat;
      G__no_exec_compile = store_no_exec_compile;
      G__asm_wholefunction = store_asm_wholefunction;
      G__asm_noverflow = store_asm_noverflow;
      G__noerr_defined = 0;
      stat = G__pp_ifdefextern(condition);
      return stat; // must be either G__IFDEF_ENDBLOCK or G__IFDEF_NORMAL
   }
   G__no_exec_compile = store_no_exec_compile;
   G__asm_wholefunction = store_asm_wholefunction;
   G__asm_noverflow = store_asm_noverflow;
   G__noerr_defined = 0;
   return G__IFDEF_NORMAL;
}

//______________________________________________________________________________
int Cint::Internal::G__pp_ifdef(int def)
{
   // -- FIXME: Describe this function!
   // def: 1 for ifdef; 0 for ifndef
   G__StrBuf temp_sb(G__LONGLINE);
   char *temp = temp_sb;
   int notfound = 1;

   G__fgetname(temp, "\n\r");

   notfound = G__defined_macro(temp) ^ 1;

   /*****************************************************************
    * false macro not found skip line until #else, #endif or #elif.
    * Then, return to evaluation
    *************************************************************/
   if (notfound == def) {
      /* SKIP */
      G__pp_skip(0);
   }
   else {
      int stat = G__pp_ifdefextern(temp);
      return(stat); /* must be either G__IFDEF_ENDBLOCK or G__IFDEF_NORMAL */
   }

   return(G__IFDEF_NORMAL);
}

/***********************************************************************
* G__exec_catch()
*
***********************************************************************/
int Cint::Internal::G__exec_catch(char *statement)
{
   int c;
   while(1) {
      fpos_t fpos;
      int line_number;

      /* catch (ehclass& obj) {  } 
      * ^^^^^^^ */
      do {
         c=G__fgetstream(statement,"(};");
      } while('}'==c);
      if('('!=c||strcmp(statement,"catch")!=0) return(1);
      fgetpos(G__ifile.fp,&fpos);
      line_number=G__ifile.line_number;

      /* catch (ehclass& obj) {  } 
      *        ^^^^^^^^ */
      c=G__fgetname_template(statement,")&*");

      if (statement[0] == '.') {
         // catch all exceptions
         // catch(...) {  }
         if (c != ')') {
            c = G__fignorestream(")");
         }
         int brace_level = 0;
         G__exec_statement(&brace_level);
         break;
      }
      else {
         int tagnum;
         tagnum=G__defined_tagname(statement,2);
         if(G__get_tagnum(G__value_typenum(G__exceptionbuffer))==tagnum || 
            -1!=G__ispublicbase(tagnum,G__get_tagnum(G__value_typenum(G__exceptionbuffer))
            ,(void*)G__exceptionbuffer.obj.i)) {
               /* catch(ehclass& obj) { match } */
               G__value store_ansipara;
               store_ansipara=G__ansipara;
               G__ansipara=G__exceptionbuffer;
               G__ansiheader=1;
               G__funcheader=1;
               G__ifile.line_number=line_number;
               fsetpos(G__ifile.fp,&fpos);
               int brace_level = 0;
               G__exec_statement(&brace_level); // declare exception handler object
               G__globalvarpointer=G__PVOID;
               G__ansiheader=0;
               G__funcheader=0;
               G__ansipara=store_ansipara;
               brace_level = 0;
               G__exec_statement(&brace_level); /* exec catch block body */
               break;
         }
         /* catch(ehclass& obj) { unmatch } */
         if(')'!=c) c=G__fignorestream(")");
         G__no_exec = 1;
         int brace_level = 0;
         G__exec_statement(&brace_level);
         G__no_exec = 0;
      }
   }
   G__free_exceptionbuffer();
   return(0);
}

/***********************************************************************
* G__skip_comment()
*
***********************************************************************/
int Cint::Internal::G__skip_comment()
{
   // -- Skip a C-style comment, must be called immediately after '/*' is scanned.
   // Return value is either 0 or EOF.
   int c0 = G__fgetc();
   if (c0 == EOF) {
      G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
      if (G__key) {
         system("key .cint_key -l execute");
      }
      G__eof = 2;
      return EOF;
   }
   int c1 = G__fgetc();
   if (c1 == EOF) {
      G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
      if (G__key) {
         system("key .cint_key -l execute");
      }
      G__eof = 2;
      return EOF;
   }
   //fprintf(stderr, "G__skip_comment: c0: '%c' c1: '%c'\n", c0, c1);
   while ((c0 != '*') || (c1 != '/')) {
#ifdef G__MULTIBYTE
      if (G__IsDBCSLeadByte(c0)) {
         c0 = '\0';
         G__CheckDBCS2ndByte(c1);
      }
      else {
         c0 = c1;
      }
#else // G__MULTIBYTE
      c0 = c1;
#endif // G__MULTIBYTE
      c1 = G__fgetc();
      if (c1 == EOF) {
         G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
         if (G__key) {
            system("key .cint_key -l execute");
         }
         G__eof = 2;
         return EOF;
      }
      //fprintf(stderr, "G__skip_comment: c0: '%c' c1: '%c'\n", c0, c1);
   }
   //fprintf(stderr, "G__skip_comment: return with c0: '%c' c1: '%c'\n", c0, c1);
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__skip_comment_peek()
{
   // -- Skip a C-style comment during a peek, must be called immediately after '/*' is scanned.
   // Return value is either 0 or EOF.
   int c0 = fgetc(G__ifile.fp);
   if (c0 == EOF) {
      G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
      if (G__key) {
         system("key .cint_key -l execute");
      }
      G__eof = 2;
      return EOF;
   }
   int c1 = fgetc(G__ifile.fp);
   if (c1 == EOF) {
      G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
      if (G__key) {
         system("key .cint_key -l execute");
      }
      G__eof = 2;
      return EOF;
   }
   while ((c0 != '*') || (c1 != '/')) {
#ifdef G__MULTIBYTE
      if (G__IsDBCSLeadByte(c0)) {
         c0 = '\0';
         G__CheckDBCS2ndByte(c1);
      }
      else {
         c0 = c1;
      }
#else // G__MULTIBYTE
      c0 = c1;
#endif // G__MULTIBYTE
      c1 = fgetc(G__ifile.fp);
      if (c1 == EOF) {
         G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
         if (G__key) {
            system("key .cint_key -l execute");
         }
         G__eof = 2;
         return EOF;
      }
   }
   return 0;
}

//______________________________________________________________________________
//
//  The beginning of parsing and execution.
//

/***********************************************************************
* G__value G__exec_statement(&brace_level);
*
*
*  Execute statement list  { ... ; ... ; ... ; }
***********************************************************************/
G__value Cint::Internal::G__exec_statement(int *mparen)
{
   int current_char = 0;
   char* conststring = 0;
   int iout = 0;
   int spaceflag = 0;
   int single_quote = 0;
   int double_quote = 0;
   int largestep = 0;
   int commentflag = 0;
   int add_fake_space = 0;
   int fake_space = 0;
   int discard_space = 0;
   int discarded_space = 0;
   G__StrBuf statement_sb(G__LONGLINE);
   char *statement = statement_sb;
   G__value result = G__null;
   //fprintf(stderr, "\nG__exec_statement: Begin.\n");
   fpos_t start_pos;
   fgetpos(G__ifile.fp, &start_pos);
   int start_line = G__ifile.line_number;
   std::stack<int> mparen_lines;
   int mparen_old = *mparen;
   while (1) {
      if (iout >= G__LONGLINE)
      {
         G__genericerror("Error: Line too long in G__exec_statement when processing ");
         return G__null;
      }
      fake_space = 0;
      if (add_fake_space && !double_quote && !single_quote) {
         current_char = ' ';
         add_fake_space = 0;
         fake_space = 1;
      }
      else {
         current_char = G__fgetc();
      }
      discard_space = 0;
read_again:
      statement[iout] = '\0';
    
      switch( current_char ) {

#ifdef G__OLDIMPLEMENTATIONxxxx_YET
         case ',' : /* column */
            if(!G__ansiheader) break;
#endif
         case '\n':
            // -- Handle a newline.
            if (*mparen != mparen_old) {
               // -- Update the line numbers of any dangling parentheses.
               int mparen_lines_size = mparen_lines.size();
               while (mparen_lines_size < *mparen) {
                  // The stream has already read the newline, so take line_number minus one.
                  mparen_lines.push(G__ifile.line_number - 1);
                  ++mparen_lines_size;
               }
               while (mparen_lines_size > *mparen) {
                  mparen_lines.pop();
                  --mparen_lines_size;
               }
            }
            // Intentionally fallthrough.
         case ' ' : /* space */
         case '\t' : /* tab */
         case '\r': /* end of line */
         case '\f': /* end of line */
            commentflag=0;
            /* ignore these character */
            if(single_quote || double_quote!=0) {
               statement[iout++] = current_char ;
            }
            else {
after_replacement:
               if (!fake_space) {
                  discard_space = 1;
               }
               if (spaceflag == 1) {
                  // -- Take action on space, even if skipping code.  Do preprocessing and look for declarations.
                  statement[iout] = '\0';
                  // search keyword
G__preproc_again:
                  if ((statement[0] == '#') && isdigit(statement[1])) {
                     // -- Handle preprocessor directive "#<number> <filename>", a CINT extension to the standard.
                     // -- # [line] <[filename]>
                     int stat = G__setline(statement, current_char, &iout);
                     if (stat) {
                        goto G__preproc_again;
                     }
                     spaceflag = 0;
                     iout = 0;
                  }

                  switch(iout) {
                     case 1:
                        // -- Handle preprocessor directive "#<number> <filename>", a CINT extension to the standard.
                        // -- # [line] <[filename]>
                        if (statement[0] == '#') {
                           int stat = G__setline(statement, current_char, &iout);
                           if (stat) {
                              goto G__preproc_again;
                           }
                           spaceflag = 0;
                           iout = 0;
                        }
                        break;
                     case 2:
                        // -- Handle preprocessor directive "#!", a CINT extension to the standard.
                        // -- comment '#! xxxxxx'
                        if (statement[0] == '#') {
                           if ((current_char != '\n') && (current_char != '\r')) {
                              G__fignoreline();
                           }
                           spaceflag = 0;
                           iout = 0;
                        }
                        break;
                     case 3:
                        // -- Handle preprocessor directive "#if".
                        //
                        // 1)
                        //  #if condition <---
                        //  #endif
                        // 2)
                        //  #if condition <---
                        //  #else
                        //  #endif
                        //
                        if (!strcmp(statement, "#if")) {
                           int stat = G__pp_if();
                           if (stat == G__IFDEF_ENDBLOCK) {
                              return G__null;
                           }
                           spaceflag = 0;
                           iout = 0;
                        }
                        break;
                     case 4:
                        // -- Handle keyword "case", even if skipping code.
                        if ((*mparen == 1) && !strcmp(statement, "case")) {
                           //fprintf(stderr, "G__exec_statement: Saw a case.\n");
                           // Scan the case label in.
                           G__StrBuf casepara_sb(G__ONELINE);
                           char *casepara = casepara_sb;
                           G__fgetstream(casepara, ":");
                           current_char = G__fgetc();
                           while (current_char == ':') {
                              strcat(casepara, "::");
                              int lenxxx = strlen(casepara);
                              G__fgetstream(casepara + lenxxx, ":");
                              current_char = G__fgetc();
                           }
                           // Backup one character.
                           fseek(G__ifile.fp, -1, SEEK_CUR);
                           // Flag that we have already displayed it, do not show it again.
                           G__disp_mask = 1;
                           //
                           //  If we are not in a switch statement,
                           //  then we are done.
                           if (G__switch) {
                              // -- We are in a switch statement, evaluate the case expression.
                              G__value local_result = G__exec_switch_case(casepara);
                              if (G__switch_searching) {
                                 // -- We are searching for a matching case, return value of case expression.
                                 return local_result;
                              }
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = 0;
                        }
                        break;
                    case 5:
                        // -- Handle a function-local const declaration, or #else, #elif, and #line.
                        {
                           int handled = G__keyword_anytime_5(statement);
                           if (handled) {
                              if (!*mparen && (statement[0] != '#')) {
                                 return G__null;
                              }
                              spaceflag = 0;
                              iout = 0;
                           }
                        }
                        break;
                     case 6:
                        // -- Handle "static", "return", "#ifdef", "#endif", "#undef", and "#ident"
                        {
                           int stat = G__keyword_anytime_6(statement);
                           if (stat) {
                              if (!*mparen && (statement[0] != '#')) {
                                 return G__null;
                              }
                              if (stat == G__IFDEF_ENDBLOCK) {
                                 return G__null;
                              }
                              spaceflag = 0;
                              iout = 0;
                           }
                        }
                        break;
                     case 7:
                        // -- Handle "default", "#define", "#ifndef", and "#pragma".
                        if ((*mparen == 1) && !strcmp(statement, "default")) {
                           // -- Handle labeled statment, "default :"
                           //                                     ^
                           G__fignorestream(":");
                           //
                           //  If we are not in a switch statement,
                           //  then we are done.
                           //
                           if (G__switch) {
                              // -- We are in a switch statement, perform semantic actions.
#ifdef G__ASM
                              if (G__asm_noverflow) {
                                 // -- We are generating bytecode.
                                 //
                                 //  Terminate previous case and backpatch its
                                 //  conditional jump on the result of its case
                                 //  expression.
                                 //
                                 if (G__prevcase) {
                                    // --  There was a previous case clause.
                                    //
                                    //  Backpatch previous case expression test to jump to this case if not equal.
                                    //
#ifdef G__ASM_DBG
                                    if (G__asm_dbg) {
                                       G__fprinterr(G__serr, "   %3x: CNDJMP %x assigned (for case expression not equal)  %s:%d\n", G__prevcase - 1, G__asm_cp, __FILE__, __LINE__);
                                    }
#endif // G__ASM_DBG
                                    G__asm_inst[G__prevcase] = G__asm_cp;
                                    G__prevcase = 0;
                                 }
                              }
#endif // G__ASM
                              if (G__switch_searching) {
                                 // -- We are searching for a matching case, return value of case expression.
                                 return G__default;
                              }
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = 0;
                           break;
                        }
                        // -- Handle "#define", "#ifndef", and "#pragma".
                        {
                           int stat = G__keyword_anytime_7(statement);
                           if (stat) {
                              if (!*mparen && (statement[0] != '#')) {
                                 return G__null;
                              }
                              if (stat == G__IFDEF_ENDBLOCK) {
                                 return G__null;
                              }
                              spaceflag = 0;
                              iout = 0;
                           }
                        }
                        break;
                     case 8:
                        // -- Handle labeled statment, "default: "
                        //
                        // BUGS:
                        //   default:printf("abc");
                        //          ^
                        // If there is no space char on
                        // either side of the ':', the
                        // default keyword is ignored.
                        //
                        if ((*mparen == 1) && !strcmp(statement, "default:")) {
                           //
                           //  If we are not in a switch statement,
                           //  then we are done.
                           //
                           if (G__switch) {
                              // -- We are in a switch statement, perform semantic actions.
                              //
                              //  Check if we are now at a breakpoint.
                              //
                              // FIXME: The other handler for 'default ' should do this as well!
                              if (
                                 !G__nobreak &&
                                 !G__disp_mask &&
                                 !G__no_exec_compile &&
                                 G__srcfile[G__ifile.filenum].breakpoint &&
                                 (G__srcfile[G__ifile.filenum].maxline > G__ifile.line_number)
                              ) {
                                 G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number] |= G__TRACED;
                              }
#ifdef G__ASM
                              if (G__asm_noverflow) {
                                 // -- We are generating bytecode.
                                 //
                                 //  Terminate previous case and backpatch its
                                 //  conditional jump on the result of its case
                                 //  expression.
                                 //
                                 if (G__prevcase) {
                                    // --  There was a previous case clause.
                                    //
                                    //  Backpatch previous case expression test to jump to this case if not equal.
                                    //
#ifdef G__ASM_DBG
                                    if (G__asm_dbg) {
                                       G__fprinterr(G__serr, "   %3x: CNDJMP %x assigned (for case expression not equal)  %s:%d\n", G__prevcase - 1, G__asm_cp, __FILE__, __LINE__);
                                    }
#endif // G__ASM_DBG
                                    G__asm_inst[G__prevcase] = G__asm_cp;
                                    G__prevcase = 0;
                                 }
                              }
#endif // G__ASM
                              if (G__switch_searching) {
                                 // -- We are searching for a matching case, return value of case expression.
                                 return G__default;
                              }
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = 0;
                           break;
                        }
                        // -- Handle "template" and "explicit" keywords.
                        {
                           int stat = G__keyword_anytime_8(statement);
                           if (stat) {
                              if (!*mparen && (statement[0] != '#')) {
                                 return G__null;
                              }
                              spaceflag = 0;
                              iout = 0;
                           }
                        }
                        break;
                  }
               }
               if ((spaceflag == 1) && !G__no_exec) {
                  // -- Take action on space, we are not skipping code.
                  //
                  // Terminate the accumulated statement text.
                  statement[iout] = '\0';
                  // Hack, fixup read of xxx** or xxx*&.
                  if (
                     // -- If we have xxx** or xxx*&.
                     (iout > 3) &&
                     (statement[iout-2] == '*') &&
                     ((statement[iout-1] == '*') || (statement[iout-1] == '&'))
                  ) {
                     // -- We have xxx** or xxx*&, undo read of '**' or '*&'.
                     // Note: Possibly char**, char*&, int**, int*&
                     //
                     // Forget we read the ** or *&.
                     statement[iout-2] = '\0';
                     iout -= 2;
                     // Undo reading the two or three characters, the ** or *&,
                     // and the space character which got us here.
                     // The space character may have been fake, that is why
                     // we must backup 2 or 3 characters.
                     // FIXME: This fails badly if G__fgetc() finished a macro.
                     // FIXME: The line number may now be wrong after seeking.
                     if (fake_space) {
                        fseek(G__ifile.fp, -2, SEEK_CUR);
                     }
                     else {
                        fseek(G__ifile.fp, -3, SEEK_CUR);
                     }
                     // Flag that we should not print the characters we backed
                     // up over a second time.
                     if (G__dispsource) {
                        // -- We are displaying source.
                        // Do not print the next two characters.
                        // FIXME: This should be two or three depending on fake_space.
                        G__disp_mask = 2;
                     }
                  }
                   switch (iout) {
                     case 2:
                        if (!strcmp(statement, "do")) {
                           // -- We have 'do stmt; while ();'.
                           //               ^
                           do_do:
                           // Note: We come to here if we have 'do {...} while ();'.
                           //                                     ^
                           result = G__exec_do();
                           if (!*mparen || (G__return > G__RETURN_NON)) {
                              return result;
                           }
                           if(G__value_typenum(result)==G__value_typenum(G__block_goto)&&
                              result.ref==G__block_goto.ref) {
                                 int found = G__search_gotolabel(0, &start_pos, start_line, mparen);
                                 // If found, continue parsing, the input file is now
                                 // positioned immediately after the colon of the label.
                                 // Otherwise, return and let our caller try to find it.
                                 if (!found) {
                                    // -- Not found, maybe our caller can find it.
                                    return G__block_goto;
                                 }
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                        }
                        break;
                     case 3:
                        // -- Handle int, new, and try.
                        if (!strcmp(statement, "int")) {
                           G__var_type = 'i' + G__unsigned;
                           G__define_var(-1,::Reflex::Type());          
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "new")) {
                           current_char = G__fgetspace();
                           if (current_char == '(') {
                              fseek(G__ifile.fp, -1, SEEK_CUR);
                              if (G__dispsource) {
                                 G__disp_mask = 1;
                              }
                              statement[iout++] = ' ';
                              spaceflag |= 1;
                              // a little later this string will be passed to subfunctions
                              // that expect the string to be terminated
                              statement[iout] = '\0';
                           }
                           else {
                              statement[0] = current_char;
                              current_char = G__fgetstream_template(statement + 1, ";");
                              result = G__new_operator(statement);
                              // Reset the statement buffer.
                              iout = 0;
                              // Flag that any following whitespace does not trigger any semantic action.
                              spaceflag = -1;
                              if (!*mparen) {
                                 return result;
                              }
                           }
                           break;
                        }
                        if (!strcmp(statement, "try")) {
                           G__exec_try(statement);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        break;
                     case 4:
                        if(strcmp(statement,"char")==0) {
                           G__DEFVAR('c');
                           break;
                        }
                        if(strcmp(statement,"FILE")==0) {
                           G__DEFVAR('e');
                           break;
                        }
                        if(strcmp(statement,"long")==0) {
                           G__DEFVAR('l');
                           break;
                        }
                        if(strcmp(statement,"void")==0) {
                           G__DEFVAR('y');
                           break;
                        }
                        if(strcmp(statement,"bool")==0) {
                           G__DEFVAR('g');
                           break;
                        }
                        if(strcmp(statement,"int*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('I');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"int&")==0) {
                           G__DEFREFVAR('i');
                           break;
                        }
                        if(strcmp(statement,"enum")==0) {
                           G__DEFSTR('e');
                           break;
                        }
                        if(strcmp(statement,"auto")==0) {
                           spaceflag = -1;
                           iout=0;
                           break;
                        }
                        if (!strcmp(statement, "(new")) {
                           // -- Handle a parenthesized new expression.
                           // Check for placement new.
                           current_char = G__fgetspace();
                           if (current_char == '(') {
                              // -- We have a placement new expression.
                              // Backup one character.
                              // FIXME: The line number, dispmask, and macro expansion state may be wrong now!
                              fseek(G__ifile.fp, -1, SEEK_CUR);
                              // And insert a fake space into the statement.
                              statement[iout++] = ' ';
                              // And terminate the buffer.
                              statement[iout] = '\0';
                              // Flag that any whitespace should now trigger a semantic action.
                              spaceflag |= 1;
                           }
                           else {
                              // Insert a fake space into the statement (right after the 'new').
                              statement[iout++] = ' ';
                              // Then add in the character that terminated the peek ahead.
                              statement[iout++] = current_char;
                              // Skip showing the next character.
                              // FIXME: This is wrong!
                              if (G__dispsource) {
                                 // -- We are display source code.
                                 G__disp_mask = 1;
                              }
                              // Scan the reset of the parenthesised new expression.
                              current_char = G__fgetstream_template(statement + iout, ")");
                              iout = strlen(statement);
                              statement[iout++] = current_char;
                              // And terminate the statement buffer.
                              statement[iout] = '\0';
                              // Flag that any whitespace should now trigger a semantic action.
                              spaceflag |= 1;
                           }
                           break;
                        }
                        if (!strcmp(statement, "goto")) {
                           // -- Handle 'goto label;'
                           //                ^
                           // Check if current security level allows the use of the goto statement.
                           G__CHECK(G__SECURE_GOTO, 1, return G__null);
                           // Scan in the goto label.
                           current_char = G__fgetstream(statement, ";");
#ifdef G__ASM
                           if (G__asm_wholefunction == G__ASM_FUNC_COMPILE) {
                              G__add_jump_bytecode(statement);
                           }
                           else {
                              // --
#ifdef G__ASM_DBG
                              if (G__asm_noverflow && G__asm_dbg) {
                                 G__fprinterr(G__serr, "bytecode compile aborted by goto statement");
                                 G__printlinenum();
                              }
#endif // G__ASM_DBG
                              if (!G__xrefflag) {
                                 // FIXME: Should goto statements disable bytecode?
                                 G__asm_noverflow = 0;
                              }
                           }
#endif // G__ASM
                           if (G__no_exec_compile) {
                              // -- We are only compiling, all done now.
                              if (!*mparen) {
                                 return G__null;
                              }
                              // Reset the statement buffer.
                              iout = 0;
                              // Flag that any following whitespace does not trigger any semantic action.
                              spaceflag = -1;
                              break;
                           }
                           int found = G__search_gotolabel(statement, &start_pos, start_line, mparen);
                           // If found, continue parsing, the input file is now
                           // positioned immediately after the colon of the label.
                           // Otherwise, return and let our caller try to find it.
                           if (!found) {
                              // -- Not found, maybe our caller can find it.
                              return G__block_goto;
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                        }
                        break;
                     case 5:
                        // -- Handle short, float, char*, char&, bool&, FILE*, long*, bool*, long&, void*, class, union, using, throw, const.
                        if(strcmp(statement,"short")==0) {
                           G__DEFVAR('s');
                           break;
                        }
                        if(strcmp(statement,"float")==0) {
                           G__DEFVAR('f');
                           break;
                        }
                        if(strcmp(statement,"char*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('C');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"char&")==0) {
                           G__DEFREFVAR('c');
                           break;
                        }
                        if(strcmp(statement,"bool&")==0) {
                           G__DEFREFVAR('g');
                           break;
                        }
                        if(strcmp(statement,"FILE*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('E');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"long*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('L');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"bool*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('G');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"long&")==0) {
                           G__DEFREFVAR('l');
                           break;
                        }
                        if(strcmp(statement,"void*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('Y');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"class")==0) {
                           G__DEFSTR('c');
                           break;
                        }
                        if(strcmp(statement,"union")==0) {
                           G__DEFSTR('u');
                           break;
                        }
                        if(strcmp(statement,"using")==0) {
                           G__using_namespace();
                           spaceflag = -1;
                           iout=0;
                           break;
                        }
                        if(strcmp(statement,"throw")==0) {
                           G__exec_throw(statement);
                           spaceflag = -1;
                           iout=0;
                        }
                        if(strcmp(statement,"const")==0) {
                           G__constvar = G__CONSTVAR;
                           spaceflag = -1;
                           iout=0;
                        }
                        break;

                     case 6:
                        // -- Handle double, struct, short*, short&, float*, return, delete, friend, extern, EXTERN, signed, inline, #error
                        if(strcmp(statement,"double")==0) {
                           G__DEFVAR('d');
                           break;
                        }
                        if(strcmp(statement,"struct")==0) {
                           G__DEFSTR('s');
                           break;
                        }
                        if(strcmp(statement,"short*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('S');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"short&")==0) {
                           G__DEFREFVAR('s');
                           break;
                        }
                        if(strcmp(statement,"float*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('F');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"float&")==0) {
                           G__DEFREFVAR('f');
                           break;
                        }
                        if (!strcmp(statement, "return")) {
                           // -- Handle 'return ...';
                           //                  ^
                           G__fgetstream_new(statement, ";");
                           result = G__return_value(statement);
                           if (G__no_exec_compile) {
                              if (!*mparen) {
                                 return G__null;
                              }
                              // Reset the statement buffer.
                              iout = 0;
                              // Flag that any following whitespace does not trigger any semantic action.
                              spaceflag = -1;
                              break;
                           }
                           return result;
                        }
                        if (!strcmp(statement, "delete")) {
                           // -- Handle 'delete ...'.
                           //                  ^
                           int delete_char = G__fgetstream(statement , "[;");
                           iout = 0;
                           if (delete_char == '[') {
                              if (!statement[0]) {
                                 delete_char = G__fgetstream(statement, "]");
                                 delete_char = G__fgetstream(statement, ";");
                                 iout = 1;
                              }
                              else {
                                 strcpy(statement + strlen(statement), "[");
                                 delete_char = G__fgetstream(statement + strlen(statement), "]");
                                 strcpy(statement + strlen(statement), "]");
                                 delete_char = G__fgetstream(statement + strlen(statement), ";");
                              }
                           }
                           // Note: iout == 1, if 'delete[]'
                           int local_largestep = 0;
                           if (G__breaksignal) {
                              int ret = G__beforelargestep(statement, &iout, &local_largestep);
                              if (ret > 1) {
                                 return G__null;
                              }
                              if (!statement[0]) {
                                 // Reset the statement buffer.
                                 iout = 0;
                                 // Flag that any following whitespace does not trigger any semantic action.
                                 spaceflag = -1;
                                 break;
                              }
                           }
                           G__delete_operator(statement, iout /* isarray */);
                           if (local_largestep) {
                              G__afterlargestep(&local_largestep);
                           }
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "friend")) {
                           // -- Handle 'friend ...'.
                           //                  ^
                           G__parse_friend();
                           if (!*mparen) {
                              return G__null;
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "extern") || !strcmp(statement, "EXTERN")) {
                           // -- Handle 'extern ...' and 'EXTERN ...'.
                           //                  ^                ^
                           G__var_type = 'p';
                           int extern_char = G__fgetspace();
                           if (extern_char != '"') {
                              // -- Handle 'extern var...' and 'EXTERN var...'.
                              // We just ignore it.
                              fseek(G__ifile.fp, -1, SEEK_CUR);
                              if (extern_char == '\n') {
                                 --G__ifile.line_number;
                              }
                              if (G__dispsource) {
                                 G__disp_mask = 1;
                              }
                              if ((G__globalcomp == G__NOLINK) && !G__parseextern) {
                                 G__fignorestream(";");
                              }
                              if (!*mparen) {
                                 return G__null;
                              }
                              // Reset the statement buffer.
                              iout = 0;
                              // Flag that any following whitespace does not trigger any semantic action.
                              spaceflag = -1;
                              break;
                           }
                           G__externignore();
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "signed")) {
                           // -- Handle 'signed ...'.
                           //                  ^
                           // We ignore it.
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "inline")) {
                           // -- Handle 'inline ...'.
                           //                  ^
                           // We ignore it.
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "#error")) {
                           // -- Handle '#error ...'.
                           //                  ^
                           G__pounderror();
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        break;
                     case 7:
                        // -- Handle typedef, double*, double&, virtual, mutable.
                        if (!strcmp(statement, "typedef")) {
                           G__var_type = 't';
                           G__define_type();
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if(strcmp(statement,"double*")==0) {
                           G__typepdecl=1;
                           G__DEFVAR('D');
                           G__typepdecl=0;
                           break;
                        }
                        if(strcmp(statement,"double&")==0) {
                           G__DEFREFVAR('d');
                           break;
                        }
                        if(strcmp(statement,"virtual")==0) {
                           G__virtual = 1;
                           spaceflag = -1;
                           iout=0;
                           break;
                        }
                        if (strcmp (statement, "mutable") == 0) {
                           spaceflag = -1;
                           iout = 0;
                           break;
                        }
                        break;          
                     case 8:
                        // -- Handle unsigned, volatile, register, delete[], operator, typename, #include
                        if (!strcmp(statement, "unsigned")) {
                           // -- Handle 'unsigned ...';
                           //                    ^
                           G__unsignedintegral();
                           if (!*mparen) {
                              return G__null;
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "volatile") || !strcmp(statement, "register")) {
                           // -- Handle 'volatile ...' and 'register ...';
                           //                    ^                  ^
                           // We just ignore them.
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "delete[]")) {
                           // Handle 'delete[] ...'.
                           //                 ^
                           G__fgetstream(statement, ";");
                           int local_largestep = 0;
                           if (G__breaksignal) {
                              int ret = G__beforelargestep(statement, &iout, &local_largestep);
                              if (ret > 1) {
                                 return G__null;
                              }
                              if (!statement[0]) {
                                 // Reset the statement buffer.
                                 iout = 0;
                                 // Flag that any following whitespace does not trigger any semantic action.
                                 spaceflag = -1;
                                 break;
                              }
                           }
                           G__delete_operator(statement, 1 /* isarray */);
                           if (local_largestep) {
                              G__afterlargestep(&local_largestep);
                           }
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "operator")) {
                           // -- Handle 'operator ...';
                           //                    ^
                           ::Reflex::Scope store_tagnum;
                           do {
                              G__StrBuf oprbuf_sb(G__ONELINE);
                              char *oprbuf = oprbuf_sb;
                              iout = strlen(statement);
                              current_char = G__fgetname(oprbuf, "(");
                              switch (oprbuf[0]) {
                                 case '*':
                                 case '&':
                                    strcpy(statement + iout, oprbuf);
                                    break;
                                 default:
                                    statement[iout] = ' ';
                                    strcpy(statement + iout + 1, oprbuf);
                              }
                           }
                           while (current_char != '(');
                           iout = strlen(statement);
                           if (statement[iout-1] == ' ') {
                              --iout;
                           }
                           statement[iout] = '\0';
                           result = G__string2type(statement + 9);
                           store_tagnum = G__tagnum;
                           G__var_type = G__get_type(result);;
                           G__typenum = G__value_typenum(result);
                           G__set_G__tagnum(result);
                           int store_constvar = G__constvar;
                           G__constvar = (short)result.obj.i; // see G__string2type
                           int store_reftype = G__reftype;
                           G__reftype = G__get_reftype(G__value_typenum(result));
                           statement[iout] = '(';
                           statement[iout+1] = '\0';
                           G__make_ifunctable(statement);
                           G__tagnum = store_tagnum;
                           G__constvar = store_constvar;
                           G__reftype = store_reftype;
#ifdef G__SECURITY
                           if (!*mparen) {
                              return G__null;
                           }
#else // G__SECURITY
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
#endif // G__SECURITY
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "typename")) {
                           // -- Handle 'typename ...'.
                           //                    ^
                           // We just ignore it.
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (statement[0] != '#') {
                           break;
                        }
                        if (!strcmp(statement, "#include")) {
                           // -- Handle '#include ...'.
                           //                    ^
                           G__include_file();
#ifdef G__SECURITY
                           if (!*mparen) {
                              return G__null;
                           }
#else // G__SECURITY
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
#endif // G__SECURITY
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                        }
                        break;
                     case 9:
                        // -- Handle namespace, unsigned*, unsigned&
                        if(strcmp(statement,"namespace")==0) {
                           G__DEFSTR('n');
                           break;
                        }
                        if (!strcmp(statement, "unsigned*")) {
                           G__var_type = 'I' - 1;
                           G__unsigned = -1;
                           G__define_var(-1, ::Reflex::Type());
                           G__unsigned = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                        }
                        else if (!strcmp(statement, "unsigned&")) {
                           G__var_type = 'i' -1;
                           G__unsigned = -1;
                           G__reftype = G__PARAREFERENCE;
                           G__define_var(-1, ::Reflex::Type());
                           G__reftype = G__PARANORMAL;
                           G__unsigned = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                        }
                     case 13:
                        // -- Handle __extension__.
                        if (!strcmp(statement, "__extension__")) {
                           // -- Handle '__extension__ ...'.
                           //                         ^
                           // We just ignore it.
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                  }
                  if (iout) {
                     // -- We have some scanned text.
                     //
                     //  Check if it is a known typedef or
                     //  class/enum/namespace/struct/union name,
                     //  and if so declare a variable.
                     //
                     int processed = G__defined_type(statement, iout);
                     if (processed) {
                        // -- Declaration was processed successfully.
                        // Flag to skip following whitespace.
                        spaceflag = -1;
                        // Reset the statement buffer.
                        iout = 0;
                        // Return if we are done, or we were told to.
                        if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                           return G__null;
                        }
                     }
                     else {
                        // -- Allow for spaces after a scope operator.
                        if ((iout >= 2) && (statement[iout-1] == ':') && (statement[iout-2] == ':')) {
                           // Flag to skip only this one space.
                           // FIXME: This was probably supposed to be -1.
                           spaceflag = 0;
                        }
                        // Allow for spaces before a scope operator.
                        int namespace_tagnum;
                        if (!strchr(statement, '.') && !strstr(statement, "->")) {
                           namespace_tagnum = G__defined_tagname(statement, 2);
                        }
                        else {
                           namespace_tagnum = -1;
                        }
                        if (
                           ((namespace_tagnum != -1) && (G__struct.type[namespace_tagnum] == 'n')) ||
                           !strcmp(statement, "std")
                        ) {
                           // Flag to skip only this one space.
                           // FIXME: This was probably supposed to be -1.
                           spaceflag = 0;
                        }
                     }
                  }
                  // FIXME: Should probably test iout here, statement may be empty.
                  // FIXME: We may have already processed statement, this should be skipped in that case.
                   {
                     char* replace = (char*) G__replacesymbol(statement);
                     if (replace != statement) {
                        strcpy(statement, replace);
                        iout = strlen(statement);
                        goto after_replacement;
                     }
                  }
                  // Record that we tried to take action on a space and we were not skipping code.
                  ++spaceflag;
                  G__var_type = 'p';
               }
            }
            // If user requested a return, then do so.
            if (G__return > G__RETURN_NORMAL) {
               return G__null;
            }
            break;
      
         case ';':
            if (single_quote || double_quote) {
               statement[iout++] = current_char;
            }
            else {
               // -- We have reached the end of the statement.
               statement[iout] = '\0';
               //fprintf(stderr, "G__exec_statement: seen ';': G__no_exec: %d mparen: %d statement: '%s'\n", G__no_exec, *mparen, statement);
               if (
                  G__breaksignal &&
                  (G__beforelargestep(statement, &iout, &largestep) > 1)
               ) {
                  return G__null;
               }
               if (!G__no_exec) {
                  // -- We are not skipping.
                  // Check for the statements which change the flow of control.
                  switch (iout) {
                     case 5:
                        // -- Check for break and throw.
                        if (!strcmp(statement, "break")) {
                           // -- Handle 'break;'
                           //                 ^
#ifdef G__ASM
                           if (G__asm_noverflow) {
                              // -- Bytecode compilation is enabled.
#ifdef G__ASM_DBG
                              if (G__asm_dbg) {
                                 G__fprinterr(G__serr, "%3x,%3x: JMP (for break, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                              }
#endif // G__ASM_DBG
                              // Generate the jump out of the mainline of the loop or switch.
                              G__asm_inst[G__asm_cp] = G__JMP;
                              // Remember where to backpatch the destination.
                              //fprintf(stderr, "G__exec_statement: Begin store for break src: %x isbreak: %d G__pbreakcontinue: %p\n", G__asm_cp + 1, 1, G__pbreakcontinue);
                              G__store_breakcontinue_list(G__asm_cp + 1, 1 /* isbreak */);
                              G__inc_cp_asm(2, 0);
                              //fprintf(stderr, "G__exec_statement: End store for break G__pbreakcontinue: %p\n", G__pbreakcontinue);
                           }
#endif // G__ASM
                           if (!G__no_exec_compile) {
                              // -- We are *not* just generating bytecode, take action.
                              // Stop parse and take semantic action now.
                              //fprintf(stderr, "End   G__exec_statement: a break statement was executed.\n");
                              return G__block_break;
                           }
                           // Reset the statement buffer.
                           statement[0] = '\0';
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = 0;
                        }
                        else if (!strcmp(statement, "throw")) {
                           G__nosupport("Exception handling");
                           G__return = G__RETURN_NORMAL;
                           return G__null;
                        }
                        break;
                     case 6:
                        // -- Check for return.
                        if (!strcmp(statement, "return")) {
                           result = G__return_value("");
                           if (G__no_exec_compile) {
                              statement[0] = '\0';
                              if (!*mparen) {
                                 return G__null;
                              }
                              // Reset the statement buffer.
                              iout = 0;
                              // Flag that any following whitespace does not trigger any semantic action.
                              // FIXME: This should be spaceflag = 0!
                              spaceflag = -1;
                              break;
                           }
                           return result;
                        }
                        break;
                     case 8:
                        // -- Check for continue.
                        if (!strcmp(statement, "continue")) {
                           // -- Handle 'continue;'
                           //                    ^
                           //fprintf(stderr, "G__exec_statement: Found a 'continue' statement.\n");
#ifdef G__ASM
                           if (G__asm_noverflow) {
                              // -- Bytecode compilation is enabled.
#ifdef G__ASM_DBG
                              if (G__asm_dbg) {
                                 G__fprinterr(G__serr, "%3x,%3x: JMP (for continue, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                              }
#endif // G__ASM_DBG
                              // Generate the jump out of the mainline of the loop or switch.
                              G__asm_inst[G__asm_cp] = G__JMP;
                              // Remember where to backpatch the destination.
                              //fprintf(stderr, "G__exec_statement: Begin store for continue src: %x isbreak: %d G__pbreakcontinue: %p\n", G__asm_cp + 1, 1, G__pbreakcontinue);
                              G__store_breakcontinue_list(G__asm_cp + 1, 0 /* isbreak */);
                              G__inc_cp_asm(2, 0);
                              //fprintf(stderr, "G__exec_statement: End store for continue G__pbreakcontinue: %p\n", G__pbreakcontinue);
                           }
#endif // G__ASM
                           if (!G__no_exec_compile) {
                              // -- We are *not* just generating bytecode, take action.
                              // Stop parse and take semantic action now.
                              //fprintf(stderr, "End   G__exec_statement: a continue statement was executed.\n");
                              return G__block_continue;
                           }
                           // Reset the statement buffer.
                           statement[0] = '\0';
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = 0;
                        }
                        break;
                  }
                  if (!strncmp(statement, "return\"", 7) || !strncmp(statement, "return'", 7)) {
                     result = G__return_value(statement + 6);
                     if (G__no_exec_compile) {
                        statement[0] = '\0';
                        if (!*mparen) {
                           return G__null;
                        }
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        // FIXME: This should be spaceflag = 0!
                        spaceflag = -1;
                        break;
                     }
                     return result;
                  }
                  // Ok, not a statement which changes the flow of control.
                  if (statement[0] && iout) {
                     // -- We have an expression statement, evaluate the expression.
#ifdef G__ASM
                     if (G__asm_noverflow) {
                        G__asm_clear();
                     }
#endif // G__ASM
                     result = G__getexpr(statement);
                     if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
                        G__free_tempobject();
                     }
                  }
               }
               if (largestep) {
                  G__afterlargestep(&largestep);
               }
               if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                  return result;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            break;

         case '=':
            if (
               !G__no_exec &&
               ((iout < 8) || (iout > 9) || strncmp(statement, "operator", 8)) &&
               !single_quote &&
               !double_quote
            ) {
               // -- Handle an assignment.
               statement[iout] = '=';
               current_char = G__fgetstream_new(statement + iout + 1, ";,{}");
               if ((current_char == '}') || (current_char == '{')) {
                  G__syntaxerror(statement);
                  --*mparen;
                  current_char = ';';
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
               if (
                  (G__breaksignal && (G__beforelargestep(statement, &iout, &largestep) > 1)) ||
                  (G__return > G__RETURN_NORMAL)
               ) {
                  return G__null;
               }
#ifdef G__ASM
               if (G__asm_noverflow) {
                  G__asm_clear();
               }
#endif // G__ASM
               result = G__getexpr(statement);
               if ((G__p_tempbuf->level >= G__templevel) && G__p_tempbuf->prev) {
                  G__free_tempobject();
               }
               if (largestep) {
                  G__afterlargestep(&largestep);
               }
               if ((!*mparen && (current_char == ';')) || (G__return > G__RETURN_NORMAL)) {
                  return result;
               }
            }
            else if (G__prerun && !single_quote && !double_quote) {
               current_char = G__fignorestream(";,}");
               if ((current_char == '}') && *mparen) {
                  --*mparen;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            else {
               statement[iout++] = current_char;
               // Flag that any following whitespace should trigger a semantic action.
               spaceflag |= 1;
            }
            break;

         case ')':
            if (G__ansiheader) {
               G__ansiheader = 0;
               return G__null;
            }
            // FIXME: We should test G__funcheader here as well!
            statement[iout++] = current_char;
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
            break;

         case '(':
            //fprintf(stderr, "\nG__exec_statement: Enter left parenthesis case.\n");
            statement[iout++] = current_char;
            statement[iout] = '\0';
            if (single_quote || double_quote) {
               break;
            }
            if (!G__no_exec) {
               // -- We are not skipping code.
               // FIXME: These tests are weird, why is a function-style macro not allowed in a class declaration?
               if (!G__def_struct_member || strncmp(statement, "ClassDef", 8)) {
                  // -- Check for a function-style macro call.
                  int found = G__execfuncmacro_noexec(statement);
                  if (found) {
                     // -- We have switched to the macro file and position, keep going.
                     // Reset the statement buffer.
                     iout = 0;
                     // Flag that any following whitespace does not trigger any semantic action.
                     spaceflag = 0;
                     break;
                  }
               }
               G__ASSERT(!G__decl || (G__decl == 1));
               if (G__prerun && !G__decl) {
                  if (iout == 12) {
                     // -- Handle _attribute_.
                     if (!strcmp(statement, "_attribute_(")) {
                        // -- Handle '_attribute_( ... )'.
                        //                       ^
                        // also get (...)
                        G__fignorestream(")");
                        // We just ignore it.
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = -1;
                        break;
                     }
                  }
                  // -- Make ifunc table at prerun and skip out.
                  // We have: 'functionname(...)'
                  G__var_type = 'i';
                  G__make_ifunctable(statement);
                  // For member function definition.
                  if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                     return G__null;
                  }
                  // Reset the statement buffer.
                  iout = 0;
                  // Flag that any following whitespace does not trigger any semantic action.
                  spaceflag = 0;
                  break;
               }
               // Handle _attribute(,return(, switch(, if(, while(, catch(, throw(, for(.
               switch (iout) {
                  case 12:
                     // -- Handle _attribute_.
                     if (!strcmp(statement, "_attribute_(")) {
                        // -- Handle '_attribute_( ... )'.
                        //                       ^
                        // also get (...)
                        G__fignorestream(")");
                        // We just ignore it.
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = -1;
                        break;
                     }
                  case 7:
                     if (!strcmp(statement, "return(")) {
                        //
                        // following part used to be
                        // G__fgetstream(statement,")");
                        //
                        fseek(G__ifile.fp, -1, SEEK_CUR);
                        if (G__dispsource) {
                           G__disp_mask = 1;
                        }
                        G__fgetstream_new(statement, ";");
                        result = G__return_value(statement);
                        if (G__no_exec_compile) {
                           statement[0] = '\0';
                           if (!*mparen) {
                              return G__null;
                           }
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           // FIXME: This should be spaceflag = 0!
                           spaceflag = -1;
                           break;
                        }
                        return result;
                     }
                     if (!strcmp(statement, "switch(")) {
                        //fprintf(stderr, "G__exec_statement: Recognized switch.\n");
                        result = G__exec_switch();
                        if (!*mparen || (G__return > G__RETURN_NON)) {
                           return result;
                        }
                        if(G__value_typenum(result)==G__value_typenum(G__block_goto) &&
                           result.ref==G__block_goto.ref) {
                           int found = G__search_gotolabel(0, &start_pos, start_line, mparen);
                           // If found, continue parsing, the input file is now
                           // positioned immediately after the colon of the label.
                           // Otherwise, return and let our caller try to find it.
                           if (!found) {
                              // -- Not found, maybe our caller can find it.
                              return G__block_goto;
                           }
                        }
                        if(G__value_typenum(result)==G__value_typenum(G__block_break)) {
                           if (result.obj.i == G__BLOCK_CONTINUE) {
                              // -- The body did a continue, return immediately.
                              //fprintf(stderr, "G__exec_statement: Switch body did a 'continue', returning now.\n");
                              return result;
                           }
                        }
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = 0;
                     }
                     break;
                   case 6:
                     if (!strcmp(statement, "while(")) {
                        result = G__exec_while();
                        if ((G__return > G__RETURN_NON) || !*mparen) {
                           return result;
                        }
                        // Check for break, continue, or goto executed during statement.
                        if(G__value_typenum(result)==G__value_typenum(G__block_goto) &&
                           result.ref==G__block_goto.ref) {
                           int found = G__search_gotolabel(0, &start_pos, start_line, mparen);
                           // If found, continue parsing, the input file is now
                           // positioned immediately after the colon of the label.
                           // Otherwise, return and let our caller try to find it.
                           if (!found) {
                              // -- Not found, maybe our caller can find it.
                              return G__block_goto;
                           }
                        }
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = 0;
                     }
                     if (!strcmp(statement, "catch(")) {
                        G__ignore_catch();
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = 0;
                     }
                     if (!strcmp(statement, "throw(")) {
                        current_char = G__fignorestream(")");
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = 0;
                     }
                     break;
                  case 5:
                     if ((*mparen == 1) && !strcmp(statement, "case(")) {
                        // -- Handle keyword "case", even if skipping code.
                        //fprintf(stderr, "G__exec_statement: Saw a case.\n");
                        // Scan the case label in.
                        G__StrBuf casepara_sb(G__ONELINE);
                        char *casepara = casepara_sb;
                        casepara[0] = '(';
                        {
                           int lencasepara = 1;
                           current_char = G__fgetstream(casepara+lencasepara, ":");
                           if (current_char == ')') {
                              lencasepara = strlen(casepara);
                              casepara[lencasepara] = ')';
                              ++lencasepara;
                              G__fgetstream(casepara+lencasepara, ":");
                           }
                           current_char = G__fgetc();
                           while (current_char == ':') {
                              strcat(casepara, "::");
                              lencasepara = strlen(casepara);
                              G__fgetstream(casepara + lencasepara, ":");
                              current_char = G__fgetc();
                           }
                        }
                        // Backup one character.
                        fseek(G__ifile.fp, -1, SEEK_CUR);
                        // Flag that we have already displayed it, do not show it again.
                        G__disp_mask = 1;
                        //
                        //  If we are not in a switch statement,
                        //  then we are done.
                        if (G__switch) {
                           // -- We are in a switch statement, evaluate the case expression.
                           G__value local_result = G__exec_switch_case(casepara);
                           if (G__switch_searching) {
                              // -- We are searching for a matching case, return value of case expression.
                              return local_result;
                           }
                        }
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = 0;
                     }
                     break;
                  case 4:
                     if (!strcmp(statement, "for(")) {
                        result = G__exec_for();
                        if ((G__return > G__RETURN_NON) || !*mparen) {
                           return result;
                        }
                        // Check for break, continue, or goto executed during statement.
                        if(G__value_typenum(result)==G__value_typenum(G__block_goto) &&
                           result.ref==G__block_goto.ref) {
                           int found = G__search_gotolabel(0, &start_pos, start_line, mparen);
                           // If found, continue parsing, the input file is now
                           // positioned immediately after the colon of the label.
                           // Otherwise, return and let our caller try to find it.
                           if (!found) {
                              // -- Not found, maybe our caller can find it.
                              return G__block_goto;
                           }
                        }
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = 0;
                     }
                     break;                     
                  case 3:
                     if (!strcmp(statement, "if(")) {
                        result = G__exec_if();
                        if ((G__return > G__RETURN_NON) || !*mparen) {
                           return result;
                        }
                        // Check for break, continue, or goto executed during statement.
                        /************************************
                        * handling break statement
                        * switch(),do,while(),for() {
                        *    if(cond) break; or continue;
                        *    if(cond) {break; or continue;}
                        * } G__fignorestream() skips until here
                        *************************************/
                        if(G__value_typenum(result)==G__value_typenum(G__block_break)) {
                           // -- Statement block was exited by break, continue, or goto.
                           if (result.ref == G__block_goto.ref) {
                              // -- Exited by goto.
                              int found = G__search_gotolabel(0, &start_pos, start_line, mparen);
                              // If found, continue parsing, the input file is now
                              // positioned immediately after the colon of the label.
                              // Otherwise, return and let our caller try to find it.
                              if (!found) {
                                 // -- Not found, maybe our caller can find it.
                                 return G__block_goto;
                              }
                           }
                           else {
                              // -- Exited by break or continue.
                              //fprintf(stderr, "G__exec_statement: break or continue executed from an if.\n");
                              //fprintf(stderr, "End   G__exec_statement (from end of if, break or continue seen).\n");
                              return result;
                           }
                        }
                        // Reset the statement buffer.
                        iout = 0;
                        // Flag that any following whitespace does not trigger any semantic action.
                        spaceflag = 0;
                     }
                     break;
               }
               //
               // Not return, switch, if, while, catch, throw, or for,
               // so this must be either a function call, a function-style
               // initializer, or an assignment where the right-hand side
               // starts with a function call.
               //
               // Note:
               //
               //      classtype a(1); done elsewhere
               //      a = b();        another case in the switch
               //
               if (
                  (iout > 1) && // Not handled above and is not empty, and
                  isalpha(statement[0]) && // First character is a letter (FIXME: We should allow underscore here!), and
                  !strchr(statement, '[') &&  // There is no subscript operator, and
                  strncmp(statement, "cout<<", 6) && // We are not 'cout<<', and
                  (
                     !strstr(statement, "<<") || // There is no left shift operator, or
                     !strcmp(statement, "operator<<(") // the function is 'operator<<'
                  ) && // and,
                  (
                     !strstr(statement, ">>") || // There is no right shift operator, or
                     !strcmp(statement, "operator>>(") // the function is 'operator>>'
                  )
               ) {
                  // Read to 'func(xxxxxx)'
                  //                     ^
                  current_char = G__fgetstream_new(statement + iout , ")");
                  iout = strlen(statement);
                  statement[iout++] = current_char;
                  statement[iout] = '\0';
                  // Skip any following whitespace.
                  current_char = G__fgetspace();
                  // if 'func(xxxxxx) \n    nextexpr'   macro
                  // if 'func(xxxxxx) ;'                func call
                  // if 'func(xxxxxx) operator xxxxx'   func call + operator
                  // if 'new (arena) type'              new operator with arena
                  if (!strncmp(statement, "new ", 4) || !strncmp(statement, "new(", 4)) {
                     // -- We have a new expression, either placement or parenthesized typename.
                     // Grab the rest of the line.
                     statement[iout++] = current_char;
                     current_char = G__fgetstream_template(statement + iout, ";");
                     // Find the position of the first open parenthesis.
                     char* pnew = strchr(statement, '(');
                     G__ASSERT(pnew);
                     // And pass it to the new expression parser.
                     // FIXME: What if this is a type with parenthesis in it, not a placement new?
                     result = G__new_operator(pnew);
                  }
                  else {
                     // -- Evaluate the expression which contains the function call.
                     //fprintf(stderr, "G__exec_statement: Calling G__exec_function: '%s'\n", statement);
                     int notcalled = G__exec_function(statement, &current_char, &iout, &largestep, &result);
                     if (notcalled) {
                        return G__null;
                     }
                  }
                  if ((!*mparen && (current_char == ';')) || (G__return > G__RETURN_NON)) {
                     return result;
                  }
                  // Reset the statement buffer.
                  iout = 0;
                  // Flag that any following whitespace does not trigger any semantic action.
                  spaceflag = 0;
               }
               else if (iout > 3) {
                  // -- Nope, just a parenthesized construct, accumulate it and keep going.
                  current_char = G__fgetstream_new(statement + iout, ")");
                  iout = strlen(statement);
                  statement[iout++] = current_char;
               }
            }
            else if (!*mparen && (iout == 3) && !strcmp(statement, "if(")) {
               // -- We are skipping code, and we have found an if at the top level.
               //  if(true);
               //  else if() ;  skip this
               //         ^
               //       else ;  skip this too
               result = G__exec_else_if();
               if (!*mparen || (G__return > G__RETURN_NON)) {
                  return result;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            } else if ((*mparen == 1) && !strcmp(statement, "case(")) {
               // -- Handle keyword "case", even if skipping code.
               //fprintf(stderr, "G__exec_statement: Saw a case.\n");
               // Scan the case label in.
               G__StrBuf casepara_sb(G__ONELINE);
               char *casepara = casepara_sb;
               casepara[0] = '(';
               {
                  int lencasepara = 1;
                  current_char = G__fgetstream(casepara+lencasepara, ":");
                  if (current_char==')') {
                     lencasepara = strlen(casepara);
                     casepara[lencasepara] = ')';
                     ++lencasepara;
                     G__fgetstream(casepara+lencasepara, ":");
                  }
                  current_char = G__fgetc();
                  while (current_char == ':') {
                     strcat(casepara, "::");
                     lencasepara = strlen(casepara);
                     G__fgetstream(casepara + lencasepara, ":");
                     current_char = G__fgetc();
                  }
               }
               // Backup one character.
               fseek(G__ifile.fp, -1, SEEK_CUR);
               // Flag that we have already displayed it, do not show it again.
               G__disp_mask = 1;
               //
               //  If we are not in a switch statement,
               //  then we are done.
               if (G__switch) {
                  // -- We are in a switch statement, evaluate the case expression.
                  G__value local_result = G__exec_switch_case(casepara);
                  if (G__switch_searching) {
                     // -- We are searching for a matching case, return value of case expression.
                     return local_result;
                  }
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            else {
               // -- We are skipping code.
               //fprintf(stderr, "G__exec_statement: Skipping a parenthesized construct.\n");
               int paren_start_line = G__ifile.line_number;
               current_char = G__fignorestream(")");
               if (current_char != ')') {
                  char* msg = new char[70];
                  sprintf(msg, "Error: Cannot find matching ')' for '(' on line %d.", paren_start_line);
                  G__genericerror(msg);
                  delete[] msg;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            break;
         case '{':
            if (G__funcheader == 1) {
               // Return if called from G__interpret_func()
               // as pass parameter declaration.
               G__unsigned = 0;
               G__constvar = 0;
               return G__start;
            }
            if (single_quote || double_quote) {
               statement[iout++] = current_char;
            }
            else {
               G__constvar = 0;
               G__static_alloc = 0;
               statement[iout] = '\0';
               ++*mparen;
               //fprintf(stderr, "G__exec_statement: seen '{': G__no_exec: %d mparen: %d: statement '%s'\n", G__no_exec, *mparen, statement);
               if (!G__no_exec) {
                  // -- We are not skipping code, check for things that could precede an open curly brace.
                  if ((iout == 2) && !strcmp(statement, "do")) {
                     // -- We have 'do {...} while (...);'.
                     //                ^
                     // Backup the curly brace.
                     fseek(G__ifile.fp, -1, SEEK_CUR);
                     // Undo the nesting increment.
                     --*mparen;
                     // Flag that we should not redisplay the curly brace.
                     if (G__dispsource) {
                        // -- Displaying source, do not display the next character, we have already shown it.
                        G__disp_mask = 1;
                     }
                     goto do_do;
                  }
                  if (
                     ((iout == 4) && !strcmp(statement, "enum"))  ||
                     ((iout == 5) && !strcmp(statement, "class")) ||
                     ((iout == 6) && !strcmp(statement, "struct"))
                  ) {
                     // Backup the curly brace.
                     fseek(G__ifile.fp, -1, SEEK_CUR);
                     // Undo the nesting increment.
                     --*mparen;
                     // Flag that we should not redisplay the curly brace.
                     if (G__dispsource) {
                        // -- Displaying source, do not display the next character, we have already shown it.
                        G__disp_mask = 1;
                     }
                     G__DEFSTR(statement[0]);
                     spaceflag=0;
                     break;
                  }
                  if ((iout == 8) && !strcmp(statement, "namespace")) {
                     // Treat unnamed namespace as global scope.
                     // This implementation may be wrong.
                     // Should fix later with using directive in global scope.
                  }
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            break;
      
         case '}':
            if (single_quote || double_quote) {
               statement[iout++] = current_char;
            }
            else {
               --*mparen;
               //fprintf(stderr, "G__exec_statement: seen '}': G__no_exec: %d mparen: %d\n", G__no_exec, *mparen);
               if (*mparen <= 0) {
                  if (iout && (G__globalcomp == G__NOLINK)) {
                     statement[iout] = '\0';
                     G__missingsemicolumn(statement);
                  }
                  if (*mparen < 0) {
                     G__genericerror("Error: Too many '}'");
                  }
                  //fprintf(stderr, "End   G__exec_statement (from '}').\n");
                  return result;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            break;

         case '"':
            // -- Handle #include "..." and return "...".
            //                    ^                ^
            if ((iout == 8) && !strcmp(statement, "#include")) {
               fseek(G__ifile.fp, -1, SEEK_CUR);
               if (G__dispsource) {
                  G__disp_mask = 1;
               }
               G__include_file();
               if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                  return G__null;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            if ((iout == 6) && !strcmp(statement, "return")) {
               fseek(G__ifile.fp, -1, SEEK_CUR);
               if (G__dispsource) {
                  G__disp_mask = 1;
               }
               G__fgetstream_new(statement, ";");
               result = G__return_value(statement);
               if (G__no_exec_compile) {
                  if (!*mparen) {
                     return G__null;
                  }
                  // Reset the statement buffer.
                  iout = 0;
                  // Flag that any following whitespace does not trigger any semantic action.
                  // FIXME: This should be spaceflag = 0;, it got cut and pasted from the semantic action on space section inappropriately!
                  spaceflag = -1;
                  break;
               }
               if (!G__prerun) {
                  return result;
               }
               G__fignorestream("}");
               return result;
            }
            statement[iout++] = current_char;
            // Flag that any following whitespace should trigger a semantic action.
            // FIXME: Why this strange value?
            spaceflag = 5;
            if (!single_quote) {
               if (!double_quote) {
                  double_quote = 1;
                  if (G__prerun) {
                     conststring = statement + (iout - 1);
                  }
               }
               else {
                  double_quote = 0;
                  if (G__prerun) {
                     statement[iout] = '\0';
                     G__strip_quotation(conststring);
                  }
               }
            }
            break;
      
         case '\'':
            if (!double_quote) {
               single_quote ^= 1;
            }
            statement[iout++] = current_char;
            // Flag that any following whitespace should trigger a semantic action.
            // FIXME: Why this strange value?
            spaceflag = 5;
            break;

         case '/':
            if ((iout > 0) && !double_quote && (statement[iout-1] == '/') && commentflag) {
               iout--;
               if (!iout) {
                  // Flag that any following whitespace should not trigger a semantic action.
                  spaceflag = 0;
               }
               G__fignoreline();
            }
            else {
               commentflag = 1;
               statement[iout++] = current_char;
               // Flag that any following whitespace should trigger a semantic action.
               spaceflag |= 1;
            }
            break;
      
         case '*':
            if ((iout > 0) && !double_quote && (statement[iout-1] == '/') && commentflag) {
               // start commenting out
               //fprintf(stderr, "G__exec_statement: Seen a C-style comment.\n");
               iout--;
               if (!iout) {
                  // Flag that any following whitespace should not trigger a semantic action.
                  spaceflag = 0;
               }
               //{
               //   char buf[128];
               //   G__fgetstream_peek(buf, 10);
               //   fprintf(stderr, "G__exec_statement: peek '%s'\n", buf);
               //}
               int stat = G__skip_comment();
               //{
               //   char buf[256];
               //   G__fgetstream_peek(buf, 10);
               //   fprintf(stderr, "G__exec_statement: peek '%s'\n", buf);
               //}
               if (stat == EOF) {
                  return G__null;
               }
            }
            else {
               statement[iout++] = current_char;
               // Flag that any following whitespace should trigger a semantic action.
               spaceflag |= 1;
               if (!double_quote && !single_quote) {
                  add_fake_space = 1;
               }
            }
            break;

         case '&':
            statement[iout++] = current_char;
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
            if (!double_quote && !single_quote) {
               add_fake_space = 1;
            }
            break;
      
         case ':':
            statement[iout++] = current_char;
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
            if (!double_quote && !single_quote) {
               statement[iout] = '\0';
               int stat = G__label_access_scope(statement, &iout, &spaceflag, *mparen);
               if (stat) {
                  return G__null;
               }
            }
            break;

#ifdef G__TEMPLATECLASS
         case '<':
            statement[iout] = '\0';
            if ((iout == 8) && !strcmp(statement, "template")) {
               G__declare_template();
               if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                  return G__null;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            else if ((iout == 8) && !strcmp(statement, "#include")) {
               fseek(G__ifile.fp, -1, SEEK_CUR);
               if (G__dispsource) {
                  G__disp_mask = 1;
               }
               G__include_file();
               if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                  return G__null;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            else {
               char* s = strchr(statement, '=');
               if (s) {
                  ++s;
               }
               else {
                  s = statement;
               }
               if (
                  ((spaceflag == 1) || (spaceflag == 2)) &&
                  !G__no_exec &&
                  (
                     (
                        (statement[0] == '~') &&
                        G__defined_templateclass(s + 1)
                     ) ||
                     G__defined_templateclass(s)
                  )
               ) {
                  // --
                  spaceflag = 1;
                  //
                  //   X  if(a<b);
                  //   X  func(a<b);
                  //   X  a=a<b;
                  //   x  cout<<a;   this is rare.
                  //   O  Vector<Array<T> >  This has to be handled here
                  //
                  statement[iout++] = current_char;
                  current_char = G__fgetstream_template(statement + iout, ">");
                  G__ASSERT(c == '>');
                  iout = strlen(statement);
                  if (statement[iout-1] == '>') {
                     statement[iout++] = ' ';
                  }
                  statement[iout++] = current_char;
                  spaceflag = 1;
                  // Try to accept statements with no space between
                  // the closing '>' and the identifier. Ugly, though.
                  {
                     fpos_t xpos;
                     fgetpos(G__ifile.fp, &xpos);
                     current_char = G__fgetspace();
                     fsetpos(G__ifile.fp, &xpos);
                     if (isalpha(current_char) || (current_char == '_')) {
                        current_char = ' ';
                        goto read_again;
                     }
                  }
               }
               else {
                  statement[iout++] = current_char;
                  // Flag that any following whitespace should trigger a semantic action.
                  spaceflag |= 1;
               }
            }
            break;
#endif /* G__TEMPLATECLASS */
      
         case EOF:
            {
               statement[iout] = '\0';
               G__eof = 1;
               int mparen_line = 0;
               if (*mparen) {
                 mparen_line = mparen_lines.top();
               }
               if (*mparen || single_quote || double_quote) {
                  // -- We still have open braces or quotes.
                  if (*mparen) {
                     // -- We have open braces.
                     if (*mparen > 1) {
                        // -- More than one open brace.
                        G__fprinterr(G__serr, "Error: %d closing braces are missing, for the block opened around line %d.\n", *mparen, mparen_line);
                     }
                     else {
                        // -- Only one open brace.
                        G__fprinterr(G__serr, "Error: Missing closing brace for the block opened around line %d.\n", mparen_line);
                     }
                  }
                  G__unexpectedEOF("G__exec_statement()");
               }
               if (
                  strcmp(statement, "") && // Not the empty statement, and
                  strcmp(statement, "#endif") && // Not an endif, and
                  (statement[0] != 0x1a /*FIXME: What is this? */) // Not ???
               ) {
                  G__fprinterr(G__serr, "Report: Unrecognized string '%s' ignored", statement);
                  G__printlinenum();
               }
            }
            return G__null;

         case '\\':
            statement[iout++] = current_char;
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
            current_char = G__fgetc();
            // Continue to default case.

         default:
            //fprintf(stderr, "G__exec_statement: Enter default case.\n");
            // Make sure that the delimiters that have not been treated
            // in the switch statement do drop the discarded_space.
            if (discarded_space && (current_char != '[') && (current_char != ']') && iout && (statement[iout-1] != ':')) {
               // -- Since the character following a discarded space
               // is *not* a separator, we have to keep the space.
               statement[iout++] = ' ';
            }
            statement[iout++] = current_char;
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
#ifdef G__MULTIBYTE
            if (G__IsDBCSLeadByte(current_char)) {
               current_char = G__fgetc();
               G__CheckDBCS2ndByte(current_char);
               statement[iout++] = current_char;
            }
#endif // G__MULTIBYTE
            break;
      }
      discarded_space = discard_space;
   }  
}

//______________________________________________________________________________
extern "C" void G__alloc_tempobject(int tagnum, int typenum)
{
   // Used for interpreted classes.
   if (G__xrefflag) {
      return;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__alloc_tempobject");
   }
#endif // G__ASM_DBG
   (void) typenum; // Force typenum to be used in case G__ASM_DBG is turned off.
   G__tempobject_list* store_p_tempbuf = G__p_tempbuf;
   G__p_tempbuf = (G__tempobject_list*) calloc(1, sizeof(G__tempobject_list));
   G__p_tempbuf->prev = store_p_tempbuf;
   G__p_tempbuf->level = G__templevel;
   G__p_tempbuf->cpplink = 0; // Note: This says we are holding an interpreted object.
   G__p_tempbuf->no_exec = G__no_exec_compile;
   G__p_tempbuf->obj.obj.i = (long) calloc(1, (size_t) G__struct.size[tagnum]);
   G__p_tempbuf->obj.ref = G__p_tempbuf->obj.obj.i;
   G__value_typenum(G__p_tempbuf->obj) = G__Dict::GetDict().GetType(tagnum);
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "\nG__alloc_tempobject: level: %d tagnum: %d  typenum: %d  typename: '%s'  addr: 0x%lx  %s:%d\n", G__p_tempbuf->level, tagnum, typenum, G__value_typenum(G__p_tempbuf->obj).Name(::Reflex::SCOPED).c_str(), G__p_tempbuf->obj.obj.i, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__alloc_tempobject");
   }
#endif // G__ASM_DBG
   // --
}

//______________________________________________________________________________
extern "C" void G__free_tempobject()
{
   // The only 2 potential risks of making this static are
   // - a destructor indirectly calls G__free_tempobject
   // - multi-thread application (but CINT is not multi-threadable anyway).
   static char statement[G__ONELINE];
   if (
      G__xrefflag || // Generating a variable cross-reference, or
      (
         G__command_eval && // Executing temporary code from the command line, and
         (G__ifswitch != G__DOWHILE) // we are not in a do {} while (); construct.
      )
   ) {
      return;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__free_tempobject");
   }
#endif // G__ASM_DBG
   while (
      (G__p_tempbuf->level >= G__templevel) && // Temp was allocated at current level or deeper, and
      G__p_tempbuf->prev // this is not the temp list head.
   ) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "G__free_tempobject: typename '%s'  addr: 0x%lx  %s:%d\n", G__value_typenum(G__p_tempbuf->obj).Name(::Reflex::SCOPED).c_str(), G__p_tempbuf->obj.obj.i, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
#ifdef G__ASM
      if (
         G__asm_noverflow
#ifndef G__ASM_IFUNC
         && G__p_tempbuf->cpplink
#endif // G__ASM_IFUNC
         // --
      ) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SETTEMP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SETTEMP;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      {
         ::Reflex::Scope store_tagnum = G__tagnum;
         char* store_struct_offset = G__store_struct_offset;
         int store_return = G__return;
         G__tagnum = G__value_typenum(G__p_tempbuf->obj).RawType();
         G__store_struct_offset = (char*) G__p_tempbuf->obj.obj.i;
         G__return = G__RETURN_NON;
         if (!G__p_tempbuf->no_exec || G__no_exec_compile) {
            if (G__dispsource) {
               G__fprinterr(G__serr, "\nG__free_tempobject: destroy temp object: typename: '%s'  addr: 0x%lx created at level: %d destroyed while at level: %d  %s:%d\n", G__struct.name[G__get_tagnum(G__tagnum)], G__p_tempbuf->obj.obj.i, G__p_tempbuf->level, G__templevel, __FILE__, __LINE__);
            }
            sprintf(statement, "~%s()", G__struct.name[G__get_tagnum(G__tagnum)]);
            int known = 0;
            G__getfunction(statement, &known, G__TRYDESTRUCTOR); // Call the destructor.
         }
         G__return = store_return;
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;
      }
#ifdef G__ASM
      if (
         G__asm_noverflow
#ifndef G__ASM_IFUNC
         && G__p_tempbuf->cpplink
#endif // G__ASM_IFUNC
         // --
      ) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: FREETEMP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__FREETEMP;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      if (
         !G__p_tempbuf->cpplink && // This is an object of an interpreted class type, and
         G__p_tempbuf->obj.obj.i // we have allocated memory for the object.
      ) {
         free((void*) G__p_tempbuf->obj.obj.i); // Free the object storage.
      }
      G__tempobject_list* tmp = G__p_tempbuf->prev;
      free(G__p_tempbuf); // Free the temp list entry.
      G__p_tempbuf = tmp; // And move to the next entry in the list.
   }
   if (G__dispsource) {
      G__fprinterr(G__serr, "\nG__free_tempobject: End of temp object list.\n");
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__free_tempobject");
   }
#endif // G__ASM_DBG
   // --
}

//______________________________________________________________________________
extern "C" void G__store_tempobject(G__value reg)
{
   // Used for precompiled classes.
   if (G__xrefflag) {
      return;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__store_tempobject");
   }
#endif // G__ASM_DBG
   G__tempobject_list* tmp = G__p_tempbuf;
   G__p_tempbuf = (G__tempobject_list*) calloc(1, sizeof(G__tempobject_list));
   G__p_tempbuf->prev = tmp;
   G__p_tempbuf->level = G__templevel;
   G__p_tempbuf->cpplink = 1; // Note: This means that this routine can only be used for compiled classes.
   G__p_tempbuf->no_exec = G__no_exec_compile;
   G__p_tempbuf->obj = reg; // Initialize the temp object.
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "\nG__store_tempobject: level: %d  cpplink: 1  no_exec: %d  typename '%s'  addr: 0x%lx  %s:%d\n", G__p_tempbuf->level, G__no_exec_compile, G__value_typenum(G__p_tempbuf->obj).Name(::Reflex::SCOPED).c_str(), G__p_tempbuf->obj.obj.i, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__store_tempobject");
   }
#endif // G__ASM_DBG
   // --
}

//______________________________________________________________________________
extern "C" void G__alloc_tempobject_val(G__value* val)
{
   if (G__xrefflag) {
      return;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__alloc_tempobject_val");
   }
#endif // G__ASM_DBG
   G__tempobject_list* tmp = G__p_tempbuf;
   G__p_tempbuf = (G__tempobject_list*) calloc(1, sizeof(G__tempobject_list));
   G__p_tempbuf->prev = tmp;
   G__p_tempbuf->level = G__templevel;
   G__p_tempbuf->cpplink = 0; // Note: This means the object we hold is of interpreted class type.
   G__p_tempbuf->no_exec = G__no_exec_compile;
   G__p_tempbuf->obj.obj.i = (long) calloc(1, (size_t) G__struct.size[G__get_tagnum(G__value_typenum(*val))]);
   G__p_tempbuf->obj.ref = G__p_tempbuf->obj.obj.i;
   G__value_typenum(G__p_tempbuf->obj) = G__value_typenum(*val).RawType();
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "\nG_alloc_tempobject_val: level: %d  typename: '%s'  addr: 0x%lx  %s:%d\n", G__p_tempbuf->level, G__value_typenum(*val).Name(::Reflex::SCOPED).c_str(), G__p_tempbuf->obj.obj.i, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__alloc_tempobject_val");
   }
#endif // G__ASM_DBG
   // --
}

//______________________________________________________________________________
static int G__pop_tempobject_imp(bool delobj)
{
   // Free the first entry of the temporary object list, and possibly delete the object as well.
   if (G__xrefflag) {
      return 0;
   }
   if (!G__p_tempbuf->prev) { // Error, attempt to pop the list head.
      return 0;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "G__pop_tempobject_imp: '%s'  addr: 0x%lx\n", G__value_typenum(G__p_tempbuf->obj).Name(::Reflex::SCOPED).c_str(), G__p_tempbuf->obj.obj.i, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__display_tempobject("G__pop_tempobject_imp");
   }
#endif // G__ASM_DBG
   // Free the object buffer only if interpreted classes are stored.
   if (
      delobj && // We have been asked to free the stored object, and
      (G__p_tempbuf->cpplink != -1) && // the store object is not of a precompiled class type, and
      G__p_tempbuf->obj.obj.i // we have an address
   ) {
      free((void*) G__p_tempbuf->obj.obj.i); // free the stored object.
   }
   G__tempobject_list* tmp = G__p_tempbuf->prev;
   free(G__p_tempbuf); // free the temporary list entry
   G__p_tempbuf = tmp;
   return 0;
}

//______________________________________________________________________________
extern "C" int G__pop_tempobject()
{
   // Free the first entry of the temporary object list, and delete the temp object.
   return G__pop_tempobject_imp(true);
}

//______________________________________________________________________________
int G__pop_tempobject_nodel()
{
   // Free the first entry of the temporary object list, and do *not* delete the temp object.
   return G__pop_tempobject_imp(false);
}

//______________________________________________________________________________
extern "C" void G__settemplevel(int val)
{
   G__templevel += val;
}

//______________________________________________________________________________
extern "C" void G__clearstack() 
{
   int store_command_eval = G__command_eval;
   G__command_eval = 0;
   ++G__templevel;
   G__free_tempobject();
   --G__templevel;
   G__command_eval = store_command_eval;
}
   
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
