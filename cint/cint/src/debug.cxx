/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file debug.c
 ************************************************************************
 * Description:
 *  Debugger capability
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include <memory>

extern "C" {

static short G__tempfilenum = G__MAXFILE - 1;

// Static functions.
static int G__findfuncposition(const char* func, int* pline, int* pfnum);
static G__value G__exec_tempfile_core(const char* file, FILE* fp);

// Externally visible functions.
int G__findposition(const char* string, G__input_file* view, int* pline, int* pfnum);
int G__display_proto(FILE* fp, const char* func);
int G__display_proto_pretty(FILE* fp, const char* func, const char friendlyStyle);
int G__beforelargestep(char* statement, int* piout, int* plargestep);
void G__afterlargestep(int* plargestep);
void G__EOFfgetc();
void G__BREAKfgetc();
void G__DISPNfgetc();
void G__DISPfgetc(int c);
int G__lock_variable(const char* varname);
int G__unlock_variable(const char* varname);
G__value G__interactivereturn();
void G__set_tracemode(char* name);
void G__del_tracemode(char* name);
void G__set_classbreak(char* name);
void G__del_classbreak(char* name);
void G__setclassdebugcond(int tagnum, int brkflag);

// Functions in the C interface.
void G__setdebugcond();
int G__gettempfilenum();
G__value G__exec_tempfile_fp(FILE* fp);
G__value G__exec_tempfile(const char* file);
G__value G__exec_text(const char* unnamedmacro);
char* G__exec_text_str(const char* unnamedmacro, char* result);
const char* G__load_text(const char* namedmacro);
int G__setbreakpoint(const char* breakline, const char* breakfile);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static int G__findfuncposition(const char* func, int* pline, int* pfnum)
{
   // -- FIXME: Describe this function!
   size_t lenfunc = strlen(func) + 1;
   G__FastAllocString funcname(func);
   G__FastAllocString scope(lenfunc);
   G__FastAllocString temp(lenfunc);
   char *pc;
   int temp1;
   int tagnum;
   struct G__ifunc_table_internal *ifunc;

   pc = strstr(funcname, "::");

   /* get appropreate scope */
   if (pc) {
      *pc = '\0';
      scope = funcname;
      temp = pc + 2;
      funcname.Swap(temp);
      tagnum = G__defined_tagname(scope, 0);
      if ('\0' == funcname[0] && -1 != tagnum) {
         /* display class declaration */
         *pline = G__struct.line_number[tagnum];
         *pfnum = G__struct.filenum[tagnum];
         return(2);
      }
      else {
         /* class scope A::func , global scope ::func */
         if (-1 == tagnum) ifunc = &G__ifunc;  /* global scope, ::func */
         else {
            G__incsetup_memfunc(tagnum);
            ifunc = G__struct.memfunc[tagnum]; /* specific class */
         }
      }
   }
   else {
      /* global scope */
      ifunc = &G__ifunc;
   }

   while (ifunc) {
      temp1 = 0;
      while (temp1 < ifunc->allifunc) {
         if (strcmp(ifunc->funcname[temp1], funcname) == 0) {
            *pline = ifunc->pentry[temp1]->line_number;
            *pfnum = ifunc->pentry[temp1]->filenum;
            return(2);
         }
         ++temp1;
      }
      ifunc = ifunc->next;
   }
   return(0);
}

struct AsmData {
   long asm_inst_g[G__MAXINST]; /* p-code instruction buffer */
   G__value asm_stack_g[G__MAXSTACK]; /* data stack */
   struct G__input_file ftemp, store_ifile;
};

//______________________________________________________________________________
static G__value G__exec_tempfile_core(const char* file, FILE* fp)
{
   // -- FIXME: Describe this function!
   // --
#ifdef G__EH_SIGNAL
   void(*fpe)();
   void(*segv)();
#ifdef SIGILL
   void(*ill)();
#endif
#ifdef SIGEMT
   void(*emt)();
#endif
#ifdef SIGBUS
   void(*bus)();
#endif
#endif

   std::auto_ptr<AsmData> pAsmData(new AsmData);

   long* asm_inst_g = pAsmData->asm_inst_g; /* p-code instruction buffer */
   G__value* asm_stack_g = pAsmData->asm_stack_g; /* data stack */
   struct G__input_file* ftemp = &pAsmData->ftemp;
   struct G__input_file* store_ifile = &pAsmData->store_ifile;

   char asm_name[G__ASM_FUNCNAMEBUF];

   long *store_asm_inst;
   G__value *store_asm_stack;
   char *store_asm_name;
   int store_asm_name_p;
   struct G__param *store_asm_param;
   /* int store_asm_exec; */
   int store_asm_noverflow;
   int store_asm_cp;
   int store_asm_dt;
   int store_asm_index; /* maybe unneccessary */

   size_t len;

   fpos_t pos;
   char store_var_type;
   G__value buf = G__null;
#ifdef G__ASM
   G__ALLOC_ASMENV;
   (void)store_asm_loopcompile; // "set but not used"
#endif

   G__LockCriticalSection();

   /*************************************************
   * delete space chars at the end of filename
   *************************************************/
   char *filename = 0;
   if (file) {
      len = strlen(file);
      filename = new char[len+1];
      strcpy(filename, file); // Okay, we allocated the right size
      while (len > 1 && isspace(filename[len-1])) {
         filename[--len] = '\0';
      }

#ifndef G__WIN32
      ftemp->fp = fopen(filename, "r");
#else
      ftemp->fp = fopen(filename, "rb");
#endif
   }
   else {
      fseek(fp, 0L, SEEK_SET);
      ftemp->fp = fp;
   }

   if (ftemp->fp) {
      ftemp->vindex = -1;
      ftemp->line_number = 1;
      if (file) {
         G__strlcpy(ftemp->name, filename, sizeof(ftemp->name));
         ftemp->name[sizeof(ftemp->name) - 1] = 0; // ensure termination
         delete [] filename;
      }
      else {
         G__strlcpy(ftemp->name, "(tmpfile)", sizeof(ftemp->name));
      }
      ftemp->filenum = G__tempfilenum;
      G__srcfile[G__tempfilenum].fp = ftemp->fp;
      G__srcfile[G__tempfilenum].filename = ftemp->name;
      G__srcfile[G__tempfilenum].hash = 0;
      G__srcfile[G__tempfilenum].maxline = 0;
      G__srcfile[G__tempfilenum].breakpoint = (char*)NULL;
      --G__tempfilenum;
      if (G__ifile.fp && G__ifile.filenum >= 0) {
         fgetpos(G__ifile.fp, &pos);
      }
      *store_ifile = G__ifile;
      G__ifile = *ftemp;

      /**********************************************
       * interrpret signal handling during inner loop asm exec
       **********************************************/
#ifdef G__ASM
      G__STORE_ASMENV;
#endif
      store_var_type = G__var_type;

      G__var_type = 'p';

#ifdef G__EH_SIGNAL
      fpe = signal(SIGFPE, G__error_handle);
      segv = signal(SIGSEGV, G__error_handle);
#ifdef SIGILL
      ill = signal(SIGILL, G__error_handle);
#endif
#ifdef SIGEMT
      emt = signal(SIGEMT, G__error_handle);
#endif
#ifdef SIGBUS
      bus = signal(SIGBUS, G__error_handle);
#endif
#endif

      store_asm_inst = G__asm_inst;
      store_asm_stack = G__asm_stack;
      store_asm_name = G__asm_name;
      store_asm_name_p = G__asm_name_p;
      store_asm_param  = G__asm_param ;
      /* store_asm_exec  = G__asm_exec ; */
      store_asm_noverflow  = G__asm_noverflow ;
      store_asm_cp  = G__asm_cp ;
      store_asm_dt  = G__asm_dt ;
      store_asm_index  = G__asm_index ;

      G__asm_inst = asm_inst_g;
      G__asm_stack = asm_stack_g;
      G__asm_name = asm_name;
      G__asm_name_p = 0;
      /* G__asm_param ; */
      G__asm_exec = 0 ;

      /* execution */
      int brace_level = 0;
      buf = G__exec_statement(&brace_level);

      G__asm_inst = store_asm_inst;
      G__asm_stack = store_asm_stack;
      G__asm_name = store_asm_name;
      G__asm_name_p = store_asm_name_p;
      G__asm_param  = store_asm_param ;
      /* G__asm_exec  = store_asm_exec ; */
      G__asm_noverflow  = store_asm_noverflow ;
      G__asm_cp  = store_asm_cp ;
      G__asm_dt  = store_asm_dt ;
      G__asm_index  = store_asm_index ;

      /**********************************************
       * restore interrpret signal handling
       **********************************************/
#ifdef G__EH_SIGNAL
      signal(SIGFPE, fpe);
      signal(SIGSEGV, segv);
#ifdef SIGILL
      signal(SIGSEGV, ill);
#endif
#ifdef SIGEMT
      signal(SIGEMT, emt);
#endif
#ifdef SIGBUS
      signal(SIGBUS, bus);
#endif
#endif

#ifdef G__ASM
      G__RECOVER_ASMENV;
#endif
      G__var_type = store_var_type;

      /* print out result */
      G__ifile = *store_ifile;
      if (G__ifile.fp && G__ifile.filenum >= 0) {
         fsetpos(G__ifile.fp, &pos);
      }
      /* Following is intentionally commented out. This has to be selectively
       * done for 'x' and 'E' command  but not for { } command */
      /* G__security = G__srcfile[G__ifile.filenum].security; */
      if (file) fclose(ftemp->fp);
      ++G__tempfilenum;
      G__srcfile[G__tempfilenum].fp = (FILE*)NULL;
      G__srcfile[G__tempfilenum].filename = (char*)NULL;
      if (G__srcfile[G__tempfilenum].breakpoint)
         free(G__srcfile[G__tempfilenum].breakpoint);
      if (G__RETURN_IMMEDIATE >= G__return) G__return = G__RETURN_NON;
      G__no_exec = 0;
      G__UnlockCriticalSection();

      return(buf);
   }
   else {
      G__fprinterr(G__serr, "Error: can not open file '%s'\n", file);
      G__UnlockCriticalSection();
      delete [] filename;
      return(G__null);
   }
}

//______________________________________________________________________________
//
//  External Functions.
//

//______________________________________________________________________________
int G__findposition(const char* string, G__input_file* view, int* pline, int* pfnum)
{
   // -- FIXME: Describe this function!
   //
   // return   0    source line not found
   // return   1    source file exists but line not exact
   // return   2    source line exactly found
   int i = 0;

   /* preset current position */
   *pline = view->line_number;
   *pfnum = view->filenum;

   /* skip space */
   while (isspace(string[i])) i++;

   if ('\0' == string[i]) {
      if ('\0' == view->name[0]) return(0);
      *pline = view->line_number;
      if (view->line_number < 1 || G__srcfile[view->filenum].maxline <= view->line_number)
         return(1);
      else
         return(2);
   }
   else if (isdigit(string[i])) {
      if ('\0' == view->name[0]) return(0);
      *pline = atoi(string + i);
   }
   else {
      return(G__findfuncposition(string + i, pline, pfnum));
   }

   if (*pfnum < 0 || G__nfile <= *pfnum) {
      *pfnum = view->filenum;
      *pline = view->line_number;
      return(0);
   }
   else if (*pline < 1) {
      *pline = 1;
      return(1);
   }
   else if (G__srcfile[*pfnum].maxline < *pline) {
      *pline = G__srcfile[*pfnum].maxline - 1;
      return(1);
   }
   return(2);
}

//______________________________________________________________________________
int G__display_proto(FILE* fp, const char* func)
{
   // -- FIXME: Describe this function!
   return G__display_proto_pretty(fp, func, 0);
}

//______________________________________________________________________________
int G__display_proto_pretty(FILE* fp, const char* func, const char friendlyStyle)
{
   // -- FIXME: Describe this function!
   size_t lenfunc = strlen(func) + 1;
   G__FastAllocString funcname(lenfunc);
   G__FastAllocString scope(lenfunc);
   G__FastAllocString temp(lenfunc);
   char *pc;
   /* int temp1; */
   int tagnum;
   struct G__ifunc_table_internal *ifunc;
   size_t i = 0;

   while (isspace(func[i])) ++i;
   funcname = func + i;

   pc = strstr(funcname, "::");

   /* get appropreate scope */
   if (pc) {
      *pc = '\0';
      scope = funcname;
      temp = pc + 2;
      funcname.Swap(temp);
      if (0 == scope[0]) tagnum = -1;
      else tagnum = G__defined_tagname(scope, 0);
      /* class scope A::func , global scope ::func */
      if (-1 == tagnum) ifunc = &G__ifunc;  /* global scope, ::func */
      else {
         G__incsetup_memfunc(tagnum);
         ifunc = G__struct.memfunc[tagnum]; /* specific class */
      }
   }
   else {
      /* global scope */
      tagnum = -1;
      ifunc = &G__ifunc;
   }

   i = strlen(funcname);
   while (i && (isspace(funcname[i-1]) || '(' == funcname[i-1])) funcname[--i] = '\0';
   if (i) {
      if (G__listfunc_pretty(fp, G__PUBLIC_PROTECTED_PRIVATE, funcname, G__get_ifunc_ref(ifunc), friendlyStyle)) return(1);
   }
   else  {
      if (G__listfunc_pretty(fp, G__PUBLIC_PROTECTED_PRIVATE, (char*)NULL, G__get_ifunc_ref(ifunc), friendlyStyle))return(1);
   }
   if (-1 != tagnum) {
      int i1;
      struct G__inheritance *baseclass = G__struct.baseclass[tagnum];
      for (i1 = 0;i1 < baseclass->basen;i1++) {
         ifunc = G__struct.memfunc[baseclass->herit[i1]->basetagnum];
         if (i) {
            if (G__listfunc_pretty(fp, G__PUBLIC_PROTECTED_PRIVATE, funcname, G__get_ifunc_ref(ifunc), friendlyStyle))
               return(1);
         }
         else  {
            if (G__listfunc_pretty(fp, G__PUBLIC_PROTECTED_PRIVATE, (char*)NULL, G__get_ifunc_ref(ifunc), friendlyStyle))
               return(1);
         }
      }
   }
   return(0);
}

//______________________________________________________________________________
int G__beforelargestep(char* statement, int* piout, int* plargestep)
{
   // -- FIXME: Describe this function!
   G__break = 0;
   G__setdebugcond();
   switch (G__pause()) {
      case 1:
         // -- ignore
         statement[0] = '\0';
         *piout = 0;
         break;
      case 3:
         // -- largestep
         if (
            strcmp(statement, "break") &&
            strcmp(statement, "continue") &&
            strcmp(statement, "return")
         ) {
            // -- Not break, continue, or return.
            *plargestep = 1;
            G__step = 0;
            G__setdebugcond();
         }
         break;
   }
   return G__return;
}

//______________________________________________________________________________
void G__afterlargestep(int* plargestep)
{
   // -- FIXME: Describe this function!
   G__step = 1;
   *plargestep = 0;
   G__setdebugcond();
}

//______________________________________________________________________________
void G__EOFfgetc()
{
   // -- FIXME: Describe this function!
   G__eof_count++;
   if (G__eof_count > 10) {
      G__unexpectedEOF("G__fgetc()");
      if (G__steptrace || G__stepover || G__break || G__breaksignal || G__debug)
         G__pause();
      G__exit(EXIT_FAILURE);
   }
   if (G__dispsource) {
      if ((G__debug || G__break || G__step
          ) &&
            ((G__prerun != 0) || (G__no_exec == 0)) &&
            (G__disp_mask == 0)) {
         G__fprinterr(G__serr, "EOF\n");
      }
      if (G__disp_mask > 0) G__disp_mask-- ;
   }
   if (G__NOLINK == G__globalcomp &&
         NULL == G__srcfile[G__ifile.filenum].breakpoint) {
      G__srcfile[G__ifile.filenum].breakpoint
      = (char*)calloc((size_t)G__ifile.line_number, 1);
      G__srcfile[G__ifile.filenum].maxline = G__ifile.line_number;
   }
}

//______________________________________________________________________________
void G__BREAKfgetc()
{
   // -- FIXME: Describe this function!
   // --
#ifdef G__ASM
   if (G__no_exec_compile) {
      G__abortbytecode();
   }
   else {
      G__break = 1;
      G__setdebugcond();
      if (G__srcfile[G__ifile.filenum].breakpoint) {
         G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number] &= G__NOCONTUNTIL;
      }
   }
#else
   G__break = 1;
   G__setdebugcond();
   G__breakpoint[G__ifile.filenum][G__ifile.line_number] &= G__NOCONTUNTIL;
#endif
   // --
}

//______________________________________________________________________________
void G__DISPNfgetc()
{
   // -- FIXME: Describe this function!
   if ((G__debug || G__break || G__step) && ((G__prerun) || (G__no_exec == 0)) && (G__disp_mask == 0)) {
      G__fprinterr(G__serr, "\n%-5d", G__ifile.line_number);
   }
   if (G__disp_mask > 0) {
      --G__disp_mask;
   }
}

//______________________________________________________________________________
void G__DISPfgetc(int c)
{
   // -- FIXME: Describe this function!
   if ((G__debug || G__break || G__step
       ) &&
         ((G__prerun != 0) || (G__no_exec == 0)) && (G__disp_mask == 0)) {
#ifndef G__OLDIMPLEMENTATION1485
      G__fputerr(c);
#else
      fputc(c, G__serr);
#endif
   }
   if (G__disp_mask > 0) G__disp_mask-- ;
}

//______________________________________________________________________________
int G__lock_variable(const char* varname)
{
   // -- FIXME: Describe this function!
   int hash, ig15;
   struct G__var_array *var;

   if (G__dispmsg >= G__DISPWARN) {
      G__fprinterr(G__serr, "Warning: lock variable obsolete feature");
      G__printlinenum();
   }

   G__hash(varname, hash, ig15)
   var = G__getvarentry(varname, hash, &ig15, &G__global, G__p_local);

   if (var) {
      var->constvar[ig15] |= G__LOCKVAR;
      G__fprinterr(G__serr, "Variable %s locked FILE:%s LINE:%d\n"
                   , varname, G__ifile.name, G__ifile.line_number);
      return(0);
   }
   else {
      G__fprinterr(G__serr, "Warining: failed locking %s FILE:%s LINE:%d\n"
                   , varname, G__ifile.name, G__ifile.line_number);
      return(1);
   }
}

//______________________________________________________________________________
int G__unlock_variable(const char* varname)
{
   // -- FIXME: Describe this function!
   int hash, ig15;
   struct G__var_array *var;

   if (G__dispmsg >= G__DISPWARN) {
      G__fprinterr(G__serr, "Warning: lock variable obsolete feature");
      G__printlinenum();
   }

   G__hash(varname, hash, ig15)
   var = G__getvarentry(varname, hash, &ig15, &G__global, G__p_local);

   if (var) {
      var->constvar[ig15] &= ~G__LOCKVAR;
      G__fprinterr(G__serr, "Variable %s unlocked FILE:%s LINE:%d\n"
                   , varname, G__ifile.name, G__ifile.line_number);
      return(0);
   }
   else {
      G__fprinterr(G__serr, "Warining: failed unlocking %s FILE:%s LINE:%d\n"
                   , varname, G__ifile.name, G__ifile.line_number);
      return(1);
   }
}

//______________________________________________________________________________
G__value G__interactivereturn()
{
   // -- FIXME: Describe this function!
   G__value result;
   result = G__null;
   if (G__interactive) {
      G__interactive = 0;
      fprintf(G__sout, "!!!Return arbitrary value by 'return [value]' command");
      G__interactive_undefined = 1;
      G__pause();
      G__interactive_undefined = 0;
      G__interactive = 1;
      result = G__interactivereturnvalue;
   }
   G__interactivereturnvalue = G__null;
   return(result);
}

//______________________________________________________________________________
void G__set_tracemode(char* name)
{
   // -- FIXME: Describe this function!
   int tagnum;
   int i = 0;
   char *p, *s;
   while (name[i] && isspace(name[i])) i++;
   if ('\0' == name[i]) {
      fprintf(G__sout, "trace all source code\n");
      G__istrace = 1;
      tagnum = -1;
   }
   else {
      s = name + i;
      while (s) {
         p = strchr(s, ' ');
         if (p) *p = '\0';
         tagnum = G__defined_tagname(s, 0);
         if (-1 != tagnum) {
            G__struct.istrace[tagnum] = 1;
            fprintf(G__sout, "trace %s object on\n", s);
         }
         if (p) s = p + 1;
         else  s = p;
      }
   }
   G__setclassdebugcond(G__memberfunc_tagnum, 0);
}

//______________________________________________________________________________
void G__del_tracemode(char* name)
{
   // -- FIXME: Describe this function!
   int tagnum;
   int i = 0;
   char *p, *s;
   while (name[i] && isspace(name[i])) i++;
   if ('\0' == name[i]) {
      G__istrace = 0;
      tagnum = -1;
      fprintf(G__sout, "trace all source code off\n");
   }
   else {
      s = name + i;
      while (s) {
         p = strchr(s, ' ');
         if (p) *p = '\0';
         tagnum = G__defined_tagname(s, 0);
         if (-1 != tagnum) {
            G__struct.istrace[tagnum] = 0;
            fprintf(G__sout, "trace %s object off\n", s);
         }
         if (p) s = p + 1;
         else  s = p;
      }
   }
   G__setclassdebugcond(G__memberfunc_tagnum, 0);
}

//______________________________________________________________________________
void G__set_classbreak(char* name)
{
   // -- FIXME: Describe this function!
   int tagnum;
   int i = 0;
   char *p, *s;
   while (name[i] && isspace(name[i])) i++;
   if (name[i]) {
      s = name + i;
      while (s) {
         p = strchr(s, ' ');
         if (p) *p = '\0';
         tagnum = G__defined_tagname(s, 0);
         if (-1 != tagnum) {
            G__struct.isbreak[tagnum] = 1;
            fprintf(G__sout, "set break point at every %s member function\n", s);
         }
         if (p) s = p + 1;
         else  s = p;
      }
   }
}

//______________________________________________________________________________
void G__del_classbreak(char* name)
{
   // -- FIXME: Describe this function!
   int tagnum;
   int i = 0;
   char *p, *s;
   while (name[i] && isspace(name[i])) i++;
   if (name[i]) {
      s = name + i;
      while (s) {
         p = strchr(s, ' ');
         if (p) *p = '\0';
         tagnum = G__defined_tagname(s, 0);
         if (-1 != tagnum) {
            G__struct.isbreak[tagnum] = 0;
            fprintf(G__sout, "delete break point at every %s member function\n", s);
         }
         if (p) s = p + 1;
         else  s = p;
      }
   }
}

//______________________________________________________________________________
void G__setclassdebugcond(int tagnum, int brkflag)
{
   // -- FIXME: Describe this function!
   if (G__cintv6) return;
   if (-1 == tagnum) {
      G__debug = G__istrace;
   }
   else {
      G__debug = G__struct.istrace[tagnum] | G__istrace;
      G__break |= G__struct.isbreak[tagnum];
   }
   G__dispsource = G__step + G__break + G__debug;
   if (G__dispsource == 0) G__disp_mask = 0;
   if (brkflag) {
      if ((G__break || G__step) && 0 == G__prerun) G__breaksignal = 1;
      else                                  G__breaksignal = 0;
   }
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
void G__setdebugcond()
{
   // -- Preset trace/debug condition for speed up.
   G__dispsource = G__break + G__step + G__debug;
   if (!G__dispsource) {
      G__disp_mask = 0;
   }
   G__breaksignal = 0;
   if ((G__break || G__step) && !G__prerun) {
      G__breaksignal = 1;
   }
}

//______________________________________________________________________________
int G__gettempfilenum()
{
   // -- FIXME: Describe this function!
   return G__tempfilenum;
}

//______________________________________________________________________________
G__value G__exec_tempfile_fp(FILE* fp)
{
   // -- FIXME: Describe this function!
   return G__exec_tempfile_core(0, fp);
}

//______________________________________________________________________________
G__value G__exec_tempfile(const char* file)
{
   // -- FIXME: Describe this function!
   return G__exec_tempfile_core(file, 0);
}

//______________________________________________________________________________
G__value G__exec_text(const char* unnamedmacro)
{
   // -- FIXME: Describe this function!
#ifndef G__TMPFILE
   G__FastAllocString tname_sb(L_tmpnam+10);
   G__FastAllocString sname_sb(L_tmpnam+10);
#else
   G__FastAllocString tname_sb(G__MAXFILENAME);
   G__FastAllocString sname_sb(G__MAXFILENAME);
#endif
   char* tname = tname_sb;
   int nest = 0, single_quote = 0, double_quote = 0;
   int ccomment = 0, cppcomment = 0;
   G__value buf;
   FILE *fp;
   size_t i, len;
   int addmparen = 0;
   int addsemicolumn = 0;
   int istmpnam = 0;

   i = 0;
   while (unnamedmacro[i] && isspace(unnamedmacro[i])) ++i;
   if (unnamedmacro[i] != '{') addmparen = 1;

   i = strlen(unnamedmacro) - 1;
   while (i && isspace(unnamedmacro[i])) --i;
   if (unnamedmacro[i] == '}')       addsemicolumn = 0;
   else if (unnamedmacro[i] == ';')  addsemicolumn = 0;
   else                   addsemicolumn = 1;

   len = (int)strlen(unnamedmacro);
   for (i = 0;i < len;i++) {
      switch (unnamedmacro[i]) {
         case '(':
         case '[':
         case '{':
            if (!single_quote && !double_quote && !ccomment && !cppcomment) ++nest;
            break;
         case ')':
         case ']':
         case '}':
            if (!single_quote && !double_quote && !ccomment && !cppcomment) --nest;
            break;
         case '\'':
            if (!double_quote && !ccomment && !cppcomment) single_quote ^= 1;
            break;
         case '"':
            if (!single_quote && !ccomment && !cppcomment) double_quote ^= 1;
            break;
         case '/':
            switch (unnamedmacro[i+1]) {
               case '/':
                  cppcomment = 1;
                  ++i;
                  break;
               case '*':
                  ccomment = 1;
                  ++i;
                  break;
               default:
                  break;
            }
            break;
         case '\n':
         case '\r':
            if (cppcomment) {
               cppcomment = 0;
               ++i;
            }
            break;
         case '*':
            if (ccomment && unnamedmacro[i+1] == '/') {
               ccomment = 0;
               ++i;
            }
            break;
         case '\\':
            ++i;
            break;
         default:
            break;
      }
   }
   if (nest != 0 || single_quote != 0 || double_quote != 0) {
      G__fprinterr(G__serr, "!!!Error in given statement!!! \"%s\"\n", unnamedmacro);
      return(G__null);
   }

   // coverity[secure_temp]: we don't care about predictable names.
   fp = tmpfile();
   if (!fp) {
      G__tmpnam(tname);  /* not used anymore 0 */
      fp = fopen(tname, "w");
      istmpnam = 1;
   }
   if (!fp) return G__null;
   if (addmparen) fprintf(fp, "{\n");
   fprintf(fp, "%s", unnamedmacro);
   if (addsemicolumn) fprintf(fp, ";");
   fprintf(fp, "\n");
   if (addmparen) fprintf(fp, "}\n");
   if (!istmpnam) fseek(fp, 0L, SEEK_SET);
   else          fclose(fp);

   if (!istmpnam) {
      G__storerewindposition();
      buf = G__exec_tempfile_fp(fp);
      G__security_recover(G__serr);
      fclose(fp);
   }
   else {
      sname_sb = tname;
      const char *sname = sname_sb;
      G__storerewindposition();
      buf = G__exec_tempfile(sname);
      G__security_recover(G__serr);
      remove(sname);
   }

   return(buf);
}

#ifndef G__OLDIMPLEMENTATION1867
//______________________________________________________________________________
char* G__exec_text_str(const char* unnamedmacro, char* result)
{
   // -- FIXME: Describe this function!
   G__FastAllocString resbuf;
   G__value buf = G__exec_text(unnamedmacro);
   G__valuemonitor(buf, resbuf);
   strcpy(result, resbuf); // Legacy interface, we don't know the buffer size
   return result;
}
#endif

#ifndef G__OLDIMPLEMENTATION1546
//______________________________________________________________________________
const char* G__load_text(const char* namedmacro)
{
   // -- FIXME: Describe this function!
   int fentry;
   char* result = (char*)NULL;
   FILE *fp;
   int istmpnam = 0;
#ifndef G__TMPFILE
   static char tname[L_tmpnam+10];
#else
   static char tname[G__MAXFILENAME];
#endif

   // coverity[secure_temp]: we don't care about predictable names.
   fp = tmpfile();
   if (!fp) {
      G__tmpnam(tname);  /* not used anymore */
      strncat(tname, G__NAMEDMACROEXT, sizeof(tname) - strlen(tname) - 1);
      tname[sizeof(tname) - 1] = 0; // ensure termination
      fp = fopen(tname, "w");
      if (!fp) return((char*)NULL);
      istmpnam = 1;
   }
   fprintf(fp, "%s", namedmacro);
   fprintf(fp, "\n");

   if (!istmpnam) {
      fseek(fp, 0L, SEEK_SET);
      fentry = G__loadfile_tmpfile(fp);
   }
   else {
      fclose(fp);
      fentry = G__loadfile(tname);
   }

   switch (fentry) {
      case G__LOADFILE_SUCCESS:
         if (!istmpnam) {
            strncpy(tname,"(tmpfile)", sizeof(tname) - 1);
            tname[sizeof(tname) - 1] = 0; // ensure termination
         }
         result = tname;
         break;
      case G__LOADFILE_DUPLICATE:
      case G__LOADFILE_FAILURE:
      case G__LOADFILE_FATAL:
         if (!istmpnam) fclose(fp);
         else           remove(tname);
         result = (char*)NULL;
         break;
      default:
         result = G__srcfile[fentry-2].filename;
         break;
   }
   return(result);
}
#endif

//______________________________________________________________________________
int G__setbreakpoint(const char* breakline, const char* breakfile)
{
   // -- FIXME: Describe this function!
   int ii;
   int line;

   if (isdigit(breakline[0])) {
      line = atoi(breakline);

      if (NULL == breakfile || '\0' == breakfile[0]) {
         G__fprinterr(G__serr, " -b : break point on line %d every file\n", line);
         for (ii = 0;ii < G__nfile;ii++) {
            if (G__srcfile[ii].breakpoint && G__srcfile[ii].maxline > line)
               G__srcfile[ii].breakpoint[line] |= G__BREAK;
         }
      }
      else {
         for (ii = 0;ii < G__nfile;ii++) {
            if (G__srcfile[ii].filename &&
                  G__matchfilename(ii, breakfile)
               ) break;
         }
         if (ii < G__nfile) {
            G__fprinterr(G__serr, " -b : break point on line %d file %s\n"
                         , line, breakfile);
            if (G__srcfile[ii].breakpoint && G__srcfile[ii].maxline > line)
               G__srcfile[ii].breakpoint[line] |= G__BREAK;
         }
         else {
            G__fprinterr(G__serr, "File %s is not loaded\n", breakfile);
            return(1);
         }
      }

   }
   else {
      if (1 < G__findfuncposition(breakline, &line, &ii)) {
         if (G__srcfile[ii].breakpoint) {
            G__fprinterr(G__serr, " -b : break point on line %d file %s\n"
                         , line, G__srcfile[ii].filename);
            G__srcfile[ii].breakpoint[line] |= G__BREAK;
         }
         else {
            G__fprinterr(G__serr, "unable to put breakpoint in %s (included file)\n"
                         , breakline);
         }
      }
      else {
         G__fprinterr(G__serr, "function %s is not loaded\n", breakline);
         return(1);
      }
   }
   return(0);
}

} // extern "C"

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
