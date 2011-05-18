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
#include <cstdlib>
#include <stack>
#include <vector>
#include <string>

using namespace std;

#if 0
class G__breakcontinue {
private:
   bool isbreak;
   int pc;
public:
   G__breakcontinue() : isbreak(false), pc(-1) {}
   G__breakcontinue(const G__breakcontinue& rhs) : isbreak(rhs.isbreak), pc(rhs.pc) {}
   G__breakcontinue& operator=(const G__breakcontinue& rhs) : isbreak(rhs.isbreak), pc(rhs.pc) {}
   ~G__breakcontinue() {}
   bool isbreak() { return isbreak; }
   void isbreak(bool flag) { isbreak = flag; }
   int pc() { return pc; }
   void pc(int val) { pc = val; }
   void setdest
};
#endif

static int G__setline(G__FastAllocString& statement, int c, int* piout);
static int G__pp_ifdefextern(G__FastAllocString& temp);
static int G__exec_try(G__FastAllocString& statement);
static int G__exec_throw(G__FastAllocString& statement);
static int G__exec_function(G__FastAllocString& statement, int* pc, int* piout, int* plargestep, G__value* presult);
static G__value G__exec_switch_case(G__FastAllocString& casepara);
static G__value G__exec_loop(const char* forinit, char* condition, const std::list<G__FastAllocString>& foraction = std::list<G__FastAllocString>());
static int G__defined_type(G__FastAllocString& type_name, int len);
static int G__keyword_anytime_5(G__FastAllocString& statement);
static int G__keyword_anytime_6(G__FastAllocString& statement);
static int G__keyword_anytime_7(G__FastAllocString& statement);
static int G__keyword_anytime_8(G__FastAllocString& statement);

extern "C" {

//______________________________________________________________________________
//
//  External functions.  (FIXME: These should be in fproto.h.)
//

extern int G__const_setnoerror(); // v6_error.cxx, in the C interface
extern int G__const_resetnoerror(); // v6_error.cxx, in the C interface
extern void G__CMP2_equal(G__value*, G__value*); // v6_pcode.cxx

//______________________________________________________________________________
//
//  Function table.
//

// statics
#ifdef G__WIN32
static void G__toUniquePath(char* s);
#endif // G__WIN32
static void G__pp_undef();
static int G__ignore_catch();
static struct G__breakcontinue_list* G__alloc_breakcontinue_list();
static void G__store_breakcontinue_list(int destination, int breakcontinue);
static void G__free_breakcontinue_list(G__breakcontinue_list* pbreakcontinue);
static void G__set_breakcontinue_destination(int break_dest, int continue_dest, G__breakcontinue_list* pbreakcontinue);
static G__value G__exec_switch();
static G__value G__exec_if();
static G__value G__exec_else_if();
static G__value G__exec_do();
static G__value G__exec_for();
static G__value G__exec_while();
static G__value G__return_value(const char* statement);
static int G__search_gotolabel(char* label, fpos_t* pfpos, int line, int* pmparen);
static int G__label_access_scope(char* statement, int* piout, int* pspaceflag, int mparen);
static int G__IsFundamentalDecl();
static void G__unsignedintegral();
static void G__externignore();
static void G__parse_friend();

// externally visible
G__value G__alloc_exceptionbuffer(int tagnum);
int G__free_exceptionbuffer();
void G__display_tempobject(const char* action);
int G__defined_macro(const char* macro);
int G__pp_command();
void G__pp_skip(int elifskip);
int G__pp_if();
int G__pp_ifdef(int def);
int G__exec_catch(char* statement);
int G__skip_comment();
int G__skip_comment_peek();
G__value G__exec_statement(int* mparen);

// in the C interface
void G__alloc_tempobject(int tagnum, int typenum);
void G__free_tempobject();
void G__store_tempobject(G__value reg);
static int G__pop_tempobject_imp(bool delobj);
int G__pop_tempobject();
int G__pop_tempobject_nodel();
void G__settemplevel(int val);
void G__clearstack();

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

//______________________________________________________________________________
static int G__prevcase = 0; // Communication between G__exec_switch() and G__exec_statement()

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Preprocessor commands.
//

#ifdef G__WIN32
//______________________________________________________________________________
static void G__toUniquePath(char* s)
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
   strcpy(s, d); // Okay we allocated enough space
   free(d);
}
#endif // G__WIN32

} // extern "C"

//______________________________________________________________________________
static int G__setline(G__FastAllocString& statement, int c, int* piout)
{
   // -- FIXME: Describe this function!
   if ((c != '\n') && (c != '\r')) {
      c = G__fgetname(statement, 1, "\n\r");
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
            c = G__fgetname(statement, 0, "\n\r");
            if (statement[0] == '"') {
               // -- We have #<line> "<filename>".
               G__getcintsysdir();
               G__FastAllocString sysinclude(G__MAXFILENAME);
               sysinclude.Format("%s/%s/include/", G__cintsysdir, G__CFG_COREVERSION);
               G__FastAllocString sysstl(G__MAXFILENAME);
               sysstl.Format("%s/%s/stl/", G__cintsysdir, G__CFG_COREVERSION);
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
               G__strlcpy(G__ifile.name, statement + 1, G__MAXFILENAME);
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
                  strcpy(G__srcfile[null_entry].filename, statement + 1); // Okay we allocated enough space
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
                     strcpy(G__srcfile[null_entry].prepname, G__srcfile[G__ifile.filenum].prepname); // Okay we allocated enough space
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
                     strcpy(G__srcfile[G__nfile].filename, statement + 1); // Okay we allocated enough space
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
                        strcpy(G__srcfile[G__nfile].prepname, G__srcfile[G__ifile.filenum].prepname); // Okay we allocated enough space
                        G__srcfile[G__nfile].fp = G__ifile.fp;
                     }
                     G__srcfile[G__nfile].included_from = G__ifile.filenum;
                     G__srcfile[G__nfile].ispermanentsl = 0;
                     G__srcfile[G__nfile].initsl = 0;
                     G__srcfile[G__nfile].hasonlyfunc = 0;
                     G__ifile.filenum = G__nfile;
                     ++G__nfile;
                     ++G__srcfile_serial;
                  }
               }
            }
            if ((c != '\n') && (c != '\r')) {
               while (((c = G__fgetc()) != '\n') && (c != '\r')) {};
            }
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
static int G__pp_ifdefextern(G__FastAllocString& temp)
{
   // -- FIXME: Describe this function!
   fpos_t pos;
   fgetpos(G__ifile.fp, &pos);
   int linenum = G__ifile.line_number;
   int cin = G__fgetname(temp, 0, "\"}#");
   if (cin == '}') {
      // -- 
      //
      //   #ifdef __cplusplus
      //   {}
      //   #endif
      //
      G__fignoreline();
      do {
         cin = G__fgetstream(temp, 0, "#");
         cin = G__fgetstream(temp, 0, "\n\r");
      }
      while (strcmp(temp, "endif"));
      return G__IFDEF_ENDBLOCK;
   }
   if ((cin != '#') && !strcmp(temp, "extern")) {
      //
      //   #ifdef __cplusplus
      //   extern "C" { ... }
      //   #endif
      //
      //
      //   #ifdef __cplusplus
      //   extern "C" {  ...  }
      //   #endif
      //
      G__var_type = 'p';
      if (cin != '{') {
         cin = G__fgetspace();
      }
      if (cin == '"') {
         // -- extern "C" {...}
         int flag = 0;
         int store_iscpp = G__iscpp;
         int store_externblock_iscpp = G__externblock_iscpp;
         G__FastAllocString fname(G__MAXFILENAME);
         cin = G__fgetstream(fname, 0, "\"");
         temp[0] = 0;
         do {
            cin = G__fgetstream(temp, 0, "{\r\n");
         }
         while (!temp[0] && ((cin == '\r') || (cin == '\n')));
         if (temp[0] || (cin != '{')) {
            goto goback;
         }
         cin = G__fgetstream(temp, 0, "\n\r");
         if ((cin == '}') && !strcmp(fname, "C")) {
            goto goback;
         }
         cin = G__fgetstream(temp, 0, "#\n\r");
         if (((cin == '\n') || (cin == '\r')) && !temp[0]) {
            cin = G__fgetstream(temp, 0, "#\n\r");
         }
         if (cin != '#') {
            goto goback;
         }
         cin = G__fgetstream(temp, 0, "\n\r");
         if (((cin == '\n') || (cin == '\r')) && !temp[0]) {
            cin = G__fgetstream(temp, 0, "#\n\r");
         }
         if (strcmp(temp, "endif")) {
            goto goback;
         }
         if (!strcmp(fname, "C")) {
            G__externblock_iscpp = (G__iscpp || G__externblock_iscpp);
            G__iscpp = 0;
         }
         else {
            G__loadfile(fname);
            G__SetShlHandle(fname);
            flag = 1;
         }
         int brace_level = 1;
         G__exec_statement(&brace_level);
         G__iscpp = store_iscpp;
         G__externblock_iscpp = store_externblock_iscpp;
         if (flag) {
            G__ResetShlHandle();
         }
         return G__IFDEF_EXTERNBLOCK;
      }
   }
   goback:
   fsetpos(G__ifile.fp, &pos);
   G__ifile.line_number = linenum;
   return G__IFDEF_NORMAL;
}

extern "C" {

//______________________________________________________________________________
static void G__pp_undef()
{
   // -- FIXME: Describe this function!
   G__FastAllocString temp(G__MAXNAME);
   G__fgetname(temp, 0, "\n\r");
   struct G__var_array* var = &G__global;
   while (var) {
      for (int i = 0; i < var->allvar; ++i) {
         if (
            // --
            var->varnamebuf[i] &&
            (temp[0] == var->varnamebuf[i][0]) &&
            !strcmp(temp, var->varnamebuf[i]) &&
            (var->type[i] == 'p')
         ) {
            var->hash[i] = 0;
            var->varnamebuf[i][0] = '\0';
         }
      }
      var = var->next;
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

} // extern "C"

//______________________________________________________________________________
static int G__exec_try(G__FastAllocString& statement)
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


extern "C" {

//______________________________________________________________________________
static int G__ignore_catch()
{
   // -- FIXME: Describe this function!
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
      fpos_t fpos1;
      fseek(G__ifile.fp, -1, SEEK_CUR);
      fseek(G__ifile.fp, -1, SEEK_CUR);
      while (fgetc(G__ifile.fp) != 'a') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         fseek(G__ifile.fp, -1, SEEK_CUR);
      }
      while (fgetc(G__ifile.fp) != 'c') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         fseek(G__ifile.fp, -1, SEEK_CUR);
      }
      fseek(G__ifile.fp, -1, SEEK_CUR);
      fgetpos(G__ifile.fp, &fpos1);
#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x: CATCH\n", G__asm_cp);
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__CATCH;
      G__asm_inst[G__asm_cp+1] = G__ifile.filenum;
      G__asm_inst[G__asm_cp+2] = G__ifile.line_number;
#if defined(G__NONSCALARFPOS2)
      G__asm_inst[G__asm_cp+3] = (long) fpos1.__pos;
#elif defined(G__NONSCALARFPOS_QNX)
      G__asm_inst[G__asm_cp+3] = (long) fpos1._Off;
#else // defined(G__NONSCALARFPOS_QNX)
      G__asm_inst[G__asm_cp+3] = (long) fpos1;
#endif // defined(G__NONSCALARFPOS_QNX)
      G__inc_cp_asm(5, 0);
      G__fignorestream("(");
   }
   // Ignore the exception clause.
   G__fignorestream(")");
   // And skip the rest of the catch clause.
   G__no_exec = 1;
   int brace_level = 0;
   G__exec_statement(&brace_level);
   G__no_exec = 0;
   return 0;
}

} // extern "C"


//______________________________________________________________________________
static int G__exec_throw(G__FastAllocString& statement)
{
   // -- Handle the "throw" expression.
   int iout = 0;
   G__FastAllocString buf(G__ONELINE);
   G__fgetstream(buf, 0, ";");
   if (isdigit(buf[0]) || (buf[0] == '.')) {
      statement = buf;
      iout = 5;
   }
   else {
      statement = "new ";
      statement += buf;
      iout = strlen(statement);
   }
   G__exceptionbuffer = G__null;
   if (iout > 4) {
      int largestep = 0;
      if (G__breaksignal) {
         int stat = G__beforelargestep(statement, &iout, &largestep);
         if (stat > 0) {
            return 1;
         }
      }
      //
      //  Evaluate the throw expression.
      //
      G__exceptionbuffer = G__getexpr(statement);
      if (G__asm_noverflow) {
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x: THROW\n", G__asm_cp);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__THROW;
         G__inc_cp_asm(1, 0);
      }
      if (largestep) {
         G__afterlargestep(&largestep);
      }
      //
      //  Change the thrown value to be by reference
      //  instead of by pointer.
      //
      G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
      if (isupper(G__exceptionbuffer.type)) {
         G__exceptionbuffer.type += 'u' - 'U';
#define G__DEREF_EXC(TYPE, MEM)                                          \
         G__exceptionbuffer.obj.MEM = *(TYPE*)G__exceptionbuffer.ref; break

         switch (G__exceptionbuffer.type) {
         case 'u': break;
         case 'd': G__DEREF_EXC(double,d);
         case 'i':
         case 'l': G__DEREF_EXC(long, i);
         case 'c': G__DEREF_EXC(char, ch);
         case 's': G__DEREF_EXC(short, sh);
         case 'f': G__DEREF_EXC(float, fl);
         case 'b': G__DEREF_EXC(unsigned char, uch);
         case 'r': G__DEREF_EXC(unsigned short, ush);
         case 'h': G__DEREF_EXC(unsigned int, uin);
         case 'k': G__DEREF_EXC(unsigned long, ulo);
         case 'n': G__DEREF_EXC(G__int64, ll);
         case 'm': G__DEREF_EXC(G__uint64, ull);
         case 'q': G__DEREF_EXC(long double, ld);
#ifdef G__BOOL4BYTE
         case 'g': G__exceptionbuffer.type = 'i'; G__DEREF_EXC(int, i);
#else // G__BOOL4BYTE
         case 'g': G__exceptionbuffer.type = 'i'; G__DEREF_EXC(unsigned char, i);
#endif // G__BOOL4BYTE
         default: G__DEREF_EXC(long, i);
#undef G__DEREF_EXC
         }
         
      }
   }
   if (!G__no_exec_compile) {
      // Flag that we should stop executing.
      G__no_exec = 1;
      // Flag that we need to go to the catch clauses
      // at the bottom of the nearest enclosing try block.
      G__return = G__RETURN_TRY;
   }
   return 0;
}

//______________________________________________________________________________
//
//  Expressions.   Function call.
//

//______________________________________________________________________________
static int G__exec_function(G__FastAllocString& statement, int* pc, int* piout, int* plargestep, G__value* presult)
{
   // -- Function call.
   //
   // Return 0 if function called, return 1 if function is *not* called.
   //
   if ((*pc == ';') || G__isoperator(*pc) || (*pc == ',') || (*pc == '.') || (*pc == '[')) {
      //fprintf(stderr, "G__exec_function: Function call is followed by an operator.\n");
      if ((*pc != ';') && (*pc != ',')) {
         statement[(*piout)++] = *pc;
         *pc = G__fgetstream_new(statement ,  (*piout), ";");
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
      *pc = G__fgetstream_newtemplate(statement, len, ")");
      len = strlen(statement);
      statement[len++] = *pc;
      statement[len] = 0;
      *pc = G__fgetspace();
      while (*pc != ';') {
         len = strlen(statement);
         statement[len++] = *pc;
         *pc = G__fgetstream_newtemplate(statement, len, ");");
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
            G__fprinterr(G__serr, "Warning: %s Missing ';'", statement());
            G__printlinenum();
         }
      }
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
   }
   if (*plargestep) {
      G__afterlargestep(plargestep);
   }
   return 0;
}

extern "C" {

//______________________________________________________________________________
//______________________________________________________________________________
//
//  The breakcontine_list structure.  (Used to make a break/continue destination stack.)
//

//______________________________________________________________________________
#ifdef G__ASM
static struct G__breakcontinue_list* G__alloc_breakcontinue_list()
{
   // -- FIXME: Describe this function!
   struct G__breakcontinue_list* oldlist = G__pbreakcontinue;
   G__pbreakcontinue = 0;
   return oldlist;
}
#endif // G__ASM

//______________________________________________________________________________
#ifdef G__ASM
static void G__store_breakcontinue_list(int idx, int isbreak)
{
   // -- FIXME: Describe this function!
   struct G__breakcontinue_list* p = (struct G__breakcontinue_list*) malloc(sizeof(struct G__breakcontinue_list));
   p->next = G__pbreakcontinue;
   p->isbreak = isbreak;
   p->idx = idx;
   G__pbreakcontinue = p;
}
#endif // G__ASM

//______________________________________________________________________________
#ifdef G__ASM
static void G__free_breakcontinue_list(G__breakcontinue_list* oldlist)
{
   // -- FIXME: Describe this function!
   while (G__pbreakcontinue) {
      struct G__breakcontinue_list* p = G__pbreakcontinue->next;
      free(G__pbreakcontinue);
      G__pbreakcontinue = p;
   }
   G__pbreakcontinue = oldlist;
}
#endif // G__ASM

//______________________________________________________________________________
#ifdef G__ASM
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
#endif // G__ASM

//______________________________________________________________________________
#ifdef G__ASM
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
#endif // G__ASM

//______________________________________________________________________________
//______________________________________________________________________________
//
//  More statements.
//

//______________________________________________________________________________
//
//  Selection statements.  if, switch
//

//______________________________________________________________________________
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
   G__FastAllocString condition(G__ONELINE);
   G__fgetstream(condition, 0, ")");
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
         (result.type != G__null.type) && // not the end of the switch block, and
         (result.type != G__default.type) && // not the default case, and
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
      if (result.type != G__null.type) {
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
         int store_G__switch = G__switch;
         G__switch = 1;
         // Tell the parser to go until the end of the switch block.
         int brace_level = 1;
         // Call the parser.
         //fprintf(stderr, "G__exec_switch: Running the case block.\n");
         //fprintf(stderr, "G__exec_switch: just before running case block: G__asm_noverflow: %d G__no_exec_compile: %d\n", G__asm_noverflow, G__no_exec_compile);
         result = G__exec_statement(&brace_level);
         //fprintf(stderr, "G__exec_switch: Case block parse has returned.\n");
         // Restore state.
         G__switch = store_G__switch;
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
         if (result.type == G__block_break.type) {
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
      if ((result.type == G__block_break.type) && (result.obj.i == G__BLOCK_BREAK)) {
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
      if ((result.type == G__block_break.type) && (result.obj.i == G__BLOCK_CONTINUE)) {
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

} //extern "C"

//______________________________________________________________________________
static G__value G__exec_switch_case(G__FastAllocString& casepara)
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

extern "C" {

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
   G__FastAllocString condition(G__LONGLINE);
   G__fgetstream_new(condition, 0, ")");
   condition.Resize(strlen(condition) + 10);
   //
   //  If the previous read hit a breakpoint, pause, and exit if requested.
   //
   if (G__breaksignal) {
      if (G__beforelargestep(condition, &iout, &largestep) > 1) {
         G__ifswitch = store_ifswitch;
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
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_if: Before condition evaluation, increment G__templevel %d --> %d  %s:%d\n"
         , G__templevel
         , G__templevel + 1
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   ++G__templevel;
   long condval = G__test(condition);
   if (largestep) {
      G__afterlargestep(&largestep);
   }
   //
   //  Destroy all temporaries created during expression evaluation.
   //
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_if: Destroy temp objects after condition evaluation, currently at G__templevel %d  %s:%d\n"
         , G__templevel
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   G__free_tempobject();
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_if: After condition evaluation, decrement G__templevel %d --> %d  %s:%d\n"
         , G__templevel
         , G__templevel - 1
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   --G__templevel;
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
         //fprintf(stderr, "---end if\n");
         return result;
      }
      //
      //  Check if the then clause terminated early due
      //  to a break, continue, or goto statement.
      //
      if (result.type == G__block_break.type) {
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
            if (system("key .cint_key -l execute")) {
               G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
            }
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
         int c = G__fgetspace_peek();
         int isblock = 0;
         if (c == '{') {
            isblock = 1;
         }
         //fprintf(stderr, "G__exec_if: isblock: %d\n", isblock);
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
         if (result.type == G__block_break.type) {
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
                  if (isblock) {
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
                  if (isblock) {
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
   // And return.
   //{
   //   char buf[128];
   //   G__fgetstream_peek(buf, 10);
   //   fprintf(stderr, "G__exec_if: peek ahead: '%s'\n", buf);
   //}
   //fprintf(stderr, "---end if ne: %d nec: %d ty: '%c'\n", G__no_exec, G__no_exec_compile, result.type);
   return result;
}

//______________________________________________________________________________
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
   if (!G__no_exec_compile) {
      if (!G__xrefflag) {
         G__asm_noverflow = 0;
      }
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
         if (G__key != 0) {
            if (system("key .cint_key -l execute")) {
               G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
            }
         }
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
      int brace_level = 0;
      result = G__exec_statement(&brace_level);
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
   int brace_level = 0;
   // Call the parser.
   G__value result = G__exec_statement(&brace_level);
   //fprintf(stderr, "G__exec_do: Just finished body.  result.type: %d isbreak: %d\n", result.type, result.type == G__block_break.type);
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
   if (result.type == G__block_break.type) {
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
            G__exec_statement(&brace_level);
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
               G__exec_statement(&brace_level);
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
               G__exec_statement(&brace_level);
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
   G__FastAllocString condition(G__ONELINE);
   G__fgetstream(condition, 0, "(");
   if (strcmp(condition, "while")) {
      G__fprinterr(G__serr, "Syntax error: do {} %s(); Should be do {} while (); FILE: %s LINE: %d\n", condition(), G__ifile.name, G__ifile.line_number);
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
   G__fgetstream(condition, 0, ")");
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
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_do: Before condition evaluation, increment G__templevel %d --> %d  %s:%d\n"
         , G__templevel
         , G__templevel + 1
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   ++G__templevel;
   //fprintf(stderr, "G__exec_do: Testing condition '%s' nec: %d\n", condition, G__no_exec_compile);
   long cond = G__test(condition);
   //fprintf(stderr, "G__exec_do: Testing condition result: %d\n", cond);
   //
   //  Destroy all temporaries created during expression evaluation.
   //
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_do: Destroy temp objects after condition evaluation, currently at G__templevel %d  %s:%d\n"
         , G__templevel
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   G__free_tempobject();
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_do: After condition evaluation, decrement G__templevel %d --> %d  %s:%d\n"
         , G__templevel
         , G__templevel - 1
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   --G__templevel;
   //
   //  If we have already terminated with a break,
   //  restore state.
   //
   if (executed_break) {
      G__no_exec_compile = store_no_exec_compile;
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
            if (result.type == G__block_break.type) {
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

//______________________________________________________________________________
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
   G__FastAllocString condition(G__LONGLINE);
   int c = G__fgetstream(condition, 0, ";)");
   if (c == ')') {
      // -- Syntax error, the third clause of the for was missing.
      G__genericerror("Error: for statement syntax error");
      G__ifswitch = store_ifswitch;
      return G__null;
   }
   // If there is no condition text, make it always true, as the standard requires.
   if (!condition[0]) {
      condition = "1";
   }
   // FIXME: Why do we make the condition buffer bigger?
   condition.Resize(strlen(condition) + 10);
   //
   //  Collect the third clause of the for,
   //  separating it on commas.
   //
   std::list<G__FastAllocString> foraction;
   do {
      // -- Collect one clause of a comma operator expression.
      // Scan until a comma or the end of the head of the for.
      foraction.push_back(G__FastAllocString());
      c = G__fgetstream(foraction.back(), 0, "),");
      if (G__return > G__RETURN_NORMAL) {
         // FIXME: Is this error message correct?
         G__fprinterr(G__serr, "Error: for statement syntax error. ';' needed\n");
         // Restore the ifswitch state.
         G__ifswitch = store_ifswitch;
         // Cleanup malloc'ed memory.
         // And exit in error.
         return G__null;
      }
   }
   while (c != ')');
   //
   //  Execute the body of the loop.
   //
   G__value result  = G__exec_loop(0, condition, foraction);
   // Cleanup malloc'ed memory.
   // Restore the ifswitch state.
   G__ifswitch = store_ifswitch;
   // And we are done.
   return result;
}

//______________________________________________________________________________
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
   G__FastAllocString condition(G__LONGLINE);
   G__fgetstream(condition, 0, ")");
   // FIXME: Why do we make the condition buffer bigger?
   condition.Resize(strlen(condition) + 10);
   //
   //  Execute the body of the loop.
   //
   G__value result = G__exec_loop(0, condition);
   // Cleanup malloc'ed memory.
   // Restore the ifswitch state.
   G__ifswitch = store_ifswitch;
   // And we are done.
   return result;
}

} // extern "C"

//______________________________________________________________________________
static G__value G__exec_loop(const char* forinit, char* condition,
                             const std::list<G__FastAllocString>& foraction)
{
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
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_loop: Before first condition evaluation, increment G__templevel %d --> %d  %s:%d\n"
         , G__templevel
         , G__templevel + 1
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   ++G__templevel;
   //fprintf(stderr, "G__exec_loop: begin first test of loop condition ...\n");
   long cond = G__test(condition);
   //fprintf(stderr, "G__exec_loop: end   first test of loop condition.\n");
   //
   //  Destroy all temporaries created during expression evaluation.
   //
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_loop: Destroy temp objects after first condition evaluation, currently at G__templevel %d  %s:%d\n"
         , G__templevel
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   G__free_tempobject();
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__exec_loop: After first condition evaluation, decrement G__templevel %d --> %d  %s:%d\n"
         , G__templevel
         , G__templevel - 1
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   --G__templevel;
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
         if (result.type == G__block_break.type) {
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
                     int store_no_exec_compile = G__no_exec_compile;
                     G__no_exec_compile = 1;
                     // Tell parser to go until a right curly brace is seen.
                     // Call the parser.
                     G__exec_statement(&brace_level);
                     G__no_exec_compile = store_no_exec_compile;
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
                        int store_no_exec_compile = G__no_exec_compile;
                        G__no_exec_compile = 1;
                        // Tell parser to go until a right curly brace is seen.
                        // And call parser.
                        G__exec_statement(&brace_level);
                        // Restore state.
                        G__no_exec_compile = store_no_exec_compile;
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
         int store_no_exec_compile = G__no_exec_compile;
         if (executed_break) {
            G__no_exec_compile = 1;
         }
         //
         //  Execute the bottom of loop expression.
         //
         if (!foraction.empty()) {
            //fprintf(stderr, "G__exec_loop: begin executing loop actions ...\n");
            std::list<G__FastAllocString>::const_iterator i = foraction.begin();
            std::list<G__FastAllocString>::const_iterator end = foraction.end();
            for (; i != end; ++i) {
               G__getexpr(*i);
            }
            //fprintf(stderr, "G__exec_loop: end executing loop actions.\n");
         }
         if (executed_break) {
            G__no_exec_compile = store_no_exec_compile;
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
            // Test the loop condition.
#ifdef G__ASM
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "\n!!!G__exec_loop: Before condition evaluation, increment G__templevel %d --> %d  %s:%d\n"
                  , G__templevel
                  , G__templevel + 1
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
#endif // G__ASM
            ++G__templevel;
            //fprintf(stderr, "G__exec_loop: Testing condition '%s'.\n", condition);
            cond = G__test(condition);
            //fprintf(stderr, "G__exec_loop: Testing condition result: %d.\n", cond);
            //
            //  Destroy all temporaries created during expression evaluation.
            //
#ifdef G__ASM
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "\n!!!G__exec_loop: Destroy temp objects after condition evaluation, currently at G__templevel %d  %s:%d\n"
                  , G__templevel
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
#endif // G__ASM
            G__free_tempobject();
#ifdef G__ASM
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "\n!!!G__exec_loop: After condition evaluation, decrement G__templevel %d --> %d  %s:%d\n"
                  , G__templevel
                  , G__templevel - 1
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
#endif // G__ASM
            --G__templevel;
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

extern "C" {

//______________________________________________________________________________
//
//  Jump statements.  return, goto
//

//______________________________________________________________________________
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
   //  Evaluate any expression which was given.
   //
   // FIXME: There are problems with the handling of G__no_exec here!
   if (!statement[0]) {
      // -- We do *not* have an expression.
      G__no_exec = 1;
      buf = G__null;
   }
   else {
      // We have an expression, evaluate it.
      G__no_exec = 0;
      buf = G__getexpr(statement);
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

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Parsing.
//

//______________________________________________________________________________
static int G__search_gotolabel(char* label, fpos_t* pfpos, int line, int* pmparen)
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
      G__strlcpy(G__gotolabel, label, G__MAXNAME);
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
      G__FastAllocString token(G__LONGLINE);
      // The extraneous punctuation is here to keep from overflowing token.
      c = G__fgetstream(token, 0, "'\"{}:();");
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
   int store_tagdefining;
   fpos_t pos;
   G__FastAllocString temp(G__ONELINE);
   // Look ahead to see if we have a "::".
   int c = G__fgetc();
   if (c == ':') {
      // X::memberfunc() {...};
      //    ^  c == ':'
      // -- Member function definition.
      if (
            G__prerun &&
            (G__func_now == -1) &&
            (
             ((G__def_tagnum == -1) || (G__struct.type[G__def_tagnum] == 'n')) ||
             memfunc_def_flag ||
             (G__tmplt_def_tagnum != -1)
            )
         ) {
         // --
         int store_def_tagnum = G__def_tagnum;
         int store_def_struct_member = G__def_struct_member;
         // X<T>::TYPE X<T>::f()
         //      ^
         fgetpos(G__ifile.fp, &pos);
         line = G__ifile.line_number;
         if (G__dispsource) G__disp_mask = 1000;
         c = G__fgetname_template(temp, 0, "(;&*");
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
         G__def_tagnum = G__defined_tagname(statement + ispntr, 0);
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

//______________________________________________________________________________
static int G__IsFundamentalDecl()
{
   // -- FIXME: Describe this function!
   // -- Used only by G__keyword_anytime_5.
   // -- FIXME: We don't check for float, double, or long double!
   // -- FIXME: We don't accept a macro which expands to a fundamental type.
   G__FastAllocString type_name(G__ONELINE);
   // Store file position.
   int linenum = G__ifile.line_number;
   fpos_t pos;
   fgetpos(G__ifile.fp, &pos);
   G__disp_mask = 1000;
   /*int c =*/ G__fgetname_template(type_name, 0, "(");
   int result = 1;
   if (!strcmp(type_name, "class") || !strcmp(type_name, "struct") || !strcmp(type_name, "union")) {
      result = 0;
   }
   else {
      int tagnum = G__defined_tagname(type_name, 1);
      if (tagnum != -1) {
         result = 0;
      }
      else {
         int typenum = G__defined_typename(type_name);
         if (typenum != -1) {
            switch (G__newtype.type[typenum]) {
               case 'b': // unsigned char
               case 'c': // char
               case 'r': // unsigned short
               case 's': // short
               case 'h': // unsigned int
               case 'i': // int
               case 'k': // unsigned long
               case 'l': // long
                  result = 1;
                  break;
               default:
                  result = 0;
            }
         }
         else {
            if (
                  !strcmp(type_name, "unsigned") ||
                  !strcmp(type_name, "char") ||
                  !strcmp(type_name, "short") ||
                  !strcmp(type_name, "int") ||
                  !strcmp(type_name, "long")
               ) {
               result = 1;
            }
            else {
               result = 0;
            }
         }
      }
   }
   // Restore file position.
   G__ifile.line_number = linenum;
   fsetpos(G__ifile.fp, &pos);
   G__disp_mask = 0;
   return result;
}

//______________________________________________________________________________
static void G__unsignedintegral()
{
   // -- FIXME: Describe this function!
   // -- FIXME: This routine can not handle 'unsigned int*GetXYZ();' [it must have a space after the *]
   
   // Remember the current file position.
   fpos_t pos;
   fgetpos(G__ifile.fp, &pos);
   G__unsigned = -1;
   // Scan the next identifier token in.
   G__FastAllocString name(G__MAXNAME);
   G__fgetname(name, 0, "(");
   //
   //  And compare against the integral types.
   //
   if (!strcmp(name, "int")) {
      G__var_type = 'i' - 1;
   }
   else if (!strcmp(name, "char")) {
      G__var_type = 'c' - 1;
   }
   else if (!strcmp(name, "short")) {
      G__var_type = 's' - 1;
   }
   else if (!strcmp(name, "long")) {
      G__var_type = 'l' - 1;
   }
   else if (!strcmp(name, "int*")) {
      G__var_type = 'I' - 1;
   }
   else if (!strcmp(name, "char*")) {
      G__var_type = 'C' - 1;
   }
   else if (!strcmp(name, "short*")) {
      G__var_type = 'S' - 1;
   }
   else if (!strcmp(name, "long*")) {
      G__var_type = 'L' - 1;
   }
   else if (!strcmp(name, "int&")) {
      G__var_type = 'i' - 1;
      G__reftype = G__PARAREFERENCE;
   }
   else if (!strcmp(name, "char&")) {
      G__var_type = 'c' - 1;
      G__reftype = G__PARAREFERENCE;
   }
   else if (!strcmp(name, "short&")) {
      G__var_type = 's' - 1;
      G__reftype = G__PARAREFERENCE;
   }
   else if (!strcmp(name, "long&")) {
      G__var_type = 'l' - 1;
      G__reftype = G__PARAREFERENCE;
   }
   else if (strchr(name, '*')) {
      // -- May have been a pointer.
      bool nomatch = false;
      if (!strncmp(name, "int*", 4) || !strncmp(name, "*", 1)) {
         G__var_type = 'I' - 1;
      }
      else if (!strncmp(name, "char*", 5)) {
         G__var_type = 'C' - 1;
      }
      else if (!strncmp(name, "short*", 6)) {
         G__var_type = 'S' -1;
      }
      else if (!strncmp(name, "long*", 5)) {
         G__var_type = 'L' -1;
      } else {
         // No match at all.
         nomatch = true;
      }
      if (nomatch) {
         // Set to unsigned int and rewind
         G__var_type = 'i' - 1;
         fsetpos(G__ifile.fp, &pos);         
      } else {
         if (strstr(name, "******")) {
            G__reftype = G__PARAP2P + 4;
         }
         else if (strstr(name, "*****")) {
            G__reftype = G__PARAP2P + 3;
         }
         else if (strstr(name, "****")) {
            G__reftype = G__PARAP2P + 2;
         }
         else if (strstr(name, "***")) {
            G__reftype = G__PARAP2P + 1;
         }
         else if (strstr(name, "**")) {
            G__reftype = G__PARAP2P;
         }
      }
   }
   else {
      // -- Just plain unsigned is an unsigned int.
      G__var_type = 'i' - 1;
      // Undo the scan of the next identifier token.
      // FIXME: The line number, dispmask, and macro expansion state could be wrong now.
      fsetpos(G__ifile.fp, &pos);
   }
   //
   //  Declare or define the variable.
   //
   G__define_var(-1, -1);
   //
   //  And reset the parse state.
   //
   // Note: We do *not* reset G__var_type here.
   G__reftype = G__PARANORMAL;
   G__unsigned = 0;
}

//______________________________________________________________________________
static void G__externignore()		
{
   // -- Handle extern "...", EXTERN "..."
   int flag = 0;
   G__FastAllocString fname(G__MAXFILENAME);
   int c = G__fgetstream(fname, 0, "\"");
   int store_iscpp = G__iscpp;
   // FIXME: We should handle "C++" as well!
   if (!strcmp(fname, "C")) {
      // -- Handle extern "C", EXTERN "C"
      G__iscpp = 0;
   }
   else {
      // -- Handle extern "filename", EXTERN "filename".
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
   // -- Handle a friend declaration.
   //
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
   G__FastAllocString classname(G__LONGLINE);
   int c = G__fgetname_template(classname, 0, ";");
   int tagtype = 0;
   if (c == ';') {
      tagtype = 'c';
   }
   else if (isspace(c)) {
      if (!strcmp(classname, "class")) {
         c = G__fgetname_template(classname, 0, ";");
         tagtype = 'c';
      }
      else if (!strcmp(classname, "struct")) {
         c = G__fgetname_template(classname, 0, ";");
         tagtype = 's';
      }
      else {
         if (!strcmp(classname, "const") || !strcmp(classname, "volatile") || !strcmp(classname, "register")) {
            c = G__fgetname_template(classname, 0, ";");
         }
         if ((c == ';') || (c == ',')) {
            tagtype = 'c';
         }
      }
   }
   int envtagnum = G__get_envtagnum();
   if (envtagnum == -1) {
      G__genericerror("Error: friend keyword appears outside class definition");
   }
   int store_tagnum = G__tagnum;
   int store_def_tagnum = G__def_tagnum;
   int store_def_struct_member = G__def_struct_member;
   int store_tagdefining = G__tagdefining;
   int store_access = G__access;
   G__friendtagnum = envtagnum;
   if (G__tagnum != -1) {
      G__tagnum = G__struct.parent_tagnum[G__tagnum];
   }
   if (G__def_tagnum != -1) {
      G__def_tagnum = G__struct.parent_tagnum[G__def_tagnum];
   }
   if (G__tagdefining != -1) {
      G__tagdefining = G__struct.parent_tagnum[G__tagdefining];
   }
   if ((G__tagdefining != -1) || (G__def_tagnum != -1)) {
      G__def_struct_member = 1;
   }
   else {
      G__def_struct_member = 0;
   }
   G__access = G__PUBLIC;
   G__var_type = 'p';
   if (tagtype) {
      while (classname[0]) {
         int def_tagnum = G__def_tagnum;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         int tagdefining = G__tagdefining;
         int friendtagnum = G__defined_tagname(classname, 2);
         G__def_tagnum = def_tagnum;
         G__tagdefining = tagdefining;
         if (friendtagnum == -1) {
            friendtagnum = G__search_tagname(classname, tagtype);
         }
         // friend class ...;
         if (envtagnum != -1 && friendtagnum != -1) {
            struct G__friendtag* friendtag = G__struct.friendtag[friendtagnum];
            if (friendtag) {
               while (friendtag->next) {
                  friendtag = friendtag->next;
               }
               friendtag->next = (struct G__friendtag*) malloc(sizeof(struct G__friendtag));
               friendtag->next->next = 0;
               friendtag->next->tagnum = envtagnum;
            }
            else {
               G__struct.friendtag[friendtagnum] = (struct G__friendtag*) malloc(sizeof(struct G__friendtag));
               friendtag = G__struct.friendtag[friendtagnum];
               friendtag->next = 0;
               friendtag->tagnum = envtagnum;
            }
         }
         if (c != ';') {
            c = G__fgetstream(classname, 0, ";,");
         }
         else {
            classname[0] = 0;
         }
      }
   }
   else {
      // friend type f() {...};
      fsetpos(G__ifile.fp, &pos);
      G__ifile.line_number = line_number;
      // friend function belongs to the inner-most namespace
      // not the parent class! In fact, this fix is not perfect, because
      // a friend function can also be a member function. This fix works
      // better only because there is no strict checking for non-member
      // function.
      if ((G__globalcomp != G__NOLINK) && (G__def_tagnum != -1) && (G__struct.type[G__def_tagnum] != 'n')) {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: This friend declaration may cause creation of wrong stub function in dictionary. Use '#pragma link off function ...;' to avoid it.");
            G__printlinenum();
         }
      }
      while ((G__def_tagnum != -1) && (G__struct.type[G__def_tagnum] != 'n')) {
         G__def_tagnum = G__struct.parent_tagnum[G__def_tagnum];
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
   G__friendtagnum = -1;
   // Restore the autoload flag.
   G__set_class_autoloading(autoload_old);
}
#endif // G__FRIEND

} // extern "C"


//______________________________________________________________________________
static int G__keyword_anytime_5(G__FastAllocString& statement)
{
   // -- Handle a function-local const declaration, or #else, #elif, and #line
   int c = 0;
   int iout = 0;
   if (
      (G__globalcomp == G__NOLINK) && // we are not making a dictionary, and
      (G__func_now >= 0) && // we are parsing a function body, and
      (
         G__prerun || // not running, or
         ((G__asm_wholefunction == G__ASM_FUNC_COMPILE) && !G__ansiheader) // compiling whole function and not currently in the (ansi) function header,
      ) && // and,
      !strcmp(statement, "const") && // we are the "const" keyword, and
      G__IsFundamentalDecl() // the following type is a fundamental type
   ) {
      // -- We have a function-local const of non-class type.
      //
      // Handle this as a static because the value may be needed
      // as a constant expression in the array index of a subsequent
      // static array declaration.  For example:
      //
      //      void f() {
      //        const int array_size = 3;
      //        static int my_array[array_size] = { 1, 2, 3 };
      //      }
      //
      // If we do not handle it as a static, then we will not have the
      // value available during prerun when we parse the static array
      // variable declaration and allocate memory for it.
      //
      G__constvar = G__CONSTVAR;
      G__const_setnoerror();
      struct G__var_array* store_local = G__p_local;
      if (G__prerun && (G__func_now != -1)) {
         G__p_local = 0;
      }
      G__static_alloc = 1;
      int store_no_exec = G__no_exec;
      G__no_exec = 0;
      int brace_level = 0;
      G__exec_statement(&brace_level);
      G__static_alloc = 0;
      G__no_exec = store_no_exec;
      G__p_local = store_local;
      G__const_resetnoerror();
      G__security_error = G__NOERROR;
      G__return = G__RETURN_NON;
      return 1;
   }
   if (statement[0] != '#') {
      // -- Reject anything that is not a preprocessor directive.
      return 0;
   }
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

//______________________________________________________________________________
static int G__keyword_anytime_6(G__FastAllocString& statement)
{
   // -- Handle "static", "return", "#ifdef", "#endif", "#undef", and "#ident"
   if (!strcmp(statement, "static")) {
      // -- We have the static keyword.
      //
      // If in prerun then allocate memory,
      // other wise are executing, so get
      // preallocated memory.
      //
      struct G__var_array* store_local = G__p_local;
      if (G__prerun && (G__func_now != -1)) {
         // -- Function local static during prerun, put into global variable array.
         G__p_local = 0;
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

//______________________________________________________________________________
static int G__keyword_anytime_7(G__FastAllocString& statement)
{
   // -- Handle "#define", "#ifndef", and "#pragma".
   /***********************************
    * 1)
    *  #ifndef macro   <---
    *  #endif
    * 2)
    *  #ifndef macro   <---
    *  #else
    *  #endif
    ***********************************/
   if (!strcmp(statement, "#define")) {
      // -- Handle #define.
      // Save state.
      int store_tagnum = G__tagnum;
      int store_typenum = G__typenum;
      struct G__var_array* store_local = G__p_local;
      //
      //  Parse the macro definition.
      //
      G__p_local = 0;
      G__var_type = 'p';
      G__definemacro = 1;
      G__define();
      //
      //  Restore state.
      //
      G__definemacro = 0;
      G__p_local = store_local;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      // And return success.
      return 1;
   }
   if (!strcmp(statement, "#ifndef")) {
      int stat = G__pp_ifdef(0);
      return(stat);
   }
   if (!strcmp(statement, "#pragma")) {
      G__pragma();
      return(1);
   }
   return 0;
}

//______________________________________________________________________________
static int G__keyword_anytime_8(G__FastAllocString& statement)
{
   // -- Handle "template" and "explicit" keywords.
   //
   // template  <class T> class A { ... };
   // template  A<int>;
   // template  class A<int>;
   //          ^
   //
   if (!strcmp(statement, "template")) {
      int c;
      fpos_t pos;
      int line_number;
      G__FastAllocString tcname(G__ONELINE);
      line_number = G__ifile.line_number;
      fgetpos(G__ifile.fp, &pos);
      c = G__fgetspace();
      if ('<' == c) {
         /* if '<' comes, this is an ordinary template declaration */
         G__ifile.line_number = line_number;
         fsetpos(G__ifile.fp, &pos);
         return(0);
      }
      /* template  A<int>; this is a template instantiation */
      tcname[0] = c;
      fseek(G__ifile.fp, -1, SEEK_CUR);
      G__disp_mask = 1;
      c = G__fgetname_template(tcname, 0, ";");
      if (strcmp(tcname, "class") == 0 ||
            strcmp(tcname, "struct") == 0) {
         c = G__fgetstream_template(tcname, 0, "0, ;");
      }
      else if (isspace(c)) {
         size_t len = strlen(tcname);
         char store_c;
         while (len && ('&' == tcname[len-1] || '*' == tcname[len-1])) --len;
         store_c = tcname[len];
         tcname[len] = 0;
         if (G__istypename(tcname)) {
            G__ifile.line_number = line_number;
            fsetpos(G__ifile.fp, &pos);
            int brace_level = 0;
            G__exec_statement(&brace_level);
            return(1);
         }
         else {
            tcname[len] = store_c;
            c = G__fgetstream_template(tcname, strlen(tcname), ";");
         }
      }
      if (!G__defined_templateclass(tcname)) {
         G__instantiate_templateclass(tcname, 0);
      }
      return 1;
   }
   if (!strcmp(statement, "explicit")) {
      G__isexplicit = 1;
      return 1;
   }
   return 0;
}

static int G__defined_type(G__FastAllocString& type_name, int len)
{
   // -- Handle a possible declaration, return 0 if not, return 1 if good.
   //
   //  Note: This routine is part of the parser proper.
   //
   int refrewind = -2;
   //
   //  Check for a destructor declaration.
   //
   if (G__prerun && (type_name[0] == '~')) {
      // -- We have found a destructor declaration in prerun.
      G__var_type = 'y';
      int cin = G__fignorestream("(");
      type_name.Resize(len + 2);
      type_name[len++] = cin;
      type_name[len] = '\0';
      G__make_ifunctable(type_name);
      return 1;
   }
   //
   //  Check for a single non-printable character.
   //
   if (!isprint(type_name[0]) && (len == 1)) {
      // -- We found a single non-printable character followed by a space, just accept it and continue.
      return 1;
   }
   //
   //  Remember the current position in case we fail.
   //
   fpos_t pos;
   fgetpos(G__ifile.fp, &pos);
   int line = G__ifile.line_number;
   // Remember the passed type_name in case we fail.
   G__FastAllocString store_typename(type_name);
   // Remember tagnum and typenum in case we fail.
   int store_tagnum = G__tagnum;
   int store_typenum = G__typenum;
   // Skip any leading whitespace.
   int cin = G__fgetspace();
   //
   // check if this is a declaration or not
   // declaration:
   //     type varname... ; type *varname
   //          ^ must be alphabet '_' , '*' or '('
   // else
   //   if not alphabet, return
   //     type (param);   function name
   //     type = expr ;   variable assignment
   //
   //--
   switch (cin) {
      case '*':
      case '&':
         cin = G__fgetc();
         fseek(G__ifile.fp, -2, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 2;
         }
         if (cin == '=') {
            return 0;
         }
         break;
      case '(':
      case '_':
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         break;
      default:
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         if (!isalpha(cin)) {
            return 0;
         }
         break;
   }
   if (type_name[len-1] == '&') {
      G__reftype = G__PARAREFERENCE;
      type_name[--len] = '\0';
      --refrewind;
   }
   //
   if ((len > 2) && (type_name[len-1] == '*') && (type_name[len-2] == '*')) {
      // -- We have a pointer to pointer.
      len -= 2;
      type_name[len] = '\0';
      // type** a;
      //     ^<<^
      fsetpos(G__ifile.fp, &pos);
      G__ifile.line_number = line;
      fseek(G__ifile.fp, -1, SEEK_CUR);
      cin = G__fgetc();
      if (cin == '*') {
         // -- We have a fake space.
         fseek(G__ifile.fp, refrewind, SEEK_CUR);
      }
      else {
         fseek(G__ifile.fp, refrewind - 1, SEEK_CUR);
      }
      if (G__dispsource) {
        G__disp_mask = 2;
      }
   }
   else if ((len > 1) && (type_name[len-1] == '*')) {
      int cin2;
      len -= 1;
      type_name[len] = 0;
      fsetpos(G__ifile.fp, &pos);
      G__ifile.line_number = line;
      // To know how much to rewind we need to know if there is a fakespace.
      fseek(G__ifile.fp, -1, SEEK_CUR);
      cin = G__fgetc();
      if (cin == '*') {
         // -- We have a fake space.
         fseek(G__ifile.fp, refrewind + 1, SEEK_CUR);
      }
      else {
         fseek(G__ifile.fp, refrewind, SEEK_CUR);
      }
      if (G__dispsource) {
         G__disp_mask = 1;
      }
      cin2 = G__fgetc();
      if (!isalnum(cin2) && (cin2 != '>')) {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
      }
   }
   //
   //  Check for a typedef name.
   //
   G__typenum = G__defined_typename(type_name);
   if (G__typenum != -1) {
      // -- We have a typedef name.
      // Note: G__var_type was set by the G__defined_typename() call above to G__newtype.type[G__typenum] + ptroffset.
      G__tagnum = G__newtype.tagnum[G__typenum];
      G__reftype += G__newtype.reftype[G__typenum];
      G__typedefnindex = G__newtype.nindex[G__typenum];
      G__typedefindex = G__newtype.index[G__typenum];
      // FIXME: We ignore G__newtype.isconst[G__typenum]!
   }
   else {
      // -- It was not a typedef name.
      //
      //  Check for a class, enum, namespace, struct, or union name.
      //
      G__tagnum = G__defined_tagname(type_name, 1);
      if (G__tagnum != -1) {
         // -- Ok, we found it, now check again as a typedef name
         // to pick up template aliases that might have been generated
         // by template instantiation during above G__defined_tagname
         G__typenum = G__defined_typename(type_name);
         if (G__typenum != -1) {
            // Note: G__var_type was set by the G__defined_typename() call above to G__newtype.type[G__typenum] + ptroffset.
            G__reftype += G__newtype.reftype[G__typenum];
            G__typedefnindex = G__newtype.nindex[G__typenum];
            G__typedefindex = G__newtype.index[G__typenum];
            // FIXME: We ignore G__newtype.isconst[G__typenum]!
         }
      }
      else {
         // -- Nope, it was not a class, enum, namespace, struct, or union name.
         if (G__fpundeftype && (cin != '(') && ((G__func_now == -1) || (G__def_tagnum != -1))) {
            // -- We have been asked to make a list of undefined type names.
            // Declare the undefined name as a class.
            G__tagnum = G__search_tagname(type_name, 'c');
            // Output the info to the given file.
            fprintf(G__fpundeftype, "class %s; /* %s %d */\n", type_name(), G__ifile.name, G__ifile.line_number);
            fprintf(G__fpundeftype, "#pragma link off class %s;\n\n", type_name());
            if (G__tagnum > -1) { // it could be -1 if we get too many classes.
               G__struct.globalcomp[G__tagnum] = G__NOLINK;
            }
         }
         else {
            // -- Was not a known type, return.
            fsetpos(G__ifile.fp, &pos);
            G__ifile.line_number = line;
            type_name = store_typename;
            G__tagnum = store_tagnum;
            G__typenum = store_typenum;
            G__reftype = G__PARANORMAL;
            return 0;
         }
      }
      G__var_type = 'u';
   }
   //
   //  Hack an enumerator.
   //
   if ((G__tagnum != -1) && (G__struct.type[G__tagnum] == 'e')) {
      // -- We have an enumerator.
      G__var_type = 'i';
   }
   //
   //  Define a variable.
   //
   G__define_var(G__tagnum, G__typenum);
   //
   //  Restore state.
   //
   G__typedefnindex = 0;
   G__typedefindex = 0;
   G__tagnum = store_tagnum;
   G__typenum = store_typenum;
   G__reftype = G__PARANORMAL;
   // And return success.
   return 1;
}

extern "C" {

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Externally visible functions.
//

//______________________________________________________________________________
G__value G__alloc_exceptionbuffer(int tagnum)
{
   // -- FIXME: Describe this function!
   G__value buf = G__null;
   /* create class object */
   buf.obj.i = (long)malloc((size_t)G__struct.size[tagnum]);
   buf.obj.reftype.reftype = G__PARANORMAL;
   buf.type = 'u';
   buf.tagnum = tagnum;
   buf.typenum = -1;
   buf.ref = G__p_tempbuf->obj.obj.i;
   return(buf);
}

//______________________________________________________________________________
int G__free_exceptionbuffer()
{
   // -- FIXME: Describe this function!
   if (G__exceptionbuffer.ref) {
      long store_struct_offset = G__store_struct_offset;
      G__store_struct_offset = G__exceptionbuffer.ref;
      if ('u' == G__exceptionbuffer.type && G__exceptionbuffer.obj.i &&
          -1 != G__exceptionbuffer.tagnum) {
         // destruct before free
         G__FastAllocString destruct(G__ONELINE);
         int store_tagnum = G__tagnum;
         int dmy = 0;
         G__tagnum = G__exceptionbuffer.tagnum;
         if (G__CPPLINK == G__struct.iscpplink[G__tagnum]) {
            G__globalvarpointer = G__store_struct_offset;
         }
         else G__globalvarpointer = G__PVOID;
         destruct.Format("~%s()", G__fulltagname(G__tagnum, 1));
         if (G__dispsource) {
            G__fprinterr(G__serr, "!!!Destructing exception buffer %s %lx"
                         , destruct(), G__exceptionbuffer.obj.i);
            G__printlinenum();
         }
         G__getfunction(destruct, &dmy , G__TRYDESTRUCTOR);
         /* do nothing here, exception object shouldn't be stored in legacy temp buf */
         G__tagnum = store_tagnum;
         G__globalvarpointer = G__PVOID;
      }
      if (G__CPPLINK != G__struct.iscpplink[G__tagnum])
         free((void*)G__store_struct_offset);
      G__store_struct_offset = store_struct_offset;
   }
   G__exceptionbuffer = G__null;
   return(0);
}

//______________________________________________________________________________
void G__display_tempobject(const char* action)
{
   // Dump the temporary object list.
   struct G__tempobject_list* ptempbuf = G__p_tempbuf;
   G__fprinterr(G__serr, "\n%s ", action);
   while (ptempbuf) {
      if (ptempbuf->obj.type) {
         G__fprinterr(
              G__serr
            , "%d:0x%lx:(%s)0x%lx "
            , ptempbuf->level
            , (long) ptempbuf
            , G__type2string(
                   ptempbuf->obj.type
                 , ptempbuf->obj.tagnum
                 , ptempbuf->obj.typenum
                 , ptempbuf->obj.obj.reftype.reftype
                 , ptempbuf->obj.isconst
              )
            , ptempbuf->obj.obj.i
         );
      }
      else {
         G__fprinterr(
              G__serr
            , "%d:0x%lx:(%s)0x%lx "
            , ptempbuf->level
            , (long) ptempbuf
            , "NULL"
            , 0L
         );
      }
      ptempbuf = ptempbuf->prev;
   }
   G__fprinterr(G__serr, "\n");
}

//______________________________________________________________________________
int G__defined_macro(const char* macro)
{
   // -- Check if a macro is defined.
   int hash = 0;
   int iout = 0;
   G__hash(macro, hash, iout);
   struct G__var_array* var = 0;
   for (var = &G__global; var; var = var->next) {
      for (iout = 0; iout < var->allvar; ++iout) {
         if (
            ((tolower(var->type[iout]) == 'p') || (var->type[iout] == 'T')) &&
            (hash == var->hash[iout]) &&
            !strcmp(macro, var->varnamebuf[iout])
         ) {
            // -- Found.
            return 1;
         }
      }
   }
   if ((hash == 682) && !strcmp(macro, "__CINT__")) {
      return 1;
   }
   if (!G__cpp && (hash == 1704) && !strcmp(macro, "__CINT_INTERNAL_CPP__")) {
      return 1;
   }
   if ((G__iscpp || G__externblock_iscpp) && (hash == 1193) && !strcmp(macro, "__cplusplus")) {
      return 1;
   }
#ifndef G__OLDIMPLEMENTATION869
   {
      // Following fix is not completely correct. It confuses typedef names as macro.
      // Look for typedef names defined by '#define foo int'.
      int save_tagnum = G__def_tagnum;
      G__def_tagnum = -1;
      int stat = G__defined_typename(macro);
      G__def_tagnum = save_tagnum;
      if (stat >= 0) {
         return 1;
      }
   }
#endif
   // Search symbol macro table.
   if (macro != G__replacesymbol(macro)) {
      return 1;
   }
   // Search function macro table.
   {
      struct G__Deffuncmacro* deffuncmacro = 0;
      for (deffuncmacro = &G__deffuncmacro; deffuncmacro; deffuncmacro = deffuncmacro->next) {
         if (deffuncmacro->name && !strcmp(macro, deffuncmacro->name)) {
            return 1;
         }
      }
   }
   // Not found.
   return 0;
}

//______________________________________________________________________________
int G__pp_command()
{
   // -- FIXME: Describe this function!
   G__FastAllocString condition(G__ONELINE);
   int c = G__fgetname(condition, 0, "\n\r");
   if (isdigit(condition[0])) {
      if ((c != '\n') && (c != '\r')) {
         G__fignoreline();
      }
      G__ifile.line_number = atoi(condition);
   }
   else if (!strncmp(condition, "el", 2)) {
      G__pp_skip(1);
   }
   else if (!strncmp(condition, "ifdef", 5)) {
      G__pp_ifdef(1);
   }
   else if (!strncmp(condition, "ifndef", 6)) {
      G__pp_ifdef(0);
   }
   else if (!strncmp(condition, "if", 2)) {
      G__pp_if();
   }
   else if ((c != '\n') && (c != '\r')) {
      G__fignoreline();
   }
   return 0;
}

//______________________________________________________________________________
void G__pp_skip(int elifskip)
{
   // -- FIXME: Describe this function!
   G__FastAllocString oneline(G__LONGLINE*2);
   G__FastAllocString argbuf(G__LONGLINE*2);
   char* arg[G__ONELINE];
   int argn;

   FILE *fp;
   int nest = 1;
   G__FastAllocString condition(G__ONELINE);
   G__FastAllocString temp(G__ONELINE);
   long i;

   fp = G__ifile.fp;

   /* elace traced mark */
   if (0 == G__nobreak && 0 == G__disp_mask &&
         G__srcfile[G__ifile.filenum].breakpoint &&
         G__srcfile[G__ifile.filenum].maxline > G__ifile.line_number) {
      G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
      &= G__NOTRACED;
   }

   /********************************************************
    * Read lines until end of conditional compilation
    ********************************************************/
   while (nest && G__readline_FastAlloc(fp, oneline, argbuf, &argn, arg) != 0) {
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

      if (argn > 0 && arg[1][0] == '#') {
         const char* directive = arg[1] + 1; // with "#if" directive will point to "if"
         int directiveArgI = 1;
         if (arg[1][1] == 0
               || strcmp(arg[1], "#pragma") == 0
            ) {
            directive = arg[2];
            directiveArgI = 2;
         }

         if (strncmp(directive, "if", 2) == 0) {
            ++nest;
         }
         else if (strncmp(directive, "else", 4) == 0) {
            if (nest == 1 && elifskip == 0) nest = 0;
         }
         else if (strncmp(directive, "endif", 5) == 0) {
            --nest;
         }
         else if (strncmp(directive, "elif", 4) == 0) {
            if (nest == 1 && elifskip == 0) {
               int store_no_exec_compile = G__no_exec_compile;
               int store_asm_wholefunction = G__asm_wholefunction;
               int store_asm_noverflow = G__asm_noverflow;
               G__no_exec_compile = 0;
               G__asm_wholefunction = 0;
               if (!G__xrefflag) {
                  G__asm_noverflow = 0;
               }
               condition = "";
               for (i = directiveArgI + 1; i <= argn; i++) {
                  condition += arg[i];
               }
               i = strlen(oneline) - 1;
               while (i >= 0 && (oneline[i] == '\n' || oneline[i] == '\r')) {
                  --i;
               }
               if (oneline[i] == '\\') {
                  int len = strlen(condition);
                  while (1) {
                     G__fgetstream(condition, len, "\n\r");
                     if (condition[len] == '\\' && (condition[len+1] == '\n' ||
                                                    condition[len+1] == '\r')) {
                        char* p = condition + len;
                        memmove(p, p + 2, strlen(p + 2) + 1);
                     }
                     len = strlen(condition) - 1;
                     while (len > 0 && (condition[len] == '\n' || condition[len] == '\r'))
                        --len;
                     if (condition[len] != '\\') break;
                  }
               }

               /* remove comments */
               char* posComment = strstr(condition, "/*");
               if (!posComment) posComment = strstr(condition, "//");
               while (posComment) {
                  if (posComment[1] == '*') {
                     char* posCXXComment = strstr(condition, "//");
                     if (posCXXComment && posCXXComment < posComment)
                        posComment = posCXXComment;
                  }
                  if (posComment[1] == '*') {
                     const char* posCommentEnd = strstr(posComment + 2, "*/");
                     // we can have
                     // #if A /*
                     //   comment */ || B
                     // #endif
                     if (!posCommentEnd) {
                        if (G__skip_comment())
                           break;
                        if (G__fgetstream(condition, posComment - condition.data(), "\r\n") == EOF)
                           break;
                     }
                     else {
                        temp = posCommentEnd + 2;
                        condition.Resize(posComment - condition.data() + strlen(temp) + 1);
                        strcpy(posComment, temp); // Okay we allocated enough space
                     }
                     posComment = strstr(posComment, "/*");
                     if (!posComment) posComment = strstr(condition, "//");
                  }
                  else {
                     posComment[0] = 0;
                     posComment = 0;
                  }
               }

               G__noerr_defined = 1;
               if (G__test(condition)) {
                  nest = 0;
               }
               G__no_exec_compile = store_no_exec_compile;
               G__asm_wholefunction = store_asm_wholefunction;
               G__asm_noverflow = store_asm_noverflow;
               G__noerr_defined = 0;
            }
         }
      }
   }

   /* set traced mark */
   if (!G__nobreak &&
         !G__disp_mask && !G__no_exec_compile &&
         G__srcfile[G__ifile.filenum].breakpoint &&
         G__srcfile[G__ifile.filenum].maxline > G__ifile.line_number) {
      G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
      |= (!G__no_exec);
   }

   if (G__dispsource) {
      if ((G__debug || G__break || G__step
          ) &&
            ((G__prerun != 0) || (G__no_exec == 0)) &&
            (G__disp_mask == 0)) {
         G__fprinterr(G__serr, "# conditional interpretation, SKIPPED");
         G__fprinterr(G__serr, "\n%-5d", G__ifile.line_number - 1);
         G__fprinterr(G__serr, "%s", arg[0]);
         G__fprinterr(G__serr, "\n%-5d", G__ifile.line_number);
      }
   }
}

//______________________________________________________________________________
int G__pp_if()
{
   // -- FIXME: Describe this function!
   G__FastAllocString condition(G__LONGLINE);
   int c, len = 0;
   int store_no_exec_compile;
   int store_asm_wholefunction;
   int store_asm_noverflow;
   int haveOpenDefined = -1; // need to convert defined FOO to defined(FOO)
   do {
      c = G__fgetstream(condition, len, " \n\r");
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
            condition.Resize(len + 2);
            condition[len] = ')';
            condition[len + 1] = 0; // this might be the end, so terminate it
            ++len;
         }
         haveOpenDefined = -1;
      }
      else
         if (c == ' ' && len > 6 && !strcmp(condition + len - 7, "defined")) {
            haveOpenDefined = len;
            condition.Resize(len + 2);
            condition[len] = '(';
            ++len;
         }
   } while ((len > 0 && '\\' == condition[len - 1]) || c == ' ');

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
int G__pp_ifdef(int def)
{
   // -- FIXME: Describe this function!
   // def: 1 for ifdef; 0 for ifndef
   G__FastAllocString temp(G__LONGLINE);
   int notfound = 1;

   G__fgetname(temp, 0, "\n\r");

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

} // extern "C"


//______________________________________________________________________________
int G__exec_catch(G__FastAllocString& statement)
{
   // -- Handle the "catch" statement.
   int c;
   while (1) {
      fpos_t fpos;
      int line_number;

      // catch (ehclass& obj) {  }
      // ^^^^^^^
      do {
         c = G__fgetstream(statement, 0, "(};");
      }
      while ('}' == c);
      if ((c != '(') || strcmp(statement, "catch")) {
         return 1;
      }
      fgetpos(G__ifile.fp, &fpos);
      line_number = G__ifile.line_number;
      // catch (ehclass& obj) {  }
      //        ^^^^^^^^
      c = G__fgetname_template(statement, 0, ")&*");
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
         std::string excType(statement);
         if (excType == "const") {
            c = G__fgetname_template(statement, 0, ")&*");
            excType += " ";
            excType += statement;
         }
         while (c == '*' || c == '&') {
            excType += c;
            c = G__fgetname_template(statement, 0, ")&*");
         }
         
         G__value sType = G__string2type(excType.c_str());
         if (G__exceptionbuffer.type == sType.type &&
             ((G__exceptionbuffer.tagnum == sType.tagnum &&
              G__exceptionbuffer.typenum == sType.typenum) ||
             (G__exceptionbuffer.type == 'u' &&
              G__ispublicbase(sType.tagnum, G__exceptionbuffer.tagnum, G__exceptionbuffer.obj.i) != -1)
              )) {
            // catch(ehclass& obj) { match }
            G__value store_ansipara = G__ansipara;
            G__ansipara = G__exceptionbuffer;
            G__ansiheader = 1;
            G__funcheader = 1;
            G__ifile.line_number = line_number;
            fsetpos(G__ifile.fp, &fpos);
            int brace_level = 0;
            G__exec_statement(&brace_level); // declare exception handler object
            G__globalvarpointer = G__PVOID;
            G__ansiheader = 0;
            G__funcheader = 0;
            G__ansipara = store_ansipara;
            brace_level = 0;
            G__exec_statement(&brace_level); // exec catch block body
            break;
         }
         // catch(ehclass& obj) { unmatch }
         if (c != ')') {
            c = G__fignorestream(")");
         }
         G__no_exec = 1;
         int brace_level = 0;
         G__exec_statement(&brace_level);
         G__no_exec = 0;
      }
   }
   G__free_exceptionbuffer();
   return 0;
}

extern "C"{

//______________________________________________________________________________
int G__skip_comment()
{
   // -- Skip a C-style comment, must be called immediately after '/*' is scanned.
   // Return value is either 0 or EOF.
   int c0 = G__fgetc();
   if (c0 == EOF) {
      G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
      if (G__key) {
         if (system("key .cint_key -l execute")) {
            G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
         }
      }
      G__eof = 2;
      return EOF;
   }
   int c1 = G__fgetc();
   if (c1 == EOF) {
      G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
      if (G__key) {
         if (system("key .cint_key -l execute")) {
            G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
         }
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
            if (system("key .cint_key -l execute")) {
               G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
            }
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
int G__skip_comment_peek()
{
   // -- Skip a C-style comment during a peek, must be called immediately after '/*' is scanned.
   // Return value is either 0 or EOF.
   int c0 = fgetc(G__ifile.fp);
   if (c0 == EOF) {
      G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
      if (G__key) {
         if (system("key .cint_key -l execute")) {
            G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
         }
      }
      G__eof = 2;
      return EOF;
   }
   int c1 = fgetc(G__ifile.fp);
   if (c1 == EOF) {
      G__genericerror("Error: File ended unexpectedly while reading a C-style comment.");
      if (G__key) {
         if (system("key .cint_key -l execute")) {
            G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
         }
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
            if (system("key .cint_key -l execute")) {
               G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
            }
         }
         G__eof = 2;
         return EOF;
      }
   }
   return 0;
}

//______________________________________________________________________________
//______________________________________________________________________________
//
//  The beginning of parsing and execution.
//

//______________________________________________________________________________
G__value G__exec_statement(int* mparen)
{
   // -- Execute statement list  { ... ; ... ; ... ; }.
   int c = 0;
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
   G__FastAllocString statement(G__LONGLINE);
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
         c = ' ';
         add_fake_space = 0;
         fake_space = 1;
      }
      else {
         c = G__fgetc();
      }
      discard_space = 0;
      read_again:
      statement.Set(iout, 0);
      if (!G__prerun) {
         //fprintf(stderr, "G__exec_statement: c: '%c' pr: %d ne: %d nec: %d ano: %d mp: %d io: %d st: '%s'\n", c, G__prerun, G__no_exec, G__no_exec_compile, G__asm_noverflow, *mparen, iout, statement);
      }
      switch (c) {
         // --
         // --
#ifdef G__OLDIMPLEMENTATIONxxxx_YET
         case ',':
            if (!G__ansiheader) {
               break;
            }
            // --
#endif // G__OLDIMPLEMENTATIONxxxx_YET
         case '\n':
            // -- Handle a newline.
            if (*mparen != mparen_old) {
               // -- Update the line numbers of any dangling parentheses.
               size_t mparen_lines_size = mparen_lines.size();
               while (mparen_lines_size < (size_t)*mparen) {
                  // The stream has already read the newline, so take line_number minus one.
                  mparen_lines.push(G__ifile.line_number - 1);
                  ++mparen_lines_size;
               }
               while (mparen_lines_size > (size_t)*mparen) {
                  mparen_lines.pop();
                  --mparen_lines_size;
               }
            }
            // Intentionally fallthrough.
         case ' ':
         case '\t':
         case '\r':
         case '\f':
            //fprintf(stderr, "G__exec_statement: Enter whitespace case. sf: %d\n", spaceflag);
            // -- Handle whitespace.
            commentflag = 0;
            // ignore these character
            if (single_quote || double_quote) {
               statement.Set(iout++, c);
            }
            else {
               after_replacement:
               if (!fake_space) {
                  discard_space = 1;
               }
               if (spaceflag == 1) {
                  // -- Take action on space, even if skipping code.  Do preprocessing and look for declarations.
                  statement.Set(iout, 0);
                  // search keyword
                  G__preproc_again:
                  if ((statement[0] == '#') && isdigit(statement[1])) {
                     // -- Handle preprocessor directive "#<number> <filename>", a CINT extension to the standard.
                     // -- # [line] <[filename]>
                     int stat = G__setline(statement, c, &iout);
                     if (stat) {
                        goto G__preproc_again;
                     }
                     spaceflag = 0;
                     iout = 0;
                  }
                  //fprintf(stderr, "G__exec_statement: whitespace case, switch on iout. iout: %d\n", iout);
                  switch (iout) {
                     case 1:
                        // -- Handle preprocessor directive "#<number> <filename>", a CINT extension to the standard.
                        // -- # [line] <[filename]>
                        if (statement[0] == '#') {
                           int stat = G__setline(statement, c, &iout);
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
                           if ((c != '\n') && (c != '\r')) {
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
                           G__FastAllocString casepara(G__ONELINE);
                           G__fgetstream(casepara, 0, ":");
                           c = G__fgetc();
                           while (c == ':') {
                              casepara += "::";
                              size_t lenxxx = strlen(casepara);
                              G__fgetstream(casepara, lenxxx, ":");
                              c = G__fgetc();
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
                              G__value result = G__exec_switch_case(casepara);
                              if (G__switch_searching) {
                                 // -- We are searching for a matching case, return value of case expression.
                                 return result;
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
                  statement.Set(iout, 0);
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
                           if ((result.type == G__block_goto.type) && (result.ref == G__block_goto.ref)) {
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
                           G__define_var(-1, -1);
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
                           c = G__fgetspace();
                           if (c == '(') {
                              fseek(G__ifile.fp, -1, SEEK_CUR);
                              if (G__dispsource) {
                                 G__disp_mask = 1;
                              }
                              statement.Set(iout++, ' ');
                              spaceflag |= 1;
                              // a little later this string will be passed to subfunctions
                              // that expect the string to be terminated
                              statement.Set(iout, '0');
                           }
                           else {
                              statement.Set(0, c);
                              c = G__fgetstream_template(statement, 1, ";");
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
                        // -- Handle char, FILE, long, void, bool, int*, int&, enum, auto, (new, goto.
                        if (!strcmp(statement, "char")) {
                           G__var_type = 'c' + G__unsigned;
                           G__define_var(-1, -1);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "FILE")) {
                           G__var_type = 'e' + G__unsigned;
                           G__define_var(-1, -1);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "long")) {
                           G__var_type = 'l' + G__unsigned;
                           G__define_var(-1, -1);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "void")) {
                           G__var_type = 'y' + G__unsigned;
                           G__define_var(-1, -1);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "bool")) {
                           G__var_type = 'g' + G__unsigned;
                           G__define_var(-1, -1);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "int*")) {
                           G__typepdecl = 1;
                           G__var_type = 'I' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "int&")) {
                           G__var_type = 'i' + G__unsigned;
                           G__reftype = G__PARAREFERENCE;
                           G__define_var(-1, -1);
                           G__reftype = G__PARANORMAL;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "enum")) {
                           G__var_type = 'u';
                           G__define_struct('e');
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "auto")) {
                           // -- Handle auto, we just ignore it.
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "(new")) {
                           // -- Handle a parenthesized new expression.
                           // Check for placement new.
                           c = G__fgetspace();
                           if (c == '(') {
                              // -- We have a placement new expression.
                              // Backup one character.
                              // FIXME: The line number, dispmask, and macro expansion state may be wrong now!
                              fseek(G__ifile.fp, -1, SEEK_CUR);
                              // And insert a fake space into the statement.
                              statement.Set(iout++, ' ');
                              // And terminate the buffer.
                              statement.Set(iout, 0);
                              // Flag that any whitespace should now trigger a semantic action.
                              spaceflag |= 1;
                           }
                           else {
                              // Insert a fake space into the statement (right after the 'new').
                              statement.Set(iout++, ' ');
                              // Then add in the character that terminated the peek ahead.
                              statement.Set(iout++, c);
                              // Skip showing the next character.
                              // FIXME: This is wrong!
                              if (G__dispsource) {
                                 // -- We are display source code.
                                 G__disp_mask = 1;
                              }
                              // Scan the reset of the parenthesised new expression.
                              c = G__fgetstream_template(statement, iout, ")");
                              iout = strlen(statement);
                              statement.Set(iout++, c);
                              // And terminate the statement buffer.
                              statement.Set(iout, 0);
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
                           c = G__fgetstream(statement, 0, ";");
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
                        if (!strcmp(statement, "short")) {
                           G__var_type = 's' + G__unsigned;
                           G__define_var(-1, -1);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "float")) {
                           G__var_type = 'f' + G__unsigned;
                           G__define_var(-1, -1);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "char*")) {
                           G__typepdecl = 1;
                           G__var_type = 'C' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "char&")) {
                           G__var_type = 'c' + G__unsigned;
                           G__reftype = G__PARAREFERENCE;
                           G__define_var(-1, -1);
                           G__reftype = G__PARANORMAL;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "bool&")) {
                           G__var_type = 'g' + G__unsigned;
                           G__reftype = G__PARAREFERENCE;
                           G__define_var(-1, -1);
                           G__reftype = G__PARANORMAL;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "FILE*")) {
                           G__typepdecl = 1;
                           G__var_type = 'E' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "long*")) {
                           G__typepdecl = 1;
                           G__var_type = 'L' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "bool*")) {
                           G__typepdecl = 1;
                           G__var_type = 'G' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "long&")) {
                           G__var_type = 'l' + G__unsigned;
                           G__reftype = G__PARAREFERENCE;
                           G__define_var(-1, -1);
                           G__reftype = G__PARANORMAL;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "void*")) {
                           G__typepdecl = 1;
                           G__var_type = 'Y' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "class")) {
                           G__var_type = 'u';
                           G__define_struct('c');
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "union")) {
                           G__var_type = 'u';
                           G__define_struct('u');
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "using")) {
                           // -- Handle 'using ...'.
                           //                 ^
                           G__using_namespace();
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "throw")) {
                           // -- Handle 'throw expr;'.
                           //                 ^
                           G__exec_throw(statement);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                        }
                        if (!strcmp(statement, "const")) {
                           // -- Handle 'const ...'.
                           //                 ^
                           // Set the 'const' seen flag.
                           G__constvar = G__CONSTVAR;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                        }
                        break;
                     case 6:
                        // -- Handle double, struct, short*, short&, float*, return, delete, friend, extern, EXTERN, signed, inline, #error
                        if (!strcmp(statement, "double")) {
                           G__var_type = 'd' + G__unsigned;
                           G__define_var(-1, -1);
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "struct")) {
                           G__var_type = 'u';
                           G__define_struct('s');
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "short*")) {
                           G__typepdecl = 1;
                           G__var_type = 'S' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "short&")) {
                           G__var_type = 's' + G__unsigned;
                           G__reftype = G__PARAREFERENCE;
                           G__define_var(-1, -1);
                           G__reftype = G__PARANORMAL;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "float*")) {
                           G__typepdecl = 1;
                           G__var_type = 'F' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "float&")) {
                           G__var_type = 'f' + G__unsigned;
                           G__reftype = G__PARAREFERENCE;
                           G__define_var(-1, -1);
                           G__reftype = G__PARANORMAL;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "return")) {
                           // -- Handle 'return ...';
                           //                  ^
                           G__fgetstream_new(statement, 0, ";");
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
                           int c = G__fgetstream(statement, 0, "[;");
                           iout = 0;
                           if (c == '[') {
                              if (!statement[0]) {
                                 c = G__fgetstream(statement, 0, "]");
                                 c = G__fgetstream(statement, 0, ";");
                                 iout = 1;
                              }
                              else {
                                 statement += "[";
                                 c = G__fgetstream(statement, strlen(statement), "]");
                                 statement += "]";
                                 c = G__fgetstream(statement, strlen(statement), ";");
                              }
                           }
                           // Note: iout == 1, if 'delete[]'
                           int largestep = 0;
                           if (G__breaksignal) {
                              int ret = G__beforelargestep(statement, &iout, &largestep);
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
                           if (largestep) {
                              G__afterlargestep(&largestep);
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
                           int c = G__fgetspace();
                           if (c != '"') {
                              // -- Handle 'extern var...' and 'EXTERN var...'.
                              // We just ignore it.
                              fseek(G__ifile.fp, -1, SEEK_CUR);
                              if (c == '\n') {
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
                        if (!strcmp(statement, "double*")) {
                           G__typepdecl = 1;
                           G__var_type = 'D' + G__unsigned;
                           G__define_var(-1, -1);
                           G__typepdecl = 0;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "double&")) {
                           G__var_type = 'd' + G__unsigned;
                           G__reftype = G__PARAREFERENCE;
                           G__define_var(-1, -1);
                           G__reftype = G__PARANORMAL;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "virtual")) {
                           G__virtual = 1;
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           break;
                        }
                        if (!strcmp(statement, "mutable")) {
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
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
                           G__fgetstream(statement, 0, ";");
                           int largestep = 0;
                           if (G__breaksignal) {
                              int ret = G__beforelargestep(statement, &iout, &largestep);
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
                           if (largestep) {
                              G__afterlargestep(&largestep);
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
                           int store_tagnum;
                           do {
                              G__FastAllocString oprbuf(G__ONELINE);
                              iout = strlen(statement);
                              c = G__fgetname(oprbuf, 0, "(");
                              switch (oprbuf[0]) {
                                 case '*':
                                 case '&':
                                    statement += oprbuf;
                                    break;
                                 default:
                                    statement += " ";
                                    statement += oprbuf;
                              }
                           }
                           while (c != '(');
                           iout = strlen(statement);
                           if (statement[iout-1] == ' ') {
                              --iout;
                           }
                           statement.Set(iout, 0);
                           result = G__string2type(statement + 9);
                           store_tagnum = G__tagnum;
                           G__var_type = result.type;
                           G__typenum = result.typenum;
                           G__tagnum = result.tagnum;
                           short store_constvar = G__constvar;
                           G__constvar = (short)result.obj.i; // see G__string2type
                           int store_reftype = G__reftype;
                           G__reftype = result.obj.reftype.reftype;
                           statement.Set(iout++, '(');
                           statement.Set(iout, 0);
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
                        if (!strcmp(statement, "namespace")) {
                           G__var_type = 'u';
                           G__define_struct('n');
                           // Reset the statement buffer.
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = -1;
                           if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                              return G__null;
                           }
                           break;
                        }
                        if (!strcmp(statement, "unsigned*")) {
                           G__var_type = 'I' - 1;
                           G__unsigned = -1;
                           G__define_var(-1, -1);
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
                           G__define_var(-1, -1);
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
                        statement = replace;
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
               statement.Set(iout++, c);
            }
            else {
               // -- We have reached the end of the statement.
               statement.Set(iout, 0);
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
                           statement[0] = 0;
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
                              statement[0] = 0;
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
                           statement[0] = 0;
                           iout = 0;
                           // Flag that any following whitespace does not trigger any semantic action.
                           spaceflag = 0;
                        }
                        break;
                  }
                  if (!strncmp(statement, "return\"", 7) || !strncmp(statement, "return'", 7)) {
                     result = G__return_value(statement + 6);
                     if (G__no_exec_compile) {
                        statement[0] = 0;
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
                     // We have an expression statement, evaluate the expression.
#ifdef G__ASM
                     if (G__asm_noverflow) {
                        // We are generating bytecode.
                        //--
                        // Take a possible breakpoint and clear the data stack
                        // and the structure offset stack.
                        G__asm_clear();
                     }
#endif // G__ASM
#ifdef G__ASM
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "\n!!!G__exec_statement: Before expression stmt, increment G__templevel %d --> %d  %s:%d\n"
                           , G__templevel
                           , G__templevel + 1
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
#endif // G__ASM
                     ++G__templevel;
                     result = G__getexpr(statement);
                     //
                     //  Destroy all temporaries created during expression evaluation.
                     //
#ifdef G__ASM
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "\n!!!G__exec_statement: Destroy temp objects after expression stmt, currently at G__templevel %d  %s:%d\n"
                           , G__templevel
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
#endif // G__ASM
                     G__free_tempobject();
#ifdef G__ASM
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "\n!!!G__exec_statement: After expression stmt, decrement G__templevel %d --> %d  %s:%d\n"
                           , G__templevel
                           , G__templevel - 1
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
#endif // G__ASM
                     --G__templevel;
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
               // Handle an assignment.
               statement.Set(iout, '=');
               c = G__fgetstream_new(statement, iout + 1, ";,{}");
               if ((c == '}') || (c == '{')) {
                  G__syntaxerror(statement);
                  --*mparen;
                  c = ';';
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
#ifdef G__ASM
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "\n!!!G__exec_statement: Before assignment expression, increment G__templevel %d --> %d  %s:%d\n"
                     , G__templevel
                     , G__templevel + 1
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
#endif // G__ASM
               ++G__templevel;
               result = G__getexpr(statement);
               if (c != ',') {
                  // Assignment expression is *not* part of a comma operator
                  // expression, so it must be the full expression.
                  //
                  //  Destroy all temporaries created during assignment expression evaluation.
                  //
#ifdef G__ASM
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "\n!!!G__exec_statement: Destroy temp objects after assignment expression, currently at G__templevel %d  %s:%d\n"
                        , G__templevel
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
#endif // G__ASM
                  G__free_tempobject();
               }
#ifdef G__ASM
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "\n!!!G__exec_statement: After assignment expression, decrement G__templevel %d --> %d  %s:%d\n"
                     , G__templevel
                     , G__templevel - 1
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
#endif // G__ASM
               --G__templevel;
               if (largestep) {
                  G__afterlargestep(&largestep);
               }
               if ((!*mparen && (c == ';')) || (G__return > G__RETURN_NORMAL)) {
                  return result;
               }
            }
            else if (G__prerun && !single_quote && !double_quote) {
               c = G__fignorestream(";,}");
               if ((c == '}') && *mparen) {
                  --*mparen;
               }
               // Reset the statement buffer.
               iout = 0;
               // Flag that any following whitespace does not trigger any semantic action.
               spaceflag = 0;
            }
            else {
               statement.Set(iout++, c);
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
            statement.Set(iout++, c);
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
            break;

         case '(':
            //fprintf(stderr, "\nG__exec_statement: Enter left parenthesis case.\n");
            statement.Set(iout++, c);
            statement.Set(iout, 0);
            if (single_quote || double_quote) {
               break;
            }
            //fprintf(stderr, "G__exec_statement: Saw '('. c: '%c' pr: %d ne: %d nec: %d ano: %d io: %d st: '%s'\n", c, G__prerun, G__no_exec, G__no_exec_compile, G__asm_noverflow, iout, statement);
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
                        // G__fgetstream(statement, 0, ")");
                        //
                        fseek(G__ifile.fp, -1, SEEK_CUR);
                        if (G__dispsource) {
                           G__disp_mask = 1;
                        }
                        G__fgetstream_new(statement, 0, ";");
                        result = G__return_value(statement);
                        if (G__no_exec_compile) {
                           statement[0] = 0;
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
                        if ((result.type == G__block_goto.type) && (result.ref == G__block_goto.ref)) {
                           int found = G__search_gotolabel(0, &start_pos, start_line, mparen);
                           // If found, continue parsing, the input file is now
                           // positioned immediately after the colon of the label.
                           // Otherwise, return and let our caller try to find it.
                           if (!found) {
                              // -- Not found, maybe our caller can find it.
                              return G__block_goto;
                           }
                        }
                        else if (result.type == G__block_break.type) {
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
                        if ((result.type == G__block_goto.type) && (result.ref == G__block_goto.ref)) {
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
                        c = G__fignorestream(")");
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
                        G__FastAllocString casepara(G__ONELINE);
                        casepara[0] = '(';
                        {
                           size_t lencasepara = 1;
                           c = G__fgetstream(casepara, lencasepara, ":");
                           if (c==')') {
                              lencasepara = strlen(casepara);
                              casepara.Resize(lencasepara + 2);
                              casepara[lencasepara] = ')';
                              ++lencasepara;
                              G__fgetstream(casepara, lencasepara, ":");
                           }
                           c = G__fgetc();
                           while (c == ':') {
                              casepara += "::";
                              lencasepara = strlen(casepara);
                              G__fgetstream(casepara, lencasepara, ":");
                              c = G__fgetc();
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
                           G__value result = G__exec_switch_case(casepara);
                           if (G__switch_searching) {
                              // -- We are searching for a matching case, return value of case expression.
                              return result;
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
                        if ((result.type == G__block_goto.type) && (result.ref == G__block_goto.ref)) {
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
                        if (result.type == G__block_break.type) {
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
                  // --
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
                  c = G__fgetstream_new(statement , iout, ")");
                  iout = strlen(statement);
                  statement.Resize(iout + 2);
                  statement[iout++] = c;
                  statement[iout] = 0;
                  // Skip any following whitespace.
                  c = G__fgetspace();
                  // if 'func(xxxxxx) \n    nextexpr'   macro
                  // if 'func(xxxxxx) ;'                func call
                  // if 'func(xxxxxx) operator xxxxx'   func call + operator
                  // if 'new (arena) type'              new operator with arena
                  if (!strncmp(statement, "new ", 4) || !strncmp(statement, "new(", 4)) {
                     // -- We have a new expression, either placement or parenthesized typename.
                     // Grab the rest of the line.
                     statement.Set(iout++, c);
                     c = G__fgetstream_template(statement, iout, ";");
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
                     int notcalled = G__exec_function(statement, &c, &iout, &largestep, &result);
                     if (notcalled) {
                        return G__null;
                     }
                  }
                  if ((!*mparen && (c == ';')) || (G__return > G__RETURN_NON)) {
                     return result;
                  }
                  // Reset the statement buffer.
                  iout = 0;
                  // Flag that any following whitespace does not trigger any semantic action.
                  spaceflag = 0;
               }
               else if (iout > 3) {
                  // -- Nope, just a parenthesized construct, accumulate it and keep going.
                  c = G__fgetstream_new(statement, iout, ")");
                  iout = strlen(statement);
                  statement.Set(iout++, c);
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
               G__FastAllocString casepara(G__ONELINE);
               casepara[0] = '(';
               {
                  size_t lencasepara = 1;
                  c = G__fgetstream(casepara, lencasepara, ":");
                  if (c==')') {
                     lencasepara = strlen(casepara);
                     casepara.Resize(lencasepara + 2);
                     casepara[lencasepara] = ')';
                     ++lencasepara;
                     G__fgetstream(casepara, lencasepara, ":");
                  }
                  c = G__fgetc();
                  while (c == ':') {
                     casepara += "::";
                     lencasepara = strlen(casepara);
                     G__fgetstream(casepara, lencasepara, ":");
                     c = G__fgetc();
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
                  G__value result = G__exec_switch_case(casepara);
                  if (G__switch_searching) {
                     // -- We are searching for a matching case, return value of case expression.
                     return result;
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
               c = G__fignorestream(")");
               if (c != ')') {
                  G__FastAllocString msg(70);
                  msg.Format("Error: Cannot find matching ')' for '(' on line %d.", paren_start_line);
                  G__genericerror(msg);
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
               G__constvar = G__VARIABLE;
               return G__start;
            }
            if (single_quote || double_quote) {
               statement.Set(iout++, c);
            }
            else {
               G__constvar = G__VARIABLE;
               G__static_alloc = 0;
               statement.Set(iout, 0);
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
                     G__var_type = 'u';
                     G__define_struct(statement[0]);  // Note: The argument is one of 'c', 'e', 's'.
                     if (!*mparen || (G__return > G__RETURN_NORMAL)) {
                        return G__null;
                     }
                     // Reset the statement buffer.
                     iout = 0;
                     // Flag that any following whitespace does not trigger any semantic action.
                     spaceflag = 0;
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
               statement.Set(iout++, c);
            }
            else {
               --*mparen;
               //fprintf(stderr, "G__exec_statement: seen '}': G__no_exec: %d mparen: %d\n", G__no_exec, *mparen);
               if (*mparen <= 0) {
                  if (iout && (G__globalcomp == G__NOLINK)) {
                     statement.Set(iout, 0);
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
               G__fgetstream_new(statement, 0, ";");
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
            statement.Set(iout++, c);
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
                     statement.Set(iout, 0);
                     G__strip_quotation(conststring);
                  }
               }
            }
            break;

         case '\'':
            if (!double_quote) {
               single_quote ^= 1;
            }
            statement.Set(iout++, c);
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
               statement.Set(iout++, c);
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
               statement.Set(iout++, c);
               // Flag that any following whitespace should trigger a semantic action.
               spaceflag |= 1;
               if (!double_quote && !single_quote) {
                  add_fake_space = 1;
               }
            }
            break;

         case '&':
            statement.Set(iout++, c);
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
            if (!double_quote && !single_quote) {
               add_fake_space = 1;
            }
            break;

         case ':':
            statement.Set(iout++, c);
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
            if (!double_quote && !single_quote) {
               statement.Set(iout, 0);
               int stat = G__label_access_scope(statement, &iout, &spaceflag, *mparen);
               if (stat) {
                  return G__null;
               }
            }
            break;

#ifdef G__TEMPLATECLASS
         case '<':
            statement.Set(iout, 0);
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
                  statement.Set(iout++, c);
                  c = G__fgetstream_template(statement, iout, ">");
                  G__ASSERT(c == '>');
                  iout = strlen(statement);
                  if (statement[iout-1] == '>') {
                     statement.Set(iout++, ' ');
                  }
                  statement.Set(iout++, c);
                  spaceflag = 1;
                  // Try to accept statements with no space between
                  // the closing '>' and the identifier. Ugly, though.
                  {
                     fpos_t xpos;
                     fgetpos(G__ifile.fp, &xpos);
                     c = G__fgetspace();
                     fsetpos(G__ifile.fp, &xpos);
                     if (isalpha(c) || (c == '_')) {
                        c = ' ';
                        goto read_again;
                     }
                  }
               }
               else {
                  statement.Set(iout++, c);
                  // Flag that any following whitespace should trigger a semantic action.
                  spaceflag |= 1;
               }
            }
            break;
#endif // G__TEMPLATECLASS

         case EOF:
            {
               statement.Set(iout, 0);
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
                  G__fprinterr(G__serr, "Report: Unrecognized string '%s' ignored", statement());
                  G__printlinenum();
               }
            }
            return G__null;

         case '\\':
            statement.Set(iout++, c);
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
            c = G__fgetc();
            // Continue to default case.

         default:
            //fprintf(stderr, "G__exec_statement: Enter default case.\n");
            // Make sure that the delimiters that have not been treated
            // in the switch statement do drop the discarded_space.
            if (discarded_space && (c != '[') && (c != ']') && iout && (statement[iout-1] != ':')) {
               // -- Since the character following a discarded space
               // is *not* a separator, we have to keep the space.
               statement.Set(iout++, ' ');
            }
            statement.Set(iout++, c);
            // Flag that any following whitespace should trigger a semantic action.
            spaceflag |= 1;
#ifdef G__MULTIBYTE
            if (G__IsDBCSLeadByte(c)) {
               c = G__fgetc();
               G__CheckDBCS2ndByte(c);
               statement.Set(iout++, c);
            }
#endif // G__MULTIBYTE
            break;
      }
      discarded_space = discard_space;
   }
}

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
void G__alloc_tempobject(int tagnum, int typenum)
{
   // Allocate a temporary interpreted object of class tagnum,
   // and insert it on the head of the temporary list.
   //
   // Note: We do not call the constructor.
   //
   G__ASSERT(tagnum != -1);
   if (G__xrefflag) {
      return;
   }
   // Create temp object buffer.
   {
      struct G__tempobject_list* store_p_tempbuf = G__p_tempbuf;
      G__p_tempbuf =
         (struct G__tempobject_list*) malloc(sizeof(struct G__tempobject_list));
      G__p_tempbuf->prev = store_p_tempbuf;
   }
   G__p_tempbuf->no_exec = G__no_exec_compile;
   G__p_tempbuf->cpplink = 0;
   G__p_tempbuf->level = G__templevel;
   // Create class object.
   G__p_tempbuf->obj.type = 'u';
   G__p_tempbuf->obj.tagnum = tagnum;
   G__p_tempbuf->obj.typenum = typenum;
   G__p_tempbuf->obj.obj.reftype.reftype = G__PARANORMAL;
   G__p_tempbuf->obj.isconst = 0;
   G__p_tempbuf->obj.obj.i = (long) malloc(G__struct.size[tagnum]);
   G__p_tempbuf->obj.ref = G__p_tempbuf->obj.obj.i;
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\nG__alloc_tempobject: no_exec: %d cpplink: %d (%s,%d,%d) 0x%lx level: %d  %s:%d\n"
         , G__p_tempbuf->no_exec
         , G__p_tempbuf->cpplink
         , G__struct.name[G__p_tempbuf->obj.tagnum]
         , G__p_tempbuf->obj.tagnum
         , G__p_tempbuf->obj.typenum
         , G__p_tempbuf->obj.obj.i
         , G__p_tempbuf->level
         , __FILE__
         , __LINE__
      );
      G__display_tempobject("After G__alloc_tempobject: ");
   }
#endif // G__ASM_DBG
#endif // G__ASM
   // --
}

//______________________________________________________________________________
void G__alloc_tempobject_val(G__value* val)
{
   // CINT7 compatible wrapper of G__alloc_tempobject, used from dictionaries
   G__alloc_tempobject(val->tagnum, val->typenum);
}

//______________________________________________________________________________
void G__store_tempobject(G__value reg)
{
   // Move the passed value into a new temporary object.
   if (G__xrefflag) {
      return;
   }
   // Create temporary object buffer.
   struct G__tempobject_list* store_p_tempbuf = G__p_tempbuf;
   G__p_tempbuf =
      (struct G__tempobject_list*) malloc(sizeof(struct G__tempobject_list));
   G__p_tempbuf->prev = store_p_tempbuf;
   G__p_tempbuf->no_exec = G__no_exec_compile;
   G__p_tempbuf->cpplink = 1;
   G__p_tempbuf->level = G__templevel;
   // Moved passed value into new temporary.
   G__p_tempbuf->obj = reg;
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\nG__store_tempobject: no_exec: %d cpplink: %d (%s,%d,%d) 0x%lx level: %d  %s:%d\n"
         , G__p_tempbuf->no_exec
         , G__p_tempbuf->cpplink
         , G__struct.name[G__p_tempbuf->obj.tagnum]
         , G__p_tempbuf->obj.tagnum
         , G__p_tempbuf->obj.typenum
         , G__p_tempbuf->obj.obj.i
         , G__p_tempbuf->level
         , __FILE__
         , __LINE__
      );
      G__display_tempobject("After G__store_tempobject: ");
   }
#endif // G__ASM_DBG
#endif // G__ASM
   // --
}

//______________________________________________________________________________
static int G__pop_tempobject_imp(bool delobj)
{
   // FIXME: Describe this function!
   //
   // Note: Used only by the following two functions:
   //
   //            G__pop_tempobject()
   //            G__pop_tempobject_nodel().
   //
   if (G__xrefflag) {
      return 0;
   }
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\nG__pop_tempobject_imp: delobj: %d no_exec: %d cpplink: %d (%s,%d,%d) 0x%lx level: %d  %s:%d\n"
         , (int) delobj
         , G__p_tempbuf->no_exec
         , G__p_tempbuf->cpplink
         , G__struct.name[G__p_tempbuf->obj.tagnum]
         , G__p_tempbuf->obj.tagnum
         , G__p_tempbuf->obj.typenum
         , G__p_tempbuf->obj.obj.i
         , G__p_tempbuf->level
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   if (delobj) {
      if (!G__p_tempbuf->cpplink && G__p_tempbuf->obj.obj.i) {
         free((void*) G__p_tempbuf->obj.obj.i);
      }
   }
   if (G__p_tempbuf->prev) {
      struct G__tempobject_list* store_p_tempbuf = G__p_tempbuf->prev;
      free((void*) G__p_tempbuf);
      G__p_tempbuf = store_p_tempbuf;
   }
   return 0;
}

//______________________________________________________________________________
int G__pop_tempobject()
{
   // -- FIXME: Describe this function!
   return G__pop_tempobject_imp(true);
}

//______________________________________________________________________________
int G__pop_tempobject_nodel()
{
   // -- FIXME: Describe this function!
   return G__pop_tempobject_imp(false);
}

//______________________________________________________________________________
void G__free_tempobject()
{
   // Destroy and free all temp objects created at G__templevel or greater.
   long store_struct_offset; /* used to be int */
   int store_tagnum;
   int store_return;
   if (
      G__xrefflag || // Generating a variable cross-reference, or
      (
         G__command_eval && // exec temp code from the cmd line, and
         (G__ifswitch != G__DOWHILE) // not in a do {} while (); construct.
      )
   ) {
      return;
   }
   if (!G__p_tempbuf->prev) {
      // Nothing to do.
      return;
   }
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__FastAllocString msg(G__ONELINE);
      msg.Format("Before G__free_tempobject: cur_level: %d ", G__templevel);
      G__display_tempobject(msg());
   }
#endif // G__ASM_DBG
#endif // G__ASM
   struct G__tempobject_list* cur = G__p_tempbuf;
   struct G__tempobject_list* previous = 0;
   while (cur->prev) {
      //fflush(stdout);
      //fprintf(stderr, "\nG__free_tempobject: previous: "
      //   "0x%lx\n", (long) previous);
      //fprintf(stderr,   "G__free_tempobject:      cur: "
      //   "0x%lx\n", (long) cur);
      if (cur->level < G__templevel) {
         // Keep this one.
         previous = cur;
         cur = cur->prev;
         continue;
      }
      //
      //  We found a temp object to delete.
      //
#ifdef G__ASM
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "\nG__free_tempobject: no_exec: %d cpplink: %d (%s,%d,%d) 0x%lx level: %d  %s:%d\n"
            , cur->no_exec
            , cur->cpplink
            , G__struct.name[cur->obj.tagnum]
            , cur->obj.tagnum
            , cur->obj.typenum
            , cur->obj.obj.i
            , cur->level
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
#endif // G__ASM
      //
      //  Remove this node from the list before calling
      //  the destructor, it may recursively call us again!
      //
      // If we are about to release head of the chain, update it.
      if (G__p_tempbuf == cur) {
         G__p_tempbuf = cur->prev;
         //fflush(stdout);
         //fprintf(stderr, "\nG__free_tempobject: new "
         //   "G__p_tempbuf: 0x%lx\n", (long) G__p_tempbuf);
      }
      // If we are removing from the middle of the list,
      // update the previous node's pointer.
      if (previous) {
         previous->prev = cur->prev;
      }
      //
      //  Call the destructor.
      //
      //  Note: This may result in this routine getting
      //        re-entered.  That is why we removed the
      //        node we are working on before making this
      //        call!
      //
#ifdef G__ASM
      if (G__asm_noverflow) {
         // We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "%3x,%3x: SETTEMP  %s:%d\n"
               , G__asm_cp
               , G__asm_dt
               , __FILE__
               , __LINE__
            );
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SETTEMP;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      store_struct_offset = G__store_struct_offset;
      G__store_struct_offset = cur->obj.obj.i;
      store_tagnum = G__tagnum;
      G__tagnum = cur->obj.tagnum;
      store_return = G__return;
      G__return = G__RETURN_NON;
      if (!cur->no_exec || G__no_exec_compile) {
         // --
#ifdef G__ASM
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "\n!!!Call temp object destructor: no_exec: %d cpplink: %d "
                 "(%s,%d,%d) 0x%lx level: %d destroylevel: %d\n"
               , cur->no_exec
               , cur->cpplink
               , G__struct.name[cur->obj.tagnum]
               , cur->obj.tagnum
               , cur->obj.typenum
               , cur->obj.obj.i
               , cur->level
               , G__templevel
            );
         }
#endif // G__ASM_DBG
#endif // G__ASM
         G__FastAllocString statement;
         statement.Format("~%s()", G__struct.name[G__tagnum]);
         int found = 0;
         G__getfunction(statement, &found, G__TRYDESTRUCTOR);
         // FIXME: We should give an error message if there was no destructor!
      }
      //
      //  Free the object storage.
      //
#ifdef G__ASM
      if (G__asm_noverflow) {
         // We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "%3x,%3x: FREETEMP  %s:%d\n"
               , G__asm_cp
               , G__asm_dt
               , __FILE__
               , __LINE__
            );
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__FREETEMP;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
      G__return = store_return;
      if (!cur->cpplink && cur->obj.obj.i) {
         // --
#ifdef G__ASM
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "\n!!!Free temp object: no_exec: %d cpplink: %d "
                 "(%s,%d,%d) 0x%lx level: %d destroylevel: %d\n"
               , cur->no_exec
               , cur->cpplink
               , G__struct.name[cur->obj.tagnum]
               , cur->obj.tagnum
               , cur->obj.typenum
               , cur->obj.obj.i
               , cur->level
               , G__templevel
            );
         }
#endif // G__ASM_DBG
#endif // G__ASM
         //fflush(stdout);
         //fprintf(stderr, "\nG__free_tempobject: Freeing "
         //   "object at: 0x%lx\n", (long) cur->obj.obj.i);
         free((void*) cur->obj.obj.i);
         cur->obj.obj.i = 0;
      }
      //
      //  Now free the list node.
      //
      //fflush(stdout);
      //fprintf(stderr, "\nG__free_tempobject: Freeing "
      //   "G__tempobject_list*: 0x%lx\n", (long) cur);
      free((void*) cur);
      //
      //  Restart the scan from the beginning, the temp list
      //  may have been modified by the destructor call and
      //  we have no other way to recover our iterator.
      //
      cur = G__p_tempbuf;
      previous = 0;
   }
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__FastAllocString msg(G__ONELINE);
      msg.Format("After G__free_tempobject: cur_level: %d  "
         "G__p_tempbuf: 0x%lx", G__templevel, (long) G__p_tempbuf);
      G__display_tempobject(msg());
   }
#endif // G__ASM_DBG
#endif // G__ASM
   // --
}

//______________________________________________________________________________
void G__settemplevel(int val)
{
   // -- FIXME: Describe this function!
   G__templevel += val;
}

//______________________________________________________________________________
void G__clearstack()
{
   // -- FIXME: Describe this function!
   int store_command_eval = G__command_eval;
   ++G__templevel;
   G__command_eval = 0;
   G__free_tempobject();
   --G__templevel;
   G__command_eval = store_command_eval;
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
