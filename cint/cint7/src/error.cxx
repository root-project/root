/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file error.c
 ************************************************************************
 * Description:
 *  Show error message
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

using namespace Cint::Internal;

int Cint::Internal::G__const_noerror = 0;

// Static functions.
#ifndef G__OLDIMPELMENTATION1174
#ifndef G__OLDIMPELMENTATION1186
static char* G__findrpos(char* s1, const char* s2);
#endif // G__OLDIMPELMENTATION1186
#endif // G__OLDIMPELMENTATION1174
#ifndef G__OLDIMPELMENTATION1174
static int G__splitmessage(char* item);
#endif // G__OLDIMPELMENTATION1174

// Cint internal functions.
namespace Cint {
namespace Internal {
void G__nosupport(const char* name);
void G__malloc_error(const char* varname);
void G__arrayindexerror(const ::Reflex::Member& var, const char* item, int index);
#ifdef G__ASM
int G__asm_execerr(const char* message, int num);
#endif // G__ASM
int G__assign_error(const char* item, G__value* pbuf);
int G__reference_error(const char* item);
int G__warnundefined(const char* item);
int G__unexpectedEOF(const char* message);
int G__shl_load_error(const char* shlname, char* message);
int G__getvariable_error(const char* item);
int G__referencetypeerror(const char* new_name);
int G__syntaxerror(const char* expr);
int G__parenthesiserror(const char* expression, const char* funcname);
int G__commenterror();
int G__changeconsterror(const char* item, const char* categ);
int G__pounderror();
int G__missingsemicolumn(const char* item);
} // namespace Internal
} // namespace Cint

// Functions in the C interface.
extern "C" int G__const_setnoerror();
extern "C" int G__const_resetnoerror();
extern "C" int G__const_whatnoerror();
extern "C" int G__printlinenum();
extern "C" int G__get_security_error();
extern "C" int G__genericerror(const char* message);
extern "C" void G__printerror(const char* funcname, int ipara, int paran);
#ifdef G__SECURITY
extern "C" int G__check_drange(int p, double low, double up, double d, G__value* result7, const char* funcname);
extern "C" int G__check_lrange(int p, long low, long up, long l, G__value* result7, const char* funcname);
extern "C" int G__check_type(int p, int t1, int t2, G__value* para, G__value* result7, const char* funcname);
extern "C" int G__check_nonull(int p, int t, G__value* para, G__value* result7, const char* funcname);
#endif // G__SECURITY

//______________________________________________________________________________
//
//  Static functions.
//

#ifndef G__OLDIMPELMENTATION1174
#ifndef G__OLDIMPELMENTATION1186
//______________________________________________________________________________
static char* G__findrpos(char* s1, const char* s2)
{
   if (!s1 || !s2) {
      return 0;
   }
   int i = strlen(s1);
   int s2len = strlen(s2);
   int nest = 0;
   int double_quote = 0;
   int single_quote = 0;
   char c = '\0';
   while (i--) {
      c = s1[i];
      switch (c) {
         case '[':
         case '(':
         case '{':
            if (!double_quote && !single_quote) {
               --nest;
            }
            break;
         case ']':
         case ')':
         case '}':
            if (!double_quote && !single_quote) {
               ++nest;
            }
            break;
      }
      if (!nest && !double_quote && !single_quote) {
         if (!strncmp(s1 + i, s2, s2len)) {
            return s1 + i;
         }
      }
   }
   return 0;
}
#endif // G__OLDIMPELMENTATION1186
#endif // G__OLDIMPELMENTATION1174

#ifndef G__OLDIMPELMENTATION1174
//______________________________________________________________________________
static int G__splitmessage(char* item)
{
   int stat = 0;
   char* dot;
   char* point;
   char* p;
   char* buf = (char*) malloc(strlen(item) + 1);
   strcpy(buf, item);
#ifndef G__OLDIMPELMENTATION1186
   dot = G__findrpos(buf, ".");
   point = G__findrpos(buf, "->");
#else // G__OLDIMPELMENTATION1186
   dot = strrchr(buf, '.');
   point = G__strrstr(buf, "->");
#endif // G__OLDIMPELMENTATION1186
   if (dot || point) {
      G__value result;
      if (!dot || (point && point > dot)) {
         p = point;
         *p = 0;
         p += 2;
      }
      else {
         p = dot;
         *p = 0;
         p += 1;
      }
      stat = 1;
      result = G__getexpr(buf);
      if (G__value_typenum(result)) {
         G__fprinterr(G__serr,
                      "Error: Failed to evaluate class member '%s' (%s)\n", p, item[0] == '$' ? item + 1 : item);
      }
      else {
         G__fprinterr(G__serr,
                      "Error: Failed to evaluate %s\n", item[0] == '$' ? item + 1 : item);
      }
   }
   free((void*) buf);
   return stat;
}
#endif // G__OLDIMPELMENTATION1174

//______________________________________________________________________________
//
//  Cint internal functions.
//

//______________________________________________________________________________
void Cint::Internal::G__nosupport(const char* name)
{
   // -- Print out error message for unsupported capability.
   G__fprinterr(G__serr, "Limitation: %s is not supported", name);
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
}

//______________________________________________________________________________
void Cint::Internal::G__malloc_error(const char* varname)
{
   G__fprinterr(G__serr, "Internal Error: malloc failed for %s", varname);
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__DANGEROUS;
#endif // G__SECURITY
   // --
}

//______________________________________________________________________________
void Cint::Internal::G__arrayindexerror(const ::Reflex::Member& var, const char* name, int index)
{
   G__fprinterr(G__serr, "Error: Array index out of range %s -> [%d] ", name, index);
   G__fprinterr(G__serr, " valid upto %s", var.Name().c_str());
   const int num_of_elements = G__get_varlabel(var, 1);
   const int stride = G__get_varlabel(var, 0);
   if (num_of_elements) {
      G__fprinterr(G__serr, "[%d]", (num_of_elements / stride) - 1);
   }
   const short num_of_dimensions = G__get_paran(var);
   for (int j = 2; j <= num_of_dimensions; ++j) {
      G__fprinterr(G__serr, "[%d]", G__get_varlabel(var, j) - 1);
   }
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   // --
}

#ifdef G__ASM
//______________________________________________________________________________
int Cint::Internal::G__asm_execerr(const char* message, int num)
{
   G__fprinterr(G__serr, "Loop Compile Internal Error: %s %d ", message, num);
   G__genericerror(0);
   G__asm_exec = 0;
   return 0;
}
#endif // G__ASM

//______________________________________________________________________________
int Cint::Internal::G__assign_error(const char* item, G__value* pbuf)
{
   if (!G__prerun) {
      if (G__value_typenum(*pbuf)) {
         G__fprinterr(G__serr, "Error: Incorrect assignment to %s, wrong type '%s'", item , G__value_typenum(*pbuf).Name(::Reflex::SCOPED).c_str());
      }
      else {
         G__fprinterr(G__serr, "Error: Incorrect assignment to %s ", item);
      }
      G__genericerror(0);
   }
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__reference_error(const char* item)
{
   G__fprinterr(G__serr, "Error: Incorrect referencing of %s ", item);
   G__genericerror(0);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__warnundefined(const char* item)
{
   if (G__prerun && G__static_alloc && G__func_now) {
      return 0;
   }
   if (G__no_exec_compile && !G__asm_noverflow) {
      return 0;
   }
   if (G__in_pause) {
      return 0;
   }
   if (G__asm_wholefunction & G__ASM_FUNC_COMPILE) {
      G__CHECK(G__SECURE_PAUSE, 1, G__pause());
      G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
   }
   else {
      G__StrBuf tmp_sb(G__ONELINE);
      char *tmp = tmp_sb;
      strcpy(tmp, item);

      if (
         !G__const_noerror
#ifndef G__OLDIMPELMENTATION1174
         && !G__splitmessage(tmp)
#endif // G__OLDIMPELMENTATION1174
      ) {

         char *p = strchr(tmp, '(');
         if (p) {
            p = G__strrstr(tmp, "::");
            if (p) {
               *p = 0;
               p += 2;
               G__fprinterr(G__serr, "Error: Function %s is not defined in %s ", p, tmp);
            }
            else {
               G__fprinterr(G__serr, "Error: Function %s is not defined in current scope ", (item[0] == '$') ? item + 1 : item);
            }
         }
         else {
           if (p) {
               *p = 0;
               p += 2;
               G__fprinterr(G__serr, "Error: Symbol %s is not defined in %s ", p, tmp);
            }
            else {
               G__fprinterr(G__serr, "Error: Symbol %s is not defined in current scope ", (item[0] == '$') ? item + 1 : item);
            }
         }
         G__genericerror(0);
      }
   }
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__unexpectedEOF(const char* message)
{
   G__eof = 2;
   G__fprinterr(G__serr, "Error: Unexpected end of file (%s)", message);
   G__genericerror(0);
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   if ((G__globalcomp != G__NOLINK) && (G__steptrace || G__stepover)) {
      while (!G__pause()) {
         ; // Intentionally empty
      }
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__shl_load_error(const char* shlname, const char* message)
{
   G__fprinterr(G__serr, "%s: Failed to load Dynamic link library %s\n", message, shlname);
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__getvariable_error(const char* item)
{
   G__fprinterr(G__serr, "Error: G__getvariable: expression %s", item);
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__referencetypeerror(const char* new_name)
{
   G__fprinterr(G__serr, "Error: Can't take address for reference type %s", new_name);
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__syntaxerror(const char* expr)
{
   G__fprinterr(G__serr, "Syntax Error: %s", expr);
   G__genericerror(0);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__parenthesiserror(const char* expression, const char* funcname)
{
   G__fprinterr(G__serr, "Syntax error: %s: Parenthesis or quotation unmatch %s", funcname, expression);
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__commenterror()
{
   G__fprinterr(G__serr, "Syntax error: Unexpected '/' Comment?");
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__changeconsterror(const char* item, const char* categ)
{
   if (G__dispmsg >= G__DISPWARN) {
      G__fprinterr(G__serr, "Warning: Re-initialization %s %s", categ, item);
      G__printlinenum();
   }
   if (!G__prerun) {
      G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
      G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__pounderror()
{
   // -- #error xxx
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   fgets(buf, G__ONELINE, G__ifile.fp);
   char* p = strchr(buf, '\n');
   if (p) {
      *p = '\0';
   }
   p = strchr(buf, '\r');
   if (p) {
      *p = '\0';
   }
   G__fprinterr(G__serr, "#error %s\n", buf);
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__missingsemicolumn(const char* item)
{
   G__fprinterr(G__serr, "Syntax Error: %s Maybe missing ';'", item);
   G__genericerror(0);
   return 0;
}

//______________________________________________________________________________
extern "C" void G__printerror(const char* funcname, int ipara, int paran)
{
   if (G__dispmsg >= G__DISPWARN) {
      G__fprinterr(G__serr, "Warning: %s() expects %d parameters, %d parameters given", funcname, ipara, paran);
      G__printlinenum();
   }
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   // --
}

#ifdef G__SECURITY
//______________________________________________________________________________
extern "C" int G__check_drange(int p, double low, double up, double d, G__value* result7, const char* funcname)
{
   // -- Check for double.
   if ((d < low) || (d > up)) {
      G__fprinterr(G__serr, "Error: %s param[%d]=%g up:%g low:%g out of range", funcname, p, d, up, low);
      G__genericerror(0);
      *result7 = G__null;
      return 1;
   }
   return 0;
}
#endif // G__SECURITY

#ifdef G__SECURITY
//______________________________________________________________________________
extern "C" int G__check_lrange(int p, long low, long up, long l, G__value* result7, const char* funcname)
{
   // -- Check for long.
   if ((l < low) || (l > up)) {
      G__fprinterr(G__serr, "Error: %s param[%d]=%ld up:%ld low:%ld out of range", funcname, p, l, up, low);
      G__genericerror(0);
      *result7 = G__null;
      return 1;
   }
   return 0;
}
#endif // G__SECURITY

#ifdef G__SECURITY
//______________________________________________________________________________
extern "C" int G__check_type(int p, int t1, int t2, G__value* para, G__value* result7, const char* funcname)
{
   // -- Check for ???
   if ((G__get_type(*para) != t1) && (G__get_type(*para) != t2)) {
      G__fprinterr(G__serr, "Error: %s param[%d] type mismatch", funcname, p);
      G__genericerror(0);
      *result7 = G__null;
      return 1;
   }
   return 0;
}
#endif // G__SECURITY

#ifdef G__SECURITY
//______________________________________________________________________________
extern "C" int G__check_nonull(int p, int t, G__value* para, G__value* result7, const char* funcname)
{
   // -- Check for null pointer.
   long l = G__int(*para);
   if (!l) {
      G__fprinterr(G__serr, "Error: %s param[%d]=%ld must not be 0", funcname, p, l);
      G__genericerror(0);
      *result7 = G__null;
      return 1;
   }
   int pt = G__get_type(*para);
   if ((t != pt) && !((t == 'C') && (pt == 'T')) && (t != 'Y')) {
      G__fprinterr(G__serr, "Error: %s parameter mismatch param[%d] %c %c", funcname, p, t, pt);
      G__genericerror(0);
      *result7 = G__null;
      return 1;
   }
   return 0;
}
#endif // G__SECURITY

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
extern "C" int G__const_setnoerror()
{
   G__const_noerror = 1;
   return 1;
}

//______________________________________________________________________________
extern "C" int G__const_resetnoerror()
{
   G__const_noerror = 0;
   return 0;
}

//______________________________________________________________________________
extern "C" int G__const_whatnoerror()
{
   return G__const_noerror;
}

//______________________________________________________________________________
extern "C" int G__printlinenum()
{
   const char* format = " FILE:%s LINE:%d\n";
#ifdef G__WIN32
   // make error msg Visual Studio compatible
   format = " %s(%d)\n";
#elif defined(G__ROOT)
   // make error msg GCC compatible
   format = " %s:%d:\n";
#endif // G__WIN32, G__ROOT
   G__fprinterr(G__serr, format, G__stripfilename(G__ifile.name), G__ifile.line_number);
   return 0;
}

//______________________________________________________________________________
extern "C" int G__get_security_error()
{
   return G__security_error;
}

//______________________________________________________________________________
extern "C" int G__genericerror(const char* message)
{
   if (G__xrefflag) {
      return 1;
   }
   if (!G__const_noerror && (G__asm_wholefunction == G__ASM_FUNC_NOP)) {
      if (message) {
         G__fprinterr(G__serr, "%s", message);
      }
      G__printlinenum();
      G__storelasterror();
   }
   G__CHECK(G__SECURE_PAUSE, 1, G__pause());
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   if (G__aterror) {
      int store_return = G__return;
      G__return = G__RETURN_NON;
      G__p2f_void_void((void*) G__aterror);
      G__return = store_return;
   }
   return 0;
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
