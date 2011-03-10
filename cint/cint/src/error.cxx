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
#include "bc_exec.h"

extern "C" {

int G__const_noerror = 0;

// Static functions.
#ifndef G__OLDIMPELMENTATION1174
#ifndef G__OLDIMPELMENTATION1186
static const char* G__findrpos(const char* s1, const char* s2);
#endif // G__OLDIMPELMENTATION1186
#endif // G__OLDIMPELMENTATION1174
#ifndef G__OLDIMPELMENTATION1174
static int G__splitmessage(char* item);
#endif // G__OLDIMPELMENTATION1174

// External functions.
void G__nosupport(const char* name);
void G__malloc_error(const char* varname);
void G__arrayindexerror(int varid, struct G__var_array* var, const char* name, int index);
#ifdef G__ASM
int G__asm_execerr(const char* message, int num);
#endif // G__ASM
int G__assign_using_null_pointer_error(const char* item);
int G__assign_error(const char* item, G__value* pbuf);
int G__reference_error(const char* item);
int G__warnundefined(const char* item);
int G__unexpectedEOF(const char* message);
int G__shl_load_error(const char* shlname, const char* message);
int G__getvariable_error(const char* item);
int G__referencetypeerror(const char* new_name);
int G__syntaxerror(const char* expr);
int G__parenthesiserror(const char* expression, const char* funcname);
int G__commenterror();
int G__changeconsterror(const char* item, const char* categ);
int G__pounderror();
int G__missingsemicolumn(const char* item);
void G__printerror(const char* funcname, int ipara, int paran);
#ifdef G__SECURITY
int G__check_drange(int p, double low, double up, double d, G__value* result7, const char* funcname);
int G__check_lrange(int p, long low, long up, long l, G__value* result7, const char* funcname);
int G__check_type(int p, int t1, int t2, G__value* para, G__value* result7, const char* funcname);
int G__check_nonull(int p, int t, G__value* para, G__value* result7, const char* funcname);
#endif // G__SECURITY

// Functions in the C interface.
int G__const_setnoerror();
int G__const_resetnoerror();
int G__const_whatnoerror();
int G__printlinenum();
int G__get_security_error();
int G__genericerror(const char* message);

//______________________________________________________________________________
//
//  Static functions.
//

#ifndef G__OLDIMPELMENTATION1174
#ifndef G__OLDIMPELMENTATION1186
//______________________________________________________________________________
static const char* G__findrpos(const char* s1, const char* s2)
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
   strcpy(buf, item); // Okay we allocated enough space.
#ifndef G__OLDIMPELMENTATION1186
   dot = (char*)G__findrpos(buf, ".");
   point = (char*)G__findrpos(buf, "->");
#else // G__OLDIMPELMENTATION1186
   dot = strrchr(buf, '.');
   point = (char*)G__strrstr(buf, "->");
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
      if (result.type) {
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
//  External functions.
//

//______________________________________________________________________________
void G__nosupport(const char* name)
{
   // -- Print out error message for unsupported capability.
   G__fprinterr(G__serr, "Limitation: %s is not supported", name);
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
}

//______________________________________________________________________________
void G__malloc_error(const char* varname)
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
void G__arrayindexerror(int varid, struct G__var_array* var, const char* name, int index)
{
   G__fprinterr(G__serr, "Error: Array index out of range %s -> [%d] ", name, index);
   G__fprinterr(G__serr, " valid upto %s", var->varnamebuf[varid]);
   const int num_of_elements = var->varlabel[varid][1];
   const int stride = var->varlabel[varid][0];
   if (num_of_elements) {
      G__fprinterr(G__serr, "[%d]", (num_of_elements / stride) - 1);
   }
   const short num_of_dimensions = var->paran[varid];
   for (int j = 2; j <= num_of_dimensions; ++j) {
      G__fprinterr(G__serr, "[%d]", var->varlabel[varid][j] - 1);
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
int G__asm_execerr(const char* message, int num)
{
   G__fprinterr(G__serr, "Loop Compile Internal Error: %s %d ", message, num);
   G__genericerror(0);
   G__asm_exec = 0;
   return 0;
}
#endif // G__ASM

//______________________________________________________________________________
int G__assign_using_null_pointer_error(const char* item)
{
   if (!G__prerun) {
      G__fprinterr(
           G__serr
         , "Error: Attempted assignment using %s, but the value of %s is NULL."
         , item
         , item
      );
      G__genericerror(0);
   }
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int G__assign_error(const char* item, G__value* pbuf)
{
   if (!G__prerun) {
      if (pbuf->type) {
         G__fprinterr(G__serr, "Error: Incorrect assignment to %s, wrong type '%s'", item, G__type2string(pbuf->type, pbuf->tagnum, pbuf->typenum, pbuf->obj.reftype.reftype, 0));
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
int G__reference_error(const char* item)
{
   G__fprinterr(G__serr, "Error: Incorrect referencing of %s ", item);
   G__genericerror(0);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int G__warnundefined(const char* item)
{
   if (G__prerun && G__static_alloc && G__func_now >= 0) return 0;
   if (G__no_exec_compile && 0 == G__asm_noverflow) return 0;
   if (G__in_pause) return 0;
   if (
      !G__cintv6 &&
      G__ASM_FUNC_COMPILE&G__asm_wholefunction) {
      G__CHECK(G__SECURE_PAUSE, 1, G__pause());
      G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
   }
   else {
      if (0 == G__const_noerror
#ifndef G__OLDIMPELMENTATION1174
            && !G__splitmessage((char*)item)
#endif // G__OLDIMPELMENTATION1174
         ) {
         char *p = (char*)strchr(item, '(');
         if (p) {
            G__FastAllocString tmp(item);
            p = (char*)G__strrstr(tmp, "::");
            if (p) {
               *p = 0;
               p += 2;
               G__fprinterr(G__serr,
                            "Error: Function %s is not defined in %s ", p, tmp());
            }
            else {
               G__fprinterr(G__serr,
                            "Error: Function %s is not defined in current scope "
                            , item[0] == '$' ? item + 1 : item);
            }
         }
         else {
            G__FastAllocString tmp(item);
            G__fprinterr(G__serr,
                         "Error: Symbol %s is not defined in current scope ", item[0] == '$' ? item + 1 : item);
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
int G__unexpectedEOF(const char* message)
{
   G__eof = 2;
   G__fprinterr(G__serr, "Error: Unexpected end of file (%s)", message);
   G__genericerror(0);
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   if (G__NOLINK != G__globalcomp && (G__steptrace || G__stepover))
      while (0 == G__pause()) ;
   return 0;
}

//______________________________________________________________________________
int G__shl_load_error(const char* shlname, const char* message)
{
   G__fprinterr(G__serr, "%s: Failed to load Dynamic link library %s\n", message, shlname);
   // No: not being able to load a library is not generally an interpreter error.
   //G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
   // #ifdef G__SECURITY
   // G__security_error = G__RECOVERABLE;
   // #endif // G__SECURITY
   G__return = G__RETURN_EXIT1;
   return 0;
}

//______________________________________________________________________________
int G__getvariable_error(const char* item)
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
int G__referencetypeerror(const char* new_name)
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
int G__syntaxerror(const char* expr)
{
   G__fprinterr(G__serr, "Syntax Error: %s", expr);
   G__genericerror(0);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int G__parenthesiserror(const char* expression, const char* funcname)
{
   G__fprinterr(G__serr, "Syntax error: %s: Parenthesis or quotation unmatch %s"
                , funcname , expression);
   G__printlinenum();
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int G__commenterror()
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
int G__changeconsterror(const char* item, const char* categ)
{
   if (G__dispmsg >= G__DISPWARN) {
      G__fprinterr(G__serr, "Warning: Re-initialization %s %s", categ, item);
      G__printlinenum();
   }
   if (0 == G__prerun) {
      G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
      G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   }
   return 0;
}

//______________________________________________________________________________
int G__pounderror()
{
   // -- #error xxx
   G__FastAllocString buf(G__ONELINE);
   if (fgets(buf, G__ONELINE, G__ifile.fp)) {
      char *p = strchr(buf, '\n');
      if (p) *p = '\0';
      p = strchr(buf, '\r');
      if (p) *p = '\0';
      G__fprinterr(G__serr, "#error %s\n", buf());
   } else {
      G__fprinterr(G__serr, "#error <can not read original file %s>\n", G__ifile.name[0] ? G__ifile.name : "temporary file");
   }
   G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return = G__RETURN_EXIT1);
#ifdef G__SECURITY
   G__security_error = G__RECOVERABLE;
#endif // G__SECURITY
   return 0;
}

//______________________________________________________________________________
int G__missingsemicolumn(const char* item)
{
   G__fprinterr(G__serr, "Syntax Error: %s Maybe missing ';'", item);
   G__genericerror(0);
   return 0;
}

//______________________________________________________________________________
void G__printerror(const char* funcname, int ipara, int paran)
{
   if (G__dispmsg >= G__DISPWARN) {
      G__fprinterr(G__serr
                   , "Warning: %s() expects %d parameters, %d parameters given"
                   , funcname, ipara, paran);
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
int G__check_drange(int p, double low, double up, double d, G__value* result7, const char* funcname)
{
   // -- Check for double.
   if (d < low || up < d) {
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
int G__check_lrange(int p, long low, long up, long l, G__value* result7, const char* funcname)
{
   // -- Check for long.
   if (l < low || up < l) {
      G__fprinterr(G__serr, "Error: %s param[%d]=%ld up:%ld low:%ld out of range"
                   , funcname, p, l, up, low);
      G__genericerror(0);
      *result7 = G__null;
      return 1;
   }
   return 0;
}
#endif // G__SECURITY

#ifdef G__SECURITY
//______________________________________________________________________________
int G__check_type(int p, int t1, int t2, G__value* para, G__value* result7, const char* funcname)
{
   // -- Check for ???
   if (para->type != t1 && para->type != t2) {
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
int G__check_nonull(int p, int t, G__value* para, G__value* result7, const char* funcname)
{
   // -- Check for nil pointer.
   long l;
   l = G__int(*para);
   if (0 == l) {
      G__fprinterr(G__serr, "Error: %s param[%d]=%ld must not be 0", funcname, p, l);
      G__genericerror(0);
      *result7 = G__null;
      return 1;
   }
   else if (t != para->type) {
      if ('Y' != t) {
         G__fprinterr(G__serr, "Error: %s parameter mismatch param[%d] %c %c", funcname, p, t, para->type);
         G__genericerror(0);
         *result7 = G__null;
         return 1;
      }
      return 0;
   }
   return 0;
}
#endif // G__SECURITY

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
int G__const_setnoerror()
{
   G__const_noerror = 1;
   return 1;
}

//______________________________________________________________________________
int G__const_resetnoerror()
{
   G__const_noerror = 0;
   return 0;
}

//______________________________________________________________________________
int G__const_whatnoerror()
{
   return G__const_noerror;
}

//______________________________________________________________________________
int G__printlinenum()
{
   const char* format = " FILE:%s LINE:%d\n";
#ifdef G__WIN32
   // make error msg Visual Studio compatible
   format = " %s(%d)\n";
#elif defined(G__ROOT)
   // make error msg GCC compatible
   format = " %s:%d:\n";
#endif // VISUAL_CPLUSPLUS, G__ROOT
   G__fprinterr(G__serr, format, G__stripfilename(G__ifile.name), G__ifile.line_number);
   return 0;
}

//______________________________________________________________________________
int G__get_security_error()
{
   return G__security_error;
}

//______________________________________________________________________________
int G__genericerror(const char* message)
{
   if (G__xrefflag) {
      return 1;
   }
   if (!G__const_noerror && (G__cintv6 || (G__asm_wholefunction == G__ASM_FUNC_NOP))) {
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
   if (G__cintv6) {
      if (G__cintv6 & G__BC_COMPILEERROR) {
         G__bc_throw_compile_error();
      }
      if (G__cintv6 & G__BC_RUNTIMEERROR) {
         G__bc_throw_runtime_error();
      }
   }
   return 0;
}

} // extern "C"

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
