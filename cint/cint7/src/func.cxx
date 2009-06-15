/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file func.c
 ************************************************************************
 * Description:
 *  Function call
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "Reflex/Builder/TypeBuilder.h"
#include "common.h"
#include "Dict.h"

#if defined(G__WIN32)
#include <windows.h>
extern "C" int getopt(int argc, char** argv, char* optlist);
#elif defined(G__POSIX)
#include <unistd.h> // already included in G__ci.h
#endif

#include <cstdlib>
#include <cstring>
#include <cstdio>

extern "C" int optind;
extern "C" char* optarg;

using namespace Cint::Internal;
using namespace std;

//______________________________________________________________________________
//
// Function Directory.
//

// Static Functions
static char* G__catparam(G__param* libp, int catn, const char* connect);
static void G__gen_PUSHSTROS_SETSTROS();
static int G__dispvalue(FILE* fp, G__value* buf);
static int G__bytecodedebugmode(int mode);
static int G__getbytecodedebugmode();
static int G__checkscanfarg(const char* fname, G__param* libp, int n);
static void G__getindexedvalue(G__value* result3, char* cindex);
#ifdef G__PTR2MEMFUNC
static G__value G__pointer2memberfunction(char* parameter0, char* parameter1, int* known3);
#endif // G__PTR2MEMFUNC
static G__value G__pointerReference(char* item, G__param* libp, int* known3);
static int G__additional_parenthesis(G__value* presult, G__param* libp);
static G__value G__operatorfunction(G__value* presult, const char* item, int* known3, char* result7, char* funcname);
static G__value G__getfunction_libp(const char* item, char* funcname, G__param* libp, int* known3, int memfunc_flag);
static void G__va_start(G__value ap);
static G__value G__va_arg(G__value ap);
static void G__va_end(G__value ap);
static int G__defined(char* tname);
static void G__printf_error();
static void G__sprintformatll(char* result, const char* fmt, void* p, char* buf);
static void G__sprintformatull(char* result, const char* fmt, void* p, char* buf);
static void G__sprintformatld(char* result, const char* fmt, void* p, char* buf);

// Internal Functions
namespace Cint {
namespace Internal {
int G__explicit_fundamental_typeconv(char* funcname, int hash, G__param* libp, G__value* presult3);
void G__gen_addstros(long addstros);
bool G__rename_templatefunc(std::string& funcname);
int G__special_func(G__value* result7, char* funcname, G__param* libp, int hash);
int G__library_func(G__value* result7, char* funcname, G__param* libp, int hash);
char* G__charformatter(int ifmt, G__param* libp, char* result);
int G__compiled_func_cxx(G__value* result7, char* funcname, struct G__param* libp, int hash);
} // namespace Internal
} // namespace Cint

// Functions in the C interface.
extern "C" {
int G__getexitcode();
int G__get_return(int* exitval);
void G__storelasterror();
char* G__lasterror_filename();
int G__lasterror_linenum();
void G__p2f_void_void(void* p2f);
void G__set_atpause(void (*p2f)());
void G__set_aterror(void (*p2f)());
void G__set_emergencycallback(void (*p2f)());
G__value G__getfunction(const char* item, int* known3, int memfunc_flag);
int G__tracemode(int tracemode);
int G__stepmode(int stepmode);
int G__gettracemode();
int G__getstepmode();
int G__optimizemode(int optimizemode);
int G__getoptimizemode();
} // extern "C"

//______________________________________________________________________________
//
//  Types.
//

typedef struct
{
   G__param* libp;
   int ip;
} G__va_list;

//______________________________________________________________________________
//
//  Static Global Variables.
//

static int G__exitcode = 0;

//______________________________________________________________________________
static char* G__catparam(G__param* libp, int catn, const char* connect)
{
   // Concatenate parameter string to libp->parameter[0] and return.
   //
   // "B<int"   "double"   "5>"     =>    "B<int,double,5>"
   //
   // B<int\0
   //      ^ => p
   char* p = libp->parameter[0] + strlen(libp->parameter[0]);
   int lenconnect = strlen(connect);
   for (int i = 1; i < catn; ++i) {
      strcpy(p, connect);
      p += lenconnect;
      strcpy(p, libp->parameter[i]);
      p += strlen(libp->parameter[i]);
   }
   return libp->parameter[0];
}

//______________________________________________________________________________
static void G__gen_PUSHSTROS_SETSTROS()
{
   // --
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x: PUSHSTROS\n", G__asm_cp);
      G__fprinterr(G__serr, "%3x: SETSTROS\n", G__asm_cp + 1);
   }
#endif
   G__asm_inst[G__asm_cp] = G__PUSHSTROS;
   G__asm_inst[G__asm_cp+1] = G__SETSTROS;
   G__inc_cp_asm(2, 0);
}

//______________________________________________________________________________
static int G__dispvalue(FILE* fp, G__value* buf)
{
   if (buf) {
      fprintf(fp
              , "\nd=%g i=%ld type=%s\n"
              , buf->obj.d
              , buf->obj.i
              , G__value_typenum(*buf).Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   }
   return(1);
}

//______________________________________________________________________________
extern "C" int G__getexitcode()
{
   int exitcode = G__exitcode;
   G__exitcode = 0 ;
   return(exitcode);
}

//______________________________________________________________________________
extern "C" int G__get_return(int* exitval)
{
   if (exitval) {
      *exitval = G__getexitcode();
   }
   return G__return;
}

//______________________________________________________________________________
static int G__bytecodedebugmode(int mode)
{
   G__asm_dbg = mode;
   return G__asm_dbg;
}

//______________________________________________________________________________
static int G__getbytecodedebugmode()
{
   return G__asm_dbg;
}

//______________________________________________________________________________
extern "C" void G__storelasterror()
{
   G__lasterrorpos = G__ifile;
}

//______________________________________________________________________________
extern "C" char* G__lasterror_filename()
{
   return G__lasterrorpos.name;
}

//______________________________________________________________________________
extern "C" int G__lasterror_linenum()
{
   return G__lasterrorpos.line_number;
}

//______________________________________________________________________________
static int G__checkscanfarg(const char* fname, G__param* libp, int n)
{
   int result = 0;
   while (n < libp->paran) {
      if (!G__value_typenum(libp->para[n]).IsPointer()) {
         G__fprinterr(G__serr, "Error: %s arg%d not a pointer", fname, n);
         G__genericerror((char*)NULL);
         ++result;
      }
      if (0 == libp->para[n].obj.i) {
         G__fprinterr(G__serr, "Error: %s arg%d is NULL", fname, n);
         G__genericerror((char*)NULL);
         ++result;
      }
      ++n;
   }
   return result;
}

//______________________________________________________________________________
extern "C" void G__p2f_void_void(void* p2f)
{
   // Pointer to function evaluation function
   switch (G__isinterpretedp2f(p2f)) {
      case G__INTERPRETEDFUNC: {
         G__StrBuf buf_sb(G__ONELINE);
         char *buf = buf_sb;
         char *fname;
         fname = G__p2f2funcname(p2f);
         sprintf(buf, "%s()", fname);
         if (G__asm_dbg) G__fprinterr(G__serr, "(*p2f)() %s interpreted\n", buf);
         G__calc_internal(buf);
      }
      break;
      case G__BYTECODEFUNC: {
         struct G__param param;
         G__value result;
#ifdef G__ANSI
         int (*ifm)(G__value*, char*, struct G__param*, int);
         ifm = (int (*)(G__value*, char*, struct G__param*, int))G__exec_bytecode;
#else
         int (*ifm)();
         ifm = (int (*)())G__exec_bytecode;
#endif
         param.paran = 0;
         if (G__asm_dbg) G__fprinterr(G__serr, "(*p2f)() bytecode\n");
         (*ifm)(&result, (char*)p2f, &param, 0);
      }
      break;
      case G__COMPILEDINTERFACEMETHOD: {
         struct G__param param;
         G__value result;
#ifdef G__ANSI
         int (*ifm)(G__value*, char*, struct G__param*, int);
         ifm = (int (*)(G__value*, char*, struct G__param*, int))p2f;
#else
         int (*ifm)();
         ifm = (int (*)())p2f;
#endif
         param.paran = 0;
         if (G__asm_dbg) G__fprinterr(G__serr, "(*p2f)() compiled interface\n");
         (*ifm)(&result, (char*)NULL, &param, 0);
      }
      break;
      case G__COMPILEDTRUEFUNC:
      case G__UNKNOWNFUNC: {
         void (*tp2f)();
         tp2f = (void (*)())p2f;
         if (G__asm_dbg) G__fprinterr(G__serr, "(*p2f)() compiled true p2f\n");
         (*tp2f)();
      }
      break;
   }
}

//______________________________________________________________________________
extern "C" void G__set_atpause(void (*p2f)())
{
   G__atpause = p2f;
}

//______________________________________________________________________________
extern "C" void G__set_aterror(void (*p2f)())
{
   G__aterror = p2f;
}

//______________________________________________________________________________
extern "C" void G__set_emergencycallback(void (*p2f)())
{
   G__emergencycallback = p2f;
}

//______________________________________________________________________________
static void G__getindexedvalue(G__value* result3, char* cindex)
{
   int size;
   int index;
   G__StrBuf sindex_sb(G__ONELINE);
   char* sindex = sindex_sb;
   strcpy(sindex, cindex);
   char* p = strstr(sindex, "][");
   if (p) {
      p[1] = 0;
      G__getindexedvalue(result3, sindex);
      p = strstr(cindex, "][");
      strcpy(sindex, p + 1);
   }
   int len = strlen(sindex);
   sindex[len-1] = '\0';
   if (G__get_type(G__value_typenum(*result3)) == 'u') {
      G__param fpara;
#ifdef G__ASM
      if (G__asm_noverflow) {
         G__gen_PUSHSTROS_SETSTROS();
      }
#endif // G__ASM
      fpara.paran = 1;
      fpara.para[0] = G__getexpr(sindex + 1);
      G__parenthesisovldobj(result3, result3, "operator[]", &fpara, 1);
      return;
   }
   index = G__int(G__getexpr(sindex + 1));
   size = G__sizeof(result3);
#ifdef G__ASM
   if (G__asm_noverflow) { // We are generating bytecode.
      //
      //  Size arithmetic is done by OP2 in bytecode execution.
      //
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: OP2  '+'  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__OP2;
      G__asm_inst[G__asm_cp+1] = (long) '+';
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   result3->obj.i += (size * index);
   *result3 = G__tovalue(*result3);
}

//______________________________________________________________________________
int Cint::Internal::G__explicit_fundamental_typeconv(char* funcname, int hash, G__param* libp, G__value* presult3)
{
   int flag = 0;

#ifndef G__OLDIMPLEMENTATION491
   /*
   if('u'==presult3->type && -1!=presult3->tagnum) {
   }
   */
#endif

   switch (hash) {
      case 3:
         if (strcmp(funcname, "int") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(int));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(int*)presult3->ref = (int)presult3->obj.i;
            flag = 1;
         }
         break;
      case 4:
         if (strcmp(funcname, "char") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(char));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(char*)presult3->ref = (char)presult3->obj.i;
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "long") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(long));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         else if (strcmp(funcname, "int*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(int)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         else if (strcmp(funcname, "bool") == 0
                 ) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(bool));
            presult3->obj.i = G__int(libp->para[0]) ? 1 : 0;
            if (presult3->ref)
               *(unsigned char*)presult3->ref = (unsigned char)presult3->obj.i ? 1 : 0;
            flag = 1;
         }
         break;
      case 5:
         if (strcmp(funcname, "short") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(short));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(short*)presult3->ref = (short)presult3->obj.i;
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "float") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(float));
            presult3->obj.d = G__double(libp->para[0]);
            if (presult3->ref) *(float*)presult3->ref = (float)presult3->obj.d;
            flag = 1;
         }
         else if (strcmp(funcname, "char*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(char)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "long*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(long)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         else if (strcmp(funcname, "void*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(void)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         break;
      case 6:
         if (strcmp(funcname, "double") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(double));
            presult3->obj.d = G__double(libp->para[0]);
            if (presult3->ref) *(double*)presult3->ref = (double)presult3->obj.d;
            flag = 1;
         }
         else if (strcmp(funcname, "short*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(short)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "float*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(float)));
            presult3->obj.d = G__double(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         break;
      case 7:
         if (strcmp(funcname, "double*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(double)));
            presult3->obj.d = G__double(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         break;
      case 8:
         if (strcmp(funcname, "unsigned") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned int));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
            flag = 1;
            break;
         }
         if (strcmp(funcname, "longlong") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(G__int64));
            presult3->obj.ll = G__Longlong(libp->para[0]);
            if (presult3->ref) *(G__int64*)presult3->ref = presult3->obj.ll;
            flag = 1;
         }
         break;
      case 9:
         if (strcmp(funcname, "unsigned*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(unsigned int)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
            break;
         }
         if (strcmp(funcname, "longlong*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(G__int64)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
            break;
         }
         if (strcmp(funcname, "long long") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(G__int64));
            presult3->obj.ll = G__Longlong(libp->para[0]);
            if (presult3->ref) *(G__int64*)presult3->ref = presult3->obj.ll;
            flag = 1;
         }
         break;
      case 10:
         if (strcmp(funcname, "longdouble") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(long double));
            presult3->obj.ld = G__Longdouble(libp->para[0]);
            if (presult3->ref) *(long double*)presult3->ref = presult3->obj.ld;
            flag = 1;
         }
         break;
      case 11:
         if (strcmp(funcname, "unsignedint") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned int));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
            flag = 1;
         }
         break;
      case 12:
         if (strcmp(funcname, "unsignedchar") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned char));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned char*)presult3->ref = (unsigned char)presult3->obj.i;
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "unsignedlong") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned long));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned long*)presult3->ref = (unsigned long)presult3->obj.i;
            flag = 1;
         }
         else if (strcmp(funcname, "unsigned int") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned int));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
            flag = 1;
         }
         break;
      case 13:
         if (strcmp(funcname, "unsignedshort") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned short));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned short*)presult3->ref = (unsigned short)presult3->obj.i;
            flag = 1;
         }
         else if (strcmp(funcname, "unsigned char") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned char));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned char*)presult3->ref = (unsigned char)presult3->obj.i;
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "unsigned long") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned long));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned long*)presult3->ref = (unsigned long)presult3->obj.i;
            flag = 1;
         }
         else if (strcmp(funcname, "unsigned int*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(unsigned int)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         break;
      case 14:
         if (strcmp(funcname, "unsigned short") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(unsigned short));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(unsigned short*)presult3->ref = (unsigned short)presult3->obj.i;
            flag = 1;
         }
         else if (strcmp(funcname, "unsigned char*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(unsigned char)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "unsigned long*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(unsigned long)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         break;
      case 15:
         if (strcmp(funcname, "unsigned short*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(unsigned short)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         break;
      case 16:
         if (strcmp(funcname, "unsignedlonglong") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(G__uint64));
            presult3->obj.ull = G__ULonglong(libp->para[0]);
            if (presult3->ref) *(G__uint64*)presult3->ref = presult3->obj.ull;
            flag = 1;
         }
         break;
      case 17:
         if (strcmp(funcname, "unsignedlonglong*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(G__uint64)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         break;
      case 18:
         if (strcmp(funcname, "unsigned long long") == 0) {
            G__value_typenum(*presult3) = Reflex::Type::ByTypeInfo(typeid(G__uint64));
            presult3->obj.ull = G__ULonglong(libp->para[0]);
            if (presult3->ref) *(G__uint64*)presult3->ref = presult3->obj.ull;
            flag = 1;
         }
         break;
      case 19:
         if (strcmp(funcname, "unsigned long long*") == 0) {
            G__value_typenum(*presult3) = Reflex::PointerBuilder(Reflex::Type::ByTypeInfo(typeid(G__uint64)));
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
            flag = 1;
         }
         break;
   }

   if (flag) {
      // Note Cint 5 code was reseting the typedef value (*presult3.typenum)
#ifdef G__ASM
      if (G__asm_noverflow) {
#ifdef G__ASM_DBG
         if (G__asm_dbg && G__asm_noverflow) {
            G__fprinterr(G__serr, "%3x,%3x: CAST to %s  %s:%d\n", G__asm_cp, G__asm_dt, G__value_typenum(*presult3).Name().c_str(), __FILE__, __LINE__);
         }
#endif
         G__asm_inst[G__asm_cp] = G__CAST;
         *(reinterpret_cast<Reflex::Type*>(&G__asm_inst[G__asm_cp+1])) = G__value_typenum(*presult3);
         // REMOVED: G__asm_inst[G__asm_cp+4]=G__PARANORMAL;
         G__inc_cp_asm(5, 0);
      }
#endif /* ASM */
   }
#ifndef G_OLDIMPLEMENTATION1128
   if (flag && (G__value_typenum(libp->para[0]).IsClass()
                || G__value_typenum(libp->para[0]).IsUnion()
                || G__value_typenum(libp->para[0]).IsEnum())) {
      int xtype = G__get_type(G__value_typenum(*presult3));
      int xreftype = 0;
      int xisconst = 0;
      *presult3 = libp->para[0];
      G__fundamental_conversion_operator(xtype, -1, ::Reflex::Type(), xreftype, xisconst, presult3);
   }
#endif
   return(flag);
}

//______________________________________________________________________________
void Cint::Internal::G__gen_addstros(long addstros)
{
   // --
#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg)
         G__fprinterr(G__serr, "%3x: ADDSTROS %d\n" , G__asm_cp, addstros);
#endif
      G__asm_inst[G__asm_cp] = G__ADDSTROS;
      G__asm_inst[G__asm_cp+1] = addstros;
      G__inc_cp_asm(2, 0);
   }
#endif
   // --
}

#ifdef G__PTR2MEMFUNC
//______________________________________________________________________________
static G__value G__pointer2memberfunction(char* parameter0, char* parameter1, int* known3)
{
   G__StrBuf buf_sb(G__LONGLINE);
   char *buf = buf_sb;
   G__StrBuf buf2_sb(G__ONELINE);
   char *buf2 = buf2_sb;
   G__StrBuf expr_sb(G__LONGLINE);
   char *expr = expr_sb;
   char* mem;
   G__value res;
   const char* opx;

   strcpy(buf, parameter0);

   if ((mem = strstr(buf, ".*"))) {
      *mem = 0;
      mem += 2;
      opx = ".";
   }
   else if ((mem = strstr(buf, "->*"))) {
      *mem = 0;
      mem += 3;
      opx = "->";
   }
   else {
      opx = "";
   }

   res = G__getexpr(mem);
   if (!G__get_type(res)) {
      G__fprinterr(G__serr, "Error: Pointer to member function %s not found"
                   , parameter0);
      G__genericerror((char*)NULL);
      return(G__null);
   }

   if (!res.obj.i || !*(char**)res.obj.i) {
      G__fprinterr(G__serr, "Error: Pointer to member function %s is NULL", parameter0);
      G__genericerror((char*)NULL);
      return(G__null);
   }

   /* For the time being, pointer to member function can only be handed as
    * function name */
   strcpy(buf2, *(char**)res.obj.i);

   sprintf(expr, "%s%s%s%s", buf, opx, buf2, parameter1);

   G__abortbytecode();
   return(G__getvariable(expr, known3, Reflex::Scope::GlobalScope(), G__p_local));
}
#endif // G__PTR2MEMFUNC

//______________________________________________________________________________
static G__value G__pointerReference(char* item, G__param* libp, int* known3)
{
   ::Reflex::Scope store_tagnum = G__tagnum;
   ::Reflex::Type store_typenum = G__typenum;
   char* store_struct_offset = G__store_struct_offset;
   G__value result3 = G__getexpr(item);
   if (!G__value_typenum(result3)) {
      return G__null;
   }
   *known3 = 1;
   if ((libp->paran == 2) && strstr(libp->parameter[1], "][")) {
      G__StrBuf arg_sb(G__ONELINE);
      char *arg = arg_sb;
      char* p = arg;
      strcpy(p, libp->parameter[1]);
      int i = 1;
      while (*p) {
         int k = 0;
         if (*p == '[') {
            ++p;
         }
         while (*p && (*p != ']')) {
            libp->parameter[i][k++] = *p++;
         }
         libp->parameter[i][k++] = 0;
         if (*p == ']') {
            ++p;
         }
         ++i;
      }
      libp->paran = i;
   }
   for (int i = 1; i < libp->paran; ++i) {
      G__StrBuf arg_sb(G__ONELINE);
      char *arg = arg_sb;
      strcpy(arg, libp->parameter[i]);
      if (arg[0] == '[') {
         int j = 0;
         while (arg[++j] && (arg[j] != ']')) {
            arg[j-1] = arg[j];
         }
         arg[j-1] = 0;
      }
      ::Reflex::Type rtype = G__value_typenum(result3).FinalType();
      if (G__get_type(G__value_typenum(result3)) == 'u') {
         // -- This is operator[] overloading.
         G__StrBuf expr_sb(G__ONELINE);
         char *expr = expr_sb;
         //
         //  Set member function environment.
         //
         //G__tagnum = G__Dict::GetDict().GetScope(result3.tagnum);
         G__tagnum = G__value_typenum(result3).RawType(); // this should be the class containing the op[]
         G__typenum = G__value_typenum(result3);
         G__store_struct_offset = (char*) result3.obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__PUSHSTROS;
            G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         //
         //  Call operator[].
         //
         *known3 = 0;
         sprintf(expr, "operator[](%s)", arg);
         result3 = G__getfunction(expr, known3, G__CALLMEMFUNC);
         //
         //  Restore environment.
         //
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         // --
      }
      else if (rtype.IsPointer()) {
         G__value varg = G__getexpr(arg);
         G__bstore('+', varg, &result3);
         result3 = G__tovalue(result3);
      }
      else {
         G__genericerror("Error: Incorrect use of operator[]");
         return G__null;
      }
   }
   return result3;
}

//______________________________________________________________________________
static int G__additional_parenthesis(G__value* presult, G__param* libp)
{
   ::Reflex::Scope store_tagnum = G__tagnum;
   char* store_struct_offset = G__store_struct_offset;
   if (G__get_tagnum(G__value_typenum(*presult)) == -1) {
      return 0;
   }
   G__tagnum = G__value_typenum(*presult).RawType();
   G__store_struct_offset = (char*) presult->obj.i;
   G__StrBuf buf_sb(G__LONGLINE);
   char* buf = buf_sb;
   sprintf(buf, "operator()%s", libp->parameter[1]);
   int known = 0;
   *presult = G__getfunction(buf, &known, G__CALLMEMFUNC);
   G__tagnum = store_tagnum;
   G__store_struct_offset = store_struct_offset;
   return known;
}

//______________________________________________________________________________
bool Cint::Internal::G__rename_templatefunc(std::string& funcname)
{
   // return true if the funcname was modified.

   char *ptmplt ;
   ptmplt = (char*)strchr(funcname.c_str(), '<');
   if (ptmplt) {
      *ptmplt = 0;
      if (G__defined_templatefunc(funcname.c_str())) {
         *ptmplt = 0;
      }
      else {
         *ptmplt = '<';
         ptmplt = (char*)0;
      }
   }
   if (ptmplt) {
      G__StrBuf funcname2_sb(G__LONGLINE);
      char *funcname2 = funcname2_sb;
      G__StrBuf buf_sb(G__ONELINE);
      char *buf = buf_sb;
      char buf2[20];
      ::Reflex::Type typenum;
      int tagnum, len;
      int ip = 1;
      int c;
      strcpy(funcname2, funcname.c_str());
      strcat(funcname2, "<");
      do {
         c = G__getstream_template(ptmplt, &ip, buf, ",>");
         len = strlen(buf) - 1;
         while ('*' == buf[len] || '&' == buf[len]) --len;
         ++len;
         if (buf[len]) {
            strcpy(buf2, buf + len);
            buf[len] = 0;
         }
         else buf2[0] = 0;
         typenum = G__find_typedef(buf);
         if (typenum) {
            strcpy(buf, typenum.Name(::Reflex::SCOPED).c_str());
         }
         else {
            tagnum = G__defined_tagname(buf, 1);
            if (-1 != tagnum) strcpy(buf, G__fulltagname(tagnum, 1));
         }
         strcat(buf, buf2);
         strcat(funcname2, buf);
         if (funcname2[strlen(funcname2)-1] == '>' && c == '>') {
            buf2[0] = ' ';
            buf2[1] = c;
            buf2[2] = 0;
         }
         else {
            buf2[0] = c;
            buf2[1] = 0;
         }
         strcat(funcname2, buf2);
      }
      while (c != '>');
      funcname = funcname2;
      return true;
   }
   return false;
}

//______________________________________________________________________________
static G__value G__operatorfunction(G__value* presult, const char* item, int* known3, char* result7, char* funcname)
{
   G__value result3 = G__null;
   struct G__param fpara;
   int ig15 = 0;
   int ig35 = 0;
   int nest = 0;
   int double_quote = 0, single_quote = 0;
   int lenitem = strlen(item);
   int overflowflag = 0;
   int castflag = 0;
   int base1 = 0;
   int nindex = 0;
   int itmp;

   fpara.paran = 0;
   G__value_typenum(fpara.para[0]) = Reflex::Type();

   /* Get Parenthesis */

   /******************************************************
    * this if statement should be always true,
    * should be able to omit.
    ******************************************************/
   if (ig15 < lenitem) {

      /*****************************************************
       * scan '(param1,param2,param3)'
       *****************************************************/
      while (ig15 < lenitem) {
         int tmpltnest = 0;

         /*************************************************
          * scan one parameter upto 'param,' or 'param)'
          * by reading upto ',' or ')'
          *************************************************/
         ig35 = 0;
         nest = 0;
         single_quote = 0;
         double_quote = 0;
         while ((((item[ig15] != ',') && (item[ig15] != ')')) ||
                 (nest > 0) ||
                 (tmpltnest > 0) ||
                 (single_quote > 0) || (double_quote > 0)) && (ig15 < lenitem)) {
            switch (item[ig15]) {
               case '"' : /* double quote */
                  if (single_quote == 0) double_quote ^= 1;
                  break;
               case '\'' : /* single quote */
                  if (double_quote == 0) single_quote ^= 1;
                  break;
               case '(':
               case '[':
               case '{':
                  if ((double_quote == 0) && (single_quote == 0)) nest++;
                  break;
               case ')':
               case ']':
               case '}':
                  if ((double_quote == 0) && (single_quote == 0)) nest--;
                  break;
               case '\\':
                  result7[ig35++] = item[ig15++];
                  break;
               case '<':
                  if (double_quote == 0 && single_quote == 0) {
                     result7[ig35] = 0;
                     if (0 == strcmp(result7, "operator") ||
                           tmpltnest ||
                           G__defined_templateclass(result7)) ++tmpltnest;
                  }
                  break;
               case '>':
                  if (double_quote == 0 && single_quote == 0) {
                     if (tmpltnest) --tmpltnest;
                  }
                  break;
            }
            result7[ig35++] = item[ig15++];
            if (ig35 >= G__ONELINE - 1) {
               if (result7[0] == '"') {
                  G__value bufv;
                  G__StrBuf bufx_sb(G__LONGLINE);
                  char *bufx = bufx_sb;
                  strncpy(bufx, result7, G__ONELINE - 1);
                  while ((((item[ig15] != ',') && (item[ig15] != ')')) ||
                          (nest > 0) || (single_quote > 0) ||
                          (double_quote > 0)) && (ig15 < lenitem)) {
                     switch (item[ig15]) {
                        case '"' : /* double quote */
                           if (single_quote == 0) double_quote ^= 1;
                           break;
                        case '\'' : /* single quote */
                           if (double_quote == 0) single_quote ^= 1;
                           break;
                        case '(':
                        case '[':
                        case '{':
                           if ((double_quote == 0) && (single_quote == 0)) nest++;
                           break;
                        case ')':
                        case ']':
                        case '}':
                           if ((double_quote == 0) && (single_quote == 0)) nest--;
                           break;
                        case '\\':
                           bufx[ig35++] = item[ig15++];
                           break;
                     }
                     bufx[ig35++] = item[ig15++];
                     if (ig35 >= G__LONGLINE - 1) {
                        G__genericerror("Limitation: Too long function argument");
                        return(G__null);
                     }
                  }
                  bufx[ig35] = 0;
                  bufv = G__strip_quotation(bufx);
                  sprintf(result7, "(char*)(%ld)", bufv.obj.i);
                  ig35 = strlen(result7) + 1;
                  break;
               }
               else if (ig35 > G__LONGLINE - 1) {
                  G__fprinterr(G__serr,
                               "Limitation: length of one function argument be less than %d"
                               , G__LONGLINE);
                  G__genericerror((char*)NULL);
                  G__fprinterr(G__serr, "Use temp variable as workaround.\n");
                  *known3 = 1;
                  return(G__null);
               }
               else {
                  overflowflag = 1;
               }
            }
         }
         /*************************************************
          * if ')' is found at the middle of expression,
          * this should be casting or pointer to function
          *
          *  v                    v            <-- this makes
          *  (type)expression  or (*p_func)();    castflag=1
          *       ^                       ^    <-- this makes
          *                                       castflag=2
          *************************************************/
         if ((item[ig15] == ')') && (ig15 < lenitem - 1)) {
            if (1 == castflag) {
               if (('-' == item[ig15+1] && '>' == item[ig15+2]) || '.' == item[ig15+1])
                  castflag = 3;
               else                                       castflag = 2;
            }
            else if (('-' == item[ig15+1] && '>' == item[ig15+2]) || '.' == item[ig15+1]) {
               castflag = 3;
               base1 = ig15 + 1;
            }
            else if (item[ig15+1] == '[') {
               nindex = fpara.paran + 1;
            }
            else if (funcname[0] && isalnum(item[ig15+1])) {
               G__fprinterr(G__serr, "Error: %s  Syntax error?", item);
               /* G__genericerror((char*)NULL); , avoid risk of side-effect */
               G__printlinenum();
            }
            else {
               ++ig15;
               result7[ig35] = '\0';
               strcpy(fpara.parameter[fpara.paran], result7);
               if (ig35) fpara.parameter[++fpara.paran][0] = '\0';
               for (itmp = 0;itmp < fpara.paran;itmp++) {
                  fpara.para[itmp] = G__getexpr(fpara.parameter[itmp]);
               }
               if (G__parenthesisovldobj(&result3, presult, "operator()"
                                         , &fpara, G__TRYNORMAL)) {
                  *known3 = 1;
                  return(G__operatorfunction(&result3, item + ig15 + 1, known3
                                             , result7, funcname));
               }
            }
         }

         /*************************************************
          * set null char to parameter list buffer.
          *************************************************/
         ig15++;
         result7[ig35] = '\0';
         if (ig35 < G__ONELINE) {
            strcpy(fpara.parameter[fpara.paran], result7);
         }
         else {
            strcpy(fpara.parameter[fpara.paran], "@");
            fpara.para[fpara.paran] = G__getexpr(result7);
         }
         if (ig35) fpara.parameter[++fpara.paran][0] = '\0';
      }
   }

#ifdef G__ASM
   if (G__asm_noverflow) G__gen_PUSHSTROS_SETSTROS();
#endif

   for (itmp = 0;itmp < fpara.paran;itmp++) {
      fpara.para[itmp] = G__getexpr(fpara.parameter[itmp]);
   }
   G__parenthesisovldobj(&result3, presult, "operator()", &fpara, 1);

   return(result3);
}

//______________________________________________________________________________
static G__value G__getfunction_libp(const char* item, char* funcname, G__param* libp, int* known3, int memfunc_flag)
{
   G__value result3 = G__null;
   G__StrBuf result7_sb(G__LONGLINE);
   char* result7 = result7_sb;
   int ipara;
   static G__param* p2ffpara = 0;
   int hash;
   int hash_2;
   int funcmatch;
   int classhash;
   ::Reflex::Scope store_tagnum = G__tagnum;
   int store_exec_memberfunc;
   int store_asm_noverflow;
   ::Reflex::Scope store_memberfunc_tagnum = G__memberfunc_tagnum;
   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   int tempstore;
   const char* pfparam;
   int nindex = 0;
   int oprp = 0;
   int store_cp_asm = 0;
   store_exec_memberfunc = G__exec_memberfunc;
   store_asm_noverflow = G__asm_noverflow;
   char* store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   *known3 = 1;
   //
   //  Check for a qualified function name.
   //
   //       ::f()
   //       A::B::f()
   //
   //  Note:
   //
   //    G__exec_memberfunc restored at return memfunc_flag is local,
   //    there should be no problem modifying these variables.
   //    store_struct_offset and store_tagnum are only used in the
   //    explicit type conversion section.  It is OK to use them here
   //    independently.
   //
   char* store_struct_offset = G__store_struct_offset;
   int intTagNum = G__get_tagnum(G__tagnum);
   {
      int which_scope = G__scopeoperator(funcname, &hash, &G__store_struct_offset, &intTagNum);
      G__tagnum = G__Dict::GetDict().GetScope(intTagNum); // might have been changed by G__scopeoperator
      switch (which_scope) {
         case G__GLOBALSCOPE:
            G__exec_memberfunc = 0;
            memfunc_flag = G__TRYNORMAL;
            G__def_tagnum = ::Reflex::Scope();
            G__tagdefining = ::Reflex::Scope();
            break;
         case G__CLASSSCOPE:
            G__exec_memberfunc = 1;
            memfunc_flag = G__CALLSTATICMEMFUNC;
            G__def_tagnum = ::Reflex::Scope();
            G__tagdefining = ::Reflex::Scope();
            G__memberfunc_tagnum = G__tagnum;
            break;
         default:
            G__hash(funcname, hash, hash_2); // FIXME: not in G__get_function???
            break;
      }
   }
   ::Reflex::Type typedf; // = hash_2; // or so it seems from the previous code! // FIXME: not in G__get_function???
#ifdef G__DUMPFILE
   if (G__dumpfile && !G__no_exec_compile) {
      //
      //  Dump that a function is called.
      //
      for (ipara = 0; ipara < G__dumpspace; ++ipara) {
         fprintf(G__dumpfile, " ");
      }
      fprintf(G__dumpfile, "%s(", funcname);
      for (ipara = 1; ipara <= libp->paran; ++ipara) {
         if (ipara != 1) {
            fprintf(G__dumpfile, ",");
         }
         G__valuemonitor(libp->para[ipara-1], result7);
         fprintf(G__dumpfile, "%s", result7);
      }
      fprintf(G__dumpfile, ");/*%s %d,%p %p*/\n", G__ifile.name, G__ifile.line_number, store_struct_offset, G__store_struct_offset);
      G__dumpspace += 3;

   }
#endif // G__DUMPFILE
   //
   //  Perform overload resolution.
   //
   //  G__EXACT = 1
   //  G__PROMOTION = 2
   //  G__STDCONV = 3
   //  G__USERCONV = 4
   //
   for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
      //
      //  Search for interpreted member function.
      //
      //  G__exec_memberfunc         ==>  memfunc();
      //  memfunc_flag!=G__TRYNORMAL ==>  a.memfunc();
      //
      if (G__exec_memberfunc || (memfunc_flag != G__TRYNORMAL)) {
         ::Reflex::Scope local_tagnum = G__tagnum;
         if (G__exec_memberfunc && (G__get_tagnum(G__tagnum) == -1)) {
            local_tagnum = G__memberfunc_tagnum;
         }
         if (G__get_tagnum(G__tagnum) != -1) {
            G__incsetup_memfunc(G__tagnum);
         }
         if (G__get_tagnum(local_tagnum) != -1) {
            int ret = G__interpret_func(&result3, funcname, libp, hash, local_tagnum, funcmatch, memfunc_flag);
            if (ret == 1) {
               // --
#ifdef G__DUMPFILE
               if (G__dumpfile && !G__no_exec_compile) {
                  G__dumpspace -= 3;
                  for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                     fprintf(G__dumpfile, " ");
                  }
                  G__valuemonitor(result3, result7);
                  fprintf(G__dumpfile, "/* return(inp) %s.%s()=%s*/\n", G__struct.name[G__get_tagnum(G__tagnum)], funcname, result7);
               }
#endif // G__DUMPFILE
               if (G__store_struct_offset != store_struct_offset) {
                  G__gen_addstros(store_struct_offset - G__store_struct_offset);
               }
               G__store_struct_offset = store_struct_offset;
               G__tagnum = store_tagnum;
               G__def_tagnum = store_def_tagnum;
               G__tagdefining = store_tagdefining;
               G__exec_memberfunc = store_exec_memberfunc;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
               G__memberfunc_struct_offset = store_memberfunc_struct_offset;
               G__setclassdebugcond(G__get_tagnum(G__memberfunc_tagnum), 0);
               if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
                  G__getindexedvalue(&result3, libp->parameter[nindex]);
               }
               if (oprp) {
                  *known3 = G__additional_parenthesis(&result3, libp);
               }
               return result3;
            }
         }
      }
      //
      //  If searching only member function.
      //
      if (memfunc_flag && (G__store_struct_offset || (memfunc_flag != G__CALLSTATICMEMFUNC))) {
         //
         //  If member function is called with a qualified name,
         //  then don't examine global functions.
         //
         //  There are 2 cases:
         //
         //                                      G__exec_memberfunc
         //    obj.memfunc();                            1
         //    X::memfunc();                             1
         //     X();              constructor            2
         //    ~X();              destructor             2
         //
         //  If G__exec_memberfunc == 2, don't display error message.
         //
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (funcmatch != G__USERCONV) {
            continue;
         }
         if (memfunc_flag == G__TRYDESTRUCTOR) {
            // destructor for base class and class members
#ifdef G__ASM
#ifdef G__SECURITY
            store_asm_noverflow = G__asm_noverflow;
            if (G__security & G__SECURE_GARBAGECOLLECTION) {
               G__abortbytecode();
            }
#endif // G__SECURITY
#endif // G__ASM
#ifdef G__VIRTUALBASE
            if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] != G__CPPLINK) {
               G__basedestructor();
            }
#else // G__VIRTUALBASE
            G__basedestructor();
#endif // G__VIRTUALBASE
#ifdef G__ASM
#ifdef G__SECURITY
            G__asm_noverflow = store_asm_noverflow;
#endif // G__SECURITY
#endif // G__ASM
            // --
         }
         else {
            switch (memfunc_flag) {
               case G__CALLCONSTRUCTOR:
               case G__TRYCONSTRUCTOR:
               case G__TRYIMPLICITCONSTRUCTOR:
                  // constructor for base class and class members default constructor only.
#ifdef G__VIRTUALBASE
                  if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] != G__CPPLINK) {
                     G__baseconstructor(0, 0);
                  }
#else // G__VIRTUALBASE
                  G__baseconstructor(0, 0);
#endif // G__VIRTUALBASE
                  // --
            }
         }
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         *known3 = 0;
         switch (memfunc_flag) {
            case G__CALLMEMFUNC:
               {
                  int ret = G__parenthesisovld(&result3, funcname, libp, G__CALLMEMFUNC);
                  if (ret) {
                     *known3 = 1;
                     if (G__store_struct_offset != store_struct_offset) {
                        G__gen_addstros(store_struct_offset - G__store_struct_offset);
                     }
                     G__store_struct_offset = store_struct_offset;
                     G__tagnum = store_tagnum;
                     G__def_tagnum = store_def_tagnum;
                     G__tagdefining = store_tagdefining;
                     if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
                        G__getindexedvalue(&result3, libp->parameter[nindex]);
                     }
                     if (oprp) {
                        *known3 = G__additional_parenthesis(&result3, libp);
                     }
                     return result3;
                  }
               }
               if (funcname[0] == '~') {
                  *known3 = 1;
                  return G__null;
               }
               { // FIXME: This whole block is not in G__get_function.
                  /******************************************************************
                   * Search template function
                   ******************************************************************/
                  G__exec_memberfunc = 1;
                  G__memberfunc_tagnum = G__tagnum;
                  G__memberfunc_struct_offset = G__store_struct_offset;
                  if ((funcmatch == G__EXACT) || (funcmatch == G__USERCONV)) {
                     int ret = G__templatefunc(&result3, funcname, libp, hash, funcmatch);
                     if (ret == 1) {
                        // --
#ifdef G__DUMPFILE
                        if (G__dumpfile && !G__no_exec_compile) {
                           G__dumpspace -= 3;
                           for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                              fprintf(G__dumpfile, " ");
                           }
                           G__valuemonitor(result3, result7);
                           fprintf(G__dumpfile , "/* return(lib) %s()=%s */\n", funcname, result7);
                        }
#endif // G__DUMPFILE
                        G__exec_memberfunc = store_exec_memberfunc;
                        G__memberfunc_tagnum = store_memberfunc_tagnum;
                        G__memberfunc_struct_offset = store_memberfunc_struct_offset;
                        if (oprp) {
                           *known3 = G__additional_parenthesis(&result3, libp);
                        }
                        else {
                           *known3 = 1;
                        }
                        return result3;
                     }
                  }
                  G__exec_memberfunc = store_exec_memberfunc;
                  G__memberfunc_tagnum = store_memberfunc_tagnum;
                  G__memberfunc_struct_offset = store_memberfunc_struct_offset;
               }
               // NOTE: Intentionally fallthrough!
            case G__CALLCONSTRUCTOR:
               if (G__globalcomp < G__NOLINK) { // If generating a dictionary, stop here.
                  break;
               }
               if (G__asm_noverflow || !G__no_exec_compile) {
                  if (!G__const_noerror) {
                     G__fprinterr(G__serr, "Error: Cannot call %s::%s in current scope (1)  %s:%d", G__struct.name[G__get_tagnum(G__tagnum)], item, __FILE__, __LINE__);
                  }
                  G__genericerror(0);
               }
               store_exec_memberfunc = G__exec_memberfunc;
               G__exec_memberfunc = 1;
               if (!G__const_noerror && (!G__no_exec_compile || G__asm_noverflow)) {
                  G__fprinterr(G__serr, "Possible candidates are...\n");
                  {
                     G__StrBuf itemtmp_sb(G__LONGLINE);
                     char* itemtmp = itemtmp_sb;
                     sprintf(itemtmp, "%s::%s", G__struct.name[G__get_tagnum(G__tagnum)], funcname);
                     G__display_proto_pretty(G__serr, itemtmp, 1);
                  }
               }
               G__exec_memberfunc = store_exec_memberfunc;
         }
#ifdef G__DUMPFILE
         if (G__dumpfile && !G__no_exec_compile) {
            G__dumpspace -= 3;
         }
#endif // G__DUMPFILE
         if (G__store_struct_offset != store_struct_offset) {
            G__gen_addstros(store_struct_offset - G__store_struct_offset);
         }
         G__store_struct_offset = store_struct_offset;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         if ( // Copy constructor not found.
            libp->paran &&
            (G__get_type(G__value_typenum(libp->para[0])) == 'u') &&
            (
               (memfunc_flag == G__TRYCONSTRUCTOR) ||
               (memfunc_flag == G__TRYIMPLICITCONSTRUCTOR)
            )
         ) { // Copy constructor not found.
            G__tagnum = store_tagnum;
            return libp->para[0];
         }
         result3 = G__null;
         G__value_typenum(result3) = G__tagnum;
         G__tagnum = store_tagnum;
         return result3; // FIXME: In cint5 this is just return G__null, result3 is not modified.
      }
      //
      //  Check for a global function.
      //
      tempstore = G__exec_memberfunc;
      G__exec_memberfunc = 0;
      { // FIXME: This is if (memfunc_flag != G__CALLSTATICMEMFUNC) in G__getfunction.
         int ret = G__interpret_func(&result3, funcname, libp, hash, G__p_ifunc, funcmatch, G__TRYNORMAL);
         if (ret == 1) {
            // --
   #ifdef G__DUMPFILE
            if (G__dumpfile && !G__no_exec_compile) {
               G__dumpspace -= 3;
               for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                  fprintf(G__dumpfile, " ");
               }
               G__valuemonitor(result3, result7);
               fprintf(G__dumpfile, "/* return(inp) %s()=%s*/\n", funcname, result7);
            }
   #endif // G__DUMPFILE
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;
            G__setclassdebugcond(G__get_tagnum(G__memberfunc_tagnum), 0);
            if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
               G__getindexedvalue(&result3, libp->parameter[nindex]);
            }
            if (oprp) {
               *known3 = G__additional_parenthesis(&result3, libp);
            }
            return result3;
         }
      }
      G__exec_memberfunc = tempstore;
      if (
         (funcmatch == G__PROMOTION) ||
         (funcmatch == G__STDCONV)
      ) {
         continue;
      }
      if (funcmatch == G__EXACT) {
         if (G__compiled_func_cxx(&result3, funcname, libp, hash) == 1) {
            // --
#ifdef G__ASM
            if (G__asm_noverflow) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_FUNC compiled '%s' paran: %d  %s:%d\n", G__asm_cp, G__asm_dt, funcname, libp->paran, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_FUNC;
               G__asm_inst[G__asm_cp+1] = 1 + 10 * hash;
               G__asm_inst[G__asm_cp+2] = (long)(&G__asm_name[G__asm_name_p]);
               G__asm_inst[G__asm_cp+3] = libp->paran;
               G__asm_inst[G__asm_cp+4] = (long) G__compiled_func_cxx;
               G__asm_inst[G__asm_cp+5] = 0;
               if ((G__asm_name_p + strlen(funcname) + 1) < G__ASM_FUNCNAMEBUF) {
                  strcpy(G__asm_name + G__asm_name_p, funcname);
                  G__asm_name_p += strlen(funcname) + 1;
                  G__inc_cp_asm(6, 0);
               }
               else {
                  G__abortbytecode();
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "COMPILE ABORT function name buffer overflow");
                     G__printlinenum();
                  }
#endif // G__ASM_DBG
                  // --
               }
            }
#endif // G__ASM
#ifdef G__DUMPFILE
            if (G__dumpfile && !G__no_exec_compile) {
               G__dumpspace -= 3;
               for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                  fprintf(G__dumpfile, " ");
               }
               G__valuemonitor(result3, result7);
               fprintf(G__dumpfile, "/* return(cmp) %s()=%s */\n", funcname, result7);
            }
#endif // G__DUMPFILE
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;
            if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
               G__getindexedvalue(&result3, libp->parameter[nindex]);
            }
            return result3;
         }
         if (G__library_func(&result3, funcname, libp, hash) == 1) {
            if (G__no_exec_compile) {
               G__value_typenum(result3) = Reflex::Type::ByTypeInfo(typeid(int)); // result3.type = 'i'
            }
#ifdef G__ASM
            if (G__asm_noverflow) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_FUNC library '%s' paran: %d  %s:%d\n", G__asm_cp, G__asm_dt, funcname, libp->paran, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_FUNC;
               G__asm_inst[G__asm_cp+1] = 1 + 10 * hash;
               G__asm_inst[G__asm_cp+2] = (long)(&G__asm_name[G__asm_name_p]);
               G__asm_inst[G__asm_cp+3] = libp->paran;
               G__asm_inst[G__asm_cp+4] = (long) G__library_func;
               G__asm_inst[G__asm_cp+5] = 0;
               if ((G__asm_name_p + strlen(funcname) + 1) < G__ASM_FUNCNAMEBUF) {
                  strcpy(G__asm_name + G__asm_name_p, funcname);
                  G__asm_name_p += strlen(funcname) + 1;
                  G__inc_cp_asm(6, 0);
               }
               else {
                  G__abortbytecode();
#ifdef G__ASM_DBG
                  if (G__asm_dbg)
                     G__fprinterr(G__serr, "COMPILE ABORT function name buffer overflow");
                  G__printlinenum();
#endif // G__ASM_DBG
                  // --
               }
            }
#endif // G__ASM
#ifdef G__DUMPFILE
            if (G__dumpfile && !G__no_exec_compile) {
               G__dumpspace -= 3;
               for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                  fprintf(G__dumpfile, " ");
               }
               G__valuemonitor(result3, result7);
               fprintf(G__dumpfile, "/* return(lib) %s()=%s */\n", funcname, result7);
            }
#endif // G__DUMPFILE
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;
            if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
               G__getindexedvalue(&result3, libp->parameter[nindex]);
            }
            return result3;
         }
      }
#ifdef G__TEMPLATEFUNC
      //
      //  Search template function.
      //
      if (G__templatefunc(&result3, funcname, libp, hash, funcmatch)) {
         // --
#ifdef G__DUMPFILE
         if (G__dumpfile && !G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ++ipara) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile, "/* return(lib) %s()=%s */\n", funcname, result7);
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, libp);
         }
         else {
            *known3 = 1;
         }
         return result3;
      }
#endif // G__TEMPLATEFUNC
      // --
   }
   //
   //  Check for function-style cast.
   //
   if ((memfunc_flag == G__TRYNORMAL) || (memfunc_flag == G__CALLSTATICMEMFUNC)) {
      int store_var_typeX = G__var_type;
      ::Reflex::Type funcnameTypedef = G__find_typedef(funcname);
      G__var_type = store_var_typeX;
      if (funcnameTypedef) {
         if (-1 != G__get_tagnum(funcnameTypedef)) {
            strcpy(funcname, G__struct.name[G__get_tagnum(funcnameTypedef)]);
         }
         else {
            result3 = libp->para[0];
            // CHECKME: or maye we should use: cursor
            if (
               G__fundamental_conversion_operator(G__get_type(funcnameTypedef), -1, typedf, G__get_reftype(funcnameTypedef), 0, &result3)
            ) {
               *known3 = 1;
               if (oprp) *known3 = G__additional_parenthesis(&result3, libp);
               return(result3);
            }
            strcpy(funcname, G__type2string(G__get_type(funcnameTypedef)
                                            , G__get_tagnum(funcnameTypedef) , -1
                                            , G__get_reftype(funcnameTypedef) , 0));
         }
         G__hash(funcname, hash, hash_2);
      }
      classhash = strlen(funcname);
      int cursor = 0;
      while (cursor < G__struct.alltag) {
         if ((G__struct.hash[cursor] == classhash) &&
               (strcmp(G__struct.name[cursor], funcname) == 0)
            ) {
            if ('e' == G__struct.type[cursor] &&
                  G__value_typenum(libp->para[0]).IsEnum()) {
               return(libp->para[0]);
            }
            store_struct_offset = G__store_struct_offset;

            /* questionable part */
            /* store_exec_memfunc=G__exec_memberfunc; */

            store_tagnum = G__tagnum;
            G__tagnum = G__Dict::GetDict().GetScope(cursor);
            if (G__CPPLINK != G__struct.iscpplink[G__get_tagnum(G__tagnum)]) {
               G__alloc_tempobject(G__get_tagnum(G__tagnum), -1);
               G__store_struct_offset = (char*)G__p_tempbuf->obj.obj.i;
#ifdef G__ASM
               if (G__asm_noverflow) {
                  if (G__throwingexception) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x: ALLOCEXCEPTION %d\n", G__asm_cp, G__get_tagnum(G__tagnum));
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__ALLOCEXCEPTION;
                     G__asm_inst[G__asm_cp+1] = G__get_tagnum(G__tagnum);
                     G__inc_cp_asm(2, 0);
                  }
                  else {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x: ALLOCTEMP %d\n", G__asm_cp, G__get_tagnum(G__tagnum));
                        G__fprinterr(G__serr, "%3x: SETTEMP\n", G__asm_cp);
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
                     G__asm_inst[G__asm_cp+1] = G__get_tagnum(G__tagnum);
                     G__asm_inst[G__asm_cp+2] = G__SETTEMP;
                     G__inc_cp_asm(3, 0);
                  }
               }
#endif // G__ASM
               // --
            }
            else {
               G__store_struct_offset = G__PVOID;
            }
            G__incsetup_memfunc(G__tagnum);
            for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
               *known3 = G__interpret_func(&result3, funcname, libp, hash, G__tagnum, funcmatch, G__TRYCONSTRUCTOR);
               if (*known3) {
                  break;
               }
            }
            if (G__CPPLINK == G__struct.iscpplink[G__get_tagnum(G__tagnum)]
                  && !G__throwingexception
               ) {
               if (G__dispsource) {
                  G__fprinterr(G__serr, "G__getfunction_libp: Create temp object: level: %d  typename '%s'  addr: 0x%lx,%d for function call '%s()'  %s:%d\n", G__templevel, G__struct.name[G__get_tagnum(G__tagnum)], G__p_tempbuf->obj.obj.i, funcname, __FILE__, __LINE__);
               }
               G__store_tempobject(result3);
#ifdef G__ASM
               if (G__asm_noverflow) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: STORETEMP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__STORETEMP;
                  G__inc_cp_asm(1, 0);
               }
#endif // G__ASM
               // --
            }
            else {
               G__value_typenum(result3) = G__tagnum;
               result3.obj.i = (long)G__store_struct_offset;
               result3.ref = (long)G__store_struct_offset;
            }
            G__tagnum = store_tagnum;

            /* questionable part */
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;

            G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
            if (G__asm_noverflow
                  && (!G__throwingexception ||
                      ((G__value_typenum(result3).RawType().IsClass()
                        || G__value_typenum(result3).RawType().IsEnum()
                        || G__value_typenum(result3).RawType().IsUnion())
                       && G__CPPLINK != G__struct.iscpplink[G__get_tagnum(G__value_typenum(result3).RawType())]))
               ) {
               G__asm_inst[G__asm_cp] = G__POPTEMP;
               if (G__throwingexception) G__asm_inst[G__asm_cp+1] = G__get_tagnum(G__value_typenum(result3).RawType());
               else                     G__asm_inst[G__asm_cp+1] = -1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) G__fprinterr(G__serr, "%3x: POPTEMP %d\n"
                                               , G__asm_cp, G__asm_inst[G__asm_cp+1]);
#endif
               G__inc_cp_asm(2, 0);
            }
#endif

            if (0 == *known3) {
               if (-1 != cursor && libp->paran == 1 && -1 != G__get_tagnum(G__value_typenum(libp->para[0]))) {
                  char* local_store_struct_offset = G__store_struct_offset;
                  char* local_store_memberfunc_struct_offset = G__memberfunc_struct_offset;
                  ::Reflex::Scope local_store_memberfunc_tagnum = G__memberfunc_tagnum;
                  int local_store_exec_memberfunc = G__exec_memberfunc;
                  store_tagnum = G__tagnum;
                  G__inc_cp_asm(-5, 0); /* cancel ALLOCTEMP, SETTEMP, POPTEMP */
                  G__pop_tempobject();
                  G__tagnum = G__value_typenum(libp->para[0]).RawType();
                  if (!G__tagnum) G__tagnum = Reflex::Scope::GlobalScope();
                  G__store_struct_offset = (char*) libp->para[0].obj.i;
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     G__asm_inst[G__asm_cp] = G__PUSHSTROS;
                     G__asm_inst[G__asm_cp+1] = G__SETSTROS;
                     G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x: PUSHSTROS\n", G__asm_cp - 2);
                        G__fprinterr(G__serr, "%3x: SETSTROS\n", G__asm_cp - 1);
                     }
#endif
                  }
#endif
                  sprintf(funcname, "operator %s", G__fulltagname(cursor, 1));
                  G__hash(funcname, hash, hash_2);
                  G__incsetup_memfunc(G__tagnum);
                  libp->paran = 0;
                  for (funcmatch = G__EXACT;funcmatch <= G__USERCONV;funcmatch++) {
                     *known3 = G__interpret_func(&result3, funcname
                                                 , libp, hash
                                                 , G__tagnum
                                                 , funcmatch
                                                 , G__TRYMEMFUNC);
                     if (*known3) {
#ifdef G__ASM
                        if (G__asm_noverflow) {
                           G__asm_inst[G__asm_cp] = G__POPSTROS;
                           G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
                           if (G__asm_dbg)
                              G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp - 1);
#endif
                        }
#endif
                        break;
                     }
                  }
                  G__memberfunc_struct_offset = local_store_memberfunc_struct_offset;
                  G__memberfunc_tagnum = local_store_memberfunc_tagnum;
                  G__exec_memberfunc = local_store_exec_memberfunc;
                  G__tagnum = store_tagnum;
                  G__store_struct_offset = local_store_struct_offset;
               }
               else if (-1 != cursor && libp->paran == 1) {
                  G__fprinterr(G__serr, "Error: No matching constructor for explicit conversion %s", item);
                  G__genericerror((char*)NULL);
               }
               /* omitted constructor, return uninitialized object */
               *known3 = 1;
               if (oprp) *known3 = G__additional_parenthesis(&result3, libp);
               return(result3);
            }
            else {
               /* Return '*this' as result */
               if (oprp) *known3 = G__additional_parenthesis(&result3, libp);
               return(result3);
            }
         }
         cursor++;
      }
      result3.ref = 0;
      if (G__explicit_fundamental_typeconv(funcname, classhash, libp, &result3)) {
         *known3 = 1;
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (oprp) *known3 = G__additional_parenthesis(&result3, libp);
         return(result3);
      }
   }
   //
   //  Check for use of operator() on an object of class type.
   //
   if (G__parenthesisovld(&result3, funcname, libp, G__TRYNORMAL)) {
      *known3 = 1;
      if (
         nindex &&
         (
            G__value_typenum(result3).FinalType().IsPointer() ||
            G__value_typenum(result3).FinalType().IsArray() ||
            G__value_typenum(result3).FinalType().IsClass() ||
            G__value_typenum(result3).FinalType().IsUnion() ||
            G__value_typenum(result3).FinalType().IsEnum()
         )
      ) {
         G__getindexedvalue(&result3, libp->parameter[nindex]);
      }
      else if (
         nindex &&
         (
            G__value_typenum(result3).RawType().IsClass() ||
            G__value_typenum(result3).RawType().IsEnum() ||
            G__value_typenum(result3).RawType().IsUnion()
         )
      ) {
         int len;
         strcpy(libp->parameter[0], libp->parameter[nindex] + 1);
         len = strlen(libp->parameter[0]);
         if (len > 1) {
            libp->parameter[0][len-1] = 0;
         }
         libp->para[0] = G__getexpr(libp->parameter[0]);
         libp->paran = 1;
         G__parenthesisovldobj(&result3, &result3, "operator[]", libp, G__TRYNORMAL);
      }
      if (oprp) {
         *known3 = G__additional_parenthesis(&result3, libp);
      }
      return result3;
   }
   //
   //  Check for pointer to function used like a normal function call.
   //
   //       int (*p2f)(void);
   //       p2f();
   //
   ::Reflex::Member var = G__getvarentry(funcname, hash, ::Reflex::Scope::GlobalScope(), G__p_local);
   if (var && G__get_type(var.TypeOf()) == '1') {
      sprintf(result7, "*%s", funcname);
      *known3 = 0;
      pfparam = strchr(item, '(');
      p2ffpara = libp;
      result3 = G__pointer2func(0, result7, /*FIXME*/(char*)pfparam, known3);
      p2ffpara = 0;
      if (*known3) {
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (
            nindex &&
            (
               G__value_typenum(result3).FinalType().IsPointer() ||
               G__value_typenum(result3).FinalType().IsArray() ||
               G__value_typenum(result3).FinalType().IsClass() ||
               G__value_typenum(result3).FinalType().IsUnion() ||
               G__value_typenum(result3).FinalType().IsEnum()
            )
         ) {
            G__getindexedvalue(&result3, libp->parameter[nindex]);
         }
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, libp);
         }
         return result3;
      }
   }
   //
   //  Check for a function-style macro invocation.
   //
   *known3 = 0; // Flag that no function was called.
#ifdef G__DUMPFILE
   if (G__dumpfile && !G__no_exec_compile) {
      G__dumpspace -= 3;
   }
#endif // G__DUMPFILE
   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   if (!G__oprovld) { // There were arguments.
      if (G__asm_noverflow && libp->paran) {
         G__asm_cp = store_cp_asm;
      }
      G__asm_clear_mask = 1;
      result3 = G__execfuncmacro(item, known3);
      G__asm_clear_mask = 0;
      if (*known3) {
         if (
            nindex &&
            (
               G__value_typenum(result3).FinalType().IsPointer() ||
               G__value_typenum(result3).FinalType().IsArray() ||
               G__value_typenum(result3).FinalType().IsClass() ||
               G__value_typenum(result3).FinalType().IsUnion() ||
               G__value_typenum(result3).FinalType().IsEnum()
            )
         ) {
            G__getindexedvalue(&result3, libp->parameter[nindex]);
         }
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, libp);
         }
         return result3;
      }
   }
   return G__null;
}

//______________________________________________________________________________
extern "C" G__value G__getfunction(const char* item, int* known3, int memfunc_flag)
{
   G__value result3 = G__null;
   G__StrBuf funcname_sb(G__LONGLINE);
   char* funcname = funcname_sb;
   int overflowflag = 0;
   G__StrBuf result7_sb(G__LONGLINE);
   char* result7 = result7_sb;
   int ipara;
   int ig35;
   int ig15;
   int lenitem;
   int nest = 0;
   int single_quote = 0;
   int double_quote = 0;
   G__param fpara;
   static G__param* p2ffpara = 0;
   int hash;
   int hash_2;
   int funcmatch;
   int classhash;
   ::Reflex::Scope store_tagnum;
   int store_exec_memberfunc;
   int store_asm_noverflow;
   int store_var_type;
   ::Reflex::Scope store_memberfunc_tagnum;
   int store_memberfunc_var_type;
   ::Reflex::Scope store_def_tagnum;
   ::Reflex::Scope store_tagdefining;
   int tempstore;
   const char* pfparam = 0;
   int nindex = 0;
   int base1 = 0;
   int oprp = 0;
   int store_cp_asm = 0;
   int memfuncenvflag = 0;
#ifdef G__DEBUG
   {
      int jdbg;
      int sizedbg = sizeof(struct G__param);
      char *pcdbg = (char*)(&fpara);
      for (jdbg = 0;jdbg < (int)sizedbg;jdbg++) {
         *(pcdbg + jdbg) = (char)0xa3;
      }
   }
#endif // G__DEBUG
   store_exec_memberfunc = G__exec_memberfunc;
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   char* store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   //
   // if string expression "string" return
   //
   if (item[0] == '"') {
      result3 = G__null;
      return(result3);
   }
   //
   // get length of expression
   //
   lenitem = strlen(item);
   //
   // Scan item[] until '(' to get function name and hash
   //
   // Separate function name
   ig15 = 0;
   hash = 0;
   while ((item[ig15] != '(') && (ig15 < lenitem)) {
      funcname[ig15] = item[ig15];
      hash += item[ig15];
      ig15++;
   }
   if ((ig15 == 8) && !strncmp(funcname, "operator", 8) && !strncmp(item + ig15, "()(", 3)) {
      strcpy(funcname + 8, "()");
      hash = hash + '(' + ')';
      ig15 += 2;
   }
   //
   // if itemp[0] == '(' this is a cast or a pointer to function.
   //
   int castflag = 0;
   if (!ig15) {
      castflag = 1;
   }
   //
   //  If '(' not found in expression, then this
   //  is not a function call, return.  This shouldn't happen.
   //
   if (item[ig15] != '(') { // if no parenthesis, this is not a function
      result3 = G__null;
      return result3;
   }
   funcname[ig15++] = '\0';
   if ((strchr(funcname, '.') || strstr(funcname, "->")) && strncmp(funcname, "operator", 8)) {
      result3 = G__null;
      return result3;
   }
   //
   // f<B>(x) -> f<NS::B>(x)
   //
   if (funcname[0]) {
      std::string tmp = funcname;
      if (G__rename_templatefunc(tmp)) {
         strcpy(funcname, tmp.c_str());
      }
   }
   //
   //  Parse arguments to function call.
   // 
   //       func(argument, ...)
   //            ^
   //
   fpara.paran = 0;
   G__value_typenum(fpara.para[0]) = Reflex::Type();
   while (ig15 < lenitem) {
      //
      //  Collect one function parameter.
      //
      ig35 = 0;
      nest = 0;
      int tmpltnest = 0;
      single_quote = 0;
      double_quote = 0;
      while (item[ig15] == ' ') {
         ++ig15;
      }
      while (
         (ig15 < lenitem) &&
         (
            (
               (item[ig15] != ',') &&
               (item[ig15] != ')')
            ) ||
            (nest > 0) ||
            (tmpltnest > 0) ||
            (single_quote > 0) ||
            (double_quote > 0)
         )
      ) {
         switch (item[ig15]) {
            case '"' :
               if (!single_quote) {
                  double_quote ^= 1;
               }
               break;
            case '\'' :
               if (!double_quote) {
                  single_quote ^= 1;
               }
               break;
            case '(':
            case '[':
            case '{':
               if (!double_quote && !single_quote) {
                  ++nest;
               }
               break;
            case ')':
            case ']':
            case '}':
               if (!double_quote && !single_quote) {
                  --nest;
               }
               break;
            case '\\':
               result7[ig35++] = item[ig15++];
               break;
            case '<':
               if (!double_quote && !single_quote) {
                  result7[ig35] = 0;
                  char* checkForTemplate = result7;
                  if (checkForTemplate && !strncmp(checkForTemplate, "const ", 6)) {
                     checkForTemplate += 6;
                  }
                  if (
                     !strcmp(result7, "operator") ||
                     tmpltnest ||
                     G__defined_templateclass(checkForTemplate)
                  ) {
                     ++tmpltnest;
                  }
               }
               break;
            case '>':
               if (!double_quote && !single_quote) {
                  if (tmpltnest) {
                     --tmpltnest;
                  }
               }
               break;
         }
         result7[ig35++] = item[ig15++];
         if (ig35 >= (G__ONELINE - 1)) {
            if (result7[0] == '"') {
               G__value bufv;
               G__StrBuf bufx_sb(G__LONGLINE);
               char* bufx = bufx_sb;
               strncpy(bufx, result7, G__ONELINE - 1);
               while (
                  (ig15 < lenitem) &&
                  (
                     (
                        (item[ig15] != ',') &&
                        (item[ig15] != ')')
                     ) ||
                     (nest > 0) ||
                     single_quote ||
                     double_quote
                  )
               ) {
                  switch (item[ig15]) {
                     case '"':
                        if (!single_quote) {
                           double_quote ^= 1;
                        }
                        break;
                     case '\'':
                        if (!double_quote) {
                           single_quote ^= 1;
                        }
                        break;
                     case '(':
                     case '[':
                     case '{':
                        if (!double_quote && !single_quote) {
                           ++nest;
                        }
                        break;
                     case ')':
                     case ']':
                     case '}':
                        if (!double_quote && !single_quote) {
                           --nest;
                        }
                        break;
                     case '\\':
                        bufx[ig35++] = item[ig15++];
                        break;
                  }
                  bufx[ig35++] = item[ig15++];
                  if (ig35 >= (G__LONGLINE - 1)) {
                     G__genericerror("Limitation: Too long function argument");
                     return G__null;
                  }
               }
               bufx[ig35] = 0;
               bufv = G__strip_quotation(bufx);
               sprintf(result7, "(char*)(%ld)", bufv.obj.i);
               ig35 = strlen(result7) + 1;
               break;
            }
            else if (ig35 > (G__LONGLINE - 1)) {
               G__fprinterr(G__serr, "Limitation: length of one function argument be less than %d" , G__LONGLINE);
               G__genericerror(0);
               G__fprinterr(G__serr, "Use temp variable as workaround.\n");
               *known3 = 1;
               return G__null;
            }
            else {
               overflowflag = 1;
            }
         }
      }
      if ((item[ig15] == ')') && (ig15 < (lenitem - 1))) { // Cast expression, or pointer to function usage.
         //
         //  v                    v            <-- this makes
         //  (type)expression  or (*p_func)();    castflag=1
         //       ^                       ^    <-- this makes
         //                                       castflag=2
         //
         if (castflag == 1) {
            if (
               (
                  (item[ig15+1] == '-') &&
                  (item[ig15+2] == '>')
               ) ||
               (item[ig15+1] == '.')
            ) {
               castflag = 3;
            }
            else {
               castflag = 2;
            }
         }
         else if (
            (
               (item[ig15+1] == '-') &&
               (item[ig15+2] == '>')
            ) ||
            (item[ig15+1] == '.')
         ) {
            castflag = 3;
            base1 = ig15 + 1;
         }
         else if (item[ig15+1] == '[') {
            nindex = fpara.paran + 1;
         }
         else if (funcname[0] && isalnum(item[ig15+1])) {
            G__fprinterr(G__serr, "Error: %s  Syntax error?", item);
            G__printlinenum();
         }
         else if (
            isalpha(item[0]) ||
            (item[0] == '_') ||
            (item[0] == '$')
         ) {
            int itmp;
            result7[ig35] = '\0';
            strcpy(fpara.parameter[fpara.paran], result7);
            if (ig35) {
               fpara.parameter[++fpara.paran][0] = '\0';
            }
            for (itmp = 0; itmp < fpara.paran; ++itmp) {
               fpara.para[itmp] = G__getexpr(fpara.parameter[itmp]);
            }
            if (G__parenthesisovld(&result3, funcname, &fpara, G__TRYNORMAL)) {
               *known3 = 1;
               return G__operatorfunction(&result3, item + ig15 + 2, known3, result7, funcname);
            }
            else {
               result3 = G__getfunction_libp(item, funcname, &fpara, known3, G__TRYNORMAL);
               if (*known3) {
                  return G__operatorfunction(&result3, item + ig15 + 2, known3, result7, funcname);
               }
            }
         }
      }
      ++ig15;
      result7[ig35] = '\0';
      if (ig35 < G__ONELINE) {
         strcpy(fpara.parameter[fpara.paran], result7);
      }
      else {
         strcpy(fpara.parameter[fpara.paran], "@");
         fpara.para[fpara.paran] = G__getexpr(result7);
      }
      fpara.parameter[++fpara.paran][0] = '\0';
   }
   if ((castflag == 1) && !funcname[0]) { // A parenthesized expression.
      //
      //  A parenthesized expression.
      //
      if ((fpara.paran == 1) && !strcmp(fpara.parameter[0], "@")) {
         result3 = fpara.para[0];
      }
      else {
         result3 = G__getexpr(result7);
      }
      *known3 = 1;
      return result3;
   }
   if (castflag == 2) { // A cast, or use of pointer to function.
      if (fpara.parameter[0][0] == '*') {
         //
         // pointer to function
         //
         //  (*p_function)(param);
         //   ^
         //  this '*' is significant
         //
         switch (fpara.parameter[1][0]) {
            case '[':
               return G__pointerReference(fpara.parameter[0], &fpara, known3);
            case '(':
            default:
               return G__pointer2func(0, fpara.parameter[0], fpara.parameter[1], known3);
         }
      }
#ifdef G__PTR2MEMFUNC
      else if (
         (fpara.parameter[1][0] == '(') &&
         (
            strstr(fpara.parameter[0], ".*") ||
            strstr(fpara.parameter[0], "->*")
         )
      ) {
         //
         // pointer to member function
         // 
         //  (obj.*p2mf)(param);
         //  (obj->*p2mf)(param);
         //
         return G__pointer2memberfunction(fpara.parameter[0], fpara.parameter[1], known3);
      }
#endif // G__PTR2MEMFUNC
      else if ((fpara.paran >= 2) && (fpara.parameter[1][0] == '[')) {
         //
         // (expr)[n]
         //
         result3 = G__getexpr(G__catparam(&fpara, fpara.paran, ""));
         *known3 = 1;
         return result3;
      }
      //
      // casting
      // 
      //  (type) expression;
      //
      if (fpara.paran > 2) {
         if (fpara.parameter[fpara.paran-1][0] != '@') {
            fpara.para[1] = G__getexpr(fpara.parameter[fpara.paran-1]);
         }
         else {
            fpara.para[1] = fpara.para[fpara.paran-1];
         }
         result3 = G__castvalue(G__catparam(&fpara, fpara.paran - 1, ","), fpara.para[1]);
      }
      else {
         if (fpara.parameter[1][0] != '@') {
            fpara.para[1] = G__getexpr(fpara.parameter[1]);
         }
         else {
            fpara.para[1] = fpara.para[1];
         }
         result3 = G__castvalue(fpara.parameter[0], fpara.para[1]);
      }
      *known3 = 1;
      return result3;
   }
   if (castflag == 3) { // A parenthesized expression followed by a member access operator.
      store_var_type = G__var_type;
      G__var_type = 'p';
      if (fpara.parameter[1][0] == '.') {
         hash_2 = 1;
      }
      else {
         hash_2 = 2;
      }
      if (base1) {
         strncpy(fpara.parameter[0], item, base1);
         fpara.parameter[0][base1] = '\0';
         strcpy(fpara.parameter[1], item + base1);
      }
      if (memfunc_flag == G__CALLMEMFUNC) {
         result3 = G__getstructmem(store_var_type, funcname, fpara.parameter[1] + hash_2, fpara.parameter[0], known3, Reflex::Scope(), hash_2);
      }
      else {
         result3 = G__getstructmem(store_var_type, funcname, fpara.parameter[1] + hash_2, fpara.parameter[0], known3, Reflex::Scope::GlobalScope(), hash_2);
      }
      G__var_type = store_var_type;
      return result3;
   }
   if (!strlen(fpara.parameter[0])) { // First arg is empty, check for usage of operator().
      if ((fpara.paran > 1) && (item[strlen(item)-1] == ')')) {
         if ((fpara.paran == 2) && (fpara.parameter[1][0] == '(')) { // We have "myobj()", operator() usage.
            oprp = 1;
         }
         else {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: Empty arg%d", 1);
               G__printlinenum();
            }
         }
      }
      fpara.paran = 0;
   }
   //
   //  Initialize the result to void.
   //  Initialized function found and executed flag to true.
   //
   G__value_typenum(result3) = Reflex::Type::ByTypeInfo(typeid(void));
   *known3 = 1;
   //
   //  Some function are handled specially, check for those
   //  first.  The special ones are sizeof, offsetof, typeid,
   //  and va_arg.
   //
   //
   if (G__special_func(&result3, funcname, &fpara, hash)) {
      G__var_type = 'p';
      return result3;
   }
   //
   //  Evaluate argument expressions.
   //
   //  Result is:
   //
   //       parameter: value as a string,
   //            para: value
   //
#ifdef G__ASM
   store_asm_noverflow = G__asm_noverflow;
   if (G__oprovld) {
      // In case of operator overloading function, arguments are already
      // evaluated. Avoid duplication in argument stack by temporarily
      // reset G__asm_noverflow.
      G__suspendbytecode();
   }
   if (
      G__asm_noverflow &&
      fpara.paran &&
      (
         (G__store_struct_offset != G__memberfunc_struct_offset) ||
         G__do_setmemfuncenv
      )
   ) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: SETMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SETMEMFUNCENV;
      G__inc_cp_asm(1, 0);
      memfuncenvflag = 1;
   }
   if (G__asm_noverflow && fpara.paran) {
      store_cp_asm = G__asm_cp;
   }
#endif // G__ASM
   char* store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_memberfunc_var_type = G__var_type;
   char* store_globalvarpointer = G__globalvarpointer;
   G__globalvarpointer = G__PVOID;
   G__tagnum = G__memberfunc_tagnum;
   G__store_struct_offset = G__memberfunc_struct_offset;
   G__var_type = 'p';
   if (p2ffpara) {
      fpara = *p2ffpara;
      p2ffpara = 0;
   }
   else {
      for (ig15 = 0; ig15 < fpara.paran; ++ig15) {
         if (fpara.parameter[ig15][0] == '[') {
            fpara.paran = ig15;
            break;
         }
         if (!fpara.parameter[ig15][0]) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: Empty arg%d", ig15 + 1);
               G__printlinenum();
            }
         }
         if (fpara.parameter[ig15][0] != '@') {
            fpara.para[ig15] = G__getexpr(fpara.parameter[ig15]);
         }
      }
   }
#ifdef G__ASM
   if (G__asm_noverflow && fpara.paran && memfuncenvflag) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: RECMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__RECMEMFUNCENV;
      G__inc_cp_asm(1, 0);
      memfuncenvflag = 0;
   }
#endif // G__ASM
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__var_type = store_memberfunc_var_type;
   G__globalvarpointer = store_globalvarpointer;
#ifdef G__ASM
   if (G__oprovld) {
      G__asm_noverflow = store_asm_noverflow;
   }
   else {
      G__asm_noverflow &= store_asm_noverflow;
   }
#endif // G__ASM
#ifdef G__SECURITY
   if ((G__return > G__RETURN_NORMAL) || (G__security_error && (G__security != G__SECURE_NONE))) {
      return G__null;
   }
#endif // G__SECURITY
   //
   //  Terminate the evaluated argument list.
   //
   fpara.para[fpara.paran] = G__null;
   if (!funcname[0]) { // No name, we are a parenthesized comma operator expression.
      //
      //  If not a function name, then this is
      //  a parenthesized comma operator expression:
      //
      //       (expr1, expr2, ..., exprn)
      //
      result3 = fpara.para[0]; // FIXME: Returns the wrong expression, instead of 0, should be paran-1.
      return result3;
   }
   //
   //  Check for a qualified function name.
   //
   //       ::f()
   //       A::B::f()
   //
   //  Note:
   //
   //    G__exec_memberfunc restored at return memfunc_flag is local,
   //    there should be no problem modifying these variables.
   //    store_struct_offset and store_tagnum are only used in the
   //    explicit type conversion section.  It is OK to use them here
   //    independently.
   //
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_def_tagnum = G__def_tagnum;
   store_tagdefining = G__tagdefining;
   int intTagNum = G__get_tagnum(G__tagnum);
   {
      int which_scope = G__scopeoperator(funcname, &hash, &G__store_struct_offset, &intTagNum);
      G__tagnum = G__Dict::GetDict().GetScope(intTagNum); // might have been changed by G__scopeoperator
      switch (which_scope) {
         case G__GLOBALSCOPE:
            G__exec_memberfunc = 0;
            memfunc_flag = G__TRYNORMAL;
            G__def_tagnum = ::Reflex::Scope();
            G__tagdefining = ::Reflex::Scope();
            break;
         case G__CLASSSCOPE:
            G__exec_memberfunc = 1;
            memfunc_flag = G__CALLSTATICMEMFUNC;
            G__def_tagnum = ::Reflex::Scope();
            G__tagdefining = ::Reflex::Scope();
            G__memberfunc_tagnum = G__tagnum;
            break;
      }
   }
#ifdef G__DUMPFILE
   if (G__dumpfile && !G__no_exec_compile) {
      //
      //  Dump that a function is called.
      //
      for (ipara = 0; ipara < G__dumpspace; ++ipara) {
         fprintf(G__dumpfile, " ");
      }
      fprintf(G__dumpfile, "%s(", funcname);
      for (ipara = 1; ipara <= fpara.paran; ++ipara) {
         if (ipara != 1) {
            fprintf(G__dumpfile, ",");
         }
         G__valuemonitor(fpara.para[ipara-1], result7);
         fprintf(G__dumpfile, "%s", result7);
      }
      fprintf(G__dumpfile, ");/*%s %d,%p %p*/\n", G__ifile.name, G__ifile.line_number, store_struct_offset, G__store_struct_offset);
      G__dumpspace += 3;
   }
#endif // G__DUMPFILE
   //
   //  Perform overload resolution.
   //
   //  G__EXACT = 1
   //  G__PROMOTION = 2
   //  G__STDCONV = 3
   //  G__USERCONV = 4
   //
   for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
      //
      //  Search for interpreted member function.
      //
      //  G__exec_memberfunc         ==>  memfunc();
      //  memfunc_flag!=G__TRYNORMAL ==>  a.memfunc();
      //
      if (G__exec_memberfunc || (memfunc_flag != G__TRYNORMAL)) {
         ::Reflex::Scope local_tagnum = G__tagnum;
         if (G__exec_memberfunc && (G__get_tagnum(G__tagnum) == -1)) {
            local_tagnum = G__memberfunc_tagnum;
         }
         // Perform any delayed dictionary loading.
         if (G__get_tagnum(G__tagnum) != -1) {
            G__incsetup_memfunc(G__tagnum);
         }
         if (G__get_tagnum(local_tagnum) != -1) {
            //
            //  Call an interpreted function.
            //
            int ret = G__interpret_func(&result3, funcname, &fpara, hash, local_tagnum, funcmatch, memfunc_flag);
            if (ret == 1) {
               // -- We found it and ran it, done.
#ifdef G__DUMPFILE
               if (G__dumpfile && !G__no_exec_compile) {
                  G__dumpspace -= 3;
                  for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                     fprintf(G__dumpfile, " ");
                  }
                  G__valuemonitor(result3, result7);
                  fprintf(G__dumpfile, "/* return(inp) %s.%s()=%s*/\n", G__struct.name[G__get_tagnum(G__tagnum)], funcname, result7);
               }
#endif // G__DUMPFILE
               if (G__store_struct_offset != store_struct_offset) {
                  G__gen_addstros(store_struct_offset - G__store_struct_offset);
               }
               G__store_struct_offset = store_struct_offset;
               G__tagnum = store_tagnum;
               G__def_tagnum = store_def_tagnum;
               G__tagdefining = store_tagdefining;
               G__exec_memberfunc = store_exec_memberfunc;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
               G__memberfunc_struct_offset = store_memberfunc_struct_offset;
               G__setclassdebugcond(G__get_tagnum(G__memberfunc_tagnum), 0);
               if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
                  G__getindexedvalue(&result3, fpara.parameter[nindex]);
               }
               if (oprp) {
                  *known3 = G__additional_parenthesis(&result3, &fpara);
               }
               return result3;
            }
         }
      }
      //
      //  If searching only member function.
      //
      if (memfunc_flag && (G__store_struct_offset || (memfunc_flag != G__CALLSTATICMEMFUNC))) {
         //
         //  If member function is called with a qualified name,
         //  then don't examine global functions.
         //
         //  There are 2 cases:
         //
         //                                      G__exec_memberfunc
         //    obj.memfunc();                            1
         //    X::memfunc();                             1
         //     X();              constructor            2
         //    ~X();              destructor             2
         //
         //  If G__exec_memberfunc == 2, don't display error message.
         //
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (funcmatch != G__USERCONV) {
            continue;
         }
         if (memfunc_flag == G__TRYDESTRUCTOR) {
            // destructor for base class and class members
#ifdef G__ASM
#ifdef G__SECURITY
            store_asm_noverflow = G__asm_noverflow;
            if (G__security & G__SECURE_GARBAGECOLLECTION) {
               G__abortbytecode();
            }
#endif // G__SECURITY
#endif // G__ASM
#ifdef G__VIRTUALBASE
            if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] != G__CPPLINK) {
               G__basedestructor();
            }
#else // G__VIRTUALBASE
            G__basedestructor();
#endif // G__VIRTUALBASE
#ifdef G__ASM
#ifdef G__SECURITY
            G__asm_noverflow = store_asm_noverflow;
#endif // G__SECURITY
#endif // G__ASM
            // --
         }
         else {
            switch (memfunc_flag) {
               case G__CALLCONSTRUCTOR:
               case G__TRYCONSTRUCTOR:
               case G__TRYIMPLICITCONSTRUCTOR:
                  // constructor for base class and class members default constructor only.
#ifdef G__VIRTUALBASE
                  if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] != G__CPPLINK) {
                     G__baseconstructor(0, 0);
                  }
#else // G__VIRTUALBASE
                  G__baseconstructor(0, 0);
#endif // G__VIRTUALBASE
                  // --
            }
         }
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         *known3 = 0;
         switch (memfunc_flag) {
            case G__CALLMEMFUNC:
               {
                  int ret = G__parenthesisovld(&result3, funcname, &fpara, G__CALLMEMFUNC);
                  if (ret) {
                     *known3 = 1;
                     if (G__store_struct_offset != store_struct_offset) {
                        G__gen_addstros(store_struct_offset - G__store_struct_offset);
                     }
                     G__store_struct_offset = store_struct_offset;
                     G__tagnum = store_tagnum;
                     G__def_tagnum = store_def_tagnum;
                     G__tagdefining = store_tagdefining;
                     if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
                        G__getindexedvalue(&result3, fpara.parameter[nindex]);
                     }
                     if (oprp) {
                        *known3 = G__additional_parenthesis(&result3, &fpara);
                     }
                     return result3;
                  }
               }
               if (funcname[0] == '~') {
                  *known3 = 1;
                  return G__null;
               }
               // NOTE: Intentionally fallthrough!
            case G__CALLCONSTRUCTOR:
               //
               // Search template function
               //
               G__exec_memberfunc = 1;
               G__memberfunc_tagnum = G__tagnum;
               G__memberfunc_struct_offset = G__store_struct_offset;
               if (
                  (funcmatch == G__EXACT) ||
                  (funcmatch == G__USERCONV)
               ) {
                  int ret = G__templatefunc(&result3, funcname, &fpara, hash, funcmatch);
                  if (ret == 1) {
                     // --
#ifdef G__DUMPFILE
                     if (G__dumpfile && !G__no_exec_compile) {
                        G__dumpspace -= 3;
                        for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                           fprintf(G__dumpfile, " ");
                        }
                        G__valuemonitor(result3, result7);
                        fprintf(G__dumpfile , "/* return(lib) %s()=%s */\n", funcname, result7);
                     }
#endif // G__DUMPFILE
                     G__exec_memberfunc = store_exec_memberfunc;
                     G__memberfunc_tagnum = store_memberfunc_tagnum;
                     G__memberfunc_struct_offset = store_memberfunc_struct_offset;
                     if (oprp) {
                        *known3 = G__additional_parenthesis(&result3, &fpara);
                     }
                     else {
                        *known3 = 1;
                     }
                     return result3;
                  }
               }
               G__exec_memberfunc = store_exec_memberfunc;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
               G__memberfunc_struct_offset = store_memberfunc_struct_offset;
               if (G__globalcomp < G__NOLINK) { // Stop here if doing dictionary generation.
                  break;
               }
               if (G__asm_noverflow || !G__no_exec_compile) {
                  if (!G__const_noerror) {
                     G__fprinterr(G__serr, "Error: Cannot call %s::%s in current scope (2)  %s:%d", G__struct.name[G__get_tagnum(G__tagnum)], item, __FILE__, __LINE__);
                  }
                  G__genericerror(0);
               }
               store_exec_memberfunc = G__exec_memberfunc;
               G__exec_memberfunc = 1;
               if (!G__const_noerror && (!G__no_exec_compile || G__asm_noverflow)) {
                  G__fprinterr(G__serr, "Possible candidates are...\n");
                  {
                     G__StrBuf itemtmp_sb(G__LONGLINE);
                     char* itemtmp = itemtmp_sb;
                     sprintf(itemtmp, "%s::%s", G__struct.name[G__get_tagnum(G__tagnum)], funcname);
                     G__display_proto_pretty(G__serr, itemtmp, 1);
                  }
               }
               G__exec_memberfunc = store_exec_memberfunc;
         }
#ifdef G__DUMPFILE
         if (G__dumpfile && !G__no_exec_compile) {
            G__dumpspace -= 3;
         }
#endif // G__DUMPFILE
         if (G__store_struct_offset != store_struct_offset) {
            G__gen_addstros(store_struct_offset - G__store_struct_offset);
         }
         G__store_struct_offset = store_struct_offset;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         if ( // Copy constructor not found.
            fpara.paran &&
            (G__get_type(G__value_typenum(fpara.para[0])) == 'u') &&
            (
               (memfunc_flag == G__TRYCONSTRUCTOR) ||
               (memfunc_flag == G__TRYIMPLICITCONSTRUCTOR)
            )
         ) { // Copy constructor not found.
            G__tagnum = store_tagnum;
            return fpara.para[0];
         }
         result3 = G__null;
         G__value_typenum(result3) = G__tagnum;
         G__tagnum = store_tagnum;
         return result3;
      }
      //
      //  Check for a global function.
      //
      tempstore = G__exec_memberfunc;
      G__exec_memberfunc = 0;
      if (memfunc_flag != G__CALLSTATICMEMFUNC) {
         int ret = G__interpret_func(&result3, funcname, &fpara, hash, G__p_ifunc, funcmatch, G__TRYNORMAL);
         if (ret == 1) {
            // --
#ifdef G__DUMPFILE
            if (G__dumpfile && !G__no_exec_compile) {
               G__dumpspace -= 3;
               for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                  fprintf(G__dumpfile, " ");
               }
               G__valuemonitor(result3, result7);
               fprintf(G__dumpfile , "/* return(inp) %s()=%s*/\n" , funcname, result7);
            }
#endif // G__DUMPFILE
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;
            G__setclassdebugcond(G__get_tagnum(G__memberfunc_tagnum), 0);
            if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
               G__getindexedvalue(&result3, fpara.parameter[nindex]);
            }
            if (oprp) {
               *known3 = G__additional_parenthesis(&result3, &fpara);
            }
            return result3;
         }
      }
      G__exec_memberfunc = tempstore;
      if (
         (funcmatch == G__PROMOTION) ||
         (funcmatch == G__STDCONV)
      ) {
         continue;
      }
      if (funcmatch == G__EXACT) {
         if (memfunc_flag != G__CALLSTATICMEMFUNC) {
            int is_compiled = G__compiled_func_cxx(&result3, funcname, &fpara, hash);
            if (is_compiled) {
               // --
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: LD_FUNC compiled '%s' paran: %d  %s:%d\n", G__asm_cp, G__asm_dt, funcname, fpara.paran, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__LD_FUNC;
                  G__asm_inst[G__asm_cp+1] = 1 + 10 * hash;
                  G__asm_inst[G__asm_cp+2] = (long)(&G__asm_name[G__asm_name_p]);
                  G__asm_inst[G__asm_cp+3] = fpara.paran;
                  G__asm_inst[G__asm_cp+4] = (long) G__compiled_func_cxx;
                  G__asm_inst[G__asm_cp+5] = 0;
                  if ((G__asm_name_p + strlen(funcname) + 1) < G__ASM_FUNCNAMEBUF) {
                     strcpy(G__asm_name + G__asm_name_p, funcname);
                     G__asm_name_p += strlen(funcname) + 1;
                     G__inc_cp_asm(6, 0);
                  }
                  else {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "COMPILE ABORT function name buffer overflow");
                        G__printlinenum();
                     }
#endif // G__ASM_DBG
                     G__abortbytecode();
                  }
               }
#endif // G__ASM
#ifdef G__DUMPFILE
               if (G__dumpfile && !G__no_exec_compile) {
                  G__dumpspace -= 3;
                  for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                     fprintf(G__dumpfile, " ");
                  }
                  G__valuemonitor(result3, result7);
                  fprintf(G__dumpfile, "/* return(cmp) %s()=%s */\n", funcname, result7);
               }
#endif // G__DUMPFILE
               G__exec_memberfunc = store_exec_memberfunc;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
               G__memberfunc_struct_offset = store_memberfunc_struct_offset;
               if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
                  G__getindexedvalue(&result3, fpara.parameter[nindex]);
               }
               return result3;
            }
         }
         if (G__library_func(&result3, funcname, &fpara, hash)) {
            if (G__no_exec_compile) {
               G__value_typenum(result3) = Reflex::Type::ByTypeInfo(typeid(int)); // result3.type == 'i'
            }
#ifdef G__ASM
            if (G__asm_noverflow) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_FUNC library '%s' paran: %d  %s:%d\n", G__asm_cp, G__asm_dt, funcname, fpara.paran, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_FUNC;
               G__asm_inst[G__asm_cp+1] = 1 + 10 * hash;
               G__asm_inst[G__asm_cp+2] = (long)(&G__asm_name[G__asm_name_p]);
               G__asm_inst[G__asm_cp+3] = fpara.paran;
               G__asm_inst[G__asm_cp+4] = (long) G__library_func;
               G__asm_inst[G__asm_cp+5] = 0;
               if ((G__asm_name_p + strlen(funcname) + 1) < G__ASM_FUNCNAMEBUF) {
                  strcpy(G__asm_name + G__asm_name_p, funcname);
                  G__asm_name_p += strlen(funcname) + 1;
                  G__inc_cp_asm(6, 0);
               }
               else {
                  G__abortbytecode();
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "COMPILE ABORT function name buffer overflow");
                  }
                  G__printlinenum();
#endif // G__ASM_DBG
                  // --
               }
            }
#endif // G__ASM
#ifdef G__DUMPFILE
            if (G__dumpfile && !G__no_exec_compile) {
               G__dumpspace -= 3;
               for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                  fprintf(G__dumpfile, " ");
               }
               G__valuemonitor(result3, result7);
               fprintf(G__dumpfile, "/* return(lib) %s()=%s */\n", funcname, result7);
            }
#endif // G__DUMPFILE
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;
            if (nindex && (isupper(G__get_type(G__value_typenum(result3))) || (G__get_type(G__value_typenum(result3)) == 'u'))) {
               G__getindexedvalue(&result3, fpara.parameter[nindex]);
            }
            return result3;
         }
      }
#ifdef G__TEMPLATEFUNC
      //
      //  Search template function
      //
      if (G__templatefunc(&result3, funcname, &fpara, hash, funcmatch)) {
         // --
#ifdef G__DUMPFILE
         if (G__dumpfile && !G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ++ipara) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile, "/* return(lib) %s()=%s */\n", funcname, result7);
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, &fpara);
         }
         else {
            *known3 = 1;
         }
         return result3;
      }
#endif // G__TEMPLATEFUNC
      // --
   }
   //
   //  Check for function-style cast.
   //
   if ((memfunc_flag == G__TRYNORMAL) || (memfunc_flag == G__CALLSTATICMEMFUNC)) {
      int store_var_typeX = G__var_type;
      ::Reflex::Type funcnameTypedef = G__find_typedef(funcname);
      G__var_type = store_var_typeX;
      int target = -1;
      if (funcnameTypedef) {
         target = G__get_tagnum(funcnameTypedef);
         if (target != -1) {
            target = G__get_tagnum(funcnameTypedef);
            strcpy(funcname, G__struct.name[G__get_tagnum(funcnameTypedef)]);
         }
         else {
            result3 = fpara.para[0];
            // CHECKME the old code was using 'i' now named 'hash_2' or cursor
            if (
               G__fundamental_conversion_operator(G__get_type(funcnameTypedef), -1, ::Reflex::Type(), G__get_reftype(funcnameTypedef), 0, &result3)
            ) {
               *known3 = 1;
               if (oprp) {
                  *known3 = G__additional_parenthesis(&result3, &fpara);
               }
               return result3;
            }
            strcpy(funcname, G__type2string(G__get_type(funcnameTypedef), G__get_tagnum(funcnameTypedef), -1, G__get_reftype(funcnameTypedef), 0));
         }
         G__hash(funcname, hash, hash_2);
      }
      if (target == -1) {
         target = G__defined_tagname(funcname, 1);
      }
      classhash = strlen(funcname);
      int cursor = target;
      while ((target != -1) && (cursor == target)) {
         // Intentionally loop only once, the loop used to loop over
         // from 0 to G__struct.alltag and exit in one case using
         // 'break', so we will need to shuffle the control flow a bit
         // to remove the while loop
         if ((G__struct.type[cursor] == 'e') && G__value_typenum(fpara.para[0]).RawType().IsEnum()) {
            return fpara.para[0];
         }
         store_struct_offset = G__store_struct_offset;
         store_tagnum = G__tagnum;
         G__tagnum = G__Dict::GetDict().GetScope(cursor);
         if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] != G__CPPLINK) {
            G__alloc_tempobject(G__get_tagnum(G__tagnum), -1);
            G__store_struct_offset = (char*) G__p_tempbuf->obj.obj.i;
#ifdef G__ASM
            if (G__asm_noverflow) {
               if (G__throwingexception) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x: ALLOCEXCEPTION %d\n", G__asm_cp, G__get_tagnum(G__tagnum));
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__ALLOCEXCEPTION;
                  G__asm_inst[G__asm_cp+1] = G__get_tagnum(G__tagnum);
                  G__inc_cp_asm(2, 0);
               }
               else {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x: ALLOCTEMP %d\n", G__asm_cp, G__get_tagnum(G__tagnum));
                     G__fprinterr(G__serr, "%3x: SETTEMP\n", G__asm_cp);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
                  G__asm_inst[G__asm_cp+1] = G__get_tagnum(G__tagnum);
                  G__asm_inst[G__asm_cp+2] = G__SETTEMP;
                  G__inc_cp_asm(3, 0);
               }
            }
#endif // G__ASM
            // --
         }
         else {
            G__store_struct_offset = G__PVOID;
         }
         G__incsetup_memfunc(G__tagnum);
         for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
            *known3 = G__interpret_func(&result3, funcname, &fpara, hash, G__tagnum, funcmatch, G__TRYCONSTRUCTOR);
            if (*known3) {
               break;
            }
         }
         if ((G__struct.iscpplink[G__get_tagnum(G__tagnum)] == G__CPPLINK) && !G__throwingexception) {
            if (G__dispsource) {
               G__fprinterr(G__serr, "G__getfunction: Create temp object: level: %d  typename: '%s'  addr: 0x%lx for function call '%s()'  %s:%d\n", G__templevel, G__struct.name[G__get_tagnum(G__tagnum)], G__p_tempbuf->obj.obj.i, funcname, __FILE__, __LINE__);
            }
            G__store_tempobject(result3);
#ifdef G__ASM
            if (G__asm_noverflow) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x: STORETEMP\n", G__asm_cp);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__STORETEMP;
               G__inc_cp_asm(1, 0);
            }
#endif // G__ASM
            // --
         }
         else {
            G__value_typenum(result3) = G__tagnum;
            result3.obj.i = (long) G__store_struct_offset;
            result3.ref = (long) G__store_struct_offset;
         }
         G__tagnum = store_tagnum;
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
         if (
            G__asm_noverflow &&
            (
               !G__throwingexception ||
               (
                  (G__get_tagnum(G__value_typenum(result3).RawType()) != -1) &&
                  (G__struct.iscpplink[G__get_tagnum(G__value_typenum(result3).RawType())] != G__CPPLINK)
               )
            )
         ) {
            G__asm_inst[G__asm_cp] = G__POPTEMP;
            if (G__throwingexception) {
               G__asm_inst[G__asm_cp+1] = G__get_tagnum(G__value_typenum(result3));
            }
            else {
               G__asm_inst[G__asm_cp+1] = -1;
            }
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: POPTEMP %d\n" , G__asm_cp, G__asm_inst[G__asm_cp+1]);
            }
#endif // G__ASM_DBG
            G__inc_cp_asm(2, 0);
         }
#endif // G__ASM
         if (!*known3) {
            if ((cursor != -1) && (fpara.paran == 1)) {
               ::Reflex::Type type = G__value_typenum(fpara.para[0]);
               int ret = G__get_tagnum(type);
               if (ret != -1) {
                  char* local_store_struct_offset = G__store_struct_offset;
                  char* local_store_memberfunc_struct_offset = G__memberfunc_struct_offset;
                  ::Reflex::Scope local_store_memberfunc_tagnum = G__memberfunc_tagnum;
                  int local_store_exec_memberfunc = G__exec_memberfunc;
                  store_tagnum = G__tagnum;
                  G__inc_cp_asm(-5, 0); // cancel ALLOCTEMP, SETTEMP, POPTEMP
                  G__pop_tempobject();
                  G__tagnum = G__value_typenum(fpara.para[0]).RawType();
                  if (!G__tagnum) {
                     G__tagnum = G__value_typenum(fpara.para[0]).DeclaringScope();
                  }
                  G__store_struct_offset = (char*) fpara.para[0].obj.i;
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     G__asm_inst[G__asm_cp] = G__PUSHSTROS;
                     G__asm_inst[G__asm_cp+1] = G__SETSTROS;
                     G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x: PUSHSTROS\n", G__asm_cp - 2);
                        G__fprinterr(G__serr, "%3x: SETSTROS\n", G__asm_cp - 1);
                     }
#endif // G__ASM_DBG
                     // --
                  }
#endif // G__ASM
                  sprintf(funcname, "operator %s", G__fulltagname(cursor, 1));
                  G__hash(funcname, hash, hash_2);
                  G__incsetup_memfunc(G__tagnum);
                  fpara.paran = 0;
                  for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
                     *known3 = G__interpret_func(&result3, funcname, &fpara, hash, G__tagnum, funcmatch, G__TRYMEMFUNC);
                     if (*known3) {
                        // --
#ifdef G__ASM
                        if (G__asm_noverflow) {
                           G__asm_inst[G__asm_cp] = G__POPSTROS;
                           G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp - 1);
                           }
#endif // G__ASM_DBG
                           // --
                        }
#endif // G__ASM
                        break;
                     }
                  }
                  G__memberfunc_struct_offset = local_store_memberfunc_struct_offset;
                  G__memberfunc_tagnum = local_store_memberfunc_tagnum;
                  G__exec_memberfunc = local_store_exec_memberfunc;
                  G__tagnum = store_tagnum;
                  G__store_struct_offset = local_store_struct_offset;
               }
            }
            else if ((cursor != -1) && (fpara.paran == 1)) {
               G__fprinterr(G__serr, "Error: No matching constructor for explicit conversion %s", item);
               G__genericerror(0);
            }
            // omitted constructor, return uninitialized object
            *known3 = 1;
            if (oprp) {
               *known3 = G__additional_parenthesis(&result3, &fpara);
            }
            return result3;
         }
         else {
            // Return '*this' as result
            if (oprp) {
               *known3 = G__additional_parenthesis(&result3, &fpara);
            }
            return result3;
         }
         ++cursor;
      }
      result3.ref = 0;
      if (G__explicit_fundamental_typeconv(funcname, classhash, &fpara, &result3)) {
         *known3 = 1;
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, &fpara);
         }
         return result3;
      }
   }
   //
   //  Check for use of operator() on an object of class type.
   //
   if (G__parenthesisovld(&result3, funcname, &fpara, G__TRYNORMAL)) {
      *known3 = 1;
      if (
         nindex &&
         (
            G__value_typenum(result3).FinalType().IsPointer() ||
            G__value_typenum(result3).FinalType().IsArray() ||
            G__value_typenum(result3).FinalType().IsClass() ||
            G__value_typenum(result3).FinalType().IsUnion() ||
            G__value_typenum(result3).FinalType().IsEnum()
         )
      ) {
         G__getindexedvalue(&result3, fpara.parameter[nindex]);
      }
      else if (
         nindex &&
         (
            G__value_typenum(result3).RawType().IsClass() ||
            G__value_typenum(result3).RawType().IsEnum() ||
            G__value_typenum(result3).RawType().IsUnion()
         )
      ) {
         int len;
         strcpy(fpara.parameter[0], fpara.parameter[nindex] + 1);
         len = strlen(fpara.parameter[0]);
         if (len > 1) {
            fpara.parameter[0][len-1] = 0;
         }
         fpara.para[0] = G__getexpr(fpara.parameter[0]);
         fpara.paran = 1;
         G__parenthesisovldobj(&result3, &result3, "operator[]", &fpara, G__TRYNORMAL);
      }
      if (oprp) {
         *known3 = G__additional_parenthesis(&result3, &fpara);
      }
      return result3;
   }
   //
   //  Check for pointer to function used like a normal function call.
   //
   //       int (*p2f)(void);
   //       p2f();
   //
   ::Reflex::Member mem = G__getvarentry(funcname, hash, ::Reflex::Scope::GlobalScope(), G__p_local);
   if (mem && G__get_type(mem.TypeOf()) == '1') {
      sprintf(result7, "*%s", funcname);
      *known3 = 0;
      pfparam = strchr(item, '(');
      p2ffpara = &fpara;
      result3 = G__pointer2func(0, result7, /*FIXME*/(char*)pfparam, known3);
      p2ffpara = 0;
      if (*known3) {
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (
            nindex &&
            (
               G__value_typenum(result3).FinalType().IsPointer() ||
               G__value_typenum(result3).FinalType().IsArray() ||
               G__value_typenum(result3).FinalType().IsClass() ||
               G__value_typenum(result3).FinalType().IsUnion() ||
               G__value_typenum(result3).FinalType().IsEnum()
            )
         ) {
            G__getindexedvalue(&result3, fpara.parameter[nindex]);
         }
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, &fpara);
         }
         return result3;
      }
   }
   //
   //  Check for a function-style macro invocation.
   //
   *known3 = 0; // Flag that no function was called.
#ifdef G__DUMPFILE
   if (G__dumpfile && !G__no_exec_compile) {
      G__dumpspace -= 3;
   }
#endif // G__DUMPFILE
   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   if (!G__oprovld) { // There were arguments.
      // Try function-style macro invocation.
      if (G__asm_noverflow && fpara.paran) {
         G__asm_cp = store_cp_asm;
      }
      G__asm_clear_mask = 1;
      result3 = G__execfuncmacro(item, known3);
      G__asm_clear_mask = 0;
      if (*known3) {
         if (
            nindex &&
            (
               G__value_typenum(result3).FinalType().IsPointer() ||
               G__value_typenum(result3).FinalType().IsArray() ||
               G__value_typenum(result3).FinalType().IsClass() ||
               G__value_typenum(result3).FinalType().IsUnion() ||
               G__value_typenum(result3).FinalType().IsEnum()
            )
         ) {
            G__getindexedvalue(&result3, fpara.parameter[nindex]);
         }
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, &fpara);
         }
         return result3;
      }
   }
   return G__null;
}

//______________________________________________________________________________
static void G__va_start(G__value ap)
{
   // --
   Reflex::Scope local = G__p_local;
   if (!local) return;
   G__va_list* va = (G__va_list*)ap.ref;
   if (!va) return;
   //!!! how do we translate libp? And ip? Check callers.
   va->libp = G__get_properties(local)->stackinfo.libp;
   va->ip = G__get_properties(local)->stackinfo.ifunc.FunctionParameterSize();
   //ifunc = G__get_ifunc_internal(local->ifunc);
   //va->ip = ifunc->para_nu[local->ifn];
}

//______________________________________________________________________________
#if defined(_MSC_VER) && (_MSC_VER>1200)
#pragma optimize("g", off)
#endif

//______________________________________________________________________________
static G__value G__va_arg(G__value ap)
{
   // --
#if defined(_MSC_VER) && (_MSC_VER==1300)
   G__genericerror("Error: You can not use va_arg because of VC++7.0(_MSC_VER==1300) problem");
   return G__null;
#else
   G__va_list* va = (G__va_list*) ap.ref;
   if (!va || !va->libp) {
      return G__null;
   }
   return va->libp->para[va->ip++];
#endif
   // --
}

//______________________________________________________________________________
static void G__va_end(G__value ap)
{
   G__va_list* va = (G__va_list*) ap.ref;
   if (!va) {
      return;
   }
   va->libp = 0;
}

//______________________________________________________________________________
int Cint::Internal::G__special_func(G__value* result7, char* funcname, G__param* libp, int hash)
{
   /*  return 1 if function is executed */
   /*  return 0 if function isn't executed */
   *result7 = G__null;
   if ((hash == 656) && !strcmp(funcname, "sizeof")) {
      if (libp->paran > 1) {
         G__letint(result7, 'i', G__Lsizeof(G__catparam(libp, libp->paran, ",")));
      }
      else {
         G__letint(result7, 'i', G__Lsizeof(libp->parameter[0]));
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: LD 0x%lx from %x  %s:%d\n", G__asm_cp, G__asm_dt, G__int(*result7), G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LD;
         G__asm_inst[G__asm_cp+1] = G__asm_dt;
         G__asm_stack[G__asm_dt] = *result7;
         G__inc_cp_asm(2, 1);
      }
#endif
      return 1;
   }
   if ((hash == 860) && !strcmp(funcname, "offsetof")) {
      if (libp->paran > 2) {
         G__letint(result7, 'i', G__Loffsetof(G__catparam(libp, libp->paran - 1, ","), libp->parameter[libp->paran-1]));
      }
      else {
         G__letint(result7, 'i', G__Loffsetof(libp->parameter[0], libp->parameter[1]));
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: LD 0x%lx from %x  %s:%d\n", G__asm_cp, G__asm_dt, G__int(*result7), G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LD;
         G__asm_inst[G__asm_cp+1] = G__asm_dt;
         G__asm_stack[G__asm_dt] = *result7;
         G__inc_cp_asm(2, 1);
      }
#endif // G__ASM
      return 1;
   }
#ifdef G__TYPEINFO
   if ((hash == 655) && !strcmp(funcname, "typeid")) {
      // --
#ifdef G__ASM
      if (G__asm_noverflow) {
         G__abortbytecode();
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "COMPILE ABORT function name buffer overflow");
            G__printlinenum();
         }
#endif // G__ASM_DBG
         // --
      }
#endif // G__ASM
      G__value_typenum(*result7) = ::Reflex::Type();
      if (G__no_exec_compile) {
         G__value_typenum(*result7) = ::Reflex::Type::ByName("type_info");
         return 1;
      }
      if (libp->paran > 1) {
         G__letint(result7, 'u', (long) G__typeid(G__catparam(libp, libp->paran, ",")));
      }
      else {
         G__letint(result7, 'u', (long) G__typeid(libp->parameter[0]));
      }
      result7->ref = result7->obj.i;
      G__value_typenum(*result7) = G__Dict::GetDict().GetType(*(int*)(result7->ref));
      return 1;
   }
#endif
   if (hash == 624 && !strcmp(funcname, "va_arg")) {
      G__value x;
      if (!G__get_type(libp->para[0])) {
         x = G__getexpr(libp->parameter[0]);
      }
      else {
         x = libp->para[0];
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         // -- We are generating bytecode.
         // FIXME: We cannot support this right now.
         G__asm_noverflow = 0;
         if (G__no_exec_compile) {
            return 0;
         }
      }
      if (G__asm_noverflow) {
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: LD_FUNC special '%s' paran: %d  %s:%d\n", G__asm_cp, G__asm_dt, funcname, 1, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LD_FUNC;
         G__asm_inst[G__asm_cp+1] = 1 + (10 * hash);
         G__asm_inst[G__asm_cp+2] = (long) &G__asm_name[G__asm_name_p];
         G__asm_inst[G__asm_cp+3] = 1;
         G__asm_inst[G__asm_cp+4] = (long) G__special_func;
         G__asm_inst[G__asm_cp+5] = 0;
         G__asm_stack[G__asm_dt] = x;
         if ((G__asm_name_p + strlen(funcname) + 1) < G__ASM_FUNCNAMEBUF) {
            strcpy(G__asm_name + G__asm_name_p, funcname);
            G__asm_name_p += strlen(funcname) + 1;
            G__inc_cp_asm(6, 1);
         }
         else {
            G__abortbytecode();
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "COMPILE ABORT function name buffer overflow");
               G__printlinenum();
            }
#endif // G__ASM_DBG
            // --
         }
      }
#endif // G__ASM
      if (G__no_exec_compile) {
         return 1;
      }
      *result7 = G__va_arg(x);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
static int G__defined(char* tname)
{
   int tagnum;
   ::Reflex::Type typenum;
   typenum = G__find_typedef(tname);
   if (typenum) return 1;
   tagnum = G__defined_tagname(tname, 2);
   if (-1 != tagnum) return 1;
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__library_func(G__value* result7, char* funcname, G__param* libp, int hash)
{
   /*  return 1 if function is executed */
   /*  return 0 if function isn't executed */
   char temp[G__LONGLINE] ;
   static int first_getopt = 1;
#ifdef G__NO_STDLIBS
   return(0);
#endif
   *result7 = G__null;

   /*********************************************************************
   * high priority
   *********************************************************************/

   if (638 == hash && strcmp(funcname, "sscanf") == 0) {
      if (G__no_exec_compile) return(1);
      /* this is a fake function. not compatible with real func */
      /* para0 scan string , para1 format , para2 var pointer */
      G__CHECKNONULL(0, 'C');
      G__CHECKNONULL(1, 'C');
      if (G__checkscanfarg("sscanf", libp, 2)) return(1);
      switch (libp->paran) {
         case 2:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1]))) ;
            break;
         case 3:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2]))) ;
            break;
         case 4:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3]))) ;
            break;
         case 5:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4]))) ;
            break;
         case 6:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5]))) ;
            break;
         case 7:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6]))) ;
            break;
         case 8:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7]))) ;
            break;
         case 9:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8]))) ;
            break;
         case 10:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8])
                               , G__int(libp->para[9]))) ;
            break;
         case 11:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8])
                               , G__int(libp->para[9])
                               , G__int(libp->para[10]))) ;
            break;
         case 12:
            G__letint(result7, 'i'
                      , sscanf((char *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8])
                               , G__int(libp->para[9])
                               , G__int(libp->para[10])
                               , G__int(libp->para[11]))) ;
            break;
         default:
            G__fprinterr(G__serr, "Limitation: sscanf only takes upto 12 arguments");
            G__genericerror((char*)NULL);
            break;
      }
      return(1);
   }

   if (625 == hash && strcmp(funcname, "fscanf") == 0) {
      if (G__no_exec_compile) return(1);
      /* this is a fake function. not compatible with real func */
      /* para0 scan string , para1 format , para2 var pointer */
      G__CHECKNONULL(0, 'E');
      G__CHECKNONULL(1, 'C');
      if (G__checkscanfarg("fscanf", libp, 2)) return(1);
      switch (libp->paran) {
         case 2:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1]))) ;
            break;
         case 3:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2]))) ;
            break;
         case 4:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3]))) ;
            break;
         case 5:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4]))) ;
            break;
         case 6:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5]))) ;
            break;
         case 7:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6]))) ;
            break;
         case 8:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7]))) ;
            break;
         case 9:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8]))) ;
            break;
         case 10:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8])
                               , G__int(libp->para[9]))) ;
            break;
         case 11:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8])
                               , G__int(libp->para[9])
                               , G__int(libp->para[10]))) ;
            break;
         case 12:
            G__letint(result7, 'i'
                      , fscanf((FILE *)G__int(libp->para[0])
                               , (char *)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8])
                               , G__int(libp->para[9])
                               , G__int(libp->para[10])
                               , G__int(libp->para[11]))) ;
            break;
         default:
            G__fprinterr(G__serr, "Limitation: fscanf only takes upto 12 arguments");
            G__genericerror((char*)NULL);
            break;
      }
      return(1);
   }

   if (523 == hash && strcmp(funcname, "scanf") == 0) {
      if (G__no_exec_compile) return(1);
      /* this is a fake function. not compatible with real func */
      /* para0 scan string , para1 format , para2 var pointer */
      G__CHECKNONULL(0, 'C');
      if (G__checkscanfarg("scanf", libp, 1)) return(1);
      switch (libp->paran) {
         case 1:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0]))) ;
            break;
         case 2:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1]))) ;
            break;
         case 3:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2]))) ;
            break;
         case 4:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3]))) ;
            break;
         case 5:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4]))) ;
            break;
         case 6:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5]))) ;
            break;
         case 7:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6]))) ;
            break;
         case 8:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7]))) ;
            break;
         case 9:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8]))) ;
            break;
         case 10:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8])
                               , G__int(libp->para[9]))) ;
            break;
         case 11:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char *)G__int(libp->para[0])
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7])
                               , G__int(libp->para[8])
                               , G__int(libp->para[9])
                               , G__int(libp->para[10]))) ;
            break;
         default:
            G__fprinterr(G__serr, "Limitation: scanf only takes upto 11 arguments");
            G__genericerror((char*)NULL);
            break;
      }
      return(1);
   }

   if (659 == hash && strcmp(funcname, "printf") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      /* para[0]:description, para[1~paran-1]: */
      G__charformatter(0, libp, temp);
      G__letint(result7, 'i', fprintf(G__intp_sout, "%s", temp));
      return(1);
   }

   if (761 == hash && strcmp(funcname, "fprintf") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'E');
      G__CHECKNONULL(1, 'C');
      /* parameter[0]:pointer ,parameter[1]:description, para[2~paran-1]: */
      G__charformatter(1, libp, temp);
      G__letint(result7, 'i',
                fprintf((FILE *)G__int(libp->para[0]), "%s", temp));
      return(1);
   }

   if (774 == hash && strcmp(funcname, "sprintf") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__CHECKNONULL(1, 'C');
      /* parameter[0]:charname ,para[1]:description, para[2~paran-1]: */
      G__charformatter(1, libp, temp);
      G__letint(result7, 'i',
                sprintf((char *)G__int(libp->para[0]), "%s", temp));
      return(1);
   }

   if (719 == hash && strcmp(funcname, "defined") == 0) {
      /********************************************************
       * modified for multiple vartable
       ********************************************************/
      G__letint(result7, 'i', G__defined_macro(libp->parameter[0]));
      return(1);
   }


   if (664 == hash && strcmp(funcname, "G__calc") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__storerewindposition();
      *result7 = G__calc_internal((char *)G__int(libp->para[0]));
      G__security_recover(G__serr);
      return(1);
   }


#ifdef G__SIGNAL
   if (525 == hash && strcmp(funcname, "alarm") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'h', alarm(G__int(libp->para[0])));
      return(1);
   }

   if (638 == hash && strcmp(funcname, "signal") == 0) {
      if (G__no_exec_compile) return(1);
      *result7 = G__null;
      if (G__int(libp->para[1]) == (long)SIG_IGN) {
         signal((int)G__int(libp->para[0]), SIG_IGN);
         return(1);
      }
      switch (G__int(libp->para[0])) {
         case SIGABRT:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGABRT, SIG_DFL);
            }
            else {
               G__SIGABRT = (char*)G__int(libp->para[1]);
               signal(SIGABRT, G__fsigabrt);
            }
            break;
         case SIGFPE:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGFPE, G__floatexception);
            }
            else {
               G__SIGFPE = (char*)G__int(libp->para[1]);
               signal(SIGFPE, G__fsigfpe);
            }
            break;
         case SIGILL:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGILL, SIG_DFL);
            }
            else {
               G__SIGILL = (char*)G__int(libp->para[1]);
               signal(SIGILL, G__fsigill);
            }
            break;
         case SIGINT:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGINT, G__breakkey);
            }
            else {
               G__SIGINT = (char*)G__int(libp->para[1]);
               signal(SIGINT, G__fsigint);
            }
            break;
         case SIGSEGV:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGSEGV, G__segmentviolation);
#ifdef SIGBUS
               signal(SIGBUS, G__buserror);
#endif
            }
            else {
               G__SIGSEGV = (char*)G__int(libp->para[1]);
               signal(SIGSEGV, G__fsigsegv);
#ifdef SIGBUS
               signal(SIGSEGV, G__fsigsegv);
#endif
            }
            break;
         case SIGTERM:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGTERM, SIG_DFL);
            }
            else {
               G__SIGTERM = (char*)G__int(libp->para[1]);
               signal(SIGTERM, G__fsigterm);
            }
            break;
#ifdef SIGHUP
         case SIGHUP:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGHUP, SIG_DFL);
            }
            else {
               G__SIGHUP = (char*)G__int(libp->para[1]);
               signal(SIGHUP, G__fsighup);
            }
            break;
#endif
#ifdef SIGQUIT
         case SIGQUIT:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGHUP, SIG_DFL);
            }
            else {
               G__SIGQUIT = (char*)G__int(libp->para[1]);
               signal(SIGQUIT, G__fsigquit);
            }
            break;
#endif
#ifdef SIGSTP
         case SIGTSTP:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGTSTP, SIG_DFL);
            }
            else {
               G__SIGTSTP = (char*)G__int(libp->para[1]);
               signal(SIGTSTP, G__fsigtstp);
            }
            break;
#endif
#ifdef SIGTTIN
         case SIGTTIN:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGTTIN, SIG_DFL);
            }
            else {
               G__SIGTTIN = (char*)G__int(libp->para[1]);
               signal(SIGTTIN, G__fsigttin);
            }
            break;
#endif
#ifdef SIGTTOU
         case SIGTTOU:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGTTOU, SIG_DFL);
            }
            else {
               G__SIGTTOU = (char*)G__int(libp->para[1]);
               signal(SIGTTOU, G__fsigttou);
            }
            break;
#endif
#ifdef SIGALRM
         case SIGALRM:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGALRM, SIG_DFL);
            }
            else {
               G__SIGALRM = (char*)G__int(libp->para[1]);
               signal(SIGALRM, G__fsigalrm);
            }
            break;
#endif
#ifdef SIGUSR1
         case SIGUSR1:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGUSR1, SIG_DFL);
            }
            else {
               G__SIGUSR1 = (char*)G__int(libp->para[1]);
               signal(SIGUSR1, G__fsigusr1);
            }
            break;
#endif
#ifdef SIGUSR2
         case SIGUSR2:
            if (G__int(libp->para[1]) == (long)SIG_DFL) {
               signal(SIGUSR2, SIG_DFL);
            }
            else {
               G__SIGUSR2 = (char*)G__int(libp->para[1]);
               signal(SIGUSR2, G__fsigusr2);
            }
            break;
#endif
         default:
            G__genericerror("Error: Unknown signal type");
            break;
      }
      return(1);
   }
#endif

   if (659 == hash && strcmp(funcname, "getopt") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(1, 'C');
      G__CHECKNONULL(2, 'C');
      if (first_getopt) {
         first_getopt = 0;
         G__globalvarpointer = (char*)(&optind);
         G__var_type = 'i';
         G__abortbytecode();
         G__getexpr("optind=1");
         G__asm_noverflow = 1;

         G__globalvarpointer = (char*)(&optarg);
         G__var_type = 'C';
         G__getexpr("optarg=");
      }
      G__letint(result7, 'c',
                (long)getopt((int)G__int(libp->para[0])
                             , (char **)G__int(libp->para[1])
                             , (char *)G__int(libp->para[2])));
      return(1);
   }

   if (hash == 868 && strcmp(funcname, "va_start") == 0) {
      if (G__no_exec_compile) return(1);
      G__va_start(libp->para[0]);
      *result7 = G__null;
      return(1);
   }

   if (hash == 621 && strcmp(funcname, "va_end") == 0) {
      if (G__no_exec_compile) return(1);
      G__va_end(libp->para[0]);
      *result7 = G__null;
      return(1);
   }

   if (hash == 1835 && strcmp(funcname, "G__va_arg_setalign") == 0) {
      if (G__no_exec_compile) return(1);
      G__va_arg_setalign((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }


   if (1093 == hash && strcmp(funcname, "G__loadfile") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'i'
                , (long)G__loadfile((char *)G__int(libp->para[0])));
      return(1);
   }

   if (1320 == hash && strcmp(funcname, "G__unloadfile") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'i', (long)G__unloadfile((char *)G__int(libp->para[0])));
      return(1);
   }

   if (1308 == hash && strcmp(funcname, "G__reloadfile") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__unloadfile((char *)G__int(libp->para[0]));
      G__letint(result7, 'i'
                , (long)G__loadfile((char *)G__int(libp->para[0])));
      return(1);
   }

   if (1882 == hash && strcmp(funcname, "G__set_smartunload") == 0) {
      if (G__no_exec_compile) return(1);
      *result7 = G__null;
      G__set_smartunload((int)G__int(libp->para[0]));
      return(1);
   }

   if (1655 == hash && strcmp(funcname, "G__charformatter") == 0) {
      if (G__no_exec_compile) {
         G__abortbytecode();
         return(1);
      }
      G__CHECKNONULL(1, 'C');
      G__charformatter((int)G__int(libp->para[0]), G__get_properties(G__p_local)->stackinfo.libp
                       , (char*)G__int(libp->para[1]));
      G__letint(result7, 'C', G__int(libp->para[1]));
      return(1);
   }

   if (1023 == hash && strcmp(funcname, "G__findsym") == 0) {
      if (G__no_exec_compile) {
         G__abortbytecode();
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'Y', (long)G__findsym((char*)G__int(libp->para[0])));
      return(1);
   }

   if (2210 == hash && strcmp(funcname, "G__set_sym_underscore") == 0) {
      if (G__no_exec_compile) {
         G__abortbytecode();
         return(1);
      }
      G__set_sym_underscore((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (2198 == hash && strcmp(funcname, "G__get_sym_underscore") == 0) {
      if (G__no_exec_compile) {
         G__abortbytecode();
         return(1);
      }
      G__letint(result7, 'i', (long)G__get_sym_underscore());
      return(1);
   }

   if (1162 == hash && strcmp(funcname, "G__IsInMacro") == 0) {
      if (G__no_exec_compile) {
         G__abortbytecode();
         return(1);
      }
      G__letint(result7, 'i', (long)G__IsInMacro());
      return(1);
   }

   if (1423 == hash && strcmp(funcname, "G__getmakeinfo") == 0) {
      if (G__no_exec_compile) {
         G__abortbytecode();
         return(1);
      }
      G__letint(result7, 'C', (long)G__getmakeinfo((char*)G__int(libp->para[0])));
      return(1);
   }

#ifndef G__OLDIMPLEMENTATION1485
   if (1249 == hash && strcmp(funcname, "G__fprinterr") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      /* para[0]:description, para[1~paran-1]: */
      G__charformatter(0, libp, temp);
      G__letint(result7, 'i', G__fprinterr(G__serr, "%s", temp));
      return(1);
   }
#endif

   /*********************************************************************
   * low priority 2
   *********************************************************************/
   if (569 == hash && strcmp(funcname, "qsort") == 0) {
      if (G__no_exec_compile) return(1);
#ifndef G__MASKERROR
      if (4 != libp->paran)
         G__printerror("qsort", 1, libp->paran);
#endif
      G__CHECKNONULL(3, 'Y');
      qsort((void *)G__int(libp->para[0])
            , (size_t)G__int(libp->para[1])
            , (size_t)G__int(libp->para[2])
            , (int (*)(const void*, const void*))G__int(libp->para[3])
            /* ,(int (*)(void *arg1,void *argv2))G__int(libp->para[3]) */
           );
      *result7 = G__null;
      return(1);
   }

   if (728 == hash && strcmp(funcname, "bsearch") == 0) {
      if (G__no_exec_compile) return(1);
#ifndef G__MASKERROR
      if (5 != libp->paran)
         G__printerror("bsearch", 1, libp->paran);
#endif
      G__CHECKNONULL(3, 'Y');
      bsearch((void *)G__int(libp->para[0])
              , (void *)G__int(libp->para[1])
              , (size_t)G__int(libp->para[2])
              , (size_t)G__int(libp->para[3])
              , (int (*)(const void*, const void*))G__int(libp->para[4])
             );
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "$read") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__textprocessing((FILE *)G__int(libp->para[0])));
      return(1);
   }

#if defined(G__REGEXP) || defined(G__REGEXP1)
   if (strcmp(funcname, "$regex") == 0) {
      if (G__no_exec_compile) return(1);
      switch (libp->paran) {
         case 1:
            G__letint(result7, 'i'
                      , (long)G__matchregex((char*)G__int(libp->para[0])
                                            , G__arg[0]));
            break;
         case 2:
            G__letint(result7, 'i'
                      , (long)G__matchregex((char*)G__int(libp->para[0])
                                            , (char*)G__int(libp->para[1])));
            break;
      }
      return(1);
   }
#endif

   if (strcmp(funcname, "$") == 0) {
      if (G__no_exec_compile) return(1);
      *result7 = G__getrsvd((int)G__int(libp->para[0]));
      return(1);
   }

   if (1631 == hash && strcmp(funcname, "G__exec_tempfile") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      *result7 = G__exec_tempfile((char *)G__int(libp->para[0]));
      return(1);
   }

#ifndef G__OLDIMPLEMENTATION1546
   if (1225 == hash && strcmp(funcname, "G__load_text") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__storerewindposition();
      G__letint(result7, 'C', (long)G__load_text((char *)G__int(libp->para[0])));
      G__security_recover(G__serr);
      return(1);
   }
#endif

   if (1230 == hash && strcmp(funcname, "G__exec_text") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__storerewindposition();
      *result7 = G__exec_text((char *)G__int(libp->para[0]));
      G__security_recover(G__serr);
      return(1);
   }

   if (1431 == hash && strcmp(funcname, "G__process_cmd") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__CHECKNONULL(1, 'C');
      G__CHECKNONULL(2, 'I');
      G__storerewindposition();
      *result7 = G__null;
      G__letint(result7, 'i', G__process_cmd((char*)G__int(libp->para[0])
                                             , (char*)G__int(libp->para[1])
                                             , (int*)G__int(libp->para[2])
                                             , (int*)G__int(libp->para[3])
                                             , (G__value*)G__int(libp->para[4])
                                            ));
      G__security_recover(G__serr);
      return(1);
   }

#ifndef G__OLDIMPLEMENTATION1867
   if (1670 == hash && strcmp(funcname, "G__exec_text_str") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__CHECKNONULL(1, 'C');
      G__storerewindposition();
      G__letint(result7, 'C',
                (long)G__exec_text_str((char *)G__int(libp->para[0]),
                                       (char *)G__int(libp->para[1])));
      G__security_recover(G__serr);
      return(1);
   }
#endif

   /*********************************************************************
   * low priority
   *********************************************************************/

   if (442 == hash && strcmp(funcname, "exit") == 0) {
      if (G__no_exec_compile) return(1);
      if (G__atexit) G__call_atexit(); /* Reduntant, also done in G__main() */
      G__return = G__RETURN_EXIT2;
      G__letint(result7, 'i', G__int(libp->para[0]));
      G__exitcode = result7->obj.i;
      return(1);
   }

   if (655 == hash && strcmp(funcname, "atexit") == 0) {
      if (G__no_exec_compile) return(1);
      if (G__int(libp->para[0]) == 0) {
         /* function wasn't registered */
         G__letint(result7, 'i', 1);
      }
      else {
         /* function was registered */
         G__atexit = (char *)G__int(libp->para[0]);
         G__letint(result7, 'i', 0);
      }
      return(1);
   }

   if (((hash == 466) && (strcmp(funcname, "ASSERT") == 0)) ||
         ((hash == 658) && (strcmp(funcname, "assert") == 0)) ||
         ((hash == 626) && (strcmp(funcname, "Assert") == 0))) {
      if (G__no_exec_compile) return(1);
      if (
         !G__bool(libp->para[0])
      ) {
         G__fprinterr(G__serr, "Assertion (%s) error: " , libp->parameter[0]);
         G__genericerror((char*)NULL);
         G__letint(result7, 'i', -1);
         G__pause();
      }
      else {
         G__letint(result7, 'i', 0);
      }
      return(1);
   }

   if (803 == hash && strcmp(funcname, "G__pause") == 0) {
      if (G__no_exec_compile) return(1);
      /* pause */
      G__letint(result7, 'i', (long)G__pause());
      return(1);
   }

   if (1443 == hash && strcmp(funcname, "G__set_atpause") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_atpause((void (*)(void))G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (1455 == hash && strcmp(funcname, "G__set_aterror") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_aterror((void (*)(void))G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (821 == hash && strcmp(funcname, "G__input") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'C', (long)G__input((char *)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__add_ipath") == 0) {
      if (G__no_exec_compile) return(1);
      G__add_ipath((char*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__delete_ipath") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__delete_ipath((char*)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__SetCINTSYSDIR") == 0) {
      if (G__no_exec_compile) return(1);
      G__SetCINTSYSDIR((char*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__set_eolcallback") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_eolcallback((void*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__set_history_size") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_history_size((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

#ifndef G__OLDIMPLEMENTATION1485
   if (strcmp(funcname, "G__set_errmsgcallback") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_errmsgcallback((void*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
#endif

   if (strcmp(funcname, "G__chdir") == 0) {
#if defined(G__WIN32) || defined(G__POSIX)
      char *stringb = (char*)G__int(libp->para[0]);
#endif
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
#if defined(G__WIN32)
      if (FALSE == SetCurrentDirectory(stringb))
         G__fprinterr(G__serr, "can not change directory to %s\n", stringb);
#elif defined(G__POSIX)
      if (0 != chdir(stringb))
         G__fprinterr(G__serr, "can not change directory to %s\n", stringb);
#endif
      *result7 = G__null;
      return(1);
   }

#ifdef G__SHMGLOBAL
   if (strcmp(funcname, "G__shminit") == 0) {
      if (G__no_exec_compile) return(1);
      G__shminit();
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__shmmalloc") == 0) {
      if (G__no_exec_compile) return(1);
      *result7 = G__null;
      G__letint(result7, 'E', (long)G__shmmalloc((int)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__shmcalloc") == 0) {
      if (G__no_exec_compile) return(1);
      *result7 = G__null;
      G__letint(result7, 'E', (long)G__shmcalloc((int)G__int(libp->para[0]), (int)G__int(libp->para[1])));
      return(1);
   }
#endif

   if (strcmp(funcname, "G__setautoconsole") == 0) {
      if (G__no_exec_compile) return(1);
      G__setautoconsole((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__AllocConsole") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__AllocConsole());
      return(1);
   }
   if (strcmp(funcname, "G__FreeConsole") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__FreeConsole());
      return(1);
   }

#ifdef G__TYPEINFO
   if (strcmp(funcname, "G__type2string") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'C' , (long)G__type2string((int)G__int(libp->para[0]),
                (int)G__int(libp->para[1]),
                (int)G__int(libp->para[2]),
                (int)G__int(libp->para[3]),
                (int)G__int(libp->para[4])));
      return(1);
   }

   if (strcmp(funcname, "G__typeid") == 0) {
      if (G__no_exec_compile) {
         G__value_typenum(*result7) = ::Reflex::Type::ByName("type_info");
         return(1);
      }
      G__letint(result7, 'u', (long)G__typeid((char*)G__int(libp->para[0])));
      result7->ref = result7->obj.i;
      G__value_typenum(*result7) = G__Dict::GetDict().GetType(*(int*)(result7->ref));
      return(1);
   }
#endif

   if (strcmp(funcname, "G__get_classinfo") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'l' , G__get_classinfo((char*)G__int(libp->para[0]),
                (int)G__int(libp->para[1])));
      return(1);
   }

   if (strcmp(funcname, "G__get_variableinfo") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'l' , G__get_variableinfo((char*)G__int(libp->para[0]),
                (long*)G__int(libp->para[1]),
                (long*)G__int(libp->para[2]),
                (int)G__int(libp->para[3])));
      return(1);
   }

   if (strcmp(funcname, "G__get_functioninfo") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'l' , G__get_functioninfo((char*)G__int(libp->para[0]),
                (long*)G__int(libp->para[1]),
                (long*)G__int(libp->para[2]),
                (int)G__int(libp->para[3])));
      return(1);
   }

   if (strcmp(funcname, "G__lasterror_filename") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'C', (long)G__lasterror_filename());
      return(1);
   }

   if (strcmp(funcname, "G__lasterror_linenum") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__lasterror_linenum());
      return(1);
   }

   if (strcmp(funcname, "G__loadsystemfile") == 0) {
      if (G__no_exec_compile) return(1);
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'i'
                , (long)G__loadsystemfile((char *)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__set_ignoreinclude") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_ignoreinclude((G__IgnoreInclude)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__ForceBytecodecompilation") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)::Cint::G__ForceBytecodecompilation((char*)G__int(libp->para[0])
                      , (char*)G__int(libp->para[1])
                                                           ));
      return(1);
   }

#ifndef G__SMALLOBJECT


#ifdef G__TRUEP2F
   if (strcmp(funcname, "G__p2f2funcname") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'C'
                , (long)G__p2f2funcname((void*)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__isinterpretedp2f") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__isinterpretedp2f((void*)G__int(libp->para[0])));
      return(1);
   }
#endif

   if (strcmp(funcname, "G__deleteglobal") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__deleteglobal((void*)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__display_tempobject") == 0) {
      if (G__no_exec_compile) return(1);
      *result7 = G__null;
      G__display_tempobject("");
      return(1);
   }

   if (strcmp(funcname, "G__cmparray") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__cmparray((short *)G__int(libp->para[0])
                                    , (short *)G__int(libp->para[1])
                                    , (int)G__int(libp->para[2])
                                    , (short)G__int(libp->para[3])));
      return(1);
   }

   if (strcmp(funcname, "G__setarray") == 0) {
      if (G__no_exec_compile) return(1);
      G__setarray((short *)G__int(libp->para[0])
                  , (int)G__int(libp->para[1])
                  , (short)G__int(libp->para[2])
                  , (char *)G__int(libp->para[3]));
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__deletevariable") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__deletevariable((char *)G__int(libp->para[0])));
      return(1);
   }

#ifndef G__NEVER
   if (strcmp(funcname, "G__split") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__split((char *)G__int(libp->para[0]),
                                 (char *)G__int(libp->para[1]),
                                 (int *)G__int(libp->para[2]),
                                 (char **)G__int(libp->para[3])));
      return(1);
   }

   if (strcmp(funcname, "G__readline") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__readline((FILE*)G__int(libp->para[0]),
                                    (char*)G__int(libp->para[1]),
                                    (char*)G__int(libp->para[2]),
                                    (int*)G__int(libp->para[3]),
                                    (char**)G__int(libp->para[4])));
      return(1);
   }
#endif

   if (strcmp(funcname, "G__setbreakpoint") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__setbreakpoint((char*)G__int(libp->para[0]), (char*)G__int(libp->para[1])));
      return(1);
   }

   if (strcmp(funcname, "G__tracemode") == 0 ||
         strcmp(funcname, "G__debugmode") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__tracemode((int)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__stepmode") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__stepmode((int)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__gettracemode") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__gettracemode());
      return(1);
   }

   if (strcmp(funcname, "G__getstepmode") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__getstepmode());
      return(1);
   }

#ifndef G__OLDIMPLEMENTATION2226
   if (strcmp(funcname, "G__setmemtestbreak") == 0) {
      if (G__no_exec_compile) return(1);
      *result7 = G__null;
      G__setmemtestbreak((int)G__int(libp->para[0]), (int)G__int(libp->para[1]));
      return(1);
   }
#endif

   if (strcmp(funcname, "G__optimizemode") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__optimizemode((int)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__getoptimizemode") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__getoptimizemode());
      return(1);
   }

   if (strcmp(funcname, "G__bytecodedebugmode") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__bytecodedebugmode((int)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__getbytecodedebugmode") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__getbytecodedebugmode());
      return(1);
   }

   if (strcmp(funcname, "G__clearerror") == 0) {
      if (G__no_exec_compile) return(1);
      G__return = G__RETURN_NON;
      G__security_error = G__NOERROR;
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__setbreakpoint") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__setbreakpoint((char*)G__int(libp->para[0])
                                         , (char*)G__int(libp->para[1])));
      return(1);
   }

   if (strcmp(funcname, "G__showstack") == 0) {
      if (G__no_exec_compile) return(1);
      G__showstack((FILE*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__graph") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__graph((double *)G__int(libp->para[0]),
                                 (double *)G__int(libp->para[1]),
                                 (int)G__int(libp->para[2]),
                                 (char*)G__int(libp->para[3]),
                                 (int)G__int(libp->para[4])));
      return(1);
   }

#ifndef G__NSEARCHMEMBER
   if (strcmp(funcname, "G__search_next_member") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'C'
                , (long)G__search_next_member((char *)G__int(libp->para[0])
                                              , (int)G__int(libp->para[1])));
      return(1);
   }

   if (strcmp(funcname, "G__what_type") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'Y', (long)G__what_type((char *)G__int(libp->para[0])
                , (char *)G__int(libp->para[1])
                , (char *)G__int(libp->para[2])
                , (char *)G__int(libp->para[3])
                                                ));
      return(1);
   }

   if (strcmp(funcname, "G__SetCatchException") == 0) {
      if (G__no_exec_compile) return(1);
      G__SetCatchException((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

#endif

#ifndef G__NSTOREOBJECT
   if (strcmp(funcname, "G__storeobject") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__storeobject((&libp->para[0]),
                (&libp->para[1])));
      return(1);
   }

   if (strcmp(funcname, "G__scanobject") == 0) {
      if (G__no_exec_compile) return(1);

      G__letint(result7, 'i', (long)G__scanobject((&libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__dumpobject") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__dumpobject((char *)G__int(libp->para[0])
                                      , (void *)G__int(libp->para[1])
                                      , (int)G__int(libp->para[2])));
      return(1);
   }

   if (strcmp(funcname, "G__loadobject") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__loadobject((char *)G__int(libp->para[0])
                                      , (void *)G__int(libp->para[1])
                                      , (int)G__int(libp->para[2])));
      return(1);
   }
#endif

   if (strcmp(funcname, "G__lock_variable") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__lock_variable((char *)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__unlock_variable") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__unlock_variable((char *)G__int(libp->para[0])));
      return(1);
   }

   if (strcmp(funcname, "G__dispvalue") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i'
                , (long)G__dispvalue((FILE*)libp->para[0].obj.i, &libp->para[1]));
      return(1);
   }

   if (strcmp(funcname, "G__set_class_autoloading_table") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_class_autoloading_table((char *)G__int(libp->para[0])
                                     , (char*)G__int(libp->para[1]));
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__defined") == 0) {
      if (G__no_exec_compile) return(1);
      G__letint(result7, 'i', (long)G__defined((char *)G__int(libp->para[0])));
      return(1);
   }


#endif /* G__SMALLOBJECT */



   if (strcmp(funcname, "G__set_alloclockfunc") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_alloclockfunc((void(*)())G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   if (strcmp(funcname, "G__set_allocunlockfunc") == 0) {
      if (G__no_exec_compile) return(1);
      G__set_allocunlockfunc((void(*)())G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }

   return(0);
}

//______________________________________________________________________________
static void G__printf_error()
{
   G__fprinterr(G__serr, "Limitation: printf string too long. Upto %d. Use fputs()", G__LONGLINE);
   G__genericerror(0);
}

//______________________________________________________________________________
static void G__sprintformatll(char* result, const char* fmt, void* p, char* buf)
{
   G__int64 *pll = (G__int64*)p;
   sprintf(buf, fmt, result, *pll);
   strcpy(result, buf);
}

//______________________________________________________________________________
static void G__sprintformatull(char* result, const char* fmt, void* p, char* buf)
{
   G__uint64 *pll = (G__uint64*)p;
   sprintf(buf, fmt, result, *pll);
   strcpy(result, buf);
}

//______________________________________________________________________________
static void G__sprintformatld(char* result, const char* fmt, void* p, char* buf)
{
   long double *pld = (long double*)p;
   sprintf(buf, fmt, result, *pld);
   strcpy(result, buf);
}

//______________________________________________________________________________
char* Cint::Internal::G__charformatter(int ifmt, G__param* libp, char* result)
{
   int ipara, ichar, lenfmt;
   int ionefmt = 0, fmtflag = 0;
   char onefmt[G__LONGLINE], fmt[G__LONGLINE];
   G__StrBuf pformat_sb(G__LONGLINE);
   char *pformat = pformat_sb;
   short dig = 0;
   int usedpara = 0;

   strcpy(pformat, (char *)G__int(libp->para[ifmt]));
   result[0] = '\0';
   ipara = ifmt + 1;
   lenfmt = strlen(pformat);
   for (ichar = 0;ichar <= lenfmt;ichar++) {
      switch (pformat[ichar]) {
         case '\0': /* end of the format */
            onefmt[ionefmt] = '\0';
            sprintf(fmt, "%%s%s", onefmt);
            sprintf(onefmt, fmt, result);
            strcpy(result, onefmt);
            ionefmt = 0;
            break;
         case 's': /* string */
            onefmt[ionefmt++] = pformat[ichar];
            if (fmtflag == 1) {
               onefmt[ionefmt] = '\0';
               if (libp->para[ipara].obj.i) {
                  if (strlen(onefmt) + strlen(result) + strlen((char*)G__int(libp->para[usedpara])) >= G__LONGLINE) {
                     G__printf_error();
                     return result;
                  }
                  sprintf(fmt, "%%s%s", onefmt);
                  sprintf(onefmt, fmt, result , (char *)G__int(libp->para[usedpara]));
                  strcpy(result, onefmt);
               }
               ipara++;
               ionefmt = 0;
               fmtflag = 0;
            }
            break;
         case 'c': /* char */
            onefmt[ionefmt++] = pformat[ichar];
            if (fmtflag == 1) {
               onefmt[ionefmt] = '\0';
               sprintf(fmt, "%%s%s", onefmt);
               sprintf(onefmt, fmt, result , (char)G__int(libp->para[usedpara]));
               strcpy(result, onefmt);
               ipara++;
               ionefmt = 0;
               fmtflag = 0;
            }
            break;
         case 'b': /* int */
            onefmt[ionefmt++] = pformat[ichar];
            if (fmtflag == 1) {
               onefmt[ionefmt-1] = 's';
               onefmt[ionefmt] = '\0';
               sprintf(fmt, "%%s%s", onefmt);
               G__logicstring(libp->para[usedpara], dig, onefmt);
               ipara++;
               sprintf(result, fmt, result, onefmt);
               ionefmt = 0;
            }
            break;
         case 'd': /* int */
         case 'i': /* int */
         case 'u': /* unsigned int */
         case 'o': /* octal */
         case 'x': /* hex */
         case 'X': /* HEX */
         case 'p': /* pointer */
            onefmt[ionefmt++] = pformat[ichar];
            if (fmtflag == 1) {
               onefmt[ionefmt] = '\0';
               sprintf(fmt, "%%s%s", onefmt);
               if ('n' == G__get_type(libp->para[usedpara])) {
                  G__value *pval = &libp->para[usedpara];
                  ipara++;
                  G__sprintformatll(result, fmt, &pval->obj.ll, onefmt);
               }
               else if ('m' == G__get_type(libp->para[usedpara])) {
                  G__value *pval = &libp->para[usedpara];
                  ipara++;
                  G__sprintformatull(result, fmt, &pval->obj.ull, onefmt);
               }
               else
                  if (
                     'u' == G__get_type(libp->para[usedpara])
                  ) {
                     char llbuf[100];
                     G__value *pval = &libp->para[usedpara];
                     ipara++;
                     ::Reflex::Type pval_type = G__value_typenum(*pval);
                     if (strcmp(pval_type.Name().c_str(), "G__longlong") == 0) {
                        sprintf(llbuf
                                , "G__printformatll((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
                                , (long)fmt, (long)onefmt, pval->obj.i);
                        G__getitem(llbuf);
                        strcat(result, fmt);
                     }
                     else if (strcmp(pval_type.Name().c_str(), "G__ulonglong") == 0) {
                        sprintf(llbuf
                                , "G__printformatull((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
                                , (long)fmt, (long)onefmt, pval->obj.i);
                        G__getitem(llbuf);
                        strcat(result, fmt);
                     }
                     else {
                        ++usedpara;
                        sprintf(onefmt, fmt , result, G__int(libp->para[usedpara]));
                        ipara++;
                        strcpy(result, onefmt);
                     }
                  }
                  else {
                     sprintf(onefmt, fmt , result, G__int(libp->para[usedpara]));
                     ipara++;
                     strcpy(result, onefmt);
                  }
               ionefmt = 0;
               fmtflag = 0;
            }
            break;
         case 'e': /* exponential form */
         case 'E': /* Exponential form */
         case 'f': /* floating */
         case 'g': /* floating or exponential */
         case 'G': /* floating or exponential */
            onefmt[ionefmt++] = pformat[ichar];
            if (fmtflag == 1) {
               onefmt[ionefmt] = '\0';
               sprintf(fmt, "%%s%s", onefmt);
               if ('q' == G__get_type(libp->para[usedpara])) {
                  G__value *pval = &libp->para[usedpara];
                  ipara++;
                  G__sprintformatld(result, fmt, &pval->obj.ld, onefmt);
               }
               else
                  if (
                     'u' == G__get_type(libp->para[usedpara])
                  ) {
                     char llbuf[100];
                     G__value *pval = &libp->para[usedpara];
                     ipara++;
                     if (strcmp(G__value_typenum(*pval).Name().c_str(), "G__longdouble") == 0) {
                        sprintf(llbuf
                                , "G__printformatld((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
                                , (long)fmt, (long)onefmt, pval->obj.i);
                        G__getitem(llbuf);
                        strcat(result, fmt);
                     }
                     else {
                        ++usedpara;
                        sprintf(onefmt, fmt, result, G__double(libp->para[usedpara]));
                        ipara++;
                        strcpy(result, onefmt);
                     }
                  }
                  else {
                     sprintf(onefmt, fmt , result, G__double(libp->para[usedpara]));
                     ipara++;
                     strcpy(result, onefmt);
                  }
               ionefmt = 0;
               fmtflag = 0;
            }
            break;
         case 'L': /* long double */
#ifdef G__OLDIMPLEMENTATION2189_YET
            if ('q' == G__get_type(libp->para[usedpara])) {
               G__value *pval = &libp->para[usedpara];
               ipara++;
               G__sprintformatld(fmt, onefmt, result, &pval->obj.ld);
            }
            break;
#endif
         case '0':
         case '1':
         case '2':
         case '3':
         case '4':
         case '5':
         case '6':
         case '7':
         case '8':
         case '9':
            dig = dig * 10 + pformat[ichar] - '0';
      case '#': // "alternate form"
         case '.':
         case '-':
         case '+':
         case 'l': /* long int */
         case 'h': /* short int unsinged int */
            onefmt[ionefmt++] = pformat[ichar];
            break;
         case '%':
            if (fmtflag == 0) {
               usedpara = ipara;
               fmtflag = 1;
            }
            else {
               fmtflag = 0;
            }
            onefmt[ionefmt++] = pformat[ichar];
            dig = 0;
            break;
         case '*': /* printf("%*s",4,"*"); */
            if (fmtflag == 1) {
               sprintf(onefmt + ionefmt, "%ld", G__int(libp->para[usedpara]));
               ipara++;
               usedpara++;
               ionefmt = strlen(onefmt);
            }
            else {
               onefmt[ionefmt++] = pformat[ichar];
            }
            break;
         case '$':
            if (fmtflag && dig) {
               usedpara = dig + ifmt;
               if (usedpara > libp->paran) {
                  /* this is an error! */
                  usedpara = ipara;
               }
               /* rewind the digit already printed */
               while (ionefmt >= 0 && isdigit(onefmt[ionefmt-1])) --ionefmt;
               dig = 0;
            }
            else {
               onefmt[ionefmt++] = pformat[ichar];
            }
            break;
         case ' ':
         case '\t' : /* tab */
         case '\n': /* end of line */
         case '\r': /* end of line */
         case '\f': /* end of line */
            if (fmtflag) {
               if ('%' != onefmt[ionefmt-1] && !isspace(onefmt[ionefmt-1])) fmtflag = 0;
               onefmt[ionefmt++] = pformat[ichar];
               break;
            }
         default:
            fmtflag = 0;
            onefmt[ionefmt++] = pformat[ichar];
            break;
      }
   }

   return(result);
}

//______________________________________________________________________________
extern "C" int G__tracemode(int tracemode)
{
   G__debug = tracemode;
   G__istrace = tracemode;
   G__setdebugcond();
   return(G__debug);
}

//______________________________________________________________________________
extern "C" int G__stepmode(int stepmode)
{
   switch (stepmode) {
      case 0:
         G__stepover = 0;
         G__step = 0;
         break;
      case 1:
         G__stepover = 0;
         G__step = 1;
         break;
      default:
         G__stepover = 3;
         G__step = 1;
         break;
   }
   G__setdebugcond();
   return(G__step);
}

//______________________________________________________________________________
extern "C" int G__gettracemode()
{
   return G__debug;
}

//______________________________________________________________________________
extern "C" int G__getstepmode()
{
   return G__step;
}

//______________________________________________________________________________
extern "C" int G__optimizemode(int optimizemode)
{
   G__asm_loopcompile = optimizemode;
   G__asm_loopcompile_mode = G__asm_loopcompile;
   return(G__asm_loopcompile);
}

//______________________________________________________________________________
extern "C" int G__getoptimizemode()
{
   return G__asm_loopcompile_mode;
}

//______________________________________________________________________________
int Cint::Internal::G__compiled_func_cxx(G__value* result7, char* funcname, struct G__param* libp, int hash)
{
   return G__compiled_func(result7, funcname, libp, hash, G__no_exec_compile, &G__p_tempbuf->obj, G__intp_sout, G__intp_sin);
}

