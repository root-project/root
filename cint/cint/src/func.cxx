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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "common.h"
#include "Api.h"

extern "C" {

void G__set_alloclockfunc(void(*foo)());
void G__set_allocunlockfunc(void(*foo)());

#if defined(G__WIN32)
#include <windows.h>
int getopt(int argc, char** argv, char* optlist);
#elif defined(G__POSIX)
#include <unistd.h> /* already included in G__ci.h */
#endif // G__WIN32, G__POSIX

#ifndef __CINT__
void G__display_tempobject(const char* action);
#endif // __CINT__

#ifndef __CINT__
int G__set_history_size(int s);
#endif // __CINT__

#ifndef __CINT__
int G__optimizemode(int optimizemode);
int G__getoptimizemode();
#endif // __CINT__

/******************************************************************
 * G__gen_PUSHSTROS_SETSTROS()
 ******************************************************************/
void G__gen_PUSHSTROS_SETSTROS()
{
   // --
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
         G__serr
         , "%3x,%3x: PUSHSTROS  %s:%d\n"
         , G__asm_cp
         , G__asm_dt
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__PUSHSTROS;
   G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
         G__serr
         , "%3x,%3x: SETSTROS  %s:%d\n"
         , G__asm_cp
         , G__asm_dt
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__SETSTROS;
   G__inc_cp_asm(1, 0);
}

/******************************************************************
 * G__dispvalue()
 ******************************************************************/
int G__dispvalue(FILE* fp, G__value* buf)
{
   if (buf) {
      fprintf(fp
              , "\nd=%g i=%ld reftype=%d type=%c tagn=%d typen=%d "
                "ref=%ld isconst=%d\n"
              , buf->obj.d
              , buf->obj.i
              , buf->obj.reftype.reftype
              , buf->type
              , buf->tagnum
              , buf->typenum
              , buf->ref
              , buf->isconst
             );
   }
   return 1;
}

static int G__exitcode = 0;

/******************************************************************
 * G__getexitcode()
 ******************************************************************/
int G__getexitcode()
{
   int exitcode = G__exitcode;
   G__exitcode = 0;
   return exitcode;
}

/******************************************************************
 * G__get_return()
 ******************************************************************/
int G__get_return(int* exitval)
{
   if (exitval) {
      *exitval = G__getexitcode();
   }
   return G__return;
}

/******************************************************************
 * G__bytecodedebugmode()
 ******************************************************************/
int G__bytecodedebugmode(int mode)
{
   G__asm_dbg = mode;
   return G__asm_dbg;
}

/******************************************************************
 * G__getoptmizemode()
 ******************************************************************/
int G__getbytecodedebugmode()
{
   return G__asm_dbg;
}

/******************************************************************
 * G__storelasterror()
 ******************************************************************/
void G__storelasterror()
{
   G__lasterrorpos = G__ifile;
}

/******************************************************************
 * G__lasterror_filename()
 ******************************************************************/
char* G__lasterror_filename()
{
   return G__lasterrorpos.name;
}

/******************************************************************
 * G__lasterror_linenum()
 ******************************************************************/
int G__lasterror_linenum()
{
   return G__lasterrorpos.line_number;
}

/******************************************************************
 * G__checkscanfarg()
 ******************************************************************/
int G__checkscanfarg(const char* fname, G__param* libp, int n)
{
   int result = 0;
   while (n < libp->paran) {
      if (islower(libp->para[n].type)) {
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

/******************************************************************
 ******************************************************************
 * Pointer to function evaluation function
 ******************************************************************
 ******************************************************************/

/******************************************************************
 * G__p2f_void_void()
 ******************************************************************/
void G__p2f_void_void(void* p2f)
{
   switch (G__isinterpretedp2f(p2f)) {
      case G__INTERPRETEDFUNC: {
            char* fname;
            fname = G__p2f2funcname(p2f);
            G__FastAllocString buf(fname);
            buf += "()";
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "(*p2f)() %s interpreted\n", buf());
            }
            G__calc_internal(buf);
         }
         break;
      case G__BYTECODEFUNC: {
            struct G__param param;
            G__value result;
#ifdef G__ANSI
            int(*ifm)(G__value*, char*, struct G__param*, int);
            ifm = (int(*)(G__value*, char*, struct G__param*, int))G__exec_bytecode;
#else // G__ANSI
            int(*ifm)();
            ifm = (int(*)())G__exec_bytecode;
#endif // G__ANSI
            param.paran = 0;
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "(*p2f)() bytecode\n");
            }
            (*ifm)(&result, (char*)p2f, &param, 0);
         }
         break;
      case G__COMPILEDINTERFACEMETHOD: {
            struct G__param param;
            G__value result;
#ifdef G__ANSI
            int(*ifm)(G__value*, char*, struct G__param*, int);
            ifm = (int(*)(G__value*, char*, struct G__param*, int))p2f;
#else // G__ANSI
            int(*ifm)();
            ifm = (int(*)())p2f;
#endif // G__ANSI
            param.paran = 0;
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "(*p2f)() compiled interface\n");
            }
            (*ifm)(&result, (char*)NULL, &param, 0);
         }
         break;
      case G__COMPILEDTRUEFUNC:
      case G__UNKNOWNFUNC: {
            void(*tp2f)();
            tp2f = (void(*)())p2f;
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "(*p2f)() compiled true p2f\n");
            }
            (*tp2f)();
         }
         break;
   }
}

/******************************************************************
 * G__set_atpause
 ******************************************************************/
void G__set_atpause(void(*p2f)())
{
   G__atpause = p2f;
}

/******************************************************************
 * G__set_aterror
 ******************************************************************/
void G__set_aterror(void(*p2f)())
{
   G__aterror = p2f;
}

/******************************************************************
 * G__set_emergencycallback
 ******************************************************************/
void G__set_emergencycallback(void(*p2f)())
{
   G__emergencycallback = p2f;
}

/******************************************************************
 * G__getindexedvalue()
 ******************************************************************/
static void G__getindexedvalue(G__value* result3, char* cindex)
{
   int size;
   int index;
   int len;
   G__FastAllocString sindex(cindex);
   char* p;
   p = strstr(sindex, "][");
   if (p) {
      *(p + 1) = 0;
      G__getindexedvalue(result3, sindex);
      p = strstr(cindex, "][");
      sindex = p + 1;
   }
   len = strlen(sindex);
#ifdef G__OLDIMPLEMENTATION424
   /* maybe unnecessary */
   if (len > 3 && '[' == sindex[0] && ']' == sindex[len-1]);
#endif // G__OLDIMPLEMENTATION424
   sindex[len-1] = '\0';
   if ('u' == result3->type) {
      struct G__param fpara;
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
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: OP2  '%c'  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , '+'
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      // Replace index on the data stack by (size * index).
      //G__letint(&G__asm_stack[G__asm_dt+1], 'i', (long) (size * index));
      G__asm_inst[G__asm_cp] = G__OP2;
      G__asm_inst[G__asm_cp+1] = (long) '+';
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   result3->obj.i += (size * index);
   *result3 = G__tovalue(*result3);
}

/******************************************************************
 * G__explicit_fundamental_typeconv()
 ******************************************************************/
int G__explicit_fundamental_typeconv(char* funcname, int hash, G__param* libp, G__value* presult3)
{
   int flag = 0;
   switch (hash) {
      case 3:
         if (strcmp(funcname, "int") == 0) {
            presult3->type = 'i';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(int*)presult3->ref = (int)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 4:
         if (strcmp(funcname, "char") == 0) {
            presult3->type = 'c';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(char*)presult3->ref = (char)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "long") == 0) {
            presult3->type = 'l';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "int*") == 0) {
            presult3->type = 'I';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "bool") == 0
                 ) {
            presult3->type = 'g';
            presult3->obj.i = G__int(libp->para[0]) ? 1 : 0;
            if (presult3->ref) {
               *(unsigned char*)presult3->ref =
                  (unsigned char) presult3->obj.i ? 1 : 0;
            }
            flag = 1;
         }
         break;
      case 5:
         if (strcmp(funcname, "short") == 0) {
            presult3->type = 's';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(short*)presult3->ref = (short)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "float") == 0) {
            presult3->type = 'f';
            presult3->obj.d = G__double(libp->para[0]);
            if (presult3->ref) {
               *(float*)presult3->ref = (float)presult3->obj.d;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "char*") == 0) {
            presult3->type = 'C';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "long*") == 0) {
            presult3->type = 'L';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "void*") == 0) {
            presult3->type = 'Y';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 6:
         if (strcmp(funcname, "double") == 0) {
            presult3->type = 'd';
            presult3->obj.d = G__double(libp->para[0]);
            if (presult3->ref) {
               *(double*)presult3->ref = (double)presult3->obj.d;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "short*") == 0) {
            presult3->type = 'S';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "float*") == 0) {
            presult3->type = 'F';
            presult3->obj.d = G__double(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 7:
         if (strcmp(funcname, "double*") == 0) {
            presult3->type = 'd';
            presult3->obj.d = G__double(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 8:
         if (strcmp(funcname, "unsigned") == 0) {
            presult3->type = 'h';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         if (strcmp(funcname, "long long") == 0) {
            presult3->type = 'n';
            presult3->obj.ll = G__Longlong(libp->para[0]);
            if (presult3->ref) {
               *(G__int64*)presult3->ref = presult3->obj.ll;
            }
            flag = 1;
         }
         break;
      case 9:
         if (strcmp(funcname, "unsigned*") == 0) {
            presult3->type = 'H';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         if (strcmp(funcname, "long long*") == 0) {
            presult3->type = 'N';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         if (strcmp(funcname, "long long") == 0) {
            presult3->type = 'n';
            presult3->obj.ll = G__Longlong(libp->para[0]);
            if (presult3->ref) {
               *(G__int64*)presult3->ref = presult3->obj.ll;
            }
            flag = 1;
         }
         break;
      case 10:
         if (strcmp(funcname, "long double") == 0) {
            presult3->type = 'n';
            presult3->obj.ld = G__Longdouble(libp->para[0]);
            if (presult3->ref) {
               *(long double*)presult3->ref = presult3->obj.ld;
            }
            flag = 1;
         }
         break;
      case 11:
         if (strcmp(funcname, "unsigned int") == 0) {
            presult3->type = 'h';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 12:
         if (strcmp(funcname, "unsignedchar") == 0) {
            presult3->type = 'b';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned char*)presult3->ref = (unsigned char)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "unsigned long") == 0) {
            presult3->type = 'k';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned long*)presult3->ref = (unsigned long)presult3->obj.i;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "unsigned int") == 0) {
            presult3->type = 'h';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 13:
         if (strcmp(funcname, "unsigned short") == 0) {
            presult3->type = 'r';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned short*)presult3->ref = (unsigned short)presult3->obj.i;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "unsigned char") == 0) {
            presult3->type = 'b';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned char*)presult3->ref = (unsigned char)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "unsigned long") == 0) {
            presult3->type = 'k';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned long*)presult3->ref = (unsigned long)presult3->obj.i;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "unsigned int*") == 0) {
            presult3->type = 'H';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 14:
         if (strcmp(funcname, "unsigned short") == 0) {
            presult3->type = 'r';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(unsigned short*)presult3->ref =
                  (unsigned short) presult3->obj.i;
            }
            flag = 1;
         }
         else if (strcmp(funcname, "unsigned char*") == 0) {
            presult3->type = 'B';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
            break;
         }
         else if (strcmp(funcname, "unsigned long*") == 0) {
            presult3->type = 'K';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 15:
         if (strcmp(funcname, "unsigned short*") == 0) {
            presult3->type = 'R';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 16:
         if (strcmp(funcname, "unsigned long long") == 0) {
            presult3->type = 'm';
            presult3->obj.ull = G__ULonglong(libp->para[0]);
            if (presult3->ref) {
               *(G__uint64*)presult3->ref = presult3->obj.ull;
            }
            flag = 1;
         }
         break;
      case 17:
         if (strcmp(funcname, "unsigned long long*") == 0) {
            presult3->type = 'M';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         break;
      case 18:
         if (strcmp(funcname, "unsigned long long") == 0) {
            presult3->type = 'm';
            presult3->obj.ull = G__ULonglong(libp->para[0]);
            if (presult3->ref) {
               *(G__uint64*)presult3->ref = presult3->obj.ull;
            }
            flag = 1;
         }
         break;
      case 19:
         if (strcmp(funcname, "unsigned long long*") == 0) {
            presult3->type = 'M';
            presult3->obj.i = G__int(libp->para[0]);
            if (presult3->ref) {
               *(long*)presult3->ref = (long)presult3->obj.i;
            }
            flag = 1;
         }
         break;
   }
   if (flag) {
      presult3->tagnum = -1;
      presult3->typenum = -1;
#ifdef G__ASM
      if (G__asm_noverflow) {
         // We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "%3x,%3x: CAST to %c  %s:%d\n"
               , G__asm_cp
               , G__asm_dt
               , presult3->type
               , __FILE__
               , __LINE__
            );
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__CAST;
         G__asm_inst[G__asm_cp+1] = presult3->type;
         G__asm_inst[G__asm_cp+2] = presult3->typenum;
         G__asm_inst[G__asm_cp+3] = presult3->tagnum;
         G__asm_inst[G__asm_cp+4] = G__PARANORMAL;
         G__inc_cp_asm(5, 0);
      }
#endif // G__ASM
      // --
   }
#ifndef G_OLDIMPLEMENTATION1128
   if (flag && 'u' == libp->para[0].type) {
      int xtype = presult3->type;
      int xreftype = 0;
      int xisconst = 0;
      *presult3 = libp->para[0];
      G__fundamental_conversion_operator(xtype, -1 , -1 , xreftype, xisconst
                                         , presult3, 0);
   }
#endif // G_OLDIMPLEMENTATION1128
   return flag;
}

/******************************************************************
 * void G__gen_addstros()
 ******************************************************************/
void G__gen_addstros(int addstros)
{
   // --
#ifdef G__ASM
   if (G__asm_noverflow) {
      // We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg)
         G__fprinterr(
            G__serr
            , "%3x,%3x: ADDSTROS %d  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , addstros
            , __FILE__
            , __LINE__
         );
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__ADDSTROS;
      G__asm_inst[G__asm_cp+1] = addstros;
      G__inc_cp_asm(2, 0);
   }
#endif
   // --
}

#ifdef G__PTR2MEMFUNC
/******************************************************************
 * G__pointer2memberfunction()
 ******************************************************************/
G__value G__pointer2memberfunction(char* parameter0, char* parameter1, int* known3)
{
   G__FastAllocString buf(parameter0);
   char* mem;
   G__value res;
   const char* opx;
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
   if (!res.type) {
      G__fprinterr(G__serr, "Error: Pointer to member function %s not found"
                   , parameter0);
      G__genericerror(0);
      return G__null;
   }
   if (!res.obj.i || !*(char**)res.obj.i) {
      G__fprinterr(G__serr, "Error: Pointer to member function %s is NULL", parameter0);
      G__genericerror(0);
      return G__null;
   }
   // For the time being, pointer to member function can only be handed as
   // function name
   G__FastAllocString buf2(*(char**)res.obj.i);
   G__FastAllocString expr(G__LONGLINE);
   expr = buf;
   expr += opx;
   expr += buf2;
   expr += parameter1;
   G__abortbytecode();
   return G__getvariable(expr, known3, &G__global, G__p_local);
}
#endif // G__PTR2MEMFUNC

/******************************************************************
 * G__pointerReference()
 ******************************************************************/
G__value G__pointerReference(char* item, G__param* libp, int* known3)
{
   G__value result3;
   int i, j;
   int store_tagnum = G__tagnum;
   int store_typenum = G__typenum;
   long store_struct_offset = G__store_struct_offset;
   result3 = G__getexpr(item);
   if (0 == result3.type) {
      return(G__null);
   }
   *known3 = 1;
   if (2 == libp->paran && strstr(libp->parameter[1], "][")) {
      G__FastAllocString arg(libp->parameter[1]);
      char* p = arg;
      i = 1;
      while (*p) {
         int k = 0;
         if (*p == '[') {
            ++p;
         }
         while (*p && *p != ']') {
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
   for (i = 1; i < libp->paran; i++) {
      G__FastAllocString arg(libp->parameter[i]);
      if ('[' == arg[0]) {
         j = 0;
         while (arg[++j] && ']' != arg[j]) {
            arg[j-1] = arg[j];
         }
         arg[j-1] = 0;
      }
      if ('u' == result3.type) { /* operator[] overloading */
         G__FastAllocString expr(G__ONELINE);
         /* Set member function environment */
         G__tagnum = result3.tagnum;
         G__typenum = result3.typenum;
         G__store_struct_offset = result3.obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // We are generating bytecode.
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
         /* call operator[] */
         *known3 = 0;
         expr.Format("operator[](%s)", arg());
         result3 = G__getfunction(expr, known3, G__CALLMEMFUNC);
         /* Restore environment */
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // We are generating bytecode.
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
      else if (isupper(result3.type)) {
         G__value varg;
         varg = G__getexpr(arg);
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

/******************************************************************
 * G__additional_parenthesis()
 ******************************************************************/
int G__additional_parenthesis(G__value* presult, struct G__param* libp)
{
   G__FastAllocString buf(G__LONGLINE);
   int known;
   int store_tagnum = G__tagnum;
   long store_struct_offset = G__store_struct_offset;
   if (-1 == presult->tagnum) {
      return 0;
   }
   G__tagnum = presult->tagnum;
   G__store_struct_offset = presult->obj.i;
   buf.Format("operator()%s", libp->parameter[1]);
   *presult = G__getfunction(buf, &known, G__CALLMEMFUNC);
   G__tagnum = store_tagnum;
   G__store_struct_offset = store_struct_offset;
   return known;
}

} // extern "C"

/******************************************************************
* G__rename_templatefunc()
******************************************************************/
char* G__rename_templatefunc(G__FastAllocString& funcname)
{
   char* ptmplt ;
   ptmplt = strchr(funcname, '<');
   if (ptmplt) {
      *ptmplt = 0;
      if (G__defined_templatefunc(funcname)) {
         *ptmplt = 0;
      }
      else {
         if (G__defined_templateclass(funcname)) {
            *ptmplt = 0;
         } else {
            *ptmplt = '<';
            ptmplt = (char*)0;
         }
      }
   }
   if (ptmplt) {
      G__FastAllocString funcname2(funcname);
      G__FastAllocString buf(G__ONELINE);
      G__FastAllocString buf2(20);
      int typenum, tagnum, len;
      int ip = 1;
      int c;
      funcname2 += "<";
      do {
         c = G__getstream_template(ptmplt, &ip, buf, 0, ",>");
         len = strlen(buf) - 1;
         while ('*' == buf[len] || '&' == buf[len]) {
            --len;
         }
         ++len;
         if (buf[len]) {
            buf2 = buf + len;
            buf[len] = 0;
         }
         else {
            buf2[0] = 0;
         }
         typenum = G__defined_typename(buf);
         if (-1 != typenum) {
            buf = G__fulltypename(typenum);
         }
         else {
            tagnum = G__defined_tagname(buf, 4);
            if (-1 != tagnum) {
               buf = G__fulltagname(tagnum, 1);
            }
         }
         buf += buf2;
         funcname2 += buf;
         if (funcname2[strlen(funcname2)-1] == '>' && c == '>') {
            buf2[0] = ' ';
            buf2[1] = c;
            buf2[2] = 0;
         }
         else {
            buf2[0] = c;
            buf2[1] = 0;
         }
         funcname2 += buf2;
      }
      while (c != '>');
      funcname2 += ptmplt + ip;
      funcname = funcname2;
   }
   return funcname;
}

extern "C" {

/******************************************************************
 * G__operatorfunction()
 ******************************************************************/
G__value G__operatorfunction(G__value* presult, const char* item, int* known3,
                             G__FastAllocString& result7, const char* funcname)
{
   G__value result3 = G__null;
   struct G__param fpara;
   int ig15 = 0;
   int ig35 = 0;
   int nest = 0;
   int double_quote = 0, single_quote = 0;
   int lenitem = strlen(item);
   int base1 = 0;
   int nindex = 0;
   int itmp;
   fpara.paran = 0;
   fpara.para[0].type = 0;
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
         while (
            (
               (
                  (item[ig15] != ',') &&
                  (item[ig15] != ')')
               ) ||
               (nest > 0) ||
               (tmpltnest > 0) ||
               (single_quote > 0) ||
               (double_quote > 0)
            ) &&
            (ig15 < lenitem)
         ) {
            switch (item[ig15]) {
               case '"' : /* double quote */
                  if (single_quote == 0) {
                     double_quote ^= 1;
                  }
                  break;
               case '\'' : /* single quote */
                  if (double_quote == 0) {
                     single_quote ^= 1;
                  }
                  break;
               case '(':
               case '[':
               case '{':
                  if ((double_quote == 0) && (single_quote == 0)) {
                     nest++;
                  }
                  break;
               case ')':
               case ']':
               case '}':
                  if ((double_quote == 0) && (single_quote == 0)) {
                     nest--;
                  }
                  break;
               case '\\':
                  result7.Set(ig35++, item[ig15++]);
                  break;
               case '<':
                  if (double_quote == 0 && single_quote == 0) {
                     result7.Set(ig35, 0);
                     if (0 == strcmp(result7, "operator") ||
                           tmpltnest ||
                           G__defined_templateclass(result7)) {
                        ++tmpltnest;
                     }
                  }
                  break;
               case '>':
                  if (double_quote == 0 && single_quote == 0) {
                     if (tmpltnest) {
                        --tmpltnest;
                     }
                  }
                  break;
            }
            result7.Set(ig35++, item[ig15++]);
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
            if (('-' == item[ig15+1] && '>' == item[ig15+2]) || '.' == item[ig15+1]) {
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
               result7.Set(ig35, 0);
               G__strlcpy(fpara.parameter[fpara.paran], result7, G__ONELINE);
               if (ig35) {
                  fpara.parameter[++fpara.paran][0] = '\0';
               }
               for (itmp = 0; itmp < fpara.paran; itmp++) {
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
         result7.Set(ig35, 0);
         if (ig35 < G__ONELINE) {
            G__strlcpy(fpara.parameter[fpara.paran], result7, G__ONELINE);
         }
         else {
            G__strlcpy(fpara.parameter[fpara.paran], "@", G__ONELINE);
            fpara.para[fpara.paran] = G__getexpr(result7);
         }
         if (ig35) {
            fpara.parameter[++fpara.paran][0] = '\0';
         }
      }
   }
#ifdef G__ASM
   if (G__asm_noverflow) {
      G__gen_PUSHSTROS_SETSTROS();
   }
#endif // G__ASM
   for (itmp = 0; itmp < fpara.paran; itmp++) {
      fpara.para[itmp] = G__getexpr(fpara.parameter[itmp]);
   }
   G__parenthesisovldobj(&result3, presult, "operator()", &fpara, 1);
   return result3;
}

/******************************************************************
 * G__value G__getfunction_libp
 ******************************************************************/
G__value G__getfunction_libp(const char* item, G__FastAllocString& funcname,
                             G__param* libp, int* known3, int memfunc_flag)
{
   G__value result3;
   G__FastAllocString result7(G__LONGLINE);
   int ig15, ipara;
   static struct G__param* p2ffpara = (struct G__param*)NULL;
   int hash;
   int funcmatch;
   int i, classhash;
   long store_struct_offset;
   int store_tagnum;
   int store_exec_memberfunc;
   int store_asm_noverflow;
   int store_memberfunc_tagnum;
   long store_memberfunc_struct_offset;
   int store_def_tagnum, store_tagdefining;
   int tempstore;
   char* pfparam;
   struct G__var_array* var;
   int store_cp_asm = 0;
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_exec_memberfunc = G__exec_memberfunc;
   store_asm_noverflow = G__asm_noverflow;
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   *known3 = 1; /* temporary solution */
   /* scope operator ::f() , A::B::f()
    * note,
    *    G__exec_memberfunc restored at return memfunc_flag is local,
    *   there should be no problem modifying these variables.
    *    store_struct_offset and store_tagnum are only used in the
    *   explicit type conversion section.  It is OK to use them here
    *   independently.
    */
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_def_tagnum = G__def_tagnum;
   store_tagdefining = G__tagdefining;
   switch (G__scopeoperator(funcname, &hash, &G__store_struct_offset, &G__tagnum)) {
      case G__GLOBALSCOPE: /* global scope */
         G__exec_memberfunc = 0;
         memfunc_flag = G__TRYNORMAL;
         G__def_tagnum = -1;
         G__tagdefining = -1;
         break;
      case G__CLASSSCOPE: /* class scope */
         memfunc_flag = G__CALLSTATICMEMFUNC;
         /* This looks very risky */
         G__def_tagnum = -1;
         G__tagdefining = -1;
         G__exec_memberfunc = 1;
         G__memberfunc_tagnum = G__tagnum;
         break;
      default:
         G__hash(funcname, hash, i);
         break;
   }
#ifdef G__DUMPFILE
   /***************************************************************
    * dump that a function is called
    ***************************************************************/
   if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
      for (ipara = 0; ipara < G__dumpspace; ipara++) {
         fprintf(G__dumpfile, " ");
      }
      fprintf(G__dumpfile, "%s(", funcname());
      for (ipara = 1; ipara <= libp->paran; ipara++) {
         if (ipara != 1) {
            fprintf(G__dumpfile, ",");
         }
         G__valuemonitor(libp->para[ipara-1], result7);
         fprintf(G__dumpfile, "%s", result7());
      }
      fprintf(G__dumpfile, ");/*%s %d,%lx %lx*/\n"
              , G__ifile.name, G__ifile.line_number
              , store_struct_offset, G__store_struct_offset);
      G__dumpspace += 3;
   }
#endif // G__DUMPFILE
   /********************************************************************
    * Begin Loop to resolve overloaded function
    ********************************************************************/
   for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; funcmatch++) {
      // -- Search for interpreted member function.
      /***************************************************************
       * if(G__exec_memberfunc)     ==>  memfunc();
       * G__TRYNORMAL!=memfunc_flag ==>  a.memfunc();
       ***************************************************************/
      if (G__exec_memberfunc || G__TRYNORMAL != memfunc_flag) {
         int local_tagnum;
         if (G__exec_memberfunc && -1 == G__tagnum) {
            local_tagnum = G__memberfunc_tagnum;
         }
         else {
            local_tagnum = G__tagnum;
         }
         if (-1 != G__tagnum) {
            G__incsetup_memfunc(G__tagnum);
         }
         if (-1 != local_tagnum && G__interpret_func(&result3, funcname, libp, hash
               , G__struct.memfunc[local_tagnum]
               , funcmatch, memfunc_flag) == 1) {
#ifdef G__DUMPFILE
            if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
               G__dumpspace -= 3;
               for (ipara = 0; ipara < G__dumpspace; ipara++) {
                  fprintf(G__dumpfile, " ");
               }
               G__valuemonitor(result3, result7);
               fprintf(G__dumpfile, "/* return(inp) %s.%s()=%s*/\n"
                       , G__tagnum >= 0 ? G__struct.name[G__tagnum] : "unknown class", funcname(), result7());
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
            G__setclassdebugcond(G__memberfunc_tagnum, 0);
            return(result3);
         }
#define G__OLDIMPLEMENTATION1159
      }
      /***************************************************************
       * If memberfunction is called explicitly by clarifying scope
       * don't examine global function and exit from G__getfunction().
       * There are 2 cases                   G__exec_memberfunc
       *   obj.memfunc();                            1
       *   X::memfunc();                             1
       *    X();              constructor            2
       *   ~X();              destructor             2
       * If G__exec_memberfunc==2, don't display error message.
       ***************************************************************/
      /* If searching only member function */
      if (memfunc_flag
            && (G__store_struct_offset || G__CALLSTATICMEMFUNC != memfunc_flag)
         ) {
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         /* If the last resolution of overloading failed */
         if (funcmatch == G__USERCONV) {
            if (memfunc_flag == G__TRYDESTRUCTOR) {
               // destructor for base calss and class members
#ifdef G__ASM
#ifdef G__SECURITY
               store_asm_noverflow = G__asm_noverflow;
               if (G__security & G__SECURE_GARBAGECOLLECTION) {
                  G__abortbytecode();
               }
#endif // G__SECURITY
#endif // G__ASM
#ifdef G__VIRTUALBASE
               if (
                  (G__tagnum != -1) &&
                  (G__struct.iscpplink[G__tagnum] != G__CPPLINK)
               ) {
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
                  case G__TRYIMPLICITCONSTRUCTOR: {
                        /* constructor for base class and class members default
                         * constructor only */
                        int store2_exec_memberfunc = G__exec_memberfunc;
                        G__exec_memberfunc = 1;
#ifdef G__VIRTUALBASE
                        if (G__tagnum != -1 && G__CPPLINK != G__struct.iscpplink[G__tagnum]) {
                           G__baseconstructor(0 , (struct G__baseparam*)NULL);
                        }
#else // G__VIRTUALBASE
                        G__baseconstructor(0 , (struct G__baseparam*)NULL);
#endif // G__VIRTUALBASE
                        G__exec_memberfunc = store2_exec_memberfunc;
                        break;
                     }
               }
            }
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;
            *known3 = 0;
            switch (memfunc_flag) {
               case G__CALLMEMFUNC:
                  if (G__parenthesisovld(&result3, funcname, libp, G__CALLMEMFUNC)) {
                     *known3 = 1;
                     if (G__store_struct_offset != store_struct_offset) {
                        G__gen_addstros(store_struct_offset - G__store_struct_offset);
                     }
                     G__store_struct_offset = store_struct_offset;
                     G__tagnum = store_tagnum;
                     G__def_tagnum = store_def_tagnum;
                     G__tagdefining = store_tagdefining;
                     return(result3);
                  }
                  if ('~' == funcname[0]) {
                     *known3 = 1;
                     return(G__null);
                  }
                  /*
                   * Search template function
                   */
                  G__exec_memberfunc = 1;
                  G__memberfunc_tagnum = G__tagnum;
                  G__memberfunc_struct_offset = G__store_struct_offset;
                  if ((G__EXACT == funcmatch || G__USERCONV == funcmatch) &&
                        G__templatefunc(&result3, funcname, libp, hash, funcmatch) == 1) {
#ifdef G__DUMPFILE
                     if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
                        G__dumpspace -= 3;
                        for (ipara = 0; ipara < G__dumpspace; ipara++) {
                           fprintf(G__dumpfile, " ");
                        }
                        G__valuemonitor(result3, result7);
                        fprintf(G__dumpfile , "/* return(lib) %s()=%s */\n"
                                , funcname(), result7());
                     }
#endif // G__DUMPFILE
                     G__exec_memberfunc = store_exec_memberfunc;
                     G__memberfunc_tagnum = store_memberfunc_tagnum;
                     G__memberfunc_struct_offset = store_memberfunc_struct_offset;
                     *known3 = 1;
                     return(result3);
                  }
                  G__exec_memberfunc = store_exec_memberfunc;
                  G__memberfunc_tagnum = store_memberfunc_tagnum;
                  G__memberfunc_struct_offset = store_memberfunc_struct_offset;
                  // We did not find the function as a regular member,
                  // let's try as a constructor.
               case G__CALLCONSTRUCTOR:
                  if (G__NOLINK > G__globalcomp) {
                     break;
                  }
                  if (!G__no_exec_compile || G__asm_noverflow) {
                     if (0 == G__const_noerror)
                        G__fprinterr(G__serr, "Error: Can't call %s::%s in current scope"
                                     , G__tagnum > 0 ? G__struct.name[G__tagnum] : "unknown class", item);
                     G__genericerror((char*)NULL);
                  }
                  store_exec_memberfunc = G__exec_memberfunc;
                  G__exec_memberfunc = 1;
                  if (0 == G__const_noerror
                        && (!G__no_exec_compile || G__asm_noverflow)
                     ) {
                     G__fprinterr(G__serr, "Possible candidates are...\n");
                     {
                        G__FastAllocString itemtmp(G__LONGLINE);
                        itemtmp.Format("%s::%s", G__tagnum != -1 ? G__struct.name[G__tagnum] : "unknown scope", funcname());
                        G__display_proto_pretty(G__serr, itemtmp, 1);
                     }
                  }
                  G__exec_memberfunc = store_exec_memberfunc;
            }
#ifdef G__DUMPFILE
            if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
               G__dumpspace -= 3;
            }
#endif // G__DUMPFILE
            if (G__store_struct_offset != store_struct_offset) {
               G__gen_addstros(store_struct_offset - G__store_struct_offset);
            }
            G__store_struct_offset = store_struct_offset;
            G__tagnum = store_tagnum;
            G__def_tagnum = store_def_tagnum;
            G__tagdefining = store_tagdefining;
            if (libp->paran && 'u' == libp->para[0].type &&
                  (G__TRYCONSTRUCTOR == memfunc_flag ||
                   G__TRYIMPLICITCONSTRUCTOR == memfunc_flag)
               ) {
               /* in case of copy constructor not found */
               return libp->para[0];
            }
            return G__null;
         }
         /* ELSE next level overloaded function resolution */
         continue;
      }
      /***************************************************************
       * reset G__exec_memberfunc for global function.
       * Original value(store_exec_memberfunc) is restored when exit
       * from this function
       ***************************************************************/
      tempstore = G__exec_memberfunc;
      G__exec_memberfunc = 0;
      /***************************************************************
       * search for interpreted global function
       ***************************************************************/
      if (G__interpret_func(&result3, funcname, libp, hash, G__p_ifunc
                            , funcmatch, G__TRYNORMAL) == 1) {
#ifdef G__DUMPFILE
         if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ipara++) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile , "/* return(inp) %s()=%s*/\n" , funcname(), result7());
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         G__setclassdebugcond(G__memberfunc_tagnum, 0);
         return(result3);
      }
      G__exec_memberfunc = tempstore;
      /* there is no function overload resolution after this point,
       * thus, if not found in G__EXACT trial, there is no chance to
       * find matched function in consequitive search
       */
      if (G__USERCONV == funcmatch) {
         goto templatefunc;
      }
      if (G__EXACT != funcmatch) {
         continue;
      }
      /***************************************************************
       * search for compiled(archived) function
       ***************************************************************/
      if (G__compiled_func(&result3, funcname, libp, hash) == 1) {
         // --
#ifdef G__ASM
         if (G__asm_noverflow) {
            /****************************************
             * LD_FUNC (compiled)
             ****************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                  G__serr
                  , "%3x,%3x: LD_FUNC compiled '%s' paran: %d  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , funcname()
                  , libp->paran
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_FUNC;
            G__asm_inst[G__asm_cp+1] = (long) &G__asm_name[G__asm_name_p];
            G__asm_inst[G__asm_cp+2] = hash;
            G__asm_inst[G__asm_cp+3] = libp->paran;
            G__asm_inst[G__asm_cp+4] = (long) G__compiled_func;
            G__asm_inst[G__asm_cp+5] = 0;
            G__asm_inst[G__asm_cp+6] = (long) G__p_ifunc;
            G__asm_inst[G__asm_cp+7] = -1;
            if (
               G__strlcpy(
                    G__asm_name + G__asm_name_p
                  , funcname
                  , G__ASM_FUNCNAMEBUF - G__asm_name_p
               ) < (size_t) (G__ASM_FUNCNAMEBUF - G__asm_name_p)
            ) {
               G__asm_name_p += strlen(funcname) + 1;
               G__inc_cp_asm(8, 0);
            }
            else {
               G__abortbytecode();
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "COMPILE ABORT function name buffer overflow"
                  );
                  G__printlinenum();
               }
#endif // G__ASM_DBG
               // --
            }
         }
#endif // G__ASM
#ifdef G__DUMPFILE
         if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ipara++) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile , "/* return(cmp) %s()=%s */\n" , funcname(), result7());
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         return result3;
      }
      /***************************************************************
       * search for library function which are included in G__ci.c
       ***************************************************************/
      if (G__library_func(&result3, funcname, libp, hash) == 1) {
         if (G__no_exec_compile) {
            result3.type = 'i';
         }
#ifdef G__ASM
         if (G__asm_noverflow) {
            /****************************************
             * LD_FUNC (library)
             ****************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: LD_FUNC library '%s' paran: %d  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , funcname()
                  , libp->paran
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_FUNC;
            G__asm_inst[G__asm_cp+1] = (long) &G__asm_name[G__asm_name_p];
            G__asm_inst[G__asm_cp+2] = hash;
            G__asm_inst[G__asm_cp+3] = libp->paran;
            G__asm_inst[G__asm_cp+4] = (long) G__library_func;
            G__asm_inst[G__asm_cp+5] = 0;
            G__asm_inst[G__asm_cp+6] = (long) G__p_ifunc;
            G__asm_inst[G__asm_cp+7] = -1;
            if (
               G__strlcpy(
                    G__asm_name + G__asm_name_p
                  , funcname
                  , G__ASM_FUNCNAMEBUF - G__asm_name_p
               ) < (size_t) G__ASM_FUNCNAMEBUF - G__asm_name_p
            ) {
               G__asm_name_p += strlen(funcname) + 1;
               G__inc_cp_asm(8, 0);
            }
            else {
               G__abortbytecode();
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "COMPILE ABORT function name buffer overflow"
                  );
               }
               G__printlinenum();
#endif // G__ASM_DBG
               // --
            }
         }
#endif // G__ASM
#ifdef G__DUMPFILE
         if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ipara++) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile , "/* return(lib) %s()=%s */\n" , funcname(), result7());
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         return result3;
      }
#ifdef G__TEMPLATEFUNC
   templatefunc:
      /******************************************************************
       * Search template function
       ******************************************************************/
      if (
         (G__EXACT == funcmatch || G__USERCONV == funcmatch) &&
         G__templatefunc(&result3, funcname, libp, hash, funcmatch) == 1
      ) {
         // --
#ifdef G__DUMPFILE
         if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ipara++) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile , "/* return(lib) %s()=%s */\n" , funcname(), result7());
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         *known3 = 1;
         return result3;
      }
#endif // G__TEMPLATEFUNC
      // --
   }
   /********************************************************************
    * Explicit type conversion by searching constructors
    ********************************************************************/
   if (
      (memfunc_flag == G__TRYNORMAL) ||
      (memfunc_flag == G__CALLSTATICMEMFUNC)
   ) {
      int store_var_typeX = G__var_type;
      i = G__defined_typename(funcname);
      G__var_type = store_var_typeX;
      if (i != -1) {
         if (G__newtype.tagnum[i] != -1) {
            funcname = G__struct.name[G__newtype.tagnum[i]];
         }
         else {
            result3 = libp->para[0];
            if (
               G__fundamental_conversion_operator(
                  G__newtype.type[i]
                  , -1
                  , i
                  , G__newtype.reftype[i]
                  , 0
                  , &result3
                  , 0
               )
            ) {
               *known3 = 1;
               return result3;
            }
            funcname =
               G__type2string(G__newtype.type[i], G__newtype.tagnum[i],
                  -1, G__newtype.reftype[i], 0);
         }
         G__hash(funcname, hash, i);
      }
      classhash = strlen(funcname);
      i = 0;
      while (i < G__struct.alltag) {
         if (
            (G__struct.hash[i] == classhash) &&
            !strcmp(G__struct.name[i], funcname)
         ) {
            if (
               (G__struct.type[i] == 'e') &&
               (libp->para[0].tagnum != -1) &&
               (G__struct.type[libp->para[0].tagnum] == 'e')
            ) {
               return libp->para[0];
            }
            store_struct_offset = G__store_struct_offset;
            store_tagnum = G__tagnum;
            G__tagnum = i;
            if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
               // compiled class
               G__store_struct_offset = G__PVOID;
            }
            else {
               // interpreted class
               G__alloc_tempobject(G__tagnum, -1);
               G__store_struct_offset = G__p_tempbuf->obj.obj.i;
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating bytecode.
                  if (G__throwingexception) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                           G__serr
                           , "%3x,%3x: ALLOCEXCEPTION %d  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , G__tagnum
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__ALLOCEXCEPTION;
                     G__asm_inst[G__asm_cp+1] = G__tagnum;
                     G__inc_cp_asm(2, 0);
                  }
                  else {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                           G__serr
                           , "%3x,%3x: ALLOCTEMP %d  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , G__tagnum
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
                     G__asm_inst[G__asm_cp+1] = G__tagnum;
                     G__inc_cp_asm(2, 0);
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
               }
#endif // G__ASM
               // --
            }
            G__incsetup_memfunc(G__tagnum);
            for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
               *known3 =
                  G__interpret_func(
                     &result3
                     , funcname
                     , libp
                     , hash
                     , G__struct.memfunc[G__tagnum]
                     , funcmatch
                     , G__TRYCONSTRUCTOR
                  );
               if (*known3) {
                  break;
               }
            }
            if (
               (G__struct.iscpplink[G__tagnum] == G__CPPLINK) &&
               !G__throwingexception
            ) {
               // --
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                        G__serr
                        , "%3x,%3x: STORETEMP  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__STORETEMP;
                  G__inc_cp_asm(1, 0);
               }
#endif // G__ASM
               G__store_tempobject(result3);
            }
            else {
               result3.type = 'u';
               result3.tagnum = G__tagnum;
               result3.typenum = -1;
               result3.obj.i = G__store_struct_offset;
               result3.ref = G__store_struct_offset;
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
                     (result3.tagnum != -1) &&
                     (G__struct.iscpplink[result3.tagnum] != G__CPPLINK)
                  )
               )
            ) {
               G__asm_inst[G__asm_cp] = G__POPTEMP;
               if (G__throwingexception) {
                  G__asm_inst[G__asm_cp+1] = result3.tagnum;
               }
               else {
                  G__asm_inst[G__asm_cp+1] = -1;
               }
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                     G__serr
                     , "%3x,%3x: POPTEMP %d  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , G__asm_inst[G__asm_cp+1]
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__inc_cp_asm(2, 0);
            }
#endif // G__ASM
            if (*known3) {
               // Return '*this' as result
               return result3;
            }
            if (-1 != i && libp->paran == 1 && -1 != libp->para[0].tagnum) {
               long store_struct_offset = G__store_struct_offset;
               long store_memberfunc_struct_offset =
                  G__memberfunc_struct_offset;
               int store_memberfunc_tagnum = G__memberfunc_tagnum;
               int store_exec_memberfunc = G__exec_memberfunc;
               store_tagnum = G__tagnum;
               // FIXME: Print a message about the cancel when G__asm_dbg!
               // FIXME: Cannot do this without testing G__ASM!
               G__inc_cp_asm(-5, 0); // cancel ALLOCTEMP, SETTEMP, POPTEMP
               G__pop_tempobject();
               G__tagnum = libp->para[0].tagnum;
               G__store_struct_offset = libp->para[0].obj.i;
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: PUSHSTROS  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__PUSHSTROS;
                  G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: SETSTROS  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__SETSTROS;
                  G__inc_cp_asm(1, 0);
               }
#endif // G__ASM
               funcname.Format("operator %s", G__fulltagname(i, 1));
               G__hash(funcname, hash, i);
               G__incsetup_memfunc(G__tagnum);
               libp->paran = 0;
               for (
                  funcmatch = G__EXACT;
                  funcmatch <= G__USERCONV;
                  ++funcmatch
               ) {
                  *known3 =
                     G__interpret_func(
                        &result3
                        , funcname
                        , libp
                        , hash
                        , G__struct.memfunc[G__tagnum]
                        , funcmatch
                        , G__TRYMEMFUNC
                     );
                  if (*known3) {
                     // Cleanup after successful function call.
#ifdef G__ASM
                     if (G__asm_noverflow) {
                        // We are generating bytecode.
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(
                              G__serr
                              , "%3x,%3x: POPSTROS  %s:%d\n"
                              , G__asm_cp
                              , G__asm_dt
                              , __FILE__
                              , __LINE__
                           );
                        }
#endif // G__ASM_DBG
                        G__asm_inst[G__asm_cp] = G__POPSTROS;
                        G__inc_cp_asm(1, 0);
                     }
#endif // G__ASM
                     break;
                  }
               }
               G__memberfunc_struct_offset = store_memberfunc_struct_offset;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
               G__exec_memberfunc = store_exec_memberfunc;
               G__tagnum = store_tagnum;
               G__store_struct_offset = store_struct_offset;
            }
            else if (-1 != i && libp->paran == 1) {
               G__fprinterr(
                  G__serr
                  , "Error: No matching constructor for "
                  "explicit conversion %s"
                  , item
               );
               G__genericerror(0);
            }
            // omitted constructor, return uninitialized object
            *known3 = 1;
            return result3;
         }
         i++;
      }
      result3.ref = 0;
      if (G__explicit_fundamental_typeconv(funcname, classhash, libp, &result3)) {
         *known3 = 1;
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         return result3;
      }
   }
   if (G__parenthesisovld(&result3, funcname, libp, G__TRYNORMAL)) {
      *known3 = 1;
      return result3;
   }
   /********************************************************************
    * pointer to function described like normal function
    * int (*p2f)(void);  p2f();
    ********************************************************************/
   var = G__getvarentry(funcname, hash, &ig15, &G__global, G__p_local);
   if (var && var->type[ig15] == '1') {
      result7.Format("*%s", funcname());
      *known3 = 0;
      pfparam = (char*)strchr(item, '(');
      if (pfparam) {
         p2ffpara = libp;
         result3 = G__pointer2func(0, result7, pfparam, known3);
      }
      p2ffpara = 0;
      if (*known3) {
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         return result3;
      }
   }
   *known3 = 0;
#ifdef G__DUMPFILE
   if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
      G__dumpspace -= 3;
   }
#endif // G__DUMPFILE
   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   if (!G__oprovld) {
      if (G__asm_noverflow && libp->paran) {
         G__asm_cp = store_cp_asm;
      }
      G__asm_clear_mask = 1;
      result3 = G__execfuncmacro(item, known3);
      G__asm_clear_mask = 0;
      if (*known3) {
         return result3;
      }
   }
   return G__null;
}

/******************************************************************
* G__value G__getfunction(item,known3,memfunc_flag)
******************************************************************/
G__value G__getfunction(const char* item, int* known3, int memfunc_flag)
{
   G__value result3 = G__null;
   int ig15, ig35, ipara;
   int lenitem, nest = 0;
   int single_quote = 0, double_quote = 0;
   struct G__param fpara;
   static struct G__param* p2ffpara = (struct G__param*)NULL;
   int hash;
   short castflag;
   int funcmatch;
   int i, classhash;
   long store_globalvarpointer;
   long store_struct_offset;
   int store_tagnum;
   int store_exec_memberfunc;
   int store_asm_noverflow;
   int store_var_type;
   int store_memberfunc_tagnum;
   long store_memberfunc_struct_offset;
   int store_memberfunc_var_type;
   int store_def_tagnum, store_tagdefining;
   int tempstore;
   char* pfparam;
   struct G__var_array* var;
   int nindex = 0;
   int base1 = 0;
   int oprp = 0;
   int store_cp_asm = 0;
   int memfuncenvflag = 0;
   store_exec_memberfunc = G__exec_memberfunc;
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   /******************************************************
    * if string expression "string"
    * return
    ******************************************************/
   if (item[0] == '"') {
      result3 = G__null;
      return result3;
   }
   /******************************************************
    * get length of expression
    ******************************************************/
   lenitem = strlen(item);
   G__FastAllocString result7(lenitem);
   G__FastAllocString funcname(lenitem + 24);
   /******************************************************
    * Scan item[] until '(' to get function name and hash
    ******************************************************/
   /* Separate function name */
   ig15 = 0;
   hash = 0;
   while ((item[ig15] != '(') && (ig15 < lenitem)) {
      funcname[ig15] = item[ig15];
      hash += item[ig15];
      ig15++;
   }
   if (8 == ig15 && strncmp(funcname, "operator", 8) == 0 &&
         strncmp(item + ig15, "()(", 3) == 0) {
      funcname.Replace(8, "()");
      hash = hash + '(' + ')';
      ig15 += 2;
   }
   /******************************************************
    * if itemp[0]=='(' this is a casting or pointer to
    * function.
    ******************************************************/
   castflag = 0;
   if (ig15 == 0) {
      castflag = 1;
   }
   /******************************************************
    * if '(' not found in expression, this is not a function
    * call , so just return.
    * this shouldn't happen.
    ******************************************************/
   if (item[ig15] != '(') {
      /* if no parenthesis , this is not a function */
      result3 = G__null;
      return(result3);
   }
   /******************************************************
    * put null char to the end of function name
    ******************************************************/
   funcname[ig15++] = '\0';
   if ((strchr(funcname, '.') || strstr(funcname, "->"))
         && 0 != strncmp(funcname, "operator", 8)
      ) {
      result3 = G__null;
      return(result3);
   }
   /******************************************************
    * conv<B>(x) -> conv<ns::B>(x)
    ******************************************************/
   G__rename_templatefunc(funcname);
   //
   // Get function call arguments.
   //
   //
   //  func(arg, arg, ...)
   //       ^
   fpara.paran = 0;
   fpara.para[0].type = 0;
   while (ig15 < lenitem) {
      // -- Scan one argument.
      int tmpltnest = 0;
      ig35 = 0;
      nest = 0;
      single_quote = 0;
      double_quote = 0;
      // Skip leading spaces.
      while (item[ig15] == ' ') {
         ++ig15;
      }
      while (
         (ig15 < lenitem) &&
         (
            ((item[ig15] != ',') && (item[ig15] != ')')) ||
            nest ||
            tmpltnest ||
            single_quote ||
            double_quote
         )
      ) {
         // -- Collect characters until a comma or right parenthesis is seen.
         switch (item[ig15]) {
            case '"' : /* double quote */
               if (single_quote == 0) {
                  double_quote ^= 1;
               }
               break;
            case '\'' : /* single quote */
               if (double_quote == 0) {
                  single_quote ^= 1;
               }
               break;
            case '(':
            case '[':
            case '{':
               if ((double_quote == 0) && (single_quote == 0)) {
                  nest++;
               }
               break;
            case ')':
            case ']':
            case '}':
               if ((double_quote == 0) && (single_quote == 0)) {
                  nest--;
               }
               break;
            case '\\':
               result7.Set(ig35++, item[ig15++]);
               break;
            case '<':
               if (double_quote == 0 && single_quote == 0) {
                  result7.Set(ig35, 0);
                  char* checkForTemplate = result7;
                  if (checkForTemplate && !strncmp(checkForTemplate, "const ", 6)) {
                     checkForTemplate += 6;
                  }
                  if (0 == strcmp(result7, "operator") ||
                        tmpltnest ||
                        G__defined_templateclass(checkForTemplate)) {
                     ++tmpltnest;
                  }
               }
               break;
            case '>':
               if (double_quote == 0 && single_quote == 0) {
                  if (tmpltnest) {
                     --tmpltnest;
                  }
               }
               break;
         }
         result7.Set(ig35++, item[ig15++]);
      }
      //
      // If a right parenthesis is found in the middle of an expression,
      // this should either be casting or a pointer to a function.
      //
      //  v                    v            <-- this makes castflag = 1
      //  (type)expression  or (*p_func)();
      //       ^                       ^    <-- this makes castflag = 2
      //
      if ((item[ig15] == ')') && (ig15 < lenitem - 1)) {
         if (castflag == 1) {
            if (('-' == item[ig15+1] && '>' == item[ig15+2]) || '.' == item[ig15+1]) {
               castflag = 3;
            }
            else {
               castflag = 2;
            }
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
         else if ((isalpha(item[0]) || (item[0] == '_') || (item[0] == '$'))) {
            int itmp;
            result7[ig35] = '\0';
            G__strlcpy(fpara.parameter[fpara.paran], result7, G__ONELINE);
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
      ig15++;
      result7[ig35] = '\0';
      if (ig35 < G__ONELINE) {
         G__strlcpy(fpara.parameter[fpara.paran], result7, G__ONELINE);
      }
      else {
         G__strlcpy(fpara.parameter[fpara.paran], "@", G__ONELINE);
         fpara.para[fpara.paran] = G__getexpr(result7);
      }
      // Initialize next argument to the empty string.
      fpara.parameter[++fpara.paran][0] = '\0';
   }
   if ((castflag == 1) && !funcname[0]) {
#ifndef G__OLDIMPLEMENTATION
      if ((fpara.paran == 1) && !strcmp(fpara.parameter[0], "@")) {
         result3 = fpara.para[0];
      }
      else {
         result3 = G__getexpr(result7);
      }
#else // G__OLDIMPLEMENTATION
      result3 = G__getexpr(result7); // THIS CAN BE DUPLICATED EVALUATION
#endif // G__OLDIMPLEMENTATION
      *known3 = 1;
      return result3;
   }
   if (castflag == 3) {
      // -- Member access by (xxx)->xxx , (xxx).xxx
      store_var_type = G__var_type;
      G__var_type = 'p';
      if (fpara.parameter[1][0] == '.') {
         i = 1;
      }
      else {
         i = 2;
      }
      if (base1) {
         strncpy(fpara.parameter[0], item, base1);
         fpara.parameter[0][base1] = '\0';
         G__strlcpy(fpara.parameter[1], item + base1, G__ONELINE);
      }
      if (memfunc_flag == G__CALLMEMFUNC) {
         result3 = G__getstructmem(store_var_type, funcname, fpara.parameter[1] + i, G__ONELINE - i - 1, fpara.parameter[0], known3, 0, i);
      }
      else {
         result3 = G__getstructmem(store_var_type, funcname, fpara.parameter[1] + i, G__ONELINE - i - 1, fpara.parameter[0], known3, &G__global, i);
      }
      G__var_type = store_var_type;
      return result3;
   }
   if (castflag == 2) {
      // -- Casting or pointer to function.
      /***************************************************************
       * pointer to function
       *
       *  (*p_function)(param);
       *   ^
       *  this '*' is significant
       ***************************************************************/
      if (fpara.parameter[0][0] == '*') {
         switch (fpara.parameter[1][0]) {
            case '[':
               // function pointer
               return G__pointerReference(fpara.parameter[0], &fpara, known3);
            case '(':
            default:
               // function pointer
               return G__pointer2func(0, fpara.parameter[0], fpara.parameter[1], known3);
         }
      }
#ifdef G__PTR2MEMFUNC
      /***************************************************************
       * pointer to member function
       *
       *  (obj.*p2mf)(param);
       *  (obj->*p2mf)(param);
       ***************************************************************/
      else if ('(' == fpara.parameter[1][0] &&
               (strstr(fpara.parameter[0], ".*") ||
                strstr(fpara.parameter[0], "->*"))) {
         return(G__pointer2memberfunction(fpara.parameter[0]
                                          , fpara.parameter[1], known3));
      }
#endif // G__PTR2MEMFUNC
      /***************************************************************
       * (expr)[n]
       ***************************************************************/
      else if ((fpara.paran >= 2) && (fpara.parameter[1][0] == '[')) {
         result3 = G__getexpr(G__catparam(&fpara, fpara.paran, ""));
         *known3 = 1;
         return result3;
      }
      /***************************************************************
       * casting
       *
       *  (type)expression;
       ***************************************************************/
      else {
         if (fpara.paran > 2) {
            if ('@' == fpara.parameter[fpara.paran-1][0]) {
               fpara.para[1] = fpara.para[fpara.paran-1];
            }
            else {
               fpara.para[1] = G__getexpr(fpara.parameter[fpara.paran-1]);
            }
            result3 = G__castvalue(G__catparam(&fpara, fpara.paran - 1, ","), fpara.para[1]);
         }
         else {
            if ('@' == fpara.parameter[1][0]) {
               ; // NO-OP: fpara.para[1] = fpara.para[1];
            }
            else {
               fpara.para[1] = G__getexpr(fpara.parameter[1]);
            }
            result3 = G__castvalue(fpara.parameter[0], fpara.para[1]);
         }
         *known3 = 1;
         return result3;
      }
   }
   if (!fpara.parameter[0][0]) {
      // -- First parameter is empty, set that there are no parameters.
      if ((fpara.paran > 1) && (item[strlen(item)-1] == ')')) {
         if ((fpara.paran == 2) && (fpara.parameter[1][0] == '(')) {
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
   result3.tagnum = -1;
   result3.typenum = -1;
   result3.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
   result3.isconst = 0;
#endif // G__OLDIMPLEMENTATION1259
   result3.obj.reftype.reftype = G__PARANORMAL;
   *known3 = 1;
   /***************************************************************
    *  Search for sizeof(),
    * before parameters are evaluated
    *  sizeof() is processed specially.
    ***************************************************************/
   if (G__special_func(&result3, funcname, &fpara, hash) == 1) {
      G__var_type = 'p';
      return(result3);
   }
   /***************************************************************
    *  Evaluate parameters  parameter:string expression ,
    *                       para     :evaluated expression
    ***************************************************************/
#ifdef G__ASM
   store_asm_noverflow = G__asm_noverflow;
   if (G__oprovld) {
      /* In case of operator overloading function, arguments are already
       * evaluated. Avoid duplication in argument stack by temporarily
       * reset G__asm_noverflow */
      /* G__asm_noverflow=0; */
      G__suspendbytecode();
   }
   /*DEBUG*/ /* fprintf(stderr,"\nSET %lx=%lx %lx %s\n",G__memberfunc_struct_offset,G__store_struct_offset,store_struct_offset,item); */
   if (
      G__asm_noverflow &&
      fpara.paran &&
      (
         (G__store_struct_offset != G__memberfunc_struct_offset) ||
         G__do_setmemfuncenv
      )
   ) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: SETMEMFUNCENV  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , __FILE__
            , __LINE__
         );
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
   /* restore base environment */
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_memberfunc_var_type = G__var_type;
   store_globalvarpointer = G__globalvarpointer;
   G__globalvarpointer = G__PVOID;
   G__tagnum = G__memberfunc_tagnum;
   G__store_struct_offset = G__memberfunc_struct_offset;
   G__var_type = 'p';
   //
   //  Evaluate parameters.
   //
   if (!p2ffpara) {
      // -- Parameter list was not pre-provided.
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
   else {
      fpara = *p2ffpara;
      p2ffpara = 0;
   }
   /* recover function call environment */
#ifdef G__ASM
   /*DEBUG*/ /* fprintf(stderr,"\nREC %lx %lx=%lx %s\n",G__memberfunc_struct_offset,G__store_struct_offset,store_struct_offset,item); */
   if (
      G__asm_noverflow &&
      fpara.paran &&
      memfuncenvflag
   ) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: RECMEMFUNCENV  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , __FILE__
            , __LINE__
         );
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
   if (
      (G__return > G__RETURN_NORMAL) ||
      (G__security_error && (G__security != G__SECURE_NONE))
   ) {
      return G__null;
   }
#endif // G__SECURITY
   fpara.para[fpara.paran] = G__null;
   /***************************************************************
    * if not function name, this is '(expr1,expr2,...,exprn)'
    * According to ANSI-C , exprn has to be returned,
    ***************************************************************/
   if (funcname[0] == '\0') {
      /*************************************************
       *  'result3 = fpara.para[fpara.paran-1] ;'
       * should be correct as ANSI-C.
       *************************************************/
      result3 = fpara.para[0];
      return(result3);
   }
   /* scope operator ::f() , A::B::f()
    * note,
    *    G__exec_memberfunc restored at return memfunc_flag is local,
    *   there should be no problem modifying these variables.
    *    store_struct_offset and store_tagnum are only used in the
    *   explicit type conversion section.  It is OK to use them here
    *   independently.
    */
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_def_tagnum = G__def_tagnum;
   store_tagdefining = G__tagdefining;
   switch (G__scopeoperator(funcname, &hash, &G__store_struct_offset, &G__tagnum)) {
      case G__GLOBALSCOPE: /* global scope */
         G__exec_memberfunc = 0;
         memfunc_flag = G__TRYNORMAL;
         G__def_tagnum = -1;
         G__tagdefining = -1;
         break;
      case G__CLASSSCOPE: /* class scope */
         memfunc_flag = G__CALLSTATICMEMFUNC;
         /* This looks very risky */
         G__def_tagnum = -1;
         G__tagdefining = -1;
         G__exec_memberfunc = 1;
         G__memberfunc_tagnum = G__tagnum;
         break;
   }
#ifdef G__DUMPFILE
   /***************************************************************
    * dump that a function is called
    ***************************************************************/
   if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
      for (ipara = 0; ipara < G__dumpspace; ipara++) {
         fprintf(G__dumpfile, " ");
      }
      fprintf(G__dumpfile, "%s(", funcname());
      for (ipara = 1; ipara <= fpara.paran; ipara++) {
         if (ipara != 1) {
            fprintf(G__dumpfile, ",");
         }
         G__valuemonitor(fpara.para[ipara-1], result7);
         fprintf(G__dumpfile, "%s", result7());
      }
      fprintf(G__dumpfile, ");/*%s %d,%lx %lx*/\n"
              , G__ifile.name, G__ifile.line_number
              , store_struct_offset, G__store_struct_offset);
      G__dumpspace += 3;
   }
#endif // G__DUMPFILE
   //
   //  Perform function overload resolution, and execute function.
   //
   for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
      // -- Try to resolve overloaded function.
      //
      // Search for interpreted member function
      // if(G__exec_memberfunc)     ==>  memfunc();
      // G__TRYNORMAL!=memfunc_flag ==>  a.memfunc();
      //
      if (G__exec_memberfunc || (memfunc_flag != G__TRYNORMAL)) {
         int local_tagnum = G__tagnum;
         if (G__exec_memberfunc && (G__tagnum == -1)) {
            local_tagnum = G__memberfunc_tagnum;
         }
         // Perform any delayed dictionary loading.
         if (G__tagnum != -1) {
            G__incsetup_memfunc(G__tagnum);
         }
         if (local_tagnum != -1) {
            //
            //
            //  Call an interpreted function.
            //
            int ret = G__interpret_func(&result3, funcname, &fpara, hash, G__struct.memfunc[local_tagnum], funcmatch, memfunc_flag);
            if (ret == 1) {
               // -- We found it and ran it, done.
#ifdef G__DUMPFILE
               if (G__dumpfile && !G__no_exec_compile) {
                  G__dumpspace -= 3;
                  for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                     fprintf(G__dumpfile, " ");
                  }
                  G__valuemonitor(result3, result7);
                  fprintf(G__dumpfile, "/* return(inp) %s.%s()=%s*/\n", G__tagnum >= 0 ? G__struct.name[G__tagnum] : "unknown class", funcname(), result7());
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
               G__setclassdebugcond(G__memberfunc_tagnum, 0);
               if (nindex && (isupper(result3.type) || (result3.type == 'u'))) {
                  G__getindexedvalue(&result3, fpara.parameter[nindex]);
               }
               if (oprp) {
                  *known3 = G__additional_parenthesis(&result3, &fpara);
               }
               return result3;
            }
         }
#define G__OLDIMPLEMENTATION1159
         // --
      }
      /***************************************************************
       * If memberfunction is called explicitly by clarifying scope
       * don't examine global function and exit from G__getfunction().
       * There are 2 cases                   G__exec_memberfunc
       *   obj.memfunc();                            1
       *   X::memfunc();                             1
       *    X();              constructor            2
       *   ~X();              destructor             2
       * If G__exec_memberfunc==2, don't display error message.
       ***************************************************************/
      /* If searching only member function */
      if (memfunc_flag
            && (G__store_struct_offset || G__CALLSTATICMEMFUNC != memfunc_flag)
         ) {
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         /* If the last resolution of overloading failed */
         if (funcmatch == G__USERCONV) {
            if (G__TRYDESTRUCTOR == memfunc_flag) {
               /* destructor for base calss and class members */
#ifdef G__ASM
#ifdef G__SECURITY
               store_asm_noverflow = G__asm_noverflow;
               if (G__security & G__SECURE_GARBAGECOLLECTION) {
                  G__abortbytecode();
               }
#endif // G__SECURITY
#endif // G__ASM
#ifdef G__VIRTUALBASE
               if (
                  (G__tagnum != -1) &&
                  (G__struct.iscpplink[G__tagnum] != G__CPPLINK)
               ) {
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
                  case G__TRYIMPLICITCONSTRUCTOR: {
                        /* constructor for base class and class members default
                         * constructor only */
                        int store2_exec_memberfunc = G__exec_memberfunc;
                        G__exec_memberfunc = 1;
#ifdef G__VIRTUALBASE
                        if (G__tagnum != -1 && G__CPPLINK != G__struct.iscpplink[G__tagnum]) {
                           G__baseconstructor(0 , (struct G__baseparam*)NULL);
                        }
#else // G__VIRTUALBASE
                        G__baseconstructor(0 , (struct G__baseparam*)NULL);
#endif // G__VIRTUALBASE
                        G__exec_memberfunc = store2_exec_memberfunc;
                        break;
                     }
               }
            }
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;
            *known3 = 0;
            switch (memfunc_flag) {
               case G__CALLMEMFUNC:
                  if (G__parenthesisovld(&result3, funcname, &fpara, G__CALLMEMFUNC)) {
                     *known3 = 1;
                     if (G__store_struct_offset != store_struct_offset) {
                        G__gen_addstros(store_struct_offset - G__store_struct_offset);
                     }
                     G__store_struct_offset = store_struct_offset;
                     G__tagnum = store_tagnum;
                     G__def_tagnum = store_def_tagnum;
                     G__tagdefining = store_tagdefining;
                     if (nindex &&
                           (isupper(result3.type) || 'u' == result3.type)
                        ) {
                        G__getindexedvalue(&result3, fpara.parameter[nindex]);
                     }
                     if (oprp) {
                        *known3 = G__additional_parenthesis(&result3, &fpara);
                     }
                     return(result3);
                  }
                  if ('~' == funcname[0]) {
                     *known3 = 1;
                     return(G__null);
                  }
                  // We did not find the function as a regular member, let's try as a constructor.
               case G__CALLCONSTRUCTOR:
                  /******************************************************************
                   * Search template function
                   ******************************************************************/
                  G__exec_memberfunc = 1;
                  G__memberfunc_tagnum = G__tagnum;
                  G__memberfunc_struct_offset = G__store_struct_offset;
                  if ((G__EXACT == funcmatch || G__USERCONV == funcmatch) &&
                        G__templatefunc(&result3, funcname, &fpara, hash, funcmatch) == 1) {
#ifdef G__DUMPFILE
                     if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
                        G__dumpspace -= 3;
                        for (ipara = 0; ipara < G__dumpspace; ipara++) {
                           fprintf(G__dumpfile, " ");
                        }
                        G__valuemonitor(result3, result7);
                        fprintf(G__dumpfile , "/* return(lib) %s()=%s */\n"
                                , funcname(), result7());
                     }
#endif // G__DUMPFILE
                     G__exec_memberfunc = store_exec_memberfunc;
                     G__memberfunc_tagnum = store_memberfunc_tagnum;
                     G__memberfunc_struct_offset = store_memberfunc_struct_offset;
                     /* don't know why if(oprp) is needed, copied from line 2111 */
                     if (oprp) {
                        *known3 = G__additional_parenthesis(&result3, &fpara);
                     }
                     else {
                        *known3 = 1;
                     }
                     return result3;
                  }
                  G__exec_memberfunc = store_exec_memberfunc;
                  G__memberfunc_tagnum = store_memberfunc_tagnum;
                  G__memberfunc_struct_offset = store_memberfunc_struct_offset;
                  if (G__NOLINK > G__globalcomp) {
                     break;
                  }
                  if (!G__no_exec_compile || G__asm_noverflow) {
                     if (!G__const_noerror) {
                        G__fprinterr(
                             G__serr
                           , "Error: Can't call %s::%s in current scope"
                           , G__struct.name[G__tagnum]
                           , item
                        );
                        G__genericerror(0);
                     }
                  }
                  store_exec_memberfunc = G__exec_memberfunc;
                  G__exec_memberfunc = 1;
                  if (
                     !G__const_noerror &&
                     (!G__no_exec_compile || G__asm_noverflow)
                  ) {
                     G__fprinterr(G__serr, "Possible candidates are...\n");
                     {
                        G__FastAllocString itemtmp(G__LONGLINE);
                        itemtmp.Format(
                             "%s::%s"
                           , G__struct.name[G__tagnum]
                           , funcname()
                        );
                        G__display_proto_pretty(G__serr, itemtmp, 1);
                     }
                  }
                  G__exec_memberfunc = store_exec_memberfunc;
            }
#ifdef G__DUMPFILE
            if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
               G__dumpspace -= 3;
            }
#endif // G__DUMPFILE
            if (G__store_struct_offset != store_struct_offset) {
               G__gen_addstros(store_struct_offset - G__store_struct_offset);
            }
            G__store_struct_offset = store_struct_offset;
            G__def_tagnum = store_def_tagnum;
            G__tagdefining = store_tagdefining;
            if (fpara.paran && 'u' == fpara.para[0].type &&
                  (G__TRYCONSTRUCTOR == memfunc_flag ||
                   G__TRYIMPLICITCONSTRUCTOR == memfunc_flag)
               ) {
               /* in case of copy constructor not found */
               G__tagnum = store_tagnum;
               return(fpara.para[0]);
            }
            else {
               result3 = G__null;
               result3.tagnum = G__tagnum;
               G__tagnum = store_tagnum;
               return(result3);
            }
         }
         /* ELSE next level overloaded function resolution */
         continue;
      }
      /***************************************************************
       * reset G__exec_memberfunc for global function.
       * Original value(store_exec_memberfunc) is restored when exit
       * from this function
       ***************************************************************/
      tempstore = G__exec_memberfunc;
      G__exec_memberfunc = 0;
      //
      //  Search for interpreted global function.
      //
      if (memfunc_flag != G__CALLSTATICMEMFUNC) {
         int called = G__interpret_func(&result3, funcname, &fpara, hash, G__p_ifunc, funcmatch, G__TRYNORMAL);
         if (called) {
            // --
#ifdef G__DUMPFILE
            if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
               G__dumpspace -= 3;
               for (ipara = 0; ipara < G__dumpspace; ++ipara) {
                  fprintf(G__dumpfile, " ");
               }
               G__valuemonitor(result3, result7);
               fprintf(G__dumpfile , "/* return(inp) %s()=%s*/\n" , funcname(), result7());
            }
#endif // G__DUMPFILE
            G__exec_memberfunc = store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__memberfunc_struct_offset = store_memberfunc_struct_offset;
            G__setclassdebugcond(G__memberfunc_tagnum, 0);
            if (nindex && (isupper(result3.type) || 'u' == result3.type)) {
               G__getindexedvalue(&result3, fpara.parameter[nindex]);
            }
            if (oprp) {
               *known3 = G__additional_parenthesis(&result3, &fpara);
            }
            return result3;
         }
      }
      G__exec_memberfunc = tempstore;
      /* there is no function overload resolution after this point,
       * thus, if not found in G__EXACT trial, there is no chance to
       * find matched function in consequitive search
       */
      if (G__USERCONV == funcmatch) {
         goto templatefunc;
      }
      if (G__EXACT != funcmatch) {
         continue;
      }
      /***************************************************************
       * search for compiled(archived) function
       ***************************************************************/
      if ((memfunc_flag != G__CALLSTATICMEMFUNC) && G__compiled_func(&result3, funcname, &fpara, hash) == 1) {
         // --
#ifdef G__ASM
         if (G__asm_noverflow) {
            // We are generating bytecode.
            /****************************************
             * LD_FUNC (compiled)
             ****************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: LD_FUNC compiled '%s' paran: %d  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , funcname()
                  , fpara.paran
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_FUNC;
            G__asm_inst[G__asm_cp+1] = (long) &G__asm_name[G__asm_name_p];
            G__asm_inst[G__asm_cp+2] = hash;
            G__asm_inst[G__asm_cp+3] = fpara.paran;
            G__asm_inst[G__asm_cp+4] = (long) G__compiled_func;
            G__asm_inst[G__asm_cp+5] = 0;
            G__asm_inst[G__asm_cp+6] = (long) G__p_ifunc;
            G__asm_inst[G__asm_cp+7] = -1;
            if (G__strlcpy(G__asm_name + G__asm_name_p, funcname, G__ASM_FUNCNAMEBUF - G__asm_name_p) < (size_t)(G__ASM_FUNCNAMEBUF - G__asm_name_p)) {
               G__asm_name_p += strlen(funcname) + 1;
               G__inc_cp_asm(8, 0);
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
         if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ipara++) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile , "/* return(cmp) %s()=%s */\n" , funcname(), result7());
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (nindex &&
               (isupper(result3.type) || 'u' == result3.type)
            ) {
            G__getindexedvalue(&result3, fpara.parameter[nindex]);
         }
         return result3;
      }
      /***************************************************************
       * search for library function which are included in G__ci.c
       ***************************************************************/
      if (G__library_func(&result3, funcname, &fpara, hash) == 1) {
         if (G__no_exec_compile) {
            result3.type = 'i';
         }
#ifdef G__ASM
         if (G__asm_noverflow) {
            /****************************************
             * LD_FUNC (library)
             ****************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD_FUNC library '%s' paran: %d  %s:%d\n", G__asm_cp, G__asm_dt, funcname(), fpara.paran, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_FUNC;
            G__asm_inst[G__asm_cp+1] = (long) &G__asm_name[G__asm_name_p];
            G__asm_inst[G__asm_cp+2] = hash;
            G__asm_inst[G__asm_cp+3] = fpara.paran;
            G__asm_inst[G__asm_cp+4] = (long) G__library_func;
            G__asm_inst[G__asm_cp+5] = 0;
            G__asm_inst[G__asm_cp+6] = (long) G__p_ifunc;
            G__asm_inst[G__asm_cp+7] = -1;
            if (G__strlcpy(G__asm_name + G__asm_name_p, funcname, G__ASM_FUNCNAMEBUF - G__asm_name_p) < (size_t)(G__ASM_FUNCNAMEBUF - G__asm_name_p)) {
               G__asm_name_p += strlen(funcname) + 1;
               G__inc_cp_asm(8, 0);
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
         if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ipara++) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile , "/* return(lib) %s()=%s */\n" , funcname(), result7());
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (nindex &&
               (isupper(result3.type) || 'u' == result3.type)
            ) {
            G__getindexedvalue(&result3, fpara.parameter[nindex]);
         }
         return(result3);
      }
#ifdef G__TEMPLATEFUNC
   templatefunc:
      /******************************************************************
       * Search template function
       ******************************************************************/
      if ((G__EXACT == funcmatch || G__USERCONV == funcmatch) &&
            G__templatefunc(&result3, funcname, &fpara, hash, funcmatch) == 1) {
#ifdef G__DUMPFILE
         if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
            G__dumpspace -= 3;
            for (ipara = 0; ipara < G__dumpspace; ipara++) {
               fprintf(G__dumpfile, " ");
            }
            G__valuemonitor(result3, result7);
            fprintf(G__dumpfile , "/* return(lib) %s()=%s */\n" , funcname(), result7());
         }
#endif // G__DUMPFILE
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, &fpara);
         }
         else {
            *known3 = 1;   /* don't know why this was missing */
         }
         return(result3);
      }
#endif // G__TEMPLATEFUNC
      /******************************************************************
       * End Loop to resolve overloaded function
       ******************************************************************/
      /* next_overload_match:
         ; */
   }
   /********************************************************************
    * Explicit type conversion by searching constructors
    ********************************************************************/
   if (G__TRYNORMAL == memfunc_flag
         || G__CALLSTATICMEMFUNC == memfunc_flag
      ) {
      int store_var_typeX = G__var_type;
      i = G__defined_typename(funcname);
      G__var_type = store_var_typeX;
      if (-1 != i) {
         if (-1 != G__newtype.tagnum[i]) {
            funcname = G__struct.name[G__newtype.tagnum[i]];
         }
         else {
            result3 = fpara.para[0];
            if (G__fundamental_conversion_operator(G__newtype.type[i], -1
                                                   , i , G__newtype.reftype[i], 0
                                                   , &result3, 0)) {
               *known3 = 1;
               if (oprp) {
                  *known3 = G__additional_parenthesis(&result3, &fpara);
               }
               return(result3);
            }
            funcname = G__type2string(G__newtype.type[i]
                                      , G__newtype.tagnum[i] , -1
                                      , G__newtype.reftype[i] , 0);
         }
         G__hash(funcname, hash, i);
      }
      classhash = strlen(funcname);
      i = 0;
      while (i < G__struct.alltag) {
         if ((G__struct.hash[i] == classhash) &&
               (strcmp(G__struct.name[i], funcname) == 0)
            ) {
            if ('e' == G__struct.type[i] &&
                  -1 != fpara.para[0].tagnum &&
                  'e' == G__struct.type[fpara.para[0].tagnum]) {
               return(fpara.para[0]);
            }
            store_struct_offset = G__store_struct_offset;
            /* questionable part */
            /* store_exec_memfunc=G__exec_memberfunc; */
            store_tagnum = G__tagnum;
            G__tagnum = i;
            G__class_autoloading(&G__tagnum); // Autoload if necessary.
            if (G__CPPLINK != G__struct.iscpplink[G__tagnum]) {
               G__alloc_tempobject(G__tagnum, -1);
               G__store_struct_offset = G__p_tempbuf->obj.obj.i;
#ifdef G__ASM
               if (G__asm_noverflow) {
                  if (G__throwingexception) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: ALLOCEXCEPTION %d  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , G__tagnum
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__ALLOCEXCEPTION;
                     G__asm_inst[G__asm_cp+1] = G__tagnum;
                     G__inc_cp_asm(2, 0);
                  }
                  else {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: ALLOCTEMP %d  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , G__tagnum
                           , __FILE__
                           , __LINE__
                        );
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
                     G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
                     G__asm_inst[G__asm_cp+1] = G__tagnum;
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
            for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; funcmatch++) {
               *known3 = G__interpret_func(&result3, funcname
                                           , &fpara, hash
                                           , G__struct.memfunc[G__tagnum]
                                           , funcmatch
                                           , G__TRYCONSTRUCTOR);
               if (*known3) {
                  break;
               }
            }
            if (
               (G__struct.iscpplink[G__tagnum] == G__CPPLINK) &&
               !G__throwingexception
            ) {
               // --
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: STORETEMP  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__STORETEMP;
                  G__inc_cp_asm(1, 0);
               }
#endif // G__ASM
               G__store_tempobject(result3);
            }
            else {
               result3.type = 'u';
               result3.tagnum = G__tagnum;
               result3.typenum = -1;
               result3.obj.i = G__store_struct_offset;
               result3.ref = G__store_struct_offset;
            }
            G__tagnum = store_tagnum;
            /* questionable part */
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
                     (result3.tagnum != -1) &&
                     (G__struct.iscpplink[result3.tagnum] != G__CPPLINK)
                  )
               )
            ) {
               G__asm_inst[G__asm_cp] = G__POPTEMP;
               if (G__throwingexception) {
                  G__asm_inst[G__asm_cp+1] = result3.tagnum;
               }
               else {
                  G__asm_inst[G__asm_cp+1] = -1;
               }
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: POPTEMP %d  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , G__asm_inst[G__asm_cp+1]
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__inc_cp_asm(2, 0);
            }
#endif // G__ASM
            if (0 == *known3) {
               if (-1 != i && fpara.paran == 1 && -1 != fpara.para[0].tagnum) {
                  long store_struct_offset = G__store_struct_offset;
                  long store_memberfunc_struct_offset = G__memberfunc_struct_offset;
                  int store_memberfunc_tagnum = G__memberfunc_tagnum;
                  int store_exec_memberfunc = G__exec_memberfunc;
                  store_tagnum = G__tagnum;
                  // FIXME: Print a message here!
                  G__inc_cp_asm(-5, 0); /* cancel ALLOCTEMP, SETTEMP, POPTEMP */
                  G__pop_tempobject();
                  G__tagnum = fpara.para[0].tagnum;
                  G__store_struct_offset = fpara.para[0].obj.i;
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // We are generating bytecode.
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: PUSHSTROS  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__PUSHSTROS;
                     G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: SETSTROS  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__SETSTROS;
                     G__inc_cp_asm(1, 0);
                  }
#endif // G__ASM
                  funcname.Format("operator %s", G__fulltagname(i, 1));
                  G__hash(funcname, hash, i);
                  G__incsetup_memfunc(G__tagnum);
                  fpara.paran = 0;
                  for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; funcmatch++) {
                     *known3 = G__interpret_func(&result3, funcname
                                                 , &fpara, hash
                                                 , G__struct.memfunc[G__tagnum]
                                                 , funcmatch
                                                 , G__TRYMEMFUNC);
                     if (*known3) {
                        // --
#ifdef G__ASM
                        if (G__asm_noverflow) {
                           // We are generating bytecode.
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(
                                   G__serr
                                 , "%3x,%3x: POPSTROS  %s:%d\n"
                                 , G__asm_cp
                                 , G__asm_dt
                                 , __FILE__
                                 , __LINE__
                              );
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__POPSTROS;
                           G__inc_cp_asm(1, 0);
                        }
#endif // G__ASM
                        break;
                     }
                  }
                  G__memberfunc_struct_offset = store_memberfunc_struct_offset;
                  G__memberfunc_tagnum = store_memberfunc_tagnum;
                  G__exec_memberfunc = store_exec_memberfunc;
                  G__tagnum = store_tagnum;
                  G__store_struct_offset = store_struct_offset;
               }
               else if (-1 != i && fpara.paran == 1) {
                  G__fprinterr(
                       G__serr
                     , "Error: No matching constructor for explicit "
                       "conversion %s"
                     , item
                  );
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
         }
         i++;
      } /* while(i<G__struct.alltag) */
      result3.ref = 0;
      if (G__explicit_fundamental_typeconv(funcname, classhash, &fpara, &result3)) {
         *known3 = 1;
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, &fpara);
         }
         return(result3);
      }
   } /* if(G__TRYNORMAL==memfunc_flag) */
   if (G__parenthesisovld(&result3, funcname, &fpara, G__TRYNORMAL)) {
      *known3 = 1;
      if (nindex &&
            (isupper(result3.type) || 'u' == result3.type)
         ) {
         G__getindexedvalue(&result3, fpara.parameter[nindex]);
      }
      else if (nindex && 'u' == result3.type) {
         G__strlcpy(fpara.parameter[0], fpara.parameter[nindex] + 1, G__ONELINE);
         int len = strlen(fpara.parameter[0]);
         if (len > 1) {
            fpara.parameter[0][len-1] = 0;
         }
         fpara.para[0] = G__getexpr(fpara.parameter[0]);
         fpara.paran = 1;
         G__parenthesisovldobj(&result3, &result3, "operator[]"
                               , &fpara, G__TRYNORMAL);
      }
      if (oprp) {
         *known3 = G__additional_parenthesis(&result3, &fpara);
      }
      return result3;
   }
   /********************************************************************
    * pointer to function described like normal function
    * int (*p2f)(void);  p2f();
    ********************************************************************/
   var = G__getvarentry(funcname, hash, &ig15, &G__global, G__p_local);
   if (var && var->type[ig15] == '1') {
      result7.Format("*%s", funcname());
      *known3 = 0;
      pfparam = (char*)strchr(item, '(');
      if (pfparam) {
         p2ffpara = &fpara;
         result3 = G__pointer2func((G__value*)NULL, result7, pfparam, known3);
      }
      p2ffpara = (struct G__param*)NULL;
      if (*known3) {
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         if (nindex &&
               (isupper(result3.type) || 'u' == result3.type)
            ) {
            G__getindexedvalue(&result3, fpara.parameter[nindex]);
         }
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, &fpara);
         }
         return(result3);
      }
   }
   *known3 = 0;
   /* bug fix, together with G__OLDIMPLEMENTATION29 */
#ifdef G__DUMPFILE
   if (G__dumpfile != NULL && 0 == G__no_exec_compile) {
      G__dumpspace -= 3;
   }
#endif // G__DUMPFILE
   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   if (!G__oprovld) {
      if (G__asm_noverflow && fpara.paran) {
         G__asm_cp = store_cp_asm;
      }
      G__asm_clear_mask = 1;
      result3 = G__execfuncmacro(item, known3);
      G__asm_clear_mask = 0;
      if (*known3) {
         if (nindex &&
               (isupper(result3.type) || 'u' == result3.type)
            ) {
            G__getindexedvalue(&result3, fpara.parameter[nindex]);
         }
         if (oprp) {
            *known3 = G__additional_parenthesis(&result3, &fpara);
         }
         return(result3);
      }
   }
   return G__null;
}

typedef struct {
   struct G__param* libp;
   int ip;
} G__va_list;

/******************************************************************
* G__va_start
*****************************************************************/
void G__va_start(G__value ap)
{
   struct G__var_array* local;
   struct G__ifunc_table_internal* ifunc;
   G__va_list* va;
   local = G__p_local;
   if (!local) {
      return;
   }
   ifunc = G__get_ifunc_internal(local->ifunc);
   if (!ifunc) {
      return;
   }
   va = (G__va_list*)ap.ref;
   if (!va) {
      return;
   }
   va->libp = local->libp;
   va->ip = ifunc->para_nu[local->ifn];
}

#if defined(_MSC_VER) && (_MSC_VER>1200)
#pragma optimize("g",off)
#endif // _MSC_VER

/******************************************************************
 * G__va_arg
 *****************************************************************/
G__value G__va_arg(G__value ap)
{
   // --
#if defined(_MSC_VER) && (_MSC_VER==1300)
   G__genericerror("Error: You can not use va_arg because of "
                   "VC++7.0(_MSC_VER==1300) problem");
   return G__null;
#else // _MSC_VER
   G__va_list* va = (G__va_list*) ap.ref;
   if (!va || !va->libp) {
      return G__null;
   }
   return va->libp->para[va->ip++];
#endif // _MSC_VER
   // --
}

/******************************************************************
 * G__va_end
 *****************************************************************/
void G__va_end(G__value ap)
{
   G__va_list* va = (G__va_list*) ap.ref;
   if (!va) {
      return;
   }
   va->libp = 0;
}

/******************************************************************
 * int G__special_func(result7,funcname,libp)
 ******************************************************************/
int G__special_func(G__value* result7, char* funcname, G__param* libp, int hash)
{
   //  return 1 if function is executed
   //  return 0 if function isn't executed
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
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: LD %ld  %s:%d\n", G__asm_cp, G__asm_dt, G__int(*result7), __FILE__, __LINE__);
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
            G__fprinterr(G__serr, "%3x,%3x: LD %ld  %s:%d\n", G__asm_cp, G__asm_dt, G__int(*result7), __FILE__, __LINE__);
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
      result7->typenum = -1;
      result7->type = 'u';
      if (G__no_exec_compile) {
         result7->tagnum = G__defined_tagname("type_info", 0);
         return(1);
      }
      if (libp->paran > 1) {
         G__letint(result7, 'u', (long)G__typeid(G__catparam(libp, libp->paran, ",")));
      }
      else {
         G__letint(result7, 'u', (long)G__typeid(libp->parameter[0]));
      }
      result7->ref = result7->obj.i;
      result7->tagnum = *(int*)(result7->ref);
      return 1;
   }
#endif // G__TYPEINFO
   if ((hash == 624) && !strcmp(funcname, "va_arg")) {
      G__value x;
      if (!libp->para[0].type) {
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
         G__asm_inst[G__asm_cp+1] = (long) &G__asm_name[G__asm_name_p];
         G__asm_inst[G__asm_cp+2] = hash;
         G__asm_inst[G__asm_cp+3] = 1;
         G__asm_inst[G__asm_cp+4] = (long) G__special_func;
         G__asm_inst[G__asm_cp+5] = 0;
         G__asm_inst[G__asm_cp+6] = (long)G__p_ifunc;
         G__asm_inst[G__asm_cp+7] = -1;
         G__asm_stack[G__asm_dt] = x;
         if (!G__p_ifunc) {
            printf("Serious trouble func 3519\n");
         }
         if (G__strlcpy(G__asm_name + G__asm_name_p, funcname, G__ASM_FUNCNAMEBUF - G__asm_name_p) < (size_t)(G__ASM_FUNCNAMEBUF - G__asm_name_p)) {
            G__asm_name_p += strlen(funcname) + 1;
            G__inc_cp_asm(8, 0);
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

/******************************************************************
 * int G__defined
 ******************************************************************/
int G__defined(char* tname)
{
   int tagnum, typenum;
   typenum = G__defined_typename(tname);
   if (-1 != typenum) {
      return 1;
   }
   tagnum = G__defined_tagname(tname, 2);
   if (-1 != tagnum) {
      return 1;
   }
   return 0;
}

/******************************************************************
 * int G__library_func(result7,funcname,libp,hash)
 ******************************************************************/
int G__library_func(G__value* result7, char* funcname, G__param* libp, int hash)
{
   /*  return 1 if function is executed */
   /*  return 0 if function isn't executed */
   char temp[G__LONGLINE] ;
   /* FILE *fopen(); */
   static int first_getopt = 1;
   extern int optind;
   extern char* optarg;
#ifdef G__NO_STDLIBS
   return(0);
#endif
   *result7 = G__null;
   /*********************************************************************
    * high priority
    *********************************************************************/
   if (638 == hash && strcmp(funcname, "sscanf") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      /* this is a fake function. not compatible with real func */
      /* para0 scan string , para1 format , para2 var pointer */
      G__CHECKNONULL(0, 'C');
      G__CHECKNONULL(1, 'C');
      if (G__checkscanfarg("sscanf", libp, 2)) {
         return(1);
      }
      switch (libp->paran) {
         case 2:
            G__fprinterr(G__serr, "sscanf: no target variable given!");
            G__genericerror((char*)NULL);
            break;
         case 3:
            G__letint(result7, 'i'
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2]))) ;
            break;
         case 4:
            G__letint(result7, 'i'
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3]))) ;
            break;
         case 5:
            G__letint(result7, 'i'
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4]))) ;
            break;
         case 6:
            G__letint(result7, 'i'
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5]))) ;
            break;
         case 7:
            G__letint(result7, 'i'
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6]))) ;
            break;
         case 8:
            G__letint(result7, 'i'
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7]))) ;
            break;
         case 9:
            G__letint(result7, 'i'
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
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
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
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
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
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
                      , sscanf((char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
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
      if (G__no_exec_compile) {
         return(1);
      }
      /* this is a fake function. not compatible with real func */
      /* para0 scan string , para1 format , para2 var pointer */
      G__CHECKNONULL(0, 'E');
      G__CHECKNONULL(1, 'C');
      if (G__checkscanfarg("fscanf", libp, 2)) {
         return(1);
      }
      switch (libp->paran) {
         case 2:
            G__fprinterr(G__serr, "fscanf: no target variable given!");
            G__genericerror((char*)NULL);
            break;
         case 3:
            G__letint(result7, 'i'
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2]))) ;
            break;
         case 4:
            G__letint(result7, 'i'
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3]))) ;
            break;
         case 5:
            G__letint(result7, 'i'
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4]))) ;
            break;
         case 6:
            G__letint(result7, 'i'
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5]))) ;
            break;
         case 7:
            G__letint(result7, 'i'
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6]))) ;
            break;
         case 8:
            G__letint(result7, 'i'
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6])
                               , G__int(libp->para[7]))) ;
            break;
         case 9:
            G__letint(result7, 'i'
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
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
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
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
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
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
                      , fscanf((FILE*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , (char*)G__int(libp->para[1])
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
      if (G__no_exec_compile) {
         return(1);
      }
      /* this is a fake function. not compatible with real func */
      /* para0 scan string , para1 format , para2 var pointer */
      G__CHECKNONULL(0, 'C');
      if (G__checkscanfarg("scanf", libp, 1)) {
         return(1);
      }
      switch (libp->paran) {
         case 1:
            G__fprinterr(G__serr, "scanf: no target variable given!");
            G__genericerror((char*)NULL);
            break;
         case 2:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , G__int(libp->para[1]))) ;
            break;
         case 3:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , G__int(libp->para[1])
                               , G__int(libp->para[2]))) ;
            break;
         case 4:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3]))) ;
            break;
         case 5:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4]))) ;
            break;
         case 6:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5]))) ;
            break;
         case 7:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
                               , G__int(libp->para[1])
                               , G__int(libp->para[2])
                               , G__int(libp->para[3])
                               , G__int(libp->para[4])
                               , G__int(libp->para[5])
                               , G__int(libp->para[6]))) ;
            break;
         case 8:
            G__letint(result7, 'i'
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
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
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
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
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
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
                      , fscanf(G__intp_sin, (char*)G__int(libp->para[0])   // This is an explicit user request ; this can not be avoided.
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
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      /* para[0]:description, para[1~paran-1]: */
      G__charformatter(0, libp, temp, sizeof(temp));
      G__letint(result7, 'i', fprintf(G__intp_sout, "%s", temp));
      return(1);
   }
   if (761 == hash && strcmp(funcname, "fprintf") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'E');
      G__CHECKNONULL(1, 'C');
      /* parameter[0]:pointer ,parameter[1]:description, para[2~paran-1]: */
      G__charformatter(1, libp, temp, sizeof(temp));
      G__letint(result7, 'i',
                fprintf((FILE*)G__int(libp->para[0]), "%s", temp));
      return(1);
   }
   if (774 == hash && strcmp(funcname, "sprintf") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__CHECKNONULL(1, 'C');
      /* parameter[0]:charname ,para[1]:description, para[2~paran-1]: */
      G__charformatter(1, libp, temp, sizeof(temp));
      G__letint(result7, 'i',
                sprintf((char*)G__int(libp->para[0]), "%s", temp));   // This is an explicit user request ; this can not be avoided.
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
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__storerewindposition();
      *result7 = G__calc_internal((char*)G__int(libp->para[0]));
      G__security_recover(G__serr);
      return(1);
   }
#ifdef G__SIGNAL
   if (525 == hash && strcmp(funcname, "alarm") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'h', alarm(G__int(libp->para[0])));
      return(1);
   }
   if (638 == hash && strcmp(funcname, "signal") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
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
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(1, 'C');
      G__CHECKNONULL(2, 'C');
      if (first_getopt) {
         first_getopt = 0;
         G__globalvarpointer = (long)(&optind);
         G__var_type = 'i';
         G__abortbytecode();
         G__getexpr("optind=1");
         G__asm_noverflow = 1;
         G__globalvarpointer = (long)(&optarg);
         G__var_type = 'C';
         G__getexpr("optarg=");
      }
      G__letint(result7, 'c',
                (long)getopt((int)G__int(libp->para[0])
                             , (char**)G__int(libp->para[1])
                             , (char*)G__int(libp->para[2])));
      return(1);
   }
   if (hash == 868 && strcmp(funcname, "va_start") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__va_start(libp->para[0]);
      *result7 = G__null;
      return(1);
   }
   if (hash == 621 && strcmp(funcname, "va_end") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__va_end(libp->para[0]);
      *result7 = G__null;
      return(1);
   }
   if (hash == 1835 && strcmp(funcname, "G__va_arg_setalign") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__va_arg_setalign((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (1093 == hash && strcmp(funcname, "G__loadfile") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'i'
                , (long)G__loadfile((char*)G__int(libp->para[0])));
      return(1);
   }
   if (1320 == hash && strcmp(funcname, "G__unloadfile") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'i', (long)G__unloadfile((char*)G__int(libp->para[0])));
      return(1);
   }
   if (1308 == hash && strcmp(funcname, "G__reloadfile") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__unloadfile((char*)G__int(libp->para[0]));
      G__letint(result7, 'i'
                , (long)G__loadfile((char*)G__int(libp->para[0])));
      return(1);
   }
   if (1882 == hash && strcmp(funcname, "G__set_smartunload") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
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
      G__charformatter((int)G__int(libp->para[0]), G__p_local->libp
                       , (char*)G__int(libp->para[1]), G__int(libp->para[2]));
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
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      /* para[0]:description, para[1~paran-1]: */
      G__charformatter(0, libp, temp, sizeof(temp));
      G__letint(result7, 'i', G__fprinterr(G__serr, "%s", temp));
      return(1);
   }
#endif
   /*********************************************************************
    * low priority 2
    *********************************************************************/
   if (569 == hash && strcmp(funcname, "qsort") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
#ifndef G__MASKERROR
      if (4 != libp->paran) {
         G__printerror("qsort", 1, libp->paran);
      }
#endif
      G__CHECKNONULL(3, 'Y');
      qsort((void*)G__int(libp->para[0])
            , (size_t)G__int(libp->para[1])
            , (size_t)G__int(libp->para[2])
            , (int(*)(const void*, const void*))G__int(libp->para[3])
            /* ,(int (*)(void *arg1,void *argv2))G__int(libp->para[3]) */
           );
      *result7 = G__null;
      return(1);
   }
   if (728 == hash && strcmp(funcname, "bsearch") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
#ifndef G__MASKERROR
      if (5 != libp->paran) {
         G__printerror("bsearch", 1, libp->paran);
      }
#endif
      G__CHECKNONULL(3, 'Y');
      void* ret = bsearch((void*)G__int(libp->para[0])
                          , (void*)G__int(libp->para[1])
                          , (size_t)G__int(libp->para[2])
                          , (size_t)G__int(libp->para[3])
                          , (int(*)(const void*, const void*))G__int(libp->para[4])
                         );
      G__letint(result7, 'Y', (long)ret);
      return(1);
   }
   if (strcmp(funcname, "$read") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__textprocessing((FILE*)G__int(libp->para[0])));
      return(1);
   }
#if defined(G__REGEXP) || defined(G__REGEXP1)
   if (strcmp(funcname, "$regex") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
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
      if (G__no_exec_compile) {
         return(1);
      }
      *result7 = G__getrsvd((int)G__int(libp->para[0]));
      return(1);
   }
   if (1631 == hash && strcmp(funcname, "G__exec_tempfile") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      *result7 = G__exec_tempfile((char*)G__int(libp->para[0]));
      return(1);
   }
#ifndef G__OLDIMPLEMENTATION1546
   if (1225 == hash && strcmp(funcname, "G__load_text") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__storerewindposition();
      G__letint(result7, 'C', (long)G__load_text((char*)G__int(libp->para[0])));
      G__security_recover(G__serr);
      return(1);
   }
#endif
   if (1230 == hash && strcmp(funcname, "G__exec_text") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__storerewindposition();
      *result7 = G__exec_text((char*)G__int(libp->para[0]));
      G__security_recover(G__serr);
      return(1);
   }
   if (1431 == hash && strcmp(funcname, "G__process_cmd") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
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
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__CHECKNONULL(1, 'C');
      G__storerewindposition();
      G__letint(result7, 'C',
                (long)G__exec_text_str((char*)G__int(libp->para[0]),
                                       (char*)G__int(libp->para[1])));
      G__security_recover(G__serr);
      return(1);
   }
#endif
   /*********************************************************************
    * low priority
    *********************************************************************/
   if ((hash == 442) && !strcmp(funcname, "exit")) {
      if (G__no_exec_compile) {
         return 1;
      }
      if (G__atexit) {
         G__call_atexit();
      }
      G__return = G__RETURN_EXIT2;
      G__letint(result7, 'i', G__int(libp->para[0]));
      G__exitcode = result7->obj.i;
      return 1;
   }
   if (655 == hash && strcmp(funcname, "atexit") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      if (G__int(libp->para[0]) == 0) {
         /* function wasn't registered */
         G__letint(result7, 'i', 1);
      }
      else {
         /* function was registered */
         G__atexit = (char*)G__int(libp->para[0]);
         G__letint(result7, 'i', 0);
      }
      return(1);
   }
   if (((hash == 466) && (strcmp(funcname, "ASSERT") == 0)) ||
         ((hash == 658) && (strcmp(funcname, "assert") == 0)) ||
         ((hash == 626) && (strcmp(funcname, "Assert") == 0))) {
      if (G__no_exec_compile) {
         return(1);
      }
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
      if (G__no_exec_compile) {
         return(1);
      }
      /* pause */
      G__letint(result7, 'i', (long)G__pause());
      return(1);
   }
   if (1443 == hash && strcmp(funcname, "G__set_atpause") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_atpause((void(*)(void))G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (1455 == hash && strcmp(funcname, "G__set_aterror") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_aterror((void(*)(void))G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (821 == hash && strcmp(funcname, "G__input") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'C', (long)G__input((char*)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__add_ipath") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__add_ipath((char*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__delete_ipath") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__delete_ipath((char*)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__SetCINTSYSDIR") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__SetCINTSYSDIR((char*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__set_eolcallback") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_eolcallback((void*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__set_history_size") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_history_size((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
#ifndef G__OLDIMPLEMENTATION1485
   if (strcmp(funcname, "G__set_errmsgcallback") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_errmsgcallback((void*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
#endif
   if (strcmp(funcname, "G__chdir") == 0) {
#if defined(G__WIN32) || defined(G__POSIX)
      char* stringb = (char*)G__int(libp->para[0]);
#endif
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
#if defined(G__WIN32)
      if (FALSE == SetCurrentDirectory(stringb)) {
         G__fprinterr(G__serr, "can not change directory to %s\n", stringb);
      }
#elif defined(G__POSIX)
      if (0 != chdir(stringb)) {
         G__fprinterr(G__serr, "can not change directory to %s\n", stringb);
      }
#endif
      *result7 = G__null;
      return(1);
   }
#ifdef G__SHMGLOBAL
   if (strcmp(funcname, "G__shminit") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__shminit();
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__shmmalloc") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      *result7 = G__null;
      G__letint(result7, 'E', (long)G__shmmalloc((int)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__shmcalloc") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      *result7 = G__null;
      G__letint(result7, 'E', (long)G__shmcalloc((int)G__int(libp->para[0]), (int)G__int(libp->para[1])));
      return(1);
   }
#endif
   if (strcmp(funcname, "G__setautoconsole") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__setautoconsole((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__AllocConsole") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__AllocConsole());
      return(1);
   }
   if (strcmp(funcname, "G__FreeConsole") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__FreeConsole());
      return(1);
   }
#ifdef G__TYPEINFO
   if (strcmp(funcname, "G__type2string") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'C' , (long)G__type2string((int)G__int(libp->para[0]),
                (int)G__int(libp->para[1]),
                (int)G__int(libp->para[2]),
                (int)G__int(libp->para[3]),
                (int)G__int(libp->para[4])));
      return(1);
   }
   if (strcmp(funcname, "G__typeid") == 0) {
      result7->typenum = -1;
      result7->type = 'u';
      if (G__no_exec_compile) {
         result7->tagnum = G__defined_tagname("type_info", 0);
         return(1);
      }
      G__letint(result7, 'u', (long)G__typeid((char*)G__int(libp->para[0])));
      result7->ref = result7->obj.i;
      result7->tagnum = *(int*)(result7->ref);
      return(1);
   }
#endif
   if (strcmp(funcname, "G__get_classinfo") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'l' , G__get_classinfo((char*)G__int(libp->para[0]),
                (int)G__int(libp->para[1])));
      return(1);
   }
   if (strcmp(funcname, "G__get_variableinfo") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'l' , G__get_variableinfo((char*)G__int(libp->para[0]),
                (long*)G__int(libp->para[1]),
                (long*)G__int(libp->para[2]),
                (int)G__int(libp->para[3])));
      return(1);
   }
   if (strcmp(funcname, "G__get_functioninfo") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'l' , G__get_functioninfo((char*)G__int(libp->para[0]),
                (long*)G__int(libp->para[1]),
                (long*)G__int(libp->para[2]),
                (int)G__int(libp->para[3])));
      return(1);
   }
   if (strcmp(funcname, "G__lasterror_filename") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'C', (long)G__lasterror_filename());
      return(1);
   }
   if (strcmp(funcname, "G__lasterror_linenum") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__lasterror_linenum());
      return(1);
   }
   if (strcmp(funcname, "G__loadsystemfile") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__CHECKNONULL(0, 'C');
      G__letint(result7, 'i'
                , (long)G__loadsystemfile((char*)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__set_ignoreinclude") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_ignoreinclude((G__IgnoreInclude)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__ForceBytecodecompilation") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__ForceBytecodecompilation((char*)G__int(libp->para[0])
                      , (char*)G__int(libp->para[1])
                                                   ));
      return(1);
   }
#ifndef G__SMALLOBJECT
#ifdef G__TRUEP2F
   if (strcmp(funcname, "G__p2f2funcname") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'C'
                , (long)G__p2f2funcname((void*)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__isinterpretedp2f") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__isinterpretedp2f((void*)G__int(libp->para[0])));
      return(1);
   }
#endif
   if (strcmp(funcname, "G__deleteglobal") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__deleteglobal((void*)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__display_tempobject") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      *result7 = G__null;
      G__display_tempobject("");
      return(1);
   }
   if (strcmp(funcname, "G__cmparray") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__cmparray((short*)G__int(libp->para[0])
                                    , (short*)G__int(libp->para[1])
                                    , (int)G__int(libp->para[2])
                                    , (short)G__int(libp->para[3])));
      return(1);
   }
   if (strcmp(funcname, "G__setarray") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__setarray((short*)G__int(libp->para[0])
                  , (int)G__int(libp->para[1])
                  , (short)G__int(libp->para[2])
                  , (char*)G__int(libp->para[3]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__deletevariable") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__deletevariable((char*)G__int(libp->para[0])));
      return(1);
   }
#ifndef G__NEVER
   if (strcmp(funcname, "G__split") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__split((char*)G__int(libp->para[0]),
                                 (char*)G__int(libp->para[1]),
                                 (int*)G__int(libp->para[2]),
                                 (char**)G__int(libp->para[3])));
      return(1);
   }
   if (strcmp(funcname, "G__readline") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
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
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__setbreakpoint((char*)G__int(libp->para[0]), (char*)G__int(libp->para[1])));
      return(1);
   }
   if (strcmp(funcname, "G__tracemode") == 0 ||
         strcmp(funcname, "G__debugmode") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__tracemode((int)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__stepmode") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__stepmode((int)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__gettracemode") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__gettracemode());
      return(1);
   }
   if (strcmp(funcname, "G__getstepmode") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__getstepmode());
      return(1);
   }
#ifndef G__OLDIMPLEMENTATION2226
   if (strcmp(funcname, "G__setmemtestbreak") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      *result7 = G__null;
      G__setmemtestbreak((int)G__int(libp->para[0]), (int)G__int(libp->para[1]));
      return(1);
   }
#endif
   if (strcmp(funcname, "G__optimizemode") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__optimizemode((int)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__getoptimizemode") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__getoptimizemode());
      return(1);
   }
   if (strcmp(funcname, "G__bytecodedebugmode") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__bytecodedebugmode((int)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__getbytecodedebugmode") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__getbytecodedebugmode());
      return(1);
   }
   if (strcmp(funcname, "G__clearerror") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__return = G__RETURN_NON;
      G__security_error = G__NOERROR;
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__setbreakpoint") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__setbreakpoint((char*)G__int(libp->para[0])
                                         , (char*)G__int(libp->para[1])));
      return(1);
   }
   if (strcmp(funcname, "G__showstack") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__showstack((FILE*)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__graph") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__graph((double*)G__int(libp->para[0]),
                                 (double*)G__int(libp->para[1]),
                                 (int)G__int(libp->para[2]),
                                 (char*)G__int(libp->para[3]),
                                 (int)G__int(libp->para[4])));
      return(1);
   }
#ifndef G__NSEARCHMEMBER
   if (strcmp(funcname, "G__search_next_member") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'C'
                , (long)G__search_next_member((char*)G__int(libp->para[0])
                                              , (int)G__int(libp->para[1])));
      return(1);
   }
   if (strcmp(funcname, "G__what_type") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'Y', (long)G__what_type((char*)G__int(libp->para[0])
                , (char*)G__int(libp->para[1])
                , (char*)G__int(libp->para[2])
                , (char*)G__int(libp->para[3])
                                                ));
      return(1);
   }
   if (strcmp(funcname, "G__SetCatchException") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__SetCatchException((int)G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
#endif
#ifndef G__NSTOREOBJECT
   if (strcmp(funcname, "G__storeobject") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__storeobject((&libp->para[0]),
                (&libp->para[1])));
      return(1);
   }
   if (strcmp(funcname, "G__scanobject") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__scanobject((&libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__dumpobject") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__dumpobject((char*)G__int(libp->para[0])
                                      , (void*)G__int(libp->para[1])
                                      , (int)G__int(libp->para[2])));
      return(1);
   }
   if (strcmp(funcname, "G__loadobject") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__loadobject((char*)G__int(libp->para[0])
                                      , (void*)G__int(libp->para[1])
                                      , (int)G__int(libp->para[2])));
      return(1);
   }
#endif
   if (strcmp(funcname, "G__lock_variable") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__lock_variable((char*)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__unlock_variable") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__unlock_variable((char*)G__int(libp->para[0])));
      return(1);
   }
   if (strcmp(funcname, "G__dispvalue") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i'
                , (long)G__dispvalue((FILE*)libp->para[0].obj.i, &libp->para[1]));
      return(1);
   }
   if (strcmp(funcname, "G__set_class_autoloading_table") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_class_autoloading_table((char*)G__int(libp->para[0])
                                     , (char*)G__int(libp->para[1]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__defined") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__letint(result7, 'i', (long)G__defined((char*)G__int(libp->para[0])));
      return(1);
   }
#endif /* G__SMALLOBJECT */
   if (strcmp(funcname, "G__set_alloclockfunc") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_alloclockfunc((void(*)())G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   if (strcmp(funcname, "G__set_allocunlockfunc") == 0) {
      if (G__no_exec_compile) {
         return(1);
      }
      G__set_allocunlockfunc((void(*)())G__int(libp->para[0]));
      *result7 = G__null;
      return(1);
   }
   return(0);
}

/******************************************************************
 * G__printf_error()
 ******************************************************************/
void G__printf_error()
{
   G__fprinterr(G__serr, "Limitation: printf string too long. "
                "Upto %d. Use fputs()", G__LONGLINE);
   G__genericerror(0);
}

#define G__PRINTF_ERROR(COND) \
if (COND) { \
   G__printf_error(); \
   return result; \
}

static void G__sprintformatll(char* result, size_t result_length, const char* fmt, void* p, G__FastAllocString& buf)
{
   G__int64* pll = (G__int64*)p;
   buf.Format(fmt, result, *pll);
   G__strlcpy(result, buf, result_length);
}

static void G__sprintformatull(char* result, size_t result_length, const char* fmt, void* p, G__FastAllocString& buf)
{
   G__uint64* pll = (G__uint64*)p;
   buf.Format(fmt, result, *pll);
   G__strlcpy(result, buf, result_length);
}

static void G__sprintformatld(char* result, size_t result_length, const char* fmt, void* p, G__FastAllocString& buf)
{
   long double* pld = (long double*)p;
   buf.Format(fmt, result, *pld);
   G__strlcpy(result, buf, result_length);
}

/******************************************************************
 * char *G__charformatter(ifmt,libp,outbuf)
 ******************************************************************/
char* G__charformatter(int ifmt, G__param* libp, char* result, size_t result_length)
{
   int ipara, ichar, lenfmt;
   int ionefmt = 0, fmtflag = 0;
   G__FastAllocString onefmt(G__LONGLINE);
   G__FastAllocString fmt(G__LONGLINE);
   G__FastAllocString pformat(G__LONGLINE);
   short dig = 0;
   int usedpara = 0;
   pformat = (char*)G__int(libp->para[ifmt]);
   result[0] = '\0';
   ipara = ifmt + 1;
   lenfmt = strlen(pformat);
   for (ichar = 0; ichar <= lenfmt; ichar++) {
      switch (pformat[ichar]) {
         case '\0': /* end of the format */
            onefmt.Set(ionefmt, 0);
            fmt = "%s";
            fmt += onefmt;
            onefmt.Format(fmt(), result);
            G__strlcpy(result, onefmt, result_length);
            ionefmt = 0;
            break;
         case 's': /* string */
            onefmt.Set(ionefmt++, pformat[ichar]);
            if (fmtflag == 1) {
               onefmt.Set(ionefmt, 0);
               if (libp->para[ipara].obj.i) {
                  G__PRINTF_ERROR(strlen(onefmt) + strlen(result) +
                                  strlen((char*)G__int(libp->para[usedpara])) >= G__LONGLINE) {
                     fmt = "%s";
                     fmt += onefmt;
                  }
                  onefmt.Format(fmt(), result , (char*)G__int(libp->para[usedpara]));
                  G__strlcpy(result, onefmt, result_length);
               }
               ipara++;
               ionefmt = 0;
               fmtflag = 0;
            }
            break;
         case 'c': /* char */
            onefmt.Set(ionefmt++, pformat[ichar]);
            if (fmtflag == 1) {
               onefmt.Set(ionefmt, 0);
               fmt = "%s";
               fmt += onefmt;
               onefmt.Format(fmt(), result , (char)G__int(libp->para[usedpara]));
               G__strlcpy(result, onefmt, result_length);
               ipara++;
               ionefmt = 0;
               fmtflag = 0;
            }
            break;
         case 'b': /* int */
            onefmt.Set(ionefmt++, pformat[ichar]);
            if (fmtflag == 1) {
               onefmt.Set(ionefmt - 1, 's');
               onefmt.Set(ionefmt, 0);
               fmt = "%s";
               fmt += onefmt;
               G__logicstring(libp->para[usedpara], dig, onefmt);
               ipara++;
               G__FastAllocString resBuf;
               resBuf.Format(fmt(), result, onefmt());
               G__strlcpy(result, resBuf, result_length);
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
            onefmt.Set(ionefmt++, pformat[ichar]);
            if (fmtflag == 1) {
               onefmt.Set(ionefmt, 0);
               fmt = "%s";
               fmt += onefmt;
               if ('n' == libp->para[usedpara].type) {
                  G__value* pval = &libp->para[usedpara];
                  ipara++;
                  G__sprintformatll(result, result_length, fmt, &pval->obj.ll, onefmt);
               }
               else if ('m' == libp->para[usedpara].type) {
                  G__value* pval = &libp->para[usedpara];
                  ipara++;
                  G__sprintformatull(result, result_length, fmt, &pval->obj.ull, onefmt);
               }
               else if (
                  'u' == libp->para[usedpara].type
               ) {
                  G__FastAllocString llbuf(100);
                  G__value* pval = &libp->para[usedpara];
                  ipara++;
                  if (strcmp(G__struct.name[pval->tagnum], "G__longlong") == 0) {
                     llbuf.Format("G__printformatll((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
                                  , (long)fmt(), (long)onefmt(), pval->obj.i);
                     G__getitem(llbuf);
                     G__strlcat(result, fmt, result_length);
                  }
                  else if (strcmp(G__struct.name[pval->tagnum], "G__ulonglong") == 0) {
                     llbuf.Format("G__printformatull((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
                                  , (long)fmt(), (long)onefmt(), pval->obj.i);
                     G__getitem(llbuf);
                     G__strlcat(result, fmt, result_length);
                  }
                  else {
                     ++usedpara;
                     onefmt.Format(fmt(), result, G__int(libp->para[usedpara]));
                     ipara++;
                     G__strlcpy(result, onefmt, result_length);
                  }
               }
               else {
                  onefmt.Format(fmt(), result, G__int(libp->para[usedpara]));
                  ipara++;
                  G__strlcpy(result, onefmt, result_length);
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
            onefmt.Set(ionefmt++, pformat[ichar]);
            if (fmtflag == 1) {
               onefmt.Set(ionefmt, 0);
               fmt = "%s";
               fmt += onefmt;
               if ('q' == libp->para[usedpara].type) {
                  G__value* pval = &libp->para[usedpara];
                  ipara++;
                  G__sprintformatld(result, result_length, fmt, &pval->obj.ld, onefmt);
               }
               else if (
                  'u' == libp->para[usedpara].type
               ) {
                  G__FastAllocString llbuf(100);
                  G__value* pval = &libp->para[usedpara];
                  ipara++;
                  if (strcmp(G__struct.name[pval->tagnum], "G__longdouble") == 0) {
                     llbuf.Format("G__printformatld((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
                                  , (long)fmt(), (long)onefmt(), pval->obj.i);
                     G__getitem(llbuf);
                     G__strlcat(result, fmt, result_length);
                  }
                  else {
                     ++usedpara;
                     onefmt.Format(fmt(), result, G__double(libp->para[usedpara]));
                     ipara++;
                     G__strlcpy(result, onefmt, result_length);
                  }
               }
               else {
                  onefmt.Format(fmt(), result, G__double(libp->para[usedpara]));
                  ipara++;
                  G__strlcpy(result, onefmt, result_length);
               }
               ionefmt = 0;
               fmtflag = 0;
            }
            break;
         case 'L': /* long double */
#ifdef G__OLDIMPLEMENTATION2189_YET
            if ('q' == libp->para[usedpara].type) {
               G__value* pval = &libp->para[usedpara];
               ipara++;
               G__sprintformatld(result, result_length, fmt(), &pval->obj.ld, result, onefmt);
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
            // intentional fall-through, need to put digit into onefmt.
         case '#': // "alternate form"
         case '.':
         case '-':
         case '+':
         case 'l': /* long int */
         case 'h': /* short int unsinged int */
            onefmt.Set(ionefmt++, pformat[ichar]);
            break;
         case '%':
            if (fmtflag == 0) {
               usedpara = ipara;
               fmtflag = 1;
            }
            else {
               fmtflag = 0;
            }
            onefmt.Set(ionefmt++, pformat[ichar]);
            dig = 0;
            break;
         case '*': /* printf("%*s",4,"*"); */
            if (fmtflag == 1) {
               onefmt.Resize(ionefmt + 100); // 100 digits for %ld should suffice
               onefmt.Format(ionefmt, "%ld", G__int(libp->para[usedpara]));
               ipara++;
               usedpara++;
               ionefmt = strlen(onefmt);
            }
            else {
               onefmt.Set(ionefmt++, pformat[ichar]);
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
               while (ionefmt >= 0 && isdigit(onefmt[ionefmt-1])) {
                  --ionefmt;
               }
               dig = 0;
            }
            else {
               onefmt.Set(ionefmt++, pformat[ichar]);
            }
            break;
         case ' ':
         case '\t' : /* tab */
         case '\n': /* end of line */
         case '\r': /* end of line */
         case '\f': /* end of line */
            if (fmtflag) {
               if ('%' != onefmt[ionefmt-1] && !isspace(onefmt[ionefmt-1])) {
                  fmtflag = 0;
               }
               onefmt.Set(ionefmt++, pformat[ichar]);
               break;
            }
         default:
            fmtflag = 0;
            onefmt.Set(ionefmt++, pformat[ichar]);
            break;
      }
   }
   return(result);
}

/******************************************************************
 * G__tracemode()
 ******************************************************************/
int G__tracemode(int tracemode)
{
   G__debug = tracemode;
   G__istrace = tracemode;
   G__setdebugcond();
   return G__debug;
}

/******************************************************************
 * G__stepmode()
 ******************************************************************/
int G__stepmode(int stepmode)
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
   return G__step;
}

/******************************************************************
 * G__gettracemode()
 ******************************************************************/
int G__gettracemode()
{
   return G__debug;
}

/******************************************************************
 * G__getstepmode()
 ******************************************************************/
int G__getstepmode()
{
   return G__step;
}

/******************************************************************
 * G__optmizemode()
 ******************************************************************/
int G__optimizemode(int optimizemode)
{
   G__asm_loopcompile = optimizemode;
   G__asm_loopcompile_mode = G__asm_loopcompile;
   return G__asm_loopcompile;
}

/******************************************************************
 * G__getoptmizemode()
 ******************************************************************/
int G__getoptimizemode()
{
   return G__asm_loopcompile_mode;
}

} // extern "C"

