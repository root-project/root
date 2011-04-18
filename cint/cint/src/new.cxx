/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file new.c
 ************************************************************************
 * Description:
 *  new delete
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Api.h"

extern "C" {

static void G__lock_noop() {}
void (*G__AllocMutexLock)() = G__lock_noop;
void (*G__AllocMutexUnLock)() = G__lock_noop;

void G__exec_alloc_lock() { G__AllocMutexLock(); }
void G__exec_alloc_unlock() { G__AllocMutexUnLock(); }

void G__set_alloclockfunc(void (*foo)()) { G__AllocMutexLock = foo; }
void G__set_allocunlockfunc(void (*foo)()) { G__AllocMutexUnLock = foo; }

//______________________________________________________________________________
G__value G__new_operator(const char* expression)
{
   // -- FIXME: Describe this function!
   // new type
   // new type[10]
   // new type(53)
   // new (arena)type
   //char expression[G__LONGLINE];
   //copy (expression,express);
   G__FastAllocString arena(G__ONELINE);
   long memarena = 0;
   int arenaflag = 0;
   G__FastAllocString construct(G__LONGLINE);
   char *type;
   char *basictype;
   char *initializer;
   char *arrayindex;
   int p = 0;
   int pinc;
   int size;
   int known;
   int i;
   long pointer = 0;
   long store_struct_offset;
   int store_tagnum;
   int store_typenum;
   int var_type = 0;
   G__value result = G__null;
   int reftype = G__PARANORMAL;
   int ispointer = 0;
   int typenum, tagnum;
   int ld_flag = 0 ;
   G__CHECK(G__SECURE_MALLOC, 1, return G__null);
   if (G__cintv6) {
      return G__bc_new_operator(expression);
   }
   //
   //  Get arena which is ignored due to limitation, however.
   //
   if (expression[0] != '(') {
      arena[0] = '\0';
   }
   else {
      G__getstream(expression + 1, &p, arena, ")");
      ++p;
      memarena = G__int(G__getexpr(arena));
      arenaflag = 1;
#ifdef G__ASM
      if (G__asm_noverflow) {
         G__asm_inst[G__asm_cp] = G__SETGVP;
         G__asm_inst[G__asm_cp+1] = 0;
         G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SETGVP 0  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         // --
      }
#endif // G__ASM
      // --
   }
   //
   //  Get initializer, arrayindex, type, pinc and size.
   //
   type = (char*)expression + p;
   initializer = strchr(type, '(');
   arrayindex = strchr(type, '[');
   // The initializer and arrayindex are exclusive.
   if (initializer && arrayindex) {
      if (initializer < arrayindex) {
         arrayindex = 0;
      }
      else {
         initializer = 0;
      }
   }
   if (initializer) {
      *initializer = 0;
   }
   basictype = type;
   {
      size_t len = strlen(type);
      unsigned int nest = 0;
      for (size_t ind = len - 1; ind > 0; --ind) {
         switch (type[ind]) {
            case '<':
               --nest;
               break;
            case '>':
               ++nest;
               break;
            case ':':
               if (!nest && (type[ind-1] == ':')) {
                  basictype = &(type[ind+1]);
                  ind = 1;
                  break;
               }
         }
      }
   }
   if (initializer) {
      *initializer = '(';
   }
   typenum = G__defined_typename(type);
   if (typenum != -1) {
      tagnum = G__newtype.tagnum[typenum];
   }
   else {
      tagnum = -1;
   }
   if (arrayindex) {
      pinc = G__getarrayindex(arrayindex);
      *arrayindex = '\0';
      if (tagnum == -1) {
         tagnum = G__defined_tagname(basictype, 1);
      }
      if (tagnum != -1) {
         construct.Format("%s()", G__struct.name[tagnum]);
      }
      else {
         construct.Format("%s()", basictype);
      }
      if (G__asm_wholefunction) {
         G__abortbytecode();
      }
   }
   else {
      // --
#ifdef G__ASM_IFUNC
      if (G__asm_noverflow) {
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: LD %d from %x  %s:%d\n", G__asm_cp, G__asm_dt, 1, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LD;
         G__asm_inst[G__asm_cp+1] = G__asm_dt;
         G__asm_stack[G__asm_dt].obj.i = 1;
         G__asm_stack[G__asm_dt].type = 'i';
         G__asm_stack[G__asm_dt].tagnum = -1;
         G__asm_stack[G__asm_dt].typenum = -1;
         ld_flag = 1 ;
      }
#endif // G__ASM_IFUNC
      if (initializer) {
         pinc = 1;
         if (tagnum == -1) {
            *initializer = 0;
            tagnum = G__defined_tagname(basictype, 2);
            *initializer = '(';
         }
         if (tagnum != -1) {
            construct.Format("%s%s", G__struct.name[tagnum], initializer);
         }
         else {
            construct = basictype;
         }
         *initializer = '\0';
      }
      else {
         pinc = 1;
         if (tagnum != -1) {
            construct.Format("%s()", G__struct.name[tagnum]);
         }
         else {
            construct.Format("%s()", basictype);
         }
      }
   }
   size = G__Lsizeof(type);
   if (size == -1) {
      G__fprinterr(G__serr, "Error: type %s not defined FILE:%s LINE:%d\n", type, G__ifile.name, G__ifile.line_number);
      return G__null;
   }
   //
   //  Store member function execution environment.
   //
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_typenum = G__typenum;
   result.ref = 0;
   //
   //  Pointer type identification.
   //
   size_t typelen = strlen(type);
   while (type[typelen-1] == '*') {
      if (!ispointer) {
         ispointer = 1;
      }
      else {
         switch (reftype) {
            case G__PARANORMAL:
               reftype = G__PARAP2P;
               break;
            case G__PARAREFERENCE:
               break;
            default:
               ++reftype;
               break;
         }
      }
      type[--typelen] = '\0';
   }
   //
   //  Identify type.
   //
   G__typenum = G__defined_typename(type);
   if (G__typenum != -1) {
      G__tagnum = G__newtype.tagnum[G__typenum];
      var_type = G__newtype.type[G__typenum];
   }
   else {
      G__tagnum = G__defined_tagname(type, 1);
      if (G__tagnum != -1) {
         var_type = 'u';
      }
      else {
         if (!strcmp(type, "int")) {
            var_type = 'i';
         }
         else if (!strcmp(type, "char")) var_type = 'c';
         else if (!strcmp(type, "short")) var_type = 's';
         else if (!strcmp(type, "long")) var_type = 'l';
         else if (!strcmp(type, "float")) var_type = 'f';
         else if (!strcmp(type, "double")) var_type = 'd';
         else if (!strcmp(type, "void")) var_type = 'y';
         else if (!strcmp(type, "FILE")) var_type = 'e';
         else if (!strcmp(type, "unsigned int")) var_type = 'h';
         else if (!strcmp(type, "unsignedchar")) var_type = 'b';
         else if (!strcmp(type, "unsigned short")) var_type = 'r';
         else if (!strcmp(type, "unsigned long")) var_type = 'l';
         else if (!strcmp(type, "size_t")) var_type = 'l';
         else if (!strcmp(type, "time_t")) var_type = 'l';
         else if (!strcmp(type, "bool")) var_type = 'g';
         else if (!strcmp(type, "long long")) var_type = 'n';
         else if (!strcmp(type, "unsigned long long")) var_type = 'm';
         else if (!strcmp(type, "long double")) var_type = 'q';
      }
   }
   if (ispointer) {
      var_type = toupper(var_type);
   }
#ifdef G__ASM
   if (G__asm_noverflow) {
      if (ld_flag) {
         if (
            (G__tagnum == -1) ||
            (G__struct.iscpplink[G__tagnum] != G__CPPLINK) ||
            ispointer ||
            isupper(var_type)
         ) {
            // -- Increment for LD 1, otherwise, cancel LD 1.
            G__inc_cp_asm(2, 1);
         }
         else {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "Cancel LD 1  %s:%d\n", __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            // --
         }
      }
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: SETMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SETMEMFUNCENV;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   //
   //  Allocate memory if this is a class object and not pre-compiled.
   //
   if (
      (G__tagnum == -1) ||
      (G__struct.iscpplink[G__tagnum] != G__CPPLINK) ||
      ispointer ||
      isupper(var_type)
   ) {
      // --
      if (G__no_exec_compile) {
         pointer = pinc;
      }
      else {
         if (arenaflag) {
            pointer = memarena;
         }
         else {
            // --
#ifdef G__ROOT
            pointer = (long) G__new_interpreted_object(size * pinc);
#else // G__ROOT
            pointer = (long) new char[size*pinc];
#endif // G__ROOT
         }
      }
      if (!pointer && !G__no_exec_compile) {
         G__fprinterr(G__serr, "Error: memory allocation for %s %s size=%d pinc=%d FILE:%s LINE:%d\n", type, expression, size, pinc, G__ifile.name, G__ifile.line_number);
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         return G__null;
      }
      G__store_struct_offset = pointer;
#ifdef G__ASM_IFUNC
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: NEWALLOC size: %d cnt: %d  %s:%d\n", G__asm_cp, G__asm_dt, size, pinc, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__NEWALLOC;
         if (memarena) {
            G__asm_inst[G__asm_cp+1] = 0;
         }
         else {
            G__asm_inst[G__asm_cp+1] = size;
         }
         G__asm_inst[G__asm_cp+2] = ((var_type == 'u') && arrayindex) ? 1 : 0;
         G__inc_cp_asm(3, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SET_NEWALLOC tagnum: %d type: '%c'  %s:%d\n", G__asm_cp, G__asm_dt, G__tagnum, toupper(var_type), __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SET_NEWALLOC;
         G__asm_inst[G__asm_cp+1] = G__tagnum;
         G__asm_inst[G__asm_cp+2] = toupper(var_type);
         G__inc_cp_asm(3, 0);
      }
#endif // G__ASM
#endif // G__ASM_IFUNC
      // --
   }
   //
   //  Call constructor if struct, class.
   //
   if (var_type == 'u') {
      if (G__struct.isabstract[G__tagnum]) {
         G__fprinterr(G__serr, "Error: abstract class object '%s' is created", G__struct.name[G__tagnum]);
         G__genericerror(0);
         G__display_purevirtualfunc(G__tagnum);
      }
      if (G__dispsource) {
         G__fprinterr(
              G__serr
            , "\n!!!Calling constructor 0x%lx.'%s' for new '%s'  %s:%d\n"
            , G__store_struct_offset
            , type
            , type
            , __FILE__
            , __LINE__
         );
      }
      if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
         // -- This is a pre-compiled class.
         long store_globalvarpointer = G__globalvarpointer;
         if (memarena) {
            G__globalvarpointer = memarena;
         }
         else {
            G__globalvarpointer = G__PVOID;
         }
         if (arrayindex) {
            G__cpp_aryconstruct = pinc;
#ifdef G__ASM
            if (G__asm_noverflow) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: SETARYINDEX  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__SETARYINDEX;
               G__asm_inst[G__asm_cp+1] = 1;
               G__inc_cp_asm(2, 0);
            }
#endif // G__ASM
            // --
         }
         result = G__getfunction(construct, &known, G__CALLCONSTRUCTOR);
#ifdef G__ASM
         if (arrayindex && G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: RESETARYINDEX  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__RESETARYINDEX;
            G__asm_inst[G__asm_cp+1] = 1;
            G__inc_cp_asm(2, 0);
         }
#endif // G__ASM
         result.type = toupper(result.type);
         result.ref = 0;
         result.isconst = G__VARIABLE;
         G__cpp_aryconstruct = 0;
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__globalvarpointer = store_globalvarpointer;
#ifdef G__ASM
         if (memarena && G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETGVP -1  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETGVP;
            G__asm_inst[G__asm_cp+1] = -1;
            G__inc_cp_asm(2, 0);
         }
#endif // G__ASM
#ifdef G__ASM
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: RECMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__RECMEMFUNCENV;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
#ifdef G__SECURITY
         if (G__security & G__SECURE_GARBAGECOLLECTION) {
            if (!G__no_exec_compile && !memarena) {
               G__add_alloctable((void*)result.obj.i, result.type, result.tagnum);
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: ADDALLOCTABLE  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__ADDALLOCTABLE;
                  G__inc_cp_asm(1, 0);
               }
#endif // G__ASM
               // --
            }
         }
#endif // G__SECURITY
         return result;
      }
      else {
         // -- This is an interpreted class.
         if (arrayindex && !G__no_exec_compile && (G__struct.type[G__tagnum] != 'e')) {
            G__alloc_newarraylist(pointer, pinc);
         }
         G__var_type = 'p';
         for (i = 0; i < pinc; ++i) {
            // --
            G__abortbytecode();
            if (G__no_exec_compile) {
               break;
            }
            G__getfunction(construct, &known, G__TRYCONSTRUCTOR);
            result.ref = 0;
            if (!known) {
               if (initializer) {
                  G__value buf;
                  char* bp = strchr(construct, '(');
                  char* ep = strrchr(construct, ')');
                  G__ASSERT(bp && ep) ;
                  *ep = 0;
                  *bp = 0;
                  ++bp;
                  {
                     int nx = 0;
                     G__FastAllocString tmpx(G__ONELINE);
                     int cx = G__getstream(bp, &nx, tmpx, "),");
                     if (cx == ',') {
                        *ep = ')';
                        *(bp - 1) = '(';
                        // only to display error message
                        G__getfunction(construct, &known, G__CALLCONSTRUCTOR);
                        break;
                     }
                  }
                  // construct = "TYPE" , bp = "ARG"
                  buf = G__getexpr(bp);
                  G__abortbytecode();
                  if ((buf.tagnum != -1) && !G__no_exec_compile) {
                     if (buf.tagnum != G__tagnum) {
                        G__fprinterr(G__serr, "Error: Illegal initialization of %s(", G__fulltagname(G__tagnum, 1));
                        G__fprinterr(G__serr, "%s)", G__fulltagname(buf.tagnum, 1));
                        G__genericerror(0);
                        return G__null;
                     }
                     memcpy((void*)G__store_struct_offset, (void*)buf.obj.i, G__struct.size[buf.tagnum]);
                  }
               }
               break;
            }
            G__store_struct_offset += size;
            //
            // WARNING: FOLLOWING PART MUST BE REDESIGNED TO SUPPORT WHOLE FUNCTION COMPILATION.
            //
#ifdef G__ASM_IFUNC
#ifdef G__ASM
            G__abortbytecode();
            if (G__asm_noverflow) {
               if (pinc > 1) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, size, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__ADDSTROS;
                  G__asm_inst[G__asm_cp+1] = size;
                  G__inc_cp_asm(2, 0);
               }
            }
#endif // G__ASM
#endif // G__ASM_IFUNC
            // --
         }
#ifdef G__ASM
         if (memarena && G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETGVP -1  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETGVP;
            G__asm_inst[G__asm_cp+1] = -1;
            G__inc_cp_asm(2, 0);
         }
#endif // G__ASM
         // --
      }
   }
   else if (initializer) {
      // -- construct = "TYPE(ARG)"
      struct G__param para;
      int typenum;
      int hash;
      char *bp = strchr(construct, '(');
      char *ep = strrchr(construct, ')');
      *ep = 0;
      *bp = 0;
      ++bp;
      // construct = "TYPE" , bp = "ARG"
      typenum = G__defined_typename(construct);
      if (typenum != -1) {
         construct = G__type2string(G__newtype.type[typenum], G__newtype.tagnum[typenum], -1, G__newtype.reftype[typenum], 0);
      }
      hash = (int)strlen(construct);
      char store_var_type = G__var_type;
      G__var_type = 'p';
      para.para[0] = G__getexpr(bp); // generates LD or LD_VAR etc...
      G__var_type = store_var_type;
      if (!G__no_exec_compile) {
         result.ref = pointer;
      }
      else {
         result.ref = 0;
      }
      // Following call generates CAST instruction.
      if ((var_type == 'U') && pointer) {
         if (!G__no_exec_compile) {
            *(long*)pointer = para.para[0].obj.i;
         }
      }
      else {
         G__explicit_fundamental_typeconv(construct, hash, &para, &result);
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: LETNEWVAL  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LETNEWVAL;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
#ifdef G__ASM
      if (memarena && G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SETGVP -1  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SETGVP;
         G__asm_inst[G__asm_cp+1] = -1;
         G__inc_cp_asm(2, 0);
      }
#endif // G__ASM
      // --
   }
   if (isupper(var_type)) {
      G__letint(&result, var_type, pointer);
      switch (reftype) {
         case G__PARANORMAL:
            result.obj.reftype.reftype = G__PARAP2P;
            break;
         case G__PARAP2P:
            result.obj.reftype.reftype = G__PARAP2P2P;
            break;
         default:
            result.obj.reftype.reftype = reftype + 1;
            break;
      }
   }
   else {
      G__letint(&result, toupper(var_type), pointer);
      result.obj.reftype.reftype = reftype;
      if (reftype == G__PARANORMAL) {
         result.ref = 0;
      }
   }
   result.tagnum = G__tagnum;
   result.typenum = G__typenum;
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__typenum = store_typenum;
#ifdef G__ASM
   if (G__asm_noverflow) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x%3x: RECMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__RECMEMFUNCENV;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
#ifdef G__SECURITY
   if (G__security & G__SECURE_GARBAGECOLLECTION) {
      if (!G__no_exec_compile && !memarena)
         G__add_alloctable((void*) result.obj.i, result.type, result.tagnum);
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ADDALLOCTABLE  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ADDALLOCTABLE;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      // --
   }
#endif // G__SECURITY
   return result;
}

//______________________________________________________________________________
int G__getarrayindex(const char* indexlist)
{
   // FIXME: Describe this function!
   // [x][y][z]     get x*y*z
   long p_inc = 1;
   int p = 1;
   G__FastAllocString index(G__ONELINE);
   int c;
   char store_var_type = G__var_type;
   G__var_type = 'p';

   c = G__getstream(indexlist, &p, index, "]");
   p_inc *= G__int(G__getexpr(index));
   while (*(indexlist + p) == '[') {
      ++p;
      c = G__getstream(indexlist, &p, index, "]");
      p_inc *= G__int(G__getexpr(index));
#ifdef G__ASM_IFUNC
#ifdef G__ASM
      if (G__asm_noverflow) {
#ifdef G__ASM_DBG
         if (G__asm_dbg) G__fprinterr(G__serr, "%3x: OP2 *\n" , G__asm_cp);
#endif
         G__asm_inst[G__asm_cp] = G__OP2;
         G__asm_inst[G__asm_cp+1] = (long)'*';
         G__inc_cp_asm(2, 0);
      }
#endif /* G__ASM */
#endif /* G__ASM_IFUNC */
   }
   G__ASSERT(']' == c);

   G__var_type = store_var_type;

   return(p_inc);
}

//______________________________________________________________________________
void G__delete_operator(char* expression, int isarray)
{
   // -- FIXME: Describe this function!
   long store_struct_offset = 0L;
   int store_tagnum = 0;
   int store_typenum = 0;
   int done = 0;
   int pinc = 0;
   int i = 0;
   int size = 0;
   int cpplink = 0;
   int zeroflag = 0;
   G__value buf;
   G__FastAllocString destruct(G__ONELINE);
   if (G__cintv6) {
      // -- THIS CASE IS NEVER USED.
      G__bc_delete_operator(expression, isarray);
      return;
   }
   buf = G__getitem(expression);
   if (islower(buf.type)) {
      G__fprinterr(G__serr, "Error: Cannot delete '%s'", expression);
      G__genericerror(0);
      return;
   }
   else if (!buf.obj.i && !G__no_exec_compile && (G__asm_wholefunction == G__ASM_FUNC_NOP)) {
      zeroflag = 1;
      G__no_exec_compile = 1;
      buf.obj.d = 0;
      buf.obj.i = 1;
   }
   G__CHECK(G__SECURE_MALLOC, 1, return);
#ifdef G__SECURITY
   if (G__security & G__SECURE_GARBAGECOLLECTION) {
      if (!G__no_exec_compile) {
         G__del_alloctable((void*)buf.obj.i);
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: DELALLOCTABLE  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__DELALLOCTABLE;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      // --
   }
#endif // G__SECURITY
   //
   //  Call destructor if struct or class
   //
   if ((buf.type == 'U') && (buf.obj.reftype.reftype == G__PARANORMAL)) {
      store_struct_offset = G__store_struct_offset;
      store_typenum = G__typenum;
      store_tagnum = G__tagnum;
      G__store_struct_offset = buf.obj.i;
      G__typenum = buf.typenum;
      G__tagnum = buf.tagnum;
      destruct.Format("~%s()", G__struct.name[G__tagnum]);
      if (G__dispsource) {
         G__fprinterr(
              G__serr
            , "\n!!!Calling destructor 0x%lx.%s for '%s'  %s:%d\n"
            , G__store_struct_offset
            , destruct()
            , expression
            , __FILE__
            , __LINE__
         );
      }
      done = 0;
      if (
         // --
         !G__no_exec_compile &&
         (G__struct.virtual_offset[G__tagnum] != -1) &&
         (G__tagnum != *(long*)(G__store_struct_offset + G__struct.virtual_offset[G__tagnum]))
      ) {
         // --
         long virtualtag = *(long*)(G__store_struct_offset + G__struct.virtual_offset[G__tagnum]);
         buf.obj.i -= G__find_virtualoffset(virtualtag, buf.obj.i);
      }
      //
      //  Push and set G__store_struct_offset.
      //
#ifdef G__ASM_IFUNC
#ifdef G__ASM
      if (G__asm_noverflow) {
         G__asm_inst[G__asm_cp] = G__PUSHSTROS;
         G__asm_inst[G__asm_cp+1] = G__SETSTROS;
         G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp - 2, G__asm_dt, __FILE__, __LINE__);
            G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp - 1, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         if (isarray) {
            G__asm_inst[G__asm_cp] = G__GETARYINDEX;
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: GETARYINDEX  %s:%d\n", G__asm_cp - 2, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__inc_cp_asm(1, 0);
         }
      }
#endif // G__ASM
#endif // G__ASM_IFUNC
      //
      //  Call destructor.
      //
      if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
         // -- Precompiled class.
         if (isarray) {
            G__cpp_aryconstruct = 1;
         }
#ifndef G__ASM_IFUNC
#ifdef G__ASM
         if (G__asm_noverflow) {
            G__asm_inst[G__asm_cp] = G__PUSHSTROS;
            G__asm_inst[G__asm_cp+1] = G__SETSTROS;
            G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp - 2, G__asm_dt, __FILE__, __LINE__);
               G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp - 1, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            // --
         }
#endif // G__ASM
#endif // G__ASM_IFUNC
         //
         //  Actually do the destructor call now.
         //
         // Note: Precompiled destructor must always exist here.
         G__getfunction(destruct, &done, G__TRYDESTRUCTOR);
#ifndef G__ASM_IFUNC
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
#endif // G__ASM_IFUNC
         G__cpp_aryconstruct = 0;
         cpplink = 1;
      }
      else {
         // -- Interpreted class.
         // WARNING: FOLLOWING PART MUST BE REDESIGNED TO SUPPORT WHOLE FUNCTION COMPILATION.
         if (!isarray) {
            G__getfunction(destruct, &done, G__TRYDESTRUCTOR);
         }
         else {
            if (!G__no_exec_compile) {
               pinc = G__free_newarraylist(G__store_struct_offset);
            }
            else {
               pinc = 1;
            }
            size = G__struct.size[G__tagnum];
            for (i = pinc - 1; i >= 0; --i) {
               G__store_struct_offset = buf.obj.i + (i * size);
               G__getfunction(destruct, &done , G__TRYDESTRUCTOR);
#ifdef G__ASM_IFUNC
#ifdef G__ASM
               if (!done) {
                  break;
               }
               G__abortbytecode(); // Disable bytecode
               if (G__asm_noverflow) {
                  // -- We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, size, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__ADDSTROS;
                  G__asm_inst[G__asm_cp+1] = (long) size;
                  G__inc_cp_asm(2, 0);
               }
#endif // G__ASM
#endif // G__ASM_IFUNC
               // --
            }
         }
      }
#ifdef G__SECURITY
#ifdef G__ASM
      if ((G__security & G__SECURE_GARBAGECOLLECTION) && G__asm_noverflow && !done) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) G__fprinterr(G__serr, "%3x: BASEDESTRUCT\n", G__asm_cp);
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__BASEDESTRUCT;
         G__asm_inst[G__asm_cp+1] = G__tagnum;
         G__asm_inst[G__asm_cp+2] = isarray;
         G__inc_cp_asm(3, 0);
      }
#endif // G__ASM
#endif // G__SECURITY
      //
      //  Restore G__store_struct_offset.
      //
#ifdef G__ASM_IFUNC
#ifdef G__ASM
      if (G__asm_noverflow) {
         if (isarray) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: RESETARYINDEX  %s:%d\n", G__asm_cp - 2, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__RESETARYINDEX;
            G__asm_inst[G__asm_cp+1] = 0;
            G__inc_cp_asm(2, 0);
         }
         if (G__struct.iscpplink[G__tagnum] != G__CPPLINK) {
            // -- Interpreted class, free memory.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: DELETEFREE  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__DELETEFREE;
            G__asm_inst[G__asm_cp+1] = isarray ? 1 : 0;
            G__inc_cp_asm(2, 0);
         }
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp + 1, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__POPSTROS;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
#endif // G__ASM_IFUNC
      G__store_struct_offset = store_struct_offset;
      G__typenum = store_typenum;
      G__tagnum = store_tagnum;
   }
   else if (G__asm_noverflow) {
      // -- We are generating code.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp - 2, G__asm_dt, __FILE__, __LINE__);
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp - 1, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: DELETEFREE  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__DELETEFREE;
      G__asm_inst[G__asm_cp+1] = 0;
      G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
   //
   //  Free memory if interpreted object.
   //
   if ((cpplink == G__NOLINK) && !G__no_exec_compile) {
#ifdef G__ROOT
      G__delete_interpreted_object((void*)buf.obj.i);
#else // G__ROOT
      delete[] (char*) buf.obj.i;
#endif // G__ROOT
   }
   //
   //  Assign zero for deleted pointer variable.
   //
   if (buf.ref && !G__no_exec && !G__no_exec_compile) {
      *(long*)buf.ref = 0;
   }
   if (zeroflag) {
      G__no_exec_compile = 0;
      buf.obj.i = 0;
   }
}

//______________________________________________________________________________
int G__alloc_newarraylist(long point, int pinc)
{
   // FIXME: Describe this function!
   struct G__newarylist *newary;

#ifdef G__MEMTEST
   fprintf(G__memhist, "G__alloc_newarraylist(%lx,%d)\n", point, pinc);
#endif

   /****************************************************
    * Find out end of list
    ****************************************************/
   newary = &G__newarray;
   while (newary->next) newary = newary->next;


   /****************************************************
    * create next list
    ****************************************************/
   newary->next = (struct G__newarylist *)malloc(sizeof(struct G__newarylist));
   /****************************************************
    * store information
    ****************************************************/
   newary = newary->next;
   newary->point = point;
   newary->pinc = pinc;
   newary->next = (struct G__newarylist *)NULL;
   return(0);
}

//______________________________________________________________________________
int G__free_newarraylist(long point)
{
   // FIXME: Describe this function!
   struct G__newarylist *newary, *prev;
   int pinc, flag = 0;

#ifdef G__MEMTEST
   fprintf(G__memhist, "G__free_newarraylist(%lx)\n", point);
#endif

   /****************************************************
    * Search point
    ****************************************************/
   prev = &G__newarray;
   newary = G__newarray.next;
   while (newary) {
      if (newary->point == point) {
         flag = 1;
         break;
      }
      prev = newary;
      newary = newary->next;
   }

   if (flag == 0) {
      G__fprinterr(G__serr, "Error: delete[] on wrong object 0x%lx FILE:%s LINE:%d\n"
                   , point, G__ifile.name, G__ifile.line_number);
      return(0);
   }

   /******************************************************
    * get malloc size information
    ******************************************************/
   pinc = newary->pinc;

   /******************************************************
    * delete newarraylist
    ******************************************************/
   prev->next = newary->next;
   free((void*)newary);

   /* return result */
   return(pinc);
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
