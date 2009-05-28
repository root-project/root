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
#include "Dict.h"

#include "Reflex/Builder/TypeBuilder.h"

using namespace Cint::Internal;

//______________________________________________________________________________
static void G__lock_noop()
{
}

//______________________________________________________________________________
static void (*G__AllocMutexLock)() = G__lock_noop;

//______________________________________________________________________________
static void (*G__AllocMutexUnLock)() = G__lock_noop;

//______________________________________________________________________________
extern "C" void G__exec_alloc_lock()
{
   G__AllocMutexLock();
}

//______________________________________________________________________________
extern "C" void G__exec_alloc_unlock()
{
   G__AllocMutexUnLock();
}

//______________________________________________________________________________
extern "C" void G__set_alloclockfunc(void (*f)())
{
   G__AllocMutexLock = f;
}

//______________________________________________________________________________
extern "C" void G__set_allocunlockfunc(void (*f)())
{
   G__AllocMutexUnLock = f;
}

//______________________________________________________________________________
G__value Cint::Internal::G__new_operator(const char* expression)
{
   // Parsing routine to handle an operator new expression.
   //
   // Returns a value containing the allocated pointer.
   //
   // Note: The standard says we should have:
   //
   //      new-expression:
   //           ::    new new-placement    new-type-id new-initializer
   //             opt                  opt                            opt
   //
   //      new-expression:
   //           ::    new new-placement    ( type-id ) new-initializer
   //             opt                  opt                            opt
   //
   //
   //      new-type-id:
   //           type-specifier-seq new-declarator
   //                                            opt
   //
   //      new-declarator:
   //           ptr-operator new-declarator
   //                                      opt
   //           direct-new-declarator
   //
   //      direct-new-declarator:
   //           [ expression ]
   //           direct-new-declarator [ constant-expression ]
   //
   //      new-initializer:
   //           ( expression-list    )
   //                            opt
   //
   // Note: Parsing state at call is one of:
   //
   //      new int
   //          ^
   //
   //      new int[10]
   //          ^
   //
   //      new int(53)
   //          ^
   //
   //      new (arena) int
   //          ^
   //
   //--
   //
   //  Issue error and exit immediately if
   //  dynamic memory allocation is disabled
   //  under the current security restrictions.
   //
   G__CHECK(G__SECURE_MALLOC, 1, return G__null);
   //
   //  If placement new syntax is used, get the
   //  adress of the memory arena we are supposed
   //  to allocate from.  Also set type to point
   //  into the expression just after the
   //  new-placement, if any.
   //
   G__StrBuf type_sb(strlen(expression));
   char* type = type_sb;
   char* memarena = 0;
   bool arenaflag = false;
   {
      G__StrBuf arena_sb(G__ONELINE);
      char* arena = arena_sb;
      int p = 0;
      if (expression[0] != '(') {
         arena[0] = '\0';
      }
      else {
         G__getstream(expression + 1, &p, arena, ")");
         ++p;
         memarena = (char*) G__int(G__getexpr(arena));
         arenaflag = true;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETGVP 0  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETGVP;
            G__asm_inst[G__asm_cp+1] = 0;
            G__inc_cp_asm(2, 0);
         }
#endif // G__ASM
         // --
      }
      strcpy(type,expression + p);
   }
   //
   //  Get the position of any initializer expression,
   //  and the position of any array bounds given.
   //
   char* initializer = strchr(type, '(');
   char* arrayindex = strchr(type, '[');
   //
   //  An initializer for an array type is forbidden.
   //
   if (initializer && arrayindex) {
      if (initializer < arrayindex) {
         arrayindex = 0;
      }
      else {
         initializer = 0;
      }
   }
   //
   //  Get the unqualified typename.
   //
   if (initializer) {
      *initializer = 0;
   }
   const char* basictype = type; // The unqualified typename.
   {
      unsigned int len = strlen(type);
      unsigned int nest = 0;
      for (unsigned int ind = len - 1; ind > 0; --ind) {
         switch (type[ind]) {
            case '<':
               --nest;
               break;
            case '>':
               ++nest;
               break;
            case ':':
               if (!nest && (type[ind-1] == ':')) { // We have "::".
                  basictype = &type[ind+1];
                  ind = 1;
                  break;
               }
         }
      }
   }
   if (initializer) {
      *initializer = '(';
   }
   //
   //  Create an initializer expression
   //  to parse in function call format.
   //  Also parse any array bounds given.
   //
   bool ld_flag = false;  // We loaded a 1 on the data stack for the array size.
   size_t pinc = 1; // Number of elements in array.
   G__StrBuf construct_sb(G__LONGLINE);
   char* construct = construct_sb;
   {
      int tagnum = -1;
      {
         ::Reflex::Type ty = G__find_typedef(type);
         if (ty) {
            tagnum = G__get_tagnum(ty);
         }
      }
      if (!arrayindex) {
         //
         //  Generate bytecode to initialize array size to one element.
         //
#ifdef G__ASM_IFUNC
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD 1 from %x  %s:%d\n", G__asm_cp, G__asm_dt, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD;
            G__asm_inst[G__asm_cp+1] = G__asm_dt;
            G__asm_stack[G__asm_dt].obj.i = 1;
            G__value_typenum(G__asm_stack[G__asm_dt]) = G__get_from_type('i', 0);
            ld_flag = true;
         }
#endif // G__ASM_IFUNC
         if (!initializer) {
            if (tagnum == -1) {
               sprintf(construct, "%s()", basictype);
            }
            else {
               sprintf(construct, "%s()", G__struct.name[tagnum]);
            }
         }
         else {
            if (tagnum == -1) {
               *initializer = 0;
               tagnum = G__defined_tagname(basictype, 2); // FIXME: The 2 here prevents template instantiation!
               *initializer = '(';
            }
            if (tagnum == -1) {
               strcpy(construct, basictype);
            }
            else {
               sprintf(construct, "%s%s", G__struct.name[tagnum], initializer);
            }
            *initializer = '\0';
         }
      }
      else {
         pinc = G__getarrayindex(arrayindex); // Parse the array bounds.
         *arrayindex = '\0';
         if (tagnum == -1) {
            tagnum = G__defined_tagname(basictype, 1);
         }
         if (tagnum != -1) {
            sprintf(construct, "%s()", G__struct.name[tagnum]);
         }
         else {
            sprintf(construct, "%s()", basictype);
         }
         if (G__asm_wholefunction) {
            G__abortbytecode();
         }
      }
   }
   //
   //  Get the amount of memory to allocate per array element.
   //
   int size = G__Lsizeof(type);
   if (size == -1) {
      G__fprinterr(G__serr, "Error: type %s not defined FILE:%s LINE:%d\n", type, G__ifile.name, G__ifile.line_number);
      return G__null;
   }
   //
   //  Save globals.
   //
   char* store_struct_offset = G__store_struct_offset;
   ::Reflex::Scope store_tagnum = G__tagnum;
   ::Reflex::Type store_typenum = G__typenum;
   //
   //  Parse pointer level from type string and remove it.
   //  Then set var_type, G__tagnum, and G__typenum from
   //  the modified type string.
   //
   int reftype = G__PARANORMAL;
   int var_type = 0;
   {
      bool ispointer = false;
      {
         int typelen = strlen(type);
         while (type[typelen-1] == '*') {
            if (!ispointer) {
               ispointer = true;
            }
            else {
               switch (reftype) {
                  case G__PARANORMAL: // Single-level pointer.
                     reftype = G__PARAP2P;
                     break;
                  case G__PARAREFERENCE: // NOT USED
                     break;
                  default: // Multi-level pointer.
                     ++reftype;
                     break;
               }
            }
            type[--typelen] = '\0';
         }
      }
      //
      //  Set var_type, G__tagnum, and G__typenum from type string.
      //
      G__typenum = G__find_typedef(type);
      if (G__typenum) {
         G__tagnum = G__typenum.RawType();
         var_type = G__get_type(G__typenum);
      }
      else {
         int int_tagnum = G__defined_tagname(type, 1);
         G__tagnum = ::Reflex::Scope();
         if (int_tagnum != -1 ) {
            G__tagnum = G__Dict::GetDict().GetScope(int_tagnum);
         }
         if (G__tagnum) {
            var_type = 'u';
         }
         else {
            if      (!strcmp(type, "int")) var_type = 'i';
            else if (!strcmp(type, "char")) var_type = 'c';
            else if (!strcmp(type, "short")) var_type = 's';
            else if (!strcmp(type, "long")) var_type = 'l';
            else if (!strcmp(type, "float")) var_type = 'f';
            else if (!strcmp(type, "double")) var_type = 'd';
            else if (!strcmp(type, "void")) var_type = 'y';
            else if (!strcmp(type, "FILE")) var_type = 'e';
            else if (!strcmp(type, "unsignedint")) var_type = 'h';
            else if (!strcmp(type, "unsignedchar")) var_type = 'b';
            else if (!strcmp(type, "unsignedshort")) var_type = 'r';
            else if (!strcmp(type, "unsignedlong")) var_type = 'l';
            else if (!strcmp(type, "size_t")) var_type = 'l';
            else if (!strcmp(type, "time_t")) var_type = 'l';
            else if (!strcmp(type, "bool")) var_type = 'g';
            else if (!strcmp(type, "longlong")) var_type = 'n';
            else if (!strcmp(type, "unsignedlonglong")) var_type = 'm';
            else if (!strcmp(type, "longdouble")) var_type = 'q';
         }
      }
      if (ispointer) {
         var_type = toupper(var_type);
      }
   }
   //
   //  Initialize return value.
   //
   G__value result; // Our return value (will contain allocated pointer).
   result.obj.i = 0; // Default to nil pointer.
   result.ref = 0; // We return a pointer, and not a reference.
   //
   //  Initialize type of return value.
   //
   ::Reflex::Type result_type;
   if (G__tagnum) {
      result_type = G__tagnum;
   }
   else {
      result_type =  G__get_from_type(var_type, 0, 0);
   }
   if (isupper(var_type)) {
      result_type = ::Reflex::PointerBuilder(result_type);
   }
   for (int i = 1; i < reftype; ++i) {
      result_type = ::Reflex::PointerBuilder(result_type);
   }
   result_type = ::Reflex::PointerBuilder(result_type);
   //
   //  TODO: Explain this.
   //
#ifdef G__ASM
   if (G__asm_noverflow && ld_flag) {
      if ( // increment for LD 1
         G__tagnum.IsTopScope() ||
         (G__struct.iscpplink[G__get_tagnum(G__tagnum)] != G__CPPLINK) ||
         isupper(var_type)
      ) { // increment for LD 1
         G__inc_cp_asm(2, 1); // increment for LD 1
      }
      else { // cancel LD 1
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "Cancel LD 1  %s:%d\n", __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         // --
      }
   }
#endif // G__ASM
   //
   //  TODO: Explain this.
   //
#ifdef G__ASM
   if (G__asm_noverflow) {
      // --
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
   //  Allocate memory if this is not a class type, or
   //  if this is a pointer to a class type, or if this
   //  is not a pre-compiled class type.
   //
   char* pointer = 0; // This is the pointer to the new object (may be a provided arena).
   if (
      !G__tagnum || // Not of class type, or
      (G__struct.iscpplink[G__get_tagnum(G__tagnum)] != G__CPPLINK) || // is interpreted class, or
      isupper(var_type) // is pointer to pre-compiled class type.
   ) {
      if (G__no_exec_compile) { // If we are only compiling, perform a dummy action.
         pointer = (char*) pinc;
      }
      else { // Do real work.
         if (arenaflag) { // If doing placement new, use provided memory.
            pointer = memarena;
         }
         else { // Not placement new, allocate memory.
            // --
#ifdef G__ROOT
            pointer = (char*) G__new_interpreted_object(size * pinc);
#else // G__ROOT
            pointer = new char[size*pinc];
#endif // G__ROOT
            // --
         }
      }
      if (!pointer && !G__no_exec_compile) { // Error, allocate failed, exit in error.
         G__fprinterr(G__serr, "Error: memory allocation for %s %s size=%d pinc=%d FILE:%s LINE:%d\n", type, expression, size, pinc, G__ifile.name, G__ifile.line_number);
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         return G__null;
      }
      G__store_struct_offset = pointer; // Set global to point at allocated memory.
      //
      //  Generate bytecode to do same work.
      //
#ifdef G__ASM_IFUNC
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: NEWALLOC %d %d  %s:%d\n", G__asm_cp, G__asm_dt, size, pinc, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__NEWALLOC;
         if (memarena) {
            G__asm_inst[G__asm_cp+1] = 0;
         }
         else {
            G__asm_inst[G__asm_cp+1] = size; // pinc is in stack
         }
         G__asm_inst[G__asm_cp+2] = ((var_type == 'u') && arrayindex) ? 1 : 0;
         G__inc_cp_asm(3, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SET_NEWALLOC  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SET_NEWALLOC;
         ::Reflex::Type &refto_type = *reinterpret_cast<Reflex::Type*>(&G__asm_inst[G__asm_cp+1]);
         switch (var_type) {
            case 'u':
            case 'U':
               refto_type = Reflex::PointerBuilder(G__tagnum);
               break;
            default:
               refto_type = G__get_from_type(toupper(var_type), 1);
         }
         G__inc_cp_asm(3, 0);
      }
#endif // G__ASM
#endif // G__ASM_IFUNC
      // --
   }
   //
   //  Initialize allocated memory,
   //  if needed.
   //
   if (var_type == 'u') { // Class type.
      if (G__struct.isabstract[G__get_tagnum(G__tagnum)]) { // Error, abstract class.
         G__fprinterr(G__serr, "Error: abstract class object '%s' is created", G__struct.name[G__get_tagnum(G__tagnum)]);
         G__genericerror(0);
         G__display_purevirtualfunc(G__get_tagnum(G__tagnum));
      }
      if (G__dispsource) {
         G__fprinterr(G__serr, "\n!!!Calling constructor for new %s  addr: 0x%lx  %s:%d", type, G__store_struct_offset, __FILE__, __LINE__);
      }
      if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] == G__CPPLINK) { // This is a pre-compiled class.
         char* store_globalvarpointer = G__globalvarpointer;
         G__globalvarpointer = G__PVOID;
         if (memarena) {
            G__globalvarpointer = memarena;
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
         int known = 0;
         result = G__getfunction(construct, &known, G__CALLCONSTRUCTOR); // The constructor call returns a pointer to the allocated object.
         // FIXME: We need to check the error code "known" to see if we found the constructor!
         result.ref = 0; // We return a pointer, not a reference.
         G__value_typenum(result) = result_type;
         //
         //  Restore state.
         //
         G__cpp_aryconstruct = 0;
         //
         //  Generate bytecode to reset G__cpp_aryconstruct.
         //
#ifdef G__ASM
         if (G__asm_noverflow && arrayindex) {
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
         //
         //  Restore state.
         //
         G__globalvarpointer = store_globalvarpointer;
         //
         //  Generate bytecode to set G__glovalvarpointer to -1 (G__PVOID).
         //
#ifdef G__ASM
         if (G__asm_noverflow && memarena) {
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
         //
         //  Restore globals.
         //
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         //
         //  Generate bytecode to restore globals.
         //
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
         //
         //  Add an entry to the allocation table
         //  for secure pointer handling.
         //
#ifdef G__SECURITY
         if (G__security & G__SECURE_GARBAGECOLLECTION) {
            if (!G__no_exec_compile && !memarena) {
               G__add_alloctable((void*)result.obj.i, G__get_type(result), G__get_tagnum(G__value_typenum(result)));
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
            }
         }
#endif // G__SECURITY
         //
         //  And we are done.
         //
         return result;
      }
      else {
         // -- This is an interpreted class.
         if ( // Remember size of allocated array.
            arrayindex &&
            !G__no_exec_compile &&
            (G__struct.type[G__get_tagnum(G__tagnum)] != 'e')
         ) { // Remember size of allocated array.
            G__alloc_newarraylist(pointer, pinc);
         }
         G__var_type = 'p';
         for (size_t i = 0; i < pinc; ++i) { // Loop over all array elements and initialize.
            G__abortbytecode(); // Disable bytecode
            if (G__no_exec_compile) { // Do not call constructor if only compiling.
               break;
            }
            //
            //  Call the constructor to initialize one array element.
            //
            int known = 0;
            G__getfunction(construct, &known, G__TRYCONSTRUCTOR);
            if (!known) { // Constructor not found, attempt bitwise copy initialization, and terminate array init early.  FIXME: We should not terminate early!
               if (initializer) {
                  G__value buf;
                  //
                  //  Isolate the initializer expression.
                  //
                  char* bp = strchr(construct, '(');
                  char* ep = strrchr(construct, ')');
                  G__ASSERT(bp && ep);
                  *ep = 0;
                  *bp = 0;
                  ++bp;
                  {
                     G__StrBuf tmpx_sb(G__ONELINE);
                     char* tmpx = tmpx_sb;
                     int nx = 0;
                     int cx = G__getstream(bp, &nx, tmpx, "),");
                     if (cx == ',') {
                        *ep = ')';
                        *(bp - 1) = '(';
                        // only to display error message
                        int dummy_known = 0;
                        G__getfunction(construct, &dummy_known, G__CALLCONSTRUCTOR);
                        break;
                     }
                  }
                  //
                  //  Evaluate intializer expression,
                  //  which must return an object of
                  //  the class type we are returning.
                  //
                  buf = G__getexpr(bp);
                  G__abortbytecode(); // Disable bytecode
                  if (G__value_typenum(buf).RawType().IsClass() && !G__no_exec_compile) {
                     if (G__tagnum != G__value_typenum(buf).RawType()) {
                        G__fprinterr(G__serr, "Error: Illegal initialization of %s(", G__tagnum.Name(::Reflex::SCOPED).c_str());
                        G__fprinterr(G__serr, "%s)", G__value_typenum(buf).Name(::Reflex::SCOPED).c_str());
                        G__genericerror(0);
                        return G__null;
                     }
                     memcpy((void*) G__store_struct_offset, (void*) buf.obj.i, G__value_typenum(buf).RawType().SizeOf());
                  }
                  break;
               }
            }
            //
            //  Move to next array element.
            //
            G__store_struct_offset += size;
            //
            //  Generate bytecode to move to next array element.
            //
            // WARNING: FOLLOWING PART MUST BE REDESIGNED TO SUPPORT WHOLE FUNCTION COMPILATION.
#ifdef G__ASM_IFUNC
#ifdef G__ASM
            G__abortbytecode(); // Disable bytecode
            if (G__asm_noverflow && (pinc > 1)) {
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
#endif // G__ASM
#endif // G__ASM_IFUNC
            // --
         }
         //
         //  Generate bytecode to reset G__globalvarpointer to -1 (G__PVOID).
         //
#ifdef G__ASM
         if (G__asm_noverflow && memarena) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETGVP -1'  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
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
   else if (initializer) { // Initialized fundamental type.
      //
      //  Isolate the initializer string.
      //
      char* bp = strchr(construct, '(');
      char* ep = strrchr(construct, ')');
      *ep = 0;
      *bp = 0;
      ++bp;
      //
      //  Lookup the type.
      //
      ::Reflex::Type typenum = G__find_typedef(construct);
      if (typenum) {
         strcpy(construct, G__type2string(G__get_type(typenum), G__get_tagnum(typenum), -1, G__get_reftype(typenum), 0));
      }
      //
      //  Evaluate the initializer expression.
      //
      struct G__param para;
      {
         int store_var_type = G__var_type;
         G__var_type = 'p';
         para.para[0] = G__getexpr(bp); // Will generate LD or LD_VAR bytecode.
         G__var_type = store_var_type;
      }
      //
      //  TODO: Explain this.
      //
      result.ref = 0;
      if (!G__no_exec_compile) {
         result.ref = (long) pointer;
      }
      //
      //  Initialize the pointed at memory.
      //
      if ((var_type == 'U') && pointer && !G__no_exec_compile) {
         *(long*)pointer = para.para[0].obj.i;
      }
      else {
         int hash = strlen(construct);
         G__explicit_fundamental_typeconv(construct, hash, &para, &result); // Will generate CAST in bytecode.
      }
      //
      //  Generate bytecode to initialize the pointed at memory.
      //
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
      //
      //  Generate bytecode to set G__globalvarpointer to -1 (G__PVOID)
      //  if we are doing placement new.
      //
#ifdef G__ASM
      if (G__asm_noverflow && memarena) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SETGVP -1''  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SETGVP;
         G__asm_inst[G__asm_cp+1] = -1;
         G__inc_cp_asm(2, 0);
      }
#endif // G__ASM
      // --
   }
   //
   //  Set our return value to a pointer to the memory
   //  we allocated above.  Set the type of our return
   //  value to the type we pre-calculated above.
   //
   G__letint(&result, 'l', (long) pointer);
   result.ref = 0; // We return a pointer, not a reference.
   G__value_typenum(result) = result_type;
   //
   //  Restore globals.
   //
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__typenum = store_typenum;
   //
   //  Generate bytecode to restore globals.
   //
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
   //
   //  Add an entry to the allocation table
   //  for secure pointer handling.
   //
#ifdef G__SECURITY
   if (G__security & G__SECURE_GARBAGECOLLECTION) {
      if (!G__no_exec_compile && !memarena) {
         G__add_alloctable((void*) result.obj.i, G__get_type(result), G__get_tagnum(G__value_typenum(result)));
      }
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
   //
   //  And we are done.
   //
   return result;
}

//______________________________________________________________________________
int Cint::Internal::G__getarrayindex(const char* indexlist)
{
   // [x][y][z]     get x*y*z
   G__StrBuf index_sb(G__ONELINE);
   char *index = index_sb;
   int store_var_type = G__var_type;
   G__var_type = 'p';
   int p = 1;
   int c = G__getstream(indexlist, &p, index, "]");
   int p_inc = 1;
   p_inc *= G__int(G__getexpr(index));
   while (*(indexlist + p) == '[') {
      ++p;
      c = G__getstream(indexlist, &p, index, "]");
      p_inc *= G__int(G__getexpr(index));
#ifdef G__ASM_IFUNC
#ifdef G__ASM
      if (G__asm_noverflow) {
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: OP2 '*'  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__OP2;
         G__asm_inst[G__asm_cp+1] = (long) '*';
         G__inc_cp_asm(2, 0);
      }
#endif // G__ASM
#endif // G__ASM_IFUNC
      // --
   }
   G__ASSERT(c == ']');
   G__var_type = store_var_type;
   return p_inc;
}

//______________________________________________________________________________
void Cint::Internal::G__delete_operator(const char* expression, int isarray)
{
   // Parsing routine to handle a delete operator expression.
   char *store_struct_offset; /* used to be int */
   ::Reflex::Scope store_tagnum;
   ::Reflex::Type store_typenum;
   int done;
   G__StrBuf destruct_sb(G__ONELINE);
   char *destruct = destruct_sb;
   G__value buf;
   int pinc, i, size;
   int cpplink = 0;
   int zeroflag = 0;

   buf = G__getitem(expression);
   if (!G__value_typenum(buf).FinalType().IsPointer()) {
      G__fprinterr(G__serr, "Error: %s cannot delete", expression);
      G__genericerror((char*)NULL);
      return;
   }
   else if (0 == buf.obj.i && 0 == G__no_exec_compile && G__ASM_FUNC_NOP == G__asm_wholefunction) {
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
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) G__fprinterr(G__serr, "%3x: DELALLOCTABLE\n", G__asm_cp);
#endif
         G__asm_inst[G__asm_cp] = G__DELALLOCTABLE;
         G__inc_cp_asm(1, 0);
      }
#endif
      // --
   }
#endif

   /*********************************************************
    * Call destructor if struct or class
    *********************************************************/
   if (
      (G__get_type(G__value_typenum(buf)) == 'U') && // pointer to class, struct, union
      (G__get_reftype(G__value_typenum(buf)) ==  G__PARANORMAL) // and, not a reference or multi-level pointer
   ) {
      store_struct_offset = G__store_struct_offset;
      store_typenum = G__typenum;
      store_tagnum = G__tagnum;

      G__store_struct_offset = (char*)buf.obj.i;
      G__typenum = G__value_typenum(buf);
      G__set_G__tagnum(buf);

      sprintf(destruct, "~%s()", G__struct.name[G__get_tagnum(G__tagnum)]);
      if (G__dispsource) {
         G__fprinterr(G__serr, "\n!!!Calling destructor 0x%lx.%s for %s"
                      , G__store_struct_offset , destruct , expression);
      }
      done = 0;

      if (0 == G__no_exec_compile && G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)] &&
            G__tagnum !=
            G__Dict::GetDict().GetScope(*(long*)(G__store_struct_offset + (size_t)G__struct.virtual_offset[G__get_tagnum(G__tagnum)]))) {
         int virtualtag =
            *(long*)(G__store_struct_offset + (size_t)G__struct.virtual_offset[G__get_tagnum(G__tagnum)]);
         buf.obj.i -= G__find_virtualoffset(virtualtag);
      }

      /*****************************************************
       * Push and set G__store_struct_offset
       *****************************************************/
#ifdef G__ASM_IFUNC
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
         if (isarray) {
            G__asm_inst[G__asm_cp] = G__GETARYINDEX;
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: GETARYINDEX\n", G__asm_cp - 2);
#endif
            G__inc_cp_asm(1, 0);
         }
      }
#endif /* G__ASM */
#endif /* G__ASM_IFUNC */

      /*****************************************************
       * Call destructor
       *****************************************************/
      if (G__CPPLINK == G__struct.iscpplink[G__get_tagnum(G__tagnum)]) {
         /* pre-compiled class */
         if (isarray) G__cpp_aryconstruct = 1;

#ifndef G__ASM_IFUNC
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
#endif /* G__ASM */
#endif /* !G__ASM_IFUNC */

         G__getfunction(destruct, &done, G__TRYDESTRUCTOR);
         /* Precompiled destructor must always exist here */

#ifndef G__ASM_IFUNC
#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp);
#endif
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif /* G__ASM */
#endif /* !G__ASM_IFUNC */

         G__cpp_aryconstruct = 0;
         cpplink = 1;
      }
      else {
         /* interpreted class */
         /* WARNING: FOLLOWING PART MUST BE REDESIGNED TO SUPPORT WHOLE
          * FUNCTION COMPILATION */
         if (isarray) {
            if (!G__no_exec_compile)
               pinc = G__free_newarraylist(G__store_struct_offset);
            else pinc = 1;
            size = G__struct.size[G__get_tagnum(G__tagnum)];
            for (i = pinc - 1;i >= 0;--i) {
               G__store_struct_offset = (char*)(buf.obj.i + size * i);
               G__getfunction(destruct, &done , G__TRYDESTRUCTOR);
#ifdef G__ASM_IFUNC
#ifdef G__ASM
               if (0 == done) break;
               G__abortbytecode(); /* Disable bytecode */
               if (G__asm_noverflow) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) G__fprinterr(G__serr, "%3x: ADDSTROS %d\n", G__asm_cp, size);
#endif
                  G__asm_inst[G__asm_cp] = G__ADDSTROS;
                  G__asm_inst[G__asm_cp+1] = (long)size;
                  G__inc_cp_asm(2, 0);
               }
#endif /* G__ASM */
#endif /* !G__ASM_IFUNC */
            }
         }
         else {
            G__getfunction(destruct, &done, G__TRYDESTRUCTOR);
         }
      }

#ifdef G__ASM
#ifdef G__SECURITY
      if (G__security&G__SECURE_GARBAGECOLLECTION && G__asm_noverflow && 0 == done) {
#ifdef G__ASM_DBG
         if (G__asm_dbg) G__fprinterr(G__serr, "%3x: BASEDESTRUCT\n", G__asm_cp);
#endif
         G__asm_inst[G__asm_cp] = G__BASEDESTRUCT;
         G__asm_inst[G__asm_cp+1] = G__get_tagnum(G__tagnum);
         G__asm_inst[G__asm_cp+2] = isarray;
         G__inc_cp_asm(3, 0);
      }
#endif
#endif

      /*****************************************************
       * Push and set G__store_struct_offset
       *****************************************************/
#ifdef G__ASM_IFUNC
#ifdef G__ASM
      if (G__asm_noverflow) {
         if (isarray) {
            G__asm_inst[G__asm_cp] = G__RESETARYINDEX;
            G__asm_inst[G__asm_cp+1] = 0;
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: RESETARYINDEX\n", G__asm_cp - 2);
#endif
            G__inc_cp_asm(2, 0);
         }
         if (G__CPPLINK != G__struct.iscpplink[G__get_tagnum(G__tagnum)]) {
            /* if interpreted class, free memory */
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: DELETEFREE\n", G__asm_cp);
#endif
            G__asm_inst[G__asm_cp] = G__DELETEFREE;
            G__asm_inst[G__asm_cp+1] = isarray ? 1 : 0;
            G__inc_cp_asm(2, 0);
         }
#ifdef G__ASM_DBG
         if (G__asm_dbg) G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp + 1);
#endif
         G__asm_inst[G__asm_cp] = G__POPSTROS;
         G__inc_cp_asm(1, 0);
      }
#endif /* G__ASM */
#endif /* G__ASM_IFUNC */

      /*****************************************************
       * Push and set G__store_struct_offset
       *****************************************************/
      G__store_struct_offset = store_struct_offset;
      G__typenum = store_typenum;
      G__tagnum = store_tagnum;

   }
   else if (G__asm_noverflow) {
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x: PUSHSTROS\n", G__asm_cp - 2);
         G__fprinterr(G__serr, "%3x: SETSTROS\n", G__asm_cp - 1);
      }
#endif
#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x: DELETEFREE\n", G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__DELETEFREE;
      G__asm_inst[G__asm_cp+1] = 0;
      G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }

   /*****************************************************
    * free memory if interpreted object
    *****************************************************/
   if (G__NOLINK == cpplink && !G__no_exec_compile) {
#ifdef G__ROOT
      G__delete_interpreted_object((void*)buf.obj.i);
#else
      delete[](char*) buf.obj.i;
#endif
   }

   /* #ifdef G__ROOT */
   /*****************************************************
    * assign NULL for deleted pointer variable
    *****************************************************/
   if (buf.ref && 0 == G__no_exec && 0 == G__no_exec_compile) *(long*)buf.ref = 0;
   /* #endif G__ROOT */

   if (zeroflag) {
      G__no_exec_compile = 0;
      buf.obj.i = 0;
   }
}

//______________________________________________________________________________
int Cint::Internal::G__alloc_newarraylist(void* point, int pinc)
{
   // Allocate and initialize a new array bounds list entry.
#ifdef G__MEMTEST
   fprintf(G__memhist, "G__alloc_newarraylist(%lx,%d)\n", point, pinc);
#endif
   //
   //  Move to the end of the list.
   //
   struct G__newarylist* newary = &G__newarray;
   while (newary->next) {
      newary = newary->next;
   }
   newary->next = (struct G__newarylist*) malloc(sizeof(struct G__newarylist));
   newary = newary->next;
   newary->point = (long) point;
   newary->pinc = pinc;
   newary->next = 0;
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__free_newarraylist(void* point)
{
   bool found = false;
#ifdef G__MEMTEST
   fprintf(G__memhist, "G__free_newarraylist(%lx)\n", point);
#endif // G__MEMTEST
   //
   //  Search point.
   //
   struct G__newarylist* prev = &G__newarray;
   struct G__newarylist* newary = G__newarray.next;
   while (newary) {
      if ((void*) newary->point == point) {
         found = true;
         break;
      }
      prev = newary;
      newary = newary->next;
   }
   if (!found) {
      G__fprinterr(G__serr, "Error: delete[] on wrong object 0x%lx FILE:%s LINE:%d\n", point, G__ifile.name, G__ifile.line_number);
      return 0;
   }
   //
   //  Get malloc size information.
   //
   int pinc = newary->pinc;
   //
   //  Delete newarraylist.
   //
   prev->next = newary->next;
   free((void*) newary);
   return pinc;
}

//______________________________________________________________________________
int Cint::Internal::G__handle_delete(int* piout, char* statement)
{
   // Parsing of "delete obj" and "delete obj[]".
   int c = G__fgetstream(statement, "[;");
   *piout = 0;
   if (c == '[') {
      if (!statement[0]) {
         c = G__fgetstream(statement , "]");
         c = G__fgetstream(statement , ";");
         *piout = 1;
      }
      else {
         strcat(statement, "[");
         c = G__fgetstream(statement + strlen(statement), "]");
         strcat(statement, "]");
         c = G__fgetstream(statement + strlen(statement), ";");
      }
   }
   return 0;
}

