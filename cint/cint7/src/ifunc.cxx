/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file ifunc.c
 ************************************************************************
 * Description:
 *  interpret function and new style compiled function
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Api.h"
#include "Dict.h"

#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Builder/NamespaceBuilder.h"
#include "Reflex/Builder/FunctionBuilder.h"
#include "bc_exec.h"
#include <vector>
#include <sstream>

using namespace Cint::Internal;

namespace Cint {
namespace Internal {
static int G__readansiproto(std::vector<Reflex::Type>& i_params_type, std::vector<std::pair<std::string, std::string> >& i_params_names, std::vector<G__value*>& i_params_default, char* i_ansi);
} // Internal
} // Cint

static int G__calldepth = 0;

//______________________________________________________________________________
char *Cint::Internal::G__savestring(char **pbuf, char *name)
{
   G__ASSERT(pbuf);
   if (*pbuf) free((void*)(*pbuf));
   *pbuf = (char*)malloc(strlen(name) + 1);
   return(strcpy(*pbuf, name));
}

#ifndef G__OLDIMPLEMENTATION1167
//______________________________________________________________________________
static void G__reftypeparam(const ::Reflex::Member &ifunc, G__param *libp)
{
   int itemp = 0;
   for (::Reflex::Type_Iterator para(ifunc.TypeOf().FunctionParameter_Begin());
         itemp < libp->paran && para != ifunc.TypeOf().FunctionParameter_End();
         ++para, ++itemp) {
      if (para->FinalType().IsReference() &&
            G__get_type(para->FinalType()) != G__get_type(libp->para[itemp])) {
         switch (G__get_type(*para)) {
            case 'c':
               libp->para[itemp].ref = (long)G__Charref(&libp->para[itemp]);
               break;
            case 's':
               libp->para[itemp].ref = (long)G__Shortref(&libp->para[itemp]);
               break;
            case 'i':
               libp->para[itemp].ref = (long)G__Intref(&libp->para[itemp]);
               break;
            case 'l':
               libp->para[itemp].ref = (long)G__Longref(&libp->para[itemp]);
               break;
            case 'b':
               libp->para[itemp].ref = (long)G__UCharref(&libp->para[itemp]);
               break;
            case 'r':
               libp->para[itemp].ref = (long)G__UShortref(&libp->para[itemp]);
               break;
            case 'h':
               libp->para[itemp].ref = (long)G__UIntref(&libp->para[itemp]);
               break;
            case 'k':
               libp->para[itemp].ref = (long)G__ULongref(&libp->para[itemp]);
               break;
            case 'f':
               libp->para[itemp].ref = (long)G__Floatref(&libp->para[itemp]);
               break;
            case 'd':
               libp->para[itemp].ref = (long)G__Doubleref(&libp->para[itemp]);
               break;
            case 'g':
               libp->para[itemp].ref = (long) G__Boolref(&libp->para[itemp]);
               break;
            case 'n':
               libp->para[itemp].ref = (long)G__Longlongref(&libp->para[itemp]);
               break;
            case 'm':
               libp->para[itemp].ref = (long)G__ULonglongref(&libp->para[itemp]);
               break;
            case 'q':
               libp->para[itemp].ref = (long)G__Longdoubleref(&libp->para[itemp]);
               break;
         }
      }
   }
}
#endif

//______________________________________________________________________________
static void G__warn_refpromotion(const ::Reflex::Member &p_ifunc, int itemp, G__param *libp)
{
   if (p_ifunc.TypeOf().FunctionParameterAt(itemp).FinalType().IsReference() &&
         !p_ifunc.TypeOf().FunctionParameterAt(itemp).IsClass() &&
         G__get_type(p_ifunc.TypeOf().FunctionParameterAt(itemp)) != G__get_type(G__value_typenum(libp->para[itemp])) &&
         0 != libp->para[itemp].obj.i &&
         G__test_const(p_ifunc.TypeOf().FunctionParameterAt(itemp), G__VARIABLE)) {
#ifdef G__OLDIMPLEMENTATION1167
      if (G__dispmsg >= G__DISPWARN) {
         G__fprinterr(G__serr, "Warning: implicit type conversion of non-const reference arg %d", itemp);
         G__printlinenum();
      }
#endif
   }
}

#ifdef G__ASM_WHOLEFUNC
//______________________________________________________________________________
void Cint::Internal::G__free_bytecode(G__bytecodefunc *bytecode)
{
   if (bytecode) {
      if (bytecode->asm_name) free((void*)bytecode->asm_name);
      if (bytecode->pstack) free((void*)bytecode->pstack);
      if (bytecode->pinst) free((void*)bytecode->pinst);
      if (bytecode->frame) {
         G__destroy_upto(bytecode->frame, G__BYTECODELOCAL_VAR, -1);
      }
      delete bytecode;
   }
}

//______________________________________________________________________________
void Cint::Internal::G__asm_storebytecodefunc(const Reflex::Member& func, const Reflex::Scope& frame, G__value *pstack, int sp, long *pinst, int instsize)
{
   struct G__bytecodefunc *bytecode;

   /* check if the function is already compiled, replace old one */
   if (G__get_funcproperties(func)->entry.bytecode) {
      G__genericerror("Internal error: G__asm_storebytecodefunc duplicated");
   }

   /* allocate bytecode buffer */
   bytecode = new G__bytecodefunc;
   G__get_funcproperties(func)->entry.bytecode = bytecode;

   /* store function ID */
   bytecode->ifunc = func;

   /* copy local variable table */
   bytecode->frame = frame;
   bytecode->varsize = G__struct.size[G__get_tagnum(G__tagdefining)];

   /* copy instruction */
   bytecode->pinst = (long*)malloc(sizeof(long) * instsize + 8);
   memcpy(bytecode->pinst, pinst, sizeof(long)*instsize + 1);
   bytecode->instsize = instsize;

   /* copy constant data stack */
   bytecode->stacksize = G__MAXSTACK - sp;
   bytecode->pstack = (G__value*)malloc(sizeof(G__value) * bytecode->stacksize);
   memcpy((void*)bytecode->pstack, (void*)(&pstack[sp])
          , sizeof(G__value)*bytecode->stacksize);

   /* copy compiled and library function name buffer */
   if (0 == G__asm_name_p) {
      if (G__asm_name) free(G__asm_name);
      bytecode->asm_name = 0;
   }
   else {
      bytecode->asm_name = G__asm_name;
   }

#ifdef G__OLDIMPLEMENtATION1578 /* Problem  t1048.cxx */
   /* store pointer to function */
   ifunc->pentry[ifn]->tp2f = (void*)bytecode;
#endif
   // --
}

//______________________________________________________________________________
static int G__noclassargument(const ::Reflex::Member &ifunc)
{
   // stops bytecode compilation if class object is passed as argument
   for (unsigned int i = 0; i < ifunc.TypeOf().FunctionParameterSize(); ++i) {
      if ((ifunc.TypeOf().FunctionParameterAt(i).IsClass()
            || ifunc.TypeOf().FunctionParameterAt(i).IsUnion()) &&
            ! ifunc.TypeOf().FunctionParameterAt(i).FinalType().IsReference())
         /* return false if class/struct object and non-reference type arg */
         return(0);
   }
   return(1);
}

//______________________________________________________________________________
extern "C" int G__compile_bytecode(struct G__ifunc_table* ifunc, int index)
{
   return G__compile_function_bytecode(G__Dict::GetDict().GetFunction(ifunc, index));
}

//______________________________________________________________________________
int Cint::G__compile_function_bytecode(const ::Reflex::Member &ifunc)
{
   G__value buf;
   struct G__param para; /* This one is only dummy */
   struct G__input_file store_ifile;
   int store_prerun = G__prerun;
   ::Reflex::Member store_asm_index = G__asm_index;
   int store_no_exec = G__no_exec;
   int store_asm_exec = G__asm_exec;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   int store_asm_noverflow = G__asm_noverflow;
   int funcstatus;
   char *store_globalvarpointer = G__globalvarpointer;
   std::string funcname;
   int store_dispsource = G__dispsource;
   if (G__step || G__stepover) G__dispsource = 0;

   if (
      G__xrefflag ||
      (
         G__get_funcproperties(ifunc)->entry.size < G__ASM_BYTECODE_FUNC_LIMIT
         && 0 == G__def_struct_member
         && ('u' != G__get_type(ifunc.TypeOf().ReturnType()) || ifunc.TypeOf().ReturnType().IsReference()
             && (0 == ifunc.TypeOf().FunctionParameterSize() ||
                 (G__get_funcproperties(ifunc)->entry.ansi && G__noclassargument(ifunc))))
      )
   ) {

      para.paran = 0;
      para.para[0] = G__null;
#ifndef __GNUC__
#pragma message(FIXME("we need a G__struct entry for bytecode!"))
#endif // __GNUC__

      // Create a scratch area for the byte code stack.
      static int bytecodeTagnum = G__search_tagname("% CINT byte code scratch arena %",'c');
      static ::Reflex::Scope bytecodeArena = G__Dict::GetDict().GetScope(bytecodeTagnum);

       //G__tagdefining = G__MAXSTRUCT-1;
      //G__struct.type[G__get_tagnum(G__tagdefining)] = 's';
      // Reset it somewhat.
      G__struct.size[bytecodeTagnum] = 0;
      G__get_properties(bytecodeArena)->builder.Class().SetSizeOf(0);
      G__get_properties(bytecodeArena)->isBytecodeArena = true;

      G__tagdefining = bytecodeArena;

      G__no_exec = 0;
      G__prerun = 0;
      G__asm_exec = 1;
      G__asm_wholefunction = G__ASM_FUNC_COMPILE;
      G__asm_noverflow = 0;
      store_ifile = G__ifile;
      G__asm_index = ifunc; // 0; // iexist;
      ++G__templevel;
      ++G__calldepth;
      funcname = ifunc.Name();
      if (ifunc.DeclaringScope().IsTopScope()) {
         funcstatus = G__TRYNORMAL;
      }
      else {
         funcstatus = G__CALLMEMFUNC;
      }
      G__init_jumptable_bytecode();
      G__interpret_func(&buf, &para
                        , 0 /* ifunc->hash[iexist] */ , ifunc
                        , G__EXACT, funcstatus);
      G__init_jumptable_bytecode();
      --G__calldepth;
      --G__templevel;
      G__tagdefining = store_tagdefining;
      G__asm_exec = store_asm_exec;
      G__no_exec = store_no_exec;
      G__prerun = store_prerun;
      G__asm_index = store_asm_index;
      G__asm_wholefunction = G__ASM_FUNC_NOP;
      G__ifile = store_ifile;
      G__asm_noverflow = store_asm_noverflow;
      G__globalvarpointer = store_globalvarpointer;
   }
   else if (G__asm_dbg) {
      G__fprinterr(G__serr, "!!!bytecode compilation %s not tried either because\n"
                   , ifunc.Name().c_str());
      G__fprinterr(G__serr, "    function is longer than %d lines\n"
                   , G__ASM_BYTECODE_FUNC_LIMIT);
      G__fprinterr(G__serr, "    function returns class object or reference type\n");
      G__fprinterr(G__serr, "    function is K&R style\n");
      G__printlinenum();
   }

   if (G__get_funcproperties(ifunc)->entry.bytecode) {
      if (0 == G__xrefflag)
         G__get_funcproperties(ifunc)->entry.bytecodestatus = G__BYTECODE_SUCCESS;
      else
         G__get_funcproperties(ifunc)->entry.bytecodestatus = G__BYTECODE_ANALYSIS;
   }
   else if (0 == G__def_struct_member)
      G__get_funcproperties(ifunc)->entry.bytecodestatus = G__BYTECODE_FAILURE;

   G__dispsource = store_dispsource;
   return(G__get_funcproperties(ifunc)->entry.bytecodestatus);
}

//______________________________________________________________________________
#define G__MAXGOTOLABEL 30

struct G__gotolabel
{
   int pc;
   char *label;
};

static int G__ngoto;
static int G__nlabel;
static struct G__gotolabel G__goto_table[G__MAXGOTOLABEL];
static struct G__gotolabel G__labeltable[G__MAXGOTOLABEL];

//______________________________________________________________________________
static void G__free_gotolabel(struct G__gotolabel *pgotolabel, int *pn)
{
   while (*pn > 0)
   {
      --(*pn);
      free((char*)pgotolabel[*pn].label);
   }
}

//______________________________________________________________________________
void Cint::Internal::G__init_jumptable_bytecode()
{
   G__free_gotolabel(G__labeltable, &G__nlabel);
   G__free_gotolabel(G__goto_table, &G__ngoto);
}

//______________________________________________________________________________
void Cint::Internal::G__add_label_bytecode(char *label)
{
   if (G__nlabel < G__MAXGOTOLABEL) {
      int len = strlen(label);
      if (len) {
         G__labeltable[G__nlabel].pc = G__asm_cp;
         label[len-1] = 0;
         G__labeltable[G__nlabel].label = (char*)malloc(strlen(label) + 1);
         strcpy(G__labeltable[G__nlabel].label, label);
         ++G__nlabel;
      }
   }
   else {
      G__abortbytecode();
   }
}

//______________________________________________________________________________
void Cint::Internal::G__add_jump_bytecode(char *label)
{
   if (G__ngoto < G__MAXGOTOLABEL) {
      int len = strlen(label);
      if (len) {
         G__goto_table[G__ngoto].pc = G__asm_cp + 1;
         G__asm_inst[G__asm_cp] = G__JMP;
         G__inc_cp_asm(2, 0);
         G__goto_table[G__ngoto].label = (char*)malloc(strlen(label) + 1);
         strcpy(G__goto_table[G__ngoto].label, label);
         ++G__ngoto;
      }
   }
   else {
      G__abortbytecode();
   }
}

//______________________________________________________________________________
void Cint::Internal::G__resolve_jumptable_bytecode()
{
   if (G__asm_noverflow) {
      int i, j;
      for (j = 0;j < G__nlabel;j++) {
         for (i = 0;i < G__ngoto;i++) {
            if (strcmp(G__goto_table[i].label, G__labeltable[j].label) == 0) {
               G__asm_inst[G__goto_table[i].pc] = G__labeltable[j].pc;
            }
         }
      }
   }
   G__init_jumptable_bytecode();
}

#endif /* G__ASM_WHOLEFUNC */

//______________________________________________________________________________
int Cint::Internal::G__istypename(char *temp)
{
   // true if fundamental type, class, struct, typedef, template class name
   if (isdigit(temp[0])) return 0;
   if (strncmp(temp, "class ", 6) == 0) temp += 6;
   else if (strncmp(temp, "struct ", 7) == 0) temp += 7;
   else if (strncmp(temp, "enum ", 5) == 0) temp += 5;
   if (strchr(temp, '(') || strchr(temp, ')') || strchr(temp, '|')) return(0);
   /* char *p; */
   /* char buf[G__MAXNAME*2]; */
   if ('\0' == temp[0]) return(0);
   if (strcmp(temp, "int") == 0 ||
         strcmp(temp, "short") == 0 ||
         strcmp(temp, "char") == 0 ||
         strcmp(temp, "long") == 0 ||
         strcmp(temp, "float") == 0 ||
         strcmp(temp, "double") == 0 ||
         (strncmp(temp, "unsigned", 8) == 0 &&
          (strcmp(temp, "unsigned") == 0 ||
           strcmp(temp, "unsignedchar") == 0 ||
           strcmp(temp, "unsignedshort") == 0 ||
           strcmp(temp, "unsignedint") == 0 ||
           strcmp(temp, "unsignedlong") == 0)) ||
         strcmp(temp, "signed") == 0 ||
         strcmp(temp, "const") == 0 ||
         strcmp(temp, "void") == 0 ||
         strcmp(temp, "FILE") == 0 ||
         strcmp(temp, "class") == 0 ||
         strcmp(temp, "struct") == 0 ||
         strcmp(temp, "union") == 0 ||
         strcmp(temp, "enum") == 0 ||
         strcmp(temp, "register") == 0 ||
         strcmp(temp, "bool") == 0 ||
         (G__iscpp && strcmp(temp, "typename") == 0) ||
         G__find_typedef(temp) ||
         -1 != G__defined_tagname(temp, 2) ||
         G__defined_templateclass(temp)) {
      return(1);
   }

   if (G__fpundeftype) return(1);

   return(0);
}

//______________________________________________________________________________
void Cint::Internal::G__make_ifunctable(char* funcheader)
{
   // -- FIXME: Describe this function!
   //
   // Note: G__no_exec is always zero on entry.
   //
#ifndef G__NEWINHERIT
   int func_now = -1; // FIXME: Used but not initialized!
#endif
   int dobody = 0;
   G__ASSERT(G__prerun);
   ::Reflex::Scope store_ifunc = G__p_ifunc;
   if (G__def_struct_member && G__def_tagnum) {
      // no need for incremental setup
      G__p_ifunc = G__def_tagnum;
   }
   // Store ifunc to check if same function already exists
   ::Reflex::Scope ifunc = G__p_ifunc;
   int numstar = 0;
   for (; funcheader[numstar] == '*'; ++numstar);
   std::string funcname = funcheader + numstar;
   if (!numstar) {
      if (!strncmp(funcheader, "operator ", 9)) {
         // -- We may have a type conversion operator, try to expand the type name.
         char* oprtype = funcheader + 9;
         if (
            strcmp(oprtype, "char") &&
            strcmp(oprtype, "short") &&
            strcmp(oprtype, "int") &&
            strcmp(oprtype, "long") &&
            strcmp(oprtype, "unsigned char") &&
            strcmp(oprtype, "unsigned short") &&
            strcmp(oprtype, "unsigned int") &&
            strcmp(oprtype, "unsigned long") &&
            strcmp(oprtype, "float") &&
            strcmp(oprtype, "double")
         ) {
            // -- Not a type conversion to one of the supported fundamental types, check for typedef and class name.
            // Modify funcheader, drop the final '(', we will put it back later.
            oprtype[strlen(oprtype)-1] = 0;
            // Is the given typename a typedef?
            ::Reflex::Type oprtypenum = G__find_typedef(oprtype);
            if (
               oprtypenum && // we found a typedef by the given name, and
               (G__get_tagnum(oprtypenum) == -1) && // it really is a typdef, and // FIXME: Waste of time?
               (oprtypenum.DeclaringScope() != ::Reflex::Scope::GlobalScope()) // not a global typdef // FIXME: Why?
            ) {
               // Rewrite the given typedef name, replace it by the underlying type.
               const char* posEndType = oprtype;
               while (isalnum(*posEndType)
                      || posEndType[0] == ':' && posEndType[1] == ':'
                      || *posEndType == '_')
                  ++posEndType;
               char* ptrrefbuf = 0;
               if (*posEndType) {
                  ptrrefbuf = new char[strlen(posEndType) + 1];
                  strcpy(ptrrefbuf, posEndType);
               }
               strcpy(oprtype, G__type2string(G__get_type(oprtypenum), -1, -1, G__get_reftype(oprtypenum), G__get_isconst(oprtypenum)));
               if (ptrrefbuf) {
                  strcat(oprtype, ptrrefbuf);
                  delete [] ptrrefbuf;
               }
            }
            else {
               // -- Not a typedef, check for a class with the given name.
               int oprtagnum = G__defined_tagname(oprtype, 2);
               if (oprtagnum > -1) {
                  // Rewrite the given class name, replace it by the fully-qualified name.
                  const char* posEndType = oprtype;
                  bool isTemplate = (strchr(G__struct.name[oprtagnum], '<'));
                  while (isalnum(*posEndType)
                         || posEndType[0] == ':' && posEndType[1] == ':' && (++posEndType)
                         || *posEndType == '_') {
                     ++posEndType;
                     if (isTemplate && *posEndType == '<') {
                        // also include "<T>" in type name
                        ++posEndType;
                        int templLevel = 1;
                        while (templLevel && *posEndType) {
                           if (*posEndType == '<') ++templLevel;
                           else if (*posEndType == '>') --templLevel;
                           ++posEndType;
                        }
                     }
                  }
                  char* ptrrefbuf = 0;
                  if (*posEndType) {
                     ptrrefbuf = new char[strlen(posEndType) + 1];
                     strcpy(ptrrefbuf, posEndType);
                  }
                  strcpy(oprtype, G__fulltagname(oprtagnum, 0));
                  if (ptrrefbuf) {
                     strcat(oprtype, ptrrefbuf);
                     delete [] ptrrefbuf;
                  }
               }
            }
            // And put the final '(' back that we removed above.
            strcat(oprtype, "(");
         }
      }
      funcname = funcheader;
      if (
         ((funcname.find(">>") != std::string::npos) && (funcname.find('<') != std::string::npos)) ||
         ((funcname.find("<<") != std::string::npos) && (funcname.find('>') != std::string::npos))
      ) {
         if (
            (funcname.find("operator<<") != std::string::npos) &&
            (funcname.find('>') != std::string::npos)
         ) {
            // we might have "operator< <>" or "operator< <double>" or "operator<< <>" or "operator<< <double>"
            // with the space missing
            std::string::size_type pos = funcname.find("<<");
            if (pos != std::string::npos) {
               if (funcname[pos+2]=='<') ++pos; // case "operator<< <double>"
               funcname.insert(pos + 1, 1, ' ');
            }
         }
         // Find the first ">>" after the initial "operator>>" or "operator<<".
         std::string::size_type pos = funcname.find(">>", 10);
         for (; pos != std::string::npos;) {
            // Turn the ">>" into "> >".  // FIXME: Need to fix the parser so it does not make this happen!
            funcname.insert(pos + 1, 1, ' ');
            pos = funcname.find(">>", pos + 2);
         }
      }
   }
   // Remove the final '(' character.
   if (funcname.size()) {
      funcname.erase(funcname.size() - 1);
   }
   // Rewrite the function name, fully qualify any template parameters.
   // Note: myfunc<B>(x) -> myfunc<ns::B>(x)
   G__rename_templatefunc(funcname);
   //
   //  Done with final parsing of the return type and the function name,
   //  now start building the function type.
   //
   G__BuilderInfo builder;
   G__funcentry builder_entry;
   fgetpos(G__ifile.fp, &builder.fProp.entry.pos);
   builder.fProp.entry.p = (void*) G__ifile.fp;
   builder.fProp.filenum = G__ifile.filenum;
   builder.fProp.linenum = G__ifile.line_number;
   builder.fProp.entry.line_number = G__ifile.line_number;
   builder.fProp.entry.filenum = G__ifile.filenum;
#ifdef G__ASM_FUNC
   builder.fProp.entry.size = 0;
#endif // G__ASM_FUNC
#ifdef G__ASM_WHOLEFUNC
   builder.fProp.entry.bytecode = 0;
   builder.fProp.entry.bytecodestatus = G__BYTECODE_NOTYET;
#endif // G__ASM_WHOLEFUNC
   if (G__get_tagnum(G__p_ifunc.DeclaringScope()) == -1) {
      builder.fProp.globalcomp = G__default_link ? G__globalcomp : G__NOLINK;
   }
   else {
      builder.fProp.globalcomp = G__globalcomp;
   }
#ifdef G__FRIEND
   if (!G__friendtagnum) {
      builder.fProp.entry.friendtag = 0;
   }
   else {
      builder.fProp.entry.friendtag = G__new_friendtag(G__get_tagnum(G__friendtagnum));
   }
#endif // G__FRIEND
   //
   //  Determine the return type of the function.
   //
   if (
      G__def_struct_member && // member of a class, enum, namespace, struct or union
      G__def_tagnum && // outer class is known
      (funcname == G__struct.name[G__get_tagnum(G__def_tagnum)]) // member name is same as class name
   ) {
      // -- This is a constructor, handle specially.
      // irregular handling not to instantiate temp object for return
#ifndef __GNUC__
#pragma message(FIXME("We no longer have the irregular constructor function setting preventing temp obj creation"))
#endif // __GNUC__
      builder.fReturnType = G__modify_type(G__def_tagnum, 0, G__reftype, G__constvar, 0, 0);
      G__struct.isctor[G__get_tagnum(G__def_tagnum)] = 1;
   }
   else if (funcname.size() && funcname.at(0) == '~') {
      // -- This is a destructor, return type is void.
      builder.fReturnType = ::Reflex::Type::ByName("void");
   }
   else if (G__typenum) {
      // -- The return type is a typedef.
      builder.fReturnType = G__typenum;
      if (G__constvar & G__PCONSTVAR) {
         // const pointer to something
         assert(builder.fReturnType.FinalType().IsPointer());
         builder.fReturnType = ::Reflex::Type(builder.fReturnType, ::Reflex::CONST, Reflex::Type::APPEND);
      } else if (G__constvar & G__CONSTVAR) {
         if (builder.fReturnType.IsPointer()) {
            // Pointer to const something (const char*).
            // FIXME we do not conserve the modifiers of the pointer to.
            Reflex::Type tmp = builder.fReturnType.ToType();
            tmp = ::Reflex::Type(tmp, ::Reflex::CONST, Reflex::Type::APPEND);
            builder.fReturnType = Reflex::PointerBuilder( tmp );
         } else {
            builder.fReturnType = ::Reflex::Type(builder.fReturnType, ::Reflex::CONST, Reflex::Type::APPEND);
         }
      }
      int ref = G__REF(G__reftype);
      int plvl = G__PLVL(G__reftype);
      if (plvl == G__PARAREFERENCE) {
         plvl = 0;
         ref = 1;
      }
      for (int i = 0; i < plvl; ++i) {
         builder.fReturnType = ::Reflex::PointerBuilder(builder.fReturnType);
      }
      for (int i = 0; i < numstar; ++i) {
         builder.fReturnType = ::Reflex::PointerBuilder(builder.fReturnType);
      }
      if (ref) {
         builder.fReturnType = ::Reflex::Type(builder.fReturnType, ::Reflex::REFERENCE, Reflex::Type::APPEND);
      }
   }
   else {
      if (numstar && (islower(G__var_type))) {
         G__var_type = toupper(G__var_type);
         --numstar;
      }
      if (numstar) {
         if (G__reftype == G__PARANORMAL) {
            G__reftype = G__PARAP2P;
            --numstar;
         }
         G__reftype += numstar;
      }
      if (G__tagnum && !G__tagnum.IsNamespace()) {
         // -- The return type is either a class, enum, struct, or union.
         builder.fReturnType = G__modify_type(G__tagnum, isupper(G__var_type), G__reftype, G__constvar, 0, 0);
      }
      else {
         // -- The return type is a fundamental type.
         builder.fReturnType = G__get_from_type(G__var_type, 1, G__constvar);
         builder.fReturnType = G__modify_type(builder.fReturnType, 0, G__reftype, G__constvar & ~G__CONSTVAR, 0, 0);
      }
   }
   builder.fIsexplicit = G__isexplicit;
   G__isexplicit = 0;
   G__reftype = G__PARANORMAL;
#ifndef G__NEWINHERIT
   G__p_ifunc->isinherit[func_now] = 0;
#endif // G__NEWINHERIT
   // member access control
   if (G__def_struct_member) {
      builder.fAccess = G__access;
   }
   else {
      builder.fAccess = G__PUBLIC;
   }
   builder.fStaticalloc = (char) G__static_alloc;
   // initiazlize baseoffset
#ifndef G__NEWINHERIT
   if (G__def_tagnum) {
      builder.fBasetagnum = G__def_tagnum;
   }
   else {
      builder.fBasetagnum = G__tagdefining;
   }
#endif // G__NEWINHERIT
   builder.fIsvirtual = G__virtual;
   builder.fIspurevirtual = 0;
   if (
      // -- We are virtual and have no virtual offset calculated yet.
#ifdef G__NEWINHERIT
      G__virtual && (G__struct.virtual_offset[G__get_tagnum(G__tagdefining)] == G__PVOID)
#else // G__NEWINHERIT
      G__virtual && (G__struct.virtual_offset[G__p_ifunc->basetagnum[func_now]] == G__PVOID)
#endif // G__NEWINHERIT
      // --
   ) {
      // -- We are a virtual function, set our virtual offset.
      ::Reflex::Scope store_tagnum = G__tagnum;
      ::Reflex::Type store_typenum = G__typenum;
      char store_type = G__var_type;
      G__tagnum = ::Reflex::Scope();
      G__typenum = ::Reflex::Type();
      G__var_type = 'l';
      int store_access = G__access;
#ifdef G__DEBUG2
      G__access = G__PUBLIC;
#else // G__DEBUG2
      G__access = G__PRIVATE;
#endif // G__DEBUG2
      G__letvariable("G__virtualinfo", G__null,::Reflex::Scope::GlobalScope(), G__p_local);
      G__access = store_access;
      G__var_type = store_type;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
#ifdef G__NEWINHERIT
      G__struct.virtual_offset[G__get_tagnum(G__tagdefining)] = (char*)(G__struct.size[G__get_tagnum(G__tagdefining)] - G__LONGALLOC);
#else // G__NEWINHERIT
      G__struct.virtual_offset[G__p_ifunc->basetagnum[func_now]] = (char*)(G__struct.size[G__p_ifunc->basetagnum[func_now]] - G__LONGALLOC);
#endif // G__NEWINHERIT
      // --
   }
   G__virtual = 0; // this position is not very best
   builder.fProp.comment.p.com = 0;
   builder.fProp.comment.filenum = -1;
   // initialize virtual table index
   //  TODO, may need to this this in other places too, need investigation
   builder.fProp.entry.vtblindex = -1;
   builder.fProp.entry.vtblbasetagnum = -1;
   builder.fProp.entry.busy = 0;
   builder.fProp.iscpplink = (char) G__iscpp;
   //
   //  Remember the file position of the beginning of the parameters.
   //
   fpos_t temppos;
   fgetpos(G__ifile.fp, &temppos);
   int store_line_number = G__ifile.line_number;
   //
   //  Try to read in a single parameter declaration.
   //
   int isparam = 0;
   G__StrBuf paraname_sb(G__LONGLINE);
   char *paraname = paraname_sb;
   int cin = G__fgetname_template(paraname, "<*&,()=");
   if (strlen(paraname) && isspace(cin)) {
      // -- There was an argument and the parsing was stopped by whitespace.
      // It is possible that we have a namespace name followed by '::', in
      // which case we have to grab more before stopping!
      G__StrBuf more_sb(G__LONGLINE);
      char *more = more_sb;
      int namespace_tagnum = G__defined_tagname(paraname, 2);
      while (
         isspace(cin) &&
         (
            ((namespace_tagnum != -1) && (G__struct.type[namespace_tagnum] == 'n')) ||
            !strcmp("std", paraname) ||
            (paraname[strlen(paraname)-1] == ':')
         )
      ) {
         cin = G__fgetname(more, "<*&,)=");
         strcat(paraname, more);
         namespace_tagnum = G__defined_tagname(paraname, 2);
      }
   }
   //
   //  Now that we have a parameter declaration, subject
   //  it to various tests to determine if this is a function
   //  taking no arguments, and whether or not this is an
   //  ansi-style function declaration.
   //
   int isvoid = 0;
   builder.fProp.entry.ansi = 0;
   if (!paraname[0]) {
      // -- No parameters given.
      isvoid = 1;
      if (G__iscpp || G__def_struct_member) {
         // -- We are C++, force ansi.
         builder.fProp.entry.ansi = 1;
      }
   }
   else {
      // -- We have parameters.
      if (!strcmp("void", paraname)) {
         // -- Check, we may have the special case of myfunc(void).
         if (isspace(cin)) {
            cin = G__fgetspace();
         }
         switch (cin) {
            case ',':
            case ')':
               // -- We have the special case of myfunc(void).
               isvoid = 1;
               builder.fProp.entry.ansi = 1;
               break;
            case '*':
            case '(':
               builder.fProp.entry.ansi = 1;
               break;
            default:
               G__genericerror("G__make_ifunctable: 823: Syntax error");
               isvoid = 1;
               break;
         }
      }
      else if (!strcmp("register", paraname)) {
         builder.fProp.entry.ansi = 1;
      }
      else if (G__istypename(paraname) || strchr(paraname, '[') || G__friendtagnum) {
         builder.fProp.entry.ansi = 1;
      }
      else {
         if (G__def_struct_member) {
            G__fprinterr(G__serr, "Error: unrecognized parameter type '%s'\n", paraname);
            G__genericerror("G__make_ifunctable: 836: Syntax error");
         }
         if ((G__globalcomp < G__NOLINK) && !G__nonansi_func
#ifdef G__ROOT
               && strncmp(funcheader, "ClassDef", 8)
#endif // G__ROOT
         ) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: Unknown type %s in function argument", paraname);
               G__printlinenum();
            }
         }
      }
   }
   //
   //  Skip to the end of the function parameters.
   //
   if (cin != ')') {
      cin = G__fignorestream(")");
   }
   G__static_alloc = 0; // FIXME: Do we need to reset this global state here?  Most likely not!
   // to get type of function parameter
   //
   // If ANSI style header, rewind file position to
   //       func(int a ,double b )   ANSI
   //            ^
   // and check type of paramaters and store it into G__ifunc
   //
   if (builder.fProp.entry.ansi) {
      if (isvoid | (funcheader[0] == '~')) {
         // -- No parameters, initialize builder.fParams_type,
         //    builder.fParams_names, and builder.fDefault_vals
         //    to empty.
      }
      else {
         // -- We are an ansi-style function which takes parameters, rewind and parse them.
         if (G__dispsource) {
            // -- Do not display the next 1000 characters read.  // FIXME: This is arbitrary and may fail!
            G__disp_mask = 1000;
         }
         //
         //  Rewind the file position.
         //
         fsetpos(G__ifile.fp, &temppos);
         G__ifile.line_number = store_line_number;
         // Parse the parameter declarations.
         //fprintf(stderr, "G__make_ifunctable: calling G__readansiproto for '%s'\n", funcheader);
         G__readansiproto(builder.fParams_type, builder.fParams_name, builder.fDefault_vals, &builder.fProp.entry.ansi);
         cin = ')';
         if (G__dispsource) {
            // -- Allow read characters to be seen again.
            G__disp_mask = 0;
         }
      }
   }
   // Set G__no_exec to skip ifunc body
   // This statement can be placed after endif.
   G__no_exec = 1;
   if (G__isfuncreturnp2f) {
      // function returning pointer to function
      //   type (*func(param1))(param2)  { } or;
      //                      ^ -----> ^
      cin = G__fignorestream(")");
      cin = G__fignorestream("(");
      cin = G__fignorestream(")");
   }
   cin = G__fgetstream_template(paraname, ",;{(");
   if (cin == '(') {
      int len = strlen(paraname);
      paraname[len++] = cin;
      cin = G__fgetstream(paraname + len, ")");
      len = strlen(paraname);
      paraname[len++] = cin;
      cin = G__fgetstream_template(paraname + len, ",;{");
   }
   // if header ignore following headers, else read func body
   if (
      (
         paraname[0] == '\0'
#ifndef G__OLDIMPLEMETATION817
         || ((strncmp(paraname, "throw", 5) == 0 || strncmp(paraname, "const throw", 11) == 0) && 0 == strchr(paraname, '='))
#endif // G__OLDIMPLEMETATION817
         // --
      ) &&
      ((cin == ',') || (cin == ';')) &&
      strncmp(funcheader, "ClassDef", 8)
   ) {
      // -- This is ANSI style func proto without param name.
      if (isparam) {
         fsetpos(G__ifile.fp, &temppos);
         G__ifile.line_number = store_line_number;
         G__readansiproto(builder.fParams_type, builder.fParams_name, builder.fDefault_vals, &builder.fProp.entry.ansi);
         cin = G__fignorestream(",;");
      }
      if (cin == ',') {
         // ignore other prototypes
         G__fignorestream(";");
         if (G__globalcomp < G__NOLINK) {
            G__genericerror("Limitation: Items in header must be separately specified");
         }
      }
      // entry fp = 0 means this is header
      builder.fProp.entry.p = 0;
      builder.fProp.entry.line_number = -1;
      builder.fProp.entry.filenum = -1;
      builder.fIspurevirtual = 0;
      // Key the class comment off of DeclFileLine rather than ClassDef
      // because ClassDef is removed by a preprocessor
      if (
         G__fons_comment && G__def_struct_member &&
         (
            !strncmp(funcname.c_str(), "DeclFileLine", 12) ||
            !strncmp(funcname.c_str(), "DeclFileLine(", 13) ||
            !strncmp(funcname.c_str(), "DeclFileLine", 12) ||
            !strncmp(funcname.c_str(), "DeclFileLine(", 13)
         )
      ) {
         G__fsetcomment(&G__struct.comment[G__get_tagnum(G__tagdefining)]);
      }
   }
   else if (
      !strncmp(paraname, "=", 1) ||
      !strncmp(paraname, "const =", 7) ||
      !strncmp(paraname, "const=", 6)
#ifndef G__OLDIMPLEMETATION817
      || ((!strncmp(paraname, "throw", 5) || !strncmp(paraname, "const throw", 11)) && strchr(paraname, '='))
#endif // G__OLDIMPLEMETATION817
      // --
   ) {
      char* p = strchr(paraname, '=');
      if (G__int(G__getexpr(p + 1)) != 0) {
         G__genericerror("Error: invalid pure virtual function initializer");
      }
      // this is ANSI style func proto without param name
      if (!builder.fProp.entry.ansi) {
         builder.fProp.entry.ansi = 1;
      }
      if (isparam) {
         fsetpos(G__ifile.fp, &temppos);
         G__ifile.line_number = store_line_number;
         G__readansiproto(builder.fParams_type, builder.fParams_name, builder.fDefault_vals, &builder.fProp.entry.ansi);
         cin = G__fignorestream(",;");
      }
      if (cin == ',') {
         // ignore other prototypes
         G__fignorestream(";");
         if (G__globalcomp < G__NOLINK) {
            G__genericerror("Limitation: Items in header must be separately specified");
         }
      }
      // entry fp = 0 means this is header
      builder.fProp.entry.p = 0;
      builder.fProp.entry.line_number = -1;
      builder.fIspurevirtual = 1;
      if (G__tagdefining) {
         //fprintf(stderr, "G__make_ifunctable: Incrementing abstract count of class '%s' cnt: %d  for: '%s'\n", G__tagdefining.Name(Reflex::SCOPED).c_str(), G__struct.isabstract[G__get_tagnum(G__tagdefining)], funcheader);
         ++G__struct.isabstract[G__get_tagnum(G__tagdefining)];
      }
      if (funcname.c_str()[0] == '~') {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: Pure virtual destructor may cause problem. Define as 'virtual %s() { }'", funcname.c_str());
            G__printlinenum();
         }
      }
      if (!strncmp(paraname, "const", 5)) {
         builder.fIsconst |= G__CONSTFUNC;
      }
   }
   else if (!strcmp(paraname, "const") || !strcmp(paraname, "const ")) {
      // this is ANSI style func proto without param name
      if (!builder.fProp.entry.ansi) {
         builder.fProp.entry.ansi = 1;
      }
      if (isparam) {
         fsetpos(G__ifile.fp, &temppos);
         G__ifile.line_number = store_line_number;
         G__readansiproto(builder.fParams_type, builder.fParams_name, builder.fDefault_vals, &builder.fProp.entry.ansi);
         cin = G__fignorestream(",;{");
      }
      if (cin == ',') {
         // ignore other prototypes
         G__fignorestream(";");
         if (G__globalcomp < G__NOLINK) {
            G__genericerror("Limitation: Items in header must be separately specified");
         }
      }
      if (cin == '{') {
         // it is possible that this is a function body.
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         dobody = 1;
      }
      else {
         // entry fp = 0 means this is header
         builder.fProp.entry.p = 0;
         builder.fProp.entry.line_number = -1;
      }
      builder.fIspurevirtual = 0;
      builder.fIsconst |= G__CONSTFUNC;
   }
   else if (
      G__def_struct_member &&
      (
         (cin == '}') ||
         ((cin == ';') && paraname[0] && (paraname[0] != ':')) ||
         ((cin == ';') && !strncmp(funcheader, "ClassDef", 8))
      )
   ) {
      // Function macro as member declaration
      // restore file position
      //   func(   int   a   ,  double   b )
      //        ^  <------------------------+
      //
      fsetpos(G__ifile.fp, &temppos);
      G__ifile.line_number = store_line_number;
      if (G__dispsource) {
         G__disp_mask = 1000;
      }
      strcpy(paraname, funcheader);
      cin = G__fgetstream(paraname + strlen(paraname), ")");
      int iin = strlen(paraname);
      paraname[iin] = ')';
      paraname[iin+1] = '\0';
      if (G__dispsource) {
         G__disp_mask = 0;
      }
      G__no_exec = 0; // must be set to 1 again after return
      G__func_now = ::Reflex::Member();
      G__p_ifunc = store_ifunc;
      G__execfuncmacro(paraname, &iin);
      if (!iin) {
         G__genericerror("Error: unrecognized language construct");
      }
      else if (
         G__fons_comment &&
         G__def_struct_member &&
         (
            !strncmp(paraname, "ClassDef", 8) ||
            !strncmp(paraname, "ClassDef(", 9) ||
            !strncmp(paraname, "ClassDefT(", 10)
         )
      ) {
         G__fsetcomment(&G__struct.comment[G__get_tagnum(G__tagdefining)]);
      }
      return;
   }
   else {
      // Body of the function, skip until
      // 'func(param)  type param;  { }'
      //                             ^
      // and rewind file to just before the '{'
      //
      if (!paraname[0] && isparam) {
         // Strange case
         //   type f(type) { };
         //          ^ <--  ^
         fsetpos(G__ifile.fp, &temppos);
         G__ifile.line_number = store_line_number;
         G__readansiproto(builder.fParams_type, builder.fParams_name, builder.fDefault_vals, &builder.fProp.entry.ansi);
         cin = G__fignorestream("{");
      }
      if (!strcmp(funcname.c_str(), "main") && (G__def_tagnum == ::Reflex::Scope::GlobalScope())) {
         G__ismain = G__MAINEXIST;
      }
      // following part is needed to detect inline new/delete in header
      if ((G__globalcomp == G__CPPLINK) || (G__globalcomp == R__CPPLINK)) {
         if (
            !strcmp(funcname.c_str(), "operator new") &&
            (builder.fParams_type.size() == 2) &&
            !(G__is_operator_newdelete & G__MASK_OPERATOR_NEW)
         ) {
            G__is_operator_newdelete |= G__IS_OPERATOR_NEW;
         }
         if (
            !strcmp(funcname.c_str(), "operator delete") &&
            !(G__is_operator_newdelete & G__MASK_OPERATOR_DELETE)
         ) {
            G__is_operator_newdelete |= G__IS_OPERATOR_DELETE;
         }
      }
      if ((paraname[0] == ':') && !builder.fProp.entry.ansi) {
         builder.fProp.entry.ansi = 1;
      }
      if (cin != '{') {
         G__fignorestream("{");
      }
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
      // skip body of the function surrounded by '{' '}'.
      // G__exec_statement(&brace_level); does the job
      builder.fIspurevirtual = 0;
      dobody = 1;
   }
   if (G__nonansi_func) {
      builder.fProp.entry.ansi = 0;
   }
#ifdef G__DETECT_NEWDEL
   //
   // operator new(size_t,void*) , operator delete(void*) detection
   // This is only needed for Linux g++
   //
   if (G__CPPLINK == G__globalcomp || R__CPPLINK == G__globalcomp) {
      if (strcmp(funcname.c_str(), "operator new") == 0 &&
            2 == G__p_ifunc->para_nu[func_now] &&
            0 == (G__is_operator_newdelete&G__MASK_OPERATOR_NEW))
         G__is_operator_newdelete |= G__IS_OPERATOR_NEW;
      if (strcmp(funcname.c_str(), "operator delete") == 0 &&
            0 == (G__is_operator_newdelete&G__MASK_OPERATOR_DELETE))
         G__is_operator_newdelete |= G__IS_OPERATOR_DELETE;
   }
#endif // G__DETECT_NEWDEL
   if (
      (strcmp(funcname.c_str(), "operator delete") == 0 || strcmp(funcname.c_str(), "operator delete[]") == 0) &&
      -1 != G__get_tagnum(G__p_ifunc)
   ) {
      builder.fStaticalloc = 1;
   }
   ::Reflex::Member newFunction = builder.Build(funcname);
   G__func_now = newFunction;
   // Set constructor,copy constructor, destructor, operator= flags.
   if (G__def_struct_member && G__def_tagnum) {
      if ('~' == funcname.c_str()[0]) {
         /* Destructor */
         G__struct.funcs[G__get_tagnum(G__def_tagnum)] |= G__HAS_DESTRUCTOR;
      }
      else if (strcmp(G__struct.name[G__get_tagnum(G__def_tagnum)], funcname.c_str()) == 0) {
         if (0 == newFunction.TypeOf().FunctionParameterSize() || newFunction.FunctionParameterDefaultAt(0).length()) {
            /* Default constructor */
            G__struct.funcs[G__get_tagnum(G__def_tagnum)] |= G__HAS_DEFAULTCONSTRUCTOR;
         }
         else if ((1 == newFunction.TypeOf().FunctionParameterSize() ||
                   newFunction.FunctionParameterDefaultAt(1).length()) &&
                  G__def_tagnum == newFunction.TypeOf().FunctionParameterAt(0).RawType() &&
                  newFunction.TypeOf().FunctionParameterAt(0).FinalType().IsReference()
                 ) {
            /* Copy constructor */
            G__struct.funcs[G__get_tagnum(G__def_tagnum)] |= G__HAS_COPYCONSTRUCTOR;
         }
         else {
            G__struct.funcs[G__get_tagnum(G__def_tagnum)] |= G__HAS_XCONSTRUCTOR;
         }
      }
      else if (strcmp("operator=", funcname.c_str()) == 0) {
         /* operator= */
         G__struct.funcs[G__get_tagnum(G__def_tagnum)] |= G__HAS_ASSIGNMENTOPERATOR;
      }
      else if (!strcmp("operator new", funcname.c_str())) {
         if (newFunction.TypeOf().FunctionParameterSize() == 1) {
            G__struct.funcs[G__get_tagnum(G__def_tagnum)] |= G__HAS_OPERATORNEW1ARG;
         }
         else if (newFunction.TypeOf().FunctionParameterSize() == 2) {
            G__struct.funcs[G__get_tagnum(G__def_tagnum)] |= G__HAS_OPERATORNEW2ARG;
         }
      }
      else if (strcmp("operator delete", funcname.c_str()) == 0) {
         G__struct.funcs[G__get_tagnum(G__def_tagnum)] |= G__HAS_OPERATORDELETE;
      }
   }
   if (dobody) {
      int store_def_struct_member = G__def_struct_member;
      G__def_struct_member = 0;
      int brace_level = 0;
      G__exec_statement(&brace_level);
      G__def_struct_member = store_def_struct_member;
#ifdef G__ASM_FUNC
      G__get_funcproperties(newFunction)->entry.size = G__ifile.line_number - G__get_funcproperties(newFunction)->entry.line_number + 1;
#endif // G__ASM_FUNC
      // --
#ifdef G__ASM_WHOLEFUNC
      /***************************************************************
      * compile as bytecode at load time if -O10 or #pragma bytecode
      ***************************************************************/
      if (G__asm_loopcompile >= 10) {
         if (ifunc) {
            G__compile_function_bytecode(newFunction);
         }
         else {
            G__compile_function_bytecode(newFunction);
         }
      }
#endif // G__ASM_WHOLEFUNC
      // --
   }
   if (G__GetShlHandle()) {
      void *shlp2f = G__FindSymbol(newFunction);
      if (shlp2f) {
         G__get_funcproperties(newFunction)->entry.tp2f = shlp2f;
         G__get_funcproperties(newFunction)->entry.p = (void*)G__DLL_direct_globalfunc;
         G__get_funcproperties(newFunction)->filenum = G__GetShlFilenum();
         G__get_funcproperties(newFunction)->entry.size = -1;
         G__get_funcproperties(newFunction)->linenum = -1;
      }
   }
   if (G__fons_comment && G__def_struct_member) {
      if ((strncmp(newFunction.Name().c_str(), "ClassDef", 8) == 0 ||
            strncmp(newFunction.Name().c_str(), "ClassDef(", 9) == 0 ||
            strncmp(newFunction.Name().c_str(), "ClassDefT(", 10) == 0 ||
            strncmp(newFunction.Name().c_str(), "DeclFileLine", 12) == 0 ||
            strncmp(newFunction.Name().c_str(), "DeclFileLine(", 13) == 0)) {
         G__fsetcomment(&G__struct.comment[G__get_tagnum(G__tagdefining)]);
      }
      else {
         G__fsetcomment(&G__get_funcproperties(newFunction)->comment);
      }
   }
   G__no_exec = 0;
   G__func_now = ::Reflex::Member();
   G__p_ifunc = store_ifunc;
   return;
}

//______________________________________________________________________________
static int Cint::Internal::G__readansiproto(std::vector<Reflex::Type>& i_params_type, std::vector<std::pair<std::string, std::string> >& i_params_names, std::vector<G__value*>& i_params_default, char* i_ansi)
{
   //  func(type , type* , ...)
   //       ^
   std::string default_str;
   G__value* default_val = 0;
   int store_var_type;
   ::Reflex::Scope store_tagnum_default;
   int store_def_struct_member_default = 0;
   int store_exec_memberfunc = 0;

   *i_ansi = 1;
   int iin = 0;
   int c = 0;
   for (; c != ')'; ++iin) {
      // --
      if (iin == G__MAXFUNCPARA) {
         G__fprinterr(G__serr, "Limitation: cint can not accept more than %d function arguments", G__MAXFUNCPARA);
         G__printlinenum();
         G__fignorestream(")");
         return 1;
      }
      int isconst = G__VARIABLE;
      G__StrBuf buf_sb(G__LONGLINE);
      char *buf = buf_sb; // Parsing I/O buffer.
      buf[0] = '\0';
      // Get first keyword, id, or separator of the type specification.
      c = G__fgetname_template(buf, "&*[(=,)");
      if (strlen(buf) && isspace(c)) {
         // -- There was an argument and the parsing stopped at white space.
         // It is possible that we have a qualified name, check.
         // FIXME: A classname should be allowed here as well, the parameter type could be a nested type.
         int namespace_tagnum = G__defined_tagname(buf, 2);
         while (
            isspace(c) &&
            (
               (buf[strlen(buf)-1] == ':') ||
               ((namespace_tagnum != -1) && (G__struct.type[namespace_tagnum] == 'n')) ||
               !strcmp("std", buf)
            )
         ) {
            G__StrBuf more_sb(G__LONGLINE);
            char *more = more_sb;
            c = G__fgetname(more, "&*[(=,)");
            strcat(buf, more);
            namespace_tagnum = G__defined_tagname(buf, 2);
         }
      }
      //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
      // Check if we have reached an ellipsis (must be at end).
      if (!strcmp(buf, "...")) {
         *i_ansi = 2;
         break;
      }
      // Check for and consume const, volatile, auto, register and typename qualifiers.
      while (
         !strcmp(buf, "const") ||
         !strcmp(buf, "volatile") ||
         !strcmp(buf, "auto") ||
         !strcmp(buf, "register") ||
         (G__iscpp && !strcmp(buf, "typename"))
      ) {
         if (!strcmp(buf, "const")) {
            isconst |= G__CONSTVAR;
         }
         c = G__fgetname_template(buf, "&*[(=,)");
      }
      //
      //  Determine type.
      //
      char type = '\0';
      int tagnum = -1;
      int typenum = -1;
      int ptrcnt = 0; // number of pointers seen (pointer count)
      int isunsigned = 0; // unsigned seen flag and offset for type code
      // Process first keyword of type specifier (most type specifiers have only one keyword).
      {
         // Partially handle unsigned and signed keywords here.  Also do some integral promotions.
         if (!strcmp(buf, "unsigned") || !strcmp(buf, "signed")) {
            if (buf[0] == 'u') {
               isunsigned = -1;
            }
            else {
               isunsigned = 0;
            }
            switch (c) {
               case ',':
               case ')':
               case '&':
               case '[':
               case '(':
               case '=':
               case '*':
                  strcpy(buf, "int"); // FIXME: We destroy the real typename here!
                  //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                  break;
               default:
                  if (isspace(c)) {
                     c = G__fgetname(buf, ",)&*[(="); // FIXME: Change to G__getname_template???
                     //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                  }
                  else {
                     fpos_t pos;
                     fgetpos(G__ifile.fp, &pos);
                     int store_line = G__ifile.line_number;
                     c = G__fgetname(buf, ",)&*[(=");
                     //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                     if (strcmp(buf, "short") && strcmp(buf, "int") && strcmp(buf, "long")) {
                        G__ifile.line_number = store_line;
                        fsetpos(G__ifile.fp, &pos);
                        strcpy(buf, "int"); // FIXME: We destroy the real typename here!
                        //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                        c = ' ';
                     }
                  }
                  break;
            }
         }
         if (!strcmp(buf, "class")) {
            c = G__fgetname_template(buf, ",)&*[(=");
            //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            tagnum = G__search_tagname(buf, 'c');
            type = 'u';
         }
         else if (!strcmp(buf, "struct")) {
            c = G__fgetname_template(buf, ",)&*[(=");
            //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            tagnum = G__search_tagname(buf, 's');
            type = 'u';
         }
         else if (!strcmp(buf, "union")) {
            c = G__fgetname_template(buf, ",)&*[(=");
            //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            tagnum = G__search_tagname(buf, 'u');
            type = 'u';
         }
         else if (!strcmp(buf, "enum")) {
            c = G__fgetname_template(buf, ",)&*[(=");
            //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            tagnum = G__search_tagname(buf, 'e');
            type = 'i';
         }
         else if (!strcmp(buf, "int")) {
            type = 'i' + isunsigned;
         }
         else if (!strcmp(buf, "char")) {
            type = 'c' + isunsigned;
         }
         else if (!strcmp(buf, "short")) {
            type = 's' + isunsigned;
         }
         else if (!strcmp(buf, "long")) {
            if ((c != ',') && (c != ')') && (c != '(')) {
               fpos_t pos;
               fgetpos(G__ifile.fp, &pos);
               int store_line = G__ifile.line_number;
               int store_c = c;
               c = G__fgetname(buf, ",)&*[(="); // FIXME: Change to G__fgetname_template???
               //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
               if (!strcmp(buf, "long") || !strcmp(buf, "double")) {
                  if (!strcmp(buf, "long")) {
                     if (isunsigned) {
                        type = 'm';
                     }
                     else {
                        type = 'n';
                     }
                  }
                  else {
                     type = 'q';
                  }
               }
               else if (!strcmp(buf, "int")) {
                  type = 'l' + isunsigned;
               }
               else {
                  G__ifile.line_number = store_line;
                  fsetpos(G__ifile.fp, &pos);
                  c = store_c;
                  type = 'l' + isunsigned;
               }
            }
            else {
               type = 'l' + isunsigned;
            }
         }
         else if (!strcmp(buf, "float")) {
            type = 'f' + isunsigned;
         }
         else if (!strcmp(buf, "double")) {
            type = 'd' + isunsigned;
         }
         else if (!strcmp(buf, "bool")) {
            type = 'g';
         }
         else if (!strcmp(buf, "void")) {
            type = 'y';
         }
         else if (!strcmp(buf, "FILE")) {
            type = 'e';
         }
         else {
            ::Reflex::Scope store_tagdefining = G__tagdefining;
            ::Reflex::Scope store_def_tagnum = G__def_tagnum;
            if (G__friendtagnum && !G__friendtagnum.IsTopScope()) {
               G__tagdefining = G__friendtagnum;
               G__def_tagnum = G__friendtagnum;
            }
            //::Reflex::Scope definer = G__get_envtagnum();
            //fprintf(stderr, "G__readansiproto: definer: '%s'\n", definer.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
            //fprintf(stderr, "G__readansiproto: search for typedef named: '%s' ... \n", buf);
            ::Reflex::Type ty = G__find_typedef(buf);
            if (ty) {
               typenum = G__get_properties(ty)->typenum;
            }
            if (typenum != -1) {
               //fprintf(stderr, "G__readansiproto: search for typedef named: '%s' found.\n", buf);
               tagnum = G__get_properties(ty)->tagnum;
               type = G__get_type(ty);
               ptrcnt += G__get_nindex(ty);
               isconst |= G__get_isconst(ty);  // TODO: This is exactly duplicating the bad behavior of cint5.
            }
            else {
               //fprintf(stderr, "G__readansiproto: search for tagname: '%s' ... \n", buf);
               tagnum = G__defined_tagname(buf, 1);
               if (tagnum != -1) {
                  if (G__struct.type[tagnum] == 'e') { // FIXME: Enumerators are not ints!
                     //fprintf(stderr, "G__readansiproto: found enumerator: '%s' ... \n", buf);
                     type = 'u'; // FIXME: cint5 has 'i' here.
                  }
                  else {
                     // re-evaluate typedef name in case of template class
                     if (strchr(buf, '<')) {
                        ::Reflex::Type ty = G__find_typedef(buf);
                        if (ty) {
                           typenum = G__get_properties(ty)->typenum;
                        }
                     }
                     type = 'u';
                  }
               }
               else {
                  //fprintf(stderr, "G__readansiproto: failed to find tagname: '%s' ... \n", buf);
                  if (G__fpundeftype) {
                     tagnum = G__search_tagname(buf, 'c');
                     fprintf(G__fpundeftype, "class %s; /* %s %d */\n", buf, G__ifile.name, G__ifile.line_number);
                     fprintf(G__fpundeftype, "#pragma link off class %s;\n\n", buf);
                     G__struct.globalcomp[tagnum] = G__NOLINK;
                     type = 'u';
                  }
                  else {
                     // In case of f(unsigned x,signed y)
                     type = 'i' + isunsigned;
                     if (!isdigit(buf[0]) && !isunsigned) {
                        if (G__dispmsg >= G__DISPWARN) {
                           G__fprinterr(G__serr, "Warning: Unknown type '%s' in function argument handled as int", buf);
                           G__printlinenum();
                        }
                     }
                  }
               }
            }
            G__tagdefining = store_tagdefining;
            G__def_tagnum = store_def_tagnum;
         }
      }
      //
      //  Process the rest of the type specifier and the parameter name.
      //
      int is_a_reference = 0;
      int has_a_default = 0;
      G__StrBuf param_name_sb(G__LONGLINE);
      char *param_name = param_name_sb;
      param_name[0] = '\0';
      {
         int arydim = 0; // Track which array bound we are processing, so we can handle an unspecified length array.
         while ((c != ',') && (c != ')')) {
            switch (c) {
               case EOF: // end-of-file
                  // -- Error, unexpected end of file, exit.
                  return 1;
               case ' ':
               case '\f':
               case '\n':
               case '\r':
               case '\t':
               case '\v':
                  // -- Whitespace, skip it.
                  c = G__fgetspace();
                  break;
               case '.': // ellipses
                  // -- We have reached an ellipses (must be at end), flag it and we are done.
                  *i_ansi = 2;
                  c = G__fignorestream(",)");
                  // Note: The enclosing loop will terminate after we break.
                  break;
               case '&': // reference
                  ++is_a_reference;
                  c = G__fgetspace();
                  break;
               case '*': // pointer
                  ++ptrcnt;
                  c = G__fgetspace();
                  break;
               case '[': // array bound
                  ++arydim;
                  if (
                     (G__globalcomp < G__NOLINK) &&
                     (
                        !param_name[0] || // No parameter name given, first array bound.
                        (param_name[0] == '[') // We are processing a second or greater array bound.
                     )
                  ) {
                     // read 'MyFunc(int [][30])' or 'MyFunc(int [])'
                     int len = strlen(param_name);
                     param_name[len++] = c;
                     param_name[len++] = ']';
                     // Ignore the given array bound value.
                     c = G__fignorestream("],)"); // <<<
                     //
                     //  Peek ahead looking for another array bound.
                     //
                     {
                        G__disp_mask = 1000;
                        fpos_t tmp_pos;
                        fgetpos(G__ifile.fp, &tmp_pos);
                        int tmp_line = G__ifile.line_number;
                        c = G__fgetstream(param_name + len, "[=,)");
                        fsetpos(G__ifile.fp, &tmp_pos);
                        G__ifile.line_number = tmp_line;
                        G__disp_mask = 0;
                     }
                     if (c == '[') {
                        // Collect all the rest of the array bounds.
                        c = G__fgetstream(param_name + len, "=,)");
                        ptrcnt = 0; // FIXME: This erases all pointers (int* ary[d][d] is broken!)
                        break;
                     }
                     else {
                        // G__fignorestream("],)") already called above <<<
                        param_name[0] = 0;
                     }
                  }
                  else {
                     c = G__fignorestream("],)");
                  }
                  ++ptrcnt;
                  c = G__fgetspace();
                  break;
               case '=': // default value
                  // -- Parameter has a default value, collect it, and we are done.
                  // Collect the rest of the parameter specification as the default text.
                  has_a_default = 1;
                  c = G__fgetstream_template(buf, ",)");
                  // Note: The enclosing loop will terminate after we break.
                  break;
               case '(': // Assume a function pointer type, e.g., MyFunc(int, int (*fp)(int, ...), int, ...)
                  // -- We have a function pointer type.
                  //
                  //  If the return type is a typedef or a class, struct, union, or enum,
                  //  then normalize the typename.
                  //
                  if (
                     (typenum != -1) ||
                     (tagnum != -1)
                  ) {
                     char* p = strrchr(buf, ' ');
                     if (p) {
                        ++p;
                     }
                     else {
                        p = buf;
                     }
                     strcpy(p, G__type2string(0, tagnum, typenum, 0, 0));
                  }
                  //
                  //  Normalize the rest of the parameter specification,
                  //  up to any default value, and collect the parameter name.
                  //
                  {
                     // Handle any ref part of the return type.
                     // FIXME: This is wrong, cannot have a pointer to a reference!
                     int i = strlen(buf);
                     if (is_a_reference) {
                        buf[i++] = '&';
                     }
                     is_a_reference = 0;
                     buf[i++] = ' ';
                     // Handle any pointer part of the return type.
                     // FIXME: This is wrong, cannot have a pointer to a reference!
                     for (int j = 0; j < ptrcnt; ++j) {
                        buf[i++] = '*';
                     }
                     ptrcnt = 0;
                     // Start constructing the parameter name part.
                     buf[i++] = '(';
                     c = G__fgetstream(buf + i, "*)");
                     if (c == '*') {
                        buf[i++] = c;
                        c = G__fgetstream(param_name, ")");
                        int j = 0;
                        for (; param_name[j] == '*'; ++j) {
                           buf[i++] = '*';
                        }
                        if (j) {
                           int k = 0;
                           while (param_name[j]) {
                              param_name[k++] = param_name[j++];
                           }
                           param_name[k] = 0;
                        }
                     }
                     if (c == ')') {
                        buf[i++] = ')';
                     }
                     // Copy out the rest of the parameter specification (up to a default value, if any).
                     c = G__fdumpstream(buf + i, ",)=");
                  }
                  buf[strlen(buf)] = '\0';
                  //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
#ifndef G__OLDIMPLEMENTATION2191
                  typenum = G__get_properties(G__declare_typedef(buf, '1', -1, 0, 0, G__NOLINK, -1, true))->typenum; // FIXME: G__search_typename(buf, '1', -1, 0)
                  type = '1';
#else // G__OLDIMPLEMENTATION2191
                  typenum = G__get_properties(G__declare_typedef(buf, 'Q', -1, 0, 0, G__NOLINK, -1, true))->typenum; // FIXME: G__search_typename(buf, 'Q', -1, 0)
                  type = 'Q';
#endif // G__OLDIMPLEMENTATION2191
                  tagnum = -1;
                  break;
               default: // Consume "long long", "const", "const*", array bounds, param name (with def val)
                  // -- Not whitespace or separator.
                  if (!strcmp(param_name, "long") && !strcmp(buf, "long")) { // Process "long long" type.
                     type = 'n';
                     tagnum = -1;
                     typenum = -1;
                  }
                  // Collect next keyword or id into param_name.
                  param_name[0] = c;
                  c = G__fgetstream(param_name + 1, "[=,)& \t");
                  if (!strcmp(param_name, "const")) { // handle const keyword
                     if (ptrcnt) {
                        isconst |= G__PCONSTVAR; // FIXME: This is intentionally wrong!  Fix the code that depends on this!
                     } else {
                        isconst |= G__CONSTVAR;
                     }
                     param_name[0] = 0;
                  }
                  else if (!strcmp(param_name, "const*")) { // handle const keyword and a single pointer spec
                     if (ptrcnt) {
                       isconst |= G__PCONSTVAR; // FIXME: This is intentionally wrong!  Fix the code that depends on this!
                     } else {
                       isconst |= G__CONSTVAR;
                     }
                     ++ptrcnt;
                     param_name[0] = 0;
                  }
                  else { // Process any array bounds and possible default value.
                     while ((c == '[') || (c == ']')) {
                        if (c == '[') { // Consume an array bound, converting it into a pointer.
                           ++ptrcnt;
                           ++arydim;
                           if ((G__globalcomp < G__NOLINK) && (arydim == 2)) {
                              // -- We have just seen the beginning of the second array bound.
                              int len = strlen(param_name);
                              if (param_name[0] == ']') {
                                 len = 0;
                              }
                              strcpy(param_name + len, "[]");
                              ptrcnt -= 2;
                              len = strlen(param_name);
                              fseek(G__ifile.fp, -1, SEEK_CUR);
                              if (G__dispsource) {
                                 G__disp_mask = 1;
                              }
                              c = G__fgetstream(param_name + len, "=,)");
                              // Note: Either we next process a default value, or enclosing loop terminates.
                              break;
                           }
                        }
                        // Skip till next array bound or end of parameter.
                        c = G__fignorestream("[=,)");
                     }
                     if (c == '=') {
                        has_a_default = 1;
                        c = G__fgetstream_template(buf, ",)");
                        //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                        // Note: Enclosing loop will terminate after we break.
                     }
                  }
                  break;
            }
         }
      }
      if (has_a_default) {
         // -- Remember the default text, and then evaluate it.
         default_val = new G__value; // FIXME: memory leak?
         default_str = buf;
         if (buf[0] == '(') {
            int len = strlen(buf);
            if (
               (len > 5) &&
               (buf[len-4] != '*') &&
               !strcmp(")()", buf + len - 3) &&
               strchr(buf, '<')
            ) {
               int i = 1;
               for (; i < len - 3; ++i) {
                  buf[i-1] = buf[i];
               }
               strcpy(buf + i - 1, "()");
               //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            }
         }
         ::Reflex::Scope store_def_tagnum = G__def_tagnum;
         ::Reflex::Scope store_tagdefining = G__tagdefining;
         int store_prerun = G__prerun;
         int store_decl = G__decl;
         store_var_type = G__var_type;
         G__var_type = 'p';
         if (G__def_tagnum) {
            store_tagnum_default = G__tagnum;
            store_exec_memberfunc = G__exec_memberfunc;
            store_def_struct_member_default = G__def_struct_member;
            G__tagnum = G__def_tagnum;
            G__exec_memberfunc = 1;
            G__def_struct_member = 0;
         }
         if (G__globalcomp == G__NOLINK) {
            G__prerun = 0;
            G__decl = 1;
         }
         {
            ::Reflex::Scope store_pifunc = G__p_ifunc;
            G__p_ifunc = ::Reflex::Scope::GlobalScope(); // FIXME: &G__ifunc
            if (G__decl && G__prerun && ((G__globalcomp == G__CPPLINK) || (G__globalcomp == R__CPPLINK))) {
               G__noerr_defined = 1;
            }
            *default_val = G__getexpr(buf);
            if (G__decl && G__prerun && ((G__globalcomp == G__CPPLINK) || (G__globalcomp == R__CPPLINK))) {
               G__noerr_defined = 0;
            }
            G__value* val = default_val;
            if (is_a_reference && !ptrcnt && (toupper(G__get_type(*val)) != toupper(type) || G__value_typenum(*val).RawType() != G__Dict::GetDict().GetType(tagnum).RawType())) {
               G__StrBuf tmp_sb(G__ONELINE);
               char *tmp = tmp_sb;
               sprintf(tmp, "%s(%s)", G__type2string(type, tagnum, -1, 0, 0), buf);
               *val = G__getexpr(tmp);
               if (G__get_type(*val) == 'u') {
                  val->ref = val->obj.i;
               }
            }
            G__p_ifunc = store_pifunc;
         }
         G__prerun = store_prerun;
         G__decl = store_decl;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         if (G__def_tagnum) {
            G__tagnum = store_tagnum_default;
            G__exec_memberfunc = store_exec_memberfunc;
            G__def_struct_member = store_def_struct_member_default;
         }
         G__var_type = store_var_type;
      }
      // Now finalize the type of the parameter based on the full parse.
      //if (isupper(type) && ptrcnt) {
      //   // Note: This can only happen if the type was a typedef to an array of pointers.
      //   ++ptrcnt;
      //}
      //if ((typenum != -1) && (G__get_reftype(G__Dict::GetDict().GetTypedef(typenum)) >= G__PARAP2P)) {
      //   ptrcnt += G__get_reftype(typenum) - G__PARAP2P + 2;
      //   type = tolower(type);
      //}
      ::Reflex::Type paramType;
      if (typenum != -1) { // a typedef was used in the parameter type
         paramType = G__Dict::GetDict().GetTypedef(typenum);
         if (isconst & G__CONSTVAR) {
            paramType = Reflex::Type(paramType, Reflex::CONST, Reflex::Type::APPEND);
         }
         int extra_ptrcnt = ptrcnt - G__get_nindex(G__Dict::GetDict().GetTypedef(typenum));
         for (int i = 0; i < extra_ptrcnt; ++i) {
            paramType = ::Reflex::PointerBuilder(paramType);
         }
         if (isconst & G__PCONSTVAR) {
            paramType = Reflex::Type(paramType, Reflex::CONST, Reflex::Type::APPEND);
         }
         if (is_a_reference) {
            paramType = Reflex::Type(paramType, Reflex::REFERENCE, Reflex::Type::APPEND);
         }
      }
      else if (tagnum != -1) { // a class, enum, struct, or union type was used in the parameter type
         paramType = G__Dict::GetDict().GetType(tagnum);
         if (!paramType) { // If not found, try again looking for the helper typdef for templates with default params.
            paramType = ::Reflex::Type::ByName(G__struct.name[tagnum]);
         }
         if (isconst & G__CONSTVAR) {
            paramType = Reflex::Type(paramType, Reflex::CONST, Reflex::Type::APPEND);
         }
         for (int i = 0; i < ptrcnt; ++i) {
            paramType = ::Reflex::PointerBuilder(paramType);
         }
         if (isconst & G__PCONSTVAR) {
            paramType = Reflex::Type(paramType, Reflex::CONST, Reflex::Type::APPEND);
         }
         if (is_a_reference) {
            paramType = Reflex::Type(paramType, Reflex::REFERENCE, Reflex::Type::APPEND);
         }
      }
      else {
         paramType = G__get_from_type(type, 1, isconst);
         for (int i = 0; i < ptrcnt; ++i) {
            paramType = ::Reflex::PointerBuilder(paramType);
         }
         if (isconst & G__PCONSTVAR) {
            paramType = Reflex::Type(paramType, Reflex::CONST, Reflex::Type::APPEND);
         }
         if (is_a_reference) {
            paramType = Reflex::Type(paramType, Reflex::REFERENCE, Reflex::Type::APPEND);
         }
      }
      //fprintf(stderr, "G__readansiproto: paramType: '%s'\n", paramType.Name(::Reflex::SCOPED | ::Reflex::QUALIFIED).c_str());
      //fprintf(stderr, "G__readansiproto: param_name: '%s'\n", param_name);
      //fprintf(stderr, "G__readansiproto: default_str: '%s'\n", default_str.c_str());
      i_params_type.push_back(paramType);
      i_params_names.push_back(std::make_pair(param_name, default_str));
      i_params_default.push_back(default_val);
      // Free any default value we may have allocated.
      //delete default_val; // FIXME: This can be a stack variable, change it.
      default_val = 0;
      default_str = "";
   }
   return 0;
}

//______________________________________________________________________________
static int G__matchpointlevel(int param_reftype, int formal_reftype)
{
   switch (param_reftype) {
      case G__PARANORMAL:
      case G__PARAREFERENCE:
         if (G__PARANORMAL == formal_reftype || G__PARAREFERENCE == formal_reftype)
            return(1);
         else
            return(0);
      default:
         return(formal_reftype == param_reftype);
   }
}

//______________________________________________________________________________
static int G__param_match(char formal_type, const ::Reflex::Scope& formal_tagnum, G__value* default_parameter, char param_type, const ::Reflex::Scope& param_tagnum, G__value* param, char* parameter, int funcmatch, int rewind_arg, int formal_reftype, int formal_isconst)
{
   // -- FIXME: Describe this function!
   int match = 0;
   static int recursive = 0; // FIXME: This is not thread-safe!
   //
   //  What is this about?
   //
   if (default_parameter && (param_type == '\0')) {
      return 2;
   }
   //
   //  First try an exact match, if allowed.
   //
   if (funcmatch >= G__EXACT) {
      if (
         !recursive && // Not recursive, and
         (G__get_tagnum(param_tagnum) == G__get_tagnum(formal_tagnum)) && // Argument tagnum matches formal parameter tagnum, and
         (
            (param_type == formal_type) || // Argument type code matches formal parameter typecode, or
            (
               (param_type == 'I') && // Argument type is int*, and
               (formal_type == 'U') && // Formal parameter type is pointer to struct, and
               (G__get_tagnum(formal_tagnum) != -1) && // Formal parameter rawtype is valid, and
               (G__get_tagtype(formal_tagnum) == 'e') // Formal parameter rawtype is an enum
            ) || // or,
            (
               (param_type == 'U') && // Argument type is pointer to struct, and
               (formal_type == 'I') && // Formal parameter type is int*, and
               (G__get_tagnum(param_tagnum) != -1) && // Argument rawtype is valid, and
               (G__get_tagtype(param_tagnum) == 'e') // Argument rawtype is an enum
            ) || // or,
            (
               (param_type == 'i') &&  // Argument type is int, and
               (formal_type == 'u') && // Formal parameter type is struct, and
               (G__get_tagnum(formal_tagnum) != -1) && // Formal parameter rawtype is valid, and
               (G__get_tagtype(formal_tagnum) == 'e') // Formal parameter rawtype is an enum
            ) || // or,
            (
               (param_type == 'u') &&  // Argument type is struct, and
               (formal_type == 'i') && // Formal parameter type is int, and
               (G__get_tagnum(param_tagnum) != -1) && // Argument rawtype is valid, and
               (G__get_tagtype(param_tagnum) == 'e') // Argument rawtype is an enum
            )
         )
      ) {
         match = 1;
      }
   }
   //
   //  If no match yet, try the standard promotions, if allowed.
   //
   if (!match && (funcmatch >= G__PROMOTION)) {
      switch (formal_type) {
         case 'd':
         case 'f':
            switch (param_type) {
               case 'b':
               case 'c':
               case 'r':
               case 's':
               case 'h':
               case 'i':
               case 'k':
               case 'l':
#define G__OLDIMPLEMENTATION1165
               case 'd':
               case 'f':
                  match = 1;
                  break;
               default:
                  match = 0;
                  break;
            }
            break;
         case 'l':
            switch (param_type) {
               case 'b':
               case 'c':
               case 'r':
               case 's':
                  /* case 'h': */
               case 'i':
                  /* case 'k': */
               case 'l':
                  match = 1;
                  break;
               default:
                  match = 0;
                  break;
            }
            break;
         case 'i':
            switch (param_type) {
               case 'b':
               case 'c':
               case 'r':
               case 's':
                  /* case 'h': */
               case 'i':
                  /* case 'k': */
                  /* case 'l': */
                  match = 1;
                  break;
               case 'u':
                  if ('e' == G__get_tagtype(param_tagnum)) {
                     if (param->ref) param->obj.i = *(long*)(param->ref);
                     match = 1;
                     break;
                  }
               default:
                  match = 0;
                  break;
            }
            break;
         case 's':
            switch (param_type) {
               case 'b':
               case 'c':
                  /* case 'r': */
               case 's':
                  /* case 'h': */
                  /* case 'i': */
                  /* case 'k': */
                  /* case 'l': */
                  match = 1;
                  break;
               default:
                  match = 0;
                  break;
            }
            break;
         case 'k':
            switch (param_type) {
               case 'b':
                  /* case 'c': */
               case 'r':
                  /* case 's': */
               case 'h':
                  /* case 'i': */
               case 'k':
                  /* case 'l': */
                  match = 1;
                  break;
               default:
                  match = 0;
                  break;
            }
            break;
         case 'h':
            switch (param_type) {
               case 'b':
                  /* case 'c': */
               case 'r':
                  /* case 's': */
               case 'h':
                  /* case 'i': */
                  /* case 'k': */
                  /* case 'l': */
                  match = 1;
                  break;
               default:
                  match = 0;
                  break;
            }
            break;
         case 'r':
            switch (param_type) {
               case 'b':
                  /* case 'c': */
               case 'r':
                  /* case 's': */
                  /* case 'h': */
                  /* case 'i': */
                  /* case 'k': */
                  /* case 'l': */
                  match = 1;
                  break;
               default:
                  match = 0;
                  break;
            }
            break;
         case 'u':
            if (
               (G__get_tagnum(formal_tagnum) >= 0) &&
               (G__get_tagtype(formal_tagnum) == 'e')
            ) {
               switch (param_type) {
                  case 'i':
                  case 's':
                  case 'l':
                  case 'c':
                  case 'h':
                  case 'r':
                  case 'k':
                  case 'b':
                     match = 1;
                     break;
                  default:
                     match = 0;
                     break;
               }
            }
            else {
               match = 0;
            }
            break;
         default:
            match = 0;
            break;
      }
   }
   //
   //  If no match yet, try the standard conversions, if allowed.
   //
   if (!match && (funcmatch >= G__STDCONV)) {
      switch (formal_type) {
         case 'b':
         case 'c':
         case 'r':
         case 's':
         case 'h':
         case 'i':
         case 'k':
         case 'l':
         case 'd':
         case 'f':
            switch (param_type) {
               case 'b':
               case 'c':
               case 'r':
               case 's':
               case 'h':
               case 'i':
               case 'k':
               case 'l':
               case 'd':
               case 'f':
                  match = 1;
                  break;
               default:
                  match = 0;
                  break;
            }
            break;
         case 'C':
            switch (param_type) {
               case 'i':
               case 'l':
                  if (!param->obj.i) {
                     match = 1;
                  }
                  else {
                     match = 0;
                  }
                  break;
               case 'Y':
                  match = 1;
                  break;
               default:
                  match = 0;
                  break;
            }
            break;
         case 'Y':
            if (isupper(param_type) || !param->obj.i) {
               match = 1;
            }
            else {
               match = 0;
            }
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case '1': // questionable
            if ((param_type == '1') || (param_type == 'C') || (param_type == 'Y')) {
               match = 1;
            }
            else {
               match = 0;
            }
            break;
#else // G__OLDIMPLEMENTATION2191
         case 'Q': // questionable
            if ((param_type == 'Q') || (param_type == 'C') || (param_type == 'Y')) {
               match = 1;
            }
            else {
               match = 0;
            }
            break;
#endif // G__OLDIMPLEMENTATION2191
#ifdef G__WORKAROUND000209_1
         case 'u':
            // reference type conversion should not be handled in this way.
            // difference was found from g++ when activating this part.
            // Added condition for formal_reftype and recursive, then things
            // are working 1999/12/5
            if ((formal_reftype == G__PARAREFERENCE) && recursive) {
               switch (param_type) {
                  case 'u':
                     // Reference to derived class can be converted to reference to base
                     // class.  Add offset, modify char* parameter and G__value* param.
                     {
                        int baseoffset = G__ispublicbase(formal_tagnum, param_tagnum, (void*) param->obj.i);
                        if (baseoffset != -1) {
                           param->tagnum = formal_tagnum;
                           param->obj.i += baseoffset;
                           param->ref += baseoffset;
                           match = 1;
                        }
                        else {
                           match = 0;
                        }
                     }
                     break;
               }
            }
            break;
#endif // G__WORKAROUND000209_1
         case 'U':
            {
               switch (param_type) {
                  case 'U':
                     {
                        // Pointer to derived class can be converted to pointer to base class.
                        // Add offset, modify char* parameter and * G__value* param.
                        int baseoffset = -1;
#ifdef G__VIRTUALBASE
                        baseoffset = G__ispublicbase(formal_tagnum, param_tagnum, (void*)param->obj.i);
#else // G__VIRTUALBASE
                        baseoffset = G__ispublicbase(formal_tagnum, param_tagnum);
#endif // G__VIRTUALBASE
                        if (baseoffset != -1) {
                           G__value_typenum(*param) = G__replace_rawtype(G__value_typenum(*param), formal_tagnum);
                           param->obj.i += baseoffset;
                           param->ref = 0;
                           match = 1;
                        }
                        else {
                           match = 0;
                        }
                     }
                     break;
                  case 'Y':
#ifndef G__OLDIMPLEMENTATION2191
                  case '1': // questionable
#else // G__OLDIMPLEMENTATION2191
                  case 'Q': // questionable
#endif // G__OLDIMPLEMENTATION2191
                     match = 1;
                     break;
                  case 'i':
                  case 0:
                     if (!param->obj.i) {
                        match = 1;
                     }
                     else {
                        match = 0;
                     }
                     break;
                  default:
                     match = 0;
                     break;
               }
            }
            break;
         default:
            // -- Questionable.
#ifndef G__OLDIMPLEMENTATION2191
            if (
               (
                  (param_type == 'Y') ||
                  (param_type == '1') ||
                  !param->obj.i
               ) &&
               (
                  isupper(formal_type) ||
                  (formal_type == 'a')
               )
            ) {
               match = 1;
            }
            else {
               match = 0;
            }
            if (
               (
                  (param_type == 'Y') ||
                  (param_type == 'Q') ||
                  !param->obj.i
               ) &&
               (
                  isupper(formal_type)
#ifndef G__OLDIMPLEMENTATION1289
                  || (formal_type == 'a')
#endif // G__OLDIMPLEMENTATION1289
               )
            ) {
               match = 1;
            }
            else {
               match = 0;
            }
#endif // G__OLDIMPLEMENTATION2191
            break;
      }
   }
   //
   //  If no match yet, try constructors and conversion operators, if allowed.
   //  
   if (!match && (funcmatch >= G__USERCONV)) {
      int rewindflag = 0;
      if (!recursive && (formal_type == 'u')) {
         if (G__struct.iscpplink[G__get_tagnum(formal_tagnum)] != G__CPPLINK) {
            // -- Create temp object buffer.
            G__alloc_tempobject(G__get_tagnum(formal_tagnum), -1);
#ifdef G__ASM
            if (G__asm_noverflow) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x: ALLOCTEMP %s %d\n", G__asm_cp, formal_tagnum.Name().c_str(), G__get_tagnum(formal_tagnum));
                  G__fprinterr(G__serr, "%3x: SETTEMP\n", G__asm_cp + 2);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
               G__asm_inst[G__asm_cp+1] = G__get_tagnum(formal_tagnum);
               G__inc_cp_asm(2, 0);
               G__asm_inst[G__asm_cp] = G__SETTEMP;
               G__inc_cp_asm(1, 0);
            }
#endif // G__ASM
            // --
         }
         //
         //  Try finding constructor.
         //
         G__StrBuf conv_sb(G__ONELINE);
         char *conv = conv_sb;
         {
            G__StrBuf tmp_sb(G__ONELINE);
            char *tmp = tmp_sb;
            if (param_type == 'u') {
               if (param->obj.i < 0) {
                  sprintf(tmp, "(%s)(%ld)", G__fulltagname(G__get_tagnum(param_tagnum), 1), param->obj.i);
               }
               else {
                  sprintf(tmp, "(%s)%ld", G__fulltagname(G__get_tagnum(param_tagnum), 1), param->obj.i);
               }
            }
            else {
               G__valuemonitor(*param, tmp);
            }
            sprintf(conv, "%s(%s)", formal_tagnum.Name().c_str(), tmp);
         }
         if (G__dispsource) {
            G__fprinterr(G__serr, "!!!Trying implicit conversion %s,%d\n", conv, G__templevel);
         }
         char* store_struct_offset = G__store_struct_offset;
         G__store_struct_offset = (char*) G__p_tempbuf->obj.obj.i;
         ::Reflex::Scope store_tagnum = G__tagnum;
         G__tagnum = formal_tagnum;
         // avoid duplicated argument evaluation in p-code stack.
         int store_oprovld = G__oprovld;
         G__oprovld = 1;
#ifdef G__ASM
         if (G__asm_noverflow && rewind_arg) {
            rewindflag = 1;
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n", G__asm_cp, rewind_arg);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__REWINDSTACK;
            G__asm_inst[G__asm_cp+1] = rewind_arg;
            G__inc_cp_asm(2, 0);
         }
#endif // G__ASM
         ++recursive;
         if (G__struct.iscpplink[G__get_tagnum(formal_tagnum)] == G__CPPLINK) {
            // -- In case of pre-compiled class.
            G__value reg = G__getfunction(conv, &match, G__TRYIMPLICITCONSTRUCTOR);
            if (match) {
               G__store_tempobject(reg);
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
               sprintf(conv, "operator %s()", formal_tagnum.Name(::Reflex::SCOPED).c_str());
               G__store_struct_offset = (char*)param->obj.i;
               G__tagnum = G__value_typenum(*param).RawType();
               if (!G__tagnum.IsTopScope()) {
                  G__getfunction(conv, &match, G__TRYMEMFUNC);
               }
               if (!match) {
                  G__store_tempobject(G__null);
               }
            }
         }
         else {
            // -- In case of interpreted class.
            G__getfunction(conv, &match, G__TRYIMPLICITCONSTRUCTOR);
            if (match) {
               if (G__asm_noverflow) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x: POPTEMP %d\n", G__asm_cp, G__get_tagnum(formal_tagnum));
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__POPTEMP;
                  G__asm_inst[G__asm_cp+1] = G__get_tagnum(formal_tagnum);
                  G__inc_cp_asm(2, 0);
               }
            }
            else {
               if (G__asm_noverflow) {
                  G__inc_cp_asm(-3, 0);
               }
               sprintf(conv, "operator %s()", formal_tagnum.Name(::Reflex::SCOPED).c_str());
               G__store_struct_offset = (char*) param->obj.i;
               G__tagnum = G__value_typenum(*param).RawType();
               G__getfunction(conv, &match, G__TRYMEMFUNC);
               if (!match) {
                  if (G__asm_noverflow) {
                     if (rewindflag) {
                        G__asm_inst[G__asm_cp-2] = G__REWINDSTACK;
                        G__asm_inst[G__asm_cp-1] = rewind_arg;
                     }
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "ALLOCTEMP,SETTEMP Cancelled %x\n", G__asm_cp);
                     }
#endif // G__ASM_DBG
                     // --
                  }
               }
            }
         }
         --recursive;
         G__oprovld = store_oprovld;
         G__tagnum = store_tagnum;
         G__store_struct_offset = store_struct_offset;
         //
         //  If no constructor, try converting to base class.
         //
         if (!match) {
            int baseoffset = -1;
#ifdef G__VIRTUALBASE
            baseoffset = G__ispublicbase(formal_tagnum, param_tagnum, (void*)param->obj.i);
#else // G__VIRTUALBASE
            baseoffset = G__ispublicbase(formal_tagnum, param_tagnum);
#endif // G__VIRTUALBASE
            if (
               (param_type == 'u') &&
               (baseoffset != -1)
            ) {
               if (G__dispsource) {
                  G__fprinterr(G__serr, "!!!Implicit conversion from %s to base %s\n", param_tagnum.Name().c_str(), formal_tagnum.Name().c_str());
               }
               G__value_typenum(*param) = formal_tagnum;
               param->obj.i += baseoffset;
               param->ref += baseoffset;
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x: BASECONV %d %d\n", G__asm_cp, G__get_tagnum(formal_tagnum), baseoffset);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__BASECONV;
                  G__asm_inst[G__asm_cp+1] = G__get_tagnum(formal_tagnum);
                  G__asm_inst[G__asm_cp+2] = baseoffset;
                  G__inc_cp_asm(3, 0);
                  if (rewind_arg) {
                     rewindflag = 1;
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n", G__asm_cp, -rewind_arg);
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                     G__asm_inst[G__asm_cp+1] = -rewind_arg;
                     G__inc_cp_asm(2, 0);
                  }
#endif // G__ASM
                  if (param->obj.i < 0) {
                     sprintf(parameter, "(%s)(%ld)", formal_tagnum.Name().c_str(), param->obj.i);
                  }
                  else {
                     sprintf(parameter, "(%s)%ld", formal_tagnum.Name().c_str(), param->obj.i);
                  }
               }
               match = 1;
               G__pop_tempobject();
            }
            else {
               // -- All conversions failed.
               if (G__dispsource) {
                  G__fprinterr(G__serr, "!!!Implicit conversion %s,%d tried, but failed\n", conv, G__templevel);
               }
               G__pop_tempobject();
#ifdef G__ASM
               if (rewindflag) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "REWINDSTACK cancelled\n");
                  }
#endif // G__ASM_DBG
                  G__inc_cp_asm(-2, 0);
               }
            }
#else // G__ASM
               // -- All conversion failed.
               if (G__dispsource) {
                  G__fprinterr(G__serr, "!!!Implicit conversion %s,%d tried, but failed\n", conv, G__templevel);
               }
               G__pop_tempobject();
#ifdef G__ASM
               if (rewindflag) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "REWINDSTACK cancelled\n");
                  }
#endif // G__ASM_DBG
                  G__inc_cp_asm(-2, 0);
               }
#endif // G__ASM
#endif // G__ASM
            // --
         }
         else {
            // -- Conversion successful.
            if (G__dispsource) {
               if (G__p_tempbuf->obj.obj.i < 0) {
                  G__fprinterr(G__serr, "!!!Create temp object (%s)(%ld),%d for implicit conversion\n", conv , G__p_tempbuf->obj.obj.i , G__templevel);
               }
               else {
                  G__fprinterr(G__serr, "!!!Create temp object (%s)%ld,%d for implicit conversion\n", conv , G__p_tempbuf->obj.obj.i , G__templevel);
               }
            }
#ifdef G__ASM
            if (G__asm_noverflow && rewind_arg) {
               rewindflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n", G__asm_cp, -rewind_arg);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__REWINDSTACK;
               G__asm_inst[G__asm_cp+1] = -rewind_arg;
               G__inc_cp_asm(2, 0);
            }
#endif // G__ASM
            *param = G__p_tempbuf->obj;
            sprintf(parameter, "(%s)%ld" , formal_tagnum.Name().c_str(), G__p_tempbuf->obj.obj.i);
         }
      }
      else if (G__get_tagnum(G__value_typenum(*param)) != -1) {
         char* store_struct_offset = G__store_struct_offset;
         ::Reflex::Scope store_tagnum = G__tagnum;
         G__StrBuf conv_sb(G__ONELINE);
         char *conv = conv_sb;
         sprintf(conv, "operator %s()", G__type2string(formal_type, G__get_tagnum(formal_tagnum), -1, 0, 0));
         G__store_struct_offset = (char*)param->obj.i;
         G__tagnum = G__value_typenum(*param).RawType();
#ifdef G__ASM
         if (G__asm_noverflow) {
            if (rewind_arg) {
               rewindflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n", G__asm_cp, rewind_arg);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__REWINDSTACK;
               G__asm_inst[G__asm_cp+1] = rewind_arg;
               G__inc_cp_asm(2, 0);
            }
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
         G__value reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
         if (!match && formal_isconst) {
            sprintf(conv, "operator const %s()", G__type2string(formal_type, G__get_tagnum(formal_tagnum), -1, 0, 0));
            G__store_struct_offset = (char*)param->obj.i;
            G__tagnum = G__value_typenum(*param).RawType();
            reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
         }
         G__tagnum = store_tagnum;
         G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
         if (G__asm_noverflow) {
            if (rewind_arg) {
               rewindflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n", G__asm_cp, -rewind_arg);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__REWINDSTACK;
               G__asm_inst[G__asm_cp+1] = -rewind_arg;
               G__inc_cp_asm(2, 0);
            }
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
         // fixing 'cout<<x' fundamental conversion opr with opr overloading
         // Not 100% sure if this is OK.
         if (match) {
            *param = reg;
         }
         else if (rewindflag) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "REWINDSTACK~ cancelled\n");
            }
#endif // G__ASM_DBG
            G__inc_cp_asm(-7, 0);
         }
         else {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "PUSHSTROS~ cancelled\n");
            }
#endif // G__ASM_DBG
            G__inc_cp_asm(-3, 0);
         }
      }
      else {
         match = 0;
         if (recursive && G__dispsource) {
            G__StrBuf tmp_sb(G__ONELINE);
            char *tmp = tmp_sb;
            G__valuemonitor(*param, tmp);
            G__fprinterr(G__serr, "!!!Recursive implicit conversion %s(%s) rejected\n", formal_tagnum.Name().c_str(), tmp);
         }
      }
   }
   //
   //  If we have a tentative match, and both the formal parameter
   //  and the given argument are pointers, then compare the pointer
   //  levels before declaring a final match.
   //
   if (
      match &&
      isupper(param_type) &&
      isupper(formal_type) &&
      (param_type != 'Y') &&
      (formal_type != 'Y')
#ifndef G__OLDIMPLEMENTATION2191
      && (param_type != '1')
#else // G__OLDIMPLEMENTATION2191
      && (param_type != 'Q')
#endif // G__OLDIMPLEMENTATION2191
      // --
   ) {
      match = G__matchpointlevel(G__get_reftype(G__value_typenum(*param)), formal_reftype);
   }
   return match;
}

//______________________________________________________________________________
#define G__NOMATCH        0xffffffff
#define G__EXACTMATCH     0x00000000
#define G__PROMOTIONMATCH 0x00000100
#define G__STDCONVMATCH   0x00010000
#define G__USRCONVMATCH   0x01000000
//#define G__CVCONVMATCH    0x00000001
#define G__CVCONVMATCH    0x00000000
#define G__BASECONVMATCH  0x00000001
#define G__C2P2FCONVMATCH 0x00000001
#define G__I02PCONVMATCH  0x00000002
#define G__V2P2FCONVMATCH 0x00000002
#define G__TOVOIDPMATCH   0x00000003

//______________________________________________________________________________
struct G__funclist* Cint::Internal::G__funclist_add(G__funclist* head, const ::Reflex::Member ifunc, int rate)
{
   // Add a new entry to a function overloading rating list at the head.
   struct G__funclist* p = (struct G__funclist*) malloc(sizeof(struct G__funclist));
   p->next = head;
   p->ifunc = ifunc;
   p->rate = rate;
   return p;
}

//______________________________________________________________________________
void Cint::Internal::G__funclist_delete(G__funclist* node)
{
   // Erase a function overloading rating list starting from a given node.
   while (node) {
      G__funclist* p = node->next;
      free(node);
      node = p;
   }
}

//______________________________________________________________________________
static unsigned int G__rate_inheritance(int basetagnum, int derivedtagnum)
{
   struct G__inheritance *derived;
   int i, n;

   if (0 > derivedtagnum || 0 > basetagnum) return(G__NOMATCH);
   if (basetagnum == derivedtagnum) return(G__EXACTMATCH);
   derived = G__struct.baseclass[derivedtagnum];
   n = derived->basen;

   for (i = 0;i < n;i++) {
      if (basetagnum == derived->basetagnum[i]) {
         if (derived->baseaccess[i] == G__PUBLIC ||
               (G__exec_memberfunc && G__tagnum == G__Dict::GetDict().GetScope(derivedtagnum) &&
                G__GRANDPRIVATE != derived->baseaccess[i])) {
            if (G__ISDIRECTINHERIT&derived->property[i]) {
               return(G__BASECONVMATCH);
            }
            else {
               int distance = 1;
               int ii = i; /* i is not 0, because !G__ISDIRECTINHERIT */
               struct G__inheritance *derived2 = derived;
               int derivedtagnum2 = derivedtagnum;
               while (0 == (derived2->property[ii]&G__ISDIRECTINHERIT)) {
                  ++distance;
                  while (ii && 0 == (derived2->property[--ii]&G__ISDIRECTINHERIT));
                  derivedtagnum2 = derived2->basetagnum[ii];
                  derived2 = G__struct.baseclass[derivedtagnum2];
                  for (ii = 0;ii < derived2->basen;ii++) {
                     if (derived2->basetagnum[ii] == basetagnum) break;
                  }
                  if (ii == derived2->basen) return(G__NOMATCH);
               }
               return(distance*G__BASECONVMATCH);
            }
         }
      }
   }
   return(G__NOMATCH);
}

//______________________________________________________________________________
static unsigned int G__rate_inheritance(const ::Reflex::Type &basetagnum, const ::Reflex::Type &derivedtagnum)
{
   return G__rate_inheritance(G__get_tagnum(basetagnum), G__get_tagnum(derivedtagnum));
}

#ifndef G__OLDIMPLEMENTATION1959
//______________________________________________________________________________
#define G__promotiongrade(f,p) G__PROMOTIONMATCH*(G__igrd(f)-G__igrd(p))

//______________________________________________________________________________
static int G__igrd(int formal_type)
{

   // this looks good but is actually nowhere to be found in the standard;
   // it's not how conversions are to be ranked.
   switch (formal_type) {
      case 'g':
         return(1);
      case 'b':
      case 'c':
         return(2);
      case 'r':
      case 's':
         return(3);
      case 'h':
      case 'i':
         return(4);
      case 'k':
      case 'l':
         return(5);
      case 'n':
      case 'm':
         return(6);
   }
   return(0);
}
#endif

//______________________________________________________________________________
#ifndef __CINT__
static ::Reflex::Member G__overload_match(char* funcname, G__param* libp, int hash, const ::Reflex::Scope p_ifunc, int memfunc_flag, int access, int isrecursive, int doconvert, int* match_error);
#endif // __CINT__

static void G__display_param(FILE* fp, const ::Reflex::Scope scopetagnum, char* funcname, G__param* libp);
static void G__display_func(FILE* fp, const ::Reflex::Member func);

//______________________________________________________________________________
static void G__rate_parameter_match(G__param* libp, const ::Reflex::Member func, G__funclist* funclist, int recursive)
{
   // Rate the conversion sequence for mapping the argument types to the formal parameter types for each function in funclist.
   //static int depth = -1;
   //++depth;
#ifdef G__DEBUG
   int i = 0xa3a3a3a3;
#else // G__DEBUG
   int i = 0;
#endif // G__DEBUG
   char arg_type = '\0';
   char formal_type = '\0';
   ::Reflex::Type arg_tagnum;
   ::Reflex::Type formal_tagnum;
   ::Reflex::Type arg_final;
   ::Reflex::Type formal_final;
   int arg_reftype = 0;
   int formal_reftype = 0;
#ifdef G__DEBUG
   int arg_isconst = 0xa3a3a3a3;
   int formal_isconst = 0xa5a5a5a5;
#else // G__DEBUG
   int arg_isconst = 0;
   int formal_isconst = 0;
#endif // G__DEBUG
   //fprintf(stderr, "\nG__rate_parameter_match: Calling: %d ", depth);
   //G__display_param(G__serr, func.TypeOf().DeclaringScope(), (char*) func.Name().c_str(), libp);
   //G__display_param(G__serr, ::Reflex::Scope::GlobalScope(), (char*) func.Name(::Reflex::SCOPED | ::Reflex::QUALIFIED).c_str(), libp);
   funclist->rate = 0;
   for (i = 0; i < libp->paran; ++i) {
      formal_tagnum = func.TypeOf().FunctionParameterAt(i);
      arg_tagnum = G__value_typenum(libp->para[i]);
      formal_final = formal_tagnum.FinalType();
      arg_final = arg_tagnum.FinalType();
      arg_type = G__get_type(arg_tagnum);
      formal_type = G__get_type(formal_tagnum);
      arg_isconst = G__get_isconst(arg_tagnum);
      formal_isconst = G__get_isconst(formal_tagnum);
      arg_reftype = G__get_reftype(arg_final);
      formal_reftype = G__get_reftype(formal_final);
      arg_tagnum = arg_tagnum.RawType();
      formal_tagnum = formal_tagnum.RawType();
      funclist->p_rate[i] = G__NOMATCH;
      //
      //  Exact Match.
      //
      //fprintf(stderr, "G__rate_parameter_match: %d Checking for exact match.\n", depth);
      if (arg_type == formal_type) {
         if (tolower(arg_type) == 'u') { // class, struct, union, or pointer to them, check rawtype
            if (arg_tagnum == formal_tagnum) {
               funclist->p_rate[i] = G__EXACTMATCH;
            }
         }
         else if (isupper(arg_type)) { // two pointers to fundamental type
            if ( // if pointer count and refness are compatible, then exact match
               (arg_reftype == formal_reftype) || // pointer count is the same, and refness is the same, or
               (
                  (arg_reftype <= G__PARAREFERENCE) && // arg is a pointer, or a ref to a pointer, and
                  (formal_reftype <= G__PARAREFERENCE) // formal param is a pointer, or a ref to a pointer
               )
            ) {
               funclist->p_rate[i] = G__EXACTMATCH;
            }
            else if (
               (
                  (arg_reftype > G__PARAREF) &&
                  (arg_reftype == (formal_reftype + G__PARAREF))
               ) ||
               (
                  (formal_reftype > G__PARAREF) &&
                  (formal_reftype == (arg_reftype + G__PARAREF))
               )
            ) {
               funclist->p_rate[i] = G__STDCONVMATCH;
            }
         }
         else if ((arg_type == 'i') && (arg_tagnum != formal_tagnum)) { // This is for enums.
            funclist->p_rate[i] = G__PROMOTIONMATCH;
         }
         else {
            funclist->p_rate[i] = G__EXACTMATCH;
         }
      }
      else if ( // special hack for pointer to enum
         (
            (arg_type == 'I') || // arg is int*, or
            (arg_type == 'U') // arg is class MyClass*,
         ) && // and,
         (
            (formal_type == 'I') || // formal func param is int*, or
            (formal_type == 'U') // formal func param is MyClass*,
         ) && // and,
         (arg_tagnum == formal_tagnum) && // arg and formal param have same tag, and
         (G__get_tagnum(formal_tagnum) != -1) && // formal param is of class, enum, struct, or union type, and
         (G__get_tagtype(formal_tagnum) == 'e') // formal param is of enum type
      ) {
         funclist->p_rate[i] = G__EXACTMATCH;
      }
      //
      //  Promotion.
      //
      if (funclist->p_rate[i] == G__NOMATCH) {
         //fprintf(stderr, "G__rate_parameter_match: %d Checking for promotion.\n", depth);
         switch (formal_type) {
         case 'd': /* 4.6: conv.fpprom */
            switch (arg_type) {
            case 'f':
               funclist->p_rate[i] = G__PROMOTIONMATCH;
               break;
            default:
               break;
            }
            break;
         case 'i': /* 4.5: conv.prom */
         case 'h': /* 4.5: conv.prom */
            switch (arg_type) {
            case 'b':
            case 'c':
            case 'r':
            case 's':
            case 'g':
               funclist->p_rate[i] = G__promotiongrade(formal_type, arg_type);
               break;
            case 'u':
               if ('e' == G__get_tagtype(arg_tagnum)) {
                  funclist->p_rate[i] = G__PROMOTIONMATCH;
               }
               break;
            default:
               break;
            }
            break;
         case 'l':
         case 'k': /* only enums get promoted to (u)long! */
            if (arg_type == 'u' && 'e' == G__get_tagtype(arg_tagnum)) {
                  funclist->p_rate[i] = G__PROMOTIONMATCH;
            }
            break;
         case 'Y':
            if (
               isupper(arg_type) ||
               0 == libp->para[i].obj.i
               // --
#ifndef G__OLDIMPLEMENTATION2191
                || '1' == arg_type
#endif // G__OLDIMPLEMENTATION2191
             ) {
               funclist->p_rate[i] = G__PROMOTIONMATCH + G__TOVOIDPMATCH;
            }
            break;
         default:
            break;
         }
      }
      //
      //  Standard Conversion.
      //
      if (funclist->p_rate[i] == G__NOMATCH) {
         //fprintf(stderr, "G__rate_parameter_match: %d Checking for a conversion.\n", depth);
         if ( // Check for integral const zero conversion to pointer
            isupper(formal_type) && // formal func param is a pointer, and
            (
               (arg_type == 'i') || // arg is int, or
               (arg_type == 'l') // arg is long
               // FIXME: we need char, short, unsigned char, unsigned short, unsigned int, unsigned long
            ) &&
            !libp->para[i].obj.i && // arg value is zero
            arg_isconst // arg is const
         ) {
            //fprintf(stderr, "G__rate_parameter_match: integral const zero passed to pointer seen for function '%s'.\n", func.Name(::Reflex::SCOPED).c_str());
            //funclist->p_rate[i] = G__STDCONVMATCH + G__I02PCONVMATCH;
            funclist->p_rate[i] = G__EXACTMATCH;
         }
         if (funclist->p_rate[i] == G__NOMATCH) {
            switch (formal_type) {
               case 'b': // unsigned char
               case 'c': // char
               case 'r': // unsigned short
               case 's': // short
               case 'h': // unsigned int
               case 'i': // int
               case 'k': // unsigned long
               case 'l': // long
               case 'g': // bool
               case 'n': // unsigned long long
               case 'm': // long long
               case 'd': // double
               case 'f': // float
                  switch (arg_type) {
                     case 'd':
                     case 'f':
                     case 'b':
                     case 'c':
                     case 'r':
                     case 's':
                     case 'h':
                     case 'i':
                     case 'k':
                     case 'l':
                     case 'g':
                     case 'n':
                     case 'm':
                     case 'q':
                        funclist->p_rate[i] = G__STDCONVMATCH;
                        break;
                     case 'u':
                        if (G__get_tagtype(arg_tagnum) == 'e') {
                           funclist->p_rate[i] = G__PROMOTIONMATCH;
                        }
                        break;
                     default:
                        break;
                  }
               case 'C': // char*
                  if (arg_type == 'Y') { // void* passed to char* FIXME: This is illegal! char* passed to void* is ok!
                     if (arg_reftype == G__PARANORMAL) {
                        funclist->p_rate[i] = G__STDCONVMATCH;
                     }
                  }
                  break;
               case 'Y': // void*
                  if (isupper(arg_type) || !libp->para[i].obj.i) {
                     funclist->p_rate[i] = G__STDCONVMATCH;
                  }
                  break;
#ifndef G__OLDIMPLEMENTATION2191
               case '1':
#else // G__OLDIMPLEMENTATION2191
               case 'Q':
#endif // G__OLDIMPLEMENTATION2191
                  // --
                  if (
                      // --
#ifndef G__OLDIMPLEMENTATION2191
                     '1' == arg_type
#else // G__OLDIMPLEMENTATION2191
                     'Q' == arg_type
#endif // G__OLDIMPLEMENTATION2191
                     // --
                  ) {
                     funclist->p_rate[i] = G__STDCONVMATCH;
                  }
                  else if ('Y' == arg_type) {
                     funclist->p_rate[i] = G__STDCONVMATCH + G__V2P2FCONVMATCH;
                  }
                  else if ('C' == arg_type) {
                     if (G__get_funcproperties(func)->entry.size >= 0) {
                        funclist->p_rate[i] = G__STDCONVMATCH - G__C2P2FCONVMATCH;
                     }
                     else {
                        funclist->p_rate[i] = G__STDCONVMATCH + G__C2P2FCONVMATCH;
                     }
                  }
                  break;
               case 'u': // MyClass
                  switch (arg_type) {
                     case 'u':
                        {
                           // reference to derived class can be converted to reference to base
                           // class. add offset, modify char* parameter and G__value* param
                           unsigned int rate_inheritance = G__rate_inheritance(formal_tagnum, arg_tagnum);
                           if (G__NOMATCH != rate_inheritance) {
                              funclist->p_rate[i] = G__STDCONVMATCH + rate_inheritance;
                           }
                        }
                        break;
                  }
                  break;
               case 'U': // MyClass*
                  switch (arg_type) {
                     case 'U':
                        {
                           // Pointer to derived class can be converted to
                           // pointer to base class.  Add offset, modify
                           // char *parameter and G__value* param.
                           unsigned int rate_inheritance = G__rate_inheritance(formal_tagnum, arg_tagnum);
                           if (G__NOMATCH != rate_inheritance) {
                              funclist->p_rate[i] = G__STDCONVMATCH + rate_inheritance;
                           }
                        }
                        break;
                     case 'Y':
                        if (G__PARANORMAL == arg_reftype) {
                           funclist->p_rate[i] = G__STDCONVMATCH;
                        }
                        break;
#ifndef G__OLDIMPLEMENTATION2191
                     case '1':
#else // G__OLDIMPLEMENTATION2191
                     case 'Q':
#endif // G__OLDIMPLEMENTATION2191
                        funclist->p_rate[i] = G__STDCONVMATCH;
                        break;
                     case 'i':
                     case 0:
                        if (0 == libp->para[0].obj.i) {
                           funclist->p_rate[i] = G__STDCONVMATCH;
                        }
                        break;
                     default:
                        break;
                  }
                  break;
               default:
                  // --
#ifndef G__OLDIMPLEMENTATION2191
                  if ((arg_type == 'Y' || arg_type == '1') && (isupper(formal_type) || 'a' == formal_type)) {
                     funclist->p_rate[i] = G__STDCONVMATCH;
                  }
#else // G__OLDIMPLEMENTATION2191
                  if ((arg_type == 'Y' || arg_type == 'Q' || 0 == libp->para[0].obj.i) &&
                        (isupper(formal_type) || 'a' == formal_type)) {
                     funclist->p_rate[i] = G__STDCONVMATCH;
                  }
#endif // G__OLDIMPLEMENTATION2191
                  break;
            }
         }
      }
      //
      //  User-Defined Conversion.
      //
      if (!recursive && (funclist->p_rate[i] == G__NOMATCH) && (formal_type == 'u')) { // Try a constructor.
         //fprintf(stderr, "G__rate_parameter_match: %d Checking for a user-defined conversion by constructor.\n", depth);
         G__incsetup_memfunc(formal_tagnum);
         struct G__param para;
         para.paran = 1;
         para.para[0] = libp->para[i];
         G__StrBuf funcname2_sb(G__ONELINE);
         char* funcname2 = funcname2_sb;
         strcpy(funcname2, formal_tagnum.Name().c_str());
         int hash2 = 0;
         int ifn2 = 0;
         G__hash(funcname2, hash2, ifn2);
         int match_error = 0;
         ::Reflex::Member func2 = G__overload_match(funcname2, &para, hash2, formal_tagnum, G__TRYCONSTRUCTOR, G__PUBLIC, 1, 1, &match_error);
         if (func2) {
            funclist->p_rate[i] = G__USRCONVMATCH;
         }
      }
      if (!recursive && (funclist->p_rate[i] == G__NOMATCH) && (arg_type == 'u') && (G__get_tagnum(arg_tagnum) != -1)) { // Try a type conversion operator function.
         //fprintf(stderr, "G__rate_parameter_match: %d Checking for a user-defined conversion by operator function.\n", depth);
         G__incsetup_memfunc(arg_tagnum);
         struct G__param para;
         para.paran = 0;
         // search for  operator type
         G__StrBuf funcname2_sb(G__ONELINE);
         char* funcname2 = funcname2_sb;
         sprintf(funcname2, "operator %s", G__type2string(formal_type, G__get_tagnum(formal_tagnum), -1, 0, 0));
         int hash2 = 0;
         int ifn2 = 0;
         G__hash(funcname2, hash2, ifn2);
         int match_error;
         ::Reflex::Member ifunc2 = G__overload_match(funcname2, &para, hash2, arg_tagnum, G__TRYMEMFUNC, G__PUBLIC, 1, 1, &match_error);
         if (!ifunc2) {
            // search for  operator const type
            sprintf(funcname2, "operator %s", G__type2string(formal_type, G__get_tagnum(formal_tagnum), -1, 0, 1));
            G__hash(funcname2, hash2, ifn2);
            ifunc2 = G__overload_match(funcname2, &para, hash2, arg_tagnum, G__TRYMEMFUNC, G__PUBLIC, 1, 1, &match_error);
         }
         if (ifunc2) {
            funclist->p_rate[i] = G__USRCONVMATCH;
         }
      }
      //
      //  Notice a const/volatile conversion (this should rank Exact Match)
      //
      //  TODO: This is unnecessary and should be removed.
      //
      if (arg_isconst != formal_isconst) { // notice const/volatile conversion
         funclist->p_rate[i] += G__CVCONVMATCH; // FIXME: Remove this!  This currently does nothing!  And it should not!
      }
      //
      //  Check for const passed to non-const ref.
      //
      if (arg_isconst && !formal_isconst && formal_final.IsReference()) { // const passed to non-const ref is bad
         //fprintf(stderr, "G__rate_parameter_match: %d No match, const passed to non-const ref.\n", depth);
         funclist->p_rate[i] = G__NOMATCH;
      }
      //fprintf(stderr, "G__rate_parameter_match: %d rate: %08X  function ", depth, funclist->p_rate[i]);
      //fprintf(stderr, "%s ", funclist->ifunc.TypeOf().ReturnType().Name(::Reflex::SCOPED |::Reflex::QUALIFIED).c_str());
      //if (!funclist->ifunc.DeclaringScope().IsTopScope()) {
      //   fprintf(stderr, "%s::", funclist->ifunc.DeclaringScope().Name(::Reflex::SCOPED).c_str());
      //}
      //fprintf(stderr, "%s(", funclist->ifunc.Name().c_str());
      //for (int j = 0; j < funclist->ifunc.TypeOf().FunctionParameterSize(); ++j) {
      //   fprintf(stderr, "%s", funclist->ifunc.TypeOf().FunctionParameterAt(i).Name(::Reflex::SCOPED |::Reflex::QUALIFIED).c_str());
      //   if (j != (funclist->ifunc.TypeOf().FunctionParameterSize() - 1)) {
      //      fprintf(stderr, ",");
      //   }
      //}
      //fprintf(stderr, ");\n");
      //
      //  Stop scanning if a param did not match, function cannot match then.
      //
      if (funclist->p_rate[i] == G__NOMATCH) { // Stop scanning, if no param match, func cannot match.
         funclist->rate = G__NOMATCH;
         break; // Stop scanning, function cannot be a match.
      }
      //
      //  Update the function rank by adding in the parameter rating.
      //
      if (funclist->rate != G__NOMATCH) {
         funclist->rate += funclist->p_rate[i];
      }
   }
   // Now adjust the rating based on the constness of the implied object parameter.
   if (
      (funclist->rate != G__NOMATCH) &&
      (
         (
            !G__isconst && // invoker is not const, and
            func.TypeOf().IsConst() // function is const
         ) || // or,
         (
            G__isconst && // invoker is const, and
            !func.TypeOf().IsConst() // function is not const
         )
      )
   ) {
      funclist->rate += G__CVCONVMATCH; // Notice const/volatile conversion of implied object parameter.
   }
   //fprintf(stderr, "G__rate_parameter_match: %d final rate: %08X\n", depth, funclist->rate);
   //--depth;
}

//______________________________________________________________________________
static int G__convert_param(G__param* libp, const ::Reflex::Member& func, G__funclist* pmatch)
{
   int i;
   unsigned int rate;
   char param_type, formal_type;
   ::Reflex::Type param_tagnum, formal_tagnum;
   int formal_reftype;
   int formal_isconst;
   G__value *param;
#ifdef G__OLDIMPLEMENTATION2195_YET
   int store_asm_cp = G__asm_cp;
#endif
   char conv[G__ONELINE], arg1[G__ONELINE], parameter[G__ONELINE];
   char *store_struct_offset; /* used to be int */
   ::Reflex::Scope store_tagnum;
   int store_isconst;
   int baseoffset;
   G__value reg;
   int store_oprovld;
   int rewindflag = 0;
   int recursive = 0;
   int rewind_arg;
   int match = 0;

   for (i = 0;i < libp->paran;i++) {
      rate = pmatch->p_rate[i];
      param_tagnum = G__value_typenum(libp->para[i]);
      formal_tagnum = func.TypeOf().FunctionParameterAt(i);
      param_type = G__get_type(param_tagnum);
      formal_type = G__get_type(formal_tagnum);
      formal_reftype = G__get_reftype(formal_tagnum);
      formal_isconst = G__get_isconst(formal_tagnum);
      param_tagnum = param_tagnum.RawType();
      formal_tagnum = formal_tagnum.RawType();
      param = &libp->para[i];
#ifndef G__OLDIMPLEMENTATION
      rewind_arg = libp->paran - i - 1;
#else
      rewind_arg = p_ifunc->para_nu[ifn] - i - 1;
#endif
      if (rate&G__USRCONVMATCH) {
         if (formal_type == 'u') {
            /* create temp object buffer */
            if (G__CPPLINK != G__struct.iscpplink[G__get_tagnum(formal_tagnum)]) {
               G__alloc_tempobject(G__get_tagnum(formal_tagnum), -1);
#ifdef G__ASM
               if (G__asm_noverflow) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x: ALLOCTEMP %s %d\n"
                                  , G__asm_cp, formal_tagnum.Name().c_str(), G__get_tagnum(formal_tagnum));
                     G__fprinterr(G__serr, "%3x: SETTEMP\n", G__asm_cp + 2);
                  }
#endif
                  G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
                  G__asm_inst[G__asm_cp+1] = G__get_tagnum(formal_tagnum);
                  G__inc_cp_asm(2, 0);
                  G__asm_inst[G__asm_cp] = G__SETTEMP;
                  G__inc_cp_asm(1, 0);
               }
#endif
            }

            /* try finding constructor */
            if ('u' == param_type) {
               if (param->obj.i < 0)
                  sprintf(arg1, "(%s)(%ld)"
                          , G__fulltagname(G__get_tagnum(param_tagnum), 1), param->obj.i);
               else
                  sprintf(arg1, "(%s)%ld", G__fulltagname(G__get_tagnum(param_tagnum), 1), param->obj.i);
            }
            else {
               G__valuemonitor(*param, arg1);
            }
            sprintf(conv, "%s(%s)", G__struct.name[G__get_tagnum(formal_tagnum)], arg1);

            if (G__dispsource) {
               G__fprinterr(G__serr, "!!!Trying implicit conversion %s,%d\n"
                            , conv, G__templevel);
            }

            store_struct_offset = G__store_struct_offset;
            G__store_struct_offset = (char*)G__p_tempbuf->obj.obj.i;

            store_tagnum = G__tagnum;
            G__tagnum = formal_tagnum;
            store_isconst = G__isconst;
            G__isconst = formal_isconst;

            /* avoid duplicated argument evaluation in p-code stack */
            store_oprovld = G__oprovld;
            G__oprovld = 1;

#ifdef G__ASM
            if (G__asm_noverflow && rewind_arg) {
               rewindflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                               , G__asm_cp, rewind_arg);
#endif
               G__asm_inst[G__asm_cp] = G__REWINDSTACK;
               G__asm_inst[G__asm_cp+1] = rewind_arg;
               G__inc_cp_asm(2, 0);
            }
#endif

            ++recursive;
            if (G__CPPLINK == G__struct.iscpplink[G__get_tagnum(formal_tagnum)]) {
               /* in case of pre-compiled class */
               reg = G__getfunction(conv, &match, G__TRYIMPLICITCONSTRUCTOR);
               if (match) {
                  G__store_tempobject(reg);
#ifdef G__ASM
                  if (G__asm_noverflow) {
#ifdef G__ASM_DBG
                     if (G__asm_dbg) G__fprinterr(G__serr, "%3x: STORETEMP\n", G__asm_cp);
#endif
                     G__asm_inst[G__asm_cp] = G__STORETEMP;
                     G__inc_cp_asm(1, 0);
                  }
#endif
               }
               else {
                  G__pop_tempobject();
                  sprintf(conv, "operator %s()", G__fulltagname(G__get_tagnum(formal_tagnum), 1));
                  G__store_struct_offset = (char*)param->obj.i;
                  G__tagnum = G__value_typenum(*param).RawType();
                  if (G__tagnum) reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
                  if (!match) G__store_tempobject(G__null);
               }
            }
            else {
               /* in case of interpreted class */
               G__getfunction(conv, &match, G__TRYIMPLICITCONSTRUCTOR);
               if (match) {
                  if (G__asm_noverflow) {
#ifdef G__ASM_DBG
                     if (G__asm_dbg) G__fprinterr(G__serr, "%3x: POPTEMP %d\n"
                                                     , G__asm_cp, G__get_tagnum(formal_tagnum));
#endif
                     G__asm_inst[G__asm_cp] = G__POPTEMP;
                     G__asm_inst[G__asm_cp+1] = G__get_tagnum(formal_tagnum);
                     G__inc_cp_asm(2, 0);
                  }
               }
               else {
                  G__pop_tempobject();
                  if (G__asm_noverflow) G__inc_cp_asm(-3, 0);
                  sprintf(conv, "operator %s()", G__fulltagname(G__get_tagnum(formal_tagnum), 1));
                  G__store_struct_offset = (char*)param->obj.i;
                  G__tagnum = G__value_typenum(*param).RawType();
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
                  reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
                  if (!match) {
                     if (G__asm_noverflow) {
                        G__inc_cp_asm(-2, 0);
                        if (rewindflag) {
                           G__asm_inst[G__asm_cp-2] = G__REWINDSTACK;
                           G__asm_inst[G__asm_cp-1] = rewind_arg;
                        }
#ifdef G__ASM_DBG
                        if (G__asm_dbg)
                           G__fprinterr(G__serr, "ALLOCTEMP,SETTEMP Cancelled %x\n", G__asm_cp);
#endif
                     }
                  }
#ifdef G__ASM
                  else if (G__asm_noverflow) {
                     G__asm_inst[G__asm_cp] = G__POPSTROS;
                     G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
                     if (G__asm_dbg) G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp - 1);
#endif
                  }
#endif
               }
            }
            --recursive;

            G__oprovld = store_oprovld;

            G__isconst = store_isconst;
            G__tagnum = store_tagnum;
            G__store_struct_offset = store_struct_offset;

            /* if no constructor, try converting to base class */


            if (match == 0) {
               if (
                  'u' == param_type &&
#ifdef G__VIRTUALBASE
                  - 1 != (baseoffset = G__ispublicbase(G__get_tagnum(formal_tagnum), G__get_tagnum(param_tagnum), (void*)param->obj.i))
#else
                  - 1 != (baseoffset = G__ispublicbase(formal_tagnum, param_tagnum))
#endif
               ) {
                  if (G__dispsource) {
                     G__fprinterr(G__serr, "!!!Implicit conversion from %s to base %s\n"
                                  , G__struct.name[G__get_tagnum(param_tagnum)]
                                  , G__struct.name[G__get_tagnum(formal_tagnum)]);
                  }
                  G__value_typenum(*param) = formal_tagnum;
                  param->obj.i += baseoffset;
                  param->ref += baseoffset;
#ifdef G__ASM
                  if (G__asm_noverflow) {
#ifdef G__ASM_DBG
                     if (G__asm_dbg) G__fprinterr(G__serr, "%3x: BASECONV %d %d\n"
                                                     , G__asm_cp, G__get_tagnum(formal_tagnum), baseoffset);
#endif
                     G__asm_inst[G__asm_cp] = G__BASECONV;
                     G__asm_inst[G__asm_cp+1] = G__get_tagnum(formal_tagnum);
                     G__asm_inst[G__asm_cp+2] = baseoffset;
                     G__inc_cp_asm(3, 0);
                     if (rewind_arg) {
                        rewindflag = 1;
#ifdef G__ASM_DBG
                        if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                        , G__asm_cp, -rewind_arg);
#endif
                        G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                        G__asm_inst[G__asm_cp+1] = -rewind_arg;
                        G__inc_cp_asm(2, 0);
                     }
#endif
                     if (param->obj.i < 0)
                        sprintf(parameter, "(%s)(%ld)", G__struct.name[G__get_tagnum(formal_tagnum)]
                                , param->obj.i);
                     else
                        sprintf(parameter, "(%s)%ld", G__struct.name[G__get_tagnum(formal_tagnum)]
                                , param->obj.i);
                  }
                  match = 1;
                  G__pop_tempobject();
               }
               else { /* all conversion failed */
                  if (G__dispsource) {
                     G__fprinterr(G__serr,
                                  "!!!Implicit conversion %s,%d tried, but failed\n"
                                  , conv, G__templevel);
                  }
                  G__pop_tempobject();
#ifdef G__ASM
                  if (rewindflag) {
#ifdef G__ASM_DBG
                     if (G__asm_dbg) G__fprinterr(G__serr, "REWINDSTACK cancelled\n");
#endif
                     G__inc_cp_asm(-2, 0);
                  }
               }

#else /* ON181 */

                  /* all conversion failed */
                  if (G__dispsource) {
                     G__fprinterr(G__serr,
                                  "!!!Implicit conversion %s,%d tried, but failed\n"
                                  , conv, G__templevel);
                  }
                  G__pop_tempobject();
#ifdef G__ASM
                  if (rewindflag) {
#ifdef G__ASM_DBG
                     if (G__asm_dbg) G__fprinterr(G__serr, "REWINDSTACK cancelled\n");
#endif
                     G__inc_cp_asm(-2, 0);
                  }
#endif
#endif
            }
            else {
               /* match==1, conversion successful */
               if (G__dispsource) {
                  if (G__p_tempbuf->obj.obj.i < 0)
                     G__fprinterr(G__serr,
                                  "!!!Create temp object (%s)(%ld),%d for implicit conversion\n"
                                  , conv , G__p_tempbuf->obj.obj.i , G__templevel);
                  else
                     G__fprinterr(G__serr,
                                  "!!!Create temp object (%s)%ld,%d for implicit conversion\n"
                                  , conv , G__p_tempbuf->obj.obj.i , G__templevel);
               }
#ifdef G__ASM
               if (G__asm_noverflow && rewind_arg) {
                  rewindflag = 1;
#ifdef G__ASM_DBG
                  if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                  , G__asm_cp, -rewind_arg);
#endif
                  G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                  G__asm_inst[G__asm_cp+1] = -rewind_arg;
                  G__inc_cp_asm(2, 0);
               }
#endif
               *param = G__p_tempbuf->obj;
               sprintf(parameter, "(%s)%ld" , G__struct.name[G__get_tagnum(formal_tagnum)]
                       , G__p_tempbuf->obj.obj.i);
            }
         }
         else if (-1 != G__get_tagnum(G__value_typenum(*param))) {
            char *store_struct_offset = G__store_struct_offset;
            ::Reflex::Scope store_tagnum = G__tagnum;
            int store_isconst = G__isconst;
            int intTagnum = G__get_tagnum(formal_tagnum);
            if (intTagnum == 0) intTagnum = -1;
            sprintf(conv, "operator %s()", G__type2string(formal_type, intTagnum, -1, 0, 0));
            G__store_struct_offset = (char*)param->obj.i;
            G__tagnum = G__value_typenum(*param).RawType();
#ifdef G__ASM
            if (G__asm_noverflow) {
               if (rewind_arg) {
                  rewindflag = 1;
#ifdef G__ASM_DBG
                  if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n", G__asm_cp, rewind_arg);
#endif
                  G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                  G__asm_inst[G__asm_cp+1] = rewind_arg;
                  G__inc_cp_asm(2, 0);
               }
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
            reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
            if (!match && 0 != formal_isconst) {
               int intTagnum = G__get_tagnum(formal_tagnum);
               if (intTagnum == 0) intTagnum = -1;
               sprintf(conv, "operator const %s()", G__type2string(formal_type, intTagnum, -1, 0, 0));
               G__store_struct_offset = (char*)param->obj.i;
               G__tagnum = G__value_typenum(*param).RawType();
               reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
            }
            G__isconst = store_isconst;
            G__tagnum = store_tagnum;
            G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
            if (G__asm_noverflow) {
               if (rewind_arg) {
                  rewindflag = 1;
#ifdef G__ASM_DBG
                  if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n", G__asm_cp, -rewind_arg);
#endif
                  G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                  G__asm_inst[G__asm_cp+1] = -rewind_arg;
                  G__inc_cp_asm(2, 0);
               }
               G__asm_inst[G__asm_cp] = G__POPSTROS;
               G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp - 1);
#endif
            }
#endif
            /* fixing 'cout<<x' fundamental conversion opr with opr overloading
             * Not 100% sure if this is OK. */
            if (match) *param = reg;
            else if (rewindflag) {
#ifdef G__ASM_DBG
               if (G__asm_dbg) G__fprinterr(G__serr, "REWINDSTACK~ cancelled\n");
#endif
               G__inc_cp_asm(-7, 0);
            }
            else {
#ifdef G__ASM_DBG
               if (G__asm_dbg) G__fprinterr(G__serr, "PUSHSTROS~ cancelled\n");
#endif
               G__inc_cp_asm(-3, 0);
            }
         }
         else {
            match = 0;
            if (recursive && G__dispsource) {
               G__valuemonitor(*param, arg1);
               G__fprinterr(G__serr, "!!!Recursive implicit conversion %s(%s) rejected\n", G__struct.name[G__get_tagnum(formal_tagnum)], arg1);
            }
         }
         continue;
      }
      switch (formal_type) {
         case 'b':
         case 'c':
         case 'r':
         case 's':
         case 'h':
         case 'i':
         case 'k':
         case 'l':
            switch (param_type) {
               case 'd':
               case 'f':
                  /* std conv */
                  if (G__PARAREFERENCE == formal_reftype) {
                     param->obj.i = (long)param->obj.d;
                     G__replace_rawtype(G__value_typenum(*param), G__get_from_type(formal_type, 1));
                     param->ref = 0;
                  }
                  break;
            }
            break;
         case 'g':
            switch (param_type) {
               case 'd':
               case 'f':
                  /* std conv */
                  if (G__PARAREFERENCE == formal_reftype) {
                     param->obj.i = param->obj.d ? 1 : 0;
                     G__replace_rawtype(G__value_typenum(*param), G__get_from_type(formal_type, 1)); // param->type = formal_type;
                     param->ref = 0;
                  }
               case 'l':
               case 'i':
               case 's':
               case 'c':
               case 'h':
               case 'k':
               case 'r':
               case 'b':
                  if (G__PARAREFERENCE == formal_reftype) {
                     param->obj.i = (long)param->obj.i ? 1 : 0;
                     G__replace_rawtype(G__value_typenum(*param), G__get_from_type(formal_type, 1)); // param->type = formal_type;
                     param->ref = 0;
                  }
                  break;
            }
            break;
         case 'n': /* long long */
            if (G__PARAREFERENCE == formal_reftype) {
               G__replace_rawtype(G__value_typenum(*param), G__get_from_type(formal_type, 1)); // param->type = formal_type;
               //if(param->type!=formal_type) param->ref = 0;
               switch (param_type) {
                  case 'd':
                  case 'f':
                     param->obj.ll = (G__int64)param->obj.d;
                     break;
                  case 'g':
                  case 'c':
                  case 's':
                  case 'i':
                  case 'l':
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k':
                     param->obj.ll = param->obj.i;
                     break;
                  case 'm':
                     param->obj.ll = param->obj.ull;
                     break;
                  case 'q':
                     param->obj.ll = (G__int64)param->obj.ld;
                     break;
               }
            }
            break;
         case 'm': /* unsigned long long */
            if (G__PARAREFERENCE == formal_reftype) {
               G__replace_rawtype(G__value_typenum(*param), G__get_from_type(formal_type, 1)); // param->type = formal_type;param->type = formal_type;
               //if(param->type!=formal_type) param->ref = 0;
               switch (param_type) {
                  case 'd':
                  case 'f':
                     param->obj.ull = (G__uint64)param->obj.d;
                     break;
                  case 'g':
                  case 'c':
                  case 's':
                  case 'i':
                  case 'l':
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k':
                     param->obj.ull = param->obj.i;
                     break;
                  case 'n':
                     param->obj.ull = param->obj.ll;
                     break;
                  case 'q':
                     param->obj.ull = (G__int64)param->obj.ld;
                     break;
               }
            }
            break;
         case 'q': /* long double */
            if (G__PARAREFERENCE == formal_reftype) {
               G__replace_rawtype(G__value_typenum(*param), G__get_from_type(formal_type, 1)); // param->type = formal_type;
               //if(param->type!=formal_type) param->ref = 0;
               switch (param_type) {
                  case 'd':
                  case 'f':
                     param->obj.ld = param->obj.d;
                     break;
                  case 'g':
                  case 'c':
                  case 's':
                  case 'i':
                  case 'l':
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k':
                     param->obj.ld = param->obj.i;
                     break;
                  case 'n':
                     param->obj.ld = (long double)param->obj.ll;
                     break;
                  case 'm':
                     param->obj.ld = (long double)param->obj.ld;
                     break;
               }
            }
            break;
         case 'd':
         case 'f':
            switch (param_type) {
               case 'b':
               case 'c':
               case 'r':
               case 's':
               case 'h':
               case 'i':
               case 'k':
               case 'l':
               case 'g':
               case 'n':
               case 'm':
                  /* std conv */
                  if (G__PARAREFERENCE == formal_reftype) {
                     param->obj.d = param->obj.i;
                     G__replace_rawtype(G__value_typenum(*param), G__get_from_type(formal_type, 1)); // param->type = formal_type;
                     param->ref = 0;
                  }
                  break;
            }
            break;
         case 'u':
            switch (param_type) {
               case 'u':
                  if (0 == (rate&0xffffff00)) {
                     /* exact */
                     if ('e' == G__struct.type[G__get_tagnum(param_tagnum)]) {
                        if (param->ref) param->obj.i = *(long*)(param->ref);
                     }
                  }
                  else /* if(G__PARAREFERENCE==formal_reftype) */ {
                     if (-1 != (baseoffset = G__ispublicbase(G__get_tagnum(formal_tagnum), G__get_tagnum(param_tagnum)
                                                             , (void*)param->obj.i))) {
                        G__replace_rawtype(G__value_typenum(*param), formal_tagnum); // param->tagnum = G__get_tagnum(formal_tagnum);
                        param->obj.i += baseoffset;
                        param->ref = param->obj.i;
#ifdef G__ASM
                        if (G__asm_noverflow) {
                           if (rewind_arg
                              ) {
#ifdef G__ASM_DBG
                              if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                              , G__asm_cp, rewind_arg);
#endif
                              G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                              G__asm_inst[G__asm_cp+1] = rewind_arg;
                              G__inc_cp_asm(2, 0);
                           }
#ifdef G__ASM_DBG
                           if (G__asm_dbg) G__fprinterr(G__serr, "%3x: BASECONV %d %d\n"
                                                           , G__asm_cp, G__get_tagnum(formal_tagnum), baseoffset);
#endif
                           G__asm_inst[G__asm_cp] = G__BASECONV;
                           G__asm_inst[G__asm_cp+1] = G__get_tagnum(formal_tagnum);
                           G__asm_inst[G__asm_cp+2] = baseoffset;
                           G__inc_cp_asm(3, 0);
                           if (rewind_arg
                              ) {
#ifdef G__ASM_DBG
                              if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                              , G__asm_cp, -rewind_arg);
#endif
                              G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                              G__asm_inst[G__asm_cp+1] = -rewind_arg;
                              G__inc_cp_asm(2, 0);
                           }
                        }
#endif
                     }
                  }
                  break;
            }
            break;
         case 'U':
            switch (param_type) {
               case 'U':
                  /* Pointer to derived class can be converted to
                   * pointer to base class.
                   * add offset, modify char *parameter and
                   * G__value *param
                   */
                  if (-1 != (baseoffset = G__ispublicbase(G__get_tagnum(formal_tagnum), G__get_tagnum(param_tagnum)
                                                          , (void*)param->obj.i))) {
                     G__replace_rawtype(G__value_typenum(*param), formal_tagnum); // param->tagnum = G__get_tagnum(formal_tagnum);
                     param->obj.i += baseoffset;
                     param->ref += baseoffset;
#ifdef G__ASM
                     if (G__asm_noverflow) {
                        if (rewind_arg
                           ) {
#ifdef G__ASM_DBG
                           if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                           , G__asm_cp, rewind_arg);
#endif
                           G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                           G__asm_inst[G__asm_cp+1] = rewind_arg;
                           G__inc_cp_asm(2, 0);
                        }
#ifdef G__ASM_DBG
                        if (G__asm_dbg) G__fprinterr(G__serr, "%3x: BASECONV %d %d\n"
                                                        , G__asm_cp, G__get_tagnum(formal_tagnum), baseoffset);
#endif
                        G__asm_inst[G__asm_cp] = G__BASECONV;
                        G__asm_inst[G__asm_cp+1] = G__get_tagnum(formal_tagnum);
                        G__asm_inst[G__asm_cp+2] = baseoffset;
                        G__inc_cp_asm(3, 0);
                        if (rewind_arg
                           ) {
#ifdef G__ASM_DBG
                           if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                           , G__asm_cp, -rewind_arg);
#endif
                           G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                           G__asm_inst[G__asm_cp+1] = -rewind_arg;
                           G__inc_cp_asm(2, 0);
                        }
                     }
#endif
                  }
                  break;
            }
#ifndef G__OLDIMPLEMENTATION2191
         case '1':
#else
         case 'Q':
#endif
            if ('C' == param_type &&
                  G__get_funcproperties(func)->entry.size < 0
               ) {
               G__genericerror("Limitation: Precompiled function can not get pointer to interpreted function as argument");
               return(-1);
            }
      }
   }
#ifdef G__OLDIMPLEMENTATION2195_YET
   if (G__asm_cp > store_asm_cp) {
      if (G__asm_dbg) G__fprinterr(G__serr, "G__convert_param instructions cancelled\n");
      G__inc_cp_asm(store_asm_cp - G__asm_cp, 0);
   }
#endif
   return(0);
}

//______________________________________________________________________________
static void G__display_param(FILE* fp, const ::Reflex::Scope scopetagnum, char* funcname, G__param* libp)
{
   int i;
#ifndef G__OLDIMPLEMENTATION1485
   if (fp == G__serr) {
      if (!scopetagnum.IsTopScope()) {
         G__fprinterr(G__serr, "%s::", scopetagnum.Name(::Reflex::SCOPED).c_str());
      }
      else {
         G__fprinterr(G__serr, "::");
      }
      G__fprinterr(G__serr, "%s(", funcname);
      for (i = 0; i < libp->paran; ++i) {
         switch (G__get_type(G__value_typenum(libp->para[i]))) {
            case 'd':
            case 'f':
               G__fprinterr(G__serr, "%s", G__type2string(G__get_type(G__value_typenum(libp->para[i]))
                            , G__get_tagnum(G__value_typenum(libp->para[i]))
                            , G__get_typenum(G__value_typenum(libp->para[i]))
                            , 0
                            , 0));
               break;
            default:
               G__fprinterr(G__serr, "%s", G__type2string(G__get_type(G__value_typenum(libp->para[i]))
                            , G__get_tagnum(G__value_typenum(libp->para[i]))
                            , G__get_typenum(G__value_typenum(libp->para[i]))
                            , G__get_reftype(G__value_typenum(libp->para[i]))
                            , 0));
               break;
         }
         if (i != (libp->paran - 1)) {
            G__fprinterr(G__serr, ",");
         }
      }
      G__fprinterr(G__serr, ");\n");
   }
   else {
      // --
#endif // G__OLDIMPLEMENTATION1485
      if (scopetagnum && !scopetagnum.IsTopScope()) {
         fprintf(fp, "%s::", scopetagnum.Name(::Reflex::SCOPED).c_str());
      }
      else {
         fprintf(fp, "::");
      }
      fprintf(fp, "%s(", funcname);
      for (i = 0; i < libp->paran; ++i) {
         switch (G__get_type(G__value_typenum(libp->para[i]))) {
            case 'd':
            case 'f':
               fprintf(fp, "%s", G__type2string(G__get_type(G__value_typenum(libp->para[i]))
                                                , G__get_tagnum(G__value_typenum(libp->para[i]))
                                                , G__get_typenum(G__value_typenum(libp->para[i]))
                                                , 0
                                                , 0));
               break;
            default:
               fprintf(fp, "%s", G__type2string(G__get_type(G__value_typenum(libp->para[i]))
                                                , G__get_tagnum(G__value_typenum(libp->para[i]))
                                                , G__get_typenum(G__value_typenum(libp->para[i]))
                                                , G__get_reftype(G__value_typenum(libp->para[i]))
                                                , 0));
               break;
         }
         if (i != (libp->paran - 1)) {
            fprintf(fp, ",");
         }
      }
      fprintf(fp, ");\n");
#ifndef G__OLDIMPLEMENTATION1485
   }
#endif // G__OLDIMPLEMENTATION1485
   // --
}

//______________________________________________________________________________
static void G__display_func(FILE* fp, const ::Reflex::Member func)
{
   unsigned int i;
   int store_iscpp = G__iscpp;
   G__iscpp = 1;

   if (!func || !G__get_funcproperties(func) /* ->pentry */) return;

#ifndef G__OLDIMPLEMENTATION1485
   if (G__serr == fp) {
      if (G__get_funcproperties(func)->filenum >= 0) { /* 2012 must leave this one */
         G__fprinterr(G__serr, "%-10s%4d ", G__stripfilename(G__srcfile[G__get_funcproperties(func)->filenum].filename), G__get_funcproperties(func)->linenum);
      }
      else {
         G__fprinterr(G__serr, "%-10s%4d ", "(compiled)", 0);
      }
      G__fprinterr(G__serr, "%s ", func.TypeOf().ReturnType().Name(::Reflex::SCOPED |::Reflex::QUALIFIED).c_str());
      //G__type2string(ifunc->type[ifn]
      //                                ,ifunc->p_tagtable[ifn]
      //                                ,G__get_typenum((ifunc->p_typetable[ifn]))
      //                                ,ifunc->reftype[ifn]
      //                                ,ifunc->isconst[ifn]));
      if (!func.DeclaringScope().IsTopScope()) G__fprinterr(G__serr, "%s::", func.DeclaringScope().Name(::Reflex::SCOPED).c_str());
      G__fprinterr(G__serr, "%s(", func.Name().c_str());
      for (i = 0;i < func.TypeOf().FunctionParameterSize();i++) {
         G__fprinterr(G__serr, "%s", func.TypeOf().FunctionParameterAt(i).Name(::Reflex::SCOPED |::Reflex::QUALIFIED).c_str());
         //G__type2string(ifunc->para_type[ifn][i]
         //                            ,ifunc->para_p_tagtable[ifn][i]
         //                            ,G__get_typenum((ifunc->para_p_typetable[ifn][i]))
         //                            ,ifunc->para_reftype[ifn][i]
         //                            ,ifunc->para_isconst[ifn][i]));
         if (i != func.TypeOf().FunctionParameterSize() - 1) G__fprinterr(G__serr, ",");
      }
      G__fprinterr(G__serr, ");\n");
   }
   else {
#endif
      if (G__get_funcproperties(func)->filenum >= 0) { /* 2012 must leave this one */
         fprintf(fp, "%-10s%4d "
                 , G__stripfilename(G__srcfile[G__get_funcproperties(func)->filenum].filename)
                 , G__get_funcproperties(func)->linenum);
      }
      else {
         fprintf(fp, "%-10s%4d ", "(compiled)", 0);
      }
      fprintf(fp, "%s ", func.TypeOf().ReturnType().Name(::Reflex::SCOPED |::Reflex::QUALIFIED).c_str());
      if (!func.DeclaringScope().IsTopScope()) fprintf(fp, "%s::", func.DeclaringScope().Name(::Reflex::SCOPED).c_str());
      fprintf(fp, "%s(", func.Name().c_str());
      for (i = 0;i < func.TypeOf().FunctionParameterSize();i++) {
         fprintf(fp, "%s", func.TypeOf().FunctionParameterAt(i).Name(::Reflex::SCOPED |::Reflex::QUALIFIED).c_str());
         if (i != func.TypeOf().FunctionParameterSize() - 1) fprintf(fp, ",");
      }
      fprintf(fp, ");\n");
#ifndef G__OLDIMPLEMENTATION1485
   }
#endif

   G__iscpp = store_iscpp;
}

//______________________________________________________________________________
static void G__display_ambiguous(const ::Reflex::Scope& scopetagnum, char* funcname, G__param* libp, G__funclist* funclist, unsigned int bestmatch)
{
   G__fprinterr(G__serr, "Calling : ");
   G__display_param(G__serr, scopetagnum, funcname, libp);
   G__fprinterr(G__serr, "Match rank: file     line  signature\n");
   while (funclist) {
      if (bestmatch == funclist->rate) G__fprinterr(G__serr, "* %8x ", funclist->rate);
      else                          G__fprinterr(G__serr, "  %8x ", funclist->rate);
      G__display_func(G__serr, funclist->ifunc);
      funclist = funclist->next;
   }
}

//______________________________________________________________________________
struct G__funclist* Cint::Internal::G__add_templatefunc(char *funcnamein, G__param* libp, int hash, G__funclist *funclist, const ::Reflex::Scope &p_ifunc, int isrecursive)
{
   // Search matching template function, search by name then parameter.
   // If match found, expand template, parse as pre-run
   struct G__Definetemplatefunc *deftmpfunc;
   struct G__Charlist call_para;
   /* int env_tagnum=G__get_envtagnum(); */
   int env_tagnum = G__get_tagnum(p_ifunc);
   struct G__inheritance *baseclass;
   ::Reflex::Scope store_friendtagnum = G__friendtagnum;
   char *funcname;
   char *ptmplt;
   char *pexplicitarg = 0;

   funcname = (char*)malloc(strlen(funcnamein) + 1);
   strcpy(funcname, funcnamein);

   if (env_tagnum != -1) baseclass = G__struct.baseclass[env_tagnum];
   else               baseclass = &G__globalusingnamespace;
   if (0 == baseclass->basen) baseclass = 0;


   call_para.string = 0;
   call_para.next = 0;
   deftmpfunc = &G__definedtemplatefunc;

   ptmplt = strchr(funcname, '<');
   if (ptmplt)
   {
      if (strncmp("operator", funcname, ptmplt - funcname) == 0) {
         /* We have operator< */
         if (ptmplt[1] == '<') ptmplt = strchr(ptmplt + 2, '<');
         else ptmplt = strchr(ptmplt + 1, '<');
      }
   }
   if (ptmplt)
   {
      if ((-1 != env_tagnum) && strcmp(funcname, G__struct.name[env_tagnum]) == 0) {
         /* this is probably a template constructor of a class template */
         ptmplt = (char*)0;
      }
      else {
         int tmp;
         *ptmplt = 0;
         if (G__defined_templatefunc(funcname)) {
            G__hash(funcname, hash, tmp);
         }
         else {
            pexplicitarg = ptmplt;
            *ptmplt = '<';
            ptmplt = (char*)0;
         }
      }
   }

   if (pexplicitarg)
   {
      /* funcname="f<int>" ->  funcname="f" , pexplicitarg="int>" */
      int tmp = 0;
      *pexplicitarg = 0;
      ++pexplicitarg;
      G__hash(funcname, hash, tmp);
   }

   /* Search matching template function name */
   while (deftmpfunc->next)
   {
      G__freecharlist(&call_para);
      if (ptmplt) {
         int itmp = 0;
         int ip = 1;
         int c;
         G__StrBuf buf_sb(G__ONELINE);
         char *buf = buf_sb;
         do {
            c = G__getstream_template(ptmplt, &ip, buf, ",>");
            G__checkset_charlist(buf, &call_para, ++itmp, 'u');
         }
         while (c != '>');
      }
      if (deftmpfunc->hash == hash && strcmp(deftmpfunc->name, funcname) == 0 &&
            (G__matchtemplatefunc(deftmpfunc, libp, &call_para, G__PROMOTION)
#ifndef G__OLDIMPLEMEMTATION2214
             || (pexplicitarg && libp->paran == 0)
#else
             || pexplicitarg
#endif
            )) {

         if (-1 != deftmpfunc->parent_tagnum &&
               env_tagnum != deftmpfunc->parent_tagnum) {
            if (baseclass) {
               int temp;
               for (temp = 0;temp < baseclass->basen;temp++) {
                  if (baseclass->basetagnum[temp] == deftmpfunc->parent_tagnum) {
                     goto match_found;
                  }
               }
            }
            deftmpfunc = deftmpfunc->next;
            continue;
         }
   match_found:
         G__friendtagnum = G__Dict::GetDict().GetScope(deftmpfunc->friendtagnum);

         if (pexplicitarg) {
            int npara = 0;
            G__gettemplatearglist(pexplicitarg, &call_para
                                  , deftmpfunc->def_para, &npara
                                  , -1, 0
                                 );
         }

         if (pexplicitarg) {
            int tmp = 0;
            G__hash(funcname, hash, tmp);
         }

         /* matches funcname and parameter,
          * then expand the template and parse as prerun */
         G__replacetemplate(
            funcname
            , funcnamein
            , &call_para /* needs to make this up */
            , deftmpfunc->def_fp
            , deftmpfunc->line
            , deftmpfunc->filenum
            , &(deftmpfunc->def_pos)
            , deftmpfunc->def_para
            , 0
            , SHRT_MAX /* large enough number */
            , deftmpfunc->parent_tagnum
         );

         G__friendtagnum = store_friendtagnum;


         /* search for instantiated template function */
         ::Reflex::Member func = p_ifunc.FunctionMemberAt(p_ifunc.FunctionMemberSize() - 1);
         if (func) {
            if (
               strcmp(funcnamein, func.Name().c_str()) == 0
            ) {
               if (ptmplt) {
                  //int tmp;
                  //*ptmplt='<';
                  //free((void*)ifunc->funcname[ifn]);
                  //ifunc->funcname[ifn] = (char*)malloc(strlen(funcnamein)+1);
                  //strcpy(ifunc->funcname[ifn],funcnamein);
                  //G__hash(funcnamein,hash,tmp);
                  //ifunc->hash[ifn] = hash;
               }
               if (0 == G__get_funcproperties(func)->entry.p && G__NOLINK == G__globalcomp) {
                  /* This was only a prototype template, search for definition
                   * template */
                  deftmpfunc = deftmpfunc->next;
                  continue;
               }
               funclist = G__funclist_add(funclist, func, 0);
               if ((int)func.FunctionParameterSize() < libp->paran ||
                     ((int)func.FunctionParameterSize() > libp->paran &&
                      !func.FunctionParameterSize(true))) {
                  funclist->rate = G__NOMATCH;
               }
               else {
                  G__rate_parameter_match(libp, func, funclist, isrecursive);
               }
            }
         }
         G__freecharlist(&call_para);
      }
      deftmpfunc = deftmpfunc->next;
   }
   G__freecharlist(&call_para);
   if (funcname) free((void*)funcname);
   return funclist;
}

//______________________________________________________________________________
static struct G__funclist* G__rate_binary_operator(const ::Reflex::Scope &p_ifunc, G__param *libp, const ::Reflex::Type &tagnum, char* funcname, int /*hash*/, G__funclist *funclist, int isrecursive)
{
   int i;
   struct G__param fpara;
#ifdef G__DEBUG
   {
      int jdbg;
      int sizedbg = sizeof(struct G__param);
      char *pcdbg = (char*)(&fpara);
      for (jdbg = 0;jdbg < (int)sizedbg;jdbg++) {
         *(pcdbg + jdbg) = (char)0xa3;
      }
   }
#endif
   /* set 1st argument as the object */
   G__value_typenum(fpara.para[0]) = G__modify_type(tagnum, 0, 0, G__isconst, 0, 0);
   fpara.para[0].obj.i = (long)G__store_struct_offset;
   fpara.para[0].ref = (long)G__store_struct_offset;
   /* set 2nd to n arguments */
   fpara.paran = libp->paran + 1;
   for (i = 0;i < libp->paran;i++) fpara.para[i+1] = libp->para[i];
   /* Search for name match
    *  if reserved func or K&R, match immediately
    *  check number of arguments and default parameters
    *  rate parameter match */
   for (::Reflex::Member_Iterator i = p_ifunc.FunctionMember_Begin();
         i != p_ifunc.FunctionMember_End();
         ++i)
   {
      if (/* hash==p_ifunc->hash[ifn]&& */ i->Name() == funcname) {
         if ((int)i->FunctionParameterSize() < fpara.paran ||
               ((int)i->FunctionParameterSize(true) <= fpara.paran)
#ifdef G__OLDIMPLEMENTATION1260_YET
               || (G__isconst && i->TypeOf().IsConst())
#endif
               || (isrecursive && i->IsExplicit())
         ) {}
         else {
            funclist = G__funclist_add(funclist, *i, 0);
            G__rate_parameter_match(&fpara, *i, funclist, isrecursive);
            funclist->ifunc = 0; /* added as dummy */
         }
      }
   }
   return funclist;
}

//______________________________________________________________________________
static int G__identical_function(G__funclist *match, G__funclist *func)
{
   unsigned int ipara;
   if (!match || !match->ifunc || !func || !func->ifunc) return(0);
   for (ipara = 0;ipara < match->ifunc.FunctionParameterSize();ipara++) {
      if (
         match->ifunc.TypeOf().FunctionParameterAt(ipara).FinalType()
         != func->ifunc.TypeOf().FunctionParameterAt(ipara).FinalType()
         //(match->ifunc->para_type[match->ifn][ipara] !=
         // func->ifunc->para_type[func->ifn][ipara]) ||
         //(match->ifunc->para_p_tagtable[match->ifn][ipara] !=
         // func->ifunc->para_p_tagtable[func->ifn][ipara]) ||
         //(match->ifunc->para_p_typetable[match->ifn][ipara] !=
         // func->ifunc->para_p_typetable[func->ifn][ipara]) ||
         //(match->ifunc->para_isconst[match->ifn][ipara] !=
         // func->ifunc->para_isconst[func->ifn][ipara]) ||
         //(match->ifunc->para_reftype[match->ifn][ipara] !=
         // func->ifunc->para_reftype[func->ifn][ipara])
      ) {
         return(0);
      }
   }

   return(1);
}

//______________________________________________________________________________
static ::Reflex::Member G__overload_match(char* funcname, G__param* libp, int hash, const ::Reflex::Scope p_ifunc, int memfunc_flag, int access, int isrecursive, int doconvert, int* match_error)
{
   struct G__funclist* funclist = 0;
   struct G__funclist* match = 0;
   unsigned int bestmatch = G__NOMATCH;
   struct G__funclist* func = 0;
   int ambiguous = 0;
   ::Reflex::Scope store_ifunc = p_ifunc;
   ::Reflex::Scope ifunc = p_ifunc;
   ::Reflex::Member result;
   int ix = 0;


   // Search for name match
   
   while (ifunc) {
      for (::Reflex::Member_Iterator ifn = ifunc.FunctionMember_Begin();
            ifn != ifunc.FunctionMember_End();
            ++ifn) {
         if (/* hash==ifunc->hash[ifn] && */ifn->Name() == funcname) {
            if (G__get_funcproperties(*ifn)->entry.ansi == 0 || /* K&R C style header */
                  G__get_funcproperties(*ifn)->entry.ansi == 2 || /* variable number of args */
                  (G__HASH_MAIN == hash && strcmp(funcname, "main") == 0)) {
               /* special match */
               G__funclist_delete(funclist);
               return(*ifn);
            }
            if (-1 != G__get_tagnum(ifunc) &&
                  (memfunc_flag == G__TRYNORMAL && doconvert)
                  && ifunc.Name() == funcname) {
               continue;
            }
            funclist = G__funclist_add(funclist, *ifn, 0);
            if ((int)ifn->FunctionParameterSize() < libp->paran ||
                  ((int)ifn->FunctionParameterSize(true) > libp->paran)
#ifdef G__OLDIMPLEMENTATION1260_YET
                  || (G__isconst && ifn->TypeOf().IsConst())
#endif
                  || (isrecursive && ifn->IsExplicit())
               ) {
               funclist->rate = G__NOMATCH;
            }
            else {
               G__rate_parameter_match(libp, *ifn, funclist, isrecursive);
            }
            if (G__EXACTMATCH == (funclist->rate & 0xffffff00)) {
               match = funclist;
            }
         }
      }

      if (store_ifunc == G__p_ifunc && ix < G__globalusingnamespace.basen) {
         ifunc = G__Dict::GetDict().GetScope(G__globalusingnamespace.basetagnum[ix]);
         ++ix;
      }
      else {
         ifunc = ::Reflex::Scope();
      }
   }

   // If exact match does not exist
   if (!match) {
      funclist =  G__add_templatefunc(funcname, libp, hash, funclist, store_ifunc, isrecursive);
   }

   if (!match && (G__TRYUNARYOPR == memfunc_flag || G__TRYBINARYOPR == memfunc_flag)) {
      for (ix = 0;ix < G__globalusingnamespace.basen;ix++) {
         funclist = G__rate_binary_operator(G__Dict::GetDict().GetScope(G__globalusingnamespace.basetagnum[ix]), libp, G__tagnum, funcname, hash, funclist, isrecursive);
      }
      funclist = G__rate_binary_operator(::Reflex::Scope::GlobalScope(), libp, G__tagnum, funcname, hash, funclist, isrecursive);
   }

   // if there is no name match, return null
   if (!funclist) {
      return ::Reflex::Member();
   }

   //  choose the best match
   func = funclist;
   ambiguous = 0;
   while (func) {
      if (func->rate < bestmatch) {
         bestmatch = func->rate;
         match = func;
         ambiguous = 0;
      }
      else if (func->rate == bestmatch && bestmatch != G__NOMATCH) {
         if (0 == G__identical_function(match, func)) {
            ++ambiguous;
         }
         match = func;
      }
      func = func->next;
   }

   if ((G__TRYUNARYOPR == memfunc_flag || G__TRYBINARYOPR == memfunc_flag) && match && !match->ifunc) {
      G__funclist_delete(funclist);
      return ::Reflex::Member();
   }

   if (!match) {
      G__funclist_delete(funclist);
      return ::Reflex::Member();
   }

   if (ambiguous && G__EXACTMATCH != bestmatch && !isrecursive) {
      if (!G__mask_error) {
         /* error, ambiguous overloading resolution */
         G__fprinterr(G__serr, "Error: Ambiguous overload resolution (%x,%d)", bestmatch, ambiguous + 1);
         G__genericerror(0);
         G__display_ambiguous(p_ifunc, funcname, libp, funclist, bestmatch);
      }
      *match_error = 1;
      G__funclist_delete(funclist);
      return ::Reflex::Member();
   }

   /* best match function found */
   result = match->ifunc;

   /*  check private, protected access rights
    *    display error if no access right
    *    do parameter conversion if needed */
   if (!G__test_access(result, access) && (!G__isfriend(G__get_tagnum(result.DeclaringScope())))
         && G__NOLINK == G__globalcomp
         && G__TRYCONSTRUCTOR !=  memfunc_flag
   ) {
      /* no access right */
      G__fprinterr(G__serr, "Error: can not call private or protected function");
      G__genericerror(0);
      G__fprinterr(G__serr, "  ");
      G__display_func(G__serr, result);
      G__display_ambiguous(p_ifunc, funcname, libp, funclist, bestmatch);
      *match_error = 1;
      G__funclist_delete(funclist);
      return ::Reflex::Member();
   }

   /* convert parameter */
   if (
      doconvert &&
      G__convert_param(libp, result, match))
      return ::Reflex::Member();

   G__funclist_delete(funclist);
   return result;
}

//______________________________________________________________________________
static ::Reflex::Scope G__create_scope(const ::Reflex::Member &ifn, const ::Reflex::Scope &calling_scope, G__param *libp)
{
   // set enclosing/enclosed scope info to m_scope
   std::stringstream sScopeName;
   // NOT the same as m_ifunc.Name(SCOPED|QUALIFIED)!
   // we need the lookup to work, so we really need
   // ns1::ns2::classA::$void myfunc(int a)$
   if (!ifn.DeclaringScope().IsTopScope()) sScopeName << "::" << ifn.DeclaringScope().Name(Reflex::SCOPED);
   sScopeName << "::$" << ifn.Name(Reflex::QUALIFIED);
   sScopeName << "$"; // hidden
   ::Reflex::Scope localvar = Reflex::NamespaceBuilder(sScopeName.str().c_str()).ToScope();

   // ::Reflex::Scope localvar = Reflex::Scope::GlobalScope().LookupScope(sScopeName.str());

   G__get_properties(localvar)->stackinfo.calling_scope = calling_scope; // G__p_local;
   G__get_properties(localvar)->stackinfo.ifunc = ifn;

#ifdef G__VAARG
   G__get_properties(localvar)->stackinfo.libp = libp;
#endif
   if (!G__tagnum.IsTopScope())
      G__get_properties(localvar)->stackinfo.tagnum = G__tagnum;

   G__get_properties(localvar)->stackinfo.struct_offset = G__store_struct_offset;
   G__get_properties(localvar)->stackinfo.exec_memberfunc = G__exec_memberfunc;
   G__get_properties(localvar)->stackinfo.prev_filenum = G__ifile.filenum;
   G__get_properties(localvar)->stackinfo.prev_line_number = G__ifile.line_number;

   return localvar;
}

//______________________________________________________________________________
int Cint::Internal::G__interpret_func(G__value *result7, struct G__param *libp, int hash, const ::Reflex::Member &func, int funcmatch, int memfunc_flag)
{
#ifndef __GNUC__
#pragma message (FIXME("Remove static buf and this func's overload"))
#endif // __GNUC__
   G__StrBuf funcname_sb(G__LONGLINE);
   char *funcname = funcname_sb;
   strcpy(funcname, func.Name().c_str());
   return G__interpret_func(result7, funcname, libp, hash, func.DeclaringScope(), funcmatch, memfunc_flag);
}

//______________________________________________________________________________
int Cint::Internal::G__interpret_func(G__value *result7, char* funcname, G__param *libp, int hash, const ::Reflex::Scope &input_ifunc, int funcmatch , int memfunc_flag)
{
   /*  return 1 if function is executed */
   /*  return 0 if function isn't executed */
   ::Reflex::Scope p_ifunc(input_ifunc);
   ::Reflex::Member ifn;
   ::Reflex::Scope G_local;
   FILE *prev_fp;
   fpos_t prev_pos /*,temppos */;
   /* paraname[][] is used only for K&R func param. length should be OK */
   G__StrBuf paraname_buf(G__MAXFUNCPARA * G__MAXNAME);
   typedef char namearray_t[G__MAXNAME];
   namearray_t *paraname = (namearray_t*) paraname_buf.data();
#ifdef G__OLDIMPLEMENTATION1802
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
#endif
   unsigned int ipara = 0;
   int cin = '\0';
   int itemp = 0;
   /* int store_linenumber; */
   int break_exit_func;
   int store_decl;
   G__value buf;
   int store_var_type;
   ::Reflex::Scope store_tagnum;
   char *store_struct_offset; /* used to be int */
   ::Reflex::Scope store_inherit_tagnum;
   char *store_inherit_offset;
   ::Reflex::Scope virtualtag;
   int store_def_struct_member;
   int store_var_typeB;
   int store_doingconstruction;
   ::Reflex::Member store_func_now;
   int store_iscpp;
   int store_exec_memberfunc;
   G__UINT32 store_security;
   int match_error = 0;
#ifdef G__ASM_IFUNC
   G__StrBuf asm_inst_g_sb(G__MAXINST * sizeof(long));
   long *asm_inst_g = (long*) asm_inst_g_sb.data(); /* p-code instruction buffer */
   G__StrBuf asm_stack_g_sb(G__MAXSTACK * sizeof(G__value));
   G__value* asm_stack_g = (G__value*) asm_stack_g_sb.data(); /* data stack */
   G__StrBuf asm_name_sb(G__ASM_FUNCNAMEBUF);
   char *asm_name = asm_name_sb;
   long *store_asm_inst;
   int store_asm_instsize;
   G__value *store_asm_stack;
   char *store_asm_name;
   int store_asm_name_p;
   struct G__param *store_asm_param;
   int store_asm_exec;
   int store_asm_noverflow;
   int store_asm_cp;
   int store_asm_dt;
   ::Reflex::Member store_asm_index; /* maybe unneccessary */
#endif
#ifdef G__ASM_WHOLEFUNC
   int store_no_exec_compile = 0;
   ::Reflex::Scope localvar;
#endif
#ifdef G__NEWINHERIT
   int basen = 0;
   int isbase;
   int access;
   int memfunc_or_friend = 0;
   struct G__inheritance *baseclass = 0;
#endif
   /* #define G__OLDIMPLEMENTATION590 */
   ::Reflex::Scope local_tagnum;
   ::Reflex::Scope store_p_ifunc = p_ifunc;
   int specialflag = 0;
   G__value *store_p_tempobject = 0;
   char *store_memberfunc_struct_offset;
   ::Reflex::Scope store_memberfunc_tagnum;
#ifdef G__NEWINHERIT
   store_inherit_offset = G__store_struct_offset;
   store_inherit_tagnum = G__tagnum;
#endif
   store_asm_noverflow = G__asm_noverflow;
#ifdef G__ASM_IFUNC
   if (G__asm_exec) {
      ifn = G__asm_index;
      /* delete 0 ~destructor ignored */
      if (0 == G__store_struct_offset && -1 != G__get_tagnum(p_ifunc) &&
            !ifn.IsStatic() && ifn.IsDestructor()) {
         return(1);
      }
      goto asm_ifunc_start;
   }
#endif
   /*******************************************************
    * searching function
    *******************************************************/
#ifdef G__NEWINHERIT
   if ((G__exec_memberfunc && (!G__tagnum.IsTopScope() || !G__memberfunc_tagnum.IsTopScope())) ||
         G__TRYNORMAL != memfunc_flag) {
      isbase = 1;
      basen = 0;
      if (G__exec_memberfunc && G__tagnum.IsTopScope()) local_tagnum = G__memberfunc_tagnum;
      else                               local_tagnum = G__tagnum;
      baseclass = G__struct.baseclass[G__get_tagnum(local_tagnum)];
      if (G__exec_memberfunc || G__isfriend(G__get_tagnum(G__tagnum))) {
         access = G__PUBLIC_PROTECTED_PRIVATE;
         memfunc_or_friend = 1;
      }
      else {
         access = G__PUBLIC;
         memfunc_or_friend = 0;
      }
   }
   else {
      access = G__PUBLIC;
      isbase = 0;
      if (p_ifunc && p_ifunc == G__p_ifunc) {
         basen = 0;
         isbase = 1;
         baseclass = &G__globalusingnamespace;
      }
   }
next_base:
#endif
   ifn = G__overload_match(funcname, libp, hash, p_ifunc, memfunc_flag, access, 0, 1, &match_error);
   /* error */
   if (match_error) {
      *result7 = G__null;
      return(1);
   }
#ifdef G__NEWINHERIT
   /**********************************************************************
    * iteration for base class member function search
    **********************************************************************/
   if (!ifn ||
         (!G__test_access(ifn, G__PUBLIC) && !G__isfriend(G__get_tagnum(G__tagnum))
          && (0 == G__exec_memberfunc ||
              (local_tagnum != G__memberfunc_tagnum
               && (!G__test_access(ifn, G__PROTECTED)
                   || -1 == G__ispublicbase(G__get_tagnum(local_tagnum), G__get_tagnum(G__memberfunc_tagnum)
                                            , (void*)store_inherit_offset))
              )))) {
      if (isbase) {
         while (baseclass && basen < baseclass->basen) {
            if (memfunc_or_friend) {
               if ((baseclass->baseaccess[basen]&G__PUBLIC_PROTECTED) ||
                     baseclass->property[basen]&G__ISDIRECTINHERIT) {
                  access = G__PUBLIC_PROTECTED;
                  G__incsetup_memfunc(baseclass->basetagnum[basen]);
                  p_ifunc = G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]);
#ifdef G__VIRTUALBASE
                  // require !G__prerun, else store_inherit_offset might not point
                  // to a valid object with vtable, and G__getvirtualbaseoffset()
                  // might fail. We should not need the voffset in this case
                  // anyway, as we don't actually call the function.
                  if (!G__prerun && baseclass->property[basen]&G__ISVIRTUALBASE) {
                     G__store_struct_offset = store_inherit_offset + G__getvirtualbaseoffset(store_inherit_offset, G__get_tagnum(G__tagnum), baseclass, basen);
                  }
                  else {
                     G__store_struct_offset = store_inherit_offset + (long)baseclass->baseoffset[basen];
                  }
#else
                  G__store_struct_offset
                  = store_inherit_offset + baseclass->baseoffset[basen];
#endif
                  G__tagnum = G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]);
                  ++basen;
                  store_p_ifunc = p_ifunc;
                  goto next_base; /* I know this is a bad manner */
               }
            }
            else {
               if (baseclass->baseaccess[basen]&G__PUBLIC) {
                  access = G__PUBLIC;
                  G__incsetup_memfunc(baseclass->basetagnum[basen]);
                  p_ifunc = G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]);
#ifdef G__VIRTUALBASE
                  if (baseclass->property[basen]&G__ISVIRTUALBASE) {
                     G__store_struct_offset = store_inherit_offset +
                                              G__getvirtualbaseoffset(store_inherit_offset, G__get_tagnum(G__tagnum)
                                                                      , baseclass, basen);
                  }
                  else {
                     G__store_struct_offset
                     = store_inherit_offset + (long)baseclass->baseoffset[basen];
                  }
#else
                  G__store_struct_offset
                  = store_inherit_offset + baseclass->baseoffset[basen];
#endif
                  G__tagnum = G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]);
                  ++basen;
                  store_p_ifunc = p_ifunc;
                  goto next_base; /* I know this is a bad manner */
               }
            }
            ++basen;
         }
         isbase = 0;
      }

      if (0 == specialflag && 1 == libp->paran && -1 != G__get_tagnum(G__value_typenum(libp->para[0])) &&
            -1 != G__struct.parent_tagnum[G__get_tagnum(G__value_typenum(libp->para[0]))]) {
         p_ifunc =
            G__Dict::GetDict().GetScope(G__struct.parent_tagnum[G__get_tagnum(G__value_typenum(libp->para[0]))]);
         switch (G__get_tagtype(p_ifunc)) {
            case 's':
            case 'c':
               store_p_ifunc = p_ifunc;
               specialflag = 1;
               goto next_base;
         }
      }

      /* not found */
      G__store_struct_offset = store_inherit_offset;
      G__tagnum = store_inherit_tagnum;
      G__asm_noverflow = store_asm_noverflow;
      return(0);
   }
#else
   /******************************************************************
   * if no such func, return 0
   *******************************************************************/
   if (!p_ifunc) {
      return(0);
   }
   /******************************************************************
   * member access control
   *******************************************************************/
   if (G__PUBLIC != p_ifunc->access[ifn] && !G__isfriend(G__tagnum)) {
      return(0);
   }
#endif
asm_ifunc_start:   /* loop compilation execution label */
   /******************************************************************
    * Constructor or destructor call in G__make_ifunctable() parameter
    * type allocation. Return without call.
    * Also, when parameter analysis with -c optoin, return without call.
    *******************************************************************/
   if (G__globalcomp) { /* with -c-1 or -c-2 option */
      result7->obj.d = 0.0;
      result7->ref = 0;
      G__value_typenum(*result7) = ifn.TypeOf().ReturnType();
      return(1);
   }
   else if (G__prerun) { /* in G__make_ifunctable parameter allocation */
      result7->obj.i = G__get_type(ifn.TypeOf().ReturnType());
      result7->ref = 0;
      G__value_typenum(*result7) = G__replace_rawtype(ifn.TypeOf().ReturnType(), G__get_from_type(G__DEFAULT_FUNCCALL, 0));
      return(1);
   }
   /******************************************************************
    * error if body not defined
    *******************************************************************/
   if (
      // --
#ifdef G__ASM_WHOLEFUNC
      (((bool)ifn) && G__get_funcproperties(ifn)->entry.p == 0) &&
      !ifn.IsAbstract() &&
      G__ASM_FUNC_NOP == G__asm_wholefunction
#else
      !G__get_funproperties(ifn)->entry.p &&
      !ifn.IsAbstract()
#endif
      // --
   ) {
      {
         if (0 == G__templatefunc(result7, funcname, libp, hash, funcmatch)) {
            if (G__USERCONV == funcmatch) {
               *result7 = G__null;
               G__fprinterr(G__serr, "Error: %s() header declared but not defined", funcname);
               G__genericerror(0);
               return(1);
            }
            else return(0);
         }
         return(1);
      }
   }
   /******************************************************************
    * function was found in interpreted function list
    *******************************************************************/
   /* p_ifunc has found */
   /******************************************************************
    * Add baseoffset if calling base class member function.
    * Resolution of virtual function is not done here. There is a
    * separate section down below. Search string 'virtual function'
    * to get there.
    *******************************************************************/
   G__tagnum = p_ifunc; // G__Dict::GetDict().GetScope(p_ifunc->tagnum);
   store_exec_memberfunc = G__exec_memberfunc;
   if (G__tagnum.IsTopScope() && !G__memberfunc_tagnum) {
      G__exec_memberfunc = 0;
   }
#define G__OLDIMPLEMENTATION1101
#ifndef G__OLDIMPLEMENTATION1101
   if (memfunc_flag == G__CALLSTATICMEMFUNC && 0 == G__store_struct_offset &&
         -1 != G__tagnum && 0 == p_ifunc->staticalloc[ifn] && 0 == G__no_exec) {
      G__fprinterr(G__serr, "Error: %s() Illegal non-static member function call", funcname);
      G__genericerror(0);
      *result7 = G__null;
      return(1);
   }
#endif
   store_var_typeB = G__var_typeB;
   G__var_typeB = 'p';
#ifdef G__NEWINHERIT
#ifdef G__ASM
   if (G__asm_noverflow && G__store_struct_offset &&
         G__store_struct_offset != store_inherit_offset) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x: ADDSTROS %ld\n"
                      , G__asm_cp, G__store_struct_offset - store_inherit_offset);
      }
#endif
      G__asm_inst[G__asm_cp] = G__ADDSTROS;
      G__asm_inst[G__asm_cp+1] = G__store_struct_offset - store_inherit_offset;
      G__inc_cp_asm(2, 0);
   }
#endif
#endif
   /******************************************************************
    * C++ compiled function
    *******************************************************************/
   if (-1 == G__get_funcproperties(ifn)->entry.size) {
      // --
#ifdef G__ROOT
      if (memfunc_flag == G__CALLCONSTRUCTOR ||
            memfunc_flag == G__TRYCONSTRUCTOR  ||
            memfunc_flag == G__TRYIMPLICITCONSTRUCTOR) {
         G__exec_alloc_lock();
#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: ROOTOBJALLOCBEGIN\n" , G__asm_cp);
#endif
            G__asm_inst[G__asm_cp] = G__ROOTOBJALLOCBEGIN;
            G__inc_cp_asm(1, 0);
         }
#endif
      }
#endif
      G__call_cppfunc(result7, libp, ifn);
#ifdef G__ROOT
      if (memfunc_flag == G__CALLCONSTRUCTOR ||
            memfunc_flag == G__TRYCONSTRUCTOR  ||
            memfunc_flag == G__TRYIMPLICITCONSTRUCTOR) {
         G__exec_alloc_unlock();
#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: ROOTOBJALLOCEND\n" , G__asm_cp);
#endif
            G__asm_inst[G__asm_cp] = G__ROOTOBJALLOCEND;
            G__inc_cp_asm(1, 0);
         }
#endif
      }
#endif

      /* recover tag environment */
      G__store_struct_offset = store_inherit_offset;
      G__tagnum = store_inherit_tagnum;
      if (!G__tagnum.IsTopScope()) {
         G__incsetup_memvar(G__tagnum);
         if (G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)] &&
               strcmp(funcname, G__struct.name[G__get_tagnum(G__tagnum)]) == 0) {
            long *pvtag
            = (long*)(result7->obj.i + G__struct.virtual_offset[G__get_tagnum(G__tagnum)]);
            *pvtag = G__get_tagnum(G__tagnum);
         }
      }
      if ('P' == store_var_typeB) G__val2pointer(result7);
#ifdef G__NEWINHERIT
#ifdef G__ASM
      if (G__asm_noverflow && G__store_struct_offset &&
            G__store_struct_offset != store_inherit_offset) {
#ifdef G__ASM_DBG
         if (G__asm_dbg)
            G__fprinterr(G__serr, "%3x: ADDSTROS %ld\n"
                         , G__asm_cp, -(long)G__store_struct_offset + store_inherit_offset);
#endif
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = -(long)G__store_struct_offset + (long)store_inherit_offset;
         G__inc_cp_asm(2, 0);
      }
#endif /* ASM */
#endif /* NEWINHERIT */
      G__exec_memberfunc = store_exec_memberfunc;
      return(1);
   }
#ifdef G__ASM
   /******************************************************************
    * create bytecode instruction for calling interpreted function
    *******************************************************************/
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: LD_IFUNC '%s' paran: %d  %s:%d\n", G__asm_cp, G__asm_dt, funcname, libp->paran, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__LD_IFUNC;
      G__asm_inst[G__asm_cp+1] = 0; // used to be (long)ifn.Name().c_str();
      G__asm_inst[G__asm_cp+2] = hash;
      G__asm_inst[G__asm_cp+3] = libp->paran;
      G__asm_inst[G__asm_cp+4] = (long)ifn.Id();
      G__asm_inst[G__asm_cp+5] = (long)funcmatch;
      G__asm_inst[G__asm_cp+6] = (long)memfunc_flag;
      G__asm_inst[G__asm_cp+7] = (long)0;
      G__inc_cp_asm(8, 0);
      if (G__store_struct_offset && G__store_struct_offset != store_inherit_offset) {
#ifdef G__ASM_DBG
         if (G__asm_dbg)
            G__fprinterr(G__serr, "%3x: ADDSTROS %ld\n"
                         , G__asm_cp, -(long)G__store_struct_offset + store_inherit_offset);
#endif
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = -(long)G__store_struct_offset + (long)store_inherit_offset;
         G__inc_cp_asm(2, 0);
      }
   }
#endif /* G__ASM */
   /* G__oprovld is set when calling operator overload function after
    * evaluating its' argument to avoid duplication in p-code stack data.
    * This must be reset when calling lower level interpreted function */
   G__oprovld = 0;
#ifdef G__ASM
   if (G__no_exec_compile) {
      G__store_struct_offset = store_inherit_offset;
      G__tagnum = store_inherit_tagnum;
      G__value_typenum(*result7) = ifn.TypeOf().ReturnType();
      if (-1 != G__get_tagnum(G__value_typenum(*result7))) result7->ref = 1;
      else result7->ref = 0;
      result7->obj.d = 0.0;
      result7->obj.i = 1;
      if ('u' == G__get_type(G__value_typenum(*result7)) && 0 == result7->ref && -1 != G__get_tagnum(G__value_typenum(*result7))) {
         G__store_tempobject(*result7); /* To free tempobject in pcode */
      }
      /* To be implemented */
      G__exec_memberfunc = store_exec_memberfunc;
      return(1);
   }
#endif /* G__ASM */
   /******************************************************************
    * virtual function
    *  If virtual function flag is set, get actual tag identity by
    * taginfo member at offset of G__struct.virtual_offset[].
    * Then search for virtual function in actual tag. If found,
    * change p_ifunc,ifn,G__store_struct_offset and G__tagnum.
    * G__store_struct_offset and G__tagnum are already stored above,
    * so no need to store it to temporary here.
    *******************************************************************/
   if (ifn.IsVirtual() && !G__fixedscope) {
      if (G__PVOID != G__struct.virtual_offset[G__get_tagnum(G__tagnum)])
         virtualtag = G__Dict::GetDict().GetScope(*(long*)(G__store_struct_offset /* NEED TO CHECK THIS PART */
                      + (long)G__struct.virtual_offset[G__get_tagnum(G__tagnum)]));
      else {
         virtualtag = G__tagnum;
      }
      if (virtualtag != G__tagnum) {
         struct G__inheritance *baseclass = G__struct.baseclass[G__get_tagnum(virtualtag)];
         int xbase[G__MAXBASE], ybase[G__MAXBASE];
         int nxbase = 0, nybase;
         int basen;
         G__incsetup_memfunc((virtualtag));
         ::Reflex::Member iexist = G__ifunc_exist(ifn, virtualtag, true);
         for (basen = 0;!iexist && basen < baseclass->basen;basen++) {
            virtualtag = G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]);
            if (0 == (baseclass->property[basen]&G__ISDIRECTINHERIT)) continue;
            xbase[nxbase++] = G__get_tagnum(virtualtag);
            G__incsetup_memfunc((virtualtag));
            iexist
            = G__ifunc_exist(ifn, virtualtag, true);
         }
         while (!iexist && nxbase) {
            int xxx;
            nybase = 0;
            for (xxx = 0;!iexist && xxx < nxbase;xxx++) {
               baseclass = G__struct.baseclass[xbase[xxx]];
               for (basen = 0;!iexist && basen < baseclass->basen;basen++) {
                  virtualtag = G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]);
                  if (0 == (baseclass->property[basen]&G__ISDIRECTINHERIT)) continue;
                  ybase[nybase++] = G__get_tagnum(virtualtag);
                  G__incsetup_memfunc((virtualtag));
                  iexist
                  = G__ifunc_exist(ifn, virtualtag, true);
               }
            }
            nxbase = nybase;
            memcpy((void*)xbase, (void*)ybase, sizeof(int)*nybase);
         }
         if (iexist) {
            if (!G__get_funcproperties(iexist)->entry.p) {
               G__fprinterr(G__serr, "Error: virtual %s() header found but not defined", funcname);
               G__genericerror(0);
               G__exec_memberfunc = store_exec_memberfunc;
               return(1);
            }
            p_ifunc = iexist.DeclaringScope();
            ifn = iexist;
            G__store_struct_offset -= G__find_virtualoffset(G__get_tagnum(virtualtag));
            G__tagnum = virtualtag;
            if ('~' == funcname[0]) {
               strcpy(funcname + 1, G__struct.name[G__get_tagnum(G__tagnum)]);
               G__hash(funcname, hash, itemp);
            }
         }
         else if (ifn.IsAbstract()) {
            G__fprinterr(G__serr, "Error: pure virtual %s() not defined", funcname);
            G__genericerror(0);
            G__exec_memberfunc = store_exec_memberfunc;
            return(1);
         }
      }
   }
#ifdef G__ASM
#ifdef G__ASM_WHOLEFUNC
   /******************************************************************
    * try bytecode compilation
    *******************************************************************/
   if (G__BYTECODE_NOTYET == G__get_funcproperties(ifn)->entry.bytecodestatus &&
         G__asm_loopcompile > 3 && G__ASM_FUNC_NOP == G__asm_wholefunction &&
#ifndef G__TO_BE_DELETED
         G__CALLCONSTRUCTOR != memfunc_flag && G__TRYCONSTRUCTOR != memfunc_flag &&
         G__TRYIMPLICITCONSTRUCTOR != memfunc_flag &&
         G__TRYDESTRUCTOR != memfunc_flag &&
#endif
         0 == G__step && (G__asm_noverflow || G__asm_exec
                          || G__asm_loopcompile > 4
                         )) {
      G__compile_function_bytecode(ifn);
   }
   /******************************************************************
    * if already compiled as bytecode run bytecode
    *******************************************************************/
   if (G__get_funcproperties(ifn)->entry.bytecode
         && G__BYTECODE_ANALYSIS != G__get_funcproperties(ifn)->entry.bytecodestatus
      ) {
      struct G__input_file store_ifile;
      store_ifile = G__ifile;
      G__ifile.filenum = G__get_funcproperties(ifn)->entry.filenum;
      G__ifile.line_number = G__get_funcproperties(ifn)->entry.line_number;
      G__exec_bytecode(result7, (char*)G__get_funcproperties(ifn)->entry.bytecode, libp, hash);
      G__ifile = store_ifile;
      G__tagnum = store_inherit_tagnum;
      G__store_struct_offset = store_inherit_offset;
      return(1);
   }
#endif /* G__ASM_WHOLEFUNC */
#endif /* G__ASM */
#ifndef G__OLDIMPLEMENTATION1167
   G__reftypeparam(ifn, libp);
#endif
#ifdef G__ASM
#ifdef G__ASM_IFUNC
   /******************************************************************
    * push bytecode environment stack
    *******************************************************************/
   /* Push loop compilation environment */
   store_asm_inst = G__asm_inst;
   store_asm_stack = G__asm_stack;
   store_asm_name = G__asm_name;
   store_asm_name_p = G__asm_name_p;
   store_asm_param  = G__asm_param;
   store_asm_exec  = G__asm_exec;
   store_asm_noverflow  = G__asm_noverflow;
   store_asm_cp  = G__asm_cp;
   store_asm_dt  = G__asm_dt;
   store_asm_index  = G__asm_index;
   store_asm_instsize = G__asm_instsize;
   G__asm_instsize = 0; /* G__asm_inst is not resizable */
   G__asm_inst = asm_inst_g;
   G__asm_stack = asm_stack_g;
   G__asm_name = asm_name;
   G__asm_name_p = 0;
   /* G__asm_param ; */
   G__asm_exec = 0;
#endif /* G__ASM_IFUNC */
#endif /* G__ASM */
#ifdef G__ASM
#ifdef G__ASM_IFUNC
#ifdef G__ASM_WHOLEFUNC
   /******************************************************************
    * bytecode function compilation start
    *******************************************************************/
   if (G__ASM_FUNC_COMPILE&G__asm_wholefunction) {
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "!!!bytecode compilation %s start"
                      , ifn.Name().c_str());
         G__printlinenum();
      }
      G__asm_name = (char*)malloc(G__ASM_FUNCNAMEBUF);
      G__asm_noverflow = 1;
      store_no_exec_compile = G__no_exec_compile;
      G__no_exec_compile = 1;

      // Initialize localvar (if anything is needed).
      localvar = G__create_scope(ifn, G__p_local, libp);
   }
   else {
      G__asm_noverflow = 0;
   }
#else /* G__ASM_WHOLEFUNC */
   G__asm_noverflow = 0;
#endif /* G__ASM_WHOLEFUNC */
   G__clear_asm();
#endif /* G__ASM_IFUNC */
#endif /* G__ASM */
   /******************************************************************
    * G__exec_memberfunc and G__memberfunc_tagnum are stored in one
    * upper level G__getfunction() and G__parenthesisovld() and restored
    * when exit from these functions.
    *******************************************************************/
   if (-1 == G__get_tagnum(p_ifunc) || p_ifunc.IsTopScope()) {
      // Note: We should try with:  if (p_ifunc.IsClass())
      G__exec_memberfunc = 0;
   }
   else {
      G__exec_memberfunc = 1;
   }
   G__setclassdebugcond(G__get_tagnum(G__tagnum), 0);
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   G__memberfunc_tagnum = G__tagnum;
   G__memberfunc_struct_offset = G__store_struct_offset;
   /**********************************************
    * If return value is struct,class,union,
    * create temp object buffer
    **********************************************/
   if (!ifn.IsConstructor()) { 
      ::Reflex::Type rtype( ifn.TypeOf().ReturnType() );
      if (G__get_type(rtype) == 'u' 
         && G__get_reftype(rtype) == G__PARANORMAL
         && G__CPPLINK != G__struct.iscpplink[G__get_tagnum(rtype)]
         && 'e' != G__get_tagtype(rtype)
         ) {
            /* create temp object buffer */

            G__alloc_tempobject(G__get_tagnum(ifn.TypeOf().ReturnType()) , G__get_typenum(ifn.TypeOf().ReturnType()));
            store_p_tempobject = &G__p_tempbuf->obj;

            if (G__dispsource) {
               G__fprinterr(G__serr, "!!!Create temp object (%s)0x%lx,%d for %s() return\n"
                  , ifn.TypeOf().ReturnType().Name().c_str()
                  , G__p_tempbuf->obj.obj.i , G__templevel , ifn.Name().c_str());
            }
         }
   }
   /**********************************************
    * increment busy flag
    **********************************************/
   G__get_funcproperties(ifn)->entry.busy++;
   /**********************************************
    * set global variable G__func_now. This is
    * used in G__malloc() to allocate static
    * variable.
    **********************************************/
   store_func_now = G__func_now;
   G__func_now = ifn;
   /* store old local to prev buffer and allocate new local variable */
   G_local = G__create_scope(ifn, G__p_local, libp);
#ifdef G__ASM_WHOLEFUNC
   if (G__ASM_FUNC_COMPILE&G__asm_wholefunction) {
      G__p_local = localvar;
   }
   else {
      G__p_local = G_local;
   }
#else
   G__p_local = G_local;
#endif
   break_exit_func = G__break_exit_func;
   G__break_exit_func = 0;
   /* store line number and filename*/
   G__ifile.line_number = G__get_funcproperties(ifn)->entry.line_number;
   G__ifile.filenum = G__get_funcproperties(ifn)->entry.filenum;
   if (G__ifile.filenum>=0) {
      strcpy(G__ifile.name, G__srcfile[G__ifile.filenum].filename);
   } else {
      strcpy(G__ifile.name, "unknown");
   }
   /* check breakpoint */
   if (0 == G__nobreak && 0 == G__no_exec_compile &&
         G__srcfile[G__ifile.filenum].breakpoint &&
         G__srcfile[G__ifile.filenum].maxline > G__ifile.line_number &&
         G__TESTBREAK&(G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number] |= G__TRACED)) {
      G__BREAKfgetc();
   }
   store_security = G__security;
   G__security = G__srcfile[G__ifile.filenum].security;
   /* store file pointer and fpos*/
   /* store_linenumber=G__ifile.line_number; */
   if (G__ifile.fp) fgetpos(G__ifile.fp, &prev_pos);
   prev_fp = G__ifile.fp;
   /* Find the right file pointer */
   if (G__mfp && (FILE*)G__get_funcproperties(ifn)->entry.p == G__mfp) {
      /* In case of macro expanded by cint, we use the tmpfile */
      G__ifile.fp = (FILE*)G__get_funcproperties(ifn)->entry.p;
   }
   else if (G__srcfile[G__ifile.filenum].fp) {
      /* The file is already open use that */
      G__ifile.fp = G__srcfile[G__ifile.filenum].fp;
   }
   else {
      /* The file had been closed, let's reopen the proper file
       * resp from the preprocessor and raw */
      if (G__srcfile[G__ifile.filenum].prepname) {
         G__ifile.fp = fopen(G__srcfile[G__ifile.filenum].prepname, "r");
      }
      else {
         G__ifile.fp = fopen(G__srcfile[G__ifile.filenum].filename, "r");
      }
      G__srcfile[G__ifile.filenum].fp =  G__ifile.fp;
      if (!G__ifile.fp) G__ifile.fp = (FILE*)G__get_funcproperties(ifn)->entry.p;
   }
   fsetpos(G__ifile.fp, &G__get_funcproperties(ifn)->entry.pos);
   /* print function header if debug mode */
   if (G__dispsource) {
      G__disp_mask = 0;
      if ((G__debug || G__break || G__step
            || (strcmp(G__breakfile, G__ifile.name) == 0) || (strcmp(G__breakfile, "") == 0)
          ) && ((G__prerun != 0) || (G__no_exec == 0))) {
         if (G__ifile.name && G__ifile.name[0])
            G__fprinterr(G__serr, "\n# %s", G__ifile.name);
         if (-1 != G__get_tagnum(p_ifunc) && !p_ifunc.IsTopScope()) {
            G__fprinterr(G__serr, "\n%-5d%s::%s(" , G__ifile.line_number
                         , p_ifunc.Name().c_str(), funcname);
         }
         else {
            G__fprinterr(G__serr, "\n%-5d%s(" , G__ifile.line_number, funcname);
         }
      }
   }
   // now came to         func( para1,para2,,,)
   //                          ^
   store_doingconstruction = G__doingconstruction;
   if (G__get_funcproperties(ifn)->entry.ansi == 0) {
      /**************************************************************
       * K&R C
       *
       *   func( para1 ,para2 ,,, )
       *        ^
       **************************************************************/
      /* read pass parameters , standard C */
      ipara = 0;
      while (cin != ')') {
#ifndef G__OLDIMPLEMENTATION1802
         G__StrBuf temp_sb(G__ONELINE);
         char *temp = temp_sb;
#endif
         cin = G__fgetstream(temp, ",)");
         if (temp[0] != '\0') {
            strcpy(paraname[ipara], temp);
            ipara++;
         }
      }

      /* read and exec pass parameter declaration , standard C
       * G__exec_statement(&brace_level); returns at '{' if G__funcheader==1
       */
      G__funcheader = 1;
      do {
         int brace_level = 0;
         buf = G__exec_statement(&brace_level);
      }
      while (!G__value_typenum(buf) /* buf.type==G__null.type */ && G__return < G__RETURN_EXIT1);

      /* set pass parameters , standard C
       * Parameters can be constant. When G__funcheader==1,
       * error message of changing const doesn't appear.
       */

      for (itemp = 0;itemp < (int)ipara;itemp++) {
         G__letvariable(paraname[itemp], libp->para[itemp]
                        ,::Reflex::Scope::GlobalScope(), G__p_local);
      }

      G__funcheader = 0;
      /* fsetpos(G__ifile.fp,&temppos) ; */
      /* G__ifile.line_number=store_linenumber; */
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) G__disp_mask = 1;
   }
   else {
      /**************************************************************
       * ANSI C
       *
       *   type func( type para1, type para2 ,,, )
       *             ^
       **************************************************************/
      G__value store_ansipara;
      store_ansipara = G__ansipara;
      G__ansiheader = 1;
      G__funcheader = 1;

      ipara = 0;
      while (G__ansiheader != 0 && G__return < G__RETURN_EXIT1) {
         /****************************************
          * for default parameter
          ****************************************/
         /****************************************
          * if parameter exists, set G__ansipara
          ****************************************/
         if ((int)ipara < libp->paran) {
            G__ansipara = libp->para[ipara];
            /* assigning reference for fundamental type reference argument */
            if (0 == G__ansipara.ref) {
               switch (G__get_type(ifn.TypeOf().FunctionParameterAt(ipara))) {
                  case 'f':
                     G__Mfloat(libp->para[ipara]);
                     G__value_typenum(libp->para[ipara]) = ifn.TypeOf().FunctionParameterAt(ipara);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.fl);
                     break;
                  case 'd':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.d);
                     break;
                  case 'c':
                     G__Mchar(libp->para[ipara]);
                     G__value_typenum(libp->para[ipara]) = ifn.TypeOf().FunctionParameterAt(ipara);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ch);
                     break;
                  case 's':
                     G__Mshort(libp->para[ipara]);
                     G__value_typenum(libp->para[ipara]) = ifn.TypeOf().FunctionParameterAt(ipara);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.sh);
                     break;
                  case 'i':
                     G__Mint(libp->para[ipara]);
                     G__value_typenum(libp->para[ipara]) = ifn.TypeOf().FunctionParameterAt(ipara);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.in);
                     break;
                  case 'l':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
                     break;
                  case 'b':
                  case 'g':
#ifdef G__BOOL4BYTE
                     G__Mint(libp->para[ipara]);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
#else
                     G__Muchar(libp->para[ipara]);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.uch);
#endif
                     G__value_typenum(libp->para[ipara]) = ifn.TypeOf().FunctionParameterAt(ipara);
                     break;
                  case 'r':
                     G__Mushort(libp->para[ipara]);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ush);
                     G__value_typenum(libp->para[ipara]) = ifn.TypeOf().FunctionParameterAt(ipara);
                     break;
                  case 'h':
                     /* G__Muint(libp->para[ipara]); */
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
                     break;
                  case 'k':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
                     break;
                  case 'u':
                     G__ansipara.ref = G__ansipara.obj.i;
                     break;
                  case 'n':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ll);
                     break;
                  case 'm':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ull);
                     break;
                  case 'q':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ld);
                     break;
                  default:
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
                     break;
               }
            }
         }
         /****************************************
          * if not, set null.
          * Default value will be used
          *  type func(type paraname=default,...)
          ****************************************/
         else {
            if (
               ifn.FunctionParameterSize() > ipara &&
               ifn.FunctionParameterDefaultAt(ipara).c_str()[0]) {
               if (G__get_type(ifn.TypeOf().FunctionParameterAt(ipara)) == G__DEFAULT_FUNCCALL) {
                  G__ASSERT(G__get_funcproperties(ifn)->entry.para_default[ipara]->ref);
                  *G__get_funcproperties(ifn)->entry.para_default[ipara] = G__getexpr((char*) G__get_funcproperties(ifn)->entry.para_default[ipara]->ref);
                  G__ansiheader = 1;
                  G__funcheader = 1;
#define G__OLDIMPLEMENTATION1558
               }
               G__ansipara = *G__get_funcproperties(ifn)->entry.para_default[ipara];
            }
            else
               G__ansipara = G__null;
         }
         G__refansipara = libp->parameter[ipara];

         if (G__ASM_FUNC_COMPILE == G__asm_wholefunction &&
               !G__get_funcproperties(ifn)->entry.para_default.empty()
               && G__get_funcproperties(ifn)->entry.para_default.size() > ipara
               && G__get_funcproperties(ifn)->entry.para_default.at(ipara)) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: ISDEFAULTPARA %x\n", G__asm_cp, G__asm_cp + 4);
               G__fprinterr(G__serr, "%3x: LD %ld %g\n", G__asm_cp + 2
                            , (G__get_funcproperties(ifn)->entry.para_default)[ipara]->obj.i
                            , (G__get_funcproperties(ifn)->entry.para_default)[ipara]->obj.d
                           );
            }
#endif
            G__asm_inst[G__asm_cp] = G__ISDEFAULTPARA;
            G__asm_wholefunc_default_cp = G__asm_cp + 1;
            G__inc_cp_asm(2, 0);

            /* set default param in stack */
            G__asm_inst[G__asm_cp] = G__LD;
            G__asm_inst[G__asm_cp+1] = G__asm_dt;
            G__asm_stack[G__asm_dt] = *G__get_funcproperties(ifn)->entry.para_default[ipara];
            G__inc_cp_asm(2, 1);

            G__asm_inst[G__asm_wholefunc_default_cp] = G__asm_cp;
            G__asm_noverflow = 0;
            int brace_level = 0;
            G__exec_statement(&brace_level); // Create var entry and ST_LVAR inst.
            //{
            //   char buf[128];
            //   G__fgetstream_peek(buf, 30);
            //   fprintf(stderr, "G__interpret_func: finished default param, peek ahead: '%s'\n", buf);
            //}
            G__asm_wholefunc_default_cp = 0;
            G__asm_noverflow = 1;
         }
         else {
            int brace_level = 0;
            G__exec_statement(&brace_level);
         }
         ipara++;
      }

      G__funcheader = 0;

      switch (memfunc_flag) {
         case G__CALLCONSTRUCTOR:
         case G__TRYCONSTRUCTOR:
#ifndef G__OLDIMPLEMENTATIO1250
         case G__TRYIMPLICITCONSTRUCTOR:
#endif
            /* read parameters for base constructors and
             * constructor for base calss and class members
             * maybe with some parameters
             */
            G__baseconstructorwp();
            G__doingconstruction = 1;
      }

      G__ansipara = store_ansipara;
   }
#ifdef G__SECURITY
   if ((G__security&G__SECURE_STACK_DEPTH) &&
         G__max_stack_depth &&
         G__calldepth > G__max_stack_depth
      ) {
      G__fprinterr(G__serr, "Error: Stack depth exceeded %d", G__max_stack_depth);
      G__genericerror(0);
      G__pause();
      G__return = G__RETURN_EXIT1;
   }
   if (G__return > G__RETURN_EXIT1) {
      G__exec_memberfunc = store_exec_memberfunc;
      G__security = store_security;
      G__memberfunc_tagnum = store_memberfunc_tagnum;
      G__memberfunc_struct_offset = store_memberfunc_struct_offset;
      return(1);
   }
#endif
   G__setclassdebugcond(G__get_tagnum(G__memberfunc_tagnum), 1);
   /**************************************************************
    * execute ifunction body
    *
    * common to standard and ANSI
    **************************************************************/
   store_iscpp = G__iscpp;
   G__iscpp = G__get_funcproperties(ifn)->iscpplink;
   ++G__templevel;
   ++G__calldepth;
#ifdef G__ASM_DBG
   if (G__istrace > 1) {
      if (G__istrace > G__templevel) {
         G__debug = 1;
         G__asm_dbg = 1;
      }
      else {
         G__debug = 0;
         G__asm_dbg = 0;
      }
   }
#endif
   G__ASSERT(0 == G__decl || 1 == G__decl);
   store_def_struct_member = G__def_struct_member;
   G__def_struct_member = 0;
   store_decl = G__decl;
   G__decl = 0;
   G__no_exec = 0;
   // 
   // Execute Body.
   int brace_level = 0;
   *result7 = G__exec_statement(&brace_level);
   G__decl = store_decl;
   G__def_struct_member = store_def_struct_member;
   G__ASSERT(0 == G__decl || 1 == G__decl);
   if (G__RETURN_IMMEDIATE == G__return &&
                                        G__get_type(G__interactivereturnvalue) && '\0' == G__get_type(G__value_typenum(*result7))) {
      *result7 = G__interactivereturnvalue;
      G__interactivereturnvalue = G__null;
   }
   --G__calldepth;
   --G__templevel;
#ifdef G__ASM_DBG
   if (G__istrace > 1) {
      if (G__istrace > G__templevel) {
         G__debug = 1;
         G__asm_dbg = 1;
      }
      else {
         G__debug = 0;
         G__asm_dbg = 0;
      }
   }
#endif
   G__iscpp = (short)store_iscpp;
   G__doingconstruction = store_doingconstruction;
   /**************************************************************
    * Error if goto label not found
    **************************************************************/
   if (G__gotolabel[0]) {
      G__fprinterr(G__serr, "Error: Goto label '%s' not found in %s()", G__gotolabel, funcname);
      G__genericerror(0);
      G__gotolabel[0] = '\0';
   }
   /**************************************************************
    * return value type conversion
    **************************************************************/
   if (
      !G__xrefflag &&
      (G__get_type(G__value_typenum(*result7)) != '\0') &&
      (G__return != G__RETURN_EXIT1) &&
      ((G__asm_wholefunction == G__ASM_FUNC_NOP) || G__asm_noverflow)
   ) {
      char rtype = G__get_type(ifn.TypeOf().ReturnType());
      if (ifn.IsConstructor()) {
         rtype = 'i';
      }
      switch (rtype) {
         case 'd': /* double */
         case 'f': /* float */
         case 'w': /* logic (original type) */
            G__letdouble(result7, G__get_type(ifn.TypeOf().ReturnType()) , G__double(*result7));
#define G__OLDIMPLEMENTATION753
#ifdef G__OLDIMPLEMENTATION753
            if (G__get_reftype(ifn.TypeOf().ReturnType()) == G__PARANORMAL) result7->ref = 0;
#endif // G__OLDIMPLEMENTATION753
#ifndef G__OLDIMPLEMENTATION1259
            if (ifn.TypeOf().ReturnType().FinalType().IsConst()) {
               G__value_typenum(*result7) = G__modify_type(G__value_typenum(*result7), 0, 0, G__get_isconst(G__value_typenum(*result7)), 0, 0);
            }
#endif // G__OLDIMPLEMENTATION1259
            break;
         case 'n':
         case 'm':
            G__letLonglong(result7, G__get_type(ifn.TypeOf().ReturnType()), G__Longlong(*result7));
            if (G__get_reftype(ifn.TypeOf().ReturnType()) == G__PARANORMAL) {
               result7->ref = 0;
            }
            if (ifn.TypeOf().ReturnType().FinalType().IsConst()) {
               G__value_typenum(*result7) = G__modify_type(G__value_typenum(*result7), 0, 0, G__get_isconst(G__value_typenum(*result7)), 0, 0);
            }
            break;
         case 'q':
            G__letLongdouble(result7, G__get_type(ifn.TypeOf().ReturnType()), G__Longdouble(*result7));
            if (G__get_reftype(ifn.TypeOf().ReturnType()) == G__PARANORMAL) {
               result7->ref = 0;
            }
            if (ifn.TypeOf().ReturnType().FinalType().IsConst()) {
               G__value_typenum(*result7) = G__modify_type(G__value_typenum(*result7), 0, 0, G__get_isconst(G__value_typenum(*result7)), 0, 0);
            }
            break;
         case 'g':
            G__letint(result7, G__get_type(ifn.TypeOf().ReturnType()), G__int(*result7) ? 1 : 0);
#ifdef G__OLDIMPLEMENTATION753
            if (G__get_reftype(ifn.TypeOf().ReturnType()) == G__PARANORMAL) {
               result7->ref = 0;
            }
#endif // G__OLDIMPLEMENTATION753
            if (ifn.TypeOf().ReturnType().FinalType().IsConst()) {
               G__value_typenum(*result7) = G__modify_type(G__value_typenum(*result7), 0, 0, G__get_isconst(G__value_typenum(*result7)), 0, 0);
            }
            break;
         case 'y': /* void */
            /***************************************************
             * in case of void, if return(); statement exists
             * it is illegal.
             * Maybe bug if return; statement exists without
             * return value.
             ***************************************************/
            if (G__RETURN_NORMAL == G__return) {
               if (G__dispmsg >= G__DISPWARN) {
                  G__fprinterr(G__serr, "Warning: Return value of void %s() ignored", ifn.Name().c_str());
                  G__printlinenum();
               }
            }
            *result7 = G__null;
            break;
         case 'u': /* struct, union, class */
            /***************************************************
             * result7 contains pointer to the local variable
             * which will be destroyed right after this.
             ***************************************************/
            {
#ifndef G__OLDIMPLEMENTATION1802
            G__StrBuf temp_sb(G__ONELINE);
            char *temp = temp_sb;
#endif // G__OLDIMPLEMENTATION1802
            /* don't call copy constructor if returning reference type */
            if (G__PARANORMAL != G__get_reftype(ifn.TypeOf().ReturnType())) {
               if (ifn.TypeOf().ReturnType().RawType() != G__value_typenum(*result7).RawType()) {
                  int offset = G__ispublicbase(ifn.TypeOf().ReturnType().RawType(), G__value_typenum(*result7).RawType(), (void*)result7->obj.i);
                  if (-1 == offset) {
                     G__fprinterr(G__serr, "Error: Return type mismatch. %s ", ifn.TypeOf().ReturnType().Name(::Reflex::SCOPED).c_str());
                     G__fprinterr(G__serr, "not a public base of %s", G__value_typenum(*result7).RawType().Name(::Reflex::SCOPED).c_str());
                     G__genericerror(0);
                     G__value_typenum(*result7) = G__replace_rawtype(G__value_typenum(*result7), ifn.TypeOf().ReturnType().RawType());
                     break;
                  }
                  else {
                     result7->obj.i += offset;
                     if (result7->ref) {
                        result7->ref += offset;
                     }
                     G__value_typenum(*result7) = G__replace_rawtype(G__value_typenum(*result7), ifn.TypeOf().ReturnType().RawType());
                  }
               }
               break;
            }
            if ('e' == G__get_tagtype(ifn.TypeOf().ReturnType())) {
               break;
            }
            if (G__get_type(*result7) == 'u' || (G__get_type(*result7) == 'i' && -1 != G__get_tagnum(G__value_typenum(*result7)))) {
               if (result7->obj.i < 0) {
                  sprintf(temp, "%s((%s)(%ld))", ifn.TypeOf().ReturnType().RawType().Name().c_str(), G__value_typenum(*result7).RawType().Name(::Reflex::SCOPED).c_str() , result7->obj.i);
               }
               else {
                  sprintf(temp, "%s((%s)%ld)", ifn.TypeOf().ReturnType().RawType().Name().c_str(), G__value_typenum(*result7).RawType().Name(::Reflex::SCOPED).c_str() , result7->obj.i);
               }
            }
            else {
               G__StrBuf buf2_sb(G__ONELINE);
               char *buf2 = buf2_sb;
               G__valuemonitor(*result7, buf2);
               sprintf(temp, "%s(%s)", ifn.TypeOf().ReturnType().RawType().Name().c_str(), buf2);
            }
            store_tagnum = G__tagnum;
            G__tagnum = ifn.TypeOf().ReturnType().RawType();
            store_var_type = G__var_type;
#ifdef G__SECURITY
            G__castcheckoff = 1;
#endif // G__SECURITY
            if (G__RETURN_IMMEDIATE >= G__return) {
               G__return = G__RETURN_NON;
            }
            itemp = 0;
            store_struct_offset = G__store_struct_offset;
            if (G__CPPLINK != G__struct.iscpplink[G__get_tagnum(G__tagnum)]) {
               // -- Interpreted class.
               if (store_p_tempobject) {
                  G__store_struct_offset = (char*)store_p_tempobject->obj.i;
               }
               else {
                  G__store_struct_offset = (char*)G__p_tempbuf->obj.obj.i;
               }
               if (G__dispsource) {
                  G__fprinterr(G__serr, "\n!!!Calling copy/conversion constructor for return temp object 0x%lx.%s", G__store_struct_offset, temp);
               }
               G__getfunction(temp, &itemp, G__TRYCONSTRUCTOR);
               if (
                  itemp &&
                  (store_p_tempobject != &G__p_tempbuf->obj) &&
                  (store_struct_offset != (char*) G__p_tempbuf->obj.obj.i)
               ) {
                  ++G__p_tempbuf->level;
                  ++G__templevel;
                  G__free_tempobject();
                  --G__templevel;
               }
            }
            else {
               // -- Precompiled class.
               char* store_globalvarpointer = G__globalvarpointer;
               G__globalvarpointer = G__PVOID;
               G__store_struct_offset = (char*) 0xffff;
               if (G__dispsource) {
                  G__fprinterr(G__serr, "\n!!!Calling copy/conversion constructor for return temp object 0x%lx.%s", G__store_struct_offset, temp);
               }
               buf = G__getfunction(temp, &itemp, G__TRYCONSTRUCTOR);
               G__globalvarpointer = store_globalvarpointer;
               if (itemp) {
                  G__free_tempobject();
                  G__store_tempobject(buf);
               }
#ifdef G__ASM
               // It is not needed to explicitly create STORETEMP instruction
               // because it is preincluded in the compiled funciton call
               // interface.
#endif // G__ASM
               if (G__dispsource) {
                  G__fprinterr(G__serr, "!!!Create temp object (%s)0x%lx,%d for %s() return\n", G__struct.name[G__get_tagnum(G__tagnum)], G__p_tempbuf->obj.obj.i, G__templevel, ifn.Name().c_str());
               }
            }
            G__store_struct_offset = store_struct_offset;
            G__tagnum = store_tagnum;
            G__var_type = store_var_type;
            // If no copy constructor, memberwise copy.
            if (!itemp && !G__xrefflag) {
               long offset = 0;
               if (G__get_tagnum(G__value_typenum(*result7)) == G__get_tagnum(ifn.TypeOf().ReturnType())) {
                  if (store_p_tempobject) {
                     memcpy((void*) store_p_tempobject->obj.i, (void*) result7->obj.i, (size_t) G__struct.size[G__get_tagnum(G__value_typenum(*result7))]);
                  }
                  else {
                     memcpy((void*) G__p_tempbuf->obj.obj.i, (void*) result7->obj.i, (size_t) G__struct.size[G__get_tagnum(G__value_typenum(*result7))]);
                  }
               }
               else if (
                  -1 != (offset = G__ispublicbase(ifn.TypeOf().ReturnType().RawType(), G__value_typenum(*result7).RawType(), (void*) result7->obj.i))
               ) {
                  sprintf(temp, "%s((%s)(%ld))", ifn.TypeOf().ReturnType().RawType().Name().c_str(), ifn.TypeOf().ReturnType().Name(::Reflex::SCOPED).c_str(), result7->obj.i + offset);
                  if (G__CPPLINK != G__struct.iscpplink[G__get_tagnum(G__tagnum)]) {
                     // -- Interpreted class.
                     if (store_p_tempobject) {
                        G__store_struct_offset = (char*)store_p_tempobject->obj.i;
                     }
                     else {
                        G__store_struct_offset = (char*)G__p_tempbuf->obj.obj.i;
                     }
                     G__getfunction(temp, &itemp, G__TRYCONSTRUCTOR);
                  }
                  else {
                     // -- Precompiled class.
                     G__store_struct_offset = (char*) 0xffff;
                     buf = G__getfunction(temp, &itemp, G__TRYCONSTRUCTOR);
                     if (itemp) {
                        G__store_tempobject(buf);
                     }
                  }
               }
            }
            if (store_p_tempobject) {
               *result7 = *store_p_tempobject;
            }
            else {
               *result7 = G__p_tempbuf->obj;
            }
#ifndef G__OLDIMPLEMENTATION1259
            if (ifn.TypeOf().ReturnType().FinalType().IsConst()) {
               G__value_typenum(*result7) = G__modify_type(G__value_typenum(*result7), 0, 0, G__get_isconst(G__value_typenum(*result7)), 0, 0);
            }
#endif // G__OLDIMPLEMENTATION1259
            break;
            }
         case 'i':
            // -- Return value of constructor.
            if (-1 != G__get_tagnum(ifn.TypeOf().ReturnType())) {
               if (
                  (G__CPPLINK != G__struct.iscpplink[G__get_tagnum(ifn.TypeOf().ReturnType())]) &&
                  ('e' != G__get_tagtype(ifn.TypeOf().ReturnType())) &&
                  G__store_struct_offset &&
                  (1 != (long)G__store_struct_offset)
               ) {
                  result7->obj.i = (long) G__store_struct_offset;
               }
               result7->ref = result7->obj.i;
#ifndef G__OLDIMPLEMENTATION1259
               if (ifn.TypeOf().ReturnType().FinalType().IsConst()) {
                  G__value_typenum(*result7) = G__modify_type(G__value_typenum(*result7), 0, 0, G__get_isconst(G__value_typenum(*result7)), 0, 0);
               }
#endif // G__OLDIMPLEMENTATION1259
               break;
            }
         case 'U':
            if ('U' == G__get_type(ifn.TypeOf().ReturnType()) && 'U' == G__get_type(*result7)) {
               if (ifn.TypeOf().ReturnType().RawType() != G__value_typenum(*result7).RawType()) {
                  int offset = G__ispublicbase(ifn.TypeOf().ReturnType().RawType(), G__value_typenum(*result7).RawType(), (void*)result7->obj.i);
                  if (-1 == offset) {
                     G__fprinterr(G__serr, "Error: Return type mismatch. %s ", ifn.TypeOf().ReturnType().Name(::Reflex::SCOPED).c_str());
                     G__fprinterr(G__serr, "not a public base of %s", G__value_typenum(*result7).RawType().Name(::Reflex::SCOPED).c_str());
                     G__genericerror(0);
                     G__value_typenum(*result7) = G__replace_rawtype(G__value_typenum(*result7), ifn.TypeOf().ReturnType().RawType());
                     break;
                  }
                  else {
                     result7->obj.i += offset;
                     if (result7->ref) {
                        result7->ref += offset;
                     }
                     G__value_typenum(*result7) = G__replace_rawtype(G__value_typenum(*result7), ifn.TypeOf().ReturnType().RawType());
                  }
               }
            }
            /* no break, this case continues to default: */
            /***************************************************
             * Everything else is returned as integer. This
             * includes char,short,int,long,unsigned version
             * of them, pointer and struct/union.
             * If return value is struct/union, malloced memory
             * area will be freed about 20 lines below by
             * G__destroy_upto(). To prevent any data loss, memory
             * area has to be copied to left hand side memory
             * area of assignment (or temp buffer of expression
             * parser which doesn't exist in this version).
             ***************************************************/
         default:
#ifdef G__SECURITY
            if (isupper(G__get_type(ifn.TypeOf().ReturnType())) && islower(G__get_type(*result7)) && result7->obj.i && 0 == G__asm_wholefunction) {
               G__fprinterr(G__serr, "Error: Return type mismatch %s()", ifn.Name().c_str());
               G__genericerror(0);
               break;
            }
#endif // G__SECURITY
            G__letint(result7, G__get_type(ifn.TypeOf().ReturnType()), G__int(*result7));
#ifdef G__OLDIMPLEMENTATION753
            if (G__get_reftype(ifn.TypeOf().ReturnType()) == G__PARANORMAL) {
               result7->ref = 0;
            }
#endif // G__OLDIMPLEMENTATION753
            G__value_typenum(*result7) = G__modify_type(G__value_typenum(*result7), 0, G__get_reftype(ifn.TypeOf().ReturnType()), G__get_isconst(G__value_typenum(*result7)), 0, 0);
#ifdef G__SECURITY
            if (isupper(G__get_type(*result7)) && (G__security & G__SECURE_GARBAGECOLLECTION) && (!G__no_exec_compile)) {
               // -- Add reference count to avoid garbage collection when pointer is returned.
               G__add_refcount((void*) result7->obj.i, 0);
            }
#endif // G__SECURITY
            break;
      }
   }
   if (G__RETURN_EXIT1 != G__return) { /* if not exit */
      // return struct and typedef identity
      G__value_typenum(*result7) = G__replace_rawtype(G__value_typenum(*result7), ifn.TypeOf().ReturnType().RawType());
   }
   if (G__RETURN_TRY != G__return) {
      G__no_exec = 0;
      /**************************************************************
       * reset return flag
       * G__return is set to 1 if interpreted function returns by
       * return(); statement.  Until G__return is reset to 0,
       * execution flow exits from G__exec_statment().
       **************************************************************/
      if (G__RETURN_IMMEDIATE >= G__return) {
         G__return = G__RETURN_NON;
      }
   }
#ifndef G__NEWINHERIT
   // recover prev local variable
   G__p_local = G_local.prev_local; // same as G__p_local->prev_local
#endif // G__NEWINHERIT
#ifdef G__ASM_WHOLEFUNC
   /**************************************************************
    * whole function bytecode compile end
    **************************************************************/
   if (G__ASM_FUNC_COMPILE&G__asm_wholefunction) {
      if (G__NOERROR != G__security_error) {
         G__resetbytecode();
      }
      if (G__asm_noverflow) {
         int pc = 0;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: RTN_FUNC  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__RTN_FUNC;
         G__asm_inst[G__asm_cp+1] = 0;
         G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: RETURN  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__RETURN;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "Bytecode compilation of '%s' successful", ifn.Name().c_str());
            G__printlinenum();
         }
#endif // G__ASM_DBG
         G__resolve_jumptable_bytecode();
         if (G__asm_loopcompile >= 2) {
            G__asm_optimize(&pc);
         }
         G__resetbytecode();
         G__no_exec_compile = store_no_exec_compile;
         G__asm_storebytecodefunc(ifn, localvar, G__asm_stack, G__asm_dt, G__asm_inst, G__asm_cp);
      }
      else {
         /* destroy temp object, before restoreing G__no_exec_compile */
         if (G__p_tempbuf->level >= G__templevel && G__p_tempbuf->prev) {
            G__free_tempobject();
         }
         free(G__asm_name);
         G__resetbytecode();
         G__no_exec_compile = store_no_exec_compile;
         /* destroy local memory area */
         localvar.Unload();
         if (G__asm_dbg) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr,
                            "Warning: Bytecode compilation of %s failed. Maybe slow"
                            , ifn.Name().c_str());
               G__printlinenum();
            }
         }
         if (G__return >= G__RETURN_IMMEDIATE) G__return = G__RETURN_NON;
         G__security_error = G__NOERROR;
      }
   }
   else {
      /**************************************************************
       * destroy malloced local memory area
       **************************************************************/
      int store_security_error = G__security_error;
      G__security_error = 0;
      //G_local.Unload();
      G__destroy_upto(G_local, G__LOCAL_VAR, -1);
      G__security_error = store_security_error;
   }
#else /* G__ASM_WHOLEFUNC */
   /**************************************************************
   * destroy malloced local memory area
   **************************************************************/
   G__destroy_upto(&G_local, G__LOCAL_VAR, -1);
#endif /* G__ASM_WHOLEFUNC */
   if (G__TRYDESTRUCTOR == memfunc_flag) {
      /* destructor for base calss and class members */
      G__basedestructor();
   }
#ifdef G__NEWINHERIT
   /* recover prev local variable */
   G__p_local = G__get_properties(G_local)->stackinfo.calling_scope; /* same as G__p_local->prev_local */
#endif

   G__tagnum = store_inherit_tagnum;
   G__store_struct_offset = store_inherit_offset;

   /* recover line number and filename*/
   G__ifile.line_number = G__get_properties(G_local)->stackinfo.prev_line_number;
   G__ifile.filenum = G__get_properties(G_local)->stackinfo.prev_filenum;
   if (-1 != G__ifile.filenum && 0 != G__srcfile[G__ifile.filenum].filename) {
      strcpy(G__ifile.name, G__srcfile[G__ifile.filenum].filename);
   }
   else {
      G__ifile.name[0] = '\0';
   }
   //G_local.Unload(); // FIXME: Wasn't this already taken care of by the G__destroy_upto(&G__local,...) call above?
   if (G__dispsource && G__ifile.name && G__ifile.name[0]) {
      G__fprinterr(G__serr, "\n# %s   ", G__ifile.name);
   }
   /* recover file pointer and fpos */
   G__ifile.fp = prev_fp;
   if (G__ifile.fp) fsetpos(G__ifile.fp, &prev_pos);
   if (G__dispsource) {
      if ((G__debug || G__break) && ((G__prerun != 0) || (G__no_exec == 0)) &&
            (G__disp_mask == 0)) {
         G__fprinterr(G__serr, "\n");
      }
   }
   if (G__break_exit_func != 0) {
      G__break = 1;
      G__break_exit_func = 0;
      G__setdebugcond();
   }
   G__break_exit_func = break_exit_func;
   /**********************************************
    * decrement busy flag
    **********************************************/
   G__get_funcproperties(ifn)->entry.busy--;
   if ('P' == store_var_typeB) G__val2pointer(result7);
   //  G__func_page = store_func_page;
   G__func_now = store_func_now;
#ifdef G__ASM_IFUNC
   /* Pop loop compilation environment */
   G__asm_inst = store_asm_inst;
   G__asm_instsize = store_asm_instsize;
   G__asm_stack = store_asm_stack;
   G__asm_name = store_asm_name;
   G__asm_name_p = store_asm_name_p;
   G__asm_param  = store_asm_param;
   G__asm_exec  = store_asm_exec;
   G__asm_noverflow  = store_asm_noverflow;
   G__asm_cp  = store_asm_cp;
   G__asm_dt  = store_asm_dt;
   G__asm_index  = store_asm_index;
#endif /* G__ASM_IFUNC */
   G__exec_memberfunc = store_exec_memberfunc;
   G__security = store_security;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   return 1;
}

//______________________________________________________________________________
int Cint::Internal::G__function_signature_match(const Reflex::Member& func1, const Reflex::Member& func2, bool check_return_type, int /* matchmode */, int *nref)
{
   if (nref) *nref = 0;

   if (func1.Name() != func2.Name()
         || (func1.FunctionParameterSize() != func2.FunctionParameterSize()
             && G__get_funcproperties(func1)->entry.ansi
             && G__get_funcproperties(func2)->entry.ansi)
         || func1.IsConst() != func2.IsConst()
         || check_return_type && (func1.TypeOf().ReturnType() != func2.TypeOf().ReturnType())) {
      return 0;
   }

   int paran;
   if (G__get_funcproperties(func1)->entry.ansi && G__get_funcproperties(func2)->entry.ansi)
      paran = func1.FunctionParameterSize();
   else
      return 1;

   for (int j = 0; j < paran; ++j) {
      // can't go via func1.TypeOf().FuncParam_Begin... - we might be looking at call args here
      if (func1.TypeOf().FunctionParameterAt(j) != func2.TypeOf().FunctionParameterAt(j)) {
         if (nref) {
            if (func1.TypeOf().IsReference() && !func2.TypeOf().IsReference()) {
               if (func1.TypeOf().ToType() == func2.TypeOf()) {
                  ++(*nref);
               }
               else {
                  return 0;
               }
            }
            else if (!func1.TypeOf().IsReference() && func2.TypeOf().IsReference()) {
               if (func1.TypeOf() == func2.TypeOf().ToType()) {
                  ++(*nref);
               }
               else {
                  return 0;
               }
            }
         }
         else {
            return 0;
         }
      }
   }
   return 1;
}

//______________________________________________________________________________
::Reflex::Member Cint::Internal::G__ifunc_exist(::Reflex::Member ifunc_now, const ::Reflex::Scope &ifunc, bool check_return_type)
{
   // compare  hash,funcname,type,p_tagtype,p_typetable,reftype,para_nu
   //          para_type[],para_p_tagtable[],para_p_typetable[],para_reftype[]
   //          para_default[]
   for (::Reflex::Member_Iterator i = ifunc.FunctionMember_Begin();
         i != ifunc.FunctionMember_End();
         ++i) {
      if ('~' == ifunc_now.Name()[0] &&
            '~' == i->Name()[0]) { /* destructor matches with ~ */
         return *i;
      }
      int nref;
      bool result = G__function_signature_match(ifunc_now, *i, check_return_type, 0, &nref);

      if (result) {
         if (nref) {
            G__fprinterr(G__serr, "Warning: %s(), parameter only differs in reference type or not"
                         , i->Name().c_str());
            G__printlinenum();
         }
         return *i;
      }
   }
   return ::Reflex::Member();
}

//______________________________________________________________________________
::Reflex::Member Cint::Internal::G__ifunc_ambiguous(const ::Reflex::Member &ifunc_now, const ::Reflex::Scope &ifunc, const ::Reflex::Type &derivedtagnum)
{
   int j, paran;
   for (::Reflex::Member_Iterator i = ifunc.FunctionMember_Begin();
         i != ifunc.FunctionMember_End();
         ++i) {
      if ('~' == ifunc_now.Name().c_str()[0] &&
            '~' == i->Name().c_str()[0]) { /* destructor matches with ~ */
         return(*i);
      }
      if (/* ifunc_now->hash[allifunc]!=ifunc->hash[i] || */
         ifunc_now.Name() == i->Name()
      ) continue; /* unmatch */
      if (ifunc_now.FunctionParameterSize() < i->FunctionParameterSize())
         paran = ifunc_now.FunctionParameterSize();
      else
         paran = i->FunctionParameterSize();
      if (paran < 0) paran = 0;
      for (j = 0;j < paran;j++) {
         if (G__get_type(ifunc_now.TypeOf().FunctionParameterAt(j)) != G__get_type(i->TypeOf().FunctionParameterAt(j)))
            break; /* unmatch */
         if (ifunc_now.TypeOf().FunctionParameterAt(j).RawType()
               == i->TypeOf().FunctionParameterAt(j).RawType()) continue; /* match */
#ifdef G__VIRTUALBASE
         if (-1 == G__ispublicbase(ifunc_now.TypeOf().FunctionParameterAt(j).RawType()
                                   , derivedtagnum, (void*)G__STATICRESOLUTION2) ||
               -1 == G__ispublicbase(i->TypeOf().FunctionParameterAt(j).RawType(), derivedtagnum
                                     , (void*)G__STATICRESOLUTION2))
#else
         if (-1 == G__ispublicbase(func_now.TypeOf().FunctionParameterAt(j).RawType()
                                   , derivedtagnum) ||
               -1 == G__ispublicbase(i->TypeOf().FunctionParameterAt(j).RawType(), derivedtagnum))
#endif
            break; /* unmatch */
         /* else match */
      }
      if ((ifunc_now.FunctionParameterSize() < i->FunctionParameterSize() &&
            i->FunctionParameterDefaultAt(paran)[0]) ||
            (ifunc_now.FunctionParameterSize() > i->FunctionParameterSize() &&
             ifunc_now.FunctionParameterDefaultAt(paran)[0])) {
         return(*i);
      }
      else if (j == paran) { /* all matched */
         return(*i);
      }
   }
   return Reflex::Member(); /* not found case */
}

//______________________________________________________________________________
::Reflex::Member G__get_ifunchandle(char *funcname, G__param *libp, int hash, const ::Reflex::Scope &p_ifunc, int access, int funcmatch)
{
   int ipara = 0;
   int itemp = 0;
   if (G__get_tagnum(p_ifunc) != -1) {
      G__incsetup_memfunc(G__get_tagnum(p_ifunc));
   }
   ::Reflex::Member_Iterator ifn;
   for (ifn = p_ifunc.FunctionMember_Begin(); ifn != p_ifunc.FunctionMember_End(); ++ifn) {
      if (strcmp(funcname, ifn->Name().c_str()) || !G__test_access(*ifn, access)) {
         continue; // name does not match, or access does not match
      }
      //  for overloading of function and operator check if parameter types match
      //  set(reset) match flag ipara temporarily
      itemp = 0;
      ipara = 1;
      if (!G__get_funcproperties(*ifn)->entry.ansi) { // K&R C style header
         break;
      }
      if ((hash == G__HASH_MAIN) && !strcmp(funcname, "main")) { // if main(), no overloading
         break;
      }
      if ((int)ifn->FunctionParameterSize() < libp->paran) { // if more args than formal params, no match
         ipara = 0;
         itemp = ifn->FunctionParameterSize(); // end of this parameter
         continue; // next function
      }
      while (itemp < (int) ifn->FunctionParameterSize()) { // scan each parameter
         if (!ifn->FunctionParameterDefaultAt(itemp).c_str()[0] && (itemp >= libp->paran)) {
            ipara = 0; // no arg given, and no default, so no match
         }
         else if (ifn->FunctionParameterDefaultAt(itemp).c_str()[0] && (itemp >= libp->paran)) {
            ipara = 2; // no arg given, but we have a default, a match
         }
         else {
            ipara = G__param_match(G__get_type(ifn->TypeOf().FunctionParameterAt(itemp)), ifn->TypeOf().FunctionParameterAt(itemp).RawType(), G__get_funcproperties(*ifn)->entry.para_default.at(itemp), G__get_type(G__value_typenum(libp->para[itemp])), G__value_typenum(libp->para[itemp]).RawType(), &(libp->para[itemp]), libp->parameter[itemp], funcmatch, ifn->FunctionParameterSize() - itemp - 1, G__get_reftype(ifn->TypeOf().FunctionParameterAt(itemp)), G__get_isconst(ifn->TypeOf().FunctionParameterAt(itemp)));
         }
         switch (ipara) {
            case 0: // no match, next function
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, " unmatch%d %c tagnum%d %p : %c tagnum%d %d\n", itemp, G__get_type(ifn->TypeOf().FunctionParameterAt(itemp)), G__get_tagnum(ifn->TypeOf().FunctionParameterAt(itemp)), (G__get_funcproperties(*ifn)->entry.para_default.at(itemp)), G__get_type(G__value_typenum(libp->para[itemp])), G__get_tagnum(G__value_typenum(libp->para[itemp])), funcmatch);
               }
#endif // G__ASM_DBG
               itemp = (int) ifn->FunctionParameterSize(); // exit from while loop
               break;
            case 1: // matched this one, next parameter
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, " match%d %c tagnum%d %p : %c tagnum%d %d\n", itemp, G__get_type(ifn->TypeOf().FunctionParameterAt(itemp)), G__get_tagnum(ifn->TypeOf().FunctionParameterAt(itemp)), (G__get_funcproperties(*ifn)->entry.para_default.at(itemp)), G__get_type(G__value_typenum(libp->para[itemp])), G__get_tagnum(G__value_typenum(libp->para[itemp])), funcmatch);
               }
#endif // G__ASM_DBG
               if (funcmatch != G__EXACT) {
                  G__warn_refpromotion(*ifn, itemp, libp);
               }
               ++itemp; // next function parameter
               break;
            case 2: // default parameter
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, " default%d %c tagnum%d %p : %c tagnum%d %d\n", itemp, G__get_type(ifn->TypeOf().FunctionParameterAt(itemp)), G__get_tagnum(ifn->TypeOf().FunctionParameterAt(itemp)), (G__get_funcproperties(*ifn)->entry.para_default.at(itemp)), G__get_type(G__value_typenum(libp->para[itemp])), G__get_tagnum(G__value_typenum(libp->para[itemp])), funcmatch);
               }
#endif // G__ASM_DBG
               itemp = ifn->FunctionParameterSize(); // exit from while loop
               break;
         }
      }
      if (ipara) { // all parameters matched, we are done.
         break;
      }
   }
   if (ifn == p_ifunc.FunctionMember_End()) {
      return ::Reflex::Member();
   }
   return *ifn;
}

//______________________________________________________________________________
::Reflex::Member G__get_ifunchandle_base(char *funcname, G__param *libp, int hash, const ::Reflex::Scope &p_ifunc, char **poffset, int access, int funcmatch, int withInheritance)
{
   int tagnum;
   ::Reflex::Member ifunc;
   int basen = 0;
   struct G__inheritance *baseclass;

   /* Search for function */
   *poffset = 0;
   ifunc = G__get_ifunchandle(funcname, libp, hash, p_ifunc, access, funcmatch);
   if (ifunc || !withInheritance) return(ifunc);

   /* Search for base class function if member function */
   tagnum = G__get_tagnum(p_ifunc);
   if (-1 != tagnum) {
      baseclass = G__struct.baseclass[tagnum];
      while (basen < baseclass->basen) {
         if (baseclass->baseaccess[basen]&G__PUBLIC) {
#ifdef G__VIRTUALBASE
            /* Can not handle virtual base class member function for ERTTI
             * because pointer to the object is not given  */
#endif
            *poffset = baseclass->baseoffset[basen];
            ifunc = G__get_ifunchandle(funcname, libp, hash, G__Dict::GetDict().GetScope(baseclass->basetagnum[basen])
                                       , access, funcmatch);
            if (ifunc) return(ifunc);
         }
         ++basen;
      }
   }

   /* Not found , ifunc=0 */
   return(ifunc);
}

//______________________________________________________________________________
void Cint::Internal::G__argtype2param(char *argtype, G__param *libp)
{
   G__StrBuf typenam_sb(G__MAXNAME*2);
   char *typenam = typenam_sb;
   int p = 0;
   int c;
   char *endmark = ",);";

   libp->paran = 0;
   libp->para[0] = G__null;

   do {
      c = G__getstream_template(argtype, &p, typenam, endmark);
      if (typenam[0]) {
         char* start = typenam;
         while (isspace(*start)) ++start;
         if (*start) {
            char* end = start + strlen(start) - 1;
            while (isspace(*end) && end != start) --end;
         }
         libp->para[libp->paran] = G__string2type(start);
         ++libp->paran;
      }
   }
   while (',' == c);
}

//______________________________________________________________________________
extern "C" struct G__ifunc_table* G__get_methodhandle(char* funcname, char* argtype, G__ifunc_table* p_ifunc, long* pifn, long* poffset, int withConversion, int withInheritance)
{
   ::Reflex::Scope ifunc;
   ::Reflex::Member ifn;
   struct G__param para;
   int hash;
   int temp;
   struct G__funclist* funclist = 0;

   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   ifunc = G__Dict::GetDict().GetScope(p_ifunc);
   G__def_tagnum = ifunc;
   G__tagdefining = G__def_tagnum;
   G__argtype2param(argtype, &para);
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__hash(funcname, hash, temp);

   *pifn = -2;

   if (withConversion) {
      int tagnum = G__get_tagnum(ifunc); // p_ifunc->tagnum;
      if (tagnum != -1) {
         G__incsetup_memfunc(tagnum);
      }
      int match_error = 0;
      ifn = G__overload_match(funcname, &para, hash, ifunc, G__TRYNORMAL, G__PUBLIC_PROTECTED_PRIVATE, 0, (withConversion & 0x2) ? 1 : 0, &match_error);
      *poffset = 0;
      *pifn = -2;
      if (ifn || !withInheritance) {
         return (struct G__ifunc_table*) ifn.Id();
      }
      if (tagnum != -1) {
         struct G__inheritance* baseclass = G__struct.baseclass[tagnum];
         for (int basen = 0; basen < baseclass->basen; ++basen) {
            if (baseclass->baseaccess[basen] & G__PUBLIC) {
               G__incsetup_memfunc(baseclass->basetagnum[basen]);
               *poffset = (long) baseclass->baseoffset[basen];
               ifunc = G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]);
               ifn = G__overload_match(funcname, &para, hash, ifunc, G__TRYNORMAL, G__PUBLIC_PROTECTED_PRIVATE, 0, 0, &match_error);
               *pifn = -2;
               if (ifn) {
                  return (struct G__ifunc_table*) ifn.Id();
               }
            }
         }
      }
   }
   else {
      /* first, search for exact match */
      ifn = G__get_ifunchandle_base(funcname, &para, hash, ifunc, (char**)poffset
            , G__PUBLIC_PROTECTED_PRIVATE, G__EXACT
            , withInheritance
            );
      if (ifn) {
         return((struct G__ifunc_table *)ifn.Id());
      }

      /* if no exact match, try to instantiate template function */
      funclist = G__add_templatefunc(funcname, &para, hash, funclist, ifunc, 0);
      if (funclist && funclist->rate == G__EXACTMATCH) {
         ifn = funclist->ifunc;
         G__funclist_delete(funclist);
         return((struct G__ifunc_table *)ifn.Id());
      }
      G__funclist_delete(funclist);

      // FIXME: Remove this code for now, we should not attempt conversions
      //        here, we have specified an exact prototype.
      //
      //for (int match = G__EXACT;match <= G__STDCONV;match++) {
      //   ifn = G__get_ifunchandle_base(funcname, &para, hash, ifunc, (char**)poffset
      //         , G__PUBLIC_PROTECTED_PRIVATE
      //         , match
      //         , withInheritance
      //         );
      //   if (ifn) return((struct G__ifunc_table *)ifn.Id());
      //}
   }

   return (struct G__ifunc_table*) ifn.Id();
}

//______________________________________________________________________________
extern "C" struct G__ifunc_table *G__get_methodhandle2(char *funcname, G__param *libp, G__ifunc_table *p_ifunc, long *pifn, long *poffset, int withConversion, int withInheritance)
{
   int hash;
   int temp;
   struct G__funclist* funclist = 0;

   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   ::Reflex::Scope ifunc = G__Dict::GetDict().GetScope(p_ifunc);
   G__def_tagnum = ifunc;
   G__tagdefining = ifunc;
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__hash(funcname, hash, temp);


   ::Reflex::Member ifn;
   *pifn = 0;

   if (withConversion)
   {
      int tagnum = G__get_tagnum(ifunc);

      if (-1 != tagnum) G__incsetup_memfunc(tagnum);

      int match_error = 0;
      ifn = G__overload_match(funcname, libp, hash, ifunc, G__TRYNORMAL, G__PUBLIC_PROTECTED_PRIVATE, 0, 0, &match_error);
      *poffset = 0;
      if (ifn || !withInheritance) return((G__ifunc_table*)ifn.Id());
      if (-1 != tagnum) {
         int basen = 0;
         struct G__inheritance *baseclass = G__struct.baseclass[tagnum];
         while (basen < baseclass->basen) {
            if (baseclass->baseaccess[basen]&G__PUBLIC) {
               G__incsetup_memfunc(baseclass->basetagnum[basen]);
               *poffset = (long)baseclass->baseoffset[basen];
               ifunc = G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]);
               ifn = G__overload_match(funcname, libp, hash, ifunc, G__TRYNORMAL, G__PUBLIC_PROTECTED_PRIVATE, 0, 0, &match_error);
               if (ifn) return((G__ifunc_table*)ifn.Id());
            }
            ++basen;
         }
      }
   }
   else
   {
      /* first, search for exact match */
      ifn = G__get_ifunchandle_base(funcname, libp, hash, ifunc, (char**)poffset
            , G__PUBLIC_PROTECTED_PRIVATE, G__EXACT
            , withInheritance
            );
      if (ifn) return((G__ifunc_table*)ifn.Id());

      /* if no exact match, try to instantiate template function */
      funclist = G__add_templatefunc(funcname, libp, hash, funclist, ifunc, 0);
      if (funclist && funclist->rate == G__EXACTMATCH) {
         ifn = funclist->ifunc;
         G__funclist_delete(funclist);
         return((G__ifunc_table*)ifn.Id());
      }
      G__funclist_delete(funclist);

      //
      // Now see if we can call it using the standard type conversions.
      //
      for (int match = G__EXACT;match <= G__STDCONV;match++) {
         ifn = G__get_ifunchandle_base(funcname, libp, hash, ifunc, (char**)poffset
               , G__PUBLIC_PROTECTED_PRIVATE
               , match
               , withInheritance
               );
         if (ifn) return((G__ifunc_table*)ifn.Id());
      }
   }

   return((G__ifunc_table*)ifn.Id());
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
