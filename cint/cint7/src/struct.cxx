/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file struct.c
 ************************************************************************
 * Description:
 *  Struct, class, enum, union handling
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Reflex/Base.h"
#include "Dict.h"

#include <cctype>
#include <cstring>

using namespace std;
using namespace Cint::Internal;

// Static functions.
static void G__add_anonymousunion(const ::Reflex::Type uniontype, int def_struct_member, ::Reflex::Scope envtagnum);
static int G__check_semicolumn_after_classdef(int isclassdef);

// Cint internal functions.
namespace Cint {
namespace Internal {
int G__using_namespace();
::Reflex::Scope G__get_envtagnum();
int G__isenclosingclass(const ::Reflex::Scope enclosingtagnum, const ::Reflex::Scope env_tagnum);
int G__isenclosingclassbase(const ::Reflex::Scope enclosingtagnum, const ::Reflex::Scope env_tagnum);
int G__isenclosingclassbase(int enclosingtagnum, int env_tagnum);
char* G__find_first_scope_operator(char* name);
char* G__find_last_scope_operator(char* name);
::Reflex::Type G__find_type(const char* type_name, int /*errorflag*/, int /*templateflag*/);
::Reflex::Member G__add_scopemember(::Reflex::Scope envvar, const char* varname, const ::Reflex::Type type, int reflex_modifiers, size_t reflex_offset, char* cint_offset, int var_access, int var_statictype);
void G__define_struct(char type);
void G__create_global_namespace();
void G__create_bytecode_arena();
#ifndef G__OLDIMPLEMENTATION2030
int G__callfunc0(G__value* result, const ::Reflex::Member ifunc, G__param* libp, void* p, int funcmatch);
int G__calldtor(void* p, const Reflex::Scope tagnum, int isheap);
#endif // G__OLDIMPLEMENTATION2030
} // namespace Internal
} // namespace Cint

// Functions in the C interface.
extern "C" int G__set_class_autoloading(int newvalue);
extern "C" void G__set_class_autoloading_callback(int (*p2f)(char*, char*));
extern "C" char* G__get_class_autoloading_table(char* classname);
extern "C" void G__set_class_autoloading_table(char* classname, char* libname);
extern "C" int G__defined_tagname(const char* tagname, int noerror);
extern "C" int G__search_tagname(const char* tagname, int type);

#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Builder/NamespaceBuilder.h"
#include "Reflex/Builder/ClassBuilder.h"
#include "Reflex/Builder/UnionBuilder.h"

//______________________________________________________________________________
static const char G__CLASS_AUTOLOAD = 'a';
static int G__enable_autoloading = 1;
int (*G__p_class_autoloading)(char*, char*);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static void G__add_anonymousunion(const ::Reflex::Type uniontype, int def_struct_member, ::Reflex::Scope envtagnum)
{
   ::Reflex::Scope envvar;
   int statictype = G__AUTO;
   int store_statictype = G__AUTO;
   int access = G__PUBLIC;
   if (def_struct_member) { // inner anonymous union
      if (!envtagnum.DataMemberSize()) {
         access = G__access;
      }
      else {
         ::Reflex::Member prev = envtagnum.DataMemberAt(envtagnum.DataMemberSize() - 1);
         if (prev.IsPublic()) {
            access = G__PUBLIC;
         }
         else if (prev.IsProtected()) {
            access = G__PROTECTED;
         }
         else if (prev.IsPrivate()) {
            access = G__PRIVATE;
         }
      }
      envvar = envtagnum;
   }
   else {
      if (G__p_local) {
         envvar = G__p_local;
      }
      else {
         envvar = ::Reflex::Scope::GlobalScope();
         statictype = G__ifile.filenum; // file scope static
      }
      store_statictype = G__COMPILEDGLOBAL;
   }
   char* cint_offset = (char*) G__malloc(1, uniontype.SizeOf(), "");
   for (unsigned int idx = 0; idx < uniontype.DataMemberSize(); ++idx) {
      ::Reflex::Member mbr = uniontype.DataMemberAt(idx);
      ::Reflex::Member new_mbr = G__add_scopemember(envvar, mbr.Name().c_str(), mbr.TypeOf(), 0, mbr.Offset(), cint_offset, access, statictype);
      *G__get_properties(new_mbr) = *G__get_properties(mbr); // WARNING: This overwrites the statictype we just passed to the function call.
      G__get_properties(new_mbr)->statictype = statictype;
      statictype = store_statictype;
   }
}

//______________________________________________________________________________
static int G__check_semicolumn_after_classdef(int isclassdef)
{
   G__StrBuf checkbuf_sb(G__ONELINE);
   char *checkbuf = checkbuf_sb;
   int store_linenum = G__ifile.line_number;
   int store_c;
   int errflag = 0;
   fpos_t store_pos;
   fgetpos(G__ifile.fp, &store_pos);
   G__disp_mask = 1000;

   store_c = G__fgetname(checkbuf, ";,(");
   if (isspace(store_c) && '*' != checkbuf[0] && 0 == strchr(checkbuf, '[')) {
      G__StrBuf checkbuf2_sb(G__ONELINE);
      char *checkbuf2 = checkbuf2_sb;
      store_c = G__fgetname(checkbuf2, ";,(");
      if (isalnum(checkbuf2[0])) errflag = 1;
   }

   G__disp_mask = 0;
   fsetpos(G__ifile.fp, &store_pos);
   G__ifile.line_number = store_linenum;
   if (errflag || (isclassdef && '(' == store_c)) {
      G__genericerror("Error: ';' missing after class/struct/enum declaration");
      return(1);
   }
   return(0);
}

//______________________________________________________________________________
//
//  Cint internal functions.
//

//______________________________________________________________________________
int Cint::Internal::G__using_namespace()
{
   // Parse using a using directive or a using declaration.
   //
   // using  namespace [ns_name];  using directive   -> inheritance
   // using  [scope]::[member];    using declaration -> reference object
   //       ^
   // 
   // Note: using directive appears in global scope is not implemented yet
   //
   int result = 0;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   int c;

   /* check if using directive or declaration */
   c = G__fgetname_template(buf, ";");

   if (strcmp(buf, "namespace") == 0) {
      /*************************************************************
      * using directive, treat as inheritance
      *************************************************************/
      ::Reflex::Scope basetagnum;
      ::Reflex::Scope envtagnum;
      c = G__fgetstream_template(buf, ";");
#ifndef G__STD_NAMESPACE /* ON676 */
      if (';' == c && strcmp(buf, "std") == 0
            && G__ignore_stdnamespace
         ) return 1;
#endif
      basetagnum = G__Dict::GetDict().GetScope(G__defined_tagname(buf, 2));
      if (!basetagnum) {
         G__fprinterr(G__serr, "Error: namespace %s is not defined", buf);
         G__genericerror((char*)NULL);
         return(0);
      }
      envtagnum = G__get_envtagnum();
      if (envtagnum)
         envtagnum.AddUsingDirective(basetagnum);
      // BEGIN OLD
      if (G__def_struct_member) {
         /* using directive in other namespace or class/struct */
         envtagnum = G__get_envtagnum();
         if (envtagnum) {
            struct G__inheritance *base = G__struct.baseclass[G__get_tagnum(envtagnum)];
            base->vec.push_back(G__inheritance::G__Entry(G__get_tagnum(basetagnum)));
         }
      }
      else {
         /* using directive in global scope, to be implemented
         * 1. global scope has baseclass information
         * 2. G__searchvariable() looks for global scope baseclass
         */
         /* first check whether we already have this directive in
         memory */
         size_t j;
         int found;
         found = 0;
         for (j = 0; j < G__globalusingnamespace.vec.size(); ++j) {
            struct G__inheritance *base = &G__globalusingnamespace;
            if (base->vec[j].basetagnum == G__get_tagnum(basetagnum)) {
               found = 1;
               break;
            }
         }
         if (!found) {
            struct G__inheritance *base = &G__globalusingnamespace;
            base->vec.push_back(G__inheritance::G__Entry(G__get_tagnum(basetagnum)));
         }
         result = 1;
      }
      // END OLD
   }

   else {
      /*************************************************************
      * using declaration, treat as reference object
      *************************************************************/
      char *struct_offset, *store_struct_offset;
      int ig15, hash;
      G__hash(buf, hash, ig15);
#ifdef __GNUC__
#else
#pragma message (FIXME("namespace FOO{using std::endl; didn't work in CINT5 - does this take care of it?"))
#endif
      ::Reflex::Member member = G__find_variable(buf, hash, G__p_local,::Reflex::Scope::GlobalScope(), &struct_offset, &store_struct_offset, &ig15, 1);
      if (member) {
         std::string varname(member.Name());
         ::Reflex::Scope varscope = ::Reflex::Scope::GlobalScope();
         if (G__p_local) {
            varscope = G__p_local;
         }
         ::Reflex::Member avar = G__add_scopemember(varscope, varname.c_str(), member.TypeOf(), 0, member.Offset(), G__get_offset(member), G__access, G__get_properties(member)->statictype);
         *G__get_properties(avar) = *G__get_properties(member);
         G__get_properties(avar)->isFromUsing = true;

      }
      else {
         int tagnum = G__defined_tagname(buf, 1);
         if (-1 != tagnum) {
            /* using scope::classname; to be implemented
            *  Now, G__tagtable is not ready */
         }
         else result = 1;
      }
   }

   return(result);
}

//______________________________________________________________________________
::Reflex::Scope Cint::Internal::G__get_envtagnum()
{
   // Return our enclosing scope.
#if 0
   if ((!G__def_tagnum || G__def_tagnum.IsTopScope()) && G__exec_memberfunc) { // No enclosing scope, in member function.
      return G__memberfunc_tagnum; // Use member function's defining class.
   }
   if (G__tagdefining) {
      return G__tagdefining;
   }
   return ::Reflex::Scope::GlobalScope(); // We are in the global namespace.
#endif // 0
   if (G__def_tagnum && !G__def_tagnum.IsTopScope()) { // We are enclosed, and not in the global namespace.
      // -- We are enclosed, and not in the global namespace.
      //
      //  In case of enclosed class definition, G__tagdefining remains
      //  as enclosing class identity, while G__def_tagnum changes to
      //  enclosed class identity. For finding environment scope, we
      //  must use G__tagdefining.
      //
      if (G__def_tagnum != G__tagdefining) {
         return G__tagdefining;
      }
      return G__def_tagnum;
   }
   if (G__exec_memberfunc) { // We are in a member function.
      return G__memberfunc_tagnum;
   }
   return ::Reflex::Scope::GlobalScope(); // We are in the global namespace.
}

//______________________________________________________________________________
int Cint::Internal::G__isenclosingclass(const ::Reflex::Scope enclosingtagnum, const ::Reflex::Scope env_tagnum)
{
   if (!env_tagnum || !enclosingtagnum
         || env_tagnum.IsTopScope() || enclosingtagnum.IsTopScope()) return(0);
   ::Reflex::Scope tagnum = env_tagnum.DeclaringScope();
   while (tagnum && !tagnum.IsTopScope()) {
      if (tagnum == enclosingtagnum) return(1);
      tagnum = tagnum.DeclaringScope();
   }
   return(0);
}

//______________________________________________________________________________
int Cint::Internal::G__isenclosingclassbase(const ::Reflex::Scope enclosingtagnum, const ::Reflex::Scope env_tagnum)
{
   // --
#ifdef __GNUC__
#else
#pragma message (FIXME("G__isenclosingclass also checks for using directives, and it really shouldn't"))
#endif
   if (!env_tagnum || !enclosingtagnum
         || env_tagnum.IsTopScope() || enclosingtagnum.IsTopScope())
      return 0;
   ::Reflex::Scope tagnum = env_tagnum.DeclaringScope();
   while (tagnum) {
      if (((::Reflex::Type)tagnum).HasBase(enclosingtagnum))
         return 1;
      if (tagnum == enclosingtagnum) return(1);
      tagnum = tagnum.DeclaringScope();
   }
   // to also take using directives into account
   return G__isenclosingclassbase(G__get_tagnum(enclosingtagnum), G__get_tagnum(env_tagnum));
}

//______________________________________________________________________________
int Cint::Internal::G__isenclosingclassbase(int enclosingtagnum, int env_tagnum)
{
   int tagnum;
   if (0 > env_tagnum || 0 > enclosingtagnum) return(0);
   tagnum = G__struct.parent_tagnum[env_tagnum];
   while (-1 != tagnum) {
      if (-1 != G__isanybase(enclosingtagnum, tagnum, (long) G__STATICRESOLUTION))
         return 1;
      if (tagnum == enclosingtagnum) return(1);
      tagnum = G__struct.parent_tagnum[tagnum];
   }
   return(0);
}

//______________________________________________________________________________
char* Cint::Internal::G__find_first_scope_operator(char* name)
{
   // Return a pointer to the first scope operator in name.
   // Only those at the outermost level of template nesting are considered.
   char* p = name;
   int single_quote = 0;
   int double_quote = 0;
   int nest = 0;

   while (*p != '\0') {

      char c = *p;

      if (0 == single_quote && 0 == double_quote) {
         if (c == '<')
            ++nest;
         else if (nest > 0 && c == '>')
            --nest;
         else if (nest == 0 && c == ':' && *(p + 1) == ':')
            return p;
      }

      if ('\'' == c && 0 == double_quote)
         single_quote = single_quote ^ 1 ;
      else if ('"' == c && 0 == single_quote)
         double_quote = double_quote ^ 1 ;

      ++p;
   }

   return 0;
}

//______________________________________________________________________________
char* Cint::Internal::G__find_last_scope_operator(char* name)
{
   // Return a pointer to the last scope operator in name.
   // Only those at the outermost level of template nesting are considered.
   char* p = name + strlen(name) - 1;
   int single_quote = 0;
   int double_quote = 0;
   int nest = 0;

   while (p > name) {

      char c = *p;

      if (0 == single_quote && 0 == double_quote) {
         if (c == '>')
            ++nest;
         else if (nest > 0 && c == '<')
            --nest;
         else if (nest == 0 && c == ':' && *(p - 1) == ':')
            return p -1;
      }

      if ('\'' == c && 0 == double_quote)
         single_quote = single_quote ^ 1 ;
      else if ('"' == c && 0 == single_quote)
         double_quote = double_quote ^ 1 ;

      --p;
   }

   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__class_autoloading(int* ptagnum)
{
   // Load the library containing the class tagnum, according to
   // G__struct.libname[tagnum] set via G__set_class_autloading_table().
   // As a request to load vector<Long64_t> can result in vector<long long>
   // beging loaded, the requested tagnum and the loaded tagnum need not
   // be identical, i.e. G__class_autolading can change the tagnum to
   // point to the valid class with dictionary. The previous G__struct entry
   // is marked as an "ex autoload entry" so no name lookup can find it anymore.

   int& tagnum = *ptagnum;
   if (!G__enable_autoloading || (tagnum < 0)) {
      return 0;
   }
   // Note: We also autoload classes that were only forward declared.
   if (
      (G__struct.type[tagnum] == G__CLASS_AUTOLOAD) ||
      ((G__struct.filenum[tagnum] == -1) && (G__struct.size[tagnum] == 0))
   ) {
      char* libname = G__struct.libname[tagnum];
      if (!libname || !libname[0]) {
         return 0;
      }
      // need to work on copy of libname, because loading a lib
      // might change the auto-load info, and thus render
      // the G__struct.libname[tagnum] value invalid.
      // E.g. ROOT (re-)reads the library's rootmap file when
      // loading it, and update that for dependent libraries
      // can change our libname.
      char* copyLibname = new char[strlen(libname) + 1];
      strcpy(copyLibname, libname);
      if (G__p_class_autoloading) {
         // -- We have a callback, use that.
         int oldAutoLoading = G__enable_autoloading;
         G__enable_autoloading = 0;
         // reset the def tagnums to not collide with dict setup
         ::Reflex::Scope store_def_tagnum = G__def_tagnum;
         ::Reflex::Scope store_tagdefining = G__tagdefining;
         G__def_tagnum = Reflex::Scope();
         G__tagdefining = Reflex::Scope();
         std::string fulltagname( G__fulltagname(tagnum, 1) );
         int res = (*G__p_class_autoloading)((char*)fulltagname.c_str(), copyLibname);
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         if (G__struct.type[tagnum] == 0 && G__struct.name[tagnum] && G__struct.name[tagnum][0]=='@') {
            // This record was already 'killed' during the autoloading.
            // Let's find the new real one!
            //FIXME: remove the char* cast
            tagnum = G__defined_tagname(fulltagname.c_str(), 3);
            
         } else if (G__struct.type[tagnum] == G__CLASS_AUTOLOAD) {
            // if (strstr(G__struct.name[tagnum], "<") != 0) 
            {
               // Kill this entry.
               store_def_tagnum = G__def_tagnum;
               store_tagdefining = G__tagdefining;
               G__tagdefining = G__def_tagnum = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[tagnum]);

               // Find the corresponding Reflex type (This must be done before
               // modifying G__struct.name since GetType uses it to find the Reflex Type!).
               ::Reflex::Type autoloadcl = G__Dict::GetDict().GetType( tagnum );

               // "hide" tagnum's name: we want to check whether this auto-loading loaded
               // another version of the same class, e.g. because of vector<Long64_t>
               // being requested but vector<long long> being loaded:
               std::string fullname( autoloadcl.Name( Reflex::SCOPED ) );
               std::string origName(G__struct.name[tagnum]);
               if (G__struct.name[tagnum][0]) {
                  G__struct.name[tagnum][0] = '@';
               }
               // Also hide the Reflex Type.
               if (autoloadcl.ToTypeBase()) {
                  autoloadcl.ToTypeBase()->HideName();
               }
               
               int found_tagnum = G__defined_tagname(fullname.c_str(), 3);
 
               G__def_tagnum = store_def_tagnum;
               G__tagdefining = store_tagdefining;
               if (found_tagnum != -1 && found_tagnum != tagnum) {
                  // The autoload has seemingly failed!
                  // This can happens in 'normal' case if the string representation of the
                  // type registered by the autoloading mechanism is actually a typedef
                  // to the real type (aka mytemp<Long64_t> vs mytemp<long long> or the
                  // stl containers with or without their (default) allocators.
                  char *old = G__struct.name[tagnum];

                  G__struct.name[tagnum] = (char*)malloc(strlen(old) + 50);
                  strcpy(G__struct.name[tagnum], "@@ ex autload entry @@");
                  strcat(G__struct.name[tagnum], old);
                  G__struct.type[tagnum] = 0;
                  free(old);
                  tagnum = found_tagnum;
               } else {
                  if (G__struct.name[tagnum][0]) {
                     G__struct.name[tagnum][0] = origName[0];
                  }
                  if (autoloadcl.ToTypeBase()) {
                     autoloadcl.ToTypeBase()->UnhideName();
                  }                  
               }                  
            }
         }
         G__enable_autoloading = oldAutoLoading;
         delete[] copyLibname;
         return res;
      }
      else if (libname && libname[0]) {
         // -- No autoload callback, try to load the library.
         int oldAutoLoading = G__enable_autoloading;
         G__enable_autoloading = 0;
         if (G__loadfile(copyLibname) >= G__LOADFILE_SUCCESS) {
            // -- Library load succeeded.
            G__enable_autoloading = oldAutoLoading;
            delete[] copyLibname;
            return 1;
         }
         else {
            // -- Library load failed.
            G__struct.type[tagnum] = G__CLASS_AUTOLOAD;
            G__enable_autoloading = oldAutoLoading;
            delete[] copyLibname;
            return -1;
         }
      }
      delete[] copyLibname;
   }
   return 0;
}

//______________________________________________________________________________
::Reflex::Type Cint::Internal::G__find_type(const char* type_name, int /*errorflag*/, int /*templateflag*/)
{
   int ispointer = 0;
   std::string temp = type_name;
   int len = temp.size();
   if (temp.size() > 0 && temp[len-1] == '*') {
      temp[--len] = '\0';
      ispointer = 'A' - 'a';
   }

   ::Reflex::Scope scope = G__get_envtagnum();
   //
   //  This is wrong, G__tmplt_def_tagnum is the tagnum of the
   //  definer of the template, we should not start our search
   //  from there.
   //
   //if (G__tmplt_def_tagnum && !G__tmplt_def_tagnum.IsTopScope()) {
   //   scope = G__tmplt_def_tagnum;
   //}
   if (!scope) {
      printf("Trying to look up struct %s in an invalid enclosing scope!\n", type_name);
      while (scope.Id() && !scope && !scope.IsTopScope())
         scope = scope.DeclaringScope();
   }

   ::Reflex::Type result = scope.LookupType(type_name);

   if (!result && strstr(type_name, "<") != 0) {
      // This may be a template that need instantiating so let's try a
      // different way.
      const char *p = G__find_last_scope_operator((char*)type_name);
      if (p && p != type_name) {
         std::string leftside(type_name, p - type_name);
         // Induce the instantiation of template if any
         if (!(leftside == "std" && G__ignore_stdnamespace)
               && G__defined_tagname(leftside.c_str(), 0) >= 0) {
            result = scope.LookupType(type_name);
         }
      }
   }
   if (!result || result.IsTypedef()) return ::Reflex::Type();

   /* This must be a bad manner. Somebody needs to reset G__var_type
    * especially when typein is 0. */
   G__var_type = G__get_type(result) + ispointer ;

   return result;
}

//______________________________________________________________________________
::Reflex::Member Cint::Internal::G__add_scopemember(::Reflex::Scope scope, const char* name, const ::Reflex::Type type, int reflex_modifiers, size_t reflex_offset, char* cint_offset, int var_access, int var_statictype)
{
   // Add a member variable to a scope.
   int modifiers = reflex_modifiers;
   if (var_access) {
      modifiers &= ~(::Reflex::PUBLIC | ::Reflex::PROTECTED | ::Reflex::PRIVATE);
      switch (var_access) {
         case G__PUBLIC:
            modifiers |= ::Reflex::PUBLIC;
            break;
         case G__PROTECTED:
            modifiers |= ::Reflex::PROTECTED;
            break;
         case G__PRIVATE:
            modifiers |= ::Reflex::PRIVATE;
            break;
      };
   }
   ::Reflex::Member d = scope.AddDataMember(name, type, reflex_offset, modifiers, cint_offset);
   G__get_properties(d)->statictype = var_statictype;
   return d;
}

//______________________________________________________________________________
void Cint::Internal::G__define_struct(char type)
{
   // Parse a class, enum, struct, or union declaration or definition.
   //
   // [struct|union|enum] tagname { member } item ;
   // [struct|union|enum]         { member } item ;
   // [struct|union|enum] tagname            item ;
   // [struct|union|enum] tagname { member }      ;
   //
   //fprintf(stderr, "G__define_struct: Begin.\n");
   //
   //  Validate type.
   //
   switch (type) {
      case 'c':
      case 'e':
      case 'n':
      case 's':
      case 'u':
         break;
      default:
         G__genericerror("Error: Illegal tagtype. struct,union,enum expected");
         break;
   }
   int c;
   G__StrBuf tagname_sb(G__LONGLINE);
   char* tagname = tagname_sb;
   G__StrBuf memname_sb(G__ONELINE);
   char* memname = memname_sb;
   G__StrBuf val_sb(G__ONELINE);
   char* val = val_sb;
   ::Reflex::Scope store_tagnum;
   int store_def_struct_member = 0;
   ::Reflex::Scope store_local;
   G__value enumval = G__null;
   int store_access;
   G__StrBuf basename_sb(G__LONGLINE);
   char* basename = basename_sb;
   int basen;
   struct G__inheritance* baseclass;
   int baseaccess;
   int newdecl;
   int store_static_alloc;
   int len;
   int ispointer = 0;
   int store_prerun;
   ::Reflex::Scope store_def_tagnum;
   int isvirtualbase = 0;
   int isclassdef = 0;
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg && G__asm_noverflow) {
      G__fprinterr(G__serr, "LOOP COMPILE ABORTED FILE:%s LINE:%d\n", G__ifile.name, G__ifile.line_number);
   }
#endif // G__ASM_DBG
   G__abortbytecode();
#endif // G__ASM
   //
   // [class|enum|struct|union] tagname { member } item;
   //                          ^
   //
   //  Read tagname.
   //
   c = G__fgetname_template(tagname, "{:;=&");
   if (strlen(tagname) >= G__LONGLINE) {
      G__fprinterr(G__serr, "Limitation: class name too long. Must be < %d", G__LONGLINE);
      G__genericerror(0);
   }
   doitagain:
   //
   // [class|enum|struct|union] tagname { member } item;
   //                                    ^
   //                     OR
   //
   // [class|enum|struct|union] { member } item;
   //                            ^
   //
   //  Now check for [:;=,)&], open curly brace, or whitespace.
   //
   if (c == '{') { //  Backup to open curly brace.
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
   }
   else if (isspace(c)) {
      //
      // [class|enum|struct|union] tagname { member } item;
      //                                  ^
      //                     OR
      //
      // [class|enum|struct|union] tagname item;
      //                                  ^
      //                     OR
      //
      // [class|enum|struct|union] tagname ;
      //                                  ^
      //
      //  Look ahead for a base class specifier,
      //  and backup if not found.
      //
      c = G__fgetspace(); // Look ahead for base class spec.
      if (c != ':') { // If no base class spec, backup.
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
      }
   }
   else if (c == ':') {
      // -- Inheritance or nested class.
      c = G__fgetc();
      if (c == ':') {
         strcat(tagname, "::");
         len = strlen(tagname);
         c = G__fgetname_template(tagname + len, "{:;=&");
         goto doitagain;
      }
      else {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         c = ':';
      }
   }
   else if (c == ';') {
      // -- Elaborated type specifier.
   }
   else if ((c == '=') && (type == 'n')) {
      // -- namespace alias = nsn; treat as typedef
      c = G__fgetstream_template(basename, ";");
      int tagdefining = G__defined_tagname(basename, 0);
      if (tagdefining != -1) {
         G__declare_typedef(tagname, 'u', tagdefining, 0, 0, G__default_link ? G__globalcomp : G__NOLINK, G__get_tagnum(G__get_envtagnum()), true);
      }
      G__var_type = 'p';
      return;
   }
   else if (G__ansiheader && ((c == ',') || (c == ')'))) {
      // -- Dummy argument for func overloading f(class A*) {}
      G__var_type = 'p';
      if (c == ')') {
         G__ansiheader = 0;
      }
      return;
   }
   else if (c == '&') {
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
      c = ' ';
   }
   else {
      G__genericerror("Syntax error in class/struct definition");
   }
   //
   //  Set default tagname if tagname is omitted.
   //
   if (!tagname[0]) {
      if (type == 'n') {
         // FIXME: What should an unnamed namespace be in Reflex?
         // unnamed namespace, treat as global scope, namespace has no effect.
         // This implementation may be wrong.
         // Should fix later with using directive in global scope.
         G__var_type = 'p';
         int brace_level = 0;
         G__exec_statement(&brace_level);
         return;
      }
      else {
         sprintf(tagname, "$G__NONAME%d_%s(%d)$", G__struct.alltag, G__ifile.name ? G__ifile.name : "<unknown file>", G__ifile.line_number); // FIXME: BAD, remove weird chars, name has to be compilable, it is written to the dictionary file.
         //sprintf(tagname, "$G__NONAME_%d$", G__struct.alltag);
      }
   }
#ifndef G__STD_NAMESPACE
   else if (
      (type == 'n') && // namespace, and
      !strcmp(tagname, "std") && // name is "std", and
      (
         G__ignore_stdnamespace || // we are ignoring the std namespace, or
         (
            G__def_tagnum && // we are nested in something, and
            !G__def_tagnum.IsTopScope() // that something is not the global namespace
         )
      )
   ) {
      // -- Namespace "std::", treat as global scope, namespace has no effect.
      // FIXME: Fix treatment of namespace std!
      G__var_type = 'p';
      int brace_level = 0;
      G__exec_statement(&brace_level);
      return;
   }
#endif // G__STD_NAMESPACE
   store_tagnum = G__tagnum;
   store_def_tagnum = G__def_tagnum;
   //
   //  Check if tagname, ends with "*",
   //  if so consume it and remember it was there.
   //
   len = strlen(tagname);
   if (len && (tagname[len-1] == '*')) {
      ispointer = 1;
      tagname[len-1] = '\0';
   }
   //
   //  Now find the tag, and create a
   //  new entry if it is not found.
   //
   switch (c) {
      case '{':
      case ':':
      case ';':
         {
            //fprintf(stderr, "G__define_struct: Creating scope '%s'\n", tagname);
            int tagnum = G__search_tagname(tagname, type + 0x100); // 0x100: define struct if not found
            G__tagnum = G__Dict::GetDict().GetScope(tagnum);
            //fprintf(stderr, "G__define_struct: Created scope '::%s'\n", G__tagnum.Name(::Reflex::SCOPED).c_str());
            //G__dumpreflex_atlevel(G__tagnum, 0);
         }
         break;
      default:
         {
            int tagnum = G__search_tagname(tagname, type);
            G__tagnum = G__Dict::GetDict().GetScope(tagnum);
         }
         break;
   }
   //
   //  If this was an elaborated type specifier, we are done.
   //
   if (c == ';') {
      // -- Elaborated type specifier.
      G__tagnum = store_tagnum;
      return;
   }
   //
   //  Handle any error.
   //
   if (!G__tagnum || G__tagnum.IsTopScope()) { // ERROR, we could not get a tagnum.
      // FIXME: Shouldn't we warn the user that we
      // FIXME: cannot find the underlying type?
      // FIXME: Or can we register a typedef to uninitialized type?
      // This case might not happen.
      G__fignorestream(";");
      G__tagnum = store_tagnum;
      return;
   }
   //
   //  Initialize the definer prior to processing
   //  any member declarations.
   //
   G__def_tagnum = G__tagnum; // Initialize the enclosing class, enum, namespace, struct, or union.
   //
   //  Judge if new declaration by size.
   //
   newdecl = 0;
   if (!G__struct.size[G__get_tagnum(G__tagnum)]) {
      newdecl = 1;
   }
   //
   //  Initialize that this is not a typedef.
   //
   G__typenum = ::Reflex::Type();
   //
   // [struct|union|enum]   tagname   { member }  item ;
   //                                 ^
   //                     OR
   // [struct|union|enum]             { member }  item ;
   //                                 ^
   //                     OR
   // [struct|union|enum]   tagname     item ;
   //                                   ^
   // member declaration if exist
   //
   //
   // base class declaration
   //
   if (c == ':') { // We have the beginning of the base class specifiers.
      c = ',';
   }
   //
   //  Process any base class specifiers.
   //
   while (c == ',') { // We have another base class specifier.
      /* [struct|class] <tagname> : <private|public> base_class
       *                           ^                                */

      /* reset virtualbase flag */
      isvirtualbase = 0;

      /* read base class name */
#ifdef G__TEMPLATECLASS
      c = G__fgetname_template(basename, "{,"); /* case 2) */
#else // G__TEMPLATECLASS
      c = G__fgetname(basename, "{,");
#endif // G__TEMPLATECLASS

      if (strlen(basename) >= G__LONGLINE) {
         G__fprinterr(G__serr, "Limitation: class name too long. Must be < %d"
                      , G__LONGLINE);
         G__genericerror((char*)NULL);
      }

      /* [struct|class] <tagname> : <private|public> base1 , base2
       *                                            ^  or ^         */

      if (strcmp(basename, "virtual") == 0) {
#ifndef G__VIRTUALBASE
         if (G__NOLINK == G__globalcomp && G__NOLINK == G__store_globalcomp)
            G__genericerror("Limitation: virtual base class not supported in interpretation");
#endif
         c = G__fgetname_template(basename, "{,");
         isvirtualbase = G__ISVIRTUALBASE;
         if (strlen(basename) >= G__LONGLINE) {
            G__fprinterr(G__serr, "Limitation: class name too long. Must be < %d"
                         , G__LONGLINE);
            G__genericerror((char*)NULL);
         }
      }

      baseaccess = G__PUBLIC;
      if (type == 'c') {
         baseaccess = G__PRIVATE;
      }
      if (!strcmp(basename, "public")) {
         baseaccess = G__PUBLIC;
#ifdef G__TEMPLATECLASS
         c = G__fgetname_template(basename, "{,");
#else
         c = G__fgetname(basename, "{,");
#endif
         if (strlen(basename) >= G__LONGLINE) {
            G__fprinterr(G__serr, "Limitation: class name too long. Must be < %d"
                         , G__LONGLINE);
            G__genericerror((char*)NULL);
         }
      }
      else if (!strcmp(basename, "private")) {
         baseaccess = G__PRIVATE;
#ifdef G__TEMPLATECLASS
         c = G__fgetname_template(basename, "{,");
#else
         c = G__fgetname(basename, "{,");
#endif
         if (strlen(basename) >= G__LONGLINE) {
            G__fprinterr(G__serr, "Limitation: class name too long. Must be < %d"
                         , G__LONGLINE);
            G__genericerror((char*)NULL);
         }
      }
      else if (!strcmp(basename, "protected")) {
         baseaccess = G__PROTECTED;
#ifdef G__TEMPLATECLASS
         c = G__fgetname_template(basename, "{,");
#else
         c = G__fgetname(basename, "{,");
#endif
         if (strlen(basename) >= G__LONGLINE) {
            G__fprinterr(G__serr, "Limitation: class name too long. Must be < %d"
                         , G__LONGLINE);
            G__genericerror((char*)NULL);
         }
      }

      if (!strcmp(basename, "virtual")) {
         // --
#ifndef G__VIRTUALBASE
         if (G__NOLINK == G__globalcomp && G__NOLINK == G__store_globalcomp) {
            G__genericerror("Limitation: virtual base class not supported in interpretation");
         }
#endif // G__VIRTUALBASE
         c = G__fgetname_template(basename, "{,");
         isvirtualbase = G__ISVIRTUALBASE;
         if (strlen(basename) >= G__LONGLINE) {
            G__fprinterr(G__serr, "Limitation: class name too long. Must be < %d", G__LONGLINE);
            G__genericerror(0);
         }
      }

      if (strlen(basename) && isspace(c)) {
         /* maybe basename is namespace that got cut because
          * G__fgetname_template stop at spaces and the user add:
          * class MyClass : public MyNamespace ::MyTopClass !
          * or
          * class MyClass : public MyNamespace:: MyTopClass !
         */
         int namespace_tagnum;
         G__StrBuf temp_sb(G__LONGLINE);
         char *temp = temp_sb;

         namespace_tagnum = G__defined_tagname(basename, 2);
         while ((((namespace_tagnum != -1)
                  && (G__struct.type[namespace_tagnum] == 'n'))
                 || (strcmp("std", basename) == 0)
                 || (basename[strlen(basename)-1] == ':'))
                && isspace(c)) {
            c = G__fgetname_template(temp, "{,");
            strcat(basename, temp);
            namespace_tagnum = G__defined_tagname(basename, 2);
         }
      }

      if (newdecl) {
         ::Reflex::Scope lstore_tagnum = G__tagnum;
         ::Reflex::Scope lstore_def_tagnum = G__def_tagnum;
         ::Reflex::Scope lstore_tagdefining = G__tagdefining;
         int lstore_def_struct_member = G__def_struct_member;
         G__tagnum = lstore_tagnum.DeclaringScope();
         G__def_tagnum = G__tagnum;
         G__tagdefining = G__tagnum;
         if (G__tagnum && !G__tagnum.IsTopScope()) {
            G__def_struct_member = 1;
         }
         else {
            G__def_struct_member = 0;
         }
         /* copy pointer for readability */
         /* member = G__struct.memvar[lstore_tagnum]; */
         baseclass = G__struct.baseclass[G__get_tagnum(lstore_tagnum)];
         basen = baseclass->vec.size();
         // Enter parsed information into base class information table.
         // Note: We are requiring the base class to exist here, we get an error message if it does not.
         // And set the base class access (private|protected|public).
         baseclass->vec.push_back(G__inheritance::G__Entry(G__defined_tagname(basename, 0), 0, baseaccess,
                                                           G__ISDIRECTINHERIT + isvirtualbase));
         // Calculate the base class offset.
         long current_size = G__struct.size[G__get_tagnum(lstore_tagnum)];
         // or? ((::Reflex::Type) lstore_tagnum).SizeOf()
         if (
            (current_size == 1) &&
            (!lstore_tagnum.MemberSize() == 0) &&
            (G__struct.baseclass[G__get_tagnum(lstore_tagnum)]->vec.empty())
         ) {
            baseclass->vec[basen].baseoffset = 0;
         }
         else {
            // FIXME: ((::Reflex::Type)lstore_tagnum).SizeOf();
            baseclass->vec[basen].baseoffset = (char*) current_size;
         }
         {
            // -- Set the base class information in the Reflex data structures.
            if (lstore_tagnum.IsClass()) {
               // -- Definer is actually a class.
               //
               // Get the scope object for the base class by looking up the basename.
               Reflex::Scope baseclass_scope = G__Dict::GetDict().GetScope(G__defined_tagname(basename, 0));
               if (baseclass_scope.IsClass()) {
                  // -- Specified base is actually a class.
                  //
                  // Construct the base class modifiers based on
                  // what we have parsed.
                  unsigned int modifiers = 0;
                  if (baseaccess == G__PRIVATE) {
                     modifiers |= Reflex::PRIVATE;
                  }
                  else if (baseaccess == G__PROTECTED) {
                     modifiers |= Reflex::PROTECTED;
                  }
                  else if (baseaccess == G__PUBLIC) {
                     modifiers |= Reflex::PUBLIC;
                  }
                  if (isvirtualbase == G__ISVIRTUALBASE) {
                     modifiers |= Reflex::VIRTUAL;
                  }
                  // Add the base class to the class we are defining.
                  lstore_tagnum.AddBase(baseclass_scope, (Reflex::OffsetFunction) 0, modifiers);
               }
            }
         }
         G__tagnum = lstore_tagnum;
         G__def_tagnum = lstore_def_tagnum;
         G__tagdefining = lstore_tagdefining;
         G__def_struct_member = lstore_def_struct_member;
         /* virtual base class for interpretation to be implemented and
          * 2 limitation messages above should be deleted. */
         if (1 == G__struct.size[baseclass->vec[basen].basetagnum]
               && 0 == G__Dict::GetDict().GetScope(baseclass->vec[basen].basetagnum).DataMemberSize()
               && G__struct.baseclass[baseclass->vec[basen].basetagnum]->vec.empty()
            ) {
            if (isvirtualbase)
               G__struct.size[G__get_tagnum(G__tagnum)] += G__DOUBLEALLOC;
            else
               G__struct.size[G__get_tagnum(G__tagnum)] += 0;
         }
         else {
            if (isvirtualbase)
               G__struct.size[G__get_tagnum(G__tagnum)]
               += (G__struct.size[baseclass->vec[basen].basetagnum] + G__DOUBLEALLOC);
            else
               G__struct.size[G__get_tagnum(G__tagnum)]
               += G__struct.size[baseclass->vec[basen].basetagnum];
         }

         /*
          * inherit base class info, variable member, function member
          */
         G__inheritclass(G__get_tagnum(G__tagnum), baseclass->vec[basen].basetagnum, baseaccess);

         /* ++(*pbasen); */
      }

      /*
       * reading remaining space
       */
      if (isspace(c)) {
         c = G__fignorestream("{,");
      }

      // rewind one character if '{' terminated read.
      if (c == '{') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) G__disp_mask = 1;
      }
   }
   //
   // virtual base class isabstract count duplication check
   //
   baseclass = G__struct.baseclass[G__get_tagnum(G__tagnum)];
   // When it is not a new declaration, updating purecount is going to
   // make us fail because the rest of the code is not going to be run.
   // Anyway we already checked once.
   if (newdecl) {
      int purecount = 0;
      size_t lastdirect = 0;
      size_t ivb;
      for (ivb = 0; ivb < baseclass->vec.size(); ++ivb) {
         ::Reflex::Scope itab;

         if (baseclass->vec[ivb].property & G__ISDIRECTINHERIT) {
            lastdirect = ivb;
         }

#ifndef G__OLDIMPLEMENTATION2037
         // Insure the loading of the memfunc.
         G__incsetup_memfunc(baseclass->vec[ivb].basetagnum);
#endif

         itab = G__Dict::GetDict().GetScope(baseclass->vec[ivb].basetagnum);
         for (unsigned int ifunc = 0; ifunc < itab.FunctionMemberSize(); ++ifunc) {

            if (itab.FunctionMemberAt(ifunc).IsAbstract()) {
               /* Search to see if this function has an overrider.
                  If we get this class through virtual derivation, search
                  all classes; otherwise, search only those derived
                  from it. */
               size_t firstb, lastb;
               size_t b2;
               int found_flag = 0;

               if (baseclass->vec[ivb].property & G__ISVIRTUALBASE) {
                  firstb = 0;
                  lastb = baseclass->vec.size();
               }
               else {
                  firstb = lastdirect;
                  lastb = ivb;
               }

               for (b2 = firstb; b2 < lastb; ++b2) {
                  ::Reflex::Member found_tab;
                  int basetag;

                  if (b2 == ivb)
                     continue;

                  basetag = baseclass->vec[b2].basetagnum;
                  if (G__isanybase(baseclass->vec[ivb].basetagnum, basetag
                                   , (long)G__STATICRESOLUTION) < 0)
                     continue;

                  found_tab = G__ifunc_exist(itab.FunctionMemberAt(ifunc),
                                             G__Dict::GetDict().GetScope(basetag),                                                         true);
                  if (found_tab) {
                     found_flag = 1;
                     break;
                  }
               }

               if (!found_flag)
                  ++purecount;
            }
         }

      }
      //fprintf(stderr, "G__define_struct: Setting abstract cnt for '%s' to: %d\n", G__tagnum.Name(Reflex::SCOPED).c_str(), purecount);
      G__struct.isabstract[G__get_tagnum(G__tagnum)] = purecount;
   }
   //
   //  Process any member declarations.
   //
   if (c == '{') {
      // -- Member declarations.
      isclassdef = 1;
      if (!newdecl && (type != 'n')) { // Do not add members to a pre-existing class, enum, struct, or union.
         G__fgetc();
         c = G__fignorestream("}");
      }
      else { // A new declaration, or we are extending a namespace.
         // -- A new declaration, or we are extending a namespace.
         //
         //  Remember the file number and line number of the declaration.
         //
         G__struct.line_number[G__get_tagnum(G__tagnum)] = G__ifile.line_number;
         G__struct.filenum[G__get_tagnum(G__tagnum)] = G__ifile.filenum;
         G__get_properties(G__tagnum)->filenum = G__ifile.filenum;
         G__get_properties(G__tagnum)->linenum = G__ifile.line_number;
         // Save state.
         store_access = G__access;
         //
         //  Set default access for members.
         //
         G__access = G__PUBLIC;
         if (type == 'c') {
            G__access = G__PRIVATE;
         }
         if (type == 'e') { // enum
            // -- Process enumerator declarations.
#ifdef G__OLDIMPLEMENTATION1386_YET
            G__struct.size[G__def_tagnum] = G__INTALLOC;
#endif // G__OLDIMPLEMENTATION1386_YET
            G__fgetc(); // skip '{'
            enumval.ref = 0;
            enumval.obj.i = -1;
            G__value_typenum(enumval) = G__tagnum;
            G__constvar = G__CONSTVAR;
            G__enumdef = 1;
            do {
               int store_decl = 0 ;
               c = G__fgetstream(memname, "=,}");
               if (c != '=') {
                  ++enumval.obj.i;
               }
               else {
                  char store_var_typeX = G__var_type;
                  ::Reflex::Scope store_tagnumX = G__tagnum;
                  ::Reflex::Scope store_def_tagnumX = G__def_tagnum;
                  G__var_type = 'p';
                  G__tagnum = ::Reflex::Scope();
                  G__def_tagnum = ::Reflex::Scope();
                  c = G__fgetstream(val, ",}");
                  store_prerun = G__prerun;
                  G__prerun = 0;
                  enumval = G__getexpr(val);
                  G__prerun = store_prerun;
                  G__var_type = store_var_typeX;
                  G__tagnum = store_tagnumX;
                  G__def_tagnum = store_def_tagnumX;
               }
               G__constvar = G__CONSTVAR;
               G__enumdef = 1;
               G__var_type = 'i';
               if (store_tagnum && !store_tagnum.IsTopScope()) {
                  store_def_struct_member = G__def_struct_member;
                  G__def_struct_member = 0;
                  G__static_alloc = 1;
                  store_decl = G__decl;
                  G__decl = 1;
               }
               //fprintf(stderr, "G__define_struct: Setting enum '%-32s' member '%-16s' to value %d\n", G__p_local.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str(), memname, enumval.obj.i);
               G__letvariable(memname, enumval,::Reflex::Scope::GlobalScope() , G__p_local);
               //G__dumpreflex_atlevel(G__p_local, 0);
               if (store_tagnum && !store_tagnum.IsTopScope()) {
                  G__def_struct_member = store_def_struct_member;
                  G__static_alloc = 0;
                  G__decl = store_decl;
               }
            }
            while (c != '}');
            G__constvar = 0;
            G__enumdef = 0;
         }
         else { // class, namespace, struct or union
            // -- Process member declaration for class, namespace, struct, or union.
            //
            //  Save state.
            //
            store_prerun = G__prerun; // TODO: Why do we save this, we don't change it?
            store_local = G__p_local;
            store_def_struct_member = G__def_struct_member;
            ::Reflex::Scope store_tagdefining = G__tagdefining;
            store_static_alloc = G__static_alloc;
            //
            //  Initialize state for parsing
            //  the member declarations.
            //
            G__p_local = G__tagnum; // The scope to put the members in is the class we are parsing.
            G__def_struct_member = 1; // Tell the parser we are declaring members.
            // G__def_tagnum was set ealier, so the enclosing class is the class we are parsing.
            G__tagdefining = G__tagnum; // The enclosed class is the class we are parsing.
            G__static_alloc = 0;
            //
            //  Now parse the member declarations all at once.
            //
            //fprintf(stderr, "G__define_struct: Starting parse of member declarations.\n");
            //fprintf(stderr, "G__define_struct:      G__tagnum: '::%s'\n", G__tagnum.Name(Reflex::SCOPED).c_str());
            //fprintf(stderr, "G__define_struct:  G__def_tagnum: '::%s'\n", G__def_tagnum.Name(Reflex::SCOPED).c_str());
            //fprintf(stderr, "G__define_struct: G__tagdefining: '::%s'\n", G__tagdefining.Name(Reflex::SCOPED).c_str());
            G__switch = 0; // TODO: Can this be removed?  We don't restore it.
            int brace_level = 0; // Tell the parser to process the entire struct block.
            G__exec_statement(&brace_level); // And call the parser.
            //fprintf(stderr, "G__define_struct: Finished parse of member declarations.\n");
            //
            //  Restore state.
            //
            G__static_alloc = store_static_alloc;
            G__tagnum = G__tagdefining; // TODO: Could this have been changed by the parse?
            G__tagdefining = store_tagdefining;
            G__def_struct_member = store_def_struct_member;
            G__p_local = store_local;
            G__prerun = store_prerun; // TODO: Why did we save this, we don't change it?
            //
            //  We have now processed all member
            //  declarations, decide on final padding.
            //
            if ( // We have only one data member and no base classes.
               (G__tagnum.DataMemberSize() == 1) && // We have only one data member, and
               G__struct.baseclass[G__get_tagnum(G__tagnum)]->vec.empty() // no base classes.
            ) {
               // TODO: this is still questionable, inherit0.c
               ::Reflex::Type var_type = G__tagnum.DataMemberAt(0).TypeOf(); // Get our only data member.
               if (G__get_type(var_type) == 'c') { // Our only data member is of type char or char*.
                  if (isupper(G__get_type(var_type))) { // First member is char*.
                     G__struct.size[G__get_tagnum(G__tagnum)] = G__get_varlabel(G__tagnum.DataMemberAt(0), 1) /* number of elements */ * G__LONGALLOC;
                  }
                  else { // First member is char.
                     G__value buf;
                     G__value_typenum(buf) = var_type;
                     G__struct.size[G__get_tagnum(G__tagnum)] = G__get_varlabel(G__tagnum.DataMemberAt(0), 1) /* number of elements */ * G__sizeof(&buf);
                  }
               }
            }
            else if (G__struct.size[G__get_tagnum(G__tagnum)] % G__DOUBLEALLOC) { // Not double aligned.
               // -- Padding for PA-RISC, Spark, etc.
               // If struct size can not be divided by G__DOUBLEALLOC the size is aligned.
               G__struct.size[G__get_tagnum(G__tagnum)] += G__DOUBLEALLOC - G__struct.size[G__get_tagnum(G__tagnum)] % G__DOUBLEALLOC;
            }
            //
            //  Force an empty class to have a size of one byte.
            //
            if (!G__struct.size[G__get_tagnum(G__tagnum)]) {
               G__struct.size[G__get_tagnum(G__tagnum)] = G__CHARALLOC;
            }
            //
            //  We now have the complete class in memory
            //  and the above code corrected the size for
            //  some special cases, so remember the new size.
            //
            if (G__tagnum.IsClass() || G__tagnum.IsUnion()) {
               Reflex::Type ty = G__tagnum;
               ty.SetSize(G__struct.size[G__get_tagnum(G__tagnum)]);
            }
            //G__tagdefining = store_tagdefining;
            //G__def_struct_member = store_def_struct_member;
            //G__p_local = store_local;
         }
         // Restore state.
         G__access = store_access;
      }
   }
   //
   // [struct|union|enum]   tagname   { member }  item ;
   //                                           ^
   //                     OR
   // [struct|union|enum]             { member }  item ;
   //                                           ^
   //                     OR
   // [struct|union|enum]   tagname     item ;
   //                                   ^
   // item declaration
   //
   G__var_type = 'u';
   if (type == 'e') {
      G__var_type = 'i';
   }
   if (ispointer) {
      G__var_type = toupper(G__var_type);
   }
   if (G__return > G__RETURN_NORMAL) {
      return;
   }
   if (type == 'u') { // union
      fpos_t pos;
      fgetpos(G__ifile.fp, &pos);
      int linenum = G__ifile.line_number;
      c = G__fgetstream(basename, ";");
      if (basename[0]) {
         fsetpos(G__ifile.fp, &pos);
         G__ifile.line_number = linenum;
         if (G__dispsource) {
            G__disp_mask = 1000;
         }
         G__define_var(G__get_tagnum(G__tagnum), ::Reflex::Type());
         G__disp_mask = 0;
      }
      else if (!strncmp(G__struct.name[G__get_tagnum(G__tagnum)], "$G__NONAME", 10)) { // anonymous union, maybe someday anonymous namespace
         // -- anonymous union
         G__add_anonymousunion(G__tagnum, G__def_struct_member, store_def_tagnum);
      }
   }
   else if (type == 'n') { // namespace
      // do nothing
   }
   else { // class or struct
      if (G__check_semicolumn_after_classdef(isclassdef)) {
         G__tagnum = store_tagnum;
         G__def_tagnum = store_def_tagnum;
         return;
      }
      G__def_tagnum = store_def_tagnum;
      G__define_var(G__get_tagnum(G__tagnum), ::Reflex::Type());
   }
   G__tagnum = store_tagnum;
   G__def_tagnum = store_def_tagnum;
}

//______________________________________________________________________________
void Cint::Internal::G__create_global_namespace()
{
   // add global scope as namespace
#ifdef __GNUC__
#else
#pragma message (FIXME("Remove this once scopes are in reflex!"))
#endif
   int i = G__struct.alltag;
   static char clnull[1] = "";
   G__struct.name[i] = clnull;
   G__struct.parent_tagnum[i] = -1;
   G__struct.userparam[i] = 0;
   G__struct.hash[i] = 0;
   G__struct.size[i] = 0;
   G__struct.type[i] = 'n';

   G__struct.baseclass[i] = new G__inheritance();
   G__struct.virtual_offset[i] = G__PVOID;
   G__struct.isabstract[i] = 0;

   G__struct.globalcomp[i] = G__NOLINK;
   G__struct.iscpplink[i] = 0;
   G__struct.protectedaccess[i] = 0;

   G__struct.line_number[i] = -1;
   G__struct.filenum[i] = -1;

   G__struct.istypedefed[i] = 0;

   G__struct.funcs[i] = 0;

   G__struct.istrace[i] = 0;
   G__struct.isbreak[i] = 0;

#ifdef G__FRIEND
   G__struct.friendtag[i] = (struct G__friendtag*)NULL;
#endif

   G__struct.incsetup_memvar[i] = 0;
   G__struct.incsetup_memfunc[i] = 0;
   G__struct.rootflag[i] = 0;
   G__struct.rootspecial[i] = (struct G__RootSpecial*)NULL;

   G__struct.isctor[i] = 0;

#ifndef G__OLDIMPLEMENTATION1503
   G__struct.defaulttypenum[i] = ::Reflex::Type();
#endif
   G__struct.vtable[i] = (void*)NULL;

   G__RflxProperties *prop = G__get_properties(Reflex::Scope::GlobalScope());
   if (prop) {
      prop->typenum = -1;
      prop->tagnum = i;
      prop->globalcomp = G__NOLINK;
      prop->autoload = 0;
   }
   G__Dict::GetDict().RegisterScope(i,Reflex::Scope::GlobalScope());
   G__struct.alltag++;
}

//______________________________________________________________________________
void Cint::Internal::G__create_bytecode_arena()
{
   // Create an artificial variable whose contents will be the storage area for bytecode.
   ::Reflex::Type ty = ::Reflex::ClassBuilder("% CINT byte code scratch arena %", typeid(::Reflex::UnknownType), 0, ::Reflex::CLASS).EnableCallback(false).ToType();
   G__Dict::GetDict().RegisterScope(1, ty);
   G__RflxProperties* prop = G__get_properties(ty);
   ty.SetSize(0);
   prop->typenum = -1;
   prop->tagnum = 1;
   prop->globalcomp = G__NOLINK;
   prop->autoload = 0;
   prop->isBytecodeArena = true;
   G__struct.parent_tagnum[1] = -1;
   G__struct.userparam[1] = 0;
   G__struct.name[1] = (char*) "% CINT byte code scratch arena %";
   G__struct.hash[1] = strlen(G__struct.name[1]);
   G__struct.size[1] = 0;
   G__struct.type[1] = 'c';
   G__struct.baseclass[1] = new G__inheritance();
   G__struct.virtual_offset[1] = G__PVOID;
   G__struct.isabstract[1] = 0;
   G__struct.globalcomp[1] = G__NOLINK;
   G__struct.iscpplink[1] = 0;
   G__struct.protectedaccess[1] = 0;
   G__struct.line_number[1] = -1;
   G__struct.filenum[1] = -1;
   G__struct.istypedefed[1] = 0;
   G__struct.funcs[1] = 0;
   G__struct.istrace[1] = 0;
   G__struct.isbreak[1] = 0;
#ifdef G__FRIEND
   G__struct.friendtag[1] = 0;
#endif // G__FRIEND
   G__struct.incsetup_memvar[1] = 0;
   G__struct.incsetup_memfunc[1] = 0;
   G__struct.rootflag[1] = 0;
   G__struct.rootspecial[1] = 0;
   G__struct.isctor[1] = 0;
#ifndef G__OLDIMPLEMENTATION1503
   G__struct.defaulttypenum[1] = ::Reflex::Type();
#endif // G__OLDIMPLEMENTATION1503
   G__struct.vtable[1] = 0;
   G__struct.alltag++;
}

#ifndef G__OLDIMPLEMENTATION2030
//______________________________________________________________________________
int Cint::Internal::G__callfunc0(G__value* result, const ::Reflex::Member ifunc, G__param* libp, void* p, int funcmatch)
{
   int stat = 0;
   char *store_struct_offset;
   int store_asm_exec;

   if (!ifunc) {
      /* The function is not defined or masked */
      *result = G__null;
      return(stat);
   }

   store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = (char*)p;
   store_asm_exec = G__asm_exec;
   G__asm_exec = 0;

   // this-pointer adjustment
   G__this_adjustment(ifunc);

#ifdef G__EXCEPTIONWRAPPER
   if (-1 == G__get_funcproperties(ifunc)->entry.size) {
      /* compiled function. should be stub */
      G__InterfaceMethod pfunc = (G__InterfaceMethod)G__get_funcproperties(ifunc)->entry.tp2f;
      stat = G__ExceptionWrapper(pfunc, result, (char*)NULL, libp, 1);
   }
   else if (G__BYTECODE_SUCCESS == G__get_funcproperties(ifunc)->entry.bytecodestatus) {
      /* bytecode function */
      struct G__bytecodefunc *pbc = G__get_funcproperties(ifunc)->entry.bytecode;
      stat = G__ExceptionWrapper(G__exec_bytecode, result, (char*)pbc, libp, 1);
   }
   else {
      /* interpreted function */
      /* stat=G__ExceptionWrapper(G__interpret_func,result,ifunc->funcname[ifn]
         ,libp,ifunc->hash[ifn]);  this was wrong! */
      G__StrBuf funcname_sb(G__LONGLINE);
      char* funcname = funcname_sb;
      strcpy(funcname, ifunc.Name().c_str());
      ::Reflex::Scope scope = ifunc.DeclaringScope();
      stat = G__interpret_func(result, funcname, libp, 0, scope, G__EXACT, funcmatch);
   }
#else // G__EXCEPTIONWRAPPER
   if (-1 == G__get_funcproperties(ifunc)->entry.size) {
      /* compiled function. should be stub */
      G__InterfaceMethod pfunc = (G__InterfaceMethod)G__get_funcproperties(ifunc)->entry.tp2f;
      stat = (*pfunc)(result, (char*)NULL, libp, 1);
   }
   else if (G__BYTECODE_SUCCESS == G__get_funcproperties(ifunc)->entry.bytecodestatus) {
      /* bytecode function */
      struct G__bytecodefunc *pbc = G__get_funcproperties(ifunc)->entry.bytecode;
      stat = G__exec_bytecode(result, (char*)pbc, libp, 1);
   }
   else {
      /* interpreted function */
      G__StrBuf funcname_sb(G__LONGLINE);
      char* funcname = funcname_sb;
      strcpy(funcname, ifunc.Name().c_str());
      ::Reflex::Scope scope = ifunc.DeclaringScope();
      stat = G__interpret_func(result, funcname, libp, 0, scope, G__EXACT, funcmatch);
   }
#endif // G__EXCEPTIONWRAPPER

   G__store_struct_offset = store_struct_offset;
   G__asm_exec = store_asm_exec;

   return stat;
}
#endif // G__OLDIMPLEMENTATION2030

#ifndef G__OLDIMPLEMENTATION2030
//______________________________________________________________________________
int Cint::Internal::G__calldtor(void* p, const Reflex::Scope tagnum, int isheap)
{
   int stat;
   G__value result;
   struct G__param para;
   long store_gvp;

   if (!tagnum) return(0);

   /* destructor must be the first function in the table. -> 2027 */
   std::string dest("~");
   dest += tagnum.Name();
   //int ifn=0;
   //ifunc = G__struct.memfunc[tagnum];
   ::Reflex::Member func(tagnum.FunctionMemberByName(dest.c_str()));

   store_gvp = G__getgvp();
   if (isheap) {
      /* p is deleted either with free() or delete */
      G__setgvp((long)G__PVOID);
   }
   else {
      /* In callfunc0, G__store_sturct_offset is also set to p.
       * Let G__operator_delete() return without doing anything */
      G__setgvp((long)p);
   }

   /* call destructor */
   para.paran = 0;
   para.parameter[0][0] = 0;
   para.para[0] = G__null;
   stat = G__callfunc0(&result, func, &para, p, G__TRYDESTRUCTOR);

   G__setgvp(store_gvp);

   if (isheap && -1 != G__get_funcproperties(func)->entry.size) {
      /* interpreted class */
      delete[](char*) p;
   }

   return(stat);
}
#endif // G__OLDIMPLEMENTATION2030

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
extern "C" int G__set_class_autoloading(int newvalue)
{
   int oldvalue =  G__enable_autoloading;
   G__enable_autoloading = newvalue;
   return oldvalue;
}

//______________________________________________________________________________
extern "C" void G__set_class_autoloading_callback(int (*p2f)(char*, char*))
{
   G__p_class_autoloading = p2f;
}

//______________________________________________________________________________
extern "C" char* G__get_class_autoloading_table(char* classname)
{
   // Return the autoload entries for the class called classname.
   int tagnum = G__defined_tagname(classname, 3);
   if (tagnum < 0) return 0;
   return G__struct.libname[tagnum];
}

//______________________________________________________________________________
extern "C" void G__set_class_autoloading_table(char* classname, char* libname)
{
   // Register the class named 'classname' as being available in library
   // 'libname' (I.e. the implementation or at least the dictionary for 
   // the class is in the given library.  The class is marked as 
   // 'autoload' to indicated that we known about it but have not yet
   // loaded its dictionary.
   // If libname==-1 then we 'undo' this behavior instead.
   
   int store_enable_autoloading = G__enable_autoloading;
   G__enable_autoloading = 0;
   // First check whether this is already defined as typedef.
   Reflex::Type typedf( G__find_typedef(classname, 3 /* no complaint if the template does not exist */ ) );
   if (typedf) {
      // The autoloading might actually be 'targeted' to the FinalType per se.
      // For example in the case of the STL, the autoload classname would be
      // vector<int> but this would be declared as a typedef to vector<int, allocator<int> >
      ::Reflex::Type final( typedf.FinalType() );
      if (final && final.SizeOf()==0) {
         //FIXME: please remove the char* cast!
         G__set_class_autoloading_table( (char*)final.Name(::Reflex::SCOPED).c_str(), libname );
      }
      // Let's do nothing in this case for now
      G__enable_autoloading = store_enable_autoloading;
      return;
   }
   int ntagnum = G__search_tagname(classname, G__CLASS_AUTOLOAD);
   if (libname == (void*)-1) {
      if (G__struct.type[ntagnum] != G__CLASS_AUTOLOAD) {
         if (G__struct.libname[ntagnum]) {
            free((void*)G__struct.libname[ntagnum]);
         }
         G__struct.libname[ntagnum] = 0;
      } else {
         if (G__struct.name[ntagnum][0]) {
            G__struct.name[ntagnum][0] = '@';
         }
         G__Dict::GetDict().GetType( ntagnum ).ToTypeBase()->HideName();
      }
      G__enable_autoloading = store_enable_autoloading;
      return;
   }
   ::Reflex::Scope tagnum = G__Dict::GetDict().GetScope(ntagnum);
   if (G__struct.libname[G__get_tagnum(tagnum)]) {
      free((void*)G__struct.libname[G__get_tagnum(tagnum)]);
   }
   G__struct.libname[G__get_tagnum(tagnum)] = (char*)malloc(strlen(libname) + 1);
   strcpy(G__struct.libname[G__get_tagnum(tagnum)], libname);

   char *p = 0;
   if ((p = strchr(classname, '<'))) {
      // If the class is a template instantiation we need
      // to also register the template itself so that the
      // properly recognize it.
      char *buf = new char[strlen(classname)+1];
      strcpy(buf, classname);
      buf[p-classname] = '\0';
      if (!G__defined_templateclass(buf)) {
         ::Reflex::Scope store_def_tagnum = G__def_tagnum;
         ::Reflex::Scope store_tagdefining = G__tagdefining;
         FILE* store_fp = G__ifile.fp;
         G__ifile.fp = (FILE*)NULL;
         G__def_tagnum = tagnum.DeclaringScope();
         G__tagdefining = tagnum.DeclaringScope();
         char *templatename = buf;
         for (int j = (p - classname); j >= 0 ; --j) {
            if (buf[j] == ':' && buf[j-1] == ':') {
               templatename = buf + j + 1;
               break;
            }
         }
         G__createtemplateclass(templatename, (struct G__Templatearg*)NULL, 0);
         G__ifile.fp = store_fp;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
      }
      delete [] buf;
   }
   G__enable_autoloading = store_enable_autoloading;
}

//______________________________________________________________________________
extern "C" int G__defined_tagname(const char* tagname, int noerror)
{
   // Scan tagname table and return tagnum.  WARNING: tagname may be modified if there is a template instantiation.
   //
   // If no match, error message is shown and -1 will be returned.
   //
   // If non zero value is given to second argument 'noerror',
   // the error message is suppressed.
   // 
   // noerror = 0   if not found try to instantiate template class
   //               if template is not found, display error
   // 
   //         = 1   if not found try to instantiate template class
   //               no error messages if template is not found
   // 
   //         = 2   if not found just return without trying template
   // 
   //         = 3   like 2, and no autoloading
   //
   // noerror & 0x1000: do not look in enclosing scope, i.e. G__tagnum is
   //               defining a fully qualified identifier. With this bit
   //               set, tagname="C" and G__tagnum=A::B will not find to A::C.
   // 
   static ::Reflex::NamespaceBuilder stdnp("std");
   int i;
   // Allow for 10 occurrences of T<S<U>> - the only case where tagname can grow
   // due to typename normalization.
   int len = strlen(tagname) + 10;
   G__StrBuf temp_sb(len);
   char* temp = temp_sb;
   G__StrBuf atom_tagname_sb(len);
   char* atom_tagname = atom_tagname_sb;
   G__StrBuf normalized_tagname_sb(len);
   char* normalized_tagname = normalized_tagname_sb;
   strcpy(normalized_tagname, tagname);
   switch (normalized_tagname[0]) {
      case '"':
      case '\'':
         return -1;
      case '\0': // Global namespace.
         return 0;
      case 'c':
         if (!strcmp(tagname, "const"))
            return -1;
   }
   bool enclosing = !(noerror & 0x1000);
   noerror &= ~0x1000;
   if (strchr(normalized_tagname, '>')) { // There is a template-id in the given tagname.
      // handles X<X<int>> as X<X<int> >
      {
         char* p = strstr(normalized_tagname, ">>");
         while (p) {
            ++p;
            strcpy(temp, p);
            *p = ' ';
            ++p;
            strcpy(p, temp);
            p = (char*) strstr(normalized_tagname, ">>");
         }
      }
      // handles X<int > as X<int>
      {
         char* p = strstr(normalized_tagname, " >");
         while (p) {
            if (p[-1] != '>') {
               strcpy(temp, p + 1);
               strcpy(p, temp);
            }
            ++p;
            p = strstr(p, " >");
         }
      }
      // handles X <int> as X<int>
      {
         char* p = strstr(normalized_tagname, " <");
         while (p) {
            strcpy(temp, p + 1);
            strcpy(p, temp);
            ++p;
            p = strstr(p, " <");
         }
      }
      // handles "X<int> "  as "X<int>"
      {
         char* p = strstr(normalized_tagname, "> ");
         while (p) {
            if (!strncmp(p, "> >", 3)) {
               p += 2;
            }
            else {
               strcpy(temp, p + 2);
               strcpy(p + 1, temp);
               ++p;
            }
            p = strstr(p, "> ");
         }
      }
      // handles X< int> as X<int>
      {
         char* p = strstr(normalized_tagname, "< ");
         while (p) {
            strcpy(temp, p + 2);
            strcpy(p + 1, temp);
            ++p;
            p = strstr(p, "< ");
         }
      }
      // handles X<int, int> as X<int,int>
      {
         char* p = strstr(normalized_tagname, ", ");
         while (p) {
            strcpy(temp, p + 2);
            strcpy(p + 1, temp);
            ++p;
            p = strstr(p, ", ");
         }
      }
   }
   // handle X<const const Y>
   {
      char* p = strstr(normalized_tagname, "const const ");
      while (p) {
         char* p1 = (p += 6);
         char* p2 = p + 6;
         while (*p2) {
            *p1++ = *p2++;
         }
         *p1 = 0;
         p = strstr(p, "const const ");
      }
   }
   if (isspace(normalized_tagname[0])) {
      strcpy(temp, normalized_tagname + 1);
   }
   else {
      strcpy(temp, normalized_tagname);
   }
   //
   //  Now get the name to lookup and
   //  the scope to look it up in.
   //
   ::Reflex::Scope env_tagnum;
   char* p = G__find_last_scope_operator(temp);
   if (!p) { // An unqualified name, use the current scope.
      strcpy(atom_tagname, temp);
      env_tagnum = G__get_envtagnum();
   }
   else { // A qualified name, find the specified scope.
      // A::B::C means we want A::B::C, not A::C, even if it exists.
      enclosing = false;
      strcpy(atom_tagname, p + 2);
      *p = '\0';
      int slen = p - temp;
      //assert(slen < G__LONGLINE);
      G__StrBuf given_scopename_sb(G__LONGLINE);
      char* given_scopename = given_scopename_sb;
      strncpy(given_scopename, temp, slen);
      // Note: Not really necessary, but make sure
      //       in the case that slen == 0, and provoke
      //       a valgrind error if slen >= G__LONGLINE.
      given_scopename[slen] = '\0';
      if (!slen) { // The last :: was at the beginning, use the global scope.
         env_tagnum = ::Reflex::Scope::GlobalScope();
      }
#ifndef G__STD_NAMESPACE
      else if (G__ignore_stdnamespace && (slen == 3) && !strcmp(given_scopename, "std")) {
         // -- A name qualified explicitly with std::, use the global scope for now.
         env_tagnum = ::Reflex::Scope::GlobalScope();
         tagname += 5;
         normalized_tagname += 5;
      }
#endif // G__STD_NAMESPACE
      else {
         // A qualified name, find the specified containing scope.
         // Recursively locate the containing scopes, from right to left.
         int tag = -1;
         // first try a typedef, so we don't trigger autoloading here:
         Reflex::Type env_typenum = G__find_typedef(given_scopename);
         if (env_typenum) {
            tag = G__get_tagnum(env_typenum.FinalType());
         } else {
            tag = G__defined_tagname(given_scopename,noerror);
         }
         if (tag == -1) {
            // Should never happen.
            // TODO: Give an error message here.
            return -1;
         }
         env_tagnum = G__Dict::GetDict().GetScope(tag);
         if (!env_tagnum) {
            // Should never happen.
            // TODO: Give an error message here.
            return -1;
         }
      }
   }
   //
   //  Now that we have the containing scope we can search it.
   //
   {
      ::Reflex::Scope scope = env_tagnum.LookupScope(atom_tagname);
      if (p && scope && (scope.DeclaringScope() != env_tagnum)) { // We found something, but not where we asked for it.
         ::Reflex::Scope decl_scope = scope.DeclaringScope();
         int dtagnum = G__get_tagnum(decl_scope);
         int etagnum = G__get_tagnum(env_tagnum);
         int tmpltagnum = G__get_tagnum(G__tmplt_def_tagnum);
         if (
            // --
#ifdef G__VIRTUALBASE
            (G__isanybase(dtagnum, etagnum, G__STATICRESOLUTION) != -1) ||
#else // G__VIRTUALBASE
            (G__isanybase(dtagnum, etagnum) != -1) ||
#endif // G__VIRTUALBASE
            (enclosing && G__isenclosingclass(decl_scope, env_tagnum)) ||
            (enclosing && G__isenclosingclassbase(decl_scope, env_tagnum)) ||
            (!p && (G__tmplt_def_tagnum == decl_scope)) ||
#ifdef G__VIRTUALBASE
            (G__isanybase(dtagnum, tmpltagnum, G__STATICRESOLUTION) != -1) ||
#else // G__VIRTUALBASE
            (G__isanybase(dtagnum, tmpltagnum) != -1) ||
#endif // G__VIRTUALBASE
            (enclosing && G__isenclosingclass(decl_scope, G__tmplt_def_tagnum)) ||
            (enclosing && G__isenclosingclassbase(decl_scope, G__tmplt_def_tagnum))
         ) {
            // -- We have found something in a base class, or an enclosing class.
         }
         else {
            scope = ::Reflex::Scope(); // Flag not found.
         }
      }
      if (scope) {
         // -- Success, we found the class/struct/union/enum/namespace.
         // Now try to autoload the class library, if requested.
         int tagnum = G__get_tagnum(scope);
         if (noerror < 3) {
            G__class_autoloading(&tagnum);
         }
         // And return the final result.
         return tagnum;
      }
   }
   if (!std::strlen(atom_tagname)) {
      //  If we searched for the empty string and failed,
      //  search for a dollar sign, which stands for an
      //  unnamed type, such as:
      //
      //       struct { ... } abc;
      //
      //  TODO: Do we still need to do this?  Does anybody actually search for the empty string?
      //
      std::strcpy(atom_tagname, "$");
      ::Reflex::Scope scope = env_tagnum.LookupScope(atom_tagname);
      if (p && (scope.DeclaringScope() != env_tagnum)) { // We found something, but not where we asked for it.
         ::Reflex::Scope decl_scope = scope.DeclaringScope();
         int dtagnum = G__get_tagnum(decl_scope);
         int etagnum = G__get_tagnum(env_tagnum);
         int tmpltagnum = G__get_tagnum(G__tmplt_def_tagnum);
         if (
            // --
#ifdef G__VIRTUALBASE
            (G__isanybase(dtagnum, etagnum, G__STATICRESOLUTION) != -1) ||
#else // G__VIRTUALBASE
            (G__isanybase(dtagnum, etagnum) != -1) ||
#endif // G__VIRTUALBASE
            (enclosing && G__isenclosingclass(decl_scope, env_tagnum)) ||
            (enclosing && G__isenclosingclassbase(decl_scope, env_tagnum)) ||
            (!p && (G__tmplt_def_tagnum == decl_scope)) ||
#ifdef G__VIRTUALBASE
            (G__isanybase(dtagnum, tmpltagnum, G__STATICRESOLUTION) != -1) ||
#else // G__VIRTUALBASE
            (G__isanybase(dtagnum, tmpltagnum) != -1) ||
#endif // G__VIRTUALBASE
            (enclosing && G__isenclosingclass(decl_scope, G__tmplt_def_tagnum)) ||
            (enclosing && G__isenclosingclassbase(decl_scope, G__tmplt_def_tagnum))
         ) {
            // We have found something in a base class, or an enclosing class.
         }
         else {
            scope = ::Reflex::Scope(); // Flag not found.
         }
      }
      if (scope) {
         // Success, we found the class/struct/union/enum/namespace.
         // Now try to autoload the class library, if requested.
         int tagnum = G__get_tagnum(scope);
         if (noerror < 3) {
            G__class_autoloading(&tagnum);
         }
         // And return the final result.
         return tagnum;
      }
   }
   //
   //  Not found, try instantiating a class template.
   //
   len = std::strlen(normalized_tagname);
   if ((normalized_tagname[len-1] == '>') &&
       (noerror < 2) &&
       ((len < 2) || (normalized_tagname[len-2] != '-'))) {
      if (G__loadingDLL) {
         G__fprinterr(G__serr, "Error: '%s' Incomplete template resolution in shared library", normalized_tagname);
         G__genericerror(0);
         G__fprinterr(G__serr, "Add following line in header for making dictionary\n");
         G__fprinterr(G__serr, "   #pragma link C++ class %s;\n", normalized_tagname);
         G__exit(-1);
         return -1;
      }

      char store_var_type = G__var_type;
      i = G__instantiate_templateclass(normalized_tagname, noerror);
      G__var_type = store_var_type;
      return i;
   }
   else if (noerror < 2) {
      G__Definedtemplateclass* deftmplt = G__defined_templateclass(normalized_tagname);
      if (deftmplt && deftmplt->def_para && deftmplt->def_para->default_parameter) {
         i = G__instantiate_templateclass(normalized_tagname, noerror);
         return i;
      }
   }
   //
   //  Not found, try a typedef.
   //
   ::Reflex::Type tp = env_tagnum.LookupType(atom_tagname);
   if (tp && tp.IsTypedef() && (!p || (tp.DeclaringScope() == env_tagnum))) {
      i = G__get_tagnum(tp);
      if (i != -1) {
         // Found a typedef.
         // Now autoload the class library, if requested.
         if (noerror < 3) {
            G__class_autoloading(&i);
         }
         return i;
      }
   }
   //
   //  Definitely not found at this point.
   //
   //  A hack, no error message if there is an
   //  operator character in the name.
   //
   // TODO: Why do we allow this?
   //
   {
      int i2 = 0;
      int cx;
      while ((cx = normalized_tagname[i2++])) {
         if (G__isoperator(cx)) {
            return -1;
         }
      }
   }
   //
   //  Not found, give error message if allowed.
   //
   if (!noerror) {
      //G__dumpreflex();
      G__fprinterr(G__serr, "G__defined_tagname: Error: class, struct, union or type '%s' is not defined  %s:%d ", tagname, __FILE__, __LINE__);
      G__genericerror(0);
   }
   // We failed, return the invalid tagnum.
   return -1;
}

//______________________________________________________________________________
extern "C" int G__search_tagname(const char* tagname, int type)
{
   // Scan tagname table and return tagnum. If not match, create
   // new tag type.  If type > 0xff, create new G__struct entry if not found.
   // autoload if !isupper(type & 0xff).  If type == 0xff means ptr but type == 0
   // (see G__parse_parameter_link).
   int i;
   int len;
   char* p;
#ifndef G__OLDIMPLEMENTATION1823
   G__StrBuf buf_sb(2 * G__BUFLEN);
   char* buf = buf_sb;
   G__StrBuf buf2_sb(2 * G__BUFLEN);
   char* buf2 = buf2_sb;
   char* temp = buf;
   char* atom_tagname = buf2;
#else // G__OLDIMPLEMENTATION1823
   G__StrBuf temp_sb(G__LONGLINE);
   char* temp = temp_sb;
   G__StrBuf atom_tagname_sb(G__LONGLINE);
   char* atom_tagname = atom_tagname_sb;
#endif // G__OLDIMPLEMENTATION1823
   int noerror = 0;
   if (type == G__CLASS_AUTOLOAD) {
      // no need to issue error message while uploading the autoloader information
      noerror = 2;
   }
   ::Reflex::Scope envtagnum;
   int isstructdecl = type > 0xff;
   type &= 0xff;
   bool isPointer = (type == 0xff) || isupper(type);
   if (type == 0xff) {
      type = 0;
   }
   type = tolower(type);
   // Search for tagname, autoload class if not ref or ptr.
   i = G__defined_tagname(tagname, isPointer ? 3 : 2); 
#ifndef G__OLDIMPLEMENTATION1823
   if (strlen(tagname) > ((G__BUFLEN * 2) - 10)) {
      temp = (char*) malloc(strlen(tagname) + 10);
      atom_tagname = (char*) malloc(strlen(tagname) + 10);
   }
#endif // G__OLDIMPLEMENTATION1823
   p = G__strrstr((char*) tagname, "::");
   if (p && !strchr(p, '>')) {
      strcpy(atom_tagname, tagname);
      p = G__strrstr(atom_tagname, "::");
      *p = 0;
      // first try a typedef, so we don't trigger autoloading here:
      Reflex::Type envtypenum = G__find_typedef(atom_tagname);
      if (envtypenum) {
         envtagnum = envtypenum.FinalType();
      } else {
         envtagnum = G__Dict::GetDict().GetScope(G__defined_tagname(atom_tagname, 1)); // Note: atom_tagname can be modified during this call.
      }      
   }
   else {
      envtagnum = G__get_envtagnum();
   }
   if ( // if new tagname, initialize tag table
      (i == -1) ||
      (
         isstructdecl &&
         (envtagnum != G__Dict::GetDict().GetScope(G__struct.parent_tagnum[i]))
      )
   ) {
      // Make sure first entry is the global namespace, and second entry is the bytecode arena.
      if (!G__struct.alltag) {
         G__create_global_namespace();
         G__create_bytecode_arena();
      }
      i = G__struct.alltag;
      ++G__struct.nactives;
      if (i == G__MAXSTRUCT) {
         G__fprinterr(G__serr, "Limitation: Number of struct/union tag exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXSTRUCT in G__ci.h and recompile %s\n", G__MAXSTRUCT, G__ifile.name, G__ifile.line_number, G__nam);
         G__eof = 1;
#ifndef G__OLDIMPLEMENTATION1823
         if (buf != temp) {
            free(temp);
         }
         if (buf2 != atom_tagname) {
            free(atom_tagname);
         }
#endif // G__OLDIMPLEMENTATION1823
         return -1;
      }
      strcpy(temp, tagname);
      p = G__find_last_scope_operator(temp);
      if (p) {
         strcpy(atom_tagname, p + 2);
         *p = '\0';
#ifndef G__STD_NAMESPACE
         if (G__ignore_stdnamespace && !strcmp(temp, "std")) {
            G__struct.parent_tagnum[i] = -1;
         }
         else {
            G__struct.parent_tagnum[i] = G__defined_tagname(temp, noerror | 0x1000);
         }
#else // G__STD_NAMESPACE
         G__struct.parent_tagnum[i] = G__defined_tagname(temp, noerror | 0x1000);
#endif // G__STD_NAMESPACE
         // --
      }
      else {
         ::Reflex::Scope env_tagnum;
         if (!G__def_tagnum.IsTopScope()) {
            if (G__def_tagnum != G__tagdefining) {
               env_tagnum = G__tagdefining;
            }
            else {
               env_tagnum = G__def_tagnum;
            }
         }
         G__struct.parent_tagnum[i] = G__get_tagnum(env_tagnum);
         strcpy(atom_tagname, temp);
      }
      if (!strncmp("$G__NONAME", atom_tagname, 10)) { // anonymous union, maybe someday anonymous namespace
         len = strlen(atom_tagname);
      }
      else {
         len = strlen(atom_tagname);
      }
      { // Do the Reflex part of the work.
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[i]);
         ::Reflex::Type cl;
         ::Reflex::Scope newscope;
         if (atom_tagname[0]) {
            newscope = G__findInScope(scope, atom_tagname);
         } 
         if (!newscope) {
            std::string fullname;
            if (G__struct.parent_tagnum[i] != -1) {
               fullname = G__fulltagname(G__struct.parent_tagnum[i], 0); // parentScope.Name(SCOPED);
               if (fullname.length()) {
                  fullname += "::";
               }
            }
            fullname += atom_tagname;
            switch (type) {
               case 0: // Unknown type.
                  // Note: When called from G__parse_parameter_link
                  //       for a function parameter with a type for
                  //       which we have not yet seen a declaration.
                  {
                     //fprintf(stderr, "G__search_tagname: New unknown type: '%s'\n", fullname.c_str());
                     cl = ::Reflex::ClassBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::CLASS).EnableCallback(false).ToType();
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 'a': // Autoloading.
                  {
                     //fprintf(stderr, "G__search_tagname: New autoloading type: '%s'\n", fullname.c_str());
                     cl = ::Reflex::ClassBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::CLASS).EnableCallback(false).ToType();
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 'c': // Class.
                  {
                     //fprintf(stderr, "G__search_tagname: New class type: '%s'\n", fullname.c_str());
                     cl = ::Reflex::ClassBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::CLASS).EnableCallback(false).ToType();
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 's': // Struct.
                  {
                     //fprintf(stderr, "G__search_tagname: New struct type: '%s'\n", fullname.c_str());
                     cl = ::Reflex::ClassBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::STRUCT).EnableCallback(false).ToType();
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 'n': // Namespace.
                  {
                     //fprintf(stderr, "G__search_tagname: New namespace scope: '%s'\n", fullname.c_str());
                     newscope = ::Reflex::NamespaceBuilder(fullname.c_str()).ToScope();
                     G__Dict::GetDict().RegisterScope(i, newscope);
                     break;
                  }
               case 'e': // Enum.
                  {
                     //fprintf(stderr, "G__search_tagname: New enum type: '%s'\n", fullname.c_str());
                     cl = ::Reflex::EnumTypeBuilder(fullname.c_str());
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 'u': // Union.
                  {
                     //fprintf(stderr, "G__search_tagname: New union type: '%s'\n", fullname.c_str());
                     cl = ::Reflex::UnionBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::UNION).EnableCallback(false).ToType();
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               default:
                  // -- Must not happen.
                  assert(false);
            }
         }
         else {
            // Reflex knows this class, but cint does not,
            // we must have been called from cintex which
            // is responding to a reflex class creation
            // callback.
            if (type != 'n') {
               cl = newscope;
            }
            switch (type) {
               case 0: // Unknown type.
                  // Note: When called from G__parse_parameter_link
                  //       for a function parameter with a type for
                  //       which we have not yet seen a declaration.
                  {
                     //fprintf(stderr, "G__search_tagname: New unknown type: '%s'\n", fullname.c_str());
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 'a': // Autoloading.
                  {
                     //fprintf(stderr, "G__search_tagname: New autoloading type: '%s'\n", fullname.c_str());
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 'c': // Class.
                  {
                     //fprintf(stderr, "G__search_tagname: New class type: '%s'\n", fullname.c_str());
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 's': // Struct.
                  {
                     //fprintf(stderr, "G__search_tagname: New struct type: '%s'\n", fullname.c_str());
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 'n': // Namespace.
                  {
                     //fprintf(stderr, "G__search_tagname: New namespace scope: '%s'\n", fullname.c_str());
                     G__Dict::GetDict().RegisterScope(i, newscope);
                    break;
                  }
               case 'e': // Enum.
                  {
                     //fprintf(stderr, "G__search_tagname: New enum type: '%s'\n", fullname.c_str());
                     //cl = ::Reflex::EnumTypeBuilder(fullname.c_str());
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               case 'u': // Union.
                  {
                     //fprintf(stderr, "G__search_tagname: New union type: '%s'\n", fullname.c_str());
                     G__Dict::GetDict().RegisterScope(i, cl);
                     break;
                  }
               default:
                  // -- Must not happen.
                  assert(false);
            }
         }
         G__RflxProperties* prop = 0;
         if (cl) {
            prop = G__get_properties(cl);
         }
         else {
            prop = G__get_properties(newscope);
         }
         if (prop) {
            prop->typenum = -1;
            prop->tagnum = i;
            prop->globalcomp = G__default_link ? G__globalcomp : G__NOLINK;
            prop->autoload = (type == 'a');
         }
      }
      G__struct.userparam[i] = 0;
      G__struct.name[i] = (char*) malloc((size_t)(len + 1));
      strcpy(G__struct.name[i], atom_tagname);
      G__struct.hash[i] = len;
      G__struct.size[i] = 0; // (type=='e') ? 4 : 0; // For consistency with Reflex (Need for cintexcompat)
      G__struct.type[i] = type; // 'c' class, 'e' enum, 'n', namespace, 's' struct, 'u' union
      G__struct.baseclass[i] = new G__inheritance();
      G__struct.virtual_offset[i] = G__PVOID; // -1 means no virtual function
      G__struct.isabstract[i] = 0;
      G__struct.globalcomp[i] = G__default_link ? G__globalcomp : G__NOLINK;
      G__struct.iscpplink[i] = 0;
      G__struct.protectedaccess[i] = 0;
      G__struct.line_number[i] = -1;
      G__struct.filenum[i] = -1;
      G__struct.istypedefed[i] = 0;
      G__struct.funcs[i] = 0;
      G__struct.istrace[i] = 0;
      G__struct.isbreak[i] = 0;
#ifdef G__FRIEND
      G__struct.friendtag[i] = 0;
#endif // G__FRIEND
      // G__setup_memfunc and G__setup_memvar pointers list initialization.
      G__struct.incsetup_memvar[i] = new std::list<G__incsetup>();
      G__struct.incsetup_memfunc[i] = new std::list<G__incsetup>();
      G__struct.rootflag[i] = 0;
      G__struct.rootspecial[i] = 0;
      G__struct.isctor[i] = 0;
#ifndef G__OLDIMPLEMENTATION1503
      G__struct.defaulttypenum[i] = ::Reflex::Type();
#endif // G__OLDIMPLEMENTATION1503
      G__struct.vtable[i] =  0;
      ++G__struct.alltag;
   }
   else if (!G__struct.type[i] || (G__struct.type[i] == 'a')) {
      G__struct.type[i] = type;
      ++G__struct.nactives;
   }
#ifndef G__OLDIMPLEMENTATION1823
   if (buf != temp) {
      free(temp);
   }
   if (buf2 != atom_tagname) {
      free(atom_tagname);
   }
#endif // G__OLDIMPLEMENTATION1823
   // Return tagnum.
   return i;
}

