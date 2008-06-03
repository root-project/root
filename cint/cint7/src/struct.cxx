
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
#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Builder/NamespaceBuilder.h"
#include "Reflex/Builder/ClassBuilder.h"
#include "Reflex/Builder/UnionBuilder.h"
#include "Dict.h"

#include <cctype>
#include <cstring>

using namespace Cint::Internal;

/******************************************************************
* G__check_semicolumn_after_classdef
******************************************************************/
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

/******************************************************************
* int G__using_namespace()
*
*  using  namespace [ns_name];  using directive   -> inheritance
*  using  [scope]::[member];    using declaration -> reference object
*        ^
*
* Note: using directive appears in global scope is not implemented yet
******************************************************************/
int Cint::Internal::G__using_namespace()
{
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
            int* pbasen;
            struct G__inheritance *base = G__struct.baseclass[G__get_tagnum(envtagnum)];
            pbasen = &base->basen;
            if (*pbasen < G__MAXBASE) {
               base->basetagnum[*pbasen] = G__get_tagnum(basetagnum);
               base->baseoffset[*pbasen] = 0;
               base->baseaccess[*pbasen] = G__PUBLIC;
               base->property[*pbasen] = 0;
               ++(*pbasen);
            }
            else {
               G__genericerror("Limitation: too many using directives");
            }
         }
      }
      else {
         /* using directive in global scope, to be implemented
         * 1. global scope has baseclass information
         * 2. G__searchvariable() looks for global scope baseclass
         */
         /* first check whether we already have this directive in
         memory */
         int j;
         int found;
         found = 0;
         for (j = 0; j < G__globalusingnamespace.basen; ++j) {
            struct G__inheritance *base = &G__globalusingnamespace;
            if (base->basetagnum[j] == G__get_tagnum(basetagnum)) {
               found = 1;
               break;
            }
         }
         if (!found) {
            if (G__globalusingnamespace.basen < G__MAXBASE) {
               struct G__inheritance *base = &G__globalusingnamespace;
               int* pbasen = &base->basen;
               base->basetagnum[*pbasen] = G__get_tagnum(basetagnum);
               base->baseoffset[*pbasen] = 0;
               base->baseaccess[*pbasen] = G__PUBLIC;
               ++(*pbasen);
            }
            else {
               G__genericerror("Limitation: too many using directives in global scope");
            }
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
      ::Reflex::Member member = G__find_variable(buf, hash, G__p_local,::Reflex::Scope::GlobalScope()
                                , &struct_offset, &store_struct_offset, &ig15, 1);
      if (member) {

         std::string varname(member.Name());
         ::Reflex::Scope varscope = ::Reflex::Scope::GlobalScope();
         if (G__p_local) varscope = G__p_local;

         ::Reflex::Member avar =
            G__add_scopemember(varscope, varname.c_str(), member.TypeOf()
                               , 0 /* should be member.Modifiers() */, member.Offset()
                               , G__get_offset(member), G__access, G__get_static(member));
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

/******************************************************************
* int G__get_envtagnum()
******************************************************************/
::Reflex::Scope Cint::Internal::G__get_envtagnum()
{
   if (G__def_tagnum && !G__def_tagnum.IsTopScope()) {
      /* In case of enclosed class definition, G__tagdefining remains
       * as enclosing class identity, while G__def_tagnum changes to
       * enclosed class identity. For finding environment scope, we
       * must use G__tagdefining */
      if (G__tagdefining != G__def_tagnum) return G__tagdefining;
      else                              return G__def_tagnum;
   }
   else if (G__exec_memberfunc) return G__memberfunc_tagnum;
   /* else if(-1!=G__func_now)    env_tagnum = -2-G__func_now; */
   return ::Reflex::Scope::GlobalScope();
}

/******************************************************************
* int G__isenclosingclass()
******************************************************************/
int Cint::Internal::G__isenclosingclass(const ::Reflex::Scope& enclosingtagnum, const ::Reflex::Scope& env_tagnum)
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

/******************************************************************
* int G__isenclosingclassbase()
******************************************************************/
int Cint::Internal::G__isenclosingclassbase(const ::Reflex::Scope& enclosingtagnum, const ::Reflex::Scope& env_tagnum)
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

/*******************************************************************/
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

/******************************************************************
* char* G__find_first_scope_operator(name) by Scott Snyder 1997/10/17
*
* Return a pointer to the first scope operator in name.
* Only those at the outermost level of template nesting are considered.
******************************************************************/
char* Cint::Internal::G__find_first_scope_operator(char *name)
{
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

/******************************************************************
* char* G__find_last_scope_operator(name)   by Scott Snyder 1997/10/17
*
* Return a pointer to the last scope operator in name.
* Only those at the outermost level of template nesting are considered.
******************************************************************/
char* Cint::Internal::G__find_last_scope_operator(char *name)
{
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

/******************************************************************
 * G__set_class_autoloading
 ******************************************************************/
static const char G__CLASS_AUTOLOAD = 'a';
static int G__enable_autoloading=1;
int (*G__p_class_autoloading)(char*,char*);

/************************************************************************
* G__set_class_autloading
************************************************************************/
extern "C" int G__set_class_autoloading(int newvalue)
{
   int oldvalue =  G__enable_autoloading;
   G__enable_autoloading = newvalue;
   return oldvalue;
}

/************************************************************************
* G__set_class_autoloading_callback
************************************************************************/
extern "C" void G__set_class_autoloading_callback(int (*p2f)(char*, char*))
{
   G__p_class_autoloading = p2f;
}

/******************************************************************
 * G__get_class_autoloading_table
 ******************************************************************/
extern "C" char* G__get_class_autoloading_table(char* classname)
{
   // Return the autoload entries for the class called classname.
   int tagnum = G__defined_tagname(classname, 3);
   if (tagnum < 0) return 0;
   return G__struct.libname[tagnum];
}

/******************************************************************
 * G__set_class_autoloading_table
 ******************************************************************/
extern "C" void G__set_class_autoloading_table(char *classname, char *libname)
{
   ::Reflex::Scope tagnum;
   G__enable_autoloading = 0;
   tagnum = G__Dict::GetDict().GetScope(G__search_tagname(classname, G__CLASS_AUTOLOAD));
   if (G__struct.libname[G__get_tagnum(tagnum)]) {
      free((void*)G__struct.libname[G__get_tagnum(tagnum)]);
   }
   G__struct.libname[G__get_tagnum(tagnum)] = (char*)malloc(strlen(libname) + 1);
   strcpy(G__struct.libname[G__get_tagnum(tagnum)], libname);
   G__enable_autoloading = 1;

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
}

/******************************************************************
 * G__class_autoloading
 ******************************************************************/
int Cint::Internal::G__class_autoloading(int tagnum)
{
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
         G__enable_autoloading = 0;
         // reset the def tagnums to not collide with dict setup
         ::Reflex::Scope store_def_tagnum = G__def_tagnum; 
         ::Reflex::Scope store_tagdefining = G__tagdefining; 
         G__def_tagnum = Reflex::Scope();
         G__tagdefining = Reflex::Scope();
         int res = (*G__p_class_autoloading)(G__fulltagname(tagnum, 1), copyLibname);
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         if (G__struct.type[tagnum] == G__CLASS_AUTOLOAD) {
            if (strstr(G__struct.name[tagnum],"<") != 0) {
               // Kill this entry.
               store_def_tagnum = G__def_tagnum; 
               store_tagdefining = G__tagdefining; 
               G__tagdefining = G__def_tagnum = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[tagnum]);
               int found_tagnum = G__defined_tagname(G__struct.name[tagnum],3);
               G__def_tagnum = store_def_tagnum;
               G__tagdefining = store_tagdefining;
               if (found_tagnum != tagnum) {
                  // The autoload has seemingly failed!
                  // This can happens in 'normal' case if the string representation of the
                  // type registered by the autoloading mechanism is actually a typedef
                  // to the real type (aka mytemp<Long64_t> vs mytemp<long long> or the
                  // stl containers with or without their (default) allocators.
                  char *old = G__struct.name[tagnum];

                  G__struct.name[tagnum] = (char*)malloc(strlen(old)+50);
                  strcpy(G__struct.name[tagnum],"@@ ex autload entry @@");
                  strcat(G__struct.name[tagnum],old);
                  G__struct.type[tagnum] = 0;
                  free(old);
               }
            }
         }
         G__enable_autoloading = 1;
         delete[] copyLibname;
         return res;
      }
      else if (libname && libname[0]) {
         // -- No autoload callback, try to load the library.
         G__enable_autoloading = 0;
         if (G__loadfile(copyLibname) >= G__LOADFILE_SUCCESS) {
            // -- Library load succeeded.
            G__enable_autoloading = 1;
            delete[] copyLibname;
            return 1;
         }
         else {
            // -- Library load failed.
            G__struct.type[tagnum] = G__CLASS_AUTOLOAD;
            G__enable_autoloading = 1;
            delete[] copyLibname;
            return -1;
         }
      }
      delete[] copyLibname;
   }
   return 0;
}

/******************************************************************/
::Reflex::Type Cint::Internal::G__find_type(const char *type_name, int /*errorflag*/, int /*templateflag*/)
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

/******************************************************************
* int G__defined_tagname(tagname,noerror)
*
* Description:
*   Scan tagname table and return tagnum. If not match, error message
*  is shown and -1 will be returned.
*  If non zero value is given to second argument 'noerror', error
*  message will be suppressed.
*
*  noerror = 0   if not found try to instantiate template class
*                if template is not found, display error
*          = 1   if not found try to instantiate template class
*                no error messages if template is not found
*          = 2   if not found just return without trying template
*          = 3   like 2, and no autoloading
*
* CAUTION:
*  If template class with constant argument is given to this function,
* tagname argument may be modified like below.
*    A<int,5*2> => A<int,10>
* This may cause unexpected side-effect.
******************************************************************/
extern "C" int G__defined_tagname(const char *tagname, int noerror)
{
   static ::Reflex::NamespaceBuilder stdnp("std");
   int i;
   int len;
   char *p;
   char temp[G__LONGLINE];
   char atom_tagname[G__LONGLINE];
   ::Reflex::Scope env_tagnum;

   switch (tagname[0]) {
      case '"':
      case '\'':
         return(-1);
      case '\0':
         // -- Global namespace.
         return 0;
   }

   if (strchr(tagname, '>')) {
      /* handles X<X<int>> as X<X<int> > */
      while ((char*)NULL != (p = (char*)strstr(tagname, ">>"))) {
         ++p;
         strcpy(temp, p);
         *p = ' ';
         ++p;
         strcpy(p, temp);
      }

      /* handles X<int > as X<int> */
      p = (char*)tagname;
      while ((char*)NULL != (p = (char*)strstr(p, " >"))) {
         if ('>' != *(p - 1)) {
            strcpy(temp, p + 1);
            strcpy(p, temp);
         }
         ++p;
      }
      /* handles X <int> as X<int> */
      p = (char*)tagname;
      while ((char*)NULL != (p = strstr(p, " <"))) {
         strcpy(temp, p + 1);
         strcpy(p, temp);
         ++p;
      }
      /* handles X<int>  as X<int> */
      p = (char*)tagname;
      while ((char*)NULL != (p = strstr(p, "> "))) {
         if (strncmp(p, "> >", 3) == 0) {
            p += 2;
         }
         else {
            strcpy(temp, p + 2);
            strcpy(p + 1, temp);
            ++p;
         }
      }
      /* handles X< int> as X<int> */
      p = (char*)tagname;
      while ((char*)NULL != (p = strstr(p, "< "))) {
         strcpy(temp, p + 2);
         strcpy(p + 1, temp);
         ++p;
      }
      // handles X<int, int> as X<int,int>
      p = (char*) tagname;
      while (0 != (p = strstr(p, ", "))) {
         strcpy(temp, p + 2);
         strcpy(p + 1, temp);
         ++p;
      }
   }

   /* handle X<const const Y> */
   p = (char*)strstr(tagname, "const const ");
   while (p) {
      char *p1 = (p += 6);
      char *p2 = p + 6;
      while (*p2) *p1++ = *p2++;
      *p1 = 0;
      p = strstr(p, "const const ");
   }

   if (isspace(tagname[0])) strcpy(temp, tagname + 1);
   else strcpy(temp, tagname);

   // Now set env_tagnum and atom_tagname.
   p = G__find_last_scope_operator(temp);
   if (!p) {
      // -- An unqualified name, use the current scope.
      strcpy(atom_tagname, temp);
      env_tagnum = G__get_envtagnum();
   }
   else {
      // -- A qualified name, find the specified scope.
      strcpy(atom_tagname, p + 2);
      *p = '\0';
      int slen = p - temp;
      //assert(slen < G__LONGLINE);
      G__StrBuf given_scopename_sb(G__LONGLINE);
      char *given_scopename = given_scopename_sb;
      strncpy(given_scopename, temp, slen);
      // Note: Not really necessary, but make sure
      //       in the case that slen == 0, and provoke
      //       a valgrind error if slen >= G__LONGLINE.
      given_scopename[slen] = '\0';
      if (!slen) {
         // -- The last :: was at the beginning, use the global scope.
         env_tagnum = ::Reflex::Scope::GlobalScope();
      }
#ifndef G__STD_NAMESPACE /* ON667 */
      else if (G__ignore_stdnamespace && (slen == 3) && !std::strcmp(given_scopename, "std")) {
         // -- A name qualified explicitly with std::, use the global scope for now.
         env_tagnum = ::Reflex::Scope::GlobalScope();
         tagname += 5;
         if (!*tagname) {
            // "std::"
         }
      }
#endif
      else {
         // -- A qualified name, find the specified containing scope.
         // Recursively locate the containing scopes, from right to left.
         // Note: use a temporary here, G__defined_tagname can alter its argument.
         strcpy(temp, given_scopename);
         int tag = G__defined_tagname(temp, noerror);
         // FIXME: If we didn't find the scope, we use the global scope,
         //       which is arguably wrong, we should just exit in error.
         env_tagnum = G__Dict::GetDict().GetScope(tag);
         if (!env_tagnum) {
            // -- Should never happen.
            // FIXME: Give an error message here.
            return -1;
         }
      }
   }
   try_again:
   // Now that we have the containing scope we can search it.
   len = std::strlen(atom_tagname);

   ::Reflex::Scope scope = env_tagnum.LookupScope(atom_tagname);
   if (scope) {
      // -- Success, we found the class/struct/union/enum/namespace.
      // Now try to autoload the class library, if requested.
      long tagnum = G__get_tagnum(scope);
      if (noerror < 3) G__class_autoloading(tagnum);
      // And return the final result.
      return tagnum;
   }

   if (!len) {
      // If we searched for the empty string and failed,
      // search for a dollar sign, which stands for an
      // unnamed type, such as:
      //      struct { ... } abc;
      // FIXME: Do we still need to do this?  Does anybody
      //        actually search for the empty string?
      std::strcpy(atom_tagname, "$");

      goto try_again;
   }

   /* if tagname not found, try instantiating class template */
   len = std::strlen(tagname);
   if ((tagname[len-1] == '>') && (noerror < 2) && ((len < 2) || (tagname[len-2] != '-'))) {
      if (G__loadingDLL) {
         G__fprinterr(G__serr, "Error: '%s' Incomplete template resolution in shared library", tagname);
         G__genericerror(0);
         G__fprinterr(G__serr, "Add following line in header for making dictionary\n");
         G__fprinterr(G__serr, "   #pragma link C++ class %s;\n", tagname);
         G__exit(-1);
         return -1;
      }
      // CAUTION: tagname may be modified in following function.
      char store_var_type = G__var_type;
      i = G__instantiate_templateclass((char*) tagname, noerror);
      G__var_type = store_var_type;
      return i;
   }
   else if (noerror < 2) {
      G__Definedtemplateclass* deftmplt = G__defined_templateclass((char*) tagname);
      if (deftmplt && deftmplt->def_para && deftmplt->def_para->default_parameter) {
         i = G__instantiate_templateclass((char*) tagname, noerror);
         return i;
      }
   }

   // Search for a typedef now.

   ::Reflex::Type tp = env_tagnum.LookupType(atom_tagname);
   if (tp && tp.IsTypedef()) {
      i = G__get_tagnum(tp);
      if (i != -1) {
         // -- Found a typedef.
         // Now autoload the class library, if requested.
         if (noerror < 3) G__class_autoloading(i);
         return i;
      }
   }

   // Definitely not found at this point.
   // A hack, no error message if there is an
   // operator character in the name.
   // FIXME: Why do we allow this?
   {
      int i2 = 0;
      int cx;
      while ((cx = tagname[i2++]))
         if (G__isoperator(cx))
            return -1;
   }
   // Not found.
   if (noerror == 0) {
      //G__dumpreflex();
      G__fprinterr(G__serr, "G__defined_tagname: Error: class, struct, union or type '%s' is not defined  %s:%d ", tagname, __FILE__, __LINE__);
      G__genericerror(0);
   }
   // We failed, return an error code.
   return -1;
}

/******************************************************************/
static void G__create_global_namespace()
{
   // add global scope as namespace
#ifdef __GNUC__
#else
#pragma message (FIXME("Remove this once scopes are in reflex!"))
#endif
   int i = G__struct.alltag;
   G__struct.name[i] = "";
   G__struct.parent_tagnum[i] = -1;
   G__struct.userparam[i] = 0;
   G__struct.hash[i] = 0;
   G__struct.size[i] = 0;
   G__struct.type[i] = 'n';

   G__struct.baseclass[i] = (struct G__inheritance *)malloc(sizeof(struct G__inheritance));
   memset(G__struct.baseclass[i], 0, sizeof(struct G__inheritance));
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

   G__struct.comment[i].p.com = (char*)NULL;
   G__struct.comment[i].filenum = -1;

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
   G__struct.alltag++;
}

/******************************************************************
* int G__search_tagname(tagname,type)
*
* Description:
*   Scan tagname table and return tagnum. If not match, create
*  new tag type.
* if type > 0xff, create new G__struct entry if not found;
* autoload if !isupper(type&0xff). type==0xff means ptr but type==0
* (see v6_newlink.cxx:G__parse_parameter_link)
*
******************************************************************/
extern "C" int G__search_tagname(const char *tagname, int type)
{
   int i , len;
   char *p;
#ifndef G__OLDIMPLEMENTATION1823
   G__StrBuf buf_sb(G__BUFLEN*2);
   char *buf = buf_sb;
   G__StrBuf buf2_sb(G__BUFLEN*2);
   char *buf2 = buf2_sb;
   char *temp = buf;
   char *atom_tagname = buf2;
#else
   G__StrBuf temp_sb(G__LONGLINE);
   char *temp = temp_sb;
   G__StrBuf atom_tagname_sb(G__LONGLINE);
   char *atom_tagname = atom_tagname_sb;
#endif
   int noerror = 0;
   if (type == G__CLASS_AUTOLOAD) {
      /* no need to issue error message while uploading
      the autoloader information */
      noerror = 2;
   }
   /* int parent_tagnum; */
   ::Reflex::Scope envtagnum;
   int isstructdecl = type > 0xff;
   type &= 0xff;
   bool isPointer = (type == 0xff) || isupper(type);
   if (type == 0xff) type = 0;
   type = tolower(type);

   // Search for old tagname
   // Only auto-load struct if not ref / ptr
   i = G__defined_tagname(tagname, isPointer ? 3 : 2);

#ifndef G__OLDIMPLEMENTATION1823
   if (strlen(tagname) > G__BUFLEN*2 - 10) {
      temp = (char*)malloc(strlen(tagname) + 10);
      atom_tagname = (char*)malloc(strlen(tagname) + 10);
   }
#endif


   p = G__strrstr((char*)tagname, "::");
   if (p
         && !strchr(p, '>')
      ) {
      strcpy(atom_tagname, tagname);
      p = G__strrstr(atom_tagname, "::");
      *p = 0;
      envtagnum = G__Dict::GetDict().GetScope(G__defined_tagname(atom_tagname, 1));
   }
   else {
      envtagnum = G__get_envtagnum();
   }

   /* if new tagname, initialize tag table */
   if (-1 == i
         || (envtagnum != G__Dict::GetDict().GetScope(G__struct.parent_tagnum[i]) && isstructdecl)
      ) {

      // Make sure first entry is the global namespace.
      if (G__struct.alltag == 0) {
         G__create_global_namespace();
      }

      i = G__struct.alltag;

      if (i == G__MAXSTRUCT) {
         G__fprinterr(G__serr,
                      "Limitation: Number of struct/union tag exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXSTRUCT in G__ci.h and recompile %s\n"
                      , G__MAXSTRUCT
                      , G__ifile.name
                      , G__ifile.line_number
                      , G__nam);

         G__eof = 1;
#ifndef G__OLDIMPLEMENTATION1823
         if (buf != temp) free((void*)temp);
         if (buf2 != atom_tagname) free((void*)atom_tagname);
#endif
         return(-1);
      }

      strcpy(temp, tagname);
      p = G__find_last_scope_operator(temp);
      if (p) {
         strcpy(atom_tagname, p + 2);
         *p = '\0';
#ifndef G__STD_NAMESPACE /* ON667 */
         if (strcmp(temp, "std") == 0
               && G__ignore_stdnamespace
            ) G__struct.parent_tagnum[i] = -1;
         else G__struct.parent_tagnum[i] = G__defined_tagname(temp, noerror);
#else
         G__struct.parent_tagnum[i] = G__defined_tagname(temp, noerror);
#endif
      }
      else {
         ::Reflex::Scope env_tagnum;
         if (!G__def_tagnum.IsTopScope()) {
            if (G__tagdefining != G__def_tagnum) env_tagnum = G__tagdefining;
            else                              env_tagnum = G__def_tagnum;
         }
         else env_tagnum = ::Reflex::Scope();
         G__struct.parent_tagnum[i] = G__get_tagnum(env_tagnum);
         strcpy(atom_tagname, temp);
      }

      if (strncmp("$G__NONAME", atom_tagname, 10) == 0) {
         //atom_tagname[0]='\0';
         //len=0;
         len = strlen(atom_tagname);
      }
      else {
         len = strlen(atom_tagname);
      }

      {
         // REFLEX_NOT_COMPLETE
         // Create (if necessary) a place holder for this
         // class/namespace in the Reflex database.
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[i]);
         ::Reflex::Type cl;
         ::Reflex::Scope newscope;

         if (atom_tagname[0]) cl = G__findInScope(scope, atom_tagname);

         if (!cl) {
            std::string fullname;
            if (G__struct.parent_tagnum[i] != -1) {
               fullname = G__fulltagname(G__struct.parent_tagnum[i], 0); // parentScope.Name(SCOPED);
               if (fullname.length())
                  fullname += "::";
            }
            fullname += atom_tagname;

            switch (type) {
               case   0:
                  // -- Unknown type.
                  // Note: When called from G__parse_parameter_link
                  //       for a function parameter with a type for
                  //       which we have not yet seen a declaration.
               {
                  ::Reflex::ClassBuilder *b = new ::Reflex::ClassBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::CLASS);
                  cl =  b->ToType();
                  G__get_properties(cl)->builder.Set(b);
                  break;
               }
               case 'a':
                  // -- Autoloading.
               {
                  ::Reflex::ClassBuilder *b = new ::Reflex::ClassBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::CLASS);
                  cl =  b->ToType();
                  G__get_properties(cl)->builder.Set(b);
                  break;
               }
               case 'c':
                  // -- Class.
               {
                  ::Reflex::ClassBuilder *b = new ::Reflex::ClassBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::CLASS);   // Should also add the privacy with the containing class.
                  cl =  b->ToType();
                  G__get_properties(cl)->builder.Set(b);
                  break;
               }
               case 's':
                  // -- Struct.
               {
                  ::Reflex::ClassBuilder *b = new ::Reflex::ClassBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::STRUCT);   // Should also add the privacy with the containing class.
                  cl =  b->ToType();
                  G__get_properties(cl)->builder.Set(b);
                  break;
               }
               case 'n':
                  // -- Namespace.
               {
                  ::Reflex::NamespaceBuilder *b = new ::Reflex::NamespaceBuilder(fullname.c_str());
                  newscope =  b->ToScope();
                  G__get_properties(newscope)->builder.Set(b);
                  break;
               }
               case 'e':
                  // -- Enum.
               {
                  //::Reflex::EnumBuilder *b = new ::Reflex::EnumBuilder( fullname.c_str() );
                  //fprintf(stderr, "G__search_tagname: Building enum '%s'\n", fullname.c_str());
                  cl = ::Reflex::EnumTypeBuilder(fullname.c_str());
                  //G__get_properties(cl)->builder.Set(b);
                  break;
               }
               case 'u':
                  // -- Union.
                  // Note: We must have the space after the '<' here because
                  // '<:' is the alternative token for '[', see ISO/IEC 14882 (1998) [lex.digraph].
               {
                  ::Reflex::UnionBuilder* b = new ::Reflex::UnionBuilder(fullname.c_str(), typeid(::Reflex::UnknownType), 0, ::Reflex::UNION);
                  cl = b->ToType();
                  G__get_properties(cl)->builder.Set(b);
                  break;
               }
               default:
                  // -- Must not happen.
                  assert(false);
            }
         }

         G__RflxProperties* prop = 0;
         if (cl) prop = G__get_properties(cl);
         else prop = G__get_properties(newscope);
         if (prop) {
            prop->typenum = -1;
            prop->tagnum = i;
            prop->globalcomp = G__default_link ? G__globalcomp : G__NOLINK;
            prop->autoload = (type == 'a');
         }
      } // end REFLEX_NOT_COMPLETE block

      G__struct.userparam[i] = 0;
      G__struct.name[i] = (char*)malloc((size_t)(len + 1));
      strcpy(G__struct.name[i], atom_tagname);
      G__struct.hash[i] = len;

      G__struct.size[i] = 0;
      G__struct.type[i] = type; /* 's' struct ,'u' union ,'e' enum , 'c' class */

      //          /***********************************************************
      //          * Allocate and initialize member variable table
      //          ************************************************************/
      //          G__struct.memvar[i] = (struct G__var_array *)malloc(sizeof(struct G__var_array));
      // #ifdef G__OLDIMPLEMENTATION1776_YET
      //          memset(G__struct.memvar[i],0,sizeof(struct G__var_array));
      // #endif
      //          G__struct.memvar[i]->ifunc = (struct G__ifunc_table*)NULL;
      //          G__struct.memvar[i]->varlabel[0][0]=0;
      //          G__struct.memvar[i]->paran[0]=0;
      //          G__struct.memvar[i]->allvar=0;
      //          G__struct.memvar[i]->next = NULL;
      //          G__struct.memvar[i]->tagnum = i;
      //          {
      //             int ix;
      //             for(ix=0;ix<G__MEMDEPTH;ix++) {
      //                G__struct.memvar[i]->varnamebuf[ix]=(char*)NULL;
      //                G__struct.memvar[i]->p[ix] = 0;
      //             }
      //          }

      /***********************************************************
      * Allocate and initialize member function table list
      ***********************************************************/
      //          G__struct.memfunc[i] = (struct G__ifunc_table *)malloc(sizeof(struct G__ifunc_table));
      //          G__struct.memfunc[i]->allifunc = 0;
      //          G__struct.memfunc[i]->next = (struct G__ifunc_table *)NULL;
      //          G__struct.memfunc[i]->page = 0;
      // #ifdef G__NEWINHERIT
      //          G__struct.memfunc[i]->tagnum = i;
      // #endif
      //          {
      //             int ix;
      //             for(ix=0;ix<G__MAXIFUNC;ix++) {
      //                G__struct.memfunc[i]->funcname[ix]=(char*)NULL;
      //             }
      //          }
      // #ifndef G__OLDIMPLEMENTATION2027
      //          /* reserve the first entry for dtor */
      //          G__struct.memfunc[i]->hash[0]=0;
      //          G__struct.memfunc[i]->funcname[0]=(char*)malloc(2);
      //          G__struct.memfunc[i]->funcname[0][0]=0;
      //          G__struct.memfunc[i]->para_nu[0]=0;
      //          G__struct.memfunc[i]->pentry[0] = &G__struct.memfunc[i]->entry[0];
      //          G__struct.memfunc[i]->pentry[0]->bytecode=(struct G__bytecodefunc*)NULL;
      //          G__struct.memfunc[i]->friendtag[0]=(struct G__friendtag*)NULL;
      // #ifndef G__OLDIMPLEMENTATION2039
      //          G__struct.memfunc[i]->pentry[0]->size = 0;
      //          G__struct.memfunc[i]->pentry[0]->filenum = 0;
      //          G__struct.memfunc[i]->pentry[0]->line_number = 0;
      //          G__struct.memfunc[i]->pentry[0]->bytecodestatus = G__BYTECODE_NOTYET;
      //          G__struct.memfunc[i]->ispurevirtual[0] = 0;
      //          G__struct.memfunc[i]->access[0] = G__PUBLIC;
      //          G__struct.memfunc[i]->ansi[0] = 1;
      //          G__struct.memfunc[i]->isconst[0] = 0;
      //          G__struct.memfunc[i]->reftype[0] = 0;
      //          G__struct.memfunc[i]->type[0] = 0;
      //          G__struct.memfunc[i]->p_tagtable[0] = -1;
      //          // G__struct.memfunc[i]->p_typetable[0] = -1;
      //          G__struct.memfunc[i]->staticalloc[0] = 0;
      //          G__struct.memfunc[i]->busy[0] = 0;
      //          G__struct.memfunc[i]->isvirtual[0] = 0;
      //          G__struct.memfunc[i]->globalcomp[0] = G__NOLINK;
      // #endif

      //          G__struct.memfunc[i]->comment[0].filenum = -1;

      //          G__struct.memfunc[i]->comment[0].filenum = -1;

      //          {
      //             struct G__ifunc_table *store_ifunc;
      //             store_ifunc = G__p_ifunc;
      //             G__p_ifunc = G__struct.memfunc[i];
      //             G__memfunc_next();
      //             G__p_ifunc = store_ifunc;
      //          }
      // #endif

      /***********************************************************
      * Allocate and initialize class inheritance table
      ***********************************************************/
      G__struct.baseclass[i] = (struct G__inheritance *)malloc(sizeof(struct G__inheritance));
      G__struct.baseclass[i]->basen = 0;

      /***********************************************************
      * Initialize iden information for virtual function
      ***********************************************************/
      G__struct.virtual_offset[i] = G__PVOID; /* -1 means no virtual function */

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
      G__struct.friendtag[i] = (struct G__friendtag*)NULL;
#endif

      G__struct.comment[i].p.com = (char*)NULL;
      G__struct.comment[i].filenum = -1;

      // G__setup_memfunc and G__setup_memvar pointers list initialisation
      G__struct.incsetup_memvar[i] = new std::list<G__incsetup>();
      G__struct.incsetup_memfunc[i] = new std::list<G__incsetup>();

      G__struct.rootflag[i] = 0;
      G__struct.rootspecial[i] = (struct G__RootSpecial*)NULL;

      G__struct.isctor[i] = 0;

#ifndef G__OLDIMPLEMENTATION1503
      G__struct.defaulttypenum[i] = ::Reflex::Type();
#endif
      G__struct.vtable[i] = (void*)NULL;

      G__struct.alltag++;
   }
   else if (0 == G__struct.type[i]
            || 'a' == G__struct.type[i]
           ) {
      G__struct.type[i] = type;
   }

   /* return tag table number */
#ifndef G__OLDIMPLEMENTATION1823
   if (buf != temp) free((void*)temp);
   if (buf2 != atom_tagname) free((void*)atom_tagname);
#endif
   return(i);
}

/******************************************************************
* G__add_scopemember()
******************************************************************/
::Reflex::Member Cint::Internal::G__add_scopemember(::Reflex::Scope &envvar, const char *varname, const ::Reflex::Type &type, int reflex_modifiers, size_t reflex_offset, char *offset, int var_access, int var_statictype)
{
   // Create the Variable!
   int modifiers = reflex_modifiers;
   if (var_access) {
      modifiers &= ~(::Reflex::PUBLIC |::Reflex::PROTECTED |::Reflex::PRIVATE);
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
   // Mutable and Register are ignored
   if (var_statictype == G__LOCALSTATIC) {
      modifiers |= ::Reflex::STATIC;
   }

   envvar.AddDataMember(varname, type, reflex_offset, modifiers);
   ::Reflex::Member d = envvar.DataMemberByName(varname);

   G__get_offset(d) = offset;
   G__get_properties(d)->isCompiledGlobal = (var_statictype == G__COMPILEDGLOBAL);
   G__get_properties(d)->statictype = var_statictype; // We need this to distinguish file-static variables.

   return d;
}

/******************************************************************
* G__add_anonymousunion()
******************************************************************/
static void G__add_anonymousunion(const ::Reflex::Type &uniontype, int def_struct_member, ::Reflex::Scope &envtagnum)
{
   ::Reflex::Scope envvar;
   int statictype = G__AUTO;
   int store_statictype = G__AUTO;

   int access = G__PUBLIC;
   if (def_struct_member) {
      /* anonymous union as class/struct member */

      assert(envtagnum.IsClass());
      if (envtagnum.DataMemberSize() == 0) access = G__access;
      else {
         ::Reflex::Member prev(envtagnum.DataMemberAt(envtagnum.DataMemberSize() - 1));
         if (prev.IsPublic()) access = G__PUBLIC;
         else if (prev.IsProtected()) access = G__PROTECTED;
         else if (prev.IsPrivate()) access = G__PRIVATE;
         else {
            access = G__GRANDPRIVATE;
            assert(0);
         }
      }
      envvar = envtagnum;
   }
   else {
      /* variable body as global or local variable */

      if (G__p_local) envvar = G__p_local;
      else {
         envvar = ::Reflex::Scope::GlobalScope();
         statictype = G__ifile.filenum; /* file scope static */
      }
      store_statictype = G__COMPILEDGLOBAL;
   }

   char *offset = (char*)G__malloc(1, uniontype.SizeOf(), "");
   for (unsigned int ig15 = 0;ig15 < uniontype.DataMemberSize();ig15++) {
      ::Reflex::Member var = uniontype.DataMemberAt(ig15);

      ::Reflex::Member d(
         G__add_scopemember(envvar, var.Name().c_str(), var.TypeOf()
                            , 0 /* should be var.Modifiers() */
                            , var.Offset(), offset, access, statictype));

      *G__get_properties(d) = *G__get_properties(var);
      G__get_offset(d) = offset;
      G__get_properties(d)->isCompiledGlobal = (statictype == G__COMPILEDGLOBAL);

      statictype = store_statictype;
   }
}

/******************************************************************
* G__define_struct(type)
*
* [struct|union|enum] tagname { member } item ;
* [struct|union|enum]         { member } item ;
* [struct|union|enum] tagname            item ;
* [struct|union|enum] tagname { member }      ;
*
******************************************************************/
void G__dumpreflex_atlevel(const ::Reflex::Scope& scope, int level);
void Cint::Internal::G__define_struct(char type)
{
   //fprintf(stderr, "G__define_struct: Begin.\n");
   // struct G__input_file *fin;
   // fpos_t rewind_fpos;
   int c;
   G__StrBuf tagname_sb(G__LONGLINE);
   char *tagname = tagname_sb;
   char category[10];
   G__StrBuf memname_sb(G__ONELINE);
   char *memname = memname_sb;
   G__StrBuf val_sb(G__ONELINE);
   char *val = val_sb;
   ::Reflex::Scope store_tagnum;
   int store_def_struct_member = 0;
   ::Reflex::Scope store_local;
   G__value enumval;
   int store_access;
   G__StrBuf basename_sb(G__LONGLINE);
   char *basename = basename_sb;
   int* pbasen;
   struct G__inheritance* baseclass;
   int baseaccess;
   int newdecl;
   /* int lenheader; */
   int store_static_alloc;
   int len;
   int ispointer = 0;
   int store_prerun;
   ::Reflex::Scope store_def_tagnum;
   int isvirtualbase = 0;
   int isclassdef = 0;

#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg && G__asm_noverflow)
      G__fprinterr(G__serr, "LOOP COMPILE ABORTED FILE:%s LINE:%d\n"
                   , G__ifile.name
                   , G__ifile.line_number);
#endif
   G__abortbytecode();
#endif

   /*
    * [struct|union|enum]   tagname  { member }  item ;
    *                    ^
    * read tagname
    */

   c = G__fgetname_template(tagname, "{:;=&");

   if (strlen(tagname) >= G__LONGLINE) {
      G__fprinterr(G__serr, "Limitation: class name too long. Must be < %d"
                   , G__LONGLINE);
      G__genericerror((char*)NULL);
   }
   doitagain:
   /*
    * [struct|union|enum]   tagname{ member }  item ;
    *                               ^
    *                     OR
    * [struct|union|enum]          { member }  item ;
    *                               ^
    * push back before '{' and fgetpos
    */
   if (c == '{') {
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) G__disp_mask = 1;
   }

   /*
    * [struct|union|enum]   tagname   { member }  item ;
    *                               ^
    *                     OR
    * [struct|union|enum]   tagname     item ;
    *                               ^
    *                     OR
    * [struct|union|enum]   tagname      ;
    *                               ^
    * skip space and push back
    */
   else if (isspace(c)) {
      c = G__fgetspace(); /* '{' , 'a-zA-Z' or ';' are expected */
      /* if(c==';') return; */
      if (c != ':') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) G__disp_mask = 1;
      }
   }
   else if (c == ':') {
      /* inheritance or nested class */
      c = G__fgetc();
      if (':' == c) {
         strcat(tagname, "::");
         len = strlen(tagname);
         c = G__fgetname_template(tagname + len, "{:;=&");
         goto doitagain;
      }
      else {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) G__disp_mask = 1;
         c = ':';
      }
   }
   else if (c == ';') {
      /* tagname declaration */
   }
   else if (c == '=' && 'n' == type) {
      /* namespace alias=nsn; treat as typedef */
      c = G__fgetstream_template(basename, ";");
      int tagdefining = G__defined_tagname(basename, 0);
      if (-1 != tagdefining) {
         G__declare_typedef(tagname, 'u', tagdefining, 0,
                            0, G__default_link ? G__globalcomp : G__NOLINK,
                            G__get_tagnum(G__get_envtagnum()), true);
      }
      G__var_type = 'p';
      return;
   }
   else if (G__ansiheader && (',' == c || ')' == c)) {
      /* dummy argument for func overloading f(class A*) { } */
      G__var_type = 'p';
      if (')' == c) G__ansiheader = 0;
      return;
   }
   else if ('&' == c) {
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) G__disp_mask = 1;
      c = ' ';
   }
   else {
      G__genericerror("Syntax error in class/struct definition");
   }

   /*
    * set default tagname if tagname is omitted
    */
   if (tagname[0] == '\0') {
      if ('n' == type) {
#ifdef __GNUC__
#else
#pragma message (FIXME("What's an unnamed namespace in Reflex?"))
#endif
         /* unnamed namespace, treat as global scope, namespace has no effect.
          * This implementation may be wrong.
          * Should fix later with using directive in global scope */
         G__var_type = 'p';
         int brace_level = 0;
         G__exec_statement(&brace_level);
         return;
      }
      else {
         sprintf(tagname, "$G__NONAME%d_%s(%d)$", G__struct.alltag, G__ifile.name ? G__ifile.name : "<unknown file>", G__ifile.line_number);
      }
   }
#ifndef G__STD_NAMESPACE /* ON667 */
   else if ('n' == type && strcmp(tagname, "std") == 0
            && (G__ignore_stdnamespace
                || (G__def_tagnum && !G__def_tagnum.IsTopScope())
               )
           ) {
#ifdef __GNUC__
#else
#pragma message (FIXME("Fix treatment of namespace std!"))
#endif
      /* namespace std, treat as global scope, namespace has no effect. */
      G__var_type = 'p';
      int brace_level = 0;
      G__exec_statement(&brace_level);
      return;
   }
#endif

   /* BUG FIX, 17 Nov 1992
    *  tagnum wasn't saved
    */
   store_tagnum = G__tagnum;
   store_def_tagnum = G__def_tagnum;
   /*
    * Get tagnum, new tagtable is allocated if new
    */
   len = strlen(tagname);
   if (len && '*' == tagname[len-1]) {
      ispointer = 1;
      tagname[len-1] = '\0';
   }
   switch (c) {
      case '{':
      case ':':
      case ';':
         // 0x100: define struct if not found
         //fprintf(stderr, "G__define_struct: Creating scope '%s'\n", tagname);
         G__tagnum = G__Dict::GetDict().GetScope(G__search_tagname(tagname, type + 0x100));
         //G__dumpreflex_atlevel(G__tagnum, 0);
         break;
      default:
         G__tagnum = G__Dict::GetDict().GetScope(G__search_tagname(tagname, type));
         break;
   }

   if (';' == c) {
      /* in case of class name declaration 'class A;' */
      G__tagnum = store_tagnum;
      return;
   }
   if (!G__tagnum || G__tagnum.IsTopScope()) {
#ifdef __GNUC__
#else
#pragma message(FIXME("Shouldn't we warn the user that we cannot find the underlying type? Or can we register a typedef to uninitialized type?"))
#endif
      /* This case might not happen */
      G__fignorestream(";");
      G__tagnum = store_tagnum;
      return;
   }
   G__def_tagnum = G__tagnum;

   /*
    * judge if new declaration by size
    */
   if (G__struct.size[G__get_tagnum(G__tagnum)] == 0) {
      newdecl = 1;
   }
   else {
      newdecl = 0;
   }

   /* typenum is -1 for struct,union,enum without typedef */
   G__typenum = ::Reflex::Type();

   /* Now came to
    * [struct|union|enum]   tagname   { member }  item ;
    *                                 ^
    *                     OR
    * [struct|union|enum]             { member }  item ;
    *                                 ^
    *                     OR
    * [struct|union|enum]   tagname     item ;
    *                                   ^
    * member declaration if exist
    */

   /**************************************************************
    * base class declaration
    **************************************************************/
   if (c == ':') {
      c = ',';
   }
   while (c == ',') {
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
         pbasen = &baseclass->basen;
         // Enter parsed information into base class information table.
         baseclass->property[*pbasen] = G__ISDIRECTINHERIT + isvirtualbase;
         // Note: We are requiring the base class to exist here, we get an error message if it does not.
         baseclass->basetagnum[*pbasen] = G__defined_tagname(basename, 0);
         // Calculate the base class offset.
         int current_size = G__struct.size[G__get_tagnum(lstore_tagnum)];
         // or? ((::Reflex::Type) lstore_tagnum).SizeOf()
         if (
            (current_size == 1) &&
            (!lstore_tagnum.MemberSize() == 0) &&
            (G__struct.baseclass[G__get_tagnum(lstore_tagnum)]->basen == 0)
         ) {
            baseclass->baseoffset[*pbasen] = 0;
         }
         else {
            // FIXME: ((::Reflex::Type)lstore_tagnum).SizeOf();
            baseclass->baseoffset[*pbasen] = (char*) current_size;
         }
         // Set the base class access (private|protected|public).
         baseclass->baseaccess[*pbasen] = baseaccess;
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
                  G__get_properties(lstore_tagnum)->builder.Class().AddBase(baseclass_scope, 0, modifiers);
               }
            }
         }
         G__tagnum = lstore_tagnum;
         G__def_tagnum = lstore_def_tagnum;
         G__tagdefining = lstore_tagdefining;
         G__def_struct_member = lstore_def_struct_member;
         /* virtual base class for interpretation to be implemented and
          * 2 limitation messages above should be deleted. */
         if (1 == G__struct.size[baseclass->basetagnum[*pbasen]]
               && 0 == G__Dict::GetDict().GetScope(baseclass->basetagnum[*pbasen]).DataMemberSize()
               && 0 == G__struct.baseclass[baseclass->basetagnum[*pbasen]]->basen
            ) {
            if (isvirtualbase)
               G__struct.size[G__get_tagnum(G__tagnum)] += G__DOUBLEALLOC;
            else
               G__struct.size[G__get_tagnum(G__tagnum)] += 0;
         }
         else {
            if (isvirtualbase)
               G__struct.size[G__get_tagnum(G__tagnum)]
               += (G__struct.size[baseclass->basetagnum[*pbasen]] + G__DOUBLEALLOC);
            else
               G__struct.size[G__get_tagnum(G__tagnum)]
               += G__struct.size[baseclass->basetagnum[*pbasen]];
         }

         /*
          * inherit base class info, variable member, function member
          */
         G__inheritclass(G__get_tagnum(G__tagnum), baseclass->basetagnum[*pbasen], baseaccess);

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
   } // End of loop over each base class.


   /**************************************************************
    * virtual base class isabstract count duplication check
    **************************************************************/
   baseclass = G__struct.baseclass[G__get_tagnum(G__tagnum)];
   /* When it is not a new declaration, updating purecount is going to
      make us fail because the rest of the code is not going to be run.
      Anyway we already checked once. */
   if (newdecl) {
      int purecount = 0;
      int lastdirect = 0;
      int ivb;
      for (ivb = 0; ivb < baseclass->basen; ++ivb) {
         ::Reflex::Scope itab;

         if (baseclass->property[ivb] & G__ISDIRECTINHERIT) {
            lastdirect = ivb;
         }

#ifndef G__OLDIMPLEMENTATION2037
         // Insure the loading of the memfunc.
         G__incsetup_memfunc(baseclass->basetagnum[ivb]);
#endif

         itab = G__Dict::GetDict().GetScope(baseclass->basetagnum[ivb]);
         for (unsigned int ifunc = 0; ifunc < itab.FunctionMemberSize(); ++ifunc) {

            if (itab.FunctionMemberAt(ifunc).IsAbstract()) {
               /* Search to see if this function has an overrider.
                  If we get this class through virtual derivation, search
                  all classes; otherwise, search only those derived
                  from it. */
               int firstb, lastb;
               int b2;
               int found_flag = 0;

               if (baseclass->property[ivb] & G__ISVIRTUALBASE) {
                  firstb = 0;
                  lastb = baseclass->basen;
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

                  basetag = baseclass->basetagnum[b2];
                  if (G__isanybase(baseclass->basetagnum[ivb], basetag
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

   // fsetpos(G__ifile.fp,&rewind_fpos);
   if (c == '{') {
      // Member declarations.

      isclassdef = 1;

      if (newdecl || 'n' == type) {

         G__struct.line_number[G__get_tagnum(G__tagnum)] = G__ifile.line_number;
         G__struct.filenum[G__get_tagnum(G__tagnum)] = G__ifile.filenum;
         G__get_properties(G__tagnum)->filenum = G__ifile.filenum;
         G__get_properties(G__tagnum)->linenum = G__ifile.line_number;

         store_access = G__access;
         G__access = G__PUBLIC;
         switch (type) {
            case 's':
               sprintf(category, "struct");
               break;
            case 'c':
               sprintf(category, "class");
               G__access = G__PRIVATE;
               break;
            case 'u':
               sprintf(category, "union");
               break;
            case 'e':
               sprintf(category, "enum");
               break;
            case 'n':
               sprintf(category, "namespace");
               break;
            default:
               G__genericerror("Error: Illegal tagtype. struct,union,enum expected");
               break;
         }

         if (type == 'e') { /* enum */

#ifdef G__OLDIMPLEMENTATION1386_YET
            G__struct.size[G__def_tagnum] = G__INTALLOC;
#endif
            G__fgetc(); /* skip '{' */
            /* Change by Philippe Canal, 1999/8/26 */
            enumval.ref = 0;
            enumval.obj.i = -1;

            //FIXME Should the type be int or the enum?
            //enumval.type = 'i' ;
            //enumval.tagnum = G__get_tagnum(G__tagnum);
            //G__value_typenum(enumval) = G__get_from_type('i', 0);
            G__value_typenum(enumval) = G__tagnum;

            G__constvar = G__CONSTVAR;
            G__enumdef = 1;
            do {
               int store_decl = 0 ;
               c = G__fgetstream(memname, "=,}");
               if (c == '=') {
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
               else {
                  enumval.obj.i++;
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
            while (c != '}') ;
            G__constvar = 0;
            G__enumdef = 0;
            G__access = store_access;
         }

         else { /* class, struct or union */
            /********************************************
             * Parsing member declaration
             ********************************************/
            store_local = G__p_local;
            G__p_local = G__tagnum;

            store_def_struct_member = G__def_struct_member;
            G__def_struct_member = 1;
            G__switch = 0; /* redundant */
            store_static_alloc = G__static_alloc;
            G__static_alloc = 0;
            store_prerun = G__prerun;
            ::Reflex::Scope store_tagdefining = G__tagdefining;
            G__tagdefining = G__tagnum;

            // Tell the parser to process the entire struct block.
            int brace_level = 0;
            // And call the parser.
            G__exec_statement(&brace_level);

            G__tagnum = G__tagdefining;
            G__access = store_access;
            G__prerun = store_prerun;
            G__static_alloc = store_static_alloc;

            /********************************************
             * Padding for PA-RISC, Spark, etc
             * If struct size can not be divided by G__DOUBLEALLOC
             * the size is aligned.
             ********************************************/
            if (1 == G__tagnum.DataMemberSize()
                  && 0 == G__struct.baseclass[G__get_tagnum(G__tagnum)]->basen
               ) {
               /* this is still questionable, inherit0.c */
               ::Reflex::Type var_type = G__tagnum.DataMemberAt(0).TypeOf();
               if ('c' == G__get_type(var_type)) {
                  if (isupper(G__get_type(var_type))) {
                     G__struct.size[G__get_tagnum(G__tagnum)] = G__get_varlabel(G__tagnum.DataMemberAt(0), 1) /* number of elements */ * G__LONGALLOC;
                  }
                  else {
                     G__value buf;
                     G__value_typenum(buf) = var_type;
                     G__struct.size[G__get_tagnum(G__tagnum)] = G__get_varlabel(G__tagnum.DataMemberAt(0), 1) /* number of elements */ * G__sizeof(&buf);
                  }
               }
            }
            else
               if (G__struct.size[G__get_tagnum(G__tagnum)] % G__DOUBLEALLOC) {
                  G__struct.size[G__get_tagnum(G__tagnum)]
                  += G__DOUBLEALLOC - G__struct.size[G__get_tagnum(G__tagnum)] % G__DOUBLEALLOC;
               }
            if (0 == G__struct.size[G__get_tagnum(G__tagnum)]) {
               G__struct.size[G__get_tagnum(G__tagnum)] = G__CHARALLOC;
            }

            // We now have the complete class in memory and the above code corrected the
            // size of for some special case, let's store it in Reflex.

            if (G__tagnum.IsClass() || G__tagnum.IsUnion()) {
               G__get_properties(G__tagnum)->builder.Class().SetSizeOf(G__struct.size[G__get_tagnum(G__tagnum)]);
            }

            G__tagdefining = store_tagdefining;

            G__def_struct_member = store_def_struct_member;
            G__p_local = store_local;
         }
      }
      else { /* of newdecl */
         G__fgetc();
         c = G__fignorestream("}");
      }
   }


   /*
    * Now came to
    * [struct|union|enum]   tagname   { member }  item ;
    *                                           ^
    *                     OR
    * [struct|union|enum]             { member }  item ;
    *                                           ^
    *                     OR
    * [struct|union|enum]   tagname     item ;
    *                                   ^
    * item declaration
    */

   G__var_type = 'u';

   /* Need to think about this */
   if (type == 'e') G__var_type = 'i';

   if (ispointer) G__var_type = toupper(G__var_type);

   if (G__return > G__RETURN_NORMAL) return;

   if ('u' == type) { /* union */
      fpos_t pos;
      int linenum;
      fgetpos(G__ifile.fp, &pos);
      linenum = G__ifile.line_number;

      c = G__fgetstream(basename, ";");
      if (basename[0]) {
         fsetpos(G__ifile.fp, &pos);
         G__ifile.line_number = linenum;
         if (G__dispsource) G__disp_mask = 1000;
         G__define_var(G__get_tagnum(G__tagnum),::Reflex::Type());
         G__disp_mask = 0;
      }
      else if (!strncmp(G__struct.name[G__get_tagnum(G__tagnum)], "$G__NONAME", 10)) {
         /* anonymous union */
         G__add_anonymousunion(G__tagnum, G__def_struct_member, store_def_tagnum);
      }
   }
   else if ('n' == type) { /* namespace */
      /* no instance object for namespace, do nothing */
   }
   else { /* struct or class instance */
      if (G__check_semicolumn_after_classdef(isclassdef)) {
         G__tagnum = store_tagnum;
         G__def_tagnum = store_def_tagnum;
         return;
      }
      G__def_tagnum = store_def_tagnum;
      G__define_var(G__get_tagnum(G__tagnum),::Reflex::Type());
   }

   G__tagnum = store_tagnum;
   G__def_tagnum = store_def_tagnum;

#ifdef G__DEBUG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "G__tagnum=%d G__def_tagnum=%d G__def_struct_member=%d\n"
                   , G__tagnum, G__def_tagnum, G__def_struct_member);
      G__printlinenum();
   }
#endif
   // --
}

#ifndef G__OLDIMPLEMENTATION2030
/******************************************************************
 * G__callfunc0()
 ******************************************************************/
int Cint::Internal::G__callfunc0(G__value *result, const ::Reflex::Member &ifunc, G__param *libp, void *p, int funcmatch)
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
      stat = G__interpret_func(result, libp, 0 /* ifunc->hash[ifn] */, ifunc, G__EXACT, funcmatch);
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
      stat = G__interpret_func(result, libp, 0 /* ifunc->hash[ifn] */, ifunc, G__EXACT, funcmatch);
   }
#endif // G__EXCEPTIONWRAPPER

   G__store_struct_offset = store_struct_offset;
   G__asm_exec = store_asm_exec;

   return stat;
}

/******************************************************************
 * G__calldtor
 ******************************************************************/
int Cint::Internal::G__calldtor(void *p, const Reflex::Scope& tagnum, int isheap)
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
#endif

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
