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
#include "Api.h"
#include "DataMemberHandle.h"
#include <string>

extern "C" {

extern G__pMethodUpdateClassInfo G__UserSpecificUpdateClassInfo;

// Static functions.
static G__var_array* G__alloc_var_array(G__var_array* var, int* pig15);
static void G__copy_unionmember(G__var_array* var, int ig15, G__var_array* envvar, int envig15, long offset, int access, int statictype);
static void G__add_anonymousunion(int tagnum, int def_struct_member, int envtagnum);

// External functions.
int G__using_namespace();
int G__get_envtagnum();
int G__isenclosingclass(int enclosingtagnum, int env_tagnum);
int G__isenclosingclassbase(int enclosingtagnum, int env_tagnum);
const char* G__find_first_scope_operator(const char* name);
const char* G__find_last_scope_operator(const char* name);
void G__define_struct(char type);
int G__callfunc0(G__value* result, G__ifunc_table* iref, int ifn, G__param* libp, void* p, int funcmatch);
int G__calldtor(void* p, int tagnum, int isheap);

// Functions in the C interface.
int G__set_class_autoloading(int newvalue);
void G__set_class_autoloading_callback(int (*p2f)(char*, char*));
void G__set_class_autoloading_table(char* classname, char* libname);
char* G__get_class_autoloading_table(char* classname);
int G__defined_tagname(const char* tagname, int noerror);
int G__search_tagname(const char* tagname, int type);


//______________________________________________________________________________
char* G__savestring(char** pbuf, char* name)
{
   // -- FIXME: Describe this function!
   G__ASSERT(pbuf);
   if (*pbuf) {
      free((void*)(*pbuf));
      *pbuf = 0;
   }
   *pbuf = (char*) malloc(strlen(name) + 1);
   return strcpy(*pbuf, name); // Okay we allocated enough space
}


//______________________________________________________________________________
#define G__CLASS_AUTOLOAD 'a'
static int G__enable_autoloading = 1;
int (*G__p_class_autoloading)(char*, char*);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static G__var_array* G__alloc_var_array(G__var_array* var, int* pig15)
{
   // -- FIXME: Describe this function!
   // --
   if (var->allvar < G__MEMDEPTH) {
      *pig15 = var->allvar;
   }
   else {
      var->next = (struct G__var_array*) malloc(sizeof(struct G__var_array));
#ifndef G__OLDIMPLEMENTATION2038
      var->next->enclosing_scope = 0;
      var->next->inner_scope = 0;
#endif // G__OLDIMPLEMENTATION2038
      var->next->ifunc = 0;
      var->next->tagnum = var->tagnum;
      var = var->next;
      var->varlabel[0][0] = 0;
      var->paran[0] = 0;
      var->next = 0;
      var->allvar = 0;
      for (int i = 0; i < G__MEMDEPTH; ++i) {
         var->varnamebuf[i] = 0;
         var->p[i] = 0;
         var->is_init_aggregate_array[i] = 0;
      }
      *pig15 = 0;
   }
   return var;
}

//______________________________________________________________________________
static void G__copy_unionmember(G__var_array* var, int ig15, G__var_array* envvar, int envig15, long offset, int access, int statictype)
{
   // -- FIXME: Describe this function!
   envvar->p[envig15] = offset;
   G__savestring(&envvar->varnamebuf[envig15], var->varnamebuf[ig15]);
   envvar->hash[envig15] = var->hash[ig15];
   for (int i = 0; i < G__MAXVARDIM; ++i) {
      envvar->varlabel[envig15][i] = var->varlabel[ig15][i];
   }
   envvar->paran[envig15] = var->paran[ig15];
   envvar->bitfield[envig15] = var->bitfield[ig15];
   envvar->type[envig15] = var->type[ig15];
   envvar->constvar[envig15] = var->constvar[ig15];
   envvar->p_tagtable[envig15] = var->p_tagtable[ig15];
   envvar->p_typetable[envig15] = var->p_typetable[ig15];
   envvar->statictype[envig15] = statictype;
   envvar->reftype[envig15] = var->reftype[ig15];
   envvar->access[envig15] = access;
   envvar->globalcomp[envig15] = var->globalcomp[ig15];
   envvar->comment[envig15].p.com = var->comment[ig15].p.com;
   envvar->comment[envig15].p.pos = var->comment[ig15].p.pos;
   envvar->comment[envig15].filenum = var->comment[ig15].filenum;
}

//______________________________________________________________________________
static void G__add_anonymousunion(int tagnum, int def_struct_member, int envtagnum)
{
   // -- FIXME: Describe this function!
   int envig15;
   int ig15;
   struct G__var_array* var;
   struct G__var_array* envvar;
   long offset;
   int access;
   int statictype = G__AUTO;
   var = G__struct.memvar[tagnum];
   if (def_struct_member) {
      // anonymous union as class/struct member
      envvar = G__struct.memvar[envtagnum];
      while (envvar->next) {
         envvar = envvar->next;
      }
      envvar = G__alloc_var_array(envvar, &envig15);
      if (!envig15) {
         access = G__access;
      }
      else {
         access = envvar->access[envig15-1];
      }
      offset = G__malloc(1, G__struct.size[tagnum], "");
      while (var) {
         for (ig15 = 0; ig15 < var->allvar; ++ig15) {
            envvar = G__alloc_var_array(envvar, &envig15);
            G__copy_unionmember(var, ig15, envvar, envig15, offset, access, statictype);
            ++envvar->allvar;
         }
         var = var->next;
      }
   }
   else {
      // variable body as global or local variable.
      if (G__p_local) {
         envvar = G__p_local;
      }
      else {
         envvar = &G__global;
         statictype = G__ifile.filenum; // file scope static
      }
      while (envvar->next) {
         envvar = envvar->next;
      }
      envvar = G__alloc_var_array(envvar, &envig15);
      access = G__PUBLIC;
      offset = G__malloc(1, G__struct.size[tagnum], "");
      while (var) {
         for (ig15 = 0; ig15 < var->allvar; ++ig15) {
            envvar = G__alloc_var_array(envvar, &envig15);
            G__copy_unionmember(var, ig15, envvar, envig15, offset, access, statictype);
            statictype = G__COMPILEDGLOBAL;
            ++envvar->allvar;
         }
         var = var->next;
      }
   }
}

//______________________________________________________________________________
//
//  External functions.
//

//______________________________________________________________________________
int G__using_namespace()
{
   // -- Handle using namespace.
   //
   //  using  namespace [ns_name];  using directive   -> inheritance
   //  using  [scope]::[member];    using declaration -> reference object
   //       ^
   //
   //  Note: using directive appears in global scope is not implemented yet
   //
   int result = 0;
   G__FastAllocString buf(G__ONELINE);
   // Check if using directive or declaration.
   int c = G__fgetname_template(buf, 0, ";");
   if (!strcmp(buf, "namespace")) {
      // -- Using directive, treat as inheritance.
      int basetagnum, envtagnum;
      c = G__fgetstream_template(buf, 0, ";");
#ifndef G__STD_NAMESPACE
      if (';' == c && strcmp(buf, "std") == 0 && G__ignore_stdnamespace) {
         return 1;
      }
#endif // G__STD_NAMESPACE
      basetagnum = G__defined_tagname(buf, 2);
      if (basetagnum == -1) {
         G__fprinterr(G__serr, "Error: namespace %s is not defined", buf());
         G__genericerror(0);
         return 0;
      }
      if (G__def_struct_member) {
         // -- Using directive in other namespace or class/struct.
         envtagnum = G__get_envtagnum();
         if (envtagnum >= 0) {
            struct G__inheritance* base = G__struct.baseclass[envtagnum];
            int* pbasen = &base->basen;
            if (*pbasen < G__MAXBASE) {
               base->herit[*pbasen]->basetagnum = basetagnum;
               base->herit[*pbasen]->baseoffset = 0;
               base->herit[*pbasen]->baseaccess = G__PUBLIC;
               base->herit[*pbasen]->property = 0;
               ++(*pbasen);
            }
            else {
               G__genericerror("Limitation: too many using directives");
            }
         }
      }
      else {
         // -- Using directive in global scope, to be implemented.
         //
         // 1. global scope has baseclass information
         // 2. G__searchvariable() looks for global scope baseclass
         //
         // First check whether we already have this directive in memory.
         int j;
         int found;
         found = 0;
         for (j = 0; j < G__globalusingnamespace.basen; ++j) {
            struct G__inheritance *base = &G__globalusingnamespace;
            if (base->herit[j]->basetagnum == basetagnum) {
               found = 1;
               break;
            }
         }
         if (!found) {
            if (G__globalusingnamespace.basen < G__MAXBASE) {
               struct G__inheritance *base = &G__globalusingnamespace;
               int* pbasen = &base->basen;
               base->herit[*pbasen]->basetagnum = basetagnum;
               base->herit[*pbasen]->baseoffset = 0;
               base->herit[*pbasen]->baseaccess = G__PUBLIC;
               ++(*pbasen);
            }
            else {
               G__genericerror("Limitation: too many using directives in global scope");
            }
         }
         result = 1;
      }
   }
   else {
      // -- Using declaration, treat as reference object.
      int hash = 0;
      int ig15 = 0;
      G__hash(buf, hash, ig15);
      long struct_offset = 0;
      long store_struct_offset = 0;
      struct G__var_array* var = G__searchvariable(buf, hash, G__p_local, &G__global, &struct_offset, &store_struct_offset, &ig15, 1);
      if (var) {
         G__FastAllocString varname(buf);
         // Allocate a variable array entry which shares value storage with the found variable.
         int store_globalvarpointer = G__globalvarpointer;
         G__globalvarpointer = var->p[ig15];
         Cint::G__DataMemberHandle member;
         G__letvariable(varname, G__null, &G__global, G__p_local, member);
         G__globalvarpointer = store_globalvarpointer;
         // Now find the variable array entry we just created.
         int aig15 = 0;
         struct G__var_array* avar = member.GetVarArray();
         aig15 = member.GetIndex();
         // copy variable information
         if (avar && ((avar != var) || (aig15 != ig15))) {
            G__savestring(&avar->varnamebuf[aig15], var->varnamebuf[ig15]);
            avar->hash[aig15] = var->hash[ig15];
            for (int i = 0; i < G__MAXVARDIM; ++i) {
               avar->varlabel[aig15][i] = var->varlabel[ig15][i];
            }
            avar->paran[aig15] = var->paran[ig15];
            avar->bitfield[aig15] = var->bitfield[ig15];
            avar->type[aig15] = var->type[ig15];
            avar->constvar[aig15] = var->constvar[ig15];
            avar->p_tagtable[aig15] = var->p_tagtable[ig15];
            avar->p_typetable[aig15] = var->p_typetable[ig15];
            // Prevent double deletion during exit.
            switch (var->statictype[ig15]) {
               case G__LOCALSTATIC:
               case G__LOCALSTATICBODY:
               case G__COMPILEDGLOBAL:
                  avar->statictype[aig15] = G__USING_STATIC_VARIABLE;
                  break;
               default:
                  avar->statictype[aig15] = G__USING_VARIABLE;
                  break;
            }
            avar->reftype[aig15] = var->reftype[ig15];
            avar->globalcomp[aig15] = var->globalcomp[ig15];
            avar->comment[aig15] = var->comment[ig15];
         }
      }
      else {
         int tagnum = G__defined_tagname(buf, 1);
         if (tagnum != -1) {
            // -- Using scope::classname; to be implemented.
            // Now, G__tagtable is not ready.
         }
         else {
            result = 1;
         }
      }
   }
   return result;
}

//______________________________________________________________________________
int G__get_envtagnum()
{
   // -- Get nearest enclosing class scope.
   int env_tagnum = -1;
   if (G__def_tagnum != -1) {
      // -- We have an outer class scope.
      env_tagnum = G__def_tagnum;
      if (G__tagdefining != G__def_tagnum) {
         // -- We have an inner class scope.
         env_tagnum = G__tagdefining;
      }
   }
   else if (G__exec_memberfunc) {
      // -- Member function scope, use class of member.
      env_tagnum = G__memberfunc_tagnum;
   }
   return env_tagnum;
}

//______________________________________________________________________________
int G__isenclosingclass(int enclosingtagnum, int env_tagnum)
{
   // -- FIXME: Describe this function!
   int tagnum;
   if ((env_tagnum < 0) || (enclosingtagnum < 0)) {
      return 0;
   }
   tagnum = G__struct.parent_tagnum[env_tagnum];
   while (tagnum != -1) {
      if (tagnum == enclosingtagnum) {
         return 1;
      }
      tagnum = G__struct.parent_tagnum[tagnum];
   }
   return 0;
}

//______________________________________________________________________________
int G__isenclosingclassbase(int enclosingtagnum, int env_tagnum)
{
   // -- FIXME: Describe this function!
   int tagnum;
   if ((env_tagnum < 0) || (enclosingtagnum < 0)) {
      return 0;
   }
   tagnum = G__struct.parent_tagnum[env_tagnum];
   while (tagnum != -1) {
      int ret = G__isanybase(enclosingtagnum, tagnum, G__STATICRESOLUTION);
      if (ret != -1) {
         return 1;
      }
      if (tagnum == enclosingtagnum) {
         return 1;
      }
      tagnum = G__struct.parent_tagnum[tagnum];
   }
   return 0;
}

//______________________________________________________________________________
const char* G__find_first_scope_operator(const char* name)
{
   // -- Return a pointer to the first scope operator in name.
   // Only those at the outermost level of template nesting are considered.
   const char* p = name;
   int single_quote = 0;
   int double_quote = 0;
   int nest = 0;
   while (*p != '\0') {
      char c = *p;
      if (!single_quote && !double_quote) {
         if (c == '<') {
            ++nest;
         }
         else if ((nest > 0) && (c == '>')) {
            --nest;
         }
         else if (!nest && (c == ':') && (*(p + 1) == ':')) {
            return p;
         }
      }
      if ((c == '\'') && !double_quote) {
         single_quote = single_quote ^ 1;
      }
      else if ((c == '"') && !single_quote) {
         double_quote = double_quote ^ 1;
      }
      ++p;
   }
   return 0;
}

//______________________________________________________________________________
const char* G__find_last_scope_operator(const char* name)
{
   // -- Return a pointer to the last scope operator in name.
   // Only those at the outermost level of template nesting are considered.
   const char* p = name + strlen(name) - 1;
   int single_quote = 0;
   int double_quote = 0;
   int nest = 0;
   while (p > name) {
      char c = *p;
      if (!single_quote && !double_quote) {
         if (c == '>') {
            ++nest;
         }
         else if ((nest > 0) && (c == '<')) {
            --nest;
         }
         else if (!nest && (c == ':') && (*(p - 1) == ':')) {
            return p - 1;
         }
      }
      if ((c == '\'') && !double_quote) {
         single_quote = single_quote ^ 1;
      }
      else if ((c == '"') && !single_quote) {
         double_quote = double_quote ^ 1;
      }
      --p;
   }
   return 0;
}

//______________________________________________________________________________
int G__class_autoloading(int* ptagnum)
{
   // Load the library containing the class tagnum, according to
   // G__struct.libname[tagnum] set via G__set_class_autloading_table().
   // As a request to load vector<Long64_t> can result in vector<long long>
   // beging loaded, the requested tagnum and the loaded tagnum need not
   // be identical, i.e. G__class_autolading can change the tagnum to
   // point to the valid class with dictionary. The previous G__struct entry
   // is marked as an "ex autoload entry" so no name lookup can find it anymore.

   int& tagnum = *ptagnum;
   if ((tagnum < 0) || !G__enable_autoloading) {
      return 0;
   }
   // also autoload classes that were only forward declared
   if (
      (G__struct.type[tagnum] == G__CLASS_AUTOLOAD) ||
      ((G__struct.filenum[tagnum] == -1) && !G__struct.size[tagnum])
   ) {
      // --
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
      strcpy(copyLibname, libname); // Okay we allocated enough sapce
      if (G__p_class_autoloading) {
         int oldAutoLoading = G__enable_autoloading;
         G__enable_autoloading = 0;
         // reset the def tagnums to not collide with dict setup
         int store_def_tagnum = G__def_tagnum;
         int store_tagdefining = G__tagdefining;
         G__def_tagnum = -1;
         G__tagdefining = -1;
         int res = (*G__p_class_autoloading)(G__fulltagname(tagnum, 1), copyLibname);
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         if (G__struct.type[tagnum] == G__CLASS_AUTOLOAD) {
            // if (1 || strstr(G__struct.name[tagnum],"<") != 0) 
            {
               // Kill this entry.
               store_def_tagnum = G__def_tagnum;
               store_tagdefining = G__tagdefining;
               G__tagdefining = G__def_tagnum = G__struct.parent_tagnum[tagnum];
               // "hide" tagnum's name: we want to check whether this auto-loading loaded
               // another version of the same class, e.g. because of vector<Long64_t>
               // being requested but vector<long long> being loaded:
               std::string origName(G__struct.name[tagnum]);
               std::string fullName(G__fulltagname(tagnum,0));
               if (G__struct.name[tagnum][0]) {
                  G__struct.namerange->Remove(G__struct.name[tagnum], tagnum);
                  G__struct.name[tagnum][0] = '@';
               }
               int found_tagnum = G__defined_tagname(fullName.c_str(),3);
               if (G__struct.name[tagnum][0]) {
                  G__struct.name[tagnum][0] = origName[0];
                  G__struct.namerange->Insert(G__struct.name[tagnum], tagnum);
               }
               G__def_tagnum = store_def_tagnum;
               G__tagdefining = store_tagdefining;
               if (found_tagnum != -1) {
                  // The autoload has seemingly failed, yielding a different tagnum!
                  // This can happens in 'normal' case if the string representation of the
                  // type registered by the autoloading mechanism is actually a typedef
                  // to the real type (aka mytemp<Long64_t> vs mytemp<long long> or the
                  // stl containers with or without their (default) allocators.
                  char *old = G__struct.name[tagnum];
                  G__struct.namerange->Remove(old, tagnum);

                  G__struct.name[tagnum] = (char*)malloc(strlen(old)+50);
                  strcpy(G__struct.name[tagnum],"@@ ex autload entry @@"); // Okay, we allocated enough space
                  strcat(G__struct.name[tagnum],old); // Okay, we allocated enough space
                  G__struct.type[tagnum] = 0;
                  free(old);
                  tagnum = found_tagnum;
               }
            }
         }
         G__enable_autoloading = oldAutoLoading;
         delete[] copyLibname;
         return res;
      }
      else {
         int oldAutoLoading = G__enable_autoloading;
         G__enable_autoloading = 0;
         int ret = G__loadfile(copyLibname);
         if (ret >= G__LOADFILE_SUCCESS) {
            G__enable_autoloading = oldAutoLoading;
            delete[] copyLibname;
            return 1;
         }
         else {
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
void G__define_struct(char type)
{
   // -- Parse a class, enum, namespace, struct, or union declaration.
   //
   //  Note: This function is part of the parser proper.
   //
   //  [struct|union|enum] tagname { member } item;
   //  [struct|union|enum]         { member } item;
   //  [struct|union|enum] tagname            item;
   //  [struct|union|enum] tagname { member };
   //
   // Note: The type must be class, enum, namespace, struct, or union.
   G__ASSERT((type == 'c') || (type == 'e') || (type == 'n') || (type == 's') || (type == 'u'));
#ifdef G__ASM
   //
   //  Abort bytecode generation on seeing a struct declaration.
   //
#ifdef G__ASM_DBG
   if (G__asm_noverflow && G__asm_dbg) {
      G__fprinterr(G__serr, "LOOP COMPILE ABORTED FILE:%s LINE:%d\n", G__ifile.name, G__ifile.line_number);
   }
#endif // G__ASM_DBG
   G__abortbytecode();
#endif // G__ASM
   //
   // We have: [struct|union|enum]   tagname  { member }  item;
   //                             ^
   // Now read tagname.
   //
   G__FastAllocString tagname(G__LONGLINE);
   int c = G__fgetname_template(tagname, 0, "{:;=&");
   while (c == ':') {
      // -- Check for and handle nested name specifier.
      c = G__fgetc();
      if (c == ':') {
         tagname += "::";
         int len = strlen(tagname);
         c = G__fgetname_template(tagname, len, "{:;=&");
      }
      else {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         c = ':';
         break;
      }
   }
   if (c == '{') {
      // -- We have an open curly brace.
      //
      // [struct|union|enum]   tagname{ member }  item;
      //                               ^
      //                     OR
      // [struct|union|enum]          { member }  item;
      //                               ^
      // push back before '{' and fgetpos
      //
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
   }
   else if (isspace(c)) {
      // -- We have whitespace.
      //
      // [struct|union|enum]   tagname   { member }  item;
      //                               ^
      //                     OR
      // [struct|union|enum]   tagname     item;
      //                               ^
      //                     OR
      // [struct|union|enum]   tagname;
      //                               ^
      // skip space and push back
      //
      c = G__fgetspace(); // '{' , 'a-zA-Z' or ';' are expected
      if (c != ':') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
      }
   }
   else if (c == ':') {
      // -- A base class specifier list is present.
      // --
   }
   else if (c == ';') {
      // -- Tagname declaration.
      // --
   }
   else if ((c == '=') && (type == 'n')) {
      // -- We have: namespace alias=nsn; treat as typedef, handle and return.
      G__FastAllocString basename(G__LONGLINE);
      c = G__fgetstream_template(basename, 0, ";");
      int tagdefining = G__defined_tagname(basename, 0);
      if (tagdefining != -1) {
         int typenum;
         typenum = G__search_typename(tagname, 'u', tagdefining, 0);
         G__newtype.parent_tagnum[typenum] = G__get_envtagnum();
      }
      G__var_type = 'p';
      return;
   }
   else if (G__ansiheader && ((c == ',') || (c == ')'))) {
      // -- Dummy argument for func overloading f(class A*) {...}, handle and return.
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
   // Set default tagname if tagname is omitted.
   //
   if (!tagname[0]) {
      // -- We do not yet have a tagname.
      if (type == 'e') {
         // -- Unnamed enumeration, give it the name '$' (which may not be unique!).
         tagname = "$";
      }
      else if (type == 'n') {
         // -- Unnamed namespace, treat as global scope, namespace has no effect, handle and return.
         // This implementation may be wrong.
         // Should fix later with using directive in global scope.
         G__var_type = 'p';
         int brace_level = 0;
         G__exec_statement(&brace_level);
         return;
      }
      else {
         // -- Otherwise name it 'G__NONAMEddd', where ddd is the tagnum that will be assigned.
         tagname.Format("G__NONAME%d", G__struct.alltag);
      }
   }
#ifndef G__STD_NAMESPACE
   else if ((type == 'n') && !strcmp(tagname, "std") && (G__ignore_stdnamespace || (G__def_tagnum != -1))) {
      // -- Namespace std, treat as global scope, namespace has no effect, handle and return.
      G__var_type = 'p';
      int brace_level = 0;
      G__exec_statement(&brace_level);
      return;
   }
#endif // G__STD_NAMESPACE
   int store_tagnum = G__tagnum;
   int store_def_tagnum = G__def_tagnum;
   //
   //  Lookup the tagnum for this tagname, if not found, create a new tagtable entry.
   //
   int ispointer = 0;
   {
      int len = strlen(tagname);
      if (len && (tagname[len-1] == '*')) {
         ispointer = 1;
         tagname[len-1] = '\0';
      }
   }
   switch (c) {
      case '{':
      case ':':
      case ';':
         G__tagnum = G__search_tagname(tagname, type + 0x100); // 0x100: define struct if not found
         break;
      default:
         G__tagnum = G__search_tagname(tagname, type);
         break;
   }
   if (c == ';') {
      // -- Case of class name declaration 'class A;', handle and return;
      G__tagnum = store_tagnum;
      return;
   }
   if (G__tagnum < 0) {
      // -- This case might not happen, handle and return.
      G__fignorestream(";");
      G__tagnum = store_tagnum;
      return;
   }
   G__def_tagnum = G__tagnum;
   //
   // Judge if new declaration by size.
   //
   int newdecl = 0;
   if (!G__struct.size[G__tagnum]) {
      newdecl = 1;
   }
   // typenum is -1 for struct,union,enum without typedef.
   G__typenum = -1;
   //
   // Now we have:
   //
   // [struct|union|enum]   tagname   { member }  item;
   //                                 ^
   //                     OR
   // [struct|union|enum]             { member }  item;
   //                                 ^
   //                     OR
   // [struct|union|enum]   tagname     item;
   //                                   ^
   // member declaration if exist
   //
   if (c == ':') {
      // -- Base class declaration.
      c = ',';
   }
   while (c == ',') {
      // -- Handle base class specifiers.
      //
      // [struct|class] <tagname> : <private|protected|public|virtual> base_class {}
      //                           ^
      // Reset virtualbase flag.
      int isvirtualbase = 0;
      // Read base class name.
      G__FastAllocString basename(G__LONGLINE);
#ifdef G__TEMPLATECLASS
      c = G__fgetname_template(basename, 0, "{,");
#else // G__TEMPLATECLASS
      c = G__fgetname(basename, 0, "{,");
#endif // G__TEMPLATECLASS
      // [struct|class] <tagname> : <private|protected|public|virtual> base1 , base2 {}
      //                                                              ^  or ^
      if (!strcmp(basename, "virtual")) {
         // --
#ifndef G__VIRTUALBASE
         if ((G__globalcomp == G__NOLINK) && (G__store_globalcomp == G__NOLINK)) {
            G__genericerror("Limitation: virtual base class not supported in interpretation");
         }
#endif // G__VIRTUALBASE
         c = G__fgetname_template(basename, 0, "{,");
         isvirtualbase = G__ISVIRTUALBASE;
      }
      int baseaccess = G__PUBLIC;
      if (type == 'c') {
         baseaccess = G__PRIVATE;
      }
      if (!strcmp(basename, "public")) {
         baseaccess = G__PUBLIC;
#ifdef G__TEMPLATECLASS
         c = G__fgetname_template(basename, 0, "{,");
#else // G__TEMPLATECLASS
         c = G__fgetname(basename, 0, "{,");
#endif // G__TEMPLATECLASS
      }
      else if (!strcmp(basename, "private")) {
         baseaccess = G__PRIVATE;
#ifdef G__TEMPLATECLASS
         c = G__fgetname_template(basename, 0, "{,");
#else // G__TEMPLATECLASS
         c = G__fgetname(basename, 0, "{,");
#endif // G__TEMPLATECLASS
      }
      else if (!strcmp(basename, "protected")) {
         baseaccess = G__PROTECTED;
#ifdef G__TEMPLATECLASS
         c = G__fgetname_template(basename, 0, "{,");
#else // G__TEMPLATECLASS
         c = G__fgetname(basename, 0, "{,");
#endif // G__TEMPLATECLASS
      }
      if (!strcmp(basename, "virtual")) {
         // --
#ifndef G__VIRTUALBASE
         if ((G__globalcomp == G__NOLINK) && (G__store_globalcomp == G__NOLINK))
            G__genericerror("Limitation: virtual base class not supported in interpretation");
#endif // G__VIRTUALBASE
         c = G__fgetname_template(basename, 0, "{,");
         isvirtualbase = G__ISVIRTUALBASE;
      }
      if (strlen(basename) && isspace(c)) {
         // Maybe basename is namespace that got cut because
         // G__fgetname_template stops at spaces and the user wrote:
         //
         //      class MyClass : public MyNamespace ::MyTopClass
         //
         // or:
         //
         //      class MyClass : public MyNamespace:: MyTopClass
         //
         G__FastAllocString temp(G__LONGLINE);
         int namespace_tagnum = G__defined_tagname(basename, 2);
         while (
            // --
            isspace(c) &&
            (
               ((namespace_tagnum != -1) && (G__struct.type[namespace_tagnum] == 'n')) ||
               !strcmp(basename, "std") ||
               (basename[strlen(basename)-1] == ':')
            )
         ) {
            // --
            c = G__fgetname_template(temp, 0, "{,");
            basename += temp;
            namespace_tagnum = G__defined_tagname(basename, 2);
         }
      }
      if (newdecl) {
         int lstore_tagnum = G__tagnum;
         int lstore_def_tagnum = G__def_tagnum;
         int lstore_tagdefining = G__tagdefining;
         int lstore_def_struct_member = G__def_struct_member;
         G__tagnum = G__struct.parent_tagnum[lstore_tagnum];
         G__def_tagnum = G__tagnum;
         G__tagdefining = G__tagnum;
         G__def_struct_member = 0;
         if (G__tagnum != -1) {
            G__def_struct_member = 1;
         }
         // Copy pointer for readability.
         // member = G__struct.memvar[lstore_tagnum];
         struct G__inheritance* baseclass = G__struct.baseclass[lstore_tagnum];
         int* pbasen = &baseclass->basen;
         // Enter parsed information into base class information table.
         baseclass->herit[*pbasen]->property = G__ISDIRECTINHERIT + isvirtualbase;
         // Note: We are requiring the base class to exist here, we get an error message if it does not.
         baseclass->herit[*pbasen]->basetagnum = G__defined_tagname(basename, 0);
         if (
            // --
            (G__struct.size[lstore_tagnum] == 1) &&
            !G__struct.memvar[lstore_tagnum]->allvar &&
            !G__struct.baseclass[lstore_tagnum]->basen
         ) {
            // --
            baseclass->herit[*pbasen]->baseoffset = 0;
         }
         else {
            // --
            baseclass->herit[*pbasen]->baseoffset = G__struct.size[lstore_tagnum];
         }
         baseclass->herit[*pbasen]->baseaccess = baseaccess;
         G__tagnum = lstore_tagnum;
         G__def_tagnum = lstore_def_tagnum;
         G__tagdefining = lstore_tagdefining;
         G__def_struct_member = lstore_def_struct_member;
         // Virtual base classes for interpretation to be implemented
         // and the two limitation messages above should be deleted.
         if (
            // --
            (G__struct.size[baseclass->herit[*pbasen]->basetagnum] == 1) &&
            !G__struct.memvar[baseclass->herit[*pbasen]->basetagnum]->allvar &&
            !G__struct.baseclass[baseclass->herit[*pbasen]->basetagnum]->basen
         ) {
            // --
            if (isvirtualbase) {
               G__struct.size[G__tagnum] += G__DOUBLEALLOC;
            }
            else {
               G__struct.size[G__tagnum] += 0;
            }
         }
         else {
            if (isvirtualbase) {
               G__struct.size[G__tagnum] += (G__struct.size[baseclass->herit[*pbasen]->basetagnum] + G__DOUBLEALLOC);
            }
            else {
               G__struct.size[G__tagnum] += G__struct.size[baseclass->herit[*pbasen]->basetagnum];
            }
         }
         // Inherit base class info, variable member, function member.
         G__inheritclass(G__tagnum, baseclass->herit[*pbasen]->basetagnum, baseaccess);
      }
      // Read remaining whitespace.
      if (isspace(c)) {
         c = G__fignorestream("{,");
      }
      // Rewind one character if '{' terminated read.
      if (c == '{') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
      }
   }
   //
   // virtual base class isabstract count duplication check
   //
   struct G__inheritance* baseclass = G__struct.baseclass[G__tagnum];
   // When it is not a new declaration, updating purecount is going to
   // make us fail because the rest of the code is not going to be run.
   // Anyway we already checked once.
   if (newdecl) {
      int purecount = 0;
      int lastdirect = 0;
      int ivb;
      for (ivb = 0; ivb < baseclass->basen; ++ivb) {
         struct G__ifunc_table_internal* itab;
         if (baseclass->herit[ivb]->property & G__ISDIRECTINHERIT) {
            lastdirect = ivb;
         }
#ifndef G__OLDIMPLEMENTATION2037
         // Insure the loading of the memfunc.
         G__incsetup_memfunc(baseclass->herit[ivb]->basetagnum);
#endif // G__OLDIMPLEMENTATION2037
         itab = G__struct.memfunc[baseclass->herit[ivb]->basetagnum];
         while (itab) {
            for (int ifunc = 0; ifunc < itab->allifunc; ++ifunc) {
               if (itab->ispurevirtual[ifunc]) {
                  // Search to see if this function has an overrider.
                  // If we get this class through virtual derivation,
                  // search all classes; otherwise, search only those
                  // derived from it.
                  int firstb;
                  int lastb;
                  int found_flag = 0;
                  if (baseclass->herit[ivb]->property & G__ISVIRTUALBASE) {
                     firstb = 0;
                     lastb = baseclass->basen;
                  }
                  else {
                     firstb = lastdirect;
                     lastb = ivb;
                  }
                  for (int b2 = firstb; b2 < lastb; ++b2) {
                     struct G__ifunc_table_internal* found_tab;
                     int found_ndx;
                     int basetag;
                     if (b2 == ivb) {
                        continue;
                     }
                     basetag = baseclass->herit[b2]->basetagnum;
                     if (G__isanybase(baseclass->herit[ivb]->basetagnum, basetag, G__STATICRESOLUTION) < 0) {
                        continue;
                     }
                     found_tab = G__ifunc_exist(itab, ifunc, G__struct.memfunc[basetag], &found_ndx, 0xffff);
                     if (found_tab) {
                        found_flag = 1;
                        break;
                     }
                  }
                  if (!found_flag) {
                     ++purecount;
                  }
               }
            }
            itab = itab->next;
         }
      }
      G__struct.isabstract[G__tagnum] = purecount;
   }
   //
   //  Parse the member declarations.
   //
   int isclassdef = 0;
   if (c == '{') {
      // -- We have member declarations.
      isclassdef = 1;
      if (!newdecl && (type != 'n')) {
         G__fgetc();
         c = G__fignorestream("}");
      }
      else {
         G__struct.line_number[G__tagnum] = G__ifile.line_number;
         G__struct.filenum[G__tagnum] = G__ifile.filenum;
         int store_access = G__access;
         G__access = G__PUBLIC;
         switch (type) {
            case 'c':
               G__access = G__PRIVATE;
               break;
            case 'e':
            case 'n':
            case 's':
            case 'u':
               break;
            default:
               G__genericerror("Error: Illegal tagtype. struct,union,enum expected");
               break;
         }
         if (type == 'e') {
            // -- enum
#ifdef G__OLDIMPLEMENTATION1386_YET
            G__struct.size[G__def_tagnum] = G__INTALLOC;
#endif // G__OLDIMPLEMENTATION1386_YET
            // skip open curly brace.
            G__fgetc();
            G__value enumval;
            enumval.obj.reftype.reftype = G__PARANORMAL;
            enumval.ref = 0;
            enumval.obj.i = -1;
            enumval.type = 'i';
            enumval.tagnum = G__tagnum;
            enumval.typenum = -1;
#ifndef G__OLDIMPLEMENTATION1259
            enumval.isconst = 0;
#endif // G__OLDIMPLEMENTATION1259
            G__constvar = G__CONSTVAR;
            G__access = store_access;
            G__enumdef = 1;
            do {
               int store_decl = 0;
               G__FastAllocString memname(G__ONELINE);
               c = G__fgetstream_new(memname, 0, "=,}");
               if (c == '=') {
                  char store_var_typeX = G__var_type;
                  int store_tagnumX = G__tagnum;
                  int store_def_tagnumX = G__def_tagnum;
                  int store_tagdefiningX = G__tagdefining;
                  G__var_type = 'p';
                  G__tagnum = -1;
                  G__def_tagnum = store_tagnum;
                  G__tagdefining = store_tagnum;
                  G__FastAllocString val(G__ONELINE);
                  c = G__fgetstream_new(val, 0, ",}");
                  int store_prerun = G__prerun;
                  G__prerun = 0;
                  enumval = G__getexpr(val);
                  G__prerun = store_prerun;
                  G__var_type = store_var_typeX;
                  G__tagnum = store_tagnumX;
                  G__def_tagnum = store_def_tagnumX;
                  G__tagdefining = store_tagdefiningX;
               }
               else {
                  enumval.obj.i++;
               }
               G__constvar = G__CONSTVAR;
               G__enumdef = 1;
               G__var_type = 'i';
               int store_def_struct_member = 0;
               if (-1 != store_tagnum) {
                  store_def_struct_member = G__def_struct_member;
                  G__def_struct_member = 0;
                  G__static_alloc = 1;
                  store_decl = G__decl;
                  G__decl = 1;
               }
               G__DataMemberHandle member;
               G__letvariable(memname, enumval, &G__global , G__p_local, member);
               if (-1 != store_tagnum) {
                  G__def_struct_member = store_def_struct_member;
                  G__static_alloc = 0;
                  G__decl = store_decl;
               }
            }
            while (c != '}');
            G__constvar = 0;
            G__enumdef = 0;
            G__access = store_access;
         }
         else {
            // -- class, struct or union
            //
            //  Parsing a member declaration.
            //
            int store_prerun = G__prerun;
            G__prerun = 1;
            struct G__var_array* store_local = G__p_local;
            G__p_local = G__struct.memvar[G__tagnum];
            G__struct.memvar[G__tagnum]->prev_local = 0;
            int store_tagdefining = G__tagdefining;
            G__tagdefining = G__tagnum;
            int store_def_struct_member = G__def_struct_member;
            G__def_struct_member = 1;
            int store_static_alloc = G__static_alloc;
            G__static_alloc = 0;
            //
            //  Do the parse.
            //
            // Tell the parser to process the entire struct block.
            int brace_level = 0;
            // And call the parser.
            G__exec_statement(&brace_level);
            //
            //
            //
            G__static_alloc = store_static_alloc;
            G__prerun = store_prerun;
            G__access = store_access;
            G__tagnum = G__tagdefining;
            //
            // Padding for PA-RISC, Spark, etc
            // If struct size can not be divided by G__DOUBLEALLOC
            // the size is aligned.
            //
            if (
               (
                  (G__struct.memvar[G__tagnum]->allvar == 1) &&
                  !G__struct.memvar[G__tagnum]->next
               ) &&
               !G__struct.baseclass[G__tagnum]->basen
            ) {
               // -- this is still questionable, inherit0.c
               struct G__var_array* v = G__struct.memvar[G__tagnum];
               if (v->type[0] == 'c') {
                  if (isupper(v->type[0])) {
                     int num_of_elements = v->varlabel[0][1] /* num of elements */;
                     if (!num_of_elements) {
                        num_of_elements = 1;
                     }
                     G__struct.size[G__tagnum] = num_of_elements * G__LONGALLOC;
                  }
                  else {
                     G__value buf;
                     buf.type = v->type[0];
                     buf.tagnum = v->p_tagtable[0];
                     buf.typenum = v->p_typetable[0];
                     int num_of_elements = v->varlabel[0][1] /* num of elements */;
                     if (!num_of_elements) {
                        num_of_elements = 1;
                     }
                     G__struct.size[G__tagnum] = num_of_elements * G__sizeof(&buf);
                  }
               }
            }
            else {
               // --
               if (G__struct.size[G__tagnum] % G__DOUBLEALLOC) {
                  G__struct.size[G__tagnum] += G__DOUBLEALLOC - G__struct.size[G__tagnum] % G__DOUBLEALLOC;
               }
            }
            if (!G__struct.size[G__tagnum]) {
               G__struct.size[G__tagnum] = G__CHARALLOC;
            }
            G__def_struct_member = store_def_struct_member;
            G__tagdefining = store_tagdefining;
            G__p_local = store_local;
         }
      }
   }
   //
   // Now came to
   // [struct|union|enum]   tagname   { member }  item;
   //                                           ^
   //                     OR
   // [struct|union|enum]             { member }  item;
   //                                           ^
   //                     OR
   // [struct|union|enum]   tagname     item;
   //                                   ^
   // item declaration
   //
   G__var_type = 'u';
   // Need to think about this.
   if (type == 'e') {
      G__var_type = 'i';
   }
   if (ispointer) {
      G__var_type = toupper(G__var_type);
   }
   if (G__return > G__RETURN_NORMAL) {
      return;
   }
   // Note: The type must be class, enum, namespace, struct, or union.
   G__ASSERT((type == 'c') || (type == 'e') || (type == 'n') || (type == 's') || (type == 'u'));
   if (type == 'u') {
      // -- Union.
      fpos_t pos;
      int linenum;
      fgetpos(G__ifile.fp, &pos);
      linenum = G__ifile.line_number;
      G__FastAllocString basename(G__LONGLINE);
      c = G__fgetstream(basename, 0, ";");
      if (basename[0]) {
         fsetpos(G__ifile.fp, &pos);
         G__ifile.line_number = linenum;
         if (G__dispsource) {
            G__disp_mask = 1000; // FIXME: Crazy!
         }
         G__define_var(G__tagnum, -1);
         G__disp_mask = 0;
      }
      else if (!G__struct.name[G__tagnum][0]) {
         // -- anonymous union
         G__add_anonymousunion(G__tagnum, G__def_struct_member, store_def_tagnum);
      }
   }
   else if (type == 'n') {
      // -- Namespace.
      // no instance object for namespace, do nothing.
   }
   else {
      // -- Class, enumeration, or struct.
      if (G__cintv6) {
         G__bc_struct(G__tagnum);
      }
      fpos_t store_pos;
      fgetpos(G__ifile.fp, &store_pos);
      int store_linenum = G__ifile.line_number;
      G__disp_mask = 1000; // FIXME: Crazy!
      G__FastAllocString buf(G__ONELINE);
      int ch = G__fgetname(buf, 0, ";,(");
      int errflag = 0;
      if (isspace(ch) && (buf[0] != '*') && !strchr(buf, '[')) {
         G__FastAllocString tmp(G__ONELINE);
         ch = G__fgetname(tmp, 0, ";,(");
         if (isalnum(tmp[0])) {
            errflag = 1;
         }
      }
      G__disp_mask = 0;
      fsetpos(G__ifile.fp, &store_pos);
      G__ifile.line_number = store_linenum;
      if (errflag || (isclassdef && (ch == '('))) {
         G__genericerror("Error: ';' missing after class/struct/enum declaration");
         G__tagnum = store_tagnum;
         G__def_tagnum = store_def_tagnum;
         return;
      }
      G__def_tagnum = store_def_tagnum;
      G__define_var(G__tagnum, -1);
   }
   G__tagnum = store_tagnum;
   G__def_tagnum = store_def_tagnum;
}

//______________________________________________________________________________
int G__callfunc0(G__value* result, G__ifunc_table* iref, int ifn, G__param* libp, void* p, int funcmatch)
{
   // -- FIXME: Describe this function!
   int stat = 0;
   long store_struct_offset;
   int store_asm_exec;
   G__ifunc_table_internal* ifunc = G__get_ifunc_internal(iref);
   if (!ifunc->hash[ifn] || !ifunc->pentry[ifn]) {
      // -- The function is not defined or masked.
      *result = G__null;
      return stat;
   }
   store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = (long) p;
   store_asm_exec = G__asm_exec;
   G__asm_exec = 0;
   // this-pointer adjustment
   G__this_adjustment(ifunc, ifn);
#ifdef G__EXCEPTIONWRAPPER
   if (ifunc->pentry[ifn]->size == -1) {
      // -- Compiled function. should be stub.
      G__InterfaceMethod pfunc = (G__InterfaceMethod) ifunc->pentry[ifn]->tp2f;
      stat = G__ExceptionWrapper(pfunc, result, 0, libp, 1);
   }
   else if (ifunc->pentry[ifn]->bytecodestatus == G__BYTECODE_SUCCESS) {
      // -- Bytecode function.
      struct G__bytecodefunc* pbc = ifunc->pentry[ifn]->bytecode;
      stat = G__ExceptionWrapper(G__exec_bytecode, result, (char*) pbc, libp, 1);
   }
   else {
      // -- Interpreted function.
      stat = G__interpret_func(result, ifunc->funcname[ifn], libp, ifunc->hash[ifn], ifunc, G__EXACT, funcmatch);
   }
#else // G__EXCEPTIONWRAPPER
   if (ifunc->pentry[ifn]->size == -1) {
      // -- Compiled function. should be stub.
      G__InterfaceMethod pfunc = (G__InterfaceMethod) ifunc->pentry[ifn]->tp2f;
      stat = (*pfunc)(result, 0, libp, 1);
   }
   else if (ifunc->pentry[ifn]->bytecodestatus == G__BYTECODE_SUCCESS) {
      // -- Bytecode function.
      struct G__bytecodefunc* pbc = ifunc->pentry[ifn]->bytecode;
      stat = G__exec_bytecode(result, (char*) pbc, libp, 1);
   }
   else {
      // -- Interpreted function.
      stat = G__interpret_func(result, ifunc->funcname[ifn], libp, ifunc->hash[ifn], ifunc, G__EXACT, funcmatch);
   }
#endif // // G__EXCEPTIONWRAPPER
   G__store_struct_offset = store_struct_offset;
   G__asm_exec = store_asm_exec;
   return stat;
}

//______________________________________________________________________________
int G__calldtor(void* p, int tagnum, int isheap)
{
   // -- FIXME: Describe this function!
   int stat;
   G__value result;
   struct G__ifunc_table_internal* ifunc;
   int ifn = 0;
   long store_gvp;
   if (tagnum == -1) {
      return 0;
   }
   // FIXME: Destructor must be the first function in the table.
   ifunc = G__struct.memfunc[tagnum];
   store_gvp = G__getgvp();
   if (isheap) {
      // p is deleted either with free() or delete.
      G__setgvp(G__PVOID);
   }
   else {
      // In callfunc0, G__store_sturct_offset is also set to p.
      // Let G__operator_delete() return without doing anything.
      G__setgvp((long) p);
   }
   // Call destructor.
   struct G__param* para = new G__param();
   para->paran = 0;
   para->parameter[0][0] = 0;
   para->para[0] = G__null;
   stat = G__callfunc0(&result, G__get_ifunc_ref(ifunc), ifn, para, p, G__TRYDESTRUCTOR);
   delete para;
   G__setgvp(store_gvp);
   if (isheap && (ifunc->pentry[ifn]->size != -1)) {
      // -- Interpreted class.
      delete[] (char*) p;
      p = 0;
   }
   return stat;
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
//
//  Autoloading.
//

//______________________________________________________________________________
int G__set_class_autoloading(int newvalue)
{
   // -- FIXME: Describe this function!
   int oldvalue =  G__enable_autoloading;
   G__enable_autoloading = newvalue;
   return oldvalue;
}

//______________________________________________________________________________
void G__set_class_autoloading_callback(int (*p2f)(char*, char*))
{
   // -- FIXME: Describe this function!
   G__p_class_autoloading = p2f;
}

//______________________________________________________________________________
char* G__get_class_autoloading_table(char* classname)
{
   // Return the autoload entries for the class called classname.
   int tagnum = G__defined_tagname(classname, 4);
   if (tagnum < 0) return 0;
   return G__struct.libname[tagnum];
}

//______________________________________________________________________________
void G__set_class_autoloading_table(char* classname, char* libname)
{
   // Register the class named 'classname' as being available in library
   // 'libname' (I.e. the implementation or at least the dictionary for 
   // the class is in the given library.  The class is marked as 
   // 'autoload' to indicated that we known about it but have not yet
   // loaded its dictionary.
   // If libname==-1 then we 'undo' this behavior instead.

   int tagnum;
   int store_enable_autoloading = G__enable_autoloading;
   G__enable_autoloading = 0;
   int store_var_type = G__var_type;
   tagnum = G__search_tagname(classname, G__CLASS_AUTOLOAD);
   if (tagnum == -1) {
      // We ran out of space in G__struct.
      return;
   }
   G__var_type = store_var_type;
   if (libname == (void*)-1) {
      if (G__struct.type[tagnum] != 'a') {
         if (G__struct.libname[tagnum]) {
            free((void*)G__struct.libname[tagnum]);
         }
         G__struct.libname[tagnum] = 0;
      } else if (G__struct.name[tagnum][0]) {
         G__struct.namerange->Remove(G__struct.name[tagnum], tagnum);
         G__struct.name[tagnum][0] = '@';
      }
      G__enable_autoloading = store_enable_autoloading;
      return;
   }
   if (G__struct.libname[tagnum]) {
      free((void*)G__struct.libname[tagnum]);
   }
   G__struct.libname[tagnum] = (char*)malloc(strlen(libname) + 1);
   strcpy(G__struct.libname[tagnum], libname); // Okay we allocated enough space
   char* p = strchr(classname, '<');
   if (p) {
      // If the class is a template instantiation we need
      // to also register the template itself so that we
      // properly recognize it.
      char* buf = new char[strlen(classname)+1];
      strcpy(buf, classname); // Okay we allocated enough space
      buf[p-classname] = '\0';
      if (!G__defined_templateclass(buf)) {
         int store_def_tagnum = G__def_tagnum;
         int store_tagdefining = G__tagdefining;
         FILE* store_fp = G__ifile.fp;
         G__ifile.fp = 0;
         G__def_tagnum = G__struct.parent_tagnum[tagnum];
         G__tagdefining = G__struct.parent_tagnum[tagnum];
         char* templatename = buf;
         for (int j = (p - classname); j >= 0; --j) {
            if ((buf[j] == ':') && (buf[j-1] == ':')) {
               templatename = buf + j + 1;
               break;
            }
         }
         G__createtemplateclass(templatename, 0, 0);
         G__ifile.fp = store_fp;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
      }
      delete[] buf;
      buf = 0;
   }
   G__enable_autoloading = store_enable_autoloading;
}

//______________________________________________________________________________
//
//  Class, enum, namespace, struct, and union lookup and definition.
//

//______________________________________________________________________________
int G__defined_tagname(const char* tagname, int noerror)
{
   // -- Scan tagname table and return tagnum.
   //
   // Description:
   //
   // Scan tagname table and return tagnum. If not match, error message
   // is shown and -1 will be returned.
   // If non zero value is given to second argument 'noerror', error
   // message will be suppressed.
   //
   // noerror = 0   if not found try to instantiate template class
   //               if template is not found, display error
   //         = 1   if not found try to instantiate template class
   //               no error messages if template is not found
   //         = 2   if not found just return without trying template
   //         = 3   like 2, and no autoloading
   //         = 4   like 3, and don't look for typedef
   // noerror & 0x1000: do not look in enclosing scope, i.e. G__tagnum is
   //               defining a fully qualified identifier. With this bit
   //               set, tagname="C" and G__tagnum=A::B will not find to A::C.
   //
   // CAUTION:
   // If template class with constant argument is given to this function,
   // tagname argument may be modified like below.
   //   A<int,5*2> => A<int,10>
   // This may cause unexpected side-effect.
   //
   int i;
   int len;
   char* p;
   G__FastAllocString temp(G__LONGLINE);
   G__FastAllocString atom_tagname(G__LONGLINE);
   int env_tagnum;
   int store_var_type;
   switch (tagname[0]) {
      case '"':
      case '\'':
         return -1;
      case 'c':
         if (!strcmp(tagname, "const"))
            return -1;
   }
   bool enclosing = !(noerror & 0x1000);
   noerror &= ~0x1000;
   if (strchr(tagname, '>')) {
      // handles X<X<int>> as X<X<int> >
      while (0 != (p = (char*) strstr(tagname, ">>"))) {
         ++p;
         temp = p;
         *p = ' ';
         ++p;
         strcpy(p, temp); // Legacy, G__defined_tagname modify its input without knowning its length ..
      }
      // handles X<int > as X<int>
      p = (char*) tagname;
      while (0 != (p = (char*) strstr(p, " >"))) {
         if ('>' != *(p - 1)) {
            temp = p + 1;
            strcpy(p, temp); // Legacy, G__defined_tagname modify its input without knowning its length ..
         }
         ++p;
      }
      // handles X <int> as X<int>
      p = (char*) tagname;
      while (0 != (p = strstr(p, " <"))) {
         temp = p + 1;
         strcpy(p, temp); // Legacy, G__defined_tagname modify its input without knowning its length ..
         ++p;
      }
      // handles X<int>  as X<int>
      p = (char*) tagname;
      while (0 != (p = strstr(p, "> "))) {
         if (strncmp(p, "> >", 3) == 0) {
            p += 2;
         }
         else {
            temp = p + 2;
            strcpy(p + 1, temp); // Legacy, G__defined_tagname modify its input without knowning its length ..
            ++p;
         }
      }
      // handles X< int> as X<int>
      p = (char*) tagname;
      while (0 != (p = strstr(p, "< "))) {
         temp = p + 2;
         strcpy(p + 1, temp); // Legacy, G__defined_tagname modify its input without knowning its length ..
         ++p;
      }
      // handles X<int, int> as X<int,int>
      p = (char*) tagname;
      while (0 != (p = strstr(p, ", "))) {
         temp = p + 2;
         strcpy(p + 1, temp); // Legacy, G__defined_tagname modify its input without knowning its length ..
         ++p;
      }
   }
   // handle X<const const Y>
   p = (char*) strstr(tagname, "const const ");
   while (p) {
      char *p1 = (p += 6);
      char *p2 = p + 6;
      while (*p2) {
         *p1++ = *p2++;
      }
      *p1 = 0;
      p = strstr(p, "const const ");
   }
   if (isspace(tagname[0])) {
      temp = tagname + 1;
   }
   else {
      temp = tagname;
   }
   p = (char*)G__find_last_scope_operator(temp);
   if (p) {
      // A::B::C means we want A::B::C, not A::C, even if it exists.
      enclosing = false;
      atom_tagname = p + 2;
      *p = '\0';
      if (p == temp) {
         env_tagnum = -1;  // global scope
      }
#ifndef G__STD_NAMESPACE
      else if (!strcmp(temp, "std") && G__ignore_stdnamespace) {
         env_tagnum = -1;
         tagname += 5;
         if (!*tagname) {
            // -- "std::"
            return -1;
         }
      }
#endif // G__STD_NAMESPACE
      else {
         // first try a typedef, so we don't trigger autoloading here:
         long env_typenum = G__defined_typename_noerror(temp, 1);
         if (env_typenum != -1 && G__newtype.type[env_typenum] == 'u')
            env_tagnum = G__newtype.tagnum[env_typenum];
         else
            env_tagnum = G__defined_tagname(temp, noerror);
         if (env_tagnum == -1) {
            return -1;
         }
      }
   }
   else {
      atom_tagname = temp;
      env_tagnum = G__get_envtagnum();
   }
   // Search for old tagname.
   len = strlen(atom_tagname);
   int candidateTag = -1;
try_again:
   NameMap::Range nameRange = G__struct.namerange->Find(atom_tagname);
   if (nameRange) {
      for (i = nameRange.Last(); i >= nameRange.First(); --i) {
         if ((len == G__struct.hash[i]) && !strcmp(atom_tagname, G__struct.name[i])) {
            if ((!p && (enclosing || env_tagnum == -1) && (G__struct.parent_tagnum[i] == -1)) || (env_tagnum == G__struct.parent_tagnum[i])) {
               if (noerror < 3) {
                  if ( G__class_autoloading(&i) < 0 && noerror < 2 ) {
                     if (G__struct.type[i] != 'a') {
                        // The autoload failed but we have a real forward declaration of the type.
                     } else {
                        // The autoloading did not load anything, let's try instantiating.
                        break;
                     }
                  }
               }
               return i;
            }
            if (
                // --
                (candidateTag == -1) &&
                (
#ifdef G__VIRTUALBASE
                 (G__isanybase(G__struct.parent_tagnum[i], env_tagnum, G__STATICRESOLUTION) != -1) ||
#else // G__VIRTUALBASE
                 (G__isanybase(G__struct.parent_tagnum[i], env_tagnum) != -1) ||
#endif // G__VIRTUALBASE
                 (enclosing && G__isenclosingclass(G__struct.parent_tagnum[i], env_tagnum)) ||
                 (enclosing && G__isenclosingclassbase(G__struct.parent_tagnum[i], env_tagnum)) ||
                 (!p && (G__tmplt_def_tagnum == G__struct.parent_tagnum[i]))
#ifdef G__VIRTUALBASE
                 || -1 != G__isanybase(G__struct.parent_tagnum[i], G__tmplt_def_tagnum, G__STATICRESOLUTION)
#else // G__VIRTUALBASE
                 || -1 != G__isanybase(G__struct.parent_tagnum[i], G__tmplt_def_tagnum)
#endif // G__VIRTUALBASE
                 || (enclosing && G__isenclosingclass(G__struct.parent_tagnum[i], G__tmplt_def_tagnum))
                 || (enclosing && G__isenclosingclassbase(G__struct.parent_tagnum[i], G__tmplt_def_tagnum))
                 )
                ) {
               // --
               candidateTag = i;
            }
         }
      }
   } else {
      i = -1;
   }
   if (!len) {
      atom_tagname = "$";
      len = 1;
      goto try_again;
   }
   if (candidateTag != -1) {
      if (noerror < 2) {
         if ( G__class_autoloading(&candidateTag) >= 0 || G__struct.type[candidateTag] != 'a') {
            // Either there was nothing to autoload or the autoload succeeded.
            return candidateTag;
         } 
         // The autoloading did not load anything, let's try instantiating.
         // (by continuing on).
         
      } else {
         if (noerror < 3) {
            G__class_autoloading(&candidateTag);
         }
         return candidateTag;
      }
   }
   // If tagname not found, try instantiating class template.
   len = strlen(tagname);
   if ((tagname[len-1] == '>') && (noerror < 2) && ((len < 2) || (tagname[len-2] != '-'))) {
      if (G__loadingDLL) {
         G__fprinterr(G__serr, "Error: '%s' Incomplete template resolution in shared library", tagname);
         G__genericerror(0);
         G__fprinterr(G__serr, "Add following line in header for making dictionary\n");
         G__fprinterr(G__serr, "   #pragma link C++ class %s;\n", tagname);
         return -1;
      }
      // CAUTION: tagname may be modified in following function.
      i = G__instantiate_templateclass((char*) tagname, noerror);
      return i;
   }
   else if (noerror < 2) {
      G__Definedtemplateclass* deftmplt = G__defined_templateclass((char*) tagname);
      if (deftmplt && deftmplt->def_para && deftmplt->def_para->default_parameter) {
         i = G__instantiate_templateclass((char*) tagname, noerror);
         return i;
      }
   }

   if (noerror == 4)
      return -1;

   // Search for typename.
   store_var_type = G__var_type;
   i = G__defined_typename(tagname);
   G__var_type = store_var_type;
   if (i != -1) {
      i = G__newtype.tagnum[i];
      if (i != -1) {
         if (noerror < 3) {
            G__class_autoloading(&i);
         }
         return i;
      }
   }
   {
      int i2 = 0;
      int cx;
      while ((cx = tagname[i2++])) {
         if (G__isoperator(cx)) {
            return -1;
         }
      }
   }
   // Not found.
   if (!noerror) {
      G__fprinterr(G__serr, "Error: class,struct,union or type %s not defined ", tagname);
      G__genericerror(0);
   }
   return -1;
}

//______________________________________________________________________________
int G__search_tagname(const char* tagname, int type)
{
   // -- Search tagtable for name, if not found create an entry.  Returns the tagnum.
   //
   // If type > 0xff, create new G__struct entry if not found;
   // autoload if !isupper(type&0xff). type==0xff means ptr but type==0
   // (see v6_newlink.cxx:G__parse_parameter_link)
   //
   int i;
   int len;
   char* p;
   G__FastAllocString temp(G__LONGLINE);
   G__FastAllocString atom_tagname(G__LONGLINE);
   int noerror = 0;
   if (type == G__CLASS_AUTOLOAD) {
      /* no need to issue error message while uploading
         the autoloader information */
      noerror = 2;
   }
   int envtagnum = -1;
   int isstructdecl = (type > 0xff);
   type &= 0xff;
   bool isPointer = (type == 0xff) || isupper(type);
   if (type == 0xff) {
      type = 0;
   }
   type = tolower(type);
   // Search for old tagname
   // Only auto-load struct if not ref / ptr
   i = G__defined_tagname(tagname, isPointer ? 3 : 2);
   p = (char*)G__strrstr( tagname, "::");
   if (p && !strchr(p, '>')) {
      atom_tagname = tagname;
      p = (char*)G__strrstr(atom_tagname, "::");
      *p = 0;
      // first try a typedef, so we don't trigger autoloading here:
      long envtypenum = G__defined_typename_noerror(atom_tagname, 1);
      if (envtypenum != -1 && G__newtype.type[envtypenum] == 'u')
         envtagnum = G__newtype.tagnum[envtypenum];
      else
         envtagnum = G__defined_tagname(atom_tagname, 1);
   }
   else {
      envtagnum = G__get_envtagnum();
   }
   // If name not found, create a tagtable entry for it.
   if ((i == -1) || (isstructdecl && (envtagnum != G__struct.parent_tagnum[i]))) {
      ++G__struct.nactives;
      i = G__struct.alltag;
      if (i == G__MAXSTRUCT) {
         G__fprinterr(G__serr, "Limitation: Number of struct/union tag exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXSTRUCT in G__ci.h and recompile %s\n", G__MAXSTRUCT, G__ifile.name, G__ifile.line_number, G__nam);
         G__eof = 1;
         return -1;
      }
      temp = tagname;
      p = (char*)G__find_last_scope_operator(temp);
      if (p) {
         atom_tagname = p + 2;
         *p = '\0';
#ifndef G__STD_NAMESPACE
         if (strcmp(temp, "std") == 0 && G__ignore_stdnamespace) {
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
         int env_tagnum = -1;
         if (G__def_tagnum != -1) {
            env_tagnum = G__def_tagnum;
            if (G__tagdefining != G__def_tagnum) {
               env_tagnum = G__tagdefining;
            }
         }
         G__struct.parent_tagnum[i] = env_tagnum;
         atom_tagname = temp;
      }
      if (!strncmp("G__NONAME", atom_tagname, 9)) {
         atom_tagname[0] = 0;
         len = 0;
      }
      else {
         len = strlen(atom_tagname);
      }
      G__struct.userparam[i] = 0;
      G__struct.name[i] = (char*) malloc((size_t)(len + 1));
      // coverity[secure_coding] we allocated enough space
      strcpy(G__struct.name[i], atom_tagname); // Okay we allocated enough space
      G__struct.namerange->Insert(G__struct.name[i], i);
      G__struct.hash[i] = len;
      G__struct.size[i] = 0;
      G__struct.type[i] = type; // 's' struct ,'u' union ,'e' enum , 'c' class
      // Allocate and initialize member variable table.
      G__struct.memvar[i] = (struct G__var_array*) malloc(sizeof(struct G__var_array));
#ifndef G__OLDIMPLEMENTATION2038
      G__struct.memvar[i]->enclosing_scope = 0;
      G__struct.memvar[i]->inner_scope = 0;
#endif // G__OLDIMPLEMENTATION2038
      G__struct.memvar[i]->ifunc = 0;
      G__struct.memvar[i]->varlabel[0][0] = 0;
      G__struct.memvar[i]->paran[0] = 0;
      G__struct.memvar[i]->allvar = 0;
      G__struct.memvar[i]->next = 0;
      G__struct.memvar[i]->tagnum = i;
      {
         for (int j = 0; j < G__MEMDEPTH; ++j) {
            G__struct.memvar[i]->varnamebuf[j] = 0;
            G__struct.memvar[i]->p[j] = 0;
            G__struct.memvar[i]->is_init_aggregate_array[j] = 0;
         }
      }
      // Allocate and initialize member function table list.
      G__struct.memfunc[i] = (struct G__ifunc_table_internal* )malloc(sizeof(struct G__ifunc_table_internal));
      memset(G__struct.memfunc[i], 0, sizeof(struct G__ifunc_table_internal));
      G__struct.memfunc[i]->allifunc = 0;
      G__struct.memfunc[i]->next = 0;
      G__struct.memfunc[i]->page = 0;
#ifdef G__NEWINHERIT
      G__struct.memfunc[i]->tagnum = i;
#endif // G__NEWINHERIT
      {
         for (int j = 0; j < G__MAXIFUNC; ++j) {
            G__struct.memfunc[i]->funcname[j] = 0;
         }
      }
      // FIXME: Reserve the first entry for dtor.
      G__struct.memfunc[i]->hash[0] = 0;
      G__struct.memfunc[i]->funcname[0] = (char*) malloc(2);
      G__struct.memfunc[i]->funcname[0][0] = 0;
      G__struct.memfunc[i]->para_nu[0] = 0;
      G__struct.memfunc[i]->pentry[0] = &G__struct.memfunc[i]->entry[0];
      G__struct.memfunc[i]->pentry[0]->bytecode = 0;
      G__struct.memfunc[i]->friendtag[0] = 0;
      G__struct.memfunc[i]->pentry[0]->size = 0;
      G__struct.memfunc[i]->pentry[0]->filenum = 0;
      G__struct.memfunc[i]->pentry[0]->line_number = 0;
      G__struct.memfunc[i]->pentry[0]->bytecodestatus = G__BYTECODE_NOTYET;
      G__struct.memfunc[i]->ispurevirtual[0] = 0;
      G__struct.memfunc[i]->access[0] = G__PUBLIC;
      G__struct.memfunc[i]->ansi[0] = 1;
      G__struct.memfunc[i]->isconst[0] = 0;
      G__struct.memfunc[i]->reftype[0] = 0;
      G__struct.memfunc[i]->type[0] = 0;
      G__struct.memfunc[i]->p_tagtable[0] = -1;
      G__struct.memfunc[i]->p_typetable[0] = -1;
      G__struct.memfunc[i]->staticalloc[0] = 0;
      G__struct.memfunc[i]->busy[0] = 0;
      G__struct.memfunc[i]->isvirtual[0] = 0;
      G__struct.memfunc[i]->globalcomp[0] = G__NOLINK;
      G__struct.memfunc[i]->comment[0].filenum = -1;
      {
         struct G__ifunc_table_internal* store_ifunc = G__p_ifunc;
         G__p_ifunc = G__struct.memfunc[i];
         G__memfunc_next();
         G__p_ifunc = store_ifunc;
      }
      // Allocate and initialize class inheritance table.
      G__struct.baseclass[i] = (struct G__inheritance*) malloc(sizeof(struct G__inheritance));
      memset(G__struct.baseclass[i], 0, sizeof(struct G__inheritance));
      // Initialize iden information for virtual function.
      G__struct.virtual_offset[i] = -1; // -1 means no virtual function
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
      G__struct.comment[i].p.com = 0;
      G__struct.comment[i].filenum = -1;
      // G__setup_memfunc and G__setup_memvar pointers list initialisation
      G__struct.incsetup_memvar[i] = new std::list<G__incsetup>();
      G__struct.incsetup_memfunc[i] = new std::list<G__incsetup>();
      G__struct.rootflag[i] = 0;
      G__struct.rootspecial[i] = 0;
      G__struct.isctor[i] = 0;
#ifndef G__OLDIMPLEMENTATION1503
      G__struct.defaulttypenum[i] = -1;
#endif // G__OLDIMPLEMENTATION1503
      G__struct.vtable[i] = 0;
      G__struct.alltag++;

      if (G__struct.type[i] != 'a'
          && G__struct.type[i] != 0
          && G__UserSpecificUpdateClassInfo) {
         G__FastAllocString fullname(G__fulltagname(i,0));
         (*G__UserSpecificUpdateClassInfo)(fullname,i);
      }
   }
   else if (G__struct.type[i]==0 || (G__struct.type[i] == 'a')) {
      if (type != G__struct.type[i]) {
         G__struct.type[i] = type;
         if (G__struct.type[i] != 'a'
             && G__struct.type[i] != 0
             && G__UserSpecificUpdateClassInfo) {
            G__FastAllocString fullname(G__fulltagname(i,0));
            (*G__UserSpecificUpdateClassInfo)(fullname,i);
         }
      }
      ++G__struct.nactives;
   }

   // Return tagnum.
   return i;
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
