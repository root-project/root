/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file scrupto.c
 ************************************************************************
 * Description:
 *  Partial cleanup function
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Api.h"
#include "Dict.h"

using namespace Cint::Internal;

//______________________________________________________________________________
//______________________________________________________________________________
//
// How to use environment rewinding feature
//
//   G__dictposition pos;
//   G__store_dictposition(&pos);
//          .
//   // Do anything, 
//          .
//   if(!G__isfuncbusy(pos.nfile)) G__scratch_upto(&pos);
//

// Static functions.
static void G__free_preprocessfilekey(G__Preprocessfilekey* pkey);
static int G__free_ifunc_table_upto(::Reflex::Scope& ifunc, int remain);
static void G__close_inputfiles_upto(G__dictposition* pos);
static int G__free_string_upto(G__ConstStringList* conststringpos);
static int G__free_typedef_upto(int typenum);
static int G__free_struct_upto(int tagnum);
static int G__scratch_upto_work(G__dictposition* dictpos, int doall);

// Cint internal functions.
namespace Cint {
namespace Internal {
struct G__friendtag* G__new_friendtag(int tagnum);
struct G__friendtag* G__copy_friendtag(const G__friendtag* orig);
void G__free_friendtag(G__friendtag* friendtag);
int G__free_ifunc_table(::Reflex::Scope& ifunc);
int G__destroy_upto(::Reflex::Scope& scope, int global, int index);
} // namespace Internal
} // namespace Cint

// Functions in the C interface.
extern "C" void G__store_dictposition(G__dictposition* dictpos);
extern "C" int G__close_inputfiles();
extern "C" void G__scratch_all();
extern "C" int G__scratch_upto(G__dictposition* dictpos);
extern "C" void G__scratch_globals_upto(G__dictposition* dictpos);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static void G__free_preprocessfilekey(G__Preprocessfilekey* pkey)
{
   // -- FIXME: Describe this function!
   if (pkey->next) {
      G__free_preprocessfilekey(pkey->next);
      free(pkey->next);
      pkey->next = 0;
   }
   if (pkey->keystring) {
      free(pkey->keystring);
      pkey->keystring = 0;
   }
}

//______________________________________________________________________________
static int G__free_ifunc_table_upto(::Reflex::Scope& scope, int remain)
{
   // -- Remove all function members from a scope newer than or equal to a given index.
   for (int itemp = scope.FunctionMemberSize() - 1; itemp >= remain; itemp--) {
      ::Reflex::Member func = scope.FunctionMemberAt(itemp);
#ifdef G__MEMTEST
      fprintf(G__memhist, "func %s\n", func.Name().c_str());
#endif // G__MEMTEST
      if (G__get_funcproperties(func)->entry.bytecode) {
         G__free_bytecode(G__get_funcproperties(func)->entry.bytecode);
      }
#ifdef G__FRIEND
      G__free_friendtag(G__get_funcproperties(func)->entry.friendtag);
#endif // G__FRIEND
      func.DeclaringScope().RemoveFunctionMember(func);
   }
   return 1;
}

//______________________________________________________________________________
static void G__close_inputfiles_upto(G__dictposition* pos)
{
   // -- FIXME: Describe this function!
   // -- Cannot replace G__close_inputfiles()
#ifdef G__SHAREDLIB
   struct G__filetable permanentsl[G__MAX_SL];
   int nperm = 0;
#endif // G__SHAREDLIB
#ifdef G__DUMPFILE
   if (G__dumpfile) {
      G__dump_tracecoverage(G__dumpfile);
   }
#endif // // G__DUMPFILE
   std::string fullname;
   Reflex::Type cl;
   int nfile = pos->nfile;
   while (G__nfile > nfile) {
      --G__nfile;
      // reset autoload struct entries
      for (int itag = 0; itag < pos->tagnum; ++itag) {
         if (G__struct.filenum[itag] == G__nfile) {
            // -- Keep name, libname and parent_tagnum; reset everything else.

            fullname = G__fulltagname( itag, 0 );
            
            char* name = G__struct.name[itag];
            int hash = G__struct.hash[itag];
            char* libname = G__struct.libname[itag];
            int parent_tagnum = G__struct.parent_tagnum[itag];
            G__struct.name[itag] = 0; // autoload entry - must not delete it, just set it to 0 FIXME: Must add this back!
            G__struct.libname[itag] = 0; // same here FIXME: Must add this back!
            int alltag = G__struct.alltag;
            G__struct.alltag = itag + 1; // to only free itag
            G__free_struct_upto(itag);
            G__struct.alltag = alltag;
            --G__struct.nactives;

            G__struct.name[itag] = name;
            G__struct.libname[itag] = libname;
            G__struct.type[itag] = 'a';
            G__struct.hash[itag] = hash;
            G__struct.size[itag] = 0;
            G__struct.baseclass[itag] = new G__inheritance();
            G__struct.virtual_offset[itag] = (char*) -1;
            G__struct.globalcomp[itag] = 0;
            G__struct.iscpplink[itag] = G__default_link ? G__globalcomp : G__NOLINK;
            G__struct.isabstract[itag] = 0;
            G__struct.protectedaccess[itag] = 0;
            G__struct.line_number[itag] = -1;
            G__struct.filenum[itag] = -1;
            G__struct.parent_tagnum[itag] = parent_tagnum;
            G__struct.funcs[itag] = 0;
            G__struct.istypedefed[itag] = 0;
            G__struct.istrace[itag] = 0;
            G__struct.isbreak[itag] = 0;
            G__struct.friendtag[itag] = 0;
            // Clean up G__setup_memfunc and G__setup_memvar pointers list
            if (G__struct.incsetup_memvar[itag])
            {
               G__struct.incsetup_memvar[itag]->clear();
               delete G__struct.incsetup_memvar[itag];
               G__struct.incsetup_memvar[itag] = 0;
            }
            if (G__struct.incsetup_memfunc[itag])
            {
               G__struct.incsetup_memfunc[itag]->clear();
               delete G__struct.incsetup_memvar[itag];
               G__struct.incsetup_memvar[itag] = 0;
            }
            G__struct.rootflag[itag] = 0;
            G__struct.rootspecial[itag] = 0;
            G__struct.isctor[itag] = 0;
            G__struct.defaulttypenum[itag] = 0;
            G__struct.vtable[itag] = 0;
         }
      }
      if (G__srcfile[G__nfile].dictpos) {
         free((void*)G__srcfile[G__nfile].dictpos);
         G__srcfile[G__nfile].dictpos = 0;
      }
      if (G__srcfile[G__nfile].hasonlyfunc) {
         free((void*)G__srcfile[G__nfile].hasonlyfunc);
         G__srcfile[G__nfile].hasonlyfunc = 0;
      }
#ifdef G__SHAREDLIB
      if (G__srcfile[G__nfile].ispermanentsl) {
         permanentsl[nperm++] = G__srcfile[G__nfile];
         G__srcfile[G__nfile].initsl = 0;
         continue;
      }
#endif // G__SHAREDLIB
      if (G__srcfile[G__nfile].initsl) {
         delete(G__srcfile[G__nfile].initsl);
         G__srcfile[G__nfile].initsl = 0;
      }
      if (G__srcfile[G__nfile].fp) {
         fclose(G__srcfile[G__nfile].fp);
         if (G__srcfile[G__nfile].prepname) {
            for (int j = G__nfile - 1; j >= 0; --j) {
               if (G__srcfile[j].fp == G__srcfile[G__nfile].fp)
                  G__srcfile[j].fp = 0;
            }
         }
         G__srcfile[G__nfile].fp = 0;
      }
      if (G__srcfile[G__nfile].breakpoint) {
         free((void*)G__srcfile[G__nfile].breakpoint);
         G__srcfile[G__nfile].breakpoint = 0;
         G__srcfile[G__nfile].maxline = 0;
      }
      if (G__srcfile[G__nfile].prepname) {
         if ('(' != G__srcfile[G__nfile].prepname[0]) {
            remove(G__srcfile[G__nfile].prepname);
         }
         free((void*)G__srcfile[G__nfile].prepname);
         G__srcfile[G__nfile].prepname = 0;
      }
      if (G__srcfile[G__nfile].filename) {
         // --
#ifndef G__OLDIMPLEMENTATION1546
         unsigned int len = strlen(G__srcfile[G__nfile].filename);
         if (
            (len > strlen(G__NAMEDMACROEXT2)) &&
            !strcmp(G__srcfile[G__nfile].filename + len - strlen(G__NAMEDMACROEXT2), G__NAMEDMACROEXT2)
         ) {
            remove(G__srcfile[G__nfile].filename);
         }
#endif // G__OLDIMPLEMENTATION1546
         free(G__srcfile[G__nfile].filename);
         G__srcfile[G__nfile].filename = 0;
      }
      G__srcfile[G__nfile].hash = 0;
   }
   G__nfile = nfile;
#ifdef G__SHAREDLIB
   while (nperm) {
      --nperm;
      G__srcfile[G__nfile++] = permanentsl[nperm];
      if (permanentsl[nperm].initsl) {
         G__input_file store_ifile = G__ifile;
         G__ifile.filenum = G__nfile - 1;
         G__ifile.line_number = -1;
         G__ifile.str = 0;
         G__ifile.pos = 0;
         G__ifile.vindex = 0;
         G__ifile.fp = G__srcfile[G__nfile-1].fp;
         strcpy(G__ifile.name, G__srcfile[G__nfile-1].filename);
         for (
            std::list<G__DLLINIT>::const_iterator iInitsl = permanentsl[nperm].initsl->begin();
            iInitsl != permanentsl[nperm].initsl->end();
            ++iInitsl
         ) {
            (*(*iInitsl))();
         }
         G__ifile = store_ifile;
      }
   }
#endif // G__SHAREDLIB
   if (G__tempc[0]) {
      remove(G__tempc);
      G__tempc[0] = '\0';
   }
   // Closing modified stdio file handles.
   if (G__serr && (G__serr != G__stderr)) {
      fclose(G__serr);
      G__serr = G__stderr;
   }
   if (G__sout && (G__sout != G__stdout)) {
      fclose(G__sout);
      G__sout = G__stdout;
   }
   if (G__sin && (G__sin != G__stdin)) {
      fclose(G__sin);
      G__sin = G__stdin;
   }
}

//______________________________________________________________________________
static int G__free_string_upto(G__ConstStringList* conststringpos)
{
   // -- FIXME: Describe this function!
   struct G__ConstStringList* pconststring = G__plastconststring;
   while (pconststring && pconststring != conststringpos) {
      G__plastconststring = pconststring;
      pconststring = pconststring->prev;
      free(G__plastconststring->string);
      G__plastconststring->string = 0;
      free(G__plastconststring);
   }
   G__plastconststring = conststringpos;
   return 0;
}

//______________________________________________________________________________
static int G__free_typedef_upto(int typenum)
{
   // -- Destroy and free typedef definitions up to a given point in time.
   // -- Can replace G__free_typedef();
   for (int n = G__Dict::GetDict().GetNumTypes() - 1; n > typenum; --n) {
      ::Reflex::Type type = G__Dict::GetDict().GetTypedef(n);
      type.Unload();
   }
   // We should somehow compact the list but ... since
   // we have both class and typedef and the index are used as
   // reference ... we can't ... So for now leave it as is.
   // And since the array should be obsolete later, this might
   // be okay.

   // CHECKME: are the type in the vector really
   // invalidated by the unload?
   // G__Dict::Dict().Compact();
   return 0;
}

//______________________________________________________________________________
static int G__free_struct_upto(int tagnum)
{
   // -- Destroy and free class, enum, namespace, struct and union definitions up to a given point in time.
   //
   //  Destroy all static data members and namespace members.
   //
   for (int ialltag = G__struct.alltag - 1; ialltag >= tagnum; --ialltag) {
      // Free the defining shared library name.
      if (G__struct.libname[ialltag]) {
         free(G__struct.libname[ialltag]);
         G__struct.libname[ialltag] = 0;
      }
      if (G__struct.iscpplink[ialltag] == G__NOLINK) { // Is interpreted.
         // -- The struct is interpreted.
         ::Reflex::Scope varscope = G__Dict::GetDict().GetScope(ialltag);
         for (unsigned int i = 0; i < varscope.DataMemberSize(); ++i) {
            ::Reflex::Member var = varscope.DataMemberAt(i);
            if (
               // -- Static data member or namespace member, not a reference.
               (
                  (G__get_properties(var)->statictype == G__LOCALSTATIC) || // data member is static, or
                  (
                     varscope.IsNamespace() && // is a namespace member, and
                     (G__get_properties(var)->statictype != G__COMPILEDGLOBAL) // not precompiled
                  )
               ) && // and,
               (G__get_reftype(var.TypeOf()) == G__PARANORMAL) // not a reference
               && (!G__get_properties(var)->isFromUsing)
            ) {
               if ((G__get_type(var.TypeOf()) == 'u') && G__get_offset(var)) {
                  // -- Static class object member try destructor.
                  G__StrBuf com_sb(G__ONELINE);
                  char *com = com_sb;
                  sprintf(com, "~%s()", var.TypeOf().RawType().Name().c_str());
                  char* store_struct_offset = G__store_struct_offset;
                  ::Reflex::Scope store_tagnum = G__tagnum;
                  G__store_struct_offset = G__get_offset(var);
                  G__set_G__tagnum(var.TypeOf());
                  int j = G__get_varlabel(var, 1) /* number of elements */;
                  if (!j) {
                     j = 1;
                  }
                  --j;
                  for (; j >= 0; --j) {
                     int done = 0;
                     G__getfunction(com, &done, G__TRYDESTRUCTOR);
                     if (!done) {
                        break;
                     }
                     G__store_struct_offset += G__struct.size[G__get_tagnum(varscope)];
                  }
                  G__store_struct_offset = store_struct_offset;
                  G__tagnum = store_tagnum;
               }
               if (
                  // -- Class is not precompiled, and variable is not a const.
                  (G__struct.iscpplink[G__get_tagnum(var.TypeOf().RawType())] != G__CPPLINK) // Class is not precompiled, and
                  && !(G__get_isconst(var.TypeOf()) & G__CONSTVAR) // do not free s in const char s[] = "...";  //FIXME: Causes memory leak?
               ) {
                  // -- Free the variable value storage.
                  free(G__get_offset(var));
                  G__get_offset(var) = 0;
               }
            }
            // Delete data member from scope.
            var.DeclaringScope().RemoveDataMember(var);
         }
      }
      else { // Is compiled.
         // -- The struct is precompiled, we need to free enumerator values even for compiled code.
         ::Reflex::Scope varscope = G__Dict::GetDict().GetScope(ialltag);
         for (unsigned int i = 0; i < varscope.DataMemberSize(); ++i) {
            ::Reflex::Member var = varscope.DataMemberAt(i);
            // -- Check for an enumerator.
            if (
               var.TypeOf().RawType().IsEnum() && // Is of enum type, and
               (G__get_properties(var)->statictype == G__LOCALSTATIC) // is static, which means enumerator.
            ) {
               // -- Free the variable value storage.
               free(G__get_offset(var));
               G__get_offset(var) = 0;
            }
            var.DeclaringScope().RemoveDataMember(var);
         }
      }
   }
   //
   //  Free the struct definitions.
   //
   for (--G__struct.alltag; G__struct.alltag >= tagnum; --G__struct.alltag) {
      // -- Free a struct definition.
#ifdef G__MEMTEST
      fprintf(G__memhist, "struct %s\n", G__struct.name[G__struct.alltag]);
#endif
      //FIXME: What happened to this?: G__reset_ifunc_refs_for_tagnum(G__struct.alltag);
      if (G__struct.rootspecial[G__struct.alltag]) {
         free((void*) G__struct.rootspecial[G__struct.alltag]);
         G__struct.rootspecial[G__struct.alltag] = 0;
      }
#ifdef G__FRIEND
      G__free_friendtag(G__struct.friendtag[G__struct.alltag]);
#endif // G__FRIEND
      // freeing class inheritance table
      delete G__struct.baseclass[G__struct.alltag];
      G__struct.baseclass[G__struct.alltag] = 0;

      // Free member functions
      {
         ::Reflex::Scope varscope = G__Dict::GetDict().GetScope(G__struct.alltag);
         G__free_ifunc_table( varscope );
      }

      // freeing _memfunc_setup and memvar_setup function pointers
      if (G__struct.incsetup_memvar[G__struct.alltag]) {
         G__struct.incsetup_memvar[G__struct.alltag]->clear();
         delete G__struct.incsetup_memvar[G__struct.alltag];
         G__struct.incsetup_memvar[G__struct.alltag] = 0;
      }
      if (G__struct.incsetup_memfunc[G__struct.alltag]) {
         G__struct.incsetup_memfunc[G__struct.alltag]->clear();
         delete G__struct.incsetup_memfunc[G__struct.alltag];
         G__struct.incsetup_memfunc[G__struct.alltag] = 0;
      }

      // Free all known information.
      G__Dict::GetDict().GetScope(G__struct.alltag).Unload();
   }
   G__struct.alltag = tagnum;
   return 0;
}

//______________________________________________________________________________
static int G__scratch_upto_work(G__dictposition* dictpos, int doall)
{
   // -- Restore interpreter state to a given point in the past, or erase all of it.
   if (!dictpos && !doall) {
      return G__scratch_count;
   }
   G__LockCriticalSection();
   ::Reflex::Scope global_scope = ::Reflex::Scope::GlobalScope();
   if (doall) {
      G__lasterrorpos.line_number = 0;
      G__lasterrorpos.filenum = -1;
      // Reset ready flag for embedded use.
      G__cintready = 0;
      // Free the local variables.
      {
         ::Reflex::Scope local = G__p_local;
         while (local) {
            G__destroy_upto(local, G__LOCAL_VAR, -1);
            local = G__get_properties(local)->stackinfo.calling_scope;
         }
      }
      // Free the temporary object;
      if (G__p_tempbuf) {
         if (G__templevel > 0) {
            G__templevel = 0;
         }
         G__free_tempobject();
      }
   }
   // Free the global variables.
   if (doall) {
      G__destroy_upto(global_scope, G__GLOBAL_VAR, -1);
   }
   else {
      G__destroy_upto(dictpos->var, G__GLOBAL_VAR, dictpos->ig15);
   }
   // Free the exception handling buffer.
   if (doall) {
      G__free_exceptionbuffer();
   }
   // Garbage collection.
#ifdef G__SECURITY
   if (G__security & G__SECURE_GARBAGECOLLECTION) {
      G__garbagecollection();
   }
#endif // G__SECURITY
   // Free struct tagname and member table.
   if (doall) {
      // Note: Be careful to keep the global namespace, at index zero.
      // Note: Be careful to keep the bytecode arena, at index one.
      G__free_struct_upto((G__struct.alltag > 0) ? 2 : 0);
   }
   else {
      G__free_struct_upto(dictpos->tagnum);
   }
   // Free string constants.
   if (doall) {
      G__free_string_upto(&G__conststringlist);
   }
   // Free the typedef table.
   if (doall) {
      G__free_typedef_upto(0);
   }
   else {
      G__free_typedef_upto(dictpos->typenum);
   }
   // Free the interpreted function table.
   if (doall) {
      G__free_ifunc_table(global_scope);
   }
   else {
      G__free_ifunc_table_upto(dictpos->ifunc, dictpos->ifn);
   }
   // Erase the local variable pointer.
   if (doall) {
      G__p_local = ::Reflex::Scope();
   }
   // Free list of include paths.
   if (doall) {
      G__includepath* p = G__ipathentry.next;
      G__includepath* nxt = 0;
      G__ipathentry.next = 0;
      free(G__ipathentry.pathname);
      G__ipathentry.pathname = 0;
      for (; p; p = nxt) {
         nxt = p->next;
         p->next = 0;
         free(p->pathname);
         p->pathname = 0;
         free(p);
      }
   }
   else {
      G__includepath* p = dictpos->ipath;
      G__includepath* nxt = 0;
      if (p) {
         nxt = p->next;
         p->next = 0;
         free(p->pathname);
         p->pathname = 0;
         p = nxt;
         for (; p; p = nxt) {
            nxt = p->next;
            p->next = 0;
            free(p->pathname);
            p->pathname = 0;
            free(p);
         }
      }
   }
   // Free shared library list.
#ifdef G__SHAREDLIB
   if (doall) {
      G__free_shl_upto(0);
   }
   else {
      G__free_shl_upto(dictpos->allsl);
   }
#endif // G__SHAREDLIB
   // Free preprocessfilekey list.
   if (doall) {
      G__free_preprocessfilekey(&G__preprocessfilekey);
   }
   else {
      G__free_preprocessfilekey(dictpos->preprocessfilekey);
   }
   // Close macro file.
   if (doall) {
      if (G__mfp) {
         G__closemfp();
         G__mfp = 0;
      }
      // Close source files.
      G__close_inputfiles();
#ifdef G__DUMPFILE
#ifdef G__MEMTEST
      if (G__dumpfile && (G__dumpfile != G__memhist)) {
         fclose(G__dumpfile);
      }
#else // G__MEMTEST
      if (G__dumpfile) {
         fclose(G__dumpfile);
      }
#endif // G__MEMTEST
      G__dumpfile = 0;
#endif // G__DUMPFILE
      // Set function key.
      if (G__key) {
         system("key .cint_key -l execute");
      }
      while (G__dumpreadline[0]) {
         fclose(G__dumpreadline[0]);
         G__popdumpinput();
      }
   }
   // Free function macro list.
   if (doall) {
      G__freedeffuncmacro(&G__deffuncmacro);
   }
   else {
      G__freedeffuncmacro(dictpos->deffuncmacro);
   }
   // Free template class list, and template function list.
   if (doall) {
      // --
#ifdef G__TEMPLATECLASS
      G__freedeftemplateclass(&G__definedtemplateclass);
#endif // G__TEMPLATECLASS
#ifdef G__TEMPLATEFUNC
      G__freetemplatefunc(&G__definedtemplatefunc);
#endif // G__TEMPLATEFUNC
      // --
   }
   else {
      // --
#ifdef G__TEMPLATECLASS
      G__freedeftemplateclass(dictpos->definedtemplateclass);
#endif // G__TEMPLATECLASS
#ifdef G__TEMPLATEFUNC
      G__freetemplatefunc(dictpos->definedtemplatefunc);
#endif // G__TEMPLATEFUNC
      // --
   }
   if (doall) {
      // Delete user defined pragma statements.
      G__freepragma(G__paddpragma);
      G__paddpragma = 0;
      if (G__allincludepath) {
         free(G__allincludepath);
         G__allincludepath = 0;
      }
      G__DeleteConstStringList(G__SystemIncludeDir);
      G__SystemIncludeDir = 0;
      // This implementation is premature in a sense that macro can not be rewound to file position.
      G__init_replacesymbol();
      // Initialize cint body global variables.
      G__init = 0;
      G__init_globals();
      G__reset_setup_funcs();
      G__clear_errordictpos();
#ifdef G__MEMTEST
      G__memresult();
#endif // G__MEMTEST
      // --
   }
   else {
      // Close input files.
      G__close_inputfiles_upto(dictpos);
      G__tagdefining = ::Reflex::Scope();
   }
   G__UnlockCriticalSection();
   return G__scratch_count;
}

//______________________________________________________________________________
//
//  Cint internal functions.
//

//______________________________________________________________________________
struct G__friendtag* Cint::Internal::G__new_friendtag(int tagnum)
{
   G__friendtag* obj = new G__friendtag;
   obj->next = 0;
   obj->tagnum = tagnum;
   return obj;
}

//______________________________________________________________________________
struct G__friendtag* Cint::Internal::G__copy_friendtag(const G__friendtag* orig)
{
   if (!orig) {
      return 0;
   }
   G__friendtag* start = G__new_friendtag(orig->tagnum);
   G__friendtag* next = start;
   const G__friendtag* cursor = orig;
   while (cursor->next) {
      next->next = G__new_friendtag(orig->next->tagnum);
      next = next->next;
      cursor = cursor->next;
   }
   return start;
}

//______________________________________________________________________________
void Cint::Internal::G__free_friendtag(G__friendtag* friendtag)
{
  if (friendtag && (friendtag != friendtag->next)) {
    G__free_friendtag(friendtag->next);
    delete friendtag;
  }
}

//______________________________________________________________________________
int Cint::Internal::G__free_ifunc_table(::Reflex::Scope& scope)
{
   // -- Free all the member functions in a given scope.
   // FIXME: This should be a Reflex::Scope member function!
   std::vector<Reflex::Member> toberemoved;
   for (::Reflex::Member_Iterator i = scope.FunctionMember_Begin(); i != scope.FunctionMember_End(); ++i) {
      toberemoved.push_back(*i);
   }
   for (std::vector<Reflex::Member>::const_iterator j = toberemoved.begin(); j != toberemoved.end(); ++j) {
      scope.RemoveFunctionMember(*j);
   }
   // Do not free ifunc because it can be a global or static object.
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__destroy_upto(::Reflex::Scope& scope, int global, int index)
{
   // -- Destroy all data members in a given scope newer than or equal to a given index.
   ++G__scratch_count;
   if (!scope) {
      return 0;
   }
   int remain = index;
   if (remain == -1) {
      remain = 0;
   }
   //
   //  Destroy the variables.
   //
   for (int itemp = scope.DataMemberSize() - 1; itemp >= remain; --itemp) {
      ::Reflex::Member var = scope.DataMemberAt(itemp);
      //G__fprinterr(G__serr, "\nG__destroy_upto: scope: '%s' var: '%s' ary: %d  %s:%d\n", scope.Name().c_str(), var.Name().c_str(), G__get_varlabel(var, 1) /* number of elements */, __FILE__, __LINE__);
      if (
         // -- Check for not static variable, not precompiled, but allow static if deleting globals.
#ifdef G__ASM_WHOLEFUNC
         (
            (index < 0) && // We are handling function local variables, and
            (
               (global != G__BYTECODELOCAL_VAR) && // We are *not* processing bytecode local variables, and
               (
                  (G__get_properties(var)->statictype != G__LOCALSTATIC) || // Not a static variable, or
                  (global == G__GLOBAL_VAR) // We are destroying globals,
               ) && // and,
               (
                  (G__get_properties(var)->statictype != G__COMPILEDGLOBAL) // Not precompiled
               )
            )
         ) || // or
         (
            (index >= 0) && // We are *not* handling function local variables, and
#endif // G__ASM_WHOLEFUNC
            (
               (G__get_properties(var)->statictype != G__LOCALSTATIC) || // Not a static variable, or
               (global == G__GLOBAL_VAR) // We are destroying globals,
            ) && // and,
            (
               (G__get_properties(var)->statictype != G__COMPILEDGLOBAL) // Not precompiled
            ) && // and,
            !G__get_properties(var)->isFromUsing
         )
      ) {
         //G__fprinterr(G__serr, "\nG__destroy_upto: Destroying variable! scope: '%s' var: '%s' ary: %d  %s:%d\n", scope.Name().c_str(), var.Name().c_str(), G__get_varlabel(var, 1) /* number of elements */, __FILE__, __LINE__);
         // Default to variable is not of a precompiled class type.
         int cpplink = 0;
         //
         //  Call a destructor if needed.
         //
         if (
            (G__get_type(var.TypeOf()) == 'u') && // Variable is of class, enum, namespace, struct, or union type, and
            //FIXME: (var->reftype[idx] == G__PARANORMAL) && // Is not a reference, and
            !G__ansiheader && // We are not in function header scope, and
            !G__prerun // We are executing.
         ) {
            // -- Variable is of class type, we must call a destructor.
            //G__fprinterr(G__serr, "\nG__destroy_upto: Destroying variable! Calling destructor! scope: '%s' var: '%s' ary: %d  %s:%d\n", scope.Name().c_str(), var.Name().c_str(), G__get_varlabel(var, 1) /* number of elements */, __FILE__, __LINE__);
            std::string temp("~");
            char* store_struct_offset = G__store_struct_offset;
            G__store_struct_offset = G__get_offset(var);
            ::Reflex::Scope store_tagnum = G__tagnum;
            G__set_G__tagnum(var.TypeOf());
            int store_return = G__return;
            G__return = G__RETURN_NON;
            temp += G__tagnum.Name();
            temp += "()";
            if (G__dispsource) {
               //G__fprinterr(G__serr, "\n!!!Calling destructor 0x%lx.%s for %s ary%d:link%d", G__store_struct_offset, temp.c_str(), var.Name().c_str(), G__get_varlabel(var, 1) /* number of elements */, G__struct.iscpplink[G__get_tagnum(G__tagnum)]);
               G__fprinterr(G__serr, "\n!!!Calling destructor '%s::%s' array elements: %d linked: %d addr: %08lX\n", var.Name(Reflex::SCOPED).c_str(), temp.c_str(), G__get_varlabel(var, 1) /* number of elements */, G__struct.iscpplink[G__get_tagnum(G__tagnum)], G__store_struct_offset);
            }
            int store_prerun = G__prerun;
            G__prerun = 0;
            if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] == G__CPPLINK) {
               // -- The class is precompiled.
               cpplink = 1;
               if (G__get_properties(var)->statictype == G__AUTOARYDISCRETEOBJ) {
                  // -- The variable is an array with auto storage duration.
                  //
                  // Note: We allocated the memory for this variable using
                  //       G__malloc() and the element count is not stored
                  //       in the allocated block.
                  char* store_globalvarpointer = G__globalvarpointer;
                  int size = ((::Reflex::Type) G__tagnum).SizeOf();
                  int num_of_elements = G__get_varlabel(var, 1) /* number of elements */;
                  if (!num_of_elements) {
                     num_of_elements = 1;
                  }
                  for (int i = num_of_elements - 1; i >= 0; --i) {
                     G__store_struct_offset = G__get_offset(var) + (i * size);
                     G__globalvarpointer = G__store_struct_offset;
                     int done = 0;
                     G__getfunction((char*) temp.c_str(), &done, G__TRYDESTRUCTOR);
                     if (!done) {
                        break;
                     }
                  }
                  G__globalvarpointer = store_globalvarpointer;
                  free(G__get_offset(var)); // Note: Was allocated by G__malloc().
                  G__get_offset(var) = 0;
               }
               else {
                  // -- The variable is *not* an array with auto storage duration.
                  G__store_struct_offset = G__get_offset(var);
                  G__cpp_aryconstruct = G__get_varlabel(var, 1) /* number of elements */;
                  int done = 0;
                  G__getfunction((char*) temp.c_str(), &done, G__TRYDESTRUCTOR);
                  G__cpp_aryconstruct = 0;
               }
            }
            else {
               // -- The class is interpreted.
               ::Reflex::Type ty(G__tagnum);
               int size = ty.SizeOf();
               int num_of_elements = G__get_varlabel(var, 1) /* number of elements */;
               if (!num_of_elements) {
                  num_of_elements = 1;
               }
               for (int i = num_of_elements - 1; i >= 0; --i) {
                  G__store_struct_offset = G__get_offset(var) + (i * size);
                  if (G__dispsource) {
                     G__fprinterr(G__serr, "\n0x%lx.%s", G__store_struct_offset, temp.c_str());
                  }
                  int done = 0;
                  G__getfunction((char*) temp.c_str(), &done, G__TRYDESTRUCTOR);
                  if (!done) {
                     // -- This class does not have a destructor, quit.
                     break;
                  }
               }
            }
            G__prerun = store_prerun;
            G__store_struct_offset = store_struct_offset;
            G__tagnum = store_tagnum;
            G__return = store_return;
         }
#ifdef G__SECURITY
         //
         //  Decrement the reference count for a pointer.
         //
         if (
            (G__security & G__SECURE_GARBAGECOLLECTION) && // We are doing secure garbage collection, and
            !G__no_exec_compile && // We are *not* just generating bytecode, we are executing, and
            isupper(G__get_type(var.TypeOf())) && // the variable is a pointer, and
            G__get_offset(var) // has memory allocated.
         ) {
            // -- Decrement the reference counts for each member of the pointer array.
            int i = G__get_varlabel(var, 1) /* number of elements */;
            if (!i) {
               i = 1;
            }
            --i;
            for (; i >= 0; --i) {
               void** address = (void**) (G__get_offset(var) + (i * G__LONGALLOC));
               if (*address) {
                  G__del_refcount(*address, address);
               }
            }
         }
#endif // G__SECURITY
         //
         //  Free the value storage for the variable.
         //
#ifdef G__MEMTEST
         fprintf(G__memhist, "Free(%s)\n", var->varnamebuf[itemp]);
#endif // G__MEMTEST
         if (
            (cpplink == G__NOLINK) && // Variable is not of precompiled class type, and
            G__get_offset(var) // We have value storage allocated.
         ) {
            // -- Free the value storage.
            // FIXME: This should probably be a delete[] (char*).
            free((void*) G__get_offset(var));
            G__get_offset(var) = 0;
         }
      }
#ifdef G__DEBUG
      else if (G__memhist) {
         fprintf(G__memhist, "0x%lx (%s) not freed localstatic or compiledglobal FILE:%s LINE:%d\n", G__get_offset(var), var->varnamebuf[itemp], G__ifile.name, G__ifile.line_number);
      }
#endif // G__DEBUG
      var.DeclaringScope().RemoveDataMember(var);
   }
   return 0;
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
extern "C" void G__store_dictposition(G__dictposition* dictpos)
{
   // -- Mark a point in time in the interpreter state.
   G__LockCriticalSection();
   // Global variable position.
   dictpos->var = ::Reflex::Scope::GlobalScope();
   dictpos->ig15 = dictpos->var.DataMemberSize();
   dictpos->tagnum = G__struct.alltag;
   dictpos->conststringpos = G__plastconststring;
   dictpos->typenum = G__Dict::GetDict().GetNumTypes();
   // Global function position.
   dictpos->ifunc = ::Reflex::Scope::GlobalScope();
   dictpos->ifn = dictpos->ifunc.FunctionMemberSize();
   // Include path.
   dictpos->ipath = &G__ipathentry;
   while (dictpos->ipath->next) {
      dictpos->ipath = dictpos->ipath->next;
   }
   // Preprocessfilekey.
   dictpos->preprocessfilekey = &G__preprocessfilekey;
   while (dictpos->preprocessfilekey->next) {
      dictpos->preprocessfilekey = dictpos->preprocessfilekey->next;
   }
#ifdef G__SHAREDLIB
   dictpos->allsl = G__allsl;
#endif // G__SHAREDLIB
   dictpos->nfile = G__nfile;
   // Function macro.
   dictpos->deffuncmacro = &G__deffuncmacro;
   while (dictpos->deffuncmacro->next) {
      dictpos->deffuncmacro = dictpos->deffuncmacro->next;
   }
   // Template class.
   dictpos->definedtemplateclass = &G__definedtemplateclass;
   while (dictpos->definedtemplateclass->next) {
      dictpos->definedtemplateclass = dictpos->definedtemplateclass->next;
   }
   // Function template.
   dictpos->definedtemplatefunc = &G__definedtemplatefunc;
   while (dictpos->definedtemplatefunc->next) {
      dictpos->definedtemplatefunc = dictpos->definedtemplatefunc->next;
   }
   dictpos->nactives = G__struct.nactives;
   G__UnlockCriticalSection();
}

//______________________________________________________________________________
extern "C" int G__close_inputfiles()
{
   // -- FIXME: Describe this function!
   int iarg;
#ifdef G__DUMPFILE
   if (G__dumpfile) {
      G__dump_tracecoverage(G__dumpfile);
   }
#endif // G__DUMPFILE
   for (iarg = 0; iarg < G__nfile; ++iarg) {
      if (G__srcfile[iarg].hasonlyfunc) {
         free((void*) G__srcfile[iarg].hasonlyfunc);
         G__srcfile[iarg].hasonlyfunc = 0;
      }
      if (G__srcfile[iarg].fp) {
         fclose(G__srcfile[iarg].fp);
         if (G__srcfile[iarg].prepname) {
            for (int j = iarg + 1; j < G__nfile; ++j) {
               if (G__srcfile[j].fp == G__srcfile[iarg].fp) {
                  G__srcfile[j].fp = 0;
               }
            }
         }
         G__srcfile[iarg].fp = 0;
      }
      if (G__srcfile[iarg].breakpoint) {
         free((void*) G__srcfile[iarg].breakpoint);
         G__srcfile[iarg].breakpoint = 0;
         G__srcfile[iarg].maxline = 0;
      }
      if (G__srcfile[iarg].prepname) {
         if (G__srcfile[iarg].prepname[0] != '(') {
            remove(G__srcfile[iarg].prepname);
         }
         free((void*) G__srcfile[iarg].prepname);
         G__srcfile[iarg].prepname = 0;
      }
      if (G__srcfile[iarg].filename) {
         // --
#ifndef G__OLDIMPLEMENTATION1546
         int len = strlen(G__srcfile[iarg].filename);
         if (
            (len > (int) strlen(G__NAMEDMACROEXT2)) &&
            !strcmp(G__srcfile[iarg].filename + len - strlen(G__NAMEDMACROEXT2), G__NAMEDMACROEXT2)
         ) {
            remove(G__srcfile[iarg].filename);
         }
#endif // G__OLDIMPLEMENTATION1546
         free(G__srcfile[iarg].filename);
         G__srcfile[iarg].filename = 0;
      }
      G__srcfile[iarg].hash = 0;
   }
   G__nfile = 0;
   if (G__xfile[0]) {
      remove(G__xfile);
      G__xfile[0] = '\0';
   }
   if (G__tempc[0]) {
      remove(G__tempc);
      G__tempc[0] = '\0';
   }
   // Close modified standard file handles.
   if (G__serr && (G__serr != G__stderr)) {
      fclose(G__serr);
      G__serr = G__stderr;
   }
   if (G__sout && (G__sout != G__stdout)) {
      fclose(G__sout);
      G__sout = G__stdout;
   }
   if (G__sin && (G__sin != G__stdin)) {
      fclose(G__sin);
      G__sin = G__stdin;
   }
   return 0;
}

//______________________________________________________________________________
extern "C" void G__scratch_all()
{
   // -- Erase all of the interpreter state.
   G__scratch_upto_work(0, 1);
}

//______________________________________________________________________________
extern "C" int G__scratch_upto(G__dictposition* dictpos)
{
   // -- Restore interpreter state to a given point in the past.
   return G__scratch_upto_work(dictpos, 0);
}

//______________________________________________________________________________
extern "C" void G__scratch_globals_upto(G__dictposition* dictpos)
{
   // -- Destroy all global variables up to a given point in the past.
   G__LockCriticalSection();
#ifdef G__MEMTEST
   fprintf(G__memhist, "Freeing global variables\n");
#endif // G__MEMTEST
   G__destroy_upto(dictpos->var, G__GLOBAL_VAR, dictpos->ig15);
   G__UnlockCriticalSection();
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
