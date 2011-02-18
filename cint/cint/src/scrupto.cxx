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

extern "C" {

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
static void G__free_friendtag(G__friendtag* friendtag);
static void G__free_preprocessfilekey(G__Preprocessfilekey* pkey);
static int G__free_ifunc_table_upto_ifunc(G__ifunc_table_internal* ifunc, G__ifunc_table_internal* dictpos, int ifn);
static int G__free_ifunc_table_upto(G__ifunc_table_internal* ifunc, G__ifunc_table_internal* dictpos, int ifn);
static void G__close_inputfiles_upto(G__dictposition* pos);
static int G__free_string_upto(G__ConstStringList* conststringpos);
static int G__free_typedef_upto(int typenum);
static int G__free_struct_upto(int tagnum);
static int G__destroy_upto_vararray(G__var_array* var, int global, int ig15);

// External functions.
int G__free_ifunc_table(G__ifunc_table_internal* passed_ifunc);
int G__destroy_upto(G__var_array* var, int global, G__var_array* dictpos /*unused*/, int index);
int G__scratch_upto_work(G__dictposition* dictpos, int doall);

// Functions in the C interface.
void G__store_dictposition(G__dictposition* dictpos);
int G__close_inputfiles();
void G__scratch_globals_upto(G__dictposition* dictpos);

static G__var_array* G__last_global = &G__global;
static G__ifunc_table_internal* G__last_ifunc = &G__ifunc;
static G__Definedtemplateclass *G__last_definedtemplateclass = &G__definedtemplateclass;

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static void G__free_friendtag(G__friendtag* friendtag)
{
   // -- FIXME: Describe this function!
   if (friendtag && friendtag != friendtag->next) {
      G__free_friendtag(friendtag->next);
      free(friendtag);
   }
}

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
static int G__free_ifunc_table_upto_ifunc(G__ifunc_table_internal* ifunc, G__ifunc_table_internal* dictpos, int ifn)
{
   // -- FIXME: Describe this function!
   int i;
   // Freeing default parameter storage.
   if ((ifunc == dictpos) && (ifn == ifunc->allifunc)) {
      return 1;
   }
   for (i = ifunc->allifunc - 1; i >= 0; --i) {
      // --
#ifdef G__MEMTEST
      fprintf(G__memhist, "func %s\n", ifunc->funcname[i]);
#endif // G__MEMTEST
      //fprintf(stderr, "\nCalling destructor for param '%s'\n", ifunc->funcname[i]);
      ifunc->param[i].~G__params();
      if (ifunc->funcname[i]) {
         free((void*) ifunc->funcname[i]);
         ifunc->funcname[i] = 0;
      }
#ifdef G__ASM_WHOLEFUNC
      if (ifunc->pentry[i] && ifunc->pentry[i]->bytecode) {
         G__free_bytecode(ifunc->pentry[i]->bytecode);
         ifunc->pentry[i]->bytecode = 0;
      }
#endif // G__ASM_WHOLEFUNC
#ifdef G__FRIEND
      G__free_friendtag(ifunc->friendtag[i]);
#endif // G__FRIEND
      if ((ifunc == dictpos) && (ifn == i)) {
         ifunc->allifunc = ifn;
         return 1;
      }
   }
   // Do not free 'ifunc' because it can be a global/static object.
   ifunc->page = 0;
   return 0;
}

//______________________________________________________________________________
static int G__free_ifunc_table_upto(G__ifunc_table_internal* ifunc, G__ifunc_table_internal* dictpos, int ifn)
{
   // -- FIXME: Describe this function!
   G__last_ifunc = &G__ifunc;
   while (ifunc && ifunc != dictpos) {
      ifunc = ifunc->next;
   }
   if (ifunc != dictpos) {
      G__fprinterr(G__serr, "G__free_ifunc_table_upto: dictpos not found in ifunc list!\n");
      return 1;
   }
   G__ifunc_table_internal* next = ifunc->next;
   int ret = G__free_ifunc_table_upto_ifunc(ifunc, dictpos, ifn);
   ifunc->next = 0;
   while (next) {
      ifunc = next;
      next = ifunc->next;
      ret += G__free_ifunc_table_upto_ifunc(ifunc, dictpos, ifn);
      ifunc->next = 0;
      free((void*) ifunc);
   }
   return ret;
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
#endif // G__DUMPFILE
   int nfile = pos->nfile;
   ++G__srcfile_serial;
   while (G__nfile > nfile) {
      --G__nfile;
      // reset autoload struct entries
      for (int itag = 0; itag < pos->tagnum; ++itag) {
         if (G__struct.filenum[itag] == G__nfile) {
            // -- Keep name, libname, parent; reset everything else.
            char* name = G__struct.name[itag];
            int hash = G__struct.hash[itag];
            char* libname = G__struct.libname[itag];
            int parent_tagnum = G__struct.parent_tagnum[itag];
            G__struct.namerange->Remove(G__struct.name[itag], itag);
            G__struct.name[itag] = 0; // autoload entry - must not delete it, just set it to 0
            G__struct.libname[itag] = 0; // same here
            int alltag = G__struct.alltag;
            G__struct.alltag = itag + 1; // to only free itag
            G__free_struct_upto(itag);
            G__struct.alltag = alltag;
            --G__struct.nactives;
            G__struct.name[itag] = name;
            G__struct.namerange->Insert(G__struct.name[itag], itag);
            G__struct.libname[itag] = libname;
            G__struct.type[itag] = 'a';
            G__struct.hash[itag] = hash;
            G__struct.size[itag] = 0;
            G__struct.memvar[itag] = (struct G__var_array *)malloc(sizeof(struct G__var_array));
            memset(G__struct.memvar[itag], 0, sizeof(struct G__var_array));
            G__struct.memvar[itag]->tagnum = itag;
            G__struct.memfunc[itag] = (struct G__ifunc_table_internal *)malloc(sizeof(struct G__ifunc_table_internal));
            memset(G__struct.memfunc[itag], 0, sizeof(struct G__ifunc_table_internal));
            G__struct.memfunc[itag]->tagnum = itag;
            G__struct.memfunc[itag]->funcname[0] = (char*)malloc(2);
            G__struct.memfunc[itag]->funcname[0][0] = 0;
            G__struct.memfunc[itag]->pentry[0] = &G__struct.memfunc[itag]->entry[0];
            G__struct.memfunc[itag]->pentry[0]->bytecodestatus = G__BYTECODE_NOTYET;
            G__struct.memfunc[itag]->access[0] = G__PUBLIC;
            G__struct.memfunc[itag]->ansi[0] = 1;
            G__struct.memfunc[itag]->p_tagtable[0] = -1;
            G__struct.memfunc[itag]->p_typetable[0] = -1;
            G__struct.memfunc[itag]->comment[0].filenum = -1;
            {
               struct G__ifunc_table_internal *store_ifunc;
               store_ifunc = G__p_ifunc;
               G__p_ifunc = G__struct.memfunc[itag];
               G__memfunc_next();
               G__p_ifunc = store_ifunc;
            }
            G__struct.baseclass[itag] = (struct G__inheritance *)malloc(sizeof(struct G__inheritance));
            memset(G__struct.baseclass[itag], 0, sizeof(struct G__inheritance));
            G__struct.virtual_offset[itag] = -1;
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
            G__struct.comment[itag].p.com = NULL;
            G__struct.comment[itag].filenum = -1;
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
               delete G__struct.incsetup_memfunc[itag];
               G__struct.incsetup_memfunc[itag] = 0;
            }
            G__struct.rootflag[itag] = 0;
            G__struct.rootspecial[itag] = 0;
            G__struct.isctor[itag] = 0;
            G__struct.defaulttypenum[itag] = 0;
            G__struct.vtable[itag] = 0;
         }
      }
      if (G__srcfile[G__nfile].dictpos) {
         free((void*) G__srcfile[G__nfile].dictpos);
         G__srcfile[G__nfile].dictpos = 0;
      }
      if (G__srcfile[G__nfile].hasonlyfunc) {
         free((void*) G__srcfile[G__nfile].hasonlyfunc);
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
            int j;
            for (j = G__nfile - 1;j >= 0;j--) {
               if (G__srcfile[j].fp == G__srcfile[G__nfile].fp)
                  G__srcfile[j].fp = (FILE*)NULL;
            }
         }
         G__srcfile[G__nfile].fp = (FILE*)NULL;
      }
      if (G__srcfile[G__nfile].breakpoint) {
         free((void*) G__srcfile[G__nfile].breakpoint);
         G__srcfile[G__nfile].breakpoint = 0;
         G__srcfile[G__nfile].maxline = 0;
      }
      if (G__srcfile[G__nfile].prepname) {
         if ('(' != G__srcfile[G__nfile].prepname[0]) {
            remove(G__srcfile[G__nfile].prepname);
         }
         free((void*) G__srcfile[G__nfile].prepname);
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
         free((void*) G__srcfile[G__nfile].filename);
         G__srcfile[G__nfile].filename = 0;
      }
      G__srcfile[G__nfile].hash = 0;
   }
   G__nfile = nfile;
#ifdef G__SHAREDLIB
   int store_nperm = nperm;
   while (nperm) {
      --nperm;
      G__srcfile[G__nfile++] = permanentsl[nperm];
   }
   ++G__srcfile_serial;  // just in case the re-init of the dictionary triggers some autoloading.
   nperm = store_nperm;
   while (nperm) {
      --nperm;
      if (permanentsl[nperm].initsl) {
         G__input_file store_ifile = G__ifile;
         G__ifile.filenum = G__nfile - 1;
         G__ifile.line_number = -1;
         G__ifile.str = 0;
         G__ifile.pos = 0;
         G__ifile.vindex = 0;
         G__ifile.fp = G__srcfile[G__nfile - 1].fp;
         G__strlcpy(G__ifile.name, G__srcfile[G__nfile - 1].filename,G__MAXFILENAME);

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
   if (G__serr != G__stderr && G__serr) {
      fclose(G__serr);
      G__serr = G__stderr;
   }
   if (G__sout != G__stdout && G__sout) {
      fclose(G__sout);
      G__sout = G__stdout;
   }
   if (G__sin != G__stdin && G__sin) {
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
      free((void*) G__plastconststring->string);
      G__plastconststring->string = 0;
      free((void*) G__plastconststring);
   }
   G__plastconststring = conststringpos;
   return 0;
}

//______________________________________________________________________________
static int G__free_typedef_upto(int typenum)
{
   // -- Destroy and free typedef definitions up to a given point in time.
   // -- Can replace G__free_typedef();
   for (--G__newtype.alltype; G__newtype.alltype >= typenum; --G__newtype.alltype) {
      // -- Free a typedef definition.
      // Free the typedef name.
      G__newtype.namerange->Remove(G__newtype.name[G__newtype.alltype], G__newtype.alltype);
      free((void*) G__newtype.name[G__newtype.alltype]);
      G__newtype.name[G__newtype.alltype] = 0;
      //
      //  Free any array dimensions.
      //
      if (G__newtype.nindex[G__newtype.alltype]) {
         free((void*) G__newtype.index[G__newtype.alltype]);
         G__newtype.nindex[G__newtype.alltype] = 0;
      }
   }
   G__newtype.alltype = typenum;
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
         free((void*) G__struct.libname[ialltag]);
         G__struct.libname[ialltag] = 0;
      }
      if (G__struct.iscpplink[ialltag] == G__NOLINK) {
         // -- The struct is interpreted.
         struct G__var_array* var = G__struct.memvar[ialltag];
         for (; var; var = var->next) {
            for (int i = 0; i < var->allvar; ++i) {
               if (
                  // -- Static data member or namespace member, not a pointer or reference.
                  (
                     (var->statictype[i] == G__LOCALSTATIC) || // data member is static, or
                     (
                        (G__struct.type[ialltag] == 'n') && // is a namespace member, and
                        (var->statictype[i] != G__COMPILEDGLOBAL && var->statictype[i] != G__USING_VARIABLE && var->statictype[i] != G__USING_STATIC_VARIABLE) // not precompiled or from 'using'
                     )
                  ) && // and,
                  (var->reftype[i] == G__PARANORMAL) // not a pointer or reference
               ) {
                  // -- Call the destructor and free the variable value storage.
                  if ((var->type[i] == 'u') && var->p[i]) {
                     // -- Static class object member try destructor.
                     G__FastAllocString com(G__ONELINE);
                     com.Format("~%s()", G__struct.name[var->p_tagtable[i]]);
                     long store_struct_offset = G__store_struct_offset;
                     int store_tagnum = G__tagnum;
                     G__store_struct_offset = var->p[i];
                     G__tagnum = var->p_tagtable[i];
                     if (G__dispsource) {
                        G__fprinterr(G__serr, "!!!Destroy static member object 0x%lx %s::~%s()\n", var->p[i], G__struct.name[ialltag], G__struct.name[i]);
                     }
                     int j = var->varlabel[i][1] /* number of elements */;
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
                        G__store_struct_offset += G__struct.size[i];
                     }
                     G__store_struct_offset = store_struct_offset;
                     G__tagnum = store_tagnum;
                  }
                  if (
                     // -- Class is not precompiled, and variable is not a const.
                     (G__struct.iscpplink[var->p_tagtable[i]] != G__CPPLINK) && // Class is not precompiled, and
                     !(var->constvar[i] & G__CONSTVAR) // do not free s in const char s[] = "...";  //FIXME: Causes memory leak?
                  ) {
                     // -- Free the variable value storage.
                     free((void*) var->p[i]);
                     var->p[i] = 0;
                  }
               }
               // Free the variable name.
               if (var->varnamebuf[i]) {
                  free((void*) var->varnamebuf[i]);
                  var->varnamebuf[i] = 0;
               }
            }
         }
      }
      else {
         // -- The struct is precompiled, we need to free enumerator values even for compiled code.
         struct G__var_array* var = G__struct.memvar[ialltag];
         for (; var; var = var->next) {
            for (int i = 0; i < var->allvar; ++i) {
               // -- Check for an enumerator.
               if (
                  (var->p_tagtable[i] != -1) && // Not a fundamental type, and
                  (G__struct.type[var->p_tagtable[i]] == 'e') && // Is of enum type, and
                  (var->statictype[i] == G__LOCALSTATIC) // is static, which means enumerator.
               ) {
                  // -- Free the variable value storage.
                  free((void*) var->p[i]);
                  var->p[i] = 0;
               }
               // Free the variable name.
               if (var->varnamebuf[i]) {
                  free((void*) var->varnamebuf[i]);
                  var->varnamebuf[i] = 0;
               }
            }
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
#endif // G__MEMTEST
      G__reset_ifunc_refs_for_tagnum(G__struct.alltag);
      G__bc_delete_vtbl(G__struct.alltag);
      if (G__struct.rootspecial[G__struct.alltag]) {
         free((void*) G__struct.rootspecial[G__struct.alltag]);
         G__struct.rootspecial[G__struct.alltag] = 0;
      }
#ifdef G__FRIEND
      G__free_friendtag(G__struct.friendtag[G__struct.alltag]);
#endif // G__FRIEND
      // freeing class inheritance table
      G__struct.baseclass[G__struct.alltag]->herit.~G__herits();
      free((void*) G__struct.baseclass[G__struct.alltag]);
      G__struct.baseclass[G__struct.alltag] = 0;
      // freeing member function table
      G__free_ifunc_table(G__struct.memfunc[G__struct.alltag]);
      free((void*) G__struct.memfunc[G__struct.alltag]);
      G__struct.memfunc[G__struct.alltag] = 0;
      // freeing member variable table
      {
         G__var_array* p = G__struct.memvar[G__struct.alltag];
         G__var_array* nxt = 0;
         for (; p; p = nxt) {
            nxt = p->next;
            p->next = 0;
            free(p);
         }
         G__struct.memvar[G__struct.alltag] = 0;
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
      // freeing tagname
      G__struct.namerange->Remove(G__struct.name[G__struct.alltag], G__struct.alltag);
      free((void*) G__struct.name[G__struct.alltag]);
      G__struct.name[G__struct.alltag] = 0;
   }
   G__struct.alltag = tagnum;
   return 0;
}

//______________________________________________________________________________
static int G__destroy_upto_vararray(G__var_array* var, int global, int ig15)
{
   // -- Destroy a page of variables down to a given index.
   int remain = ig15;
   if (remain < 0) {
      remain = 0;
   }
   for (int idx = var->allvar - 1; idx >= remain; --idx) {
      if (
         // -- Check for not static variable, not precompiled, but allow static if deleting globals.
#ifdef G__ASM_WHOLEFUNC
         (
            (ig15 < 0) && // We are handling function local variables, and
            (
               (global != G__BYTECODELOCAL_VAR) && // We are *not* processing bytecode local variables, and
               (
                  (var->statictype[idx] != G__LOCALSTATIC) || // Not a static variable, or
                  (global == G__GLOBAL_VAR) // We are destroying globals,
               ) && // and,
               (
                  (var->statictype[idx] != G__COMPILEDGLOBAL) //|| // Not precompiled, or
                  //(
                  //  isupper(var->type[idx]) || // Is a pointer, or
                  //  (var->reftype[idx] == G__PARANORMAL) // Is a reference
                  //)
               )
            )
         ) || // or,
         ( (ig15 >= 0) && // We are *not* handling function local variables, and
#endif // G__ASM_WHOLEFUNC
         (
            (var->statictype[idx] != G__LOCALSTATIC) || // Not a static variable, or
            (global == G__GLOBAL_VAR) // We are destroying globals,
         ) && // and,
         (
            (var->statictype[idx] != G__COMPILEDGLOBAL) //|| // Not precompiled, or
            //(
            //  isupper(var->type[idx]) || // Is a pointer, or
            //  (var->reftype[idx] == G__PARANORMAL) // Is a reference
            //)
         )
#ifdef G__ASM_WHOLEFUNC
         )
#endif
      ) {
         // Default to variable is not of a precompiled class type.
         int cpplink = 0;
         //
         //  Call a destructor if needed.
         //
         if (
            (var->type[idx] == 'u') && // Variable is of class, enum, namespace, struct, or union type, and
            (var->reftype[idx] == G__PARANORMAL) && // Is not a reference, and
            !var->is_init_aggregate_array[idx] && // Is not an initialized aggregate array, and
            !G__ansiheader && // We are not in function header scope, and
            !G__prerun // We are executing.
         ) {
            // -- Variable is of class type, we must call a destructor.
            G__FastAllocString temp(G__BUFLEN);
            long store_struct_offset = G__store_struct_offset;
            G__store_struct_offset = var->p[idx];
            int store_tagnum = G__tagnum;
            G__tagnum = var->p_tagtable[idx];
            int store_return = G__return;
            G__return = G__RETURN_NON;
            temp.Format("~%s()", G__struct.name[G__tagnum]);
            if (G__dispsource) {
               G__fprinterr(
                    G__serr
                  , "\n!!!Calling destructor (%s) 0x%lx for %s "
                    "len: %d iscpplink: %d  %s:%d\n"
                  , temp()
                  , G__store_struct_offset
                  , var->varnamebuf[idx]
                  , var->varlabel[idx][1] /* number of elements */
                  , G__struct.iscpplink[G__tagnum]
                  , __FILE__
                  , __LINE__
               );
            }
            int store_prerun = G__prerun;
            G__prerun = 0;
            if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
               // -- The class is precompiled.
               cpplink = 1;
               if (var->statictype[idx] == G__AUTOARYDISCRETEOBJ) {
                  // -- The variable is an array with auto storage duration.
                  // FIXME: Do we really need this special case?
                  long store_globalvarpointer = G__globalvarpointer;
                  int size = G__struct.size[G__tagnum];
                  int num_of_elements = var->varlabel[idx][1] /* number of elements */;
                  if (!num_of_elements) {
                     num_of_elements = 1;
                  }
                  for (int i = num_of_elements - 1; i >= 0; --i) {
                     G__store_struct_offset = var->p[idx] + (i * size);
                     G__globalvarpointer = G__store_struct_offset;
                     int known = 0;
                     G__getfunction(temp, &known, G__TRYDESTRUCTOR);
                     if (!known) {
                        break;
                     }
                  }
                  G__globalvarpointer = store_globalvarpointer;
                  // FIXME: This should probably be a delete[] (char*).
                  free((void*) var->p[idx]);
                  var->p[idx] = 0;
               }
               else {
                  // -- The variable is *not* and array with auto storage duration.
                  G__store_struct_offset = var->p[idx];
                  int i = var->varlabel[idx][1] /* number of elements */;
                  if (i > 0)  {
                     G__cpp_aryconstruct = i;
                  }
                  int known = 0;
                  G__getfunction(temp, &known, G__TRYDESTRUCTOR);
                  G__cpp_aryconstruct = 0;
               }
            }
            else {
               // -- The class is interpreted.
               int size = G__struct.size[G__tagnum];
               int num_of_elements = var->varlabel[idx][1] /* number of elements */;
               if (!num_of_elements) {
                  num_of_elements = 1;
               }
               for (int i = num_of_elements - 1; i >= 0; --i) {
                  G__store_struct_offset = var->p[idx] + (i * size);
                  if (G__dispsource) {
                     G__fprinterr(G__serr, "\n0x%lx.%s", G__store_struct_offset, temp());
                  }
                  int known = 0;
                  G__getfunction(temp, &known, G__TRYDESTRUCTOR);
                  if (!known) {
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
            // -- Variable is a reference-counted pointer.
            (G__security & G__SECURE_GARBAGECOLLECTION) && // We are doing secure garbage collection, and
            !G__no_exec_compile && // We are *not* just generating bytecode, we are executing, and
            isupper(var->type[idx]) && // the variable is a pointer, and
            var->p[idx] // has memory allocated.
         ) {
            // -- Decrement the reference counts for each member of the pointer array.
            int i = var->varlabel[idx][1] /* number of elements */;
            if (!i) {
               i = 1;
            }
            --i;
            for (; i >= 0; --i) {
               long* address = (long*) (var->p[idx] + (i * G__LONGALLOC));
               if (*address) {
                  G__del_refcount((void*) *address, (void**) address);
               }
            }
         }
#endif // G__SECURITY
         //
         //  Free the value storage for the variable.
         //
#ifdef G__MEMTEST
         fprintf(G__memhist, "Free(%s)\n", var->varnamebuf[idx]);
#endif // G__MEMTEST
         if (
            (
               (cpplink == G__NOLINK) || // Variable is not of precompiled class type, or
               var->is_init_aggregate_array[idx] // Variable is an initialized aggregate array
            ) && // and,
            var->p[idx] && // We have value storage allocated, and
            (var->p[idx] != -1) // FIXME: This probably is not needed.
         ) {
            // -- Free the value storage.
            // FIXME: This should probably be a delete[] (char*).
            free((void*) var->p[idx]);
            var->p[idx] = 0;
         }
      }
#ifdef G__DEBUG
      else if (G__memhist) {
         fprintf(G__memhist, "0x%lx (%s) not freed localstatic or compiledglobal FILE:%s LINE:%d\n", var->p[idx], var->varnamebuf[idx], G__ifile.name, G__ifile.line_number);
      }
#endif // G__DEBUG
      // Clear the array dimensions.
      for (int j = 0; j < G__MAXVARDIM; ++j) {
         var->varlabel[idx][j] = 0;
      }
      // Free the variable name.
      if (var->varnamebuf[idx]) {
         // FIXME: This should probably be a delete[] (char*).
         free((void*) var->varnamebuf[idx]);
         var->varnamebuf[idx] = 0;
      }
   }
   var->allvar = remain;
   return 0;
}

//______________________________________________________________________________
//
//  External functions.
//

//______________________________________________________________________________
int G__free_ifunc_table(G__ifunc_table_internal* passed_ifunc)
{
   // -- Loop over the passed ifunc chain and free it.
   G__last_ifunc = &G__ifunc;
   G__ifunc_table_internal* ifunc = passed_ifunc;
   G__ifunc_table_internal* nxt_func = 0;
   for (; ifunc; ifunc = nxt_func) {
      nxt_func = ifunc->next;
      // Loop over all ifunc entries in memory page.
      for (int i = ifunc->allifunc - 1; i >= 0; --i) {
         // --
#ifdef G__MEMTEST
         fprintf(G__memhist, "func %s\n", ifunc->funcname[i]);
#endif // G__MEMTEST
         if (ifunc->funcname[i]) {
            ifunc->param[i].reset();
            free(ifunc->funcname[i]);
            ifunc->funcname[i] = 0;
#ifdef G__ASM_WHOLEFUNC
            if (
               ifunc->pentry[i] &&
               ifunc->pentry[i]->bytecode
            ) {
               G__free_bytecode(ifunc->pentry[i]->bytecode);
               ifunc->pentry[i]->bytecode = 0;
            }
#endif // G__ASM_WHOLEFUNC
#ifdef G__FRIEND
            G__free_friendtag(ifunc->friendtag[i]);
#endif // G__FRIEND
            // --
         }
      }
      ifunc->page = 0;
      ifunc->next = 0;
      // Do not free the passed_ifunc because it can be a global or static object.
      if (ifunc != passed_ifunc) {
         free(ifunc);
      }
   }
   return 0;
}

//______________________________________________________________________________
int G__destroy_upto(G__var_array* var, int global, G__var_array* /* dictpos*/, int index)
{
   // -- Destroy a variable chain up to a given index in the first page.
   ++G__scratch_count;
   if (!var) {
      return 0;
   }
   if (global == G__GLOBAL_VAR) {
      G__last_global = &G__global;
   }

   //
   //  Destroy any bytecode inner local variables first,
   //  they are chained off of the first variable.
   //
#ifndef G__OLDIMPLEMENTATION2038
   if (index == -1) {
      // -- We are destroying an entire local variable table.
      // Note: Enclosing_scope and inner_scope members are assigned
      // only in the local variable table for a bytecode function.
      var->enclosing_scope = 0;
      if (var->inner_scope) {
         for (int i = 0; var->inner_scope[i]; ++i) {
            G__destroy_upto(var->inner_scope[i], global, 0, -1);
            free((void*) var->inner_scope[i]);
            var->inner_scope[i] = 0;
         }
      }
   }
#endif // G__OLDIMPLEMENTATION2038
   //
   //  Reverse the variable list in place,
   //  so that destructors are run in the
   //  reverse order of creation.
   //
   G__var_array* tail = var;
   G__var_array* prev = 0;
   while (tail->next) {
      if (tail->allvar != G__MEMDEPTH) {
         fprintf(stderr, "!!!Fatal Error: Interpreter memory overwritten by illegal access.!!!\n");
         fprintf(stderr, "!!!Terminate session!!!\n");
      }
      // make tail->next point to prev instead
      G__var_array* next = tail->next;
      tail->next = prev;
      prev = tail;
      tail = next;
   }
   tail->next = prev;
   //
   //  Destroy the variables.
   //
   int ret = 0;
   G__var_array* next = 0;
   for (; tail; tail = next) {
      int upto_index = 0;
      if (!tail->next || (index < 0)) {
         upto_index = index;
      }
      ret += G__destroy_upto_vararray(tail, global, upto_index);
      next = tail->next;
      tail->next = 0;
      // Do not free the last page of variables.  FIXME: Is this a memory leak?
      if (next) {
         free(tail);
      }
   }
   return ret;
}

//______________________________________________________________________________
int G__scratch_upto_work(G__dictposition* dictpos, int doall)
{
   // -- Restore interpreter state to a given point in the past, or erase all of it.
   if (!dictpos && !doall) {
      return G__scratch_count;
   }
   G__LockCriticalSection();
   if (doall) {
      G__lasterrorpos.line_number = 0;
      G__lasterrorpos.filenum = -1;
      // Reset ready flag for embedded use.
      G__cintready = 0;
      // Free the local variables.
      {
         struct G__var_array* local = G__p_local;
         while (local) {
            G__destroy_upto(local, G__LOCAL_VAR, 0, -1);
            local = local->prev_local;
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
      G__destroy_upto(&G__global, G__GLOBAL_VAR, 0, -1);
   }
   else {
      G__destroy_upto(dictpos->var, G__GLOBAL_VAR, dictpos->var, dictpos->ig15);
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
      G__free_struct_upto(0);
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
      G__free_ifunc_table(&G__ifunc);
      G__ifunc.allifunc = 0;
   }
   else {
      G__free_ifunc_table_upto(&G__ifunc, G__get_ifunc_internal(dictpos->ifunc), dictpos->ifn);
   }
   // Erase the local variable pointer.
   if (doall) {
      G__p_local = 0;
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
         if (system("key .cint_key -l execute")) {
            G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
         }
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
   G__last_definedtemplateclass = &G__definedtemplateclass;
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
      G__tagdefining = -1;
   }
   G__UnlockCriticalSection();
   return G__scratch_count;
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
void G__store_dictposition(G__dictposition* dictpos)
{
   // -- Mark a point in time in the interpreter state.
   G__LockCriticalSection();
   // Global variable position.
   dictpos->var = G__last_global;
   while (dictpos->var->next) {
      dictpos->var = dictpos->var->next;
   }
   G__last_global = dictpos->var;

   dictpos->ig15 = dictpos->var->allvar;
   dictpos->tagnum = G__struct.alltag;
   dictpos->conststringpos = G__plastconststring;
   dictpos->typenum = G__newtype.alltype;
   // Global function position.
   G__ifunc_table_internal* lastifunc = G__last_ifunc;
   while (lastifunc->next) {
      lastifunc = lastifunc->next;
   }
   G__last_ifunc = lastifunc;
   dictpos->ifunc = G__get_ifunc_ref(lastifunc);
   dictpos->ifn = lastifunc->allifunc;
   // Include path.
   dictpos->ipath = &G__ipathentry;
   while (dictpos->ipath->next) dictpos->ipath = dictpos->ipath->next;
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
   dictpos->definedtemplateclass = G__last_definedtemplateclass;
   while (dictpos->definedtemplateclass->next) {
      dictpos->definedtemplateclass = dictpos->definedtemplateclass->next;
   }
   G__last_definedtemplateclass = dictpos->definedtemplateclass;
   // Function template.
   dictpos->definedtemplatefunc = &G__definedtemplatefunc;
   while (dictpos->definedtemplatefunc->next) {
      dictpos->definedtemplatefunc = dictpos->definedtemplatefunc->next;
   }
   dictpos->nactives = G__struct.nactives;
   G__UnlockCriticalSection();
}

//______________________________________________________________________________
int G__close_inputfiles()
{
   // -- FIXME: Describe this function!
   int iarg;
#ifdef G__DUMPFILE
   if (G__dumpfile) {
      G__dump_tracecoverage(G__dumpfile);
   }
#endif // G__DUMPFILE
   ++G__srcfile_serial;
   for (iarg = 0;iarg < G__nfile;iarg++) {
      if (G__srcfile[iarg].dictpos) {
         free((void*)G__srcfile[iarg].dictpos);
         G__srcfile[iarg].dictpos = (struct G__dictposition*)NULL;
      }
      if (G__srcfile[iarg].hasonlyfunc) {
         free((void*)G__srcfile[iarg].hasonlyfunc);
         G__srcfile[iarg].hasonlyfunc = (struct G__dictposition*)NULL;
      }
      if (G__srcfile[iarg].fp) {
         fclose(G__srcfile[iarg].fp);
         if (G__srcfile[iarg].prepname) {
            int j;
            for (j = iarg + 1;j < G__nfile;j++) {
               if (G__srcfile[j].fp == G__srcfile[iarg].fp)
                  G__srcfile[j].fp = (FILE*)NULL;
            }
         }
         G__srcfile[iarg].fp = (FILE*)NULL;
      }
      if (G__srcfile[iarg].breakpoint) {
         free((void*)G__srcfile[iarg].breakpoint);
         G__srcfile[iarg].breakpoint = (char*)NULL;
         G__srcfile[iarg].maxline = 0;
      }
      if (G__srcfile[iarg].prepname) {
         if ('(' != G__srcfile[iarg].prepname[0]) remove(G__srcfile[iarg].prepname);
         free((void*)G__srcfile[iarg].prepname);
         G__srcfile[iarg].prepname = (char*)NULL;
      }
      if (G__srcfile[iarg].filename) {
         // --
#ifndef G__OLDIMPLEMENTATION1546
         int len = strlen(G__srcfile[iarg].filename);
         if (len > (int)strlen(G__NAMEDMACROEXT2) &&
               strcmp(G__srcfile[iarg].filename + len - strlen(G__NAMEDMACROEXT2),
                      G__NAMEDMACROEXT2) == 0) {
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

   /*****************************************************************
    * Closing modified STDIOs.  May need to modify here.
    *  init.c, end.c, scrupto.c, pause.c
    *****************************************************************/
   if (G__serr != G__stderr && G__serr) {
      fclose(G__serr);
      G__serr = G__stderr;
   }
   if (G__sout != G__stdout && G__sout) {
      fclose(G__sout);
      G__sout = G__stdout;
   }
   if (G__sin != G__stdin && G__sin) {
      fclose(G__sin);
      G__sin = G__stdin;
   }
   return 0;
}

//______________________________________________________________________________
void G__scratch_all()
{
   // -- Erase all of the interpreter state.
   if (!G__struct.namerange)
      G__struct.namerange = new NameMap;
   if (!G__newtype.namerange)
      G__newtype.namerange = new NameMap;

   G__scratch_upto_work(0, 1);
}

} // extern "C"

//______________________________________________________________________________
int G__scratch_upto(G__dictposition* dictpos)
{
   // -- Restore interpreter state to a given point in the past.
   return G__scratch_upto_work(dictpos, 0);
}

//______________________________________________________________________________
void G__scratch_globals_upto(G__dictposition* dictpos)
{
   // -- Destroy all global variables up to a given point in the past.
   G__LockCriticalSection();
#ifdef G__MEMTEST
   fprintf(G__memhist, "Freeing global variables\n");
#endif // G__MEMTEST
   G__destroy_upto(dictpos->var, G__GLOBAL_VAR, dictpos->var /*unused*/, dictpos->ig15);
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
