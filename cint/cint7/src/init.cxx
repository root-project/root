/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file init.c
 ************************************************************************
 * Description:
 *  Entry functions
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "G__ci.h"
#include "Api.h"
#include "Dict.h"
#include "common.h"
#include "rflx_gendict.h"
#include "Reflex/Builder/ClassBuilder.h"
#include "Reflex/Builder/NamespaceBuilder.h"
#include "Reflex/Kernel.h"
#include "Reflex/Scope.h"
#include <list>

using namespace Cint::Internal;
using namespace std;

#if defined(G__ROOT) && !defined(G__NATIVELONGLONG)
void G__cpp_setuplongif();
#endif // G__ROOT && !G__NATIVELONGLONG

//______________________________________________________________________________
struct G__setup_func_struct
{
   char* libname;
   G__incsetup func;
   int filenum;
   int inited;
};

//______________________________________________________________________________
static G__setup_func_struct** G__setup_func_list;

//______________________________________________________________________________
static int G__max_libs;

//______________________________________________________________________________
static int G__nlibs;

//______________________________________________________________________________
typedef void G__parse_hook_t();

//______________________________________________________________________________
static G__parse_hook_t* G__afterparse_hook;

//______________________________________________________________________________
static G__parse_hook_t* G__beforeparse_hook;

//______________________________________________________________________________
static void G__platformMacro();

//______________________________________________________________________________
static char G__memsetup_init;

//______________________________________________________________________________
extern "C" void G__add_setup_func(const char* libname, G__incsetup func)
{
   int islot = -1;

   if (!G__memsetup_init) {
      for (int i = 0; i < G__MAXSTRUCT; i++) {
         G__struct.incsetup_memvar[i] = 0;
         G__struct.incsetup_memfunc[i] = 0;
      }
      G__memsetup_init = 1;
      Reflex::Instance initialize_reflex; // This causes Reflex to be initialized.
      G__init_globals();
      G__platformMacro();
   }

   if (!G__setup_func_list) {
      G__max_libs = 10;
      G__setup_func_list = (G__setup_func_struct**)calloc(G__max_libs, sizeof(G__setup_func_struct*));
   }
   if (G__nlibs >= G__max_libs) {
      G__max_libs += 10;
      G__setup_func_list = (G__setup_func_struct**)realloc(G__setup_func_list,
                           G__max_libs * sizeof(G__setup_func_struct*));
      for (int libi = G__nlibs; libi < G__max_libs; libi++)
         G__setup_func_list[libi] = 0;
   }

   /* if already in table: ignore (could also print warning) */
   for (int libi = 0; libi < G__nlibs; libi++)
      if (G__setup_func_list[libi] &&
            !strcmp(G__setup_func_list[libi]->libname, libname)) return;

   /* find empty slot */
   for (int libi = 0; libi < G__nlibs; libi++)
      if (!G__setup_func_list[libi]) {
         islot = libi;
         break;
      }
   if (islot == -1) islot = G__nlibs++;

   G__setup_func_list[islot] = (G__setup_func_struct*)malloc(sizeof(G__setup_func_struct));
   G__setup_func_list[islot]->libname = (char*) malloc(strlen(libname) + 1);
   G__setup_func_list[islot]->func    = func;
   G__setup_func_list[islot]->inited  = 0;
   strcpy(G__setup_func_list[islot]->libname, libname);
   G__setup_func_list[islot]->filenum = G__RegisterLibrary(func);
}

//______________________________________________________________________________
extern "C" void G__remove_setup_func(const char* libname)
{
   int i;

   for (i = 0; i < G__nlibs; i++)
      if (G__setup_func_list[i] &&
            !strcmp(G__setup_func_list[i]->libname, libname)) {
         G__UnregisterLibrary( G__setup_func_list[i]->func );
         free(G__setup_func_list[i]->libname);
         free(G__setup_func_list[i]);
         G__setup_func_list[i] = 0;
         if (i == G__nlibs - 1) G__nlibs--;
         return;
      }
}


//______________________________________________________________________________
extern "C" int G__call_setup_funcs()
{
   if ( ! G__tagtable::inited ) { 
      // Don't do anything until G__struct (at least) is initialized
      return 0;
   }
   if (Reflex::Instance::HasShutdown()) {
      return 0;
   }

   int init_counter = 0; // Number of initializers run.
   ::Reflex::Scope store_p_local = G__p_local; // changed by setupfuncs
   G__LockCriticalSection();
#ifdef G__SHAREDLIB
   if (!G__initpermanentsl) {
      G__initpermanentsl = new std::list<G__DLLINIT>;
   }
#endif //G__SHAREDLIB
   // Call G__RegisterLibrary() again, after it got called already
   // in G__init_setup_funcs(), because G__scratchall might have been
   // called in between.
   // Do a separate loop so we don't re-load because of A->B->A
   // dependencies introduced by autoloading during dictionary
   // initialization
   for (int i = 0; i < G__nlibs; ++i) {
      if (G__setup_func_list[i] && !G__setup_func_list[i]->inited) {
         G__setup_func_list[i]->filenum = G__RegisterLibrary(G__setup_func_list[i]->func);
      }
   }

   for (int i = 0; i < G__nlibs; ++i) {
      if (G__setup_func_list[i] && !G__setup_func_list[i]->inited) {
         // Run setup function for dictionary file.
#ifdef G__DEBUG
         fprintf(G__sout, "Initializing dictionary for '%s'.\n", G__setup_func_list[i]->libname);
#endif // G__DEBUG
         // Temporarily set G__ifile to the shared library.
         G__input_file store_ifile = G__ifile;
         int fileno = G__setup_func_list[i]->filenum;
         G__ifile.filenum = fileno;
         G__ifile.line_number = 1;
         G__ifile.str = 0;
         G__ifile.pos = 0;
         G__ifile.vindex = 0;
         
         if (fileno != -1) {
            G__ifile.fp = G__srcfile[fileno].fp;
            strcpy(G__ifile.name,G__srcfile[fileno].filename);
         }

         (G__setup_func_list[i]->func)();

         G__ifile = store_ifile;

         G__setup_func_list[i]->inited = 1; // FIXME: Should set before calling func to make sure we run func only once, but because of stupid way G__get_linked_tagnum calls back into root, G__setup_tagtable needs this to allow double calling.
         G__initpermanentsl->push_back(G__setup_func_list[i]->func);
         ++init_counter;
      }
   }
   G__UnlockCriticalSection();
   G__p_local = store_p_local;
   return init_counter;
}

//______________________________________________________________________________
extern "C" void G__reset_setup_funcs()
{
   int i;

   for (i = 0; i < G__nlibs; i++)
      if (G__setup_func_list[i])
         G__setup_func_list[i]->inited = 0;
}

//______________________________________________________________________________
struct G__libsetup_list
{
   void (*p2f)();
   struct G__libsetup_list* next;
};

//______________________________________________________________________________
static struct G__libsetup_list G__p2fsetup;

//______________________________________________________________________________
extern "C" void G__set_p2fsetup(void (*p2f)())
{
   G__libsetup_list* setuplist = &G__p2fsetup;
   while (setuplist->next) {
      setuplist = setuplist->next;
   }
   setuplist->p2f = p2f;
   setuplist->next = (G__libsetup_list*) malloc(sizeof(G__libsetup_list));
   setuplist->next->next = 0;
}

//______________________________________________________________________________
static void G__free_p2fsetuplist(G__libsetup_list* setuplist)
{
   if (setuplist->next) {
      G__free_p2fsetuplist(setuplist->next);
      free(setuplist->next);
      setuplist->next = 0;
   }
   setuplist->p2f = 0;
}

//______________________________________________________________________________
extern "C" void G__free_p2fsetup()
{
   G__free_p2fsetuplist(&G__p2fsetup);
}

//______________________________________________________________________________
static void G__do_p2fsetup()
{
   G__libsetup_list* setuplist = &G__p2fsetup;
   while (setuplist->next) {
      (*setuplist->p2f)();
      setuplist = setuplist->next;
   }
}

//______________________________________________________________________________
static void G__read_include_env(char* envname)
{
   char* env = getenv(envname);
   if (env) {
      char* pc;
      char* tmp = (char*)malloc(strlen(env) + 2);
      strcpy(tmp, env);
      char* p = tmp;
      while ((pc = strchr(p, ';')) || (pc = strchr(p, ','))) {
         *pc = 0;
         if (p[0]) {
            if (!strncmp(p, "-I", 2)) {
               G__add_ipath(p + 2);
            }
            else {
               G__add_ipath(p);
            }
         }
         p = pc + 1;
      }
      if (p[0]) {
         if (!strncmp(p, "-I", 2)) {
            G__add_ipath(p + 2);
         }
         else {
            G__add_ipath(p);
         }
      }
      free(tmp);
   }
}

//______________________________________________________________________________
extern "C" int G__getcintready()
{
   return G__cintready;
}

//______________________________________________________________________________
extern "C" void G__setothermain(int othermain)
{
   G__othermain = (short) othermain;
}

//______________________________________________________________________________
extern "C" int G__setglobalcomp(int globalcomp)
{
   int oldvalue = G__globalcomp;
   G__globalcomp = globalcomp;
   return oldvalue;
}

//______________________________________________________________________________
extern "C" G__parse_hook_t* G__set_afterparse_hook(G__parse_hook_t* hook)
{
   G__parse_hook_t* old = G__afterparse_hook;
   G__afterparse_hook = hook;
   return old;
}

//______________________________________________________________________________
extern "C" G__parse_hook_t* G__set_beforeparse_hook(G__parse_hook_t* hook)
{
   G__parse_hook_t* old = G__beforeparse_hook;
   G__beforeparse_hook = hook;
   return old;
}

//______________________________________________________________________________
void Cint::Internal::G__display_note()
{
   G__more(G__sout, "\n");
   G__more(G__sout, "Note1: Cint is not aimed to be a 100%% ANSI/ISO compliant C/C++ language\n");
   G__more(G__sout, " processor. It rather is a portable script language environment which\n");
   G__more(G__sout, " is close enough to the standard C++.\n");
   G__more(G__sout, "\n");
   G__more(G__sout, "Note2: Regularly check either of /tmp /usr/tmp /temp /windows/temp directory\n");
   G__more(G__sout, " and remove temp-files which are accidentally left by cint.\n");
   G__more(G__sout, "\n");
   G__more(G__sout, "Note3: Cint reads source file on-the-fly from the file system. Do not change\n");
   G__more(G__sout, " the active source during cint run. Use -C option or C1 command otherwise.\n");
   G__more(G__sout, "\n");
   G__more(G__sout, "Note4: In source code trace mode, cint sometimes displays extra-characters.\n");
   G__more(G__sout, " This is harmless. Please ignore.\n");
   G__more(G__sout, "\n");
}

//______________________________________________________________________________
int G__optind = 1;

//______________________________________________________________________________
char* G__optarg;

//______________________________________________________________________________
#define optind G__optind
#define optarg G__optarg
#define getopt G__getopt

//______________________________________________________________________________
extern "C" int G__getopt(int argc, char** argv, const char* optlist)
{
   if (optind >= argc) {
      return EOF;
   }
   if (argv[optind][0] != '-') {
      return EOF;
   }
   int optkey = argv[optind][1];
   for (const char* p = optlist; *p; ++p) {
      if ((*p) != optkey) {
         continue;
      }
      ++p;
      if ((*p) == ':') { // option with argument
         if (argv[optind][2]) { // -aARGUMENT
            optarg = argv[optind] + 2;
            optind += 1;
            return argv[optind-1][1];
         }
         // -a ARGUMENT
         optarg = argv[optind+1];
         optind += 2;
         return argv[optind-2][1];
      }
      // option without argument
      ++optind;
      optarg =  0;
      return argv[optind-1][1];
   }
   G__fprinterr(G__serr, "Error: Unknown option %s\n", argv[optind]);
   ++optind;
   return 0;
}

//______________________________________________________________________________
extern int G__quiet;

//______________________________________________________________________________
int Cint::Internal::G__init_globals()
{
   Reflex::Instance initReflex;
   // Explicit initialization of all necessary global variables.
   if (G__init) {
      return 1;
   }
   G__init = 1;
   G__p_ifunc = ::Reflex::Scope::GlobalScope();
   G__exec_memberfunc = 0;
   G__memberfunc_tagnum = ::Reflex::Scope::GlobalScope();
   G__memberfunc_struct_offset = 0;
   G__atpause = 0;
#ifdef G__ASM
   G__asm_name_p = 0;
   G__asm_loopcompile = 4;
   G__asm_loopcompile_mode = G__asm_loopcompile;
#ifdef G__ASM_WHOLEFUNC
   G__asm_wholefunction = G__ASM_FUNC_NOP;
#endif // G__ASM_WHOLEFUNC
   G__asm_wholefunc_default_cp = 0;
   G__abortbytecode();
   G__asm_dbg = 0;
   G__asm_cp = 0; /* compile program counter */
   G__asm_dt = G__MAXSTACK - 1; /* constant data address */
#ifdef G__ASM_IFUNC
   G__asm_inst = G__asm_inst_g;
   G__asm_instsize = 0; /* 0 means G__asm_inst is not resizable */
   G__asm_stack = G__asm_stack_g;
   G__asm_name = G__asm_name_g;
   G__asm_name_p = 0;
#endif // G__ASM_IFUNC
#endif // G__ASM
   G__debug = 0;          /* trace input file */
   G__breakdisp = 0;      /* display input file at break point */
   G__break = 0;          /* break flab */
   G__step = 0;           /* step execution flag */
   G__charstep = 0;       /* char step flag */
   G__break_exit_func = 0;  /* break at function exit */
   G__no_exec = 0;        /* no execution(ignore) flag */
   G__no_exec_compile = 0;
   G__var_type = 'p';    /* variable decralation type */
   G__var_typeB = 'p';
   G__prerun = 0;         /* pre-run flag */
   G__funcheader = 0;     /* function header mode */
   G__return = G__RETURN_NON; /* return flag of function */
   G__disp_mask = 0;      /* temporary read count */
   G__temp_read = 0;      /* temporary read count */
   G__switch = 0;           /* in a switch, parser should evaluate case expressions */
   G__switch_searching = 0; /* in a switch, parser should return after evaluating a case expression */
   G__eof_count = 0;      /* end of file error flag */
   G__ismain = G__NOMAIN; /* is there a main function */
   G__globalcomp = G__NOLINK;  /* make compiled func's global table */
   G__store_globalcomp = G__NOLINK;
   G__globalvarpointer = G__PVOID; /* make compiled func's global table */
   G__nfile = 0;
   G__key = 0; /* user function key on/off */
   G__xfile[0] = '\0';
   G__tempc[0] = '\0';
   G__doingconstruction = 0;
#ifdef G__DUMPFILE
   G__dumpfile = 0;
   G__dumpspace = 0;
#endif // G__DUMPFILE
   G__nobreak = 0;
#ifdef G__FRIEND
   G__friendtagnum = ::Reflex::Scope();
#endif // G__FRIEND
   G__def_struct_member = 0;
   G__tmplt_def_tagnum = ::Reflex::Scope();
   G__def_tagnum = ::Reflex::Scope::GlobalScope();
   G__tagdefining = ::Reflex::Scope::GlobalScope();
   G__tagnum = ::Reflex::Scope::GlobalScope();
   G__typenum = ::Reflex::Type();
   G__iscpp = 1;
   G__cpplock = 0;
   G__clock = 0;
   G__isconst = 0;
   G__constvar = 0;
   G__isexplicit = 0;
   G__unsigned = 0;
   G__ansiheader = 0;
   G__enumdef = 0;
   G__store_struct_offset = 0;
   G__decl = 0;
   G__longjump = 0;
   G__coredump = 0;
   G__definemacro = 0;
   G__noerr_defined = 0;
   G__static_alloc = 0;
   G__twice = 0;
   G__cpp = 0;
#ifndef G__OLDIMPLEMENTATOIN136
   G__include_cpp = 0;
#endif // G__OLDIMPLEMENTATOIN136
   G__ccom[0] = '\0';
   G__cppsrcpost[0] = '\0';
   G__csrcpost[0] = '\0';
   G__cpphdrpost[0] = '\0';
   G__chdrpost[0] = '\0';
   G__dispsource = 0;
   G__breaksignal = 0;
   G__bitfield = 0;
   G__atexit = 0;
#ifdef G__SIGNAL
   G__SIGINT = 0;
   G__SIGILL = 0;
   G__SIGFPE = 0;
   G__SIGABRT = 0;
   G__SIGSEGV = 0;
   G__SIGTERM = 0;
#ifdef SIGHUP
   G__SIGHUP = 0;
#endif
#ifdef SIGQUIT
   G__SIGQUIT = 0;
#endif
#ifdef SIGTSTP
   G__SIGTSTP = 0;
#endif
#ifdef SIGTTIN
   G__SIGTTIN = 0;
#endif
#ifdef SIGTTOU
   G__SIGTTOU = 0;
#endif
#ifdef SIGALRM
   G__SIGALRM = 0;
#endif
#ifdef SIGUSR1
   G__SIGUSR1 = 0;
#endif
#ifdef SIGUSR2
   G__SIGUSR2 = 0;
#endif
#endif
#ifdef G__SHAREDLIB
   G__allsl = 0;
#endif // G__SHAREDLIB
   memset(&G__tempbuf, 0, sizeof(G__tempobject_list));
   G__p_tempbuf = &G__tempbuf;
   G__templevel = 0;
   G__reftype = G__PARANORMAL;
   G__access = G__PUBLIC;
   G__stepover = 0;
   G__newarray.next = 0;
   G__ifile.fp = 0;
   G__ifile.name[0] = '\0';
   G__ifile.line_number = 0;
   G__ifile.filenum = -1;
   G__steptrace = 0;
   G__debugtrace = 0;
   G__ipathentry.pathname = 0;
   G__ipathentry.next = 0;
   G__mfp = 0;
   G__deffuncmacro.hash = 0;
   G__deffuncmacro.callfuncmacro.call_fp = 0;
   G__deffuncmacro.callfuncmacro.call_filenum = -1;
   G__deffuncmacro.callfuncmacro.next = 0;
   G__deffuncmacro.next = 0;
   G__cintsysdir[0] = '*';
   G__cintsysdir[1] = '\0';
   G__p_local = 0;
   G__cpp_aryconstruct = 0;
   G__cppconstruct = 0;
   G__breakfile[0] = '\0';
   G__assertion[0] = '\0';
   G__letint(&G__null, '\0', 0);
   G__value_typenum(G__null) = ::Reflex::Type();
   G__null.ref = 0;
   G__letint(&G__one, 'i', 1);
   G__one.ref = 0;
   {
      //fprintf(stderr, "G__init_globals: calling Reflex::TypedefTypeBuilder for 'switchStart$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("switchStart$", ::Reflex::Type::ByName("int"), (Reflex::REPRESTYPE)'a'); // 'a' type
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__init_globals: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }
   G__letint(&G__start, 'a', G__SWITCH_START);
   G__start.ref = 0;
   {
      //fprintf(stderr, "G__init_globals: calling Reflex::TypedefTypeBuilder for 'switchDefault$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("switchDefault$", ::Reflex::Type::ByName("int"), (Reflex::REPRESTYPE)'z'); // 'z' type
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__init_globals: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }

   //new TypeBase("rootSpecial$", sizeof(void*) * 2, FUNDAMENTAL, typeid(::Reflex::UnknownType), Type(), (Reflex::REPRESTYPE)'Z'); // 'Z' type
   {
      //fprintf(stderr, "G__init_globals: calling Reflex::TypedefTypeBuilder for 'rootSpecial$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("rootSpecial$", ::Reflex::ArrayBuilder( ::Reflex::PointerBuilder(Reflex::Type::ByName("void")), 2 ), (Reflex::REPRESTYPE)'Z'); // 'Z' type
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__init_globals: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }      
   }

   // new TypeBase("blockBreakContinueGoto$", sizeof(int), FUNDAMENTAL, typeid(::Reflex::UnknownType), Type(), (Reflex::REPRESTYPE)'\001'); // was also 'Z' type (confusing)
   {
      //fprintf(stderr, "G__init_globals: calling Reflex::TypedefTypeBuilder for 'blockBreakContinueGoto$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("blockBreakContinueGoto$", ::Reflex::Type::ByName("int"), (Reflex::REPRESTYPE)'\001'); // was also 'Z' type (confusing)
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__init_globals: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }

   G__letint(&G__default, 'z', G__SWITCH_DEFAULT);
   G__default.ref = 0;
   G__letint(&G__block_break, '\001', G__BLOCK_BREAK);
   G__block_break.ref = 0;
   G__letint(&G__block_continue, '\001', G__BLOCK_CONTINUE);
   G__block_continue.ref = 0;
   G__letint(&G__block_goto, '\001', G__BLOCK_BREAK);
   G__block_goto.ref = 1;
   G__gotolabel[0] = '\0';
   {
      //fprintf(stderr, "G__init_globals: calling Reflex::TypedefTypeBuilder for 'defaultFunccall$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("defaultFunccall$", ::Reflex::Type::ByName("int"), (Reflex::REPRESTYPE)'\011'); // '\011' type
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__init_globals: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }
   G__exceptionbuffer = G__null;
   G__virtual = 0;
#ifdef G__AUTOCOMPILE
   G__fpautocc = 0;
   G__compilemode = 1;
#endif // G__AUTOCOMPILE
   G__typedefnindex = 0;
   G__oprovld = 0;
   G__p2arylabel[0] = 0;
   G__interactive = 0;
   G__interactivereturnvalue = G__null;
#ifdef G__TEMPLATECLASS
   G__definedtemplateclass.next = 0;
   G__definedtemplateclass.def_para = 0;
   G__definedtemplateclass.def_fp = 0;
#ifdef G__TEMPLATEMEMFUNC
   G__definedtemplateclass.memfunctmplt.next = 0;
#endif // G__TEMPLATEMEMFUNC
   G__definedtemplateclass.parent_tagnum = -1;
   G__definedtemplateclass.isforwarddecl = 0;
   G__definedtemplateclass.instantiatedtagnum = 0;
   G__definedtemplateclass.specialization = 0;
   G__definedtemplateclass.spec_arg = 0;
#ifdef G__TEMPLATEFUNC
   G__definedtemplatefunc.next = 0;
   G__definedtemplatefunc.def_para = 0;
   G__definedtemplatefunc.name = 0;
   for (int i = 0; i < G__MAXFUNCPARA; ++i) {
      G__definedtemplatefunc.func_para.ntarg[i] = 0;
      G__definedtemplatefunc.func_para.nt[i] = 0;
   }
#endif // G__TEMPLATEFUNC
#endif // G__TEMPLATECLASS
   G__isfuncreturnp2f = 0;
   G__typepdecl = 0;
   G__macroORtemplateINfile = 0;
   G__macro_defining = 0;
   G__nonansi_func = 0;
   G__parseextern = 0;
   G__istrace = 0;
   G__pbreakcontinue = 0;
   G__fons_comment = 0;
   G__setcomment = 0;
#ifdef G__SECURITY
   G__security = G__SECURE_LEVEL0;
   G__castcheckoff = 0;
   G__security_error = G__NOERROR;
   G__max_stack_depth = G__DEFAULT_MAX_STACK_DEPTH;
#endif // G__SECURITY
   G__preprocessfilekey.keystring = 0;
   G__preprocessfilekey.next = 0;
   G__precomp_private = 0;
   // The first entry in the const string is a blank string which is never used.
   static char clnull[1] = ""; 
   G__conststringlist.string = clnull;
   G__conststringlist.hash = 0;
   G__conststringlist.prev = 0;
   G__plastconststring = &G__conststringlist;
#ifdef G__ROOT
   if (!G__GetSpecialObject) {
      G__GetSpecialObject = (G__value(*)(char*, void**, void**)) G__getreserved;
   }
#else // G__ROOT
   G__GetSpecialObject = (G__value(*)(char*, void**, void**)) G__getreserved;
#endif // G__ROOT
   G__is_operator_newdelete = G__DUMMYARG_NEWDELETE | G__NOT_USING_2ARG_NEW;
   G__fpundeftype = 0;
   G__ispermanentsl = 0;
   G__boolflag = 0;
   return 0;
}

//______________________________________________________________________________
static void G__defineMacro(const char* name, long value, const char* cintname = 0, bool cap = true, bool compiler = false)
{
   //  Add a macro called name with its value to the known macros.
   //  Also add a CINT version, which transforms
   //  [_]*xyz[_]* to G__XYZ
   //  If called with cap=false, capitalization does not happen,
   //  i.e. [_]*xyz[_]* is transformed to G__xyz.
   //  If cintname is given, it will be used instead of the
   //  converted name G__XYZ.
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;

   if (G__globalcomp != G__NOLINK && !compiler)
      // not a compiler, and !=G__NOLINK - already dealt with in first pass
      return;

   sprintf(temp + 2, "!%s=%ld", name, value);

   if (!compiler || G__globalcomp != G__NOLINK)
      // add system version, which starts with a '!'
      G__add_macro(temp + 2);

   if (G__globalcomp != G__NOLINK)
      // already dealt with in first pass
      return;

   char* start = temp;
   if (cintname) {
      start += 3;
      sprintf(start, "%s=%ld", cintname, value);
   }
   else {
      // generate CINT name:
      // leading '_' are skipped:
      char* end = start + 3 + strlen(name) - 1;
      while (start[3] == '_') ++start;
      // it starts with a "G__":
      memcpy(start, "G__", 3);
      // trailing '_' are removed
      while (*end == '_') --end;

      sprintf(end + 1, "=%ld", value);
      while (cap && end != start) {
         // capitalize the CINT macro name
         *end = toupper(*end);
         --end;
      }
   }
   // add the CINT version of the macro
   G__add_macro(start);
}

//______________________________________________________________________________
// Define macro with value, both system macro and CINT macro.
#define G__DEFINE_MACRO(macro) \
   G__defineMacro(#macro, (long)macro)

//______________________________________________________________________________
// Define compiler macro with value, both system macro and CINT macro.
#define G__DEFINE_MACRO_C(macro) \
   G__defineMacro(#macro, (long)macro, 0, true, true)

//______________________________________________________________________________
// Define macro with value, both system macro and CINT macro, specifying the CINT macro name.
#define G__DEFINE_MACRO_N(macro, name) \
   G__defineMacro(#macro, (long)macro, name)

//______________________________________________________________________________
// Define compiler macro with value, both system macro and CINT macro, specifying the CINT macro name.
#define G__DEFINE_MACRO_N_C(macro, name) \
   G__defineMacro(#macro, (long)macro, name, true, true)

//______________________________________________________________________________
// Define macro with value, both system macro and CINT macro, preventing capitalization of the CINT macro name.
#define G__DEFINE_MACRO_S(macro) \
   G__defineMacro(#macro, (long)macro, 0, false)

//______________________________________________________________________________
// Define compiler macro with value, both system macro and CINT macro, preventing capitalization of the CINT macro name.
#define G__DEFINE_MACRO_S_C(macro) \
   G__defineMacro(#macro, (long)macro, 0, false, true)

//______________________________________________________________________________
static void G__platformMacro()
{
   //  (G__globalcomp == G__NOLINK) means first pass, before G__globalcomp has been defined.
   // Those are not really 'types' but so far they are used as such  .
   // They correspond to the cint 'type' value:
   /****************************************************
    * Automatic variable and macro
    *   p : macro int
    *   P : macro double
    *   o : auto int
    *   O : auto double
    ****************************************************/
   {
      //fprintf(stderr, "G__platformMacro: calling Reflex::TypedefTypeBuilder for 'macroInt$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("macroInt$", ::Reflex::ConstBuilder(::Reflex::Type::ByName("int")), (Reflex::REPRESTYPE) 'p'); // type 'p'
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__platformMacro: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }

   {
      //fprintf(stderr, "G__platformMacro: calling Reflex::TypedefTypeBuilder for 'macroDouble$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("macroDouble$", ::Reflex::ConstBuilder(::Reflex::Type::ByName("double")), (Reflex::REPRESTYPE) 'P'); // type 'P'
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__platformMacro: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }

   {
      //fprintf(stderr, "G__platformMacro: calling Reflex::TypedefTypeBuilder for 'autoInt$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("autoInt$", ::Reflex::Type::ByName("int"), (Reflex::REPRESTYPE) 'o'); // type 'o'
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__platformMacro: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }

   {
      //fprintf(stderr, "G__platformMacro: calling Reflex::TypedefTypeBuilder for 'autoDouble$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("autoDouble$", ::Reflex::Type::ByName("double"), (Reflex::REPRESTYPE)'O'); // type 'O'
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__platformMacro: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }

   {
      //fprintf(stderr, "G__platformMacro: calling Reflex::TypedefTypeBuilder for 'macroChar*$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("macroChar*$", ::Reflex::PointerBuilder(::Reflex::ConstBuilder(::Reflex::Type::ByName("char"))),  (Reflex::REPRESTYPE)'T'); // type 'T'
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__platformMacro: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }

   {
      //fprintf(stderr, "G__platformMacro: calling Reflex::TypedefTypeBuilder for 'macro$'\n");
      ::Reflex::Type result = ::Reflex::TypedefTypeBuilder("macro$", ::Reflex::PointerBuilder(::Reflex::ConstBuilder(::Reflex::Type::ByName("char"))), (Reflex::REPRESTYPE) 'j'); // type 'j'
      G__RflxProperties* prop = G__get_properties(result);
      if (prop) {
         //fprintf(stderr, "G__platformMacro: registering typedef '%s'\n", result.Name(Reflex::QUALIFIED).c_str());
         prop->typenum = G__Dict::GetDict().Register(result);
      }
   }


   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   /***********************************************************************
    * operating system
    ***********************************************************************/
#if defined(__linux__)  /* Linux */
   G__DEFINE_MACRO(__linux__);
#elif defined(__linux)
   G__DEFINE_MACRO(__linux);
#elif defined(linux)
   G__DEFINE_MACRO(linux);
#endif
#if defined(_FreeBSD_) && !defined(__FreeBSD__)
# define __FreeBSD__ _FreeBSD_
#endif
#ifdef __FreeBSD__   /* FreeBSD */
   G__DEFINE_MACRO_N(__FreeBSD__, "G__FBSD");
#endif
#ifdef __OpenBSD__   /* OpenBSD */
   G__DEFINE_MACRO_N(__OpenBSD__, "G__OBSD");
#endif
#ifdef __hpux        /* HP-UX */
   G__DEFINE_MACRO(__hpux__);
#endif
#ifdef __sun         /* SunOS and Solaris */
   G__DEFINE_MACRO(__sun);
#endif
#ifdef _WIN32        /* Windows 32bit */
   G__DEFINE_MACRO(_WIN32);
#endif
#ifdef _WINDOWS_     /* Windows */
   G__DEFINE_MACRO(_WINDOWS_);
#endif
#ifdef __APPLE__     /* Apple MacOS X */
   G__DEFINE_MACRO(__APPLE__);
#endif
#ifdef __VMS         /* DEC/Compac VMS */
   G__DEFINE_MACRO(__VMS);
#endif
#ifdef _AIX          /* IBM AIX */
   G__DEFINE_MACRO(_AIX);
#endif
#ifdef __sgi         /* SGI IRIX */
   G__DEFINE_MACRO(__sgi);
#endif
#if defined(__alpha) && !defined(__linux) && !defined(__linux__) && !defined(linux) /* DEC/Compac Alpha-OSF operating system */
   G__DEFINE_MACRO(__alpha);
#endif
#ifdef __QNX__         /* QNX realtime OS */
   G__DEFINE_MACRO(__QNX__);
#endif
   /***********************************************************************
    * compiler and library
    ***********************************************************************/
#ifdef G__MINGW /* Mingw */
   G__DEFINE_MACRO_C(G__MINGW);
#endif
#ifdef __CYGWIN__ /* Cygwin */
   G__DEFINE_MACRO(__CYGWIN__);
#endif
#ifdef __GNUC__  /* gcc/g++  GNU C/C++ compiler major version */
   G__DEFINE_MACRO_C(__GNUC__);
#endif
#ifdef __GNUC_MINOR__  /* gcc/g++ minor version */
   G__DEFINE_MACRO_C(__GNUC_MINOR__);
#endif
#if defined(__GNUC__) && defined(__GNUC_MINOR__)
   if (G__globalcomp == G__NOLINK) {
      sprintf(temp, "G__GNUC_VER=%ld", (long)__GNUC__*1000 + __GNUC_MINOR__);
      G__add_macro(temp);
   }
#endif
#ifdef __GLIBC__   /* GNU C library major version */
   G__DEFINE_MACRO(__GLIBC__);
#endif
#ifdef __GLIBC_MINOR__  /* GNU C library minor version */
   G__DEFINE_MACRO(__GLIBC_MINOR__);
#endif
#ifdef __HP_aCC     /* HP aCC C++ compiler */
   if (G__globalcomp == G__NOLINK) {
      sprintf(temp, "G__HP_aCC=%ld", (long)__HP_aCC);
      G__add_macro(temp);
   }
   G__DEFINE_MACRO_S_C(__HP_aCC);
#if __HP_aCC > 15000
   if (G__globalcomp == G__NOLINK) {
      sprintf(temp, "G__ANSIISOLIB=1");
      G__add_macro(temp);
   }
#endif
#endif
#ifdef __SUNPRO_CC  /* Sun C++ compiler */
   G__DEFINE_MACRO_C(__SUNPRO_CC);
#endif
#ifdef __SUNPRO_C   /* Sun C compiler */
   G__DEFINE_MACRO_C(__SUNPRO_C);
#endif
#ifdef _STLPORT_VERSION
   // stlport version, used on e.g. SUN
   G__DEFINE_MACRO_C(_STLPORT_VERSION);
#endif
#ifdef G__VISUAL    /* Microsoft Visual C++ compiler */
   if (G__globalcomp == G__NOLINK) {
      sprintf(temp, "G__VISUAL=%ld", (long)G__VISUAL);
      G__add_macro(temp);
   }
#endif
#ifdef _MSC_VER     /* Microsoft Visual C++ version */
   G__DEFINE_MACRO_C(_MSC_VER);
   if (G__globalcomp == G__NOLINK) {
#ifdef _HAS_ITERATOR_DEBUGGING
      sprintf(temp, "G__HAS_ITERATOR_DEBUGGING=%d", _HAS_ITERATOR_DEBUGGING);
      G__add_macro(temp);
#endif
#ifdef _SECURE_SCL
      sprintf(temp, "G__SECURE_SCL=%d", _SECURE_SCL);
      G__add_macro(temp);
#endif
   }
#endif
#ifdef __SC__       /* Symantec C/C++ compiler */
   G__DEFINE_MACRO_N_C(__SC__, "G__SYMANTEC");
#endif
#ifdef __BORLANDC__ /* Borland C/C++ compiler */
   G__DEFINE_MACRO_C(__BORLANDC__);
#endif
#ifdef __BCPLUSPLUS__  /* Borland C++ compiler */
   G__DEFINE_MACRO_C(__BCPLUSPLUS__);
#endif
#ifdef G__BORLANDCC5 /* Borland C/C++ compiler 5.5 */
   G__DEFINE_MACRO_C(__BORLANDCC5__);
#endif
#ifdef __KCC        /* KCC  C++ compiler */
   G__DEFINE_MACRO_C(__KCC__);
#endif
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER<810) /* icc and ecc C++ compilers */
   G__DEFINE_MACRO_C(__INTEL_COMPILER);
#endif
#ifndef _AIX
#ifdef __xlc__ /* IBM xlc compiler */
   if (G__globalcomp == G__NOLINK) {
      sprintf(temp, "G__GNUC=%ld", (long)3 /*__GNUC__*/);
      G__add_macro(temp);
      sprintf(temp, "G__GNUC_MINOR=%ld", (long)3 /*__GNUC_MINOR__*/);
      G__add_macro(temp);
   }
   G__DEFINE_MACRO_C(__xlc__);
#endif
#endif //G__SIGNAL

   if (G__globalcomp == G__NOLINK) {
      G__initcxx();
   }

   /***********************************************************************
    * micro processor
    ***********************************************************************/
#ifdef __hppa__ /* HP-PA , Hewlett Packard Precision Architecture */
   G__DEFINE_MACRO_S(__hppa__);
#endif
#ifdef __i386__ /* Intel 386,486,586 */
   G__DEFINE_MACRO_S(__i386__);
#endif
#ifdef __i860__ /* Intel 860 */
   G__DEFINE_MACRO_S(__i860__);
#endif
#ifdef __i960__ /* Intel 960 */
   G__DEFINE_MACRO_N(__i960__, "G__i860");
#endif
#ifdef __ia64__ /* Intel Itanium */
   G__DEFINE_MACRO_S(__ia64__);
#endif
#ifdef __m88k__ /* Motorola 88000 */
   G__DEFINE_MACRO_S(__m88k__);
#endif
#ifdef __m68k__ /* Motorola 68000 */
   G__DEFINE_MACRO_S(__m68k__);
#endif
#ifdef __ppc__  /* Motorola Power-PC */
   G__DEFINE_MACRO_S(__ppc__);
#endif
#ifdef __PPC__  /* IBM Power-PC */
   G__DEFINE_MACRO(__PPC__);
#endif
#ifdef __mips__ /* MIPS architecture */
   G__DEFINE_MACRO_S(__mips__);
#endif
#ifdef __alpha__ /* DEC/Compac Alpha */
   G__DEFINE_MACRO_S(__alpha__);
#endif
#if defined(__sparc) /* Sun Microsystems SPARC architecture */
   G__DEFINE_MACRO(__sparc);
#elif  defined(__sparc__)
   G__DEFINE_MACRO(__sparc__);
#endif
#ifdef __arc__  /* ARC architecture */
   G__DEFINE_MACRO_S(__arc__);
#endif
#ifdef __M32R__
   G__DEFINE_MACRO_S(__M32R__);
#endif
#ifdef __sh__   /* Hitachi SH micro-controller */
   G__DEFINE_MACRO_S(__sh__);
#endif
#ifdef __arm__  /* ARM , Advanced Risk Machines */
   G__DEFINE_MACRO_S(__arm__);
#endif
#ifdef __s390__ /* IBM S390 */
   G__DEFINE_MACRO_S(__s390__);
#endif
   if (G__globalcomp != G__NOLINK)
      return;

   /***********************************************************************
    * application environment
    ***********************************************************************/
#ifdef G__ROOT
   sprintf(temp, "G__ROOT=%ld", (long)G__ROOT);
   G__add_macro(temp);
#endif
#ifdef G__NO_STDLIBS
   sprintf(temp, "G__NO_STDLIBS=%ld", (long)G__NO_STDLIBS);
   G__add_macro(temp);
#endif
#ifdef G__NATIVELONGLONG
   sprintf(temp, "G__NATIVELONGLONG=%ld", (long)G__NATIVELONGLONG);
   G__add_macro(temp);
#endif

   sprintf(temp, "int& G__cintv6=*(int*)(%ld);", (long) &G__cintv6);
   G__exec_text(temp);

   // setup size_t, ssize_t
   int size_t_type = 0;
   if (sizeof(size_t) == G__INTALLOC)
      size_t_type = 'i';
   else if (sizeof(size_t) == G__LONGALLOC)
      size_t_type = 'l';
   else if (sizeof(size_t) == G__LONGLONGALLOC)
      size_t_type = 'n';
   else G__fprinterr(G__serr, "On your platform, size_t has a weird size of %d which is not handled yet!\n",
                        sizeof(size_t));

   G__search_typename2("size_t", size_t_type - 1, -1, 0, -1);
   G__setnewtype(-1, NULL, 0);

   G__search_typename2("ssize_t", size_t_type, -1, 0, -1);
   G__setnewtype(-1, NULL, 0);

   if (G__ignore_stdnamespace) {
      Reflex::NamespaceBuilder("std");
      Reflex::Scope std = Reflex::Scope::ByName("std");
      if (std)
         Reflex::Scope::GlobalScope().AddUsingDirective(std);
   }
}

//______________________________________________________________________________
void Cint::Internal::G__set_stdio()
{
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;

   G__globalvarpointer = G__PVOID ;

   G__intp_sout = G__sout;
   G__intp_serr = G__serr;
   G__intp_sin = G__sin;

   // FILE is a fundamental type for CINT
   ::Reflex::ClassBuilder("FILE", typeid(FILE), 0, ::Reflex::STRUCT).EnableCallback(false);

   G__var_type = 'E';
   sprintf(temp, "stdout=(FILE*)(%ld)", (long)G__intp_sout);
   G__getexpr(temp);

   G__var_type = 'E';
   sprintf(temp, "stderr=(FILE*)(%ld)", (long)G__intp_serr);
   G__getexpr(temp);

   G__var_type = 'E';
   sprintf(temp, "stdin=(FILE*)(%ld)", (long)G__intp_sin);
   G__getexpr(temp);

   G__platformMacro();
   G__definemacro = 1;
   sprintf(temp, "EOF=%ld", (long)EOF);
   G__getexpr(temp);
   sprintf(temp, "NULL=%ld", (long)NULL);
   G__getexpr(temp);
#ifdef G__SHAREDLIB
   sprintf(temp, "G__SHAREDLIB=1");
   G__getexpr(temp);
#endif
#if defined(G__P2FCAST) || defined(G__P2FDECL)
   sprintf(temp, "G__P2F=1");
   G__getexpr(temp);
#endif
#ifdef G__NEWSTDHEADER
   sprintf(temp, "G__NEWSTDHEADER=1");
   G__getexpr(temp);
#endif
   G__definemacro = 0;

   /* G__constvar=G__CONSTVAR; G__var_type='g'; G__getexpr("TRUE=1"); */
   /* G__constvar=G__CONSTVAR; G__var_type='g'; G__getexpr("FALSE=0"); */
   G__constvar = G__CONSTVAR;
   G__var_type = 'g';
   G__getexpr("true=1");
   G__constvar = G__CONSTVAR;
   G__var_type = 'g';
   G__getexpr("false=0");
   G__constvar = 0;

#ifdef G__DUMPFILE
   G__globalvarpointer = (char*)(&G__dumpfile);
   G__var_type = 'E';
   G__getexpr("G__dumpfile=0");
#endif
   G__globalvarpointer = G__PVOID;

   G__var_type = 'p';
   G__tagnum = ::Reflex::Scope::GlobalScope();
   G__typenum = ::Reflex::Type();
}

//______________________________________________________________________________
extern "C" void G__set_stdio_handle(FILE* sout, FILE* serr, FILE* sin)
{
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;

   G__sout = G__stdout = sout;
   G__serr = G__stderr = serr;
   G__sin  = G__stdin  = sin;

   G__var_type = 'E';
   sprintf(temp, "stdout=(FILE*)(%ld)", (long)G__intp_sout);
   G__getexpr(temp);

   G__var_type = 'E';
   sprintf(temp, "stderr=(FILE*)(%ld)", (long)G__intp_serr);
   G__getexpr(temp);

   G__var_type = 'E';
   sprintf(temp, "stdin=(FILE*)(%ld)", (long)G__intp_sin);
   G__getexpr(temp);
}

//______________________________________________________________________________
extern "C" const char* G__cint_version()
{
   static std::string static_version;
   static_version = G__CINTVERSIONSTR;
   return static_version.c_str(); // For example: "5.14.34, Mar 10 2000"
}

//______________________________________________________________________________
int Cint::Internal::G__cintrevision(FILE* fp)
{
   fprintf(fp, "\n");
   fprintf(fp, "cint : C/C++ interpreter  (mailing list 'cint@root.cern.ch')\n");
   fprintf(fp, "   Copyright(c) : 1995~2005 Masaharu Goto (gotom@hanno.jp)\n");
   fprintf(fp, "   revision     : %s by M.Goto\n\n", G__cint_version());
#ifdef G__DEBUG
   fprintf(fp, "   MEMORY LEAK TEST ACTIVATED!!! MAYBE SLOW.\n\n");
#endif // G__DEBUG
   return 0;
}

//______________________________________________________________________________
G__ConstStringList* Cint::Internal::G__AddConstStringList(G__ConstStringList* current, char* str, int islen)
{
   int itemp;
   struct G__ConstStringList* next;

   next = (struct G__ConstStringList*)malloc(sizeof(struct G__ConstStringList));

   next->string = (char*)malloc(strlen(str) + 1);
   strcpy(next->string, str);

   if (islen) {
      next->hash = strlen(str);
   }
   else {
      G__hash(str, next->hash, itemp);
   }

   next->prev = current;

   return(next);
}

//______________________________________________________________________________
void Cint::Internal::G__DeleteConstStringList(G__ConstStringList* current)
{
   struct G__ConstStringList* tmp;
   while (current) {
      if (current->string) free((void*)current->string);
      tmp = current->prev;
      free((void*)current);
      current = tmp;
   }
}

//______________________________________________________________________________
extern "C" void G__LockCpp()
{
   /* Same as option -A */
   G__cpplock = 1;
   G__iscpp = 1;
}

//______________________________________________________________________________
extern "C" void G__SetCatchException(int mode)
{
   G__catchexception = mode;
}

//______________________________________________________________________________
extern "C" int G__init_cint(const char* command)
{
   //  Initialize cint, read source file (if any) and return to host program.
   //
   //  After calling this function, G__calc() or G__pause() has to be explicitly
   //  called to start interpretation or interactive interface.
   //
   //  Called by nothing.  User host application should call this.
   //
   //  Returns:
   //
   //    G__CINT_INIT_FAILURE   (-1) : If initialization failed
   //    G__CINT_INIT_SUCCESS      0 : Initialization success.
   //                                  no main() found in given source.
   //    G__CINT_INIT_SUCCESS_MAIN 1 : Initialization + main() executed successfully.
   //
   //  Example user source code:
   //
   //     G__init_cint("cint source.c"); // read source.c and allocate globals
   //     G__calc("func1()"); // interpret func1() which should be in source.c
   //     while (!G__pause()) {} // start interactive interface
   //     G__calc("func2()"); // G__calc(), G__pause() can be called multiple times
   //     G__scratch_all(); // terminate cint
   //     G__init_cint("cint source2.c"); // you can start cint again
   //     G__calc("func3()"); // G__calc(), G__pause() can be called multiple times
   //     G__scratch_all(); // terminate cint
   //
   int argn = 0;
   int i;
   int result;
   char* arg[G__MAXARG];
   G__StrBuf argbuf_sb(G__LONGLINE);
   char* argbuf = argbuf_sb;
   G__LockCriticalSection();
   /***************************************
   * copy command to argbuf. argbuf will
   * be modified by G__split().
   ***************************************/
   if (G__commandline != command) {
      strcpy(G__commandline, command);
   }
   strcpy(argbuf, command);
   /***************************************
   * split arguments as follows
   *
   *        arg[0]
   *   <--------------->
   *  "cint -s source1.c"
   *    ^   ^        ^
   * arg[1] arg[2]  arg[3]
   ***************************************/
   G__split(G__commandline, argbuf, &argn, arg);
   /***************************************
   * shift argument as follows
   *
   *  "cint -s source1.c"
   *    ^   ^        ^
   * arg[0] arg[1]  arg[2]
   ***************************************/
   for (i = 0;i < argn;i++) arg[i] = arg[i+1];
   while (i < G__MAXARG) {
      arg[i] = (char*)NULL;
      ++i;
   }
   /***************************************
   * G__othermain is set to 2 so that
   * G__main() returns without start
   * interpretation.
   *
   *  G__main() returns 0 if it is OK,
   * -1 if error occur.
   ***************************************/
   G__othermain = 2;
   result = G__main(argn, arg);
   if (G__MAINEXIST == G__ismain) {
      G__UnlockCriticalSection();
      return(G__INIT_CINT_SUCCESS_MAIN);
   }
   else if (result == EXIT_SUCCESS) {
      G__UnlockCriticalSection();
      return(G__INIT_CINT_SUCCESS);
   }
   else {
      G__UnlockCriticalSection();
      return(G__INIT_CINT_FAILURE);
   }
}

//______________________________________________________________________________
extern "C" int G__load(char* commandfile)
{
   //  USING THIS FUNCTION IS NOT RECOMMENDED
   //
   //  Called by nothing. User host application should call this
   //
   //  If main() exists in user compiled function, G__load() is an entry.
   //  Command file has to contain
   //     cint < [options] > [file1] < [file2] < [file3] <...>>>
   //
   int argn = 0, i;
   char *arg[G__MAXARG], line[G__LONGLINE*2], argbuf[G__LONGLINE*2];
   FILE *cfp;
   cfp = fopen(commandfile, "r");
   if (cfp == NULL) {
      fprintf(stderr, "Error: command file \"%s\" doesn't exist\n"
              , commandfile);
      fprintf(stderr, "  Make command file : [comID] <[cint options]> [file1] <[file2]<[file3]<...>>>\n");
      return(-1);
   }
   while (G__readline(cfp, line, argbuf, &argn, arg) != 0) {
      for (i = 0;i < argn;i++) arg[i] = arg[i+1];
      arg[argn] = NULL;

      /*******************************************************
       * Ignore line start with # and blank lines.
       * else call G__main()
       *******************************************************/
      if ((argn > 0) && (arg[0][0] != '#')) {
         G__othermain = 1;
         G__main(argn, arg);
         if (G__return > G__RETURN_EXIT1) return(0);
         G__return = G__RETURN_NON;
      }
   }
   fclose(cfp);
   return 0;
}

//______________________________________________________________________________
extern "C" int G__main(int argc, char** argv)
{
   // Main entry of the C/C++ interpreter.
   int stepfrommain = 0;
   char* forceassignment = 0;
   int xfileflag = 0;
   G__StrBuf sourcefile_sb(G__MAXFILENAME);
   char* sourcefile = sourcefile_sb;
   extern int optind;
   extern char* optarg;
   static char usage[] = "Usage: %s [options] [sourcefiles|suboptions] [arguments]\n";
   static char* progname;
   char* ppc;
   int c;
   int iarg;
   G__StrBuf temp_sb(G__ONELINE);
   char* temp = temp_sb;
   long G__argpointer[G__MAXARG];
   G__StrBuf dumpfile_sb(G__MAXFILENAME);
   char* dumpfile = dumpfile_sb;
   G__value result = G__null;
   const char* linkfilename = 0;
   int linkflag = 0;
   const char* dllid = 0;
   G__dictposition stubbegin;
   char* icom = 0;
   //
   // Setting STDIOs.  May need to modify in init.c, end.c, scrupto.c, pause.c.
   //
   if (!G__stdin) {
      G__stdin = stdin;
   }
   if (!G__stdout) {
      G__stdout = stdout;
   }
   if (!G__stderr) {
      G__stderr = stderr;
   }
   G__sin = G__stdin;
   G__sout = G__stdout;
   G__serr = G__stderr;
#ifdef G__MEMTEST
   G__memanalysis();
#endif // G__MEMTEST
   G__scratch_all();
   if (G__beforeparse_hook) {
      G__beforeparse_hook();
   }
#ifdef G__SECURITY
   G__init_garbagecollection();
#endif // G__SECURITY
   for (int i = 0; i <= 5; ++i) {
      G__dumpreadline[i] = 0;
      G__Xdumpreadline[i] = 0;
   }
   // Get command name.
   if (argv[0]) {
      sprintf(G__nam, "%s", argv[0]);
   }
   else {
      strcpy(G__nam, "cint");
   }
   G__macros[0] = '\0';
   G__set_stdio();
#if defined(G__ROOT) && !defined(G__NATIVELONGLONG)
   {
      int xtagnum, xtypenum;
      G__cpp_setuplongif();
      xtagnum = G__defined_tagname("G__longlong", 2);
      xtypenum = G__search_typename("long long", 'u', xtagnum, G__PARANORMAL);
      xtagnum = G__defined_tagname("G__ulonglong", 2);
      xtypenum = G__search_typename("unsigned long long", 'u', xtagnum, G__PARANORMAL);
      xtagnum = G__defined_tagname("G__longdouble", 2);
      xtypenum = G__search_typename("long double", 'u', xtagnum, G__PARANORMAL);
   }
#endif // G__ROOT && !G__NATIVELONGLONG
   //
   //  Signal handling moved after getopt
   //  to enable core dump with 'E'.
   //
#ifdef G__HSTD
   //
   // TEE software error handling
   //
   if (setjmp(EH_env)) {
      G__errorprompt("TEE software error");
      G__scratch_all();
      return EXIT_FAILURE;
   }
#endif // G__HSTD
   G__breakline[0] = '\0';
   G__breakfile[0] = '\0';
   if (G__allincludepath) {
      free(G__allincludepath);
   }
   G__allincludepath = (char*) malloc(1);
   G__allincludepath[0] = '\0';
   G__prerun = 1;
   G__setdebugcond();
   G__globalsetup();
   G__call_setup_funcs();
   G__do_p2fsetup();
   G__prerun = 0;
   G__setdebugcond();
   progname = argv[0];
   dumpfile[0] = '\0';
#ifndef G__TESTMAIN
   optind = 1;
#endif // G__TESTMAIN
   //
   //  Get command options.
   //
   while (
      (c = getopt(argc, argv, "a:b:c:d:ef:gij:kl:mn:pq:rstu:vw:x:y:z:AB:CD:EF:G:H:I:J:KM:N:O:P:QRSTU:VW:X:Y:Z:-:@+:")) != EOF
   ) {
      switch (c) {
         case '+':
            G__setmemtestbreak(atoi(optarg) / 10000, atoi(optarg) % 10000);
            break;
         case '@': // OBSOLETE, was here to set G__cintv6, the flag controlling the never-implemented bytecode machine.
            break;
         case 'H': // limit on level of inclusion for dictionary generation
            G__gcomplevellimit = atoi(optarg);
            break;
         case 'J': // 0 - 5
            G__dispmsg = atoi(optarg);
            break;
         case 'j':
            G__multithreadlibcint = atoi(optarg);
            break;
         case 'm':
            G__lang = G__ONEBYTE;
            break;
         case 'Q':
            G__quiet = 1;
            break;
         case 'B':
            G__setInitFunc(optarg);
            break;
         case 'C':
            G__setcopyflag(1);
            break;
         case 'y':
            G__setCINTLIBNAME(optarg);
            break;
         case 'z':
            G__setPROJNAME(optarg);
            break;
         case 'w': /* set Windows-NT DLL mode */
            G__setDLLflag(atoi(optarg));
            break;
         case 'M':
            G__is_operator_newdelete = (int) G__int(G__calc_internal((optarg)));
            if (G__is_operator_newdelete & G__NOT_USING_2ARG_NEW) {
               G__fprinterr(G__serr, "!!!-M option may not be needed any more. A new scheme has been implemented.\n");
               G__fprinterr(G__serr, "!!!Refer to $CINTSYSDIR/doc/makecint.txt for the detail.\n");
            }
            break;
         case 'V': /* Create precompiled private member dictionary */
            G__precomp_private = 1;
            break;
         case 'q':
            // --
#ifdef G__SECURITY
            G__security = G__getsecuritycode(optarg);
#endif // G__SECURITY
            break;
         case 'e': /* Parse all extern declaration */
            G__parseextern = 1;
            break;
         case 'i': /* interactive return */
            G__interactive = 1;
            break;
         case 'x':
            if (xfileflag) {
               G__fprinterr(G__serr, "Error: only one -x option is allowed\n");
               G__exit(EXIT_FAILURE);
            }
            else {
               xfileflag = optind - 1;
            }
            break;
         case 'I': /* Include path */
            if (optarg[0] == ',') {
               G__read_include_env(optarg + 1);
            }
            else {
               G__add_ipath(optarg);
            }
            break;
         case 'F': /* force assignment is evaluated after pre-RUN */
            forceassignment = optarg ;
            break;
         case 'Y': /* Do not ignore std namespace or not */
            G__ignore_stdnamespace = atoi(optarg);
            break;
         case 'Z': /* Generate archived header */
            G__autoload_stdheader = atoi(optarg);
            break;
         case 'n': /* customize G__cpplink file name, G__cppXXX?.C , G__cXXX.c */
            if (linkflag) {
               G__fprinterr(G__serr, "Warning: option -n[linkname] must be given prior to -c[linklevel]\n");
            }
            linkfilename = optarg;
            break;
         case 'N': /* customize DLL identification name */
            if (linkflag) {
               G__fprinterr(G__serr, "Warning: option -N[DLL_Name] must be given prior to -c[linklevel]\n");
            }
            dllid = optarg;
            break;
         case 'U':
            G__SystemIncludeDir = G__AddConstStringList(G__SystemIncludeDir, optarg, 1);
            break;
         case 'u':
            if (!G__fpundeftype) {
               G__security = 0;
               G__fpundeftype = fopen(optarg, "w");
               fprintf(G__fpundeftype, "/* List of possible undefined type names. This list is not perfect.\n");
               fprintf(G__fpundeftype, "* It is user's responsibility to modify this file. \n");
               fprintf(G__fpundeftype, "* There are cases that the undefined type names can be\n");
               fprintf(G__fpundeftype, "*   class name\n");
               fprintf(G__fpundeftype, "*   struct name\n");
               fprintf(G__fpundeftype, "*   union name\n");
               fprintf(G__fpundeftype, "*   enum name\n");
               fprintf(G__fpundeftype, "*   typedef name\n");
               fprintf(G__fpundeftype, "*   not a typename but object name by CINT's mistake\n");
               fprintf(G__fpundeftype, "* but CINT can not distinguish between them. So it outputs 'class xxx;' \n");
               fprintf(G__fpundeftype, "* for all.\n");
               fprintf(G__fpundeftype, "*/\n");
            }
            break;
         case 'l': // dynamic link file, shared library file
            {
               // --
#ifdef G__SHAREDLIB
               int err = G__shl_load(optarg);
               if (err == -1) {
                  if (G__key) {
                     system("key .cint_key -l execute");
                  }
                  G__scratch_all();
                  return EXIT_FAILURE;
               }
#else // G__SHAREDLIB
               G__fprinterr(G__serr, "Error: %s is not compiled with dynamic link capability\n", argv[0]);
#endif // G__SHAREDLIB
               // --
            }
            break;
         case 'p': /* use preprocessor */
            G__cpp = 1;
            break;
         case 'P': /* specify what preprocessor to use */
            switch (optarg[1]) {
               case 'p':
                  strcpy(G__ppopt, optarg + 2);
                  ppc = G__ppopt;
                  while ((ppc = strchr(ppc, ','))) * ppc = ' ';
                  break;
               case 'l':
                  break;
            }
            break;
         case 'W': /* specify what preprocessor to use */
            switch (optarg[1]) {
               case 'p':
                  strcpy(G__ppopt, optarg + 2);
                  ppc = G__ppopt;
                  while ((ppc = strchr(ppc, ','))) * ppc = ' ';
                  break;
               case 'l':
                  break;
            }
            break;
         case 'A': /* C++ mode lock */
            G__cpplock = 1;
            G__iscpp = 1;
            break;
         case 'K': /* C mode lock */
            G__clock = 0;
            if (G__globalcomp == G__CLINK) {
               G__clock = 1;
            }
            G__iscpp = 0;
            break;
         case 'g': /* whole function compile off */
            G__asm_loopcompile = 3;
            G__asm_loopcompile_mode = G__asm_loopcompile;
            break;
         case 'O': /* loop compiler on */
            G__asm_loopcompile = atoi(optarg);
            G__asm_loopcompile_mode = G__asm_loopcompile;
            break;
         case 'v': /* loop compiler debug mode */
            G__asm_dbg = 1;
            break;
         case 'D': /* define macro */
            G__add_macro(optarg);
            break;
         case 'E': /* Dump core at error */
            if (G__catchexception == 1) {
               G__catchexception = 0;
            }
            else if (!G__catchexception) {
               G__catchexception = 2;
            }
            else {
               ++G__catchexception;
            }
            ++G__coredump;
            break;
         case 'X': /* readline dumpfile execution */
            G__dumpreadline[0] = fopen(optarg, "r");
            if (G__dumpreadline[0]) {
               G__Xdumpreadline[0] = 1;
               G__fprinterr(G__serr, " -X : readline dumpfile %s executed\n", optarg);
            }
            else {
               G__fprinterr(G__serr, "Readline dumpfile %s can not open\n" , optarg);
               return EXIT_FAILURE;
            }
            break;
         case 'd': /* dump file */
            // --
#ifdef G__DUMPFILE
            G__fprinterr(G__serr, " -d : dump function call history to %s\n", optarg);
            if (
               !strcmp(optarg + strlen(optarg) - 2, ".c") ||
               !strcmp(optarg + strlen(optarg) - 2, ".C") ||
               !strcmp(optarg + strlen(optarg) - 2, ".h") ||
               !strcmp(optarg + strlen(optarg) - 2, ".H")
            ) {
               G__fprinterr(G__serr, "-d %s : Improper history dump file name\n", optarg);
            }
            else {
               sprintf(dumpfile, "%s", optarg);
            }
#else // G__DUMPFILE
            G__fprinterr(G__serr, " -d : func call dump not supported now\n");
#endif // G__DUMPFILE
            break;
         case 'k': /* user function key */
            system("key .cint_key -l pause");
            G__key = 1;
            break;
         case 'c': /* set dictionary generation flags, global compile */
            // c-1 : create global variable & function information from
            //      C++ header file. Default link on for pure cint, off for ROOT
            // c-2 : create global variable & function information from
            //      C header file. Default link on for pure cint, off for ROOT
            // c-10 : create global variable & function information from
            //      C++ header file. Default link off
            // c-20 : create global variable & function information from
            //      C header file. Default link off
            // c-11 : create global variable & function information from
            //      C++ header file. Default link on
            // c-21 : create global variable & function information from
            //      C header file. Default link on
            G__globalcomp = atoi(optarg);
            if (abs(G__globalcomp) >= 10) {
               G__default_link = abs(G__globalcomp) % 10;
               G__globalcomp /= 10;
            }
            linkflag = 1;
            if (!linkfilename) {
               switch (G__globalcomp) {
                  case G__CPPLINK:
                     linkfilename = "G__cpplink.C";
                     G__iscpp = 1;
                     G__cpplock = 1;
                     break;
                  case G__CLINK:
                     linkfilename = "G__clink.c";
                     G__iscpp = 0;
                     G__clock = 1;
                     break;
                  case R__CPPLINK:
                     linkfilename = "G__cpplink_rflx.cxx";
                     G__iscpp = 1;
                     G__cpplock = 1;
                     break;
                  default:
                     linkfilename = "G__cpplink.cxx";
                     break;
               }
            }
            if (!dllid) {
               dllid = "";
            }
            G__set_globalcomp(optarg, linkfilename, dllid);
            break;
         case 's': /* step into mode */
            G__fprinterr(G__serr, " -s : Step into function/loop mode\n");
            G__steptrace = 1;
            break;
         case 'S': /* step over mode */
            G__stepover = 3;
            G__fprinterr(G__serr, " -S : Step over function/loop mode\n");
            stepfrommain = 1;
            break;
         case 'b': /* break line */
            strcpy(G__breakline, optarg);
            break;
         case 'f': /* break file */
            strcpy(G__breakfile, optarg);
            break;
         case 'a': /* assertion */
            strcpy(G__assertion, optarg);
            break;
         case 'T': /* trace of input file */
            G__fprinterr(G__serr, " -T : trace from pre-run\n");
            G__debugtrace = 1;
            G__istrace = 1;
            G__debug = 1;
            G__setdebugcond();
            break;
         case 'G': /* trace of input file with redirected error output */
            G__serr = fopen(optarg, "w");
            G__fprinterr(G__serr, " -t : trace execution\n");
            G__istrace = 1;
            G__debugtrace = 1;
            break;
         case 't': /* trace of input file */
            G__fprinterr(G__serr, " -t : trace execution\n");
            G__istrace = 1;
            G__debugtrace = 1;
            break;
         case 'R': /* displays input file at the break point*/
            G__fprinterr(G__serr, " -d : display at break point mode\n");
            G__breakdisp = 1;
            break;
         case 'r': /* revision */
            G__revprint(G__sout);
            if (G__key) {
               system("key .cint_key -l execute");
            }
            return EXIT_SUCCESS;
         case '-':
            icom = optarg;
            break;
         default:
            // --
#ifndef G__SMALLOBJECT
            G__more_pause(0, 0);
            fprintf(G__sout, usage, progname);
            G__more_pause(G__sout, 0);
            G__display_note();
            G__more(G__sout, "options     (* : used only with makecint or -c option)\n");
            G__more(G__sout, "  -A : ANSI C++ mode(default)\n");
            G__more(G__sout, "  -b [line] : set break line\n");
            G__more(G__sout, "* -c -1: make C++ precompiled interface method files\n");
            G__more(G__sout, "* -c -2: make C precompiled interface method files\n");
            G__more(G__sout, "* -c -10: make C++ precompiled interface method files. Default link off\n");
            G__more(G__sout, "* -c -20: make C precompiled interface method files. Default link off\n");
            G__more(G__sout, "* -c -11: make C++ precompiled interface method files. Default link on\n");
            G__more(G__sout, "* -c -21: make C precompiled interface method files. Default link on\n");
            G__more(G__sout, "  -C : copy src to $TMPDIR so that src can be changed during cint run\n");
#ifdef G__DUMPFILE
            G__more(G__sout, "  -d [dumpfile] : dump function call history\n");
#endif // G__DUMPFILE
            G__more(G__sout, "  -D [macro] : define macro symbol for #ifdef\n");
            G__more(G__sout, "  -e : Not ignore extern declarations\n");
            G__more(G__sout, "  -E : Dump core at error\n");
            G__more(G__sout, "  -E -E : Exit process at error and uncaught exception\n");
            G__more(G__sout, "  -f [file] : set break file\n");
            G__more(G__sout, "  -F [assignement] : set global variable\n");
            G__more(G__sout, "  -G [tracedmp] : dump exec trace into file\n");
            G__more(G__sout, "* -H[1-100] : level of header inclusion activated for dictionary generation\n");
            G__more(G__sout, "  -i : interactively return undefined symbol value\n");
            G__more(G__sout, "  -I [includepath] : set include file search path\n");
            G__more(G__sout, "* -j [0|1]: Create multi-thread safe DLL(experimental)\n");
            G__more(G__sout, "  -J[0-4] : Display nothing(0)/error(1)/warning(2)/note(3)/all(4)\n");
            /* G__more(G__sout,"  -k : function key on\n"); */
            G__more(G__sout, "  -K : C mode\n");
#ifdef G__SHAREDLIB
            G__more(G__sout, "  -l [dynamiclinklib] : link dynamic link library\n");
#endif // G__SHAREDLIB
            G__more(G__sout, "  -m : Support ISO-8859-x Eurpoean char set (disabling multi-byte char)\n");
            G__more(G__sout, "* -M [newdelmask] : operator new/delete mask for precompiled interface method\n");
            G__more(G__sout, "* -n [linkname] : Specify precompiled interface method filename\n");
            G__more(G__sout, "* -N [DLL_name] : Specify DLL interface method name\n");
            G__more(G__sout, "  -O [0~4] : Loop compiler on(1~5) off(0). Default on(4)\n");
            G__more(G__sout, "  -p : use preprocessor prior to interpretation\n");
            G__more(G__sout, "  -q [security] : Set security level(default 0)\n");
            G__more(G__sout, "  -Q : Quiet mode (no prompt)\n");
            G__more(G__sout, "  -r : revision and linked function/global info\n");
            G__more(G__sout, "  -R : display input file at break point\n");
            G__more(G__sout, "  -s : step execution mode\n");
            G__more(G__sout, "  -S : step execution mode, First stop in main()\n");
            G__more(G__sout, "  -t : trace execution\n");
            G__more(G__sout, "  -T : trace from pre-run\n");
            G__more(G__sout, "  -u [undefout] : listup undefined typenames\n");
            G__more(G__sout, "* -U [dir] : directory to disable interface method generation\n");
            G__more(G__sout, "* -V : Generate symbols for non-public member with -c option\n");
            G__more(G__sout, "  -v : Bytecode compiler debug mode\n");
            G__more(G__sout, "  -X [readlinedumpfile] : Execute readline dumpfile\n");
            G__more(G__sout, "  -x 'main() {...}' : Execute argument as source code\n");
            G__more(G__sout, "  -Y [0|1]: ignore std namespace (default=1:ignore)\n");
            G__more(G__sout, "  -Z [0|1]: automatic loading of standard header files with DLL\n");
            G__more(G__sout, "  --'command': Execute interactive command and terminate Cint\n");
            G__more(G__sout, "suboptions\n");
            G__more(G__sout, "  +V : turn on class title comment mode for following source fies\n");
            G__more(G__sout, "  -V : turn off class title comment mode for following source fies\n");
            G__more(G__sout, "  +P : turn on preprocessor for following source files\n");
            G__more(G__sout, "  -P : turn off preprocessor for following source files\n");
            G__more(G__sout, "* +STUB : stub function header begin\n");
            G__more(G__sout, "* -STUB : stub function header end\n");
            G__more(G__sout, "sourcefiles\n");
            G__more(G__sout, "  Any valid C/C++ source or header files\n");
            G__more(G__sout, "EXAMPLES\n");
            G__more(G__sout, "  $ cint prog.c main.c\n");
            G__more(G__sout, "  $ cint -S prog.c main.c\n");
            G__more(G__sout, "\n");
            if (G__key) {
               system("key .cint_key -l execute");
            }
#endif // G__SMALLOBJECT
            return EXIT_FAILURE;
      }
   }
   //
   //  Redefine the platform macros
   //  if doing dictionary generation.
   //
   if (G__globalcomp != G__NOLINK) {
      G__platformMacro(); // second round, now taking G__globalcomp into account
   }
   //
   //  Create the G__CINTVERSION macro.
   //
   {
      int oldglobalcomp = G__globalcomp;
      G__globalcomp = G__NOLINK;
      sprintf(temp, "G__CINTVERSION=%ld", (long) G__CINTVERSION);
      G__add_macro(temp);
      G__globalcomp = oldglobalcomp;
   }
   //
   //  Catch signals if not embedded in ROOT.
   //
#ifndef G__ROOT
#ifdef G__SIGNAL
#ifndef G__DONT_CATCH_SIGINT
   signal(SIGINT, G__breakkey);
#endif // G__DONT_CATCH_SIGINT
   if (!G__coredump) {
      signal(SIGFPE, G__floatexception);
      signal(SIGSEGV, G__segmentviolation);
#ifdef SIGEMT
      signal(SIGEMT, G__outofmemory);
#endif // SIGEMT
#ifdef SIGBUS
      signal(SIGBUS, G__buserror);
#endif // SIGBUS
      // --
   }
   else if (G__coredump >= 2) {
      signal(SIGFPE, G__errorexit);
      signal(SIGSEGV, G__errorexit);
#ifdef SIGEMT
      signal(SIGEMT, G__errorexit);
#endif // SIGEMT
#ifdef SIGBUS
      signal(SIGBUS, G__errorexit);
#endif // SIGBUS
      // --
   }
#endif // G__SIGNAL
#endif // G__ROOT
   //
   //  Initialize pointer to member function size.
   //
   if (!G__sizep2memfunc) {
      G__sizep2memfunc = G__LONGALLOC + (G__SHORTALLOC * 2);
   }
   //
   //  Open the function call history dumpfile.
   //
#ifdef G__DUMPFILE
   if (dumpfile[0]) {
      G__dumpfile = fopen(dumpfile, "w");
   }
#endif // G__DUMPFILE
   //
   //  If doing dictionary generation,
   //  output the header.
   //
   if (G__globalcomp < G__NOLINK) {
      G__gen_cppheader(0);
   }
   //
   //  Begin of prerun.
   //
   //  Read whole file to allocate global variables,
   //  function statics, and parse function prototypes.
   //
   while (((G__ismain != G__MAINEXIST) && (optind < argc)) || xfileflag) {
      if (!xfileflag) {
         strcpy(sourcefile, argv[optind]);
         ++optind;
      }
      else {
         FILE* tmpf = tmpfile();
         if (!tmpf) {
            xfileflag = 0;
         }
         else {
            fprintf(tmpf, "%s\n", argv[xfileflag]);
            xfileflag = 0;
            fseek(tmpf, 0L, SEEK_SET);
            int load_status = G__loadfile_tmpfile(tmpf);
            if (load_status || (G__eof == 2)) {
               // file not found or unexpected EOF
               if (G__CPPLINK == G__globalcomp || G__CLINK == G__globalcomp) {
                  G__cleardictfile(-1);
               }
               G__scratch_all();
               return EXIT_FAILURE;
            }
            continue;
         }
      }
      if (!strcmp(sourcefile, "+P")) {
         G__cpp = 1;
         continue;
      }
      else if (strcmp(sourcefile, "-P") == 0) {
         G__cpp = 0;
         continue;
      }
      else if (strcmp(sourcefile, "+V") == 0) {
         G__fons_comment = 1;
         continue;
      }
      else if (strcmp(sourcefile, "-V") == 0) {
         G__fons_comment = 0;
         continue;
      }
      else if (strcmp(sourcefile, "+STUB") == 0) {
         G__store_dictposition(&stubbegin);
         continue;
      }
      else if (strcmp(sourcefile, "-STUB") == 0) {
         G__set_stubflags(&stubbegin);
         continue;
      }
      if (G__globalcomp < G__NOLINK) {
         G__gen_cppheader(sourcefile);
      }
      int load_status = G__loadfile(sourcefile);
      if ((load_status < 0) || (G__eof == 2)) {
         // file not found or unexpected EOF
         if ((G__globalcomp == G__CPPLINK) || (G__globalcomp == G__CLINK)) {
            G__cleardictfile(-1);
         }
         G__scratch_all();
         return EXIT_FAILURE;
      }
      if (G__return > G__RETURN_NORMAL) {
         G__scratch_all();
         return EXIT_SUCCESS;
      }
   }
   if (icom) {
      int more = 0;
      G__redirect_on();
      G__init_process_cmd();
      G__StrBuf prompt_sb(G__ONELINE);
      char* prompt = prompt_sb;
      strcpy(prompt, "cint>");
      G__process_cmd(icom, prompt, &more, 0, 0);
      G__scratch_all();
      return EXIT_SUCCESS;
   }
   if (G__afterparse_hook) {
      G__afterparse_hook();
   }
   if (G__security_error) {
      G__fprinterr(G__serr, "Warning: Error occurred during reading source files\n");
   }
   //
   //  If doing dictionary generation,
   //  add the includes of the extra
   //  include files to the dictionary
   //  header file.
   //
   G__gen_extra_include();
   //
   //  If doing C++ dictionary generation,
   //  then do the work and exit.
   //
   if (G__globalcomp == G__CPPLINK) { // C++ header
      if (G__steptrace || G__stepover) {
         while (!G__pause()) {}
      }
      G__gen_cpplink();
#if !defined(G__ROOT) && !defined(G__D0)
      G__scratch_all();
#endif // !G__ROOT && !G__D0
      if (G__security_error) {
         G__fprinterr(G__serr, "Warning: Error occurred during dictionary source generation\n");
         G__cleardictfile(-1);
         return -1;
      }
      G__cleardictfile(EXIT_SUCCESS);
      return EXIT_SUCCESS;
   }
   //
   //  If doing C dictionary generation,
   //  then do the work and exit.
   //
   if (G__globalcomp == G__CLINK) { // C header
      if (G__steptrace || G__stepover) {
         while (!G__pause()) {}
      }
      G__gen_clink();
#if !defined(G__ROOT) && !defined(G__D0)
      G__scratch_all();
#endif // !G__ROOT && !G__D0
      if (G__security_error) {
         G__fprinterr(G__serr, "Warning: Error occured during dictionary source generation\n");
         G__cleardictfile(-1);
         return -1;
      }
      G__cleardictfile(EXIT_SUCCESS);
      return EXIT_SUCCESS;
   }
#ifdef G__ROOT
   //
   //  If doing Reflex dictionary generation,
   //  then do the work and exit.
   //
   if (G__globalcomp == R__CPPLINK) {
      rflx_gendict(linkfilename, sourcefile);
      return EXIT_SUCCESS;
   }
#endif // G__ROOT
   //
   //
   //
   --optind;
   //
   //  End of prerun.
   //
   //  Note:  If we were doing dictionary generation
   //         then we have done it and exited before
   //         getting here.  The following code is
   //         only executed for the interpretor.
   //
   if (G__debugtrace) {
      G__fprinterr(G__serr, "PRE-RUN END\n");
   }
   if (G__breakline[0]) {
      G__setbreakpoint(G__breakline, G__breakfile);
   }
   G__eof = 0;
   G__prerun = 0;
   G__debug = G__debugtrace;
   G__step = G__steptrace;
   if (stepfrommain) {
      G__step = 1;
   }
   G__setdebugcond();
   //
   //  Forceassignment is given by -F command line option.
   //
   //     cint -F(expr1,expr2,...)
   //
   //  Expressions are evaluated after pre-RUN.
   //
   if (forceassignment) {
      G__calc_internal(forceassignment);
   }
   //
   //  Set ready flag for embedded use.
   //
   G__cintready = 1;
   //
   //  If we are operating as an embedded
   //  interpreter, then return to the host.
   //
   switch (G__othermain) {
      case 2:
         if (G__ismain == G__NOMAIN) {
            return EXIT_SUCCESS;
         }
         break;
      case 3: // for drcc
         return EXIT_SUCCESS;
   }
   //
   //  If we were generating a list of undefined types,
   //  then closed output file and exit, we are done.
   //
   //  TODO: Should this go before the test of G__othermain?
   //
   if (G__fpundeftype) {
      fclose(G__fpundeftype);
      G__fpundeftype = 0;
      return EXIT_SUCCESS;
   }
   //
   //  Interpretation begins.
   //
   //  If there is a main() function in the
   //  input file, call that, otherwise enter
   //  the interactive interface.
   //
   if (G__ismain == G__MAINEXIST) {
      if (G__breaksignal) {
         G__fprinterr(G__serr, "\nCALL main()\n");
      }
      //
      //  Initialize arg pointer array.
      //
      for (iarg = optind; iarg < argc; ++iarg) {
         G__argpointer[iarg-optind] = (long) argv[iarg];
      }
      while (iarg < G__MAXARG) {
         G__argpointer[iarg-optind] = 0L;
         ++iarg;
      }
      // Make function name.
      sprintf(temp, "main");
      //
      //  Initialize arguments.
      //
      //  FIXME:  This can be very wrong if there were no arguments.
      //
      G__param para;
      para.paran = 2;
      G__letint(&para.para[0], 'i', argc - optind);
      para.para[0].ref = 0;
      G__letint(&para.para[1], 'C', (long) G__argpointer);
      para.para[1].ref = 0;
      //
      //  Call main().
      //
      G__interpret_func(&result, temp, &para, G__HASH_MAIN, G__p_ifunc, G__EXACT, G__TRYNORMAL);
      //
      if (!G__get_type(G__value_typenum(result))) {
         result = G__null;
      }
      // After main() function, break if step or break mode.
      if (G__breaksignal || (G__return == G__RETURN_EXIT1)) {
         G__return = G__RETURN_NON;
         G__break = 0;
         G__setdebugcond();
         G__fprinterr(G__serr, "!!! return from main() function\n");
#ifdef SIGALRM
         if (G__return == G__RETURN_EXIT1) {
            G__fprinterr(G__serr, "Press return or process will be terminated in %dsec by timeout\n", G__TIMEOUT);
            signal(SIGALRM, G__timeout);
            alarm(G__TIMEOUT);
         }
#endif // SIGALRM
         if (G__catchexception != 2) {
            G__pause();
         }
#ifdef SIGALRM
         if (G__return == G__RETURN_EXIT1) {
            alarm(0);
            G__fprinterr(G__serr, "Time out cancelled\n");
         }
#endif // SIGALRM
         if (G__catchexception != 2) {
            G__pause();
         }
      }
      if (G__stepover) {
         G__step = 0;
         G__setdebugcond();
      }
      //
      //  If G__main() was called from G__init_cint() then
      //  return from G__main() without destroying data.
      //
      if (G__othermain == 2) {
         return G__int(result);
      }
      G__interpretexit();
      return G__int(result);
   }
#ifdef G__TCLTK
   if (G__ismain == G__TCLMAIN) {
      if (G__othermain == 2) {
         return EXIT_SUCCESS;
      }
      G__interpretexit();
      return EXIT_SUCCESS;
   }
#endif // G__TCLTK
   //
   //  No main() function in the input file,
   //  enter the interactive interface.
   //
   if (!G__quiet) {
      G__cintrevision(G__sout);
      fprintf(G__sout, "No main() function found in given source file. Interactive interface started.\n");
      switch (G__ReadInputMode()) {
         case G__INPUTROOTMODE:
            fprintf(G__sout, "'?':help, '.q':quit, 'statement','{statements;}' or '.p [expr]' to evaluate\n");
            break;
         case G__INPUTCXXMODE:
            fprintf(G__sout, "'?':help, '.q':quit, 'statement;','{statements;}' or '.p [expr]' to evaluate\n");
            break;
         case G__INPUTCINTMODE:
         default:
            fprintf(G__sout, "'h':help, 'q':quit, '{statements;}' or 'p [expr]' to evaluate\n");
            break;
      }
   }
   while (1) {
      G__pause();
      if ((G__return > G__RETURN_NORMAL) && (G__return != G__RETURN_EXIT1)) {
         break;
      }
   }
   G__return = G__RETURN_NON;
   G__scratch_all();
   return EXIT_SUCCESS;
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
