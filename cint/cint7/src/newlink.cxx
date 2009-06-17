/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file newlink.c
 ************************************************************************
 * Description:
 *  New style compiled object linkage
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "configcint.h"
#include "Api.h"
#include "Dict.h"
#include "dllrev.h"
#include "value.h"
#include "../../reflex/src/FunctionMember.h"
#include "Reflex/internal/MemberBase.h"
#include "Reflex/Builder/TypeBuilder.h"

#ifndef G__TESTMAIN
#include <sys/stat.h>
#endif // G__TESTMAIN

#include <cstdlib>
#include <cstring>
#include <string>
#include <stack>

//______________________________________________________________________________
#ifdef _WIN32

#include "windows.h"
#include <errno.h>

//______________________________________________________________________________
extern "C" FILE* FOpenAndSleep(const char* filename, const char* mode)
{
   int tries = 0;
   FILE* ret = 0;
   while (!ret && (++tries < 51)) {
      ret = fopen(filename, mode);
      if (!ret && (tries < 50)) {
         if ((errno != EACCES) && (errno != EEXIST)) {
            return 0;
         }
         else {
            Sleep(200);
         }
      }
   }
   if (tries > 1) {
      printf("fopen slept for %g seconds until it succeeeded.\n", (tries - 1) / 5.0);
   }
   return ret;
}

//______________________________________________________________________________
#include "Reflex/Tools.h"

//______________________________________________________________________________
#ifdef fopen
#undef fopen
#endif // fopen

//______________________________________________________________________________
#define fopen(A,B) FOpenAndSleep((A),(B))

//______________________________________________________________________________
#endif // _WIN32

//______________________________________________________________________________
extern "C" {
   // dummy implementation
   void G__enable_wrappers(int) {};
   int G__wrappers_enabled() {return 1;};
}

//______________________________________________________________________________
using namespace Cint::Internal;

//______________________________________________________________________________
int(*G__p_ioctortype_handler)(const char*);

//______________________________________________________________________________
extern "C" void G__set_ioctortype_handler(int(*p2f)(const char*))
{
   // -- G__set_class_autoloading_callback.
   G__p_ioctortype_handler = p2f;
}

//______________________________________________________________________________
#define G__PROTECTEDACCESS 1
#define G__PRIVATEACCESS 2
static int G__privateaccess = 0;

//______________________________________________________________________________
//
// CAUTION:
//
//   Following macro G__BUILTIN must not be defined at normal cint
//   installation. This macro must be deleted only when you generate following
//   source files.
//       src/libstrm.cxx  in lib/stream    'make'
//       src/gcc3strm.cxx in lib/gcc3strm  'make'
//       src/iccstrm.cxx  in lib/iccstrm   'make'
//       src/vcstrm.cxx   in lib/vcstream  'make' , 'make -f Makefileold'
//       src/vc7strm.cxx  in lib/vc7stream 'make'
//       src/bcstrm.cxx   in lib/bcstream  'make'
//       src/cbstrm.cpp   in lib/cbstream  'make'
//       src/sunstrm.cxx  in lib/snstream  'make'
//       src/kccstrm.cxx  (lib/kcc_work not included in the package)
//       src/stdstrct.c  in lib/stdstrct  'make'
//       src/Apiif.cxx   in src           'make -f Makeapi' , 'make -f Makeapiold'
//   g++ has a bug of distinguishing 'static operator delete(void* p)' in
//   different file scope. Deleting this macro will avoid this problem.
//   Note:  The C++ standard explicitly forbids a static operator delete
//          at global scope.

//#define G__BUILTIN

#if !defined(G__DECCXX) && !defined(G__BUILTIN) && !defined(__hpux)
#define G__DEFAULTASSIGNOPR
#endif // !G__DECCXX && !G__BUILTIN && !__hpux

#if !defined(G__DECCXX) && !defined(G__BUILTIN)
#define G__N_EXPLICITDESTRUCTOR
#endif // ! G__DECCXX && !G__BUILTIN

#ifndef G__N_EXPLICITDESTRUCTOR

#ifdef G__P2FCAST
#undef G__P2FCAST
#endif // G__P2FCAST

#ifdef G__P2FDECL
#undef G__P2FDECL
#endif // G__P2FDECL

#endif // G__N_EXPLICITDESTRUCTOR

//______________________________________________________________________________
//
//  If this is Windows-NT/95, create G__PROJNAME.DEF file
//
#if  defined(G__WIN32) && !defined(G__BORLAND)
#define G__GENWINDEF
#endif

//______________________________________________________________________________
//
//  Following static variables must be protected by semaphore for multi-threading.
//

// Class to store several variables
// Due to recursion problems these variables are not global anymore
   class G__IncSetupStack {

   public:

      Reflex::Scope G__incset_p_ifunc;
      Reflex::Scope G__incset_tagnum;
      Reflex::Member G__incset_func_now;
      Reflex::Scope G__incset_p_local;
      int G__incset_def_struct_member;
      Reflex::Scope  G__incset_tagdefining;
      Reflex::Scope  G__incset_def_tagnum;
      char *G__incset_globalvarpointer;
      int G__incset_var_type;
      Reflex::Scope G__incset_typenum;
      int G__incset_static_alloc;
      int G__incset_access;
 
  };
 

//G__stack_instance
//Several problems in the mefunc_setup recursive calling, due to global variables using,
//was introduced. Now a stack stores the variables (G__IncSetupStack class)
//Each memfunc_setup call will have its own copy of the variables in the stack
//RETURN: This function return a pointer to the static variable stack
   std::stack<G__IncSetupStack>* G__stack_instance()
   {

      // Variables Stack
      static std::stack<G__IncSetupStack>* G__stack = 0;

      // If the stack has not been initialized yet
      if (G__stack==0)
         G__stack = new std::stack<G__IncSetupStack>();

      return G__stack;

   }

  // Supress Stub Functions
  static int G__suppress_methods = 0;
  static int G__nestedclass = 0;
  static int G__nestedtypedef = 0;
  static int G__store_asm_noverflow;
  static int G__store_no_exec_compile;
  static int G__store_asm_exec;
  static int G__extra_inc_n = 0;
  static char** G__extra_include = 0; /*  [G__MAXFILENAME] = NULL;  */
 

//______________________________________________________________________________
//
//  G__CurrentCall
//

static int s_CurrentCallType;
static void* s_CurrentCall;
static int s_CurrentIndex;

//______________________________________________________________________________
extern "C" void G__CurrentCall(int call_type, void* call_ifunc, long* ifunc_idx)
{
   switch (call_type) {
      case G__NOP:
         s_CurrentCallType = call_type;
         s_CurrentCall     = 0;
         s_CurrentIndex    = -1;
         break;
      case G__SETMEMFUNCENV:
         s_CurrentCallType = call_type;
         s_CurrentCall     = call_ifunc;
         s_CurrentIndex    = *ifunc_idx;
         break;
      case G__DELETEFREE:
         s_CurrentCallType = call_type;
         s_CurrentCall     = call_ifunc;
         s_CurrentIndex    = *ifunc_idx;
         break;
      case G__RECMEMFUNCENV:
         if (call_ifunc) *(void**)call_ifunc = s_CurrentCall;
         if (ifunc_idx)  *ifunc_idx = s_CurrentIndex;
         break;
      case G__RETURN:
         if (call_ifunc) *(void**)call_ifunc = 0;
         if (ifunc_idx)  *ifunc_idx  = s_CurrentCallType;
         break;
   }
}

//______________________________________________________________________________
//
//  Checking private constructor
//

#define G__CTORDTOR_UNINITIALIZED     0x00000000
#define G__CTORDTOR_PRIVATECTOR       0x00000001
#define G__CTORDTOR_NOPRIVATECTOR     0x00000002
#define G__CTORDTOR_PRIVATECOPYCTOR   0x00000010
#define G__CTORDTOR_NOPRIVATECOPYCTOR 0x00000020
#define G__CTORDTOR_PRIVATEDTOR       0x00000100
#define G__CTORDTOR_NOPRIVATEDTOR     0x00000200
#define G__CTORDTOR_PRIVATEASSIGN     0x00001000
#define G__CTORDTOR_NOPRIVATEASSIGN   0x00002000

static int* G__ctordtor_status;


//______________________________________________________________________________
//
//  G__cpplink file name
//

static char* G__CPPLINK_H;
static char* G__CPPLINK_C;

static char* G__CLINK_H;
static char* G__CLINK_C;

#ifdef G__GENWINDEF
static char* G__WINDEF;
static int G__nexports;
static FILE* G__WINDEFfp;
static int G__isDLL;
static char G__CINTLIBNAME[10] = "LIBCINT";
#endif // G__GENWINDEF

#define G__MAXDLLNAMEBUF 512

static char G__PROJNAME[G__MAXNAME];
static char G__DLLID[G__MAXDLLNAMEBUF];
static char* G__INITFUNC;

static char G__NEWID[G__MAXDLLNAMEBUF];

//______________________________________________________________________________
#ifdef G__BORLANDCC5
static void G__ctordtor_initialize(void);
static void G__fileerror(char* fname);
static void G__ctordtor_destruct(void);
void G__cpplink_protected_stub(FILE *fp, FILE *hfp);
void G__gen_cppheader(char *headerfilein);
static void G__gen_headermessage(FILE *fp, char *fname);
void G__add_macro(char *macroin);
int G__isnonpublicnew(int tagnum);
static int G__isprotecteddestructoronelevel(int tagnum);
void  G__if_ary_union(FILE *fp, int ifn, struct G__ifunc_table *ifunc);
const char *G__mark_linked_tagnum(int tagnum);
static int G__isprivateconstructorifunc(int tagnum, int iscopy);
static int G__isprivateconstructorvar(int tagnum, int iscopy);
static int G__isprivatedestructorifunc(int tagnum);
static int G__isprivatedestructorvar(int tagnum);
static int G__isprivateassignoprifunc(int tagnum);
static int G__isprivateassignoprvar(int tagnum);
void G__cppif_gendefault(FILE *fp, FILE *hfp, int tagnum, int ifn, struct G__ifunc_table *ifunc, int isconstructor, int iscopyconstructor, int isdestructor, int isassignmentoperator, int isnonpublicnew);
static char* G__vbo_funcname(int tagnum, int basetagnum, int basen);
static void G__declaretruep2f(FILE *fp, struct G__ifunc_table *ifunc, int j);
static void G__printtruep2f(FILE *fp, struct G__ifunc_table *ifunc, int j);
int G__tagtable_setup(int tagnum, int size, int cpplink, int isabstract, char *comment, G__incsetup setup_memvar, G__incsetup setup_memfunc);
int G__tag_memfunc_setup(int tagnum);
int G__memfunc_setup(char *funcname, int hash, int(*funcp)(), int type, int tagnum, int typenum, int reftype, int para_nu, int ansi, int accessin, int isconst, char *paras, char *comment
#ifdef G__TRUEP2F
                     , void* truep2f, int isvirtual
#endif // G__TRUEP2F
                    );
int G__memfunc_next(void);
static void G__pragmalinkenum(int tagnum, int globalcomp);
void G__incsetup_memvar(int tagnum);
void G__incsetup_memfunc(int tagnum);
#endif // G__BORLANDCC5

//______________________________________________________________________________
extern "C" void G__check_setup_version(int version, const char* func)
{
   // -- Verify CINT and DLL version.
   G__init_globals();
   if (version > G__ACCEPTDLLREV_UPTO || version < G__ACCEPTDLLREV_FROM) {
      fprintf(G__sout, "\n\
              !!!!!!!!!!!!!!   W A R N I N G    !!!!!!!!!!!!!\n\n\
              The internal data structures have been changed.\n\
              Please regenerate and recompile your dictionary which\n\
              contains the definition \"%s\"\n\
              using CINT version %s.\n\
              your dictionary=%d. This version accepts=%d-%d\n\
              and creates %d\n\n\
              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n",
              func, G__cint_version(), version
              , G__ACCEPTDLLREV_FROM
              , G__ACCEPTDLLREV_UPTO
              , G__CREATEDLLREV
             );
      exit(1);
   }
   G__store_asm_noverflow = G__asm_noverflow;
   G__store_no_exec_compile = G__no_exec_compile;
   G__store_asm_exec = G__asm_exec;
   G__abortbytecode();
   G__no_exec_compile = 0;
   G__asm_exec = 0;
}

//______________________________________________________________________________
static void G__fileerror(char* fname)
{
   char *buf = (char*)malloc(strlen(fname) + 80);
   sprintf(buf, "Error opening %s", fname);
   perror(buf);
   exit(2);
}

//______________________________________________________________________________
const std::string Cint::Internal::G__fulltypename(::Reflex::Type typenum)
{
   if (!typenum) return("");
   return typenum.Name(::Reflex::SCOPED);
}

//______________________________________________________________________________
static void G__ctordtor_initialize()
{
   int i;
   G__ctordtor_status = (int*)malloc(sizeof(int) * (G__struct.alltag + 1));
   for (i = 0;i < G__struct.alltag;i++) {
      /* If link for this class is turned off but one or more member functions
      * are explicitly turned on, set G__ONLYMETHODLINK flag for the class */
      if (G__NOLINK == G__struct.globalcomp[i]) {
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(i);
         for (::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
               ifunc != scope.FunctionMember_End();
               ++ifunc) {
            if (G__METHODLINK == G__get_funcproperties(*ifunc)->globalcomp) {
               G__struct.globalcomp[i] = G__ONLYMETHODLINK;
            }
         }
      }
      G__ctordtor_status[i] = G__CTORDTOR_UNINITIALIZED;
   }
}

//______________________________________________________________________________
static void G__ctordtor_destruct()
{
   if (G__ctordtor_status) free(G__ctordtor_status);
}

//______________________________________________________________________________
void Cint::Internal::G__gen_clink()
{
   // -- Generate C++ interface routine source file.
#ifndef G__SMALLOBJECT
   /*
   *   include header files
   *   struct G__tagtable G__struct;
   *   struct G__typedef G__newtype;
   *   struct G__var_array G__struct.*memvar;
   *   struct G__var_array G__global;
   *   struct G__ifunc_table G__ifunc;
   */
   FILE *fp;
   FILE *hfp;

   G__ctordtor_initialize();

   fp = fopen(G__CLINK_C, "a");
   if (!fp) G__fileerror(G__CLINK_C);
   fprintf(fp, "  G__c_reset_tagtable%s();\n", G__DLLID);
   fprintf(fp, "}\n");

   hfp = fopen(G__CLINK_H, "a");
   if (!hfp) G__fileerror(G__CLINK_H);

#ifdef G__BUILTIN
   fprintf(fp, "#include \"dllrev.h\"\n");
   fprintf(fp, "int G__c_dllrev%s() { return(G__CREATEDLLREV); }\n", G__DLLID);
#else
   fprintf(fp, "int G__c_dllrev%s() { return(%d); }\n", G__DLLID, G__CREATEDLLREV);
#endif

   G__cppif_func(fp, hfp);
   G__cppstub_func(fp);

   G__cpplink_typetable(fp, hfp);
   G__cpplink_memvar(fp);
   G__cpplink_global(fp);
   G__cpplink_func(fp);
   G__cpplink_tagtable(fp, hfp);
   fprintf(fp, "void G__c_setup%s() {\n", G__DLLID);
#ifdef G__BUILTIN
   fprintf(fp, "  G__check_setup_version(G__CREATEDLLREV,\"G__c_setup%s()\");\n",
           G__DLLID);
#else
   fprintf(fp, "  G__check_setup_version(%d,\"G__c_setup%s()\");\n",
           G__CREATEDLLREV, G__DLLID);
#endif
   fprintf(fp, "  G__set_c_environment%s();\n", G__DLLID);
   fprintf(fp, "  G__c_setup_tagtable%s();\n\n", G__DLLID);
   fprintf(fp, "  G__c_setup_typetable%s();\n\n", G__DLLID);
   fprintf(fp, "  G__c_setup_memvar%s();\n\n", G__DLLID);
   fprintf(fp, "  G__c_setup_global%s();\n", G__DLLID);
   fprintf(fp, "  G__c_setup_func%s();\n", G__DLLID);
   fprintf(fp, "  return;\n");
   fprintf(fp, "}\n");
   fclose(fp);
   fclose(hfp);
   G__ctordtor_destruct();
#endif // G__SMALLOBJECT
   // --
}

static void AllocateRootSpecial( int tagnum )
{
  if(G__struct.rootspecial[tagnum]) return;
  G__struct.rootspecial[tagnum]
    =(struct G__RootSpecial*)malloc(sizeof(struct G__RootSpecial));
  G__struct.rootspecial[tagnum]->deffile=(char*)NULL;
  G__struct.rootspecial[tagnum]->impfile=(char*)NULL;
  G__struct.rootspecial[tagnum]->defline=0;
  G__struct.rootspecial[tagnum]->impline=0;
  G__struct.rootspecial[tagnum]->version=0;
  G__struct.rootspecial[tagnum]->instancecount=0;
  G__struct.rootspecial[tagnum]->heapinstancecount=0;
  G__struct.rootspecial[tagnum]->defaultconstructor = 0;
  // G__struct.rootspecial[tagnum]->defaultconstructorifunc = 0;
}

#ifdef G__ROOT
//______________________________________________________________________________
static void G__cpp_initialize(FILE* fp)
{
   // Do not do this for cint/src/Apiif.cxx and cint/src/Apiifold.cxx.
   if (!strcmp(G__DLLID, "G__API")) {
      return;
   }
   fprintf(fp, "class G__cpp_setup_init%s {\n", G__DLLID);
   fprintf(fp, "  public:\n");
   if (G__DLLID[0]) {
      fprintf(fp, "    G__cpp_setup_init%s() { G__add_setup_func(\"%s\",(G__incsetup)(&G__cpp_setup%s)); G__call_setup_funcs(); }\n", G__DLLID, G__DLLID, G__DLLID);
      fprintf(fp, "   ~G__cpp_setup_init%s() { G__remove_setup_func(\"%s\"); }\n", G__DLLID, G__DLLID);
   }
   else {
      fprintf(fp, "    G__cpp_setup_init() { G__add_setup_func(\"G__Default\",(G__incsetup)(&G__cpp_setup)); }\n");
      fprintf(fp, "   ~G__cpp_setup_init() { G__remove_setup_func(\"G__Default\"); }\n");
   }
   fprintf(fp, "};\n");
   fprintf(fp, "G__cpp_setup_init%s G__cpp_setup_initializer%s;\n\n", G__DLLID, G__DLLID);
}
#endif // G__ROOT

//______________________________________________________________________________
void Cint::Internal::G__gen_cpplink()
{
   // -- Generate C++ interface routine source file.
#ifndef G__SMALLOBJECT
   /*
   *   include header files
   *   struct G__tagtable G__struct;
   *   struct G__typedef G__newtype;
   *   struct G__var_array G__struct.*memvar;
   *   struct G__ifunc_table G__struct.*memfunc;
   *   struct G__var_array G__global;
   *   struct G__ifunc_table G__ifunc;
   */
   FILE *fp;
   FILE *hfp;

   G__ctordtor_initialize();

   fp = fopen(G__CPPLINK_C, "a");
   if (!fp) G__fileerror(G__CPPLINK_C);
   fprintf(fp, "  G__cpp_reset_tagtable%s();\n", G__DLLID);
   fprintf(fp, "}\n");

   hfp = fopen(G__CPPLINK_H, "a");
   if (!hfp) G__fileerror(G__CPPLINK_H);

   {
      int algoflag = 0;
      int filen;
      char *fname;
      int lenstl;
      char *sysstl;
      G__getcintsysdir();
      sysstl = (char*)malloc(strlen(G__cintsysdir) + 20);
      sprintf(sysstl,"%s%s%s%sstl%s",G__cintsysdir,G__psep,G__CFG_COREVERSION,G__psep,G__psep);
      lenstl = strlen(sysstl);
      for (filen = 0;filen < G__nfile;filen++) {
         fname = G__srcfile[filen].filename;
         if (strncmp(fname, sysstl, lenstl) == 0) fname += lenstl;
         if (strcmp(fname, "vector") == 0 || strcmp(fname, "list") == 0 ||
               strcmp(fname, "deque") == 0 || strcmp(fname, "map") == 0 ||
               strcmp(fname, "multimap") == 0 || strcmp(fname, "set") == 0 ||
               strcmp(fname, "multiset") == 0 || strcmp(fname, "stack") == 0 ||
               strcmp(fname, "queue") == 0 || strcmp(fname, "climits") == 0 ||
               strcmp(fname, "valarray") == 0) {
            algoflag |= 1;
         }
         if (strcmp(fname, "vector.h") == 0 || strcmp(fname, "list.h") == 0 ||
               strcmp(fname, "deque.h") == 0 || strcmp(fname, "map.h") == 0 ||
               strcmp(fname, "multimap.h") == 0 || strcmp(fname, "set.h") == 0 ||
               strcmp(fname, "multiset.h") == 0 || strcmp(fname, "stack.h") == 0 ||
               strcmp(fname, "queue.h") == 0) {
            algoflag |= 2;
         }
      }
      if (algoflag&1) {
         fprintf(hfp, "#include <algorithm>\n");
         if (G__ignore_stdnamespace) {
            /* fprintf(hfp,"#ifndef __hpux\n"); */
            fprintf(hfp, "namespace std { }\n");
            fprintf(hfp, "using namespace std;\n");
            /* fprintf(hfp,"#endif\n"); */
         }
      }
      else if (algoflag&2) fprintf(hfp, "#include <algorithm.h>\n");
      if (sysstl) free((void*)sysstl);
   }

#if !defined(G__ROOT) || defined(G__OLDIMPLEMENTATION1817)
   if (G__CPPLINK == G__globalcomp && -1 != G__defined_tagname("G__longlong", 2)) {
#if defined(__hpux) && !defined(G__ROOT)
      G__getcintsysdir();
      fprintf(hfp, "\n#include \"%s/%s/lib/longlong/longlong.h\"\n", G__cintsysdir, G__CFG_COREVERSION);
#else
      fprintf(hfp, "\n#include \"%s/lib/longlong/longlong.h\"\n", G__CFG_COREVERSION);
#endif
   }
#endif /* G__ROOT */

   fprintf(fp, "#include <new>\n");

#ifdef G__BUILTIN
   fprintf(fp, "#include \"dllrev.h\"\n");
   fprintf(fp, "extern \"C\" int G__cpp_dllrev%s() { return(G__CREATEDLLREV); }\n", G__DLLID);
#else
   fprintf(fp, "extern \"C\" int G__cpp_dllrev%s() { return(%d); }\n", G__DLLID, G__CREATEDLLREV);
#endif

   fprintf(hfp, "\n#ifndef G__MEMFUNCBODY\n");

   if (!G__suppress_methods) {
      G__cppif_memfunc(fp, hfp);
   }

   G__cppif_func(fp, hfp);

   if (!G__suppress_methods) {
      G__cppstub_memfunc(fp);
   }

   G__cppstub_func(fp);

   fprintf(hfp, "#endif\n\n");

   G__cppif_p2memfunc(fp);

#ifdef G__VIRTUALBASE
   G__cppif_inheritance(fp);
#endif

   G__cpplink_inheritance(fp);
   G__cpplink_typetable(fp, hfp);
   G__cpplink_memvar(fp);

   if (!G__suppress_methods) {
      G__cpplink_memfunc(fp);
   }

   G__cpplink_global(fp);
   G__cpplink_func(fp);
   G__cpplink_tagtable(fp, hfp);
   fprintf(fp, "extern \"C\" void G__cpp_setup%s(void) {\n", G__DLLID);
#ifdef G__BUILTIN
   fprintf(fp, "  G__check_setup_version(G__CREATEDLLREV,\"G__cpp_setup%s()\");\n",
           G__DLLID);
#else
   fprintf(fp, "  G__check_setup_version(%d,\"G__cpp_setup%s()\");\n",
           G__CREATEDLLREV, G__DLLID);
#endif
   fprintf(fp, "  G__set_cpp_environment%s();\n", G__DLLID);
   fprintf(fp, "  G__cpp_setup_tagtable%s();\n\n", G__DLLID);
   fprintf(fp, "  G__cpp_setup_inheritance%s();\n\n", G__DLLID);
   fprintf(fp, "  G__cpp_setup_typetable%s();\n\n", G__DLLID);
   fprintf(fp, "  G__cpp_setup_memvar%s();\n\n", G__DLLID);
   if (!G__suppress_methods)
      fprintf(fp, "  G__cpp_setup_memfunc%s();\n", G__DLLID);
   fprintf(fp, "  G__cpp_setup_global%s();\n", G__DLLID);
   fprintf(fp, "  G__cpp_setup_func%s();\n", G__DLLID);
   G__set_sizep2memfunc(fp);
   fprintf(fp, "  return;\n");
   fprintf(fp, "}\n");

#ifdef G__ROOT
   /* Only activated for ROOT at this moment. Need to come back */
   G__cpp_initialize(fp);
#endif

   fclose(fp);
   fclose(hfp);
#ifdef G__GENWINDEF
   fprintf(G__WINDEFfp, "\n");
   fclose(G__WINDEFfp);
#endif

   G__ctordtor_destruct();
#endif // G__SMALLOBJECT
   // --
}

//______________________________________________________________________________
int Cint::Internal::G__cleardictfile(int flag)
{
   if (EXIT_SUCCESS != flag) {
      G__fprinterr(G__serr, "!!!Removing ");
      if (G__CPPLINK_C) {
         remove(G__CPPLINK_C);
         G__fprinterr(G__serr, "%s ", G__CPPLINK_C);
      }
      if (G__CPPLINK_H) {
         remove(G__CPPLINK_H);
         G__fprinterr(G__serr, "%s ", G__CPPLINK_H);
      }
      if (G__CLINK_C) {
         remove(G__CLINK_C);
         G__fprinterr(G__serr, "%s ", G__CLINK_C);
      }
      if (G__CLINK_H) {
         remove(G__CLINK_H);
         G__fprinterr(G__serr, "%s ", G__CLINK_H);
      }
      G__fprinterr(G__serr, "!!!\n");
   }
#ifdef G__GENWINDEF
   if (G__WINDEF) {
      /* unlink(G__WINDEF); */
      free(G__WINDEF);
   }
#endif
   if (G__CPPLINK_H) free(G__CPPLINK_H);
   if (G__CPPLINK_C) free(G__CPPLINK_C);
   if (G__CLINK_H) free(G__CLINK_H);
   if (G__CLINK_C) free(G__CLINK_C);

#ifdef G__GENWINDEF
   G__WINDEF = (char*)NULL;
#endif
   G__CPPLINK_C = (char*)NULL;
   G__CPPLINK_H = (char*)NULL;
   G__CLINK_C = (char*)NULL;
   G__CLINK_H = (char*)NULL;
   return(0);
}

//______________________________________________________________________________
void Cint::Internal::G__clink_header(FILE* fp)
{
   fprintf(fp, "#include <stddef.h>\n");
   fprintf(fp, "#include <stdio.h>\n");
   fprintf(fp, "#include <stdlib.h>\n");
   fprintf(fp, "#include <math.h>\n");
   fprintf(fp, "#include <string.h>\n");
   if (G__multithreadlibcint)
      fprintf(fp, "#define G__MULTITHREADLIBCINTC\n");
   fprintf(fp, "#define G__ANSIHEADER\n");
#if defined(G__VAARG_COPYFUNC) || !defined(G__OLDIMPLEMENTATION1530)
   fprintf(fp, "#define G__DICTIONARY\n");
   fprintf(fp,"#define G__PRIVATE_GVALUE\n");
#endif
#if defined(__hpux) && !defined(G__ROOT)
   G__getcintsysdir();
   fprintf(fp, "#include \"%s/%s/inc/G__ci.h\"\n", G__cintsysdir, G__CFG_COREVERSION);
#elif defined(G__ROOT)
   //fprintf(fp,"#include \"cint7/G__ci.h\"\n");
   fprintf(fp, "#include \"G__ci.h\"\n");
#else
   fprintf(fp, "#include \"G__ci.h\"\n");
#endif

   if (G__multithreadlibcint)
      fprintf(fp, "#undef G__MULTITHREADLIBCINTC\n");

#if defined(G__BORLAND) || defined(G__VISUAL)
   fprintf(fp, "extern G__DLLEXPORT int G__c_dllrev%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__set_c_environment%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__c_setup_tagtable%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__c_setup_typetable%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__c_setup_memvar%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__c_setup_global%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__c_setup_func%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__c_setup%s();\n", G__DLLID);
   if (G__multithreadlibcint) {
      fprintf(fp, "extern G__DLLEXPORT void G__SetCCintApiPointers(void* a[G__NUMBER_OF_API_FUNCTIONS]);\n");
   }
#else
   fprintf(fp, "extern void G__c_setup_tagtable%s();\n", G__DLLID);
   fprintf(fp, "extern void G__c_setup_typetable%s();\n", G__DLLID);
   fprintf(fp, "extern void G__c_setup_memvar%s();\n", G__DLLID);
   fprintf(fp, "extern void G__c_setup_global%s();\n", G__DLLID);
   fprintf(fp, "extern void G__c_setup_func%s();\n", G__DLLID);
   fprintf(fp, "extern void G__set_c_environment%s();\n", G__DLLID);
   if (G__multithreadlibcint) {
      fprintf(fp, "extern void G__SetCCintApiPointers(void* a[G__NUMBER_OF_API_FUNCTIONS]);\n");
   }
#endif


   fprintf(fp, "\n");
   fprintf(fp, "\n");
}

//______________________________________________________________________________
void Cint::Internal::G__cpplink_header(FILE* fp)
{
   fprintf(fp, "#include <stddef.h>\n");
   fprintf(fp, "#include <stdio.h>\n");
   fprintf(fp, "#include <stdlib.h>\n");
   fprintf(fp, "#include <math.h>\n");
   fprintf(fp, "#include <string.h>\n");
   if (G__multithreadlibcint)
      fprintf(fp, "#define G__MULTITHREADLIBCINTCPP\n");
   fprintf(fp, "#define G__ANSIHEADER\n");
#if defined(G__VAARG_COPYFUNC) || !defined(G__OLDIMPLEMENTATION1530)
   fprintf(fp, "#define G__DICTIONARY\n");
#endif
#if defined(__hpux) && !defined(G__ROOT)
   G__getcintsysdir();
   fprintf(fp, "#include \"%s/%s/inc/G__ci.h\"\n", G__cintsysdir, G__CFG_COREVERSION);
#elif defined(G__ROOT)
   //fprintf(fp,"#include \"cint7/G__ci.h\"\n");
   fprintf(fp, "#include \"G__ci.h\"\n");
#else
   fprintf(fp, "#include \"G__ci.h\"\n");
#endif

   if (G__multithreadlibcint)
      fprintf(fp, "#undef G__MULTITHREADLIBCINTCPP\n");

   fprintf(fp, "extern \"C\" {\n");

#if defined(G__BORLAND) || defined(G__VISUAL)
   fprintf(fp, "extern G__DLLEXPORT int G__cpp_dllrev%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__set_cpp_environment%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__cpp_setup_tagtable%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__cpp_setup_inheritance%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__cpp_setup_typetable%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__cpp_setup_memvar%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__cpp_setup_global%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__cpp_setup_memfunc%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__cpp_setup_func%s();\n", G__DLLID);
   fprintf(fp, "extern G__DLLEXPORT void G__cpp_setup%s();\n", G__DLLID);
   if (G__multithreadlibcint) {
      fprintf(fp, "extern G__DLLEXPORT void G__SetCppCintApiPointers(void* a[G__NUMBER_OF_API_FUNCTIONS]);\n");
   }
#else
   fprintf(fp, "extern void G__cpp_setup_tagtable%s();\n", G__DLLID);
   fprintf(fp, "extern void G__cpp_setup_inheritance%s();\n", G__DLLID);
   fprintf(fp, "extern void G__cpp_setup_typetable%s();\n", G__DLLID);
   fprintf(fp, "extern void G__cpp_setup_memvar%s();\n", G__DLLID);
   fprintf(fp, "extern void G__cpp_setup_global%s();\n", G__DLLID);
   fprintf(fp, "extern void G__cpp_setup_memfunc%s();\n", G__DLLID);
   fprintf(fp, "extern void G__cpp_setup_func%s();\n", G__DLLID);
   fprintf(fp, "extern void G__set_cpp_environment%s();\n", G__DLLID);
   if (G__multithreadlibcint) {
      fprintf(fp, "extern void G__SetCppCintApiPointers(void* a[G__NUMBER_OF_API_FUNCTIONS]);\n");
   }
#endif
   fprintf(fp, "}\n");
   fprintf(fp, "\n");
   fprintf(fp, "\n");
}

//______________________________________________________________________________
extern "C" char *G__map_cpp_name(const char* in)
{
   static char out[G__MAXNAME*6];
   int i = 0, j = 0, c;
   while ((c = in[i])) {
      switch (c) {
         case '+':
            strcpy(out + j, "pL");
            j += 2;
            break;
         case '-':
            strcpy(out + j, "mI");
            j += 2;
            break;
         case '*':
            strcpy(out + j, "mU");
            j += 2;
            break;
         case '/':
            strcpy(out + j, "dI");
            j += 2;
            break;
         case '&':
            strcpy(out + j, "aN");
            j += 2;
            break;
         case '%':
            strcpy(out + j, "pE");
            j += 2;
            break;
         case '|':
            strcpy(out + j, "oR");
            j += 2;
            break;
         case '^':
            strcpy(out + j, "hA");
            j += 2;
            break;
         case '>':
            strcpy(out + j, "gR");
            j += 2;
            break;
         case '<':
            strcpy(out + j, "lE");
            j += 2;
            break;
         case '=':
            strcpy(out + j, "eQ");
            j += 2;
            break;
         case '~':
            strcpy(out + j, "wA");
            j += 2;
            break;
         case '.':
            strcpy(out + j, "dO");
            j += 2;
            break;
         case '(':
            strcpy(out + j, "oP");
            j += 2;
            break;
         case ')':
            strcpy(out + j, "cP");
            j += 2;
            break;
         case '[':
            strcpy(out + j, "oB");
            j += 2;
            break;
         case ']':
            strcpy(out + j, "cB");
            j += 2;
            break;
         case '!':
            strcpy(out + j, "nO");
            j += 2;
            break;
         case ',':
            strcpy(out + j, "cO");
            j += 2;
            break;
         case '$':
            strcpy(out + j, "dA");
            j += 2;
            break;
         case ' ':
            strcpy(out + j, "sP");
            j += 2;
            break;
         case ':':
            strcpy(out + j, "cL");
            j += 2;
            break;
         case '"':
            strcpy(out + j, "dQ");
            j += 2;
            break;
         case '@':
            strcpy(out + j, "aT");
            j += 2;
            break;
         case '\'':
            strcpy(out + j, "sQ");
            j += 2;
            break;
         case '\\':
            strcpy(out + j, "fI");
            j += 2;
            break;
         default:
            out[j++] = c;
            break;
      }
      ++i;
   }
   out[j] = '\0';
   return(out);
}

//______________________________________________________________________________
static char *G__map_cpp_funcname(const ::Reflex::Member& ifunc)
{
   // Mapping between C++ function and parameter name to cint interface
   // function name. This routine handles mapping of function and operator
   // overloading in linked C++ object.
   return G__map_cpp_funcname(G__get_tagnum(ifunc.DeclaringScope()), (char*)0x0, (long)ifunc.Id(), 0);
}

//______________________________________________________________________________
char* Cint::Internal::G__map_cpp_funcname(int tagnum, char* /*funcname*/, long ifn, int page)
{
   static char mapped_name[G__MAXNAME];
   const char *dllid;

   if (G__DLLID[0]) dllid = G__DLLID;
   else if (G__PROJNAME[0]) dllid = G__PROJNAME;
   else dllid = "";

   if (-1 == tagnum) {
      sprintf(mapped_name, "G__%s__%ld_%d", G__map_cpp_name(dllid), ifn, page);
   }
   else {
      sprintf(mapped_name, "G__%s_%d_%ld_%d", G__map_cpp_name(dllid), tagnum, ifn, page);
   }
   return mapped_name;
}

//______________________________________________________________________________
static void G__cpplink_protected_stub_ctor(int tagnum, FILE* hfp)
{
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
   for (::Reflex::Member_Iterator func_mbr_iter = scope.FunctionMember_Begin(); func_mbr_iter != scope.FunctionMember_End(); ++func_mbr_iter) {
      if (func_mbr_iter->Name() == G__struct.name[tagnum]) {
         unsigned int i = 0;
         fprintf(hfp, "  %s_PR(", G__get_link_tagname(tagnum));
         for (i = 0; i < func_mbr_iter->FunctionParameterSize(); ++i) {
            if (i) {
               fprintf(hfp, ",");
            }
            ::Reflex::Type param_type = func_mbr_iter->TypeOf().FunctionParameterAt(i);
            char type = '\0';
            int local_tagnum = -1;
            int typenum = -1;
            int reftype = 0;
            int isconst = 0;
            G__get_cint5_type_tuple(param_type, &type, &local_tagnum, &typenum, &reftype, &isconst);
            fprintf(hfp, "%s a%d", G__type2string(type, tagnum, typenum, reftype, isconst), i);
         }
         fprintf(hfp, ")\n");
         fprintf(hfp, ": %s(" , G__fulltagname(tagnum, 1));
         for (i = 0; i < func_mbr_iter->FunctionParameterSize(); ++i) {
            if (i) {
               fprintf(hfp, ",");
            }
         }
         fprintf(hfp, "a%d", i);
      }
      fprintf(hfp, ") {}\n");
   }
}

//______________________________________________________________________________
static void G__cpplink_protected_stub(FILE* fp, FILE* hfp)
{
   int i;
   /* Create stub derived class for protected member access */
   fprintf(hfp, "\n/* STUB derived class for protected member access */\n");
   for (i = 0;i < G__struct.alltag;i++) {
      if (G__CPPLINK == G__struct.globalcomp[i] && G__struct.hash[i] &&
            G__struct.protectedaccess[i]) {
         unsigned int n;
         fprintf(hfp, "class %s_PR : public %s {\n"
                 , G__get_link_tagname(i), G__fulltagname(i, 1));
         fprintf(hfp, " public:\n");
         if (((G__struct.funcs[i]&G__HAS_XCONSTRUCTOR) ||
               (G__struct.funcs[i]&G__HAS_COPYCONSTRUCTOR))
               && 0 == (G__struct.funcs[i]&G__HAS_DEFAULTCONSTRUCTOR)) {
            G__cpplink_protected_stub_ctor(i, hfp);
         }
         /* member function */
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(i);

         for (::Reflex::Member_Iterator memfunc = scope.FunctionMember_Begin();
               memfunc != scope.FunctionMember_End();
               ++memfunc) {
            if ((G__test_access(*memfunc, G__PROTECTED)
                  || ((G__PRIVATEACCESS&G__struct.protectedaccess[i]) &&
                      G__test_access(*memfunc, G__PRIVATE))
                ) &&
                  memfunc->Name() == G__struct.name[i]) {
               fprintf(hfp, "  %s_PR(", G__get_link_tagname(i));
               if (0 == memfunc->FunctionParameterSize()) {
                  fprintf(hfp, "void");
               }
               else {
                  for (n = 0;n < memfunc->FunctionParameterSize();n++) {
                     if (n != 0) fprintf(hfp, ",");
                     fprintf(hfp, "%s G__%d", memfunc->TypeOf().FunctionParameterAt(n).Name(::Reflex::SCOPED).c_str()
                             , n);
                  }
               }
               fprintf(hfp, ") : %s(", G__fulltagname(i, 1));
               if (0 < memfunc->FunctionParameterSize()) {
                  for (n = 0;n < memfunc->FunctionParameterSize();n++) {
                     if (n != 0) fprintf(hfp, ",");
                     fprintf(hfp, "G__%d", n);
                  }
               }
               fprintf(hfp, ") { }\n");
            }
            if ((G__test_access(*memfunc, G__PROTECTED)
                  || ((G__PRIVATEACCESS&G__struct.protectedaccess[i]) &&
                      G__test_access(*memfunc, G__PRIVATE))
                ) &&
                  memfunc->Name() != G__struct.name[i] &&
                  '~' != memfunc->Name()[0]) {
               if (memfunc->IsStatic()) fprintf(hfp, "  static ");
               fprintf(hfp, "  %s G__PT_%s(", memfunc->TypeOf().ReturnType().Name(::Reflex::SCOPED).c_str()
                       , memfunc->Name().c_str());
               if (0 == memfunc->FunctionParameterSize()) {
                  fprintf(hfp, "void");
               }
               else {
                  for (n = 0;n < memfunc->FunctionParameterSize();n++) {
                     if (n != 0) fprintf(hfp, ",");
                     fprintf(hfp, "%s G__%d", memfunc->TypeOf().FunctionParameterAt(n).Name(::Reflex::SCOPED).c_str()
                             , n);
                  }
               }
               fprintf(hfp, ") {\n");
               if (::Reflex::Tools::FundamentalType(memfunc->TypeOf().ReturnType().FinalType()) !=::Reflex::kVOID) fprintf(hfp, "    return(");
               else                                                                                  fprintf(hfp, "    ");
               fprintf(hfp, "%s(", memfunc->Name().c_str());
               for (n = 0;n < memfunc->FunctionParameterSize();n++) {
                  if (n != 0) fprintf(hfp, ",");
                  fprintf(hfp, "G__%d", n);
               }
               fprintf(hfp, ")");
               if (::Reflex::Tools::FundamentalType(memfunc->TypeOf().ReturnType().FinalType()) !=::Reflex::kVOID) fprintf(hfp, ")");
               fprintf(hfp, ";\n");
               fprintf(hfp, "  }\n");
            }
         }
         /* data member */
         for (::Reflex::Member_Iterator memvar = scope.DataMember_Begin();
               memvar != scope.DataMember_End();
               ++memvar) {
            if (G__test_access(*memvar, G__PROTECTED)) {
               if (G__get_properties(*memvar)->statictype == G__AUTO)
                  fprintf(hfp, "  long G__OS_%s(){return((long)(&%s)-(long)this);}\n"
                          , memvar->Name().c_str(), memvar->Name().c_str());
               else
                  fprintf(hfp, "  static long G__OS_%s(){return((long)(&%s));}\n"
                          , memvar->Name().c_str(), memvar->Name().c_str());
            }
         }
         fprintf(hfp, "};\n");
      }
   }
   fprintf(fp, "\n");
}

//______________________________________________________________________________
void Cint::Internal::G__cpplink_linked_taginfo(FILE* fp, FILE* hfp)
{
   int i;
   G__StrBuf buf_sb(G__MAXFILENAME);
   char *buf = buf_sb;
   FILE* pfp;
   if (G__privateaccess) {
      char *xp;
      strcpy(buf, G__CPPLINK_H);
      xp = strstr(buf, ".h");
      if (xp) strcpy(xp, "P.h");
      pfp = fopen(buf, "r");
      if (pfp) {
         fclose(pfp);
         remove(buf);
      }
      pfp = fopen(buf, "w");
      fprintf(pfp, "#ifdef PrivateAccess\n");
      fprintf(pfp, "#undef PrivateAccess\n");
      fprintf(pfp, "#endif\n");
      fprintf(pfp, "#define PrivateAccess(name) PrivateAccess_##name\n");
      fclose(pfp);
   }
   fprintf(fp, "/* Setup class/struct taginfo */\n");
   for (i = 0;i < G__struct.alltag;i++) {
      if ((G__NOLINK > G__struct.globalcomp[i]
            || G__ONLYMETHODLINK == G__struct.globalcomp[i]
          ) &&
            (
               (G__struct.hash[i] || 0 == G__struct.name[i][0])
               || -1 != G__struct.parent_tagnum[i])) {
         fprintf(fp, "G__linked_taginfo %s = { \"%s\" , %d , -1 };\n"
                 , G__get_link_tagname(i), G__fulltagname(i, 0), G__struct.type[i]);
         fprintf(hfp, "extern G__linked_taginfo %s;\n", G__get_link_tagname(i));
         if (G__privateaccess) {
            pfp = fopen(buf, "a");
            if (pfp) {
               if (G__PRIVATEACCESS&G__struct.protectedaccess[i])
                  fprintf(pfp, "#define PrivateAccess_%s  friend class %s_PR;\n"
                          , G__fulltagname(i, 1), G__get_link_tagname(i));
               else
                  fprintf(pfp, "#define PrivateAccess_%s \n", G__fulltagname(i, 1));
               fclose(pfp);
            }
         }
      }
   }
   fprintf(fp, "\n");

   fprintf(fp, "/* Reset class/struct taginfo */\n");
   switch (G__globalcomp) {
      case G__CLINK:
         fprintf(fp, "void G__c_reset_tagtable%s() {\n", G__DLLID);
         break;
      case G__CPPLINK:
      default:
         fprintf(fp, "extern \"C\" void G__cpp_reset_tagtable%s() {\n", G__DLLID);
         break;
   }

   for (i = 0;i < G__struct.alltag;i++) {
      if ((G__NOLINK > G__struct.globalcomp[i]
            || G__ONLYMETHODLINK == G__struct.globalcomp[i]
          ) &&
            (
               (G__struct.hash[i] || 0 == G__struct.name[i][0])
               || -1 != G__struct.parent_tagnum[i])) {
         fprintf(fp, "  %s.tagnum = -1 ;\n", G__get_link_tagname(i));
      }
   }

   fprintf(fp, "}\n\n");

   G__cpplink_protected_stub(fp, hfp);
}

//______________________________________________________________________________
extern "C" int G__get_linked_tagnum(G__linked_taginfo* p)
{
   // Setup and return tagnum.
   if (!p) {
      return -1;
   }
   if (p->tagnum == -1) {
      p->tagnum = G__search_tagname(p->tagname, p->tagtype);
      if (G__UserSpecificUpdateClassInfo) {
         char* varp = G__globalvarpointer;
         G__globalvarpointer = G__PVOID;
         (*G__UserSpecificUpdateClassInfo)((char*) p->tagname, p->tagnum);
         G__globalvarpointer = varp;
      }
   }
   return p->tagnum;
}

//______________________________________________________________________________
int G__get_linked_tagnum_fwd(G__linked_taginfo* p)
{
   // Setup and return tagnum; no autoloading.
   if (!p) {
      return -1;
   }
   int type = p->tagtype;
   p->tagtype = toupper(type);
   int ret = G__get_linked_tagnum(p);
   p->tagtype = type;
   return ret;
}

//______________________________________________________________________________
char* Cint::Internal::G__get_link_tagname(int tagnum)
{
   // -- Setup and return tagnum.
   static char mapped_tagname[G__MAXNAME*6];
   if (G__struct.hash[tagnum]) {
      sprintf(mapped_tagname, "G__%sLN_%s"  , G__DLLID
              , G__map_cpp_name(G__fulltagname(tagnum, 0)));
   }
   else {
      sprintf(mapped_tagname, "G__%sLN_%s%d"  , G__DLLID
              , G__map_cpp_name(G__fulltagname(tagnum, 0)), tagnum);
   }
   return(mapped_tagname);
}

//______________________________________________________________________________
static const char* G__mark_linked_tagnum(int tagnum)
{
   // -- Setup and return tagnum.
   int tagnumorig = tagnum;
   if (tagnum < 0) {
      G__fprinterr(G__serr, "Internal error: G__mark_linked_tagnum() Illegal tagnum %d\n", tagnum);
      return("");
   }

   while (tagnum > 0) {
      if (G__NOLINK == G__struct.globalcomp[tagnum]) {
         /* this class is unlinked but tagnum interface requested.
         * G__globalcomp is already G__CLINK=-2 or G__CPPLINK=-1,
         * Following assignment will decrease the value by 2 more */
         G__struct.globalcomp[tagnum] = G__globalcomp - 2;
      }
      tagnum = G__struct.parent_tagnum[tagnum];
   }
   return(G__get_link_tagname(tagnumorig));
}

//______________________________________________________________________________
#ifdef G__GENWINDEF
void Cint::Internal::G__setDLLflag(int flag)
{
   G__isDLL = flag;
}
#else
void Cint::Internal::G__setDLLflag(int /*flag*/)
{}
#endif

//______________________________________________________________________________
void Cint::Internal::G__setPROJNAME(char* proj)
{
   strcpy(G__PROJNAME, G__map_cpp_name(proj));
}

//______________________________________________________________________________
#ifdef G__GENWINDEF
void Cint::Internal::G__setCINTLIBNAME(const char* cintlib)
{
   strcpy(G__CINTLIBNAME, cintlib);
}
#else
void Cint::Internal::G__setCINTLIBNAME(const char* /*cintlib*/)
{}
#endif

#ifdef G__GENWINDEF
//______________________________________________________________________________
static void G__write_windef_header()
{
   FILE* fp;

   fp = fopen(G__WINDEF, "w");
   if (!fp) G__fileerror(G__WINDEF);
   G__WINDEFfp = fp;

   if (G__isDLL)
      fprintf(fp, "LIBRARY           \"%s\"\n", G__PROJNAME);
   else
      fprintf(fp, "NAME              \"%s\" WINDOWAPI\n", G__PROJNAME);
   fprintf(fp, "\n");
#if defined(G__OLDIMPLEMENTATION1971) || !defined(G__VISUAL)
   fprintf(fp, "DESCRIPTION       '%s'\n", G__PROJNAME);
   fprintf(fp, "\n");
#endif
#if !defined(G__VISUAL) && !defined(G__CYGWIN)
   fprintf(fp, "EXETYPE           NT\n");
   fprintf(fp, "\n");
   if (G__isDLL)
      fprintf(fp, "SUBSYSTEM        WINDOWS\n");
   else
      fprintf(fp, "SUBSYSTEM   CONSOLE\n");
   fprintf(fp, "\n");
   fprintf(fp, "STUB              'WINSTUB.EXE'\n");
   fprintf(fp, "\n");
#endif        /* G__VISUAL */
   fprintf(fp, "VERSION           1.0\n");
   fprintf(fp, "\n");
#if defined(G__OLDIMPLEMENTATION1971) || !defined(G__VISUAL)
   fprintf(fp, "CODE               EXECUTE READ\n");
   fprintf(fp, "\n");
   fprintf(fp, "DATA               READ WRITE\n");
   fprintf(fp, "\n");
#endif
   fprintf(fp, "HEAPSIZE  1048576,4096\n");
   fprintf(fp, "\n");
#ifndef G__VISUAL
   fprintf(fp, "IMPORTS\n");
   fprintf(fp, "        _G__main=%s.G__main\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__setothermain=%s.G__setothermain\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__getnumbaseclass=%s.G__getnumbaseclass\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__setnewtype=%s.G__setnewtype\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__setnewtypeindex=%s.G__setnewtypeindex\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__resetplocal=%s.G__resetplocal\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__getgvp=%s.G__getgvp\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__resetglobalenv=%s.G__resetglobalenv\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__lastifuncposition=%s.G__lastifuncposition\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__resetifuncposition=%s.G__resetifuncposition\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__setnull=%s.G__setnull\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__getstructoffset=%s.G__getstructoffset\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__getaryconstruct=%s.G__getaryconstruct\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__gettempbufpointer=%s.G__gettempbufpointer\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__setsizep2memfunc=%s.G__setsizep2memfunc\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__getsizep2memfunc=%s.G__getsizep2memfunc\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__get_linked_tagnum=%s.G__get_linked_tagnum\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__tagtable_setup=%s.G__tagtable_setup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__search_tagname=%s.G__search_tagname\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__search_typename=%s.G__search_typename\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__defined_typename=%s.G__defined_typename\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__tag_memvar_setup=%s.G__tag_memvar_setup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__memvar_setup=%s.G__memvar_setup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__tag_memvar_reset=%s.G__tag_memvar_reset\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__tag_memfunc_setup=%s.G__tag_memfunc_setup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__memfunc_setup=%s.G__memfunc_setup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__memfunc_next=%s.G__memfunc_next\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__memfunc_para_setup=%s.G__memfunc_para_setup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__tag_memfunc_reset=%s.G__tag_memfunc_reset\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__letint=%s.G__letint\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__letdouble=%s.G__letdouble\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__store_tempobject=%s.G__store_tempobject\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__inheritance_setup=%s.G__inheritance_setup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__add_compiledheader=%s.G__add_compiledheader\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__add_ipath=%s.G__add_ipath\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__add_macro=%s.G__add_macro\n", G__CINTLIBNAME);
   fprintf(fp
           , "        _G__check_setup_version=%s.G__check_setup_version\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__int=%s.G__int\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__double=%s.G__double\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__calc=%s.G__calc\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__loadfile=%s.G__loadfile\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__unloadfile=%s.G__unloadfile\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__init_cint=%s.G__init_cint\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__scratch_all=%s.G__scratch_all\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__setdouble=%s.G__setdouble\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__setint=%s.G__setint\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__stubstoreenv=%s.G__stubstoreenv\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__stubrestoreenv=%s.G__stubrestoreenv\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__getstream=%s.G__getstream\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__type2string=%s.G__type2string\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__alloc_tempobject_val=%s.G__alloc_tempobject_val\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__set_p2fsetup=%s.G__set_p2fsetup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__free_p2fsetup=%s.G__free_p2fsetup\n", G__CINTLIBNAME);
   fprintf(fp, "        _G__search_typename2=%s.G__search_typename2\n", G__CINTLIBNAME);
   fprintf(fp, "\n");
#endif /* G__VISUAL */
   fprintf(fp, "EXPORTS\n");
   if (G__CPPLINK == G__globalcomp) {
      fprintf(fp, "        G__cpp_dllrev%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__set_cpp_environment%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__cpp_setup_tagtable%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__cpp_setup_inheritance%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__cpp_setup_typetable%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__cpp_setup_memvar%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__cpp_setup_memfunc%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__cpp_setup_global%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__cpp_setup_func%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__cpp_setup%s @%d\n", G__DLLID, ++G__nexports);
      if (G__multithreadlibcint)
         fprintf(fp, "        G__SetCppCintApiPointers @%d\n", ++G__nexports);
   }
   else {
      fprintf(fp, "        G__c_dllrev%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__set_c_environment%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__c_setup_tagtable%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__c_setup_typetable%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__c_setup_memvar%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__c_setup_global%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__c_setup_func%s @%d\n", G__DLLID, ++G__nexports);
      fprintf(fp, "        G__c_setup%s @%d\n", G__DLLID, ++G__nexports);
      if (G__multithreadlibcint)
         fprintf(fp, "        G__SetCCintApiPointers @%d\n", ++G__nexports);
   }
}
#endif // G__GENWINDEF

//______________________________________________________________________________
void Cint::Internal::G__set_globalcomp(const char* mode, const char* linkfilename, const char* dllid)
{
   FILE *fp;
   G__StrBuf buf_sb(G__LONGLINE);
   char *buf = buf_sb;
   G__StrBuf linkfilepref_sb(G__LONGLINE);
   char *linkfilepref = linkfilepref_sb;
   char linkfilepostf[20];
   char *p;

   strcpy(linkfilepref, linkfilename);
   p = strrchr(linkfilepref, '/'); /* ../aaa/bbb/ccc.cxx */
#ifdef G__WIN32
   if (!p) p = strrchr(linkfilepref, '\\'); /* in case of Windows pathname */
#endif
   if (!p) p = linkfilepref;      /*  /ccc.cxx */
   p = strrchr(p, '.');          /*  .cxx     */
   if (p) {
      strcpy(linkfilepostf, p + 1);
      *p = '\0';
   }
   else {
      sprintf(linkfilepostf, "C");
   }

   G__globalcomp = atoi(mode); /* this is redundant */
   if (abs(G__globalcomp) >= 10) {
      G__default_link = abs(G__globalcomp) % 10;
      G__globalcomp /= 10;
   }
   G__store_globalcomp = G__globalcomp;

   strcpy(G__DLLID, G__map_cpp_name(dllid));

   if (0 == strncmp(linkfilename, "G__cpp_", 7))
      strcpy(G__NEWID, G__map_cpp_name(linkfilename + 7));
   else if (0 == strncmp(linkfilename, "G__", 3))
      strcpy(G__NEWID, G__map_cpp_name(linkfilename + 3));
   else
      strcpy(G__NEWID, G__map_cpp_name(linkfilename));

   switch (G__globalcomp) {
      case G__CPPLINK:
         sprintf(buf, "%s.h", linkfilepref);
         G__CPPLINK_H = (char*)malloc(strlen(buf) + 1);
         strcpy(G__CPPLINK_H, buf);

         sprintf(buf, "%s.%s", linkfilepref, linkfilepostf);
         G__CPPLINK_C = (char*)malloc(strlen(buf) + 1);
         strcpy(G__CPPLINK_C, buf);

#ifdef G__GENWINDEF
         if (G__PROJNAME[0])
            sprintf(buf, "%s.def", G__PROJNAME);
         else if (G__DLLID[0])
            sprintf(buf, "%s.def", G__DLLID);
         else
            sprintf(buf, "%s.def", "G__lib");
         G__WINDEF = (char*)malloc(strlen(buf) + 1);
         strcpy(G__WINDEF, buf);
         G__write_windef_header();
#endif

         fp = fopen(G__CPPLINK_C, "w");
         if (!fp) G__fileerror(G__CPPLINK_C);
         fprintf(fp, "/********************************************************\n");
         fprintf(fp, "* %s\n", G__CPPLINK_C);
         fprintf(fp, "* CAUTION: DON'T CHANGE THIS FILE. THIS FILE IS AUTOMATICALLY GENERATED\n");
         fprintf(fp, "*          FROM HEADER FILES LISTED IN G__setup_cpp_environmentXXX().\n");
         fprintf(fp, "*          CHANGE THOSE HEADER FILES AND REGENERATE THIS FILE.\n");
         fprintf(fp, "********************************************************/\n");
         fprintf(fp, "#include \"%s\"\n", G__CPPLINK_H);

         fprintf(fp, "\n");
         fprintf(fp, "#ifdef G__MEMTEST\n");
         fprintf(fp, "#undef malloc\n");
         fprintf(fp, "#undef free\n");
         fprintf(fp, "#endif\n");
         fprintf(fp, "\n");

#ifdef __GNUC__
         fprintf(fp, "#if defined(__GNUC__) && (__GNUC__ > 3) && (__GNUC_MINOR__ > 1)\n");
         fprintf(fp, "#pragma GCC diagnostic ignored \"-Wstrict-aliasing\"\n");
         fprintf(fp, "#endif\n");
         fprintf(fp, "\n");
#endif // __GNUC__

         fprintf(fp, "extern \"C\" void G__cpp_reset_tagtable%s();\n", G__DLLID);

         fprintf(fp, "\nextern \"C\" void G__set_cpp_environment%s() {\n", G__DLLID);
         fclose(fp);
         break;
      case G__CLINK:
         sprintf(buf, "%s.h", linkfilepref);
         G__CLINK_H = (char*)malloc(strlen(buf) + 1);
         strcpy(G__CLINK_H, buf);

         sprintf(buf, "%s.c", linkfilepref);
         G__CLINK_C = (char*)malloc(strlen(buf) + 1);
         strcpy(G__CLINK_C, buf);

#ifdef G__GENWINDEF
         sprintf(buf, "%s.def", G__PROJNAME);
         G__WINDEF = (char*)malloc(strlen(buf) + 1);
         strcpy(G__WINDEF, buf);
         G__write_windef_header();
#endif

         fp = fopen(G__CLINK_C, "w");
         if (!fp) G__fileerror(G__CLINK_C);
         fprintf(fp, "/********************************************************\n");
         fprintf(fp, "* %s\n", G__CLINK_C);
         fprintf(fp, "********************************************************/\n");
         fprintf(fp, "#include \"%s\"\n", G__CLINK_H);
         fprintf(fp, "void G__c_reset_tagtable%s();\n", G__DLLID);
         fprintf(fp, "void G__set_c_environment%s() {\n", G__DLLID);
         fclose(fp);
         break;
      case R__CPPLINK:
         sprintf(buf, "%s.h", linkfilepref);
         G__CPPLINK_H = (char*)malloc(strlen(buf) + 1);
         strcpy(G__CPPLINK_H, buf);

         sprintf(buf, "%s.%s", linkfilepref, linkfilepostf);
         G__CPPLINK_C = (char*)malloc(strlen(buf) + 1);
         strcpy(G__CPPLINK_C, buf);

#ifdef G__GENWINDEF
         if (G__PROJNAME[0])
            sprintf(buf, "%s.def", G__PROJNAME);
         else if (G__DLLID[0])
            sprintf(buf, "%s.def", G__DLLID);
         else
            sprintf(buf, "%s.def", "G__lib");
         G__WINDEF = (char*)malloc(strlen(buf) + 1);
         strcpy(G__WINDEF, buf);
         G__write_windef_header();
#endif

         fp = fopen(G__CPPLINK_C, "w");
         if (!fp) G__fileerror(G__CPPLINK_C);
         fprintf(fp, "/********************************************************\n");
         fprintf(fp, "* %s\n", G__CPPLINK_C);
         fprintf(fp, "* CAUTION: DON'T CHANGE THIS FILE. THIS FILE IS AUTOMATICALLY GENERATED\n");
         fprintf(fp, "*          FROM HEADER FILES LISTED IN G__setup_cpp_environmentXXX().\n");
         fprintf(fp, "*          CHANGE THOSE HEADER FILES AND REGENERATE THIS FILE.\n");
         fprintf(fp, "********************************************************/\n");
         fprintf(fp, "#include \"%s\"\n", G__CPPLINK_H);

         fprintf(fp, "\n");
         fclose(fp);
         break;
   }
}

//______________________________________________________________________________
static void G__gen_headermessage(FILE* fp, char* fname)
{
   fprintf(fp, "/********************************************************************\n");
   fprintf(fp, "* %s\n", fname);
   fprintf(fp, "* CAUTION: DON'T CHANGE THIS FILE. THIS FILE IS AUTOMATICALLY GENERATED\n");
   fprintf(fp, "*          FROM HEADER FILES LISTED IN G__setup_cpp_environmentXXX().\n");
   fprintf(fp, "*          CHANGE THOSE HEADER FILES AND REGENERATE THIS FILE.\n");
   fprintf(fp, "********************************************************************/\n");
   fprintf(fp, "#ifdef __CINT__\n");
   fprintf(fp, "#error %s/C is only for compilation. Abort cint.\n"
           , fname);
   fprintf(fp, "#endif\n");
}

//______________________________________________________________________________
int Cint::Internal::G__gen_linksystem(char* headerfile)
{
   FILE *fp;
   switch (G__globalcomp) {
      case G__CPPLINK: /* C++ link */
         fp = fopen(G__CPPLINK_C, "a");
         break;
      case G__CLINK:   /* C link */
         fp = fopen(G__CLINK_C, "a");
         break;
      default:
         return(0);
   }
   fprintf(fp, "  G__add_compiledheader(\"<%s\");\n", headerfile);
   fclose(fp);
   return 0;
}

//______________________________________________________________________________
void Cint::Internal::G__gen_cppheader(char* headerfilein)
{
   FILE *fp;
   static char hdrpost[10] = "";
   G__StrBuf headerfile_sb(G__ONELINE);
   char *headerfile = headerfile_sb;
   char* p;

   switch (G__globalcomp) {
      case G__CPPLINK: /* C++ link */
      case G__CLINK:   /* C link */
      case R__CPPLINK: /* C++ link (reflex) */
         break;
      default:
         return;
   }

   if (headerfilein) {
      /*************************************************************
      * if header file is already created
      *************************************************************/

      strcpy(headerfile, headerfilein);
      /*************************************************************
      * if preprocessed file xxx.i is given rename as xxx.h
      *************************************************************/
      if (strlen(headerfile) > 2 &&
            (strcmp(".i", headerfile + strlen(headerfile) - 2) == 0 ||
             strcmp(".I", headerfile + strlen(headerfile) - 2) == 0)) {
         if ('\0' == hdrpost[0]) {
            switch (G__globalcomp) {
               case G__CPPLINK: /* C++ link */
                  strcpy(hdrpost, G__getmakeinfo1("CPPHDRPOST"));
                  break;
               case R__CPPLINK:
                  break;
               case G__CLINK: /* C link */
                  strcpy(hdrpost, G__getmakeinfo1("CHDRPOST"));
                  break;
            }
         }
         strcpy(headerfile + strlen(headerfile) - 2, hdrpost);
      }

      /* backslash escape sequence */
      p = strchr(headerfile, '\\');
      if (p) {
         G__StrBuf temp2_sb(G__ONELINE);
         char *temp2 = temp2_sb;
         int i = 0, j = 0;
         while (headerfile[i]) {
            switch (headerfile[i]) {
               case '\\':
                  temp2[j++] = headerfile[i];
                  temp2[j++] = headerfile[i++];
                  break;
               default:
                  temp2[j++] = headerfile[i++];
                  break;
            }
         }
         temp2[j] = '\0';
         strcpy(headerfile, temp2);
      }

#ifdef G__ROOT
      if (!((strstr(headerfile, "LinkDef") || strstr(headerfile, "Linkdef") ||
             strstr(headerfile, "linkdef")) && strstr(headerfile, ".h"))) {
#endif
         /* if(strstr(headerfile,".h")||strstr(headerfile,".H")) { */
         switch (G__globalcomp) {
            case G__CPPLINK:
               fp = fopen(G__CPPLINK_H, "a");
               if (!fp) G__fileerror(G__CPPLINK_H);
               fprintf(fp, "#include \"%s\"\n", headerfile);
               fclose(fp);
               fp = fopen(G__CPPLINK_C, "a");
               if (!fp) G__fileerror(G__CPPLINK_C);
               fprintf(fp, "  G__add_compiledheader(\"%s\");\n", headerfile);
               fclose(fp);
               break;
            case G__CLINK:
               fp = fopen(G__CLINK_H, "a");
               if (!fp) G__fileerror(G__CLINK_H);
               fprintf(fp, "#include \"%s\"\n", headerfile);
               fclose(fp);
               fp = fopen(G__CLINK_C, "a");
               if (!fp) G__fileerror(G__CLINK_C);
               fprintf(fp, "  G__add_compiledheader(\"%s\");\n", headerfile);
               fclose(fp);
               break;
            case R__CPPLINK:
               fp = fopen(G__CPPLINK_H, "a");
               if (!fp) G__fileerror(G__CPPLINK_H);
               fprintf(fp, "#include \"%s\"\n", headerfile);
               fclose(fp);
               break;
         }
         /* } */
#ifdef G__ROOT
      }
#endif
   }
   else {
      /*************************************************************
      * if header file is not created yet
      *************************************************************/
      switch (G__globalcomp) {
         case G__CPPLINK:
            fp = fopen(G__CPPLINK_H, "w");
            if (!fp) G__fileerror(G__CPPLINK_H);
            G__gen_headermessage(fp, G__CPPLINK_H);
            G__cpplink_header(fp);
            fclose(fp);
            break;
         case G__CLINK:
            fp = fopen(G__CLINK_H, "w");
            if (!fp) G__fileerror(G__CLINK_H);
            G__gen_headermessage(fp, G__CLINK_H);
            G__clink_header(fp);
            fclose(fp);
            break;
         case R__CPPLINK:
            fp = fopen(G__CPPLINK_H, "w");
            if (!fp) G__fileerror(G__CPPLINK_H);
            G__gen_headermessage(fp, G__CPPLINK_H);
            fclose(fp);
            break;
      }
   }
}

//______________________________________________________________________________
extern "C" void G__add_compiledheader(const char* headerfile)
{
   if (headerfile && headerfile[0] == '<' && G__autoload_stdheader) {
      ::Reflex::Scope store_tagnum = G__tagnum;
      ::Reflex::Scope store_def_tagnum = G__def_tagnum;
      ::Reflex::Scope store_tagdefining = G__tagdefining;
      int store_def_struct_member = G__def_struct_member;
#ifdef G__OLDIMPLEMENTATION1284_YET
      if (G__def_struct_member && 'n' == G__struct.type[G__tagdefining]) {
         G__def_struct_member = 1;
      }
      else {
         G__tagnum = -1;
         G__def_tagnum = -1;
         G__tagdefining = -1;
         G__def_struct_member = 0;
      }
#else
      G__tagnum = ::Reflex::Scope();
      G__def_tagnum = ::Reflex::Scope();
      G__tagdefining = ::Reflex::Scope();
      G__def_struct_member = 0;
#endif
      G__loadfile(headerfile + 1);
      G__tagnum = store_tagnum;
      G__def_tagnum = store_def_tagnum;
      G__tagdefining = store_tagdefining;
      G__def_struct_member = store_def_struct_member;
   }
}

//______________________________________________________________________________
extern "C" void G__add_macro(const char* macroin)
{
   // Macros starting with '!' are assumed to be system level macros
   // that will not be passed to an external preprocessor.
   G__StrBuf temp_sb(G__LONGLINE);
   char *temp = temp_sb;
   FILE *fp;
   char *p;
   G__StrBuf macro_sb(G__LONGLINE);
   char *macro = macro_sb;
   ::Reflex::Scope store_tagnum = G__tagnum;
   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   int store_def_struct_member = G__def_struct_member;
   int store_var_type = G__var_type;
   ::Reflex::Scope store_p_local = G__p_local;
   G__tagnum = ::Reflex::Scope::GlobalScope();
   G__def_tagnum = ::Reflex::Scope::GlobalScope();
   G__tagdefining = ::Reflex::Scope::GlobalScope();
   G__def_struct_member = 0;
   G__var_type = 'p';
   G__p_local = ::Reflex::Scope();

   const char* macroname = macroin;
   if (macroname[0] == '!')
      ++macroname;
   strcpy(macro, macroname);
   G__definemacro = 1;
   if ((p = strchr(macro, '='))) {
      if (G__cpp && '"' == *(p + 1)) {
         G__add_quotation(p + 1, temp);
         strcpy(p + 1, temp + 1);
         macro[strlen(macro)-1] = 0;
      }
      else {
         strcpy(temp, macro);
      }
   }
   else {
      sprintf(temp, "%s=1", macro);
   }
   G__getexpr(temp);
   G__definemacro = 0;

   if (macroin[0] == '!')
      goto end_add_macro;

   sprintf(temp, "\"-D%s\" ", macro);
   p = strstr(G__macros, temp);
   /*   " -Dxxx -Dyyy -Dzzz"
   *       p  ^              */
   if (p) goto end_add_macro;
   strcpy(temp, G__macros);
   if (strlen(temp) + strlen(macro) + 3 > G__LONGLINE) {
      if (G__dispmsg >= G__DISPWARN) {
         G__fprinterr(G__serr, "Warning: can not add any more macros in the list\n");
         G__printlinenum();
      }
   }
   else {
      sprintf(G__macros, "%s\"-D%s\" ", temp, macro);
   }

   switch (G__globalcomp) {
      case G__CPPLINK:
         fp = fopen(G__CPPLINK_C, "a");
         if (!fp) G__fileerror(G__CPPLINK_C);
         fprintf(fp, "  G__add_macro(\"%s\");\n", macro);
         fclose(fp);
         break;
      case G__CLINK:
         fp = fopen(G__CLINK_C, "a");
         if (!fp) G__fileerror(G__CLINK_C);
         fprintf(fp, "  G__add_macro(\"%s\");\n", macro);
         fclose(fp);
         break;
   }
end_add_macro:
   G__tagnum = store_tagnum;
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__def_struct_member = store_def_struct_member;
   G__var_type = store_var_type;
   G__p_local = store_p_local;
}

//______________________________________________________________________________
extern "C" void G__add_ipath(const char* path)
{
   struct G__includepath *ipath;
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   FILE *fp;
   char *p;
   char *store_allincludepath;

   /* strip double quotes if exist */
   if ('"' == path[0]) {
      strcpy(temp, path + 1);
      if ('"' == temp[strlen(temp)-1]) temp[strlen(temp)-1] = '\0';
   }
   else {
      strcpy(temp, path);
   }

   /* to the end of list */
   ipath = &G__ipathentry;
   while (ipath->next) {
      if (ipath->pathname && strcmp(ipath->pathname, temp) == 0) return;
      ipath = ipath->next;
   }

   /* G__allincludepath will be given to real preprocessor */
   if (!G__allincludepath) {
      G__allincludepath = (char*)malloc(1);
      G__allincludepath[0] = '\0';
   }
   store_allincludepath = (char*)realloc((void*)G__allincludepath
                                         , strlen(G__allincludepath) + strlen(temp) + 6);
   if (store_allincludepath) {
      int i = 0, flag = 0;
      while (temp[i]) if (isspace(temp[i++])) flag = 1;
      G__allincludepath = store_allincludepath;
      if (flag)
         sprintf(G__allincludepath + strlen(G__allincludepath) , "-I\"%s\" ", temp);
      else
         sprintf(G__allincludepath + strlen(G__allincludepath) , "-I%s ", temp);
   }
   else {
      G__genericerror("Internal error: memory allocation failed for includepath buffer");
   }


   /* copy the path name */
   ipath->pathname = (char *)malloc((size_t)(strlen(temp) + 1));
   strcpy(ipath->pathname, temp);

   /* allocate next entry */
   ipath->next = (struct G__includepath *)malloc(sizeof(struct G__includepath));
   ipath->next->next = (struct G__includepath *)NULL;
   ipath->next->pathname = (char *)NULL;

   /* backslash escape sequence */
   p = strchr(temp, '\\');
   if (p) {
      G__StrBuf temp2_sb(G__ONELINE);
      char *temp2 = temp2_sb;
      int i = 0, j = 0;
      while (temp[i]) {
         switch (temp[i]) {
            case '\\':
               temp2[j++] = temp[i];
               temp2[j++] = temp[i++];
               break;
            default:
               temp2[j++] = temp[i++];
               break;
         }
      }
      temp2[j] = '\0';
      strcpy(temp, temp2);
   }

   /* output include path information to interface routine */
   switch (G__globalcomp) {
      case G__CPPLINK:
         fp = fopen(G__CPPLINK_C, "a");
         if (!fp) G__fileerror(G__CPPLINK_C);
         fprintf(fp, "  G__add_ipath(\"%s\");\n", temp);
         fclose(fp);
         break;
      case G__CLINK:
         fp = fopen(G__CLINK_C, "a");
         if (!fp) G__fileerror(G__CLINK_C);
         fprintf(fp, "  G__add_ipath(\"%s\");\n", temp);
         fclose(fp);
         break;
   }
}

//______________________________________________________________________________
extern "C" int G__delete_ipath(const char* path)
{
   struct G__includepath *ipath;
   struct G__includepath *previpath;
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   G__StrBuf temp2_sb(G__ONELINE);
   char *temp2 = temp2_sb;
   int i = 0, flag = 0;
   char *p;

   /* strip double quotes if exist */
   if ('"' == path[0]) {
      strcpy(temp, path + 1);
      if ('"' == temp[strlen(temp)-1]) temp[strlen(temp)-1] = '\0';
   }
   else {
      strcpy(temp, path);
   }

   /* to the end of list */
   ipath = &G__ipathentry;
   previpath = (struct G__includepath*)NULL;
   while (ipath->next) {
      if (ipath->pathname && strcmp(ipath->pathname, temp) == 0) {
         /* delete this entry */
         free((void*)ipath->pathname);
         ipath->pathname = (char*)NULL;
         if (previpath) {
            previpath->next = ipath->next;
            free((void*)ipath);
         }
         else if (ipath->next) {
            G__ipathentry.pathname = (char*)calloc(1, 1);
         }
         else {
            free((void*)G__ipathentry.pathname);
            G__ipathentry.pathname = (char*)NULL;
         }
         break;
      }
      previpath = ipath;
      ipath = ipath->next;
   }

   /* G__allincludepath will be given to real preprocessor */
   if (!G__allincludepath) return(0);
   i = 0;
   while (temp[i]) if (isspace(temp[i++])) flag = 1;
   if (flag) sprintf(temp2, "-I\"%s\" ", temp);
   else     sprintf(temp2, "-I%s ", temp);

   p = strstr(G__allincludepath, temp2);
   if (p) {
      char *p2 = p + strlen(temp2);
      while (*p2) *p++ = *p2++;
      *p = *p2;
      return(1);
   }

   return(0);
}

//______________________________________________________________________________
static int G__isnonpublicnew(int tagnum)
{
   int i;
   int hash;
   const char *namenew = "operator new";

   G__hash(namenew, hash, i);

   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
   for (::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
         ifunc != scope.FunctionMember_End();
         ++ifunc) {
      if (/* hash==ifunc->hash[i] && */
         ifunc->Name() == namenew &&
         !G__test_access(*ifunc, G__PUBLIC)) {
         return(1);
      }
   }
   return(0);
}

//______________________________________________________________________________
int Cint::Internal::G__method_inbase(const Reflex::Member mbr)
{
   // This function search for the method ifunc in the base classes.
   // RETURN -> NULL Method not found
   //          NOT NULL Method Found. Method's ifunc table pointer
   // tagnum's Base Classes structure
   char name[4096];
   strcpy(name, mbr.Name(Reflex::SCOPED).c_str());
   Reflex::FunctionMember* fm = dynamic_cast<Reflex::FunctionMember*>(static_cast<Reflex::MemberBase*>(mbr.Id()));
   if (!fm) { // Passed member is not a Reflex::FunctionMember, quit.
      //fprintf(stderr, "G__method_inbase: %s: invalid function member, skipping ...\n", name);
      return 0;
   }
   G__inheritance* bases = G__struct.baseclass[G__get_tagnum(fm->DeclaringScope())];
   if (!bases) { // Declaring class has no bases, quit.
      //fprintf(stderr, "G__method_inbase: %s: class of member has no base classes, done.\n", name);
      return 0;
   }
   for (size_t i = 0; i < bases->vec.size(); ++i) { // loop over all base classes
      Reflex::Type base_type = G__Dict::GetDict().GetType(bases->vec[i].basetagnum);
      if (!base_type) { // invalid base, skip it
         //fprintf(stderr, "G__method_inbase: %s: invalid base, skipping ...\n", name);
         continue;
      }
      Reflex::Member base_mbr = G__ifunc_exist(*fm, base_type, true);
      if (!base_mbr) { // method not found in base, next base
         //fprintf(stderr, "G__method_inbase: %s: method not in base '%s', next base.\n", name, base_type.Name(Reflex::SCOPED).c_str());
         continue;
      }
      Reflex::FunctionMember* baseFunc = dynamic_cast<Reflex::FunctionMember*>(static_cast<Reflex::MemberBase*>(base_mbr.Id()));
      if (!baseFunc) { // base member is not a Reflex::FunctionMember, next base
         //fprintf(stderr, "G__method_inbase: %s: invalid base member function, skipping ...\n", name);
         continue;
      }
      //
      //  If the number of default parameters is the same in the
      //  base class version of the method then we have a match.
      //
      int base_def_cnt = baseFunc->FunctionParameterSize() - baseFunc->FunctionParameterSize(true);
      int derived_def_cnt = fm->FunctionParameterSize() - fm->FunctionParameterSize(true);
      if (base_def_cnt == derived_def_cnt) {
         //fprintf(stderr, "G__method_inbase: %s: method found in base '%s'.\n", name, base_type.Name(Reflex::SCOPED).c_str());
         return 1;
      }
   }
   return 0;
}

//______________________________________________________________________________
void Cint::Internal::G__cppif_memfunc(FILE* fp, FILE* hfp)
{
   // -- TODO: Describe this function!
#ifndef G__SMALLOBJECT
   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Member function Interface Method\n");
   fprintf(fp, "*********************************************************/\n");
   //
   //  Loop over all known classes, enums, namespaces, structs and unions.
   //
   for (int i = 0; i < G__struct.alltag; ++i) {
      // -- Loop over all known classes, enums, namespaces, structs and unions.
      // Only generate dictionary info for marked classes.
      if ( // class is marked, and passes nesting test, not an enum, is valid, and we have source info
         (
            (G__struct.globalcomp[i] == G__CPPLINK) || // class is marked for C++ link, or
            (G__struct.globalcomp[i] == G__CLINK) || // class is marked for C link, or
            (G__struct.globalcomp[i] == G__ONLYMETHODLINK) // linking only explicitly marked member functions,
         ) && // and,
         (
            ((int) G__struct.parent_tagnum[i] == -1) || // not a nested class, or
            G__nestedclass // #pragma link nestedclass;
         ) && // and,
         (G__struct.line_number[i] != -1) && // we have a line number for the class declaration (ClassDef macro), and
         G__struct.hash[i] && // the class has a non-empty name, and
         (G__struct.name[i][0] != '$') && // class is not an unnamed enum
         (G__struct.type[i] != 'e') // class is not an enum
      ) {
         int isconstructor = 0;
         int iscopyconstructor = 0;
         int isdestructor = 0;
         int isassignmentoperator = 0;
         int isnonpublicnew = G__isnonpublicnew(i);
         // Print class name for reference (help the poor human who has to read the dictionary!).
         fprintf(fp, "\n/* %s */\n", G__fulltagname(i, 0));
         // Get the corresponding Reflex scope.
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(i);
         //
         //  Loop over all of the member functions.
         //
         for ( // loop over all member functions of the class
            ::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
            ifunc != scope.FunctionMember_End();
            ++ifunc
         ) {
            // -- Loop over all of the member functions.
            //fprintf(stderr, "-----> %s\n", ifunc->Name(Reflex::SCOPED).c_str());
            if ( // member function is accessible
               G__test_access(*ifunc, G__PUBLIC) || // function is marked public, or
               (
                  G__test_access(*ifunc, G__PROTECTED) && // function is marked protected, and
                  (G__struct.protectedaccess[i] & G__PROTECTEDACCESS) // #pragma link C++ class+protected MyClass;
               ) || // or,
               (G__struct.protectedaccess[i] & G__PRIVATEACCESS) // #pragma link C++ class+private MyClass;
            ) {
               // -- Public member function, or selected by pragma on protected or private.
               if ( // only linking explicitly marked methods and not marked, then skip it
                  (G__struct.globalcomp[i] == G__ONLYMETHODLINK) && // linking only explicitly marked functions, and
                  (G__get_funcproperties(*ifunc)->globalcomp != G__METHODLINK) // function is not marked
               ) {
                  // -- Skip this function, we are only linking marked ones, and this one is not marked.
                  //fprintf(stderr, " -- skipped onlymethodlink\n");
                  continue;
               }
               if (G__get_funcproperties(*ifunc)->entry.size < 0) {
                  // -- Skip this function, it is precompiled.
                  //fprintf(stderr, " -- skipped negativepentrysize\n");
                  continue;
               }
               if (ifunc->Name() == G__struct.name[i]) { // constructor
                  // -- Constructor needs special handling.
                  if (!G__struct.isabstract[i] && !isnonpublicnew) {
                     //fprintf(stderr, " -- ok, is constructor, or is copy constructor\n");
                     G__cppif_genconstructor(fp, hfp, i, *ifunc);
                  } else {
                     //fprintf(stderr, " -- skipped, is constructor, or is copy constructor, abstract class or non-public new\n");
                  }
                  ++isconstructor;
                  if ((ifunc->FunctionParameterSize() >= 1) && ifunc->IsCopyConstructor()) { // copy constructor
                     ++iscopyconstructor;
                  }
               }
               else if (ifunc->Name()[0] == '~') { // destructor, skip it, handle it later
                  // -- The destructor is created in gendefault later.
                  if (G__test_access(*ifunc, G__PUBLIC)) {
                     isdestructor = -1;
                  }
                  else {
                     ++isdestructor;
                  }
                  //fprintf(stderr, " -- skipped destructor\n");
                  continue;
               }
               else {
                  // -- Normal function, or operator=.
#ifdef G__DEFAULTASSIGNOPR
                  if ( // operator=
                     (ifunc->Name() == "operator=") &&
                     (ifunc->TypeOf().FunctionParameterAt(0).RawType() == (::Reflex::Type) scope) &&
                     (G__get_type(ifunc->TypeOf().FunctionParameterAt(0)) == 'u')
                  ) {
                     ++isassignmentoperator;
                  }
#endif // G__DEFAULTASSIGNOPR
                  if (ifunc->IsVirtual() && G__method_inbase(*ifunc)) {
                     //fprintf(stderr, " -- skipped implementedinbase\n");
                  } else if (!ifunc->IsVirtual()) {
                     //fprintf(stderr, " -- ok, not virtual\n");
                     G__cppif_genfunc(fp, hfp, i, *ifunc);
                  } else {
                     //fprintf(stderr, " -- ok, virtual and not in base class\n");
                     G__cppif_genfunc(fp, hfp, i, *ifunc);
                  }
               }
            }
            else { // member function is not accessible, just accumulate flags
               // -- No access.
               //fprintf(stderr, " -- skipped noaccess\n");
               if (ifunc->Name() == G__struct.name[i]) { // constructor
                  ++isconstructor;
                  if (ifunc->IsCopyConstructor()) { // copy constructor
                     ++iscopyconstructor;
                  }
               }
               else if (ifunc->Name()[0] == '~') { // destructor
                  ++isdestructor;
               }
               else if (ifunc->Name() == "operator new") { // operator new
                  ++isconstructor;
                  ++iscopyconstructor;
               }
               else if (ifunc->Name() == "operator delete") { // operator delete
                  ++isdestructor;
               }
#ifdef G__DEFAULTASSIGNOPR
               else if (
                  (ifunc->Name() == "operator=") &&
                  (ifunc->TypeOf().FunctionParameterAt(0).RawType() == (::Reflex::Type) scope) &&
                  (G__get_type(ifunc->TypeOf().FunctionParameterAt(0)) == 'u')
               ) { // operator=
                  ++isassignmentoperator;
               }
#endif // G__DEFAULTASSIGNOPR
               // --
            }
         }
         //
         //  Now handle compiler-supplied member functions.
         //
         if ((G__struct.iscpplink[i] == G__NOLINK) && (G__struct.globalcomp[i] != G__ONLYMETHODLINK)) {
            // -- Create interfaces to the compiler-supplied member functions.
            G__cppif_gendefault(fp, hfp, i, isconstructor, iscopyconstructor, isdestructor, isassignmentoperator, isnonpublicnew);
         }
      }
   }
#endif // G__SMALLOBJECT
   // --
}

//______________________________________________________________________________
void Cint::Internal::G__cppif_func(FILE* fp, FILE* hfp)
{
   // -- Output the stubs for all the functions in the global namespace.
   fprintf(fp, "\n/* Setting up global function */\n");
   ::Reflex::Scope scope = ::Reflex::Scope::GlobalScope();
   ::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
   for (; ifunc != scope.FunctionMember_End(); ++ifunc) {
      if ( // -- Function is marked for geneartion and has public access.
         (G__get_funcproperties(*ifunc)->globalcomp < G__NOLINK) && // Flagged for stub generation, and
         G__test_access(*ifunc, G__PUBLIC) // Has public access
      ) {
         // -- Generate the stub.
         G__cppif_genfunc(fp, hfp, -1, *ifunc);
      }
   }
}

//______________________________________________________________________________
void Cint::Internal::G__cppif_dummyfuncname(FILE* fp)
{
   // --
#ifndef G__IF_DUMMY
   fprintf(fp, "   return(1);\n");
#else
   fprintf(fp, "   return(1 || funcname || hash || result7 || libp) ;\n");
#endif
   // --
}

//______________________________________________________________________________
static void G__if_ary_union(FILE* fp, const ::Reflex::Member& ifunc)
{
   int k, m;
   char* p;

   m = ifunc.FunctionParameterSize();

   for (k = 0; k < m; ++k) {
      if (ifunc.FunctionParameterNameAt(k)[0]) {
         p = strchr((char*)ifunc.FunctionParameterNameAt(k).c_str(), '[');
         ::Reflex::Type ptype(ifunc.TypeOf().FunctionParameterAt(k));
         if (p) {
            fprintf(fp, "  struct G__aRyp%d { %s a[1]%s; }* G__Ap%d = (struct G__aRyp%d*) G__int(libp->para[%d]);\n", k,
                    G__type2string(G__get_type(ptype), G__get_tagnum(ptype.RawType()), G__get_typenum(ptype), 0 , 0)
                    , p + 2, k, k, k);
         }
      }
   }
}


#ifdef G__CPPIF_EXTERNC
//______________________________________________________________________________
char* Cint::Internal::G__p2f_typedefname(int ifn, short page, int k)
{
   static char buf[G__ONELINE];
   sprintf(buf, "G__P2F%d_%d_%d%s", ifn, page, k, G__PROJNAME);
   return(buf);
}
#endif // G__CPPIF_EXTERNC

#ifdef G__CPPIF_EXTERNC
//______________________________________________________________________________
void Cint::Internal::G__p2f_typedef(FILE* fp, int ifn, struct G__ifunc_table* ifunc)
{
   G__StrBuf buf_sb(G__LONGLINE);
   char *buf = buf_sb;
   char *p;
   int k;
   if (G__CPPLINK != G__globalcomp) return;
   for (k = 0;k < ifunc.FunctionParameterSize();k++)
   {
      /*DEBUG*/ printf("%s %d\n", ifunc.Name().c_str(), k);
      if (
#ifndef G__OLDIMPLEMENTATION2191
         '1' == ifunc->para_type[ifn][k]
#else
         'Q' == ifunc->para_type[ifn][k]
#endif
      ) {
         strcpy(buf, G__type2string(ifunc->para_type[ifn][k],
                                    ifunc->para_p_tagtable[ifn][k],
                                    ifunc->para_p_typetable[ifn][k], 0,
                                    ifunc->para_isconst[ifn][k]));
         /*DEBUG*/
         printf("buf=%s\n", buf);
         p = strstr(buf, "(*)(");
         if (p) {
            p += 2;
            *p = 0;
            fprintf(fp, "typedef %s%s", buf, G__p2f_typedefname(ifn, ifunc->page, k));
            *p = ')';
            fprintf(fp, "%s;\n", p);
         }
      }
   }
}
#endif // G__CPPIF_EXTERNC

//______________________________________________________________________________
static int G__isprotecteddestructoronelevel(int tagnum)
{
   char *dtorname = (char*)malloc(strlen(G__struct.name[tagnum]) + 2);
   dtorname[0] = '~';
   strcpy(dtorname + 1, G__struct.name[tagnum]);

   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
   for (::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
         ifunc != scope.FunctionMember_End();
         ++ifunc) {
      if (strcmp(dtorname, ifunc->Name().c_str()) == 0) {
         if (G__test_access(*ifunc, G__PRIVATE)
               || G__test_access(*ifunc, G__PROTECTED)) {
            free((void*)dtorname);
            return(1);
         }
      }
   }

   free((void*)dtorname);
   return(0);
}

#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
//______________________________________________________________________________
static void G__x8664_vararg(FILE* fp, const ::Reflex::Member& ifunc, const char* fn, int tagnum, const char* cls)
{
   // This function sets up vararg calls on X86_64 (AMD64 and EM64T).
   // On these platforms arguments are passed by register and not on
   // the stack. The Linux ABI specifies that the first 6 integer and
   // 8 double arguments are passed via registers, while any remaining
   // arguments are passed via the stack. In this function we use inline
   // assembler to set the arguments in the right registers before
   // calling the vararg function.
   const int umax = 20;   // maximum number of extra vararg stack arguments
   int i;

   fprintf(fp, "  const int imax = 6, dmax = 8, umax = 20;\n");
   fprintf(fp, "  int objsize, type, i, icnt = 0, dcnt = 0, ucnt = 0;\n");
   fprintf(fp, "  G__value *pval;\n");
   fprintf(fp, "  G__int64 lval[imax];\n");
   fprintf(fp, "  double dval[dmax];\n");
   fprintf(fp, "  union { G__int64 lval; double dval; } u[umax];\n");
   
   int type    = G__get_type(ifunc.TypeOf().ReturnType());
   int reftype = G__get_reftype(ifunc.TypeOf().ReturnType());
   if (type == 'u' && reftype==0) {
      // The function returns an object by value, so we need to reserve space
      // for it and pass it to the function.
      fprintf(fp, "  char returnValue[sizeof(%s)];\n", ifunc.TypeOf().ReturnType().Name(Reflex::SCOPED).c_str());
      fprintf(fp, "  lval[icnt] = (G__int64)returnValue; icnt++; // Object returned by value\n");
   }
      
   if (tagnum != -1 && !ifunc.IsStatic())
      fprintf(fp, "  lval[icnt] = G__getstructoffset(); icnt++;  // this pointer\n");

   fprintf(fp, "  for (i = 0; i < libp->paran; i++) {\n");
   fprintf(fp, "    type = G__value_get_type(&libp->para[i]);\n");
   fprintf(fp, "    pval = &libp->para[i];\n");
   fprintf(fp, "    if (isupper(type))\n");
   fprintf(fp, "      objsize = G__LONGALLOC;\n");
   fprintf(fp, "    else\n");
   fprintf(fp, "      objsize = G__sizeof(pval);\n");

   fprintf(fp, "    switch (type) {\n");
   fprintf(fp, "      case 'c': case 'b': case 's': case 'r': objsize = sizeof(int); break;\n");
   fprintf(fp, "      case 'f': objsize = sizeof(double); break;\n");
   fprintf(fp, "    }\n");

#ifdef G__VAARG_PASS_BY_REFERENCE
   fprintf(fp, "    if (objsize > %d /* G__VAARG_PASS_BY_REFERENCE */ ) {\n", G__VAARG_PASS_BY_REFERENCE);
   fprintf(fp, "      if (pval->ref > 0x1000) {\n");
   fprintf(fp, "        if (icnt < imax) {\n");
   fprintf(fp, "          lval[icnt] = pval->ref; icnt++;\n");
   fprintf(fp, "        } else {\n");
   fprintf(fp, "          u[ucnt].lval = pval->ref; ucnt++;\n");
   fprintf(fp, "        }\n");
   fprintf(fp, "      } else {\n");
   fprintf(fp, "        if (icnt < imax) {\n");
   fprintf(fp, "          lval[icnt] = G__int(*pval); icnt++;\n");
   fprintf(fp, "        } else {\n");
   fprintf(fp, "          u[ucnt].lval = G__int(*pval); ucnt++;\n");
   fprintf(fp, "        }\n");
   fprintf(fp, "      }\n");
   fprintf(fp, "      type = 'z';\n");
   fprintf(fp, "    }\n");
#endif

   fprintf(fp, "    switch (type) {\n");
   fprintf(fp, "      case 'n': case 'm':\n");
   fprintf(fp, "        if (icnt < imax) {\n");
   fprintf(fp, "          lval[icnt] = (G__int64)G__Longlong(*pval); icnt++;\n");
   fprintf(fp, "        } else {\n");
   fprintf(fp, "          u[ucnt].lval = (G__int64)G__Longlong(*pval); ucnt++;\n");
   fprintf(fp, "        } break;\n");
   fprintf(fp, "      case 'f': case 'd':\n");
   fprintf(fp, "        if (dcnt < dmax) {\n");
   fprintf(fp, "          dval[dcnt] = G__double(*pval); dcnt++;\n");
   fprintf(fp, "        } else {\n");
   fprintf(fp, "          u[ucnt].dval = G__double(*pval); ucnt++;\n");
   fprintf(fp, "        } break;\n");
   fprintf(fp, "      case 'z': break;\n");
   fprintf(fp, "      case 'u':\n");
   fprintf(fp, "        if (objsize >= 16) {\n");
   fprintf(fp, "          memcpy(&u[ucnt].lval, (void*)pval->obj.i, objsize);\n");
   fprintf(fp, "          ucnt += objsize/8;\n");
   fprintf(fp, "          break;\n");
   fprintf(fp, "        }\n");
   fprintf(fp, "        // objsize < 16 -> fall through\n");
   fprintf(fp, "      case 'g': case 'c': case 'b': case 'r': case 's': case 'h': case 'i':\n");
   fprintf(fp, "      case 'k': case 'l':\n");
   fprintf(fp, "      default:\n");
   fprintf(fp, "        if (icnt < imax) {\n");
   fprintf(fp, "          lval[icnt] = G__int(*pval); icnt++;\n");
   fprintf(fp, "        } else {\n");
   fprintf(fp, "          u[ucnt].lval = G__int(*pval); ucnt++;\n");
   fprintf(fp, "        } break;\n");
   fprintf(fp, "    }\n");
   fprintf(fp, "    if (ucnt >= %d) printf(\"%s: more than %d var args\\n\");\n", umax, fn, umax);
   fprintf(fp, "  }\n");

   // example of what we try to generate:
   //    void (TQObject::*fptr)(const char *, Int_t, ...) = &TQObject::EmitVA;

   //int type    = ifunc->type[ifn];
   //int ptagnum = ifunc->p_tagtable[ifn];
   //::Reflex::Type typenum = ifunc->p_typetable[ifn];
   //int reftype = ifunc->reftype[ifn];
   //int isconst = ifunc->isconst[ifn];

   int m = ifunc.FunctionParameterSize();
   if (tagnum != -1) {
      if (!strcmp(fn, cls)) {
         // variadic constructor case, not yet supported
         printf("G__x8664_vararg: variadic constructors not yet supported\n");
         return;
      }
      else {
         // write return type
         const char *typestring = ifunc.TypeOf().ReturnType().Name(::Reflex::SCOPED).c_str(); // G__type2string(type, ptagnum, G__get_typenum(typenum), reftype, isconst);
         if (ifunc.IsStatic() || ifunc.DeclaringScope().IsNamespace()) {
            fprintf(fp, "  %s (*fptr)(", typestring);
         }
         else {
            // class method
            fprintf(fp, "  %s (%s::*fptr)(", typestring, cls);
         }

         // write arguments
         for (int k = 0; k < m; k++) {
            //type    = ifunc->para_type[ifn][k];
            //ptagnum = ifunc->para_p_tagtable[ifn][k];
            //typenum = ifunc->para_p_typetable[ifn][k];
            //reftype = ifunc->para_reftype[ifn][k];
            //isconst = ifunc->para_isconst[ifn][k];

            if (k)
               fprintf(fp, ", ");
            fprintf(fp, "%s", ifunc.TypeOf().FunctionParameterAt(k).Name(::Reflex::SCOPED | ::Reflex::QUALIFIED).c_str());
         }
         fprintf(fp, ", ...) %s = &%s::%s;\n",
                 (ifunc.IsConst() /* ->isconst[ifn] & G__CONSTFUNC */) ? "const" : "", cls, fn);
      }
   }
   else {
      fprintf(fp, "  long fptr = (long)&%s;\n", fn);
   }

   if (tagnum != -1 && ifunc.IsVirtual()) {
      fprintf(fp, "  // special prologue since virtual member function pointers contain\n");
      fprintf(fp, "  // only an offset into the virtual table and not a function address\n");
      fprintf(fp, "  long faddr;\n");

      fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%rax\"  :: \"m\" (lval[0]) : \"%%rax\", \"%%rdx\");\n");
      fprintf(fp, "  __asm__ __volatile__(\"movq %%rax, %%rdx\");\n");
      fprintf(fp, "  __asm__ __volatile__(\"leaq %%0, %%%%rax\" :: \"m\" (fptr));\n");
      fprintf(fp, "  __asm__ __volatile__(\"movq 8(%%rax), %%rax\");  //multiple inheritance offset\n");
#if defined(__GNUC__) && __GNUC__ > 3
      fprintf(fp, "  __asm__ __volatile__(\"leaq (%%rdx,%%rax), %%rax\");\n");
      fprintf(fp, "  __asm__ __volatile__(\"movq (%%rax), %%rdx\");\n");
      fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%rax\"  :: \"m\" (fptr));  //virtual member function offset\n");
      fprintf(fp, "  __asm__ __volatile__(\"leaq (%%rdx,%%rax), %%rax\");\n");
#else
      fprintf(fp, "  __asm__ __volatile__(\"addq %%rax, %%rdx\");\n");
      fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%rax\"  :: \"m\" (fptr));  //virtual member function offset\n");
      fprintf(fp, "  __asm__ __volatile__(\"addq (%%rdx), %%rax\");\n");
#endif
      fprintf(fp, "  __asm__ __volatile__(\"decq %%rax\");\n");
      fprintf(fp, "  __asm__ __volatile__(\"movq (%%rax), %%rax\");\n");
      fprintf(fp, "  __asm__ __volatile__(\"movq %%%%rax, %%0\"  : \"=m\" (faddr) :: \"memory\");\n\n");
   }

   fprintf(fp, "  __asm__ __volatile__(\"movlpd %%0, %%%%xmm0\"  :: \"m\" (dval[0]) : \"%%xmm0\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movlpd %%0, %%%%xmm1\"  :: \"m\" (dval[1]) : \"%%xmm1\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movlpd %%0, %%%%xmm2\"  :: \"m\" (dval[2]) : \"%%xmm2\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movlpd %%0, %%%%xmm3\"  :: \"m\" (dval[3]) : \"%%xmm3\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movlpd %%0, %%%%xmm4\"  :: \"m\" (dval[4]) : \"%%xmm4\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movlpd %%0, %%%%xmm5\"  :: \"m\" (dval[5]) : \"%%xmm5\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movlpd %%0, %%%%xmm6\"  :: \"m\" (dval[6]) : \"%%xmm6\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movlpd %%0, %%%%xmm7\"  :: \"m\" (dval[7]) : \"%%xmm7\");\n");

   fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%rdi\" :: \"m\" (lval[0]) : \"%%rdi\");\n");
#if defined(__GNUC__) && __GNUC__ < 4
   if (tagnum != -1 && ifunc.IsVirtual()) {
      fprintf(fp, "  __asm__ __volatile__(\"leaq %%0, %%%%rax\" :: \"m\" (fptr));\n");
      fprintf(fp, "  __asm__ __volatile__(\"movq 8(%%rax), %%rax\");  //multiple inheritance offset\n");
      fprintf(fp, "  __asm__ __volatile__(\"addq %%rax, %%rdi\");\n");
   }
#endif
   fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%rsi\" :: \"m\" (lval[1]) : \"%%rsi\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%rdx\" :: \"m\" (lval[2]) : \"%%rdx\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%rcx\" :: \"m\" (lval[3]) : \"%%rcx\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%r8\"  :: \"m\" (lval[4]) : \"%%r8\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%r9\"  :: \"m\" (lval[5]) : \"%%r9\");\n");

   int istck = 0;
   fprintf(fp, "  __asm__ __volatile__(\"subq %%0, %%%%rsp\" :: \"i\" ((umax+2)*8));\n");
   for (i = 0; i < umax; i++) {
      fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%rax \\n\\t\"\n");
      fprintf(fp, "                       \"movq %%%%rax, %d(%%%%rsp)\" :: \"m\" (u[%d].lval) : \"%%rax\");\n", istck, i);
      istck += 8;
   }
   if (tagnum != -1 && ifunc.IsVirtual())
      fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%r10\"  :: \"m\" (faddr) : \"%%r10\");\n");
   else
      fprintf(fp, "  __asm__ __volatile__(\"movq %%0, %%%%r10\"  :: \"m\" (fptr) : \"%%r10\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movl $8, %%eax\");  // number of used xmm registers\n");
   fprintf(fp, "  __asm__ __volatile__(\"call *%%r10\");\n");
   fprintf(fp, "  __asm__ __volatile__(\"movq %%%%rax, %%0\" : \"=m\" (u[0].lval) :: \"memory\");  // get return value\n");
   fprintf(fp, "  __asm__ __volatile__(\"addq %%0, %%%%rsp\" :: \"i\" ((umax+2)*8));\n");
}
#endif // __x86_64__ && (__linux || __APPLE__)

#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
//______________________________________________________________________________
static void G__x8664_vararg_epilog(FILE* fp, const ::Reflex::Member& ifunc)
{
   char *typestring;

   int type    = G__get_type(ifunc.TypeOf().ReturnType());
   int tagnum  = G__get_tagnum(ifunc.TypeOf().ReturnType().RawType());
   int reftype = G__get_reftype(ifunc.TypeOf().ReturnType());
   int isconst = G__get_isconst(ifunc.TypeOf().ReturnType());

   // Function return type is a reference, handle and return.
   if (reftype == G__PARAREFERENCE) {
      if (isconst & G__CONSTFUNC) {
         if (isupper(type)) {
            isconst |= G__PCONSTVAR;
         }
         else {
            isconst |= G__CONSTVAR;
         }
      }
      fprintf(fp, "(%s) u[0].lval", ifunc.TypeOf().ReturnType().Name(::Reflex::SCOPED).c_str());
      return;
   }

   // Function return type is a pointer, handle and return.
   if (isupper(type)) {
      fprintf(fp, "u[0].lval");
      return;
   }

   // Function returns an object or a fundamental type.
   switch (type) {
      case 'y':
         break;
      case '1':
      case 'e':
      case 'c':
      case 's':
      case 'i':
      case 'l':
      case 'b':
      case 'r':
      case 'h':
      case 'k':
      case 'g':
      case 'n':
      case 'm':
         fprintf(fp, "u[0].lval");
         break;
      case 'q':
      case 'f':
      case 'd':
         fprintf(fp, "u[0].dval");
         break;
      case 'u':
         switch (G__struct.type[tagnum]) {
            case 'c':
            case 's':
            case 'u':
               typestring = G__type2string(type, tagnum, G__get_typenum(ifunc.TypeOf().ReturnType()), 0, 0);
               if (reftype) {
                  fprintf(fp, "(%s&) u[0].lval", typestring);
               }
               else {
                  if (G__globalcomp == G__CPPLINK) {
                     fprintf(fp, "*(%s*) u[0].lval", typestring);
                  }
                  else {
                     fprintf(fp, "(%s*) u[0].lval", typestring);
                  }
               }
               break;
            default:
               fprintf(fp, "u[0].lval");
               break;
         }
         break;
      default:
         break;
   }
}
#endif // __x86_64__ && (__linux || __APPLE__)

//______________________________________________________________________________
void Cint::Internal::G__cppif_genconstructor(FILE* fp, FILE* /*hfp*/, int tagnum, const ::Reflex::Member& memfunc)
{
   // -- FIXME: Describe this function!
   // Write a special constructor wrapper that handles placement new
   // using G__getgvp().  All calls to the constructor get routed here and we
   // eventually call the real constructor with the appropriate arguments.
#ifndef G__SMALLOBJECT
   int k, m;
   int isprotecteddtor = G__isprotecteddestructoronelevel(tagnum);
   G__StrBuf buf_sb(G__LONGLINE);
   char *buf = buf_sb; /* 1481 */

   G__ASSERT(tagnum != -1);

   if (G__test_access(memfunc, G__PROTECTED) || G__test_access(memfunc, G__PRIVATE)) {
      sprintf(buf, "%s_PR", G__get_link_tagname(tagnum));
   }
   else {
      strcpy(buf, G__fulltagname(tagnum, 1));
   }

#ifdef G__CPPIF_EXTERNC
   G__p2f_typedef(fp, ifn, memfunc);
#endif

#ifdef G__CPPIF_STATIC
   fprintf(fp,               "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", ::G__map_cpp_funcname(memfunc));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
   fprintf(G__WINDEFfp,      "        %s @%d\n", G__map_cpp_funcname(tagnum, G__struct.name[tagnum], memfunc), ++G__nexports);
#endif
   fprintf(hfp,              "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", ::G__map_cpp_funcname(memfunc));

   fprintf(fp,               "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", ::G__map_cpp_funcname(memfunc));
#endif /* G__CPPIF_STATIC */


   fprintf(fp,               "\n{\n");


#if !defined(G__BORLAND)
   fprintf(fp,               "   %s* p = NULL;\n", G__type2string('u', tagnum, -1, 0, 0));
#else
   fprintf(fp,               "   %s* p;\n", G__type2string('u', tagnum, -1, 0, 0));
#endif


#ifndef G__VAARG_COPYFUNC
   if (G__get_funcproperties(memfunc)->entry.ansi == 2) {
      // Handle a variadic function (variable number of arguments).
      fprintf(fp,             "   G__va_arg_buf G__va_arg_bufobj;\n");
      fprintf(fp,             "   G__va_arg_put(&G__va_arg_bufobj, libp, %ld);\n", (long)memfunc.FunctionParameterSize());
   }
#endif
#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
   if (G__get_funcproperties(memfunc)->entry.ansi == 2)
      G__x8664_vararg(fp, memfunc, buf, tagnum, buf);
#endif

   G__if_ary_union(fp, memfunc);

   fprintf(fp,               "   char* gvp = (char*)G__getgvp();\n");

   bool has_a_new = G__struct.funcs[tagnum] & (G__HAS_OPERATORNEW1ARG | G__HAS_OPERATORNEW2ARG);
   bool has_a_new1arg = G__struct.funcs[tagnum] & G__HAS_OPERATORNEW1ARG;
   bool has_a_new2arg = G__struct.funcs[tagnum] & G__HAS_OPERATORNEW2ARG;

   bool has_own_new1arg = false;
   bool has_own_new2arg = false;

   {
      struct G__ifunc_table* ifunc;
      long index;
      long offset;
      ifunc = G__get_methodhandle("operator new", "size_t", (G__ifunc_table*)G__Dict::GetDict().GetScope(tagnum).Id(), &index, &offset, 0, 0);
      has_own_new1arg = (ifunc != 0);
      ifunc = G__get_methodhandle("operator new", "size_t, void*", (G__ifunc_table*)G__Dict::GetDict().GetScope(tagnum).Id(), &index, &offset, 0, 0);
      has_own_new2arg = (ifunc != 0);
   }

   //FIXME: debugging code
   //fprintf(fp,               "   //\n");
   //fprintf(fp,               "   //has_a_new1arg: %d\n", has_a_new1arg);
   //fprintf(fp,               "   //has_a_new2arg: %d\n", has_a_new2arg);
   //fprintf(fp,               "   //has_own_new1arg: %d\n", has_own_new1arg);
   //fprintf(fp,               "   //has_own_new2arg: %d\n", has_own_new2arg);
   //fprintf(fp,               "   //\n");

   m = memfunc.FunctionParameterSize() ;

   if ((m > 0) && memfunc.FunctionParameterDefaultAt(m - 1).c_str()[0]) {
      // Handle a constructor with arguments where some of them are defaulted.
      fprintf(fp,             "   switch (libp->paran) {\n");
      do {
         fprintf(fp,           "   case %d:\n", m);
         if (m == 0) {
            // Caller gave us no arguments.
            //
            // Handle array new.
            fprintf(fp,         "     int n = G__getaryconstruct();\n");
            fprintf(fp,         "     if (n) {\n");
            if (isprotecteddtor) {
               fprintf(fp,       "       p = 0;\n");
               fprintf(fp,       "       G__genericerror(\"Error: Array construction with private/protected destructor is illegal\");\n");
            }
            else {
               fprintf(fp,       "       if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
               if (!has_a_new) {
                  fprintf(fp,     "         p = new %s[n];\n", buf);
               }
               else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
                  fprintf(fp,     "         p = new %s[n];\n", buf);
               }
               else {
                  fprintf(fp,     "         p = ::new %s[n];\n", buf);
               }
               fprintf(fp,       "       } else {\n");
               if (!has_a_new) {
                  fprintf(fp,     "         p = new((void*) gvp) %s[n];\n", buf);
               }
               else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
                  fprintf(fp,     "         p = new((void*) gvp) %s[n];\n", buf);
               }
               else {
                  fprintf(fp,     "         p = ::new((void*) gvp) %s[n];\n", buf);
               }
               fprintf(fp,       "       }\n");
            }
            fprintf(fp,         "     } else {\n");
            // Handle regular new.
            fprintf(fp,         "       if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
            if (!has_a_new) {
               fprintf(fp,         "         p = new %s;\n", buf);
            }
            else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
               fprintf(fp,       "         p = new %s;\n", buf);
            }
            else {
               fprintf(fp,       "         p = ::new %s;\n", buf);
            }
            fprintf(fp,         "       } else {\n");
            if (!has_a_new) {
               fprintf(fp,       "         p = new((void*) gvp) %s;\n", buf);
            }
            else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
               fprintf(fp,       "         p = new((void*) gvp) %s;\n", buf);
            }
            else {
               fprintf(fp,       "         p = ::new((void*) gvp) %s;\n", buf);
            }
            fprintf(fp,         "       }\n");
            fprintf(fp,         "     }\n");
         }
         else {
            // Caller gave us some of the arguments.
            //
            // Note: We do not have to handle array new here because there
            //       can be no initializer in an array new.
            fprintf(fp,         "     //m: %d\n", m);
            fprintf(fp,         "     if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
            if (!has_a_new) {
               fprintf(fp,         "       p = new %s(", buf);
            }
            else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
               fprintf(fp,       "       p = new %s(", buf);
            }
            else {
               fprintf(fp,       "       p = ::new %s(", buf);
            }
            if (m > 2) {
               fprintf(fp,       "\n");
            }
            // Copy in the arguments the caller gave us.
            for (k = 0; k < m; ++k) {
               G__cppif_paratype(fp, memfunc, k);
            }
            if (G__get_funcproperties(memfunc)->entry.ansi == 2 && memfunc.FunctionParameterSize()) {
               // Handle a variadic constructor (varying number of arguments).
#if defined(G__VAARG_COPYFUNC)
               fprintf(fp,       ", libp, %d", k);
#elif defined(__hpux)
               //FIXME:  This loops only 99 times, the other clause loops 100 times.
               int i;
               for (i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
                  fprintf(fp,     ", G__va_arg_bufobj.x.i[%d]", i);
               }
#elif (defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || \
       defined(__SUNPRO_CC)) || \
      ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
               int i;
               for (i = 0; i < 100; ++i) {
                  fprintf(fp,     ", G__va_arg_bufobj.x.i[%d]", i);
               }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
               // see G__x8664_vararg()
#else
               fprintf(fp,       ", G__va_arg_bufobj");
#endif
            }
            fprintf(fp,         ");\n");
            fprintf(fp,         "     } else {\n");
            if (!has_a_new) {
               fprintf(fp,       "       p = new((void*) gvp) %s(", buf);
            }
            else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
               fprintf(fp,       "       p = new((void*) gvp) %s(", buf);
            }
            else {
               fprintf(fp,       "       p = ::new((void*) gvp) %s(", buf);
            }
            if (m > 2) {
               fprintf(fp,       "\n");
            }
            // Copy in the arguments the caller gave us.
            for (k = 0; k < m; ++k) {
               G__cppif_paratype(fp, memfunc, k);
            }
            if (G__get_funcproperties(memfunc)->entry.ansi == 2 && memfunc.FunctionParameterSize()) {
               // Handle a variadic constructor (varying number of arguments).
#if defined(G__VAARG_COPYFUNC)
               fprintf(fp,       ", libp, %d", k);
#elif defined(__hpux)
               //FIXME:  This loops only 99 times, the other clause loops 100 times.
               int i;
               for (i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
                  fprintf(fp,     ", G__va_arg_bufobj.x.i[%d]", i);
               }
#elif (defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || \
       defined(__SUNPRO_CC)) || \
      ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
               int i;
               for (i = 0; i < 100; ++i) {
                  fprintf(fp,     ", G__va_arg_bufobj.x.i[%d]", i);
               }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
               // see G__x8664_vararg()
#else
               fprintf(fp,       ", G__va_arg_bufobj");
#endif
            }
            fprintf(fp,         ");\n");
            fprintf(fp,         "     }\n");
         }
         fprintf(fp,           "     break;\n");
         --m;
      }
      while ((m >= 0) && memfunc.FunctionParameterDefaultAt(m).c_str()[0]);
      fprintf(fp,             "   }\n");
   }
   else if (m > 0) {
      // Handle a constructor with arguments where none of them are defaulted.
      //
      // Note: We do not have to handle an array new here because initializers
      //       are not allowed for array new.
      fprintf(fp,             "   //m: %d\n", m);
      fprintf(fp,             "   if ((gvp == (char*) G__PVOID) || (gvp == 0)) {\n");
      if (!has_a_new) {
         fprintf(fp,             "     p = new %s(", buf);
      }
      else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
         fprintf(fp,           "     p = new %s(", buf);
      }
      else {
         fprintf(fp,           "     p = ::new %s(", buf);
      }
      if (m > 2) {
         fprintf(fp,           "\n");
      }
      for (k = 0; k < m; ++k) {
         G__cppif_paratype(fp, memfunc, k);
      }
      if (G__get_funcproperties(memfunc)->entry.ansi == 2 && memfunc.FunctionParameterSize()) {
         // handle a varadic constructor (varying number of arguments)
#if defined(G__VAARG_COPYFUNC)
         fprintf(fp,           ", libp, %d", k);
#elif defined(__hpux)
         //FIXME:  This loops only 99 times, the other clause loops 100 times.
         int i;
         for (i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
            fprintf(fp,         ", G__va_arg_bufobj.x.i[%d]", i);
         }
#elif (defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || \
       defined(__SUNPRO_CC)) || \
    ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
         int i;
         for (i = 0; i < 100; ++i) {
            fprintf(fp,         ", G__va_arg_bufobj.x.i[%d]", i);
         }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
         // see G__x8664_vararg()
#else
         fprintf(fp,           ", G__va_arg_bufobj");
#endif
      }
      fprintf(fp,             ");\n");
      fprintf(fp,             "   } else {\n");
      if (!has_a_new) {
         fprintf(fp,           "     p = new((void*) gvp) %s(", buf);
      }
      else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
         fprintf(fp,           "     p = new((void*) gvp) %s(", buf);
      }
      else {
         fprintf(fp,           "     p = ::new((void*) gvp) %s(", buf);
      }
      if (m > 2) {
         fprintf(fp,           "\n");
      }
      for (k = 0; k < m; ++k) {
         G__cppif_paratype(fp, memfunc, k);
      }
      if (G__get_funcproperties(memfunc)->entry.ansi == 2 && memfunc.FunctionParameterSize()) {
         // handle a varadic constructor (varying number of arguments)
#if defined(G__VAARG_COPYFUNC)
         fprintf(fp,           ", libp, %d", k);
#elif defined(__hpux)
         //FIXME:  This loops only 99 times, the other clause loops 100 times.
         int i;
         for (i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
            fprintf(fp,         ", G__va_arg_bufobj.x.i[%d]", i);
         }
#elif (defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || \
       defined(__SUNPRO_CC)) || \
    ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
         int i;
         for (i = 0; i < 100; ++i) {
            fprintf(fp,         ", G__va_arg_bufobj.x.i[%d]", i);
         }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
         // see G__x8664_vararg()
#else
         fprintf(fp,           ", G__va_arg_bufobj");
#endif
      }
      fprintf(fp,             ");\n");
      fprintf(fp,             "   }\n");
   }
   else {
      // Handle a constructor with no arguments.
      //
      // Handle array new.
      fprintf(fp,             "   int n = G__getaryconstruct();\n");
      fprintf(fp,             "   if (n) {\n");
      if (isprotecteddtor) {
         fprintf(fp,           "     p = 0;\n");
         fprintf(fp,           "     G__genericerror(\"Error: Array construction with private/protected destructor is illegal\");\n");
      }
      else {
         fprintf(fp,           "     if ((gvp == (char*) G__PVOID) || (gvp == 0)) {\n");
         if (!has_a_new) {
            fprintf(fp,         "       p = new %s[n];\n", buf);
         }
         else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
            fprintf(fp,         "       p = new %s[n];\n", buf);
         }
         else {
            fprintf(fp,         "       p = ::new %s[n];\n", buf);
         }
         fprintf(fp,           "     } else {\n");
         if (!has_a_new) {
            fprintf(fp,         "       p = new((void*) gvp) %s[n];\n", buf);
         }
         else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
            fprintf(fp,         "       p = new((void*) gvp) %s[n];\n", buf);
         }
         else {
            fprintf(fp,         "       p = ::new((void*) gvp) %s[n];\n", buf);
         }
         fprintf(fp,           "     }\n");
      }
      fprintf(fp,             "   } else {\n");
      //
      // Handle regular new.
      fprintf(fp,             "     if ((gvp == (char*) G__PVOID) || (gvp == 0)) {\n");
      if (!has_a_new) {
         fprintf(fp,             "       p = new %s;\n", buf);
      }
      else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
         fprintf(fp,           "       p = new %s;\n", buf);
      }
      else {
         fprintf(fp,           "       p = ::new %s;\n", buf);
      }
      fprintf(fp,             "     } else {\n");
      if (!has_a_new) {
         fprintf(fp,           "       p = new((void*) gvp) %s;\n", buf);
      }
      else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
         fprintf(fp,           "       p = new((void*) gvp) %s;\n", buf);
      }
      else {
         fprintf(fp,           "       p = ::new((void*) gvp) %s;\n", buf);
      }
      fprintf(fp,             "     }\n");
      fprintf(fp,             "   }\n");
   }

   fprintf(fp,               "   result7->obj.i = (long) p;\n");
   fprintf(fp,               "   result7->ref = (long) p;\n");
   fprintf(fp,               "   G__set_tagnum(result7,G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(tagnum));

   G__cppif_dummyfuncname(fp);

   fprintf(fp,               "}\n\n");
#endif // G__SMALLOBJECT
   // --
}

//______________________________________________________________________________
#ifndef __CINT__
static int G__isprivateconstructorifunc(int tagnum, int iscopy);
static int G__isprivateconstructorifunc(int tagnum, int iscopy);
static int G__isprivateconstructorclass(int tagnum, int iscopy);
#endif

//______________________________________________________________________________
static int G__isprivateconstructorifunc(int tagnum, int iscopy)
{
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
   for (
      ::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
      ifunc != scope.FunctionMember_End();
      ++ifunc
   ) {
      if (ifunc->IsConstructor()) {
         if (iscopy) { // Check copy constructor
            if (ifunc->IsCopyConstructor() && ifunc->IsPrivate()) {
               return 1;
            }
         }
         else { // Check default constructor
            if (
               (
                  !ifunc->FunctionParameterSize() ||
                  ifunc->FunctionParameterDefaultAt(0).c_str()[0]
               ) &&
               ifunc->IsPrivate()
            ) {
               return 1;
            }
            // Following solution may not be perfect
            if (ifunc->IsCopyConstructor() && ifunc->IsPrivate()) {
               return 1;
            }
         }
      }
      else if (!strcmp("operator new", ifunc->Name().c_str())) {
         if (ifunc->IsPrivate() || ifunc->IsProtected()) {
            return 1;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
static int G__isprivateconstructorvar(int scope_tagnum, int iscopy)
{
   // -- Check if private constructor exists in this particular class.
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(scope_tagnum);
   for (
      ::Reflex::Member_Iterator var = scope.DataMember_Begin();
      var != scope.DataMember_End();
      ++var
   ) {
      ::Reflex::Type var_type = var->TypeOf();
      ::Reflex::Type var_rawtype = var_type.RawType();
      if (
         (G__get_type(var_type) == 'u') && // Note: This means we do not follow pointers.
         var_rawtype &&
         !var_rawtype.IsEnum() &&
         (var_rawtype != scope) &&
         !var_type.FinalType().IsReference()
      ) {
         int tagnum = G__get_tagnum(var_rawtype);
         if (G__isprivateconstructorclass(tagnum, iscopy)) {
            return 1;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
static int G__isprivateconstructorclass(int tagnum, int iscopy)
{
   // -- Check if private constructor exists in this particular class.
   int t = 0;
   int f = 0;
   if (iscopy) {
      t = G__CTORDTOR_PRIVATECOPYCTOR;
      f = G__CTORDTOR_NOPRIVATECOPYCTOR;
   }
   else {
      t = G__CTORDTOR_PRIVATECTOR;
      f = G__CTORDTOR_NOPRIVATECTOR;
   }
   if (G__ctordtor_status[tagnum] & t) {
      return 1;
   }
   if (G__ctordtor_status[tagnum] & f) {
      return 0;
   }
   if (G__isprivateconstructorifunc(tagnum, iscopy) || G__isprivateconstructor(tagnum, iscopy)) {
      G__ctordtor_status[tagnum] |= t;
      return 1;
   }
   G__ctordtor_status[tagnum] |= f;
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__isprivateconstructor(int tagnum, int iscopy)
{
   // -- Check if private constructor exists in base class or class of member obj.
   // Check base class private constructor
   struct G__inheritance* baseclass = G__struct.baseclass[tagnum];
   for (size_t basen = 0; basen < baseclass->vec.size(); ++basen) {
      int basetagnum = baseclass->vec[basen].basetagnum;
      if (G__isprivateconstructorclass(basetagnum, iscopy)) {
         return 1;
      }
   }
   // Check Data member object
   if (G__isprivateconstructorvar(tagnum, iscopy)) {
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
#ifndef __CINT__
static int G__isprivatedestructorclass(int tagnum);
static int G__isprivatedestructor(int tagnum);
#endif

//______________________________________________________________________________
static int G__isprivatedestructorifunc(int tagnum)
{
   int ret = 0;
   char* dtorname = (char*) malloc(strlen(G__struct.name[tagnum]) + 2);
   dtorname[0] = '~';
   strcpy(dtorname + 1, G__struct.name[tagnum]);
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
   for (
      ::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
      ifunc != scope.FunctionMember_End();
      ++ifunc
   ) {
      if (ifunc->IsDestructor() || !strcmp(dtorname, ifunc->Name().c_str())) {
         if (ifunc->IsPrivate()) {
            ret = 1;
            break;
         }
      }
      else if (!strcmp("operator delete", ifunc->Name().c_str())) {
         if (ifunc->IsPrivate() || ifunc->IsProtected()) {
            ret = 1;
            break;
         }
      }
   }
   free(dtorname);
   return ret;
}

//______________________________________________________________________________
static int G__isprivatedestructorvar(int scope_tagnum)
{
   // -- Check if private destructor exists in this particular class.
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(scope_tagnum);
   for (
      ::Reflex::Member_Iterator var = scope.DataMember_Begin();
      var != scope.DataMember_End();
      ++var
   ) {
      ::Reflex::Type var_type = var->TypeOf();
      ::Reflex::Type var_rawtype = var_type.RawType();
      if (
         (G__get_type(var_type) == 'u') && // Note: This means we do not follow pointers.
         var_rawtype &&
         !var_rawtype.IsEnum() &&
         (var_rawtype != scope) &&
         !var_type.FinalType().IsReference()
      ) {
         int tagnum = G__get_tagnum(var_rawtype);
         if (G__isprivatedestructorclass(tagnum)) {
            return 1;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
static int G__isprivatedestructorclass(int tagnum)
{
   // -- Check if private destructor exists in this particular class.
   int t = G__CTORDTOR_PRIVATEDTOR;
   int f = G__CTORDTOR_NOPRIVATEDTOR;
   if (G__ctordtor_status[tagnum] & t) {
      return 1;
   }
   if (G__ctordtor_status[tagnum] & f) {
      return 0;
   }
   if (G__isprivatedestructorifunc(tagnum) || G__isprivatedestructor(tagnum)) {
      G__ctordtor_status[tagnum] |= t;
      return 1;
   }
   G__ctordtor_status[tagnum] |= f;
   return 0;
}

//______________________________________________________________________________
static int G__isprivatedestructor(int tagnum)
{
   // -- Check if private destructor exists in base class or class of member obj.
   // Check base class private destructor
   struct G__inheritance* baseclass = G__struct.baseclass[tagnum];
   for (size_t basen = 0; basen < baseclass->vec.size(); ++basen) {
      int basetagnum = baseclass->vec[basen].basetagnum;
      if (G__isprivatedestructorclass(basetagnum)) {
         return 1;
      }
   }
   // Check Data member object
   if (G__isprivatedestructorvar(tagnum)) {
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
#ifdef G__DEFAULTASSIGNOPR
#ifndef __CINT__
static int G__isprivateassignoprifunc(const ::Reflex::Type& scope);
static int G__isprivateassignoprclass(int tagnum);
static int G__isprivateassignopr(int tagnum);
#endif // __CINT__
#endif // G__DEFAULTASSIGNOPR

#ifdef G__DEFAULTASSIGNOPR
//______________________________________________________________________________
static int G__isprivateassignoprifunc(const ::Reflex::Type& scope)
{
   for (
      ::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
      ifunc != scope.FunctionMember_End();
      ++ifunc
   ) {
      if (!strcmp("operator=", ifunc->Name().c_str())) {
         ::Reflex::Type func_type = ifunc->TypeOf();
         if (
            (ifunc->IsPrivate() || ifunc->IsProtected()) &&
            func_type.FunctionParameterAt(0).RawType().IsClass() &&
            (func_type.FunctionParameterAt(0).RawType() == scope)
         ) {
            return 1;
         }
      }
   }
   return 0;
}
#endif // G__DEFAULTASSIGNOPR

#ifdef G__DEFAULTASSIGNOPR
//______________________________________________________________________________
static int G__isprivateassignoprvar(int given_tagnum)
{
   // Check if private operator= exists in this particular class.
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(given_tagnum);
   for (
      ::Reflex::Member_Iterator mbr_iter = scope.DataMember_Begin();
      mbr_iter != scope.DataMember_End();
      ++mbr_iter
   ) {
      char type = '\0';
      int tagnum = -1;
      int typenum = -1;
      int reftype = 0;
      int isconst = 0;
      G__get_cint5_type_tuple(mbr_iter->TypeOf(), &type, &tagnum, &typenum, &reftype, &isconst);
      if (
         (type == 'u') && // mbr is of class type, and
         (tagnum != -1) && // mbr tag is valid, and
         !mbr_iter->TypeOf().RawType().IsEnum() && // mbr is not of enum type, and
         (tagnum != given_tagnum) && // mbr class tag does not match given tag, and
         (reftype != G__PARAREFERENCE) // mbr is not a reference
      ) {
         if (G__isprivateassignoprclass(tagnum)) {
            return 1;
         }
      }
      if (
         (reftype == G__PARAREFERENCE) && // mbr is a reference, and
         (G__get_properties(*mbr_iter)->statictype != G__LOCALSTATIC) // not static
      ) {
         return 1;
      }
      if (
         isconst && // mbr is const, and
         (G__get_properties(*mbr_iter)->statictype != G__LOCALSTATIC) // not static
      ) {
         return 1;
      }
   }
   return 0;
}
#endif // G__DEFAULTASSIGNOPR

#ifdef G__DEFAULTASSIGNOPR
//______________________________________________________________________________
static int G__isprivateassignoprclass(int tagnum)
{
   // -- Check if private assignopr exists in this particular class.
   int t = G__CTORDTOR_PRIVATEASSIGN;
   int f = G__CTORDTOR_NOPRIVATEASSIGN;
   if (G__ctordtor_status[tagnum] & t) {
      return 1;
   }
   if (G__ctordtor_status[tagnum] & f) {
      return 0;
   }
   ::Reflex::Type type(G__Dict::GetDict().GetType(tagnum));
   if (G__isprivateassignoprifunc(type) || G__isprivateassignopr(tagnum)) {
      G__ctordtor_status[tagnum] |= t;
      return 1;
   }
   G__ctordtor_status[tagnum] |= f;
   return 0;
}
#endif // G__DEFAULTASSIGNOPR

#ifdef G__DEFAULTASSIGNOPR
//______________________________________________________________________________
static int G__isprivateassignopr(int tagnum)
{
   // -- Check if private assignopr exists in base class or class of member obj.
   // Check base class private assignopr
   struct G__inheritance* baseclass = G__struct.baseclass[tagnum];
   for (size_t basen = 0; basen < baseclass->vec.size(); ++basen) {
      int basetagnum = baseclass->vec[basen].basetagnum;
      if (G__isprivateassignoprclass(basetagnum)) {
         return 1;
      }
   }
   // Check Data member object
   if (G__isprivateassignoprvar(tagnum)) {
      return 1;
   }
   return 0;
}
#endif // G__DEFAULTASSIGNOPR

//______________________________________________________________________________
void Cint::Internal::G__cppif_gendefault(FILE* fp, FILE* /*hfp*/, int tagnum, int isconstructor, int iscopyconstructor, int isdestructor, int isassignmentoperator, int isnonpublicnew)
{
   // Create default constructor and destructor. If default constructor is
   // given in the header file, the interface function created here for
   // the default constructor will be redundant and won't be used.
   //
   // Copy constructor and operator=(), if not explisitly specified in the
   // header file, are handled as memberwise copy by cint parser. Validity of
   // this handling is questionalble especially when base class has explicit
   // copy constructor or operator=().
#ifndef G__SMALLOBJECT
#define G__OLDIMPLEMENtATION1972
#ifndef G__OLDIMPLEMENtATION1972
   G__StrBuf buf1_sb(G__MAXNAME);
   char *buf1 = buf1_sb;
   G__StrBuf buf2_sb(G__MAXNAME);
   char *buf2 = buf2_sb;
   G__StrBuf buf3_sb(G__MAXNAME);
   char *buf3 = buf3_sb;
   char *funcname = buf1;
   char *temp = buf2;
   char *dtorname = buf3;
#else
   G__StrBuf funcname_sb(G__MAXNAME*6);
   char *funcname = funcname_sb;
   G__StrBuf temp_sb(G__MAXNAME*6);
   char *temp = temp_sb;
#endif
   int isprotecteddtor = G__isprotecteddestructoronelevel(tagnum);
#ifdef G__OLDIMPLEMENtATION1972
   G__StrBuf dtorname_sb(G__LONGLINE);
   char *dtorname = dtorname_sb;
#endif

   G__ASSERT(tagnum != -1);

   {
      ::Reflex::Scope scope(G__Dict::GetDict().GetScope(tagnum));
      if (!scope || scope.IsNamespace()) {
         return;
      }
      if (G__struct.type[tagnum] == 'n') { // FIXME: This is a hack until we fix the problem with G__search_tagname creating autoloading entries as Reflex::Class even for namespaces.
         return;
      }
   }

   int extra_pages = 0;

#ifndef G__OLDIMPLEMENtATION1972
   if (strlen(G__struct.name[tagnum]) > G__MAXNAME - 2) {
      funcname = (char*)malloc(strlen(G__struct.name[tagnum]) + 5);
      dtorname = (char*)malloc(strlen(G__struct.name[tagnum]) + 5);
   }
   if (strlen(G__fulltagname(tagnum, 1)) > G__MAXNAME - 2) {
      dtorname = (char*)malloc(strlen(G__fulltagname(tagnum, 1)) + 5);
   }
#endif

   /*********************************************************************
   * default constructor
   *********************************************************************/

   if (!isconstructor) {
      isconstructor = G__isprivateconstructor(tagnum, 0);
   }

   if (!isconstructor && !G__struct.isabstract[tagnum] && !isnonpublicnew) {

      G__StrBuf buf_sb(G__LONGLINE);
      char *buf = buf_sb;
      strcpy(buf, G__fulltagname(tagnum, 1));

      strcpy(funcname, G__struct.name[tagnum]);
      fprintf(fp,         "// automatic default constructor\n");

#ifdef G__CPPIF_STATIC
      fprintf(fp,         "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
      fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages), ++G__nexports);
#endif
      fprintf(hfp,        "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
      fprintf(fp,         "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
#endif /* G__CPPIF_STATIC */
      fprintf(fp,         "\n{\n");
      fprintf(fp,         "   %s *p;\n", G__fulltagname(tagnum, 1));

      fprintf(fp,         "   char* gvp = (char*)G__getgvp();\n");

      bool has_a_new = G__struct.funcs[tagnum] & (G__HAS_OPERATORNEW1ARG | G__HAS_OPERATORNEW2ARG);
      bool has_a_new1arg = G__struct.funcs[tagnum] & G__HAS_OPERATORNEW1ARG;
      bool has_a_new2arg = G__struct.funcs[tagnum] & G__HAS_OPERATORNEW2ARG;

      bool has_own_new1arg = false;
      bool has_own_new2arg = false;

      {
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
         struct G__ifunc_table* ifunc;
         long index;
         long offset;
         ifunc = G__get_methodhandle("operator new", "size_t", (G__ifunc_table*)scope.Id(), &index, &offset, 0, 0);
         has_own_new1arg = (ifunc != 0);
         ifunc = G__get_methodhandle("operator new", "size_t, void*", (G__ifunc_table*)scope.Id(), &index, &offset, 0, 0);
         has_own_new2arg = (ifunc != 0);
      }

      //FIXME: debugging code
      //fprintf(fp,         "   //\n");
      //fprintf(fp,         "   //has_a_new1arg: %d\n", has_a_new1arg);
      //fprintf(fp,         "   //has_a_new2arg: %d\n", has_a_new2arg);
      //fprintf(fp,         "   //has_own_new1arg: %d\n", has_own_new1arg);
      //fprintf(fp,         "   //has_own_new2arg: %d\n", has_own_new2arg);
      //fprintf(fp,         "   //\n");

      //
      // Handle array new.
      fprintf(fp,         "   int n = G__getaryconstruct();\n");
      fprintf(fp,         "   if (n) {\n");
      if (isprotecteddtor) {
         fprintf(fp,       "     p = 0;\n");
         fprintf(fp,       "     G__genericerror(\"Error: Array construction with private/protected destructor is illegal\");\n");
      }
      else {
         fprintf(fp,       "     if ((gvp == (char*) G__PVOID) || (gvp == 0)) {\n");
         if (!has_a_new) {
            fprintf(fp,     "       p = new %s[n];\n", buf);
         }
         else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
            fprintf(fp,     "       p = new %s[n];\n", buf);
         }
         else {
            fprintf(fp,     "       p = ::new %s[n];\n", buf);
         }
         fprintf(fp,       "     } else {\n");
         if (!has_a_new) {
            fprintf(fp,     "       p = new((void*) gvp) %s[n];\n", buf);
         }
         else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
            fprintf(fp,     "       p = new((void*) gvp) %s[n];\n", buf);
         }
         else {
            fprintf(fp,     "       p = ::new((void*) gvp) %s[n];\n", buf);
         }
         fprintf(fp,       "     }\n");
      }
      fprintf(fp,         "   } else {\n");
      //
      // Handle regular new.
      fprintf(fp,         "     if ((gvp == (char*) G__PVOID) || (gvp == 0)) {\n");
      if (!has_a_new) {
         fprintf(fp,       "       p = new %s;\n", buf);
      }
      else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
         fprintf(fp,       "       p = new %s;\n", buf);
      }
      else {
         fprintf(fp,       "       p = ::new %s;\n", buf);
      }
      fprintf(fp,         "     } else {\n");
      if (!has_a_new) {
         fprintf(fp,       "       p = new((void*) gvp) %s;\n", buf);
      }
      else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
         fprintf(fp,       "       p = new((void*) gvp) %s;\n", buf);
      }
      else {
         fprintf(fp,       "       p = ::new((void*) gvp) %s;\n", buf);
      }
      fprintf(fp,         "     }\n");
      fprintf(fp,         "   }\n");

      fprintf(fp,         "   result7->obj.i = (long) p;\n");
      fprintf(fp,         "   result7->ref = (long) p;\n");
      fprintf(fp,         "   G__set_tagnum(result7,G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(tagnum));

      G__cppif_dummyfuncname(fp);

      fprintf(fp,         "}\n\n");

      ++extra_pages;
   } /* if (isconstructor) */

   /*********************************************************************
   * copy constructor
   *********************************************************************/

   if (!iscopyconstructor) {
      iscopyconstructor = G__isprivateconstructor(tagnum, 1);
   }

   if (!iscopyconstructor && !G__struct.isabstract[tagnum] && !isnonpublicnew) {

      sprintf(funcname, "%s", G__struct.name[tagnum]);

      fprintf(fp,     "// automatic copy constructor\n");

#ifdef G__CPPIF_STATIC
      fprintf(fp,     "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
      fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages), ++G__nexports);
#endif
      fprintf(hfp,    "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
      fprintf(fp,    "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
#endif /* G__CPPIF_STATIC */

      fprintf(fp,     "\n{\n");
      fprintf(fp,     "   %s* p;\n", G__fulltagname(tagnum, 1));

      strcpy(temp, G__fulltagname(tagnum, 1));

      fprintf(fp,     "   void* tmp = (void*) G__int(libp->para[0]);\n");
      fprintf(fp,     "   p = new %s(*(%s*) tmp);\n", temp, temp);

      fprintf(fp,     "   result7->obj.i = (long) p;\n");
      fprintf(fp,     "   result7->ref = (long) p;\n");
      fprintf(fp,     "   G__set_tagnum(result7,G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(tagnum));

      G__cppif_dummyfuncname(fp);

      fprintf(fp,     "}\n\n");

      ++extra_pages;
   }


   /*********************************************************************
   * destructor
   *********************************************************************/

   if (0 >= isdestructor) {
      isdestructor = G__isprivatedestructor(tagnum);
   }

   if ((0 >= isdestructor) && (G__struct.type[tagnum] != 'n')) {

      G__StrBuf buf_sb(G__LONGLINE);
      char *buf = buf_sb;
      strcpy(buf, G__fulltagname(tagnum, 1));

      bool has_a_delete = G__struct.funcs[tagnum] & G__HAS_OPERATORDELETE;

      bool has_own_delete1arg = false;
      bool has_own_delete2arg = false;

      {
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
         struct G__ifunc_table* ifunc;
         long index;
         long offset;
         ifunc = G__get_methodhandle("operator delete", "void*", (G__ifunc_table*)scope.Id(), &index, &offset, 0, 0);
         has_own_delete1arg = (ifunc != 0);
         ifunc = G__get_methodhandle("operator delete", "void*, size_t", (G__ifunc_table*)scope.Id(), &index, &offset, 0, 0);
         has_own_delete2arg = (ifunc != 0);
      }

      sprintf(funcname, "~%s", G__struct.name[tagnum]);
      sprintf(dtorname, "G__T%s", G__map_cpp_name(G__fulltagname(tagnum, 0)));

      fprintf(fp, "// automatic destructor\n");
      fprintf(fp, "typedef %s %s;\n", G__fulltagname(tagnum, 0), dtorname);

#ifdef G__CPPIF_STATIC
      fprintf(fp, "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
      fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages), ++G__nexports);
#endif
      fprintf(hfp, "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
      fprintf(fp, "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
#endif /* G__CPPIF_STATIC */

      fprintf(fp,   "\n{\n");
      fprintf(fp,   "   char* gvp = (char*) G__getgvp();\n");
      fprintf(fp,   "   long soff = G__getstructoffset();\n");
      fprintf(fp,   "   int n = G__getaryconstruct();\n");

      fprintf(fp,   "   //\n");
      fprintf(fp,   "   //has_a_delete: %d\n", has_a_delete);
      fprintf(fp,   "   //has_own_delete1arg: %d\n", has_own_delete1arg);
      fprintf(fp,   "   //has_own_delete2arg: %d\n", has_own_delete2arg);
      fprintf(fp,   "   //\n");

      fprintf(fp,   "   if (!soff) {\n");
      fprintf(fp,   "     return(1);\n");
      fprintf(fp,   "   }\n");

      fprintf(fp,   "   if (n) {\n");
      fprintf(fp,   "     if (gvp == (char*)G__PVOID) {\n");
      fprintf(fp,   "       delete[] (%s*) soff;\n", buf);
      fprintf(fp,   "     } else {\n");
      fprintf(fp,   "       G__setgvp((long) G__PVOID);\n");
      fprintf(fp,   "       for (int i = n - 1; i >= 0; --i) {\n");
      fprintf(fp,   "         ((%s*) (soff+(sizeof(%s)*i)))->~%s();\n", buf, buf, dtorname);
      fprintf(fp,   "       }\n");
      fprintf(fp,   "       G__setgvp((long)gvp);\n");
      fprintf(fp,   "     }\n");
      fprintf(fp,   "   } else {\n");
      fprintf(fp,   "     if (gvp == (char*) G__PVOID) {\n");
      //fprintf(fp, "       G__operator_delete((void*) soff);\n");
      fprintf(fp,   "       delete (%s*) soff;\n", buf);
      fprintf(fp,   "     } else {\n");
      fprintf(fp,   "       G__setgvp((long) G__PVOID);\n");
      fprintf(fp,   "       ((%s*) (soff))->~%s();\n", buf, dtorname);
      fprintf(fp,   "       G__setgvp((long)gvp);\n");
      fprintf(fp,   "     }\n");
      fprintf(fp,   "   }\n");

      fprintf(fp,   "   G__setnull(result7);\n");

      G__cppif_dummyfuncname(fp);

      fprintf(fp,   "}\n\n");

      ++extra_pages;
   }


#ifdef G__DEFAULTASSIGNOPR

   /*********************************************************************
   * assignment operator
   *********************************************************************/

   if (!isassignmentoperator) {
      isassignmentoperator = G__isprivateassignopr(tagnum);
   }

   if (!isassignmentoperator) {
      sprintf(funcname, "operator=");
      fprintf(fp,   "// automatic assignment operator\n");

#ifdef G__CPPIF_STATIC
      fprintf(fp,   "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
      fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages), ++G__nexports);
#endif
      fprintf(hfp,  "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
      fprintf(fp,  "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, 0, extra_pages));
#endif /* G__CPPIF_STATIC */
      fprintf(fp,   "\n{\n");
      strcpy(temp, G__type2string('u', tagnum, -1, 0, 0));
      fprintf(fp,   "   %s* dest = (%s*) G__getstructoffset();\n", temp, temp);
      if ((1 >= G__struct.size[tagnum]) && (0 == G__Dict::GetDict().GetScope(tagnum).DataMemberSize())) {}
      else {
         fprintf(fp, "   *dest = *(%s*) libp->para[0].ref;\n", temp);
      }
      fprintf(fp,   "   const %s& obj = *dest;\n", temp);
      fprintf(fp,   "   result7->ref = (long) (&obj);\n");
      fprintf(fp,   "   result7->obj.i = (long) (&obj);\n");
      G__cppif_dummyfuncname(fp);
      fprintf(fp,   "}\n\n");

      ++extra_pages;
   }
#endif

#ifndef G__OLDIMPLEMENtATION1972
   if (funcname != buf1) free((void*)funcname);
   if (temp != buf2) free((void*)temp);
   if (dtorname != buf3) free((void*)dtorname);
   //
#endif
   //
#endif // G__SMALLOBJECT
   // --
}

//______________________________________________________________________________
void Cint::Internal::G__cppif_genfunc(FILE* fp, FILE* /*hfp*/, int tagnum, const ::Reflex::Member& ifunc)
{
   // -- Output the stub for a function.
   //
#ifndef G__SMALLOBJECT
   int k = 0;
   int m = 0;
#ifndef G__OLDIMPLEMENTATION1823
   G__StrBuf buf2_sb(G__LONGLINE);
   char *buf2 = buf2_sb;
   char *endoffunc = buf2;
#else // G__OLDIMPLEMENTATION1823
   G__StrBuf endoffunc_sb(G__LONGLINE);
   char *endoffunc = endoffunc_sb;
#endif // G__OLDIMPLEMENTATION1823
#ifndef G__OLDIMPLEMENTATION1823
   G__StrBuf buf_sb(G__BUFLEN*4);
   char *buf = buf_sb;
   char *castname = buf;
#else // G__OLDIMPLEMENTATION1823
   G__StrBuf castname_sb(G__ONELINE);
   char *castname = castname_sb;
#endif // G__OLDIMPLEMENTATION1823
#ifndef G__OLDIMPLEMENTATION1823
   // Expand castname and endoffunc buffers if necessary.
   if (tagnum != -1) {
      int len = strlen(G__fulltagname(tagnum, 1));
      if (len > (G__BUFLEN * 4) - 30) {
         castname = (char*) malloc(len + 30);
      }
      if (len > (G__LONGLINE - 256)) {
         endoffunc = (char*) malloc(len + 256);
      }
   }
#endif // G__OLDIMPLEMENTATION1823
#ifdef G__CPPIF_EXTERNC
   G__p2f_typedef(fp, ifn, ifunc);
#endif // G__CPPIF_EXTERNC
#ifdef G__VAARG_COPYFUNC
   if ((G__get_funcproperties(ifunc)->entry.ansi == 2) && (ifunc->pentry[ifn]->line_number > 0)) {
      G__va_arg_copyfunc(fp, ifunc, ifn);
   }
#endif // G__VAARG_COPYFUNC
#ifndef G__CPPIF_STATIC
#ifdef G__GENWINDEF
   fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, ifunc.Name().c_str(), ifn, ifunc->page), ++G__nexports);
#endif // G__GENWINDEF
   if (G__globalcomp == G__CPPLINK) {
      fprintf(hfp, "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, ifunc.Name().c_str(), ifn, ifunc->page));
   }
   else {
      fprintf(hfp, "int %s();\n", G__map_cpp_funcname(tagnum, ifunc.Name().c_str(), ifn, ifunc->page));
   }
#endif // G__CPPIF_STATIC
#ifdef G__CPPIF_STATIC
   fprintf(fp, "static ");
#else // G__CPPIF_STATIC
   if (G__globalcomp == G__CPPLINK) {
      fprintf(fp, "extern \"C\" ");
   }
#endif // G__CPPIF_STATIC
   if (G__clock) {
      // -- K&R style header
      fprintf(fp, "int %s(result7, funcname, libp, hash)\n", ::G__map_cpp_funcname(ifunc));
      fprintf(fp, "G__value* result7;\n");
      fprintf(fp, "char* funcname;\n");
      fprintf(fp, "struct G__param* libp;\n");
      fprintf(fp, "int hash;\n");
   }
   else {
      // -- ANSI style header.
      fprintf(fp, "int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", ::G__map_cpp_funcname(ifunc));
   }
   fprintf(fp, "\n{\n");
   G__if_ary_union(fp, ifunc);
   if (tagnum != -1) {
      if ((ifunc.IsProtected()) || ((ifunc.IsPrivate()) && (G__struct.protectedaccess[tagnum] & G__PRIVATEACCESS))) {
         sprintf(castname, "%s_PR", G__get_link_tagname(tagnum));
      }
      else {
         strcpy(castname, G__fulltagname(tagnum, 1));
      }
   }
#ifndef G__VAARG_COPYFUNC
   if (G__get_funcproperties(ifunc)->entry.ansi == 2) {
      fprintf(fp, "   G__va_arg_buf G__va_arg_bufobj;\n");
      fprintf(fp, "   G__va_arg_put(&G__va_arg_bufobj, libp, %ld);\n", (long)ifunc.FunctionParameterSize());
   }
#endif // G__VAARG_COPYFUNC
#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
   if (G__get_funcproperties(ifunc)->entry.ansi == 2) {
      G__x8664_vararg(fp, ifunc, ifunc.Name().c_str(), tagnum, castname);
   }
#endif // __x86_64__ && (__linux  || __APPLE__)
   m = ifunc.FunctionParameterSize() ;
   if ((m > 0) && ifunc.FunctionParameterDefaultAt(m - 1).c_str()[0]) {
      // -- Handle a function with parameters, some of which have defaults.
      fprintf(fp, "   switch (libp->paran) {\n");
      do {
         // -- One case for each possible number of supplied parameters.
         fprintf(fp, "   case %d:\n", m);
         //
         // Output the return type.
         G__cppif_returntype(fp, ifunc, endoffunc);
         //
         // Output the function name.
         if (-1 != tagnum) {
            if ('n' == G__struct.type[tagnum]) {
               fprintf(fp, "%s::", G__fulltagname(tagnum, 1));
            }
            else {
               if (ifunc.IsStatic()) {
                  fprintf(fp, "%s::", castname);
               }
               else {
                  if (ifunc.IsConst()) {
                     fprintf(fp, "((const %s*) G__getstructoffset())->", castname);
                  }
                  else {
                     fprintf(fp, "((%s*) G__getstructoffset())->", castname);
                  }
               }
            }
         }
         if ((ifunc.IsProtected()) || ((ifunc.IsPrivate()) && (G__struct.protectedaccess[tagnum] & G__PRIVATEACCESS))) {
            fprintf(fp, "G__PT_%s(", ifunc.Name().c_str());
         }
         else {
            fprintf(fp, "%s(", ifunc.Name().c_str());
         }
         //
         // Output the parameters.
         if (m > 6) {
            fprintf(fp, "\n");
         }
         for (k = 0; k < m; ++k) {
            G__cppif_paratype(fp, ifunc, k);
         }
         if (G__get_funcproperties(ifunc)->entry.ansi == 2 && ifunc.FunctionParameterSize()) {
            // --
#if defined(G__VAARG_COPYFUNC)
            fprintf(fp, ", libp, %d", k);
#elif defined(__hpux)
            //FIXME:  This loops only 99 times, the other clause loops 100 times.
            for (int i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; i--) {
               fprintf(fp, ", G__va_arg_bufobj.x.i[%d]", i);
            }
#elif (defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || defined(__SUNPRO_CC)) || ((defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__)))
            for (int i = 0; i < 100; i++) fprintf(fp, ", G__va_arg_bufobj.x.i[%d]", i);
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
            // -- See G__x8664_vararg().
#else
            fprintf(fp, ", G__va_arg_bufobj");
#endif
            // --
         }
         //
         // Output the function body.
         fprintf(fp, ")%s\n", endoffunc);
         //
         // End the case for m number of parameters given.
         fprintf(fp, "      break;\n");
         --m;
      }
      while ((m >= 0) && ifunc.FunctionParameterDefaultAt(m).c_str()[0]);
      //
      // End of switch on number of parameters provided by call.
      fprintf(fp, "   }\n");
   }
   else {
      // -- Handle a function with parameters, none of which have defaults.
      //
      // Output the return type.
      G__cppif_returntype(fp, ifunc, endoffunc);
#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
      if (G__get_funcproperties(ifunc)->entry.ansi == 2 && ifunc.FunctionParameterSize()) {
         // all code is already generated by G__x8664_vararg()
         G__x8664_vararg_epilog(fp, ifunc);
         fprintf(fp, "%s\n", endoffunc);
      }
      else {
#endif // __x86_64__ && (__linux || __APPLE__)
         //
         // Output the function name.
         if (-1 != tagnum) {
            if (G__struct.type[tagnum] == 'n')
               fprintf(fp, "%s::", G__fulltagname(tagnum, 1));
            else {
               if (ifunc.IsStatic()) {
                  fprintf(fp, "%s::", castname);
               }
               else {
                  if (ifunc.IsConst()) {
                     fprintf(fp, "((const %s*) G__getstructoffset())->", castname);
                  }
                  else {
                     fprintf(fp, "((%s*) G__getstructoffset())->", castname);
                  }
               }
            }
         }
         if ((ifunc.IsProtected()) || ((ifunc.IsPrivate()) && (G__struct.protectedaccess[tagnum] & G__PRIVATEACCESS))) {
            fprintf(fp, "G__PT_%s(", ifunc.Name().c_str());
         }
         else {
            // we need to convert A::operator T() to A::operator ::T, or
            // the context will be the one of tagnum, i.e. A::T instead of ::T
            if (
               (tolower(G__get_type(ifunc.TypeOf().ReturnType())) == 'u') &&
               !strncmp(ifunc.Name().c_str(), "operator ", 9) &&
               (isalpha(ifunc.Name().c_str()[9]) || ifunc.Name().c_str()[9] == '_')
            ) {
               if (!strncmp(ifunc.Name().c_str() + 9, "const ", 6)) {
                  fprintf(fp, "operator const ::%s(", ifunc.Name().c_str() + 15);
               }
               else {
                  fprintf(fp, "operator ::%s(", ifunc.Name().c_str() + 9);
               }
            }
            else {
               fprintf(fp, "%s(", ifunc.Name().c_str());
            }
         }
         //
         // Output the parameters.
         if (m > 6) {
            fprintf(fp, "\n");
         }
         for (k = 0; k < m; k++) {
            G__cppif_paratype(fp, ifunc, k);
         }
         if (G__get_funcproperties(ifunc)->entry.ansi == 2 && ifunc.FunctionParameterSize()) {
            // --
#if defined(G__VAARG_COPYFUNC)
            fprintf(fp, ", libp, %d", k);
#elif defined(__hpux)
            //FIXME:  This loops only 99 times, the other clause loops 100 times.
            for (int i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
               fprintf(fp, ", G__va_arg_bufobj.x.i[%d]", i);
            }
#elif (defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || defined(__SUNPRO_CC)) || ((defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__)))
            for (int i = 0; i < 100; i++) {
               fprintf(fp, ", G__va_arg_bufobj.x.i[%d]", i);
            }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
            // -- See G__x8664_vararg().
#else // G__VAARG_COPYFUNC, __hpux, (__sparc || __sparc__ || __SUNPRO_C || __SUNPRO_CC) || ((__PPC__ || __ppc__) && (_AIX || __APPLE__)), __x86_64__ && (__linux || __APPLE__)
            fprintf(fp, ", G__va_arg_bufobj");
#endif // G__VAARG_COPYFUNC, __hpux, (__sparc || __sparc__ || __SUNPRO_C || __SUNPRO_CC) || ((__PPC__ || __ppc__) && (_AIX || __APPLE__)), __x86_64__ && (__linux || __APPLE__)
            // --
         }
         // Output the function body.
         fprintf(fp, ")%s\n", endoffunc);
#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__))
      }  // end G__x8664_vararg_epilog
#endif // __x86_64__ && (__linux || __APPLE__)
      // --
   }
   G__cppif_dummyfuncname(fp);
   fprintf(fp, "}\n\n");
#ifndef G__OLDIMPLEMENTATION1823
   if (castname != buf) {
      free((void*) castname);
   }
   if (endoffunc != buf2) {
      free((void*) endoffunc);
   }
#endif // G__OLDIMPLEMENTATION1823
#endif // G__SMALLOBJECT
   // --
}

//______________________________________________________________________________
int Cint::Internal::G__cppif_returntype(FILE* fp, const ::Reflex::Member& ifunc, char* endoffunc)
{
   // -- FIXME: Describe this function!
#ifndef G__SMALLOBJECT
#ifndef G__OLDIMPLEMENTATION1503
   int deftyp = -1;
#endif // G__OLDIMPLEMENTATION1503
   char* typestring = 0;
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
   char* ptr = 0;
#endif // (_MSC_VER)  && (_MSC_VER < 1310)
   const char* indent = "      ";
   ::Reflex::Type ret_type = ifunc.TypeOf().ReturnType();
   int type = G__get_type(ret_type);
   int tagnum = G__get_tagnum(ret_type.RawType());
   int typenum = G__get_typenum(ret_type);
   int reftype = G__get_reftype(ret_type);
   int isconst = G__get_isconst(ret_type);
   if (ifunc.IsConst()) {
      isconst |= G__CONSTFUNC;
   }

   /* Promote link-off typedef to link-on if used in function */
   if (
      ret_type.IsTypedef() &&
      G__get_properties(ret_type) &&
      (G__get_properties(ret_type)->globalcomp == G__NOLINK) &&
      (G__get_properties(ret_type)->iscpplink == G__NOLINK)
   ) {
      G__get_properties(ret_type)->globalcomp = G__globalcomp;
   }

#ifdef G__OLDIMPLEMENTATION1859 /* questionable with 1859 */
   /* return type is a reference */
   if (ret_type.IsTypedef() && (G__get_reftype(ret_type.ToType()) == G__PARAREFERENCE)) {
      reftype = G__PARAREFERENCE;
      typenum = -1;
   }
#endif

   // Function return type is a reference, handle and return.
   if ((reftype == G__PARAREFERENCE) || (ret_type.IsTypedef() && G__get_reftype(ret_type.FinalType()))) {
      fprintf(fp, "%s{\n", indent);
      if (isconst & G__CONSTFUNC) {
         if (isupper(type)) {
            isconst |= G__PCONSTVAR;
         }
         else {
            isconst |= G__CONSTVAR;
         }
      }
      typestring = G__type2string(type, tagnum, typenum, reftype, isconst);
      std::string rettypestring = ret_type.Name(Reflex::SCOPED | Reflex::QUALIFIED);
      if (ret_type.IsReference() && ret_type.ToType().FinalType().IsReference()) {
         // We have a reference to a reference.
         rettypestring = Reflex::Type(ret_type,Reflex::REFERENCE,Reflex::Type::MASK).Name(Reflex::SCOPED | Reflex::QUALIFIED);
      }
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
      // For old Microsoft compilers, replace "long long" with " __int64 ".
      ptr = strstr(typestring, "long long");
      if (ptr) {
         memcpy(ptr, " __int64 ", 9);
      }
      ptr = strstr(rettypestring.c_str(), "long long");
      if (ptr) {
         memcpy(ptr, " __int64 ", 9);
      }
#endif
      //
      // Output the left-hand side of the assignment.
      if (islower(type) && !isconst) {
         // Reference to a non-const object.
         // Note:  The type string already has the ampersand in it.
         fprintf(fp, "%s   const %s obj = ", indent, typestring);
      }
      else {
         // Reference to a pointer or to a const object.
         // Note:  The type string already has the ampersand in it.
         fprintf(fp, "%s   %s obj = ", indent, typestring);
      }
      if (ret_type.IsTypedef() && ret_type.ToType().FinalType().IsArray()) {
         sprintf(endoffunc, ";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.i = (long) (obj);\n%s}", indent, indent, indent);
         return 0;
      }
      switch (type) {
         case 'd':
         case 'f':
            sprintf(endoffunc, ";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.d = (double) (obj);\n%s}", indent, indent, indent);
            break;
         case 'u':
            if (G__struct.type[tagnum] == 'e') {
               sprintf(endoffunc, ";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.i = (long) (obj);\n%s}", indent, indent, indent);
            }
            else {
               sprintf(endoffunc, ";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.i = (long) (&obj);\n%s}", indent, indent, indent);
            }
            break;
         default:
            sprintf(endoffunc, ";\n%s   result7->ref = (long) (&obj);\n%s   G__letint(result7, '%c', (long)obj);\n%s}", indent, indent, type, indent);
            break;
      }
      return 0;
   }

   // Function return type is a pointer, handle and return.
   if (isupper(type) || (ret_type.IsTypedef() && ret_type.FinalType().IsPointer())) {
      fprintf(fp, "%sG__letint(result7, %d, (long) ", indent, type);
      sprintf(endoffunc, ");");
      return(0);
   }

   // Function returns an object or a fundamental type.
   if (ret_type.IsTypedef()) {
      type = G__get_type(ret_type.FinalType());
   }
   if (ret_type.IsTypedef()) {
      reftype = G__get_reftype(ret_type.FinalType());
   }
   switch (type) {
      case 'y':
         fprintf(fp, "%s", indent);
         sprintf(endoffunc, ";\n%sG__setnull(result7);", indent);
         return 0;
      case '1':
         fprintf(fp, "%sG__letint(result7, %d, (long) ", indent, type);
         sprintf(endoffunc, ");");
         return 0;
      case 'e':
      case 'c':
      case 's':
      case 'i':
      case 'l':
      case 'b':
      case 'r':
      case 'h':
      case 'k':
      case 'g':
         fprintf(fp, "%sG__letint(result7, %d, (long) ", indent, type);
         sprintf(endoffunc, ");");
         return 0;
      case 'n':
         fprintf(fp, "%sG__letLonglong(result7, %d, (G__int64) ", indent, type);
         sprintf(endoffunc, ");");
         return 0;
      case 'm':
         fprintf(fp, "%sG__letULonglong(result7, %d, (G__uint64) ", indent, type);
         sprintf(endoffunc, ");");
         return 0;
      case 'q':
         fprintf(fp, "%sG__letLongdouble(result7, %d, (long double) ", indent, type);
         sprintf(endoffunc, ");");
         return 0;
      case 'f':
      case 'd':
         fprintf(fp, "%sG__letdouble(result7, %d, (double) ", indent, type);
         sprintf(endoffunc, ");");
         return 0;
      case 'u':
         switch (G__struct.type[tagnum]) {
            case 'a':
               G__class_autoloading(&tagnum);
            case 'c':
            case 's':
            case 'u':
               deftyp = typenum;
               if (reftype) {
                  fprintf(fp, "%s{\n", indent);
                  typestring = G__type2string('u', tagnum, deftyp, 0, 0);
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and v7.0*/
                  // For old Microsoft compilers, replace "long long" by " __int64 ".
                  ptr = strstr(typestring, "long long");
                  if (ptr) {
                     memcpy(ptr, " __int64 ", 9);
                  }
#endif
                  fprintf(fp, "%sconst %s& obj = ", indent, typestring);
                  sprintf(endoffunc, ";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.i = (long) (&obj);\n%s}", indent, indent, indent);
               }
               else {
                  if (G__globalcomp == G__CPPLINK) {
                     fprintf(fp, "%s{\n", indent);
                     if (isconst & G__CONSTFUNC) {
                        fprintf(fp, "%s   const %s* pobj;\n", indent, G__type2string('u', tagnum, deftyp, 0, 0));
                        fprintf(fp, "%s   const %s xobj = ", indent, G__type2string('u', tagnum, deftyp, 0, 0));
                     }
                     else {
                        fprintf(fp, "%s   %s* pobj;\n", indent, G__type2string('u', tagnum, deftyp, 0, 0));
                        fprintf(fp, "%s   %s xobj = ", indent, G__type2string('u', tagnum, deftyp, 0, 0));
                     }
                     sprintf(endoffunc, ";\n"
                             "%s   pobj = new %s(xobj);\n"
                             "%s   result7->obj.i = (long) ((void*) pobj);\n"
                             "%s   result7->ref = result7->obj.i;\n"
                             "%s   G__store_tempobject(*result7);\n"
                             "%s}", indent, G__type2string('u', tagnum, deftyp, 0, 0), indent, indent, indent, indent);
                  }
                  else {
                     fprintf(fp, "%sG__alloc_tempobject_val(result7);\n", indent);
                     fprintf(fp, "%sresult7->obj.i = G__gettempbufpointer();\n", indent);
                     fprintf(fp, "%sresult7->ref = G__gettempbufpointer();\n", indent);
                     fprintf(fp, "%s*((%s *) result7->obj.i) = ", indent, G__type2string(type, tagnum, typenum, reftype, 0));
                     sprintf(endoffunc, ";");
                  }
               }
               break;
            default:
               fprintf(fp, "%sG__letint(result7, %d, (long) ", indent, type);
               sprintf(endoffunc, ");");
               break;
         }
         return 0;
   } // switch(type)
   return 1; /* never happen, avoiding lint error */
#endif // G__SMALLOBJECT
   // --
}

//______________________________________________________________________________
void Cint::Internal::G__cppif_paratype(FILE* fp, const ::Reflex::Member& ifunc, int k)
{
   // TODO: Describe this function!
#ifndef G__SMALLOBJECT
   ::Reflex::Type param_type(ifunc.TypeOf().FunctionParameterAt(k));
   std::string param_name(ifunc.FunctionParameterNameAt(k));
   char type = '\0';
   int tagnum = -1;
   int typenum = -1;
   int reftype = 0;
   int isconst = 0;
   G__get_cint5_type_tuple(param_type, &type, &tagnum, &typenum, &reftype, &isconst);
   // Promote link-off typedef to link-on if used in function.
   if (typenum != -1) {
      ::Reflex::Type ty = G__Dict::GetDict().GetTypedef(typenum);
      if (
         (G__get_properties(ty)->globalcomp == G__NOLINK) &&
         (G__get_properties(ty)->iscpplink == G__NOLINK)
      ) {
         G__get_properties(ty)->globalcomp = G__globalcomp;
      }
   }
   if (k && !(k % 2)) {
      fprintf(fp, "\n");
   }
   if (k) {
      fprintf(fp, ", ");
   }
   if (!param_name.empty()) {
      const char* p = strchr(param_name.c_str(), '[');
      if (p) {
         fprintf(fp, "G__Ap%d->a", k);
         return;
      }
   }
   if (
      // --
#ifndef G__OLDIMPLEMENTATION2191
      (type != '1') && (type != 'a')
#else // G__OLDIMPLEMENTATION2191
      (type != 'Q') && (type != 'a')
#endif // G__OLDIMPLEMENTATION2191
      // --
   ) {
      switch (reftype) {
         case G__PARANORMAL:
            if ((typenum != -1) && G__get_reftype(G__Dict::GetDict().GetTypedef(typenum)) == G__PARAREFERENCE) {
               reftype = G__PARAREFERENCE;
               typenum = -1;
            }
            else {
               break;
            }
            // Note: Intentionally fall through here!
         case G__PARAREFERENCE: 
            {
               std::string castname(G__type2string(type, tagnum, typenum, 0, 0));
               if (islower(type)) {
                  switch (type) {
                     case 'u':
                        fprintf(fp, "*(%s*) libp->para[%d].ref", castname.c_str(), k);
                        break;
#ifndef G__OLDIMPLEMENTATION1167
                     case 'd':
                        fprintf(fp, "*(%s*) G__Doubleref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'l':
                        fprintf(fp, "*(%s*) G__Longref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'i':
                        if (tagnum == -1) { // int
                           fprintf(fp, "*(%s*) G__Intref(&libp->para[%d])", castname.c_str(), k);
                        }
                        else { // enum type
                           fprintf(fp, "*(%s*) libp->para[%d].ref", castname.c_str() , k);
                        }
                        break;
                     case 's':
                        fprintf(fp, "*(%s*) G__Shortref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'c':
                        fprintf(fp, "*(%s*) G__Charref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'h':
                        fprintf(fp, "*(%s*) G__UIntref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'r':
                        fprintf(fp, "*(%s*) G__UShortref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'b':
                        fprintf(fp, "*(%s*) G__UCharref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'k':
                        fprintf(fp, "*(%s*) G__ULongref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'n':
                        fprintf(fp, "*(%s*) G__Longlongref(&libp->para[%d])",
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
                           "__int64",
#else // _MSC_VER && (_MSC_VER < 1310)
                           castname.c_str(),
#endif // _MSC_VER && (_MSC_VER < 1310)
                        k);
                        break;
                     case 'm':
                        fprintf(fp, "*(%s*) G__ULonglongref(&libp->para[%d])",
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
                           "unsigned __int64",
#else // _MSC_VER && (_MSC_VER < 1310)
                           castname.c_str(),
#endif // _MSC_VER && (_MSC_VER < 1310)
                        k);
                        break;
                     case 'q':
                        fprintf(fp, "*(%s*) G__Longdoubleref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'g':
                        fprintf(fp, "*(%s*) G__Boolref(&libp->para[%d])", castname.c_str(), k);
                        break;
                     case 'f':
                        fprintf(fp, "*(%s*) G__Floatref(&libp->para[%d])", castname.c_str(), k);
                        break;
#else // G__OLDIMPLEMENTATION1167
                     case 'd':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mdouble(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'l':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mlong(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'i':
                        if (-1 == tagnum) { // int
                           fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mint(libp->para[%d])", k, castname.c_str(), k, k);
                        }
                        else { // enum type
                           fprintf(fp, "*(%s*) libp->para[%d].ref", castname.c_str(), k);
                        }
                        break;
                     case 's':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mshort(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'c':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mchar(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'h':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Muint(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'r':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mushort(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'b':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Muchar(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'k':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mulong(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'n':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mlonglong(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'm':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mulonglong(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'q':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mlongdouble(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
                     case 'g':
                        // --
#ifdef G__BOOL4BYTE
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mint(libp->para[%d])", k, castname.c_str(), k, k);
#else // G__BOOL4BYTE
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Muchar(libp->para[%d])", k, castname.c_str(), k, k);
#endif // G__BOOL4BYTE
                        break;
                     case 'f':
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mfloat(libp->para[%d])", k, castname.c_str(), k, k);
                        break;
#endif // G__OLDIMPLEMENTATION1167
                        // --
                     default:
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : (%s) G__int(libp->para[%d])", k, castname.c_str(), k, castname.c_str(), k);
                        break;
                  }
               }
               else {
                  if ((typenum != -1) && isupper(G__get_type(G__Dict::GetDict().GetTypedef(typenum)))) {
                     // This part is not perfect. Cint data structure bug.
                     // typedef char* value_type;
                     // void f(value_type& x);  // OK
                     // void f(value_type x);   // OK
                     // void f(value_type* x);  // OK
                     // void f(value_type*& x); // bad
                     //  reference and pointer to pointer can not happen at once
                     fprintf(
                          fp
                        , "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : *(%s*) (&G__Mlong(libp->para[%d]))"
                        , k
                        , G__type2string(type, tagnum, typenum, 0, isconst & G__CONSTVAR)
                        , k
                        , G__type2string(type, tagnum, typenum, 0, isconst & G__CONSTVAR)
                        , k
                     );
                  }
                  else {
                     fprintf(
                          fp
                        , "libp->para[%d].ref ? *(%s) libp->para[%d].ref : *(%s) (&G__Mlong(libp->para[%d]))"
                        , k
                        , G__type2string(type, tagnum, typenum, 2, isconst&G__CONSTVAR)
                        , k
                        , G__type2string(type, tagnum, typenum, 2, isconst&G__CONSTVAR)
                        , k
                     );
                     // above is , in fact, not good. G__type2string returns pointer to
                     // static buffer. This relies on the fact that the 2 calls are
                     // identical
                  }
               }
               return;
            }
         case G__PARAREFP2P:
         case G__PARAREFP2P2P:
            reftype = G__PLVL(reftype);
            if ((typenum != -1) && isupper(G__get_type(G__Dict::GetDict().GetTypedef(typenum)))) {
               fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : *(%s*) (&G__Mlong(libp->para[%d]))"
                       , k, G__type2string(type, tagnum, typenum, reftype, isconst)
                       , k, G__type2string(type, tagnum, typenum, reftype, isconst), k
               );
            }
            else {
               fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : *(%s*) (&G__Mlong(libp->para[%d]))"
                       , k, G__type2string(type, tagnum, typenum, reftype, isconst)
                       , k, G__type2string(type, tagnum, typenum, reftype, isconst), k
               );
            }
            return;
         case G__PARAP2P:
            G__ASSERT(isupper(type));
            fprintf(fp, "(%s) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isconst), k);
            return;
         case G__PARAP2P2P:
            G__ASSERT(isupper(type));
            fprintf(fp, "(%s) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isconst), k);
            return;
      }
   }
   switch (type) {
      // --
#ifndef G__OLDIMPLEMENTATION2191
      case '1': // Pointer to function
#else // G__OLDIMPLEMENTATION2191
      case 'Q': // Pointer to function
#endif // G__OLDIMPLEMENTATION2191
#ifdef G__CPPIF_EXTERNC
         fprintf(fp, "(%s) G__int(libp->para[%d])", G__p2f_typedefname(ifn, ifunc->page, k), k);
         break;
#endif // G__CPPIF_EXTERNC
      case 'c':
      case 'b':
      case 's':
      case 'r':
      case 'i':
      case 'h':
      case 'l':
      case 'k':
      case 'g':
      case 'F':
      case 'D':
      case 'E':
      case 'Y':
      case 'U':
         fprintf(fp, "(%s) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isconst), k);
         break;
      case 'a':
         // Pointer to member , THIS IS BAD , WON'T WORK
         fprintf(fp, "*(%s *) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, 0, isconst), k);
         break;
      case 'n':
         fprintf(fp, "(%s) G__Longlong(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isconst), k);
         break;
      case 'm':
         fprintf(fp, "(%s) G__ULonglong(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isconst), k);
         break;
      case 'q':
         fprintf(fp, "(%s) G__Longdouble(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isconst), k);
         break;
      case 'f':
      case 'd':
         fprintf(fp, "(%s) G__double(libp->para[%d])", G__type2string(type, tagnum, typenum, 0, isconst), k);
         break;
      case 'u':
         if (G__struct.type[tagnum] == 'e') {
            fprintf(fp, "(%s) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, 0, isconst), k);
         }
         else {
            fprintf(fp, "*((%s*) G__int(libp->para[%d]))", G__type2string(type, tagnum, typenum, 0, isconst), k);
         }
         break;
      default:
         fprintf(fp, "(%s) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, 0, isconst), k);
         break;
   }
#endif // G__SMALLOBJECT
   // --
}

//______________________________________________________________________________
void Cint::Internal::G__cpplink_tagtable(FILE* fp, FILE* hfp)
{
   // --
#ifndef G__SMALLOBJECT
   int i;
   G__StrBuf tagname_sb(G__MAXNAME*8);
   char *tagname = tagname_sb;
   G__StrBuf mappedtagname_sb(G__MAXNAME*6);
   char *mappedtagname = mappedtagname_sb;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;

   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Class,struct,union,enum tag information setup\n");
   fprintf(fp, "*********************************************************/\n");

   if (G__CPPLINK == G__globalcomp) {
      G__cpplink_linked_taginfo(fp, hfp);
      fprintf(fp, "extern \"C\" void G__cpp_setup_tagtable%s() {\n", G__DLLID);
   }
   else {
      G__cpplink_linked_taginfo(fp, hfp);
      fprintf(fp, "void G__c_setup_tagtable%s() {\n", G__DLLID);
   }

   fprintf(fp, "\n   /* Setting up class,struct,union tag entry */\n");
   for (i = 2;i < G__struct.alltag;i++) {
      if (
         (G__struct.hash[i] || 0 == G__struct.name[i][0]) &&
         (G__CPPLINK == G__struct.globalcomp[i]
          || G__CLINK == G__struct.globalcomp[i]
          || G__ONLYMETHODLINK == G__struct.globalcomp[i]
         )) {
         if (!G__nestedclass) {
            if (0 <= G__struct.parent_tagnum[i] &&
                  -1 != G__struct.parent_tagnum[G__struct.parent_tagnum[i]])
               continue;
            if (G__CLINK == G__struct.globalcomp[i] && -1 != G__struct.parent_tagnum[i])
               continue;
         }

         if (-1 == G__struct.line_number[i]) {
            /* Philippe and Fons's request to display this */
            if (i != 0 // No message for global namespace
                  && G__dispmsg >= G__DISPERR /*G__DISPNOTE*/) {
               if (G__NOLINK == G__struct.iscpplink[i]) {
                  G__fprinterr(G__serr, "Note: Link requested for undefined class %s (ignore this message)"
                               , G__fulltagname(i, 1));
               }
               else {
                  G__fprinterr(G__serr,
                               "Note: Link requested for already precompiled class %s (ignore this message)"
                               , G__fulltagname(i, 1));
               }
               G__printlinenum();
            }
            /* G__genericerror((char*)NULL); */
         }

         G__getcommentstring(buf, i, &G__get_properties(G__Dict::GetDict().GetScope(i))->comment);

         strcpy(tagname, G__fulltagname(i, 0));
         if (-1 != G__struct.line_number[i]
               && (-1 == G__struct.parent_tagnum[i] || G__nestedclass)
            ) {
            if ('e' == G__struct.type[i])
               fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),sizeof(%s),%d,%d,%s,NULL,NULL);\n"
                       , G__mark_linked_tagnum(i) , "int" , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                       , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                       + G__struct.rootflag[i]*0x10000
#else
                       , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                       , buf);
            else if ('n' == G__struct.type[i]) {
               strcpy(mappedtagname, G__map_cpp_name(tagname));
               fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),0,%d,%d,%s,G__setup_memvar%s,G__setup_memfunc%s);\n"
                       , G__mark_linked_tagnum(i)
                       /* ,G__type2string('u',i,-1,0,0) */
                       , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                       , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                       + G__struct.rootflag[i]*0x10000
#else
                       , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                       , buf, mappedtagname, mappedtagname);
            }
            else if (!G__struct.name[i][0] || (G__struct.name[i][strlen(G__struct.name[i])-1] == '$')) { // anonymous union
               strcpy(mappedtagname, G__map_cpp_name(tagname));
               if (G__CPPLINK == G__globalcomp) {
                  fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),%s,%d,%d,%s,G__setup_memvar%s,G__setup_memfunc%s);\n"
                          , G__mark_linked_tagnum(i)
                          , "0" /* G__type2string('u',i,-1,0,0) */
                          , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                          , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                          + G__struct.rootflag[i]*0x10000
#else
                          , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                          , buf , mappedtagname, mappedtagname);
               }
               else {
                  fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),%s,%d,%d,%s,G__setup_memvar%s,NULL);\n"
                          , G__mark_linked_tagnum(i)
                          , "0" /* G__type2string('u',i,-1,0,0) */
                          , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                          , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                          + G__struct.rootflag[i]*0x10000
#else
                          , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                          , buf , mappedtagname);
               }
            }
            else {
               strcpy(mappedtagname, G__map_cpp_name(tagname));
               if (G__CPPLINK == G__globalcomp && '$' != G__struct.name[i][0]) {
                  if (G__ONLYMETHODLINK == G__struct.globalcomp[i])
                     fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),sizeof(%s),%d,%d,%s,NULL,G__setup_memfunc%s);\n"
                             , G__mark_linked_tagnum(i)
                             , G__type2string('u', i, -1, 0, 0)
                             , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                             , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                             + G__struct.rootflag[i]*0x10000
#else
                             , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                             , buf , mappedtagname);
                  else
                     if (G__suppress_methods)
                        fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),sizeof(%s),%d,%d,%s,G__setup_memvar%s,NULL);\n"
                                , G__mark_linked_tagnum(i)
                                , G__type2string('u', i, -1, 0, 0)
                                , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                                , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                                + G__struct.rootflag[i]*0x10000
#else
                                , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                                , buf , mappedtagname);
                     else
                        fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),sizeof(%s),%d,%d,%s,G__setup_memvar%s,G__setup_memfunc%s);\n"
                                , G__mark_linked_tagnum(i)
                                , G__type2string('u', i, -1, 0, 0)
                                , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                                , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                                + G__struct.rootflag[i]*0x10000
#else
                                , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                                , buf , mappedtagname, mappedtagname);
               }
               else if ('$' == G__struct.name[i][0] &&
                        isupper(G__get_type(G__find_typedef(G__struct.name[i] + 1)))) {
                  fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),sizeof(%s),%d,%d,%s,NULL,NULL);\n"
                          , G__mark_linked_tagnum(i)
                          , G__type2string('u', i, -1, 0, 0)
                          , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                          , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                          + G__struct.rootflag[i]*0x10000
#else
                          , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                          , buf);
               }
               else {
                  fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),sizeof(%s),%d,%d,%s,G__setup_memvar%s,NULL);\n"
                          , G__mark_linked_tagnum(i)
                          , G__type2string('u', i, -1, 0, 0)
                          , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                          , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                          + G__struct.rootflag[i]*0x10000
#else
                          , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                          , buf , mappedtagname);
               }

            }
         }
         else {
            fprintf(fp, "   G__tagtable_setup(G__get_linked_tagnum(&%s),0,%d,%d,%s,NULL,NULL);\n"
                    , G__mark_linked_tagnum(i)
                    , G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                    , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
                    + G__struct.rootflag[i]*0x10000
#else
                    , G__struct.isabstract[i] + G__struct.funcs[i]*0x100
#endif
                    , buf);
         }
         if ('e' != G__struct.type[i]) {
            if (strchr(tagname, '<')) { /* template class */
               fprintf(hfp, "typedef %s G__%s;\n", tagname, G__map_cpp_name(tagname));
            }
         }
      }
      else if ((G__struct.hash[i] || 0 == G__struct.name[i][0]) &&
               (G__CPPLINK - 2) == G__struct.globalcomp[i]) {
         fprintf(fp, "   G__get_linked_tagnum_fwd(&%s);\n" , G__mark_linked_tagnum(i));
      }
   }

   fprintf(fp, "}\n");
#endif
   // --
}

#ifdef G__VIRTUALBASE
//______________________________________________________________________________
static char* G__vbo_funcname(int tagnum, int basetagnum, int basen)
{
   static char result[G__LONGLINE*2];
   G__StrBuf temp_sb(G__LONGLINE);
   char *temp = temp_sb;
   strcpy(temp, G__map_cpp_name(G__fulltagname(tagnum, 1)));
   sprintf(result, "G__2vbo_%s_%s_%d", temp
           , G__map_cpp_name(G__fulltagname(basetagnum, 1)), basen);
   return(result);
}
#endif // G__VIRTUALBASE

#ifdef G__VIRTUALBASE
//______________________________________________________________________________
void Cint::Internal::G__cppif_inheritance(FILE* fp)
{
   int i;
   size_t basen;
   int basetagnum;
   G__StrBuf temp_sb(G__LONGLINE*2);
   char *temp = temp_sb;

   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* virtual base class offset calculation interface\n");
   fprintf(fp, "*********************************************************/\n");

   fprintf(fp, "\n   /* Setting up class inheritance */\n");
   for (i = 0;i < G__struct.alltag;i++) {
      if (G__NOLINK > G__struct.globalcomp[i] &&
            (-1 == (int)G__struct.parent_tagnum[i]
             || G__nestedclass
            )
            && -1 != G__struct.line_number[i] && G__struct.hash[i] &&
            ('$' != G__struct.name[i][0])) {
         switch (G__struct.type[i]) {
            case 'c': /* class */
            case 's': /* struct */
               if (!G__struct.baseclass[i]->vec.empty()) {
                  for (basen = 0;basen < G__struct.baseclass[i]->vec.size();basen++) {
                     if (G__PUBLIC != G__struct.baseclass[i]->vec[basen].baseaccess ||
                           0 == (G__struct.baseclass[i]->vec[basen].property&G__ISVIRTUALBASE))
                        continue;
                     basetagnum = G__struct.baseclass[i]->vec[basen].basetagnum;
                     fprintf(fp, "static long %s(long pobject) {\n"
                             , G__vbo_funcname(i, basetagnum, basen));
                     strcpy(temp, G__fulltagname(i, 1));
                     fprintf(fp, "  %s *G__Lderived=(%s*)pobject;\n", temp, temp);
                     fprintf(fp, "  %s *G__Lbase=G__Lderived;\n", G__fulltagname(basetagnum, 1));
                     fprintf(fp, "  return((long)G__Lbase-(long)G__Lderived);\n");
                     fprintf(fp, "}\n\n");
                  }
               }
               break;
            default: /* enum */
               break;
         }
      }
   }
}
#endif // G__VIRTUALBASE

//______________________________________________________________________________
void Cint::Internal::G__cpplink_inheritance(FILE* fp)
{
   // --
#ifndef G__SMALLOBJECT
   int i;
   size_t basen;
   int basetagnum;
   G__StrBuf temp_sb(G__MAXNAME*6);
   char *temp = temp_sb;
   int flag;

   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Inheritance information setup/\n");
   fprintf(fp, "*********************************************************/\n");

   if (G__CPPLINK == G__globalcomp) {
      fprintf(fp, "extern \"C\" void G__cpp_setup_inheritance%s() {\n", G__DLLID);
   }
   else  {}

   fprintf(fp, "\n   /* Setting up class inheritance */\n");
   for (i = 0;i < G__struct.alltag;i++) {
      if (G__NOLINK > G__struct.globalcomp[i] &&
            (-1 == (int)G__struct.parent_tagnum[i]
             || G__nestedclass
            )
            && -1 != G__struct.line_number[i] && G__struct.hash[i] &&
            ('$' != G__struct.name[i][0])) {
         switch (G__struct.type[i]) {
            case 'c': /* class */
            case 's': /* struct */
               if (!G__struct.baseclass[i]->vec.empty()) {
                  fprintf(fp, "   if(0==G__getnumbaseclass(G__get_linked_tagnum(&%s))) {\n"
                          , G__get_link_tagname(i));
                  flag = 0;
                  for (basen = 0;basen < G__struct.baseclass[i]->vec.size();basen++) {
                     if (0 == (G__struct.baseclass[i]->vec[basen].property&G__ISVIRTUALBASE))
                        ++flag;
                  }
                  if (flag) {
                     fprintf(fp, "     %s *G__Lderived;\n", G__fulltagname(i, 0));
                     fprintf(fp, "     G__Lderived=(%s*)0x1000;\n", G__fulltagname(i, 1));
                  }
                  for (basen = 0;basen < G__struct.baseclass[i]->vec.size();basen++) {
                     basetagnum = G__struct.baseclass[i]->vec[basen].basetagnum;
                     fprintf(fp, "     {\n");
#ifdef G__VIRTUALBASE
                     strcpy(temp, G__mark_linked_tagnum(basetagnum));
                     if (G__struct.baseclass[i]->vec[basen].property&G__ISVIRTUALBASE) {
                        G__StrBuf temp2_sb(G__LONGLINE*2);
                        char *temp2 = temp2_sb;
                        strcpy(temp2, G__vbo_funcname(i, basetagnum, basen));
                        fprintf(fp, "       G__inheritance_setup(G__get_linked_tagnum(&%s),G__get_linked_tagnum(&%s),(long)%s,%d,%ld);\n"
                                , G__mark_linked_tagnum(i)
                                , temp
                                , temp2
                                , G__struct.baseclass[i]->vec[basen].baseaccess
                                , (long)G__struct.baseclass[i]->vec[basen].property
                               );
                     }
                     else {
                        size_t basen2, flag2 = 0;
                        for (basen2 = 0;basen2 < G__struct.baseclass[i]->vec.size();basen2++) {
                           if (basen2 != basen &&
                                 (G__struct.baseclass[i]->vec[basen].basetagnum
                                  == G__struct.baseclass[i]->vec[basen2].basetagnum) &&
                                 ((G__struct.baseclass[i]->vec[basen].property&G__ISVIRTUALBASE)
                                  == 0 ||
                                  (G__struct.baseclass[i]->vec[basen2].property&G__ISVIRTUALBASE)
                                  == 0)) {
                              flag2 = 1;
                           }
                        }
                        strcpy(temp, G__fulltagname(basetagnum, 1));
                        if (!flag2)
                           fprintf(fp, "       %s *G__Lpbase=(%s*)G__Lderived;\n"
                                   , temp, G__fulltagname(basetagnum, 1));
                        else {
                           G__fprinterr(G__serr,
                                        "Warning: multiple ambiguous inheritance %s and %s. Cint will not get correct base object address\n"
                                        , temp, G__fulltagname(i, 1));
                           fprintf(fp, "       %s *G__Lpbase=(%s*)((long)G__Lderived);\n"
                                   , temp, G__fulltagname(basetagnum, 1));
                        }
                        strcpy(temp, G__mark_linked_tagnum(basetagnum));
                        fprintf(fp, "       G__inheritance_setup(G__get_linked_tagnum(&%s),G__get_linked_tagnum(&%s),(long)G__Lpbase-(long)G__Lderived,%d,%ld);\n"
                                , G__mark_linked_tagnum(i)
                                , temp
                                , G__struct.baseclass[i]->vec[basen].baseaccess
                                , (long)G__struct.baseclass[i]->vec[basen].property
                               );
                     }
#else
                     strcpy(temp, G__fulltagname(basetagnum, 1));
                     if (G__struct.baseclass[i]->vec[basen].property&G__ISVIRTUALBASE) {
                        fprintf(fp, "       %s *pbase=(%s*)0x1000;\n"
                                , temp, G__fulltagname(basetagnum, 1));
                     }
                     else {
                        fprintf(fp, "       %s *pbase=(%s*)G__Lderived;\n"
                                , temp, G__fulltagname(basetagnum, 1));
                     }
                     strcpy(temp, G__mark_linked_tagnum(basetagnum));
                     fprintf(fp, "       G__inheritance_setup(G__get_linked_tagnum(&%s),G__get_linked_tagnum(&%s),(long)pbase-(long)G__Lderived,%d,%ld);\n"
                             , G__mark_linked_tagnum(i)
                             , temp
                             , G__struct.baseclass[i]->vec[basen].baseaccess
                             , G__struct.baseclass[i]->vec[basen].property
                            );
#endif
                     fprintf(fp, "     }\n");
                  }
                  fprintf(fp, "   }\n");
               }
               break;
            default: /* enum */
               break;
         }
      } /* if() */
   } /* for(i) */

   fprintf(fp, "}\n");
#endif
   // --
}

//______________________________________________________________________________
void Cint::Internal::G__cpplink_typetable(FILE* fp, FILE* /*hfp*/)
{
   G__StrBuf temp_sb(G__ONELINE);
   char* temp = temp_sb;
   G__StrBuf buf_sb(G__ONELINE);
   char* buf = buf_sb;
   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* typedef information setup/\n");
   fprintf(fp, "*********************************************************/\n");
   if (G__globalcomp == G__CPPLINK) {
      fprintf(fp, "extern \"C\" void G__cpp_setup_typetable%s() {\n", G__DLLID);
   }
   else {
      fprintf(fp, "void G__c_setup_typetable%s() {\n", G__DLLID);
   }
   fprintf(fp, "\n   /* Setting up typedef entry */\n");
   //
   //  Mark the parents of all nested typedefs to be
   //  included in the dictionary.
   //
   size_t max_type = ::Reflex::Type::TypeSize();
   if (G__nestedtypedef) { // We are including nested typedefs in the dictionary.
      for (size_t i = 0; i < max_type; ++i) {
         ::Reflex::Type ty = ::Reflex::Type::TypeAt(i);
         if (!ty.IsTypedef()) {
            continue;
         }
         G__RflxProperties* prop = G__get_properties(ty);
         if (prop && (prop->globalcomp < G__NOLINK)) { // We will include this typedef.
            if (ty.DeclaringScope() != ::Reflex::Scope::GlobalScope()) { // And it is nested.
               G__mark_linked_tagnum(G__get_tagnum(ty.DeclaringScope())); // Mark the parent linked.
            }
         }
      }
   }
   for (size_t i = 0; i < max_type; ++i) {
      ::Reflex::Type ty = ::Reflex::Type::TypeAt(i);
      if (!ty.IsTypedef()) {
         continue;
      }
      G__RflxProperties* prop = G__get_properties(ty);
      //fprintf(stderr, "G__cpplink_typetable: name: '%s'\n", ty.Name(::Reflex::SCOPED).c_str());
      //fprintf(stderr, "G__cpplink_typetable:   gc: %d\n", (int) G__struct.globalcomp[G__get_tagnum(ty.DeclaringScope())]);
      //fprintf(stderr, "G__cpplink_typetable:   gc: %d\n", (int) prop->globalcomp);
      if (prop && (prop->globalcomp < G__NOLINK)) {
         if (
            (ty.DeclaringScope() != ::Reflex::Scope::GlobalScope()) &&
            (
               !G__nestedtypedef ||
               (G__struct.globalcomp[G__get_tagnum(ty.DeclaringScope())] >= G__NOLINK)
            )
         ) {
            continue;
         }
#ifdef __GNUC__
#else
#pragma message(FIXME("Commented out code that need porting (typedef of functions)"))
#endif
#if 0 // FIXME
         if (!strncmp("G__p2mf", G__newtype.name[i], 7) && (G__globalcomp == G__CPPLINK)) {
            G__ASSERT(i > 0);
            strcpy(temp, G__newtype.name[i-1]);
            p = strstr(temp, "::*");
            *(p + 3) = '\0';
            fprintf(hfp, "typedef %s%s)%s;\n", temp, G__newtype.name[i], p + 4);
         }
#endif // 0
         if (tolower(G__get_type(ty)) == 'u') {
            fprintf(fp, "   G__search_typename2(\"%s\",%d,G__get_linked_tagnum(&%s),%d,", ty.Name().c_str(), G__get_type(ty), G__mark_linked_tagnum(G__get_tagnum(ty)), G__get_reftype(ty) | (G__get_isconst(ty) * 0x100));
         }
         else {
            fprintf(fp, "   G__search_typename2(\"%s\",%d,-1,%d,", ty.Name().c_str(), G__get_type(ty), G__get_reftype(ty) | (G__get_isconst(ty) * 0x100));
         }
         if (ty.DeclaringScope() == ::Reflex::Scope::GlobalScope()) {
            fprintf(fp, "-1);\n");
         }
         else {
            fprintf(fp, "G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(G__get_tagnum(ty.DeclaringScope())));
         }
         if (prop->comment.filenum != -1) {
            G__getcommenttypedef(temp, &prop->comment, ty);
            if (temp[0]) {
               G__add_quotation(temp, buf);
            }
            else {
               strcpy(buf, "NULL");
            }
         }
         else {
            strcpy(buf, "NULL");
         }
         if (!ty.FinalType().IsArray()) {
            fprintf(fp, "   G__setnewtype(%d,%s,0);\n", G__globalcomp, buf);
         }
         else {
            std::vector<int> bounds;
            ::Reflex::Type typ = ty.FinalType();
            for (; typ && typ.IsArray(); typ = typ.ToType()) {
               bounds.push_back(typ.ArrayLength());
            }
            for (int tind = bounds.size() - 1; tind > -1; --tind) {
               fprintf(fp, "   G__setnewtypeindex(%d,%d);\n", tind, bounds[tind]);
            }
         }
      }
   }
   fprintf(fp, "}\n");
}

//______________________________________________________________________________
void Cint::Internal::G__cpplink_memvar(FILE* fp)
{
   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Data Member information setup/\n");
   fprintf(fp, "*********************************************************/\n");
   fprintf(fp, "\n   /* Setting up class,struct,union tag member variable */\n");
   //
   //  Loop over all known classes, enums, namespaces, structs and unions.
   //
   for (int i = 0; i < G__struct.alltag; ++i) {
      if ( // Class is marked for dictionary generation.
         (
            (G__struct.globalcomp[i] == G__CPPLINK) || // Class is marked for c++ dictionary, or
            (G__struct.globalcomp[i] == G__CLINK) // Class is marked for c dictionary,
         ) && // and,
         (
            (G__struct.parent_tagnum[i] == -1) || // Not an innerclass, or
            G__nestedclass // Is an inner class,
         ) && // and,
         (G__struct.line_number[i] != -1) && // Class has line number information, and
         (
            G__struct.hash[i] || // Class has a name with a non-zero hash, or
            !G__struct.name[i][0] // Class is unnamed
         )
      ) { // Class is marked for dictionary generation.
         //
         //  FIXME: What is this block doing?
         //
         if (G__struct.name[i][0] == '$') {
            ::Reflex::Type typenum = G__find_typedef(G__struct.name[i] + 1);
            if (isupper(G__get_type(typenum))) {
               continue;
            }
         }
         //
         //  Skip all enums.
         //
         if (G__struct.type[i] == 'e') {
            continue;
         }
         //
         //  Write the class name to the dictionary file
         //  for the poor humans who have to read it later.
         //
         fprintf(fp, "\n   /* %s */\n", G__type2string('u', i, -1, 0, 0));
         //
         //  Write out member variable setup function
         //  header to dictionary file.
         //
         if (G__globalcomp == G__CPPLINK) { // C++ style.
            fprintf(fp, "static void G__setup_memvar%s(void) {\n", G__map_cpp_name(G__fulltagname(i, 0)));
         }
         else {
            if (G__clock) { // C style.
               fprintf(fp, "static void G__setup_memvar%s() {\n", G__map_cpp_name(G__fulltagname(i, 0)));
            }
            else { // C++ style by default.
               fprintf(fp, "static void G__setup_memvar%s(void) {\n", G__map_cpp_name(G__fulltagname(i, 0)));
            }
         }
         //
         //  Write out call to member variable setup
         //  initialization to the dictionary file.
         //
         fprintf(fp, "   G__tag_memvar_setup(G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(i));
         //
         //  We need a fake this pointer, except for namespace, unnamed union, and unnamed enum members.
         //
         if ((G__struct.type[i] == 'n') || !G__struct.name[i][0] || (G__struct.name[i][strlen(G__struct.name[i])-1] == '$')) {
            fprintf(fp, "   {\n");
         }
         else {
            fprintf(fp, "   { %s *p; p=(%s*)0x1000; if (p) { }\n", G__type2string('u', i, -1, 0, 0), G__type2string('u', i, -1, 0, 0));
         }
         //
         //  Loop over all the data members and write setup info to the file.
         //
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(i);
         for (::Reflex::Member_Iterator mbr_iter = scope.DataMember_Begin(); mbr_iter != scope.DataMember_End(); ++mbr_iter) {
            if ( // Data member is accessible and is not a bitfield, or G__precomp_private flag is set.
               (
                  (
                     mbr_iter->IsPublic() || // Data member is public, or
                     (
                        mbr_iter->IsProtected() && // Data member is protected, and
                        (G__struct.protectedaccess[i] & G__PROTECTEDACCESS) // Class is marked for protected access.
                     ) || // or,
                     (G__struct.protectedaccess[i] & G__PRIVATEACCESS) // Class is marked for private access.
                  ) && // and,
                  !G__get_bitfield_width(*mbr_iter) // Data member is not a bitfield.
               ) ||
               G__precomp_private
            ) { // Data member is accessible and is not a bitfield, or G__precomp_private flag is set.
               //
               //  Write a data member setup call to the dictionary file.
               //
               char type = '\0';
               int tagnum = -1;
               int typenum = -1;
               int reftype = 0;
               int isconst = 0;
               G__get_cint5_type_tuple(mbr_iter->TypeOf(), &type, &tagnum, &typenum, &reftype, &isconst);
               G__RflxVarProperties* prop = G__get_properties(*mbr_iter);
               int pvoidflag = 0;
               if ( // Is enumerator or unaddressable bool or const static fundamental.
                  (
                     islower(type) && // not a pointer, and
                     isconst && // is const, and
                     (tagnum != -1) && // class tag is valid, and
                     mbr_iter->TypeOf().RawType().IsEnum() // data member of an enum
                  ) || // or,
#ifdef G__UNADDRESSABLEBOOL
                  (type == 'g') || // or, is an unaddressable bool
#endif // G__UNADDRESSABLEBOOL
                  (
                     prop->statictype == G__LOCALSTATIC && // static, and
                     isconst && // const, and
                     islower(type) && // not a pointer, and
                     (type != 'u') && // not a class, enum, struct, or union, and
                     G__get_offset(*mbr_iter) // has allocated memory (is initialized???)
                   )
               ) { // Is enumerator or unaddressable bool or const static fundamental.
                  pvoidflag = 1; // Pass G__PVOID as the address to force G__malloc to allocate storage.
               }
               fprintf(fp, "   G__memvar_setup(");
               //
               //  Offset in object for a non-static data member, or
               //  the address of a member for a static data
               //  member, or a namespace member.
               //
               if (mbr_iter->IsPublic() && !G__get_bitfield_width(*mbr_iter)) { // Public member, not a bitfield.
                  if (!G__struct.name[i][0] || (G__struct.name[i][strlen(G__struct.name[i])-1] == '$')) {
                     // Anonymous union or namespace, we pass a null pointer.
                     // We pass a null pointer, which means no data allocation (unfortunate,
                     // but we have no way to take the address!).
                     fprintf(fp, "(void*)0,");
                  }
                  else if ( // Static member or namespace member.
                     (prop->statictype == G__LOCALSTATIC) || // Static member, or
                     scope.IsNamespace() // Namespace member
                  ) { // Static member or namespace member.
                     // We pass the special G__PVOID flag, or the address of the member.
                     if (pvoidflag) {
                        // Special case, is enumerator, unaddressable bool, or static.
                        fprintf(fp, "(void*)G__PVOID,"); // Pass G__PVOID to force G__malloc to allocate storage.
                     }
                     else {
                        // We pass the address of the member.
                        fprintf(fp, "(void*)(&%s::%s),", G__fulltagname(i, 1), mbr_iter->Name().c_str());
                     }
                  }
                  else {
                     // We pass the offset of the data member in the class.
                     fprintf(fp, "(void*)((long)(&p->%s)-(long)(p)),", mbr_iter->Name().c_str());
                  }
               }
               else if (mbr_iter->IsProtected() && G__struct.protectedaccess[i]) { // Protected member.
                  fprintf(fp, "(void*)((%s_PR*)p)->G__OS_%s(),", G__get_link_tagname(i), mbr_iter->Name().c_str());
               }
               else {
                  // Private or protected member, we pass a null pointer, unfortunate,
                  // but we have no way to take the address of these.
                  fprintf(fp, "(void*)0,");
               }
               //
               //  Type code, referenceness, and constness.
               //
               fprintf(fp, "%d,", type);
               fprintf(fp, "%d,", reftype);
               fprintf(fp, "%d,", isconst);
               //
               //  Tagnum of data type, if not fundamental.
               //
               if (tagnum != -1) {
                  fprintf(fp, "G__get_linked_tagnum(&%s),", G__mark_linked_tagnum(tagnum));
               }
               else {
                  fprintf(fp, "-1,");
               }
               //
               //  Typenum of data type, if it is a typedef.
               //
               if (typenum != -1) {
                  ::Reflex::Type ty = mbr_iter->TypeOf();
                  for (; !ty.IsTypedef(); ty = ty.ToType()) {}
                  std::string tmp = ty.Name();
                  // Remove any array bounds in the name.
                  std::string::size_type pos = tmp.find("[");
                  if (pos != std::string::npos) {
                     tmp.erase(pos);
                  }
                  fprintf(fp, "G__defined_typename(\"%s\"),", tmp.c_str());
               }
               else {
                  fprintf(fp, "-1,");
               }
               //
               //  Storage duration and staticness, member access.
               //
               fprintf(fp, "%d,", prop->statictype);
               fprintf(fp, "%d,", G__get_access(*mbr_iter));
               //
               //  Name and array dimensions (quoted) as the
               //  left hand side of an assignment expression.
               //
               fprintf(fp, "\"%s", mbr_iter->Name().c_str());
               //
               if (G__get_varlabel(*mbr_iter, 1) /* number of elements */ == INT_MAX /* unspecified length flag */) {
                  fprintf(fp, "[]");
               }
               else if (G__get_varlabel(*mbr_iter, 1) /* number of elements */) {
                  fprintf(fp, "[%d]", G__get_varlabel(*mbr_iter, 1) /* number of elements */ / G__get_varlabel(*mbr_iter, 0) /* stride */);
               }
               for (int k = 1; k < G__get_paran(*mbr_iter); ++k) {
                  fprintf(fp, "[%d]", G__get_varlabel(*mbr_iter, k + 1));
               }
               if ( // Enumerator in a static enum or const static fundamental.
                  pvoidflag && // Is enumerator or unaddressable bool, and
                  (prop->statictype == G__LOCALSTATIC) // is static
                  && (
#ifdef G__UNADDRESSABLEBOOL
                      type != 'g' || // and is unaddressable bool FIXME: Should be or here?
#endif // G__UNADDRESSABLEBOOL
                      (isconst && // const static
                       islower(type) && type != 'u' && // of fundamental
                       G__get_offset(*mbr_iter) // with initializer
                      )
                  )
                  // --
               ) { // Enumerator in a static enum.
                  // Enumerator in a static enum has a special initializer.
                  // CAUTION: This implementation cause error on enum in nested class.
                  G__StrBuf ttt_sb(G__MAXNAME*6);
                  char* ttt = ttt_sb;
                  sprintf(ttt, "%s::%s", G__fulltagname(i, 1), mbr_iter->Name().c_str());
                  int store_var_type = G__var_type;
                  G__var_type = 'p';
                  G__value buf;
                  G__StrBuf value_sb(G__MAXNAME*6);
                  char* value = value_sb;
                  if (isconst && // const static
                      (isupper(type) || type != 'u') && // of fundamental
                      G__get_offset(*mbr_iter) // with initializer
                      ) {
                     // local static, can be private thus cannot call G__getitem.
                     // Take the value from var->p instead. If var is an enum constant
                     // it will be stored as an int, so convert it accordingly.
                     bool isInt = (type == 'i');
                     sprintf(value, "*(%s*)0x%lx",
                             isInt ? "int" : G__type2string(type, tagnum, typenum, 0, 0),
                             (unsigned long)G__get_offset(*mbr_iter));
                     buf = G__calc_internal(value);
                  } else {
                     buf = G__getitem(ttt);
                  }
                  G__var_type = store_var_type;
                  G__string(buf, value);
                  G__quotedstring(value, ttt);
                  fprintf(fp, "=%s\"", ttt);
               }
               else {
                  fprintf(fp, "=\"");
               }
               //
               //  Define macro flag (always zero).
               //
               fprintf(fp, ",0");
               //
               //  Comment string.
               //
               {
                  G__StrBuf commentbuf_sb(G__LONGLINE);
                  char* commentbuf = commentbuf_sb;
                  G__getcommentstring(commentbuf, i, &prop->comment);
                  fprintf(fp, ",%s);\n", commentbuf);
               }
            }
            G__var_type = 'p';
         }
         fprintf(fp, "   }\n");
         //
         //  Write out a call to the shutdown routine for member
         //  variable initialization to the dictionary file.
         //
         fprintf(fp, "   G__tag_memvar_reset();\n");
         fprintf(fp, "}\n\n");
      }
   }
   if (G__globalcomp == G__CPPLINK) {
      fprintf(fp, "extern \"C\" void G__cpp_setup_memvar%s() {\n", G__DLLID);
   }
   else {
      fprintf(fp, "void G__c_setup_memvar%s() {\n", G__DLLID);
   }
   fprintf(fp, "}\n");
   // Following dummy comment string is needed to clear rewinded part of the
   // interface method source file.
   fprintf(fp, "/***********************************************************\n");
   fprintf(fp, "************************************************************\n");
   fprintf(fp, "************************************************************\n");
   fprintf(fp, "************************************************************\n");
   fprintf(fp, "************************************************************\n");
   fprintf(fp, "************************************************************\n");
   fprintf(fp, "************************************************************\n");
   fprintf(fp, "***********************************************************/\n");
}

//______________________________________________________________________________
static int G__isprivatectordtorassgn(int /* tagnum */, const ::Reflex::Member& ifunc)
{
   if (ifunc.IsPublic()) return 0;
   if ('~' == ifunc.Name().c_str()[0]) return(1);
   if (ifunc.Name() == ifunc.DeclaringScope().Name()) return(1);
   if (strcmp(ifunc.Name().c_str(), "operator=") == 0) return(1);
   return(0);
}

//______________________________________________________________________________
void Cint::Internal::G__cpplink_memfunc(FILE* fp)
{
   // --
#ifndef G__SMALLOBJECT
   int i;
   unsigned int k;
   int hash;
   G__StrBuf funcname_sb(G__MAXNAME*6);
   char *funcname = funcname_sb;
   int isconstructor, iscopyconstructor, isdestructor, isassignmentoperator;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   int isnonpublicnew;
   int virtualdtorflag;
   int dtoraccess = G__PUBLIC;

   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Member function information setup for each class\n");
   fprintf(fp, "*********************************************************/\n");

   if (G__CPPLINK == G__globalcomp) {}
   else {}

   for (i = 0;i < G__struct.alltag;i++) {
      dtoraccess = G__PUBLIC;
      if ((G__CPPLINK == G__struct.globalcomp[i]
            || G__ONLYMETHODLINK == G__struct.globalcomp[i]
          ) &&
            (-1 == (int)G__struct.parent_tagnum[i]
             || G__nestedclass
            )
            && -1 != G__struct.line_number[i] &&
            (G__struct.hash[i] || 0 == G__struct.name[i][0])
            &&
            '$' != G__struct.name[i][0] && 'e' != G__struct.type[i]) {
         isconstructor = 0;
         iscopyconstructor = 0;
         isdestructor = 0;
         isassignmentoperator = 0;
         isnonpublicnew = G__isnonpublicnew(i);
         virtualdtorflag = 0;

         if (G__clock)
            fprintf(fp, "static void G__setup_memfunc%s() {\n" , G__map_cpp_name(G__fulltagname(i, 0)));
         else
            fprintf(fp, "static void G__setup_memfunc%s(void) {\n" , G__map_cpp_name(G__fulltagname(i, 0)));

         /* link member function information */
         fprintf(fp, "   /* %s */\n", G__type2string('u', i, -1, 0, 0));

         fprintf(fp, "   G__tag_memfunc_setup(G__get_linked_tagnum(&%s));\n" , G__mark_linked_tagnum(i));

         if (0 == G__struct.name[i][0]) {
            fprintf(fp, "}\n");
            continue;
         }

         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(i);
         for (::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
               ifunc != scope.FunctionMember_End();
               ++ifunc) {

            if ((ifunc->IsPublic()) || G__precomp_private || G__isprivatectordtorassgn(i, *ifunc) || ((ifunc->IsProtected()) && (G__struct.protectedaccess[i] & G__PROTECTEDACCESS)) || (G__struct.protectedaccess[i] & G__PRIVATEACCESS)) {
               // public

               int ifunc_globalcomp = G__get_funcproperties(*ifunc)->globalcomp;
               if ((G__struct.globalcomp[i] == G__ONLYMETHODLINK) && ( ifunc_globalcomp != G__METHODLINK)) {
                  // not marked for link, skip it.
                  continue;
               }

               if ( ifunc_globalcomp == G__CSTUB || ifunc_globalcomp == G__CPPSTUB) {
                  // Do not generate a stub around the 'stubbed' method which are supposed to be interpreted!
                  continue;
               }

#ifndef G__OLDIMPLEMENTATION1656
               if (G__get_funcproperties(*ifunc)->entry.size < 0) {
                  // already precompiled, skip it
                  continue;
               }
#endif // G__OLDIMPLEMENTATION1656

               // Check for constructor, destructor, or operator=.
               if (!strcmp(ifunc->Name().c_str(), G__struct.name[i])) {
                  // We have a constructor.
                  if (G__struct.isabstract[i]) {
                     continue;
                  }
                  if (isnonpublicnew) {
                     continue;
                  }
                  ++isconstructor;
                  if (ifunc->IsCopyConstructor()) {
                     ++iscopyconstructor;
                  }
               }
               else if (ifunc->Name().c_str()[0] == '~') {
                  // We have a destructor.
                  dtoraccess = G__get_access(*ifunc);
                  virtualdtorflag = 1 * ifunc->IsVirtual() + (ifunc->IsAbstract() * 2);
                  if (!ifunc->IsPublic()) {
                     ++isdestructor;
                  }
                  if ((ifunc->IsProtected()) && G__struct.protectedaccess[i] && !G__precomp_private) {
                     G__fprinterr(G__serr, "Limitation: can not generate dictionary for protected destructor for %s\n", G__fulltagname(i, 1));
                     continue;
                  }
                  continue;
               }

#ifdef G__DEFAULTASSIGNOPR
               else if (!strcmp(ifunc->Name().c_str(), "operator=") && ('u' == G__get_type(ifunc->TypeOf().FunctionParameterAt(0))) && (i == G__get_tagnum(ifunc->TypeOf().FunctionParameterAt(0).RawType()))) {
                  // We have an operator=.
                  ++isassignmentoperator;
               }
#endif // G__DEFAULTASSIGNOPR

               /****************************************************************
               * setup normal function
               ****************************************************************/
               /* function name and return type */
               fprintf(fp, "   G__memfunc_setup(");
               {
                  int local_hash = 0;
                  int junk = 0;
                  G__hash(ifunc->Name().c_str(), local_hash, junk)
                  fprintf(fp, "\"%s\",%d,", ifunc->Name().c_str(), local_hash);
               }
               if (G__test_access(*ifunc, G__PUBLIC)
                     || (((ifunc->IsProtected() &&
                           (G__PROTECTEDACCESS&G__struct.protectedaccess[i])) ||
                          (G__PRIVATEACCESS&G__struct.protectedaccess[i])) &&
                         '~' != ifunc->Name().c_str()[0])
                  ) {
                  // If the method is virtual. Is it overridden? -> Does it exist in the base classes? 
                     //  Virtual method found in the base classes(we have an overridden virtual method)so it
                     //  has not stub function in its dictionary
                     if ((ifunc->IsVirtual())&&(G__method_inbase(*ifunc)))

                        // Null Stub Pointer
                        fprintf(fp, "(G__InterfaceMethod) NULL," );

                     else
                        // If the method isn't virtual or it belongs to a base class. 
                        // The method has its own Stub Function in its dictionary
                        // Normal Stub Pointer
                        fprintf(fp, "%s, ", ::G__map_cpp_funcname(*ifunc));
               }
               else {
                  fprintf(fp, "(G__InterfaceMethod) NULL, ");
               }
               fprintf(fp, "%d, ", G__get_type(ifunc->TypeOf().ReturnType()));

               if (-1 != G__get_tagnum(ifunc->TypeOf().ReturnType().RawType()))
                  fprintf(fp, "G__get_linked_tagnum(&%s), ", G__mark_linked_tagnum(G__get_tagnum(ifunc->TypeOf().ReturnType().RawType())));
               else
                  fprintf(fp, "-1, ");

               if (G__get_cint5_typenum(ifunc->TypeOf().ReturnType()) != -1) {
                  ::Reflex::Type ty = ifunc->TypeOf().ReturnType();
                  for ( ; !ty.IsTypedef(); ty = ty.ToType()) {
                     // Intentionally empty
                  }
                  fprintf(fp, "G__defined_typename(\"%s\"), ", ty.Name(::Reflex::SCOPED).c_str());
               }
               else {
                  fprintf(fp, "-1, ");
               }

               fprintf(fp, "%d, ", G__get_reftype(ifunc->TypeOf().ReturnType()));

               fprintf(fp, "%ld, ", (long)ifunc->FunctionParameterSize());

               if (2 == G__get_funcproperties(*ifunc)->entry.ansi)
                  fprintf(fp, "%d, ", 8 + ifunc->IsStatic()*2 + ifunc->IsExplicit()*4);
               else
                  fprintf(fp, "%d, ", G__get_funcproperties(*ifunc)->entry.ansi + ifunc->IsStatic()*2 + ifunc->IsExplicit()*4);

               fprintf(fp, "%d, ", G__get_access(*ifunc));
               fprintf(fp, "%d, ", G__get_isconst(ifunc->TypeOf()));

               /* newline to avoid lines more than 256 char for CMZ */
               if (ifunc->FunctionParameterSize() > 1) fprintf(fp, "\n");
               fprintf(fp, "\"");

               /****************************************************************
               * function parameter
               ****************************************************************/

               for (k = 0; k < ifunc->FunctionParameterSize(); ++k) {
                  char type = '\0';
                  int tagnum = -1;
                  int typenum = -1;
                  int reftype = 0;
                  int isconst = 0;
                  G__get_cint5_type_tuple(ifunc->TypeOf().FunctionParameterAt(k), &type, &tagnum, &typenum, &reftype, &isconst);
                  // newline to avoid lines more than 256 char for CMZ
                  if ((G__globalcomp == G__CPPLINK) && k && !(k % 2)) {
                     fprintf(fp, "\"\n\"");
                  }
                  if (isprint(type)) {
                     fprintf(fp, "%c ", type);
                  }
                  else {
                     G__fprinterr(G__serr, "Internal error: function parameter type\n");
                     fprintf(fp, "%d ", type);
                  }

                  if (tagnum != -1) {
                     fprintf(fp, "'%s' ", G__fulltagname(tagnum, 0));
                     G__mark_linked_tagnum(tagnum);
                  }
                  else {
                     fprintf(fp, "- ");
                  }

                  if (typenum != -1) {
                     fprintf(fp, "'%s' ", G__fulltypename(G__Dict::GetDict().GetTypedef(typenum)).c_str());
                  }
                  else {
                     fprintf(fp, "- ");
                  }

                  fprintf(fp, "%d ", reftype + (isconst * 10));
                  if (ifunc->FunctionParameterDefaultAt(k).c_str()[0])
                     fprintf(fp, "'%s' ", G__quotedstring((char*)ifunc->FunctionParameterDefaultAt(k).c_str(), buf));
                  else
                     fprintf(fp, "- ");
                  if (ifunc->FunctionParameterNameAt(k).c_str()[0])
                     fprintf(fp, "%s", ifunc->FunctionParameterNameAt(k).c_str());
                  else
                     fprintf(fp, "-");
                  if (k != ifunc->FunctionParameterSize() - 1) fprintf(fp, " ");
               }
               fprintf(fp, "\"");

               G__getcommentstring(buf, i, &G__get_funcproperties(*ifunc)->comment);
               fprintf(fp, ", %s", buf);
#ifdef G__TRUEP2F
#if defined(G__OLDIMPLEMENTATION1289_YET) || !defined(G__OLDIMPLEMENTATION1993)
               if (
#ifndef G__OLDIMPLEMENTATION1993
                  (ifunc->IsStatic() || scope.IsNamespace())
#else // G__OLDIMPLEMENTATION1993
                  ifunc->IsStatic()
#endif // G__OLDIMPLEMENTATION1993
#ifndef G__OLDIMPLEMENTATION1292
                  && ifunc->IsPublic()
#endif // G__OLDIMPLEMENTATION1292
                  && G__MACROLINK != G__get_funcproperties(*ifunc)->globalcomp
               ) {
#ifndef G__OLDIMPLEMENTATION1993
                  fprintf(fp, ", (void*) G__func2void( (%s (*)("
                          , G__type2string(G__get_type(ifunc->TypeOf().ReturnType())
                                           , G__get_tagnum(ifunc->TypeOf().ReturnType().RawType())
                                           , G__get_typenum(ifunc->TypeOf().ReturnType())
                                           , G__get_reftype(ifunc->TypeOf().ReturnType())
                                           , G__get_isconst(ifunc->TypeOf().ReturnType()) /* g++ may have problem */
                                          )
                         );
                  for (unsigned int fpind = 0; fpind < ifunc->FunctionParameterSize(); fpind++) {
                     if (fpind) fprintf(fp, ", ");
                     fprintf(fp, "%s"
                             , G__type2string(G__get_type(ifunc->TypeOf().FunctionParameterAt(fpind))
                                              , G__get_tagnum(ifunc->TypeOf().FunctionParameterAt(fpind).RawType())
                                              , G__get_typenum(ifunc->TypeOf().FunctionParameterAt(fpind))
                                              , G__get_reftype(ifunc->TypeOf().FunctionParameterAt(fpind))
                                              , G__get_isconst(ifunc->TypeOf().FunctionParameterAt(fpind))));
                  }
                  fprintf(fp, "))(&%s::%s) ) ", ifunc->DeclaringScope().Name(::Reflex::SCOPED).c_str(), ifunc->Name().c_str());
#else // G__OLDIMPLEMENTATION1993
                  fprintf(fp, ", (void*)%s::%s", ifunc->DeclaringScope().Name(::Reflex::SCOPED).c_str(), ifunc->Name().c_str());
#endif // G__OLDIMPLEMENTATION1993
               }
               else
                  fprintf(fp, ", (void*) NULL");

               fprintf(fp, ", %d", 1*ifunc->IsVirtual() + (ifunc->IsAbstract() * 2));
#else // 1289_YET || !1993
               fprintf(fp, ", (void*) NULL, %d", 1*ifunc->IsVirtual() + (ifunc->IsAbstract() * 2));
#endif // 1289_YET
#endif // G__TRUEP2F
               fprintf(fp, ");\n");
            } /* end of if access public && not pure virtual func */
            else { /* in case of protected,private or pure virtual func */
               // protected, private, pure virtual
               if (!strcmp(ifunc->Name().c_str(), G__struct.name[i])) {
                  ++isconstructor;
                  if (ifunc->IsCopyConstructor()) {
                     // copy constructor
                     ++iscopyconstructor;
                  }
               }
               else if ('~' == ifunc->Name().c_str()[0]) {
                  // destructor
                  ++isdestructor;
               }
               else if (!strcmp(ifunc->Name().c_str(), "operator new")) {
                  ++isconstructor;
                  ++iscopyconstructor;
               }
               else if (!strcmp(ifunc->Name().c_str(), "operator delete")) {
                  // destructor
                  ++isdestructor;
               }
#ifdef G__DEFAULTASSIGNOPR
               else if (!strcmp(ifunc->Name().c_str(), "operator=") && 'u' == G__get_type(ifunc->TypeOf().FunctionParameterAt(0)) && i == G__get_tagnum(ifunc->TypeOf().FunctionParameterAt(0).RawType())) {
                  // operator=
                  ++isassignmentoperator;
               }
#endif // G__DEFAULTASSIGNOPR
            } /* end of if access not public */
         } /* end for(j), loop over all ifuncs */

         int extra_pages = 0;
         if (
#ifndef G__OLDIMPLEMENTATON1656
            G__NOLINK == G__struct.iscpplink[i]
#endif // G__OLDIMPLEMENTATON1656
#ifndef G__OLDIMPLEMENTATON1730
            && G__ONLYMETHODLINK != G__struct.globalcomp[i]
#endif // G__OLDIMPLEMENTATON1730
         ) {

            /****************************************************************
            * setup default constructor
            ****************************************************************/

            if (0 == isconstructor) isconstructor = G__isprivateconstructor(i, 0);
            if ('n' == G__struct.type[i]) isconstructor = 1;
            if (0 == isconstructor && 0 == G__struct.isabstract[i] && 0 == isnonpublicnew) {
               sprintf(funcname, "%s", G__struct.name[i]);
               G__hash(funcname, hash, k);
               fprintf(fp, "   // automatic default constructor\n");
               fprintf(fp, "   G__memfunc_setup(");
               fprintf(fp, "\"%s\", %d, ", funcname, hash);
               fprintf(fp, "%s, ", G__map_cpp_funcname(i, funcname, 0, extra_pages));
               fprintf(fp, "(int) ('i'), ");
               if (strlen(G__struct.name[i]) > 25) fprintf(fp, "\n");
               fprintf(fp, "G__get_linked_tagnum(&%s), ", G__mark_linked_tagnum(i));
               fprintf(fp, "-1, "); /* typenum */
               fprintf(fp, "0, "); /* reftype */
               fprintf(fp, "0, "); /* para_nu */
               fprintf(fp, "1, "); /* ansi */
               fprintf(fp, "%d, 0", G__PUBLIC);
#ifdef G__TRUEP2F
               fprintf(fp, ", \"\", (char*) NULL, (void*) NULL, %d);\n", 0);
#else // G__TRUEP2F
               fprintf(fp, ", \"\", (char*) NULL);\n");
#endif // G__TRUEP2F
               ++extra_pages;
            } /* if(isconstructor) */

            /****************************************************************
            * setup copy constructor
            ****************************************************************/

            if (0 == iscopyconstructor) iscopyconstructor = G__isprivateconstructor(i, 1);
            if ('n' == G__struct.type[i]) iscopyconstructor = 1;
            if (0 == iscopyconstructor && 0 == G__struct.isabstract[i] && 0 == isnonpublicnew) {
               sprintf(funcname, "%s", G__struct.name[i]);
               G__hash(funcname, hash, k);
               fprintf(fp, "   // automatic copy constructor\n");
               fprintf(fp, "   G__memfunc_setup(");
               fprintf(fp, "\"%s\", %d, ", funcname, hash);
               fprintf(fp, "%s, ", G__map_cpp_funcname(i, funcname, 0, extra_pages));
               fprintf(fp, "(int) ('i'), ");
               if (strlen(G__struct.name[i]) > 20) fprintf(fp, "\n");
               fprintf(fp, "G__get_linked_tagnum(&%s), ", G__mark_linked_tagnum(i));
               fprintf(fp, "-1, "); /* typenum */
               fprintf(fp, "0, "); /* reftype */
               fprintf(fp, "1, "); /* para_nu */
               fprintf(fp, "1, "); /* ansi */
               fprintf(fp, "%d, 0", G__PUBLIC);
#ifdef G__TRUEP2F
               fprintf(fp, ", \"u '%s' - 11 - -\", (char*) NULL, (void*) NULL, %d);\n", G__fulltagname(i, 0), 0);
#else // G__TRUEP2F
               fprintf(fp, ", \"u '%s' - 11 - -\", (char*) NULL);\n", G__fulltagname(i, 0));
#endif // G__TRUEP2F
               ++extra_pages;
            }

            /****************************************************************
            * setup destructor
            ****************************************************************/

            if (0 == isdestructor) isdestructor = G__isprivatedestructor(i);
            if ('n' == G__struct.type[i]) isdestructor = 1;
            if ('n' != G__struct.type[i]) {
               sprintf(funcname, "~%s", G__struct.name[i]);
               G__hash(funcname, hash, k);
               fprintf(fp, "   // automatic destructor\n");
               fprintf(fp, "   G__memfunc_setup(");
               fprintf(fp, "\"%s\", %d, ", funcname, hash);
               if (0 == isdestructor)
                  fprintf(fp, "%s, ", G__map_cpp_funcname(i, funcname, 0, extra_pages));
               else
                  fprintf(fp, "(G__InterfaceMethod) NULL, ");
               fprintf(fp, "(int) ('y'), ");
               fprintf(fp, "-1, "); /* tagnum */
               fprintf(fp, "-1, "); /* typenum */
               fprintf(fp, "0, "); /* reftype */
               fprintf(fp, "0, "); /* para_nu */
               fprintf(fp, "1, "); /* ansi */
               fprintf(fp, "%d, 0", dtoraccess);
#ifdef G__TRUEP2F
               fprintf(fp, ", \"\", (char*) NULL, (void*) NULL, %d);\n", virtualdtorflag);
#else // G__TRUEP2F
               fprintf(fp, ", \"\", (char*) NULL);\n");
#endif // G__TRUEP2F
               if (0 == isdestructor) ++extra_pages;
            }

#ifdef G__DEFAULTASSIGNOPR
            /****************************************************************
            * setup assignment operator
            ****************************************************************/

            if (0 == isassignmentoperator) isassignmentoperator = G__isprivateassignopr(i);
            if ('n' == G__struct.type[i]) isassignmentoperator = 1;
            if (0 == isassignmentoperator) {
               sprintf(funcname, "operator=");
               G__hash(funcname, hash, k);
               fprintf(fp, "   // automatic assignment operator\n");
               fprintf(fp, "   G__memfunc_setup(");
               fprintf(fp, "\"%s\", %d, ", funcname, hash);
               fprintf(fp, "%s, ", G__map_cpp_funcname(i, funcname, 0, extra_pages));
               fprintf(fp, "(int) ('u'), ");
               fprintf(fp, "G__get_linked_tagnum(&%s), ", G__mark_linked_tagnum(i));
               fprintf(fp, "-1, "); /* typenum */
               fprintf(fp, "1, "); /* reftype */
               fprintf(fp, "1, "); /* para_nu */
               fprintf(fp, "1, "); /* ansi */
               fprintf(fp, "%d, 0", G__PUBLIC);
#ifdef G__TRUEP2F
               fprintf(fp, ", \"u '%s' - 11 - -\", (char*) NULL, (void*) NULL, %d);\n", G__fulltagname(i, 0), 0);
#else // G__TRUEP2F
               fprintf(fp, ", \"u '%s' - 11 - -\", (char*) NULL);\n", G__fulltagname(i, 0));
#endif // G__TRUEP2F
            }
#endif // G__DEFAULTASSIGNOPR
         } /* end while(ifunc) */
         fprintf(fp, "   G__tag_memfunc_reset();\n");
         fprintf(fp, "}\n\n");
      } /* end if(globalcomp) */
   } /* end for(i) */

   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Member function information setup\n");
   fprintf(fp, "*********************************************************/\n");

   if (G__globalcomp == G__CPPLINK) {
      fprintf(fp, "extern \"C\" void G__cpp_setup_memfunc%s() {\n", G__DLLID);
   }
   else {
      /* fprintf(fp, "void G__c_setup_memfunc%s() {\n", G__DLLID); */
   }

   fprintf(fp, "}\n");

#endif // G__SMALLOBJECT
   // --
}

//______________________________________________________________________________
void Cint::Internal::G__cpplink_global(FILE* fp)
{
   int divn = 0;
   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Global variable information setup for each class\n");
   fprintf(fp, "*********************************************************/\n");
#ifdef G__BORLANDCC5
   fprintf(fp, "static void G__cpp_setup_global%d(void) {\n", divn++);
#else // G__BORLANDCC5
   fprintf(fp, "static void G__cpp_setup_global%d() {\n", divn++);
#endif // G__BORLANDCC5
   fprintf(fp, "\n   /* Setting up global variables */\n");
   fprintf(fp, "   G__resetplocal();\n\n");
   //
   //  Loop over all known global variables.
   //
   int maxfnc = 100;
   int fnc = 0;
   ::Reflex::Scope varscope = ::Reflex::Scope::GlobalScope();
   for (::Reflex::Member_Iterator mbr_iter = varscope.DataMember_Begin(); mbr_iter != varscope.DataMember_End(); ++mbr_iter) {
      if (fnc++ > maxfnc) { // Make a new section.
         fnc = 0;
         fprintf(fp, "}\n\n");
#ifdef G__BORLANDCC5
         fprintf(fp, "static void G__cpp_setup_global%d(void) {\n", divn++);
#else // G__BORLANDCC5
         fprintf(fp, "static void G__cpp_setup_global%d() {\n", divn++);
#endif // G__BORLANDCC5
         // --
      }
      if ( // Variable is marked for dictionary generation.
         (
            (G__get_properties(*mbr_iter)->statictype == G__AUTO) || // not static, or
            ( // extern type v[];
               !G__get_offset(*mbr_iter) && // no storage, and
               (G__get_properties(*mbr_iter)->statictype == G__COMPILEDGLOBAL) && // marked as having preallocated storage, and
               (G__get_varlabel(*mbr_iter, 1) /* number of elements */ == INT_MAX /* unspecified length flag */) // is an unspecified length array
            )
         ) && // and,
         (G__get_properties(*mbr_iter)->globalcomp < G__NOLINK) && // is marked for dictionary generation, and
#ifndef G__OLDIMPLEMENTATION2191
         (tolower(G__get_type(mbr_iter->TypeOf())) != 'j') &&
#else // G__OLDIMPLEMENTATION2191
         (tolower(G__get_type(mbr_iter->TypeOf())) != 'm') &&
#endif // G__OLDIMPLEMENTATION2191
         mbr_iter->Name().c_str()[0] // is named
      ) { // Variable is marked for dictionary generation.
         //
         //  Write a data member setup call to the dictionary file.
         //
         char type = '\0';
         int tagnum = -1;
         int typenum = -1;
         int reftype = 0;
         int isconst = 0;
         G__get_cint5_type_tuple(mbr_iter->TypeOf(), &type, &tagnum, &typenum, &reftype, &isconst);
         G__RflxVarProperties* prop = G__get_properties(*mbr_iter);
         int pvoidflag = 0;
         if ( // Is enumerator or unaddressable bool.
            (
               islower(type) && // not a pointer, and
               isconst && // const, and
               mbr_iter->TypeOf().RawType().IsEnum() // is an enumerator member
            ) || // or,
            (tolower(type) == 'p') ||
            (type == 'T')
#ifdef G__UNADDRESSABLEBOOL
            || (type == 'g') // or, is an unaddressable bool
#endif // G__UNADDRESSABLEBOOL
            || (prop->statictype == G__LOCALSTATIC && isconst && // const static
                islower(type) && type != 'u' && // of fundamental
                G__get_offset(*mbr_iter)) // with initializer
               // --
         ) { // Is enumerator or unaddressable bool.
            pvoidflag = 1; // Pass a null pointer as the address of these things.
         }
         fprintf(fp, "   G__memvar_setup(");
         if (pvoidflag) { // Special case, is enumerator or unaddressable bool, pass G__PVOID (is -1).
            fprintf(fp, "(void*)G__PVOID,"); // No storage, pass address of data member as G__PVOID (is -1).
         }
         else {
            fprintf(fp, "(void*)(&%s),", mbr_iter->Name().c_str()); // Address of data member.
#ifdef G__GENWINDEF
            fprintf(G__WINDEFfp, "        %s @%d\n", mbr_iter->Name().c_str(), ++G__nexports);
#endif // G__GENWINDEF
            // --
         }
         //
         //  Type code, referenceness, pointer level, and constness.
         //
         fprintf(fp, "%d,", type);
         fprintf(fp, "%d,", reftype);
         fprintf(fp, "%d,", isconst);
         //
         //  Tagnum of data type, if not fundamental.
         //
         if (tagnum != -1) {
            fprintf(fp, "G__get_linked_tagnum(&%s),", G__mark_linked_tagnum(tagnum));
         }
         else {
            fprintf(fp, "-1,");
         }
         //
         //  Typenum of data type, if it is a typedef.
         //
         if (typenum != -1) {
               ::Reflex::Type ty = mbr_iter->TypeOf();
               for (; !ty.IsTypedef(); ty = ty.ToType()) {}
               std::string tmp = ty.Name();
               // Remove any array bounds in the name.
               std::string::size_type pos = tmp.find("[");
               if (pos != std::string::npos) {
                  tmp.erase(pos);
               }
               fprintf(fp, "G__defined_typename(\"%s\"),", tmp.c_str());
         }
         else {
            fprintf(fp, "-1,");
         }
         //
         //  Storage duration and staticness, member access.
         //
         fprintf(fp, "%d,", prop->statictype);
         fprintf(fp, "%d,", G__get_access(*mbr_iter));
         //
         //  Name and array dimensions (quoted) as the
         //  left hand side of an assignment expression.
         //
         fprintf(fp, "\"%s", mbr_iter->Name().c_str());
         if (G__get_varlabel(*mbr_iter, 1) /* number of elements */ == INT_MAX /* unspecified length flag */) {
            fprintf(fp, "[]");
         }
         else if (G__get_varlabel(*mbr_iter, 1) /* number of elements */) {
            fprintf(fp, "[%d]", G__get_varlabel(*mbr_iter, 1) / G__get_varlabel(*mbr_iter, 0));
         }
         for (int k = 1; k < G__get_paran(*mbr_iter); ++k) {
            fprintf(fp, "[%d]", G__get_varlabel(*mbr_iter, k + 1));
         }
         if (pvoidflag) { // Is enumerator or unaddressable bool.
            G__value buf = G__getitem((char*) mbr_iter->Name().c_str());
            char value[G__ONELINE];
            G__string(buf, value);
            char ttt[G__ONELINE];
            G__quotedstring(value, ttt);
            if ((tolower(type) == 'p') || (type == 'T')) {
               fprintf(fp, "=%s\",1,(char*)NULL);\n", ttt);
            }
            else {
               fprintf(fp, "=%s\",0,(char*)NULL);\n", ttt);
            }
         }
         else {
            fprintf(fp, "=\",0,(char*)NULL);\n");
         }
      }
      G__var_type = 'p';
   }
   fprintf(fp, "\n");
   fprintf(fp, "   G__resetglobalenv();\n");
   fprintf(fp, "}\n");
   if (G__globalcomp == G__CPPLINK) {
      fprintf(fp, "extern \"C\" void G__cpp_setup_global%s() {\n", G__DLLID);
   }
   else {
      fprintf(fp, "void G__c_setup_global%s() {\n", G__DLLID);
   }
   for (int i = 0; i < divn; ++i) {
      fprintf(fp, "  G__cpp_setup_global%d();\n", i);
   }
   fprintf(fp, "}\n");
}

#ifdef G__P2FDECL
//______________________________________________________________________________
static void G__declaretruep2f(FILE* fp, G__ifunc_table& ifunc, int j)
{
   // --
#ifdef G__P2FDECL
   int i;
   int ifndefflag = 1;
   if (strncmp(ifunc->Name().c_str(), "operator", 8) == 0) ifndefflag = 0;
   if (ifndefflag) {
      switch (G__globalcomp) {
         case G__CPPLINK:
            if (G__MACROLINK == G__get_funcproperties(*ifunc)->globalcomp ||
                  0 ==
                  strcmp("iterator_category", ifunc->Name().c_str())) fprintf(fp, "#if 0\n");
            else fprintf(fp, "#ifndef %s\n", ifunc->Name().c_str());
            fprintf(fp, "%s (*%sp2f)("
                    , G__type2string(G__get_type(ifunc->TypeOf().ReturnType())
                                     , G__get_tagnum(ifunc->TypeOf().ReturnType().RawType())
                                     , ifunc->TypeOf().ReturnType()
                                     , G__get_reftype(ifunc->TypeOf().ReturnType())
                                     , G__get_isconst(ifunc->TypeOf().ReturnType())
                                     /* ,0  avoiding g++ bug */
                                    )
                    , G__map_cpp_funcname(-1, ifunc->Name().c_str(), j, ifunc->page)
                   );
            for (i = 0;i < ifunc->FunctionParameterSize();i++) {
               if (i) fprintf(fp, ", ");
               fprintf(fp, "%s"
                       , G__type2string(ifunc->para_type[j][i]
                                        , ifunc->para_p_tagtable[j][i]
                                        , ifunc->para_p_typetable[j][i]
                                        , ifunc->para_reftype[j][i]
                                        , ifunc->para_isconst[j][i]));
            }
            fprintf(fp, ") = %s;\n", ifunc->Name().c_str());
            fprintf(fp, "#else\n");
            fprintf(fp, "void* %sp2f = (void*) NULL;\n"
                    , G__map_cpp_funcname(-1, ifunc->Name().c_str(), j, ifunc->page));
            fprintf(fp, "#endif\n");
            break;
         default:
            break;
      }
   }
#else // G__P2FDECL
   if (fp && ifunc && j) return;
#endif // G__P2FDECL
   // --
}
#endif // G__P2FDECL

#ifdef G__TRUEP2F
//______________________________________________________________________________
static void G__printtruep2f(FILE* fp, const ::Reflex::Member& ifunc)
{
#if defined(G__P2FCAST)
   int i;
#endif // G__P2FCAST
   int ifndefflag = 1;
#if defined(G__FUNCPOINTER)
   if (strncmp(ifunc.Name().c_str(), "operator", 8) == 0)
      ifndefflag = 0;
#else // G__FUNCPOINTER
   ifndefflag = 0;
#endif // G__FUNCPOINTER
   if (ifndefflag) {
      switch (G__globalcomp) {
         case G__CPPLINK:
#if defined(G__P2FDECL)
            fprintf(fp, ", (void*) %sp2f);\n"
                    , G__map_cpp_funcname(-1, ifunc.Name().c_str(), j, 0 /*ifunc.page*/));
#elif defined(G__P2FCAST)
            if (
               (G__get_funcproperties(ifunc)->globalcomp == G__MACROLINK) ||
               !strcmp("iterator_category", ifunc.Name().c_str())
            ) {
               fprintf(fp, "#if 0\n");
            }
            else {
               fprintf(fp, "#ifndef %s\n", ifunc.Name().c_str());
            }
            fprintf(fp,
                    ", (void*) (%s (*)("
                    , G__type2string(
                       G__get_type(ifunc.TypeOf().ReturnType())
                       , G__get_tagnum(ifunc.TypeOf().ReturnType().RawType())
                       , G__get_typenum(ifunc.TypeOf().ReturnType())
                       , G__get_reftype(ifunc.TypeOf().ReturnType())
                       , G__get_isconst(ifunc.TypeOf().ReturnType())
                    )
                    //, G__map_cpp_funcname(-1,ifunc.Name().c_str(),j,ifunc.page)
                   );
            for (i = 0; i < ifunc.FunctionParameterSize(); ++i) {
               if (i) {
                  fprintf(fp, ", ");
               }
               //fprintf(fp, "%s", ifunc.TypeOf().FunctionParameterAt(i).Name(::Reflex::SCOPED).c_str());
               fprintf(fp, "%s", G__type2string(
                          G__get_type(ifunc.TypeOf().FunctionParameterAt(i))
                          , G__get_tagnum(ifunc.TypeOf().FunctionParameterAt(i).RawType())
                          , G__get_typenum(ifunc.TypeOf().FunctionParameterAt(i))
                          , G__get_reftype(ifunc.TypeOf().FunctionParameterAt(i))
                          , G__get_isconst(ifunc.TypeOf().FunctionParameterAt(i))
                       )
                      );
               //,G__type2string(ifunc.para_type[j][i]
               //                ,ifunc.para_p_tagtable[j][i]
               //                ,G__get_typenum(ifunc.para_p_typetable[j][i])
               //                ,ifunc.para_reftype[j][i]
               //                ,ifunc.para_isconst[j][i]));
            }
            fprintf(fp,
                    "))%s, %d);\n"
                    , ifunc.Name().c_str()
                    , (ifunc.IsVirtual() * 1) + (ifunc.IsAbstract() * 2)
                   );
            fprintf(fp, "#else\n");
            fprintf(fp, ", (void*) NULL, %d);\n", (ifunc.IsVirtual()) * 1 + (ifunc.IsAbstract() * 2));
            fprintf(fp, "#endif\n");
#else // G__P2FCAST
            fprintf(fp, ", (void*) NULL, %d);\n", (ifunc.IsVirtual() * 1) + (ifunc.IsAbstract() * 2));
#endif // G__P2FDECL, G__P2FCAST
            break;
         case G__CLINK:
         default:
            fprintf(fp, "#ifndef %s\n", ifunc.Name().c_str());
            fprintf(fp, ", (void*) %s, %d);\n", ifunc.Name().c_str()
                    , ifunc.IsVirtual()*1 + ifunc.IsAbstract()*2);
            fprintf(fp, "#else\n");
            fprintf(fp, ", (void*) NULL, %d);\n"
                    , ifunc.IsVirtual()*1 + ifunc.IsAbstract()*2);
            fprintf(fp, "#endif\n");
            break;
      }
   }
   else {
      fprintf(fp, ", (void*) NULL, %d);\n"
              , ifunc.IsVirtual()*1 + ifunc.IsAbstract()*2);
   }
}
#endif // G__TRUEP2F

//______________________________________________________________________________
void Cint::Internal::G__cpplink_func(FILE* fp)
{
   // -- Making C++ link routine to global function.
   unsigned int k;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   int divn = 0;
   int maxfnc = 100;
   int fnc = 0;

   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Global function information setup for each class\n");
   fprintf(fp, "*********************************************************/\n");

#ifdef G__BORLANDCC5
   fprintf(fp, "static void G__cpp_setup_func%d(void) {\n", divn++);
#else
   fprintf(fp, "static void G__cpp_setup_func%d() {\n", divn++);
#endif

   fprintf(fp, "   G__lastifuncposition();\n\n");

   ::Reflex::Scope scope(::Reflex::Scope::GlobalScope());
   for (::Reflex::Member_Iterator ifunc = scope.FunctionMember_Begin();
         ifunc != scope.FunctionMember_End();
         ++ifunc) {
      if (fnc++ > maxfnc) {
         fnc = 0;
         fprintf(fp, "}\n\n");
#ifdef G__BORLANDCC5
         fprintf(fp, "static void G__cpp_setup_func%d(void) {\n", divn++);
#else
         fprintf(fp, "static void G__cpp_setup_func%d() {\n", divn++);
#endif
      }
      if (G__NOLINK > G__get_funcproperties(*ifunc)->globalcomp &&  /* with -c-1 option */
            G__test_access(*ifunc, G__PUBLIC) && /* public, this is always true */
            !ifunc->IsStatic()
            /* && ifunc->hash[j] */) {   /* not static */

         if (strcmp(ifunc->Name().c_str(), "operator new") == 0 &&
               (ifunc->FunctionParameterSize() == 2 || ifunc->FunctionParameterDefaultAt(2).c_str()[0])) {
            G__is_operator_newdelete |= G__IS_OPERATOR_NEW;
         }
         else if (strcmp(ifunc->Name().c_str(), "operator delete") == 0) {
            G__is_operator_newdelete |= G__IS_OPERATOR_DELETE;
         }

#ifdef G__P2FDECL  /* used to be G__TRUEP2F */
         G__declaretruep2f(fp, ifunc, j);
#endif

         /* function name and return type */
         fprintf(fp, "   G__memfunc_setup(");
         {
            int hash = 0;
            int junk = 0;
            G__hash(ifunc->Name().c_str(), hash, junk)
            fprintf(fp, "\"%s\", %d, ", ifunc->Name().c_str(), hash);
         }
         fprintf(fp, "%s, ",::G__map_cpp_funcname(*ifunc));
         fprintf(fp, "%d, ", G__get_type(ifunc->TypeOf().ReturnType()));

         if (-1 != G__get_tagnum(ifunc->TypeOf().ReturnType().RawType()))
            fprintf(fp, "G__get_linked_tagnum(&%s), "
                    , G__mark_linked_tagnum(G__get_tagnum(ifunc->TypeOf().ReturnType().RawType())));
         else
            fprintf(fp, "-1, ");

         if (G__get_cint5_typenum(ifunc->TypeOf().ReturnType()) != -1) {
            ::Reflex::Type ty = ifunc->TypeOf().ReturnType();
            for ( ; !ty.IsTypedef(); ty = ty.ToType()) {
               // Intentionally empty
            }
            fprintf(fp, "G__defined_typename(\"%s\"), ", ty.Name().c_str());
         }
         else {
            fprintf(fp, "-1, ");
         }

         fprintf(fp, "%d, ", G__get_reftype(ifunc->TypeOf().ReturnType()));

         fprintf(fp, "%ld, ", (long)ifunc->FunctionParameterSize());

         if (2 == G__get_funcproperties(*ifunc)->entry.ansi)
            fprintf(fp, "%d, ", 8 + ifunc->IsStatic()*2);
         else
            fprintf(fp, "%d, ", G__get_funcproperties(*ifunc)->entry.ansi + ifunc->IsStatic()*2);
         fprintf(fp, "%d, ", G__get_access(*ifunc));
         fprintf(fp, "%d, ", G__get_isconst(ifunc->TypeOf()));

         /* newline to avoid lines more than 256 char for CMZ */
         if (ifunc->FunctionParameterSize() > 1) fprintf(fp, "\n");
         fprintf(fp, "\"");

         /****************************************************************
         * function parameter
         ****************************************************************/
         for (k = 0; k < ifunc->FunctionParameterSize(); ++k) {
            char type = '\0';
            int tagnum = -1;
            int typenum = -1;
            int reftype = 0;
            int isconst = 0;
            G__get_cint5_type_tuple(ifunc->TypeOf().FunctionParameterAt(k), &type, &tagnum, &typenum, &reftype, &isconst);
            // newline to avoid lines more than 256 char for CMZ
            if ((G__globalcomp == G__CPPLINK) && k && !(k % 2)) {
               fprintf(fp, "\"\n\"");
            }
            if (isprint(type)) {
               fprintf(fp, "%c ", type);
            }
            else {
               G__fprinterr(G__serr, "Internal error: function parameter type\n");
               fprintf(fp, "%d ", type);
            }

            if (tagnum != -1) {
               fprintf(fp, "'%s' ", G__fulltagname(tagnum, 0));
               G__mark_linked_tagnum(tagnum);
            }
            else {
               fprintf(fp, "- ");
            }

            if (typenum != -1) {
               fprintf(fp, "'%s' ", G__fulltypename(G__Dict::GetDict().GetTypedef(typenum)).c_str());
            }
            else {
               fprintf(fp, "- ");
            }

            fprintf(fp, "%d ", reftype + (isconst * 10));
            if (ifunc->FunctionParameterDefaultAt(k).c_str()[0])
               fprintf(fp, "'%s' ", G__quotedstring((char*)ifunc->FunctionParameterDefaultAt(k).c_str(), buf));
            else fprintf(fp, "- ");
            if (ifunc->FunctionParameterNameAt(k).c_str()[0])
               fprintf(fp, "%s", ifunc->FunctionParameterNameAt(k).c_str());
            else fprintf(fp, "-");
            if (k != ifunc->FunctionParameterSize() - 1) fprintf(fp, " ");
         }
#ifdef G__TRUEP2F
         fprintf(fp, "\", (char*) NULL\n");
         G__printtruep2f(fp, *ifunc);
#else
         fprintf(fp, "\", (char*) NULL);\n");
#endif

      }
   } /* end for(j) */

   fprintf(fp, "\n");
   fprintf(fp, "   G__resetifuncposition();\n");

   /********************************************************
   * call user initialization function if specified
   ********************************************************/
   if (G__INITFUNC) {
      fprintf(fp, "  %s();\n", G__INITFUNC);
   }

   fprintf(fp, "}\n\n");

   if (G__CPPLINK == G__globalcomp) {
      fprintf(fp, "extern \"C\" void G__cpp_setup_func%s() {\n", G__DLLID);
   }
   else {
      fprintf(fp, "void G__c_setup_func%s() {\n", G__DLLID);
   }
   for (fnc = 0;fnc < divn;fnc++) {
      fprintf(fp, "  G__cpp_setup_func%d();\n", fnc);
   }
   fprintf(fp, "}\n");
}

//______________________________________________________________________________
char G__incsetup_exist(std::list<G__incsetup>* incsetuplist, G__incsetup incsetup)
{
   if(incsetuplist->empty()) return 0;
   std::list<G__incsetup>::iterator iter;
   for (iter=incsetuplist->begin(); iter != incsetuplist->end(); ++iter)
      if (*iter==incsetup)
         return 1;

   return 0;
}

//______________________________________________________________________________
extern "C" int G__tagtable_setup(int tagnum, int size, int cpplink, int isabstract, const char* comment, G__incsetup setup_memvar, G__incsetup setup_memfunc)
{
   // -- FIXME: Describe this function!
   char* p;
#ifndef G__OLDIMPLEMENTATION1823
   G__StrBuf xbuf_sb(G__BUFLEN);
   char* xbuf = xbuf_sb;
   char* buf = xbuf;
#else // G__OLDIMPLEMENTATION1823
   G__StrBuf buf_sb(G__ONELINE);
   char* buf = buf_sb;
#endif // G__OLDIMPLEMENTATION1823
   if (!G__struct.incsetup_memvar[tagnum]) {
      G__struct.incsetup_memvar[tagnum] = new std::list<G__incsetup>();
   }
   if (!G__struct.incsetup_memfunc[tagnum]) {
      G__struct.incsetup_memfunc[tagnum] = new std::list<G__incsetup>();
   }
   if (!size && G__struct.size[tagnum] && (G__struct.type[tagnum] != 'n')) {
      return 0;
   }
   if (
      ((G__struct.type[tagnum] != 'n') && G__struct.size[tagnum]) || // Class already setup
      ((G__struct.type[tagnum] == 'n') && G__struct.iscpplink[tagnum]) // Namespace already setup
   ) {
      // --
#ifndef G__OLDIMPLEMENTATION1656
      char found = G__incsetup_exist(G__struct.incsetup_memvar[tagnum], setup_memvar);
      // If setup_memvar is not NULL we push the G__setup_memvarXXX pointer into the list
      if (setup_memvar && !found) {
         G__struct.incsetup_memvar[tagnum]->push_back(setup_memvar);
      }
      found = G__incsetup_exist(G__struct.incsetup_memfunc[tagnum], setup_memfunc);
      // If setup_memfunc is not NULL we push the G__setup_memfuncXXX pointer into the list
      if (setup_memfunc && !found) {
         G__struct.incsetup_memfunc[tagnum]->push_back(setup_memfunc);
      }
#endif // G__OLDIMPLEMENTATION1656
      if (G__asm_dbg) {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: Try to reload %s from DLL\n", G__fulltagname(tagnum, 1));
         }
      }
      return 0;
   }
   G__struct.size[tagnum] = size;
   Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
   G__RflxProperties* prop = G__get_properties(scope);
   if (scope.IsClass() || scope.IsUnion()) {
      Reflex::Type type = scope;
      type.SetSize(size);
   }
   prop->iscpplink = cpplink;
   G__struct.iscpplink[tagnum] = cpplink;
#ifndef G__OLDIMPLEMENTATION1545
   G__struct.rootflag[tagnum] = (isabstract / 0x10000) % 0x100;
   G__struct.funcs[tagnum] = (isabstract / 0x100) % 0x100;
   G__struct.isabstract[tagnum] = isabstract % 0x100;
#else // G__OLDIMPLEMENTATION1545
   G__struct.funcs[tagnum] = isabstract / 0x100;
   G__struct.isabstract[tagnum] = isabstract % 0x100;
#endif // G__OLDIMPLEMENTATION1545
   G__struct.filenum[tagnum] = G__ifile.filenum;
   prop->filenum = G__ifile.filenum;
   prop->comment.p.com = (char*) comment;
   if (comment) {
      prop->comment.filenum = -2;
   }
   else {
      prop->comment.filenum = -1;
   }
   if (!scope.DataMemberSize() || scope.IsNamespace()) {
         char found = G__incsetup_exist(G__struct.incsetup_memvar[tagnum], setup_memvar);
         if (setup_memvar && !found) {
            G__struct.incsetup_memvar[tagnum]->push_back(setup_memvar);
         }
   }
   if (
      !G__Dict::GetDict().GetScope(tagnum).FunctionMemberSize() ||
      (G__struct.type[tagnum]  == 'n') ||
      (
         // --
#ifndef G__OLDIMPLEMENTATION2027
         (G__get_funcproperties(G__Dict::GetDict().GetScope(tagnum).FunctionMemberAt(0))->entry.size != -1) &&
#else // G__OLDIMPLEMENTATION2027
         (G__struct.memfunc[tagnum]->pentry[0]->size != -1) &&
#endif // G__OLDIMPLEMENTATION2027
         (G__Dict::GetDict().GetScope(tagnum).FunctionMemberSize() <= 2)
      )
   ) {
      char found = 0;
      found = G__incsetup_exist(G__struct.incsetup_memfunc[tagnum], setup_memfunc);
      if (setup_memfunc && !found) {
         G__struct.incsetup_memfunc[tagnum]->push_back(setup_memfunc);
      }
   }
   // add template names
#ifndef G__OLDIMPLEMENTATION1823
   if (strlen(G__struct.name[tagnum]) > (G__BUFLEN - 10)) {
      buf = (char*) malloc(strlen(G__struct.name[tagnum]) + 10);
   }
#endif // G__OLDIMPLEMENTATION1823
   strcpy(buf, G__struct.name[tagnum]);
   if ((p = strchr(buf, '<'))) {
      *p = '\0';
      if (!G__defined_templateclass(buf)) {
         ::Reflex::Scope store_def_tagnum = G__def_tagnum;
         ::Reflex::Scope store_tagdefining = G__tagdefining;
         FILE* store_fp = G__ifile.fp;
         G__ifile.fp = 0;
         G__def_tagnum = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[tagnum]);
         G__tagdefining = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[tagnum]);
         G__createtemplateclass(buf, 0, 0);
         G__ifile.fp = store_fp;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
      }
   }
#ifndef G__OLDIMPLEMENTATION1823
   if (buf != xbuf) {
      free(buf);
   }
#endif // G__OLDIMPLEMENTATION1823
   return 0;
}

//______________________________________________________________________________
extern "C" int G__inheritance_setup(int tagnum, int basetagnum, long baseoffset, int baseaccess, int property)
{
   // --
#ifndef G__SMALLOBJECT
   G__ASSERT(0 <= tagnum && 0 <= basetagnum);
   G__struct.baseclass[tagnum]->vec.push_back(G__inheritance::G__Entry(basetagnum,(char*)baseoffset, baseaccess, property));
#endif // G__SMALLOBJECT
   return(0);
}

//______________________________________________________________________________
extern "C" int G__tag_memvar_setup(int tagnum)
{
   // --
#ifndef G__OLDIMPLEMENTATON285
  
   /* Variables stack storing */
   G__IncSetupStack incsetup_stack;
   std::stack<G__IncSetupStack> *var_stack = G__stack_instance(); 

   incsetup_stack.G__incset_tagnum = G__tagnum;
   incsetup_stack.G__incset_p_local = G__p_local;
   incsetup_stack.G__incset_def_struct_member = G__def_struct_member;
   incsetup_stack.G__incset_tagdefining = G__tagdefining;
   incsetup_stack.G__incset_globalvarpointer = G__globalvarpointer;
   incsetup_stack.G__incset_var_type = G__var_type ;
   incsetup_stack.G__incset_typenum = G__typenum ;
   incsetup_stack.G__incset_static_alloc = G__static_alloc ;
   incsetup_stack.G__incset_access = G__access ;
 
#endif // G__OLDIMPLEMENTATON285
   G__tagnum = G__Dict::GetDict().GetScope(tagnum);
   G__p_local= G__tagnum;
   G__def_struct_member = 1;
   incsetup_stack.G__incset_def_tagnum = G__def_tagnum;
   G__def_tagnum = G__tagnum.DeclaringScope();
   G__tagdefining=G__tagnum;

   var_stack->push(incsetup_stack);

   return 0;
}

//______________________________________________________________________________
extern "C" int G__memvar_setup(void* p, int type, int reftype, int constvar, int tagnum, int typenum, int statictype, int accessin, const char* expr, int definemacro, const char* comment)
{
   int store_in_memvar_setup = G__in_memvar_setup;
   G__in_memvar_setup = 1;
   int store_def_struct_member = G__def_struct_member;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   ::Reflex::Scope store_p_local = G__p_local;
   if ((type == 'p') && G__def_struct_member) {
      G__def_struct_member = 0;
      G__tagdefining = ::Reflex::Scope();
      G__p_local = ::Reflex::Scope();
   }
   G__setcomment = (char*) comment; // FIXME: We don't save this!
   G__globalvarpointer = (char*) p; // FIXME: We don't save this!
   G__var_type = type; // FIXME: We don't save this!
   G__tagnum = G__Dict::GetDict().GetScope(tagnum); // FIXME: We don't save this!
   G__typenum = G__Dict::GetDict().GetTypedef(typenum); // FIXME: We don't save this!
   G__reftype = reftype; // FIXME: We don't save this!
   int store_constvar = G__constvar;
   G__constvar = constvar;
   //int save_static_alloc = G__static_alloc; // FIXME: We probably need a line like this here.
   if ((statictype == G__AUTO) || (statictype == G__AUTOARYDISCRETEOBJ)) {
      G__static_alloc = 0;
   }
   else if (statictype == G__LOCALSTATIC) {
      G__static_alloc = 1;
   }
   else if (statictype == G__COMPILEDGLOBAL) {
      G__static_alloc = 1; // FIXME: This is probaby wrong!
   }
   else {
      G__static_alloc = 1; // File scope static variable, which is actually a static data member.
   }
   //int store_access = G__access; // FIXME: We probably need a line like this here.
   G__access = accessin; // FIXME: We don't save this!
   G__definemacro = definemacro;
   int store_asm_noverflow = G__asm_noverflow;
   G__asm_noverflow = 0;
   int store_prerun = G__prerun;
   G__prerun = 1;
   int store_asm_wholefunction = G__asm_wholefunction;
   G__asm_wholefunction = G__ASM_FUNC_NOP;
   //
   G__getexpr((char*)expr);
   //
   G__asm_wholefunction = store_asm_wholefunction;
   G__prerun = store_prerun;
   G__asm_noverflow = store_asm_noverflow;
   G__definemacro = 0; // FIXME: Shouldn't we restore this?
   //G__access = store_access; // FIXME: We probably need a line like this here.
   //G__static_alloc = save_static_alloc; // FIXME: We probably need a line like this here.
   G__constvar = store_constvar;
   G__reftype = G__PARANORMAL; // FIXME: We should probably restore this instead.
   G__setcomment = 0; // FIXME: Shouldn't we restore this?
   if ((type == 'p') && store_def_struct_member) {
      G__def_struct_member = store_def_struct_member;
      G__tagdefining = store_tagdefining;
      G__p_local = store_p_local;
   }
   G__in_memvar_setup = store_in_memvar_setup;
   return 0;
}

//______________________________________________________________________________
extern "C" int G__tag_memvar_reset()
{
   /* Variables stack restoring */
   std::stack<G__IncSetupStack> *var_stack = G__stack_instance(); 

   G__IncSetupStack *incsetup_stack = &var_stack->top();

   G__p_local = incsetup_stack->G__incset_p_local ;
   G__def_struct_member = incsetup_stack->G__incset_def_struct_member ;
   G__tagdefining = incsetup_stack->G__incset_tagdefining ;
   G__def_tagnum = incsetup_stack->G__incset_def_tagnum;

   G__globalvarpointer = incsetup_stack->G__incset_globalvarpointer ;
   G__var_type = incsetup_stack->G__incset_var_type ;
   G__tagnum = incsetup_stack->G__incset_tagnum ;
   G__typenum = incsetup_stack->G__incset_typenum ;
   G__static_alloc = incsetup_stack->G__incset_static_alloc ;
   G__access = incsetup_stack->G__incset_access ;

   var_stack->pop();

   return(0);
}

//______________________________________________________________________________
extern "C" int G__usermemfunc_setup(char* funcname, int hash, int (*funcp)(), int type, int tagnum, int typenum, int reftype, int para_nu, int ansi, int accessin, int isconst, char* paras, char* comment, void* truep2f, int isvirtual, void* userparam)
{
   int ret = G__memfunc_setup(funcname, hash, (G__InterfaceMethod) funcp, type, tagnum, typenum, reftype, para_nu, ansi, accessin, isconst, paras, comment, truep2f, isvirtual);
   G__get_funcproperties(*(--G__p_ifunc.FunctionMember_REnd()))->entry.userparam = userparam; // Add the userparam to the last created function member.
   return ret;
}

//______________________________________________________________________________
extern "C" int G__tag_memfunc_setup(int tagnum)
{
   /* Variables stack storing */
   G__IncSetupStack incsetup_stack;
   std::stack<G__IncSetupStack>* var_stack = G__stack_instance();

   incsetup_stack.G__incset_p_ifunc = G__p_ifunc;
   incsetup_stack.G__incset_tagnum = G__tagnum;
   incsetup_stack.G__incset_func_now = G__func_now;
   incsetup_stack.G__incset_tagdefining = G__tagdefining;
   incsetup_stack.G__incset_var_type = G__var_type;
   incsetup_stack.G__incset_def_tagnum = G__def_tagnum;

   var_stack->push(incsetup_stack);

   G__tagdefining =  G__Dict::GetDict().GetScope(G__struct.parent_tagnum[tagnum]);
   G__def_tagnum = G__tagdefining;
   G__tagnum = G__Dict::GetDict().GetScope(tagnum);
   G__p_ifunc = G__tagnum;

   G__memfunc_next();
   return(0);
}

//______________________________________________________________________________
extern "C" int G__memfunc_setup(const char* funcname, int /*hash*/, G__InterfaceMethod funcp, int type, int tagnum, int typenum, int reftype, int para_nu, int ansi, int accessin, int isconst, const char* paras, const char* comment, void* truep2f, int isvirtual)
{
   // Create a dictionary information for a member function.
   G__BuilderInfo builder;
   builder.fProp.entry.p = (void*) funcp;
   builder.fProp.entry.size = -1;
   if (G__p_ifunc && !G__p_ifunc.IsTopScope()) { // Parent is a class or namespace, and not the global namespace.
      builder.fProp.filenum = G__get_properties(G__p_ifunc)->filenum;
   }
   else {
      builder.fProp.filenum = G__ifile.filenum;
   }
   builder.fProp.entry.line_number = -1;
   builder.fProp.entry.bytecode = 0;
   if (truep2f) {
      builder.fProp.entry.tp2f = truep2f;
   }
   else {
      builder.fProp.entry.tp2f = (void*) funcp;
   }
   builder.fReturnType = G__cint5_tuple_to_type(type, tagnum, typenum, reftype, isconst);
   //{
   //   bool return_is_const = isconst & ~G__CONSTFUNC;
   //   builder.fReturnType = G__cint5_tuple_to_type(type, tagnum, typenum, reftype, isconst);
   //}
   builder.fIsconst = isconst & G__CONSTFUNC;
   //{
   //   bool return_is_pointer = (G__var_type == 'U');
   //   builder.fReturnType = G__modify_type(builder.fReturnType, return_is_pointer, reftype, isconst & ~G__CONSTVAR, 0, 0);
   //}
   if (ansi & 0x08) {
      builder.fProp.entry.ansi = 2; // ansi-style prototype and varadic
   }
   else if (ansi & 0x01) {
      builder.fProp.entry.ansi = 1; // ansi-style prototype, not varadic
   }
   builder.fAccess = accessin;
   builder.fIsexplicit = (ansi & 0x04) >> 2;
   builder.fStaticalloc = (ansi & 0x02) >> 1;
   builder.fIsvirtual = isvirtual & 0x01;
   builder.fIspurevirtual = (isvirtual & 0x02) >> 1;
   builder.fProp.entry.friendtag = 0;
   builder.fProp.comment.p.com = (char*) comment;
   if (comment) {
      builder.fProp.comment.filenum = -2;
   }
   else {
      builder.fProp.comment.filenum = -1;
   }
   builder.ParseParameterLink(paras); // parse parameter setup information
   //
   //  Now that we have all the information,
   //  create dictionary entry for function.
   //
   ::Reflex::Member newfunc = builder.Build(funcname); // Create new function entry in dictionary.
   //
   //
   //
   builder.fProp.entry.ptradjust = 0; // FIXME: Why now after we created it?
   if (!funcp && builder.fIsvirtual && (builder.fAccess == G__PUBLIC)) {
      //
      //  No pointer and the function is
      //  public virtual, so the stub must
      //  be in a base class.
      //
      G__inheritance* cbases = G__struct.baseclass[G__get_tagnum(G__p_ifunc)];
      if (cbases) {
         void* base_memberfunc_addr = 0;
         for (size_t idx = 0; (idx < cbases->vec.size()) && !base_memberfunc_addr; ++idx) {
            int basetagnum = cbases->vec[idx].basetagnum;
            {
               Reflex::Scope store_ifunc = G__p_ifunc;
               G__incsetup_memfunc(basetagnum); // Force memfunc_setup for the base.
               G__p_ifunc = store_ifunc;
            }
            ::Reflex::Scope base = G__Dict::GetDict().GetScope(basetagnum);
            // Look for the method in the base class.
            // FIXME: Should we be looking in ALL bases classes instead of just one level?
            ::Reflex::Member base_memberfunc = G__ifunc_exist(newfunc, base, false);
            if (base_memberfunc) {
               G__RflxFuncProperties* base_memberfunc_prop = G__get_funcproperties(base_memberfunc);
               G__RflxFuncProperties* newfunc_prop = G__get_funcproperties(newfunc);
               G__value ptr = G__null;
               G__value_typenum(ptr) = Reflex::PointerBuilder(Reflex::Type(newfunc.DeclaringScope()));
               ptr.obj.i = 0;
               //{
               //   char buf[G__ONELINE];
               //   fprintf(stderr, "G__memfunc_setup: newfunc.Name(): '%s'  %s:%d\n", newfunc.Name(::Reflex::SCOPED).c_str(), __FILE__, __LINE__);
               //   fprintf(stderr, "G__memfunc_setup: Calling G__castvalue('%s', '%s')  %s:%d\n", base_memberfunc.DeclaringScope().Name(Reflex::SCOPED).c_str(), G__valuemonitor(ptr, buf), __FILE__, __LINE__);
               //}
               ptr = G__castvalue_bc((char*) base_memberfunc.DeclaringScope().Name(Reflex::SCOPED).c_str(), ptr, 0);
               //{
               //   fprintf(stderr, "G__memfunc_setup: base_memberfunc_prop->entry.ptradjust: %016lx  %s:%d\n", (unsigned long) base_memberfunc_prop->entry.ptradjust, __FILE__, __LINE__);
               //   fprintf(stderr, "G__memfunc_setup: ptr.obj.i: %016lx  %s:%d\n", (unsigned long) ptr.obj.i, __FILE__, __LINE__);
               //}
               newfunc_prop->entry.ptradjust = base_memberfunc_prop->entry.ptradjust + ptr.obj.i;
               base_memberfunc_addr = base_memberfunc_prop->entry.p;
               newfunc_prop->entry.p = (void*) base_memberfunc_addr;
               if (truep2f) {
                  newfunc_prop->entry.tp2f = truep2f;
               }
               else {
                  newfunc_prop->entry.tp2f = (void*) base_memberfunc_addr;
               }
            }
         }
      }
   }
   //
   //  Create some fake member functions.
   //
   {
      const char* isTemplate = strchr(funcname, '<');
      if (
         isTemplate &&
         (
            !strncmp(funcname, "operator", 8) || // operator<(), or
            ((tagnum != -1) && !strcmp(funcname, G__struct.name[tagnum])) // template constructor
         )
      ) {
         isTemplate = 0;
      }
      if (isTemplate) {
         //
         //  Allocate a buffer for
         //  the new function name.
         //
         G__StrBuf funcname_notmplt_sb(strlen(funcname));
         char* funcname_notmplt = funcname_notmplt_sb;
         strcpy(funcname_notmplt, funcname);
         //
         //  Remove the template arguments
         //  from the original name, creating
         //  the new function name.
         //
         *(funcname_notmplt + (isTemplate - funcname)) = 0;
         //
         //  Recalculate the hash code.
         //
         const char* p = funcname_notmplt;
         int notmplt_hash = 0;
         while (*p) {
            notmplt_hash += *p;
            ++p;
         }
         //
         //  Check to see if new name
         //  already exists in parent.
         //
         bool found = false;
         for (
            ::Reflex::Member_Iterator mem_iter = G__p_ifunc.FunctionMember_Begin();
            mem_iter != G__p_ifunc.FunctionMember_End();
            ++mem_iter
         ) {
            if ((funcname_notmplt[0] == '~') && (mem_iter->Name().c_str()[0] == '~')) {
               found = true;
               break;
            }
            if (strcmp(funcname_notmplt, mem_iter->Name().c_str())) { // No name match.
               continue;
            }
            if ( // ansi function, no match on number of parameters.
               G__get_funcproperties(*mem_iter)->entry.ansi &&
               G__get_funcproperties(newfunc)->entry.ansi &&
               mem_iter->FunctionParameterSize() != newfunc.FunctionParameterSize()
            ) { // ansi function, no match on number of parameters.
               continue;
            }
            if (mem_iter->IsConst() != newfunc.IsConst()) { // No constness match.
               continue;
            }
            // Note: We do not check return type on purpose.
            //if (mem_iter->TypeOf().ReturnType() != newfunc.TypeOf().ReturnType()) { // No return type match.
            //   continue;
            //}
            if (!G__get_funcproperties(*mem_iter)->entry.ansi || !G__get_funcproperties(newfunc)->entry.ansi) { // Match now for non-ansi.
               found = true;
               break;
            }
            found = true;
            int paran = newfunc.FunctionParameterSize(); // Note: We already verified that parameter count matches.
            for (int i = 0; i < paran; ++i) {
               if (mem_iter->TypeOf().FunctionParameterAt(i) != newfunc.TypeOf().FunctionParameterAt(i)) { // No parameter type match.
                  found = false;
                  break;
               }
            }
            if (found) { //  We have a match, all done.
               break;
            }
         }
         //
         //  Create new function identical
         //  to the current function, with
         //  a non-template name by calling
         //  ourselves recursively.
         //
         if (!found) {
            G__memfunc_setup(funcname_notmplt, notmplt_hash, funcp, type, tagnum, typenum, reftype, para_nu, ansi, accessin, isconst, paras, comment, truep2f, isvirtual);
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__separate_parameter(const char* original, int* pos, char* param)
{
   int single_quote = 0;
   int double_quote = 0;
   int single_arg_quote = 0;
   bool argStartsWithSingleQuote = false;

   int startPos = (*pos);
   if (original[startPos] == '\'') {
      // don't put beginning ' into param
      ++startPos;
      argStartsWithSingleQuote = true;
      single_arg_quote = 1;
   }

   int i = startPos;
   bool done = false;
   for (; !done; ++i) {
      int c = original[i];
      switch (c) {
         case '\'':
            if (!double_quote) {
               if (single_quote) {
                  single_quote = 0;
            // only turn on single_quote if at the beginning!
               } else if (i == startPos)  {
                  single_quote = 1;
               } else if (single_arg_quote) {
                  single_arg_quote = 0;
               }
            }
            break;
         case '"':
            if (!single_quote) double_quote ^= 1;
            break;
         case ' ':
            if (!single_quote && !double_quote && !single_arg_quote) {
               c = 0;
               done = true;
            }
            break;
         case '\\':
            if (single_quote || double_quote) {
               // prevent special treatment of next char
               *(param++) = c;
               c = original[++i];
            }
            break;
         case 0:
            done = true;
            break;
      }

      *(param++) = c;
   }

   if (argStartsWithSingleQuote && ! *(param - 1) && *(param - 2) == '\'')
      *(param - 2) = 0; // skip trailing '
   *pos = i;

   if (i > startPos) --i;
   else i = startPos;

   return original[i];
}

//______________________________________________________________________________
extern "C" int G__memfunc_next()
{
   // -- FIXME: Describe this function!
   return 0;
}

//______________________________________________________________________________
extern "C" int G__tag_memfunc_reset()
{
   /* Variables stack restoring */
   std::stack<G__IncSetupStack> *var_stack = G__stack_instance();
   G__IncSetupStack *incsetup_stack = &var_stack->top();

   G__tagnum = incsetup_stack->G__incset_tagnum;
   G__p_ifunc = incsetup_stack->G__incset_p_ifunc;
   G__func_now = incsetup_stack->G__incset_func_now;
   G__var_type = incsetup_stack->G__incset_var_type;
   G__tagdefining = incsetup_stack->G__incset_tagdefining;
   G__def_tagnum = incsetup_stack->G__incset_def_tagnum;

   var_stack->pop();

   return(0);
}

//______________________________________________________________________________
int Cint::Internal::G__cppif_p2memfunc(FILE* fp)
{
   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Get size of pointer to member function\n");
   fprintf(fp, "*********************************************************/\n");
   fprintf(fp, "class G__Sizep2memfunc%s {\n", G__DLLID);
   fprintf(fp, " public:\n");
   fprintf(fp, "  G__Sizep2memfunc%s(): p(&G__Sizep2memfunc%s::sizep2memfunc) {}\n"
           , G__DLLID, G__DLLID);
   fprintf(fp, "    size_t sizep2memfunc() { return(sizeof(p)); }\n");
   fprintf(fp, "  private:\n");
   fprintf(fp, "    size_t (G__Sizep2memfunc%s::*p)();\n", G__DLLID);
   fprintf(fp, "};\n\n");

   fprintf(fp, "size_t G__get_sizep2memfunc%s()\n", G__DLLID);
   fprintf(fp, "{\n");
   fprintf(fp, "  G__Sizep2memfunc%s a;\n", G__DLLID);
   fprintf(fp, "  G__setsizep2memfunc((int)a.sizep2memfunc());\n");
   fprintf(fp, "  return((size_t)a.sizep2memfunc());\n");
   fprintf(fp, "}\n\n");
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__set_sizep2memfunc(FILE* fp)
{
   fprintf(fp, "\n   if(0==G__getsizep2memfunc()) G__get_sizep2memfunc%s();\n", G__DLLID);
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__getcommentstring(char* buf, int tagnum, G__comment_info* pcomment)
{
   G__StrBuf temp_sb(G__LONGLINE);
   char *temp = temp_sb;
   G__getcomment(temp, pcomment, tagnum);
   if ('\0' == temp[0]) {
      sprintf(buf, "(char*)NULL");
   }
   else {
      G__add_quotation(temp, buf);
   }
   return(1);
}

//______________________________________________________________________________
static void G__pragmalinkenum(int tagnum, int globalcomp)
{
   // double check tagnum points to a enum
   if (-1 == tagnum || 'e' != G__struct.type[tagnum]) return;

   /* enum in global scope */
   if (-1 == G__struct.parent_tagnum[tagnum]
         || G__nestedclass
      ) {
      ::Reflex::Scope scope = ::Reflex::Scope::GlobalScope();
      for (::Reflex::Member_Iterator memvar = scope.DataMember_Begin();
            memvar != scope.DataMember_End();
            ++memvar) {
         if (G__get_tagnum(memvar->TypeOf().RawType()) == tagnum) {
            G__get_properties(*memvar)->globalcomp = globalcomp;
         }
      }
   }
   else {
      /* enum enclosed in class  */
      /* do nothing, should already be OK. */
   }
}

#if !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
//______________________________________________________________________________
static void G__linknestedtypedef(int tagnum, int globalcomp)
{
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
   for (::Reflex::Type_Iterator it = scope.SubType_Begin(); it != scope.SubType_End(); ++it) {
      if (it->IsTypedef()) {
         G__RflxProperties *prop = G__get_properties(*it);
         if (prop) {
            prop->globalcomp = globalcomp;
         }
      }
   }
}
#endif // !G__OLDIMPLEMENTATION1955 && G__ROOT

//______________________________________________________________________________
static void G__link_unamed_nested_types(int tagnum, int globalcomp)
{
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
   for (::Reflex::Type_Iterator it = scope.SubType_Begin(); it != scope.SubType_End(); ++it) {
      std::string s(it->Name());
      if (s.length() && s[s.length()-1] == '$') {
         G__RflxProperties *prop = G__get_properties(*it);
         if (prop) {
            prop->globalcomp = globalcomp;
         }
      }
   }
}

//______________________________________________________________________________
void G__link_enclosing(const std::string& buf, int globalcomp)
{
   const char *p = G__find_last_scope_operator((char*) buf.c_str());
   if (p) {
      if (p != buf.c_str()) {
         std::string sub(buf.substr(0, (p - buf.c_str())));
         G__link_enclosing(sub, globalcomp);
         //int tagnum = G__defined_tagname(sub.c_str(),2);
         //if (tagnum!=-1) G__struct.globalcomp[tagnum] = globalcomp;
         ::Reflex::Type typedf = G__find_typedef(sub.c_str());
         if (typedf) {
            G__RflxProperties *prop = G__get_properties(typedf);
            if (prop) prop->globalcomp = globalcomp;
         }
      }
   }
}

//______________________________________________________________________________
//
//  #pragma link C++ class ClassName;      can use regexp
//  #pragma link C   class ClassName;      can use regexp
//  #pragma link off class ClassName;      can use regexp
//  #ifdef G__ROOTSPECIAL
//  #pragma link off class ClassName-;     set ROOT specific flag
//  #pragma link off options=OPTION,OPTION class ClassName; set ROOT specific flag
//  #endif
//  #pragma link C++ enum ClassName;      can use regexp
//  #pragma link C   enum ClassName;      can use regexp
//  #pragma link off enum ClassName;      can use regexp
//
//  #pragma link C++ nestedclass;
//  #pragma link C   nestedclass;
//  #pragma link off nestedclass;
//
//  #pragma link C++ nestedtypedef;
//  #pragma link C   nestedtypedef;
//  #pragma link off nestedtypedef;
//
//  #pragma link C++ function funcname;    can use regexp
//  #pragma link C   function funcname;    can use regexp
//  #pragma link off function funcname;    can use regexp
//  #pragma link MACRO function funcname;  can use regexp
//  #pragma stub C++ function funcname;    can use regexp
//  #pragma stub C   function funcname;    can use regexp
//
//  #pragma link C++ global variablename;  can use regexp
//  #pragma link C   global variablename;  can use regexp
//  #pragma link off global variablename;  can use regexp
//
//  #pragma link C++ defined_in filename;
//  #pragma link C   defined_in filename;
//  #pragma link off defined_in filename;
//  #pragma link C++ defined_in classname;
//  #pragma link C   defined_in classname;
//  #pragma link off defined_in classname;
//  #pragma link C++ defined_in [class|struct|namespace] classname;
//  #pragma link C   defined_in [class|struct|namespace] classname;
//  #pragma link off defined_in [class|struct|namespace] classname;
//
//  #pragma link C++ typedef TypeName;      can use regexp
//  #pragma link C   typedef TypeName;      can use regexp
//  #pragma link off typedef TypeName;      can use regexp
//
//  #pragma link off all classes;
//  #pragma link off all functions;
//  #pragma link off all variables;
//  #pragma link off all typedefs;
//  #pragma link off all methods;
//
//  #pragma link [C++|off] all_method     ClassName;
//  #pragma link [C++|off] all_datamember ClassName;
//               ^
//
//  #pragma link postprocess file func;
//
//  For ROOT only:
//  #praga link C++ ioctortype ClassName;
//

//______________________________________________________________________________
void Cint::Internal::G__specify_link(int link_stub)
{
   int c;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   int globalcomp = G__NOLINK;
   /* int store_globalcomp; */
   int i;
   int hash;
   ::Reflex::Scope ifunc;
   ::Reflex::Scope var;
#ifdef G__REGEXP
   regex_t re;
   int regstat;
#endif
#ifdef G__REGEXP1
   char *re;
#endif
   int os;
   char *p;
   int done = 0;


   /* Get link language interface */
   c = G__fgetname_template(buf, ";\n\r");

   if (strncmp(buf, "postproc", 5) == 0) {
      int store_globalcomp2 = G__globalcomp;
      int store_globalcomp3 = G__store_globalcomp;
      int store_prerun = G__prerun;
      G__globalcomp = G__NOLINK;
      G__store_globalcomp = G__NOLINK;
      G__prerun = 0;
      c = G__fgetname_template(buf, ";");
      if (G__LOADFILE_SUCCESS <= G__loadfile(buf)) {
         G__StrBuf buf2_sb(G__ONELINE);
         char *buf2 = buf2_sb;
         c = G__fgetstream(buf2, ";");
         G__calc(buf2);
         G__unloadfile(buf);
      }
      G__globalcomp = store_globalcomp2;
      G__store_globalcomp = store_globalcomp3;
      G__prerun = store_prerun;
      if (';' != c) G__fignorestream(";");
      return;
   }

   if (strncmp(buf, "default", 3) == 0) {
      c = G__read_setmode(&G__default_link);
      if ('\n' != c && '\r' != c) G__fignoreline();
      return;
   }

   if (G__SPECIFYLINK == link_stub) {
      if (strcmp(buf, "C++") == 0) {
         globalcomp = G__CPPLINK;
         if (G__CLINK == G__globalcomp) {
            G__fprinterr(G__serr, "Warning: '#pragma link C++' ignored. Use '#pragma link C'");
            G__printlinenum();
         }
      }
      else if (strcmp(buf, "C") == 0) {
         globalcomp = G__CLINK;
         if (G__CPPLINK == G__globalcomp) {
            G__fprinterr(G__serr, "Warning: '#pragma link C' ignored. Use '#pragma link C++'");
            G__printlinenum();
         }
      }
      else if (strcmp(buf, "MACRO") == 0) globalcomp = G__MACROLINK;
      else if (strcmp(buf, "off") == 0) globalcomp = G__NOLINK;
      else if (strcmp(buf, "OFF") == 0) globalcomp = G__NOLINK;
      else {
         G__genericerror("Error: '#pragma link' syntax error");
         globalcomp = G__NOLINK; /* off */
      }
   }
   else {
      if (strcmp(buf, "C++") == 0) {
         globalcomp = G__CPPSTUB;
         if (G__CLINK == G__globalcomp) {
            G__fprinterr(G__serr, "Warning: '#pragma stub C++' ignored. Use '#pragma stub C'");
            G__printlinenum();
         }
      }
      else if (strcmp(buf, "C") == 0) {
         globalcomp = G__CSTUB;
         if (G__CPPLINK == G__globalcomp) {
            G__fprinterr(G__serr, "Warning: '#pragma stub C' ignored. Use '#pragma stub C++'");
            G__printlinenum();
         }
      }
      else if (strcmp(buf, "MACRO") == 0) globalcomp = G__MACROLINK;
      else if (strcmp(buf, "off") == 0) globalcomp = G__NOLINK;
      else if (strcmp(buf, "OFF") == 0) globalcomp = G__NOLINK;
      else {
         G__genericerror("Error: '#pragma link' syntax error");
         globalcomp = G__NOLINK; /* off */
      }
   }

   if (';' == c)  return;

   /* Get type of language construct */
   c = G__fgetname_template(buf, ";\n\r");

   if (G__MACROLINK == globalcomp && strncmp(buf, "function", 3) != 0) {
      G__fprinterr(G__serr, "Warning: #pragma link MACRO only valid for global function. Ignored\n");
      G__printlinenum();
      c = G__fignorestream(";\n");
      return;
   }

   int rfNoStreamer = 0;
   int rfNoInputOper = 0;
   int rfUseBytecount = 0;
   int rfNoMap = 0;
   int rfVersionNumber = -1;

   /*************************************************************************
   * #pragma link [spec] options=...
   * possible options:
   *   nostreamer: set G__NOSTREAMER flag
   *   noinputoper: set G__NOINPUTOPERATOR flag
   *   evolution: set G__USEBYTECOUNT flag
   *   nomap: (irgnored by CINT; prevents entry in ROOT's rootmap file)
   *   version(x): sets the version number of the class to x
   *************************************************************************/
   if (!strncmp(buf, "options=", 8) || !strncmp(buf, "option=", 7)) {
      const char* optionStart = buf + 7;
      if (*optionStart == '=') ++optionStart;

      std::list<std::string> options;
      while (optionStart) {
         const char* next = strchr(optionStart, ',');
         options.push_back(std::string(optionStart));
         if (next) options.back().erase(next - optionStart);
         optionStart = next;
         if (optionStart) ++optionStart; // skip ','
      }

      for (std::list<std::string>::iterator iOpt = options.begin();
            iOpt != options.end(); ++iOpt)
         if (*iOpt == "nomap") rfNoMap = 1; // ignored
         else if (*iOpt == "nostreamer") rfNoStreamer = 1;
         else if (*iOpt == "noinputoper") rfNoInputOper = 1;
         else if (*iOpt == "evolution") rfUseBytecount = 1;
         else if (iOpt->size() >= 7 && !strncmp( iOpt->c_str(), "version", 7 )) {
            std::string::size_type fb = iOpt->find( '(' );
            std::string::size_type lb = iOpt->find( ')' );
            if( fb == std::string::npos || lb == std::string::npos ||
                fb+1 >= lb ) {
               G__fprinterr(G__serr, "Malformed version option \"%s\"\n", iOpt->c_str() );
               G__fprinterr(G__serr, "Should be specified as follows: version(x)\n" );
            }
            else {
               std::string verStr = iOpt->substr( fb+1, lb-fb-1 );
               bool noDigit       = false;
               for( std::string::size_type di = 0; di<verStr.size(); ++di )
                  if( !isdigit( verStr[di] ) ) noDigit = true;
 
               if( noDigit )
                  G__fprinterr(G__serr, "Malformed version option! \"%s\" is not a non-negative number!\n", verStr.c_str() );
               else
                  rfVersionNumber = atoi( verStr.c_str() );
            }
         }
         else {
            G__printlinenum();
            G__fprinterr(G__serr, "Warning: ignoring unknown #pragma link option=%s\n", iOpt->c_str());
         }

      // fetch next token
      c = G__fgetname_template(buf, ";\n\r");
   }

   /*************************************************************************
   * #pragma link [spec] nestedclass;
   *************************************************************************/
   if (strncmp(buf, "nestedclass", 3) == 0) {
      G__nestedclass = globalcomp;
   }

   if (strncmp(buf, "nestedtypedef", 7) == 0) {
      G__nestedtypedef = globalcomp;
   }

   if (';' == c)  return;

   switch (globalcomp) {
      case G__CPPSTUB:
      case G__CSTUB:
         if (strncmp(buf, "function", 3) != 0) {
            G__fprinterr(G__serr, "Warning: #pragma stub only valid for global function. Ignored\n");
            c = G__fignorestream(";\n");
            return;
         }
         break;
      default:
         break;
   }

   /*************************************************************************
   * #pragma link [spec] class [name];
   *************************************************************************/
   if (strncmp(buf, "class", 3) == 0 || strncmp(buf, "struct", 3) == 0 ||
         strncmp(buf, "union", 3) == 0 || strncmp(buf, "enum", 3) == 0
#ifndef G__OLDIKMPLEMENTATION1242
         || strncmp(buf, "namespace", 3) == 0
#endif
      ) {
      int len;
      char* p2;
      int iirf;
#ifndef G__OLDIKMPLEMENTATION1334
      char protectedaccess = 0;
#ifndef G__OLDIKMPLEMENTATION1483
      if (strncmp(buf, "class+protected", 10) == 0)
         protectedaccess = G__PROTECTEDACCESS;
      else if (strncmp(buf, "class+private", 10) == 0) {
         protectedaccess = G__PRIVATEACCESS;
         G__privateaccess = 1;
      }
#else
      if (strncmp(buf, "class+protected", 6) == 0) protectedaccess = 1;
#endif
#endif
      c = G__fgetstream_template(buf, ";\n\r");
      for (iirf = 0; iirf < 3; iirf++) {
         if (buf[strlen(buf)-1] == '-') {
            rfNoStreamer = 1;
            buf[strlen(buf)-1] = '\0';
         }
         if (buf[strlen(buf)-1] == '!') {
            rfNoInputOper = 1;
            buf[strlen(buf)-1] = '\0';
         }
         if (buf[strlen(buf)-1] == '+') {
            rfUseBytecount = 1;
            buf[strlen(buf)-1] = '\0';
         }
      }
      len = strlen(buf);
      p2 = strchr(buf, '[');
      p = strrchr(buf, '*');
      if (len && p && (p2 || '*' == buf[len-1] || ('>' != buf[len-1] && '-' != buf[len-1]))) {
         if (*(p + 1) == '>') p = (char*)NULL;
         else p = p;
      }
      else p = (char*)NULL;
      if (p) {
#if defined(G__REGEXP)
#ifndef G__OLDIKMPLEMENTATION1583
         if ('.' != buf[len-2]) {
            buf[len-1] = '.';
            buf[len++] = '*';
            buf[len] = 0;
         }
#endif
         regstat = regcomp(&re, buf, REG_EXTENDED | REG_NOSUB);
         if (regstat != 0) {
            G__genericerror("Error: regular expression error");
            return;
         }
         for (i = 0;i < G__struct.alltag;i++) {
            if ('$' == G__struct.name[i][0]) os = 1;
            else                          os = 0;
            if (0 == regexec(&re, G__struct.name[i] + os, (size_t)0, (regmatch_t*)NULL, 0)) {
               G__struct.globalcomp[i] = globalcomp;
#ifndef G__OLDIKMPLEMENTATION1334
               G__struct.protectedaccess[i] = protectedaccess;
#endif
               ++done;
               if ('e' == G__struct.type[i]) G__pragmalinkenum(i, globalcomp);
               else {
#if  !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
                  if (G__NOLINK > G__nestedtypedef)
                     G__linknestedtypedef(i, globalcomp);
#endif
                  G__link_unamed_nested_types(i, globalcomp);
               }
            }
         }
         regfree(&re);
#elif defined(G__REGEXP1)
         re = regcmp(buf, NULL);
         if (re == 0) {
            G__genericerror("Error: regular expression error");
            return;
         }
         for (i = 0;i < G__struct.alltag;i++) {
            if ('$' == G__struct.name[i][0]) os = 1;
            else                          os = 0;
            if (0 != regex(re, G__struct.name[i] + os)) {
               G__struct.globalcomp[i] = globalcomp;
#ifndef G__OLDIKMPLEMENTATION1334
               G__struct.protectedaccess[i] = protectedaccess;
#endif
               ++done;
               if ('e' == G__struct.type[i]) G__pragmalinkenum(i, globalcomp);
               else {
#if  !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
                  if (G__NOLINK > G__nestedtypedef)
                     G__linknestedtypedef(i, globalcomp);
#endif
                  G__link_unamed_nested_types(i, globalcomp);
               }
            }
         }
         free(re);
#else /* G__REGEXP */
         *p = '\0';
         hash = strlen(buf);
         for (i = 0;i < G__struct.alltag;i++) {
            if ('$' == G__struct.name[i][0]) os = 1;
            else                             os = 0;
            const char *fullname = G__fulltagname(i,1);
            if (strncmp(buf, G__struct.name[i] + os, hash) == 0
                || ('*' == buf[0] && strstr(G__struct.name[i], buf + 1))
                || strncmp(buf, fullname, hash) == 0
               ) {
               G__struct.globalcomp[i] = globalcomp;
#ifndef G__OLDIKMPLEMENTATION1334
               G__struct.protectedaccess[i] = protectedaccess;
#endif
               ++done;
               if ('e' == G__struct.type[i]) G__pragmalinkenum(i, globalcomp);
               else {
#if  !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
                  if (G__NOLINK > G__nestedtypedef)
                     G__linknestedtypedef(i, globalcomp);
#endif
                  G__link_unamed_nested_types(i, globalcomp);
               }
            }
         }
#endif /* G__REGEXP */
      } /* if(p) */
      else {
         i = G__defined_tagname(buf, 1);
         if (i >= 0) {
            G__struct.globalcomp[i] = globalcomp;
#ifndef G__OLDIKMPLEMENTATION1334
            G__struct.protectedaccess[i] = protectedaccess;
#endif
            ++done;
            if ('e' == G__struct.type[i]) G__pragmalinkenum(i, globalcomp);
            else {
#if  !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
               if (G__NOLINK > G__nestedtypedef)
                  G__linknestedtypedef(i, globalcomp);
#endif
               G__link_unamed_nested_types(i, globalcomp);
            }
            G__struct.rootflag[i] = 0;
            if (rfNoStreamer == 1) G__struct.rootflag[i] = G__NOSTREAMER;
            if (rfNoInputOper == 1) G__struct.rootflag[i] |= G__NOINPUTOPERATOR;
            if (rfUseBytecount == 1) {
               G__struct.rootflag[i] |= G__USEBYTECOUNT;
               if (rfNoStreamer) {
                  G__struct.rootflag[i] &= ~G__NOSTREAMER;
                  G__fprinterr(G__serr, "option + mutual exclusive with -, + prevails\n");
               }
            }
            if( rfVersionNumber > -1 )
            {
               AllocateRootSpecial( i );
               G__struct.rootflag[i] |= G__HASVERSION;
               G__struct.rootspecial[i]->version = rfVersionNumber;
            }
         }
      }
      if (!done && G__NOLINK != globalcomp) {
#ifdef G__ROOT
         if (G__dispmsg >= G__DISPERR) {
            G__fprinterr(G__serr, "Error: link requested for unknown class %s", buf);
            G__genericerror((char*)NULL);
         }
#else
         if (G__dispmsg >= G__DISPNOTE) {
            G__fprinterr(G__serr, "Note: link requested for unknown class %s", buf);
            G__printlinenum();
         }
#endif
      }
   }

   /*************************************************************************
   * #pragma link [spec] function [name];
   *************************************************************************/
   else if (strncmp(buf, "function", 3) == 0) {
      ::Reflex::Scope x_ifunc = ::Reflex::Scope::GlobalScope();
#ifndef G__OLDIMPLEMENTATION828
      fpos_t pos;
      int store_line = G__ifile.line_number;
      fgetpos(G__ifile.fp, &pos);
      c = G__fgetstream_template(buf, ";\n\r<>");

      if (G__CPPLINK == globalcomp) globalcomp = G__METHODLINK;

      if (('<' == c || '>' == c)
            && (strcmp(buf, "operator") == 0 || strstr(buf, "::operator"))) {
         int len = strlen(buf);
         buf[len++] = c;
         store_line = G__ifile.line_number;
         fgetpos(G__ifile.fp, &pos);
         buf[len] = G__fgetc();
         if (buf[len] == c || '=' == buf[len]) c = G__fgetstream_template(buf + len + 1, ";\n\r");
         else {
            fsetpos(G__ifile.fp, &pos);
            G__ifile.line_number = store_line;
            if (G__dispsource) G__disp_mask = 1;
            c = G__fgetstream_template(buf + len, ";\n\r");
         }
      }
      else {
         fsetpos(G__ifile.fp, &pos);
         G__ifile.line_number = store_line;
         c = G__fgetstream_template(buf, ";\n\r");
      }


#else /* 828 */
      c = G__fgetstream_template(buf, ";\n\r");
#endif /* 828 */


      /* if the function is specified with paramters */
      p = strchr(buf, '(');
      if (p && strstr(buf, "operator()") == 0) {
         if (strncmp(p, ")(", 2) == 0) p += 2;
         else if (strcmp(p, ")") == 0) p = 0;
      }
      if (p) {
         G__StrBuf funcname_sb(G__LONGLINE);
         char *funcname = funcname_sb;
         G__StrBuf param_sb(G__LONGLINE);
         char *param = param_sb;
         if (')' == *(p + 1) && '(' == *(p + 2)) p = strchr(p + 1, '(');
         *p = '\0';
         strcpy(funcname, buf);
         strcpy(param, p + 1);
         p = strrchr(param, ')');
         *p = '\0';
         G__SetGlobalcomp(funcname, param, globalcomp);
         ++done;
         return;
      }

      p = G__strrstr(buf, "::");
      if (p) {
         int ixx = 0;
         if (x_ifunc.IsTopScope() /* CHECKME */) {
            int tagnum;
            *p = 0;
            tagnum = G__defined_tagname(buf, 0);
            if (-1 != tagnum) {
               x_ifunc = G__Dict::GetDict().GetScope(tagnum);
            }
            *p = ':';
         }
         p += 2;
         while (*p) buf[ixx++] = *p++;
         buf[ixx] = 0;
      }

      /* search for wildcard character */
      p = strrchr(buf, '*');

      /* in case of operator*  */
      if (strncmp(buf, "operator", 8) == 0) p = (char*)NULL;

      if (p) {
#if defined(G__REGEXP)
#ifndef G__OLDIKMPLEMENTATION1583
         int len = strlen(buf);
         if ('.' != buf[len-2]) {
            buf[len-1] = '.';
            buf[len++] = '*';
            buf[len] = 0;
         }
#endif
         regstat = regcomp(&re, buf, REG_EXTENDED | REG_NOSUB);
         if (regstat != 0) {
            G__genericerror("Error: regular expression error");
            return;
         }
         ifunc = x_ifunc;
         for (::Reflex::Member_Iterator memfunc = ifunc.FunctionMember_Begin();
               memfunc != ifunc.FunctionMember_End();
               ++memfunc) {
            if (0 == regexec(&re, memfunc->Name().c_str(), (size_t)0, (regmatch_t*)NULL, 0)
                     && (memfunc->FunctionParameterSize() < 2 ||
                         strncmp(memfunc->TypeOf().FunctionParameterAt(1).RawType().Name().c_str()
                              , "G__CINT_", 8) != 0)
                  ) {
                  G__get_funcproperties(*memfunc)->globalcomp = globalcomp;
                  ++done;
               }
         }
         regfree(&re);
#elif defined(G__REGEXP1)
         re = regcmp(buf, NULL);
         if (re == 0) {
            G__genericerror("Error: regular expression error");
            return;
         }
         ifunc = x_ifunc;
         while (ifunc) {
            for (i = 0;i < ifunc->allifunc;i++) {
               if (0 != regex(re, ifunc->funcname[i])
                     && (ifunc->para_nu[i] < 2 ||
                         -1 == ifunc->para_p_tagtable[i][1] ||
                         strncmp(G__struct.name[ifunc->para_p_tagtable[i][1]]
                                 , "G__CINT_", 8) != 0)
                  ) {
                  ifunc->globalcomp[i] = globalcomp;
                  ++done;
               }
            }
            ifunc = ifunc->next;
         }
         free(re);
#else /* G__REGEXP */
         *p = '\0';
         hash = strlen(buf);
         ifunc = x_ifunc;
         for (::Reflex::Member_Iterator memfunc = ifunc.FunctionMember_Begin();
               memfunc != ifunc.FunctionMember_End();
               ++memfunc) {
            if ((memfunc->Name() == buf
                  || ('*' == buf[0] && strstr(memfunc->Name().c_str(), buf + 1)))
                  && (memfunc->FunctionParameterSize() < 2  ||
                      strncmp(memfunc->TypeOf().FunctionParameterAt(1).RawType().Name().c_str()
                              , "G__CINT_", 8) != 0)
               ) {
               G__get_funcproperties(*memfunc)->globalcomp = globalcomp;
               ++done;
            }
         }
#endif /* G__REGEXP */
      }
      else {
         ifunc = x_ifunc;
         for (::Reflex::Member_Iterator memfunc = ifunc.FunctionMember_Begin();
               memfunc != ifunc.FunctionMember_End();
               ++memfunc) {
            if (strcmp(buf, memfunc->Name().c_str()) == 0
                  && (memfunc->FunctionParameterSize() < 2 ||
                      strncmp(memfunc->TypeOf().FunctionParameterAt(1).RawType().Name().c_str()
                              , "G__CINT_", 8) != 0)
               ) {
               G__get_funcproperties(*memfunc)->globalcomp = globalcomp;
               ++done;
            }
         }
      }
      if (!done && (p = strchr(buf, '<'))) {
         struct G__param fpara;
         struct G__funclist *funclist = (struct G__funclist*)NULL;
         int tmp = 0;

         fpara.paran = 0;

         G__hash(buf, hash, tmp);
         funclist = G__add_templatefunc(buf, &fpara, hash, funclist, x_ifunc, 0);
         if (funclist) {
            G__get_funcproperties(funclist->ifunc)->globalcomp = globalcomp;
            G__funclist_delete(funclist);
            ++done;
         }
      }
      if (!done && G__NOLINK != globalcomp) {
#ifdef G__ROOT
         if (G__dispmsg >= G__DISPERR) {
            G__fprinterr(G__serr, "Error: link requested for unknown function %s", buf);
            G__genericerror((char*)NULL);
         }
#else
         if (G__dispmsg >= G__DISPNOTE) {
            G__fprinterr(G__serr, "Note: link requested for unknown function %s", buf);
            G__printlinenum();
         }
#endif
      }
   }

   /*************************************************************************
   * #pragma link [spec] global [name];
   *************************************************************************/
   else if (strncmp(buf, "global", 3) == 0) {
      c = G__fgetname_template(buf, ";\n\r");
      p = strrchr(buf, '*');
      if (p) {
#if defined(G__REGEXP)
#ifndef G__OLDIKMPLEMENTATION1583
         int len = strlen(buf);
         if ('.' != buf[len-2]) {
            buf[len-1] = '.';
            buf[len++] = '*';
            buf[len] = 0;
         }
#endif
         regstat = regcomp(&re, buf, REG_EXTENDED | REG_NOSUB);
         if (regstat != 0) {
            G__genericerror("Error: regular expression error");
            return;
         }
         var = ::Reflex::Scope::GlobalScope();
         for (::Reflex::Member_Iterator memvar = var.DataMember_Begin();
               memvar != var.DataMember_End();
               ++memvar) {
            if (0 == regexec(&re, memvar->Name().c_str(), (size_t)0, (regmatch_t*)NULL, 0)) {
               G__get_properties(*memvar)->globalcomp = globalcomp;
               ++done;
            }
         }
         regfree(&re);
#elif defined(G__REGEXP1)
         re = regcmp(buf, NULL);
         if (re == 0) {
            G__genericerror("Error: regular expression error");
            return;
         }
         var = &G__global;
         while (var) {
            for (i = 0;i < var->allvar;i++) {
               if (0 != regex(re, var->varnamebuf[i])) {
                  var->globalcomp[i] = globalcomp;
                  ++done;
               }
            }
            var = var->next;
         }
         free(re);
#else /* G__REGEXP */
         *p = '\0';
         hash = strlen(buf);
         var = ::Reflex::Scope::GlobalScope();
         for (::Reflex::Member_Iterator memvar = var.DataMember_Begin();
               memvar != var.DataMember_End();
               ++memvar) {
            if (memvar->Name() == buf
                  || ('*' == buf[0] && strstr(memvar->Name().c_str(), buf + 1))
               ) {
               G__get_properties(*memvar)->globalcomp = globalcomp;
               ++done;
            }
         }
#endif /* G__REGEXP */
      }
      else {
         G__hash(buf, hash, i);
         ::Reflex::Member memvar = G__getvarentry(buf, hash,::Reflex::Scope::GlobalScope(),::Reflex::Scope());
         if (memvar) {
            G__get_properties(memvar)->globalcomp = globalcomp;
            ++done;
         }
      }
      if (!done && G__NOLINK != globalcomp) {
         if (G__dispmsg >= G__DISPNOTE) {
            G__fprinterr(G__serr, "Note: link requested for unknown global variable %s", buf);
            G__printlinenum();
         }
      }
   }

   /*************************************************************************
   * #pragma link [spec] all_datamember [classname];
   *  This is not needed because G__METHODLINK and G__ONLYMETHODLINK are
   *  introduced. Keeping this just for future needs.
   *************************************************************************/
   else if (strncmp(buf, "all_datamembers", 5) == 0) {
      if (';' != c) c = G__fgetstream_template(buf, ";\n\r");
      if (buf[0]) {
         ::Reflex::Scope localvar;
         if (strcmp(buf, "::") == 0) {
            localvar = ::Reflex::Scope::GlobalScope();
         }
         else {
            int tagnum = G__defined_tagname(buf, 0);
            if (-1 != tagnum) {
               localvar = G__Dict::GetDict().GetScope(tagnum);
            }
            else { /* must be an error */
               return;
            }
         }
         for (::Reflex::Member_Iterator memvar = localvar.DataMember_Begin();
               memvar != localvar.DataMember_End();
               ++memvar) {
            G__get_properties(*memvar)->globalcomp = globalcomp;
            //The following require a modification of the privacy, why?
            //For now, explicitly kill the program, if we reach this.
            assert(0 /* code not yet port from cint5 */);
            //if(G__NOLINK==globalcomp) var->access[ig15] = G__PRIVATE;
            //else                      var->access[ig15] = G__PUBLIC;
         }
      }
   }

   /*************************************************************************
   * #pragma link [spec] all_function|all_method [classname];
   *  This is not needed because G__METHODLINK and G__ONLYMETHODLINK are
   *  introduced. Keeping this just for future needs.
   *************************************************************************/
   else if (strncmp(buf, "all_methods", 5) == 0 || strncmp(buf, "all_functions", 5) == 0) {
      if (';' != c) c = G__fgetstream_template(buf, ";\n\r");
      if (G__CPPLINK == globalcomp) globalcomp = G__METHODLINK;
      if (buf[0]) {
         ::Reflex::Scope funcscope;
         if (strcmp(buf, "::") == 0) {
            funcscope = ::Reflex::Scope::GlobalScope();
         }
         else {
            int tagnum = G__defined_tagname(buf, 0);
            if (-1 != tagnum) {
               funcscope = G__Dict::GetDict().GetScope(tagnum);
            }
            else { /* must be an error */
               return;
            }
         }
         for (::Reflex::Member_Iterator memfunc = funcscope.FunctionMember_Begin();
               memfunc != funcscope.FunctionMember_End();
               ++memfunc) {
            G__get_properties(*memfunc)->globalcomp = globalcomp;
            //The following require a modification of the privacy, why?
            //For now, explicitly kill the program, if we reach this.
            assert(0 /* code not yet ported from cint5 */);
            //if(G__NOLINK==globalcomp) memfunc->access[ifn] = G__PRIVATE;
            //else                      memfunc->access[ifn] = G__PUBLIC;
         }
      }
      else {
         G__suppress_methods = (globalcomp == G__NOLINK);
      }
   }

   /*************************************************************************
   * #pragma link [spec] methods;
   *************************************************************************/
   else if (strncmp(buf, "methods", 3) == 0) {
      G__suppress_methods = (globalcomp == G__NOLINK);
   }

   /*************************************************************************
   * #pragma link [spec] typedef [name];
   *************************************************************************/
   else if (strncmp(buf, "typedef", 3) == 0) {
      c = G__fgetname_template(buf, ";\n\r");
      p = strrchr(buf, '*');
      if (p && *(p + 1) == '>') p = (char*)NULL;
      if (p) {
#if defined(G__REGEXP)
#ifndef G__OLDIKMPLEMENTATION1583
         int len = strlen(buf);
         if ('.' != buf[len-2]) {
            buf[len-1] = '.';
            buf[len++] = '*';
            buf[len] = 0;
         }
#endif
         regstat = regcomp(&re, buf, REG_EXTENDED | REG_NOSUB);
         if (regstat != 0) {
            G__genericerror("Error: regular expression error");
            return;
         }
         ::Reflex::Type_Iterator iter;
         for (iter = ::Reflex::Type::Type_Begin();
               iter != ::Reflex::Type::Type_End();
               ++iter) {
            ::Reflex::Type typedf = *iter;
            if (typedf.IsTypedef()) {
               if (0 == regexec(&re, typedf.Name().c_str(), (size_t)0, (regmatch_t*)NULL, 0)) {

                  G__RflxProperties *prop = G__get_properties(typedf);
                  prop->globalcomp = globalcomp;
                  int tagnum_target = G__get_tagnum(typedf);
                  if (-1 != tagnum_target &&
                        '$' == G__struct.name[tagnum_target][0]) {
                     G__struct.globalcomp[tagnum_target] = globalcomp;
                  }
                  ++done;
               }
            }
         }
         regfree(&re);
#elif defined(G__REGEXP1)
         re = regcmp(buf, NULL);
         if (re == 0) {
            G__genericerror("Error: regular expression error");
            return;
         }
         ::Reflex::Type_Iterator iter;
         for (iter = ::Reflex::Type::Type_Begin();
               iter != ::Reflex::Type::Type_End();
               ++iter) {
            ::Reflex::Type typedf = *iter;
            if (0 != regex(re, typdef.Name().c_str())) {
               G__RflxProperties *prop = G__get_properties(typedf);
               prop->globalcomp = globalcomp;
               int tagnum_target = G__get_tagnum(typedf);
               if (-1 != tagnum_target &&
                     '$' == G__struct.name[tagnum_target][0]) {
                  G__struct.globalcomp[tagnum_target] = globalcomp;
               }
               ++done;
            }
         }
         free(re);
#else /* G__REGEXP */
         *p = '\0';
         hash = strlen(buf);
         ::Reflex::Type_Iterator iter;
         for (iter = ::Reflex::Type::Type_Begin();
               iter != ::Reflex::Type::Type_End();
               ++iter) {
            ::Reflex::Type typedf = *iter;
            if (typedf.IsTypedef()) {
               std::string name(typedf.Name(::Reflex::SCOPED));
               if (strncmp(buf, name.c_str(), hash) == 0
                     || ('*' == buf[0] && strstr(name.c_str(), buf + 1))
                  ) {
                  G__RflxProperties *prop = G__get_properties(typedf);
                  prop->globalcomp = globalcomp;
                  int tagnum_target = G__get_tagnum(typedf);
                  if (-1 != tagnum_target &&
                        '$' == G__struct.name[tagnum_target][0]) {
                     G__struct.globalcomp[tagnum_target] = globalcomp;
                  }
                  ++done;
               }
            }
         }
#endif /* G__REGEXP */
      }
      else {
         ::Reflex::Type typedf = G__find_typedef(buf);
         if (typedf) {
            G__RflxProperties *prop = G__get_properties(typedf);
            prop->globalcomp = globalcomp;
            int tagnum_target = G__get_tagnum(typedf);
            if (-1 != tagnum_target &&
                  '$' == G__struct.name[tagnum_target][0]) {
               G__struct.globalcomp[tagnum_target] = globalcomp;
            }
            if (strstr(buf, "<")) {
               // When using reflex, the typedef name will always
               // be the long template name, including the
               // expliciting of the default argument.
               G__link_enclosing(buf, globalcomp);
            }
            ++done;
         }
      }
      if (!done && G__NOLINK != globalcomp) {
#ifdef G__ROOT
         if (G__dispmsg >= G__DISPERR) {
            G__fprinterr(G__serr, "Error: link requested for unknown typedef %s", buf);
            G__genericerror((char*)NULL);
         }
#else
         if (G__dispmsg >= G__DISPNOTE) {
            G__fprinterr(G__serr, "Note: link requested for unknown typedef %s", buf);
            G__printlinenum();
         }
#endif
      }
   }
#ifdef G__ROOT
   /*************************************************************************
   * #pragma link [spec] ioctortype [item];
   *************************************************************************/
   else if (strncmp(buf, "ioctortype", 3) == 0) {
      c = G__fgetname(buf, ";\n\r");
      if (G__p_ioctortype_handler) G__p_ioctortype_handler(buf);
   }
#endif
   /*************************************************************************
   * #pragma link [spec] defined_in [item];
   *************************************************************************/
   else if (strncmp(buf, "defined_in", 3) == 0) {
      fpos_t pos;
      int tagflag = 0;
      int ifile = 0;
      struct stat statBufItem;
      struct stat statBuf;
#ifdef G__WIN32
      char fullItem[_MAX_PATH], fullIndex[_MAX_PATH];
#endif
      fgetpos(G__ifile.fp, &pos);
      c = G__fgetname(buf, ";\n\r");
      if (strcmp(buf, "class") == 0 || strcmp(buf, "struct") == 0 ||
            strcmp(buf, "namespace") == 0) {
         if (isspace(c)) c = G__fgetstream_template(buf, ";\n\r");
         tagflag = 1;
      }
      else {
         fsetpos(G__ifile.fp, &pos);
         c = G__fgetstream_template(buf, ";\n\r<>");
         unsigned int buflen = strlen(buf) - 1;
         if (buf[0]=='"' && buf[buflen]=='"') {
            // Skip the quotes (that allowed us to keep the spaces.
            for(unsigned int bufind = 1; bufind < buflen; ++bufind) {
               buf[bufind-1] = buf[bufind];
            }
            buf[buflen-1]='\0';
         }
      }
      if (
         0 == tagflag &&
         0 == G__statfilename(buf, & statBufItem)) {
#ifdef G__WIN32
         _fullpath(fullItem, buf, _MAX_PATH);
#endif
         for (ifile = 0;ifile < G__nfile;ifile++) {
            if (0 == stat(G__srcfile[ifile].filename, & statBuf)) {
               // --
#ifdef G__WIN32
               _fullpath(fullIndex, G__srcfile[ifile].filename, _MAX_PATH);
               /* Windows is case insensitive! */
#endif // G__WIN32
               if (
#ifndef G__WIN32
                  statBufItem.st_ino == statBuf.st_ino
#else
                  0 == stricmp(fullItem, fullIndex)
#endif
               ) {
                  ++done;
                  /* link class,struct */
                  for (i = 0;i < G__struct.alltag;i++) {
                     if (G__struct.filenum[i] == ifile) {
                        ::Reflex::Scope scope = G__Dict::GetDict().GetScope(i);
                        for (::Reflex::Member_Iterator variter = scope.DataMember_Begin();
                              variter != scope.DataMember_End();
                              ++variter) {
                           if (G__get_properties(*variter)->filenum == ifile
#define G__OLDIMPLEMENTATION1740
                              ) {
                              G__get_properties(*variter)->globalcomp = globalcomp;
                           }
                        }
                        for (::Reflex::Member_Iterator func = scope.FunctionMember_Begin();
                              func != scope.FunctionMember_End();
                              ++func) {
                           if (G__get_funcproperties(*func)->filenum == ifile
                              ) {
                              G__get_funcproperties(*func)->globalcomp = globalcomp;
                           }
                        }
                        G__struct.globalcomp[i] = globalcomp;
                        /* Note this make the equivalent of '+' the
                        default for defined_in type of linking */
                        if (0 == (G__struct.rootflag[i] & G__NOSTREAMER)) {
                           G__struct.rootflag[i] |= G__USEBYTECOUNT;
                        }
                     }
                  }
                  /* link global function */
                  for (::Reflex::Member_Iterator func = ::Reflex::Scope::GlobalScope().FunctionMember_Begin();
                        func != ::Reflex::Scope::GlobalScope().FunctionMember_End();
                        ++func) {
                     if (G__get_funcproperties(*func)->filenum == ifile) {
                        G__get_funcproperties(*func)->globalcomp = globalcomp;
                     }
                  }
#ifdef G__VARIABLEFPOS
                  /* link global variable */
                  for (::Reflex::Member_Iterator variter = ::Reflex::Scope::GlobalScope().DataMember_Begin();
                        variter != ::Reflex::Scope::GlobalScope().DataMember_End();
                        ++variter) {
                     if (G__get_properties(*variter)->filenum == ifile) {
                        G__get_properties(*variter)->globalcomp = globalcomp;
                     }
                  }
#endif
#ifdef G__TYPEDEFFPOS
                  /* link typedef */
                  ::Reflex::Type_Iterator iter;
                  for (iter = ::Reflex::Type::Type_Begin();
                        iter != ::Reflex::Type::Type_End();
                        ++iter) {
                     if (iter->IsTypedef()) {
                        G__RflxProperties *prop = G__get_properties(*iter);
                        if (prop->filenum == ifile) {
                           prop->globalcomp = globalcomp;
                        }
                     }
                  }
#endif
                  break;
               }
            }
         }
      }
      /* #pragma link [C|C++|off] defined_in [class|struct|namespace] name; */
      if (!done) {
         int parent_tagnum = G__defined_tagname(buf, 0);
         int j, flag;
         if (-1 != parent_tagnum) {
            ::Reflex::Scope scope = G__Dict::GetDict().GetScope(parent_tagnum);
            for (::Reflex::Member_Iterator variter = scope.DataMember_Begin();
                  variter != scope.DataMember_End();
                  ++variter) {
               // NOTE: the test is here on the filnum.  It sounds like
               // it should on the tagnum of the parent scope
               if (G__get_properties(*variter)->filenum == ifile) {
                  G__get_properties(*variter)->globalcomp = globalcomp;
               }
            }
            scope = G__Dict::GetDict().GetScope(parent_tagnum);
            for (::Reflex::Member_Iterator func = scope.FunctionMember_Begin();
                  func != scope.FunctionMember_End();
                  ++func) {
               if (G__get_properties(*func)->filenum == ifile) {
                  G__get_properties(*func)->globalcomp = globalcomp;
               }
            }
            for (i = 0;i < G__struct.alltag;i++) {
               done = 1;
               flag = 0;
               j = i;
               G__struct.globalcomp[parent_tagnum] = globalcomp;
               while (-1 != G__struct.parent_tagnum[j]) {
                  if (G__struct.parent_tagnum[j] == parent_tagnum) flag = 1;
                  j = G__struct.parent_tagnum[j];
               }
               if (flag) {
                  ::Reflex::Scope localscope = G__Dict::GetDict().GetScope(i);
                  done = 1;
                  for (::Reflex::Member_Iterator variter = localscope.DataMember_Begin();
                        variter != localscope.DataMember_End();
                        ++variter) {
                     if (G__get_properties(*variter)->filenum == ifile
                           && G__test_access(*variter, G__PUBLIC)
                        ) {
                        G__get_properties(*variter)->globalcomp = globalcomp;
                     }
                  }
                  localscope = G__Dict::GetDict().GetScope(i);
                  for (::Reflex::Member_Iterator func = ::Reflex::Scope::GlobalScope().FunctionMember_Begin();
                        func != ::Reflex::Scope::GlobalScope().FunctionMember_End();
                        ++func) {
                     if (G__get_properties(*func)->filenum == ifile
                           && G__test_access(*func, G__PUBLIC)
                        ) {
                        G__get_properties(*func)->globalcomp = globalcomp;
                     }
                  }
                  G__struct.globalcomp[i] = globalcomp;
                  /* Note this make the equivalent of '+' the
                  default for defined_in type of linking */
                  if ((G__struct.rootflag[i] & G__NOSTREAMER) == 0) {
                     G__struct.rootflag[i] |= G__USEBYTECOUNT;
                  }
               }
            }
#ifdef __GNUC__
#else
#pragma message (FIXME("Is there a better way to get the list of type(def)s defined in scope or any of its sub-scopes"))
#endif
            for (::Reflex::Type_Iterator iTypedef =::Reflex::Type::Type_Begin();
                  iTypedef !=::Reflex::Type::Type_End();++iTypedef) {
               if (!iTypedef->IsTypedef()) continue;
               flag = 0;
               j = G__get_tagnum(*iTypedef);
               do {
                  if (j == parent_tagnum) flag = 1;
                  j = G__struct.parent_tagnum[j];
               }
               while (-1 != j);
               if (flag) {
                  G__RflxProperties* props = G__get_properties(*iTypedef);
                  if (props) {
                     props->globalcomp = globalcomp;
                     //CHECKME the old code contained:
                     //if ( (G__struct.rootflag[i] & G__NOSTREAMER) == 0 ) {
                     //   G__struct.rootflag[i] |= G__USEBYTECOUNT;
                     //}
                     //but maybe we should have something more like:
                     //if((-1!=G__newtype.tagnum[i] &&
                     //   '$'==G__struct.name[G__newtype.tagnum[i]][0])) {
                     //      G__struct.globalcomp[G__newtype.tagnum[i]] = globalcomp;
                     //}
                  }
               }
            }
         }
      }
      if (!done && G__NOLINK != globalcomp) {
         G__fprinterr(G__serr, "Warning: link requested for unknown srcfile %s", buf);
         G__printlinenum();
      }
   }

   /*************************************************************************
   * #pragma link [spec] all [item];
   *************************************************************************/
   else if (strncmp(buf, "all", 2) == 0) {
      c = G__fgetname_template(buf, ";\n\r");
      if (strncmp(buf, "class", 3) == 0) {
         for (i = 0;i < G__struct.alltag;i++) {
            if (G__NOLINK == globalcomp ||
                  (G__NOLINK == G__struct.iscpplink[i] &&
                   (-1 != G__struct.filenum[i] &&
                    0 == (G__srcfile[G__struct.filenum[i]].hdrprop&G__CINTHDR))))
               G__struct.globalcomp[i] = globalcomp;
         }
      }
      else if (strncmp(buf, "function", 3) == 0) {
         for (::Reflex::Member_Iterator func = ::Reflex::Scope::GlobalScope().FunctionMember_Begin();
               func != ::Reflex::Scope::GlobalScope().FunctionMember_End();
               ++func) {
            if (G__NOLINK == globalcomp ||
                  (
                     0 <= G__get_funcproperties(*func)->entry.size && 0 <= G__get_properties(*func)->filenum &&
                     0 == (G__srcfile[G__get_properties(*func)->filenum].hdrprop&G__CINTHDR))
               ) {
               G__get_funcproperties(*func)->globalcomp = globalcomp;
            }
         }
      }
      else if (strncmp(buf, "global", 3) == 0) {
         for (::Reflex::Member_Iterator variter = ::Reflex::Scope::GlobalScope().DataMember_Begin();
               variter != ::Reflex::Scope::GlobalScope().DataMember_End();
               ++variter) {
            if (G__NOLINK == globalcomp ||
                  (0 <= G__get_properties(*variter)->filenum &&
                   0 == (G__srcfile[G__get_properties(*variter)->filenum].hdrprop&G__CINTHDR))
               ) {
               G__get_properties(*variter)->globalcomp = globalcomp;
            }
         }
      }
      else if (strncmp(buf, "typedef", 3) == 0) {
         for (::Reflex::Type_Iterator iTypedef =::Reflex::Type::Type_Begin();
               iTypedef !=::Reflex::Type::Type_End();++iTypedef) {
            if (!iTypedef->IsTypedef()) continue;
            G__RflxProperties* props = G__get_properties(*iTypedef);
            if (props == 0) continue;
            if (G__NOLINK == globalcomp ||
                  (G__NOLINK == props->iscpplink &&
                   0 <= props->filenum &&
                   0 == (G__srcfile[props->filenum].hdrprop&G__CINTHDR))) {
               props->globalcomp = globalcomp;
               int target_tagnum = G__get_tagnum(*iTypedef);
               if ((-1 != target_tagnum &&
                     '$' == G__struct.name[target_tagnum][0])) {
                  G__struct.globalcomp[target_tagnum] = globalcomp;
               }
            }
         }
      }
      /*************************************************************************
      * #pragma link [spec] all methods;
      *************************************************************************/
      else if (strncmp(buf, "methods", 3) == 0) {
         G__suppress_methods = (globalcomp == G__NOLINK);
      }
   }
   if (';' != c) c = G__fignorestream("#;");
   if (';' != c) G__genericerror("Syntax error: #pragma link");
}

//______________________________________________________________________________
void f1(int /*link_stub*/)
{}

//______________________________________________________________________________
void Cint::Internal::G__incsetup_memvar(::Reflex::Scope& scope)
{
   int tagnum = G__get_tagnum(scope);
   if (tagnum > 0) G__incsetup_memvar(tagnum);
}

//______________________________________________________________________________
void Cint::Internal::G__incsetup_memvar(::Reflex::Type& scope)
{
   int tagnum = G__get_tagnum(scope);
   if (tagnum > 0) G__incsetup_memvar(tagnum);
}

//______________________________________________________________________________
void Cint::Internal::G__incsetup_memvar(int tagnum)
{
   int store_asm_exec;
   char store_var_type;
   int store_static_alloc = G__static_alloc;
   int store_constvar = G__constvar;
  if (G__struct.incsetup_memvar[tagnum]==0) return;
  
  if(!G__struct.incsetup_memvar[tagnum]->empty()) {
      store_asm_exec = G__asm_exec;
      G__asm_exec = 0;
      store_var_type = G__var_type;
      G__input_file store_ifile = G__ifile;
      int fileno = G__struct.filenum[tagnum];
      G__ifile.filenum = fileno;
      G__ifile.line_number = -1;
      G__ifile.str = 0;
      G__ifile.pos = 0;
      G__ifile.vindex = 0;

      if (fileno != -1) {
         G__ifile.fp = G__srcfile[fileno].fp;
         strcpy(G__ifile.name, G__srcfile[fileno].filename);
      }
#ifdef G__OLDIMPLEMENTATION1125_YET
      if (0 == G__struct.memvar[tagnum]->allvar
            || 'n' == G__struct.type[tagnum]){

       // G__setup_memvarXXX execution
       std::list<G__incsetup>::iterator iter;
       for (iter=G__struct.incsetup_memvar[tagnum]->begin(); iter != G__struct.incsetup_memvar[tagnum]->end(); iter ++)
          (*iter)();
    }
#else
    // G__setup_memvarXXX execution
      std::list<G__incsetup>::iterator iter;
      for (iter=G__struct.incsetup_memvar[tagnum]->begin(); iter != G__struct.incsetup_memvar[tagnum]->end(); iter ++)
         (*iter)();
#endif
      // The G__setup_memvarXXX functions have been executed. We don't need the pointers anymore. We clean the list
      G__struct.incsetup_memvar[tagnum]->clear();
      delete G__struct.incsetup_memvar[tagnum];
      G__struct.incsetup_memvar[tagnum] = 0;

#ifdef G__DEBUG
      if (G__var_type != store_var_type)
         G__fprinterr(G__serr, "Cint internal error: G__incsetup_memvar %c %c\n"
                      , G__var_type, store_var_type);
#endif
      G__var_type = store_var_type;
      G__asm_exec = store_asm_exec;
      G__constvar = store_constvar;
      G__ifile = store_ifile;
      G__static_alloc = store_static_alloc;
   }
}

//______________________________________________________________________________
void Cint::Internal::G__incsetup_memfunc(::Reflex::Scope& scope)
{
   int tagnum = G__get_tagnum(scope);
   if (tagnum > 0) G__incsetup_memfunc(tagnum);
}

//______________________________________________________________________________
void Cint::Internal::G__incsetup_memfunc(::Reflex::Type& scope)
{
   int tagnum = G__get_tagnum(scope);
   if (tagnum > 0) G__incsetup_memfunc(tagnum);
}

//______________________________________________________________________________
void Cint::Internal::G__incsetup_memfunc(int tagnum)
{
   char store_var_type;
   int store_asm_exec;

   if (G__struct.incsetup_memfunc[tagnum]==0) return;

   if(!G__struct.incsetup_memfunc[tagnum]->empty()) {
      store_asm_exec = G__asm_exec;
      G__asm_exec = 0;
      store_var_type = G__var_type;
      G__input_file store_ifile = G__ifile;
      int fileno = G__struct.filenum[tagnum];
      if (fileno >= G__nfile) {
         // Most likely we are in the unloading of the shared library holding this
         // dictionary.  Let's avoid a spurrious message (Function can not be defined in a command line or a tempfile).
         fileno = -1;
      }
      G__ifile.filenum = fileno;
      G__ifile.line_number = -1;
      G__ifile.str = 0;
      G__ifile.pos = 0;
      G__ifile.vindex = 0;

      if (fileno != -1) {
         G__ifile.fp = G__srcfile[fileno].fp;
         if (G__srcfile[fileno].filename) strcpy(G__ifile.name, G__srcfile[fileno].filename);
      }
#ifdef G__OLDIMPLEMENTATION1125_YET /* G__PHILIPPE26 */
      if (0 == G__struct.memfunc[tagnum]->allifunc
         || 'n' == G__struct.type[tagnum]
      || (
         -1 != G__struct.memfunc[tagnum]->pentry[0]->size
         && 2 >= G__struct.memfunc[tagnum]->allifunc)){

            // G__setup_memfuncXXX execution
            std::list<G__incsetup>::iterator iter;
            for (iter=G__struct.incsetup_memfunc[tagnum]->begin(); iter != G__struct.incsetup_memfunc[tagnum]->end(); iter ++)
               (*iter)();
      }
#else
      // G__setup_memfuncXXX execution
      std::list<G__incsetup>::iterator iter;
      for (iter=G__struct.incsetup_memfunc[tagnum]->begin(); iter != G__struct.incsetup_memfunc[tagnum]->end(); iter ++)
         (*iter)();
#endif
      // The G__setup_memfuncXXX functions have been executed. We don't need the pointers anymore. We clean the list
      G__struct.incsetup_memfunc[tagnum]->clear();
      delete G__struct.incsetup_memfunc[tagnum];
      G__struct.incsetup_memfunc[tagnum] = 0;

      G__var_type = store_var_type;
      G__asm_exec = store_asm_exec;
      G__ifile = store_ifile;
   }
}

//______________________________________________________________________________
extern "C" int G__getnumbaseclass(int tagnum)
{
   return(G__struct.baseclass[tagnum]->vec.size());
}

//______________________________________________________________________________
static ::Reflex::Type G__setnewtype_typenum;
//static int G__setnewtype_nindex = 0;
//static std::vector<int> G__setnewtype_index;

namespace Cint {
namespace Internal {
//______________________________________________________________________________
void G__setnewtype_settypenum(::Reflex::Type typenum)
{
   G__setnewtype_typenum = typenum;
}
} // namespace Internal
} // namespace Cint

//______________________________________________________________________________
extern "C" void G__setnewtype(int globalcomp, const char* comment, int /*nindex*/)
{
   G__RflxProperties* prop = G__get_properties(G__setnewtype_typenum);
   if (prop) {
      prop->iscpplink = globalcomp;
      prop->comment.p.com = (char*) comment;
      if (comment) {
         prop->comment.filenum = -2;
      }
      else {
         prop->comment.filenum = -1;
      }
   }
   //G__setnewtype_nindex = nindex;
   //G__setnewtype_index.clear();
}

//______________________________________________________________________________
extern "C" void G__setnewtypeindex(int /*j*/, int index)
{
   ::Reflex::Type ty = G__setnewtype_typenum;
   std::string name = ty.Name(::Reflex::SCOPED);
   G__RflxProperties properties = *G__get_properties(ty);
   
   ty = ty.ToType();
   ty = ::Reflex::ArrayBuilder(ty, index);
   
   G__setnewtype_typenum.Unload();
   
   ty = ::Reflex::TypedefTypeBuilder(name.c_str(), ty);
   properties.typenum = G__Dict::GetDict().Register(ty);
   *G__get_properties(ty) = properties;
   G__setnewtype_typenum = ty;
}

//______________________________________________________________________________
extern "C" long G__getgvp()
{
   return((long)G__globalvarpointer);
}

//______________________________________________________________________________
extern "C" void G__setgvp(long gvp)
{
   G__globalvarpointer = (char*)gvp;
}

//______________________________________________________________________________
extern "C" void G__resetplocal()
{
   G__IncSetupStack incsetup_stack;
   std::stack<G__IncSetupStack> *var_stack = G__stack_instance(); 
   if (G__def_struct_member && 'n' == G__struct.type[G__get_tagnum(G__tagdefining)]) {
      incsetup_stack.G__incset_tagnum = G__tagnum;
      incsetup_stack.G__incset_p_local = G__p_local;
      incsetup_stack.G__incset_def_struct_member = G__def_struct_member;
      incsetup_stack.G__incset_tagdefining = G__tagdefining;
      incsetup_stack.G__incset_globalvarpointer = G__globalvarpointer ;
      incsetup_stack.G__incset_var_type = G__var_type ;
      incsetup_stack.G__incset_typenum = G__typenum ;
      incsetup_stack.G__incset_static_alloc = G__static_alloc ;
      incsetup_stack.G__incset_access = G__access ;

      G__tagnum = G__tagdefining;
      G__p_local = G__tagnum;
      G__def_struct_member = 1;
      G__tagdefining = G__tagnum;
      /* G__static_alloc = 1; */
   }
   else {
      G__p_local = ::Reflex::Scope();
      incsetup_stack.G__incset_def_struct_member = 0;
   }
   var_stack->push(incsetup_stack);
}

//______________________________________________________________________________
extern "C" void G__resetglobalenv()
{
   /* Variables stack restoring */
   std::stack<G__IncSetupStack> *var_stack = G__stack_instance(); 
   G__IncSetupStack *incsetup_stack = &var_stack->top();

   if(incsetup_stack->G__incset_def_struct_member && incsetup_stack->G__incset_tagdefining.IsNamespace()){
      G__p_local = incsetup_stack->G__incset_p_local;
      G__def_struct_member = incsetup_stack->G__incset_def_struct_member ;
      G__tagdefining = incsetup_stack->G__incset_tagdefining ;

      G__globalvarpointer = incsetup_stack->G__incset_globalvarpointer ;
      G__var_type = incsetup_stack->G__incset_var_type ;
      G__tagnum = incsetup_stack->G__incset_tagnum ;
      G__typenum = incsetup_stack->G__incset_typenum ;
      G__static_alloc = incsetup_stack->G__incset_static_alloc ;
      G__access = incsetup_stack->G__incset_access ;

   }
   else {
      G__globalvarpointer = G__PVOID;
      G__var_type = 'p';
      G__tagnum = ::Reflex::Scope();
      G__typenum = ::Reflex::Scope();
      G__static_alloc = 0;
      G__access = G__PUBLIC;
   }

   var_stack->pop();
}

//______________________________________________________________________________
extern "C" void G__lastifuncposition()
{
   /* Variables stack storing */
   std::stack<G__IncSetupStack> *var_stack = G__stack_instance(); 
   G__IncSetupStack incsetup_stack;

   if(G__def_struct_member && G__tagdefining.IsNamespace()) {
      incsetup_stack.G__incset_def_struct_member = G__def_struct_member;
      incsetup_stack.G__incset_tagnum = G__tagnum;
      incsetup_stack.G__incset_p_ifunc = G__p_ifunc;
      incsetup_stack.G__incset_func_now = G__func_now;
      incsetup_stack.G__incset_var_type = G__var_type;
      incsetup_stack.G__incset_tagdefining = G__tagdefining;
      G__tagnum = G__tagdefining;
      G__p_ifunc = G__tagnum;
   }
   else {
      G__p_ifunc = ::Reflex::Scope::GlobalScope();
      incsetup_stack.G__incset_def_struct_member = 0;
   }

   var_stack->push(incsetup_stack);
}

//______________________________________________________________________________
extern "C" void G__resetifuncposition()
{
   /* Variables stack restoring */
   std::stack<G__IncSetupStack>* var_stack = G__stack_instance(); 
   G__IncSetupStack *incsetup_stack = &var_stack->top();

   if(incsetup_stack->G__incset_def_struct_member && incsetup_stack->G__incset_tagdefining.IsNamespace()){
      G__tagnum = incsetup_stack->G__incset_tagnum;
      G__p_ifunc = incsetup_stack->G__incset_p_ifunc;
      G__func_now = incsetup_stack->G__incset_func_now;
      G__var_type = incsetup_stack->G__incset_var_type;
   }
   else {
      G__tagnum = Reflex::Scope();
      G__p_ifunc = Reflex::Scope::GlobalScope();
      G__func_now = Reflex::Member();
      G__var_type = 'p';
   }
   G__globalvarpointer = G__PVOID;
   G__static_alloc = 0;
   G__access = G__PUBLIC;
   G__typenum = Reflex::Type();

   var_stack->pop();
}

//______________________________________________________________________________
extern "C" void G__setnull(G__value* result)
{
   *result = G__null;
}

//______________________________________________________________________________
extern "C" long G__getstructoffset()
{
   return((long)G__store_struct_offset);
}

//______________________________________________________________________________
extern "C" int G__getaryconstruct()
{
   return(G__cpp_aryconstruct);
}

//______________________________________________________________________________
extern "C" long G__gettempbufpointer()
{
   return(G__p_tempbuf->obj.obj.i);
}

//______________________________________________________________________________
extern "C" void G__setsizep2memfunc(int sizep2memfunc)
{
   G__sizep2memfunc = sizep2memfunc;
}

//______________________________________________________________________________
extern "C" int G__getsizep2memfunc()
{
   G__asm_noverflow = G__store_asm_noverflow;
   G__no_exec_compile = G__store_no_exec_compile;
   G__asm_exec = G__store_asm_exec;
   return(G__sizep2memfunc);
}

//______________________________________________________________________________
void Cint::Internal::G__setInitFunc(char* initfunc)
{
   G__INITFUNC = initfunc;
}

#ifdef G__WILDCARD  // FIXME: What do these functions have to do with wildcards?
//______________________________________________________________________________
extern "C" FILE* G__getIfileFp()
{
   return(G__ifile.fp);
}

//______________________________________________________________________________
extern "C" void G__incIfileLineNumber()
{
   ++G__ifile.line_number;
}

//______________________________________________________________________________
extern "C" int G__getIfileLineNumber()
{
   return(G__ifile.line_number);
}

//______________________________________________________________________________
extern "C" void G__setReturn(int rtn)
{
   G__return = rtn;
}

//______________________________________________________________________________
extern "C" long G__getFuncNow()
{
   return((long)G__func_now.Id());
}

//______________________________________________________________________________
extern "C" int G__getPrerun()
{
   return(G__prerun);
}

//______________________________________________________________________________
extern "C" void G__setPrerun(int prerun)
{
   G__prerun = prerun;
}

//______________________________________________________________________________
extern "C" short G__getDispsource()
{
   return(G__dispsource);
}

//______________________________________________________________________________
extern "C" FILE* G__getSerr()
{
   return(G__serr);
}

//______________________________________________________________________________
extern "C" int G__getIsMain()
{
   return(G__ismain);
}

//______________________________________________________________________________
extern "C" void G__setIsMain(int ismain)
{
   G__ismain = ismain;
}

//______________________________________________________________________________
extern "C" void G__setStep(int step)
{
   G__step = step;
}

//______________________________________________________________________________
extern "C" int G__getStepTrace()
{
   return(G__steptrace);
}

//______________________________________________________________________________
extern "C" void G__setDebug(int dbg)
{
   G__debug = dbg;
}

//______________________________________________________________________________
extern "C" int G__getDebugTrace()
{
   return(G__debugtrace);
}

//______________________________________________________________________________
extern "C" void G__set_asm_noverflow(int novfl)
{
   G__asm_noverflow = novfl;
}

//______________________________________________________________________________
extern "C" int G__get_no_exec()
{
   return(G__no_exec);
}

//______________________________________________________________________________
extern "C" int G__get_no_exec_compile()
{
   return(G__no_exec_compile);
}
#endif // G__WILDCARD  // FIXME: What do the previous functions have to do with wildcards?

//______________________________________________________________________________
template<class T> inline T* G__refT(G__value* buf)
{
   char type = G__get_type(*buf);
   if (type == G__gettypechar<T>() && buf->ref) {
      return (T*) buf->ref;
   }
   G__setvalue(buf, G__convertT<T>(buf));
   return &G__value_ref<T>(*buf);
}



extern "C" {
   char* G__Charref(G__value *buf) {return G__refT<char>(buf);}
   short* G__Shortref(G__value *buf) {return G__refT<short>(buf);}
   int* G__Intref(G__value *buf) {return G__refT<int>(buf);}
   long* G__Longref(G__value *buf) {return G__refT<long>(buf);}
   unsigned char* G__UCharref(G__value *buf) {return G__refT<unsigned char>(buf);}
#ifdef G__BOOL4BYTE
   int* G__Boolref(G__value *buf) {return (int*)G__refT<bool>(buf);}
#else // G__BOOL4BYTE
   unsigned char* G__Boolref(G__value *buf) {return (unsigned char*)G__refT<bool>(buf);}
#endif // G__BOOL4BYTE
   unsigned short* G__UShortref(G__value *buf) {return G__refT<unsigned short>(buf);}
   unsigned int* G__UIntref(G__value *buf) {return G__refT<unsigned int>(buf);}
   unsigned long* G__ULongref(G__value *buf) {return G__refT<unsigned long>(buf);}
   float* G__Floatref(G__value *buf) {return G__refT<float>(buf);}
   double* G__Doubleref(G__value *buf) {return G__refT<double>(buf);}
   G__int64* G__Longlongref(G__value *buf) {return G__refT<G__int64>(buf);}
   G__uint64* G__ULonglongref(G__value *buf) {return G__refT<G__uint64>(buf);}
   long double* G__Longdoubleref(G__value *buf) {return G__refT<long double>(buf);}
} // extern "C"

//______________________________________________________________________________
void Cint::Internal::G__specify_extra_include()
{
   int i;
   int c;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   char *tobecopied;
   if (!G__extra_include) {
      G__extra_include = (char**)malloc(G__MAXFILE * sizeof(char*));
      for (i = 0;i < G__MAXFILE;i++)
         G__extra_include[i] = (char*)malloc(G__MAXFILENAME * sizeof(char));
   };
   c = G__fgetstream_template(buf, ";\n\r<>");
   if (1) { /* should we check if the file exist ? */
      tobecopied = buf;
      if (buf[0] == '\"' || buf[0] == '\'') tobecopied++;
      i = strlen(buf);
      if (buf[i-1] == '\"' || buf[i-1] == '\'') buf[i-1] = '\0';
      strcpy(G__extra_include[G__extra_inc_n++], tobecopied);
   }
}

//______________________________________________________________________________
void Cint::Internal::G__gen_extra_include()
{
   // -- Prepend the extra header files to the C or CXX file.
   char * tempfile;
   FILE *fp, *ofp;
   char line[BUFSIZ];
   int i;

   if (G__extra_inc_n) {
#ifndef G__ADD_EXTRA_INCLUDE_AT_END
      /* because of a bug in (at least) the KAI compiler we have to
         add the files at the beginning of the dictionary header file
         (Specifically, the extra include files have to be include
         before any forward declarations!) */

      if (!G__CPPLINK_H) return;

      tempfile = (char*) malloc(strlen(G__CPPLINK_H) + 6);
      sprintf(tempfile, "%s.temp", G__CPPLINK_H);
      rename(G__CPPLINK_H, tempfile);

      fp = fopen(G__CPPLINK_H, "w");
      if (!fp) G__fileerror(G__CPPLINK_H);
      ofp = fopen(tempfile, "r");
      if (!ofp) G__fileerror(tempfile);

      /* Add the extra include ad the beginning of the files */
      fprintf(fp, "\n/* Includes added by #pragma extra_include */\n");
      for (i = 0; i < G__extra_inc_n; i++) {
         fprintf(fp, "#include \"%s\"\n", G__extra_include[i]);
      }

      /* Copy rest of the header file */
      while (fgets(line, BUFSIZ, ofp)) {
         fprintf(fp, "%s", line);
      }
      fprintf(fp, "\n");

      fclose(fp);
      fclose(ofp);
      unlink(tempfile);
      free(tempfile);

#else
      fp = fopen(G__CPPLINK_H, "a");
      if (!fp) G__fileerror(G__CPPLINK_H);

      fprintf(fp, "\n/* Includes added by #pragma extra_include */\n");
      for (i = 0; i < G__extra_inc_n; i++) {
         fprintf(fp, "#include \"%s\"\n", G__extra_include[i]);
      }
      fprintf(fp, "\n");
      fclose(fp);
#endif

   }
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
