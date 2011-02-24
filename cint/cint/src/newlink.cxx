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

/* #define G__OLDIMPLEMENTATION2047 */
#define G__OLDIMPLEMENTATION2044 /* Problem with t980.cxx */


#include "common.h"
#include "configcint.h"
#include "value.h"
#include "dllrev.h"
#include "Api.h"
#include <cstdlib>
#include <cstring>
#include <stack>
#include <string>
#include <stdexcept>

#ifdef G__NOSTUBS
# include <ext/hash_map>
# include <dlfcn.h> // needed for dlsym
# include <typeinfo>
# include <cxxabi.h>
#endif

#ifndef G__TESTMAIN
#include <sys/stat.h>
#endif

#if defined(__GNUC__) && __GNUC__ > 3 && ((__GNUC_MINOR__ > 3 &&__GNUC_PATCHLEVEL__ > 3) || __GNUC_MINOR__ > 4)
#define G__LOCALS_REFERENCED_FROM_RSP
#endif

extern "C"
void G__enable_wrappers(int set) {
   // enable wrappers
   G__wrappers = set;
}

extern "C"
int G__wrappers_enabled() {
   // whether wrappers are enabled
   return G__wrappers;
}

#ifdef G__NOSTUBS

// 03-08-08
// Due to G__struct.hash inneficiency we are forced to create the same
// but in an efficient way
//extern std::map<std::string, int> G__structmap;
typedef __gnu_cxx::hash_map<const char*, int>::value_type G__structmap_pair;

// 03-08-07
// Create a hash_map to contain the association between
// mangled_names (classes) and tagnums
static
void* G__get_struct_map()
{
  static __gnu_cxx::hash_map<const char*, int> structmap;
  return &structmap;
}
#endif

#ifdef _WIN32
#include "windows.h"
#include <errno.h>
extern "C"
extern const char *G__libname;

FILE *FOpenAndSleep(const char *filename, const char *mode) {
   int tries=0;
   FILE *ret=0;
   while (!ret && ++tries<51)
      if (!(ret=fopen(filename, mode)) && tries<50)
         if (errno!=EACCES && errno!=EEXIST) return 0;
         else Sleep(200);
   if (tries>1)  printf("fopen slept for %g seconds until it succeeeded.\n", (tries-1)/5.);
   return ret;
}

# ifdef fopen
#  undef fopen
# endif
# define fopen(A,B) FOpenAndSleep((A),(B))
#endif

// cross-compiling for iOS and iOS simulator (assumes host is Intel Mac OS X)
#if defined(R__IOSSIM) || defined(R__IOS)
#ifdef __x86_64__
#undef __x86_64__
#endif
#ifdef __i386__
#undef __i386
#endif
#ifdef R__IOSSIM
#define __i386 1
#endif
#ifdef R__IOS
#define __arm__ 1
#endif
#endif

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
  G__struct.rootspecial[tagnum]->defaultconstructorifunc = 0;
}

#ifdef G__ROOT

/******************************************************************
 * G__set_class_autoloading
 ******************************************************************/
int (*G__p_ioctortype_handler) G__P((const char*));

/************************************************************************
* G__set_class_autoloading_callback
************************************************************************/
extern "C" void G__set_ioctortype_handler(int (*p2f) G__P((const char*)))
{
  G__p_ioctortype_handler = p2f;
}
#endif

extern "C" {

#define G__OLDIMPLEMENTATION1702
#define G__OLDIMPLEMENTATION1714

/* This is a very complicated decision. The change 1714 avoids compiled stub
 * function registeration to the dictionary. From the interpreter, only
 * interpreted stub function should be visible. If there is a
 */
#undef G__OLDIMPLEMENTATION1714

#define G__OLDIMPLEMENTATION1336

#define G__PROTECTEDACCESS   1
#define G__PRIVATEACCESS     2
static int G__privateaccess = 0;

/**************************************************************************
* CAUTION:
*  Following macro G__BUILTIN must not be defined at normal cint
* installation. This macro must be deleted only when you generate following
* source files.
*     src/libstrm.cxx  in lib/stream    'make'
*     src/gcc3strm.cxx in lib/gcc3strm  'make'
*     src/iccstrm.cxx  in lib/iccstrm   'make'
*     src/vcstrm.cxx   in lib/vcstream  'make' , 'make -f Makefileold'
*     src/vc7strm.cxx  in lib/vc7stream 'make'
*     src/bcstrm.cxx   in lib/bcstream  'make'
*     src/cbstrm.cpp   in lib/cbstream  'make'
*     src/sunstrm.cxx  in lib/snstream  'make'
*     src/kccstrm.cxx  (lib/kcc_work not included in the package)
*     src/stdstrct.c  in lib/stdstrct  'make'
*     src/Apiif.cxx   in src           'make -f Makeapi' , 'make -f Makeapiold'
* g++ has a bug of distinguishing 'static operator delete(void* p)' in
* different file scope. Deleting this macro will avoid this problem.
*
* Note:  The C++ standard explicitly forbids a static operator delete
*        at global scope.
**************************************************************************/
/* #define G__BUILTIN */

#if !defined(G__DECCXX) && !defined(G__BUILTIN) && !defined(__hpux)
#define G__DEFAULTASSIGNOPR
#endif

#if !defined(G__DECCXX) && !defined(G__BUILTIN)
#define G__N_EXPLICITDESTRUCTOR
#endif

#ifndef G__N_EXPLICITDESTRUCTOR
#ifdef G__P2FCAST
#undef G__P2FCAST
#endif
#ifdef G__P2FDECL
#undef G__P2FDECL
#endif
#endif


/**************************************************************************
* If this is Windows-NT/95, create G__PROJNAME.DEF file
**************************************************************************/
#if  defined(G__WIN32) && !defined(G__BORLAND)
#define G__GENWINDEF
#endif




/**************************************************************************
* Following static variables must be protected by semaphore for
* multi-threading.
**************************************************************************/
// Class to store several variables
// Due to recursion problems these variables are not global anymore
   class G__IncSetupStack {

   public:
      G__IncSetupStack() {
         memset(this, 0, sizeof(G__IncSetupStack));
      }

      struct G__ifunc_table_internal *G__incset_p_ifunc;
      int G__incset_tagnum;

      int G__incset_func_now;
      int G__incset_func_page;
      struct G__var_array *G__incset_p_local;
      int G__incset_def_struct_member;
      int G__incset_tagdefining;
      int G__incset_def_tagnum;
      long G__incset_globalvarpointer;
      int G__incset_var_type;
      int G__incset_typenum;
      int G__incset_static_alloc;
      int G__incset_access;
      short G__incset_definemacro;

      void store() {
         G__incset_tagnum = G__tagnum;
         G__incset_typenum = G__typenum ;
         G__incset_p_ifunc = G__p_ifunc;
         G__incset_func_now = G__func_now;
         G__incset_func_page = G__func_page;
         G__incset_p_local = G__p_local;
         G__incset_globalvarpointer = G__globalvarpointer;
         G__incset_var_type = G__var_type;
         G__incset_tagdefining = G__tagdefining;
         G__incset_static_alloc = G__static_alloc;
         G__incset_access = G__access;
         G__incset_definemacro = G__definemacro;
         G__incset_def_tagnum = G__def_tagnum;
         G__incset_def_struct_member = G__def_struct_member;
      }

      void restore() {
         G__tagnum = G__incset_tagnum;
         G__typenum = G__incset_typenum;
         G__p_ifunc = G__incset_p_ifunc;
         G__func_now = G__incset_func_now;
         G__func_page = G__incset_func_page;
         G__p_local = G__incset_p_local;
         G__globalvarpointer = G__incset_globalvarpointer;
         G__var_type = G__incset_var_type;
         G__tagdefining = G__incset_tagdefining;
         G__static_alloc = G__incset_static_alloc;
         G__access = G__incset_access;
         G__definemacro = G__incset_definemacro;
         G__def_tagnum = G__incset_def_tagnum;
         G__def_struct_member = G__incset_def_struct_member;
      }

      static void push();

      static void pop();
  };

/**************************************************************************
* G__stack_instance
* Several problems in the mefunc_setup recursive calling, due to global variables using,
* was introduced. Now a stack stores the variables (G__IncSetupStack class)
* Each memfunc_setup call will have its own copy of the variables in the stack
* RETURN: This function return a pointer to the static variable stack
**************************************************************************/
   std::stack<G__IncSetupStack>* G__stack_instance(){

      // Variables Stack
      static std::stack<G__IncSetupStack>* G__stack = 0;

      // If the stack has not been initialized yet
      if (G__stack==0)
         G__stack = new std::stack<G__IncSetupStack>();

      return G__stack;

   }

   void G__IncSetupStack::push() {
      std::stack<G__IncSetupStack> *var_stack = G__stack_instance();
      G__IncSetupStack incsetup_stack;
      incsetup_stack.store();
      var_stack->push(incsetup_stack);
   }  

   void G__IncSetupStack::pop() {
      std::stack<G__IncSetupStack> *var_stack = G__stack_instance();
      G__IncSetupStack *incsetup_stack = &var_stack->top();
      incsetup_stack->restore();
      var_stack->pop();
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

/**************************************************************************
* G__CurrentCall
**************************************************************************/
static int   s_CurrentCallType = 0;
static void* s_CurrentCall  = 0;
static int   s_CurrentIndex = 0;
void G__CurrentCall(int call_type, void* call_ifunc, long *ifunc_idx)
{
  switch( call_type )   {
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
    if ( call_ifunc) *(void**)call_ifunc = s_CurrentCall;
    if ( ifunc_idx)  *ifunc_idx = s_CurrentIndex;
    break;
  case G__RETURN:
    assert(0);
    if ( call_ifunc) *(void**)call_ifunc = 0;
    if ( ifunc_idx)  *ifunc_idx  = s_CurrentCallType;
    break;
  }
}


/**************************************************************************
* Checking private constructor
**************************************************************************/
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


/**************************************************************************
* G__cpplink file name
**************************************************************************/
static char *G__CPPLINK_H;
static char *G__CPPLINK_C;

static char *G__CLINK_H;
static char *G__CLINK_C;

#ifdef G__GENWINDEF
static char *G__WINDEF;
static int G__nexports = 0;
static FILE* G__WINDEFfp = (FILE*)NULL;
static int G__isDLL=0;
static G__FastAllocString G__CINTLIBNAME("LIBCINT");
#endif

#define G__MAXDLLNAMEBUF 512

static G__FastAllocString G__PROJNAME("");
static char G__DLLID[G__MAXDLLNAMEBUF];
static char *G__INITFUNC;

static char G__NEWID[G__MAXDLLNAMEBUF];

#ifdef G__BORLANDCC5
int G__debug_compiledfunc_arg(FILE *fout,struct G__ifunc_table *ifunc,int ifn,struct G__param *libp);
static void G__ctordtor_initialize(void);
static void G__fileerror(char* fname);
static void G__ctordtor_destruct(void);
void G__cpplink_protected_stub(FILE *fp,FILE *hfp);
void G__gen_cppheader(char *headerfilein);
static void G__gen_headermessage(FILE *fp,char *fname);
void G__add_macro(char *macroin);
int G__isnonpublicnew(int tagnum);
void  G__if_ary_union_reset(int ifn,struct G__ifunc_table *ifunc);
static int G__isprotecteddestructoronelevel(int tagnum);
void  G__if_ary_union(FILE *fp,int ifn,struct G__ifunc_table *ifunc);
const char *G__mark_linked_tagnum(int tagnum);
static int G__isprivateconstructorifunc(int tagnum,int iscopy);
static int G__isprivateconstructorvar(int tagnum,int iscopy);
static int G__isprivatedestructorifunc(int tagnum);
static int G__isprivatedestructorvar(int tagnum);
static int G__isprivateassignoprifunc(int tagnum);
static int G__isprivateassignoprvar(int tagnum);
void G__cppif_gendefault(FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table *ifunc,int isconstructor,int iscopyconstructor,int isdestructor,int isassignmentoperator,int isnonpublicnew);
static char* G__vbo_funcname(int tagnum,int basetagnum,int basen);
//static int G__hascompiledoriginalbase(int tagnum);
static void G__declaretruep2f(FILE *fp,struct G__ifunc_table *ifunc,int j);
static void G__printtruep2f(FILE *fp,struct G__ifunc_table *ifunc,int j);
int G__tagtable_setup(int tagnum,int size,int cpplink,int isabstract,char *comment,G__incsetup setup_memvar,G__incsetup setup_memfunc);
int G__tag_memfunc_setup(int tagnum);
int G__memfunc_setup(char *funcname,int hash,int (*funcp)(),int type,int tagnum,int typenum,int reftype,int para_nu,int ansi,int accessin,int isconst,char *paras,char *comment
#ifdef G__TRUEP2F
                     ,void* truep2f, int isvirtual
#endif
);
int G__memfunc_setup2(char *funcname,int hash,char *mangled_name,int (*funcp)(),int type,int tagnum,int typenum,int reftype,int para_nu,int ansi,int accessin,int isconst,char *paras,char *comment
#ifdef G__TRUEP2F
                     ,void* truep2f, int isvirtual
#endif
);
int G__memfunc_next(void);
static void G__pragmalinkenum(int tagnum,int globalcomp);
void G__incsetup_memvar(int tagnum);
void G__incsetup_memfunc(int tagnum);
#endif

int G__isprivatectordtorassgn(int tagnum, G__ifunc_table_internal *ifunc, int ifn);
static int G__isprivatedestructorifunc(int tagnum);
int G__isprivatedestructor(int tagnum);
int G__isprivateassignopr(int tagnum);

int  G__execute_call(G__value *result7,G__param *libp,G__ifunc_table_internal *ifunc,int ifn);
void G__cppif_change_globalcomp();

/**************************************************************************
* G__check_setup_version()
*
*  Verify CINT and DLL version
**************************************************************************/
extern const char *G__cint_version();

void G__check_setup_version(int version,const char *func)
{
   G__init_globals();
   if (version > G__ACCEPTDLLREV_UPTO || version < G__ACCEPTDLLREV_FROM) {
      fprintf(G__sout,"\n\
!!!!!!!!!!!!!!   W A R N I N G    !!!!!!!!!!!!!\n\n\
The internal data structures have been changed.\n\
Please regenerate and recompile your dictionary which\n\
contains the definition \"%s\"\n\
using CINT version %s.\n\
your dictionary=%d. This version accepts=%d-%d\n\
and creates %d\n\n\
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n",
            func, G__cint_version(),version
              ,G__ACCEPTDLLREV_FROM
              ,G__ACCEPTDLLREV_UPTO
              ,G__CREATEDLLREV
              );
      std::string errmsg("CINT: dictionary ");
      errmsg += std::string(func) + " was built with incompatible CINT version!";
      throw std::runtime_error(errmsg);
   }
   G__store_asm_noverflow = G__asm_noverflow;
   G__store_no_exec_compile = G__no_exec_compile;
   G__store_asm_exec = G__asm_exec;
   G__abortbytecode();
   G__no_exec_compile =0;
   G__asm_exec = 0;
}

/**************************************************************************
* G__fileerror()
**************************************************************************/
static void G__fileerror(char *fname)
{
   G__FastAllocString buf(G__MAXFILENAME);
   buf.Format("Error opening %s",fname);
   perror(buf);
   throw std::runtime_error(std::string("CINT: error opening ") + fname);
}

/**************************************************************************
* G__fulltypename
**************************************************************************/
const char* G__fulltypename(int typenum)
{
  if(-1==typenum) { 
     static const char *nullstr = "";
     return nullstr;
  }
  if(-1==G__newtype.parent_tagnum[typenum]) return(G__newtype.name[typenum]);
  else {
    static G__FastAllocString buf(G__ONELINE);
    buf = G__fulltagname(G__newtype.parent_tagnum[typenum],0);
    buf += "::";
    buf += G__newtype.name[typenum];
    return(buf);
  }
}

/**************************************************************************
* G__debug_compiledfunc_arg(ifunc,ifn,libp);
*
*  Show compiled function call parameters
**************************************************************************/
int G__debug_compiledfunc_arg(FILE *fout,G__ifunc_table_internal *ifunc,int ifn,G__param *libp)
{
  G__FastAllocString temp(G__ONELINE);
  int i;
  fprintf(fout,"\n!!!Calling compiled function %s()\n",ifunc->funcname[ifn]);
  G__in_pause=1;
  for(i=0;i<libp->paran;i++) {
    G__valuemonitor(libp->para[i],temp);
    fprintf(fout,"  arg%d = %s\n",i+1,temp());
#ifdef G__NEVER
    if('u'==libp->para[i].type && -1!=libp->para[i].tagnum &&
       libp->para[i].obj.i) {
       G__varmonitor(fout,G__struct.memvar[libp->para[i].tagnum],""
                     ,"    ",libp->para[i].obj.i);
    }
#endif
  }
  G__in_pause=0;
  return(G__pause());
}


#ifdef G__NOSTUBS

/**************************************************************************
 **************************************************************************
 * Calling C++ compiled function
 **************************************************************************
 **************************************************************************/

/**************************************************************************
 * G__isbaseclass()
 *
 * check if dertag (derived tagnum) is a descendant of basetag (basetagnum)
 **************************************************************************/
int G__isbaseclass(int dertag, int basetag)
{
  int basen;
  int basetagnum;
  struct G__inheritance *baseclass;

  baseclass = G__struct.baseclass[dertag];

  /* Check every base*/
  for(basen=0;basen<baseclass->basen;basen++) {
    basetagnum = baseclass->herit[basen]->basetagnum;
    if(basetagnum==basetag) {
      return(1);
    }
  }
  return(0);
}

/**************************************************************************
 * G__ifunc_exist_base
 *
 * 09-08-08
 * look for an ifunc in all the inher. hierarchy.
 * This function will be called in the dictionary generation.
 * Not at run time.
 *
 * It will return the page of the function (its index) ...
 * or it will return -1 when there is more than one possible index
 * (we call it an ambiguous function).
 * This is merely an optimization artifact for the function lookup
 **************************************************************************/
int G__ifunc_exist_base(int ifn, G__ifunc_table_internal *ifunc)
{
  if (ifunc->tagnum<0)
    return 0;

  G__inheritance* cbases = 0;

  G__ifunc_table_internal *ifunc_res = G__struct.memfunc[ifunc->tagnum];

  if (ifunc_res){
    int base=-1;
    // Look for it in the actual class
    ifunc_res = G__ifunc_exist(ifunc, ifn, ifunc_res, &base, 0xffff);

    // it was found
    if (base!=-1 && ifunc_res)
      goto end;
  }

  // tagnum's Base Classes structure
  cbases = G__struct.baseclass[ifunc->tagnum];

  // If there are still base classes
  if (cbases){
    // Go through the base tagnums (tagnum = index in G__struct structure)
    for (int idx=0; idx < cbases->basen; ++idx){
      // Current tagnum
      int basetagnum=cbases->herit[idx]->basetagnum;

      // Current tagnum's ifunc table
      ifunc_res = G__struct.memfunc[basetagnum];

      // Continue if there are still ifuncs and the method 'ifn' is not found yet
      if (ifunc_res){
        int base=-1;

        // Does the Method 'ifn' (in ifunc) exist in the current ifunct?
        ifunc_res = G__ifunc_exist(ifunc, ifn, ifunc_res, &base, 0xffff);

        //If the number of default parameters numbers is different between the base and the derived
        //class we generete the stub
        if (base!=-1 && ifunc_res)
          goto end;

      }
    }
  }
  return 0;

 end:
  // If we end up here is because we found the ifunc and we want to know weather
  // it's an ambiguous function (could be troublesome when we try to look for it
  // using G__ifunc_page)

  // If the page has been set then we dont need to do this
  if(!ifunc_res->page_base) {

    G__ifunc_table_internal* ifuncb = G__struct.memfunc[ifunc_res->tagnum];
    G__ifunc_table_internal* ifuncf = 0;

    int n=0;
    int i;
    while(ifuncb) {
      for(i=0;i<ifuncb->allifunc;i++) {
        if(ifuncb->hash[ifn]==ifunc_res->hash[ifn] &&
           ifuncb->page_base==ifunc_res->page_base &&
           strcmp(ifuncb->funcname[ifn], ifunc_res->funcname[ifn]) == 0) {

          // 08-08-08
          // dont return just increase the counter
          ++n;

          // We have an ambiguous function...
          // mark its page_base as negative
          if(n>1){
            ifuncb->page_base *= -1;
            ifunc_res->page_base *= -1;
            return -1;
          }

        }
      }
      ifuncb=ifuncb->next;
    }

    // I guess we have to look for it in the base classes also... :/

    cbases = G__struct.baseclass[ifunc->tagnum];

    // If there are still base classes
    if (cbases){
      // Go through the base tagnums (tagnum = index in G__struct structure)
      for (int idx=0; idx < cbases->basen; ++idx){
        // Current tagnum
        int basetagnum=cbases->herit[idx]->basetagnum;

        // Current tagnum's ifunc table
        ifuncb = G__struct.memfunc[basetagnum];
        ifuncf = 0;

        // Continue if there are still ifuncs and the method 'ifn' is not found yet
        while(ifuncb) {
          for(i=0;i<ifuncb->allifunc;i++) {
            if(ifuncb->hash[ifn]==ifunc_res->hash[ifn] &&
               ifuncb->page_base==ifunc_res->page_base &&
               strcmp(ifuncb->funcname[ifn], ifunc_res->funcname[ifn]) == 0) {

              // 08-08-08
              // dont return just increase the counter
              ++n;

              // We have an ambiguous function...
              // mark its page_base as negative
              if(n>1){
                ifuncb->page_base *= -1;
                ifunc_res->page_base *= -1;
                return -1;
              }

            }
          }
          ifuncb=ifuncb->next;
        }
      }
    }

    // should not enter here
    if (n>1)
      return -1;
  }

  // in the normal case return the page number
  return ifunc_res->page_base;
}

/**************************************************************************
 * G__ifunc_page_old_dict
 *
 * 23-10-07
 * This is here just for compatibility reasons...
 * When we have an old dicionary, the page_base field will be -1...
 * in that case we need to do a full search
 **************************************************************************/
struct G__ifunc_table_internal* G__ifunc_page_old_dict(char *funcname, int hash, G__ifunc_table_internal *ifunc, int allifunc)
{
  // Look for a function with this name and special index in the given ifunc
  // (and its bases)
  int i;
  while(ifunc) {
    for(i=0;i<ifunc->allifunc;i++) {
      if((ifunc->hash[allifunc]==hash &&
          ifunc->page_base==-1 &&
          strcmp(ifunc->funcname[allifunc], funcname) == 0)
         || (ifunc->funcname[allifunc][0]=='~' && funcname[0]=='~') ) {
        return(ifunc);
      }
    }
    ifunc=ifunc->next;
  }
  return 0;
}

/**************************************************************************
 * G__ifunc_page()
 *
 * 06-08-07
 * Look for a function with this name and special index in the given ifunc
 **************************************************************************/
struct G__ifunc_table_internal* G__ifunc_page(char *funcname, int hash, int page_base, G__ifunc_table_internal *ifunc, int allifunc)
{
  int i;
  while(ifunc) {
    for(i=0;i<ifunc->allifunc;i++) {
      if((ifunc->hash[allifunc]==hash &&
          ifunc->page_base==page_base &&
          strcmp(ifunc->funcname[allifunc], funcname) == 0)
         || (ifunc->funcname[allifunc][0]=='~' && funcname[0]=='~') ) {
        return(ifunc);
      }
    }
    ifunc=ifunc->next;
  }
  return 0;
}

/**************************************************************************
 * G__ifunc_page_base()
 *
 * 06-08-07
 * Look for a function with this name and special index in the given ifunc
 * (and look for it in its bases too)
 **************************************************************************/
struct G__ifunc_table_internal * G__ifunc_page_base(char *funcname, int hash,int page_base, G__ifunc_table_internal *ifunc, int allifunc)
{
  G__ifunc_table_internal *ifunc_res = G__ifunc_page(funcname,hash,page_base,ifunc,allifunc);

  if(ifunc_res)
    return ifunc_res;

  // 23-10-07
  // This is here for compatibility reasons... and is only needed when we are mixing new dicitionaries
  // (in ROOT, for example)... with old dictionaries (like in roottest/root/io/multipleInherit)
  // where the base clase has the new scheme and the derived classes are in old dictionaries 
  ifunc_res = G__ifunc_page_old_dict(funcname,hash,ifunc,allifunc);
  if(ifunc_res)
    return ifunc_res;

  // If not.. look for it in the base classes

  // tagnum's Base Classes structure
  G__inheritance* cbases = G__struct.baseclass[ifunc->tagnum];

  // If there are still base classes
  if (cbases){
    // Go through the base tagnums (tagnum = index in G__struct structure)
    for (int idx=0; idx < cbases->basen; ++idx){
      // Current tagnum
      int basetagnum=cbases->herit[idx]->basetagnum;

      // Current tagnum's ifunc table
      ifunc_res = G__struct.memfunc[basetagnum];

      // Continue if there are still ifuncs and the method 'ifn' is not found yet
      if (ifunc_res){
        // Does the Method 'ifn' (in ifunc) exist in the current ifunct?
        ifunc_res = G__ifunc_page(funcname,hash,page_base,ifunc_res, allifunc);

        if(ifunc_res)
          return ifunc_res;
      }
    }
  }
  return 0;
}

/**************************************************************************
 * G__get_symbol_address()
 *
 * It will return a memory address that should contain the function 
 * specified by the mangled name (symbol).
 * It will look for it in all the libraries registered in G__sl_handle[i]
 * and in those loaded by using RTLD_DEFAULT.
 * It migth return null in case it's not found
 **************************************************************************/
void* G__get_symbol_address(const char* mangled_name)
{
  void *address=G__findsym(mangled_name);   

#ifdef G__OSFDLL
  if (!address)
     address = dlsym(RTLD_DEFAULT, mangled_name);
#endif
  
  return address;
}

#ifdef __x86_64__
#define ASM_X86_64_ARGS_PASSING(dval, lval) { \
__asm__ __volatile__("movlpd %0, %%xmm0"  :: "m" (dval[0]) : "%xmm0"); \
__asm__ __volatile__("movlpd %0, %%xmm1"  :: "m" (dval[1]) : "%xmm1"); \
__asm__ __volatile__("movlpd %0, %%xmm2"  :: "m" (dval[2]) : "%xmm2"); \
__asm__ __volatile__("movlpd %0, %%xmm3"  :: "m" (dval[3]) : "%xmm3"); \
__asm__ __volatile__("movlpd %0, %%xmm4"  :: "m" (dval[4]) : "%xmm4"); \
__asm__ __volatile__("movlpd %0, %%xmm5"  :: "m" (dval[5]) : "%xmm5"); \
__asm__ __volatile__("movlpd %0, %%xmm6"  :: "m" (dval[6]) : "%xmm6"); \
__asm__ __volatile__("movlpd %0, %%xmm7"  :: "m" (dval[7]) : "%xmm7"); \
__asm__ __volatile__("movq %0, %%rdi" :: "m" (lval[0]) : "%rdi"); \
__asm__ __volatile__("movq %0, %%rsi" :: "m" (lval[1]) : "%rsi"); \
__asm__ __volatile__("movq %0, %%rdx" :: "m" (lval[2]) : "%rdx"); \
__asm__ __volatile__("movq %0, %%rcx" :: "m" (lval[3]) : "%rcx"); \
__asm__ __volatile__("movq %0, %%r8"  :: "m" (lval[4]) : "%r8"); \
__asm__ __volatile__("movq %0, %%r9"  :: "m" (lval[5]) : "%r9"); \
__asm__ __volatile__("movl $8, %eax");  \
}
#endif //__x86_64__

/**************************************************************************
 * G__stub_method_asm_x86_64
 *
 * 08-08-07
 * Differentiate between the asm call of a function and the
 * logic required before it (virtual table handling and stuff)
 * "para" is the parameters (like libp) and param are the
 * formal parameters (like ifunc->param)
 * At this point the parameters have already been evaluated and we
 * only need to push them to the stack and make the function call
 **************************************************************************/
#ifdef __x86_64__
int G__stub_method_asm_x86_64(G__ifunc_table_internal *ifunc, int ifn, void* this_ptr, G__param* rpara, G__value *result7){

  void *vaddress = G__get_funcptr(ifunc, ifn);
  int paran = rpara->paran;

  int type  = ifunc->type[ifn];
  int reftype = ifunc->reftype[ifn];
  int isref = (reftype == G__PARAREFERENCE || isupper(type));
  G__params *fpara = &ifunc->param[ifn];
  int ansi = ifunc->ansi[ifn];
   
  const int imax = 6, dmax = 8;
  int objsize, i, icnt = 0, dcnt = 0;
  G__value *pval;
  G__int64 lval[imax];
  double dval[dmax];
  int isextra[rpara->paran];
   std::vector<char> returnHolder;

  if (type == 'u' && !isref) {
     int osize;
     G__value otype;
     otype.type   = 'u';
     otype.tagnum = result7->tagnum; // size of the return type!!!
     
     // Class Size
     osize = G__sizeof(&otype);
     returnHolder.reserve( osize );
     
     lval[icnt] = (G__int64)&(returnHolder[0]); icnt++;  // Object returned by value      
  }
  if (this_ptr) {
    lval[icnt] = G__getstructoffset(); icnt++;  // this pointer
  }

  for (i = 0; i < rpara->paran; i++) {
    isextra[i] = 0;
    
    int type = rpara->para[i].type;
    pval = &rpara->para[i];
    if (isupper(type))
      objsize = G__LONGALLOC;
    else
      objsize = G__sizeof(pval);
    switch (type) {
      case 'c': case 'b': case 's': case 'r': objsize = sizeof(int); break;
      case 'f': objsize = sizeof(double); break;
    }
#ifdef G__VAARG_PASS_BY_REFERENCE
    if (objsize > G__VAARG_PASS_BY_REFERENCE) {
      if (pval->ref > 0x1000) {
        if (icnt < imax) {
          lval[icnt] = pval->ref; icnt++;
        } else {
          //utmp.lval = pval->ref; ucnt++;
	  //u.push_back(utmp);
	  //u[ucnt].lval = pval->ref; ucnt++;
	  isextra[i] = 1;
        }
      } else {
        if (icnt < imax) {
          lval[icnt] = G__int(*pval); icnt++;
        } else {
          utmp.lval = G__int(*pval); ucnt++;
	  //u.push_back(utmp);
	  //u[ucnt].lval = G__int(*pval); ucnt++;
	  isextra[i] = 1;
        }
      }
      type = 'z';
    }
#endif
    switch (type) {
      case 'n': case 'm':
        if (icnt < imax) {
          lval[icnt] = (G__int64)G__Longlong(*pval); icnt++;
        } else {
          //utmp.lval = (G__int64)G__Longlong(*pval); ucnt++;
	  //u.push_back(utmp);
	  //u[ucnt].lval = (G__int64)G__Longlong(*pval); ucnt++;
	  isextra[i] = 1;
        } break;
      case 'f': case 'd':
        if (dcnt < dmax) {
          dval[dcnt] = G__double(*pval); dcnt++;
        } else {
	  //utmp.dval = G__double(*pval); ucnt++;
	  //u.push_back(utmp);
	  //u[ucnt].dval = G__double(*pval); ucnt++;
	  isextra[i] = 1;
        } break;
      case 'z': break;
      case 'u':
        if (objsize >= 16) {
          //memcpy(&utmp.lval, (void*)pval->obj.i, objsize);
	  //u.push_back(utmp);
	  //memcpy(&u[ucnt].lval, (void*)pval->obj.i, objsize);
          //ucnt += objsize/8;
	  isextra[i] = 1;
          break;
        }
        // objsize < 16 -> fall through
      case 'g': case 'c': case 'b': case 'r': case 's': case 'h': case 'i':
      case 'k': case 'l':
      default:
        if (icnt < imax) {
          lval[icnt] = G__int(*pval); icnt++;
        } else {
          //utmp.lval = G__int(*pval); ucnt++;
	  //u.push_back(utmp);
          //u[ucnt].lval = G__int(*pval); ucnt++;
	  isextra[i] = 1;
        } break;
    }
    //if (ucnt >= 20) printf("Info: more than 20 var args\n");
  }

  for (int k=paran-1; k>=0; k--) {
    if(isextra[k]) {
    void *paramref = 0;
    int isref = 0;

    G__value param = rpara->para[k];
    G__paramfunc *formal_param = fpara->operator[](k);

    if(ansi!=2 && formal_param->reftype!=G__PARANORMAL){
      isref = 1;
      paramref = (void *) param.ref;
    }

    if(ansi==2 && param.type!='i' && param.type!='d' && param.type!='f' &&  param.ref){
      isref = 1;
      paramref = (void *) param.ref;
    }

    // This means the parameter is a pointer
    if(isupper(param.type) || isupper(formal_param->type)){
      isref = 1;
      paramref = (void *)(param.obj.i);
    }

    // This means the parameter is a pointer
    if(formal_param->type=='u'){
      //isref = 1;
      paramref = (void *)(param.obj.i);
    }

    // Pushing Parameter
    // By Value or By Reference?
    if (!isref){// By Value
      unsigned char para_type = formal_param->type;

      // If we have more parameters than the declarations allows
      // (variadic function) then take the type of the actual parameter
      // ... forget to check the declaration (will be null)
      if(ansi==2)
        para_type = param.type;

      // Parameter's type? Push is different for each type
      switch(para_type){

      case 'd' : // Double = Double Word
        if((param.type=='d')||(param.type=='q')){
           double dparam = (double) G__double(param);
           __asm__ __volatile__("push %0" :: "g" (dparam));
        }
        else if(param.type=='f') {
           double fparam = (double) G__double(param);  
           __asm__ __volatile__("push %0" :: "g" (fparam));
        }
        else{
           long iparam = (long) G__int(param);
           G__value otype;
           otype.type   = 'u';
           otype.tagnum = G__tagnum;
           __asm__ __volatile__("push %0" :: "g" (iparam));
        }

        break;

      case 'i' : // Integer = Single Word
      {
        long valuei = (long) G__int(param);
        __asm__ __volatile__("push %0" :: "g" (valuei));
      }
      break;

      case 'b' : // Unsigned Char ????
      {
        unsigned char valueb = param.obj.uch;
        __asm__ __volatile__("push %0" :: "g" (valueb));
      }
      break;

      case 'c' : // Char
      {
        char valuec = param.obj.ch;
        __asm__ __volatile__("push %0" :: "g" (valuec));
      }
      break;

      case 's' : // Short
      {
        short values = param.obj.sh;
        __asm__ __volatile__("push %0" :: "g" (values));
      }
      break;

      case 'r' : // Unsigned Short
      {
        unsigned short valuer = param.obj.ush;
        __asm__ __volatile__("push %0" :: "g" (valuer));
      }
      break;

      case 'h' : // Unsigned Int
      {
        long valueh = G__uint(param);
        __asm__ __volatile__("push %0" :: "g" (valueh));
      }
      break;

      case 'l' : // Long
      {
        long valuel = G__int(param);
        __asm__ __volatile__("push %0" :: "g" (valuel));
      }
      break;

      case 'k': // Unsigned Long
      {
        long valuekb = G__uint(param);
        __asm__ __volatile__("push %0" :: "g" (valuekb));
      }
      break;

      case 'f' : // Float // Shouldnt it be treated as a double?
      {
        float valuef = (float) G__double(param);

        // Casting a single precision to a doeble precision should be safe
        // 31-01-08: We do this because the compiler complains in x86-64 (optimized)
        double valued = (double) valuef;
        __asm__ __volatile__("push %0" :: "g" (valued));
      }
      break;

      case 'n' : // Long Long
      {
        G__int64 fparam = (G__int64) G__Longlong(param);
        __asm__ __volatile__("push %0" :: "g" (fparam));

      }
      break;

      case 'm' : // unsigned Long Long
      {
        G__int64 valuem = G__Longlong(param);
        __asm__ __volatile__("push %0" :: "A" (valuem));
      }
      break;

      case 'q' : // should this be treated with two resgisters two?
      {
        long double valueq = G__Longdouble(param);
        __asm__ __volatile__("push %0" :: "g" (valueq));
      }
      break;

      case 'g' : // bool
      {
        long valueb = G__bool(param);
        __asm__ __volatile__("push %0" :: "g" (valueb));
      }
      break;

      case 'u' : // a class... treat it as a reference
      {
        __asm__ __volatile__("push %0" :: "g" ((void*)paramref));
      }
      break;

      default:
        G__fprinterr(G__serr,"Type %c not known yet (asm push)\n", para_type);
      }
    }
    else{
      //int parama = *paramref;
      __asm__ __volatile__("push %0" :: "g" ((void*)paramref));
    }
  }
  }
  
  // By Value or By Reference?
  if (!isref){// By Value

    // Although the call to the function is the same,
    // the way to pick un the result depends on the return type
    // (which makes the call look different)
    switch(type){

    case 'd' : // Double = Double Word
      if((type=='d')||(type=='q')){
        double result_val;
        ASM_X86_64_ARGS_PASSING(dval, lval)
        __asm__ __volatile__("call *%1" : "=t" (result_val) : "g" (vaddress));
        G__letdouble(result7, 100, (double) (result_val));
      }
      else{
        G__value otype;
        otype.type   = 'u';
        otype.tagnum = G__tagnum;

        ASM_X86_64_ARGS_PASSING(dval, lval)
        __asm__ __volatile__("call *%1" : "=t" (result7->obj.d) : "g" (vaddress));
      }

      break;

    case 'i' : // Integer = Single Word
    {
      int return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (return_val) : "g" (vaddress));
      G__letint(result7, 'i', (long) (return_val));
    }
    break;

    case 'b' : // Unsigned Char ????
    {
      unsigned char result_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (result_val) : "g" (vaddress));
      G__letint(result7, 'b', (long) result_val);
    }
    break;

    case 'c' : // Char
    {
      char return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (return_val) : "g" (vaddress));
      G__letint(result7, 'c', (long) (return_val));
    }
    break;

    case 's' : // Short
    {
      short return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (return_val): "g" (vaddress));
      G__letint(result7, 's', (long) (return_val));
    }
    break;

    case 'r' : // Unsigned Short
    {
      unsigned short return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (return_val): "g" (vaddress));
      G__letint(result7, 'r', (long) (return_val));
    }
    break;

    case 'h' : // Unsigned Int
    {
      unsigned int return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (return_val): "g" (vaddress));
      G__letint(result7, 'h', (long) (return_val));
    }
    break;

    case 'l' : // Long
    {
      long return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (return_val): "g" (vaddress));
      G__letint(result7, 'l', return_val);
    }
    break;

    case 'k': // Unsigned Long
    {
      unsigned long return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (result7->obj.ulo): "g" (vaddress));
      G__letint(result7, 'k', (long) (return_val));
    }
    break;

    case 'f' : // Float
    {
      float return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=t" (return_val): "g" (vaddress));
      G__letdouble(result7, 'f', (double) return_val);
    }
    break;

    case 'n' : // Long Long
    {
      long long return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=A" (return_val): "g" (vaddress));
      G__letLonglong(result7, 'n', (G__int64) (return_val));
    }
    break;

    case 'm' : // unsigned Long Long
    {
      unsigned long long return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=A" (return_val): "g" (vaddress));
      G__letLonglong(result7, 'm', (G__uint64) (return_val));
    }
    break;

    case 'q' : // long double
    {
      long double return_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=t" (return_val): "g" (vaddress));
      G__letLongdouble (result7, 'q', (long double) (return_val));
    }
    break;

    case 'g' : // bool
    {
      int result_val;
      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (result_val): "g" (vaddress));
      G__letint(result7, 'g', (long) (result_val));
    }
    break;

    case 'u' : // This is a class.. treat it as a reference
    {
      // 20-11-07
      // This means we have to return a given object (i.e) a new object..
      // This will be a temporary object for the compiles
      // and it used to look like:
      //
      // /////////////////////////
      // const Track* pobj;
      // const Track xobj = ((const Event*) G__getstructoffset())->GetTrackCopy((int) G__int(libp->para[0]));
      // pobj = new Track(xobj);
      // result7->obj.i = (long) ((void*) pobj);
      // result7->ref = result7->obj.i;
      // G__store_tempobject(*result7);
      // /////////////////////////
      //
      // In the stubs of a dictionary... we have to recreate the exact same thing here.

      //the first thing we need to find is the size
      // of the object (since we have to handle the allocation)
      // Getting the Class size
      int osize;
      G__value otype;
      otype.type   = 'u';
      otype.tagnum = result7->tagnum; // size of the return type!!!

      // Class Size
      osize = G__sizeof(&otype);

      // The second thing we need is a new object of type T (the place holder)
      void* pobject = operator new(osize);

      // The place holder is the last parameter we have to push !!!
      __asm__ __volatile__("push %0" :: "g" ((void*) pobject));

      long res=0;

      ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%1" : "=a" (res): "g" (vaddress));
      result7->obj.i = (long) ((void*) pobject);
      result7->ref = result7->obj.i;
      G__store_tempobject(*result7);
    }
    break;

    case 'y' : // treat it as void function (what is 'y' ???)
    {

       ASM_X86_64_ARGS_PASSING(dval, lval)
      __asm__ __volatile__("call *%0" :: "g" (vaddress));

      // if this a void function the return type must be 0
      // why isnt it 'y'?
      G__setnull(result7);
    }
    break;

    default:
      G__fprinterr(G__serr,"Type %c not known yet (asm call)\n", type);
    }
  }
  else{
    //int parama = *paramref;
    long res=0;
    __asm__ __volatile__("call *%1" : "=a" (res): "g" (vaddress));
    result7->obj.i = (long)res;

    if(!isupper(type)) {
      result7->ref = result7->obj.i;
      result7->obj.reftype.reftype = G__PARAREFERENCE;
    }
  }

  return 0;
}
#endif // VAARG_PASS_BY_REFERENCE


/**************************************************************************
 * G__stub_method_asm
 *
 * 08-08-07
 * Differentiate between the asm call of a function and the
 * logic required before it (virtual table handling and stuff)
 * "para" is the parameters (like libp) and param are the
 * formal parameters (like ifunc->param)
 * At this point the parameters have already been evaluated and we
 * only need to push them to the stack and make the function call
 **************************************************************************/
int G__stub_method_asm(G__ifunc_table_internal *ifunc, int ifn, void* this_ptr, G__param* rpara, G__value *result7){

   void *vaddress = G__get_funcptr(ifunc, ifn);
   int paran = rpara->paran;

   int reftype = ifunc->reftype[ifn];
   G__params *fpara = &ifunc->param[ifn];
   int ansi = ifunc->ansi[ifn];

   /* Pushing Parameters in the stack */
   for (int k=paran-1; k>=0; k--) {
      void *paramref = 0;
      int isref = 0;

      const G__value &param = rpara->para[k];
      G__paramfunc *formal_param = fpara->operator[](k);

      if(ansi!=2 && formal_param->reftype!=G__PARANORMAL){
         isref = 1;
         paramref = (void *) param.ref;
      }

      if(ansi==2 && param.type!='i' && param.type!='d' && param.type!='f' &&  param.ref){
         isref = 1;
         paramref = (void *) param.ref;
      }

      // This means the parameter is a pointer
      if(isupper(param.type) || isupper(formal_param->type)){
         isref = 1;
         paramref = (void *)(param.obj.i);
      }

      // This means the parameter is a pointer
      if(formal_param->type=='u'){
         //isref = 1;
         paramref = (void *)(param.obj.i);
      }

      // Pushing Parameter
      // By Value or By Reference?
      if (!isref){// By Value
         unsigned char para_type = formal_param->type;

         // If we have more parameters than the declarations allows
         // (variadic function) then take the type of the actual parameter
         // ... forget to check the declaration (will be null)
         if(ansi==2)
            para_type = param.type;

         // Parameter's type? Push is different for each type
         switch(para_type){

         case 'd' : // Double = Double Word

            double value;
            int *paddr;
         
            if((param.type=='d')||(param.type=='q')||(param.type=='f')) 
               value = (double) G__double(param);
            else
               value = (double) G__int(param);
            
            // Parameter Pointer
            paddr = (int*) &value;

            /* Highest Word */
            __asm__ __volatile__("push %0" :: "g" (*(paddr+1)));
            /* Lowest Word */
            __asm__ __volatile__("push %0" :: "g" (*paddr));

            break;

         case 'i' : // Integer = Single Word
         {
            long valuei = (long) G__int(param);
            __asm__ __volatile__("push %0" :: "g" (valuei));
         }
         break;

         case 'b' : // Unsigned Char ????
         {
            __asm__ __volatile__("push %0" :: "g" (param.obj.uch));
         }
         break;

         case 'c' : // Char
         {
            __asm__ __volatile__("push %0" :: "g" (param.obj.ch));
         }
         break;

         case 's' : // Short
         {
            __asm__ __volatile__("push %0" :: "g" (param.obj.sh));
         }
         break;

         case 'r' : // Unsigned Short
         {
            __asm__ __volatile__("push %0" :: "g" (param.obj.ush));
         }
         break;

         case 'h' : // Unsigned Int
         {
            long valueh = G__uint(param);
            __asm__ __volatile__("push %0" :: "g" (valueh));
         }
         break;

         case 'l' : // Long
         {
            long valuel = G__int(param);
            __asm__ __volatile__("push %0" :: "g" (valuel));
         }
         break;

         case 'k': // Unsigned Long
         {
            long valuekb = G__uint(param);
            __asm__ __volatile__("push %0" :: "g" (valuekb));
         }
         break;

         case 'f' : // Float // Shouldnt it be treated as a double?
         {
            float valuef = (float) G__double(param);
            __asm__ __volatile__("push %0" :: "g" (valuef));
         }
         break;

         case 'n' : // Long Long
         {
            G__int64 fparam = (G__int64) G__Longlong(param);
            // Parameter Pointer
            int *paddr = (int *) &fparam;

            /* Highest Word */
            __asm__ __volatile__("push %0" :: "g" (*(paddr+1)));
            /* Lowest Word */
            __asm__ __volatile__("push %0" :: "g" (*paddr));
         }
         break;

         case 'm' : // unsigned Long Long
         {
            G__int64 valuem = G__Longlong(param);
            __asm__ __volatile__("push %0" :: "A" (valuem));
         }
         break;

         case 'q' : // should this be treated with two resgisters two?
         {
            long double valueq = G__Longdouble(param);

            // Parameter Pointer
            int *paddr = (int *) &valueq;

            // This has to be checked
            __asm__ __volatile__("push %0" :: "g" (*(paddr+2)));
            /* Highest Word */
            __asm__ __volatile__("push %0" :: "g" (*(paddr+1)));
            /* Lowest Word */
            __asm__ __volatile__("push %0" :: "g" (*paddr));
         }
         break;

         case 'g' : // bool
         {
            long valueb = G__bool(param);
            __asm__ __volatile__("push %0" :: "g" (valueb));
         }
         break;

         case 'u' : // a class... treat it as a reference
         {
            __asm__ __volatile__("push %0" :: "g" ((void*)paramref));
         }
         break;

         default:
            G__fprinterr(G__serr,"Type %c not known yet (asm push)\n", para_type);
         }
      }
      else
         __asm__ __volatile__("push %0" :: "g" ((void*)paramref));
    
   }

   // Here we push the this pointer as the last parameter
   // BUT DO NOT do it if it's a static function
   if (this_ptr)
      __asm__ __volatile__("push %0" :: "g" ((void*) this_ptr));

   int type = ifunc->type[ifn];
   int isref   = 0;

   if (reftype == G__PARAREFERENCE || isupper(type))
      isref = 1;

   // By Value or By Reference?
   if (!isref){// By Value

      // Although the call to the function is the same,
      // the way to pick un the result depends on the return type
      // (which makes the call look different)
      switch(type){

      case 'd' : // Double = Double Word
         if((type=='d')||(type=='q')){
            double result_val;
            __asm__ __volatile__("call *%1" : "=t" (result_val) : "g" (vaddress));
            G__letdouble(result7, 100, (double) (result_val));
         }
         else
            __asm__ __volatile__("call *%1" : "=t" (result7->obj.d) : "g" (vaddress));

         break;

      case 'i' : // Integer = Single Word
      {
         int return_val;
         __asm__ __volatile__("call *%1" : "=a" (return_val) : "g" (vaddress));
         G__letint(result7, 'i', (long) (return_val));
      }
      break;

      case 'b' : // Unsigned Char ????
      {
         unsigned char result_val;
         __asm__ __volatile__("call *%1" : "=a" (result_val) : "g" (vaddress));
         G__letint(result7, 'b', (long) result_val);
      }
      break;

      case 'c' : // Char
      {
         char return_val;
         __asm__ __volatile__("call *%1" : "=a" (return_val) : "g" (vaddress));
         G__letint(result7, 'c', (long) (return_val));
      }
      break;

      case 's' : // Short
      {
         short return_val;
         __asm__ __volatile__("call *%1" : "=a" (return_val): "g" (vaddress));
         G__letint(result7, 's', (long) (return_val));
      }
      break;

      case 'r' : // Unsigned Short
      {
         unsigned short return_val;
         __asm__ __volatile__("call *%1" : "=a" (return_val): "g" (vaddress));
         G__letint(result7, 'r', (long) (return_val));
      }
      break;

      case 'h' : // Unsigned Int
      {
         unsigned int return_val;
         __asm__ __volatile__("call *%1" : "=a" (return_val): "g" (vaddress));
         G__letint(result7, 'h', (long) (return_val));
      }
      break;

      case 'l' : // Long
      {
         long return_val;
         __asm__ __volatile__("call *%1" : "=a" (return_val): "g" (vaddress));
         G__letint(result7, 'l', return_val);
      }
      break;

      case 'k':  // Unsigned Long
      {
         unsigned long return_val;
         __asm__ __volatile__("call *%1" : "=a" (result7->obj.ulo): "g" (vaddress));
         G__letint(result7, 'k', (long) (return_val));
      }
      break;

      case 'f' : // Float
      {
         float return_val;
         __asm__ __volatile__("call *%1" : "=t" (return_val): "g" (vaddress));
         G__letdouble(result7, 'f', (double) return_val);
      }
      break;

      case 'n' : // Long Long
      {
         long long return_val;
         __asm__ __volatile__("call *%1" : "=A" (return_val): "g" (vaddress));
         G__letLonglong(result7, 'n', (G__int64) (return_val));
      }
      break;

      case 'm' : // unsigned Long Long
      {
         unsigned long long return_val;
         __asm__ __volatile__("call *%1" : "=A" (return_val): "g" (vaddress));
         G__letLonglong(result7, 'm', (G__uint64) (return_val));
      }
      break;

      case 'q' : // long double
      {
         long double return_val;
         __asm__ __volatile__("call *%1" : "=t" (return_val): "g" (vaddress));
         G__letLongdouble (result7, 'q', (long double) (return_val));
      }
      break;

      case 'g' : // bool
      {
         int result_val;
         __asm__ __volatile__("call *%1" : "=a" (result_val): "g" (vaddress));
         G__letint(result7, 'g', (long) (result_val));
      }
      break;

      case 'u' : // This is a class.. treat it as a reference
      {
         // 20-11-07
         // This means we have to return a given object (i.e) a new object..
         // This will be a temporary object for the compiles
         // and it used to llok like:
         //
         // /////////////////////////
         // const Track* pobj;
         // const Track xobj = ((const Event*) G__getstructoffset())->GetTrackCopy((int) G__int(libp->para[0]));
         // pobj = new Track(xobj);
         // result7->obj.i = (long) ((void*) pobj);
         // result7->ref = result7->obj.i;
         // G__store_tempobject(*result7);
         // /////////////////////////
         //
         // In the stubs of a dictionary... we have to recreate the exact same thing here.

         //the first thing we need to find is the size
         // of the object (since we have to handle the allocation)
         // Getting the Class size

         // if Function is a constructor we don't return any value
         if ((ifunc->tagnum > -1) && (strcmp(ifunc->funcname[ifn], G__struct.name[ifunc->tagnum])== 0)) {
            __asm__ __volatile__("call *%0" :: "g" (vaddress));

            result7->obj.i = (long) this_ptr;
            // Object's Reference
            result7->ref = (long) this_ptr;
    
         }
         else{
            int osize;
            G__value otype;
            otype.type   = 'u';
            otype.tagnum = result7->tagnum; // size of the return type!!!

            // Class Size
            osize = G__sizeof(&otype);

            // The second thing we need is a new object of type T (the place holder)
            void* pobject = operator new(osize);

            // The place holder is the last parameter we have to push !!!
            __asm__ __volatile__("push %0" :: "g" ((void*) pobject));

            long res=0;
            __asm__ __volatile__("call *%1" : "=a" (res): "g" (vaddress));
            result7->obj.i = (long) ((void*) pobject);
            result7->ref = result7->obj.i;
            G__store_tempobject(*result7);
         }
      }
      break;

      case 'y' : // treat it as void function (what is 'y' ???)
      {
         __asm__ __volatile__("call *%0" :: "g" (vaddress));

         // if this a void function the return type must be 0
         // why isnt it 'y'?
         G__setnull(result7);
      }
      break;

      default:
         G__fprinterr(G__serr,"Type %c not known yet (asm call)\n", type);
      }
   }
   else{
      //int parama = *paramref;
      long res=0;
      __asm__ __volatile__("call *%1" : "=a" (res): "g" (vaddress));

      if(!isupper(type)) {
         result7->ref = (long) (res);
         if(type=='u') {
            result7->obj.i = (long) (res);
         }
         else {
            G__letint(result7, result7->type, (*(long*)(res)));
         }
      }
      else {
         G__letint(result7, result7->type, res);
	 result7->obj.reftype.reftype = reftype;
      }
   }

   return 0;
}

/**************************************************************************
 * G__evaluate_libp
 *
 * This function will put the parameter of libp in rpara,
 * but it will also evaluate the default parameters of ifunc
 * putting them in rpara... at the end, rpara should be everything
 * we need to execute a given function
 * it returns 0 if everything is ok or a -1 if an error is found
 **************************************************************************/
int G__evaluate_libp(G__param* rpara, G__param *libp, G__ifunc_table_internal *ifunc, int ifn)
{
  // 08-08-07
  // We need to instantiate the default parameters and create a variable
  // containing all of them...
  //struct G__param rpara;
  rpara->paran=0;

  // for methods with "..." libp->paran>ifunc->para_nu[ifn]
  // but for methods with optional parameters the opposite
  // is true;
  int paran = ifunc->para_nu[ifn];
  if ((ifunc->ansi[ifn] == 2) && libp->paran>ifunc->para_nu[ifn])
    paran = libp->paran;

  for (int counter=0; counter<paran; counter++) {
    if (counter < libp->paran) {
      // This means it's a given param (not by default)
      rpara->para[rpara->paran] = libp->para[counter];
      rpara->paran++;
    }
    else {
      // This happens when it's by default

      // If it's a parameter by default we have to get its real value
      //
      // 26/04/07
      // Note: when pdefault!=0 and pdefault!=-1 it means it's a valid
      //       pointer so I can only assume that it would be the
      //       reference of an object but when can we encounter such
      //       situation?
      G__paramfunc *formal_param = ifunc->param[ifn][counter];

      if((G__value *)(-1)==formal_param->pdefault) {
         G__dictgenmode store_dicttype = G__dicttype;
         G__dicttype = (G__dictgenmode) -1;
         rpara->para[rpara->paran] = G__getexpr(formal_param->def);
         G__dicttype = store_dicttype;
         rpara->paran++;
      }
      else if((G__value *)(0)==formal_param->pdefault){
        G__fprinterr(G__serr,"Error in G__evaluate_libp: default param not found\n");
        return -1;
      }
      else{
        G__fprinterr(G__serr,"Error in G__evaluate_libp: unknown param\n");
        return -1;
      }
    }
  }
  return 0;
}

/**************************************************************************
 * G__stub_method_calling
 *
 * Non Dictionariy (Stub functions) Assembler Method Calling
 *
 * This is the first step of the function execution without the stubs.
 * We will find the right virtual function and let everything
 * ready for the real evaluation
 *
 * result7  = Method's return
 * libp     = Method's Parameters
 * ifunc    = Interpreted Functions Table
 * ifn      = Method's index in ifunc
 *
 * See: common.h and G__c.h for types information
 **************************************************************************/
int G__stub_method_calling(G__value *result7, G__param *libp,
                           G__ifunc_table_internal *ifunc, int ifn)
{

   /**************************************************************************
    * Create a dummy strct to be able to perform a typeif(void*)
    * based on DynamicType from Reflex
    **************************************************************************/
   struct G__dyntype { virtual ~G__dyntype() {} };

   long store_struct_offset = G__store_struct_offset;
   int store_tagnum = G__tagnum;

   // Getting the class's name
   int gtagnum = ifunc->tagnum;

   // If wee don't get it from the ifunc, try getting it from
   // the environment (is it caused by CallFunc ?)
   if(gtagnum < 0)
      gtagnum = G__tagnum;
  
   // this is redundant if gtagnum < 0
   // but I want to say that G__tagnum is always changed
   G__tagnum = gtagnum; //G_getexpr will use it to find a variable within a class
  
   // Return values for new and delete parameters.
   G__value op_return;

   // 19-11-07
   // We need to evaluate the parameters by default here, since we want to use those
   // given by the static type and in the next function we will recursively call 
   // G__stub_method_calling with a different type
   struct G__param rpara;
   if(G__evaluate_libp(&rpara, libp, ifunc, ifn)==-1){
      G__fprinterr(G__serr,"Error in G__stub_method_calling: problem with the default parameters\n");
      return -1;
   }

   // We will try to do the special case for constructors here...
   // I couldnt find a field in the ifunc table telling us if it's acons.
   // so we would have to verify that the name of the function is actually
   // the name of the class (Axel said that constructors return an 'int' so
   // we can do that check to speed things up a bit)
   // Method's name == Class Name?
   if((ifunc->type[ifn] == 'i') && (gtagnum > -1) && (strcmp(ifunc->funcname[ifn], G__struct.name[gtagnum])== 0)){
      // ----------------------
      // CONSTRUCTOR PROCEDURE
      // ----------------------
      // #1: Memory location trough the new operator (Important: Is the new operator overriden?)
      // #2: Constructor Call to set up the located memory by the new operator

      // If this is a cons. the first thing we need to find is the size
      // of the object (since we have to handle the allocation)
      // Getting the Class size
      int osize;
      G__value otype;
      otype.type   = 'u';
      otype.tagnum = gtagnum;

      // Class Size
      osize = G__sizeof(&otype);   

      // Variable's pointer
      long gvp = G__getgvp();

      // Pointer to new object
      void* pobject = (void *) 0;

      /* ---------------------- */
      /*  NEW Operator calling  */
      /* ---------------------- */

      // We need to check if the new operator has been overidden by any class in the hierarchy
      // Here the new parameters
      G__param para_new;

      // index in the ifunc
      long pifn;
      long poffset;

      // Constructor arity => Number of objects to initialize. Objects array?
      // Constructor Arity. Array Constructor? Single Constructor?
      int arity = G__getaryconstruct();

      // Space to allocate
      long allocatedsize;
      if (arity)
         allocatedsize = ((long) osize)*arity;
      else
         allocatedsize = (long) osize;
    
      // New Operator ifunc entry
      G__ifunc_table_internal * new_oper;

      // New operator has at least one parameter. The size of the allocated space
      para_new.paran = 1;

      para_new.para[0].typenum = 0;
      para_new.para[0].type = 'h';
      para_new.para[0].tagnum = 0;

      G__letint(&para_new.para[0],(int) 'h', allocatedsize);

      // Do we have already an address for the object?. Is the "this" pointer valid?
      if ((gvp != G__PVOID) && (gvp != 0)) { // We have a valid space, so we will call the placement new
     
         // We already have a valid space for the object
         // Now we have a second parameter (placement new). The first is the size of the object and the second one is the address.
         para_new.paran = 2;
     
         // In this case we have a second parameter: The address for the allocated space.
         para_new.para[1].typenum = 0;
         para_new.para[1].type = 'Y';
         para_new.para[1].tagnum = 0;
       
         G__letint(&para_new.para[1],(int) 'Y', (long) gvp);
               
      }

      // We look for the "new operator" ifunc in the current class and in its bases
      if (arity)
         new_oper = G__get_methodhandle4("operator new[]", &para_new, ifunc, &pifn, &poffset, 0, 1,0);
      else 
         new_oper = G__get_methodhandle4("operator new", &para_new, ifunc, &pifn, &poffset, 0, 1,0);

      // is the new operator overriden?
      if (!new_oper) { // No, it's not
         // We use the default new operator
         if ((gvp == G__PVOID) || (gvp == 0)){ // Valid space? Placement new?

            // No valid space we have to request/allocate a new address
            if (arity)
               pobject = operator new[](osize*arity);
            else 
               pobject = operator new(osize);
         }
         else{ // We have an address already. placement New is executed
            if (arity)
               pobject = operator new[](osize*arity,(void*) gvp);
            else 
               pobject = operator new(osize,(void*) gvp);
         }
      }
      else
      { // Yes, we have a nice overriden operator new yeah c'mon!. Hack me baby! 
           
         op_return.type = 'U';
           
#ifdef __x86_64__
         G__stub_method_asm_x86_64(new_oper, pifn, 0, &para_new, &op_return);
#else      
         G__stub_method_asm(new_oper, pifn, 0, &para_new, &op_return);
#endif   
         // Allocated Address
         pobject = (void *) op_return.obj.i;
      }
  
      if (!arity)
         arity=1; // We have to execute the constructor only one time
    
      // 27-04-07
      // The actual version of Cint has a limitation with
      // the array constructor: it cant receive paramaters (or at least,
      // it ignores them). Here I will try to pass the parameters of
      // the constructors for every object (to handle something like
      // string *str = new string[10]("strriiiinnnggg")   ).

      /* ------------------- */
      /*  Constructor call   */
      /* ------------------- */

      /* n = constructor arity (see array constructor) */
      for (int i=0;i<arity;i++){
#ifdef __x86_64__
         G__stub_method_asm_x86_64(ifunc, ifn, ((void*)((long)pobject + (i*osize))), &rpara, result7);
#else      
         G__stub_method_asm(ifunc, ifn, ((void*)((long)pobject + (i*osize))), &rpara, result7);
#endif
      }

      result7->obj.i = (long) pobject;
      // Object's Reference
      result7->ref = (long) pobject;
      // Object's Type
      result7->type = 'u';
   }
   else{// Not Constructor
      char * finalclass = 0;

      // Let's try to find the final type if we have a virtual function
      // tagnum cant be -1 here because a free standing function cant be
      // virtual also.
      if(ifunc->isvirtual[ifn] && G__getstructoffset()){
         G__dyntype *dyntype = (G__dyntype *) G__getstructoffset();
         const char *mangled = typeid(*dyntype).name();

         // 08-08-07
         // look for this mangled name in our hash_map 
         // (speed things up, it could also be looked up in the cint structs)
         int tagnum = -1;
         __gnu_cxx::hash_map<const char*, int> *gmap = (__gnu_cxx::hash_map<const char*, int>*) G__get_struct_map();
         __gnu_cxx::hash_map<const char*, int>::iterator  iter = gmap->find(mangled);
         if (iter != gmap->end() ) {
            tagnum = iter->second;
         }

         // if we dont find it in the map then we have to
         // look for it in Cint and add it to the map
         if(tagnum==-1) {
            int status = 0;
            finalclass = abi::__cxa_demangle(mangled, 0, 0, &status);
            if (!finalclass) {
               G__fprinterr(G__serr,"** error demangling typeid \n");
            }
            else {
               // printf(" ** type id mangled   : %s \n", mangled);
               // printf(" ** type id demangled : %s \n", finalclass);
            }

            // Before continuingwe check that the method is implemented in the
            // the final class...
            // consider:
            //
            // TH1F h1 = new TH1F()
            // h1->Draw()
            //
            // when it gets here, the classname is TH1 because Draw()
            // is implemented there, not in TH1F. That will confuse our
            // basic implementation.

            // The 3 as a parameter means we disble autoloading... this could be a problem in certain cases
            // The case we have in mind is when we have:
            // B inherits from A, B *b=new A AND the dict lib is separated from the normal
            // lib. Only in such weird cases this might be a problem.
            tagnum = G__defined_tagname(finalclass, 3);

            // if we find it in cint then let's add it to the map
            if(tagnum>=0)
               gmap->insert(G__structmap_pair((mangled),tagnum));

            // rem to free everything returned by abi::__cxa_demangle
            if(finalclass)
               free(finalclass);
         }

         // Axel said isbaseclass is not necessary
         // 28-01-08
         // This is a bit tricky because we can find the (illegal) case of a class
         // that inherits from TObject but hasn't declared a ClassDef, in that case, CInt thinks
         // that that class doesnt inherith from TObject... but if we don't do that then the 
         // (also illegal) case of sibling casting won't be catched... which one is worse for us?
         //if (tagnum>=0 && (G__isbaseclass(tagnum, gtagnum) || G__isbaseclass(gtagnum, tagnum))  /*tagnum!=gtagnum*/) {
         if (tagnum>=0 && tagnum!=gtagnum) {
            struct G__ifunc_table_internal *new_ifunc;
            long poffset;
            long pifn = ifn;

            if(!(G__isbaseclass(tagnum, gtagnum) || G__isbaseclass(gtagnum, tagnum))){
               G__fprinterr(G__serr,"Warning: static type is %s but dynamic type is %s. Are you casting two different objects? \n", 
                            G__struct.name[gtagnum], G__struct.name[tagnum]);
            }

            new_ifunc = G__struct.memfunc[tagnum];
            G__incsetup_memfunc(tagnum);

            // 23-10-12
            // We need a way to know that we need with an old dictionary
            if(ifunc)
               new_ifunc = G__ifunc_page_base(ifunc->funcname[ifn], ifunc->hash[ifn], ifunc->page_base, new_ifunc, ifn);

            // 08-08-07
            // in case of collisions do the whole matching
            if(new_ifunc->page_base<0 ){
               G__paramfunc *parfunc = ifunc->param[ifn].fparams;
               struct G__param fpara;
               fpara.paran=0;
               fpara.para[0]=G__null;

               while (parfunc) {
                  if (parfunc->type) {
                     fpara.para[fpara.paran].tagnum  = parfunc->p_tagtable;
                     fpara.para[fpara.paran].obj.reftype.reftype = parfunc->reftype;
                     fpara.para[fpara.paran].isconst = parfunc->isconst;
                     fpara.para[fpara.paran].type    = parfunc->type;
                     fpara.paran++;
                  }
                  parfunc = parfunc->next;
               }
               new_ifunc = G__struct.memfunc[tagnum];

	       G__FastAllocString funcname(G__MAXNAME);
	       if(ifunc->funcname[ifn][0]=='~')
                  funcname.Format("~%s", G__struct.name[new_ifunc->tagnum]);
	       else
                  funcname = ifunc->funcname[ifn];

               new_ifunc = G__get_methodhandle4(funcname, &fpara, new_ifunc, &pifn, &poffset, 1, 1, 0);
            }
        
            if(new_ifunc && (ifunc!=new_ifunc)){
               // we have an animal that looks like a dog and we have
               // to make him bark...
               // i.e. we have a derived class casted as a base class
               // and we have to execute the method of the derived class not
               // the one from the base class
               int intres;
               int old_tag;
               tagnum = new_ifunc->tagnum;
               int offset = G__isanybase(gtagnum, tagnum, 0);

               // 12/04/07
               // How can I change the this ptr cleanly?
               // hacking through it doesn't look like the best solution :/
               if(offset > 0){
                  G__store_struct_offset -= offset;
               }

               // Be careful....
               old_tag = G__tagnum;
               G__tagnum = tagnum;

               if(G__get_funcptr(new_ifunc, ifn))
                  intres = G__stub_method_calling(result7, &rpara, new_ifunc, ifn); // Default params already evaluated
               else
                  intres = G__execute_call(result7, &rpara, new_ifunc, ifn); // Default params already evaluated

               G__tagnum = old_tag;
               // change back the this pointer
               if(offset != 0){
                  G__store_struct_offset += offset;
               }
               return intres;
            }
         }
         else if(tagnum==-1 && gtagnum!=tagnum){
            G__fprinterr(G__serr,"Warning: CInt doesn't know about the class %s but it knows about %s (did you forget the ClassDef?)\n", finalclass, G__struct.name[gtagnum]);
         }
      }

      // We are in serious trouble here...
      // we are trying to execute a pure virtual function.
      // This should never happen... all pure virtual functions
      // should had been redirected in the last if
      if(ifunc->ispurevirtual[ifn] && !G__get_funcptr(ifunc, ifn)) {
         G__fprinterr(G__serr,"Fatal Error: Trying to execute pure virtual function %s", ifunc->funcname[ifn]);
         return -1;
      }

      // Destructor? Method's Name == ~ClassName?
      if (ifunc->funcname[ifn][0]=='~'){
         long gvp = G__getgvp();
         long soff = G__getstructoffset();
         int arity = G__getaryconstruct();
         G__param para_del;

         int osize;
         long pifn;
         long poffset;
         G__value otype;
         otype.type   = 'u';
         otype.tagnum = gtagnum;

         osize = G__sizeof(&otype);
         if (!soff) {
            return(1);
         }

         if (!arity) 
            arity = 1;

         // delete operator ifunc pointer
         G__ifunc_table_internal *del_oper;

         // In the first step we call the destructor for the objects
         for(int idx = 0; idx < arity; idx++){
         
            if ((gvp != G__PVOID) && (gvp != 0))
               G__setgvp((long) G__PVOID);

#ifdef __x86_64__
            G__stub_method_asm_x86_64(ifunc, ifn, ((void*)((long)soff + (idx*osize))), libp, result7);     
#else          
            G__stub_method_asm(ifunc, ifn, ((void*)((long)soff + (idx*osize))), libp, result7);

#endif      

            if ((gvp != G__PVOID) && (gvp != 0))
               G__setgvp(gvp);

         }
      
         if ((gvp == G__PVOID) || (gvp == 0)) {

            // 27-07-07  
            // This means we have to delete this object from the heap
            // (calling both delete and destructor)

            para_del.paran = 1;
            para_del.para[0].typenum = 0;
            para_del.para[0].type = 'Y';
            para_del.para[0].tagnum = 0;      
      
            // We look for the "delete operator" ifunc in the current class and in its bases
            if (arity > 1)
               del_oper = G__get_methodhandle4("operator delete[]", &para_del, ifunc, &pifn, &poffset,0,1,0);
            else 
               del_oper = G__get_methodhandle4("operator delete", &para_del, ifunc, &pifn, &poffset,0,1,0);
      
            // Setting up parameter
            G__letint(&para_del.para[0],(int) 'Y', (long) soff);
      
            // Return parameter
            op_return.type = 'Y';
                            
            // is the delete operator overriden?
            if (!del_oper) { // No, it's not
               // We use the default delete operator
               if (arity > 1)
                  operator delete[]((void*) soff);
               else 
                  operator delete((void*) soff);
            }
            else{ // Yes, we have a nice overriden operator delete and we have its symbol yhea c'mon. Hack me baby!
                     
#ifdef __x86_64__
               G__stub_method_asm_x86_64(del_oper, pifn, 0, &para_del, &op_return);
#else       
               G__stub_method_asm(del_oper, pifn, 0, &para_del, &op_return);
#endif
            }
              
         }
         // destructors doesn't return anything
         G__setnull(result7);
      }
      else{
         // We dont have a this ptr if this is a static function
         void* this_ptr = 0;

         // Here get this pointer (to be passed to the asm call)
         // BUT DO NOT do it if it's a static function
         if ((gtagnum > -1) && (!ifunc->staticalloc[ifn]))
            this_ptr = (void*) G__getstructoffset();

         // Return Structure
         result7->type    = ifunc->type[ifn];
         result7->tagnum  = ifunc->p_tagtable[ifn];
         result7->typenum = ifunc->p_typetable[ifn];
         result7->isconst = ifunc->isconst[ifn];

         // 08-08-07
         // Now let's call our lower level asm function
#ifdef __x86_64__
         G__stub_method_asm_x86_64(ifunc, ifn, this_ptr, &rpara, result7);
#else      
         G__stub_method_asm(ifunc, ifn, this_ptr, &rpara, result7);
#endif
      }
   }

   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   return 0;
}
#endif // defined G__NOSTUBS

/**************************************************************************
 * G__execute_call
 *
 * Method/Function call final execution
 *
 * This function will execute a function via either the stub or the assembler call
 *
 * result7  = Method's return
 * libp     = Method's Parameters
 * ifunc    = Interpreted Functions Table
 * ifn      = Method's index in ifunc
 *
 * See: common.h and G__c.h for types information
 **************************************************************************/
int G__execute_call(G__value *result7,G__param *libp,G__ifunc_table_internal *ifunc,int ifn)
{
   G__InterfaceMethod cppfunc = (G__InterfaceMethod)ifunc->pentry[ifn]->p;

#ifdef G__NOSTUBS
    /* 15/03/2007 */
    // 1 Parameter && Registered Method in ifunc && Neither static method nor function
    // (G__tagnum > -1) is not needed because G__tagnum can be -1 when we have free
    // standing functions
    if ( ((libp->paran>=0) &&
          G__get_funcptr(ifunc, ifn) &&
          /*!(G__struct.type[ifunc->tagnum] == 'n') &&*/
          ((!G__wrappers_enabled() && !G__nostubs) || (!cppfunc && G__nostubs))
	  )){ // DMS Use the stub if there is one
      // Registered Method in ifunc. Then We Can Call the method without the stub function
      G__stub_method_calling(result7, libp, ifunc, ifn);
    }
    else 
#endif
    if (cppfunc) {
      /* 15/03/2007 */
      // this-pointer adjustment
      G__this_adjustment(ifunc, ifn);
#ifdef G__EXCEPTIONWRAPPER
      G__ExceptionWrapper((G__InterfaceMethod)cppfunc,result7,(char*)ifunc,libp,ifn);
#else
      // Stub calling
      (*cppfunc)(result7,(char*)ifunc,libp,ifn);
#endif
    }
    else if (!cppfunc && !G__get_funcptr(ifunc, ifn)) {
      G__fprinterr(G__serr,"Error in G__call_cppfunc: There is no stub nor mangled name for function: %s \n", ifunc->funcname[ifn]);

      if(ifunc->tagnum != -1)
        G__fprinterr(G__serr,"Error in G__call_cppfunc: For class: %s \n", G__struct.name[ifunc->tagnum]);   

      return -1;
    }
    else {
      // It shouldn't be here
      G__fprinterr(G__serr,"Error in G__call_cppfunc: Function %s could not be called. \n", ifunc->funcname[ifn]);
      return -1;
    }
    
    // Restore the correct type
//     fprintf(stderr,"%c:%c %d:%d %d:%d %d:%d\n",
//             result7->type, ifunc->type[ifn],
//             result7->tagnum, ifunc->p_tagtable[ifn],
//             result7->typenum, ifunc->p_typetable[ifn],
//             result7->obj.reftype.reftype, ifunc->reftype[ifn]);
            
    if (ifunc->type[ifn]!='y'
        && !(result7->type=='u' && ifunc->type[ifn]=='i' /* constructor */) ) {
       result7->type = ifunc->type[ifn];
    }
    result7->tagnum = ifunc->p_tagtable[ifn];
    result7->typenum = ifunc->p_typetable[ifn];
    if ((result7->typenum != -1) && G__newtype.nindex[result7->typenum]) {
       result7->type = toupper(result7->type);
    }
    if (isupper(ifunc->type[ifn]) && ifunc->reftype[ifn]) {
       result7->obj.reftype.reftype = ifunc->reftype[ifn];
    }

    return 1;

}

/**************************************************************************
 * G__get_funcptr()
 *
 * returns the function pointer "contained" in an ifunc.
 * if it has not been set up yet but we have the mangled name, then
 * we fetch the address using this symbol
 **************************************************************************/
void* G__get_funcptr(G__ifunc_table_internal *ifunc, int ifn)
{
  // returns the funcptr of an ifunc
  // and it's case it's null, it tries to get it
  // from the mangled_name

#ifdef G__NOSTUBS
  if( !ifunc )
    return 0;

  if(ifunc->funcptr[ifn] && ifunc->funcptr[ifn]!=(void*)-1)
    return ifunc->funcptr[ifn];

  if(!ifunc->mangled_name[ifn])
    return 0;

  ifunc->funcptr[ifn] = G__get_symbol_address(ifunc->mangled_name[ifn]);
  return ifunc->funcptr[ifn];
#else
  (void) ifunc;
  (void) ifn;
  return 0;
#endif
}

/**************************************************************************
 * G__call_cppfunc()
 *
 * Here we choose if the function will be called thorugh the stubs or
 * directly with asm calls
 **************************************************************************/
int G__call_cppfunc(G__value *result7,G__param *libp,G__ifunc_table_internal *ifunc,int ifn)
{
  G__InterfaceMethod cppfunc;
  int result;
  cppfunc = (G__InterfaceMethod)ifunc->pentry[ifn]->p;
#ifdef G__ASM
  if (G__asm_noverflow) {
    if (cppfunc == (G__InterfaceMethod) G__DLL_direct_globalfunc) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: LD_FUNC direct global function '%s' paran: %d  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , ifunc->funcname[ifn]
            , libp->paran
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__LD_FUNC;
      G__asm_inst[G__asm_cp+1] = (long) ifunc;
      G__asm_inst[G__asm_cp+2] = ifn;
      G__asm_inst[G__asm_cp+3] = libp->paran;
      G__asm_inst[G__asm_cp+4] = (long) cppfunc;
      G__asm_inst[G__asm_cp+5] = 0;
      if (ifunc->pentry[ifn]) {
         G__asm_inst[G__asm_cp+5] = ifunc->pentry[ifn]->ptradjust;
      }
      G__asm_inst[G__asm_cp+6] = (long) ifunc;
      G__asm_inst[G__asm_cp+6] = (long) ifn;
      G__inc_cp_asm(8, 0);
    }
    else {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: LD_FUNC C++ compiled '%s' paran: %d  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , ifunc->funcname[ifn]
            , libp->paran
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__LD_FUNC;
      G__asm_inst[G__asm_cp+1] = ifunc->p_tagtable[ifn];
      G__asm_inst[G__asm_cp+2] = -(ifunc->type[ifn]);
      G__asm_inst[G__asm_cp+3] = libp->paran;
      G__asm_inst[G__asm_cp+4] = (long) cppfunc;
      G__asm_inst[G__asm_cp+5] = 0;
      if (ifunc->pentry[ifn]) {
         G__asm_inst[G__asm_cp+5] = ifunc->pentry[ifn]->ptradjust;
      }
      G__asm_inst[G__asm_cp+6] = (long) ifunc;
      G__asm_inst[G__asm_cp+7] = (long) ifn;
      G__inc_cp_asm(8, 0);
    }
  }
#endif // G__ASM
  *result7 = G__null;
  result7->tagnum = ifunc->p_tagtable[ifn];
  result7->typenum = ifunc->p_typetable[ifn];
#ifndef G__OLDIMPLEMENTATION1259
  result7->isconst = ifunc->isconst[ifn];
#endif // G__OLDIMPLEMENTATION1259
  if(-1!=result7->tagnum&&'e'!=G__struct.type[result7->tagnum]) {
    if(isupper(ifunc->type[ifn])) result7->type='U';
    else                          result7->type='u';
  }
  else
    result7->type = ifunc->type[ifn];

#ifdef G__ASM
  if(G__no_exec_compile) {
    if(isupper(ifunc->type[ifn])) result7->obj.i = G__PVOID;
    else                          result7->obj.i = 0;
    result7->ref = ifunc->reftype[ifn];
    if('u'==ifunc->type[ifn]&&0==result7->ref&&-1!=result7->tagnum) {
      G__store_tempobject(*result7); /* To free tempobject in pcode */
    }
    if('u'==result7->type&&-1!=result7->tagnum) {
      result7->ref = 1;
      result7->obj.i=1;
    }
    return(1);
  }
#endif

  /* show function arguments when step into mode */
  if(G__breaksignal) {
    if(G__PAUSE_IGNORE==G__debug_compiledfunc_arg(G__sout,ifunc,ifn,libp)) {
      return(0);
    }
  }

  if('~'==ifunc->funcname[ifn][0] && 1==G__store_struct_offset &&
     -1!=ifunc->tagnum && 0==ifunc->staticalloc[ifn]) {
    /* Object is constructed when 1==G__no_exec_compile at loop compilation
     * and destructed at 0==G__no_exec_compile at 2nd iteration.
     * G__store_struct_offset is set to 1. Need to avoid calling destructor. */
    return(1);
  }

  {
    int store_asm_noverflow = G__asm_noverflow;
    G__suspendbytecode();

    long lifn = ifn;
    G__CurrentCall(G__SETMEMFUNCENV, ifunc, &lifn);

    // We store the this-pointer
    long save_offset = G__store_struct_offset;

    // We launch the method/function here!!! Either stubs or wrappers (via G__ExceptionWrapper) or direct call
    if (!G__execute_call(result7,libp,ifunc,ifn))
       return -1;

    // This-pointer restoring
    G__store_struct_offset = save_offset;

    G__CurrentCall(G__NOP, 0, 0);
    result = 1;

    if(isupper(ifunc->type[ifn]))
      result7->obj.reftype.reftype=ifunc->reftype[ifn];

    G__asm_noverflow = store_asm_noverflow;
  }
  return(result);
}

/**************************************************************************
* G__ctordtor_initialize()
**************************************************************************/
static void G__ctordtor_initialize()
{
  int i;
  G__ctordtor_status=(int*)malloc(sizeof(int)*(G__struct.alltag+1));
  for(i=0;i<G__struct.alltag+1;i++) {
    /* If link for this class is turned off but one or more member functions
     * are explicitly turned on, set G__ONLYMETHODLINK flag for the class */
    struct G__ifunc_table_internal *ifunc=G__struct.memfunc[i];
    int ifn;
    if(G__NOLINK==G__struct.globalcomp[i]) {
      while(ifunc) {
        for(ifn=0;ifn<ifunc->allifunc;ifn++) {
          if(G__METHODLINK==ifunc->globalcomp[ifn]) {
            G__struct.globalcomp[i] = G__ONLYMETHODLINK;
          }
        }
        ifunc=ifunc->next;
      }
    }
    G__ctordtor_status[i]=G__CTORDTOR_UNINITIALIZED;
  }
}
/**************************************************************************
* G__ctordtor_destruct()
**************************************************************************/
static void G__ctordtor_destruct()
{
  if(G__ctordtor_status) free(G__ctordtor_status);
}


#ifdef G__SMALLOBJECT

void G__gen_clink() {}
void G__gen_cpplink() {}

#else

/**************************************************************************
**************************************************************************
* Function to generate C interface routine G__clink.C
**************************************************************************
**************************************************************************/

/**************************************************************************
* G__gen_clink()
*
*  Generate C++ interface routine source file.
*
**************************************************************************/
void G__gen_clink()
{
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

  fp = fopen(G__CLINK_C,"a");
  if(!fp) G__fileerror(G__CLINK_C);
  fprintf(fp,"  G__c_reset_tagtable%s();\n",G__DLLID);
  fprintf(fp,"}\n");

  hfp = fopen(G__CLINK_H,"a");
  if(!hfp) G__fileerror(G__CLINK_H);

#ifdef G__BUILTIN
  fprintf(fp,"#include \"dllrev.h\"\n");
  fprintf(fp,"int G__c_dllrev%s() { return(G__CREATEDLLREV); }\n",G__DLLID);
#else
  fprintf(fp,"int G__c_dllrev%s() { return(%d); }\n",G__DLLID,G__CREATEDLLREV);
#endif

  G__cppif_func(fp,hfp);
  G__cppstub_func(fp);

  G__cpplink_typetable(fp,hfp);
  G__cpplink_memvar(fp);
  G__cpplink_global(fp);
  G__cpplink_func(fp);
  G__cpplink_tagtable(fp,hfp);
  fprintf(fp,"void G__c_setup%s() {\n",G__DLLID);
#ifdef G__BUILTIN
  fprintf(fp,"  G__check_setup_version(G__CREATEDLLREV,\"G__c_setup%s()\");\n",
          G__DLLID);
#else
  fprintf(fp,"  G__check_setup_version(%d,\"G__c_setup%s()\");\n",
          G__CREATEDLLREV,G__DLLID);
#endif
  fprintf(fp,"  G__set_c_environment%s();\n",G__DLLID);
  fprintf(fp,"  G__c_setup_tagtable%s();\n\n",G__DLLID);
  fprintf(fp,"  G__c_setup_typetable%s();\n\n",G__DLLID);
  fprintf(fp,"  G__c_setup_memvar%s();\n\n",G__DLLID);
  fprintf(fp,"  G__c_setup_global%s();\n",G__DLLID);
  fprintf(fp,"  G__c_setup_func%s();\n",G__DLLID);
  fprintf(fp,"  return;\n");
  fprintf(fp,"}\n");
  fclose(fp);
  fclose(hfp);
  G__ctordtor_destruct();
}

#ifdef G__ROOT
/**************************************************************************
* G__cpp_initialize()
*
**************************************************************************/
void G__cpp_initialize(FILE *fp)
{
  // Do not do this for cint/src/Apiif.cxx and cint/src/Apiifold.cxx.
  if (!strcmp(G__DLLID, "G__API")) {
     return;
  }
  fprintf(fp,"class G__cpp_setup_init%s {\n",G__DLLID);
  fprintf(fp,"  public:\n");
  if (G__DLLID[0]) {
    fprintf(fp,"    G__cpp_setup_init%s() { G__add_setup_func(\"%s\",(G__incsetup)(&G__cpp_setup%s)); G__call_setup_funcs(); }\n",G__DLLID,G__DLLID,G__DLLID);
    fprintf(fp,"   ~G__cpp_setup_init%s() { G__remove_setup_func(\"%s\"); }\n",G__DLLID,G__DLLID);
  } else {
    fprintf(fp,"    G__cpp_setup_init() { G__add_setup_func(\"G__Default\",(G__incsetup)(&G__cpp_setup)); }\n");
    fprintf(fp,"   ~G__cpp_setup_init() { G__remove_setup_func(\"G__Default\"); }\n");
  }
  fprintf(fp,"};\n");
  fprintf(fp,"G__cpp_setup_init%s G__cpp_setup_initializer%s;\n\n",G__DLLID,G__DLLID);
}
#endif

/**************************************************************************
* G__gen_cpplink()
*
*  Generate C++ interface routine source file.
*
**************************************************************************/
void G__gen_cpplink()
{
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

  fp = fopen(G__CPPLINK_C,"a");
  if(!fp) G__fileerror(G__CPPLINK_C);

  if(G__dicttype!=kFunctionSymbols)
     fprintf(fp,"  G__cpp_reset_tagtable%s();\n",G__DLLID);
  fprintf(fp,"}\n");

  hfp=fopen(G__CPPLINK_H,"a");
  if(!hfp) G__fileerror(G__CPPLINK_H);

  {
     int algoflag=0;
     int filen;
     char *fname;
     int lenstl;
     G__getcintsysdir();
     G__FastAllocString sysstl(strlen(G__cintsysdir)+20);
     
     sysstl.Format("%s%s%s%sstl%s",G__cintsysdir,G__psep,G__CFG_COREVERSION,G__psep,G__psep);
     lenstl=strlen(sysstl);
     for(filen=0;filen<G__nfile;filen++) {
        fname = G__srcfile[filen].filename;
        if(strncmp(fname,sysstl,lenstl)==0) fname += lenstl;
        if(strcmp(fname,"vector")==0 || strcmp(fname,"list")==0 ||
           strcmp(fname,"deque")==0 || strcmp(fname,"map")==0 ||
           strcmp(fname,"multimap")==0 || strcmp(fname,"set")==0 ||
           strcmp(fname,"multiset")==0 || strcmp(fname,"stack")==0 ||
           strcmp(fname,"queue")==0 || strcmp(fname,"climits")==0 ||
           strcmp(fname,"valarray")==0) {
           algoflag |= 1;
        }
        if(strcmp(fname,"vector.h")==0 || strcmp(fname,"list.h")==0 ||
           strcmp(fname,"deque.h")==0 || strcmp(fname,"map.h")==0 ||
           strcmp(fname,"multimap.h")==0 || strcmp(fname,"set.h")==0 ||
           strcmp(fname,"multiset.h")==0 || strcmp(fname,"stack.h")==0 ||
           strcmp(fname,"queue.h")==0) {
           algoflag |= 2;
        }
     }
     if(algoflag&1) {
        fprintf(hfp,"#include <algorithm>\n");
        if(G__ignore_stdnamespace) {
           /* fprintf(hfp,"#ifndef __hpux\n"); */
           fprintf(hfp,"namespace std { }\n");
           fprintf(hfp,"using namespace std;\n");
           /* fprintf(hfp,"#endif\n"); */
        }
     }
     else if(algoflag&2) fprintf(hfp,"#include <algorithm.h>\n");
  }

  fprintf(fp,"#include <new>\n");

#ifdef G__BUILTIN
    fprintf(fp,"#include \"dllrev.h\"\n");
    fprintf(fp,"extern \"C\" int G__cpp_dllrev%s() { return(G__CREATEDLLREV); }\n",G__DLLID);
#else
    fprintf(fp,"extern \"C\" int G__cpp_dllrev%s() { return(%d); }\n",G__DLLID,G__CREATEDLLREV);
#endif

  fprintf(hfp,"\n#ifndef G__MEMFUNCBODY\n");

  // Member Function Interface Method Ej: G__G__Hist_95_0_2()
  // Stub Functions
  if (!G__suppress_methods) {
    if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
      G__cppif_memfunc(fp,hfp);

    // 09-10-07
    // The stubs are not printed and the internal status is not changed
    if(G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
      G__cppif_change_globalcomp();
  }

  if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
    G__cppif_func(fp,hfp);

  if (!G__suppress_methods) {
    if(G__dicttype==kCompleteDictionary || G__dicttype==kNoWrappersDictionary)
      G__cppstub_memfunc(fp);
  }

  if(G__dicttype==kCompleteDictionary || G__dicttype==kNoWrappersDictionary)
    G__cppstub_func(fp);

  fprintf(hfp,"#endif\n\n");

  if(G__dicttype==kCompleteDictionary || G__dicttype==kNoWrappersDictionary) {
    G__cppif_p2memfunc(fp);

#ifdef G__VIRTUALBASE
    G__cppif_inheritance(fp);
#endif
    G__cpplink_inheritance(fp);
    G__cpplink_typetable(fp,hfp);
    G__cpplink_memvar(fp);
    if (!G__suppress_methods) G__cpplink_memfunc(fp);
    G__cpplink_global(fp);
    G__cpplink_func(fp);
    G__cpplink_tagtable(fp,hfp);
    fprintf(fp,"extern \"C\" void G__cpp_setup%s(void) {\n",G__DLLID);
#ifdef G__BUILTIN
    fprintf(fp,"  G__check_setup_version(G__CREATEDLLREV,\"G__cpp_setup%s()\");\n",
            G__DLLID);
#else
    fprintf(fp,"  G__check_setup_version(%d,\"G__cpp_setup%s()\");\n",
            G__CREATEDLLREV,G__DLLID);
#endif
    fprintf(fp,"  G__set_cpp_environment%s();\n",G__DLLID);
    fprintf(fp,"  G__cpp_setup_tagtable%s();\n\n",G__DLLID);
    fprintf(fp,"  G__cpp_setup_inheritance%s();\n\n",G__DLLID);
    fprintf(fp,"  G__cpp_setup_typetable%s();\n\n",G__DLLID);
    fprintf(fp,"  G__cpp_setup_memvar%s();\n\n",G__DLLID);
    if(!G__suppress_methods)
      fprintf(fp,"  G__cpp_setup_memfunc%s();\n",G__DLLID);
    fprintf(fp,"  G__cpp_setup_global%s();\n",G__DLLID);
    fprintf(fp,"  G__cpp_setup_func%s();\n",G__DLLID);
    G__set_sizep2memfunc(fp);
    fprintf(fp,"  return;\n");
    fprintf(fp,"}\n");

#ifdef G__ROOT
    /* Only activated for ROOT at this moment. Need to come back */
    G__cpp_initialize(fp);
#endif

  }
    fclose(fp);
    fclose(hfp);
#ifdef G__GENWINDEF
    fprintf(G__WINDEFfp,"\n");
    fclose(G__WINDEFfp);
#endif

    G__ctordtor_destruct();
  
}

/**************************************************************************
* G__cleardictfile()
**************************************************************************/
int G__cleardictfile(int flag)
{
  if(EXIT_SUCCESS!=flag) {
    G__fprinterr(G__serr,"!!!Removing ");
    if(G__CPPLINK_C) {
      remove(G__CPPLINK_C);
      G__fprinterr(G__serr,"%s ",G__CPPLINK_C);
    }
    if(G__CPPLINK_H) {
      remove(G__CPPLINK_H);
      G__fprinterr(G__serr,"%s ",G__CPPLINK_H);
    }
    if(G__CLINK_C) {
      remove(G__CLINK_C);
      G__fprinterr(G__serr,"%s ",G__CLINK_C);
    }
    if(G__CLINK_H) {
      remove(G__CLINK_H);
      G__fprinterr(G__serr,"%s ",G__CLINK_H);
    }
    G__fprinterr(G__serr,"!!!\n");
  }
#ifdef G__GENWINDEF
  if(G__WINDEF) {
    /* unlink(G__WINDEF); */
    free(G__WINDEF);
  }
#endif
  if(G__CPPLINK_H) free(G__CPPLINK_H);
  if(G__CPPLINK_C) free(G__CPPLINK_C);
  if(G__CLINK_H) free(G__CLINK_H);
  if(G__CLINK_C) free(G__CLINK_C);

#ifdef G__GENWINDEF
  G__WINDEF = (char*)NULL;
#endif
  G__CPPLINK_C = (char*)NULL;
  G__CPPLINK_H = (char*)NULL;
  G__CLINK_C = (char*)NULL;
  G__CLINK_H = (char*)NULL;
  return(0);
}


/**************************************************************************
* G__clink_header()
*
**************************************************************************/
void G__clink_header(FILE *fp)
{
  int i;
  fprintf(fp,"#include <stddef.h>\n");
  fprintf(fp,"#include <stdio.h>\n");
  fprintf(fp,"#include <stdlib.h>\n");
  fprintf(fp,"#include <math.h>\n");
  fprintf(fp,"#include <string.h>\n");
  if(G__multithreadlibcint)
    fprintf(fp,"#define G__MULTITHREADLIBCINTC\n");
  fprintf(fp,"#define G__ANSIHEADER\n");
  fprintf(fp,"#define G__DICTIONARY\n");
#if defined(__hpux) && !defined(G__ROOT)
  G__getcintsysdir();
  fprintf(fp,"#include \"%s/%s/inc/G__ci.h\"\n",G__cintsysdir, G__CFG_COREVERSION);
#else
  fprintf(fp,"#include \"G__ci.h\"\n");
#endif
  if(G__multithreadlibcint)
    fprintf(fp,"#undef G__MULTITHREADLIBCINTC\n");

#if defined(G__BORLAND) || defined(G__VISUAL)
  fprintf(fp,"extern G__DLLEXPORT int G__c_dllrev%s();\n",G__DLLID);
  fprintf(fp,"extern G__DLLEXPORT void G__set_c_environment%s();\n",G__DLLID);
  fprintf(fp,"extern G__DLLEXPORT void G__c_setup_tagtable%s();\n",G__DLLID);
  fprintf(fp,"extern G__DLLEXPORT void G__c_setup_typetable%s();\n",G__DLLID);
  fprintf(fp,"extern G__DLLEXPORT void G__c_setup_memvar%s();\n",G__DLLID);
  fprintf(fp,"extern G__DLLEXPORT void G__c_setup_global%s();\n",G__DLLID);
  fprintf(fp,"extern G__DLLEXPORT void G__c_setup_func%s();\n",G__DLLID);
  fprintf(fp,"extern G__DLLEXPORT void G__c_setup%s();\n",G__DLLID);
  if(G__multithreadlibcint) {
    fprintf(fp,"extern G__DLLEXPORT void G__SetCCintApiPointers G__P((\n");
#if !defined(G__OLDIMPLEMENTATION1485)
    for(i=0;i<125;i++) {
      fprintf(fp,"\tvoid*");
      if(i!=124) fprintf(fp,",\n");
    }
#elif !defined(G__OLDIMPLEMENTATION1546)
    for(i=0;i<124;i++) {
      fprintf(fp,"\tvoid*");
      if(i!=123) fprintf(fp,",\n");
    }
#else
    for(i=0;i<122;i++) {
      fprintf(fp,"\tvoid*");
      if(i!=121) fprintf(fp,",\n");
    }
#endif
    fprintf(fp,"));\n");
  }
#else
  fprintf(fp,"extern void G__c_setup_tagtable%s();\n",G__DLLID);
  fprintf(fp,"extern void G__c_setup_typetable%s();\n",G__DLLID);
  fprintf(fp,"extern void G__c_setup_memvar%s();\n",G__DLLID);
  fprintf(fp,"extern void G__c_setup_global%s();\n",G__DLLID);
  fprintf(fp,"extern void G__c_setup_func%s();\n",G__DLLID);
  fprintf(fp,"extern void G__set_c_environment%s();\n",G__DLLID);
  if(G__multithreadlibcint) {
    fprintf(fp,"extern void G__SetCCintApiPointers G__P((\n");
#if !defined(G__OLDIMPLEMENTATION1485)
    for(i=0;i<125;i++) {
      fprintf(fp,"\tvoid*");
      if(i!=124) fprintf(fp,",\n");
    }
#elif !defined(G__OLDIMPLEMENTATION1546)
    for(i=0;i<124;i++) {
      fprintf(fp,"\tvoid*");
      if(i!=123) fprintf(fp,",\n");
    }
#else
    for(i=0;i<122;i++) {
      fprintf(fp,"\tvoid*");
      if(i!=121) fprintf(fp,",\n");
    }
#endif
    fprintf(fp,"));\n");
  }
#endif


  fprintf(fp,"\n");
  fprintf(fp,"\n");
}

/**************************************************************************
* G__cpplink_header()
*
**************************************************************************/
void G__cpplink_header(FILE *fp)
{
  int i;
  fprintf(fp,"#include <stddef.h>\n");
  fprintf(fp,"#include <stdio.h>\n");
  fprintf(fp,"#include <stdlib.h>\n");
  fprintf(fp,"#include <math.h>\n");
  fprintf(fp,"#include <string.h>\n");
  if(G__multithreadlibcint)
    fprintf(fp,"#define G__MULTITHREADLIBCINTCPP\n");
  fprintf(fp,"#define G__ANSIHEADER\n");
  fprintf(fp,"#define G__DICTIONARY\n");
  fprintf(fp,"#define G__PRIVATE_GVALUE\n");
#if defined(__hpux) && !defined(G__ROOT)
  G__getcintsysdir();
  fprintf(fp,"#include \"%s/%s/inc/G__ci.h\"\n",G__cintsysdir, G__CFG_COREVERSION);
  fprintf(fp,"#include \"%s/%s/inc/FastAllocString.h\"\n", G__cintsysdir, G__CFG_COREVERSION);
#else
  fprintf(fp,"#include \"G__ci.h\"\n");
  fprintf(fp,"#include \"FastAllocString.h\"\n");
#endif
  if(G__multithreadlibcint)
    fprintf(fp,"#undef G__MULTITHREADLIBCINTCPP\n");

  // 10-07-07
  if(G__dicttype==kCompleteDictionary || G__dicttype==kNoWrappersDictionary) {
     fprintf(fp,"extern \"C\" {\n");

#if defined(G__BORLAND) || defined(G__VISUAL)
     fprintf(fp,"extern G__DLLEXPORT int G__cpp_dllrev%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__set_cpp_environment%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__cpp_setup_tagtable%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__cpp_setup_inheritance%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__cpp_setup_typetable%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__cpp_setup_memvar%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__cpp_setup_global%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__cpp_setup_memfunc%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__cpp_setup_func%s();\n",G__DLLID);
     fprintf(fp,"extern G__DLLEXPORT void G__cpp_setup%s();\n",G__DLLID);
     if(G__multithreadlibcint) {
        fprintf(fp,"extern G__DLLEXPORT void G__SetCppCintApiPointers G__P((\n");
#if !defined(G__OLDIMPLEMENTATION1485)
        for(i=0;i<125;i++) {
           fprintf(fp,"\tvoid*");
           if(i!=124) fprintf(fp,",\n");
        }
#elif !defined(G__OLDIMPLEMENTATION1546)
        for(i=0;i<124;i++) {
           fprintf(fp,"\tvoid*");
           if(i!=123) fprintf(fp,",\n");
        }
#else
        for(i=0;i<122;i++) {
           fprintf(fp,"\tvoid*");
           if(i!=121) fprintf(fp,",\n");
        }
#endif
        fprintf(fp,"));\n");
     }
#else
     fprintf(fp,"extern void G__cpp_setup_tagtable%s();\n",G__DLLID);
     fprintf(fp,"extern void G__cpp_setup_inheritance%s();\n",G__DLLID);
     fprintf(fp,"extern void G__cpp_setup_typetable%s();\n",G__DLLID);
     fprintf(fp,"extern void G__cpp_setup_memvar%s();\n",G__DLLID);
     fprintf(fp,"extern void G__cpp_setup_global%s();\n",G__DLLID);
     fprintf(fp,"extern void G__cpp_setup_memfunc%s();\n",G__DLLID);
     fprintf(fp,"extern void G__cpp_setup_func%s();\n",G__DLLID);
     fprintf(fp,"extern void G__set_cpp_environment%s();\n",G__DLLID);
     if(G__multithreadlibcint) {
        fprintf(fp,"extern void G__SetCppCintApiPointers G__P((\n");
#if !defined(G__OLDIMPLEMENTATION1485)
        for(i=0;i<125;i++) {
           fprintf(fp,"\tvoid*");
           if(i!=124) fprintf(fp,",\n");
        }
#elif !defined(G__OLDIMPLEMENTATION1546)
        for(i=0;i<124;i++) {
           fprintf(fp,"\tvoid*");
           if(i!=123) fprintf(fp,",\n");
        }
#else
        for(i=0;i<122;i++) {
           fprintf(fp,"\tvoid*");
           if(i!=121) fprintf(fp,",\n");
        }
#endif
        fprintf(fp,"));\n");
     }
#endif


     fprintf(fp,"}\n");
     fprintf(fp,"\n");
     fprintf(fp,"\n");
  }
}

/**************************************************************************
**************************************************************************
* Function to generate C++ interface routine G__cpplink.C
**************************************************************************
**************************************************************************/


/**************************************************************************
* G__map_cpp_name()
**************************************************************************/
char *G__map_cpp_name(const char *in)
{
   static G__FastAllocString out(G__MAXNAME*6);
   unsigned int i=0,j=0,c;
   while((c=in[i])) {
      if (out.Capacity() < (j+3)) {
         out.Resize(2*j);
      }
      switch(c) {
         case '+': strcpy(out+j,"pL"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '-': strcpy(out+j,"mI"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '*': strcpy(out+j,"mU"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '/': strcpy(out+j,"dI"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '&': strcpy(out+j,"aN"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '%': strcpy(out+j,"pE"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '|': strcpy(out+j,"oR"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '^': strcpy(out+j,"hA"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '>': strcpy(out+j,"gR"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '<': strcpy(out+j,"lE"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '=': strcpy(out+j,"eQ"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '~': strcpy(out+j,"wA"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '.': strcpy(out+j,"dO"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '(': strcpy(out+j,"oP"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ')': strcpy(out+j,"cP"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '[': strcpy(out+j,"oB"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ']': strcpy(out+j,"cB"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '!': strcpy(out+j,"nO"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ',': strcpy(out+j,"cO"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '$': strcpy(out+j,"dA"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ' ': strcpy(out+j,"sP"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ':': strcpy(out+j,"cL"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '"': strcpy(out+j,"dQ"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '@': strcpy(out+j,"aT"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '\'': strcpy(out+j,"sQ"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '\\': strcpy(out+j,"fI"); j+=2; break; // Okay: we resized the underlying buffer if needed
         default: out[j++]=c; break;
      }
      ++i;
   }
   out[j]='\0';
   return(out);
}


/**************************************************************************
* G__map_cpp_funcname()
*
* Mapping between C++ function and parameter name to cint interface
* function name. This routine handles mapping of function and operator
* overloading in linked C++ object.
**************************************************************************/
char *G__map_cpp_funcname(int tagnum,const char * /* funcname */,int ifn,int page)
{
   static G__FastAllocString mapped_name(G__MAXNAME);
  const char *dllid;

  if(G__DLLID[0]) dllid=G__DLLID;
  else if(G__PROJNAME[0]) dllid=G__PROJNAME;
  else dllid="";

  if(-1==tagnum) {
     mapped_name.Format("G__%s__%d_%d",G__map_cpp_name(dllid),ifn,page);
  }
  else {
     mapped_name.Format("G__%s_%d_%d_%d",G__map_cpp_name(dllid),tagnum,ifn,page);
  }
  return(mapped_name);

}

/**************************************************************************
* G__cpplink_protected_stub_ctor
*
**************************************************************************/
void G__cpplink_protected_stub_ctor(int tagnum,FILE *hfp)
{
  struct G__ifunc_table_internal *memfunc = G__struct.memfunc[tagnum];
  int ifn;

  while(memfunc) {
    for(ifn=0;ifn<memfunc->allifunc;ifn++) {
      if(strcmp(G__struct.name[tagnum],memfunc->funcname[ifn])==0) {
        int i;
        fprintf(hfp,"  %s_PR(" ,G__get_link_tagname(tagnum));
        for(i=0;i<memfunc->para_nu[ifn];i++) {
          if(i) fprintf(hfp,",");
          fprintf(hfp,"%s a%d"
                  ,G__type2string(memfunc->param[ifn][i]->type
                                  ,memfunc->param[ifn][i]->p_tagtable
                                  ,memfunc->param[ifn][i]->p_typetable
                                  ,memfunc->param[ifn][i]->reftype
                                  ,memfunc->param[ifn][i]->isconst)
                  ,i);
        }
        fprintf(hfp,")\n");
        fprintf(hfp,": %s(" ,G__fulltagname(tagnum,1));
        for(i=0;i<memfunc->para_nu[ifn];i++) {
          if(i) fprintf(hfp,",");
          fprintf(hfp,"a%d",i);
        }
        fprintf(hfp,") {}\n");
      }
    }
    memfunc=memfunc->next;
  }
}

/**************************************************************************
* G__cpplink_protected_stub
*
**************************************************************************/
void G__cpplink_protected_stub(FILE *fp,FILE *hfp)
{
  int i;
  /* Create stub derived class for protected member access */
  fprintf(hfp,"\n/* STUB derived class for protected member access */\n");
  for(i=0;i<G__struct.alltag;i++) {
    if(G__CPPLINK == G__struct.globalcomp[i] && G__struct.hash[i] &&
       G__struct.protectedaccess[i] ) {
      int ig15,ifn,n;
      struct G__var_array *memvar = G__struct.memvar[i];
      struct G__ifunc_table_internal *memfunc = G__struct.memfunc[i];
      fprintf(hfp,"class %s_PR : public %s {\n"
              ,G__get_link_tagname(i),G__fulltagname(i,1));
      fprintf(hfp," public:\n");
      if(((G__struct.funcs[i]&G__HAS_XCONSTRUCTOR) ||
          (G__struct.funcs[i]&G__HAS_COPYCONSTRUCTOR))
         && 0==(G__struct.funcs[i]&G__HAS_DEFAULTCONSTRUCTOR)) {
        G__cpplink_protected_stub_ctor(i,hfp);
      }
      /* member function */
      while(memfunc) {
        for(ifn=0;ifn<memfunc->allifunc;ifn++) {
          if((G__PROTECTED==memfunc->access[ifn]
              || ((G__PRIVATEACCESS&G__struct.protectedaccess[i]) &&
                  G__PRIVATE==memfunc->access[ifn])
              ) &&
             strcmp(memfunc->funcname[ifn],G__struct.name[i])==0) {
            fprintf(hfp,"  %s_PR(",G__get_link_tagname(i));
            if(0==memfunc->para_nu[ifn]) {
              fprintf(hfp,"void");
            }
            else {
              for(n=0;n<memfunc->para_nu[ifn];n++) {
                if(n!=0) fprintf(hfp,",");
                fprintf(hfp,"%s G__%d"
                        ,G__type2string(memfunc->param[ifn][n]->type
                                        ,memfunc->param[ifn][n]->p_tagtable
                                        ,memfunc->param[ifn][n]->p_typetable
                                        ,memfunc->param[ifn][n]->reftype
                                        ,memfunc->param[ifn][n]->isconst),n);
              }
            }
            fprintf(hfp,") : %s(",G__fulltagname(i,1));
            if(0<memfunc->para_nu[ifn]) {
              for(n=0;n<memfunc->para_nu[ifn];n++) {
                if(n!=0) fprintf(hfp,",");
                fprintf(hfp,"G__%d",n);
              }
            }
            fprintf(hfp,") { }\n");
          }
          if((G__PROTECTED==memfunc->access[ifn]
              || ((G__PRIVATEACCESS&G__struct.protectedaccess[i]) &&
                  G__PRIVATE==memfunc->access[ifn])
              ) &&
             strcmp(memfunc->funcname[ifn],G__struct.name[i])!=0 &&
             '~'!=memfunc->funcname[ifn][0]) {
            if(memfunc->staticalloc[ifn]) fprintf(hfp,"  static ");
            fprintf(hfp,"  %s G__PT_%s("
                    ,G__type2string(memfunc->type[ifn]
                                    ,memfunc->p_tagtable[ifn]
                                    ,memfunc->p_typetable[ifn]
                                    ,memfunc->reftype[ifn]
                                    ,memfunc->isconst[ifn])
                    ,memfunc->funcname[ifn]);
            if(0==memfunc->para_nu[ifn]) {
              fprintf(hfp,"void");
            }
            else {
              for(n=0;n<memfunc->para_nu[ifn];n++) {
                if(n!=0) fprintf(hfp,",");
                fprintf(hfp,"%s G__%d"
                        ,G__type2string(memfunc->param[ifn][n]->type
                                        ,memfunc->param[ifn][n]->p_tagtable
                                        ,memfunc->param[ifn][n]->p_typetable
                                        ,memfunc->param[ifn][n]->reftype
                                        ,memfunc->param[ifn][n]->isconst),n);
              }
            }
            fprintf(hfp,") {\n");
            if('y'!=memfunc->type[ifn]) fprintf(hfp,"    return(");
            else                        fprintf(hfp,"    ");
            fprintf(hfp,"%s(",memfunc->funcname[ifn]);
            for(n=0;n<memfunc->para_nu[ifn];n++) {
              if(n!=0) fprintf(hfp,",");
              fprintf(hfp,"G__%d",n);
            }
            fprintf(hfp,")");
            if('y'!=memfunc->type[ifn]) fprintf(hfp,")");
            fprintf(hfp,";\n");
            fprintf(hfp,"  }\n");
          }
        }
        memfunc = memfunc->next;
      }
      /* data member */
      while(memvar) {
        for(ig15=0;ig15<memvar->allvar;ig15++) {
          if(G__PROTECTED==memvar->access[ig15]) {
            if(G__AUTO==memvar->statictype[ig15])
              fprintf(hfp,"  long G__OS_%s(){return((long)(&%s)-(long)this);}\n"
                      ,memvar->varnamebuf[ig15],memvar->varnamebuf[ig15]);
            else
              fprintf(hfp,"  static long G__OS_%s(){return((long)(&%s));}\n"
                      ,memvar->varnamebuf[ig15],memvar->varnamebuf[ig15]);
          }
        }
        memvar = memvar->next;
      }
      fprintf(hfp,"};\n");
    }
  }
  fprintf(fp,"\n");
}

/**************************************************************************
* G__cpplink_linked_taginfo
*
**************************************************************************/
void G__cpplink_linked_taginfo(FILE *fp,FILE *hfp)
{
   int i;
   G__FastAllocString buf(G__MAXFILENAME);
   FILE* pfp;
   if(G__privateaccess) {
      char *xp;
      buf = G__CPPLINK_H;
      xp = strstr(buf,".h");
      if (xp) {
         size_t pos = xp - buf.data();
         buf[pos] = '\0';
         buf += "P.h";
      }
      pfp = fopen(buf,"r");
      if(pfp) {
         fclose(pfp);
         remove(buf);
      }
      pfp = fopen(buf,"w");
      fprintf(pfp,"#ifdef PrivateAccess\n");
      fprintf(pfp,"#undef PrivateAccess\n");
      fprintf(pfp,"#endif\n");
      fprintf(pfp,"#define PrivateAccess(name) PrivateAccess_##name\n");
      fclose(pfp);
   }
   fprintf(fp,"/* Setup class/struct taginfo */\n");
   for(i=0;i<G__struct.alltag;i++) {
      if((G__NOLINK > G__struct.globalcomp[i]
          || G__ONLYMETHODLINK==G__struct.globalcomp[i]
          ) &&
         (
          (G__struct.hash[i] || 0==G__struct.name[i][0])
          || -1!=G__struct.parent_tagnum[i])) {
            fprintf(fp,"G__linked_taginfo %s = { \"%s\" , %d , -1 };\n"
                    ,G__get_link_tagname(i),G__fulltagname(i,0),G__struct.type[i]);
            fprintf(hfp,"extern G__linked_taginfo %s;\n",G__get_link_tagname(i));
            if(G__privateaccess) {
               pfp = fopen(buf,"a");
               if(pfp) {
                  if(G__PRIVATEACCESS&G__struct.protectedaccess[i])
                     fprintf(pfp,"#define PrivateAccess_%s  friend class %s_PR;\n"
                             ,G__fulltagname(i,1),G__get_link_tagname(i));
                  else
                     fprintf(pfp,"#define PrivateAccess_%s \n",G__fulltagname(i,1));
                  fclose(pfp);
               }
            }
         }
   }
   fprintf(fp,"\n");
   
   fprintf(fp,"/* Reset class/struct taginfo */\n");
   switch(G__globalcomp) {
      case G__CLINK:
         fprintf(fp,"void G__c_reset_tagtable%s() {\n",G__DLLID);
         break;
      case G__CPPLINK:
      default:
         fprintf(fp,"extern \"C\" void G__cpp_reset_tagtable%s() {\n",G__DLLID);
         break;
   }
   
   for(i=0;i<G__struct.alltag;i++) {
      if((G__NOLINK > G__struct.globalcomp[i]
          || G__ONLYMETHODLINK==G__struct.globalcomp[i]
          ) &&
         (
          (G__struct.hash[i] || 0==G__struct.name[i][0])
          || -1!=G__struct.parent_tagnum[i])) {
            fprintf(fp,"  %s.tagnum = -1 ;\n",G__get_link_tagname(i));
         }
   }

   fprintf(fp,"}\n\n");

   G__cpplink_protected_stub(fp,hfp);
}
   
G__pMethodUpdateClassInfo G__UserSpecificUpdateClassInfo;

/**************************************************************************
* G__get_linked_tagnum
*
*  Setup and return tagnum
**************************************************************************/
int G__get_linked_tagnum(G__linked_taginfo *p)
{
  if(!p) return(-1);
  if(-1==p->tagnum) {
     p->tagnum = G__search_tagname(p->tagname,p->tagtype);
  }
  return(p->tagnum);
}

/**************************************************************************
* G__get_linked_tagnum_fwd
*
*  Setup and return tagnum; no autoloading
**************************************************************************/
int G__get_linked_tagnum_fwd(G__linked_taginfo *p)
{
  if(!p) return(-1);
  int type = p->tagtype;
  p->tagtype = toupper(type);
  int ret = G__get_linked_tagnum(p);
  p->tagtype = type;
  return ret;
}

/**************************************************************************
* G__get_linked_tagnum_with_param
*
*  Setup and return tagnum; also set user parameter
**************************************************************************/
int G__get_linked_tagnum_with_param(G__linked_taginfo *p,void* param)
{
  int tag = G__get_linked_tagnum(p);
  if(tag != -1) {
    G__struct.userparam[tag] = param;
    return tag;
  }
  return -1;
}

/**************************************************************************
* G__get_linked_user_param
*
*  Retrieve user parameter
**************************************************************************/
void* G__get_linked_user_param(int tag_num)
{
  if ( tag_num<0 || tag_num>=G__MAXSTRUCT ) return 0;
  return G__struct.userparam[tag_num];
}

/**************************************************************************
* G__get_link_tagname
*
*  Setup and return tagnum
**************************************************************************/
char *G__get_link_tagname(int tagnum)
{
  static G__FastAllocString mapped_tagname(G__MAXNAME);
  if(G__struct.hash[tagnum]) {
     mapped_tagname.Format("G__%sLN_%s"  ,G__DLLID
                           ,G__map_cpp_name(G__fulltagname(tagnum,0)));
  }
  else {
    mapped_tagname.Format("G__%sLN_%s%d"  ,G__DLLID
           ,G__map_cpp_name(G__fulltagname(tagnum,0)),tagnum);
  }
  return(mapped_tagname);
}

/**************************************************************************
* G__mark_linked_tagnum
*
*  Setup and return tagnum
**************************************************************************/
const char *G__mark_linked_tagnum(int tagnum)
{
  int tagnumorig = tagnum;
  if(tagnum<0) {
    G__fprinterr(G__serr,"Internal error: G__mark_linked_tagnum() Illegal tagnum %d\n",tagnum);
    return("");
  }

  while(tagnum>=0) {
    if(G__NOLINK == G__struct.globalcomp[tagnum]) {
      /* this class is unlinked but tagnum interface requested.
       * G__globalcomp is already G__CLINK=-2 or G__CPPLINK=-1,
       * Following assignment will decrease the value by 2 more */
      G__struct.globalcomp[tagnum] = G__globalcomp-2;
    }
    tagnum = G__struct.parent_tagnum[tagnum];
  }
  return(G__get_link_tagname(tagnumorig));
}


/**************************************************************************
* G__set_DLLflag()
*
*
**************************************************************************/
#ifdef G__GENWINDEF
void G__setDLLflag(int flag)
{
  G__isDLL = flag;
}
#else
void G__setDLLflag(int /* flag */) {}
#endif

/**************************************************************************
* G__setPROJNAME()
*
*
**************************************************************************/
void G__setPROJNAME(char *proj)
{
   G__PROJNAME = G__map_cpp_name(proj);
}

/**************************************************************************
* G__setCINTLIBNAME()
*
*
**************************************************************************/
#ifdef G__GENWINDEF
void G__setCINTLIBNAME(char * cintlib)
{
   G__CINTLIBNAME = cintlib;
}
#else
void G__setCINTLIBNAME(char * /*cintlib*/) {}
#endif

#ifdef G__GENWINDEF
/**************************************************************************
* G__write_windef_header()
*
*
**************************************************************************/
static void G__write_windef_header()
{
  FILE* fp;

  fp = fopen(G__WINDEF,"w");
  if(!fp) G__fileerror(G__WINDEF);
  G__WINDEFfp=fp;

  if(G__isDLL)
    fprintf(fp,"LIBRARY           \"%s\"\n",G__PROJNAME.data());
  else
    fprintf(fp,"NAME              \"%s\" WINDOWAPI\n",G__PROJNAME.data());
  fprintf(fp,"\n");
#if defined(G__OLDIMPLEMENTATION1971) || !defined(G__VISUAL)
  fprintf(fp,"DESCRIPTION       '%s'\n",G__PROJNAME.data());
  fprintf(fp,"\n");
#endif
#if !defined(G__VISUAL) && !defined(G__CYGWIN)
  fprintf(fp,"EXETYPE           NT\n");
  fprintf(fp,"\n");
  if(G__isDLL)
    fprintf(fp,"SUBSYSTEM        WINDOWS\n");
  else
    fprintf(fp,"SUBSYSTEM   CONSOLE\n");
  fprintf(fp,"\n");
  fprintf(fp,"STUB              'WINSTUB.EXE'\n");
  fprintf(fp,"\n");
#endif        /* G__VISUAL */
  fprintf(fp,"VERSION           1.0\n");
  fprintf(fp,"\n");
#if defined(G__OLDIMPLEMENTATION1971) || !defined(G__VISUAL)
  fprintf(fp,"CODE               EXECUTE READ\n");
  fprintf(fp,"\n");
  fprintf(fp,"DATA               READ WRITE\n");
  fprintf(fp,"\n");
#endif
  fprintf(fp,"HEAPSIZE  1048576,4096\n");
  fprintf(fp,"\n");
#ifndef G__VISUAL
  fprintf(fp,"IMPORTS\n");
  fprintf(fp,"        _G__main=%s.G__main\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__setothermain=%s.G__setothermain\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__getnumbaseclass=%s.G__getnumbaseclass\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__setnewtype=%s.G__setnewtype\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__setnewtypeindex=%s.G__setnewtypeindex\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__resetplocal=%s.G__resetplocal\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__getgvp=%s.G__getgvp\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__resetglobalenv=%s.G__resetglobalenv\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__lastifuncposition=%s.G__lastifuncposition\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__resetifuncposition=%s.G__resetifuncposition\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__setnull=%s.G__setnull\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__getstructoffset=%s.G__getstructoffset\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__getaryconstruct=%s.G__getaryconstruct\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__gettempbufpointer=%s.G__gettempbufpointer\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__setsizep2memfunc=%s.G__setsizep2memfunc\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__getsizep2memfunc=%s.G__getsizep2memfunc\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__get_linked_tagnum=%s.G__get_linked_tagnum\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__tagtable_setup=%s.G__tagtable_setup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__search_tagname=%s.G__search_tagname\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__search_typename=%s.G__search_typename\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__defined_typename=%s.G__defined_typename\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__tag_memvar_setup=%s.G__tag_memvar_setup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__memvar_setup=%s.G__memvar_setup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__tag_memvar_reset=%s.G__tag_memvar_reset\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__tag_memfunc_setup=%s.G__tag_memfunc_setup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__memfunc_setup=%s.G__memfunc_setup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__memfunc_next=%s.G__memfunc_next\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__memfunc_para_setup=%s.G__memfunc_para_setup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__tag_memfunc_reset=%s.G__tag_memfunc_reset\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__letint=%s.G__letint\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__letdouble=%s.G__letdouble\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__store_tempobject=%s.G__store_tempobject\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__inheritance_setup=%s.G__inheritance_setup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__add_compiledheader=%s.G__add_compiledheader\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__add_ipath=%s.G__add_ipath\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__add_macro=%s.G__add_macro\n",G__CINTLIBNAME.data());
  fprintf(fp
          ,"        _G__check_setup_version=%s.G__check_setup_version\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__int=%s.G__int\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__double=%s.G__double\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__calc=%s.G__calc\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__loadfile=%s.G__loadfile\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__unloadfile=%s.G__unloadfile\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__init_cint=%s.G__init_cint\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__scratch_all=%s.G__scratch_all\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__setdouble=%s.G__setdouble\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__setint=%s.G__setint\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__stubstoreenv=%s.G__stubstoreenv\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__stubrestoreenv=%s.G__stubrestoreenv\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__getstream=%s.G__getstream\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__type2string=%s.G__type2string\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__alloc_tempobject_val=%s.G__alloc_tempobject_val\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__set_p2fsetup=%s.G__set_p2fsetup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__free_p2fsetup=%s.G__free_p2fsetup\n",G__CINTLIBNAME.data());
  fprintf(fp,"        _G__search_typename2=%s.G__search_typename2\n",G__CINTLIBNAME.data());
  fprintf(fp,"\n");
#endif /* G__VISUAL */
  fprintf(fp,"EXPORTS\n");
  if(G__CPPLINK==G__globalcomp) {
    fprintf(fp,"        G__cpp_dllrev%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__set_cpp_environment%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__cpp_setup_tagtable%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__cpp_setup_inheritance%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__cpp_setup_typetable%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__cpp_setup_memvar%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__cpp_setup_memfunc%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__cpp_setup_global%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__cpp_setup_func%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__cpp_setup%s @%d\n",G__DLLID,++G__nexports);
    if(G__multithreadlibcint)
      fprintf(fp,"        G__SetCppCintApiPointers @%d\n",++G__nexports);
  }
  else {
    fprintf(fp,"        G__c_dllrev%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__set_c_environment%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__c_setup_tagtable%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__c_setup_typetable%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__c_setup_memvar%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__c_setup_global%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__c_setup_func%s @%d\n",G__DLLID,++G__nexports);
    fprintf(fp,"        G__c_setup%s @%d\n",G__DLLID,++G__nexports);
    if(G__multithreadlibcint)
      fprintf(fp,"        G__SetCCintApiPointers @%d\n",++G__nexports);
  }
}
#endif /* G__GENWINDEF */

/**************************************************************************
* G__set_globalcomp()
*
*
**************************************************************************/
void G__set_globalcomp(const char *mode,const char *linkfilename,const char *dllid)
{
  FILE *fp;
  G__FastAllocString buf(G__LONGLINE);
  G__FastAllocString linkfilepref(linkfilename);
  G__FastAllocString linkfilepostf(20);
  char *p;

  p = strrchr(linkfilepref,'/'); /* ../aaa/bbb/ccc.cxx */
#ifdef G__WIN32
  if (!p) p = strrchr(linkfilepref,'\\'); /* in case of Windows pathname */
#endif
  if (!p) p = linkfilepref;      /*  /ccc.cxx */
  p = strrchr (p, '.');          /*  .cxx     */
  if(p) {
    linkfilepostf = p+1;
    *p = '\0';
  }
  else {
    linkfilepostf = "C";
  }

  G__globalcomp = atoi(mode); /* this is redundant */
  if(abs(G__globalcomp)>=10) {
     G__default_link = abs(G__globalcomp)%10;
     G__globalcomp /= 10;
  }
  G__store_globalcomp=G__globalcomp;

  G__strlcpy(G__DLLID,G__map_cpp_name(dllid),sizeof(G__DLLID));

    if(0==strncmp(linkfilename,"G__cpp_",7))
      G__strlcpy(G__NEWID,G__map_cpp_name(linkfilename+7),sizeof(G__NEWID));
    else if(0==strncmp(linkfilename,"G__",3))
      G__strlcpy(G__NEWID,G__map_cpp_name(linkfilename+3),sizeof(G__NEWID));
    else
      G__strlcpy(G__NEWID,G__map_cpp_name(linkfilename),sizeof(G__NEWID));

  switch(G__globalcomp) {
  case G__CPPLINK:
    buf = linkfilepref;
    buf += ".h";
    G__CPPLINK_H = (char*)malloc(strlen(buf)+1);
    strcpy(G__CPPLINK_H,buf); // Okay, we allocated the right size

    buf.Format("%s.%s",linkfilepref(),linkfilepostf());
    G__CPPLINK_C = (char*)malloc(strlen(buf)+1);
    strcpy(G__CPPLINK_C,buf); // Okay, we allocated the right size

#ifdef G__GENWINDEF
    if (G__PROJNAME[0])
      buf.Format("%s.def",G__PROJNAME.data());
    else if (G__DLLID[0])
      buf.Format("%s.def",G__DLLID);
    else
      buf.Format("%s.def","G__lib");
    G__WINDEF = (char*)malloc(strlen(buf)+1);
    strcpy(G__WINDEF,buf); // Okay, we allocated the right size
    G__write_windef_header();
#endif

    // 10-07-07
    // if G__dicttype==kCompleteDictionary we want to generate the ShowMembers only
    // but there is some kind of problem with globals and we still
    // need to execute this function
    if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary){
      fp = fopen(G__CPPLINK_C,"w");
      if(!fp) G__fileerror(G__CPPLINK_C);
      fprintf(fp,"/********************************************************\n");
      fprintf(fp,"* %s\n",G__CPPLINK_C);
      fprintf(fp,"* CAUTION: DON'T CHANGE THIS FILE. THIS FILE IS AUTOMATICALLY GENERATED\n");
      fprintf(fp,"*          FROM HEADER FILES LISTED IN G__setup_cpp_environmentXXX().\n");
      fprintf(fp,"*          CHANGE THOSE HEADER FILES AND REGENERATE THIS FILE.\n");
      fprintf(fp,"********************************************************/\n");
      fprintf(fp,"#include \"%s\" //newlink 3678 \n",G__CPPLINK_H); // THIS COMMENT IS IMPORTANT - rootcint triggers on it!

      fprintf(fp,"\n");
      fprintf(fp,"#ifdef G__MEMTEST\n");
      fprintf(fp,"#undef malloc\n");
      fprintf(fp,"#undef free\n");
      fprintf(fp,"#endif\n");
      fprintf(fp,"\n");

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
      fprintf(fp,"#if defined(__GNUC__) && __GNUC__ >= 4 && ((__GNUC_MINOR__ == 2 && __GNUC_PATCHLEVEL__ >= 1) || (__GNUC_MINOR__ >= 3))\n");
      fprintf(fp,"#pragma GCC diagnostic ignored \"-Wstrict-aliasing\"\n");
      fprintf(fp,"#endif\n");
      fprintf(fp,"\n");
#endif

#ifdef __INTEL_COMPILER
      fprintf(fp,"#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1100)\n");
      fprintf(fp,"# pragma warning (disable 21)\n");
      fprintf(fp,"# pragma warning (disable 191)\n");
      fprintf(fp,"#endif\n");
      fprintf(fp,"\n");
      
#endif

      if(G__dicttype!=kFunctionSymbols)
         fprintf(fp,"extern \"C\" void G__cpp_reset_tagtable%s();\n",G__DLLID);

      fprintf(fp,"\nextern \"C\" void G__set_cpp_environment%s() {\n",G__DLLID);
      fclose(fp);
    }
    break;
  case G__CLINK:
     buf.Format("%s.h",linkfilepref());
    G__CLINK_H = (char*)malloc(strlen(buf)+1);
    strcpy(G__CLINK_H,buf); // Okay, we allocated the right size

    buf.Format("%s.c",linkfilepref());
    G__CLINK_C = (char*)malloc(strlen(buf)+1);
    strcpy(G__CLINK_C,buf); // Okay, we allocated the right size

#ifdef G__GENWINDEF
    buf.Format("%s.def",G__PROJNAME.data());
    G__WINDEF = (char*)malloc(strlen(buf)+1);
    strcpy(G__WINDEF,buf); // Okay, we allocated the right size
    G__write_windef_header();
#endif

    // 10-07-07
    if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary){
      fp = fopen(G__CLINK_C,"w");
      if(!fp) G__fileerror(G__CLINK_C);
      fprintf(fp,"/********************************************************\n");
      fprintf(fp,"* %s\n",G__CLINK_C);
      fprintf(fp,"********************************************************/\n");
      fprintf(fp,"#include \"%s\"\n",G__CLINK_H);

      if(G__dicttype!=kFunctionSymbols)
        fprintf(fp,"void G__c_reset_tagtable%s();\n",G__DLLID);

      fprintf(fp,"void G__set_c_environment%s() {\n",G__DLLID);
      fclose(fp);
    }
    break;
  case R__CPPLINK:
    buf.Format("%s.h",linkfilepref());
    G__CPPLINK_H = (char*)malloc(strlen(buf)+1);
    strcpy(G__CPPLINK_H,buf); // Okay, we allocated the right size

    buf.Format("%s.%s",linkfilepref(),linkfilepostf());
    G__CPPLINK_C = (char*)malloc(strlen(buf)+1);
    strcpy(G__CPPLINK_C,buf); // Okay, we allocated the right size

#ifdef G__GENWINDEF
    if (G__PROJNAME[0])
      buf.Format("%s.def",G__PROJNAME.data());
    else if (G__DLLID[0])
      buf.Format("%s.def",G__DLLID);
    else
      buf.Format("%s.def","G__lib");
    G__WINDEF = (char*)malloc(strlen(buf)+1);
    strcpy(G__WINDEF,buf); // Okay, we allocated the right size
    G__write_windef_header();
#endif

    // 10-07-07
    if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary){
      fp = fopen(G__CPPLINK_C,"w");
      if(!fp) G__fileerror(G__CPPLINK_C);
      fprintf(fp,"/********************************************************\n");
      fprintf(fp,"* %s\n",G__CPPLINK_C);
      fprintf(fp,"* CAUTION: DON'T CHANGE THIS FILE. THIS FILE IS AUTOMATICALLY GENERATED\n");
      fprintf(fp,"*          FROM HEADER FILES LISTED IN G__setup_cpp_environmentXXX().\n");
      fprintf(fp,"*          CHANGE THOSE HEADER FILES AND REGENERATE THIS FILE.\n");
      fprintf(fp,"********************************************************/\n");
      fprintf(fp,"#include \"%s\"\n",G__CPPLINK_H);

      fprintf(fp,"\n");
      fclose(fp);
    }
    break;
  }
}

/**************************************************************************
* G__gen_headermessage()
*
**************************************************************************/
static void G__gen_headermessage(FILE *fp,char *fname)
{
  fprintf(fp,"/********************************************************************\n");
  fprintf(fp,"* %s\n",fname);
  fprintf(fp,"* CAUTION: DON'T CHANGE THIS FILE. THIS FILE IS AUTOMATICALLY GENERATED\n");
  fprintf(fp,"*          FROM HEADER FILES LISTED IN G__setup_cpp_environmentXXX().\n");
  fprintf(fp,"*          CHANGE THOSE HEADER FILES AND REGENERATE THIS FILE.\n");
  fprintf(fp,"********************************************************************/\n");
  fprintf(fp,"#ifdef __CINT__\n");
  fprintf(fp,"#error %s/C is only for compilation. Abort cint.\n"
          ,fname);
  fprintf(fp,"#endif\n");
}

/**************************************************************************
* G__gen_linksystem()
*
**************************************************************************/
int G__gen_linksystem(const char *headerfile)
{
  FILE *fp;

  /* if(G__autoload_stdheader) return(0); */

  switch(G__globalcomp) {
  case G__CPPLINK: /* C++ link */
    fp = fopen(G__CPPLINK_C,"a");
    break;
  case G__CLINK:   /* C link */
    fp = fopen(G__CLINK_C,"a");
    break;
  default:
    return(0);
  }
  fprintf(fp,"  G__add_compiledheader(\"<%s\");\n",headerfile);
  fclose(fp);

  return(0);
}

/**************************************************************************
* G__gen_cppheader()
*
*
**************************************************************************/
void G__gen_cppheader(char *headerfilein)
{
   FILE *fp;
   static char hdrpost[10]="";
   G__FastAllocString headerfile(G__ONELINE);
   char* p;
   
   switch(G__globalcomp) {
      case G__CPPLINK: /* C++ link */
      case G__CLINK:   /* C link */
      case R__CPPLINK: /* C++ link (reflex) */
         break;
      default:
         return;
   }
   
   if(headerfilein) {
      /*************************************************************
       * if header file is already created
       *************************************************************/
      
      headerfile = headerfilein;
      size_t headerfilelen = strlen(headerfile);
      /*************************************************************
       * if preprocessed file xxx.i is given rename as xxx.h
       *************************************************************/
      if(headerfilelen>2 &&
         (strcmp(".i",headerfile+headerfilelen-2)==0 ||
          strcmp(".I",headerfile+headerfilelen-2)==0)) {
            if('\0'==hdrpost[0]) {
               switch(G__globalcomp) {
                  case G__CPPLINK: /* C++ link */
                     G__strlcpy(hdrpost,G__getmakeinfo1("CPPHDRPOST"),sizeof(hdrpost));
                     break;
                  case R__CPPLINK:
                     break;
                  case G__CLINK: /* C link */
                     G__strlcpy(hdrpost,G__getmakeinfo1("CHDRPOST"),sizeof(hdrpost));
                     break;
               }
            }
            headerfile.Replace(headerfilelen-2,hdrpost);
         }
      
      /* backslash escape sequence */
      p=strchr(headerfile,'\\');
      if(p) {
         G__FastAllocString temp2(G__ONELINE);
         int i=0,j=0;
         while(headerfile[i]) {
            switch(headerfile[i]) {
               case '\\':
                  temp2.Set(j++, headerfile[i]);
                  temp2.Set(j++, headerfile[i++]);
                  break;
               default:
                  temp2.Set(j++, headerfile[i++]);
                  break;
            }
         }
         temp2.Set(j, 0);
         headerfile.Swap(temp2);
      }
      
#ifdef G__ROOT
      /* if (!strstr(headerfile,"LinkDef.h")&&!strstr(headerfile,"Linkdef.h") &&
       !strstr(headerfile,"linkdef.h")) { */
      if (!((strstr(headerfile,"LinkDef") || strstr(headerfile,"Linkdef") ||
             strstr(headerfile,"linkdef")) && strstr(headerfile,".h"))) {
#endif
         /* if(strstr(headerfile,".h")||strstr(headerfile,".H")) { */
         switch(G__globalcomp) {
            case G__CPPLINK:
               fp = fopen(G__CPPLINK_H,"a");
               if(!fp) G__fileerror(G__CPPLINK_H);
               fprintf(fp,"#include \"%s\"\n",headerfile());
               fclose(fp);
               
               // 10-07-07
               if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary) {
                  fp = fopen(G__CPPLINK_C,"a");
                  if(!fp) G__fileerror(G__CPPLINK_C);
                  fprintf(fp,"  G__add_compiledheader(\"%s\");\n",headerfile());
                  fclose(fp);
               }
               break;
            case G__CLINK:
               fp = fopen(G__CLINK_H,"a");
               if(!fp) G__fileerror(G__CLINK_H);
               fprintf(fp,"#include \"%s\"\n",headerfile());
               fclose(fp);
               
               // 10-07-07
               if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary) {
                  fp = fopen(G__CLINK_C,"a");
                  if(!fp) G__fileerror(G__CLINK_C);
                  fprintf(fp,"  G__add_compiledheader(\"%s\");\n",headerfile());
                  fclose(fp);
               }
               break;
            case R__CPPLINK:
               fp = fopen(G__CPPLINK_H,"a");
               if(!fp) G__fileerror(G__CPPLINK_H);
               fprintf(fp,"#include \"%s\"\n",headerfile());
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
      switch(G__globalcomp) {
         case G__CPPLINK:
            fp = fopen(G__CPPLINK_H,"w");
            if(!fp) G__fileerror(G__CPPLINK_H);
            G__gen_headermessage(fp,G__CPPLINK_H);
            G__cpplink_header(fp);
            fclose(fp);
            break;
         case G__CLINK:
            fp = fopen(G__CLINK_H,"w");
            if(!fp) G__fileerror(G__CLINK_H);
            G__gen_headermessage(fp,G__CLINK_H);
            G__clink_header(fp);
            fclose(fp);
            break;
         case R__CPPLINK:
            fp = fopen(G__CPPLINK_H,"w");
            if(!fp) G__fileerror(G__CPPLINK_H);
            G__gen_headermessage(fp,G__CPPLINK_H);
            fclose(fp);
            break;
      }
   }
}
   
/**************************************************************************
* G__add_compiledheader()
*
**************************************************************************/
void G__add_compiledheader(const char *headerfile)
{
  if(headerfile && headerfile[0]=='<' && G__autoload_stdheader ) {
    int store_tagnum = G__tagnum;
    int store_def_tagnum = G__def_tagnum;
    int store_tagdefining = G__tagdefining;
    int store_def_struct_member = G__def_struct_member;
#ifdef G__OLDIMPLEMENTATION1284_YET
    if(G__def_struct_member && 'n'==G__struct.type[G__tagdefining]) {
      G__def_struct_member = 1;
    }
    else {
      G__tagnum = -1;
      G__def_tagnum = -1;
      G__tagdefining = -1;
      G__def_struct_member = 0;
    }
#else
    G__tagnum = -1;
    G__def_tagnum = -1;
    G__tagdefining = -1;
    G__def_struct_member = 0;
#endif
    G__loadfile(headerfile+1);
    G__tagnum = store_tagnum;
    G__def_tagnum = store_def_tagnum;
    G__tagdefining = store_tagdefining;
    G__def_struct_member = store_def_struct_member;
  }
}

/**************************************************************************
* G__add_macro()
*
* Macros starting with '!' are assumed to be system level macros
* that will not be passed to an external preprocessor.
**************************************************************************/
void G__add_macro(const char *macroin)
{
   G__FastAllocString temp(G__LONGLINE);

   int store_tagnum = G__tagnum;
   int store_def_tagnum = G__def_tagnum;
   int store_tagdefining = G__tagdefining;
   int store_def_struct_member = G__def_struct_member;
   int store_var_type = G__var_type;
   struct G__var_array *store_p_local = G__p_local;
   G__tagnum = -1;
   G__def_tagnum = -1;
   G__tagdefining = -1;
   G__def_struct_member = 0;
   G__var_type = 'p';
   G__p_local = (struct G__var_array*)0;
   
   const char* macroname = macroin;
   if (macroname[0] == '!')
      ++macroname;
   G__FastAllocString macro(macroname);
   G__definemacro=1;
   {
      char *p = strchr(macro,'=');
      if( p ) {
         if(G__cpp && '"'==*(p+1)) {
            G__add_quotation(p+1,temp);
            macro.Replace(p+1-macro.data(),temp+1);
            macro[strlen(macro)-1]=0;
         }
         else {
            temp = macro;
         }
      }
      else {
         temp = macro;
         temp += "=1";
      }
   }
   G__getexpr(temp);
   G__definemacro=0;
   
   if (macroin[0] == '!')
      goto end_add_macro;
   
   temp.Format("\"-D%s\" ", macro());

   /*   " -Dxxx -Dyyy -Dzzz"
    *       p  ^              */
   if(strstr(G__macros,temp)) goto end_add_macro;
   temp = G__macros;
   if(strlen(temp)+strlen(macro)+5 > sizeof(G__macros)) {
      if(G__dispmsg>=G__DISPWARN) {
         G__fprinterr(G__serr,"Warning: can not add any more macros in the list\n");
         G__printlinenum();
      }
   }
   else {
      snprintf(G__macros,sizeof(G__macros),"%s\"-D%s\" ",temp(),macro());
   }
   
   switch(G__globalcomp) {
      case G__CPPLINK: {
         FILE *fp=fopen(G__CPPLINK_C,"a");
         if(!fp) G__fileerror(G__CPPLINK_C);
         fprintf(fp,"  G__add_macro(\"%s\");\n",macro());
         fclose(fp);
         break;
      }
      case G__CLINK: {
         FILE *fp=fopen(G__CLINK_C,"a");
         if(!fp) G__fileerror(G__CLINK_C);
         fprintf(fp,"  G__add_macro(\"%s\");\n",macro());
         fclose(fp);
         break;
      }
   }
end_add_macro:
   G__tagnum = store_tagnum;
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__def_struct_member = store_def_struct_member;
   G__var_type = store_var_type;
   G__p_local = store_p_local;
}
   
/**************************************************************************
* G__add_ipath()
*
**************************************************************************/
void G__add_ipath(const char *path)
{
  struct G__includepath *ipath;
  G__FastAllocString temp(G__ONELINE);
  FILE *fp;
  char *p;
  char *store_allincludepath;

  /* strip double quotes if exist */
  if('"'==path[0]) {
    temp = path+1;
    if('"'==temp[strlen(temp)-1]) temp[strlen(temp)-1]='\0';
  }
  else {
    temp = path;
  }

  /* to the end of list */
  ipath = &G__ipathentry;
  while(ipath->next) {
    if(ipath->pathname&&strcmp(ipath->pathname,temp)==0) return;
    ipath = ipath->next;
  }

  /* G__allincludepath will be given to real preprocessor */
  if(!G__allincludepath) {
    G__allincludepath = (char*)malloc(1);
    G__allincludepath[0] = '\0';
  }
  size_t allincludepath_oldlen = strlen(G__allincludepath);
  size_t allincludepath_newlen = allincludepath_oldlen+strlen(temp)+6;
  store_allincludepath = (char*)realloc((void*)G__allincludepath,allincludepath_newlen);
  if(store_allincludepath) {
    int i=0,flag=0;
    while(temp[i]) if(isspace(temp[i++])) flag=1;
    G__allincludepath = store_allincludepath;
    size_t increase = allincludepath_newlen - allincludepath_oldlen;
    if(flag)
       G__snprintf(G__allincludepath+allincludepath_oldlen,increase,"-I\"%s\" ",temp());
    else
       G__snprintf(G__allincludepath+allincludepath_oldlen,increase,"-I%s ",temp());
  }
  else {
    G__genericerror("Internal error: memory allocation failed for includepath buffer");
  }


  /* copy the path name */
  size_t templen = (size_t)(strlen(temp)+1);
  ipath->pathname = (char *)malloc(templen);
  G__strlcpy(ipath->pathname,temp,templen); // Okay, we allocated the right size

  /* allocate next entry */
  ipath->next=(struct G__includepath *)malloc(sizeof(struct G__includepath));
  ipath->next->next=(struct G__includepath *)NULL;
  ipath->next->pathname = (char *)NULL;

  /* backslash escape sequence */
  p=strchr(temp,'\\');
  if(p) {
    G__FastAllocString temp2(G__ONELINE);
    int i=0,j=0;
    while(temp[i]) {
      switch(temp[i]) {
      case '\\':
         temp2.Set(j++, temp[i]);
         temp2.Set(j++, temp[i++]);
        break;
      default:
         temp2.Set(j++, temp[i++]);
        break;
      }
    }
    temp2.Set(j, 0);
    temp.Swap(temp2);
  }

  /* output include path information to interface routine */
  switch(G__globalcomp) {
  case G__CPPLINK:
    fp=fopen(G__CPPLINK_C,"a");
    if(!fp) G__fileerror(G__CPPLINK_C);
    fprintf(fp,"  G__add_ipath(\"%s\");\n",temp());
    fclose(fp);
    break;
  case G__CLINK:
    fp=fopen(G__CLINK_C,"a");
    if(!fp) G__fileerror(G__CLINK_C);
    fprintf(fp,"  G__add_ipath(\"%s\");\n",temp());
    fclose(fp);
    break;
  }
}


/**************************************************************************
* G__delete_ipath()
*
**************************************************************************/
int G__delete_ipath(const char *path)
{
  struct G__includepath *ipath;
  struct G__includepath *previpath;
  G__FastAllocString temp(G__ONELINE);
  G__FastAllocString temp2(G__ONELINE);
  int i=0,flag=0;
  char *p;

  /* strip double quotes if exist */
  if('"'==path[0]) {
    temp = path+1;
    if('"'==temp[strlen(temp)-1]) temp[strlen(temp)-1]='\0';
  }
  else {
    temp = path;
  }

  /* to the end of list */
  ipath = &G__ipathentry;
  previpath = (struct G__includepath*)NULL;
  while(ipath->next) {
    if(ipath->pathname&&strcmp(ipath->pathname,temp)==0) {
      /* delete this entry */
      free((void*)ipath->pathname);
      ipath->pathname=(char*)NULL;
      if(previpath) {
        previpath->next = ipath->next;
        free((void*)ipath);
      }
      else if(ipath->next) {
        G__ipathentry.pathname = (char*)calloc(1,1);
      }
      else {
        free((void*)G__ipathentry.pathname);
        G__ipathentry.pathname=(char*)NULL;
      }
      break;
    }
    previpath=ipath;
    ipath = ipath->next;
  }

  /* G__allincludepath will be given to real preprocessor */
  if(!G__allincludepath) return(0);
  i=0;
  while(temp[i]) if(isspace(temp[i++])) flag=1;
  if(flag) temp2.Format("-I\"%s\" ",temp());
  else     temp2.Format("-I%s ",temp());

  p = strstr(G__allincludepath,temp2);
  if(p) {
    char *p2 = p+strlen(temp2);
    while(*p2) *p++ = *p2++;
    *p = *p2;
    return(1);
  }

  return(0);
}




/**************************************************************************
**************************************************************************
* Generate C++ function access entry function
**************************************************************************
**************************************************************************/


/**************************************************************************
* G__isnonpublicnew()
*
**************************************************************************/
int G__isnonpublicnew(int tagnum)
{
  int i;
  int hash;
  const char *namenew = "operator new";
  struct G__ifunc_table_internal *ifunc;

  G__hash(namenew,hash,i);
  ifunc = G__struct.memfunc[tagnum];

  while(ifunc) {
    for(i=0;i<ifunc->allifunc;i++) {
      if(hash==ifunc->hash[i] && strcmp(ifunc->funcname[i],namenew)==0) {
         if(G__PUBLIC!=ifunc->access[i]) return(1);
       }
    }
    ifunc = ifunc->next;
  }
  return(0);
}

/**************************************************************************
 * G__cppif_change_globalcomp()
 *
 * 09-10-07
 * We need this silly method just to change the value of a field that
 * was changed in the stub generation (but since we got rid of the
 * stubs... the state isn't changed)
 **************************************************************************/
void G__cppif_change_globalcomp()
{
#ifndef G__SMALLOBJECT
  int i,j;
  struct G__ifunc_table_internal *ifunc;

  for(i=0;i<G__struct.alltag;i++) {
    if(
      (G__CPPLINK==G__struct.globalcomp[i]||
       G__CLINK==G__struct.globalcomp[i]
       || G__ONLYMETHODLINK==G__struct.globalcomp[i]
        ) &&
      (-1==(int)G__struct.parent_tagnum[i]
       || G__nestedclass
        )
      &&
      -1!=G__struct.line_number[i]&&G__struct.hash[i]&&
      '$'!=G__struct.name[i][0] && 'e'!=G__struct.type[i]) {
      ifunc = G__struct.memfunc[i];

      while(ifunc) {
        for(j=0;j<ifunc->allifunc;j++) {
          if(G__PUBLIC==ifunc->access[j]
             || (G__PROTECTED==ifunc->access[j] &&
                 (G__PROTECTEDACCESS&G__struct.protectedaccess[i]))
             || (G__PRIVATEACCESS&G__struct.protectedaccess[i])
            ) {
            if(G__ONLYMETHODLINK==G__struct.globalcomp[i]&&
               G__METHODLINK!=ifunc->globalcomp[j]) continue;
            if(0==ifunc->hash[j]) continue;

            /* Promote link-off typedef to link-on if used in function */
            // 09-10-07
            // This is just wrong... wrong
            // A function that 'just' prints a value should not change its internal settings...
            // As a result this fields have to be change by hand at a non-specified part of
            // the file when this method is not called...
            //
            // look at the function G__cppif_returntype: This was changed there in the old scheme
            //
            // Can we really change it here?
            if ((ifunc->p_typetable[j] != -1) &&
                (G__newtype.globalcomp[ifunc->p_typetable[j]] == G__NOLINK) &&
                (G__newtype.iscpplink[ifunc->p_typetable[j]] == G__NOLINK)) {
              G__newtype.globalcomp[ifunc->p_typetable[j]] = G__globalcomp;
            }
          }
        }
        ifunc=ifunc->next;
      } /* while(ifunc) */
    } /* if(globalcomp) */
  } /* for(i) */
#endif // G__SMALLOBJECT
}

/**************************************************************************
*  G__if_ary_union_constructor()
*
**************************************************************************/
void G__if_ary_union_constructor(FILE *fp, int ifn, G__ifunc_table_internal *ifunc)
{
  int k, m;
  char* p;

  m = ifunc->para_nu[ifn];

  for (k = 0; k < m; ++k) {
    if (ifunc->param[ifn][k]->name) {
      p = strchr(ifunc->param[ifn][k]->name, '[');
      if (p) {
        fprintf(fp, "  struct G__aRyp%d { %s a[1]%s; }* G__Ap%d = (struct G__aRyp%d*) 0x64;\n", k, G__type2string(ifunc->param[ifn][k]->type, ifunc->param[ifn][k]->p_tagtable, ifunc->param[ifn][k]->p_typetable, 0 , 0), p + 2, k, k);
      }
    }
  }
}


/**************************************************************************
 *  G__is_tagnum_safe(int i)
 *
 * 08-02-2008
 * We have some problems to create the tmp dictionaries of certain classes.
 * Here we create an artificial condition telling us if the passed tagnum
 * is one of this classes...
 * This should be avoided, what we need is a more precise condition for
 * the cases where we fail creating the dictionary
 **************************************************************************/
bool G__is_tagnum_safe(int i)
{
#ifdef G__NOSTUBS
  return (strncmp(G__fulltagname(i,0),"string", strlen("string"))!=0 &&
          strncmp(G__fulltagname(i,0),"vector", strlen("vector"))!=0 &&
          strncmp(G__fulltagname(i,0),"list", strlen("list"))!=0 &&
          strncmp(G__fulltagname(i,0),"deque", strlen("deque"))!=0 &&
          strncmp(G__fulltagname(i,0),"set", strlen("set"))!=0 &&
          strncmp(G__fulltagname(i,0),"multiset", strlen("multiset"))!=0 &&
          strncmp(G__fulltagname(i,0),"allocator", strlen("allocator"))!=0 &&
          strncmp(G__fulltagname(i,0),"map", strlen("map"))!=0 &&
          strncmp(G__fulltagname(i,0),"multimap", strlen("multimap"))!=0 &&
          strncmp(G__fulltagname(i,0),"complex", strlen("complex"))!=0 );
#else
  (void) i;
  return true;
#endif
}

/**************************************************************************
 *  G__write_preface()
 *
 * 21-01-2008
 * When creting the second tmp file, we want to end up with all the symbols
 * in the file. For this, we create a pointer to a function when we have 
 * non-virtual functions a function call when we do have virtual functions.
 * For the latter, it's easier if we have just one object and then we call
 * all the functions with it, so what we need is a big function doing that
 * for every class.
 *
 * This preface will try to create the declaration of this function
 * containing all the function calls. It will also include a dummy pointer
 * used by all the others. (try to reduce code replication)
 **************************************************************************/
void G__write_preface(FILE *fp, struct G__ifunc_table_internal *ifunc, int i)
{
  // Write the prototype of the function
  // Let's keep it simple G__function_class
  const char *dllid;
  if(G__DLLID[0]) dllid=G__DLLID;
  else if(G__PROJNAME[0]) dllid=G__PROJNAME;
  else dllid="";

  (void) ifunc;
  fprintf(fp, "void G__function_%d_%s() \n{\n", i, G__map_cpp_name(dllid));
}


/**************************************************************************
 *  G__write_dummy_ptr()
 *
 * 23-01-2008
 *
 * This should part of the preface, the problem is that sometimes we don't
 * need the dummy ptr (when there are no function calls)
 *
 **************************************************************************/
void G__write_dummy_ptr(FILE *fp, struct G__ifunc_table_internal *ifunc, int i)
{
  // Now print the dummy pointer we will use..
  // be sure to rem the name
  (void) ifunc;
  if(G__struct.type[i]!='n'){  // This is only for classes (we can't have an object of a namespace)
     fprintf(fp,"  %s* ptr_%d=0;\n",G__fulltagname(i,0), i);
  }
}

/**************************************************************************
 *  G__write_postface()
 *
 * 21-01-2008
 * When creting the second tmp file, we want to end up with all the symbols
 * in the file. For this we create a pointer to a function when we have 
 * non-virtual functions a function call when we do have virtual functions.
 * For the latter, it's easier if we have just one object and then we call
 * all the functions with it, so what we need is a big function doing that
 * for every class.
 *
 * This postface just has to finish what we started in G__write_preface
 * for the moment it should be onlz a parenthesis closing the function
 **************************************************************************/
void G__write_postface(FILE *fp, struct G__ifunc_table_internal *ifunc, int i)
{
  (void) ifunc;
  (void) i;
  fprintf(fp, "}\n");
}

/**************************************************************************
 * G__cppif_geninline
 *
 * 29-05-06
 * Rem we have to deal with inlined functions.
 * And saddly CInt doesnt care about them.
 * Axel said that we would be able to find out
 * what symbols were in a library before generating the
 * wrappers (he said that a dictionary will be divided in
 * two parts).
 * As an inmediate solution, force every single member function
 * to be inlined by declaring a pointer to it
 **************************************************************************/
void G__cppif_geninline(FILE *fp, struct G__ifunc_table_internal *ifunc, int i,int j)
{
  // 06-11-12
  // Since we are now registering the symbols for the second dictionary too...
  // We can try to inline all the functions without symbol
  if( G__NOLINK>ifunc->globalcomp[j] &&  /* with -c-1 option */
      G__PUBLIC==ifunc->access[j] && /* public, this is always true */
      /*0==ifunc->staticalloc[j] &&*/
      ifunc->hash[j] ){//&&
    //!ifunc->mangled_name[j]) { DMS
    if(G__dicttype==kFunctionSymbols) {
      int hash;
      int idx;
      if(i != -1) {
        G__hash(G__fulltagname(i,0),hash,idx);
      }

      // Print the return type
      // we need to convert A::operator T() to A::operator ::T, or
      // the context will be the one of tagnum, i.e. A::T instead of ::T
      if (tolower(ifunc->type[j]) == 'u'
          && !strncmp(ifunc->funcname[j], "operator ", 8)
          && (isalpha(ifunc->funcname[j][9]) || ifunc->funcname[j][9] == '_')) {
        if (!strncmp(ifunc->funcname[j] + 9, "const ", 6))
          fprintf(fp, "const ::%s ", ifunc->funcname[j] + 15);
        else
          fprintf(fp, "::%s ", ifunc->funcname[j] + 9);
      } else
        fprintf(fp, "%s ", G__type2string(ifunc->type[j], ifunc->p_tagtable[j], 
                                          ifunc->p_typetable[j], ifunc->reftype[j],
                                          ifunc->isconst[j]));

      // Static functions are not casted as member-functions
      // but as normal functions
      if(ifunc->staticalloc[j]
         || (i == -1) // global functions
         || G__struct.type[i] == 'n'
         || strcmp(ifunc->funcname[j], "operator new")==0 /*operator new must be treated as a static function*/
         || strcmp(ifunc->funcname[j], "operator new[]")==0) {

        fprintf(fp," (*fmptr_%s)(", G__map_cpp_funcname(ifunc->tagnum, ifunc->funcname[j], j, ifunc->page));
      }
      else {
        fprintf(fp," (%s::*fmptr_%s)(", G__fulltagname(i,0), G__map_cpp_funcname(ifunc->tagnum, ifunc->funcname[j], j, ifunc->page));
      }
      // print the params
      int paran = ifunc->para_nu[j];
      int parai = 0;
      for (parai = 0; parai < paran; ++parai) {
        fprintf(fp, " %s", G__type2string(ifunc->param[j][parai]->type,
                                          ifunc->param[j][parai]->p_tagtable,
                                          ifunc->param[j][parai]->p_typetable,
                                          ifunc->param[j][parai]->reftype,
                                          ifunc->param[j][parai]->isconst));
        if (ifunc->param[j][parai]->name) {
          char *p = strchr(ifunc->param[j][parai]->name, '[');
          if (p) {
            fprintf(fp, " [1]%s",p + 2);
          }
        }
        if(parai < paran-1)
          fprintf(fp, ",");
      }
      if (ifunc->ansi[j]==2)
        fprintf(fp,", ... ");
      fprintf(fp,") ");

      // print the rest of the assign
      if(ifunc->isconst[j] & G__CONSTFUNC)
        fprintf(fp," %s", " const ");
      if(ifunc->isconst[j] & G__FUNCTHROW)
        fprintf(fp," %s", " throw() ");

      if(i == -1) { // global functions
        // we need to convert A::operator T() to A::operator ::T, or
        // the context will be the one of tagnum, i.e. A::T instead of ::T
        if (tolower(ifunc->type[j]) == 'u'
            && !strncmp(ifunc->funcname[j], "operator ", 8)
            && (isalpha(ifunc->funcname[j][9]) || ifunc->funcname[j][9] == '_')) {
          if (!strncmp(ifunc->funcname[j] + 9, "const ", 6))
            fprintf(fp, " = &operator const ::%s;\n", ifunc->funcname[j] + 15);
          else
            fprintf(fp, " = &operator ::%s;\n", ifunc->funcname[j] + 9);
        } else
          fprintf(fp," = &%s; \n", ifunc->funcname[j]);
      }
      else{
        // we need to convert A::operator T() to A::operator ::T, or
        // the context will be the one of tagnum, i.e. A::T instead of ::T
        if (tolower(ifunc->type[j]) == 'u'
            && !strncmp(ifunc->funcname[j], "operator ", 8)
            && (isalpha(ifunc->funcname[j][9]) || ifunc->funcname[j][9] == '_')) {
          if (!strncmp(ifunc->funcname[j] + 9, "const ", 6))
            fprintf(fp, " = &%s::operator const ::%s;\n", G__fulltagname(i,0), ifunc->funcname[j] + 15);
          else
            fprintf(fp, " = &%s::operator ::%s;\n", G__fulltagname(i,0), ifunc->funcname[j] + 9);
        } else
          fprintf(fp," = &%s::%s; \n", G__fulltagname(i,0), ifunc->funcname[j]);
      }
      
      // Don't do this, if we are outside a function scope
      if(i != -1 )
        fprintf(fp," (void)(fmptr_%s);\n", G__map_cpp_funcname(ifunc->tagnum, ifunc->funcname[j], j, ifunc->page));
    }
  }
}

/**************************************************************************
 * G__write_dummy_param
 *
 * This will print a dummy parameterof the type given in formal_param
 * It's required in the second stage of the temporary dictionary.
 * It's used to simulate a function call, which will in turn force
 * the function symbol to be included in the dictionary.
 * All that is done to avoid using all the *.o from the directory.
 *
 * This function is used by G__cpp_methodcall
 **************************************************************************/
void G__write_dummy_param(FILE *fp, G__paramfunc *formal_param)
{
   int ispointer = 0;
   char para_type = formal_param->type;
   
   if(isupper(formal_param->type)){
      ispointer = 1;
   }  
   
   // By Value or By Reference?
   if (!ispointer) {// By Value
      if (formal_param->reftype==1&&(formal_param->p_typetable!=-1||formal_param->p_tagtable!=-1)){
         if(formal_param->p_typetable==-1)
            fprintf(fp,"*(%s*) 0x64",G__fulltagname(formal_param->p_tagtable,0));   
         else  
            fprintf(fp,"*(%s*) 0x64",G__fulltypename(formal_param->p_typetable));   
      }
      else {
         
         if (formal_param->reftype==1||para_type=='u'||para_type=='a')
            fprintf(fp,"*");
         
         fprintf(fp,"(");
         
         // Parameter's type
         switch(para_type) {
               
               // Double = Double Word
            case 'a' : fprintf(fp,"%s",G__fulltypename(formal_param->p_typetable)); 
               break;
               
               // Double = Double Word
            case 'd' : fprintf(fp, "double");
               break;
               
               // Integer = Single Word
            case 'i' :
            {
               if (formal_param->p_tagtable==-1)
                  fprintf(fp, "int");
               else  
                  fprintf(fp, " %s ", G__fulltagname(formal_param->p_tagtable,0));
            }
               break;
               
               // Unsigned Char ????
            case 'b' : fprintf(fp, "unsigned char");
               break;
               
               // Char
            case 'c' : fprintf(fp, "char");
               break;
               
               // Short
            case 's' : fprintf(fp, "short");
               break;
               
               // Unsigned Short
            case 'r' : fprintf(fp, "unsigned short");
               break;
               
               // Unsigned Int
            case 'h' : fprintf(fp, "unsigned int");
               break;
               
               // Long
            case 'l' : fprintf(fp, "long");                 
               break;
               
               // Unsigned Long
            case 'k': fprintf(fp, "unsigned long");
               break;
               
               // Float // Shouldnt it be treated as a double?
            case 'f' : fprintf(fp, "float");
               break;
               
               // Long Long
            case 'n' : fprintf(fp, "long long");
               break;
               
               // unsigned Long Long
            case 'm' : fprintf(fp, "unsigned long long");
               break;
               
               // long double
            case 'q' : fprintf(fp, "long double");
               break;
               
               // bool 
            case 'g' : fprintf(fp, "bool");
               break;
               
            case '1': // Function Pointer
            {
               if(formal_param->p_typetable==-1)
                  fprintf(fp, "void");
               else
                  fprintf(fp,"%s",G__fulltypename(formal_param->p_typetable)); 
            }
               break;
               
               // a class... treat it as a reference
            case 'u' : fprintf(fp,"%s",G__fulltagname(formal_param->p_tagtable,0)); 
               break;
               
            default:
               fprintf(fp, " Unkown: %c", formal_param->type);
               G__fprinterr(G__serr,"Type %c not known yet (methodcall)\n", para_type);
         }
         
         
         if (formal_param->reftype==1||para_type=='u'||para_type=='a')
            fprintf(fp, "*) 0x64");
         else     
            fprintf(fp, ") 0");
         
      }
   }
   else {
      // If this is something like "int*&"
      // we deference it first
      if (formal_param->reftype==1 || (formal_param->p_tagtable==-1 && formal_param->p_typetable!=-1 &&
                                       formal_param->type=='Y')) {
         fprintf(fp,"*");
      }
      fprintf(fp,"(");
      if (formal_param->isconst&G__CONSTVAR)
         fprintf(fp,"const ");
      
      if(formal_param->p_tagtable!=-1) {
         if (formal_param->reftype==1)
            fprintf(fp,"%s", G__type2string(formal_param->type,formal_param->p_tagtable,formal_param->p_typetable,0,formal_param->isconst&G__CONSTVAR));
         else
            fprintf(fp,"%s*", G__fulltagname(formal_param->p_tagtable,0));
      }
      else {  
         switch(formal_param->type) {
               
               // Double* 
            case 'D': fprintf(fp, "double");
               break;
               
               // Unsigned Integer*
            case 'H': fprintf(fp, "unsigned int");
               break;
               
               // Integer*
            case 'I': fprintf(fp, "int");
               break;
               
               // *UChar
            case 'B': fprintf(fp,"unsigned char"); 
               break;
               
               // *Char
            case 'C': fprintf(fp,"char");
               break;
               
               // FILE
            case 'E': fprintf(fp,"FILE"); 
               break;
               
               // (void*)
            case 'Y':
            {
               if(formal_param->p_typetable==-1)
                  fprintf(fp,"void");
               else
                  fprintf(fp,"%s",G__fulltypename(formal_param->p_typetable));
            }
               break;
               
               // *float
            case 'F': fprintf(fp,"float"); 
               break;
               
               // *long long
            case 'N': fprintf(fp,"long long"); 
               break;
               
               // *long
            case 'L': fprintf(fp,"long"); 
               break;
               
               // *short
            case 'S': fprintf(fp,"short"); 
               break;
               
               // Unsigned Long
            case 'K': fprintf(fp,"unsigned long"); 
               break;
               
               // bool
            case 'G': fprintf(fp,"bool"); 
               break;
               
               // Unsigned short
            case 'R': fprintf(fp,"unsigned short"); 
               break;
               
               // Unsigned Long Long
            case 'M': fprintf(fp,"unsigned long long"); 
               break;
               
               // long double
            case 'Q': fprintf(fp,"long double"); 
               break;
               
            default:
               fprintf(fp, " Unkown: %c", formal_param->type);
               G__fprinterr(G__serr,"Type %c not known yet (G__write_dummy_param)\n",formal_param->type);
         }  
         fprintf(fp,"*");
      }
      // Put the stars and give it a bogus value
      if (formal_param->reftype==1)
         fprintf(fp,"*");
      for(int id=1;id<formal_param->reftype;id++)
         fprintf(fp,"*");
      fprintf(fp,") 0x64");  
   }        
}


/**************************************************************************
 * G__cpp_methodcall
 *
 * It's used to simulate a function call, which will in turn force
 * the function symbol to be included in the dictionary.
 * All that is done to avoid using all the *.o from the directory.
 *
 * This has to be used when we have a virtual function since just having
 * a pointer to it won't generate the symbol name
 **************************************************************************/
void G__cpp_methodcall(FILE *fp, struct G__ifunc_table_internal *ifunc, int i,int j)
{
   // Output the function name.
   
   assert(i>=0);
   
   // 06-11-12
   // Since we are now registering the symbols for the second dictionary too...
   // We can try to inline all the functions without symbol
   if( G__NOLINK>ifunc->globalcomp[j] &&  /* with -c-1 option */
      G__PUBLIC==ifunc->access[j] && /* public, this is always true */
      /*0==ifunc->staticalloc[j] &&*/
      ifunc->hash[j] ){//&&
      //!ifunc->mangled_name[j]) { DMS
      if(G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary) {
         int hash;
         int idx;

         G__hash(G__fulltagname(i,0),hash,idx);
         
         if(G__struct.type[i]!='n'){  // This is only for classes (we can't have an object of a namespace)
            fprintf(fp,"  ptr_%d->", i);
            fprintf(fp,"%s::%s(",G__fulltagname(i,0),ifunc->funcname[j]);
         } 
         else {
            fprintf(fp,"%s::%s(",G__fulltagname(i,0),ifunc->funcname[j]);
         }
         
         int paran = ifunc->para_nu[j];
         int k = 0;
         for (int counter=paran-1; counter>-1; counter--) {
            k = (paran-1) - counter;
            G__paramfunc *formal_param = ifunc->param[j][k];
            
            if (counter!=paran-1)
               fprintf(fp,",");
            
            if (formal_param->name){
               if(strchr(formal_param->name,'[')){
                  fprintf(fp,"G__Ap%d->a",counter);
                  continue;
               }
            }
            
            // Write every dummy parameter for this function
            G__write_dummy_param(fp, formal_param);
         }
         fprintf(fp,");\n");
      }
   }
}  

/**************************************************************************
 *  G__cppif_dummyobj()
 *  
 * We cannont create a pointer to a constructor. 
 * To get the symbol in the .o file we create an object (this should 
 * generate the constructor and destructor of a class).
 * This is only for classes! (we can't have an object of a namespace))
 **************************************************************************/
void G__cppif_dummyobj(FILE *fp, struct G__ifunc_table_internal *ifunc, int i,int j)
{
  static int func_cod = 0;

  if (i!=-1&&!strcmp(ifunc->funcname[j] ,G__struct.name[i]) && G__struct.type[i]!='n') {
    //if(ifunc->para_nu[j]==1&&ifunc->param[j][0]->p_tagtable!=-1&&ifunc->param[j][0]->reftype==1)
    //  return;

    // We cannot create an object which belongs to an abstract class.
    if (ifunc->tagnum==-1 || G__struct.isabstract[ifunc->tagnum]) {
       return;
    } 

    // We can't create a dummy object if its destructor is private...
    // we tried by creating the object in the heap instead of the heap (with new)
    // but it doesn't work. It seems the compiler is smart enough to ignore the
    // object if it isnt used
    if (G__isprivatedestructorifunc(ifunc->tagnum))
      return;

    int paran = ifunc->para_nu[j];
    // our flag for globalfunctions
    int globalfunc = 0;
    // The other important point is that variadic functions take the parameters
    // in the opposite order
    if (ifunc->tagnum < 0)
      globalfunc = 1;

    // if this is a variadic func then pass the parameters
    // in the same order of methods not the one of globals functions
    if(ifunc->ansi[j] == 2)
      globalfunc = 0;

    G__if_ary_union_constructor(fp, 0, ifunc);

    // Print the object class and name
    fprintf(fp, "  %s obj_%s(",G__fulltagname(ifunc->tagnum,0), G__map_cpp_funcname(ifunc->tagnum, ifunc->funcname[j], j, ifunc->page));

    int k = 0;
    for (int counter=paran-1; counter>-1; counter--) {
      int ispointer = 0;
      k = (paran-1) - counter;

      G__paramfunc *formal_param = ifunc->param[j][k];

      if(isupper(formal_param->type)) {
        ispointer = 1;
      }

      if (counter!=paran-1)
        fprintf(fp,",");

      if (formal_param->name) {
        if(strchr(formal_param->name,'[')) {
          fprintf(fp,"G__Ap%d->a",counter);
          continue;
        }
      }
      // Write every dummy parameter for this function
      G__write_dummy_param(fp, formal_param);
    }
    fprintf(fp, ");\n");
    
    // Avoid warnings for non-used objects
    fprintf(fp, "  (void) obj_%s;\n", G__map_cpp_funcname(ifunc->tagnum, ifunc->funcname[j], j, ifunc->page));
  }
  func_cod++;
}


/**************************************************************************
 * G__make_default_ifunc()
 *
 * 25-07-07
 *
 * Create the deafult cons, dests, etc as ifunc entries before writing
 * them to the file
 **************************************************************************/
void G__make_default_ifunc(G__ifunc_table_internal *ifunc_copy)
{
  struct G__ifunc_table_internal *ifunc = ifunc_copy;
  struct G__ifunc_table_internal *ifunc_destructor=0;
  G__FastAllocString funcname(G__MAXNAME*6);
  int hash;
  int j=0,k=0;
  int i = ifunc->tagnum;
  int isnonpublicnew;
  int isconstructor,iscopyconstructor,isdestructor,isassignmentoperator;
  int virtualdtorflag;
  int dtoraccess=G__PUBLIC;

  dtoraccess=G__PUBLIC;

  struct G__ifunc_table_internal *store_p_ifunc = G__p_ifunc;

  if((G__CPPLINK==G__struct.globalcomp[i]
      || G__ONLYMETHODLINK==G__struct.globalcomp[i]
       )&&
     (-1==(int)G__struct.parent_tagnum[i]
      || G__nestedclass
       )
     && -1!=G__struct.line_number[i]&&
     (G__struct.hash[i] || 0==G__struct.name[i][0])
     &&
     '$'!=G__struct.name[i][0] && 'e'!=G__struct.type[i]) {

    ifunc_destructor = G__struct.memfunc[i];
    isconstructor=0;
    iscopyconstructor=0;
    isdestructor=0;
    /* isvirtualdestructor=0; */
    isassignmentoperator=0;
    isnonpublicnew=G__isnonpublicnew(i);
    virtualdtorflag=0;

    while (ifunc) {
      for (j = 0; j < ifunc->allifunc; ++j) {
        if ((ifunc->access[j] == G__PUBLIC) || G__precomp_private
            || G__isprivatectordtorassgn(i, ifunc, j)
            || ((ifunc->access[j] == G__PROTECTED) && (G__struct.protectedaccess[i] & G__PROTECTEDACCESS)) 
            || (G__struct.protectedaccess[i] & G__PRIVATEACCESS)) {
          // public

          if ((G__struct.globalcomp[i] == G__ONLYMETHODLINK) && (ifunc->globalcomp[j] != G__METHODLINK)) {
            // not marked for link, skip it.
            continue;
          }

#ifndef G__OLDIMPLEMENTATION2039
          if (!ifunc->hash[j]) {
            // no hash, skip it
            continue;
          }
#endif

#ifndef G__OLDIMPLEMENTATION1656
          // 08-10-07
          // Here we haven't introduced the default functions...
          // should we skip it or not?
          if (ifunc->pentry[j]->size < 0) {
            // already precompiled, skip it
            continue;
          }
#endif

          // Check for constructor, destructor, or operator=.
          if (!strcmp(ifunc->funcname[j], G__struct.name[i])) {
            // We have a constructor.
            if (G__struct.isabstract[i]) {
              continue;
            }
            if (isnonpublicnew) {
              continue;
            }
            ++isconstructor;

            if ((ifunc->para_nu[j] >= 1) && (ifunc->param[j][0]->type == 'u') 
                && (i == ifunc->param[j][0]->p_tagtable) && (ifunc->param[j][0]->reftype == G__PARAREFERENCE) 
                && ((ifunc->para_nu[j] == 1) || ifunc->param[j][1]->pdefault)) {
              ++iscopyconstructor;
            }
          } else if (ifunc->funcname[j][0] == '~') {
            // We have a destructor.
            dtoraccess = ifunc->access[j];
            virtualdtorflag = ifunc->isvirtual[j] + (ifunc->ispurevirtual[j] * 2);
            if (G__PUBLIC != ifunc->access[j]) {
              ++isdestructor;
            }
            else
              ifunc_destructor = ifunc;

            if ((G__PROTECTED == ifunc->access[j]) && G__struct.protectedaccess[i] && !G__precomp_private) {
              G__fprinterr(G__serr, "Limitation: can not generate dictionary for protected destructor for %s\n", G__fulltagname(i, 1));
              continue;
            }
            continue;
          }

#ifdef G__DEFAULTASSIGNOPR
          else if (!strcmp(ifunc->funcname[j], "operator=")
                   && ('u' == ifunc->param[j][0]->type)
                   && (i == ifunc->param[j][0]->p_tagtable)) {
            // We have an operator=.
            ++isassignmentoperator;
          }
#endif

        } // end of if access public && not pure virtual func
        else { // in case of protected,private or pure virtual func
               // protected, private, pure virtual
          if (!strcmp(ifunc->funcname[j], G__struct.name[i])) {
            ++isconstructor;
            if ('u' == ifunc->param[j][0]->type && i == ifunc->param[j][0]->p_tagtable 
                && G__PARAREFERENCE == ifunc->param[j][0]->reftype 
                && (1 == ifunc->para_nu[j] || ifunc->param[j][1]->pdefault)) {
              // copy constructor
              ++iscopyconstructor;
            }
          } else if ('~' == ifunc->funcname[j][0]) {
            // destructor
            ++isdestructor;
            ifunc_destructor = ifunc;
          } else if (!strcmp(ifunc->funcname[j], "operator new")) {
            ++isconstructor;
            ++iscopyconstructor;
          } else if (!strcmp(ifunc->funcname[j], "operator delete")) {
            // destructor
            ++isdestructor;
          }
#ifdef G__DEFAULTASSIGNOPR
          else if (!strcmp(ifunc->funcname[j], "operator=") && 'u' == ifunc->param[j][0]->type && i == ifunc->param[j][0]->p_tagtable) {
            ++isassignmentoperator;
          }
#endif
        } // end of if access not public
      }

      if(ifunc->next == 0)
        break;

      ifunc = ifunc->next;
    } /* end while(ifunc) */

    if ( ifunc && ifunc->next == 0
         // dummy
#ifndef G__OLDIMPLEMENTATON1656
         && G__NOLINK == G__struct.iscpplink[i]
#endif
#ifndef G__OLDIMPLEMENTATON1730
         && G__ONLYMETHODLINK != G__struct.globalcomp[i]
#endif
      ) {

      G__p_ifunc = ifunc;

      /****************************************************************
       * setup default constructor
       ****************************************************************/
      if (0 == isconstructor) isconstructor = G__isprivateconstructor(i, 0);
      if ('n' == G__struct.type[i]) isconstructor = 1;
      if (0 == isconstructor && 0 == G__struct.isabstract[i] && 0 == isnonpublicnew) {
        // putting automatic default constructor in the ifunc table
        funcname = G__struct.name[i];
        G__hash(funcname, hash, k);

        ifunc = G__p_ifunc;
#ifdef G__TRUEP2F
        G__memfunc_setup(funcname, hash, (G__InterfaceMethod) NULL
                         ,(int) ('i') , i
                         ,-1,0,0,1,G__PUBLIC,0
                         , "", (char*) NULL, (void*) NULL, 0);
#else
        G__memfunc_setup(funcname, hash, (G__InterfaceMethod) NULL
                         ,(int) ('i') , i
                         ,-1,0,0,1,G__PUBLIC,0
                         , "", (char*) NULL);
#endif

        // 08-11-07
        // Flag this method as an automatic one.
        ifunc->funcptr[0] = (void*)-1 ;
      } /* if(isconstructor) */

      /****************************************************************
       * setup copy constructor
       ****************************************************************/
      if (0 == iscopyconstructor) iscopyconstructor = G__isprivateconstructor(i, 1);
      if ('n' == G__struct.type[i]) iscopyconstructor = 1;
      if (0 == iscopyconstructor && 0 == G__struct.isabstract[i] && 0 == isnonpublicnew) {
        // putting automatic copy constructor
        funcname = G__struct.name[i];
        G__hash(funcname, hash, k);

        G__FastAllocString paras(G__MAXNAME*6);
        paras.Format("u '%s' - 11 - -",  G__fulltagname(i, 0));

        ifunc = G__p_ifunc;
#ifdef G__TRUEP2F
        G__memfunc_setup(funcname, hash, (G__InterfaceMethod) NULL
                         ,(int) ('i') ,i
                         ,-1,0,1,1,G__PUBLIC,0
                         , paras, (char*) NULL, (void*) NULL, 0);
#else
        G__memfunc_setup(funcname, hash, (G__InterfaceMethod) NULL
                         ,(int) ('i') ,i
                         ,-1,0,1,1,G__PUBLIC,0
                         , paras, (char*) NULL);
#endif
        // 08-11-07
        // Flag this method as an automatic one.
        ifunc->funcptr[0] = (void*)-1 ;
      } /* if(iscopyconstructor) */


      /****************************************************************
       * setup destructor
       ****************************************************************/
      if (0 == isdestructor) isdestructor = G__isprivatedestructor(i);
      if ('n' == G__struct.type[i]) isdestructor = 1;
      if (ifunc_destructor && 'n' != G__struct.type[i]) {
        // putting automatic destructor
         funcname.Format("~%s", G__struct.name[i]);
        G__hash(funcname, hash, k);

        // 25-07-07
        // the ifunc field has already been created for the destructor... lets just fill it up
        /* set entry pointer */
        if(!(*ifunc_destructor->funcname[j]))
          G__savestring(&ifunc_destructor->funcname[j],funcname);

        if(!ifunc_destructor->hash[j])
          ifunc_destructor->hash[j] = hash;
        ifunc_destructor->type[j] = (int) ('y');
        ifunc_destructor->p_tagtable[j] = -1;
        ifunc_destructor->p_typetable[j] = -1;
        ifunc_destructor->reftype[j] = 0;
        ifunc_destructor->para_nu[j] = 0;
        ifunc_destructor->ansi[j] = 1;
        ifunc_destructor->access[j] = dtoraccess;
        ifunc_destructor->isconst[j] = 0;

        // 08-11-07
        // Flag this method as an automatic one.
        ifunc_destructor->funcptr[j] = (void*)-1 ;
      } /* if(isdestructor) */


#ifdef G__DEFAULTASSIGNOPR
      /****************************************************************
       * setup assignment operator
       ****************************************************************/

      if (0 == isassignmentoperator) isassignmentoperator = G__isprivateassignopr(i);
      if ('n' == G__struct.type[i]) isassignmentoperator = 1;
      if (0 == isassignmentoperator) {
        // putting automatic assignment operator
        funcname =  "operator=";
        G__hash(funcname, hash, k);

        G__FastAllocString paras(G__MAXNAME*6);
        paras.Format("u '%s' - 11 - -",  G__fulltagname(i, 0));

        ifunc = G__p_ifunc;
#ifdef G__TRUEP2F
        G__memfunc_setup(funcname, hash, (G__InterfaceMethod) NULL
                         ,(int) ('u'),i
                         ,-1,1,1,1,G__PUBLIC,0
                         , paras, (char*) NULL, (void*) NULL, 0);
#else
        G__memfunc_setup(funcname, hash, (G__InterfaceMethod) NULL
                         ,(int) ('u'),i
                         ,-1,1,1,1,G__PUBLIC,0
                         , paras, (char*) NULL);
#endif

        // 08-11-07
        // Flag this method as an automatic one.
        ifunc->funcptr[0] = (void*)-1 ;
      } /* if(isassignmentoperator) */

#endif // G__DEFAULTASSIGNOPR
    } /* end of ifunc->next */
  }/* end if(globalcomp) */
  G__p_ifunc = store_p_ifunc;
}

/**************************************************************************
* G__cppif_memfunc() working
* Writes the stub functions in the dictioinary
*
**************************************************************************/
void G__cppif_memfunc(FILE *fp, FILE *hfp)
{
#ifndef G__SMALLOBJECT
  int i,j;
  struct G__ifunc_table_internal *ifunc;
  struct G__ifunc_table_internal *ifunc_default;
  int isconstructor,iscopyconstructor,isdestructor,isassignmentoperator;
  int isnonpublicnew;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Member function Interface Method\n");
  fprintf(fp,"*********************************************************/\n");

  /* This loop goes through all the classes loaded in the G__struct */
  for(i=0;i<G__struct.alltag;i++) {
    if( /* Linkage */
      (G__CPPLINK==G__struct.globalcomp[i]||
       G__CLINK==G__struct.globalcomp[i]
       || G__ONLYMETHODLINK==G__struct.globalcomp[i]
        ) &&
      (-1==(int)G__struct.parent_tagnum[i]
       || G__nestedclass
        )
      &&
      -1!=G__struct.line_number[i]&&G__struct.hash[i]&&
      '$'!=G__struct.name[i][0] && 'e'!=G__struct.type[i]) { /* Not enums */
      ifunc = G__struct.memfunc[i];
      isconstructor=0;
      iscopyconstructor=0;
      isdestructor=0;
      isassignmentoperator=0;
      isnonpublicnew=G__isnonpublicnew(i);

      ifunc_default = ifunc;

#ifdef G__NOSTUBS
      // 28-01-08
      // Why are the defaults method introduced after the class registering?
      // If we do that, we won't be able to avoid their stubs...
      if( (G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary) &&
          G__is_tagnum_safe(i))
        G__make_default_ifunc(ifunc_default);

      // 03-07-07
      // Trigger the symbol registering to have them at hand
      // Do it here when we have the library and the class
      if(G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
        G__register_class(G__libname, G__type2string('u',i,-1,0,0));
#endif

      /* member function interface */
      fprintf(fp,"\n/* %s */\n",G__fulltagname(i,0));

      if(G__dicttype==kFunctionSymbols)
        G__write_preface(fp, ifunc, i);

      // Flag to see if we have written the dummy pointer
      int ptr_written = 0;
      // Go through all the function members in the class 
      while(ifunc) {
        for(j=0;j<ifunc->allifunc;j++) {
          if(G__PUBLIC==ifunc->access[j] // Public Method?
             || (G__PROTECTED==ifunc->access[j] &&
                 (G__PROTECTEDACCESS&G__struct.protectedaccess[i]))
             || (G__PRIVATEACCESS&G__struct.protectedaccess[i])
            ) {
            if(G__ONLYMETHODLINK==G__struct.globalcomp[i]&&
               G__METHODLINK!=ifunc->globalcomp[j]) continue;
            if(0==ifunc->hash[j]) continue;
#ifndef G__OLDIMPLEMENTATION1656
            // 08-10-07
            // With no-stubs (ifunc->pentry[j]->size < 0) by default.
            if (G__dicttype==kCompleteDictionary || !G__is_tagnum_safe(i)){
              if (ifunc->pentry[j]->size < 0) {
                // already precompiled, skip it
                continue;
              }
            }
#endif
            // Constructor? ClassName == FunctionName
            if(strcmp(ifunc->funcname[j],G__struct.name[i])==0) {
              // Constructor and dictionary #2
              // Generating dummy object for getting the symbols of default constructor and destructor
              if(G__dicttype==kFunctionSymbols) {
                if(G__is_tagnum_safe(i))
                  G__cppif_dummyobj(fp, ifunc, i, j);
              }
              else if( !(!ifunc->mangled_name[j] /*&& ifunc->funcptr[j]==(void*)-1*/) || // No mangled name and No function Pointer
                       (!ifunc->mangled_name[j] && G__dicttype!=kFunctionSymbols ) // No mangled name and no dicttype=2         
                ){
                /* constructor needs special handling */
                if(0==G__struct.isabstract[i]&&0==isnonpublicnew) {
                  // No Constructor Stub
                  if(((!ifunc->mangled_name[j] || /*if there is no symbol*/
                       !G__is_tagnum_safe(i) )
                      && G__dicttype==kNoWrappersDictionary) 
                     || G__dicttype==kCompleteDictionary || G__dicttype==kNoWrappersDictionary ) {
                    if( !ifunc->mangled_name[j] /*|| ifunc->funcptr[j]>0*/ || !G__nostubs) /* if there no is a symbol or the no stubs flag is not activated */
                      G__cppif_genconstructor(fp,hfp,i,j,ifunc);
                  }
                }
                ++isconstructor;
                if(ifunc->para_nu[j]>=1&&
                   'u'==ifunc->param[j][0]->type&&
                   i==ifunc->param[j][0]->p_tagtable&&
                   G__PARAREFERENCE==ifunc->param[j][0]->reftype&&
                   (1==ifunc->para_nu[j]||ifunc->param[j][1]->pdefault)) {
                  ++iscopyconstructor;
                }
              }
            }
            else if('~'==ifunc->funcname[j][0]) { // Destructor?
              /* destructor is created in gendefault later */
              if(G__PUBLIC==ifunc->access[j]){
                if(G__dicttype==kNoWrappersDictionary){
                  if(G__is_tagnum_safe(i)) {
                    if( !ifunc->mangled_name[j] /*|| ifunc->funcptr[j]>0*/ || !G__nostubs ) /* if there no is a symbol or the no stubs flag is not activated */
                      G__cppif_gendefault(fp,hfp,i,j,ifunc,1,1,isdestructor,1,1);
                    ++isdestructor; // don't try to create it later on
                  }
                }
                else if((G__dicttype==kNoWrappersDictionary) && ifunc->mangled_name[j] /*if there is not a symbol*/)
                  ++isdestructor;
                else if(G__dicttype!=kNoWrappersDictionary){
                  isdestructor = -1;
                }
              }
              else{
                ++isdestructor;
              }
              continue;
            }
            else if('\0'==ifunc->funcname[j][0] && j==0) {
              /* this must be the place holder for the destructor.
               * let's skip it! */
              continue;
            }
            else {
#ifdef G__DEFAULTASSIGNOPR
              if(strcmp(ifunc->funcname[j],"operator=")==0
                 && 'u'==ifunc->param[j][0]->type
                 && i==ifunc->param[j][0]->p_tagtable) {
              if( G__dicttype!=kNoWrappersDictionary ||
                  (G__dicttype==kNoWrappersDictionary &&
                   !(!ifunc->mangled_name[j] && ifunc->funcptr[j]==(void*)-1)))
                ++isassignmentoperator;
              }
#endif
              // 06-11-07
              // Try to rewrite the condition now that the symbols are available
              // also for the second dictionary

              // If there is no symbol and this is the second dictionary
              // generate the inline code

              // 14-02-08 When ifunc->funcptr[j]==(void*)-2 we have to force the creation of the stub
              if(G__dicttype==kFunctionSymbols && ifunc->funcptr[j]!=(void*)-2){
                if (G__is_tagnum_safe(i)) {
                  if(G__struct.isabstract[ifunc->tagnum]) {
                    G__cppif_geninline(fp, ifunc, i, j);
                  }
                  else {
                    if (ifunc->isvirtual[j] || ifunc->ispurevirtual[j]) {
                      // the first thing is to write the pointer if it hasn't been written
                       if(!ptr_written){
                          G__write_dummy_ptr(fp, ifunc, i);
                          ptr_written = 1;
                       }
                       // Now write the function call to force the generation of the symbol 
                       G__cpp_methodcall(fp, ifunc, i, j); 
                    }                    
                    else {
                      G__cppif_geninline(fp, ifunc, i, j);
                    }
                  }
                }
              }

              // The ifunc->funcptr[j]==(void*)-1 means that they were generated automatically,
              // if that is the case and we are writing the final dictionary (kCompleteDictionary),
              // we can't write the stubs for the operator=.
              // 14-02-08 When ifunc->funcptr[j]==(void*)-2 we have to force the creation of the stub
              if((ifunc->funcptr[j]==(void*)-2) || !ifunc->mangled_name[j] || 
                // 26-10-07
                // Generate the stubs for those function needing a pointer to a reference (see TCLonesArray "virtual TObject*&	operator[](Int_t idx)")
                // Is this condition correct and/or sufficient?
                ((ifunc->reftype[j] == G__PARAREFERENCE) && isupper(ifunc->type[j])) ||
		 !G__nostubs){
                if(G__dicttype==kNoWrappersDictionary){
                  // Now we have no symbol but we are in the third or fourth
                  // dictionary... which means that the second one already tried to create it...
                  // If that's the case we have no other choice but to generate the stub
                  if(!(strcmp(ifunc->funcname[j],"operator=")==0 && ifunc->funcptr[j]==(void*)-1))
                    G__cppif_genfunc(fp,hfp,i,j,ifunc);
                }
                else if(G__dicttype==kCompleteDictionary){
                  // This is the old case...
                  // just do what we did before
                  G__cppif_genfunc(fp,hfp,i,j,ifunc);
                }
              }
        
            }
          } /* if PUBLIC */
          else { /* if PROTECTED or PRIVATE */
            if(strcmp(ifunc->funcname[j],G__struct.name[i])==0) {
              ++isconstructor;
              if (G__dicttype==kFunctionSymbols)
                G__cppif_geninline(fp, ifunc, i, j);
              if(
                ifunc->para_nu[j]>0 &&
                'u'==ifunc->param[j][0]->type&&i==ifunc->param[j][0]->p_tagtable&&
                G__PARAREFERENCE==ifunc->param[j][0]->reftype&&
                (1==ifunc->para_nu[j]||ifunc->param[j][1]->pdefault)) {
                ++iscopyconstructor;
              }
            }
            else if('~'==ifunc->funcname[j][0]) {
              ++isdestructor;
            }
            else if(strcmp(ifunc->funcname[j],"operator new")==0) {
              ++isconstructor;
              ++iscopyconstructor;
            }
            else if(strcmp(ifunc->funcname[j],"operator delete")==0) {
              ++isdestructor;
            }
#ifdef G__DEFAULTASSIGNOPR
            else if(strcmp(ifunc->funcname[j],"operator=")==0
                    && 'u'==ifunc->param[j][0]->type
                    && i==ifunc->param[j][0]->p_tagtable
              ) {
              ++isassignmentoperator;
              if (G__dicttype==kFunctionSymbols)
                G__cppif_geninline(fp, ifunc, i, j);
            }
#endif
          }
        } /* for(j) */
        if(NULL==ifunc->next
#ifndef G__OLDIMPLEMENTATON1656
           && G__NOLINK==G__struct.iscpplink[i]
#endif
           && G__ONLYMETHODLINK!=G__struct.globalcomp[i]
          ){

          // 21-06-07
          // for the stubs don't generate default memebers
          // Note: we don't want neither constructor nor copyconstructor 
          // nor destructor nor asign. operator stubs       

          // 06-07-07
          // We need wrappers for default functions (constructor and assignment operator...)
          // except for the destructor because of mangled names in the dicitionary,
          // but deal with that in G__cppif_gendefault
          if(  G__dicttype==kCompleteDictionary || G__dicttype==kNoWrappersDictionary ||
               (( G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
                && G__is_tagnum_safe(i)) ||
               (G__dicttype==kNoWrappersDictionary && !G__is_tagnum_safe(i)) ){ // rem to create inlines for default functions
            if( !ifunc->mangled_name[j] /*|| ifunc->funcptr[j]>0*/ || !G__nostubs )
              G__cppif_gendefault(fp,hfp,i,j,ifunc
                                  ,isconstructor
                                  ,iscopyconstructor
                                  ,isdestructor
                                  ,isassignmentoperator
                                  ,isnonpublicnew);
          }
          break;
        }
        ifunc=ifunc->next;
      } /* while(ifunc) */

      if(G__dicttype==kFunctionSymbols)
        G__write_postface(fp, ifunc, i);
    } /* if(globalcomp) */
  } /* for(i) */
#endif // G__SMALLOBJECT
}

/**************************************************************************
 * G__cppif_func() working
 *
 *
 **************************************************************************/
void G__cppif_func(FILE *fp, FILE *hfp)
{
  int j;
  struct G__ifunc_table_internal *ifunc;

#ifdef G__NOSTUBS
  // 30-07-07
  // Trigger the symbol registering to have them at hand
  // Do it here when we have the library and the class
  if(G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
    G__register_class(G__libname, 0);
#endif

  fprintf(fp,"\n/* Setting up global function */\n");
  ifunc = &G__ifunc;

  /* member function interface */
  while(ifunc) {
    for(j=0;j<ifunc->allifunc;j++) {
      if(G__NOLINK>ifunc->globalcomp[j] &&
         G__PUBLIC==ifunc->access[j] &&
         0==ifunc->staticalloc[j] &&
         ifunc->hash[j]) {

        // 24/05/07
        // Generate the stubs only for operators
        // (when talking about global functions)
        // 11-07-07
        if(G__dicttype==kCompleteDictionary || G__dicttype==kNoWrappersDictionary) {
          // The stubs where generated for functions needing temporal objects too... use them
          if  ( !ifunc->mangled_name[j] ||
                // 26-10-07
                // Generate the stubs for those function needing a pointer to a reference (see TCLonesArray "virtual TObject*&	operator[](Int_t idx)")
                // Is this condition correct and/or sufficient?
                ((ifunc->reftype[j] == G__PARAREFERENCE) && isupper(ifunc->type[j]))
                || !G__nostubs)
            G__cppif_genfunc(fp,hfp,-1,j,ifunc);
        }
        else {
          // we can't write the stubs for the operator=.
          // 14-02-08 When ifunc->funcptr[j]==(void*)-2 we have to force the creation of the stub
          if (!ifunc->mangled_name[j] && ifunc->funcptr[j]!=(void*)-2)
            G__cppif_geninline(fp, ifunc, -1, j);
        }
      } /* if(access) */
    } /* for(j) */
    ifunc=ifunc->next;
  } /* while(ifunc) */
}

/**************************************************************************
* G__cppif_dummyfuncname()
*
**************************************************************************/
void G__cppif_dummyfuncname(FILE *fp)
{
#ifndef G__IF_DUMMY
  fprintf(fp,"   return(1);\n");
#else
  fprintf(fp,"   return(1 || funcname || hash || result7 || libp) ;\n");
#endif
}
/**************************************************************************
*  G__if_ary_union()
*
**************************************************************************/
void G__if_ary_union(FILE *fp, int ifn, G__ifunc_table_internal *ifunc)
{
  int k, m;
  char* p;

  m = ifunc->para_nu[ifn];

  for (k = 0; k < m; ++k) {
    if (ifunc->param[ifn][k]->name) {
      p = strchr(ifunc->param[ifn][k]->name, '[');
      if (p) {
        fprintf(fp, "  struct G__aRyp%d { %s a[1]%s; }* G__Ap%d = (struct G__aRyp%d*) G__int(libp->para[%d]);\n", k, G__type2string(ifunc->param[ifn][k]->type, ifunc->param[ifn][k]->p_tagtable, ifunc->param[ifn][k]->p_typetable, 0 , 0), p + 2, k, k, k);
      }
    }
  }
}

/**************************************************************************
*  G__if_ary_union_reset()
*
**************************************************************************/
void G__if_ary_union_reset(int ifn, G__ifunc_table_internal* ifunc)
{
   int m = ifunc->para_nu[ifn];
   for (int k = 0; k < m; ++k) {
      if (!ifunc->param[ifn][k]->name) {
         continue;
      }
      char* p = strchr(ifunc->param[ifn][k]->name, '[');
      if (!p) {
         continue;
      }
      *p = 0;
      int pointlevel = 1;
      p = strchr(p + 1, '[');
      while (p) {
         ++pointlevel;
         p = strchr(p + 1, '[');
      }
      char type = ifunc->param[ifn][k]->type;
      if (isupper(type)) {
         switch (pointlevel) {
            case 2:
               ifunc->param[ifn][k]->reftype = G__PARAP2P2P;
               break;
            default:
               G__genericerror("Cint internal error ary parameter dimension");
               break;
         }
      }
      else {
         ifunc->param[ifn][k]->type = toupper(type);
         switch (pointlevel) {
            case 2:
               ifunc->param[ifn][k]->reftype = G__PARAP2P;
               break;
            case 3:
               ifunc->param[ifn][k]->reftype = G__PARAP2P2P;
               break;
            default:
               G__genericerror("Cint internal error ary parameter dimension");
               break;
         }
      }
   }
}

#ifdef G__CPPIF_EXTERNC
/**************************************************************************
* G__p2f_typedefname
**************************************************************************/
char* G__p2f_typedefname(ifn,page,k)
int ifn;
short page;
int k)
{
  static G__FastAllocString buf(G__ONELINE);
  buf.Format("G__P2F%d_%d_%d%s",ifn,page,k,G__PROJNAME.data());
  return(buf);
}

/**************************************************************************
* G__p2f_typedef
**************************************************************************/
void G__p2f_typedef(fp,ifn,ifunc)
FILE *fp;
int ifn;
struct G__ifunc_table_internal *ifunc)
{
  G__FastAllocString buf(G__LONGLINE);
  char *p;
  int k;
  if(G__CPPLINK!=G__globalcomp) return;
  for(k=0;k<ifunc->para_nu[ifn];k++) {
    /*DEBUG*/ printf("%s %d\n",ifunc->funcname[ifn],k);
    if(
#ifndef G__OLDIMPLEMENTATION2191
       '1'==ifunc->param[ifn][k]->type
#else
       'Q'==ifunc->param[ifn][k]->type
#endif
       ) {
      buf = G__type2string(ifunc->param[ifn][k]->type,
                                ifunc->param[ifn][k]->p_tagtable,
                                ifunc->param[ifn][k]->p_typetable,0,
                                ifunc->param[ifn][k]->isconst));
    /*DEBUG*/ printf("buf=%s\n",buf());
      p = strstr(buf,"(*)(");
      if(p) {
        p += 2;
        *p = 0;
        fprintf(fp,"typedef %s%s",buf(),G__p2f_typedefname(ifn,ifunc->page,k));
        *p = ')';
        fprintf(fp,"%s;\n",p);
      }
    }
  }
}
#endif // G__CPPIF_EXTERNC

/**************************************************************************
* G__isprotecteddestructoronelevel()
*
**************************************************************************/
static int G__isprotecteddestructoronelevel(int tagnum)
{
  char *dtorname;
  struct G__ifunc_table_internal *ifunc;
  int ifn;
  ifunc=G__struct.memfunc[tagnum];
  dtorname = (char*)malloc(strlen(G__struct.name[tagnum])+2);
  dtorname[0]='~';
  strcpy(dtorname+1,G__struct.name[tagnum]); // Okay, we allocated the right size
  do {
    for(ifn=0;ifn<ifunc->allifunc;ifn++) {
      if(strcmp(dtorname,ifunc->funcname[ifn])==0) {
        if(G__PRIVATE==ifunc->access[ifn]||G__PROTECTED==ifunc->access[ifn]) {
          free((void*)dtorname);
          return(1);
        }
      }
    }
    ifunc=ifunc->next;
  } while(ifunc);
  free((void*)dtorname);
  return(0);
}


#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
/**************************************************************************
* G__x8664_vararg()
*
* This function sets up vararg calls on X86_64 (AMD64 and EM64T).
* On these platforms arguments are passed by register and not on
* the stack. The Linux ABI specifies that the first 6 integer and
* 8 double arguments are passed via registers, while any remaining
* arguments are passed via the stack. In this function we use inline
* assembler to set the arguments in the right registers before
* calling the vararg function.
*
**************************************************************************/
static void G__x8664_vararg(FILE *fp, int ifn, G__ifunc_table_internal *ifunc,
                            const char *fn, int tagnum, const char * /*cls*/)
{
   //const int dmax = 8;    // number of floating point xmm registers
   const int imax = 6;    // number of integer registers
   const int umax = 50;   // maximum number of extra vararg stack arguments

   fprintf(fp, "   const int imax = 6, dmax = 8, umax = 50;\n");
   fprintf(fp, "   int objsize, type, i, icnt = 0, dcnt = 0, ucnt = 0;\n");
   fprintf(fp, "   G__value *pval;\n");
   fprintf(fp, "   G__int64 lval[imax];\n");
   fprintf(fp, "   double dval[dmax];\n");
   fprintf(fp, "   union { G__int64 lval; double dval; } u[umax];\n");
              
   if (tagnum != -1 && !ifunc->staticalloc[ifn])
      fprintf(fp, "   lval[icnt] = G__getstructoffset(); icnt++; // this pointer\n");

   fprintf(fp, "   for (i = 0; i < libp->paran; i++) {\n");
   fprintf(fp, "      type = G__value_get_type(&libp->para[i]);\n");
   fprintf(fp, "      pval = &libp->para[i];\n");
   fprintf(fp, "      if (isupper(type))\n");
   fprintf(fp, "         objsize = G__LONGALLOC;\n");
   fprintf(fp, "      else\n");
   fprintf(fp, "         objsize = G__sizeof(pval);\n");

   fprintf(fp, "      switch (type) {\n");
   fprintf(fp, "         case 'c': case 'b': case 's': case 'r': objsize = sizeof(int); break;\n");
   fprintf(fp, "         case 'f': objsize = sizeof(double); break;\n");
   fprintf(fp, "      }\n");

   fprintf(fp, "#ifdef G__VAARG_PASS_BY_REFERENCE\n");
   fprintf(fp, "      if (objsize > G__VAARG_PASS_BY_REFERENCE) {\n");
   fprintf(fp, "         if (pval->ref > 0x1000) {\n");
   fprintf(fp, "            if (icnt < imax) {\n");
   fprintf(fp, "               lval[icnt] = pval->ref; icnt++;\n");
   fprintf(fp, "            } else {\n");
   fprintf(fp, "               u[ucnt].lval = pval->ref; ucnt++;\n");
   fprintf(fp, "            }\n");
   fprintf(fp, "         } else {\n");
   fprintf(fp, "            if (icnt < imax) {\n");
   fprintf(fp, "               lval[icnt] = G__int(*pval); icnt++;\n");
   fprintf(fp, "            } else {\n");
   fprintf(fp, "               u[ucnt].lval = G__int(*pval); ucnt++;\n");
   fprintf(fp, "            }\n");
   fprintf(fp, "         }\n");
   fprintf(fp, "         type = 'z';\n");
   fprintf(fp, "      }\n");
   fprintf(fp, "#endif\n");

   fprintf(fp, "      switch (type) {\n");
   fprintf(fp, "         case 'n': case 'm':\n");
   fprintf(fp, "            if (icnt < imax) {\n");
   fprintf(fp, "               lval[icnt] = (G__int64)G__Longlong(*pval); icnt++;\n");
   fprintf(fp, "            } else {\n");
   fprintf(fp, "               u[ucnt].lval = (G__int64)G__Longlong(*pval); ucnt++;\n");
   fprintf(fp, "            } break;\n");
   fprintf(fp, "         case 'f': case 'd':\n");
   fprintf(fp, "            if (dcnt < dmax) {\n");
   fprintf(fp, "               dval[dcnt] = G__double(*pval); dcnt++;\n");
   fprintf(fp, "            } else {\n");
   fprintf(fp, "               u[ucnt].dval = G__double(*pval); ucnt++;\n");
   fprintf(fp, "            } break;\n");
   fprintf(fp, "         case 'z': break;\n");
   fprintf(fp, "         case 'u':\n");
   fprintf(fp, "            if (objsize >= 16) {\n");
   fprintf(fp, "               memcpy(&u[ucnt].lval, (void*)pval->obj.i, objsize);\n");
   fprintf(fp, "               ucnt += objsize/8;\n");
   fprintf(fp, "               break;\n");
   fprintf(fp, "            }\n");
   fprintf(fp, "            // objsize < 16 -> fall through\n");
   fprintf(fp, "         case 'g': case 'c': case 'b': case 'r': case 's': case 'h': case 'i':\n");
   fprintf(fp, "         case 'k': case 'l':\n");
   fprintf(fp, "         default:\n");
   fprintf(fp, "            if (icnt < imax) {\n");
   fprintf(fp, "               lval[icnt] = G__int(*pval); icnt++;\n");
   fprintf(fp, "            } else {\n");
   fprintf(fp, "               u[ucnt].lval = G__int(*pval); ucnt++;\n");
   fprintf(fp, "            } break;\n");
   fprintf(fp, "      }\n");
   fprintf(fp, "      if (ucnt >= %d) printf(\"%s: more than %d var args\\n\");\n", umax, fn, umax+imax);
   fprintf(fp, "   }\n");
}

static void G__x8664_vararg_write(FILE *fp, int xmm, int reg)
{
   // Write out explicit argument list in vararg function/method.
   // See also G__x8664_vararg().
   
   const int imax = 6, dmax = 8, umax = 50;
   int i;
   for (i = xmm; i < dmax; i++)
      fprintf(fp, ", dval[%d]", i);
   for (i = reg; i < imax; i++)
      fprintf(fp, ", lval[%d]", i);
   for (i = 0; i < umax; i++)
      fprintf(fp, ", u[%d].lval", i);
}
#endif

/**************************************************************************
* G__cppif_genconstructor()
*
* Write a special constructor wrapper that handles placement new
* using G__getgvp().  All calls to the constructor get routed here and we
* eventually call the real constructor with the appropriate arguments.
*
**************************************************************************/
void G__cppif_genconstructor(FILE *fp, FILE * /* hfp */, int tagnum, int ifn, G__ifunc_table_internal *ifunc)
{
#ifndef G__SMALLOBJECT
  int k, m, ret, reg = 0, xmm = 0;
  int isprotecteddtor = G__isprotecteddestructoronelevel(tagnum);
  G__FastAllocString buf(G__LONGLINE);

  G__ASSERT( tagnum != -1 );

  if ((G__PROTECTED == ifunc->access[ifn]) || (G__PRIVATE == ifunc->access[ifn])) {
     buf.Format("%s_PR", G__get_link_tagname(tagnum));
  } else {
    buf =  G__fulltagname(tagnum, 1);
  }

#ifdef G__CPPIF_EXTERNC
  G__p2f_typedef(fp, ifn, ifunc);
#endif

#ifdef G__CPPIF_STATIC
  fprintf(fp,               "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, G__struct.name[tagnum], ifn, ifunc->page));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
  fprintf(G__WINDEFfp,      "        %s @%d\n", G__map_cpp_funcname(tagnum, G__struct.name[tagnum], ifn, ifunc->page), ++G__nexports);
#endif
  fprintf(hfp,              "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, G__struct.name[tagnum], ifn, ifunc->page));

  fprintf(fp,               "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, G__struct.name[tagnum], ifn, ifunc->page));
#endif /* G__CPPIF_STATIC */


  fprintf(fp,               "\n{\n");


#if !defined(G__BORLAND)
  fprintf(fp,               "   %s* p = NULL;\n", G__type2string('u', tagnum, -1, 0, 0));
#else
  fprintf(fp,               "   %s* p;\n", G__type2string('u', tagnum, -1, 0, 0));
#endif


#ifndef G__VAARG_COPYFUNC
  if (ifunc->ansi[ifn] == 2) {
    // Handle a variadic function (variable number of arguments).
#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
    G__x8664_vararg(fp, ifn, ifunc, buf, tagnum, buf);
#else
    fprintf(fp, "   G__va_arg_buf G__va_arg_bufobj;\n");
    fprintf(fp, "   G__va_arg_put(&G__va_arg_bufobj, libp, %d);\n", ifunc->para_nu[ifn]);
#endif
  }
#endif

  G__if_ary_union(fp, ifn, ifunc);

  fprintf(fp,               "   char* gvp = (char*) G__getgvp();\n");

  bool has_a_new = G__struct.funcs[tagnum] & (G__HAS_OPERATORNEW1ARG | G__HAS_OPERATORNEW2ARG);
  bool has_a_new1arg = G__struct.funcs[tagnum] & G__HAS_OPERATORNEW1ARG;
  bool has_a_new2arg = G__struct.funcs[tagnum] & G__HAS_OPERATORNEW2ARG;

  bool has_own_new1arg = false;
  bool has_own_new2arg = false;

  {
    struct G__ifunc_table* iref = G__get_ifunc_ref(G__struct.memfunc[tagnum]);
    struct G__ifunc_table* ireffound = 0;
    long index;
    long offset;
    ireffound = G__get_methodhandle("operator new", "size_t", iref,
                                &index, &offset, 0, 0);
    has_own_new1arg = (ireffound != 0);
    ireffound = G__get_methodhandle("operator new", "size_t, void*", iref,
                                &index, &offset, 0, 0);
    has_own_new2arg = (ireffound != 0);
  }

  //FIXME: debugging code
  //fprintf(fp,               "   //\n");
  //fprintf(fp,               "   //has_a_new1arg: %d\n", has_a_new1arg);
  //fprintf(fp,               "   //has_a_new2arg: %d\n", has_a_new2arg);
  //fprintf(fp,               "   //has_own_new1arg: %d\n", has_own_new1arg);
  //fprintf(fp,               "   //has_own_new2arg: %d\n", has_own_new2arg);
  //fprintf(fp,               "   //\n");

  m = ifunc->para_nu[ifn] ;

  if ((m > 0) && ifunc->param[ifn][m-1]->pdefault) {
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
        } else {
          fprintf(fp,       "       if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
          if (!has_a_new) {
             fprintf(fp,     "         p = new %s[n];\n", buf());
          } else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
             fprintf(fp,     "         p = new %s[n];\n", buf());
          } else {
             fprintf(fp,     "         p = ::new %s[n];\n", buf());
          }
          fprintf(fp,       "       } else {\n");
          if (!has_a_new) {
             fprintf(fp,     "         p = new((void*) gvp) %s[n];\n", buf());
          } else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
             fprintf(fp,     "         p = new((void*) gvp) %s[n];\n", buf());
          } else {
             fprintf(fp,     "         p = ::new((void*) gvp) %s[n];\n", buf());
          }
          fprintf(fp,       "       }\n");
        }
        fprintf(fp,         "     } else {\n");
        // Handle regular new.
        fprintf(fp,         "       if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
        if (!has_a_new) {
           fprintf(fp,         "         p = new %s;\n", buf());
        } else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
           fprintf(fp,       "         p = new %s;\n", buf());
        } else {
           fprintf(fp,       "         p = ::new %s;\n", buf());
        }
        fprintf(fp,         "       } else {\n");
        if (!has_a_new) {
           fprintf(fp,       "         p = new((void*) gvp) %s;\n", buf());
        } else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
           fprintf(fp,       "         p = new((void*) gvp) %s;\n", buf());
        } else {
           fprintf(fp,       "         p = ::new((void*) gvp) %s;\n", buf());
        }
        fprintf(fp,         "       }\n");
        fprintf(fp,         "     }\n");
      } else {
        // Caller gave us some of the arguments.
        //
        // Note: We do not have to handle array new here because there
        //       can be no initializer in an array new.
        fprintf(fp,         "     //m: %d\n", m);
        fprintf(fp,         "     if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
        if (!has_a_new) {
           fprintf(fp,         "       p = new %s(", buf());
        } else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
           fprintf(fp,       "       p = new %s(", buf());
        } else {
           fprintf(fp,       "       p = ::new %s(", buf());
        }
        if (m > 2) {
          fprintf(fp,       "\n");
        }
        // Copy in the arguments the caller gave us.
        for (k = 0; k < m; ++k) {
          ret = G__cppif_paratype(fp, ifn, ifunc, k);
          if (ret == 0) reg++;
          if (ret == 1) xmm++;
        }
        if (ifunc->ansi[ifn] == 2 && m==ifunc->para_nu[ifn]) {
          // Handle a variadic constructor (varying number of arguments).
#if defined(G__VAARG_COPYFUNC)
          fprintf(fp,       ", libp, %d", k);
#elif defined(__hpux)
          //FIXME:  This loops only 99 times, the other clause loops 100 times.
          int i;
          for (i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
            fprintf(fp,     ", G__va_arg_bufobj.x.i[%d]", i);
          }
#elif ((defined(__sparc) || defined(__i386)) && defined(__SUNPRO_CC)) || \
      ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
          int i;
          for (i = 0; i < 100; ++i) {
            fprintf(fp,     ", G__va_arg_bufobj.x.i[%d]", i);
          }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
          G__x8664_vararg_write(fp, xmm, reg);
#else
          fprintf(fp,       ", G__va_arg_bufobj");
#endif
        }
        fprintf(fp,         ");\n");
        fprintf(fp,         "     } else {\n");
        if (!has_a_new) {
           fprintf(fp,       "       p = new((void*) gvp) %s(", buf());
        } else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
           fprintf(fp,       "       p = new((void*) gvp) %s(", buf());
        } else {
           fprintf(fp,       "       p = ::new((void*) gvp) %s(", buf());
        }
        if (m > 2) {
          fprintf(fp,       "\n");
        }
        // Copy in the arguments the caller gave us.
        for (k = 0; k < m; ++k) {
          ret = G__cppif_paratype(fp, ifn, ifunc, k);
          if (ret == 0) reg++;
          if (ret == 1) xmm++;
        }
        if (ifunc->ansi[ifn] == 2 && m==ifunc->para_nu[ifn]) {
          // Handle a variadic constructor (varying number of arguments).
#if defined(G__VAARG_COPYFUNC)
          fprintf(fp,       ", libp, %d", k);
#elif defined(__hpux)
          //FIXME:  This loops only 99 times, the other clause loops 100 times.
          int i;
          for (i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
            fprintf(fp,     ", G__va_arg_bufobj.x.i[%d]", i);
          }
#elif ((defined(__sparc) || defined(__i386)) && defined(__SUNPRO_CC)) || \
      ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
          int i;
          for (i = 0; i < 100; ++i) {
            fprintf(fp,     ", G__va_arg_bufobj.x.i[%d]", i);
          }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
          G__x8664_vararg_write(fp, xmm, reg);
#else
          fprintf(fp,       ", G__va_arg_bufobj");
#endif
        }
        fprintf(fp,         ");\n");
        fprintf(fp,         "     }\n");
      }
      fprintf(fp,           "     break;\n");
      --m;
    } while ((m >= 0) && ifunc->param[ifn][m]->pdefault);
    fprintf(fp,             "   }\n");
  } else if (m > 0) {
    // Handle a constructor with arguments where none of them are defaulted.
    //
    // Note: We do not have to handle an array new here because initializers
    //       are not allowed for array new.
    fprintf(fp,             "   //m: %d\n", m);
    fprintf(fp,             "   if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
    if (!has_a_new) {
       fprintf(fp,             "     p = new %s(", buf());
    } else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
       fprintf(fp,           "     p = new %s(", buf());
    } else {
       fprintf(fp,           "     p = ::new %s(", buf());
    }
    if (m > 2) {
      fprintf(fp,           "\n");
    }
    for (k = 0; k < m; ++k) {
      ret = G__cppif_paratype(fp, ifn, ifunc, k);
      if (ret == 0) reg++;
      if (ret == 1) xmm++;
    }
    if (ifunc->ansi[ifn] == 2 && m==ifunc->para_nu[ifn]) {
      // handle a varadic constructor (varying number of arguments)
#if defined(G__VAARG_COPYFUNC)
      fprintf(fp,           ", libp, %d", k);
#elif defined(__hpux)
      //FIXME:  This loops only 99 times, the other clause loops 100 times.
      int i;
      for (i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
        fprintf(fp,         ", G__va_arg_bufobj.x.i[%d]", i);
      }
#elif ((defined(__sparc) || defined(__i386)) && defined(__SUNPRO_CC)) || \
      ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
      int i;
      for (i = 0; i < 100; ++i) {
        fprintf(fp,         ", G__va_arg_bufobj.x.i[%d]", i);
      }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
      G__x8664_vararg_write(fp, xmm, reg);
#else
      fprintf(fp,           ", G__va_arg_bufobj");
#endif
    }
    fprintf(fp,             ");\n");
    fprintf(fp,             "   } else {\n");
    if (!has_a_new) {
       fprintf(fp,           "     p = new((void*) gvp) %s(", buf());
    } else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
       fprintf(fp,           "     p = new((void*) gvp) %s(", buf());
    } else {
       fprintf(fp,           "     p = ::new((void*) gvp) %s(", buf());
    }
    if (m > 2) {
      fprintf(fp,           "\n");
    }
    for (k = 0; k < m; ++k) {
      ret = G__cppif_paratype(fp, ifn, ifunc, k);
      if (ret == 0) reg++;
      if (ret == 1) xmm++;
    }
    if (ifunc->ansi[ifn] == 2 && m==ifunc->para_nu[ifn]) {
      // handle a varadic constructor (varying number of arguments)
#if defined(G__VAARG_COPYFUNC)
      fprintf(fp,           ", libp, %d", k);
#elif defined(__hpux)
      //FIXME:  This loops only 99 times, the other clause loops 100 times.
      int i;
      for (i = G__VAARG_SIZE / sizeof(long) - 1; i > G__VAARG_SIZE / sizeof(long) - 100; --i) {
        fprintf(fp,         ", G__va_arg_bufobj.x.i[%d]", i);
      }
#elif ((defined(__sparc) || defined(__i386)) && defined(__SUNPRO_CC)) || \
      ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
      int i;
      for (i = 0; i < 100; ++i) {
        fprintf(fp,         ", G__va_arg_bufobj.x.i[%d]", i);
      }
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
      G__x8664_vararg_write(fp, xmm, reg);
#else
      fprintf(fp,           ", G__va_arg_bufobj");
#endif
    }
    fprintf(fp,             ");\n");
    fprintf(fp,             "   }\n");
  } else {
    // Handle a constructor with no arguments.
    //
    // Handle array new.
    fprintf(fp,             "   int n = G__getaryconstruct();\n");
    fprintf(fp,             "   if (n) {\n");
    if (isprotecteddtor) {
      fprintf(fp,           "     p = 0;\n");
      fprintf(fp,           "     G__genericerror(\"Error: Array construction with private/protected destructor is illegal\");\n");
    } else {
      fprintf(fp,           "     if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
      if (!has_a_new) {
         fprintf(fp,         "       p = new %s[n];\n", buf());
      } else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
         fprintf(fp,         "       p = new %s[n];\n", buf());
      } else {
         fprintf(fp,         "       p = ::new %s[n];\n", buf());
      }
      fprintf(fp,           "     } else {\n");
      if (!has_a_new) {
         fprintf(fp,         "       p = new((void*) gvp) %s[n];\n", buf());
      } else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
         fprintf(fp,         "       p = new((void*) gvp) %s[n];\n", buf());
      } else {
         fprintf(fp,         "       p = ::new((void*) gvp) %s[n];\n", buf());
      }
      fprintf(fp,           "     }\n");
    }
    fprintf(fp,             "   } else {\n");
    //
    // Handle regular new.
    fprintf(fp,             "     if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
    if (!has_a_new) {
       fprintf(fp,             "       p = new %s;\n", buf());
    } else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
       fprintf(fp,           "       p = new %s;\n", buf());
    } else {
       fprintf(fp,           "       p = ::new %s;\n", buf());
    }
    fprintf(fp,             "     } else {\n");
    if (!has_a_new) {
       fprintf(fp,           "       p = new((void*) gvp) %s;\n", buf());
    } else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
       fprintf(fp,           "       p = new((void*) gvp) %s;\n", buf());
    } else {
       fprintf(fp,           "       p = ::new((void*) gvp) %s;\n", buf());
    }
    fprintf(fp,             "     }\n");
    fprintf(fp,             "   }\n");
  }

  fprintf(fp,               "   result7->obj.i = (long) p;\n");
  fprintf(fp,               "   result7->ref = (long) p;\n");
  fprintf(fp,               "   G__set_tagnum(result7,G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(tagnum));

  G__if_ary_union_reset(ifn, ifunc);
  G__cppif_dummyfuncname(fp);

  fprintf(fp,               "}\n\n");
#endif // G__SMALLOBJECT
}

/**************************************************************************
* G__isprivateconstructorifunc()
*
**************************************************************************/
static int G__isprivateconstructorifunc(int tagnum,int iscopy)
{
  struct G__ifunc_table_internal *ifunc;
  int ifn;
  ifunc=G__struct.memfunc[tagnum];
  do {
    for(ifn=0;ifn<ifunc->allifunc;ifn++) {
      if(strcmp(G__struct.name[tagnum],ifunc->funcname[ifn])==0) {
        if(iscopy) { /* Check copy constructor */
          if((1<=ifunc->para_nu[ifn]&&'u'==ifunc->param[ifn][0]->type&&
              tagnum==ifunc->param[ifn][0]->p_tagtable) &&
             (1==ifunc->para_nu[ifn]||ifunc->param[ifn][1]->pdefault)
             && G__PRIVATE==ifunc->access[ifn]
             ) {
            return(1);
          }
        }
        else { /* Check default constructor */
          if((0==ifunc->para_nu[ifn]||ifunc->param[ifn][0]->pdefault)
             && G__PRIVATE==ifunc->access[ifn]
             ) {
            return(1);
          }
          /* Following solution may not be perfect */
          if((1<=ifunc->para_nu[ifn]&&'u'==ifunc->param[ifn][0]->type&&
              tagnum==ifunc->param[ifn][0]->p_tagtable) &&
             (1==ifunc->para_nu[ifn]||ifunc->param[ifn][1]->pdefault)
             &&G__PRIVATE==ifunc->access[ifn]
             ) {
            return(1);
          }
        }
      }
      else if(strcmp("operator new",ifunc->funcname[ifn])==0) {
        if(G__PRIVATE==ifunc->access[ifn]||G__PROTECTED==ifunc->access[ifn])
          return(1);
      }
    }
    ifunc=ifunc->next;
  } while(ifunc);
  return(0);
}

#ifndef __CINT__
static int G__isprivateconstructorclass G__P((int tagnum,int iscopy));
int G__isprivateconstructor G__P((int tagnum,int iscopy));
#endif

/**************************************************************************
* G__isprivateconstructorvar()
*
* check if private constructor exists in this particular class
**************************************************************************/
static int G__isprivateconstructorvar(int tagnum,int iscopy)
{
  int ig15;
  struct G__var_array *var;
  int memtagnum;
  var=G__struct.memvar[tagnum];
  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if('u'==var->type[ig15] 
         && -1!=(memtagnum=var->p_tagtable[ig15]) 
         && 'e'!=G__struct.type[memtagnum]
         && memtagnum!=tagnum
         && var->reftype[ig15]!=G__PARAREFERENCE
         ) {
        if(G__isprivateconstructorclass(memtagnum,iscopy)) return(1);
      }
    }
    var=var->next;
  }
  return(0);
}

/**************************************************************************
* G__isprivateconstructorclass()
*
* check if private constructor exists in this particular class
**************************************************************************/
static int G__isprivateconstructorclass(int tagnum, int iscopy)
{
  int t,f;
  if(iscopy) {
    t=G__CTORDTOR_PRIVATECOPYCTOR;
    f=G__CTORDTOR_NOPRIVATECOPYCTOR;
  }
  else {
    t=G__CTORDTOR_PRIVATECTOR;
    f=G__CTORDTOR_NOPRIVATECTOR;
  }
  if(G__ctordtor_status[tagnum]&t) return(1);
  if(G__ctordtor_status[tagnum]&f) return(0);
  if(G__isprivateconstructorifunc(tagnum,iscopy)||
     G__isprivateconstructor(tagnum,iscopy)
     ) {
    G__ctordtor_status[tagnum]|=t;
    return(1);
  }
  G__ctordtor_status[tagnum]|=f;
  return(0);
}

/**************************************************************************
* G__isprivateconstructor()
*
* check if private constructor exists in base class or class of member obj
**************************************************************************/
int G__isprivateconstructor(int tagnum, int iscopy)
{
  int basen;
  int basetagnum;
  struct G__inheritance *baseclass;

  baseclass = G__struct.baseclass[tagnum];

  /* Check base class private constructor */
  for(basen=0;basen<baseclass->basen;basen++) {
    basetagnum = baseclass->herit[basen]->basetagnum;
    if(G__isprivateconstructorclass(basetagnum,iscopy)) return(1);
  }

  /* Check Data member object */
  if(G__isprivateconstructorvar(tagnum,iscopy)) return(1);

  return(0);
}


/**************************************************************************
* G__isprivatedestructorifunc()
*
**************************************************************************/
static int G__isprivatedestructorifunc(int tagnum)
{
  char *dtorname;
  struct G__ifunc_table_internal *ifunc;
  int ifn;
  ifunc=G__struct.memfunc[tagnum];
  dtorname = (char*)malloc(strlen(G__struct.name[tagnum])+2);
  dtorname[0]='~';
  strcpy(dtorname+1,G__struct.name[tagnum]); // Okay, we allocated the right size
  do {
    for(ifn=0;ifn<ifunc->allifunc;ifn++) {
      if(strcmp(dtorname,ifunc->funcname[ifn])==0) {
        if(G__PRIVATE==ifunc->access[ifn]) {
          free((void*)dtorname);
          return(1);
        }
      }
      else if(strcmp("operator delete",ifunc->funcname[ifn])==0) {
        if(G__PRIVATE==ifunc->access[ifn]||G__PROTECTED==ifunc->access[ifn]) {
          free((void*)dtorname);
          return(1);
        }
      }
    }
    ifunc=ifunc->next;
  } while(ifunc);
  free((void*)dtorname);
  return(0);
}

#ifndef __CINT__
static int G__isprivatedestructorclass G__P((int tagnum));
int G__isprivatedestructor G__P((int tagnum));
#endif

/**************************************************************************
* G__isprivatedestructorvar()
*
* check if private destructor exists in this particular class
**************************************************************************/
static int G__isprivatedestructorvar(int tagnum)
{
  int ig15;
  struct G__var_array *var;
  int memtagnum;
  var=G__struct.memvar[tagnum];
  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if('u'==var->type[ig15] 
         && -1!=(memtagnum=var->p_tagtable[ig15]) 
         && 'e'!=G__struct.type[memtagnum]
         && memtagnum!=tagnum
         && var->reftype[ig15]!=G__PARAREFERENCE
         ) {
        if(G__isprivatedestructorclass(memtagnum)) return(1);
      }
    }
    var=var->next;
  }
  return(0);
}

/**************************************************************************
* G__isprivatedestructorclass()
*
* check if private destructor exists in this particular class
**************************************************************************/
static int G__isprivatedestructorclass(int tagnum)
{
  int t,f;
  t=G__CTORDTOR_PRIVATEDTOR;
  f=G__CTORDTOR_NOPRIVATEDTOR;
  if(G__ctordtor_status[tagnum]&t) return(1);
  if(G__ctordtor_status[tagnum]&f) return(0);
  if(G__isprivatedestructorifunc(tagnum)||
     G__isprivatedestructor(tagnum)
     ) {
    G__ctordtor_status[tagnum]|=t;
    return(1);
  }
  G__ctordtor_status[tagnum]|=f;
  return(0);
}
/**************************************************************************
* G__isprivatedestructor()
*
* check if private destructor exists in base class or class of member obj
**************************************************************************/
int G__isprivatedestructor(int tagnum)
{
  int basen;
  int basetagnum;
  struct G__inheritance *baseclass;

  baseclass = G__struct.baseclass[tagnum];

  /* Check base class private destructor */
  for(basen=0;basen<baseclass->basen;basen++) {
    basetagnum = baseclass->herit[basen]->basetagnum;
    if(G__isprivatedestructorclass(basetagnum)) {
      return(1);
    }
  }

  /* Check Data member object */
  if(G__isprivatedestructorvar(tagnum)) return(1);

  return(0);
}

#ifdef G__DEFAULTASSIGNOPR
/**************************************************************************
* G__isprivateassignoprifunc()
*
**************************************************************************/
static int G__isprivateassignoprifunc(int tagnum)
{
  struct G__ifunc_table_internal *ifunc;
  int ifn;
  ifunc=G__struct.memfunc[tagnum];
  do {
    for(ifn=0;ifn<ifunc->allifunc;ifn++) {
      if(strcmp("operator=",ifunc->funcname[ifn])==0) {
        if((G__PRIVATE==ifunc->access[ifn]||G__PROTECTED==ifunc->access[ifn])
           && 'u'==ifunc->param[ifn][0]->type
           && tagnum==ifunc->param[ifn][0]->p_tagtable
            ) {
          return(1);
        }
      }
    }
    ifunc=ifunc->next;
  } while(ifunc);
  return(0);
}

#ifndef __CINT__
static int G__isprivateassignoprclass G__P((int tagnum));
int G__isprivateassignopr G__P((int tagnum));
#endif

/**************************************************************************
* G__isprivateassignoprvar()
*
* check if private assignopr exists in this particular class
**************************************************************************/
static int G__isprivateassignoprvar(int tagnum)
{
  int ig15;
  struct G__var_array *var;
  int memtagnum;
  var=G__struct.memvar[tagnum];
  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if('u'==var->type[ig15] 
         && -1!=(memtagnum=var->p_tagtable[ig15])
         && 'e'!=G__struct.type[memtagnum]
         && memtagnum!=tagnum
         && var->reftype[ig15]!=G__PARAREFERENCE
         ) {
        if(G__isprivateassignoprclass(memtagnum)) return(1);
      }
      if(G__PARAREFERENCE==var->reftype[ig15] &&
        G__LOCALSTATIC!=var->statictype[ig15]) {
        return(1);
      }
      if(var->constvar[ig15] &&
        G__LOCALSTATIC!=var->statictype[ig15]) {
        return(1);
      }
    }
    var=var->next;
  }
  return(0);
}

/**************************************************************************
* G__isprivateassignoprclass()
*
* check if private assignopr exists in this particular class
**************************************************************************/
static int G__isprivateassignoprclass(int tagnum)
{
  int t,f;
  t=G__CTORDTOR_PRIVATEASSIGN;
  f=G__CTORDTOR_NOPRIVATEASSIGN;
  if(G__ctordtor_status[tagnum]&t) return(1);
  if(G__ctordtor_status[tagnum]&f) return(0);
  if(G__isprivateassignoprifunc(tagnum)||G__isprivateassignopr(tagnum)) {
    G__ctordtor_status[tagnum]|=t;
    return(1);
  }
  G__ctordtor_status[tagnum]|=f;
  return(0);
}
/**************************************************************************
* G__isprivateassignopr()
*
* check if private assignopr exists in base class or class of member obj
**************************************************************************/
int G__isprivateassignopr(int tagnum)
{
  int basen;
  int basetagnum;
  struct G__inheritance *baseclass;

  baseclass = G__struct.baseclass[tagnum];

  /* Check base class private assignopr */
  for(basen=0;basen<baseclass->basen;basen++) {
    basetagnum = baseclass->herit[basen]->basetagnum;
    if(G__isprivateassignoprclass(basetagnum)) {
      return(1);
    }
  }

  /* Check Data member object */
  if(G__isprivateassignoprvar(tagnum)) return(1);

  return(0);
}
#endif


/**************************************************************************
* G__cppif_gendefault()
*
* Create default constructor and destructor. If default constructor is
* given in the header file, the interface function created here for
* the default constructor will be redundant and won't be used.
*
* Copy constructor and operator=(), if not explisitly specified in the
* header file, are handled as memberwise copy by cint parser. Validity of
* this handling is questionalble especially when base class has explicit
* copy constructor or operator=().
*
**************************************************************************/
void G__cppif_gendefault(FILE *fp, FILE* /*hfp*/, int tagnum,
                         int ifn, G__ifunc_table_internal* ifunc,
                         int isconstructor, int iscopyconstructor,
                         int isdestructor,
                         int isassignmentoperator, int isnonpublicnew)
{
#ifndef G__SMALLOBJECT
  /* int k,m; */
  int page;
  G__FastAllocString funcname(G__MAXNAME);
  G__FastAllocString temp(G__MAXNAME);
  G__FastAllocString dtorname(G__MAXNAME);
  int isprotecteddtor = G__isprotecteddestructoronelevel(tagnum);

  G__ASSERT( tagnum != -1 );

  if('n'==G__struct.type[tagnum]) return;

  page = ifunc->page;
  if(ifn>=G__MAXIFUNC) {
    ifn=0;
    ++page;
  }

  /*********************************************************************
  * default constructor
  *********************************************************************/

  if (!isconstructor) {
    isconstructor = G__isprivateconstructor(tagnum, 0);
  }

#ifdef G__NOSTUBS
  int isconstused = 0;
#endif

  if (!isconstructor && !G__struct.isabstract[tagnum] && !isnonpublicnew) {
#ifdef G__NOSTUBS
    if(G__dicttype==kNoWrappersDictionary){
      // index in the ifunc
      long pifn;
      long poffset;

      // Single Constructor has only a parameter. The size of the object
      G__param para_cons;
      para_cons.paran = 0;

      // We look for the "new operator" ifunc in the current class and in its bases
      G__ifunc_table_internal * cons_oper = G__get_methodhandle4(G__struct.name[tagnum], &para_cons, G__struct.memfunc[tagnum], &pifn, &poffset, 0, 1,0);

      // Look for it in the ifunc table is it was already create in make_default_ifunc
      if(cons_oper && !(!cons_oper->mangled_name[pifn] /*&& cons_oper->funcptr[pifn]!=(void*)-1*/))
        page = cons_oper->page;
    }
#endif

#ifdef G__NOSTUBS
    if(G__dicttype==kFunctionSymbols){
      // 01-11-07
      // Force the outlining of functions even if the weren't declared
      // (CInt will try to declare them later on anyways)

      if(G__is_tagnum_safe(tagnum)) {
        // if we didn't force the outlining for the default constructor 	 
        // we have to do it here because we need the object created there 	 
        // to be able to use a copy constructor 
        if ( !(!isconstructor && !G__struct.isabstract[tagnum] && !isnonpublicnew) ) { 	 
          // I don't know how to get the pointer to a constructor so the only thing 	 
          // I can think of to use the default constructor is to create an object 	 
	  	 
          // index in the ifunc 	 
          long pifn; 	 
          long poffset; 	 
	  	 
          // Single Constructor has only a parameter. The size of the object 	 
          G__param para_new; 	 
          para_new.paran = 0; 	 
	  	 
          // We look for the "new operator" ifunc in the current class and in its bases 	 
          G__ifunc_table_internal* new_oper = G__get_methodhandle4(G__struct.name[tagnum], &para_new, G__struct.memfunc[tagnum], &pifn, &poffset, 0, 1,0);
      
          if(new_oper && (!isconstructor && !G__struct.isabstract[tagnum] && !isnonpublicnew))
            isconstused = 1;
        }

        // I don't know how to get the pointer to a constructor so the only thing
        // I can think of to use the default constructor is to create an object
        if(isconstused)
          fprintf(fp,"  %s G__cons_%s;\n", G__fulltagname(tagnum, 0),  G__map_cpp_funcname(tagnum, funcname, ifn, page));
      }
    }
    else
#endif
    {
       G__FastAllocString buf(G__fulltagname(tagnum, 1));

      funcname = G__struct.name[tagnum];
      fprintf(fp,         "// automatic default constructor\n");

#ifdef G__CPPIF_STATIC
      fprintf(fp,         "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, ifn, page));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
      fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, funcname, ifn, page), ++G__nexports);
#endif
      fprintf(hfp,        "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, funcname, ifn, page));
      fprintf(fp,         "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, ifn, page));
#endif /* G__CPPIF_STATIC */
      fprintf(fp,         "\n{\n");
      fprintf(fp,         "   %s *p;\n", G__fulltagname(tagnum, 1));

      fprintf(fp,         "   char* gvp = (char*) G__getgvp();\n");

      bool has_a_new = G__struct.funcs[tagnum] & (G__HAS_OPERATORNEW1ARG | G__HAS_OPERATORNEW2ARG);
      bool has_a_new1arg = G__struct.funcs[tagnum] & G__HAS_OPERATORNEW1ARG;
      bool has_a_new2arg = G__struct.funcs[tagnum] & G__HAS_OPERATORNEW2ARG;

      bool has_own_new1arg = false;
      bool has_own_new2arg = false;

      {
        struct G__ifunc_table* iref = G__get_ifunc_ref(G__struct.memfunc[tagnum]);
        struct G__ifunc_table* ireffound = 0;
        long index;
        long offset;
        ireffound = G__get_methodhandle("operator new", "size_t", iref, &index, &offset, 0, 0);
        has_own_new1arg = (ireffound != 0);
        ireffound = G__get_methodhandle("operator new", "size_t, void*", iref, &index, &offset, 0, 0);
        has_own_new2arg = (ireffound != 0);
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
      } else {
        fprintf(fp,       "     if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
        if (!has_a_new) {
           fprintf(fp,     "       p = new %s[n];\n", buf());
        } else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
           fprintf(fp,     "       p = new %s[n];\n", buf());
        } else {
           fprintf(fp,     "       p = ::new %s[n];\n", buf());
        }
        fprintf(fp,       "     } else {\n");
        if (!has_a_new) {
           fprintf(fp,     "       p = new((void*) gvp) %s[n];\n", buf());
        } else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
           fprintf(fp,     "       p = new((void*) gvp) %s[n];\n", buf());
        } else {
           fprintf(fp,     "       p = ::new((void*) gvp) %s[n];\n", buf());
        }
        fprintf(fp,       "     }\n");
      }
      fprintf(fp,         "   } else {\n");
      //
      // Handle regular new.
      fprintf(fp,         "     if ((gvp == (char*)G__PVOID) || (gvp == 0)) {\n");
      if (!has_a_new) {
         fprintf(fp,       "       p = new %s;\n", buf());
      } else if (has_a_new1arg && (has_own_new1arg || !has_own_new2arg)) {
         fprintf(fp,       "       p = new %s;\n", buf());
      } else {
         fprintf(fp,       "       p = ::new %s;\n", buf());
      }
      fprintf(fp,         "     } else {\n");
      if (!has_a_new) {
         fprintf(fp,       "       p = new((void*) gvp) %s;\n", buf());
      } else if (has_a_new2arg && (has_own_new2arg || !has_own_new1arg)) {
         fprintf(fp,       "       p = new((void*) gvp) %s;\n", buf());
      } else {
         fprintf(fp,       "       p = ::new((void*) gvp) %s;\n", buf());
      }
      fprintf(fp,         "     }\n");
      fprintf(fp,         "   }\n");

      fprintf(fp,         "   result7->obj.i = (long) p;\n");
      fprintf(fp,         "   result7->ref = (long) p;\n");
      fprintf(fp,         "   G__set_tagnum(result7,G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(tagnum));
 
      G__cppif_dummyfuncname(fp);

      fprintf(fp,         "}\n\n");

      ++ifn;
      if (ifn >= G__MAXIFUNC) {
        ifn = 0;
        ++page;
      }
    }
  } /* if (isconstructor) */

  /*********************************************************************
   * copy constructor
   *********************************************************************/

  if (!iscopyconstructor) {
    iscopyconstructor = G__isprivateconstructor(tagnum, 1);
  }

  if (!iscopyconstructor && !G__struct.isabstract[tagnum] && !isnonpublicnew) {
#ifdef G__NOSTUBS
    if(G__dicttype==kNoWrappersDictionary){
      // index in the ifunc
      long pifn;
      long poffset;

      // Single Constructor has only a parameter. The size of the object
      G__param para_cons;
      para_cons.paran = 1;
      para_cons.para[0].typenum = 0;
      para_cons.para[0].type = 'u';
      para_cons.para[0].tagnum = tagnum;

      // We look for the "new operator" ifunc in the current class and in its bases
      G__ifunc_table_internal * cons_oper = G__get_methodhandle4(G__struct.name[tagnum], &para_cons, G__struct.memfunc[tagnum], &pifn, &poffset, 0, 1,0);

      if(cons_oper && !(!cons_oper->mangled_name[pifn] /*&& cons_oper->funcptr[pifn]!=(void*)-1*/))
        page = cons_oper->page;
    }
#endif

#ifdef G__NOSTUBS
    if(G__dicttype==kFunctionSymbols) {
      // 01-11-07
      // Force the outlining of functions even if the weren't declared
      // (CInt will try to declare them later on anyways)

      if(G__is_tagnum_safe(tagnum)) {
        // if we didn't force the outlining for the default constructor 	 
        // we have to do it here because we need the object created there 	 
        // to be able to use a copy constructor 
        if (!(!isconstructor && !G__struct.isabstract[tagnum] && !isnonpublicnew)) { 	 
          // I don't know how to get the pointer to a constructor so the only thing 	 
          // I can think of to use the default constructor is to create an object 	 
	  	 
          // index in the ifunc 	 
          long pifn; 	 
          long poffset; 	 
	  	 
          // Single Constructor has only a parameter. The size of the object 	 
          G__param para_new; 	 
          para_new.paran = 0; 	 
	  	 
          // We look for the "new operator" ifunc in the current class and in its bases 	 
          G__ifunc_table_internal* new_oper = G__get_methodhandle4(G__struct.name[tagnum], &para_new, G__struct.memfunc[tagnum], &pifn, &poffset, 0, 1,0);
      
          if(!isconstused && new_oper && (!isconstructor && !G__struct.isabstract[tagnum] && !isnonpublicnew)){
            fprintf(fp, "  %s G__cons_%s;\n", G__fulltagname(tagnum, 0),  G__map_cpp_funcname(tagnum, funcname, ifn, page));
            isconstused = 1;
          }
        }
	  	 
        if(isconstused)
          fprintf(fp,"  %s G__copycons_%s(G__cons_%s);\n", G__fulltagname(tagnum, 0), G__map_cpp_funcname(tagnum, funcname, ifn, page),G__map_cpp_funcname(tagnum, funcname, ifn, page));
      }
    }
    else
#endif // G__NOSTUBS
    {
      funcname = G__struct.name[tagnum];

      fprintf(fp,     "// automatic copy constructor\n");

#ifdef G__CPPIF_STATIC
      fprintf(fp,     "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)\n", G__map_cpp_funcname(tagnum, funcname, ifn, page));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
      fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, funcname, ifn, page), ++G__nexports);
#endif
      fprintf(hfp,    "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, funcname, ifn, page));
      fprintf( fp,    "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)\n", G__map_cpp_funcname(tagnum, funcname, ifn, page));
#endif /* G__CPPIF_STATIC */

      fprintf(fp,     "\n{\n");
      fprintf(fp,     "   %s* p;\n", G__fulltagname(tagnum, 1));

      temp = G__fulltagname(tagnum, 1);

      fprintf(fp,     "   void* tmp = (void*) G__int(libp->para[0]);\n");
      fprintf(fp,     "   p = new %s(*(%s*) tmp);\n", temp(), temp());

      fprintf(fp,     "   result7->obj.i = (long) p;\n");
      fprintf(fp,     "   result7->ref = (long) p;\n");
      fprintf(fp,     "   G__set_tagnum(result7,G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(tagnum));

      G__cppif_dummyfuncname(fp);

      fprintf(fp,     "}\n\n");

      ++ifn;
      if (ifn >= G__MAXIFUNC) {
        ifn = 0;
        ++page;
      }
    }
  }
  
  /*********************************************************************
   * destructor
   *********************************************************************/

  // 06-7-07
  // Dont generate the wrappers for the general case of the destructors
  // do it only for all those stranges classes we dont support yet
  if (0 >= isdestructor && G__dicttype==kCompleteDictionary) {
    isdestructor = G__isprivatedestructor(tagnum);
  }

  if ((0 >= isdestructor) && (G__struct.type[tagnum] != 'n')) {
#ifdef G__NOSTUBS
    if(G__dicttype==kNoWrappersDictionary){
      // index in the ifunc
      long pifn;
      long poffset;
      G__FastAllocString funcname(G__MAXNAME*6);
      funcname.Format("~%s", G__struct.name[tagnum]);

      // Single Constructor has only a parameter. The size of the object
      G__param para_des;
      para_des.paran = 0;

      // We look for the "new operator" ifunc in the current class and in its bases
      G__ifunc_table_internal * des_oper = G__get_methodhandle4(funcname, &para_des, G__struct.memfunc[tagnum], &pifn, &poffset, 0, 1,0);

      // Look for it in the ifunc table is it was already create in make_default_ifunc
      if(des_oper && !(!des_oper->mangled_name[pifn] /*&& des_oper->funcptr[pifn]!=(void*)-1*/) && G__is_tagnum_safe(tagnum))
        page = des_oper->page;
    }
#endif

    int isdestdefined = 1;
    if(!G__struct.memfunc[tagnum]->mangled_name[0])
      isdestdefined = 0;



#ifdef G__NOSTUBS
    if(G__dicttype==kFunctionSymbols) {
      // 01-11-07
      // Force the outlining of functions even if the weren't declared
      // (CInt will try to declare them later on anyways)

      //if(G__is_tagnum_safe(tagnum))
      // 28-01-08
      // How can we force the destructor symbol to be included?
      // is it enough with the object creation?
    }
    else
#endif
    {
       G__FastAllocString buf(G__fulltagname(tagnum, 1));

      bool has_a_delete = G__struct.funcs[tagnum] & G__HAS_OPERATORDELETE;

      bool has_own_delete1arg = false;
      bool has_own_delete2arg = false;

      {
        struct G__ifunc_table* iref = G__get_ifunc_ref(G__struct.memfunc[tagnum]);
        struct G__ifunc_table* ireffound = 0;
        long index;
        long offset;
        ireffound = G__get_methodhandle("operator delete", "void*", iref, &index, &offset, 0, 0);
        has_own_delete1arg = (ireffound != 0);
        ireffound = G__get_methodhandle("operator delete", "void*, size_t", iref, &index, &offset, 0, 0);
        has_own_delete2arg = (ireffound != 0);
      }

      funcname.Format("~%s", G__struct.name[tagnum]);
      dtorname.Format("G__T%s", G__map_cpp_name(G__fulltagname(tagnum, 0)));

      fprintf(fp,"// automatic destructor\n");
      fprintf(fp, "typedef %s %s;\n", G__fulltagname(tagnum, 0), dtorname());

#ifdef G__CPPIF_STATIC
      fprintf(fp,"static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, ifn, page));
#else /* G__CPPIF_STATIC */
#ifdef G__GENWINDEF
      fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, funcname, ifn, page), ++G__nexports);
#endif
      fprintf(hfp,"extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, funcname, ifn, page));
      fprintf( fp,"extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, ifn, page));
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
      fprintf(fp,   "       delete[] (%s*) soff;\n", buf());
      fprintf(fp,   "     } else {\n");
      fprintf(fp,   "       G__setgvp((long) G__PVOID);\n");
      fprintf(fp,   "       for (int i = n - 1; i >= 0; --i) {\n");
      fprintf(fp,   "         ((%s*) (soff+(sizeof(%s)*i)))->~%s();\n", buf(), buf(), dtorname());
      fprintf(fp,   "       }\n");
      fprintf(fp,   "       G__setgvp((long)gvp);\n");
      fprintf(fp,   "     }\n");
      fprintf(fp,   "   } else {\n");
      fprintf(fp,   "     if (gvp == (char*)G__PVOID) {\n");
      //fprintf(fp, "       G__operator_delete((void*) soff);\n");
      fprintf(fp,   "       delete (%s*) soff;\n", buf());
      fprintf(fp,   "     } else {\n");
      fprintf(fp,   "       G__setgvp((long) G__PVOID);\n");
      fprintf(fp,   "       ((%s*) (soff))->~%s();\n", buf(), dtorname());
      fprintf(fp,   "       G__setgvp((long)gvp);\n");
      fprintf(fp,   "     }\n");
      fprintf(fp,   "   }\n");

      fprintf(fp,   "   G__setnull(result7);\n");

      G__cppif_dummyfuncname(fp);

      fprintf(fp,   "}\n\n");

      ++ifn;
      if (ifn >= G__MAXIFUNC) {
        ifn = 0;
        ++page;
      }
    }
  }

#ifdef G__DEFAULTASSIGNOPR

  /*********************************************************************
   * assignment operator
   *********************************************************************/

  if (!isassignmentoperator) {
    isassignmentoperator = G__isprivateassignopr(tagnum);
  }

  if (!isassignmentoperator) {
#ifdef G__NOSTUBS
    if(G__dicttype==kNoWrappersDictionary){
      // index in the ifunc
      long pifn;
      long poffset;

      // Single Constructor has only a parameter. The size of the object
      G__param para_op;
      para_op.paran = 1;
      para_op.para[0].typenum = 0;
      para_op.para[0].type = 'u';
      para_op.para[0].tagnum = tagnum;
      
      // We look for the "new operator" ifunc in the current class and in its bases
      G__ifunc_table_internal * op_oper = G__get_methodhandle4("operator=", &para_op, G__struct.memfunc[tagnum], &pifn, &poffset, 0, 1,0);

      if(op_oper && !(!op_oper->mangled_name[pifn] /*&& op_oper->funcptr[pifn]!=(void*)-1*/))
        page = op_oper->page;
    }
#endif

#ifdef G__NOSTUBS
    if(G__dicttype==kFunctionSymbols) {
      // 01-11-07
      // Force the outlining of functions even if the weren't declared
      // (CInt will try to declare them later on anyways)

      if(G__is_tagnum_safe(tagnum)) {
        funcname = "operator=";
        fprintf(fp,"%s& (%s::*G__assignop_%s)(const %s&) = &%s::operator=;\n", G__fulltagname(tagnum, 0), G__fulltagname(tagnum, 0), G__map_cpp_funcname(tagnum, funcname, ifn, page), G__fulltagname(tagnum, 0), G__fulltagname(tagnum, 0) );
        fprintf(fp," (void)(G__assignop_%s);\n", G__map_cpp_funcname(tagnum, funcname, ifn, page));
      }
    }
    else
#endif
    {
      funcname = "operator=";
      fprintf(fp,   "// automatic assignment operator\n");

#ifdef G__CPPIF_STATIC
      fprintf(fp,   "static int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, ifn, page));
#else /* G__CPPIF_STATIC */

#ifdef G__GENWINDEF
      fprintf(G__WINDEFfp,"        %s @%d\n", G__map_cpp_funcname(tagnum, funcname, ifn, page),++G__nexports);
#endif
      fprintf(hfp,  "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, funcname, ifn, page));
      fprintf( fp,  "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, funcname, ifn, page));
#endif /* G__CPPIF_STATIC */
      fprintf(fp,   "\n{\n");
      temp = G__type2string('u', tagnum, -1, 0, 0);
      fprintf(fp,   "   %s* dest = (%s*) G__getstructoffset();\n", temp(), temp());
      if ((1 >= G__struct.size[tagnum]) && (0 == G__struct.memvar[tagnum]->allvar)) {
      } else {
         fprintf(fp, "   *dest = *(%s*) libp->para[0].ref;\n", temp());
      }
      fprintf(fp,   "   const %s& obj = *dest;\n", temp());
      fprintf(fp,   "   result7->ref = (long) (&obj);\n");
      fprintf(fp,   "   result7->obj.i = (long) (&obj);\n");
      G__cppif_dummyfuncname(fp);
      fprintf(fp,   "}\n\n");

      ++ifn;
      if (ifn >= G__MAXIFUNC) {
        ifn = 0;
        ++page;
      }
    }
  }
#endif

  //
#endif // G__SMALLOBJECT
}


/**************************************************************************
 * G__method_inbase()
 * This function will search for the method ifn (index in ifunc) 
 * in the ifunc->tagnum's base classes
 * RETURN -> NULL Method not found
 *           NOT NULL Method Found. Method's ifunc table pointer
 *
 *
 * 06-08-08
 * Changed to return the page instead of just not null
 * (since 0 means not found then the index starts at 1)
 **************************************************************************/
int G__method_inbase(int ifn, G__ifunc_table_internal *ifunc)
{
  // tagnum's Base Classes structure
  G__inheritance* cbases = G__struct.baseclass[ifunc->tagnum];

  // If there are still base classes
  if (cbases){
    // Go through the base tagnums (tagnum = index in G__struct structure)
    for (int idx=0; idx < cbases->basen; ++idx){
      // Current tagnum
      int basetagnum=cbases->herit[idx]->basetagnum;

      // Current tagnum's ifunc table
      G__ifunc_table_internal * ifunct = G__struct.memfunc[basetagnum];
      
      // Continue if there are still ifuncs and the method 'ifn' is not found yet
      if (ifunct){
        int base=-1;

        // Does the Method 'ifn' (in ifunc) exist in the current ifunct?
        ifunct = G__ifunc_exist(ifunc, ifn, ifunct, &base, 0xffff);
        
        //If the number of default parameters numbers is different between the base and the derived
        //class we generete the stub
        if (base!=-1 && ifunct){
          int derived_def_n = -1;
          
          // Counting derived class default parameters
          for(int i = ifunc->para_nu[ifn] - 1; i >= 0; --i)
            if (ifunc->param[ifn][i]->def)
              derived_def_n = i;
            else break;

          //Counting base class default parameters
          if (derived_def_n != -1
              && !ifunct->param[base][derived_def_n]->def)
            return 0;

          return ifunct->page+1;
        }
      }
    }
  }
  return 0;
}

/**************************************************************************
 * G__method_inbase2()
 *
 * 16-11-07
 * -1 now means than it was found in two or more parents...
 * the association can not be done here... we have to
 * do it at runtime
 * if "onlyparents" then we will check only the direct ascendants, not
 * the full hierarchy
 **************************************************************************///
int G__method_inbase2(int ifn, G__ifunc_table_internal *ifunc, int onlyparents)
{
  int page_base = 0; // the result... 0 if not found (the index otehrwise)
  int found = 0;

  // tagnum's Base Classes structure
  G__inheritance* cbases = G__struct.baseclass[ifunc->tagnum];

  // If there are still base classes
  if (cbases){
    // Go through the base tagnums (tagnum = index in G__struct structure)
    for (int idx=0; idx < cbases->basen; ++idx){
      // Current tagnum
      int basetagnum=cbases->herit[idx]->basetagnum;

      // Do it only for its parents
      if( (onlyparents && cbases->herit[idx]->property&G__ISDIRECTINHERIT) || !onlyparents) {
        // Current tagnum's ifunc table
        G__ifunc_table_internal * ifunct = G__struct.memfunc[basetagnum];

        // Continue if there are still ifuncs and the method 'ifn' is not found yet
        if (ifunct){
          int base=-1;

          // Does the Method 'ifn' (in ifunc) exist in the current ifunct?
          ifunct = G__ifunc_exist(ifunc, ifn, ifunct, &base, 0xffff);

          //If the number of default parameters numbers is different between the base and the derived
          //class we generete the stub
          if (base!=-1 && ifunct) {
	    page_base = G__method_inbase2(ifn, ifunct, onlyparents);
	    if(page_base!=0) // if page_base==-1 it means it found the same method in different parents
	      ++found;
          }
        }
      }
    }
  }

  if(!found) {
    if(onlyparents)
      page_base = G__method_inbase2(ifn, ifunc, 0);
    
    if(!page_base) {
      ifunc->page_base = ifunc->page+1;
      page_base = ifunc->page_base;
    }
    return page_base; // not found
  }
  
  if (found>1 && onlyparents)
    return -1; // found in multiple parents

  return page_base;   
}

/**************************************************************************
 * G__cppif_genfunc()
 *
 **************************************************************************/
void G__cppif_genfunc(FILE *fp, FILE * /* hfp */, int tagnum, int ifn, G__ifunc_table_internal *ifunc)
{

  // If the virtual method 'ifn' (in ifunc) exists in any Base Clase then we have
  // an overridden virtual method,so the stub function for it is not generated
  if (G__dicttype!=kNoWrappersDictionary && (ifunc->isvirtual[ifn]) && (G__method_inbase(ifn, ifunc)))
    return;

#ifndef G__SMALLOBJECT
  int k, m, ret, xmm = 0, reg = 0;

  G__FastAllocString endoffunc(G__LONGLINE);
  G__FastAllocString castname(G__ONELINE);

#ifdef G__CPPIF_EXTERNC
  G__p2f_typedef(fp, ifn, ifunc) ;
#endif

#ifdef G__VAARG_COPYFUNC
  if ((ifunc->ansi[ifn] == 2) && (ifunc->pentry[ifn]->line_number > 0)) {
    G__va_arg_copyfunc(fp, ifunc, ifn);
  }
#endif

#ifndef G__CPPIF_STATIC
#ifdef G__GENWINDEF
  fprintf(G__WINDEFfp, "        %s @%d\n", G__map_cpp_funcname(tagnum, ifunc->funcname[ifn], ifn, ifunc->page), ++G__nexports);
#endif
  if (G__globalcomp == G__CPPLINK) {
    fprintf(hfp, "extern \"C\" int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash);\n", G__map_cpp_funcname(tagnum, ifunc->funcname[ifn], ifn, ifunc->page));
  } else {
    fprintf(hfp, "int %s();\n", G__map_cpp_funcname(tagnum, ifunc->funcname[ifn], ifn, ifunc->page));
  }
#endif // G__CPPIF_STATIC

#ifdef G__CPPIF_STATIC
  fprintf(fp, "static ");
#else
  if (G__globalcomp == G__CPPLINK) {
    fprintf(fp, "extern \"C\" ");
  }
#endif

  if (G__clock) {
    /* K&R style header */
    fprintf(fp, "int %s(result7, funcname, libp, hash)\n", G__map_cpp_funcname(tagnum, ifunc->funcname[ifn], ifn, ifunc->page));
    fprintf(fp, "G__value* result7;\n");
    fprintf(fp, "char* funcname;\n");
    fprintf(fp, "struct G__param* libp;\n");
    fprintf(fp, "int hash;\n");
  } else {
    /* ANSI style header */
    fprintf(fp, "int %s(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)", G__map_cpp_funcname(tagnum, ifunc->funcname[ifn], ifn, ifunc->page));
  }

  fprintf(fp, "\n{\n");

  G__if_ary_union(fp,ifn,ifunc);

  if (-1 != tagnum) {
    if ((ifunc->access[ifn] == G__PROTECTED) || ((ifunc->access[ifn] == G__PRIVATE) && (G__struct.protectedaccess[tagnum] & G__PRIVATEACCESS))) {
       castname.Format("%s_PR", G__get_link_tagname(tagnum));
    } else {
      castname = G__fulltagname(tagnum, 1);
    }
  }

#ifndef G__VAARG_COPYFUNC
  if (ifunc->ansi[ifn] == 2) {
#if defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
    G__x8664_vararg(fp, ifn, ifunc, ifunc->funcname[ifn], tagnum, castname);
#else
    fprintf(fp, "   G__va_arg_buf G__va_arg_bufobj;\n");
    fprintf(fp, "   G__va_arg_put(&G__va_arg_bufobj, libp, %d);\n", ifunc->para_nu[ifn]);
#endif
  }
#endif

  /*************************************************************
  * compact G__cpplink.C
  *************************************************************/
  m = ifunc->para_nu[ifn] ;
  if ((m > 0) && ifunc->param[ifn][m-1]->pdefault) {
    // Handle a function with parameters, some of which have defaults.
    fprintf(fp, "   switch (libp->paran) {\n");
    do {
      // One case for each possible number of supplied parameters.
      fprintf(fp, "   case %d:\n", m);
      //
      // Output the return type.
      G__cppif_returntype(fp, ifn, ifunc, endoffunc);
      //
      // Output the function name.
      if (-1 != tagnum) {
        if ('n' == G__struct.type[tagnum]) {
          fprintf(fp,"%s::", G__fulltagname(tagnum, 1));
        } else {
          if (ifunc->staticalloc[ifn]) {
             fprintf(fp, "%s::", castname());
          } else {
            reg++;
            if (ifunc->isconst[ifn] & G__CONSTFUNC) {
               fprintf(fp, "((const %s*) G__getstructoffset())->", castname());
            } else {
               fprintf(fp, "((%s*) G__getstructoffset())->", castname());
            }
          }
        }
      }
      if ((ifunc->access[ifn] == G__PROTECTED) || ((ifunc->access[ifn] == G__PRIVATE) && (G__struct.protectedaccess[tagnum] & G__PRIVATEACCESS))) {
        fprintf(fp, "G__PT_%s(", ifunc->funcname[ifn]);
      } else {
        fprintf(fp, "%s(", ifunc->funcname[ifn]);
      }
      //
      // Output the parameters.
      if (m > 6) {
        fprintf(fp, "\n");
      }
      for (k = 0; k < m; ++k) {
         ret = G__cppif_paratype(fp, ifn, ifunc, k);
         if (ret == 0) reg++;
         if (ret == 1) xmm++;
      }
      if (ifunc->ansi[ifn] == 2 && m==ifunc->para_nu[ifn]) {
#if defined(G__VAARG_COPYFUNC)
        fprintf(fp, ", libp, %d", k);
#elif defined(__hpux)
        //FIXME:  This loops only 99 times, the other clause loops 100 times.
        int i;
        for (i = G__VAARG_SIZE/sizeof(long) - 1; i > G__VAARG_SIZE/sizeof(long) - 100; i--)
          fprintf(fp, ", G__va_arg_bufobj.x.i[%d]", i);
#elif ((defined(__sparc) || defined(__i386)) && defined(__SUNPRO_CC)) || \
      ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
        int i;
        for (i = 0; i < 100; i++) fprintf(fp, ", G__va_arg_bufobj.x.i[%d]", i);
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
        G__x8664_vararg_write(fp, xmm, reg);
#else
        fprintf(fp, ", G__va_arg_bufobj");
#endif
      }
      //
      // Output the function body.
      fprintf(fp, ")%s\n", endoffunc());
      //
      // End the case for m number of parameters given.
      fprintf(fp, "      break;\n");
      --m;
    } while ((m >= 0) && ifunc->param[ifn][m]->pdefault);
    //
    // End of switch on number of parameters provided by call.
    fprintf(fp, "   }\n");
  } else {
    // Handle a function with parameters, none of which have defaults.
    //
    // Output the return type.
    G__cppif_returntype(fp, ifn, ifunc, endoffunc);

    //
    // Output the function name.
    if (-1 != tagnum) {
      if (G__struct.type[tagnum] == 'n')
        fprintf(fp, "%s::", G__fulltagname(tagnum, 1));
      else {
        if (ifunc->staticalloc[ifn]) {
           fprintf(fp, "%s::", castname());
        } else {
          reg++;
          if (ifunc->isconst[ifn] & G__CONSTFUNC) {
             fprintf(fp, "((const %s*) G__getstructoffset())->", castname());
          } else {
             fprintf(fp,"((%s*) G__getstructoffset())->", castname());
          }
        }
      }
    }
    if ((ifunc->access[ifn] == G__PROTECTED) || ((ifunc->access[ifn] == G__PRIVATE) && (G__struct.protectedaccess[tagnum] & G__PRIVATEACCESS))) {
      fprintf(fp, "G__PT_%s(", ifunc->funcname[ifn]);
    } else {
       // we need to convert A::operator T() to A::operator ::T, or
       // the context will be the one of tagnum, i.e. A::T instead of ::T
       if (tolower(ifunc->type[ifn]) == 'u'
           && !strncmp(ifunc->funcname[ifn], "operator ", 8)
           && (isalpha(ifunc->funcname[ifn][9]) || ifunc->funcname[ifn][9] == '_')) {
          if (!strncmp(ifunc->funcname[ifn] + 9, "const ", 6))
             fprintf(fp, "operator const ::%s(", ifunc->funcname[ifn] + 15);
          else
             fprintf(fp, "operator ::%s(", ifunc->funcname[ifn] + 9);
       } else
          fprintf(fp, "%s(", ifunc->funcname[ifn]);
    }
    //
    // Output the parameters.
    if (m > 6) {
      fprintf(fp, "\n");
    }
    for (k = 0; k < m; k++) {
      ret = G__cppif_paratype(fp, ifn, ifunc, k);
      if (ret == 0) reg++;
      if (ret == 1) xmm++;
    }
    if (ifunc->ansi[ifn] == 2 && m==ifunc->para_nu[ifn]) {
#if defined(G__VAARG_COPYFUNC)
      fprintf(fp, ", libp, %d", k);
#elif defined(__hpux)
      //FIXME:  This loops only 99 times, the other clause loops 100 times.
      int i;
      for (i = G__VAARG_SIZE/sizeof(long) - 1; i > G__VAARG_SIZE/sizeof(long) - 100; --i) fprintf(fp, ", G__va_arg_bufobj.x.i[%d]", i);
#elif ((defined(__sparc) || defined(__i386)) && defined(__SUNPRO_CC)) || \
      ((defined(__PPC__) || defined(__ppc__)) && (defined(_AIX) || defined(__APPLE__)))
      int i;
      for (i = 0; i < 100; i++) fprintf(fp, ", G__va_arg_bufobj.x.i[%d]", i);
#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || defined(__sun))
      G__x8664_vararg_write(fp, xmm, reg);
#else
      fprintf(fp, ", G__va_arg_bufobj");
#endif
    }
    //
    // Output the function body.
    fprintf(fp, ")%s\n", endoffunc());
  }

  G__if_ary_union_reset(ifn, ifunc);
  G__cppif_dummyfuncname(fp);
  fprintf(fp,"}\n\n");

  //
#endif // G__SMALLOBJECT
}

} // extern "C"

/**************************************************************************
* G__cppif_returntype()
*
**************************************************************************/
int G__cppif_returntype(FILE *fp, int ifn, G__ifunc_table_internal *ifunc, G__FastAllocString& endoffunc)
{
#ifndef G__SMALLOBJECT
  int type, tagnum, typenum, reftype, isconst;
#ifndef G__OLDIMPLEMENTATION1503
  int deftyp = -1;
#endif
  char *typestring;
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
  char *ptr;
#endif
  const char* indent = "      ";
  type = ifunc->type[ifn];
  tagnum = ifunc->p_tagtable[ifn];
  typenum = ifunc->p_typetable[ifn];
  reftype = ifunc->reftype[ifn];
  isconst = ifunc->isconst[ifn];

  /* Promote link-off typedef to link-on if used in function */
  // 09-10-07
  // This is just wrong... wrong!
  // A functions that 'just' prints a value should NOT change its internal settings...
  // As a result this field has to be changed by hand at a non-specified part of the
  // file when this method is not called...
  if ((typenum != -1) && (G__newtype.globalcomp[typenum] == G__NOLINK) && (G__newtype.iscpplink[typenum] == G__NOLINK)) {
    G__newtype.globalcomp[typenum] = G__globalcomp;
  }

#ifdef G__OLDIMPLEMENTATION1859 /* questionable with 1859 */
  /* return type is a reference */
  if ((typenum != -1) && (G__newtype.reftype[typenum] == G__PARAREFERENCE)) {
    reftype = G__PARAREFERENCE;
    typenum = -1;
  }
#endif

  // Function return type is a reference, handle and return.
  if (reftype == G__PARAREFERENCE) {
    fprintf(fp, "%s{\n", indent);
    if (isconst & G__CONSTFUNC) {
      if (isupper(type)) {
        isconst |= G__PCONSTVAR;
      } else {
        isconst |= G__CONSTVAR;
      }
    }
    typestring = G__type2string(type, tagnum, typenum, reftype, isconst);
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
    // For old Microsoft compilers, replace "long long" with " __int64 ".
    ptr = strstr(typestring, "long long");
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
    } else {
      // Reference to a pointer or to a const object.
      // Note:  The type string already has the ampersand in it.
      fprintf(fp, "%s   %s obj = ", indent, typestring);
    }
    if ((typenum != -1) && G__newtype.nindex[typenum]) {
       endoffunc.Format(";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.i = (long) (obj);\n%s}", indent, indent, indent);
      return 0;
    }
    switch (type) {
      case 'd':
      case 'f':
         endoffunc.Format(";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.d = (double) (obj);\n%s}", indent, indent, indent);
        break;
      case 'u':
        if (G__struct.type[tagnum] == 'e') {
           endoffunc.Format(";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.i = (long) (obj);\n%s}", indent, indent, indent);
        } else {
           endoffunc.Format(";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.i = (long) (&obj);\n%s}", indent, indent, indent);
        }
        break;
      default:
         endoffunc.Format(";\n%s   result7->ref = (long) (&obj);\n%s   G__letint(result7, '%c', (long)obj);\n%s}", indent, indent, type, indent);
        break;
    }
    return 0;
  }

  // Function return type is a pointer, handle and return.
  if (isupper(type)) {
    fprintf(fp, "%sG__letint(result7, %d, (long) ", indent, type);
    endoffunc = ");";
    return(0);
  }

  // Function returns an object or a fundamental type.
  switch (type) {
    case 'y':
      fprintf(fp, "%s", indent);
      endoffunc.Format(";\n%sG__setnull(result7);", indent);
      return 0;
    case '1':
      fprintf(fp, "%sG__letint(result7, %d, (long) ", indent, type);
      endoffunc = ");";
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
      endoffunc = ");";
      return 0;
    case 'n':
      fprintf(fp, "%sG__letLonglong(result7, %d, (G__int64) ", indent, type);
      endoffunc = ");";
      return 0;
    case 'm':
      fprintf(fp, "%sG__letULonglong(result7, %d, (G__uint64) ", indent, type);
      endoffunc = ");";
      return 0;
    case 'q':
      fprintf(fp, "%sG__letLongdouble(result7, %d, (long double) ", indent, type);
      endoffunc = ");";
      return 0;
    case 'f':
    case 'd':
      fprintf(fp, "%sG__letdouble(result7, %d, (double) ", indent, type);
      endoffunc = ");";
      return 0;
    case 'u':
      switch (G__struct.type[tagnum]) {
        case 'a':
           G__class_autoloading(&tagnum);
           // After attempting the autoloading, processing as a class.
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
              endoffunc.Format(";\n%s   result7->ref = (long) (&obj);\n%s   result7->obj.i = (long) (&obj);\n%s}", indent, indent, indent);
           } else {
              if (G__globalcomp == G__CPPLINK) {
                 fprintf(fp, "%s{\n", indent);
                 if (isconst & G__CONSTFUNC) {
                    fprintf(fp, "%s   const %s* pobj;\n", indent, G__type2string('u', tagnum, deftyp, 0, 0));
                    fprintf(fp, "%s   const %s xobj = ", indent, G__type2string('u', tagnum, deftyp, 0, 0));
                 } else {
                    fprintf(fp, "%s   %s* pobj;\n", indent, G__type2string('u', tagnum, deftyp, 0, 0));
                    fprintf(fp, "%s   %s xobj = ", indent, G__type2string('u', tagnum, deftyp, 0, 0));
                 }
                 endoffunc.Format(";\n"
                                  "%s   pobj = new %s(xobj);\n"
                                  "%s   result7->obj.i = (long) ((void*) pobj);\n"
                                  "%s   result7->ref = result7->obj.i;\n"
                                  "%s   G__store_tempobject(*result7);\n"
                                  "%s}", indent, G__type2string('u', tagnum, deftyp, 0, 0), indent, indent, indent, indent);
              } else {
                 fprintf(fp, "%sG__alloc_tempobject_val(result7);\n", indent);
                 fprintf(fp, "%sresult7->obj.i = G__gettempbufpointer();\n", indent);
                 fprintf(fp, "%sresult7->ref = G__gettempbufpointer();\n", indent);
                 fprintf(fp, "%s*((%s *) result7->obj.i) = ", indent, G__type2string(type, tagnum, typenum, reftype, 0));
                 endoffunc = ";";
              }
           }
           break;
	 default:
      fprintf(fp, "%sG__letint(result7, %d, (long) ", indent, type);
      endoffunc = ");";
      break;
    }
    return 0;
  } // switch(type)
  return 1; /* never happen, avoiding lint error */
#endif // G__SMALLOBJECT
}

extern "C" {

/**************************************************************************
* G__cppif_paratype()
*
**************************************************************************/
int G__cppif_paratype(FILE *fp, int ifn, G__ifunc_table_internal *ifunc, int k)
{
#ifndef G__SMALLOBJECT
   // returns 1 when paratype was double (will go in X8664 xmm register)
   // 0 otherwise
   int retval = 0;
   char type = ifunc->param[ifn][k]->type;
   int tagnum = ifunc->param[ifn][k]->p_tagtable;
   int typenum = ifunc->param[ifn][k]->p_typetable;
   int reftype = ifunc->param[ifn][k]->reftype;
   int isconst = ifunc->param[ifn][k]->isconst;
   // Promote link-off typedef to link-on if used in function.
   if (
      (typenum != -1) &&
      (G__newtype.globalcomp[typenum] == G__NOLINK) &&
      (G__newtype.iscpplink[typenum] == G__NOLINK)
   ) {
      G__newtype.globalcomp[typenum] = G__globalcomp;
   }
   if (k && !(k % 2)) {
      fprintf(fp, "\n");
   }
   if (k) {
      fprintf(fp, ", ");
   }
   if (ifunc->param[ifn][k]->name) {
      char* p = strchr(ifunc->param[ifn][k]->name, '[');
      if (p) {
         fprintf(fp, "G__Ap%d->a", k);
         return 0;
      }
   }
   if (
      // --
#ifndef G__OLDIMPLEMENTATION2191
      (type != '1') && (type != 'a')
#else
      (type != 'Q') && (type != 'a')
#endif
      // --
   ) {
      switch (reftype) {
         case G__PARANORMAL:
            if ((-1 != typenum) && (G__PARAREFERENCE == G__newtype.reftype[typenum])) {
               reftype = G__PARAREFERENCE;
               typenum = -1;
            }
            else {
               break;
            }
         case G__PARAREFERENCE:
            if (islower(type)) {
               switch (type) {
                  case 'u':
                     fprintf(fp, "*(%s*) libp->para[%d].ref", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
#ifndef G__OLDIMPLEMENTATION1167
                  case 'd':
                     fprintf(fp, "*(%s*) G__Doubleref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'l':
                     fprintf(fp, "*(%s*) G__Longref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'i':
                     if (tagnum == -1) { // int
                        fprintf(fp, "*(%s*) G__Intref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     }
                     else { // enum type
                        fprintf(fp, "*(%s*) libp->para[%d].ref", G__type2string(type, tagnum, typenum, 0, 0), k);
                     }
                     break;
                  case 's':
                     fprintf(fp, "*(%s*) G__Shortref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'c':
                     fprintf(fp, "*(%s*) G__Charref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'h':
                     fprintf(fp, "*(%s*) G__UIntref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'r':
                     fprintf(fp, "*(%s*) G__UShortref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'b':
                     fprintf(fp, "*(%s*) G__UCharref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'k':
                     fprintf(fp, "*(%s*) G__ULongref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'n':
                     fprintf(fp, "*(%s*) G__Longlongref(&libp->para[%d])",
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
                             "__int64",
#else
                             G__type2string(type, tagnum, typenum, 0, 0),
#endif
                     k);
                     break;
                  case 'm':
                     fprintf(fp, "*(%s*) G__ULonglongref(&libp->para[%d])",
#if defined(_MSC_VER) && (_MSC_VER < 1310) /*vc6 and vc7.0*/
                             "unsigned __int64",
#else
                             G__type2string(type, tagnum, typenum, 0, 0),
#endif
                     k);
                     break;
                  case 'q':
                     fprintf(fp, "*(%s*) G__Longdoubleref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'g':
                     fprintf(fp, "*(%s*) G__Boolref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
                  case 'f':
                     fprintf(fp, "*(%s*) G__Floatref(&libp->para[%d])", G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
#else
                  case 'd':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mdouble(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'l':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mlong(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'i':
                     if (tagnum == -1) { // int
                        fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mint(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     }
                     else { // enum type
                        fprintf(fp, "*(%s*) libp->para[%d].ref", G__type2string(type, tagnum, typenum, 0, 0), k);
                     }
                     break;
                  case 's':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mshort(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'c':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mchar(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'h':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Muint(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'r':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mushort(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'b':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Muchar(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'k':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mulong(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'n':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mlong long(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'm':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mulonglong(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'q':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mlong double(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
                  case 'g':
#ifdef G__BOOL4BYTE
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mint(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
#else
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Muchar(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
#endif
                     break;
                  case 'f':
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : G__Mfloat(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, k);
                     break;
#endif
                  default:
                     fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : (%s) G__int(libp->para[%d])", k, G__type2string(type, tagnum, typenum, 0, 0), k, G__type2string(type, tagnum, typenum, 0, 0), k);
                     break;
               }
            }
            else {
               if ((typenum != -1) && isupper(G__newtype.type[typenum])) {
                  /* This part is not perfect. Cint data structure bug.
                   * typedef char* value_type;
                   * void f(value_type& x);  // OK
                   * void f(value_type x);   // OK
                   * void f(value_type* x);  // OK
                   * void f(value_type*& x); // bad
                   *  reference and pointer to pointer can not happen at once */
                  fprintf(
                       fp
                     , "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : *(%s*) (void*) (&G__Mlong(libp->para[%d]))"
                     , k
                     , G__type2string(type, tagnum, typenum, 0, isconst & G__CONSTVAR)
                     , k
                     , G__type2string(type, tagnum, typenum, 0, isconst & G__CONSTVAR)
                     , k
                  );
                  /* above is , in fact, not good. G__type2string returns pointer to
                   * static buffer. This relies on the fact that the 2 calls are
                   * identical */
               }
               else {
                  fprintf(
                       fp
                     , "libp->para[%d].ref ? *(%s) libp->para[%d].ref : *(%s) (void*) (&G__Mlong(libp->para[%d]))"
                     , k
                     , G__type2string(type, tagnum, typenum, 2, isconst&G__CONSTVAR)
                     , k
                     , G__type2string(type, tagnum, typenum, 2, isconst&G__CONSTVAR)
                     , k
                  );
                  /* above is , in fact, not good. G__type2string returns pointer to
                   * static buffer. This relies on the fact that the 2 calls are
                   * identical */
               }
            }
            return 0;
         case G__PARAREFP2P:
         case G__PARAREFP2P2P:
            reftype = G__PLVL(reftype);
            if ((typenum != -1) && isupper(G__newtype.type[typenum])) {
               fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : *(%s*) (void*) (&G__Mlong(libp->para[%d]))"
                       , k, G__type2string(type, tagnum, typenum, reftype, isconst)
                       , k, G__type2string(type, tagnum, typenum, reftype, isconst) , k);
            }
            else {
               fprintf(fp, "libp->para[%d].ref ? *(%s*) libp->para[%d].ref : *(%s*) (void*) (&G__Mlong(libp->para[%d]))"
                       , k, G__type2string(type, tagnum, typenum, reftype, isconst)
                       , k, G__type2string(type, tagnum, typenum, reftype, isconst), k);
            }
            return 0;
         case G__PARAP2P:
            G__ASSERT(isupper(type));
            fprintf(fp, "(%s) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isconst), k);
            return 0;
         case G__PARAP2P2P:
            G__ASSERT(isupper(type));
            fprintf(fp, "(%s) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isconst), k);
            return 0;
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
         fprintf(fp, "(%s) G__int(libp->para[%d])", G__type2string(type, tagnum, typenum, reftype, isupper(type) ? (isconst & ~(G__PCONSTVAR | G__PCONSTCONSTVAR)) : isconst), k);
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
         retval = 1;
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
   return retval;
#endif // G__SMALLOBJECT
   // --
}

/**************************************************************************
**************************************************************************
* Generate C++ symbol binding routine calls
**************************************************************************
**************************************************************************/

/**************************************************************************
* G__cpplink_tagtable()
*
**************************************************************************/
void G__cpplink_tagtable(FILE *fp, FILE *hfp)
{
#ifndef G__SMALLOBJECT
  int i;
  G__FastAllocString tagname(G__MAXNAME*8);
  G__FastAllocString mappedtagname(G__MAXNAME*6);
  G__FastAllocString buf(G__ONELINE);

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Class,struct,union,enum tag information setup\n");
  fprintf(fp,"*********************************************************/\n");

  if(G__CPPLINK == G__globalcomp) {
    G__cpplink_linked_taginfo(fp,hfp);
    fprintf(fp,"extern \"C\" void G__cpp_setup_tagtable%s() {\n",G__DLLID);
  }
  else {
    G__cpplink_linked_taginfo(fp,hfp);
    fprintf(fp,"void G__c_setup_tagtable%s() {\n",G__DLLID);
  }

  fprintf(fp,"\n   /* Setting up class,struct,union tag entry */\n");
  for(i=0;i<G__struct.alltag;i++) {
    if(
       (G__struct.hash[i] || 0==G__struct.name[i][0]) &&
       (G__CPPLINK==G__struct.globalcomp[i]
        ||G__CLINK==G__struct.globalcomp[i]
        ||G__ONLYMETHODLINK==G__struct.globalcomp[i]
        )) {
      if(!G__nestedclass) {
        if(0<=G__struct.parent_tagnum[i] &&
           -1!=G__struct.parent_tagnum[G__struct.parent_tagnum[i]])
          continue;
        if(G__CLINK==G__struct.globalcomp[i] && -1!=G__struct.parent_tagnum[i])
          continue;
      }

      if(-1==G__struct.line_number[i]) {
        /* Philippe and Fons's request to display this */
        if(G__dispmsg>= G__DISPERR /*G__DISPNOTE*/) {
          if(G__NOLINK==G__struct.iscpplink[i]) {
            G__fprinterr(G__serr,"Note: Link requested for undefined class %s (ignore this message)"
                         ,G__fulltagname(i,1));
          }
          else {
            G__fprinterr(G__serr,
                         "Note: Link requested for already precompiled class %s (ignore this message)"
                         ,G__fulltagname(i,1));
          }
          G__printlinenum();
        }
        /* G__genericerror((char*)NULL); */
      }

      G__getcommentstring(buf,i,&G__struct.comment[i]);

      tagname = G__fulltagname(i,0);
      if(-1!=G__struct.line_number[i]
         && (-1==G__struct.parent_tagnum[i]||G__nestedclass)
         ) {
        if('e'==G__struct.type[i])
          fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),sizeof(%s),%d,%d,%s,NULL,NULL);\n"
                  ,G__mark_linked_tagnum(i) ,"int" ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                  ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                  +G__struct.rootflag[i]*0x10000
#else
                  ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                  ,buf());
        else if('n'==G__struct.type[i]) {
          mappedtagname = G__map_cpp_name(tagname);
          fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),0,%d,%d,%s,G__setup_memvar%s,G__setup_memfunc%s);\n"
                  ,G__mark_linked_tagnum(i)
                  /* ,G__type2string('u',i,-1,0,0) */
                  ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                  ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                  +G__struct.rootflag[i]*0x10000
#else
                  ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                  ,buf(),mappedtagname(),mappedtagname());
        }
        else if(0==G__struct.name[i][0]) {
          mappedtagname = G__map_cpp_name(tagname);
          if(G__CPPLINK==G__globalcomp) {
            fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),%s,%d,%d,%s,G__setup_memvar%s,G__setup_memfunc%s);\n"
                    ,G__mark_linked_tagnum(i)
                    ,"0" /* G__type2string('u',i,-1,0,0) */
                    ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                    +G__struct.rootflag[i]*0x10000
#else
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                    ,buf() ,mappedtagname(),mappedtagname());
          }
          else {
            fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),%s,%d,%d,%s,G__setup_memvar%s,NULL);\n"
                    ,G__mark_linked_tagnum(i)
                    ,"0" /* G__type2string('u',i,-1,0,0) */
                    ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                    +G__struct.rootflag[i]*0x10000
#else
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                    ,buf(),mappedtagname());
          }
        }
        else {
          mappedtagname = G__map_cpp_name(tagname);
          if(G__CPPLINK==G__globalcomp && '$'!=G__struct.name[i][0]) {
            if(G__ONLYMETHODLINK==G__struct.globalcomp[i])
              fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),sizeof(%s),%d,%d,%s,NULL,G__setup_memfunc%s);\n"
                      ,G__mark_linked_tagnum(i)
                      ,G__type2string('u',i,-1,0,0)
                      ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                      ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                      +G__struct.rootflag[i]*0x10000
#else
                      ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                      ,buf(),mappedtagname());
            else
            if(G__suppress_methods)
              fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),sizeof(%s),%d,%d,%s,G__setup_memvar%s,NULL);\n"
                      ,G__mark_linked_tagnum(i)
                      ,G__type2string('u',i,-1,0,0)
                      ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                      ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                      +G__struct.rootflag[i]*0x10000
#else
                      ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                      ,buf(),mappedtagname());
            else
              fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),sizeof(%s),%d,%d,%s,G__setup_memvar%s,G__setup_memfunc%s);\n"
                      ,G__mark_linked_tagnum(i)
                      ,G__type2string('u',i,-1,0,0)
                      ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                      ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                      +G__struct.rootflag[i]*0x10000
#else
                      ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                      ,buf(),mappedtagname(),mappedtagname());
          }
          else if('$'==G__struct.name[i][0]&&
                  G__defined_typename(G__struct.name[i]+1)>0&&
                  isupper(G__newtype.type[G__defined_typename(G__struct.name[i]+1)])) {
            fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),sizeof(%s),%d,%d,%s,NULL,NULL);\n"
                    ,G__mark_linked_tagnum(i)
                    ,G__type2string('u',i,-1,0,0)
                    ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                    +G__struct.rootflag[i]*0x10000
#else
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                    ,buf());
          }
          else {
            fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),sizeof(%s),%d,%d,%s,G__setup_memvar%s,NULL);\n"
                    ,G__mark_linked_tagnum(i)
                    ,G__type2string('u',i,-1,0,0)
                    ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                    +G__struct.rootflag[i]*0x10000
#else
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                    ,buf(),mappedtagname());
          }

        }
      }
      else {
        fprintf(fp,"   G__tagtable_setup(G__get_linked_tagnum_fwd(&%s),0,%d,%d,%s,NULL,NULL);\n"
                ,G__mark_linked_tagnum(i)
                ,G__globalcomp
#if  !defined(G__OLDIMPLEMENTATION1545)
                    ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
                    +G__struct.rootflag[i]*0x10000
#else
                ,G__struct.isabstract[i]+G__struct.funcs[i]*0x100
#endif
                ,buf());
      }
      if('e'!=G__struct.type[i]) {
        if(strchr(tagname,'<')) { /* template class */
           fprintf(hfp,"typedef %s G__%s;\n",tagname(),G__map_cpp_name(tagname));
        }
      }
    }
    else if((G__struct.hash[i] || 0==G__struct.name[i][0]) &&
            (G__CPPLINK-2)==G__struct.globalcomp[i]) {
      fprintf(fp,"   G__get_linked_tagnum_fwd(&%s);\n" ,G__mark_linked_tagnum(i));
    }
  }

  fprintf(fp,"}\n");
#endif
}

#ifdef G__VIRTUALBASE
/**************************************************************************
* G__vbo_funcname()
*
**************************************************************************/
static char* G__vbo_funcname(int tagnum, int basetagnum, int basen)
{
  static G__FastAllocString result(G__LONGLINE);
  G__FastAllocString temp(G__LONGLINE);
  temp = G__map_cpp_name(G__fulltagname(tagnum,1));
  result.Format("G__2vbo_%s_%s_%d",temp()
                ,G__map_cpp_name(G__fulltagname(basetagnum,1)),basen);
  return(result);
}

/**************************************************************************
* G__cpplink_inheritance()
*
**************************************************************************/
void G__cppif_inheritance(FILE *fp)
{
  int i;
  int basen;
  int basetagnum;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* virtual base class offset calculation interface\n");
  fprintf(fp,"*********************************************************/\n");

  fprintf(fp,"\n   /* Setting up class inheritance */\n");
  for(i=0;i<G__struct.alltag;i++) {
    if(G__NOLINK>G__struct.globalcomp[i]&&
       (-1==(int)G__struct.parent_tagnum[i]
        || G__nestedclass
        )
       && -1!=G__struct.line_number[i]&&G__struct.hash[i]&&
       ('$'!=G__struct.name[i][0])) {
      switch(G__struct.type[i]) {
      case 'c': /* class */
      case 's': /* struct */
        if(G__struct.baseclass[i]->basen>0) {
          for(basen=0;basen<G__struct.baseclass[i]->basen;basen++) {
            if(G__PUBLIC!=G__struct.baseclass[i]->herit[basen]->baseaccess ||
               0==(G__struct.baseclass[i]->herit[basen]->property&G__ISVIRTUALBASE))
              continue;
            basetagnum=G__struct.baseclass[i]->herit[basen]->basetagnum;
            fprintf(fp,"static long %s(long pobject) {\n"
                    ,G__vbo_funcname(i,basetagnum,basen));
            G__FastAllocString temp(G__fulltagname(i,1));
            fprintf(fp,"  %s *G__Lderived=(%s*)pobject;\n",temp(),temp());
            fprintf(fp,"  %s *G__Lbase=G__Lderived;\n",G__fulltagname(basetagnum,1));
            fprintf(fp,"  return((long)G__Lbase-(long)G__Lderived);\n");
            fprintf(fp,"}\n\n");
          }
        }
        break;
      default: /* enum */
        break;
      }
    } /* if() */
  } /* for(i) */
}
#endif

/**************************************************************************
* G__cpplink_inheritance()
*
**************************************************************************/
void G__cpplink_inheritance(FILE *fp)
{
#ifndef G__SMALLOBJECT
  int i;
  int basen;
  int basetagnum;
  G__FastAllocString temp(G__MAXNAME*6);
  int flag;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Inheritance information setup/\n");
  fprintf(fp,"*********************************************************/\n");

  if(G__CPPLINK == G__globalcomp) {
    fprintf(fp,"extern \"C\" void G__cpp_setup_inheritance%s() {\n",G__DLLID);
  }
  else  {
  }

  fprintf(fp,"\n   /* Setting up class inheritance */\n");
  for(i=0;i<G__struct.alltag;i++) {
    if(G__NOLINK>G__struct.globalcomp[i]&&
       (-1==(int)G__struct.parent_tagnum[i]
       || G__nestedclass
        )
       && -1!=G__struct.line_number[i]&&G__struct.hash[i]&&
       ('$'!=G__struct.name[i][0])) {
      switch(G__struct.type[i]) {
      case 'c': /* class */
      case 's': /* struct */
        if(G__struct.baseclass[i]->basen>0) {
          fprintf(fp,"   if(0==G__getnumbaseclass(G__get_linked_tagnum(&%s))) {\n"
                  ,G__get_link_tagname(i));
          flag=0;
          for(basen=0;basen<G__struct.baseclass[i]->basen;basen++) {
            if(0==(G__struct.baseclass[i]->herit[basen]->property&G__ISVIRTUALBASE))
              ++flag;
          }
          if(flag) {
            fprintf(fp,"     %s *G__Lderived;\n",G__fulltagname(i,0));
            fprintf(fp,"     G__Lderived=(%s*)0x1000;\n",G__fulltagname(i,1));
          }
          for(basen=0;basen<G__struct.baseclass[i]->basen;basen++) {
            basetagnum=G__struct.baseclass[i]->herit[basen]->basetagnum;
            fprintf(fp,"     {\n");
#ifdef G__VIRTUALBASE
            temp = G__mark_linked_tagnum(basetagnum);
            if(G__struct.baseclass[i]->herit[basen]->property&G__ISVIRTUALBASE) {
              G__FastAllocString temp2(G__vbo_funcname(i,basetagnum,basen));
              fprintf(fp,"       G__inheritance_setup(G__get_linked_tagnum(&%s),G__get_linked_tagnum(&%s),(long)%s,%d,%ld);\n"
                      ,G__mark_linked_tagnum(i)
                      ,temp()
                      ,temp2()
                      ,G__struct.baseclass[i]->herit[basen]->baseaccess
                      ,(long)G__struct.baseclass[i]->herit[basen]->property
                      );
            }
            else {
              int basen2,flag2=0;
              for(basen2=0;basen2<G__struct.baseclass[i]->basen;basen2++) {
                if(basen2!=basen &&
                   (G__struct.baseclass[i]->herit[basen]->basetagnum
                    == G__struct.baseclass[i]->herit[basen2]->basetagnum) &&
                   ((G__struct.baseclass[i]->herit[basen]->property&G__ISVIRTUALBASE)
                    ==0 ||
                    (G__struct.baseclass[i]->herit[basen2]->property&G__ISVIRTUALBASE)
                    ==0 )) {
                  flag2=1;
                }
              }
              temp = G__fulltagname(basetagnum,1);
              if(!flag2)
                fprintf(fp,"       %s *G__Lpbase=(%s*)G__Lderived;\n"
                        ,temp(),G__fulltagname(basetagnum,1));
              else {
                G__fprinterr(G__serr,
                             "Warning: multiple ambiguous inheritance %s and %s. Cint will not get correct base object address\n"
                             ,temp(),G__fulltagname(i,1));
                fprintf(fp,"       %s *G__Lpbase=(%s*)((long)G__Lderived);\n"
                        ,temp(),G__fulltagname(basetagnum,1));
              }
              temp = G__mark_linked_tagnum(basetagnum);
              fprintf(fp,"       G__inheritance_setup(G__get_linked_tagnum(&%s),G__get_linked_tagnum(&%s),(long)G__Lpbase-(long)G__Lderived,%d,%ld);\n"
                      ,G__mark_linked_tagnum(i)
                      ,temp()
                      ,G__struct.baseclass[i]->herit[basen]->baseaccess
                      ,(long)G__struct.baseclass[i]->herit[basen]->property
                      );
            }
#else
            temp = G__fulltagname(basetagnum,1);
            if(G__struct.baseclass[i]->herit[basen]->property&G__ISVIRTUALBASE) {
              fprintf(fp,"       %s *pbase=(%s*)0x1000;\n"
                      ,temp(),G__fulltagname(basetagnum,1));
            }
            else {
              fprintf(fp,"       %s *pbase=(%s*)G__Lderived;\n"
                      ,temp(),G__fulltagname(basetagnum,1));
            }
            temp = G__mark_linked_tagnum(basetagnum);
            fprintf(fp,"       G__inheritance_setup(G__get_linked_tagnum(&%s),G__get_linked_tagnum(&%s),(long)pbase-(long)G__Lderived,%d,%ld);\n"
                    ,G__mark_linked_tagnum(i)
                    ,temp()
                    ,G__struct.baseclass[i]->herit[basen]->baseaccess
                    ,G__struct.baseclass[i]->herit[basen]->property
                    );
#endif
            fprintf(fp,"     }\n");
          }
          fprintf(fp,"   }\n");
        }
        break;
      default: /* enum */
        break;
      }
    } /* if() */
  } /* for(i) */

  fprintf(fp,"}\n");
#endif
}

/**************************************************************************
* G__cpplink_typetable()
*
**************************************************************************/
void G__cpplink_typetable(FILE *fp, FILE *hfp)
{
  int i;
  int j;
  G__FastAllocString temp(G__ONELINE);
  char *p;
  G__FastAllocString buf(G__ONELINE);


  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* typedef information setup/\n");
  fprintf(fp,"*********************************************************/\n");

  if(G__CPPLINK == G__globalcomp) {
    fprintf(fp,"extern \"C\" void G__cpp_setup_typetable%s() {\n",G__DLLID);
  }
  else {
    fprintf(fp,"void G__c_setup_typetable%s() {\n",G__DLLID);
  }

  fprintf(fp,"\n   /* Setting up typedef entry */\n");
  for(i=0;i<G__newtype.alltype;i++) {
    if(G__NOLINK>G__newtype.globalcomp[i]) {
      if(!(G__newtype.parent_tagnum[i] == -1 ||
           (G__nestedtypedef &&
            (G__struct.globalcomp[G__newtype.parent_tagnum[i]]<G__NOLINK
#define G__OLDIMPLEMENTATION1830
             )
            )))
        continue;
      if(strncmp("G__p2mf",G__newtype.name[i],7)==0 &&
         G__CPPLINK==G__globalcomp){
        G__ASSERT(i>0);
        temp = G__newtype.name[i-1];
        p = strstr(temp,"::*");
        *(p+3)='\0';
        fprintf(hfp,"typedef %s%s)%s;\n",temp(),G__newtype.name[i],p+4);
      }
      if('u'==tolower(G__newtype.type[i]))
        fprintf(fp,"   G__search_typename2(\"%s\",%d,G__get_linked_tagnum(&%s),%d,"
                ,G__newtype.name[i]
                ,G__newtype.type[i]
                ,G__mark_linked_tagnum(G__newtype.tagnum[i])
#if !defined(G__OLDIMPLEMENTATION1861)
                ,G__newtype.reftype[i] | (G__newtype.isconst[i]*0x100)
#else
                ,G__newtype.reftype[i] & (G__newtype.isconst[i]*0x100)
#endif
                );
      else
        fprintf(fp,"   G__search_typename2(\"%s\",%d,-1,%d,"
                ,G__newtype.name[i]
                ,G__newtype.type[i]
#if !defined(G__OLDIMPLEMENTATION1861)
                ,G__newtype.reftype[i] | (G__newtype.isconst[i]*0x100)
#else
                ,G__newtype.reftype[i] & (G__newtype.isconst[i]*0x100)
#endif
                );
      if(G__newtype.parent_tagnum[i] == -1)
        fprintf(fp,"-1);\n");
      else
        fprintf(fp,"G__get_linked_tagnum(&%s));\n"
               ,G__mark_linked_tagnum(G__newtype.parent_tagnum[i]));

      if(-1!=G__newtype.comment[i].filenum) {
        G__getcommenttypedef(temp,&G__newtype.comment[i],i);
        if(temp[0]) G__add_quotation(temp,buf);
        else buf = "NULL";
      }
      else buf = "NULL";
      if(G__newtype.nindex[i]>G__MAXVARDIM) {
        /* This is just a work around */
        G__fprinterr(G__serr,"CINT INTERNAL ERROR? typedef %s[%d] 0x%lx\n"
                ,G__newtype.name[i],G__newtype.nindex[i]
                ,(long)G__newtype.index[i]);
        G__newtype.nindex[i] = 0;
        if(G__newtype.index[i]) free((void*)G__newtype.index[i]);
      }
      fprintf(fp,"   G__setnewtype(%d,%s,%d);\n",G__globalcomp,buf()
              ,G__newtype.nindex[i]);
      if(G__newtype.nindex[i]) {
        for(j=0;j<G__newtype.nindex[i];j++) {
          fprintf(fp,"   G__setnewtypeindex(%d,%d);\n"
                  ,j,G__newtype.index[i][j]);
        }
      }

    }
  }
  fprintf(fp,"}\n");
}

/**************************************************************************
* G__hascompiledoriginalbase()
*
**************************************************************************/
/* unused:
static int G__hascompiledoriginalbase(int tagnum)
{
  struct G__ifunc_table_internal *memfunc;
  struct G__inheritance *baseclass = G__struct.baseclass[tagnum];
  int basen,ifn;
  for(basen=0;basen<baseclass->basen;basen++) {
    if(G__CPPLINK!=G__struct.iscpplink[baseclass->herit[basen]->basetagnum])
      continue;
    memfunc=G__struct.memfunc[baseclass->herit[basen]->basetagnum];
    while(memfunc) {
      for(ifn=0;ifn<memfunc->allifunc;ifn++) {
        if(memfunc->isvirtual[ifn]) return(1);
      }
      memfunc=memfunc->next;
    }
  }
  return(0);
}
*/

/**************************************************************************
* G__cpplink_memvar()
*
**************************************************************************/
void G__cpplink_memvar(FILE *fp)
{
   G__FastAllocString commentbuf(G__LONGLINE);
   fprintf(fp, "\n/*********************************************************\n");
   fprintf(fp, "* Data Member information setup/\n");
   fprintf(fp, "*********************************************************/\n");
   fprintf(fp, "\n   /* Setting up class,struct,union tag member variable */\n");
   //
   //  Loop over all known classes, enums, namespaces, structs and unions.
   //
   for (int i = 0; i < G__struct.alltag; ++i) {
      if (
         // -- Class is marked for dictionary generation.
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
      ) {
         //
         //  FIXME: What is this block doing?
         //
         if (G__struct.name[i][0] == '$') {
            int typenum = G__defined_typename(G__struct.name[i] + 1);
            if (typenum!=-1 && isupper(G__newtype.type[typenum])) {
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
         if (G__globalcomp == G__CPPLINK) {
            // -- C++ style.
            fprintf(fp, "static void G__setup_memvar%s(void) {\n", G__map_cpp_name(G__fulltagname(i, 0)));
         }
         else {
            if (G__clock) {
               // -- C style.
               fprintf(fp, "static void G__setup_memvar%s() {\n", G__map_cpp_name(G__fulltagname(i, 0)));
            }
            else {
               // -- C++ style by default.
               fprintf(fp, "static void G__setup_memvar%s(void) {\n", G__map_cpp_name(G__fulltagname(i, 0)));
            }
         }
         //
         //  Write out call to member variable setup
         //  initialization to the dictionary file.
         //
         fprintf(fp, "   G__tag_memvar_setup(G__get_linked_tagnum(&%s));\n", G__mark_linked_tagnum(i));
         //
         //  We need a fake this pointer, except for namespace and unnamed struct members.
         //
         if ((G__struct.type[i] == 'n') || !G__struct.name[i][0]) {
            fprintf(fp, "   {\n");
         }
         else {
            fprintf(fp, "   { %s *p; p=(%s*)0x1000; if (p) { }\n", G__type2string('u', i, -1, 0, 0), G__type2string('u', i, -1, 0, 0));
         }
         //
         //  Loop over all the data members and write setup info to the file.
         //
         struct G__var_array* var = G__struct.memvar[i];
         while (var) { // Loop over all variable pages.
            for (int j = 0; j < var->allvar; ++j) { // Loop over all variables on page.
               if ( // Data member is not a bitfield and is accessible.
                  G__precomp_private || // option -V given on the command line, or
                  (
                     !var->bitfield[j] && // not a bitfield, and
                     (
                        (var->access[j] == G__PUBLIC) || // is public, or
                        (
                           (var->access[j] == G__PROTECTED) && // is protected, and
                           (G__struct.protectedaccess[i] & G__PROTECTEDACCESS) // enabled by pragma link
                        ) || // or,
                        (G__struct.protectedaccess[i] & G__PRIVATEACCESS) // is private, and enabled by pragma link
                     )
                  )
               ) { // Data member is not a bitfield and is accessible.
                  //
                  //  Write a data member setup call to the dictionary file.
                  //
                  int pvoidflag = 0; // if true, pass G__PVOID as the addr to force G__malloc to allocate storage.
                  if (
                     // Is a special data member for representing
                     // an enumerator value (has static const enum
                     // type, with type code 'i').
                     // FIXME: Collides with a data member of static const enumeration type!
                     (var->statictype[j] == G__LOCALSTATIC) && // static member, and
                     var->constvar[j] && // is const, and
                     (var->type[j] == 'i') && // not a pointer, type int, and
                     (var->reftype[j] == G__PARANORMAL) && // not a ref, and
                     (
                        // no elements, no dimensions, not an array
                        !var->varlabel[j][1] /* number of elements */ &&
                        !var->paran[j]
                     ) && // and,
                     (
                        (var->p_tagtable[j] != -1) && // class tag is valid, and
                        (G__struct.type[var->p_tagtable[j]] == 'e') // type is an enum type.
                     )
                  ) {
                     // Pass G__PVOID as the address to force G__malloc to allocate storage.
                     pvoidflag = 1;
                  }
                  if (
                     // Is static const integral type.
                     (var->statictype[j] == G__LOCALSTATIC) && // static member, and
                     var->constvar[j] && // is const, and
                     (var->p_tagtable[j] == -1) && // is fundamental type, and
                     islower(var->type[j]) && // not a pointer, and
                     (var->reftype[j] == G__PARANORMAL) && // not a ref, and
                     (
                        // no elements, no dimensions, not an array
                        !var->varlabel[j][1] /* number of elements */ &&
                        !var->paran[j]
                     ) && // and,
                     ( // of integral type
                        (var->type[j] == 'g') || // bool
                        (var->type[j] == 'c') || // char
                        (var->type[j] == 'b') || // unsigned char
                        (var->type[j] == 's') || // short
                        (var->type[j] == 'r') || // unsigned short
                        (var->type[j] == 'i') || // int
                        (var->type[j] == 'h') || // unsigned int
                        (var->type[j] == 'l') || // long
                        (var->type[j] == 'k') || // unsigned long
                        (var->type[j] == 'n') || // long long
                        (var->type[j] == 'm')    // unsigned long long
                     )
                  ) {
                     // Pass G__PVOID as the address to force G__malloc to allocate storage.
                     pvoidflag = 1;
                  }
                  //
                  //  Begin writing the member setup call.
                  //
                  fprintf(fp, "   G__memvar_setup(");
                  //
                  //  Offset in object for a non-static data member, or
                  //  the address of the member for a static data
                  //  member, or a namespace member.
                  //
                  if (pvoidflag) {
                     // Special case, special enumerator data member or
                     // static const integral data member.
                     // Pass G__PVOID to force G__malloc to allocate storage.
                     fprintf(fp, "(void*)G__PVOID,");
                  }
                  else {
                     if ((var->access[j] == G__PUBLIC) && !var->bitfield[j]) {
                        // Public member and not a bitfield.
                        if (!G__struct.name[i][0]) {
                           // Unnamed class or namespace.
                           // We pass a null pointer, which means
                           // no data allocation (unfortunate, but
                           // we have no way to take the address).
                           fprintf(fp, "(void*)0,");
                        }
                        else if ( // Static member or namespace member.
                           (var->statictype[j] == G__LOCALSTATIC) || // static member, or
                           (G__struct.type[i] == 'n') // namespace member.
                        ) { // Static member or namespace member.
                           // Pass the addr of the member.
                           fprintf(fp, "(void*)(&%s::%s),", G__fulltagname(i, 1), var->varnamebuf[j]);
                        }
                        else {
                           // Pass the offset of the member in the class.
                           fprintf(fp, "(void*)((long)(&p->%s)-(long)(p)),", var->varnamebuf[j]);
                        }
                     }
                     else if ((var->access[j] == G__PROTECTED) && G__struct.protectedaccess[i]) {
                        // Protected member, and enabled by pragma link.
                        // FIXME: Need code for enum, bool, and static const.
                        fprintf(fp, "(void*)((%s_PR*)p)->G__OS_%s(),", G__get_link_tagname(i), var->varnamebuf[j]);
                     }
                     else {
                        // Private or protected member, we pass a null pointer, unfortunate,
                        // but we have no way to take the address of these.
                        fprintf(fp, "(void*)0,");
                     }
                  }
                  //
                  //  Type code, referenceness, and constness.
                  //
                  fprintf(fp, "%d,", var->type[j]);
                  fprintf(fp, "%d,", var->reftype[j]);
                  fprintf(fp, "%d,", var->constvar[j]);
                  //
                  //  Tagnum of data type, if not fundamental.
                  //
                  if (var->p_tagtable[j] != -1) {
                     fprintf(fp, "G__get_linked_tagnum(&%s),", G__mark_linked_tagnum(var->p_tagtable[j]));
                  }
                  else {
                     fprintf(fp, "-1,");
                  }
                  //
                  //  Typenum of data type, if it is a typedef.
                  //
                  if (var->p_typetable[j] != -1) {
                     fprintf(fp, "G__defined_typename(\"%s\"),", G__newtype.name[var->p_typetable[j]]);
                  }
                  else {
                     fprintf(fp, "-1,");
                  }
                  //
                  //  Storage duration and staticness, member access.
                  //
                  fprintf(fp, "%d,", var->statictype[j]);
                  fprintf(fp, "%d,", var->access[j]);
                  //
                  //  Name and array dimensions (quoted) as the
                  //  left hand side of an assignment expression.
                  //
                  if (!pvoidflag || (G__globalcomp == G__CLINK)) {
                     // No special initializer needed.
                     fprintf(fp, "\"%s", var->varnamebuf[j]);
                     if (var->varlabel[j][1] /* num of elements */ == INT_MAX /* unspecified length array */) {
                        fprintf(fp, "[]");
                     }
                     else if (var->varlabel[j][1] /* num of elements */) {
                        fprintf(fp, "[%d]", var->varlabel[j][1] /* num of elements */ / var->varlabel[j][0] /* stride */);
                     }
                     for (int k = 1; k < var->paran[j]; ++k) {
                        fprintf(fp, "[%d]", var->varlabel[j][k+1]);
                     }
                     fprintf(fp, "=\"");
                  }
                  else {
                     // Special enumerator, or static const integral type.
                     if (var->access[j] == G__PUBLIC) {
                        // Public, let the compiler provide the value.
                        fprintf(
                             fp
                           , "G__FastAllocString(%d).Format(\""
                           , G__LONGLINE
                        );
                        fprintf(fp, "%s=", var->varnamebuf[j]);
#ifdef G_WIN32
                        if (
                           (var->type[j] == 'g') || // bool
                           (var->type[j] == 'c') || // char
                           (var->type[j] == 's') || // short
                           (var->type[j] == 'i') || // int
                           (var->type[j] == 'l') || // long
                           (var->type[j] == 'n')    // long long
                        ) {
                           fprintf(
                                fp
                              , "%%I64dLL\",(G__int64)%s::%s).data()"
                              , G__fulltagname(i, 1)
                              , var->varnamebuf[j]
                           );
                        }
                        else {
                           fprintf(
                                fp
                              , "%%I64uULL\",(G__uint64)%s::%s).data()"
                              , G__fulltagname(i, 1)
                              , var->varnamebuf[j]
                           );
                        }
#else // G_WIN32
                        if (
                           (var->type[j] == 'g') || // bool
                           (var->type[j] == 'c') || // char
                           (var->type[j] == 's') || // short
                           (var->type[j] == 'i') || // int
                           (var->type[j] == 'l') || // long
                           (var->type[j] == 'n')    // long long
                        ) {
                           fprintf(
                                fp
                              , "%%lldLL\",(long long)%s::%s).data()"
                              , G__fulltagname(i, 1)
                              , var->varnamebuf[j]
                           );
                        }
                        else {
                           fprintf(
                                fp
                              , "%%lluULL\",(unsigned long long)%s::%s).data()"
                              , G__fulltagname(i, 1)
                              , var->varnamebuf[j]
                           );
                        }
#endif // G_WIN32
                        // --
                     }
                     else {
                        // Not public, so compiler cannot access the value,
                        // get it from the interpreter.
                        fprintf(fp, "\"%s=", var->varnamebuf[j]);
                        switch (var->type[j]) {
                           case 'g': // bool
                              // --
#ifdef G__BOOL4BYTE
                              fprintf(fp, "%lldLL", (long long) *(int*)var->p[j]);
#else // G__BOOL4BYTE
                              fprintf(fp, "%lldULL", (unsigned long long) *(unsigned char*)var->p[j]);
#endif // G__BOOL4BYTE
                              break;
                           case 'c': // char
                              fprintf(fp, "%lldLL", (long long) *(char*)var->p[j]);
                              break;
                           case 'b': // unsigned char
                              fprintf(fp , "%lluULL", (unsigned long long) *(unsigned char*)var->p[j]);
                              break;
                           case 's': // short
                              fprintf(fp, "%lldLL", (long long) *(short*)var->p[j]);
                              break;
                           case 'r': // unsigned short
                              fprintf(fp, "%lluULL", (unsigned long long) *(unsigned short*)var->p[j]);
                              break;
                           case 'i': // int
                              fprintf(fp, "%lldLL", (long long) *(int*)var->p[j]);
                              break;
                           case 'h': // unsigned int
                              fprintf(fp , "%lluULL", (unsigned long long) *(unsigned int*)var->p[j]);
                              break;
                           case 'l': // long
                              fprintf(fp, "%lldLL", (long long) *(long*)var->p[j]);
                              break;
                           case 'k': // unsigned long
                              fprintf(fp , "%lluULL", (unsigned long long) *(unsigned long*)var->p[j]);
                              break;
                           case 'n': // long long
                              fprintf(fp, "%lldLL", *(long long*)var->p[j]);
                              break;
                           case 'm': // unsigned long long
                              fprintf(fp , "%lluULL", *(unsigned long long*)var->p[j]);
                              break;
                        }
                        fprintf(fp, "\"");
                     }
                  }
                  //
                  //  Define macro flag (always zero).
                  //
                  fprintf(fp, ",0");
                  //
                  //  Comment string.
                  //
                  G__getcommentstring(commentbuf, i, &var->comment[j]);
                  fprintf(fp, ",%s);\n", commentbuf());
               }
               G__var_type = 'p';
            }
            var = var->next;
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

/**************************************************************************
* G__isprivatectordtorassgn()
*
**************************************************************************/
int G__isprivatectordtorassgn(int tagnum, G__ifunc_table_internal *ifunc, int ifn)
{
  /* if(G__PRIVATE!=ifunc->access[ifn]) return(0); */
  if(G__PUBLIC==ifunc->access[ifn]) return(0);
  if('~'==ifunc->funcname[ifn][0]) return(1);
  if(strcmp(ifunc->funcname[ifn],G__struct.name[tagnum])==0) return(1);
  if(strcmp(ifunc->funcname[ifn],"operator=")==0) return(1);
  return(0);
}

/**************************************************************************
 * G__cpplink_memfunc()
 *
 **************************************************************************/
void G__cpplink_memfunc(FILE *fp)
{
#ifndef G__SMALLOBJECT
  int i,j,k;
  int hash,page;
  struct G__ifunc_table_internal *ifunc;
  G__FastAllocString funcname(G__MAXNAME*6);
  int isconstructor,iscopyconstructor,isdestructor,isassignmentoperator;
  /* int isvirtualdestructor; */
  G__FastAllocString buf(G__ONELINE);
  int isnonpublicnew;
  /* struct G__ifunc_table *baseifunc; */
  /* int baseifn; */
  /* int alltag=0; */
  int virtualdtorflag;
  int dtoraccess=G__PUBLIC;
  struct G__ifunc_table_internal *ifunc_destructor=0;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Member function information setup for each class\n");
  fprintf(fp,"*********************************************************/\n");

  if(G__CPPLINK == G__globalcomp) {
  }
  else {
  }

  for(i=0;i<G__struct.alltag;i++) {
    dtoraccess=G__PUBLIC;
    if((G__CPPLINK==G__struct.globalcomp[i]
        || G__ONLYMETHODLINK==G__struct.globalcomp[i]
        )&&
       (-1==(int)G__struct.parent_tagnum[i]
        || G__nestedclass
        )
       && -1!=G__struct.line_number[i]&&
       (G__struct.hash[i] || 0==G__struct.name[i][0])
       &&
       '$'!=G__struct.name[i][0] && 'e'!=G__struct.type[i]) {
      ifunc = G__struct.memfunc[i];
      isconstructor=0;
      iscopyconstructor=0;
      isdestructor=0;
      /* isvirtualdestructor=0; */
      isassignmentoperator=0;
      isnonpublicnew=G__isnonpublicnew(i);
      virtualdtorflag=0;

      if(G__clock)
        fprintf(fp,"static void G__setup_memfunc%s() {\n"
                ,G__map_cpp_name(G__fulltagname(i,0)));
      else
        fprintf(fp,"static void G__setup_memfunc%s(void) {\n"
                ,G__map_cpp_name(G__fulltagname(i,0)));

      /* link member function information */
      fprintf(fp,"   /* %s */\n",G__type2string('u',i,-1,0,0));

      fprintf(fp,"   G__tag_memfunc_setup(G__get_linked_tagnum(&%s));\n"
              ,G__mark_linked_tagnum(i));

      if(0==G__struct.name[i][0]) {
        fprintf(fp,"}\n");
        continue;
      }

      while (ifunc) {

        for (j = 0; j < ifunc->allifunc; ++j) {
          if ((ifunc->access[j] == G__PUBLIC) || G__precomp_private || G__isprivatectordtorassgn(i, ifunc, j) || ((ifunc->access[j] == G__PROTECTED) && (G__struct.protectedaccess[i] & G__PROTECTEDACCESS)) || (G__struct.protectedaccess[i] & G__PRIVATEACCESS)) {
            // public

            if ((G__struct.globalcomp[i] == G__ONLYMETHODLINK) && (ifunc->globalcomp[j] != G__METHODLINK)) {
              // not marked for link, skip it.
              continue;
            }

            if (!ifunc->hash[j]) {
              // no hash, skip it
              continue;
            }

#ifndef G__OLDIMPLEMENTATION1656
            // 08-10-07
            // Dont try to evaluate this condition in case we have the new scheme...
            // the automatic methods are created with size<0 by default so it
            // mmesses it up
            if (G__dicttype==kCompleteDictionary || !G__is_tagnum_safe(i)){
              if (ifunc->pentry[j]->size < 0) {
                // already precompiled, skip it
                continue;
              }
            }
#endif

            // Check for constructor, destructor, or operator=.
            if ( !strcmp(ifunc->funcname[j], G__struct.name[i]) && 
                 (G__dicttype==kNoWrappersDictionary || G__dicttype==kCompleteDictionary || !G__is_tagnum_safe(i)) ) {
              // We have a constructor.
              // If this was generated automatically.. print it... but dont re-do it at the end
              if (ifunc->funcptr[j]==(void*)-1)
                ++isconstructor;

              if (G__struct.isabstract[i]) {
                // Dont continue in this step because we used make_default_ifunc which
                // will create them as normal functions
                if(G__dicttype!=kNoWrappersDictionary)
                  continue;
              }
              if (isnonpublicnew) {
                // Dont continue in this step because we used make_default_ifunc which
                // will create them as normal functions
                if(G__dicttype!=kNoWrappersDictionary)
                  continue;
              }
              ++isconstructor;

              if ((ifunc->para_nu[j] >= 1) && (ifunc->param[j][0]->type == 'u') && (i == ifunc->param[j][0]->p_tagtable) && (ifunc->param[j][0]->reftype == G__PARAREFERENCE) && ((ifunc->para_nu[j] == 1) || ifunc->param[j][1]->pdefault)) {
                ++iscopyconstructor;
              }
            } else if ((G__dicttype==kNoWrappersDictionary || G__dicttype==kCompleteDictionary || !G__is_tagnum_safe(i))
                       && ifunc->funcname[j][0] == '~') {
              // If this was generated automatically.. print it... but dont re-do it at the end
              if (ifunc->funcptr[j]==(void*)-1)
                ++isdestructor;

              // We have a destructor.
              dtoraccess = ifunc->access[j];
              virtualdtorflag = ifunc->isvirtual[j] + (ifunc->ispurevirtual[j] * 2);
              // Why do we need this condition?
              if (G__PUBLIC != ifunc->access[j]) {
                ++isdestructor;
              }
              else
                ifunc_destructor = ifunc;

              if ((G__PROTECTED == ifunc->access[j]) && G__struct.protectedaccess[i] && !G__precomp_private) {
                G__fprinterr(G__serr, "Limitation: can not generate dictionary for protected destructor for %s\n", G__fulltagname(i, 1));
                // Dont continue in this step because we used make_default_ifunc which
                // will create them as normal functions
                if(G__dicttype!=kNoWrappersDictionary)
                  continue;
              }
              // Dont continue in this step because we used make_default_ifunc which
              // will create them as normal functions
              if(G__dicttype!=kNoWrappersDictionary)
                continue;
            }

#ifdef G__DEFAULTASSIGNOPR
            else if ( !strcmp(ifunc->funcname[j], "operator=")
                      && ('u' == ifunc->param[j][0]->type)
                      && (i == ifunc->param[j][0]->p_tagtable)) {
              // We have an operator=.
              if( G__dicttype!=kNoWrappersDictionary ||
                  (G__dicttype==kNoWrappersDictionary &&
                   !(!ifunc->mangled_name[j] && ifunc->funcptr[j]==(void*)-1)))
                ++isassignmentoperator;
            }
#endif
            /****************************************************************
             * setup normal function
             ****************************************************************/
            /* function name and return type */


            // Note: when this conditions are fullfilled, we don't generate the stub
            // for the assigment operator (when it's automatic)... so dont't generate its
            // memfunc either
            // The ifunc->funcptr[j]==(void*)-1 means that they were generated automatically,
            // if that is the case and we are writing the final dictionary (kCompleteDictionary),
            // we can't write the stubs for the operator=
            if(/*!(ifunc->funcptr[j]==(void*)-1) &&*/ (!ifunc->mangled_name[j] || !G__nostubs)){
              if(G__dicttype==kNoWrappersDictionary){
                // Now we have no symbol but we are in the third or fourth
                // dictionary... which means that the second one already tried to create it...
                // If that's the case we have no other choice but to generate the stub
                if((strcmp(ifunc->funcname[j],"operator=")==0 && ifunc->funcptr[j]==(void*)-1) /*&& !G__is_tagnum_safe(i)*/)
                  continue;
              }
            }

            if(/*!(ifunc->funcptr[j]==(void*)-1) &&*/ (!ifunc->mangled_name[j] || !G__nostubs)){
              if(G__dicttype==kNoWrappersDictionary){
                // Now we have no symbol but we are in the third or fourth
                // dictionary... which means that the second one already tried to create it...
                // If that's the case we have no other choice but to generate the stub
                if(ifunc->funcname[j][0]=='~' &&  !G__is_tagnum_safe(i))
                  continue;
              }
            }

            // 11-07-07 create a switch for the normal dictionaries
            if(G__dicttype==kCompleteDictionary)
              fprintf(fp,"   G__memfunc_setup(");
            else
              fprintf(fp,"   G__memfunc_setup2(");

            fprintf(fp,"\"%s\",%d,",ifunc->funcname[j],ifunc->hash[j]);

            // 04-07-07 print the mangled name after the funcname and hash
            if(G__dicttype!=kCompleteDictionary) {
              if(ifunc->mangled_name[j])
                fprintf(fp,"\"%s\",", ifunc->mangled_name[j]);
              else
                fprintf(fp,"0,");
            }


            // 24-05-07
            // Dont try to use the stubs functions since we just removed them
            if(G__PUBLIC==ifunc->access[j]
               || (((G__PROTECTED==ifunc->access[j] &&
                     (G__PROTECTEDACCESS&G__struct.protectedaccess[i])) ||
                    (G__PRIVATEACCESS&G__struct.protectedaccess[i])) /*&&
                                                                       '~'!=ifunc->funcname[j][0]*/)
              ) {
              // If the method is virtual. Is it overridden? -> Does it exist in the base classes? 
              //  Virtual method found in the base classes(we have an overridden virtual method)so it
              //  has not stub function in its dictionary
              if (G__dicttype!=kNoWrappersDictionary && (ifunc->isvirtual[j]) && (G__method_inbase(j, ifunc)))
                // Null Stub Pointer
                fprintf(fp, "(G__InterfaceMethod) NULL," );
              else {
                // If the method isn't virtual or it belongs to a base class.
                // The method has its own Stub Function in its dictionary
                // Normal Stub Pointer

                // 06-11-12
                // Second attempt...
                // Since the second dictionary registered the symbols and now we should had registered
                // those plus the one in the objects; we dont have to assume anything...
                // If the mangled name is not there the stub MUST have been created
                // 28-01-08: What we just said is false again because we don't have the *.o
                // anymore... so we have to continue with our assumptions

		// 29-01-08
		// Let's do it in the same way we did for genfunc... like that
		// we shouldnt get any difference between the stubs and the memfuns

		// If they werent generated automatically...
		if(/*!(ifunc->funcptr[j]==(void*)-1) && */ 
		   (!ifunc->mangled_name[j] || 
		    // 26-10-07
		    // Generate the stubs for those function needing a pointer to a reference (see TCLonesArray "virtual TObject*&	operator[](Int_t idx)")
		    // Is this condition correct and/or sufficient?
		    ((ifunc->reftype[j] == G__PARAREFERENCE) && isupper(ifunc->type[j])) ||
		    !G__nostubs)){
		  if(G__dicttype==kNoWrappersDictionary){
		    if(strcmp(ifunc->funcname[j],G__struct.name[i])==0) {
		      // constructor need special handling
		      if(0==G__struct.isabstract[i]&&0==isnonpublicnew) {
			fprintf(fp, "%s, ", G__map_cpp_funcname(i, ifunc->funcname[j], j, ifunc->page));
		      }
		      else
			fprintf(fp, "(G__InterfaceMethod) NULL, ");
		    }
		    else {
		      // Now we have no symbol but we are in the third or fourth
		      // dictionary... which means that the second one already tried to create it...
		      // If that's the case we have no other choice but to generate the stub
		      fprintf(fp, "%s, ", G__map_cpp_funcname(i, ifunc->funcname[j], j, ifunc->page));
		    }
		  }
		  else if(G__dicttype==kCompleteDictionary){
		    // This is the old case...
		    // just do what we did before
		    fprintf(fp, "%s, ", G__map_cpp_funcname(i, ifunc->funcname[j], j, ifunc->page));
		  }
		  else
		    fprintf(fp, "(G__InterfaceMethod) NULL, ");
		}
		else
		  fprintf(fp, "(G__InterfaceMethod) NULL, ");

                // Why do we have to put the isabstract here?
                // it doesnt seem to be necesary in the original code
                //
/*
		if( !ifunc->ispurevirtual[j] ||
                    (!ifunc->mangled_name[j] || !G__nostubs)
                  ) {
                  if(strcmp(ifunc->funcname[j],G__struct.name[i])==0) {
                    // constructor need special handling
                    if(0==G__struct.isabstract[i]&&0==isnonpublicnew) {
                      fprintf(fp, "%s, ", G__map_cpp_funcname(i, ifunc->funcname[j], j, ifunc->page));
                    }
                    else
                      fprintf(fp, "(G__InterfaceMethod) NULL, ");
                  }
                  else
                    fprintf(fp, "%s, ", G__map_cpp_funcname(i, ifunc->funcname[j], j, ifunc->page));
                }
                else
                  fprintf(fp, "(G__InterfaceMethod) NULL, ");
*/
              }
            }
            else
              fprintf(fp, "(G__InterfaceMethod) NULL, ");

            fprintf(fp, "%d, ", ifunc->type[j]);

            if (-1 != ifunc->p_tagtable[j]) {
                fprintf(fp, "G__get_linked_tagnum(&%s), ", G__mark_linked_tagnum(ifunc->p_tagtable[j]));
            }
            else
              fprintf(fp, "-1, ");

            if (-1 != ifunc->p_typetable[j])
              fprintf(fp, "G__defined_typename(\"%s\"), ", G__fulltypename(ifunc->p_typetable[j]));
            else
              fprintf(fp, "-1, ");

            fprintf(fp, "%d, ", ifunc->reftype[j]);

            /* K&R style if para_nu==-1, force it to 0 */
            if (0 > ifunc->para_nu[j]) fprintf(fp, "0, ");
            else                       fprintf(fp, "%d, ", ifunc->para_nu[j]);

            if (2 == ifunc->ansi[j])
              fprintf(fp, "%d, ", 8 + ifunc->staticalloc[j]*2 + ifunc->isexplicit[j]*4);
            else
              fprintf(fp, "%d, ", ifunc->ansi[j] + ifunc->staticalloc[j]*2 + ifunc->isexplicit[j]*4);

            fprintf(fp, "%d, ", ifunc->access[j]);
            fprintf(fp, "%d, ", ifunc->isconst[j]);

            /* newline to avoid lines more than 256 char for CMZ */
            if (ifunc->para_nu[j] > 1) fprintf(fp, "\n");
            fprintf(fp, "\"");

            /****************************************************************
             * function parameter
             ****************************************************************/

            for (k = 0; k < ifunc->para_nu[j]; k++) {
              /* newline to avoid lines more than 256 char for CMZ */
              if (G__CPPLINK == G__globalcomp && k && 0 == (k%2)) fprintf(fp, "\"\n\"");
              if (isprint(ifunc->param[j][k]->type)) {
                fprintf(fp, "%c ", ifunc->param[j][k]->type);
              }
              else {
                G__fprinterr(G__serr, "Internal error: function parameter type\n");
                fprintf(fp, "%d ", ifunc->param[j][k]->type);
              }

              if (-1 != ifunc->param[j][k]->p_tagtable) {
                fprintf(fp, "'%s' ", G__fulltagname(ifunc->param[j][k]->p_tagtable, 0));
                G__mark_linked_tagnum(ifunc->param[j][k]->p_tagtable);
              }
              else
                fprintf(fp, "- ");

              if (-1 != ifunc->param[j][k]->p_typetable)
                fprintf(fp, "'%s' ", G__fulltypename(ifunc->param[j][k]->p_typetable));
              else
                fprintf(fp, "- ");

              fprintf(fp, "%d ", ifunc->param[j][k]->reftype + ifunc->param[j][k]->isconst*10);
              if (ifunc->param[j][k]->def) {
#ifdef G__NOSTUBS
                // 15-02-08: When evaluating a default parameter in the stub-less calls,
                // its expression must be known to CInt, which migth not be true at run-time
                // for CPP defines. So replace macros with their actual values when generating
                // the dictionary
                G__FastAllocString res(G__ONELINE);
                G__FastAllocString tmp(G__ONELINE);
                G__FastAllocString value(G__ONELINE);
                char *str = ifunc->param[j][k]->def;
                int pos_res=0;
                int pos_str=0;

                while(str[pos_str]!='\0') {
                  int pos_tmp=0;
                  
                  // Copy everything if it doesnt start with a letter
                  while(!isalpha(str[pos_str]) && str[pos_str]!='\0'){
                     res.Set(pos_res, str[pos_str]);
                    pos_res++; pos_str++;
                  }
                  res.Set(pos_res, 0);

                  // if it's a letter, then look for the end of the name
                  while(isdigit(str[pos_str]) || isalpha(str[pos_str]) || str[pos_str]=='_'){
                     tmp.Set(pos_tmp, str[pos_str]);
                    pos_tmp++;  pos_str++;
                  }
                  tmp.Set(pos_tmp, 0);
                  pos_tmp++;

                  if(pos_tmp>1){
                    // Now look for the variable
                    long struct_offset = 0;
                    long store_struct_offset = 0;
                    int ig15 = 0;
                    int varhash = 0;
                    int isdecl = 0;
                    struct G__var_array* var = 0;
                    G__hash(tmp, varhash, ig15);
                    var = G__searchvariable((char*)tmp, varhash, 0, &G__global, &struct_offset, &store_struct_offset, &ig15, isdecl);
                    if (var && (var->type[ig15]=='P' || var->type[ig15]=='p')) {
                      G__value result3;
                      int known = 0;
                      result3 = G__getvariable(tmp, &known, &G__global, G__p_local);
                      if(var->type[ig15]=='P')
                         value.Format("%e", G__double(result3));
                      else
                         value.Format("%ld", G__int(result3));
                      res += value;
                      pos_res = strlen(res);
                    }
                    else {
                      // If we dont think this is a macro, the copy the same thing
                       res.Set(pos_res, 0);
                       res += tmp;
                       pos_res += strlen(tmp);
                    }
                  }
                }
                fprintf(fp, "'%s' ", G__quotedstring(res, buf));
#else                
                fprintf(fp, "'%s' ", G__quotedstring(ifunc->param[j][k]->def, buf));
#endif // G__NOSTUBS                
              }
              else
                fprintf(fp, "- ");
              if (ifunc->param[j][k]->name)
                fprintf(fp, "%s", ifunc->param[j][k]->name);
              else
                fprintf(fp, "-");
              if (k != ifunc->para_nu[j] - 1) fprintf(fp, " ");
            }
            fprintf(fp, "\"");

            G__getcommentstring(buf, i, &ifunc->comment[j]);
            fprintf(fp, ", %s", buf());
#ifdef G__TRUEP2F
            if (
               (ifunc->staticalloc[j] || 'n' == G__struct.type[i])
#ifndef G__OLDIMPLEMENTATION1292
               && G__PUBLIC == ifunc->access[j]
#endif // G__OLDIMPLEMENTATION1292
               && G__MACROLINK != ifunc->globalcomp[j]
              ) {
              int k;
              fprintf(fp, ", (void*) G__func2void( (%s (*)("
                      , G__type2string(ifunc->type[j]
                                       ,ifunc->p_tagtable[j]
                                       ,ifunc->p_typetable[j]
                                       ,ifunc->reftype[j]
                                       ,ifunc->isconst[j] // g++ may have problem
                        )
                );
              for (k = 0; k < ifunc->para_nu[j]; k++) {
                if (k) fprintf(fp, ", ");
                fprintf(fp, "%s"
                        ,G__type2string(ifunc->param[j][k]->type
                                        ,ifunc->param[j][k]->p_tagtable
                                        ,ifunc->param[j][k]->p_typetable
                                        ,ifunc->param[j][k]->reftype
                                        ,ifunc->param[j][k]->isconst));
              }
              fprintf(fp, "))(&%s::%s) )", G__fulltagname(ifunc->tagnum, 1), ifunc->funcname[j]);
            }
            else
              fprintf(fp, ", (void*) NULL");

            int virtflag = 0;
            if(G__dicttype==kCompleteDictionary) {
              virtflag = ifunc->isvirtual[j] + ifunc->ispurevirtual[j]*2;
            }
            else {
#ifdef G__NOSTUBS
              if(ifunc->isvirtual[j]){
                // 06-08-07
                // Trying to optimize the virtual function execution.
                // We are going to print ifunc page for a method that has been inherithed
                int page_base = ifunc->page_base;
                if(!page_base) {
                  // Look for it only in the parents, to check if we have the
		  // same method in both of them
		  page_base = G__method_inbase2(j, ifunc, 1);
		  
		  // If we don't find it in its parents and it's not ambiguous (not -1)
		  // look for it in the whole hierarchy
		  if(!page_base)
		    page_base = G__method_inbase2(j, ifunc, 0);

		  // If not found... start a page sequence
                  // This shouldn't be needed now that we do it
		  // inside G__method_inbase2
		  if(page_base==0) {
                    ifunc->page_base = ifunc->page+1;
                    page_base = ifunc->page_base;
                  }
                }

                // put "ispurevirtual" in the less significant bit and shift the rest to the left
                virtflag = 2*page_base + ifunc->ispurevirtual[j];
              }
#endif
            }

            fprintf(fp, ", %d", virtflag);
#endif // G__TRUEP2F
            fprintf(fp, ");\n");
          } // end of if access public && not pure virtual func
          else { // in case of protected,private or pure virtual func
            // protected, private, pure virtual
            if (!strcmp(ifunc->funcname[j], G__struct.name[i])) {
              ++isconstructor;
              if ('u' == ifunc->param[j][0]->type && i == ifunc->param[j][0]->p_tagtable && G__PARAREFERENCE == ifunc->param[j][0]->reftype && (1 == ifunc->para_nu[j] || ifunc->param[j][1]->pdefault)) {
                // copy constructor
                ++iscopyconstructor;
              }
            } else if ('~' == ifunc->funcname[j][0]) {
              // destructor
              ++isdestructor;
              ifunc_destructor = ifunc;
            } else if (!strcmp(ifunc->funcname[j], "operator new")) {
              ++isconstructor;
              ++iscopyconstructor;
            } else if (!strcmp(ifunc->funcname[j], "operator delete")) {
              // destructor
              ++isdestructor;
            }
#ifdef G__DEFAULTASSIGNOPR
            else if (!strcmp(ifunc->funcname[j], "operator=") && 'u' == ifunc->param[j][0]->type && i == ifunc->param[j][0]->p_tagtable) {
              // operator=
              ++isassignmentoperator;
              //ifunc_assignmentoperator = ifunc;
            }
#endif
          } // end of if access not public
        } // end for(j), loop over all ifuncs

        if (ifunc->next == 0
           // dummy
#ifndef G__OLDIMPLEMENTATON1656
           && G__NOLINK == G__struct.iscpplink[i]
#endif
#ifndef G__OLDIMPLEMENTATON1730
           && G__ONLYMETHODLINK != G__struct.globalcomp[i]
#endif
           ) {
          page = ifunc->page;
          if (j >= G__MAXIFUNC) {
            j = 0;
            ++page;
          }

          if (G__dicttype==kCompleteDictionary || 
              ( G__dicttype==kNoWrappersDictionary /*&& !G__is_tagnum_safe(i)*/) ){
            /****************************************************************
             * setup default constructor
             ****************************************************************/
            if (0 == isconstructor) isconstructor = G__isprivateconstructor(i, 0);
            if ('n' == G__struct.type[i]) isconstructor = 1;
            if (0 == isconstructor && 0 == G__struct.isabstract[i] && 0 == isnonpublicnew) {
              funcname = G__struct.name[i];
              G__hash(funcname, hash, k);
              fprintf(fp, "   // automatic default constructor\n");

              if(G__dicttype==kCompleteDictionary)
                fprintf(fp, "   G__memfunc_setup(");
              else
                fprintf(fp, "   G__memfunc_setup2(");

              fprintf(fp, "\"%s\", %d, ", funcname(), hash);

              // 04-07-07 print the mangled name after the funcname and hash
              if(G__dicttype!=kCompleteDictionary) {
                if(ifunc->mangled_name[j])
                  fprintf(fp,"\"%s\",", ifunc->mangled_name[j]);
                else
                  fprintf(fp,"0,");
              }

              fprintf(fp, "%s, ", G__map_cpp_funcname(i, funcname, j, page));

              fprintf(fp, "(int) ('i'), ");
              if (strlen(G__struct.name[i]) > 25) fprintf(fp,"\n");
              fprintf(fp, "G__get_linked_tagnum(&%s), ", G__mark_linked_tagnum(i));
              fprintf(fp, "-1, "); /* typenum */
              fprintf(fp, "0, "); /* reftype */
              fprintf(fp, "0, "); /* para_nu */
              fprintf(fp, "1, "); /* ansi */
              fprintf(fp, "%d, 0",G__PUBLIC);
#ifdef G__TRUEP2F
              fprintf(fp, ", \"\", (char*) NULL, (void*) NULL, %d);\n", 0);
#else
              fprintf(fp, ", \"\", (char*) NULL);\n");
#endif

            ++j;
            if (j >= G__MAXIFUNC) {
              j = 0;
              ++page;
            }
          } /* if(isconstructor) */

            /****************************************************************
             * setup copy constructor
             ****************************************************************/

            if (0 == iscopyconstructor) iscopyconstructor = G__isprivateconstructor(i, 1);
            if ('n' == G__struct.type[i]) iscopyconstructor = 1;
            if (0 == iscopyconstructor && 0 == G__struct.isabstract[i] && 0 == isnonpublicnew) {
              funcname = G__struct.name[i];
              G__hash(funcname, hash, k);
              fprintf(fp, "   // automatic copy constructor\n");

              if(G__dicttype==kCompleteDictionary)
                fprintf(fp, "   G__memfunc_setup(");
              else
                fprintf(fp, "   G__memfunc_setup2(");

              fprintf(fp, "\"%s\", %d, ", funcname(), hash);

              // 04-07-07 print the mangled name after the funcname and hash
              if(G__dicttype!=kCompleteDictionary) {
                if(ifunc->mangled_name[j])
                  fprintf(fp,"\"%s\",", ifunc->mangled_name[j]);
                else
                  fprintf(fp,"0,");
              }

              fprintf(fp, "%s, ", G__map_cpp_funcname(i, funcname, j, page));

              fprintf(fp, "(int) ('i'), ");
              if (strlen(G__struct.name[i]) > 20) fprintf(fp, "\n");
              fprintf(fp, "G__get_linked_tagnum(&%s), ", G__mark_linked_tagnum(i));
              fprintf(fp,"-1, "); /* typenum */
              fprintf(fp,"0, "); /* reftype */
              fprintf(fp,"1, "); /* para_nu */
              fprintf(fp,"1, "); /* ansi */
              fprintf(fp,"%d, 0",G__PUBLIC);
#ifdef G__TRUEP2F
              fprintf(fp, ", \"u '%s' - 11 - -\", (char*) NULL, (void*) NULL, %d);\n", G__fulltagname(i, 0), 0);
#else
              fprintf(fp, ", \"u '%s' - 11 - -\", (char*) NULL);\n", G__fulltagname(i, 0));
#endif
            ++j;
            if (j >= G__MAXIFUNC) {
              j = 0;
              ++page;

            }
            }

            /****************************************************************
             * setup destructor
             ****************************************************************/

            if (0 == isdestructor) isdestructor = G__isprivatedestructor(i);
            if ('n' == G__struct.type[i]) isdestructor = 1;
            if (0 == isdestructor && 'n' != G__struct.type[i]) {
              funcname.Format("~%s", G__struct.name[i]);
              G__hash(funcname, hash, k);
              fprintf(fp, "   // automatic destructor\n");

              if(G__dicttype==kCompleteDictionary)
                fprintf(fp, "   G__memfunc_setup(");
              else
                fprintf(fp, "   G__memfunc_setup2(");

              fprintf(fp, "\"%s\", %d, ", funcname(), hash);
              //04-07-07 print the mangled name after the funcname and hash
              if(G__dicttype!=kCompleteDictionary){
                fprintf(fp,"0,");
              }

              fprintf(fp, "%s, ", G__map_cpp_funcname(i, funcname, j, page));

              fprintf(fp, "(int) ('y'), ");
              fprintf(fp, "-1, "); /* tagnum */
              fprintf(fp, "-1, "); /* typenum */
              fprintf(fp, "0, "); /* reftype */
              fprintf(fp, "0, "); /* para_nu */
              fprintf(fp, "1, "); /* ansi */
              fprintf(fp, "%d, 0", dtoraccess);
#ifdef G__TRUEP2F
              fprintf(fp, ", \"\", (char*) NULL, (void*) NULL, %d);\n", virtualdtorflag);
#else
              fprintf(fp, ", \"\", (char*) NULL);\n");
#endif
              // LF
              if (0 == isdestructor)
                ++j;
              if (j >= G__MAXIFUNC) {
                j = 0;
                ++page;
              }
            }

#ifdef G__DEFAULTASSIGNOPR
            /****************************************************************
             * setup assignment operator
             ****************************************************************/

            if (0 == isassignmentoperator) isassignmentoperator = G__isprivateassignopr(i);
            if ('n' == G__struct.type[i]) isassignmentoperator = 1;
            if (0 == isassignmentoperator) {
              funcname = "operator=";
              G__hash(funcname, hash, k);
              fprintf(fp, "   // automatic assignment operator\n");

              if(G__dicttype==kCompleteDictionary)
                fprintf(fp, "   G__memfunc_setup(");
              else
                fprintf(fp, "   G__memfunc_setup2(");

              fprintf(fp, "\"%s\", %d, ", funcname(), hash);

              // 04-07-07 print the mangled name after the funcname and hash
              if(G__dicttype!=kCompleteDictionary) {
                if(ifunc->mangled_name[j])
                  fprintf(fp,"\"%s\",", ifunc->mangled_name[j]);
                else
                  fprintf(fp,"0,");
              }

              fprintf(fp, "%s, ", G__map_cpp_funcname(i, funcname, j, page));

              fprintf(fp, "(int) ('u'), ");
              fprintf(fp, "G__get_linked_tagnum(&%s), ", G__mark_linked_tagnum(i));
              fprintf(fp, "-1, "); /* typenum */
              fprintf(fp, "1, "); /* reftype */
              fprintf(fp, "1, "); /* para_nu */
              fprintf(fp, "1, "); /* ansi */
              fprintf(fp, "%d, 0", G__PUBLIC);
#ifdef G__TRUEP2F
              fprintf(fp, ", \"u '%s' - 11 - -\", (char*) NULL, (void*) NULL, %d);\n", G__fulltagname(i, 0), 0);
#else
              fprintf(fp, ", \"u '%s' - 11 - -\", (char*) NULL);\n", G__fulltagname(i, 0));
#endif
            }
          }
#endif
        } /* end of ifunc->next */
        ifunc = ifunc->next;
      } /* end while(ifunc) */
      fprintf(fp, "   G__tag_memfunc_reset();\n");
      fprintf(fp, "}\n\n");
    } /* end if(globalcomp) */
  } /* end for(i) */

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Member function information setup\n");
  fprintf(fp,"*********************************************************/\n");

  if (G__globalcomp == G__CPPLINK) {
    fprintf(fp, "extern \"C\" void G__cpp_setup_memfunc%s() {\n", G__DLLID);
  } else {
    /* fprintf(fp, "void G__c_setup_memfunc%s() {\n", G__DLLID); */
  }

  fprintf(fp, "}\n");

#endif
}


/**************************************************************************
* G__cpplink_global()
*
**************************************************************************/
void G__cpplink_global(FILE *fp)
{
#ifndef G__SMALLOBJECT
  int j,k;
  struct G__var_array *var;
  int pvoidflag;
  G__value buf;
  G__FastAllocString value(G__ONELINE);
  G__FastAllocString ttt(G__ONELINE);
  int divn=0;
  int maxfnc=100;
  int fnc=0;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Global variable information setup for each class\n");
  fprintf(fp,"*********************************************************/\n");

#ifdef G__BORLANDCC5
  fprintf(fp,"static void G__cpp_setup_global%d(void) {\n",divn++);
#else
  fprintf(fp,"static void G__cpp_setup_global%d() {\n",divn++);
#endif

  fprintf(fp,"\n   /* Setting up global variables */\n");
  var = &G__global;
  fprintf(fp,"   G__resetplocal();\n\n");

  while((struct G__var_array*)NULL!=var) {
    for(j=0;j<var->allvar;j++) {
      if(fnc++>maxfnc) {
        fnc=0;
        fprintf(fp,"}\n\n");
#ifdef G__BORLANDCC5
        fprintf(fp,"static void G__cpp_setup_global%d(void) {\n",divn++);
#else
        fprintf(fp,"static void G__cpp_setup_global%d() {\n",divn++);
#endif
      }
      if (
        (
          (var->statictype[j] == G__AUTO) /* not static */ ||
          (
            !var->p[j] &&
            (var->statictype[j] == G__COMPILEDGLOBAL) &&
            (var->varlabel[j][1] /* num of elements */ == INT_MAX /* unspecified length array flag */)
          )
        ) && /* extern type v[]; */
        (G__NOLINK > var->globalcomp[j]) && /* with -c-1 or -c-2 option */
#ifndef G__OLDIMPLEMENTATION2191
        ('j' != tolower(var->type[j])) && /* questionable */
#else
        ('m' != tolower(var->type[j])) &&
#endif
        var->varnamebuf[j][0]
      ) {

        if((-1!=var->p_tagtable[j]&&
            islower(var->type[j])&&var->constvar[j]&&
            'e'==G__struct.type[var->p_tagtable[j]])
           || 'p'==tolower(var->type[j])
           || 'T'==var->type[j]
#ifdef G__UNADDRESSABLEBOOL
           || 'g'==var->type[j]
#endif
           || (var->statictype[j] == G__LOCALSTATIC && var->constvar[j] && // const static
               islower(var->type[j]) && var->type[j] != 'u' && // of fundamental
               var->p[j]) // with initializer
           )
          pvoidflag=1;
        else
          pvoidflag=0;

        fprintf(fp,"   G__memvar_setup(");
        if(pvoidflag) fprintf(fp,"(void*)G__PVOID,");
        else {
          fprintf(fp,"(void*)(&%s),",var->varnamebuf[j]);
#ifdef G__GENWINDEF
          fprintf(G__WINDEFfp,"        %s @%d\n"
                  ,var->varnamebuf[j] ,++G__nexports);
#endif
        }
        fprintf(fp,"%d,",var->type[j]);
        fprintf(fp,"%d,",var->reftype[j]);
        fprintf(fp,"%d,",var->constvar[j]);

        if(-1!=var->p_tagtable[j])
          fprintf(fp,"G__get_linked_tagnum(&%s),"
                  ,G__mark_linked_tagnum(var->p_tagtable[j]));
        else
          fprintf(fp,"-1,");

        if(-1!=var->p_typetable[j])
          fprintf(fp,"G__defined_typename(\"%s\"),"
                  ,G__newtype.name[var->p_typetable[j]]);
        else
          fprintf(fp,"-1,");

        fprintf(fp, "%d,", var->statictype[j]);
        fprintf(fp, "%d,", var->access[j]);
        fprintf(fp, "\"%s", var->varnamebuf[j]);
        if (var->varlabel[j][1] /* num of elements */ == INT_MAX /* unspecified length flag */) {
          fprintf(fp, "[]");
        }
        else if (var->varlabel[j][1] /* num of elements */) {
          fprintf(fp, "[%d]", var->varlabel[j][1] /* num of elements */ / var->varlabel[j][0] /* stride */);
        }
        for (k = 1; k < var->paran[j]; ++k) {
          fprintf(fp, "[%d]", var->varlabel[j][k+1]);
        }
        if (pvoidflag) {
          buf = G__getitem(var->varnamebuf[j]);
          G__string(buf, value);
          G__quotedstring(value, ttt);
          if ((tolower(var->type[j]) == 'p') || (var->type[j] == 'T')) {
             fprintf(fp, "=%s\",1,(char*)NULL);\n", ttt());
          }
          else {
             fprintf(fp, "=%s\",0,(char*)NULL);\n", ttt());
          }
        }
        else {
          fprintf(fp, "=\",0,(char*)NULL);\n");
        }
      } /* end if(G__PUBLIC) */
      G__var_type='p';
    } /* end for(j) */
    var=var->next;
  }  /* end while(var) */

  fprintf(fp,"\n");
  fprintf(fp,"   G__resetglobalenv();\n");

  fprintf(fp,"}\n");

  if(G__CPPLINK == G__globalcomp) {
    fprintf(fp,"extern \"C\" void G__cpp_setup_global%s() {\n",G__DLLID);
  }
  else {
    fprintf(fp,"void G__c_setup_global%s() {\n",G__DLLID);
  }
  for(fnc=0;fnc<divn;fnc++) {
    fprintf(fp,"  G__cpp_setup_global%d();\n",fnc);
  }
  fprintf(fp,"}\n");


#endif
}

#ifdef G__P2FDECL  /* used to be G__TRUEP2F */
/**************************************************************************
* G__declaretruep2f()
*
**************************************************************************/
static void G__declaretruep2f(FILE *fp, G__ifunc_table_internal &ifunc, int j)
{
#ifdef G__P2FDECL
  int i;
  int ifndefflag=1;
  if(strncmp(ifunc->funcname[j],"operator",8)==0) ifndefflag=0;
  if(ifndefflag) {
    switch(G__globalcomp) {
    case G__CPPLINK:
      if(G__MACROLINK==ifunc->globalcomp[j]||
         0==
         strcmp("iterator_category",ifunc->funcname[j])) fprintf(fp,"#if 0\n");
      else fprintf(fp,"#ifndef %s\n",ifunc->funcname[j]);
      fprintf(fp,"%s (*%sp2f)("
              ,G__type2string(ifunc->type[j]
                              ,ifunc->p_tagtable[j]
                              ,ifunc->p_typetable[j]
                              ,ifunc->reftype[j]
                              ,ifunc->isconst[j]
                              /* ,0  avoiding g++ bug */
                              )
              ,G__map_cpp_funcname(-1,ifunc->funcname[j],j,ifunc->page)
              );
      for(i=0;i<ifunc->para_nu[j];i++) {
        if(i) fprintf(fp,", ");
        fprintf(fp,"%s"
                ,G__type2string(ifunc->param[j][i]->type
                                ,ifunc->param[j][i]->p_tagtable
                                ,ifunc->param[j][i]->p_typetable
                                ,ifunc->param[j][i]->reftype
                                ,ifunc->param[j][i]->isconst));
      }
      fprintf(fp,") = %s;\n",ifunc->funcname[j]);
      fprintf(fp,"#else\n");
      fprintf(fp,"void* %sp2f = (void*) NULL;\n"
              ,G__map_cpp_funcname(-1,ifunc->funcname[j],j,ifunc->page));
      fprintf(fp,"#endif\n");
      break;
    default:
      break;
    }
  }
#else
  if(fp && ifunc && j) return;
#endif
}
#endif

#ifdef G__TRUEP2F
/**************************************************************************
* G__printtruep2f()
*
**************************************************************************/
static void G__printtruep2f(FILE *fp, G__ifunc_table_internal *ifunc, int j)
{
#if defined(G__P2FCAST)
  int i;
#endif
  int ifndefflag=1;
#if defined(G__FUNCPOINTER)
  if(strncmp(ifunc->funcname[j],"operator",8)==0)
    ifndefflag=0;
#else
  ifndefflag=0;
#endif
  if(ifndefflag) {
    switch(G__globalcomp) {
    case G__CPPLINK:
#if defined(G__P2FDECL)
      fprintf(fp,", (void*) %sp2f);\n"
              ,G__map_cpp_funcname(-1,ifunc->funcname[j],j,ifunc->page));
#elif defined(G__P2FCAST)
      if(G__MACROLINK==ifunc->globalcomp[j]||
         0==
         strcmp("iterator_category",ifunc->funcname[j])) fprintf(fp,"#if 0\n");
      else fprintf(fp,"#ifndef %s\n",ifunc->funcname[j]);

      fprintf(fp,", (void*) (%s (*)("
              ,G__type2string(ifunc->type[j]
                              ,ifunc->p_tagtable[j]
                              ,ifunc->p_typetable[j]
                              ,ifunc->reftype[j]
                              ,ifunc->isconst[j] /* g++ may have problem */
                              )
              /* ,G__map_cpp_funcname(-1,ifunc->funcname[j],j,ifunc->page) */
              );
      for(i=0;i<ifunc->para_nu[j];i++) {
        if(i) fprintf(fp,", ");
        fprintf(fp,"%s"
                ,G__type2string(ifunc->param[j][i]->type
                                ,ifunc->param[j][i]->p_tagtable
                                ,ifunc->param[j][i]->p_typetable
                                ,ifunc->param[j][i]->reftype
                                ,ifunc->param[j][i]->isconst));
      }
      fprintf(fp,"))%s, %d);\n",ifunc->funcname[j]
              ,ifunc->isvirtual[j]+ifunc->ispurevirtual[j]*2);
      fprintf(fp,"#else\n");
      fprintf(fp,", (void*) NULL, %d);\n"
              ,ifunc->isvirtual[j]+ifunc->ispurevirtual[j]*2);
      fprintf(fp,"#endif\n");
#else
      fprintf(fp,", (void*) NULL, %d);\n"
              ,ifunc->isvirtual[j]+ifunc->ispurevirtual[j]*2);
#endif
      break;
    case G__CLINK:
    default:
       fprintf(fp, ", funcptr._read, %d);\n",ifunc->isvirtual[j]+ifunc->ispurevirtual[j]*2);
      // fprintf(fp,"#ifndef %s\n",ifunc->funcname[j]);
      // fprintf(fp,", (void*) %s, %d);\n",ifunc->funcname[j]
      //         ,ifunc->isvirtual[j]+ifunc->ispurevirtual[j]*2);
      // fprintf(fp,"#else\n");
      // fprintf(fp,", (void*) NULL, %d);\n"
      //         ,ifunc->isvirtual[j]+ifunc->ispurevirtual[j]*2);
      // fprintf(fp,"#endif\n");
      break;
    }
  }
  else {
    fprintf(fp,", (void*) NULL, %d);\n"
              ,ifunc->isvirtual[j]+ifunc->ispurevirtual[j]*2);
  }
}
#endif

/**************************************************************************
* G__cpplink_func()
*
*  making C++ link routine to global function
**************************************************************************/
void G__cpplink_func(FILE *fp)
{
  int j,k;
  struct G__ifunc_table_internal *ifunc;
  G__FastAllocString buf(G__ONELINE);
  int divn=0;
  int maxfnc=100;
  int fnc=0;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Global function information setup for each class\n");
  fprintf(fp,"*********************************************************/\n");

#ifdef G__BORLANDCC5
  fprintf(fp,"static void G__cpp_setup_func%d(void) {\n",divn++);
#else
  fprintf(fp,"static void G__cpp_setup_func%d() {\n",divn++);
#endif

  ifunc = &G__ifunc;
  
  if (G__globalcomp == G__CLINK) {
     fprintf(fp," funcptr_and_voidptr funcptr;\n");
  }
  fprintf(fp,"   G__lastifuncposition();\n\n");
  
  while((struct G__ifunc_table_internal*)NULL!=ifunc) {
    for(j=0;j<ifunc->allifunc;j++) {
      if(fnc++>maxfnc) {
        fnc=0;
        fprintf(fp,"}\n\n");
#ifdef G__BORLANDCC5
        fprintf(fp,"static void G__cpp_setup_func%d(void) {\n",divn++);
#else
        fprintf(fp,"static void G__cpp_setup_func%d() {\n",divn++);
#endif
        if (G__globalcomp == G__CLINK) {
           fprintf(fp," funcptr_and_voidptr funcptr;\n");
        }
      }
      if(G__NOLINK>ifunc->globalcomp[j] &&  /* with -c-1 option */
         G__PUBLIC==ifunc->access[j] && /* public, this is always true */
         0==ifunc->staticalloc[j] &&
         ifunc->hash[j]) {   /* not static */

        if(strcmp(ifunc->funcname[j],"operator new")==0 &&
           (ifunc->para_nu[j]==2 || ifunc->param[j][2]->pdefault)) {
          G__is_operator_newdelete |= G__IS_OPERATOR_NEW;
        }
        else if(strcmp(ifunc->funcname[j],"operator delete")==0) {
          G__is_operator_newdelete |= G__IS_OPERATOR_DELETE;
        }

#ifdef G__P2FDECL  /* used to be G__TRUEP2F */
        G__declaretruep2f(fp,ifunc,j);
#endif
        if (G__globalcomp == G__CLINK) {
           fprintf(fp,"#ifndef %s\n",ifunc->funcname[j]); 
           fprintf(fp,"   funcptr._write = (void (*)())%s;\n",ifunc->funcname[j]);
           fprintf(fp,"#else\n");
           fprintf(fp,"   funcptr._write = 0;\n");
           fprintf(fp,"#endif\n");
        }

        if(G__dicttype==kCompleteDictionary)
          fprintf(fp, "   G__memfunc_setup(");
        else
          fprintf(fp,"   G__memfunc_setup2(");

        fprintf(fp,"\"%s\", %d, ",ifunc->funcname[j],ifunc->hash[j]);
        // 04-07-07 print the mangled name after the funcname and hash
        if(G__dicttype!=kCompleteDictionary){
          if(ifunc->mangled_name[j])
            fprintf(fp,"\"%s\",", ifunc->mangled_name[j]);
          else
            fprintf(fp,"0,");
        }

        // 24-05-07
        // Remember we only have stubs for global operators...
        // for the rest just point to null
        /* function name and return type */
        if(  G__dicttype==kCompleteDictionary ||
             ((G__dicttype==kNoWrappersDictionary) &&
              // The stubs where generated for functions needing temporal objects too... use them
              ( !ifunc->mangled_name[j] || 
                // 26-10-07
                // Generate the stubs for those function needing a pointer to a reference (see TCLonesArray "virtual TObject*&	operator[](Int_t idx)")
                // Is this condition correct and/or sufficient?
                ((ifunc->reftype[j] == G__PARAREFERENCE) && isupper(ifunc->type[j]))
                || !G__nostubs)
               )
          )
          fprintf(fp,"%s, ",G__map_cpp_funcname(-1
                                                ,ifunc->funcname[j]
                                                ,j,ifunc->page));
        else
          fprintf(fp, "(G__InterfaceMethod) NULL, ");

        fprintf(fp,"%d, ",ifunc->type[j]);

        if(-1!=ifunc->p_tagtable[j])
          fprintf(fp,"G__get_linked_tagnum(&%s), "
                  ,G__mark_linked_tagnum(ifunc->p_tagtable[j]));
        else
          fprintf(fp,"-1, ");

        if(-1!=ifunc->p_typetable[j])
          fprintf(fp,"G__defined_typename(\"%s\"), "
                  ,G__newtype.name[ifunc->p_typetable[j]]);
        else
          fprintf(fp,"-1, ");

        fprintf(fp,"%d, ",ifunc->reftype[j]);

        /* K&R style if para_nu==-1, force it to 0 */
        if(0>ifunc->para_nu[j]) fprintf(fp,"0, ");
        else                    fprintf(fp,"%d, ",ifunc->para_nu[j]);

        if(2==ifunc->ansi[j])
          fprintf(fp,"%d, ",8 + ifunc->staticalloc[j]*2);
        else
          fprintf(fp,"%d, ",ifunc->ansi[j] + ifunc->staticalloc[j]*2);
        fprintf(fp,"%d, ",ifunc->access[j]);
        fprintf(fp,"%d, ",ifunc->isconst[j]);

        /* newline to avoid lines more than 256 char for CMZ */
        if(ifunc->para_nu[j]>1) fprintf(fp,"\n");
        fprintf(fp,"\"");

        /****************************************************************
         * function parameter
         ****************************************************************/
        for(k=0;k<ifunc->para_nu[j];k++) {
          /* newline to avoid lines more than 256 char for CMZ */
          if(G__CPPLINK==G__globalcomp&&k&&0==(k%2)) fprintf(fp,"\"\n\"");
          if(isprint(ifunc->param[j][k]->type)) {
            fprintf(fp,"%c ",ifunc->param[j][k]->type);
          }
          else {
            G__fprinterr(G__serr,"Internal error: function parameter type\n");
            fprintf(fp,"%d ",ifunc->param[j][k]->type);
          }

          if(-1!=ifunc->param[j][k]->p_tagtable) {
            fprintf(fp,"'%s' "
                    ,G__fulltagname(ifunc->param[j][k]->p_tagtable,0));
            G__mark_linked_tagnum(ifunc->param[j][k]->p_tagtable);
          }
          else
            fprintf(fp,"- ");

          if(-1!=ifunc->param[j][k]->p_typetable)
            fprintf(fp,"'%s' "
                    ,G__newtype.name[ifunc->param[j][k]->p_typetable]);
          else
            fprintf(fp,"- ");

          fprintf(fp,"%d "
                ,ifunc->param[j][k]->reftype+ifunc->param[j][k]->isconst*10);
          if(ifunc->param[j][k]->def)
             fprintf(fp,"'%s' ",G__quotedstring(ifunc->param[j][k]->def,buf));
          else fprintf(fp,"- ");
          if(ifunc->param[j][k]->name)
            fprintf(fp,"%s",ifunc->param[j][k]->name);
          else fprintf(fp,"-");
          if(k!=ifunc->para_nu[j]-1) fprintf(fp," ");
        }
#ifdef G__TRUEP2F
        fprintf(fp,"\", (char*) NULL\n");
        G__printtruep2f(fp,ifunc,j);
#else
        fprintf(fp,"\", (char*) NULL);\n");
#endif

      }
    } /* end for(j) */
    ifunc = ifunc->next;
  } /* end while(ifunc) */

  fprintf(fp,"\n");
  fprintf(fp,"   G__resetifuncposition();\n");

  /********************************************************
  * call user initialization function if specified
  ********************************************************/
  if(G__INITFUNC) {
    fprintf(fp,"  %s();\n",G__INITFUNC);
  }

  fprintf(fp,"}\n\n");

  if(G__CPPLINK == G__globalcomp) {
    fprintf(fp,"extern \"C\" void G__cpp_setup_func%s() {\n",G__DLLID);
  }
  else {
    fprintf(fp,"void G__c_setup_func%s() {\n",G__DLLID);
  }
  for(fnc=0;fnc<divn;fnc++) {
    fprintf(fp,"  G__cpp_setup_func%d();\n",fnc);
  }
  fprintf(fp,"}\n");
}

/**************************************************************************
**************************************************************************
*  Functions used in G__cpplink.C to bind C++ symbols
*
**************************************************************************
**************************************************************************/

char G__incsetup_exist(std::list<G__incsetup> *incsetuplist, G__incsetup incsetup)
{

   if(incsetuplist->empty()) return 0;
   std::list<G__incsetup>::iterator iter;
   for (iter=incsetuplist->begin(); iter != incsetuplist->end(); ++iter)
      if (*iter==incsetup)
         return 1;

   return 0;
}


/**************************************************************************
* G__tagtable_setup()
*
*  Used in G__cpplink.C
**************************************************************************/
int G__tagtable_setup(int tagnum,int size,int cpplink,int isabstract,const char *comment
                      ,G__incsetup setup_memvar, G__incsetup setup_memfunc)
{
  if (tagnum < 0) return 0;
   
  char *p;
  G__FastAllocString buf(G__ONELINE);

  if (G__struct.incsetup_memvar[tagnum]==0)
     G__struct.incsetup_memvar[tagnum] = new std::list<G__incsetup>();

  if (G__struct.incsetup_memfunc[tagnum]==0)
     G__struct.incsetup_memfunc[tagnum] = new std::list<G__incsetup>();

   if(0==size && 0!=G__struct.size[tagnum]
     && 'n'!=G__struct.type[tagnum]
     ) return(0);

  if(0!=G__struct.size[tagnum]
     && 'n'!=G__struct.type[tagnum]
     ) {
     if (G__struct.filenum[tagnum]!=-1 &&
	!G__struct.incsetup_memvar[tagnum]->empty() &&
	strcmp(G__srcfile[G__struct.filenum[tagnum]].filename,"{CINTEX dictionary translator}")==0) {
        // Something was already registered by Cintex, let's not add more
        return 0;
     }
#ifndef G__OLDIMPLEMENTATION1656

     char found = G__incsetup_exist(G__struct.incsetup_memvar[tagnum],setup_memvar);
     // If setup_memvar is not NULL we push the G__setup_memvarXXX pointer into the list 
     if (setup_memvar&&!found)
        G__struct.incsetup_memvar[tagnum]->push_back(setup_memvar);

     found = G__incsetup_exist(G__struct.incsetup_memfunc[tagnum],setup_memfunc);
     // If setup_memfunc is not NULL we push the G__setup_memfuncXXX pointer into the list 
     if (setup_memfunc&&!found)
        G__struct.incsetup_memfunc[tagnum]->push_back(setup_memfunc);

#endif /* 1656 */
    if(G__asm_dbg ) {
      if(G__dispmsg>=G__DISPWARN) {
        G__fprinterr(G__serr,"Warning: Try to reload %s from DLL\n"
                     ,G__fulltagname(tagnum,1));
      }
    }
    return(0);
  }
  G__struct.size[tagnum] = size;
  G__struct.iscpplink[tagnum] = cpplink;
#if  !defined(G__OLDIMPLEMENTATION1545)
  G__struct.rootflag[tagnum] = (isabstract/0x10000)%0x100;
  G__struct.funcs[tagnum] = (isabstract/0x100)%0x100;
  G__struct.isabstract[tagnum] = isabstract%0x100;
#else
  G__struct.funcs[tagnum] = isabstract/0x100;
  G__struct.isabstract[tagnum] = isabstract%0x100;
#endif
  G__struct.filenum[tagnum] = G__ifile.filenum;

  G__struct.comment[tagnum].p.com = (char*)comment;
  if(comment) G__struct.comment[tagnum].filenum = -2;
  else        G__struct.comment[tagnum].filenum = -1;

     if(0==G__struct.memvar[tagnum]->allvar
        || 'n'==G__struct.type[tagnum]){
         char found = G__incsetup_exist(G__struct.incsetup_memvar[tagnum],setup_memvar);
         // If setup_memvar is not NULL we push the G__setup_memvarXXX pointer into the list 
         if (setup_memvar&&!found)
            G__struct.incsetup_memvar[tagnum]->push_back(setup_memvar);
     }
     //  else
     // G__struct.incsetup_memvar[tagnum]->clear();

  if(
     1==G__struct.memfunc[tagnum]->allifunc
     || 'n'==G__struct.type[tagnum]
#if G__MAXIFUNC > 1
     || (
         -1!=G__struct.memfunc[tagnum]->pentry[1]->size
         && 2>=G__struct.memfunc[tagnum]->allifunc)
#endif
     ){
     char found = 0;
     found = G__incsetup_exist(G__struct.incsetup_memfunc[tagnum], setup_memfunc);
     if (setup_memfunc&&!found)
        G__struct.incsetup_memfunc[tagnum]->push_back(setup_memfunc);
  }
  /* add template names */
  buf = G__struct.name[tagnum];
  if((p=strchr(buf,'<'))) {
    *p='\0';
    if(!G__defined_templateclass(buf)) {
      int store_def_tagnum = G__def_tagnum;
      int store_tagdefining = G__tagdefining;
      FILE* store_fp = G__ifile.fp;
      G__ifile.fp = (FILE*)NULL;
      G__def_tagnum = G__struct.parent_tagnum[tagnum];
      G__tagdefining = G__struct.parent_tagnum[tagnum];
      G__createtemplateclass(buf,(struct G__Templatearg*)NULL,0);
      G__ifile.fp = store_fp;
      G__def_tagnum = store_def_tagnum;
      G__tagdefining = store_tagdefining;
    }
  }
  return(0);
}

/**************************************************************************
* G__inheritance_setup()
*
*  Used in G__cpplink.C
**************************************************************************/
int G__inheritance_setup(int tagnum,int basetagnum
                         ,long baseoffset,int baseaccess
                         ,int property
                         )
{
#ifndef G__SMALLOBJECT
   if (0<=tagnum && 0<=basetagnum) {
      int basen=G__struct.baseclass[tagnum]->basen;
      //G__herit *herit = G__struct.baseclass[tagnum]->herit;
      //printf("G__inheritance_setup, tagnum=%d, basetagnum=%d, basen=%d\n",tagnum,basetagnum,basen);
      //printf("    herit=%x, herit.id=%x\n",herit,herit->id);
      G__struct.baseclass[tagnum]->herit[basen]->basetagnum = basetagnum;
      G__struct.baseclass[tagnum]->herit[basen]->baseoffset=baseoffset;
      G__struct.baseclass[tagnum]->herit[basen]->baseaccess=baseaccess;
      G__struct.baseclass[tagnum]->herit[basen]->property=property;
      ++G__struct.baseclass[tagnum]->basen;
   }
#endif
  return(0);
}


/**************************************************************************
* G__tag_memvar_setup()
*
* Used in G__cpplink.C
**************************************************************************/
int G__tag_memvar_setup(int tagnum)
{

   /* Variables stack storing */
   G__IncSetupStack::push();

   if (tagnum >= 0) {
      G__tagnum = tagnum;
      G__p_local=G__struct.memvar[G__tagnum];
      G__def_struct_member = 1;
      G__def_tagnum = G__struct.parent_tagnum[G__tagnum];
      G__tagdefining=G__tagnum;
   }

   return(0);
}

/**************************************************************************
* G__memvar_setup()
*
* Used in G__cpplink.C
**************************************************************************/
int G__memvar_setup(void* p, int type, int reftype, int constvar, int tagnum, int typenum, int statictype, int accessin, const char* expr, int definemacro, const char* comment)
{
   // -- Recreate a member variable from a dictionary entry.
   //  Save some state.
   int store_constvar = G__constvar;
   //
   //  Special case of an enumerator in a static enum,
   //  it has a special initializer, and should be
   //  treated as a global variable.
   //
   int store_def_struct_member = G__def_struct_member;
   int store_tagdefining = G__tagdefining;
   struct G__var_array* store_p_local = G__p_local;
   if ((type == 'p') && G__def_struct_member) {
      // -- Special case of an enumerator in a static enum.
      G__def_struct_member = 0;
      G__tagdefining = -1;
      G__p_local = 0;
   }
   // Note: This may be a global variable address,
   // or it may be an offset into an object, possibly zero.
   G__globalvarpointer = (long) p;
   G__var_type = type;
   G__reftype = reftype;
   G__constvar = constvar;
   G__tagnum = tagnum;
   G__typenum = typenum;
   if ((statictype == G__AUTO) || (statictype == G__AUTOARYDISCRETEOBJ)) {
      G__static_alloc = 0;
   } 
   else if (statictype == G__USING_VARIABLE) {
      G__using_alloc = 1;
   }
   else if (statictype == G__USING_STATIC_VARIABLE) {
      G__using_alloc = 1;
      G__static_alloc = 1;
   }
   else if (statictype == G__LOCALSTATIC) {
      G__static_alloc = 1;
   }
   else if (statictype == G__COMPILEDGLOBAL) {
      G__static_alloc = 1;  // FIXME: This is probaby wrong!
   }
   else {
      // -- File scope static variable, which is actually a static data member.
      G__static_alloc = 1;
   }
   G__access = accessin;
   G__definemacro = definemacro;
   G__setcomment = (char*) comment;
   //
   //  Evaluate the expression to create
   //  the member variable.
   //
   // We are not doing whole function bytecode generation.
   int store_asm_wholefunction = G__asm_wholefunction;
   G__asm_wholefunction = G__ASM_FUNC_NOP;
   // Do not generate bytecode.
   int store_asm_noverflow = G__asm_noverflow;
   G__asm_noverflow = 0;
   // We are not executing, we are declaring.
   int store_prerun = G__prerun;
   G__prerun = 1;
   // Do the actual variable creation.
   G__getexpr((char*)expr);
   if (statictype == G__USING_VARIABLE) {
      G__getexpr((char*)expr);
      
   }
   if ((type == 'p') && store_def_struct_member) {
      // -- Special case of an enumerator in a static enum.
      // Restore state.
      G__def_struct_member = store_def_struct_member;
      G__tagdefining = store_tagdefining;
      G__p_local = store_p_local;
   }
   //
   //  Restore state.
   G__asm_wholefunction = store_asm_wholefunction;
   G__prerun = store_prerun;
   G__asm_noverflow = store_asm_noverflow;
   G__using_alloc = 0;
   //
   //  Force some state.
   //
   G__definemacro = 0;
   G__setcomment = 0;
   G__reftype = G__PARANORMAL;
   //  Restore more state.
   G__constvar = store_constvar;
   // and return.
   return 0;
}

/**************************************************************************
* G__tag_memvar_reset()
*
* Used in G__cpplink.C
**************************************************************************/
int G__tag_memvar_reset()
{

 /* Variables stack restoring */
  G__IncSetupStack::pop();

  return(0);

}

/**************************************************************************
* G__usermemfunc_setup
*
**************************************************************************/
int G__usermemfunc_setup(char *funcname,int hash,int (*funcp)(),int type,
                         int tagnum,int typenum,int reftype,
                         int para_nu,int ansi,int accessin,int isconst,
                         char *paras, char *comment
#ifdef G__TRUEP2F
                         ,void *truep2f,int isvirtual
#endif
                         ,void *userparam
                         )
{
  G__p_ifunc->userparam[G__p_ifunc->allifunc] = userparam;
  return G__memfunc_setup(funcname,hash,(G__InterfaceMethod)funcp,type,tagnum,typenum,reftype,
                          para_nu,ansi,accessin,isconst,paras,comment
#ifdef G__TRUEP2F
                     ,truep2f,isvirtual
#endif
                     );
}

/**************************************************************************
* G__usermemfunc_setup
*
**************************************************************************/
int G__usermemfunc_setup2(char *funcname,int hash,char *mangled_name,
                         int (*funcp)(),int type,
                         int tagnum,int typenum,int reftype,
                         int para_nu,int ansi,int accessin,int isconst,
                         char *paras, char *comment
#ifdef G__TRUEP2F
                         ,void *truep2f,int isvirtual
#endif
                         ,void *userparam
                         )
{
  G__p_ifunc->userparam[G__p_ifunc->allifunc] = userparam;
  return G__memfunc_setup2(funcname,hash,mangled_name,(G__InterfaceMethod)funcp,type,tagnum,typenum,reftype,
                          para_nu,ansi,accessin,isconst,paras,comment
#ifdef G__TRUEP2F
                     ,truep2f,isvirtual
#endif
                     );
}

/**************************************************************************
* G__tag_memfunc_setup()
*
* Used in G__cpplink.C
**************************************************************************/
int G__tag_memfunc_setup(int tagnum)
{
   
   G__IncSetupStack::push();
   
   if (tagnum >= 0) { 
      G__tagdefining = G__struct.parent_tagnum[tagnum];
      G__def_tagnum = G__tagdefining;
      G__tagnum = tagnum;
      G__p_ifunc = G__struct.memfunc[G__tagnum];
   
      while(G__p_ifunc->next) G__p_ifunc=G__p_ifunc->next;
   
      --G__p_ifunc->allifunc;
      G__memfunc_next();
   }
   return(0);
}

/**************************************************************************
* G__memfunc_setup_imp()
* Common part for G__memfunc_setup and G__memfunc_setup2
* Since isvirtual calculation iss different for both of them,
* the code is incompatible now
**************************************************************************/
int G__memfunc_setup_imp(const char *funcname,int hash
                         ,G__InterfaceMethod funcp
                         ,int type,int tagnum,int typenum,int reftype
                         ,int para_nu,int ansi,int accessin,int isconst
                         ,const char *paras, const char *comment
#ifdef G__TRUEP2F
                         ,void *truep2f
                         ,int isvirtual
#endif // G__TRUEP2F
  )
{
#ifndef G__SMALLOBJECT
  int store_func_now = -1;
  struct G__ifunc_table_internal *store_p_ifunc = 0;
  int dtorflag=0;
  (void) isvirtual;

  if('~'==funcname[0] && 0==G__struct.memfunc[G__p_ifunc->tagnum]->hash[0]) {
    store_func_now = G__func_now;
    store_p_ifunc = G__p_ifunc;
    G__p_ifunc = G__struct.memfunc[G__p_ifunc->tagnum];
    G__func_now = 0;
    dtorflag=1;
  }
  G__savestring(&G__p_ifunc->funcname[G__func_now],(char*)funcname);
  G__p_ifunc->hash[G__func_now] = hash;

  // 23/02/2007 
  // We want to have a direct pointer to the function.
  // and we start that value with 0... later we have
  // to register a proper ptr
  G__p_ifunc->funcptr[G__func_now] = 0;

#ifndef G__OLDIMLEMENTATION2012
  if(-1!=G__p_ifunc->tagnum)
    G__p_ifunc->entry[G__func_now].filenum=G__struct.filenum[G__p_ifunc->tagnum];
  else G__p_ifunc->entry[G__func_now].filenum = G__ifile.filenum;
  G__p_ifunc->entry[G__func_now].size = -1;
#else // G__OLDIMLEMENTATION2012
  G__p_ifunc->entry[G__func_now].filenum = -1;
#endif // G__OLDIMLEMENTATION2012
  G__p_ifunc->entry[G__func_now].line_number = -1;
#ifdef G__ASM_WHOLEFUNC
  G__p_ifunc->entry[G__func_now].bytecode = (struct G__bytecodefunc*)NULL;
#endif // G__ASM_WHOLEFUNC
#ifdef G__TRUEP2F
  if(truep2f)
    G__p_ifunc->entry[G__func_now].tp2f=truep2f;
  else G__p_ifunc->entry[G__func_now].tp2f=(void*)funcp;
#endif // G__TRUEP2F

  G__p_ifunc->type[G__func_now] = type;
  G__p_ifunc->p_tagtable[G__func_now] = tagnum;
  G__p_ifunc->p_typetable[G__func_now] = typenum;
  G__p_ifunc->reftype[G__func_now] = reftype;

  G__p_ifunc->para_nu[G__func_now] = para_nu;
  if(ansi&8) G__p_ifunc->ansi[G__func_now] = 2;
  else if(ansi&1) G__p_ifunc->ansi[G__func_now] = 1;

  G__p_ifunc->isconst[G__func_now] = isconst;

  G__p_ifunc->busy[G__func_now] = 0;

  G__p_ifunc->access[G__func_now] = accessin;

  G__p_ifunc->globalcomp[G__func_now] = G__NOLINK;
  G__p_ifunc->isexplicit[G__func_now] = (ansi&4)/4;
  G__p_ifunc->staticalloc[G__func_now] = (ansi&2)/2;

  G__p_ifunc->param[G__func_now][0]->name=(char*)NULL;
  /* parse parameter setup information */
  G__parse_parameter_link((char*)paras);

#ifdef G__FRIEND
  G__p_ifunc->friendtag[G__func_now] = (struct G__friendtag*)NULL;
#endif // G__FRIEND

  G__p_ifunc->comment[G__func_now].p.com = (char*)comment;
  if(comment) G__p_ifunc->comment[G__func_now].filenum = -2;
  else        G__p_ifunc->comment[G__func_now].filenum = -1;

  /* end */

  /* set entry pointer */
  G__p_ifunc->pentry[G__func_now] = &G__p_ifunc->entry[G__func_now];
  G__p_ifunc->entry[G__func_now].p=(void*)funcp;

  /* Stub Pointer Adjustement */
  G__p_ifunc->entry[G__func_now].ptradjust = 0;

  // Stub Pointer initialisation.
  // If funcp parameter is null and there is no mangled name. It means that the stub is in a base class.
  // We have to look for the stub pointer in the base classes.
  if (!G__p_ifunc->mangled_name[G__func_now]&&!funcp&&(G__p_ifunc->isvirtual[G__func_now])&&(G__p_ifunc->access[G__func_now]==G__PUBLIC)){

    // tagnum's Base Classes structure
    G__inheritance* cbases = G__struct.baseclass[G__p_ifunc->tagnum];

    // If there's any base class
    if (cbases){

      // Ifunc Method's index in the base class
      int base = -1;

      // Stub function pointer
      void * basefuncp = 0;

      // Go through the bases tagnums if there are base classes and a valid stub is not found yet
      for (int idx=0; (idx < cbases->basen)&&(!basefuncp); ++idx){

        // Current tagnum
        int basetagnum = cbases->herit[idx]->basetagnum;

        // Warning: Global G__p_ifunc is modified in G__incsetup_memfunc
        // We save and later we restore the G__p_ifunc's value
        G__ifunc_table_internal* store_ifunc = G__p_ifunc;

        // We force memfunc_setup for the base classes
        G__incsetup_memfunc(basetagnum);

        // Restore G__p_ifunc
        G__p_ifunc = store_ifunc;

        // Current Base Class ifunc table
        G__ifunc_table_internal* ifunct = G__struct.memfunc[basetagnum];

        // Look for the method in the base class
        G__ifunc_table_internal* found = G__ifunc_exist(G__p_ifunc, G__func_now, ifunct, &base, 0xffff); 

        // Method found
        if((base!=-1)&&found){

          // Method's stub pointer
          basefuncp = found->entry[base].p;

          G__value ptr = G__null;

          ptr.tagnum = G__p_ifunc->tagnum;
          ptr.type = 'C';
          ptr.typenum = -1;
          ptr.obj.i = 0;

          ptr = G__castvalue_bc(G__fulltagname(found->tagnum, 0),ptr, 0);

          // Pointer Adjustement
          G__p_ifunc->entry[G__func_now].ptradjust = found->entry[base].ptradjust + ptr.obj.i;

          // Method's stub pointer found.
          // We update the current method stub pointer
          G__p_ifunc->entry[G__func_now].p= (void *) basefuncp;

          if(truep2f)
            G__p_ifunc->entry[G__func_now].tp2f=truep2f;
          else G__p_ifunc->entry[G__func_now].tp2f=(void*) basefuncp;

        }

      }

    }

  }
#ifndef G__OLDIMPLEMENTATION1702
  {
    struct G__ifunc_table_internal *ifunc;
    int iexist;
    if(-1==G__p_ifunc->tagnum)
      ifunc = G__ifunc_exist(G__p_ifunc,G__func_now
                             ,&G__ifunc,&iexist,0xffff);
    else
      ifunc = G__ifunc_exist(G__p_ifunc,G__func_now
                             ,G__struct.memfunc[G__p_ifunc->tagnum],&iexist
                             ,0xffff);

    if(ifunc) {
      /* Overriding old function definition */
      int func_now = G__func_now;
      int paranu,iin;
      /* Overriding old function definition */
      ifunc->ansi[iexist]=G__p_ifunc->ansi[func_now];
      if(-1==G__p_ifunc->para_nu[func_now]) paranu=0;
      else paranu=ifunc->para_nu[iexist];
      if(0==ifunc->ansi[iexist])
        ifunc->para_nu[iexist] = G__p_ifunc->para_nu[func_now];
      ifunc->type[iexist]=G__p_ifunc->type[func_now];
      ifunc->p_tagtable[iexist]=G__p_ifunc->p_tagtable[func_now];
      ifunc->p_typetable[iexist]=G__p_ifunc->p_typetable[func_now];
      ifunc->reftype[iexist]=G__p_ifunc->reftype[func_now];
      ifunc->isconst[iexist]|=G__p_ifunc->isconst[func_now];
      ifunc->isexplicit[iexist]|=G__p_ifunc->isexplicit[func_now];
      for(iin=0;iin<paranu;iin++) {
        ifunc->param[iexist][iin]->reftype
          =G__p_ifunc->param[func_now][iin]->reftype;
        ifunc->param[iexist][iin]->p_typetable
          =G__p_ifunc->param[func_now][iin]->p_typetable;
        if(G__p_ifunc->param[func_now][iin]->pdefault) {
          if(-1!=(long)G__p_ifunc->param[func_now][iin]->pdefault)
            free((void*)G__p_ifunc->param[func_now][iin]->pdefault);
          free((void*)G__p_ifunc->param[func_now][iin]->def);
        }
        G__p_ifunc->param[func_now][iin]->pdefault=(G__value*)NULL;
        G__p_ifunc->param[func_now][iin]->def=(char*)NULL;
        if(ifunc->param[iexist][iin]->name) {
          if(G__p_ifunc->param[func_now][iin]->name) {
            free((void*)G__p_ifunc->param[func_now][iin]->name);
            G__p_ifunc->param[func_now][iin]->name=(char*)NULL;
          }
        }
        else {
          ifunc->param[iexist][iin]->name=G__p_ifunc->param[func_now][iin]->name;
          G__p_ifunc->param[func_now][iin]->name=(char*)NULL;
        }
      }
      ifunc->entry[iexist]=G__p_ifunc->entry[func_now];
      /* The copy in previous get the wrong tp2f ... let's restore it */
      ifunc->entry[iexist].tp2f = (void*)ifunc->funcname[iexist];
      ifunc->pentry[iexist]= &ifunc->entry[iexist];
      if(1==ifunc->ispurevirtual[iexist]) {
        ifunc->ispurevirtual[iexist]=G__p_ifunc->ispurevirtual[func_now];
        if(G__tagdefining>=0) --G__struct.isabstract[G__tagdefining];
      }
      else if(1==G__p_ifunc->ispurevirtual[func_now]) {
        ifunc->ispurevirtual[iexist]=G__p_ifunc->ispurevirtual[func_now];
      }
      if((ifunc!=G__p_ifunc || iexist!=func_now) &&
         G__p_ifunc->funcname[func_now]) {
        free((void*)G__p_ifunc->funcname[func_now]);
        G__p_ifunc->funcname[func_now] = (char*)NULL;
      }
      if((ifunc!=G__p_ifunc || iexist!=func_now) &&
         G__p_ifunc->mangled_name[func_now]) {
        free((void*)G__p_ifunc->mangled_name[func_now]);
        G__p_ifunc->mangled_name[func_now] = (char*)NULL;
      }
    }
    else {
      if(dtorflag) {
        G__func_now = store_func_now;
        G__p_ifunc = store_p_ifunc;
      }
      else {
        G__memfunc_next();
      }
    }
  }
#else // G__OLDIMPLEMENTATION1702
 {
   const char* isTemplate = strchr(funcname, '<');
   if (isTemplate
       // no op <()
       && ( !strncmp(funcname, "operator", 8)
            // no A<int>()
            || (tagnum != -1 && !strcmp(funcname, G__struct.name[tagnum]))))
      isTemplate = 0;

   if (isTemplate) {
      G__FastAllocString funcname_notmplt(funcname);
      *(funcname_notmplt + (isTemplate - funcname)) = 0; // cut at template arg
      isTemplate = funcname_notmplt;
      int tmplthash = 0;
      while (*isTemplate) {
         tmplthash += *isTemplate;
         ++isTemplate;
      }

      struct G__ifunc_table_internal *ifunc;
      int iexist;
      char *oldname = G__p_ifunc->funcname[G__func_now];
      G__p_ifunc->funcname[G__func_now] = funcname_notmplt;
      G__p_ifunc->hash[G__func_now] = tmplthash;

      if(-1==G__p_ifunc->tagnum)
         ifunc = G__ifunc_exist(G__p_ifunc,G__func_now
                                ,&G__ifunc,&iexist,0xffff);
      else
         ifunc = G__ifunc_exist(G__p_ifunc,G__func_now
                                ,G__struct.memfunc[G__p_ifunc->tagnum],&iexist
                                ,0xffff);

      G__p_ifunc->funcname[G__func_now] = oldname;
      G__p_ifunc->hash[G__func_now] = hash;

      if(dtorflag) {
         G__func_now = store_func_now;
         G__p_ifunc = store_p_ifunc;
      }
      else {
         G__memfunc_next();
      }

      if (!ifunc) {
         // create a copy of this function, name without template arguments, if
         // that function doesn't exist yet.
         G__memfunc_setup_imp(funcname_notmplt, tmplthash, funcp, type, tagnum, typenum, reftype
                         , para_nu, ansi, accessin, isconst , paras,  comment
#ifdef G__TRUEP2F
                         ,truep2f
                         ,isvirtual
#endif // G__TRUEP2F
                           );
      }
   } else {
      if(dtorflag) {
         G__func_now = store_func_now;
         G__p_ifunc = store_p_ifunc;
      }
      else {
         G__memfunc_next();
      }
   }
 }
#endif // G__OLDIMPLEMENTATION1702


#endif // G__SMALLOBJECT
  //if (G__p_ifunc->param[0][para_nu-1]) return (1);
  return(0);
}


/**************************************************************************
 * G__memfunc_setup()
 *
 * Used in G__cpplink.C
 **************************************************************************/
int G__memfunc_setup(const char *funcname,int hash
                     ,G__InterfaceMethod funcp
                     ,int type,int tagnum,int typenum,int reftype
                     ,int para_nu,int ansi,int accessin,int isconst
                     ,const char *paras, const char *comment
#ifdef G__TRUEP2F
                     ,void *truep2f
                     ,int isvirtual
#endif // G__TRUEP2F
  )
{
#ifndef G__SMALLOBJECT
  int store_func_now = -1;
  struct G__ifunc_table_internal *store_p_ifunc = 0;
  int dtorflag=0;

  if (G__p_ifunc->allifunc == G__MAXIFUNC) {
    G__p_ifunc->next=(struct G__ifunc_table_internal *)malloc(sizeof(struct G__ifunc_table_internal));
    memset(G__p_ifunc->next,0,sizeof(struct G__ifunc_table_internal));
    G__p_ifunc->next->allifunc=0;
    G__p_ifunc->next->next=(struct G__ifunc_table_internal *)NULL;
    G__p_ifunc->next->page = G__p_ifunc->page+1;
    G__p_ifunc->next->tagnum = G__p_ifunc->tagnum;
    G__p_ifunc = G__p_ifunc->next;
    {
      int ix;
      for(ix=0;ix<G__MAXIFUNC;ix++) {
        G__p_ifunc->funcname[ix] = (char*)NULL;
        G__p_ifunc->userparam[ix] = 0;
      }
    }

    //G__fprinterr(G__serr, "Attempt to add function %s failed - ifunc_table overflow!\n",
    //     funcname);
    // return 0;
  }
  G__func_now=G__p_ifunc->allifunc;

  if('~'==funcname[0] && 0==G__struct.memfunc[G__p_ifunc->tagnum]->hash[0]) {
    store_func_now = G__func_now;
    store_p_ifunc = G__p_ifunc;
    G__p_ifunc = G__struct.memfunc[G__p_ifunc->tagnum];
    G__func_now = 0;
    dtorflag=1;
  }
  //G__savestring(&G__p_ifunc->funcname[G__func_now],(char*)funcname);
  //G__p_ifunc->hash[G__func_now] = hash;

#ifdef G__TRUEP2F
  G__p_ifunc->isvirtual[G__func_now] = isvirtual&0x01;
  G__p_ifunc->ispurevirtual[G__func_now] = (isvirtual&0x02)/2;

  G__p_ifunc->page_base = -1; // indicates this is function called with the old memfunc
#else
  G__p_ifunc->isvirtual[G__func_now] = 0;
  G__p_ifunc->ispurevirtual[G__func_now] = 0;

  G__p_ifunc->page_base = -1; // indicates this is function called with the old memfunc
#endif

  if(dtorflag) {
    G__func_now = store_func_now;
    G__p_ifunc = store_p_ifunc;
  }

  return G__memfunc_setup_imp(funcname, hash, funcp, type, tagnum, typenum, reftype, para_nu, ansi, 
                              accessin, isconst, paras, comment
#ifdef G__TRUEP2F
                              ,truep2f
                              ,isvirtual
#endif
    );

#endif
}


/**************************************************************************
 * G__memfunc_setup2()
 *
 * Used in G__cpplink.C
 *
 * 10-07-07
 * This is the same this as G__memfunc_setup but since we can not overload
 * methods in C we have to duplicate code
 **************************************************************************/
int G__memfunc_setup2(const char *funcname,int hash,const char *mangled_name
                      ,G__InterfaceMethod funcp
                      ,int type,int tagnum,int typenum,int reftype
                      ,int para_nu,int ansi,int accessin,int isconst
                      ,const char *paras, const char *comment
#ifdef G__TRUEP2F
                      ,void *truep2f
                      ,int isvirtual
#endif
  )
{
  struct G__ifunc_table_internal * store_p_ifunc = G__p_ifunc;
  int store_func_now = G__func_now;
  int dtorflag=0;

#ifndef G__SMALLOBJECT
  if (G__p_ifunc->allifunc == G__MAXIFUNC) {
    G__p_ifunc->next=(struct G__ifunc_table_internal *)malloc(sizeof(struct G__ifunc_table_internal));
    memset(G__p_ifunc->next,0,sizeof(struct G__ifunc_table_internal));
    G__p_ifunc->next->allifunc=0;
    G__p_ifunc->next->next=(struct G__ifunc_table_internal *)NULL;
    G__p_ifunc->next->page = G__p_ifunc->page+1;
    G__p_ifunc->next->tagnum = G__p_ifunc->tagnum;
    G__p_ifunc = G__p_ifunc->next;
    {
      int ix;
      for(ix=0;ix<G__MAXIFUNC;ix++) {
        G__p_ifunc->funcname[ix] = (char*)NULL;
        G__p_ifunc->userparam[ix] = 0;
      }
    }

    //G__fprinterr(G__serr, "Attempt to add function %s failed - ifunc_table overflow!\n",
    //     funcname);
    // return 0;
  }
  G__func_now=G__p_ifunc->allifunc;

  // 02-10-07
  // Do this only in setup_impl()
  if('~'==funcname[0] && 0==G__struct.memfunc[G__p_ifunc->tagnum]->hash[0]) {
    store_func_now = G__func_now;
    store_p_ifunc = G__p_ifunc;
    G__p_ifunc = G__struct.memfunc[G__p_ifunc->tagnum];
    G__func_now = 0;
    dtorflag=1;
  }
  //G__savestring(&G__p_ifunc->funcname[G__func_now],(char*)funcname);
  //G__p_ifunc->hash[G__func_now] = hash;

  // 06-07-07
  // Keep the mangled name in addition to everything else
  
  // 24-01-08
  // Don't copy the string... just point to it since it should 
  // in the .o
  //if(mangled_name)
  //  G__savestring(&G__p_ifunc->mangled_name[G__func_now],(char*)mangled_name);
  // Be careful... this pointer can't be changed
  G__p_ifunc->mangled_name[G__func_now] = (char *)mangled_name;

  // 06-08-07
  // new virtual flags
#ifdef G__TRUEP2F
  G__p_ifunc->ispurevirtual[G__func_now] = isvirtual&0x01;
  isvirtual = isvirtual/2;
  G__p_ifunc->page_base = isvirtual;
  G__p_ifunc->isvirtual[G__func_now] = (isvirtual?0x01:0x00);
#else // G__TRUEP2F
  G__p_ifunc->isvirtual[G__func_now] = 0;
  G__p_ifunc->ispurevirtual[G__func_now] = 0;
#endif // G__TRUEP2F

  // 02-10-07
  // Do this only in setup_impl()
  if(dtorflag) {
    G__func_now = store_func_now;
    G__p_ifunc = store_p_ifunc;
  }

  return G__memfunc_setup_imp(funcname, hash, funcp, type, tagnum, typenum, reftype, para_nu, ansi, 
                              accessin, isconst, paras, comment
#ifdef G__TRUEP2F
                              ,truep2f
                              ,isvirtual
#endif
    );

#endif
}

} // extern "C"

/**************************************************************************
* G__separate_parameter()
*
**************************************************************************/
int G__separate_parameter(const char *original,int *pos, G__FastAllocString& param)
{
#ifndef G__SMALLOBJECT
   int single_quote=0;
   int double_quote=0;
   int single_arg_quote=0;
   bool argStartsWithSingleQuote = false;

   int startPos = (*pos);
   if (original[startPos] == '\'') {
      // don't put beginning ' into param
      ++startPos;
      argStartsWithSingleQuote = true;
      single_arg_quote = 1;
   }

   int iParam = 0;
   int i = startPos;
   bool done = false;
   for(; !done; ++i) {
      int c = original[i];
      switch(c) {
       case '\'':
          if (!double_quote) {
             if (single_quote) single_quote = 0;
             // only turn on single_quote if at the beginning!
             else if (i == startPos)  single_quote = 1;
             else if (single_arg_quote) single_arg_quote = 0;
          }
          break;
       case '"':
          if (!single_quote) double_quote ^= 1;
          break;
       case ' ':
          if(!single_quote && !double_quote && !single_arg_quote) {
             c = 0;
             done = true;
          }
          break;
       case '\\':
          if (single_quote || double_quote) {
             // prevent special treatment of next char

             param.Set(iParam++, c);
             c = original[++i];
          }
          break;
       case 0:
          done = true;
          break;
      }

      param.Set(iParam++, c);
   }

   if (argStartsWithSingleQuote && ! param[iParam - 1] && param[iParam - 2] == '\'' )
      param.Set(iParam - 2, 0); // skip trailing '
   *pos = i;

   if (i > startPos) --i;
   else i = startPos;

   return original[i];

#endif
}

extern "C" {

/**************************************************************************
* G__parse_parameter_link()
*
* Used in G__cpplink.C
**************************************************************************/
int G__parse_parameter_link(char* paras)
{
#ifndef G__SMALLOBJECT
  int type;
  int tagnum;
  int reftype_const;
  int typenum;
  G__value* para_default = 0;
  G__FastAllocString c_type(10);
  G__FastAllocString tagname(G__MAXNAME*6);
  G__FastAllocString type_name(G__MAXNAME*6);
  G__FastAllocString c_reftype_const(10);
  G__FastAllocString c_default(G__MAXNAME*2);
  G__FastAllocString c_paraname(G__MAXNAME*2);
  int os = 0;

  int store_loadingDLL = G__loadingDLL;
  G__loadingDLL = 1;

  int store_var_type = G__var_type;

  char ch = paras[0];
  for (int ifn = 0; ch != '\0'; ++ifn) {
    G__separate_parameter(paras, &os, c_type);
    type = c_type[0];
    G__separate_parameter(paras, &os, tagname);
    if (tagname[0] == '-') {
      tagnum = -1;
    }
    else {
      // -- We have a tagname.
      // save G__p_ifunc, G__search_tagname might change it when autoloading
      G__ifunc_table_internal* current_G__p_ifunc = G__p_ifunc;
      // Note: This is the only place we call G__search_tagname
      //       with a type of 0, this is causing problems with
      //       the switch to reflex.
      tagnum = G__search_tagname(tagname, isupper(type) ? 0xff : 0);
      G__p_ifunc = current_G__p_ifunc;
    }
    G__separate_parameter(paras, &os, type_name);
    if (type_name[0] == '-') {
      typenum = -1;
    }
    else {
      if (type_name[0] == '\'') {
        type_name[::strlen(type_name)-1] = '\0';
        typenum = G__defined_typename(type_name + 1);
      }
      else {
        typenum = G__defined_typename(type_name);
      }
    }
    G__separate_parameter(paras, &os, c_reftype_const);
    reftype_const = ::atoi(c_reftype_const);
#ifndef G__OLDIMPLEMENTATION1861
    //if (typenum != -1) {
    // NO - this is already taken into account when writing the dictionary
    //  reftype_const += G__newtype.isconst[typenum] * 10;
    //}
#endif
    G__separate_parameter(paras, &os, c_default);
    if ((c_default[0] == '-') && (c_default[1] == '\0')) {
      para_default = 0;
      c_default[0] = '\0';
    }
    else {
      para_default = (G__value*) -1;
    }
    ch = G__separate_parameter(paras, &os, c_paraname);
    if (c_paraname[0] == '-') {
      c_paraname[0] = '\0';
    }
    G__memfunc_para_setup(ifn, type, tagnum, typenum, reftype_const, para_default, c_default, c_paraname);
  }
  G__var_type = store_var_type;
  G__loadingDLL = store_loadingDLL;
#endif
  return 0;
}

/**************************************************************************
* G__memfunc_para_setup()
*
* Used in G__cpplink.C
**************************************************************************/
int G__memfunc_para_setup(int ifn,int type,int tagnum,int typenum,int reftype_const,G__value *para_default
                          ,char *para_def,char *para_name
                          )
{
#ifndef G__SMALLOBJECT
  G__p_ifunc->param[G__func_now][ifn]->type = type;
  G__p_ifunc->param[G__func_now][ifn]->p_tagtable = tagnum;
  G__p_ifunc->param[G__func_now][ifn]->p_typetable = typenum;
#if !defined(G__OLDIMPLEMENTATION1975)
  G__p_ifunc->param[G__func_now][ifn]->isconst = (reftype_const/10)%10;
  G__p_ifunc->param[G__func_now][ifn]->reftype = reftype_const-(reftype_const/10%10*10);
#else
  G__p_ifunc->param[G__func_now][ifn]->reftype = reftype_const%10;
  G__p_ifunc->param[G__func_now][ifn]->isconst = reftype_const/10;
#endif
  G__p_ifunc->param[G__func_now][ifn]->pdefault = para_default;
  if(para_def[0]
     || (G__value*)NULL!=para_default
     ) {
    G__p_ifunc->param[G__func_now][ifn]->def=(char*)malloc(strlen(para_def)+1);
    strcpy(G__p_ifunc->param[G__func_now][ifn]->def,para_def); // Okay, we allocated the right size
  }
  else {
    G__p_ifunc->param[G__func_now][ifn]->def=(char*)NULL;
  }
  if(para_name[0]) {
    G__p_ifunc->param[G__func_now][ifn]->name=(char*)malloc(strlen(para_name)+1);
    strcpy(G__p_ifunc->param[G__func_now][ifn]->name,para_name); // Okay, we allocated the right size
  }
  else {
    G__p_ifunc->param[G__func_now][ifn]->name=(char*)NULL;
  }
#endif
  return(0);
}

/**************************************************************************
* G__memfunc_next()
*
* Used in G__cpplink.C
**************************************************************************/
int G__memfunc_next()
{
#ifndef G__SMALLOBJECT
  /* increment count */
  if(G__p_ifunc->allifunc < G__MAXIFUNC)
     ++G__p_ifunc->allifunc;
  /***************************************************************
   * Allocate and initialize function table list
   ***************************************************************/
  if(G__p_ifunc->allifunc >= G__MAXIFUNC) {
    G__p_ifunc->next=(struct G__ifunc_table_internal *)malloc(sizeof(struct G__ifunc_table_internal));
    memset(G__p_ifunc->next,0,sizeof(struct G__ifunc_table_internal));
    G__p_ifunc->next->allifunc=0;
    G__p_ifunc->next->next=(struct G__ifunc_table_internal *)NULL;
    G__p_ifunc->next->page = G__p_ifunc->page+1;
    G__p_ifunc->next->tagnum = G__p_ifunc->tagnum;

    /* set next G__p_ifunc */
    G__p_ifunc = G__p_ifunc->next;
    {
      int ix;
      for(ix=0;ix<G__MAXIFUNC;ix++) {
        G__p_ifunc->funcname[ix] = (char*)NULL;
        G__p_ifunc->userparam[ix] = 0;
      }
    }
    {
      int ix;
      for(ix=0;ix<G__MAXIFUNC;ix++) G__p_ifunc->userparam[ix] = 0;
    }
  }
#endif
  return(0);
}

/**************************************************************************
* G__tag_memfunc_reset()
*
* Used in G__cpplink.C
**************************************************************************/
int G__tag_memfunc_reset()
{
   /* Variables stack restoring */
   G__IncSetupStack::pop();
   
   return(0);
}
#ifdef G__NEVER
/**************************************************************************
* G__p2ary_setup()
*
* Used in G__cpplink.C
**************************************************************************/
int G__p2ary_setup(n,...)
int n)
{
   return(0);
}
#endif


/**************************************************************************
* G__cppif_p2memfunc(fp)
*
* Used in G__cpplink.C
**************************************************************************/
int G__cppif_p2memfunc(FILE *fp)
{
  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Get size of pointer to member function\n");
  fprintf(fp,"*********************************************************/\n");
  fprintf(fp,"class G__Sizep2memfunc%s {\n",G__DLLID);
  fprintf(fp," public:\n");
  fprintf(fp,"  G__Sizep2memfunc%s(): p(&G__Sizep2memfunc%s::sizep2memfunc) {}\n"
          ,G__DLLID,G__DLLID);
  fprintf(fp,"    size_t sizep2memfunc() { return(sizeof(p)); }\n");
  fprintf(fp,"  private:\n");
  fprintf(fp,"    size_t (G__Sizep2memfunc%s::*p)();\n",G__DLLID);
  fprintf(fp,"};\n\n");

  fprintf(fp,"size_t G__get_sizep2memfunc%s()\n",G__DLLID);
  fprintf(fp,"{\n");
  fprintf(fp,"  G__Sizep2memfunc%s a;\n",G__DLLID);
  fprintf(fp,"  G__setsizep2memfunc((int)a.sizep2memfunc());\n");
  fprintf(fp,"  return((size_t)a.sizep2memfunc());\n");
  fprintf(fp,"}\n\n");
  return(0);
}

/**************************************************************************
* G__set_sizep2memfunc(fp)
*
* Used in G__cpplink.C
**************************************************************************/
int G__set_sizep2memfunc(FILE *fp)
{
  fprintf(fp,"\n   if(0==G__getsizep2memfunc()) G__get_sizep2memfunc%s();\n"
          ,G__DLLID);
  return(0);
}

} // extern "C"

/**************************************************************************
* G__getcommentstring()
*
**************************************************************************/
int G__getcommentstring(G__FastAllocString& buf,int tagnum,G__comment_info *pcomment)
{
   G__FastAllocString temp(G__LONGLINE);
   G__getcomment(temp,pcomment,tagnum);
   if('\0'==temp[0]) {
      buf = "(char*)NULL";
   }
   else {
      G__add_quotation(temp,buf);
   }
   return(1);
}

extern "C" {

/**************************************************************************
* G__pragmalinkenum()
**************************************************************************/
static void G__pragmalinkenum(int tagnum,int globalcomp)
{
  /* double check tagnum points to a enum */
  if(-1==tagnum || 'e'!=G__struct.type[tagnum]) return;

  /* enum in global scope */
  if(-1==G__struct.parent_tagnum[tagnum]
     || G__nestedclass
     ) {
    int ig15;
    struct G__var_array *var = &G__global;
    while(var) {
      for(ig15=0;ig15<var->allvar;ig15++) {
        /* modify globalcomp flag if enum member */
        if(tagnum==var->p_tagtable[ig15]) var->globalcomp[ig15]=globalcomp;
      }
      var=var->next;
    }
  }
  /* enum enclosed in class  */
  else {
    /* do nothing, should already be OK. */
  }
}

#if !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
/**************************************************************************
* G__linknestedtypedef()
**************************************************************************/
static void G__linknestedtypedef(int tagnum,int globalcomp)
{
  int i;
  for(i=0;i<G__newtype.alltype;i++) {
    if (G__newtype.parent_tagnum[i] == -1) continue;
    if (G__newtype.parent_tagnum[i]==tagnum) {
      G__newtype.globalcomp[i] = globalcomp;
    }
  }
}
#endif

/**************************************************************************
* G__specify_link()
*
* #pragma link C++ class ClassName;      can use regexp
* #pragma link C   class ClassName;      can use regexp
* #pragma link off class ClassName;      can use regexp
* #ifdef G__ROOTSPECIAL
* #pragma link off class ClassName-;     set ROOT specific flag
* #pragma link off options=OPTION,OPTION class ClassName; set ROOT specific flag
* #endif
* #pragma link C++ enum ClassName;      can use regexp
* #pragma link C   enum ClassName;      can use regexp
* #pragma link off enum ClassName;      can use regexp
*
* #pragma link C++ nestedclass;
* #pragma link C   nestedclass;
* #pragma link off nestedclass;
*
* #pragma link C++ nestedtypedef;
* #pragma link C   nestedtypedef;
* #pragma link off nestedtypedef;
*
* #pragma link C++ function funcname;    can use regexp
* #pragma link C   function funcname;    can use regexp
* #pragma link off function funcname;    can use regexp
* #pragma link MACRO function funcname;  can use regexp
* #pragma stub C++ function funcname;    can use regexp
* #pragma stub C   function funcname;    can use regexp
*
* #pragma link [C++|off] operators classname;
*
* #pragma link C++ global variablename;  can use regexp
* #pragma link C   global variablename;  can use regexp
* #pragma link off global variablename;  can use regexp
*
* #pragma link C++ defined_in filename;
* #pragma link C   defined_in filename;
* #pragma link off defined_in filename;
* #pragma link C++ defined_in classname;
* #pragma link C   defined_in classname;
* #pragma link off defined_in classname;
* #pragma link C++ defined_in [class|struct|namespace] classname;
* #pragma link C   defined_in [class|struct|namespace] classname;
* #pragma link off defined_in [class|struct|namespace] classname;
*
* #pragma link C++ typedef TypeName;      can use regexp
* #pragma link C   typedef TypeName;      can use regexp
* #pragma link off typedef TypeName;      can use regexp
*
* #pragma link off all classes;
* #pragma link off all functions;
* #pragma link off all variables;
* #pragma link off all typedefs;
* #pragma link off all methods;
*
* #pragma link [C++|off] all_method     ClassName;
* #pragma link [C++|off] all_datamember ClassName;
*              ^
*
* #pragma link postprocess file func;
*
* For ROOT only:
* #praga link C++ ioctortype ClassName;
*
**************************************************************************/
void G__specify_link(int link_stub)
{
  int c;
  G__FastAllocString buf(G__ONELINE);
  int globalcomp=G__NOLINK;
  /* int store_globalcomp; */
  int i;
  int hash;
  struct G__ifunc_table_internal *ifunc;
  struct G__var_array *var;
#ifdef G__REGEXP
  regex_t re;
  int regstat;
#endif
#ifdef G__REGEXP1
  char *re;
#endif
  int os;
  char *p;
  int done=0;

  /* Get link language interface */
  c = G__fgetname_template(buf, 0, ";\n\r");

  if(strncmp(buf,"postproc",5)==0) {
    int store_globalcomp2 = G__globalcomp;
    int store_globalcomp3 = G__store_globalcomp;
    int store_prerun = G__prerun;
    G__globalcomp = G__NOLINK;
    G__store_globalcomp = G__NOLINK;
    G__prerun = 0;
    c = G__fgetname_template(buf, 0, ";");
    if(G__LOADFILE_SUCCESS<=G__loadfile(buf)) {
      G__FastAllocString buf2(G__ONELINE);
      c = G__fgetstream(buf2, 0, ";");
      G__calc(buf2);
      G__unloadfile(buf);
    }
    G__globalcomp = store_globalcomp2;
    G__store_globalcomp = store_globalcomp3;
    G__prerun = store_prerun;
    if(';'!=c) G__fignorestream(";");
    return;
  }

  if(strncmp(buf,"default",3)==0) {
    c=G__read_setmode(&G__default_link);
    if('\n'!=c&&'\r'!=c) G__fignoreline();
    return;
  }

  if(G__SPECIFYLINK==link_stub) {
    if(strcmp(buf,"C++")==0) {
      globalcomp = G__CPPLINK;
      if(G__CLINK==G__globalcomp) {
        G__fprinterr(G__serr,"Warning: '#pragma link C++' ignored. Use '#pragma link C'");
        G__printlinenum();
      }
    }
    else if(strcmp(buf,"C")==0) {
      globalcomp = G__CLINK;
      if(G__CPPLINK==G__globalcomp) {
        G__fprinterr(G__serr,"Warning: '#pragma link C' ignored. Use '#pragma link C++'");
        G__printlinenum();
      }
    }
    else if(strcmp(buf,"MACRO")==0) globalcomp = G__MACROLINK;
    else if(strcmp(buf,"off")==0) globalcomp = G__NOLINK;
    else if(strcmp(buf,"OFF")==0) globalcomp = G__NOLINK;
    else {
      G__genericerror("Error: '#pragma link' syntax error");
      globalcomp = G__NOLINK; /* off */
    }
  }
  else {
    if(strcmp(buf,"C++")==0) {
      globalcomp = G__CPPSTUB;
      if(G__CLINK==G__globalcomp) {
        G__fprinterr(G__serr,"Warning: '#pragma stub C++' ignored. Use '#pragma stub C'");
        G__printlinenum();
      }
    }
    else if(strcmp(buf,"C")==0) {
      globalcomp = G__CSTUB;
      if(G__CPPLINK==G__globalcomp) {
        G__fprinterr(G__serr,"Warning: '#pragma stub C' ignored. Use '#pragma stub C++'");
        G__printlinenum();
      }
    }
    else if(strcmp(buf,"MACRO")==0) globalcomp = G__MACROLINK;
    else if(strcmp(buf,"off")==0) globalcomp = G__NOLINK;
    else if(strcmp(buf,"OFF")==0) globalcomp = G__NOLINK;
    else {
      G__genericerror("Error: '#pragma link' syntax error");
      globalcomp = G__NOLINK; /* off */
    }
  }

  if(';'==c)  return;

  /* Get type of language construct */
  c = G__fgetname_template(buf, 0, ";\n\r");

  if(G__MACROLINK==globalcomp&&strncmp(buf,"function",3)!=0) {
    G__fprinterr(G__serr,"Warning: #pragma link MACRO only valid for global function. Ignored\n");
    G__printlinenum();
    c=G__fignorestream(";\n");
    return;
  }

  int rfNoStreamer = 0;
  int rfNoInputOper = 0;
  int rfUseBytecount = 0;
  int rfNoMap = 0;
  int rfUseStubs = 0;
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
  if (!strncmp(buf,"options=", 8) || !strncmp(buf,"option=", 7)) {
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
        else if (*iOpt == "stub") rfUseStubs = 1;
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
              for( std::string::size_type i = 0; i<verStr.size(); ++i )
                 if( !isdigit( verStr[i] ) ) noDigit = true;

              if( noDigit )
                 G__fprinterr(G__serr, "Malformed version option! \"%s\" is not a non-negative number!\n", verStr.c_str() );
              else
                 rfVersionNumber = atoi( verStr.c_str() );
           }
        }
        else {
           G__printlinenum();
           G__fprinterr(G__serr,"Warning: ignoring unknown #pragma link option=%s\n", iOpt->c_str());
        }

     // fetch next token
     c = G__fgetname_template(buf, 0, ";\n\r");
  }


  /*************************************************************************
  * #pragma link [spec] nestedclass;
  *************************************************************************/
  if(strncmp(buf,"nestedclass",3)==0) {
    G__nestedclass = globalcomp;
  }

  if(strncmp(buf,"nestedtypedef",7)==0) {
    G__nestedtypedef=globalcomp;
  }

  if(';'==c)  return;

  switch(globalcomp) {
  case G__CPPSTUB:
  case G__CSTUB:
    if(strncmp(buf,"function",3)!=0) {
      G__fprinterr(G__serr,"Warning: #pragma stub only valid for global function. Ignored\n");
      c=G__fignorestream(";\n");
      return;
    }
    break;
  default:
    break;
  }

  /*************************************************************************
  * #pragma link [spec] class [name];
  *************************************************************************/
  if(strncmp(buf,"class",3)==0||strncmp(buf,"struct",3)==0||
     strncmp(buf,"union",3)==0||strncmp(buf,"enum",3)==0
#ifndef G__OLDIKMPLEMENTATION1242
     || strncmp(buf,"namespace",3)==0
#endif
     ) {
    int len;
    char* p2;
    int iirf;
#ifndef G__OLDIKMPLEMENTATION1334
    char protectedaccess=0;
#ifndef G__OLDIKMPLEMENTATION1483
    if(strncmp(buf,"class+protected",10)==0)
      protectedaccess=G__PROTECTEDACCESS;
    else if(strncmp(buf,"class+private",10)==0) {
      protectedaccess=G__PRIVATEACCESS;
      G__privateaccess = 1;
    }
#else
    if(strncmp(buf,"class+protected",6)==0) protectedaccess = 1;
#endif
#endif
    c = G__fgetstream_template(buf, 0, ";\n\r");
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
    p2 = strchr(buf,'[');
    p = strrchr(buf,'*');
    if(len&&p&&(p2||'*'==buf[len-1]||('>'!=buf[len-1]&&'-'!=buf[len-1]))) {
      if(*(p+1)=='>') p=(char*)NULL;
      //else p=p;
    }
    else p=(char*)NULL;
    if(p) {
       if (p - buf > 2 && p[-1] == ':' && p[-2] == ':') {
          // pragma link C++ class A::*
          p[-2] = 0;
          int scopetag = G__defined_tagname(buf,1);
          if (scopetag >= 0) {
             ++done;
             // A exists. A::B must be after A in G__struct:
             for (int t = scopetag + 1; t < G__struct.alltag; ++t) {
                int parenttag = t;
                while ((parenttag = G__struct.parent_tagnum[parenttag]) > scopetag)
                   {;}
                if (parenttag == scopetag) {
                   if('e'==G__struct.type[scopetag]) {
                      G__pragmalinkenum(scopetag,globalcomp);
                   } else {
                      // t is A::B::C, so set globalcomp
                      G__struct.globalcomp[t] = globalcomp;

                      G__struct.rootflag[t] = 0;
                      if (rfNoStreamer == 1) G__struct.rootflag[t] = G__NOSTREAMER;
                      if (rfNoInputOper == 1) G__struct.rootflag[t] |= G__NOINPUTOPERATOR;
                      if (rfUseBytecount == 1) {
                         G__struct.rootflag[t] |= G__USEBYTECOUNT;
                         if(rfNoStreamer) {
                            G__struct.rootflag[t] &= ~G__NOSTREAMER;
                            G__fprinterr(G__serr, "option + mutual exclusive with -, + prevails\n");
                         }
                      }
                      if( rfVersionNumber > -1 ) {
                         AllocateRootSpecial( t );
                         G__struct.rootflag[t] |= G__HASVERSION;
                         G__struct.rootspecial[t]->version = rfVersionNumber;
                      }
#if !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
                      if (G__NOLINK>G__nestedtypedef) {
                         G__linknestedtypedef(t, globalcomp);
                      }
#endif
                   } // enum or not
                   break;
                } // is within linked parent
             }
          } else {
             G__fprinterr(G__serr,"Error: unknown class %s in \"#pragma link C++ class %s::*\"! Egnoring it.\n", buf.data(), buf.data());
             c=G__fignorestream(";\n");
             return;
          }
       } else {
#if defined(G__REGEXP)
#ifndef G__OLDIKMPLEMENTATION1583
      if('.'!=buf[len-2]) {
        buf[len-1] = '.';
        buf[len++] = '*';
        buf[len] = 0;
      }
#endif
      regstat=regcomp(&re,buf,REG_EXTENDED|REG_NOSUB);
      if(regstat!=0) {
        G__genericerror("Error: regular expression error");
        return;
      }
      for(i=0;i<G__struct.alltag;i++) {
        if('$'==G__struct.name[i][0]) os=1;
        else                          os=0;
        if(0==regexec(&re,G__struct.name[i]+os,(size_t)0,(regmatch_t*)NULL,0)){
          G__struct.globalcomp[i] = globalcomp;
#ifndef G__OLDIKMPLEMENTATION1334
          G__struct.protectedaccess[i] = protectedaccess;
#endif
          ++done;
          if('e'==G__struct.type[i]) G__pragmalinkenum(i,globalcomp);
#if  !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
          else if (G__NOLINK>G__nestedtypedef)
            G__linknestedtypedef(i,globalcomp);
#endif
        }
      }
      regfree(&re);
#elif defined(G__REGEXP1)
      re = regcmp(buf, NULL);
      if (re==0) {
        G__genericerror("Error: regular expression error");
        return;
      }
      for(i=0;i<G__struct.alltag;i++) {
        if('$'==G__struct.name[i][0]) os=1;
        else                          os=0;
        if(0!=regex(re,G__struct.name[i]+os)){
          G__struct.globalcomp[i] = globalcomp;
#ifndef G__OLDIKMPLEMENTATION1334
          G__struct.protectedaccess[i] = protectedaccess;
#endif
          ++done;
          if('e'==G__struct.type[i]) G__pragmalinkenum(i,globalcomp);
#if  !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
          else if (G__NOLINK>G__nestedtypedef)
            G__linknestedtypedef(i,globalcomp);
#endif
        }
      }
      free(re);
#else /* G__REGEXP */
      *p='\0';
      hash=strlen(buf);
      for(i=0;i<G__struct.alltag;i++) {
        if('$'==G__struct.name[i][0]) os=1;
        else                          os=0;
        if(strncmp(buf,G__struct.name[i]+os,hash)==0
           || ('*'==buf[0]&&strstr(G__struct.name[i],buf+1))
           ) {
          G__struct.globalcomp[i] = globalcomp;
#ifndef G__OLDIKMPLEMENTATION1334
          G__struct.protectedaccess[i] = protectedaccess;
#endif
          ++done;
          /*G__fprinterr(G__serr,"#pragma link changed %s\n",G__struct.name[i]);*/
          if('e'==G__struct.type[i]) G__pragmalinkenum(i,globalcomp);
#if  !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
          else if (G__NOLINK>G__nestedtypedef)
            G__linknestedtypedef(i,globalcomp);
#endif
        }
      }
#endif /* G__REGEXP */
       } // if "A::*"
    } /* if(p) */
    else {
      i = G__defined_tagname(buf,1);
      if(i>=0) {
        G__struct.globalcomp[i] = globalcomp;
#ifndef G__OLDIKMPLEMENTATION1334
        G__struct.protectedaccess[i] = protectedaccess;
#endif
        ++done;
        if('e'==G__struct.type[i]) G__pragmalinkenum(i,globalcomp);
#if  !defined(G__OLDIMPLEMENTATION1955) && defined(G__ROOT)
        else if (G__NOLINK>G__nestedtypedef)
          G__linknestedtypedef(i,globalcomp);
#endif
        G__struct.rootflag[i] = 0;
        if (rfNoStreamer == 1) G__struct.rootflag[i] = G__NOSTREAMER;
        if (rfNoInputOper == 1) G__struct.rootflag[i] |= G__NOINPUTOPERATOR;
        if (rfUseBytecount == 1) {
          G__struct.rootflag[i] |= G__USEBYTECOUNT;
          if(rfNoStreamer) {
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
    if(!done && G__NOLINK!=globalcomp) {
#ifdef G__ROOT
      if(G__dispmsg>=G__DISPERR) {
         G__fprinterr(G__serr,"Error: link requested for unknown class %s",buf());
        G__genericerror((char*)NULL);
#else
      if(G__dispmsg>=G__DISPNOTE) {
         G__fprinterr(G__serr,"Note: link requested for unknown class %s",buf());
        G__printlinenum();
#endif
      }
    }
  }

  /*************************************************************************
  * #pragma link [spec] function [name];
  *************************************************************************/
  else if(strncmp(buf,"function",3)==0) {
    struct G__ifunc_table_internal *x_ifunc = &G__ifunc;
#ifndef G__OLDIMPLEMENTATION828
    fpos_t pos;
    int store_line = G__ifile.line_number;
    fgetpos(G__ifile.fp,&pos);
    c = G__fgetstream_template(buf, 0, ";\n\r<>");

    if(G__CPPLINK==globalcomp) globalcomp=G__METHODLINK;

    if(('<'==c || '>'==c)
       &&(strcmp(buf,"operator")==0||strstr(buf,"::operator"))) {
      int len=strlen(buf);
      buf[len++]=c;
      store_line = G__ifile.line_number;
      fgetpos(G__ifile.fp,&pos);
      buf[len] = G__fgetc();
      if(buf[len]==c||'='==buf[len]) c=G__fgetstream_template(buf, len+1, ";\n\r");
      else {
        fsetpos(G__ifile.fp,&pos);
        G__ifile.line_number = store_line;
        if(G__dispsource) G__disp_mask = 1;
        c = G__fgetstream_template(buf, len, ";\n\r");
      }
    }
    else {
      fsetpos(G__ifile.fp,&pos);
      G__ifile.line_number = store_line;
      c = G__fgetstream_template(buf, 0, ";\n\r");
    }


#else /* 828 */
    c = G__fgetstream_template(buf, 0, ";\n\r");
#endif /* 828 */


    /* if the function is specified with parameters */
    p = strchr(buf,'(');
    if(p && strstr(buf,"operator()")==0) {
      if(strncmp(p,")(",2)==0) p+=2;
      else if(strcmp(p,")")==0) p=0;
    }
    if(p) {
      G__FastAllocString funcname(G__LONGLINE);
      G__FastAllocString param_sb(G__LONGLINE);

      if(')' == *(p+1) && '('== *(p+2) ) p += 2;
      *p='\0';
      funcname = buf;
      param_sb = p+1;
      char* param = param_sb;
      p = strrchr(param,')');
      if (p==0) {
         return;
      }
      *p='\0';
      G__SetGlobalcomp(funcname,param,globalcomp);
      if(rfUseStubs) G__SetForceStub(funcname,param);
      ++done;
      return;
    }

    p = (char*)G__strrstr(buf,"::");
    if(p) {
      int ixx=0;
      if(-1==x_ifunc->tagnum) {
        int tagnum;
        *p = 0;
        tagnum = G__defined_tagname(buf,0);
        if(-1!=tagnum) {
          x_ifunc = G__struct.memfunc[tagnum];
        }
        *p = ':';
      }
      p+=2;
      while(*p) buf[ixx++] = *p++;
      buf[ixx] = 0;
    }

    /* search for wildcard character */
    p = strrchr(buf,'*');

    /* in case of operator*  */
    if(strncmp(buf,"operator",8)==0) p = (char*)NULL;

    if(p) {
#if defined(G__REGEXP)
#ifndef G__OLDIKMPLEMENTATION1583
      int len = strlen(buf);
      if('.'!=buf[len-2]) {
        buf[len-1] = '.';
        buf[len++] = '*';
        buf[len] = 0;
      }
#endif
      regstat=regcomp(&re,buf,REG_EXTENDED|REG_NOSUB);
      if(regstat!=0) {
        G__genericerror("Error: regular expression error");
        return;
      }
      ifunc = x_ifunc;
      while(ifunc) {
        for(i=0;i<ifunc->allifunc;i++) {
          if(0==regexec(&re,ifunc->funcname[i],(size_t)0,(regmatch_t*)NULL,0)
             && (ifunc->para_nu[i]<2 ||
                 -1==ifunc->param[i][1]->p_tagtable ||
                 strncmp(G__struct.name[ifunc->param[i][1]->p_tagtable]
                         ,"G__CINT_",8)!=0)
             ){
            ifunc->globalcomp[i] = globalcomp;
            if(rfUseStubs) ifunc->funcptr[i]=(void*)-2;
            ++done;
          }
        }
        ifunc = ifunc->next;
      }
      regfree(&re);
#elif defined(G__REGEXP1)
      re = regcmp(buf, NULL);
      if (re==0) {
        G__genericerror("Error: regular expression error");
        return;
      }
      ifunc = x_ifunc;
      while(ifunc) {
        for(i=0;i<ifunc->allifunc;i++) {
           if(0!=regex(re,ifunc->funcname[i])
             && (ifunc->para_nu[i]<2 ||
                 -1==ifunc->param[i][1]->p_tagtable ||
                 strncmp(G__struct.name[ifunc->param[i][1]->p_tagtable]
                         ,"G__CINT_",8)!=0)
             ){
            ifunc->globalcomp[i] = globalcomp;
            if(rfUseStubs) ifunc->funcptr[i]=(void*)-2;
            ++done;
          }
        }
        ifunc = ifunc->next;
      }
      free(re);
#else /* G__REGEXP */
      *p='\0';
      hash=strlen(buf);
      ifunc = x_ifunc;
      while(ifunc) {
        for(i=0;i<ifunc->allifunc;i++) {
           if((strncmp(buf,ifunc->funcname[i],hash)==0
              || ('*'==buf[0]&&strstr(ifunc->funcname[i],buf+1)))
             && (ifunc->para_nu[i]<2 ||
                 -1==ifunc->param[i][1]->p_tagtable ||
                 strncmp(G__struct.name[ifunc->param[i][1]->p_tagtable]
                         ,"G__CINT_",8)!=0)
             ) {
            ifunc->globalcomp[i] = globalcomp;
            if(rfUseStubs) ifunc->funcptr[i]=(void*)-2;
            ++done;
            /*G__fprinterr(G__serr,"#pragma link changed %s\n",ifunc->funcname[i]);*/
          }
        }
        ifunc = ifunc->next;
      }
#endif /* G__REGEXP */
    }
    else {
      ifunc = x_ifunc;
      while(ifunc) {
        for(i=0;i<ifunc->allifunc;i++) {
          if(strcmp(buf,ifunc->funcname[i])==0
             && (ifunc->para_nu[i]<2 ||
                 -1==ifunc->param[i][1]->p_tagtable ||
                 strncmp(G__struct.name[ifunc->param[i][1]->p_tagtable]
                         ,"G__CINT_",8)!=0)
             ) {
            ifunc->globalcomp[i] = globalcomp;
            if(rfUseStubs) ifunc->funcptr[i]=(void*)-2;
            ++done;
          }
        }
        ifunc = ifunc->next;
      }
    }
    if(!done && strchr(buf,'<')!=0) {
      struct G__param fpara;
      struct G__funclist *funclist=(struct G__funclist*)NULL;
      int tmp=0;

      fpara.paran=0;

      G__hash(buf,hash,tmp);
      funclist=G__add_templatefunc(buf,&fpara,hash,funclist,x_ifunc,0);
      if(funclist) {
        funclist->ifunc->globalcomp[funclist->ifn] = globalcomp;
        if(rfUseStubs) funclist->ifunc->funcptr[i]=(void*)-2;
        G__funclist_delete(funclist);
        ++done;
      }
    }
    if(!done && G__NOLINK!=globalcomp) {
#ifdef G__ROOT
      if(G__dispmsg>=G__DISPERR) {
         G__fprinterr(G__serr,"Error: link requested for unknown function %s",buf());
        G__genericerror((char*)NULL);
#else
      if(G__dispmsg>=G__DISPNOTE) {
         G__fprinterr(G__serr,"Note: link requested for unknown function %s",buf());
        G__printlinenum();
#endif
      }
    }
  }

  /*************************************************************************
  * #pragma link [spec] operators [classname];
  *************************************************************************/
  else if(strncmp(buf,"operators",3)==0) {
     c = G__fgetname_template(buf, 0, ";\n\r");

     // Do not (yet) support wildcarding
     int cltag = G__defined_tagname(buf,1);
     if (cltag<0) {
#ifdef G__ROOT
        if(G__dispmsg>=G__DISPERR) {
           G__fprinterr(G__serr,"Error: link requested for operators or unknown class %s",buf());
           G__genericerror((char*)NULL);
        }
#else
        if(G__dispmsg>=G__DISPNOTE) {
           G__fprinterr(G__serr,"Note: link requested for unknown class %s",buf());
           G__printlinenum();
        }
#endif
     } else if (G__struct.type[cltag] == 'n') {
        // Nothing to do for namespace (should we issue an error message?)
     } else {
        // Look for function in the declaring namespace of the class
        // that are name 'operator' and have an argument which is of the
        // requested type.
        short scope = cltag;
        do {
           scope = G__struct.parent_tagnum[scope];
           struct G__ifunc_table_internal *x_ifunc = &G__ifunc;
           if (scope != -1) {
              x_ifunc =  G__struct.memfunc[scope];
           }
           ifunc = x_ifunc;
           while(ifunc) {
              for(i=0;i<ifunc->allifunc;i++) {
                 bool opmatch = false;
                 if (strncmp( ifunc->funcname[i], "operator", 8)==0) {
                    for(short narg=0; narg<ifunc->para_nu[i]; ++narg) {
                       if ( ifunc->param[i][narg]->p_tagtable == cltag ) {
                          // note we do not test whether the argument is a reference, value or pointer
                          // nor its constness.
                          opmatch = true;
                       }
                    }
                    if (opmatch) {
                       ifunc->globalcomp[i] = globalcomp;
                    }
                 }
              }
              ifunc = ifunc->next;
           }
        } while (scope != -1);
     }
  }
  /*************************************************************************
  * #pragma link [spec] global [name];
  *************************************************************************/
  else if(strncmp(buf,"global",3)==0) {
    c = G__fgetname_template(buf, 0, ";\n\r");
    p = strrchr(buf,'*');
    if(p) {
#if defined(G__REGEXP)
#ifndef G__OLDIKMPLEMENTATION1583
      int len = strlen(buf);
      if('.'!=buf[len-2]) {
        buf[len-1] = '.';
        buf[len++] = '*';
        buf[len] = 0;
      }
#endif
      regstat=regcomp(&re,buf,REG_EXTENDED|REG_NOSUB);
      if(regstat!=0) {
        G__genericerror("Error: regular expression error");
        return;
      }
      var = &G__global;
      while(var) {
        for(i=0;i<var->allvar;i++) {
          if(0==regexec(&re,var->varnamebuf[i],(size_t)0,(regmatch_t*)NULL,0)){
            var->globalcomp[i] = globalcomp;
            ++done;
          }
        }
        var=var->next;
      }
      regfree(&re);
#elif defined(G__REGEXP1)
      re = regcmp(buf, NULL);
      if (re==0) {
        G__genericerror("Error: regular expression error");
        return;
      }
      var = &G__global;
      while(var) {
        for(i=0;i<var->allvar;i++) {
           if(0!=regex(re,var->varnamebuf[i])){
            var->globalcomp[i] = globalcomp;
            ++done;
          }
        }
        var=var->next;
      }
      free(re);
#else /* G__REGEXP */
      *p = '\0';
      hash = strlen(buf);
      var = &G__global;
      while(var) {
        for(i=0;i<var->allvar;i++) {
          if(strncmp(buf,var->varnamebuf[i],hash)==0
             || ('*'==buf[0]&&strstr(var->varnamebuf[i],buf+1))
             ) {
            var->globalcomp[i] = globalcomp;
            ++done;
            /*G__fprinterr(G__serr,"#pragma link changed %s\n",var->varnamebuf[i]);*/
          }
        }
        var=var->next;
      }
#endif /* G__REGEXP */
    }
    else {
      G__hash(buf,hash,i);
      var = G__getvarentry(buf,hash,&i,&G__global,(struct G__var_array*)NULL);
      if(var) {
        var->globalcomp[i] = globalcomp;
        ++done;
      }
    }
    if(!done && G__NOLINK!=globalcomp) {
      if(G__dispmsg>=G__DISPNOTE) {
         G__fprinterr(G__serr,"Note: link requested for unknown global variable %s",buf());
        G__printlinenum();
      }
    }
  }

  /*************************************************************************
  * #pragma link [spec] all_datamember [classname];
  *  This is not needed because G__METHODLINK and G__ONLYMETHODLINK are
  *  introduced. Keeping this just for future needs.
  *************************************************************************/
  else if(strncmp(buf,"all_datamembers",5)==0) {
    if(';'!=c) c = G__fgetstream_template(buf, 0, ";\n\r");
    if(buf[0]) {
      struct G__var_array *var;
      int ig15;
      if(strcmp(buf,"::")==0) {
        var = &G__global;
      }
      else {
        int tagnum = G__defined_tagname(buf,0);
        if(-1!=tagnum) {
          var= G__struct.memvar[tagnum];
        }
        else { /* must be an error */
          return;
        }
      }
      while(var) {
        for(ig15=0;ig15<var->allvar;ig15++) {
          var->globalcomp[ig15] = globalcomp;
          if(G__NOLINK==globalcomp) var->access[ig15] = G__PRIVATE;
          else                      var->access[ig15] = G__PUBLIC;
        }
        var=var->next;
      }
    }
  }

  /*************************************************************************
  * #pragma link [spec] all_function|all_method [classname];
  *  This is not needed because G__METHODLINK and G__ONLYMETHODLINK are
  *  introduced. Keeping this just for future needs.
  *************************************************************************/
  else if(strncmp(buf,"all_methods",5)==0||
          strncmp(buf,"all_functions",5)==0) {
    if(';'!=c) c = G__fgetstream_template(buf, 0, ";\n\r");
    if(G__CPPLINK==globalcomp) globalcomp=G__METHODLINK;
    if(buf[0]) {
      struct G__ifunc_table_internal *ifunc;
      int ifn;
      if(strcmp(buf,"::")==0) {
        ifunc = &G__ifunc;
      }
      else {
        int tagnum = G__defined_tagname(buf,0);
        if(-1!=tagnum) {
          ifunc = G__struct.memfunc[tagnum];
        }
        else { /* must be an error */
          return;
        }
      }
      while(ifunc) {
        for(ifn=0;ifn<ifunc->allifunc;ifn++) {
          ifunc->globalcomp[ifn] = globalcomp;
          if(G__NOLINK==globalcomp) ifunc->access[ifn] = G__PRIVATE;
          else                      ifunc->access[ifn] = G__PUBLIC;
        }
        ifunc=ifunc->next;
      }
    }
    else {
      G__suppress_methods = (globalcomp==G__NOLINK);
    }
  }

  /*************************************************************************
  * #pragma link [spec] methods;
  *************************************************************************/
  else if(strncmp(buf,"methods",3)==0) {
    G__suppress_methods = (globalcomp==G__NOLINK);
  }

  /*************************************************************************
  * #pragma link [spec] typedef [name];
  *************************************************************************/
  else if(strncmp(buf,"typedef",3)==0) {
    c = G__fgetname_template(buf, 0, ";\n\r");
    p = strrchr(buf,'*');
    if(p && *(p+1)=='>') p=(char*)NULL;
    if(p) {
#if defined(G__REGEXP)
#ifndef G__OLDIKMPLEMENTATION1583
      int len = strlen(buf);
      if('.'!=buf[len-2]) {
        buf[len-1] = '.';
        buf[len++] = '*';
        buf[len] = 0;
      }
#endif
      regstat=regcomp(&re,buf,REG_EXTENDED|REG_NOSUB);
      if(regstat!=0) {
        G__genericerror("Error: regular expression error");
        return;
      }
      for(i=0;i<G__newtype.alltype;i++) {
        if(0==regexec(&re,G__newtype.name[i],(size_t)0,(regmatch_t*)NULL,0)){
          G__newtype.globalcomp[i] = globalcomp;
          if(-1!=G__newtype.tagnum[i] &&
             '$'==G__struct.name[G__newtype.tagnum[i]][0]) {
            G__struct.globalcomp[G__newtype.tagnum[i]] = globalcomp;
          }
          ++done;
        }
      }
      regfree(&re);
#elif defined(G__REGEXP1)
      re = regcmp(buf, NULL);
      if (re==0) {
        G__genericerror("Error: regular expression error");
        return;
      }
      for(i=0;i<G__newtype.alltype;i++) {
        if(0!=regex(re,G__newtype.name[i])){
          G__newtype.globalcomp[i] = globalcomp;
          if(-1!=G__newtype.tagnum[i] &&
             '$'==G__struct.name[G__newtype.tagnum[i]][0]) {
            G__struct.globalcomp[G__newtype.tagnum[i]] = globalcomp;
          }
          ++done;
        }
      }
      free(re);
#else /* G__REGEXP */
      *p='\0';
      hash=strlen(buf);
      for(i=0;i<G__newtype.alltype;i++) {
        if(strncmp(buf,G__newtype.name[i],hash)==0
           || ('*'==buf[0]&&strstr(G__newtype.name[i],buf+1))
           ) {
          G__newtype.globalcomp[i] = globalcomp;
          if(-1!=G__newtype.tagnum[i] &&
             '$'==G__struct.name[G__newtype.tagnum[i]][0]) {
            G__struct.globalcomp[G__newtype.tagnum[i]] = globalcomp;
          }
          ++done;
        }
      }
#endif /* G__REGEXP */
    }
    else {
      i = G__defined_typename(buf);
      if(-1!=i) {
        G__newtype.globalcomp[i] = globalcomp;
        if(-1!=G__newtype.tagnum[i] &&
           '$'==G__struct.name[G__newtype.tagnum[i]][0]) {
          G__struct.globalcomp[G__newtype.tagnum[i]] = globalcomp;
        }
          ++done;
      }
    }
    if(!done && G__NOLINK!=globalcomp) {
#ifdef G__ROOT
      if(G__dispmsg>=G__DISPERR) {
         G__fprinterr(G__serr,"Error: link requested for unknown typedef %s",buf());
        G__genericerror((char*)NULL);
#else
      if(G__dispmsg>=G__DISPNOTE) {
         G__fprinterr(G__serr,"Note: link requested for unknown typedef %s",buf());
        G__printlinenum();
#endif
      }
    }
  }
#ifdef G__ROOT
  /*************************************************************************
  * #pragma link [spec] ioctortype [item];
  *************************************************************************/
  else if(strncmp(buf,"ioctortype",3)==0) {
      c = G__fgetname(buf, 0, ";\n\r");
      if (G__p_ioctortype_handler) G__p_ioctortype_handler(buf);
  }
#endif
  /*************************************************************************
  * #pragma link [spec] defined_in [item];
  *************************************************************************/
  else if(strncmp(buf,"defined_in",3)==0) {
    fpos_t pos;
    int tagflag = 0;
    int ifile=0;
    struct stat statBufItem;
    struct stat statBuf;
#ifdef G__WIN32
    G__FastAllocString fullItem(_MAX_PATH);
    G__FastAllocString fullIndex(_MAX_PATH);
#endif
    fgetpos(G__ifile.fp,&pos);
    c = G__fgetname(buf, 0, ";\n\r");
    if(strcmp(buf,"class")==0||strcmp(buf,"struct")==0||
        strcmp(buf,"namespace")==0) {
      if(isspace(c)) c = G__fgetstream_template(buf, 0, ";\n\r");
      tagflag = 1;
    }
    else {
      fsetpos(G__ifile.fp,&pos);
      c = G__fgetstream_template(buf, 0, ";\n\r<>");
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
        0==tagflag &&
        0 == G__statfilename( buf, & statBufItem ) ) {
#ifdef G__WIN32
      _fullpath( fullItem, buf, _MAX_PATH );
#endif
      for(ifile=0;ifile<G__nfile;ifile++) {
        if (0 == stat( G__srcfile[ifile].filename, & statBuf ) ) {
#ifndef G__WIN32
          if ( statBufItem.st_ino == statBuf.st_ino ) {
#else
          _fullpath( fullIndex, G__srcfile[ifile].filename, _MAX_PATH );
          /* Windows is case insensitive! */
          if (0==stricmp(fullItem,fullIndex)) {
#endif
            ++done;
            /* link class,struct */
            for(i=0;i<G__struct.alltag;i++) {
              if(G__struct.filenum[i]==ifile) {
                struct G__var_array *var = G__struct.memvar[i];
                int ifn;
                while(var) {
                  for(ifn=0;ifn<var->allvar;ifn++) {
                    if(var->filenum[ifn]==ifile
#define G__OLDIMPLEMENTATION1740
                       ) {
                      var->globalcomp[ifn] = globalcomp;
                    }
                  }
                  var = var->next;
                }
                ifunc=G__struct.memfunc[i];
                while(ifunc) {
                  for(ifn=0;ifn<ifunc->allifunc;ifn++) {
                    if(ifunc->pentry[ifn]&&ifunc->pentry[ifn]->filenum==ifile
                       ) {
                      ifunc->globalcomp[ifn] = globalcomp;
                    }
                  }
                  ifunc = ifunc->next;
                }
                G__struct.globalcomp[i]=globalcomp;
                /* Note this make the equivalent of '+' the
                   default for defined_in type of linking */
                if ( 0 == (G__struct.rootflag[i] & G__NOSTREAMER) ) {
                  G__struct.rootflag[i] |= G__USEBYTECOUNT;
                }
              }
            }
            /* link global function */
            ifunc = &G__ifunc;
            while(ifunc) {
              for(i=0;i<ifunc->allifunc;i++) {
                if(ifunc->pentry[i]&&ifunc->pentry[i]->filenum==ifile) {
                  ifunc->globalcomp[i] = globalcomp;
                }
              }
              ifunc = ifunc->next;
            }
#ifdef G__VARIABLEFPOS
            /* link global variable */
            {
              struct G__var_array *var = &G__global;
              while(var) {
                for(i=0;i<var->allvar;i++) {
                  if(var->filenum[i]==ifile) {
                    var->globalcomp[i] = globalcomp;
                  }
                }
                var = var->next;
              }
            }
#endif
#ifdef G__TYPEDEFFPOS
            /* link typedef */
            for(i=0;i<G__newtype.alltype;i++) {
              if(G__newtype.filenum[i]==ifile) {
                G__newtype.globalcomp[i] = globalcomp;
              }
            }
#endif
            break;
          }
        }
      }
    }
    /* #pragma link [C|C++|off] defined_in [class|struct|namespace] name; */
    if(!done) {
      int parent_tagnum = G__defined_tagname(buf,0);
      int j,flag;
      if(-1!=parent_tagnum) {
        for(i=0;i<G__struct.alltag;i++) {
          struct G__var_array *var = G__struct.memvar[parent_tagnum];
          int ifn;
          done = 1;
          while(var) {
            for(ifn=0;ifn<var->allvar;ifn++) {
              if(var->filenum[ifn]==ifile
                 ) {
                var->globalcomp[ifn] = globalcomp;
              }
            }
            var = var->next;
          }
          ifunc=G__struct.memfunc[i];
          while(ifunc) {
            for(ifn=0;ifn<ifunc->allifunc;ifn++) {
              if(ifunc->pentry[ifn]&&ifunc->pentry[ifn]->filenum==ifile
                 ) {
                ifunc->globalcomp[ifn] = globalcomp;
              }
            }
            ifunc = ifunc->next;
          }
          flag = 0;
          j = i;
          G__struct.globalcomp[parent_tagnum]=globalcomp;
          while(-1!=G__struct.parent_tagnum[j]) {
            if(G__struct.parent_tagnum[j]==parent_tagnum) flag=1;
            j = G__struct.parent_tagnum[j];
          }
          if(flag) {
            var = G__struct.memvar[i];
            while(var) {
              for(ifn=0;ifn<var->allvar;ifn++) {
                if(var->filenum[ifn]==ifile
                   &&G__PUBLIC==var->access[ifn]
                   ) {
                  var->globalcomp[ifn] = globalcomp;
                }
              }
              var = var->next;
            }
            ifunc=G__struct.memfunc[i];
            while(ifunc) {
              for(ifn=0;ifn<ifunc->allifunc;ifn++) {
                if(ifunc->pentry[ifn]&&ifunc->pentry[ifn]->filenum==ifile
                   &&G__PUBLIC==ifunc->access[ifn]
                   ) {
                  ifunc->globalcomp[ifn] = globalcomp;
                }
              }
              ifunc = ifunc->next;
            }
            G__struct.globalcomp[i]=globalcomp;
            /* Note this make the equivalent of '+' the
               default for defined_in type of linking */
            if ( (G__struct.rootflag[i] & G__NOSTREAMER) == 0 ) {
              G__struct.rootflag[i] |= G__USEBYTECOUNT;
            }
          }
        }
        for(i=0;i<G__newtype.alltype;i++) {
          flag = 0;
          j = G__newtype.parent_tagnum[i];
          do {
            if(j == parent_tagnum) flag = 1;
            j = G__struct.parent_tagnum[j];
          } while(-1 != j);
          if(flag) {
            G__struct.globalcomp[i]=globalcomp;
            /* Note this make the equivalent of '+' the
               default for defined_in type of linking */
            if ( 0 == (G__struct.rootflag[i] & G__NOSTREAMER) ) {
              G__struct.rootflag[i] |= G__USEBYTECOUNT;
            }
          }
        }
      }
    }
    if(!done && G__NOLINK!=globalcomp) {
       G__fprinterr(G__serr,"Warning: link requested for unknown srcfile %s",buf());
      G__printlinenum();
    }
  }

  /*************************************************************************
  * #pragma link [spec] all [item];
  *************************************************************************/
  else if(strncmp(buf,"all",2)==0) {
    c = G__fgetname_template(buf, 0, ";\n\r");
    if(strncmp(buf,"class",3)==0) {
      for(i=0;i<G__struct.alltag;i++) {
        if(G__NOLINK==globalcomp ||
           (G__NOLINK==G__struct.iscpplink[i] &&
            (-1!=G__struct.filenum[i] &&
             0==(G__srcfile[G__struct.filenum[i]].hdrprop&G__CINTHDR))))
          G__struct.globalcomp[i] = globalcomp;
      }
    }
    else if(strncmp(buf,"function",3)==0) {
      ifunc = &G__ifunc;
      while(ifunc) {
        for(i=0;i<ifunc->allifunc;i++) {
        if(G__NOLINK==globalcomp ||
           (
            0<=ifunc->pentry[i]->size && 0<=ifunc->pentry[i]->filenum &&
            0==(G__srcfile[ifunc->pentry[i]->filenum].hdrprop&G__CINTHDR)))
          ifunc->globalcomp[i] = globalcomp;
        }
        ifunc = ifunc->next;
      }
    }
    else if(strncmp(buf,"global",3)==0) {
      var = &G__global;
      while(var) {
        for(i=0;i<var->allvar;i++) {
          if(G__NOLINK==globalcomp ||
             (0<=var->filenum[i] &&
              0==(G__srcfile[var->filenum[i]].hdrprop&G__CINTHDR)))
            var->globalcomp[i] = globalcomp;
        }
        var = var->next;
      }
    }
    else if(strncmp(buf,"typedef",3)==0) {
      for(i=0;i<G__newtype.alltype;i++) {
        if(G__NOLINK==globalcomp ||
           (G__NOLINK==G__newtype.iscpplink[i] &&
            0<=G__newtype.filenum[i] &&
            0==(G__srcfile[G__newtype.filenum[i]].hdrprop&G__CINTHDR))) {
          G__newtype.globalcomp[i] = globalcomp;
          if((-1!=G__newtype.tagnum[i] &&
              '$'==G__struct.name[G__newtype.tagnum[i]][0])) {
            G__struct.globalcomp[G__newtype.tagnum[i]] = globalcomp;
          }
        }
      }
    }
    /*************************************************************************
     * #pragma link [spec] all methods;
     *************************************************************************/
    else if(strncmp(buf,"methods",3)==0) {
      G__suppress_methods = (globalcomp==G__NOLINK);
  }
  }

  if(';'!=c) c=G__fignorestream("#;");
  if(';'!=c) G__genericerror("Syntax error: #pragma link");
}

#endif /* G__SMALLOBJECT */

/**************************************************************************
* G__incsetup_memvar()
*
**************************************************************************/
void G__incsetup_memvar(int tagnum)
{
  int store_asm_exec;
  char store_var_type;
  int store_static_alloc = G__static_alloc;
  int store_constvar = G__constvar;

  if (G__struct.incsetup_memvar[tagnum]==0) return;

  if(!G__struct.incsetup_memvar[tagnum]->empty()) {
    store_asm_exec = G__asm_exec;
    G__asm_exec=0;
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
      G__strlcpy(G__ifile.name,G__srcfile[fileno].filename,G__MAXFILENAME);
    }

#ifdef G__OLDIMPLEMENTATION1125_YET
    if(0==G__struct.memvar[tagnum]->allvar
       || 'n'==G__struct.type[tagnum]){

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
    if(G__var_type!=store_var_type)
      G__fprinterr(G__serr,"Cint internal error: G__incsetup_memvar %c %c\n"
                   ,G__var_type,store_var_type);
#endif
    G__var_type = store_var_type;
    G__asm_exec = store_asm_exec;
    G__constvar = store_constvar;
    G__ifile = store_ifile;
    G__static_alloc = store_static_alloc;
  }
}

/**************************************************************************
* G__incsetup_memfunc()
*
**************************************************************************/
void G__incsetup_memfunc(int tagnum)
{
  char store_var_type;
  int store_asm_exec;

  // DIEGO
  if (G__struct.incsetup_memfunc[tagnum]==0)
    G__struct.incsetup_memfunc[tagnum] = new std::list<G__incsetup>();

  if(!G__struct.incsetup_memfunc[tagnum]->empty()) {
    store_asm_exec = G__asm_exec;
    G__asm_exec=0;
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
       G__strlcpy(G__ifile.name,G__srcfile[fileno].filename,G__MAXFILENAME);
    }

#ifdef G__OLDIMPLEMENTATION1125_YET /* G__PHILIPPE26 */
    if(0==G__struct.memfunc[tagnum]->allifunc
       || 'n'==G__struct.type[tagnum]
       || (
           -1!=G__struct.memfunc[tagnum]->pentry[0]->size
           && 2>=G__struct.memfunc[tagnum]->allifunc)){
       // G__setup_memfuncXXX execution
       std::list<G__incsetup>::iterator iter;
       if(!G__struct.incsetup_memfunc[tagnum]->empty())
          for (iter=G__struct.incsetup_memfunc[tagnum]->begin(); iter != G__struct.incsetup_memfunc[tagnum]->end(); ++iter)
             (*iter)();
   }
#else
    // G__setup_memfuncXXX execution
    std::list<G__incsetup> *store_memfunc = G__struct.incsetup_memfunc[tagnum];
    G__struct.incsetup_memfunc[tagnum] = 0;

    std::list<G__incsetup>::iterator iter;
    if(!store_memfunc->empty()){
      for (iter=store_memfunc->begin(); iter != store_memfunc->end(); ++iter) {
        if (*iter) (*iter)();
      }
    }

    if (G__struct.incsetup_memfunc[tagnum]){
      G__struct.incsetup_memfunc[tagnum]->clear();
      delete G__struct.incsetup_memfunc[tagnum];
      G__struct.incsetup_memfunc[tagnum] = 0;
    }
    G__struct.incsetup_memfunc[tagnum] = store_memfunc;
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



/**************************************************************************
* G__getnumbaseclass()
*
**************************************************************************/
int G__getnumbaseclass(int tagnum)
{
   if (tagnum >= 0) {
      return(G__struct.baseclass[tagnum]->basen);
   } else {
      return 0;
   }
}

/**************************************************************************
* G__setnewtype_settypeum()
*
**************************************************************************/
static int G__setnewtype_typenum = -1;
void G__setnewtype_settypeum(int typenum)
{
   G__setnewtype_typenum=typenum;
}

/**************************************************************************
* G__setnewtype()
*
**************************************************************************/
void G__setnewtype(int globalcomp,const char *comment,int nindex)
{
  int typenum =
    (-1!=G__setnewtype_typenum)? G__setnewtype_typenum:G__newtype.alltype-1;
  G__newtype.iscpplink[typenum] = globalcomp;
  G__newtype.comment[typenum].p.com = (char*)comment;
  if(comment) G__newtype.comment[typenum].filenum = -2;
  else        G__newtype.comment[typenum].filenum = -1;
  G__newtype.nindex[typenum] = nindex;
  if(nindex)
    G__newtype.index[typenum]=(int*)malloc(G__INTALLOC*nindex);
}

/**************************************************************************
* G__setnewtypeindex()
*
**************************************************************************/
void G__setnewtypeindex(int j,int index)
{
  int typenum =
    (-1!=G__setnewtype_typenum)? G__setnewtype_typenum:G__newtype.alltype-1;
  G__newtype.index[typenum][j] = index;
}

/**************************************************************************
* G__getgvp()
*
**************************************************************************/
long G__getgvp()
{
  return(G__globalvarpointer);
}

/**************************************************************************
* G__setgvp()
*
**************************************************************************/
void G__setgvp(long gvp)
{
  G__globalvarpointer=gvp;
}

/**************************************************************************
* G__resetplocal()
*
**************************************************************************/
void G__resetplocal()
{
  if(G__def_struct_member && 'n'==G__struct.type[G__tagdefining]) {
     G__IncSetupStack::push();

    G__tagnum = G__tagdefining;
    G__p_local=G__struct.memvar[G__tagnum];
    while(G__p_local->next) G__p_local = G__p_local->next;
    G__def_struct_member = 1;
    G__tagdefining=G__tagnum;
    /* G__static_alloc = 1; */
  }
  else {
    G__p_local = (struct G__var_array*)NULL;
    int store_def_struct_member = G__def_struct_member;
    G__def_struct_member = 0;
    G__IncSetupStack::push();
    G__def_struct_member = store_def_struct_member;
  }
}

/**************************************************************************
* G__resetglobalenv()
*
**************************************************************************/
void G__resetglobalenv()
{
  /* Variables stack restoring */
  std::stack<G__IncSetupStack> *var_stack = G__stack_instance();
  G__IncSetupStack *incsetup_stack = &var_stack->top();

  if(incsetup_stack->G__incset_def_struct_member && 'n'==G__struct.type[incsetup_stack->G__incset_tagdefining]){
     G__IncSetupStack::pop();
  }
  else {
    G__globalvarpointer = G__PVOID;
    G__var_type = 'p';
    G__tagnum = -1;
    G__typenum = -1;
    G__static_alloc = 0;
    G__access = G__PUBLIC;
    var_stack->pop();
  }
}

/**************************************************************************
* G__lastifuncposition()
*
**************************************************************************/
void G__lastifuncposition()
{

/* Variables stack storing */
  if(G__def_struct_member && 'n'==G__struct.type[G__tagdefining]) {
     G__IncSetupStack::push();
     G__tagnum = G__tagdefining;
     G__p_ifunc = G__struct.memfunc[G__tagnum];
     while(G__p_ifunc->next) G__p_ifunc=G__p_ifunc->next;
  }
  else {
     G__p_ifunc = &G__ifunc;
     while(G__p_ifunc->next) G__p_ifunc=G__p_ifunc->next;
     int store_def_struct_member = G__def_struct_member;
     G__def_struct_member = 0;
     G__IncSetupStack::push();
     G__def_struct_member = store_def_struct_member;
  }
}

/**************************************************************************
* G__resetifuncposition()
*
**************************************************************************/
void G__resetifuncposition()
{

 /* Variables stack restoring */
  std::stack<G__IncSetupStack>* var_stack = G__stack_instance();
  G__IncSetupStack *incsetup_stack = &var_stack->top();

  if(incsetup_stack->G__incset_def_struct_member && 'n'==G__struct.type[incsetup_stack->G__incset_tagdefining]){
     incsetup_stack->restore();
  }
  else {
    G__tagnum = -1;
    G__p_ifunc = &G__ifunc;
    G__func_now = -1;
    G__func_page = 0;
    G__var_type = 'p';
  }
  G__globalvarpointer = G__PVOID;
  G__static_alloc = 0;
  G__access = G__PUBLIC;
  G__typenum = -1;

  var_stack->pop();
}

/**************************************************************************
* G__setnull()
*
**************************************************************************/
void G__setnull(G__value *result)
{
  *result = G__null;
}

/**************************************************************************
* G__getstructoffset()
*
**************************************************************************/
long G__getstructoffset()
{
  return(G__store_struct_offset);
}

/**************************************************************************
* G__getaryconstruct()
*
**************************************************************************/
int G__getaryconstruct()
{
  return(G__cpp_aryconstruct);
}

/**************************************************************************
* G__gettempbufpointer()
*
**************************************************************************/
long G__gettempbufpointer()
{
  return(G__p_tempbuf->obj.obj.i);
}

/**************************************************************************
* G__setsizep2memfunc()
*
**************************************************************************/
void G__setsizep2memfunc(int sizep2memfunc)
{
  G__sizep2memfunc = sizep2memfunc;
}

/**************************************************************************
* G__getsizep2memfunc()
*
**************************************************************************/
int G__getsizep2memfunc()
{
   G__asm_noverflow = G__store_asm_noverflow;
   G__no_exec_compile = G__store_no_exec_compile;
   G__asm_exec = G__store_asm_exec;
  return(G__sizep2memfunc);
}


/**************************************************************************
* G__setInitFunc()
*
**************************************************************************/
void G__setInitFunc(char *initfunc)
{
  G__INITFUNC=initfunc;
}

#ifdef G__WILDCARD
/**************************************************************************
* Access functions for WildCard interpreter
**************************************************************************/

/**************************************************************************
* G__getIfileFp()
**************************************************************************/
FILE *G__getIfileFp()
{
   return(G__ifile.fp);
}

/**************************************************************************
* G__incIfileLineNumber()
**************************************************************************/
void G__incIfileLineNumber()
{
  ++G__ifile.line_number;
}

/**************************************************************************
* G__getIfileLineNumber()
**************************************************************************/
int G__getIfileLineNumber()
{
  return(G__ifile.line_number);
}

/**************************************************************************
* G__setReturn()
**************************************************************************/
void G__setReturn(int rtn)
{
  G__return=rtn;
}

/**************************************************************************
* G__getFuncNow()
**************************************************************************/
int G__getFuncNow()
{
  return(G__func_now);
}

/**************************************************************************
* G__getPrerun
**************************************************************************/
int G__getPrerun()
{
  return(G__prerun);
}

/**************************************************************************
* G__setPrerun()
**************************************************************************/
void G__setPrerun(int prerun)
{
  G__prerun=prerun;
}

/**************************************************************************
* G__getDispsource()
**************************************************************************/
short G__getDispsource()
{
  return(G__dispsource);
}

/**************************************************************************
* G__getSerr()
**************************************************************************/
FILE* G__getSerr()
{
  return(G__serr);
}

/**************************************************************************
* G__getIsMain()
**************************************************************************/
int G__getIsMain()
{
  return(G__ismain);
}

/**************************************************************************
* G__setIsMain()
**************************************************************************/
void G__setIsMain(int ismain)
{
  G__ismain=ismain;
}

/**************************************************************************
* G__setStep()
**************************************************************************/
void G__setStep(int step)
{
  G__step=step;
}

/**************************************************************************
* G__getStepTrace()
**************************************************************************/
int G__getStepTrace()
{
  return(G__steptrace);
}

/**************************************************************************
* G__setDebug
**************************************************************************/
void G__setDebug(int dbg)
{
  G__debug=dbg;
}

/**************************************************************************
* G__getDebugTrace
**************************************************************************/
int G__getDebugTrace()
{
  return(G__debugtrace);
}

/**************************************************************************
* G__set_asm_noverflow
**************************************************************************/
void G__set_asm_noverflow(int novfl)
{
  G__asm_noverflow=novfl;
}

/**************************************************************************
* G__get_no_exec()
**************************************************************************/
int G__get_no_exec()
{
  return(G__no_exec);
}

/**************************************************************************
* int G__get_no_exec_compile
**************************************************************************/
int G__get_no_exec_compile()
{
  return(G__no_exec_compile);
}
#endif /* G__WILDCARD */

/* #ifndef G__OLDIMPLEMENTATION1167 */

} // extern "C"
template <typename T>
inline T* G__refT(G__value* buf)
{
   if (buf->type == G__gettypechar<T>() && buf->ref)
      return (T*)buf->ref;
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


/**************************************************************************
* G__specify_extra_include()
* has to be called from the pragma decoding!
**************************************************************************/
void G__specify_extra_include() {
   int i;
   int c;
   G__FastAllocString buf(G__ONELINE);
   char *tobecopied;
   if (!G__extra_include) {
      G__extra_include = (char**)malloc(G__MAXFILE*sizeof(char*));
      for(i=0;i<G__MAXFILE;i++)
         G__extra_include[i]=(char*)malloc(G__MAXFILENAME*sizeof(char));
   };
   c = G__fgetstream_template(buf, 0, ";\n\r<>");
   if ( 1 ) { /* should we check if the file exist ? */
      tobecopied = buf;
      if (buf[0]=='\"' || buf[0]=='\'') tobecopied++;
      i = strlen(buf);
      if (buf[i-1]=='\"' || buf[i-1]=='\'') buf[i-1]='\0';
      G__strlcpy(G__extra_include[G__extra_inc_n++],tobecopied,G__MAXFILENAME);
   }
}
   
/**************************************************************************
 * G__gen_extra_include()
 * prepend the extra header files to the C or CXX file
 **************************************************************************/
void G__gen_extra_include() {
  char * tempfile;
  FILE *fp,*ofp;
  G__FastAllocString line_sb(BUFSIZ);
  char* line = line_sb;
  int i;

  if (G__extra_inc_n) {
#ifndef G__ADD_EXTRA_INCLUDE_AT_END
    /* because of a bug in (at least) the KAI compiler we have to
       add the files at the beginning of the dictionary header file
       (Specifically, the extra include files have to be include
       before any forward declarations!) */

    if (!G__CPPLINK_H) return;

    tempfile = (char*) malloc(strlen(G__CPPLINK_H)+6);
    sprintf(tempfile,"%s.temp", G__CPPLINK_H);
    rename(G__CPPLINK_H,tempfile);

    fp = fopen(G__CPPLINK_H,"w");
    if(!fp) G__fileerror(G__CPPLINK_H);
    ofp = fopen(tempfile,"r");
    if(!ofp) G__fileerror(tempfile);

    /* Add the extra include ad the beginning of the files */
    fprintf(fp,"\n/* Includes added by #pragma extra_include */\n");
    for(i=0; i< G__extra_inc_n; i++) {
      fprintf(fp,"#include \"%s\"\n",G__extra_include[i]);
    }

    /* Copy rest of the header file */
    while (fgets(line, BUFSIZ, ofp)) {
      fprintf(fp, "%s", line);
    }
    fprintf(fp,"\n");

    fclose(fp);
    fclose(ofp);
    unlink(tempfile);
    free(tempfile);

#else
    fp = fopen(G__CPPLINK_H,"a");
    if(!fp) G__fileerror(G__CPPLINK_H);

    fprintf(fp,"\n/* Includes added by #pragma extra_include */\n");
    for(i=0; i< G__extra_inc_n; i++) {
      fprintf(fp,"#include \"%s\"\n",G__extra_include[i]);
    }
    fprintf(fp,"\n");
    fclose(fp);
#endif

  }
}


} /* extern "C" */


/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
