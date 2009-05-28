/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file global2.c
 ************************************************************************
 * Description:
 *  Cint parser global variables.
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

/**************************************************************************
* $xxx user specific scope object
**************************************************************************/
#ifdef G__ANSI
G__value (*G__GetSpecialObject)(char *name,void** pptr,void** ppdict);
#else
G__value (*G__GetSpecialObject)();
#endif

#ifdef G__SECURITY
/**************************************************************************
* Secure C++ mode
**************************************************************************/
G__UINT32 G__security;
namespace Cint {
   namespace Internal {
      int G__castcheckoff;
      int G__security_error;
      int G__max_stack_depth;
      char G__commandline[G__LONGLINE];
   }
}
#endif

/********************************************************************
* include path list
********************************************************************/
G__includepath Cint::Internal::G__ipathentry;
extern "C" struct G__includepath *G__getipathentry() { return &Cint::Internal::G__ipathentry; }

namespace Cint {
   namespace Internal {

/**************************************************************************
*  flags
**************************************************************************/
/* int G__error_flag=0; */ /* syntax error flag */
int G__debug;            /* trace input file */
int G__breakdisp;        /* display input file at break point */
/* int G__cont = -1;  */        /* continue G__cont lines */
int G__break;            /* break flab */
int G__step;             /* step execution flag */
int G__charstep;         /* char step flag */
int G__break_exit_func;  /* break at function exit */
/* int G__no_exec_stack ; */ /* stack for G__no_exec */
int G__eof;                /* end of file flag */
int G__no_exec;          /* no execution(ignore) flag */
int G__no_exec_compile;
char G__var_type;      /* variable decralation type */
char G__var_typeB;      /* variable decralation type for function return*/
int G__prerun;           /* pre-run flag */
int G__funcheader;       /* function header mode */
int G__return;           /* 0:normal,1:return,2:exit()or'q',3:'qq'(not used)*/
/* int G__extern_global;    number of globals defined in c function */
int G__disp_mask;        /* temporary read count */
int G__temp_read;        /* temporary read count */
int G__switch;           /* in a switch, parser should evaluate case expressions */
int G__switch_searching; /* in a switch, parser should return after evaluating a case expression */
int G__eof_count;        /* end of file error flag */
int G__ismain;           /* is there a main function */
int G__globalcomp;       /* make compiled func's global table */
int G__store_globalcomp;
char *G__globalvarpointer = G__PVOID; /* make compiled func's global table */

struct G__filetable G__srcfile[G__MAXFILE];
int G__nfile;

int G__nobreak;
char G__breakline[G__MAXNAME];
char G__breakfile[G__MAXFILENAME];
#define G__TESTBREAK     0x30
#define G__BREAK         0x10
#define G__NOBREAK       0xef
#define G__CONTUNTIL     0x20
#define G__NOCONTUNTIL   0xdf
#define G__TRACED        0x01
#define G__NOTRACED       0xfe

int G__key;              /* user function key on/off */
FILE *G__dumpreadline[6];
short G__Xdumpreadline[6];

#ifdef G__TMPFILE
char G__xfile[G__MAXFILENAME];
char G__tempc[G__MAXFILENAME];
#else
char G__xfile[L_tmpnam+10];
char G__tempc[L_tmpnam+10];
#endif

int G__doingconstruction;

#ifdef G__DUMPFILE
FILE *G__dumpfile;
short G__dumpspace;
#endif

int G__def_struct_member;

#ifdef G__FRIEND
::Reflex::Scope G__friendtagnum;
#endif
::Reflex::Scope G__tmplt_def_tagnum;
::Reflex::Scope G__def_tagnum;
::Reflex::Scope G__tagdefining;
::Reflex::Scope G__tagnum ; /* -1 */
::Reflex::Type G__typenum ;
short G__iscpp ;
short G__cpplock ;
short G__clock ;
short G__constvar ;
short G__isexplicit ;
short G__unsigned ;
short G__ansiheader ;
G__value G__ansipara;
short G__enumdef;

char G__tagname[G__MAXNAME];
char *G__store_struct_offset=0;
FILE *G__header,*G__temp1,*G__temp3,*G__temp5,*G__temp7,*G__temp8;
FILE *G__header2;
int G__decl;

#ifndef G__OLDIMPLEMENTATION1259
G__SIGNEDCHAR_T G__isconst;
#endif


char G__nam[G__MAXFILENAME];
char G__assertion[G__ONELINE];
short G__longjump;
short G__coredump;
short G__definemacro;

short G__noerr_defined;

short G__static_alloc;
Reflex::Member G__func_now;
//int G__func_page;

short G__twice;
short G__othermain;
int G__cpp;
int G__include_cpp;
char G__ccom[G__MAXFILENAME];
char G__cppsrcpost[G__LENPOST];
char G__csrcpost[G__LENPOST];
char G__cpphdrpost[G__LENPOST];
char G__chdrpost[G__LENPOST];

short G__dispsource;
short G__breaksignal;

short G__bitfield;

char *G__atexit ;

int G__cpp_aryconstruct;
int G__cppconstruct;

int G__access;

int G__steptrace,G__debugtrace;

int G__in_pause=0;
int G__stepover;


/* This must be set everytime before G__interpret_func() is called */
int G__fixedscope;

int G__isfuncreturnp2f;

int G__virtual;

/* #define G__OLDIMPLEMENTATION78 */

int G__oprovld;

int G__p2arylabel[G__MAXVARDIM];


/**************************************************************************
* If G__exec_memberfunc==1, G__getfunction will search memberfunction
* in addition to global functions.
**************************************************************************/
int G__exec_memberfunc;
::Reflex::Scope G__memberfunc_tagnum;
char *G__memberfunc_struct_offset;



/**************************************************************************
* buffer to store default parameter value
**************************************************************************/
G__value G__default_parameter;
char G__def_parameter[G__MAXNAME];


/**************************************************************************
* temporary object buffer
**************************************************************************/
struct G__tempobject_list *G__p_tempbuf,G__tempbuf;
int G__templevel;

/**************************************************************************
* reference type buffer
**************************************************************************/
int G__reftype;
char *G__refansipara;


/*********************************************************************
* #pragma compile feature utilizing DLL
*********************************************************************/
#ifdef G__AUTOCOMPILE
FILE *G__fpautocc;
char G__autocc_c[G__MAXNAME];
char G__autocc_h[G__MAXNAME];
char G__autocc_sl[G__MAXNAME];
char G__autocc_mak[G__MAXNAME];
int G__autoccfilenum = -1;
int G__compilemode;
#endif


/**************************************************************************
* Interactive debugging mode support and error recover
**************************************************************************/
int G__interactive;
G__value G__interactivereturnvalue;

/**************************************************************************
* Having K&R function in new linking
**************************************************************************/
int G__nonansi_func;


/**************************************************************************
* pointer to member function
**************************************************************************/
int G__sizep2memfunc=0;

/**************************************************************************
* flag to parse extern variables '-e' command line option
**************************************************************************/
int G__parseextern;

/**************************************************************************
* class specific debugging
**************************************************************************/
int G__istrace;

/**************************************************************************
* break, continue compilation
**************************************************************************/
struct G__breakcontinue_list *G__pbreakcontinue;

/*********************************************************************
* const string list
*********************************************************************/
struct G__ConstStringList G__conststringlist;
struct G__ConstStringList *G__plastconststring;

FILE *G__stderr;
FILE *G__stdout;
FILE *G__stdin;

FILE *G__serr;
FILE *G__sout;
FILE *G__sin;

FILE *G__intp_serr;
FILE *G__intp_sout;
FILE *G__intp_sin;

FILE *G__fpundeftype;

/**************************************************************************
* Class/struct comment title enhancement
**************************************************************************/
int G__fons_comment;
char *G__setcomment;

int G__precomp_private;

/**************************************************************************
* Preprocessed file keystring list
**************************************************************************/
struct G__Preprocessfilekey G__preprocessfilekey;

/**************************************************************************
* Flag to check global operator new/delete()
**************************************************************************/
int G__is_operator_newdelete ;

/**************************************************************************
* Path separator
**************************************************************************/
#if defined(G__NONANSI)
char *G__psep = "/";
#elif defined(G__CYGWIN)
const char *G__psep = "/";
#elif defined(G__WIN32)
const char *G__psep = "\\";
#elif defined(__MWERKS__)
const char *G__psep = ":";
#else
const char *G__psep = "/";
#endif

/**************************************************************************
* add user defined pragma statement
**************************************************************************/
struct G__AppPragma *G__paddpragma;

#ifdef G__MULTIBYTE
/**************************************************************************
* multi-byte coding system selection
**************************************************************************/
short G__lang = G__UNKNOWNCODING;
#endif

/**************************************************************************
* A flag no notify cint ready status to embedding program
* This flag is set by cint internal only, and read by G__getcintready() API
**************************************************************************/
int G__cintready=0;

/**************************************************************************
* interactive return for undefined symbol and G__pause
**************************************************************************/
int G__interactive_undefined=0;

/**************************************************************************
* STL Allocator workaround data
**************************************************************************/
char G__Allocator[G__ONELINE] = "Allocator";

/**************************************************************************
* Exception object buffer
**************************************************************************/
G__value G__exceptionbuffer;

/**************************************************************************
* Exception object buffer
**************************************************************************/
int G__ispragmainclude=0;

/**************************************************************************
* automatic variable on/off
**************************************************************************/
int G__automaticvar=1;

/**************************************************************************
* Local variable , bytecode compiler workaround
**************************************************************************/
int G__xrefflag=0;

/**************************************************************************
* Local variable , bytecode compiler workaround
**************************************************************************/
#ifdef G__ROOT
int G__do_smart_unload=1;
#else
int G__do_smart_unload=1;
#endif

#ifdef G__ROOT
int G__autoload_stdheader = 0;
#else
int G__autoload_stdheader = 1;
#endif

int G__ignore_stdnamespace = 1;

int G__decl_obj=0;

struct G__ConstStringList* G__SystemIncludeDir=0;

int G__command_eval=0 ;

#ifdef G__MULTITHREADLIBCINT
int G__multithreadlibcint = 1;
#else
int G__multithreadlibcint = 0;
#endif

void (*G__emergencycallback)();

int G__asm_clear_mask = 0;

int G__boolflag;

int G__init=0;

int G__last_error = 0;

int G__dispmsg = G__DISPALL;

int G__default_link = 1;

/* 1713 */
int G__gettingspecial = 0;

int G__gcomplevellimit=1000;

int G__catchexception=1;

int G__eval_localstatic=0;

int G__loadingDLL=0;


int G__mask_error=0;

G__eolcallback_t G__eolcallback;

int G__throwingexception=0;

int G__do_setmemfuncenv = 0;

int G__scopelevel=0;
int G__cintv6 = 0;

struct G__input_file G__lasterrorpos;

/**************************************************************************
* Incremented every time the cint dictionary is rewound in scrupto.
* Can be used to see if cached information derived from the dictionary
* is still valid.
**************************************************************************/
int G__scratch_count = 0;

int G__in_memvar_setup;

} // namespace Internal
} // namespace Cint

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
