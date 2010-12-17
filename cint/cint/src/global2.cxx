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

extern "C" {

/**************************************************************************
 *  flags
 **************************************************************************/
int G__debug;            /* trace input file */
int G__breakdisp;        /* display input file at break point */
int G__break;            /* break flab */
int G__step;             /* step execution flag */
int G__charstep;         /* char step flag */
int G__break_exit_func;  /* break at function exit */
int G__eof;              /* end of file flag */
int G__no_exec;          /* no execution(ignore) flag */
int G__no_exec_compile;
char G__var_type;        /* variable decralation type */
char G__var_typeB;       /* variable decralation type for function return*/
int G__prerun;           /* pre-run flag */
int G__funcheader;       /* function header mode */
int G__return;           /* 0:normal,1:return,2:exit()or'q',3:'qq'(not used)*/
int G__disp_mask;        /* temporary read count */
int G__temp_read;        /* temporary read count */
int G__switch;           /* in a switch, parser should evaluate case expressions */
int G__switch_searching; /* in a switch, parser should return after evaluating a case expression */
int G__eof_count;        /* end of file error flag */
int G__ismain;           /* is there a main function */
int G__globalcomp;       /* make compiled func's global table */
int G__store_globalcomp;
long G__globalvarpointer = G__PVOID; /* make compiled func's global table */

// 10-07-07
// indicate if we have a bundle file in rootcint 
// (impt for printing the temp dicts)
int G__isfilebundled = 0;
G__dictgenmode G__dicttype = kCompleteDictionary;

#ifdef __cplusplus
struct G__filetable G__srcfile[G__MAXFILE];
#endif
int G__nfile;
int G__srcfile_serial;
int G__nobreak;
char G__breakline[G__MAXNAME];
char G__breakfile[G__MAXFILENAME];
#define G__TESTBREAK     0x30
#define G__BREAK         0x10
#define G__NOBREAK       0xef
#define G__CONTUNTIL     0x20
#define G__NOCONTUNTIL   0xdf
#define G__TRACED        0x01
#define G__NOTRACED      0xfe
int G__key; // User-defined function key on/off.
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
FILE* G__dumpfile;
short G__dumpspace;
#endif
int G__def_struct_member;
#ifdef G__FRIEND
int G__friendtagnum;
#endif
int G__tmplt_def_tagnum;
int G__def_tagnum;
int G__tagdefining;
int G__tagnum;
int G__typenum;
short G__iscpp;
short G__cpplock;
short G__clock;
short G__constvar;
short G__isexplicit;
short G__unsigned;
short G__ansiheader;
G__value G__ansipara;
short G__enumdef;
char G__tagname[G__MAXNAME];
long G__store_struct_offset;
FILE* G__header;
FILE* G__temp1;
FILE* G__temp3;
FILE* G__temp5;
FILE* G__temp7;
FILE* G__temp8;
FILE* G__header2;
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
short G__using_alloc;
short G__static_alloc;
int G__func_now;
int G__func_page;
char* G__varname_now;
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
char* G__atexit;
int G__cpp_aryconstruct;
int G__cppconstruct;
int G__access;
int G__steptrace;
int G__debugtrace;
int G__in_pause;
int G__stepover;
int G__fixedscope;
int G__isfuncreturnp2f;
int G__virtual;
int G__oprovld;
int G__p2arylabel[G__MAXVARDIM];
/**************************************************************************
 * If G__exec_memberfunc==1, G__getfunction will search memberfunction
 * in addition to global functions.
 **************************************************************************/
int G__exec_memberfunc;
int G__memberfunc_tagnum;
long G__memberfunc_struct_offset;
/**************************************************************************
 * buffer to store default parameter value
 **************************************************************************/
G__value G__default_parameter;
char G__def_parameter[G__MAXNAME];
//
//  Temporary object buffer
//
struct G__tempobject_list G__tempbuf; // temp list anchor
struct G__tempobject_list* G__p_tempbuf; // temp list head
int G__templevel; // current temp nesting level
//
//  Reference type buffer.
//
int G__reftype;
char* G__refansipara;
//
//  Include path list
//
struct G__includepath G__ipathentry;
struct G__includepath* G__getipathentry() {
   return &G__ipathentry;
}
//
// #pragma compile feature for using a *.dll file.
//
#ifdef G__AUTOCOMPILE
FILE *G__fpautocc;
char G__autocc_c[G__MAXNAME];
char G__autocc_h[G__MAXNAME];
char G__autocc_sl[G__MAXNAME];
char G__autocc_mak[G__MAXNAME];
int G__autoccfilenum = -1;
int G__compilemode;
#endif
//
// Interactive debugging mode support and error recovery.
//
int G__interactive;
G__value G__interactivereturnvalue;
int G__nonansi_func; // Having K&R function in new linking.
int G__sizep2memfunc; // Pointer to member function.
int G__parseextern; // Flag to parse extern variables, -e command line option.
int G__istrace; // Class-specific debugging.
struct G__breakcontinue_list* G__pbreakcontinue; // Break and continue, compilation.
struct G__ConstStringList G__conststringlist;
struct G__ConstStringList* G__plastconststring;
FILE* G__stderr;
FILE* G__stdout;
FILE* G__stdin;
FILE* G__serr;
FILE* G__sout;
FILE* G__sin;
FILE* G__intp_serr;
FILE* G__intp_sout;
FILE* G__intp_sin;
FILE* G__fpundeftype;
//
// Class/struct comment title enhancement.
//
int G__fons_comment;
char* G__setcomment;
int G__precomp_private;
#ifdef G__SECURITY
//
// Secure C++ mode
//
G__UINT32 G__security;
int G__castcheckoff;
int G__security_error;
int G__max_stack_depth;
char G__commandline[G__LONGLINE];
#endif
struct G__Preprocessfilekey G__preprocessfilekey; // Preprocessed file keystring list.
int G__is_operator_newdelete;  // Flag to check for presence of global operator new/operatore delete
//
// $xxx user specific scope object
//
#ifdef G__ANSI
G__value (*G__GetSpecialObject)(const char *name, void** pptr, void** ppdict);
#else
G__value (*G__GetSpecialObject)();
#endif
//
// Path separator
//
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
struct G__AppPragma* G__paddpragma; // Add user defined pragma statement.
#ifdef G__MULTIBYTE
short G__lang = G__UNKNOWNCODING;
#endif
int G__cintready = 0; // A flag to communicate cint ready status to an embedding program.
int G__interactive_undefined = 0;
char G__Allocator[G__ONELINE] = "Allocator";
G__value G__exceptionbuffer;
int G__ispragmainclude = 0;
int G__automaticvar = 1; // automatic variable on/off
int G__xrefflag = 0;
#ifdef G__ROOT
int G__do_smart_unload = 1;
#else
int G__do_smart_unload = 1;
#endif
#ifdef G__ROOT
int G__autoload_stdheader = 0;
#else
int G__autoload_stdheader = 1;
#endif
int G__ignore_stdnamespace = 1;
int G__decl_obj = 0;
struct G__ConstStringList* G__SystemIncludeDir = 0;
int G__command_eval = 0 ;
#ifdef G__MULTITHREADLIBCINT
int G__multithreadlibcint = 1;
#else
int G__multithreadlibcint = 0;
#endif
void (*G__emergencycallback)();
int G__asm_clear_mask = 0;
int G__boolflag;
int G__init = 0;
int G__last_error = 0;
int G__dispmsg = G__DISPALL;
int G__default_link = 1;
int G__gettingspecial = 0;
int G__gcomplevellimit = 1000;
int G__catchexception = 1;
int G__eval_localstatic = 0;
int G__loadingDLL = 0;
int G__mask_error = 0;
G__eolcallback_t G__eolcallback;
int G__throwingexception = 0;
int G__do_setmemfuncenv = 0;
int G__scopelevel = 0;
int G__cintv6 = 0;
struct G__input_file G__lasterrorpos;
int G__nlibs_highwatermark = 0;  // Use to indicates which of the members of G__setup_func_list pertain to the current (set of) library being loaded
int G__nlibs = 0; // Number of values in G__setup_func_list

/**************************************************************************
* Incremented every time the cint dictionary is rewound in scrupto.
* Can be used to see if cached information derived from the dictionary
* is still valid.
**************************************************************************/
int G__scratch_count = 0;
 
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
