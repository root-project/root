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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "common.h"

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
int G__switch;           /* switch case control flag */
int G__mparen;           /* switch case break nesting control */
int G__eof_count;        /* end of file error flag */
int G__ismain;           /* is there a main function */
int G__globalcomp;       /* make compiled func's global table */
int G__store_globalcomp;
long G__globalvarpointer = G__PVOID; /* make compiled func's global table */

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
int G__friendtagnum;
#endif
#ifndef G__OLDIMPLEMENTATION440
int G__tmplt_def_tagnum;
#endif
int G__def_tagnum;
int G__tagdefining;
int G__tagnum ; /* -1 */
int G__typenum ;
short G__iscpp ;
short G__cpplock ;
short G__clock ;
short G__constvar ;
#ifndef G__OLDIMPLEMENTATION1250
short G__isexplicit ;
#endif
short G__unsigned ;
short G__ansiheader ;
G__value G__ansipara;
short G__enumdef;

char G__tagname[G__MAXNAME];
long G__store_struct_offset=0;
FILE *G__header,*G__temp1,*G__temp3,*G__temp5,*G__temp7,*G__temp8;
FILE *G__header2;
int G__decl;

#ifndef G__OLDIMPLEMENTATION1259
G__SIGNEDCHAR_T G__isconst;
#endif

#ifdef G__OLDIMPLEMENTATION435
char *G__conststring[G__MAXSTRING];
int   G__conststringhash[G__MAXSTRING];
short G__allstring;
#endif

char G__nam[G__MAXFILENAME];
char G__assertion[G__ONELINE];
short G__longjump;
short G__coredump;
short G__definemacro;

short G__noerr_defined;

short G__static_alloc;
int G__func_now ;
int G__func_page;
char *G__varname_now;

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
struct G__ifunc_table *G__ifunc_exist();
struct G__ifunc_table *G__ifunc_ambiguous();

/* #define G__OLDIMPLEMENTATION78 */

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

/********************************************************************
* include path list
********************************************************************/
struct G__includepath G__ipathentry;
struct G__includepath *G__getipathentry() { return &G__ipathentry; }


/*********************************************************************
* #pragma compile feature utilizing DLL
*********************************************************************/
#ifdef G__AUTOCOMPILE
FILE *G__fpautocc;
#ifndef G__OLDIMPLEMENTATION486
char G__autocc_c[G__MAXNAME];
char G__autocc_h[G__MAXNAME];
char G__autocc_sl[G__MAXNAME];
char G__autocc_mak[G__MAXNAME];
int G__autoccfilenum = -1;
#else
char *G__autocc_c="G__autocc.c";
char *G__autocc_h="G__autocc.h";
char *G__autocc_sl="G__autocc.sl";
#endif
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
#ifdef G__OLDIMPLEMENTATION1636
#endif

FILE *G__stderr;
FILE *G__stdout;
FILE *G__stdin;

FILE *G__serr;
FILE *G__sout;
FILE *G__sin;

#ifndef G__OLDIMPLEMENTATION713
FILE *G__intp_serr;
FILE *G__intp_sout;
FILE *G__intp_sin;
#endif

#ifndef G__OLDIMPLEMENTATION411
FILE *G__fpundeftype;
#endif

#ifdef G__FONS_COMMENT
/**************************************************************************
* Class/struct comment title enhancement
**************************************************************************/
int G__fons_comment;
char *G__setcomment;
#endif

int G__precomp_private;

#ifdef G__SECURITY
/**************************************************************************
* Secure C++ mode
**************************************************************************/
G__UINT32 G__security;
int G__castcheckoff;
int G__security_error;
int G__max_stack_depth;
char G__commandline[G__LONGLINE];
#endif

/**************************************************************************
* Preprocessed file keystring list
**************************************************************************/
struct G__Preprocessfilekey G__preprocessfilekey;

/**************************************************************************
* Flag to check global operator new/delete()
**************************************************************************/
int G__is_operator_newdelete ;

/**************************************************************************
* $xxx user specific scope object
**************************************************************************/
#ifdef G__ANSI
#if !defined(G__OLDIMPLEMENTATION481)
G__value (*G__GetSpecialObject)(char *name,void** pptr,void** ppdict);
#elif !defined(G__OLDIMPLEMENTATION455)
G__value (*G__GetSpecialObject)(char *name,void* ptr);
#else
G__value (*G__GetSpecialObject)(char *name);
#endif
#else
G__value (*G__GetSpecialObject)();
#endif

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

#ifndef G__OLDIMPLEMENTATION451
/**************************************************************************
* add user defined pragma statement
**************************************************************************/
struct G__AppPragma *G__paddpragma;
#endif

#ifdef G__MULTIBYTE
/**************************************************************************
* multi-byte coding system selection
**************************************************************************/
short G__lang = G__UNKNOWNCODING;
#endif

#ifndef G__OLDIMPLEMENTATION563
/**************************************************************************
* A flag no notify cint ready status to embedding program
* This flag is set by cint internal only, and read by G__getcintready() API
**************************************************************************/
int G__cintready=0;
#endif

#ifndef G__OLDIMPLEMENTATION630
/**************************************************************************
* interactive return for undefined symbol and G__pause
**************************************************************************/
int G__interactive_undefined=0;
#endif

#ifndef G__OLDIMPLEMENTATION734
/**************************************************************************
* STL Allocator workaround data
**************************************************************************/
char G__Allocator[G__ONELINE] = "Allocator";
#endif

#ifndef G__OLDIMPLEMENTATION754
/**************************************************************************
* Exception object buffer
**************************************************************************/
G__value G__exceptionbuffer;
#endif

#ifndef G__OLDIMPLEMENTATION782
/**************************************************************************
* Exception object buffer
**************************************************************************/
int G__ispragmainclude=0;
#endif

#ifndef G__OLDIMPLEMENTATION1097
/**************************************************************************
* automatic variable on/off
**************************************************************************/
int G__automaticvar=1;
#endif

#ifndef G__OLDIMPLEMENTATION1164
/**************************************************************************
* Local variable , bytecode compiler workaround
**************************************************************************/
int G__xrefflag=0;
#endif

#ifndef G__OLDIMPLEMENTATION1273
/**************************************************************************
* Local variable , bytecode compiler workaround
**************************************************************************/
#ifdef G__ROOT
int G__do_smart_unload=1;
#else
int G__do_smart_unload=1;
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1278
#ifdef G__ROOT
int G__autoload_stdheader = 0;
#else
int G__autoload_stdheader = 1;
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1285
int G__ignore_stdnamespace = 1;
#endif

#ifndef G__OLDIMPLEMENTATION1349
int G__decl_obj=0;
#endif

#ifndef G__OLDIMPLEMENTATION1451
struct G__ConstStringList* G__SystemIncludeDir=0;
#endif

#ifndef G__OLDIMPLEMENTATION1476
int G__command_eval=0 ;
#endif

#ifndef G__OLDIMPLEMENTATION1525
#ifdef G__MULTITHREADLIBCINT
int G__multithreadlibcint = 1;
#else
int G__multithreadlibcint = 0;
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1548
void (*G__emergencycallback)();
#endif

#ifndef G__OLDIMPLEMENTATION1570
int G__asm_clear_mask = 0;
#endif

#ifndef G__OLDIMPLEMENTATION1593
int G__boolflag;
#endif

#ifndef G__OLDIMPLEMENTATION1599
int G__init=0;
#endif

#ifndef G__OLDIMPLEMENTATION1600
int G__last_error = 0;
#endif

int G__dispmsg = G__DISPALL;

#ifndef G__OLDIMPLEMENTATION1700
int G__default_link = 1;
#endif

/* 1713 */
int G__gettingspecial = 0;

#ifndef G__OLDIMPLEMENTATION1725
int G__gcomplevellimit=1000;
#endif

#ifndef G__OLDIMPLEMENTATION1726
int G__catchexception=1;
#endif

#ifndef G__OLDIMPLEMENTATION1551
int G__eval_localstatic=0;
#endif

#ifndef G__OLDIMPLEMENTATION1854
int G__loadingDLL=0;
#endif

#ifndef G__OLDIMPLEMENTATION1986
int G__stubcall=0;
#endif

#ifndef G__OLDIMPLEMENTATION2002
int G__mask_error=0;
#endif

#ifndef G__OLDIMPLEMENTATION2005
G__eolcallback_t G__eolcallback;
#endif

#ifndef G__OLDIMPLEMENTATION2111
int G__throwingexception=0;
#endif

#ifndef G__OLDIMPLEMENTATION2155
int G__do_setmemfuncenv = 0;
#endif

int G__scopelevel=0;
int G__cintv6 = 0;

struct G__input_file G__lasterrorpos;


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
