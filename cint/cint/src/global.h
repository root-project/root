/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file global.h
 ************************************************************************
 * Description:
 *  Cint parser global variables.
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__GLOBAL_H
#define G__GLOBAL_H

#ifndef __MAKECINT__

#ifndef __CINT__
#ifdef __cplusplus
extern "C" {
#endif
#endif

/***********************************************************************
* EH_env and err_env are only variable name which isn't escaped by 'G__' 
***********************************************************************/
#ifdef G__HSTD
extern jmp_buf EH_env;
/* jmp_buf err_env; */ /* old nucleus */
#endif /* G__HSTD */


/**************************************************************************
* get CINTSYSDIR environment variable here
**************************************************************************/
extern char G__cintsysdir[G__MAXFILENAME] ; 


/**************************************************************************
* make sutpi file
**************************************************************************/
extern  FILE *G__sutpi;      /* sutpi file pointer */


extern int G__typepdecl; /* to be commented */

/**************************************************************************
* Loop compilation instruction, stack and temporaly variables
**************************************************************************/
#ifdef G__ASM

/********************************************************
* whole function bytecode compilation flag
*********************************************************/
extern int G__asm_cond_cp; /* avoid wrong bytecode optimization */

#ifdef G__ASM_WHOLEFUNC
extern int G__asm_wholefunction;
#endif // G__ASM_WHOLEFUNC

extern int G__asm_wholefunc_default_cp;

#ifdef G__ASM_IFUNC
extern long *G__asm_inst; /* p-code instruction buffer */
extern int G__asm_instsize;
extern G__value *G__asm_stack; /* data stack */
extern char *G__asm_name;
extern long G__asm_inst_g[G__MAXINST]; /* p-code instruction buffer */
extern G__value G__asm_stack_g[G__MAXSTACK]; /* data stack */
extern char G__asm_name_g[]; /* buffer to store function names which 
				* is called within the compiled loop */
extern int G__asm_name_p; /* pointer to the function name buffer */
#else // G__ASM_IFUNC
extern long G__asm_inst[G__MAXINST]; /* p-code instruction buffer */
extern G__value G__asm_stack[G__MAXSTACK]; /* data stack */
extern char G__asm_name[]; /* buffer to store function names which 
				* is called within the compiled loop */
extern int G__asm_name_p; /* pointer to the function name buffer */
#endif // G__ASM_IFUNC

extern struct G__param *G__asm_param; /* pointer of parameter buffer to 
				* bring function parameter */
extern int G__asm_exec; /* p-code execution flag */
extern int G__asm_loopcompile; /* loop compilation mode. default on(1). 
			   * This is set to 0 by -O0 */
extern int G__asm_loopcompile_mode; 
extern int G__asm_noverflow; /* When this is set to 1, compilation starts. 
			 * If any error found, reset */
extern int G__asm_dbg; /* p-code debugging flag, only valid when compiled with
		   * G__ASM_DBG */
#ifdef G__ASM_DBG
extern const char *G__LOOPCOMPILEABORT;
#endif // G__ASM_DBG
extern int G__asm_cp;               /* compile time program counter */
extern int G__asm_dt;   /* compile time stack pointer */
extern int G__asm_index;              /* variable index */

// Stubs removal activation status
// Run-time variable for swaping from between old behaviour and the new one (symbol calls)
extern char G__wrappers;
// This variable is set by the configure system
extern char G__nostubs;
#endif // G__ASM

/**************************************************************************
* signal handling
**************************************************************************/
#ifdef G__SIGNAL
extern char *G__SIGINT; 
extern char *G__SIGILL; 
extern char *G__SIGFPE; 
extern char *G__SIGABRT; 
extern char *G__SIGSEGV; 
extern char *G__SIGTERM; 
#ifdef SIGHUP
extern char *G__SIGHUP;
#endif
#ifdef SIGQUIT
extern char *G__SIGQUIT;
#endif
#ifdef SIGTSTP
extern char *G__SIGTSTP;
#endif
#ifdef SIGTTIN
extern char *G__SIGTTIN;
#endif
#ifdef SIGTTOU
extern char *G__SIGTTOU;
#endif
#ifdef SIGALRM
extern char *G__SIGALRM;
#endif
#ifdef SIGUSR1
extern char *G__SIGUSR1;
#endif
#ifdef SIGUSR2
extern char *G__SIGUSR2;
#endif
#endif


/**************************************************************************
* class template 
**************************************************************************/
#ifdef G__TEMPLATECLASS
extern struct G__Definedtemplateclass G__definedtemplateclass;
#endif
extern int G__macroORtemplateINfile;

#ifdef G__TEMPLATEFUNC
extern struct G__Definetemplatefunc G__definedtemplatefunc;
#endif


/**************************************************************************
* Macro statement support
**************************************************************************/
extern FILE *G__mfp;
extern fpos_t G__nextmacro;
extern int G__mline;
extern const char *G__macro;
extern struct G__Deffuncmacro G__deffuncmacro;
extern char G__macros[16*G__LONGLINE];
#ifdef __cplusplus
extern G__FastAllocString G__ppopt; 
#endif
extern char *G__allincludepath;
extern const char *G__undeflist;
struct G__funcmacro_stackelt;
extern struct G__funcmacro_stackelt* G__funcmacro_stack;

/**************************************************************************
* Macro constant
**************************************************************************/
extern int G__macro_defining;

/**************************************************************************
* Array type typedef support, 'typedef int a[10];'
**************************************************************************/
extern int G__typedefnindex;
extern int *G__typedefindex;


/**************************************************************************
* Text processing capability
*
*   fp=fopen("xxx","r");
*   while($read(fp)) {
*      printf("%d %s %s\n",$#,$1,$2);
*   }
**************************************************************************/
extern char G__oline[G__LONGLINE*2],G__argb[G__LONGLINE*2],*G__arg[G__ONELINE];
extern int G__argn;


/**************************************************************************
* structure for global and local variables
*
**************************************************************************/
extern struct G__var_array G__global ;
extern struct G__var_array *G__p_local;
extern struct G__inheritance G__globalusingnamespace;

/**************************************************************************
* structure for struct,union tag information
* structure for typedef information
**************************************************************************/
extern struct G__tagtable G__struct;
extern struct G__typedef  G__newtype;

// 03-07-07
// This is not the libname but the .nm file that is passed to newlink
// with all the symbols needed to register the functions
extern const char *G__libname;

/**************************************************************************
* structure for input file
**************************************************************************/
extern struct G__input_file G__ifile;
#ifdef G__NONANSI
extern char *G__psep;
#else
/* contains path sequencer UNIX / , Windows \ , Mac : */
extern const char *G__psep;
#endif


/**************************************************************************
* structure for ifunc (Interpreted FUNCtion) table
**************************************************************************/
extern struct G__ifunc_table_internal G__ifunc ;
extern struct G__ifunc_table_internal *G__p_ifunc;


/**************************************************************************
* cint G__value constants
**************************************************************************/
extern G__value G__null,G__start,G__default,G__one;
extern G__value G__block_break,G__block_continue;
extern G__value G__block_goto;
extern char G__gotolabel[G__MAXNAME];

/**************************************************************************
* allocation of array by new operator ?
**************************************************************************/
extern struct G__newarylist G__newarray;



/**************************************************************************
*  flags
**************************************************************************/
/* int G__error_flag; */ /* syntax error flag */
extern int G__debug;            /* trace input file */
extern int G__breakdisp;        /* display input file at break point */
/* int G__cont ;  */        /* continue G__cont lines */
extern int G__break;            /* break flab */
extern int G__step;             /* step execution flag */
extern int G__charstep;         /* char step flag */
extern int G__break_exit_func;  /* break at function exit */
/* int G__no_exec_stack ; */ /* stack for G__no_exec */
extern int G__eof;                /* end of file flag */
extern int G__no_exec;          /* no execution(ignore) flag */
extern int G__no_exec_compile;
extern char G__var_type;      /* variable decralation type */
extern char G__var_typeB;      /* variable decralation type for function return*/
extern int G__prerun;           /* pre-run flag */
extern int G__funcheader;       /* function header mode */
extern int G__return;           /* 0:normal,1:return,2:exit()or'q',3:'qq'(not used)*/
/* int G__extern_global;    number of globals defined in c function */
extern int G__disp_mask;        /* temporary read count */
extern int G__temp_read;        /* temporary read count */
extern int G__switch;           /* in a switch, parser should evaluate case expressions */
extern int G__switch_searching; /* in a switch, parser should return after evaluating a case expression */
extern int G__eof_count;        /* end of file error flag */
extern int G__ismain;           /* is there a main function */
extern int G__globalcomp;       /* make compiled func's global table */
extern int G__store_globalcomp;
extern long G__globalvarpointer; /* make compiled func's global table */
extern int G__isfilebundled;       /* indicate if we have a bundle file in rootcint */

#ifdef __cplusplus
extern struct G__filetable G__srcfile[G__MAXFILE];
#endif
extern int G__nfile;
extern int G__srcfile_serial; /* Serial/Unique number for the state of G__srcfile, increased anytime a file is added or removed */
extern int G__globals_serial; /* Serial/Unique number for the state of G__global, increased anytime a variable is added or removed */

extern int G__nobreak;
extern char G__breakline[G__MAXNAME];
extern char G__breakfile[G__MAXFILENAME];

extern int G__key;              /* user function key on/off */
extern  FILE *G__dumpreadline[6];
extern short G__Xdumpreadline[6];

extern char G__xfile[];
extern char G__tempc[];

extern int G__doingconstruction;

#ifdef G__DUMPFILE
extern FILE *G__dumpfile;
extern short G__dumpspace;
#endif

extern int G__def_struct_member;

#ifdef G__FRIEND
extern int G__friendtagnum;
#endif
extern int G__tmplt_def_tagnum;
extern int G__def_tagnum;
extern int G__tagdefining;
extern int G__tagnum ; /* -1 */
extern int G__typenum ;
extern short G__iscpp ;
extern short G__cpplock ;
extern short G__clock ;
extern short G__constvar ;
extern short G__isexplicit ;
extern short G__unsigned ;
extern short G__ansiheader ;
extern G__value G__ansipara;
extern short G__enumdef;

// 10-07-07
// Can tell if we are writing the temp. dicts or the real one
extern G__dictgenmode G__dicttype;
extern char G__tagname[G__MAXNAME];
extern long G__store_struct_offset; /* used to be int */
extern FILE *G__header,*G__temp1,*G__temp3,*G__temp5,*G__temp7,*G__temp8;
extern FILE *G__header2;
extern int G__decl;

#ifndef G__OLDIMPLEMENTATION1259
extern G__SIGNEDCHAR_T G__isconst;
#endif


extern char G__nam[G__MAXFILENAME];
extern char G__assertion[G__ONELINE];
extern short G__longjump;
extern short G__coredump;
extern short G__definemacro;

extern short G__noerr_defined;

extern short G__using_alloc;
extern short G__static_alloc;
extern int G__func_now ;
extern int G__func_page;
extern char *G__varname_now;

extern short G__twice;
extern short G__othermain;
extern int G__cpp;
extern int G__include_cpp;
extern char G__ccom[G__MAXFILENAME];
extern char G__cppsrcpost[G__LENPOST];
extern char G__csrcpost[G__LENPOST];
extern char G__cpphdrpost[G__LENPOST];
extern char G__chdrpost[G__LENPOST];

extern short G__dispsource;
extern short G__breaksignal;

extern short G__bitfield;

extern char *G__atexit ;

extern int G__cpp_aryconstruct;
extern int G__cppconstruct;

extern int G__access;

extern int G__steptrace,G__debugtrace;

extern int G__in_pause;
extern int G__stepover;


/* This must be set everytime before G__interpret_func() is called */
extern int G__fixedscope;

extern int G__isfuncreturnp2f;

extern int G__virtual;


/* #define G__OLDIMPLEMENTATION78 */

extern int G__oprovld;

extern int G__p2arylabel[G__MAXVARDIM];


/**************************************************************************
* If G__exec_memberfunc==1, G__getfunction will search memberfunction
* in addition to global functions.
**************************************************************************/
extern int G__exec_memberfunc;
extern int G__memberfunc_tagnum;
extern long G__memberfunc_struct_offset;



/**************************************************************************
* buffer to store default parameter value
**************************************************************************/
//extern G__value G__default_parameter;
extern char G__def_parameter[G__MAXNAME];


/**************************************************************************
* temporary object buffer
**************************************************************************/
extern struct G__tempobject_list *G__p_tempbuf,G__tempbuf;
extern int G__templevel;

/**************************************************************************
* reference type buffer
**************************************************************************/
extern int G__reftype;
extern char *G__refansipara;

/********************************************************************
* include path list
********************************************************************/
extern struct G__includepath G__ipathentry;


/*********************************************************************
* dynamic link library(shared library) enhancement
*********************************************************************/
#ifdef G__SHAREDLIB
extern short G__allsl;
#endif 

/*********************************************************************
* #pragma compile feature utilizing DLL
*********************************************************************/
#ifdef G__AUTOCOMPILE
extern FILE *G__fpautocc;
extern char G__autocc_c[];
extern char G__autocc_h[];
extern char G__autocc_sl[];
extern char G__autocc_mak[];
extern int G__autoccfilenum;
extern int G__compilemode;
#endif


/**************************************************************************
* Interactive debugging mode support and error recover
**************************************************************************/
extern int G__interactive;
extern G__value G__interactivereturnvalue;


/**************************************************************************
* Old style compiled function name/pointer table
**************************************************************************/
extern G__COMPLETIONLIST G__completionlist[];


#ifdef G__MEMTEST
/**************************************************************************
* memory leak test
**************************************************************************/
extern FILE *G__memhist;
#endif

/**************************************************************************
* Having K&R function in new linking 
**************************************************************************/
extern int G__nonansi_func;

/**************************************************************************
* pointer to member function
**************************************************************************/
extern int G__sizep2memfunc;

/**************************************************************************
* flag to parse extern variables '-e' command line option
**************************************************************************/
extern int G__parseextern;

/**************************************************************************
* class specific debugging 
**************************************************************************/
extern int G__istrace;

/**************************************************************************
* break, continue compilation
**************************************************************************/
extern struct G__breakcontinue_list *G__pbreakcontinue;

extern FILE *G__stderr;
extern FILE *G__stdout;
extern FILE *G__stdin;

extern FILE *G__serr;
extern FILE *G__sout;
extern FILE *G__sin;

extern FILE *G__intp_serr;
extern FILE *G__intp_sout;
extern FILE *G__intp_sin;

extern FILE* G__fpundeftype;

/**************************************************************************
* Class/struct comment title enhancement
**************************************************************************/
extern int G__fons_comment;
extern char *G__setcomment;

extern int G__precomp_private;

/*********************************************************************
* const string list
*********************************************************************/
extern struct G__ConstStringList G__conststringlist ;
extern struct G__ConstStringList *G__plastconststring ;

#ifdef G__SECURITY
/**************************************************************************
* Secure C++ mode
**************************************************************************/
/* extern G__UINT32 G__security;  declared in security.h */
extern int G__castcheckoff;
extern int G__security_error;
extern int G__max_stack_depth;
extern char G__commandline[];
#endif

/**************************************************************************
* Preprocessed file keystring list
**************************************************************************/
extern struct G__Preprocessfilekey G__preprocessfilekey;

/**************************************************************************
* Flag to check global operator new/delete()
**************************************************************************/
extern int G__is_operator_newdelete ;


/**************************************************************************
* Path separator
**************************************************************************/
#if defined(G__NONANSI)
extern char *G__psep;
#else
extern const char *G__psep;
#endif

#ifndef G__OLDIMPLEMENtATION451
/**************************************************************************
* add user defined pragma statement
**************************************************************************/
extern struct G__AppPragma *G__paddpragma;
#endif

#ifdef G__MULTIBYTE
/**************************************************************************
* multi-byte coding system selection
**************************************************************************/
extern short G__lang;
#endif

/**************************************************************************
* A flag no notify cint ready status to embedding program
* This flag is set by cint internal only, and read by G__getcintready() API
**************************************************************************/
extern int G__cintready;

/**************************************************************************
* interactive return for undefined symbol and G__pause
**************************************************************************/
extern int G__interactive_undefined;

/**************************************************************************
* STL Allocator workaround data
**************************************************************************/
extern char G__Allocator[];

/**************************************************************************
* Exception object buffer
**************************************************************************/
extern G__value G__exceptionbuffer;

/**************************************************************************
* Exception object buffer
**************************************************************************/
extern int G__ispragmainclude;

/**************************************************************************
* automatic variable on/off
**************************************************************************/
extern int G__automaticvar;

/**************************************************************************
* Local variable , bytecode compiler workaround
**************************************************************************/
extern int G__xrefflag;

/**************************************************************************
* Local variable , bytecode compiler workaround
**************************************************************************/
extern int G__do_smart_unload;

extern int G__autoload_stdheader;

extern int G__ignore_stdnamespace;

extern int G__decl_obj;

extern struct G__ConstStringList* G__SystemIncludeDir;

extern int G__command_eval ;

extern int G__multithreadlibcint ;

extern void (*G__emergencycallback)();

extern int G__asm_clear_mask;

extern int G__boolflag;

extern int G__init;

extern int G__last_error;
extern int G__const_noerror;

extern int G__dispmsg;

extern int G__default_link;

/* 1713 */
extern int G__gettingspecial;

extern int G__gcomplevellimit;

extern int G__catchexception;

extern int G__eval_localstatic;

extern int G__loadingDLL;


extern int G__mask_error;

typedef void (*G__eolcallback_t) G__P((const char* fname,int linenum));
extern G__eolcallback_t G__eolcallback;

extern int G__throwingexception;

extern int G__do_setmemfuncenv;

extern int G__scopelevel;
extern int G__cintv6;

extern struct G__input_file G__lasterrorpos;

extern int G__initval_eval;
extern int G__dynconst;

/*extern std::list<G__DLLINIT>* G__initpermanentsl;*/
extern int G__ispermanentsl;
extern int G__nlibs;
extern int G__nlibs_highwatermark;
   
/**************************************************************************
* Incremented every time the cint dictionary is rewound in scrupto.
* Can be used to see if cached information derived from the dictionary
* is still valid.
**************************************************************************/
extern int G__scratch_count;

#ifndef __CINT__
#ifdef __cplusplus
}
#endif
#endif

#endif /* __MAKECINT__ */

#endif /* G__GLOBAL_H */

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
