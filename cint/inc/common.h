/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file common.h
 ************************************************************************
 * Description:
 * Common header file for cint parser.
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__COMMON_H
#define G__COMMON_H


/**************************************************************************
* HSTD local facilities are turned on by defining G__HSTD. 
* Usually doesn't matter
**************************************************************************/
#ifndef __CINT__
/* #define G__HSTD */
#endif

/**************************************************************************
* include header file which includes macros and structs.
*
**************************************************************************/
#include <time.h>
#define G__CINTBODY
#include "G__ci.h"

/**************************************************************************
* Windows tempfile
*
*  Borland C++ does not have _tmpnam, need to reimplement by GetTempPath()
**************************************************************************/
#ifdef G__WIN32
#ifndef G__BORLAND
#define G__TMPFILE
#endif
#endif

/**************************************************************************
* GNU readline
*
*  G__GNUREADLINE will be defined only if GNU readline library is compiled
* successfully.
**************************************************************************/
/* #define G__GNUREADLINE */

/**************************************************************************
* Memory leakage test. Usually, commented out
*
**************************************************************************/
#ifdef G__DEBUG
#include "memtest.h"
#endif


/**************************************************************************
* On line file loading
*
**************************************************************************/
#define G__ONLINELOAD

/**************************************************************************
* standard include file <>
**************************************************************************/
#define G__STDINCLUDE

/**************************************************************************
* make sutpi file
**************************************************************************/
#define G__SUTPI

#define G__SUT_PROTOTYPE  1
#define G__SUT_REALFUNC   2

/**************************************************************************
* Array initialization
**************************************************************************/
#define G__INITARY


/**************************************************************************
* Scope operator category
**************************************************************************/
#define G__NOSCOPEOPR    0
#define G__GLOBALSCOPE   1
#define G__CLASSSCOPE    2


/**************************************************************************
* Scope operator category
**************************************************************************/
#define G__UNARYOP 'U'

#define G__OPR_ADD          '+'
#define G__OPR_SUB          '-'
#define G__OPR_MUL          '*'
#define G__OPR_DIV          '/'

#define G__OPR_RSFT         'R'
#define G__OPR_LSFT         'L'

#define G__OPR_LT           '<'
#define G__OPR_GT           '>'
#define G__OPR_LE           'l'
#define G__OPR_GE           'G'
#define G__OPR_EQ           'E'
#define G__OPR_NE           'N'

#define G__OPR_AND          'A'
#define G__OPR_OR           'O'

#define G__OPR_INCREMENT    'I'
#define G__OPR_DECREMENT    'D'

#define G__OPR_ADDASSIGN     1
#define G__OPR_SUBASSIGN     2
#define G__OPR_MODASSIGN     3
#define G__OPR_MULASSIGN     4
#define G__OPR_DIVASSIGN     5
#define G__OPR_RSFTASSIGN    6
#define G__OPR_LSFTASSIGN    7
#define G__OPR_BANDASSIGN    8
#define G__OPR_BORASSIGN     9
#define G__OPR_EXORASSIGN   10
#define G__OPR_ANDASSIGN    11
#define G__OPR_ORASSIGN     12

#define G__OPR_POSTFIXINC   13
#define G__OPR_PREFIXINC    14
#define G__OPR_POSTFIXDEC   15
#define G__OPR_PREFIXDEC    16

#ifndef G__OLDIMPLEMENTATION572
#define G__OPR_POSTFIXINC_I  0x110
#define G__OPR_PREFIXINC_I   0x111
#define G__OPR_POSTFIXDEC_I  0x112
#define G__OPR_PREFIXDEC_I   0x113

#define G__OPR_POSTFIXINC_S  0x410
#define G__OPR_PREFIXINC_S   0x411
#define G__OPR_POSTFIXDEC_S  0x412
#define G__OPR_PREFIXDEC_S   0x413

#define G__OPR_POSTFIXINC_L  0x510
#define G__OPR_PREFIXINC_L   0x511
#define G__OPR_POSTFIXDEC_L  0x512
#define G__OPR_PREFIXDEC_L   0x513

#define G__OPR_POSTFIXINC_H  0x610
#define G__OPR_PREFIXINC_H   0x611
#define G__OPR_POSTFIXDEC_H  0x612
#define G__OPR_PREFIXDEC_H   0x613

#define G__OPR_POSTFIXINC_R  0x710
#define G__OPR_PREFIXINC_R   0x711
#define G__OPR_POSTFIXDEC_R  0x712
#define G__OPR_PREFIXDEC_R   0x713

#define G__OPR_POSTFIXINC_K  0x810
#define G__OPR_PREFIXINC_K   0x811
#define G__OPR_POSTFIXDEC_K  0x812
#define G__OPR_PREFIXDEC_K   0x813

#define G__OPR_ADD_II        0x100
#define G__OPR_SUB_II        0x101
#define G__OPR_MUL_II        0x102
#define G__OPR_DIV_II        0x103
#define G__OPR_LT_II         0x104
#define G__OPR_GT_II         0x105
#define G__OPR_LE_II         0x106
#define G__OPR_GE_II         0x107
#define G__OPR_EQ_II         0x108
#define G__OPR_NE_II         0x109
#define G__OPR_ADDASSIGN_II  0x10a
#define G__OPR_SUBASSIGN_II  0x10b
#define G__OPR_MULASSIGN_II  0x10c
#define G__OPR_DIVASSIGN_II  0x10d

#define G__OPR_POSTFIXINC_D  0x210
#define G__OPR_PREFIXINC_D   0x211
#define G__OPR_POSTFIXDEC_D  0x212
#define G__OPR_PREFIXDEC_D   0x213
#define G__OPR_POSTFIXINC_F  0x310
#define G__OPR_PREFIXINC_F   0x311
#define G__OPR_POSTFIXDEC_F  0x312
#define G__OPR_PREFIXDEC_F   0x313

#define G__OPR_ADD_DD        0x200
#define G__OPR_SUB_DD        0x201
#define G__OPR_MUL_DD        0x202
#define G__OPR_DIV_DD        0x203
#define G__OPR_LT_DD         0x204
#define G__OPR_GT_DD         0x205
#define G__OPR_LE_DD         0x206
#define G__OPR_GE_DD         0x207
#define G__OPR_EQ_DD         0x208
#define G__OPR_NE_DD         0x209
#define G__OPR_ADDASSIGN_DD  0x20a
#define G__OPR_SUBASSIGN_DD  0x20b
#define G__OPR_MULASSIGN_DD  0x20c
#define G__OPR_DIVASSIGN_DD  0x20d

#define G__OPR_ADDASSIGN_FD  0x30a
#define G__OPR_SUBASSIGN_FD  0x30b
#define G__OPR_MULASSIGN_FD  0x30c
#define G__OPR_DIVASSIGN_FD  0x30d
#endif /* ON572 */

/**************************************************************************
* G__reftype, var->reftype[], ifunc->reftype[] flag
**************************************************************************/
#define G__PARANORMAL       0
#define G__PARAREFERENCE    1
#define G__PARAP2P          2
#define G__PARAP2P2P        3

#define G__POINTER2FUNC    0
#define G__FUNCRETURNP2F   1
#define G__POINTER2MEMFUNC 2
#define G__CONSTRUCTORFUNC 3


/**************************************************************************
* var->constvar[]
**************************************************************************/
#define G__VARIABLE       0
#define G__CONSTVAR       1
#define G__LOCKVAR        2
#define G__DYNCONST       2
#define G__PCONSTVAR      4
#define G__PCONSTCONSTVAR 5
#define G__CONSTFUNC      8

/**************************************************************************
* Class charasteristics
**************************************************************************/
#define G__HAS_DEFAULTCONSTRUCTOR  0x01
#define G__HAS_COPYCONSTRUCTOR     0x02
#define G__HAS_DESTRUCTOR          0x04
#define G__HAS_ASSIGNMENTOPERATOR  0x08

/**************************************************************************
* Default parameter expression as function
**************************************************************************/
#define G__DEFAULT_FUNCCALL 9

/**************************************************************************
* break,continue,goto,default statement 
**************************************************************************/
#define G__SWITCH_START    -3229
#define G__SWITCH_DEFAULT  -1000
#define G__BLOCK_BREAK     -1001
#define G__BLOCK_CONTINUE  -1002

/**************************************************************************
* ByteCode compiler (loops only)
*
*  Following macro definitions accelarates the most inner loop execution 
* by factor of 4.
*  Search macro symble G__ASM to figure out related source for the inner
* loop compiler.
*  With -O0 command line option, the inner loop compilation mode can be 
* turned off.
**************************************************************************/

/*********************************************
* loop compile mode turned on
*********************************************/
#define G__ASM

/*********************************************
* Old style compiled function name buffer size
*********************************************/
#define G__ASM_FUNCNAMEBUF 200

/*********************************************
* Loop compile optimizer turned on
*********************************************/
#define G__ASM_OPTIMIZE

/*********************************************
* Nested loop compilation
*********************************************/
#define G__ASM_NESTING

/*********************************************
* ifunc compilation
*********************************************/
#define G__ASM_IFUNC


/*********************************************
* loop compile debug mode
*  Not defined usually. Needed only when
* debugging of inner loop compiler.
*********************************************/
#ifdef G__DEBUG
#define G__ASM_DBG
#endif


#ifdef G__ASM

/*********************************************
* p-code instructions
*  Instructions which appears frequently in
* typical application is assigned to a smaller 
* number. This speeds up the execution by
* 10~20%.
*********************************************/
#define G__LDST_VAR_P         0x0000
#define G__LDST_LVAR_P        0x0001
#define G__LDST_MSTR_P        0x0002
#define G__LDST_VAR_INDEX     0x0003
#define G__LDST_VAR_INDEX_OPR 0x0004
#define G__OP2_OPTIMIZED      0x0005
#define G__OP1_OPTIMIZED      0x0006
#define G__LD                 0x0007
#define G__CL                 0x0008
#define G__OP2                0x0009
#define G__CMPJMP             0x000a
#define G__INCJMP             0x000b
#define G__CNDJMP             0x000c
#define G__JMP                0x000d
#define G__POP                0x000e
#define G__LD_FUNC            0x000f
#define G__RETURN             0x0010
#define G__CAST               0x0011
#define G__OP1                0x0012
#define G__LETVVAL            0x0013
#define G__ADDSTROS           0x0014
#define G__LETPVAL            0x0015
#define G__TOPNTR	      0x0016
#define G__NOT                0x0017
#define G__BOOL               0x0018
#define G__ISDEFAULTPARA      0x0019

#define G__LD_VAR             0x001a
#define G__ST_VAR             0x001b
#define G__LD_MSTR            0x001c
#define G__ST_MSTR            0x001d
#define G__LD_LVAR            0x001e
#define G__ST_LVAR            0x001f
#define G__CMP2               0x0020
#define G__PUSHSTROS          0x0021
#define G__SETSTROS           0x0022
#define G__POPSTROS           0x0023
#define G__SETTEMP            0x0024
#define G__FREETEMP           0x0025
#define G__GETRSVD            0x0026
#define G__REWINDSTACK        0x0027
#define G__CND1JMP            0x0028
#define G__LD_IFUNC	      0x0029
#define G__NEWALLOC           0x002a
#define G__SET_NEWALLOC       0x002b
#define G__DELETEFREE         0x002c
#define G__SWAP               0x002d
#define G__BASECONV           0x002e
#define G__STORETEMP          0x002f
#define G__ALLOCTEMP          0x0030
#define G__POPTEMP            0x0031
#define G__REORDER            0x0032
#define G__LD_THIS            0x0033
#define G__RTN_FUNC           0x0034
#define G__SETMEMFUNCENV      0x0035
#define G__RECMEMFUNCENV      0x0036
#define G__ADDALLOCTABLE      0x0037
#define G__DELALLOCTABLE      0x0038
/* #define G__BASECONSTRUCT     0x00XX */
#define G__BASEDESTRUCT       0x0039
#define G__REDECL             0x003a
#define G__TOVALUE            0x003b
#define G__INIT_REF           0x003c
#define G__PUSHCPY            0x003d
#define G__LETNEWVAL          0x003e
#define G__SETGVP             0x003f
#define G__TOPVALUE           0x0040
#define G__CTOR_SETGVP        0x0041

#define G__THROW              0x0042
#define G__CATCH              0x0043

#define G__NOP                0xffff

struct G__breakcontinue_list {
  int destination;
  int breakcontinue;
  struct G__breakcontinue_list *prev;
};

/*********************************************
* loop compiler
*  limit numbers
*********************************************/
#define G__MAXINST     0x1000
#define G__MAXSTACK    0x100
#define G__MAXSTRSTACK  0x10

/*********************************************
* macros for loop compiler
*********************************************/
#define G__ALLOC_ASMENV                            \
  int store_asm_exec,store_asm_loopcompile

#define G__STORE_ASMENV                            \
    store_asm_exec = G__asm_exec;                  \
    store_asm_loopcompile=G__asm_loopcompile;      \
    if(store_asm_exec) G__asm_loopcompile=0;       \
    G__asm_exec = 0

#ifndef G__OLDIMPLEMENTATION1155
#define G__RECOVER_ASMENV                          \
    G__asm_exec=store_asm_exec;                    \
    G__asm_loopcompile=G__asm_loopcompile_mode
#else
#define G__RECOVER_ASMENV                          \
    G__asm_exec=store_asm_exec;                    \
    G__asm_loopcompile=store_asm_loopcompile
#endif


/*********************************************
* whole function bytecode compilation flag
*********************************************/
#ifdef G__ASM_WHOLEFUNC
#define G__ASM_FUNC_NOP         0x00
#define G__ASM_FUNC_COMPILE     0x01
#define G__ASM_FUNC_EXEC        0x02
#define G__ASM_FUNC_COMPILEEXEC 0x03

/* number of line to try bytecode compilation */
#define G__ASM_BYTECODE_FUNC_LIMIT  G__MAXINST

#define G__ASM_VARGLOBAL     0x00
#define G__ASM_VARLOCAL      0x01
#endif /* G__ASM_WHOLEFUNC */

#define G__LOCAL_VAR           0
#define G__GLOBAL_VAR          1
#define G__BYTECODELOCAL_VAR   2

#endif /* of G__ASM */



/**************************************************************************
* signal handling
**************************************************************************/
#define G__SIGNAL



/**************************************************************************
* class template 
**************************************************************************/
#define G__TEMPLATECLASS
#define G__TEMPLATEMEMFUNC
#define G__TEMPLATEFUNC    /* Experimental */

#ifdef G__TEMPLATECLASS

#define G__TMPLT_CLASSARG     'u'
#define G__TMPLT_TMPLTARG     't'
#define G__TMPLT_SIZEARG      'o'

#define G__TMPLT_CHARARG      'c'
#define G__TMPLT_UCHARARG     'b'
#define G__TMPLT_INTARG       'i'
#define G__TMPLT_UINTARG      'h'
#define G__TMPLT_SHORTARG     's'
#define G__TMPLT_USHORTARG    'r'
#define G__TMPLT_LONGARG      'l'
#define G__TMPLT_ULONGARG     'k'
#define G__TMPLT_FLOATARG     'f'
#define G__TMPLT_DOUBLEARG    'd'

#define G__TMPLT_POINTERARG1   1
#define G__TMPLT_POINTERARG2   2
#define G__TMPLT_POINTERARG3   3

#ifdef G__TEMPLATEMEMFUNC

#ifndef G__OLDIMPLEMENTATION691
/* Doubly linked list of long int, methods are described in tmplt.c */
struct G__IntList {
  long i;
  struct G__IntList *prev;
  struct G__IntList *next;
};
#endif

struct G__Definedtemplatememfunc {
  int line;
  int filenum;
  FILE *def_fp;
  fpos_t def_pos;
  struct G__Definedtemplatememfunc *next;
};
#endif

struct G__Templatearg {
  int type;
  char *string;
  char *default_parameter;
  struct G__Templatearg *next;
};

struct G__Definedtemplateclass {
  char *name;
  int hash;
  int line;
  int filenum;
  FILE *def_fp;
  fpos_t def_pos;
  struct G__Templatearg *def_para;
#ifdef G__TEMPLATEMEMFUNC
  struct G__Definedtemplatememfunc memfunctmplt;
#endif
  struct G__Definedtemplateclass *next;
#ifndef G__OLDIMPLEMENTATION682
  int parent_tagnum;
#endif
#ifndef G__OLDIMPLEMENTATION691
  struct G__IntList *instantiatedtagnum;
  int isforwarddecl;
#endif
#ifndef G__OLDIMPLEMENTATION972
  int friendtagnum;
#endif
};


#ifdef G__TEMPLATEFUNC

struct G__Templatefuncarg {
  int paran;
  char type[G__MAXFUNCPARA];
  int tagnum[G__MAXFUNCPARA];
  int typenum[G__MAXFUNCPARA];
  int reftype[G__MAXFUNCPARA];
  char paradefault[G__MAXFUNCPARA];
  int argtmplt[G__MAXFUNCPARA];
#ifndef G__OLDIMPLEMENTATION727
  int *ntarg[G__MAXFUNCPARA];
  int nt[G__MAXFUNCPARA];
#endif
#ifndef G__OLDIMPLEMENTATION750
  char **ntargc[G__MAXFUNCPARA];
#endif
};

struct G__Definetemplatefunc {
  char *name;
  int hash;
  struct G__Templatearg *def_para;
  struct G__Templatefuncarg func_para;  /* need to refine here */
  int line;
  int filenum;
  FILE *def_fp;
  fpos_t def_pos;
  struct G__Definetemplatefunc *next;
#ifndef G__OLDIMPLEMENTATION687
  int parent_tagnum;
#endif
#ifndef G__OLDIMPLEMENTATION972
  int friendtagnum;
#endif
};

#endif /* G__TEMPLATEFUNC */

#endif /* G__TEMPLATECLASS */


/**************************************************************************
* Macro statement support
**************************************************************************/
/* print out warning for macro statement and function form macro */
#define G__WARNPREP
#define G__MACROSTATEMENT
#define G__FUNCMACRO

struct G__Charlist {
  char *string;
  struct G__Charlist *next;
};

struct G__Callfuncmacro{
  FILE *call_fp;
  fpos_t call_pos;
  int line;
  fpos_t mfp_pos;
  struct G__Callfuncmacro *next;
#ifndef G__OLDIMPLEMENTATION1179
  short call_filenum;
#endif
} ;

struct G__Deffuncmacro {
  char *name;
  int hash;
  int line;
  FILE *def_fp;
  fpos_t def_pos;
  struct G__Charlist def_para;
  struct G__Callfuncmacro callfuncmacro;
  struct G__Deffuncmacro *next;
#ifndef G__OLDIMPLEMENTATION1179
  short def_filenum;
#endif
} ;


/**************************************************************************
* Text processing capability
*
*   fp=fopen("xxx","r");
*   while($read(fp)) {
*      printf("%d %s %s\n",$#,$1,$2);
*   }
**************************************************************************/
#define G__TEXTPROCESSING


#define G__RSVD_LINE        -1
#define G__RSVD_FILE        -2
#define G__RSVD_ARG         -3
#define G__RSVD_DATE        -4
#define G__RSVD_TIME        -5

/**************************************************************************
* preprocessed file keystring list
**************************************************************************/
struct G__Preprocessfilekey {
  char *keystring;
  struct G__Preprocessfilekey *next;
};


/**************************************************************************
* allocation of array by new operator ?
**************************************************************************/
struct G__newarylist {
  long point;
  int pinc;
  struct G__newarylist *next;
};


/**************************************************************************
* integration of G__atpause() function
**************************************************************************/
#define G__ATPAUSE


/**************************************************************************
* struct for storing base class constructor
**************************************************************************/
struct G__baseparam {
	char *name[G__MAXBASE];
	char *param[G__MAXBASE];
};


/**************************************************************************
* flag argument to G__getfunction()
**************************************************************************/
#define G__TRYNORMAL         0
#define G__CALLMEMFUNC       1
#define G__TRYMEMFUNC        2
#define G__CALLCONSTRUCTOR   3  
#define G__TRYCONSTRUCTOR    4  
#define G__TRYDESTRUCTOR     5  
#define G__CALLSTATICMEMFUNC 6
#define G__TRYUNARYOPR       7
#define G__TRYBINARYOPR      8

#ifndef G__OLDIMPLEMENTATINO1250
#define G__TRYIMPLICITCONSTRUCTOR 7
#endif


/********************************************************************
* include path by -I option
* Used in G__main() and G__loadfile()
********************************************************************/
struct G__includepath {
  char *pathname;
  struct G__includepath *next;
};

/*********************************************************************
* Sharedlibrary table
*********************************************************************/
#ifdef G__SHAREDLIB

#define G__AUTOCOMPILE

#endif /* G__SHAREDLIB */

#define G__MAX_SL 150

/*********************************************************************
* debugging flag
*********************************************************************/
#define G__TESTBREAK     0x30
#define G__BREAK         0x10
#define G__NOBREAK       0xef
#define G__CONTUNTIL     0x20
#define G__NOCONTUNTIL   0xdf
#define G__TRACED        0x01
#define G__NOTRACED       0xfe

/*********************************************************************
* const string list
*********************************************************************/
struct G__ConstStringList {
  char *string;
  int hash;
  struct G__ConstStringList *prev;
};

/*********************************************************************
* scratch upto dictionary position
*********************************************************************/
struct G__dictposition {
  /* global variable table position */
  struct G__var_array *var;
  int ig15;
  /* struct tagnum */
  int tagnum;
  /* const string table */
  struct G__ConstStringList *conststringpos;
  /* typedef table */
  int typenum;
  /* global function table position */
  struct G__ifunc_table *ifunc;
  int ifn;
  /* include path */
  struct G__includepath *ipath;
  /* shared library file */
  int allsl;
  /* preprocessfilekey */
  struct G__Preprocessfilekey *preprocessfilekey;
  /* input file */
  int nfile;
  /* macro table */
  struct G__Deffuncmacro *deffuncmacro;
  /* template class */
  struct G__Definedtemplateclass *definedtemplateclass;
  /* function template */
  struct G__Definetemplatefunc *definedtemplatefunc;   
};

#ifdef G__SECURITY
#ifdef G__64BIT
typedef unsigned int G__UINT32 ;
#else
typedef unsigned long G__UINT32 ;
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1207
typedef void (*G__DLLINIT)();
#endif

struct G__filetable {
  FILE *fp;
  int hash;
  char *filename;
  char *prepname;
  char  *breakpoint;
  int maxline;
  struct G__dictposition *dictpos;
  G__UINT32 security;
#ifndef G__OLDIMPLEMENTATION952
  int included_from; /* filenum of the file which first include this one */
#endif
#ifndef G__OLDIMPLEMENTATION1207
  int ispermanentsl;
  G__DLLINIT initsl;
#endif
#ifndef G__OLDIMPLEMENTATION1273
  struct G__dictposition *hasonlyfunc;
#endif
};

/**************************************************************************
* user specified pragma statement
**************************************************************************/
#ifndef G__OLDIMPLEMENTATION451
struct G__AppPragma {
  char *name;
  void *p2f;
  struct G__AppPragma *next;
};
#endif

/**************************************************************************
* Flag to check global operator new/delete()
**************************************************************************/
#define G__IS_OPERATOR_NEW        0x01
#define G__IS_OPERATOR_DELETE     0x02
#define G__MASK_OPERATOR_NEW      0x04
#define G__MASK_OPERATOR_DELETE   0x08
#define G__NOT_USING_2ARG_NEW     0x10
#define G__DUMMYARG_NEWDELETE        0x100
#define G__DUMMYARG_NEWDELETE_STATIC 0x200

/**************************************************************************
* Stub function mode
**************************************************************************/
#define G__SPECIFYLINK 1
#define G__SPECIFYSTUB 2


/**********************************************************************
* Multi-byte character handling in comment and string
**********************************************************************/
#ifdef G__MULTIBYTE
#define G__UNKNOWNCODING 0
#define G__EUC           1
#define G__SJIS          2
#define G__JIS           3
#define G__ONEBYTE       4

/* checking both EUC and S-JIS by flag */
#define G__IsDBCSLeadByte(c) ((0x80&c)&&G__EUC!=G__lang&&G__CodingSystem(c)) 

/* Checking multi-byte coding system by 2nd byte, 
 * MSB of 2nd byte may be 0 in S-JIS */
#define G__CheckDBCS2ndByte(c) if(0==(0x80&c)) G__lang=G__SJIS
#endif

/**********************************************************************
* hash token for some symbol
**********************************************************************/
#define G__HASH_MAIN      421
#define G__HASH_OPERATOR  876

/*********************************************************************
* return status flag
*********************************************************************/
#ifndef G__OLDIMPLEMENTATION754
#define G__RETURN_NON       0
#define G__RETURN_NORMAL    1 
#define G__RETURN_IMMEDIATE 2
#define G__RETURN_TRY       3 
#define G__RETURN_EXIT1     4
#define G__RETURN_EXIT2     5
#else
#define G__RETURN_NON       0
#define G__RETURN_NORMAL    1 
#define G__RETURN_IMMEDIATE 2
#define G__RETURN_EXIT1     3
#define G__RETURN_EXIT2     4
#endif

/*********************************************************************
* cint parser function and global variable prototypes
*********************************************************************/
#include "security.h"
#include "fproto.h"
#include "global.h"

#endif /* G__COMMON_H */

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

