/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file common.h
 ************************************************************************
 * Description:
 * Common header file for cint parser.
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
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
* Note, Warning message display flag
**************************************************************************/
#define G__DISPNONE  0
#define G__DISPERR   1
#define G__DISPWARN  2
#define G__DISPNOTE  3
#define G__DISPALL   4
#define G__DISPSTRICT 5
#define G__DISPROOTSTRICT 5

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

#ifdef G__OLDIMPLEMENTATION1921
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

#define G__OPR_ADDVOIDPTR   17

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

#ifndef G__OLDIMPLEMENTATION1491
#define G__OPR_ADD_UU        0xa00
#define G__OPR_SUB_UU        0xa01
#define G__OPR_MUL_UU        0xa02
#define G__OPR_DIV_UU        0xa03
#define G__OPR_LT_UU         0xa04
#define G__OPR_GT_UU         0xa05
#define G__OPR_LE_UU         0xa06
#define G__OPR_GE_UU         0xa07
#define G__OPR_EQ_UU         0xa08
#define G__OPR_NE_UU         0xa09
#define G__OPR_ADDASSIGN_UU  0xa0a
#define G__OPR_SUBASSIGN_UU  0xa0b
#define G__OPR_MULASSIGN_UU  0xa0c
#define G__OPR_DIVASSIGN_UU  0xa0d
#endif

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

#ifndef G__OLDIMPLEMENTATION1967
#define G__PARAREF         100
#define G__PARAREFP2P      102
#define G__PARAREFP2P2P    103

#define G__PLVL(x)        (x%10)
#define G__REF(x)         ((x/100)*100)
#endif

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
#define G__STATICCONST 0x10

/**************************************************************************
* Class charasteristics
**************************************************************************/
#define G__HAS_DEFAULTCONSTRUCTOR  0x01
#define G__HAS_COPYCONSTRUCTOR     0x02
#define G__HAS_CONSTRUCTOR         0x03
#define G__HAS_XCONSTRUCTOR        0x80
#define G__HAS_DESTRUCTOR          0x04
#define G__HAS_ASSIGNMENTOPERATOR  0x08
#define G__HAS_OPERATORNEW1ARG     0x10
#define G__HAS_OPERATORNEW2ARG     0x20
#define G__HAS_OPERATORNEW         0x30
#define G__HAS_OPERATORDELETE      0x40

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
#define G__LDST_VAR_P         (long)0x7fff0000
#define G__LDST_LVAR_P        (long)0x7fff0001
#define G__LDST_MSTR_P        (long)0x7fff0002
#define G__LDST_VAR_INDEX     (long)0x7fff0003
#define G__LDST_VAR_INDEX_OPR (long)0x7fff0004
#define G__OP2_OPTIMIZED      (long)0x7fff0005
#define G__OP1_OPTIMIZED      (long)0x7fff0006
#define G__LD                 (long)0x7fff0007
#define G__CL                 (long)0x7fff0008
#define G__OP2                (long)0x7fff0009
#define G__CMPJMP             (long)0x7fff000a
#define G__INCJMP             (long)0x7fff000b
#define G__CNDJMP             (long)0x7fff000c
#define G__JMP                (long)0x7fff000d
#define G__POP                (long)0x7fff000e
#define G__LD_FUNC            (long)0x7fff000f
#define G__RETURN             (long)0x7fff0010
#define G__CAST               (long)0x7fff0011
#define G__OP1                (long)0x7fff0012
#define G__LETVVAL            (long)0x7fff0013
#define G__ADDSTROS           (long)0x7fff0014
#define G__LETPVAL            (long)0x7fff0015
#define G__TOPNTR	      (long)0x7fff0016
#define G__NOT                (long)0x7fff0017
#define G__BOOL               (long)0x7fff0018
#define G__ISDEFAULTPARA      (long)0x7fff0019

#define G__LD_VAR             (long)0x7fff001a
#define G__ST_VAR             (long)0x7fff001b
#define G__LD_MSTR            (long)0x7fff001c
#define G__ST_MSTR            (long)0x7fff001d
#define G__LD_LVAR            (long)0x7fff001e
#define G__ST_LVAR            (long)0x7fff001f
#define G__CMP2               (long)0x7fff0020
#define G__PUSHSTROS          (long)0x7fff0021
#define G__SETSTROS           (long)0x7fff0022
#define G__POPSTROS           (long)0x7fff0023
#define G__SETTEMP            (long)0x7fff0024
#define G__FREETEMP           (long)0x7fff0025
#define G__GETRSVD            (long)0x7fff0026
#define G__REWINDSTACK        (long)0x7fff0027
#define G__CND1JMP            (long)0x7fff0028
#define G__LD_IFUNC	      (long)0x7fff0029
#define G__NEWALLOC           (long)0x7fff002a
#define G__SET_NEWALLOC       (long)0x7fff002b
#define G__DELETEFREE         (long)0x7fff002c
#define G__SWAP               (long)0x7fff002d
#define G__BASECONV           (long)0x7fff002e
#define G__STORETEMP          (long)0x7fff002f
#define G__ALLOCTEMP          (long)0x7fff0030
#define G__POPTEMP            (long)0x7fff0031
#define G__REORDER            (long)0x7fff0032
#define G__LD_THIS            (long)0x7fff0033
#define G__RTN_FUNC           (long)0x7fff0034
#define G__SETMEMFUNCENV      (long)0x7fff0035
#define G__RECMEMFUNCENV      (long)0x7fff0036
#define G__ADDALLOCTABLE      (long)0x7fff0037
#define G__DELALLOCTABLE      (long)0x7fff0038
/* #define G__BASECONSTRUCT     (long)0x7fff00XX */
#define G__BASEDESTRUCT       (long)0x7fff0039
#define G__REDECL             (long)0x7fff003a
#define G__TOVALUE            (long)0x7fff003b
#define G__INIT_REF           (long)0x7fff003c
#define G__PUSHCPY            (long)0x7fff003d
#define G__LETNEWVAL          (long)0x7fff003e
#define G__SETGVP             (long)0x7fff003f
#define G__TOPVALUE           (long)0x7fff0040
#define G__CTOR_SETGVP        (long)0x7fff0041

#define G__TRY                (long)0x7fff0042
#define G__TYPEMATCH          (long)0x7fff0043
#define G__ALLOCEXCEPTION     (long)0x7fff0044
#define G__DESTROYEXCEPTION   (long)0x7fff0045
#define G__THROW              (long)0x7fff0046
#define G__CATCH              (long)0x7fff0047 /* never used */
#define G__SETARYINDEX        (long)0x7fff0048
#define G__RESETARYINDEX      (long)0x7fff0049
#define G__GETARYINDEX        (long)0x7fff004a

#define G__ENTERSCOPE         (long)0x7fff004b
#define G__EXITSCOPE          (long)0x7fff004c
#define G__PUTAUTOOBJ         (long)0x7fff004d
#define G__PUTHEAPOBJ         (long)0x7fff004e /* not implemented yet */
#define G__CASE               (long)0x7fff004f
/* #define G__SETARYCTOR         (long)0x7fff0050 */
#define G__MEMCPY             (long)0x7fff0050
#define G__MEMSETINT          (long)0x7fff0051
#define G__JMPIFVIRTUALOBJ    (long)0x7fff0052
#define G__VIRTUALADDSTROS    (long)0x7fff0053

#define G__PAUSE              (long)0x7fff00fe

#define G__NOP                (long)0x7fff00ff

#ifdef G__NEVER

#define G__INSTMASK       (long)0x7fff000000ff
#define G__LINENUMMASK    (long)0x7fffffffff00
/* #define G__INST(x) (x&G__INSTMASK) */ /* not ready yet */
#define G__LINE(x) ((x&G__LINEMASK)/0x100)

#else

#define G__INST(x)  x

#endif

/********************************************
* G__TRY G__bc_exec_try_bytecode return value
********************************************/
#define G__TRY_NORMAL                 1
#define G__TRY_INTERPRETED_EXCEPTION  2
#define G__TRY_COMPILED_EXCEPTION     3
#define G__TRY_UNCAUGHT               9

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

#ifndef G__OLDIMPLEMENTATION2132
/*********************************************
* G__CL  line+filenum offset
*********************************************/
#define G__CL_LINEMASK  0x000fffff
#define G__CL_FILEMASK  0x00000fff
#define G__CL_FILESHIFT 0x00100000
#endif


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

#ifndef G__OLDIMPLEMENTATION1587
#define G__TMPLT_POINTERARG1   0x10000
#define G__TMPLT_POINTERARG2   0x20000
#define G__TMPLT_POINTERARG3   0x30000
#define G__TMPLT_POINTERARGMASK 0xffff0000
#else
#define G__TMPLT_POINTERARG1   1
#define G__TMPLT_POINTERARG2   2
#define G__TMPLT_POINTERARG3   3
#endif

#define G__TMPLT_CONSTARG      0x100
#define G__TMPLT_REFERENCEARG  0x200

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
#ifndef G__OLDIMPLEMENTATION1587
  struct G__Definedtemplateclass *specialization;
  struct G__Templatearg *spec_arg;
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
#ifndef G__OLDIMPLEMENTATION1870
  char *name;
  char *param;
  struct G__baseparam *next;
#else
  char *name[G__MAXBASE];
  char *param[G__MAXBASE];
#endif
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
#define G__NOTRACED      0xfe

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

#ifndef G__OLDIMPLEMENTATION2014
  char* ptype; /* struct,union,enum,class */
#endif
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

#ifndef G__OLDIMPLEMENTATION1536
#define G__NONCINTHDR   0x01
#define G__CINTHDR      0x10
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
#ifndef G__OLDIMPLEMENTATION1536
  char hdrprop;
#endif
#ifndef G__OLDIMPLEMENTATION1649
  char *str;
  int vindex;
#endif
#ifndef G__OLDIMPLEMENTATION1756
  int parent_tagnum;
#endif
#ifndef G__OLDIMPLEMENTATION1908
  int slindex;
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
#define G__ONEBYTE       4 /* ISO-8859-x */

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
#ifndef G__OLDIMPLEMENTATION1844
#define G__RETURN_TRY      -1 
#else
#define G__RETURN_TRY       3 
#endif
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
* G__isanybase, G__ispublicbase static resolution
*********************************************************************/
#ifndef G__OLDIMPLEMENTATION1719
#define G__STATICRESOLUTION   2  /* for G__isanybase */
#define G__STATICRESOLUTION2  2  /* for G__ispublicbase */
#else
#define G__STATICRESOLUTION   0
#define G__STATICRESOLUTION2  0
#endif

/*********************************************************************
* x
*********************************************************************/
#define G__NAMEDMACROEXT  "NM"
#define G__NAMEDMACROEXT2 "_cintNM"


/*********************************************************************
* G__rootmode and G__ReadInputMode() in pause.c
*********************************************************************/
#define G__INPUTCXXMODE  3
#define G__INPUTROOTMODE 1
#define G__INPUTCINTMODE 0

/***********************************************************************
* for function overloading
**********************************************************************/
struct G__funclist {
  struct G__ifunc_table *ifunc;
  int ifn;
  unsigned int rate;
  unsigned int p_rate[G__MAXFUNCPARA];
  struct G__funclist *prev;
};

#ifndef G__OLDIMPLEMENTATION1782
/*********************************************************************
* variable length string buffer
*********************************************************************/
/* #define G__BUFLEN 34 */
#define G__BUFLEN 80
#endif

#ifndef G__OLDIMPLEMENTATION1836
/*********************************************************************
* variable length string buffer
*********************************************************************/
#define G__LONGLONG    1
#define G__ULONGLONG   2
#define G__LONGDOUBLE  3
#endif

/*********************************************************************
* cintv6, flags
*********************************************************************/
/* G__cintv6 flags */
/* #define G__CINT_VER6 1 */ /* defined in platform configuration */
#define G__BC_CINTVER6     0x01
#define G__BC_COMPILEERROR 0x02
#define G__BC_RUNTIMEERROR 0x04

#define G__BC_DEBUG        0x08

/*********************************************************************
* debug interface
*********************************************************************/
#ifndef G__OLDIMPLEMENTATION2137
struct G__store_env {
  struct G__var_array *var_local;
  long struct_offset;
  int tagnum;
  int exec_memberfunc;
};

struct G__view {
  struct G__input_file file;
  struct G__var_array *var_local;
  long struct_offset;
  int tagnum;
  int exec_memberfunc;
#ifndef G__OLDIMPLEMENTATION2159
  long localmem;
#endif
};
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

