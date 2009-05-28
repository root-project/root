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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__COMMON_H
#define G__COMMON_H

#ifndef __CINT__
#include "Reflex/Type.h"
#include "Reflex/Member.h"

#else
namespace Reflex {
   class Type;
   class Scope;
   class Type_Iterator;
} // namespace Reflex
#endif

#include "vector"

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
#include "G__ci.h"


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

#ifdef __cplusplus
#include <list>
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


/**************************************************************************
* G__reftype, var->reftype[], ifunc->reftype[] flag
**************************************************************************/
#define G__PLVL(x)        (x%10)
#define G__REF(x)         ((x/100)*100)

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
//#ifndef G__ASM_DBG
//#define G__ASM_DBG
//#endif

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
#define G__LD_IFUNC	         (long)0x7fff0029
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

#define G__ROOTOBJALLOCBEGIN  (long)0x7fff0054
#define G__ROOTOBJALLOCEND    (long)0x7fff0055

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

/*********************************************
* loop compiler
*  limit numbers
*********************************************/
#define G__MAXINST     0x1000
#define G__MAXSTACK    0x100
#define G__MAXSTRSTACK  0x10

/*********************************************
* G__CL  line+filenum offset
*********************************************/
#define G__CL_LINEMASK  0x000fffff
#define G__CL_FILEMASK  0x00000fff
#define G__CL_FILESHIFT 0x00100000


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

#define G__RECOVER_ASMENV                          \
    G__asm_exec=store_asm_exec;                    \
    G__asm_loopcompile=G__asm_loopcompile_mode


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
* class inheritance
**************************************************************************/
#define G__ISDIRECTINHERIT         0x0001
#define G__ISVIRTUALBASE           0x0002
#define G__ISINDIRECTVIRTUALBASE   0x0004


/**************************************************************************
* class template 
**************************************************************************/
#define G__TEMPLATECLASS
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

#define G__TMPLT_POINTERARG1   0x10000
#define G__TMPLT_POINTERARG2   0x20000
#define G__TMPLT_POINTERARG3   0x30000
#define G__TMPLT_POINTERARGMASK 0xffff0000

#define G__TMPLT_CONSTARG      0x100
#define G__TMPLT_REFERENCEARG  0x200

#ifdef G__TEMPLATEFUNC

#endif /* G__TEMPLATEFUNC */

struct G__Templatefuncarg {
  int paran;
  char type[G__MAXFUNCPARA];
  int tagnum[G__MAXFUNCPARA];
  ::Reflex::Type typenum[G__MAXFUNCPARA];
  int reftype[G__MAXFUNCPARA];
  char paradefault[G__MAXFUNCPARA];
  int argtmplt[G__MAXFUNCPARA];
  int *ntarg[G__MAXFUNCPARA];
  int nt[G__MAXFUNCPARA];
  char **ntargc[G__MAXFUNCPARA];
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
  int parent_tagnum;
  int friendtagnum;
};

#endif /* G__TEMPLATECLASS */

/**************************************************************************
* Macro statement support
**************************************************************************/
/* print out warning for macro statement and function form macro */
#define G__WARNPREP
#define G__MACROSTATEMENT
#define G__FUNCMACRO

/**************************************************************************
* Text processing capability
*
*   fp=fopen("xxx","r");
*   while($read(fp)) {
*      printf("%d %s %s\n",$#,$1,$2);
*   }
**************************************************************************/
#define G__TEXTPROCESSING


/**************************************************************************
* integration of G__atpause() function
**************************************************************************/
#define G__ATPAUSE


/*********************************************************************
* Sharedlibrary table
*********************************************************************/
#ifdef G__SHAREDLIB

#define G__AUTOCOMPILE

#endif /* G__SHAREDLIB */

#define G__MAX_SL 1024

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
* G__isanybase, G__ispublicbase static resolution
*********************************************************************/
#define G__STATICRESOLUTION   2  /* for G__isanybase */
#define G__STATICRESOLUTION2  2  /* for G__ispublicbase */

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

#define G__RSVD_LINE        -1
#define G__RSVD_FILE        -2
#define G__RSVD_ARG         -3
#define G__RSVD_DATE        -4
#define G__RSVD_TIME        -5

/*********************************************************************
* variable length string buffer
*********************************************************************/
/* #define G__BUFLEN 34 */
#define G__BUFLEN 180

/*********************************************************************
* OBSOLETE, cint bytecode machine flags
*********************************************************************/
#define G__BC_CINTVER6     0x01
#define G__BC_COMPILEERROR 0x02
#define G__BC_RUNTIMEERROR 0x04
#define G__BC_DEBUG        0x08

/**************************************************************************
* ROOT special requirement
*
**************************************************************************/
#define G__NOSTREAMER      0x01
#define G__NOINPUTOPERATOR 0x02
#define G__USEBYTECOUNT   0x04



/**************************************************************************
*
* STRUCT DEFINITIONS
*
**************************************************************************/

struct G__Charlist {
   G__Charlist() : string(0), next(0) {}
   char *string;
   struct G__Charlist *next;
};

struct G__Callfuncmacro{
   G__Callfuncmacro() : call_fp(0),line(-1),next(0),call_filenum(-1) {}
   
   FILE *call_fp;
   fpos_t call_pos;
   int line;
   fpos_t mfp_pos;
   struct G__Callfuncmacro *next;
   short call_filenum;
} ;

struct G__Deffuncmacro {
   G__Deffuncmacro() : name(0), hash(0), line(-1), def_fp(0), next(0), def_filenum(-1) {}
   
   char *name;
   int hash;
   int line;
   FILE *def_fp;
   fpos_t def_pos;
   G__Charlist def_para;
   struct G__Callfuncmacro callfuncmacro;
   struct G__Deffuncmacro *next;
   short def_filenum;
} ;

extern "C" {

struct G__breakcontinue_list {
   struct G__breakcontinue_list* next; // next entry in list
   int isbreak; // is it a break or a continue
   int idx; // index into bytecode array to patch
};

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
* struct for storing base class constructor
**************************************************************************/
struct G__baseparam {
  char *name;
  char *param;
  struct G__baseparam *next;
};

/*********************************************************************
* const string list
*********************************************************************/
struct G__ConstStringList {
  char *string;
  int hash;
  struct G__ConstStringList *prev;
};

typedef void (*G__DLLINIT)();

#define G__NONCINTHDR   0x01
#define G__CINTHDR      0x10

#ifdef __cplusplus
struct G__filetable {
  FILE *fp;
  int hash;
  char *filename;
  char *prepname;
  char  *breakpoint;
  int maxline;
  struct G__dictposition *dictpos;
  G__UINT32 security;
  int included_from; /* filenum of the file which first include this one */
  int ispermanentsl;
  std::list<G__DLLINIT>* initsl;
  struct G__dictposition *hasonlyfunc;
  int definedStruct; /* number of struct/class/namespace defined in this file */
  char hdrprop;
#ifndef G__OLDIMPLEMENTATION1649
  char *str;
  int vindex;
#endif
  int parent_tagnum;
  int slindex;
};
#endif /* __cplusplus */

/**************************************************************************
* user specified pragma statement
**************************************************************************/
struct G__AppPragma {
  char *name;
  void *p2f;
  struct G__AppPragma *next;
};

/***********************************************************************
* for function overloading
**********************************************************************/
struct G__funclist {
  struct G__funclist* next;
  ::Reflex::Member ifunc;
  int ifn;
  unsigned int rate;
  unsigned int p_rate[G__MAXFUNCPARA];
};

/*********************************************************************
* debug interface
*********************************************************************/
struct G__store_env {
  ::Reflex::Scope var_local;
  char* struct_offset;
  ::Reflex::Scope tagnum;
  int exec_memberfunc;
};

struct G__view {
  struct G__input_file file;
  ::Reflex::Scope var_local;
  char* struct_offset;
  ::Reflex::Scope tagnum;
  int exec_memberfunc;
  char* localmem;
};

/**************************************************************************
* comment information
*
**************************************************************************/
namespace Cint { namespace Internal { extern int G__nfile; } } 
struct G__comment_info {
   G__comment_info()  {
      filenum = -1;
      p.com = 0;
   }
   union {
      char  *com;
      fpos_t pos;
   } p;
   int   filenum;
};

/**************************************************************************
* ROOT special requirement
*
**************************************************************************/
struct G__RootSpecial {
  char* deffile;
  int defline;
  char* impfile;
  int impline;
  int version;
  unsigned int instancecount;
  unsigned int heapinstancecount;
  void* defaultconstructor;
};

#ifdef G__ASM_WHOLEFUNC
/**************************************************************************
* bytecode compiled interpreted function
*
**************************************************************************/
struct G__bytecodefunc {
  /*struct G__ifunc_table *ifunc;
  int ifn;*/
  Reflex::Member ifunc;
  Reflex::Scope  frame;
  int varsize;
  G__value *pstack; /* upper part of stack to store numerical constants */
  int stacksize;
  long *pinst;      /* instruction buffer */
  int instsize;
  char *asm_name;   /* name of used ANSI library function */
};
#endif

/**************************************************************************
 *                                                      1
 *                               2            2          
 *                        proto   interpreted   bytecode  compiled
 * fpos_t pos;            hdrpos  src_fpos      src_fpos  ??
 * void* p;            2  NULL    src_fp        src_fp    ifmethod
 * int line_number;       -1      line          line      -1
 * short filenum;      1  fnum    fnum          fnum      fnum <<<change
 * ushort size;           0       size          size      -1   <<<change
 * void*  tp2f;           fname   fname         bytecode  (*p2f)|ifmethod
 * bcf* bytecode;      2  NULL    NULL          bytecode  NULL
 * int bytecodestatus;    NOTYET  NOTYET|FAIL   SUCCESS   ??
 **************************************************************************/

class G__funcentry
{
public: // -- Data Members
   fpos_t pos; // if source file, file position of function definition.
   void* p; // if source file, FILE*
            // if compiled function, int (*)()
            // if prototype, 0
   int line_number; // -1 if no function body or compiled function
   short filenum; // -1 if compiled function, otherwise interpreted func
   long ptradjust; // Object Pointer Adjustement to call the stub function (entry)
#ifdef G__ASM_FUNC
   int size; // size (number of lines) of function
#endif // G__ASM_FUNC
#ifdef G__TRUEP2F
   std::string for_tp2f; // FIXME: INTERFACE CHANGE
   void* tp2f;
#endif // G__TRUEP2F
#ifdef G__ASM_WHOLEFUNC
   G__bytecodefunc* bytecode;
   int bytecodestatus;
#endif // G__ASM_WHOLEFUNC

   //
   //  This part is from G__ifunc_table_internal.
   //

   // ANSI or standard header format
   // 0 indicates K&R
   // 1 indicates ansi
   // 2 indicates the presence of elipsis (...)
   char ansi;
   short busy; // if function is called, busy[] is incremented
#ifdef G__FRIEND
   G__friendtag* friendtag;
#endif // G__FRIEND
   void* userparam; // user parameter array
   short vtblindex;
   short vtblbasetagnum;
   std::vector<G__value*> para_default;
public: // -- Member Functions.
   G__funcentry()
   : p(0)
   , line_number(-1)
   , filenum(-1)
   , ptradjust(0)
#ifdef G__ASM_FUNC
   , size(0)
#endif // G__ASM_FUNC
#ifdef G__TRUEP2F
   , tp2f(0)
#endif // G__TRUEP2F
#ifdef G__ASM_WHOLEFUNC
   , bytecode(0)
   , bytecodestatus(G__BYTECODE_NOTYET)
#endif // G__ASM_WHOLEFUNC
   , ansi(1)
   , busy(0)
#ifdef G__FRIEND
   , friendtag(0)
#endif // G__FRIEND
   , userparam(0)
   , vtblindex(-1)
   , vtblbasetagnum(0)
   , para_default(0)
   {
   }
   G__funcentry& copy(const G__funcentry& orig); // This function must NOT be made virtual
   void clear();
   G__funcentry(const G__funcentry& orig)
   {
      copy(orig);
   }
   G__funcentry& operator=(const G__funcentry& orig)
   {
      clear();
      return copy(orig);
   }
   ~G__funcentry();

};

/**************************************************************************
* structure for class inheritance
*
**************************************************************************/
struct G__inheritance {
   struct G__Entry {
      G__Entry(short tag = 0, char* off = 0, G__SIGNEDCHAR_T acc = G__PUBLIC, char prop = 0):
         basetagnum(tag), baseoffset(off), baseaccess(acc), property(prop) {}
      short basetagnum;
      char* baseoffset;
      G__SIGNEDCHAR_T baseaccess;
      char property;
   };
   std::vector<G__Entry> vec;
};

/**************************************************************************
* structure struct,union tag information
*
**************************************************************************/

#ifdef __cplusplus

namespace Cint { namespace Internal { extern int G__global1_init; } }

struct G__tagtable {
  static int inited;
  G__tagtable(); // Implemented in global1.cxx for now.

  /* tag entry information */
  char type[G__MAXSTRUCT]; /* struct,union,enum,class */

  char *name[G__MAXSTRUCT];
  int  hash[G__MAXSTRUCT];
  int  size[G__MAXSTRUCT];
  /* member information */

  struct G__inheritance *baseclass[G__MAXSTRUCT];
  char* virtual_offset[G__MAXSTRUCT];
  G__SIGNEDCHAR_T globalcomp[G__MAXSTRUCT];
  G__SIGNEDCHAR_T iscpplink[G__MAXSTRUCT];
  char isabstract[G__MAXSTRUCT];
  char protectedaccess[G__MAXSTRUCT];

  int  line_number[G__MAXSTRUCT];
  short filenum[G__MAXSTRUCT];

  short parent_tagnum[G__MAXSTRUCT];
  unsigned char funcs[G__MAXSTRUCT];
  char istypedefed[G__MAXSTRUCT];
  char istrace[G__MAXSTRUCT];
  char isbreak[G__MAXSTRUCT];
  int  alltag;
  int  nactives; // List of active (i.e. non disable, non autoload entries).

#ifdef G__FRIEND
  struct G__friendtag *friendtag[G__MAXSTRUCT];
#endif

  std::list<G__incsetup> *incsetup_memvar[G__MAXSTRUCT];
  std::list<G__incsetup> *incsetup_memfunc[G__MAXSTRUCT];

  char rootflag[G__MAXSTRUCT];
  struct G__RootSpecial *rootspecial[G__MAXSTRUCT];

  char isctor[G__MAXSTRUCT];

#ifndef G__OLDIMPLEMENTATION1503
  ::Reflex::Type defaulttypenum[G__MAXSTRUCT];
#endif
  void* userparam[G__MAXSTRUCT];     /* user parameter array */
  char* libname[G__MAXSTRUCT];
  void* vtable[G__MAXSTRUCT];
  /* short vtabledepth[G__MAXSTRUCT]; */
};
#else
struct G__tagtable;
#endif

/**************************************************************************
* tempobject list
*
**************************************************************************/
#ifdef G__CLINK
struct G__tempobject_list {
  G__value obj;
  int  level;
#ifdef G__CPPLINK3
  int cpplink;
#endif
  int no_exec;
  struct G__tempobject_list *prev;
};
#endif

} /* extern "C" */

/*********************************************************************
* Reflex Merge Utility Funcs
*********************************************************************/
class G__RflxStackProperties {
public:
   G__RflxStackProperties() : prev_filenum(-1), prev_line_number(-1), struct_offset(0), exec_memberfunc(0)
#ifdef G__VAARG
   , libp(0)
#endif // G__VAARG
   {
   }
   G__RflxStackProperties(const G__RflxStackProperties& rhs) : prev_filenum(rhs.prev_filenum), prev_line_number(rhs.prev_line_number), struct_offset(rhs.struct_offset), tagnum(rhs.tagnum), exec_memberfunc(rhs.exec_memberfunc), ifunc(rhs.ifunc), calling_scope(rhs.calling_scope)
#ifdef G__VAARG
   , libp(rhs.libp)
#endif // G__VAARG
   {
   }
   ~G__RflxStackProperties();
   G__RflxStackProperties& operator=(const G__RflxStackProperties& rhs)
   {
      if (this != &rhs) {
         prev_filenum = rhs.prev_filenum;
         prev_line_number = rhs.prev_line_number;
         struct_offset = rhs.struct_offset;
         tagnum = rhs.tagnum;
         exec_memberfunc = rhs.exec_memberfunc;
         ifunc = rhs.ifunc;
         calling_scope = rhs.calling_scope;
#ifdef G__VAARG
         libp = rhs.libp;
#endif // G__VAARG
         // --
      }
      return *this;
   }
public:
   int prev_filenum; // File number of the calling pointing that induced this stack level.
   short prev_line_number;
   char* struct_offset;
   ::Reflex::Scope tagnum; 
   int exec_memberfunc;
   ::Reflex::Member ifunc;
   ::Reflex::Scope calling_scope; // aka prev_local
#ifdef G__VAARG
   struct G__param* libp; // store the values of function parameters as passed
#endif // G__VAARG
   // --
}; 

class G__RflxProperties {
public:
   G__RflxProperties() : autoload(0), filenum(-1), linenum(-1), globalcomp(G__NOLINK), iscpplink(G__NOLINK), typenum(-1), tagnum(-1), isFromUsing(false), vtable(0), isBytecodeArena(0), statictype(0) {}
   G__RflxProperties(const G__RflxProperties& rhs) : autoload(rhs.autoload), filenum(rhs.filenum), linenum(rhs.linenum), globalcomp(rhs.globalcomp), iscpplink(rhs.iscpplink), typenum(rhs.typenum), tagnum(rhs.tagnum), isFromUsing(rhs.isFromUsing), comment(rhs.comment), stackinfo(rhs.stackinfo), vtable(rhs.vtable), isBytecodeArena(rhs.isBytecodeArena), statictype(rhs.statictype) {}
   virtual ~G__RflxProperties();
   G__RflxProperties& operator=(const G__RflxProperties& rhs)
   {
      if (this != &rhs) {
         autoload = rhs.autoload;
         filenum = rhs.filenum;
         linenum = rhs.linenum;
         globalcomp = rhs.globalcomp;
         iscpplink = rhs.iscpplink;
         typenum = rhs.typenum;
         tagnum = rhs.tagnum;
         isFromUsing = rhs.isFromUsing;
         comment = rhs.comment;
         stackinfo = rhs.stackinfo;
         vtable = rhs.vtable;
         isBytecodeArena = rhs.isBytecodeArena;
         statictype = rhs.statictype;
      }
      return *this;
   }
public:
   int autoload;
   int filenum; // entry in G__srcfile
   int linenum;
   char globalcomp;
   char iscpplink;
   int typenum;
   int tagnum;
   bool isFromUsing;
   G__comment_info comment;
   G__RflxStackProperties stackinfo;
   void* vtable;
   bool isBytecodeArena;
   int statictype;
};

class G__RflxVarProperties : public G__RflxProperties {
public:
   G__RflxVarProperties(): G__RflxProperties(), bitfield_start(0), bitfield_width(0), lock(false) {}
   G__RflxVarProperties(const G__RflxVarProperties& rhs): G__RflxProperties(rhs), bitfield_start(rhs.bitfield_start), bitfield_width(rhs.bitfield_width), lock(rhs.lock) {}
   virtual ~G__RflxVarProperties();
   G__RflxVarProperties& operator=(const G__RflxVarProperties& rhs)
   {
      if (this != &rhs) {
         this->G__RflxProperties::operator=(rhs);
         bitfield_start = rhs.bitfield_start;
         bitfield_width = rhs.bitfield_width;
         lock = rhs.lock;
      }
      return *this;
   }
public:
   short bitfield_start;
   short bitfield_width;
   bool lock;
};

class G__RflxFuncProperties : public G__RflxProperties {
public:
   G__RflxFuncProperties() : G__RflxProperties() {}
   G__RflxFuncProperties(const G__RflxFuncProperties& rhs) : G__RflxProperties(rhs), ifmethod(rhs.ifmethod), entry(rhs.entry) {}
   virtual ~G__RflxFuncProperties();
   G__RflxFuncProperties& operator=(const G__RflxFuncProperties& rhs)
   {
      if (this != &rhs) {
         this->G__RflxProperties::operator=(rhs);
         ifmethod = rhs.ifmethod;
         entry = rhs.entry;
      }
      return *this;
   }
public:
   G__InterfaceMethod ifmethod;
   G__funcentry entry;
};

namespace Cint {
namespace Internal {

class G__BuilderInfo {
public: // Public Types
   typedef std::vector<std::pair<std::string, std::string> > names_t;
public: // Public Interface
   G__BuilderInfo();
   G__BuilderInfo(const G__BuilderInfo& rhs); // NOT IMPLEMENTED
   ~G__BuilderInfo() {}
   G__BuilderInfo& operator=(const G__BuilderInfo& rhs); // NOT IMPLEMENTED
   std::string GetParamNames(); // "p1=val1; p2=val2", Internal, called by Build().
   void ParseParameterLink(const char* paras); // Dictionary Interface, called by v6_newlink.cxx(G__memfunc_setup)
   void AddParameter(int ifn, int type, int tagnum, int typenum, int reftype_const, G__value* para_default, char* para_def, char* para_name); // Internal, called by ParseParameterLink()
// called by v6_ifunc.cxx(G__make_ifunctable)
   ::Reflex::Member Build(const std::string name); // Called by v6_newlink.cxx(G__memfunc_setup), v6_ifunc.cxx(G__make_ifunctable).
public: // Public Data Members
   // -- The containing class of this member, despite the name.
   ::Reflex::Scope fBasetagnum; // This is our containing class, despite the name.
   // -- Offset of data member in most derived class.
   int fBaseoffset; // beginning of this data member's class in most derived class
   // -- Access.
   int fAccess; // public, protected, private
   // -- Modifiers.
   int fIsconst; // int f() const;
   int fIsexplicit; // explict MyClass();
   char fStaticalloc; // static f();
   int fIsvirtual; // virtual f();
   int fIspurevirtual; // virtual f() = 0;
   // -- Type of return value.
   ::Reflex::Type fReturnType; // type of return value
   //
   //  Function parameter names, parameter types,
   //  parameter default texts, and parameter default values
   //
   //  Note: Set by G__readansiproto().
   //
   std::vector<Reflex::Type> fParams_type; // [ type1, type2, ... ]
   names_t fParams_name; // [ (nm1, def1), (nm2, def2), ... ]
   std::vector<G__value*> fDefault_vals; // [ def_val1, def_val2, ... ]
   //
   //  Extended properties beyond what Reflex supports.
   //
   G__RflxFuncProperties fProp;
};

} // namespace Internal
} // namespace Cint


/*********************************************************************
* cint parser function and global variable prototypes
*********************************************************************/
#include "fproto.h"
#include "global.h"
#include "vararg.h"

/*********************************************************************
* New C++ Utility classes
*********************************************************************/
namespace Cint {
   namespace Internal {
      class G__CriticalSection {
      public:
         G__CriticalSection() { G__LockCriticalSection(); }
         G__CriticalSection(const G__CriticalSection& rhs); // NOT IMPLEMENTED
         ~G__CriticalSection() { G__UnlockCriticalSection(); }
         G__CriticalSection& operator=(const G__CriticalSection& rhs); // NOT IMPLEMENTED
      };
   }
}

#ifdef __cplusplus
#include "strbuf.h"
using namespace Cint::Internal;
#endif

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

