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

#ifdef __cplusplus
#include <set>
#include <map>
extern "C"
#else
extern 
#endif
G__value G__default_parameter;

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
#define G__STATICCONST   16
#define G__FUNCTHROW     32

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
#ifndef G__ASM_DBG
#define G__ASM_DBG
#endif // G__ASM_DBG

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

struct G__breakcontinue_list {
  struct G__breakcontinue_list* next; // next entry in list
  int isbreak; // is it a break or a continue
  int idx; // index into bytecode array to patch
};

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
* Dictionary Generation Modes
*
* It is used for managing the flavour of the generated dictionary by rootcint
* 0 kCompleteDictionary = Complete dictionary with all stubs/wrappers (old style)
* 1 kShadowMembers = This dictionary contains the shadow members (Temporary dictionary #1)
* 2 kFunctionSymbols = This dictionary allows us to get the symbols of the member functions
* (Temporary dictionary #2) via pointers to the member functions, constructor dummy calls and 
* forcing inline member functions to be outline.
* 3 kNoWrappersDictionary = No Wrappers Dictionary (New Style) (#include temporary dictionaries)
*
**************************************************************************/

typedef enum { kCompleteDictionary = 0, kShadowMembers = 1, kFunctionSymbols = 2, kNoWrappersDictionary = 3} G__dictgenmode;


/**************************************************************************
* signal handling
**************************************************************************/
#define G__SIGNAL



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

struct G__Templatefuncarg {
  int paran;
  char type[G__MAXFUNCPARA];
  int tagnum[G__MAXFUNCPARA];
  int typenum[G__MAXFUNCPARA];
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
  char* string;
  struct G__Charlist* next;
};

struct G__Callfuncmacro {
  FILE* call_fp;
  fpos_t call_pos;
  int line;
  fpos_t mfp_pos;
  struct G__Callfuncmacro* next;
  short call_filenum;
};

struct G__Deffuncmacro {
  char* name;
  int hash;
  int line;
  FILE* def_fp;
  fpos_t def_pos;
  struct G__Charlist def_para;
  struct G__Callfuncmacro callfuncmacro;
  struct G__Deffuncmacro* next;
  short def_filenum;
};


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
  char *name;
  char *param;
  struct G__baseparam *next;
};


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

/***********************************************************************
* for function overloading
**********************************************************************/
struct G__funclist {
  struct G__ifunc_table_internal *ifunc;
  int ifn;
  unsigned int rate;
  unsigned int p_rate[G__MAXFUNCPARA];
  struct G__funclist *prev;
};

/*********************************************************************
* variable length string buffer
*********************************************************************/
/* #define G__BUFLEN 34 */
#define G__BUFLEN 180

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
  long localmem;
};

/**************************************************************************
*
* STRUCT DEFINITIONS
*
**************************************************************************/

#ifdef G__ASM_WHOLEFUNC
/**************************************************************************
* bytecode compiled interpreted function
*
**************************************************************************/
struct G__bytecodefunc {
  struct G__ifunc_table_internal *ifunc;
  int ifn;
  struct G__var_array *var;
  int varsize;
  G__value *pstack; /* upper part of stack to store numerical constants */
  int stacksize;
  long *pinst;      /* instruction buffer */
  int instsize;
  char *asm_name;   /* name of used ANSI library function */
};
#endif

/**************************************************************************
* structure for function entry
*
**************************************************************************/

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

struct G__funcentry {
   /* file position and pointer for restoring start point */
   fpos_t pos; /* Set if interpreted func body defined, unknown otherwise */
   void *p;  /* FILE* for source file or  int (*)() for compiled function
              * (void*)NULL if no function body */
   int  line_number; /* -1 if no function body or compiled function */
   short filenum;    /* -1 if compiled function, otherwise interpreted func */
   
   // Object Pointer Adjustement to call the stub function (entry) 
   long ptradjust;
   
#ifdef G__ASM_FUNC
   int size; /* size (number of lines) of function */
#endif
#ifdef G__TRUEP2F
   void *tp2f;
#endif
#ifdef G__ASM_WHOLEFUNC
   struct G__bytecodefunc *bytecode;
   int bytecodestatus;
#endif
#ifdef __cplusplus
   G__funcentry() : p(0),line_number(-1),filenum(-1),ptradjust(0),size(0),tp2f(0),bytecode(0),bytecodestatus(0) { 
      memset(&pos,0,sizeof(pos)); }
#endif
};

/**************************************************************************
* structure for ifunc (Interpreted FUNCtion) table
*
**************************************************************************/
struct G__paramfunc {
  short p_tagtable;
  short p_typetable;
  char  reftype;
  char  type;
  char  isconst;
  char *name;
  char *def;
  char  id;
  G__value *pdefault;
  struct G__paramfunc *next;
};

struct G__params {
   struct G__paramfunc* fparams;
#ifdef __cplusplus  
   G__params()
   : fparams(0)
   {
   }
   void reset() {
      struct G__paramfunc* params = fparams;
      struct G__paramfunc* next = 0;
      for (; params; params = next) {
         if (params->name) {
            free(params->name);
            params->name = 0;
         }
         if (params->def) {
            free(params->def);
            params->def = 0;
         }
         if (params->pdefault) {
            if ((params->pdefault != &G__default_parameter) && (params->pdefault != (G__value*) (-1))) {
               free(params->pdefault);
            }
            params->pdefault = 0;
         }
         next = params->next;
         params->next = 0;
         free(params);
      }
      fparams = 0;      
   }
   ~G__params()
   {
      reset();
   }
   struct G__paramfunc* operator[](char idx)
   {
      if (!fparams) {
         fparams = (struct G__paramfunc*) malloc(sizeof (struct G__paramfunc));
         memset(fparams, 0, sizeof (struct G__paramfunc));
         fparams->id = idx;
         return fparams;
      }
      struct G__paramfunc* params = fparams;
      for (; params; params = params->next) {
         if (params->id == idx) {
            return params;
         }
         struct G__paramfunc* nparams = params->next;
         if (!nparams) {
            nparams = (struct G__paramfunc*) malloc(sizeof (struct G__paramfunc));
            memset(nparams, 0, sizeof (struct G__paramfunc));
            nparams->id = idx;
            params->next = nparams;
            return nparams;
         }
      }
      return 0;
   }
#endif
};

struct G__ifunc_table {
   int tagnum;
   int page; /* G__struct.ifunc->next->next->... */
   struct G__ifunc_table_internal* ifunc_cached;
#ifdef __cplusplus
   bool operator < (const struct G__ifunc_table& right) const {
      return tagnum < right.tagnum 
         || (tagnum == right.tagnum && page < right.page);
   }
#endif
};
struct G__ifunc_table_internal {
#ifdef __cplusplus
   G__ifunc_table_internal() : inited(true),allifunc(0),next(0),page(0),page_base(0),tagnum(-1) {
      for(unsigned i = 0; i < G__MAXIFUNC; ++i) {
         funcname[i]       = 0;
         hash[i]           = 0;
         funcptr[i]        = 0;
         mangled_name[i]   = 0;
         pentry[i]         = 0;
         type[i]           = 0;
         p_tagtable[i]     = 0;
         p_typetable[i]    = 0;
         reftype[i]        = 0;
         para_nu[i]        = 0;
         isconst[i]        = 0;
         isexplicit[i]     = 0;
         iscpp[i]          = 0;
         ansi[i]           = 0;
         busy[i]           = 0;
         access[i]         = 0;
         staticalloc[i]    = 0;
         isvirtual[i]      = 0;
         ispurevirtual[i]  = 0;
         friendtag[i]      = 0;
         globalcomp[i]     = 0;
         userparam[i]      = 0;
         vtblindex[i]      = 0;
         vtblbasetagnum[i] = 0;
      }
   };
#endif
   
  /* true if the constructor was run */
  char inited;
   
  /* number of interpreted function */
  int allifunc;

  /* function name and hash for identification */
  char *funcname[G__MAXIFUNC];
  int  hash[G__MAXIFUNC];

  // 23/02/2007
  // We want to have a direct pointer to the function.
  // It's used for the stub-less calls.
  // And the mangled name is used to get this ptr
  void *funcptr[G__MAXIFUNC];
  char *mangled_name[G__MAXIFUNC];

  struct G__funcentry entry[G__MAXIFUNC],*pentry[G__MAXIFUNC];

  /* type of return value */
  G__SIGNEDCHAR_T type[G__MAXIFUNC];
  short p_tagtable[G__MAXIFUNC];
  short p_typetable[G__MAXIFUNC];
  G__SIGNEDCHAR_T reftype[G__MAXIFUNC];
  short para_nu[G__MAXIFUNC];
  G__SIGNEDCHAR_T isconst[G__MAXIFUNC];
  G__SIGNEDCHAR_T isexplicit[G__MAXIFUNC];

  /* number and type of function parameter */
  /* G__inheritclass() depends on type of following members */
  struct G__params param[G__MAXIFUNC];

  /* C or C++ */
  char iscpp[G__MAXIFUNC];

  /* ANSI or standard header format */
  char ansi[G__MAXIFUNC];

  /**************************************************
   * if function is called, busy[] is incremented
   **************************************************/
  short busy[G__MAXIFUNC];

  struct G__ifunc_table_internal *next;
  int page;

  // 06-08-07
  // an additional 'page' indicating the index
  // of the function in the base class
  int page_base;

  G__SIGNEDCHAR_T access[G__MAXIFUNC];  /* private, protected, public */
  char staticalloc[G__MAXIFUNC];

  int tagnum;
  char isvirtual[G__MAXIFUNC]; /* virtual function flag */
  char ispurevirtual[G__MAXIFUNC]; /* virtual function flag */

#ifdef G__FRIEND
  struct G__friendtag *friendtag[G__MAXIFUNC];
#endif

  G__SIGNEDCHAR_T globalcomp[G__MAXIFUNC];

  struct G__comment_info comment[G__MAXIFUNC];

  void* userparam[G__MAXIFUNC]; /* user parameter array */
  short vtblindex[G__MAXIFUNC];
  short vtblbasetagnum[G__MAXIFUNC];
};

/**************************************************************************
* structure for class inheritance
*
**************************************************************************/
struct G__herit {
  short basetagnum;
#ifdef G__VIRTUALBASE
  long baseoffset;
#else
  int baseoffset;
#endif
  G__SIGNEDCHAR_T baseaccess;
  char property;
  char  id;
  struct G__herit* next;
};

struct G__herits {
  struct G__herit* fherits;
#ifdef __cplusplus
   G__herits()
   : fherits(0)
   {
   }
   ~G__herits()
   {
      struct G__herit* herits = fherits;
      struct G__herit* nxt = 0;
      for (; herits; herits = nxt) {
         nxt = herits->next;
         herits->next = 0;
         free(herits);
      }
      fherits = 0;
   }
   struct G__herit* operator[](char idx)
   {
      if (!fherits) {
         fherits = (struct G__herit*) malloc(sizeof(struct G__herit));
         memset(fherits, 0, sizeof(struct G__herit));
         fherits->id = idx;
         return fherits;
      }
      struct G__herit* herits = fherits;
      while (herits) {
         if (herits->id == idx) {
            return herits;
         }
         struct G__herit* nherits  = herits->next;
         if (!nherits) {
            nherits = (struct G__herit*) malloc(sizeof(struct G__herit));
            memset(nherits, 0, sizeof(struct G__herit));
            nherits->id = idx;
            herits->next = nherits;
            return nherits;
         }
         herits = herits->next;
      }
      return 0;
   }
#endif
};

struct G__inheritance {
  int basen;
  struct G__herits herit;
};


/**************************************************************************
* structure for variable table
*
**************************************************************************/
struct G__var_array {
  /* union for variable pointer */
  long p[G__MEMDEPTH]; /* used to be int */
  int allvar;
  char *varnamebuf[G__MEMDEPTH]; /* variable name */
  int hash[G__MEMDEPTH];                    /* hash table of varname */
  size_t varlabel[G__MEMDEPTH+1][G__MAXVARDIM];  /* points varpointer */
  short paran[G__MEMDEPTH];
  char bitfield[G__MEMDEPTH];
  char is_init_aggregate_array[G__MEMDEPTH];
#ifdef G__VARIABLEFPOS
  int filenum[G__MEMDEPTH];
  int linenum[G__MEMDEPTH];
#endif

  /* type information,
     if pointer : Char,Int,Short,Long,Double,U(struct,union)
     if value   : char,int,short,long,double,u(struct,union) */
  G__SIGNEDCHAR_T  type[G__MEMDEPTH];
  G__SIGNEDCHAR_T constvar[G__MEMDEPTH];
  short p_tagtable[G__MEMDEPTH];        /* tagname if struct,union */
  short p_typetable[G__MEMDEPTH];       /* typename if typedef */
  short statictype[G__MEMDEPTH];
  G__SIGNEDCHAR_T reftype[G__MEMDEPTH];

  /* chain for next G__var_array */
  struct G__var_array *next;

  G__SIGNEDCHAR_T access[G__MEMDEPTH];  /* private, protected, public */

#ifdef G__SHOWSTACK /* not activated */
  struct G__ifunc_table *ifunc;
  int ifn;
  struct G__var_array *prev_local;
  int prev_filenum;
  short prev_line_number;
  long struct_offset;
  int tagnum;
  int exec_memberfunc;
#endif
#ifdef G__VAARG
  struct G__param *libp;
#endif

#ifndef G__NEWINHERIT
  char isinherit[G__MEMDEPTH];
#endif
  G__SIGNEDCHAR_T globalcomp[G__MEMDEPTH];

  struct G__comment_info comment[G__MEMDEPTH];

#ifndef G__OLDIMPLEMENTATION2038
  struct G__var_array *enclosing_scope;
  struct G__var_array **inner_scope;
#endif

} ;

/**************************************************************************
* structure struct,union tag information
*
**************************************************************************/

#ifdef __cplusplus

class NameMap {
public:
   class Range {
   public:
      Range(): fFirst(-1), fLast(-1) {}
      Range(const std::set<int>& s): fFirst(*s.begin()), fLast(*s.rbegin()) {}
      int First() const { return fFirst; }
      int Last() const { return fLast; }
      bool Empty() const { return fFirst == -1; }
      operator bool() const { return fFirst != -1; }
   private:
      int fFirst;
      int fLast;
   };

   NameMap() {}
   void Insert(const char* name, int idx) {
      fMap[name].insert(idx); 
   }
   void Remove(const char* name, int idx, char **namepool);

   Range Find(const char* name) {
      NameMap_t::const_iterator iMap = fMap.find(name);
      if (iMap != fMap.end() && !iMap->second.empty())
         return Range(iMap->second);
      return Range();
   }
   void Print() {
      NameMap_t::iterator iMap = fMap.begin();
      while( iMap != fMap.end() ) {
         fprintf(stderr,"key=%s size=%ld\n",iMap->first,(long)iMap->second.size());
         ++iMap;
      }  
   }
   
private:
   struct G__charptr_less {
      bool operator() (const char* a, const char* b) const {
         return !a || (b && (strcmp(a, b) < 0));
      }
   };

   typedef std::map<const char*, std::set<int>, G__charptr_less> NameMap_t;
   NameMap_t fMap;
};

struct G__tagtable {
  /* tag entry information */
  char type[G__MAXSTRUCT]; /* struct,union,enum,class */

  char *name[G__MAXSTRUCT];
  int  hash[G__MAXSTRUCT];
  int  size[G__MAXSTRUCT];
  /* member information */
  struct G__var_array *memvar[G__MAXSTRUCT];
  struct G__ifunc_table_internal *memfunc[G__MAXSTRUCT];
  struct G__inheritance *baseclass[G__MAXSTRUCT];
  int virtual_offset[G__MAXSTRUCT];
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

  struct G__comment_info comment[G__MAXSTRUCT];

   std::list<G__incsetup> *incsetup_memvar[G__MAXSTRUCT];
   std::list<G__incsetup> *incsetup_memfunc[G__MAXSTRUCT];

  char rootflag[G__MAXSTRUCT];
  struct G__RootSpecial *rootspecial[G__MAXSTRUCT];

  char isctor[G__MAXSTRUCT];

#ifndef G__OLDIMPLEMENTATION1503
  int defaulttypenum[G__MAXSTRUCT];
#endif
  void* userparam[G__MAXSTRUCT];     /* user parameter array */
  char* libname[G__MAXSTRUCT];
  void* vtable[G__MAXSTRUCT];
  /* short vtabledepth[G__MAXSTRUCT]; */
  NameMap* namerange;
};

#else /* ifdef __cplusplus */

struct G__tagtable;

#endif

/**************************************************************************
* structure typedef information
*
**************************************************************************/

struct G__typedef {
  char type[G__MAXTYPEDEF];
  char *name[G__MAXTYPEDEF];
  int  hash[G__MAXTYPEDEF];
  short  tagnum[G__MAXTYPEDEF];
  char reftype[G__MAXTYPEDEF];
#ifdef G__CPPLINK1
  G__SIGNEDCHAR_T globalcomp[G__MAXTYPEDEF];
#endif
  int nindex[G__MAXTYPEDEF];
  int *index[G__MAXTYPEDEF];
  short parent_tagnum[G__MAXTYPEDEF];
  char iscpplink[G__MAXTYPEDEF];
  struct G__comment_info comment[G__MAXTYPEDEF];
#ifdef G__TYPEDEFFPOS
  int filenum[G__MAXTYPEDEF];
  int linenum[G__MAXTYPEDEF];
#endif
  int alltype;
  G__SIGNEDCHAR_T isconst[G__MAXTYPEDEF];
#ifdef __cplusplus
  NameMap* namerange;
#else
  void* namerange;
#endif
};

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

/*********************************************************************
* cint parser function and global variable prototypes
*********************************************************************/
#include "security.h"
#include "fproto.h"
#include "global.h"
#include "config/snprintf.h"
#include "config/strlcpy.h"

#ifdef __cplusplus
#include "FastAllocString.h"
using namespace Cint::Internal;

/**************************************************************************
 * user specified pragma statement
 **************************************************************************/
extern "C" {
   typedef void (*G__AppPragma_func_t)(char*);
}

struct G__AppPragma {

   G__FastAllocString name;
   G__AppPragma_func_t p2f;
   struct G__AppPragma *next;

   G__AppPragma(char *comname, G__AppPragma_func_t p2f);
   ~G__AppPragma();
};

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

