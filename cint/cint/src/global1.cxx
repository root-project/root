/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file global1.c
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

/***********************************************************************
* EH_env and err_env are only variable name which isn't escaped by 'G__' 
***********************************************************************/
#ifdef G__HSTD
jmp_buf EH_env;
/* jmp_buf err_env; */ /* old nucleus */
#endif /* G__HSTD */


/**************************************************************************
* get CINTSYSDIR environment variable here
**************************************************************************/
char G__cintsysdir[G__MAXFILENAME] ; 


/**************************************************************************
* make sutpi file
**************************************************************************/
FILE *G__sutpi;      /* sutpi file pointer */


int G__typepdecl; /* to be commented */

/**************************************************************************
* Loop compilation instruction, stack and temporaly variables
**************************************************************************/
#ifdef G__ASM

/********************************************************
* whole function bytecode compilation flag
*********************************************************/
int G__asm_cond_cp = -1; /* avoid wrong bytecode optimization */

#ifdef G__ASM_WHOLEFUNC
int G__asm_wholefunction;
#endif

int G__asm_wholefunc_default_cp;

#ifdef G__ASM_IFUNC

long *G__asm_inst; /* p-code instruction buffer */
int G__asm_instsize;
G__value *G__asm_stack; /* data stack and constant buffer */
char *G__asm_name;

long G__asm_inst_g[G__MAXINST]; /* p-code instruction buffer */
G__value G__asm_stack_g[G__MAXSTACK]; /* data stack */
char G__asm_name_g[G__ASM_FUNCNAMEBUF]; /* buffer to store function names 
                                         * which is called within the 
                                         * compiled loop */
int G__asm_name_p=0; /* pointer to the function name buffer */

#else /* G__ASM_IFUNC */
long G__asm_inst[G__MAXINST]; /* p-code instruction buffer */
G__value G__asm_stack[G__MAXSTACK]; /* data stack */
char G__asm_name[G__LONGLINE*2]; /* buffer to store function names which 
                                * is called within the compiled loop */
int G__asm_name_p=0; /* pointer to the function name buffer */
#endif /* G__ASM_IFUNC */

/* Compile time program counter and constant data stack pointer */
int G__asm_cp=0;               /* compile time program counter */
int G__asm_dt=G__MAXSTACK-1;   /* compile time stack pointer */

/* Global variables to bring compilation data */
int G__asm_index;              /* variable index */
struct G__param *G__asm_param; /* pointer of parameter buffer to 
                                * bring function parameter */

/* Loop compiler flags */
int G__asm_loopcompile; /* loop compilation mode. default on(4). 
                           * This is set to 0 by -O0 */
int G__asm_loopcompile_mode; 
int G__asm_exec=0; /* p-code execution flag */
int G__asm_noverflow=0; /* When this is set to 1, compilation starts. 
                         * If any error found, reset */

int G__asm_dbg=0; /* p-code debugging flag, only valid when compiled with
                   * G__ASM_DBG */

char G__wrappers = 0;
char G__nostubs = 0;

#ifdef G__ASM_DBG
const char *G__LOOPCOMPILEABORT="LOOP COMPILE ABORTED";
#endif

#endif /* G__ASM */

/**************************************************************************
* signal handling
**************************************************************************/
#ifdef G__SIGNAL
char *G__SIGINT; 
char *G__SIGILL; 
char *G__SIGFPE; 
char *G__SIGABRT; 
char *G__SIGSEGV; 
char *G__SIGTERM; 
#ifdef SIGHUP
char *G__SIGHUP;
#endif
#ifdef SIGQUIT
char *G__SIGQUIT;
#endif
#ifdef SIGTSTP
char *G__SIGTSTP;
#endif
#ifdef SIGTTIN
char *G__SIGTTIN;
#endif
#ifdef SIGTTOU
char *G__SIGTTOU;
#endif
#ifdef SIGALRM
char *G__SIGALRM;
#endif
#ifdef SIGUSR1
char *G__SIGUSR1;
#endif
#ifdef SIGUSR2
char *G__SIGUSR2;
#endif
#endif


/**************************************************************************
* class template 
**************************************************************************/
#ifdef G__TEMPLATECLASS
struct G__Definedtemplateclass G__definedtemplateclass;
#endif
int G__macroORtemplateINfile;

#ifdef G__TEMPLATEFUNC
struct G__Definetemplatefunc G__definedtemplatefunc;
#endif


/**************************************************************************
* Macro statement support
**************************************************************************/
FILE *G__mfp;
fpos_t G__nextmacro;
int G__mline;
const char *G__macro="tmpfile";
struct G__Deffuncmacro G__deffuncmacro;
char G__macros[16 * G__LONGLINE];
G__FastAllocString G__ppopt("");
char *G__allincludepath=(char*)NULL;
const char *G__undeflist="";

/**************************************************************************
* Macro constant
**************************************************************************/
int G__macro_defining;

/**************************************************************************
* Array type typedef support, 'typedef int a[10];'
**************************************************************************/
int G__typedefnindex;
int *G__typedefindex;


/**************************************************************************
* Text processing capability
*
*   fp=fopen("xxx","r");
*   while($read(fp)) {
*      printf("%d %s %s\n",$#,$1,$2);
*   }
**************************************************************************/
char G__oline[G__LONGLINE*2],G__argb[G__LONGLINE*2],*G__arg[G__ONELINE];
int G__argn;


/**************************************************************************
* structure for global and local variables
*
**************************************************************************/
struct G__var_array  G__global ;
struct G__var_array *G__p_local;
struct G__inheritance G__globalusingnamespace;

/**************************************************************************
* structure for struct,union tag information
* structure for typedef information
**************************************************************************/
struct G__tagtable  G__struct;
struct G__typedef   G__newtype;


/**************************************************************************
* structure for input file
**************************************************************************/
struct G__input_file  G__ifile;


/**************************************************************************
* structure for ifunc (Interpreted FUNCtion) table
**************************************************************************/
struct G__ifunc_table_internal G__ifunc ;
struct G__ifunc_table_internal *G__p_ifunc;


/**************************************************************************
* cint G__value constants
**************************************************************************/
G__value G__null,G__start,G__default,G__one;
G__value G__block_break,G__block_continue;
G__value G__block_goto;
char G__gotolabel[G__MAXNAME];

/**************************************************************************
* allocation of array by new operator ?
**************************************************************************/
struct G__newarylist G__newarray;


/**************************************************************************
* pointer to function which is evaluated at pause 
**************************************************************************/
void (*G__atpause)();

/**************************************************************************
* pointer to function which is evaluated in G__genericerror()
**************************************************************************/
void (*G__aterror)();

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
