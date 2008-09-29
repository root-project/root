/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file varrag.h
 ************************************************************************
 * Description:
 * Header file for var arg definitions.
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__VARRARG_H
#define G__VARRARG_H

/* G__va_arg_buf(TAG) is defined in G__ci.h */

#if (defined(__i386__) && (defined(__linux) || defined(__APPLE__))) || \
    defined(_WIN32) || defined(G__CYGWIN)
/**********************************************
 * Intel architecture, aligns in multiple of 4 
 *    |1111|22  |3   |44444444|55555555555555  |
 **********************************************/
#define G__VAARG_INC_COPY_N 4

#elif (defined(__linux)&&defined(__ia64__))
/**********************************************
 * Itanium/linux, aligns in multiple of 8 
 **********************************************/

#define G__VAARG_INC_COPY_N 8
#define G__VAARG_PASS_BY_REFERENCE 8

#elif defined(__hpux) || defined(__hppa__)
/**********************************************
 * HP-Precision Architecture, 
 *  Args > 8 bytes are passed by reference.  Args > 4 and <= 8 are
 *  right-justified in 8 bytes.  Args <= 4 are right-justified in
 *  4 bytes. 
 **********************************************/
/* #define G__VAARG_NOSUPPORT */

#ifdef __ia64__
#define G__VAARG_INC_COPY_N 8
#else
#define G__VAARG_INC_COPY_N 4
#endif
#define G__VAARG_PASS_BY_REFERENCE 8

#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || \
      defined(__FreeBSD__))
/**********************************************
 * AMD64/EM64T
 * It turned out it is quite difficult to support this
 * platform as it uses registers for passing arguments (first 6 long
 * and first 8 double arguments in registers, the remaining on the stack)
 * for Linux/gcc.
 **********************************************/

#define G__VAARG_INC_COPY_N 8
/* #define G__VAARG_PASS_BY_REFERENCE 8 */

#elif defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || \
      defined(__SUNPRO_CC)
/**********************************************
 * Sun Sparc architecture
 * Alignment is similar to Intel, but class/struct
 * objects are passed by reference
 **********************************************/
/* #define G__VAARG_NOSUPPORT */

#define G__VAARG_INC_COPY_N 4
#define G__VAARG_PASS_BY_REFERENCE 8

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
/**********************************************
 * PowerPC, AIX and Apple Mac
 * It turned out it is quite difficult if not impossible to support PowerPC.
 * PPC uses registers for passing arguments (general purpose 3-10, floating 1)
 **********************************************/
#if !defined(__GNUC__)
/* Looks like gcc3.3 doesn't use registers. */
#define G__VAARG_NOSUPPORT
#endif
#define G__VAARG_INC_COPY_N 4
#define G__VAARG_PASS_BY_REFERENCE 8

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))
/**********************************************
 * PowerPC, Linux
 **********************************************/
#define G__VAARG_INC_COPY_N 4
#define G__VAARG_PASS_BY_REFERENCE 8

#elif (defined(__mips)&&defined(linux))
/**********************************************
* MIPS, Linux
**********************************************/
# define G__VAARG_INC_COPY_N 4
# define G__VAARG_PASS_BY_REFERENCE 8

#else
/**********************************************
 * Other platforms, 
 *  Try copying object as value.
 **********************************************/
#define G__VAARG_NOSUPPORT
#define G__VAARG_INC_COPY_N 4
/* #define G__VAARG_PASS_BY_REFERENCE 8 */

#endif


struct G__va_list_para {
  struct G__param *libp;
  int i;
};

#endif /* def G__VARRARG_H */
