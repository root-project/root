/* @(#)root/zip:$Id$ */
/* Author: */
/*

 Copyright (C) 1990-1993 Mark Adler, Richard B. Wales, Jean-loup Gailly,
 Kai Uwe Rommel and Igor Mandrichenko.
 For conditions of distribution and use, see copyright notice in zlib.h

*/

/*
 *  zip.h by Mark Adler.
 */


#define ZIP   /* for crypt.c:  include zip password functions, not unzip */

/* Set up portability */
/* #include "tailor.h" */
#include "Tailor.h"

#define MIN_MATCH  3
#define MAX_MATCH  258
/* The minimum and maximum match lengths */

#ifndef WSIZE
#  define WSIZE  ((unsigned)32768)
#endif
/* Maximum window size = 32K. If you are really short of memory, compile
 * with a smaller WSIZE but this reduces the compression ratio for files
 * of size > WSIZE. WSIZE must be a power of two in the current implementation.
 */

#define MIN_LOOKAHEAD (MAX_MATCH+MIN_MATCH+1)
/* Minimum amount of lookahead, except at the end of the input file.
 * See deflate.c for comments about the MIN_MATCH+1.
 */

#define MAX_DIST  (WSIZE-MIN_LOOKAHEAD)
/* In order to simplify the code, particularly on 16 bit machines, match
 * distances are limited to MAX_DIST instead of WSIZE.
 */


/* Define fseek() commands */
#ifndef SEEK_SET
#  define SEEK_SET 0
#endif /* !SEEK_SET */

#ifndef SEEK_CUR
#  define SEEK_CUR 1
#endif /* !SEEK_CUR */

/* Types centralized here for easy modification */
#define local static            /* More meaningful outside functions */
typedef unsigned char uch;      /* unsigned 8-bit value */
typedef unsigned short ush;     /* unsigned 16-bit value */
typedef unsigned long ulg;      /* unsigned 32-bit value */

/* internal file attribute */
#define UNKNOWN (-1)
#define BINARY  0
#define ASCII   1

#define BEST -1                 /* Use best method (deflation or store) */
#define STORE 0                 /* Store method */
#define DEFLATE 8               /* Deflation method*/

static int verbose=0;           /* Report oddities in zip file structure */
static int level=6;             /* Compression level */

/* Diagnostic functions */
#ifdef DEBUG
# ifdef MSDOS
#  undef  stderr
#  define stderr stdout
# endif
#  define diag(where) fprintf(stderr, "zip diagnostic: %s\n", where)
#  define Assert(cond,msg) {if(!(cond)) error(msg);}
#  define Trace(x) fprintf x
#  define Tracev(x) {if (verbose) fprintf x ;}
#  define Tracevv(x) {if (verbose>1) fprintf x ;}
#  define Tracec(c,x) {if (verbose && (c)) fprintf x ;}
#  define Tracecv(c,x) {if (verbose>1 && (c)) fprintf x ;}
#else
#  define diag(where)
#  define Assert(cond,msg)
#  define Trace(x)
#  define Tracev(x)
#  define Tracevv(x)
#  define Tracec(c,x)
#  define Tracecv(c,x)
#endif

#ifndef UTIL
typedef struct bits_internal_state bits_internal_state;
typedef struct tree_internal_state tree_internal_state;
        /* in deflate.c */
int R__lm_init OF((bits_internal_state *state,int pack_level, ush *flags));
void R__lm_free OF((void));
ulg  R__Deflate OF((bits_internal_state *state,int *errorflag));

        /* in trees.c */
int  R__ct_init     OF((tree_internal_state *t_state, ush *attr, int *method));
int  R__ct_tally    OF((bits_internal_state *state, int dist, int lc));
ulg  R__flush_block OF((bits_internal_state *state, char far *buf, ulg stored_len, int eof,int *errorflag));
tree_internal_state *R__get_thread_tree_state   OF((void));

        /* in bits.c */
int      R__bi_init    OF((bits_internal_state *state));
void     R__send_bits  OF((bits_internal_state *state,int value, int length));
unsigned R__bi_reverse OF((unsigned value, int length));
void     R__bi_windup  OF((bits_internal_state *state));
void     R__copy_block OF((bits_internal_state *state,char far *buf, unsigned len, int header));
int      R__seekable   OF((void));
/* On some platform (MacOS) marking this thread local does not work,
 however in our use this is a constant, so we do not really need to make it
 thread local */
#ifdef _MSC_VER
extern /* __declspec( thread ) */ int (*R__read_buf) OF((char *buf, unsigned size));
#else
extern /* __thread */ int (*R__read_buf) OF((char *buf, unsigned size));
#endif
ulg      R__memcompress OF((char *tgt, ulg tgtsize, char *src, ulg srcsize));
void     R__error      OF((char *h));

#endif /* !UTIL */

/* end of zip.h */

