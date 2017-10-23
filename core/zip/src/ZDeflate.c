/* @(#)root/zip:$Id$ */
/* Author: */
/*

 Copyright (C) 1990-1993 Mark Adler, Richard B. Wales, Jean-loup Gailly,
 Kai Uwe Rommel and Igor Mandrichenko.
 For conditions of distribution and use, see copyright notice in zlib.h

*/
#include "Bits.h"

/*
 *  deflate.c by Jean-loup Gailly.
 *
 *  PURPOSE
 *
 *      Identify new text as repetitions of old text within a fixed-
 *      length sliding window trailing behind the new text.
 *
 *  DISCUSSION
 *
 *      The "deflation" process depends on being able to identify portions
 *      of the input text which are identical to earlier input (within a
 *      sliding window trailing behind the input currently being processed).
 *
 *      The most straightforward technique turns out to be the fastest for
 *      most input files: try all possible matches and select the longest.
 *      The key feature of this algorithm is that insertions into the string
 *      dictionary are very simple and thus fast, and deletions are avoided
 *      completely. Insertions are performed at each input character, whereas
 *      string matches are performed only when the previous match ends. So it
 *      is preferable to spend more time in matches to allow very fast string
 *      insertions and avoid deletions. The matching algorithm for small
 *      strings is inspired from that of Rabin & Karp. A brute force approach
 *      is used to find longer strings when a small match has been found.
 *      A similar algorithm is used in comic (by Jan-Mark Wams) and freeze
 *      (by Leonid Broukhis).
 *         A previous version of this file used a more sophisticated algorithm
 *      (by Fiala and Greene) which is guaranteed to run in linear amortized
 *      time, but has a larger average cost, uses more memory and is patented.
 *      However the F&G algorithm may be faster for some highly redundant
 *      files if the parameter max_chain_length (described below) is too large.
 *
 *  ACKNOWLEDGEMENTS
 *
 *      The idea of lazy evaluation of matches is due to Jan-Mark Wams, and
 *      I found it in 'freeze' written by Leonid Broukhis.
 *      Thanks to many info-zippers for bug reports and testing.
 *
 *  REFERENCES
 *
 *      APPNOTE.TXT documentation file in PKZIP 1.93a distribution.
 *
 *      A description of the Rabin and Karp algorithm is given in the book
 *         "Algorithms" by R. Sedgewick, Addison-Wesley, p252.
 *
 *      Fiala,E.R., and Greene,D.H.
 *         Data Compression with Finite Windows, Comm.ACM, 32,4 (1989) 490-595
 *
 *  INTERFACE
 *
 *      void lm_init (int pack_level, ush *flags)
 *          Initialize the "longest match" routines for a new file
 *
 *      ulg deflate (void)
 *          Processes a new input file and return its compressed length. Sets
 *          the compressed length, crc, deflate flags and internal file
 *          attributes.
 */

/* #include "zip.h" */
/* #include "ZIP.h" */

/* ===========================================================================
 * Configuration parameters
 */

/* Compile with MEDIUM_MEM to reduce the memory requirements or
 * with SMALL_MEM to use as little memory as possible. Use BIG_MEM if the
 * entire input file can be held in memory (not possible on 16 bit systems).
 * Warning: defining these symbols affects HASH_BITS (see below) and thus
 * affects the compression ratio. The compressed output
 * is still correct, and might even be smaller in some cases.
 */

#if BITS_NOT_INCLUDED
#ifdef SMALL_MEM
#   define HASH_BITS  13  /* Number of bits used to hash strings */
#endif
#ifdef MEDIUM_MEM
#   define HASH_BITS  14
#endif
#ifndef HASH_BITS
#   define HASH_BITS  15
/* For portability to 16 bit machines, do not use values above 15. */
#endif
#endif

/* #define HASH_SIZE (unsigned)(1<<HASH_BITS) now in Bits.h */
#define HASH_MASK (HASH_SIZE-1)
#define WMASK     (WSIZE-1)
/* HASH_SIZE and WSIZE must be powers of two */

#define NIL 0
/* Tail of hash chains */

#define FAST 4
#define SLOW 2
/* speed options for the general purpose bit flag */

#ifndef TOO_FAR
#  define TOO_FAR 4096
#endif
/* Matches of length 3 are discarded if their distance exceeds TOO_FAR */

#ifdef ATARI_ST
#  undef MSDOS /* avoid the processor specific parts */
   /* (but the Atari should never define MSDOS anyway ...) */
#endif
#if defined(MSDOS) && !defined(NO_ASM) && !defined(ASMV) && !defined(WIN32)
#  define ASMV
#endif
#if defined(ASMV) && !defined(MSDOS) && defined(DYN_ALLOC)
  error: DYN_ALLOC not yet supported in match.s
#endif
#if defined(MSDOS) && !defined(__32BIT__)
#  define MAXSEG_64K
#endif



/* Values for state->max_lazy_match, good_match and max_chain_length, depending on
 * the desired pack level (0..9). The values given below have been tuned to
 * exclude worst case performance for pathological files. Better values may be
 * found for specific files.
 */

typedef struct config {
   ush good_length; /* reduce lazy search above this match length */
   ush max_lazy;    /* do not perform lazy search above this match length */
   ush nice_length; /* quit search above this match length */
   ush max_chain;
} config;


local config configuration_table[10] = {
/*      good lazy nice chain */
/* 0 */ {0,    0,  0,    0},  /* store only */
/* 1 */ {4,    4,  8,    4},  /* maximum speed, no lazy matches */
/* 2 */ {4,    5, 16,    8},
/* 3 */ {4,    6, 32,   32},

/* 4 */ {4,    4, 16,   16},  /* lazy matches */
/* 5 */ {8,   16, 32,   32},
/* 6 */ {8,   16, 128, 128},
/* 7 */ {8,   32, 128, 256},
/* 8 */ {32, 128, 258, 1024},
/* 9 */ {32, 258, 258, 4096}}; /* maximum compression */

/* Note: the deflate() code requires max_lazy >= MIN_MATCH and max_chain >= 4
 * For deflate_fast() (levels <= 3) good is ignored and lazy has a different
 * meaning.
 */

#define EQUAL 0
/* result of memcmp for equal strings */

/* ===========================================================================
 *  Prototypes for local functions.
 */

local void R__fill_window    OF((bits_internal_state *state));
local ulg  R__Deflate_fast   OF((bits_internal_state *state,int *errorflag));

      int  R__longest_match  OF((bits_internal_state *state, IPos cur_match));
#ifdef ASMV
      void match_init OF((void)); /* asm code initialization */
#endif

#ifdef DEBUG
local  void check_match OF((IPos start, IPos match, int length));
#endif

/* ===========================================================================
 * Update a hash value with the given input byte
 * IN  assertion: all calls to to UPDATE_HASH are made with consecutive
 *    input characters, so that a running hash key can be computed from the
 *    previous key instead of complete recalculation each time.
 */
#define UPDATE_HASH(h,c) (h = (((h)<<H_SHIFT) ^ (c)) & HASH_MASK)

/* ===========================================================================
 * Insert string s in the dictionary and set match_head to the previous head
 * of the hash chain (the most recent string with same hash key). Return
 * the previous length of the hash chain.
 * IN  assertion: all calls to to INSERT_STRING are made with consecutive
 *    input characters and the first MIN_MATCH bytes of s are valid
 *    (except for the last MIN_MATCH-1 bytes of the input file).
 */
#define INSERT_STRING(s, match_head) \
   (UPDATE_HASH(state->ins_h, state->R__window[(s) + MIN_MATCH-1]), \
    state->R__prev[(s) & WMASK] = match_head = state->R__head[state->ins_h], \
    state->R__head[state->ins_h] = (s))

/* ===========================================================================
 * Initialize the "longest match" routines for a new file
 *
 * IN assertion: window_size is > 0 if the input file is already read or
 *    mmap'ed in the window[] array, 0 otherwise. In the first case,
 *    window_size is sufficient to contain the whole input file plus
 *    MIN_LOOKAHEAD bytes (to avoid referencing memory beyond the end
 *    of window[] when looking for matches towards the end).
 */
int R__lm_init (bits_internal_state *state, int pack_level, ush *flags)
    /* int pack_level;  0: store, 1: best speed, 9: best compression */
    /* ush *flags;      general purpose bit flag */
{
    register unsigned j;

    if (pack_level < 1 || pack_level > 9) {
      R__error("bad pack level");
      return 1;
    }

    /* Do not slide the window if the whole input is already in memory
     * (window_size > 0)
     */
    state->sliding = 0;
    if (state->R__window_size == 0L) {
        state->sliding = 1;
        state->R__window_size = (ulg)2L*WSIZE;
    }

    /* Use dynamic allocation if compiler does not like big static arrays: */
#ifdef DYN_ALLOC
    if (state->R__window == NULL) {
        state->R__window = (uch*) fcalloc(WSIZE,   2*sizeof(uch));
        if (state->R__window == NULL)  { R__error("window allocation"); return 1; }
    }
    if (state->R__prev == NULL) {
        state->R__prev   = (Pos*) fcalloc(WSIZE,     sizeof(Pos));
        state->R__head   = (Pos*) fcalloc(HASH_SIZE, sizeof(Pos));
        if (state->R__prev == NULL || state->R__head == NULL) {
            R__error("hash table allocation");
            return 1;
        }
    }
#endif /* DYN_ALLOC */

    /* Initialize the hash table (avoiding 64K overflow for 16 bit systems).
     * prev[] will be initialized on the fly.
     */
    state->R__head[HASH_SIZE-1] = NIL;
    memset((char*)state->R__head, NIL, (unsigned)(HASH_SIZE-1)*sizeof(*state->R__head));

    /* Set the default configuration parameters:
     */
    state->max_lazy_match   = configuration_table[pack_level].max_lazy;
    state->R__good_match    = configuration_table[pack_level].good_length;
#ifndef FULL_SEARCH
    state->R__nice_match    = configuration_table[pack_level].nice_length;
#endif
    state->R__max_chain_length = configuration_table[pack_level].max_chain;
    if (pack_level == 1) {
       *flags |= FAST;
    } else if (pack_level == 9) {
       *flags |= SLOW;
    }
    /* ??? reduce max_chain_length for binary files */

    state->R__strstart = 0;
    state->R__block_start = 0L;
#ifdef ASMV
    match_init(); /* initialize the asm code */
#endif

    j = WSIZE;
#ifndef MAXSEG_64K
    if (sizeof(int) > 2) j <<= 1; /* Can read 64K in one step */
#endif
    state->lookahead = R__mem_read(state,(char*)state->R__window, j);

    if (state->lookahead == 0 || state->lookahead == (unsigned)EOF) {
       state->eofile = 1, state->lookahead = 0;
       return 0;
    }
    state->eofile = 0;
    /* Make sure that we always have enough state->lookahead. This is important
     * if input comes from a device such as a tty.
     */
    while (state->lookahead < MIN_LOOKAHEAD && !state->eofile) R__fill_window(state);

    state->ins_h = 0;
    for (j=0; j<MIN_MATCH-1; j++) UPDATE_HASH(state->ins_h, state->R__window[j]);
    /* If state->lookahead < MIN_MATCH, state->ins_h is garbage, but this is
     * not important since only literal bytes will be emitted.
     */
    return 0;
}

/* ===========================================================================
 * Free the window and hash table
 */
void R__lm_free()
{
#ifdef DYN_ALLOC
    if (state->R__window != NULL) {
        fcfree(state->R__window);
        state->R__window = NULL;
    }
    if (state->R__prev != NULL) {
        fcfree(state->R__prev);
        fcfree(state->R__head);
        state->R__prev = state->R__head = NULL;
    }
#endif /* DYN_ALLOC */
}

/* ===========================================================================
 * Set match_start to the longest match starting at the given string and
 * return its length. Matches shorter or equal to prev_length are discarded,
 * in which case the result is equal to prev_length and match_start is
 * garbage.
 * IN assertions: cur_match is the head of the hash chain for the current
 *   string (strstart) and its distance is <= MAX_DIST, and prev_length >= 1
 */
#ifndef ASMV
/* For MSDOS, OS/2 and 386 Unix, an optimized version is in match.asm or
 * match.s. The code is functionally equivalent, so you can use the C version
 * if desired.  A 68000 version is in amiga/match_68.a -- this could be used
 * with other 68000 based systems such as Macintosh with a little effort.
 */
int R__longest_match(bits_internal_state *state, IPos cur_match)
    /* IPos cur_match; */                       /* current match */
{
    unsigned chain_length = state->R__max_chain_length;   /* max hash chain length */
    register uch *scan = state->R__window + state->R__strstart;     /* current string */
    register uch *match;                        /* matched string */
    register int len;                           /* length of current match */
    int best_len = state->R__prev_length;              /* best match length so far */
    IPos limit = state->R__strstart > (IPos)MAX_DIST ? state->R__strstart - (IPos)MAX_DIST : NIL;
    /* Stop when cur_match becomes <= limit. To simplify the code,
     * we prevent matches with the string of window index 0.
     */

/* The code is optimized for HASH_BITS >= 8 and MAX_MATCH-2 multiple of 16.
 * It is easy to get rid of this optimization if necessary.
 */
#if HASH_BITS < 8 || MAX_MATCH != 258
   error: Code too clever
#endif

#ifdef UNALIGNED_OK
    /* Compare two bytes at a time. Note: this is not always beneficial.
     * Try with and without -DUNALIGNED_OK to check.
     */
    register uch *strend = state->R__window + state->R__strstart + MAX_MATCH - 1;
    register ush scan_start = *(ush*)scan;
    register ush scan_end   = *(ush*)(scan+best_len-1);
#else
    register uch *strend = state->R__window + state->R__strstart + MAX_MATCH;
    register uch scan_end1  = scan[best_len-1];
    register uch scan_end   = scan[best_len];
#endif

    /* Do not waste too much time if we already have a good match: */
    if (state->R__prev_length >= state->R__good_match) {
        chain_length >>= 2;
    }
    Assert(state->R__strstart <= state->R__window_size-MIN_LOOKAHEAD, "insufficient lookahead");

    do {
        Assert(cur_match < state->R__strstart, "no future");
        match = state->R__window + cur_match;

        /* Skip to next match if the match length cannot increase
         * or if the match length is less than 2:
         */
#if (defined(UNALIGNED_OK) && MAX_MATCH == 258)
        /* This code assumes sizeof(unsigned short) == 2. Do not use
         * UNALIGNED_OK if your compiler uses a different size.
         */
        if (*(ush*)(match+best_len-1) != scan_end ||
            *(ush*)match != scan_start) continue;

        /* It is not necessary to compare scan[2] and match[2] since they are
         * always equal when the other bytes match, given that the hash keys
         * are equal and that HASH_BITS >= 8. Compare 2 bytes at a time at
         * strstart+3, +5, ... up to strstart+257. We check for insufficient
         * lookahead only every 4th comparison; the 128th check will be made
         * at strstart+257. If MAX_MATCH-2 is not a multiple of 8, it is
         * necessary to put more guard bytes at the end of the window, or
         * to check more often for insufficient lookahead.
         */
        scan++, match++;
        do {
        } while (*(ush*)(scan+=2) == *(ush*)(match+=2) &&
                 *(ush*)(scan+=2) == *(ush*)(match+=2) &&
                 *(ush*)(scan+=2) == *(ush*)(match+=2) &&
                 *(ush*)(scan+=2) == *(ush*)(match+=2) &&
                 scan < strend);
        /* The funny "do {}" generates better code on most compilers */

        /* Here, scan <= window+state->R__strstart+257 */
        Assert(scan <= state->R__window+(unsigned)(state->R__window_size-1), "wild scan");
        if (*scan == *match) scan++;

        len = (MAX_MATCH - 1) - (int)(strend-scan);
        scan = strend - (MAX_MATCH-1);

#else /* UNALIGNED_OK */

        if (match[best_len]   != scan_end  ||
            match[best_len-1] != scan_end1 ||
            *match            != *scan     ||
            *++match          != scan[1])      continue;

        /* The check at best_len-1 can be removed because it will be made
         * again later. (This heuristic is not always a win.)
         * It is not necessary to compare scan[2] and match[2] since they
         * are always equal when the other bytes match, given that
         * the hash keys are equal and that HASH_BITS >= 8.
         */
        scan += 2, match++;

        /* We check for insufficient lookahead only every 8th comparison;
         * the 256th check will be made at strstart+258.
         */
        do {
        } while (*++scan == *++match && *++scan == *++match &&
                 *++scan == *++match && *++scan == *++match &&
                 *++scan == *++match && *++scan == *++match &&
                 *++scan == *++match && *++scan == *++match &&
                 scan < strend);

        len = MAX_MATCH - (int)(strend - scan);
        scan = strend - MAX_MATCH;

#endif /* UNALIGNED_OK */

        if (len > best_len) {
            state->R__match_start = cur_match;
            best_len = len;
            if (len >= state->R__nice_match) break;
#ifdef UNALIGNED_OK
            scan_end = *(ush*)(scan+best_len-1);
#else
            scan_end1  = scan[best_len-1];
            scan_end   = scan[best_len];
#endif
        }
    } while ((cur_match = state->R__prev[cur_match & WMASK]) > limit
             && --chain_length != 0);

    return best_len;
}
#endif /* ASMV */

#ifdef DEBUG
/* ===========================================================================
 * Check that the match at match_start is indeed a match.
 */
local int check_match(IPos start, IPos match, int length)
{
    /* check that the match is indeed a match */
    if (memcmp((char*)state->R__window + match,
                (char*)state->R__window + start, length) != EQUAL) {
        fprintf(stderr,
            " start %d, match %d, length %d\n",
            start, match, length);
        R__error("invalid match");
        return 1;
    }
/*
    if (verbose > 1) {
        fprintf(stderr,"\\[%d,%d]", start-match, length);
        do { putc(state->R__window[start++], stderr); } while (--length != 0);
    }
*/
    return 0;
}
#else
#  define check_match(start, match, length)
#endif

/* ===========================================================================
 * Fill the window when the lookahead becomes insufficient.
 * Updates strstart and lookahead, and sets state->eofile if end of input file.
 *
 * IN assertion: state->lookahead < MIN_LOOKAHEAD && strstart + state->lookahead > 0
 * OUT assertions: at least one byte has been read, or state->eofile is set;
 *    file reads are performed for at least two bytes (required for the
 *    translate_eol option).
 */
local void R__fill_window(bits_internal_state *state)
{
    register unsigned n, m;
    unsigned more = (unsigned)(state->R__window_size - (ulg)state->lookahead - (ulg)state->R__strstart);
    /* Amount of free space at the end of the window. */

    /* If the window is almost full and there is insufficient lookahead,
     * move the upper half to the lower one to make room in the upper half.
     */
    if (more == (unsigned)EOF) {
        /* Very unlikely, but possible on 16 bit machine if strstart == 0
         * and state->lookahead == 1 (input done one byte at time)
         */
        more--;

    /* For MMAP or BIG_MEM, the whole input file is already in memory
     * so we must not perform sliding. We must however call file_read
     * in order to compute the crc, update state->lookahead and possibly set state->eofile.
     */
    } else if (state->R__strstart >= WSIZE+MAX_DIST && state->sliding) {

        /* By the IN assertion, the window is not empty so we can't confuse
         * more == 0 with more == 64K on a 16 bit machine.
         */
        memcpy((char*)state->R__window, (char*)state->R__window+WSIZE, (unsigned)WSIZE);
        state->R__match_start -= WSIZE;
        state->R__strstart    -= WSIZE; /* we now have strstart >= MAX_DIST: */

        state->R__block_start -= (long) WSIZE;

        for (n = 0; n < HASH_SIZE; n++) {
            m = state->R__head[n];
            state->R__head[n] = (Pos)(m >= WSIZE ? m-WSIZE : NIL);
        }
        for (n = 0; n < WSIZE; n++) {
            m = state->R__prev[n];
            state->R__prev[n] = (Pos)(m >= WSIZE ? m-WSIZE : NIL);
            /* If n is not on any hash chain, prev[n] is garbage but
             * its value will never be used.
             */
        }
        more += WSIZE;
        //if (verbose) putc('.', stderr);
    }
    /* At this point, more >= 2 */
    if (!state->eofile) {
        n = R__mem_read(state,(char*)state->R__window+state->R__strstart+state->lookahead, more);
        if (n == 0 || n == (unsigned)EOF) {
            state->eofile = 1;
        } else {
            state->lookahead += n;
        }
    }
}

/* ===========================================================================
 * Flush the current block, with given end-of-file flag.
 * IN assertion: strstart is set to the end of the current match.
 */
#define FLUSH_BLOCK(eof) \
   R__flush_block(state, state->R__block_start >= 0L ? (char*)&state->R__window[(unsigned)state->R__block_start] : \
                (char*)NULL, (long)state->R__strstart - state->R__block_start, (eof),errorflag)

/* ===========================================================================
 * Processes a new input file and return its compressed length. This
 * function does not perform lazy evaluationof matches and inserts
 * new strings in the dictionary only for unmatched strings or for short
 * matches. It is used only for the fast compression options.
 */
local ulg R__Deflate_fast(bits_internal_state *state,int *errorflag)
{
    IPos hash_head; /* head of the hash chain */
    int flush;      /* set if current block must be flushed */
    unsigned match_length = 0;  /* length of best match */

    state->R__prev_length = MIN_MATCH-1;
    while (state->lookahead != 0) {
        /* Insert the string window[strstart .. strstart+2] in the
         * dictionary, and set hash_head to the head of the hash chain:
         */
        INSERT_STRING(state->R__strstart, hash_head);

        /* Find the longest match, discarding those <= prev_length.
         * At this point we have always match_length < MIN_MATCH
         */
        if (hash_head != NIL && state->R__strstart - hash_head <= MAX_DIST) {
            /* To simplify the code, we prevent matches with the string
             * of window index 0 (in particular we have to avoid a match
             * of the string with itself at the start of the input file).
             */
            match_length = R__longest_match (state, hash_head);
            /* R__longest_match() sets match_start */
            if (match_length > state->lookahead) match_length = state->lookahead;
        }
        if (match_length >= MIN_MATCH) {
            check_match(state->R__strstart, state->R__match_start, match_length);

            flush = R__ct_tally(state,state->R__strstart-state->R__match_start, match_length - MIN_MATCH);

            state->lookahead -= match_length;

            /* Insert new strings in the hash table only if the match length
             * is not too large. This saves time but degrades compression.
             */
            if (match_length <= max_insert_length) {
                match_length--; /* string at strstart already in hash table */
                do {
                    state->R__strstart++;
                    INSERT_STRING(state->R__strstart, hash_head);
                    /* strstart never exceeds WSIZE-MAX_MATCH, so there are
                     * always MIN_MATCH bytes ahead. If state->lookahead < MIN_MATCH
                     * these bytes are garbage, but it does not matter since
                     * the next lookahead bytes will be emitted as literals.
                     */
                } while (--match_length != 0);
                state->R__strstart++;
            } else {
                state->R__strstart += match_length;
                match_length = 0;
                state->ins_h = state->R__window[state->R__strstart];
                UPDATE_HASH(state->ins_h, state->R__window[state->R__strstart+1]);
#if MIN_MATCH != 3
                Call UPDATE_HASH() MIN_MATCH-3 more times
#endif
            }
        } else {
            /* No match, output a literal byte */
            Tracevv((stderr,"%c",state->R__window[state->R__strstart]));
            flush = R__ct_tally (state, 0, state->R__window[state->R__strstart]);
            state->lookahead--;
            state->R__strstart++;
        }
        if (flush) FLUSH_BLOCK(0), state->R__block_start = state->R__strstart;

        /* Make sure that we always have enough lookahead, except
         * at the end of the input file. We need MAX_MATCH bytes
         * for the next match, plus MIN_MATCH bytes to insert the
         * string following the next match.
         */
        while (state->lookahead < MIN_LOOKAHEAD && !state->eofile) R__fill_window(state);

    }
    return FLUSH_BLOCK(1); /* eof */
}

/* ===========================================================================
 * Same as above, but achieves better compression. We use a lazy
 * evaluation for matches: a match is finally adopted only if there is
 * no better match at the next window position.
 */
ulg R__Deflate(bits_internal_state *state,int *errorflag)
{
    IPos hash_head;          /* head of hash chain */
    IPos R__prev_match;      /* previous match */
    int flush;               /* set if current block must be flushed */
    int match_available = 0; /* set if previous match exists */
    register unsigned match_length = MIN_MATCH-1; /* length of best match */
#ifdef DEBUG
    /* extern ulg R__isize; */ /* byte length of input file, for debug only */
#endif

    if (gCompressionLevel <= 3) return R__Deflate_fast(state,errorflag); /* optimized for speed */

    /* Process the input block. */
    while (state->lookahead != 0) {

        /* Insert the string window[strstart .. strstart+2] in the
         * dictionary, and set hash_head to the head of the hash chain:
         */
        INSERT_STRING(state->R__strstart, hash_head);

        /* Find the longest match, discarding those <= prev_length.
         */
        state->R__prev_length = match_length, R__prev_match = state->R__match_start;
        match_length = MIN_MATCH-1;

        if (hash_head != NIL && state->R__prev_length < state->max_lazy_match &&
            state->R__strstart - hash_head <= MAX_DIST) {
            /* To simplify the code, we prevent matches with the string
             * of window index 0 (in particular we have to avoid a match
             * of the string with itself at the start of the input file).
             */
            match_length = R__longest_match (state,hash_head);
            /* R__longest_match() sets match_start */
            if (match_length > state->lookahead) match_length = state->lookahead;

            /* Ignore a length 3 match if it is too distant: */
            if (match_length == MIN_MATCH && state->R__strstart-state->R__match_start > TOO_FAR){
                /* If prev_match is also MIN_MATCH, match_start is garbage
                 * but we will ignore the current match anyway.
                 */
                match_length--;
            }
        }
        /* If there was a match at the previous step and the current
         * match is not better, output the previous match:
         */
        if (state->R__prev_length >= MIN_MATCH && match_length <= state->R__prev_length) {

            check_match(state->R__strstart-1, R__prev_match, state->R__prev_length);

            flush = R__ct_tally(state,state->R__strstart-1-R__prev_match, state->R__prev_length - MIN_MATCH);

            /* Insert in hash table all strings up to the end of the match.
             * strstart-1 and strstart are already inserted.
             */
            state->lookahead -= state->R__prev_length-1;
            state->R__prev_length -= 2;
            do {
                state->R__strstart++;
                INSERT_STRING(state->R__strstart, hash_head);
                /* strstart never exceeds WSIZE-MAX_MATCH, so there are
                 * always MIN_MATCH bytes ahead. If lookahead < MIN_MATCH
                 * these bytes are garbage, but it does not matter since the
                 * next lookahead bytes will always be emitted as literals.
                 */
            } while (--state->R__prev_length != 0);
            match_available = 0;
            match_length = MIN_MATCH-1;
            state->R__strstart++;
            if (flush) FLUSH_BLOCK(0), state->R__block_start = state->R__strstart;

        } else if (match_available) {
            /* If there was no match at the previous position, output a
             * single literal. If there was a match but the current match
             * is longer, truncate the previous match to a single literal.
             */
            Tracevv((stderr,"%c",state->R__window[state->R__strstart-1]));
            if (R__ct_tally (state, 0, state->R__window[state->R__strstart-1])) {
                FLUSH_BLOCK(0), state->R__block_start = state->R__strstart;
            }
            state->R__strstart++;
            state->lookahead--;
        } else {
            /* There is no previous match to compare with, wait for
             * the next step to decide.
             */
            match_available = 1;
            state->R__strstart++;
            state->lookahead--;
        }
#ifdef DEBUG
        Assert (state->R__strstart <= state->R__isize && state->lookahead <= state->R__isize, "a bit too far");
#endif

        /* Make sure that we always have enough lookahead, except
         * at the end of the input file. We need MAX_MATCH bytes
         * for the next match, plus MIN_MATCH bytes to insert the
         * string following the next match.
         */
        while (state->lookahead < MIN_LOOKAHEAD && !state->eofile) R__fill_window(state);
    }
    if (match_available) R__ct_tally (state, 0, state->R__window[state->R__strstart-1]);

    return FLUSH_BLOCK(1); /* eof */
}

