// @(#)root/base:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Author:    Allen I. Holub                                            //
//                                                                      //
// (c) C Gazette. May be used freely as long as author and publication  //
// are acknowledged.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <ctype.h>
#include <string.h>


#include "Match.h"


// Metacharacters in the input:
#define BOL     '^'             // start-of-line anchor
#define EOL     '$'             // end-of-line anchor
#define ANY     '.'             // matches any character
#define CCL     '['             // start a character class
#define CCLEND  ']'             // end a character class
#define NCCL    '^'             // negates character class if 1st character
#define CLOSURE '*'             // Kleene closure (matches 0 or more)
#define PCLOSE  '+'             // Positive closure (1 or more)
#define OPT     '?'             // Optional closure (0 or 1)

enum Eaction {                  // These are put in the pattern string
                                // to represent metacharacters.
   kBOL    = (0x8000 | '^'),
   kEOL    = (0x8000 | '$'),
   kANY    = (0x8000 | '.'),
   kCCL    = (0x8000 | '['),
   kOPT    = (0x8000 | '?'),
   kCLOSE  = (0x8000 | '*'),
   kPCLOSE = (0x8000 | '+'),
   kEND    = (0x8000 | 0)      // end of pattern

};

#if 1
#define ISHEXDIGIT(x) isxdigit((unsigned char)x)
#else
#define ISHEXDIGIT(x) (isdigit(x)                       \
                            || ('a'<=(x) && (x)<='f')   \
                            || ('A'<=(x) && (x)<='F')   )
#endif

#define ISOCTDIGIT(x) ('0'<=(x) && (x)<='7')

// ----------------------------------------------------------------------
#define MAPSIZE 16              // need this many Pattern_t elements for
                                // character class bit map

//______________________________________________________________________________
inline void ADVANCE(const Pattern_t*& pat)
{
   // Advance a pointer into the pattern template to the next pattern element,
   // this is a+1 for all pattern elements but kCCL, where you have to skip
   // past both the kCCL character and the bitmap that follows that character.

   if (*pat++ == (Pattern_t) kCCL)
      pat += MAPSIZE;
}

//
// Bitmap functions. Set bit b in the map and
// test bit b to see if it was set previously.
//

//______________________________________________________________________________
#ifdef SETBIT // from Rtypes.h
#undef SETBIT
#endif

static void SETBIT(unsigned char b, Pattern_t* map)
{
   map[b >> 4] |= 1 << (b & 0x0f);
}

//______________________________________________________________________________
static int TSTBIT(unsigned char b, const Pattern_t* map)
{
   return map[b >> 4] & (1 << (b & 0x0f));
}

// ----------------------------------------------------------------------

#define E_NONE       0          // Possible return values from pat_err
#define E_ILLEGAL    1          // Set in Makepat() to indicate prob-
#define E_NOMEM      2          // lems that came up while making the
#define E_PAT        3          // pattern template.

// ----------------------------------------------------------------------

static const char*  doccl(Pattern_t*, const char*);
static int          hex2bin(int);
static int          oct2bin(int);
static int          omatch(const char**, size_t*, const Pattern_t*, const char*);
static const char*  patcmp(const char*, size_t, const Pattern_t*, const char*);
static int          esc(const char**);

// ----------------------------------------------------------------------

//______________________________________________________________________________
int Makepat(const char*     exp,        // Regular expression
            Pattern_t*      pat,        // Assembled compiled pattern
            int             maxpat)     // Length of pat
{
   // Make a pattern template from the string pointed to by exp. Stop when
   // '\0' is found in exp.  The pattern template is assembled
   // in pat whose length is given by maxpat.
   //
   // Return:
   // E_ILLEGAL       Illegal input pattern.
   // E_NOMEM         out of memory.
   // E_PAT           pattern too long.

   Pattern_t* cur;           // pointer to current pattern element
   Pattern_t* prev;          // pointer to previous pattern element
   int        theError = E_ILLEGAL;

   if (!*exp)
      goto exit;

   if (*exp == CLOSURE || *exp == PCLOSE || *exp == OPT)
      goto exit;

   theError = E_NOMEM;
   if (!pat) goto exit;          // Check for bad pat

   prev = cur = pat;
   theError = E_PAT;

   while (*exp) {

      if (cur >= &pat[maxpat - 1]) goto exit;

      switch (*exp) {
      case ANY:
         *cur = (Pattern_t)kANY;
         prev = cur++;
         ++exp;
         break;

      case BOL:
         *cur = (cur == pat) ? (Pattern_t)kBOL : (unsigned char)*exp;
         prev = cur++;
         ++exp;
         break;

      case EOL:
         *cur = (!exp[1]) ? (Pattern_t)kEOL : (unsigned char)*exp;
         prev = cur++;
         ++exp;
         break;

      case CCL:
         if (((cur - pat) + MAPSIZE) >= maxpat)
            goto exit;              // not enough room for bit map
         prev = cur;
         *cur++ = (Pattern_t)kCCL;
         exp = doccl(cur, exp);
         if (*exp != CCLEND) goto exit;
         ++exp;
         cur += MAPSIZE;
         break;

      case OPT:
      case CLOSURE:
      case PCLOSE:
         switch (*prev) {
         case kBOL:
         case kEOL:
         case kOPT:
         case kPCLOSE:
         case kCLOSE:
            goto exit;
         }

         memmove( prev+1, prev, (cur-prev)*sizeof(*cur));
         *prev = (Pattern_t) (*exp == OPT) ?    kOPT :
                             (*exp == PCLOSE) ? kPCLOSE :
                                                kCLOSE;
         ++cur;
         ++exp;
         break;

      default:
         prev = cur;
         *cur++ = esc(&exp);
         break;
      }
   }

   *cur = (Pattern_t)kEND;
   theError = E_NONE;

exit:
   return theError;
}

//______________________________________________________________________________
const char *Matchs(const char*       str,
                   size_t            slen,
                   const Pattern_t*  pat,
                   const char**      startpat)
{
   // Match a string with a pattern.

   if (!pat) return 0;
   const char* endp = 0;
   if (*pat == (Pattern_t)kBOL) {
      // the rest has to match directly
      endp = patcmp(str, slen, pat+1, str);
   } else {
      // scoot along the string until it matches, or no more string
      const char* start = str;
      while ((endp = patcmp(str, slen, pat, start)) == 0 && slen != 0)
         ++str, --slen;
   }
   *startpat = str;
   return endp;
}

//______________________________________________________________________________
static const char *doccl(Pattern_t*  map, const char* src)
{
   // Set bits in the map corresponding to characters specified in the src
   // character class.

   int negative;

   ++src;                        // skip past the [
   negative = (*src == NCCL);
   if (negative)                 // check for negative ccl
      ++src;
   memset(map, 0, MAPSIZE*sizeof(*map)); // bitmap initially empty

   while (*src && *src != CCLEND) {
      int isCCL = (*src == CCL);  // support [] in pattern, don't put single [ before closing ]
      unsigned char first = esc(&src);
      SETBIT(first, map);
      if (isCCL && *src && *src == CCLEND) {
         first = esc(&src);
         SETBIT(first, map);
      }
      if (*src == '-' && src[1] && src[1] != CCLEND) {
         ++src;                    // skip to end-of-sequence char
         unsigned char last = esc(&src);
         if (first <= last)  while (first < last ) SETBIT(++first, map);
         else                while (last  < first) SETBIT(last++,  map);
      }
   }

   if (negative) {
      for (int i = MAPSIZE; --i >= 0;)
         *map++ ^= ~0;             // invert all bits
   }

   return src;
}

//______________________________________________________________________________
static const char *patcmp(const char*      str,
                          size_t           slen,
                          const Pattern_t* pat,
                          const char*      start)
{
   // Like strcmp, but compares str against pat. Each element of str is
   // compared with the template until either a mis-match is found or the end
   // of the template is reached. In the former case a 0 is returned; in the
   // latter, a pointer into str (pointing after the last character in the
   // matched pattern) is returned. start points at the first character in
   // the string, which might not be the same thing as str if the search
   // started in the middle of the string.

   if (!pat)                     // make sure pattern is valid
      return 0;

   while (*pat != (Pattern_t)kEND) {
      if (*pat == (Pattern_t)kOPT) {

         // Zero or one matches. It doesn't matter if omatch fails---it will
         // advance str past the character on success, though. Always advance
         // the pattern past both the kOPT and the operand.
         omatch(&str, &slen, ++pat, start);
         ADVANCE(pat);

      } else if (*pat != (Pattern_t)kCLOSE &&
                 *pat != (Pattern_t)kPCLOSE)    {

         // Do a simple match. Note that omatch() fails if there's still
         // something in pat but we're at end of string.

         if (!omatch(&str, &slen, pat, start))
            return 0;
         ADVANCE(pat);

      } else {                    // Process a Kleene or positive closure

         if (*pat++ == (Pattern_t)kPCLOSE)    // one match required
            if (!omatch(&str, &slen, pat, start))
               return 0;

         // Match as many as possible, zero is okay

         const char* bocl = str;           // beginning of closure string.
         while (slen && omatch(&str, &slen, pat, start))
            ;
         ADVANCE(pat);  // skip over the closure
         if (*pat == (Pattern_t)kEND)
            break;

         // 'str' now points to the character that made made us fail. Try to
         // process the rest of the string. If the character following the
         // closure could have been in the closure (as in the pattern "[a-z]*t")
         // the final 't' will be sucked up in the while loop. So, if the match
         // fails, back up a notch and try to match the rest of the string
         // again, repeating this process until we get back to the
         // beginning of the closure. The recursion goes as many levels
         // deep as there are closures in the pattern.

         const char* end;
         while ((end = patcmp(str, slen, pat, start)) == 0) {
            ++slen, --str;
            if (str < bocl) break;
         }
         return end;

      }  // closure
   }  // while (*pat != kEND)

   //
   // omatch() advances str to point at the next character to be matched. So
   // str points at the character following the last character matched when
   // you reach the end of the template. The exceptions are templates
   // containing only a BOLN or EOLN token. In these cases omatch doesn't
   // advance.
   //

   return str;
}

//______________________________________________________________________________
static int omatch(const char**      strp,
                  size_t*           slenp,
                  const Pattern_t*  pat,
                  const char*       start)
{
   // Match one pattern element, pointed at by pat, against the character at
   // **strp. Return 0 on a failure, 1 on success. *strp is advanced to skip
   // over the matched character on a successful match. Closure is handled one
   // level up by patcmp().
   //
   // "start" points at the character at the left edge of the line. This might
   // not be the same thing as *strp if the search is starting in the middle
   // of the string. An end-of- line anchor matches end of string only.

   switch (*pat) {
   // Match beginning of line
   case kBOL:   return (*strp == start);

   // Match end of line
   case kEOL:   return (*slenp == 0);

   // Notice: cases above don't advance.
   // Match any except newline
   case kANY: if (**strp == '\n') return 0;
      break;

   // Set match
   case kCCL: if (*slenp == 0 || !TSTBIT(**strp, pat + 1)) return 0;
      break;

   // Literal match
   default:    if (*slenp == 0 || (unsigned char) **strp != *pat)  return 0;
      break;
   }

   if (*slenp) {
      ++*strp;
      --*slenp;
   }
   return 2;
}

//______________________________________________________________________________
static int hex2bin(int c)
{
   // Convert the hex digit represented by 'c' to an int. 'c'
   // must be one of: 0123456789abcdefABCDEF

   return (isdigit(c) ? (c)-'0': ((toupper((unsigned char)c))-'A')+10)  & 0xf;
}

//______________________________________________________________________________
static int oct2bin(int c)
{
   // Convert the hex digit represented by 'c' to an int. 'c'
   // must be a digit in the range '0'-'7'.

   return ( ((c)-'0')  &  0x7 );
}

//______________________________________________________________________________
static int esc(const char** s)
{
   // Map escape sequences into their equivalent symbols. Return
   // the equivalent ASCII character. *s is advanced past the
   // escape sequence. If no escape sequence is present, the
   // current character is returned and the string is advanced by
   // one. The following are recognized:
   //
   //  \b     backspace
   //  \f     formfeed
   //  \n     newline
   //  \r     carriage return
   //  \s     space
   //  \t     tab
   //  \e     ASCII ESC character ('\033')
   //  \DDD   number formed of 1-3 octal digits
   //  \xDD   number formed of 1-2 hex digits
   //  \^C    C = any letter. Control code

   int rval;

   if (**s != '\\')
      rval = *((*s)++);
   else {
      ++(*s);                                 // Skip the backslash (\)
      switch (toupper((unsigned char)**s)) {
      case '\0':  rval = '\\';             break;
      case 'B':   rval = '\b';             break;
      case 'F':   rval = '\f';             break;
      case 'N':   rval = '\n';             break;
      case 'R':   rval = '\r';             break;
      case 'S':   rval = ' ';              break;
      case 'T':   rval = '\t';             break;
      case 'E':   rval = '\033';           break;

      case '^':
         rval = *++(*s) ;
         rval = toupper((unsigned char)rval) - '@';
         break;

      case 'X':
         rval = 0;
         ++(*s);
         if (ISHEXDIGIT(**s)) {
            rval  = hex2bin((unsigned char) *(*s)++);
         }
         if (ISHEXDIGIT(**s)) {
            rval <<= 4;
            rval  |= hex2bin((unsigned char) *(*s)++);
         }
         --(*s);
         break;

      default:
         if (!ISOCTDIGIT(**s))
            rval = **s;
         else {
            rval = oct2bin(*(*s)++);
            if (ISOCTDIGIT(**s)) {
               rval <<= 3;
               rval  |= oct2bin(*(*s)++);
            }
            if (ISOCTDIGIT(**s)) {
               rval <<= 3;
               rval  |= oct2bin(*(*s)++);
            }
            --(*s);
         }
         break;
      }
   ++(*s);
   }
   return (unsigned char)rval;
}
