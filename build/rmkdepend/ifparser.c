/*
 * $XConsortium: ifparser.c /main/10 1996/09/28 16:15:18 rws $
 * $XFree86: xc/config/makedepend/ifparser.c,v 3.6 1996/12/30 13:57:55 dawes Exp $
 *
 * Copyright 1992 Network Computing Devices, Inc.
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted, provided
 * that the above copyright notice appear in all copies and that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Network Computing Devices may not be
 * used in advertising or publicity pertaining to distribution of the software
 * without specific, written prior permission.  Network Computing Devices makes
 * no representations about the suitability of this software for any purpose.
 * It is provided ``as is'' without express or implied warranty.
 *
 * NETWORK COMPUTING DEVICES DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS,
 * IN NO EVENT SHALL NETWORK COMPUTING DEVICES BE LIABLE FOR ANY SPECIAL,
 * INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 *
 * Author:  Jim Fulton
 *          Network Computing Devices, Inc.
 *
 * Simple if statement processor
 *
 * This module can be used to evaluate string representations of C language
 * if constructs.  It accepts the following grammar:
 *
 *     EXPRESSION := VALUE
 *     | VALUE  BINOP EXPRESSION
 *     | VALUE '?' EXPRESSION ':' EXPRESSION
 *
 *     VALUE  := '('  EXPRESSION  ')'
 *     | '!'  VALUE
 *     | '-'  VALUE
 *     | '~'  VALUE
 *     | 'defined'  '('  variable  ')'
 *     | 'defined'  variable
 *     | # variable '(' variable-list ')'
 *     | variable
 *     | number
 *
 *     BINOP  := '*' |  '/' |  '%'
 *     | '+' |  '-'
 *     | '<<' |  '>>'
 *     | '<' |  '>' |  '<='  |  '>='
 *     | '==' |  '!='
 *     | '&' |  '^'  |  '|'
 *     | '&&' |  '||'
 *
 * The normal C order of precedence is supported.
 *
 *
 * External Entry Points:
 *
 *     ParseIfExpression  parse a string for #if
 */

#include "ifparser.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

/****************************************************************************
     Internal Macros and Utilities for Parser
 ****************************************************************************/

#define DO(val) if (!(val)) return NULL
#define CALLFUNC(ggg,fff) (*((ggg)->funcs.fff))
#define SKIPSPACE(ccc) while (isspace(*ccc)) ccc++
#define isvarfirstletter(ccc) (isalpha(ccc) || (ccc) == '_')


static const char *
parse_variable(g, cp, varp)
IfParser *g;
const char *cp;
const char **varp;
{
   SKIPSPACE(cp);

   if (!isvarfirstletter(*cp))
      return CALLFUNC(g, handle_error)(g, cp, "variable name");

   *varp = cp;
   /* EMPTY */
   for (cp++; isalnum(*cp) || *cp == '_'; cp++) ;
   return cp;
}


static const char *
parse_number(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   SKIPSPACE(cp);

   if (!isdigit(*cp))
      return CALLFUNC(g, handle_error)(g, cp, "number");

   *valp = strtol(cp, &cp, 0);
   /* skip trailing qualifiers */
   while (*cp == 'U' || *cp == 'u' || *cp == 'L' || *cp == 'l') cp++;
#if 0
   *valp = atoi(cp);
   /* EMPTY */
   for (cp++; isdigit(*cp); cp++) ;
#endif
   return cp;
}

static const char *
parse_character(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   char val;
   if (g) { }   /* use argument */

   SKIPSPACE(cp);
   if (*cp == '\\')
      switch (cp[1]) {
         case 'n':
            val = '\n';
            break;
         case 't':
            val = '\t';
            break;
         case 'v':
            val = '\v';
            break;
         case 'b':
            val = '\b';
            break;
         case 'r':
            val = '\r';
            break;
         case 'f':
            val = '\f';
            break;
         case 'a':
            val = '\a';
            break;
         case '\\':
            val = '\\';
            break;
         case '?':
            val = '\?';
            break;
         case '\'':
            val = '\'';
            break;
         case '\"':
            val = '\"';
            break;
         case 'x':
            val = (char) strtol(cp + 2, NULL, 16);
            break;
         default:
            val = (char) strtol(cp + 1, NULL, 8);
            break;
      }
   else
      val = *cp;
   while (*cp != '\'') cp++;
   *valp = (long) val;
   return cp;
}

static const char *
parse_value(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   const char *var;

   *valp = 0;
   var = 0;

   SKIPSPACE(cp);
   if (!*cp)
      return cp;

   switch (*cp) {
      case '(':
         DO(cp = ParseIfExpression(g, cp + 1, valp));
         SKIPSPACE(cp);
         if (*cp != ')')
            return CALLFUNC(g, handle_error)(g, cp, ")");

         return cp + 1;   /* skip the right paren */

      case '!':
         DO(cp = parse_value(g, cp + 1, valp));
         *valp = !(*valp);
         return cp;

      case '-':
         DO(cp = parse_value(g, cp + 1, valp));
         *valp = -(*valp);
         return cp;

      case '~':
         DO(cp = parse_value(g, cp + 1, valp));
         *valp = ~(*valp);
         return cp;

      case '#':
         DO(cp = parse_variable(g, cp + 1, &var));
         SKIPSPACE(cp);
         if (*cp != '(')
            return CALLFUNC(g, handle_error)(g, cp, "(");
         do {
            DO(cp = parse_variable(g, cp + 1, &var));
            SKIPSPACE(cp);
         } while (*cp && *cp != ')');
         if (*cp != ')')
            return CALLFUNC(g, handle_error)(g, cp, ")");
         *valp = 1; /* XXX */
         return cp + 1;

      case '\'':
         DO(cp = parse_character(g, cp + 1, valp));
         if (*cp != '\'')
            return CALLFUNC(g, handle_error)(g, cp, "'");
         return cp + 1;

      case 'd':
         if (strncmp(cp, "defined", 7) == 0 && !isalnum(cp[7])) {
            int paren = 0;
            int len;

            cp += 7;
            SKIPSPACE(cp);
            if (*cp == '(') {
               paren = 1;
               cp++;
            }
            DO(cp = parse_variable(g, cp, &var));
            len = cp - var;
            SKIPSPACE(cp);
            if (paren && *cp != ')')
               return CALLFUNC(g, handle_error)(g, cp, ")");
            *valp = (*(g->funcs.eval_defined))(g, var, len);
            return cp + paren;  /* skip the right paren */
         }
         /* fall out */
      case '_':
         if (strncmp(cp, "__has_", 6) == 0) {
            cp += 6;
            while (isalnum(*cp) || *cp == '_')
               ++cp;

            int paren = 0;
            SKIPSPACE(cp);
            if (*cp == '(') {
               paren = 1;
               cp++;
            }
            DO(cp = parse_variable(g, cp, &var));
            SKIPSPACE(cp);
            if (paren && *cp != ')')
               return CALLFUNC(g, handle_error)(g, cp, ")");
            *valp = 0; /* no features / attributes by default */
           return cp + paren;  /* skip the right paren */            
         } else if (strncmp(cp, "__GNUC_PREREQ", 13) == 0 && (cp += 7)) {
            cp += 13;
            while (*cp != '(' && *cp) ++cp;
            while (*cp != ')' && *cp) ++cp;
            *valp = 0; // assume old GCC
            return cp;
         }

         /* fall out */
   }

   if (isdigit(*cp)) {
      DO(cp = parse_number(g, cp, valp));
   } else if (!isvarfirstletter(*cp))
      return CALLFUNC(g, handle_error)(g, cp, "variable or number");
   else {
      DO(cp = parse_variable(g, cp, &var));
      *valp = (*(g->funcs.eval_variable))(g, var, cp - var);
   }

   return cp;
}



static const char *
parse_product(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_value(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '*':
         DO(cp = parse_product(g, cp + 1, &rightval));
         *valp = (*valp * rightval);
         break;

      case '/':
         DO(cp = parse_product(g, cp + 1, &rightval));
         if (rightval)
            *valp = (*valp / rightval);
         break;

      case '%':
         DO(cp = parse_product(g, cp + 1, &rightval));
         if (rightval)
            *valp = (*valp % rightval);
         break;
   }
   return cp;
}


static const char *
parse_sum(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_product(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '+':
         DO(cp = parse_sum(g, cp + 1, &rightval));
         *valp = (*valp + rightval);
         break;

      case '-':
         DO(cp = parse_sum(g, cp + 1, &rightval));
         *valp = (*valp - rightval);
         break;
   }
   return cp;
}


static const char *
parse_shift(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_sum(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '<':
         if (cp[1] == '<') {
            DO(cp = parse_shift(g, cp + 2, &rightval));
            *valp = (*valp << rightval);
         }
         break;

      case '>':
         if (cp[1] == '>') {
            DO(cp = parse_shift(g, cp + 2, &rightval));
            *valp = (*valp >> rightval);
         }
         break;
   }
   return cp;
}


static const char *
parse_inequality(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_shift(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '<':
         if (cp[1] == '=') {
            DO(cp = parse_inequality(g, cp + 2, &rightval));
            *valp = (*valp <= rightval);
         } else {
            DO(cp = parse_inequality(g, cp + 1, &rightval));
            *valp = (*valp < rightval);
         }
         break;

      case '>':
         if (cp[1] == '=') {
            DO(cp = parse_inequality(g, cp + 2, &rightval));
            *valp = (*valp >= rightval);
         } else {
            DO(cp = parse_inequality(g, cp + 1, &rightval));
            *valp = (*valp > rightval);
         }
         break;
   }
   return cp;
}


static const char *
parse_equality(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_inequality(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '=':
         if (cp[1] == '=')
            cp++;
         DO(cp = parse_equality(g, cp + 1, &rightval));
         *valp = (*valp == rightval);
         break;

      case '!':
         if (cp[1] != '=')
            break;
         DO(cp = parse_equality(g, cp + 2, &rightval));
         *valp = (*valp != rightval);
         break;
   }
   return cp;
}


static const char *
parse_band(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_equality(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '&':
         if (cp[1] != '&') {
            DO(cp = parse_band(g, cp + 1, &rightval));
            *valp = (*valp & rightval);
         }
         break;
   }
   return cp;
}


static const char *
parse_bxor(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_band(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '^':
         DO(cp = parse_bxor(g, cp + 1, &rightval));
         *valp = (*valp ^ rightval);
         break;
   }
   return cp;
}


static const char *
parse_bor(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_bxor(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '|':
         if (cp[1] != '|') {
            DO(cp = parse_bor(g, cp + 1, &rightval));
            *valp = (*valp | rightval);
         }
         break;
   }
   return cp;
}


static const char *
parse_land(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_bor(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '&':
         if (cp[1] != '&')
            return CALLFUNC(g, handle_error)(g, cp, "&&");
         DO(cp = parse_land(g, cp + 2, &rightval));
         *valp = (*valp && rightval);
         break;
   }
   return cp;
}


static const char *
parse_lor(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long rightval;

   DO(cp = parse_land(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '|':
         if (cp[1] != '|')
            return CALLFUNC(g, handle_error)(g, cp, "||");
         DO(cp = parse_lor(g, cp + 2, &rightval));
         *valp = (*valp || rightval);
         break;
   }
   return cp;
}


static const char *
parse_cond(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   long trueval, falseval;

   DO(cp = parse_lor(g, cp, valp));
   SKIPSPACE(cp);

   switch (*cp) {
      case '?':
         DO(cp = parse_cond(g, cp + 1, &trueval));
         SKIPSPACE(cp);
         if (*cp != ':')
            return CALLFUNC(g, handle_error)(g, cp, ":");
         DO(cp = parse_cond(g, cp + 1, &falseval));
         *valp = (*valp ? trueval : falseval);
         break;
   }
   return cp;
}


/****************************************************************************
        External Entry Points
 ****************************************************************************/

const char *
ParseIfExpression(g, cp, valp)
IfParser *g;
const char *cp;
long *valp;
{
   return parse_cond(g, cp, valp);
}
