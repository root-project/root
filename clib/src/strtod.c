/* @(#)root/clib:$Name$:$Id$ */
/* Author: */

#if defined(WIN32) && defined(SYMANTEC)

/*
** Copyright (C) 1993 DJ Delorie, 24 Kirsten Ave, Rochester NH 03867-2954
**
** This file is distributed under the terms listed in the document
** "copying.dj", available from DJ Delorie at the address above.
** A copy of "copying.dj" should accompany this file; if not, a copy
** should be available from where this file was obtained.  This file
** may not be distributed without a verbatim copy of "copying.dj".
**
** This file is distributed WITHOUT ANY WARRANTY; without even the implied
** warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <math.h>
#include <stdlib.h>

double strtod(const char *s, char **sret)
{
  double r;     /* result */
  int e;        /* exponent */
  double d;     /* scale */
  int sign;     /* +- 1.0 */
  int esign;
  int i;
  int flags=0;

  r = 0.0;
  sign = 1.0;
  e = 0;
  esign = 1;

  while ((*s == ' ') || (*s == '\t'))
    s++;

  if (*s == '+')
    s++;
  else if (*s == '-')
  {
    sign = -1;
    s++;
  }

  while ((*s >= '0') && (*s <= '9'))
  {
    flags |= 1;
    r *= 10.0;
    r += *s - '0';
    s++;
  }

  if (*s == '.')
  {
    d = 0.1;
    s++;
    while ((*s >= '0') && (*s <= '9'))
    {
      flags |= 2;
      r += d * (*s - '0');
      s++;
      d *= 0.1;
    }
  }

  if (flags == 0)
  {
    if (sret) *sret = (char *)s;
    return 0;
  }

  if ((*s == 'e') || (*s == 'E'))
  {
    s++;
    if (*s == '+')
      s++;
    else if (*s == '-')
    {
      *s++;
      esign = -1;
    }
    if ((*s < '0') || (*s > '9'))
    {
      if (sret) *sret = (char *)s;
      return r;
    }

    while ((*s >= '0') && (*s <= '9'))
    {
      e *= 10.0;
      e += *s - '0';
      s++;
    }
  }

  if (esign < 0)
    for (i = 1; i <= e; i++)
      r *= 0.1;
  else
    for (i = 1; i <= e; i++)
      r *= 10.0;

  if (sret) *sret = (char *)s;
  return r * sign;
}
#else

#ifndef __GNUC__
/* Prevent "empty translation unit" warnings. */
static char file_intentionally_empty = 69;
#endif

#endif
