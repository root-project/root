/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file stdstr.h
 ************************************************************************
 * Description:
 *  Stub file for making ANSI C standard structs
 ************************************************************************
 * Copyright(c) 1991~1999,   Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef G__STDSTRUCT
#define G__STDSTRUCT

#ifndef __CINT__

#include <locale.h>
#include <time.h>
#include <stdlib.h>

#else

/* Structure giving information about numeric and monetary notation.  */
struct lconv
{
  /* Numeric (non-monetary) information.  */
  char *currency_symbol;	/* Local currency symbol.  */
  char *decimal_point;		/* Decimal point character.  */
  char frac_digits;		/* Local fractional digits.  */
  char *grouping;
  char *int_curr_symbol;
  char *mon_decimal_point;	/* Decimal point character.  */
  char *mon_grouping;		/* Like `grouping' element (above).  */
  /* char *mon_thousands_sep;*/	/* Thousands separator.  */
  char n_cs_precedes;
  char n_sep_by_space;
  char n_sign_posn;
  char *negative_sign;		/* Sign for negative values.  */
  char p_cs_precedes;
  char p_sep_by_space;
  char p_sign_posn;
  char *positive_sign;		/* Sign for positive values.  */
  char *thousands_sep;		/* Thousands separator.  */

 /* char int_frac_digits; */ /* Int'l fractional digits.  */
};


struct tm {
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
};


/* Returned by `div'.  */
typedef struct {
  int quot;	/* Quotient.  */
  int rem;	/* Remainder.  */
} div_t;

/* Returned by `ldiv'.  */
typedef struct {
  long quot;	/* Quotient.  */
  long rem;	/* Remainder.  */
} ldiv_t;

#endif /* __CINT__ */

#endif
