#ifndef G__LIMITS_H
#define G__LIMITS_H
#define 	CHAR_BIT (8)
#define 	CHAR_MAX (127)
#define 	CHAR_MIN (-128)
#define 	INT_MAX (2147483647)
#define 	INT_MIN (-2147483648)
#define 	SCHAR_MAX (127)
#define 	SCHAR_MIN (-128)
#define 	SHRT_MAX (32767)
#define 	SHRT_MIN (-32768)
#define 	UCHAR_MAX (255U)
#if !defined(G__EXTERNAL_CPP)
# if sizeof(long) == sizeof(int)
#  define 	LONG_MAX (2147483647)
# else
#  define 	LONG_MAX (9223372036854775807)
# endif
#else
# ifdef __LONG_MAX__
#  define     LONG_MAX  __LONG_MAX__
# else
#  define     LONG_MAX  2147483647
# endif
#endif
#define         LONG_MIN (-LONG_MAX - 1)
const unsigned int  	UINT_MAX = 1u + 2u*(unsigned int)(INT_MAX);
const unsigned long 	ULONG_MAX = 1u + 2u*(unsigned long)(LONG_MAX);
#define 	USHRT_MAX (65535U)
#endif
