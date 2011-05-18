/* config.h for CMake builds */

/* #undef HAVE_DIRENT_H */
/* #undef HAVE_UNISTD_H */
#define HAVE_SYS_STAT_H
#define HAVE_SYS_TYPES_H
/* #undef HAVE_TYPE_TRAITS_H */
/* #undef HAVE_BITS_TYPE_TRAITS_H */

/* #undef HAVE_BCOPY */
#define HAVE_MEMMOVE
#define HAVE_STRERROR

#define PCRE_STATIC

/* #undef SUPPORT_UTF8 */
/* #undef SUPPORT_UCP */
/* #undef EBCDIC */
/* #undef BSR_ANYCRLF */
/* #undef NO_RECURSE */

#define NEWLINE			10
#define POSIX_MALLOC_THRESHOLD	10
#define LINK_SIZE		2
#define MATCH_LIMIT		10000000
#define MATCH_LIMIT_RECURSION	MATCH_LIMIT

#define MAX_NAME_SIZE	32
#define MAX_NAME_COUNT	10000

/* end config.h for CMake builds */
