/* glibconfig.h.win32 */
/* Handcrafted for Microsoft C and gcc -mno-cygwin ("mingw32"). */

#ifndef __G_LIBCONFIG_H__
#define __G_LIBCONFIG_H__

#include <gmacros.h>

#ifdef _MSC_VER
/* Make MSVC more pedantic, this is a recommended pragma list
 * from _Win32_Programming_ by Rector and Newcomer.
 */

#pragma warning(error:4002)
#pragma warning(error:4003)
#pragma warning(1:4010)
#pragma warning(error:4013)
#pragma warning(1:4016)
#pragma warning(error:4020)
#pragma warning(error:4021)
#pragma warning(error:4027)
#pragma warning(error:4029)
#pragma warning(error:4033)
#pragma warning(error:4035)
#pragma warning(error:4045)
#pragma warning(error:4047)
#pragma warning(error:4049)
#pragma warning(error:4053)
#pragma warning(error:4071)
#pragma warning(disable:4101)
#pragma warning(error:4150)

#pragma warning(disable:4244)	// No possible loss of data warnings
#pragma warning(disable:4305)   // No truncation from int to char warnings

#endif /* _MSC_VER */

#include <limits.h>
#include <float.h>

#define G_MINFLOAT	FLT_MIN
#define G_MAXFLOAT	FLT_MAX
#define G_MINDOUBLE	DBL_MIN
#define G_MAXDOUBLE	DBL_MAX
#define G_MINSHORT	SHRT_MIN
#define G_MAXSHORT	SHRT_MAX
#define G_MAXUSHORT	USHRT_MAX
#define G_MININT	INT_MIN
#define G_MAXINT	INT_MAX
#define G_MAXUINT	UINT_MAX
#define G_MINLONG	LONG_MIN
#define G_MAXLONG	LONG_MAX
#define G_MAXULONG	ULONG_MAX

G_BEGIN_DECLS

typedef signed char gint8;
typedef unsigned char guint8;
typedef signed short gint16;
typedef unsigned short guint16;
#define G_GINT16_FORMAT "hi"
#define G_GUINT16_FORMAT "hu"
typedef signed int gint32;
typedef unsigned int guint32;
#define G_GINT32_FORMAT "i"
#define G_GUINT32_FORMAT "u"

#define G_HAVE_GINT64 1

/* These are compiler specific */
#ifdef _MSC_VER
typedef __int64 gint64;
typedef unsigned __int64 guint64;
#define G_GINT64_CONSTANT(val)	(val##i64)
#elif __GNUC__
typedef long long gint64;
typedef unsigned long long guint64;
#define G_GINT64_CONSTANT(val)	(val##LL)
#endif

/* These depend on the C library. Using this file means the we
 * use the (bundled) Microsoft msvcrt.dll.
 */
#define G_GINT64_FORMAT "I64i"
#define G_GUINT64_FORMAT "I64u"

typedef gint32 gssize;
typedef guint32 gsize;

#define GPOINTER_TO_INT(p)	((gint)(p))
#define GPOINTER_TO_UINT(p)	((guint)(p))

#define GINT_TO_POINTER(i)	((gpointer)(i))
#define GUINT_TO_POINTER(u)	((gpointer)(u))

#define g_ATEXIT(proc)	(atexit (proc))

#define g_memmove(d,s,n) G_STMT_START { memmove ((d), (s), (n)); } G_STMT_END

#define GLIB_MAJOR_VERSION 1
#define GLIB_MINOR_VERSION 3
#define GLIB_MICRO_VERSION 2

#define G_OS_WIN32

#ifdef	__cplusplus
#define	G_HAVE_INLINE	1
#else	/* !__cplusplus */
#define G_HAVE___INLINE 1
#endif

#define G_THREADS_ENABLED
/*
 * The following program can be used to determine the magic values below:
 * #include <stdio.h>
 * #include <pthread.h>
 * main(int argc, char **argv)
 * {
 *   int i;
 *   pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
 *   printf ("sizeof (pthread_mutex_t) = %d\n", sizeof (pthread_mutex_t));
 *   printf ("sizeof (pthread_t) = %d\n", sizeof (pthread_t));
 *   printf ("PTHREAD_MUTEX_INITIALIZER = ");
 *   for (i = 0; i < sizeof (pthread_mutex_t); i++)
 *     printf ("%u, ", ((unsigned char *) &m)[i]);
 *   printf ("\n");
 *   exit(0);
 * }
 */

#define G_THREADS_IMPL_POSIX
typedef struct _GStaticMutex GStaticMutex;
struct _GStaticMutex
{
  struct _GMutex *runtime_mutex;
  union {
    /* The size of the pad array should be sizeof (pthread_mutex_t) */
    /* This value corresponds to the 1999-05-30 version of pthreads-win32 */
    char   pad[4];
    double dummy_double;
    void  *dummy_pointer;
    long   dummy_long;
  } aligned_pad_u;
};
/* This should be NULL followed by the bytes in PTHREAD_MUTEX_INITIALIZER */
#define	G_STATIC_MUTEX_INIT	{ NULL, { { 255, 255, 255, 255 } } }
#define	g_static_mutex_get_mutex(mutex) \
  (g_thread_use_default_impl ? ((GMutex*) &((mutex)->aligned_pad_u)) : \
   g_static_mutex_get_mutex_impl (&((mutex)->runtime_mutex)))
/* This represents a system thread as used by the implementation. An
 * alien implementaion, as loaded by g_thread_init can only count on
 * "sizeof (gpointer)" bytes to store their info. We however need more
 * for some of our native implementations. */
typedef union _GSystemThread GSystemThread;
union _GSystemThread
{
  /* The size of the data array should be sizeof (pthread_t) */
  /* This value corresponds to the 1999-05-30 version of pthreads-win32 */
  char   data[4];
  double dummy_double;
  void  *dummy_pointer;
  long   dummy_long;
};

#define GINT16_TO_LE(val)	((gint16) (val))
#define GUINT16_TO_LE(val)	((guint16) (val))
#define GINT16_TO_BE(val)	((gint16) GUINT16_SWAP_LE_BE (val))
#define GUINT16_TO_BE(val)	(GUINT16_SWAP_LE_BE (val))

#define GINT32_TO_LE(val)	((gint32) (val))
#define GUINT32_TO_LE(val)	((guint32) (val))
#define GINT32_TO_BE(val)	((gint32) GUINT32_SWAP_LE_BE (val))
#define GUINT32_TO_BE(val)	(GUINT32_SWAP_LE_BE (val))

#define GINT64_TO_LE(val)	((gint64) (val))
#define GUINT64_TO_LE(val)	((guint64) (val))
#define GINT64_TO_BE(val)	((gint64) GUINT64_SWAP_LE_BE (val))
#define GUINT64_TO_BE(val)	(GUINT64_SWAP_LE_BE (val))

#define GLONG_TO_LE(val)	((glong) GINT32_TO_LE (val))
#define GULONG_TO_LE(val)	((gulong) GUINT32_TO_LE (val))
#define GLONG_TO_BE(val)	((glong) GINT32_TO_BE (val))
#define GULONG_TO_BE(val)	((gulong) GUINT32_TO_BE (val))

#define GINT_TO_LE(val)		((gint) GINT32_TO_LE (val))
#define GUINT_TO_LE(val)	((guint) GUINT32_TO_LE (val))
#define GINT_TO_BE(val)		((gint) GINT32_TO_BE (val))
#define GUINT_TO_BE(val)	((guint) GUINT32_TO_BE (val))
#define G_BYTE_ORDER G_LITTLE_ENDIAN

#define GLIB_SYSDEF_POLLIN	= 1
#define GLIB_SYSDEF_POLLOUT	= 4
#define GLIB_SYSDEF_POLLPRI	= 2
#define GLIB_SYSDEF_POLLERR	= 8
#define GLIB_SYSDEF_POLLHUP	= 16
#define GLIB_SYSDEF_POLLNVAL	= 32

#define G_MODULE_SUFFIX "dll"

G_END_DECLS

#endif /* __G_LIBCONFIG_H__ */
