#ifndef ROOT_RVERSION_HXX
#define ROOT_RVERSION_HXX

/* Update on release: */
#define ROOT_VERSION_MAJOR 6
#define ROOT_VERSION_MINOR 35
#define ROOT_VERSION_PATCH 1
#define ROOT_RELEASE_DATE "Nov 5 2024"

/* Don't change the lines below. */

/*
 * These macros can be used in the following way:
 *
 *    #if ROOT_VERSION_CODE >= ROOT_VERSION(6,32,4)
 *       #include <newheader.h>
 *    #else
 *       #include <oldheader.h>
 *    #endif
 *
*/

#define ROOT_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#define ROOT_VERSION_CODE ROOT_VERSION(ROOT_VERSION_MAJOR, ROOT_VERSION_MINOR, ROOT_VERSION_PATCH)

#define R__VERS_QUOTE1_MAJOR(P) #P
#define R__VERS_QUOTE_MAJOR(P) R__VERS_QUOTE1_MAJOR(P)


#if ROOT_VERSION_MINOR < 10
#define R__VERS_QUOTE1_MINOR(P) "0" #P
#else
#define R__VERS_QUOTE1_MINOR(P) #P
#endif
#define R__VERS_QUOTE_MINOR(P) R__VERS_QUOTE1_MINOR(P)

#if ROOT_VERSION_PATCH < 10
#define R__VERS_QUOTE1_PATCH(P) "0" #P
#else
#define R__VERS_QUOTE1_PATCH(P) #P
#endif
#define R__VERS_QUOTE_PATCH(P) R__VERS_QUOTE1_PATCH(P)

#define ROOT_RELEASE R__VERS_QUOTE_MAJOR(ROOT_VERSION_MAJOR) \
   "." R__VERS_QUOTE_MINOR(ROOT_VERSION_MINOR) \
   "." R__VERS_QUOTE_PATCH(ROOT_VERSION_PATCH)

#endif // ROOT_RVERSION_H
