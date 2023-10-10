#ifndef ROOT_RVERSION_HXX
#define ROOT_RVERSION_HXX

/* Update on release: */
#define ROOT_VERSION_MAJOR 6
#define ROOT_VERSION_MINOR 31
#define ROOT_VERSION_PATCH 01
#define ROOT_RELEASE_DATE "Oct 10 2023"

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

#define R__VERS_QUOTE1(P) #P
#define R__VERS_QUOTE(P) R__VERS_QUOTE1(P)

#define ROOT_RELEASE R__VERS_QUOTE(ROOT_VERSION_MAJOR) \
   "." R__VERS_QUOTE(ROOT_VERSION_MINOR) \
   "/" R__VERS_QUOTE(ROOT_VERSION_PATCH)

#endif // ROOT_RVERSION_H
