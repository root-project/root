#ifndef __XRDPOSIXOSDEP_H__
#define __XRDPOSIXOSDEP_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d P o s i x O s D e p . h h                       */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/* Modified by Frank Winklmeier to add the full Posix file system definition. */
/******************************************************************************/
  
//           $Id$

// Solaris does not have a statfs64 structure. So all interfaces use statfs.
//
#ifdef __solaris__
#define statfs64 statfs
#endif

// We need to avoid using dirent64 for MacOS platforms. We would normally
// include XrdSysPlatform.hh for this but this include file needs to be
// standalone. So, we replicate the dirent64 redefinition here, Additionally,
// off64_t, normally defined in Solaris and Linux, is cast as long long (the
// appropriate type for the next 25 years). The Posix interface only supports
// 64-bit offsets.
//
#if  defined(__macos__)
#if !defined(dirent64)
#define dirent64 dirent
#endif
#if !defined(off64_t)
#define off64_t long long
#endif

#if defined(__DARWIN_VERS_1050) && !__DARWIN_VERS_1050
#if !defined(stat64)
#define stat64 stat
#endif
#if !defined(statfs64)
#define statfs64 statfs
#endif
#endif

#if !defined(statvfs64)
#define statvfs64 statvfs
#endif
#define ELIBACC ESHLIBVERS
#endif

#ifdef __FreeBSD__
#define	dirent64 dirent
#define	ELIBACC EFTYPE
#endif

#endif
