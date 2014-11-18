#ifndef __XRDPOSIX_H__
#define __XRDPOSIX_H__
/******************************************************************************/
/*                                                                            */
/*                           X r d P o s i x . h h                            */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/* Modified by Frank Winklmeier to add the full Posix file system definition. */
/******************************************************************************/
  
//           $Id$

// The following defines substitute our names for the common system names. We
// would have liked to use wrappers but each platform uses a different mechanism
// to accomplish this. So, redefinition is the most portable way of doing this.
//

#define access(a,b)      XrdPosix_Access(a,b)

#define chdir(a)         XrdPosix_Chdir(a)

#define close(a)         XrdPosix_Close(a)

#define closedir(a)      XrdPosix_Closedir(a)

#define lseek(a,b,c)     XrdPosix_Lseek(a,b,c)

#define fopen(a,b)       XrdPosix_Fopen(a,b)

#define fread(b,s,n,f)   XrdPosix_Fread(b,s,n,f)

#define fseek(a,b,c)     XrdPosix_Fseek(a,b,c)

#define fseeko(a,b,c)    XrdPosix_Fseeko(a,b,c)

#define fstat(a,b)       XrdPosix_Fstat(a,b)

#define fsync(a)         XrdPosix_Fsync(a)

#define ftell(a)         XrdPosix_Ftell(a)

#define ftello(a)        XrdPosix_Ftello(a)

#define ftruncate(a,b)   XrdPosix_Ftruncate(a,b)

#define fwrite(b,s,n,f)  XrdPosix_Fwrite(b,s,n,f)

#define mkdir(a,b)       XrdPosix_Mkdir(a,b)

#define open             XrdPosix_Open

#define opendir(a)       XrdPosix_Opendir(a)
  
#define pread(a,b,c,d)   XrdPosix_Pread(a,b,c,d)

#define read(a,b,c)      XrdPosix_Read(a,b,c)
  
#define readv(a,b,c)     XrdPosix_Readv(a,b,c)

#define readdir(a)       XrdPosix_Readdir(a)
#define readdir64(a)     XrdPosix_Readdir64(a)

#define readdir_r(a,b,c)   XrdPosix_Readdir_r(a,b,c)
#define readdir64_r(a,b,c) XrdPosix_Readdir64_r(a,b,c)

#define rename(a,b)      XrdPosix_Rename(a,b)

#define rewinddir(a)     XrdPosix_Rewinddir(a)

#define rmdir(a)         XrdPosix_Rmdir(a)

#define seekdir(a,b)     XrdPosix_Seekdir(a,b)

#define stat(a,b)        XrdPosix_Stat(a,b)

#define statfs(a,b)      XrdPosix_Statfs(a,b)

#define statvfs(a,b)     XrdPosix_Statvfs(a,b)

#define pwrite(a,b,c,d)  XrdPosix_Pwrite(a,b,c,d)

#define telldir(a)       XrdPosix_Telldir(a)

#define truncate(a,b)    XrdPosix_Truncate(a,b)

#define unlink(a)        XrdPosix_Unlink(a)

#define write(a,b,c)     XrdPosix_Write(a,b,c)

#define writev(a,b,c)    XrdPosix_Writev(a,b,c)

// Now define the external interfaces (not C++ but OS compatabile)
//
#include "XrdPosix/XrdPosixExtern.hh"

#endif
