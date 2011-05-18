/******************************************************************************/
/*                                                                            */
/*                    X r d P o s i x L i n k a g e . c c                     */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdPosixLinkageCVSID = "$Id$";

// Bypass Solaris ELF madness
//
#ifdef __solaris__
#include <sys/isa_defs.h>
#if defined(_ILP32) && (_FILE_OFFSET_BITS != 32)
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#undef  _LARGEFILE_SOURCE
#endif
#endif

#include <dlfcn.h>
#if !defined(__macos__) && !defined(__CYGWIN__)
#include <link.h>
#endif

#include <errno.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdPosix/XrdPosixLinkage.hh"
 
/******************************************************************************/
/*                   G l o b a l   D e c l a r a t i o n s                    */
/******************************************************************************/
  
XrdPosixLinkage Xunix;
 
/******************************************************************************/
/*                          M a c r o   L o a d e r                           */
/******************************************************************************/
  
#define LOOKUP_UNIX(symb) symb = (Retv_ ## symb (*)(Args_ ## symb)) \
                                 dlsym(RTLD_NEXT, Symb_ ## symb); \
                          if (!symb) {symb = Xrd_U_ ## symb; \
                                      Missing(Symb_ ## symb);}
 
/******************************************************************************/
/*          U n r e s o l v e d   R e f e r e n c e   L i n k a g e           */
/******************************************************************************/

      Retv_Access      Xrd_U_Access(Args_Access)
                         {return (Retv_Access)Xunix.Load_Error("access");}
      Retv_Acl         Xrd_U_Acl(Args_Acl)
                         {return (Retv_Acl)Xunix.Load_Error("acl");}
      Retv_Chdir       Xrd_U_Chdir(Args_Chdir) 
                         {return (Retv_Chdir)Xunix.Load_Error("chdir");}
      Retv_Close       Xrd_U_Close(Args_Close) 
                         {return (Retv_Close)Xunix.Load_Error("close");}
      Retv_Closedir    Xrd_U_Closedir(Args_Closedir) 
                         {return (Retv_Closedir)Xunix.Load_Error("closedir");}
      Retv_Fclose      Xrd_U_Fclose(Args_Fclose)
                         {return (Retv_Fclose)Xunix.Load_Error("fclose");}
      Retv_Fcntl       Xrd_U_Fcntl(Args_Fcntl) 
                         {Xunix.Load_Error("fcntl"); return (Retv_Fcntl)0;}
      Retv_Fcntl64     Xrd_U_Fcntl64(Args_Fcntl64)
                         {Xunix.Load_Error("fcntl"); return (Retv_Fcntl64)0;}
      Retv_Fdatasync   Xrd_U_Fdatasync(Args_Fdatasync)
                         {return (Retv_Fdatasync)Xunix.Load_Error("fdatasync");}
      Retv_Fflush      Xrd_U_Fflush(Args_Fflush)
                         {return (Retv_Fflush)Xunix.Load_Error("fflush");}
      Retv_Fopen       Xrd_U_Fopen(Args_Fopen) 
                         {Xunix.Load_Error("fopen"); return (Retv_Fopen)0;}
      Retv_Fopen64     Xrd_U_Fopen64(Args_Fopen64)
                         {Xunix.Load_Error("fopen"); return (Retv_Fopen64)0;}
      Retv_Fread       Xrd_U_Fread(Args_Fread)
                         {return (Retv_Fread)Xunix.Load_Error("fread");}
      Retv_Fseek       Xrd_U_Fseek(Args_Fseek)
                         {return (Retv_Fseek)Xunix.Load_Error("fseek");}
      Retv_Fseeko      Xrd_U_Fseeko(Args_Fseeko)
                         {return (Retv_Fseeko)Xunix.Load_Error("fseeko");}
      Retv_Fseeko64    Xrd_U_Fseeko64(Args_Fseeko64)
                         {return (Retv_Fseeko64)Xunix.Load_Error("fseeko64");}
      Retv_Fstat       Xrd_U_Fstat(Args_Fstat) 
                         {return (Retv_Fstat)Xunix.Load_Error("fstat");}
      Retv_Fstat64     Xrd_U_Fstat64(Args_Fstat64)
                         {return (Retv_Fstat64)Xunix.Load_Error("fstat64");}
      Retv_Fsync       Xrd_U_Fsync(Args_Fsync) 
                         {return (Retv_Fsync)Xunix.Load_Error("fsync");}
      Retv_Ftell       Xrd_U_Ftell(Args_Ftell)
                         {return (Retv_Ftell)Xunix.Load_Error("ftell");}
      Retv_Ftello      Xrd_U_Ftello(Args_Ftello)
                         {return (Retv_Ftello)Xunix.Load_Error("ftello");}
      Retv_Ftello64    Xrd_U_Ftello64(Args_Ftello64)
                         {return (Retv_Ftello64)Xunix.Load_Error("ftello64");}
      Retv_Ftruncate   Xrd_U_Ftruncate(Args_Ftruncate)
                         {return (Retv_Ftruncate)Xunix.Load_Error("ftruncate");}
      Retv_Ftruncate64 Xrd_U_Ftruncate64(Args_Ftruncate64)
                         {return (Retv_Ftruncate64)Xunix.Load_Error("ftruncate64");}
      Retv_Fwrite      Xrd_U_Fwrite(Args_Fwrite)
                         {return (Retv_Fwrite)Xunix.Load_Error("fwrite");}
      Retv_Fgetxattr   Xrd_U_Fgetxattr(Args_Fgetxattr)
                         {return (Retv_Fgetxattr)Xunix.Load_Error("fgetxattr");}
      Retv_Getxattr    Xrd_U_Getxattr(Args_Getxattr)
                         {return (Retv_Getxattr)Xunix.Load_Error("getxattr");}
      Retv_Lgetxattr   Xrd_U_Lgetxattr(Args_Lgetxattr)
                         {return (Retv_Lgetxattr)Xunix.Load_Error("lgetxattr");}
      Retv_Lseek       Xrd_U_Lseek(Args_Lseek) 
                         {return (Retv_Lseek)Xunix.Load_Error("lseek");}
      Retv_Lseek64     Xrd_U_Lseek64(Args_Lseek64)
                         {return (Retv_Lseek64)Xunix.Load_Error("lseek");}
      Retv_Lstat       Xrd_U_Lstat(Args_Lstat)
                         {return (Retv_Lstat)Xunix.Load_Error("lstat");}
      Retv_Lstat64     Xrd_U_Lstat64(Args_Lstat64)
                         {return (Retv_Lstat64)Xunix.Load_Error("lstat");}
      Retv_Mkdir       Xrd_U_Mkdir(Args_Mkdir) 
                         {return (Retv_Mkdir)Xunix.Load_Error("mkdir");}
      Retv_Open        Xrd_U_Open(Args_Open) 
                         {return (Retv_Open)Xunix.Load_Error("open");}
      Retv_Open64      Xrd_U_Open64(Args_Open64)
                         {return (Retv_Open64)Xunix.Load_Error("open");}
      Retv_Opendir     Xrd_U_Opendir(Args_Opendir) 
                         {Xunix.Load_Error("opendir"); return (Retv_Opendir)0;}
      Retv_Pathconf    Xrd_U_Pathconf(Args_Pathconf)
                         {return (Retv_Pathconf)Xunix.Load_Error("pathconf");}
      Retv_Pread       Xrd_U_Pread(Args_Pread)
                         {return (Retv_Pread)Xunix.Load_Error("pread");}
      Retv_Pread64     Xrd_U_Pread64(Args_Pread64)
                         {return (Retv_Pread64)Xunix.Load_Error("pread");}
      Retv_Pwrite      Xrd_U_Pwrite(Args_Pwrite) 
                         {return (Retv_Pwrite)Xunix.Load_Error("pwrite");}
      Retv_Pwrite64    Xrd_U_Pwrite64(Args_Pwrite64)
                         {return (Retv_Pwrite64)Xunix.Load_Error("pwrite");}
      Retv_Read        Xrd_U_Read(Args_Read) 
                         {return (Retv_Read)Xunix.Load_Error("read");}
      Retv_Readv       Xrd_U_Readv(Args_Readv) 
                         {return (Retv_Readv)Xunix.Load_Error("readv");}
      Retv_Readdir     Xrd_U_Readdir(Args_Readdir) 
                         {Xunix.Load_Error("readdir"); return (Retv_Readdir)0;}
      Retv_Readdir64   Xrd_U_Readdir64(Args_Readdir64)
                         {Xunix.Load_Error("readdir64");return (Retv_Readdir64)0;}
      Retv_Readdir_r   Xrd_U_Readdir_r(Args_Readdir_r) 
                         {return (Retv_Readdir_r)Xunix.Load_Error("readdir_r", ELIBACC);}
      Retv_Readdir64_r Xrd_U_Readdir64_r(Args_Readdir64_r)
                         {return (Retv_Readdir64_r)Xunix.Load_Error("readdir64_r", ELIBACC);}
      Retv_Rename      Xrd_U_Rename(Args_Rename)
                         {return (Retv_Rename)Xunix.Load_Error("rename");}
      Retv_Rewinddir   Xrd_U_Rewinddir(Args_Rewinddir) 
                         {       Xunix.Load_Error("rewinddir"); abort();}
      Retv_Rmdir       Xrd_U_Rmdir(Args_Rmdir) 
                         {return (Retv_Rmdir)Xunix.Load_Error("rmdir");}
      Retv_Seekdir     Xrd_U_Seekdir(Args_Seekdir) 
                         {       Xunix.Load_Error("seekdir"); abort();}
      Retv_Stat        Xrd_U_Stat(Args_Stat) 
                         {return (Retv_Stat)Xunix.Load_Error("stat");}
      Retv_Stat64      Xrd_U_Stat64(Args_Stat64)
                         {return (Retv_Stat64)Xunix.Load_Error("stat");}
      Retv_Statfs      Xrd_U_Statfs(Args_Statfs)
                         {return (Retv_Statfs)Xunix.Load_Error("statfs");}
      Retv_Statfs64    Xrd_U_Statfs64(Args_Statfs64)
                         {return (Retv_Statfs64)Xunix.Load_Error("statfs64");}
      Retv_Statvfs     Xrd_U_Statvfs(Args_Statvfs)
                         {return (Retv_Statvfs)Xunix.Load_Error("statvfs");}
      Retv_Statvfs64   Xrd_U_Statvfs64(Args_Statvfs64)
                         {return (Retv_Statvfs64)Xunix.Load_Error("statvfs64");}
      Retv_Telldir     Xrd_U_Telldir(Args_Telldir) 
                         {return (Retv_Telldir)Xunix.Load_Error("telldir");}
      Retv_Truncate    Xrd_U_Truncate(Args_Truncate)
                         {return (Retv_Truncate)Xunix.Load_Error("truncate");}
      Retv_Truncate64  Xrd_U_Truncate64(Args_Truncate64)
                         {return (Retv_Truncate64)Xunix.Load_Error("truncate64");}
      Retv_Unlink      Xrd_U_Unlink(Args_Unlink) 
                         {return (Retv_Unlink)Xunix.Load_Error("unlink");}
      Retv_Write       Xrd_U_Write(Args_Write) 
                         {return (Retv_Write)Xunix.Load_Error("write");}
      Retv_Writev      Xrd_U_Writev(Args_Writev) 
                         {return (Retv_Writev)Xunix.Load_Error("writev");}
  
/******************************************************************************/
/*           X r d P o s i x L i n k a g e   C o n s t r u c t o r            */
/******************************************************************************/
  
int XrdPosixLinkage::Resolve()
{
  LOOKUP_UNIX(Access)
  LOOKUP_UNIX(Acl)
  LOOKUP_UNIX(Chdir)
  LOOKUP_UNIX(Close)
  LOOKUP_UNIX(Closedir)
  LOOKUP_UNIX(Fclose)
  LOOKUP_UNIX(Fcntl)
  LOOKUP_UNIX(Fcntl64)
  LOOKUP_UNIX(Fdatasync)
  LOOKUP_UNIX(Fflush)
  LOOKUP_UNIX(Fopen)
  LOOKUP_UNIX(Fopen64)
  LOOKUP_UNIX(Fread)
  LOOKUP_UNIX(Fseek)
  LOOKUP_UNIX(Fseeko)
  LOOKUP_UNIX(Fseeko64)
  LOOKUP_UNIX(Fstat)
  LOOKUP_UNIX(Fstat64)
  LOOKUP_UNIX(Fsync)
  LOOKUP_UNIX(Ftell)
  LOOKUP_UNIX(Ftello)
  LOOKUP_UNIX(Ftello64)
  LOOKUP_UNIX(Ftruncate)
  LOOKUP_UNIX(Ftruncate64)
  LOOKUP_UNIX(Fwrite)
  LOOKUP_UNIX(Fgetxattr)
  LOOKUP_UNIX(Getxattr)
  LOOKUP_UNIX(Lgetxattr)
  LOOKUP_UNIX(Lseek)
  LOOKUP_UNIX(Lseek64)
  LOOKUP_UNIX(Lstat)
  LOOKUP_UNIX(Lstat64)
  LOOKUP_UNIX(Fsync)
  LOOKUP_UNIX(Mkdir)
  LOOKUP_UNIX(Open)
  LOOKUP_UNIX(Open64)
  LOOKUP_UNIX(Opendir)
  LOOKUP_UNIX(Pathconf)
  LOOKUP_UNIX(Pread)
  LOOKUP_UNIX(Pread64)
  LOOKUP_UNIX(Pwrite)
  LOOKUP_UNIX(Pwrite64)
  LOOKUP_UNIX(Read)
  LOOKUP_UNIX(Readv)
  LOOKUP_UNIX(Readdir)
  LOOKUP_UNIX(Readdir64)
  LOOKUP_UNIX(Readdir_r)
  LOOKUP_UNIX(Readdir64_r)
  LOOKUP_UNIX(Rename)
  LOOKUP_UNIX(Rewinddir)
  LOOKUP_UNIX(Rmdir)
  LOOKUP_UNIX(Seekdir)
  LOOKUP_UNIX(Stat)
  LOOKUP_UNIX(Stat64)
  LOOKUP_UNIX(Statfs)
  LOOKUP_UNIX(Statfs64)
  LOOKUP_UNIX(Statvfs)
  LOOKUP_UNIX(Statvfs64)
  LOOKUP_UNIX(Telldir)
  LOOKUP_UNIX(Truncate)
  LOOKUP_UNIX(Truncate64)
  LOOKUP_UNIX(Unlink)
  LOOKUP_UNIX(Write)
  LOOKUP_UNIX(Writev)
  if (getenv("XRDPOSIX_REPORT")) Missing(0);
  return 1;
}

/******************************************************************************/
/*           X r d P o s i x L i n k a g e : : L o a d _ E r r o r            */
/******************************************************************************/
  
int XrdPosixLinkage::Load_Error(const char *epname, int retv)
{
    if (*Write != &Xrd_U_Write && *Writev != &Xrd_U_Writev)
       cerr << "PosixPreload: Unable to resolve Unix '" <<epname <<"()'" <<endl;
    errno = ELIBACC;
    return retv;
}

/******************************************************************************/
/*                               M i s s i n g                                */
/******************************************************************************/
  
void XrdPosixLinkage::Missing(const char *epname)
{
   struct Missing
         {struct Missing *Next;
          const char     *What;

                          Missing(Missing *Prev, const char *That)
                                 : Next(Prev), What(That) {}
                         ~Missing() {}
         };

   static Missing *epList = 0;

   if (epname) epList = new Missing(epList, epname);
      else {Missing *np = epList;
            while(np) cerr << "PosixPreload: Unable to resolve Unix '" 
                           <<epname <<"()'" <<endl;
            np = np->Next;
           }
}
