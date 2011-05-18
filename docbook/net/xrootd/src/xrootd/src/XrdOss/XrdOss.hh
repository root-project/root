#ifndef _XRDOSS_H
#define _XRDOSS_H
/******************************************************************************/
/*                                                                            */
/*                     X r d O s s   &   X r d O s s D F                      */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <dirent.h>
#include <errno.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>

class XrdOucEnv;
class XrdSysLogger;
class XrdSfsAio;

#ifndef XrdOssOK
#define XrdOssOK 0
#endif

/******************************************************************************/
/*                              X r d O s s D F                               */
/******************************************************************************/

// This class defines the object that handles directory as well as file
// oriented requests. It is instantiated for each file/dir to be opened.
// The object is obtained by calling newDir() or newFile() in class XrdOss.
// This allows flexibility on how to structure an oss plugin.
  
class XrdOssDF
{
public:
                // Directory oriented methods
virtual int     Opendir(const char *)                        {return -ENOTDIR;}
virtual int     Readdir(char *buff, int blen)                {return -ENOTDIR;}

                // File oriented methods
virtual int     Fchmod(mode_t mode)                          {return -EISDIR;}
virtual int     Fstat(struct stat *)                         {return -EISDIR;}
virtual int     Fsync()                                      {return -EISDIR;}
virtual int     Fsync(XrdSfsAio *aiop)                       {return -EISDIR;}
virtual int     Ftruncate(unsigned long long)                {return -EISDIR;}
virtual int     getFD()                                      {return -1;}
virtual off_t   getMmap(void **addr)                         {return 0;}
virtual int     isCompressed(char *cxidp=0)                  {return -EISDIR;}
virtual int     Open(const char *, int, mode_t, XrdOucEnv &) {return -EISDIR;}
virtual ssize_t Read(off_t, size_t)                          {return (ssize_t)-EISDIR;}
virtual ssize_t Read(void *, off_t, size_t)                  {return (ssize_t)-EISDIR;}
virtual int     Read(XrdSfsAio *aoip)                        {return (ssize_t)-EISDIR;}
virtual ssize_t ReadRaw(    void *, off_t, size_t)           {return (ssize_t)-EISDIR;}
virtual ssize_t Write(const void *, off_t, size_t)           {return (ssize_t)-EISDIR;}
virtual int     Write(XrdSfsAio *aiop)                       {return (ssize_t)-EISDIR;}

                // Methods common to both
virtual int     Close(long long *retsz=0)=0;
inline  int     Handle() {return fd;}

                XrdOssDF() {fd = -1;}
virtual        ~XrdOssDF() {}

protected:

int     fd;      // The associated file descriptor.
};

/******************************************************************************/
/*                                X r d O s s                                 */
/******************************************************************************/

// Options that can be passed to Create()
//
#define XRDOSS_mkpath 0x01
#define XRDOSS_new    0x02
#define XRDOSS_Online 0x04
#define XRDOSS_isPFN  0x08
#define XRDOSS_isMIG  0x10

// Options that can be passed to Stat()
//
#define XRDOSS_resonly 0x01
#define XRDOSS_updtatm 0x02

// Class passed to StatVS()
//
class XrdOssVSInfo
{
public:
long long Total;   // Total bytes
long long Free;    // Total bytes free
long long Large;   // Total bytes in largest partition
long long LFree;   // Max   bytes free in contiguous chunk
long long Usage;   // Used  bytes (if usage enabled)
long long Quota;   // Quota bytes (if quota enabled)
int       Extents; // Number of partitions/extents
int       Reserved;

          XrdOssVSInfo() : Total(0),Free(0),Large(0),LFree(0),Usage(-1),
                           Quota(-1),Extents(0),Reserved(0) {}
         ~XrdOssVSInfo() {}
};
  
class XrdOss
{
public:
virtual XrdOssDF *newDir(const char *tident)=0;
virtual XrdOssDF *newFile(const char *tident)=0;

virtual int     Chmod(const char *, mode_t mode)=0;
virtual int     Create(const char *, const char *, mode_t, XrdOucEnv &, 
                       int opts=0)=0;
virtual int     Init(XrdSysLogger *, const char *)=0;
virtual int     Mkdir(const char *, mode_t mode, int mkpath=0)=0;
virtual int     Reloc(const char *, const char *, const char *, const char *x=0)
                      {return -ENOTSUP;}
virtual int     Remdir(const char *, int Opts=0)=0;
virtual int     Rename(const char *, const char *)=0;
virtual int     Stat(const char *, struct stat *, int opts=0)=0;
virtual int     StatFS(const char *path, char *buff, int &blen) 
                      {return -ENOTSUP;}
virtual int     StatLS(XrdOucEnv &env, const char *cgrp, char *buff, int &blen)
                      {return -ENOTSUP;}
virtual int     StatXA(const char *path, char *buff, int &blen)
                      {return -ENOTSUP;}
virtual int     StatXP(const char *path, unsigned long long &attr)
                      {return -ENOTSUP;}
virtual int     Truncate(const char *, unsigned long long)=0;
virtual int     Unlink(const char *, int Opts=0)=0;

virtual int     Stats(char *bp, int bl) {return 0;}

virtual int     StatVS(XrdOssVSInfo *sP, const char *sname=0, int updt=0)
                      {return -ENOTSUP;}

virtual int     Lfn2Pfn(const char *Path, char *buff, int blen)
                       {if ((int)strlen(Path) >= blen) return -ENAMETOOLONG;
                        strcpy(buff, Path); return 0;
                       }

                XrdOss() {}
virtual        ~XrdOss() {}
};

/******************************************************************************/
/*           S t o r a g e   S y s t e m   I n s t a n t i a t o r            */
/******************************************************************************/

// This function is called to obtain an instance of a configured XrdOss object.
// It is passed the object that would have been used as the storage system.
// The object is not initialized (i.e., Init() has not yet been called).
// This allows one to easily wrap the native implementation or to completely 
// replace it, as needed. The name of the config file and any parameters
// specified after the path on the ofs.osslib directive are also passed (note
// that if no parameters exist, parms may be null).

extern "C"
{
XrdOss *XrdOssGetStorageSystem(XrdOss       *native_oss,
                               XrdSysLogger *Logger,
                               const char   *config_fn,
                               const char   *parms);
}
#endif
