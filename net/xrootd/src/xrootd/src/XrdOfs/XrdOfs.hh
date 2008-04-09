#ifndef __OFS_API_H__
#define __OFS_API_H__
/******************************************************************************/
/*                                                                            */
/*                             X r d O f s . h h                              */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <string.h>
#include <dirent.h>
#include <sys/time.h>
#include <sys/types.h>
  
#include "XrdOfs/XrdOfsEvr.hh"
#include "XrdOfs/XrdOfsHandle.hh"
#include "XrdOdc/XrdOdcFinder.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucPList.hh"
#include "XrdSfs/XrdSfsInterface.hh"

class XrdOfsEvs;
class XrdOssDir;
class XrdOucEnv;
class XrdSysError;
class XrdSysLogger;
class XrdOucStream;
class XrdSfsAio;

/******************************************************************************/
/*                       X r d O f s D i r e c t o r y                        */
/******************************************************************************/
  
class XrdOfsDirectory : public XrdSfsDirectory
{
public:

        int         open(const char              *dirName,
                         const XrdSecEntity      *client,
                         const char              *opaque = 0);

        const char *nextEntry();

        int         close();

inline  void        copyError(XrdOucErrInfo &einfo) {einfo = error;}

const   char       *FName() {return (const char *)fname;}

                    XrdOfsDirectory(const char *user) : XrdSfsDirectory(user)
                          {dp     = 0;
                           tident = (user ? user : "");
                           fname=0; atEOF=0;
                          }
virtual            ~XrdOfsDirectory() {if (dp) close();}

protected:
const char    *tident;
char          *fname;

private:
XrdOssDF      *dp;
int            atEOF;
char           dname[MAXNAMLEN];
};

/******************************************************************************/
/*                            X r d O f s F i l e                             */
/******************************************************************************/
  
class XrdOfsFile : public XrdSfsFile
{
public:

        int          open(const char                *fileName,
                                XrdSfsFileOpenMode   openMode,
                                mode_t               createMode,
                          const XrdSecEntity        *client,
                          const char                *opaque = 0);

        int          close();

        const char  *FName() {return (oh ? oh->Name() : "?");}

        int          getMmap(void **Addr, off_t &Size);

        int            read(XrdSfsFileOffset   fileOffset,   // Preread only
                            XrdSfsXferSize     amount);

        XrdSfsXferSize read(XrdSfsFileOffset   fileOffset,
                            char              *buffer,
                            XrdSfsXferSize     buffer_size);

        int            read(XrdSfsAio *aioparm);

        XrdSfsXferSize write(XrdSfsFileOffset   fileOffset,
                             const char        *buffer,
                             XrdSfsXferSize     buffer_size);

        int            write(XrdSfsAio *aioparm);

        int            sync();

        int            sync(XrdSfsAio *aiop);

        int            stat(struct stat *buf);

        int            truncate(XrdSfsFileOffset   fileOffset);

        int            getCXinfo(char cxtype[4], int &cxrsz);

                       XrdOfsFile(const char *user) : XrdSfsFile(user)
                                 {oh = (XrdOfsHandle *)0; dorawio = 0;
                                  tident = (user ? user : "");
                                  gettimeofday(&tod, 0);
                                 }

virtual               ~XrdOfsFile() {if (oh) close();}

protected:
const char    *tident;

private:

       void         setCXinfo(XrdSfsFileOpenMode mode);
       void         TimeStamp() {gettimeofday(&tod, 0);}
        int         Unclose();

XrdOfsHandle  *oh;
int            dorawio;
struct timeval tod;
};

/******************************************************************************/
/*                          C l a s s   X r d O f s                           */
/******************************************************************************/

class XrdAccAuthorize;
  
class XrdOfs : public XrdSfsFileSystem
{
friend class XrdOfsDirectory;
friend class XrdOfsFile;

public:

// Object allocation
//
        XrdSfsDirectory *newDir(char *user=0)
                        {return (XrdSfsDirectory *)new XrdOfsDirectory(user);}

        XrdSfsFile      *newFile(char *user=0)
                        {return      (XrdSfsFile *)new XrdOfsFile(user);}

// Other functions
//
        int            chmod(const char             *Name,
                                   XrdSfsMode        Mode,
                                   XrdOucErrInfo    &out_error,
                             const XrdSecEntity     *client,
                             const char             *opaque = 0);

        int            exists(const char                *fileName,
                                    XrdSfsFileExistence &exists_flag,
                                    XrdOucErrInfo       &out_error,
                              const XrdSecEntity        *client,
                              const char                *opaque = 0);

        int            fsctl(const int               cmd,
                             const char             *args,
                                   XrdOucErrInfo    &out_error,
                             const XrdSecEntity     *client);

        int            getStats(char *buff, int blen) {return 0;}

const   char          *getVersion();

        int            mkdir(const char             *dirName,
                                   XrdSfsMode        Mode,
                                   XrdOucErrInfo    &out_error,
                             const XrdSecEntity     *client,
                             const char             *opaque = 0);

        int            prepare(      XrdSfsPrep       &pargs,
                                     XrdOucErrInfo    &out_error,
                               const XrdSecEntity     *client = 0);

        int            rem(const char             *path,
                                 XrdOucErrInfo    &out_error,
                           const XrdSecEntity     *client,
                           const char             *info = 0)
                          {return remove('f', path, out_error, client, info);}

        int            remdir(const char             *dirName,
                                    XrdOucErrInfo    &out_error,
                              const XrdSecEntity     *client,
                              const char             *info = 0)
                             {return remove('d',dirName,out_error,client,info);}

        int            rename(const char             *oldFileName,
                              const char             *newFileName,
                                    XrdOucErrInfo    &out_error,
                              const XrdSecEntity     *client,
                              const char             *infoO = 0,
                               const char            *infoN = 0);

        int            stat(const char             *Name,
                                  struct stat      *buf,
                                  XrdOucErrInfo    &out_error,
                            const XrdSecEntity     *client,
                            const char             *opaque = 0);

        int            stat(const char             *Name,
                                  mode_t           &mode,
                                  XrdOucErrInfo    &out_error,
                            const XrdSecEntity     *client,
                            const char             *opaque = 0);

// Management functions
//
virtual int            Configure(XrdSysError &);

        void           Config_Display(XrdSysError &);

virtual int            Unopen(XrdOfsHandle *);

                       XrdOfs();
virtual               ~XrdOfs() {}  // Too complicate to delete :-)

/******************************************************************************/
/*                  C o n f i g u r a t i o n   V a l u e s                   */
/******************************************************************************/
  
// Configuration values for this filesystem
//
int   Options;        //    Various options
int   myPort;         //    Port number being used

// Forward options
//
const char *fwdCHMOD;
const char *fwdMKDIR;
const char *fwdMKPATH;
const char *fwdMV;
const char *fwdRM;
const char *fwdRMDIR;

int   FDConn;         //    Number of conn file descriptors
int   FDOpen;         //    Number of open file descriptors
int   FDOpenMax;      //    Max open FD's before we do scan
int   FDMinIdle;      //    Min idle time in seconds
int   FDMaxIdle;      //    Max idle time before close

int   MaxDelay;       //    Max delay imposed during staging

int   LockTries;      //    Number of times to try for a lock
int   LockWait;       //    Number of milliseconds to wait for lock

char *HostName;       //    ->Our hostname
char *HostPref;       //    ->Our hostname with domain removed
char *ConfigFN;       //    ->Configuration filename
char *OssLib;         //    ->Oss Library

/******************************************************************************/
/*                       P r o t e c t e d   I t e m s                        */
/******************************************************************************/

protected:

// The following structure defines an anchor for the valid file list. There is
// one entry in the list for each validpath directive. When a request comes in,
// the named path is compared with entries in the VFlist. If no prefix match is
// found, the request is treated as being invalid (i.e., EACCES).
//
XrdOucPListAnchor VPlist;     // -> Valid file list

XrdOfsEvr         evrObject;  // Event receiver

virtual int   ConfigXeq(char *var, XrdOucStream &, XrdSysError &);
static  int   Emsg(const char *, XrdOucErrInfo  &, int, const char *x,
                   const char *y="");
XrdOdcFinder     *Finder;         //    ->Distrib Cluster Service
static  int   fsError(XrdOucErrInfo &myError, int rc);
        int   Stall(XrdOucErrInfo  &, int, const char *);
        char *WaitTime(int, char *, int);
  
/******************************************************************************/
/*                 P r i v a t e   C o n f i g u r a t i o n                  */
/******************************************************************************/

private:
  
char             *AuthLib;        //    ->Authorization   Library
char             *AuthParm;       //    ->Authorization   Parameters
char             *myRole;
XrdAccAuthorize  *Authorization;  //    ->Authorization   Service
XrdOdcFinderTRG  *Balancer;       //    ->Server Balancer Interface
XrdOfsEvs        *evsObject;      //    ->Event Notifier
char             *locResp;        //    ->Locate Response
int               locRlen;        //      Length of locResp;

/******************************************************************************/
/*                            O t h e r   D a t a                             */
/******************************************************************************/

// Common functions
//
        int   Close(XrdOfsHandle *, const char *trid=0);
        void  Detach_Name(const char *);
        int   remove(const char type, const char *path,
                     XrdOucErrInfo &out_error, const XrdSecEntity     *client,
                     const char *opaque);

// Function used during Configuration
//
int           ConfigRedir(XrdSysError &Eroute);
const char   *Fname(const char *);
int           setupAuth(XrdSysError &);
const char   *theRole(int opts);
void          List_VPlist(char *, XrdOucPListAnchor &, XrdSysError &);
int           xalib(XrdOucStream &, XrdSysError &);
int           xfdscan(XrdOucStream &, XrdSysError &);
int           xforward(XrdOucStream &, XrdSysError &);
int           xlocktry(XrdOucStream &, XrdSysError &);
int           xmaxd(XrdOucStream &, XrdSysError &);
int           xnot(XrdOucStream &, XrdSysError &);
int           xolib(XrdOucStream &, XrdSysError &);
int           xred(XrdOucStream &, XrdSysError &);
int           xrole(XrdOucStream &, XrdSysError &);
int           xtrace(XrdOucStream &, XrdSysError &);
};
#endif
