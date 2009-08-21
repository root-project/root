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
#include <sys/types.h>
  
#include "XrdOfs/XrdOfsEvr.hh"
#include "XrdOfs/XrdOfsHandle.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucPList.hh"
#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdCms/XrdCmsClient.hh"

class XrdOfsEvs;
class XrdOfsPocq;
class XrdOss;
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

virtual int          fctl(const int               cmd,
                          const char             *args,
                                XrdOucErrInfo    &out_error);

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

                       XrdOfsFile(const char *user);

virtual               ~XrdOfsFile() {viaDel = 1; if (oh) close();}

protected:
       const char   *tident;

private:

void           GenFWEvent();

XrdOfsHandle  *oh;
int            dorawio;
char           viaDel;
};

/******************************************************************************/
/*                          C l a s s   X r d O f s                           */
/******************************************************************************/

class XrdAccAuthorize;
class XrdCmsClient;
class XrdOfsPoscq;
  
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

        int            getStats(char *buff, int blen);

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

        int            truncate(const char             *Name,
                                      XrdSfsFileOffset fileOffset,
                                      XrdOucErrInfo    &out_error,
                                const XrdSecEntity     *client = 0,
                                const char             *opaque = 0);
// Management functions
//
virtual int            Configure(XrdSysError &);

        void           Config_Cluster(XrdOss *);

        void           Config_Display(XrdSysError &);

                       XrdOfs();
virtual               ~XrdOfs() {}  // Too complicate to delete :-)

/******************************************************************************/
/*                  C o n f i g u r a t i o n   V a l u e s                   */
/******************************************************************************/
  
// Configuration values for this filesystem
//
enum {Authorize = 0x0001,    // Authorization wanted
      isPeer    = 0x0050,    // Role peer
      isProxy   = 0x0020,    // Role proxy
      isManager = 0x0040,    // Role manager
      isServer  = 0x0080,    // Role server
      isSuper   = 0x00C0,    // Role supervisor
      isMeta    = 0x0100,    // Role meta + above
      haveRole  = 0x01F0,    // A role is present
      Forwarding= 0x1000     // Fowarding wanted
     };                      // These are set in Options below

int   Options;               // Various options
int   myPort;                // Port number being used

// Forward options
//
struct fwdOpt
      {const char *Cmd;
             char *Host;
             int   Port;
             void  Reset() {Cmd = 0; Port = 0;
                            if (Host) {free(Host); Host = 0;}
                           }
                   fwdOpt() : Cmd(0), Host(0), Port(0) {}
                  ~fwdOpt() {}
      };

struct fwdOpt fwdCHMOD;
struct fwdOpt fwdMKDIR;
struct fwdOpt fwdMKPATH;
struct fwdOpt fwdMV;
struct fwdOpt fwdRM;
struct fwdOpt fwdRMDIR;
struct fwdOpt fwdTRUNC;

static int MaxDelay;  //    Max delay imposed during staging
static int OSSDelay;  //    Delay to impose when oss interface times out

char *HostName;       //    ->Our hostname
char *HostPref;       //    ->Our hostname with domain removed
char *ConfigFN;       //    ->Configuration filename
char *OssLib;         //    ->Oss Library

/******************************************************************************/
/*                       P r o t e c t e d   I t e m s                        */
/******************************************************************************/

protected:

XrdOfsEvr     evrObject;      // Event receiver
XrdCmsClient *Finder;         // ->Cluster Management Service

virtual int   ConfigXeq(char *var, XrdOucStream &, XrdSysError &);
static  int   Emsg(const char *, XrdOucErrInfo  &, int, const char *x,
                   XrdOfsHandle *hP);
static  int   Emsg(const char *, XrdOucErrInfo  &, int, const char *x,
                   const char *y="");
static  int   fsError(XrdOucErrInfo &myError, int rc);
        int   Stall(XrdOucErrInfo  &, int, const char *);
        void  Unpersist(XrdOfsHandle *hP, int xcev=1);
        char *WaitTime(int, char *, int);

/******************************************************************************/
/*                 P r i v a t e   C o n f i g u r a t i o n                  */
/******************************************************************************/

private:
  
char             *AuthLib;        //    ->Authorization   Library
char             *AuthParm;       //    ->Authorization   Parameters
char             *myRole;
XrdAccAuthorize  *Authorization;  //    ->Authorization   Service
XrdCmsClient     *Balancer;       //    ->Cluster Local   Interface
XrdOfsEvs        *evsObject;      //    ->Event Notifier
char             *locResp;        //    ->Locate Response
int               locRlen;        //      Length of locResp;

XrdOfsPoscq      *poscQ;          //    -> poscQ if  persist on close enabled
char             *poscLog;        //    -> Directory for posc recovery log
int               poscHold;       //       Seconds to hold a forced close
int               poscAuto;       //  1 -> Automatic persist on close

static XrdOfsHandle     *dummyHandle;
XrdSysMutex              ocMutex; // Global mutex for open/close

/******************************************************************************/
/*                            O t h e r   D a t a                             */
/******************************************************************************/

// Common functions
//
        int   remove(const char type, const char *path,
                     XrdOucErrInfo &out_error, const XrdSecEntity     *client,
                     const char *opaque);

// Function used during Configuration
//
int           ConfigDispFwd(char *buff, struct fwdOpt &Fwd);
int           ConfigPosc(XrdSysError &Eroute);
int           ConfigRedir(XrdSysError &Eroute);
const char   *Fname(const char *);
int           Forward(int &Result, XrdOucErrInfo &Resp, struct fwdOpt &Fwd,
                      const char *arg1=0, const char *arg2=0,
                      const char *arg3=0, const char *arg4=0);
int           setupAuth(XrdSysError &);
const char   *theRole(int opts);
int           xalib(XrdOucStream &, XrdSysError &);
int           xforward(XrdOucStream &, XrdSysError &);
int           xmaxd(XrdOucStream &, XrdSysError &);
int           xnmsg(XrdOucStream &, XrdSysError &);
int           xnot(XrdOucStream &, XrdSysError &);
int           xolib(XrdOucStream &, XrdSysError &);
int           xpers(XrdOucStream &, XrdSysError &);
int           xred(XrdOucStream &, XrdSysError &);
int           xrole(XrdOucStream &, XrdSysError &);
int           xtrace(XrdOucStream &, XrdSysError &);
};
#endif
