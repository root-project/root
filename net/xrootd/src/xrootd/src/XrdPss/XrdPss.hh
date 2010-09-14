#ifndef _XRDPSS_API_H
#define _XRDPSS_API_H
/******************************************************************************/
/*                                                                            */
/*                             X r d P s s . h h                              */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/types.h>
#include <errno.h>
#include "XrdSys/XrdSysHeaders.hh"

#include "XrdOss/XrdOss.hh"

/******************************************************************************/
/*                             X r d P s s D i r                              */
/******************************************************************************/

class XrdPssDir : public XrdOssDF
{
public:
int     Close(long long *retsz=0);
int     Opendir(const char *);
int     Readdir(char *buff, int blen);

        // Constructor and destructor
        XrdPssDir(const char *tid) 
                 {lclfd=0; ateof=0; tident=tid;}
       ~XrdPssDir() {if (lclfd) Close();}
private:
         DIR       *lclfd;
const    char      *tident;
         int        ateof;
};
  
/******************************************************************************/
/*                            X r d P s s F i l e                             */
/******************************************************************************/

class XrdSfsAio;
  
class XrdPssFile : public XrdOssDF
{
public:

// The following two are virtual functions to allow for upcasting derivations
// of this implementation
//
virtual int     Close(long long *retsz=0);
virtual int     Open(const char *, int, mode_t, XrdOucEnv &);

int     Fstat(struct stat *);
int     Fsync();
int     Fsync(XrdSfsAio *aiop);
int     Ftruncate(unsigned long long);
off_t   getMmap(void **addr);
int     isCompressed(char *cxidp=0);
ssize_t Read(               off_t, size_t);
ssize_t Read(       void *, off_t, size_t);
int     Read(XrdSfsAio *aiop);
ssize_t ReadRaw(    void *, off_t, size_t);
ssize_t Write(const void *, off_t, size_t);
int     Write(XrdSfsAio *aiop);
 
         // Constructor and destructor
         XrdPssFile(const char *tid) {fd = -1; tident = tid;}

virtual ~XrdPssFile() {if (fd >= 0) Close();}

private:

const char *tident;
const char *crPath;
      int   crOpts;
};

/******************************************************************************/
/*                             X r d P s s S y s                              */
/******************************************************************************/
  
class XrdOucEnv;
class XrdSysError;
class XrdOucStream;
class XrdOucTList;

class XrdPssSys : public XrdOss
{
public:
virtual XrdOssDF *newDir(const char *tident)
                       {return (XrdOssDF *)new XrdPssDir(tident);}
virtual XrdOssDF *newFile(const char *tident)
                       {return (XrdOssDF *)new XrdPssFile(tident);}

int       Chmod(const char *, mode_t mode);
virtual
int       Create(const char *, const char *, mode_t, XrdOucEnv &, int opts=0);
int       Init(XrdSysLogger *, const char *);
int       Mkdir(const char *, mode_t mode, int mkpath=0);
int       Remdir(const char *, int Opts=0);
int       Rename(const char *, const char *);
int       Stat(const char *, struct stat *, int resonly=0);
int       Truncate(const char *, unsigned long long);
int       Unlink(const char *, int Opts=0);

static int P2URL(char *pbuff,int pblen,const char *path,XrdOucEnv *env=0);

static const char  *ConfigFN;       // -> Pointer to the config file name
static const char  *myHost;
static const char  *myName;
static XrdOucTList *PanList;
static char        *hdrData;
static char         hdrLen;
static long         rdAheadSz;
static long         rdCacheSz;
static long         numStream;
   
         XrdPssSys() {}
virtual ~XrdPssSys() {}

private:

int    buildHdr();
int    Configure(const char *);
int    ConfigProc(const char *ConfigFN);
int    ConfigXeq(char*, XrdOucStream&);
int    xmang(XrdSysError *errp,   XrdOucStream &Config);
int    xsopt(XrdSysError *Eroute, XrdOucStream &Config);
int    xtrac(XrdSysError *Eroute, XrdOucStream &Config);
};
#endif
