#ifndef _XRDPSS_API_H
#define _XRDPSS_API_H
/******************************************************************************/
/*                                                                            */
/*                             X r d P s s . h h                              */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOuc/XrdOucName2Name.hh"
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
        XrdPssDir(const char *tid) : tident(tid), dirVec(0) {}
       ~XrdPssDir() {if (dirVec) Close();}
private:
const    char      *tident;
         char     **dirVec;
         int        curEnt;
         int        numEnt;
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

static char *P2URL(char *pbuff, int pblen,   const char *path, int Split=0,
             const char *Cgi=0, int CgiLn=0, const char *tIdent=0);
static int   T2UID(const char *Ident);

static const char  *ConfigFN;       // -> Pointer to the config file name
static const char  *myHost;
static const char  *myName;
static uid_t        myUid;
static gid_t        myGid;
static XrdOucTList *ManList;
static const char  *urlPlain;
static int          urlPlen;
static int          hdrLen;
static const char  *hdrData;
static int          Workers;

static char         allChmod;
static char         allMkdir;
static char         allMv;
static char         allRmdir;
static char         allRm;
static char         allTrunc;

         XrdPssSys() {}
virtual ~XrdPssSys() {}

private:

char            *N2NLib;   // -> Name2Name Library Path
char            *N2NParms; // -> Name2Name Object Parameters
XrdOucName2Name *theN2N;   // -> File mapper object

int    buildHdr();
int    Configure(const char *);
int    ConfigProc(const char *ConfigFN);
int    ConfigXeq(char*, XrdOucStream&);
int    ConfigN2N();
int    xconf(XrdSysError *Eroute, XrdOucStream &Config);
int    xorig(XrdSysError *errp,   XrdOucStream &Config);
int    xsopt(XrdSysError *Eroute, XrdOucStream &Config);
int    xtrac(XrdSysError *Eroute, XrdOucStream &Config);
int    xnml (XrdSysError *Eroute, XrdOucStream &Config);
};
#endif
