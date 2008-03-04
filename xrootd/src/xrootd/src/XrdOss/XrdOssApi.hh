#ifndef _XRDOSS_API_H
#define _XRDOSS_API_H
/******************************************************************************/
/*                                                                            */
/*                          X r d O s s A p i . h h                           */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/types.h>
#include <errno.h>
#include <iostream.h>

#include "XrdOss/XrdOss.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucExport.hh"
#include "XrdOuc/XrdOucPList.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucStream.hh"

/******************************************************************************/
/*                              o o s s _ D i r                               */
/******************************************************************************/

class XrdOssDir : public XrdOssDF
{
public:
int     Close();
int     Opendir(const char *);
int     Readdir(char *buff, int blen);

        // Constructor and destructor
        XrdOssDir(const char *tid) 
                 {lclfd=0; mssfd=0; pflags=ateof=isopen=0; tident=tid;}
       ~XrdOssDir() {if (isopen > 0) Close(); isopen = 0;}
private:
         DIR       *lclfd;
         void      *mssfd;
const    char      *tident;
unsigned long long  pflags;
         int        ateof;
         int        isopen;
};
  
/******************************************************************************/
/*                             o o s s _ F i l e                              */
/******************************************************************************/

class oocx_CXFile;
class XrdSfsAio;
class XrdOssMioFile;
  
class XrdOssFile : public XrdOssDF
{
public:

// The following two are virtual functions to allow for upcasting derivations
// of this implementation
//
virtual int     Close();
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
        XrdOssFile(const char *tid)
                  {cxobj = 0; rawio = 0; cxpgsz = 0; cxid[0] = '\0';
                   mmFile = 0; tident = tid;
                  }

virtual ~XrdOssFile() {if (fd >= 0) Close();}

private:
int     Open_ufs(const char *, int, int, unsigned long long);

static int   AioFailure;
oocx_CXFile *cxobj;
XrdOssMioFile *mmFile;
const char  *tident;
int          rawio;
int          cxpgsz;
char         cxid[4];
};

/******************************************************************************/
/*                              o o s s _ S y s                               */
/******************************************************************************/
  
class XrdOucMsubs;
class XrdOucName2Name;
class XrdOucProg;

class XrdOssSys : public XrdOss
{
public:
virtual XrdOssDF *newDir(const char *tident)
                       {return (XrdOssDF *)new XrdOssDir(tident);}
virtual XrdOssDF *newFile(const char *tident)
                       {return (XrdOssDF *)new XrdOssFile(tident);}

int       Chmod(const char *, mode_t mode);
void     *CacheScan(void *carg);
int       Configure(const char *, XrdSysError &);
void      Config_Display(XrdSysError &);
virtual
int       Create(const char *, const char *, mode_t, XrdOucEnv &, int opts=0);
int       GenLocalPath(const char *, char *);
int       GenRemotePath(const char *, char *);
int       Init(XrdSysLogger *, const char *);
int       IsRemote(const char *path) 
                  {return (RPList.Find(path) & XRDEXP_REMOTE) != 0;}
int       Mkdir(const char *, mode_t mode, int mkpath=0);
int       Mkpath(const char *, mode_t mode);
unsigned long long PathOpts(const char *path) {return RPList.Find(path);}
int       Remdir(const char *) {return -ENOTSUP;}
int       Rename(const char *, const char *);
virtual 
int       Stage(const char *, const char *, XrdOucEnv &, int, mode_t);
void     *Stage_In(void *carg);
int       Stat(const char *, struct stat *, int resonly=0);
int       Unlink(const char *);

static int   AioInit();
static int   AioAllOk;

static char  tryMmap;           // Memory mapped files enabled
static char  chkMmap;           // Memory mapped files are selective
   
int       MSS_Closedir(void *);
int       MSS_Create(const char *path, mode_t, XrdOucEnv &);
void     *MSS_Opendir(const char *, int &rc);
int       MSS_Readdir(void *fd, char *buff, int blen);
int       MSS_Remdir(const char *, const char *) {return -ENOTSUP;}
int       MSS_Rename(const char *, const char *);
int       MSS_Stat(const char *, struct stat *);
int       MSS_Unlink(const char *);

static const int MaxArgs = 15;

char     *ConfigFN;       // -> Pointer to the config file name
int       Hard_FD_Limit;  //    Hard file descriptor limit
int       MaxTwiddle;     //    Maximum seconds of internal wait
char     *LocalRoot;      // -> Path prefix for local  filename
char     *RemoteRoot;     // -> Path prefix for remote filename
int       StageRealTime;  //    If 1, Invoke stage command on demand
int       StageAsync;     //    If 1, return EINPROGRESS to the caller
int       StageCreate;    //    If 1, use open path to create files
char     *StageCmd;       // -> Staging command to use
char     *StageMsg;       // -> Staging message to be passed
XrdOucMsubs *StageSnd;    // -> Parsed Message

char     *StageEvents;    // -> file:////<adminpath> if async staging
int       StageEvSize;    //    Length of above
int       StageActLen;    //    Length of below
char     *StageAction;    // -> "wq " if sync | "wfn " if async

char     *StageArg[MaxArgs];
int       StageAln[MaxArgs];
int       StageAnum;      //    Count of valid Arg/Aln array elements
char     *MSSgwCmd;       // -> MSS Gateway command to use
long long MaxDBsize;      //    Maximum database size
int       FDFence;        //    Smallest file FD number allowed
int       FDLimit;        //    Largest  file FD number allowed
unsigned long long DirFlags;//  Default directory settings
int       Trace;          //    Trace flags
int       ConvertFN;      //    If 1 filenames need to be converted
char     *CompSuffix;     // -> Compressed file suffix or null for autodetect
int       CompSuflen;     //    Length of suffix
int       OptFlags;       //    General option flags
char     *DeprLine;       //    Temporrary to catch deprecated options

char             *N2N_Lib;   // -> Name2Name Library Path
char             *N2N_Parms; // -> Name2Name Object Parameters
XrdOucName2Name  *lcl_N2N;   // -> File mapper for local  files
XrdOucName2Name  *rmt_N2N;   // -> File mapper for remote files
XrdOucName2Name  *the_N2N;   // -> File mapper object
XrdOucPListAnchor RPList;    //    The real path list
   
         XrdOssSys();
virtual ~XrdOssSys() {}

protected:
// Cache management related data and methods
//
long long minalloc;          //    Minimum allocation
int       ovhalloc;          //    Allocation overage
int       fuzalloc;          //    Allocation fuzz
int       cscanint;          //    Seconds between cache scans
int       xfrspeed;          //    Average transfer speed (bytes/second)
int       xfrovhd;           //    Minimum seconds to get a file
int       xfrhold;           //    Second hold limit on failing requests
int       xfrkeep;           //    Second keep queued requests
int       xfrthreads;        //    Number of threads for staging
int       xfrtcount;         //    Actual count of threads (used for dtr)
long long pndbytes;          //    Total bytes to be staged (pending)
long long stgbytes;          //    Total bytes being staged (active)
long long totbytes;          //    Total bytes were  staged (active+pending)
int       totreqs;           //    Total   successful requests
int       badreqs;           //    Total unsuccessful requests

friend class XrdOssCache_FSData;
friend class XrdOssCache_FS;
friend class XrdOssCache_Group;
friend class XrdOssCache_Lock;

XrdOssCache_FSData *fsdata;   // -> Filesystem data
XrdOssCache_FS     *fsfirst;  // -> First  filesystem
XrdOssCache_FS     *fslast;   // -> Last   filesystem
XrdOssCache_FS     *fscurr;   // -> Curent filesystem (global allocation only)
XrdOssCache_Group  *fsgroups; // -> Cache group list

XrdOssCache_FSData *xsdata;   // -> Filesystem data   (config time only)
XrdOssCache_FS     *xsfirst;  // -> First  filesystem (config time only)
XrdOssCache_FS     *xslast;   // -> Last   filesystem (config time only)
XrdOssCache_FS     *xscurr;   // -> Curent filesystem (config time only)
XrdOssCache_Group  *xsgroups; // -> Cache group list  (config time only)

XrdOssCache_Req StageQ;       //    Queue of staging requests
XrdOucProg     *StageProg;    //    Command or manager than handles staging
XrdOucProg     *MSSgwProg;    //    Command for MSS meta-data operations

XrdSysMutex     CacheContext;
XrdSysSemaphore ReadyRequest;

char **sfx;                  // -> Valid filename suffixes

int                Alloc_Cache(const char *, int, mode_t, XrdOucEnv &);
int                Alloc_Local(const char *, int, mode_t, XrdOucEnv &);
int                BreakLink(const char *local_path, struct stat &statbuff);
int                CalcTime();
int                CalcTime(XrdOssCache_Req *req);
void               doScrub();
int                Find(XrdOssCache_Req *req, void *carg);
int                GetFile(XrdOssCache_Req *req);
time_t             HasFile(const char *fn, const char *sfx);
void               List_Cache(char *lname, int self, XrdSysError &Eroute);
void               ReCache();
int                Stage_QT(const char *, const char *, XrdOucEnv &, int, mode_t);
int                Stage_RT(const char *, const char *, XrdOucEnv &);

// Configuration related methods
//
off_t  Adjust(dev_t devid, off_t size);
int    chkDep(const char *var);
void   ConfigMio(XrdSysError &Eroute);
int    ConfigN2N(XrdSysError &Eroute);
int    ConfigProc(XrdSysError &Eroute);
int    ConfigStage(XrdSysError &Eroute);
int    ConfigXeq(char *, XrdOucStream &, XrdSysError &);
void   List_Path(const char *, char *, unsigned long long, XrdSysError &);
int    xalloc(XrdOucStream &Config, XrdSysError &Eroute);
int    xcache(XrdOucStream &Config, XrdSysError &Eroute);
int    xcacheBuild(char *grp, char *fn, XrdSysError &Eroute);
int    xcompdct(XrdOucStream &Config, XrdSysError &Eroute);
int    xcachescan(XrdOucStream &Config, XrdSysError &Eroute);
int    xdefault(XrdOucStream &Config, XrdSysError &Eroute);
int    xfdlimit(XrdOucStream &Config, XrdSysError &Eroute);
int    xmaxdbsz(XrdOucStream &Config, XrdSysError &Eroute);
int    xmemf(XrdOucStream &Config, XrdSysError &Eroute);
int    xnml(XrdOucStream &Config, XrdSysError &Eroute);
int    xpath(XrdOucStream &Config, XrdSysError &Eroute);
int    xstg(XrdOucStream &Config, XrdSysError &Eroute);
int    xtrace(XrdOucStream &Config, XrdSysError &Eroute);
int    xxfr(XrdOucStream &Config, XrdSysError &Eroute);

// Mass storage related methods
//
int    tranmode(char *);
int    MSS_Xeq(XrdOucStream **xfd, int okerr,
               const char *cmd, const char *arg1=0, const char *arg2=0);

// Other methods
//
int    RenameLink(char *old_path, char *new_path);
};

/******************************************************************************/
/*                  A P I   S p e c i f i c   D e f i n e s                   */
/******************************************************************************/
  
// The following defines are used to translate symbolicly linked paths
//
#define XrdOssTPC '%'
#define XrdOssTAMP(dst, src) \
   while(*src) {*dst = (*src == '/' ? XrdOssTPC : *src); src++; dst++;}; *dst='\0'
#define XrdOssTRNP(src) \
   while(*src) {if (*src == '/') *src = XrdOssTPC; src++;}

// The Check_RO macro is valid only for XrdOssSys objects.
//
#define Check_RO(act, flags, path, opname) \
   XRDEXP_REMOTE & (flags = PathOpts(path)); \
   if (flags & XRDEXP_NOTRW) \
      return OssEroute.Emsg(#act, -XRDOSS_E8005, opname, path)
#endif
