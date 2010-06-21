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
#include "XrdSys/XrdSysHeaders.hh"

#include "XrdOss/XrdOss.hh"
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
int     Close(long long *retsz=0);
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
class XrdOssCache_FS;
class XrdOssMioFile;
  
class XrdOssFile : public XrdOssDF
{
public:

// The following two are virtual functions to allow for upcasting derivations
// of this implementation
//
virtual int     Close(long long *retsz=0);
virtual int     Open(const char *, int, mode_t, XrdOucEnv &);

int     Fchmod(mode_t mode);
int     Fstat(struct stat *);
int     Fsync();
int     Fsync(XrdSfsAio *aiop);
int     Ftruncate(unsigned long long);
int     getFD() {return fd;}
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

static int      AioFailure;
oocx_CXFile    *cxobj;
XrdOssCache_FS *cacheP;
XrdOssMioFile  *mmFile;
const char     *tident;
long long       FSize;
int             rawio;
int             cxpgsz;
char            cxid[4];
};

/******************************************************************************/
/*                              o o s s _ S y s                               */
/******************************************************************************/
  
class XrdFrmProxy;
class XrdOssCache_Group;
class XrdOssCache_Space;
class XrdOucMsubs;
class XrdOucName2Name;
class XrdOucProg;
class XrdOssSpace;
class XrdOssStage_Req;

class XrdOssSys : public XrdOss
{
public:
virtual XrdOssDF *newDir(const char *tident)
                       {return (XrdOssDF *)new XrdOssDir(tident);}
virtual XrdOssDF *newFile(const char *tident)
                       {return (XrdOssDF *)new XrdOssFile(tident);}

int       Chmod(const char *, mode_t mode);
int       Configure(const char *, XrdSysError &);
void      Config_Display(XrdSysError &);
virtual
int       Create(const char *, const char *, mode_t, XrdOucEnv &, int opts=0);
int       GenLocalPath(const char *, char *);
int       GenRemotePath(const char *, char *);
int       Init(XrdSysLogger *, const char *);
int       IsRemote(const char *path) 
                  {return (RPList.Find(path) & XRDEXP_REMOTE) != 0;}
int       Lfn2Pfn(const char *Path, char *buff, int blen);
int       Mkdir(const char *, mode_t mode, int mkpath=0);
int       Mkpath(const char *, mode_t mode);
unsigned long long PathOpts(const char *path) {return RPList.Find(path);}
int       Reloc(const char *tident, const char *path,
                const char *cgName, const char *anchor=0);
int       Remdir(const char *, int Opts=0);  // In Unlink()
int       Rename(const char *, const char *);
virtual 
int       Stage(const char *, const char *, XrdOucEnv &, int, mode_t, unsigned long long );
void     *Stage_In(void *carg);
int       Stat(const char *, struct stat *, int opts=0);
int       StatFS(const char *path, char *buff, int &blen);
int       StatFS(const char *path, unsigned long long &Opt,
                 long long &fSize, long long &fSpace);
int       StatLS(XrdOucEnv &env, const char *path, char *buff, int &blen);
int       StatVS(XrdOssVSInfo *sP, const char *sname=0, int updt=0);
int       StatXA(const char *path, char *buff, int &blen);
int       StatXP(const char *path, unsigned long long &attr);
int       Truncate(const char *, unsigned long long Size);
int       Unlink(const char *, int Opts=0);

int       Stats(char *bp, int bl);

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
int       MSS_Stat(const char *, struct stat *buff=0);
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
int       StageFormat;    //    Format for default stagecmd
char     *StageCmd;       // -> Staging command to use
char     *StageMsg;       // -> Staging message to be passed
XrdOucMsubs *StageSnd;    // -> Parsed Message
XrdFrmProxy *StageFrm;    // -> Built-in stagecmd or zero

char     *StageEvents;    // -> file:////<adminpath> if async staging
int       StageEvSize;    //    Length of above
int       StageActLen;    //    Length of below
char     *StageAction;    // -> "wq " if sync | "wfn " if async

char     *StageArg[MaxArgs];
int       StageAln[MaxArgs];
int       StageAnum;      //    Count of valid Arg/Aln array elements
char     *RSSCmd;         // -> Remote Storage Service Command
int       isMSSC;         //    RSSCmd is old-style msscmd
int       RSSTout;        //    RSSCmd response timeout
long long MaxSize;        //    Maximum file size (*obsolete*)
int       FDFence;        //    Smallest file FD number allowed
int       FDLimit;        //    Largest  file FD number allowed
unsigned long long DirFlags;//  Default directory settings
int       Trace;          //    Trace flags
int       Solitary;       //    True if running in stand-alone mode
char     *CompSuffix;     // -> Compressed file suffix or null for autodetect
int       CompSuflen;     //    Length of suffix
int       OptFlags;       //    General option flags

char             *N2N_Lib;   // -> Name2Name Library Path
char             *N2N_Parms; // -> Name2Name Object Parameters
XrdOucName2Name  *lcl_N2N;   // -> File mapper for local  files
XrdOucName2Name  *rmt_N2N;   // -> File mapper for remote files
XrdOucName2Name  *the_N2N;   // -> File mapper object
XrdOucPListAnchor RPList;    //    The real path list
OssDPath         *DPList;    //    The stat path list
int               lenDP;
short             numDP;
short             numCG;
   
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

XrdOucProg     *StageProg;    //    Command or manager than handles staging
XrdOucProg     *RSSProg;      //    Command for Remote Storage Services

char           *UDir;         // -> Usage logdir
char           *QFile;        // -> Quota file

int                Alloc_Cache(const char *, mode_t, XrdOucEnv &);
int                Alloc_Local(const char *, mode_t, XrdOucEnv &);
int                BreakLink(const char *local_path, struct stat &statbuff);
int                CalcTime();
int                CalcTime(XrdOssStage_Req *req);
void               doScrub();
int                Find(XrdOssStage_Req *req, void *carg);
int                getCname(const char *path, struct stat *sbuff, char *cgbuff);
int                getStats(char *buff, int blen);
int                GetFile(XrdOssStage_Req *req);
int                getID(const char *, XrdOucEnv &, char *, int);
time_t             HasFile(const char *fn, const char *sfx, time_t *mTime=0);
int                Stage_QT(const char *, const char *, XrdOucEnv &, int, mode_t);
int                Stage_RT(const char *, const char *, XrdOucEnv &, unsigned long long);

// Configuration related methods
//
void   ConfigMio(XrdSysError &Eroute);
int    ConfigN2N(XrdSysError &Eroute);
int    ConfigProc(XrdSysError &Eroute);
void   ConfigSpace();
void   ConfigSpace(const char *Lfn);
void   ConfigSpath(XrdSysError &Eroute, const char *Pn,
                   unsigned long long &Fv, int noMSS);
int    ConfigStage(XrdSysError &Eroute);
int    ConfigStageC(XrdSysError &Eroute);
void   ConfigStats(XrdSysError &Eroute);
void   ConfigStats(dev_t Devnum, char *lP);
int    ConfigXeq(char *, XrdOucStream &, XrdSysError &);
void   List_Path(const char *, const char *, unsigned long long, XrdSysError &);
int    xalloc(XrdOucStream &Config, XrdSysError &Eroute);
int    xcache(XrdOucStream &Config, XrdSysError &Eroute);
int    xcompdct(XrdOucStream &Config, XrdSysError &Eroute);
int    xcachescan(XrdOucStream &Config, XrdSysError &Eroute);
int    xdefault(XrdOucStream &Config, XrdSysError &Eroute);
int    xfdlimit(XrdOucStream &Config, XrdSysError &Eroute);
int    xmaxsz(XrdOucStream &Config, XrdSysError &Eroute);
int    xmemf(XrdOucStream &Config, XrdSysError &Eroute);
int    xnml(XrdOucStream &Config, XrdSysError &Eroute);
int    xpath(XrdOucStream &Config, XrdSysError &Eroute);
int    xspace(XrdOucStream &Config, XrdSysError &Eroute, int *isCD=0);
int    xspaceBuild(char *grp, char *fn, int isxa, XrdSysError &Eroute);
int    xstg(XrdOucStream &Config, XrdSysError &Eroute);
int    xusage(XrdOucStream &Config, XrdSysError &Eroute);
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
int    RenameLink2(int Llen, char *oLnk, char *old_path,
                             char *nLnk, char *new_path);
};

/******************************************************************************/
/*                  A P I   S p e c i f i c   D e f i n e s                   */
/******************************************************************************/

// The Check_RO macro is valid only for XrdOssSys objects.
//
#define Check_RO(act, flags, path, opname) \
   XRDEXP_REMOTE & (flags = PathOpts(path)); \
   if (flags & XRDEXP_NOTRW) \
      return OssEroute.Emsg(#act, -XRDOSS_E8005, opname, path)
#endif
