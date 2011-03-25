#ifndef __XROOTD_PROTOCOL_H__
#define __XROOTD_PROTOCOL_H__
/******************************************************************************/
/*                                                                            */
/*                  X r d X r o o t d P r o t o c o l . h h                   */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$
 
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSec/XrdSecInterface.hh"

#include "Xrd/XrdObject.hh"
#include "Xrd/XrdProtocol.hh"
#include "XrdXrootd/XrdXrootdReqID.hh"
#include "XrdXrootd/XrdXrootdResponse.hh"
#include "XProtocol/XProtocol.hh"

/******************************************************************************/
/*                               D e f i n e s                                */
/******************************************************************************/
  
#define XROOTD_VERSBIN 0x00000291

#define XROOTD_VERSION "2.9.1"

#define ROOTD_PQ 2012

#define XRD_LOGGEDIN       1
#define XRD_NEED_AUTH      2
#define XRD_ADMINUSER      4
#define XRD_BOUNDPATH      8

#ifndef __GNUC__
#define __attribute__(x)
#endif

/******************************************************************************/
/*                   x r d _ P r o t o c o l _ X R o o t d                    */
/******************************************************************************/

class XrdNetSocket;
class XrdOucErrInfo;
class XrdOucStream;
class XrdOucTokenizer;
class XrdOucTrace;
class XrdSfsFileSystem;
class XrdSecProtocol;
class XrdBuffer;
class XrdLink;
class XrdXrootdAioReq;
class XrdXrootdFile;
class XrdXrootdFileLock;
class XrdXrootdFileTable;
class XrdXrootdJob;
class XrdXrootdMonitor;
class XrdXrootdPio;
class XrdXrootdStats;
class XrdXrootdXPath;

class XrdXrootdProtocol : public XrdProtocol
{
friend class XrdXrootdAdmin;
friend class XrdXrootdAioReq;
public:

static int           Configure(char *parms, XrdProtocol_Config *pi);

       void          DoIt() {(*this.*Resume)();}

       XrdProtocol  *Match(XrdLink *lp);

       int           Process(XrdLink *lp); //  Sync: Job->Link.DoIt->Process

       void          Recycle(XrdLink *lp, int consec, const char *reason);

       int           Stats(char *buff, int blen, int do_sync=0);

static int           StatGen(struct stat &buf, char *xxBuff);

              XrdXrootdProtocol operator =(const XrdXrootdProtocol &rhs);
              XrdXrootdProtocol();
             ~XrdXrootdProtocol() {Cleanup();}

private:

// Note that Route[] structure (below) must have RD_Num elements!
//
enum RD_func {RD_chmod = 0, RD_dirlist, RD_locate,RD_mkdir, RD_mv,
              RD_prepare,   RD_prepstg, RD_rm,    RD_rmdir, RD_stat,
              RD_open1,     RD_open2,   RD_open3, RD_open4, RD_Num};

       int   do_Admin();
       int   do_Auth();
       int   do_Bind();
       int   do_Chmod();
       int   do_CKsum(int canit);
       int   do_Close();
       int   do_Dirlist();
       int   do_Endsess();
       int   do_Getfile();
       int   do_Login();
       int   do_Locate();
       int   do_Mkdir();
       int   do_Mv();
       int   do_Offload(int pathID, int isRead);
       int   do_OffloadIO();
       int   do_Open();
       int   do_Ping();
       int   do_Prepare();
       int   do_Protocol();
       int   do_Putfile();
       int   do_Qconf();
       int   do_Qfh();
       int   do_Qopaque(short);
       int   do_Qspace();
       int   do_Query();
       int   do_Qxattr();
       int   do_Read();
       int   do_ReadV();
       int   do_ReadAll(int asyncOK=1);
       int   do_ReadNone(int &retc, int &pathID);
       int   do_Rm();
       int   do_Rmdir();
       int   do_Set();
       int   do_Set_Mon(XrdOucTokenizer &setargs);
       int   do_Stat();
       int   do_Statx();
       int   do_Sync();
       int   do_Truncate();
       int   do_Write();
       int   do_WriteAll();
       int   do_WriteCont();
       int   do_WriteNone();

       int   aio_Error(const char *op, int ecode);
       int   aio_Read();
       int   aio_Write();
       int   aio_WriteAll();
       int   aio_WriteCont();

       void  Assign(const XrdXrootdProtocol &rhs);
       void  Cleanup();
static int   Config(const char *fn);
       int   fsError(int rc, XrdOucErrInfo &myError);
       int   getBuff(const int isRead, int Quantum);
       int   getData(const char *dtype, char *buff, int blen);
static int   mapMode(int mode);
static void  PidFile();
       int   Process2();
       void  Reset();
static int   rpCheck(char *fn, const char **opaque);
       int   rpEmsg(const char *op, char *fn);
       int   vpEmsg(const char *op, char *fn);
static int   Squash(char *);
static int   xapath(XrdOucStream &Config);
static int   xasync(XrdOucStream &Config);
static int   xcksum(XrdOucStream &Config);
static int   xexp(XrdOucStream &Config);
static int   xexpdo(char *path, int popt=0);
static int   xfsl(XrdOucStream &Config);
static int   xpidf(XrdOucStream &Config);
static int   xprep(XrdOucStream &Config);
static int   xlog(XrdOucStream &Config);
static int   xmon(XrdOucStream &Config);
static int   xred(XrdOucStream &Config);
static void  xred_set(RD_func func, const char *rHost, int rPort);
static int   xsecl(XrdOucStream &Config);
static int   xtrace(XrdOucStream &Config);

static XrdObjectQ<XrdXrootdProtocol> ProtStack;
XrdObject<XrdXrootdProtocol>         ProtLink;

protected:

static XrdXrootdXPath        RPList;    // Redirected paths
static XrdXrootdXPath        XPList;    // Exported   paths
static XrdSfsFileSystem     *osFS;      // The filesystem
static XrdSecService        *CIA;       // Authentication Server
static XrdXrootdFileLock    *Locker;    // File lock handler
static XrdScheduler         *Sched;     // System scheduler
static XrdBuffManager       *BPool;     // Buffer manager
static XrdSysError           eDest;     // Error message handler
static const char           *myInst;
static const char           *TraceID;
static       char           *pidPath;
static int                   myPID;

// Admin control area
//
static XrdNetSocket         *AdminSock;

// Processing configuration values
//
static int                 hailWait;
static int                 readWait;
static int                 Port;
static int                 Window;
static int                 WANPort;
static int                 WANWindow;
static char               *SecLib;
static char               *FSLib;
static char               *Notify;
static char                isRedir;
static char                chkfsV;
static XrdXrootdJob       *JobCKS;
static char               *JobCKT;

// Static redirection
//
static struct RD_Table {char *Host; int Port;} Route[RD_Num];

// async configuration values
//
static int                 as_maxperlnk; // Max async requests per link
static int                 as_maxperreq; // Max async ops per request
static int                 as_maxpersrv; // Max async ops per server
static int                 as_miniosz;   // Min async request size
static int                 as_minsfsz;   // Min sendf request size
static int                 as_segsize;   // Aio quantum (optimal)
static int                 as_maxstalls; // Maximum stalls we will tolerate
static int                 as_force;     // aio to be forced
static int                 as_noaio;     // aio is disabled
static int                 as_nosf;      // sendfile is disabled
static int                 as_syncw;     // writes to be synchronous
static int                 maxBuffsz;    // Maximum buffer size we can have
static int                 maxTransz;    // Maximum transfer size we can have
static const int           maxRvecsz = 1024;   // Maximum read vector size

// Statistical area
//
static XrdXrootdStats     *SI;
int                        numReads;     // Count
int                        numReadP;     // Count
int                        numWrites;    // Count
int                        numFiles;     // Count

int                        cumReads;     // Count less numReads
int                        cumReadP;     // Count less numReadP
int                        cumWrites;    // Count less numWrites
long long                  totReadP;     // Bytes

// Data local to each protocol/link combination
//
XrdLink                   *Link;
XrdBuffer                 *argp;
XrdXrootdFileTable        *FTab;
XrdXrootdMonitor          *Monitor;
kXR_unt32                  monUID;
char                       monFILE;
char                       monIO;
char                       Status;
unsigned char              CapVer;

// Authentication area
//
XrdSecEntity              *Client;
XrdSecProtocol            *AuthProt;
XrdSecEntity               Entity;

// Buffer information, used to drive DoIt(), getData(), and (*Resume)()
//
XrdXrootdAioReq           *myAioReq;
char                      *myBuff;
int                        myBlen;
int                        myBlast;
int                       (XrdXrootdProtocol::*Resume)();
XrdXrootdFile             *myFile;
long long                  myOffset;
int                        myIOLen;
int                        myStalls;

// Buffer resize control area
//
static int                 hcMax;
       int                 hcPrev;
       int                 hcNext;
       int                 hcNow;
       int                 halfBSize;

// This area is used for parallel streams
//
static const int           maxStreams = 16;
XrdSysMutex                streamMutex;
XrdSysSemaphore           *reTry;
XrdXrootdProtocol         *Stream[maxStreams];
unsigned int               mySID;
char                       isActive;
char                       isDead;
char                       isBound;
char                       isNOP;

static const int           maxPio = 4;
XrdXrootdPio              *pioFirst;
XrdXrootdPio              *pioLast;
XrdXrootdPio              *pioFree;

short                      PathID;
char                       doWrite;
char                       doWriteC;

// Buffers to handle client requests
//
XrdXrootdReqID             ReqID;
ClientRequest              Request;
XrdXrootdResponse          Response;
};
#endif
