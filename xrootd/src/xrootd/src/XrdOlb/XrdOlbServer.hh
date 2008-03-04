#ifndef __OLB_SERVER__H
#define __OLB_SERVER__H
/******************************************************************************/
/*                                                                            */
/*                       X r d O l b S e r v e r . h h                        */
/*                                                                            */
/* (c) 2002 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <string.h>
#include <unistd.h>
#include <sys/uio.h>
  
#include "XrdNet/XrdNetLink.hh"
#include "XrdOlb/XrdOlbTypes.hh"
#include "XrdOlb/XrdOlbReq.hh"
#include "XrdOlb/XrdOlbRRQ.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdOlbCInfo;
class XrdOlbDrop;
class XrdOlbPInfo;
class XrdOlbPrepArgs;

class XrdOlbServer
{
public:
friend class XrdOlbManager;

       char  isDisable;    // Set via admin command to temporarily remove server
       char  isOffline;    // Set when a link failure occurs
       char  isNoStage;    // Set upon a nostage event
       char  isSpecial;    // Set when server can be redirected
       char  isMan;        // Set when server can act as manager
       char  isPeer;       // Set when server can act as peer manager
       char  isProxy;      // Set when server can act as a proxy
       char  isSuspend;    // Set upon a suspend event
       char  isActive;     // Set when server is functioning
       char  isBound;      // Set when server is in the configuration
       char  isRW;         // Set when server can write or stage data
       char  isKnown;      // Set when we have recieved a "state"
       char  isBusy;       // Set when server has an active thread
       char  isGone;       // Set when server must be deleted

       int   do_Locate(char *, const char *, SMask_t hfVec, SMask_t rwVec);

inline int   Inst() {return Instance;}

inline int   isServer(SMask_t smask) {return (smask & ServMask) != 0;}
inline int   isServer(const char *hn)
                      {return Link && !strcmp(Link->Name(), hn);}
inline int   isServer(unsigned int ipa)
                      {return ipa == IPAddr;}
inline int   isServer(unsigned int ipa, int port)
                      {return ipa == IPAddr && port == Port && port;}
inline int   isServer(unsigned int ipa, int port, char *sid)
                      {if (sid) return ipa == IPAddr && !strcmp(mySID, sid);
                                return ipa == IPAddr && port == Port && port;
                      }
inline       char *Name()   {return (myName ? myName : (char *)"?");}
inline const char *Nick()   {return (myNick ? myNick : (char *)"?");}
inline void    Lock() {myMutex.Lock();}
inline void  UnLock() {myMutex.UnLock();}

       int  Login(int Port, int Status, int Lvl);

       void Process_Director(void);
       int  Process_Requests(void);
       int  Process_Responses(void);

static int  Resume(XrdOlbPrepArgs *pargs);

       int  Send(const char *buff, int blen=0);
       int  Send(const struct iovec *iov, int iovcnt);

       void setName(XrdNetLink *lnkp, int port);

static void setRelay(XrdNetLink *rlyp) {Relay = rlyp;}

static void setSpace(int Dfree, int Dutil) {dsk_free = Dfree; dsk_totu = Dutil;}

            XrdOlbServer(XrdNetLink *lnkp, int port=0, char *sid=0);
           ~XrdOlbServer();

private:
       int   do_AvKb(char *rid);
       int   do_Chmod(char *rid, int do4real);
       int   do_Delay(char *rid);
       int   do_Disc(char *rid, int sendDisc);
       int   do_Gone(char *rid);
       int   do_Have(char *rid);
       int   do_Load(char *rid);
       int   do_Mkdir(char *rid, int do4real);
       int   do_Mkpath(char *rid, int do4real);
       int   do_Mv(char *rid, int do4real);
       int   do_Ping(char *rid);
       int   do_Pong(char *rid);
       int   do_Port(char *rid);
       int   do_PrepAdd(char *rid, int server=0);
static int   do_PrepAdd4Real(XrdOlbPrepArgs *pargs);
       int   do_PrepDel(char *rid, int server=0);
static int   do_PrepSel(XrdOlbPrepArgs *pargs, int stage);
       int   do_Rm(char *rid, int do4real);
       int   do_Rmdir(char *rid, int do4real);
       int   do_RST(char *rid);
       int   do_Select(char *rid, int refresh=0);
       int   do_Space(char *rid);
       int   do_State(char *rid, int reset);
       int   do_StateFWD(char *tp, int reset);
       int   do_Stats(char *rid, int wantdata);
       int   do_StNst(char *rid, int Resume);
       int   do_SuRes(char *rid, int Resume);
       int   do_Try(char *rid);
       int   do_Usage(char *rid);
       int   getMode(const char *, const char *, const char *, mode_t &);
static int   Inform(const char *cmd, XrdOlbPrepArgs *pargs);
static int   isOnline(char *path, int upt=1, XrdNetLink *lnk=0);
       int   Mkpath(char *local_path, mode_t mode);
       char *prepScan(char **Line,XrdOlbPrepArgs *pargs,const char *Etxt);
       char *Receive(char *idbuff, int blen);
       int   Reissue(char *rid, const char *op, char *arg1, char *path, char *arg3=0);

static XrdNetLink   *Relay;
XrdSysMutex       myMutex;
XrdNetLink       *Link;
unsigned int      IPAddr;
XrdOlbServer     *Next;
time_t            DropTime;
XrdOlbDrop       *DropJob;
XrdOlbRRQInfo     Info;
XrdOlbReq         Req;

SMask_t    ServMask;
int        ServID;
int        Instance;
int        Port;
int        myLevel;
char      *mySID;
char      *myName;
char      *myNick;
char      *Stype;

static     const int     redr_iov_cnt = 3;
           struct iovec  redr_iov[redr_iov_cnt];

int        pingpong;     // Keep alive field
int        newload;
int        logload;
int        DiskFree;     // Largest free KB
int        DiskNums;     // Number of file systems
int        DiskTotu;     // Total disk utilization
int        myCost;       // Overall cost (determined by location)
int        myLoad;       // Overall load
int        RefA;         // Number of times used for allocation
int        RefTotA;
int        RefR;         // Number of times used for redirection
int        RefTotR;

// The following fields are used to keep the supervisor's load values
//
static XrdSysMutex mlMutex;
static int         xeq_load;
static int         cpu_load;
static int         mem_load;
static int         pag_load;
static int         net_load;
static int         dsk_free;
static int         dsk_totu;
};
#endif
