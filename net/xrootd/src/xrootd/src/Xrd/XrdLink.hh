#ifndef __XRD_LINK_H__
#define __XRD_LINK_H__
/******************************************************************************/
/*                                                                            */
/*                            X r d L i n k . h h                             */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/socket.h>
#include <sys/types.h>
#include <fcntl.h>
#include <time.h>

#include "XrdSys/XrdSysPthread.hh"

#include "Xrd/XrdJob.hh"
#include "Xrd/XrdLinkMatch.hh"
#include "Xrd/XrdProtocol.hh"
  
/******************************************************************************/
/*                       X r d L i n k   O p t i o n s                        */
/******************************************************************************/
  
#define XRDLINK_RDLOCK  0x0001
#define XRDLINK_NOCLOSE 0x0002

/******************************************************************************/
/*                      C l a s s   D e f i n i t i o n                       */
/******************************************************************************/
  
class XrdNetBuffer;
class XrdNetPeer;
class XrdPoll;

class XrdLink : XrdJob
{
public:
friend class XrdLinkScan;
friend class XrdPoll;
friend class XrdPollPoll;
friend class XrdPollDev;
friend class XrdPollE;

static XrdLink *Alloc(XrdNetPeer &Peer, int opts=0);

void          Bind();
void          Bind(pthread_t tid);

int           Client(char *buff, int blen);

int           Close(int defer=0);

void          DoIt();

int           FDnum() {return FD;}

static XrdLink *fd2link(int fd)
                {if (fd < 0) fd = -fd; 
                 return (fd <= LTLast && LinkBat[fd] ? LinkTab[fd] : 0);
                }

static XrdLink *fd2link(int fd, unsigned int inst)
                {if (fd < 0) fd = -fd; 
                 if (fd <= LTLast && LinkBat[fd] && LinkTab[fd]
                 && LinkTab[fd]->Instance == inst) return LinkTab[fd];
                 return (XrdLink *)0;
                }

static XrdLink *Find(int &curr, XrdLinkMatch *who=0);

       int    getIOStats(long long &inbytes, long long &outbytes,
                              int  &numstall,     int  &numtardy)
                        { inbytes = BytesIn + BytesInTot;
                         outbytes = BytesOut+BytesOutTot;
                         numstall = stallCnt + stallCntTot;
                         numtardy = tardyCnt + tardyCntTot;
                         return InUse;
                        }

static int    getName(int &curr, char *bname, int blen, XrdLinkMatch *who=0);

XrdProtocol  *getProtocol() {return Protocol;} // opmutex must be locked

void          Hold(int lk) {(lk ? opMutex.Lock() : opMutex.UnLock());}

char         *ID;      // This is referenced a lot

unsigned int  Inst() {return Instance;}

int           isFlawed() {return Etext != 0;}

int           isInstance(unsigned int inst)
                        {return FD >= 0 && Instance == inst;}

const char   *Name(sockaddr *ipaddr=0)
                     {if (ipaddr) memcpy(ipaddr, &InetAddr, sizeof(sockaddr));
                      return (const char *)Lname;
                     }

const char   *Host(sockaddr *ipaddr=0)
                     {if (ipaddr) memcpy(ipaddr, &InetAddr, sizeof(sockaddr));
                      return (const char *)HostName;
                     }

int           Peek(char *buff, int blen, int timeout=-1);

int           Recv(char *buff, int blen);
int           Recv(char *buff, int blen, int timeout);

int           RecvAll(char *buff, int blen, int timeout=-1);

int           Send(const char *buff, int blen);
int           Send(const struct iovec *iov, int iocnt, int bytes=0);

struct sfVec {union {char *buffer;    // ->Data if fdnum < 0
                     off_t offset;    // File offset      of data
                    };
              int   sendsz;           // Length of data at offset
              int   fdnum;            // File descriptor for data
             };
static const int sfMax = 8;

static int    sfOK;                   // True if Send(sfVec) enabled

int           Send(const struct sfVec *sdP, int sdn); // Iff sfOK > 0

void          Serialize();                              // ASYNC Mode

int           setEtext(const char *text);

void          setID(const char *userid, int procid);

static void   setKWT(int wkSec, int kwSec);

XrdProtocol  *setProtocol(XrdProtocol *pp);

void          setRef(int cnt);                          // ASYNC Mode

static int    Setup(int maxfd, int idlewait);

static int    Stats(char *buff, int blen, int do_sync=0);

       void   syncStats(int *ctime=0);

       int    Terminate(const XrdLink *owner, int fdnum, unsigned int inst);

time_t        timeCon() {return conTime;}

int           UseCnt() {return InUse;}

              XrdLink();
             ~XrdLink() {}  // Is never deleted!

private:

void   Reset();
int    sendData(const char *Buff, int Blen);

static XrdSysMutex   LTMutex;    // For the LinkTab only LTMutex->IOMutex allowed
static XrdLink     **LinkTab;
static char         *LinkBat;
static unsigned int  LinkAlloc;
static int           LTLast;
static const char   *TraceID;
static int           devNull;
static short         killWait;
static short         waitKill;

// Statistical area (global and local)
//
static long long    LinkBytesIn;
static long long    LinkBytesOut;
static long long    LinkConTime;
static long long    LinkCountTot;
static int          LinkCount;
static int          LinkCountMax;
static int          LinkTimeOuts;
static int          LinkStalls;
static int          LinkSfIntr;
       long long        BytesIn;
       long long        BytesInTot;
       long long        BytesOut;
       long long        BytesOutTot;
       int              stallCnt;
       int              stallCntTot;
       int              tardyCnt;
       int              tardyCntTot;
       int              SfIntr;
static XrdSysMutex  statsMutex;

// Identification section
//
struct sockaddr     InetAddr;
char                Uname[24];  // Uname and Lname must be adjacent!
char                Lname[232];
char               *HostName;
int                 HNlen;
pthread_t           TID;

XrdSysMutex         opMutex;
XrdSysMutex         rdMutex;
XrdSysMutex         wrMutex;
XrdSysSemaphore     IOSemaphore;
XrdSysCondVar      *KillcvP;        // Protected by opMutex!
XrdLink            *Next;
XrdNetBuffer       *udpbuff;
XrdProtocol        *Protocol;
XrdProtocol        *ProtoAlt;
XrdPoll            *Poller;
struct pollfd      *PollEnt;
char               *Etext;
int                 FD;
unsigned int        Instance;
time_t              conTime;
int                 InUse;
int                 doPost;
char                LockReads;
char                KeepFD;
char                isEnabled;
char                isIdle;
char                inQ;
char                tBound;
char                KillCnt;        // Protected by opMutex!
static const char   KillMax =   60;
static const char   KillMsk = 0x7f;
static const char   KillXwt = 0x80;
};
#endif
