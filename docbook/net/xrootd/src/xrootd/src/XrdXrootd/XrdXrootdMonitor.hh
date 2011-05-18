#ifndef __XRDXROOTDMONITOR__
#define __XRDXROOTDMONITOR__
/******************************************************************************/
/*                                                                            */
/*                   X r d X r o o t d M o n i t o r . h h                    */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include <inttypes.h>
#include <time.h>
#include <netinet/in.h>
#include <sys/types.h>

#include "XrdNet/XrdNetPeer.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdXrootd/XrdXrootdMonData.hh"
#include "XProtocol/XPtypes.hh"
  
/******************************************************************************/
/*                            X r d M o n i t o r                             */
/******************************************************************************/

#define XROOTD_MON_ALL      1
#define XROOTD_MON_FILE     2
#define XROOTD_MON_IO       4
#define XROOTD_MON_INFO     8
#define XROOTD_MON_STAGE   16
#define XROOTD_MON_USER    32
#define XROOTD_MON_PATH    (XROOTD_MON_IO   | XROOTD_MON_FILE)

class XrdScheduler;
  
class XrdXrootdMonitor
{
public:
friend class XrdXrootdMonitorDummy; // Avoid stupid gcc warnings

// All values for Add_xx() must be passed in network byte order
//
inline void              Add_rd(kXR_unt32 dictid,
                                kXR_int32 rlen,
                                kXR_int64 offset)
                               {Add_io(dictid, rlen, offset);}

inline void              Add_wr(kXR_unt32 dictid,
                                kXR_int32 wlen, 
                                kXR_int64 offset)
                               {unsigned int temp = ~ntohl(wlen)+1;
                                Add_io(dictid,
                                      (kXR_int32)htonl(temp), offset);
                               }

       void              appID(char *id);

static XrdXrootdMonitor *Alloc(int force=0);

       void              Close(kXR_unt32 dictid, long long rTot, long long wTot);

       void              Disc(kXR_unt32 dictid, int csec);

static void              Defaults(char *dest1, int m1, char *dest2, int m2);
static void              Defaults(int msz, int wsz, int flush);

       void              Dup(XrdXrootdMonTrace *mrec);

static int               Init(XrdScheduler *sp, XrdSysError *errp);

static kXR_unt32         Map(const char code,const char *uname,const char *path);

       void              Open(kXR_unt32 dictid, off_t fsize);

static time_t            Tick();

static void              unAlloc(XrdXrootdMonitor *monp);

static XrdXrootdMonitor *altMon;
static char              monIO;
static char              monINFO;
static char              monFILE;
static char              monSTAGE;
static char              monUSER;

                         XrdXrootdMonitor();

private:
                        ~XrdXrootdMonitor(); 

inline void              Add_io(kXR_unt32 dictid, kXR_int32 buflen,
                                kXR_int64 offset);
       unsigned char     do_Shift(long long xTot, unsigned int &xVal);
static void              fillHeader(XrdXrootdMonHeader *hdr,
                                    const char id, int size);
       void              Flush();
       void              Mark();
static int               Send(int mmode, void *buff, int size);
static void              startClock();

static XrdScheduler      *Sched;
static XrdSysError       *eDest;
static XrdSysMutex        windowMutex;
static int                monFD;
static char              *Dest1;
static int                monMode1;
static struct sockaddr    InetAddr1;
static char              *Dest2;
static int                monMode2;
static struct sockaddr    InetAddr2;
       XrdXrootdMonBuff  *monBuff;
static int                monBlen;
       int                nextEnt;
static int                lastEnt;
static int                autoFlush;
static int                FlushTime;
static kXR_int32          startTime;
       kXR_int32          lastWindow;
static kXR_int32          currWindow;
static kXR_int32          sizeWindow;
static int                isEnabled;
static int                numMonitor;
};

/******************************************************************************/
/*                      I n l i n e   F u n c t i o n s                       */
/******************************************************************************/
/******************************************************************************/
/*                                A d d _ i o                                 */
/******************************************************************************/
  
void XrdXrootdMonitor::Add_io(kXR_unt32 dictid,kXR_int32 blen,kXR_int64 offset)
     {if (lastWindow != currWindow) Mark();
         else if (nextEnt == lastEnt) Flush();
      monBuff->info[nextEnt].arg0.val      = offset;
      monBuff->info[nextEnt].arg1.buflen   = blen;
      monBuff->info[nextEnt++].arg2.dictid = dictid;
     }
#endif
