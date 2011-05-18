/******************************************************************************/
/*                                                                            */
/*                   X r d X r o o t d M o n i t o r . c c                    */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

const char *XrdXrootdMonitorCVSID = "$Id$";

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#if !defined(__macos__) && !defined(__FreeBSD__)
#include <malloc.h>
#endif

#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

#include "Xrd/XrdScheduler.hh"
#include "XrdXrootd/XrdXrootdMonitor.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"

/******************************************************************************/
/*                     S t a t i c   A l l o c a t i o n                      */
/******************************************************************************/
  
XrdScheduler      *XrdXrootdMonitor::Sched      = 0;
XrdSysError       *XrdXrootdMonitor::eDest      = 0;
int                XrdXrootdMonitor::monFD;
char              *XrdXrootdMonitor::Dest1      = 0;
int                XrdXrootdMonitor::monMode1   = 0;
struct sockaddr    XrdXrootdMonitor::InetAddr1;
char              *XrdXrootdMonitor::Dest2      = 0;
int                XrdXrootdMonitor::monMode2   = 0;
struct sockaddr    XrdXrootdMonitor::InetAddr2;
XrdXrootdMonitor  *XrdXrootdMonitor::altMon     = 0;
XrdSysMutex        XrdXrootdMonitor::windowMutex;
kXR_int32          XrdXrootdMonitor::startTime  = 0;
int                XrdXrootdMonitor::monBlen    = 0;
int                XrdXrootdMonitor::lastEnt    = 0;
int                XrdXrootdMonitor::isEnabled  = 0;
int                XrdXrootdMonitor::numMonitor = 0;
int                XrdXrootdMonitor::autoFlush  = 600;
int                XrdXrootdMonitor::FlushTime  = 0;
kXR_int32          XrdXrootdMonitor::currWindow = 0;
kXR_int32          XrdXrootdMonitor::sizeWindow = 60;
char               XrdXrootdMonitor::monINFO    = 0;
char               XrdXrootdMonitor::monIO      = 0;
char               XrdXrootdMonitor::monFILE    = 0;
char               XrdXrootdMonitor::monSTAGE   = 0;
char               XrdXrootdMonitor::monUSER    = 0;

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
extern          XrdOucTrace       *XrdXrootdTrace;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
/******************************************************************************/
/*           C l a s s   X r d X r o o t d M o n i t o r _ T i c k            */
/******************************************************************************/

class XrdXrootdMonitor_Tick : public XrdJob
{
public:

void          DoIt() {
#ifndef NODEBUG
                      const char *TraceID = "MonTick";
#endif
                      time_t Now = XrdXrootdMonitor::Tick();
                      if (Window && Now)
                         Sched->Schedule((XrdJob *)this, Now+Window);
                         else {TRACE(DEBUG, "Monitor clock stopping.");}
                     }

void          Set(XrdScheduler *sp, int intvl) {Sched = sp; Window = intvl;}

      XrdXrootdMonitor_Tick() : XrdJob("monitor window clock")
                                  {Sched = 0; Window = 0;}
     ~XrdXrootdMonitor_Tick() {}

private:
XrdScheduler  *Sched;     // System scheduler
int            Window;
};

/******************************************************************************/
/*            C l a s s   X r d X r o o t d M o n i t o r L o c k             */
/******************************************************************************/
  
class XrdXrootdMonitorLock
{
public:

static void Lock()   {monLock.Lock();}

static void UnLock() {monLock.UnLock();}

       XrdXrootdMonitorLock(XrdXrootdMonitor *theMonitor)
                {if (theMonitor != XrdXrootdMonitor::altMon) unLock = 0;
                    else {unLock = 1; monLock.Lock();}
                }
      ~XrdXrootdMonitorLock() {if (unLock) monLock.UnLock();}

private:

static XrdSysMutex monLock;
       char        unLock;
};

XrdSysMutex XrdXrootdMonitorLock::monLock;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdXrootdMonitor::XrdXrootdMonitor()
{
   kXR_int32 localWindow;

// Initialize the local window
//
   windowMutex.Lock();
   localWindow = currWindow;
   windowMutex.UnLock();

// Allocate a monitor buffer
//
   if (!(monBuff = (XrdXrootdMonBuff *)memalign(getpagesize(), monBlen)))
      eDest->Emsg("Monitor", "Unable to allocate monitor buffer.");
      else {nextEnt = 1;
            monBuff->info[0].arg0.rTot[0] = 0;
            monBuff->info[0].arg0.id[0]   = XROOTD_MON_WINDOW;
            monBuff->info[0].arg1.Window  =
            monBuff->info[0].arg2.Window  =
                     static_cast<kXR_int32>(ntohl(localWindow));
           }
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdXrootdMonitor::~XrdXrootdMonitor()
{
// Release buffer
   if (monBuff) {Flush(); free(monBuff);}
}

/******************************************************************************/
/*                                 a p p I D                                  */
/******************************************************************************/
  
void XrdXrootdMonitor::appID(char *id)
{

// Application ID's are only meaningful for io event recording
//
   if (this == altMon || !*id) return;

// Fill out the monitor record
//
   if (lastWindow != currWindow) Mark();
      else if (nextEnt == lastEnt) Flush();
   monBuff->info[nextEnt].arg0.id[0]  = XROOTD_MON_APPID;
   strncpy((char *)&monBuff->info[nextEnt].arg0.id[4], id,
           sizeof(XrdXrootdMonTrace)-4);
}

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdXrootdMonitor *XrdXrootdMonitor::Alloc(int force)
{
   XrdXrootdMonitor *mp;
   int lastVal;

// If enabled, create a new object (if possible). If we are not monitoring
// i/o then return the global object.
//
   if (!isEnabled || (isEnabled < 0 && !force)) mp = 0;
      else if (!monIO) mp = altMon;
              else if ((mp = new XrdXrootdMonitor()))
                      if (!(mp->monBuff)) {delete mp; mp = 0;}

// Check if we should turn on the monitor clock
//
   if (mp && isEnabled < 0)
      {windowMutex.Lock();
       lastVal = numMonitor; numMonitor++;
       if (!lastVal) startClock();
       windowMutex.UnLock();
      }

// All done
//
   return mp;
}

/******************************************************************************/
/*                                 C l o s e                                  */
/******************************************************************************/

void XrdXrootdMonitor::Close(kXR_unt32 dictid, long long rTot, long long wTot)
{
  XrdXrootdMonitorLock mLock(this);
  unsigned int rVal, wVal;

// Fill out the monitor record (we allow the compiler to correctly cast data)
//
   if (lastWindow != currWindow) Mark();
      else if (nextEnt == lastEnt) Flush();
   monBuff->info[nextEnt].arg0.id[0]    = XROOTD_MON_CLOSE;
   monBuff->info[nextEnt].arg0.id[1]    = do_Shift(rTot, rVal);
   monBuff->info[nextEnt].arg0.rTot[1]  = htonl(rVal);
   monBuff->info[nextEnt].arg0.id[2]    = do_Shift(wTot, wVal);
   monBuff->info[nextEnt].arg0.id[3]    = 0;
   monBuff->info[nextEnt].arg1.wTot     = htonl(wVal);
   monBuff->info[nextEnt++].arg2.dictid = dictid;

// Check if we need to duplicate this entry
//
   if (altMon && this != altMon) altMon->Dup(&monBuff->info[nextEnt-1]);
}

/******************************************************************************/
/*                                  D i s c                                   */
/******************************************************************************/

void XrdXrootdMonitor::Disc(kXR_unt32 dictid, int csec)
{
  XrdXrootdMonitorLock mLock(this);

// Check if this should not be included in the io trace
//
   if (this != altMon && monUSER == 1 && altMon)
      {altMon->Disc(dictid, csec); return;}

// Fill out the monitor record (let compiler cast the data correctly)
//
   if (lastWindow != currWindow) Mark();
      else if (nextEnt == lastEnt) Flush();
   monBuff->info[nextEnt].arg0.rTot[0]  = 0;
   monBuff->info[nextEnt].arg0.id[0]    = XROOTD_MON_DISC;
   monBuff->info[nextEnt].arg1.wTot     = htonl(csec);
   monBuff->info[nextEnt++].arg2.dictid = dictid;

// Check if we need to duplicate this entry
//
   if (altMon && this != altMon && monUSER == 3)
      altMon->Dup(&monBuff->info[nextEnt-1]);
}

/******************************************************************************/
/*                              D e f a u l t s                               */
/******************************************************************************/

void XrdXrootdMonitor::Defaults(char *dest1, int mode1, char *dest2, int mode2)
{
   int mmode;

// Make sure if we have a dest1 we have mode
//
   if (!dest1)
      {mode1 = (dest1 = dest2) ? mode2 : 0;
       dest2 = 0; mode2 = 0;
      } else if (!dest2) mode2 = 0;


// Set the default destinations (caller supplied strdup'd strings)
//
   if (Dest1) free(Dest1);
   Dest1 = dest1; monMode1 = mode1;
   if (Dest2) free(Dest2);
   Dest2 = dest2; monMode2 = mode2;

// Set overall monitor mode
//
   mmode     = mode1 | mode2;
   isEnabled = (mmode & XROOTD_MON_ALL  ? 1 :-1);
   monIO     = (mmode & XROOTD_MON_IO   ? 1 : 0);
   monINFO   = (mmode & XROOTD_MON_INFO ? 1 : 0);
   monFILE   = (mmode & XROOTD_MON_FILE ? 1 : 0) | monIO;
   monSTAGE  = (mmode & XROOTD_MON_STAGE? 1 : 0);
   monUSER   = (mmode & XROOTD_MON_USER ? 1 : 0);

// Check where user information should go
//
   if (((mode1 & XROOTD_MON_IO) && (mode1 & XROOTD_MON_USER))
   ||  ((mode2 & XROOTD_MON_IO) && (mode2 & XROOTD_MON_USER)))
      {if ((!(mode1 & XROOTD_MON_IO) && (mode1 & XROOTD_MON_USER))
       ||  (!(mode2 & XROOTD_MON_IO) && (mode2 & XROOTD_MON_USER))) monUSER = 3;
          else monUSER = 2;
      }

// Do final check
//
   if (Dest1 == 0 && Dest2 == 0) isEnabled = 0;
}

/******************************************************************************/

void XrdXrootdMonitor::Defaults(int msz, int wsz, int flush)
{

// Set default window size
//
   sizeWindow = (wsz <= 0 ? 60 : wsz);
   autoFlush  = (flush <= 0 ? 600 : flush);

// Set default monitor buffer size
//
   if (msz <= 0) msz = 8192;
      else if (msz < 1024) msz = 1024;
   lastEnt = (msz-sizeof(XrdXrootdMonHeader))/sizeof(XrdXrootdMonTrace);
   monBlen =  (lastEnt*sizeof(XrdXrootdMonTrace))+sizeof(XrdXrootdMonHeader);
   lastEnt--;
   startTime = htonl(time(0));
}
  
/******************************************************************************/
/*                                   D u p                                    */
/******************************************************************************/
  
void XrdXrootdMonitor::Dup(XrdXrootdMonTrace *mrec)
{
  XrdXrootdMonitorLock mLock(this);

// Fill out the monitor record
//
   if (lastWindow != currWindow) Mark();
      else if (nextEnt == lastEnt) Flush();
   memcpy(&monBuff->info[nextEnt],(const void *)mrec,sizeof(XrdXrootdMonTrace));
   nextEnt++;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdXrootdMonitor::Init(XrdScheduler *sp, XrdSysError *errp)
{
   XrdNet     myNetwork(errp, 0);
   XrdNetPeer monDest;
   char      *etext;

// Set various statics
//
   Sched = sp;
   eDest = errp;

// There is nothing to do unless we have been enabled via Defaults()
//
   if (!isEnabled) return 1;

// Allocate a socket for the primary destination
//
   if (!myNetwork.Relay(monDest, Dest1, XRDNET_SENDONLY)) return 0;
   monFD = monDest.fd;

// Get the address of the primary destination
//
   if (!XrdNetDNS::Host2Dest(Dest1, InetAddr1, &etext))
      {eDest->Emsg("Monitor", "setup monitor collector;", etext);
       return 0;
      }

// Get the address of the alternate destination, if we happen to have one
//
   if (Dest2 && !XrdNetDNS::Host2Dest(Dest2, InetAddr2, &etext))
      {eDest->Emsg("Monitor", "setup monitor collector;", etext);
       return 0;
      }

// If there is a destination that is only collecting file events, then
// allocate a global monitor object but don't start the timer just yet.
//
   if ((monMode1 && !(monMode1 & XROOTD_MON_IO))
   ||  (monMode2 && !(monMode2 & XROOTD_MON_IO)))
       if (!(altMon = new XrdXrootdMonitor()) || !altMon->monBuff)
          {if (altMon) {delete altMon; altMon = 0;}
           eDest->Emsg("Monitor","allocate monitor; insufficient storage.");
           return 0;
          }

// Turn on the monitoring clock if we need it running all the time
//
   if (isEnabled > 0) startClock();

// All done
//
   return 1;
}

/******************************************************************************/
/*                                   M a p                                    */
/******************************************************************************/
  
kXR_unt32 XrdXrootdMonitor::Map(const char code,
                                   const char *uname, const char *path)
{
     static XrdSysMutex  seqMutex;
     static unsigned int monSeqID = 1;
     XrdXrootdMonMap     map;
     int                 size, montype;
     unsigned int        mySeqID;

// Assign a unique ID for this entry
//
   seqMutex.Lock();
   mySeqID = monSeqID++;
   seqMutex.UnLock();

// Copy in the username and path
//
   map.dictid = htonl(mySeqID);
   strcpy(map.info, uname);
   size = strlen(uname);
   if (path)
      {*(map.info+size) = '\n';
       strlcpy(map.info+size+1, path, sizeof(map.info)-size-1);
       size = size + strlen(path) + 1;
      }

// Fill in the header
//
   size = sizeof(XrdXrootdMonHeader)+sizeof(kXR_int32)+size;
   fillHeader(&map.hdr, code, size);

// Route the packet to all destinations that need them
//
        if (code == XROOTD_MON_MAPUSER) montype = XROOTD_MON_USER;
   else if (code == XROOTD_MON_MAPPATH) montype = XROOTD_MON_PATH;
   else if (code == XROOTD_MON_MAPSTAG) montype = XROOTD_MON_STAGE;
   else                                 montype = XROOTD_MON_INFO;
   Send(montype, (void *)&map, size);

// Return the dictionary id
//
   return map.dictid;
}
  
  
/******************************************************************************/
/*                                  O p e n                                   */
/******************************************************************************/
  
void XrdXrootdMonitor::Open(kXR_unt32 dictid, off_t fsize)
{
  XrdXrootdMonitorLock mLock(this);

  if (lastWindow != currWindow) Mark();
     else if (nextEnt == lastEnt) Flush();
  h2nll(fsize, monBuff->info[nextEnt].arg0.val);
  monBuff->info[nextEnt].arg0.id[0]    = XROOTD_MON_OPEN;
  monBuff->info[nextEnt].arg1.buflen   = 0;
  monBuff->info[nextEnt++].arg2.dictid = dictid;

// Check if we need to duplicate this entry
//
   if (altMon && this != altMon) altMon->Dup(&monBuff->info[nextEnt-1]);
}

/******************************************************************************/
/*                                  T i c k                                   */
/******************************************************************************/
  
time_t XrdXrootdMonitor::Tick()
{
   time_t Now;
   windowMutex.Lock();
   Now = time(0);
   currWindow = static_cast<kXR_int32>(Now);
   if (isEnabled < 0 && !numMonitor) Now = 0;
   windowMutex.UnLock();

// Check to see if we should flush the alternate monitor
//
   if (altMon && currWindow >= FlushTime)
      {XrdXrootdMonitorLock::Lock();
       if (currWindow >= FlushTime)
          {if (altMon->nextEnt > 1) altMon->Flush();
              else FlushTime = currWindow + autoFlush;
          }
       XrdXrootdMonitorLock::UnLock();
      }

// All done
//
   return Now;
}

/******************************************************************************/
/*                               u n A l l o c                                */
/******************************************************************************/
  
void XrdXrootdMonitor::unAlloc(XrdXrootdMonitor *monp)
{

// We must delete this object if we are de-allocating the local monitor.
//
   if (monp != altMon) delete monp;

// Decrease number being monitored if in selective mode
//
   if (isEnabled < 0)
      {windowMutex.Lock();
       numMonitor--;
       windowMutex.UnLock();
      }
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                              d o _ S h i f t                               */
/******************************************************************************/
  
unsigned char XrdXrootdMonitor::do_Shift(long long xTot, unsigned int &xVal)
{
  const long long smask = 0x7fffffff00000000LL;
  unsigned char xshift = 0;

  while(xTot & smask) {xTot = xTot >> 1LL; xshift++;}
  xVal = static_cast<unsigned int>(xTot);

  return xshift;
}

/******************************************************************************/
/*                            f i l l H e a d e r                             */
/******************************************************************************/
  
void XrdXrootdMonitor::fillHeader(XrdXrootdMonHeader *hdr,
                                  const char          id, int size)
{  static XrdSysMutex seqMutex;
   static int         seq = 0;
          int         myseq;

// Generate a new sequence number
//
   seqMutex.Lock();
   myseq = 0x00ff & (seq++);
   seqMutex.UnLock();

// Fill in the header
//
   hdr->code = static_cast<kXR_char>(id);
   hdr->pseq = static_cast<kXR_char>(myseq);
   hdr->plen = htons(static_cast<uint16_t>(size));
   hdr->stod = startTime;
}
  
/******************************************************************************/
/*                                 F l u s h                                  */
/******************************************************************************/
  
void XrdXrootdMonitor::Flush()
{
   int       size;
   kXR_int32 localWindow, now;

// Do not flush if the buffer is empty
//
   if (nextEnt <= 1) return;

// Get the current window marker. Since it might be updated while
// we are getting it, get a mutex to make sure it's fully updated
//
   windowMutex.Lock();
   localWindow = currWindow;
   windowMutex.UnLock();

// Fill in the header and in the process we will have the current time
//
   size = (nextEnt+1)*sizeof(XrdXrootdMonTrace)+sizeof(XrdXrootdMonHeader);
   fillHeader(&monBuff->hdr, XROOTD_MON_MAPTRCE, size);

// Punt on the right ending time. We are trying to keep same-sized windows
//
   if (monBuff->info[0].arg2.Window  != localWindow) now = localWindow;
      else now = localWindow + sizeWindow;

// Place the ending timing mark, send off the buffer and reinitialize it
//
   monBuff->info[nextEnt].arg0.rTot[0] = 0;
   monBuff->info[nextEnt].arg0.id[0]   = XROOTD_MON_WINDOW;
   monBuff->info[nextEnt].arg1.Window  =
   monBuff->info[nextEnt].arg2.Window  = htonl(now);

   if (this != altMon) Send(XROOTD_MON_IO, (void *)monBuff, size);
      else {Send(XROOTD_MON_FILE, (void *)monBuff, size);
            FlushTime = localWindow + autoFlush;
           }

   monBuff->info[0].arg0.rTot[0] = 0;
   monBuff->info[0].arg0.id[0]   = XROOTD_MON_WINDOW;
   monBuff->info[0].arg1.Window  =
   monBuff->info[0].arg2.Window  = htonl(localWindow);
   nextEnt = 1;
}

/******************************************************************************/
/*                                  M a r k                                   */
/******************************************************************************/
  
void XrdXrootdMonitor::Mark()
{
   kXR_int32 localWindow;

// Get the current window marker. Since it might be updated while
// we are getting it, get a mutex to make sure it's fully updated
//
   windowMutex.Lock();
   localWindow = currWindow;
   windowMutex.UnLock();

// Now, optimize placing the window mark in the buffer
//
   if (monBuff->info[nextEnt-1].arg0.id[0] == XROOTD_MON_WINDOW)
      monBuff->info[nextEnt-1].arg2.Window =
               static_cast<kXR_int32>(ntohl(localWindow));
      else if (nextEnt+8 > lastEnt) Flush();
              else {monBuff->info[nextEnt].arg0.rTot[0] = 0;
                    monBuff->info[nextEnt].arg0.id[0]   = XROOTD_MON_WINDOW;
                    monBuff->info[nextEnt].arg1.Window  =
                             static_cast<kXR_int32>(ntohl(lastWindow));
                    monBuff->info[nextEnt].arg2.Window  =
                             static_cast<kXR_int32>(ntohl(localWindow));
                    nextEnt++;
                   }
     lastWindow = localWindow;
}
 
/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/
  
int XrdXrootdMonitor::Send(int monMode, void *buff, int blen)
{
#ifndef NODEBUG
    const char *TraceID = "Monitor";
#endif
    static XrdSysMutex sendMutex;
    int rc1, rc2;

    sendMutex.Lock();
    if (monMode & monMode1) 
       {rc1  = (int)sendto(monFD, buff, blen, 0,
                        (const struct sockaddr *)&InetAddr1, sizeof(sockaddr));
        TRACE(DEBUG,blen <<" bytes sent to " <<Dest1 <<" rc=" <<(rc1 ? errno : 0));
       }
       else rc1 = 0;
    if (monMode & monMode2) 
       {rc2 = (int)sendto(monFD, buff, blen, 0,
                        (const struct sockaddr *)&InetAddr2, sizeof(sockaddr));
        TRACE(DEBUG,blen <<" bytes sent to " <<Dest2 <<" rc=" <<(rc2 ? errno : 0));
       }
       else rc2 = 0;
    sendMutex.UnLock();

    return (rc1 > rc2 ? rc1 : rc2);
}

/******************************************************************************/
/*                            s t a r t C l o c k                             */
/******************************************************************************/
  
void XrdXrootdMonitor::startClock()
{
   static XrdXrootdMonitor_Tick MonTick;
          time_t Now;

// Start the clock (caller must have windowMutex locked, if necessary
//
   Now = time(0);
   currWindow = static_cast<kXR_int32>(Now);
   MonTick.Set(Sched, sizeWindow);
   FlushTime = autoFlush + currWindow;
   if (Sched) Sched->Schedule((XrdJob *)&MonTick, Now+sizeWindow);
}
