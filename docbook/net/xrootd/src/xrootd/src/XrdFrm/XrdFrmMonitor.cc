/******************************************************************************/
/*                                                                            */
/*                      X r d F r m M o n i t o r . c c                       */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

const char *XrdFrmMonitorCVSID = "$Id$";

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmMonitor.hh"
#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                     S t a t i c   A l l o c a t i o n                      */
/******************************************************************************/
  
char              *XrdFrmMonitor::Dest1      = 0;
int                XrdFrmMonitor::monFD1     = -1;
int                XrdFrmMonitor::monMode1   = 0;
struct sockaddr    XrdFrmMonitor::InetAddr1;
char              *XrdFrmMonitor::Dest2      = 0;
int                XrdFrmMonitor::monFD2     = -1;
int                XrdFrmMonitor::monMode2   = 0;
struct sockaddr    XrdFrmMonitor::InetAddr2;
kXR_int32          XrdFrmMonitor::startTime  = 0;
int                XrdFrmMonitor::isEnabled  = 0;
char               XrdFrmMonitor::monSTAGE   = 0;

/******************************************************************************/
/*                              D e f a u l t s                               */
/******************************************************************************/

void XrdFrmMonitor::Defaults(char *dest1, int mode1, char *dest2, int mode2)
{

// Make sure if we have a proper destinations and modes
//
   if (dest1 && !mode1) {free(dest1); dest1 = 0; mode1 = 0;}
   if (dest2 && !mode2) {free(dest2); dest2 = 0; mode2 = 0;}

// Propogate the destinations
//
   if (!dest1)
      {mode1 = (dest1 = dest2) ? mode2 : 0;
       dest2 = 0; mode2 = 0;
      }

// Set the default destinations (caller supplied strdup'd strings)
//
   if (Dest1) free(Dest1);
   Dest1 = dest1; monMode1 = mode1;
   if (Dest2) free(Dest2);
   Dest2 = dest2; monMode2 = mode2;

// Set overall monitor mode
//
   monSTAGE  = ((mode1 | mode2) & XROOTD_MON_STAGE ? 1 : 0);

// Do final check
//
   isEnabled = (Dest1 == 0 && Dest2 == 0 ? 0 : 1);
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdFrmMonitor::Init()
{
   XrdNet     myNetwork(&Say, 0);
   XrdNetPeer monDest;
   char      *etext;

// There is nothing to do unless we have been enabled via Defaults()
//
   if (!isEnabled) return 1;

// Get the address of the primary destination
//
   if (!XrdNetDNS::Host2Dest(Dest1, InetAddr1, &etext))
      {Say.Emsg("Monitor", "setup monitor collector;", etext);
       return 0;
      }

// Allocate a socket for the primary destination
//
   if (!myNetwork.Relay(monDest, Dest1, XRDNET_SENDONLY)) return 0;
   monFD1 = monDest.fd;

// Do the same for the secondary destination
//
   if (Dest2)
      {if (!XrdNetDNS::Host2Dest(Dest2, InetAddr2, &etext))
          {Say.Emsg("Monitor", "setup monitor collector;", etext);
           return 0;
          }
       if (!myNetwork.Relay(monDest, Dest2, XRDNET_SENDONLY)) return 0;
       monFD2 = monDest.fd;
      }

// All done
//
   startTime = htonl(time(0));
   return 1;
}

/******************************************************************************/
/*                                   M a p                                    */
/******************************************************************************/
  
kXR_unt32 XrdFrmMonitor::Map(const char code,
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
        if (code == XROOTD_MON_MAPSTAG) montype = XROOTD_MON_STAGE;
   else                                 montype = XROOTD_MON_INFO;
   Send(montype, (void *)&map, size);

// Return the dictionary id
//
   return map.dictid;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                            f i l l H e a d e r                             */
/******************************************************************************/
  
void XrdFrmMonitor::fillHeader(XrdXrootdMonHeader *hdr,
                               const char id, int size)
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
/*                                  S e n d                                   */
/******************************************************************************/
  
int XrdFrmMonitor::Send(int monMode, void *buff, int blen)
{
    EPNAME("Send");
    static XrdSysMutex sendMutex;
    int rc1, rc2;

    sendMutex.Lock();
    if (monMode & monMode1 && monFD1 >= 0)
       {rc1  = (int)sendto(monFD1, buff, blen, 0,
                        (const struct sockaddr *)&InetAddr1, sizeof(sockaddr));
        DEBUG(blen <<" bytes sent to " <<Dest1 <<" rc=" <<(rc1 ? errno : 0));
       }
       else rc1 = 0;
    if (monMode & monMode2 && monFD2 >= 0)
       {rc2 = (int)sendto(monFD2, buff, blen, 0,
                        (const struct sockaddr *)&InetAddr2, sizeof(sockaddr));
        DEBUG(blen <<" bytes sent to " <<Dest2 <<" rc=" <<(rc2 ? errno : 0));
       }
       else rc2 = 0;
    sendMutex.UnLock();

    return (rc1 > rc2 ? rc1 : rc2);
}
