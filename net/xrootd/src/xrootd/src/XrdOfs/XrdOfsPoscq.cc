/******************************************************************************/
/*                                                                            */
/*                        X r d O f s P o s c q . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdOfsPoscqCVSID = "$Id$";

#include <string.h>
#include <strings.h>
#include <stddef.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdOfs/XrdOfsPoscq.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdOfsPoscq::XrdOfsPoscq(XrdSysError *erp, XrdOss *oss, const char *fn)
{
   eDest = erp;
   ossFS = oss;
   pocFN = strdup(fn);
   pocFD = -1;
   pocSZ = 0;
   pocIQ = 0;
   SlotList = SlotLust = 0;
}
  
/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
int XrdOfsPoscq::Add(const char *Tident, const char *Lfn)
{
   XrdOfsPoscq::Request tmpReq;
   FileSlot *freeSlot;
   int fP;

// Construct the request
//
   tmpReq.addT = 0;
   strlcpy(tmpReq.LFN,  Lfn,    sizeof(tmpReq.LFN));
   strlcpy(tmpReq.User, Tident, sizeof(tmpReq.User));
   memset(tmpReq.Reserved, 0, sizeof(tmpReq.Reserved));

// Obtain a free slot
//
   myMutex.Lock();
   if ((freeSlot = SlotList))
      {fP = freeSlot->Offset;
       SlotList = freeSlot->Next;
       freeSlot->Next = SlotLust;
       SlotLust = freeSlot;
      } else {fP = pocSZ; pocSZ += ReqSize;}
   pocIQ++;
   myMutex.UnLock();

// Write out the record
//
   if (!reqWrite((void *)&tmpReq, sizeof(tmpReq), fP))
      {eDest->Emsg("Add", Lfn, "not added to the persist queue.");
       myMutex.Lock(); pocIQ--; myMutex.UnLock();
       return -EIO;
      }

// Return the record offset
//
   return fP;
}
  
/******************************************************************************/
/*                                C o m m i t                                 */
/******************************************************************************/

int XrdOfsPoscq::Commit(const char *Lfn, int Offset)
{
   long long addT = static_cast<long long>(time(0));

// Verify the offset it must be correct
//
   if (!VerOffset(Lfn, Offset)) return -EINVAL;

// Indicate the record is free
//
   if (reqWrite((void *)&addT, sizeof(addT), Offset)) return 0;
   eDest->Emsg("Commit", Lfn, "not commited to the persist queue.");
   return -EIO;
}

/******************************************************************************/
/*                                   D e l                                    */
/******************************************************************************/

int XrdOfsPoscq::Del(const char *Lfn, int Offset, int Unlink)
{
   static int Zero = 0;
   FileSlot *freeSlot;
   int retc;

// Verify the offset it must be correct
//
   if (!VerOffset(Lfn, Offset)) return -EINVAL;

// Unlink the file if need be
//
   if (Unlink && (retc = ossFS->Unlink(Lfn)) && retc != -ENOENT)
      {eDest->Emsg("Del", retc, "remove", Lfn);
       return (retc < 0 ? retc : -retc);
      }

// Indicate the record is free
//
   if (!reqWrite((void *)&Zero, sizeof(Zero), Offset+offsetof(Request,LFN)))
      {eDest->Emsg("Del", Lfn, "not removed from the persist queue.");
       return -EIO;
      }

// Serialize and place this on the free queue
//
   myMutex.Lock();
   if ((freeSlot = SlotLust)) SlotLust = freeSlot->Next;
      else freeSlot = new FileSlot;
   freeSlot->Offset = Offset;
   freeSlot->Next   = SlotList;
   SlotList         = freeSlot;
   if (pocIQ > 0) pocIQ--;
   myMutex.UnLock();

// All done
//
   return 0;
}
  
/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/

XrdOfsPoscq::recEnt *XrdOfsPoscq::Init(int &Ok)
{
   static const int Mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   Request     tmpReq;
   struct stat buf, Stat;
   recEnt     *First = 0;
   char        Buff[80];
   int         rc, Offs, numreq = 0;

// Assume we will fail
//
   Ok = 0;

// Open the file first in r/w mode
//
   if ((pocFD = open(pocFN, O_RDWR|O_CREAT, Mode)) < 0)
      {eDest->Emsg("Init",errno,"open",pocFN);
       return 0;
      }

// Get file status
//
   if (fstat(pocFD, &buf)) {FailIni("stat"); return 0;}

// Check for a new file here
//
   if (buf.st_size < ReqSize)
      {pocSZ = ReqOffs;
       if (ftruncate(pocFD, ReqOffs)) FailIni("trunc");
          else Ok = 1;
       return 0;
      }

// Read the full file
//
   for (Offs = ReqOffs; Offs < buf.st_size; Offs += ReqSize)
       {do {rc = pread(pocFD, (void *)&tmpReq, ReqSize, Offs);}
           while(rc < 0 && errno == EINTR);
        if (rc < 0) {eDest->Emsg("Init",errno,"read",pocFN); return First;}
        if (*tmpReq.LFN == '\0'
        ||  ossFS->Stat(tmpReq.LFN, &Stat)
        ||  !(S_ISREG(Stat.st_mode) || !(Stat.st_mode & S_ISUID))) continue;
        First = new recEnt(tmpReq, Stat.st_mode & S_IAMB, First); numreq++;
       }

// Now write out the file and return
//
   sprintf(Buff, " %d pending create%s", numreq, (numreq != 1 ? "s" : ""));
   eDest->Say("Init", Buff, " recovered from ", pocFN);
   if (ReWrite(First)) Ok = 1;
   return First;
}
  
/******************************************************************************/
/*                                  L i s t                                   */
/******************************************************************************/
  
XrdOfsPoscq::recEnt *XrdOfsPoscq::List(XrdSysError *Say, const char *theFN)
{
   XrdOfsPoscq::Request tmpReq;
   struct stat buf;
   recEnt *First = 0;
   int    rc, theFD, Offs;

// Open the file first in r/o mode
//
   if ((theFD = open(theFN, O_RDONLY)) < 0)
      {Say->Emsg("Init",errno,"open",theFN);
       return 0;
      }

// Get file status
//
   if (fstat(theFD, &buf))
      {Say->Emsg("Init",errno,"stat",theFN);
       close(theFD);
       return 0;
      }
   if (buf.st_size < ReqSize) buf.st_size = 0;

// Read the full file
//
   for (Offs = ReqOffs; Offs < buf.st_size; Offs += ReqSize)
       {do {rc = pread(theFD, (void *)&tmpReq, ReqSize, Offs);}
           while(rc < 0 && errno == EINTR);
        if (rc < 0) {Say->Emsg("List",errno,"read",theFN); return First;}
        if (*tmpReq.LFN != '\0') First = new recEnt(tmpReq, 0, First);
       }

// All done
//
   close(theFD);
   return First;
}

/******************************************************************************/
/*                               F a i l I n i                                */
/******************************************************************************/

void XrdOfsPoscq::FailIni(const char *txt)
{
   eDest->Emsg("Init", errno, txt, pocFN);
}

/******************************************************************************/
/*                              r e q W r i t e                               */
/******************************************************************************/
  
int XrdOfsPoscq::reqWrite(void *Buff, int Bsz, int Offs)
{
   int rc = 0;

   do {rc = pwrite(pocFD, Buff, Bsz, Offs);} while(rc < 0 && errno == EINTR);

   if (rc >= 0 && Bsz > 8) rc = fsync(pocFD);

   if (rc < 0) {eDest->Emsg("reqWrite",errno,"write", pocFN); return 0;}
   return 1;
}

/******************************************************************************/
/*                               R e W r i t e                                */
/******************************************************************************/
  
int XrdOfsPoscq::ReWrite(XrdOfsPoscq::recEnt *rP)
{
   static const int Mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   char newFN[MAXPATHLEN], *oldFN;
   int  newFD, oldFD, Offs = ReqOffs, aOK = 1;

// Construct new file and open it
//
   strcpy(newFN, pocFN); strcat(newFN, ".new");
   if ((newFD = open(newFN, O_RDWR|O_CREAT|O_TRUNC, Mode)) < 0)
      {eDest->Emsg("ReWrite",errno,"open",newFN); return 0;}

// Setup to write/swap the file
//
   oldFD = pocFD; pocFD = newFD;
   oldFN = pocFN; pocFN = newFN;

// Rewrite all records if we have any
//
   while(rP)
        {rP->Offset = Offs;
         if (!reqWrite((void *)&rP->reqData, ReqSize, Offs)) {aOK = 0; break;}
         Offs += ReqSize;
         rP = rP->Next;
        }

// If all went well, rename the file
//
   if (aOK && rename(newFN, oldFN) < 0)
      {eDest->Emsg("ReWrite",errno,"rename",newFN); aOK = 0;}

// Perform post processing
//
   if (aOK)  close(oldFD);
      else  {close(newFD); pocFD = oldFD;}
   pocFN = oldFN;
   pocSZ = Offs;
   return aOK;
}

/******************************************************************************/
/*                             V e r O f f s e t                              */
/******************************************************************************/
  
int XrdOfsPoscq::VerOffset(const char *Lfn, int Offset)
{

// Verify the offset
//
   if (Offset < ReqOffs || (Offset-ReqOffs)%ReqSize)
      {char buff[128];
       sprintf(buff, "Invalid slot %d for", Offset);
       eDest->Emsg("VerOffset", buff, Lfn);
       return 0;
      }
   return 1;
}
