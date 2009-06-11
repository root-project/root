/******************************************************************************/
/*                                                                            */
/*                      X r d F r m P s t g R e q . c c                       */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmPstgReq.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdFrmPstgReq     *XrdFrm::rQueue[XrdFrmPstgReq::maxPrty];

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdFrmPstgReq::XrdFrmPstgReq(const char *fn)
{
   char buff[1200];

   memset((void *)&HdrData, 0, sizeof(HdrData));
   reqFN = strdup(fn);
   strcpy(buff, fn); strcat(buff, ".lock");
   lokFN = strdup(buff);
   lokFD = reqFD = -1;
}
  
/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
void XrdFrmPstgReq::Add(XrdFrmPstgReq::Request *rP)
{
   XrdFrmPstgReq::Request tmpReq;
   char *qP;
   int fP;

// Lock the file
//
   if (!FileLock()) {FailAdd(rP->LFN, 0); return;}

// Process the Opaque information
//
   if (!(qP = index(rP->LFN, '?'))) rP->Opaque = 0;
      else {*qP = '\0';
            if (*(qP+1)) rP->Opaque = qP-(rP->LFN)+1;
               else rP->Opaque = 0;
           }

// Obtain a free slot
//
   if ((fP = HdrData.Free))
      {if (!reqRead((void *)&tmpReq, fP)) {FailAdd(rP->LFN, 1); return;}
       HdrData.Free = tmpReq.Next;
      } else {
       struct stat buf;
       if (fstat(reqFD, &buf))
          {Say.Emsg("Add",errno,"stat",reqFN); FailAdd(rP->LFN, 1); return;}
       fP = buf.st_size;
      }

// Chain in the request
//
   if (HdrData.First && HdrData.Last)
      {if (!reqRead((void *)&tmpReq, HdrData.Last))
          {FailAdd(rP->LFN, 1); return;}
       tmpReq.Next = fP;
       if (!reqWrite((void *)&tmpReq, HdrData.Last, 0))
          {FailAdd(rP->LFN, 1); return;}
      } else HdrData.First = fP;
    HdrData.Last = fP;

// Write out the file
//
   rP->This = fP; rP->Next = 0;
   if (!reqWrite(rP, fP)) FailAdd(rP->LFN, 0);
   FileLock(lkNone);
}
  
/******************************************************************************/
/*                                   C a n                                    */
/******************************************************************************/

void XrdFrmPstgReq::Can(XrdFrmPstgReq::Request *rP)
{
   XrdFrmPstgReq::Request tmpReq;
   int Offs, numCan = 0, numBad = 0;
   struct stat buf;
   char txt[128];

// Lock the file and get its size
//
   if (!FileLock() || fstat(reqFD, &buf)) {FailCan(rP->ID, 0); return;}

// Run through all of the file entries removing matching requests
//
   for (Offs = ReqSize; Offs < buf.st_size; Offs += ReqSize)
       {if (!reqRead((void *)&tmpReq, Offs)) return FailCan(rP->ID);
        if (!strcmp(tmpReq.ID, rP->ID))
           {tmpReq.LFN[0] = '\0';
            if (!reqWrite((void *)&tmpReq, Offs, 0)) numBad++;
               else numCan++;
           }
       }

// Make sure this is written to disk
//
   if (numCan) fsync(reqFD);

// Document the action
//
   if (numCan || numBad)
      {sprintf(txt, "has %d entries; %d removed (%d failures).",
                    numCan+numBad, numCan, numBad);
       Say.Emsg("Can", rP->ID, txt);
      }
   FileLock(lkNone);
}
  
/******************************************************************************/
/*                                   D e l                                    */
/******************************************************************************/

void XrdFrmPstgReq::Del(XrdFrmPstgReq::Request *rP)
{
   XrdFrmPstgReq::Request tmpReq;

// Lock the file
//
   if (!FileLock()) {FailDel(rP->LFN, 0); return;}

// Put entry on the free chain
//
   memset(&tmpReq, 0, sizeof(tmpReq));
   tmpReq.Next  = HdrData.Free;
   HdrData.Free = rP->This;
   if (!reqWrite((void *)&tmpReq, rP->This)) FailDel(rP->LFN, 0);
   FileLock(lkNone);
}

/******************************************************************************/
/*                                   G e t                                    */
/******************************************************************************/
  
int XrdFrmPstgReq::Get(XrdFrmPstgReq::Request *rP)
{
   int fP, rc;

// Lock the file
//
   if (!FileLock()) return 0;

// Get the next request
//
   while((fP = HdrData.First))
        {if (!reqRead((void *)rP, fP)) {FileLock(lkNone); return 0;}
         HdrData.First= rP->Next;
         if (*(rP->LFN)) break;
         rP->Next     = HdrData.Free;
         HdrData.Free = fP;
         if (!reqWrite(rP, fP)) {fP = 0; break;}
      }
   reqWrite(0,0,1);
   if (fP) rc = (HdrData.First ? 1 : -1);
      else rc = 0;
   FileLock(lkNone);
   return rc;
}
  
/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/

int XrdFrmPstgReq::Init()
{
   EPNAME("Init");
   static const int Mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   XrdFrmPstgReq::Request tmpReq;
   struct stat buf;
   recEnt *First = 0, *rP, *pP, *tP;
   int    Offs, rc, numreq = 0;

// Open the lock file first in r/w mode
//
   if ((lokFD = open(lokFN, O_RDWR|O_CREAT, Mode)) < 0)
      {Say.Emsg("Init",errno,"open",lokFN); return 0;}

// Obtain a lock
//
   if (!FileLock(lkInit)) return 0;

// Open the file first in r/w mode
//
   if ((reqFD = open(reqFN, O_RDWR|O_CREAT, Mode)) < 0)
      {FileLock(lkNone);
       Say.Emsg("Init",errno,"open",reqFN); 
       return 0;
      }

// Check for a new file here
//
   if (fstat(reqFD, &buf)) return FailIni("stat");
   if (buf.st_size < ReqSize)
      {memset(&tmpReq, 0, sizeof(tmpReq));
       HdrData.Free = ReqSize;
       if (!reqWrite((void *)&tmpReq, ReqSize)) return FailIni("init file");
       FileLock(lkNone);
       return 1;
      }

// We are done if this is a agent
//
   if (Config.isAgent)
      {FileLock(lkNone);
       return 1;
      }

// Read the full file
//
   for (Offs = ReqSize; Offs < buf.st_size; Offs += ReqSize)
       {if (!reqRead((void *)&tmpReq, Offs)) return FailIni("read file");
        if (*tmpReq.LFN == '\0' || !tmpReq.addTOD
        ||  tmpReq.Opaque >= int(sizeof(tmpReq.LFN))) continue;
        pP = 0; rP = First; tP = new recEnt(tmpReq); numreq++;
        while(rP && rP->reqData.addTOD < tmpReq.addTOD) {pP=rP; rP=rP->Next;}
        if (pP) pP->Next = tP;
           else First    = tP;
        tP->Next = rP;
       }

// Now write out the file
//
   DEBUG(numreq <<" request(s) recovered from " <<reqFN);
   rc = ReWrite(First);

// Delete all the entries in memory
//
   while((tP = First)) {First = tP->Next; delete tP;}

// All done
//
   FileLock(lkNone);
   return rc;
}
  
/******************************************************************************/
/*                                  L i s t                                   */
/******************************************************************************/
  
char  *XrdFrmPstgReq::List(char *Buff, int bsz, int &Offs,
                           Item *ITList, int ITNum)
{
   XrdFrmPstgReq::Request tmpReq;
   int rc;

// Set Offs argument
//
   if (Offs < ReqSize) Offs = ReqSize;

// Lock the file
//
   if (!FileLock(lkShare)) return 0;

// Return next valid filename
//
   do{do {rc = pread(reqFD, (void *)&tmpReq, ReqSize, Offs);}
          while(rc < 0 && errno == EINTR);
      if (rc == ReqSize)
         {Offs += ReqSize;
          if (*tmpReq.LFN == '\0' || !tmpReq.addTOD
          ||  tmpReq.Opaque >= int(sizeof(tmpReq.LFN))) continue;
          FileLock(lkNone);
          if (!ITNum || !ITList) strlcpy(Buff, tmpReq.LFN, bsz);
             else ListL(tmpReq, Buff, bsz, ITList, ITNum);
          return Buff;
         }
     } while(rc == ReqSize);

// Diagnose ending condition
//
   if (rc < 0) Say.Emsg("List",errno,"read",reqFN);

// Return end of list
//
   FileLock(lkNone);
   return 0;
}

/******************************************************************************/
/*                                 L i s t L                                  */
/******************************************************************************/
  
void XrdFrmPstgReq::ListL(XrdFrmPstgReq::Request tmpReq, char *Buff, int bsz,
                          Item *ITList, int ITNum)
{
   char What, tbuf[32];
   long long tval;
   int i, n, bln = bsz-2;

   for (i = 0; i < ITNum && bln; i++)
       {switch(ITList[i])
              {case getLFN:     n = strlen(tmpReq.LFN);
                                strlcpy(Buff, tmpReq.LFN, bln);
                                break;
               case getLFNCGI:  n = strlen(tmpReq.LFN); tmpReq.LFN[n] = '?';
                                if (!tmpReq.Opaque) tmpReq.LFN[n+1] = '\0';
                                strlcpy(Buff, tmpReq.LFN, bln);
                                n = strlen(tmpReq.LFN);
                                tmpReq.LFN[n] = '\0';
                                break;
               case getMODE:    n = 0;
                                What = (tmpReq.Options & stgRW ? 'w' : 'r');
                                if (bln) {Buff[n] = What; n++;}
                                if (tmpReq.Options & msgFail)
                                if (bln-n > 0) {Buff[n] = 'f'; n++;}
                                if (tmpReq.Options & msgSucc)
                                if (bln-n > 0) {Buff[n] = 'n'; n++;}
                                break;
               case getNOTE:    n = strlen(tmpReq.Notify);
                                strlcpy(Buff, tmpReq.Notify, bln);
                                break;
               case getPRTY:    if (tmpReq.Prty == 2) What = '2';
                                   else if (tmpReq.Prty == 1) What = '1';
                                           else What = '0';
                                n = 1;
                                if (bln) *Buff = What;
                                break;
               case getQWT:
               case getTOD:     tval = tmpReq.addTOD;
                                if (ITList[i] == getQWT) tval = time(0)-tval;
                                if ((n = sprintf(tbuf, "%lld", tval)) >= 0)
                                   strlcpy(Buff, tbuf, bln);
                                break;
               case getRID:     n = strlen(tmpReq.ID);
                                strlcpy(Buff, tmpReq.ID, bln);
                                break;
               case getUSER:    n = strlen(tmpReq.User);
                                strlcpy(Buff, tmpReq.User, bln);
                                break;
               default:         n = 0; break;
              }
        if (bln > 0) {bln -= n; Buff += n;}
        if (bln > 0) {*Buff++ = ' '; bln--;}
       }
   *Buff = '\0';
}

/******************************************************************************/
/*                               F a i l A d d                                */
/******************************************************************************/
  
void XrdFrmPstgReq::FailAdd(char *lfn, int unlk)
{
   Say.Emsg("Add", lfn, "not added to prestage queue.");
   if (unlk) FileLock(lkNone);
}
  
/******************************************************************************/
/*                               F a i l C a n                                */
/******************************************************************************/
  
void XrdFrmPstgReq::FailCan(char *rid, int unlk)
{
   Say.Emsg("Can", rid, "request not removed from prestage queue.");
   if (unlk) FileLock(lkNone);
}
  
/******************************************************************************/
/*                               F a i l D e l                                */
/******************************************************************************/
  
void XrdFrmPstgReq::FailDel(char *lfn, int unlk)
{
   Say.Emsg("Del", lfn, "not removed from prestage queue.");
   if (unlk) FileLock(lkNone);
}

/******************************************************************************/
/*                               F a i l I n i                                */
/******************************************************************************/

int XrdFrmPstgReq::FailIni(const char *txt)
{
   Say.Emsg("Init", errno, txt, reqFN);
   FileLock(lkNone);
   return 0;
}
  
/******************************************************************************/
/*                              F i l e L o c k                               */
/******************************************************************************/
  
int XrdFrmPstgReq::FileLock(LockType lktype)
{
   FLOCK_t lock_args;
   const char *What;
   int rc;

// Establish locking options
//
   bzero(&lock_args, sizeof(lock_args));
   if (lktype == lkNone)
      {lock_args.l_type = F_UNLCK; What = "unlock";
       if (Config.isAgent && reqFD >= 0) {close(reqFD); reqFD = -1;}
      }
      else {lock_args.l_type = (lktype == lkShare ? F_RDLCK : F_WRLCK);
            What = "lock";
           }

// Perform action.
//
   do {rc = fcntl(lokFD,F_SETLKW,&lock_args);}
       while(rc < 0 && errno == EINTR);
   if (rc < 0) {Say.Emsg("FileLock", errno, What , lokFN); return 0;}

// Refresh the header
//
   if (lktype == lkExcl || lktype == lkShare)
      {if (reqFD < 0 && (reqFD = open(reqFN, O_RDWR)) < 0)
          {Say.Emsg("FileLock",errno,"open",reqFN);
           FileLock(lkNone);
           return 0;
          }
       do {rc = pread(reqFD, (void *)&HdrData, sizeof(HdrData), 0);}
           while(rc < 0 && errno == EINTR);
       if (rc < 0) {Say.Emsg("reqRead",errno,"refresh hdr from", reqFN);
                    FileLock(lkNone); return 0;
                   }
      }

// All done
//
   return 1;
}

/******************************************************************************/
/*                               r e q R e a d                                */
/******************************************************************************/
  
int XrdFrmPstgReq::reqRead(void *Buff, int Offs)
{
   int rc;

   do {rc = pread(reqFD, Buff, ReqSize, Offs);} while(rc < 0 && errno == EINTR);
   if (rc < 0) {Say.Emsg("reqRead",errno,"read",reqFN); return 0;}
   return 1;
}

/******************************************************************************/
/*                              r e q W r i t e                               */
/******************************************************************************/
  
int XrdFrmPstgReq::reqWrite(void *Buff, int Offs, int updthdr)
{
   int rc = 0;

   if (Buff && Offs)       do {rc = pwrite(reqFD, Buff, ReqSize, Offs);}
                              while(rc < 0 && errno == EINTR);
   if (rc >= 0 && updthdr){do {rc = pwrite(reqFD,&HdrData, sizeof(HdrData), 0);}
                              while(rc < 0 && errno == EINTR);
                           if (rc >= 0) rc = fsync(reqFD);
                          }
   if (rc < 0) {Say.Emsg("reqWrite",errno,"write", reqFN); return 0;}
   return 1;
}

/******************************************************************************/
/*                               R e W r i t e                                */
/******************************************************************************/
  
int XrdFrmPstgReq::ReWrite(XrdFrmPstgReq::recEnt *rP)
{
   static const int Mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   char newFN[MAXPATHLEN], *oldFN;
   int  newFD, oldFD, Offs = ReqSize, aOK = 1;

// Construct new file and open it
//
   strcpy(newFN, reqFN); strcat(newFN, ".new");
   if ((newFD = open(newFN, O_RDWR|O_CREAT|O_TRUNC, Mode)) < 0)
      {Say.Emsg("ReWrite",errno,"open",newFN); FileLock(lkNone); return 0;}

// Setup to write/swap the file
//
   oldFD = reqFD; reqFD = newFD;
   oldFN = reqFN; reqFN = newFN;

// Rewrite all records if we have any
//
   if (rP)
      {HdrData.First = Offs;
       while(rP)
            {rP->reqData.This = Offs;
             rP->reqData.Next = (rP->Next ? Offs+ReqSize : 0);
             if (!reqWrite((void *)&rP->reqData, Offs, 0)) {aOK = 0; break;}
             Offs += ReqSize;
             rP = rP->Next;
            }
       HdrData.Last = Offs - ReqSize;
      } else {
       HdrData.First = HdrData.Last = 0;
       if (ftruncate(newFD, ReqSize) < 0)
          {Say.Emsg("ReWrite",errno,"trunc",newFN); aOK = 0;}
      }

// Update the header
//
   HdrData.Free = 0;
   if (aOK && !(aOK = reqWrite(0, 0)))
      Say.Emsg("ReWrite",errno,"write header",newFN);

// If all went well, rename the file
//
   if (aOK && rename(newFN, oldFN) < 0)
      {Say.Emsg("ReWrite",errno,"rename",newFN); aOK = 0;}

// Perform post processing
//
   if (aOK)  close(oldFD);
      else  {close(newFD); reqFD = oldFD;}
   reqFN = oldFN;
   return aOK;
}

/******************************************************************************/
/*                                U n i q u e                                 */
/******************************************************************************/
  
  
int XrdFrmPstgReq::Unique(const char *lkfn)
{
   static const int Mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   FLOCK_t lock_args;
   int myFD, rc;

// Open the lock file first in r/w mode
//
   if ((myFD = open(lkfn, O_RDWR|O_CREAT, Mode)) < 0)
      {Say.Emsg("Unique",errno,"open",lkfn); return 0;}

// Establish locking options
//
   bzero(&lock_args, sizeof(lock_args));
   lock_args.l_type =  F_WRLCK;

// Perform action.
//
   do {rc = fcntl(myFD,F_SETLK,&lock_args);}
       while(rc < 0 && errno == EINTR);
   if (rc < 0) 
      {Say.Emsg("Unique", errno, "obtain the run lock on", lkfn);
       Say.Emsg("Unique", "Another", Config.myProg, "may already be running!");
       close(myFD);
       return 0;
      }

// All done
//
   return 1;
}
