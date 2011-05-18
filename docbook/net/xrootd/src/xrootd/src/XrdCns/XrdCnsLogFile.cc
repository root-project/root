/******************************************************************************/
/*                                                                            */
/*                      X r d C n s L o g F i l e . c c                       */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdCnsLogFileCVSID = "$Id$";

#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "XrdCns/XrdCnsLogFile.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
int XrdCnsLogFile::logRMax = 1024;
int XrdCnsLogFile::logBMax = 1024 * sizeof(XrdCnsLogRec);

namespace XrdCns
{
extern XrdSysError MLog;
}

using namespace XrdCns;
  
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdCnsLogFile::~XrdCnsLogFile()
{
// Wait until it is safe to delete ourselves
//
   synSem.Wait();

// Do clean-up
//
   if (logFD >= 0) close(logFD);
   if (logFN)      free(logFN);
   if (logBuff)    free(logBuff);
}

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
int XrdCnsLogFile::Add(XrdCnsLogRec *lrP, int doSync)
{
   XrdCnsLogFile *lfP = subNext;
   char *bP;
   int   rc, bL;

// Compute full record length
//
   bL = lrP->setLen() + XrdCnsLogRec::MinSize;
   bP = lrP->Record();

// Write out record
//
   do {do {rc = write(logFD, bP, bL);} while (rc < 0 && errno == EINTR);
       if (rc < 0) {MLog.Emsg("Add", errno, "add log rec to", logFN);
                    return 0;
                   }
       bP += rc; bL -= rc;
      } while(bL > 0);

// Make sure data is on disk
//
   if (doSync && fdatasync(logFD)) MLog.Emsg("Add", errno, "fsync log", logFN);

// Notify all subscribers
//
   while(lfP) {lfP->logSem.Post(); lfP = lfP->subNext;}

// All done
//
   return 1;
}

/******************************************************************************/
/*                                C o m m i t                                 */
/******************************************************************************/
  
int XrdCnsLogFile::Commit()
{
   static char dVal = 1;
   int dOffs, rc;

// Commit the previous record if we have returned one
//
   if (logOffset)
      {Rec.setDone(logRdr);
       dOffs = recOffset + XrdCnsLogRec::OffDone + logRdr;
       do {rc=pwrite(logFD, &dVal, 1, dOffs);} while(rc < 0 && errno == EINTR);
       if (rc > 0) fdatasync(logFD);
          else {MLog.Emsg("Commit", errno, "commit log rec in", logFN);
                return 0;
               }
      }
   return 1;
}

/******************************************************************************/
/*                                   E o l                                    */
/******************************************************************************/
  
int XrdCnsLogFile::Eol()
{
   XrdCnsLogFile *lfX, *lfP = subNext;
   XrdCnsLogRec   lRec(XrdCnsLogRec::lrEOL);
   int rc, bL = XrdCnsLogRec::MinSize + lRec.DLen();
   char *bP = (char *)&lRec;

// Write out record end of log record
//
   do {do {rc = write(logFD, bP, bL);} while (rc < 0 && errno == EINTR);
       if (rc < 0) {MLog.Emsg("Eol", errno, "eol log file", logFN);
                    break;
                   }
       bP += rc; bL -= rc;
      } while(bL > 0);

// Allow subscribers to end
//
   while(lfP)
        {lfP->logSem.Post();
         lfP->logWait = 0;
         lfX = lfP;
         lfP = lfP->subNext;
         lfX->synSem.Post();
        }

// All done
//
   return 1;
}

/******************************************************************************/
/*                                g e t R e c                                 */
/******************************************************************************/
  
XrdCnsLogRec *XrdCnsLogFile::getRec()
{
   char *bP, *nP;
   int   bL;

// Wait for a record if we must wait, read it, return if it's not committed
//
   do {if (logWait) logSem.Wait();
       bP = Rec.Record();
       bL = XrdCnsLogRec::MinSize; recOffset = logOffset;

       if (!Read(bP, bL) || !((bL = Rec.DLen()))) return 0;
       if (bL < XrdCnsLogRec::FixDLen)
          {MLog.Emsg("getRec", "Invalid record length detected in", logFN);
           return 0;
          }

       bP = bP + XrdCnsLogRec::MinSize;
       if (!Read(bP, bL)) return 0;

       memcpy(logNext, bP, bL);
       nP = logNext + XrdCnsLogRec::FixDLen + Rec.L1sz();
       if (!Rec.L2sz()) *nP = '\n';
          else {*nP = ' ';
                *(nP + Rec.L2sz() + 1) = '\n';
               }
       logNext += bL;
      } while(Rec.Done(logRdr));

// Return the record or nil pointer if there is no associated file
//
   return (Rec.L1sz() ? &Rec : 0);
}

/******************************************************************************/
/*                                  O p e n                                   */
/******************************************************************************/
  
int XrdCnsLogFile::Open(int allocbuff, off_t thePos)
{
   static const int AMode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

// Open the file
//
   if ((logFD = open(logFN, O_CREAT|O_RDWR, AMode)) < 0)
      {MLog.Emsg("Open", errno, "open", logFN); return 0;}

// Set starting position if need be
//
   if (thePos && (lseek(logFD, thePos, SEEK_SET) == (off_t)-1))
      {MLog.Emsg("Open", errno, "seek into", logFN); close(logFD), logFD = -1;
       return 0;
      }

// Allocate a memory buffer if so wanted
//
   if (allocbuff)
      {struct stat Stat;
       if (fstat(logFD, &Stat)) MLog.Emsg("Open", errno, "stat", logFN);
          else logNext=logBuff=(char *)malloc(logWait ? logBMax : Stat.st_size);
      }

// All done
//
   return 1;
}

/******************************************************************************/
/* Private:                         R e a d                                   */
/******************************************************************************/
  
int XrdCnsLogFile::Read(char *bP, int bL)
{
   int rc;

// Read the data
//
   do{do {rc = pread(logFD,bP,bL,logOffset);} while(rc < 0 && errno == EINTR);
      if (rc < 0) {MLog.Emsg("getRec", errno, "read", logFN); return 0;}
      bP += rc; bL -= rc; logOffset += rc;
     } while(bL > 0);

// All done
//
   return 1;
}

/******************************************************************************/
/*                             S u b s c r i b e                              */
/******************************************************************************/
  
XrdCnsLogFile *XrdCnsLogFile::Subscribe(const char *Path, int cNum)
{
   XrdCnsLogFile *lfP;
   int rc;

// Create hard link to our log file
//
   do {rc = link(logFN, Path);} while(rc && errno == EINTR);
   if (rc) {MLog.Emsg("Subscribe", errno, "create hard link", Path);
            return 0;
           }

// Indicate wait is needed and that delete must wait as well
//
   lfP = new XrdCnsLogFile(Path, cNum);
   lfP->logWait = 1;
   lfP->synSem.Wait();

// Chain in the subscriber
//
   lfP->subNext = subNext;
   subNext = lfP;
   return lfP;
}

/******************************************************************************/
/*                                U n l i n k                                 */
/******************************************************************************/
  
int XrdCnsLogFile::Unlink()
{
   int rc;

   do {rc = unlink(logFN);} while(rc < 0 && errno == EINTR);
   if (rc < 0) MLog.Emsg("Unlink", errno, "remove log", logFN);
   return rc >= 0;
}
