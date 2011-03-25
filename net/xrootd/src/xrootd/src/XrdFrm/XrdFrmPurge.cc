/******************************************************************************/
/*                                                                            */
/*                        X r d F r m P u r g e . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <utime.h>
#include <sys/param.h>
#include <sys/types.h>

#include "XrdNet/XrdNetCmsNotify.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmPurge.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
/******************************************************************************/
/*                  C l a s s   X r d F r m P u r g e D i r                   */
/******************************************************************************/
  
class XrdFrmPurgeDir : XrdOucNSWalk::CallBack
{
public:

void isEmpty(struct stat *dStat, const char *dPath, const char *lkFN);

void Reset(time_t dExp)
          {expDirTime = dExp; lowDirTime = 0; numRMD = numEMD = 0;}

time_t expDirTime;
time_t lowDirTime;
int    numRMD;
int    numEMD;

     XrdFrmPurgeDir() {}
    ~XrdFrmPurgeDir() {}
};
  
/******************************************************************************/
/*                               i s E m p t y                                */
/******************************************************************************/

void XrdFrmPurgeDir::isEmpty(struct stat *dStat, const char *dPath,
                             const char *lkFN)
{
   static const int ossOpts = XRDOSS_isPFN | XRDOSS_resonly;
   static const char *What = (Config.Test ? "Zorch  " : "Purged ");
   struct stat pStat;
   struct utimbuf times;
   char Parent[MAXPATHLEN+1], *Slash;
   int  n, rc;

// Check if this directory is still considered active
//
   numEMD++;
   if (dStat->st_mtime > expDirTime)
      {if (!lowDirTime || lowDirTime > dStat->st_mtime)
          lowDirTime = dStat->st_mtime;
       return;
      }

// We can expire the directory. However, we need to get the parent mtime
// because removing this directory should not change the parent's mtime.
//
   strcpy(Parent, dPath);
   n = strlen(Parent);
   if (Parent[n-1] == '/') Parent[--n] = '\0';
   if ((Slash = rindex(Parent, '/')))
      {*Slash = '\0';
       if (stat(Parent, &pStat)) Slash = 0;
      }

// Delete the directory
//
   if (Config.Test) rc = 0;
      else if (!(rc = Config.ossFS->Remdir(dPath, ossOpts)) && Slash)
              {times.actime  = pStat.st_atime;
               times.modtime = pStat.st_mtime;
               utime(Parent, &times);
              }

// Report if successful
//
   if (!rc)
      {numRMD++;
       if (Config.Verbose)
          {char sbuff[32];
           struct tm tNow;
           localtime_r(&(dStat->st_mtime), &tNow);
           sprintf(sbuff, "%02d%02d%02d %02d:%02d:%02d ",
                          tNow.tm_year-100, tNow.tm_mon+1, tNow.tm_mday,
                          tNow.tm_hour,     tNow.tm_min,   tNow.tm_sec);
           Say.Say(What, "empty dir ", sbuff, dPath);
          }
      }
}

/******************************************************************************/
/*                     C l a s s   X r d F r m P u r g e                      */
/******************************************************************************/
/******************************************************************************/
/*                        S t a t i c   M e m b e r s                         */
/******************************************************************************/
  
XrdFrmPurge      *XrdFrmPurge::First     = 0;
XrdFrmPurge      *XrdFrmPurge::Default   = 0;

XrdOucProg       *XrdFrmPurge::PolProg   = 0;
XrdOucStream     *XrdFrmPurge::PolStream = 0;

int               XrdFrmPurge::Left2Do   = 0;

time_t            XrdFrmPurge::lastReset = 0;
time_t            XrdFrmPurge::nextReset = 0;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdFrmPurge::XrdFrmPurge(const char *snp, XrdFrmPurge *spp) : FSTab(1)
{
   strncpy(SName, snp, sizeof(SName)-1); SName[sizeof(SName)-1] = '\0';
   Next = spp;
   freeSpace = 0;
   usedSpace =-1;
   pmaxSpace = 0;
   totlSpace = 0;
   contSpace = 0;
   minFSpace = 0;
   maxFSpace = 0;
   Enabled   = 0;
   Stop      = 0;
   SNlen     = strlen(SName);
   memset(DeferQ, 0, sizeof(DeferQ));
   Clear();
}
  
/******************************************************************************/
/* Private:                          A d d                                    */
/******************************************************************************/
  
void XrdFrmPurge::Add(XrdFrmFileset *sP)
{
   EPNAME("Add");
   XrdOucNSWalk::NSEnt *baseFile = sP->baseFile();
   XrdFrmPurge *psP = Default;
   const char *Why;
   time_t xTime;

// First, get the space name associated with the base file
//
   if ((baseFile->Link))
      {char snBuff[XrdOssSpace::minSNbsz];
       XrdOssPath::getCname(0, snBuff, baseFile->Link, baseFile->Lksz);
       if (!(psP = Find(snBuff))) psP = Default;
      }

// Ignore the file is the space is not enabled for purging
//
   if (!(psP->Enabled)) {delete sP; return;}
   psP->numFiles++;

// Check to see if the file is really eligible for purging
//
   if ((Why = psP->Eligible(sP, xTime)))
      {DEBUG(sP->basePath() <<"cannot be purged; " <<Why);
       delete sP;
       return;
      }

// Add the file to the purge table or the defer queue based on access time
//
   if (xTime >= psP->Hold) psP->FSTab.Add(sP);
      else psP->Defer(sP, xTime);
}
  
/******************************************************************************/
/* Private:                      A d v a n c e                                */
/******************************************************************************/

XrdFrmFileset *XrdFrmPurge::Advance()
{
   XrdFrmFileset *fP, *xP;
   int n;

// Find a defer queue entry that meets the hold threshold
//
   for (n = DeferQsz-1; n >= 0 && !DeferQ[n]; n--) {}
   if (n < 0) return 0;
   if (time(0) - DeferT[n] > Hold) return 0;
   fP = DeferQ[n]; DeferQ[n] = 0; DeferT[n] = 0;

// Try to re-add everything in this queue
//
   while((xP = fP))
        {fP = fP->Next;
         if (xP->Refresh(0,0)) Add(xP);
            else delete xP;
        }

// Return based on whether something now exists in the purge table
//
   return FSTab.Oldest();
}
  
/******************************************************************************/
/* Private:                        C l e a r                                  */
/******************************************************************************/
  
void XrdFrmPurge::Clear()
{
   XrdFrmFileset *fP;
   int n;

// Zero out the defer queue
//
   for (n = 0; n < DeferQsz; n++)
       while ((fP = DeferQ[n])) {DeferQ[n] = fP->Next; delete fP;}
   memset(DeferT, 0, sizeof(DeferT));

// Purge the eligible file table
//
   FSTab.Purge();

// Clear counters
//
   numFiles = 0; prgFiles = 0; purgBytes = 0;
}
  
/******************************************************************************/
/* Private:                        D e f e r                                  */
/******************************************************************************/

void XrdFrmPurge::Defer(XrdFrmFileset *sP, time_t xTime)
{
   time_t aTime = sP->baseFile()->Stat.st_atime;
   int n = xTime/DeferQsz;

// Slot the entry into the defer queue vector
//
   if (n >= DeferQsz) n = DeferQsz-1;
   if (!DeferQ[n] || aTime < DeferT[n]) DeferT[n] = aTime;
   sP->Next = DeferQ[n];
   DeferQ[n] = sP;
}

/******************************************************************************/
/*                               D i s p l a y                                */
/******************************************************************************/

void XrdFrmPurge::Display()
{
   XrdFrmConfig::VPInfo *vP = Config.pathList;
   XrdFrmPurge *spP = First;
   XrdOucTList *tP;
   const char *isExt;
   char buff[1024], minfsp[32], maxfsp[32];

// Type header
//
   Say.Say("=====> ", "Purge configuration:");

// Display what we will scan
//
   while(vP)
        {Say.Say("=====> ", "Scanning ", (vP->Val?"r/w: ":"r/o: "), vP->Name);
         tP = vP->Dir;
         while(tP) {Say.Say("=====> ", "Excluded ", tP->text); tP = tP->next;}
         vP = vP->Next;
        }

// Display directory hold value
//
   if (Config.dirHold < 0) strcpy(buff, "forever");
      else sprintf(buff, "%d", Config.dirHold);
   Say.Say("=====> ", "Directory hold: ", buff);

// Run through all of the policies, displaying each one
//
   spP = First;
   while(spP)
        {if (spP->Enabled)
            {XrdOucUtils::fmtBytes(spP->minFSpace, minfsp, sizeof(minfsp));
             XrdOucUtils::fmtBytes(spP->maxFSpace, maxfsp, sizeof(maxfsp));
         isExt = spP->Ext ? " polprog" : "";
             sprintf(buff, "policy %s min %s max %s free; hold: %d%s",
                     spP->SName, minfsp, maxfsp, spP->Hold, isExt);
            } else sprintf(buff, "policy %s nopurge", spP->SName);
         Say.Say("=====> ", buff);
         spP = spP->Next;
        }
}
  
/******************************************************************************/
/* Private:                     E l i g i b l e                               */
/******************************************************************************/
  
const char *XrdFrmPurge::Eligible(XrdFrmFileset *sP, time_t &xTime, int hTime)
{
   XrdOucNSWalk::NSEnt *baseFile = sP->baseFile();
   time_t aTime, mTime, nowTime = time(0);

// Get the acess time and modification time
//
   aTime = baseFile->Stat.st_atime;
   mTime = baseFile->Stat.st_mtime;

// File is not eligible if it's been accessed too recently
//
   xTime = static_cast<int>(nowTime - aTime);
   if (hTime && xTime <= hTime) return "is in hold";

// File is ineligible if it has a fail file
//
   if (sP->failFile()) return "has fail file";

// If there is a lock file and the file has not migrated, then ineligible
// Note that entries were pre-screened for copy file requirements.
//
   if (sP->cpyInfo.Attr.cpyTime
   &&  sP->cpyInfo.Attr.cpyTime < static_cast<long long>(mTime))
      return "not migrated";

// If there is no pin info, then it is eligible subject to external policy
//
   if (!(sP->pinInfo.Attr.Flags)) return 0;

// See if pin is permanent
//
   if (sP->pinInfo.Attr.Flags & XrdFrmXAttrPin::pinPerm)
       return "is perm pinned";

// See if the file is pinned until a certain time
//
   if (sP->pinInfo.Attr.Flags & XrdFrmXAttrPin::pinKeep
   &&  sP->pinInfo.Attr.pinTime > static_cast<long long>(nowTime))
      return "is time pinned";

// Check if the file can only be unpinned after going idle
//
   if (sP->pinInfo.Attr.Flags & XrdFrmXAttrPin::pinIdle
   &&  sP->pinInfo.Attr.pinTime > static_cast<long long>(xTime))
      return "is pin defered";
   return 0;
}

/******************************************************************************/
/* Private:                         F i n d                                   */
/******************************************************************************/

XrdFrmPurge *XrdFrmPurge::Find(const char *snp)
{
   XrdFrmPurge *spP = First;
  
// See if we already have this space defined
//
   while(spP && strcmp(snp, spP->SName)) spP = spP->Next;
   return spP;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/

int XrdFrmPurge::Init(XrdOucTList *sP, long long minV, int hVal)
{
   static char pVec[] = {char(XrdFrmConfig::PP_sname),
                         char(XrdFrmConfig::PP_pfn),
                         char(XrdFrmConfig::PP_fsize),
                         char(XrdFrmConfig::PP_atime),
                         char(XrdFrmConfig::PP_mtime)
                        };

   XrdFrmConfig::VPInfo *vP;
   XrdOssVSInfo vsInfo;
   XrdFrmPurge *xP, *ppP = 0, *spP = First;
   XrdOucTList  *tP;
   char xBuff[32];
   int setIt, rc, haveExt = 0;

// The first step is to remove any defined policies for which there is no space
//
   while(spP)
        {vP = Config.VPList;
         while(vP && strcmp(spP->SName, vP->Name)) vP = vP->Next;
         if (!vP && strcmp("public", spP->SName))
            {Say.Emsg("Init", "Purge policy", spP->SName,
                              "deleted; space not defined.");
             if (ppP) ppP->Next = spP->Next;
                else  First =     spP->Next;
             xP = spP; spP = spP->Next; delete xP;
            } else {ppP = spP; spP = spP->Next;}
        }

// For each space enable it and optionally over-ride policy
//
   spP = First;
   while(spP)
        {setIt = 1;
         if ((tP = sP))
            {while(tP && strcmp(tP->text, spP->SName)) tP = tP->next;
             if (!tP) setIt = 0;
            }
         if (setIt)
            {if (minV) spP->minFSpace = spP->maxFSpace = minV;
             if (hVal >= 0) {spP->Hold = hVal; spP->Hold2x = hVal*2;}
             if (spP->minFSpace && spP->Hold >= 0)
                {spP->Enabled = 1; haveExt |= spP->Ext;}
            }
         spP = spP->Next;
        }

// Go through each space policy getting the actual space and calculating
// the targets based on the policy (we need to do this only once)
//
   spP = First; ppP = 0;
   while(spP)
        {if ((rc = Config.ossFS->StatVS(&vsInfo, spP->SName, 1)))
            {Say.Emsg("Init", rc, "calculate space for", spP->SName);
             if (ppP) ppP->Next = spP->Next;
                else  First =     spP->Next;
             xP = spP; spP = spP->Next; delete xP; continue;
            }
         spP->totlSpace = vsInfo.Total;
         spP->spaceTLen = sprintf(xBuff, "%lld", vsInfo.Total);
         spP->spaceTotl =  strdup(xBuff);
         spP->pmaxSpace = vsInfo.Large;
         spP->spaceTLep = sprintf(xBuff, "%lld", vsInfo.Large);
         spP->spaceTotP =  strdup(xBuff);
         if (spP->minFSpace < 0)
            {spP->minFSpace = vsInfo.Total * XRDABS(spP->minFSpace) / 100LL;
             spP->maxFSpace = vsInfo.Total * XRDABS(spP->maxFSpace) / 100LL;
            } else if (vsInfo.Total < spP->minFSpace
                   ||  vsInfo.Total < spP->maxFSpace)
                      Say.Emsg("Init", "Warning: ", spP->SName, " min/max free "
                               "space policy exceeds total available!");
         ppP = spP; spP = spP->Next;
        }

// Make sure "public" still exists. While this should not happen, we check for
// this possibility anyway.
//
   if (!(Default = Find("public")))
      {Say.Emsg("Init", "Unable to start purge; no public policy found.");
       return 0;
      }

// If a policy program is present, then we need to verify it
//
   if (Config.pProg && haveExt)
      {PolProg = new XrdOucProg(&Say);
       if (PolProg->Setup(Config.pProg) || PolProg->Start()) return 0;
       PolStream = PolProg->getStream();
       if (!Config.pVecNum)
          {memcpy(Config.pVec, pVec, sizeof(pVec));
           Config.pVecNum = sizeof(pVec);
          }
      }

// All went well
//
   return 1;
}
  
/******************************************************************************/
/* Private:                   L o w O n S p a c e                             */
/******************************************************************************/

// This method *must* be called prior to Purge() and returns:
// =0 -> Purge not needed.
//!>0 -> Purge is  needed.
  
int XrdFrmPurge::LowOnSpace()
{
   XrdOssVSInfo VSInfo;
   XrdFrmPurge *psP = First;
   time_t eNow;

// Recalculate free space and set initial status
//
   Left2Do = 0;
   while(psP)
        {if (psP->Enabled)
            {if (Config.ossFS->StatVS(&VSInfo, psP->SName, 1)) psP->Stop = 1;
                else {psP->freeSpace = VSInfo.Free;
                      psP->contSpace = VSInfo.LFree;
                      psP->usedSpace = VSInfo.Usage;
                      if (psP->freeSpace >= psP->minFSpace) psP->Stop = 1;
                         else {Left2Do++; psP->Stop = 0;}
                     }
            } else psP->Stop = 1;
         psP = psP->Next;
        }

// If enough space then indicate no purge is needed
//
   if (!Left2Do) return 0;

// Reset all policies to prepare for purging
//
   psP = First;
   while(psP)
        {psP->Clear();
         psP = psP->Next;
        }

// We must check whether or not a full name space scan is required. This is
// based on the last time we did one and whether or not a space needs one now.
//
   eNow = time(0);
   if (eNow >= nextReset) {lastReset = eNow; nextReset = 0; Scan();}
   return 1;
}

/******************************************************************************/
/*                                P o l i c y                                 */
/******************************************************************************/

XrdFrmPurge *XrdFrmPurge::Policy(const char *sname, long long minV,
                                 long long maxV, int hVal, int xVal)
{
   XrdFrmPurge *psP;
  
// Find or create a new policy
//
   if (!(psP = Find(sname))) First = psP = new XrdFrmPurge(sname, First);

// Fill out the policy
//
   psP->minFSpace = minV;
   psP->maxFSpace = maxV;
   psP->Hold      = hVal;
   psP->Hold2x    = hVal*2;
   psP->Ext       = xVal;
   return psP;
}

/******************************************************************************/
/*                                 P u r g e                                  */
/******************************************************************************/
  
void XrdFrmPurge::Purge()
{
   XrdFrmPurge *psP = First;

// Check if are low on space, if not, ignore the call
//
   if (!LowOnSpace())
      {Say.Emsg("Purge", "Purge cycle skipped; all policies met.");
       return;
      }

// Report data at the start of the purge cycle
//
   Say.Emsg("Purge", "Purge cycle started.");
   Stats(0);

// Cycle through each space until we no longer can cycle
//
do{psP = First;
   while(psP && Left2Do)
        {if (!(psP->Stop) && (psP->Stop = psP->PurgeFile())) Left2Do--;
         psP = psP->Next;
        }
  } while(Left2Do);

// Report data at the end of the purge cycle
//
   Stats(1);
   Say.Emsg("Purge", "Purge cycle ended.");
}

/******************************************************************************/
/* Private:                    P u r g e F i l e                              */
/******************************************************************************/
  
int XrdFrmPurge::PurgeFile()
{
   EPNAME("PurgeFile");
   static const int unOpts = XRDOSS_isPFN|XRDOSS_isMIG;
   XrdFrmFileset *fP;
   const char *fn, *Why;
   time_t xTime;
   int rc, FilePurged = 0;

// If we have don't have a file, see if we can grab some from the defer queue
//
do{if (!(fP = FSTab.Oldest()) && !(fP = Advance()))
      {time_t nextScan = time(0)+Hold;
       if (!nextReset || nextScan < nextReset) nextReset = nextScan;
       return 1;
      }
   Why = "file in use";
   if (fP->Refresh() && !(Why = Eligible(fP, xTime, Hold))
   && (!Ext || !(Why = XPolOK(fP))))
      {fn = fP->basePath();
       if (Config.Test) rc = 0;
          else if (!(rc = Config.ossFS->Unlink(fn, unOpts))
               && Config.cmsPath) Config.cmsPath->Gone(fn);
       if (!rc) {prgFiles++; FilePurged = 1;
                 freeSpace += fP->baseFile()->Stat.st_size;
                 purgBytes += fP->baseFile()->Stat.st_size;
                 if (Config.Verbose) Track(fP);
                }
      } else {DEBUG("Purge " <<SName <<": keeping " <<fP->basePath() <<"; " <<Why);}
   delete fP;
  } while(!FilePurged && !Stop);

// All done, indicate whether we should stop now
//
   return freeSpace >= maxFSpace || Stop;
}
  
/******************************************************************************/
/* Private:                         S c a n                                   */
/******************************************************************************/
  
void XrdFrmPurge::Scan()
{
   static const int Opts = XrdFrmFiles::Recursive | XrdFrmFiles::CompressD
                         | XrdFrmFiles::NoAutoDel;
   static time_t lastHP = time(0), nextDP = 0, nowT = time(0);
   static XrdFrmPurgeDir purgeDir;
   static XrdOucNSWalk::CallBack *cbP;

   XrdFrmConfig::VPInfo *vP = Config.pathList;
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   const char *Extra;
   char buff[128];
   int needLF, ec = 0, Bad = 0, aFiles = 0, bFiles = 0;

// Purge that bad file table evey 24 hours to keep complaints down
//
   if (nowT - lastHP >= 86400) {XrdFrmFileset::Purge(); lastHP = nowT;}

// Determine if we need to do an empty directory trim
//
   if (Config.dirHold < 0 || nowT < nextDP) {cbP = 0; Extra = 0;}
      else {nextDP = nowT + Config.dirHold;
            purgeDir.Reset(nowT - Config.dirHold);
            cbP = (XrdOucNSWalk::CallBack *)&purgeDir;
            Extra = "and empty directory";
           }

// Indicate scan started
//
   VMSG("Scan", "Name space", Extra, "scan started. . .");

// Process each directory
//
   do {fP = new XrdFrmFiles(vP->Name, Opts, vP->Dir, cbP);
       needLF = vP->Val;
       while((sP = fP->Get(ec,1)))
            {aFiles++;
             if (sP->Screen(needLF)) Add(sP);
                else {delete sP; bFiles++;}
            }
       if (ec) Bad = 1;
       delete fP;
      } while((vP = vP->Next));

// If we did a directory purge, schedule the next one and say what we did
//
   if (cbP)
      {if ((purgeDir.numEMD - purgeDir.numRMD) > 0
       &&   purgeDir.lowDirTime + Config.dirHold < nextDP)
          nextDP = purgeDir.lowDirTime + Config.dirHold;
       sprintf(buff, "%d of %d empty dir%s removed", purgeDir.numRMD,
                     purgeDir.numEMD, (purgeDir.numEMD != 1 ? "s":""));
       VMSG("Scan", "Empty directory space scan ended;", buff);
      }

// Indicate scan ended
//
   sprintf(buff, "%d file%s with %d error%s", aFiles, (aFiles != 1 ? "s":""),
                                              bFiles, (bFiles != 1 ? "s":""));
   VMSG("Scan", "Name space scan ended;", buff);

// Issue warning if we encountered errors
//
   if (Bad) Say.Emsg("Scan", "Errors encountered while scanning for "
                             "purgeable files.");
}
  
/******************************************************************************/
/* Private:                        S t a t s                                  */
/******************************************************************************/
  
void XrdFrmPurge::Stats(int Final)
{
   XrdFrmPurge *xsP, *psP = First;
   long long pVal, xBytes, zBytes;
   const char *xWhat, *nWhat, *zWhat;
   char fBuff[64], uBuff[80], sBuff[512], xBuff[64], zBuff[64];
   int nFiles;

// Report data for each enabled space
//
   while((xsP = psP))
        {psP = psP->Next;
         if (!(xsP->Enabled)) continue;
         if (xsP->usedSpace >= 0)
            {if (Final) xsP->usedSpace -= xsP->purgBytes;
             XrdOucUtils::fmtBytes(xsP->usedSpace, fBuff, sizeof(fBuff));
             pVal = xsP->usedSpace*100/xsP->totlSpace;
             sprintf(uBuff, "used %s (%lld%%) ", fBuff, pVal);
            } else *uBuff = '\0';
         XrdOucUtils::fmtBytes(xsP->freeSpace, fBuff, sizeof(fBuff));
         pVal = xsP->freeSpace*100/xsP->totlSpace;
         if (Final)
            {xBytes = xsP->purgBytes; xWhat = "freed";
             if ((zBytes = xsP->maxFSpace - xsP->freeSpace) > 0)
                {XrdOucUtils::fmtBytes(zBytes, zBuff, sizeof(zBuff));
                 zWhat = " deficit";
                } else {*zBuff = '\0'; zWhat = "need met";}
             nFiles = xsP->prgFiles;  nWhat = "prgd";
            } else {
             xBytes = (xsP->freeSpace < xsP->minFSpace
                    ?  xsP->maxFSpace - xsP->freeSpace : 0);
             nFiles = xsP->FSTab.Count(); 
             xWhat = "needed"; nWhat = "idle"; *zBuff = '\0'; zWhat = "";
           }
         XrdOucUtils::fmtBytes(xBytes, xBuff, sizeof(xBuff));
         sprintf(sBuff, " %sfree %s (%lld%%) %d files %d %s; %s %s %s%s",
                 uBuff,fBuff,pVal,xsP->numFiles,nFiles,nWhat,
                 xBuff,xWhat,zBuff,zWhat);
         Say.Say("++++++ ", xsP->SName, sBuff);
        }
}

/******************************************************************************/
/* Private:                        T r a c k                                  */
/******************************************************************************/
  
void XrdFrmPurge::Track(XrdFrmFileset *sP)
{
   XrdOucNSWalk::NSEnt *fP = sP->baseFile();
   const char *What = (Config.Test ? "Zorch  " : "Purged ");
   char sbuff[128], fszbf[16];
   struct tm tNow;

// Format the size
//
   XrdOucUtils::fmtBytes(static_cast<long long>(fP->Stat.st_size),
                         fszbf, sizeof(fszbf));

// Format the information and display it
//
   localtime_r(&(fP->Stat.st_atime), &tNow);
   sprintf(sbuff, " %8s %02d%02d%02d %02d:%02d:%02d ", fszbf,
                  tNow.tm_year-100, tNow.tm_mon+1, tNow.tm_mday,
                  tNow.tm_hour,     tNow.tm_min,   tNow.tm_sec);

   Say.Say(What, SName, sbuff, sP->basePath());
}

/******************************************************************************/
/* Private:                       X P o l O K                                 */
/******************************************************************************/
  
const char *XrdFrmPurge::XPolOK(XrdFrmFileset *fsP)
{
   static char neg1[] = {'-','1','\0'};
   XrdOucNSWalk::NSEnt *fP = fsP->baseFile();
   char *Data[sizeof(Config.pVec)*2+2];
   int   Dlen[sizeof(Config.pVec)*2+2];
   char  atBuff[32], ctBuff[32], mtBuff[32], fsBuff[32], spBuff[32], usBuff[32];
   char *Resp;
   int i, k = 0;

// Construct the data to be sent (not mt here)
//
   for (i = 0; i < Config.pVecNum; i++)
       {switch(Config.pVec[i])
              {case XrdFrmConfig::PP_atime:
                    Data[k] = atBuff;
                    Dlen[k] = sprintf(atBuff, "%lld",
                              static_cast<long long>(fP->Stat.st_atime));
                    break;
               case XrdFrmConfig::PP_ctime:
                    Data[k] = ctBuff;
                    Dlen[k] = sprintf(ctBuff, "%lld",
                              static_cast<long long>(fP->Stat.st_ctime));
                    break;
               case XrdFrmConfig::PP_fname:
                    Data[k] = fP->File;  Dlen[k] = strlen(fP->File);
                    break;
               case XrdFrmConfig::PP_fsize:
                    Data[k] = fsBuff;
                    Dlen[k] = sprintf(fsBuff, "%lld",
                              static_cast<long long>(fP->Stat.st_size));
                    break;
               case XrdFrmConfig::PP_fspace:
                    Data[k] = spBuff;
                    Dlen[k] = sprintf(spBuff, "%lld", freeSpace);
                    break;
               case XrdFrmConfig::PP_mtime:
                    Data[k] = mtBuff;
                    Dlen[k] = sprintf(mtBuff, "%lld",
                              static_cast<long long>(fP->Stat.st_mtime));
                    break;
               case XrdFrmConfig::PP_pfn:
                    Data[k] = (char *)fsP->basePath();
                    Dlen[k] = strlen(Data[k]);
                    break;
               case XrdFrmConfig::PP_sname:
                    Data[k] = SName;     Dlen[k] = SNlen;
                    break;
               case XrdFrmConfig::PP_tspace:
                    Data[k] = spaceTotl; Dlen[k] = spaceTLen;
                    break;
               case XrdFrmConfig::PP_usage:
                    if (usedSpace < 0) {Data[k] = neg1; Dlen[k]=2;}
                       else {Dlen[k] = sprintf(usBuff, "%lld",
                                               usedSpace - purgBytes);
                             Data[k] = usBuff;
                            }
                    break;
               default: break;
              }
        Data[++k] = (char *)" "; Dlen[k] = 1; k++;
       }

// Now finish up the vector
//
   Data[k-1] = (char *)"\n"; Data[k] = 0; Dlen[k] = 0;

// Feed the program this information get the response
//
   if (PolProg->Feed((const char **)Data, Dlen) || !(Resp=PolStream->GetLine()))
      {Stop = 1; return "external policy failed";}

// Decode the response (single line with a charcater y|n|a)
//
   if (*Resp == 'y') return 0;
   if (*Resp == 'n') return "external policy reject";
   Stop = 1;
   return "external policy stop";
}
