/******************************************************************************/
/*                                                                            */
/*                    X r d C n s I n v e n t o r y . c c                     */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdCnsInventoryCVSID = "$Id$";

#include "XrdCns/XrdCnsConfig.hh"
#include "XrdCns/XrdCnsInventory.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdSys/XrdSysError.hh"

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/
  
namespace XrdCns
{
extern XrdSysError  MLog;

extern XrdCnsConfig Config;
}

using namespace XrdCns;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdCnsInventory::XrdCnsInventory()
               : dRec(XrdCnsLogRec::lrInvD),
                 fRec(XrdCnsLogRec::lrInvF),
                 mRec(XrdCnsLogRec::lrMount),
                 sRec(XrdCnsLogRec::lrSpace),
                 Mount(Config.LCLRoot ? Config.LCLRoot : "/",0),
                 Space("public",0), lfP(0)
{
// Do some initialization
//
   dRec.setMode(0755);
   dRec.setSize(-1);
   mDflt = Mount.Default();
   sDflt = Space.Default();
}
  
/******************************************************************************/
/*                               C o n d u c t                                */
/******************************************************************************/
  
int XrdCnsInventory::Conduct(const char *dPath)
{
   static const int nsOpts = XrdOucNSWalk::Recurse | XrdOucNSWalk::retFile
                           | XrdOucNSWalk::retStat | XrdOucNSWalk::skpErrs
                           | XrdOucNSWalk::noPath  | XrdOucNSWalk::retLink;
   XrdOucNSWalk nsObj(&MLog, dPath, 0, nsOpts);
   XrdOucNSWalk::NSEnt *nP, *fP;
   int n, aOK = 1, rc;

// Index all directories here
//
   do {if (!(nP = nsObj.Index(rc, &cwdP))) return 1;
       if (!Config.LogicPath(cwdP, lfnBuff, sizeof(lfnBuff))) break;
       if ((n = strlen(lfnBuff)) && lfnBuff[n-1] == '/') lfnBuff[n-1] = '\0';
       dRec.setLfn1(lfnBuff);
       if (!(aOK = lfP->Add(&dRec, 0))) break;
       do {if (XrdOssPath::pathType(nP->Path) == XrdOssPath::isBase)
              {fRec.setMode(nP->Stat.st_mode);
               fRec.setSize(nP->Stat.st_size);
               fRec.setLfn1(nP->Path);
               if (!(aOK = Xref(nP) && lfP->Add(&fRec, 0))) break;
              }
            fP = nP; nP = nP->Next; delete fP;
          } while(nP);
      } while(aOK);

// Determine how we ended
//
   rc = (nP || rc ? 0 : 1);

// Clean up and return
//
   while((fP = nP)) {nP = nP->Next; delete fP;}
   return rc;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdCnsInventory::Init(XrdCnsLogFile *theLF)
{
   XrdCnsLogRec tRec(XrdCnsLogRec::lrTOD);

// Establish the log file we will be using
//
   lfP = theLF;

// Insert time stamp
//
   lfP->Add(&tRec, 0);

// Add default Mount record
//
   mRec.setLfn1(Mount.Key(mDflt));
   mRec.setMount(mDflt);
   if (!(lfP->Add(&mRec, 0))) return 0;

// Add default space record
//
   sRec.setLfn1(Space.Key(sDflt));
   sRec.setSpace(sDflt);
   if (!(lfP->Add(&sRec, 0))) return 0;

// All done
//
   return 1;
}

/******************************************************************************/
/* Private:                         X r e f                                   */
/******************************************************************************/
  
int XrdCnsInventory::Xref(XrdOucNSWalk::NSEnt *nP)
{
   const char *cP;
   char pBuff[MAXPATHLEN+1], xCode;
   int n;

// Add actual space information
//
   if (nP->Link && (n = nP->Lksz))
      {cP = XrdOssPath::Extract(0, nP->Link, n);
       if (!(xCode = Mount.Find(nP->Link)))
          {xCode = Mount.Add(nP->Link); mRec.setMount(xCode);
           mRec.setLfn1(nP->Link);
           if (!(lfP->Add(&mRec, 0))) return 0;
          }
       fRec.setMount(xCode);
       if (!(xCode = Space.Find(cP)))
          {xCode = Space.Add(cP); sRec.setSpace(xCode);
           sRec.setLfn1(cP);
           if (!(lfP->Add(&sRec, 0))) return 0;
          }
        fRec.setSpace(xCode);
        return 1;
       }

// Add constructed space information
//
   if (!Config.MountPath(lfnBuff, pBuff, sizeof(pBuff))) xCode = mDflt;
      else if (!(xCode = Mount.Find(pBuff)))
              {xCode = Mount.Add(pBuff);
               mRec.setMount(xCode);
               mRec.setLfn1(pBuff);
               if (!(lfP->Add(&mRec, 0))) return 0;
              }
   fRec.setMount(xCode);
   fRec.setSpace(sDflt);
   return 1;
}
