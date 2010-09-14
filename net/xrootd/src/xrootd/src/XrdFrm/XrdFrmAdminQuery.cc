/******************************************************************************/
/*                                                                            */
/*                      X r d A d m i n Q u e r y . c c                       */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "XrdFrm/XrdFrmAdmin.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmProxy.hh"
#include "XrdFrm/XrdFrmRequest.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOuc/XrdOucArgs.hh"
#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucTList.hh"

const char *XrdFrmAdminQueryCVSID = "$Id$";

using namespace XrdFrm;

/******************************************************************************/
/*                              Q u e r y P f n                               */
/******************************************************************************/
  
int XrdFrmAdmin::QueryPfn(XrdOucArgs &Spec)
{
   char *lfn, pfn[MAXPATHLEN];

// Get the first lfn
//
   if (!(lfn = Spec.getarg())) {Emsg("lfn not specified."); return 1;}

// Process all of the files
//
   do {if (Config.LocalPath(lfn, pfn, sizeof(pfn))) Msg(pfn);
          else finalRC = 4;
      } while((lfn = Spec.getarg()));
   return 0;
}

/******************************************************************************/
/*                              Q u e r y R f n                               */
/******************************************************************************/
  
int XrdFrmAdmin::QueryRfn(XrdOucArgs &Spec)
{
   char *lfn, rfn[MAXPATHLEN];

// Get the first lfn
//
   if (!(lfn = Spec.getarg())) {Emsg("lfn not specified."); return 1;}

// Process all of the files
//
   do {if (Config.RemotePath(lfn, rfn, sizeof(rfn))) Msg(rfn);
          else finalRC = 4;
      } while((lfn = Spec.getarg()));
   return 0;
}

/******************************************************************************/
/*                            Q u e r y S p a c e                             */
/******************************************************************************/
  
int XrdFrmAdmin::QuerySpace(XrdOucArgs &Spec)
{
   XrdFrmConfig::VPInfo *vP = Config.VPList;
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   XrdOucTList   *tP;
   struct stat Stat;
   char buff[2048], pfn[MAXPATHLEN], *lfn;
   int opts = 0, ec = 0;

// If no cache configured say so
//
   if (!vP) {Emsg("No outplace space has been configured."); return 0;}

// Get the first lfn (optional)
//
   lfn = Spec.getarg();

// List the cache we have if no lfn exists
//
   if (!lfn)
      {while(vP)
            {tP = vP->Dir;
             while(tP)
                  {sprintf(buff, "%s %s", vP->Name, tP->text);
                   Msg(buff, (tP->val ? " xa" : 0));
                   tP = tP->next;
                  }
             vP = vP->Next;
            }
       return 0;
      }

// Check if this is '-recursive'
//
   if (!strncmp(lfn, "-recursive", strlen(lfn)))
      {opts = XrdFrmFiles::Recursive;
       if (!(lfn = Spec.getarg()))
          {Emsg("lfn not specified."); return 0;}
      }

// Here we display thespace name of each lfn
//
   do {Opt.All = VerifyAll(lfn);
            if (!Config.LocalPath(lfn, pfn, sizeof(pfn))) finalRC = 4;
       else if (stat(pfn, &Stat)) Emsg(errno, "query ", pfn);
       else if ((Stat.st_mode & S_IFMT) != S_IFDIR)
               {if (Opt.All) Emsg(ENOTDIR, "query ", lfn);
                   else QuerySpace(pfn);
               }
       else{fP = new XrdFrmFiles(pfn, opts);
            while((sP = fP->Get(ec,1)))
                 {if (sP->baseFile())
                     QuerySpace(sP->basePath(),
                                sP->baseFile()->Link,
                                sP->baseFile()->Lksz);
                 }
            if (ec) finalRC = 4;
            delete fP;
           }
      } while((lfn = Spec.getarg()));

// All done
//
   return 0;
}
  
/******************************************************************************/

int XrdFrmAdmin::QuerySpace(const char *Pfn, char *Lnk, int Lsz)
{
   char SName[XrdOssSpace::minSNbsz];

// Get the space name
//
   XrdOssPath::getCname(Pfn, SName, Lnk, Lsz);
   Msg(SName, " ", Pfn);
   return 0;
}

/******************************************************************************/
/*                            Q u e r y U s a g e                             */
/******************************************************************************/
  
int XrdFrmAdmin::QueryUsage(XrdOucArgs &Spec)
{
   XrdOssSpace::uEnt myUsage;
   XrdFrmConfig::VPInfo myVP((char *)""), *vP = Config.VPList;
   long long Actual;
   char buff[4096];

// Check if usage has been configured
//
   if (!(XrdOssSpace::Init() & XrdOssSpace::haveUsage))
      {Emsg("Usage is not being tracked."); return 0;}

// Get the optional space name
//
   if ((myVP.Name = Spec.getarg())) {myVP.Next = 0; vP = &myVP;}
      else if (!vP) {Emsg("No outplace space has been configured."); return 0;}

// Process all of the files
//
   do {if (XrdOssSpace::Usage(vP->Name, myUsage, 1) < 0)
          Emsg("Space ", vP->Name, " not found.");
          else
         {Actual = myUsage.Bytes[XrdOssSpace::Serv]
                 + myUsage.Bytes[XrdOssSpace::Pstg]
                 - myUsage.Bytes[XrdOssSpace::Purg]
                 + myUsage.Bytes[XrdOssSpace::Admin];
          sprintf(buff,"Space %s\n%20lld Used\n%20lld Staged\n"
                       "%20lld Purged\n%20lld Adjust\n%20lld Effective",
                  vP->Name, myUsage.Bytes[XrdOssSpace::Serv],
                            myUsage.Bytes[XrdOssSpace::Pstg],
                            myUsage.Bytes[XrdOssSpace::Purg],
                            myUsage.Bytes[XrdOssSpace::Admin], Actual);
          Msg(buff);
         }
      } while((vP = vP->Next));
   return 0;
}
  
/******************************************************************************/
/*                             Q u e r y X f r Q                              */
/******************************************************************************/
  
int XrdFrmAdmin::QueryXfrQ(XrdOucArgs &Spec)
{
   static struct {const char *qName; char qType;} qN2T[] =
                 {{"all",    XrdFrmProxy::opAll},
                  {"get",    XrdFrmProxy::opGet},
                  {"migr",   XrdFrmProxy::opMig},
                  {"put",    XrdFrmProxy::opPut},
                  {"migrate",XrdFrmProxy::opMig},
                  {"stage",  XrdFrmProxy::opStg},
                  {0, 0}};

   XrdFrmRequest::Item Items[XrdFrmRequest::getLast];
   XrdFrmProxy::Queues xfrQ(0);
   char *qName;
   int i, qPrty, QList = 0;

// Check for proxy initialization
//
   if (!frmProxy && !frmProxz) ConfigProxy();

// Get the first q-type
//
   while((qName = Spec.getarg()))
        {i = 0;
         while(qN2T[i].qName && strcmp(qN2T[i].qName, qName)) i++;
         if (qN2T[i].qName) QList |= qN2T[i].qType;
            else break;
        }

// Set queue if none specified
//
   if (!QList) QList = XrdFrmProxy::opAll;

// Check if priority
//
   if (qName && strlen(qName) == 1 && *qName >= '0' && *qName <= '9')
      {qPrty = *qName - '0';
       if (qPrty > XrdFrmRequest::maxPrty)
          {Emsg("Invalid xfrq priority - ", qName); return 1;}
       qName = Spec.getarg();
      } else qPrty = -1;

// Process variable is we have an unmatched name
//
   i = 0;
   if (qName)
      {do {if (XrdFrmUtils::MapV2I(qName, Items[i])) i++;
              else {Emsg("Invalid xfrq variable - ", qName); return 1;}
          } while((qName = Spec.getarg()) && i < XrdFrmRequest::getLast);
       if (qName) {Emsg("Too many xfrq variables starting at ",qName);return 1;}
      } else Items[i++] = XrdFrmRequest::getLFN;

// Produce the listing
//
   if (!frmProxy) {Emsg("Unable to list the xfrq."); return 1;}
   if (!frmProxy->List(QList, qPrty, Items, i)) Msg("No entries found.");
   return 0;
}
