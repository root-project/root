/******************************************************************************/
/*                                                                            */
/*                        X r d F r m F i l e s . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include <string.h>
#include <sys/types.h>

#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmTrace.hh"

const char *XrdFrmFilesCVSID = "$Id$";

using namespace XrdFrm;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmFiles::XrdFrmFiles(const char *dname, int opts)
            : nsObj(&Say, dname, Config.lockFN,
                    XrdOucNSWalk::retFile | XrdOucNSWalk::retLink
                   |XrdOucNSWalk::retStat | XrdOucNSWalk::skpErrs
                   | (opts & Recursive  ?   XrdOucNSWalk::Recurse : 0)),
              fsList(0)
{}

/******************************************************************************/
/*                                   G e t                                    */
/******************************************************************************/
  
XrdFrmFileset *XrdFrmFiles::Get(int &rc, int noBase)
{
   XrdOucNSWalk::NSEnt *nP;
   XrdFrmFileset *fsetP;

// Check if we have something to return
//
do{while ((fsetP = fsList))
         {fsList = fsetP->Next; fsetP->Next = 0;
          if (fsetP->File[XrdOssPath::isBase] || noBase) {rc = 0; return fsetP;}
         }

// Start with next directory (we return when no directories left)
//
   do {if (!(nP = nsObj.Index(rc))) return 0;
       fsTab.Purge(); fsList = 0;
      } while(!Process(nP));

  } while(1);

// To keep the compiler happy
//
   return 0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               P r o c e s s                                */
/******************************************************************************/
  
int XrdFrmFiles::Process(XrdOucNSWalk::NSEnt *nP)
{
   XrdOucNSWalk::NSEnt *fP;
   XrdFrmFileset       *sP;
   char *dotP;
   int fType;

// Process the file list
//
   while((fP = nP))
        {nP = fP->Next; fP->Next = 0;
         if (!strcmp(fP->File, Config.lockFN)) {delete fP; continue;}
         if ((dotP = rindex(fP->File, '.'))) *dotP = '\0';
         if (!(sP = fsTab.Find(fP->File)))
            {sP = fsList = new XrdFrmFileset(fsList); fsTab.Add(fP->File, sP);}
         if (dotP) *dotP = '.';
         fType = (int)XrdOssPath::pathType(fP->File);
         sP->File[fType] = fP;
        }

// Indicate whether we have anything here
//
   return fsList != 0;
}
