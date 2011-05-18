/******************************************************************************/
/*                                                                            */
/*                       X r d A d m i n F i n d . c c                        */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/param.h>

#include "XrdFrm/XrdFrmAdmin.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdOuc/XrdOucArgs.hh"
#include "XrdOuc/XrdOucNSWalk.hh"

const char *XrdFrmAdminFindCVSID = "$Id$";

using namespace XrdFrm;

/******************************************************************************/
/*                              F i n d F a i l                               */
/******************************************************************************/
  
int XrdFrmAdmin::FindFail(XrdOucArgs &Spec)
{
   XrdOucNSWalk::NSEnt *nP, *fP;
   XrdOucNSWalk *nsP;
   char pDir[MAXPATHLEN], *dotP, *dirFN, *lDir = Opt.Args[1];
   int opts = XrdOucNSWalk::retFile | (Opt.Recurse ? XrdOucNSWalk::Recurse : 0);
   int ec, rc = 0, num = 0;

// Process each directory
//
   do {if (!Config.LocalPath(lDir, pDir, sizeof(pDir))) continue;
       nsP = new XrdOucNSWalk(&Say, pDir, Config.lockFN, opts);
       while((nP = nsP->Index(ec)) || ec)
            {while((fP = nP))
                  {if ((dotP = rindex(fP->File,'.')) && !strcmp(dotP,".fail"))
                      {Msg(fP->Path); num++;}
                   nP = fP->Next; delete fP;
                  }
            if (ec) rc = 4;
            }
        delete nsP;
       } while((dirFN = Spec.getarg()));

// All done
//
   sprintf(pDir, "%d fail file%s found.", num, (num == 1 ? "" : "s"));
   Msg(pDir);
   return rc;
}

/******************************************************************************/
/*                              F i n d N o l k                               */
/******************************************************************************/
  
int XrdFrmAdmin::FindNolk(XrdOucArgs &Spec)
{
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char pDir[MAXPATHLEN], *lDir = Opt.Args[1];
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int ec = 0, rc = 0, num = 0;

// Process each directory
//
   do {if (!Config.LocalPath(lDir, pDir, sizeof(pDir))) continue;
       fP = new XrdFrmFiles(pDir, opts);
       while((sP = fP->Get(ec)))
            {if (!(sP->lockFile())) {Msg(sP->basePath()); num++;}
            }
       if (ec) rc = 4;
       delete fP;
      } while((lDir = Spec.getarg()));

// All done
//
   sprintf(pDir,"%d missing lock file%s found.", num, (num == 1 ? "" : "s"));
   Msg(pDir);
   return rc;
}
  
/******************************************************************************/
/*                              F i n d U n m i                               */
/******************************************************************************/
  
int XrdFrmAdmin::FindUnmi(XrdOucArgs &Spec)
{
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   const char *Why;
   char buff[128], pDir[MAXPATHLEN], *lDir = Opt.Args[1];
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int ec = 0, rc = 0, num = 0;

// Process each directory
//
   do {if (!Config.LocalPath(lDir, pDir, sizeof(pDir))) continue;
       fP = new XrdFrmFiles(pDir, opts);
       while((sP = fP->Get(ec)))
            {     if (!(sP->lockFile()))
                     Why = "Unmigrated; no lock file: ";
             else if (sP->baseFile()->Stat.st_mtime >
                      sP->lockFile()->Stat.st_mtime)
                     Why="Unmigrated; modified: ";
             else continue;
             Msg(Why, sP->basePath()); num++;
            }
       if (ec) rc = 4;
       delete fP;
      } while((lDir = Spec.getarg()));

// All done
//
   sprintf(buff,"%d file%s unmigrated files found.",num,(num == 1 ? "" : "s"));
   Msg(buff);
   return rc;
}
