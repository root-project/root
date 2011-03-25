/******************************************************************************/
/*                                                                            */
/*                  X r d F r m A d m i n U n l i n k . c c                   */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmAdmin.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdNet/XrdNetCmsNotify.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOuc/XrdOucNSWalk.hh"

using namespace XrdFrm;
  
/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdFrmAdminNSE
{public: 

 XrdOucNSWalk::NSEnt *nP, *dP;
 XrdOucNSWalk        *nsP;

 XrdFrmAdminNSE() : nP(0), dP(0), nsP(0) {}
~XrdFrmAdminNSE() {XrdOucNSWalk::NSEnt *fP;
                   while((fP = dP)) {dP = dP->Next; delete fP;}
                   while((fP = nP)) {nP = nP->Next; delete fP;}
                   if (nsP) delete nsP;
                  }
};
  
/******************************************************************************/
/*                                U n l i n k                                 */
/******************************************************************************/
  
int XrdFrmAdmin::Unlink(const char *Path)
{
   static const int ulOpts = XRDOSS_Online | XRDOSS_isPFN;
   XrdOucNSWalk::NSEnt *fP;
   XrdFrmAdminNSE NSE;
   struct stat Stat;
   char Resp, lclPath[MAXPATHLEN+8];
   int aOK = 1, rc;

// Get the actual pfn for the base file
//
   if (!Config.LocalPath(Path, lclPath, sizeof(lclPath)-8))
      {numProb++; return -1;}

// Make sure the base file exists
//
   if (stat(lclPath, &Stat))
      {Emsg(errno,"remove ",lclPath); numProb++; return -1;}

// Check if the file is actually a directory or a plain file
//
   if ((Stat.st_mode & S_IFMT) != S_IFDIR)
      {if (Opt.All) {Emsg(ENOTDIR, "remove ", Path); numProb++; return -1;}
       return UnlinkFile(lclPath);
      }

// This is a directory, see if a non-recursive delete wanted
//
   if (!Opt.Recurse) return UnlinkDir(Path, lclPath);

// Get confirmation unless not wanted
//
   if (!Opt.Force)
      {Resp = XrdFrmUtils::Ask('n', "Remove EVERYTHING starting at ",Path,"?");
       if (Resp != 'y') return Resp != 'a';
      }

// Create the name space object to return the contents of each directory
//
   NSE.nsP = new XrdOucNSWalk(&Say, lclPath, 0, XrdOucNSWalk::Recurse
                             |XrdOucNSWalk::retAll | XrdOucNSWalk::retStat);

// Process each directory
//
   while((NSE.nP = NSE.nsP->Index(rc)) && !rc)
        {if ((rc = UnlinkDir(NSE.nP, NSE.dP)) < 0) break;
         rc = 0;
        }
   aOK = !rc;

// Check if we can now delete the directories
//
   while((fP = NSE.dP))
        {if (aOK)
            {if ((rc = Config.ossFS->Remdir(fP->Path, ulOpts)))
                {Emsg(-rc, "remove directory ", fP->Path); aOK = 0; numProb++;}
                else {if (Opt.Echo) Msg("Local directory ",fP->Path," removed.");
                      numDirs++;
                     }
            }
         NSE.dP = NSE.dP->Next; delete fP;
        }

// Now remove the base directory
//
   if (aOK)
      {if ((rc = Config.ossFS->Remdir(lclPath, ulOpts)))
          {Emsg(-rc, "remove directory ", lclPath); aOK = 0;}
          else {numDirs++;
                if (Opt.Echo) Msg("Local directory ", lclPath, " removed.");
               }
      }

// All done
//
   return aOK ? 1 : -1;
}

/******************************************************************************/
/*                             U n l i n k D i r                              */
/******************************************************************************/
  
int XrdFrmAdmin::UnlinkDir(const char *Path, const char *lclPath)
{
   static const int ulOpts = XRDOSS_Online | XRDOSS_isPFN;
   XrdFrmAdminNSE NSE;
   XrdOucNSWalk::NSEnt *fP;
   char Resp;
   int rc;

// Create the name space object to return the contents of each directory
//
   NSE.nsP = new XrdOucNSWalk(&Say, lclPath, 0, XrdOucNSWalk::retAll 
                                              | XrdOucNSWalk::retStat);

// Get the entries in this directory
//
   NSE.nP = NSE.nsP->Index(rc);
   if (rc) {numProb++; return -1;}

// If the only entry is the DIR_LOCK file then we can remove it and the
// directory without asking
//
   if (!Opt.All)
      {if (NSE.nP && !NSE.nP->Next && !strcmp(Config.lockFN, NSE.nP->Path))
          if (unlink(NSE.nP->Path)) {Emsg(-rc, "remove ", lclPath); return -1;}
       if ((rc = Config.ossFS->Remdir(lclPath, ulOpts)))
          {Emsg(-rc, "remove directory ", lclPath); numProb++; return -1;}
       if (Opt.Echo) Msg("Local directory ", Path, " removed.");
       numDirs++;
       return 1;
      }

// Run through the list looking to see if we have any directories
//
   fP = NSE.nP;
   while(fP)
        {if (fP->Type != XrdOucNSWalk::NSEnt::isDir) fP = fP->Next;
            else {Emsg(EISDIR, "remove ", fP->Path); numProb++; return -1;}
        }

// If neither 'all' nor 'force' not specified, then we must ask for permission
//
   if (!Opt.Force)
      {Resp = XrdFrmUtils::Ask('n', "Remove EVERYTHING in ",Path,"?");
       if (Resp != 'y') return Resp != 'a';
      }
                                                                               
// Remove all items in this directory
//
   if ((rc = UnlinkDir(NSE.nP, NSE.dP)) < 0) return -1;
   return 1;
}

/******************************************************************************/
  
int XrdFrmAdmin::UnlinkDir(XrdOucNSWalk::NSEnt *&nP, XrdOucNSWalk::NSEnt *&dP)
{

   XrdOucNSWalk::NSEnt *fP;
   int retval = 1;

// Remove each entry but remember any directories
//
   while((fP = nP))
        {nP = fP->Next;
         if (fP->Type == XrdOucNSWalk::NSEnt::isDir)
            {fP->Next = dP; dP = fP;}
            else {if (UnlinkFile(fP->Path) < 0) retval = -1;
                  delete fP;
                 }
        }

// All done
//
   return retval;
}

/******************************************************************************/
/*                            U n l i n k F i l e                             */
/******************************************************************************/
  
int XrdFrmAdmin::UnlinkFile(const char *lclPath)
{
   static const int ulOpts = XRDOSS_Online | XRDOSS_isMIG | XRDOSS_isPFN;
   int rc;

// Remove this file as needed
//
   if (XrdOssPath::pathType(lclPath))
      {if (!unlink(lclPath) || errno == ENOENT) return 1;
       rc = -errno;
      } else {
       if (!(rc = Config.ossFS->Unlink(lclPath, ulOpts)))
          {if (Opt.Echo) Msg("Local file ", lclPath, " removed.");
           if (Config.cmsPath) Config.cmsPath->Gone(lclPath);
           numFiles++;
           return 1;
          }
      }

// Unlink failed
//
   Emsg(-rc, "remove ", lclPath);
   numProb++;
   return -1;
}
