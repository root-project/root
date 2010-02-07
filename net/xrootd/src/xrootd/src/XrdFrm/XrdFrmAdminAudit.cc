/******************************************************************************/
/*                                                                            */
/*                   X r d F r m A d m i n A u d i t . c c                    */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <stdio.h>
#include <string.h>
#include <sys/param.h>

#include "XrdFrm/XrdFrmAdmin.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucTList.hh"

const char *XrdFrmAdminAuditCVSID = "$Id$";

using namespace XrdFrm;

/******************************************************************************/
/*                           A u d i t N a m e N B                            */
/******************************************************************************/

int XrdFrmAdmin::AuditNameNB(XrdFrmFileset *sP)
{
   char Resp, buff[80];
   int num = 0, rem;

// Report what is orphaned
//
   if (sP->failFile())
      {num++; Msg("Orphaned fail file: ", sP->failPath());}
   if (sP->lockFile())
      {num++; Msg("Orphaned lock file: ", sP->lockPath());}
   if (sP->pfnFile() )
      {num++; Msg("Orphaned pfn  file: ", sP->pfnPath());
              Msg("PFN file refers to: ", sP->pfnFile()->Link);
                 }
   if (sP->pinFile() )
      {num++; Msg("Orphaned pin  file: ", sP->pinPath());}

// Return if no fix is needed, otherwise check if we should ask before removal
//
   numProb += num;
   if (!Opt.Fix || !num) return 1;
   if (!Opt.Force)
      {Resp = XrdFrmUtils::Ask('n', "Remove orphaned files?");
       if (Resp != 'y') return Resp != 'a';
      }

// Remove the orphaned files
//
   rem = AuditRemove(sP);
   numFix += rem;

// Indicate final resolution
//
   sprintf(buff, "%d of %d orphaned files removed.", rem, num);
   Msg(buff);
   return 1;
}
  
/******************************************************************************/
/*                           A u d i t N a m e N F                            */
/******************************************************************************/

int XrdFrmAdmin::AuditNameNF(XrdFrmFileset *sP)
{
   char Resp;

// Indicate what is wrong
//
   Msg("Dangling link:  ", sP->basePath());
   Msg("Missing target: ", sP->baseFile()->Link);
   numProb++;

// Return if no fix is needed, otherwise check if we should ask before removal
//
   if (!Opt.Fix) return 1;
   if (!Opt.Force)
      {Resp = XrdFrmUtils::Ask('n', "Remove symlink?");
       if (Resp != 'y') return Resp != 'a';
      }

// Remove the symlink and associated files
//
   if (unlink(sP->basePath()))
      Emsg(errno,"remove symlink", sP->basePath());
      else if (AuditRemove(sP))
              {Msg("Symlink removed.");
               numFix++;
               return 1;
              }
   return 1;
}
  
/******************************************************************************/
/*                           A u d i t N a m e N L                            */
/******************************************************************************/

int XrdFrmAdmin::AuditNameNL(XrdFrmFileset *sP)
{
   char Resp;

// Indicate what is wrong
//
   Msg("Missing lock file: ", sP->basePath());
   numProb++;

// Return if no fix is needed, otherwise check if we should ask before removal
//
   if (!Opt.Fix) return -1;
   if (!Opt.Force)
      {Resp = XrdFrmUtils::Ask('y', "Create lock file?");
       if (Resp != 'y') return Resp != 'a';
      }

// Create the lock file
//
   if (!mkFile(mkLF|isPFN, sP->basePath())) return 1;
   Msg("Lock file created.");
   numFix++;
   return 1;
}

/******************************************************************************/
/*                            A u d i t N a m e s                             */
/******************************************************************************/
  
int XrdFrmAdmin::AuditNames()
{
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char pDir[MAXPATHLEN], *lDir = Opt.Args[1];
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int ec = 0, Act = 1;

// Initialization
//
   numProb = 0; numFix = 0;
   if (VerifyMP("audit", lDir) != 'y') return 0;

// Process the directory
//
   if (!Config.LocalPath(lDir, pDir, sizeof(pDir))) {finalRC = 4; return 1;}
   fP = new XrdFrmFiles(pDir, opts);
   while(Act && (sP = fP->Get(ec,1)))
        {if (!(sP->baseFile())) Act = AuditNameNB(sP);
             else {if (sP->baseFile()->Type == XrdOucNSWalk::NSEnt::isLink)
                      Act = AuditNameNF(sP);
                   if (Act && Opt.MPType && !(sP->lockFile()))
                      Act = AuditNameNL(sP);
                   if (Act && sP->baseFile()->Link && isXA(sP->baseFile()))
                      Act = AuditNameXA(sP);
                  }
         }
    if (ec) finalRC = 4;
    delete fP;

// All done
//
   if (!Act) Msg("Audit names aborted!");
   sprintf(pDir,"%d problem%s found; %d fixed.", numProb,
                (numProb == 1 ? "" : "s"), numFix);
   Msg(pDir);
   return !Act;
}

/******************************************************************************/
/*                           A u d i t N a m e X A                            */
/******************************************************************************/
  
int XrdFrmAdmin::AuditNameXA(XrdFrmFileset *sP)
{
   struct stat buf;
   char Path[1032], lkbuff[1032];
   int n;

// Make sure there is a PFN file here
//
   strcpy(Path, sP->baseFile()->Link); strcat(Path, ".pfn");
   if (lstat(Path,&buf))
      {if (errno != ENOENT)
          {Emsg(errno, "stat ", Path); return AuditNameXL(sP,-1);}
       Msg("Missing pfn link to ", sP->basePath());
       return AuditNameXL(sP,0);
      }

// Make sure the PFN file is a link
//
   if ((buf.st_mode & S_IFMT) != S_IFLNK)
      {Msg("Invalid pfn file for ", sP->basePath());
       return AuditNameXL(sP,1);
      }

// Make sure the link points to the right file
//
   if ((n = readlink(Path, lkbuff, sizeof(lkbuff)-1)) < 0)
      {Emsg(errno, "read link from ", Path); return AuditNameXL(sP,-1);}
   lkbuff[n] = '\0';
   if (strcmp(sP->basePath(), lkbuff))
      {Msg("Incorrect pfn link to ", sP->basePath());
       return AuditNameXL(sP,1);
      }

// All is well
//
   return 1;
}

/******************************************************************************/
/*                           A u d i t N a m e X L                            */
/******************************************************************************/
  
int XrdFrmAdmin::AuditNameXL(XrdFrmFileset *sP, int dorm)
{
   char Resp, Path[1032];

// Return if no fix is needed, otherwise check if we should ask before doing it
//
   numProb++;
   if (!Opt.Fix || dorm < 0) return 1;
   if (!Opt.Force)
      {if (dorm)
          Resp = XrdFrmUtils::Ask('n', "Recreate pfn symlink?");
          else
          Resp = XrdFrmUtils::Ask('y',   "Create pfn symlink?");
       if (Resp != 'y') return Resp != 'a';
      }

// Create the pfn symlink
//
   strcpy(Path, sP->baseFile()->Link); strcat(Path, ".pfn");
   if (dorm) unlink(Path);
   if (symlink(sP->basePath(), Path))
      {Emsg(errno, "create symlink ", Path); return 1;}
   Msg("pfn symlink created.");
   numFix++;
   return 1;
}

/******************************************************************************/
/*                           A u d i t R e m o v e                            */
/******************************************************************************/
  
int XrdFrmAdmin::AuditRemove(XrdFrmFileset *sP)
{
   int rem = 0;

// Remove the orphaned files
//
   if (sP->failFile())
      {if (unlink(sP->failPath())) Emsg(errno,"remove fail file.");
          else rem++;
      }
   if (sP->lockFile())
      {if (unlink(sP->lockPath())) Emsg(errno,"remove lock file.");
          else rem++;
      }
   if (sP-> pinFile())
      {if (unlink(sP-> pinPath())) Emsg(errno,"remove pin  file.");
          else rem++;
      }
   if (sP-> pfnFile())
      {if (unlink(sP-> pinPath())) Emsg(errno,"remove pfn  file.");
          else rem++;
      }

   return rem;
}

/******************************************************************************/
/*                            A u d i t S p a c e                             */
/******************************************************************************/
  
int XrdFrmAdmin::AuditSpace()
{
   XrdOucTList   *pP;
   char buff[256], *Path = 0, *Space = Opt.Args[1];
   int Act;

// Parse the space specification
//
   if (!(pP = ParseSpace(Space, &Path))) return 4;

// Initialize
//
   numBytes = 0; numFiles = 0; numProb = 0; numFix = 0;

// Index the space via filesets
//
   do {Act = (pP->val ? AuditSpaceXA(Space, pP->text) : AuditSpaceAX(pP->text));
       pP = pP->next;
      } while(pP && !Path && Act);

// All done
//
   sprintf(buff,"%d problem%s found; %d fixed.", numProb,
                (numProb == 1 ? "" : "s"), numFix);
   Msg(buff);
   if (!Act) Msg("Audit space aborted!");
      else {if (Path) *(--Path) = ':';
            sprintf(buff, "Space %s has %d file%s with %lld byte%s in use.",
                    Space,
                    numFiles, (numFiles == 1 ? "" : "s"),
                    numBytes, (numBytes == 1 ? "" : "s"));
            Msg(buff);
           }
   return (Act ? 0 : 4);
}

/******************************************************************************/
/*                          A u d i t S p a c e A X                           */
/******************************************************************************/
  
int XrdFrmAdmin::AuditSpaceAX(const char *Path)
{
   XrdOucNSWalk nsWalk(&Say, Path, Config.lockFN, XrdOucNSWalk::retFile
                                                | XrdOucNSWalk::retStat
                                                | XrdOucNSWalk::skpErrs);
   XrdOucNSWalk::NSEnt *nP, *pP;
   char buff[1032];
   int ec, Act = 1;

// Get the files in this directory
//
   if (!(nP = nsWalk.Index(ec))) {if (ec) finalRC = 4; return 1;}
   pP = nP;

// Now traverse through all of the files
//
   while(nP && Act)
        {Act = (XrdOssPath::genPFN(buff, sizeof(buff), nP->Path)
             ? AuditSpaceAXDC(buff, nP) : AuditSpaceAXDB(nP->Path));
         nP = nP->Next;
        }

// Delete the entries and return
//
   while(pP) {nP = pP; pP = pP->Next; delete nP;}
   return Act;
}

/******************************************************************************/
/*                        A u d i t S p a c e A X D B                         */
/******************************************************************************/
  
int XrdFrmAdmin::AuditSpaceAXDB(const char *Path)
{
   char Resp;

// Indicate the problem
//
   Msg("Invalid name for data file ", Path);
   numProb++;

// Return if no fix is needed, otherwise check if we should ask before doing it
//
   if (Opt.Fix)
      {if (!Opt.Force)
          {Resp = XrdFrmUtils::Ask('n', "Delete file?");
           if (Resp != 'y') return Resp != 'a';
          }
       if (unlink(Path)) Emsg(errno, "remove ", Path);
          else numFix++;
      }
   return 1;
}

/******************************************************************************/
/*                        A u d i t S p a c e A X D C                         */
/******************************************************************************/
  
int XrdFrmAdmin::AuditSpaceAXDC(const char *Path, XrdOucNSWalk::NSEnt *nP)
{
   struct stat buf;
   char lkbuff[1032], *Dest = nP->Path;
   int n;

// Assume we have a problem
//
   numProb++;

// Verify that the link to the file exists
//
   if (lstat(Path,&buf))
      {if (errno != ENOENT) {Emsg(errno, "stat ", Path); return 1;}
       Msg("Missing pfn data link ", Path);
       return AuditSpaceAXDL(0, Path, Dest);
      }

// Make sure the PFN file is a link
//
   if ((buf.st_mode & S_IFMT) != S_IFLNK)
      {Msg("Invalid pfn data link ", Path);
       return AuditSpaceAXDL(1, Path, Dest);
      }

// Make sure tyhe link points to the right file
//
   if ((n = readlink(Path, lkbuff, sizeof(lkbuff)-1)) < 0)
      {Emsg(errno, "read link from ", Path); return 1;}
   lkbuff[n] = '\0';
   if (strcmp(Path, lkbuff))
      {Msg("Incorrect pfn data link ", Path);
       return AuditSpaceAXDL(1, Path, Dest);
      }

// All went well
//
   numProb--; numFiles++; numBytes += nP->Stat.st_size;
   return 1;
}

/******************************************************************************/
/*                        A u d i t S p a c e A X D L                         */
/******************************************************************************/
  
int XrdFrmAdmin::AuditSpaceAXDL(int dorm, const char *Path, const char *Dest)
{
   char Resp;

// Return if no fix is needed, otherwise check if we should ask before doing it
//
   if (!Opt.Fix) return -1;
   if (!Opt.Force)
      {if (dorm)
          Resp = XrdFrmUtils::Ask('n', "Recreate pfn symlink?");
          else
          Resp = XrdFrmUtils::Ask('y',   "Create pfn symlink?");
       if (Resp != 'y') return Resp != 'a';
      }

// Create the pfn symlink
//
   if (dorm) unlink(Path);
   if (symlink(Dest, Path))
      {Emsg(errno, "create symlink ", Path); return 1;}
   Msg("pfn symlink created.");
   numFix++;
   return 1;
}

/******************************************************************************/
/*                          A u d i t S p a c e X A                           */
/******************************************************************************/
  
int XrdFrmAdmin::AuditSpaceXA(const char *Space, const char *Path)
{
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char tmpv[8], *buff;
   int ec = 0, Act = 1;

// Construct the right space path and get a files object
//
   buff = XrdOssPath::genPath(Path, Space, tmpv);
   fP = new XrdFrmFiles(buff, XrdFrmFiles::Recursive);

// Go and check out the files
//
   while(Act && (sP = fP->Get(ec,1)))
        {     if (!(sP->baseFile())) Act = AuditNameNB(sP);
         else if (!(sP-> pfnFile())) Act = AuditSpaceXANB(sP);
         else {numFiles++; numBytes += sP->baseFile()->Stat.st_size; continue;}
        }

// All done
//
   if (ec) finalRC = 4;
   free(buff);
   delete fP;
   return Act;
}

/******************************************************************************/
/*                        A u d i t S p a c e X A N B                         */
/******************************************************************************/

int XrdFrmAdmin::AuditSpaceXANB(XrdFrmFileset *sP)
{
   char Resp;

// Update statistics which we may have to correct later
//
   numProb++; numFiles++; numBytes += sP->baseFile()->Stat.st_size;

// Report what is orphaned
//
   Msg("Missing pfn file for data file ", sP->basePath());

// Return if no fix is needed, otherwise check if we should ask before removal
//
   if (!Opt.Fix) return -1;
   if (!Opt.Force)
      {Resp = XrdFrmUtils::Ask('n', "Remove data file?");
       if (Resp != 'y') return Resp != 'a';
      }

// Remove the file
//
   if (unlink(sP->basePath())) Emsg(errno,"remove data file ", sP->basePath());
      else {numFix++; numFiles--; numBytes -= sP->baseFile()->Stat.st_size;}
   return 1;
}
  
/******************************************************************************/
/*                            A u d i t U s a g e                             */
/******************************************************************************/
  
int XrdFrmAdmin::AuditUsage()
{
   XrdFrmConfig::VPInfo *vP = Config.VPList;
   char Sbuff[1024];
   int retval, rc;

// Check if we have a space or we should do all spaces
//
   if (Opt.Args[1]) return AuditUsage(Opt.Args[1]);

// If no cache configured say so
//
   if (!vP) {Emsg("No outplace space has been configured."); return -1;}

// Audit usage for each space
//
   retval = 1;
   while(vP) 
        {strcpy(Sbuff, vP->Name);
         if (!(rc = AuditUsage(Sbuff))) return 0;
         if (rc < 0) retval = rc;
         vP = vP->Next;
        }
   return retval;
}
  
/******************************************************************************/
  
int XrdFrmAdmin::AuditUsage(char *Space)
{
   XrdOucTList   *pP;
   const char *Sfx;
   char Resp, buff[256], *Path = 0;
   long long theClaim, theDiff;
   int haveUsage, Probs = 0;

// Parse the space specification
//
   if (!(pP = ParseSpace(Space, &Path))) return -1;
   if (Path) {Emsg("Path not allowed for audit usage."); return -1;}

// Initialize
//
   numBytes = 0; numFiles = 0; numProb = 0;
   haveUsage = XrdOssSpace::Init();

// Index the space via filesets
//
   do {Probs |= (pP->val ? AuditUsageXA(pP->text, Space)
                         : AuditUsageAX(pP->text));
       pP = pP->next;
      } while(pP);

// Print ending condition
//
   sprintf(buff, "Audit of %d file%s in %s space completed with %serrors.",
                 numFiles, (numFiles == 1 ? "" : "s"), Space,
                 (Probs ? "" : "no "));
   Msg(buff);

// Print what is in the usage file
//
   if (haveUsage)
      {XrdOssSpace::uEnt myEnt;
       XrdOssSpace::Usage(Space, myEnt);
       theClaim = myEnt.Bytes[XrdOssSpace::Serv]
                + myEnt.Bytes[XrdOssSpace::Pstg]
                - myEnt.Bytes[XrdOssSpace::Purg]
                + myEnt.Bytes[XrdOssSpace::Admin];
       sprintf(buff, "%12lld", theClaim);
       Msg("Claimed: ", buff);
      } else theClaim = numBytes;

// Print what we came up with
//
   sprintf(buff, "%12lld", numBytes);
   Msg("Actual:  ", buff);

// Check if fix is required and wanted
//
   if (numBytes == theClaim || !Opt.Fix) return 1;
   if (!haveUsage)
      {Emsg(0, "No usage file present to fix!"); return -1;}

// Compute difference
//
   if (theClaim < numBytes) theDiff = numBytes - theClaim;
      else                  theDiff = theClaim - numBytes;

// See if we should fix this
//
   if (!Opt.Force)
      {if (theDiff < 500000) Sfx = "byte";
          {theDiff = (theDiff+512)/1024; Sfx = "KB";}
       sprintf(buff, "Fix %lld %s difference?", theDiff, Sfx);
       Resp = XrdFrmUtils::Ask('n', "Fix usage information?");
       if (Resp != 'y') return Resp != 'a';
      }

// Fix the problem
//
   XrdOssSpace::Adjust(Space, numBytes-theClaim, XrdOssSpace::Admin);
   return 1;
}

/******************************************************************************/
/*                          A u d i t U s a g e A X                           */
/******************************************************************************/
  
int XrdFrmAdmin::AuditUsageAX(const char *Path)
{
   XrdOucNSWalk nsWalk(&Say, Path, Config.lockFN, XrdOucNSWalk::retFile
                                                | XrdOucNSWalk::retStat
                                                | XrdOucNSWalk::skpErrs);
   XrdOucNSWalk::NSEnt *nP, *pP;
   int ec;

// Get the files in this directory
//
   if (!(nP = nsWalk.Index(ec))) {if (ec) finalRC = 4; return 1;}

// Now traverse through all of the files
//
   while(nP)
        {numBytes += nP->Stat.st_size;
         numFiles++;
         pP = nP;
         nP = nP->Next;
         delete pP;
        }

// All done
//
   return 0;
}

/******************************************************************************/
/*                          A u d i t U s a g e X A                           */
/******************************************************************************/
  
int XrdFrmAdmin::AuditUsageXA(const char *Path, const char *Space)
{
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char tmpv[8], *buff;
   int ec = 0;

// Construct the right space path and get a files object
//
   buff = XrdOssPath::genPath(Path, Space, tmpv);
   fP = new XrdFrmFiles(buff, XrdFrmFiles::Recursive);

// Go and check out the files
//
   while((sP = fP->Get(ec)))
        {if ((sP->baseFile()))
            {numFiles++; numBytes += sP->baseFile()->Stat.st_size;}
        }

// All done
//
   free(buff);
   delete fP;
   return ec;
}
  
/******************************************************************************/
/*                                  i s X A                                   */
/******************************************************************************/
  
int XrdFrmAdmin::isXA(XrdOucNSWalk::NSEnt *nP)
{
   char *lP;

   if (!(nP->Link)) return 0;
   lP = nP->Link + nP->Lksz -1;
   return (*lP == XrdOssPath::xChar);
}
