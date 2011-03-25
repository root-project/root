/******************************************************************************/
/*                                                                            */
/*                 X r d F r m A d m i n C o n v e r t . c c                  */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <sys/param.h>

#include "XrdFrm/XrdFrmAdmin.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdFrm/XrdFrmXAttr.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOuc/XrdOucArgs.hh"
#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucPList.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucXAttr.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                               C o n v e r t                                */
/******************************************************************************/
  
int XrdFrmAdmin::Convert()
{
   static const char *cHelp =
   "Usage: convert [-a[utoreply]] [-fix] [-t[est]] old2new [names] [spaces]";
   static XrdOucArgs Spec(&Say, "frm_admin: ",    "", 
                                "autoreply",   1, "F",
                                "fix",         3, "f",
                                "test",        1, "l",
                                (const char *)0);

   static const char *Reqs[] = {"mode", 0};
   int Warn, doNames = 0, doSpaces = 0, o2n = 1;

// Parse the request
//
   if (!Parse("convert ", Spec, Reqs)) return 1;

// Process the correct find
//
        if (!strncmp(Opt.Args[0], "old2new", 4)) o2n = 1;
   else if (!strncmp(Opt.Args[0], "new2Old", 4)) o2n = 0;
   else {Emsg("Unknown conversion mode - ", Opt.Args[0]); Msg(cHelp); return 4;}

// Now see what to convert
//
   while((Opt.Args[0] = Spec.getarg()))
        {     if (!strcmp(Opt.Args[0], "names"))  doNames = 1;
         else if (!strcmp(Opt.Args[0], "spaces")) doSpaces= 1;
         else {Emsg("Unknown conversion space - ", Opt.Args[0]);
               Msg(cHelp); return 4;
              }
        }

// Set conversion options
//
   if (!doNames && !doSpaces) doNames = doSpaces = 1;

// Check for actual conversion
//
   if (!Opt.Local)
      return (o2n ? Old2New(doNames, doSpaces) : New2Old(doNames, doSpaces));

// This is only a test. See if conversion could be tried
//
   if (!(o2n = ConvTest(doNames, doSpaces))) Msg("Conversion would fail!");
      else {if ((Warn  = (doNames && Config.runOld)))
               Msg("Remember to remove the 'oss.runmodeold' directive.");
            if ((Warn |= (doSpaces && Config.nonXA)))
               Msg("Change all occurrences of 'oss.cache' to 'oss.space'.");
            if (Warn)
               Msg("Then kill or restart all daemons before converting.");
            Msg("Conversion may succeed.");
           }
   return (o2n ? 0 : 8);
}

/******************************************************************************/
/*                              C o n v T e s t                               */
/******************************************************************************/

int XrdFrmAdmin::ConvTest(int doNames, int doSpaces)
{
   XrdSysError *errP;
   XrdOucTList *tP, *pP, *pList;
   const char *What;
   char pDir[MAXPATHLEN+8];
   int pdSz, rc, chkFD, Ok = 1;

// Get the paths we need to try
//
   if (doNames) {pList = x2xPaths();  What  = "namespace";}
      else      {pList = x2xSpaces(); What  = "dataspace";}
   if (!(pP = pList)) return 0;

// Turn off messages from the attribute setter
//
   errP = XrdSysFAttr::Msg(0);

// For each path, see if it supports extended attributes
//
do{if (!doNames) strcpy(pDir, pP->text+pP->val);
      else if (!Config.LocalPath(pP->text,pDir,sizeof(pDir))) {Ok = 0; break;}

   pdSz = strlen(pDir); strcpy(pDir+pdSz, (doNames ? "/.\b" : ".\b"));
   if ((chkFD = open(pDir, O_CREAT|O_RDWR, 0740)) < 0) rc = -errno;
      else {rc = XrdSysFAttr::Set("XrdFrmTest", "?", 1, pDir, chkFD);
            close(chkFD); unlink(pDir);
           }
   *(pDir+pdSz) = 0;
   if (!rc) Msg("Verified ", What, " at ", pDir);
      else {Ok = 0;
            if (rc == -ENOTSUP)
               Msg("Extended attributes disabled for ",What," at ",pDir);
               else Emsg(-rc, "determine xattr status for ", pDir);
           }
   tP = pP; pP = pP->next; delete tP;
  } while(pP);

// Cleanup
//
   XrdSysFAttr::Msg(errP);
   while(pP) {tP = pP; pP = pP->next; delete tP;}

// Check if we should do a space check and return final result
//
   return Ok & (doNames && doSpaces ? ConvTest(0, 1) : 1);
}
  
/******************************************************************************/
/*                               N e w 2 O l d                                */
/******************************************************************************/
  
int XrdFrmAdmin::New2Old(int doNames, int doSpaces)
{
   Emsg("Backward conversion is not currently supported.");
   return 4;
}

/******************************************************************************/
/*                               O l d 2 N e w                                */
/******************************************************************************/
  
int XrdFrmAdmin::Old2New(int doNames, int doSpaces)
{
   static const char *fMsg = "Do you want to proceed?";
   XrdOucTList *tP, *pP, *pList;
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   const char *What, *Only = "Only ";
   char Resp = 'y', pDir[MAXPATHLEN+8];
   int numActs = 0, numOld = 0, ec = 0, Act = 1, fsetOpts;

// Verify that we are in the correct runmode
//
   if (Config.runOld)
      {Msg("The system is still configured with 'oss.runmodeold'!");
       Msg("Remove the directive, kill or restart all daemons, and try again.");
       finalRC = 4;
       return 0;
      }

// The following establishes what we are doing
//
   fsetOpts = XrdFrmFiles::GetCpyTim | XrdFrmFiles::NoAutoDel
            | (doNames ? XrdFrmFiles::Recursive : 0);
   if (doNames) {pList = x2xPaths();  What  = "namespace";}
      else      {pList = x2xSpaces(); What  = "dataspace";}

// Bag out of we have no paths to convert run on old mode
//
   if (!(pP = pList)) return 0;
   Config.runOld = 1; Config.runNew = 0; numProb = 0; numOld = 0;

// Process each directory
//
do{if (!doNames) strcpy(pDir, pP->text+pP->val);
      else if (!Config.LocalPath(pP->text,pDir,sizeof(pDir))) {Act = 0; break;}
   Msg("Converting ", What, " starting at ", pDir);
   if (!Opt.Force && (Resp = XrdFrmUtils::Ask('y', fMsg)) == 'a') break;
   if (Resp == 'y')
      {fP = new XrdFrmFiles(pDir, fsetOpts);
       while(Act && (sP = fP->Get(ec,1)))
            {numFix = 0;
             Act = (doNames ? o2nFiles(sP, numOld) : o2nSpace(sP, pP->text));
             if (Act && numFix) numActs++;
             delete sP;
            }
       delete fP;
       if (ec || !Act) break;
      } else numProb++;
   tP = pP; pP = pP->next; delete tP;
  } while(pP && !ec && Act);

// Cleanup
//
   if (pP) {Act = 0; while(pP) {tP = pP; pP = pP->next; delete tP;}}
      else if (ec) Act = 0;

// Check for old-style spaces
//
   if (numOld && doNames)
      {sprintf(pDir, "%d old-style dataspace file%s found.", numOld,
                     (numOld == 1 ? "" : "s"));
       Msg(pDir);
      }

// Print ending status here
//
   if (!Act) Msg("Conversion aborted!");
      else Only = "";
   if (!numActs) Msg("No ", What ," conversions performed.");
      else {sprintf(pDir,"%s%d %s conversion%s performed.", Only,
                         numActs, What, (numActs == 1 ? "" : "s"));
            Msg(pDir);
            if (numProb) Msg("Warning! ", What, " conversion is incomplete.");
           }

// Re-establish new run mode now
//
   Config.runOld = 0; Config.runNew = 1;

// If no space conversion, check if it is really needed
//
   if (!doSpaces && Act)
      {if (Config.hasCache) 
          Msg("Please change 'oss.cache' to 'oss.space' directives.");
       if (numOld) Msg("You are encouraged to run 'convert old2new spaces'.");
       return 0;
      }

// Check if we should and can run space conversion
//
   if (doSpaces == 1)
      {if (Act && !numProb) return Old2New(0,2);
       Msg("Space conversion bypassed until namespace problems are fixed.");
      }

// All done
//
   return (Act && !numProb ? 0 : 2);
}
  
/******************************************************************************/
/*                              o 2 n F i l e s                               */
/******************************************************************************/
  
int XrdFrmAdmin::o2nFiles(XrdFrmFileset *sP, int &numOld)
{
   const char *basePath = sP->basePath();
   char Resp, pfnFile[1032], *linkPath;

// If we have a base and a lock file, then set the copy time attribute. But
// we must first validate that this is not a dangling symlink
//
   if (sP->baseFile())
      {if (sP->baseFile()->Type == XrdOucNSWalk::NSEnt::isLink)
          {if (Config.Verbose || !Opt.Fix)
              {Msg("Dangling link:  ",     basePath);
               Msg("Missing target: ", sP->baseFile()->Link);
              }
           if (Opt.Force) Resp = (Opt.Fix ? 'y' : 'n');
              else if ((Resp = XrdFrmUtils::Ask('y', "Remove symlink?")) == 'a')
                      return 0;
           if (Resp == 'n')
              {if (sP->lockFile())
                  {Msg("Corresponding lock file cannot be converted!");
                   numProb++;
                  }
              } else if (!x2xRemove("symlink", basePath)
                     ||  (sP->lockFile() &&
                          !x2xRemove("lock file",sP->lockPath()))) return 0;
          } else if (sP->lockFile())
                    {if (sP->cpyInfo.Set(basePath)
                     ||  !x2xRemove("lock file",sP->lockPath(), 1)) return 0;
                    }
      }

// Remove the pin file if it exists
//
   if (sP->pinFile() && !x2xRemove("pin file", sP->pinPath())) return 0;

// Remove all mmap type of files if they exist
//
   if (sP->xyzFile(XrdOssPath::isMmap)
   &&  !x2xRemove("mmap file",  sP->xyzPath(XrdOssPath::isMmap)))  return 0;
   if (sP->xyzFile(XrdOssPath::isMlock)
   &&  !x2xRemove("mlock file", sP->xyzPath(XrdOssPath::isMlock))) return 0;
   if (sP->xyzFile(XrdOssPath::isMkeep)
   &&  !x2xRemove("mkeep file", sP->xyzPath(XrdOssPath::isMkeep))) return 0;

// Check if this is a symlink elsewhere
//
   if (sP->baseFile() && (linkPath = sP->baseFile()->Link))
      {if (!isXA(sP->baseFile())) {numOld++; return 1;}
       if (XrdSysFAttr::Set(XrdFrmXAttrPfn::Name(), basePath,
                                strlen(basePath)+1, basePath))
          {Msg("Unable to set pfn xattr for ", basePath, "->", linkPath);
           return 0;
          }
       strcpy(pfnFile, linkPath); strcat(pfnFile,".pfn");
       return x2xRemove("pfn file", pfnFile, 1);
      }

// All done
//
   return 1;
}

/******************************************************************************/
/*                              o 2 n S p a c e                               */
/******************************************************************************/
  
int XrdFrmAdmin::o2nSpace(XrdFrmFileset *sP, const char *Space)
{
   XrdFrmXAttrPfn pfnInfo;
   int rc, oldNP;

// If we have no basefile or this is not an old-style file, skip it
//
   if (!sP->baseFile() || sP->baseFile()->File[0] != XrdOssPath::xChar)
      return 1;

// Get the pfn for this file
//
   if (!XrdOssPath::genPFN(pfnInfo.Pfn, sizeof(pfnInfo.Pfn), sP->basePath()))
      {Msg("Unable to get pfn for ", sP->basePath()); return 0;}

// Now make sure the Pfn exists and is a symlink
//
   oldNP = numProb;
   if ((rc = AuditSpaceAXDC(pfnInfo.Pfn, sP->baseFile())) <= 0) return rc;
   numProb = oldNP; numFix = 0;

// We must now set the pfn attribute as the reloc process will destroy it as
// we are asking it to do a pure relocation which assumes the pfn is already set
//
   if ((rc = XrdSysFAttr::Set(pfnInfo.Name(), pfnInfo.Pfn, 
                              pfnInfo.sizeSet(), sP->basePath()))
   ||  (rc = Config.ossFS->Reloc("admin", pfnInfo.Pfn, Space, ".")) < 0)
      {Emsg(-rc, "convert ", sP->basePath()); return 0;}

// All went well
//
   VSAY("Converted space for ", pfnInfo.Pfn);
   numFix = 1;
   return 1;
}

/******************************************************************************/
/*                              x 2 x P a t h s                               */
/******************************************************************************/

XrdOucTList *XrdFrmAdmin::x2xPaths()
{
   extern XrdOucPListAnchor *XrdOssRPList;
   XrdOucTList *nP, *pP, *tP, *mypList = 0;
   char *Path;
   int   Plen;

// Verify that we have an RPList
//
   if (!XrdOssRPList)
      {Say.Emsg("Convert", "Cannot determine paths to convert."); return 0;}

// Get complete path list trimmed to prevent rescans
//
   XrdOucPList *fP = XrdOssRPList->First();
   while(fP)
        {Path = fP->Path(); Plen = fP->Plen();
         nP = mypList, pP = 0;
         while(nP && Plen <= nP->val)
              {if (strncmp(Path, nP->text, Plen)) {pP = nP; nP = nP->next;}
                  else {tP = nP; nP = nP->next; delete tP;}
              }
         tP = new XrdOucTList(fP->Path(), Plen, nP);
         if (pP) pP->next = tP;
            else mypList  = tP;
         fP = fP->Next();
        }

// Make sure we have something here
//
   if (!mypList) Emsg("No convertable namespaces found.");

// All done
//
   return mypList;
}

/******************************************************************************/
/*                             x 2 x R e m o v e                              */
/******************************************************************************/
  
int XrdFrmAdmin::x2xRemove(const char *Type, const char *Path, int cvt)
{
    if (unlink(Path) && errno != ENOENT)
       {Emsg(errno, "remove ", Type, " ", Path); return 0;}
    VSAY((cvt ? "Converted " : "Removed "), Type, " ", Path);
    numFix++;
    return 1;
}

/******************************************************************************/
/*                             x 2 x S p a c e s                              */
/******************************************************************************/

XrdOucTList *XrdFrmAdmin::x2xSpaces()
{
   struct XrdFrmConfig::VPInfo *nP = Config.VPList;
   XrdOucTList *tP, *sList = 0;
   char theSpace[1024+256], *tS;
   int pOff;

// There is still an old space defined. Complain and give specific details
//
   if (!Opt.Local && Config.nonXA)
      {Msg("Configuration file still defines one or more old-style spaces.");
       Msg("You must change all occurrences of 'oss.cache' to 'oss.space'.");
       Msg("Then kill or restart all daemons, and try again.");
       finalRC = 4;
       return 0;
      }

// We will construct a list of paths that we need to scan
//
   while(nP)
        {strcpy(theSpace, nP->Name); strcat(theSpace, ":");
         pOff = strlen(theSpace);
         tS = theSpace+pOff; tP = nP->Dir;
         while(tP)
              {strcpy(tS, tP->text);
               sList = new XrdOucTList(theSpace, pOff, sList);
               tP = tP->next;
              }
         nP = nP->Next;
        }

// Return what we have
//
   if (!sList) Emsg("No convertable dataspaces found.");
   return sList;
}
