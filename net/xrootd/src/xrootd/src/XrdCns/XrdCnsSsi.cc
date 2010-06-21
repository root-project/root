/******************************************************************************/
/*                                                                            */
/*                          X r d C n s S s i . c c                           */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCnsSsiCVSID = "$Id$";

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/uio.h>

#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucSxeq.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucUtils.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include "XrdCns/XrdCnsLog.hh"
#include "XrdCns/XrdCnsSsi.hh"
#include "XrdCns/XrdCnsSsiCfg.hh"
#include "XrdCns/XrdCnsSsiSay.hh"
#include "XrdCns/XrdCnsXref.hh"
#include "XrdCns/XrdCnsLogRec.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

struct XrdCnsSsiFRec
{
char Info[XrdCnsLogRec::FixDLen];

void Updt(const char *nInfo) {strncpy(Info, nInfo, sizeof(Info));}

     XrdCnsSsiFRec(const char *Data) {if (!Data) Data = XrdCnsLogRec::iArg;
                                      strncpy(Info, Data, sizeof(Info));
                                      *Info = 'i';
                                     }
    ~XrdCnsSsiFRec() {}
};
  
struct XrdCnsSsiDRec
{
XrdOucHash<XrdCnsSsiFRec> *Files;
char                       Info[XrdCnsLogRec::FixDLen];

     XrdCnsSsiDRec(const char *Data) {if (!Data) Data = XrdCnsLogRec::IArg;
                                      strncpy(Info, Data, sizeof(Info));
                                      *Info = 'I';
                                      Files = new XrdOucHash<XrdCnsSsiFRec>;
                                     }
    ~XrdCnsSsiDRec() {if (Files) delete Files;}
};

/******************************************************************************/
/*           G l o b a l   C o n f i g u r a t i o n   O b j e c t            */
/******************************************************************************/

namespace XrdCns
{
extern XrdCnsSsiCfg               Config;

extern XrdSysError                MLog;

extern XrdCnsSsiSay               Say;

       XrdOucHash<XrdCnsSsiDRec> *hInv;
       XrdCnsXref             *mountP;
       XrdCnsXref             *spaceP;
}

int    XrdCnsSsi::nErrs = 0;
int    XrdCnsSsi::nDirs = 0;
int    XrdCnsSsi::nFiles= 0;

using namespace XrdCns;
  
/******************************************************************************/
/*                   E x t e r n a l   I n t e r f a c e s                    */
/******************************************************************************/

int XrdCnsSsiApplyF(const char *Path, XrdCnsSsiFRec *fP, void *Arg)
{
   static struct iovec iov[3] = {{0,sizeof(fP->Info)},{0,0},{(char *)"\n",1}};
   int n, iFD = *(int *)Arg;

   iov[0].iov_base = (char *)fP->Info;
   iov[1].iov_base = (char *)Path; n = strlen(Path);
   iov[1].iov_len  = n;
   XrdCnsSsi::nFiles++;

   return !XrdCnsSsi::Write(iFD, iov, 3, sizeof(fP->Info)+n+1);
}

int XrdCnsSsiApplyD(const char *Path, XrdCnsSsiDRec *dP, void *Arg)
{
   static struct iovec iov[3] = {{0,sizeof(dP->Info)},{0,0},{(char *)"\n",1}};
   int n, iFD = *(int *)Arg;

// Return if there are no files in this directory
//
   if (dP->Files->Num() <= 0) return 0;

// Write out a directory record. Terminate processing upon error
//
   iov[0].iov_base = (char *)dP->Info;
   iov[1].iov_base = (char *)Path; n = strlen(Path);
   iov[1].iov_len  = n;
   XrdCnsSsi::nDirs++;

   if (!XrdCnsSsi::Write(iFD, iov, 3, sizeof(dP->Info)+n+1)) return 1;

// Now index through all of the file in this directory
//
   return (dP->Files->Apply(XrdCnsSsiApplyF, Arg) ? 1 : 0);
}

int XrdCnsSsiApplyM(const char *Mount, char *xP, void *Arg)
{
   static char Hdr[XrdCnsLogRec::FixDLen];
   static XrdCnsLogRec::Arg *aP = (XrdCnsLogRec::Arg *)Hdr;
   static struct iovec iov[3] = {{Hdr,sizeof(Hdr)},{0,0},{(char *)"\n",1}};
   static int doInit = 1;
   int n, iFD = *(int *)Arg;

// Initialize the header (needs to be done once)
//
   if (doInit)
      {memset(Hdr, ' ', sizeof(Hdr));
       aP->Type = XrdCnsLogRec::lrMount;
       doInit = 0;
      }

// Write out a directory record. Terminate processing upon error
//
   aP->Mount = *xP;
   iov[1].iov_base = (char *)Mount; n = strlen(Mount);
   iov[1].iov_len  = n;

   return !XrdCnsSsi::Write(iFD, iov, 3, sizeof(Hdr)+n+1);
}

int XrdCnsSsiApplyS(const char *Space, char *xP, void *Arg)
{
   static char Hdr[XrdCnsLogRec::FixDLen];
   static XrdCnsLogRec::Arg *aP = (XrdCnsLogRec::Arg *)Hdr;
   static struct iovec iov[3] = {{Hdr,sizeof(Hdr)},{0,0},{(char *)"\n",1}};
   static int doInit = 1;
   int n, iFD = *(int *)Arg;

// Initialize the header (needs to be done once
//
   if (doInit)
      {memset(Hdr, ' ', sizeof(Hdr));
       aP->Type = XrdCnsLogRec::lrSpace;
       doInit = 0;
      }

// Write out a directory record. Terminate processing upon error
//
   aP->Space = *xP;
   iov[1].iov_base = (char *)Space; n = strlen(Space);
   iov[1].iov_len  = n;

   return !XrdCnsSsi::Write(iFD, iov, 3, sizeof(Hdr)+n+1);
}
  
/******************************************************************************/
/*                                  L i s t                                   */
/******************************************************************************/
  
int XrdCnsSsi::List(const char *Host, const char *Path)
{
   XrdOucStream myIF;
   XrdCnsXref Mount("/",0), Space("public",0);
   XrdCnsLogRec::Arg *aP = 0;
   XrdOucNSWalk::NSEnt *nsP, *nsL, *nsI = 0;
   struct stat Stat;
   char pfxBuff[512], *pfxP = pfxBuff, *omP, *osP, *oSP;
   char hBuff[256], oBuff[MAXPATHLEN+1], *ofP = oBuff;
   char *tP, *lP;
   int pendLog, iFD;
   int dName = Config.Lopt & XrdCnsSsiCfg::Lname;
   int dMount= Config.Lopt & XrdCnsSsiCfg::Lmount;

// First step is to get the files in this directory
//
   nsL = XrdCnsLog::List(Path, &nsI, 1);
   pendLog = (nsL != 0);
   if (nsI) delete nsI;
   while((nsP = nsL)) {nsL = nsL->Next; delete nsP;}

// If we have no inventory say so and return
//
   if (!nsI) {Say.M("No inventory found for ", Host); return 4;}

// Open the inventory file and attach it to a stream
//
   strcpy(oBuff, Path); strcat(oBuff,"/"); strcat(oBuff, XrdCnsLog::invFNz);
   if ((iFD = open(oBuff, O_RDONLY)) < 0)
      {Say.M("Unable to process ",oBuff,"; ",
             XrdOucUtils::eText(errno, pfxBuff, sizeof(pfxBuff)));
       return 1;
      }
   myIF.Attach(iFD, 4096);

// Preformat the output buffer
//
   if (Config.Lopt & XrdCnsSsiCfg::Lhost)
      {strcpy(pfxP, Host); pfxP += strlen(pfxP); *pfxP++ = ' ';}

   if (Config.Lopt & XrdCnsSsiCfg::Lmode)
      {omP = pfxP; pfxP += sizeof(aP->Mode); *pfxP++ = ' ';
      } else omP = 0;

   oSP = osP = 0;
   if (Config.Lopt & XrdCnsSsiCfg::Lsize)
      {if (Config.Lopt & XrdCnsSsiCfg::Lfmts) oSP = pfxP;
          else osP = pfxP;
       pfxP += sizeof(aP->SorT); *pfxP++ = ' ';
      }
   *pfxP = '\0';

// The first line should be a time stamp. If it is not us the file's ctime
//
   if ((lP = myIF.GetLine()) && *lP)
      {aP = (struct XrdCnsLogRec::Arg *)lP;
       if (aP->Type != XrdCnsLogRec::lrTOD) fstat(iFD,&Stat);
          else {Stat.st_ctime = strtol(aP->SorT,0,10) + XrdCnsLogRec::tBase;
                if (*(aP->lfn) && *(aP->lfn) != ' ')
                   {strcpy(hBuff, aP->lfn); Host = hBuff;}
                lP = myIF.GetLine();
               }
      } else fstat(iFD,&Stat);
   tP = ctime(&Stat.st_ctime); tP[strlen(tP)-1] = '\0';

 // Produce the header
 //
   cout <<Host <<(pendLog? " in":" ") <<"complete inventory as of " <<tP <<endl;

// Produce the listing
//
   if (lP && *lP)
   do {aP = (struct XrdCnsLogRec::Arg *)lP;
       switch(aP->Type)
             {case XrdCnsLogRec::lrMount: Mount.Add(aP->lfn, aP->Mount);
                                          continue;
              case XrdCnsLogRec::lrSpace: Space.Add(aP->lfn, aP->Space);
                                          continue;
              case XrdCnsLogRec::lrInvD:  strcpy(oBuff, aP->lfn);
                                          ofP = oBuff + strlen(oBuff);
                                          *ofP++ = '/';
                                          continue;
              default: break;
             }
       if (omP) memcpy(omP, aP->Mode, sizeof(aP->Mode));
       if (oSP) FSize( oSP, aP->SorT, sizeof(aP->SorT));
       if (osP) memcpy(osP, aP->SorT, sizeof(aP->SorT));
       if (*pfxBuff) cout <<pfxBuff;

       if (dName) cout <<Space.Key(aP->Space) <<' ';
       strcpy(ofP, aP->lfn);
       if (dMount) cout <<oBuff <<" -> " <<Mount.Key(aP->Mount) <<endl;
          else     cout <<oBuff <<endl;

      } while((lP = myIF.GetLine()) && *lP);

// All done
//
   return 0;
}

/******************************************************************************/
/*                                  U p d t                                   */
/******************************************************************************/
  
int XrdCnsSsi::Updt(const char *Host, const char *Path)
{
   static const int AMode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
   class nsHelper
        {public:
         void Set(XrdOucNSWalk::NSEnt *nsP) {nsBase = nsP;}
              nsHelper() : nsBase(0) {}
             ~nsHelper() {XrdOucNSWalk::NSEnt *nsP;
                          while((nsP = nsBase))
                               {nsBase = nsP->Next; delete nsP;}
                         }
         private:
         XrdOucNSWalk::NSEnt *nsBase;
        };

   XrdOucStream myIF;
   XrdCnsXref Mount("/",0), Space("public",0);
   XrdOucSxeq   myUP(".cns_ssi_updt.", Host);
   XrdOucHash<XrdCnsSsiDRec> myInv;
   XrdCnsSsiDRec *curDir = 0;
   XrdCnsSsiFRec *curFile;
   XrdCnsLogRec::Arg *aP = 0;
   XrdOucNSWalk::NSEnt *nsP, *nsL, *nsI = 0;
   nsHelper theNS;
   struct stat Stat;
   char cSave, iBuff[MAXPATHLEN+1], oBuff[MAXPATHLEN+1], *lP;
   int iFD, rc, TOD = 0;

// Make sure we are the only ones running here for this directory
//
   if (!myUP.Serialize(XrdOucSxeq::noWait|XrdOucSxeq::Unlink))
      {rc = myUP.lastError();
       if (rc == EAGAIN)
          Say.M(Host, " inventory is already being updated.");
          else Say.M("Unable to update ", Host, " inventory; ",
                     XrdOucUtils::eText(rc, oBuff, sizeof(oBuff)));
       return 8;
      }

// Now get the files in this directory
//
   nsL = XrdCnsLog::List(Path, &nsI, 1);

// If we have no inventory say so and return
//
   if (!nsI) {Say.M("No inventory found for ", Host);
              theNS.Set(nsL);
              return 0;
             }

// If no pending log files, no need to update the inventory
//
   if (!nsL) {Say.V(Host," inventory is up to date.");
              delete nsI;
              return 0;
             }

// Make sure that the full ns list is deleted
//
   nsI->Next = nsL;
   theNS.Set(nsI);
   hInv = &myInv;
   mountP = &Mount;
   spaceP = &Space;
   nErrs = nDirs = nFiles = 0;

// Open the inventory file and attach it to a stream
//
   strcpy(iBuff, Path); strcat(iBuff,"/"); strcat(iBuff, XrdCnsLog::invFNz);
   if ((iFD = open(iBuff, O_RDONLY)) < 0)
      {Say.M("Unable to process ",iBuff,"; ",
             XrdOucUtils::eText(errno, oBuff, sizeof(oBuff)));
       return 1;
      }
   myIF.Attach(iFD, 4096);

// The first line should be a time stamp. If it is, throw it away.
//
   if ((lP = myIF.GetLine()) && *lP)
      {aP = (struct XrdCnsLogRec::Arg *)lP;
       if (aP->Type == XrdCnsLogRec::lrTOD) lP = myIF.GetLine();
      }

// Populate the hash table with the inventory
//
   if (lP && *lP)
   do {aP = (struct XrdCnsLogRec::Arg *)lP;
       switch(aP->Type)
             {case XrdCnsLogRec::lrMount: Mount.Add(aP->lfn, aP->Mount); break;
              case XrdCnsLogRec::lrSpace: Space.Add(aP->lfn, aP->Space); break;
              case XrdCnsLogRec::lrInvD:  curDir = AddDir(aP->lfn, lP);  break;
              default: if (curDir)
                          {curFile = new XrdCnsSsiFRec(lP);
                           curDir->Files->Rep(aP->lfn, curFile);
                          } else Say.M("Ignoring file '", aP->lfn,
                                       "'; missing directory in inventory.");
                       break;
             }
      } while((lP = myIF.GetLine()) && *lP);

// Done with the inventory
//
   fstat(iFD, &Stat);
   myIF.Close();

// Now apply each log file against the inventory
//
   nsP = nsL;
   while(nsP)
        {if (nsP->Stat.st_ctime <= Stat.st_ctime)
            Say.V("Skipping ",nsP->File,"; too old.");
            else if (!(TOD = ApplyLog(nsP->Path))) return 8;
         nsP = nsP->Next;
        }

// Now we can open a shadow inventory file
//
   strcpy(oBuff, iBuff); lP = rindex(oBuff, '/')+1;
   cSave = *lP; *lP = 'i';
   if ((iFD = open(oBuff, O_CREAT|O_TRUNC|O_WRONLY, AMode)) < 0)
      {Say.M("Unable to create ", oBuff, "; ",
             XrdOucUtils::eText(errno, iBuff, sizeof(iBuff)));
       return 8;
      }

// Create a TOD record based on the last TOD received
//
   if (!Write(iFD, TOD, Host)) return 8;

// Output the space names ans mount points
//
   Mount.Apply(XrdCnsSsiApplyM, (void *)&iFD);
   Space.Apply(XrdCnsSsiApplyS, (void *)&iFD);

// Now output the whole name space into the inventory file
//
   if (myInv.Apply(XrdCnsSsiApplyD, (void *)&iFD)) {close(iFD); return 8;}

// Close the file and rename it
//
   close(iFD);
   if (rename(oBuff, iBuff))
      {Say.M("Unable to rename ",oBuff," to the inventory; ",
             XrdOucUtils::eText(errno, iBuff, sizeof(iBuff)));
       return 8;
      }

// Now unlink all of the log files we processed
//
   while(nsL) {unlink(nsL->Path); nsL = nsL->Next;}

// Success. All resources will be deleted upon return
//
   sprintf(oBuff, "%d director%s and %d file%s updated with %d error%s.",
           nDirs, (nDirs != 1 ? "ies" : "y"), nFiles, (nFiles != 1 ? "s" : ""),
           nErrs, (nErrs != 1 ? "s" : ""));
   Say.M(Host, " inventory with ", oBuff);
   return 0;
}

/******************************************************************************/
/*                              A p p l y L o g                               */
/******************************************************************************/
  
int XrdCnsSsi::ApplyLog(const char *Path)
{
   XrdOucStream myLog;
   XrdCnsLogRec::Arg *aP = 0;
   char eBuff[64], *lP;
   int logFD, TOD = 0, oldErrs = nErrs;

// Open the log file
//
   if ((logFD = open(Path, O_RDONLY)) < 0)
      {Say.M("Unable to process ",Path,"; ",
             XrdOucUtils::eText(errno, eBuff, sizeof(eBuff)));
       return 0;
      }
   myLog.Attach(logFD, 4096);
   Say.V("Processing log file ", Path);

// Update the hash table with the log file
//
   while((lP = myLog.GetLine()) && *lP)
        {aP = (struct XrdCnsLogRec::Arg *)lP;
              if (*(aP->lfn) == '/') ApplyLogRec(lP);
         else if (aP->Type == XrdCnsLogRec::lrEOL
              ||  aP->Type == XrdCnsLogRec::lrTOD) TOD = atoi(aP->SorT);
         else {Say.V("Invalid log record: ", lP); nErrs++;}
        }

// Check if we need to issue a warning
//
   if (oldErrs != nErrs) Say.M("Errors encountered processing log ", Path);

// Check if we should manufacture a TOD
//
   if (!TOD)
      {struct stat Stat;
       fstat(logFD, &Stat);
       TOD = Stat.st_ctime - XrdCnsLogRec::tBase;
      }

// All done, the file will be closed by the stream on exit
//
   return TOD;
}

/******************************************************************************/
/*                           A p p l y L o g R e c                            */
/******************************************************************************/
  
void XrdCnsSsi::ApplyLogRec(char *lP)
{
   XrdCnsLogRec::Arg *aP = (XrdCnsLogRec::Arg *)lP;
   XrdCnsSsiDRec *theDir;
   char Type = aP->Type, *lfn = aP->lfn, *fnP = 0;

// Preprosess the record to establish dir/fn relationships
//
   if (aP->Type != XrdCnsLogRec::lrMkdir
   &&  aP->Type != XrdCnsLogRec::lrRmdir
   &&  aP->Type != XrdCnsLogRec::lrCreate
   &&  aP->Type != XrdCnsLogRec::lrMv)
      {if (!(fnP = rindex(aP->lfn+1, '/')) || !(*(fnP+1))) Type = 0;
          else *fnP++ = '\0';
      }

// Switch on record type
//
   switch (Type)
          {case XrdCnsLogRec::lrClosew: AddSize(lfn,  fnP, lP); break;
           case XrdCnsLogRec::lrCreate: AddFile(lfn,       lP); break;
           case XrdCnsLogRec::lrMkdir:  AddDir (lfn,       lP); break;
           case XrdCnsLogRec::lrRm:     if ((theDir  = hInv->Find(lfn)))
                                           theDir->Files->Del(fnP);
                                                                break;
           case XrdCnsLogRec::lrRmdir:  hInv->Del(lfn);         break;
           case XrdCnsLogRec::lrMv:     if (AddDel(lfn, lP))    break;
           default: if (fnP)  *(fnP -1) = '/';
                    Say.V("Invalid log record ", lP);
                    nErrs++;
          }
}

/******************************************************************************/
/* Private:                       A d d D i r                                 */
/******************************************************************************/
  
XrdCnsSsiDRec *XrdCnsSsi::AddDir(char *dP, char *lP)
{
   XrdCnsSsiDRec *theDir;

// Find the directory or create one
//
   if (!(theDir  = hInv->Find(dP)))
      {theDir = new XrdCnsSsiDRec(lP);
       hInv->Add(dP, theDir);
      }
   return theDir;
}

/******************************************************************************/
/* Private:                       A d d D e l                                 */
/******************************************************************************/
  
int XrdCnsSsi::AddDel(char *pPo, char *lP)
{
   XrdCnsSsiDRec *newDir, *oldDir = hInv->Find(pPo);
   XrdCnsSsiFRec *oldFile, *newFile;
   char *diP = 0, *fnPo = 0, *fnPn = 0, *pPn = 0;

// Isolate the two lfn's
//
   if (!(pPn = index(pPo, ' '))) return 0;
   *pPn++ = '\0';

// First see if this is a directory rename
//
   if (oldDir)
      {newDir = AddDir(pPn, oldDir->Info);
       delete newDir->Files;
       newDir->Files = oldDir->Files;
       oldDir->Files = 0;
       hInv->Del(pPo);
       return 1;
      }

// Prepare for a file rename
//
   if (!(fnPo = rindex(pPo, '/')) || !(*(fnPo+1))
   ||  !(fnPn = rindex(pPn, '/')) || !(*(fnPn+1))) {*(pPn-1) = ' '; return 0;}
   *fnPo++ = '\0'; *fnPn++ = '\0';

// Now delete the old file
//
   if ((oldDir = hInv->Find(pPo)) && (oldFile = oldDir->Files->Find(fnPo)))
      {newFile = new XrdCnsSsiFRec(oldFile->Info);
       diP = oldDir->Info;
       oldDir->Files->Del(fnPo);
      } else newFile = new XrdCnsSsiFRec(0);

// Add the new file
//
   newDir = AddDir(pPn, diP);
   newDir->Files->Add(fnPn, newFile);

// All done
//
   return 1;
}

/******************************************************************************/
/* Private:                      A d d F i l e                                */
/******************************************************************************/

XrdCnsSsiFRec *XrdCnsSsi::AddFile(char *lfn, char *lP)
{
   XrdCnsLogRec::Arg *aP = (XrdCnsLogRec::Arg *)lP;
   XrdCnsSsiDRec *theDir;
   XrdCnsSsiFRec *theFile;
   char *fP, *mP, *sP;

// Extract out the directory, file name, space name and mount point, if any
//
   if ((sP = index(lfn, ' '))) *sP++ = '\0';
   if (!(fP = rindex(lfn+1, '/')) || !(*(fP+1)))
      {if (sP) *(sP-1) = ' ';
       Say.V("Invalid log record ", lP); nErrs++;
       return 0;
      }
   *fP++ = '\0';
   if (sP)
      {if (!(mP = index(sP, ' '))) aP->Mount = mountP->Default();
          else {*mP++ = '\0';      aP->Mount = mountP->Add(mP);}
       if (*sP)                    aP->Space = spaceP->Add(sP);
          else                     aP->Space = spaceP->Default();
      } else {                     aP->Mount = mountP->Default();
                                   aP->Space = spaceP->Default();
      }


// Add the file if it does not exist
//
   theDir = AddDir(lfn, 0);
   if ((theFile = theDir->Files->Find(fP))) theFile->Updt(lP);
      else {theFile = new XrdCnsSsiFRec(lP);
            theDir->Files->Add(fP, theFile);
           }
   return theFile;
}

/******************************************************************************/

XrdCnsSsiFRec *XrdCnsSsi::AddFile(char *dP, char *fP, char *lP)
{
   XrdCnsSsiDRec *theDir = AddDir(dP, 0);
   XrdCnsSsiFRec *theFile;

// Add the file if it does not exist
//
   if ((theFile = theDir->Files->Find(fP))) theFile->Updt(lP);
      else {theFile = new XrdCnsSsiFRec(lP);
            theDir->Files->Add(fP, theFile);
           }
   return theFile;
}

/******************************************************************************/
/* Private:                      A d d S i z e                                */
/******************************************************************************/
  
void XrdCnsSsi::AddSize(char *dP, char *fP, char *lP)
{
   XrdCnsLogRec::Arg *nP = (XrdCnsLogRec::Arg *)lP;
   XrdCnsSsiDRec *theDir = hInv->Find(dP);
   XrdCnsSsiFRec *theFile;

// Find directory
//
   if (!theDir || !(theFile = theDir->Files->Find(fP)))
      theFile = AddFile(dP, fP, 0);
   XrdCnsLogRec::Arg *aP = (XrdCnsLogRec::Arg *)(theFile->Info);
   strncpy(aP->SorT, nP->SorT, sizeof(aP->SorT));
}

/******************************************************************************/
/* Private:                        F S i z e                                  */
/******************************************************************************/
  
void XrdCnsSsi::FSize(char *oP, char *iP, int bsz)
{
   static const long long Kval = 1024LL;
   static const long long Mval = 1024LL*1024LL;
   static const long long Gval = 1024LL*1024LL*1024LL;
   static const long long Tval = 1024LL*1024LL*1024LL*1024LL;
   long long val;
   char buff[32], sName = ' ';
   int n, resid;

// Convert the number
//
   val = strtoll(iP, 0, 10);

// Get correct scaling
//
        if (val < 1024) {memcpy(oP, iP, bsz); return;}
        if (val < Mval) {val = val*10/Kval; sName = 'K';}
   else if (val < Gval) {val = val*10/Mval; sName = 'M';}
   else if (val < Tval) {val = val*10/Gval; sName = 'G';}
   else                 {val = val*10/Tval; sName = 'T';}
   resid = val%10LL; val = val/10LL;

// Format it
//
   n = sprintf(buff,"%lld.%d%c", val, resid, sName);
   memset(oP, ' ', bsz);
   strncpy(oP+(bsz-n), buff, n);
}

/******************************************************************************/
/*                                 W r i t e                                  */
/******************************************************************************/
  
int XrdCnsSsi::Write(int xFD, char *bP, int bL)
{
   char eBuff[64];
   int rc;

   do {do {rc = write(xFD, bP, bL);} while (rc < 0 && errno == EINTR);
       if (rc < 0) {Say.M("Unable to update inventory; ",
                          XrdOucUtils::eText(errno, eBuff, sizeof(eBuff)));
                    return 0;
                   }
       bP += rc; bL -= rc;
      } while(bL > 0);

   return 1;
}

/******************************************************************************/
  
int XrdCnsSsi::Write(int xFD, struct iovec *iov, int n, int Bytes)
{
   char eBuff[64], *Buff;
   int rc, i, Blen;

   do {rc = writev(xFD, iov, n);} while(rc < 0 && errno == EINTR);
   if (rc < 0) {Say.M("Unable to update inventory; ",
                      XrdOucUtils::eText(errno, eBuff, sizeof(eBuff)));
                return 0;
               }

   if (rc == Bytes) return 1;

   for (i = 0; i < n; i++)
       {if (Bytes >= (int)iov[i].iov_len) Bytes -= iov[i].iov_len;
           else {Buff = Bytes + (char *)iov[i].iov_base;
                 Blen = iov[i].iov_len  - Bytes;
                 if (!Write(xFD, Buff, Blen)) return 0;
                 Bytes = 0;
                }
       }

   return 1;
}

/******************************************************************************/
  
int XrdCnsSsi::Write(int iFD, int TOD, const char *Host)
{
   XrdCnsLogRec::Arg tRec;
   char buff[32];
   int n;

   memset(&tRec, ' ', XrdCnsLogRec::FixDLen);
   tRec.Type = XrdCnsLogRec::lrTOD;
   memset(tRec.Mode, '0', sizeof(tRec.Mode));
   n = sprintf(buff, "%d", TOD);
   memcpy(tRec.SorT+sizeof(tRec.SorT)-n, buff, n);
   strcpy(tRec.lfn, Host);
   n = strlen(tRec.lfn);
   tRec.lfn[n] = '\n';
   return Write(iFD, (char *)&tRec, XrdCnsLogRec::FixDLen+n+1);
}
