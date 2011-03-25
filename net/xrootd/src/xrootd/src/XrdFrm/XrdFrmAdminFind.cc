/******************************************************************************/
/*                                                                            */
/*                       X r d A d m i n F i n d . c c                        */
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
#include "XrdOuc/XrdOucArgs.hh"
#include "XrdOuc/XrdOucNSWalk.hh"

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
       nsP = new XrdOucNSWalk(&Say, pDir, 0, opts);
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
/*                              F i n d M m a p                               */
/******************************************************************************/
  
int XrdFrmAdmin::FindMmap(XrdOucArgs &Spec)
{
   XrdOucXAttr<XrdFrmXAttrMem> memInfo;
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char buff[128], pDir[MAXPATHLEN], *lDir = Opt.Args[1];
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int mFlag, ec = 0, rc = 0, num = 0;

// Process each directory
//
   do {if (!Config.LocalPath(lDir, pDir, sizeof(pDir))) continue;
       fP = new XrdFrmFiles(pDir, opts | XrdFrmFiles::NoAutoDel);
       while((sP = fP->Get(ec)))
            {if (memInfo.Get(sP->basePath()) >= 0
             &&  (mFlag = memInfo.Attr.Flags))
                {const char *Kp = (mFlag&XrdFrmXAttrMem::memKeep ? "-keep ":0);
                 const char *Lk = (mFlag&XrdFrmXAttrMem::memLock ? "-lock ":0);
                 Msg("mmap ", Kp, Lk, sP->basePath()); num++;
                }
             delete sP;
            }
       if (ec) rc = 4;
       delete fP;
      } while((lDir = Spec.getarg()));

// All done
//
   sprintf(buff,"%d mmapped file%s found.",num,(num == 1 ? "" : "s"));
   Msg(buff);
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

// This mode is not supported in new-style spaces
//
   if (Config.runNew)
      {Msg("New runmode does not have any lock files!"); return 0;}

// Process each directory
//
   do {if (!Config.LocalPath(lDir, pDir, sizeof(pDir))) continue;
       fP = new XrdFrmFiles(pDir, opts | XrdFrmFiles::NoAutoDel);
       while((sP = fP->Get(ec)))
            {if (!(sP->lockFile())) {Msg(sP->basePath()); num++;} // runOld
             delete sP;
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
/*                              F i n d P i n s                               */
/******************************************************************************/
  
int XrdFrmAdmin::FindPins(XrdOucArgs &Spec)
{
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char buff[128], pDir[MAXPATHLEN], *lDir = Opt.Args[1];
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int ec = 0, rc = 0, num = 0;

// Process each directory
//
   do {if (!Config.LocalPath(lDir, pDir, sizeof(pDir))) continue;
       fP = new XrdFrmFiles(pDir, opts | XrdFrmFiles::NoAutoDel);
       while((sP = fP->Get(ec)))
            {if (sP->pinInfo.Get(sP->basePath()) >= 0 && FindPins(sP)) num++;
             delete sP;
            }
       if (ec) rc = 4;
       delete fP;
      } while((lDir = Spec.getarg()));

// All done
//
   sprintf(buff,"%d pinned file%s found.",num,(num == 1 ? "" : "s"));
   Msg(buff);
   return rc;
}
  
/******************************************************************************/

int XrdFrmAdmin::FindPins(XrdFrmFileset *sP)
{
   static const int Week = 7*24*60*60;
   time_t pinVal;
   const char *Pfx = "+";
   char How[256], Scale;
   int pinFlag;

// If no pif flags are set then no pin value exists
//
   if (!(pinFlag = sP->pinInfo.Attr.Flags)) return 0;

// If it's pinned forever, we can blither the message right away
//
   if (pinFlag & XrdFrmXAttrPin::pinPerm)
      {Msg("pin -k forever ", sP->basePath()); return 1;}

// Be optimistic and get the pin time value
//
   pinVal = static_cast<time_t>(sP->pinInfo.Attr.pinTime);
   *How = 0;

// If it's a keep then decide how to format it. If the keep has been exceeed
// then just delete the attribute, since it isn't pinned anymore.
//
   if (pinFlag & XrdFrmXAttrPin::pinKeep)
      {time_t nowT = time(0);
       if (pinVal <= nowT) {sP->pinInfo.Del(sP->basePath()); return 0;}
       if ((pinVal - nowT) <= Week)
          {pinFlag = XrdFrmXAttrPin::pinIdle;
           pinVal = pinVal - nowT;
           Pfx = "";
          } else {
           struct tm *lclTime = localtime(&pinVal);
           sprintf(How, "-k %02d/%02d/%04d", lclTime->tm_mon+1,
                        lclTime->tm_mday, lclTime->tm_year+1900);
          }
      }

// Check for idle pin or convert keep pin to suedo-idle formatting
//
   if (pinFlag & XrdFrmXAttrPin::pinIdle)
      {     if ( pinVal        <= 180) Scale = 's';
       else if ((pinVal /= 60) <=  90) Scale = 'm';
       else if ((pinVal /= 60) <=  45) Scale = 'h';
       else {    pinVal /= 24;         Scale = 'd';}
       sprintf(How, "-k %s%d%c", Pfx, static_cast<int>(pinVal), Scale);
      } else if (!*How) return 0;

// Print the result
//
    Msg("pin ", How, " ", sP->basePath());
    return 1;
}

/******************************************************************************/
/*                              F i n d U n m i                               */
/******************************************************************************/
  
int XrdFrmAdmin::FindUnmi(XrdOucArgs &Spec)
{
   static const char *noCPT = (Config.runNew ? "Unmigrated; no copy time: "
                                             : "Unmigrated; no lock file: ");
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   const char *Why;
   char buff[128], pDir[MAXPATHLEN], *lDir = Opt.Args[1];
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0)|XrdFrmFiles::GetCpyTim;
   int ec = 0, rc = 0, num = 0;

// Process each directory
//
   do {if (!Config.LocalPath(lDir, pDir, sizeof(pDir))) continue;
       fP = new XrdFrmFiles(pDir, opts | XrdFrmFiles::NoAutoDel);
       while((sP = fP->Get(ec)))
            {     if (!(sP->cpyInfo.Attr.cpyTime))
                     Why = noCPT;
             else if (static_cast<long long>(sP->baseFile()->Stat.st_mtime) >
                      sP->cpyInfo.Attr.cpyTime)
                     Why="Unmigrated; modified: ";
             else Why = 0;
             if (Why) {Msg(Why, sP->basePath()); num++;}
             delete sP;
            }
       if (ec) rc = 4;
       delete fP;
      } while((lDir = Spec.getarg()));

// All done
//
   sprintf(buff,"%d unmigrated file%s found.",num,(num == 1 ? "" : "s"));
   Msg(buff);
   return rc;
}
