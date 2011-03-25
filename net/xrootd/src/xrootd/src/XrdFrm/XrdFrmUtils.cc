/******************************************************************************/
/*                                                                            */
/*                        X r d F r m U t i l s . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <utime.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdFrm/XrdFrmXLock.hh"
#include "XrdFrm/XrdFrmXAttr.hh"

#include "XrdOuc/XrdOucSxeq.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdOuc/XrdOucXAttr.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;
  
/******************************************************************************/
/*                                   A s k                                    */
/******************************************************************************/
  
char XrdFrmUtils::Ask(char dflt, const char *Msg1, const char *Msg2,
                                 const char *Msg3)
{
   const char *Hint;
   char Answer[8];
   int n;

   Hint = (dflt == 'y' ? " (y | n | a): " : " (n | y | a): ");

   do {cerr <<"frm_admin: " <<Msg1 <<Msg2 <<Msg3 <<Hint;
       cin.getline(Answer, sizeof(Answer));
       if (!*Answer) return dflt;

       n = strlen(Answer);
       if (!strncmp("yes",  Answer, n)) return 'y';
       if (!strncmp("no",   Answer, n)) return 'n';
       if (!strncmp("abort",Answer, n)) return 'a';
      } while(1);
   return 'a';
}
  
/******************************************************************************/
/*                                c h k U R L                                 */
/******************************************************************************/
  
int XrdFrmUtils::chkURL(const char *Url)
{
   const char *Elem;

// Verify that this is a valid url and return offset to the lfn
//
   if (!(Elem = index(Url, ':'))) return 0;
   if (Elem[1] != '/' || Elem[2] != '/') return 0;
   if (!(Elem = index(Elem+3, '/')) || Elem[1] != '/') return 0;
   Elem++;

// At this point ignore all leading slashes but one
//
   while(Elem[1] == '/') Elem++;
   return Elem - Url;
}

/******************************************************************************/
/*                              m a k e P a t h                               */
/******************************************************************************/
  
char *XrdFrmUtils::makePath(const char *iName, const char *Path, int Mode)
{
   char *bPath;
   int rc;

// Generate an frm-specific admin path
//
   bPath = XrdOucUtils::genPath(Path, iName, "frm");

// Create the admin directory if it does not exists and a mode supplied
//
   if (Mode > 0 && (rc = XrdOucUtils::makePath(bPath, Mode)))
      {Say.Emsg("makePath", rc, "create directory", bPath);
       return 0;
      }

// Return the actual adminpath we are to use (this has been strduped).
//
   return bPath;
}

/******************************************************************************/
/*                              m a k e Q D i r                               */
/******************************************************************************/
  
char *XrdFrmUtils::makeQDir(const char *Path, int Mode)
{
   char qPath[1032], qLink[2048];
   int n, lksz, rc;

// Generate an frm-specific queue path
//
   strcpy(qPath, Path);
   n = strlen(qPath);
   if (qPath[n-1] != '/') qPath[n++] = '/';
   strcpy(qPath+n, "Queues/");

// If the target is a symlink, optimize the path
//
   if ((lksz = readlink(qPath, qLink, sizeof(qLink))) > 0)
      {qLink[lksz] = '\0';
       if (qLink[lksz-1] != '/') {qLink[lksz++] = '/'; qLink[lksz++] = '\0';}
       if (*qLink == '/') strcpy(qPath, qLink);
          else strcpy(qPath+n, qLink);
      }

// Create the queue directory if it does not exists
//
   if (Mode > 0 && (rc = XrdOucUtils::makePath(qPath, Mode)))
      {Say.Emsg("makeQDir", rc, "create directory", qPath);
       return 0;
      }

// Return the actual adminpath we are to use
//
   return strdup(qPath);
}

/******************************************************************************/
/*                                M a p M 2 O                                 */
/******************************************************************************/
  
int XrdFrmUtils::MapM2O(const char *Nop, const char *Pop)
{
   int Options = 0;

// Map processing options to request options
//
   if (index(Pop, 'w')) Options |= XrdFrmRequest::makeRW;
      if (*Nop != '-')
         {if (index(Pop, 's') ||  index(Pop, 'n'))
             Options |= XrdFrmRequest::msgSucc;
          if (index(Pop, 'f') || !index(Pop, 'q'))
             Options |= XrdFrmRequest::msgFail;
         }

// All done
//
   return Options;
}
  
/******************************************************************************/
/*                                M a p R 2 Q                                 */
/******************************************************************************/
  
int XrdFrmUtils::MapR2Q(char Opc, int *Flags)
{

// Simply map the request code to the relevant queue
//
   switch(Opc)
         {case 0  :
          case '+': return XrdFrmRequest::stgQ;
          case '^': if (Flags) *Flags = XrdFrmRequest::Purge;
          case '&': return XrdFrmRequest::migQ;
          case '<': return XrdFrmRequest::getQ;
          case '=': if (Flags) *Flags |= XrdFrmRequest::Purge;
          case '>': return XrdFrmRequest::putQ;
          default:  break;
         }
   return XrdFrmRequest::nilQ;
}
  
/******************************************************************************/
/*                                M a p V 2 I                                 */
/******************************************************************************/
  
int XrdFrmUtils::MapV2I(const char *vName, XrdFrmRequest::Item &ICode)
{
   static struct ITypes {const char *IName; XrdFrmRequest::Item ICode;}
                 ITList[] = {{"lfn",    XrdFrmRequest::getLFN},
                             {"lfncgi", XrdFrmRequest::getLFNCGI},
                             {"mode",   XrdFrmRequest::getMODE},
                             {"obj",    XrdFrmRequest::getOBJ},
                             {"objcgi", XrdFrmRequest::getOBJCGI},
                             {"op",     XrdFrmRequest::getOP},
                             {"prty",   XrdFrmRequest::getPRTY},
                             {"qwt",    XrdFrmRequest::getQWT},
                             {"rid",    XrdFrmRequest::getRID},
                             {"tod",    XrdFrmRequest::getTOD},
                             {"note",   XrdFrmRequest::getNOTE},
                             {"tid",    XrdFrmRequest::getUSER}};
   static const int ITNum = sizeof(ITList)/sizeof(struct ITypes);
   int i;

// Simply map the variable name to the item code
//
   for (i = 0; i < ITNum; i++)
       if (!strcmp(vName, ITList[i].IName))
          {ICode = ITList[i].ICode; return 1;}
   return 0;
}
  
/******************************************************************************/
/*                                U n i q u e                                 */
/******************************************************************************/
  
int XrdFrmUtils::Unique(const char *lkfn, const char *myProg)
{
   static const int Mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   FLOCK_t lock_args;
   int myFD, rc;

// Open the lock file first in r/w mode
//
   if ((myFD = open(lkfn, O_RDWR|O_CREAT, Mode)) < 0)
      {Say.Emsg("Unique",errno,"open",lkfn); return 0;}

// Establish locking options
//
   bzero(&lock_args, sizeof(lock_args));
   lock_args.l_type =  F_WRLCK;

// Perform action.
//
   do {rc = fcntl(myFD,F_SETLK,&lock_args);}
       while(rc < 0 && errno == EINTR);
   if (rc < 0) 
      {Say.Emsg("Unique", errno, "obtain the run lock on", lkfn);
       Say.Emsg("Unique", "Another", myProg, "may already be running!");
       close(myFD);
       return 0;
      }

// All done
//
   return 1;
}
  
/******************************************************************************/
/*                               u p d t C p y                                */
/******************************************************************************/
  
int XrdFrmUtils::updtCpy(const char *Pfn, int Adj)
{
   XrdOucXAttr<XrdFrmXAttrCpy> cpyInfo;
   struct stat Stat;

// Make sure the base file exists
//
   if (stat(Pfn, &Stat)) {Say.Emsg("updCpy", errno,"stat pfn ",Pfn); return 0;}

// Set correct copy time based on need
//
   cpyInfo.Attr.cpyTime = static_cast<long long>(Stat.st_mtime + Adj);
   return cpyInfo.Set(Pfn) == 0;
}

/******************************************************************************/
/*                                 U t i m e                                  */
/******************************************************************************/
  
int XrdFrmUtils::Utime(const char *Path, time_t tVal)
{
   struct utimbuf tbuf = {tVal, tVal};
   int rc;

// Set the time
//
   do {rc = utime(Path, &tbuf);} while(rc && errno == EINTR);
   if (rc) Say.Emsg("Utils", errno, "set utime for pfn", Path);

// All done
//
   return rc == 0;
}
