#ifndef __FRMCID_H__
#define __FRMCID_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d F r m C I D . h h                           */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <stdlib.h>
#include <string.h>

#include "XrdSys/XrdSysPthread.hh"

class XrdOucEnv;
class XrdOucStream;

class XrdFrmCID
{
public:
       int    Add(const char *iName, const char *cName, time_t addT, pid_t Pid);

       int    Get(const char *iName, char *buff, int blen);

       int    Get(const char *iName, const char *vName, XrdOucEnv *evP);

       int    Init(const char *qPath);

       void   Ref(const char *iName);

              XrdFrmCID() : Dflt(0), First(0), cidFN(0), cidFN2(0) {}
             ~XrdFrmCID() {}

private:

struct cidEnt
      {cidEnt *Next;
       char   *iName;
       char   *cName;
       time_t  addT;
       pid_t   Pid;
       int     useCnt;
       short   iNLen;
       short   cNLen;

               cidEnt(cidEnt *epnt,const char *iname,const char *cname,
                      time_t addt, pid_t idp)
                     : Next(epnt), iName(strdup(*iname ? iname : "anon")),
                       cName(strdup(cname)), addT(addt), Pid(idp), useCnt(0),
                       iNLen(strlen(iName)), cNLen(strlen(cName)) {}
              ~cidEnt() {if (iName) free(iName); if (cName) free(cName);}

      };

class  cidMon {public:
               cidMon() {cidMutex.Lock();}
              ~cidMon() {cidMutex.UnLock();}
               private:
               static XrdSysMutex cidMutex;
              };

cidEnt *Find(const char *iName);
int     Init(XrdOucStream &cidFile);
int     Update();

cidEnt *Dflt;
cidEnt *First;
char   *cidFN;
char   *cidFN2;
};

namespace XrdFrm
{
extern XrdFrmCID CID;
}
#endif
