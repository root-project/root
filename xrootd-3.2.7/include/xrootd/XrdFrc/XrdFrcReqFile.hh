#ifndef __FRCREQFILE_H__
#define __FRCREQFILE_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d F r c R e q F i l e . h h                       */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdFrc/XrdFrcRequest.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdFrcReqFile
{
public:

       void   Add(XrdFrcRequest *rP);

       void   Can(XrdFrcRequest *rP);

       void   Del(XrdFrcRequest *rP);

       int    Get(XrdFrcRequest *rP);

       int    Init();

       char  *List(char *Buff, int bsz, int &Offs,
                    XrdFrcRequest::Item *ITList=0, int ITNum=0);

       void   ListL(XrdFrcRequest &tmpReq, char *Buff, int bsz,
                    XrdFrcRequest::Item *ITList, int ITNum);

              XrdFrcReqFile(const char *fn, int aVal);
             ~XrdFrcReqFile() {}

private:
enum LockType {lkNone, lkShare, lkExcl, lkInit};

static const int ReqSize  = sizeof(XrdFrcRequest);

void   FailAdd(char *lfn, int unlk=1);
void   FailCan(char *rid, int unlk=1);
void   FailDel(char *lfn, int unlk=1);
int    FailIni(const char *lfn);
int    FileLock(LockType ltype=lkExcl);
int    reqRead(void *Buff, int Offs);
int    reqWrite(void *Buff, int Offs, int updthdr=1);

struct FileHdr
{
int    First;
int    Last;
int    Free;
}      HdrData;

char  *lokFN;
int    lokFD;
int    reqFD;
char  *reqFN;

int    isAgent;

struct recEnt {recEnt       *Next;
               XrdFrcRequest reqData;
               recEnt(XrdFrcRequest &reqref) {Next = 0; reqData = reqref;}
              };
int    ReWrite(recEnt *rP);

class rqMonitor
{
public:
      rqMonitor(int isAgent) : doUL(isAgent)
                  {if (isAgent) rqMutex.Lock();}
     ~rqMonitor() {if (doUL)    rqMutex.UnLock();}
private:
static XrdSysMutex rqMutex;
int                doUL;
};
};
#endif
