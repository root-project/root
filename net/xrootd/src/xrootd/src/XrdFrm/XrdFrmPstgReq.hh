#ifndef __FRMPSTGREQ_H__
#define __FRMPSTGREQ_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d F r m P s t g R e q . h h                       */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

class XrdFrmPstgReq
{
public:

struct Request
{
char      LFN[2176];    // Logical File Name (optional '\0' opaque)
char      User[256];    // User trace identifier
char      ID[64];       // Request ID
char      Notify[512];  // Notification path
char      Reserved[40]; // Reserved for future
long long addTOD;       // Time added to queue
int       This;         // Offset to this request
int       Next;         // Offset to next request
int       Opaque;       // Offset to '?' in LFN if exists, 0 o/w
short     Options;      // Processing options (see class definitions)
short     Prty;         // Request priority
};

static const int msgFail  = 0x0001;
static const int msgSucc  = 0x0002;
static const int stgRW    = 0x0004;

static const int maxPrty  = 2;

static const int ReqSize  = sizeof(Request);

       void   Add(Request *rP);

       void   Can(Request *rP);

       void   Del(Request *rP);

       int    Get(Request *rP);

       int    Init();

enum Item {getLFN=0, getLFNCGI, getMODE, getNOTE, getPRTY, getQWT,
           getRID,   getTOD,    getUSER};

       char  *List(char *Buff, int bsz, int &Offs, Item *ITList=0, int ITNum=0);

       void   ListL(XrdFrmPstgReq::Request tmpReq, char *Buff, int bsz,
                    Item *ITList, int ITNum);

static int    Unique(const char *lkfn);

              XrdFrmPstgReq(const char *fn);
             ~XrdFrmPstgReq() {}

private:
enum LockType {lkNone, lkShare, lkExcl, lkInit};

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

struct recEnt {recEnt *Next;
               struct Request reqData;
               recEnt(struct Request &reqref) {Next = 0; reqData = reqref;}
              };
int    ReWrite(recEnt *rP);
};
namespace XrdFrm
{
extern XrdFrmPstgReq     *rQueue[XrdFrmPstgReq::maxPrty];
}
#endif
