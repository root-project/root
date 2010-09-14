#ifndef __XRDFRMPROXY__
#define __XRDFRMPROXY__
/******************************************************************************/
/*                                                                            */
/*                        X r d F r m P r o x y . h h                         */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XrdFrm/XrdFrmRequest.hh"

class XrdFrmReqAgent;
class XrdOucStream;
class XrdSysLogger;
  
class XrdFrmProxy
{
public:

int   Add(char Opc, const char *Lfn, const char *Opq, const char *Usr,
                    const char *Rid, const char *Nop, const char *Pop,
          int Prty=1);

int   Del(char Opc, const char *Rid);

static const int opGet =  1;
static const int opPut =  2;
static const int opMig =  4;
static const int opStg =  8;
static const int opAll = 15;

class Queues
      {friend class XrdFrmProxy;
       int   Offset;
       char  Prty;
       char  QList;
       char  QNow;
       char  Active;
       public:
       Queues(int opX) : Offset(0), Prty(0), QList(opX), QNow(0), Active(0) {}
      ~Queues() {}
      };

int   List(Queues &State, char *Buff, int Bsz);

int   List(int qType, int qPrty, XrdFrmRequest::Item *Items, int Num);

int   Init(int opX, const char *aPath, int aMode, const char *qPath=0);

      XrdFrmProxy(XrdSysLogger *lP, const char *iName, int Debug=0);
     ~XrdFrmProxy() {}

private:

int Init2(const char *cfgFN);
int qChk(XrdOucStream &cFile);

struct o2qMap {const char *qName; int qType; int oType;};

static o2qMap   oqMap[];
static int      oqNum;

XrdFrmReqAgent *Agent[XrdFrmRequest::numQ];
const char     *insName;
char           *intName;
char           *QPath;
};
#endif
