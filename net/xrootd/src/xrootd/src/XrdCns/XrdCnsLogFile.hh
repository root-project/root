#ifndef __XRDCnsLogFile_H_
#define __XRDCnsLogFile_H_
/******************************************************************************/
/*                                                                            */
/*                      X r d C n s L o g F i l e . h h                       */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdCns/XrdCnsLogRec.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdCnsLogFile
{
public:

XrdCnsLogFile *Next;

int            Add(XrdCnsLogRec *Rec, int doSync=1);

int            Commit();

int            Eol();

const char    *FName() {return logFN;}

char          *getLog(int &Dlen) {Dlen = logNext-logBuff; return logBuff;}

XrdCnsLogRec  *getRec();

static void    maxRecs(int nRecs) {logRMax = nRecs;
                                   logBMax = nRecs * sizeof(XrdCnsLogRec);
                                  }

int            Open(int aBuff=1, off_t thePos=0);

XrdCnsLogFile *Subscribe(const char *Path, int cNum);

int            Unlink();

               XrdCnsLogFile(const char *Path, int cnum=0, int Wait=1)
                            : Next(0), logSem(0), subNext(0),
                              logBuff(0),logNext(0), logFN(strdup(Path)),
                              logFD(-1), logRdr(cnum), logWait(Wait),
                              logOffset(0), recOffset(0) {}
              ~XrdCnsLogFile();

private:
int                   Read(char *buff, int blen);

static int            logRMax;
static int            logBMax;

XrdSysMutex           logMutex;
XrdSysSemaphore       logSem;
XrdSysSemaphore       synSem;
XrdCnsLogFile        *subNext;

XrdCnsLogRec          Rec;

char                 *logBuff;
char                 *logNext;

char                 *logFN;
int                   logFD;
int                   logRdr;
int                   logWait;
int                   logOffset;
int                   recOffset;
};
#endif
