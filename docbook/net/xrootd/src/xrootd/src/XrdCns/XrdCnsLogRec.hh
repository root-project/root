#ifndef __XRDCnsLogRec_H_
#define __XRDCnsLogRec_H_
/******************************************************************************/
/*                                                                            */
/*                       X r d C n s L o g R e c . h h                        */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/param.h>
#include <sys/types.h>

#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdCnsLogRec
{
public:

static const int maxClients = 4;

struct Ctl
      {short     dataLen;           // Length of data in Arg (eof when 0)
       short     lfn1Len;           // strlen(lfn1)
       short     lfn2Len;           // strlen(lfn2)
       short     Mode;              // File mode (create and mkdir)
       char      Done[maxClients];  // 1->Record processed
       int       Rsvd;              // Reserved field
       long long Size;              // Valid when Closew
      };

struct Arg
      {char Type;                   // Event code
       char Mode[3];                // Mode (create, inv, mkdir)
       char SorT[12];               // Size (closew, inv) | TOD (eol)
       char Mount;                  // Mount Index
       char Space;                  // Space Index
       char Rsvd[6];                // Reserved (blank filled)
       char lfn[MAXPATHLEN*2+3];    // lfn1 [lfn2] \n
      };

struct LogRec
      {struct Ctl Hdr;
       struct Arg Data;
      };

static const int  OffDone  = offsetof(LogRec, Hdr.Done);
static const int  FixDLen  = offsetof(Arg, lfn);
static const int  MinSize  = sizeof(Ctl);  // Header
static const int  MaxSize  = sizeof(Arg);  // Data
static const long tBase    = 1248126834L;

static const char lrClosew = 't';
static const char lrCreate = 'c';
static const char lrEOL    = 'E'; // Internal
static const char lrInvD   = 'I'; // Internal dir   inventory
static const char lrInvF   = 'i'; // Internal file  inventory
static const char lrMkdir  = 'd';
static const char lrMount  = 'M'; // Internal Mount inventory
static const char lrMv     = 'm';
static const char lrRm     = 'r';
static const char lrRmdir  = 'D';
static const char lrSpace  = 'S'; // Internal Space inventory
static const char lrTOD    = 'T'; // Internal TOD

static const char *IArg;
static const char *iArg;

static XrdCnsLogRec *Alloc();

inline const char   *Data() {return (const char *)&Rec.Data;}

inline int           DLen() {return Rec.Hdr.dataLen;}

inline int           Done(int iPos) {return Rec.Hdr.Done[iPos];}

static XrdCnsLogRec *Get(char &lrType);

inline const char   *Lfn1() {return Rec.Data.lfn;}

inline const char   *Lfn1(int &Len) {Len=Rec.Hdr.lfn1Len; return Rec.Data.lfn;}

inline const char   *Lfn2() {return Rec.Data.lfn+Rec.Hdr.lfn1Len+1;}

inline mode_t        Mode() {return static_cast<mode_t>(Rec.Hdr.Mode);}

       void          Queue();

       void          Recycle();

inline char         *Record() {return (char *)&Rec;}

// setLfn1() must be called prior to calling setLfn2() or setData()
//
       int           setData(const char *dP1, const char *dP2=0);

       void          setDone(int iPos, char Val=1) {Rec.Hdr.Done[iPos] = Val;}

       int           setLen() {if (Rec.Hdr.lfn1Len)
                                  {Rec.Hdr.dataLen = FixDLen+Rec.Hdr.lfn1Len+1;
                                   if (Rec.Hdr.lfn2Len)
                                      Rec.Hdr.dataLen += Rec.Hdr.lfn2Len+1;
                                  } else Rec.Hdr.dataLen = 0;
                               return static_cast<int>(Rec.Hdr.dataLen);
                              }

inline int           setLfn1(const char *lfn)
                            {int n;
                             n = strlcpy(Rec.Data.lfn, lfn, MAXPATHLEN+1);
                             if (n > MAXPATHLEN) return 0;
                             Rec.Hdr.lfn1Len  = n;
                             return n;
                            }

inline int           setLfn2(const char *lfn)
                            {int n;
                             setSize(static_cast<long long>(Rec.Hdr.lfn1Len));
                             n = strlcpy(Rec.Data.lfn + Rec.Hdr.lfn1Len + 1,
                                        lfn, MAXPATHLEN+1);
                             if (n > MAXPATHLEN) return 0;
                             Rec.Hdr.lfn2Len  = n;
                             return n;
                            }

inline void          setMode(mode_t    Mode) {char Save = *Rec.Data.SorT;
                                              Rec.Hdr.Mode = Mode;
                                              sprintf(Rec.Data.Mode, "%03o",
                                              511 & static_cast<int>(Mode));
                                              *Rec.Data.SorT = Save;
                                             }

inline void          setMount(char mCode) {Rec.Data.Mount = mCode;}

inline void          setSize(long long Size) {char Save = Rec.Data.Mount;
                                              Rec.Hdr.Size = Size;
                                              sprintf(Rec.Data.SorT, "%12lld",
                                              (Size > 0 ? Size & 0x7fffffffffLL
                                                        : Size));
                                              Rec.Data.Mount = Save;
                                             }

inline void          setSpace(char sCode) {Rec.Data.Space = sCode;}

inline void          setTime(long TOD=time(0)){char Save = Rec.Data.Mount;
                                               sprintf(Rec.Data.SorT, "%12ld",
                                                       TOD-tBase);
                                               Rec.Data.Mount = Save;
                                              }

inline void          setType(const char evt) {Rec.Data.Type = evt;}

       int           setType(const char *evt);

inline long long     Size() {return Rec.Hdr.Size;}

inline char          Space(){return Rec.Data.Space;}

inline int           L1sz() {return Rec.Hdr.lfn1Len;}

inline int           L2sz() {return Rec.Hdr.lfn2Len;}

       char          Type() {return Rec.Data.Type;}

                     XrdCnsLogRec(const char rType=0) : Next(0)
                                    {memset(&Rec, 0, sizeof(Rec.Hdr));
                                     memset(&Rec.Data,' ', FixDLen);
                                     Rec.Data.Type = rType;
                                     if (rType == lrEOL || rType == lrTOD)
                                        {setTime();
                                         Rec.Hdr.dataLen=FixDLen;
                                         if (rType == lrTOD)
                                            {Rec.Data.lfn[0] = ' ';
                                             Rec.Data.lfn[1] = '\0';
                                             Rec.Hdr.lfn1Len = 1;
                                            }
                                        }
                                    }
                    ~XrdCnsLogRec() {}

private:

static XrdSysSemaphore qSem;
static XrdSysMutex     qMutex;
static XrdSysMutex     fMutex;
static XrdCnsLogRec   *freeRec;
static XrdCnsLogRec   *frstRec;
static XrdCnsLogRec   *lastRec;
static int             Running;

XrdCnsLogRec *Next;
      LogRec  Rec;
};
#endif
