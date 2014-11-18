#ifndef __XRDPSSCKS_HH__
#define __XRDPSSCKS_HH__
/******************************************************************************/
/*                                                                            */
/*                          X r d P s s C k s . h h                           */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <errno.h>

#include "XrdCks/XrdCks.hh"
#include "XrdCks/XrdCksData.hh"

class XrdSysError;

class XrdPssCks : public XrdCks
{
public:

virtual int         Calc( const char *Pfn, XrdCksData &Cks, int doSet=1)
                        {return Get(Pfn, Cks);}

virtual int         Del(  const char *Pfn, XrdCksData &Cks)
                       {return -ENOTSUP;}

virtual int         Get(  const char *Pfn, XrdCksData &Cks);

virtual int         Config(const char *Token, char *Line) {return 1;}

virtual int         Init(const char *ConfigFN, const char *DfltCalc=0);

virtual char       *List(const char *Pfn, char *Buff, int Blen, char Sep=' ')
                        {return 0;}

virtual const char *Name(int seqNum=0);

virtual int         Size( const char  *Name=0);

virtual int         Set(  const char *Pfn, XrdCksData &Cks, int myTime=0)
                       {return -ENOTSUP;}

virtual int         Ver(  const char *Pfn, XrdCksData &Cks);

           XrdPssCks(XrdSysError *erP);
virtual   ~XrdPssCks() {}

private:

struct csInfo
      {char          Name[XrdCksData::NameSize];
       int           Len;
                     csInfo() : Len(0) {memset(Name, 0, sizeof(Name));}
      };

csInfo *Find(const char *Name);

static const int csMax = 4;
csInfo           csTab[csMax];
int              csLast;
};
#endif
