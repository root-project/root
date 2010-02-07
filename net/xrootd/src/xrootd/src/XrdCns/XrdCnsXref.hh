#ifndef __XRDCnsXref_H_
#define __XRDCnsXref_H_
/******************************************************************************/
/*                                                                            */
/*                         X r d C n s X r e f . h h                          */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdCnsXref
{
public:


char           Add(const char *Key, char xref=0);

char          *Apply(int (*func)(const char *, char *, void *), void *Arg)
                    {return xTable.Apply(func, Arg);}

char           Default(const char *Dflt=0);

char          *Key (char  xref);

char           Find(const char *xref);

               XrdCnsXref(const char *Dflt=0, int MTProt=1);
              ~XrdCnsXref();

private:

int              availI();
int              c2i(char xCode);
XrdSysMutex      xMutex;
XrdOucHash<char> xTable;
static char     *xIndex;

static const int yTSize = '~'-'0'+1;
char            *yTable[yTSize];
int              availIdx;
int              isMT;
};
#endif
