#ifndef __FRCUTILS__HH
#define __FRCUTILS__HH
/******************************************************************************/
/*                                                                            */
/*                        X r d F r c U t i l s . h h                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdlib.h>
#include <time.h>

#include "XrdFrc/XrdFrcRequest.hh"

class  XrdFrcXAttrPin;

class XrdFrcUtils
{
public:

static       char  Ask(char dflt, const char *Msg1, const char *Msg2="",
                                  const char *Msg3="");

static       int   chkURL(const char *Url);

static       char *makePath(const char *iName, const char *Path, int Mode);

static       char *makeQDir(const char *Path, int Mode);

static       int   MapM2O(const char *Nop, const char *Pop);

static       int   MapR2Q(char Opc, int *Flags=0);

static       int   MapV2I(const char *Opc, XrdFrcRequest::Item &ICode);

static       int   Unique(const char *lkfn, const char *myProg);

static       int   updtCpy(const char *Pfn, int Adj);

static       int   Utime(const char *Path, time_t tVal);

                   XrdFrcUtils() {}
                  ~XrdFrcUtils() {}
private:
};
#endif
