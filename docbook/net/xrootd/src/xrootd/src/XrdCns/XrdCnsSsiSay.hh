#ifndef __XRDCnsSsiSay_H_
#define __XRDCnsSsiSay_H_
/******************************************************************************/
/*                                                                            */
/*                          X r d C n s S a y . h h                           */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdSys/XrdSysError.hh"

class XrdCnsSsiSay
{
public:

inline void M(const char *txt1,   const char *txt2=0, const char *txt3=0,
              const char *txt4=0, const char *txt5=0)
             {eDest->Say("cns_ssi: ", txt1, txt2, txt3, txt4, txt5);}

inline void V(const char *txt1,   const char *txt2=0, const char *txt3=0,
              const char *txt4=0, const char *txt5=0)
             {if (Verbose) M(txt1, txt2, txt3, txt4, txt5);}

inline void setV(int val) {Verbose = val;}

       XrdCnsSsiSay(XrdSysError *erp) : eDest(erp), Verbose(0) {}
      ~XrdCnsSsiSay() {}

private:

XrdSysError *eDest;
int          Verbose;
};
#endif
