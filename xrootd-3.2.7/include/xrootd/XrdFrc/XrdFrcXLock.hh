#ifndef __FRCXLOCK_HH__
#define __FRCXLOCK_HH__
/******************************************************************************/
/*                                                                            */
/*                        X r d F r c X L o c k . h h                         */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOuc/XrdOucSxeq.hh"
  
class XrdFrcXLock
{
public:

static int Init(const char *aPath)
               {XrdOucSxeq mySxeq(".frmxeq", 0, aPath);
                return ((lkFD = mySxeq.Detach()) >= 0);
               }

           XrdFrcXLock() {XrdOucSxeq::Serialize(lkFD, 0);}
          ~XrdFrcXLock() {XrdOucSxeq::Release(lkFD);}

private:

static int lkFD;
};

#ifndef __FRCXLOCK_CC__
#define __FRCXLOCK_CC__
int XrdFrcXLock::lkFD = -1;
#endif

#endif
