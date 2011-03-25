#ifndef __FRMXLOCK_HH__
#define __FRMXLOCK_HH__
/******************************************************************************/
/*                                                                            */
/*                        X r d F r m X L o c k . h h                         */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOuc/XrdOucSxeq.hh"
  
class XrdFrmXLock
{
public:

static int Init(const char *aPath)
               {XrdOucSxeq mySxeq(".frmxeq", 0, aPath);
                return ((lkFD = mySxeq.Detach()) >= 0);
               }

           XrdFrmXLock() {XrdOucSxeq::Serialize(lkFD, 0);}
          ~XrdFrmXLock() {XrdOucSxeq::Release(lkFD);}

private:

static int lkFD;
};

#ifndef __FRMXLOCK_CC__
#define __FRMXLOCK_CC__
int XrdFrmXLock::lkFD = -1;
#endif

#endif
