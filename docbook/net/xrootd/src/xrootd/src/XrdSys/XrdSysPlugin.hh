#ifndef __XRDSYSPLUGIN__
#define __XRDSYSPLUGIN__
/******************************************************************************/
/*                                                                            */
/*                       X r d S y s P l u g i n . h h                        */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

class XrdSysError;

class XrdSysPlugin
{
public:

void *getPlugin(const char *pname, int errok=0);
void *getPlugin(const char *pname, int errok, bool global);

      XrdSysPlugin(XrdSysError *erp, const char *path)
                  {eDest = erp; libPath = path; libHandle = 0;}
     ~XrdSysPlugin();

private:

XrdSysError *eDest;
const char  *libPath;
void        *libHandle;
};
#endif
