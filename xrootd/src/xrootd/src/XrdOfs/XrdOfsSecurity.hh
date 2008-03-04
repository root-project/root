#ifndef ___XrdOfsSECURITY_H___
#define ___XrdOfsSECURITY_H___
/******************************************************************************/
/*                                                                            */
/*                     X r d O f s S e c u r i t y . h h                      */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdAcc/XrdAccAuthorize.hh"

#define AUTHORIZE(usr, env, optype, action, pathp, edata) \
    if (usr && XrdOfsFS.Authorization \
    &&  !XrdOfsFS.Authorization->Access(usr, pathp, optype, env)) \
       {XrdOfsFS.Emsg(epname, edata, EACCES, action, pathp); return SFS_ERROR;}

#define AUTHORIZE2(usr,edata,opt1,act1,path1,env1,opt2,act2,path2,env2) \
       {AUTHORIZE(usr, env1, opt1, act1, path1, edata); \
        AUTHORIZE(usr, env2, opt2, act2, path2, edata); \
       }

#define OOIDENTENV(usr, env) \
    if (usr) {if (usr->name) env.Put(SEC_USER, usr->name); \
              if (usr->host) env.Put(SEC_HOST, usr->host);}
#endif
