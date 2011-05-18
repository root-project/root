#ifndef __CMS_ROUTING_H__
#define __CMS_ROUTING_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d C m s R o u t i n g . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include "XProtocol/YProtocol.hh"

class XrdCmsRouting
{
public:

enum {isInvalid = 0x00,
      isSync    = 0x02,
      Forward   = 0x04,
      noArgs    = 0x08,
      Delayable = 0x10,
      Repliable = 0x20,
      AsyncQ0   = 0x40,
      AsyncQ1   = 0x80
     };

struct      theRouting {int reqCode; int reqOpts;};

inline int  getRoute(int reqCode)
                      {return reqCode < XrdCms::kYR_MaxReq
                                      ? valVec[reqCode] : isInvalid;
                      }

            XrdCmsRouting(theRouting *initP)
                          {memset(valVec, 0, sizeof(valVec));
                           do {valVec[initP->reqCode] = initP->reqOpts;
                              } while((++initP)->reqCode);
                           }
            ~XrdCmsRouting() {}

private:
int          valVec[XrdCms::kYR_MaxReq];
};

/******************************************************************************/
/*                    X r d C m s R o u t e r   C l a s s                     */
/******************************************************************************/

class XrdCmsNode;
class XrdCmsRRData;
  
class XrdCmsRouter
{
public:

typedef const char *(XrdCmsNode::*NodeMethod_t)(XrdCmsRRData &);

struct  theRoute {int reqCode; const char *reqName; NodeMethod_t reqMeth;};

inline  NodeMethod_t getMethod(int Code)
                           {return Code < XrdCms::kYR_MaxReq
                                        ? methVec[Code] : (NodeMethod_t)0;
                           }

inline  const char  *getName(int Code)
                            {return Code < XrdCms::kYR_MaxReq && nameVec[Code]
                                         ? nameVec[Code] : "?";
                            }

              XrdCmsRouter(theRoute *initP)
                          {memset(methVec, 0, sizeof(methVec));
                           do {nameVec[initP->reqCode] = initP->reqName;
                               methVec[initP->reqCode] = initP->reqMeth;
                              } while((++initP)->reqCode);
                           }
             ~XrdCmsRouter() {}

private:

const  char         *nameVec [XrdCms::kYR_MaxReq];
       NodeMethod_t  methVec [XrdCms::kYR_MaxReq];
};

namespace XrdCms
{
extern XrdCmsRouter  Router;
extern XrdCmsRouting rdrVOps;
extern XrdCmsRouting rspVOps;
extern XrdCmsRouting srvVOps;
extern XrdCmsRouting supVOps;
}
#endif
