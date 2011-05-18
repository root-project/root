#ifndef __XRDCMSPARSER_H__
#define __XRDCMSPARSER_H__
/******************************************************************************/
/*                                                                            */
/*                       X r d C m s P a r s e r . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XProtocol/YProtocol.hh"

#include "XrdCms/XrdCmsRRData.hh"
#include "XrdOuc/XrdOucPup.hh"

/******************************************************************************/
/*                    C l a s s   X r d C m s P a r s e r                     */
/******************************************************************************/

class XrdOucErrInfo;
  
class XrdCmsParser
{
public:

static int            Decode(const char *Man, XrdCms::CmsRRHdr &hdr, 
                                   char *data, int dlen, XrdOucErrInfo *eInfo);

static int            mapError(const char *ecode);

static int            mapError(int ecode);

static int            Pack(int rnum, struct iovec *iovP, struct iovec *iovE,
                           char *Base, char *Work);

inline int            Parse(XrdCms::CmsLoginData *Data, 
                            const char *Aps, const char *Apt)
                           {Data->SID = Data->Paths = 0;
                            return Pup.Unpack(Aps,Apt,vecArgs[XrdCms::kYR_login],
                                              (char *)Data);
                           }

inline int            Parse(int rnum, const char *Aps, const char *Apt, 
                            XrdCmsRRData *Data)
                           {Data->Opaque = Data->Opaque2 = Data->Path = 0;
                            return rnum < XrdCms::kYR_MaxReq 
                                   && vecArgs[rnum] != 0
                                   && Pup.Unpack(Aps, Apt,
                                      vecArgs[rnum], (char *)Data);
                           }

static XrdOucPup      Pup;

static XrdOucPupArgs *PupArgs(int rnum)
                             {return rnum < XrdCms::kYR_MaxReq ? vecArgs[rnum] : 0;}

       XrdCmsParser();
      ~XrdCmsParser() {}

private:

static const char   **PupNVec;
static XrdOucPupNames PupName;

static XrdOucPupArgs  fwdArgA[];  // chmod | mkdir | mkpath | trunc
static XrdOucPupArgs  fwdArgB[];  // mv
static XrdOucPupArgs  fwdArgC[];  // rm | rmdir
static XrdOucPupArgs  locArgs[];  // locate | select
static XrdOucPupArgs  padArgs[];  // prepadd
static XrdOucPupArgs  pdlArgs[];  // prepdel
static XrdOucPupArgs  avlArgs[];  // avail
static XrdOucPupArgs  pthArgs[];  // statfs | try
static XrdOucPupArgs  lodArgs[];  // load
static XrdOucPupArgs  logArgs[];  // login

static XrdOucPupArgs *vecArgs[XrdCms::kYR_MaxReq];
};

namespace XrdCms
{
extern    XrdCmsParser Parser;
}
#endif
