#ifndef __CMS_SELECT_HH
#define __CMS_SELECT_HH
/******************************************************************************/
/*                                                                            */
/*                       X r d C m s S e l e c t . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include "XrdCms/XrdCmsKey.hh"

/******************************************************************************/
/*                    C l a s s   X r d C m s S e l e c t                     */
/******************************************************************************/

class XrdCmsRRQInfo;
  
class XrdCmsSelect
{
public:
XrdCmsKey      Path;    //  In: Path to select or lookup in the cache
XrdCmsRRQInfo *InfoP;   //  In: Fast redirect routing
SMask_t        nmask;   //  In: Nodes to avoid
SMask_t        smask;   // Out: Nodes selected
struct iovec  *iovP;    //  In: Prepare notification I/O vector
int            iovN;    //  In: Prepare notification I/O vector count
int            Opts;    //  In: One or more of the following enums

enum {Write   = 0x0001, // File will be open in write mode     (select & cache)
      NewFile = 0x0002, // File will be created may not exist  (select)
      Online  = 0x0004, // Only consider online files          (select & prep)
      Trunc   = 0x0008, // File will be truncated              (Select   only)
      Create  = 0x000A, // Create file, truncate if exists
      Defer   = 0x0010, // Do not select a server now          (prep     only)
      Peers   = 0x0020, // Peer clusters may be selected       (select   only)
      Refresh = 0x0040, // Cache should be refreshed           (all)
      Asap    = 0x0080, // Respond as soon as possible         (locate   only)
      noBind  = 0x0100, // Do not new bind file to a server    (select   only)
      isMeta  = 0x0200, // Only inode information being changed(select   only)
      Freshen = 0x0400, // Freshen access times                (prep     only)
      Replica = 0x0800, // File will be replicated (w/ Create) (select   only)
      Advisory= 0x4000, // Cache A/D is advisory (no delay)    (have   & cache)
      Pending = 0x8000  // File being staged                   (have   & cache)
     };

struct {SMask_t wf;     // Out: Writable locations
        SMask_t hf;     // Out: Existing locations
        SMask_t pf;     // Out: Pending  locations
        SMask_t bf;     // Out: Bounced  locations
       }        Vec;

struct {int  Port;      // Out: Target node port number
        char Data[256]; // Out: Target node or error message
        int  DLen;      // Out: Length of Data including null byte
       }     Resp;

             XrdCmsSelect(int opts=0, char *thePath=0, int thePLen=0)
                         : Path(thePath,thePLen), smask(0), Opts(opts)
                         {Resp.Port = 0; *Resp.Data = '\0'; Resp.DLen = 0;}
            ~XrdCmsSelect() {}
};

/******************************************************************************/
/*                  C l a s s   X r d C m s S e l e c t e d                   */
/******************************************************************************/
  
class XrdCmsSelected   // Argument to List() after select or locate
{
public:

XrdCmsSelected *next;
char           *Name;
SMask_t         Mask;
int             Id;
unsigned int    IPAddr;
int             Port;
int             IPV6Len;  // 12345678901234567890123456
char            IPV6[28]; // [::123.123.123.123]:123456
int             Load;
int             Util;
int             Free;
int             RefTotA;
int             RefTotR;
int             Status;      // One of the following

enum           {Disable = 0x0001,
                NoStage = 0x0002,
                Offline = 0x0004,
                Suspend = 0x0008,
                NoSpace = 0x0020,
                isRW    = 0x0040,
                Reservd = 0x0080,
                isMangr = 0x0100,
                isPeer  = 0x0200,
                isProxy = 0x0400,
                noServr = 0x0700
               };

               XrdCmsSelected(const char *sname, XrdCmsSelected *np=0)
                         {Name = (sname ? strdup(sname) : 0); next=np;}

              ~XrdCmsSelected() {if (Name) free(Name);}
};
#endif
