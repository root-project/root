#ifndef __CMS_PROTOCOL_H__
#define __CMS_PROTOCOL_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d C m s P r o t o c o l . h h                      */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include "Xrd/XrdProtocol.hh"
#include "XrdCms/XrdCmsParser.hh"
#include "XrdCms/XrdCmsTypes.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdLink;
class XrdCmsNode;
class XrdCmsRRData;
class XrdCmsRouting;

class XrdCmsProtocol : public XrdProtocol
{
friend class XrdCmsJob;
public:

static XrdCmsProtocol *Alloc(const char *theRole = "",
                             const char *theMan  = 0, int thePort=0);

       void            DoIt();

       int             Execute(XrdCmsRRData &Data);

       XrdProtocol    *Match(XrdLink *lp);   // Upon    accept

       int             Process(XrdLink *lp); // Initial entry

       void            Recycle(XrdLink *lp, int consec, const char *reason);

       int             Stats(char *buff, int blen, int do_sync=0) {return 0;}

              XrdCmsProtocol() : XrdProtocol("cms protocol handler"),
                                 ProtLink(0), myRole("?"), myNode(0), RSlot(0)
                               {}
             ~XrdCmsProtocol() {}

private:

XrdCmsRouting  *Admit();
XrdCmsRouting  *Admit_DataServer(int);
XrdCmsRouting  *Admit_Redirector(int);
XrdCmsRouting  *Admit_Supervisor(int);
SMask_t         AddPath(XrdCmsNode *nP, const char *pType, const char *Path);
int             Authenticate();
void            ConfigCheck(unsigned char *theConfig);
enum Bearing    {isDown, isLateral, isUp};
const char     *Dispatch(Bearing cDir, int maxWait, int maxTries);
XrdCmsRouting  *Login_Failed(const char *Reason);
void            Pander(const char *manager, int mport);
void            Reissue(XrdCmsRRData &Data);
void            Reply_Delay(XrdCmsRRData &Data, kXR_unt32 theDelay);
void            Reply_Error(XrdCmsRRData &Data, int ecode, const char *etext);

static XrdSysMutex     ProtMutex;
static XrdCmsProtocol *ProtStack;
static XrdCmsParser    ProtArgs;
       XrdCmsProtocol *ProtLink;

       XrdCmsRouting  *Routing;   // Request routing for this instance

static const int       maxReqSize = 16384;

       XrdLink        *Link;
static int             readWait;
const  char           *myRole;
const  char           *myMan;
       int             myManPort;
       XrdCmsNode     *myNode;
       short           RSlot;      // True only for redirectors
       char            loggedIn;   // True of login succeeded
};
#endif
