#ifndef __CMS_CLUSTER__H
#define __CMS_CLUSTER__H
/******************************************************************************/
/*                                                                            */
/*                      X r d C m s C l u s t e r . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdlib.h>
#include <string.h>
#include <strings.h>
  
#include "XrdCms/XrdCmsTypes.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdLink;
class XrdCmsDrop;
class XrdCmsNode;
class XrdCmsSelect;
namespace XrdCms
{
struct CmsRRHdr;
}
 
/******************************************************************************/
/*                          O p t i o n   F l a g s                           */
/******************************************************************************/

namespace XrdCms
{

// Flags passed to Add()
//
static const int CMS_noStage =  1;
static const int CMS_Suspend =  2;
static const int CMS_Perm    =  4;
static const int CMS_isMan   =  8;
static const int CMS_Lost    = 16;
static const int CMS_isPeer  = 32;
static const int CMS_isProxy = 64;
static const int CMS_noSpace =128;

// Class passed to Space()
//
class SpaceData
{
public:

long long Total;    // Total space
int       wMinF;    // Free space minimum to select wFree node
int       wFree;    // Free space for nodes providing r/w access
int       wNum;     // Number of      nodes providing r/w access
int       wUtil;    // Average utilization
int       sFree;    // Free space for nodes providing staging
int       sNum;     // Number of      nodes providing staging
int       sUtil;    // Average utilization

          SpaceData() : Total(0), wMinF(0),
                        wFree(0), wNum(0), wUtil(0),
                        sFree(0), sNum(0), sUtil(0) {}
         ~SpaceData() {}
};
}
  
/******************************************************************************/
/*                   C l a s s   X r d C m s C l u s t e r                    */
/******************************************************************************/
  
// This a single-instance global class
//
class XrdCmsSelected;

class XrdCmsCluster
{
public:
friend class XrdCmsDrop;

int             NodeCnt;       // Number of active nodes

// Called to add a new node to the cluster. Status values are defined above.
//
XrdCmsNode     *Add(XrdLink *lp, int dport, int Status,
                    int sport, const char *theNID);

// Sends a message to all nodes matching smask (three forms for convenience)
//
SMask_t         Broadcast(SMask_t, const struct iovec *, int, int tot=0);

SMask_t         Broadcast(SMask_t smask, XrdCms::CmsRRHdr &Hdr,
                          char *Data,    int Dlen=0);

SMask_t         Broadcast(SMask_t smask, XrdCms::CmsRRHdr &Hdr,
                          void *Data,    int Dlen);

// Returns the node mask matching the given IP address
//
SMask_t         getMask(unsigned int IPv4adr);

// Returns the node mask matching the given cluster ID
//
SMask_t         getMask(const char *Cid);

// Extracts out node information. Opts are one or more of CmsLSOpts
//
enum            CmsLSOpts {LS_Best = 0x0001, LS_All  = 0x0002,
                           LS_IPO  = 0x0004, LS_IPV6 = 0x0008};

XrdCmsSelected *List(SMask_t mask, CmsLSOpts opts);

// Returns the location of a file
//
int             Locate(XrdCmsSelect &Sel);

// Always run as a separate thread to monitor subscribed node performance
//
void           *MonPerf();

// Alwats run as a separate thread to maintain the node reference count
//
void           *MonRefs();

// Called to remove a node from the cluster
//
void            Remove(const char *reason, XrdCmsNode *theNode, int immed=0);

// Called to reset the node reference counts for nodes matching smask
//
void            ResetRef(SMask_t smask);

// Called to select the best possible node to serve a file (two forms)
//
int             Select(XrdCmsSelect &Sel);

int             Select(int isrw, SMask_t pmask, int &port, 
                       char *hbuff, int &hlen);

// Called to get cluster space (for managers and supervisors only)
//
void            Space(XrdCms::SpaceData &sData, SMask_t smask);

// Called to returns statistics (not really implemented)
//
int             Stats(char *bfr, int bln);

                XrdCmsCluster();
               ~XrdCmsCluster() {} // This object should never be deleted

private:
int         Assign(const char *Cid);
XrdCmsNode *calcDelay(int nump, int numd, int numf, int numo,
                      int nums, int &delay, const char **reason);
int         Drop(int sent, int sinst, XrdCmsDrop *djp=0);
void        Record(char *path, const char *reason);
int         Multiple(SMask_t mVec);
enum        {eExists, eDups, eROfs, eNoRep}; // Passed to SelFail
int         SelFail(XrdCmsSelect &Sel, int rc);
int         SelNode(XrdCmsSelect &Sel, SMask_t pmask, SMask_t amask);
XrdCmsNode *SelbyCost(SMask_t, int &, int &, const char **, int);
XrdCmsNode *SelbyLoad(SMask_t, int &, int &, const char **, int);
XrdCmsNode *SelbyRef (SMask_t, int &, int &, const char **, int);
void        sendAList(XrdLink *lp);
void        setAltMan(int snum, unsigned int ipaddr, int port);


static const  int AltSize = 24; // Number of IP:Port characters per entry

XrdSysMutex   cidMutex;         // Protects to cid list
XrdOucTList  *cidFirst;         // Cluster ID to cluster number map

XrdSysMutex   XXMutex;          // Protects cluster summary state variables
XrdSysMutex   STMutex;          // Protects all node information  variables
XrdCmsNode   *NodeTab[STMax];   // Current  set of nodes

int           STHi;             // NodeTab high watermark
int           SelAcnt;          // Total number of r/w selections
int           SelRcnt;          // Total number of r/o selections
int           doReset;          // Must send reset event to Managers[resetMask]

// The following is a list of IP:Port tokens that identify supervisor nodes.
// The information is sent via the try request to redirect nodes; as needed.
// The list is alays rotated by one entry each time it is sent.
//
char          AltMans[STMax*AltSize]; // ||123.123.123.123:12345|| = 21
char         *AltMend;
int           AltMent;

// The foloowing three variables are protected by the STMutex
//
SMask_t       resetMask;        // Nodes to receive a reset event
SMask_t       peerHost;         // Nodes that are acting as peers
SMask_t       peerMask;         // Always ~peerHost
};

namespace XrdCms
{
extern    XrdCmsCluster Cluster;
}
#endif
