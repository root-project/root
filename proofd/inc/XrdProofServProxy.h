// @(#)root/proofd:$Name:  $:$Id: XrdProofServProxy.h,v 1.5 2006/08/05 20:04:47 brun Exp $
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofServProxy
#define ROOT_XrdProofServProxy

#include <string.h>
#include <unistd.h>
#include <sys/uio.h>

#include <list>
#include <map>
#include <vector>

#include "Xrd/XrdLink.hh"
#include "XrdOuc/XrdOucPthread.hh"
#include "XrdOuc/XrdOucSemWait.hh"

#include "XrdProofdResponse.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdSrvBuffer                                                         //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// The following structure is used to store buffers to be sent or       //
// received from clients                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class XrdSrvBuffer {
public:
   int   fSize;
   char *fBuff;

   XrdSrvBuffer(char *bp=0, int sz=0, bool dup=0) {
      if (dup && bp && sz > 0) {
         fMembuf = (char *)malloc(sz);
         if (fMembuf) {
            memcpy(fMembuf, bp, sz);
            fBuff = fMembuf;
            fSize = sz;
         }
      } else {
         fBuff = fMembuf = bp;
         fSize = sz;
      }}
   ~XrdSrvBuffer() {if (fMembuf) free(fMembuf);}

private:
   char *fMembuf;
};


class XrdProofdProtocol;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientID                                                          //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Mapping of clients and stream IDs                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class XrdClientID {
public:
   XrdProofdProtocol *fP;
   unsigned short     fSid;

   XrdClientID(XrdProofdProtocol *pt = 0, unsigned short id = 0)
            { fP = pt; fSid = id; }
   ~XrdClientID() { }

   bool   IsValid() const { return (fP != 0); }
   void   Reset() { fP = 0; fSid = 0; }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofServProxy                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// This class represent an instance of TProofServ                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#define kXPROOFSRVTAGMAX   64
#define kXPROOFSRVALIASMAX 256

class XrdProofWorker;
class XrdNet;

class XrdProofServProxy
{

friend class XrdProofdProtocol;

public:
   XrdProofServProxy();
   ~XrdProofServProxy();

   inline const char  *Alias() const { return fAlias; }
   inline const char  *Client() const { return fClient; }
   inline const char  *Fileout() const { return fFileout; }
   inline bool         IsParent(XrdProofdProtocol *p) const
                                 { return (fParent && fParent->fP == p); }
   inline bool         Match(short int id) const { return (id == fID); }
   inline XrdOucMutex *Mutex() { return &fMutex; }
   inline const char  *Ordinal() const { return (const char *)fOrdinal; }
   inline int          SrvID() const { return fSrvID; }
   inline int          SrvType() const { return fSrvType; }
   inline void         SetID(short int id) { fID = id;}
   inline void         SetSrv(int id) { fSrvID = id; }
   inline void         SetSrvType(int id) { fSrvType = id; }
   inline void         SetStatus(int st) { fStatus = st; }
   inline void         SetValid(bool valid = 1) { fIsValid = valid; }
   inline int          Status() const { return fStatus;}
   inline const char  *Tag() const { return fTag; }

   XrdClientID        *GetClientID(int cid);
   int                 GetFreeID();
   int                 GetNClients();

   int                 GetNWorkers() { return (int) fWorkers.size(); }
   void                AddWorker(XrdProofWorker *w) { fWorkers.push_back(w); }
   void                RemoveWorker(XrdProofWorker *w) { fWorkers.remove(w); }

   void                SetAlias(const char *a, int l = 0)
                          { XrdProofServProxy::SetCharValue(&fAlias, a, l); }
   void                SetClient(const char *c, int l = 0)
                          { XrdProofServProxy::SetCharValue(&fClient, c, l); }
   void                SetFileout(const char *f, int l = 0)
                          { XrdProofServProxy::SetCharValue(&fFileout, f, l); }
   void                SetOrdinal(const char *o, int l = 0)
                          { XrdProofServProxy::SetCharValue(&fOrdinal, o, l); }
   void                SetTag(const char *t, int l = 0)
                          { XrdProofServProxy::SetCharValue(&fTag, t, l); }

   bool                IsValid() const { return fIsValid; }
   const char         *StatusAsString() const;

   void                Reset();

 private:

   XrdOucRecMutex            fMutex;
   XrdLink                  *fLink;      // Link to proofsrv
   XrdProofdResponse         fProofSrv;  // Utility to talk to proofsrv

   XrdClientID              *fParent;    // Parent creating this session
   std::vector<XrdClientID *> fClients;  // Attached clients stream ids
   std::list<XrdProofWorker *> fWorkers; // Workers assigned to the session

   XrdOucSemWait            *fPingSem;   // To sychronize ping requests

   XrdSrvBuffer             *fQueryNum;  // Msg with sequential number of currebt query
   XrdSrvBuffer             *fStartMsg;  // Msg with start processing info

   int                       fStatus;
   int                       fSrvID;     // Srv process ID
   int                       fSrvType;
   short int                 fID;
   char                      fProtVer;
   char                     *fFileout;

   bool                      fIsValid;   // Validity flag

   char                     *fAlias;     // Session alias
   char                     *fClient;    // Client name
   char                     *fTag;       // Session unique tag
   char                     *fOrdinal;   // Session ordinal number

   void                      ClearWorkers();

   static void               SetCharValue(char **carr, const char *v, int len = 0);
};
#endif
