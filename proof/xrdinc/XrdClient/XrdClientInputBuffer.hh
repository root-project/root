#ifndef XRC_INPUTBUFFER_H
#define XRC_INPUTBUFFER_H
/******************************************************************************/
/*                                                                            */
/*             X r d C l i e n t I n p u t B u f f e r . h h                  */
/*                                                                            */
/* Author: Fabrizio Furano (INFN Padova, 2004)                                */
/* Adapted from TXNetFile (root.cern.ch) originally done by                   */
/*  Alvise Dorigo, Fabrizio Furano                                            */
/*          INFN Padova, 2003                                                 */
/*                                                                            */
/* This file is part of the XRootD software suite.                            */
/*                                                                            */
/* XRootD is free software: you can redistribute it and/or modify it under    */
/* the terms of the GNU Lesser General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your     */
/* option) any later version.                                                 */
/*                                                                            */
/* XRootD is distributed in the hope that it will be useful, but WITHOUT      */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       */
/* License for more details.                                                  */
/*                                                                            */
/* You should have received a copy of the GNU Lesser General Public License   */
/* along with XRootD in a file called COPYING.LESSER (LGPL license) and file  */
/* COPYING (GPL license).  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                            */
/* The copyright holder's institutional names and contributor's names may not */
/* be used to endorse or promote products derived from this software without  */
/* specific prior written permission of the institution or contributor.       */
/******************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Buffer for incoming messages (responses)                             //
//  Handles the waiting (with timeout) for a message to come            //
//   belonging to a logical streamid                                    //
//  Multithread friendly                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdClient/XrdClientMessage.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysSemWait.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdClient/XrdClientVector.hh"

using namespace std;

class XrdClientInputBuffer {

private:

   XrdClientVector<XrdClientMessage*> fMsgQue;      // queue for incoming messages
   int                                fMsgIter;     // an iterator on it

   XrdSysRecMutex                        fMutex;       // mutex to protect data structures

   XrdOucHash<XrdSysSemWait>          fSyncobjRepo;
                                             // each streamid counts on a condition
                                             // variable to make the caller wait
                                             // until some data is available


   XrdSysSemWait                      *GetSyncObjOrMakeOne(int streamid);

   int             MsgForStreamidCnt(int streamid);

public:
   XrdClientInputBuffer();
  ~XrdClientInputBuffer();

   inline bool     IsMexEmpty() { return (MexSize() == 0); }
   inline bool     IsSemEmpty() { return (SemSize() == 0); }
   inline int      MexSize() { 
                       XrdSysMutexHelper mtx(fMutex);
                       return fMsgQue.GetSize();
                       }
   int             PutMsg(XrdClientMessage *msg);
   inline int      SemSize() {
                       XrdSysMutexHelper mtx(fMutex);
                       return fSyncobjRepo.Num();
                       }

   int             WipeStreamid(int streamid);

   XrdClientMessage      *GetMsg(int streamid, int secstimeout);
};
#endif
