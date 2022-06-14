#ifndef XRC_UNSOLMSG_H
#define XRC_UNSOLMSG_H
/******************************************************************************/
/*                                                                            */
/*                X r d C l i e n t U n s o l M s g . h h                     */
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
// Base classes for unsolicited msg senders/receivers                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class XrdClientMessage;
class XrdClientUnsolMsgSender;

// The processing result for an unsolicited response
enum UnsolRespProcResult {
   kUNSOL_CONTINUE = 0, // Dispatching must continue to other interested handlers
   kUNSOL_KEEP,         // Dispatching ended, but stream still alive (must keep the SID)
   kUNSOL_DISPOSE       // Dispatching ended, stream no more to be used
};

// Handler

class XrdClientAbsUnsolMsgHandler {
 public:
  
   virtual ~XrdClientAbsUnsolMsgHandler() { }
   // To be called when an unsolicited response arrives from the lower layers
   virtual UnsolRespProcResult ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *sender, 
				      XrdClientMessage *unsolmsg) = 0;

};

// Sender

class XrdClientUnsolMsgSender {

 public:

   virtual ~XrdClientUnsolMsgSender() { }

   // The upper level handler for unsolicited responses
   XrdClientAbsUnsolMsgHandler *UnsolicitedMsgHandler;

   inline UnsolRespProcResult SendUnsolicitedMsg(XrdClientUnsolMsgSender *sender, XrdClientMessage *unsolmsg) {
      // We simply send the event
      if (UnsolicitedMsgHandler)
	 return (UnsolicitedMsgHandler->ProcessUnsolicitedMsg(sender, unsolmsg));

      return kUNSOL_CONTINUE;
   }

   inline XrdClientUnsolMsgSender() { UnsolicitedMsgHandler = 0; }
};
#endif
