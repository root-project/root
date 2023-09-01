#ifndef _XRC_URLINFO_H
#define _XRC_URLINFO_H
/******************************************************************************/
/*                                                                            */
/*                 X r d C l i e n t U r l I n f o . h h                      */
/*                                                                            */
/* Author: Fabrizio Furano (INFN Padova, 2004)                                */
/* Adapted from TXNetFile (root.cern.ch) originally done by                   */
/* Alvise Dorigo, Fabrizio Furano, INFN Padova, 2003                          */
/* Revised by G. Ganis, CERN,  June 2005                                      */
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
// Class handling information about an url                              //
// The purpose of this class is to allow:                               //
//   - parsing a string url into its components                         //
//   - reading/writing the single components                            //
//   - reading the modified full url                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdOuc/XrdOucString.hh"

//
// The information an url may contain
// Plus utilities for parsing and rebuilding an url
//

class XrdClientUrlInfo {
 public:
   XrdOucString Proto;
   XrdOucString Passwd;
   XrdOucString User;
   XrdOucString Host;
   int Port;
   XrdOucString HostAddr;
   XrdOucString HostWPort;
   XrdOucString File;

   void Clear();
   void TakeUrl(XrdOucString url);
   XrdOucString GetUrl();

   XrdClientUrlInfo(const char *url);
   XrdClientUrlInfo(const XrdOucString &url);
   XrdClientUrlInfo(const XrdClientUrlInfo &url);
   XrdClientUrlInfo();

   void SetAddrFromHost();

   inline bool IsValid() { return (Port >= 0); }

   XrdClientUrlInfo &operator=(const XrdOucString &url);
   XrdClientUrlInfo &operator=(const XrdClientUrlInfo &url);
};
#endif
