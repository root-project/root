#ifndef __ACC_PRIVS__
#define __ACC_PRIVS__
/******************************************************************************/
/*                                                                            */
/*                        X r d A c c P r i v s . h h                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

/******************************************************************************/
/*                           X r d A c c P r i v s                            */
/******************************************************************************/
  
// Recognized privileges
//
enum XrdAccPrivs {XrdAccPriv_All    = 0x07f,
                  XrdAccPriv_Chmod  = 0x063,  // Insert + Open r/w + Delete
                  XrdAccPriv_Chown  = 0x063,  // Insert + Open r/w + Delete
                  XrdAccPriv_Create = 0x062,  // Insert + Open r/w
                  XrdAccPriv_Delete = 0x001,
                  XrdAccPriv_Insert = 0x002,
                  XrdAccPriv_Lock   = 0x004,
                  XrdAccPriv_Mkdir  = 0x002,  // Insert
                  XrdAccPriv_Lookup = 0x008,
                  XrdAccPriv_Rename = 0x010,
                  XrdAccPriv_Read   = 0x020,
                  XrdAccPriv_Readdir= 0x020,
                  XrdAccPriv_Write  = 0x040,
                  XrdAccPriv_Update = 0x060,
                  XrdAccPriv_None   = 0x000
                 };
  
/******************************************************************************/
/*                        X r d A c c P r i v S p e c                         */
/******************************************************************************/
  
// The following are the 1-letter privileges that we support.
//
enum XrdAccPrivSpec {   All_Priv = 'a',
                     Delete_Priv = 'd',
                     Insert_Priv = 'i',
                       Lock_Priv = 'k',
                     Lookup_Priv = 'l',
                     Rename_Priv = 'n',
                       Read_Priv = 'r',
                      Write_Priv = 'w',
                        Neg_Priv = '-'
                    };

/******************************************************************************/
/*                        X r d A c c P r i v C a p s                         */
/******************************************************************************/

struct XrdAccPrivCaps {XrdAccPrivs pprivs;     // Positive privileges
                       XrdAccPrivs nprivs;     // Negative privileges

                       XrdAccPrivCaps() {pprivs = XrdAccPriv_None;
                                         nprivs = XrdAccPriv_None;
                                        }
                      ~XrdAccPrivCaps() {}

                      };
#endif
