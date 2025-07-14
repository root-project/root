/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_MessageTypes
#define ROOT_MessageTypes


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MessageTypes                                                         //
//                                                                      //
// System predefined message types. Message types are constants that    //
// indicate what kind of message it is. Make sure your own message      //
// types don't clash whith the ones defined in this file. ROOT reserves //
// all message ids between 0 - 10000. Make sure your message            //
// id < 200000000.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

enum EMessageTypes {
   kMESS_ZIP             = 0x20000000,   //OR with kMESS_ZIP to compress message
   kMESS_ACK             = 0x10000000,   //OR with kMESS_ACK to force each
                                         //message to be acknowledged
   kMESS_ANY             = 0,            //generic message type
   kMESS_OK              = 1,            //everything OK
   kMESS_NOTOK           = 2,            //things are NOT OK
   kMESS_STRING          = 3,            //string follows
   kMESS_OBJECT          = 4,            //object follows
   kMESS_CINT            = 5,            //cint command follows
   kMESS_STREAMERINFO    = 6,            //TStreamerInfo object follows
   kMESS_PROCESSID       = 7,            //TProcessID object follows

   //---- ROOTD message opcodes (2000 - 2099)
   kROOTD_USER           = 2000,         //user id follows
   kROOTD_PASS           = 2001,         //passwd follows
   kROOTD_AUTH           = 2002,         //authorization status (to client)
   kROOTD_FSTAT          = 2003,         //filename follows
   kROOTD_OPEN           = 2004,         //filename follows + mode
   kROOTD_PUT            = 2005,         //offset, number of bytes and buffer
   kROOTD_GET            = 2006,         //offset, number of bytes
   kROOTD_FLUSH          = 2007,         //flush file
   kROOTD_CLOSE          = 2008,         //close file
   kROOTD_STAT           = 2009,         //return rootd statistics
   kROOTD_ACK            = 2010,         //acknowledgement (all OK)
   kROOTD_ERR            = 2011,         //error code and message follow
   kROOTD_PROTOCOL       = 2012,         //returns rootd protocol
   kROOTD_PUTFILE        = 2020,         //store file
   kROOTD_GETFILE        = 2021,         //retrieve file
   kROOTD_CHDIR          = 2022,         //change directory
   kROOTD_MKDIR          = 2023,         //make directory
   kROOTD_RMDIR          = 2024,         //delete directory
   kROOTD_LSDIR          = 2025,         //list directory
   kROOTD_PWD            = 2026,         //pwd
   kROOTD_MV             = 2027,         //rename file
   kROOTD_RM             = 2028,         //delete file
   kROOTD_CHMOD          = 2029,         //change permission
   kROOTD_PROTOCOL2      = 2031,         //client proto follows, returns rootd proto
   kROOTD_BYE            = 2032,         //terminate rootd
   kROOTD_CLEANUP        = 2034,         //cleanup things
   kROOTD_NEGOTIA        = 2037,         //negotiation follows
   kROOTD_RSAKEY         = 2038,         //RSA public key exchange
   kROOTD_ENCRYPT        = 2039,         //an encrypted message follows
   kROOTD_OPENDIR        = 2040,         //open directory
   kROOTD_FREEDIR        = 2041,         //free directory
   kROOTD_DIRENTRY       = 2042,         //get directory entry
   kROOTD_ACCESS         = 2043,         //test Access
   kROOTD_GETS           = 2044          //multiple offset, number of byte pairs
};

#endif
