/* @(#)root/base:$Name:  $:$Id: MessageTypes.h,v 1.9 2002/02/12 17:53:18 rdm Exp $ */

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
   kMESS_ACK             = 0x10000000,   //OR with kMESS_ACK to force each
                                         //message to be acknowledged
   kMESS_ANY             = 0,            //generic message type
   kMESS_OK,                             //everything OK
   kMESS_NOTOK,                          //things are NOT OK
   kMESS_STRING,                         //string follows
   kMESS_OBJECT,                         //object follows
   kMESS_CINT,                           //cint command follows

   //---- PROOF message opcodes (1000 - 1999)
   kPROOF_GROUPVIEW       = 1000,        //groupview follows
   kPROOF_STOP,                          //stop proof server
   kPROOF_FATAL,                         //server got fatal error and died
   kPROOF_LOGLEVEL,                      //loglevel follows
   kPROOF_LOGFILE,                       //log file length and content follows
   kPROOF_LOGDONE,                       //log file received, status follows
   kPROOF_STATUS,                        //print status of slave
   kPROOF_PING,                          //ping slave
   kPROOF_PRINT,                         //ask master to print config
   kPROOF_RESET,                         //reset slave
   kPROOF_GETOBJECT,                     //ask for object with given name
   kPROOF_TREEDRAW,                      //tree draw command follows
   kPROOF_GETPACKET,                     //ask for next packet
   kPROOF_LIMITS,                        //ask for histogram limits
   kPROOF_CHECKFILE,                     //filename and md5 follows
   kPROOF_SENDFILE,                      //filename, length and file follows
   kPROOF_PARALLEL,                      //number of parallel slaves follows
   kPROOF_OPENFILE,                      //type of file, name and option follows
   kPROOF_CLOSEFILE,                     //name of file follows
   kPROOF_PROCESS,                       //process events, DSet and input list follow
   kPROOF_OUTPUTLIST,                    //return the output list from Process()

   //---- ROOTD message opcodes (2000 - 2099)
   kROOTD_USER             = 2000,       //user id follows
   kROOTD_PASS,                          //passwd follows
   kROOTD_AUTH,                          //authorization status (to client)
   kROOTD_FSTAT,                         //filename follows
   kROOTD_OPEN,                          //filename follows + mode
   kROOTD_PUT,                           //offset, number of bytes and buffer
   kROOTD_GET,                           //offset, number of bytes
   kROOTD_FLUSH,                         //flush file
   kROOTD_CLOSE,                         //close file
   kROOTD_STAT,                          //return rootd statistics
   kROOTD_ACK,                           //acknowledgement (all OK)
   kROOTD_ERR,                           //error code and message follow
   kROOTD_PROTOCOL,                      //return rootd protocol id
   kROOTD_SRPUSER,                       //user id for SRP authentication follows
   kROOTD_SRPN,                          //SRP n follows
   kROOTD_SRPG,                          //SRP g follows
   kROOTD_SRPSALT,                       //SRP salt follows
   kROOTD_SRPA,                          //SRP a follows
   kROOTD_SRPB,                          //SRP b follows
   kROOTD_SRPRESPONSE,                   //SRP final response
   kROOTD_PUTFILE,                       //store file
   kROOTD_GETFILE,                       //retrieve file
   kROOTD_CHDIR,                         //change directory
   kROOTD_MKDIR,                         //make directory
   kROOTD_RMDIR,                         //delete directory
   kROOTD_LSDIR,                         //list directory
   kROOTD_PWD,                           //pwd
   kROOTD_MV,                            //rename file
   kROOTD_RM,                            //delete file
   kROOTD_CHMOD                          //change permission
};

#endif
