/* @(#)root/base:$Name:  $:$Id: MessageTypes.h,v 1.34 2006/11/20 15:56:35 rdm Exp $ */

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
   kPROOF_STATUS,                        //print status of slave - (OBSOLETE Message)
   kPROOF_PING,                          //ping slave
   kPROOF_PRINT,                         //ask master to print config
   kPROOF_RESET,                         //reset slave
   kPROOF_GETOBJECT,                     //ask for object with given name
   kPROOF_GETPACKET,                     //ask for next packet
   kPROOF_CHECKFILE,                     //filename and md5 follows
   kPROOF_SENDFILE,                      //filename, length and file follows
   kPROOF_PARALLEL,                      //number of parallel slaves follows
   kPROOF_PROCESS,                       //process events, DSet and input list follow
   kPROOF_OUTPUTLIST,                    //return the output list from Process()
   kPROOF_AUTOBIN,                       //callback for auto binning
   kPROOF_CACHE,                         //cache and package handling messages
   kPROOF_GETENTRIES,                    //report back number of entries to master
   kPROOF_PROGRESS,                      //event loop progress
   kPROOF_FEEDBACK,                      //intermediate version of objects
   kPROOF_STOPPROCESS,                   //stop or abort the current process call
   kPROOF_HOSTAUTH,                      //HostAuth info follows
   kPROOF_GETSLAVEINFO,                  //get slave info from master
   kPROOF_GETTREEHEADER,                 //get tree object
   kPROOF_GETOUTPUTLIST,                 //get the output list
   kPROOF_GETSTATS,                      //get statistics of slaves
   kPROOF_GETPARALLEL,                   //get number of parallel slaves
   kPROOF_VALIDATE_DSET,                 //validate a TDSet
   kPROOF_DATA_READY,                    //ask if the data is ready on nodes
   kPROOF_QUERYLIST,                     //ask/send the list of queries
   kPROOF_RETRIEVE,                      //asynchronous retrieve of query results
   kPROOF_ARCHIVE,                       //archive query results
   kPROOF_REMOVE,                        //remove query results from the lists
   kPROOF_STARTPROCESS,                  //signals the start of query processing
   kPROOF_SETIDLE,                       //signals idle state of session
   kPROOF_QUERYSUBMITTED,                //signals querysubmission
   kPROOF_SESSIONTAG,                    //message with unique session tag
   kPROOF_MAXQUERIES,                    //message with max number of queries
   kPROOF_CLEANUPSESSION,                //cleanup session query area
   kPROOF_SERVERSTARTED,                 //signal completion of a server startup
   kPROOF_DATASETS,                      //dataset management
   kPROOF_PACKAGE_LIST,                  //a list of package names (TObjString's) follows
   kPROOF_MESSAGE,                       //a message for the client follows
   kPROOF_LIB_INC_PATH,                  //a list of lib/inc paths follows
   kPROOF_WORKERLISTS,                   //an action on any of the worker list follows
   kPROOF_DATASET_STATUS,                //status of data set preparation before processing
   kPROOF_OUTPUTOBJECT,                  //output object follows
   kPROOF_SETENV,                        //buffer with env vars to set
   kPROOF_REALTIMELOG,                   //switch on/off real-time retrieval of log messages

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
   kROOTD_PROTOCOL,                      //returns rootd protocol
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
   kROOTD_CHMOD,                         //change permission
   kROOTD_KRB5,                          //krb5 authentication follows
   kROOTD_PROTOCOL2,                     //client proto follows, returns rootd proto
   kROOTD_BYE,                           //terminate rootd
   kROOTD_GLOBUS,                        //Globus authetication follows
   kROOTD_CLEANUP,                       //cleanup things
   kROOTD_SSH,                           //SSH-like authentication follows
   kROOTD_RFIO,                          //RFIO-like authentication follows
   kROOTD_NEGOTIA,                       //negotiation follows
   kROOTD_RSAKEY,                        //RSA public key exchange
   kROOTD_ENCRYPT,                       //an encrypted message follows
   kROOTD_OPENDIR,                       //open directory
   kROOTD_FREEDIR,                       //free directory
   kROOTD_DIRENTRY,                      //get directory entry
   kROOTD_ACCESS,                        //test Access
   kROOTD_GETS                           //multiple offset, number of byte pairs
};

#endif
