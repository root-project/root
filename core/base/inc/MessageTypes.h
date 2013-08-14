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

   //---- PROOF message opcodes (1000 - 1999)
   kPROOF_GROUPVIEW      = 1000,         //groupview follows
   kPROOF_STOP           = 1001,         //stop proof server
   kPROOF_FATAL          = 1002,         //server got fatal error and died
   kPROOF_LOGLEVEL       = 1003,         //loglevel follows
   kPROOF_LOGFILE        = 1004,         //log file length and content follows
   kPROOF_LOGDONE        = 1005,         //log file received, status follows
   kPROOF_STATUS         = 1006,         //print status of worker - (OBSOLETE Message)
   kPROOF_PING           = 1007,         //ping worker
   kPROOF_PRINT          = 1008,         //ask master to print config
   kPROOF_RESET          = 1009,         //reset worker
   kPROOF_GETOBJECT      = 1010,         //ask for object with given name
   kPROOF_GETPACKET      = 1011,         //ask for next packet
   kPROOF_CHECKFILE      = 1012,         //filename and md5 follows
   kPROOF_SENDFILE       = 1013,         //filename, length and file follows
   kPROOF_PARALLEL       = 1014,         //number of parallel workers follows
   kPROOF_PROCESS        = 1015,         //process events, DSet and input list follow
   kPROOF_OUTPUTLIST     = 1016,         //return the output list from Process()
   kPROOF_AUTOBIN        = 1017,         //callback for auto binning
   kPROOF_CACHE          = 1018,         //cache and package handling messages
   kPROOF_GETENTRIES     = 1019,         //report back number of entries to master
   kPROOF_PROGRESS       = 1020,         //event loop progress
   kPROOF_FEEDBACK       = 1021,         //intermediate version of objects
   kPROOF_STOPPROCESS    = 1022,         //stop or abort the current process call
   kPROOF_HOSTAUTH       = 1023,         //HostAuth info follows
   kPROOF_GETSLAVEINFO   = 1024,         //get worker info from master
   kPROOF_GETTREEHEADER  = 1025,         //get tree object
   kPROOF_GETOUTPUTLIST  = 1026,         //get the output list names
   kPROOF_GETSTATS       = 1027,         //get statistics of workers
   kPROOF_GETPARALLEL    = 1028,         //get number of parallel workers
   kPROOF_VALIDATE_DSET  = 1029,         //validate a TDSet
   kPROOF_DATA_READY     = 1030,         //ask if the data is ready on nodes
   kPROOF_QUERYLIST      = 1031,         //ask/send the list of queries
   kPROOF_RETRIEVE       = 1032,         //asynchronous retrieve of query results
   kPROOF_ARCHIVE        = 1033,         //archive query results
   kPROOF_REMOVE         = 1034,         //remove query results from the lists
   kPROOF_STARTPROCESS   = 1035,         //signals the start of query processing
   kPROOF_SETIDLE        = 1036,         //signals idle state of session
   kPROOF_QUERYSUBMITTED = 1037,         //signals querysubmission
   kPROOF_SESSIONTAG     = 1038,         //message with unique session tag
   kPROOF_MAXQUERIES     = 1039,         //message with max number of queries
   kPROOF_CLEANUPSESSION = 1040,         //cleanup session query area
   kPROOF_SERVERSTARTED  = 1041,         //signal completion of a server startup
   kPROOF_DATASETS       = 1042,         //dataset management
   kPROOF_PACKAGE_LIST   = 1043,         //a list of package names (TObjString's) follows
   kPROOF_MESSAGE        = 1044,         //a message for the client follows
   kPROOF_LIB_INC_PATH   = 1045,         //a list of lib/inc paths follows
   kPROOF_WORKERLISTS    = 1046,         //an action on any of the worker list follows
   kPROOF_DATASET_STATUS = 1047,         //status of data set preparation before processing
   kPROOF_OUTPUTOBJECT   = 1048,         //output object follows
   kPROOF_SETENV         = 1049,         //buffer with env vars to set
   kPROOF_REALTIMELOG    = 1050,         //switch on/off real-time retrieval of log messages
   kPROOF_VERSARCHCOMP   = 1051,         //String with worker version/architecture/compiler follows
   kPROOF_ENDINIT        = 1052,         //signals end of initialization on worker
   kPROOF_TOUCH          = 1053,         //touch the client admin file
   kPROOF_FORK           = 1054,         //ask the worker to clone itself
   kPROOF_GOASYNC        = 1055,         //switch to asynchronous mode
   kPROOF_SUBMERGER      = 1056,         //sub-merger based approach in finalization
   kPROOF_ECHO           = 1057,         //object echo request from client
   kPROOF_SENDOUTPUT     = 1058,         //control output sending

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
   kROOTD_SRPUSER        = 2013,         //user id for SRP authentication follows
   kROOTD_SRPN           = 2014,         //SRP n follows
   kROOTD_SRPG           = 2015,         //SRP g follows
   kROOTD_SRPSALT        = 2016,         //SRP salt follows
   kROOTD_SRPA           = 2017,         //SRP a follows
   kROOTD_SRPB           = 2018,         //SRP b follows
   kROOTD_SRPRESPONSE    = 2019,         //SRP final response
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
   kROOTD_KRB5           = 2030,         //krb5 authentication follows
   kROOTD_PROTOCOL2      = 2031,         //client proto follows, returns rootd proto
   kROOTD_BYE            = 2032,         //terminate rootd
   kROOTD_GLOBUS         = 2033,         //Globus authetication follows
   kROOTD_CLEANUP        = 2034,         //cleanup things
   kROOTD_SSH            = 2035,         //SSH-like authentication follows
   kROOTD_RFIO           = 2036,         //RFIO-like authentication follows
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
