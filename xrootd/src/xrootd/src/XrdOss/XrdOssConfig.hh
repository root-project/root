#ifndef _XRDOSS_CONFIG_H
#define _XRDOSS_CONFIG_H
/******************************************************************************/
/*                                                                            */
/*                       X r d O s s C o n f i g . h h                        */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$
  
#define  XRDOSS_VERSION "1.0.0"

/* Constant to indicate all went well.
*/
#ifndef XrdOssOK
#define XrdOssOK 0
#endif

/* The following defines are used to control path name and directory name
   construction. They should be ste to Min(local filesystem, MSS) limits.
*/
#define XrdOssMAX_PATH_LEN       1024
  
#define XrdOssMAXDNAME           MAXNAMLEN

/* Minimum amount of free space required on selected filesystem
*/
#define XrdOssMINALLOC 0

/* Percentage of requested space to be added to the requested space amount
*/
#define XrdOssOVRALLOC 0

/* The percentage difference between two free spaces in order for them to be
   considered equivalent in terms of allocation. Specify 0 to 100, where:
     0 - Always uses cache with the most space
   100 - Always does round robin allocation regardless of free space
*/
#define XrdOssFUZALLOC 0

/* The location of the configuration file. This can be overidden by setting
   XrdOssCONFIGFN environmental variable to the name.
*/
#define XrdOssCONFIGFN (char *)""

// Set the following value to establish the ulimit for FD numbers. Zero
// sets it to whatever the current hard limit is. Negative leaves it alone.
//
#define XrdOssFDLIMIT     -1
#define XrdOssFDMINLIM    64

/* The MAXDBSIZE value sets the maximum number of bytes a database can have
   (actually applies to the seek limit). A value of zero imposes no limit.
*/
#define XrdOssMAXDBSIZE 0

// Flags set in OptFlags
//
#define XrdOss_ROOTDIR   0x00000001
#define XrdOss_USRPRTY   0x00000002
#define XrdOss_EXPORT    0x00000004


/* Set the following:
   XrdOssSCANINT    - Number of seconds between cache scans
   XrdOssXFRTHREADS - maximum number of threads used for staging
   XrdOssXFRSPEED   - average bytes/second to transfer a file
   XrdOssXFROVHD    - minimum number of seconds to get a file
   XrdOssXFRWAIT    - number of seconds to hold a failing stage request
*/
#define XrdOssCSCANINT    600
#define XrdOssXFRTHREADS    1
#define XrdOssXFRSPEED      9*1024*1024
#define XrdOssXFROVHD      30
#define XrdOssXFRHOLD       3*60*60

/******************************************************************************/
/*               r e m o t e   f l i s t   d e f i n i t i o n                */
/******************************************************************************/

/* The RPlist object defines an entry in the remote file list which is anchored
   at XrdOssSys::RPlist. There is one entry in the list for each remotepath
   directive. When a request comes in, the named path is compared with entries
   in the RPList. If no prefix match is found, the request is treated as being
   directed to the local filesystem. No path prefixing occurs and no remote
   filesystem is invoked. The list is searched in reverse of specification.
   No defaults can be specified for this list.
*/
#endif
