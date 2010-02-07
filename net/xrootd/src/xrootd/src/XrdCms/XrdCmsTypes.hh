#ifndef XRDCMSTYPES__H
#define XRDCMSTYPES__H
/******************************************************************************/
/*                                                                            */
/*                        X r d C m s T y p e s . h h                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$
  
typedef unsigned long long SMask_t;

#define FULLMASK 0xFFFFFFFFFFFFFFFFULL

// The following defines our cell size (maximum subscribers)
//
#define STMax 64

// The following defines the maximum number of redirectors. It is one greater
// than the actual maximum as the zeroth is never used.
//
#define maxRD 65

#define XrdCmsMAX_PATH_LEN 1024

#define XrdCmsVERSION "1.0.0"
#endif
