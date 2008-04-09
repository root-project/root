#ifndef __OSS_OPAQUE_H__
#define __OSS_OPAQUE_H__
/******************************************************************************/
/*                                                                            */
/*                       X r d O s s O p a q u e . h h                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$


/******************************************************************************/
/*                E x t e r n a l   C o n f i g u r a t i o n                 */
/******************************************************************************/
  
#define OSS_ASIZE           (char *)"oss.asize"
#define OSS_CGROUP          (char *)"oss.cgroup"
#define OSS_USRPRTY         (char *)"oss.sprty"
#define OSS_SYSPRTY         (char *)"oss&sprty"
#define OSS_CGROUP_DEFAULT  (char *)"public"

#define OSS_VARLEN          32

#define OSS_MAX_PRTY        15
#define OSS_USE_PRTY         7

#endif
