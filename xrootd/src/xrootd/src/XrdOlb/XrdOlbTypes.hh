#ifndef __OLB__TYPES__H
#define __OLB__TYPES__H
/******************************************************************************/
/*                                                                            */
/*                        X r d O l b T y p e s . h h                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$
  
typedef unsigned long long SMask_t;

#define BUFF(x) char buff[x]; buff[x-1] = '\0';

#define XrdOlbMAX_PATH_LEN 1024

#define XrdOlbVERSION "2.0.0"
#endif
