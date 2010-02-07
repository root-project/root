#ifndef __XRDSYS_HEADERS_H__
#define __XRDSYS_HEADERS_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d S y s H e a d e r s . h h                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//        $Id$

// This header has been introduced to help the transition to new versions
// of the gcc compiler which deprecates or even not support some standard
// headers in the form <header_name>.h
//

#if !defined(HAVE_OLD_HDRS) || defined(WIN32)

// gcc >= 4.3, cl require this
#  include <iostream>
using namespace std;

#else

#  include <iostream.h>

#endif



#endif  // __XRDSYS_HEADERS_H__
