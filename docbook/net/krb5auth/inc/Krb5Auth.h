// @(#)root/krb5auth:$Id$
// Author: Johannes Muelmenstaedt  17/03/2002

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/* Parts of this file are copied from the MIT krb5 distribution and
 * are subject to the following license:
 *
 * Copyright 1990,1991 by the Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * Export of this software from the United States of America may
 *   require a specific license from the United States Government.
 *   It is the responsibility of any person or organization contemplating
 *   export to obtain such a license before exporting.
 *
 * WITHIN THAT CONSTRAINT, permission to use, copy, modify, and
 * distribute this software and its documentation for any purpose and
 * without fee is hereby granted, provided that the above copyright
 * notice appear in all copies and that both that copyright notice and
 * this permission notice appear in supporting documentation, and that
 * the name of M.I.T. not be used in advertising or publicity pertaining
 * to distribution of the software without specific, written prior
 * permission.  Furthermore if you modify this software you must label
 * your software as modified software and not distribute it in such a
 * fashion that it might be confused with the original M.I.T. software.
 * M.I.T. makes no representations about the suitability of
 * this software for any purpose.  It is provided "as is" without express
 * or implied warranty.
 *
 */

#ifndef ROOT_Krb5Auth
#define ROOT_Krb5Auth

extern "C" {
   #define KRB5_PRIVATE 1
   #if (defined(__sun) || defined(__sgi) || defined(__APPLE__)) && !defined(linux)
      #include <com_err.h>
   #endif
   #include <krb5.h>
   #if defined(linux) && !defined(__COM_ERR_H)
      #include <com_err.h>
   #endif
   int krb5_net_read(krb5_context, int, char *, int); // ow ow ow!
}

#endif
