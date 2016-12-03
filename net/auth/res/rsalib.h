/* @(#)root/auth:$Id$ */
/* Author: Martin Nicolay  22/11/1988 */

/******************************************************************************
Copyright (C) 2006 Martin Nicolay <m.nicolay@osm-gmbh.de>

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later
version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free
Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
MA  02110-1301  USA
******************************************************************************/

/*******************************************************************************
*                                                                              *
*       Simple RSA public key code.                                            *
*       Adaptation in library for ROOT by G. Ganis, July 2003                  *
*       (gerardo.ganis@cern.ch)                                                *
*                                                                               *
*       Header used by internal rsa functions                                   *
*                                                                               *
*******************************************************************************/

#ifndef   _RSALIB_H
#define   _RSALIB_H

#ifndef _RSADEF_H
#include "rsadef.h"
#endif

#include <stdio.h>

rsa_NUMBER rsa_genprim(int, int);
int    rsa_genrsa(rsa_NUMBER, rsa_NUMBER, rsa_NUMBER *, rsa_NUMBER *, rsa_NUMBER *);
int    rsa_encode(char *, int, rsa_NUMBER, rsa_NUMBER);
int    rsa_decode(char *, int, rsa_NUMBER, rsa_NUMBER);

int    rsa_encode_size(rsa_NUMBER);

/******************
 * nio.h          *
 ******************/

int   rsa_cmp( rsa_NUMBER*, rsa_NUMBER* );
void   rsa_assign( rsa_NUMBER*, rsa_NUMBER* );

int   rsa_num_sput( rsa_NUMBER*, char*, int );
int   rsa_num_fput( rsa_NUMBER*, FILE* );
int   rsa_num_sget( rsa_NUMBER*, char* );
int   rsa_num_fget( rsa_NUMBER*, FILE* );

#endif


