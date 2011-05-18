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
*                                            									       *
*       Simple RSA public key code.                                            *
*       Adaptation in library for ROOT by G. Ganis, July 2003                  *
*       (gerardo.ganis@cern.ch)                                                *
*									                                                    *
*       Header used by internal rsa functions                                   *
*									                                                    *
*******************************************************************************/

#include <stdio.h>

#ifndef ROOT_rsafun
#define ROOT_rsafun

#ifndef _RSADEF_H
#include "rsadef.h"
#endif


typedef  rsa_NUMBER (*RSA_genprim_t)(int, int);
typedef  int    (*RSA_genrsa_t)(rsa_NUMBER, rsa_NUMBER, rsa_NUMBER *, rsa_NUMBER *, rsa_NUMBER *);
typedef  int    (*RSA_encode_t)(char *, int, rsa_NUMBER, rsa_NUMBER);
typedef  int    (*RSA_decode_t)(char *, int, rsa_NUMBER, rsa_NUMBER);
typedef  int	(*RSA_num_sput_t)(rsa_NUMBER*, char*, int );
typedef  int	(*RSA_num_fput_t)(rsa_NUMBER*, FILE* );
typedef  int	(*RSA_num_sget_t)(rsa_NUMBER*, char* );
typedef  int	(*RSA_num_fget_t)(rsa_NUMBER*, FILE* );
typedef  void   (*RSA_assign_t)(rsa_NUMBER *, rsa_NUMBER *);
typedef  int    (*RSA_cmp_t)(rsa_NUMBER *, rsa_NUMBER *);


class TRSA_fun {

private:
   static RSA_genprim_t   fg_rsa_genprim;
   static RSA_genrsa_t    fg_rsa_genrsa;
   static RSA_encode_t    fg_rsa_encode;
   static RSA_decode_t    fg_rsa_decode;
   static RSA_num_sput_t  fg_rsa_num_sput;
   static RSA_num_fput_t  fg_rsa_num_fput;
   static RSA_num_sget_t  fg_rsa_num_sget;
   static RSA_num_fget_t  fg_rsa_num_fget;
   static RSA_assign_t    fg_rsa_assign;
   static RSA_cmp_t       fg_rsa_cmp;

public:
   static RSA_genprim_t   RSA_genprim();
   static RSA_genrsa_t    RSA_genrsa();
   static RSA_encode_t    RSA_encode();
   static RSA_decode_t    RSA_decode();
   static RSA_num_sput_t  RSA_num_sput();
   static RSA_num_fput_t  RSA_num_fput();
   static RSA_num_sget_t  RSA_num_sget();
   static RSA_num_fget_t  RSA_num_fget();
   static RSA_assign_t    RSA_assign();
   static RSA_cmp_t       RSA_cmp();

   TRSA_fun(RSA_genprim_t, RSA_genrsa_t, RSA_encode_t, RSA_decode_t,
           RSA_num_sput_t, RSA_num_fput_t, RSA_num_sget_t, RSA_num_fget_t, RSA_assign_t, RSA_cmp_t);
};

#endif
