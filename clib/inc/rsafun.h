// @(#)root/clib:$Name:  $:$Id: TSocket.cxx,v 1.10 2002/05/18 08:22:00 brun Exp $
// Author:

/*******************************************************************************
*									       *
*	Copyright (c) Martin Nicolay,  22. Nov. 1988			       *
*									       *
*	Wenn diese (oder sinngemaess uebersetzte) Copyright-Angabe enthalten   *
*	bleibt, darf diese Source fuer jeden nichtkomerziellen Zweck weiter    *
*	verwendet werden.						       *
*									       *
*	martin@trillian.megalon.de					       *
*									       *
*       ftp://ftp.funet.fi/pub/crypt/cryptography/asymmetric/rsa               *
*									       *
*       Simple RSA public key code.                                            *
*       Adaptation in library for ROOT by G. Ganis, July 2003                  *
*       (gerardo.ganis@cern.ch)                                                *
*									       *
*       Hooks for useful rsa funtions                                          *
**									       *
*******************************************************************************/

#include <stdio.h>

#ifndef ROOT_rsafun
#define ROOT_rsafun

extern "C" {
#ifndef _RSADEF_H
#include "rsadef.h"
#endif
}


typedef  rsa_NUMBER (*rsa_genprim_t)(int, int);
typedef  int    (*rsa_genrsa_t)(rsa_NUMBER, rsa_NUMBER, rsa_NUMBER *, rsa_NUMBER *, rsa_NUMBER *);
typedef  int    (*rsa_encode_t)(char *, int, rsa_NUMBER, rsa_NUMBER);
typedef  int    (*rsa_decode_t)(char *, int, rsa_NUMBER, rsa_NUMBER);
typedef  int	(*rsa_num_sput_t)(rsa_NUMBER*, char*, int );
typedef  int	(*rsa_num_fput_t)(rsa_NUMBER*, FILE* );
typedef  int	(*rsa_num_sget_t)(rsa_NUMBER*, char* );
typedef  int	(*rsa_num_fget_t)(rsa_NUMBER*, FILE* );
typedef  int    (*rsa_assign_t)(rsa_NUMBER *, rsa_NUMBER *);
typedef  int    (*rsa_cmp_t)(rsa_NUMBER *, rsa_NUMBER *);


class rsa_fun {

public:
   static rsa_genprim_t   fg_rsa_genprim;
   static rsa_genrsa_t    fg_rsa_genrsa;
   static rsa_encode_t    fg_rsa_encode;
   static rsa_decode_t    fg_rsa_decode;
   static rsa_num_sput_t  fg_rsa_num_sput;
   static rsa_num_fput_t  fg_rsa_num_fput;
   static rsa_num_sget_t  fg_rsa_num_sget;
   static rsa_num_fget_t  fg_rsa_num_fget;
   static rsa_assign_t    fg_rsa_assign;
   static rsa_cmp_t       fg_rsa_cmp;

   rsa_fun(rsa_genprim_t, rsa_genrsa_t, rsa_encode_t, rsa_decode_t,
           rsa_num_sput_t, rsa_num_fput_t, rsa_num_sget_t, rsa_num_fget_t, rsa_assign_t, rsa_cmp_t);
};

#endif
