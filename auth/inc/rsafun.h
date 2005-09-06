// Author: Gerardo Ganis  07/07/2003

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
* 									       *
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
