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
*									       *
*******************************************************************************/

#include "rsafun.h"


extern "C" {
   rsa_NUMBER rsa_genprim(int, int);
   int    rsa_genrsa(rsa_NUMBER, rsa_NUMBER, rsa_NUMBER *, rsa_NUMBER *, rsa_NUMBER *);
   int    rsa_encode(char *, int, rsa_NUMBER, rsa_NUMBER);
   int    rsa_decode(char *, int, rsa_NUMBER, rsa_NUMBER);
   int	  rsa_num_sput( rsa_NUMBER*, char*, int );
   int	  rsa_num_fput( rsa_NUMBER*, FILE* );
   int	  rsa_num_sget( rsa_NUMBER*, char* );
   int	  rsa_num_fget( rsa_NUMBER*, FILE* );
   int	  rsa_assign( rsa_NUMBER*, rsa_NUMBER* );
   int	  rsa_cmp( rsa_NUMBER*, rsa_NUMBER* );
}

rsa_genprim_t  rsa_fun::fg_rsa_genprim;
rsa_genrsa_t   rsa_fun::fg_rsa_genrsa;
rsa_encode_t   rsa_fun::fg_rsa_encode;
rsa_decode_t   rsa_fun::fg_rsa_decode;
rsa_num_sput_t rsa_fun::fg_rsa_num_sput;
rsa_num_fput_t rsa_fun::fg_rsa_num_fput;
rsa_num_sget_t rsa_fun::fg_rsa_num_sget;
rsa_num_fget_t rsa_fun::fg_rsa_num_fget;
rsa_assign_t   rsa_fun::fg_rsa_assign;
rsa_cmp_t      rsa_fun::fg_rsa_cmp;

// Static instantiation to load hooks during dynamic load
static rsa_fun  rsa_init(&rsa_genprim,&rsa_genrsa,&rsa_encode,&rsa_decode,
                         &rsa_num_sput,&rsa_num_fput,&rsa_num_sget,&rsa_num_fget,&rsa_assign,&rsa_cmp);

rsa_fun::rsa_fun(rsa_genprim_t genprim, rsa_genrsa_t genrsa, rsa_encode_t encode, rsa_decode_t decode,
                 rsa_num_sput_t num_sput, rsa_num_fput_t num_fput, rsa_num_sget_t num_sget, rsa_num_fget_t num_fget,
                 rsa_assign_t assign, rsa_cmp_t cmp)
{
  // ctor

  fg_rsa_genprim = genprim;
  fg_rsa_genrsa  = genrsa;
  fg_rsa_encode  = encode;
  fg_rsa_decode  = decode;
  fg_rsa_num_sput = num_sput;
  fg_rsa_num_fput = num_fput;
  fg_rsa_num_sget = num_sget;
  fg_rsa_num_fget = num_fget;
  fg_rsa_assign   = assign;
  fg_rsa_cmp      = cmp;
}
