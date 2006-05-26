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

rsa_NUMBER rsa_genprim(int, int);
int rsa_genrsa(rsa_NUMBER, rsa_NUMBER, rsa_NUMBER *, rsa_NUMBER *, rsa_NUMBER *);
int rsa_encode(char *, int, rsa_NUMBER, rsa_NUMBER);
int rsa_decode(char *, int, rsa_NUMBER, rsa_NUMBER);
int rsa_num_sput( rsa_NUMBER*, char*, int );
int rsa_num_fput( rsa_NUMBER*, FILE* );
int rsa_num_sget( rsa_NUMBER*, char* );
int rsa_num_fget( rsa_NUMBER*, FILE* );
void rsa_assign( rsa_NUMBER*, rsa_NUMBER* );
int rsa_cmp( rsa_NUMBER*, rsa_NUMBER* );

RSA_genprim_t  TRSA_fun::fg_rsa_genprim;
RSA_genrsa_t   TRSA_fun::fg_rsa_genrsa;
RSA_encode_t   TRSA_fun::fg_rsa_encode;
RSA_decode_t   TRSA_fun::fg_rsa_decode;
RSA_num_sput_t TRSA_fun::fg_rsa_num_sput;
RSA_num_fput_t TRSA_fun::fg_rsa_num_fput;
RSA_num_sget_t TRSA_fun::fg_rsa_num_sget;
RSA_num_fget_t TRSA_fun::fg_rsa_num_fget;
RSA_assign_t   TRSA_fun::fg_rsa_assign;
RSA_cmp_t      TRSA_fun::fg_rsa_cmp;

RSA_genprim_t  TRSA_fun::RSA_genprim() { return fg_rsa_genprim; }
RSA_genrsa_t   TRSA_fun::RSA_genrsa() { return fg_rsa_genrsa; }
RSA_encode_t   TRSA_fun::RSA_encode() { return fg_rsa_encode; }
RSA_decode_t   TRSA_fun::RSA_decode() { return fg_rsa_decode; }
RSA_num_sput_t TRSA_fun::RSA_num_sput() { return fg_rsa_num_sput; }
RSA_num_fput_t TRSA_fun::RSA_num_fput() { return fg_rsa_num_fput; }
RSA_num_sget_t TRSA_fun::RSA_num_sget() { return fg_rsa_num_sget; }
RSA_num_fget_t TRSA_fun::RSA_num_fget() { return fg_rsa_num_fget; }
RSA_assign_t   TRSA_fun::RSA_assign() { return fg_rsa_assign; }
RSA_cmp_t      TRSA_fun::RSA_cmp() { return fg_rsa_cmp; }

// Static instantiation to load hooks during dynamic load
static TRSA_fun  g_rsa_init(&rsa_genprim,&rsa_genrsa,&rsa_encode,&rsa_decode,
                            &rsa_num_sput,&rsa_num_fput,&rsa_num_sget,
                            &rsa_num_fget,&rsa_assign,&rsa_cmp);

TRSA_fun::TRSA_fun(RSA_genprim_t genprim, RSA_genrsa_t genrsa, RSA_encode_t encode,
                   RSA_decode_t decode, RSA_num_sput_t num_sput, RSA_num_fput_t num_fput,
                   RSA_num_sget_t num_sget, RSA_num_fget_t num_fget,
                   RSA_assign_t assign, RSA_cmp_t cmp)
{
   // constructor

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
