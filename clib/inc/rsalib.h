/* @(#)root/clib:$Name:  $:$Id: rsalib.h,v 1.1 2003/08/29 10:38:18 rdm Exp $ */
/* Author: */

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
*       prototypes for rsa funtions of public interest                         *
*									       *
*******************************************************************************/

#ifndef	_RSALIB_H
#define	_RSALIB_H

rsa_NUMBER rsa_genprim(int, int);
int    rsa_genrsa(rsa_NUMBER, rsa_NUMBER, rsa_NUMBER *, rsa_NUMBER *, rsa_NUMBER *);
int    rsa_encode(char *, int, rsa_NUMBER, rsa_NUMBER);
int    rsa_decode(char *, int, rsa_NUMBER, rsa_NUMBER);


/******************
 * nio.h          *
 ******************/

int	rsa_cmp( rsa_NUMBER*, rsa_NUMBER* );
void	rsa_assign( rsa_NUMBER*, rsa_NUMBER* );

int	rsa_num_sput( rsa_NUMBER*, char*, int );
int	rsa_num_fput( rsa_NUMBER*, FILE* );
int	rsa_num_sget( rsa_NUMBER*, char* );
int	rsa_num_fget( rsa_NUMBER*, FILE* );

#endif


