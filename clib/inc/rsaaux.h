/* @(#)root/clib:$Name:  $:$Id: mmalloc.c,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $ */
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
*       Header used by internal rsa funtions                                   *
*									       *
*******************************************************************************/
#ifndef	_RSAAUX_H
#define	_RSAAUX_H

#ifndef	_RSADEF_H
#include "rsadef.h"
#endif

extern rsa_NUMBER a_one,a_two;

/*
 * Prototypes
 */

void	a_add		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
void	a_assign	P(( rsa_NUMBER*, rsa_NUMBER* ));
int	a_cmp		P(( rsa_NUMBER*, rsa_NUMBER* ));
void	a_div		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
void	a_div2		P(( rsa_NUMBER* ));
void	a_ggt		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
void	a_imult		P(( rsa_NUMBER*, rsa_INT, rsa_NUMBER* ));
void	a_mult		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
void	a_sub		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
void	m_init		P(( rsa_NUMBER*, rsa_NUMBER* ));
void	m_add		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
void	m_mult		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
void	m_exp		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
int	n_bits		P(( rsa_NUMBER*, int));
void	n_div		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));
int	n_cmp		P(( rsa_INT*, rsa_INT*, int ));
int	n_mult		P(( rsa_INT*, rsa_INT, rsa_INT*, int ));
int	n_sub		P(( rsa_INT*, rsa_INT*, rsa_INT*, int, int ));
int	n_bitlen	P(( rsa_NUMBER* ));



/******************
 * prim.h         *
 ******************/

int	p_prim		P(( rsa_NUMBER*, int ));
void	inv		P(( rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER* ));


/******************
 * rnd.h          *
 ******************/

void	gen_number	P(( int, rsa_NUMBER* ));
void	init_rnd	P(( void ));


/******************
 * aux.h          *
 ******************/

void   do_crypt(char *, char *, int, rsa_NUMBER *);

/*
int    get_clear(char *, FILE *);
int    get_enc(char *, FILE *);
int    put_clear(char *, FILE *);
int    put_enc(char *, FILE *);
*/

#endif
