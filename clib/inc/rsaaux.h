/* @(#)root/clib:$Name:  $:$Id: rsaaux.h,v 1.1 2003/08/29 10:38:18 rdm Exp $ */
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

void	a_add(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
void	a_assign(rsa_NUMBER*, rsa_NUMBER*);
int	a_cmp(rsa_NUMBER*, rsa_NUMBER*);
void	a_div(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
void	a_div2(rsa_NUMBER*);
void	a_ggt(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
void	a_imult(rsa_NUMBER*, rsa_INT, rsa_NUMBER*);
void	a_mult(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
void	a_sub(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
void	m_init(rsa_NUMBER*, rsa_NUMBER*);
void	m_add(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
void	m_mult(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
void	m_exp(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
int	n_bits(rsa_NUMBER*, int);
void	n_div(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);
int	n_cmp(rsa_INT*, rsa_INT*, int);
int	n_mult(rsa_INT*, rsa_INT, rsa_INT*, int);
int	n_sub(rsa_INT*, rsa_INT*, rsa_INT*, int, int);
int	n_bitlen(rsa_NUMBER*);



/******************
 * prim.h         *
 ******************/

int	p_prim(rsa_NUMBER*, int);
void	inv(rsa_NUMBER*, rsa_NUMBER*, rsa_NUMBER*);


/******************
 * rnd.h          *
 ******************/

void	gen_number(int, rsa_NUMBER*);
void	init_rnd(void);


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
