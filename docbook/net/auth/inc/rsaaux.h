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
