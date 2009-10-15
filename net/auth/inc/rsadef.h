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

#ifndef	_RSADEF_H
#define	_RSADEF_H

typedef	unsigned short rsa_INT;		/* muss MAXINT fassen		*/
typedef	unsigned long rsa_LONG;		/* muss (MAXINT+1)^2 -1 fassen	*/

/*
 *	(MAXINT+1)-adic Zahlen
 */

/*
 *	MAXINT		Maximale Zahl pro Element (muss int sein)
 *	MAXBIT		Maximales Bit von MAXINT
 *	LOWBITS		Anzahl der consekutiven low Bits von MAXINT
 *	HIGHBIT		Hoechsten Bit von MAXINT
 *	TOINT		muss (INT)( (x) % MAXINT) ergeben
 *	MAXLEN		Laenge der INT Array in jeder NUMBER
 */

#define rsa_MAXINT	0xFFFF

#if rsa_MAXINT == 99
#define	rsa_MAXBIT	7
#define	rsa_LOWBITS 	2
#endif
#if rsa_MAXINT == 9
#define	rsa_MAXBIT	4
#define	rsa_LOWBITS 	1
#endif
#if rsa_MAXINT == 1
#define rsa_MAXBIT	1
#endif
#if rsa_MAXINT == 0xFF
#define rsa_MAXBIT	8
#define	rsa_TOINT(x)	((rsa_INT)(x))		/* ACHTUNG !!!!! */
#endif
#if rsa_MAXINT == 0xFFFF
#define rsa_MAXBIT	16
#define	rsa_TOINT(x)	((rsa_INT)(x))		/* ACHTUNG !!!!! */
#endif

#ifndef	rsa_MAXBIT
#include	"<< ERROR: rsa_MAXBIT must be defined >>"
#endif
#ifndef	rsa_LOWBITS
#if rsa_MAXINT == (1 << rsa_MAXBIT) - 1
#define	rsa_LOWBITS	rsa_MAXBIT
#else
#include	"<< ERROR: rsa_LOWBITS must be defined >>"
#endif
#endif

#define	rsa_MAXLEN	(300*8/(rsa_MAXBIT + 1))
#define	rsa_STRLEN	(rsa_MAXLEN*rsa_MAXBIT/4)
#define	rsa_HIGHBIT	(1 << (rsa_MAXBIT-1) )

#if rsa_LOWBITS == rsa_MAXBIT
#define	rsa_DIVMAX1(x)	((x) >> rsa_MAXBIT)
#define	rsa_MODMAX1(x)	((x) & rsa_MAXINT)
#define	rsa_MULMAX1(x)	((x) << rsa_MAXBIT)
#else
#define	rsa_DIVMAX1(x)	((x) / (rsa_MAXINT+1))
#define	rsa_MODMAX1(x)	((x) % (rsa_MAXINT+1))
#define	rsa_MULMAX1(x)	((x) * (unsigned)(rsa_MAXINT+1))
#endif

#ifndef	rsa_TOINT
#define	rsa_TOINT(x)	((rsa_INT)rsa_MODMAX1(x))
#endif

typedef struct {
	int	n_len;			/* Hoechster benutzter Index */
	rsa_INT	n_part[rsa_MAXLEN];
} rsa_NUMBER;

#define	rsa_NUM0P	((rsa_NUMBER *)0)		/* Abkuerzung */

/* Key structures */
typedef struct {
        rsa_NUMBER n;   /* modulus */
        rsa_NUMBER e;   /* private or public exponent */
} rsa_KEY;
typedef struct {
        int   len;      /*  length of 'data' in bytes */
        char *keys;     /* 'HEX[n]#HEX[d]\0' */
} rsa_KEY_export;


#endif
