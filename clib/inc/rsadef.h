/* @(#)root/clib:$Name:  $:$Id: rsadef.h,v 1.2 2003/08/29 17:23:31 rdm Exp $ */
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
*       General rsa definitions header                                         *
*									       *
*******************************************************************************/

#ifndef	_RSADEF_H
#define	_RSADEF_H

typedef	unsigned short rsa_INT;		/* muss MAXINT fassen		*/
typedef	unsigned long rsa_LONG;		/* muss (MAXINT+1)^2 -1 fassen	*/

/*
 *	(MAXINT+1)-adic Zahlen
 */

/*
 *	MAXINT		Maximale Zahl pro Elemenmt (muss int sein)
 *	MAXBIT		Maximales Bit von MAXINT
 *	LOWBITS		Anzahl der consekutiven low Bits von MAXINT
 *	HIGHBIT		Hoechsten Bit von MAXINT
 *	TOINT		muss (INT)( (x) % MAXINT) ergeben
 *	MAXLEN		Laenge der INT Array in jeder NUMBER
 */

#define rsa_MAXINT		0xFFFF

#if rsa_MAXINT == 99
#define	rsa_MAXBIT		7
#define	rsa_LOWBITS 	2
#endif
#if rsa_MAXINT == 9
#define	rsa_MAXBIT		4
#define	rsa_LOWBITS 	1
#endif
#if rsa_MAXINT == 1
#define rsa_MAXBIT		1
#endif
#if rsa_MAXINT == 0xFF
#define rsa_MAXBIT		8
#define	rsa_TOINT(x)	((rsa_INT)(x))		/* ACHTUNG !!!!! */
#endif
#if rsa_MAXINT == 0xFFFF
#define rsa_MAXBIT		16
#define	rsa_TOINT(x)	((rsa_INT)(x))		/* ACHTUNG !!!!! */
#endif

#ifndef	rsa_MAXBIT
#include	"<< ERROR: rsa_MAXBIT must be defined >>"
#endif
#ifndef	rsa_LOWBITS
#if rsa_MAXINT == (1 << rsa_MAXBIT) - 1
#define	rsa_LOWBITS		rsa_MAXBIT
#else
#include	"<< ERROR: rsa_LOWBITS must be defined >>"
#endif
#endif

#define	rsa_MAXLEN		(300*8/(rsa_MAXBIT + 1))
#define	rsa_STRLEN		(rsa_MAXLEN*rsa_MAXBIT/4)
#define	rsa_HIGHBIT		(1 << (rsa_MAXBIT-1) )

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


