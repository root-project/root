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
*       rsa funtions of public interest                                        *
*									       *
*******************************************************************************/

#include	<stdio.h>
#include	<string.h>
#include	<ctype.h>
#include	<stdlib.h>
#include        <errno.h>

#include	"rsaaux.h"
#include	"rsalib.h"

static int	clear_siz;		/* clear-text blocksize		*/
static int	enc_siz;		/* encoded blocksize		*/
					/* clear_siz < enc_siz		*/

int gLog = 0;

rsa_NUMBER rsa_genprim(int len, int prob)
{
        rsa_NUMBER a_three,a_four;
	rsa_NUMBER prim;
	int i;

	a_add( &a_one, &a_two, &a_three );
	a_add( &a_two, &a_two, &a_four );

	init_rnd();

	do {
		gen_number( len, &prim );
	} while ( !prim.n_len );

	a_mult( &prim, &a_two, &prim );
	a_mult( &prim, &a_three, &prim );
	a_add( &prim, &a_one, &prim );

	for (i=1 ;; i++) {

		if (p_prim( &prim, prob ))
			break;
		if (i % 2)
			a_add( &prim, &a_four, &prim );
		else
			a_add( &prim, &a_two, &prim );
	}

	return prim;
}

int rsa_genrsa(rsa_NUMBER p1, rsa_NUMBER p2, rsa_NUMBER *n, rsa_NUMBER *e, rsa_NUMBER *d)
{
	rsa_NUMBER phi, *max_p;
	int len;

	if ( !a_cmp( &p1, &p2) ) return 1;

	if (a_cmp( &p1, &p2) > 0)
		max_p = &p1;
	else
		max_p = &p2;


	a_mult( &p1, &p2, n );
	a_sub( &p1, &a_one, &phi );
	a_sub( &p2, &a_one, e );
	a_mult( &phi, e, &phi );

	len = n_bitlen( &phi );
	len = ( len + 3) / 4;

	a_assign( &p1, &phi );
	a_sub( &p1, &a_one, &p1 );

	init_rnd();

	do {
		do {
			gen_number( len, d );
		} while (a_cmp( d, max_p) <= 0 || a_cmp( d, &p1) >= 0);

		a_ggt( d, &phi, e );
	} while ( a_cmp( e, &a_one) );

	inv( d, &phi, e );

        return 0;
}


int rsa_encode(char *bufin, int lin, rsa_NUMBER n, rsa_NUMBER e)
{
  /* Encodes plain string in 'bufin' (output in 'bufin')
     Returns length of encoded string
     (key validity is not checked) */

    char  buf[ rsa_STRLEN*2 ];
    char  bufout[ rsa_STRLEN*2 ];
    int   i, j, lout;
    char *pout;

    enc_siz = ( n_bitlen( &n) + 7) / 8;
    clear_siz = enc_siz -1;
    m_init( &n, rsa_NUM0P );

    pout = bufout;
    lout = 0;
    for ( i = 0; i < lin; i += clear_siz) {

      memcpy(buf,bufin+i,clear_siz);

      j = ((lin-i) < clear_siz) ? lin-i : clear_siz;
      memset(buf+j,0,(enc_siz-j));

      do_crypt( buf, buf, enc_siz, &e );

      memcpy(pout,buf,enc_siz);

      pout += enc_siz;
      lout += enc_siz;
    }

    memcpy(bufin,bufout,lout);

    return lout;

}

int rsa_decode(char *bufin, int lin, rsa_NUMBER n, rsa_NUMBER e)
{
  /* Decodes string in 'bufin' (output in 'bufin')
     Returns length of plaintext string
     (key validity is not checked) */

    char  buf[ rsa_STRLEN*2 ];
    char  bufout[ rsa_STRLEN*2 ];
    int   i, lout;
    char *pout;

    enc_siz = ( n_bitlen( &n) + 7) / 8;
    clear_siz = enc_siz -1;
    m_init( &n, rsa_NUM0P );

    pout = bufout;
    lout = 0;
    for ( i = 0; i < lin; i += enc_siz) {

      memcpy(buf,bufin+i,enc_siz);

      do_crypt( buf, buf, enc_siz, &e );

      memcpy(pout,buf,clear_siz);

      pout += clear_siz;
      lout += clear_siz;
    }

    memcpy(bufin,bufout,lout);

    return lout;

}


/*******************************************************************************
*									       *
* nio.c                                                                        *
*                                                                              *
********************************************************************************/


/*
 *	rsa_NUMBER io
 */

/*
 *		Funktionen
 *
 * int	num_sput( n, s, l)
 *		rsa_NUMBER *n;
 *		char s[l];
 *			schreibt *n als Hex-Zahl in s
 *
 * int	num_fput( n, f )
 *		rsa_NUMBER *n;
 *		FILE *f;
 *			schreibt *n als Hex-Zahl in File f
 *
 * int	num_sget( n, s )
 *		rsa_NUMBER *n;
 *		char *s;
 *			liest Hex-Zahl s in *n ein
 *
 * int	num_fget( n, f )
 *		rsa_NUMBER *n;
 *		FILE *f;
 *			liest eine Hex-Zahl von f in *n ein
 *
 */


static char *HEX="0123456789ABCDEF";
static char *hex="0123456789abcdef";

static rsa_NUMBER bits[9];
static rsa_NUMBER int16[16];

static int init = 0;

void num_init()
{
	int i;

        if (init) return;

	a_assign( &bits[0], &a_one );
	for ( i=1; i<9; i++)
		a_add( &bits[i-1], &bits[i-1], &bits[i] );

	a_assign( &int16[0], &a_one );
	for ( i=1; i<16; i++)
		a_add( &int16[i-1], &a_one, &int16[i] );

	init = 1;
}


int rsa_num_sput( n, s, l)
rsa_NUMBER *n;
char *s;
int l;
{
#if rsa_MAXINT == ( (1 << rsa_MAXBIT) - 1 )
	rsa_INT *p;
	int bi,ab,i;
	long b;
	int first = 1;

	bi = rsa_MAXBIT * n->n_len;
	ab = 4 - (bi + 3) % 4 -1;
	p  = &n->n_part[n->n_len -1];

	if ( (bi+3) / 4 >= l )
		return(EOF);

	b  = 0;
	while (bi) {
		b <<= (rsa_MAXBIT);
		b |= (unsigned long)*p--;
		bi -= rsa_MAXBIT;
		ab += rsa_MAXBIT;
		while (ab >= 4) {
			i = (b >> (ab - 4));
			b &= ( 1L << (ab - 4)) -1L;
			ab -= 4;

			if (first && !i)
				continue;
			first = 0;
			*s++ = HEX[ i ];
		}
	}
	if (b)
		abort();
	*s = '\0';

	return (0);
#else
	rsa_NUMBER r,q;
	int i,b,p,len,low,high;
	char *np;

	if (! init)
		num_init();

	a_assign( &q, n);
	len = l;
	np = s + l;

	for (; q.n_len && len > 1; len --) {
		a_div( &q, &bits[4], &q, &r );
		for (p=8, b=0, i=3; i >= 0; i--, p /= 2) {
			if ( a_cmp( &r, &bits[i]) >= 0) {
				a_sub( &r, &bits[i], &r );
				b += p;
			}
		}
		*--np = HEX[ b ];
	}
	if (q.n_len)
		return(EOF);

	l -= len;
	len = l;
	for (; l--; )
		*s++ = *np++;

	*s = '\0';

	return (0);
#endif
}


int rsa_num_fput( n, f )
rsa_NUMBER *n;
FILE *f;
{
	int j;
	char *np;
	char n_print[ rsa_STRLEN + 1 ];

	if ( rsa_num_sput( n, n_print, sizeof( n_print) ) == EOF )
		return(EOF);

	for (j=0, np=n_print; *np ; np++, j++) {
		if (j==64) {
			fputs("\n",f);
			j = 0;
		}
		putc((int)*np,f);
	}

	if (j)
		putc('\n',f);

	return(0);
}


int rsa_num_sget( n, s )
rsa_NUMBER *n;
char *s;
{
#if rsa_MAXINT == ( (1 << rsa_MAXBIT) - 1 )
	rsa_INT *p;
	char *hp;
	int bi,ab,i;
	long b;
	int first = 1;

	bi = 4 * strlen(s);
	ab = rsa_MAXBIT - (bi + rsa_MAXBIT -1) % rsa_MAXBIT -1;
	i  =  (bi + rsa_MAXBIT-1) / rsa_MAXBIT;
	p  = &n->n_part[ i -1 ];
	n->n_len = i;

	if ( i > rsa_MAXLEN )
		return(EOF);

	b  = 0;
	while (bi > 0) {
		if ( (hp = strchr( HEX, *s )) )
			i = hp - HEX;
		else if ((hp = strchr( hex, *s )) )
			i = hp - hex;
		else
			return(EOF);
		s++;

		b <<= 4;
		b |= (unsigned long)i;
		bi -= 4;
		ab += 4;
		while (ab >= rsa_MAXBIT) {
			i = (b >> (ab - rsa_MAXBIT));
			b &= ( 1L << (ab - rsa_MAXBIT)) -1L;
			ab -= rsa_MAXBIT;
			if (first && !i) {
				p--;
				n->n_len--;
			}
			else {
				first = 0;
				*p-- = i;
			}
		}
	}
	if (b)
		abort();
	*s = '\0';

	return (0);
#else
	char *p;
	int i,c;

	if (! init)
		num_init();

	n->n_len = 0;
	while ( (c = *s++ & 0xFF)) {
		if ( p= strchr( HEX, c) )
			i = p - HEX;
		else if ( p= strchr( hex, c) )
			i = p - hex;
		else
			return(EOF);

		a_mult( n, &bits[4], n );
		if (i)
			a_add( n, &int16[i-1], n );
	}

	return(0);
#endif
}

int rsa_num_fget( n, f )
rsa_NUMBER *n;
FILE *f;
{
	int j,c;
	char *np;
	char n_print[ rsa_STRLEN + 1 ];

	np = n_print;
	j = sizeof(n_print);
	while ( (c=getc(f)) != EOF && ( isxdigit(c) || isspace(c)) ) {
		if (isspace(c))
			continue;
		if (! --j)
			return(EOF);
		*np++ = (char)c;
	}
	*np = '\0';

	if (c != EOF)
		ungetc(c,f);

	if ( rsa_num_sget( n, n_print) == EOF )
		return( EOF );

	return(0);
}

int rsa_cmp( c1, c2 )
rsa_NUMBER *c1,*c2;
{
	int l;
					/* bei verschiedener Laenge klar*/
	if ( (l=c1->n_len) != c2->n_len)
		return( l - c2->n_len);

					/* vergleiche als arrays	*/
	return( n_cmp( c1->n_part, c2->n_part, l) );
}

void rsa_assign( d, s )
rsa_NUMBER *d,*s;
{
	int l;

	if (s == d)			/* nichts zu kopieren		*/
		return;

	if ((l=s->n_len))
		memcpy( d->n_part, s->n_part, sizeof(rsa_INT)*l);

	d->n_len = l;
}

