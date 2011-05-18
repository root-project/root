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
*******************************************************************************/

#include	<stdio.h>
#include	<string.h>
#include	<ctype.h>
#include	<stdlib.h>
#include <errno.h>

#include	"rsaaux.h"
#include	"rsalib.h"

static int	g_clear_siz;            /* clear-text blocksize	        */
static int	g_enc_siz;              /* encoded blocksize            */
                                    /* g_clear_siz < g_enc_siz      */

int gLog = 0;
int kMAXT = 100;

rsa_NUMBER rsa_genprim(int len, int prob)
{
   rsa_NUMBER a_three,a_four;
   rsa_NUMBER prim;
   int i;

   a_add( &a_one, &a_two, &a_three );
   a_add( &a_two, &a_two, &a_four );

   /* This is done elsewhere to allow different initialization of
      rand seed (GGa - Sep 15, 2003) */
   /* init_rnd(); */

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
   int ii, jj;

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

   /* This is done elsewhere to allow different initialization of
      rand seed (GGa - Sep 15, 2003) */
   /* init_rnd(); */

   ii = 0;
   do {
      ii++;
      jj = 0;
      do {
         jj++;
         gen_number( len, d );
      } while (((a_cmp( d, max_p) <= 0 || a_cmp( d, &p1) >= 0)) && jj < kMAXT);

      a_ggt( d, &phi, e );
   } while ( a_cmp( e, &a_one) && ii < kMAXT);

   if (ii >= kMAXT || jj >= kMAXT)
      return  2;

   inv( d, &phi, e );

   return 0;

}

int rsa_encode_size(rsa_NUMBER n)
{
   // Returns length unit block of output

   return ( n_bitlen( &n) + 7) / 8;
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

   g_enc_siz = ( n_bitlen( &n) + 7) / 8;
   g_clear_siz = g_enc_siz -1;
   m_init( &n, rsa_NUM0P );

   pout = bufout;
   lout = 0;
   for ( i = 0; i < lin; i += g_clear_siz) {

      memcpy(buf,bufin+i,g_clear_siz);

      j = ((lin-i) < g_clear_siz) ? lin-i : g_clear_siz;
      memset(buf+j,0,(g_enc_siz-j));

      do_crypt( buf, buf, g_enc_siz, &e );

      memcpy(pout,buf,g_enc_siz);

      pout += g_enc_siz;
      lout += g_enc_siz;
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

   g_enc_siz = ( n_bitlen( &n) + 7) / 8;
   g_clear_siz = g_enc_siz -1;
   m_init( &n, rsa_NUM0P );

   pout = bufout;
   lout = 0;
   for ( i = 0; i < lin; i += g_enc_siz) {

      memcpy(buf,bufin+i,g_enc_siz);

      do_crypt( buf, buf, g_enc_siz, &e );

      memcpy(pout,buf,g_clear_siz);

      pout += g_clear_siz;
      lout += g_clear_siz;
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


static const char *gHEX="0123456789ABCDEF";
static const char *ghex="0123456789abcdef";

static rsa_NUMBER gbits[9];
static rsa_NUMBER gint16[16];

static int ginit = 0;

void num_init()
{
   int i;

   if (ginit) return;

   a_assign( &gbits[0], &a_one );
   for ( i=1; i<9; i++)
      a_add( &gbits[i-1], &gbits[i-1], &gbits[i] );

   a_assign( &gint16[0], &a_one );
   for ( i=1; i<16; i++)
      a_add( &gint16[i-1], &a_one, &gint16[i] );

   ginit = 1;
}


int rsa_num_sput(rsa_NUMBER *n, char *s, int l)
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
         *s++ = gHEX[ i ];
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

   if (! ginit)
      num_init();

   a_assign( &q, n);
   len = l;
   np = s + l;

   for (; q.n_len && len > 1; len --) {
      a_div( &q, &gbits[4], &q, &r );
      for (p=8, b=0, i=3; i >= 0; i--, p /= 2) {
         if ( a_cmp( &r, &gbits[i]) >= 0) {
            a_sub( &r, &gbits[i], &r );
            b += p;
         }
      }
      *--np = gHEX[ b ];
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


int rsa_num_fput(rsa_NUMBER *n, FILE *f)
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


int rsa_num_sget(rsa_NUMBER *n, char *s)
{
#if rsa_MAXINT == ( (1 << rsa_MAXBIT) - 1 )
   rsa_INT *p;
   const char *hp;
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
      if ( (hp = strchr( gHEX, *s )) )
         i = hp - gHEX;
      else if ((hp = strchr( ghex, *s )) )
         i = hp - ghex;
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

   if (! ginit)
      num_init();

   n->n_len = 0;
   while ( (c = *s++ & 0xFF)) {
      if ( p= strchr( gHEX, c) )
         i = p - gHEX;
      else if ( p= strchr( ghex, c) )
         i = p - ghex;
      else
         return(EOF);

      a_mult( n, &gbits[4], n );
      if (i)
         a_add( n, &gint16[i-1], n );
   }

   return(0);
#endif
}

int rsa_num_fget(rsa_NUMBER *n, FILE *f)
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

int rsa_cmp(rsa_NUMBER *c1, rsa_NUMBER *c2)
{
   int l;
   /* bei verschiedener Laenge klar*/
   if ( (l=c1->n_len) != c2->n_len)
      return( l - c2->n_len);

   /* vergleiche als arrays	*/
   return( n_cmp( c1->n_part, c2->n_part, l) );
}

void rsa_assign(rsa_NUMBER *d, rsa_NUMBER *s)
{
   int l;

   if (s == d)			/* nichts zu kopieren		*/
      return;

   if ((l=s->n_len))
      memcpy( d->n_part, s->n_part, sizeof(rsa_INT)*l);

   d->n_len = l;
}
