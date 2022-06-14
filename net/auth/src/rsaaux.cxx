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
*                                                                              *
*       Simple RSA public key code.                                            *
*       Adaptation in library for ROOT by G. Ganis, July 2003                  *
*       (gerardo.ganis@cern.ch)                                                *
*                                                                               *
*******************************************************************************/

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef WIN32
#  include <io.h>
typedef long off_t;
#else
#  include <unistd.h>
#  include <sys/time.h>
#endif

#include "rsaaux.h"
#include "rsalib.h"

/********************************************************************************
 *                                                                                *
 * arith.c                                                                      *
 *                                                                              *
 ********************************************************************************/

/*
 *   !!!!!!!!!!!!!!!!!!!!!!!!!!!! ACHTUNG !!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *   Es findet keinerlei Ueberpruefung auf Bereichsueberschreitung
 *   statt. Alle Werte muessen natuerliche Zahlen aus dem Bereich
 *      0 ... (rsa_MAXINT+1)^rsa_MAXLEN-1 sein.
 *
 *
 *   Bei keiner Funktion oder Hilsfunktion werden Annahmen getroffen,
 *   ueber die Verschiedenheit von Eingabe- & Ausgabe-Werten.
 *
 *
 *      Funktionen:
 *
 *   a_add( s1, s2, d )
 *      rsa_NUMBER *s1,*s2,*d;
 *         *d = *s1 + *s2;
 *
 *   a_assign( *d, *s )
 *      rsa_NUMBER *d,*s;
 *         *d = *s;
 *
 * int   a_cmp( c1, c2 )
 *      rsa_NUMBER *c1,*c2;
 *          1 :   falls *c1 >  *c2
 *          0 :   falls *c1 == *c2
 *         -1 :   falls *c1 <  *c2
 *
 *   a_div( d1, d2, q, r )
 *      rsa_NUMBER *d1,*d2,*q,*r;
 *         *q = *d1 / *d2 Rest *r;
 *
 *   a_div2( n )
 *      rsa_NUMBER *n;
 *         *n /= 2;
 *
 *   a_ggt( a, b, f )
 *      rsa_NUMBER *a,*b,*f;
 *         *f = ( *a, *b );
 *
 *   a_imult( n, m, d )
 *      rsa_NUMBER *n;
 *      rsa_INT m;
 *      rsa_NUMBER *d;
 *         *d = *n * m
 *
 *   a_mult( m1, m2, d )
 *      rsa_NUMBER *m1,*m2,*d;
 *         *d = *m1 * *m2;
 *
 *   a_sub( s1, s2, d )
 *      rsa_NUMBER *s1,*s2,*d;
 *         *d = *s1 - *s2;
 *
 *      Modulare Funktionen
 *   m_init( n, o )
 *      rsa_NUMBER *n,*o;
 *         Initialsierung der Modularen Funktionen
 *         o != 0 : *o = alter Wert
 *
 *   m_add( s1, s2, d )
 *      rsa_NUMBER *s1, *s2, *d;
 *         *d = *s1 + *s2;
 *
 *   m_mult( m1, m2, d )
 *      rsa_NUMBER *m1,*m2,*d;
 *
 *   m_exp( x, n, z )
 *      rsa_NUMBER *x,*n,*z;
 *         *z = *x exp *n;
 *
 *
 *      Hilfs-Funktionen:
 *
 * int   n_bits( n, b )
 *      rsa_NUMBER *n;
 *      int b;
 *         return( unterste b Bits der Dualdarstellung von n)
 *
 *   n_div( d1, z2, q, r )
 *      rsa_NUMBER *d1,z2[rsa_MAXBIT],*q,*r;
 *         *q = *d1 / z2[0] Rest *r;
 *         z2[i] = z2[0] * 2^i,  i=0..rsa_MAXBIT-1
 *
 * int   n_cmp( i1, i2, l )
 *      rsa_INT i1[l], i2[l];
 *          1 :   falls i1 >  i2
 *          0 :   falls i1 == i2
 *         -1 :   falls i1 <  i2
 *
 * int   n_mult( n, m, d, l)
 *      rsa_INT n[l], m, d[];
 *         d = m * n;
 *         return( sizeof(d) ); d.h. 'l' oder 'l+1'
 *
 * int   n_sub( p1, p2, p3, l, lo )
 *      rsa_INT p1[l], p2[lo], p3[];
 *         p3 = p1 - p2;
 *         return( sizeof(p3) ); d.h. '<= min(l,lo)'
 *
 * int   n_bitlen( n )
 *       rsa_NUMBER *n;
 *         return( sizeof(n) in bits )
 *
 */


////////////////////////////////////////////////////////////////////////////////
/// rand() implementation using /udev/random or /dev/random, if available

static int aux_rand()
{
#ifndef WIN32
   int frnd = open("/dev/urandom", O_RDONLY);
   if (frnd < 0) frnd = open("/dev/random", O_RDONLY);
   int r;
   if (frnd >= 0) {
      ssize_t rs = read(frnd, (void *) &r, sizeof(int));
      close(frnd);
      if (r < 0) r = -r;
      if (rs == sizeof(int)) return r;
   }
   printf("+++ERROR+++ : aux_rand: neither /dev/urandom nor /dev/random are available or readable!\n");
   struct timeval tv;
   if (gettimeofday(&tv,0) == 0) {
      int t1, t2;
      memcpy((void *)&t1, (void *)&tv.tv_sec, sizeof(int));
      memcpy((void *)&t2, (void *)&tv.tv_usec, sizeof(int));
      r = t1 + t2;
      if (r < 0) r = -r;
      return r;
   }
   return -1;
#else
   // No special random device available: use rand()
   return rand();
#endif
}

/*
 * Konstante 1, 2
 */
rsa_NUMBER a_one = {
   1,
   { (rsa_INT)1, },
};

rsa_NUMBER a_two = {
#if rsa_MAXINT == 1
   2,
   { 0, (rsa_INT)1, },
#else
   1,
   { (rsa_INT)2, },
#endif
};


/*
 * Vergleiche zwei rsa_INT arrays der Laenge l
 */
int n_cmp(rsa_INT *i1, rsa_INT *i2, int l)
{
   i1 += (l-1);         /* Pointer ans Ende      */
   i2 += (l-1);

   for (;l--;)
      if ( *i1-- != *i2-- )
         return( i1[1] > i2[1] ? 1 : -1 );

   return(0);
}

/*
 * Vergleiche zwei rsa_NUMBER
 */
int a_cmp(rsa_NUMBER *c1, rsa_NUMBER *c2)
{
   int l;
   /* bei verschiedener Laenge klar*/
   if ( (l=c1->n_len) != c2->n_len)
      return( l - c2->n_len);

   /* vergleiche als arrays   */
   return( n_cmp( c1->n_part, c2->n_part, l) );
}

/*
 * Zuweisung einer rsa_NUMBER (d = s)
 */
void a_assign(rsa_NUMBER *d, rsa_NUMBER *s)
{
   int l;

   if (s == d)         /* nichts zu kopieren      */
      return;

   if ((l=s->n_len))
      memcpy( d->n_part, s->n_part, sizeof(rsa_INT)*l);

   d->n_len = l;
}

/*
 * Addiere zwei rsa_NUMBER (d = s1 + s2)
 */
void a_add(rsa_NUMBER *s1, rsa_NUMBER *s2, rsa_NUMBER *d)
{
   int l,lo,ld,same;
   rsa_LONG sum;
   rsa_INT *p1,*p2,*p3;
   rsa_INT b;

   /* setze s1 auch die groessere Zahl   */
   l = s1->n_len;
   if ( (l=s1->n_len) < s2->n_len) {
      rsa_NUMBER *tmp = s1;

      s1 = s2;
      s2 = tmp;

      l = s1->n_len;
   }

   ld = l;
   lo = s2->n_len;
   p1 = s1->n_part;
   p2 = s2->n_part;
   p3 = d->n_part;
   same = (s1 == d);
   sum = 0;

   while (l --) {
      if (lo) {      /* es ist noch was von s2 da   */
         lo--;
         b = *p2++;
      }
      else
         b = 0;      /* ansonten 0 nehmen      */

      sum += (rsa_LONG)*p1++ + (rsa_LONG)b;
      *p3++ = rsa_TOINT(sum);

      if (sum > (rsa_LONG)rsa_MAXINT) {   /* carry merken      */
         sum = 1;
      }
      else
         sum = 0;

      if (!lo && same && !sum)   /* nichts mehr zu tuen   */
         break;
   }

   if (sum) {      /* letztes carry beruecksichtigen   */
      ld++;
      *p3 = sum;
   }

   d->n_len = ld;         /* Laenge setzen      */
}

/*
 * Subtrahiere zwei rsa_INT arrays. return( Laenge Ergebniss )
 * l == Laenge p1
 * lo== Laenge p3
 */
int n_sub(rsa_INT *p1, rsa_INT *p2, rsa_INT *p3, int l, int lo)
{
   int ld,lc,same;
   int over = 0;
   rsa_LONG dif;
   rsa_LONG a,b;

   same = (p1 == p3);         /* frueher Abbruch moeglich */

   for (lc=1, ld=0; l--; lc++) {
      a = (rsa_LONG)*p1++;
      if (lo) {         /* ist noch was von p2 da ? */
         lo--;
         b = (rsa_LONG)*p2++;
      }
      else
         b=0;         /* ansonten 0 nehmen   */

      if (over)         /* frueherer Overflow   */
         b++;
      if ( b > a) {         /* jetzt Overflow ?   */
         over = 1;
         dif = (rsa_MAXINT +1) + a;
      }
      else {
         over = 0;
         dif = a;
      }
      dif -= b;
      *p3++ = (rsa_INT)dif;

      if (dif)         /* Teil != 0 : Laenge neu */
         ld = lc;
      if (!lo && same && !over) {   /* nichts mehr zu tuen   */
         if (l > 0)      /* Laenge korrigieren   */
            ld = lc + l;
         break;
      }
   }

   return( ld );
}

/*
 * Subtrahiere zwei rsa_NUMBER (d= s1 - s2)
 */
void a_sub(rsa_NUMBER *s1, rsa_NUMBER *s2, rsa_NUMBER *d)
{
   d->n_len = n_sub( s1->n_part, s2->n_part, d->n_part
                     ,s1->n_len, s2->n_len );
}

/*
 * Mulitipliziere rsa_INT array der Laenge l mit einer rsa_INT (d = n * m)
 * return neue Laenge
 */
int n_mult(rsa_INT *n, rsa_INT m, rsa_INT *d, int l)
{
   int i;
   rsa_LONG mul;

   for (i=l,mul=0; i; i--) {
      mul += (rsa_LONG)m * (rsa_LONG)*n++;
      *d++ = rsa_TOINT(mul);
      mul  = rsa_DIVMAX1( mul );
   }

   if (mul) {      /* carry  ? */
      l++;
      *d = mul;
   }

   return( l );
}

/*
 * Mulitipliziere eine rsa_NUMBER mit einer rsa_INT (d = n * m)
 */
void a_imult(rsa_NUMBER *n, rsa_INT m, rsa_NUMBER *d)
{
   if (m == 0)
      d->n_len=0;
   else if (m == 1)
      a_assign( d, n );
   else
      d->n_len = n_mult( n->n_part, m, d->n_part, n->n_len );
}

/*
 * Multipliziere zwei rsa_NUMBER (d = m1 * m2)
 */
void a_mult(rsa_NUMBER *m1, rsa_NUMBER *m2, rsa_NUMBER *d)
{
   static rsa_INT id[ rsa_MAXLEN ];      /* Zwischenspeicher   */
   rsa_INT *vp;          /* Pointer darin   */
   rsa_LONG sum;         /* Summe fuer jede Stelle */
   rsa_LONG tp1;         /* Zwischenspeicher fuer m1 */
   rsa_INT *p2;
   rsa_INT *p1;
   int l1,l2,ld,lc,l,i,j;

   l1 = m1->n_len;
   l2 = m2->n_len;
   l = l1 + l2;
   if (l >= rsa_MAXLEN)
      abort();

   for (i=l, vp=id; i--;)
      *vp++ = 0;

   /* ohne Uebertrag in Zwischenspeicher multiplizieren */
   for ( p1 = m1->n_part, i=0; i < l1 ; i++, p1++) {

      tp1 = (rsa_LONG)*p1;
      vp = &id[i];
      sum = 0;
      for ( p2 = m2->n_part, j = l2; j--;) {
         sum += (rsa_LONG)*vp + (tp1 * (rsa_LONG)*p2++);
         *vp++ = rsa_TOINT( sum );
         sum = rsa_DIVMAX1(sum);
      }
      *vp++ += (rsa_INT)sum;
   }

   /* jetzt alle Uebertraege beruecksichtigen   */
   ld = 0;
   for (lc=0, vp=id, p1=d->n_part; lc++ < l;) {
      if ( (*p1++ = *vp++))
         ld = lc;
   }

   d->n_len = ld;
}


/*
 * Dividiere Zwei rsa_NUMBER mit Rest (q= d1 / z2[0] Rest r)
 * z2[i] = z2[0] * 2^i,  i=0..rsa_MAXBIT-1
 * r = 0 : kein Rest
 * q = 0 : kein Quotient
 */
void n_div(rsa_NUMBER *d1, rsa_NUMBER *z2, rsa_NUMBER *q, rsa_NUMBER *r)
{
   static   rsa_NUMBER dummy_rest;  /* Dummy Variable, falls r = 0 */
   static   rsa_NUMBER dummy_quot;  /* Dummy Variable, falla q = 0 */
   rsa_INT *i1,*i1e,*i3;
   int l2,ld,l,lq;
#if rsa_MAXINT != 1
   rsa_INT z;
   int pw,l2t;
#endif

   if (!z2->n_len)
      abort();

   if (!r)
      r = &dummy_rest;
   if (!q)
      q = &dummy_quot;

   a_assign( r, d1 );   /* Kopie von d1 in den Rest      */

   l2= z2->n_len;      /* Laenge von z2[0]         */
   l = r->n_len - l2;   /* Laenge des noch ''rechts'' liegenden
                           Stuecks von d1         */
   lq= l +1;      /* Laenge des Quotienten      */
   i3= q->n_part + l;
   i1= r->n_part + l;
   ld = l2;      /* aktuelle Laenge des ''Vergleichsstuecks''
                           von d1            */
   i1e= i1 + (ld-1);

   for (; l >= 0; ld++, i1--, i1e--, l--, i3--) {
      *i3 = 0;

      if (ld == l2 && ! *i1e) {
         ld--;
         continue;
      }

      if ( ld > l2 || (ld == l2 && n_cmp( i1, z2->n_part, l2) >= 0) ) {
#if rsa_MAXINT != 1
         /* nach 2er-Potenzen zerlegen   */
         for (pw=rsa_MAXBIT-1, z=(rsa_INT)rsa_HIGHBIT; pw >= 0; pw--, z /= 2) {
            if ( ld > (l2t= z2[pw].n_len)
                 || (ld == l2t
                     && n_cmp( i1, z2[pw].n_part, ld) >= 0)) {
               ld = n_sub( i1, z2[pw].n_part, i1, ld, l2t );
               (*i3) += z;
            }
         }
#else
         /* bei rsa_MAXINT == 1 alles viel einfacher   */
         ld = n_sub( i1, z2->n_part, i1, ld, l2 );
         (*i3) ++;
#endif
      }
   }

   /* Korrektur, falls l von Anfang an Negativ war */
   l ++;
   lq -= l;
   ld += l;

   if (lq>0 && !q->n_part[lq -1])   /* evtl. Laenge korrigieren   */
      lq--;

   q->n_len = lq;
   r->n_len = ld -1;
}

/*
 * Dividiere Zwei rsa_NUMBER mit Rest (q= d1 / z2[0] Rest r)
 * z2[i] = z2[0] * 2^i,  i=0..rsa_MAXBIT-1
 * r = 0 : kein Rest
 * q = 0 : kein Quotient
 */
void a_div(rsa_NUMBER *d1, rsa_NUMBER *d2, rsa_NUMBER *q, rsa_NUMBER *r)
{
#if rsa_MAXINT != 1
   rsa_NUMBER z2[rsa_MAXBIT];
   rsa_INT z;
   int i;

   a_assign( &z2[0], d2 );
   for (i=1,z=2; i < rsa_MAXBIT; i++, z *= 2)
      a_imult( d2, z, &z2[i] );

   d2 = z2;
#endif

   n_div( d1, d2, q, r );
}

/*
 * Dividiere eine rsa_NUMBER durch 2
 */
void a_div2(rsa_NUMBER *n)
{
#if rsa_MAXBIT == rsa_LOWBITS
   rsa_INT *p;
   int i;

#if rsa_MAXINT != 1
   rsa_INT h;
   int c;

   c=0;
   i= n->n_len;
   p= &n->n_part[i-1];

   for (; i--;) {
      if (c) {
         c = (h= *p) & 1;
         h /= 2;
         h |= rsa_HIGHBIT;
      }
      else {
         c = (h= *p) & 1;
         h /= 2;
      }

      *p-- = h;
   }

   if ( (i= n->n_len) && n->n_part[i-1] == 0 )
      n->n_len = i-1;

#else  /* rsa_MAXBIT != 1 */
   p = n->n_part;
   i = n->n_len;

   if (i) {
      n->n_len = i-1;
      for (; --i ; p++)
         p[0] = p[1];
   }
#endif /* rsa_MAXBIT != 1 */
#else  /* rsa_MAXBIT == rsa_LOWBITS */
   a_div( n, &a_two, n, rsa_NUM0P );
#endif /* rsa_MAXBIT == rsa_LOWBITS */
}


/*
 *   MODULO-FUNKTIONEN
 */

static rsa_NUMBER g_mod_z2[ rsa_MAXBIT ];

/*
 * Init
 */
void m_init(rsa_NUMBER *n, rsa_NUMBER *o)
{
   rsa_INT z;
   int i;

   if (o)
      a_assign( o, &g_mod_z2[0] );

   if (! a_cmp( n, &g_mod_z2[0]) )
      return;

   for (i=0,z=1; i < rsa_MAXBIT; i++, z *= 2)
      a_imult( n, z, &g_mod_z2[i] );
}

void m_add(rsa_NUMBER *s1, rsa_NUMBER *s2, rsa_NUMBER *d)
{
   a_add( s1, s2, d );
   if (a_cmp( d, g_mod_z2) >= 0)
      a_sub( d, g_mod_z2, d );
}

void m_mult(rsa_NUMBER *m1, rsa_NUMBER *m2, rsa_NUMBER *d)
{
   a_mult( m1, m2, d );
   n_div( d, g_mod_z2, rsa_NUM0P, d );
}

/*
 * Restklassen Exponent
 */
void m_exp(rsa_NUMBER *x, rsa_NUMBER *n, rsa_NUMBER *z)
{
   rsa_NUMBER xt,nt;

   a_assign( &nt, n );
   a_assign( &xt, x );
   a_assign( z, &a_one );

   while (nt.n_len) {
      while ( ! (nt.n_part[0] & 1)) {
         m_mult( &xt, &xt, &xt );
         a_div2( &nt );
      }
      m_mult( &xt, z, z );
      a_sub( &nt, &a_one, &nt );
   }
}

/*
 * GGT
 */
void a_ggt(rsa_NUMBER *a, rsa_NUMBER *b, rsa_NUMBER *f)
{
   rsa_NUMBER t[2];
   int at,bt, tmp;

   a_assign( &t[0], a ); at= 0;
   a_assign( &t[1], b ); bt= 1;

   if ( a_cmp( &t[at], &t[bt]) < 0) {
      tmp= at; at= bt; bt= tmp;
   }
   /* euklidischer Algorithmus      */
   while ( t[bt].n_len) {
      a_div( &t[at], &t[bt], rsa_NUM0P, &t[at] );
      tmp= at; at= bt; bt= tmp;
   }

   a_assign( f, &t[at] );
}

/*
 * die untersten b bits der Dualdarstellung von n
 * die bits muessen in ein int passen
 */
int n_bits(rsa_NUMBER *n, int b)
{
   rsa_INT *p;
   int l;
   unsigned r;
   int m = (1<<b) -1;

   if ( n->n_len == 0)
      return(0);

   if (rsa_LOWBITS >= b)
      return( n->n_part[0] & m );

#if rsa_LOWBITS != 0
   l = (b-1) / rsa_LOWBITS;
#else
   l = n->n_len -1;
#endif
   for (p= &n->n_part[l],r=0; l-- >= 0 && b > 0; b-= rsa_LOWBITS, p--) {
      r  = rsa_MULMAX1( r );
      r += (unsigned)*p;
   }

   return( r & m );
}

/*
 * Anzahl der bits von n bei Dualdarstellung
 */
int n_bitlen(rsa_NUMBER *n)
{
   rsa_NUMBER b;
   int i;

   a_assign( &b, &a_one );

   for (i=0; a_cmp( &b, n) <= 0; a_mult( &b, &a_two, &b ), i++)
      ;

   return(i);
}


/*******************************************************************************
 *                                  *
 * prim.c                                                                       *
 *                                                                              *
 ********************************************************************************/

/*
 *      RSA
 *
 *   p,q prim
 *   p != q
 *   n = p*q
 *   phi = (p -1)*(q -1)
 *   e,d aus 0...n-1
 *   e*d == 1 mod phi
 *
 *   m aus 0...n-1 sei eine Nachricht
 *
 *   Verschluesseln:
 *      E(x) = x^e mod n      ( n,e oeffendlich )
 *
 *   Entschluesseln:
 *      D(x) = x^d mod n      ( d geheim )
 *
 *
 *   Sicherheit:
 *
 *      p,q sollten bei mind. 10^100 liegen.
 *      (d,phi) == 1, das gilt fuer alle Primzahlen > max(p,q).
 *      Allerdings sollte d moeglichst gross sein ( d < phi-1 )
 *      um direktes Suchen zu verhindern.
 */


/*
 *      FUNKTIONEN um RSA Schluessel zu generieren.
 *
 * int   p_prim( n, m )
 *      rsa_NUMBER *n;
 *      int m;
 *         0 : n ist nicht prim
 *         1 : n ist mit Wahrscheinlichkeit (1-1/2^m) prim
 *      ACHTUNG !!!!
 *      p_prim benutzt m_init
 *
 *   inv( d, phi, e )
 *      rsa_NUMBER *d,*phi,*e;
 *         *e = *d^-1 (mod phi)
 *      ACHTUNG !!!!
 *      p_prim benutzt m_init
 */

/*
 * Prototypes
 */
static int   jak_f( rsa_NUMBER* );
static int   jak_g( rsa_NUMBER*, rsa_NUMBER* );
static int   jakobi( rsa_NUMBER*, rsa_NUMBER* );

/*
 * Hilfs-Funktion fuer jakobi
 */
static int jak_f(rsa_NUMBER *n)
{
   int f,ret;

   f = n_bits( n, 3 );

   ret = ((f == 1) || (f == 7)) ? 1 : -1;

   return(ret);
}

/*
 * Hilfs-Funktuion fuer jakobi
 */
static int jak_g(rsa_NUMBER *a, rsa_NUMBER *n)
{
   int ret;

   if ( n_bits( n, 2) == 1 ||
        n_bits( a, 2) == 1 )
      ret = 1;
   else
      ret = -1;

   return(ret);
}

/*
 * Jakobi-Symbol
 */
static int jakobi(rsa_NUMBER *a, rsa_NUMBER *n)
{
   rsa_NUMBER t[2];
   int at,nt, ret;

   a_assign( &t[0], a ); at = 0;
   a_assign( &t[1], n ); nt = 1;

   /*
    * b > 1
    *
    * J( a, b) =
    * a == 1   :   1
    * a == 2   :   f(n)
    * a == 2*b   :   J(b,n)*J(2,n) ( == J(b,n)*f(n) )
    * a == 2*b -1   :   J(n % a,a)*g(a,n)
    *
    */

   ret = 1;
   while (1) {
      if (! a_cmp(&t[at],&a_one)) {
         break;
      }
      if (! a_cmp(&t[at],&a_two)) {
         ret *= jak_f( &t[nt] );
         break;
      }
      if ( ! t[at].n_len )      /* Fehler :-)   */
         abort();
      if ( t[at].n_part[0] & 1) {   /* a == 2*b -1   */
         int tmp;

         ret *= jak_g( &t[at], &t[nt] );
         a_div( &t[nt], &t[at], rsa_NUM0P, &t[nt] );
         tmp = at; at = nt; nt = tmp;
      }
      else {            /* a == 2*b   */
         ret *= jak_f( &t[nt] );
         a_div2( &t[at] );
      }

   }

   return(ret);
}

/*
 * Probabilistischer Primzahltest
 *
 *  0 -> n nicht prim
 *  1 -> n prim mit  (1-1/2^m) Wahrscheinlichkeit.
 *
 *   ACHTUNG !!!!!!
 *   p_prim benutzt m_init !!
 *
 */
int p_prim(rsa_NUMBER *n, int m)
{
   rsa_NUMBER gt,n1,n2,a;
   rsa_INT *p;
   int i,w,j;

   if (a_cmp(n,&a_two) <= 0 || m <= 0)
      abort();

   a_sub( n, &a_one, &n1 );   /* n1 = -1    mod n      */
   a_assign( &n2, &n1 );
   a_div2( &n2 );         /* n2 = ( n -1) / 2      */

   m_init( n, rsa_NUM0P );

   w = 1;
   for (; w && m; m--) {
      /* ziehe zufaellig a aus 2..n-1      */
      do {
         for (i=n->n_len-1, p=a.n_part; i; i--)
            *p++ = (rsa_INT)aux_rand();
         if ((i=n->n_len) )
            *p = (rsa_INT)( aux_rand() % ((unsigned long)n->n_part[i-1] +1) );
         while ( i && ! *p )
            p--,i--;
         a.n_len = i;
      } while ( a_cmp( &a, n) >= 0 || a_cmp( &a, &a_two) < 0 );

      /* jetzt ist a fertig         */

      /*
       * n ist nicht prim wenn gilt:
       *   (a,n) != 1
       * oder
       *   a^( (n-1)/2) != J(a,n)   mod n
       *
       */

      a_ggt( &a, n, &gt );
      if ( a_cmp( &gt, &a_one) == 0) {

         j= jakobi( &a, n );
         m_exp( &a, &n2, &a );

         if  (   ( a_cmp( &a, &a_one) == 0 && j ==  1 )
                 || ( a_cmp( &a, &n1   ) == 0 && j == -1) )
            w = 1;
         else
            w = 0;
      }
      else
         w = 0;
   }

   return( w );
}

/*
 * Berechne mulitiplikatives Inverses zu d (mod phi)
 *   d relativ prim zu phi ( d < phi )
 *   d.h. (d,phi) == 1
 *
 *   ACHTUNG !!!!
 *   inv benutzt m_init
 */
void inv(rsa_NUMBER *d, rsa_NUMBER *phi, rsa_NUMBER *e)
{
   int k, i0, i1, i2;
   rsa_NUMBER r[3],p[3],c;

   /*
    * Berlekamp-Algorithmus
    *   ( fuer diesen Spezialfall vereinfacht )
    */

   if (a_cmp(phi,d) <= 0)
      abort();

   m_init( phi, rsa_NUM0P );

   p[1].n_len = 0;
   a_assign( &p[2], &a_one );
   a_assign( &r[1], phi );
   a_assign( &r[2], d );

   k = -1;
   do {
      k++;
      i0=k%3; i1=(k+2)%3; i2=(k+1)%3;
      a_div( &r[i2], &r[i1], &c, &r[i0] );
      m_mult( &c, &p[i1], &p[i0] );
      m_add( &p[i0], &p[i2], &p[i0] );
   } while (r[i0].n_len);

   if ( a_cmp( &r[i1], &a_one) )   /* r[i1] == (d,phi) muss 1 sein   */
      abort();

   if ( k & 1 )   /* falsches ''Vorzeichen''   */
      a_sub( phi, &p[i1], e );
   else
      a_assign( e, &p[i1] );
}


/*******************************************************************************
 *                                  *
 * rnd.c                                                                       *
 *                                                                              *
 ********************************************************************************/

void gen_number(int len, rsa_NUMBER *n)
{
   const char *hex = "0123456789ABCDEF" ;
   char num[ rsa_STRLEN +1 ];
   char *p;
   int i,l;

   p=&num[ sizeof(num) -1];
   *p-- = '\0';

   for (l=len; l--; p--) {
      i = aux_rand() % 16;
      *p = hex[ i ];
   }
   p++;

   while (len-- && *p == '0')
      p++;

   rsa_num_sget( n, p );
}

void init_rnd()
{
   const char *randdev = "/dev/urandom";

   int fd;
   unsigned int seed;
   if ((fd = open(randdev, O_RDONLY)) != -1) {
      if (read(fd, &seed, sizeof(seed))) {;}
      close(fd);
   } else {
      seed = (unsigned int)time(0);   //better use times() + win32 equivalent
   }
   srand( seed );
}


/*******************************************************************************
 *                                  *
 * aux.c                                                                       *
 *                                                                              *
 ********************************************************************************/

/* These are not needed, for the moment

int get_clear(char *p, FILE *fp)
{
int n;

n = fread( p, 1, clear_siz, fp );

if (n <= 0)
return(0);

memset( p + n, 0, enc_siz - n );

return(1);
}

int get_enc(char *p, FILE *fp)
{
   int n;

   n = fread( p, 1, enc_siz, fp );

   if (n != enc_siz)
      return(0);

   return(1);
}

int put_clear(char *p, FILE *fp)
{
   int n;

   n = fwrite( p, 1, clear_siz, fp );

   if (n != clear_siz)
      return(0);

   return(1);
}

int put_enc(char *p, FILE *fp)
{
   int n;

   n = fwrite( p, 1, enc_siz, fp );

   if (n != enc_siz)
      return(0);

   return(1);
}

*/

void do_crypt(char *s, char *d, int len, rsa_NUMBER *e)
{
   static char hex[] = "0123456789ABCDEF";
   rsa_NUMBER n;
   char buf[ rsa_STRLEN + 1 ];
   char *ph;
   int i,c;

   ph = buf + rsa_STRLEN - 1;
   ph[1] = '\0';

   for (i=len; i; i--) {
      c = *s++;
      *ph-- = hex[ (c >> 4) & 0xF ];
      *ph-- = hex[ c & 0xF ];
   }
   ph++;

   rsa_num_sget( &n, ph );

   m_exp( &n, e, &n );

   rsa_num_sput( &n, buf, rsa_STRLEN +1 );

   ph = buf + (i=strlen(buf)) -1;

   for (; len; len--) {
      if (i-- > 0) {
         c = (strchr( hex, *ph) - hex) << 4;
         ph--;
      }
      else
         c=0;
      if (i-- > 0) {
         c |= strchr( hex, *ph) - hex;
         ph--;
      }

      *d++ = c;
   }
}

