// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// The set of methods to work with the plain matrix / vector
// "derived" from  http://wwwinfo.cern.ch/asdoc/shortwrupsdir/f110/top.html
// "derived" from  http://wwwinfo.cern.ch/asdoc/shortwrupsdir/f112/top.html
//
// Revision 1.7  2006/05/21 18:05:26  brun
// Fix more coding conventions violations
//
// Revision 1.6  2006/05/20 14:06:09  brun
// Fix a VERY long list of coding conventions violations
//
// Revision 1.5  2003/09/30 09:52:49  brun
// Add references to the original CERNLIB packages
//
// Revision 1.4  2003/05/28 15:17:03  brun
// From Valeri Fine. A new version of the table package.
// It fixes a couple of memory leaks:
//  class TTableDescriptorm
//  class TVolumePosition
// and provides some clean up
// for the TCL class interface.
//
// Revision 1.3  2003/04/03 17:39:39  fine
// Make merge with ROOT 3.05.03 and add TR package
//122
// Revision 1.2  2003/02/04 23:35:20  fine
// Clean up
//
// Revision 1.1  2002/04/15 20:23:39  fine
// NEw naming schema for RootKErnel classes and a set of classes to back geometry OO
//
// Revision 1.2  2001/05/29 19:08:08  brun
// New version of some STAR classes from Valery.
//
// Revision 1.2  2001/05/27 02:38:14  fine
// New method trsedu to solev Ax=B from Victor
//
// Revision 1.1.1.1  2000/11/27 22:57:14  fisyak
//
//
// Revision 1.1.1.1  2000/05/16 17:00:48  rdm
// Initial import of ROOT into CVS
//
////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include "TCernLib.h"
#include "TMath.h"
#include "TArrayD.h"
#include "TError.h"

ClassImp(TCL)

#define TCL_MXMAD(n_,a,b,c,i,j,k)                       \
    /* Local variables */                                \
    int l, m, n, ia, ic, ib, ja, jb, iia, iib, ioa, iob; \
                                                         \
    /* Parameter adjuTments */                          \
    --a;  --b;  --c;                                     \
    /* Function Body */                                  \
/*                      MXMAD MXMAD1 MXMAD2 MXMAD3 MXMPY MXMPY1 MXMPY2 MXMPY3 MXMUB MXMUB1 MXMUB2 MXMUB3 */ \
/*  const int iandj1[] = {21,   22,    23,    24,   11,    12,    13,    14,    31,   32,   33,    34 }; */ \
    const int iandj1[] = {2,    2 ,    2 ,    2 ,   1 ,    1 ,    1 ,    1 ,    3 ,   3 ,   3 ,    3  }; \
    const int iandj2[] = { 1,    2,     3,     4,    1,     2,     3,     4,     1,    2,    3,     4 }; \
    int n1 = iandj1[n_];                                  \
    int n2 = iandj2[n_];                                  \
    if (i == 0 || k == 0) return 0;                       \
                                                          \
    switch (n2) {                                         \
      case 1: iia = 1; ioa = j; iib = k; iob = 1; break;  \
      case 2: iia = 1; ioa = j; iib = 1; iob = j; break;  \
      case 3: iia = i; ioa = 1; iib = k; iob = 1; break;  \
      case 4: iia = i; ioa = 1; iib = 1; iob = j; break;  \
      default: iia = ioa = iib = iob = 0; assert(iob);    \
    };                                                    \
                                                          \
    ia = 1; ic = 1;                                       \
    for (l = 1; l <= i; ++l) {                            \
            ib = 1;                                           \
            for (m = 1; m <= k; ++m,++ic) {                   \
              switch (n1) {                                   \
                      case 1:  c[ic] = 0.;      break;            \
                      case 3:  c[ic] = -c[ic];  break;            \
              };                                              \
              if (j == 0) continue;                           \
              ja = ia; jb = ib;                               \
          double cic = c[ic];                             \
              for (n = 1; n <= j; ++n, ja+=iia, jb+=iib)      \
                       cic += a[ja] * b[jb];                      \
          c[ic] = cic;                                    \
              ib += iob;                                      \
            }                                                 \
            ia += ioa;                                        \
    }

//___________________________________________________________________________
float *TCL::mxmad_0_(int n_, const float *a, const float *b, float *c, int i, int j, int k)
{
  TCL_MXMAD(n_,a,b,c,i,j,k)
  return c;
} /* mxmad_ */

//___________________________________________________________________________
double *TCL::mxmad_0_(int n_, const double *a, const double *b, double *c, int i, int j, int k)
{
   TCL_MXMAD(n_,a,b,c,i,j,k)
   return c;
} /* mxmad_ */

#undef TCL_MXMAD

//___________________________________________________________________________
//
//             Matrix Multiplication
//___________________________________________________________________________

#define TCL_MXMLRT( n__, a, b, c,  ni,nj) \
  if (ni <= 0 || nj <= 0) return 0;        \
  double x;                                \
  int ia, ib, ic, ja, kc, ii, jj, kj, ki, ia1, ib1, ic1, ja1; \
  int ipa = 1;  int jpa = nj;              \
  if (n__ == 1) { ipa = ni;  jpa = 1; }    \
                                           \
  --a;  --b;  --c;                         \
                                           \
  ic1 = 1;  ia1 = 1;                       \
  for (ii = 1; ii <= ni; ++ii, ic1+=ni, ia1+=jpa) { \
    ic = ic1;                                       \
    for (kc = 1; kc <= ni; ++kc,ic++) c[ic] = 0.;   \
    ib1 = 1;  ja1 = 1;                              \
    for (jj = 1; jj <= nj; ++jj,++ib1,ja1 += ipa) { \
      ib = ib1;  ia = ia1;                          \
      x = 0.;                                       \
      for (kj = 1;kj <= nj;++kj,ia+=ipa,ib += nj)   \
                    x += a[ia] * b[ib];                     \
      ja = ja1;  ic = ic1;                          \
      for (ki = 1; ki <= ni; ++ki,++ic,ja += jpa)   \
                    c[ic] += x * a[ja];                     \
    }                                               \
  }

//___________________________________________________________________________
float *TCL::mxmlrt_0_(int n__, const float *a, const float *b, float *c, int ni,int nj)
{
 // Matrix Multiplication
 // CERN PROGLIB# F110    MXMLRT          .VERSION KERNFOR  2.00  720707
 // ORIG. 01/01/64 RKB
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f110/top.html">F110</A>
 <!--*/
 // -->END_HTML


// --      ENTRY MXMLRT */
// --                C = A(I,J) X B(J,J) X A*(J,I) */
// --                A* TANDS FOR A-TRANSPOSED */
//             mxmlrt (A,B,C,NI,NJ)     IS EQUIVALENT TO */
//             CALL MXMPY (A,B,X,NI,NJ,NJ) */
//             CALL MXMPY1 (X,A,C,NI,NJ,NI) */

/*        OR   CALL MXMPY1 (B,A,Y,NJ,NJ,NI) */
/*             CALL MXMPY (A,Y,C,NI,NJ,NI) */


// --                C = A*(I,J) X B(J,J) X A(J,I)

//        CALL MXMLTR (A,B,C,NI,NJ)     IS EQUIVALENT TO
//             CALL MXMPY2 (A,B,X,NI,NJ,NJ)
//             CALL MXMPY (X,A,C,NI,NJ,NI)

//        OR   CALL MXMPY (B,A,Y,NJ,NJ,NI)
//             CALL MXMPY2 (A,Y,C,NI,NJ,NI)
   TCL_MXMLRT( n__, a, b, c,  ni,nj)
   return c;
} /* mxmlrt_ */

//___________________________________________________________________________
double *TCL::mxmlrt_0_(int n__, const double *a, const double *b, double *c, int ni,int nj)
{
 // Matrix Multiplication (double precision)

   TCL_MXMLRT( n__, a, b, c,  ni,nj)
   return c;

} /* mxmlrt_ */

#undef TCL_MXMLRT

//___________________________________________________________________________
//
//             Matrix Transposition
//___________________________________________________________________________

#define TCL_MXTRP(a, b, i, j)     \
  if (i == 0 || j == 0) return 0; \
  --b;  --a;                      \
  int ib = 1;                     \
  for (int k = 1; k <= j; ++k)    \
  { int ia = k;                   \
    for (int l = 1; l <= i; ++l,ia += j,++ib) b[ib] = a[ia]; }

//___________________________________________________________________________
float *TCL::mxtrp(const float *a, float *b, int i, int j)
{
//
//  Matrix Transposition
// CERN PROGLIB# F110    MXTRP           .VERSION KERNFOR  1.0   650809
// ORIG. 01/01/64 RKB
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f110/top.html">F110</A>
 <!--*/
 // -->END_HTML

   TCL_MXTRP(a, b, i, j)
   return b;
} /* mxtrp */

//___________________________________________________________________________
double *TCL::mxtrp(const double *a, double *b, int i, int j)
{
//  Matrix Transposition (double precision)
// CERN PROGLIB# F110    MXTRP           .VERSION KERNFOR  1.0   650809
// ORIG. 01/01/64 RKB
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f110/top.html">F110</A>
 <!--*/
 // -->END_HTML

   TCL_MXTRP(a, b, i, j)
   return b;

} /* mxtrp */
#undef TCL_MXTRP

//___________________________________________________________________________
//___________________________________________________________________________
//
//            TRPACK
//___________________________________________________________________________
//___________________________________________________________________________

#define TCL_TRAAT(a, s, m, n)           \
   /* Local variables */                \
   int ipiv, i, j, ipivn, ia, is, iat;  \
   double sum;                          \
   --s;    --a;                         \
   ia = 0;   is = 0;                    \
   for (i = 1; i <= m; ++i) {           \
     ipiv = ia;                         \
     ipivn = ipiv + n;                  \
     iat = 0;                           \
     for (j = 1; j <= i; ++j) {         \
       ia = ipiv;                       \
       sum = 0.;                        \
       do {                             \
         ++ia;  ++iat;                  \
         sum += a[ia] * a[iat];         \
       } while (ia < ipivn);            \
       ++is;                            \
       s[is] = sum;                     \
     }                                  \
   }                                    \
   s++;


//____________________________________________________________
float *TCL::traat(const float *a, float *s, int m, int n)
{
   //
   // Symmetric Multiplication of Rectangular Matrices
   // CERN PROGLIB# F112    TRAAT           .VERSION KERNFOR  4.15  861204
   // ORIG. 18/12/74 WH */
   // traat.F -- translated by f2c (version 19970219).
   //
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRAAT(a, s, m, n)
   return s;
} /* traat_ */

//____________________________________________________________
double *TCL::traat(const double *a, double *s, int m, int n)
{
   //  Symmetric Multiplication of Rectangular Matrices
   // CERN PROGLIB# F112    TRAAT           .VERSION KERNFOR  4.15  861204
   // ORIG. 18/12/74 WH */
   // traat.F -- translated by f2c (version 19970219).
   //
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRAAT(a, s, m, n)
   return s;
} /* traat_ */

#undef TCL_TRAAT

#define TCL_TRAL(a, u, b, m,  n)   \
   int indu, i, j, k, ia, ib, iu;  \
   double sum;                     \
   --b;    --u;    --a;            \
   ib = 1;                         \
   for (i = 1; i <= m; ++i) {      \
      indu = 0;                    \
      for (j = 1; j <= n; ++j) {   \
         indu += j;                \
         ia = ib;                  \
         iu = indu;                \
         sum = 0.;                 \
         for (k = j; k <= n; ++k) {\
            sum += a[ia] * u[iu];  \
            ++ia;                  \
            iu += k;               \
         }                         \
         b[ib] = sum;              \
         ++ib;                     \
      }                            \
   }                               \
   b++;

//____________________________________________________________
float *TCL::tral(const float *a, const float *u, float *b, int m, int n)
{
   // Triangular - Rectangular Multiplication
   // CERN PROGLIB# F112    TRAL            .VERSION KERNFOR  4.15  861204
   // ORIG. 18/12/74 WH
   // tral.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRAL(a, u, b, m,  n)
   return b;
} /* tral_ */

//____________________________________________________________
double *TCL::tral(const double *a, const double *u, double *b, int m, int n)
{
   // Triangular - Rectangular Multiplication
   // tral.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRAL            .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRAL(a, u, b, m,  n)
   return b;
} /* tral_ */

#undef TCL_TRAL

//____________________________________________________________
#define TCL_TRALT(a, u, b, m, n)  \
   int indu, j, k, ia, ib, iu;    \
   double sum;                    \
   --b;    --u;    --a;           \
   ib = m * n;                    \
   indu = (n * n + n) / 2;        \
   do {                           \
      iu = indu;                  \
      for (j = 1; j <= n; ++j) {  \
         ia = ib;                 \
         sum = 0.;                \
        for (k = j; k <= n; ++k) {\
           sum += a[ia] * u[iu];  \
           --ia;   --iu;          \
        }                         \
        b[ib] = sum;              \
        --ib;                     \
      }                           \
   } while (ib > 0);              \
   ++b;

//____________________________________________________________
float *TCL::tralt(const float *a, const float *u, float *b, int m, int n)
{
   // Triangular - Rectangular Multiplication
   // CERN PROGLIB# F112    TRALT           .VERSION KERNFOR  4.15  861204
   // ORIG. 18/12/74 WH
   // tralt.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRALT(a, u, b, m, n)
   return b;
} /* tralt_ */

//____________________________________________________________
double *TCL::tralt(const double *a, const double *u, double *b, int m, int n)
{
   // Triangular - Rectangular Multiplication
   // CERN PROGLIB# F112    TRALT           .VERSION KERNFOR  4.15  861204
   // ORIG. 18/12/74 WH
   // tralt.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRALT(a, u, b, m, n)
   return b;
} /* tralt_ */

#undef TCL_TRALT

//____________________________________________________________

#define TCL_TRAS(a, s, b, m, n)     \
   int inds, i__, j, k, ia, ib, is; \
   double sum;                      \
   --b;    --s;    --a;             \
   ib = 0; inds = 0; i__ = 0;       \
   do {                             \
      inds += i__;                  \
      ia = 0;                       \
      ib = i__ + 1;                 \
      for (j = 1; j <= m; ++j) {    \
         is = inds;                 \
         sum = 0.;                  \
         k = 0;                     \
         do {                       \
            if (k > i__) is += k;   \
            else        ++is;       \
            ++ia;                   \
            sum += a[ia] * s[is];   \
            ++k;                    \
         } while (k < n);           \
         b[ib] = sum;               \
         ib += n;                   \
      }                             \
      ++i__;                        \
   } while (i__ < n);               \
   ++b;

//____________________________________________________________
float *TCL::tras(const float *a, const float *s, float *b, int m, int n)
{
   // Symmetric - Rectangular Multiplication
   // CERN PROGLIB# F112    TRAS            .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
   // tras.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRAS(a, s, b, m, n)
   return b;
} /* tras_ */

//____________________________________________________________
double *TCL::tras(const double *a, const double *s, double *b, int m, int n)
{
   // Symmetric - Rectangular Multiplication
   // CERN PROGLIB# F112    TRAS            .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
   // tras.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRAS(a, s, b, m, n)
   return b;
} /* tras_ */

#undef TCL_TRAS


//____________________________________________________________
#define TCL_TRASAT(a, s, r__, m, n) \
   int imax,  k;                    \
   int ia, mn, ir, is, iaa;         \
   double sum;                      \
   --r__;    --s;    --a;           \
   imax = (m * m + m) / 2;          \
   vzero(&r__[1], imax);            \
   mn = m * n;                      \
   int ind = 0;                     \
   int i__ = 0;                     \
   do {                             \
      ind += i__;                   \
      ia = 0; ir = 0;               \
      do {                          \
         is = ind;                  \
         sum = 0.;   k = 0;         \
         do {                       \
            if (k > i__) is += k;   \
            else         ++is;      \
            ++ia;                   \
            sum += s[is] * a[ia];   \
            ++k;                    \
         } while (k < n);           \
         iaa = i__ + 1;             \
         do {                       \
            ++ir;                   \
            r__[ir] += sum * a[iaa];\
            iaa += n;               \
         } while (iaa <= ia);       \
      } while (ia < mn);            \
      ++i__;                        \
   } while (i__ < n);               \
   ++r__;

//____________________________________________________________
float *TCL::trasat(const float *a, const float *s, float *r__, int m, int n)
{
   // Transformation of Symmetric Matrix
   // CERN PROGLIB# F112    TRASAT          .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
   // trasat.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRASAT(a, s, r__, m, n)
   return r__;
} /* trasat_ */

//____________________________________________________________
double *TCL::trasat(const double *a, const double *s, double *r__, int m, int n)
{
   // Transformation of Symmetric Matrix
   // CERN PROGLIB# F112    TRASAT          .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
   // trasat.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRASAT(a, s, r__, m, n)
   return r__;
} /* trasat_ */

//____________________________________________________________
float *TCL::trasat(const double *a, const float *s, float *r__, int m, int n)
{
   // Transformation of Symmetric Matrix
   // CERN PROGLIB# F112    TRASAT          .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
   // trasat.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   TCL_TRASAT(a, s, r__, m, n)
   return r__;
} /* trasat_ */

#undef TCL_TRASAT

//____________________________________________________________
float *TCL::trata(const float *a, float *r__, int m, int n)
{
   // trata.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRATA           .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int i__, j, ia, mn, ir, iat;
   double sum;

   /* Parameter adjuTments */
   --r__;    --a;

   /* Function Body */
   mn = m * n;
   ir = 0;

   for (i__ = 1; i__ <= m; ++i__) {
      for (j = 1; j <= i__; ++j) {
         ia = i__;
         iat = j;
         sum = 0.;
         do {
            sum += a[ia] * a[iat];
            ia +=  m;
            iat += m;
         } while  (ia <= mn);
         ++ir;
         r__[ir] = sum;
      }
   }
   ++r__;
   return r__;
} /* trata_ */

//____________________________________________________________
// trats.F -- translated by f2c (version 19970219).
float *TCL::trats(const float *a, const float *s, float *b, int m, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int inds, i__, j, k, ia, ib, is;
   double sum;

   /* CERN PROGLIB# F112    TRATS           .VERSION KERNFOR  4.15  861204 */
   /* ORIG. 18/12/74 WH */

   /* Parameter adjuTments */
   --b;    --s;    --a;

   /* Function Body */
   ib = 0;    inds = 0;    i__ = 0;
   do {
      inds += i__;
      ib = i__ + 1;

      for (j = 1; j <= m; ++j) {
         ia = j;
         is = inds;
         sum = 0.;
         k = 0;

         do {
            if (k > i__) is += k;
            else         ++is;
            sum += a[ia] * s[is];
            ia += m;
            ++k;
         } while (k < n);

         b[ib] = sum;
         ib += n;
      }
      ++i__;
   } while (i__ < n);
   ++b;
   return b;
} /* trats_ */

//____________________________________________________________
// tratsa.F -- translated by f2c (version 19970219).
/* Subroutine */float *TCL::tratsa(const float *a, const float *s, float *r__, int m, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int imax, i__, j, k;
   int ia, ir, is, iaa, ind;
   double sum;

   /* CERN PROGLIB# F112    TRATSA          .VERSION KERNFOR  4.15  861204 */
   /* ORIG. 18/12/74 WH */


   /* Parameter adjuTments */
   --r__;    --s;    --a;

   /* Function Body */
   imax = (m * m + m) / 2;
   vzero(&r__[1], imax);
   ind = 0;
   i__ = 0;

   do {
      ind += i__;
      ir = 0;

      for (j = 1; j <= m; ++j) {
         is = ind;
         ia = j;
         sum = 0.;
         k = 0;

         do {
            if (k > i__) is += k;
            else         ++is;
            sum += s[is] * a[ia];
            ia += m;
            ++k;
         } while  (k < n);
         iaa = i__ * m;

         for (k = 1; k <= j; ++k) {
            ++iaa;
            ++ir;
            r__[ir] += sum * a[iaa];
         }
      }
      ++i__;
   } while (i__ < n);
   ++r__;
   return r__;
} /* tratsa_ */

//____________________________________________________________
// trchlu.F -- translated by f2c (version 19970219).
float *TCL::trchlu(const float *a, float *b, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int ipiv, kpiv, i__, j;
   double r__, dc;
   int id, kd;
   double sum;


   /* CERN PROGLIB# F112    TRCHLU          .VERSION KERNFOR  4.16  870601 */
   /* ORIG. 18/12/74 W.HART */


   /* Parameter adjuTments */
   --b;    --a;

   /* Function Body */
   ipiv = 0;

   i__ = 0;

   do {
      ++i__;
      ipiv += i__;
      kpiv = ipiv;
      r__ = a[ipiv];

      for (j = i__; j <= n; ++j) {
         sum = 0.;
         if (i__ == 1)           goto L40;
         if (r__ == 0.)      goto L42;
         id = ipiv - i__ + 1;
         kd = kpiv - i__ + 1;

         do {
            sum += b[kd] * b[id];
            ++kd;       ++id;
         } while (id < ipiv);

L40:
         sum = a[kpiv] - sum;
L42:
         if (j != i__) b[kpiv] = sum * r__;
         else {
            dc = TMath::Sqrt(sum);
            b[kpiv] = dc;
            if (r__ > 0.)  r__ = 1. / dc;
         }
         kpiv += j;
      }

   } while  (i__ < n);
   ++b;
   return b;
} /* trchlu_ */

//____________________________________________________________
// trchul.F -- translated by f2c (version 19970219).
/* Subroutine */float *TCL::trchul(const float *a, float *b, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int ipiv, kpiv, i__;
   double r__;
   int nTep;
   double dc;
   int id, kd;
   double sum;


   /* CERN PROGLIB# F112    TRCHUL          .VERSION KERNFOR  4.16  870601 */
   /* ORIG. 18/12/74 WH */


   /* Parameter adjuTments */
   --b;    --a;

   /* Function Body */
   kpiv = (n * n + n) / 2;

   i__ = n;
   do {
      ipiv = kpiv;
      r__ = a[ipiv];

      do {
         sum = 0.;
         if (i__ == n)   goto L40;
         if (r__ == 0.)  goto L42;
         id = ipiv;
         kd = kpiv;
         nTep = i__;

         do {
            kd += nTep;
            id += nTep;
            ++nTep;
            sum += b[id] * b[kd];
         } while  (nTep < n);

L40:
         sum = a[kpiv] - sum;
L42:
         if (kpiv < ipiv) b[kpiv] = sum * r__;
         else {
            dc = TMath::Sqrt(sum);
            b[kpiv] = dc;
            if (r__ > 0.)         r__ = 1. / dc;
         }
         --kpiv;
      } while (kpiv > ipiv - i__);

      --i__;
   } while  (i__ > 0);

   ++b;
   return b;
} /* trchul_ */

//____________________________________________________________
/* Subroutine */float *TCL::trinv(const float *t, float *s, int n)
{
   // trinv.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRINV           .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   int lhor, ipiv, lver, j;
   double sum = 0;
   double r__ = 0;
   int mx, ndTep, ind;


   /* Parameter adjuTments */
   --s;    --t;

   /* Function Body */
   mx = (n * n + n) / 2;
   ipiv = mx;

   int i = n;
   do {
      r__ = 0.;
      if (t[ipiv] > 0.) r__ = 1. / t[ipiv];
      s[ipiv] = r__;
      ndTep = n;
      ind = mx - n + i;

      while (ind != ipiv) {
         sum = 0.;
         if (r__ != 0.) {
            lhor = ipiv;
            lver = ind;
            j = i;

            do {
               lhor += j;
               ++lver;
               sum += t[lhor] * s[lver];
               ++j;
            } while  (lhor < ind);
         }
         s[ind] = -sum * r__;
         --ndTep;
         ind -= ndTep;
      }

      ipiv -= i;
      --i;
   } while (i > 0);

   ++s;
   return s;
} /* trinv_ */

//____________________________________________________________
// trla.F -- translated by f2c (version 19970219).
/* Subroutine */float *TCL::trla(const float *u, const float *a, float *b, int m, int n)
{
   int ipiv, ia, ib, iu;
   double sum;

   /* CERN PROGLIB# F112    TRLA            .VERSION KERNFOR  4.15  861204 */
   /* ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML


   /* Parameter adjuTments */
   --b;    --a;    --u;

   /* Function Body */
   ib = m * n;
   ipiv = (m * m + m) / 2;

   do {
      do {
         ia = ib;
         iu = ipiv;

         sum = 0.;
         do {
            sum += a[ia] * u[iu];
            --iu;
            ia -= n;
         } while (ia > 0);

         b[ib] = sum;
         --ib;
      } while (ia > 1 - n);

      ipiv = iu;
   } while (iu > 0);

   ++b;
   return b;
} /* trla_ */

//____________________________________________________________
/* trlta.F -- translated by f2c (version 19970219).
// Subroutine */float *TCL::trlta(const float *u, const float *a, float *b, int m, int n)
{
   int ipiv, mxpn, i__, nTep, ia, ib, iu, mx;
   double sum;

   /* CERN PROGLIB# F112    TRLTA           .VERSION KERNFOR  4.15  861204 */
   /* ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML


   /* Parameter adjuTments */
   --b;    --a;    --u;

   /* Function Body */
   ipiv = 0;
   mx = m * n;
   mxpn = mx + n;
   ib = 0;

   i__ = 0;
   do {
      ++i__;
      ipiv += i__;

      do {
         iu = ipiv;
         nTep = i__;
         ++ib;
         ia = ib;

         sum = 0.;
         do {
            sum += a[ia] * u[iu];
            ia += n;
            iu += nTep;
            ++nTep;
         } while (ia <= mx);

         b[ib] = sum;
      } while (ia < mxpn);

   } while (i__ < m);

   ++b;
   return b;
} /* trlta_ */

//____________________________________________________________
float *TCL::trpck(const float *s, float *u, int n)
{
   // trpck.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRPCK           .VERSION KERNFOR  2.08  741218 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   int i__, ia, ind, ipiv;

   /* Parameter adjuTments */
   --u;    --s;

   /* Function Body */
   ia = 0;
   ind = 0;
   ipiv = 0;

   for (i__ = 1; i__ <= n; ++i__) {
      ipiv += i__;
      do {
         ++ia;
         ++ind;
         u[ind] = s[ia];
      } while (ind < ipiv);
      ia = ia + n - i__;
   }

   ++u;
   return u;
} /* trpck_ */

//____________________________________________________________
float *TCL::trqsq(const float *q, const float *s, float *r__, int m)
{
   // trqsq.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRQSQ           .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   int indq, inds, imax, i__, j, k, l;
   int iq, ir, is, iqq;
   double sum;

   /* Parameter adjuTments */
   --r__;    --s;    --q;

   /* Function Body */
   imax = (m * m + m) / 2;
   vzero(&r__[1], imax);
   inds = 0;
   i__ = 0;

   do {
      inds += i__;
      ir = 0;
      indq = 0;
      j = 0;

      do {
         indq += j;
         is = inds;
         iq = indq;
         sum = (float)0.;
         k = 0;

         do {
            if (k > i__)  is += k;
            else          ++is;

            if (k > j)    iq += k;
            else        ++iq;

            sum += s[is] * q[iq];
            ++k;
         } while (k < m);
         iqq = inds;
         l = 0;

         do {
            ++ir;
            if (l > i__)  iqq += l;
            else          ++iqq;
            r__[ir] += q[iqq] * sum;
            ++l;
         } while (l <= j);
         ++j;
      } while (j < m);
      ++i__;
   } while (i__ < m);

   ++r__;
   return r__;
} /* trqsq_ */

//____________________________________________________________
float *TCL::trsa(const float *s, const float *a, float *b, int m, int n)
{
   // trsa.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRSA            .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int inds, i__, j, k, ia, ib, is;
   double sum;

   /* Parameter adjuTments */
   --b;    --a;    --s;

   /* Function Body */
   inds = 0;
   ib = 0;
   i__ = 0;

   do {
      inds += i__;

      for (j = 1; j <= n; ++j) {
         ia = j;
         is = inds;
         sum = 0.;
         k = 0;

         do {
            if (k > i__) is += k;
            else         ++is;
            sum += s[is] * a[ia];
            ia += n;
            ++k;
         } while (k < m);
         ++ib;
         b[ib] = sum;
      }
      ++i__;
   } while (i__ < m);

   ++b;
   return b;
} /* trsa_ */

//____________________________________________________________
/* Subroutine */float *TCL::trsinv(const float *g, float *gi, int n)
{
   // trsinv.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRSINV          .VERSION KERNFOR  2.08  741218
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Function Body */
   trchlu(g, gi, n);
   trinv(gi, gi, n);
   return trsmul(gi, gi, n);
} /* trsinv_ */

//____________________________________________________________
/* Subroutine */float *TCL::trsmlu(const float *u, float *s, int n)
{
   // trsmlu.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRSMLU          .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int lhor, lver, i__, k, l, ind;
   double sum;

   /* Parameter adjuTments */
   --s;    --u;

   /* Function Body */
   ind = (n * n + n) / 2;

   for (i__ = 1; i__ <= n; ++i__) {
      lver = ind;

      for (k = i__; k <= n; ++k,--ind) {
         lhor = ind;    sum = 0.;
         for (l = k; l <= n; ++l,--lver,--lhor)
            sum += u[lver] * u[lhor];
         s[ind] = sum;
      }
   }
   ++s;
   return s;
} /* trsmlu_ */

//____________________________________________________________
/* Subroutine */float *TCL::trsmul(const float *g, float *gi, int n)
{
   // trsmul.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRSMUL          .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int lhor, lver, lpiv, i__, j, k, ind;
   double sum;

   /* Parameter adjuTments */
   --gi;    --g;

   /* Function Body */
   ind = 1;
   lpiv = 0;
   for (i__ = 1; i__ <= n; ++i__) {
      lpiv += i__;
      for (j = 1; j <= i__; ++j,++ind) {
         lver = lpiv;
         lhor = ind;
         sum = 0.;
         for (k = i__; k <= n; lhor += k,lver += k,++k)
            sum += g[lver] * g[lhor];
         gi[ind] = sum;
      }
   }
   ++gi;
   return gi;
} /* trsmul_ */

//____________________________________________________________
float *TCL::trupck(const float *u, float *s, int m)
{
   // trupck.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRUPCK          .VERSION KERNFOR  2.08  741218
   // ORIG. 18/12/74 WH
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML


   int i__, im, is, iu, iv, ih, m2;

   /* Parameter adjuTments */
   --s;    --u;

   /* Function Body */
   m2 = m * m;
   is = m2;
   iu = (m2 + m) / 2;
   i__ = m - 1;

   do {
      im = i__ * m;
      do {
         s[is] = u[iu];
         --is;
         --iu;
      } while (is > im);
      is = is - m + i__;
      --i__;
   } while (i__ >= 0);

   is = 1;
   do {
      iv = is;
      ih = is;
      while (1) {
         iv += m;
         ++ih;
         if (iv > m2)    break;
         s[ih] = s[iv];
      }
      is = is + m + 1;
   } while (is < m2);

   ++s;
   return s;
} /* trupck_ */

//____________________________________________________________
/* trsat.F -- translated by f2c (version 19970219).
// Subroutine */ float *TCL::trsat(const float *s, const float *a, float *b, int m, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int inds, i__, j, k, ia, ib, is;
   double sum;


   /* CERN PROGLIB# F112    TRSAT           .VERSION KERNFOR  4.15  861204 */
   /* ORIG. 18/12/74 WH */


   /* Parameter adjuTments */
   --b;    --a;    --s;

   /* Function Body */
   inds = 0;
   ib = 0;
   i__ = 0;

   do {
      inds += i__;
      ia = 0;

      for (j = 1; j <= n; ++j) {
         is = inds;
         sum = 0.;
         k = 0;

         do {
            if (k > i__) is += k;
            else         ++is;
            ++ia;
            sum += s[is] * a[ia];
            ++k;
         } while (k < m);
         ++ib;
         b[ib] = sum;
      }
      ++i__;
   } while (i__ < m);

   ++b;
   return b;
} /* trsat_ */

// ------  double

//____________________________________________________________
// trata.F -- translated by f2c (version 19970219).
double *TCL::trata(const double *a, double *r__, int m, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int i__, j, ia, mn, ir, iat;
   double sum;


   /* CERN PROGLIB# F112    TRATA           .VERSION KERNFOR  4.15  861204 */
   /* ORIG. 18/12/74 WH */


   /* Parameter adjuTments */
   --r__;    --a;

   /* Function Body */
    mn = m * n;
   ir = 0;

   for (i__ = 1; i__ <= m; ++i__) {

      for (j = 1; j <= i__; ++j) {
         ia = i__;
         iat = j;

         sum = (double)0.;
         do {
            sum += a[ia] * a[iat];
            ia +=  m;
            iat += m;
         } while  (ia <= mn);
         ++ir;
         r__[ir] = sum;
      }
   }

   return 0;
} /* trata_ */

//____________________________________________________________
// trats.F -- translated by f2c (version 19970219).
double *TCL::trats(const double *a, const double *s, double *b, int m, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int inds, i__, j, k, ia, ib, is;
   double sum;


   /* CERN PROGLIB# F112    TRATS           .VERSION KERNFOR  4.15  861204 */
   /* ORIG. 18/12/74 WH */

   /* Parameter adjuTments */
   --b;    --s;    --a;

   /* Function Body */
   ib = 0;    inds = 0;    i__ = 0;

   do {
      inds += i__;
      ib = i__ + 1;

      for (j = 1; j <= m; ++j) {
         ia = j;
         is = inds;
         sum = (double)0.;
         k = 0;

         do {
            if (k > i__) is += k;
            else         ++is;
            sum += a[ia] * s[is];
            ia += m;
            ++k;
         } while (k < n);

         b[ib] = sum;
         ib += n;
      }
      ++i__;
   } while (i__ < n);

   return 0;
} /* trats_ */

//____________________________________________________________
// tratsa.F -- translated by f2c (version 19970219).
/* Subroutine */double *TCL::tratsa(const double *a, const double *s, double *r__, int m, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int imax, i__, j, k;
   int ia, ir, is, iaa, ind;
   double sum;

   /* CERN PROGLIB# F112    TRATSA          .VERSION KERNFOR  4.15  861204 */
   /* ORIG. 18/12/74 WH */


   /* Parameter adjuTments */
   --r__;    --s;    --a;

   /* Function Body */
   imax = (m * m + m) / 2;
   vzero(&r__[1], imax);
   ind = 0;
   i__ = 0;

   do {
      ind += i__;
      ir = 0;

      for (j = 1; j <= m; ++j) {
         is = ind;
         ia = j;
         sum = (double)0.;
         k = 0;

         do {
            if (k > i__) is += k;
            else         ++is;
            sum += s[is] * a[ia];
            ia += m;
            ++k;
         } while  (k < n);
         iaa = i__ * m;

         for (k = 1; k <= j; ++k) {
            ++iaa;
            ++ir;
            r__[ir] += sum * a[iaa];
         }
      }
      ++i__;
   } while (i__ < n);

   return 0;
} /* tratsa_ */

//____________________________________________________________
double *TCL::trchlu(const double *a, double *b, int n)
{
   // trchlu.F -- translated by f2c (version 19970219).
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int ipiv, kpiv, i__, j;
   double r__, dc;
   int id, kd;
   double sum;


   /* CERN PROGLIB# F112    TRCHLU          .VERSION KERNFOR  4.16  870601 */
   /* ORIG. 18/12/74 W.HART */


   /* Parameter adjuTments */
   --b;    --a;

   /* Function Body */
   ipiv = 0;

   i__ = 0;

   do {
      ++i__;
      ipiv += i__;
      kpiv = ipiv;
      r__ = a[ipiv];

      for (j = i__; j <= n; ++j) {
         sum = 0.;
         if (i__ == 1)       goto L40;
         if (r__ == 0.)      goto L42;
         id = ipiv - i__ + 1;
         kd = kpiv - i__ + 1;

         do {
            sum += b[kd] * b[id];
            ++kd;   ++id;
         } while (id < ipiv);

L40:
         sum = a[kpiv] - sum;
L42:
         if (j != i__) b[kpiv] = sum * r__;
         else {
            dc = TMath::Sqrt(sum);
            b[kpiv] = dc;
            if (r__ > 0.)  r__ = (double)1. / dc;
         }
         kpiv += j;
      }

   } while  (i__ < n);

   return 0;
} /* trchlu_ */

//____________________________________________________________
// trchul.F -- translated by f2c (version 19970219).
double *TCL::trchul(const double *a, double *b, int n)
{
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int ipiv, kpiv, i__;
   double r__;
   int nTep;
   double dc;
   int id, kd;
   double sum;


   /* CERN PROGLIB# F112    TRCHUL          .VERSION KERNFOR  4.16  870601 */
   /* ORIG. 18/12/74 WH */


   /* Parameter adjuTments */
   --b;    --a;

   /* Function Body */
   kpiv = (n * n + n) / 2;

   i__ = n;
   do {
      ipiv = kpiv;
      r__ = a[ipiv];

      do {
         sum = 0.;
         if (i__ == n)           goto L40;
         if (r__ == (double)0.)  goto L42;
         id = ipiv;
         kd = kpiv;
         nTep = i__;

         do {
            kd += nTep;
            id += nTep;
            ++nTep;
            sum += b[id] * b[kd];
         } while  (nTep < n);

L40:
         sum = a[kpiv] - sum;
L42:
         if (kpiv < ipiv) b[kpiv] = sum * r__;
         else {
            dc = TMath::Sqrt(sum);
            b[kpiv] = dc;
            if (r__ > (double)0.)         r__ = (double)1. / dc;
         }
         --kpiv;
      } while (kpiv > ipiv - i__);

      --i__;
   } while  (i__ > 0);

   return 0;
} /* trchul_ */

//____________________________________________________________
double *TCL::trinv(const double *t, double *s, int n)
{
   // trinv.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRINV           .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
   //
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   int lhor, ipiv, lver,  j;
   double r__;
   int mx, ndTep, ind;
   double sum;

   /* Parameter adjuTments */
   --s;    --t;

   /* Function Body */
   mx = (n * n + n) / 2;
   ipiv = mx;

   int i = n;
   do {
      r__ = 0.;
      if (t[ipiv] > 0.)  r__ = (double)1. / t[ipiv];
      s[ipiv] = r__;
      ndTep = n;
      ind = mx - n + i;

      while (ind != ipiv) {
         sum = 0.;
         if (r__ != 0.) {
            lhor = ipiv;
            lver = ind;
            j = i;

            do {
               lhor += j;
               ++lver;
               sum += t[lhor] * s[lver];
               ++j;
            } while  (lhor < ind);
         }
         s[ind] = -sum * r__;
         --ndTep;
         ind -= ndTep;
      }

      ipiv -= i;
      --i;
   } while (i > 0);

   return 0;
} /* trinv_ */

//____________________________________________________________
/* Subroutine */double *TCL::trla(const double *u, const double *a, double *b, int m, int n)
{
   //
   // trla.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRLA            .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
   //
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   int ipiv, ia, ib, iu;
   double sum;

   /* Parameter adjuTments */
   --b;    --a;    --u;

   /* Function Body */
   ib = m * n;
   ipiv = (m * m + m) / 2;

   do {
      do {
         ia = ib;
         iu = ipiv;

         sum = 0.;
         do {
            sum += a[ia] * u[iu];
            --iu;
            ia -= n;
         } while (ia > 0);

         b[ib] = sum;
         --ib;
      } while (ia > 1 - n);

      ipiv = iu;
   } while (iu > 0);

   return 0;
} /* trla_ */

//____________________________________________________________
double *TCL::trlta(const double *u, const double *a, double *b, int m, int n)
{
   // trlta.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRLTA           .VERSION KERNFOR  4.15  861204
   // ORIG. 18/12/74 WH
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   int ipiv, mxpn, i__, nTep, ia, ib, iu, mx;
   double sum;

   /* Parameter adjuTments */
   --b;    --a;    --u;

   /* Function Body */
   ipiv = 0;
   mx = m * n;
   mxpn = mx + n;
   ib = 0;

   i__ = 0;
   do {
      ++i__;
      ipiv += i__;

      do {
         iu = ipiv;
         nTep = i__;
         ++ib;
         ia = ib;

         sum = 0.;
         do {
            sum += a[ia] * u[iu];
            ia += n;
            iu += nTep;
            ++nTep;
         } while (ia <= mx);

         b[ib] = sum;
      } while (ia < mxpn);

   } while (i__ < m);

   return 0;
} /* trlta_ */

//____________________________________________________________
/* Subroutine */double *TCL::trpck(const double *s, double *u, int n)
{
   // trpck.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRPCK           .VERSION KERNFOR  2.08  741218 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   int i__, ia, ind, ipiv;

   /* Parameter adjuTments */
   --u;    --s;

   /* Function Body */
   ia = 0;
   ind = 0;
   ipiv = 0;

   for (i__ = 1; i__ <= n; ++i__) {
      ipiv += i__;
      do {
         ++ia;
         ++ind;
         u[ind] = s[ia];
      } while (ind < ipiv);
      ia = ia + n - i__;
   }

   return 0;
} /* trpck_ */

//____________________________________________________________
double *TCL::trqsq(const double *q, const double *s, double *r__, int m)
{
   // trqsq.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRQSQ           .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   int indq, inds, imax, i__, j, k, l;
   int iq, ir, is, iqq;
   double sum;

   /* Parameter adjuTments */
   --r__;    --s;    --q;

   /* Function Body */
   imax = (m * m + m) / 2;
   vzero(&r__[1], imax);
   inds = 0;
   i__ = 0;

   do {
      inds += i__;
      ir = 0;
      indq = 0;
      j = 0;

      do {
         indq += j;
         is = inds;
         iq = indq;
         sum = 0.;
         k = 0;

         do {
            if (k > i__)  is += k;
            else          ++is;

            if (k > j)    iq += k;
            else        ++iq;

            sum += s[is] * q[iq];
            ++k;
         } while (k < m);
         iqq = inds;
         l = 0;

         do {
            ++ir;
            if (l > i__)  iqq += l;
            else          ++iqq;
            r__[ir] += q[iqq] * sum;
            ++l;
         } while (l <= j);
         ++j;
      } while (j < m);
      ++i__;
   } while (i__ < m);

   return 0;
} /* trqsq_ */

//____________________________________________________________
double *TCL::trsa(const double *s, const double *a, double *b, int m, int n)
{
   // trsa.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRSA            .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML
   /* Local variables */
   int inds, i__, j, k, ia, ib, is;
   double sum;

   /* Parameter adjuTments */
   --b;    --a;    --s;

   /* Function Body */
   inds = 0;
   ib = 0;
   i__ = 0;

   do {
      inds += i__;

      for (j = 1; j <= n; ++j) {
         ia = j;
         is = inds;
         sum = 0.;
         k = 0;

         do {
            if (k > i__) is += k;
            else         ++is;
            sum += s[is] * a[ia];
            ia += n;
            ++k;
         } while (k < m);
         ++ib;
         b[ib] = sum;
      }
      ++i__;
   } while (i__ < m);

   return 0;
} /* trsa_ */

//____________________________________________________________
/* Subroutine */double *TCL::trsinv(const double *g, double *gi, int n)
{
   // trsinv.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRSINV          .VERSION KERNFOR  2.08  741218
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Function Body */
   trchlu(g, gi, n);
   trinv(gi, gi, n);
   trsmul(gi, gi, n);

   return 0;
} /* trsinv_ */

//____________________________________________________________
/* Subroutine */double *TCL::trsmlu(const double *u, double *s, int n)
{
   // trsmlu.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRSMLU          .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int lhor, lver, i__, k, l, ind;
   double sum;

   /* Parameter adjuTments */
   --s;    --u;

   /* Function Body */
   ind = (n * n + n) / 2;

   for (i__ = 1; i__ <= n; ++i__) {
      lver = ind;

      for (k = i__; k <= n; ++k,--ind) {
         lhor = ind;    sum = 0.;
         for (l = k; l <= n; ++l,--lver,--lhor)
            sum += u[lver] * u[lhor];
         s[ind] = sum;
      }
   }

   return 0;
} /* trsmlu_ */

//____________________________________________________________
/* Subroutine */double *TCL::trsmul(const double *g, double *gi, int n)
{
   // trsmul.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRSMUL          .VERSION KERNFOR  4.15  861204 */
   // ORIG. 18/12/74 WH */
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int lhor, lver, lpiv, i__, j, k, ind;
   double sum;

   /* Parameter adjuTments */
   --gi;    --g;

   /* Function Body */
   ind = 1;
   lpiv = 0;
   for (i__ = 1; i__ <= n; ++i__) {
      lpiv += i__;
      for (j = 1; j <= i__; ++j,++ind) {
         lver = lpiv;
         lhor = ind;
         sum = 0.;
         for (k = i__; k <= n;lhor += k,lver += k,++k)
            sum += g[lver] * g[lhor];
         gi[ind] = sum;
      }
   }

   return 0;
} /* trsmul_ */

//____________________________________________________________
/* Subroutine */double *TCL::trupck(const double *u, double *s, int m)
{
   // trupck.F -- translated by f2c (version 19970219).
   // CERN PROGLIB# F112    TRUPCK          .VERSION KERNFOR  2.08  741218
   // ORIG. 18/12/74 WH
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML


   int i__, im, is, iu, iv, ih, m2;

   /* Parameter adjuTments */
   --s;    --u;

   /* Function Body */
   m2 = m * m;
   is = m2;
   iu = (m2 + m) / 2;
   i__ = m - 1;

   do {
      im = i__ * m;
      do {
         s[is] = u[iu];
         --is;
         --iu;
      } while (is > im);
      is = is - m + i__;
      --i__;
   } while (i__ >= 0);

   is = 1;
   do {
      iv = is;
      ih = is;
      while (1) {
         iv += m;
         ++ih;
         if (iv > m2)    break;
         s[ih] = s[iv];
      }
      is = is + m + 1;
   } while (is < m2);

   return 0;
} /* trupck_ */

//____________________________________________________________
double *TCL::trsat(const double *s, const double *a, double *b, int m, int n)
{
   // trsat.F -- translated by f2c (version 19970219)
   // CERN PROGLIB# F112    TRSAT           .VERSION KERNFOR  4.15  861204
   // ORIG. 18/12/74 WH
 //BEGIN_HTML <!--
 /* -->
  <b>see original documentation of CERNLIB package</b> <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/f112/top.html">F112</A>
 <!--*/
 // -->END_HTML

   /* Local variables */
   int inds, i__, j, k, ia, ib, is;
   double sum;

   /* Parameter adjuTments */
   --b;    --a;    --s;

   /* Function Body */
   inds = 0;
   ib = 0;
   i__ = 0;

   do {
      inds += i__;
      ia = 0;

      for (j = 1; j <= n; ++j) {
         is = inds;
         sum = 0.;
         k = 0;

         do {
            if (k > i__) is += k;
            else         ++is;
            ++ia;
            sum += s[is] * a[ia];
            ++k;
         } while (k < m);
         ++ib;
         b[ib] = sum;
      }
      ++i__;
   } while (i__ < m);

   return 0;
} /* trsat_ */

// ------------ Victor Perevoztchikov's addition

//_____________________________________________________________________________
float *TCL::trsequ(float *smx, int m, float *b, int n)
{
   // Linear Equations, Matrix Inversion
   // trsequ solves the matrix equation
   //
   //             SMX*x = B
   //
   // which represents a system of m simultaneous linear equations with n right-hand sides:
   // SMX is an  unpacked symmetric matrix (all  elements) (m x m)
   // B is an unpacked matrix of right-hand sides (n x m)
   //
   float *mem = new float[(m*(m+1))/2+m];
   float *v = mem;
   float *s = v+m;
   if (!b) n=0;
   TCL::trpck (smx,s    ,m);
   TCL::trsinv(s  ,s,    m);

   for (int i=0;i<n;i++) {
      TCL::trsa  (s  ,b+i*m, v, m, 1);
      TCL::ucopy (v  ,b+i*m, m);}
   TCL::trupck(s  ,smx,  m);
   delete [] mem;
   return b;
}
//_____________________________________________________________________________
double *TCL::trsequ(double *smx, int m, double *b, int n)
{
   // Linear Equations, Matrix Inversion
   // trsequ solves the matrix equation
   //
   //             SMX*x = B
   //
   // which represents a system of m simultaneous linear equations with n right-hand sides:
   // SMX is an  unpacked symmetric matrix (all  elements) (m x m)
   // B is an unpacked matrix of right-hand sides (n x m)
   //
   double *mem = new double[(m*(m+1))/2+m];
   double *v = mem;
   double *s = v+m;
   if (!b) n=0;
   TCL::trpck (smx,s    ,m);
   TCL::trsinv(s  ,s,    m);

   for (int i=0;i<n;i++) {
      TCL::trsa  (s  ,b+i*m, v, m, 1);
      TCL::ucopy (v  ,b+i*m, m);}
   TCL::trupck(s  ,smx,  m);
   delete [] mem;
   return b;
}

