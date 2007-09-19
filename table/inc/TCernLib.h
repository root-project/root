// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCernLib
#define ROOT_TCernLib

#include "Rtypes.h"
#include <string.h>

// http://wwwinfo.cern.ch/asdoc/shortwrupsdir/f110/top.html

///////////////////////////////////////////////////////////////////////////////////////
//                                                                                   //
// The routines of MXPACK compute the product of two matrices or the product of      //
// their transposed matrices and may add or subtract to the resultant matrix         //
// a third one, add or subtract one matrix from another, or transfer a matrix,       //
// its negative, or a multiple of it, transpose a given matrix, build up a unit      //
// matrix, multiply a matrix by a diagonal (from left or from right) and may         //
// add the result to another matrix, add to square matrix the multiple of a diagonal //
// matrix, compute the products <IMG WIDTH=79 HEIGHT=12 ALIGN=BOTTOM ALT="tex2html_wrap_inline191" SRC="gif/mxpack_ABAt.gif"> (<IMG WIDTH=16 HEIGHT=12 ALIGN=BOTTOM ALT="tex2html_wrap_inline193" SRC="gif/mxpack_At.gif"> denotes the transpose of <IMG WIDTH=1
// It is assumed that matrices are begin_html <B>row-wise without gaps</B> end_html without gaps.                     //
//                                                                                   //
///////////////////////////////////////////////////////////////////////////////////////

class TArrayD;

class TCL  {
public:
   virtual ~TCL() { }

   static int    *ucopy(const int    *a, int    *b, int n);
   static float  *ucopy(const float  *a, float  *b, int n);
   static double *ucopy(const float  *a, double *b, int n);
   static float  *ucopy(const double *a, float  *b, int n);
   static double *ucopy(const double *a, double *b, int n);
   static void  **ucopy(const void **a, void  **b, int n);

   static float  *vzero(float *a,  int n2);
   static double *vzero(double *a, int n2);
   static void  **vzero(void **a,  int n2);

   static float  *vadd(const float *b,  const float  *c,  float *a, int n);
   static double *vadd(const double *b, const double *c, double *a, int n);

   static float  *vadd(const float *b,  const double *c, float *a, int n);
   static double *vadd(const double *b, const float  *c,double *a, int n);

   static float   vdot(const float  *b, const float  *a, int n);
   static double  vdot(const double *b, const double *a, int n);

   static float  *vsub(const float  *a, const float  *b, float  *x, int n);
   static double *vsub(const double *a, const double *b, double *x, int n);
   static float  *vsub(const float  *b, const double *c, float  *a, int n);
   static double *vsub(const double *b, const float  *c, double *a, int n);

   static float  *vcopyn(const float *a,  float *x, int n);
   static double *vcopyn(const double *a, double *x, int n);

   static float  *vscale(const float  *a, float  scale, float  *b, int n);
   static double *vscale(const double *a, double scale, double *b, int n);

   static float  *vlinco(const float  *a, float  fa, const float  *b, float  fb,float  *x, int n);
   static double *vlinco(const double *a, double fa, const double *b, double fb,double *x, int n);

   static float  *vmatl(const float  *g, const float  *c, float  *x, int n=3,int m=3);
   static double *vmatl(const double *g, const double *c, double *x, int n=3,int m=3);

   static float  *vmatr(const float  *c, const float  *g, float  *x, int n=3,int m=3);
   static double *vmatr(const double *c, const double *g, double *x, int n=3,int m=3);

   static float *mxmad_0_(int n, const float *a, const float *b, float *c, int i, int j, int k);

   static float *mxmad( const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmad1(const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmad2(const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmad3(const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmpy( const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmpy1(const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmpy2(const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmpy3(const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmub( const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmub1(const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmub2(const float *a, const float *b, float *c, int i, int j, int k);
   static float *mxmub3(const float *a, const float *b, float *c, int i, int j, int k);

   static float *mxmlrt_0_(int n__, const float *a, const float *b, float *c, int ni,int nj);
   static float *mxmlrt(const float *a, const float *b, float *c, int ni, int nj);
   static float *mxmltr(const float *a, const float *b, float *c, int ni, int nj);
   static float *mxtrp(const float *a, float *b, int i, int j);

   static double *mxmad_0_(int n, const double *a, const double *b, double *c, int i, int j, int k);

   static double *mxmad (const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmad1(const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmad2(const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmad3(const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmpy (const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmpy1(const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmpy2(const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmpy3(const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmub (const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmub1(const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmub2(const double *a, const double *b, double *c, int i, int j, int k);
   static double *mxmub3(const double *a, const double *b, double *c, int i, int j, int k);

   static double *mxmlrt_0_(int n__, const double *a, const double *b, double *c, int ni,int nj);
   static double *mxmlrt(const double *a, const double *b, double *c, int ni, int nj);
   static double *mxmltr(const double *a, const double *b, double *c, int ni, int nj);
   static double *mxtrp(const double *a, double *b, int i, int j);

// * TR pack

   static float *traat(const float *a, float *s, int m, int n);
   static float *tral(const float *a, const float *u, float *b, int m, int n);
   static float *tralt(const float *a, const float *u, float *b, int m, int n);
   static float *tras(const float *a, const float *s, float *b, int m, int n);
   static float *trasat(const float *a, const float *s, float *r, int m, int n);
   static float *trasat(const double *a, const float *s, float *r, int m, int n);
   static float *trata(const float *a, float *r, int m, int n);
   static float *trats(const float *a, const float *s, float *b, int m, int n);
   static float *tratsa(const float *a, const float *s, float *r, int m, int n);
   static float *trchlu(const float *a, float *b, int n);
   static float *trchul(const float *a, float *b, int n);
   static float *trinv(const float *t, float *s, int n);
   static float *trla(const float *u, const float *a, float *b, int m, int n);
   static float *trlta(const float *u, const float *a, float *b, int m, int n);
   static float *trpck(const float *s, float *u, int n);
   static float *trqsq(const float *q, const float *s, float *r, int m);
   static float *trsa(const float *s, const float *a, float *b, int m, int n);
   static float *trsinv(const float *g, float *gi, int n);
   static float *trsmlu(const float *u, float *s, int n);
   static float *trsmul(const float *g, float *gi, int n);
   static float *trupck(const float *u, float *s, int m);
   static float *trsat(const float *s, const float *a, float *b, int m, int n);

// Victor Perevoztchikov's addition:
   static float *trsequ(float *smx, int m=3, float *b=0, int n=1);

// ---   double version

   static double *traat (const double *a, double *s, int m, int n);
   static double *tral  (const double *a, const double *u, double *b, int m, int n);
   static double *tralt (const double *a, const double *u, double *b, int m, int n);
   static double *tras  (const double *a, const double *s, double *b, int m, int n);
   static double *trasat(const double *a, const double *s, double *r, int m, int n);
   static double *trata (const double *a, double *r, int m, int n);
   static double *trats (const double *a, const double *s, double *b, int m, int n);
   static double *tratsa(const double *a, const double *s, double *r, int m, int n);
   static double *trchlu(const double *a, double *b, int n);
   static double *trchul(const double *a, double *b, int n);
   static double *trinv (const double *t, double *s, int n);
   static double *trla  (const double *u, const double *a, double *b, int m, int n);
   static double *trlta (const double *u, const double *a, double *b, int m, int n);
   static double *trpck (const double *s, double *u, int n);
   static double *trqsq (const double *q, const double *s, double *r, int m);
   static double *trsa  (const double *s, const double *a, double *b, int m, int n);
   static double *trsinv(const double *g, double *gi, int n);
   static double *trsmlu(const double *u, double *s, int n);
   static double *trsmul(const double *g, double *gi, int n);
   static double *trupck(const double *u, double *s, int m);
   static double *trsat (const double *s, const double *a, double *b, int m, int n);

//  Victor Perevoztchikov's addition:
   static double *trsequ(double *smx, int m=3, double *b=0, int n=1);

   ClassDef(TCL,0)  //C++ replacement for CERNLIB matrix / triangle matrix packages: F110 and F112

};

//___________________________________________________________________________
inline float *TCL::mxmad(const float *a, const float *b, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(0, a, b, c, i, j, k);   }

//___________________________________________________________________________
inline float *TCL::mxmad1(const float *a, const float *q, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad1.gif"> </P> End_Html //
   return mxmad_0_(1, a, q, c, i, j, k);  }

//___________________________________________________________________________
inline float *TCL::mxmad2(const float *p, const float *b, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad2.gif"> </P> End_Html //
   return mxmad_0_(2, p, b, c, i, j, k);  }

//___________________________________________________________________________
inline float *TCL::mxmad3(const float *p, const float *q, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad3.gif"> </P> End_Html //
   return mxmad_0_(3, p, q, c, i, j, k);  }

//___________________________________________________________________________
inline float *TCL::mxmpy(const float *a, const float *b, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmpy.gif"> </P> End_Html //
   return mxmad_0_(4, a, b, c, i, j, k); }

//___________________________________________________________________________
inline float *TCL::mxmpy1(const float *a, const float *q, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmpy1.gif"> </P> End_Html //
   return mxmad_0_(5, a, q, c, i, j, k);  }

//___________________________________________________________________________
inline float *TCL::mxmpy2(const float *p, const float *b, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmpy2.gif"> </P> End_Html //
   return mxmad_0_(6, p, b, c, i, j, k); }

//___________________________________________________________________________
inline float *TCL::mxmpy3(const float *p, const float *q, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmpy3.gif"> </P> End_Html //
   return mxmad_0_(7, p, q, c, i, j, k); }

//___________________________________________________________________________
inline float *TCL::mxmub(const float *a, const float *b, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmub.gif"> </P> End_Html //
   return mxmad_0_(8, a, b, c, i, j, k);  }

//___________________________________________________________________________
inline float *TCL::mxmub1(const float *a, const float *q, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmub1.gif"> </P> End_Html //
   return mxmad_0_(9, a, q, c, i, j, k); }

//___________________________________________________________________________
inline float *TCL::mxmub2(const float *p, const float *b, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmub2.gif"> </P> End_Html //
   return mxmad_0_(10, p, b, c, i, j, k); }

//___________________________________________________________________________
inline float *TCL::mxmub3(const float *p, const float *q, float *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmub3.gif"> </P> End_Html //
   return mxmad_0_(11, p, q, c, i, j, k); }

//___________________________________________________________________________
inline float *TCL::mxmlrt(const float *a, const float *b, float *x, int ni, int nj)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmlrt.gif"> </P> End_Html //
   return mxmlrt_0_(0, a, b, x, ni, nj); }

//___________________________________________________________________________
inline float *TCL::mxmltr(const float *a, const float *b, float *x, int ni, int nj)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmltr.gif"> </P> End_Html //
   return mxmlrt_0_(1, a, b, x, ni, nj);   }


//--   double version --

//___________________________________________________________________________
inline double *TCL::mxmad(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(0, a, b, c, i, j, k);   }

//___________________________________________________________________________
inline double *TCL:: mxmad1(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(1, a, b, c, i, j, k);  }

//___________________________________________________________________________
inline double *TCL::mxmad2(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(2, a, b, c, i, j, k);  }

//___________________________________________________________________________
inline double *TCL::mxmad3(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(3, a, b, c, i, j, k);  }

//___________________________________________________________________________
inline double *TCL::mxmpy(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(4, a, b, c, i, j, k); }

//___________________________________________________________________________
inline double *TCL::mxmpy1(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(5, a, b, c, i, j, k);  }

//___________________________________________________________________________
inline double *TCL::mxmpy2(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(6, a, b, c, i, j, k); }

//___________________________________________________________________________
inline double *TCL::mxmpy3(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(7, a, b, c, i, j, k); }

//___________________________________________________________________________
inline double *TCL::mxmub(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(8, a, b, c, i, j, k);  }

//___________________________________________________________________________
inline double *TCL::mxmub1(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(9, a, b, c, i, j, k); }

//___________________________________________________________________________
inline double *TCL::mxmub2(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(10, a, b, c, i, j, k); }

//___________________________________________________________________________
inline double *TCL::mxmub3(const double *a, const double *b, double *c, int i, int j, int k)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmad_0_(11, a, b, c, i, j, k); }

//___________________________________________________________________________
inline double *TCL::mxmlrt(const double *a, const double *b, double *c, int ni, int nj)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return  mxmlrt_0_(0, a, b, c, ni, nj); }

//___________________________________________________________________________
inline double *TCL::mxmltr(const double *a, const double *b, double *c, int ni, int nj)
{
   // Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/mxpack_mxmad.gif"> </P> End_Html //
   return mxmlrt_0_(1, a, b, c, ni, nj);   }

// ----

//________________________________________________________
inline int  *TCL::ucopy(const int  *b, int  *a, int n)
{
   //to be documented
   if (n <= 0) return 0; memcpy(a,b,n*sizeof(int)); return a;
}

//________________________________________________________
inline float *TCL::ucopy(const float *b, float *a, int n)
{
   //to be documented
   if (n <= 0) return 0; memcpy(a,b,n*sizeof(float)); return a;
}

//________________________________________________________
inline float *TCL::ucopy(const double *b, float *a, int n)
{
   //to be documented
   if (n <= 0) return 0;
   for (int i=0;i<n;i++,a++,b++) *a = float(*b);
   return a;
}

//________________________________________________________
inline double *TCL::ucopy(const float *b, double *a, int n)
{
   //to be documented
   if (n <= 0) return 0;
   for (int i=0;i<n;i++,a++,b++) *a = double(*b);
   return a;
}

//________________________________________________________
inline double *TCL::ucopy(const double *b, double *a, int n)
{
   //to be documented
   if (n <= 0) return 0; memcpy(a,b,n*sizeof(double)); return a;
}

//________________________________________________________
inline void **TCL::ucopy(const void **b, void  **a, int n)
{
   //to be documented
   if (n <= 0) return 0; memcpy(a,b,n*sizeof(void *)); return a;
}


//________________________________________________________
inline float *TCL::vadd(const float *b, const float *c,  float *a, int n)
{
   //to be documented
   if (n <= 0)  return 0;
   for (int i=0;i<n;i++) a[i] = b[i] + c[i];
   return a;
}

//________________________________________________________
inline double *TCL::vadd(const double *b, const double *c,  double *a, int n)
{
   //to be documented
   if (n <= 0)  return 0;
   for (int i=0;i<n;i++) a[i] = b[i] + c[i];
   return a;
}

//________________________________________________________
inline float  *TCL::vadd(const float *b, const double *c,  float *a, int n)
{
   //to be documented
   if (n <= 0)  return 0;
   for (int i=0;i<n;i++) a[i] = b[i] + c[i];
   return a;
}

//________________________________________________________
inline double *TCL::vadd(const double *b, const float *c,  double *a, int n)
{
   //to be documented
   if (n <= 0)  return 0;
   for (int i=0;i<n;i++) a[i] = b[i] + c[i];
   return a;
}

//________________________________________________________
inline float  TCL::vdot(const float  *b, const float *a, int n)
{
   //to be documented
   float x=0;
   if (n>0)
      for (int i=0;i<n;i++,a++,b++) x += (*a) * (*b);
   return x;
}
//________________________________________________________
inline double TCL::vdot(const double *b, const double *a, int n)
{
   //to be documented
   double  x=0;
   if (n>0)
      for (int i=0;i<n;i++,a++,b++) x += (*a) * (*b);
   return x;
}
//________________________________________________________
inline float *TCL::vsub(const float *a, const float *b, float *x, int n)
{
   //to be documented
   if (n <= 0) return 0;
   for (int i=0;i<n;i++) x[i] = a[i]-b[i];
   return x;
}

//________________________________________________________
inline double *TCL::vsub(const double *a, const double *b, double *x, int n)
{
  //to be documented
   if (n <= 0) return 0;
   for (int i=0;i<n;i++) x[i] = a[i]-b[i];
   return x;
}
//________________________________________________________
inline float  *TCL::vsub(const float *b, const double *c,  float *a, int n)
{
   //to be documented
   if (n <= 0)  return 0;
   for (int i=0;i<n;i++) a[i] = b[i] - c[i];
   return a;
}

//________________________________________________________
inline double *TCL::vsub(const double *b, const float *c,  double *a, int n)
{
   //to be documented
   if (n <= 0)  return 0;
   for (int i=0;i<n;i++) a[i] = b[i] - c[i];
   return a;
}
//________________________________________________________
inline float *TCL::vcopyn(const float *a, float *x, int n)
{
   //to be documented
   if (n <= 0) return 0;
   for (int i=0;i<n;i++) x[i] = -a[i];
   return x;
}
//________________________________________________________
inline double *TCL::vcopyn(const double *a, double *x, int n)
{
   //to be documented
   if (n <= 0) return 0;
   for (int i=0;i<n;i++) x[i] = -a[i];
   return x;
}

//________________________________________________________
inline float *TCL::vzero(float *a, int n1)
{
   //to be documented
   if (n1 <= 0) return 0;
   return (float *)memset(a,0,n1*sizeof(float));
}

//________________________________________________________
inline double *TCL::vzero(double *a, int n1)
{
   //to be documented
   if (n1 <= 0) return 0;
   return (double *)memset(a,0,n1*sizeof(double));
}

//________________________________________________________
inline void **TCL::vzero(void **a, int n1)
{
   //to be documented
   if (n1 <= 0) return 0;
   return (void **)memset(a,0,n1*sizeof(void *));
}

//________________________________________________________
inline float *TCL::vscale(const float *a, float scale, float *b, int n)
{
   //to be documented
   for (int i=0;i<n;i++) b[i]=scale*a[i];
   return b;
}

//________________________________________________________
inline double *TCL::vscale(const double *a, double scale, double *b, int n)
{
   //to be documented
   for (int i=0;i<n;i++) b[i]=scale*a[i];
   return b;
}

//________________________________________________________
inline float *TCL::vlinco(const float *a, float fa, const float *b, float fb, float *x, int n)
{
   //to be documented
   for (int i=0;i<n;i++){x[i]=a[i]*fa+b[i]*fb;};
   return x;
}

//________________________________________________________
inline double *TCL::vlinco(const double *a, double fa, const double *b, double fb,double *x, int n)
{
   //to be documented
   for (int i=0;i<n;i++) x[i]=a[i]*fa+b[i]*fb;
   return x;
}

//_____________________________________________________________________________
inline float *TCL::vmatl(const float *G, const float *c, float *x, int n,int m)
{
   //  x = G*c
   for (int i=0; i<n; i++) {
      double sum = 0;
      for (int j=0; j<m; j++) sum += G[j + m*i]*c[j];
      x[i] = sum;
   }
   return x;
}

//_____________________________________________________________________________
inline double *TCL::vmatl(const double *G, const double *c, double *x, int n,int m)
{
   //  x = G*c
   for (int i=0; i<n; i++) {
      double sum = 0;
      for (int j=0; j<m; j++) sum += G[j + m*i]*c[j];
      x[i] = sum;
   }
   return x;
}

//_____________________________________________________________________________
inline float *TCL::vmatr(const float *c, const float *G, float *x, int n,int m)
{
   //  x = c*G
   for (int j=0; j<m; j++) {
      double sum = 0;
      for (int i=0; i<n; i++) sum += G[j + n*i]*c[i];
      x[j] = sum;
   }
   return x;
}

//_____________________________________________________________________________
inline double *TCL::vmatr(const double *c, const double *G, double *x, int n,int m)
{
   //  x = c*G
   for (int j=0; j<m; j++) {
      double sum = 0;
      for (int i=0; i<n; i++) sum += G[j + n*i]*c[i];
      x[j] = sum;
   }
   return x;
}

#endif
