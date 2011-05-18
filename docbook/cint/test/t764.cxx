/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* #define CGI */
double          besj0(double);
double          besj1(double);

int main()
{
  static double   x, bj0, bj1;
  int             i;
 
#if !defined(G__MSC_VER) && !defined(_MSC_VER)
   system("clear");
#endif

  /* printf("\x01b[2J"); */	/* clear screan */
  printf("                 x           besj0                  besj1\n");

  for (i = 0; i < 11; i++) {
    x = (double) i;
    bj0 = besj0(x);
    bj1 = besj1(x);
    printf("              %5.2f       %12.6E         %12.6E\n", x, bj0, bj1);
  }
  printf("\n\n");
  return 0;
}

double besj0(double x)
{
  int             i, j;
  double          bj0, w, wx, wx4, wp, wq;
  static double   a[] = {0, 1.00, -3.9999998721, 3.9999973021, -1.7777560599,
			   0.4443584263, -0.0709253492, 0.0076771853},
  c[] = {0, 0.3989422793, -0.001753062, 0.00017343,
	   -0.00004877613, 0.0000173565},
  k[] = {0, -.0124669441, 0.0004564324, -0.0000869791,
	   0.0000342468, -0.0000142078};

  if (x < 0)
    return (999.0);
  wx = x;
  if (x <= 4) {
    w = -.0005014415;
    wx4 = (wx / 4.0) * (wx / 4.0);
    for (i = 1; i <= 7; i++) {
      j = 8 - i;
      w = w * wx4 + a[j];
    }
    bj0 = w;
    return (bj0);
  }
  wx4 = (4.0 / wx) * (4.0 / wx);
  wp = -.0000037043;
  wq = .0000032312;
  for (i = 1; i <= 5; i++) {
    j = 6 - i;
    wp = wp * wx4 + c[j];
    wq = wq * wx4 + k[j];
  }
  w = wx - 0.7853981633974483;
  wp = wp * cos(w);
  wq = wq * sin(w);
  w = 2 / sqrt(wx);
  bj0 = w * (wp - 4 * wq / wx);
  return (bj0);

}

double besj1(double x)
{
  int             i, j;
  double          bj1, w, wx, wx4, wp, wq;
  static double   a[] = {0.0, 1.9999999998, -3.999999971, 2.6666660544,
			   -0.8888839649, 0.1777582922, -0.0236616773,
			   0.0022069155},
  c[] = {0.0, 0.3989422819, 0.0029218256, -0.000223203,
	   0.0000580759, -0.000020092},
  k[] = {0.0, 0.0374008364, -0.00063904, 0.0001064741,
	   -0.0000398708, 0.00001622};

  if (x < 0)
    return (999);
  wx = x;
  if (x <= 4.0) {
    w = -0.0001289769;
    wx4 = (wx / 4.0) * (wx / 4.0);
    for (i = 1; i <= 7; i++) {
      j = 8 - i;
      w = w * wx4 + a[j];
    }
    bj1 = w * wx / 4.0;
    return (bj1);
  }
  wx4 = (4.0 / wx) * (4.0 / wx);
  wp = 0.000004214;
  wq = -0.0000036594;
  for (i = 1; i <= 5; i++) {
    j = 6 - i;
    wp = wp * wx4 + c[j];
    wq = wq * wx4 + k[j];
  }
  w = wx - 2.356194490192345;
  wp = wp * cos(w);
  wq = wq * sin(w);
  w = 2.0 / sqrt(wx);
  bj1 = w * (wp - 4.0 * wq / wx);
  return (bj1);
}


