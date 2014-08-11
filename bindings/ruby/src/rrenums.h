// @(#)root/ruby:$Id$
// Author:  Elias Athanasopoulos, May 2004

/*
 * ruby-root global enumeration methods
 *
 * ROOT enumerations are Ruby methods in ruby-root and
 * not constants, in order to keep the low-case 'k'.
 *
 * --elathan  <elathan@phys.uoa.gr>
 *
 * (c) 2003, 2004
 */

static VALUE
rr_kWhite (void)
{
  return INT2NUM (kWhite);
}

static VALUE
rr_kBlack (void)
{
  return INT2NUM (kBlack);
}

static VALUE
rr_kRed (void)
{
  return INT2NUM (kRed);
}

static VALUE
rr_kGreen (void)
{
  return INT2NUM (kGreen);
}

static VALUE
rr_kBlue (void)
{
  return INT2NUM (kBlue);
}

static VALUE
rr_kYellow (void)
{
  return INT2NUM (kYellow);
}

static VALUE
rr_kMagenta (void)
{
  return INT2NUM (kMagenta);
}

static VALUE
rr_kCyan (void)
{
  return INT2NUM (kCyan);
}

static VALUE
rr_kSolid (void)
{
  return INT2NUM (kSolid);
}

static VALUE
rr_kDashed (void)
{
  return INT2NUM (kDashed);
}

static VALUE
rr_kDotted (void)
{
  return INT2NUM (kDotted);
}

static VALUE
rr_kDashDotted (void)
{
  return INT2NUM (kDashDotted);
}

static VALUE
rr_kDot (void)
{
  return INT2NUM (kDot);
}

static VALUE
rr_kPlus (void)
{
  return INT2NUM (kPlus);
}

static VALUE
rr_kStar (void)
{
  return INT2NUM (kStar);
}

static VALUE
rr_kCircle (void)
{
  return INT2NUM (kCircle);
}

static VALUE
rr_kMultiply (void)
{
  return INT2NUM (kMultiply);
}

static VALUE
rr_kFullDotSmall (void)
{
  return INT2NUM (kFullDotSmall);
}

static VALUE
rr_kFullDotMedium (void)
{
  return INT2NUM (kFullDotMedium);
}

static VALUE
rr_kFullDotLarge (void)
{
  return INT2NUM (kFullDotLarge);
}


static VALUE
rr_kFullCircle (void)
{
  return INT2NUM (kFullCircle);
}

static VALUE
rr_kFullSquare (void)
{
  return INT2NUM (kFullSquare);
}

static VALUE
rr_kFullTriangleUp (void)
{
  return INT2NUM (kFullTriangleUp);
}

static VALUE
rr_kFullTriangleDown (void)
{
  return INT2NUM (kFullTriangleDown);
}

static VALUE
rr_kOpenCircle (void)
{
  return INT2NUM (kOpenCircle);
}

static VALUE
rr_kOpenSquare (void)
{
  return INT2NUM (kOpenSquare);
}

static VALUE
rr_kOpenTriangleUp (void)
{
  return INT2NUM (kOpenTriangleUp);
}

static VALUE
rr_kOpenDiamond (void)
{
  return INT2NUM (kOpenDiamond);
}

static VALUE
rr_kOpenCross (void)
{
  return INT2NUM (kOpenCross);
}

static VALUE
rr_kFullStar (void)
{
  return INT2NUM (kFullStar);
}

static VALUE
rr_kOpenStar (void)
{
  return INT2NUM (kOpenStar);
}

void init_global_enums (void)
{
  rb_define_method (rb_cObject, "kWhite", VALUEFUNC (rr_kWhite), 0);
  rb_define_method (rb_cObject, "kBlack", VALUEFUNC (rr_kBlack), 0);
  rb_define_method (rb_cObject, "kRed", VALUEFUNC (rr_kRed), 0);
  rb_define_method (rb_cObject, "kGreen", VALUEFUNC (rr_kGreen), 0);
  rb_define_method (rb_cObject, "kBlue", VALUEFUNC (rr_kBlue), 0);
  rb_define_method (rb_cObject, "kYellow", VALUEFUNC (rr_kYellow), 0);
  rb_define_method (rb_cObject, "kMagenta", VALUEFUNC (rr_kMagenta), 0);
  rb_define_method (rb_cObject, "kCyan", VALUEFUNC (rr_kCyan), 0);
  rb_define_method (rb_cObject, "kSolid", VALUEFUNC (rr_kSolid), 0);
  rb_define_method (rb_cObject, "kDashed", VALUEFUNC (rr_kDashed), 0);
  rb_define_method (rb_cObject, "kDotted", VALUEFUNC (rr_kDotted), 0);
  rb_define_method (rb_cObject, "kDashDotted", VALUEFUNC (rr_kDashDotted), 0);
  rb_define_method (rb_cObject, "kDot", VALUEFUNC (rr_kDot), 0);
  rb_define_method (rb_cObject, "kPlus", VALUEFUNC (rr_kPlus), 0);
  rb_define_method (rb_cObject, "kStar", VALUEFUNC (rr_kStar), 0);
  rb_define_method (rb_cObject, "kCircle", VALUEFUNC (rr_kCircle), 0);
  rb_define_method (rb_cObject, "kMultiply", VALUEFUNC (rr_kMultiply), 0);
  rb_define_method (rb_cObject, "kFullDotSmall", VALUEFUNC (rr_kFullDotSmall),
          0);
  rb_define_method (rb_cObject, "kFullDotMedium",
          VALUEFUNC (rr_kFullDotMedium), 0);
  rb_define_method (rb_cObject, "kFullDotLarge", VALUEFUNC (rr_kFullDotLarge),
          0);
  rb_define_method (rb_cObject, "kFullCircle", VALUEFUNC (rr_kFullCircle), 0);
  rb_define_method (rb_cObject, "kFullSquare", VALUEFUNC (rr_kFullSquare), 0);
  rb_define_method (rb_cObject, "kFullTriangleUp",
          VALUEFUNC (rr_kFullTriangleUp), 0);
  rb_define_method (rb_cObject, "kFullTriangleDown",
          VALUEFUNC (rr_kFullTriangleDown), 0);
  rb_define_method (rb_cObject, "kOpenCircle", VALUEFUNC (rr_kOpenCircle), 0);
  rb_define_method (rb_cObject, "kOpenSquare", VALUEFUNC (rr_kOpenSquare), 0);
  rb_define_method (rb_cObject, "kOpenTriangleUp",
          VALUEFUNC (rr_kOpenTriangleUp), 0);
  rb_define_method (rb_cObject, "kOpenDiamond", VALUEFUNC (rr_kOpenDiamond),
          0);
  rb_define_method (rb_cObject, "kOpenCross", VALUEFUNC (rr_kOpenCross), 0);
  rb_define_method (rb_cObject, "kFullStar", VALUEFUNC (rr_kFullStar), 0);
  rb_define_method (rb_cObject, "kOpenStar", VALUEFUNC (rr_kOpenStar), 0);
}
