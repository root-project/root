#include <cassert>
#include <TError.h>
#include <TGeoMatrix.h>

void myassert(bool condition, const char *msg)
{
  if (!condition)
    ::Fatal("", "%s", msg);
}

void TestMatrix()
{
  TGeoHMatrix identity;
  /** translations **/
  TGeoTranslation tr1(1., 2., 3.);
  // Copy ctor
  TGeoTranslation tr2(tr1);
  myassert(tr2 == tr1, "translation copy ctor wrong");
  // Assignment
  tr2 = tr1;
  myassert(tr2 == tr1, "translation assignment wrong");
  myassert(tr2.IsTranslation(), "translation flag not set");
  // Composition
  TGeoTranslation trref1(2., 4., 6.);
  TGeoTranslation tr3 = tr1 * tr2;
  myassert(tr3 == trref1, "translation multiplication wrong");
  tr2 *= tr1;
  myassert(tr2 == trref1, "translation inplace multiplication wrong");
  tr2 *= tr2.Inverse();
  myassert(tr2 == identity, "translation inverse wrong");

  /** rotations **/
  TGeoRotation r1;
  r1.RotateZ(90.);
  // Copy ctor
  TGeoRotation r2(r1);
  myassert(r2 == r1, "rotation copy ctor wrong");
  // Assignment
  r2 = r1;
  myassert(r2 == r1, "rotation assignment wrong");
  myassert(r2.IsRotation(), "rotation flag not set");
  // Composition
  TGeoRotation r3  = r1 * r1 * r1 * r1;
  myassert(r3 == identity, "rotation multiplication wrong");
  r2 *= r2.Inverse();
  myassert(r2 == identity, "rotation inplace multiplication wrong");

   /** scale **/
  TGeoScale scl1(1., 2., 3.);
  // Copy ctor
  TGeoScale scl2(scl1);
  myassert(scl2 == scl1, "scale copy ctor wrong");
  // Assignment
  scl2 = scl1;
  myassert(scl2 == scl1, "scale assignment wrong");
  myassert(scl2.IsScale(), "scale flag not set");
  // Composition
  TGeoScale sclref1(1., 4., 9.);
  TGeoScale scl3 = scl1 * scl2;
  myassert(scl3 == sclref1, "scale multiplication wrong");
  scl2 *= scl1;
  myassert(scl2 == sclref1, "scale inplace multiplication wrong");
  scl2 *= scl2.Inverse();
  myassert(scl2 == identity, "scale inverse wrong");

  /** HMatrix **/
  // Copy constructor
  TGeoHMatrix h1(tr1);
  myassert(h1 == tr1 && h1.IsTranslation(), "hmatrix constructor from translation wrong");

  // Assignment
  TGeoHMatrix h2 = r1;
  TGeoHMatrix h3 = scl1;
  myassert(h2 == r1 && h3 == scl1 && h2.IsRotation() && h3.IsScale(), "hmatrix assignment wrong");

  TGeoHMatrix h4 = h1 * h2;
  myassert(tr1 == h4 && r1 == h4 && h4.IsTranslation() && h4.IsRotation(), "hmatrix multiplication wrong");

  h4 *= h4.Inverse();
  myassert(h4 == identity, "hmatrix inverse wrong");

  /** Combi trans **/
  // Copy constructor
  TGeoCombiTrans c1(tr1);
  myassert(c1 == tr1 && c1.IsTranslation(), "combi trans constructor from translation wrong");

  // Assignment
  TGeoCombiTrans c2 = r1;
  myassert(c2 == r1 && c2.IsRotation(), "combi trans assignment wrong");

  TGeoCombiTrans c3 = c1 * c2;
  myassert(tr1 == c3 && r1 == c3 && c3.IsTranslation() && c3.IsRotation(), "combi trans multiplication wrong");

  c3 *= c3.Inverse();
  myassert(c3 == identity, "combi trans inverse wrong");

  TGeoCombiTrans c4(1., 2., 3., &r1);
  c4.Print();
  TGeoCombiTrans c5(c4);
  c5.Print();
  myassert(c4.GetRotation() == &r1 && c5.GetRotation() != &r1, "combi trans copy constructor wrong");

  // Test for Wolfgang Korsch's case
  TGeoHMatrix href = tr1;
  href *= TGeoRotation("", 0, 120, 0);
  TGeoRotation r4;
  r4.SetAngles(0, 90, 0);
  TGeoRotation r5 = r4;
  r4.SetAngles(0, 30, 0);
  r5 = r5 * r4;
  TGeoHMatrix combiH1TB = tr1 * r5;
  myassert(combiH1TB == href, "translation multiplication demoted");

  // Test for David Rohr's case
  TGeoHMatrix h5 = href, h6 = href;
  h5 *= h6.Inverse();
  h6.Multiply(h6.Inverse());
  myassert(h5 == h6 && h5 == identity, "inverse not matching");
}
