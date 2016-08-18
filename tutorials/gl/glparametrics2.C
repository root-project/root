/// \file
/// \ingroup tutorial_gl
/// Show rendering of parametric surfaces.
///
/// A parametric surface is defined by three functions:
/// S(u, v) : {x(u, v), y(u, v), z(u, v)}.
/// To create parametric surface and draw it one has to:
///  1. Create canvas, which support OpenGL drawing (two ways):
///     - Call gStyle->SetCanvasPreferGL(kTRUE)
///     - Or create canvas with name, wich contains "gl".
///  2. create TGLParametricEquation object.
/// ~~~{.cpp}
///     TGLParametricEquation *eq = new TGLParametricEquation("name",
///     "some FORMULA here - x(u, v)",
///     "some FORMULA here - y(u, v)",
///     "some FORMULA here - z(u, v)",
///     uMin, uMax, vMin, vMax);
/// ~~~
///     where FORMULA is the same string (mathematical expression),
///     as in TF2, but you should use 'u' (or 'U') instead of 'x'
///     and 'v' (or 'V') instead of 'y'.
///  3. Call equation->Draw();
///     Parametric surfaces support 21 color "schemes", you can change
///     the color:
///     - place mouse cursor above surface (surface is selected in pad)
///     - press 's' or 'S'.
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \author  Timur Pocheptsov

void klein_bottle(TGLVertex3 &dst, Double_t u, Double_t v)
{
    using namespace TMath;

    const Double_t r = 4. * (1. - Cos(u) / 2.);
    if (u < Pi()) {
        dst.X() = 6 * Cos(u) * (1. + Sin(u)) + r * Cos(u) * Cos(v);
        dst.Y() = 16 * Sin(u) + r * Sin(u) * Cos(v);
    } else {
        dst.X() = 6 * Cos(u) * (1. + Sin(u)) + r * Cos(v + Pi());
        dst.Y() = 16 * Sin(u);
    }
    dst.Z() = r * Sin(v);
}

void glparametrics2()
{
   gStyle->SetCanvasPreferGL(kTRUE);
   TCanvas *c = new TCanvas("canvas","Parametric surfaces with gl", 100, 10, 700, 700);

   c->Divide(2, 2);
   c->cd(1);
   TGLParametricEquation *p1 = new TGLParametricEquation("Shell",
                                "1.2 ^ v * sin(u) ^ 2 * sin(v)",
                                "1.2 ^ v * sin(u) * cos(u)",
                                "1.2 ^ v * sin(u) ^ 2 * cos(v)",
                                0., TMath::Pi(), // 0 <= u <= pi
                                -TMath::Pi() / 4., 5 * TMath::Pi() / 2.); // -pi/4 <= v <= 5*pi/2
   p1->Draw("");

   c->cd(2);
   TGLParametricEquation *p2 = new TGLParametricEquation("Limpet torus",
                                    "cos(u) / (sqrt(2) + sin(v))",
                                    "sin(u) / (sqrt(2) + sin(v))",
                                    "1. / (sqrt(2) + cos(v))",
                                    -TMath::Pi(), TMath::Pi(),
                                    -TMath::Pi(), TMath::Pi());
   p2->Draw();

   c->cd(3);
   TGLParametricEquation *p3 = new TGLParametricEquation("Klein bottle",
                                        klein_bottle,
                                        0., TMath::TwoPi(),
                                        0., TMath::TwoPi());
   p3->Draw();

   c->cd(4);
   TGLParametricEquation *p4 = new TGLParametricEquation("Helicoid",
                                                         "v * cos(u)",
                                                         "v * sin(u)",
                                                         "u",
                                                         -3., 3.,
                                                         -3., 3.);
   p4->Draw();
}
