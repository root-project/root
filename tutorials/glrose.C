void glrose()
{
   const Int_t paletteSize = 10;
   Float_t rgb[paletteSize * 3] = 
      {0.80f, 0.55f, 0.40f, 
       0.85f, 0.60f, 0.45f, 
       0.90f, 0.65f, 0.50f, 
       0.95f, 0.70f, 0.55f, 
       1.f,   0.75f, 0.60f,
       1.f,   0.80f, 0.65f,
       1.f,   0.85f, 0.70f,
       1.f,   0.90f, 0.75f,
       1.f,   0.95f, 0.80f,
       1.f,   1.f,   0.85f};

   Int_t palette[paletteSize] = {0};

   for (Int_t i = 0; i < paletteSize; ++i)
      palette[i] = TColor::GetColor(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);

   gStyle->SetPalette(paletteSize, palette);
   gStyle->SetCanvasPreferGL(true);

   TF2 *fun = new TF2("a", "cos(y)*sin(x)+cos(x)*sin(y)", -6, 6, -6, 6);
   fun->SetContour(paletteSize);
   fun->SetNpx(30);
   fun->SetNpy(30);
   fun->Draw("glsurf2pol");
}
