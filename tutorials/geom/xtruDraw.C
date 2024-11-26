/// \file
/// \ingroup tutorial_geom
/// Draw a "representative" TXTRU shape
///
/// \macro_image
/// \macro_code
///
/// \author Robert Hatcher (rhatcher@fnal.gov) 2000.09.06

void xtruDraw() {
  TCanvas *canvas = new TCanvas("xtru","Example XTRU object",200,10,640,640);

// Create a new geometry
  TGeometry* geometry = new TGeometry("geometry","geometry");
  geometry->cd();

  TXTRU* atxtru = new TXTRU("atxtru","atxtru","void",5,2);

// outline and z segment specifications

  Float_t x[] =
    {   -177.292,   -308.432,   -308.432,   -305.435,   -292.456,    -280.01
    ,    -241.91,    -241.91,   -177.292,   -177.292,    177.292,    177.292
    ,     241.91,     241.91,     280.06,    297.942,    305.435,    308.432
    ,    308.432,    177.292,    177.292,   -177.292 };
  Float_t y[] =
    {    154.711,    23.5712,     1.1938,     1.1938,     8.6868,     8.6868
    ,    -3.7592,   -90.0938,   -154.711,   -190.602,   -190.602,   -154.711
    ,   -90.0938,    -3.7592,     8.6868,     8.6868,     1.1938,     1.1938
    ,    23.5712,    154.711,    190.602,    190.602 };
  Float_t z[] =
    {       0.00,      500.0 };
  Float_t scale[] =
    {       1.00,       1.00 };
  Float_t x0[] =
    {          0,          0 };
  Float_t y0[] =
    {          0,          0 };

  Int_t i;

  Int_t nxy = sizeof(x)/sizeof(Float_t);
  for (i=0; i<nxy; i++) {
     atxtru->DefineVertex(i,x[i],y[i]);
  }

  Int_t nz = sizeof(z)/sizeof(Float_t);
  for (i=0; i<nz; i++) {
     atxtru->DefineSection(i,z[i],scale[i],x0[i],y0[i]);
  }

// Define a TNode where this example resides in the TGeometry
// Draw the TGeometry

  TNode* anode = new TNode("anode","anode",atxtru);
  anode->SetLineColor(1);

  geometry->Draw();

// Tweak the pad scales so as not to distort the shape

  TVirtualPad *thisPad = gPad;
  if (thisPad) {
    TView *view = thisPad->GetView();
    if (!view) return;
    Double_t min[3],max[3],center[3];
    view->GetRange(min,max);
    int i;
    // Find the boxed center
    for (i=0;i<3; i++) center[i] = 0.5*(max[i]+min[i]);
    Double_t maxSide = 0;
    // Find the largest side
    for (i=0;i<3; i++) maxSide = TMath::Max(maxSide,max[i]-center[i]);
    file://Adjust scales:
    for (i=0;i<3; i++) {
       max[i] = center[i] + maxSide;
       min[i] = center[i] - maxSide;
    }
    view->SetRange(min,max);
    Int_t ireply;
    thisPad->Modified();
    thisPad->Update();
  }

}
