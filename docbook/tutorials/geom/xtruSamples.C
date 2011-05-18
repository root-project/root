void xtruSamples()
{
// Draw a sample of TXTRU shapes some convex, concave (and possibly malformed)
// Change Bool_t's to test alternative specifications
// Author: Robert Hatcher (rhatcher@fnal.gov) 2000.09.06

// One normally specifies the x-y points in counter-clockwise order;
// flip this to TRUE to test that it doesn't matter.
  Bool_t makecw      = kFALSE;

// One normally specifies the z points in increasing z order;
// flip this to TRUE to test that it doesn't matter.
  Bool_t reversez    = kFALSE;

// One shouldn't be creating malformed polygons
// but to test what happens when one does here's a flag.
// The effect will be only apparent in solid rendering mode
  Bool_t domalformed = kFALSE;
//  domalformed = kTRUE;

  c1 = new TCanvas("c1","sample TXTRU Shapes",200,10,640,640);

// Create a new geometry
  TGeometry* geom = new TGeometry("sample","sample");
  geom->cd();

// Define the complexity of the drawing
  Float_t zseg   = 6;  // either 2 or 6
  Int_t extravis = 0;  // make extra z "arrow" visible

  Float_t unit = 1;

// Create a large BRIK to embed things into
  Float_t bigdim = 12.5*unit;
  TBRIK* world = new TBRIK("world","world","void",bigdim,bigdim,bigdim);

  // Create the main node, make it invisible
  TNode* worldnode = new TNode("worldnode","world node",world);
  worldnode->SetVisibility(0);
  worldnode->cd();

// Canonical shape ... gets further modified by scale factors
// to create convex (and malformed) versions
  Float_t x[] = { -0.50, -1.20,  1.20,  0.50,  0.50,  1.20, -1.20, -0.50 };
  Float_t y[] = { -0.75, -2.00, -2.00, -0.75,  0.75,  2.00,  2.00,  0.75 };
  Float_t z[] = { -0.50, -1.50, -1.50,  1.50,  1.50,  0.50 };
  Float_t s[] = {  0.50,  1.00,  1.50,  1.50,  1.00,  0.50 };
  Int_t   nxy = sizeof(x)/sizeof(Float_t);
  Float_t convexscale[] = { 7.0, -1.0, 1.5 };

  Int_t icolor[] = { 1, 2, 3, 2, 2, 2, 4, 2, 6 };

// xycase and zcase:  0=convex, 1=malformed, 2=concave
// this will either create a 2x2 matrix of shapes
// or a 3x3 array (if displaying malformed versions)
  for (Int_t zcase = 0; zcase<3; zcase++) {
     if (zcase == 1 && !domalformed) continue;
     for (Int_t xycase = 0; xycase<3; xycase++) {
        if (xycase == 1 && !domalformed) continue;

        Char_t *name = "txtruXYZ";
        sprintf(name,"txtru%1d%1d%1d",xycase,zcase,zseg);
        TXTRU* mytxtru = new TXTRU(name,name,"void",8,2);

        Int_t i, j;
        Float_t xsign = (makecw) ? -1 : 1;
        Float_t zsign = (reversez) ? -1 : 1;

        // set the vertex points
        for (i=0; i<nxy; i++) {
           Float_t xtmp = x[i]*xsign;
           Float_t ytmp = y[i];
           if (i==0||i==3||i==4||i==7) xtmp *= convexscale[xycase];
           if (xycase==2) xtmp *=2;
           mytxtru->DefineVertex(i,xtmp,ytmp);
        }
        // set the z segment positions and scales
        for (i=0, j=0; i<zseg; i++) {
           Float_t ztmp = z[i]*zsign;
           if (i==0||i==5) ztmp *= convexscale[zcase];
           if (zcase==2) ztmp *= 2.5;
           if (zseg>2 && zcase!=2 && (i==1||i==4)) continue;
           mytxtru->DefineSection(j,ztmp,s[i]);
           j++;
        }

        TNode* txtrunode = new TNode(name,name,mytxtru);
        txtrunode->SetLineColor(icolor[3*zcase+xycase]);
        Float_t pos_scale = (domalformed) ? 10 : 6;
        Float_t xpos = (xycase-1)*pos_scale*unit;
        Float_t ypos = (zcase-1)*pos_scale*unit;
        txtrunode->SetPosition(xpos,ypos,0.);
     }
  }


// Some extra shapes to show the direction of "z"
  Float_t zhalf = 0.5*bigdim;
  Float_t rmax = 0.03*bigdim;
  TCONE* zcone = new TCONE("zcone","zcone","void",zhalf,0.,rmax,0.,0.);
  zcone->SetVisibility(extravis);
  TNode* zconenode = new TNode("zconenode","zconenode",zcone);
  zconenode->SetLineColor(3);

  Float_t dzstub = 2*rmax; 
  TBRIK* zbrik = new TBRIK("zbrik","zbrik","void",rmax,rmax,dzstub);
  zbrik->SetVisibility(extravis);
  TNode* zbriknode = new TNode("zbriknode","zbriknode",zbrik);
  zbriknode->SetPosition(0.,0.,zhalf+dzstub);
  zbriknode->SetLineColor(3);

//  geom->ls();

  geom->Draw();

// Tweak the pad so that it displays the entire geometry undistorted
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
    thisPad->Modified();
    thisPad->Update();
  }

}
