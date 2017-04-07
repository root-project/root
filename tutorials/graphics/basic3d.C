/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Show 3-D polylines and markers.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void basic3d(){
   TCanvas *c1 = new TCanvas("c1","PolyLine3D & PolyMarker3D Window",200,10,700,500);

   // create a pad
   TPad *p1 = new TPad("p1","p1",0.05,0.02,0.95,0.82,46,3,1);
   p1->Draw();
   p1->cd();

   // creating a view
   TView *view = TView::CreateView(1);
   view->SetRange(5,5,5,25,25,25);

   // create a first PolyLine3D
   TPolyLine3D *pl3d1 = new TPolyLine3D(5);

   // set points
   pl3d1->SetPoint(0, 10, 10, 10);
   pl3d1->SetPoint(1, 15, 15, 10);
   pl3d1->SetPoint(2, 20, 15, 15);
   pl3d1->SetPoint(3, 20, 20, 20);
   pl3d1->SetPoint(4, 10, 10, 20);
   // set attributes
   pl3d1->SetLineWidth(3);
   pl3d1->SetLineColor(5);

   // create a second PolyLine3D
   TPolyLine3D *pl3d2 = new TPolyLine3D(4);

   // set points
   pl3d2->SetPoint(0, 5, 10, 5);
   pl3d2->SetPoint(1, 10, 15, 8);
   pl3d2->SetPoint(2, 15, 15, 18);
   pl3d2->SetPoint(3, 5, 20, 20);
   pl3d2->SetPoint(4, 10, 10, 5);

   // set attributes
   pl3d2->SetLineWidth(5);
   pl3d2->SetLineColor(2);

   // create a first PolyMarker3D
   TPolyMarker3D *pm3d1 = new TPolyMarker3D(12);

   // set points
   pm3d1->SetPoint(0, 10, 10, 10);
   pm3d1->SetPoint(1, 11, 15, 11);
   pm3d1->SetPoint(2, 12, 15, 9);
   pm3d1->SetPoint(3, 13, 17, 20);
   pm3d1->SetPoint(4, 14, 16, 15);
   pm3d1->SetPoint(5, 15, 20, 15);
   pm3d1->SetPoint(6, 16, 18, 10);
   pm3d1->SetPoint(7, 17, 15, 10);
   pm3d1->SetPoint(8, 18, 22, 15);
   pm3d1->SetPoint(9, 19, 28, 25);
   pm3d1->SetPoint(10, 20, 12, 15);
   pm3d1->SetPoint(11, 21, 12, 15);

   // set marker size, color & style
   pm3d1->SetMarkerSize(2);
   pm3d1->SetMarkerColor(4);
   pm3d1->SetMarkerStyle(2);

   // create a second PolyMarker3D
   TPolyMarker3D *pm3d2 = new TPolyMarker3D(8);

   pm3d2->SetPoint(0, 22, 15, 15);
   pm3d2->SetPoint(1, 23, 18, 21);
   pm3d2->SetPoint(2, 24, 26, 13);
   pm3d2->SetPoint(3, 25, 17, 15);
   pm3d2->SetPoint(4, 26, 20, 15);
   pm3d2->SetPoint(5, 27, 15, 18);
   pm3d2->SetPoint(6, 28, 20, 10);
   pm3d2->SetPoint(7, 29, 20, 20);

   // set marker size, color & style
   pm3d2->SetMarkerSize(2);
   pm3d2->SetMarkerColor(1);
   pm3d2->SetMarkerStyle(8);

   // draw
   pl3d1->Draw();
   pl3d2->Draw();
   pm3d1->Draw();
   pm3d2->Draw();
   //
   // draw a title/explanation in the canvas pad
   c1->cd();
   TPaveText *title = new TPaveText(0.1,0.85,0.9,0.97);
   title->SetFillColor(24);
   title->AddText("Examples of 3-D primitives");
   TText *click=title->AddText("Click anywhere on the picture to rotate");
   click->SetTextColor(4);
   title->Draw();
}
