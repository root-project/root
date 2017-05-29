/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// This example show how to put some annotation on a 3D plot using 3D
/// polylines. It also demonstrates how the axis labels can be modified.
/// It was created for the book:
/// [Statistical Methods for Data Analysis in Particle Physics](http://www.springer.com/la/book/9783319201757)
/// \macro_image
/// \macro_code
///
/// \author Luca Lista

void annotation3d()
{
   TCanvas *c = new TCanvas("c", "c", 600, 600);
   c->SetTheta(30);
   c->SetPhi(50);
   gStyle->SetOptStat(0);
   gStyle->SetHistTopMargin(0);
   gStyle->SetOptTitle(kFALSE);

   // Define and draw a surface
   TF2 *f = new TF2("f", "[0]*cos(x)*cos(y)", -1, 1, -1, 1);
   f->SetParameter(0, 1);
   double s = 1./f->Integral(-1, 1, -1, 1);
   f->SetParameter(0, s);
   f->SetNpx(50);
   f->SetNpy(50);

   f->GetXaxis()->SetTitle("x");
   f->GetXaxis()->SetTitleOffset(1.4);
   f->GetXaxis()->SetTitleSize(0.04);
   f->GetXaxis()->CenterTitle();
   f->GetXaxis()->SetNdivisions(505);
   f->GetXaxis()->SetTitleOffset(1.3);
   f->GetXaxis()->SetLabelSize(0.03);
   f->GetXaxis()->ChangeLabel(2,-1,-1,-1,kRed,-1,"X_{0}");

   f->GetYaxis()->SetTitle("y");
   f->GetYaxis()->CenterTitle();
   f->GetYaxis()->SetTitleOffset(1.4);
   f->GetYaxis()->SetTitleSize(0.04);
   f->GetYaxis()->SetTitleOffset(1.3);
   f->GetYaxis()->SetNdivisions(505);
   f->GetYaxis()->SetLabelSize(0.03);

   f->GetZaxis()->SetTitle("dP/dx");
   f->GetZaxis()->CenterTitle();
   f->GetZaxis()->SetTitleOffset(1.3);
   f->GetZaxis()->SetNdivisions(505);
   f->GetZaxis()->SetTitleSize(0.04);
   f->GetZaxis()->SetLabelSize(0.03);

   f->SetLineWidth(1);
   f->SetLineColorAlpha(kAzure-2, 0.3);

   f->Draw("surf1 fb");

   // Lines for 3D annotation
   double x[11] = {-0.500, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.500};
   double y[11] = {-0.985, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4,  0.6,  0.8,  0.985};
   double z[11];
   for (int i = 0; i < 11; ++i) z[i] = s*cos(x[i])*cos(y[i]);
   TPolyLine3D *g2 = new TPolyLine3D(11, x, y, z);

   double xx[2] = {-0.5, -0.5};
   double yy[2] = {-0.985, -0.985};
   double zz[2] = {0.11, s*cos(-0.5)*cos(-0.985)};
   TPolyLine3D *l2 = new TPolyLine3D(2, xx, yy, zz);

   g2->SetLineColor(kRed);
   g2->SetLineWidth(3);
   g2->Draw();

   l2->SetLineColor(kRed);
   l2->SetLineStyle(2);
   l2->SetLineWidth(1);
   l2->Draw();

   // Draw text Annotations
   TLatex *txt = new TLatex(0.05, 0, "f(y,x_{0})");
   txt->SetTextFont(42);
   txt->SetTextColor(kRed);
   txt->Draw();

   TLatex *txt1 = new TLatex(0.12, 0.52, "f(x,y)");
   txt1->SetTextColor(kBlue);
   txt1->SetTextFont(42);
   txt1->Draw();
}
