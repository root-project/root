// variants: 0  - custom values
//           1  - default values
//           2+ - old default values
void canvas_divide_example(int use_variant = 0)
{
   auto wx = 600; // width and heigh
   auto wy = 400;

   auto nx = 3; // top-level pad division
   auto ny = 2;

   auto ml = 0.30; // top-level pad margins
   auto mb = 0.10;
   auto mr = 0.05;
   auto mt = 0.10;

   auto c = new TCanvas("canvas_divide", "canvas_divide", wx, wy);
   c->SetFillColor(19);

   if (use_variant == 0) {
      c->SetMargin(ml, mr, mb, mt);
      c->Divide(nx, ny, 0.03, 0.05, 46);
   } else if (use_variant == 1) {
      c->Divide(nx, ny, 0.01, 0.01, 46);
   } else {
      c->SetMargin(0.01, 0.01, 0.01, 0.01);
      c->Divide(nx, ny, 0.02, 0.02, 46);
   }

   ml = c->GetLeftMargin();
   mb = c->GetBottomMargin();
   mr = c->GetRightMargin();
   mt = c->GetTopMargin();

   auto h = new TH1F("", "", 100, -3.3, 3.3);
   h->GetXaxis()->SetLabelFont(43);
   h->GetXaxis()->SetLabelSize(12);
   h->GetYaxis()->SetLabelFont(43);
   h->GetYaxis()->SetLabelSize(12);
   h->GetYaxis()->SetNdivisions(505);
   h->SetMaximum(30 * nx * ny);
   h->SetFillColor(41);

   Int_t number = 0;
   for (Int_t i = 0; i < nx * ny; i++) {
      number++;
      c->cd(number);
      h->FillRandom("gaus", 1000);
      h->DrawCopy();
   }

   c->cd();

   TArrow arr;

   arr.DrawArrow(0, 0.5, ml, 0.5, 0.01, "<|>");
   arr.DrawArrow(0.5, 0, 0.5, mb, 0.01, "<|>");
   arr.DrawArrow(1 - mr, 0.5, 1, 0.5, 0.01, "<|>");
   arr.DrawArrow(0.5, 1 - mt, 0.5, 1, 0.01, "<|>");

   TLatex tex_x;
   tex_x.SetNDC(1);
   tex_x.SetTextSize(0.03);
   tex_x.SetTextAlign(12);
   tex_x.SetTextAngle(90);

   TLatex tex_y;
   tex_y.SetNDC(1);
   tex_y.SetTextSize(0.03);
   tex_y.SetTextAlign(12);

   tex_x.DrawLatex(ml / 2, 0.5, TString::Format(" ml = %.2f", ml));
   tex_x.DrawLatex(1 - mr / 2, 0.5, TString::Format(" mr = %.2f", mr));

   tex_y.DrawLatex(0.5, mb / 2, TString::Format(" mb = %.2f", mb));
   tex_y.DrawLatex(0.5, 1 - mt / 2, TString::Format(" mt = %.2f", mt));

   for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {

         float x1, x2, xc, y1, y2, yc;

         auto spad = c->GetPad(1 + j * nx + i); // current pad
         x1 = spad->GetXlowNDC() + spad->GetWNDC();
         xc = spad->GetXlowNDC() + spad->GetWNDC() / 2;
         if (i < (nx - 1)) {
            auto spad_nx = c->GetPad(1 + j * nx + (i + 1)); // next pad in x
            x2 = spad_nx->GetXlowNDC();
         }
         auto xm = x2 - x1;

         if (j < (ny - 1)) {
            auto spad_ny = c->GetPad(1 + (j + 1) * nx + i); // next pad in y
            y1 = spad_ny->GetYlowNDC() + spad->GetHNDC();
         }
         y2 = spad->GetYlowNDC();
         yc = spad->GetYlowNDC() + spad->GetHNDC() / 2;
         auto ym = y2 - y1;

         if (i < (nx - 1)) {
            arr.DrawArrow(x1, yc, x2, yc, 0.01, "<|>");
            tex_x.DrawLatex((x1 + x2) / 2, yc, TString::Format(" xm = %.2f", xm));
         }
         if (j < (ny - 1)) {
            arr.DrawArrow(xc, y1, xc, y2, 0.01, "<|>");
            tex_y.DrawLatex(xc, (y1 + y2) / 2, TString::Format(" ym = %.2f", ym));
         }
      }
   }

   TText text;
   text.SetTextSize(0.03);
   text.SetTextFont(102);
   text.SetNDC(1);

   text.DrawText(0.01, 0.97, "c->SetMargin(ml, mr, mb, mt);");
   text.DrawText(0.01, 0.94, "c->Divide(nx, ny, xm, ym);");
}
