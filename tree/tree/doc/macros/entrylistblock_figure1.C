{
   TCanvas *c = new TCanvas("c", "c",49,123,752,435);
   c->Range(0,0,1,1);
   c->SetBorderSize(2);
   c->SetFrameFillColor(0);

   TPaveText *pt = new TPaveText(0.00313972,0.650146,0.22135,0.772595,"br");
   pt->SetFillColor(kWhite);
   TText *text = pt->AddText("TEntryListBlock");
   pt->Draw();

   pt = new TPaveText(0.00313972,0.827988,0.675039,0.994169,"br");
   pt->SetFillColor(kWhite);
   pt->SetTextColor(4);
   text = pt->AddText("Indices representation in a TEntryListBlock");
   pt->Draw();

   pt = new TPaveText(0.00410678,0.412955,0.221766,0.651822,"br");
   pt->SetFillColor(kWhite);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.048583);
   text = pt->AddText("UShort_t* fIndices");
   text = pt->AddText("Int_t fType");
   pt->Draw();

   pt = new TPaveText(0.324961,0.708455,0.959184,0.804665,"br");
   pt->SetFillColor(kWhite);
   text = pt->AddText("Suppose,that this block stores entries");
   text = pt->AddText("0, 2, 4, 10, 11, 12");
   pt->Draw();

   pt = new TPaveText(0.232227,0.541176,0.333333,0.641176,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fType=0");
   text->SetTextAngle(-15);
   pt->Draw();

   pt = new TPaveText(0.355114,0.189066,0.457386,0.255125,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fIndices[0]");
   pt->Draw();

   pt = new TPaveText(0.521193,0.38484,0.77708,0.48105,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fIndices[0]");
   pt->Draw();

   pt = new TPaveText(0.355619,0.239726,0.458037,0.305936,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   pt->SetTextSize(0.0342466);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.264241,0.383481,0.363924,0.486726,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fType=1");
   text->SetTextAngle(-50);
   pt->Draw();

   pt = new TPaveText(0.458807,0.173121,0.559659,0.273349,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fIndices[1]");
   pt->Draw();

   pt = new TPaveText(0.473684,0.251142,0.540541,0.299087,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   pt->SetTextSize(0.0342466);
   text = pt->AddText("2");
   pt->Draw();

   pt = new TPaveText(0.556818,0.193622,0.659091,0.250569,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fIndices[2]");
   pt->Draw();

   pt = new TPaveText(0.55761,0.244292,0.660028,0.30137,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   pt->SetTextSize(0.0342466);
   text = pt->AddText("4");
   pt->Draw();

   pt = new TPaveText(0.659091,0.191344,0.758523,0.255125,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fIndices[3]");
   pt->Draw();

   pt = new TPaveText(0.657183,0.239726,0.756757,0.303653,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   pt->SetTextSize(0.0342466);
   text = pt->AddText("10");
   pt->Draw();

   pt = new TPaveText(0.759943,0.189066,0.859375,0.255125,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fIndices[4]");
   pt->Draw();

   pt = new TPaveText(0.758179,0.239726,0.857752,0.305936,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   pt->SetTextSize(0.0342466);
   text = pt->AddText("11");
   pt->Draw();

   pt = new TPaveText(0.859943,0.189066,0.959375,0.255125,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("fIndices[5]");
   pt->Draw();

   pt = new TPaveText(0.852063,0.239726,0.951636,0.305936,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   pt->SetTextSize(0.0342466);
   text = pt->AddText("12");
   pt->Draw();

   pt = new TPaveText(0.786325,0.503432,0.830484,0.549199,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("1");
   pt->Draw();

   pt = new TPaveText(0.750712,0.503432,0.796296,0.549199,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("1");
   pt->Draw();

   pt = new TPaveText(0.825472,0.5,0.871069,0.54386,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.863208,0.5,0.908805,0.54386,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.902516,0.5,0.948113,0.54386,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.677673,0.5,0.72327,0.54386,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.639937,0.5,0.685535,0.54386,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.602201,0.5,0.647799,0.54386,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.564465,0.5,0.610063,0.54386,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.529874,0.5,0.575472,0.54386,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.45283,0.502924,0.498428,0.546784,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.380503,0.502924,0.426101,0.546784,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("0");
   pt->Draw();

   pt = new TPaveText(0.710826,0.503432,0.766382,0.549199,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("1");
   pt->Draw();

   pt = new TPaveText(0.487179,0.505721,0.532764,0.551487,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   pt->SetTextSize(0.0389016);
   text = pt->AddText("1");
   pt->Draw();

   pt = new TPaveText(0.413105,0.501144,0.460114,0.549199,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   pt->SetTextSize(0.0389016);
   text = pt->AddText("1");
   pt->Draw();

   pt = new TPaveText(0.344729,0.505721,0.393162,0.551487,"br");
   pt->SetBorderSize(0);
   pt->SetFillColor(kWhite);
   text = pt->AddText("1");
   pt->Draw();
   TArrow *arrow = new TArrow(0.225552,0.572271,0.35489,0.283186,0.03,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   arrow = new TArrow(0.22082,0.581121,0.344937,0.519174,0.03,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   TLine *line = new TLine(0.35,0.5,0.95,0.5);
   line->Draw();
   line = new TLine(0.35,0.49,0.35,0.51);
   line->Draw();
   line = new TLine(0.3875,0.49,0.3875,0.51);
   line->Draw();
   line = new TLine(0.419,0.49,0.419,0.51);
   line->Draw();
   line = new TLine(0.4565,0.49,0.4565,0.51);
   line->Draw();
   line = new TLine(0.494,0.49,0.494,0.51);
   line->Draw();
   line = new TLine(0.5315,0.49,0.5315,0.51);
   line->Draw();
   line = new TLine(0.569,0.49,0.569,0.51);
   line->Draw();
   line = new TLine(0.6065,0.49,0.6065,0.51);
   line->Draw();
   line = new TLine(0.644,0.48,0.644,0.52);
   line->Draw();
   line = new TLine(0.6815,0.49,0.6815,0.51);
   line->Draw();
   line = new TLine(0.719,0.49,0.719,0.51);
   line->Draw();
   line = new TLine(0.7565,0.49,0.7565,0.51);
   line->Draw();
   line = new TLine(0.794,0.49,0.794,0.51);
   line->Draw();
   line = new TLine(0.8315,0.49,0.8315,0.51);
   line->Draw();
   line = new TLine(0.869,0.49,0.869,0.51);
   line->Draw();
   line = new TLine(0.9065,0.49,0.9065,0.51);
   line->Draw();
   line = new TLine(0.944,0.49,0.944,0.51);
   line->Draw();
   line = new TLine(0.944,0.49,0.944,0.51);
   line->Draw();
   line = new TLine(0.36,0.251142,0.96,0.251142);
   line->Draw();
   line = new TLine(0.36,0.24,0.36,0.26);
   line->Draw();
   line = new TLine(0.46,0.24,0.46,0.26);
   line->Draw();
   line = new TLine(0.56,0.24,0.56,0.26);
   line->Draw();
   line = new TLine(0.66,0.24,0.66,0.26);
   line->Draw();
   line = new TLine(0.76,0.24,0.76,0.26);
   line->Draw();
   line = new TLine(0.86,0.24,0.86,0.26);
   line->Draw();
   line = new TLine(0.96,0.24,0.96,0.26);
   line->Draw();
}
