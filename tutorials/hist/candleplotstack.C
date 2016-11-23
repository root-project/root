/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Example showing how a THStack with candle plot option.
///
/// \macro_image
/// \macro_code
///
/// \authors Georg Troska, Olivier Couet

void candleplotstack()
{
   gStyle->SetTimeOffset(0);
   TRandom *randnum      = new TRandom();
   TDatime *dateBegin = new TDatime(2010,1,1,0,0,0);
   TDatime *dateEnd   = new TDatime(2011,1,1,0,0,0);
   TH2I *h1 = new TH2I("h1","Machine A",12,dateBegin->Convert(),dateEnd->Convert(),1000,0,1000);
   TH2I *h2 = new TH2I("h2","Machine B",12,dateBegin->Convert(),dateEnd->Convert(),1000,0,1000);
   TH2I *h3 = new TH2I("h3","Machine C",12,dateBegin->Convert(),dateEnd->Convert(),1000,0,1000);

   float Rand;
   for (int i = dateBegin->Convert(); i < dateEnd->Convert(); i+=86400*30) {
      for (int j = 0; j < 1000; j++) {
         Rand = randnum->Gaus(500+sin(i/10000000.)*100,50); h1->Fill(i,Rand);
         Rand = randnum->Gaus(500+sin(i/11000000.)*100,70); h2->Fill(i,Rand);
         Rand = randnum->Gaus(500+sin(i/11000000.)*100,90); h3->Fill(i,Rand);
      }
   }

   h2->SetLineColor(kRed);
   h3->SetLineColor(kRed+3);
   TCanvas *c1 = new TCanvas();

   THStack *hs = new THStack("hs","Machine A+B+C");
   hs->Add(h1);
   hs->Add(h2);
   hs->Add(h3,"candle2");
   hs->Draw("candle3");

   hs->GetXaxis()->SetTimeDisplay(1);
   hs->GetXaxis()->SetTimeFormat("%m/%y");
   hs->GetXaxis()->SetTitle("Date [month/year]");

   c1->Modified();

   gPad->BuildLegend(0.75,0.75,0.95,0.95,"");
}
