{
//=========Macro generated from canvas: c/c
//=========  (Tue Jan 23 16:58:56 2007) by ROOT version5.15/01
   TCanvas *c = new TCanvas("c", "c",213,172,460,253);
   c->Range(0,0,1,1);
   c->SetBorderSize(2);
   c->SetFrameFillColor(0);
   
   TPaveText *pt = new TPaveText(0.00518135,0.810811,0.507772,0.989189,"br");
   pt->SetFillColor(19);
   pt->SetTextColor(4);
   TText *text = pt->AddText("TEntryList for a TTree");
   pt->Draw();
   
   pt = new TPaveText(0.0387597,0.483696,0.307494,0.657609,"br");
   pt->SetFillColor(19);
   text = pt->AddText("TEntryList");
   pt->Draw();
   
   pt = new TPaveText(0.0363636,0.107527,0.306494,0.489247,"br");
   pt->SetFillColor(19);
   pt->SetTextFont(42);
   text = pt->AddText("fBlocks");
   text = pt->AddText("fLists = 0");
   pt->Draw();
   
   pt = new TPaveText(0.338501,0.23913,0.627907,0.375,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(19);
   text = pt->AddText("Info on entries 0-63999");
   pt->Draw();
   
   pt = new TPaveText(0.643411,0.23913,0.989664,0.375,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(19);
   text = pt->AddText("entries 64000-127999");
   pt->Draw();
   
   pt = new TPaveText(0.423773,0.423913,0.870801,0.576087,"br");
   pt->SetFillColor(19);
   text = pt->AddText("TObjArray of TEntryListBlock objects");
   pt->Draw();
   TArrow *arrow = new TArrow(0.277202,0.356757,0.418605,0.505435,0.05,">");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   return c;
}
