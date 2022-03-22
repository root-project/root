#include "TCanvas.h"
#include "TPaveText.h"
#include "TArrow.h"

void entrylist_figure2()
{
   TCanvas *c = new TCanvas("c", "c",172,104,447,249);
   c->Range(0,0,1,1);
   c->SetBorderSize(2);
   c->SetFrameFillColor(0);

   TPaveText *pt = new TPaveText(0.0026738,0.790055,0.417112,0.994475,"br");
   pt->SetFillColor(kWhite);
   pt->SetTextColor(4);
   TText *text = pt->AddText("TEntryList for a TChain");
   pt->Draw();

   pt = new TPaveText(0.00802139,0.541436,0.294118,0.701657,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   pt->SetTextFont(42);
   text = pt->AddText("TEntryList");
   pt->Draw();

   pt = new TPaveText(0.0106952,0.237569,0.294118,0.546961,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   pt->SetTextFont(42);
   text = pt->AddText("fBlocks = 0");
   text = pt->AddText("fLists");
   pt->Draw();

   pt = new TPaveText(0.483957,0.607735,0.68984,0.773481,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   pt->SetTextFont(42);
   text = pt->AddText("TChain");
   pt->Draw();

   pt = new TPaveText(0.347594,0.475138,0.494652,0.596685,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   pt->SetTextFont(42);
   text = pt->AddText("TTree_1");
   pt->Draw();

   pt = new TPaveText(0.508021,0.475138,0.660428,0.59116,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   pt->SetTextFont(42);
   text = pt->AddText("TTree_2");
   pt->Draw();

   pt = new TPaveText(0.673797,0.469613,0.826203,0.59116,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   pt->SetTextFont(42);
   text = pt->AddText("TTree_3");
   pt->Draw();

   pt = new TPaveText(0.251337,0.0331492,0.483957,0.165746,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   text = pt->AddText("TEntryList for TTree_1");
   pt->Draw();

   pt = new TPaveText(0.491979,0.038674,0.729947,0.171271,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   text = pt->AddText("TEntryList for TTree_2");
   pt->Draw();

   pt = new TPaveText(0.737968,0.038674,0.97861,0.171271,"br");
   pt->SetBorderSize(1);
   pt->SetFillColor(kWhite);
   text = pt->AddText("TEntryList for TTree_3");
   pt->Draw();

   pt = new TPaveText(0.410667,0.21978,0.816,0.395604,"br");
   pt->SetFillColor(kWhite);
   text = pt->AddText("TList of TEntryList* objects");
   pt->Draw();
   TArrow *arrow = new TArrow(0.224,0.296703,0.4,0.296703,0.05,">");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
}
