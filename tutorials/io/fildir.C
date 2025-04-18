/// \file
/// \ingroup tutorial_io
/// \notebook
/// This macro displays the ROOT Directory data structure
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void fildir(){


   TCanvas *c1 = new TCanvas("c1","ROOT FilDir description",700,900);
   c1->Range(1,1,19,24.5);
   TPaveLabel *title = new TPaveLabel(4,23,16,24.2,"ROOT File/Directory/Key description");
   title->SetFillColor(16);
   title->Draw();

   Int_t keycolor = 42;
   Int_t dircolor = 21;
   Int_t objcolor = 46;
   TPaveText *file = new TPaveText(2,19,6,22);
   file->SetFillColor(39);
   file->Draw();
   file->SetTextSize(0.04);
   file->AddText("TFile");
   file->AddText("Header");
   TArrow *arrow = new TArrow(6,20.5,17,20.5,0.02,"|>");
   arrow->SetFillStyle(1001);
   arrow->SetLineWidth(2);
   arrow->Draw();
   TPaveText *free1 = new TPaveText(8,20,11,21);
   free1->SetFillColor(18);
   free1->Draw();
   free1->AddText("First:Last");
   TPaveText *free2 = new TPaveText(12,20,15,21);
   free2->SetFillColor(18);
   free2->Draw();
   free2->AddText("First:Last");
   TText *tfree = new TText(6.2,21.2,"fFree = TList of free blocks");
   tfree->SetTextSize(0.02);
   tfree->Draw();
   TText *tkeys = new TText(5.2,18.2,"fKeys = TList of Keys");
   tkeys->SetTextSize(0.02);
   tkeys->Draw();
   TText *tmemory = new TText(3.2,15.2,"fListHead = TList of Objects in memory");
   tmemory->SetTextSize(0.02);
   tmemory->Draw();

   arrow->DrawArrow(5,17,17,17,0.02,"|>");
   TLine *line = new TLine(5,19,5,17);
   line->SetLineWidth(2);
   line->Draw();
   TPaveText *key0 = new TPaveText(7,16,10,18);
   key0->SetTextSize(0.04);
   key0->SetFillColor(keycolor);
   key0->AddText("Key 0");
   key0->Draw();
   TPaveText *key1 = new TPaveText(12,16,15,18);
   key1->SetTextSize(0.04);
   key1->SetFillColor(keycolor);
   key1->AddText("Key 1");
   key1->Draw();
   line->DrawLine(3,19,3,14);
   line->DrawLine(3,14,18,14);
   TPaveText *obj0 = new TPaveText(5,13,8,15);
   obj0->SetFillColor(objcolor);
   obj0->AddText("Object");
   obj0->Draw();
   TPaveText *dir1 = new TPaveText(10,13,13,15);
   dir1->SetFillColor(dircolor);
   dir1->AddText("SubDir");
   dir1->Draw();
   TPaveText *obj1 = new TPaveText(15,13,18,15);
   obj1->SetFillColor(objcolor);
   obj1->AddText("Object");
   obj1->Draw();
   arrow->DrawArrow(12,11,17,11,0.015,"|>");
   arrow->DrawArrow(11,9,17,9,0.015,"|>");
   line->DrawLine(12,13,12,11);
   line->DrawLine(11,13,11,9);
   TPaveText *key2 = new TPaveText(14,10.5,16,11.5);
   key2->SetFillColor(keycolor);
   key2->AddText("Key 0");
   key2->Draw();
   TPaveText *obj2 = new TPaveText(14,8.5,16,9.5);
   obj2->SetFillColor(objcolor);
   obj2->AddText("Object");
   obj2->Draw();
   TLine *ldot = new TLine(10,15,2,11);
   ldot->SetLineStyle(kDashed);
   ldot->Draw();
   ldot->DrawLine(13,15,8,11);
   ldot->DrawLine(13,13,8,5);
   TPaveText *dirdata = new TPaveText(2,5,8,11);
   dirdata->SetTextAlign(12);
   dirdata->SetFillColor(dircolor);
   dirdata->Draw();
   dirdata->SetTextSize(0.015);
   dirdata->AddText("fModified: True if directory is modified");
   dirdata->AddText("fWritable: True if directory is writable");
   dirdata->AddText("fDatimeC: Creation Date/Time");
   dirdata->AddText("fDatimeM: Last mod Date/Time");
   dirdata->AddText("fNbytesKeys: Number of bytes of key");
   dirdata->AddText("fNbytesName : Header length up to title");
   dirdata->AddText("fSeekDir: Start of Directory on file");
   dirdata->AddText("fSeekParent: Start of Parent Directory");
   dirdata->AddText("fSeekKeys: Pointer to Keys record");
   TPaveText *keydata = new TPaveText(10,2,17,7);
   keydata->SetTextAlign(12);
   keydata->SetFillColor(keycolor);
   keydata->Draw();
   ldot->DrawLine(14,11.5,10,7);
   ldot->DrawLine(16,11.5,17,7);
   keydata->SetTextSize(0.015);
   keydata->AddText("fNbytes: Size of compressed Object");
   keydata->AddText("fObjLen: Size of uncompressed Object");
   keydata->AddText("fDatime: Date/Time when written to store");
   keydata->AddText("fKeylen: Number of bytes for the key");
   keydata->AddText("fCycle : Cycle number");
   keydata->AddText("fSeekKey: Pointer to Object on file");
   keydata->AddText("fSeekPdir: Pointer to directory on file");
   keydata->AddText("fClassName: 'TKey'");
   keydata->AddText("fName: Object name");
   keydata->AddText("fTitle: Object Title");
   c1->Print("fildir.png");
}
