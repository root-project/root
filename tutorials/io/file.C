/// \file
/// \ingroup tutorial_io
/// \notebook
/// This macro displays the physical ROOT file structure
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void file(){

   TCanvas *c1 = new TCanvas("c1","ROOT File description",200,10,700,550);

   c1->Range(0,-0.25,21,14);
   TPaveLabel *title = new TPaveLabel(5,12,15,13.7,c1->GetTitle());
   title->SetFillColor(16);
   title->Draw();

   // horizonthal file layout
   TPave *file = new TPave(1,8.5,20,11);
   file->SetFillColor(11);
   file->Draw();
   TPave *fileh = new TPave(1,8.5,2.5,11);
   fileh->SetFillColor(44);
   fileh->Draw();
   TPave *lrh = new TPave(2.5,8.5,3.3,11,1);
   lrh->SetFillColor(33);
   lrh->Draw();
   lrh->DrawPave(6.9,8.5,7.7,11,1);
   lrh->DrawPave(10.5,8.5,11.3,11,1);
   lrh->DrawPave(14.5,8.5,15.3,11,1);
   TLine *ldot = new TLine(1,8.5,0.5,6.5);
   ldot->SetLineStyle(2);
   ldot->Draw();
   ldot->DrawLine(2.5, 8.5, 9.4, 6.5);
   ldot->DrawLine(10.5, 8.5, 10, 6.5);
   ldot->DrawLine(11.3, 8.5, 19.5, 6.5);
   TLine *line = new TLine(2.6,11,2.6,11.5);
   line->Draw();
   line->DrawLine(2.6,11.5,7,11.5);
   TArrow *arrow = new TArrow(7,11.5,7,11.1,0.01,"|>");
   arrow->SetFillStyle(1001);
   arrow->Draw();
   line->DrawLine( 7, 8.5, 7, 8.0);
   line->DrawLine( 7, 8.0, 10.6, 8);
   arrow->DrawArrow( 10.6,8, 10.6, 8.4,0.01,"|>");
   line->DrawLine( 10.6, 11, 10.6, 11.5);
   line->DrawLine( 10.6, 11.5, 14.6, 11.5);
   arrow->DrawArrow( 14.6,11.5, 14.6,11.1,0.01,"|>");
   line->DrawLine( 14.6, 8.5, 14.6, 8.0);
   line->DrawLine( 14.6, 8.0, 16, 8);
   ldot->DrawLine(16, 8, 19, 8);
   TText *vert = new TText(1.5,9.75,"File");
   vert->SetTextAlign(21);
   vert->SetTextAngle(90);
   vert->SetTextSize(0.025);
   vert->Draw();
   vert->DrawText(2.0, 9.75,"Header");
   vert->DrawText(2.9, 9.75,"Logical Record");
   vert->DrawText(3.2, 9.75,"Header");
   vert->DrawText(7.3, 9.75,"Logical Record");
   vert->DrawText(7.6, 9.75,"Header");
   vert->DrawText(10.9,9.75,"Logical Record");
   vert->DrawText(11.2,9.75,"Header");
   vert->DrawText(14.9,9.75,"Logical Record");
   vert->DrawText(15.2,9.75,"Header");
   TText *hori = new TText(4.75,10,"Object");
   hori->SetTextAlign(22);
   hori->SetTextSize(0.035);
   hori->Draw();
   hori->DrawText(4.75, 9.5,"Data");
   hori->DrawText(9.2, 10,"Deleted");
   hori->DrawText(9.2, 9.5,"Object");
   line->DrawLine( 6.9, 8.5, 10.5, 11);
   line->DrawLine( 6.9, 11, 10.5, 8.5);
   TText *tbig = new TText(17,9.75,"............");
   tbig->SetTextAlign(22);
   tbig->SetTextSize(0.03);
   tbig->Draw();
   tbig->DrawText(2.6, 7, "fBEGIN");
   tbig->DrawText(20., 7, "fEND");
   arrow->DrawArrow( 2.6,7, 2.6,8.4,0.01,"|>");
   arrow->DrawArrow( 20,7, 20,8.4,0.01,"|>");

   //file header
   TPaveText *header = new TPaveText(0.5,.2,9.4,6.5);
   header->SetFillColor(44);
   header->Draw();
   TText *fh=header->AddText("File Header");
   fh->SetTextAlign(22);
   fh->SetTextSize(0.04);
   header->SetTextSize(0.027);
   header->SetTextAlign(12);
   header->AddText(" ");
   header->AddLine(0,0,0,0);
   header->AddText("\"root\": Root File Identifier");
   header->AddText("fVersion: File version identifier");
   header->AddText("fBEGIN: Pointer to first data record");
   header->AddText("fEND: Pointer to first free word at EOF");
   header->AddText("fSeekFree: Pointer to FREE data record");
   header->AddText("fNbytesFree: Number of bytes in FREE");
   header->AddText("fNfree: Number of free data records");
   header->AddText("fNbytesName: Number of bytes in name/title");
   header->AddText("fUnits: Number of bytes for pointers");
   header->AddText("fCompress: Compression level");

   //logical record header
   TPaveText *lrecord = new TPaveText(10,0.2,19.5,6.5);
   lrecord->SetFillColor(33);
   lrecord->Draw();
   TText *tlrh=lrecord->AddText("Logical Record Header (TKEY)");
   tlrh->SetTextAlign(22);
   tlrh->SetTextSize(0.04);
   lrecord->SetTextSize(0.027);
   lrecord->SetTextAlign(12);
   lrecord->AddText(" ");
   lrecord->AddLine(0,0,0,0);
   lrecord->AddText("fNbytes: Length of compressed object");
   lrecord->AddText("fVersion: Key version identifier");
   lrecord->AddText("fObjLen: Length of uncompressed object");
   lrecord->AddText("fDatime: Date/Time when written to store");
   lrecord->AddText("fKeylen: Number of bytes for the key");
   lrecord->AddText("fCycle : Cycle number");
   lrecord->AddText("fSeekKey: Pointer to object on file");
   lrecord->AddText("fSeekPdir: Pointer to directory on file");
   lrecord->AddText("fClassName: class name of the object");
   lrecord->AddText("fName: name of the object");
   lrecord->AddText("fTitle: title of the object");

   c1->Update();
   c1->Print("file.png");
}
