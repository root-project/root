/// \file
/// \ingroup tutorial_graphics
/// \notebook -js
/// Illustrate some basic primitives.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void event(){
   TCanvas *c1 = new TCanvas("c1","ROOT Event description",700,500);
   c1->Range(0,0,14,15.5);
   TPaveText *event = new TPaveText(1,13,3,15);
   event->SetFillColor(11);
   event->Draw();
   event->AddText("Event");
   TLine *line = new TLine(1.1,13,1.1,1.5);
   line->SetLineWidth(2);
   line->Draw();
   line->DrawLine(1.3,13,1.3,3.5);
   line->DrawLine(1.5,13,1.5,5.5);
   line->DrawLine(1.7,13,1.7,7.5);
   line->DrawLine(1.9,13,1.9,9.5);
   line->DrawLine(2.1,13,2.1,11.5);
   TArrow *arrow = new TArrow(1.1,1.5,3.9,1.5,0.02,"|>");
   arrow->SetFillStyle(1001);
   arrow->SetFillColor(1);
   arrow->Draw();
   arrow->DrawArrow(1.3,3.5,3.9,3.5,0.02,"|>");
   arrow->DrawArrow(1.5,5.5,3.9,5.5,0.02,"|>");
   arrow->DrawArrow(1.7,7.5,3.9,7.5,0.02,"|>");
   arrow->DrawArrow(1.9,9.5,3.9,9.5,0.02,"|>");
   arrow->DrawArrow(2.1,11.5,3.9,11.5,0.02,"|>");
   TPaveText *p1 = new TPaveText(4,1,11,2);
   p1->SetTextAlign(12);
   p1->SetFillColor(42);
   p1->AddText("1 Mbyte");
   p1->Draw();
   TPaveText *p2 = new TPaveText(4,3,10,4);
   p2->SetTextAlign(12);
   p2->SetFillColor(42);
   p2->AddText("100 Kbytes");
   p2->Draw();
   TPaveText *p3 = new TPaveText(4,5,9,6);
   p3->SetTextAlign(12);
   p3->SetFillColor(42);
   p3->AddText("10 Kbytes");
   p3->Draw();
   TPaveText *p4 = new TPaveText(4,7,8,8);
   p4->SetTextAlign(12);
   p4->SetFillColor(42);
   p4->AddText("1 Kbytes");
   p4->Draw();
   TPaveText *p5 = new TPaveText(4,9,7,10);
   p5->SetTextAlign(12);
   p5->SetFillColor(42);
   p5->AddText("100 bytes");
   p5->Draw();
   TPaveText *p6 = new TPaveText(4,11,6,12);
   p6->SetTextAlign(12);
   p6->SetFillColor(42);
   p6->AddText("10 bytes");
   p6->Draw();
   TText text;
   text.SetTextAlign(12);
   text.SetTextSize(0.04);
   text.SetTextFont(72);
   text.DrawText(6.2,11.5,"Header:Event_flag");
   text.DrawText(7.2,9.5,"Trigger_Info");
   text.DrawText(8.2,7.5,"Muon_Detector: TOF");
   text.DrawText(9.2,5.5,"Calorimeters");
   text.DrawText(10.2,3.5,"Forward_Detectors");
   text.DrawText(11.2,1.5,"TPCs");
}
