{
   gROOT->Reset();
   TCanvas *nut = new TCanvas("nut", "Shared libraries",100,10,700,900);
   nut->Range(0,0,20,24);
   nut->SetFillColor(10);
   nut->SetBorderSize(2);

   TPaveLabel *pl = new TPaveLabel(3,22.2,17,23.7,"Dynamic linking from Shared libraries","br");
   pl->SetFillColor(18);
   pl->SetTextSize(0.4);
   pl->Draw();
   TText t(0,0,"a");
   t.SetTextFont(62);
   t.SetTextSize(0.025);
   t.SetTextAlign(12);
   t.DrawText(2,20,"The \"standard\" ROOT executable module can dynamically");
   t.DrawText(2,19,"load user@'s specific code from shared libraries.");
   t.SetTextFont(72);
   t.SetTextSize(0.026);
   t.DrawText(3,16,"Root >  gSystem->Load(\"na49.sl\")");
   t.DrawText(3,15,"Root >  gSystem->Load(\"mylib.sl\")");
   t.DrawText(3,14,"Root >  T49Event event");
   t.DrawText(3,13,"Root >  event.xxxxxxx");

   TEllipse el(5,8.5,2,1.3);
   el.SetFillColor(17);
   el.Draw();
   t.SetTextFont(62);
   t.SetTextAlign(22);
   t.SetTextSize(0.025);
   t.DrawText(5,9.0,"ROOT");
   t.DrawText(5,8.0,"executable");

   TPaveLabel roots(10,10,16,11,"ROOT Shared libraries");
   roots.SetFillColor(17);
   roots.Draw();
   TPaveLabel na49s(10,8,16,9,"NA49 Shared libraries");
   na49s.SetFillColor(17);
   na49s.Draw();
   TPaveLabel users(10,6,16,7,"User Shared libraries");
   users.SetFillColor(17);
   users.Draw();
   TArrow *arrow = new TArrow(9.91329,10.4982,7.05202,9.17895,0.025,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   arrow = new TArrow(9.88439,8.47719,7.39884,8.47719,0.025,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   arrow = new TArrow(9.88439,6.45614,7.19653,7.77544,0.025,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();

  //--signature
   TText sig(.2,.2,"/user/brun/root/aihep/shared.C");
   sig.SetTextFont(72);
   sig.SetTextSize(0.020);
   sig.Draw();

   nut->Modified();
   nut->Print("shared.ps");
   nut->cd();
}
