{
gROOT->Reset();
c1 = new TCanvas("c1","ROOT Environment Canvas",720,840);
c1->Range(0,-0.25,19,29);
TPaveLabel title(3,27.1,15,28.7,"ROOT Environment and Tools");
title.SetFillColor(42);
title.SetTextColor(5);
title.SetTextFont(62);
title.Draw();

//
TArrow ardash(2,15,2,3.5,0.015,"|>");
ardash.SetLineStyle(2);
ardash.SetFillColor(1);
ardash.Draw();
TLine l1(2.5,4.5,15.5,4.5);
l1.Draw();
l1.DrawLine(4.5,15,4.5,11);
l1.DrawLine(13,10,13,15.5);
l1.DrawLine(14,10,13,10);
l1.DrawLine(14,15.5,13,15.5);
TArrow ar(9,23,9,21.6,0.015,"|>");
ar.SetFillColor(1);
//
TPavesText UserChtml(0.5,0.5,4.5,3,5,"tr");
UserChtml.AddText("Files with hyperlinks");
TText *t1=UserChtml.AddText("*User.C.html");
TText *t2=UserChtml.AddText("*User.mac.html");
t1->SetTextColor(4);
t2->SetTextColor(4);
UserChtml.Draw();
ar.DrawArrow(2.5,4.5,2.5,3.5,0.015,"|>");
//
TPavesText UserTree(7,0.5,11,3,5,"tr");
UserTree.AddText("Dictionary");
UserTree.AddText("Inheritance graphs");
TText *t3=UserTree.AddText("*User_Tree.ps");
t3->SetTextColor(4);
UserTree.Draw();
ar.DrawArrow(9,5.5,9,3.5,0.015,"|>");
//
TPavesText Userhtml(13.5,0.5,17.5,3,5,"tr");
Userhtml.AddText("Class Description");
Userhtml.AddText("with references");
TText *t4=Userhtml.AddText("*User.html");
t4->SetTextColor(4);
Userhtml.Draw();
ar.DrawArrow(15.5,4.5,15.5,3.5,0.015,"|>");
//
TPavesText Macros(0.5,8,3.5,11,5,"tr");
Macros.AddText("Macros");
Macros.AddText("Log files");
TText *t5=Macros.AddText("*User.mac");
TText *t5a=Macros.AddText("*User.log");
t5->SetTextColor(4);
t5a->SetTextColor(4);
Macros.Draw();
//
TPavesText UserC(1,15,5,18,5,"tr");
UserC.AddText("C++ application");
UserC.AddText("source code");
TText *t6=UserC.AddText("*User.C");
t6->SetTextColor(4);
UserC.Draw();
ar.DrawArrow(4.5,11,5.8,11,0.015,"|>");
//
TPavesText Userh(6,23,12,26,5,"tr");
Userh.AddText("C++ header files");
TText *t7=Userh.AddText("*User.h");
t7->SetTextColor(4);
Userh.SetFillColor(11);
Userh.Draw();
ar.DrawArrow(9,23,9,21.6,0.015,"|>");
//
TPavesText UserUI(6.5,14,11.5,17,5,"tr");
UserUI.AddText("C++ code for");
UserUI.AddText("User Interface and I/O");
TText *t8=UserUI.AddText("*UserUI.C");
t8->SetTextColor(4);
UserUI.Draw();
ar.DrawArrow(9,18.5,9,17.3,0.015,"|>");
ar.DrawArrow(9,14,9,12.6,0.015,"|>");
//
TPavesText Usersl(14,14,17.5,17,5,"tr");
Usersl.AddText("User");
Usersl.AddText("Libraries");
TText *t9=Usersl.AddText("*User.sl");
t9->SetTextColor(4);
Usersl.Draw();
ar.DrawArrow(13,11,12.1,11,0.015,"|>");
//
TPavesText Rootlib(14,8.5,17.5,11.5,5,"tr");
Rootlib.AddText("Root Library");
Rootlib.AddText("and Includes");
TText *t10=Rootlib.AddText("Root.sl");
TText *t11=Rootlib.AddText("Root/include");
t10->SetTextColor(4);
t11->SetTextColor(4);
Rootlib.Draw();
//
TEllipse dict(9,20,3,1.5);
dict.SetFillColor(43);
dict.SetFillStyle(1001);
dict.SetLineColor(1);
dict.SetLineWidth(3);
dict.Draw();
TText gen(9,20.7,"rootcint");
gen.SetTextAlign(22);
gen.SetTextSize(0.025);
gen.Draw();
gen.DrawText(9,19.5,"ROOT compiler");
ar.DrawArrow(9,18.5,9,17.3,0.015,"|>");
//
TEllipse compiler(9,11,3,1.5);
compiler.SetFillColor(43);
compiler.SetFillStyle(1001);
compiler.SetLineColor(1);
compiler.SetLineWidth(3);
compiler.Draw();
TText gen2(9,11.4,"C++ compiler");
gen2.SetTextAlign(22);
gen2.SetTextSize(0.025);
gen2.Draw();
gen2.DrawText(9,10.3,"and Linker");
ar.DrawArrow(9,9.5,9,8.2,0.015,"|>");
//
TPaveText exe(6,5.5,12,8);
exe.SetFillColor(41);
exe.AddText("ROOT-based Application");
exe.AddText("Interactive or Batch");
TText *t12=exe.AddText("User.exe");
t12->SetTextColor(2);
exe.Draw();

c1->Modified();
c1->Print("rootenv.ps");
}
