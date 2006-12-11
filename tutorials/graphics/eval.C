{
   //
   // This macro produces the flowchart of TFormula::Eval
   //Author: Rene Brun
   
   gROOT->Reset();
   c1 = new TCanvas("c1");
   c1->Range(0,0,20,10);
   TPaveLabel pt1(0.2,4,3,6,"Eval");
   pt1.SetTextSize(0.5);
   pt1.SetFillColor(42);
   pt1.Draw();
   TPaveText pt2(4.5,4,7.8,6);
   pt2.Draw();
   TText *t1=pt2.AddText("Read Operator");
   TText *t2=pt2.AddText("number i");
   TPaveText pt3(9,3.5,17.5,6.5);
   TText *t4=pt3.AddText("Apply Operator to current stack values");
   TText *t5=pt3.AddText("Example: if operator +");
   TText *t6=pt3.AddText("value[i] += value[i-1]");
   t4.SetTextAlign(22);
   t5.SetTextAlign(22);
   t6.SetTextAlign(22);
   t5.SetTextColor(4);
   t6.SetTextColor(2);
   pt3.Draw();
   TPaveLabel pt4(4,0.5,12,2.5,"return result = value[i]");
   pt4.Draw();
   TArrow ar1(6,4,6,2.7,0.02,"|>");
   ar1.Draw();
   TText t7(6.56,2.7,"if i = number of stack elements");
   t7.SetTextSize(0.04);
   t7.Draw();
   ar1.DrawArrow(6,8,6,6.2,0.02,"|>");
   TLine l1(12,6.6,12,8);
   l1.Draw();
   l1.DrawLine(12,8,6,8);
   ar1.DrawArrow(3,5,4.4,5,0.02,"|>");
   ar1.DrawArrow(7.8,5,8.9,5,0.02,"|>");
}

