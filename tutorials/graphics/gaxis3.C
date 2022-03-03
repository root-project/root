/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Example illustrating how to modify individual labels of a TGaxis. The method
/// `ChangeLabel` allows to do that.
///
/// The first parameter of this method is the label number to be modified. If
/// this number is negative labels are numbered from the last one. The other
/// parameters are (in order):
///  - the new angle value,
///  - the new size (0 erase the label),
///  - the new text alignment,
///  - the new label color,
///  - the new label text.
///
/// \macro_image
/// \macro_code
///
/// \author  Olivier Couet

void gaxis3() {
   TCanvas* c1 = new TCanvas("c1","Examples of TGaxis",10,10,800,400);
   c1->Range(-6,-0.1,6,0.1);

   TGaxis *axis = new TGaxis(-5.5,0.,5.5,0.,0.0,100,510,"");
   axis->SetName("axis");
   axis->SetTitle("Axis Title");
   axis->SetTitleSize(0.05);
   axis->SetTitleColor(kBlue);
   axis->SetTitleFont(42);

   // Change the  1st label color to red.
   axis->ChangeLabel(1,-1,-1,-1,2);

   // Erase the 3rd label
   axis->ChangeLabel(3,-1,0.);

   // 5th label is drawn with an angle of 30 degrees
   axis->ChangeLabel(5,30.,-1,0);

   // Change the text of the 6th label.
   axis->ChangeLabel(6,-1,-1,-1,3,-1,"6th label");

   // Change the text of the 2nd label to the end.
   axis->ChangeLabel(-2,-1,-1,-1,3,-1,"2nd to last label");

   axis->Draw();
}
