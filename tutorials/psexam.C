{
// Example of text produced with PostScript illustrating how to use
// the various text control characters, national accents, sub and superscripts

   gROOT->Reset();
   c1 = new TCanvas("c1","PostScript examples",100,10,600,700);

   title = new TPaveLabel(.2,.9,.8,.95,"Printed text with PostScript");
   title->SetFillColor(16);
   title->Draw();

   pt1 = new TPaveText(.1,.5,.9,.8);
   pt1->SetTextFont(61);
   pt1->SetFillColor(18);
   pt1->AddText("K\355nstler in den gr\345\373ten st\311dten");
   pt1->AddText("\253\265 l@'\372uvre on conna\333t l@'artisan\273");
   pt1->AddText("(proverbe fran\321ais)");
   pt1->AddText("\252\241Ma\337ana\41 \322ag&\306!das&\313!\272, dit l@'\323l\325ve.");
   pt1->Draw();


   pt2 = new TPaveText(.1,.1,.9,.4);
   pt2->SetTextFont(61);
   pt2->SetFillColor(18);
   pt2->SetTextSize(0.04);
   pt2->AddText("e^+!e^-! '5# Z^0! '5# ll&^-!, qq&^\261!");
   pt2->AddText("| a&^`\256#! \267 b&^`\256#! | = `\345# a^i?jk!+b^kj?i");
   pt2->AddText("i ('d#?`m!y#&^\261!`g^m#! + m `y#&^\261! = 0' r# (~r# + m^2!)`y# = 0");
   pt2->AddText("L?em! = e J^`m#?em! A?`m#! , J^`m#?em!=l&^\261!` g?m#!l , M^j?i! = `\345&?a#! A?`a! t^a#j?i! ");
   pt2->Draw();

   c1->Print("psexam.ps");
}
