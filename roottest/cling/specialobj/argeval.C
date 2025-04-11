 {
   new TCanvas();
   int Current_Pad=0;
   gPad->cd(Current_Pad++);
   if (Current_Pad!=1) {
     cerr << "Error: Arguments of gPad->cd evaluate twice!" << endl;
     gApplication->Terminate(1);
   }
}
