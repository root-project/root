int runCanvases()
{
   TMessage m;
   TCanvas *c1 = new TCanvas("1", "1");
   TLine* l = new TLine(0,0,1,1);
   l->Draw();


   TCanvas *c2 = new TCanvas("2", "2");
   TLine* l2 = new TLine(1,0,0,1);
   l2->Draw();

   c1->Streamer(m);
   c2->Streamer(m);
   m.SetReadMode();
   m.Reset();
   delete c1;
   delete c2;
   c1 = new TCanvas(kFALSE);
   c2 = new TCanvas(kFALSE);
   c1->Streamer(m);
   c2->Streamer(m);
   int result1 = c1->GetListOfPrimitives()->GetSize();
   int result2 = c2->GetListOfPrimitives()->GetSize();
   if (result1 != 1 || result2 != 1) {
      printf("twocanvas...................................... failed\n");
   } else {
      printf("twocanvas...................................... OK\n");
   }
   // Need to return 0 in case of success
   return !(10*result1+result2);
}
