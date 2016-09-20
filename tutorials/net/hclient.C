/// \file
/// \ingroup tutorial_net
/// Client program which creates and fills a histogram. Every 1000 fills
/// the histogram is send to the server which displays the histogram.
///
/// To run this demo do the following:
///   - Open three windows
///   - Start ROOT in all three windows
///   - Execute in the first window: .x hserv.C (or hserv2.C)
///   - Execute in the second and third windows: .x hclient.C
/// If you want to run the hserv.C on a different host, just change
/// "localhost" in the TSocket ctor below to the desired hostname.
///
/// The script argument "evol" can be used when using a modified version
/// of the script where the clients and server are on systems with
/// different versions of ROOT. When evol is set to kTRUE the socket will
/// support automatic schema evolution between the client and the server.
///
/// \macro_code
///
/// \author Fons Rademakers

void hclient(Bool_t evol=kFALSE)
{
   gBenchmark->Start("hclient");

   // Open connection to server
   TSocket *sock = new TSocket("localhost", 9090);

   // Wait till we get the start message
   char str[32];
   sock->Recv(str, 32);

   // server tells us who we are
   int idx = !strcmp(str, "go 0") ? 0 : 1;

   Float_t messlen  = 0;
   Float_t cmesslen = 0;
   if (idx == 1)
      sock->SetCompressionLevel(1);

   TH1 *hpx;
   if (idx == 0) {
      // Create the histogram
      hpx = new TH1F("hpx","This is the px distribution",100,-4,4);
      hpx->SetFillColor(48);  // set nice fill-color
   } else {
      hpx = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   }

   TMessage::EnableSchemaEvolutionForAll(evol);
   TMessage mess(kMESS_OBJECT);
   //TMessage mess(kMESS_OBJECT | kMESS_ACK);

   // Fill histogram randomly
   gRandom->SetSeed();
   Float_t px, py;
   const int kUPDATE = 1000;
   for (int i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      if (idx == 0)
         hpx->Fill(px);
      else
         hpx->Fill(px,py);
      if (i && (i%kUPDATE) == 0) {
         mess.Reset();              // re-use TMessage object
         mess.WriteObject(hpx);     // write object in message buffer
         sock->Send(mess);          // send message
         messlen  += mess.Length();
         cmesslen += mess.CompLength();
      }
   }
   sock->Send("Finished");          // tell server we are finished

   if (cmesslen > 0)
      printf("Average compression ratio: %g\n", messlen/cmesslen);

   gBenchmark->Show("hclient");

   // Close the socket
   sock->Close();
}
