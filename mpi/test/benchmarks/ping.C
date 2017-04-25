// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
//Measuring message size using TMpiMessage, Raw MPI Message and TMpiMessage with compression

using namespace ROOT::Mpi;

void ping()
{
   TEnvironment env;          //environment to start communication system
   env.SyncOutput();

   if (COMM_WORLD.GetSize() != 2) {
      env.Finalize();
   }

   Float_t elements[10];
   TH1F *msgh;  //Serialized message size in Kb
   TH1F *cmsgh; //Compressed serialized message size Kb
   TH1F *rawmsgh; //Raw MPI message size Kb

   Int_t width = 20;
   const Char_t separator    = ' ';

   //Measuring normal serialized message
   if (COMM_WORLD.GetRank() == 0) {
      cout << "Sending TMpiMessages" << endl;
      auto counter = 10;
      for (auto n = 1024; n < pow(2, 20); n = pow(2, counter)) {
         TVectorT<Double_t> vec(n);
         COMM_WORLD.Send(vec, 1, n);
         counter++;
      }
   } else {
      cout << "Recieving TMpiMessages" << endl;
      cout << left << setw(width) << setfill(separator) << "Number of Elements" << right << setw(width) << setfill(separator) << "Size(Kb)" << endl;
      auto counter = 10;
      Float_t msgsizes[10];
      msgh  = new TH1F("msgh", "", 10, 0, pow(2, 19));

      for (auto n = 1024; n < pow(2, 20); n = pow(2, counter)) {
         elements[counter - 10] = n;
         TVectorT<Double_t> vec(n);

         TStatus status;
         COMM_WORLD.Recv(vec, 0, n, status);
         msgsizes[counter - 10] = status.GetMsgSize() / 1000.0;
         cout << left << setw(width) << setfill(separator) << n << right << setw(width) << setfill(separator) << std::fixed << msgsizes[counter - 10] << endl;
         msgh->GetXaxis()->SetBinLabel(counter - 9, Form("%d", n));
         msgh->SetBinContent(counter - 9, msgsizes[counter - 10]);
         msgh->SetLineColorAlpha(kGreen, 0.35);
         counter++;
      }
   }

   env.SetCompression(1);//enabling compression in the serialized messages

   //Measuring compressed and serialized message
   if (COMM_WORLD.GetRank() == 0) {
      cout << "Sending TMpiMessages Compressed" << endl;
      auto counter = 10;
      for (auto n = 1024; n < pow(2, 20); n = pow(2, counter)) {
         TVectorT<Double_t> vec(n);
         COMM_WORLD.Send(vec, 1, n);
//          std::cout << "Sending compressed message of " << n << " Double_t in std::TVectorT<Double_t>" << std::endl;
         counter++;
      }
   } else {
      cout << "Recieving Compressed TMpiMessages " << endl;
      cout << left << setw(width) << setfill(separator) << "Number of Elements" << right << setw(width) << setfill(separator) << "Size(Kb)" << endl;
      auto counter = 10;
      Float_t msgsizes[10];
      cmsgh  = new TH1F("cmsgh", "", 10, 0, pow(2, 19));

      for (auto n = 1024; n < pow(2, 20); n = pow(2, counter)) {
         elements[counter - 10] = n;
         TVectorT<Double_t> vec(n);

         TStatus status;
         COMM_WORLD.Recv(vec, 0, n, status);
         msgsizes[counter - 10] = status.GetMsgSize() / 1000.0;
         cout << left << setw(width) << setfill(separator) << n << right << setw(width) << setfill(separator) << std::fixed << msgsizes[counter - 10] << endl;
         cmsgh->GetXaxis()->SetBinLabel(counter - 9, Form("%d", n));
         cmsgh->SetBinContent(counter - 9, msgsizes[counter - 10]);
         cmsgh->SetLineColorAlpha(kRed, 0.35);
         counter++;
      }
   }



   //Measuring raw MPI  message
   if (COMM_WORLD.GetRank() == 0) {
      cout << "Sending Raw MPI Messages" << endl;
      auto counter = 10;
      for (auto n = 1024; n < pow(2, 20); n = pow(2, counter)) {
         Double_t dvec[n];

         COMM_WORLD.Send(dvec, n, 1, n + 1);
         counter++;
      }
   } else {
      cout << "Recieving Raw MPI Messages" << endl;
      cout << left << setw(width) << setfill(separator) << "Number of Elements" << right << setw(width) << setfill(separator) << "Size(Kb)" << endl;
      auto counter = 10;
      Float_t rawmsgsizes[10];
      rawmsgh  = new TH1F("rawmsgh", "", 10, 0, pow(2, 19));

      for (auto n = 1024; n < pow(2, 20); n = pow(2, counter)) {
         Double_t dvec[n];

         TStatus status;

         COMM_WORLD.Recv(dvec, n, 0, n + 1, status);
         rawmsgsizes[counter - 10] = status.GetMsgSize() / 1000.0;
         cout << left << setw(width) << setfill(separator) << n << right << setw(width) << setfill(separator) << std::fixed << rawmsgsizes[counter - 10] << endl;
         rawmsgh->GetXaxis()->SetBinLabel(counter - 9, Form("%d", n));
         rawmsgh->SetBinContent(counter - 9, rawmsgsizes[counter - 10]);
         rawmsgh->SetLineColorAlpha(kBlue, 0.35);

         counter++;
      }
   }
   //plotting restuls
   if (COMM_WORLD.GetRank() == 1) {
      auto canvas = new TCanvas("canvas");
      gStyle->SetOptStat(000000);
      gROOT->SetStyle("Plain");

      auto mg = new TMultiGraph();

      mg->SetTitle("Message sizes;Number of elements;Size of message(Kb)");

      auto gr1 = new TGraph(msgh);
      auto gr2 = new TGraph(cmsgh);
      auto gr3 = new TGraph(rawmsgh);

      mg->Add(gr1, "*AC");
      mg->Add(gr2, "*AC");
      mg->Add(gr3, "*AC");

      mg->Draw("AP");
      TLegend *leg = new TLegend(0.1, 0.7, 0.3, 0.9);
      leg->SetHeader("Message type");
      leg->AddEntry(gr1, "TMpiMessage", "lp");
      leg->AddEntry(gr3, "Raw MPI Message", "lp");
      leg->AddEntry(gr2, "TMpiMessage Compressed", "lp");
      leg->SetFillColor(0);
      leg->Draw();

      gPad->SetLogy();

      canvas->SaveAs("plot.C");
   }

}

