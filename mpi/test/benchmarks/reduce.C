// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
//Measuring Aproximation of FLOPs using matrix operations.

//NOTE: this is an example to compute floating point operations (FLOP) that over the time in seconds are FLOPs.
// for this example I am using square matrix operations in a reduce algorithm to get an aproximations of FLOPs.
// We know the theorical number of FLOP for matrix operations:
//  n^2  for sums of matrices
//  2n^3 for multiplication matrices.
//then the aproximation that I am using to get for FLOPs  is measuring the time in every rank to get the maximum in all proccesses called t,
//in parallel I am calling a number of proccesses called p,
//To que the total FLOPs used in all proccesses the formula is  FLOPs=(p-1)*FLOP/t where FLOP is the theorical formula for matrix operation and is p-1 because
// I am using reduce operation.
// An important note is that using message compression the time can change and it can modified the value of the FLOPs measured.

using namespace ROOT::Mpi;

//flops for square matrix multiplication 2n^3
Double_t mflops(Double_t n, Double_t t)
{
   return n * n / t / 1000000.0;
}

template<class T> void FillMatrix(T *mdata, Int_t n)
{
   for (auto i = 0; i < n * n; i++) {
      mdata[i] = 0.1;
   }
}


Int_t size[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
void reduce()
{
   TEnvironment env;          //environment to start communication system
   env.SyncOutput();
//    env.SetCompression(1);

   if (COMM_WORLD.GetSize() <= 1) {
      env.Finalize();
   }

   Double_t flops[9];
   Int_t counter = 0;

   TMpiTimer timer(COMM_WORLD);
   for (auto &n : size) {
      Double_t maxtime = 0;
      Float_t mdata[n * n];
      if (COMM_WORLD.GetRank() == 0) {
         FillMatrix(mdata, n);
      }
      COMM_WORLD.Bcast(mdata, n * n, 0);

      TMatrixF send_mat(n, n, mdata); //filling matrix with floats, by defualt is filling with zero
      TMatrixF recv_mat(n, n, mdata);

      COMM_WORLD.Barrier();
      timer.Start();
      COMM_WORLD.Reduce(send_mat, recv_mat, SUM, 0); //result in rank 0
      COMM_WORLD.Barrier();
      auto elapsed = timer.GetElapsed();

      COMM_WORLD.Reduce(elapsed, maxtime, MAX, 0); //taking the worst time to compute flops


      if (COMM_WORLD.GetRank() == 0) {
         flops[counter] = (COMM_WORLD.GetSize() - 1) * mflops(n, maxtime);
         std::cout << left << fixed << "MaxTime(sec) = " << setw(10) << maxtime << setw(20) << Form(" matrix[%d][%d] ", n, n) << " MFlops(Aprox)=" << setw(16) << flops[counter] << " Total jobs = " << COMM_WORLD.GetSize() << std::endl;
         counter++;
      }


   }

   if (COMM_WORLD.GetRank() == 0) {
      //plots here
      auto canvas = new TCanvas("canvas", "ROOTMpi Performance", 700, 800);

      auto h1 = new TH1F(Form("h%d", COMM_WORLD.GetSize()), "ROOTMpi Performance", 9, 0, 200);

      counter = 1;
      for (auto &n : size) {
         h1->SetBinContent(counter, flops[counter - 1]);
         h1->GetXaxis()->SetBinLabel(counter, Form("%dx%d", n, n));
         counter++;
      }

      h1->GetXaxis()->SetTitle(" Matrix size ");
      h1->GetXaxis()->SetTitleSize(0.045);
      h1->GetXaxis()->CenterTitle();
      h1->GetXaxis()->SetTitleOffset(1.05);

      h1->GetYaxis()->SetTitle(" Performance (MFLOPs)");
      h1->GetYaxis()->SetTitleSize(0.045);
      h1->GetYaxis()->CenterTitle();
      h1->GetYaxis()->SetTitleOffset(1.05);



      h1->SetBarWidth(0.97);
//       h1->SetFillColor(TColor::GetColor("#006600"));
      h1->SetFillColor(COMM_WORLD.GetSize());
      h1->SaveAs(Form("reduce%d.root", COMM_WORLD.GetSize()), "RECREATE");
      h1->Draw();
      gStyle->SetOptStat(000000);
      canvas->SaveAs("plot.C");
   }

}

