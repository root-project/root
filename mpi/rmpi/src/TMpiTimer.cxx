#include<Mpi/TMpiTimer.h>
#include<iostream>
#include<Mpi/TIntraCommunicator.h>
#include<Mpi/TEnvironment.h>
#include<TColor.h>
#include<TStyle.h>
using namespace ROOT::Mpi;
//______________________________________________________________________________
TMpiTimer::TMpiTimer(MPI_Comm  comm): fComm(comm) {};

//______________________________________________________________________________
void TMpiTimer::Start()
{
   fStarTime = MPI_Wtime();
}

//______________________________________________________________________________
void TMpiTimer::ReStart()
{
   Start();
}

//______________________________________________________________________________
Double_t TMpiTimer::GetElapsed() const
{
   return MPI_Wtime() - fStarTime;
}

//______________________________________________________________________________
Double_t TMpiTimer::GetTick()
{
   return  MPI_Wtick();
}

//______________________________________________________________________________
void TMpiTimer::Sleep(Double_t msec)
{
   gSystem->Sleep(msec);
}

//______________________________________________________________________________
Bool_t TMpiTimer::IsGlobal() const
{
   Int_t *global;
   Int_t flag = 0;
   MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &global, &flag);
   return flag == false ? kFALSE : *global != 0;
}

//______________________________________________________________________________
void TMpiTimer::Print() const
{
   if (fComm != COMM_NULL) {
      std::cout << std::flush \
                << "Comm    = " << fComm.GetCommName() << std::endl\
                << "Rank    = " << fComm.GetRank() << std::endl\
                << "Size    = " << fComm.GetSize() << std::endl\
                << "Global  = " << IsGlobal() << std::endl\
                << "Host    = " << TEnvironment::GetProcessorName() << std::endl\
                << "Elapsed = " << GetElapsed() << std::endl \
                << std::flush;
   } else {
      std::cout << std::flush \
                << "Global  = " << IsGlobal() << std::endl\
                << "Host    = " << TEnvironment::GetProcessorName() << std::endl\
                << "Elapsed = " << GetElapsed() << std::endl \
                << std::flush;
   }
}

//______________________________________________________________________________
TH1F *TMpiTimer::GetElapsedHist(Int_t root) const
{
   if (fComm != COMM_NULL) {
      TH1F *hist = nullptr;
      Double_t elapsed = GetElapsed();
      Double_t *vtimes = nullptr;
      Double_t max;
      auto size = fComm.GetSize();
      auto rank = fComm.GetRank();
      if (rank == root) vtimes = new Double_t[size];
      fComm.Gather(&elapsed, 1, vtimes, size, root);
      fComm.Reduce(elapsed, max, MAX, root);
      if (rank == root) {
         hist = new TH1F("times", "", size, 0, size);
         gStyle->SetOptStat(000000);
         gStyle->SetTitleXOffset(0.4);
         gStyle->SetTitleXOffset(1.2);
         for (auto i = 1; i < size + 1; ++i) {
            hist->GetXaxis()->SetBinLabel(i, Form("%d", i - 1));
            hist->SetBinContent(i, vtimes[i - 1]);
         }
         hist->SetBarWidth(0.97);
         hist->LabelsOption("v <", "X");
         hist->SetBarWidth(0.97);
         hist->SetFillColor(TColor::GetColor("#006600"));

         hist->GetXaxis()->SetTitle(" Rank ");
         hist->GetXaxis()->SetTitleSize(0.045);
         hist->GetXaxis()->CenterTitle();

         hist->GetYaxis()->SetTitle(" Time (sec)");
         hist->GetYaxis()->SetTitleSize(0.045);
         hist->GetYaxis()->CenterTitle();

         hist->GetYaxis()->SetRangeUser(0, max);
         hist->SetDirectory(0);
      }
      return hist;
   } else {
      //TODO: added error here
      return nullptr;
   }
}
