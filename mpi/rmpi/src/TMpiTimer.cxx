#include <Mpi/TMpiTimer.h>
#include <iostream>
#include <Mpi/TIntraCommunicator.h>
#include <Mpi/TEnvironment.h>
#include <TColor.h>
#include <TStyle.h>
using namespace ROOT::Mpi;
//______________________________________________________________________________
TMpiTimer::TMpiTimer(MPI_Comm comm) : fComm(comm){};

//______________________________________________________________________________
/**
 * Method to start a time counter
 */
void TMpiTimer::Start()
{
   fStarTime = MPI_Wtime();
}

//______________________________________________________________________________
/**
 * Method to restart the time counter on the calling processor.
 */
void TMpiTimer::ReStart()
{
   Start();
}

//______________________________________________________________________________
/**
 * Returns an elapsed time on the calling processor.
 * \return time elapse (double)
 */
Double_t TMpiTimer::GetElapsed() const
{
   return MPI_Wtime() - fStarTime;
}

//______________________________________________________________________________
/**
 * Returns  the  resolution  of ROOT::Mpi::TMpiTimer::GetElapsed in seconds.
 * That is, it returns, as a double-precision value, the number of seconds
 * between successive clock ticks. For example, if the clock is implemented by
 * the hardware as a counter that is incremented every millisecond, the value
 * returned by MPI_Wtick should  be 10^-3.
 * \return Time in seconds of resolution
 */
Double_t TMpiTimer::GetTick()
{
   return MPI_Wtick();
}

//______________________________________________________________________________
/**
 * Utility to sleep in miliseconds
 */
void TMpiTimer::Sleep(Double_t msec)
{
   gSystem->Sleep(msec);
}

//______________________________________________________________________________
/**
 * Clock synchronization The value returned for ROOT::Mpi::WTIME_IS_GLOBAL is 1
 * if clocks ROOT::Mpi::WTIME_IS_GLOBAL at all processes in
 * ROOT::Mpi::COMM_WORLD are synchronized, 0 otherwise.
 * \return boolean
 */
Bool_t TMpiTimer::IsGlobal() const
{
   Int_t *global;
   Int_t flag = 0;
   // TODO: not suppoted by OpenMPI, needs error handling here
   MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &global, &flag);
   return flag == false ? kFALSE : *global != 0;
}

//______________________________________________________________________________
/**
 * Utility method to print time information.
 */
void TMpiTimer::Print() const
{
   if (fComm != COMM_NULL) {
      std::cout << std::flush << "Comm    = " << fComm.GetCommName() << std::endl
                << "Rank    = " << fComm.GetRank() << std::endl
                << "Size    = " << fComm.GetSize() << std::endl
                << "Global  = " << IsGlobal() << std::endl
                << "Host    = " << TEnvironment::GetProcessorName() << std::endl
                << "Elapsed = " << GetElapsed() << std::endl
                << std::flush;
   } else {
      std::cout << std::flush << "Global  = " << IsGlobal() << std::endl
                << "Host    = " << TEnvironment::GetProcessorName() << std::endl
                << "Elapsed = " << GetElapsed() << std::endl
                << std::flush;
   }
}

//______________________________________________________________________________
/**
 * Utility method to get time histogram, it only works if
 * a valid communicator (non ROOT::Mpi::COMM_NULL) was provided.
 * \return TH1F object with the histogram of times for every rank.
 */
TH1F *TMpiTimer::GetElapsedHist(Int_t root) const
{
   if (fComm != COMM_NULL) {
      TH1F *hist = nullptr;
      Double_t elapsed = GetElapsed();
      Double_t *vtimes = nullptr;
      Double_t max;
      auto size = fComm.GetSize();
      auto rank = fComm.GetRank();
      if (rank == root)
         vtimes = new Double_t[size];
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
      // TODO: added error here
      return nullptr;
   }
}
