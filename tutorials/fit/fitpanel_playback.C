/// \file
/// \ingroup tutorial_fit
/// This file will test all the transient frames (aka Dialog windows)
/// displayed in the fitpanel, as the rest of the functionality is
/// tried automatically with the UnitTest.C unit.
///
/// This implies trying the Set Parameters dialog and the Advanced one.
///
/// At every operation, a png file will be saved. These files will be
/// later on compared with some references values, to have an estimation
/// of the goodness of the test.
///
/// \macro_code
///
/// \author David Gonzalez Maline

#include "TSystem.h"
#include "TSystemFile.h"
#include "TRecorder.h"
#include "Riostream.h"

int file_size(const char *filename)
{
   FileStat_t fs;
   gSystem->GetPathInfo(filename, fs);
   return (int)fs.fSize;
}

void fitpanel_playback()
{
   auto * r = new TRecorder();
   r->Replay("http://root.cern.ch/files/fitpanel_playback.root");

   // wait for the recorder to finish the replay
   while (r->GetState() == TRecorder::kReplaying) {
      gSystem->ProcessEvents();
      gSystem->Sleep(1);
   }

   int Step_Err    = 100;
   int Step1_Ref   = 15691;
   int Step2_Ref   = 15691;
   int Step3_Ref   = 17632;
   int Step4_Ref   = 12305;
   int Step5_Ref   = 11668;

   int Step1_Size   =  file_size("Step1.png");
   int Step2_Size   =  file_size("Step2.png");
   int Step3_Size   =  file_size("Step3.png");
   int Step4_Size   =  file_size("Step4.png");
   int Step5_Size   =  file_size("Step5.png");


   std::cout << "**********************************************************************" << std::endl;
   std::cout << "*  Report of fitpanel_playback.C                                     *" << std::endl;
   std::cout << "**********************************************************************" << std::endl;

   if (TMath::Abs(Step1_Ref-Step1_Size) <= Step_Err) {
      std::cout << "Step1: ............................................................ OK" << std::endl;
   } else {
      std::cout << "Step1: ........................................................ FAILED" << std::endl;
   }

   if (TMath::Abs(Step2_Ref-Step2_Size) <= Step_Err) {
      std::cout << "Step2: ............................................................ OK" << std::endl;
   } else {
      std::cout << "Step2: ........................................................ FAILED" << std::endl;
   }

   if (TMath::Abs(Step3_Ref-Step3_Size) <= Step_Err) {
      std::cout << "Step3: ............................................................ OK" << std::endl;
   } else {
      std::cout << "Step3: ........................................................ FAILED" << std::endl;
   }

   if (TMath::Abs(Step4_Ref-Step4_Size) <= Step_Err) {
      std::cout << "Step4: ............................................................ OK" << std::endl;
   } else {
      std::cout << "Step4: ........................................................ FAILED" << std::endl;
   }

   if (TMath::Abs(Step5_Ref-Step5_Size) <= Step_Err) {
      std::cout << "Step5: ............................................................ OK" << std::endl;
   } else {
      std::cout << "Step5: ........................................................ FAILED" << std::endl;
   }
   std::cout << "**********************************************************************" << std::endl;

}
