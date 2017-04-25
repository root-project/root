// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2016 http://oproject.org
#ifndef ROOT_Mpi_TEnvironment
#define ROOT_Mpi_TEnvironment

#include<Mpi/Globals.h>
namespace ROOT {

   namespace Mpi {
      /**
      \class TEnvironment
         Class manipulate mpi environment, with this class you can to start/stop the communication system and
         to hanlde some information about the communication environment.
         \ingroup Mpi
       */

      //TODO: added error handing callback to flush stderr/stdout in case premature exit by signals o Abort call
      class TErrorHandler;
      class TEnvironment: public TObject {
         friend class TCommunicator;
      private:
         TString fStdOut;
         TString fStdErr;
         Bool_t  fSyncOutput;
         Int_t   fStdOutPipe[2];
         Int_t   fStdErrPipe[2];
         Int_t   fSavedStdErr;
         Int_t   fSavedStdOut;
         static TErrorHandler fErrorHandler;
         static Int_t fCompressionAlgorithm;
         static Int_t fCompressionLevel;
      protected:
         void InitCapture();
         void EndCapture();
         void Flush();
         void ClearBuffers();
         static Int_t GetCompressionAlgorithm();
         static Int_t GetCompressionLevel();
      public:
         TEnvironment(Int_t level = ROOT::Mpi::THREAD_SINGLE);
         /**
         Constructor thar reciev command line arguments
              */
         TEnvironment(Int_t argc, Char_t **argv, Int_t level = ROOT::Mpi::THREAD_SINGLE);
         ~TEnvironment();

         /**
         Method to synchronize stdout/stderr output.
              */
         void SyncOutput(Bool_t status = kTRUE);

         /**
         Method to finalize the environment.
              */
         void Finalize();

         // static public functions TODO
         /**
         Method to check if the communication system is finalized.
              */
         static Bool_t IsFinalized();
         /**
         Method to check if the communication system is initialized.
              */
         static Bool_t IsInitialized();

         static TString GetProcessorName();

         static Int_t GetThreadLevel();
         static Bool_t IsMainThread();


         static void SetCompression(Int_t level, Int_t algorithm = 0);

         ClassDef(TEnvironment, 1)
      };
   }
}

#endif
