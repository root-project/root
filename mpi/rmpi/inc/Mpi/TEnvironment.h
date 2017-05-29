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
      class TMpiSignalHandler;
      class TEnvironment: public TObject {
         friend class TCommunicator;
         friend class TMpiSignalHandler;
         friend class TErrorHandler;
      private:
         static TString fStdOut;
         static TString fStdErr;
         static Bool_t  fSyncOutput;
         static Int_t   fStdOutPipe[2];
         static Int_t   fStdErrPipe[2];
         static Int_t   fSavedStdErr;
         static Int_t   fSavedStdOut;

         static TErrorHandler fErrorHandler;
         static Int_t fCompressionAlgorithm;
         static Int_t fCompressionLevel;

         static FILE *fOutput;
         
         TMpiSignalHandler *fInterruptSignal;
         TMpiSignalHandler *fTerminationSignal;
         TMpiSignalHandler *fSigSegmentationViolationSignal;
      protected:
         void InitSignalHandlers();
         static void InitCapture();
         static void EndCapture();
         static void Flush();
         static void ClearBuffers();
         static TString GetStdOut();
         static TString GetStdErr();
         static Bool_t IsSyncOutput();
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
         Method to finalize the environment.
              */
         void Finalize();

         // static public functions TODO
         /**
         Method to synchronize stdout/stderr output.
         \param status enable/disable output synchronization
         \param output FILE pointer to merge stdout and stderr
         by default is merged in stdout but if NULL stdout will be printed asynchronous respect to stderr
              */
         static void SyncOutput(Bool_t status = kTRUE, FILE *output = stdout);

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
