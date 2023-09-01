#include "gtest/gtest.h"

#include "TException.h"
#include "TSysEvtHandler.h"
#include "Rtypes.h"

#include <csignal>

class TestExceptionHandler : public TExceptionHandler {
private:
   Int_t currentSignal;
public:
   void HandleException(Int_t sig) override
   {
      EXPECT_EQ(sig, currentSignal);
      Throw(sig);
   }

   void SetCurrentSignal(Int_t sig)
   {
      currentSignal = sig;
   }
};


TEST(TExceptionHandler, HandleException)
{
   // Set exception handler
   auto handler = new TestExceptionHandler();
   gExceptionHandler = handler;

   // Trigger signals
   handler->SetCurrentSignal(kSigSegmentationViolation);
   TRY {
      std::raise(SIGSEGV);
   } CATCH(excode) {
      EXPECT_EQ(excode, kSigSegmentationViolation);
   } ENDTRY;

   handler->SetCurrentSignal(kSigAbort);
   TRY {
      std::raise(SIGABRT);
   } CATCH(excode) {
      EXPECT_EQ(excode, kSigAbort);
   } ENDTRY;
}
