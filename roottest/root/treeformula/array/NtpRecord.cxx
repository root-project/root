////////////////////////////////////////////////////////////////////////
//
//                       NtpRecord test class
//
////////////////////////////////////////////////////////////////////////

#include "NtpRecord.h"

ClassImp(NtpShower)
ClassImp(NtpEvent)
ClassImp(NtpRecord)

TClonesArray *NtpRecord::fgShowers = 0;
TClonesArray *NtpRecord::fgEvents = 0;

//____________________________________________________________________________
NtpRecord::NtpRecord() {
   // Create a NtpRecord object.
   // When the constructor is invoked for the first time, the class static
   // variable fgShowers/Events is 0 and the TClonesArrays are created.

   if (!fgShowers) fgShowers = new TClonesArray("NtpShower", 1000);
   if (!fgEvents)  fgEvents = new TClonesArray("NtpEvent", 1000);

   fShowers = fgShowers;
   fEvents = fgEvents;

}

//_____________________________________________________________________________
NtpRecord::~NtpRecord() {
   Clear();
}

//____________________________________________________________________________
void NtpRecord::Clear(Option_t * /*option*/) {
   fShowers->Clear("C"); //will also call NtpShower::Clear
   fEvents->Clear("C"); //will also call NtpEvent::Clear
}

