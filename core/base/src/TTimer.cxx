// @(#)root/base:$Id$
// Author: Fons Rademakers   28/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TTimer
\ingroup Base

Handles synchronous and a-synchronous timer events.
1. synchronous timer is registered into TSystem and is processed
   within the standard ROOT event-loop.
2. asynchronous timer is passed to the operating system which sends
   an external signal to ROOT and thus interrupts its event-loop.

You can use this class in one of the following ways:
   - Sub-class TTimer and override the Notify() method.
   - Re-implement the TObject::HandleTimer() method in your class
     and pass a pointer to this object to timer, see the SetObject()
     method.
   - Pass an interpreter command to timer, see SetCommand() method.
   - Create a TTimer, connect its Timeout() signal to the
     appropriate methods. Then when the time is up it will emit a
     Timeout() signal and call connected slots.

Minimum timeout interval is defined in TSystem::ESysConstants as
`kItimerResolution` (currently 10 ms).

Signal/slots example:
~~~{.cpp}
      TTimer *timer = new TTimer();
      timer->Connect("Timeout()", "myObjectClassName",
                     myObject, "TimerDone()");
      timer->Start(2000, kTRUE);   // 2 seconds single-shot
~~~
To emit the Timeout signal repeatedly with minimum timeout:
~~~ {.cpp}
      timer->Start(0, kFALSE);
~~~
*/

#include "TTimer.h"
#include "TSystem.h"
#include "TROOT.h"

ClassImp(TTimer);


class TSingleShotCleaner : public TTimer {
private:
   TList  *fGarbage;
public:
   TSingleShotCleaner() : TTimer(10, kTRUE) { fGarbage = new TList(); }
   virtual ~TSingleShotCleaner() { fGarbage->Delete(); delete fGarbage; }
   void TurnOn() {
      TObject *obj = (TObject*) gTQSender;
      fGarbage->Add(obj);
      Reset();
      if (gSystem)
         gSystem->AddTimer(this);
   }
   Bool_t Notify() {
      fGarbage->Delete();
      Reset();
      if (gSystem)
         gSystem->RemoveTimer(this);
      return kTRUE;
   }
};

////////////////////////////////////////////////////////////////////////////////
/// Create timer that times out in ms milliseconds. If milliSec is 0
/// then the timeout will be the minimum timeout (see TSystem::ESysConstants,
/// i.e. 10 ms). If mode == kTRUE then the timer is synchronous else
/// a-synchronous. The default is synchronous. Add a timer to the system
/// eventloop by calling TurnOn(). Set command to be executed from Notify()
/// or set the object whose HandleTimer() method will be called via Notify(),
/// derive from TTimer and override Notify() or connect slots to the
/// signals Timeout(), TurnOn() and TurnOff().

TTimer::TTimer(Long_t ms, Bool_t mode) : fTime(ms)
{
   fObject      = nullptr;
   fCommand     = "";
   fSync        = mode;
   fIntSyscalls = kFALSE;
   Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// Create timer that times out in ms milliseconds. If mode == kTRUE then
/// the timer is synchronous else a-synchronous. The default is synchronous.
/// Add a timer to the system eventloop by calling TurnOn().
/// The object's HandleTimer() will be called by Notify().

TTimer::TTimer(TObject *obj, Long_t ms, Bool_t mode) : fTime(ms)
{
   fObject      = obj;
   fCommand     = "";
   fSync        = mode;
   fIntSyscalls = kFALSE;
   Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// Create timer that times out in ms milliseconds. If mode == kTRUE then
/// the timer is synchronous else a-synchronous. The default is synchronous.
/// Add a timer to the system eventloop by calling TurnOn().
/// The interpreter will execute command from Notify().

TTimer::TTimer(const char *command, Long_t ms, Bool_t mode) : fTime(ms)
{
   fObject      = nullptr;
   fCommand     = command;
   fSync        = mode;
   fIntSyscalls = kFALSE;
   Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// Check if timer timed out.

Bool_t TTimer::CheckTimer(const TTime &now)
{
   if (fAbsTime <= now) {
      fTimeout = kTRUE;
      Notify();
      return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Notify when timer times out. The timer is always reset. To stop
/// the timer call TurnOff(). Make sure to call Reset() also in derived
/// Notify() so timers will keep working repeatedly.

Bool_t TTimer::Notify()
{
   Timeout();       // emit Timeout() signal
   if (fObject) fObject->HandleTimer(this);
   if (fCommand && fCommand.Length() > 0)
      gROOT->ProcessLine(fCommand);

   Reset();
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the timer.

void TTimer::Reset()
{
   // make sure gSystem exists
   ROOT::GetROOT();

   fTimeout = kFALSE;
   fAbsTime = fTime;
   if (gSystem) {
      fAbsTime += gSystem->Now();
      if (!fSync) gSystem->ResetTimer(this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the interpreter command to be executed at time out. Removes the
/// object to be notified (if it was set).

void TTimer::SetCommand(const char *command)
{
   fObject  = nullptr;
   fCommand = command;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the object to be notified  at time out. Removes the command to
/// be executed (if it was set).

void TTimer::SetObject(TObject *object)
{
   fObject  = object;
   fCommand = "";
}

////////////////////////////////////////////////////////////////////////////////
/// When the argument is true the a-synchronous timer (SIGALRM) signal
/// handler is set so that interrupted syscalls will not be restarted
/// by the kernel. This is typically used in case one wants to put a
/// timeout on an I/O operation. By default interrupted syscalls will
/// be restarted.

void TTimer::SetInterruptSyscalls(Bool_t set)
{
   fIntSyscalls = set;
}

////////////////////////////////////////////////////////////////////////////////
/// Starts the timer with a milliSec timeout. If milliSec is 0
/// then the timeout will be the minimum timeout (see TSystem::ESysConstants,
/// i.e. 10 ms), if milliSec is -1 then the time interval as previously
/// specified (in ctor or SetTime()) will be used.
/// If singleShot is kTRUE, the timer will be activated only once,
/// otherwise it will continue until it is stopped.
/// See also TurnOn(), Stop(), TurnOff().

void TTimer::Start(Long_t milliSec, Bool_t singleShot)
{
   if (milliSec >= 0)
      SetTime(milliSec);
   Reset();
   TurnOn();
   if (singleShot)
      Connect(this, "Timeout()", "TTimer", this, "TurnOff()");
   else
      Disconnect(this, "Timeout()", this, "TurnOff()");
}

////////////////////////////////////////////////////////////////////////////////
/// Remove timer from system timer list. This requires that a timer
/// has been placed in the system timer list (using TurnOn()).
/// If a TTimer subclass is placed on another list, override TurnOff() to
/// remove the timer from the correct list.

void TTimer::TurnOff()
{
   if (gSystem)
      if (gSystem->RemoveTimer(this))
         Emit("TurnOff()");
}

////////////////////////////////////////////////////////////////////////////////
/// Add the timer to the system timer list. If a TTimer subclass has to be
/// placed on another list, override TurnOn() to add the timer to the correct
/// list.

void TTimer::TurnOn()
{
   // might have been set in a previous Start()
   Disconnect(this, "Timeout()", this, "TurnOff()");

   if (gSystem) {
      gSystem->AddTimer(this);
      Emit("TurnOn()");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// This static function calls a slot after a given time interval.
/// Created internal timer will be deleted after that.

void TTimer::SingleShot(Int_t milliSec, const char *receiver_class,
                        void *receiver, const char *method)
{
   TTimer *singleShotTimer = new TTimer(milliSec);
   TQObject::Connect(singleShotTimer, "Timeout()",
                     receiver_class, receiver, method);

   static TSingleShotCleaner singleShotCleaner;  // single shot timer cleaner

   // gSingleShotCleaner will delete singleShotTimer a
   // short period after Timeout() signal is emitted
   TQObject::Connect(singleShotTimer, "Timeout()",
                     "TTimer", &singleShotCleaner, "TurnOn()");

   singleShotTimer->Start(milliSec, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// This function checks if the timer is running within gSystem 
/// (Has been started and did not finish yet).

bool TTimer::IsRunning()
{
   if (gSystem && gSystem->GetListOfTimers())
      return gSystem->GetListOfTimers()->IndexOf(this) != -1;
   return false;
}
