## Core Libraries

### General

#### Platform support

ROOT now works on linuxarm64 / AArch64 / ARMv8 64-bit - thanks, David Abdurachmanov!

ROOT supports GCC 5.0 (a preview thereof) and XCode 6.3, Mac OSX 10.10.3


#### Thread-Safety

A lot of effort went into improving the thread-safety of Core and Meta classes / functions. A special thanks to Chris Jones from CMS!


### Meta library

#### Backward Incompatibilities

TIsAProxy's constructor no longer take the optional and unused 2nd argument which was reserved for a 'context'.  This context was unused in TIsAProxy itself and was not accessible from derived classes.

#### Interpreter

The new interface `TInterpreter::Declare(const char* code)` will declare the
code to the interpreter with all interpreter extensions disabled, i.e. as
"proper" C++ code. No autoloading or synamic lookup will be performed.

A new R__LOAD_LIBRARY(libWhatever) will load libWhatever at parse time. This allows ROOT to resolve symbols from this library very early on. It is a work-around for the following code from ROOT 5:

``` {.cpp}
  // ROOT 5:
  void func() {
    gSystem->Load("libEvent");
    Event* e = new Event;
  }
```

Instead, write:

``` {.cpp}
  // ROOT 6:
  R__LOAD_LIBRARY(libEvent)
  #include "Event.h"

  void func() {
    Event* e = new Event;
  }
```


#### TClass

Introduced new overload for calculating the TClass CheckSum:

``` {.cpp}
   UInt_t TClass::GetCheckSum(ECheckSum code, Bool_t &isvalid) const;
```

which indicates via the 'isvalid' boolean whether the checksum could be
calculated correctly or not.

### TROOT

Implemented new gROOT->GetTutorialsDir() static method to return the actual location of the tutorials directory.
This is $ROOTSYS/tutorials when not configuring with --prefix  or -Dgnuinstall for CMake.

### Interpreter Library

Many, many bugs have been fixed; thanks to everyone who has reported them!

#### Cling

Cling is now using a new just-in-time compilation engine called OrcJIT, a development based on MCJIT. It enables interpretation of inline assembly and exceptions; it will hopefully in the near future also support interpreting thread local storage (but doesn't at the moment).

Thanks to the new JIT, cling also comes with debug symbols for interpreted code; you can enable them with ".debug".

#### Function evaluation

Function calls through TMethodCall etc have been accelerated.

#### llvm / clang

llvm / clang were updated to r227800. This includes everything from the clang 3.6 release.
