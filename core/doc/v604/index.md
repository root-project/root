## Core Libraries

### Interpreter

The new interface `TInterpreter::Declare(const char* code)` will declare the
code to the interpreter with all interpreter extensions disabled, i.e. as
"proper" C++ code. No autoloading or synamic lookup will be performed.

### Meta library

#### Backward Incompatibilities

TIsAProxy's constructor no longer take the optional and unused 2nd argument which was reserved for a 'context'.  This context was unused in TIsAProxy itself and was not accessible from derived classes.

### TROOT

Implemented new gROOT->GetTutorialsDir() static method to return the actual location of the tutorials directory.
This is $ROOTSYS/tutorials when not configuring with --prefix  or -Dgnuinstall for CMake.

### TClass

Introduced new overload for calculating the TClass CheckSum:

``` {.cpp}
   UInt_t TClass::GetCheckSum(ECheckSum code, Bool_t &isvalid) const;
```

which indicates via the 'isvalid' boolean whether the checksum could be
calculated correctly or not.

