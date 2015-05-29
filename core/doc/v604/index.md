## Core Libraries

### General

#### Platform support

ROOT now works on linuxarm64 / AArch64 / ARMv8 64-bit - thanks, David Abdurachmanov!

ROOT supports GCC 5.0 (a preview thereof) and XCode 6.3, Mac OSX 10.10.3


#### Thread-Safety

A lot of effort went into improving the thread-safety of Core and Meta classes / functions. A special thanks to Chris Jones from CMS!

#### std::string_view

Introduce a preview of C++17's std::string_view.  To take advantage of this new
class use:
```{.cpp}
#include "RStringView.h"
```
The documentation of this can be found at `http://en.cppreference.com/w/cpp/experimental/basic_string_view`
The implementation provided is extracted from libcxx.  Whenever the current
compiler and standard library provide an implmentation, it is used.

The type string_view describes an object that can refer to a constant contiguous sequence of char-like objects with the first element of the sequence at position zero.

This type is used throughout the ROOT code to avoid copying strings when a
sub-string is needed and to extent interfaces that uses to take a const char*
to take a std::string_view as thus be able to be directly directly passed a
TString, a std::string or a std::string_view.   Usage example:

``` {.cpp}
// With SetName(std::string_view)
std::string str; …
obj.SetName( std );
obj.SetName( {str.data()+pos, len} );
```

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

### TColor

Add an enum to access the palette by name.

Add new palettes with 255 colors. Names and colors' definitions have been taken from
[here](http://www.rcnp.osaka-u.ac.jp/~noji/colormap). Except for the `kBird` palette.
These palettes can be accessed with `gStyle->SetPalette(num)`. `num` can be taken
within the following enum:

* kDeepSea = 51
* kGreyScale = 52
* kDarkBodyRadiator = 53
* kBlueYellow =  54
* kRainBow = 55
* kInvertedDarkBodyRadiator = 56
* kBird = 57
* kCubehelix = 58
* kGreenRedViolet = 59
* kBlueRedYellow = 60
* kOcean = 61
* kColorPrintableOnGrey = 62
* kAlpine = 63
* kAquamarine = 64
* kArmy = 65
* kAtlantic = 66
* kAurora = 67
* kAvocado = 68
* kBeach = 69
* kBlackBody = 70
* kBlueGreenYellow = 71
* kBrownCyan = 72
* kCMYK = 73
* kCandy = 74
* kCherry = 75
* kCoffee = 76
* kDarkRainBow = 77
* kDarkTerrain = 78
* kFall = 79
* kFruitPunch = 80
* kFuchsia = 81
* kGreyYellow = 82
* kGreenBrownTerrain = 83
* kGreenPink = 84
* kIsland = 85
* kLake = 86
* kLightTemperature = 87
* kLightTerrain = 88
* kMint = 89
* kNeon = 90
* kPastel = 91
* kPearl = 92
* kPigeon = 93
* kPlum = 94
* kRedBlue = 95
* kRose = 96
* kRust = 97
* kSandyTerrain = 98
* kSienna = 99
* kSolar = 100
* kSouthWest = 101
* kStarryNight = 102
* kSunset = 103
* kTemperatureMap = 104
* kThermometer = 105
* kValentine = 106
* kVisibleSpectrum = 107
* kWaterMelon = 108
* kCool = 109
* kCopper = 110
* kGistEarth = 111

![DeepSea](palette_51.png)
![GreyScale](palette_52.png)
![DarkBodyRadiator](palette_53.png)
![BlueYellow=](palette_54.png)
![RainBow](palette_55.png)
![InvertedDarkBodyRadiator](palette_56.png)
![Bird](palette_57.png)
![Cubehelix](palette_58.png)
![GreenRedViolet](palette_59.png)
![BlueRedYellow](palette_60.png)
![Ocean](palette_61.png)
![ColorPrintableOnGrey](palette_62.png)
![Alpine](palette_63.png)
![Aquamarine](palette_64.png)
![Army](palette_65.png)
![Atlantic](palette_66.png)
![Aurora](palette_67.png)
![Avocado](palette_68.png)
![Beach](palette_69.png)
![BlackBody](palette_70.png)
![BlueGreenYellow](palette_71.png)
![BrownCyan](palette_72.png)
![CMYK](palette_73.png)
![Candy](palette_74.png)
![Cherry](palette_75.png)
![Coffee](palette_76.png)
![DarkRainBow](palette_77.png)
![DarkTerrain](palette_78.png)
![Fall](palette_79.png)
![FruitPunch](palette_80.png)
![Fuchsia](palette_81.png)
![GreyYellow](palette_82.png)
![GreenBrownTerrain](palette_83.png)
![GreenPink](palette_84.png)
![Island](palette_85.png)
![Lake](palette_86.png)
![LightTemperature](palette_87.png)
![LightTerrain](palette_88.png)
![Mint](palette_89.png)
![Neon](palette_90.png)
![Pastel](palette_91.png)
![Pearl](palette_92.png)
![Pigeon](palette_93.png)
![Plum](palette_94.png)
![RedBlue](palette_95.png)
![Rose](palette_96.png)
![Rust](palette_97.png)
![SandyTerrain](palette_98.png)
![Sienna](palette_99.png)
![Solar](palette_100.png)
![SouthWest](palette_101.png)
![StarryNight](palette_102.png)
![Sunset](palette_103.png)
![TemperatureMap](palette_104.png)
![Thermometer](palette_105.png)
![Valentine](palette_106.png)
![VisibleSpectrum](palette_107.png)
![WaterMelon](palette_108.png)
![Cool](palette_109.png)
![Copper](palette_110.png)
![GistEart](palette_111.png)


### Interpreter Library

Many, many bugs have been fixed; thanks to everyone who has reported them!

#### Cling

Cling is now using a new just-in-time compilation engine called OrcJIT, a development based on MCJIT. It enables interpretation of inline assembly and exceptions; it will hopefully in the near future also support interpreting thread local storage (but doesn't at the moment).

Thanks to the new JIT, cling also comes with debug symbols for interpreted code; you can enable them with ".debug".

#### Function evaluation

Function calls through TMethodCall etc have been accelerated.

#### llvm / clang

llvm / clang were updated to r227800. This includes everything from the clang 3.6 release.

### Dictionary Generation

Detect usage of #pragma once for inlined headers.

Turn on verbosity of genreflex if the VERBOSE environment variable is defined.

Optimise forward declarations in rootmap files in order to make their interpretation faster.

Propagate attributes specified in xml selection files to selected classes even when selected through typedefs.

Optimise selection procedure caching selected declarations in the selection rules, therewith avoiding to query the AST twice.

Include in the PCH all the STL and C headers to guarantee portability of binaries from SLC6 to CC7.



