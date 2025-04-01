# qtweb example

Demonstration how different ROOT web-based widgets can be embed into qt application

## Compile on Linux

Create build directory and call:

    cmake $ROOTSYS/tutorials/visualisation/webgui/qtweb
    make -j

ROOT should be compiled with configured `-Dqt6web=ON`.
As a result, `qtweb` application should be created.

## Compile on Windows

Run x86 native tools shell from MS VC.  Configure Qt5 pathes:

    set PATH=%PATH%;C:\Qt6\6.8.2\msvc2019\bin

Compile ROOT with qt6web support in Release mode:

    cd C:\
    mkdir root
    cd C:\root
    cmake -G"Visual Studio 16 2019" -A Win32 -Thost=x64 c:\git\root -Droot7=ON -DCMAKE_CXX_STANDARD=17 -Dwebgui=ON -Dqt6web=ON
    cmake --build . --config Release -- /maxcpucount

Configure ROOT, create build directory and build qtweb tutorial:

    call C:\root\bin\thisroot.bat
    cd C:\
    mkdir qtweb
    cd C:\qtweb
    cmake -G"Visual Studio 16 2019" -A Win32 -Thost=x64 c:\root\tutorials\visualisation\webgui\qtweb
    cmake --build . --config Release -- /maxcpucount

As a result, `Release\qtweb.exe` executable should be created.


## Demo application

Application based on `QTabWidget` with four tabs - standard Qt widget,
TCanvas, RCanvas and geometry drawing. Both canvas variants include different histograms drawing.


## How to include RCanvas/TCanvas into other Qt-based project

Most easy way - just include `RCanvasWidget.h` and `RCanvasWidget.cpp` files
in the project and let compile, linking with ROOT basic libraries `root-config --libs` plus `-lROOTWebDisplay -lROOTGpadv7`.
`RCanvasWidget` is just `QWidget` which internally embed `RCanvas` drawing.
See `ExampleWidget.ui` file how to embed such custom widget in normal qt ui file.

To let ROOT work inside Qt event loop, one should instantiate `TApplication` object and
regularly call `gSystem->ProcessEvents()` - see how it is done in `ExampleMain.cpp`.

Author: Sergey Linev
