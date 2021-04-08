# qt5web example

Demonstration how different ROOT web-based widgets can be embed into qt application

## Compile on linux

Create build directory and call:

    cmake $ROOTSYS/tutorials/webgui/qt5web
    make -j

ROOT should be compiled with configured `-Dqt5web=ON`.
As a result, `qt5web` application should be created.


## Compile on Windows

Run x86 native tools shell from MS VC. Configure ROOT and Qt5 pathes like:

    set PATH=%PATH%;C:\Qt\5.12.6\msvc2017\bin
    call C:\root\bin\thisroot.bat
    cmake -G"Visual Studio 16 2019" -A Win32 -Thost=x64 c:\root\tutorials\webgui\qt5web
    cmake --build . --config Debug -- /maxcpucount

As a result, `Debug\qt5web.exe` executable should be created.


## Demo application

Application based on `QTabWidget` with four tabs - standard Qt5 widget,
TCanvas, RCanvas and geometry drawing. Both canvas variants include different histograms drawing.


## How to include RCanvas into other Qt-based project

Most easy way - just include `RCanvasWidget.h` and `RCanvasWidget.cpp` files
in the project and let compile, linking with ROOT basic libraries `root-config --libs` plus `-lROOTWebDisplay -lROOTGpadv7`.
`RCanvasWidget` is just `QWidget` which internally embed `RCanvas` drawing. See `ExampleWidget.ui` file how to embed such custom widget in normal qt ui file.

To let ROOT work inside Qt5 event loop, one should instantiate `TApplication` object and
regularly call `gSystem->ProcessEvents()` - see how it is done in `ExampleMain.cpp`.

Author: Sergey Linev
