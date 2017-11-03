## [Compilation with CEF support](https://bitbucket.org/chromiumembedded/cef)

1. Current code tested with CEF3 3163, should work with other releases

2. Download binary code from [http://opensource.spotify.com/cefbuilds/index.html](http://opensource.spotify.com/cefbuilds/index.html)
   and unpack it in directory without spaces and special symbols:

~~~
     $ mkdir /d/cef
     $ cd /d/cef/
     $ wget http://opensource.spotify.com/cefbuilds/cef_binary_3.3163.1671.g700dc25_linux64.tar.bz2
     $ tar xjf cef_binary_3.3163.1671.g700dc25_linux64.tar.bz2
~~~

3. Set `CEF_PATH` shell variable to unpacked directory:

~~~
     $ export CEF_PATH=/d/cef/cef_binary_3.3163.1671.g700dc25_linux64
~~~

4. Install prerequisites - See comments in `$CEF_PATH/CMakeLists.txt`.
   For the linux it is `build-essential`, `libgtk2.0-dev`, `libgtkglext1-dev`

5. Compile all tests (required for the libcef_dll_wrapper)

~~~
     $ cd $CEF_PATH
     $ mkdir build
     $ cd build
     $ cmake $CEF_PATH
     $ make -j8
~~~

6. Compile ROOT from the same shell (CEF_PATH variable should be set)
   Check that files icudtl.dat, natives_blob.bin, snapshot_blob.bin copied into ROOT binaries directory

7. Run ROOT from the same shell (CEF_PATH and JSROOTSYS variables should be set)


## Using CEF in batch mode on Linux

CEF under Linux uses X11 functionality and therefore requires configured display and running X11 server
On the long run there is hope, that CEF introduces true headless mode - chromium itself
[already supports it](https://chromium.googlesource.com/chromium/src/+/lkgr/headless/README.md).

There is simple workaround for this problem.
One could use [Xvfb](https://en.wikipedia.org/wiki/Xvfb) as X11 server.
It does not require any graphics adapter, screen or input device.
CEF works with  Xvfb without problem.

1. Start Xvfb

~~~
     $ Xvfb :99 &
     $ export DISPLAY=:99
~~~

2. Run macro in batch mode:

~~~
     $ root -l -b draw_v6.cxx -q
~~~

## Compilation with QT5 WebEngine support

This is alternative implementation of local display, using chromium from Qt5.
It should provide very similar functionality as CEF (beside batch mode).
Biggest advantage - one gets Qt5 libraris for all platforms through normal package managers.
Later one will provide possibility to embed ROOT panels (TCanvas, TBrowser, TFitPanel) directly in Qt applications.

1. Install libqt5-qtwebengine and libqt5-qtwebengine-devel packages

2. Compile ROOT and call thisroot.sh (ROOTSYS variable should be set)

3. Compile rootqt5 main program (standard Rint plus QApplication)

~~~
     $ cd gui/canvaspainter/v7/qt5; qmake-qt5 rootqt5.pro; make
~~~

4. Run ROOT macros, using rootqt5 executable:

~~~
     $ rootqt5 -l hsimple.C
~~~
