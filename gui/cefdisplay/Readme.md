\page cefpage ROOT installation with CEF

## Compilation with CEF support

See details about [Chromium Embedded Framework](https://bitbucket.org/chromiumembedded/cef)

1. Current code tested with CEF3 branch 778, Chromium 148 (June 2026)

2. Download binary code from [https://cef-builds.spotifycdn.com/index.html](https://cef-builds.spotifycdn.com/index.html)
   and unpack it in directory without spaces and special symbols:

~~~
     $ mkdir /d/cef
     $ cd /d/cef/
     $ wget https://cef-builds.spotifycdn.com/cef_binary_148.0.10%2Bg7ee53f5%2Bchromium-148.0.7778.218_linux64.tar.bz2
     $ tar xjf cef_binary_148.0.10+g7ee53f5+chromium-148.0.7778.218_linux64.tar.bz2
~~~


3 Install prerequisites - see comments in package `CMakeLists.txt`.
   For the linux these are: `build-essential`, `libgtk3.0-dev`

4. Compile CEF to produce `libcef_dll_wrapper`:

~~~
     $ cd xjf cef_binary_148.0.10+g7ee53f5+chromium-148.0.7778.218_linux64
     $ mkdir build
     $ cd build
     $ cmake ..
     $ make -j libcef_dll_wrapper
~~~

5. Set CEF_ROOT variable to unpacked directory:

~~~
     $ export CEF_ROOT=/d/cef/xjf cef_binary_148.0.10+g7ee53f5+chromium-148.0.7778.218_linux64
~~~

6. When configure ROOT compilation with `cmake -Dwebgui=ON -Dcefweb=ON ...`, CEF_ROOT shell variable should be set appropriately. During compilation library `$ROOTSYS/lib/libROOTCefDisplay.so` and executable `$ROOTSYS/bin/cef_main` should be created. Also check that several files like `icudtl.dat`, `v8_context_snapshot_blob.bin`, `snapshot_blob.bin` copied into ROOT library directory

7. Run ROOT with `--web=cef` argument to use CEF web display like:

~~~
   $ root --web=cef $ROOTSYS/tutorials/hsimple.C
~~~


## Compile libcef_dll_wrapper on Windows

1. Download binary win32 build like [win64](https://cef-builds.spotifycdn.com/cef_binary_148.0.10%2Bg7ee53f5%2Bchromium-148.0.7778.218_windows64.tar.bz2)

2. Extract in directory without spaces like `C:\Soft\cef`

3. Modify `cmake/cef_variables.cmake` to set dynamic linking, replace "/MT" by "/MD" in approx line 389

4. Start "x86 Native tools Command Prompt for VS 2019". Do:
~~~
   $ cd C:\Soft\cef
   $ mkdir build
   $ cd build
   $ cmake ..
   $ cmake --build . --config Release --target libcef_dll_wrapper
~~~

5. Before compiling ROOT, `set CEF_ROOT=C:\Soft\cef` variable


## Using plain CEF in ROOT batch mode on Linux and Windows

Default CEF builds, provided by [https://cef-builds.spotifycdn.com/index.html](https://cef-builds.spotifycdn.com/index.html), now (June 2026) **INCLUDES!** support of Ozone framework, which allows to run CEF in headless mode on Linux and Windows. So one can run different ROOT macros and tests in batch mode like

~~~
      $ cd $ROOTSYS/test
      $ ./stressGraphics -b --web=cef
~~~
