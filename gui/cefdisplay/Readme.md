## Compilation with CEF support

See details about [Chromimum Embeded Framework](https://bitbucket.org/chromiumembedded/cef)

1. Current code tested with CEF3 branch 4147, Chromium 84 (August 2020)

2. Download binary code from [http://opensource.spotify.com/cefbuilds/index.html](http://opensource.spotify.com/cefbuilds/index.html) and unpack it in directory without spaces and special symbols:

~~~
     $ mkdir /d/cef
     $ cd /d/cef/
     $ wget http://opensource.spotify.com/cefbuilds/cef_binary_84.4.1+gfdc7504+chromium-84.0.4147.105_linux32.tar.bz2
     $ tar xjf cef_binary_84.4.1+gfdc7504+chromium-84.0.4147.105_linux32.tar.bz2
~~~


3. As it is on 21.08.2020, CEF has problem to compile with latest gcc, issuing error because of warnings in some standard libraries.
   Therefore one has to modify cmake/cef_variables.cmake, removing code at line ~90:

~~~
   -Werror                         # Treat warnings as errors
~~~

4. Set `CEF_ROOT` shell variable to unpacked directory:

~~~
     $ export CEF_ROOT=/d/cef/cef_binary_84.4.1+gfdc7504+chromium-84.0.4147.105_linux64
~~~

5. Install prerequisites - see comments in `$CEF_ROOT/CMakeLists.txt`.
   For the linux these are: `build-essential`, `libgtk2.0-dev`, `libgtkglext1-dev`

6. Compile CEF to produce `libcef_dll_wrapper`:

~~~
     $ cd $CEF_ROOT
     $ mkdir build
     $ cd build
     $ cmake $CEF_ROOT
     $ make -j
~~~

7. When configure ROOT compilation with `cmake -Dwebgui=ON -Dcefweb=ON ...`, CEF_ROOT shell variable should be set appropriately.
   During compilation library `$ROOTSYS/lib/libROOTCefDisplay.so` and executable `$ROOTSYS/bin/cef_main`
   should be created. Also check that several files like `icudtl.dat`, `v8_context_snapshot_blob.bin`, `snapshot_blob.bin`
   copied into ROOT library directory

8. Run ROOT with `--web=cef` argument to use CEF web display like

~~~
   $ root --web=cef $ROOTSYS/tutorials/v7/draw_rh2.cxx
~~~


## Using plain CEF in ROOT batch mode on Linux

CEF under Linux uses X11 functionality and therefore requires configured display and running X11 server
Unfortunately, CEF developers [will not support](https://bitbucket.org/chromiumembedded/cef/issues/2349/)
chrome headless mode, where X11 server can be avoided completely.

Nevertheless, ROOT allow to use CEF in batch mode, running `Xvfb` server. Most simple way is to use
`xvfb-run` utility like:

~~~
      $ xvfb-run --server-args='-screen 0, 1024x768x16'  root.exe -l --web=cef $ROOTSYS/tutorials/v7/line.cxx -q
~~~

Or run `Xvfb` before starting ROOT:

~~~
     $ Xvfb :99 &
     $ export DISPLAY=:99
     $ root.exe -l --web=cef $ROOTSYS/tutorials/v7/line.cxx -q
~~~


## Using CEF with ozone driver in headless mode

Since March 2019 one can compile [CEF without X11](https://bitbucket.org/chromiumembedded/cef/issues/2296/), but this
is not included in available binary builds. Therefore to be able use it, one should compile CEF from source.
[Here](https://bitbucket.org/chromiumembedded/cef/wiki/AutomatedBuildSetup.md) is complete comilation documentation.
As usual - main problem is to provide all necessary dependency to be able compile it.



