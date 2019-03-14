## Compilation with CEF support

See details about [Chromimum Embeded Framework](https://bitbucket.org/chromiumembedded/cef)

1. Current code tested with CEF3 3.3325, should work with other recent releases (April 2018)

2. Download binary code from [http://opensource.spotify.com/cefbuilds/index.html](http://opensource.spotify.com/cefbuilds/index.html) and unpack it in directory without spaces and special symbols:

~~~
     $ mkdir /d/cef
     $ cd /d/cef/
     $ wget http://opensource.spotify.com/cefbuilds/cef_binary_3.3626.1895.g7001d56_linux64_minimal.tar.bz2
     $ tar xjf cef_binary_3.3626.1895.g7001d56_linux64_minimal.tar.bz2
~~~


3. As it is on 14.03.2019, CEF has problem to compile with gcc. Master already [patched](https://bitbucket.org/chromiumembedded/cef/commits/84a5749), but patch is not yet appeared in the distribution. Therefore one has to modify cmake/cef_variables.cmake, iserting code at line approx 164:

~~~
   if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
     list(APPEND CEF_CXX_COMPILER_FLAGS
        -Wno-attributes             # The cfi-icall attribute is not supported by the GNU C++ compiler
     )
   endif()
~~~

4. Set `CEF_PATH` shell variable to unpacked directory:

~~~
     $ export CEF_PATH=/d/cef/cef_binary_3.3626.1895.g7001d56_linux64_minimal
~~~

5. Install prerequisites - see comments in `$CEF_PATH/CMakeLists.txt`.
   For the linux these are: `build-essential`, `libgtk2.0-dev`, `libgtkglext1-dev`

6. Compile to produce libcef_dll_wrapper:

~~~
     $ cd $CEF_PATH
     $ mkdir build
     $ cd build
     $ cmake $CEF_PATH
     $ make -j8
~~~

7. Compile ROOT from the same shell (CEF_PATH variable should be set)
   Check that files icudtl.dat, natives_blob.bin, snapshot_blob.bin copied into ROOT binaries directory

8. Run ROOT from the same shell (CEF_PATH variable should be set)



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
     $ root -l -b --web cef draw_v6.cxx -q
~~~

Or one can start with special `xvfb-run` script which starts Xvfb, executes root macro and then stop Xvfb

~~~
     $ xvfb-run --server-args='-screen 0, 1024x768x16' root -l -b --web cef draw_file.cxx -q
~~~



