## [Compilation with CEF support](https://bitbucket.org/chromiumembedded/cef)

1. Current code tested with CEF3 3.3282, should work with other releases

2. Download binary code from [http://opensource.spotify.com/cefbuilds/index.html](http://opensource.spotify.com/cefbuilds/index.html)
   and unpack it in directory without spaces and special symbols:

~~~
     $ mkdir /d/cef
     $ cd /d/cef/
     $ wget http://opensource.spotify.com/cefbuilds/cef_binary_3.3282.1732.g973997f_linux32.tar.bz2
     $ tar xjf cef_binary_3.3282.1732.g973997f_linux32.tar.bz2
~~~

3. Set `CEF_PATH` shell variable to unpacked directory:

~~~
     $ export CEF_PATH=/d/cef/cef_binary_3.3282.1732.g973997f_linux32
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

