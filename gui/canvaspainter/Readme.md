## Compilation with CEF support (https://bitbucket.org/chromiumembedded/cef)     

1. Current code developed with CEF3 3071.

2  Download binary code from http://opensource.spotify.com/cefbuilds/index.html and 
   unpack it in directory without spaces and special symbols: 
  
     [shell] mkdir /d/cef
     [shell] cd /d/cef/
     [shell] wget http://opensource.spotify.com/cefbuilds/cef_binary_3.3071.1649.g98725e6_linux64.tar.bz2 
     [shell] tar xjf cef_binary_3.3071.1649.g98725e6_linux64.tar.bz2

3. Set `CEF_PATH` shell variable to unpacked directory:
  
     [shell] export CEF_PATH=/d/cef/cef_binary_3.3071.1649.g98725e6_linux64
     
4. Install prerequicities - see comments in $CEF_PATH/CMakeLists.txt. 
   For the linux it is `build-essential`, `libgtk2.0-dev`, `libgtkglext1-dev`

5. Compile all tests (required for the libcef_dll_wrapper)
     
     [shell] cd $CEF_PATH
     [shell] mkdir build
     [shell] cd build
     [shell] cmake $CEF_PATH
     [shell] make -j8

6. Compile ROOT from the same shell (CEF_PATH variable should be set)

7. Run ROOT from the same shell (CEF_PATH and JSROOTSYS variables should be set)

8. Only single canvas is supported at the moment, one get different warnings       



## Compilation with QT5 WebEngine support

1. Install libqt5-qtwebengine and libqt5-qtwebengine-devel packages

2. Compile ROOT and call thisroot.sh (ROOTSYS variable should be set)

3. Compile rootqt5 main program (standard Rint plus QApplication)
  
     [shell] cd gui/canvaspainter/v7/qt5; qmake-qt5 rootqt5.pro; make     

4. Run ROOT macros, using rootqt5 executable:

     [shell] rootqt5 -l hsimple.C

     