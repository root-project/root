## How to use TWebCanvas prototype

1. Checkout **webgui** branch and compile ROOT with http support 

    [shell] cmake -Dhttp=ON
    
2. Enable web gui factory in .rootrc file

     Gui.Factory:                web
     
3. Checkout latest JSROOT (works from 5.2.0) 
  
     [shell] git clone https://github.com/root-project/jsroot.git       
     
4. Set JSROOTSYS shell variable like

     [shell] export JSROOTSYS=/home/user/git/jsroot
     
5. Run ROOT macro with TCanvas, where TCanvas::Update is regularly called.
   One could use gui/web/demo/hsimple.C as example      
     
6. To change http server port number do:

     [shell] export WEBGUI_PORT=8877


## Compilation with CEF support (https://bitbucket.org/chromiumembedded/cef)     

1. Download binary code from http://opensource.spotify.com/cefbuilds/index.html

2. Unpack it in directory without spaces and special symbols like
  
     [shell] mkdir /d/cef
     [shell] cd /d/cef/
     [shell] wget http://opensource.spotify.com/cefbuilds/cef_binary_3.3029.1604.g364cd86_linux64.tar.bz2
     [shell] tar xjf cef_binary_3.3029.1604.g364cd86_linux64.tar.bz2  

3. Set shell variable to unpacked directory
  
     [shell] export CEF_PATH=/d/cef/cef_binary_3.3029.1604.g364cd86_linux64
     
4. Install prerequicities - see comments in $CEF_PATH/CMakeLists.txt. For the linux
   it is `build-essential`, `libgtk2.0-dev`, `libgtkglext1-dev`

5. Compile all tests (required for the libcef_dll_wrapper)
     
     [shell] mkdir /d/cef/tests
     [shell] cd /d/cef/tests
     [shell] cmake $CEF_PATH
     [shell] make -j8
     [shell] cp -r libcef_dll_wrapper $CEF_PATH


6. Compile ROOT from the same shell (CEF_PATH variable should be set)

7. Run ROOT from the same shell (CEF_PATH and JSROOTSYS variables should be set)

8. Only single canvas is supported at the moment, one get different warnings       



## Compilation with QT5 WebEngine support

1. Install libqt5-qtwebengine and libqt5-qtwebengine-devel packages

2. Compile rootqt5 main program (standard Rint plus QApplication)
  
     [shell] cd gui/web/qt5; qmake-qt5 rootqt5.pro; make     

3. Run macro in "demo" folder like:

     [shell] cd gui/web/demo;  ../qt5/rootqt5 -l hsimple.C
     

     