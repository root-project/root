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
