demo/qt/README.txt

This directory contains Qt-Cint demo programs.

 Qt-Cint is still experimantal. There are limitations. Interface 
currently provided may be changed.  

Platform:

  Currently, Qt-Cint is supported only on Windows with Visual C++ 6.0/7.x
 compiler.

Preparation:

  In order to run programs in this directory, you need to build 
 include\qtcint.dll in lib\qt directory. Please refer to lib\qt\README.txt. 

How to run:

  c:\>  cint test1.cxx
  c:\>  cint test2.cxx
             .
             .


Limitation and required limitation:

  In order to use Qt library in Cint, you need to modify source code as
 follows.

 1. Qt header file
     Cint does not recognize Qt header files as they are. Instead, you 
    just include <qtcint.dll>. 

         #ifdef __CINT__

         #pragma include <qtcint.dll>

         #else

         #include <qapplication.h>
         #include <qpushbutton.h>
         #include <qslider.h>
         #include <qlcdnumber.h>
         #include <qfont.h>
         #include <qvbox.h>
         #include <qgrid.h>
         
         #endif

 2. qApp object
     I do not know the reason why, but makecint fails to find a linkage to
    qApp object. qApp object you see in Qt-Cint is a dummy which you can 
    set manually.

        int main( int argc, char **argv ) {
           QApplication a( argc, argv );

        #ifdef __CINT__
           //Cint workaround, don't know why linker can not find qApp 
           //Set a dummy version declared in qtstatic.cxx
           qApp = &a; 
        #endif
             .
             .
        }


 3. connect() command 
     Cint can not process command macros used with connect() statement.
    You need to modify them as follows.

         #ifdef __CINT__
             connect( slider, "2valueChanged(int)", lcd, "1display(int)" );
         #else
             connect( slider,SIGNAL(valueChanged(int)),lcd,SLOT(display(int)));
         #endif

         #ifdef __CINT__
             connect( quit, "2clicked()", qApp, "1quit()" );
         #else
             connect( quit, SIGNAL(clicked()), qApp, SLOT(quit()) );
         #endif

     METHOD(x) is replaced as "0x"
     SLOT(x)   is replaced as "1x"
     SIGNAL(x) is replaced as "2x"

