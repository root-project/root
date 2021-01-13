all: qt5web

ROOTINCPATH := $(shell root-config --incdir)
ROOTCFLAGS  := $(shell root-config --cflags)
ROOTLIBS    := $(shell root-config --libs)
ROOTGLIBS   := $(shell root-config --glibs)

ifneq ($(findstring -std=c++11,$(ROOTCFLAGS)),)
   QMAKEOPTFLAGS := "CONFIG+=c++11"
endif
ifneq ($(findstring -std=c++14,$(ROOTCFLAGS)),)
    QMAKEOPTFLAGS := "CONFIG+=c++14"
endif
ifneq ($(findstring -std=c++17,$(ROOTCFLAGS)),)
    QMAKEOPTFLAGS := "CONFIG+=c++17"
endif

HDRS = ExampleWidget.h TCanvasWidget.h RCanvasWidget.h
SRCS = ExampleWidget.cpp TCanvasWidget.cpp RCanvasWidget.cpp ExampleMain.cpp  
UIS = ExampleWidget.ui

Makefile.qt: qt5web.pro $(UIS)
	@echo "Generating Makefile.qt"
	qmake-qt5 qt5web.pro -o Makefile.qt $(QMAKEOPTFLAGS) "LIBS+=$(ROOTGLIBS) -lROOTWebDisplay -lWebGui6 -lROOTGpadv7" "INCLUDEPATH+=$(ROOTINCPATH)" "DEPENDPATH+=$(ROOTINCPATH)" 

qt5web: Makefile.qt $(HDRS) $(SRCS)
	@echo "Compiling qt5web..."
	$(MAKE) -f Makefile.qt

clean:
	@echo "Clean generated files"
	rm -f Makefile.qt ui_ExampleWidget.h .qmake.stash qt5web
	rm -rf .obj .moc 
