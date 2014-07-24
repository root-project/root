
QT += qt3support
CONFIG += uic3

SOURCES += main.cpp
FORMS = qtrootexample1.ui
IMAGES = images/qtroot_canvas.png images/h1_t.png images/h2_t.png
TEMPLATE =app
CONFIG += qt warn_on thread
INCLUDEPATH += $(ROOTSYS)/include
LIBS += -L$(ROOTSYS)/lib -lCore -lRIO -lNet -lMathCore -lCint -lHist -lGraf -lGraf3d -lGpad -lGui -lTree -lRint -lPostscript -lMatrix -lPhysics -lThread -lQtGSI -lnsl -lm -ldl -rdynamic $(SYSLIBS)
DBFILE = qtrootexample.db
LANGUAGE = C++
