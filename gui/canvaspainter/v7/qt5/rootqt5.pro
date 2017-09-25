TEMPLATE = app

QT += webengine webenginewidgets

HEADERS += rootwebpage.h \
           rootwebview.h

SOURCES += rootwebpage.cpp \
           rootwebview.cpp \
           rootqt5.cpp

DESTDIR = $$system(root-config --bindir)

INCLUDEPATH += $$system(root-config --incdir)

LIBS += $$system(root-config --glibs) -lRHTTP
