TEMPLATE = app

QT += webengine webenginewidgets

HEADERS += rootwebview.h

SOURCES += rootwebview.cpp rootqt5.cpp

target.path = .

INCLUDEPATH += $$system(root-config --incdir)

LIBS += $$system(root-config --glibs) -lRHTTP
