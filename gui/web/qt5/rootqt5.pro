TEMPLATE = app

QT += webengine webenginewidgets

HEADERS += rootwebview.h

SOURCES += rootwebview.cpp rootqt5.cpp

# RESOURCES += qml.qrc

target.path = .
# INSTALLS += target

INCLUDEPATH += $$system(root-config --incdir)

LIBS += $$system(root-config --glibs) -lRHTTP

#LIBS += -Wl,-rpath,/home/linev/soft/build6/lib -L/home/linev/soft/build6/lib -lGui -lCore -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -pthread -lm -ldl -rdynamic