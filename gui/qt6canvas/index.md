\defgroup qt6canvas QT6 Canvas Display
\ingroup gpad
\brief Classes for canvas display using QT6

CAUTION! Highly experimental, constantly changing code

Provides alternative graphics implementation in the ROOT.
Allows non-GL painting in Qt6 QWidget. Implementes:
   - painting via TVirtualPadPainter interface
   - canvas menu
   - status bar
   - tools bar
   - objects context menu
   - basics graphical editors

To use this class one need to compile ROOT with `-Dqt6canvas=ON` flag.
Qt6 gui library should be installed on the system.
To activate Qt6 canvase in ROOT application one should create .rootrc file with following entries:

```
Gui.Factory: qt6
Root.PadEditor: qt6
```
