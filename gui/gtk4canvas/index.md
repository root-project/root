\defgroup gtk4canvas GTK4 Canvas Display
\ingroup gpad
\brief Classes for canvas display using GTK4

CAUTION! Highly experimental, constantly changing code

Provides alternative graphics implementation in the ROOT.
Allows non-GL painting in GTK4. Implementes:
   - painting via TVirtualPadPainter interface
   - canvas menu
   - status bar
   - objects context menu
   - basics graphical editors

To use this class one need to compile ROOT with `-Dgtk4canv=ON` flag.
GTK4 library should be installed on the system.
To activate GTK4 canvas in ROOT application one should create .rootrc file with following entries:

```
Gui.Factory: gtk4
Root.PadEditor: gtk4
```
