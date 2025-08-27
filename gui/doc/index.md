\defgroup gui GUI
\brief Graphical User Interface

The ROOT GUI classes support an extensive and rich set of widgets. The widget classes
interface to the underlying graphics system via a single abstract class making the ROOT GUI
fully cross-platform.


\defgroup webwidgets Web Widgets
\brief A Graphical User Interface based on WEB technology

In case of using web browsers based on snap sandboxing, if you see a runtime error about unauthorized access to the system /tmp/ folder, try callign `export TMPDIR=/home/user/mytemp` (adapt path to a real folder) before running ROOT.