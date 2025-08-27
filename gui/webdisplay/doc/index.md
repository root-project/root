\defgroup webdisplay Web Display
\brief A Graphical User Interface based on WEB technology

In case of using web browsers based on snap sandboxing, if you see a runtime error about unauthorized access to the system /tmp/ folder, try callign `export TMPDIR=/home/user/mytemp` (adapt path to a real folder) before running ROOT.