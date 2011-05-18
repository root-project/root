import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *


import ROOT
import sip

class window(QMainWindow):
   def __init__(self):
       # Init the main window.
        QMainWindow.__init__(self)
        self.resize(350, 350)
  
     # Create the central widget.
        self.CentralWidget = QWidget(self)
        self.setCentralWidget(self.CentralWidget)
        self.Layout = QGridLayout(self.CentralWidget)
  
    # Create a button.
        self.QuitButton    = QPushButton(self.centralWidget())
        self.QuitButton.setText('Quit')
        self.Layout.addWidget(self.QuitButton, 1, 0)
    # Connect the button.
        QObject.connect(self.QuitButton, SIGNAL('clicked()'), self.quit)
     
    # Create a root histogram.
        self.hist = ROOT.TH1F("pipo","pipo", 100, 0, 100)
  
    # Create the main TQtWidget (using sip to get the pointer to the central widget).
        self.Address = sip.unwrapinstance(self.CentralWidget)
        self.Canvas = ROOT.TQtWidget(sip.voidptr(self.Address).ascobject())
        ROOT.SetOwnership( self.Canvas, False )
  
    # Place the TQtWidget in the main grid layout and draw the histogram.
       
        self.Layout.addWidget(sip.wrapinstance(ROOT.AddressOf(self.Canvas)[0],QWidget), 0, 0)
        self.hist.Draw()

   def quit(self):
       print 'Bye bye...'
       self.close()
       ROOT.gApplication.Terminate()


if __name__ == '__main__':
   application = qApp
   terminator = ROOT.TQtRootSlot.CintSlot()
   termAddress = sip.wrapinstance(ROOT.AddressOf(terminator)[0],QObject)
   QObject.connect(application, SIGNAL("lastWindowClosed()"),termAddress ,SLOT("Terminate()"))
   w = window()
   w.show()
   ROOT.gApplication.Run(1)
   print 'Bye forever!'
