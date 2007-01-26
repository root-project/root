import sys
import qt

import ROOT
import sip

class window(qt.QMainWindow):
   def __init__(self):
       # Init the main window.
        qt.QMainWindow.__init__(self)
        self.resize(350, 350)
  
     # Create the central widget.
        self.CentralWidget = qt.QWidget(self)
        self.setCentralWidget(self.CentralWidget)
        self.Layout = qt.QGridLayout(self.CentralWidget)
  
    # Create a button.
        self.QuitButton    = qt.QPushButton(self.centralWidget())
        self.QuitButton.setText('Quit')
        self.Layout.addWidget(self.QuitButton, 1, 0)
    # Connect the button.
        qt.QObject.connect(self.QuitButton, qt.SIGNAL('clicked()'), self.quit)
     
    # Create a root histogram.
        self.hist = ROOT.TH1F("pipo","pipo", 100, 0, 100)
  
    # Create the main TQtWidget (using sip to get the pointer to the central widget).
        self.Address = sip.unwrapinstance(self.CentralWidget)
        self.Canvas = ROOT.TQtWidget(sip.voidptr(self.Address).ascobject())
  
    # Place the TQtWidget in the main grid layout and draw the histogram.
       
        self.Layout.addWidget(sip.wrapinstance(ROOT.AddressOf(self.Canvas)[0],qt.QWidget), 0, 0)
        self.hist.Draw()

   def quit(self):
       print 'Bye bye...'
       self.close()
       
       
if __name__ == '__main__':
   application = qt.qApp
   terminator = ROOT.TQtRootSlot.CintSlot()
   termAddress = sip.wrapinstance(ROOT.AddressOf(terminator)[0],qt.QObject)
   qt.QObject.connect(application, qt.SIGNAL("lastWindowClosed()"),termAddress ,qt.SLOT("Terminate()"))
   w = window()
   w.show()
   ROOT.gApplication.Run(1)
   print 'Bye forever!'
