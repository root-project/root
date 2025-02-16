## \file
## \ingroup tutorial_graphics
##
##
## This script plays a recorded ROOT session showing how to perform various
## interactive graphical editing operations. The initial graphic setup
## was created using the following root commands:
##
## ~~~{.cpp}
##     TRecorder *t = new TRecorder();
##     t->Start("graphedit_playback.root");
##     gStyle->SetPalette(1);
##     TCanvas *c2 = new TCanvas("c2","c2",0,0,700,500);
##     TH2F* h2 = new TH2F("h2","Random 2D Gaussian",40,-4,4,40,-4,4);
##     h2->SetDirectory(0);
##     TRandom r;
##     for (int i=0;i<50000;i++) h2->Fill(r.Gaus(),r.Gaus());
##     h2->Draw();
##     gPad->Update();
##     TCanvas *c1 = new TCanvas("c1","c1",0,0,700,500);
##     TH1F* h1 = new TH1F("h1","Random 1D Gaussian",100,-4,4);
##     h1->SetDirectory(0);
##     h1->FillRandom("gaus",10000);
##     h1->Draw();
##     gPad->Update();
##
##     # Here the following "sketch" was done.
##
##     t->Stop();
## ~~~
## or in Python version.
## ~~~{.py}
##     import ROOT
##     TH1F = ROOT.TH1F
##     TH2F = ROOT.TH2F
##     TCanvas = ROOT.TCanvas
##     TRecorder = ROOT.TRecorder
##     TRandom = ROOT.TRandom
##     gPad = ROOT.gPad
## 
##     t = TRecorder()
##     t.Start("graphedit_playback.root")
##     gStyle.SetPalette(1)
##     c2 = TCanvas("c2","c2",0,0,700,500)
##     h2 = TH2F("h2","Random 2D Gaussian",40,-4,4,40,-4,4)
##     h2.SetDirectory(0)
##     r = TRandom()
##     for i in range(50000): h2.Fill(r.Gaus(),r.Gaus())
##     h2.Draw()
##     gPad.Update()
##     c1 = TCanvas("c1","c1",0,0,700,500)
##     h1 = TH1F("h1","Random 1D Gaussian",100,-4,4)
##     h1.SetDirectory(0)
##     h1.FillRandom("gaus",10000)
##     h1.Draw()
##     gPad.Update()
##
##     # Here the following "sketch" was done.
##
##     t.Stop()
## ~~~
## 
## Note: The previous commands(whichever one of) should be copy/pasted into a ROOT session, not
## executed as a macro. In Python, it should be loaded by using the IPy[ ]: session.
##
## ### The interactive editing shows:
##     - Object editing using object editors
##     - Direct editing on the graphics canvas
##     - Saving PS and bitmap files.
##     - Saving as a .C file: C++ code corresponding to the modifications is saved.
##       Note: It can be saved as Python-script since the TCanvas core is still running 
##             on ROOT.
##
## ### The sketch of the recorded actions is
##
##    #### On the canvas c1
##      - Open View/Editor
##      - Select histogram
##      - Change fill style
##      - Change fill color
##      - Move stat box
##      - Change fill color
##      - Move title
##      - Change fill color using wheel color
##      - Select Y axis
##      - Change axis title
##      - Select X axis
##      - Change axis title
##      - Select histogram
##      - Go in binning
##      - Change range
##      - Move range
##      - On the canvas menu set grid Y
##      - On the canvas menu set grid X
##      - On the canvas menu set log Y
##      - Increase the range
##      - Close View/Editor
##      - Open the Tool Bar
##      - Create a text "Comment"
##      - Create an arrow
##      - Change the arrow size
##      - Close the Tool Bar
##      - Save as PS file
##      - Save as C file
##      - Close c1
##    #### On the canvas c2
##      - Open View/Editor
##      - Select histogram
##      - Select COL
##      - Select Palette
##      - Move Stats
##      - Select Overflows
##      - Select histogram
##      - Select 3D
##      - Select SURF1
##      - Rotate Surface
##      - Go in binning
##      - Change X range
##      - Change Y range
##      - Close View/Editor
##      - Save as GIF file
##      - Save as C file
##      - Close c2
## \author Olivier Couet
## \translator P. P.


import ROOT

#classes
TRecorder = ROOT.TRecorder
TMath = ROOT.TMath

#types
FileStat_t = ROOT.FileStat_t
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
char = ROOT.char

#globals
gSystem = ROOT.gSystem

# Int_t
def file_size(filename : char) :
   fs = FileStat_t()
   ROOT.gSystem.GetPathInfo(filename, fs)
   return fs.fSize # Int_t
   

# void
def graph_edit_playback() :
   
   global r
   r = TRecorder()
   r.Replay("http://root.cern.ch/files/graphedit_playback.root")
   
   # wait for the recorder to finish the replay
   counter = 0
   while (r.GetState() == TRecorder.kReplaying) :
      ROOT.gSystem.ProcessEvents()
      ROOT.gSystem.Sleep(1)
      #print("r.GetState() : ", r.GetState() )
      counter += 1
      if counter == 1000: print("r.GetState() : ", r.GetState() )
      if counter > 100000: break #Otherwise, the play never ends.
      
   
   c1_ps_Ref = 11592;    c1_ps_Err = 600
   c1_C_Ref = 4729;    c1_C_Err = 200
   c2_gif_Ref = 21184;    c2_gif_Err = 500
   c2_C_Ref = 35471;    c2_C_Err = 1500
   
   c1_ps = file_size("c1.ps")
   c1_C = file_size("c1.C")
   c2_gif = file_size("c2.gif")
   c2_C = file_size("c2.C")
   
   print(f"**********************************************************************" )
   print(f"*  Report of graph_edit_playback.py                                  *" )
   print(f"**********************************************************************" )
   
   if (TMath.Abs(c1_ps_Ref-c1_ps) <= c1_ps_Err):
      print(f"Canvas c1: PS output............................................... OK" )
      
   else:
      print(f"Canvas c1: PS output........................................... FAILED" )
      
   if (TMath.Abs(c1_C_Ref-c1_C) <= c1_C_Err):
      print(f"           C output................................................ OK" )
      
   else:
      print(f"           C output............................................ FAILED" )
      
   if (TMath.Abs(c2_gif_Ref-c2_gif) <= c2_gif_Err):
      print(f"Canvas c2: GIF output.............................................. OK" )
      
   else:
      print(f"Canvas c2: GIF output.......................................... FAILED" )
      
   if (TMath.Abs(c2_C_Ref-c2_C) <= c2_C_Err):
      print(f"           C output................................................ OK" )
      
   else:
      print(f"           C output............................................ FAILED" )
      
   print(f"**********************************************************************" )
   
   


if __name__ == "__main__":
   graph_edit_playback()
