## \file
## \ingroup tutorial_hist
## \notebook
##
## This creates a grey scale of `200 x 200` boxes.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TBox = ROOT.TBox
TPad = ROOT.TPad
TText = ROOT.TText
TColor = ROOT.TColor

#types
Float_t = ROOT.Float_t

#globals
gEnv = ROOT.gEnv

def greyscale():

   global c
   c = TCanvas("grey", "Grey Scale", 500, 500)
   c.SetBorderMode(0)
   
   n = 200; # tunable parameter
   n1 = 1./n
   #for (int i = 0; i < n; i++) {
   #   for (int j = 0; j < n; j++) {
   global b_list, grey_list
   b_list = grey_list = []
   for i in range(0, n, 1):
      for j in range(0, n, 1):

         b = TBox(n1*j, n1*(n-1-i), n1*(j+1), n1*(n-i))
         b_list.append( b )
         grey = Float_t(i*n+j)/(n*n)
         grey_list.append( grey )

         b.SetFillColor(TColor.GetColor(grey, grey, grey))
         b.Draw()
         
      
   global p, guibackgroudn
   p = TPad("p","p",0.3, 0.3, 0.7,0.7)
   guibackground = gEnv.GetValue("Gui.BackgroundColor", "")
   p.SetFillColor(TColor.GetColor(guibackground))
   p.Draw()
   p.cd()

   global t
   t = TText(0.5, 0.5, "GUI Background Color")
   t.SetTextAlign(22)
   t.SetTextSize(.09)
   t.Draw()
   
   c.SetEditable(False)
   


if __name__ == "__main__":
   greyscale()
