## \file
## \ingroup tutorial_graphics
## \notebook
## This macro displays the ROOT architecture.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas 
TPaveLabel = ROOT.TPaveLabel
TPavesText = ROOT.TPavesText
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow
TWbox = ROOT.TWbox
TPad = ROOT.TPad
TBox = ROOT.TBox
TPad = ROOT.TPad



# void
def archi() :

   global c1
   c1 = TCanvas("c1","Dictionary Architecture",20,10,750,930)
   c1.SetBorderSize(0)
   c1.Range(0,0,20.5,26)
   
   global title
   title = TPaveLabel(4,24,16,25.5,c1.GetTitle())
   title.SetFillColor(46)
   title.Draw()
   
   global dll
   dll = TPavesText(0.5,19,4.5,23,5,"tr")
   dll.SetFillColor(39)
   dll.SetTextSize(0.023)
   dll.AddText(" ")
   dll.AddText("Dynamically")
   dll.AddText("Linked")
   dll.AddText("Libraries")
   dll.Draw()
   
   global dlltitle
   dlltitle = TPaveLabel(1.5,22.6,3.5,23.3,"DLLs")
   dlltitle.SetFillColor(28)
   dlltitle.Draw()
   
   global cpp
   cpp = TPavesText(5.5,19,9.5,23,5,"tr")
   cpp.SetTextSize(0.023)
   cpp.AddText(" ")
   cpp.AddText("Commented")
   cpp.AddText("Header")
   cpp.AddText("Files")
   cpp.Draw()
   
   global cpptitle
   cpptitle = TPaveLabel(6.5,22.6,8.5,23.3,"C++")
   cpptitle.SetFillColor(28)
   cpptitle.Draw()
   
   global odl
   odl = TPavesText(10.5,19,14.5,23,5,"tr")
   odl.SetTextSize(0.023)
   odl.AddText(" ")
   odl.AddText("Objects")
   odl.AddText("Description")
   odl.AddText("Files")
   odl.Draw()
   
   global odltitle
   odltitle = TPaveLabel(11.5,22.6,13.5,23.3,"ODL")
   odltitle.SetFillColor(28)
   odltitle.Draw()
   
   global idl
   idl = TPavesText(15.5,19,19.5,23,5,"tr")
   idl.SetTextSize(0.023)
   idl.AddText(" ")
   idl.AddText("Interface")
   idl.AddText("Definition")
   idl.AddText("Language")
   idl.Draw()

   global idltitle
   idltitle = TPaveLabel(16.5,22.6,18.5,23.3,"IDL")
   idltitle.SetFillColor(28)
   idltitle.Draw()
   
   global p1
   p1 = TWbox(7.8,10,13.2,17,11,12,1)
   p1.Draw()

   global pro1
   pro1 = TText(10.5,15.8,"Process 1")
   pro1.SetTextAlign(21)
   pro1.SetTextSize(0.03)
   pro1.Draw()

   global p1dict
   p1dict = TPaveText(8.8,13.8,12.2,15.6)
   p1dict.SetTextSize(0.023)
   p1dict.AddText("Dictionary")
   p1dict.AddText("in memory")
   p1dict.Draw()

   global p1object
   p1object = TPavesText(8.6,10.6,12.1,13.0,5,"tr")
   p1object.SetTextSize(0.023)
   p1object.AddText("Objects")
   p1object.AddText("in memory")
   p1object.Draw()

   global p2
   p2 = TWbox(15.5,10,20,17,11,12,1)
   p2.Draw()

   global pro2
   pro2 = TText(17.75,15.8,"Process 2")
   pro2.SetTextAlign(21)
   pro2.SetTextSize(0.03)
   pro2.Draw()

   global p2dict
   p2dict = TPaveText(16,13.8,19.5,15.6)
   p2dict.SetTextSize(0.023)
   p2dict.AddText("Dictionary")
   p2dict.AddText("in memory")
   p2dict.Draw()

   global p2object
   p2object = TPavesText(16.25,10.6,19.25,13.0,5,"tr")
   p2object.SetTextSize(0.023)
   p2object.AddText("Objects")
   p2object.AddText("in memory")
   p2object.Draw()

   global stub1
   stub1 = TWbox(12.9,11.5,13.6,15.5,49,3,1)
   stub1.Draw()

   global tstub1
   tstub1 = TText(13.25,13.5,"Stub1")
   tstub1.SetTextSize(0.025)
   tstub1.SetTextAlign(22)
   tstub1.SetTextAngle(90)
   tstub1.Draw()

   global stub2
   stub2 = TWbox(15.1,11.5,15.8,15.5,49,3,1)
   stub2.Draw()

   global tstub2
   tstub2 = TText(15.45,13.5,"Stub2")
   tstub2.SetTextSize(0.025)
   tstub2.SetTextAlign(22)
   tstub2.SetTextAngle(-90)
   tstub2.Draw()
   
   global ar1
   ar1 = TArrow()
   ar1.SetLineWidth(6)
   ar1.SetLineColor(1)
   ar1.SetFillStyle(1001)
   ar1.SetFillColor(1)
   ar1.DrawArrow(13.5,14,15,14,0.012,"|>")
   ar1.DrawArrow(15.1,13,13.51,13,0.012,"|>")
   

   global cint
   cint = TPaveText(1.0,15.0,8.0,17.5)
   cint.SetFillColor(39)
   cint.SetBorderSize(1)
   cint.SetTextSize(0.023)
   cint.AddText("C++ Interpreter")
   cint.AddText("and program builder")
   cint.Draw()

   global command
   command = TPaveText(2.5,13.4,8.0,14.5)
   command.SetTextSize(0.023)
   command.SetFillColor(39)
   command.SetBorderSize(1)
   command.AddText("Command Thread")
   command.Draw()

   global view
   view = TPavesText(1.0,9.5,7.7,12.6,3,"tr")
   view.SetFillColor(39)
   view.SetBorderSize(2)
   view.SetTextSize(0.023)
   view.AddText("Viewer Thread(s)")
   view.AddText("Picking")
   view.AddText("Context Menus")
   view.AddText("Inspector/Browser")
   view.Draw()
   

   global web
   web = TPavesText(0.5,5,6,8.5,5,"tr")
   web.SetTextSize(0.023)
   web.AddText(" ")
   web.AddText("generated")
   web.AddText("automatically")
   web.AddText("from dictionary")
   web.AddText("and source files")
   web.Draw()

   global webtitle
   webtitle = TPaveLabel(1.5,8.1,5.0,8.8,"HTML Files")
   webtitle.SetFillColor(28)
   webtitle.Draw()
   

   global printed
   printed = TPavesText(0.5,1.0,6,4,5,"tr")
   printed.SetTextSize(0.023)
   printed.AddText(" ")
   printed.AddText("generated")
   printed.AddText("automatically")
   printed.AddText("from HTML files")
   printed.Draw()

   global printedtitle
   printedtitle = TPaveLabel(1.5,3.6,5.0,4.3,"Printed Docs")
   printedtitle.SetFillColor(28)
   printedtitle.Draw()
   

   global box1
   box1 = TBox(0.2,9.2,14.25,17.8)
   box1.SetFillStyle(0)
   box1.SetLineStyle(2)
   box1.Draw()
   

   global box2
   box2 = TBox(10.2,18.7,20.2,23.6)
   box2.SetFillStyle(0)
   box2.SetLineStyle(3)
   box2.Draw()
   
   #Note:
   #      TArrow() is used as a container of many arrows.
   #      Each time we in invoke .DrawArrow, TArrow stores
   #      an arrow and draws on the default canvas.
   #      It would be useful to have names for each one.
   #      However, for sakes of simplicity we just use one

   global ar2
   ar2 = TArrow()
   ar2.SetLineWidth(6)
   ar2.SetLineColor(1)
   ar2.SetFillStyle(1001)
   ar2.SetFillColor(1)
   ar2.DrawArrow(2.5,17.5,2.5,18.9,0.012,"|>")
   ar2.DrawArrow(5.5,9.2,5.5,8.7,0.012,"|>")
   ar2.DrawArrow(5.5,5,5.5,4.2,0.012,"|>")
   ar2.DrawArrow(8.5,9.2,8.5,8.2,0.012,"|>")
   ar2.DrawArrow(9.5,8.1,9.5,9.0,0.012,"|>")
   ar2.DrawArrow(6.5,19,6.5,17.6,0.012,"|>")
   ar2.DrawArrow(8.5,19,8.5,17.1,0.012,"|>")
   ar2.DrawArrow(11.5,19,11.5,17.1,0.012,"|>")
   
   

   global ootitle
   ootitle = TPaveLabel(10.5,7.8,17,8.8,"Objects Data Base")
   ootitle.SetFillColor(28)
   ootitle.Draw()
   

   global pio
   pio = TPad("pio","pio",0.37,0.02,0.95,0.31,49)
   pio.Range(0,0,12,8)
   pio.Draw()
   pio.cd()

   global raw
   raw = TPavesText(0.5,1,2.5,6,7,"tr")
   raw.Draw()

   global dst1
   dst1 = TPavesText(4,1,5,3,7,"tr")
   dst1.Draw()

   global dst2
   dst2 = TPavesText(6,1,7,3,7,"tr")
   dst2.Draw()

   global dst3
   dst3 = TPavesText(4,4,5,6,7,"tr")
   dst3.Draw()

   global dst4
   dst4 = TPavesText(6,4,7,6,7,"tr")
   dst4.Draw()



   xlow = 8.5
   ylow = 1
   dx = 0.5
   dy = 0.5

   global analysis_list, analysis_i
   analysis_list = []
   #for (Int_t j=1; j<9; j++) {
   for j in range(1, 9, 1):
      y0 = ylow + (j-1)*0.7
      y1 = y0 + dy
      #for (Int_t i=1; i<5; i++) {
      for i in range(1, 5, 1):
         x0 = xlow +(i-1)*0.6
         x1 = x0 + dx

         analysis_i = TPavesText(x0,y0,x1,y1,7,"tr")
         analysis_i.Draw()
         analysis_list.append( analysis_i )
         
      
   
   # Just a temporary object; it is used for common attributes.
   global daq
   daq = TText()
   daq.SetTextSize(0.07)
   daq.SetTextAlign(22)
   daq.DrawText(1.5,7.3,"DAQ")
   daq.DrawText(6,7.3,"DST")
   daq.DrawText(10.,7.3,"Physics Analysis")
   daq.DrawText(1.5,0.7,"Events")
   daq.DrawText(1.5,0.3,"Containers")
   daq.DrawText(6,0.7,"Tracks/Hits")
   daq.DrawText(6,0.3,"Containers")
   daq.DrawText(10.,0.7,"Attributes")
   daq.DrawText(10.,0.3,"Containers")
   
   c1.cd()
   


if __name__ == "__main__":
   archi()
