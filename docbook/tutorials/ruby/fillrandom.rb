# ruby-root testsuite
#
# Original $ROOT/tutorials/fillrandom.C using ruby-root.
#
# 15/12/2003    --elathan


c1 = TCanvas.new("c1","The FillRandom example",200,10,700,900)
c1.SetFillColor(18)

pad1 = TPad.new("pad1","The pad with the function" ,0.05,0.50, 0.95,0.95,21)
pad2 = TPad.new("pad2","The pad with the histogram", 0.05,0.05,0.95,0.45,21)
pad1.Draw
pad2.Draw
pad1.cd 0

form1 = TFormula.new("form1","abs(sin(x)/x)")
sqroot = TF1.new("sqroot","x*gaus(0) + [3]*form1",0,10)
sqroot.SetParameters [10,4,1,20]
pad1.SetGridx
pad1.SetGridy
   
pad1.GetFrame.SetFillColor(42)
pad1.GetFrame.SetBorderMode(-1)
pad1.GetFrame.SetBorderSize(5)
   
sqroot.SetLineColor(4)
sqroot.SetLineWidth(6)
sqroot.Draw
   
lfunction = TPaveLabel.new(5,39,9.8,46,"The sqroot function", "br")
lfunction.SetFillColor(41)
lfunction.Draw

c1.Update

# Create a one dimensional histogram (one float per bin)
# and fill it following the distribution in function sqroot.
   
pad2.cd(0)
pad2.GetFrame.SetFillColor(42)
pad2.GetFrame.SetBorderMode(-1)
pad2.GetFrame.SetBorderSize(5)
h1f = TH1F.new("h1f","Test random numbers",200,0,10)
h1f.SetFillColor(45)
h1f.FillRandom("sqroot",10000)
h1f.Draw
c1.Update
    
gApplication.Run
