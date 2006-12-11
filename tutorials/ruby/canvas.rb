
# ruby-root testsuite
#
# A simple canvas demostration script.
#
# 15/12/2003    --elathan

tc = TCanvas.new("tc", "canvas example", 200, 10, 700, 500)

tc.SetFillColor(42)
tc.SetGrid

x = Array.new
y = Array.new

for i in 0..19 do
    x[i] = i*0.1
    y[i] = 10*Math.sin(x[i] + 0.2)
end

tg = TGraph.new(20, x, y)
tg.SetLineColor(2)
tg.SetLineWidth(4)
tg.SetMarkerColor(4)
tg.SetMarkerStyle(21)
tg.Draw("ACP")

tc.Update
tc.GetFrame.SetFillColor(21)
tc.GetFrame.SetBorderSize(12)
tc.Modified
gApplication.Run
