
# ruby-root testsuite
# port of the original $ROOT/multigraph.C tutorial
# (02/02/2004)  --elathan  <elathan@phys.uoa.gr>

gStyle.SetOptFit

c1 = TCanvas.new("c1", "multigraph", 200, 10, 700, 500)
    c1.SetGrid

# draw a frame to define the range
mg = TMultiGraph.new
   
    # create first graph
    n1 = 10
    x1 = [-0.1, 0.05, 0.25, 0.35, 0.5, 0.61,0.7,0.85,0.89,0.95]
    y1 = [-1,2.9,5.6,7.4,9,9.6,8.7,6.3,4.5,1]
    ex1 = [0.05,0.1,0.07,0.07,0.04,0.05,0.06,0.07,0.08,0.05]
    ey1 = [0.8,0.7,0.6,0.5,0.4,0.4,0.5,0.6,0.7,0.8]
    
gr1 = TGraphErrors.new n1, x1, y1, ex1, ey1
    gr1.SetMarkerColor kBlue 
    gr1.SetMarkerStyle 21
    gr1.Fit "pol6", "q"
  
mg.Add gr1 

    # create second graph
    n2 = 10
    x2 = [-0.28, 0.005, 0.19, 0.29, 0.45, 0.56,0.65,0.80,0.90,1.01]
    y2 = [2.1,3.86,7,9,10,10.55,9.64,7.26,5.42,2]
    ex2 = [0.04,0.12,0.08,0.06,0.05,0.04,0.07,0.06,0.08,0.04]
    ey2 = [0.6,0.8,0.7,0.4,0.3,0.3,0.4,0.5,0.6,0.7]
    
gr2 = TGraphErrors.new n2, x2, y2, ex2, ey2
    gr2.SetMarkerColor kRed 
    gr2.SetMarkerStyle 20 
    gr2.Fit "pol5", "q" 
   
mg.Add gr2 

mg.Draw "ap"
   
    # force drawing of canvas to generate the fit TPaveStats
    c1.Update
    
    # see how ROOT Collections have been turned up to Ruby arrays.  --elathan
    stats1 = gr1.GetListOfFunctions.find { |f| f.GetName == "stats" }.as("TPaveStats")
    stats2 = gr2.GetListOfFunctions.find { |f| f.GetName == "stats" }.as("TPaveStats")

    stats1.SetTextColor kBlue
    stats2.SetTextColor kRed 
    stats1.SetX1NDC(0.12); stats1.SetX2NDC(0.32); stats1.SetY1NDC(0.75);
    stats2.SetX1NDC(0.72); stats2.SetX2NDC(0.92); stats2.SetY1NDC(0.78);
    c1.Modified
gApplication.Run
