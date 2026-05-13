# \file
# \ingroup tutorial_geom
# Macro that demonstrates usage of radioactive elements/materials/mixtures
# with TGeo package.
# 
# How to run: `%run RadioNuclides.py` in ipython3 interpreter, it'll show you 
# two graphics "C14 Decay" and "Mixture Decay". 
#
# Information about the ROOT-Classes used in this macro.
# A radionuclide (TGeoElementRN) derives from the class TGeoElement and
# provides additional information related to its radioactive properties and
# decay modes.
#
# The radionuclides table is loaded on demand by any call on ROOT:
# ~~~{.py}
#    myElement = ROOT.TGElementTable.GetElemntRN( atomic_numbe = int,
#                                                 atomic_charge = int,                   
#                                                 isomeric_number = int)                  
# ~~~
# The `isomeric number` is optional and the default value is 0.
#
# To create a radioactive material based on a radionuclide, one should use the
# constructor:
# ~~~{.py}
#    myMaterial = TGeoMaterial(name = str, elem = TGeoElement, density = Double_t )
# ~~~
#
# To create a radioactive mixture, one can use radionuclides as well as stable
# elements:
# ~~~{.py}
#    myMixture = TGeoMixture(name = str, nelements = Int_t, density = Double_t)
#    myMixture.AddElement(elem = TGeoElement, weight_fraction = Double_t)
# ~~~
#
# Once defined a TGeoMaterial object, one can retrieve the time evolution of its products
# (radioactive materials/mixtures) by using:
# ~~~{.py}
#    myElement.FillMaterialEvolution(population = TObjArray,
#                                    precision=0.001)
# ~~~
# To use this method `FillMaterial`, one has to provide an empty TObjArray object(`population`) that will
# be filled with all elements coming from the decay chain of the initial
# radionuclides contained in the material/mixture. The `precision` argument 
# represents the cumulative branching ratio for which decay products are still considered.
# The `population` list may contain stable elements as well as radionuclides,
# depending on the initial elements. To test if an element is a radionuclide you
# can you use `IsRadioNuclide` method:
#
# ~~~{.py}
#    myMaterial.IsRadioNuclide() # bool
# ~~~
#
# All radionuclides, in the output population list, have attached "objects" to themselves;
# each of them represent the time evolution of their nuclei fraction(with respect to the
# top radionuclide in the decay chain). These objects (Bateman solutions) can be
# retrieved and drawn using TGeoBateManSol, as in:
#
# ~~~{.py}
#    myBatemanSol = TGeoElementRN.Ratio() # TGeoBatemanSoli()
#    myBatemanSol.Draw()
# ~~~
#
# Another method allows to create the evolution of a given radioactive
# material/mixture at a given moment in time:
#
# ~~~{.py}
#    TGeoMaterial.DecayMaterial(time = Double_t, precision=0.001)
# ~~~
#
# The above method `DecayMaterial` will create a "mixture" whichs results from the decay of an initial
# material/mixture after passed some period `time`; while all resulting elements, having 
# a fractional weight less than the `precision` argument, will be excluded.
#
# \macro_image
# \macro_code
#
# \author Mihaela Gheata
# \translator P. P.

import ROOT

TGeoManager = 		 ROOT.TGeoManager
gGeoManager = 		 ROOT.gGeoManager
TGeoMaterial = 		 ROOT.TGeoMaterial
TGeoMixture = 		 ROOT.TGeoMixture
TObjArray = 		 ROOT.TObjArray
TCanvas = ROOT.TCanvas 
Bool_t = ROOT.Bool_t
Double_t = ROOT.Double_t
TGeoElementRN = ROOT.TGeoElementRN
TGeoBatemanSol = ROOT.TGeoBatemanSol
Form =ROOT.Form
strcmp = ROOT.strcmp
TLatex = ROOT.TLatex
TPaveText = ROOT.TPaveText
kTRUE = ROOT.kTRUE
TArrow = ROOT.TArrow

Declare = 		 ROOT.gInterpreter.Declare 
ProcessLine = 		 ROOT.gInterpreter.ProcessLine 
gGeoManager = ROOT.gGeoManager

def DrawPopulation(
                   vect = TObjArray(),
                   can = TCanvas(False),
                   tmin = Double_t(),
                   tmax = Double_t(),
                   logx = Bool_t()   
   ):

   n = vect.GetEntriesFast()
   elem = TGeoElementRN()
   sol = TGeoBatemanSol()
   can.SetLogy()
   
   if (logx) :
      can.SetLogx()
   
   
   for i in range(n):
      el = vect.At(i)
      if not el.IsRadioNuclide(): 
         continue
      elem = el # TGeoElementRN
      sol = elem.Ratio() # TGeoBateManSol
      if sol:
         sol.SetLineColor(1+(i%9))
         sol.SetLineWidth(2)
         if tmax > 0.:
            sol.SetRange(tmin,tmax)
         if i==0:
            sol.Draw()
            # fun is TF1
            func = can.FindObject(
               Form("conc{:s}".format( sol.GetElement().GetName() ) )
               )
            if func:
               if not strcmp(can.GetName(),"c1"): 
                  func.SetTitle(
                     "Concentration of C14 derived elements;time[s];Ni/N0(C14)")
               else:
                  func.SetTitle(
                     "Concentration of elements derived from mixture Ca53+Sr78;\
                     time[s];Ni/N0(Ca53)")
               
            
         else: # indentation: if i==0
            sol.Draw("SAME")
            pass
   can.Draw()
   can.Update()     
      

#def RadioNuclides():
class RadioNuclides:
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager

   #geom =  TGeoManager("","")
   ProcessLine('''
   TGeoManager *geom = new TGeoManager("","");
   ''')
   global geom
   geom = ROOT.geom

   table = gGeoManager.GetElementTable()
   c14 = table.GetElementRN(14,6)
   el1 = table.GetElementRN(53,20)
   el2 = table.GetElementRN(78,38)
   # Radioactive material
   mat =  TGeoMaterial("C14", c14, 1.3)
   print("___________________________________________________________\n")
   print("Radioactive material:\n")
   mat.Print()
   time = 1.5e11 # seconds
   decaymat = mat.DecayMaterial(time)
   print("Radioactive material evolution after %g years:\n", time/3.1536e7)
   decaymat.Print()
   
   # Radioactive mixture
   mix =  TGeoMixture("mix", 2, 7.3)
   mix.AddElement(el1, 0.35)
   mix.AddElement(el2, 0.65)
   print("___________________________________________________________\n")
   print("Radioactive mixture:\n")
   mix.Print()
   time = 1000.
   decaymat = mix.DecayMaterial(time)
   print("Radioactive mixture evolution after {} seconds:\n".format(time) )
   decaymat.Print()
   vect = TObjArray() # TObjArray
   global c1 # Canvas keeps on screen after closing function.
   c1 = TCanvas("c1","C14 decay", 800,600) # TCanvas
   c1.SetGrid()
   mat.FillMaterialEvolution(vect)
   DrawPopulation(vect, c1, 0, 1.4e12)
   tex = TLatex(8.35e11,0.564871,"C_{N^{14}_{7}}") # TLatex
   tex.SetTextSize(0.0388601)
   tex.SetLineWidth(2)
   tex.Draw()
   tex = TLatex(3.33e11,0.0620678,"C_{C^{14}_{6}}")
   tex.SetTextSize(0.0388601)
   tex.SetLineWidth(2)
   tex.Draw()
   tex = TLatex(9.4e11,0.098,"C_{X}=#frac{N_{X}(t)}{N_{0}(t=0)}=\
   #sum_{j}#alpha_{j}e^{-#lambda_{j}t}")
   tex.SetTextSize(0.0388601)
   tex.SetLineWidth(2)
   tex.Draw()
   pt = TPaveText(2.6903e+11,0.0042727,1.11791e+12,0.0325138,"br") # TPaveText
   pt.SetFillColor(5)
   pt.SetTextAlign(12)
   pt.SetTextColor(4)
   pt.AddText("Time evolution of a population of radionuclides.")
   pt.AddText("The concentration of a nuclide X represent the  ")
   pt.AddText("ratio between the number of X nuclei and the    ")
   pt.AddText("number of nuclei of the top element of the decay")
   pt.AddText("from which X derives from at T=0.               ")
   pt.Draw()
   c1.Modified()
   vect.Clear()
   global c2 # Canvas keeps on screen after closing function.
   c2 = TCanvas("c2","Mixture decay", 1000,800) # TCanvas
   c2.SetGrid()
   mix.FillMaterialEvolution(vect)
   DrawPopulation(vect, c2, 0.01, 1000., kTRUE)
   tex = TLatex(0.019,0.861,"C_{Ca^{53}_{20}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(1)
   tex.Draw()
   tex = TLatex(0.0311,0.078064,"C_{Sc^{52}_{21}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(2)
   tex.Draw()
   tex = TLatex(0.1337,0.010208,"C_{Ti^{52}_{22}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(3)
   tex.Draw()
   tex = TLatex(1.54158,0.00229644,"C_{V^{52}_{23}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(4)
   tex.Draw()
   tex = TLatex(25.0522,0.00135315,"C_{Cr^{52}_{24}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(5)
   tex.Draw()
   tex = TLatex(0.1056,0.5429,"C_{Sc^{53}_{21}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(6)
   tex.Draw()
   tex = TLatex(0.411,0.1044,"C_{Ti^{53}_{22}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(7)
   tex.Draw()
   tex = TLatex(2.93358,0.0139452,"C_{V^{53}_{23}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(8)
   tex.Draw()
   tex = TLatex(10.6235,0.00440327,"C_{Cr^{53}_{24}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(9)
   tex.Draw()
   tex = TLatex(15.6288,0.782976,"C_{Sr^{78}_{38}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(1)
   tex.Draw()
   tex = TLatex(20.2162,0.141779,"C_{Rb^{78}_{37}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(2)
   tex.Draw()
   tex = TLatex(32.4055,0.0302101,"C_{Kr^{78}_{36}}")
   tex.SetTextSize(0.0388601)
   tex.SetTextColor(3)
   tex.Draw()
   tex = TLatex(117.,1.52,"C_{X}=#frac{N_{X}(t)}{N_{0}(t=0)}=#sum_{j}\
   #alpha_{j}e^{-#lambda_{j}t}")
   tex.SetTextSize(0.03)
   tex.SetLineWidth(2)
   tex.Draw()
   arrow = TArrow(0.0235313,0.74106,0.0385371,0.115648,0.02,">") # TArrow
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(0.0543138,0.0586338,0.136594,0.0146596,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(0.31528,0.00722919,1.29852,0.00306079,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(4.13457,0.00201942,22.5047,0.00155182,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(0.0543138,0.761893,0.0928479,0.67253,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(0.238566,0.375717,0.416662,0.154727,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(0.653714,0.074215,2.41863,0.0213142,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(5.58256,0.00953882,10.6235,0.00629343,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(22.0271,0.601935,22.9926,0.218812,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   arrow = TArrow(27.2962,0.102084,36.8557,0.045686,0.02,">")
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.SetLineWidth(2)
   arrow.SetAngle(30)
   arrow.Draw()
   
   #DelROOTObjs(self) 
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   myvars = [x for x in dir() if not x.startswith("__")]
   #myvars = [x for x in vars(self) ] 
   for var in myvars: 
      try:
         exec(f"ROOT.gROOT.Remove({var})")
         #exec(f"ROOT.gROOT.Remove(self.{var})")
         #print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!


if __name__ == "__main__" :
   RadioNuclides()
