require(ROOT)
gApplication$ProcessLine('#include<Math/SpecFuncMathMore.h>')#loading MathMore to use Special Functions in TRF1 (required for airy)

c1    <- TCanvas('c1')
dilog <- TF1('dilog','TMath::DiLog(x)')
dilog$Draw()

c2     <- TCanvas('c2')
lgamma <- TF1('lgamma','ROOT::Math::lgamma(x)')
lgamma$Draw()
# c3     <- TRCanvas('c2')
# airy   <- TRF1('lgamma','ROOT::Math::airy_Ai(x)')
# airy$Draw()