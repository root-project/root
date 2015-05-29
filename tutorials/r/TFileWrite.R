require(ROOT)
LoadModule('RIO')
LoadModule('Hist')
LoadModule('Graf')

rfile <- new(TFile,'gamma.root','recreate')
gamma <- new(TF1,'gamma','TMath::Gamma(x)',0.1,2*pi)
gamma$Write('gamma')
rfile$Flush()
rfile$Close()

