require(ROOT)
ROOT::LoadModule('Hist')
ROOT::LoadModule('Graf')

c1    <- new(TCanvas,'c1','dilog',1)
dilog <- new(TF1,'dilog','TMath::DiLog(x)',0,0)
dilog$SetRange(0,2*pi)
dilog$Draw('') #plotting with ROOT's graphics system
c1$Update()

x<-seq(0,2*pi,by=.1)
gamma<- new(TF1,'gamma','TMath::Gamma(x)',0,2*pi)
plot(x,gamma$Eval(x)) #plotting with R's graphics system
