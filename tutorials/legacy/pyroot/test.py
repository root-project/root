## \file
## \ingroup tutorial_pyroot_legacy
## \notebook -nodraw
##
## \macro_code
##
## \author Wim Lavrijsen

from cmath import exp

from cmath import tanh

class test:
   def value(self,index,in0,in1,in2):
      self.input0 = (in0 - 0.459987)/0.0509152
      self.input1 = (in1 - 0.188581)/0.0656804
      self.input2 = (in2 - 134.719)/16.5033
      if index==0: return ((self.neuron0xad38cb8()*1)+0);
      return 0.
   def neuron0xad37278(self):
      return self.input0
   def neuron0xad37408(self):
      return self.input1
   def neuron0xad375e0(self):
      return self.input2
   def neuron0xad378d8(self):
      input = -0.125959
      input = input + self.synapse0xad3c090()
      input = input + self.synapse0xad37a68()
      input = input + self.synapse0xad37a90()
      return ((1/(1+exp(-input)))*1)+0
   def neuron0xad37ab8(self):
      input = -1.28521
      input = input + self.synapse0xad37c90()
      input = input + self.synapse0xad37cb8()
      input = input + self.synapse0xad37ce0()
      return ((1/(1+exp(-input)))*1)+0
   def neuron0xad37d08(self):
      input = 0.869267
      input = input + self.synapse0xad37ee0()
      input = input + self.synapse0xad37f08()
      input = input + self.synapse0xad37f30()
      return ((1/(1+exp(-input)))*1)+0
   def neuron0xad37f58(self):
      input = -0.244184
      input = input + self.synapse0xad38150()
      input = input + self.synapse0xad38178()
      input = input + self.synapse0xad381a0()
      return ((1/(1+exp(-input)))*1)+0
   def neuron0xad381c8(self):
      input = 0.264279
      input = input + self.synapse0xad383c0()
      input = input + self.synapse0xad383e8()
      input = input + self.synapse0xad38410()
      return ((1/(1+exp(-input)))*1)+0
   def neuron0xad38460(self):
      input = -0.460612
      input = input + self.synapse0xad38438()
      input = input + self.synapse0xad38658()
      input = input + self.synapse0xad38708()
      input = input + self.synapse0xad38730()
      input = input + self.synapse0xad38758()
      return ((1/(1+exp(-input)))*1)+0
   def neuron0xad38780(self):
      input = 1.34716
      input = input + self.synapse0xad38930()
      input = input + self.synapse0xad38958()
      input = input + self.synapse0xad38980()
      input = input + self.synapse0xad389a8()
      input = input + self.synapse0xad389d0()
      return ((1/(1+exp(-input)))*1)+0
   def neuron0xad389f8(self):
      input = -0.550286
      input = input + self.synapse0xad38bf0()
      input = input + self.synapse0xad38c18()
      input = input + self.synapse0xad38c40()
      input = input + self.synapse0xad38c68()
      input = input + self.synapse0xad38c90()
      return ((1/(1+exp(-input)))*1)+0
   def neuron0xad38cb8(self):
      input = -0.960531
      input = input + self.synapse0xad38eb0()
      input = input + self.synapse0xad38ed8()
      input = input + self.synapse0xad38f00()
      return (input*1)+0
   def synapse0xad3c090(self):
      return (self.neuron0xad37278()*1.6232)
   def synapse0xad37a68(self):
      return (self.neuron0xad37408()*-1.0136)
   def synapse0xad37a90(self):
      return (self.neuron0xad375e0()*-0.539065)
   def synapse0xad37c90(self):
      return (self.neuron0xad37278()*0.264157)
   def synapse0xad37cb8(self):
      return (self.neuron0xad37408()*-2.25461)
   def synapse0xad37ce0(self):
      return (self.neuron0xad375e0()*1.94399)
   def synapse0xad37ee0(self):
      return (self.neuron0xad37278()*0.0608883)
   def synapse0xad37f08(self):
      return (self.neuron0xad37408()*-0.0482581)
   def synapse0xad37f30(self):
      return (self.neuron0xad375e0()*-1.07958)
   def synapse0xad38150(self):
      return (self.neuron0xad37278()*-0.0811102)
   def synapse0xad38178(self):
      return (self.neuron0xad37408()*-2.58659)
   def synapse0xad381a0(self):
      return (self.neuron0xad375e0()*1.56508)
   def synapse0xad383c0(self):
      return (self.neuron0xad37278()*-0.770005)
   def synapse0xad383e8(self):
      return (self.neuron0xad37408()*0.388095)
   def synapse0xad38410(self):
      return (self.neuron0xad375e0()*0.619588)
   def synapse0xad38438(self):
      return (self.neuron0xad378d8()*0.695049)
   def synapse0xad38658(self):
      return (self.neuron0xad37ab8()*0.400539)
   def synapse0xad38708(self):
      return (self.neuron0xad37d08()*0.631674)
   def synapse0xad38730(self):
      return (self.neuron0xad37f58()*-0.66193)
   def synapse0xad38758(self):
      return (self.neuron0xad381c8()*1.00913)
   def synapse0xad38930(self):
      return (self.neuron0xad378d8()*-0.182205)
   def synapse0xad38958(self):
      return (self.neuron0xad37ab8()*-0.920062)
   def synapse0xad38980(self):
      return (self.neuron0xad37d08()*-0.464498)
   def synapse0xad389a8(self):
      return (self.neuron0xad37f58()*-0.222692)
   def synapse0xad389d0(self):
      return (self.neuron0xad381c8()*-0.546376)
   def synapse0xad38bf0(self):
      return (self.neuron0xad378d8()*-0.446571)
   def synapse0xad38c18(self):
      return (self.neuron0xad37ab8()*2.22403)
   def synapse0xad38c40(self):
      return (self.neuron0xad37d08()*1.02097)
   def synapse0xad38c68(self):
      return (self.neuron0xad37f58()*-2.35373)
   def synapse0xad38c90(self):
      return (self.neuron0xad381c8()*4.67134)
   def synapse0xad38eb0(self):
      return (self.neuron0xad38460()*4.18467)
   def synapse0xad38ed8(self):
      return (self.neuron0xad38780()*-4.31992)
   def synapse0xad38f00(self):
      return (self.neuron0xad389f8()*0.916293)
