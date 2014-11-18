from math import exp

from math import tanh

class test:
	def value(self,index,in0,in1,in2):
		self.input0 = (in0 - 0.459987)/0.0509152
		self.input1 = (in1 - 0.188581)/0.0656804
		self.input2 = (in2 - 134.719)/16.5033
		if index==0: return self.neuron0x7fd144992720();
		return 0.
	def neuron0x7fd14498e050(self):
		return self.input0
	def neuron0x7fd14498e2d0(self):
		return self.input1
	def neuron0x7fd14498e5d0(self):
		return self.input2
	def neuron0x7fd144990b40(self):
		input = -2.64856
		input = input + self.synapse0x7fd1449a8700()
		input = input + self.synapse0x7fd14498df10()
		input = input + self.synapse0x7fd14493b120()
		if input<-709. : return 0
		return ((1/(1+exp(-input)))*1)+0
	def neuron0x7fd144990cc0(self):
		input = -2.64856
		input = input + self.synapse0x7fd144990fc0()
		input = input + self.synapse0x7fd144990ff0()
		input = input + self.synapse0x7fd144991020()
		if input<-709. : return 0
		return ((1/(1+exp(-input)))*1)+0
	def neuron0x7fd144991050(self):
		input = -2.64856
		input = input + self.synapse0x7fd144991350()
		input = input + self.synapse0x7fd144991380()
		input = input + self.synapse0x7fd1449913b0()
		if input<-709. : return 0
		return ((1/(1+exp(-input)))*1)+0
	def neuron0x7fd1449913e0(self):
		input = -2.64856
		input = input + self.synapse0x7fd1449916e0()
		input = input + self.synapse0x7fd144991710()
		input = input + self.synapse0x7fd144991740()
		if input<-709. : return 0
		return ((1/(1+exp(-input)))*1)+0
	def neuron0x7fd144991770(self):
		input = -2.64856
		input = input + self.synapse0x7fd144991a70()
		input = input + self.synapse0x7fd144991aa0()
		input = input + self.synapse0x7fd144991ad0()
		if input<-709. : return 0
		return ((1/(1+exp(-input)))*1)+0
	def neuron0x7fd144991b00(self):
		input = -1.46999
		input = input + self.synapse0x7fd144991e00()
		input = input + self.synapse0x7fd144991e30()
		input = input + self.synapse0x7fd14498df40()
		input = input + self.synapse0x7fd144991f60()
		input = input + self.synapse0x7fd144991f90()
		if input<-709. : return 0
		return ((1/(1+exp(-input)))*1)+0
	def neuron0x7fd144991fc0(self):
		input = -1.46999
		input = input + self.synapse0x7fd144992240()
		input = input + self.synapse0x7fd144992270()
		input = input + self.synapse0x7fd1449922a0()
		input = input + self.synapse0x7fd1449922d0()
		input = input + self.synapse0x7fd144992300()
		if input<-709. : return 0
		return ((1/(1+exp(-input)))*1)+0
	def neuron0x7fd144992330(self):
		input = -1.46999
		input = input + self.synapse0x7fd144992630()
		input = input + self.synapse0x7fd144992660()
		input = input + self.synapse0x7fd144992690()
		input = input + self.synapse0x7fd1449926c0()
		input = input + self.synapse0x7fd1449926f0()
		if input<-709. : return 0
		return ((1/(1+exp(-input)))*1)+0
	def neuron0x7fd144992720(self):
		input = -1.81984
		input = input + self.synapse0x7fd144992920()
		input = input + self.synapse0x7fd144992950()
		input = input + self.synapse0x7fd144992980()
		return (input*1)+0
	def synapse0x7fd1449a8700(self):
		return (self.neuron0x7fd14498e050()*0.0207841)
	def synapse0x7fd14498df10(self):
		return (self.neuron0x7fd14498e2d0()*-4.05985)
	def synapse0x7fd14493b120(self):
		return (self.neuron0x7fd14498e5d0()*3.01615)
	def synapse0x7fd144990fc0(self):
		return (self.neuron0x7fd14498e050()*0.0207841)
	def synapse0x7fd144990ff0(self):
		return (self.neuron0x7fd14498e2d0()*-4.05985)
	def synapse0x7fd144991020(self):
		return (self.neuron0x7fd14498e5d0()*3.01615)
	def synapse0x7fd144991350(self):
		return (self.neuron0x7fd14498e050()*0.0207841)
	def synapse0x7fd144991380(self):
		return (self.neuron0x7fd14498e2d0()*-4.05985)
	def synapse0x7fd1449913b0(self):
		return (self.neuron0x7fd14498e5d0()*3.01615)
	def synapse0x7fd1449916e0(self):
		return (self.neuron0x7fd14498e050()*0.0207841)
	def synapse0x7fd144991710(self):
		return (self.neuron0x7fd14498e2d0()*-4.05985)
	def synapse0x7fd144991740(self):
		return (self.neuron0x7fd14498e5d0()*3.01615)
	def synapse0x7fd144991a70(self):
		return (self.neuron0x7fd14498e050()*0.0207841)
	def synapse0x7fd144991aa0(self):
		return (self.neuron0x7fd14498e2d0()*-4.05985)
	def synapse0x7fd144991ad0(self):
		return (self.neuron0x7fd14498e5d0()*3.01615)
	def synapse0x7fd144991e00(self):
		return (self.neuron0x7fd144990b40()*0.085646)
	def synapse0x7fd144991e30(self):
		return (self.neuron0x7fd144990cc0()*0.085646)
	def synapse0x7fd14498df40(self):
		return (self.neuron0x7fd144991050()*0.085646)
	def synapse0x7fd144991f60(self):
		return (self.neuron0x7fd1449913e0()*0.085646)
	def synapse0x7fd144991f90(self):
		return (self.neuron0x7fd144991770()*0.085646)
	def synapse0x7fd144992240(self):
		return (self.neuron0x7fd144990b40()*0.085646)
	def synapse0x7fd144992270(self):
		return (self.neuron0x7fd144990cc0()*0.085646)
	def synapse0x7fd1449922a0(self):
		return (self.neuron0x7fd144991050()*0.085646)
	def synapse0x7fd1449922d0(self):
		return (self.neuron0x7fd1449913e0()*0.085646)
	def synapse0x7fd144992300(self):
		return (self.neuron0x7fd144991770()*0.085646)
	def synapse0x7fd144992630(self):
		return (self.neuron0x7fd144990b40()*0.085646)
	def synapse0x7fd144992660(self):
		return (self.neuron0x7fd144990cc0()*0.085646)
	def synapse0x7fd144992690(self):
		return (self.neuron0x7fd144991050()*0.085646)
	def synapse0x7fd1449926c0(self):
		return (self.neuron0x7fd1449913e0()*0.085646)
	def synapse0x7fd1449926f0(self):
		return (self.neuron0x7fd144991770()*0.085646)
	def synapse0x7fd144992920(self):
		return (self.neuron0x7fd144991b00()*3.3132)
	def synapse0x7fd144992950(self):
		return (self.neuron0x7fd144991fc0()*3.3132)
	def synapse0x7fd144992980(self):
		return (self.neuron0x7fd144992330()*3.3132)
