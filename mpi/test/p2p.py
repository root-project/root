from ROOT import TMatrixD, gROOT, TString, Mpi
from ROOT.Mpi import COMM_WORLD, TEnvironment


env  = Mpi.TEnvironment()        

class Matrix:
    def __init__(self, value):
        self.matrix = TMatrixD(1,1)
        self.matrix[0][0]=value
        self.value = value
    def Print(self):
        self.matrix.Print()


def p2p():
    
    #testing python dataypes
    if COMM_WORLD.IsMainProcess():#filling the matrix in main rank
        COMM_WORLD.Send(1,1,0)       #int
        COMM_WORLD.Send(1.1,1,0)     #float
        COMM_WORLD.Send("hi",1,0)    #string
        COMM_WORLD.Send([0,1],1,0)   #list
        COMM_WORLD.Send((1,0),1,0)   #tuple
        COMM_WORLD.Send({"key":0.12},1,0)   #dict
    else:
        msg=COMM_WORLD.Recv(0,0)
        print(msg)
        assert(msg==1)
        msg=COMM_WORLD.Recv(0,0)
        print(msg)
        assert(msg==1.1)
        msg=COMM_WORLD.Recv(0,0)
        print(msg)
        assert(msg=="hi")
        msg=COMM_WORLD.Recv(0,0)
        print(msg)
        assert(msg==[0,1])
        msg=COMM_WORLD.Recv(0,0)
        print(msg)
        assert(msg==(1,0))
        msg=COMM_WORLD.Recv(0,0)
        print(msg)
        assert(msg=={"key":0.12})
        

    #testing ROOT dataypes
    if COMM_WORLD.IsMainProcess():#filling the matrix in main rank
        mat=TMatrixD(2,2)
        mat[0][0]=1
        COMM_WORLD.Send(mat,1,0)
        COMM_WORLD.Send(TString("hi rootmpi py"),1,0)
    else:
        mat=COMM_WORLD.Recv(0,0)
        assert(mat[0][0]==1)
        assert(mat.GetNrows()==2)
        assert(mat.GetNcols()==2)
        msgstr=COMM_WORLD.Recv(0,0)
        assert(msgstr=="hi rootmpi py")
        

    ###texting mixed types
    if COMM_WORLD.IsMainProcess():#filling the matrix in main rank
        msg=[Matrix(1.0),1,"Hola"]
        COMM_WORLD.Send(msg,1,0)
    else:
        msg=COMM_WORLD.Recv(0,0)
        print(msg)
        assert(msg[0].matrix[0][0]==1)
        assert(msg[0].matrix.GetNrows()==1)
        assert(msg[0].matrix.GetNcols()==1)
        assert(msg[0].value==1)
        assert(msg[1]==1)
        assert(msg[2]=="Hola")
        
if __name__ == "__main__":
    p2p()
