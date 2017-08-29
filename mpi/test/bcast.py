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


def bcast():
    
    #testing python dataypes
    msgint=0
    msgfloat=0
    msgstr=""
    msglist=[]
    msgtuple=()
    msgdict={}
    if COMM_WORLD.IsMainProcess():#filling the matrix in main rank
        msgint=123         #int
        msgfloat=0.11      #float
        msgstr="hi"        #string
        msglist=[1,2,3]    #list
        msgtuple=(3,2,1)   #tuple
        msgdict={"key":0.01}#dict

    msg=COMM_WORLD.Bcast(msgint,0)
    print(msg)
    assert(msg==123)
    msg=COMM_WORLD.Bcast(msgfloat,0)
    print(msg)
    assert(msg==0.11)
    msg=COMM_WORLD.Bcast(msgstr,0)
    print(msg)
    assert(msg=="hi")
    msg=COMM_WORLD.Bcast(msglist,0)
    print(msg)
    assert(msg==[1,2,3])
    msg=COMM_WORLD.Bcast(msgtuple,0)
    print(msg)
    assert(msg==(3,2,1))
    msg=COMM_WORLD.Bcast(msgdict,0)
    print(msg)
    assert(msg=={"key":0.01})

    ##testing ROOT dataypes
    mat=TMatrixD(2,2)
    tstr=TString()
    if COMM_WORLD.IsMainProcess():#filling the matrix in main rank
        mat[0][0]=1
        tstr="hi rootmpi py"

    recv=COMM_WORLD.Bcast(mat,0)
    assert(recv[0][0]==1)
    assert(recv.GetNrows()==2)
    assert(recv.GetNcols()==2)
    
    msgstr=COMM_WORLD.Bcast(tstr,0)
    assert(msgstr=="hi rootmpi py")
        

    ###texting mixed types
    msg=[]
    if COMM_WORLD.IsMainProcess():#filling the matrix in main rank
        msg=[Matrix(1.0),1,"Hola"]
    
    msg=COMM_WORLD.Bcast(msg,0)
    print(msg)
    assert(msg[0].matrix[0][0]==1)
    assert(msg[0].matrix.GetNrows()==1)
    assert(msg[0].matrix.GetNcols()==1)
    assert(msg[0].value==1)
    assert(msg[1]==1)
    assert(msg[2]=="Hola")
    
    
    ##nonblocking send/recv ibcast in any order
    msg1=COMM_WORLD.IBcast(123,0)
    msg2=COMM_WORLD.IBcast(321,0)
    msg3=COMM_WORLD.IBcast(0,0)
    
    msg1.Wait()
    assert(msg1.GetMsg()==123)

    msg3.Wait()
    assert(msg3.GetMsg()==0)

    msg2.Wait()
    assert(msg2.GetMsg()==321)

        
if __name__ == "__main__":
    bcast()
