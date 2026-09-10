######################################/
#   This class has been translated into Python from the automatically generated class
#     (Wed Apr 19 21:47:55 2000 by ROOT version 2.24/02)
#   from TTree h42/
#   found on file: Memory Directory
#
#   The example was modfied for the new TSelector version
#   (Thu Sep 25 06:44:10 EDT 2003)
######################################/

import ROOT
import ctypes
from array import array


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   Info,
                   addressof,
                   TROOT,
                   TChain,
                   TEntryList,
                   TFile,
                   TSelector,
                   TTree,
                   TBranch,
                   TObject,
                   TList,
                   TH1F,
                   TH2F,
                   TCanvas,
                   TH1F,
                   TGraph,
                   TLatex,
)

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
)

# types
from ROOT import (
                   Double_t,
                   Bool_t,
                   Float_t,
                   Int_t,
                   nullptr,
                   UChar_t,
                   char,
                   )
#
from ctypes import (
                     c_double,
                     c_longlong,
)
#
Long64_t = c_longlong


# utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args, end="")
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

# constants
from ROOT import (
                   kBlue,
                   kRed,
                   kGreen,
)

# globals
from ROOT import (
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)





#class h1analysis : public TSelector {
class h1analysis( TSelector ):

   #public:
   
   hdmd = TH1F() # *
   h2 = TH2F() # *
   
   useList = Bool_t()
   fillList = Bool_t()
   elist = TEntryList() # *
   fProcessed = Long64_t()
   
   fChain = TTree() # *
   nrun = Int_t()
   nevent = Int_t()
   nentry = Int_t()
   trelem = [ UChar_t() for _ in range( 192 ) ] # [192]
   subtr = [ UChar_t() for _ in range( 128 ) ] # [128]
   rawtr = [ UChar_t() for _ in range( 128 ) ] # [128]
   L4subtr = [ UChar_t() for _ in range( 128 ) ] # [128]
   L5class = [ UChar_t() for _ in range( 32 ) ] # [32]
   E33 = Float_t()
   de33 = Float_t()
   x33 = Float_t()
   dx33 = Float_t()
   y33 = Float_t()
   dy33 = Float_t()
   E44 = Float_t()
   de44 = Float_t()
   x44 = Float_t()
   dx44 = Float_t()
   y44 = Float_t()
   dy44 = Float_t()
   Ept = Float_t()
   dept = Float_t()
   xpt = Float_t()
   dxpt = Float_t()
   ypt = Float_t()
   dypt = Float_t()
   pelec = [ Float_t() for _ in range( 4 ) ] # [4]
   flagelec = Int_t()
   xeelec = Float_t()
   yeelec = Float_t()
   Q2eelec = Float_t()
   nelec = Int_t()
   Eelec = [ Float_t() for _ in range( 20 ) ] # [20]
   thetelec = [ Float_t() for _ in range( 20 ) ] # [20]
   phielec = [ Float_t() for _ in range( 20 ) ] # [20]
   xelec = [ Float_t() for _ in range( 20 ) ] # [20]
   Q2elec = [ Float_t() for _ in range( 20 ) ] # [20]
   xsigma = [ Float_t() for _ in range( 20 ) ] # [20]
   Q2sigma = [ Float_t() for _ in range( 20 ) ] # [20]
   sumc = [ Float_t() for _ in range( 4 ) ] # [4]
   sumetc = Float_t()
   yjbc = Float_t()
   Q2jbc = Float_t()
   sumct = [ Float_t() for _ in range( 4 ) ] # [4]
   sumetct = Float_t()
   yjbct = Float_t()
   Q2jbct = Float_t()
   Ebeamel = Float_t()
   Ebeampr = Float_t()
   pvtx_d = [ Float_t() for _ in range( 3 ) ] # [3]
   cpvtx_d = [ Float_t() for _ in range( 6 ) ] # [6]
   pvtx_t = [ Float_t() for _ in range( 3 ) ] # [3]
   cpvtx_t = [ Float_t() for _ in range( 6 ) ] # [6]
   ntrkxy_t = Int_t()
   prbxy_t = Float_t()
   ntrkz_t = Int_t()
   prbz_t = Float_t()
   nds = Int_t()
   rankds = Int_t()
   qds = Int_t()
   pds_d = [ Float_t() for _ in range( 4 ) ] # [4]
   ptds_d = Float_t()
   etads_d = Float_t()
   dm_d = Float_t()
   ddm_d = Float_t()
   pds_t = [ Float_t() for _ in range( 4 ) ] # [4]
   dm_t = Float_t()
   ddm_t = Float_t()
   ik = Int_t()
   ipi = Int_t()
   ipis = Int_t()
   pd0_d = [ Float_t() for _ in range( 4 ) ] # [4]
   ptd0_d = Float_t()
   etad0_d = Float_t()
   md0_d = Float_t()
   dmd0_d = Float_t()
   pd0_t = [ Float_t() for _ in range( 4 ) ] # [4]
   md0_t = Float_t()
   dmd0_t = Float_t()
   pk_r = [ Float_t() for _ in range( 4 ) ] # [4]
   ppi_r = [ Float_t() for _ in range( 4 ) ] # [4]
   pd0_r = [ Float_t() for _ in range( 4 ) ] # [4]
   md0_r = Float_t()
   Vtxd0_r = [ Float_t() for _ in range( 3 ) ] # [3]
   cvtxd0_r = [ Float_t() for _ in range( 6 ) ] # [6]
   dxy_r = Float_t()
   dz_r = Float_t()
   psi_r = Float_t()
   rd0_d = Float_t()
   drd0_d = Float_t()
   rpd0_d = Float_t()
   drpd0_d = Float_t()
   rd0_t = Float_t()
   drd0_t = Float_t()
   rpd0_t = Float_t()
   drpd0_t = Float_t()
   rd0_dt = Float_t()
   drd0_dt = Float_t()
   prbr_dt = Float_t()
   prbz_dt = Float_t()
   rd0_tt = Float_t()
   drd0_tt = Float_t()
   prbr_tt = Float_t()
   prbz_tt = Float_t()
   ijetd0 = Int_t()
   ptr3d0_j = Float_t()
   ptr2d0_j = Float_t()
   ptr3d0_3 = Float_t()
   ptr2d0_3 = Float_t()
   ptr2d0_2 = Float_t()
   Mimpds_r = Float_t()
   Mimpbk_r = Float_t()
   ntracks = Int_t()
   pt = [ Float_t() for _ in range( 200 ) ] # [200]
   kappa = [ Float_t() for _ in range( 200 ) ] # [200]
   phi = [ Float_t() for _ in range( 200 ) ] # [200]
   theta = [ Float_t() for _ in range( 200 ) ] # [200]
   dca = [ Float_t() for _ in range( 200 ) ] # [200]
   z0 = [ Float_t() for _ in range( 200 ) ] # [200]
   #Float_t         covar[200][15];#!
   covar = [ [ Float_t() for _ in range(15) ] for _ in range( 15 ) ] # [200][15]
   nhitrp = [ Int_t() for _ in range( 200 ) ] # [200]
   prbrp = [ Float_t() for _ in range( 200 ) ] # [200]
   nhitz = [ Int_t() for _ in range( 200 ) ] # [200]
   prbz = [ Float_t() for _ in range( 200 ) ] # [200]
   rstart = [ Float_t() for _ in range( 200 ) ] # [200]
   rend = [ Float_t() for _ in range( 200 ) ] # [200]
   lhk = [ Float_t() for _ in range( 200 ) ] # [200]
   lhpi = [ Float_t() for _ in range( 200 ) ] # [200]
   nlhk = [ Float_t() for _ in range( 200 ) ] # [200]
   nlhpi = [ Float_t() for _ in range( 200 ) ] # [200]
   dca_d = [ Float_t() for _ in range( 200 ) ] # [200]
   ddca_d = [ Float_t() for _ in range( 200 ) ] # [200]
   dca_t = [ Float_t() for _ in range( 200 ) ] # [200]
   ddca_t = [ Float_t() for _ in range( 200 ) ] # [200]
   muqual = [ Int_t() for _ in range( 200 ) ] # [200]
   imu = Int_t()
   imufe = Int_t()
   njets = Int_t()
   E_j = [ Float_t() for _ in range( 20 ) ] # [20]
   pt_j = [ Float_t() for _ in range( 20 ) ] # [20]
   theta_j = [ Float_t() for _ in range( 20 ) ] # [20]
   eta_j = [ Float_t() for _ in range( 20 ) ] # [20]
   phi_j = [ Float_t() for _ in range( 20 ) ] # [20]
   m_j = [ Float_t() for _ in range( 20 ) ] # [20]
   thrust = Float_t()
   pthrust = [ Float_t() for _ in range( 4 ) ] # [4]
   thrust2 = Float_t()
   pthrust2 = [ Float_t() for _ in range( 4 ) ] # [4]
   spher = Float_t()
   aplan = Float_t()
   plan = Float_t()
   nnout = [ Float_t() for _ in range( 1 ) ] # [1]
   
   
   b_nrun = TBranch() # *
   b_nevent = TBranch() # *
   b_nentry = TBranch() # *
   b_trelem = TBranch() # *
   b_subtr = TBranch() # *
   b_rawtr = TBranch() # *
   b_L4subtr = TBranch() # *
   b_L5class = TBranch() # *
   b_E33 = TBranch() # *
   b_de33 = TBranch() # *
   b_x33 = TBranch() # *
   b_dx33 = TBranch() # *
   b_y33 = TBranch() # *
   b_dy33 = TBranch() # *
   b_E44 = TBranch() # *
   b_de44 = TBranch() # *
   b_x44 = TBranch() # *
   b_dx44 = TBranch() # *
   b_y44 = TBranch() # *
   b_dy44 = TBranch() # *
   b_Ept = TBranch() # *
   b_dept = TBranch() # *
   b_xpt = TBranch() # *
   b_dxpt = TBranch() # *
   b_ypt = TBranch() # *
   b_dypt = TBranch() # *
   b_pelec = TBranch() # *
   b_flagelec = TBranch() # *
   b_xeelec = TBranch() # *
   b_yeelec = TBranch() # *
   b_Q2eelec = TBranch() # *
   b_nelec = TBranch() # *
   b_Eelec = TBranch() # *
   b_thetelec = TBranch() # *
   b_phielec = TBranch() # *
   b_xelec = TBranch() # *
   b_Q2elec = TBranch() # *
   b_xsigma = TBranch() # *
   b_Q2sigma = TBranch() # *
   b_sumc = TBranch() # *
   b_sumetc = TBranch() # *
   b_yjbc = TBranch() # *
   b_Q2jbc = TBranch() # *
   b_sumct = TBranch() # *
   b_sumetct = TBranch() # *
   b_yjbct = TBranch() # *
   b_Q2jbct = TBranch() # *
   b_Ebeamel = TBranch() # *
   b_Ebeampr = TBranch() # *
   b_pvtx_d = TBranch() # *
   b_cpvtx_d = TBranch() # *
   b_pvtx_t = TBranch() # *
   b_cpvtx_t = TBranch() # *
   b_ntrkxy_t = TBranch() # *
   b_prbxy_t = TBranch() # *
   b_ntrkz_t = TBranch() # *
   b_prbz_t = TBranch() # *
   b_nds = TBranch() # *
   b_rankds = TBranch() # *
   b_qds = TBranch() # *
   b_pds_d = TBranch() # *
   b_ptds_d = TBranch() # *
   b_etads_d = TBranch() # *
   b_dm_d = TBranch() # *
   b_ddm_d = TBranch() # *
   b_pds_t = TBranch() # *
   b_dm_t = TBranch() # *
   b_ddm_t = TBranch() # *
   b_ik = TBranch() # *
   b_ipi = TBranch() # *
   b_ipis = TBranch() # *
   b_pd0_d = TBranch() # *
   b_ptd0_d = TBranch() # *
   b_etad0_d = TBranch() # *
   b_md0_d = TBranch() # *
   b_dmd0_d = TBranch() # *
   b_pd0_t = TBranch() # *
   b_md0_t = TBranch() # *
   b_dmd0_t = TBranch() # *
   b_pk_r = TBranch() # *
   b_ppi_r = TBranch() # *
   b_pd0_r = TBranch() # *
   b_md0_r = TBranch() # *
   b_Vtxd0_r = TBranch() # *
   b_cvtxd0_r = TBranch() # *
   b_dxy_r = TBranch() # *
   b_dz_r = TBranch() # *
   b_psi_r = TBranch() # *
   b_rd0_d = TBranch() # *
   b_drd0_d = TBranch() # *
   b_rpd0_d = TBranch() # *
   b_drpd0_d = TBranch() # *
   b_rd0_t = TBranch() # *
   b_drd0_t = TBranch() # *
   b_rpd0_t = TBranch() # *
   b_drpd0_t = TBranch() # *
   b_rd0_dt = TBranch() # *
   b_drd0_dt = TBranch() # *
   b_prbr_dt = TBranch() # *
   b_prbz_dt = TBranch() # *
   b_rd0_tt = TBranch() # *
   b_drd0_tt = TBranch() # *
   b_prbr_tt = TBranch() # *
   b_prbz_tt = TBranch() # *
   b_ijetd0 = TBranch() # *
   b_ptr3d0_j = TBranch() # *
   b_ptr2d0_j = TBranch() # *
   b_ptr3d0_3 = TBranch() # *
   b_ptr2d0_3 = TBranch() # *
   b_ptr2d0_2 = TBranch() # *
   b_Mimpds_r = TBranch() # *
   b_Mimpbk_r = TBranch() # *
   b_ntracks = TBranch() # *
   b_pt = TBranch() # *
   b_kappa = TBranch() # *
   b_phi = TBranch() # *
   b_theta = TBranch() # *
   b_dca = TBranch() # *
   b_z0 = TBranch() # *
   b_covar = TBranch() # *
   b_nhitrp = TBranch() # *
   b_prbrp = TBranch() # *
   b_nhitz = TBranch() # *
   b_prbz = TBranch() # *
   b_rstart = TBranch() # *
   b_rend = TBranch() # *
   b_lhk = TBranch() # *
   b_lhpi = TBranch() # *
   b_nlhk = TBranch() # *
   b_nlhpi = TBranch() # *
   b_dca_d = TBranch() # *
   b_ddca_d = TBranch() # *
   b_dca_t = TBranch() # *
   b_ddca_t = TBranch() # *
   b_muqual = TBranch() # *
   b_imu = TBranch() # *
   b_imufe = TBranch() # *
   b_njets = TBranch() # *
   b_E_j = TBranch() # *
   b_pt_j = TBranch() # *
   b_theta_j = TBranch() # *
   b_eta_j = TBranch() # *
   b_phi_j = TBranch() # *
   b_m_j = TBranch() # *
   b_thrust = TBranch() # *
   b_pthrust = TBranch() # *
   b_thrust2 = TBranch() # *
   b_pthrust2 = TBranch() # *
   b_spher = TBranch() # *
   b_aplan = TBranch() # *
   b_plan = TBranch() # *
   b_nnout = TBranch() # *

   

   # # #
   # constructor and destructor methods
   #

   #h1analysis(TTree *tree=nullptr);
   #~h1analysis() override { }
   def __init__(self, tree : TTree = TTree() ):
      TSelector.__init__( self ) 
      pass
   def __del__(self, ):
      pass



   # # #
   # methods
   #

   # void    
   def Reset( self, ) : pass

   # int
   def Version( self, ):
      return 1

   # void    
   def Begin( self, tree : TTree = TTree()) : pass

   # void    
   def SlaveBegin( self, tree : TTree = TTree()) : pass

   # void    
   def Init( self, tree : TTree = TTree()) : pass

   # Bool_t  
   def Notify( self, ) : pass

   # Bool_t  
   def Process( self, entry : Long64_t) : pass

   # void    
   def SetOption( self, option : char) :
      fOption = option

   # void    
   def SetObject( self, obj : TObject) :
      fObject = obj

   # void    
   def SetInputList( self, Input : TList) :
      fInput = Input

   # # TList  *
   # def GetOutputList( self, ):
   #    return self.fOutput

   # void    
   def SlaveTerminate( self, ) : pass

   # void    
   def Terminate( self, ) : pass
   
   #ClassDefOverride(h1analysis,2);
   
   



#_____________________________________________________________________
# 
def h1analysis__h1analysis( self,  tree : TTree = TTree() ):
   TSelector.__init__( self ) 

   # Constructor.
   
   # TODO  does Reset come here?
   # self.Reset()
   pass
   # or
   #self.fChain = tree 
   
#_____________________________________________________________________
# void
def h1analysis__Reset( self, ) :

   # Reset the data members to their initial value.
   
   self.hdmd = nullptr
   self.h2 = nullptr
   self.fChain = nullptr
   self.elist = nullptr
   self.fillList = False
   self.useList  = False
   self.fProcessed = 0
   

#_____________________________________________________________________
# void
def h1analysis__Init( self, tree : TTree = TTree()) :

   #   Set branch addresses
   
   #Info( "Init", "tree: %p" % tree ) # %p not supported
   Info( "Init", "tree: %s" % hex( addressof( tree ) ) )
   
   if (tree == nullptr) : return

   self.fChain    = tree

   
   print( "not using self.fChain.SetBranchAddress" )
   #self.fChain.SetBranchAddress( "nrun",         self.nrun,         self.b_nrun,      )
   #self.fChain.SetBranchAddress( "nevent",       self.nevent,       self.b_nevent,    )
   #self.fChain.SetBranchAddress( "nentry",       self.nentry,       self.b_nentry,    )
   #self.fChain.SetBranchAddress( "trelem",       self.trelem,       self.b_trelem,    )
   #self.fChain.SetBranchAddress( "subtr",        self.subtr,        self.b_subtr,     )
   #self.fChain.SetBranchAddress( "rawtr",        self.rawtr,        self.b_rawtr,     )
   #self.fChain.SetBranchAddress( "L4subtr",      self.L4subtr,      self.b_L4subtr,   )
   #self.fChain.SetBranchAddress( "L5class",      self.L5class,      self.b_L5class,   )
   #self.fChain.SetBranchAddress( "E33",          self.E33,          self.b_E33,       )
   #self.fChain.SetBranchAddress( "de33",         self.de33,         self.b_de33,      )
   #self.fChain.SetBranchAddress( "x33",          self.x33,          self.b_x33,       )
   #self.fChain.SetBranchAddress( "dx33",         self.dx33,         self.b_dx33,      )
   #self.fChain.SetBranchAddress( "y33",          self.y33,          self.b_y33,       )
   #self.fChain.SetBranchAddress( "dy33",         self.dy33,         self.b_dy33,      )
   #self.fChain.SetBranchAddress( "E44",          self.E44,          self.b_E44,       )
   #self.fChain.SetBranchAddress( "de44",         self.de44,         self.b_de44,      )
   #self.fChain.SetBranchAddress( "x44",          self.x44,          self.b_x44,       )
   #self.fChain.SetBranchAddress( "dx44",         self.dx44,         self.b_dx44,      )
   #self.fChain.SetBranchAddress( "y44",          self.y44,          self.b_y44,       )
   #self.fChain.SetBranchAddress( "dy44",         self.dy44,         self.b_dy44,      )
   #self.fChain.SetBranchAddress( "Ept",          self.Ept,          self.b_Ept,       )
   #self.fChain.SetBranchAddress( "dept",         self.dept,         self.b_dept,      )
   #self.fChain.SetBranchAddress( "xpt",          self.xpt,          self.b_xpt,       )
   #self.fChain.SetBranchAddress( "dxpt",         self.dxpt,         self.b_dxpt,      )
   #self.fChain.SetBranchAddress( "ypt",          self.ypt,          self.b_ypt,       )
   #self.fChain.SetBranchAddress( "dypt",         self.dypt,         self.b_dypt,      )
   #self.fChain.SetBranchAddress( "pelec",        self.pelec,        self.b_pelec,     )
   #self.fChain.SetBranchAddress( "flagelec",     self.flagelec,     self.b_flagelec,  )
   #self.fChain.SetBranchAddress( "xeelec",       self.xeelec,       self.b_xeelec,    )
   #self.fChain.SetBranchAddress( "yeelec",       self.yeelec,       self.b_yeelec,    )
   #self.fChain.SetBranchAddress( "Q2eelec",      self.Q2eelec,      self.b_Q2eelec,   )
   #self.fChain.SetBranchAddress( "nelec",        self.nelec,        self.b_nelec,     )
   #self.fChain.SetBranchAddress( "Eelec",        self.Eelec,        self.b_Eelec,     )
   #self.fChain.SetBranchAddress( "thetelec",     self.thetelec,     self.b_thetelec,  )
   #self.fChain.SetBranchAddress( "phielec",      self.phielec,      self.b_phielec,   )
   #self.fChain.SetBranchAddress( "xelec",        self.xelec,        self.b_xelec,     )
   #self.fChain.SetBranchAddress( "Q2elec",       self.Q2elec,       self.b_Q2elec,    )
   #self.fChain.SetBranchAddress( "xsigma",       self.xsigma,       self.b_xsigma,    )
   #self.fChain.SetBranchAddress( "Q2sigma",      self.Q2sigma,      self.b_Q2sigma,   )
   #self.fChain.SetBranchAddress( "sumc",         self.sumc,         self.b_sumc,      )
   #self.fChain.SetBranchAddress( "sumetc",       self.sumetc,       self.b_sumetc,    )
   #self.fChain.SetBranchAddress( "yjbc",         self.yjbc,         self.b_yjbc,      )
   #self.fChain.SetBranchAddress( "Q2jbc",        self.Q2jbc,        self.b_Q2jbc,     )
   #self.fChain.SetBranchAddress( "sumct",        self.sumct,        self.b_sumct,     )
   #self.fChain.SetBranchAddress( "sumetct",      self.sumetct,      self.b_sumetct,   )
   #self.fChain.SetBranchAddress( "yjbct",        self.yjbct,        self.b_yjbct,     )
   #self.fChain.SetBranchAddress( "Q2jbct",       self.Q2jbct,       self.b_Q2jbct,    )
   #self.fChain.SetBranchAddress( "Ebeamel",      self.Ebeamel,      self.b_Ebeamel,   )
   #self.fChain.SetBranchAddress( "Ebeampr",      self.Ebeampr,      self.b_Ebeampr,   )
   #self.fChain.SetBranchAddress( "pvtx_d",       self.pvtx_d,       self.b_pvtx_d,    )
   #self.fChain.SetBranchAddress( "cpvtx_d",      self.cpvtx_d,      self.b_cpvtx_d,   )
   #self.fChain.SetBranchAddress( "pvtx_t",       self.pvtx_t,       self.b_pvtx_t,    )
   #self.fChain.SetBranchAddress( "cpvtx_t",      self.cpvtx_t,      self.b_cpvtx_t,   )
   #self.fChain.SetBranchAddress( "ntrkxy_t",     self.ntrkxy_t,     self.b_ntrkxy_t,  )
   #self.fChain.SetBranchAddress( "prbxy_t",      self.prbxy_t,      self.b_prbxy_t,   )
   #self.fChain.SetBranchAddress( "ntrkz_t",      self.ntrkz_t,      self.b_ntrkz_t,   )
   #self.fChain.SetBranchAddress( "prbz_t",       self.prbz_t,       self.b_prbz_t,    )
   #self.fChain.SetBranchAddress( "nds",          self.nds,          self.b_nds,       )
   #self.fChain.SetBranchAddress( "rankds",       self.rankds,       self.b_rankds,    )
   #self.fChain.SetBranchAddress( "qds",          self.qds,          self.b_qds,       )
   #self.fChain.SetBranchAddress( "pds_d",        self.pds_d,        self.b_pds_d,     )
   #self.fChain.SetBranchAddress( "ptds_d",       self.ptds_d,       self.b_ptds_d,    )
   #self.fChain.SetBranchAddress( "etads_d",      self.etads_d,      self.b_etads_d,   )
   #self.fChain.SetBranchAddress( "dm_d",         self.dm_d,         self.b_dm_d,      )
   #self.fChain.SetBranchAddress( "ddm_d",        self.ddm_d,        self.b_ddm_d,     )
   #self.fChain.SetBranchAddress( "pds_t",        self.pds_t,        self.b_pds_t,     )
   #self.fChain.SetBranchAddress( "dm_t",         self.dm_t,         self.b_dm_t,      )
   #self.fChain.SetBranchAddress( "ddm_t",        self.ddm_t,        self.b_ddm_t,     )
   #self.fChain.SetBranchAddress( "ik",           self.ik,           self.b_ik,        )
   #self.fChain.SetBranchAddress( "ipi",          self.ipi,          self.b_ipi,       )
   #self.fChain.SetBranchAddress( "ipis",         self.ipis,         self.b_ipis,      )
   #self.fChain.SetBranchAddress( "pd0_d",        self.pd0_d,        self.b_pd0_d,     )
   #self.fChain.SetBranchAddress( "ptd0_d",       self.ptd0_d,       self.b_ptd0_d,    )
   #self.fChain.SetBranchAddress( "etad0_d",      self.etad0_d,      self.b_etad0_d,   )
   #self.fChain.SetBranchAddress( "md0_d",        self.md0_d,        self.b_md0_d,     )
   #self.fChain.SetBranchAddress( "dmd0_d",       self.dmd0_d,       self.b_dmd0_d,    )
   #self.fChain.SetBranchAddress( "pd0_t",        self.pd0_t,        self.b_pd0_t,     )
   #self.fChain.SetBranchAddress( "md0_t",        self.md0_t,        self.b_md0_t,     )
   #self.fChain.SetBranchAddress( "dmd0_t",       self.dmd0_t,       self.b_dmd0_t,    )
   #self.fChain.SetBranchAddress( "pk_r",         self.pk_r,         self.b_pk_r,      )
   #self.fChain.SetBranchAddress( "ppi_r",        self.ppi_r,        self.b_ppi_r,     )
   #self.fChain.SetBranchAddress( "pd0_r",        self.pd0_r,        self.b_pd0_r,     )
   #self.fChain.SetBranchAddress( "md0_r",        self.md0_r,        self.b_md0_r,     )
   #self.fChain.SetBranchAddress( "Vtxd0_r",      self.Vtxd0_r,      self.b_Vtxd0_r,   )
   #self.fChain.SetBranchAddress( "cvtxd0_r",     self.cvtxd0_r,     self.b_cvtxd0_r,  )
   #self.fChain.SetBranchAddress( "dxy_r",        self.dxy_r,        self.b_dxy_r,     )
   #self.fChain.SetBranchAddress( "dz_r",         self.dz_r,         self.b_dz_r,      )
   #self.fChain.SetBranchAddress( "psi_r",        self.psi_r,        self.b_psi_r,     )
   #self.fChain.SetBranchAddress( "rd0_d",        self.rd0_d,        self.b_rd0_d,     )
   #self.fChain.SetBranchAddress( "drd0_d",       self.drd0_d,       self.b_drd0_d,    )
   #self.fChain.SetBranchAddress( "rpd0_d",       self.rpd0_d,       self.b_rpd0_d,    )
   #self.fChain.SetBranchAddress( "drpd0_d",      self.drpd0_d,      self.b_drpd0_d,   )
   #self.fChain.SetBranchAddress( "rd0_t",        self.rd0_t,        self.b_rd0_t,     )
   #self.fChain.SetBranchAddress( "drd0_t",       self.drd0_t,       self.b_drd0_t,    )
   #self.fChain.SetBranchAddress( "rpd0_t",       self.rpd0_t,       self.b_rpd0_t,    )
   #self.fChain.SetBranchAddress( "drpd0_t",      self.drpd0_t,      self.b_drpd0_t,   )
   #self.fChain.SetBranchAddress( "rd0_dt",       self.rd0_dt,       self.b_rd0_dt,    )
   #self.fChain.SetBranchAddress( "drd0_dt",      self.drd0_dt,      self.b_drd0_dt,   )
   #self.fChain.SetBranchAddress( "prbr_dt",      self.prbr_dt,      self.b_prbr_dt,   )
   #self.fChain.SetBranchAddress( "prbz_dt",      self.prbz_dt,      self.b_prbz_dt,   )
   #self.fChain.SetBranchAddress( "rd0_tt",       self.rd0_tt,       self.b_rd0_tt,    )
   #self.fChain.SetBranchAddress( "drd0_tt",      self.drd0_tt,      self.b_drd0_tt,   )
   #self.fChain.SetBranchAddress( "prbr_tt",      self.prbr_tt,      self.b_prbr_tt,   )
   #self.fChain.SetBranchAddress( "prbz_tt",      self.prbz_tt,      self.b_prbz_tt,   )
   #self.fChain.SetBranchAddress( "ijetd0",       self.ijetd0,       self.b_ijetd0,    )
   #self.fChain.SetBranchAddress( "ptr3d0_j",     self.ptr3d0_j,     self.b_ptr3d0_j,  )
   #self.fChain.SetBranchAddress( "ptr2d0_j",     self.ptr2d0_j,     self.b_ptr2d0_j,  )
   #self.fChain.SetBranchAddress( "ptr3d0_3",     self.ptr3d0_3,     self.b_ptr3d0_3,  )
   #self.fChain.SetBranchAddress( "ptr2d0_3",     self.ptr2d0_3,     self.b_ptr2d0_3,  )
   #self.fChain.SetBranchAddress( "ptr2d0_2",     self.ptr2d0_2,     self.b_ptr2d0_2,  )
   #self.fChain.SetBranchAddress( "Mimpds_r",     self.Mimpds_r,     self.b_Mimpds_r,  )
   #self.fChain.SetBranchAddress( "Mimpbk_r",     self.Mimpbk_r,     self.b_Mimpbk_r,  )
   #self.fChain.SetBranchAddress( "ntracks",      self.ntracks,      self.b_ntracks,   )
   #self.fChain.SetBranchAddress( "pt",           self.pt,           self.b_pt,        )
   #self.fChain.SetBranchAddress( "kappa",        self.kappa,        self.b_kappa,     )
   #self.fChain.SetBranchAddress( "phi",          self.phi,          self.b_phi,       )
   #self.fChain.SetBranchAddress( "theta",        self.theta,        self.b_theta,     )
   #self.fChain.SetBranchAddress( "dca",          self.dca,          self.b_dca,       )
   #self.fChain.SetBranchAddress( "z0",           self.z0,           self.b_z0,        )
   #self.fChain.SetBranchAddress( "covar",        self.covar,        self.b_covar,     )
   #self.fChain.SetBranchAddress( "nhitrp",       self.nhitrp,       self.b_nhitrp,    )
   #self.fChain.SetBranchAddress( "prbrp",        self.prbrp,        self.b_prbrp,     )
   #self.fChain.SetBranchAddress( "nhitz",        self.nhitz,        self.b_nhitz,     )
   #self.fChain.SetBranchAddress( "prbz",         self.prbz,         self.b_prbz,      )
   #self.fChain.SetBranchAddress( "rstart",       self.rstart,       self.b_rstart,    )
   #self.fChain.SetBranchAddress( "rend",         self.rend,         self.b_rend,      )
   #self.fChain.SetBranchAddress( "lhk",          self.lhk,          self.b_lhk,       )
   #self.fChain.SetBranchAddress( "lhpi",         self.lhpi,         self.b_lhpi,      )
   #self.fChain.SetBranchAddress( "nlhk",         self.nlhk,         self.b_nlhk,      )
   #self.fChain.SetBranchAddress( "nlhpi",        self.nlhpi,        self.b_nlhpi,     )
   #self.fChain.SetBranchAddress( "dca_d",        self.dca_d,        self.b_dca_d,     )
   #self.fChain.SetBranchAddress( "ddca_d",       self.ddca_d,       self.b_ddca_d,    )
   #self.fChain.SetBranchAddress( "dca_t",        self.dca_t,        self.b_dca_t,     )
   #self.fChain.SetBranchAddress( "ddca_t",       self.ddca_t,       self.b_ddca_t,    )
   #self.fChain.SetBranchAddress( "muqual",       self.muqual,       self.b_muqual,    )
   #self.fChain.SetBranchAddress( "imu",          self.imu,          self.b_imu,       )
   #self.fChain.SetBranchAddress( "imufe",        self.imufe,        self.b_imufe,     )
   #self.fChain.SetBranchAddress( "njets",        self.njets,        self.b_njets,     )
   #self.fChain.SetBranchAddress( "E_j",          self.E_j,          self.b_E_j,       )
   #self.fChain.SetBranchAddress( "pt_j",         self.pt_j,         self.b_pt_j,      )
   #self.fChain.SetBranchAddress( "theta_j",      self.theta_j,      self.b_theta_j,   )
   #self.fChain.SetBranchAddress( "eta_j",        self.eta_j,        self.b_eta_j,     )
   #self.fChain.SetBranchAddress( "phi_j",        self.phi_j,        self.b_phi_j,     )
   #self.fChain.SetBranchAddress( "m_j",          self.m_j,          self.b_m_j,       )
   #self.fChain.SetBranchAddress( "thrust",       self.thrust,       self.b_thrust,    )
   #self.fChain.SetBranchAddress( "pthrust",      self.pthrust,      self.b_pthrust,   )
   #self.fChain.SetBranchAddress( "thrust2",      self.thrust2,      self.b_thrust2,   )
   #self.fChain.SetBranchAddress( "pthrust2",     self.pthrust2,     self.b_pthrust2,  )
   #self.fChain.SetBranchAddress( "spher",        self.spher,        self.b_spher,     )
   #self.fChain.SetBranchAddress( "aplan",        self.aplan,        self.b_aplan,     )
   #self.fChain.SetBranchAddress( "plan",         self.plan,         self.b_plan,      )
   #self.fChain.SetBranchAddress( "nnout",        self.nnout,        self.b_nnout,     )
   

#_____________________________________________________________________
# Bool_t
def h1analysis__Notify( self, ) :

   #   Called when loading a new file.
   #   It gets the branch pointers.
   
   Info("Notify", 
        "processing file: %s" % ( 
           self.fChain.GetCurrentFile().GetName(),
           ),
        )
   
   if (self.elist and self.fChain)  :
      if (fillList)  :
         self.elist.SetTree(self.fChain)
      elif (useList)  :
         self.fChain.SetEntryList(self.elist)
         
      
   return True


# Loading function methods in class
h1analysis.__init__ = h1analysis__h1analysis
h1analysis.Reset = h1analysis__Reset
h1analysis.Init = h1analysis__Init
h1analysis.Notify = h1analysis__Notify
   


if __name__ == "__main__":

   #h1analysis_h()
   h1analysis()
   pass

   # test
   #my_h1analysis = h1analysis()
   #my_h1analysis.__init__()
   #my_h1analysis.Reset()
   #my_h1analysis.Init()
   #my_h1analysis.Notify() # error
