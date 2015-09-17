#
# To see the output of this macro, click begin_html <a href="gif/shapes.gif" >here</a> end_html
#

import ROOT

c1 = ROOT.TCanvas( 'c1', 'Geometry Shapes', 200, 10, 700, 500 )

# delete previous geometry objects in case this script is reexecuted
if hasattr(ROOT, 'gGeometry') and ROOT.gGeometry:
   ROOT.gGeometry.GetListOfNodes().Delete()
   ROOT.gGeometry.GetListOfShapes().Delete()

#  Define some volumes
brik = ROOT.TBRIK( 'BRIK', 'BRIK', 'void', 200, 150, 150 )
trd1 = ROOT.TTRD1( 'TRD1', 'TRD1', 'void', 200, 50, 100, 100 )
trd2 = ROOT.TTRD2( 'TRD2', 'TRD2', 'void', 200, 50, 200, 50, 100 )
trap = ROOT.TTRAP( 'TRAP', 'TRAP', 'void', 190, 0, 0, 60, 40, 90, 15, 120, 80, 180, 15 )
para = ROOT.TPARA( 'PARA', 'PARA', 'void', 100, 200, 200, 15, 30, 30 )
gtra = ROOT.TGTRA( 'GTRA', 'GTRA', 'void', 390, 0, 0, 20, 60, 40, 90, 15, 120, 80, 180, 15 )
tube = ROOT.TTUBE( 'TUBE', 'TUBE', 'void', 150, 200, 400 )
tubs = ROOT.TTUBS( 'TUBS', 'TUBS', 'void', 80, 100, 100, 90, 235 )
cone = ROOT.TCONE( 'CONE', 'CONE', 'void', 100, 50, 70, 120, 150 )
cons = ROOT.TCONS( 'CONS', 'CONS', 'void', 50, 100, 100, 200, 300, 90, 270 )
sphe  = ROOT.TSPHE( 'SPHE',  'SPHE',  'void', 25, 340, 45, 135,  0, 270 )
sphe1 = ROOT.TSPHE( 'SPHE1', 'SPHE1', 'void',  0, 140,  0, 180,  0, 360 )
sphe2 = ROOT.TSPHE( 'SPHE2', 'SPHE2', 'void',  0, 200, 10, 120, 45, 145 )

pcon = ROOT.TPCON( 'PCON', 'PCON', 'void', 180, 270, 4 )
pcon.DefineSection( 0, -200, 50, 100 )
pcon.DefineSection( 1,  -50, 50,  80 )
pcon.DefineSection( 2,   50, 50,  80 )
pcon.DefineSection( 3,  200, 50, 100 )

pgon = ROOT.TPGON( 'PGON', 'PGON', 'void', 180, 270, 8, 4 )
pgon.DefineSection( 0, -200, 50, 100 )
pgon.DefineSection( 1,  -50, 50,  80 )
pgon.DefineSection( 2,   50, 50,  80 )
pgon.DefineSection( 3,  200, 50, 100 )

#  Set shapes attributes
brik.SetLineColor( 1 )
trd1.SetLineColor( 2 )
trd2.SetLineColor( 3 )
trap.SetLineColor( 4 )
para.SetLineColor( 5 )
gtra.SetLineColor( 7 )
tube.SetLineColor( 6 )
tubs.SetLineColor( 7 )
cone.SetLineColor( 2 )
cons.SetLineColor( 3 )
pcon.SetLineColor( 6 )
pgon.SetLineColor( 2 )
sphe.SetLineColor( ROOT.kRed )
sphe1.SetLineColor( ROOT.kBlack )
sphe2.SetLineColor( ROOT.kBlue )

#  Build the geometry hierarchy
node1 = ROOT.TNode( 'NODE1', 'NODE1', 'BRIK' )
node1.cd()

node2  = ROOT.TNode(  'NODE2',  'NODE2', 'TRD1',     0,     0, -1000 )
node3  = ROOT.TNode(  'NODE3',  'NODE3', 'TRD2',     0,     0,  1000 )
node4  = ROOT.TNode(  'NODE4',  'NODE4', 'TRAP',     0, -1000,     0 )
node5  = ROOT.TNode(  'NODE5',  'NODE5', 'PARA',     0,  1000,     0 )
node6  = ROOT.TNode(  'NODE6',  'NODE6', 'TUBE', -1000,     0,     0 )
node7  = ROOT.TNode(  'NODE7',  'NODE7', 'TUBS',  1000,     0,     0 )
node8  = ROOT.TNode(  'NODE8',  'NODE8', 'CONE',  -300,  -300,     0 )
node9  = ROOT.TNode(  'NODE9',  'NODE9', 'CONS',   300,   300,     0 )
node10 = ROOT.TNode( 'NODE10', 'NODE10', 'PCON',     0, -1000, -1000 )
node11 = ROOT.TNode( 'NODE11', 'NODE11', 'PGON',     0,  1000,  1000 )
node12 = ROOT.TNode( 'NODE12', 'NODE12', 'GTRA',     0,  -400,   700 )
node13 = ROOT.TNode( 'NODE13', 'NODE13', 'SPHE',    10,  -400,   500 )
node14 = ROOT.TNode( 'NODE14', 'NODE14', 'SPHE1',   10,   250,   300 )
node15 = ROOT.TNode( 'NODE15', 'NODE15', 'SPHE2',   10,  -100,  -200 )

# for memory management
for l, o in locals().items():
   if isinstance( o, ROOT.TShape ) or isinstance( o, ROOT.TNode ):
      ROOT.SetOwnership( o, False )

# Draw this geometry in the current canvas
node1.cd()
node1.Draw( 'gl' )
c1.Update()
#
#  Draw the geometry using the x3d viewver.
#  Note that this viewver may also be invoked from the "View" menu in
#  the canvas tool bar
#
# once in x3d viewer, type m to see the menu.
# For example typing r will show a solid model of this geometry.
