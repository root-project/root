#   Set visibility attributes for the NA49 geometry
#        Set Shape attributes

import ROOT

ROOT.YK01.SetVisibility( 0 )
ROOT.YK03.SetLineColor( 2 )
ROOT.YK04.SetLineColor( 5 )
ROOT.SEC1.SetLineColor( 6 )
ROOT.SEC2.SetLineColor( 6 )
ROOT.SEC3.SetLineColor( 3 )
ROOT.SEC4.SetLineColor( 3 )
ROOT.TOFR.SetLineColor( 5 )
ROOT.COI1.SetLineColor( 4 )
ROOT.COI2.SetLineColor( 4 )
ROOT.COI3.SetLineColor( 4 )
ROOT.COI4.SetLineColor( 4 )
ROOT.CS38.SetLineColor( 5 )
ROOT.CS28.SetLineColor( 5 )
ROOT.CS18.SetLineColor( 5 )
ROOT.TF4D.SetLineColor( 3 )
ROOT.OGB4.SetLineColor( 3 )
ROOT.TF3D.SetLineColor( 3 )
ROOT.OGB3.SetLineColor( 3 )
ROOT.TF4A.SetLineColor( 3 )
ROOT.OGB4.SetLineColor( 3 )
ROOT.TF3A.SetLineColor( 3 )
ROOT.OGB3.SetLineColor( 3 )

#   Copy shape attributes (colors,etc) in nodes referencing the shapse
CAVE1 = ROOT.gGeometry.FindObject( 'CAVE1' )
CAVE1.ImportShapeAttributes( )

#  Set Node attributes
CAVE1.SetVisibility( 2 )   # node is not drawn but its sons are drawn
ROOT.gGeometry.FindObject( 'VT1_1' ).SetVisibility( -4 )  # Node is not drawn.
                                                          # Its immediate sons are drawn
ROOT.gGeometry.FindObject( 'VT2_1' ).SetVisibility( -4 )
ROOT.gGeometry.FindObject( 'MTL_1' ).SetVisibility( -4 )
ROOT.gGeometry.FindObject( 'MTR_1' ).SetVisibility( -4 )
ROOT.gGeometry.FindObject( 'TOFR1' ).SetVisibility( -4 )

