print( 'loading MyModule.py ... ', flush=True )

class MyPyClass( object ):
   def __init__( self ):
      print( 'in MyModule.MyPyClass.__init__', flush=True )

   def gime( self, what ):
      return what
