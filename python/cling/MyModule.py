from force_flush import print_flushed

print_flushed( 'loading MyModule.py ... ' )

class MyPyClass( object ):
   def __init__( self ):
      print_flushed( 'in MyModule.MyPyClass.__init__' )

   def gime( self, what ):
      return what
