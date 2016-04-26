from force_flush import print_flushed

print_flushed( 'creating class MyPyClass ... ' )

class MyPyClass:
   def __init__( self ):
      print_flushed( 'in MyPyClass.__init__' )

   def gime( self, what ):
      return what
