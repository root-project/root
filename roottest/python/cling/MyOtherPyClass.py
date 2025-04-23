print( 'creating class MyOtherPyClass ... ', flush=True )

class MyOtherPyClass:
   count = 0

   def __init__( self ):
      print( 'in MyOtherPyClass.__init__', flush=True )
      MyOtherPyClass.count += 1

   def __del__( self ):
      print( 'in MyOtherPyClass.__del__', flush=True )
      MyOtherPyClass.count -= 1

   def hop( self ):
      print( 'hop', flush=True )

   def duck( self ):
      print( 'quack', flush=True )


# include a class that may interfere with the previous one due to
# same-named method in it
class MyYetAnotherPyClass:
   def hop( self ):
      print( 'another hop', flush=True )
