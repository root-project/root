print 'creating class MyOtherPyClass ... '

class MyOtherPyClass:
   count = 0

   def __init__( self ):
      print 'in MyOtherPyClass.__init__'
      MyOtherPyClass.count += 1

   def __del__( self ):
      print 'in MyOtherPyClass.__del__'
      MyOtherPyClass.count -= 1

   def hop( self ):
      print 'hop'

   def duck( self ):
      print 'quack'


# include a class that may interfere with the previous one due to
# same-named method in it
class MyYetAnotherPyClass:
   def hop( self ):
      print 'another hop'
