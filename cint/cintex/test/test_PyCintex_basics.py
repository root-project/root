#!/usr/bin/env python
#
# PyLCGDict unittest programs
# ---------------------------
# Author: Pere Mato
# (June 2003)
#
import unittest
import sys

import PyCintex

class BasicsTestCase(unittest.TestCase):
  def setUp(self):
    PyCintex.makeClass('A::B::C::MyClass')     # This is needed to force loading the dictionary
    self.A   = PyCintex.makeNamespace('A')
    self.std = PyCintex.makeNamespace('std')
    self.gbl = PyCintex.makeNamespace('')
    
  def tearDown(self):
    pass

  def test01DoSomething(self):
    object = self.A.B.C.MyClass()
    self.failUnless( isinstance(object, self.A.B.C.MyClass),
                     'unable to create an instance of A::B::C::MyClass')
    result = object.doSomething('Hello World')
    self.failUnless( result == len('Hello World'),
                     'incorrect return value from doSomething')
    self.failUnlessEqual( self.A.B.C.ValidityKeyMin, 0)  
    
  
  def test02PrimitiveArgTypes(self):
    p = self.A.B.C.Primitives()
    self.failUnless( isinstance(p, self.A.B.C.Primitives) )
    p.set_b(False)
    self.failUnless( p.b() == False , 'fail to set bool false')
    p.set_b(True)
    self.failUnless( p.b() == True , 'fail to set bool true')
    self.failUnlessRaises( TypeError, p.set_b, 10 )
    p.set_c('h')
    self.failUnless( p.c() == 'h', 'fail to set char')
    p.set_c(40)
    self.failUnless( p.c() == chr(40), 'fail to set char')
    self.failUnlessRaises( TypeError, p.set_c, 'ssss')    
    p.set_s(-8)
    self.failUnless( p.s() == -8, 'fail to set short' )
    self.failUnlessRaises( TypeError, p.set_s, 1.5)    
    self.failUnlessRaises( TypeError, p.set_s, 'ssss')    
    p.set_i(8)
    self.failUnless( p.i() == 8, 'fail to set int' )
    p.set_l(-8)
    self.failUnless( p.l() == -8, 'fail to set long' )
    p.set_ll(-8)
    self.failUnless( p.ll() == -8, 'fail to set long long' )
    p.set_uc(8)
    self.failUnless( p.uc() == chr(8), 'fail to set unsigned char' )
    p.set_us(8)
    self.failUnless( p.us() == 8, 'fail to set unsigned short' )
    p.set_ui(8)
    self.failUnless( p.ui() == 8, 'fail to set unsigned int' )
    p.set_ul(88)
    p.set_f(8.8)
    self.failUnless( abs(p.f() - 8.8) < 0.00001, 'fail to set float' )
    p.set_d(8.8)
    self.failUnless( p.d() == 8.8, 'fail to set double' )
    p.set_str('This is a string')
    self.failUnless( p.str() == 'This is a string', 'fail to set string' )
    self.failUnless( type(p.str()) is str, 'fail to comprate type of strings' )
    p.set_cstr('This is a C string')
    self.failUnless( p.cstr() == 'This is a C string', 'fail to set string' )
    self.failUnless( p.ccstr() == 'This is a C string', 'fail to set string' )
    p.set_all(1,'g',7,7,7,7.7,7.7,'Another string')
    self.failUnless( (p.b(), p.c(),p.s(),p.i(),p.l(), p.str()) ==
                     (1,'g',7,7,7,'Another string'),
                     'fail to set multiple argument types' )


  def test03ReturnModes(self):
    myobj = self.A.B.C.MyClass()
    myobj.setMagic(1234567890)
    self.failUnless( myobj.magic() == 1234567890 )
    calling = self.A.B.C.Calling()
    calling.setByReference(myobj)
    self.failUnless( calling.retByValue().magic() == 1234567890 , 'fail return by value')
    self.failUnless( calling.retByPointer().magic() == 1234567890 , 'fail return by pointer')
    self.failUnless( calling.retByReference().magic() == 1234567890 , 'fail return by reference')
    self.failUnless( calling.retByRefPointer().magic() == 1234567890 , 'fail return by reference pointer')
    self.failUnless( calling.retByVoidPointer(), 'fail return by void pointer')

    myobj.setMagic(111111111)
    self.failUnless( myobj.magic() == 111111111 )
    calling = self.A.B.C.Calling()
    calling.setByConstReference(myobj)
    self.failUnless( calling.retByValue().magic() == 111111111 , 'fail return by value')
    self.failUnless( calling.retByPointer().magic() == 111111111 , 'fail return by pointer')
    self.failUnless( calling.retByReference().magic() == 111111111 , 'fail return by reference')
    self.failUnless( calling.retByRefPointer().magic() == 111111111 , 'fail return by reference pointer')
    self.failUnless( calling.retByVoidPointer(), 'fail return by void pointer')

  def test04UnknownTypes(self) :
    calling = self.A.B.C.Calling()
    #---Returning unknown types
    rp = calling.retUnknownTypePointer()
    self.failUnless( rp )
    rr = calling.retUnknownTypeReference()
    self.failUnless( rr )
    #---Passing unknown types
    self.failUnlessEqual( calling.setByUnknownTypePointer(rp), 0x12345678)
    self.failUnlessEqual( calling.setByUnknownTypeReference(rr), 0x12345678)
    self.failUnlessEqual( calling.setByUnknownConstTypePointer(rp), 0x12345678)
    self.failUnlessEqual( calling.setByUnknownConstTypeReference(rr), 0x12345678)

  
  def test05ReturnDynamicType(self):
    dfact = self.A.B.C.DiamondFactory()
    d = dfact.getDiamond()
    self.failUnless(d)
    self.failUnlessEqual(d.vf(), 999.999)
    v = dfact.getVirtual()
    self.failUnless(v)
    self.failUnlessEqual(v.magic(), 987654321 )
    v.setMagic(22222)
    self.failUnlessEqual(d.magic(), 22222 )


  def test06CallingModes(self):
    self.failUnlessEqual( self.A.B.C.MyClass.instances(), 0)  
    self.failUnlessEqual( self.A.B.C.s_public_instances, 0)  
    myobj = self.A.B.C.MyClass()
    calling = self.A.B.C.Calling()
    myobj.setMagic(22222222)
    #---Check calling modes-------------
    calling.setByReference(myobj)
    self.failUnless( calling.retByPointer().magic() == 22222222 , 'fail set by value')
    myobj.setMagic(33333333)
    calling.setByPointer(myobj)
    self.failUnless( calling.retByPointer().magic() == 33333333 , 'fail set by pointer')
    self.failUnless( myobj.magic() == 999999 , 'fail set by pointer')
    # None not acceptable; user ROOT.NULL instead ...
    #calling.setByPointer(None)
    #calling.setByPointer( ROOT.NULL )
    calling.setByPointer(PyCintex.NULL)
    self.failUnless( calling.retByPointer().magic() == 0 , 'fail set by null pointer')
    myobj.setMagic(44444444)
    calling.setByReference(myobj)
    self.failUnless( calling.retByPointer().magic() == 44444444 , 'fail set by reference')
    self.failUnless( myobj.magic() == 999999 , 'fail set by reference')
    myobj.setMagic(55555555)
    calling.setByRefPointer(myobj)
    self.failUnless( calling.retByPointer().magic() == 55555555 , 'fail set by reference pointer')
    self.failUnless( myobj.magic() == 999999 , 'fail set by reference pointer')
    self.failUnlessEqual( calling.retStrByValue(), 'value' )
    self.failUnlessEqual( calling.retStrByRef(), 'reference' )
    self.failUnlessEqual( calling.retStrByConstRef(), 'const reference' )
    self.failUnlessEqual( calling.retConstCStr(), 'const pointer' )
    self.failUnlessEqual( calling.retCStr(), 'pointer' )
    del myobj, calling
    self.failUnlessEqual( self.A.B.C.MyClass.instances(), 0)

  
  def test07CallingChecks(self):
    calling = self.A.B.C.Calling()
    #---Check invalid argument type-------------
    self.assertRaises(TypeError, calling.setByValue, calling)
    self.assertRaises(TypeError, calling.setByPointer, calling)
    #---Implicit cast conversions-------
    xobj = self.A.B.C.Diamond()
    xobj.setMagic(88888888)
    calling.setByReference(xobj)
    self.failUnlessEqual( calling.retByPointer().magic(), 88888888 )
    xobj.setMagic(99999999)
    calling.setByPointer(xobj)
    self.failUnlessEqual( calling.retByPointer().magic(), 99999999 )
    del calling, xobj
    self.failUnlessEqual( self.A.B.C.MyClass.instances(), 0)


  def test08VoidPointerArgument(self):
    calling = self.A.B.C.Calling()
    obj = self.A.B.C.MyClass()
    obj.setMagic(77777777)
    calling.setByVoidPointer(obj)
    self.failUnless( calling.retByPointer().magic() == 77777777 , 'fail set by void pointer')    
    del calling, obj
    self.failUnlessEqual( self.A.B.C.MyClass.instances(), 0 , 'MyClass instances not deleted')

  def test09TemplatedClasses(self) :
    tc1 = getattr(self.A.B.C, 'Template<A::B::C::MyClass,float>')
    tc2 = self.A.B.C.Template(self.A.B.C.MyClass,'float')
    tc3 = self.A.B.C.Template(self.A.B.C.MyClass,float)
    tc4 = self.gbl.GlobalTemplate(self.A.B.C.MyClass, float)
    self.failUnless( str(tc2) == str(tc3), 'template instanciation using TemplateGenerator')
    #[ Bug #2239 ] Invalid python identity on templated classes
    self.failUnless( tc2 is self.A.B.C.Template(self.A.B.C.MyClass,'float'))
    self.failUnless( tc2 is tc3 )
    self.failUnless( tc2 is self.A.B.C.Template(self.A.B.C.MyClass,float))
    self.failUnless( tc4 is self.gbl.GlobalTemplate(self.A.B.C.MyClass, float))
    # Object instantiation
    object = self.A.B.C.Template(self.A.B.C.MyClass,'float')()
    result = object.doSomething('Hello World')
    self.failUnless( result == len('Hello World'), 'incorrect return value from doSomething')
    object = self.gbl.GlobalTemplate(self.A.B.C.MyClass,'float')()
    result = object.doSomething('Hello World')
    self.failUnless( result == len('Hello World'), 'incorrect return value from doSomething')
    del object
    self.failUnless( self.A.B.C.MyClass.instances() == 0 , 'MyClass instances not deleted')

  def test10VectorSemantics(self) :
    vi = self.std.vector('int')()
    for i in range(30) : vi.push_back(i*i)
    self.failUnless( len(vi) == vi.size() , 'invalid vector size')
    vs = vi[6:9]
    self.failUnless( vs.__class__ == vi.__class__, 'slice type is unchanged')
    self.failUnlessEqual(vs[0], 36 )

  """

  def test11ListSemantics(self) :
    li = self.std.list('int')()
    for i in range(30) : li.push_back(i*i)
    self.failUnless( len(li) == li.size() , 'invalid list size')
    self.failUnless( 25 in li )
    self.failUnlessEqual( [i for i in li], [x*x for x in range(30)])
    lo = self.std.list('A::B::C::MyClass*')()
    objs = [self.A.B.C.MyClass() for i in range(10)]
    for i in objs : lo.push_back(i)
    self.failUnlessEqual(len(lo), 10)
    for o in lo : o.setMagic(222)
    it = iter(lo)
    self.failUnlessEqual(it.next().magic(), 222)
    self.failUnlessEqual(it.next().magic(), 222)

  def test12MapSemantics(self) :
    Key = self.gbl.Key
    ma = self.std.map(Key,str)()
    self.failUnlessEqual( len(ma), 0)
    for i in range(30) : ma[Key(i)]='%d'%i
    self.failUnlessEqual( len(ma), 30)
    k1, k2 = Key(10), Key(20)
    self.failUnless( k1 in ma )
    self.failUnless( k1 in ma.keys())
    self.failUnless( k2 in ma )
    self.failUnless( k2 in ma.keys())
    for k in ma :
      self.failUnlessEqual( ma[k], '%d'%k.value())
    del ma[k1]
    del ma[k2]
    self.failUnlessEqual( len(ma), 28)
  """
  
  def test13VirtualInheritance(self) :
    d = self.A.B.C.Diamond()
    self.failUnlessEqual(d.vf(), 999.999 )
    self.failUnlessEqual(d.magic(), 987654321)
    del d
    self.failUnlessEqual(self.A.B.C.MyClass.instances(), 0)

  def test14MethodOverloading(self) :
    #int overloaded( int ) { return 1; }
    #int overloaded( float ) { return 2; }
    #int overloaded( int, float ) { return 3; }
    #int overloaded( float, int ) { return 4; }
    calling = self.A.B.C.Calling()
    self.failUnlessEqual(calling.overloaded(10), 1)
    self.failUnlessEqual(calling.overloaded(10.0), 2)
    self.failUnlessEqual(calling.overloaded(10, 10.0), 3)
    self.failUnlessEqual(calling.overloaded(10.0, 10), 4)

  
  def test15OperatorOverloading(self) :
    Number = self.A.B.C.Number;
    self.failUnlessEqual(Number(20) + Number(10), Number(30) )
    self.failUnlessEqual(Number(20) - Number(10), Number(10) )
    self.failUnlessEqual(Number(20) / Number(10), Number(2) )
    self.failUnlessEqual(Number(20) * Number(10), Number(200) )
    self.failUnlessEqual(Number(5)  & Number(14), Number(4) )
    self.failUnlessEqual(Number(5)  | Number(14), Number(15) )
    self.failUnlessEqual(Number(5)  ^ Number(14), Number(11) )
    self.failUnlessEqual(Number(5)  << 2, Number(20) )
    self.failUnlessEqual(Number(20) >> 2, Number(5) )
    n  = Number(20)
    n += Number(10)
    n -= Number(10)
    n *= Number(10)
    n /= Number(2)
    self.failUnlessEqual(n ,Number(100) )
    self.failUnlessEqual(Number(20) >  Number(10), 1 )
    self.failUnlessEqual(Number(20) <  Number(10), 0 )
    self.failUnlessEqual(Number(20) >= Number(20), 1 )
    self.failUnlessEqual(Number(20) <= Number(10), 0 )
    self.failUnlessEqual(Number(20) != Number(10), 1 )
    self.failUnlessEqual(Number(20) == Number(10), 0 )

 
  def test16DataMembers(self) :
    dm = self.A.B.C.DataMembers()
    # testing get
    self.failUnlessEqual(dm.i, 0 )
    self.failUnlessEqual(dm.f, 0.0 )
    self.failUnlessEqual(dm.myclass.magic(), self.A.B.C.MyClass().magic() )
    # testing set
    dm.i = 8
    self.failUnlessEqual(dm.i, 8 )
    dm.f = 8.8
    self.failUnless( abs(dm.f - 8.8) < 0.00001)
    mc = self.A.B.C.MyClass()
    mc.setMagic(99999)
    dm.myclass.setMagic( 123456 )
    dm.myclass = mc
    self.failUnlessEqual(dm.myclass.magic(), 99999 )
    dm.p_myclass.setMagic(555555)
    self.failUnlessEqual(dm.p_myclass.magic(), 555555 )
    dm.d = 88.88
    self.failUnless( abs(dm.d - 88.88) < 0.00001)
    self.failUnlessEqual(dm.s, '')
    dm.s = 'small'
    self.failUnlessEqual(dm.s, 'small')
    s = 'a rather long or very long string this time instead of a using a small one that can not be stored locally'
    dm.s = s
    self.failUnlessEqual(dm.s, s)
    # testing inherited members
    xm = self.A.B.C.ExtDataMembers()
    xm.i  = 5
    xm.d  = 10.0
    xm.f  = 5.0
    xm.xi = 55
    self.failUnlessEqual(xm.i, 5 )
    self.failUnlessEqual(xm.d, 10.0 )
    self.failUnlessEqual(xm.f, 5.0 )
    self.failUnlessEqual(xm.xi, 55 )
    del dm, mc, xm
    self.failUnless( self.A.B.C.MyClass.instances() == 0 , 'MyClass instances not deleted')


  def test17EnumConversion(self) :
    # note that setAnswer in testclasses.h has been changed const Answer& -> Answer, as
    # enums are UInt_ts and references of built-in types are not (yet) supported
    myobj = self.A.B.C.MyClass()
    myobj.setAnswer(0)
    self.failUnlessEqual(myobj.answer(), 0 )
    myobj.setAnswer(1)
    self.failUnlessEqual(myobj.answer(), 1 )
 
  def test18DefaultArguments(self) :
    m0 = self.A.B.C.DefaultArguments()  # default arguments (1, 0.0)
    self.failUnlessEqual((m0.i(),m0.f()), (1,0.0))
    m1 = self.A.B.C.DefaultArguments(99)
    self.failUnlessEqual((m1.i(),m1.f()), (99,0.0))
    m2 = self.A.B.C.DefaultArguments(88,8.8)
    self.failUnlessEqual( m1.function('string'), 1004.0)
    self.failUnlessEqual( m1.function('string',10.0), 1015.0)
    self.failUnlessEqual( m1.function('string',20.0,20), 46.0)
    self.assertRaises(TypeError, m2.function, (20.0,20))
    self.assertRaises(TypeError, m2.function, ('a',20.0,20.5))
    self.assertRaises(TypeError, m2.function, ('a',20.0,20,'b'))


  def test19ObjectIdentity(self) :
    c1 = self.A.B.C.Calling()
    c2 = self.A.B.C.Calling()
    # PyROOT objects have no exposed object _theObject (b/c of performance)
    # AddressOf() yields a ptr-to-ptr ('**'), hence deref[0] gives address as long
    #self.failUnless(c1.retByPointer()._theObject == c1.retByReference()._theObject)
    #self.failUnless(c1.retByPointer()._theObject != c2.retByPointer()._theObject)
    self.failUnless(PyCintex.addressOf(c1.retByPointer()) == PyCintex.addressOf(c1.retByReference()))
    self.failUnless(PyCintex.addressOf(c1.retByPointer())!= PyCintex.addressOf(c2.retByPointer()))

  def test20AbstractInterfaces(self) :
    d = self.gbl.I_get()
    self.failUnless(d)
    self.gbl.I_set(d)
    
  def test21TemplatedWithString(self) :
    p = self.gbl.SimpleProperty('string','Verifier<string>')('hello')
    self.failUnless(p)
    del p
 
  def test22VectorPointers(self) :
    v = self.gbl.std.vector('const Pbase*')()
    self.failUnless(v)
    coll = [ self.gbl.PPbase(i) for i in range(10) ]
    for p in coll : v.push_back(p)
    self.failUnlessEqual(len(v),10)
    p = v.at(1)
    self.failUnlessEqual(p.get(),1)

  def test23ExceptionsInCPPCode(self) :
    g = self.gbl.ExceptionGenerator(False)  # should not throw exception
    g.doThrow(False)                        # should not thoow exception
    self.assertRaises(RuntimeError, g.doThrow, True)
    self.assertRaises(RuntimeError, g.intThrow, True)
    self.assertRaises(TypeError, self.gbl.ExceptionGenerator, True)

  def test24STLArgTypes(self):
    p = self.A.B.C.Primitives()
    self.failUnless( isinstance(p, self.A.B.C.Primitives) )
    v = self.std.vector('double')()
    v.push_back(1.0)
    p.set_doubles('a', v)
    self.failUnlessEqual(p.doubles().size(), 1 )
    self.failUnlessEqual(p.doubles()[0], 1.0 )

  def test25STLIterator(self):
    vector = PyCintex.makeClass('std::vector<MyA>')
    self.failUnless( vector )
    self.failUnless( PyCintex.makeClass('std::vector<MyA>::iterator') )
    self.failUnless( PyCintex.makeClass('std::vector<MyA>::reverse_iterator') )
	
  def test26TypedefRet(self):
    self.failUnlessEqual( self.gbl.MyNS.theFunction(), 1)
    self.failUnlessEqual( self.gbl.theFunction(), 1 )  
   
  def test27Enums(self):
    self.failUnlessEqual( self.gbl.one, 1)
    self.failUnlessEqual( self.gbl.MyNS.one, 1)
    self.failUnlessEqual( self.gbl.MyClass1.one, 1)
    self.failUnlessEqual( self.gbl.MyClass2.one, 1)
    self.failUnlessEqual( self.gbl.MyClass3.one, 1)
    self.failUnlessEqual( self.gbl.MyClass4.one, 1)
    self.failUnless('unknown' not in str(self.gbl.MyClass1()) )
    self.failUnless('unknown' not in str(self.gbl.MyClass2()) )
    self.failUnless('unknown' not in str(self.gbl.MyClass3()) )
    self.failUnless('unknown' not in str(self.gbl.MyClass4()) )

  def test28PrimitiveArgumentsByReference(self):
    c = PyCintex.libPyROOT.Double(10.0+0.0)
    d = PyCintex.libPyROOT.Double(c)
    calling = self.A.B.C.Calling()
    self.failUnlessEqual( calling.GetByPrimitiveReference(c), 10.0 )
    self.failUnlessEqual( c, 999.99 )
    
  def test29MarcoClemencic(self) :
    a = self.gbl.MarcoCl.MyClass()
    i = 0
    self.failUnlessEqual( a.echo("hi there!"), 1)
    self.failUnlessEqual( a.echo(i), 2)

  def test30VectorArguments(self) :
    calling = self.A.B.C.Calling()
    self.gbl.gEnv.SetValue("Root.ErrorIgnoreLevel", "Error")
    self.failUnlessEqual(calling.vectorargument(self.std.vector('double')(3)), 3)
    self.failUnlessEqual(calling.vectorargument(self.std.vector('unsigned long')(4)), 4)
    self.failUnlessEqual(calling.vectorargument(self.std.vector('string')(2)), 2)
    #self.gbl.gEnv.SetValue("Root.ErrorIgnoreLevel", "Warning")
   
  def test31VectorPrimitiveElementAsignments(self) :
    # Not working for 'char', 'short', 'unsigned long',...
    for p in (('int',0, 66) , ('long',0, 77),  
             ('float', 0., 22.0), ('double', 0., 33.3) ) :
      v = self.std.vector(p[0])(3)
      self.failUnlessEqual(v[2], p[1])
      v[2] = p[2]
      self.failUnlessEqual(v[2], p[2])

  def test32CannotfindShowMembers(self) :
    obj = self.gbl.TrackingRecHit()
    self.failUnless(self)
    self.failUnless(hasattr(obj,'ShowMembers'))

suite = unittest.makeSuite(BasicsTestCase,'test')
if __name__ == '__main__':
  ret = unittest.TextTestRunner( sys.stdout, verbosity = 2 ).run(suite)
  raise SystemExit, not ret.wasSuccessful()
   
    
  #unittest.main()

