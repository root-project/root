# @(#)root/gdml:$Id$
# Author: Witold Pokorski   05/06/2006
# This is the application-independent part of the GDML 'writer' implementation.
# It contains the 'writeFile' method (at the end of the file) which does the actual
# formating and writing out of the GDML file as well as the specialized 'add-element'
# methods for all the supported GDML elements. These methods are used to build
# the content of the GDML document, which is then generatd using the 'writeFile' method.

# The constructor of this class takes the output file (.gdml) as argument.
# An instance of this class should be passed to the constructor of application-specific
# 'writer binding' (in the present case ROOTwriter) as argument.

# For any question or remarks concerning this code, please send an email to
# Witold.Pokorski@cern.ch.

class writer(object):

    def __init__(self, fname):

        self.gdmlfile = fname
        self.define = ['define',{},[]]
        self.materials = ['materials',{},[]]
        self.solids = ['solids',{},[]]
        self.structure = ['structure',{},[]]
        self.document = ['gdml',{'xmlns:gdml':"http://cern.ch/2001/Schemas/GDML",
                                 'xmlns:xsi':"http://www.w3.org/2001/XMLSchema-instance",
                                 'xsi:noNamespaceSchemaLocation':"gdml.xsd"},
                         [self.define, self.materials, self.solids, self.structure]]

    def addPosition(self, name, x, y, z):
        self.define[2].append(['position',{'name':name, 'x':x, 'y':y, 'z':z, 'unit':'cm'},[]])

    def addRotation(self, name, x, y, z):
        self.define[2].append(['rotation',{'name':name, 'x':x, 'y':y, 'z':z, 'unit':'deg'},[]])

    def addMaterial(self, name, a, z, rho):
        self.materials[2].append(['material', {'name':name, 'Z':z},
                                           [['D',{'value':rho},[]], ['atom',{'value':a},[]]] ] )

    def addMixture(self, name, rho, elems):
        subel = [ ['D',{'value':rho},[]] ]
        for el in elems.keys():
            subel.append(['fraction',{'n':elems[el],'ref':el}, []])

        self.materials[2].append(['material',{'name':name},
                                           subel])

    def addElement(self, symb, name, z, a):
        self.materials[2].append(['element', {'name':name, 'formula':symb, 'Z':z},
                                  [['atom', {'value':a},[]] ]])

    def addReflSolid(self, name, solid, dx, dy, dz, sx, sy, sz, rx, ry, rz):
        self.solids[2].append(['reflectedSolid',{'name':name, 'solid':solid, 'dx':dx, 'dy':dy, 'dz':dz, 'sx':sx, 'sy':sy, 'sz':sz, 'rx':rx, 'ry':ry, 'rz':rz},[]])

    def addBox(self, name, dx, dy, dz):
        self.solids[2].append(['box',{'name':name, 'x':dx, 'y':dy, 'z':dz, 'lunit':'cm'},[]])
	
    def addParaboloid(self, name, rlo, rhi, dz):
        self.solids[2].append(['paraboloid',{'name':name, 'rlo':rlo, 'rhi':rhi, 'dz':dz, 'lunit':'cm'},[]])
	
    def addArb8(self, name, v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y, v5x, v5y, v6x, v6y, v7x, v7y, v8x, v8y, dz):
        self.solids[2].append(['arb8',{'name':name, 'v1x':v1x, 'v1y':v1y, 'v2x':v2x, 'v2y':v2y, 'v3x':v3x, 'v3y':v3y, 'v4x':v4x, 'v4y':v4y, 'v5x':v5x, 'v5y':v5y, 'v6x':v6x, 'v6y':v6y, 'v7x':v7x, 'v7y':v7y, 'v8x':v8x, 'v8y':v8y, 'dz':dz, 'lunit':'cm'},[]])

    def addSphere(self, name, rmin, rmax, startphi, deltaphi, starttheta, deltatheta):
        self.solids[2].append(['sphere',{'name':name, 'rmin':rmin, 'rmax':rmax,
                                         'startphi':startphi, 'deltaphi':deltaphi,
                                         'starttheta':starttheta, 'deltatheta':deltatheta,
                                         'aunit':'deg', 'lunit':'cm'},[]])

    def addCone(self, name, z, rmin1, rmin2, rmax1, rmax2, sphi, dphi):
        self.solids[2].append(['cone',{'name':name, 'z':z, 'rmin1':rmin1, 'rmin2':rmin2,
                                       'rmax1':rmax1, 'rmax2':rmax2,
                                       'startphi':sphi, 'deltaphi':dphi, 'lunit':'cm', 'aunit':'deg'}, []] )

    def addPara(self, name, x, y, z, alpha, theta, phi):
        self.solids[2].append(['para',{'name':name, 'x':x, 'y':y, 'z':z,
                                       'alpha':alpha, 'theta':theta, 'phi':phi, 'lunit':'cm', 'aunit':'deg'}, []] )

    def addTrap(self, name, z, theta, phi, y1, x1, x2, alpha1, y2, x3, x4, alpha2):
        self.solids[2].append(['trap', {'name':name, 'z':z, 'theta':theta, 'phi':phi,
                                        'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4,
                                        'y1':y1, 'y2':y2, 'alpha1':alpha1, 'alpha2':alpha2, 'lunit':'cm', 'aunit':'deg'}, []])
					
    def addTwistedTrap(self, name, z, theta, phi, y1, x1, x2, alpha1, y2, x3, x4, alpha2, twist):
        self.solids[2].append(['twistTrap', {'name':name, 'z':z, 'theta':theta, 'phi':phi,
                                             'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4,
                                             'y1':y1, 'y2':y2, 'alpha1':alpha1, 'alpha2':alpha2, 'twist':twist, 'aunit':'deg', 'lunit':'cm'}, []])

    def addTrd(self, name, x1, x2, y1, y2, z):
        self.solids[2].append(['trd',{'name':name, 'x1':x1, 'x2':x2,
                                      'y1':y1, 'y2':y2, 'z':z, 'lunit':'cm'}, []])

    def addTube(self, name, rmin, rmax, z, startphi, deltaphi):
        self.solids[2].append(['tube',{'name':name, 'rmin':rmin, 'rmax':rmax,
                                       'z':z, 'startphi':startphi, 'deltaphi':deltaphi, 'lunit':'cm', 'aunit':'deg'},[]])
				       
    def addCutTube(self, name, rmin, rmax, z, startphi, deltaphi, lowX, lowY, lowZ, highX, highY, highZ):
        self.solids[2].append(['cutTube',{'name':name, 'rmin':rmin, 'rmax':rmax,
                                          'z':z, 'startphi':startphi, 'deltaphi':deltaphi,
					  'lowX':lowX, 'lowY':lowY, 'lowZ':lowZ, 'highX':highX, 'highY':highY, 'highZ':highZ, 'lunit':'cm', 'aunit':'deg'},[]])

    def addPolycone(self, name, startphi, deltaphi, zplanes):
        zpls = []
        for zplane in zplanes:
            zpls.append( ['zplane',{'z':zplane[0], 'rmin':zplane[1], 'rmax':zplane[2]},[]] )
        self.solids[2].append(['polycone',{'name':name,
                                           'startphi':startphi, 'deltaphi':deltaphi, 'lunit':'cm', 'aunit':'deg'}, zpls])

    def addTorus(self, name, r, rmin, rmax, startphi, deltaphi):
        self.solids[2].append( ['torus',{'name':name, 'rtor':r, 'rmin':rmin, 'rmax':rmax,
                                         'startphi':startphi, 'deltaphi':deltaphi, 'lunit':'cm', 'aunit':'deg'},[]] )

    def addPolyhedra(self, name, startphi, deltaphi, numsides, zplanes):
        zpls = []
        for zplane in zplanes:
            zpls.append( ['zplane',{'z':zplane[0], 'rmin':zplane[1], 'rmax':zplane[2]},[]] )
        self.solids[2].append(['polyhedra',{'name':name,
                                            'startphi':startphi, 'deltaphi':deltaphi,
                                            'numsides':numsides, 'lunit':'cm', 'aunit':'deg'}, zpls])
					    
    def addXtrusion(self, name, vertices, sections):
        elems = []
	for vertex in vertices:
	    elems.append( ['twoDimVertex',{'x':vertex[0], 'y':vertex[1]},[]] )
	for section in sections:
	    elems.append( ['section',{'zOrder':section[0], 'zPosition':section[1], 'xOffset':section[2], 'yOffset':section[3], 'scalingFactor':section[4]},[]] )
	self.solids[2].append(['xtru',{'name':name, 'lunit':'cm'}, elems])

    def addEltube(self, name, x, y, z):
        self.solids[2].append( ['eltube', {'name':name, 'dx':x, 'dy':y, 'dz':z, 'lunit':'cm'},[]] )

    def addHype(self, name, rmin, rmax, inst, outst, z):
        self.solids[2].append( ['hype', {'name':name, 'rmin':rmin, 'rmax':rmax,
                                         'inst':inst, 'outst':outst, 'z':z, 'lunit':'cm', 'aunit':'deg'},[]] )

    def addPos(self, subels, type, name, v):
        if v[0]!=0.0 or v[1]!=0.0 or v[2]!=0.0:
            subels.append( [type,{'name':name, 'x':v[0], 'y':v[1], 'z':v[2], 'unit':'cm'},[]] )

    def addRot(self, subels, type, name, v):
        if v[0]!=0.0 or v[1]!=0.0 or v[2]!=0.0:
            subels.append( [type,{'name':name, 'x':v[0], 'y':v[1], 'z':v[2], 'unit':'deg'},[]] )

    def addUnion(self, name, lname, ltr, lrot, rname, rtr, rrot):
        subels = [['first',{'ref':lname},[]],
                ['second',{'ref':rname},[]]]
        self.addPos(subels, 'position', rname+'pos', rtr)
        self.addRot(subels, 'rotation', rname+'rot', rrot)
        self.addPos(subels, 'firstposition', lname+'pos', ltr)
        self.addRot(subels, 'firstrotation', lname+'rot', lrot)
        self.solids[2].append( ['union',{'name':name}, subels])

    def addSubtraction(self, name, lname, ltr, lrot, rname, rtr, rrot):
        subels = [['first',{'ref':lname},[]],
                  ['second',{'ref':rname},[]]]
        self.addPos(subels, 'position', rname+'pos', rtr)
        self.addRot(subels, 'rotation', rname+'rot', rrot)
        self.addPos(subels, 'firstposition', lname+'pos', ltr)
        self.addRot(subels, 'firstrotation', lname+'rot', lrot)
        self.solids[2].append( ['subtraction',{'name':name}, subels])

    def addIntersection(self, name, lname, ltr, lrot, rname, rtr, rrot):
        subels = [['first',{'ref':lname},[]],
                  ['second',{'ref':rname},[]]]
        self.addPos(subels, 'position', rname+'pos', rtr)
        self.addRot(subels, 'rotation', rname+'rot', rrot)
        self.addPos(subels, 'firstposition', lname+'pos', ltr)
        self.addRot(subels, 'firstrotation', lname+'rot', lrot)
        self.solids[2].append( ['intersection',{'name':name}, subels])

    def addVolume(self, volume, solid, material, daughters):
        subels = [['materialref',{'ref':material},[]],
                  ['solidref',{'ref':solid},[]]]
        for child in daughters:
            subsubels = [['volumeref',{'ref':child[0]},[]],
                         ['positionref',{'ref':child[1]},[]]]
            if child[2]!='':
                subsubels.append( ['rotationref',{'ref':child[2]},[]] )

            subels.append( ['physvol',{}, subsubels])

        used = 0
        self.structure[2].append(['volume',{'name':volume}, subels, used])

    def addAssembly(self, volume, daughters):
        subels = []
        for child in daughters:
            subsubels = [['volumeref',{'ref':child[0]},[]],
                         ['positionref',{'ref':child[1]},[]]]
            if child[2]!='':
                subsubels.append( ['rotationref',{'ref':child[2]},[]] )

            subels.append( ['physvol',{}, subsubels])

        self.structure[2].append(['assembly',{'name':volume}, subels])

    def addSetup(self, name, version, world):
        self.document[2].append( ['setup',{'name':name, 'version':version},
                                   [ ['world',{'ref':world},[]]] ] )

    def writeFile(self):
        file = open(self.gdmlfile,'w')
        offset = ''

        def writeElement(elem, offset):
            offset = offset + '  '
            file.write(offset+'<%s' %(elem[0]))
            for attr in elem[1].keys():
                file.write(' %s="%s"' %(attr,elem[1][attr]))
            if elem[2].__len__()>0:
                file.write('>\n')
                for subel in elem[2]:
                    writeElement(subel, offset)

                file.write(offset+'</%s>\n' %(elem[0]))
            else:
                file.write('/>\n')

        file.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
        writeElement(self.document,'')
	

