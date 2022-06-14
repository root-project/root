# @(#)root/gdml:$Id$
# Author: Witold Pokorski   05/06/2006

from math import *

import libPyROOT
import ROOT
import math
import re

# This class provides ROOT binding for the 'writer' class. It implements specific
# methods for all the supported TGeo classes which call the appropriate 'add-element'
# methods from the 'writer' class.

# The list of presently supported classes is the following:

# Materials:
# TGeoElement
# TGeoMaterial
# GeoMixture

# Solids:
# TGeoBBox
# TGeoArb8
# TGeoTubeSeg
# TGeoConeSeg
# TGeoCtub
# TGeoPcon
# TGeoTrap
# TGeoGtra
# TGeoTrd2
# TGeoSphere
# TGeoPara
# TGeoTorus
# TGeoHype
# TGeoPgon
# TGeoXtru
# TGeoEltu
# TGeoParaboloid
# TGeoCompositeShape (subtraction, union, intersection)

# Geometry:
# TGeoVolume

# In addition the class contains three methods 'dumpMaterials', 'dumpSolids' and 'examineVol'
# which retrieve from the memory the materials, the solids and the geometry tree
# respectively. The user should instanciate this class passing and instance of 'writer'
# class as argument. In order to export the geometry in the form of a GDML file,
# the three methods (dumpMaterials, dumpSolids and examineVol) should be called.
# The argument of 'dumpMaterials' method should be the list of materials,
# the argument of the 'dumpSolids' method should be the list of solids and
# the argument of the 'examineVol' method should be the top volume of
# the geometry tree.

# For any question or remarks concerning this code, please send an email to
# Witold.Pokorski@cern.ch.

class ROOTwriter(object):

    def __init__(self, writer):
        self.writer = writer
        self.elements = []
	self.volumeCount = 0
	self.nodeCount = 0
	self.shapesCount = 0	
	self.bvols = []
	self.vols = []
	self.volsUseCount = {}
	self.sortedVols = []
	self.nodes = []
	self.bnodes = []
	self.solList = []
        self.geomgr = ROOT.gGeoManager
        self.geomgr.SetAllIndex()
        pass



    def genName(self, name):

        re.sub('$', '', name)

        return name


    def rotXYZ(self, r):

        rad = 180/acos(-1)
        
        cosb = math.sqrt( r[0]*r[0] + r[1]*r[1] )
        if cosb > 0.00001 : #I didn't find a proper constant to use here, so I just put a value that works with all the examples on a linux machine (P4)
            a = atan2( r[5], r[8] ) * rad
            b = atan2( -r[2], cosb ) * rad
            c = atan2( r[1], r[0] ) * rad
        else:
            a = atan2( -r[7], r[4] ) * rad
            b = atan2( -r[2], cosb ) * rad
            c = 0.
        return (a, b, c)

    def TGeoBBox(self, solid):
        self.writer.addBox(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), 2*solid.GetDX(), 2*solid.GetDY(), 2*solid.GetDZ())

    def TGeoParaboloid(self, solid):
        self.writer.addParaboloid(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetRlo(), solid.GetRhi(), solid.GetDz())

    def TGeoSphere(self, solid):
        self.writer.addSphere(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetRmin(), solid.GetRmax(),
                             solid.GetPhi1(), solid.GetPhi2() - solid.GetPhi1(),
                             solid.GetTheta1(), solid.GetTheta2() - solid.GetTheta1())
			 
    def TGeoArb8(self, solid):
        self.writer.addArb8(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]),
			    solid.GetVertices()[0],
			    solid.GetVertices()[1],
			    solid.GetVertices()[2],
			    solid.GetVertices()[3],
			    solid.GetVertices()[4],
			    solid.GetVertices()[5],
			    solid.GetVertices()[6],
			    solid.GetVertices()[7],
			    solid.GetVertices()[8],
			    solid.GetVertices()[9],
			    solid.GetVertices()[10],
			    solid.GetVertices()[11],
			    solid.GetVertices()[12],
			    solid.GetVertices()[13],
			    solid.GetVertices()[14],
			    solid.GetVertices()[15],
			    solid.GetDz())

    def TGeoConeSeg(self, solid):
        self.writer.addCone(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), 2*solid.GetDz(), solid.GetRmin1(), solid.GetRmin2(),
                           solid.GetRmax1(), solid.GetRmax2(), solid.GetPhi1(), solid.GetPhi2() - solid.GetPhi1())

    def TGeoCone(self, solid):
        self.writer.addCone(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), 2*solid.GetDz(), solid.GetRmin1(), solid.GetRmin2(), 
	                   solid.GetRmax1(), solid.GetRmax2(), 0, 360)

    def TGeoPara(self, solid):
        self.writer.addPara(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetX(), solid.GetY(), solid.GetZ(),
                           solid.GetAlpha(), solid.GetTheta(), solid.GetPhi())

    def TGeoTrap(self, solid):
        self.writer.addTrap(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), 2*solid.GetDz(), solid.GetTheta(), solid.GetPhi(),
                           2*solid.GetH1(), 2*solid.GetBl1(), 2*solid.GetTl1(), solid.GetAlpha1(),
                           2*solid.GetH2(), 2*solid.GetBl2(), 2*solid.GetTl2(), solid.GetAlpha2())
			   
    def TGeoGtra(self, solid):
        self.writer.addTwistedTrap(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), 2*solid.GetDz(), solid.GetTheta(), solid.GetPhi(),
                                   2*solid.GetH1(), 2*solid.GetBl1(), 2*solid.GetTl1(), solid.GetAlpha1(),
                                   2*solid.GetH2(), 2*solid.GetBl2(), 2*solid.GetTl2(), solid.GetAlpha2(), solid.GetTwistAngle())

    def TGeoTrd1(self, solid):
        self.writer.addTrd(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), 2*solid.GetDx1(), 2*solid.GetDx2(), 2*solid.GetDy(),
                          2*solid.GetDy(), 2*solid.GetDz())
    
    def TGeoTrd2(self, solid):
        self.writer.addTrd(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), 2*solid.GetDx1(), 2*solid.GetDx2(), 2*solid.GetDy1(),
                          2*solid.GetDy2(), 2*solid.GetDz())

    def TGeoTubeSeg(self, solid):
        self.writer.addTube(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetRmin(), solid.GetRmax(),
                           2*solid.GetDz(), solid.GetPhi1(), solid.GetPhi2()-solid.GetPhi1())
			   
    def TGeoCtub(self, solid):
        self.writer.addCutTube(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetRmin(), solid.GetRmax(),
                              2*solid.GetDz(), solid.GetPhi1(), solid.GetPhi2()-solid.GetPhi1(),
			      solid.GetNlow()[0],
			      solid.GetNlow()[1],
			      solid.GetNlow()[2],
			      solid.GetNhigh()[0],
			      solid.GetNhigh()[1],
			      solid.GetNhigh()[2])
			   
    def TGeoTube(self, solid):
        self.writer.addTube(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetRmin(), solid.GetRmax(),
                           2*solid.GetDz(), 0, 360)

    def TGeoPcon(self, solid):
        zplanes = []
        for i in range(solid.GetNz()):
            zplanes.append( (solid.GetZ(i), solid.GetRmin(i), solid.GetRmax(i)) )
        self.writer.addPolycone(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetPhi1(), solid.GetDphi(), zplanes)

    def TGeoTorus(self, solid):
        self.writer.addTorus(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetR(), solid.GetRmin(), solid.GetRmax(),
                            solid.GetPhi1(), solid.GetDphi())

    def TGeoPgon(self, solid):
        zplanes = []
        for i in range(solid.GetNz()):
            zplanes.append( (solid.GetZ(i), solid.GetRmin(i), solid.GetRmax(i)) )
        self.writer.addPolyhedra(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetPhi1(), solid.GetDphi(),
                                solid.GetNedges(), zplanes)
				
    def TGeoXtru(self, solid):
        vertices = []
	sections = []
        for i in range(solid.GetNvert()):
            vertices.append( (solid.GetX(i), solid.GetY(i)) )
	for i in range(solid.GetNz()):
            sections.append( (i, solid.GetZ(i), solid.GetXOffset(i), solid.GetYOffset(i), solid.GetScale(i)) )    
        self.writer.addXtrusion(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), vertices, sections)

    def TGeoEltu(self, solid):
        self.writer.addEltube(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetA(), solid.GetB(), solid.GetDz())

    def TGeoHype(self, solid):
        self.writer.addHype(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]), solid.GetRmin(), solid.GetRmax(),
                           solid.GetStIn(), solid.GetStOut(), 2*solid.GetDz())

    def TGeoUnion(self, solid):
        lrot = self.rotXYZ(solid.GetBoolNode().GetLeftMatrix().Inverse().GetRotationMatrix())
        rrot = self.rotXYZ(solid.GetBoolNode().GetRightMatrix().Inverse().GetRotationMatrix())
	
	if ([solid.GetBoolNode().GetLeftShape(), 0]) in self.solList:
	    self.solList[self.solList.index([solid.GetBoolNode().GetLeftShape(), 0])][1] = 1
            eval('self.'+solid.GetBoolNode().GetLeftShape().__class__.__name__)(solid.GetBoolNode().GetLeftShape())
	    self.shapesCount = self.shapesCount + 1
	if ([solid.GetBoolNode().GetRightShape(), 0]) in self.solList:
	    self.solList[self.solList.index([solid.GetBoolNode().GetRightShape(), 0])][1] = 1
            eval('self.'+solid.GetBoolNode().GetRightShape().__class__.__name__)(solid.GetBoolNode().GetRightShape())
	    self.shapesCount = self.shapesCount + 1

        self.writer.addUnion(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]),
                            solid.GetBoolNode().GetLeftShape().GetName()+'_'+str(libPyROOT.AddressOf(solid.GetBoolNode().GetLeftShape())[0]),
                            solid.GetBoolNode().GetLeftMatrix().GetTranslation(),
                            lrot,
                            solid.GetBoolNode().GetRightShape().GetName()+'_'+str(libPyROOT.AddressOf(solid.GetBoolNode().GetRightShape())[0]),
                            solid.GetBoolNode().GetRightMatrix().GetTranslation(),
                            rrot)

    def TGeoIntersection(self, solid):
        lrot = self.rotXYZ(solid.GetBoolNode().GetLeftMatrix().Inverse().GetRotationMatrix())
        rrot = self.rotXYZ(solid.GetBoolNode().GetRightMatrix().Inverse().GetRotationMatrix())
	
	if ([solid.GetBoolNode().GetLeftShape(), 0]) in self.solList:
	    self.solList[self.solList.index([solid.GetBoolNode().GetLeftShape(), 0])][1] = 1
            eval('self.'+solid.GetBoolNode().GetLeftShape().__class__.__name__)(solid.GetBoolNode().GetLeftShape())
	    self.shapesCount = self.shapesCount + 1
	if ([solid.GetBoolNode().GetRightShape(), 0]) in self.solList:
	    self.solList[self.solList.index([solid.GetBoolNode().GetRightShape(), 0])][1] = 1
            eval('self.'+solid.GetBoolNode().GetRightShape().__class__.__name__)(solid.GetBoolNode().GetRightShape())
	    self.shapesCount = self.shapesCount + 1

        self.writer.addIntersection(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]),
                                   solid.GetBoolNode().GetLeftShape().GetName()+'_'+str(libPyROOT.AddressOf(solid.GetBoolNode().GetLeftShape())[0]),
                                   solid.GetBoolNode().GetLeftMatrix().GetTranslation(),
                                   lrot,
                                   solid.GetBoolNode().GetRightShape().GetName()+'_'+str(libPyROOT.AddressOf(solid.GetBoolNode().GetRightShape())[0]),
                                   solid.GetBoolNode().GetRightMatrix().GetTranslation(),
                                   rrot)

    def TGeoSubtraction(self, solid):
        lrot = self.rotXYZ(solid.GetBoolNode().GetLeftMatrix().Inverse().GetRotationMatrix())
        rrot = self.rotXYZ(solid.GetBoolNode().GetRightMatrix().Inverse().GetRotationMatrix())	
	
        if ([solid.GetBoolNode().GetLeftShape(), 0]) in self.solList:
	    self.solList[self.solList.index([solid.GetBoolNode().GetLeftShape(), 0])][1] = 1
            eval('self.'+solid.GetBoolNode().GetLeftShape().__class__.__name__)(solid.GetBoolNode().GetLeftShape())
	    self.shapesCount = self.shapesCount + 1
	if ([solid.GetBoolNode().GetRightShape(), 0]) in self.solList:
	    self.solList[self.solList.index([solid.GetBoolNode().GetRightShape(), 0])][1] = 1
            eval('self.'+solid.GetBoolNode().GetRightShape().__class__.__name__)(solid.GetBoolNode().GetRightShape())
	    self.shapesCount = self.shapesCount + 1

        self.writer.addSubtraction(self.genName(solid.GetName())+'_'+str(libPyROOT.AddressOf(solid)[0]),
                                  solid.GetBoolNode().GetLeftShape().GetName()+'_'+str(libPyROOT.AddressOf(solid.GetBoolNode().GetLeftShape())[0]),
                                  solid.GetBoolNode().GetLeftMatrix().GetTranslation(),
                                  lrot,
                                  solid.GetBoolNode().GetRightShape().GetName()+'_'+str(libPyROOT.AddressOf(solid.GetBoolNode().GetRightShape())[0]),
                                  solid.GetBoolNode().GetRightMatrix().GetTranslation(),
                                  rrot)

    def TGeoCompositeShape(self, solid):
        eval('self.'+solid.GetBoolNode().__class__.__name__)(solid)

    def dumpMaterials(self, matlist):
        print 'Info in <TPython::Exec>: Found ', matlist.GetSize(),' materials'
        for mat in matlist:
            if not mat.IsMixture():
                self.writer.addMaterial(self.genName(mat.GetName()), mat.GetA(), mat.GetZ(), mat.GetDensity())
            else:
                elems = {}
                for index in range(mat.GetNelements()):
                    elems[mat.GetElement(index).GetName()] = mat.GetWmixt()[index]
                    el = mat.GetElement(index)
                    if el not in self.elements:
                        self.elements.append(el)
                        self.writer.addElement(mat.GetElement(index).GetTitle(), mat.GetElement(index).GetName(), mat.GetZmixt()[index], mat.GetAmixt()[index])
			
                self.writer.addMixture(self.genName(mat.GetName()), mat.GetDensity(), elems)

    def dumpSolids(self, shapelist):
        print 'Info in <TPython::Exec>: Found ', shapelist.GetEntries(), ' shapes'
        for shape in shapelist:
	    self.solList.append([shape, 0])
	for sol in self.solList:
	    if sol[1] == 0:
	        sol[1] = 1
                #print eval('self.'+sol[0].__class__.__name__)(sol[0])
                eval('self.'+sol[0].__class__.__name__)(sol[0])
	        self.shapesCount = self.shapesCount + 1
	print 'Info in <TPython::Exec>: Dumped ', self.shapesCount, ' shapes'
	    
    
    def orderVolumes(self, volume):
        index = str(volume.GetNumber())+"_"+str(libPyROOT.AddressOf(volume)[0])
	daughters = volume.GetNodes()
	if len(self.sortedVols)<len(self.vols) and self.volsUseCount[index]>0:
	    self.volsUseCount[index] = self.volsUseCount[index]-1
	    if self.volsUseCount[index]==0:
	        self.sortedVols.append(volume)
	    if daughters:
	        for node in daughters:
	            self.orderVolumes(node.GetVolume())
		    self.nodeCount = self.nodeCount+1
	            if self.nodeCount%10000==0:
	                 print '[FIRST STAGE] Node count: ', self.nodeCount
        elif len(self.sortedVols)<len(self.volsUseCount) and self.volsUseCount[index]==0:
	    self.sortedVols.append(volume)
	    if daughters:
	        for node in daughters:
	            self.orderVolumes(node.GetVolume())
		    self.nodeCount = self.nodeCount+1
	            if self.nodeCount%10000==0:
	                 print '[FIRST STAGE] Node count: ', self.nodeCount
           
    def getNodes(self, volume):
        nd = volume.GetNdaughters()
	if nd:
	    for i in range(nd):
	        currentNode = volume.GetNode(i)
	        nextVol = currentNode.GetVolume()
		index = str(nextVol.GetNumber())+"_"+str(libPyROOT.AddressOf(nextVol)[0])
		self.volsUseCount[index] = self.volsUseCount[index]+1
                self.nodes.append(currentNode)
	        self.getNodes(nextVol)
		self.nodeCount = self.nodeCount+1
	        if self.nodeCount%10000==0:
	            print '[ZEROTH STAGE] Analysing node: ', self.nodeCount
    
    def examineVol2(self, volume): #use with geometries containing many volumes
        print ''
	print '[RETRIEVING VOLUME LIST]'

	self.bvols = geomgr.GetListOfVolumes()
	print ''
	print '[INITIALISING VOLUME USE COUNT]'
        for vol in self.bvols:
	    self.vols.append(vol)
	    self.volsUseCount[str(vol.GetNumber())+"_"+str(libPyROOT.AddressOf(vol)[0])]=0
	print ''
	print '[CALCULATING VOLUME USE COUNT]'
	self.nodeCount = 0
	self.getNodes(volume)
	print ''
	print '[ORDERING VOLUMES]'
	self.nodeCount = 0
        self.orderVolumes(volume)
	print ''
	print '[DUMPING GEOMETRY TREE]'
	self.sortedVols.reverse()
	self.nodeCount = 0
        self.dumpGeoTree()
	print ''
	print '[FINISHED!]'
	print ''
    
    def examineVol(self, volume): #use with geometries containing very few volumes and many nodes
        daughters = []
        if volume.GetNodes():
            for node in volume.GetNodes():
                subvol = node.GetVolume()

                #if bit not set, set and save primitive
                if not subvol.TestAttBit(524288): #value referring to TGeoAtt::kSavePrimitiveAtt (1 << 19)
                    subvol.SetAttBit(524288) 
                    self.vols.append(subvol)
                    self.examineVol(subvol)
                name = node.GetName()+str(libPyROOT.AddressOf(subvol)[0])+'in'+volume.GetName()+str(libPyROOT.AddressOf(volume)[0])
                pos = node.GetMatrix().GetTranslation()
                self.writer.addPosition(name+'pos', pos[0], pos[1], pos[2])
                r = self.rotXYZ(node.GetMatrix().GetRotationMatrix())
                rotname = ''
                if r[0]!=0.0 or r[1]!=0.0 or r[2]!=0.0:
                    self.writer.addRotation(name+'rot', r[0], r[1], r[2])
                    rotname = name+'rot'

                reflection = node.GetMatrix().IsReflection()#check if this daughter has a reflection matrix
                
                if reflection:
                    rotmat =  node.GetMatrix().GetRotationMatrix()

                    #add new 'reflectedSolid' shape to solids
                    self.writer.addReflSolid('refl_'+node.GetVolume().GetShape().GetName()+'_'+str(libPyROOT.AddressOf(node.GetVolume().GetShape())[0]), node.GetVolume().GetShape().GetName()+'_'+str(libPyROOT.AddressOf(node.GetVolume().GetShape())[0]), 0, 0, 0, rotmat[0], rotmat[4], rotmat[8], 0, 0, 0)

                    #add new volume with correct solidref to the new reflectedSolid
                    emptyd = []
                    self.writer.addVolume('refl_'+node.GetVolume().GetName()+'_'+str(libPyROOT.AddressOf(node.GetVolume())[0]), 'refl_'+node.GetVolume().GetShape().GetName()+'_'+str(libPyROOT.AddressOf(node.GetVolume().GetShape())[0]), self.genName(node.GetVolume().GetMaterial().GetName()), emptyd)
                    
                    #add new volume as volumeref to this physvol
                    daughters.append( ('refl_'+node.GetVolume().GetName()+'_'+str(libPyROOT.AddressOf(node.GetVolume())[0]), name+'pos', rotname) )
                    
                else:
                    daughters.append( (node.GetVolume().GetName()+'_'+str(libPyROOT.AddressOf(node.GetVolume())[0]), name+'pos', rotname) )

        if volume.IsTopVolume():
	   if not volume.IsAssembly():
	       self.writer.addVolume(volume.GetName(), volume.GetShape().GetName()+'_'+str(libPyROOT.AddressOf(volume.GetShape())[0]), self.genName(volume.GetMaterial().GetName()), daughters)
	   else:
	       self.writer.addAssembly(volume.GetName(), daughters)
	else: 
	   if not volume.IsAssembly():
	       self.writer.addVolume(volume.GetName()+'_'+str(libPyROOT.AddressOf(volume)[0]), volume.GetShape().GetName()+'_'+str(libPyROOT.AddressOf(volume.GetShape())[0]), self.genName(volume.GetMaterial().GetName()), daughters)
           else:
	       self.writer.addAssembly(volume.GetName()+'_'+str(libPyROOT.AddressOf(volume)[0]), daughters)    
	    
    def dumpGeoTree(self):
	for volume in self.sortedVols:
	    nd = volume.GetNdaughters()
	    daughters = []
	    if nd:
                for i in range(nd):
		    node = volume.GetNode(i)
		    name = node.GetName()+'in'+volume.GetName()
                    pos = node.GetMatrix().GetTranslation()
                    self.writer.addPosition(name+'pos', pos[0], pos[1], pos[2])
	            r = self.rotXYZ(node.GetMatrix().GetRotationMatrix())
                    rotname = ''
                    if r[0]!=0.0 or r[1]!=0.0 or r[2]!=0.0:
                        self.writer.addRotation(name+'rot', r[0], r[1], r[2])
                        rotname = name+'rot'			
                    daughters.append( (node.GetVolume().GetName()+'_'+str(libPyROOT.AddressOf(node.GetVolume())[0]), name+'pos', rotname) )
		    self.nodeCount = self.nodeCount+1
	            if self.nodeCount%100==0:
	                print '[SECOND STAGE] Volume Count: ', self.nodeCount, node.GetVolume().GetName()+'_'+str(libPyROOT.AddressOf(node.GetVolume())[0])
		    	
	    if volume.IsTopVolume():
		if not volume.IsAssembly():
	            self.writer.addVolume(volume.GetName(), volume.GetShape().GetName()+'_'+str(libPyROOT.AddressOf(volume.GetShape())[0]), self.genName(volume.GetMaterial().GetName()), daughters)
		else:
		    self.writer.addAssembly(volume.GetName(), daughters)
	    else: 
		if not volume.IsAssembly():
	            self.writer.addVolume(volume.GetName()+'_'+str(libPyROOT.AddressOf(volume)[0]), volume.GetShape().GetName()+'_'+str(libPyROOT.AddressOf(volume.GetShape())[0]), self.genName(volume.GetMaterial().GetName()), daughters)
                else:
		    self.writer.addAssembly(volume.GetName()+'_'+str(libPyROOT.AddressOf(volume)[0]), daughters)
		
