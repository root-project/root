/// @file JSRootGeoPainter.js
/// JavaScript ROOT 3D geometry painter

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['JSRootPainter', 'THREE_ALL'], factory );
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootGeoPainter.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter is not defined', 'JSRootGeoPainter.js');

      if (typeof THREE == 'undefined')
         throw new Error('THREE is not defined', 'JSRootGeoPainter.js');

      factory(JSROOT);
   }
} (function(JSROOT) {

   if ( typeof define === "function" && define.amd )
      JSROOT.loadScript('$$$style/JSRootGeoPainter.css');

   // === functions to create THREE.Geometry for TGeo shapes ========================

   JSROOT.GEO = {};

   JSROOT.GEO.createCube = function( shape ) {

      // instead of BoxGeometry create all vertices and faces ourself
      // reduce number of allocated objects

      //return new THREE.BoxGeometry( 2*shape.fDX, 2*shape.fDY, 2*shape.fDZ );

      var geom = new THREE.Geometry();

      geom.vertices.push( new THREE.Vector3( shape.fDX,  shape.fDY,  shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3( shape.fDX,  shape.fDY, -shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3( shape.fDX, -shape.fDY,  shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3( shape.fDX, -shape.fDY, -shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3(-shape.fDX,  shape.fDY, -shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3(-shape.fDX,  shape.fDY,  shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3(-shape.fDX, -shape.fDY, -shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3(-shape.fDX, -shape.fDY,  shape.fDZ ) );

      var indicies = [0,2,1, 2,3,1, 4,6,5, 6,7,5, 4,5,1, 5,0,1, 7,6,2, 6,3,2, 5,7,0, 7,2,0, 1,3,4, 3,6,4];

      // normals for each  pair of faces
      var normals = [ 1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1,  0,0,-1 ];

      var color = new THREE.Color();
      var norm = null;
      for (var n=0; n < indicies.length; n+=3) {
          if (n % 6 === 0) norm = new THREE.Vector3(normals[n/2], normals[n/2+1], normals[n/2+2]);
          var face = new THREE.Face3( indicies[n], indicies[n+1], indicies[n+2], norm, color, 0);
          geom.faces.push(face);
      }

      return geom;

   }

   JSROOT.GEO.createPara = function( shape ) {

      var txy = shape.fTxy, txz = shape.fTxz, tyz = shape.fTyz;

      var verticesOfShape = [
          -shape.fZ*txz-txy*shape.fY-shape.fX, -shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz+txy*shape.fY-shape.fX,  shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz+txy*shape.fY+shape.fX,  shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz-txy*shape.fY+shape.fX, -shape.fY-shape.fZ*tyz,  -shape.fZ,
           shape.fZ*txz-txy*shape.fY-shape.fX, -shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz+txy*shape.fY-shape.fX,  shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz+txy*shape.fY+shape.fX,  shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz-txy*shape.fY+shape.fX, -shape.fY+shape.fZ*tyz,   shape.fZ ];

      var indicesOfFaces = [ 4,5,6,   4,7,6,   0,3,7,   7,4,0,
                             4,5,1,   1,0,4,   6,2,1,   1,5,6,
                             7,3,2,   2,6,7,   1,2,3,   3,0,1 ];

      var geom = new THREE.Geometry();

      for (var i = 0; i < verticesOfShape.length; i += 3)
         geom.vertices.push( new THREE.Vector3( verticesOfShape[i], verticesOfShape[i+1], verticesOfShape[i+2] ) );

      var color = new THREE.Color();

      for (var i = 0; i < indicesOfFaces.length; i += 3)
         geom.faces.push( new THREE.Face3( indicesOfFaces[i], indicesOfFaces[i+1], indicesOfFaces[i+2], null, color, 0 ) );

      geom.computeFaceNormals();

      return geom;
   }


   JSROOT.GEO.createTrapezoid = function( shape ) {

      var verticesOfShape;

      if (shape._typename == "TGeoTrd1")
         verticesOfShape = [
            -shape.fDx1,  shape.fDY, -shape.fDZ,
             shape.fDx1,  shape.fDY, -shape.fDZ,
             shape.fDx1, -shape.fDY, -shape.fDZ,
            -shape.fDx1, -shape.fDY, -shape.fDZ,
            -shape.fDx2,  shape.fDY,  shape.fDZ,
             shape.fDx2,  shape.fDY,  shape.fDZ,
             shape.fDx2, -shape.fDY,  shape.fDZ,
            -shape.fDx2, -shape.fDY,  shape.fDZ
         ];
      else
         verticesOfShape = [
            -shape.fDx1,  shape.fDy1, -shape.fDZ,
             shape.fDx1,  shape.fDy1, -shape.fDZ,
             shape.fDx1, -shape.fDy1, -shape.fDZ,
            -shape.fDx1, -shape.fDy1, -shape.fDZ,
            -shape.fDx2,  shape.fDy2,  shape.fDZ,
             shape.fDx2,  shape.fDy2,  shape.fDZ,
             shape.fDx2, -shape.fDy2,  shape.fDZ,
            -shape.fDx2, -shape.fDy2,  shape.fDZ
         ];

      var indicesOfFaces = [
          4,5,6,   4,7,6,   0,3,7,   7,4,0,
          4,5,1,   1,0,4,   6,2,1,   1,5,6,
          7,3,2,   2,6,7,   1,2,3,   3,0,1 ];

      var geometry = new THREE.Geometry();
      for (var i = 0; i < 24; i += 3)
         geometry.vertices.push( new THREE.Vector3( verticesOfShape[i], verticesOfShape[i+1], verticesOfShape[i+2] ) );

      var color = new THREE.Color();

      for (var i = 0; i < 36; i += 3)
         geometry.faces.push( new THREE.Face3( indicesOfFaces[i], indicesOfFaces[i+1], indicesOfFaces[i+2], null, color, 0 ) );

      geometry.computeFaceNormals();
      return geometry;
   }


   JSROOT.GEO.createArb8 = function( shape ) {

      var verticesOfShape = [
            shape.fXY[0][0], shape.fXY[0][1], -shape.fDZ,
            shape.fXY[1][0], shape.fXY[1][1], -shape.fDZ,
            shape.fXY[2][0], shape.fXY[2][1], -shape.fDZ,
            shape.fXY[3][0], shape.fXY[3][1], -shape.fDZ,
            shape.fXY[4][0], shape.fXY[4][1],  shape.fDZ,
            shape.fXY[5][0], shape.fXY[5][1],  shape.fDZ,
            shape.fXY[6][0], shape.fXY[6][1],  shape.fDZ,
            shape.fXY[7][0], shape.fXY[7][1],  shape.fDZ
         ];

      var indicies = [];

      var indicesOfFaces = [
          4,5,6,   4,7,6,   0,3,7,   7,4,0,
          4,5,1,   1,0,4,   6,2,1,   1,5,6,
          7,3,2,   2,6,7,   1,2,3,   3,0,1 ];

      var geometry = new THREE.Geometry();
      for (var i = 0; i < 8; ++i) {
         var ii = i*3;
         if ((i>0) && (verticesOfShape[ii] === verticesOfShape[ii-3]) &&
             (verticesOfShape[ii+1] === verticesOfShape[ii-2]) &&
             (verticesOfShape[ii+2] === verticesOfShape[ii-1])) {
            indicies[i] = indicies[i-1];
            continue;
         }

         indicies[i] = geometry.vertices.length;

         geometry.vertices.push( new THREE.Vector3( verticesOfShape[ii], verticesOfShape[ii+1], verticesOfShape[ii+2] ) );
      }

      var color = new THREE.Color();

      for (var i = 0; i < 36; i += 3) {
         var a = indicies[indicesOfFaces[i]],
             b = indicies[indicesOfFaces[i+1]],
             c = indicies[indicesOfFaces[i+2]];
         if ((a!==b) && (b!==c) && (a!==c))
            geometry.faces.push( new THREE.Face3( a, b, c, null, color, 0 ) );
      }

      geometry.computeFaceNormals();
      return geometry;
   }


   JSROOT.GEO.createSphere = function( shape ) {
      var outerRadius = shape.fRmax;
      var innerRadius = shape.fRmin;
      var phiStart = shape.fPhi1 + 180;
      var phiLength = shape.fPhi2 - shape.fPhi1;
      var thetaStart = shape.fTheta1;
      var thetaLength = shape.fTheta2 - shape.fTheta1;
      var widthSegments = shape.fNseg;
      var heightSegments = shape.fNz;

      var sphere = new THREE.SphereGeometry( outerRadius, widthSegments, heightSegments,
                                             phiStart*Math.PI/180, phiLength*Math.PI/180, thetaStart*Math.PI/180, thetaLength*Math.PI/180);
      sphere.applyMatrix( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );

      var geometry = new THREE.Geometry();
      var color = new THREE.Color();

      // add outer sphere
      for (var n=0; n < sphere.vertices.length; ++n)
         geometry.vertices.push(sphere.vertices[n]);

      // add faces
      for (var n=0; n < sphere.faces.length; ++n) {
         var face = sphere.faces[n];
         geometry.faces.push(new THREE.Face3( face.a, face.b, face.c, null, color, 0 ) );
      }

      var shift = geometry.vertices.length;
      var noInside = (innerRadius <= 0);

      if (noInside) {
         // simple sphere without inner cut
         if ((thetaLength === 180) && (phiLength === 360)) {
            geometry.computeFaceNormals();
            return geometry;
         }

         geometry.vertices.push(new THREE.Vector3(0, 0, 0));
      } else {
         var k = innerRadius / outerRadius;

         // add inner sphere
         for (var n=0; n < sphere.vertices.length; ++n) {
            var v = sphere.vertices[n];
            geometry.vertices.push(new THREE.Vector3(k*v.x, k*v.y, k*v.z));
         }
         for (var n=0; n < sphere.faces.length; ++n) {
            var face = sphere.faces[n];
            geometry.faces.push(new THREE.Face3( shift+face.a, shift+face.b, shift+face.c, null, color, 0 ) );
         }
      }

      if (thetaLength !== 180) {
         // add top cap
         for (var i = 0; i < widthSegments; ++i) {
            if (noInside) {
               geometry.faces.push( new THREE.Face3( i+0, i+1, shift, null, color, 0 ) );
            } else {
               geometry.faces.push( new THREE.Face3( i+0, i+1, i+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i+1, i+shift+1, i+shift, null, color, 0 ) );
            }
         }

         var dshift = sphere.vertices.length - widthSegments - 1;

         // add bottom cap
         for (var i = dshift; i < dshift + widthSegments; ++i) {
            if (noInside) {
               geometry.faces.push( new THREE.Face3( i+0, i+1, shift, null, color, 0 ) );
            } else {
               geometry.faces.push( new THREE.Face3( i+0, i+1, i+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i+1, i+shift+1, i+shift, null, color, 0 ) );
            }
         }
      }

      if (phiLength !== 360) {
         // one cuted side
         for (var j=0; j<heightSegments; j++) {
            var i1 = j*(widthSegments+1);
            var i2 = (j+1)*(widthSegments+1);
            if (noInside) {
               geometry.faces.push( new THREE.Face3( i1, i2, shift, null, color, 0 ) );
            } else {
               geometry.faces.push( new THREE.Face3( i1, i2, i1+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i2, i2+shift, i1+shift, null, color, 0 ));
            }
         }
         // another cuted side
         for (var j=0;j<heightSegments;j++) {
            var i1 = (j+1)*(widthSegments+1) - 1;
            var i2 = (j+2)*(widthSegments+1) - 1;
            if (noInside) {
               geometry.faces.push( new THREE.Face3( i1, i2, shift, null, color, 0 ) );
            } else {
               geometry.faces.push( new THREE.Face3( i1, i2, i1+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i2, i2+shift, i1+shift, null, color, 0));
            }
         }
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   JSROOT.GEO.createTube = function( shape ) {
      var outerRadius1, innerRadius1, outerRadius2, innerRadius2;
      if ((shape._typename == "TGeoCone") || (shape._typename == "TGeoConeSeg")) {
         outerRadius1 = shape.fRmax2;
         innerRadius1 = shape.fRmin2;
         outerRadius2 = shape.fRmax1;
         innerRadius2 = shape.fRmin1;
      } else {
         outerRadius1 = outerRadius2 = shape.fRmax;
         innerRadius1 = innerRadius2 = shape.fRmin;
      }

      var hasrmin = (innerRadius1 > 0) || (innerRadius2 > 0);

      if (hasrmin) {
         if (innerRadius1 <= 0) { innerRadius1 = 0.0000001; console.warn('zero inner radius1 in tube - not yet supported'); }
         if (innerRadius2 <= 0) { innerRadius2 = 0.0000001; console.warn('zero inner radius1 in tube - not yet supported'); }
      }

      var thetaStart = 0, thetaLength = 360;
      if ((shape._typename == "TGeoConeSeg") || (shape._typename == "TGeoTubeSeg") || (shape._typename == "TGeoCtub")) {
         thetaStart = shape.fPhi1;
         thetaLength = shape.fPhi2 - shape.fPhi1;
      }

      var radiusSegments = Math.floor(thetaLength/6);
      if (radiusSegments < 4) radiusSegments = 4;

      var extrapnt = (thetaLength < 360) ? 1 : 0;

      var nsegm = radiusSegments + extrapnt;

      var phi0 = thetaStart*Math.PI/180, dphi = thetaLength/radiusSegments*Math.PI/180;

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(nsegm), _cos = new Float32Array(nsegm);
      for (var seg=0; seg<nsegm; ++seg) {
         _cos[seg] = Math.cos(phi0+seg*dphi);
         _sin[seg] = Math.sin(phi0+seg*dphi);
      }

      var geometry = new THREE.Geometry();

      // add inner tube vertices

      if (hasrmin) {
         for (var seg=0; seg<nsegm; ++seg)
            geometry.vertices.push( new THREE.Vector3( innerRadius1*_cos[seg], innerRadius1*_sin[seg], shape.fDZ));
         for (var seg=0; seg<nsegm; ++seg)
            geometry.vertices.push( new THREE.Vector3( innerRadius2*_cos[seg], innerRadius2*_sin[seg], -shape.fDZ));
      } else {
         geometry.vertices.push( new THREE.Vector3( 0, 0, shape.fDZ));
         geometry.vertices.push( new THREE.Vector3( 0, 0, -shape.fDZ));
      }

      var shift = geometry.vertices.length;

      // add outer tube vertices
      for (var seg=0; seg<nsegm; ++seg)
         geometry.vertices.push( new THREE.Vector3( outerRadius1*_cos[seg], outerRadius1*_sin[seg], shape.fDZ));
      for (var seg=0; seg<nsegm; ++seg)
         geometry.vertices.push( new THREE.Vector3( outerRadius2*_cos[seg], outerRadius2*_sin[seg], -shape.fDZ));

      // recalculate Z of all vertices for ctub shape
      if (shape._typename == "TGeoCtub")
         for (var n=0;n<geometry.vertices.length;++n) {
            var vertex = geometry.vertices[n];
            if (vertex.z<0) vertex.z = -shape.fDz-(vertex.x*shape.fNlow[0]+vertex.x*shape.fNlow[1])/shape.fNlow[2];
                       else vertex.z = shape.fDz-(vertex.y*shape.fNhigh[0]+vertex.y*shape.fNhigh[1])/shape.fNhigh[2];
         }

      var color = new THREE.Color(); // make dummy color for all faces

      // add inner tube faces
      if (hasrmin)
         for (var seg=0; seg<radiusSegments; ++seg) {
            var seg1 = (extrapnt === 1) ? (seg + 1) : (seg + 1) % radiusSegments;
            geometry.faces.push( new THREE.Face3( seg, nsegm + seg, seg1, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( nsegm + seg, nsegm + seg1, seg1, null, color, 0 ) );
         }

      // add outer tube faces
      for (var seg=0; seg<radiusSegments; ++seg) {
         var seg1 = (extrapnt === 1) ? (seg + 1) : (seg + 1) % radiusSegments;
         geometry.faces.push( new THREE.Face3( shift+seg, shift + nsegm + seg, shift + seg1, null, color, 0 ) );
         geometry.faces.push( new THREE.Face3( shift + nsegm + seg, shift + nsegm + seg1, shift + seg1, null, color, 0 ) );
      }


      // add top cap
      for (var i = 0; i < radiusSegments; ++i){
         var i1 = (extrapnt === 1) ? (i+1) : (i+1) % radiusSegments;
         if (hasrmin) {
            geometry.faces.push( new THREE.Face3( i, i+shift, i1, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( i+shift, i1+shift, i1, null, color, 0 ) );
         } else {
            geometry.faces.push( new THREE.Face3( i+shift, 0, i1+shift, null, color, 0 ) );
         }
      }

      // add bottom cap
      for (var i = 0; i < radiusSegments; ++i) {
         var i1 = (extrapnt === 1) ? (i+1) : (i+1) % radiusSegments;
         if (hasrmin) {
            geometry.faces.push( new THREE.Face3( nsegm+i, nsegm+i+shift, nsegm+i1, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( nsegm+i+shift, nsegm+i1+shift, nsegm+i1, null, color, 0 ) );
         } else {
            geometry.faces.push( new THREE.Face3( nsegm+i+shift, 1, nsegm+i1+shift, null, color, 0 ) );
         }
      }

      // close cut regions
      if (extrapnt === 1) {
          if (hasrmin) {
             geometry.faces.push( new THREE.Face3( 0, nsegm, shift+nsegm, null, color, 0 ) );
             geometry.faces.push( new THREE.Face3( 0, shift+nsegm, shift, null, color, 0 ) );
          } else {
             geometry.faces.push( new THREE.Face3( 0, 1, shift+nsegm, null, color, 0 ) );
             geometry.faces.push( new THREE.Face3( 0, shift+nsegm, shift, null, color, 0 ) );
             console.log('very special case of tube with cut');
          }

          if (hasrmin) {
             geometry.faces.push( new THREE.Face3( radiusSegments, 2*radiusSegments+1, shift+2*radiusSegments+1, null, color, 0 ) );
             geometry.faces.push( new THREE.Face3( radiusSegments, shift+2*radiusSegments+1, shift + radiusSegments, null, color, 0 ) );
          } else {
             geometry.faces.push( new THREE.Face3( 0, 1, shift+2*radiusSegments+1, null, color, 0 ) );
             geometry.faces.push( new THREE.Face3( 0, shift+2*radiusSegments+1, shift + radiusSegments, null, color, 0 ) );
          }
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   JSROOT.GEO.createEltu = function( shape ) {
      var geometry = new THREE.Geometry();

      var radiusSegments = Math.floor(360/6);

      // calculate all sin/cos tables in advance
      var x = new Float32Array(radiusSegments),
          y = new Float32Array(radiusSegments);
      for (var seg=0; seg<radiusSegments; ++seg) {
         var phi = seg/radiusSegments*2*Math.PI;
         x[seg] = shape.fRmin*Math.cos(phi);
         y[seg] = shape.fRmax*Math.sin(phi);
      }

      // create vertices
      for (var seg=0; seg<radiusSegments; ++seg)
         geometry.vertices.push( new THREE.Vector3( x[seg], y[seg], -shape.fDZ));
      geometry.vertices.push( new THREE.Vector3( 0, 0, -shape.fDZ));

      for (var seg=0; seg<radiusSegments; ++seg)
         geometry.vertices.push( new THREE.Vector3( x[seg], y[seg], +shape.fDZ));
      geometry.vertices.push( new THREE.Vector3( 0, 0, shape.fDZ));

      var color = new THREE.Color();

      // create tube faces
      for (var seg=0; seg<radiusSegments; ++seg) {
         var seg1 = (seg + 1) % radiusSegments;
         geometry.faces.push( new THREE.Face3( seg, seg+radiusSegments+1, seg1, null, color, 0 ) );
         geometry.faces.push( new THREE.Face3( seg+radiusSegments+1, seg1+radiusSegments+1, seg1, null, color, 0 ) );
      }

      // create bottom cap
      for (var seg=0; seg<radiusSegments; ++seg)
         geometry.faces.push( new THREE.Face3( seg, (seg + 1) % radiusSegments, radiusSegments, null, color, 0 ));

      // create upper cap
      var shift = radiusSegments + 1;
      for (var seg=0; seg<radiusSegments; ++seg)
         geometry.faces.push( new THREE.Face3( shift+seg, shift+ (seg + 1) % radiusSegments, shift+radiusSegments, null, color, 0 ));

      geometry.computeFaceNormals();
      return geometry;
   }


   JSROOT.GEO.createTorus = function( shape ) {
      var radius = shape.fR;
      var innerTube = shape.fRmin;
      var outerTube = shape.fRmax;
      var arc = shape.fDphi - shape.fPhi1;
      var rotation = shape.fPhi1;
      var radialSegments = 30;
      var tubularSegments = Math.floor(arc/6);
      if (tubularSegments < 8) tubularSegments = 8;

      var geometry = new THREE.Geometry();
      var color = new THREE.Color();

      var outerTorus = new THREE.TorusGeometry( radius, outerTube, radialSegments, tubularSegments, arc*Math.PI/180);
      outerTorus.applyMatrix( new THREE.Matrix4().makeRotationZ( rotation*Math.PI/180) );

      var innerTorus = new THREE.TorusGeometry( radius, innerTube, radialSegments, tubularSegments, arc*Math.PI/180);
      innerTorus.applyMatrix( new THREE.Matrix4().makeRotationZ( rotation*Math.PI/180 ) );

      // add inner torus
      for (var n=0; n < innerTorus.vertices.length; ++n)
         geometry.vertices.push(innerTorus.vertices[n]);

      for (var n=0; n < innerTorus.faces.length; ++n) {
         var face = innerTorus.faces[n];
         geometry.faces.push(new THREE.Face3( face.a, face.b, face.c, null, color, 0 ) );
      }

      var shift = geometry.vertices.length;

      // add outer torus
      for (var n=0; n < outerTorus.vertices.length; ++n)
         geometry.vertices.push(outerTorus.vertices[n]);

      for (var n=0; n < outerTorus.faces.length; ++n) {
         var face = outerTorus.faces[n];
         geometry.faces.push(new THREE.Face3( shift+face.a, shift+face.b, shift+face.c, null, color, 0 ) );
      }

      if (arc !== 360) {
         // one cuted side
         for (var j=0;j<radialSegments;j++) {
            var i1 = j*(tubularSegments+1);
            var i2 = (j+1)*(tubularSegments+1);
            geometry.faces.push( new THREE.Face3( i1, i2, i1+shift, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( i2, i2+shift, i1+shift, null, color, 0 ));
         }

         // another cuted side
         for (var j=0;j<radialSegments;j++) {
            var i1 = (j+1)*(tubularSegments+1) - 1;
            var i2 = (j+2)*(tubularSegments+1) - 1;
            geometry.faces.push( new THREE.Face3( i1, i2, i1+shift, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( i2, i2+shift, i1+shift, null, color, 0 ));
         }
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   JSROOT.GEO.createPolygon = function( shape ) {

      var thetaStart = shape.fPhi1, thetaLength = shape.fDphi;

      var radiusSegments = 60;
      if ( shape._typename == "TGeoPgon" ) {
         radiusSegments = shape.fNedges;
      } else {
         radiusSegments = Math.floor(thetaLength/6);
         if (radiusSegments < 4) radiusSegments = 4;
      }

      var geometry = new THREE.Geometry();

      var color = new THREE.Color();

      var phi0 = thetaStart*Math.PI/180, dphi = thetaLength/radiusSegments*Math.PI/180;

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments+1), _cos = new Float32Array(radiusSegments+1);
      for (var seg=0;seg<=radiusSegments;++seg) {
         _cos[seg] = Math.cos(phi0+seg*dphi);
         _sin[seg] = Math.sin(phi0+seg*dphi);
      }

      var indxs = [[],[]], pnts = null, edges = null; // remember indexes for each layer
      var layerVerticies = radiusSegments; // how many verticies in one layer

      if (thetaLength !== 360) {
         pnts = []; // coordinate of point on cut edge (x,z)
         edges = [];  // number of layer for that points
         layerVerticies+=1; // one need one more vertice
      }

      for (var side = 0; side < 2; ++side) {

         var rside = (side === 0) ? 'fRmax' : 'fRmin';
         var prev_indx = geometry.vertices.length;

         for (var layer=0; layer < shape.fNz; ++layer) {

            indxs[side][layer] = geometry.vertices.length;

            // first create points for the layer
            var layerz = shape.fZ[layer], rad = shape[rside][layer];

            if ((layer > 0) && (layer < shape.fNz-1)) {
               if (((shape.fZ[layer-1] === layerz) && (shape[rside][layer-1] === rad)) ||
                   ((shape[rside][layer+1] === rad) && (shape[rside][layer-1] === rad))) {

                  // same Z and R as before - ignore
                  // or same R before and after
                  indxs[side][layer] = indxs[side][layer-1];
                  // if (len) len[side][layer] = len[side][layer-1];
                  continue;
               }
            }

            if (rad <= 0.) rad = 0.000001;

            var curr_indx = geometry.vertices.length;

            // create vertices for the layer
            for (var seg=0; seg < layerVerticies; ++seg)
               geometry.vertices.push( new THREE.Vector3( rad*_cos[seg], rad*_sin[seg], layerz ));

            if (pnts !== null) {
               if (side === 0) {
                  pnts.push(new THREE.Vector2(rad, layerz));
                  edges.push(curr_indx);
               } else {
                  pnts.unshift(new THREE.Vector2(rad, layerz));
                  edges.unshift(curr_indx);
               }
            }

            if (layer>0)  // create faces
               for (var seg=0;seg < radiusSegments;++seg) {
                  var seg1 = (seg + 1) % layerVerticies;
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg, curr_indx + seg1, null, color, 0 ) );
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg1, prev_indx + seg1, null, color, 0 ));
               }

            prev_indx = curr_indx;
         }
      }

      // add faces for top and bottom side
      for (var top = 0; top < 2; ++top) {
         var inside = (top === 0) ? indxs[1][0] : indxs[1][shape.fNz-1];
         var outside = (top === 0) ? indxs[0][0] : indxs[0][shape.fNz-1];
         for (var seg=0; seg < radiusSegments; ++seg) {
            var seg1 = (seg + 1) % layerVerticies;
            geometry.faces.push( new THREE.Face3( outside + seg, inside + seg, inside + seg1, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( outside + seg, inside + seg1, outside + seg1, null, color, 0 ));
         }
      }

      if (pnts!==null) {
         var faces = [];
         if (pnts.length === shape.fNz * 2) {
            // special case - all layers are there, create faces ourself
            for (var layer = shape.fNz-1; layer>0; --layer) {
               if (shape.fZ[layer] === shape.fZ[layer-1]) continue;
               var right = 2*shape.fNz - 1 - layer;
               faces.push([right, layer, layer - 1]);
               faces.push([right, layer-1, right + 1]);
            }

         } else {
            // let three.js calculate our faces
            faces = THREE.ShapeUtils.triangulateShape(pnts, []);
         }

         for (var i = 0; i < faces.length; ++i) {
            var f = faces[i];
            geometry.faces.push( new THREE.Face3( edges[f[0]], edges[f[1]], edges[f[2]], null, color, 0) );
         }
         for (var i = 0; i < faces.length; ++i) {
            var f = faces[i];
            geometry.faces.push( new THREE.Face3( edges[f[0]] + radiusSegments, edges[f[2]] + radiusSegments, edges[f[1]] + radiusSegments, null, color, 0) );
         }
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   JSROOT.GEO.createXtru = function( shape ) {

      var geometry = new THREE.Geometry();

      var fcolor = new THREE.Color();

      var prev = 0, curr = 0;
      for (var layer = 0; layer < shape.fNz; ++layer) {
         var layerz = shape.fZ[layer], scale = shape.fScale[layer];

         prev = curr;
         curr = geometry.vertices.length;

         // add vertices
         for (var vert = 0; vert < shape.fNvert; ++vert)
            geometry.vertices.push( new THREE.Vector3( scale * shape.fX[vert], scale * shape.fY[vert], layerz ));

         if (layer>0)  // create faces for sides
            for (var vert = 0; vert < shape.fNvert; ++vert) {
               var vert1 = (vert + 1) % shape.fNvert;
               geometry.faces.push( new THREE.Face3( prev + vert, curr + vert, curr + vert1, null, fcolor, 0 ) );
               geometry.faces.push( new THREE.Face3( prev + vert, curr + vert1, prev + vert1, null, fcolor, 0 ));
            }
      }

      // now try to make shape - use standard THREE.js utils

      var pnts = [];
      for (var vert = 0; vert < shape.fNvert; ++vert)
         pnts.push( new THREE.Vector2(shape.fX[vert], shape.fY[vert]));
      var faces = THREE.ShapeUtils.triangulateShape(pnts, []);

      for (var i = 0; i < faces.length; ++i) {
         face = faces[ i ];
         geometry.faces.push( new THREE.Face3( face[0], face[1], face[2], null, fcolor, 0) );
         geometry.faces.push( new THREE.Face3( face[0] + curr, face[1] + curr, face[2] + curr, null, fcolor, 0) );
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   JSROOT.GEO.createParaboloid = function( shape ) {

      var radiusSegments = Math.round(360/6);
      var heightSegments = 30;

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments), _cos = new Float32Array(radiusSegments);
      for (var seg=0;seg<radiusSegments;++seg) {
         _cos[seg] = Math.cos(seg/radiusSegments*2*Math.PI);
         _sin[seg] = Math.sin(seg/radiusSegments*2*Math.PI);
      }

      var geometry = new THREE.Geometry();
      var fcolor = new THREE.Color();

      var zmin = -shape.fDZ, zmax = shape.fDZ, rmin = shape.fRlo, rmax = shape.fRhi;

      // if no radius at -z, find intersection
      if (shape.fA >= 0) {
         if (shape.fB > zmin) zmin = shape.fB;
      } else {
         if (shape.fB < zmax) zmax = shape.fB;
      }

      var ttmin = Math.atan2(zmin, rmin), ttmax = Math.atan2(zmax, rmax);

      var prev_indx = 0, prev_radius = 0;

      for (var layer = 0; layer <= heightSegments + 1; ++layer) {
         var layerz = zmax, radius = 0;

         if ((layer === heightSegments + 1) && (prev_radius === 0)) break;

         switch (layer) {
            case 0: layerz = zmin; radius = rmin; break;
            case heightSegments: layerz = zmax; radius = rmax; break;
            case heightSegments + 1: layerz = zmax; radius = 0; break;
            default: {
               var tt = Math.tan(ttmin + (ttmax-ttmin) * layer / heightSegments);
               var delta = tt*tt - 4*shape.fA*shape.fB; // should be always positive (a*b<0)
               radius = 0.5*(tt+Math.sqrt(delta))/shape.fA;
               if (radius < 1e-6) radius = 0;
               layerz = radius*tt;
            }
         }

         var curr_indx = geometry.vertices.length;

         if (radius === 0) {
            geometry.vertices.push( new THREE.Vector3( 0, 0, layerz ));
         } else {
            for (var seg=0; seg<radiusSegments; ++seg)
               geometry.vertices.push( new THREE.Vector3( radius*_cos[seg], radius*_sin[seg], layerz));
         }

         // add faces of next layer
         if (layer>0) {
            for (var seg=0; seg<radiusSegments; ++seg) {
               var seg1 = (seg+1) % radiusSegments;
               if (prev_radius === 0) {
                  geometry.faces.push( new THREE.Face3( prev_indx, curr_indx + seg, curr_indx + seg1, null, fcolor, 0) );
               } else
               if (radius == 0) {
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx, prev_indx + seg1, null, fcolor, 0) );
               } else {
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg, curr_indx + seg1, null, fcolor, 0) );
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg1, prev_indx + seg1, null, fcolor, 0) );
               }
            }
         }

         prev_radius = radius;
         prev_indx = curr_indx;
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   JSROOT.GEO.createHype = function( shape ) {

      if ((shape.fTin===0) && (shape.fTout===0))
         return JSROOT.GEO.createTube(shape);

      var radiusSegments = Math.round(360/6);
      var heightSegments = 30;

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments), _cos = new Float32Array(radiusSegments);
      for (var seg=0;seg<radiusSegments;++seg) {
         _cos[seg] = Math.cos(seg/radiusSegments*2*Math.PI);
         _sin[seg] = Math.sin(seg/radiusSegments*2*Math.PI);
      }

      var geometry = new THREE.Geometry();
      var fcolor = new THREE.Color();

      var indexes = [[],[]];

      // in-out side
      for (var side=0;side<2;++side) {

         // add only points, no faces
         if ((side===0) && (shape.fRmin <= 0)) {
            indexes[side][0] = geometry.vertices.length;
            geometry.vertices.push( new THREE.Vector3( 0, 0, -shape.fDz ) );
            indexes[side][heightSegments] = geometry.vertices.length;
            geometry.vertices.push( new THREE.Vector3( 0, 0, shape.fDz ) );
            continue;
         }

         var prev_indx = 0;
         var r0 = (side===0) ? shape.fRmin : shape.fRmax;
         var tsq = (side===0) ? shape.fTinsq : shape.fToutsq;

         // vertical layers
         for (var layer=0;layer<=heightSegments;++layer) {
            var layerz = -shape.fDz + layer/heightSegments*2*shape.fDz;

            var radius = Math.sqrt(r0*r0+tsq*layerz*layerz);
            var curr_indx = geometry.vertices.length;

            indexes[side][layer] = curr_indx;

            for (var seg=0; seg<radiusSegments; ++seg)
               geometry.vertices.push( new THREE.Vector3( radius*_cos[seg], radius*_sin[seg], layerz));

            // add faces of next layer
            if (layer>0) {
               for (var seg=0; seg<radiusSegments; ++seg) {
                  var seg1 = (seg+1) % radiusSegments;
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg, curr_indx + seg1, null, fcolor, 0) );
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg1, prev_indx + seg1, null, fcolor, 0) );
               }
            }

            prev_indx = curr_indx;
         }
      }

      // add caps
      for(var layer=0; layer<=heightSegments; layer+=heightSegments) {
         var inside = indexes[0][layer], outside = indexes[1][layer];
         for (var seg=0; seg<radiusSegments; ++seg) {
            var seg1 = (seg+1) % radiusSegments;
            if (shape.fRmin <= 0) {
               geometry.faces.push( new THREE.Face3( inside, outside + seg, outside + seg1, null, fcolor, 0) );
            } else {
               geometry.faces.push( new THREE.Face3( inside + seg, outside + seg, outside + seg1, null, fcolor, 0) );
               geometry.faces.push( new THREE.Face3( inside + seg, outside + seg1, inside + seg1, null, fcolor, 0) );
            }
         }
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   JSROOT.GEO.createComposite = function ( shape ) {

      return null;

      var geom1 = JSROOT.GEO.createGeometry(shape.fNode.fLeft);

      var geom2 = JSROOT.GEO.createGeometry(shape.fNode.fRight);

      // var geometry = new THREE.Geometry();

      return geom1;
   }


   JSROOT.GEO.createGeometry = function( shape ) {

      switch (shape._typename) {
         case "TGeoBBox": return JSROOT.GEO.createCube( shape );
         case "TGeoPara": return JSROOT.GEO.createPara( shape );
         case "TGeoTrd1":
         case "TGeoTrd2": return JSROOT.GEO.createTrapezoid( shape );
         case "TGeoArb8":
         case "TGeoTrap":
         case "TGeoGtra": return JSROOT.GEO.createArb8( shape );
         case "TGeoSphere": return JSROOT.GEO.createSphere( shape );
         case "TGeoCone":
         case "TGeoConeSeg":
         case "TGeoTube":
         case "TGeoTubeSeg":
         case "TGeoCtub": return JSROOT.GEO.createTube( shape );
         case "TGeoEltu": return JSROOT.GEO.createEltu( shape );
         case "TGeoTorus": return JSROOT.GEO.createTorus( shape );
         case "TGeoPcon":
         case "TGeoPgon": return JSROOT.GEO.createPolygon( shape );
         case "TGeoXtru": return JSROOT.GEO.createXtru( shape );
         case "TGeoParaboloid": return JSROOT.GEO.createParaboloid( shape );
         case "TGeoHype": return JSROOT.GEO.createHype( shape );
         case "TGeoCompositeShape": return JSROOT.GEO.createComposite( shape );
      }

      return null;
   }

   /**
    * @class JSROOT.TGeoPainter Holder of different functions and classes for drawing geometries
    */

   // ======= Geometry painter================================================



   JSROOT.EGeoVisibilityAtt = {
         kVisOverride     : JSROOT.BIT(0),           // volume's vis. attributes are overidden
         kVisNone         : JSROOT.BIT(1),           // the volume/node is invisible, as well as daughters
         kVisThis         : JSROOT.BIT(2),           // this volume/node is visible
         kVisDaughters    : JSROOT.BIT(3),           // all leaves are visible
         kVisOneLevel     : JSROOT.BIT(4),           // first level daughters are visible
         kVisStreamed     : JSROOT.BIT(5),           // true if attributes have been streamed
         kVisTouched      : JSROOT.BIT(6),           // true if attributes are changed after closing geom
         kVisOnScreen     : JSROOT.BIT(7),           // true if volume is visible on screen
         kVisContainers   : JSROOT.BIT(12),          // all containers visible
         kVisOnly         : JSROOT.BIT(13),          // just this visible
         kVisBranch       : JSROOT.BIT(14),          // only a given branch visible
         kVisRaytrace     : JSROOT.BIT(15)           // raytracing flag
      };

   JSROOT.TestGeoAttBit = function(volume, f) {
      if (!('fGeoAtt' in volume)) return false;
      return (volume['fGeoAtt'] & f) != 0;
   }

   JSROOT.ToggleGeoAttBit = function(volume, f) {
      if (!('fGeoAtt' in volume)) return false;

      volume['fGeoAtt'] = volume['fGeoAtt'] ^ (f & 0xffffff);
   }

   JSROOT.TGeoPainter = function( geometry ) {
      JSROOT.TObjectPainter.call( this, geometry );

      this.Cleanup(true);

      if ((geometry !== null) && (geometry['_typename'].indexOf('TGeoVolume') === 0))
         geometry = { _typename:"TGeoNode", fVolume: geometry, fName:"TopLevel" };

      this._geometry = geometry;
   }

   JSROOT.TGeoPainter.prototype = Object.create( JSROOT.TObjectPainter.prototype );

   JSROOT.TGeoPainter.prototype.CreateToolbar = function(args) {
      if ( this._toolbar !== null ) return;
      var painter = this;
      var buttonList = [{
         name: 'toImage',
         title: 'Save as PNG',
         icon: JSROOT.ToolbarIcons.camera,
         click: function() {
            var dataUrl = painter._renderer.domElement.toDataURL("image/png");
            dataUrl.replace("image/png", "image/octet-stream");
            var link = document.createElement('a');
            if (typeof link.download === 'string') {
               document.body.appendChild(link); //Firefox requires the link to be in the body
               link.download = "geometry.png";
               link.href = dataUrl;
               link.click();
               document.body.removeChild(link); //remove the link when done
            }
         }
      }];
      this._toolbar = new JSROOT.Toolbar( this.select_main(), [buttonList] );
   }

   JSROOT.TGeoPainter.prototype.GetObject = function() {
      return this._geometry;
   }

   JSROOT.TGeoPainter.prototype.decodeOptions = function(opt) {
      var res = { _grid: false, _bound: false, _debug: false, _full: false, maxlvl: -1, _axis:false, scale: new THREE.Vector3(1,1,1) };

      var _opt = JSROOT.GetUrlOption('_grid');
      if (_opt !== null && _opt == "true") res._grid = true;
      var _opt = JSROOT.GetUrlOption('_debug');
      if (_opt !== null && _opt == "true") { res._debug = true; res._grid = true; }
      if (_opt !== null && _opt == "bound") { res._debug = true; res._grid = true; res._bound = true; }
      if (_opt !== null && _opt == "full") { res._debug = true; res._grid = true; res._full = true; res._bound = true; }

      opt = opt.toLowerCase();

      if (opt.indexOf("all")>=0) {
         res.maxlvl = 9999;
         opt = opt.replace("all", " ");
      }
      if (opt.indexOf("limit")>=0) {
         res.maxlvl = 1111;
         opt = opt.replace("limit", " ");
      }
      if (opt.indexOf("invx")>=0) {
         res.scale.x = -1;
         opt = opt.replace("invx", " ");
      }
      if (opt.indexOf("invy")>=0) {
         res.scale.y = -1;
         opt = opt.replace("invy", " ");
      }
      if (opt.indexOf("invz")>=0) {
         res.scale.z = -1;
         opt = opt.replace("invz", " ");
      }

      var p = opt.indexOf("maxlvl");
      if (p>=0) {
         res.maxlvl = parseInt(opt.substr(p+6, 1));
         opt = opt.replace("maxlvl" + res.maxlvl, " ");
      }

      if (opt.indexOf("d")>=0) res._debug = true;
      if (opt.indexOf("g")>=0) res._grid = true;
      if (opt.indexOf("b")>=0) res._bound = true;
      if (opt.indexOf("f")>=0) res._full = true;
      if (opt.indexOf("a")>=0) { res._axis = true; res._yup = false; }
      if (opt.indexOf("y")>=0) res._yup = true;
      if (opt.indexOf("z")>=0) res._yup = false;

      return res;
   }


   JSROOT.TGeoPainter.prototype.addControls = function() {

      if (this._controls !== null) return;

      var painter = this;

      this.select_main().property('flex_block_drag', true);

      this._controls = new THREE.OrbitControls(this._camera, this._renderer.domElement);
      this._controls.enableDamping = true;
      this._controls.dampingFactor = 0.25;
      this._controls.enableZoom = true;
      this._controls.target.copy(this._lookat);
      this._controls.update();

      this._controls.addEventListener( 'change', function() { painter.Render3D(0); } );

      if ( this.options._debug || this.options._grid ) {
         this._tcontrols = new THREE.TransformControls( this._camera, this._renderer.domElement );
         this._scene.add( this._tcontrols );
         this._tcontrols.attach( this._toplevel );
         //this._tcontrols.setSize( 1.1 );

         window.addEventListener( 'keydown', function ( event ) {
            switch ( event.keyCode ) {
               case 81: // Q
                  painter._tcontrols.setSpace( painter._tcontrols.space === "local" ? "world" : "local" );
                  break;
               case 17: // Ctrl
                  painter._tcontrols.setTranslationSnap( Math.ceil( painter._overall_size ) / 50 );
                  painter._tcontrols.setRotationSnap( THREE.Math.degToRad( 15 ) );
                  break;
               case 84: // T (Translate)
                  painter._tcontrols.setMode( "translate" );
                  break;
               case 82: // R (Rotate)
                  painter._tcontrols.setMode( "rotate" );
                  break;
               case 83: // S (Scale)
                  painter._tcontrols.setMode( "scale" );
                  break;
               case 187:
               case 107: // +, =, num+
                  painter._tcontrols.setSize( painter._tcontrols.size + 0.1 );
                  break;
               case 189:
               case 109: // -, _, num-
                  painter._tcontrols.setSize( Math.max( painter._tcontrols.size - 0.1, 0.1 ) );
                  break;
            }
         });
         window.addEventListener( 'keyup', function ( event ) {
            switch ( event.keyCode ) {
               case 17: // Ctrl
                  painter._tcontrols.setTranslationSnap( null );
                  painter._tcontrols.setRotationSnap( null );
                  break;
            }
         });

         this._tcontrols.addEventListener( 'change', function() { painter.Render3D(0); } );
      }

      var raycaster = new THREE.Raycaster(), INTERSECTED = null;

      function findIntersection(mouse) {
         // find intersections

         // if (!JSROOT.gStyle.Tooltip) return tooltip.hide();

         raycaster.setFromCamera( mouse, painter._camera );
         var intersects = raycaster.intersectObjects(painter._scene.children, true);
         if (intersects.length > 0) {
            var pick = null;
            for (var i = 0; i < intersects.length; ++i) {
               if ('emissive' in intersects[i].object.material) {
                  pick = intersects[i].object;
                  break;
               }
            }
            if (pick && INTERSECTED != pick) {
               INTERSECTED = pick;

               var name = INTERSECTED.name;

               var p = INTERSECTED.parent;
               while ((p!==undefined) && (p!==null)) {
                  if ('name' in p) name = p.name+'/'+name;
                  p = p.parent;
               }

               // console.log('intersect ' + name);
            }
         } else {
            // INTERSECTED = null;
         }
      };

      function mousemove(e) {
         var mouse_x = ('offsetX' in e) ? e.offsetX : e.layerX;
         var mouse_y = ('offsetY' in e) ? e.offsetY : e.layerY;
         var mouse = { x: (mouse_x / painter._renderer.domElement.width) * 2 - 1,
                   y: -(mouse_y / painter._renderer.domElement.height) * 2 + 1 };

         findIntersection(mouse);
         e.preventDefault();
      }

      this._renderer.domElement.addEventListener('mousemove', mousemove);
   }

   JSROOT.TGeoPainter.prototype.accountClear = function() {
      this._num_geom = 0;
      this._num_vertices = 0;
      this._num_faces = 0;
      this._num_meshes = 0;
   }

   JSROOT.TGeoPainter.prototype.accountGeom = function(geom, shape_typename) {
      // used to calculate statistic over created geometry
      if (geom === null) {
         if (!('unsupported_shapes' in this)) this.unsupported_shapes = [];
         if ((shape_typename !== undefined) && (this.unsupported_shapes.indexOf(shape_typename) < 0)) {
             this.unsupported_shapes.push(shape_typename);
             console.warn('Not supported ' + shape_typename);
         }
         return;
      }

      this._num_geom++;
      if (('vertices' in geom) && ('faces' in geom)) {
         this._num_vertices += geom.vertices.length;
         this._num_faces += geom.faces.length;
      } else {

         var attr = geom.getAttribute('position');
         // this._num_vertices += attr.count() / 3;
         // this._num_faces += geom.index.count() / 3;
      }
   }

   JSROOT.TGeoPainter.prototype.accountMesh = function(mesh) {
      // used to calculate statistic over created meshes
      if (mesh !== null) this._num_meshes++;
   }

   JSROOT.TGeoPainter.prototype.checkFlipping = function(parent, matrix, shape, geom, mesh_has_childs) {
      // check if matrix of element should be flipped

      var m = new THREE.Matrix4();
      m.multiplyMatrices( parent.matrixWorld, matrix);

      if (m.determinant() > -0.9) return geom;

      // we could not transform matrix of mesh with childs, need workaround
      if (mesh_has_childs) return null;

      var cnt = 0, flip = new THREE.Vector3(1,1,1);

      if (m.elements[0]===-1 &&  m.elements[1]=== 0 &&  m.elements[2]=== 0) { flip.x = -1; cnt++; }
      if (m.elements[4]=== 0  && m.elements[5]===-1 &&  m.elements[6]=== 0) { flip.y = -1; cnt++; }
      if (m.elements[8]=== 0  && m.elements[9]=== 0 && m.elements[10]===-1) { flip.z = -1; cnt++; }

      if ((cnt===0) || (cnt ===2)) {
         flip.set(1,1,1); cnt = 0;
         if (m.elements[0] + m.elements[1] + m.elements[2] === -1) { flip.x = -1; cnt++; }
         if (m.elements[4] + m.elements[5] + m.elements[6] === -1) { flip.y = -1; cnt++; }
         if (m.elements[8] + m.elements[9] + m.elements[10] === -1) { flip.z = -1; cnt++; }
         if ((cnt === 0) || (cnt === 2)) {
            // console.log('not found proper axis, use Z ' + JSON.stringify(flip) + '  m = ' + JSON.stringify(m.elements));
            flip.z = -flip.z;
         }
      }

      matrix.scale(flip);

      var gname = "_geom";
      if (flip.x<0) gname += "X";
      if (flip.y<0) gname += "Y";
      if (flip.z<0) gname += "Z";

      // if geometry with such flipping already was created - use it again
      if (gname in shape) return shape[gname];

      geom = geom.clone();

      geom.scale(flip.x, flip.y, flip.z);

      var face, d;
      for (var n=0;n<geom.faces.length;++n) {
         face = geom.faces[n];
         d = face.b; face.b = face.c; face.c = d;
      }

      //geom.computeBoundingSphere();
      geom.computeFaceNormals();

      shape[gname] = geom;

      this.accountGeom(geom);

      return geom;
   }

   JSROOT.TGeoPainter.prototype.getNodeProperties = function(node, visible) {
      // function return matrix, shape and material

      var volume = node['fVolume'];

      var prop = { shape: volume.fShape };

      var translation_matrix = null; // [0, 0, 0];
      var rotation_matrix = null;//[1, 0, 0, 0, 1, 0, 0, 0, 1];

      if (typeof node['fMatrix'] != 'undefined' && node['fMatrix'] !== null) {
         if (node.fMatrix._typename == 'TGeoTranslation') {
            translation_matrix = node.fMatrix.fTranslation;
         }
         else if (node.fMatrix._typename == 'TGeoRotation') {
            rotation_matrix = node.fMatrix.fRotationMatrix;
         }
         else if (node.fMatrix._typename == 'TGeoCombiTrans') {
            if (typeof node.fMatrix.fTranslation != 'undefined' &&
                node.fMatrix.fTranslation !== null)
               translation_matrix = node.fMatrix.fTranslation;
            if (typeof node.fMatrix.fRotation != 'undefined' &&
                node.fMatrix.fRotation !== null) {
               rotation_matrix = node.fMatrix.fRotation.fRotationMatrix;
            }
         }
      }
      if ((node['_typename'] == "TGeoNodeOffset") && (node['fFinder'] != null)) {
         // if (node['fFinder']['_typename'] == 'TGeoPatternParaX') { }
         // if (node['fFinder']['_typename'] == 'TGeoPatternParaY') { }
         // if (node['fFinder']['_typename'] == 'TGeoPatternParaZ') { }
         // if (node['fFinder']['_typename'] == 'TGeoPatternTrapZ') { }
         // if (node['fFinder']['_typename'] == 'TGeoPatternCylR') { }
         // if (node['fFinder']['_typename'] == 'TGeoPatternSphR') { }
         // if (node['fFinder']['_typename'] == 'TGeoPatternSphTheta') { }
         // if (node['fFinder']['_typename'] == 'TGeoPatternSphPhi') { }
         // if (node['fFinder']['_typename'] == 'TGeoPatternHoneycomb') { }
         if ((node['fFinder']['_typename'] == 'TGeoPatternX') ||
             (node['fFinder']['_typename'] == 'TGeoPatternY') ||
             (node['fFinder']['_typename'] == 'TGeoPatternZ')) {
            var _shift = node.fFinder.fStart + (node.fIndex + 0.5) * node.fFinder.fStep;

            if (translation_matrix !== null)
               console.warn('translation exists - should we combine with ' + node['fFinder']['_typename']);
            else
            switch (node['fFinder']['_typename'].charAt(11)) {
               case 'X': translation_matrix = [_shift, 0, 0]; break;
               case 'Y': translation_matrix = [0, _shift, 0]; break;
               case 'Z': translation_matrix = [0, 0, _shift]; break;
            }
         } else
         if (node['fFinder']['_typename'] == 'TGeoPatternCylPhi') {

            var _cos = 1., _sin = 0.;

            if (typeof node['fFinder']['fSinCos'] === 'undefined') {
               _cos = Math.cos((Math.PI / 180.0)*(node['fFinder']['fStart']+(node.fIndex+0.5)*node['fFinder']['fStep']));
               _sin = Math.sin((Math.PI / 180.0)*(node['fFinder']['fStart']+(node.fIndex+0.5)*node['fFinder']['fStep']));
            } else {
               _cos = node['fFinder']['fSinCos'][2*node.fIndex+1];
               _sin = node['fFinder']['fSinCos'][2*node.fIndex];
            }

            if (rotation_matrix === null) {
               rotation_matrix = [_cos, -_sin, 0,
                                  _sin,  _cos, 0,
                                     0,     0, 1];
            } else {
               console.warn('should we multiply rotation matrixes here??');
               rotation_matrix[0] = _cos;
               rotation_matrix[1] = -_sin;
               rotation_matrix[3] = _sin;
               rotation_matrix[4] = _cos;
            }
         } else {
            console.warn('Unsupported pattern type ' + node['fFinder']['_typename']);
         }
      }

      prop.material = null;

      if (visible) {
         var _transparent = false, _opacity = 1.0, fillcolor;
         if (volume['fLineColor'] >= 0)
            fillcolor = JSROOT.Painter.root_colors[volume['fLineColor']];
         if (typeof volume['fMedium'] != 'undefined' && volume['fMedium'] !== null &&
               typeof volume['fMedium']['fMaterial'] != 'undefined' &&
               volume['fMedium']['fMaterial'] !== null) {
            var fillstyle = volume['fMedium']['fMaterial']['fFillStyle'];
            var transparency = (fillstyle < 3000 || fillstyle > 3100) ? 0 : fillstyle - 3000;
            if (transparency > 0) {
               _transparent = true;
               _opacity = (100.0 - transparency) / 100.0;
            }
            if (fillcolor === undefined)
               fillcolor = JSROOT.Painter.root_colors[volume['fMedium']['fMaterial']['fFillColor']];
         }
         if (fillcolor === undefined)
            fillcolor = "lightgrey";

         prop.material = new THREE.MeshLambertMaterial( { transparent: _transparent,
                              opacity: _opacity, wireframe: false, color: fillcolor,
                              side: THREE.DoubleSide, vertexColors: THREE.NoColors /*THREE.VertexColors*/,
                              overdraw: 0. } );
      }

      prop.matrix = new THREE.Matrix4();

      if (rotation_matrix !== null)
         prop.matrix.set(rotation_matrix[0], rotation_matrix[1], rotation_matrix[2],   0,
                         rotation_matrix[3], rotation_matrix[4], rotation_matrix[5],   0,
                         rotation_matrix[6], rotation_matrix[7], rotation_matrix[8],   0,
                                          0,                  0,                  0,   1);

      if (translation_matrix !== null)
         prop.matrix.setPosition(new THREE.Vector3(translation_matrix[0], translation_matrix[1], translation_matrix[2]));

      return prop;
   }

   JSROOT.TGeoPainter.prototype.getEveNodeProperties = function(node, visible) {

      var prop = { shape: node.fShape };

      prop.material = null;

      if (visible) {
         var _transparent = false, _opacity = 1.0;
         if ( node['fRGBA'][3] < 1.0) {
            _transparent = true;
            _opacity = node['fRGBA'][3];
         }
         var linecolor = new THREE.Color( node['fRGBALine'][0], node['fRGBALine'][1], node['fRGBALine'][2] );
         var fillcolor = new THREE.Color( node['fRGBA'][0], node['fRGBA'][1], node['fRGBA'][2] );
         prop.material = new THREE.MeshLambertMaterial( { transparent: _transparent,
                          opacity: _opacity, wireframe: false, color: fillcolor,
                          side: THREE.DoubleSide, vertexColors: THREE.NoColors /*THREE.VertexColors */,
                          overdraw: 0. } );
      }

      prop.matrix = new THREE.Matrix4().set(
                           node.fTrans[0],  node.fTrans[4],  node.fTrans[8],  0,
                           node.fTrans[1],  node.fTrans[5],  node.fTrans[9],  0,
                           node.fTrans[2],  node.fTrans[6],  node.fTrans[10], 0,
                                        0,               0,                0, 1);

      // second - set position with proper sign
      prop.matrix.setPosition({ x: node.fTrans[12], y: node.fTrans[13], z: node.fTrans[14] });
      return prop;
   }


   JSROOT.TGeoPainter.prototype.drawNode = function() {

      if ((this._stack == null) || (this._stack.length == 0)) return false;

      var arg = this._stack[this._stack.length - 1];

      // cut all volumes below 0 level
      // if (arg.lvl===0) { this._stack.pop(); return true; }

      var kind = this.NodeKind(arg.node);
      if (kind < 0) return false;
      var chlds = null;

      if (kind === 0) {
         chlds = (arg.node.fVolume.fNodes !== null) ? arg.node.fVolume.fNodes.arr : null;
      } else {
         chlds = (arg.node.fElements !== null) ? arg.node.fElements.arr : null;
      }

      if ('nchild' in arg) {
         // add next child
         if ((chlds === null) || (chlds.length <= arg.nchild)) {
            this._stack.pop();
         } else {
            this._stack.push({ toplevel: (arg.mesh ? arg.mesh : arg.toplevel),
                               node: chlds[arg.nchild++] });
         }
         return true;
      }

      var prop = null;

      //if (kind === 0)
      //   arg.node._mesh = JSROOT.GEO.createNodeMesh(arg.node, arg.toplevel);
      //else
      //   arg.node._mesh = JSROOT.GEO.createEveNodeMesh(arg.node, arg.toplevel);

      if (kind === 0)
         prop = this.getNodeProperties(arg.node, arg.node._visible);
      else
         prop = this.getEveNodeProperties(arg.node, arg.node._visible);

      var geom = null;

      if ((prop.shape === null) && arg.node._visible)
         arg.node._visible = false;

      if (arg.node._visible) {
         if (typeof prop.shape._geom === 'undefined') {
            prop.shape._geom = JSROOT.GEO.createGeometry(prop.shape);
            this.accountGeom(prop.shape._geom, prop.shape._typename);
         }

         geom = prop.shape._geom;

      } else {
         if (this._dummy_material === undefined)
            this._dummy_material =
               new THREE.MeshLambertMaterial( { transparent: true, opacity: 0, wireframe: false,
                                                color: 'white', vertexColors: THREE.NoColors,
                                                overdraw: 0., depthWrite : false, depthTest: false, visible: false } );

         prop.material = this._dummy_material;
      }

      var has_childs = (chlds !== null) && (chlds.length > 0);
      var work_around = false;

      // this is only for debugging - test invertion of whole geometry
      if (arg.main && (this.options.scale !== null)) {
         if ((this.options.scale.x<0) || (this.options.scale.y<0) || (this.options.scale.z<0)) {
            prop.matrix.scale(this.options.scale);
         }
      }

      if (arg.node._visible && (geom !== null)) {
         geom = this.checkFlipping(arg.toplevel, prop.matrix, prop.shape, geom, has_childs);
         work_around = has_childs && (geom === null);
      }

      if (geom === null) geom = new THREE.Geometry();

      var mesh = new THREE.Mesh( geom, prop.material );

      mesh.applyMatrix(prop.matrix);

      this.accountMesh(mesh);

      mesh.name = arg.node.fName;

      // add the mesh to the scene
      arg.toplevel.add(mesh);

      mesh.updateMatrixWorld();

      if (work_around) {
         JSROOT.console('perform workaroud for flipping mesh with childs');

         prop.matrix.identity(); // set to 1

         geom = this.checkFlipping(mesh, prop.matrix, prop.shape, prop.shape._geom, false);

         var dmesh = new THREE.Mesh( geom, prop.material );

         dmesh.applyMatrix(prop.matrix);

         dmesh.name = "..";

         // add the mesh to the scene
         mesh.add(dmesh);

         dmesh.updateMatrixWorld();
      }

      if (this.options._debug && (arg.node._visible || this.options._full)) {
         var helper = new THREE.WireframeHelper(mesh);
         helper.material.color.set(JSROOT.Painter.root_colors[arg.node.fVolume['fLineColor']]);
         helper.material.linewidth = arg.node.fVolume['fLineWidth'];
         arg.toplevel.add(helper);
      }

      if (this.options._bound && (arg.node._visible || this.options._full)) {
         var boxHelper = new THREE.BoxHelper( mesh );
         arg.toplevel.add( boxHelper );
      }

      arg.mesh = mesh;

      if ((chlds === null) || (chlds.length == 0)) {
         // do not draw childs
         this._stack.pop();
      } else {
         arg.nchild = 0; // specify that childs should be extracted
      }

      return true;
   }

   JSROOT.TGeoPainter.prototype.NodeKind = function(obj) {
      if ((obj === undefined) || (obj === null) || (typeof obj !== 'object')) return -1;
      return ('fShape' in obj) && ('fTrans' in obj) ? 1 : 0;
   }

   JSROOT.TGeoPainter.prototype.CountGeoVolumes = function(obj, arg, lvl) {
      // count number of volumes, numver per hierarchy level, reference count, number of childs
      // also check if volume shape can be drawn

      var kind = this.NodeKind(obj);
      if (kind < 0) return 0;

      if (lvl === undefined) {
         lvl = 0;
         if (!arg) arg = { erase: true };
         if (!('map' in arg)) arg.map = [];
         arg.viscnt = 0;
         if (!('clear' in arg))
            arg.clear = function() {
               for (var n=0;n<this.map.length;++n) {
                  delete this.map[n]._refcnt;
                  delete this.map[n]._numchld;
                  delete this.map[n]._visible;
               }
               this.map = [];
               this.viscnt = 0;
            };
      }

      var chlds = null, shape = null, vis = false;
      if (kind === 0) {
         if ((obj.fVolume === undefined) || (obj.fVolume === null)) return 0;
         shape = obj.fVolume.fShape;
         chlds = (obj.fVolume.fNodes !== null) ? obj.fVolume.fNodes.arr : null;
         vis = JSROOT.TestGeoAttBit(obj.fVolume, JSROOT.EGeoVisibilityAtt.kVisOnScreen)
                || ((lvl < arg.maxlvl) && JSROOT.TestGeoAttBit(obj.fVolume, JSROOT.EGeoVisibilityAtt.kVisThis));
      } else {
         if (obj.fShape === undefined) return 0;
         shape = obj.fShape;
         chlds = (obj.fElements !== null) ? obj.fElements.arr : null;
         vis = obj['fRnrSelf'];
      }

      if ('cnt' in arg) {
         if (arg.cnt[lvl] === undefined) arg.cnt[lvl] = 0;
         arg.cnt[lvl] += 1;
      }

      if ('_refcnt' in obj) {
          obj._refcnt++;
      } else {
         obj._refcnt = 1;
         obj._numchld = 0;
         arg.map.push(obj);

         /*
         // kVisNone         : JSROOT.BIT(1),           // the volume/node is invisible, as well as daughters
         // kVisThis         : JSROOT.BIT(2),           // this volume/node is visible
         // kVisDaughters    : JSROOT.BIT(3),           // all leaves are visible
         // kVisOneLevel     : JSROOT.BIT(4),           // first level daughters are visible

         if (JSROOT.TestGeoAttBit(obj.fVolume, JSROOT.EGeoVisibilityAtt.kVisNone))
            console.log('not visible');
         else
         if (JSROOT.TestGeoAttBit(obj.fVolume, JSROOT.EGeoVisibilityAtt.kVisOneLevel))
            console.log('only one level');
         }
         */

         if (vis && !('_visible' in obj) && (shape!==null)) {
            obj._visible = true;
            arg.viscnt++;
         }

         if (chlds !== null)
            for (var i = 0; i < chlds.length; ++i)
               obj._numchld += this.CountGeoVolumes(chlds[i], arg, lvl+1);
      }

      if ((lvl === 0) && arg.erase) arg.clear();

      return 1 + obj._numchld;
   }

   JSROOT.TGeoPainter.prototype.SameMaterial = function(node1, node2) {

      if ((node1===null) || (node2===null)) return node1 === node2;

      if (node1.fVolume.fLineColor >= 0)
         return (node1.fVolume.fLineColor === node2.fVolume.fLineColor);

       var m1 = (node1.fVolume['fMedium'] !== null) ? node1.fVolume['fMedium']['fMaterial'] : null;
       var m2 = (node2.fVolume['fMedium'] !== null) ? node2.fVolume['fMedium']['fMaterial'] : null;

       if (m1 === m2) return true;

       if ((m1 === null) || (m2 === null)) return false;

       return (m1.fFillStyle === m2.fFillStyle) && (m1.fFillColor === m2.fFillColor);
    }

   JSROOT.TGeoPainter.prototype.ScanUniqueVisVolumes = function(obj, lvl, arg) {
      if ((obj === undefined) || (obj===null) || (typeof obj !== 'object') ||
          (obj.fVolume === undefined) || (obj.fVolume == null)) return 0;

      if (lvl === 0) {
         arg.master = null;
         arg.vis_unique = true;
         arg.vis_master = null; // master used to verify material attributes
         arg.same_material = true;
      }

      var res = obj._visible ? 1 : 0;

      if (obj._refcnt > 1) arg.vis_unique = false;
      if (arg.master!==null)
         if (!this.SameMaterial(arg.master, obj)) arg.same_material = false;

      var top_unique = arg.vis_unique;
      arg.vis_unique = true;

      var top_master = arg.master, top_same = arg.same_material;

      arg.master = obj._visible ? obj : null;
      arg.same_material = true;

      var arr = (obj.fVolume.fNodes !== null) ? obj.fVolume.fNodes.arr : null;

      var numvis = 0;
      if (arr !== null)
         for (var i = 0; i < arr.length; ++i)
            numvis += this.ScanUniqueVisVolumes(arr[i], lvl+1, arg);

      obj._numvis = numvis;
      obj._visunique  = arg.vis_unique;
      obj._samematerial = arg.same_material;

      if (obj._samematerial) {
         if (top_same && (top_master!=null) && (arg.master!==null))
            arg.same_material = this.SameMaterial(top_master, arg.master);
         else
            arg.same_material = top_same;

         if (top_master !== null) arg.master = top_master;
      } else {
         arg.master = null; // when material differ, no need to preserve master
         arg.same_material = false;
      }

      arg.vis_unique = top_unique && obj._visunique;

      return res + numvis;
   }

   JSROOT.TGeoPainter.prototype.createScene = function(webgl, w, h, pixel_ratio) {
      // three.js 3D drawing
      this._scene = new THREE.Scene();
      this._scene.fog = new THREE.Fog(0xffffff, 500, 300000);

      this._scene_width = w;
      this._scene_height = h;

      this._camera = new THREE.PerspectiveCamera(25, w / h, 1, 100000);

      this._renderer = webgl ?
                        new THREE.WebGLRenderer({ antialias : true, logarithmicDepthBuffer: true,
                                                  preserveDrawingBuffer: true }) :
                        new THREE.CanvasRenderer({antialias : true });
      this._renderer.setPixelRatio(pixel_ratio);
      this._renderer.setClearColor(0xffffff, 1);
      this._renderer.setSize(w, h);

      var pointLight = new THREE.PointLight(0xefefef);
      this._camera.add( pointLight );
      pointLight.position.set(10, 10, 10);
      this._camera.up = this.options._yup ? new THREE.Vector3(0,1,0) : new THREE.Vector3(0,0,1);
      this._scene.add( this._camera );

      this._toplevel = new THREE.Object3D();

      this._scene.add(this._toplevel);

      this._overall_size = 10;
   }


   JSROOT.TGeoPainter.prototype.startDrawGeometry = function() {
      if (this._geometry['_typename'] == "TGeoNode")  {
         this._nodedraw = true;
         this._stack = [ { toplevel: this._toplevel, node: this._geometry, main: true } ];
      }
      else if (this._geometry['_typename'] == 'TEveGeoShapeExtract') {
         this._nodedraw = false;
         this._stack = [ { toplevel: this._toplevel, node: this._geometry, main: true } ];
      }

      this.accountClear();
   }

   JSROOT.TGeoPainter.prototype.adjustCameraPosition = function() {

      var box = new THREE.Box3().setFromObject(this._toplevel);

      var sizex = box.max.x - box.min.x,
          sizey = box.max.y - box.min.y,
          sizez = box.max.z - box.min.z,
          midx = (box.max.x + box.min.x)/2,
          midy = (box.max.y + box.min.y)/2,
          midz = (box.max.z + box.min.z)/2;

      this._overall_size = 2 * Math.max( sizex, sizey, sizez);

      this._camera.near = this._overall_size / 200;
      this._camera.far = this._overall_size * 500;
      this._camera.updateProjectionMatrix();

      if (this.options._yup)
         this._camera.position.set(midx-this._overall_size, midy+this._overall_size, midz-this._overall_size);
      else
         this._camera.position.set(midx-this._overall_size, midy-this._overall_size, midz+this._overall_size);

      this._lookat = new THREE.Vector3(midx, midy, midz);
      this._camera.lookAt(this._lookat);

      if (this._controls !== null) {
         this._controls.target.copy(this._lookat);
         this._controls.update();
      }
   }

   JSROOT.TGeoPainter.prototype.completeScene = function() {
      if ( this.options._debug || this.options._grid ) {
         if ( this.options._full ) {
            var boxHelper = new THREE.BoxHelper(this._toplevel);
            this._scene.add( boxHelper );
         }
         this._scene.add( new THREE.AxisHelper( 2 * this._overall_size ) );
         this._scene.add( new THREE.GridHelper( Math.ceil( this._overall_size), Math.ceil( this._overall_size ) / 50 ) );
         this.helpText("<font face='verdana' size='1' color='red'><center>Transform Controls<br>" +
               "'T' translate | 'R' rotate | 'S' scale<br>" +
               "'+' increase size | '-' decrease size<br>" +
               "'W' toggle wireframe/solid display<br>"+
         "keep 'Ctrl' down to snap to grid</center></font>");
      }
   }

   JSROOT.TGeoPainter.prototype.drawCount = function() {

      var tm1 = new Date();

      var arg = { cnt : [], maxlvl: -1 };
      var cnt = this.CountGeoVolumes(this._geometry, arg);

      var res = 'Total number: ' + cnt + '<br/>';
      for (var lvl=0;lvl<arg.cnt.length;++lvl) {
         if (arg.cnt[lvl] !== 0)
            res += ('  lvl' + lvl + ': ' + arg.cnt[lvl] + '<br/>');
      }
      res += "Unique volumes: " + arg.map.length + '<br/>';

      if (arg.viscnt === 0) {
         arg.clear(); arg.maxlvl = 9999;
         cnt = this.CountGeoVolumes(this._geometry, arg);
      }

      res += "Visible volumes: " + arg.viscnt + '<br/>';

      if (cnt<200000) {
         this.ScanUniqueVisVolumes(this._geometry, 0, arg);

         for (var n=0;n<arg.map.length;++n)
            if (arg.map[n]._refcnt > 1) {
               res += (arg.map[n]._visible ? "vis" : "map") + n + " " + arg.map[n].fName + "  nref:"+arg.map[n]._refcnt +
               ' chld:'+ arg.map[n]._numvis + "(" + arg.map[n]._numchld + ')' +
               " unique:" + arg.map[n]._visunique + " same:" + arg.map[n]._samematerial;

               if (arg.map[n]._samematerial) {
                  if (arg.map[n]._visunique && (arg.map[n]._numvis>0)) res+=" (can merge with childs in Worker)"; else
                     if ((arg.map[n]._refcnt > 4) && (arg.map[n]._numvis>1)) res+=" (make sense merge in main thread)";
               }

               res += "<br/>";
            }
      }

      var tm2 = new Date();

      res +=  "Elapsed time: " + (tm2.getTime() - tm1.getTime()) + "ms <br/>";

      this.select_main().style('overflow', 'auto').html(res);

      return this.DrawingReady();
   }


   JSROOT.TGeoPainter.prototype.DrawGeometry = function(opt) {
      if (typeof opt !== 'string') opt = "";

      if (opt === 'count')
         return this.drawCount();

      var size = this.size_for_3d();

      this.options = this.decodeOptions(opt);

      if (!('_yup' in this.options))
         this.options._yup = this.svg_canvas().empty();

      this._webgl = JSROOT.Painter.TestWebGL();

      this._data = { cnt: [], maxlvl : this.options.maxlvl }; // now count volumes which should go to the processing

      var total = this.CountGeoVolumes(this._geometry, this._data);

      // if no any volume was selected, probably it is because of visibility flags
      if ((total>0) && (this._data.viscnt == 0) && (this.options.maxlvl < 0)) {
         this._data.clear();
         this._data.maxlvl = 1111;
         total = this.CountGeoVolumes(this._geometry, this._data);
      }

      var maxlimit = this._webgl ? 1e7 : 1e4;

      if ((this._data.maxlvl === 1111) && (total > maxlimit))  {
         var sum = 0;
         for (var lvl=1; lvl < this._data.cnt.length; ++lvl) {
            sum += this._data.cnt.cnt[lvl];
            if (sum > maxlimit) {
               this._data.maxlvl = lvl - 1;
               this._data.clear();
               this.CountGeoVolumes(this._geometry, this._data);
               break;
            }
         }
      }

      this.createScene(this._webgl, size.width, size.height, window.devicePixelRatio);

      this.add_3d_canvas(this._renderer.domElement);

      this.startDrawGeometry();

      this._startm = new Date().getTime();

      this._drawcnt = 0; // counter used to build meshes

      this.CreateToolbar( { container: this.select_main().node() } );

      return this.continueDraw();
   }

   JSROOT.TGeoPainter.prototype.continueDraw = function() {
      var curr = new Date().getTime();

      var log = "";

      while(true) {
         if (this.drawNode()) {
            this._drawcnt++;
            log = "Creating meshes " + this._drawcnt;
         } else
            break;

         var now = new Date().getTime();

         if (now - curr > 300) {
            JSROOT.progress(log);
            setTimeout(this.continueDraw.bind(this), 0);
            return this;
         }

         // stop creation, render as is
         if (now - this._startm > 1e5) break;
      }

      var t2 = new Date().getTime();
      JSROOT.console('Create tm = ' + (t2-this._startm) + ' geom ' + this._num_geom + ' vertices ' + this._num_vertices + ' faces ' + this._num_faces + ' meshes ' + this._num_meshes);

      if (t2 - this._startm > 300) {
         JSROOT.progress('Rendering geometry');
         setTimeout(this.completeDraw.bind(this, true), 0);
         return this;
      }

      return this.completeDraw();
   }

   JSROOT.TGeoPainter.prototype.Render3D = function(tmout) {
      if (tmout === undefined) tmout = 5; // by default, rendering happens with timeout

      if (tmout <= 0) {
         if ('render_tmout' in this)
            clearTimeout(this['render_tmout']);

         var tm1 = new Date();

         // do rendering, most consuming time
         this._renderer.render(this._scene, this._camera);

         var tm2 = new Date();

         delete this['render_tmout'];

         if (this.first_render_tm === 0) {
            this.first_render_tm = tm2.getTime() - tm1.getTime();
            JSROOT.console('First render tm = ' + this.first_render_tm);
            this.addControls();
         }

         return;
      }

      // no need to shoot rendering once again
      if ('render_tmout' in this) return;

      this['render_tmout'] = setTimeout(this.Render3D.bind(this,0), tmout);
   }

   JSROOT.TGeoPainter.prototype.completeDraw = function(close_progress) {

      this.adjustCameraPosition();

      this.completeScene();

      if (this.options._axis) {
         var axis = JSROOT.Create("TNamed");
         axis._typename = "TAxis3D";
         axis._main = this;
         JSROOT.draw(this.divid, axis); // it will include drawing of
      }

      this.Render3D();

      if (close_progress) JSROOT.progress();

      this._data.clear();

      // pointer used in the event handlers
      var pthis = this;
      var dom = this.select_main().node();

      if (dom !== null) {
         dom.tabIndex = 0;
         dom.focus();
         dom.onkeypress = function(e) {
            if (!e) e = event;
            switch ( e.keyCode ) {
               case 87:  // W
               case 119: // w
                  pthis.toggleWireFrame(pthis._scene);
                  break;
            }
         };
         dom.onclick = function(e) {
            dom.focus();
         };
      }

      return this.DrawingReady();
   }


   JSROOT.TGeoPainter.prototype.Cleanup = function(first_time) {

      if (first_time === undefined) {
         this.helpText();
         if (this._scene !== null)
            this.deleteChildren(this._scene);
         if ( this._tcontrols !== null)
            this._tcontrols.dispose();
         if (this._controls !== null)
            this._controls.dispose();
      }

      this._scene = null;
      this._scene_width = 0;
      this._scene_height = 0;
      this._renderer = null;
      this._toplevel = null;
      this._stack = null;
      this._camera = null;

      this.first_render_tm = 0;

      this._controls = null;
      this._tcontrols = null;
      this._toolbar = null;
   }

   JSROOT.TGeoPainter.prototype.helpText = function(msg) {
      var id = "jsroot_helptext";
      var box = d3.select("#"+id);
      var newmsg = true;
      if ((typeof msg == "undefined") || (msg==null)) {
         if (box.empty())
            return;
         box.property('stack').pop();
         if (box.property('stack').length==0)
            return box.remove();
         msg = box.property('stack')[box.property('stack').length-1]; // show prvious message
         newmsg = false;
      }
      if (box.empty()) {
         box = d3.select(document.body)
           .append("div")
           .attr("id", id)
           .attr("class","progressbox")
           .property("stack",new Array);

         box.append("p");
      }
      box.select("p").html(msg);
      if (newmsg) {
         box.property('stack').push(msg);
         box.property("showtm", new Date);
      }
   }

   JSROOT.TGeoPainter.prototype.CheckResize = function() {

      var pad_painter = this.pad_painter();

      // firefox is the only browser which correctly supports resize of embedded canvas,
      // for others we should force canvas redrawing at every step
      if (pad_painter)
         if (!pad_painter.CheckCanvasResize(size, JSROOT.browser.isFirefox ? false : true)) return false;

      var size3d = this.size_for_3d();

      if ((this._scene_width === size3d.width) && (this._scene_height === size3d.height)) return false;
      if ((size3d.width<10) || (size3d.height<10)) return;

      this._scene_width = size3d.width;
      this._scene_height = size3d.height;

      this._camera.aspect = this._scene_width / this._scene_height;
      this._camera.updateProjectionMatrix();

      this._renderer.setSize( this._scene_width, this._scene_height );

      this.Render3D();

      return true;
   }

   JSROOT.TGeoPainter.prototype.ownedByTransformControls = function(child) {
      var obj = child.parent;
      while (obj && !(obj instanceof THREE.TransformControls) ) {
         obj = obj.parent;
      }
      return (obj && (obj instanceof THREE.TransformControls));
   }

   JSROOT.TGeoPainter.prototype.toggleWireFrame = function(obj) {
      var painter = this;

      var f = function(obj2) {
         if ( obj2.hasOwnProperty("material") && !(obj2 instanceof THREE.GridHelper) ) {
            if (!painter.ownedByTransformControls(obj2))
               obj2.material.wireframe = !obj2.material.wireframe;
         }
      }
      obj.traverse(f);
      this.Render3D();
   }

   JSROOT.TGeoPainter.prototype.deleteChildren = function(obj) {
      if ((typeof obj['children'] != 'undefined') && (obj['children'] instanceof Array)) {
         for ( var i=obj.children.length-1; i>=0; i-- ) {
            var ob = obj.children[i];
            this.deleteChildren(ob);
            try {
               obj.remove(obj.children[i]);
            } catch(e) {}
            try {
               ob.geometry.dispose();
               ob.geometry = null;
            } catch(e) {}
            try {
               ob.material.dispose();
               ob.material = null;
            } catch(e) {}
            try {
               ob.texture.dispose();
               ob.texture = null;
            } catch(e) {}
            ob = null;
            obj.children[i] = null;
         }
         obj.children = null;
      }
      obj = null;
   }

   JSROOT.Painter.drawGeometry = function(divid, geometry, opt) {

      // create painter and add it to canvas
      JSROOT.extend(this, new JSROOT.TGeoPainter(geometry));

      this.SetDivId(divid, 5);

      return this.DrawGeometry(opt);
   }

   // ===================================================================================

   JSROOT.TAxis3DPainter = function( axis ) {
      JSROOT.TObjectPainter.call( this, axis );

      this.axis = axis;
   }

   JSROOT.TAxis3DPainter.prototype = Object.create( JSROOT.TObjectPainter.prototype );

   JSROOT.TAxis3DPainter.prototype.GetObject = function() {
      return this.axis;
   }

   JSROOT.TAxis3DPainter.prototype.Draw3DAxis = function() {
      var main = this.main_painter();

      if ((main === null) && ('_main' in this.axis))
         main = this.axis._main; // simple workaround to get geo painter

      if ((main === null) || (main._toplevel === undefined))
         return console.warn('no geo object found for 3D axis drawing');

      var box = new THREE.Box3().setFromObject(main._toplevel);

      this.xmin = box.min.x; this.xmax = box.max.x;
      this.ymin = box.min.y; this.ymax = box.max.y;
      this.zmin = box.min.z; this.zmax = box.max.z;

      this.options = { Logx: false, Logy: false, Logz: false };

      this.size3d = 0; // use min/max values directly as graphical coordinates

      this['DrawXYZ'] = JSROOT.Painter.HPainter_DrawXYZ;

      this.toplevel = main._toplevel;

      this.DrawXYZ();

      main.adjustCameraPosition();

      main.Render3D();
   }

   JSROOT.Painter.drawAxis3D = function(divid, axis, opt) {

      JSROOT.extend(this, new JSROOT.TAxis3DPainter(axis));

      if (!('_main' in axis))
         this.SetDivId(divid);

      this.Draw3DAxis();

      return this.DrawingReady();
   }

   // ===============================================================================

   JSROOT.expandGeoList = function(item, lst) {
      if ((lst==null) || !('arr' in lst) || (lst.arr.length==0)) return;

      item['_more'] = true;
      item['_geolst'] = lst;

      item['_get'] = function(item, itemname, callback) {
         if ('_geolst' in item)
            JSROOT.CallBack(callback, item, item._geolst);

         if ('_geoobj' in item)
            return JSROOT.CallBack(callback, item, item._geoobj);

         JSROOT.CallBack(callback, item, null);
      }
      item['_expand'] = function(node, lst) {
         // only childs
         if (!('arr' in lst)) return false;

         node['_childs'] = [];

         for (var n in lst.arr) {
            var obj = lst.arr[n];
            var sub = {
               _kind : "ROOT." + obj._typename,
               _name : obj.fName,
               _title : obj.fTitle,
               _parent : node,
               _geoobj : obj
            };

            if (obj._typename == "TGeoMaterial") sub._icon = "img_geomaterial"; else
            if (obj._typename == "TGeoMedium") sub._icon = "img_geomedium"; else
            if (obj._typename == "TGeoMixture") sub._icon = "img_geomixture";

            node['_childs'].push(sub);
         }

         return true;
      }
   };

   JSROOT.provideGeoVisStyle = function(volume) {
      var res = "";

      if (JSROOT.TestGeoAttBit(volume, JSROOT.EGeoVisibilityAtt.kVisThis))
         res += " geovis_this";

      if (JSROOT.TestGeoAttBit(volume, JSROOT.EGeoVisibilityAtt.kVisDaughters))
         res += " geovis_daughters";

      return res;
   }

   JSROOT.provideGeoMenu = function(menu, item, hpainter) {
      if (! ('_volume' in item)) return false;

      menu.add("separator");
      var vol = item._volume;

      function ToggleMenuBit(arg) {
         JSROOT.ToggleGeoAttBit(vol, arg);
         item._icon = item._icon.split(" ")[0] + JSROOT.provideGeoVisStyle(vol);
         hpainter.UpdateTreeNode(item);
      }

      menu.addchk(JSROOT.TestGeoAttBit(vol, JSROOT.EGeoVisibilityAtt.kVisNone), "Invisible",
            JSROOT.EGeoVisibilityAtt.kVisNone, ToggleMenuBit);
      menu.addchk(JSROOT.TestGeoAttBit(vol, JSROOT.EGeoVisibilityAtt.kVisThis), "Visible",
            JSROOT.EGeoVisibilityAtt.kVisThis, ToggleMenuBit);
      menu.addchk(JSROOT.TestGeoAttBit(vol, JSROOT.EGeoVisibilityAtt.kVisDaughters), "Daughters",
            JSROOT.EGeoVisibilityAtt.kVisDaughters, ToggleMenuBit);
      menu.addchk(JSROOT.TestGeoAttBit(vol, JSROOT.EGeoVisibilityAtt.kVisOneLevel), "1lvl daughters",
            JSROOT.EGeoVisibilityAtt.kVisOneLevel, ToggleMenuBit);

      return true;
   }

   JSROOT.geoIconClick = function(hitem) {
      if ((hitem==null) || (hitem._volume == null)) return false;
      JSROOT.ToggleGeoAttBit(hitem._volume, JSROOT.EGeoVisibilityAtt.kVisDaughters);
      hitem._icon = hitem._icon.split(" ")[0] + JSROOT.provideGeoVisStyle(hitem._volume);
      return true; // hpainter.UpdateTreeNode(hitem);
   }

   JSROOT.expandGeoVolume = function(parent, volume, arg) {

      if ((parent == null) || (volume==null)) return false;

      var item = {
         _kind : "ROOT.TGeoVolume",
         _name : (arg!=null) ? arg : volume.fName,
         _title : volume.fTitle,
         _parent : parent,
         _volume : volume, // keep direct reference
         _more : (volume['fNodes'] !== undefined) && (volume['fNodes'] !== null),
         _menu : JSROOT.provideGeoMenu,
         _icon_click : JSROOT.geoIconClick,
         _get : function(item, itemname, callback) {
            if ((item!=null) && (item._volume != null))
               return JSROOT.CallBack(callback, item, item._volume);

            JSROOT.CallBack(callback, item, null);
         },
      };

      if (item['_more'])
        item['_expand'] = function(node, obj) {
           var subnodes = obj['fNodes']['arr'];
           for (var i in subnodes)
              JSROOT.expandGeoVolume(node, subnodes[i]['fVolume']);
           return true;
        }

      if (item._title == "")
         if (volume._typename != "TGeoVolume") item._title = volume._typename;

      if (volume['fShape']!=null) {
         if (item._title == "")
            item._title = volume['fShape']._typename;

         switch (volume['fShape']._typename) {
            case "TGeoArb8" : item._icon = "img_geoarb8"; break;
            case "TGeoCone" : item._icon = "img_geocone"; break;
            case "TGeoConeSeg" : item._icon = "img_geoconeseg"; break;
            case "TGeoCompositeShape" : item._icon = "img_geocomposite"; break;
            case "TGeoTube" : item._icon = "img_geotube"; break;
            case "TGeoTubeSeg" : item._icon = "img_geotubeseg"; break;
            case "TGeoPara" : item._icon = "img_geopara"; break;
            case "TGeoParaboloid" : item._icon = "img_geoparab"; break;
            case "TGeoPcon" : item._icon = "img_geopcon"; break;
            case "TGeoPgon" : item._icon = "img_geopgon"; break;
            case "TGeoShapeAssembly" : item._icon = "img_geoassembly"; break;
            case "TGeoSphere" : item._icon = "img_geosphere"; break;
            case "TGeoTorus" : item._icon = "img_geotorus"; break;
            case "TGeoTrd1" : item._icon = "img_geotrd1"; break;
            case "TGeoTrd2" : item._icon = "img_geotrd2"; break;
            case "TGeoXtru" : item._icon = "img_geoxtru"; break;
            case "TGeoTrap" : item._icon = "img_geotrap"; break;
            case "TGeoGtra" : item._icon = "img_geogtra"; break;
            case "TGeoEltu" : item._icon = "img_geoeltu"; break;
            case "TGeoHype" : item._icon = "img_geohype"; break;
            case "TGeoCtub" : item._icon = "img_geoctub"; break;
         }
      }

      if (!('_childs' in parent)) parent['_childs'] = [];

      if (!('_icon' in item))
         item._icon = item['_more'] ? "img_geocombi" : "img_geobbox";

      item._icon += JSROOT.provideGeoVisStyle(volume);

      // avoid name duplication of the items
      for (var cnt=0;cnt<1000000;cnt++) {
         var curr_name = item._name;
         if (curr_name.length == 0) curr_name = "item";
         if (cnt>0) curr_name+= "_"+cnt;
         // avoid name duplication
         for (var n in parent['_childs']) {
            if (parent['_childs'][n]['_name'] == curr_name) {
               curr_name = ""; break;
            }
         }
         if (curr_name.length > 0) {
            if (cnt>0) item._name = curr_name;
            break;
         }
      }

      parent['_childs'].push(item);

      return true;
   }

   JSROOT.expandGeoManagerHierarchy = function(hitem, obj) {
      if ((hitem==null) || (obj==null)) return false;

      hitem['_childs'] = [];

      var item1 = { _name : "Materials", _kind : "Folder", _title : "list of materials" };
      JSROOT.expandGeoList(item1, obj.fMaterials);
      hitem['_childs'].push(item1);

      var item2 = { _name : "Media", _kind : "Folder", _title : "list of media" };
      JSROOT.expandGeoList(item2, obj.fMedia);
      hitem['_childs'].push(item2);

      var item3 = { _name : "Tracks", _kind : "Folder", _title : "list of tracks" };
      JSROOT.expandGeoList(item3, obj.fTracks);
      hitem['_childs'].push(item3);

      JSROOT.expandGeoVolume(hitem, obj.fMasterVolume, "Volume");

      return true;
   }

   JSROOT.addDrawFunc({ name: "TGeoVolumeAssembly", icon: 'img_geoassembly', func: JSROOT.Painter.drawGeometry, expand: "JSROOT.expandGeoVolume", opt : "all;count;limit;maxlvl2" });
   JSROOT.addDrawFunc({ name: "TAxis3D", func: JSROOT.Painter.drawAxis3D });

   return JSROOT.Painter;

}));

