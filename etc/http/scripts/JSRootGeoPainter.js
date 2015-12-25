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
      JSROOT.TBasePainter.call( this, geometry );
      this._debug = false;
      this._full = false;
      this._bound = false;
      this._grid = false;
      //this._transformControl = null;
      //this._translationSnap = 100;
      this._geometry = geometry;
      this._scene = null;
      this._renderer = null;
      var _opt = JSROOT.GetUrlOption('_grid');
      if (_opt !== null && _opt == "true") this._grid = true;
      var _opt = JSROOT.GetUrlOption('_debug');
      if (_opt !== null && _opt == "true") { this._debug = true; this._grid = true; }
      if (_opt !== null && _opt == "bound") { this._debug = true; this._grid = true; this._bound = true; }
      if (_opt !== null && _opt == "full") { this._debug = true; this._grid = true; this._full = true; this._bound = true; }
   }

   JSROOT.TGeoPainter.prototype = Object.create( JSROOT.TBasePainter.prototype );

   JSROOT.TGeoPainter.prototype.GetObject = function() {
      return this._geometry;
   }

   JSROOT.TGeoPainter.prototype.addControls = function(renderer, scene, camera) {

      if (typeof renderer.domElement.trackballControls !== 'undefined' &&
          renderer.domElement.trackballControls !== null) return;

      // add 3D mouse interactive functions
      renderer.domElement.clock = new THREE.Clock();
      renderer.domElement.trackballControls = new THREE.TrackballControls(camera, renderer.domElement);
      renderer.domElement.trackballControls.rotateSpeed = 5.0;
      renderer.domElement.trackballControls.zoomSpeed = 0.8;
      renderer.domElement.trackballControls.panSpeed = 0.2;
      renderer.domElement.trackballControls.noZoom = false;
      renderer.domElement.trackballControls.noPan = false;
      renderer.domElement.trackballControls.staticMoving = false;
      renderer.domElement.trackballControls.dynamicDampingFactor = 0.25;
      renderer.domElement.trackballControls.target.set(0,0,0);
      renderer.domElement.transformControl = null;

      renderer.domElement.render = function() {
         var delta = renderer.domElement.clock.getDelta();
         if ( renderer.domElement.transformControl !== null )
            renderer.domElement.transformControl.update();
         renderer.domElement.trackballControls.update(delta);
         renderer.render(scene, camera);
      }

      if ( this._debug || this._grid ) {
         renderer.domElement.transformControl = new THREE.TransformControls( camera, renderer.domElement );
         renderer.domElement.transformControl.addEventListener( 'change', renderer.domElement.render );
         scene.add( renderer.domElement.transformControl );
         //renderer.domElement.transformControl.setSize( 1.1 );

         window.addEventListener( 'keydown', function ( event ) {
            switch ( event.keyCode ) {
               case 81: // Q
                  renderer.domElement.transformControl.setSpace( renderer.domElement.transformControl.space === "local" ? "world" : "local" );
                  break;
               case 17: // Ctrl
                  renderer.domElement.transformControl.setTranslationSnap( renderer.domElement._translationSnap );
                  renderer.domElement.transformControl.setRotationSnap( THREE.Math.degToRad( 15 ) );
                  break;
               case 84: // T (Translate)
                  renderer.domElement.transformControl.setMode( "translate" );
                  break;
               case 82: // R (Rotate)
                  renderer.domElement.transformControl.setMode( "rotate" );
                  break;
               case 83: // S (Scale)
                  renderer.domElement.transformControl.setMode( "scale" );
                  break;
               case 187:
               case 107: // +, =, num+
                  renderer.domElement.transformControl.setSize( renderer.domElement.transformControl.size + 0.1 );
                  break;
               case 189:
               case 109: // -, _, num-
                  renderer.domElement.transformControl.setSize( Math.max( renderer.domElement.transformControl.size - 0.1, 0.1 ) );
                  break;
            }
         });
         window.addEventListener( 'keyup', function ( event ) {
            switch ( event.keyCode ) {
               case 17: // Ctrl
                  renderer.domElement.transformControl.setTranslationSnap( null );
                  renderer.domElement.transformControl.setRotationSnap( null );
                  break;
            }
         });

      }
      renderer.domElement._timeoutFunc = null;
      renderer.domElement._animationId = null;
      var mouseover = true;
      function animate() {
         if ( mouseover === true ) {
            renderer.domElement._timeoutFunc = setTimeout(function() {
               renderer.domElement._animationId = requestAnimationFrame(animate, renderer.domElement);
            }, 1000 / 30);
         }
         renderer.domElement.render();
      }
      /*
      $(renderer.domElement).on('mouseover', function(e) {
         mouseover = true;
         animate();
      }).on('mouseout', function(){
         mouseover = false;
      });
      */
      animate();
   }

   JSROOT.TGeoPainter.prototype.createCube = function( shape, material, volume ) {
      var geometry = new THREE.BoxGeometry( shape['fDX'], shape['fDY'], shape['fDZ'] );
      return new THREE.Mesh( geometry, material );
   }

   JSROOT.TGeoPainter.prototype.createPolygon = function( shape, material, rotation_matrix ) {
      var radiusSegments = 60;
      if ( shape['_typename'] == "TGeoPgon" )
         radiusSegments = shape['fNedges'];
      var outerRadius = [];
      var innerRadius = [];
      var tube = [], tubeMesh = [];
      var face = [], faceMesh = [];
      var end = [], endMesh = [];
      var thetaStart = 0
      var thetaLength = 360;
      thetaStart = shape['fPhi1'] + 90;
      thetaLength = shape['fDphi'];
      var draw_faces = (thetaLength < 360) ? true : false;
      if (rotation_matrix !== null && rotation_matrix[4] === -1 &&
          rotation_matrix[0] === 1 && rotation_matrix[8] === 1)
         thetaStart += 180;
      thetaStart *= (Math.PI / 180.0);
      thetaLength *= (Math.PI / 180.0);
      var geometry = new THREE.Geometry();

      for (var i=0; i<shape['fNz']; ++i) {
         outerRadius[i] = shape['fRmax'][i]/2;
         innerRadius[i] = shape['fRmin'][i]/2;
         if (innerRadius[i] <= 0) innerRadius[i] = 0.0000001;
      }
      for (var n=0; n<shape['fNz']; ++n) {
         var seg = n*2;
         var DZ = (shape['fZ'][n+1]-shape['fZ'][n])/2;
         tube[seg] = new THREE.CylinderGeometry(outerRadius[n+1], outerRadius[n],
                  DZ, radiusSegments, 1, true, thetaStart, thetaLength);
         tube[seg].applyMatrix( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );
         tube[seg].faceVertexUvs[0] = [];  // workaround to avoid warnings from three.js
         tubeMesh[seg] = new THREE.Mesh( tube[seg], material );
         tubeMesh[seg].translateZ( 0.5 * (shape['fZ'][n] + DZ) );

         tube[seg+1] = new THREE.CylinderGeometry(innerRadius[n+1], innerRadius[n],
                  DZ, radiusSegments, 1, true, thetaStart, thetaLength);
         tube[seg+1].applyMatrix( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );
         tube[seg+1].faceVertexUvs[0] = [];  // workaround to avoid warnings from three.js

         tubeMesh[seg+1] = new THREE.Mesh( tube[seg+1], material );
         tubeMesh[seg+1].translateZ( 0.5 * (shape['fZ'][n] + DZ) );

         if ( n >= (shape['fNz']-2) ) {
            end[seg] = new THREE.Geometry();
            for (i = 0; i < radiusSegments; ++i){
               var j = i;
               var k = i*6;
               end[seg].vertices.push(tube[seg].vertices[j+0]/*.clone()*/);
               end[seg].vertices.push(tube[seg].vertices[j+1]/*.clone()*/);
               end[seg].vertices.push(tube[seg+1].vertices[j+0]/*.clone()*/);
               end[seg].faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
               end[seg].vertices.push(tube[seg+1].vertices[j+0]/*.clone()*/);
               end[seg].vertices.push(tube[seg+1].vertices[j+1]/*.clone()*/);
               end[seg].vertices.push(tube[seg].vertices[j+1]/*.clone()*/);
               end[seg].faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
            };
            end[seg].mergeVertices();
            end[seg].computeFaceNormals();
            endMesh[seg] = new THREE.Mesh( end[seg], material );
            endMesh[seg].translateZ( 0.5 * (shape['fZ'][n] + DZ) );
         }

         if ( draw_faces ) {
            face[seg] = new THREE.Geometry();
            face[seg].vertices.push(tube[seg].vertices[0]/*.clone()*/);
            face[seg].vertices.push(tube[seg].vertices[tube[seg].vertices.length/2]/*.clone()*/);
            face[seg].vertices.push(tube[seg+1].vertices[tube[seg].vertices.length/2]/*.clone()*/);
            face[seg].faces.push( new THREE.Face3( 0, 1, 2 ) );
            face[seg].vertices.push(tube[seg+1].vertices[0]/*.clone()*/);
            face[seg].vertices.push(tube[seg+1].vertices[tube[seg].vertices.length/2]/*.clone()*/);
            face[seg].vertices.push(tube[seg].vertices[0]/*.clone()*/);
            face[seg].faces.push( new THREE.Face3( 3, 4, 5 ) );
            face[seg].mergeVertices();
            face[seg].computeFaceNormals();
            faceMesh[seg] = new THREE.Mesh( face[seg], material );
            faceMesh[seg].translateZ( 0.5 * (shape['fZ'][n] + DZ) );

            face[seg+1] = new THREE.Geometry();
            face[seg+1].vertices.push(tube[seg].vertices[radiusSegments]/*.clone()*/);
            face[seg+1].vertices.push(tube[seg].vertices[tube[seg].vertices.length-1]/*.clone()*/);
            face[seg+1].vertices.push(tube[seg+1].vertices[tube[seg].vertices.length-1]/*.clone()*/);
            face[seg+1].faces.push( new THREE.Face3( 0, 1, 2 ) );
            face[seg+1].vertices.push(tube[seg+1].vertices[radiusSegments]/*.clone()*/);
            face[seg+1].vertices.push(tube[seg+1].vertices[tube[seg].vertices.length-1]/*.clone()*/);
            face[seg+1].vertices.push(tube[seg].vertices[radiusSegments]/*.clone()*/);
            face[seg+1].faces.push( new THREE.Face3( 3, 4, 5 ) );
            face[seg+1].mergeVertices();
            face[seg+1].computeFaceNormals();
            faceMesh[seg+1] = new THREE.Mesh( face[seg+1], material );
            faceMesh[seg+1].translateZ( 0.5 * (shape['fZ'][n] + DZ) );
         }
         if ( n == 0 ) {
            end[seg+1] = new THREE.Geometry();
            for (i = 0; i < radiusSegments; ++i) {
               var j = i;
               var k = i*6;
               end[seg+1].vertices.push(tube[seg].vertices[tube[seg].vertices.length-2-j+0]/*.clone()*/);
               end[seg+1].vertices.push(tube[seg].vertices[tube[seg].vertices.length-2-j+1]/*.clone()*/);
               end[seg+1].vertices.push(tube[seg+1].vertices[tube[seg].vertices.length-2-j+0]/*.clone()*/);
               end[seg+1].faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
               end[seg+1].vertices.push(tube[seg+1].vertices[tube[seg].vertices.length-2-j+0]/*.clone()*/);
               end[seg+1].vertices.push(tube[seg+1].vertices[tube[seg].vertices.length-2-j+1]/*.clone()*/);
               end[seg+1].vertices.push(tube[seg].vertices[tube[seg].vertices.length-2-j+1]/*.clone()*/);
               end[seg+1].faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
            }
            end[seg+1].mergeVertices();
            end[seg+1].computeFaceNormals();
            endMesh[seg+1] = new THREE.Mesh( end[seg+1], material );
            endMesh[seg+1].translateZ( 0.5 * (shape['fZ'][n] + DZ) );
         }

         tubeMesh[seg].updateMatrix();
         geometry.merge(tubeMesh[seg].geometry, tubeMesh[seg].matrix);
         tubeMesh[seg+1].updateMatrix();
         geometry.merge(tubeMesh[seg+1].geometry, tubeMesh[seg+1].matrix);
         if ( draw_faces ) {
            faceMesh[seg].updateMatrix();
            geometry.merge(faceMesh[seg].geometry, faceMesh[seg].matrix);
            faceMesh[seg+1].updateMatrix();
            geometry.merge(faceMesh[seg+1].geometry, faceMesh[seg+1].matrix);
         }
         if ( n >= (shape['fNz']-2) ) {
            endMesh[seg].updateMatrix();
            geometry.merge(endMesh[seg].geometry, endMesh[seg].matrix);
         }
         if ( n == 0 ) {
            endMesh[seg+1].updateMatrix();
            geometry.merge(endMesh[seg+1].geometry, endMesh[seg+1].matrix);
         }
      }
      //geometry.computeFaceNormals();
      return new THREE.Mesh( geometry, material );
   }

   JSROOT.TGeoPainter.prototype.createSphere = function( shape, material ) {
      var widthSegments = 32;
      var heightSegments = 32;
      var outerRadius = shape['fRmax'];
      var innerRadius = shape['fRmin'];
      var phiStart = shape['fPhi1'] + 180;
      var phiLength = shape['fPhi2'] - shape['fPhi1'];
      var thetaStart = shape['fTheta1'];
      var thetaLength = shape['fTheta2'] - shape['fTheta1'];
      thetaStart *= (Math.PI / 180.0);
      thetaLength *= (Math.PI / 180.0);
      phiStart *= (Math.PI / 180.0);
      phiLength *= (Math.PI / 180.0);
      var geometry = new THREE.Geometry();
      if (innerRadius <= 0) innerRadius = 0.0000001;

      var outerSphere = new THREE.SphereGeometry( outerRadius/2, widthSegments,
            heightSegments, phiStart, phiLength, thetaStart, thetaLength );
      outerSphere.applyMatrix( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );
      var outerSphereMesh = new THREE.Mesh( outerSphere, material );

      var innerSphere = new THREE.SphereGeometry( innerRadius/2, widthSegments,
            heightSegments, phiStart, phiLength, thetaStart, thetaLength );
      innerSphere.applyMatrix( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );
      var innerSphereMesh = new THREE.Mesh( innerSphere, material );

      var first = new THREE.Geometry();
      for (i = 0; i < widthSegments; ++i){
         var j = i;
         var k = i*6;
         first.vertices.push(outerSphere.vertices[j+0]/*.clone()*/);
         first.vertices.push(outerSphere.vertices[j+1]/*.clone()*/);
         first.vertices.push(innerSphere.vertices[j+0]/*.clone()*/);
         first.faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
         first.vertices.push(innerSphere.vertices[j+0]/*.clone()*/);
         first.vertices.push(innerSphere.vertices[j+1]/*.clone()*/);
         first.vertices.push(outerSphere.vertices[j+1]/*.clone()*/);
         first.faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
      };
      first.mergeVertices();
      first.computeFaceNormals();
      var firstMesh = new THREE.Mesh( first, material );

      var second = new THREE.Geometry();
      for (i = 0; i < widthSegments; ++i) {
         var j = i;
         var k = i*6;
         second.vertices.push(outerSphere.vertices[outerSphere.vertices.length-2-j+0]/*.clone()*/);
         second.vertices.push(outerSphere.vertices[outerSphere.vertices.length-2-j+1]/*.clone()*/);
         second.vertices.push(innerSphere.vertices[outerSphere.vertices.length-2-j+0]/*.clone()*/);
         second.faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
         second.vertices.push(innerSphere.vertices[outerSphere.vertices.length-2-j+0]/*.clone()*/);
         second.vertices.push(innerSphere.vertices[outerSphere.vertices.length-2-j+1]/*.clone()*/);
         second.vertices.push(outerSphere.vertices[outerSphere.vertices.length-2-j+1]/*.clone()*/);
         second.faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
      };
      second.mergeVertices();
      second.computeFaceNormals();
      var secondMesh = new THREE.Mesh( second, material );

      var face1 = new THREE.Geometry();
      for (i = 0; i < widthSegments; ++i){
         var j = widthSegments*i;
         var k = i*6;
         face1.vertices.push(outerSphere.vertices[j+i]/*.clone()*/);
         face1.vertices.push(outerSphere.vertices[j+widthSegments+i+1]/*.clone()*/);
         face1.vertices.push(innerSphere.vertices[j+i]/*.clone()*/);
         face1.faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
         face1.vertices.push(innerSphere.vertices[j+i]/*.clone()*/);
         face1.vertices.push(outerSphere.vertices[j+widthSegments+i+1]/*.clone()*/);
         face1.vertices.push(innerSphere.vertices[j+widthSegments+i+1]/*.clone()*/);
         face1.faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
      }
      face1.mergeVertices();
      face1.computeFaceNormals();
      var face1Mesh = new THREE.Mesh( face1, material );

      var face2 = new THREE.Geometry();
      for (i = 0; i < widthSegments; ++i){
         var j = widthSegments*(i+1);
         var k = i*6;
         face2.vertices.push(outerSphere.vertices[j+i]/*.clone()*/);
         face2.vertices.push(outerSphere.vertices[j+widthSegments+i+1]/*.clone()*/);
         face2.vertices.push(innerSphere.vertices[j+i]/*.clone()*/);
         face2.faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
         face2.vertices.push(innerSphere.vertices[j+i]/*.clone()*/);
         face2.vertices.push(outerSphere.vertices[j+widthSegments+i+1]/*.clone()*/);
         face2.vertices.push(innerSphere.vertices[j+widthSegments+i+1]/*.clone()*/);
         face2.faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
      }
      face2.mergeVertices();
      face2.computeFaceNormals();
      var face2Mesh = new THREE.Mesh( face2, material );

      outerSphereMesh.updateMatrix();
      geometry.merge(outerSphereMesh.geometry, outerSphereMesh.matrix);
      innerSphereMesh.updateMatrix();
      geometry.merge(innerSphereMesh.geometry, innerSphereMesh.matrix);
      firstMesh.updateMatrix();
      geometry.merge(firstMesh.geometry, firstMesh.matrix);
      secondMesh.updateMatrix();
      geometry.merge(secondMesh.geometry, secondMesh.matrix);
      face1Mesh.updateMatrix();
      geometry.merge(face1Mesh.geometry, face1Mesh.matrix);
      face2Mesh.updateMatrix();
      geometry.merge(face2Mesh.geometry, face2Mesh.matrix);
      //geometry.computeFaceNormals();

      return new THREE.Mesh( geometry, material );
   }

   JSROOT.TGeoPainter.prototype.createTorus = function( shape, material ) {
      var radius = shape['fR'];
      var innerTube = shape['fRmin'];
      var outerTube = shape['fRmax'];
      var radialSegments = 30;
      var tubularSegments = 60;
      var arc = shape['fDphi'] - shape['fPhi1'];
      var rotation = shape['fPhi1'];
      rotation *= (Math.PI / 180.0);
      arc *= (Math.PI / 180.0);

      var geometry = new THREE.Geometry();

      var outerTorus = new THREE.TorusGeometry( radius/2, outerTube/2, radialSegments, tubularSegments, arc );
      outerTorus.applyMatrix( new THREE.Matrix4().makeRotationZ( rotation ) );
      var outerTorusMesh = new THREE.Mesh( outerTorus, material );

      var innerTorus = new THREE.TorusGeometry( radius/2, innerTube/2, radialSegments, tubularSegments, arc );
      innerTorus.applyMatrix( new THREE.Matrix4().makeRotationZ( rotation ) );
      var innerTorusMesh = new THREE.Mesh( innerTorus, material );

      var first = new THREE.Geometry();
      for (i = 0; i < radialSegments; ++i) {
         var j = i*(tubularSegments+1);
         var k = i*6;
         var l = (i+1)*(tubularSegments+1);
         first.vertices.push(outerTorus.vertices[j]/*.clone()*/);
         first.vertices.push(outerTorus.vertices[l]/*.clone()*/);
         first.vertices.push(innerTorus.vertices[j]/*.clone()*/);
         first.faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
         first.vertices.push(innerTorus.vertices[j]/*.clone()*/);
         first.vertices.push(innerTorus.vertices[l]/*.clone()*/);
         first.vertices.push(outerTorus.vertices[l]/*.clone()*/);
         first.faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
      }
      first.mergeVertices();
      first.computeFaceNormals();
      var firstMesh = new THREE.Mesh( first, material );

      var second = new THREE.Geometry();
      for (i = 0; i < radialSegments; ++i) {
         var j = (i+1)*tubularSegments;
         var k = i*6;
         var l = (i+2)*tubularSegments;
         second.vertices.push(outerTorus.vertices[j+i]/*.clone()*/);
         second.vertices.push(outerTorus.vertices[l+i+1]/*.clone()*/);
         second.vertices.push(innerTorus.vertices[j+i]/*.clone()*/);
         second.faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
         second.vertices.push(innerTorus.vertices[j+i]/*.clone()*/);
         second.vertices.push(innerTorus.vertices[l+i+1]/*.clone()*/);
         second.vertices.push(outerTorus.vertices[l+i+1]/*.clone()*/);
         second.faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
      }

      second.mergeVertices();
      second.computeFaceNormals();
      var secondMesh = new THREE.Mesh( second, material );

      outerTorusMesh.updateMatrix();
      geometry.merge(outerTorusMesh.geometry, outerTorusMesh.matrix);
      innerTorusMesh.updateMatrix();
      geometry.merge(innerTorusMesh.geometry, innerTorusMesh.matrix);
      firstMesh.updateMatrix();
      geometry.merge(firstMesh.geometry, firstMesh.matrix);
      secondMesh.updateMatrix();
      geometry.merge(secondMesh.geometry, secondMesh.matrix);
      //geometry.computeFaceNormals();

      return new THREE.Mesh( geometry, material );
   }

   JSROOT.TGeoPainter.prototype.createTrapezoid = function( shape, material ) {
      if (shape['_typename'] == "TGeoArb8" || shape['_typename'] == "TGeoTrap") {
         // Arb8
         var verticesOfShape = [
            shape['fXY'][0][0], shape['fXY'][0][1], -1*shape['fDZ'],
            shape['fXY'][1][0], shape['fXY'][1][1], -1*shape['fDZ'],
            shape['fXY'][2][0], shape['fXY'][2][1], -1*shape['fDZ'],
            shape['fXY'][3][0], shape['fXY'][3][1], -1*shape['fDZ'],
            shape['fXY'][4][0], shape['fXY'][4][1],    shape['fDZ'],
            shape['fXY'][5][0], shape['fXY'][5][1],    shape['fDZ'],
            shape['fXY'][6][0], shape['fXY'][6][1],    shape['fDZ'],
            shape['fXY'][7][0], shape['fXY'][7][1],    shape['fDZ'],
         ];
      }
      else if (shape['_typename'] == "TGeoTrd1") {
         var verticesOfShape = [
            -shape['fDx1'],  shape['fDY'], -shape['fDZ'],
             shape['fDx1'],  shape['fDY'], -shape['fDZ'],
             shape['fDx1'], -shape['fDY'], -shape['fDZ'],
            -shape['fDx1'], -shape['fDY'], -shape['fDZ'],
            -shape['fDx2'],  shape['fDY'],  shape['fDZ'],
             shape['fDx2'],  shape['fDY'],  shape['fDZ'],
             shape['fDx2'], -shape['fDY'],  shape['fDZ'],
            -shape['fDx2'], -shape['fDY'],  shape['fDZ']
         ];
      }
      else if (shape['_typename'] == "TGeoTrd2") {
         var verticesOfShape = [
            -shape['fDx1'],  shape['fDy1'], -shape['fDZ'],
             shape['fDx1'],  shape['fDy1'], -shape['fDZ'],
             shape['fDx1'], -shape['fDy1'], -shape['fDZ'],
            -shape['fDx1'], -shape['fDy1'], -shape['fDZ'],
            -shape['fDx2'],  shape['fDy2'],  shape['fDZ'],
             shape['fDx2'],  shape['fDy2'],  shape['fDZ'],
             shape['fDx2'], -shape['fDy2'],  shape['fDZ'],
            -shape['fDx2'], -shape['fDy2'],  shape['fDZ']
         ];
      }
      var indicesOfFaces = [
          4,5,6,   4,7,6,   0,3,7,   7,4,0,
          4,5,1,   1,0,4,   6,2,1,   1,5,6,
          7,3,2,   2,6,7,   1,2,3,   3,0,1,
      ];

      var geometry = new THREE.Geometry();
      for (var i = 0; i < 24; i += 3) {
         geometry.vertices.push( new THREE.Vector3( 0.5*verticesOfShape[i], 0.5*verticesOfShape[i+1], 0.5*verticesOfShape[i+2] ) );
      }
      for (var i = 0; i < 36; i += 3) {
         geometry.faces.push( new THREE.Face3( indicesOfFaces[i], indicesOfFaces[i+1], indicesOfFaces[i+2] ) );
      }
      geometry.computeFaceNormals();
      return new THREE.Mesh( geometry, material );
   }

   JSROOT.TGeoPainter.prototype.createTube = function( shape, material, rotation_matrix ) {
      var radiusSegments = 60;
      var outerRadius1, innerRadius1, outerRadius2, innerRadius2;
      if ((shape['_typename'] == "TGeoCone") || (shape['_typename'] == "TGeoConeSeg")) {
         outerRadius1 = shape['fRmax2'];
         innerRadius1 = shape['fRmin2'];
         outerRadius2 = shape['fRmax1'];
         innerRadius2 = shape['fRmin1'];
      }
      else {
         outerRadius1 = outerRadius2 = shape['fRmax'];
         innerRadius1 = innerRadius2 = shape['fRmin'];
      }
      if (innerRadius1 <= 0) innerRadius1 = 0.0000001;
      if (innerRadius2 <= 0) innerRadius2 = 0.0000001;
      var thetaStart = 0
      var thetaLength = 360;
      if ((shape['_typename'] == "TGeoConeSeg") || (shape['_typename'] == "TGeoTubeSeg") ||
           (shape['_typename'] == "TGeoCtub")) {
         thetaStart = shape['fPhi1'] + 90;
         thetaLength = shape['fPhi2'] - shape['fPhi1'];
         if (rotation_matrix !== null && rotation_matrix[4] === -1 &&
             rotation_matrix[0] === 1 && rotation_matrix[8] === 1)
            thetaStart += 180;
      }
      thetaStart *= (Math.PI / 180.0);
      thetaLength *= (Math.PI / 180.0);
      var geometry = new THREE.Geometry();

      var outerTube = new THREE.CylinderGeometry(outerRadius1/2, outerRadius2/2,
               shape['fDZ'], radiusSegments, 1, true, thetaStart, thetaLength);
      outerTube.applyMatrix( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );
      outerTube.faceVertexUvs[0] = [];  // workaround to avoid warnings from three.js
      var outerTubeMesh = new THREE.Mesh( outerTube, material );

      var innerTube = new THREE.CylinderGeometry(innerRadius1/2, innerRadius2/2,
               shape['fDZ'], radiusSegments, 1, true, thetaStart, thetaLength);
      innerTube.applyMatrix( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );
      innerTube.faceVertexUvs[0] = [];  // workaround to avoid warnings from three.js
      var innerTubeMesh = new THREE.Mesh( innerTube, material );

      var first = new THREE.Geometry();
      for (i = 0; i < radiusSegments; ++i){
         var j = i;
         var k = i*6;
         first.vertices.push(outerTube.vertices[j+0]/*.clone()*/);
         first.vertices.push(outerTube.vertices[j+1]/*.clone()*/);
         first.vertices.push(innerTube.vertices[j+0]/*.clone()*/);
         first.faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
         first.vertices.push(innerTube.vertices[j+0]/*.clone()*/);
         first.vertices.push(innerTube.vertices[j+1]/*.clone()*/);
         first.vertices.push(outerTube.vertices[j+1]/*.clone()*/);
         first.faces.push( new THREE.Face3( k+3, k+4, k+5 ) );
      };
      first.mergeVertices();
      first.computeFaceNormals();
      var firstMesh = new THREE.Mesh( first, material );

      var face1Mesh, face2Mesh;

      if ((shape['_typename'] == "TGeoConeSeg") || (shape['_typename'] == "TGeoTubeSeg") ||
          (shape['_typename'] == "TGeoCtub")) {
         var face1 = new THREE.Geometry();
         face1.vertices.push(outerTube.vertices[0]/*.clone()*/);
         face1.vertices.push(outerTube.vertices[outerTube.vertices.length/2]/*.clone()*/);
         face1.vertices.push(innerTube.vertices[outerTube.vertices.length/2]/*.clone()*/);
         face1.faces.push( new THREE.Face3( 0, 1, 2 ) );
         face1.vertices.push(innerTube.vertices[0]/*.clone()*/);
         face1.vertices.push(innerTube.vertices[outerTube.vertices.length/2]/*.clone()*/);
         face1.vertices.push(outerTube.vertices[0]/*.clone()*/);
         face1.faces.push( new THREE.Face3( 3, 4, 5 ) );
         face1.mergeVertices();
         face1.computeFaceNormals();
         face1Mesh = new THREE.Mesh( face1, material );

         var face2 = new THREE.Geometry();
         face2.vertices.push(outerTube.vertices[radiusSegments]/*.clone()*/);
         face2.vertices.push(outerTube.vertices[outerTube.vertices.length-1]/*.clone()*/);
         face2.vertices.push(innerTube.vertices[outerTube.vertices.length-1]/*.clone()*/);
         face2.faces.push( new THREE.Face3( 0, 1, 2 ) );
         face2.vertices.push(innerTube.vertices[radiusSegments]/*.clone()*/);
         face2.vertices.push(innerTube.vertices[outerTube.vertices.length-1]/*.clone()*/);
         face2.vertices.push(outerTube.vertices[radiusSegments]/*.clone()*/);
         face2.faces.push( new THREE.Face3( 3, 4, 5 ) );
         face2.mergeVertices();
         face2.computeFaceNormals();
         face2Mesh = new THREE.Mesh( face2, material );
      }

      var second = new THREE.Geometry();
      for (i = 0; i < radiusSegments; ++i) {
         var j = i;
         var k = i*6;
         second.vertices.push(outerTube.vertices[outerTube.vertices.length-2-j+0]/*.clone()*/);
         second.vertices.push(outerTube.vertices[outerTube.vertices.length-2-j+1]/*.clone()*/);
         second.vertices.push(innerTube.vertices[outerTube.vertices.length-2-j+0]/*.clone()*/);
         second.faces.push( new THREE.Face3( k+0, k+1, k+2 ) );
         second.vertices.push(innerTube.vertices[outerTube.vertices.length-2-j+0]/*.clone()*/);
         second.vertices.push(innerTube.vertices[outerTube.vertices.length-2-j+1]/*.clone()*/);
         second.vertices.push(outerTube.vertices[outerTube.vertices.length-2-j+1]/*.clone()*/);
         second.faces.push( new THREE.Face3( k+3, k+4, k+5 ) );

      };
      second.mergeVertices();
      second.computeFaceNormals();
      var secondMesh = new THREE.Mesh( second, material );

      outerTubeMesh.updateMatrix();
      geometry.merge(outerTubeMesh.geometry, outerTubeMesh.matrix);
      innerTubeMesh.updateMatrix();
      geometry.merge(innerTubeMesh.geometry, innerTubeMesh.matrix);

      if (face1Mesh && face2Mesh) {
         face1Mesh.updateMatrix();
         geometry.merge(face1Mesh.geometry, face1Mesh.matrix);
         face2Mesh.updateMatrix();
         geometry.merge(face2Mesh.geometry, face2Mesh.matrix);
      }
      firstMesh.updateMatrix();
      geometry.merge(firstMesh.geometry, firstMesh.matrix);
      secondMesh.updateMatrix();
      geometry.merge(secondMesh.geometry, secondMesh.matrix);
      //geometry.computeFaceNormals();

      return new THREE.Mesh( geometry, material );
   }

   JSROOT.TGeoPainter.prototype.createMesh = function( shape, material, rotation_matrix ) {
      var mesh = null;
      if (shape['_typename'] == "TGeoBBox") {
         // Cube
         mesh = this.createCube( shape, material );
      }
      else if ((shape['_typename'] == "TGeoArb8") || (shape['_typename'] == "TGeoTrd1") ||
          (shape['_typename'] == "TGeoTrd2") || (shape['_typename'] == "TGeoTrap")) {
         mesh = this.createTrapezoid( shape, material );
      }
      else if ((shape['_typename'] == "TGeoSphere")) {
         mesh = this.createSphere( shape, material );
      }
      else if ((shape['_typename'] == "TGeoCone") || (shape['_typename'] == "TGeoConeSeg") ||
          (shape['_typename'] == "TGeoTube") || (shape['_typename'] == "TGeoTubeSeg")) {
         mesh = this.createTube( shape, material, rotation_matrix );
      }
      else if (shape['_typename'] == "TGeoTorus") {
         mesh = this.createTorus( shape, material );
      }
      else if ( shape['_typename'] == "TGeoPcon" || shape['_typename'] == "TGeoPgon" ) {
         mesh = this.createPolygon( shape, material, rotation_matrix );
      }
      return mesh;
   }

   JSROOT.TGeoPainter.prototype.drawNode = function(scene, toplevel, node, visible) {
      var container = toplevel;
      var volume = node['fVolume'];

      var translation_matrix = [0, 0, 0];
      var rotation_matrix = null;//[1, 0, 0, 0, 1, 0, 0, 0, 1];
      if (typeof node['fMatrix'] != 'undefined' && node['fMatrix'] != null) {
         if (node['fMatrix']['_typename'] == 'TGeoTranslation') {
            translation_matrix = node['fMatrix']['fTranslation'];
         }
         else if (node['fMatrix']['_typename'] == 'TGeoRotation') {
            rotation_matrix = node['fMatrix']['fRotationMatrix'];
         }
         else if (node['fMatrix']['_typename'] == 'TGeoCombiTrans') {
            if (typeof node['fMatrix']['fTranslation'] != 'undefined' &&
                node['fMatrix']['fTranslation'] != null)
               translation_matrix = node['fMatrix']['fTranslation'];
            if (typeof node['fMatrix']['fRotation'] != 'undefined' &&
                node['fMatrix']['fRotation'] != null)
               rotation_matrix = node['fMatrix']['fRotation']['fRotationMatrix'];
         }
      }
      if (node['_typename'] == "TGeoNodeOffset") {
         if (node['fFinder']['_typename'] == 'TGeoPatternX') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternY') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternZ') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternParaX') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternParaY') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternParaZ') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternTrapZ') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternCylR') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternSphR') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternSphTheta') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternSphPhi') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternHoneycomb') {
         }
         if (node['fFinder']['_typename'] == 'TGeoPatternCylPhi') {
            if (typeof node['fFinder']['fSinCos'] === 'undefined') {
               node['fFinder']['fSinCos'] = [];
               for (var i = 0; i<node['fFinder']['fNdivisions']; ++i) {
                  node['fFinder']['fSinCos'][2*i] = Math.sin((Math.PI / 180.0)*(node['fFinder']['fStart']+0.5*node['fFinder']['fStep']+i*node['fFinder']['fStep']));
                  node['fFinder']['fSinCos'][2*i+1] = Math.cos((Math.PI / 180.0)*(node['fFinder']['fStart']+0.5*node['fFinder']['fStep']+i*node['fFinder']['fStep']));
               }
            }
            if (rotation_matrix == null)
               rotation_matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1];
            rotation_matrix[0] = node['fFinder']['fSinCos'][(2*node['fIndex'])+1];
            rotation_matrix[1] = -node['fFinder']['fSinCos'][(2*node['fIndex'])];
            rotation_matrix[3] = node['fFinder']['fSinCos'][(2*node['fIndex'])];
            rotation_matrix[4] = node['fFinder']['fSinCos'][(2*node['fIndex'])+1];
         }
      }
      var fillcolor = JSROOT.Painter.root_colors[volume['fLineColor']];

      var _transparent = true, _helper = false, _opacity = 0.0, _isdrawn = false;
      if (this._debug) _helper = true;
      if (JSROOT.TestGeoAttBit(volume, JSROOT.BIT(7)) || (visible>0)) {
         _transparent = false;
         _opacity = 1.0;
         _isdrawn = true;
      }
      if (typeof volume['fMedium'] != 'undefined' && volume['fMedium'] != null &&
          typeof volume['fMedium']['fMaterial'] != 'undefined' &&
          volume['fMedium']['fMaterial'] != null) {
         var fillstyle = volume['fMedium']['fMaterial']['fFillStyle'];
         var transparency = (fillstyle < 3000 || fillstyle > 3100) ? 0 : fillstyle - 3000;
         if (transparency > 0) {
            _transparent = true;
            _opacity = (100.0 - transparency) / 100.0;
         }
         if (typeof fillcolor == "undefined")
            fillcolor = JSROOT.Painter.root_colors[volume['fMedium']['fMaterial']['fFillColor']];
      }

      var material = new THREE.MeshLambertMaterial( { transparent: _transparent,
               opacity: _opacity, wireframe: false, color: fillcolor,
               side: THREE.DoubleSide, vertexColors: THREE.VertexColors,
               overdraw: false } );
      if ( !_isdrawn ) {
         material.depthWrite = false;
         material.depthTest = false;
         material.visible = false;
      }
      var mesh = this.createMesh(volume['fShape'], material, rotation_matrix);
      if (typeof mesh != 'undefined' && mesh != null) {
         mesh.position.x = 0.5 * translation_matrix[0];
         mesh.position.y = 0.5 * translation_matrix[1];
         mesh.position.z = 0.5 * translation_matrix[2];

         if (rotation_matrix !== null) {
            mesh.rotation.setFromRotationMatrix(
               new THREE.Matrix4().set( rotation_matrix[0], rotation_matrix[1], rotation_matrix[2],   0,
                                        rotation_matrix[3], rotation_matrix[4], rotation_matrix[5],   0,
                                        rotation_matrix[6], rotation_matrix[7], rotation_matrix[8],   0,
                                        0,                                   0,                  0,   1 ) );
         }
         if (_isdrawn && _helper) {
            var helper = new THREE.WireframeHelper(mesh);
            helper.material.color.set(JSROOT.Painter.root_colors[volume['fLineColor']]);
            helper.material.linewidth = volume['fLineWidth'];
            scene.add(helper);
         }
         if (this._debug && this._bound) {
            if (_isdrawn || this._full) {
               var boxHelper = new THREE.BoxHelper( mesh );
               toplevel.add( boxHelper );
            }
         }
         mesh['name'] = node['fName'];
         // add the mesh to the scene
         toplevel.add(mesh);
         //if ( this._debug && this._renderer.domElement.transformControl !== null)
         //   this._renderer.domElement.transformControl.attach( mesh );
         container = mesh;
      }
      if (typeof volume['fNodes'] != 'undefined' && volume['fNodes'] != null) {
         var nodes = volume['fNodes']['arr'];
         for (var i in nodes)
            this.drawNode(scene, container, nodes[i], visible-1);
      }
   }

   JSROOT.TGeoPainter.prototype.drawEveNode = function(scene, toplevel, node) {
      var container = toplevel;
      var shape = node['fShape'];
      var rotation_matrix = null;
      var mesh = null;
      var linecolor = new THREE.Color( node['fRGBALine'][0], node['fRGBALine'][1], node['fRGBALine'][2] );
      var fillcolor = new THREE.Color( node['fRGBA'][0], node['fRGBA'][1], node['fRGBA'][2] );
      var _transparent = true;
      var _helper = false;
      if (this._debug) _helper = true;
      var _opacity = 0.0;
      var _isdrawn = false;
      if (node['fRnrSelf'] == true) {
         _transparent = false;
         _opacity = 1.0;
         _isdrawn = true;
      }
      if ( node['fRGBA'][3] < 1.0) {
         _transparent = true;
         _opacity = node['fRGBA'][3];
      }
      var material = new THREE.MeshLambertMaterial( { transparent: _transparent,
               opacity: _opacity, wireframe: false, color: fillcolor,
               side: THREE.DoubleSide, vertexColors: THREE.VertexColors,
               overdraw: false } );
      if ( !_isdrawn ) {
         material.depthWrite = false;
         material.depthTest = false;
         material.visible = false;
      }
      material.polygonOffset = true;
      material.polygonOffsetFactor = -1;
      if (shape !== null)
         mesh = this.createMesh(shape, material, rotation_matrix);
      if (typeof mesh != 'undefined' && mesh != null) {
         mesh.position.x = 0.5 * node['fTrans'][12];
         mesh.position.y = 0.5 * node['fTrans'][13];
         mesh.position.z = 0.5 * node['fTrans'][14];

         mesh.rotation.setFromRotationMatrix( new THREE.Matrix4().set(
               node['fTrans'][0],  node['fTrans'][4],  node['fTrans'][8],  0,
               node['fTrans'][1],  node['fTrans'][5],  node['fTrans'][9],  0,
               node['fTrans'][2],  node['fTrans'][6],  node['fTrans'][10], 0,
               0, 0, 0, 1 ) );

         if (_isdrawn && _helper) {
            var helper = new THREE.WireframeHelper(mesh);
            helper.material.color.set(JSROOT.Painter.root_colors[volume['fLineColor']]);
            helper.material.linewidth = volume['fLineWidth'];
            scene.add(helper);
         }
         if (this._debug && this._bound) {
            if (_isdrawn || this._full) {
               var boxHelper = new THREE.BoxHelper( mesh );
               toplevel.add( boxHelper );
            }
         }
         mesh['name'] = node['fName'];
         // add the mesh to the scene
         toplevel.add(mesh);
         //if ( this._debug && renderer.domElement.transformControl !== null)
         //   renderer.domElement.transformControl.attach( mesh );
         container = mesh;
      }
      if (typeof node['fElements'] != 'undefined' && node['fElements'] != null) {
         var nodes = node['fElements']['arr'];
         for (var i = 0; i < nodes.length; ++i) {
            var inode = node['fElements']['arr'][i];
            this.drawEveNode(scene, container, inode);
         }
      }
   }

   JSROOT.TGeoPainter.prototype.computeBoundingBox = function( mesh, any ) {
      var bbox = null;
      for (var i = 0; i < mesh.children.length; ++i) {
         var node = mesh.children[i];
         if ( node instanceof THREE.Mesh ) {
            if ( any || node['material']['visible'] ) {
               bbox = new THREE.Box3().setFromObject( node );
               return bbox;
            } else {
               bbox = this.computeBoundingBox( node, any );
               if (bbox != null) return bbox;
            }
         }
      }
      return bbox;
   }

   JSROOT.TGeoPainter.prototype.drawGeometry = function(opt) {
      var rect = this.select_main().node().getBoundingClientRect();

      var w = rect.width, h = rect.height, size = 100;

      if (h < 10) { h = parseInt(0.66*w); this.select_main().style('height', h +"px"); }

      var dom = this.select_main().node();

      // three.js 3D drawing
      this._scene = new THREE.Scene();
      this._scene.fog = new THREE.Fog(0xffffff, 500, 300000);

      this._camera = new THREE.PerspectiveCamera(25, w / h, 1, 100000);
      var pointLight = new THREE.PointLight(0xefefef);
      this._camera.add( pointLight );
      pointLight.position.set( 10, 10, 10 );
      this._scene.add( this._camera );

      /**
       * @author alteredq / http://alteredqualia.com/
       * @author mr.doob / http://mrdoob.com/
       */
      var Detector = {
            canvas : !!window.CanvasRenderingContext2D,
            webgl : (function() {
               try {
                  return !!window.WebGLRenderingContext
                  && !!document.createElement('canvas')
                  .getContext('experimental-webgl');
               } catch (e) {
                  return false;
               }
            })(),
            workers : !!window.Worker,
            fileapi : window.File && window.FileReader
            && window.FileList && window.Blob
      };

      this._renderer = Detector.webgl ? new THREE.WebGLRenderer({ antialias : true, logarithmicDepthBuffer: true  }) :
                       new THREE.CanvasRenderer({antialias : true });
      this._renderer.setPixelRatio( window.devicePixelRatio );
      this._renderer.setClearColor(0xffffff, 1);
      this._renderer.setSize(w, h);

      dom.appendChild(this._renderer.domElement);

      this.SetDivId(); // now one could set painter pointer in child element

      this.addControls(this._renderer, this._scene, this._camera);

      var toplevel = new THREE.Object3D();
      //toplevel.rotation.x = 30 * Math.PI / 180;
      toplevel.rotation.y = 90 * Math.PI / 180;
      this._scene.add(toplevel);

      var overall_size = 10;

      if ((this._geometry['_typename'] == 'TGeoVolume') || (this._geometry['_typename'] == 'TGeoVolumeAssembly'))  {
         var shape = this._geometry['fShape'];
         var top = new THREE.BoxGeometry( shape['fDX'], shape['fDY'], shape['fDZ'] );
         var cube = new THREE.Mesh( top, new THREE.MeshBasicMaterial( {
                  visible: false, transparent: true, opacity: 0.0 } ) );
         toplevel.add(cube);

         this.drawNode(this._scene, cube, { _typename:"TGeoNode", fVolume:this._geometry, fName:"TopLevel" }, opt=="all" ? 9999 : 0);

         top.computeBoundingBox();
         var overall_size = 3 * Math.max( Math.max(Math.abs(top.boundingBox.max.x), Math.abs(top.boundingBox.max.y)),
                                          Math.abs(top.boundingBox.max.z));
         //var boundingBox = this.computeBoundingBox(toplevel, false);
         //if (boundingBox!=null)
         //   overall_size = 10 * Math.max( Math.max(Math.abs(boundingBox.max.x), Math.abs(boundingBox.max.y)),
         //                                Math.abs(boundingBox.max.z));
      }
      else if (this._geometry['_typename'] == 'TEveGeoShapeExtract') {
         if (typeof this._geometry['fElements'] != 'undefined' && this._geometry['fElements'] != null) {
            var nodes = this._geometry['fElements']['arr'];
            for (var i = 0; i < nodes.length; ++i) {
               var node = this._geometry['fElements']['arr'][i];
               this.drawEveNode(this._scene, toplevel, node)
            }
         }
         var boundingBox = this.computeBoundingBox(toplevel, true);
         overall_size = 10 * Math.max( Math.max(Math.abs(boundingBox.max.x), Math.abs(boundingBox.max.y)),
                                       Math.abs(boundingBox.max.z));
      }
      if ( this._debug || this._grid ) {
         if ( this._full ) {
            var boxHelper = new THREE.BoxHelper( cube );
            this._scene.add( boxHelper );
         }
         this._scene.add( new THREE.AxisHelper( 2 * overall_size ) );
         this._scene.add( new THREE.GridHelper( Math.ceil( overall_size), Math.ceil( overall_size ) / 50 ) );
         this._renderer.domElement._translationSnap = Math.ceil( overall_size ) / 50;
         if ( this._renderer.domElement.transformControl !== null )
            this._renderer.domElement.transformControl.attach( toplevel );
         this.helpText("<font face='verdana' size='1' color='red'><center>Transform Controls<br>" +
                       "'T' translate | 'R' rotate | 'S' scale<br>" +
                       "'+' increase size | '-' decrease size<br>" +
                       "'W' toggle wireframe/solid display<br>"+
                       "keep 'Ctrl' down to snap to grid</center></font>");
      }
      this._camera.near = overall_size / 200;
      this._camera.far = overall_size * 500;
      this._camera.updateProjectionMatrix();
      this._camera.position.x = overall_size * Math.cos( 135.0 );
      this._camera.position.y = overall_size * Math.cos( 45.0 );
      this._camera.position.z = overall_size * Math.sin( 45.0 );
      this._renderer.render(this._scene, this._camera);

      // pointer used in the event handlers
      var pthis = this;

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
         this.focus();
      };

      return this.DrawingReady();
   }

   JSROOT.TGeoPainter.prototype.Cleanup = function() {
      this.helpText();
      if (this._scene === null ) return;

      this._renderer.domElement.clock = null;
      if (this._renderer.domElement._timeoutFunc != null)
         clearTimeout( this._renderer.domElement._timeoutFunc );
      if (this._renderer.domElement._animationId != null)
         cancelAnimationFrame( this._renderer.domElement._animationId );

      this.deleteChildren(this._scene);
      //this._renderer.initWebGLObjects(this._scene);
      delete this._scene;
      this._scene = null;
      if ( this._renderer.domElement.transformControl !== null )
         this._renderer.domElement.transformControl.dispose();
      this._renderer.domElement.transformControl = null;
      this._renderer.domElement.trackballControls = null;
      this._renderer.domElement.render = null;
      this._renderer = null;
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

   JSROOT.TGeoPainter.prototype.CheckResize = function(size) {

      var rect = this.select_main().node().getBoundingClientRect();

      if ((size!=null) && ('width' in size) && ('height' in size)) rect = size;

      if ((rect.width<10) || (rect.height<10)) return;

      this._camera.aspect = rect.width / rect.height;
      this._camera.updateProjectionMatrix();

      this._renderer.setSize( rect.width, rect.height );

   }

   ownedByTransformControls = function(child) {
      var obj = child.parent;
      while (obj && !(obj instanceof THREE.TransformControls) ) {
         obj = obj.parent;
      }
      return (obj && (obj instanceof THREE.TransformControls));
   }

   JSROOT.TGeoPainter.prototype.toggleWireFrame = function(obj) {
      var f = function(obj2) {
         if ( obj2.hasOwnProperty("material") && !(obj2 instanceof THREE.GridHelper) ) {
            if (!ownedByTransformControls(obj2))
               obj2.material.wireframe = !obj2.material.wireframe;
         }
      }
      obj.traverse(f);
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

      this.SetDivId(divid);

      return this.drawGeometry(opt);
   }

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
         _more : (typeof volume['fNodes'] != 'undefined') && (volume['fNodes']!=null),
         _menu : JSROOT.provideGeoMenu,
         _icon_click : JSROOT.geoIconClick,
         // this is special case of expand of geo volume
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
      if ((hitem==null) || (obj==null)) {
         return false;
      }

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

   JSROOT.addDrawFunc({ name: "TGeoVolumeAssembly", icon: 'img_geoassembly', func: JSROOT.Painter.drawGeometry, expand: "JSROOT.expandGeoVolume", painter_kind : "base" });


   return JSROOT.Painter;

}));

