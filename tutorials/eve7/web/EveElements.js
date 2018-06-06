/** @file EveElements.js */

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( [ 'JSRootCore', 'threejs' ], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
      factory(require("./JSRootCore.js"), require("./three.min.js"));
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'EveElements.js');

      if (typeof JSROOT.EVE == 'undefined')
         throw new Error('JSROOT.EVE is not defined', 'EveElements.js');

      if (typeof THREE == 'undefined')
         throw new Error('THREE is not defined', 'EveElements.js');

      factory(JSROOT, THREE);
   }
} (function( JSROOT, THREE ) {

   "use strict";

   function EveElements() {
      
   }
   
   EveElements.prototype.makeHit = function(hit, rnrData) {
      console.log("drawHit ", hit, "this type ", this.viewType);
      // console.log("marker size ", hit.fMarkerSize)
      var hit_size = 8*rnrData.fMarkerSize,
          size = rnrData.vtxBuff.length/3,
          pnts = new JSROOT.Painter.PointsCreator(size, true, hit_size);
      
      for (var i=0;i<size;i++) {
          pnts.AddPoint(rnrData.vtxBuff[i*3],rnrData.vtxBuff[i*3+1],rnrData.vtxBuff[i*3+2]);
         // console.log("add vertex ", rnrData.vtxBuff[i*3],rnrData.vtxBuff[i*3+1],rnrData.vtxBuff[i*3+2]);
      }
      var mesh = pnts.CreatePoints(JSROOT.Painter.root_colors[rnrData.fMarkerColor] );

      mesh.highlightMarkerSize = hit_size*3;
      mesh.normalMarkerSize = hit_size;

      mesh.geo_name = hit.fName;
      mesh.geo_object = hit;

      mesh.visible = hit.fRnrSelf;
      mesh.material.sizeAttenuation = false;
      return mesh;
  }
   
   EveElements.prototype.makeTrack = function(track, rnrData) {
      var N = rnrData.vtxBuff.length/3;
      var track_width = track.fLineWidth || 1,
          track_color = JSROOT.Painter.root_colors[track.fLineColor] || "rgb(255,0,255)";

      var buf = new Float32Array(N*3*2), pos = 0;
      for (var k=0;k<(N-1);++k) {
          buf[pos]   = rnrData.vtxBuff[k*3];
          buf[pos+1] = rnrData.vtxBuff[k*3+1];
          buf[pos+2] = rnrData.vtxBuff[k*3+2];

          var breakTrack = 0;
          if (rnrData.idxBuff) {
              for (var b = 0; b < rnrData.idxBuff.length; b++)
              {
                  if ( (k+1) == rnrData.idxBuff[b]) {
                      breakTrack = 1;
                  }
              }
          }
          
          if (breakTrack) {
              buf[pos+3] = rnrData.vtxBuff[k*3];
              buf[pos+4] = rnrData.vtxBuff[k*3+1];
              buf[pos+5] = rnrData.vtxBuff[k*3+2];
          }
          else {
              buf[pos+3] = rnrData.vtxBuff[k*3+3];
              buf[pos+4] = rnrData.vtxBuff[k*3+4];
              buf[pos+5] = rnrData.vtxBuff[k*3+5];
          }

          // console.log(" vertex ", buf[pos],buf[pos+1], buf[pos+2],buf[pos+3], buf[pos+4],  buf[pos+5]);
          pos+=6;
      }
      var lineMaterial = new THREE.LineBasicMaterial({ color: track_color, linewidth: track_width });
      var geom = new THREE.BufferGeometry();
      geom.addAttribute( 'position', new THREE.BufferAttribute( buf, 3 )  );
      var line = new THREE.LineSegments(geom, lineMaterial);

      line.geo_name = track.fName;
      line.geo_object = track;
      line.visible = track.fRnrSelf;
      console.log("make track ", track, line.visible);
      return line;
  }
   
   EveElements.prototype.makeJet = function(jet, rnrData) {
      console.log("make jet ", jet);
      var jet_ro = new THREE.Object3D();
      //var geo = new EveJetConeGeometry(jet.geoBuff);
      var pos_ba = new THREE.BufferAttribute( rnrData.vtxBuff, 3 );
      var N      = rnrData.vtxBuff.length / 3;

      var geo_body = new THREE.BufferGeometry();
      geo_body.addAttribute('position', pos_ba);
      {
          var idcs = [];
          idcs.push( 0 );  idcs.push( N - 1 );  idcs.push( 1 );
          for (var i = 1; i < N - 1; ++i)
          {
              idcs.push( 0 );  idcs.push( i );  idcs.push( i + 1 );
          }
          geo_body.setIndex( idcs );
      }
      var geo_rim = new THREE.BufferGeometry();
      geo_rim.addAttribute('position', pos_ba);
      {
          var idcs = [];
          for (var i = 1; i < N; ++i)
          {
              idcs.push( i );
          }
          geo_rim.setIndex( idcs );
      }
      var geo_rays = new THREE.BufferGeometry();
      geo_rays.addAttribute('position', pos_ba);
      {
          var idcs = [];
          for (var i = 1; i < N; i += 4)
          {
              idcs.push( 0 ); idcs.push( i );
          }
          geo_rays.setIndex( idcs );
      }

      jet_ro.add( new THREE.Mesh        (geo_body, new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.5 })) );
      jet_ro.add( new THREE.LineLoop    (geo_rim,  new THREE.LineBasicMaterial({ linewidth: 2,   color: 0x00ffff, transparent: true, opacity: 0.5 })) );
      jet_ro.add( new THREE.LineSegments(geo_rays, new THREE.LineBasicMaterial({ linewidth: 0.5, color: 0x00ffff, transparent: true, opacity: 0.5 })) );
      jet_ro.geo_name = jet.fName;
      jet_ro.geo_object = jet;
      jet_ro.visible = jet.fRnrSelf;

      return jet_ro;
  }
   
   JSROOT.EVE.EveElements = EveElements;
   
   console.log("LOADING EVE ELEMENTS");
   
   return JSROOT;

}));

