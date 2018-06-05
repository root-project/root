
/*
function EveJetConeGeometry(vertices)
{
    THREE.BufferGeometry.call( this );

    this.addAttribute( 'position', new THREE.BufferAttribue( vertices, 3 ) );

    var N = vertices.length / 3;
    var idcs = [];
    for (var i = 1; i < N - 1; ++i)
    {
        idcs.push( i ); idcs.push( 0 ); idcs.push( i + 1 );
    }
    this.setIndex( idcs );
}

EveJetConeGeometry.prototype = Object.create( THREE.BufferGeometry.prototype );
EveJetConeGeometry.prototype.constructor = EveJetConeGeometry;
*/


sap.ui.define([
    'sap/ui/core/mvc/Controller',
    'sap/ui/model/json/JSONModel',
    "sap/ui/core/ResizeHandler"
], function (Controller, JSONModel, ResizeHandler) {
    "use strict";

    return Controller.extend("eve.GL", {
        // function called from GuiPanelController
        onInit : function() {
            var id = this.getView().getId();
            console.log("eve.GL.onInit id = ", id );

            var cstm = this.getView().getViewData();
            console.log("CUSTOM", cstm);
            
            ResizeHandler.register(this.getView(), this.onResize.bind(this));
            this.fast_event = [];
        },

        // function called from GuiPanelController
        onExit : function() {
        },
        
        geometry:function(data) {
            var pthis = this;
            var id = this.getView().getId() + "--panelGL";
            this.viewType = this.getView().data("type");


            
	    JSROOT.draw(id, data, "", function(painter) {
                console.log('GL painter initialized', painter);
                pthis.geo_painter = painter;

                if (pthis.viewType != "3D") {
                    var a = 651;
                    painter._camera =  new THREE.OrthographicCamera(a, -a, a, -a, a, -a);
                    painter._camera.position.x = 0;
                    painter._camera.position.y = 0;
                    painter._camera.position.z = +200;
                    painter._controls = JSROOT.Painter.CreateOrbitControl(painter, painter._camera, painter._scene, painter._renderer, painter._lookat);
                }

                
                if (pthis.fast_event) pthis.drawExtra();
                pthis.geo_painter.Render3D();

	    });
        },
        event: function(data) {
            /*
            if (this.drawExtra(data)) {
                this.geo_painter.Render3D();
            }*/
        },
        endChanges: function(val) {

            this.needRedraw = true;
        },
        drawExtra : function(el) {
            if (!this.geo_painter) {
                // no painter - no draw of event
                console.log("fast event geo not initialized append element",  this.getView().getId())
                this.fast_event.push(el);
                return false;
            }
            else {
               // this.geo_painter.clearExtras(); // remove old three.js container with tracks and hits
                var len = this.fast_event.length;
                for(var i = 0; i < len;  i++){
                    var x = this.fast_event[i];
                    console.log("draw extra ... catchup fast event ", x, this.getView().getId());
                    var rnrData = x[this.viewType];
                    if (rnrData) {
                        // console.log("calling rendere ",rnrData.rnrFunc, rnrData );
                        var mesh = this[rnrData.rnrFunc](x, rnrData);
                        this.geo_painter.getExtrasContainer().add(mesh);
                    }
                }
                this.fast_event = [];
                
                if (el) {
                    console.log("draw extra SINGLE");

                    var rnrData = el[this.viewType];
                    if (rnrData) {
                        // console.log("calling rendere ",rnrData.rnrFunc, rnrData );
                        var mesh = this[rnrData.rnrFunc](el, rnrData);
                        this.geo_painter.getExtrasContainer().add(mesh);
                    }
                }
                if (this.needRedraw) {
                    this.geo_painter.Render3D();
                    this.needRedraw = false;
                }
                // console.log("PAINTER ", this.geo_painter);

                return true;
            }
        },
        makeHit: function(hit, rnrData) {
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
        },
        makeTrack: function(track, rnrData) {
            if (this.viewType == "RhoZ") {
                console.log("RhoZ track ", rnrData.idxBuff);
            }
            var N = rnrData.vtxBuff.length/3;
            var track_width = track.fLineWidth || 1,
                track_color = JSROOT.Painter.root_colors[track.fLineColor] || "rgb(255,0,255)";

            var buf = new Float32Array(N*3*2), pos = 0;
            for (var k=0;k<(N-1);++k) {
                buf[pos]   = rnrData.vtxBuff[k*3];
                buf[pos+1] = rnrData.vtxBuff[k*3+1];
                buf[pos+2] = rnrData.vtxBuff[k*3+2];

                var breakTrack = 0;
                if (this.viewType == "RhoZ" && rnrData.idxBuff) {
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
        },
        makeJet: function(jet, rnrData) {
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
        },

        replaceElement:function(oldEl, newEl) {
            console.log("GL controller replace element  OLD", oldEl,  oldEl.fRnrSelf);
            console.log("GL controller replace element  NEW",  newEl, newEl.fRnrSelf);

            var ec = this.geo_painter.getExtrasContainer();
            var chld = ec.children;
            var idx = -1;
            for (var i = 0; i < chld.length; ++i) {
                if (chld[i].geo_object.guid == newEl.guid)
                {
                    idx = i;
                    break;
                }
            }


            var rnrData = oldEl[this.viewType];

            
            console.log("calling draw",  newEl, newEl.fRnrSelf);
           chld[idx] = this[rnrData.rnrFunc](newEl, rnrData);
            
            this.geo_painter.Render3D();
           console.log("------------- rnrstate ", this.geo_painter, newEl );
            //this.geo_painter._renderer.render(this.geo_painter._scene, this.geo_painter._camera);


        },
	onResize: function(event) {
            // use timeout
            // console.log("resize painter")
            if (this.resize_tmout) clearTimeout(this.resize_tmout);
            this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 300); // minimal latency
	},

	onResizeTimeout: function() {
            delete this.resize_tmout;
            if (this.geo_painter) {
		this.geo_painter.CheckResize();
	    }
	}

    });

});
