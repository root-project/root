
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
            console.log("VIEW DATA", cstm);
            
            ResizeHandler.register(this.getView(), this.onResize.bind(this));
            this.fast_event = [];
            
            this.creator = new JSROOT.EVE.EveElements();
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
                        var mesh = this.creator[rnrData.rnrFunc](x, rnrData);
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
