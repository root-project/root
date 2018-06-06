
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
        
        onInit : function() {
            var id = this.getView().getId();
            console.log("eve.GL.onInit id = ", id );

            var data = this.getView().getViewData();
            console.log("VIEW DATA", data);
            
            this.mgr = data.mgr;
            this.elementid = data.elementid;
            
            // this is configured view
            var element = this.mgr.GetElement(this.elementid);
            
            // loop over scene and add dependency
            for (var k=0;k<element.childs.length;++k) {
               var scene = element.childs[k];
               console.log("FOUND scene", scene.fSceneId);
               
               this.mgr.Register(scene.fSceneId, this, "onElementChanged")
            }
            
            ResizeHandler.register(this.getView(), this.onResize.bind(this));
            this.fast_event = [];
            
            this.creator = new JSROOT.EVE.EveElements();
            
            // this.checkScences();
        },

        // function called from GuiPanelController
        onExit : function() {
           if (this.mgr) this.mgr.Unregister(this);
        },
        
        onElementChanged: function(id, element) {
           console.log("!!!CHANGED", id);
           
           this.checkScences();
        },
        
        onAfterRendering: function() {
           
           console.log("Did rendering");
           
           this.checkScences();
        },
        
        checkScences: function() {
           // start drawing only when all scenece has childs 
           // this is configured view
           var element = this.mgr.GetElement(this.elementid), allok = true;
           
           // loop over scene and add dependency
           for (var k=0;k<element.childs.length;++k) {
              var scene = element.childs[k];
              if (!scene) { allok = false; break; }
              var realscene = this.mgr.GetElement(scene.fSceneId);
              
              console.log("check scene", scene.fSceneId);
              if (!realscene || !realscene.childs) { allok = false; break; }
              console.log("scene ok", scene.fSceneId);
              
           }
           
           if (allok) this.drawGeometry();
        },
        
        
        drawGeometry: function() {
           
           console.log("start geometry drawing", this.getView().getId()); 
           
           var shape = {
              _typename: "TGeoBBox",
              fUniqueID: 0,
              fBits: 0x3000000,
              fName: "BOX",
              fTitle: "",
              fShapeId: 256,
              fShapeBits: 1024,
              fDX: 200,
              fDY: 300,
              fDZ: 400,
              fOrigin: [0,0,0]
           };
           
           var obj = JSROOT.extend(JSROOT.Create("TEveGeoShapeExtract"),
                 { fTrans: null, fShape: shape, fRGBA: [0, 1, 0, 0.2], fElements: null, fRnrSelf: true });
           
           
           this.geo_painter = JSROOT.Painter.CreateGeoPainter(this.getView().getDomRef(), obj, "");
           
           this.geo_painter.WhenReady(this.onGeomertyDrawn.bind(this));
           
           this.geo_painter.prepareObjectDraw(obj); // and start it
           
           // JSROOT.draw(this.getView().getDomRef(), obj, "", this.onGeomertyDrawn.bind(this));
           
        },
        
        onGeomertyDrawn: function(painter) {
           // this.geo_painter = painter;
           
           // top scene element
           var element = this.mgr.GetElement(this.elementid);
           
           // loop over scene and add dependency
           for (var k=0;k<element.childs.length;++k) {
              var scene = element.childs[k];
              if (!scene) continue;
              var realscene = this.mgr.GetElement(scene.fSceneId);
              
              console.log("check scene", scene.fSceneId);
              if (realscene && realscene.childs && (k>0)) 
                 this.drawExtras(realscene.childs, true); 
           }
        },
        
        drawExtras: function(arr, toplevel) {
           if (!arr) return;
           for (var k=0;k<arr.length;++k) {
              var elem = arr[k];
              if (elem.render_data) {
                 var fname = elem.render_data.rnr_func;
                 var obj3d = this.creator[fname](elem, elem.render_data);
                 if (obj3d) this.geo_painter.getExtrasContainer().add(obj3d);
                 
              }
              
              this.drawExtras(elem.childs);
           }
           
           if (toplevel) this.geo_painter.Render3D();
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
        
      onResize: function(event) {
            // use timeout
            // console.log("resize painter")
            if (this.resize_tmout) clearTimeout(this.resize_tmout);
            this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 300); // minimal latency
      },

       onResizeTimeout: function() {
          delete this.resize_tmout;
          if (this.geo_painter)
             this.geo_painter.CheckResize();
       
   }

    });

});
