
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
            
            ResizeHandler.register(this.getView(), this.onResize.bind(this));
            this.fast_event = [];
            
            this._load_scripts = false;
            this._render_html = false;
            
            JSROOT.AssertPrerequisites("geom;user:evedir/EveElements.js", this.onLoadScripts.bind(this));
            
            // this.checkScences();
        },
        
        onLoadScripts: function() {
           this._load_scripts = true;
           // only when scripts loaded, one could create objects
           this.creator = new JSROOT.EVE.EveElements();
           this.checkScences();
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
           
           this._render_html = true;
           
           // only when rendering completed - register for modify events
           var element = this.mgr.GetElement(this.elementid);
           
           // loop over scene and add dependency
           for (var k=0;k<element.childs.length;++k) {
              var scene = element.childs[k];
              console.log("FOUND scene", scene.fSceneId);
              
              this.mgr.Register(scene.fSceneId, this, "onElementChanged")
           }
           
           
           this.checkScences();
        },
        
        checkScences: function() {
           
           if (!this._load_scripts || !this._render_html) return;
           
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
              fUniqueID: 0, fBits: 0x3000000, fName: "BOX", fTitle: "",
              fShapeId: 256, fShapeBits: 1024, fDX: 200, fDY: 300, fDZ: 400, fOrigin: [0,0,0]
           };
           
           var geom_obj = JSROOT.extend(JSROOT.Create("TEveGeoShapeExtract"),
                 { fTrans: null, fShape: shape, fRGBA: [0, 1, 0, 0.2], fElements: null, fRnrSelf: true });
           
           this.geo_painter = JSROOT.Painter.CreateGeoPainter(this.getView().getDomRef(), null, "");
           
           // assign callback function - when needed 
           this.geo_painter.WhenReady(this.onGeomertyDrawn.bind(this));
           
           // now loop over all  scene and create three.js objects
           
           // top scene element
           var element = this.mgr.GetElement(this.elementid);
           
           // loop over scene and add dependency
           for (var k=0;k<element.childs.length;++k) {
              var scene = element.childs[k];
              if (!scene) continue;
              var realscene = this.mgr.GetElement(scene.fSceneId);
              
              console.log("EVE check scene", scene.fSceneId);
              if (realscene && realscene.childs) 
                 this.createExtras(realscene.childs);
              console.log("EVE check scene done", scene.fSceneId);
           }

           // if geometry detected in the scenes, it will be used to display 
           geom_obj = null;

           this.geo_painter.AssignObject(geom_obj);
           
           this.geo_painter.prepareObjectDraw(geom_obj); // and now start everything
           
           // JSROOT.draw(this.getView().getDomRef(), obj, "", this.onGeomertyDrawn.bind(this));
           
        },
        
        onGeomertyDrawn: function(painter) {
           console.log("Drawing completed");
        },
        
        createExtras: function(arr, toplevel) {
           if (!arr) return;
           for (var k=0;k<arr.length;++k) {
              var elem = arr[k];
              if (elem.render_data) {
                 var fname = elem.render_data.rnr_func, obj3d = null;
                 if (!this.creator[fname]) {
                    console.error("Function " +fname + " missing in creator");
                 } else {
                    console.log("creating ", fname);
                    obj3d = this.creator[fname](elem, elem.render_data);
                 }
                 if (obj3d) {
                    obj3d._typename = "THREE.Mesh";
                    this.geo_painter.addExtra(obj3d);
                 }
              }
              
              this.createExtras(elem.childs);
           }
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
