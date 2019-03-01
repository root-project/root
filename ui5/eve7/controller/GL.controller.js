sap.ui.define([
   'sap/ui/core/Component',
   'sap/ui/core/UIComponent',
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   "sap/ui/core/ResizeHandler",
   'rootui5/eve7/lib/EveManager'
], function (Component, UIComponent, Controller, JSONModel, ResizeHandler, EveManager) {

   "use strict";

   // for debug purposes - do not create geometry painter, just three.js renderer 
   var direct_threejs = false;
   
   var EveScene = null;

   return Controller.extend("rootui5.eve7.controller.GL", {

      onInit : function()
      {
         var id = this.getView().getId();
         console.log("eve.GL.onInit id = ", id);

         var viewData = this.getView().getViewData();
         if (viewData) {
            console.log("Create with view data");
            this.createXXX(viewData);
         }
         
         var oRouter = UIComponent.getRouterFor(this);
         oRouter.getRoute("View").attachPatternMatched(this._onObjectMatched, this);

         ResizeHandler.register(this.getView(), this.onResize.bind(this));
         this.fast_event = [];

         this._load_scripts = false;
         this._render_html = false;
         this.geo_painter = null;
         // this.painter_ready = false;

         JSROOT.AssertPrerequisites("geom", this.onLoadScripts.bind(this));
      },
      
      onLoadScripts: function()
      {
         var pthis = this;
         
         sap.ui.define(['rootui5/eve7/lib/EveScene'], function (_handle) {
            EveScene = _handle;
            pthis._load_scripts = true;
            pthis.checkViewReady();
         });
      },
      
      _onObjectMatched: function(oEvent) {
         var args = oEvent.getParameter("arguments");
         this.createXXX(Component.getOwnerComponentFor(this.getView()).getComponentData(), args.viewName);
      },

      createXXX: function(data, viewName) {

         if (viewName) {
            data.standalone = viewName;
            data.kind = viewName;
         }
         //var data = this.getView().getViewData();
         // console.log("VIEW DATA", data);

         if (data.standalone && data.conn_handle)
         {
            this.mgr = new EveManager();
            this.mgr.UseConnection(data.conn_handle);
            this.standalone = data.standalone;
            this.mgr.RegisterUpdate(this, "onManagerUpdate");
         }
         else
         {
            this.mgr = data.mgr;
            this.elementid = data.elementid;
            this.kind = data.kind;
         }

      },

      // MT-HAKA
      createThreejsRenderer: function()
      {
         if (!direct_threejs || this.renderer) return;

         this.scene      = new THREE.Scene();
         this.camera     = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 100000 );
         this.rot_center = new THREE.Vector3(0,0,0);

         // this.controls = new THREE.OrbitControls( this.camera );
         //var controls = new THREE.FirstPersonControls( camera );

         this.renderer = new THREE.WebGLRenderer();
         this.renderer.setPixelRatio( window.devicePixelRatio );
         this.renderer.setSize( window.innerWidth, window.innerHeight );

         // this.scene.fog = new THREE.FogExp2( 0xaaaaaa, 0.05 );
         this.renderer.setClearColor( 0xffffff, 1 );

         this.dom_registered = false;
         //document.body.appendChild( this.renderer.domElement );

         //document.getElementById('EveViewer9').appendChild( this.renderer.domElement );

         // this.getView().getDomRef().appendChild( this.renderer.domElement );

         // -------

         // var sphere = new THREE.SphereGeometry( 0.1, 8, 8 );
         // var lamp = new THREE.DirectionalLight( 0xff5050, 0.5 );
         var lampR = new THREE.PointLight( 0xff5050, 0.7 );
         // lampR.add(new THREE.Mesh( sphere, new THREE.MeshBasicMaterial( { color: lampR.color } ) ));
         lampR.position.set(2, 2, -2);
         this.scene.add( lampR );

         var lampG = new THREE.PointLight( 0x50ff50, 0.7 );
         // lampG.add(new THREE.Mesh( sphere, new THREE.MeshBasicMaterial( { color: lampG.color } ) ));
         lampG.position.set(-2, 2, 2);
         this.scene.add( lampG );

         var lampB = new THREE.PointLight( 0x5050ff, 0.7 );
         // lampB.add(new THREE.Mesh( sphere, new THREE.MeshBasicMaterial( { color: lampB.color } ) ));
         lampB.position.set(2, 2, 2);
         this.scene.add( lampB );

         //var plane = new THREE.GridHelper(20, 20, 0x80d080, 0x8080d0);
         //this.scene.add(plane);
      },

      /** returns container for 3d objects */
      getThreejsContainer: function(name)
      {
         var prnt = null;

         if (!direct_threejs)
            prnt = this.geo_painter.getExtrasContainer();
         else
            prnt = this.scene;

         for (var k=0;k<prnt.children.length;++k)
            if (prnt.children[k]._eve_name === name)
               return prnt.children[k];

         var obj3d = new THREE.Object3D();
         obj3d._eve_name = name;
         prnt.add(obj3d);
         return obj3d;
      },

      // MT-HAKA
      render: function()
      {
         if (!direct_threejs) {
            if (this.geo_painter) {
               if (!this.first_time_render) {
                  this.first_time_render = true;
                  this.geo_painter.adjustCameraPosition(true);
               }
               this.geo_painter.Render3D();
            }
            return;
         }

         if ( ! this.dom_registered)
         {
            this.getView().getDomRef().appendChild( this.renderer.domElement );

            //this.controls = new THREE.OrbitControls( this.camera);
            this.controls = new THREE.OrbitControls( this.camera, this.getView().getDomRef() );

            this.controls.addEventListener( 'change', this.render.bind(this) );

            this.dom_registered = true;

            // Setup camera
            var sbbox = new THREE.Box3();
            sbbox.setFromObject( this.scene );

            //var center = boundingBox.getCenter();
            this.controls.target = this.rot_center;

            var maxV = new THREE.Vector3; maxV.subVectors(sbbox.max, this.rot_center);
            var minV = new THREE.Vector3; minV.subVectors(sbbox.min, this.rot_center);

            var posV = new THREE.Vector3; posV = maxV.multiplyScalar(2);

            this.camera.position.set( posV.x, posV.y, posV.z );
            this.camera.lookAt(this.rot_center);

            console.log("scene bbox ", sbbox, ", camera_pos ", posV, ", look_at ", this.rot_center);
         }

         // console.log(this.camera);

         //console.log(this.controls);
         //console.log(this.getView().getDomRef());
         //console.log(this.renderer.domElement);

         // requestAnimationFrame( this.render.bind(this) );

         // this.controls.update( );

         this.renderer.render( this.scene, this.camera );
      },

      onManagerUpdate: function()
      {
         // called when manager was updated, need only in standalone modes to detect own element id
         if (!this.standalone || this.elementid) return;

         var viewers = this.mgr.FindViewers();

         // first check number of views to create
         var found = null;
         for (var n=0;n<viewers.length;++n) {
            if (viewers[n].fName.indexOf(this.standalone) == 0) { found = viewers[n]; break; }
         }
         if (!found) return;

         this.elementid = found.fElementId;
         this.kind = (found.fName == "Default Viewer") ? "3D" : "2D";
         this.checkViewReady();

      },

      // function called from GuiPanelController
      onExit: function()
      {
         if (this.mgr) this.mgr.Unregister(this);
      },

      onAfterRendering: function()
      {

         this._render_html = true;

         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%").parent().css("overflow", "hidden");

         this.checkViewReady();
      },

      createScenes: function()
      {
         if (this.created_scenes !== undefined) return;
         this.created_scenes = [];

         // only when rendering completed - register for modify events
         var element = this.mgr.GetElement(this.elementid);

         // loop over scene and add dependency
         for (var k=0;k<element.childs.length;++k)
         {
            var scene = element.childs[k];

            var handler = new EveScene(this.mgr, scene, this);

            this.created_scenes.push(handler);
            this.mgr.addSceneHandler(handler);
         }
      },

      redrawScenes: function() {
         for (var k=0;k<this.created_scenes.length;++k)
            this.created_scenes[k].redrawScene();
      },


      /** checks if all initialization is performed */
      checkViewReady: function()
      {
         if (!this._load_scripts || !this._render_html || !this.elementid) return;

         if (direct_threejs) {
            this.createThreejsRenderer();
            this.createScenes();
            this.redrawScenes();
            return;
         }

         if (this.geo_painter) {
            this.redrawScenes();
            return;
         }


         var options = "";
         if (this.kind != "3D") options = "ortho_camera";


         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%");

         this.geo_painter = JSROOT.Painter.CreateGeoPainter(this.getView().getDomRef(), null, options);

         this.painter_ready = false;
         // assign callback function - when needed
         this.geo_painter.WhenReady(this.onGeoPainterReady.bind(this));

         this.geo_painter.AssignObject(null);

         this.geo_painter.prepareObjectDraw(null); // and now start everything
      },

      onGeoPainterReady: function(painter) {

         // AMT temporary here, should be set in camera instantiation time
         if (this.geo_painter._camera.type == "OrthographicCamera") {
            this.geo_painter._camera.left = -this.getView().$().width();
            this.geo_painter._camera.right = this.getView().$().width();
            this.geo_painter._camera.top = this.getView().$().height();
            this.geo_painter._camera.bottom = -this.getView().$().height();
            this.geo_painter._camera.updateProjectionMatrix();
         }

         this.painter_ready = true;
         // this.geo_painter._highlight_handlers = [ this ]; // register ourself for highlight handling
         this.last_highlight = null;

         // create only when geo painter is ready
         this.createScenes();
         this.redrawScenes();
      },

      /// invoked from the manager
      onResize: function(event) {
         // use timeout
         // console.log("resize painter")
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 300); // minimal latency
      },

      onResizeTimeout: function() {
         delete this.resize_tmout;

         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%");
         if (this.geo_painter)
            this.geo_painter.CheckResize();
      }

   });

});
