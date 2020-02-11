sap.ui.define([
   'sap/ui/core/Component',
   'sap/ui/core/UIComponent',
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   "sap/ui/core/ResizeHandler",
   'rootui5/eve7/lib/EveManager'
], function (Component, UIComponent, Controller, JSONModel, ResizeHandler, EveManager) {

   "use strict";

   // EveScene constructor function.
   var EveScene = null;
   var GlViewer = null;

   let viewer_class = "JSRoot"; // JSRoot Three RCore

   let maybe_proto = Controller.extend("rootui5.eve7.Controller.GL", {

      //==============================================================================
      // Initialization, bootstrap, destruction & cleanup.
      //==============================================================================

      onInit : function()
      {
         this.resize_handler = ResizeHandler;

         // var id = this.getView().getId();

         let viewData = this.getView().getViewData();
         if (viewData)
         {
            this.setupManagerAndViewType(viewData);
         }
         else
         {
            UIComponent.getRouterFor(this).getRoute("View").attachPatternMatched(this.onViewObjectMatched, this);
         }

         this._load_scripts = false;
         this._render_html  = false;

         if (viewer_class === "RCore")
         {
            var pthis = this;
            import("../../eve7/rnr_core/RenderCore.js").then((module) => {
               console.log("GLC onInit RenderCore loaded");
              // alert("Step 1: controller says: RnrCore loaded")
               pthis.RCore = module;

               let orc = new pthis.RCore.Object3D;
               console.log("RCore::Object3D", orc);

               JSROOT.AssertPrerequisites("geom", pthis.onLoadScripts.bind(pthis));
            });
         }
         else
         {
            JSROOT.AssertPrerequisites("geom", this.onLoadScripts.bind(this));
         }
      },

      onLoadScripts: function()
      {
         var pthis = this;

         // console.log("GLC::onLoadScripts, now loading EveScene and GlViewer" + viewer_class);

         // one only can load EveScene after geometry painter

         sap.ui.require(['rootui5/eve7/lib/EveScene',
                         'rootui5/eve7/lib/GlViewer' + viewer_class
                        ],
                        function (eve_scene, gl_viewer) {
                           EveScene = eve_scene;
                           GlViewer = gl_viewer;
                           pthis._load_scripts = true;
                           pthis.checkViewReady();
                        });
      },

      onViewObjectMatched: function(oEvent)
      {
         let args = oEvent.getParameter("arguments");

         this.setupManagerAndViewType(Component.getOwnerComponentFor(this.getView()).getComponentData(),
                                      args.viewName, JSROOT.$eve7tmp);

         delete JSROOT.$eve7tmp;
      },

      // Initialization that can be done immediately onInit or later through UI5 bootstrap callbacks.
      setupManagerAndViewType: function(data, viewName, moredata)
      {
         if (viewName)
         {
            data.standalone = viewName;
            data.kind       = viewName;
         }

         // console.log("VIEW DATA", data);

         if (moredata && moredata.mgr)
         {
            this.mgr        = moredata.mgr;
            this.eveViewerId  = moredata.eveViewerId;
            this.kind       = moredata.kind;
            this.standalone = viewName;

            this.checkViewReady();
         }
         else if (data.standalone && data.conn_handle)
         {
            this.mgr        = new EveManager();
            this.standalone = data.standalone;
            this.mgr.UseConnection(data.conn_handle);
         }
         else
         {
            this.mgr       = data.mgr;
            this.eveViewerId = data.eveViewerId;
            this.kind      = data.kind;
         }

         this.mgr.RegisterController(this);
         this.mgr.RegisterGlController(this);
      },

      // Called when HTML parent/container rendering is complete.
      onAfterRendering: function()
      {
         this._render_html = true;

         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%").parent().css("overflow", "hidden");

         this.checkViewReady();
      },

      OnEveManagerInit: function()
      {
         // called when manager was updated, need only in standalone modes to detect own element id
         if (!this.standalone) return;

         let viewers = this.mgr.FindViewers();

         // first check number of views to create
         let found = null;
         for (let n = 0; n < viewers.length; ++n)
         {
            if (viewers[n].fName.indexOf(this.standalone) == 0)
            {
               found = viewers[n];
               break;
            }
         }
         if ( ! found) return;

         this.eveViewerId = found.fElementId;
         this.kind      = (found.fName == "Default Viewer") ? "3D" : "2D";

         this.checkViewReady();
      },

      // Function called from GuiPanelController.
      onExit: function()
      {
         // QQQQ EveManager does not have Unregister ... nor UnregisterController
         if (this.mgr) this.mgr.Unregister(this);
         // QQQQ plus, we should unregister this as gl-controller, too
      },

      // Checks if all initialization is performed and startup renderer.
      checkViewReady: function()
      {
         if ( ! this._load_scripts || ! this._render_html || ! this.eveViewerId)
         {
            return;
         }

         this.JsRootRender = !this.mgr.handle.GetUserArgs("JsRootRender");
         this.htimeout = this.mgr.handle.GetUserArgs("HTimeout");
         if (this.htimeout === undefined) this.htimeout = 250;

         // console.log("GLC::checkViewReady, instantiating GLViewer" + viewer_class);

         this.viewer = new GlViewer(viewer_class);
         this.viewer.init(this);
      },


      //==============================================================================
      // Common functions between THREE and GeoPainter
      //==============================================================================

      /** returns container for 3d objects */
      getSceneContainer: function(scene_name)
      {
         let parent = this.viewer.get_top_scene();

         for (let k = 0; k < parent.children.length; ++k)
         {
            if (parent.children[k]._eve_name === scene_name)
            {
               return parent.children[k];
            }
         }

         let obj3d = this.viewer.make_object();
         obj3d._eve_name = scene_name;
         parent.add(obj3d);
         return obj3d;
      },

      createScenes: function()
      {
         if (this.created_scenes !== undefined) return;
         this.created_scenes = [];

         // only when rendering completed - register for modify events
         let element = this.mgr.GetElement(this.eveViewerId);

         // loop over scene and add dependency
         for (let scene of element.childs)
         {
            this.created_scenes.push(new EveScene(this.mgr, scene, this));
         }
      },

      redrawScenes: function()
      {
         for (let s of this.created_scenes)
         {
            s.redrawScene();
         }
      },

      /// invoked from ResizeHandler
      onResize: function(event)
      {
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 250); // minimal latency
      },

      onResizeTimeout: function()
      {
         delete this.resize_tmout;

         // console.log("onResizeTimeout", this.camera);

         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%");

         this.viewer.onResizeTimeout();
      },

   });

   return maybe_proto;
});
