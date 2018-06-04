sap.ui.define(['sap/ui/core/mvc/Controller'],
	      function(Controller) {
	          "use strict";

	          var SplitterController = Controller.extend("eve.Main", {
		      onInit: function () {
		          {
                              var sv =  this.getView().byId("ViewAreaSplitter");
                              // console.log("view Name ", sv);
		              // console.log("SPLIT CONTROLLER == ", sv.getContentAreas());
                              var ca = sv.getContentAreas();
                              // console.log("primary ",ca[0].data("type"), ca[0] );
                              viewManager.addView(ca[0].getId(), ca[0].data("type"));
                          }
                          {
                              var sv =  this.getView().byId("SecondaryView");
			      if (sv) {
                                  var ca = sv.getContentAreas();
                                  for (var i = 0; i < ca.length; ++i) {
                                      console.log("seconary  ",  i ,  ca[i].data("type"), ca[i].getId());
                                      viewManager.addView(ca[i].getId(), ca[i].data("type"));

                                  }
			      }
                          }

                          //DOCUMENT_READY = true;
                          //this.processWaitingMsg();

			  this.handle = JSROOT.eve_handle;
			  delete JSROOT.eve_handle;
                          this.handle.SetReceiver(this);
                          this.handle.Connect();

                          this.mgr = new JSROOT.EveManager();

                          //this.getView().byId("Summary").SetMgr(this.mgr);
			  
		      },
		      
                      getHandle: function () {
                          return this.handle;
                      },
		      
	              OnWebsocketMsg: function(handle, msg)
                      {
                         // this.handle = handle;
                          
                          if (typeof msg != "string")
                          {
                             return;
                              // console.log('TestPanel ArrayBuffer size ' +  msg.byteLength);
                              var textSize = 11;
                              {
                                  var sizeArr = new Int32Array(msg, 0, 4);
                                  textSize = sizeArr[0];                            
                                  // console.log("textsize 4", textSize);
                              }
                              
                              var arr = new Int8Array(msg, 4, textSize);
                              var str = String.fromCharCode.apply(String, arr);
                              // console.log("core header = ", str);

                              var off = 4+ textSize;
                              var renderData = JSON.parse(str);

                              off = 4 * Math.ceil(off/4.0);

                              var vtArr = [];
                              var el = this.findElementWithId(renderData.guid, this._event);

                              for (var i = 0; i < renderData["hsArr"].length; ++i)
                              {
                                  console.log(">>>>>>>> LOOP view type ", i, off);
                                  var vha = new Int8Array(msg, off,renderData["hsArr"][i]);
                                  str = String.fromCharCode.apply(String, vha);
                                  console.log("HEADER ", str);
                                  var vo = JSON.parse(str);
                                  
                                  var headOff =  4*Math.ceil(renderData["hsArr"][i]/4.0);
                                  off += headOff;
                                  var totalSizeVT = renderData["bsArr"][i];
                                  var arrSize = totalSizeVT - headOff;

                                  console.log("array size off", arrSize, off);
                                  if (vo.vertexN) {
                                      console.log("vertex array size off", vo.vertexN);
                                      var fArr = new Float32Array(msg, off, vo.vertexN);
                                      off+=vo.vertexN*4;
                                      // console.log("vertex arr off ", fArr, off);                            
                                      vo["vtxBuff"] = fArr;
                                  }

                                  if (vo.normalN) {
                                      console.log("vertex array size off", vo.normalN);
                                      var fArr = new Float32Array(msg, off, vo.normalN);
                                      off+=vo.nornalN*4;
                                      // console.log("normal arr off ", fArr, off);                            
                                      vo["normalBuff"] = fArr;
                                  }

                                  if (vo.indexN) {
                                      console.log("index array size", vo.indexN, "off", off);
                                      var iArr = new Int32Array(msg, off, vo.indexN);
                                      off+=vo.indexN*4;
                                      console.log("index arr == ", iArr);                            
                                      vo["idxBuff"] = iArr;
                                  }

                                  
                                  el[vo.viewType] = vo;
                                  // console.log("add render info ", el);
                              }

                              viewManager.addElementRnrInfo(el);
                              // console.log("element with rendering info ", el);

                              return;
                          }

                          console.log("txt:", msg);
                          
                          // console.log("OnWebsocketMsg response ", msg);
                          var resp = JSON.parse(msg);

                          if (resp && resp[0] && resp[0].content == "TEveScene::StreamElements") {
                             
                             this.mgr.Update(resp);

                             // console.log('element', this.getView().byId("Summary").getController());

                             this.getView().byId("Summary").getController().UpdateMgr(this.mgr);

                             // console.log('Mgr', this.mgr.childs);
                          
                             
                          } else if (resp.function === "geometry")
                          {
                              console.log("GEO");
                              viewManager.setGeometry( resp);
                          }
                      
                          else if (resp.function === "event")
                          {
                              console.log("EVE ", resp);
                              this._event = resp.args[0];
                              this.event();
                          }
                          else if (resp.function === "replaceElement")
                          {
                              var oldEl = this.findElementWithId(resp.guid, this._event);
                              var newEl = resp;
                              viewManager.replace(oldEl, newEl);

                              this.event(); 
                          }
                          else if (resp.function === "endChanges") {
                              this.endChanges = resp.val;
                              if (resp.val)
                              {
			          /*
                                    var ele =  this.getView().byId("GL");
                                    var cont = ele.getController();
                                    cont.endChanges(resp.val);
			          */
			          viewManager.envokeViewFunc("endChanges", resp.val);
                              }
                          }
                      },
                           /*     
                      processWaitingMsg: function() {
                          for ( var i = 0; i < msgToWait.length; ++i ) {
                              this.OnWebsocketMsg(handleToWait, msgToWait[i]);
                          }
                          handleToWait = 0;
                          msgToWait = [];
                      },
                           */
                      event: function() {
                          //  this._event = lst;
		          /*
                            {
			    
                            var ele =  this.getView().byId("GL");
                            console.log("ele GL >>>> ", ele);
                            if (!ele) return;
                            var cont = ele.getController();
                            cont["event"]( this._event);
                            }
		          */
		          viewManager.envokeViewFunc("event", this._event);
                          {
                              var ele =  this.getView().byId("Summary");
                              // console.log("ele Sum", ele);
                              if (!ele) return;
                              var cont = ele.getController();
                              
                              // console.log("ele Sum cont", cont);
                              cont.event( this._event);
                          }
                      },
		      setMainVerticalSplitterHeight: function(){
                          var mainViewHeight = document.body.clientHeight;
		          var mainToolbarHeight = 49;
		          var height = mainViewHeight - mainToolbarHeight;    
		          var splitter =  this.getView().byId("MainAreaSplitter");
		          if (splitter) {
		              //console.log("set splitter height >>>  " , height);		
                              splitter.setHeight(height + "px");
                          }
		      },
		      onAfterRendering: function(){
		          var me = this;
		          setTimeout(
			      function(){
			          $(window).on("resize", function(){
				      me.setMainVerticalSplitterHeight();
			          });
			          me.setMainVerticalSplitterHeight();
			      },
			      100
		          );
		          
		      },
                      findElementWithId: function(valueToSearch, el) {
                          if (!el) {
                              el = this._event;
                          }
                          // console.log("serach ",valueToSearch, "in", el )
                          if (el.guid == valueToSearch) {
                              // console.log("found it findElementWithId ", el)
                              return el;
                          }
                          if ( el.arr) {
                              for (var i = 0; i < el.arr.length; i++) {
                                  var x = this.findElementWithId(valueToSearch, el.arr[i]);
                                  if (x) return x; 
                              }
                          }
                          return 0;
                      }
	          });

	          return SplitterController;

              }
             );
