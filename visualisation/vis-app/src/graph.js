import {HttpClient} from 'aurelia-fetch-client';

export class Graph {

  constructor() {
    this.client = new HttpClient();
    this.graph = "<div>Loading graph...</div>";
  }

  determineActivationStrategy(){
    return "replace";
  }

  activate(params, settings) {
    console.log(params, settings);
    this.treatment = params.name;
    this.graph_type = params.graph;
  }

  attached() {
    return this.client.fetch(
        'http://localhost:8765/graphs/' + this.treatment + '/' + this.graph_type)
    .then(response => response.json())
    .then(data => {
      this.graph = data.div;
      return data.script;
    })
    .then(script_data => {
      // In theory, I don't see why this timeout is needed, but the binding
      // of this.graph to the innerHTML appears to not update immediately
      setTimeout(function(){
        let script = document.createElement("script");
        script.innerHTML = script_data;
        document.head.appendChild(script);
      }, 10)
    })
  }

  detached() {
    document.head.removeChild(document.head.lastChild);
  }

}
