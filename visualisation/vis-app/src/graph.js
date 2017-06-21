import {HttpClient} from 'aurelia-fetch-client';
import {bindable} from 'aurelia-framework';

export class Graph {

  @bindable treatment:string;

  constructor() {
    this.client = new HttpClient();
  }

  attached() {
    return this.client.fetch('http://localhost:8765/graphs/' + this.treatment)
    .then(response => response.json())
    .then(data => {
      this.graph = data.div;
      return data.script;
    })
    .then(script_data => {
      // In theory, I don't see why this timeout is needed, but stuff doesn't
      // work without it.
      setTimeout(function(){
        let script = document.createElement("script");
        script.innerHTML = script_data;
        document.head.appendChild(script);
      }, 10)
    })
  }
}
