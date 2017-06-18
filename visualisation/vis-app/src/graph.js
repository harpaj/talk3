import {HttpClient} from 'aurelia-fetch-client';

export class Graph {
  constructor() {
    this.client = new HttpClient();
    this.script_data;
  }

  activate() {
    return this.client.fetch('http://localhost:8765/')
    .then(response => response.json())
    .then(data => {
      this.graph = data.div;
      this.script_data = data.script;
    });
  }
  attached() {
    let script = document.createElement("script");
    script.innerHTML = this.script_data;
    document.head.appendChild(script);
  }
}
