import {HttpClient} from 'aurelia-fetch-client';

export class Overview {

  configureRouter(config, router){
    config.map([
      {route: '', redirect: 'popularity'},
      {route: ':graph', moduleId: 'graph', name: 'graph_page'},
    ]);
    this.router = router;
  }

  constructor() {
    this.client = new HttpClient();
    this.treatment;
    this.name;
    this.active_graph = "popularity";
  }

  activate(params) {
    this.name = params.name;
    return this.client.fetch('http://localhost:8765/treatment/' + params.name)
    .then(response => response.json())
    .then(data => {
      this.treatment = data;
    });
  }

  graph(name) {
    this.active_graph = name;
    this.router.navigateToRoute("graph_page", {graph: name});
  }
}
