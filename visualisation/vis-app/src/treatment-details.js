import {HttpClient} from 'aurelia-fetch-client';
import environment from './environment';

export class Overview {

  configureRouter(config, router){
    config.map([
      {route: '', redirect: 'count'},
      {route: ':graph', moduleId: 'graph', name: 'graph_page'},
    ]);
    this.router = router;
  }

  constructor() {
    this.client = new HttpClient();
    this.treatment;
    this.name;
    this.active_graph = "count";
  }

  activate(params, routeConfig) {
    this.name = params.name;
    routeConfig.navModel.setTitle(params.name);
    return this.client.fetch(environment.api_base_url + '/treatment/' + params.name)
    .then(response => response.json())
    .then(data => {
      this.treatment = data;
    });
  }

  graph(name) {
    this.active_graph = name;
    this.router.navigateToRoute("graph_page", {graph: name});
  }

  decimals(value) {
    if(value < 10) return value.toFixed(1);
    return value.toFixed(0);
  }

  trend(from, to, type) {
    var elements = {
      sentiment: ["pos", "neu", "neg"],
      arrow: ["up", "right", "down"],
      description: ["increased", "constant", "decreased"],
    }[type]
    if(to > (from * 1.02)) return elements[0];
    if(to < (from * 0.98)) return elements[2];
    return elements[1];
  }
}
