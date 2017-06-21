
export class App {
  configureRouter(config, router){
    config.title = 'Talk3';
    config.map([
      {route: '', moduleId: 'overview', title: 'Overview', name: 'overview'},
      {route: 'treatment/:name', moduleId: 'treatment-details', name: 'treatment'}
    ]);
    this.router = router;
  }
}
