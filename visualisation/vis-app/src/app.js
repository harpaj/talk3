
export class App {
  configureRouter(config, router){
    config.title = 'Talk3';
    config.map([
      {route: '', moduleId: 'graph', title: 'Graph'},
    ]);

    this.router = router;
  }
}
