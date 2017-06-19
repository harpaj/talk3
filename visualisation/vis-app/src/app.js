
export class App {
  configureRouter(config, router){
    config.title = 'Talk3';
    config.map([
      {route: '', moduleId: 'overview', title: 'Overview'},
    ]);
    this.router = router;
  }
}
