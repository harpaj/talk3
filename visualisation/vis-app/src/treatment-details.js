import {HttpClient} from 'aurelia-fetch-client';

export class Overview {
  constructor() {
    this.client = new HttpClient();
    this.treatment;
  }

  activate(params) {
    return this.client.fetch('http://localhost:8765/treatment/' + params.name)
    .then(response => response.json())
    .then(data => {
      this.treatment = data;
    });
  }
}
