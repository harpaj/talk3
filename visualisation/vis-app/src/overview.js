import {HttpClient} from 'aurelia-fetch-client';

export class Overview {
  constructor() {
    this.client = new HttpClient();
    this.treatments = [];
    this.all_treatments = [];
    this.filter = "";
  }

  activate() {
    return this.client.fetch('http://localhost:8765/treatment_summary')
    .then(response => response.json())
    .then(data => {
      this.treatments = this.all_treatments = data.treatment_list;
    });
  }

  filterTreatments() {
    var self = this;
    if(this.filter){
      self.treatments = self.all_treatments.filter(function(el) {
        return (
          el.treatment.toLowerCase().startsWith(self.filter.toLowerCase()) ||
          el.names.some(function(li) {return li.toLowerCase().startsWith(self.filter.toLowerCase())})
        )
      });
    }
    else{
      self.treatments = self.all_treatments;
    }
  }
}
