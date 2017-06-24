import {HttpClient} from 'aurelia-fetch-client';

export class Overview {
  constructor() {
    this.client = new HttpClient();
    this.treatments = [];
    this.all_treatments = [];
    this.filter = "";
    this.order = -1;
    this.key = "last_year_cnt";
    this.sort_label = "Frequency";
  }

  activate() {
    var self = this;
    return this.client.fetch('http://localhost:8765/treatment_summary')
    .then(response => response.json())
    .then(data => {
      this.treatments = this.all_treatments = data.treatment_list.sort(
        function(a, b){return (a[self.key] - b[self.key]) * self.order})
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

  sort() {
    var self = this;
    function compare(a, b) {
      if(self.key == "treatment") return a.treatment.localeCompare(b.treatment) * self.order;
      return (a[self.key] - b[self.key]) * self.order;
    };
    this.treatments.sort(compare);
    this.all_treatments.sort(compare);
  }

  changeOrder() {
    this.order *= -1;
    this.sort();
  }

  changeKey(new_key) {
    this.key = new_key;
    this.sort();
    this.sort_label = {
      last_year_cnt: "Frequency",
      total_score: "Score",
      treatment: "Name"
    }[new_key];
  }
}
