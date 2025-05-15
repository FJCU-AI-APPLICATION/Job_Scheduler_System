import Vue from "vue";
import { BootstrapVue, BootstrapVueIcons } from "bootstrap-vue";
import { SchedulePlugin } from '@syncfusion/ej2-vue-schedule';

import App from "./App.vue";
import router from "./router";
import store from "./store";
import "./registerServiceWorker";
import "@/assets/global.css";

// Bootstrap & BootstrapVue
import "bootstrap/dist/css/bootstrap.css";
import "bootstrap-vue/dist/bootstrap-vue.css";

// FontAwesome
import "@fortawesome/fontawesome-free/css/all.css";

// Syncfusion Scheduler (UI library styles)
import '@syncfusion/ej2-base/styles/material.css';
import '@syncfusion/ej2-buttons/styles/material.css';
import '@syncfusion/ej2-calendars/styles/material.css';
import '@syncfusion/ej2-dropdowns/styles/material.css';
import '@syncfusion/ej2-inputs/styles/material.css';
import '@syncfusion/ej2-navigations/styles/material.css';
import '@syncfusion/ej2-popups/styles/material.css';
import '@syncfusion/ej2-schedule/styles/material.css';

// Global SCSS for sidebar theme
// import "vue-bootstrap-sidebar/dist/vue-bootstrap-sidebar.css";
// import "vue-bootstrap-sidebar/src/scss/default-theme.scss";

// import { CHECK_AUTH } from "./store/actions.type";
import ApiService from "./common/api.service";
import DateFilter from "./common/date.filter";
import ErrorFilter from "./common/error.filter";

Vue.config.productionTip = false;
Vue.filter("date", DateFilter);
Vue.filter("error", ErrorFilter);

ApiService.init();

// Ensure we checked auth before each page load.
// router.beforeEach((to, from, next) =>
//   Promise.all([store.dispatch(CHECK_AUTH)]).then(next)
// );

Vue.use(BootstrapVue);
Vue.use(BootstrapVueIcons);
Vue.use(SchedulePlugin);

import { library } from "@fortawesome/fontawesome-svg-core";
import {
  faUsers,
  faCalendarAlt,
  faClipboardList
} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/vue-fontawesome";

library.add(faUsers, faCalendarAlt, faClipboardList);
Vue.component("font-awesome-icon", FontAwesomeIcon);

new Vue({
  router,
  store,
  render: (h) => h(App)
}).$mount("#app");
