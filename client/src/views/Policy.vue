<template>
  <div class="policy-view container py-4">
    <b-table
      :items="paginatedPolicies"
      :fields="policyFields"
      striped
      hover
      responsive
      @row-clicked="selectPolicy"
    >
      <template #cell(policy_name)="data">
        <strong>{{ data.value }}</strong>
      </template>
    </b-table>

    <b-pagination
      v-model="currentPage"
      :total-rows="policies.length"
      :per-page="perPage"
      align="center"
      class="my-3"
    ></b-pagination>

    <div v-if="selectedPolicy" class="shift-table mt-4">
      <h4>Shifts for {{ selectedPolicy.policy_name }}</h4>
      <b-table
        :items="selectedPolicy.shifts"
        :fields="shiftFields"
        small
        bordered
        responsive
      ></b-table>
    </div>
  </div>
</template>

<script>
import { BTable, BPagination } from "bootstrap-vue";
import axios from "axios";
import store from "@/store";
import { mapGetters } from "vuex";

//import { mapGetters } from "vuex";

export default {
  name: "PolicyView",
  components: {
    BTable,
    BPagination
  },
  data() {
    return {
      policies: [],
      loading: false,
      error: null,
      perPage: 5,
      currentPage: 1,
      selectedPolicy: null,
      policyFields: [
        { key: "policy_name", label: "Policy Name" },
        { key: "description", label: "Description" }
      ],
      shiftFields: [
        { key: "start_time", label: "Start Time" },
        { key: "end_time", label: "End Time" }
      ]
    };
  },
  computed: {
    paginatedPolicies() {
      const start = (this.currentPage - 1) * this.perPage;
      return this.policies.slice(start, start + this.perPage);
    }
  },
  created() {
    this.fetchPolicies();
  },
  methods: {
    async fetchPolicies() {
      try {
        const res = await axios.get("/api/policy-view");
        const flatList = res.data;
        const grouped = {};
        flatList.forEach((item) => {
          if (!grouped[item.policy_id]) {
            grouped[item.policy_id] = {
              policy_id: item.policy_id,
              policy_name: item.policy_name,
              description: item.description,
              shifts: []
            };
          }
          if (item.detail_id) {
            grouped[item.policy_id].shifts.push({
              detail_id: item.detail_id,
              start_time: item.start_time,
              end_time: item.end_time
            });
          }
        });
        this.policies = Object.values(grouped);
        // auto-select first policy on load
        this.selectedPolicy = this.policies.length ? this.policies[0] : null;
      } catch (err) {
        this.error = err;
      } finally {
      }
    },
    selectPolicy(policy) {
      this.selectedPolicy = policy;
    }
  }
};
</script>

<style scoped>
.policy-view {
  max-width: 800px;
}
.shift-table h4 {
  margin-bottom: 0.75rem;
}
</style>
