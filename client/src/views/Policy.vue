<template>  
  <div class="policy-view">
    <h1>排班規則</h1>

    <!-- 新增 / 重新載入 按鈕 -->
    <div class="button-field mb-2">
      <b-button variant="success" @click="createPolicy">+ 新增班別</b-button>
      <b-button variant="primary" @click="refreshList">重新載入</b-button>
    </div>

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

      <template #cell(description)="data">
        <strong>{{ data.value || '（尚無資料）' }}</strong>
      </template>

      <template #cell(actions)="row">
        <b-button size="sm" variant="danger" @click="deletePolicy(row.item)">
          刪除
        </b-button>
      </template>

    </b-table>

    <b-pagination
      v-model="currentPage"
      :total-rows="totalCount"
      :per-page="pageSize"
      align="center"
      class="my-3"
    ></b-pagination>
    <!-- :total-rows="policies.length" -->
    <!-- :total-rows="totalCount" -->

    <div v-if="selectedPolicy" class="shift-table mt-4">
      <h4>Shifts for {{ selectedPolicy.policy_name }}</h4>
      <b-table
        :items="selectedPolicy.shifts"
        :fields="shiftFields"
        small
        bordered
        responsive
      ></b-table>
      <!-- // url = "policy/shiftpolicy" -->
    </div>
  </div>
</template>

<script>
import { BTable, BPagination } from "bootstrap-vue";
import axios from "axios";
import store from "@/store";
import { mapGetters } from "vuex";
import {
  FETCH_POLICIES,
  FETCH_POLICY,
  CREATE_POLICY,
  UPDATE_POLICY,
  DELETE_POLICY
} from "../store/actions.type";

export default {
  name: "PolicyView",
  components: {
    BTable,
    BPagination
  },
  data() {
    return {
      // policies: [],
      loading: false,
      error: null,
      // pageSize: 5,
      // currentPage: 1,
      selectedPolicy: null,
      policyFields: [
        { key: "policy_name", label: "Policy Name" },
        { key: "description", label: "Description" },
        { key: "actions", label: "操作" }
      ],
      shiftFields: [
        { key: "start_time", label: "Start Time" },
        { key: "end_time", label: "End Time" }
      ]
    };
  },

  computed: {
    ...mapGetters ("policy", [
      // raw items for this page
      "policies",
      // pagination metadata
      "currentPage",
      "pageSize",
      "totalPages",
      "totalCount",
      "nextPageUrl",
      "prevPageUrl",
      // client‑side helpers
      "hasNext",
      "hasPrev",
      "isLoading",
      "showModal"
    ]),
    paginatedPolicies() {
      const start = (this.currentPage - 1) * this.pageSize;
      const end = start + this.pageSize;
      // return this.policies.slice(start, start + this.pageSize);
      return this.policies.slice(start, end);
    }
  },

  created() {
    // this.fetchPolicies();
    this.refreshList();
  },

  watch: {
    currentPage() {
      const start = (this.currentPage - 1) * this.perPage;
      this.selectedPolicy = this.policies[start] || null;
    }
  },

  methods: {
    refreshList() {
      this.$store.dispatch(`policy/${FETCH_POLICIES}`, {
        page: this.currentPage,
        page_size: this.pageSize
      }).then(() => {
        this.selectedPolicy = this.policies.length ? this.policies[0] : null;
      });
    },
    async fetchPolicies() {
      try {
        // const res = await axios.get("/api/policy-view");
        const res = await axios.get("policy/");
        console.log("API 回傳資料：", res.data);
        // const flatList = res.data;
        const flatList = res.data.results;
        const grouped = {};
        flatList.forEach((item) => {
          if (!grouped[item.id]) {
            grouped[item.id] = {
              policy_id: item.id,
              policy_name: item.policy_name,
              description: item.description,
              shifts: []
            };
          }
          if (item.id) {
            grouped[item.id].shifts.push({
              detail_id: item.id,
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
    async createPolicy() {
      const newPolicy = {
        policy_name: "早班",
        description: "正職班別"
      };
      await this.$store.dispatch(`policy/${CREATE_POLICY}`, 
        { policy: newPolicy }
      );
    },
    async updatePolicy(policy) {
      const updated = {
        ...policy,
        description: "更新後的描述"
      };
      await this.$store.dispatch(`policy/${UPDATE_POLICY}`,
        { policy: updated }
      );
    },
    async deletePolicy(policy) {
      await this.$store.dispatch(`policy/${DELETE_POLICY}`,
        { policy }
      );
    },
    selectPolicy(policy) {
      this.selectedPolicy = policy;
    }
  },
  // mounted() {
  //   this.refreshList();
  // }
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