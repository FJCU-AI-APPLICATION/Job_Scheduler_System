<template>  
  <div class="policy-view">
    <h1>排班規則</h1>

    <!-- 新增 / 重新載入 按鈕 -->
    <div class="button-field mb-2">
      <b-button variant="success" @click="addPolicy">+ 新增班別</b-button>
      <span>&nbsp;&nbsp;&nbsp;</span>
      <b-button variant="primary" @click="refreshList">重新載入</b-button>
    </div>

    <PolicyModal
      :modelValue="modalVisible"
      :isEdit="isEdit"
      :initialPolicy="selectedPolicy"
      @update:modelValue="modalVisible = $event"
      @submit="submitPolicy"
    />

    <b-table
      :items="paginatedPolicies"
      :fields="policyFields"
      striped
      hover
      responsive
      head-variant="dark"
      @row-clicked="selectPolicy"
    >
      <template #cell(policy_name)="data">
        <strong>{{ data.value }}</strong>
      </template>

      <template #cell(description)="data">
        <strong>{{ data.value || '（尚無資料）' }}</strong>
      </template>

      <!-- <template #cell(actions)="row">
        <b-button size="sm" variant="danger" @click="deletePolicy(row.item)">
          刪除
        </b-button>
      </template> -->

      <template #cell(actions)="row">
        <b-dropdown right size="sm" variant="link" no-caret>
          <!-- 三點按鈕 -->
          <template #button-content>
            <i class="fas fa-ellipsis-v"></i>
          </template>
          <b-dropdown-item @click="editPolicy(row.item)">
            編輯
          </b-dropdown-item>
          <b-dropdown-item @click="confirmDelete(row.item)">
            刪除
          </b-dropdown-item>
        </b-dropdown>
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
        :items="[selectedPolicy]"
        :fields="shiftFields"
        striped        
        hover
        bordered
        responsive
        head-variant="dark"
      >
      <!-- small -->
      <!-- :items="selectedPolicy.shifts || []" -->
        <template #cell(start_time)="data">
          <strong>{{ data.item.start_time || "尚無時間" }}</strong>
        </template>
        <template #cell(end_time)="data">
          <strong>{{ data.item.end_time || "尚無時間" }}</strong>
        </template>
      </b-table>
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
import PolicyModal from "@/components/PolicyModal.vue";

export default {
  name: "PolicyView",
  components: {
    BTable,
    BPagination,
    PolicyModal
  },
  data() {
    return {
      // policies: [],
      loading: false,
      error: null,
      // pageSize: 5,
      // currentPage: 1,
      modalVisible: false,
      isEdit: false,
      newPolicy: {
        policy_name: '',
        description: ''
      },
      selectedPolicy: null,
      // selectedPolicy: {
      //   policy_name: "全家_1",
      //   shifts: [
      //     { start_time: "08:00", end_time: "12:00" },
      //     { start_time: "13:00", end_time: "17:00" }
      //   ]
      // },
      policyFields: [
        { key: "policy_name", label: "Policy Name" },
        { key: "description", label: "Description" },
        { key: "actions", label: "操作", thStyle: { width: "5%" } }
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
      const start = (this.currentPage - 1) * this.pageSize;
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

    addPolicy() {
      this.selectedPolicy = {
        policy_name: "",
        description: ""
      };
      this.isEdit = false;
      this.modalVisible = true;
    },

    editPolicy(policy) {
      this.selectedPolicy = { ...policy }; // 儲存要編輯的資料
      this.modalVisible = true;
      this.isEdit = true;
    },

    selectPolicy(policy) {
      this.selectedPolicy = policy;
      // console.log("👉 selectedPolicy:", policy);
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
              // detail_id: item.id,
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
    
    async submitPolicy(policy) {
      if (!policy.policy_name) {
        alert("班別名稱不得為空");
        return;
      }

      console.log("🧾 送出資料的 ID：", policy.id);
      // console.log("🧾 送出的資料：", policy);

      const action = this.isEdit ? `policy/${UPDATE_POLICY}` : `policy/${CREATE_POLICY}`;
      const payload = this.isEdit
        ? {
            id: policy.id,
            payload: policy
          }
        : policy;

      try {
        const updated = await this.$store.dispatch(action, payload);
        alert(this.isEdit ? "班別資料已更新" : "新班別已新增");
        this.selectedPolicy = updated;
        this.modalVisible = false;
        this.refreshList();
        // this.$refs.employeeModal.closeUniqueModal();
      } catch (error) {
        console.error("❌ 儲存失敗", error);
        alert("儲存失敗，請稍後再試");
      }
    },

    // async deletePolicy(policy) {
    async confirmDelete(policy) {
      if (!confirm(`確定刪除 ${policy.policy_name}？`)) return;
      // console.log("🧹 刪除項目：", policy);
      // console.log("🔍 policy JSON:", JSON.stringify(policy, null, 2));
      // console.log("🆔 policy.id：", policy.id);
      // await this.$store.dispatch(`policy/${DELETE_POLICY}`, { policy });
      await this.$store.dispatch(`policy/${DELETE_POLICY}`, policy.id);
      this.refreshList();
    }
  },
  // mounted() {
  //   this.refreshList();
  // }
};
</script>

<style scoped>
.fas.fa-ellipsis-v {
  font-size: 1.2rem;
  color: #555;
}
.policy-view {
  max-width: 800px;
}
.shift-table h4 {
  margin-bottom: 0.75rem;
}
</style>