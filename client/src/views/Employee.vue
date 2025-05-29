<template>
  
  <section id="member-settings">
    <h1>成員設定</h1>

    <!-- 新增 / 重新載入 按鈕 -->
    <div class="button-field mb-2">
      <b-button variant="success" @click="openCreate">+ 新增成員</b-button>
      <b-button variant="primary" @click="refreshList">重新載入</b-button>
    </div>
    
    <!-- 員工表格 -->
    <EmployeeTable
      :employees="employees"
      @edit="openEdit"
      @delete="confirmDelete"
      :current-page="currentPage"
      :page-size="pageSize"
    />

    <!-- 分頁元件 -->
    <b-pagination
      v-model="currentPage"
      :per-page="pageSize"
      :total-rows="employeesCount"
      class="my-3"      
      @change="onPageChange"
    />
      <!-- :value="currentPage" -->
      <!-- @input="onPageChange" -->
    <!-- 員工 Modal -->
    <EmployeeModal
      :modelValue="showModal"
      :is-edit="isEdit"
      :initial-employee="editingEmployee"
      @submit="handleSubmit"
      @update:modelValue="showModal = $event"
    />
    <!-- <p>showModal: {{ showModal }}</p> -->
    <!-- <EmployeeModal
      ref="employeeModal"
      :visible="showModal"
      :is-edit="isEdit"
      :initial-employee="editingEmployee"
      @submit="handleSubmit"
      @update:modelValue="showModal = $event"
    /> -->
  </section>
</template>

<script>
import store from "@/store";
import { mapGetters } from "vuex";
import EmployeeTable from "@/components/Employee/EmployeeTable.vue";
import EmployeeModal from "@/components/Employee/EmployeeModal.vue";
import {
  CREATE_EMPLOYEE, 
  FETCH_EMPLOYEES, 
  UPDATE_EMPLOYEE, 
  DELETE_EMPLOYEE
} from "@/store/actions.type";
import { SET_CURRENT_PAGE, SET_CURRENT_EMPLOYEE } from "@/store/mutations.type";

export default {
  name: "EmployeePage",
  components: {
    EmployeeTable,
    EmployeeModal
  },
  data() {
    return {
      isEdit: false,
      showModal: false,
      editingEmployee: {}
    };
  },

  computed: {
    ...mapGetters("employee", [
      "currentPage",
      "pageSize",
      "employeesCount",
      "totalPages",
      // raw items for this page
      "employees",

      // pagination metadata
      "totalCount",
      "nextPageUrl",
      "prevPageUrl",

      // client‑side helpers
      "hasNext",
      "hasPrev",
      "isLoading"
    ]),
    currentPage: {
      get() {
        // console.log("👉 讀取 currentPage:", this.$store.state.employee.currentPage);
        return this.$store.state.employee.currentPage;
      },
      set(newPage) {
        // console.log("✏️ 設定 currentPage:", newPage);
        this.$store.commit(`employee/${SET_CURRENT_PAGE}`, newPage);
        // this.refreshList();
      }
    }
  },

  methods: {
    refreshList() {
      // console.log("🔄 執行 refreshList，當前頁碼是:", this.currentPage);
      store.dispatch(`employee/${FETCH_EMPLOYEES}`, {
        page: this.currentPage,
        page_size: this.pageSize
      });
    },

    onPageChange(newPage) {
      // console.log("頁碼切換到：", newPage);      
      this.$store.commit(`employee/${SET_CURRENT_PAGE}`, newPage);
      this.refreshList();
    },

    openCreate() {
      this.isEdit = false;
      this.editingEmployee = {
        name: "",
        age: null,
        phone: "",
        identity: "FULL",
        salary_type: "MONTH"
      };
      this.showModal = true;
      // this.$store.commit(`employee/${SET_CURRENT_EMPLOYEE}`, null);
      // this.showModal = false; // Show the modal when creating

      // this.$nextTick(() => {
      //   this.showModal = true;
      // });
    },
    openEdit(emp) {
      this.isEdit = true;
      this.editingEmployee = emp;
      // this.$store.commit(`employee/${SET_CURRENT_EMPLOYEE}`, emp);
      this.showModal = true; // Show the modal when editing
    },

    confirmDelete(emp) {
      if (!confirm(`確定刪除 ${emp.name}？`)) return;
      this.$store.dispatch(`employee/${DELETE_EMPLOYEE}`, emp.id)
        .then(() => {
          this.refreshList();
        });
    },

    handleSubmit(empData) {
      console.log("🧾 送出資料的 ID：", empData.id);
      // console.log("🧾 送出的資料：", empData);

      const action = this.isEdit ? `employee/${UPDATE_EMPLOYEE}` : `employee/${CREATE_EMPLOYEE}`;
      this.$store
        .dispatch(
          action,
          this.isEdit
            ? {
                id: empData.id,
                payload: empData
              }
            : empData
        )
        .then(() => {
          alert(this.isEdit ? "員工資料已更新" : "新員工已新增");
          this.refreshList();
          // this.$refs.employeeModal.closeUniqueModal();
          this.showModal = false;
        })
        .catch(err => {
          console.error("❌ 儲存失敗", err);
          alert("儲存失敗，請稍後再試");
        });
    }
  },

  mounted() {
    this.refreshList();
    // console.log("Employee module state:", this.$store.state.employee);
  }
};
</script>
