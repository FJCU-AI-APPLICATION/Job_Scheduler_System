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
import EmployeeList from "@/components/Employee/EmployeeList.vue";
import EmployeeTable from "@/components/Employee/EmployeeTable.vue";
import EmployeeModal from "@/components/Employee/EmployeeModal.vue";
import { FETCH_EMPLOYEES } from "@/store/actions.type";

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
    ...mapGetters([
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
        return this.$store.state.employee.currentPage;
      },
      set(newPage) {
        this.$store.commit("SET_CURRENT_PAGE", newPage);
        // this.refreshList();
      }
    }
  },

  methods: {
    refreshList() {
      store.dispatch(FETCH_EMPLOYEES, {
        page: this.currentPage,
        pagesize: this.pageSize
      });
    },

    onPageChange(newPage) {
      console.log("頁碼切換到：", newPage);
      this.$store.commit("SET_CURRENT_PAGE", newPage);
      this.refreshList();
    },

    openCreate() {
      this.isEdit = false;
      this.editingEmployee = {};
      this.showModal = true;
      // this.showModal = false; // Show the modal when creating

      // this.$nextTick(() => {
      //   this.showModal = true;
      // });
    },
    openEdit(emp) {
      this.isEdit = true;
      this.editingEmployee = emp;
      this.showModal = true; // Show the modal when editing
    },

    confirmDelete(emp) {
      if (!confirm(`確定刪除 ${emp.name}？`)) return;
      this.DELETE_EMPLOYEE(emp.id).then(() => this.refreshList());
    },

    handleSubmit(empData) {
      const action = this.isEdit ? "UPDATE_EMPLOYEE" : "CREATE_EMPLOYEE";
      store
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
          this.refreshList();
          // this.$refs.employeeModal.closeUniqueModal();
          this.showModal = false;
        });
    }
  },

  mounted() {
    this.refreshList();
    // console.log("Employee module state:", this.$store.state.employee);
  }
};
</script>
