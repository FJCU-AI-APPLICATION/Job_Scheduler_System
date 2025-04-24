<template>
  <b-table
    :items="items"
    :fields="fields"
    responsive
    hover
    striped
    small
    bordered
    head-variant="dark"
    empty-text="目前尚無員工資料"
  >
    <!-- Index 列 -->
    <template #cell(index)="row">
      {{ row.item.index }}
    </template>

    <!-- 操作 列：三點選單 -->
    <template #cell(actions)="row">
      <b-dropdown right size="sm" variant="link" no-caret>
        <!-- 三點按鈕 -->
        <template #button-content>
          <i class="fas fa-ellipsis-v"></i>
        </template>
        <b-dropdown-item @click="$emit('edit', row.item)">
          編輯
        </b-dropdown-item>
        <b-dropdown-item @click="$emit('delete', row.item)">
          刪除
        </b-dropdown-item>
      </b-dropdown>
    </template>
  </b-table>
</template>

<script>
export default {
  name: "EmployeeTable",
  props: {
    employees: {
      type: Array,
      required: true
    },
    currentPage: {
      type: Number,
      required: true
    },
    pageSize: {
      type: Number,
      required: true
    }
  },
  computed: {
    fields() {
      return [
        { key: "index", label: "#", thStyle: { width: "5%" } },
        { key: "name", label: "姓名" },
        { key: "age", label: "年齡", thStyle: { width: "10%" } },
        { key: "phone", label: "電話", thStyle: { width: "15%" } },
        { key: "identity", label: "身份別", thStyle: { width: "15%" } },
        { key: "salary_type", label: "薪別", thStyle: { width: "15%" } },
        { key: "actions", label: "操作", thStyle: { width: "5%" } }
      ];
    },
    items() {
      return this.employees.map((emp, idx) => ({
        ...emp,
        index: (this.currentPage - 1) * this.pageSize + idx + 1
      }));
    }
  }
};
</script>

<style scoped>
.fas.fa-ellipsis-v {
  font-size: 1.2rem;
  color: #555;
}
</style>
