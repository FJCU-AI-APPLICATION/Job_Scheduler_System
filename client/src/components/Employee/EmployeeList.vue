<template>
  <div class="table-wrapper">
    <table id="employeeTable">
      <thead>
        <tr>
          <th>#</th>
          <th>姓名</th>
          <th>年齡</th>
          <th>電話</th>
          <th>身份別</th>
          <th>薪別</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(emp, idx) in employees" :key="emp.id">
          <td>{{ startIndex + idx + 1 }}</td>
          <td>{{ emp.name }}</td>
          <td>{{ emp.age }}</td>
          <td>{{ emp.phone }}</td>
          <td>{{ emp.identity }}</td>
          <td>{{ emp.salary_type }}</td>
        </tr>
      </tbody>
    </table>

    <div class="table-button">
      <button @click="$emit('prev-page')" :disabled="currentPage === 1">
        上一頁
      </button>
      <span>第 {{ currentPage }} / {{ totalPages }} 頁</span>
      <button
        @click="$emit('next-page')"
        :disabled="currentPage === totalPages"
      >
        下一頁
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: "EmployeeList",
  props: {
    employees: {
      type: Array,
      required: true
    },
    startIndex: {
      type: Number,
      required: true
    },
    currentPage: {
      type: Number,
      required: true
    },
    totalPages: {
      type: Number,
      required: true
    }
  }
};
</script>

<style scoped>
.table-wrapper {
  margin: 20px 0;
}
.table-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin-top: 10px;
}
</style>
