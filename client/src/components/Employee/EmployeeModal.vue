<template>
  <b-modal
    v-model="visible"
    :title="modalTitle"
    @hidden="resetForm"
    @ok="onOk"
    ok-title="送出"
    cancel-title="取消"
    :ok-disabled="!isFormValid"
  >
    <b-form ref="form" @submit.stop.prevent="onOk">
      <b-form-group
        label="姓名"
        label-for="name-input"
        invalid-feedback="姓名為必填"
        :state="nameState"
      >
        <b-form-input
          id="name-input"
          v-model="form.name"
          :state="nameState"
          required
        />
      </b-form-group>

      <b-form-group
        label="年齡"
        label-for="age-input"
        invalid-feedback="年齡需大於 0"
        :state="ageState"
      >
        <b-form-input
          id="age-input"
          type="number"
          v-model.number="form.age"
          :state="ageState"
          min="1"
          required
        />
      </b-form-group>

      <b-form-group label="電話" label-for="phone-input">
        <b-form-input id="phone-input" v-model="form.phone" />
      </b-form-group>

      <b-form-group label="身份別" label-for="identity-input">
        <b-form-select
          id="identity-input"
          v-model="form.identity"
          :options="[
            { value: 'FULL', text: '正職' },
            { value: 'PART', text: '兼職' }
          ]"
        />
      </b-form-group>

      <b-form-group label="薪別" label-for="salary-input">
        <b-form-select
          id="salary-input"
          v-model="form.salary_type"
          :options="[
            { value: 'MONTH', text: '月薪' },
            { value: 'HOUR', text: '時薪' }
          ]"
        />
      </b-form-group>
    </b-form>
  </b-modal>
</template>

<script>
export default {
  name: "EmployeeModal",
  props: {
    modelValue: {
      type: Boolean,
      default: false
    },
    isEdit: {
      type: Boolean,
      default: false
    },
    initialEmployee: {
      type: Object,
      default: () => ({
        name: "",
        age: null,
        phone: "",
        identity: "FULL",
        salary_type: "MONTH"
      })
    }
  },
  emits: ["update:modelValue", "submit"],
  data() {
    return {
      form: { ...this.initialEmployee },
      nameState: null,
      ageState: null
    };
  },
  computed: {
    visible: {
      get() {
        return this.modelValue;
      },
      set(v) {
        this.$emit("update:modelValue", v);
      }
    },
    modalTitle() {
      return this.isEdit ? "編輯成員" : "新增成員";
    },
    isFormValid() {
      return this.checkValidity();
    }
  },
  watch: {
    initialEmployee: {
      immediate: true,
      handler(emp) {
        this.form = { ...emp };
      }
    }
  },
  methods: {
    resetForm() {
      this.form = { ...this.initialEmployee };
      this.nameState = null;
      this.ageState = null;
    },
    checkValidity() {
      const validName = !!this.form.name.trim();
      this.nameState = validName;
      const validAge = Number.isInteger(this.form.age) && this.form.age > 0;
      this.ageState = validAge;
      return validName && validAge;
    },
    onOk(evt) {
      if (!this.checkValidity()) {
        evt.preventDefault();
        return;
      }
      // emit submit up to parent
      this.$emit("submit", { ...this.form });
    }
  }
};
</script>

<style scoped>
</style>
