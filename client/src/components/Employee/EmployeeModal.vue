<template>
  <b-modal
    v-if="modelValue"
    :title="modalTitle"
    @hidden="resetForm"
    @ok="onOk"
    ok-title="送出"
    cancel-title="取消"
    :ok-disabled="!isFormValid"
  >
    <b-form @submit.stop.prevent="onOk">
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
    </b-form>
  </b-modal>
</template>

<script>
export default {
  name: "EmployeeModal",
  props: {
    visible: {
      type: Boolean,
      default: false
    },
    isEdit: { type: Boolean, default: false },
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
      ageState: null,
      modelValue: this.visible
    };
  },
  computed: {
    isFormValid() {
      return this.form.name && this.form.age > 0;
    }
  },
  methods: {
    resetForm() {
      this.form = { ...this.initialEmployee };
      this.nameState = null;
      this.ageState = null;
    },
    onOk(evt) {
      if (!this.isFormValid) {
        evt.preventDefault();
        return;
      }
      this.$emit("submit", { ...this.form });
    }
  },
  beforeMount() {
    this.modelValue = this.visible;
  }
};
</script>
