<template>
  
  <transition name="modal">
    <div class="modal-mask" v-if="modelValue">
      <div class="modal-wrapper">
        <div class="modal-container">
          <div class="modal-header">
            <h3>{{ modalTitle }}</h3>
          </div>

          <div class="modal-body">
            <form @submit.prevent="onOk">
              <div class="form-group">
                <label for="name-input">姓名</label>
                <input
                  id="name-input"
                  v-model="form.name"
                  :class="['form-control', { 'is-invalid': nameState === false }]"
                  @input="validateName"
                  required
                />
                <div class="invalid-feedback" v-if="nameState === false">姓名為必填</div>
              </div>

              <div class="form-group">
                <label for="age-input">年齡</label>
                <input
                  id="age-input"
                  type="number"
                  v-model="form.age"
                  :class="['form-control', { 'is-invalid': ageState === false }]"
                  @input="validateAge"
                  min="1"
                  required
                />
                <div class="invalid-feedback" v-if="ageState === false">年齡需大於 0</div>
              </div>

              <div class="form-group">
                <label for="phone-input">電話</label>
                <input
                  id="phone-input"
                  type="text"
                  v-model="form.phone"
                  :class="['form-control', { 'is-invalid': phoneState === false }]"
                  @input="validatePhone"
                  min="1"
                  required
                />
                <div class="invalid-feedback" v-if="phoneState === false">請輸入有效手機號碼(09xxxxxxxx)</div>
              </div>

              <div class="form-group">
                <label for="identity-input">身份別</label>
                <select
                  id="identity-input"
                  v-model="form.identity"
                  :class="['form-control', { 'is-invalid': identityState === false }]"
                  @change="validateIdentity"
                  required
                >
                  <option value="">請選擇</option>
                  <option value="FULL">正職</option>
                  <option value="PART">兼職</option>
                </select>
                <div class="invalid-feedback" v-if="identityState === false">請選擇身份別</div>
              </div>

              <div class="form-group">
                <label for="salary_type-input">薪別</label>
                <select
                  id="salary_type-input"
                  v-model="form.salary_type"
                  :class="['form-control', { 'is-invalid': salaryTypeState === false }]"
                  @change="validateSalaryType"
                  required
                >
                  <option value="">請選擇</option>
                  <option value="MONTH">月薪</option>
                  <option value="HOUR">時薪</option>
                </select>
                <div class="invalid-feedback" v-if="salaryTypeState === false">請選擇薪別</div>
              </div>
            </form>
          </div>

          <div class="modal-footer">
            <button class="btn btn-secondary" @click="$emit('update:modelValue', false)">取消</button>
            <button class="btn btn-primary" :disabled="!isFormValid" @click="onOk">送出</button>
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<script>
export default {
  name: "EmployeeModal",
  props: {
    modelValue: {
      type: Boolean,
      default: false
    },
    // visible: {
    //   type: Boolean,
    //   default: false
    // },
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
      phoneState: null,
      identityState: null,
      salaryTypeState: null,
      // modelValue: this.visible
    };
  },
  watch: {
    // visible(val) {
    //   this.modelValue = val;
    // },
    modelValue(val) {
      this.$emit("update:modelValue", val);
      if (!val) this.resetForm();
    }
  },
  computed: {
    modalTitle() {
      return this.isEdit ? "編輯員工" : "新增員工";
    },
    isFormValid() {
      return (
        this.form.name &&
        this.form.age > 0 &&
        this.phoneState !== false &&
        this.identityState !== false &&
        this.salaryTypeState !== false
      );
    }
  },
  methods: {
    resetForm() {
      this.form = { ...this.initialEmployee };
      this.nameState = null;
      this.ageState = null;
      this.phoneState = null;
      this.identityState = null;
      this.salaryTypeState = null;
    },
    onOk(evt) {
      if (!this.isFormValid) {
        if (evt && typeof evt.preventDefault === 'function') {
          evt.preventDefault();
        }    
        return;
      }
      this.$emit("submit", { ...this.form });
      this.$emit("update:modelValue", false);
    },
    validateName() {
      this.nameState = this.form.name.trim() !== "";
    },
    validateAge() {
      this.ageState = this.form.age > 0;
    },
    validatePhone() {
      const regex = /^09\d{8}$/;
      this.phoneState = regex.test(this.form.phone.toString().trim());
    },
    validateIdentity() {
      this.identityState = this.form.identity === "FULL" || this.form.identity === "PART";
    },
    validateSalaryType() {
      this.salaryTypeState =
        this.form.salary_type === "MONTH" || this.form.salary_type === "HOUR";
    }
  },
  // beforeMount() {
  //   this.modelValue = this.visible;
  // }
};
</script>

<style scoped>
.modal-mask {
  position: fixed;
  z-index: 9998;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-container {
  width: 400px;
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.33);
}

.modal-header,
.modal-footer {
  padding: 10px 20px;
  background: #f1f1f1;
}

.modal-body {
  padding: 20px;
}

.modal-enter-active,
.modal-leave-active {
  transition: all 0.3s ease;
}

.modal-enter,
.modal-leave-to {
  opacity: 0;
  transform: scale(1.05);
}
</style>



<!-- <template>
  <transition name="modal">
    <div class="modal-mask">
      <div class="modal-wrapper">
        <div class="modal-container">
          <div class="modal-header">
            <slot name="header">
              default header
            </slot>
          </div>

          <div class="modal-body">
            <slot name="body">
              default body
            </slot>
          </div>

          <div class="modal-footer">
            <slot name="footer">
              default footer
              <button class="modal-default-button" @click="$emit('close')">
                OK
              </button>
            </slot>
          </div>
        </div>
      </div>
    </div>
  </transition>

  <b-modal
    :visible="modelValue"
    :title="modalTitle"
    @hidden="resetForm"
    @ok="onOk"
    ok-title="送出"
    cancel-title="取消"
    :ok-disabled="!isFormValid"
    :modal-class="'modal-zfix'"
    :content-class="'modal-content-fix'"
    :dialog-class="'modal-dialog-fix'"
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

  watch: {
    modelValue(val) {
      this.$emit("update:modelValue", val);
    },
    visible(val) {
      this.modelValue = val;
    }
  },

  computed: {
    modalTitle() {
      return this.isEdit ? "編輯員工" : "新增員工";
    },
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
</script> -->