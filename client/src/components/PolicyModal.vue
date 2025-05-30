<template>
  <transition name="modal">
    <div class="modal-mask" v-if="modelValue">
      <div class="modal-wrapper">
        <div class="modal-container">
          <div class="modal-header">
            <h3>{{ modalTitle }}</h3>
          </div>

          <div class="modal-body">
            <form>
              <div class="form-group">
                <label for="policy-name">班別名稱</label>
                <input
                  id="policy-name"
                  v-model="form.policy_name"
                  :class="['form-control', { 'is-invalid': nameState === false }]"
                  @input="validateName"
                  required
                />
                <div class="invalid-feedback" v-if="nameState === false">班別名稱不得為空</div>
              </div>

              <div class="form-group">
                <label for="description">描述</label>
                <textarea
                  id="description"
                  v-model="form.description"
                  class="form-control"
                ></textarea>
              </div>
            </form>
          </div>

          <div class="modal-footer">
            <button class="btn btn-secondary" @click="onCancel">取消</button>
            <button class="btn btn-primary" :disabled="!isFormValid" @click="onOk">送出</button>
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<script>
export default {
  name: "PolicyModal",
  props: {
    modelValue: {
      type: Boolean,
      default: false
    },
    isEdit: {
      type: Boolean,
      default: false
    },
    initialPolicy: {
      type: Object,
      default: () => ({
        policy_name: "",
        description: ""
      })
    }
  },
  emits: ["update:modelValue", "submit"],
  data() {
    return {
        form: {
        policy_name: "",
        description: "",  
    },
    nameState: null
    };
  },
  computed: {
    modalTitle() {
      return this.isEdit ? "編輯班別" : "新增班別";
    },
    isFormValid() {
      return this.form.policy_name.trim().length > 0;
    }
  },
  watch: {
    modelValue(val) {
    //   this.$emit("update:modelValue", val);
      if (!val) this.resetForm();
    },
    initialPolicy: {
      handler(newVal) {
        this.form = { ...newVal };
      },
      immediate: true
    }
  },
  methods: {
    onOk(evt) {
        if (!this.isFormValid) {
            if (evt && typeof evt.preventDefault === 'function') {
            evt.preventDefault();
            }
            return;
        }
        // console.log("🚀 要送出的 form", this.form); 
        // console.log("🚀 policy_name:", this.form.policy_name);
        // console.log("🚀 description:", this.form.description);
        this.$emit("submit", { ...this.form });
        this.$emit("update:modelValue", false);
    },
    onCancel() {
        this.$emit("update:modelValue", false);
    },
    resetForm() {
        // this.form = { ...this.initialPolicy };
        this.form = {
            policy_name: "",
            description: ""
        };
        this.nameState = null;
    },
    validateName() {
        this.nameState = this.form.policy_name.trim() !== "";
    }
  }
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