<template>
  <div class="auth-page">
    <div class="container-page">
      <!-- <div class=""> -->
        <div class="container-column">
          <h1 class="">ShiftX AI排班系統應用於便利商店人力管理</h1>
          <!-- <p class="text-xs-center"> -->
            <!-- <router-link :to="{ name: 'register' }">
              Need an account?
            </router-link> -->
          <!-- </p> -->

          <div class="error-block">
            <ul v-if="errors" class="error-messages">
              <li v-for="(v, k) in errors" :key="k">{{ k }} {{ v | error }}</li>
            </ul>
          </div>
          
          <form @submit.prevent="onSubmit(empNumber, password)">
            <fieldset class="form-group">
              <div class="form-label">
                員工編號：
              </div>
              <input
                class="form-control form-control-lg form-block"
                type="text"
                v-model="empNumber"
                placeholder="員工編號"
              />
            </fieldset>
            <fieldset class="form-group">
              <div class="form-label">
                密碼：
              </div>
              <input
                class="form-control form-control-lg form-block"
                type="password"
                v-model="password"
                placeholder="密碼"
              />
            </fieldset>
            <fieldset class="form-group">
              <span class="btn-span"></span>
              <button class="btn btn-lg btn-primary pull-xs-right">
                登入
              </button>
            </fieldset>
          </form>
        </div>
      <!-- </div> -->
    </div>
  </div>
</template>

<script>
import { mapState } from "vuex";
import { LOGIN } from "@/store/actions.type";

export default {
  name: "RwvLogin",
  data() {
    return {
      empNumber: null,
      password: null,
      errors: null
    };
  },
  methods: {
    onSubmit(empNumber, password) {
      // 固定帳密
      if (empNumber === "Fju003" && password === "pbai113") {
        alert("登入成功！");
        // localStorage.setItem("loggedIn", "true");
        sessionStorage.setItem("loggedIn", "true");
        this.$router.push({ name: "employee" });
      } else {
        this.errors = { login: ["帳密錯誤，員工編號：Fju003；密碼：pbai113"] };
      }
    }
  },
  //   onSubmit(email, password) {
  //     this.$store
  //       .dispatch(LOGIN, { email, password })
  //       .then(() => this.$router.push({ name: "home" }));
  //   }
  // },
  // computed: {
  //   ...mapState({
  //     errors: state => state.auth.errors
  //   })
  // }
};
</script>


<style>
html, body {
  height: 100vh;
  width: 100vw;
  margin: 0;
}

.auth-page {
  height: 100%;
  width: 100%;
  /* width: 800px;
  min-height: 100vh;           撐滿整個畫面 */
  margin: 0 auto;              /* 水平置中 */
  /* display: flex; */
  align-items: center;         /* 垂直置中 */
  justify-content: center;     /* 可選，若內容需水平置中 */
  /* background-color: #f8f9fa;   可選背景色讓畫面更明顯 */
}

.container-page {
  height: 100%;
  width: 100%;  
}

.container-column {
  width: 100%;
}

.error-block {
  display: flex;
  align-items: center;
  height: 30px;
  margin: 10px;
}

.form-group {
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  margin-bottom: 1rem;
  gap: 2rem;
}

.form-group .form-label {
  width: 20%;
  font-size: 24px;
  flex-wrap: nowrap;
  text-align: right;
  
}

.form-group .form-block {
  /* width: 50%; */
  flex: 1;
}

.form-group .btn-span {
  flex: 1;
}

</style>