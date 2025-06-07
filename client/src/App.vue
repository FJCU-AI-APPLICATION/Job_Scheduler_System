<template>
  <div id="app" class="container">
    <!-- 測試模式開啟才顯示 -->
    <!-- <div v-if="!isLoginPage"> -->
      <div>
        <TestModal v-if="isTestingModal">
          <template #header>          
          </template>
          <template #body>
          
          </template>
          <template #footer>
            <button @click="alert('送出')">送出</button>
            <button @click="close">關閉</button>
          </template>
        </TestModal>
      
        <!-- 正式內容 -->
        <div v-else>
          <Sidebar />
        </div>
      </div>
    <!-- </div> -->
    
    <!-- <router-view /> -->
  </div>
</template>

<script>
import TestModal from "@/components/TestModal.vue";
import Sidebar from "@/components/Sidebar.vue";
import "@/assets/global.css";


export default {
  name: "App",
  components: {
    TestModal,
    Sidebar
  },
  computed: {
    isLoginPage() {
      return this.$route.path === "/login";
    }
  },
  methods: {
    close() {
      this.isTestingModal = false;
    }
  },
  data () {
    return {
      isTestingModal: false
    };
  },
  mounted() {
    window.addEventListener('keydown', (e) => {
      if (e.key === 'm' && !this.isLoginPage) {
        this.isTestingModal = !this.isTestingModal;
      }
    });    
  }
};
</script>

<style></style>
