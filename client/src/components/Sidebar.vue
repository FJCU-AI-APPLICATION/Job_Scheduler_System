<template>
  <BootstrapSidebar
    :initial-show="initialShow"
    :header="header"
    :links="links"    
    :fa="true"
    @sidebarChanged="onSidebarChanged"
  >
  
    <!-- <template v-slot:header>
      <img src="/ShiftX_Logo_20250503.png" alt="Logo" style="max-height: 50px;" />
    </template> -->

    <template v-slot:navbar>
      <b-navbar
        id="mainNavbar"
        toggleable="lg"
        type="light"
        variant="light"
        fixed="top"
      >
        <b-navbar-nav class="navbar-items">
          <b-nav-item>
            <img src="/Fu_Jen_Catholic_University_logo.png" alt="Logo" style="height: 42px;" />
          </b-nav-item>
          <b-nav-item class="">FJU 113 AI專班 專題報告</b-nav-item>
        </b-navbar-nav>
      </b-navbar>
    </template>

    <template v-slot:content>
      <b-container style="margin-top: 56px">
        <router-view />
      </b-container>
    </template>
  </BootstrapSidebar>
</template>

<script>
import BootstrapSidebar from "vue-bootstrap-sidebar";

export default {
  name: "Sidebar",
  components: { BootstrapSidebar },
  props: {
    initialShow: {
      type: Boolean,
      default: true
    },
    header: {
      type: String,
      // default: "<h3>Sidebar</h3>"      
      default: `<img src="/ShiftX_Logo_20250503.png" alt="Logo" style="max-height: 50px;" />`
    },
    links: {
      type: Array,
      default: () => [
        {
          name: "成員設定",
          href: { name: "employee" },
          faIcon: ["fas", "users"]
        },
        {
          name: "班表查詢",
          href: { name: "schedule" },
          faIcon: ["fas", "calendar-alt"]
        },
        {
          name: "排班規則",
          href: { name: "policy" },
          faIcon: ["fas", "clipboard-list"]
        }
      ]
    }
  },  
  methods: {
    onSidebarChanged(show) {
      this.$emit("sidebarChanged", show);
    }
  }
};
</script>

<style lang="scss">
@import "node_modules/vue-bootstrap-sidebar/src/scss/default-theme";

.navbar-items {
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  justify-content: flex-end;
  width: 100%;  
}

.navbar-items > *:not(:last-child) {
  margin-right: -1.5rem;
}
</style>
