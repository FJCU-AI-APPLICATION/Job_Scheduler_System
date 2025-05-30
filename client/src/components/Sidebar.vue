<template>
  <BootstrapSidebar
    :initial-show="initialShow"
    :links="links"
    :header="header"
    :fa="true"
    @sidebarChanged="onSidebarChanged"
  >
  
    <template v-slot:header>
      <img src="/ShiftX_Logo_20250503.png" alt="Logo" style="max-height: 50px;" />
    </template>

    <template v-slot:navbar>
      <b-navbar
        id="mainNavbar"
        toggleable="lg"
        type="light"
        variant="light"
        fixed="top"
      >
        <b-navbar-nav>
          <b-nav-item> Navbar </b-nav-item>
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
</style>
