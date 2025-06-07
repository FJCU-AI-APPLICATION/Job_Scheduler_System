import Vue from "vue";
import Router from "vue-router";

Vue.use(Router);

const router = new Router({
  mode: "history",
  routes: [
    {
      path: '/',
      redirect: '/login'
    },
    {
      path: "/employee",
      name: "employee",
      component: () => import("@/views/Employee"),
      children: [
        // {
        //   path: "",
        //   name: "home",
        //   component: () => import("@/views/HomeGlobal")
        // },
        // {
        //   path: "my-feed",
        //   name: "home-my-feed",
        //   component: () => import("@/views/HomeMyFeed")
        // },
        // {
        //   path: "tag/:tag",
        //   name: "home-tag",
        //   component: () => import("@/views/HomeTag")
        // },
      ]
    },
    {
      path: "/policy",
      name: "policy",
      component: () => import("@/views/Policy")
    },
    {
      path: "/schedule",
      name: "schedule",
      component: () => import("@/views/Schedule")
    },
    {
      name: "login",
      path: "/login",
      component: () => import("@/views/Login")
    }
    // {
    //   name: "register",
    //   path: "/register",
    //   component: () => import("@/views/Register")
    // },
    // {
    //   name: "settings",
    //   path: "/settings",
    //   component: () => import("@/views/Settings")
    // },
    // Handle child routes with a default, by giving the name to the
    // child.
    // SO: https://github.com/vuejs/vue-router/issues/777
    // {
    //   path: "/@:username",
    //   component: () => import("@/views/Profile"),
    //   children: [
    //     {
    //       path: "",
    //       name: "profile",
    //       component: () => import("@/views/ProfileArticles")
    //     },
    //     {
    //       name: "profile-favorites",
    //       path: "favorites",
    //       component: () => import("@/views/ProfileFavorited")
    //     }
    //   ]
    // },
    // {
    //   name: "article",
    //   path: "/articles/:slug",
    //   component: () => import("@/views/Article"),
    //   props: true
    // },
    // {
    //   name: "article-edit",
    //   path: "/editor/:slug?",
    //   props: true,
    //   component: () => import("@/views/ArticleEdit")
    // }
  ]
});

router.beforeEach((to, from, next) => {
  // const isLoggedIn = localStorage.getItem("loggedIn"); // 假設登入後有設定這個
  const isLoggedIn = sessionStorage.getItem("loggedIn");
  if (to.path !== "/login" && !isLoggedIn) {
    next("/login");
  } else if (to.path === "/login" && isLoggedIn) {
    next("/employee");
  } else {
    next();
  }
});

export default router;