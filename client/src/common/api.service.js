import Vue from "vue";
import axios from "axios";
import VueAxios from "vue-axios";
import JwtService from "@/common/jwt.service";
import { API_URL } from "@/common/config";

const ApiService = {
  init() {
    Vue.use(VueAxios, axios);
    Vue.axios.defaults.baseURL = API_URL;
  },

  setHeader() {
    Vue.axios.defaults.headers.common[
      "Authorization"
    ] = `Token ${JwtService.getToken()}`;
  },

  query(resource, params) {
    // console.log("📌 resource:", resource);
    // console.log("📌 params:", params);
    if (typeof params !== "object") {
      throw new Error(`[ApiService] params 必須是物件，你傳的是：${typeof params}`);
    }
    const queryString = new URLSearchParams(params).toString();
    const fullUrl = `${Vue.axios.defaults.baseURL}${resource}?${queryString}`;
    console.log(`QUERY 請求網址：`, fullUrl);

    return Vue.axios.get(resource, { params }).catch((error) => {
      throw new Error(`[RWV] ApiService ${error}`);
    });
  },

  get(resource, slug = "") {
    return Vue.axios.get(`${resource}/${slug}`).catch((error) => {
      throw new Error(`[RWV] ApiService ${error}`);
    });
  },

  post(resource, params) {
    // return Vue.axios.post(`${resource}`, params);
    return Vue.axios.post(`${resource}`, params).catch(error => {
      console.error(`🚨 [POST] ${resource} 失敗`, (error.response && error.response.data) || error.message);
      throw error;
    });
  },

  update(resource, slug, params) {
    return Vue.axios.put(`${resource}/${slug}/`, params);
  },

  put(resource, params) {
    return Vue.axios.put(`${resource}`, params);
  },

  delete(resource) {
    return Vue.axios.delete(resource).catch((error) => {
      throw new Error(`[RWV] ApiService ${error}`);
    });
  }
};

export default ApiService;

export const TagsService = {
  get() {
    return ApiService.get("tags");
  }
};

export const ArticlesService = {
  query(type, params) {
    return ApiService.query("articles" + (type === "feed" ? "/feed" : ""), {
      params: params
    });
  },
  get(slug) {
    return ApiService.get("articles", slug);
  },
  create(params) {
    return ApiService.post("articles", { article: params });
  },
  update(slug, params) {
    return ApiService.update("articles", slug, { article: params });
  },
  destroy(slug) {
    return ApiService.delete(`articles/${slug}`);
  }
};

export const CommentsService = {
  get(slug) {
    if (typeof slug !== "string") {
      throw new Error(
        "[RWV] CommentsService.get() article slug required to fetch comments"
      );
    }
    return ApiService.get("articles", `${slug}/comments`);
  },

  post(slug, payload) {
    return ApiService.post(`articles/${slug}/comments`, {
      comment: { body: payload }
    });
  },

  destroy(slug, commentId) {
    return ApiService.delete(`articles/${slug}/comments/${commentId}`);
  }
};

export const FavoriteService = {
  add(slug) {
    return ApiService.post(`articles/${slug}/favorite`);
  },
  remove(slug) {
    return ApiService.delete(`articles/${slug}/favorite`);
  }
};

export const EmployeeService = {
  // 取得員工列表，可帶分頁或篩選參數
  query(params) {
    return ApiService.query("employee/", params);
  },
  // 取得單一員工
  get(id) {
    return ApiService.get("employee/", id);
  },
  // 新增員工
  create(data) {
    return ApiService.post("employee/", data);
  },
  // 更新員工
  update(id, data) {
    return ApiService.update("employee", id, data);
    // return ApiService.update("employee", id, data).then(res => res.data);
  },
  // 刪除員工
  destroy(id) {
    return ApiService.delete(`employee/${id}/`);
  }
};

export const PolicyService = {
  // 取得政策列表
  query(params) {
    return ApiService.query("policy/", { params });
  },
  // 取得單一政策
  get(id) {
    return ApiService.get("policy", id);
  },
  // 新增政策
  create(data) {
    return ApiService.post("policy", data);
  },
  // 更新政策
  update(id, data) {
    return ApiService.update("policy", id, data);
  },
  // 刪除政策
  destroy(id) {
    return ApiService.delete(`policy/${id}`);
  }
};

export const ScheduleService = {
  // 查詢班表（可用 startDate/endDate 等參數篩選）
  query(params) {
    return ApiService.query("schedules", { params });
  },
  // 取得單筆班表
  get(id) {
    return ApiService.get("schedules", id);
  },
  // 產生新排班（AI 排班）
  generate(range) {
    return ApiService.post("schedules/generate", range);
  },
  // 匯出班表
  export(params) {
    return ApiService.get("schedules/export", { params });
  },
  // 刪除班表（如有此需求）
  destroy(id) {
    return ApiService.delete(`schedules/${id}`);
  },

  create(data) {
    return ApiService.post("schedules", data);
  }
};



