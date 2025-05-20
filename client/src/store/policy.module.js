// src/store/modules/employee.js
import Vue from "vue";
import { PolicyService } from "@/common/api.service";
import {
  FETCH_POLICIES,
  FETCH_POLICY,
  CREATE_POLICY,
  UPDATE_POLICY,
  DELETE_POLICY
} from "@/store/actions.type";
import {
  SET_POLICIES,
  SET_POLICY,
  ADD_POLICY,
  UPDATE_POLICY_IN_LIST,
  REMOVE_POLICY,
  SET_LOADING,
  SET_CURRENT_PAGE,
  SET_NEXT_URL,
  SET_PREV_URL,
  SET_POLICIES_COUNT
} from "@/store/mutations.type";


const initialState = {
  results: [],
  count: 0,
  next: null,
  previous: null,
  currentPage: 1,
  pageSize: 10,
  isLoading: false,
  selectedPolicy: null
};

export const state = { ...initialState };

export const actions = {
  async [FETCH_POLICIES](context, params) {
    context.commit(SET_LOADING, true);
    console.log("actions - FETCH_POLICIES 參數", params);
    try {
      const { data } = await PolicyService.query(params);
      console.log("actions - FETCH_POLICIES 資料", data);
      context.commit(SET_POLICIES, data.results);
      context.commit(SET_POLICIES_COUNT, data.count);
      context.commit(SET_NEXT_URL, data.next);
      context.commit(SET_PREV_URL, data.previous);
      context.commit(SET_CURRENT_PAGE, params.page);
    } catch (err) {
      console.error("Failed fetching policies", err);
    } finally {
      context.commit(SET_LOADING, false);
    }
  },

  async [FETCH_POLICY](context, id) {
    try {
      const { data } = await PolicyService.get(id);
      context.commit(SET_POLICY, data);
    } catch (error) {
      console.error("Failed fetching single policy", error);
      throw new Error(error);
    }
  },

  async [CREATE_POLICY](context, payload) {
    const { data } = await PolicyService.create(payload);
    context.commit(ADD_POLICY, data);
    return data;
  },

  async [UPDATE_POLICY](context, { id, payload }) {
    const { data } = await PolicyService.update(id, payload);
    context.commit(UPDATE_POLICY_IN_LIST, data);
    return data;
  },

  async [DELETE_POLICY](context, id) {
    await PolicyService.destroy(id);
    context.commit(REMOVE_POLICY, id);
  }
};

export const mutations = {
  [SET_POLICIES](state, payload) {
    state.results = Array.isArray(payload) ? payload : [];
    console.log("mutations - SET_POLICIES: ", state.results);
  },

  [ADD_POLICY](state, policy) {
    state.results = [policy, ...state.results];
  },

  [UPDATE_POLICY_IN_LIST](state, policy) {
    const idx = state.results.findIndex((e) => e.id === policy.id);
    if (idx !== -1) {
      Vue.set(state.results, idx, policy);
    }
    if (state.current.id === policy.id) {
      state.current = policy;
    }
  },

  [REMOVE_POLICY](state, id) {
    state.results = state.results.filter((e) => e.id !== id);
    if (state.current.id === id) {
      Object.assign(state.current, initialState.current);
    }
  },

  [SET_CURRENT_PAGE](state, page) {
    state.currentPage = page;
  },

  [SET_NEXT_URL](state, url) {
    state.next = url;
  },

  [SET_PREV_URL](state, url) {
    state.previous = url;
  },

  [SET_POLICIES_COUNT](state, count) {
    state.count = count;
  },

  [SET_LOADING](state, flag) {
    state.isLoading = flag;
  },
  // [SET_POLICIES](state, policy) {
  //   state.results = policy.results;
  //   state.count = policy.count;
  //   state.next = policy.next;
  //   state.previous = policy.previous;
  //   state.currentPage = policy.currentPage;
  //   state.pageSize = policy.pageSize;
  // }
};

export const getters = {
  currentPage: (state) => state.currentPage,
  pageSize: (state) => state.pageSize,  
  totalPages: (state) => Math.ceil(state.count / state.pageSize) || 1,
  // raw items for this page
  policies: (state) => state.results,
  // pagination metadata
  totalCount: (state) => state.count,
  nextPageUrl: (state) => state.next,
  prevPageUrl: (state) => state.previous,

  // client‑side helpers
  hasNext: (state) => state.next !== null,
  hasPrev: (state) => state.previous !== null,
  isLoading: (state) => state.isLoading,
  showModal: (state) => state.showModal
};

// const initialState = {
//   results: [],
//   count: 0,
//   next: null,
//   previous: null,
//   currentPage: 1,
//   pageSize: 10,
//   isLoading: false
// };

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
};
