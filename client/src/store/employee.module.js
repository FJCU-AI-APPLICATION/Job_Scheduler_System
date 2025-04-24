// src/store/modules/employee.js
import Vue from "vue";
import { EmployeeService } from "@/common/api.service";
import {
  FETCH_EMPLOYEES,
  FETCH_EMPLOYEE,
  CREATE_EMPLOYEE,
  UPDATE_EMPLOYEE,
  DELETE_EMPLOYEE
} from "@/store/actions.type";
import {
  SET_EMPLOYEES,
  SET_EMPLOYEE,
  ADD_EMPLOYEE,
  UPDATE_EMPLOYEE_IN_LIST,
  REMOVE_EMPLOYEE,
  RESET_EMPLOYEE_STATE,
  SET_LOADING,
  SET_CURRENT_PAGE
} from "@/store/mutations.type";

const initialState = {
  results: [],
  count: 0,
  next: null,
  previous: null,
  currentPage: 1,
  pageSize: 10,
  isLoading: false
};


export const state = { ...initialState };

export const actions = {
  async [FETCH_EMPLOYEES](context, params) {
    context.commit(SET_LOADING, true);
    try {
      const { data } = await EmployeeService.query("employee/", params);
      console.log("params", params);
      console.log("data", data);
      context.commit(SET_EMPLOYEES, data.results);
      context.commit(SET_CURRENT_PAGE, params.page);
    } catch (err) {
      console.error("Failed fetching employees", err);
    } finally {
      context.commit(SET_LOADING, false);
    }
  },

  async [FETCH_EMPLOYEE](context, id) {
    const { data } = await EmployeeService.get(id);
    context.commit(SET_EMPLOYEE, data);
    return data;
  },

  async [CREATE_EMPLOYEE](context, payload) {
    const { data } = await EmployeeService.create(payload);
    context.commit(ADD_EMPLOYEE, data);
    return data;
  },

  async [UPDATE_EMPLOYEE](context, { id, payload }) {
    const { data } = await EmployeeService.update(id, payload);
    context.commit(UPDATE_EMPLOYEE_IN_LIST, data);
    return data;
  },

  async [DELETE_EMPLOYEE](context, id) {
    await EmployeeService.destroy(id);
    context.commit(REMOVE_EMPLOYEE, id);
  }
};

export const mutations = {
  [SET_EMPLOYEES](state, payload) {
    state.results = Array.isArray(payload) ? payload : [];
  },
  [ADD_EMPLOYEE](state, emp) {
    state.list = [emp, ...state.list];
  },

  [UPDATE_EMPLOYEE_IN_LIST](state, emp) {
    const idx = state.list.findIndex((e) => e.id === emp.id);
    if (idx !== -1) {
      Vue.set(state.list, idx, emp);
    }
    if (state.current.id === emp.id) {
      state.current = emp;
    }
  },

  [REMOVE_EMPLOYEE](state, id) {
    state.list = state.list.filter((e) => e.id !== id);
    if (state.current.id === id) {
      Object.assign(state.current, initialState.current);
    }
  },
  [SET_CURRENT_PAGE](state, page) {
    state.currentPage = page;
  },
  [RESET_EMPLOYEE_STATE](state) {
    Object.keys(initialState).forEach((key) => {
      Vue.set(state, key, JSON.parse(JSON.stringify(initialState[key])));
    });
  },
  [SET_LOADING](state, flag) {
    state.isLoading = flag;
  }
};

export const getters = {
  currentPage: (state) => state.currentPage,
  pageSize: (state) => state.pageSize,
  employeesCount: (state) => state.list.length,
  totalPages: (state) => Math.ceil(state.list.length / state.pageSize) || 1,
  // raw items for this page
  employees: (state) => state.results,

  // pagination metadata
  totalCount: (state) => state.count,
  nextPageUrl: (state) => state.next,
  prevPageUrl: (state) => state.previous,

  // client‑side helpers
  hasNext: (state) => state.next !== null,
  hasPrev: (state) => state.previous !== null,
  isLoading: (state) => state.isLoading
};

export default {
  state,
  actions,
  mutations,
  getters
};
