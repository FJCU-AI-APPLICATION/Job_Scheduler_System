// src/store/modules/employee.js
import Vue from "vue";
import { PolicyService } from "@/common/api.service";
import {
  FETCH_POLICIES,
  CREATE_POLICY,
  UPDATE_POLICY,
  DELETE_POLICY
} from "@/store/actions.type";
import { SET_POLICIES } from "@/store/mutations.type";

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
  [FETCH_POLICIES](context, params) {
    return PolicyService.get(params)
      .then(({ data }) => {
        context.commit(SET_POLICIES, data.policy);
      })
      .catch((error) => {
        throw new Error(error);
      });
  },
  [CREATE_POLICY](context, { policy }) {
    return PolicyService.post(policy)
      .then(({ data }) => {
        context.commit(SET_POLICIES, data.policy);
      })
      .catch((error) => {
        throw new Error(error);
      });
  },
  [UPDATE_POLICY](context, { policy }) {
    return PolicyService.put(policy)
      .then(({ data }) => {
        context.commit(SET_POLICIES, data.policy);
      })
      .catch((error) => {
        throw new Error(error);
      });
  },
  [DELETE_POLICY](context, { policy }) {
    return PolicyService.delete(policy)
      .then(() => {
        context.commit(SET_POLICIES, null);
      })
      .catch((error) => {
        throw new Error(error);
      });
  }
};

export const mutations = {
  [SET_POLICIES](state, policy) {
    state.policy = policy;
  }
};

export default {
  state,
  actions,
  mutations
};
