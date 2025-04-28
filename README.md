# Job_Scheduler_System

job-scheduling-frontend/
├── README.md
├── package.json
├── public/
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── assets/                   # 靜態資源（圖片、字型等）
│   ├── components/               # 通用可重複使用元件
│   │   ├── Employee/
│   │   │   ├── EmployeeList.vue      # 員工清單（列出 / 搜尋 / 刪除）
│   │   │   ├── EmployeeForm.vue      # 新增/編輯員工資料
│   │   │   └── UnavailabilityForm.vue # 設定員工不可排班時間
│   │   ├── Schedule/
│   │   │   ├── ScheduleList.vue      # 既有排班表檢視
│   │   │   └── ScheduleGenerator.vue # 根據政策與員工產生新排班
│   │   └── Policy/
│   │       ├── PolicyList.vue        # 排班政策列表
│   │       └── PolicyForm.vue        # 新增/編輯排班政策 (開始/結束時間)
│   ├── views/                     # 路由對應的頁面（View）
│   │   ├── EmployeePage.vue          # /employees
│   │   ├── SchedulePage.vue          # /schedules
│   │   └── PolicyPage.vue            # /policies
│   ├── router/                    # Vue Router 設定
│   │   └── index.js
│   ├── store/                     # Vuex / Pinia 狀態管理
│   │   └── index.js
│   ├── services/                  # API 呼叫封裝
│   │   └── api.js
│   ├── utils/                     # 公共工具函式
│   │   └── dateFormatter.js
│   ├── App.vue
│   └── main.js
└── tests/                        # 單元測試、元件測試
    ├── components/
    └── views/
