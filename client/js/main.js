// import 'bootstrap/dist/css/bootstrap.min.css';  // 引入 Bootstrap 樣式
// import 'bootstrap-icons/font/bootstrap-icons.css'; // 引入 Bootstrap Icon
// import 'bootstrap';  // 可選，載入 Bootstrap JS（如果有互動功能）
import { fetchEmployees_2 } from "./apiClient.js"
import { API_URL } from "./env.js";

document.addEventListener("DOMContentLoaded", async function () {
    console.log("DOM 已經載入完成！");

    // 漢堡選單測試
    const sidebar = document.getElementById("Sidebar");
    const toggleButton = document.createElement("button");

    toggleButton.innerText = "☰";
    toggleButton.id = "SidebarToggle";
    toggleButton.style.position = "absolute";
    toggleButton.style.top = "500px";
    toggleButton.style.left = "10px";
    toggleButton.style.fontSize = "24px";
    toggleButton.style.background = "transparent";
    toggleButton.style.border = "none";
    toggleButton.style.cursor = "pointer";

    document.body.appendChild(toggleButton);


    
    // 綁定查詢按鈕
    const searchButton = document.getElementById("btn_search_member");

    if (!searchButton) {
        console.error("找不到 #btn_search_member 按鈕，請確認 HTML 結構！");
        return;
    }

    searchButton.addEventListener("click", async () => {
        console.log("🔍 查詢按鈕被點擊，開始獲取員工資料...");
        // 動態產生表格的函式
        await generateEmployeeTable();
    });

    
    let currentUrl = `${API_URL}/api/employee/`;
    let nextUrl = null;
    let prevUrl = null;

    // 中文表頭對應順序
    const column_names = [
        "項次",
        "姓名",
        "年齡",
        "電話",
        "身分別",
        "薪別",
    ];

    // 分頁按鈕
    const nextBtn = document.getElementById("nextPageBtn");
    const prevBtn = document.getElementById("prevPageBtn");

    // **將表格生成的邏輯封裝成函式**
    async function generateEmployeeTable(url = currentUrl) {

        // 動態產生表格
        const table = document.getElementById("employeeTable");
        const tableHead = document.querySelector("#employeeTable thead tr");
        const tableBody = document.querySelector("#employeeTable tbody");

        if (!table || !tableHead || !tableBody) {
            console.error("找不到表格結構，請確認 HTML 結構！");
            return;
        }

        try {
            const { employees, next, previous } = await fetchEmployees_2(url);  // 呼叫 API 獲取員工資料
            currentUrl = url; // 更新目前頁面網址
            nextUrl = next;
            prevUrl = previous;

            console.log("📦 即將請求 URL：", url);
            console.log("收到員工的資料", employees);

            if (!Array.isArray(employees) || employees.length === 0) {
                console.error("employees不是陣列，無法處理", employees);
                return;
            }

            // 取得 `results` 第一筆資料的 `keys` 作為表頭
            const keys = Object.keys(employees[0]);
            console.log("表頭 keys:", keys);

            // 先清空原本的表頭，避免重複生成
            tableHead.innerHTML = "";
            tableBody.innerHTML = "";

            // 動態產生<th>表頭
            column_names.forEach(name => {
                const th = document.createElement("th");
                th.textContent = name;
                tableHead.appendChild(th);
            });            

            // 動態產生員工資料<td>
            employees.forEach(employee => {
                const row = document.createElement("tr");

                // 排除 insert_date 和 update_date 欄位
                keys.slice(0, -2).forEach(key => {
                    const td = document.createElement("td");
                    td.textContent = employee[key];
                    row.appendChild(td);
                });
                tableBody.appendChild(row);
            });

            // 控制分頁按鈕顯示與否
            //nextBtn.style.display = nextUrl ? "inline-block" : "none";
            nextBtn.style.display = "inline-block";
            //nextBtn.style.backgroundColor = nextUrl ? "#28a745" : "#ffc107";
            nextBtn.style.opacity = nextUrl ? 1 : 0.5;
            
            //prevBtn.style.display = prevUrl ? "inline-block" : "none";
            prevBtn.style.display = "inline-block";
            //prevBtn.style.backgroundColor = prevUrl ? "#28a745" : "#ffc107";
            prevBtn.style.opacity = prevUrl ? 1 : 0.5;

        } catch (error) {
            console.error("載入員工資料失敗:", error);
        }
    };
    
    // 分頁按鈕事件
    nextBtn.addEventListener("click", () => {
        if (nextUrl) {
            generateEmployeeTable(`${API_URL}${nextUrl}`);
        }
    });

    prevBtn.addEventListener("click", () => {
        if (prevUrl) {
            generateEmployeeTable(`${API_URL}${prevUrl}`);
        }
    });


    // 導覽列選單
    if (sidebar) {
        toggleButton.addEventListener("click", function () {
            sidebar.classList.toggle("active");
        });
    }; 

    // 獲取所有選單連結和內容區塊
    const menuLinks = document.querySelectorAll('.sidebar ul li a');
    const sections = document.querySelectorAll('.main > div');

    // 點擊事件處理
    menuLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault(); // 防止預設跳轉行為

            // 切換選單的 active 樣式
            menuLinks.forEach(link => link.classList.remove('active'));
            link.classList.add('active');

            // 根據 href 顯示對應的區塊
            const targetId = link.getAttribute('href').substring(1); // 去掉「#」
            sections.forEach(section => {
                if (section.id === targetId) {
                    section.classList.add('active');
                } else {
                    section.classList.remove('active');
                }
            });
        });
    });   
});