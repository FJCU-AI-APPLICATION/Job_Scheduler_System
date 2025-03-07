// import 'bootstrap/dist/css/bootstrap.min.css';  // 引入 Bootstrap 樣式
// import 'bootstrap-icons/font/bootstrap-icons.css'; // 引入 Bootstrap Icon
// import 'bootstrap';  // 可選，載入 Bootstrap JS（如果有互動功能）
import { fetchEmployees_2 } from "./apiClient.js"

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


    // 動態產生表格
    const table = document.getElementById("employeeTable");
    const tableHead = document.querySelector("#employeeTable thead tr");
    const tableBody = document.querySelector("#employeeTable tbody");

    if (!table || !tableHead || !tableBody) {
        console.error("找不到表格結構，請確認 HTML 結構！");
        return;
    }

    try {
        const { employees } = await fetchEmployees_2();  // 呼叫 API 獲取員工資料
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
        keys.forEach(key => {
            const th = document.createElement("th");
            th.textContent = key;
            tableHead.appendChild(th);
        })

        // 動態產生員工資料<td>
        employees.forEach(employee => {
            const row = document.createElement("tr");
            keys.forEach(key => {
                const td = document.createElement("td");
                td.textContent = employee[key];
                row.appendChild(td);
            });
            tableBody.appendChild(row);
        });
    } catch (error) {
        console.error("載入員工資料失敗:", error);
    }

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