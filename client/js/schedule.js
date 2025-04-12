import { API_URL } from "./env.js";
import { fetchSchedules } from "./apiClient.js";
import {
    identityMap,
    salaryTypeMap,
    getIdentityDisplay,
    getSalaryTypeDisplay
} from "./formatter.js";

async function searchSchedule() {

    let endpoint = "schedule";
    let currentUrl = `${API_URL}/api/${endpoint}/`;
    let nextUrl = null;
    let prevUrl = null;

    // 中文表頭對應順序
    /* const column_names = [
        "員工ID",
        "姓名",
        "年齡",
        "電話",
        "身分別",
        "薪別",
    ]; */

    // 分頁按鈕
    const nextBtn = document.getElementById("btnSchedule_next");
    const prevBtn = document.getElementById("btnSchedule_prev");

    await generateScheduleTable(currentUrl);    

    // **將表格生成的邏輯封裝成函式**
    async function generateScheduleTable(url) {

        // 動態產生表格
        const table = document.getElementById("scheduleTable");
        const tableHead = document.querySelector("#scheduleTable thead tr");
        const tableBody = document.querySelector("#scheduleTable tbody");

        if (!table || !tableHead || !tableBody) {
            console.error("找不到表格結構，請確認 HTML 結構！");
            return;
        }

        try {
            const { schedules , next, previous } = await fetchSchedules(url);  // 呼叫 API 獲取員工資料
            //const { schedules, next, previous } = await fetchSchedules(url);  // 呼叫 API 獲取員工資料
            // currentUrl = url; // 更新目前頁面網址
            // nextUrl = next;
            // prevUrl = previous;

            console.log("📦 即將請求 URL：", url);
            console.log("收到員工的資料", schedules);

            if (!Array.isArray(schedules) || schedules.length === 0) {
                console.error("schedules", schedules);
                return;
            }

            // 取得 `results` 第一筆資料的 `keys` 作為表頭
            const keys = Object.keys(schedules[0]);
            console.log("表頭 keys:", keys);

            // 先清空原本的表頭，避免重複生成
            tableHead.innerHTML = "";
            tableBody.innerHTML = "";

            // 動態產生<th>表頭
            keys.forEach(key => {
                const th = document.createElement("th");
                th.textContent = key;
                tableHead.appendChild(th);
            });            

            // 動態產生員工資料<td>
            schedules.forEach(schedule => {
                const row = document.createElement("tr");

                keys.forEach(key => {
                    const td = document.createElement("td");
                    
                    if (key === "identity") {
                        td.textContent = getIdentityDisplay(schedule[key]);
                    } else if (key === "salary_type") {
                        td.textContent = getSalaryTypeDisplay(schedule[key]);
                    } else {
                        td.textContent = schedule[key];
                    }

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
};

export { searchSchedule };