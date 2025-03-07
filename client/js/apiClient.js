
// const API_URL = "http://211.20.21.35:8002/api/employee/";
const API_URL = "/api/employee/"; // 透過代理請求

async function fetchEmployees() {
    try {
        console.log(`Fetching: ${API_URL}`);
        
        const response = await fetch(API_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // 取得回應的表頭 (Headers)
        const headers = {};
        response.headers.forEach((value, name) => {
            headers[name] = value;
        });

        const data = await response.json();
        console.log("API 回傳資料:", data);
        console.log("API 回傳表頭:", headers);

        // 確保回傳的是陣列
        if (!Array.isArray(data.results)) {
            console.error("API 沒回傳陣列，可能是錯誤的格式", data.results);
            return {headers, employees: []}; // 回傳空陣列，避免forEach錯誤
        }

        return {headers, employees: data.results};
    } catch (error) {
        console.error("Error fetching employees:", error);
        return {headers, employees: []};
    }
}

export { fetchEmployees };
