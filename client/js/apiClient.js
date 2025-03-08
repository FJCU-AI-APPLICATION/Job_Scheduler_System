// let API_URL = "";


// const API_URL = "https://cors-anywhere.herokuapp.com/http://xxx.xx.xx.xx:8002/api/employee/";
const API_URL = "http://localhost:8080/http://210.xx.xx.xx:8002/api/employee/";

// const API_URL = "http://xxx.xx.xx.xx:8002/api/employee/";
// const API_URL = "/api/employee/"; // 透過代理請求

async function fetchEmployees_2() {
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

export { fetchEmployees_2 };



// Load API URL from config.json
async function loadConfig() {
    try {
        const response = await fetch("config.json");
        const config = await response.json();
        API_URL = config.API_URL;
        console.log("Loaded API URL:", API_URL);
    } catch (error) {
        console.error("Error loading config:", error);
    }
}

// Fetch employees
async function fetchEmployees() {
    await loadConfig();
    try {
        const response = await fetch(API_URL);
        const data = await response.json();
        return data; // Return JSON data
    } catch (error) {
        console.error("Error fetching employees:", error);
        return null;
    }
}

// Create new employee
async function createEmployee(employeeData) {
    await loadConfig();
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(employeeData)
        });
        const data = await response.json();
        return data; // Return JSON response
    } catch (error) {
        console.error("Error creating employee:", error);
        return null;
    }
}

// Edit existing employee
async function editEmployee(id, updatedData) {
    await loadConfig();
    try {
        const response = await fetch(API_URL + id + "/", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(updatedData)
        });
        const data = await response.json();
        return data; // Return JSON response
    } catch (error) {
        console.error("Error updating employee:", error);
        return null;
    }
}

// Delete an employee
async function deleteEmployee(id) {
    await loadConfig();
    try {
        const response = await fetch(API_URL + id + "/", { method: "DELETE" });
        return response.ok ? { success: true } : { success: false }; // Return JSON response
    } catch (error) {
        console.error("Error deleting employee:", error);
        return { success: false };
    }
}

// Fetch all schedules
async function fetchSchedules() {
    await loadConfig();
    try {
        const response = await fetch(API_URL + "schedule/");
        return await response.json();
    } catch (error) {
        console.error("Error fetching schedules:", error);
        return null;
    }
}

// Create a new schedule
async function createSchedule(scheduleData) {
    await loadConfig();
    try {
        const response = await fetch(API_URL + "schedule/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(scheduleData)
        });
        return await response.json();
    } catch (error) {
        console.error("Error creating schedule:", error);
        return null;
    }
}

// Edit an existing schedule
async function editSchedule(id, updatedData) {
    await loadConfig();
    try {
        const response = await fetch(API_URL + "schedule/" + id + "/", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(updatedData)
        });
        return await response.json();
    } catch (error) {
        console.error("Error updating schedule:", error);
        return null;
    }
}

// Delete a schedule
async function deleteSchedule(id) {
    await loadConfig();
    try {
        const response = await fetch(API_URL + "schedule/" + id + "/", { method: "DELETE" });
        return response.ok ? { success: true } : { success: false };
    } catch (error) {
        console.error("Error deleting schedule:", error);
        return { success: false };
    }
}

export { fetchEmployees, createEmployee, editEmployee, deleteEmployee,fetchSchedules, createSchedule, editSchedule, deleteSchedule };