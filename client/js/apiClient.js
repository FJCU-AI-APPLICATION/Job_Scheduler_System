let API_URL = "";

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

// Export functions (for modular usage)
export { fetchEmployees, createEmployee, editEmployee, deleteEmployee };