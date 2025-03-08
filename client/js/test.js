fetch("http://localhost:8080/http://211.20.21.35:8002/api/employee/", { mode: "cors" })
  .then(response => {
    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
  
      // 檢查是否是 JSON 格式
      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        return response.text().then(text => {
          throw new Error(`非 JSON 回應: ${text}`);
        });
      }
  
    return response.json();
    })
  .then(data => {
    console.log("API 回應", data);
    const employees = data.results;
    console.log("員工列表", employees);
    })
  .catch(error => console.error("錯誤:", error));