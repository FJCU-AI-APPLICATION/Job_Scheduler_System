// import 'bootstrap/dist/css/bootstrap.min.css';  // 引入 Bootstrap 樣式
// import 'bootstrap-icons/font/bootstrap-icons.css'; // 引入 Bootstrap Icon
// import 'bootstrap';  // 可選，載入 Bootstrap JS（如果有互動功能）


document.addEventListener("DOMContentLoaded", function () {
    console.log("DOM 已經載入完成！");
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
